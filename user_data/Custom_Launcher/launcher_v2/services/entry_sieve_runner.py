from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
from queue import Empty, Queue
import sys
import time
from typing import Any

from explorer.explorer_catalog import load_catalog
from explorer.explorer_commands import (
    build_child_env,
    filter_best_params,
    load_json as explorer_load_json,
    merge_params_into_snapshot,
    run_hyperopt,
    save_json as explorer_save_json,
    score_objective,
    strategy_parameter_spaces,
)
from explorer.explorer_support import (
    build_backtest_args,
    latest_result_file,
    parse_backtest_metrics,
    result_file_snapshot,
    run_command,
)
from explorer.explorer_targets import resolve_params
from explorer.explorer_windows import compact_window, load_window_manifest, resolve_windows, window_label


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return deepcopy(default)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run neutral Entry Sieve hyperopt/backtest jobs.")
    parser.add_argument("--job-file", required=True)
    return parser.parse_args(argv)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value)).strip("_") or "item"


def _metric(metrics: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return value
    return None


def _base_snapshot(strategy_class: str) -> dict[str, Any]:
    return {
        "strategy_name": strategy_class,
        "params": {},
        "ft_stratparam_v": 1,
        "export_time": datetime.now().astimezone().isoformat(),
    }


def _build_preset(base_preset: dict[str, Any], strategy_file: Path, strategy_class: str, runtime_dir: Path) -> dict[str, Any]:
    preset = deepcopy(base_preset)
    preset["strategy_file"] = str(strategy_file)
    preset["strategy_class"] = strategy_class
    preset["hyperopt_spaces"] = "buy"
    preset["backtest_export"] = "trades"
    preset["backtest_directory"] = str(runtime_dir / "backtests")
    return preset


@dataclass
class BacktestTask:
    validation_window: dict[str, Any]
    take_profit_pct: str
    stoploss_pct: str
    target_sweep: bool
    env: dict[str, str]
    output_dir: Path


@dataclass
class PendingBacktests:
    job_id: str
    strategy: dict[str, str]
    training_window: dict[str, Any]
    tasks: list[BacktestTask]
    preset: dict[str, Any]
    cwd: Path
    hyperopt_file: Path
    params_file: Path
    best_params_count: int
    epoch_count: int
    hyperopt_loss: float | None
    future: Future[list[dict[str, Any]]] | None = None


def _prepare_runtime_strategy(source_file: Path, run_dir: Path) -> Path:
    strategy_dir = run_dir / "strategy"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    runtime_file = strategy_dir / source_file.name
    runtime_file.write_bytes(source_file.read_bytes())
    return runtime_file


def _child_env(
    job: dict[str, Any],
    cwd: Path,
    original_strategy_dir: Path,
    take_profit_pct: str | None = None,
    stoploss_pct: str | None = None,
) -> dict[str, str]:
    env = build_child_env(os.environ.copy(), cwd)
    extra_paths = [str(original_strategy_dir), str(original_strategy_dir.parent)]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(extra_paths) + (os.pathsep + existing if existing else "")
    env["ENTRY_SIEVE_TAKE_PROFIT_PCT"] = str(take_profit_pct or job.get("take_profit_pct") or "2")
    env["ENTRY_SIEVE_STOPLOSS_PCT"] = str(stoploss_pct or job.get("stoploss_pct") or "2")
    return env


def _result_row(
    *,
    job_id: str,
    strategy: dict[str, str],
    training_window: dict[str, Any],
    validation_window: dict[str, Any],
    status: str,
    hyperopt_file: Path | None = None,
    backtest_file: Path | None = None,
    params_file: Path | None = None,
    best_params_count: int = 0,
    epoch_count: int = 0,
    hyperopt_loss: float | None = None,
    take_profit_pct: str = "",
    stoploss_pct: str = "",
    target_sweep: bool = False,
    metrics: dict[str, Any] | None = None,
    error: str = "",
) -> dict[str, Any]:
    metrics = metrics or {}
    objective = None
    if metrics:
        try:
            objective = score_objective(metrics)
        except Exception:
            objective = None
    return {
        "job_id": job_id,
        "finished_at": datetime.now().astimezone().isoformat(),
        "strategy": strategy.get("name") or Path(strategy.get("strategy_file", "")).stem,
        "strategy_class": strategy.get("strategy_class", ""),
        "strategy_file": strategy.get("strategy_file", ""),
        "side": strategy.get("side", ""),
        "core_behavior": strategy.get("core_behavior", ""),
        "training_window": window_label(training_window),
        "training_timerange": str(training_window.get("timerange") or ""),
        "validation_window": window_label(validation_window),
        "validation_timerange": str(validation_window.get("timerange") or ""),
        "take_profit_pct": str(take_profit_pct),
        "stoploss_pct": str(stoploss_pct),
        "target_sweep": bool(target_sweep),
        "status": status,
        "hyperopt_loss": hyperopt_loss,
        "objective": objective,
        "best_params_count": best_params_count,
        "epoch_count": epoch_count,
        "profit_total_abs": _metric(metrics, "profit_total_abs"),
        "profit_total": _metric(metrics, "profit_total", "profit_total_pct"),
        "profit_total_pct": _metric(metrics, "profit_total_pct"),
        "trade_count": _metric(metrics, "total_trades", "trade_count"),
        "winrate": _metric(metrics, "winrate"),
        "wins": _metric(metrics, "wins"),
        "draws": _metric(metrics, "draws"),
        "losses": _metric(metrics, "losses"),
        "profit_factor": _metric(metrics, "profit_factor"),
        "max_drawdown_pct": _metric(metrics, "max_relative_drawdown", "max_drawdown_account"),
        "final_balance": _metric(metrics, "final_balance"),
        "market_change": _metric(metrics, "market_change"),
        "hyperopt_file": str(hyperopt_file) if hyperopt_file is not None else "",
        "backtest_file": str(backtest_file) if backtest_file is not None else "",
        "params_file": str(params_file) if params_file is not None else "",
        "metrics": metrics,
        "error": error,
    }


def _append_result(runtime_dir: Path, row: dict[str, Any]) -> None:
    job_id = str(row.get("job_id") or "unknown")
    results_file = runtime_dir / "results" / f"{_safe_name(job_id)}.json"
    payload = load_json(results_file, {"rows": []})
    rows = payload.get("rows") if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        rows = []
    rows.append(row)
    updated_at = datetime.now().astimezone().isoformat()
    save_json(
        results_file,
        {
            "schema_version": 2,
            "job_id": job_id,
            "created_at": payload.get("created_at") if isinstance(payload, dict) else "",
            "updated_at": updated_at,
            "status": payload.get("status", "running") if isinstance(payload, dict) else "running",
            "phase": payload.get("phase", "backtest") if isinstance(payload, dict) else "backtest",
            "rows": rows,
        },
    )
    status = payload.get("status", "running") if isinstance(payload, dict) else "running"
    phase = payload.get("phase", "backtest") if isinstance(payload, dict) else "backtest"
    save_json(
        runtime_dir / "latest.json",
        {
            "job_id": job_id,
            "path": str(results_file),
            "updated_at": updated_at,
            "status": status,
            "phase": phase,
        },
    )


def _result_file(runtime_dir: Path, job_id: str) -> Path:
    return runtime_dir / "results" / f"{_safe_name(job_id)}.json"


def _initialize_result_batch(runtime_dir: Path, job: dict[str, Any]) -> None:
    job_id = str(job.get("job_id") or "unknown")
    now = datetime.now().astimezone().isoformat()
    results_file = _result_file(runtime_dir, job_id)
    if results_file.exists():
        payload = load_json(results_file, {"rows": []})
        rows = payload.get("rows") if isinstance(payload, dict) else []
        created_at = payload.get("created_at") if isinstance(payload, dict) else now
    else:
        rows = []
        created_at = str(job.get("created_at") or now)
    if not isinstance(rows, list):
        rows = []
    save_json(
        results_file,
        {
            "schema_version": 2,
            "job_id": job_id,
            "created_at": created_at,
            "updated_at": now,
            "status": "running",
            "phase": "starting",
            "rows": rows,
        },
    )
    save_json(runtime_dir / "latest.json", {"job_id": job_id, "path": str(results_file), "updated_at": now, "status": "running"})


def _update_result_batch_status(runtime_dir: Path, job_id: str, status_payload: dict[str, Any]) -> None:
    results_file = _result_file(runtime_dir, job_id)
    payload = load_json(results_file, {"rows": []})
    rows = payload.get("rows") if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        rows = []
    updated_at = str(status_payload.get("updated_at") or datetime.now().astimezone().isoformat())
    save_json(
        results_file,
        {
            "schema_version": 2,
            "job_id": job_id,
            "created_at": payload.get("created_at") if isinstance(payload, dict) else "",
            "updated_at": updated_at,
            "status": str(status_payload.get("status") or "running"),
            "phase": str(status_payload.get("phase") or ""),
            "rows": rows,
        },
    )
    save_json(
        runtime_dir / "latest.json",
        {
            "job_id": job_id,
            "path": str(results_file),
            "updated_at": updated_at,
            "status": str(status_payload.get("status") or "running"),
            "phase": str(status_payload.get("phase") or ""),
        },
    )


def _write_run_status(runtime_dir: Path, job_id: str, **fields: Any) -> None:
    now = datetime.now().astimezone().isoformat()
    status_dir = runtime_dir / "status"
    status_file = status_dir / f"{_safe_name(job_id)}.json"
    payload = load_json(status_file, {})
    if not isinstance(payload, dict):
        payload = {}
    payload.update(fields)
    payload.update(
        {
            "job_id": job_id,
            "pid": os.getpid(),
            "updated_at": now,
            "heartbeat_at": now,
        }
    )
    save_json(status_file, payload)
    save_json(runtime_dir / "active.json", {"job_id": job_id, "status_file": str(status_file), "updated_at": now, "status": payload.get("status"), "phase": payload.get("phase")})
    _update_result_batch_status(runtime_dir, job_id, payload)


def _run_backtest_task(batch: PendingBacktests, task: BacktestTask, python_exe: str) -> dict[str, Any]:
    preset = deepcopy(batch.preset)
    preset.pop("backtest_directory", None)
    task.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = result_file_snapshot(task.output_dir, ("*.zip", "*.json"))
    command = [
        python_exe,
        "-u",
        "-m",
        "freqtrade",
        *build_backtest_args(preset, str(task.validation_window.get("timerange") or "")),
        "--backtest-directory",
        str(task.output_dir),
    ]
    started_at = time.time()
    completed = run_command(command, batch.cwd, task.env, dry_run=False, stream_output=True)
    if completed is None or completed.returncode != 0:
        raise RuntimeError(f"Backtest failed for {window_label(task.validation_window)} at {task.take_profit_pct}/{task.stoploss_pct}.")
    backtest_file = latest_result_file(task.output_dir, ("*.zip", "*.json"), started_at, None, snapshot)
    if backtest_file is None:
        raise RuntimeError("Backtest completed but no isolated result file was found.")
    metrics = parse_backtest_metrics(backtest_file)
    return _result_row(
        job_id=batch.job_id,
        strategy=batch.strategy,
        training_window=batch.training_window,
        validation_window=task.validation_window,
        status="ok",
        hyperopt_file=batch.hyperopt_file,
        backtest_file=backtest_file,
        params_file=batch.params_file,
        best_params_count=batch.best_params_count,
        epoch_count=batch.epoch_count,
        hyperopt_loss=batch.hyperopt_loss,
        take_profit_pct=task.take_profit_pct,
        stoploss_pct=task.stoploss_pct,
        target_sweep=task.target_sweep,
        metrics=metrics,
    )


def _error_row_for_task(batch: PendingBacktests, task: BacktestTask, error: str) -> dict[str, Any]:
    return _result_row(
        job_id=batch.job_id,
        strategy=batch.strategy,
        training_window=batch.training_window,
        validation_window=task.validation_window,
        status="error",
        hyperopt_file=batch.hyperopt_file,
        params_file=batch.params_file,
        best_params_count=batch.best_params_count,
        epoch_count=batch.epoch_count,
        hyperopt_loss=batch.hyperopt_loss,
        take_profit_pct=task.take_profit_pct,
        stoploss_pct=task.stoploss_pct,
        target_sweep=task.target_sweep,
        error=error,
    )


def _run_backtest_batch(batch: PendingBacktests, python_exe: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in batch.tasks:
        try:
            rows.append(_run_backtest_task(batch, task, python_exe))
        except Exception as exc:
            rows.append(_error_row_for_task(batch, task, str(exc)))
    return rows


def _error_rows_for_batch(batch: PendingBacktests, error: str) -> list[dict[str, Any]]:
    return [
        _error_row_for_task(batch, task, error)
        for task in batch.tasks
    ]


def _finish_pending(
    runtime_dir: Path,
    pending: PendingBacktests | None,
    *,
    job_id: str,
    completed_backtests: int,
    total_backtests: int,
) -> int:
    if pending is None or pending.future is None:
        return 0
    try:
        rows = pending.future.result()
    except Exception as exc:
        rows = _error_rows_for_batch(pending, str(exc))
    written = 0
    for row in rows:
        _append_result(runtime_dir, row)
        written += 1
        done = completed_backtests + written
        _write_run_status(
            runtime_dir,
            job_id,
            status="running",
            phase="backtest",
            completed_backtests=done,
            total_backtests=total_backtests,
            current_strategy=str(row.get("strategy") or pending.strategy.get("name") or pending.strategy.get("strategy_class") or ""),
            current_training_window=str(row.get("training_window") or window_label(pending.training_window)),
            message=f"Backtests {done}/{total_backtests}: {row.get('strategy') or pending.strategy.get('name')}",
        )
    return written


def _prepare_strategy_window(
    *,
    job: dict[str, Any],
    base_preset: dict[str, Any],
    runtime_dir: Path,
    state: dict[str, Any],
    strategy: dict[str, str],
    training_window: dict[str, Any],
    validation_windows: list[dict[str, Any]],
) -> PendingBacktests:
    source_file = Path(strategy["strategy_file"]).resolve()
    strategy_class = str(strategy["strategy_class"])
    run_name = f"{_safe_name(strategy.get('name') or source_file.stem)}__{_safe_name(window_label(training_window))}"
    run_dir = runtime_dir / "runs" / run_name
    runtime_strategy_file = _prepare_runtime_strategy(source_file, run_dir)
    params_file = runtime_strategy_file.with_suffix(".json")

    project_root = Path(str(base_preset.get("project_root") or "")).expanduser()
    if not project_root:
        raise RuntimeError("Entry Sieve preset must include project_root.")
    cwd = project_root.resolve()
    python_exe = str(base_preset.get("python_exe") or job.get("python_exe") or sys.executable)
    preset = _build_preset(base_preset, runtime_strategy_file, strategy_class, runtime_dir)
    env = _child_env(job, cwd, source_file.parent)

    catalog = load_catalog(source_file, strategy_class)
    strategy_spaces = strategy_parameter_spaces(source_file, strategy_class)
    target = resolve_params(catalog, "family", "entries", "targeted", strategy_spaces)
    resolved_params = target.get("resolved_params") or []
    epochs = _resolve_epochs(job, resolved_params)
    best_epoch, hyperopt_file, epoch_count = run_hyperopt(
        python_exe=python_exe,
        cwd=cwd,
        preset=preset,
        target=target,
        timerange=str(training_window.get("timerange") or ""),
        epochs=epochs,
        random_state=str(job.get("random_state") or "").strip() or None,
        env=env,
    )
    candidate_params = filter_best_params(best_epoch, target)
    if not candidate_params:
        raise RuntimeError("Hyperopt completed but returned no entry parameters.")

    snapshot = merge_params_into_snapshot(_base_snapshot(strategy_class), candidate_params)
    explorer_save_json(params_file, snapshot)
    archive_params_file = runtime_dir / "params" / f"{_safe_name(strategy_class)}__{_safe_name(window_label(training_window))}.json"
    explorer_save_json(archive_params_file, snapshot)

    hyperopt_loss = best_epoch.get("loss")
    best_params_count = sum(len(values) for values in candidate_params.values() if isinstance(values, dict))
    target_pairs = _target_pairs(job)
    tasks: list[BacktestTask] = []
    for validation_window in validation_windows:
        validation_name = _safe_name(window_label(validation_window))
        for pair in target_pairs:
            take_profit_pct = str(pair["take_profit_pct"])
            stoploss_pct = str(pair["stoploss_pct"])
            target_name = _safe_name(f"tp_{take_profit_pct}_sl_{stoploss_pct}")
            tasks.append(
                BacktestTask(
                    validation_window=validation_window,
                    take_profit_pct=take_profit_pct,
                    stoploss_pct=stoploss_pct,
                    target_sweep=bool(pair.get("target_sweep")),
                    env=_child_env(job, cwd, source_file.parent, take_profit_pct, stoploss_pct),
                    output_dir=runtime_dir / "backtests" / run_name / validation_name / target_name,
                )
            )
    state.setdefault("completed_runs", []).append(
        {
            "strategy": strategy.get("name") or source_file.stem,
            "strategy_class": strategy_class,
            "training_window": compact_window(training_window),
            "validation_windows": [compact_window(window) for window in validation_windows],
            "target_pairs": [{"take_profit_pct": task.take_profit_pct, "stoploss_pct": task.stoploss_pct} for task in tasks],
            "hyperopt_file": str(hyperopt_file),
            "params_file": str(archive_params_file),
            "best_params_count": best_params_count,
            "epoch_count": epoch_count,
            "finished_at": datetime.now().astimezone().isoformat(),
        }
    )
    return PendingBacktests(
        job_id=str(job.get("job_id") or ""),
        strategy=strategy,
        training_window=training_window,
        tasks=tasks,
        preset=preset,
        cwd=cwd,
        hyperopt_file=hyperopt_file,
        params_file=archive_params_file,
        best_params_count=best_params_count,
        epoch_count=epoch_count,
        hyperopt_loss=float(hyperopt_loss) if isinstance(hyperopt_loss, (int, float)) else None,
    )


def _resolve_epochs(job: dict[str, Any], resolved_params: list[str]) -> str:
    if bool(job.get("auto_epochs")):
        value = max(1, len(resolved_params) * 20)
        cap_text = str(job.get("auto_epochs_cap") or "").strip()
        if cap_text:
            try:
                cap = int(cap_text)
            except (TypeError, ValueError):
                cap = 0
            if cap > 0:
                value = min(value, cap)
        return str(value)
    return str(job.get("epochs") or "200")


def _format_pct(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _parse_pct(value: Any) -> float:
    return abs(float(str(value).strip().lstrip("+")))


def _target_pairs(job: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_tp = _parse_pct(job.get("take_profit_pct") or "2")
    baseline_sl = _parse_pct(job.get("stoploss_pct") or "2")
    pairs: list[dict[str, Any]] = [
        {
            "take_profit_pct": _format_pct(baseline_tp),
            "stoploss_pct": _format_pct(baseline_sl),
            "target_sweep": False,
        }
    ]
    seen = {(pairs[0]["take_profit_pct"], pairs[0]["stoploss_pct"])}
    if not bool(job.get("target_sweep_enabled")):
        return pairs

    text = str(job.get("target_sweep_pairs") or "").replace(";", ",").replace("\n", ",")
    for raw_token in text.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if "/" in token:
            left, right = token.split("/", 1)
        elif ":" in token:
            left, right = token.split(":", 1)
        else:
            parts = token.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid target sweep pair '{token}'. Use TP/SL, for example 2/2.")
            left, right = parts
        take_profit = _parse_pct(left)
        stoploss = _parse_pct(right)
        if take_profit <= 0.0 or stoploss <= 0.0:
            raise ValueError(f"Invalid target sweep pair '{token}'. TP and SL must be greater than 0.")
        normalized = (_format_pct(take_profit), _format_pct(stoploss))
        if normalized in seen:
            continue
        seen.add(normalized)
        pairs.append(
            {
                "take_profit_pct": normalized[0],
                "stoploss_pct": normalized[1],
                "target_sweep": True,
            }
        )
    return pairs


def _backtest_lanes(*, base_python_exe: str, backtest_python_exe: str, split_venv_pipeline: bool) -> list[str]:
    lanes = [str(backtest_python_exe or base_python_exe or sys.executable)]
    if split_venv_pipeline:
        primary = str(base_python_exe or sys.executable)
        try:
            same_exe = Path(primary).resolve() == Path(lanes[0]).resolve()
        except OSError:
            same_exe = primary == lanes[0]
        if not same_exe:
            lanes.insert(0, primary)
    return lanes


def _run_backtest_lane(
    *,
    lane_index: int,
    python_exe: str,
    work_items: list[tuple[PendingBacktests, BacktestTask]],
    result_queue: Queue[dict[str, Any]],
) -> int:
    completed = 0
    total = len(work_items)
    for index, (batch, task) in enumerate(work_items, start=1):
        print(
            f"Target sweep lane {lane_index}: {index}/{total} | "
            f"{batch.strategy.get('name') or batch.strategy.get('strategy_class')} | "
            f"{window_label(task.validation_window)} | TP/SL={task.take_profit_pct}/{task.stoploss_pct}"
        )
        try:
            row = _run_backtest_task(batch, task, python_exe)
        except Exception as exc:
            row = _error_row_for_task(batch, task, str(exc))
        result_queue.put(row)
        completed += 1
    return completed


def _run_target_sweep_backtests(
    *,
    runtime_dir: Path,
    job_id: str,
    batches: list[PendingBacktests],
    lanes: list[str],
    total_backtests: int,
) -> int:
    work_items = [(batch, task) for batch in batches for task in batch.tasks]
    if not work_items:
        return 0
    lane_count = max(1, len(lanes))
    assigned: list[list[tuple[PendingBacktests, BacktestTask]]] = [[] for _ in range(lane_count)]
    for index, item in enumerate(work_items):
        assigned[index % lane_count].append(item)
    print(f"\nEntry Sieve target-sweep backtests: {len(work_items)} tasks across {lane_count} lane(s)")
    result_queue: Queue[dict[str, Any]] = Queue()
    with ThreadPoolExecutor(max_workers=lane_count, thread_name_prefix="entry-sieve-target-sweep") as executor:
        futures = [
            executor.submit(
                _run_backtest_lane,
                lane_index=index + 1,
                python_exe=lanes[index],
                work_items=items,
                result_queue=result_queue,
            )
            for index, items in enumerate(assigned)
            if items
        ]
        completed_backtests = 0
        while completed_backtests < len(work_items):
            try:
                row = result_queue.get(timeout=1.0)
            except Empty:
                for future in futures:
                    if future.done() and future.exception() is not None:
                        raise future.exception()
                if all(future.done() for future in futures):
                    break
                continue
            _append_result(runtime_dir, row)
            completed_backtests += 1
            _write_run_status(
                runtime_dir,
                job_id,
                status="running",
                phase="target_sweep_backtest",
                completed_backtests=completed_backtests,
                total_backtests=total_backtests,
                current_strategy=str(row.get("strategy") or ""),
                current_training_window=str(row.get("training_window") or ""),
                message=f"Target-sweep backtests {completed_backtests}/{total_backtests}: {row.get('strategy')}",
            )
        for future in futures:
            future.result()
        if completed_backtests < len(work_items):
            raise RuntimeError("Target-sweep backtests finished before all result rows were reported.")
    return completed_backtests


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    job_file = Path(args.job_file).resolve()
    job = load_json(job_file, {})
    if not isinstance(job, dict):
        raise SystemExit(f"Job file is not an object: {job_file}")

    runtime_dir = Path(str(job["runtime_dir"])).resolve()
    presets = load_json(Path(str(job["preset_file"])), {})
    preset_name = str(job.get("preset_name") or "LauncherV2-auto")
    if preset_name not in presets:
        preset_name = "BackTest2021-26"
    if preset_name not in presets:
        raise SystemExit("Entry Sieve requires LauncherV2-auto or BackTest2021-26 in presets.json.")

    all_windows = load_window_manifest(job["market_windows_file"])
    training_windows = resolve_windows(all_windows, job.get("training_windows") or [], label="training")
    validation_windows = resolve_windows(all_windows, job.get("validation_windows") or [], label="validation")
    strategies = [strategy for strategy in job.get("strategies") or [] if isinstance(strategy, dict)]
    base_preset = presets[preset_name]
    job_id = str(job.get("job_id") or job_file.stem)
    _initialize_result_batch(runtime_dir, job)
    split_venv_pipeline = bool(job.get("split_venv_pipeline"))
    target_sweep_enabled = bool(job.get("target_sweep_enabled"))
    base_python_exe = str(base_preset.get("python_exe") or job.get("python_exe") or sys.executable)
    backtest_python_exe = str(job.get("backtest_python_exe") or base_preset.get("python_exe") or sys.executable)
    if split_venv_pipeline and not Path(backtest_python_exe).expanduser().exists():
        raise SystemExit(f"Entry Sieve split-venv backtest Python does not exist: {backtest_python_exe}")
    state_file = runtime_dir / "state.json"
    state = explorer_load_json(state_file, {})
    if not isinstance(state, dict):
        state = {}

    total_runs = len(strategies) * len(training_windows)
    target_pair_count = len(_target_pairs(job))
    total_backtests = total_runs * len(validation_windows) * target_pair_count
    print(f"Entry Sieve job: {job_id}")
    print(f"Strategies: {len(strategies)} | training windows: {len(training_windows)} | validation windows: {len(validation_windows)} | runs: {total_runs}")
    if split_venv_pipeline:
        print(f"Split-venv backtests: {backtest_python_exe}")
    if target_sweep_enabled:
        pairs = _target_pairs(job)
        pair_text = ", ".join(f"{pair['take_profit_pct']}/{pair['stoploss_pct']}" for pair in pairs)
        print(f"Target sweep enabled: {pair_text}")
    _write_run_status(
        runtime_dir,
        job_id,
        status="running",
        phase="starting",
        total_hyperopts=total_runs,
        completed_hyperopts=0,
        total_backtests=total_backtests,
        completed_backtests=0,
        strategy_count=len(strategies),
        training_window_count=len(training_windows),
        validation_window_count=len(validation_windows),
        target_pair_count=target_pair_count,
        message="Entry Sieve run started",
    )

    run_index = 0
    pending_batches: list[PendingBacktests] = []
    final_completed_backtests = 0
    if target_sweep_enabled:
        for strategy in strategies:
            strategy_name = strategy.get("name") or Path(str(strategy.get("strategy_file") or "")).stem
            for training_window in training_windows:
                run_index += 1
                print(f"\nEntry Sieve {run_index}/{total_runs}: {strategy_name} | train={window_label(training_window)}")
                _write_run_status(
                    runtime_dir,
                    job_id,
                    status="running",
                    phase="hyperopt",
                    run_index=run_index,
                    total_hyperopts=total_runs,
                    completed_hyperopts=run_index - 1,
                    total_backtests=total_backtests,
                    completed_backtests=0,
                    current_strategy=strategy_name,
                    current_training_window=window_label(training_window),
                    message=f"Hyperopt {run_index}/{total_runs}: {strategy_name}",
                )
                try:
                    pending_batches.append(
                        _prepare_strategy_window(
                            job=job,
                            base_preset=base_preset,
                            runtime_dir=runtime_dir,
                            state=state,
                            strategy=strategy,
                            training_window=training_window,
                            validation_windows=validation_windows,
                        )
                    )
                    _write_run_status(
                        runtime_dir,
                        job_id,
                        status="running",
                        phase="hyperopt",
                        run_index=run_index,
                        total_hyperopts=total_runs,
                        completed_hyperopts=run_index,
                        total_backtests=total_backtests,
                        completed_backtests=0,
                        current_strategy=strategy_name,
                        current_training_window=window_label(training_window),
                        message=f"Hyperopt complete {run_index}/{total_runs}: {strategy_name}",
                    )
                except Exception as exc:
                    print(f"Entry Sieve run failed: {exc}")
                    for validation_window in validation_windows:
                        _append_result(
                            runtime_dir,
                            _result_row(
                                job_id=job_id,
                                strategy=strategy,
                                training_window=training_window,
                                validation_window=validation_window,
                                take_profit_pct=str(job.get("take_profit_pct") or "2"),
                                stoploss_pct=str(job.get("stoploss_pct") or "2"),
                                status="error",
                                error=str(exc),
                            ),
                        )
                    _write_run_status(
                        runtime_dir,
                        job_id,
                        status="running",
                        phase="hyperopt",
                        run_index=run_index,
                        total_hyperopts=total_runs,
                        completed_hyperopts=run_index,
                        total_backtests=total_backtests,
                        completed_backtests=0,
                        current_strategy=strategy_name,
                        current_training_window=window_label(training_window),
                        message=f"Hyperopt failed {run_index}/{total_runs}: {strategy_name}",
                    )
        _write_run_status(
            runtime_dir,
            job_id,
            status="running",
            phase="target_sweep_backtest",
            total_hyperopts=total_runs,
            completed_hyperopts=total_runs,
            total_backtests=total_backtests,
            completed_backtests=0,
            message="Target-sweep backtests started",
        )
        final_completed_backtests = _run_target_sweep_backtests(
            runtime_dir=runtime_dir,
            job_id=job_id,
            batches=pending_batches,
            lanes=_backtest_lanes(
                base_python_exe=base_python_exe,
                backtest_python_exe=backtest_python_exe,
                split_venv_pipeline=split_venv_pipeline,
            ),
            total_backtests=total_backtests,
        )
    else:
        pending: PendingBacktests | None = None
        completed_backtests = 0
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="entry-sieve-backtest") as executor:
            for strategy in strategies:
                strategy_name = strategy.get("name") or Path(str(strategy.get("strategy_file") or "")).stem
                for training_window in training_windows:
                    run_index += 1
                    print(f"\nEntry Sieve {run_index}/{total_runs}: {strategy_name} | train={window_label(training_window)}")
                    _write_run_status(
                        runtime_dir,
                        job_id,
                        status="running",
                        phase="hyperopt",
                        run_index=run_index,
                        total_hyperopts=total_runs,
                        completed_hyperopts=run_index - 1,
                        total_backtests=total_backtests,
                        completed_backtests=completed_backtests,
                        current_strategy=strategy_name,
                        current_training_window=window_label(training_window),
                        message=f"Hyperopt {run_index}/{total_runs}: {strategy_name}",
                    )
                    try:
                        batch = _prepare_strategy_window(
                            job=job,
                            base_preset=base_preset,
                            runtime_dir=runtime_dir,
                            state=state,
                            strategy=strategy,
                            training_window=training_window,
                            validation_windows=validation_windows,
                        )
                        if split_venv_pipeline:
                            completed_backtests += _finish_pending(
                                runtime_dir,
                                pending,
                                job_id=job_id,
                                completed_backtests=completed_backtests,
                                total_backtests=total_backtests,
                            )
                            batch.future = executor.submit(_run_backtest_batch, batch, backtest_python_exe)
                            pending = batch
                        else:
                            rows = _run_backtest_batch(batch, backtest_python_exe)
                            for row in rows:
                                _append_result(runtime_dir, row)
                            completed_backtests += len(rows)
                        _write_run_status(
                            runtime_dir,
                            job_id,
                            status="running",
                            phase="backtest" if split_venv_pipeline else "hyperopt_backtest",
                            run_index=run_index,
                            total_hyperopts=total_runs,
                            completed_hyperopts=run_index,
                            total_backtests=total_backtests,
                            completed_backtests=completed_backtests,
                            current_strategy=strategy_name,
                            current_training_window=window_label(training_window),
                            message=f"Run complete {run_index}/{total_runs}: {strategy_name}",
                        )
                    except Exception as exc:
                        print(f"Entry Sieve run failed: {exc}")
                        for validation_window in validation_windows:
                            _append_result(
                                runtime_dir,
                                _result_row(
                                    job_id=job_id,
                                    strategy=strategy,
                                    training_window=training_window,
                                    validation_window=validation_window,
                                    take_profit_pct=str(job.get("take_profit_pct") or "2"),
                                    stoploss_pct=str(job.get("stoploss_pct") or "2"),
                                    status="error",
                                    error=str(exc),
                                ),
                            )
                        completed_backtests += len(validation_windows)
                        _write_run_status(
                            runtime_dir,
                            job_id,
                            status="running",
                            phase="error",
                            run_index=run_index,
                            total_hyperopts=total_runs,
                            completed_hyperopts=run_index,
                            total_backtests=total_backtests,
                            completed_backtests=completed_backtests,
                            current_strategy=strategy_name,
                            current_training_window=window_label(training_window),
                            message=f"Run failed {run_index}/{total_runs}: {strategy_name}",
                        )
            completed_backtests += _finish_pending(
                runtime_dir,
                pending,
                job_id=job_id,
                completed_backtests=completed_backtests,
                total_backtests=total_backtests,
            )
        final_completed_backtests = completed_backtests
    _write_run_status(
        runtime_dir,
        job_id,
        status="finished",
        phase="finished",
        total_hyperopts=total_runs,
        completed_hyperopts=total_runs,
        total_backtests=total_backtests,
        completed_backtests=final_completed_backtests,
        message="Entry Sieve run finished",
    )
    state["updated_at"] = datetime.now().astimezone().isoformat()
    explorer_save_json(state_file, state)
    print(f"\nEntry Sieve results: {runtime_dir / 'results' / f'{_safe_name(job_id)}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

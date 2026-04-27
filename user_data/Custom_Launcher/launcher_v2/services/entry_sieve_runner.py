from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any

from explorer.explorer_catalog import load_catalog
from explorer.explorer_commands import (
    build_child_env,
    filter_best_params,
    load_json as explorer_load_json,
    merge_params_into_snapshot,
    run_backtest,
    run_hyperopt,
    save_json as explorer_save_json,
    score_objective,
    strategy_parameter_spaces,
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


def _prepare_runtime_strategy(source_file: Path, run_dir: Path) -> Path:
    strategy_dir = run_dir / "strategy"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    runtime_file = strategy_dir / source_file.name
    runtime_file.write_bytes(source_file.read_bytes())
    return runtime_file


def _child_env(job: dict[str, Any], cwd: Path, original_strategy_dir: Path) -> dict[str, str]:
    env = build_child_env(os.environ.copy(), cwd)
    extra_paths = [str(original_strategy_dir), str(original_strategy_dir.parent)]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(extra_paths) + (os.pathsep + existing if existing else "")
    env["ENTRY_SIEVE_TAKE_PROFIT_PCT"] = str(job.get("take_profit_pct") or "2")
    env["ENTRY_SIEVE_STOPLOSS_PCT"] = str(job.get("stoploss_pct") or "2")
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
        "training_window": window_label(training_window),
        "training_timerange": str(training_window.get("timerange") or ""),
        "validation_window": window_label(validation_window),
        "validation_timerange": str(validation_window.get("timerange") or ""),
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
    results_file = runtime_dir / "results.json"
    payload = load_json(results_file, {"rows": []})
    rows = payload.get("rows") if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        rows = []
    rows.append(row)
    save_json(
        results_file,
        {
            "schema_version": 2,
            "updated_at": datetime.now().astimezone().isoformat(),
            "rows": rows,
        },
    )


def _run_strategy_window(
    *,
    job: dict[str, Any],
    base_preset: dict[str, Any],
    runtime_dir: Path,
    state: dict[str, Any],
    strategy: dict[str, str],
    training_window: dict[str, Any],
    validation_windows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
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
    best_epoch, hyperopt_file, epoch_count = run_hyperopt(
        python_exe=python_exe,
        cwd=cwd,
        preset=preset,
        target=target,
        timerange=str(training_window.get("timerange") or ""),
        epochs=str(job.get("epochs") or "200"),
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
    rows: list[dict[str, Any]] = []
    for validation_window in validation_windows:
        metrics, backtest_file = run_backtest(
            python_exe=python_exe,
            cwd=cwd,
            preset=preset,
            timerange=str(validation_window.get("timerange") or ""),
            env=env,
        )
        rows.append(
            _result_row(
                job_id=str(job.get("job_id") or ""),
                strategy=strategy,
                training_window=training_window,
                validation_window=validation_window,
                status="ok",
                hyperopt_file=hyperopt_file,
                backtest_file=backtest_file,
                params_file=archive_params_file,
                best_params_count=sum(len(values) for values in candidate_params.values() if isinstance(values, dict)),
                epoch_count=epoch_count,
                hyperopt_loss=float(hyperopt_loss) if isinstance(hyperopt_loss, (int, float)) else None,
                metrics=metrics,
            )
        )
    state.setdefault("completed_runs", []).append(
        {
            "strategy": strategy.get("name") or source_file.stem,
            "strategy_class": strategy_class,
            "training_window": compact_window(training_window),
            "validation_windows": [compact_window(window) for window in validation_windows],
            "hyperopt_file": str(hyperopt_file),
            "params_file": str(archive_params_file),
            "best_params_count": sum(len(values) for values in candidate_params.values() if isinstance(values, dict)),
            "epoch_count": epoch_count,
            "finished_at": datetime.now().astimezone().isoformat(),
        }
    )
    return rows


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
    state_file = runtime_dir / "state.json"
    state = explorer_load_json(state_file, {})
    if not isinstance(state, dict):
        state = {}

    total_runs = len(strategies) * len(training_windows)
    print(f"Entry Sieve job: {job_id}")
    print(f"Strategies: {len(strategies)} | training windows: {len(training_windows)} | validation windows: {len(validation_windows)} | runs: {total_runs}")

    run_index = 0
    for strategy in strategies:
        strategy_name = strategy.get("name") or Path(str(strategy.get("strategy_file") or "")).stem
        for training_window in training_windows:
            run_index += 1
            print(f"\nEntry Sieve {run_index}/{total_runs}: {strategy_name} | train={window_label(training_window)}")
            try:
                rows = _run_strategy_window(
                    job=job,
                    base_preset=base_preset,
                    runtime_dir=runtime_dir,
                    state=state,
                    strategy=strategy,
                    training_window=training_window,
                    validation_windows=validation_windows,
                )
                for row in rows:
                    _append_result(runtime_dir, row)
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
                            status="error",
                            error=str(exc),
                        ),
                    )
    state["updated_at"] = datetime.now().astimezone().isoformat()
    explorer_save_json(state_file, state)
    print(f"\nEntry Sieve results: {runtime_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Split-venv Explorer pipeline runner.

This runner keeps normal Explorer writeback semantics under one controller while
allowing the next hyperopt to run during the previous loop's backtests.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

from .explorer_catalog import load_catalog
from .explorer_commands import (
    build_child_env,
    filter_best_params,
    load_json,
    merge_params_into_snapshot,
    run_backtest,
    run_hyperopt,
    save_json,
    score_objective,
    strategy_parameter_spaces,
)
from .explorer_metadata import emit_status, latest_summary, write_audit, write_latest_summary
from .explorer_runner import DEFAULT_STATE, _json_arg, print_score_table, print_validation_table, rejected_comparison
from .explorer_scoring import aggregate, compare, score_window
from .explorer_support import flatten_params
from .explorer_targets import choose_target, resolve_params, update_usage_counts
from .explorer_windows import compact_window, load_window_manifest, resolve_windows, window_label


@dataclass
class HyperoptHandoff:
    loop_index: int
    target_name: str
    target: dict[str, Any]
    training_window: dict[str, Any]
    validation_windows: list[dict[str, Any]]
    base_snapshot: dict[str, Any]
    candidate_snapshot: dict[str, Any]
    candidate_params: dict[str, dict[str, Any]]
    changes: list[dict[str, Any]]
    hyperopt_file: Path
    hyperopt_requested_epochs: int
    hyperopt_epoch_count: int
    handoff_file: Path
    champion_file: Path
    challenger_file: Path


@dataclass
class ValidationResult:
    champion_scored: list[Any]
    challenger_scored: list[Any]
    champion_records: list[dict[str, Any]]
    challenger_records: list[dict[str, Any]]


@dataclass
class PendingBacktest:
    handoff: HyperoptHandoff
    future: Future[ValidationResult]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split-venv Explorer pipeline runner.")
    parser.add_argument("--preset", required=True)
    parser.add_argument("--preset-file", required=True)
    parser.add_argument("--market-windows-file", required=True)
    parser.add_argument("--target-type", choices=["family", "mode"], required=True)
    parser.add_argument("--target-selection", choices=["random", "specific"], required=True)
    parser.add_argument("--target-name", default="")
    parser.add_argument("--search-breadth", choices=["targeted", "open"], default="targeted")
    parser.add_argument("--training-windows-json", type=_json_arg, required=True)
    parser.add_argument("--validation-windows-json", type=_json_arg, required=True)
    parser.add_argument("--max-loops", type=int, default=0)
    parser.add_argument("--epochs", default="")
    parser.add_argument("--auto-epochs", action="store_true")
    parser.add_argument("--auto-epochs-cap", default="")
    parser.add_argument("--random-state", default="")
    parser.add_argument("--sampling-seed", type=int, default=None)
    parser.add_argument("--backtest-workers", type=int, default=1)
    parser.add_argument("--metadata-file", required=True)
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--strategy-param-file", default="")
    parser.add_argument("--backtest-python-exe", required=True)
    parser.add_argument("--handoff-dir", required=True)
    return parser.parse_args(argv)


def _json_stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _same_snapshot(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _json_stable(left) == _json_stable(right)


def _safe_name(value: Any) -> str:
    text = str(value or "")
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text).strip("_") or "item"


def _verify_command(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        tail = "\n".join((completed.stdout or "").splitlines()[-20:])
        raise SystemExit(f"Backtest venv verification failed: {' '.join(command)}\n{tail}")


def verify_backtest_python(python_exe: str, cwd: Path) -> None:
    path = Path(python_exe).expanduser()
    if not path.exists():
        raise SystemExit(f"Split-venv backtest Python does not exist: {path}")
    commands = [
        [str(path), "--version"],
        [str(path), "-m", "freqtrade", "--version"],
        [str(path), "-m", "freqtrade", "hyperopt", "--help"],
        [str(path), "-m", "freqtrade", "backtesting", "--help"],
    ]
    for command in commands:
        _verify_command(command, cwd)
    print(f"Backtest venv verified: {path}")


def _snapshot_values(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    params = snapshot.get("params", {})
    return flatten_params(params if isinstance(params, dict) else {})


def _base_snapshot(strategy_class: str) -> dict[str, Any]:
    return {
        "strategy_name": strategy_class,
        "params": {},
        "ft_stratparam_v": 1,
        "export_time": datetime.now().astimezone().isoformat(),
    }


def normalize_strategy_snapshot(snapshot: dict[str, Any], strategy_class: str) -> dict[str, Any]:
    normalized = deepcopy(snapshot) if isinstance(snapshot, dict) else {}
    if not normalized:
        normalized = _base_snapshot(strategy_class)
    params = normalized.get("params")
    if not isinstance(params, dict):
        normalized["params"] = {}
    normalized["strategy_name"] = str(normalized.get("strategy_name") or strategy_class)
    normalized["ft_stratparam_v"] = int(normalized.get("ft_stratparam_v") or 1)
    normalized["export_time"] = str(normalized.get("export_time") or datetime.now().astimezone().isoformat())
    return normalized


def _is_strategy_snapshot(value: Any) -> bool:
    try:
        version = int(value.get("ft_stratparam_v") or 0) if isinstance(value, dict) else 0
    except (TypeError, ValueError):
        version = 0
    return (
        isinstance(value, dict)
        and isinstance(value.get("strategy_name"), str)
        and isinstance(value.get("params"), dict)
        and version >= 1
    )


def _write_handoff(handoff: HyperoptHandoff) -> None:
    save_json(
        handoff.handoff_file,
        {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "loop_index": handoff.loop_index,
            "target": handoff.target,
            "training_window": compact_window(handoff.training_window),
            "validation_windows": [compact_window(window) for window in handoff.validation_windows],
            "hyperopt_file": str(handoff.hyperopt_file),
            "hyperopt_requested_epochs": handoff.hyperopt_requested_epochs,
            "hyperopt_epoch_count": handoff.hyperopt_epoch_count,
            "candidate_params": handoff.candidate_params,
            "champion_file": str(handoff.champion_file),
            "challenger_file": str(handoff.challenger_file),
        },
    )


def _prepare_runtime_strategy(
    *,
    source_strategy_file: Path,
    strategy_class: str,
    snapshot_file: Path,
    run_dir: Path,
) -> tuple[Path, Path]:
    strategy_dir = run_dir / "strategy"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    runtime_strategy = strategy_dir / source_strategy_file.name
    runtime_params = runtime_strategy.with_suffix(".json")
    runtime_strategy.write_bytes(source_strategy_file.read_bytes())
    runtime_params.write_bytes(snapshot_file.read_bytes())
    return runtime_strategy, runtime_params


def _backtest_env(cwd: Path, source_strategy_dir: Path, userdir: Path) -> dict[str, str]:
    env = build_child_env(os.environ.copy(), cwd)
    extra_paths = [str(source_strategy_dir), str(source_strategy_dir.parent), str(userdir)]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(extra_paths) + (os.pathsep + existing if existing else "")
    return env


def _backtest_snapshot(
    *,
    python_exe: str,
    cwd: Path,
    preset: dict[str, Any],
    source_strategy_file: Path,
    strategy_class: str,
    snapshot_file: Path,
    validation_windows: list[dict[str, Any]],
    role: str,
    run_dir: Path,
) -> tuple[list[Any], list[dict[str, Any]]]:
    runtime_strategy, runtime_params = _prepare_runtime_strategy(
        source_strategy_file=source_strategy_file,
        strategy_class=strategy_class,
        snapshot_file=snapshot_file,
        run_dir=run_dir / role,
    )
    backtest_preset = deepcopy(preset)
    backtest_preset["strategy_file"] = str(runtime_strategy)
    backtest_preset["strategy_class"] = strategy_class
    userdir = Path(str(backtest_preset.get("userdir") or "user_data")).expanduser()
    env = _backtest_env(cwd, source_strategy_file.parent, userdir)

    scored = []
    records: list[dict[str, Any]] = []
    for window in validation_windows:
        timerange = str(window.get("timerange") or "")
        print(f"Starting split-venv {role} backtest: {window_label(window)} {timerange}")
        metrics, result_file = run_backtest(
            python_exe=python_exe,
            cwd=cwd,
            preset=backtest_preset,
            timerange=timerange,
            env=env,
        )
        objective = score_objective(metrics)
        scored_window = score_window(window, metrics, objective)
        scored.append(scored_window)
        records.append(
            {
                "role": role,
                "window": compact_window(window),
                "result_file": str(result_file),
                "params_file": str(runtime_params),
                "metrics": metrics,
                "score": scored_window.__dict__,
            }
        )
    return scored, records


def run_validation_pair(
    *,
    handoff: HyperoptHandoff,
    backtest_python_exe: str,
    cwd: Path,
    preset: dict[str, Any],
    source_strategy_file: Path,
    strategy_class: str,
    handoff_dir: Path,
) -> ValidationResult:
    run_dir = handoff_dir / f"loop_{handoff.loop_index:05d}" / "backtests"
    champion_scored, champion_records = _backtest_snapshot(
        python_exe=backtest_python_exe,
        cwd=cwd,
        preset=preset,
        source_strategy_file=source_strategy_file,
        strategy_class=strategy_class,
        snapshot_file=handoff.champion_file,
        validation_windows=handoff.validation_windows,
        role="champion",
        run_dir=run_dir,
    )
    challenger_scored, challenger_records = _backtest_snapshot(
        python_exe=backtest_python_exe,
        cwd=cwd,
        preset=preset,
        source_strategy_file=source_strategy_file,
        strategy_class=strategy_class,
        snapshot_file=handoff.challenger_file,
        validation_windows=handoff.validation_windows,
        role="challenger",
        run_dir=run_dir,
    )
    return ValidationResult(champion_scored, challenger_scored, champion_records, challenger_records)


def build_handoff(
    *,
    args: argparse.Namespace,
    loop_index: int,
    cwd: Path,
    preset: dict[str, Any],
    python_exe: str,
    env: dict[str, str],
    catalog: dict[str, Any],
    strategy_spaces: dict[str, str],
    current_snapshot: dict[str, Any],
    target_name: str,
    target: dict[str, Any],
    training_window: dict[str, Any],
    validation_windows: list[dict[str, Any]],
    loop_epochs_text: str,
    handoff_dir: Path,
) -> HyperoptHandoff | None:
    best_epoch, hyperopt_file, epoch_count = run_hyperopt(
        python_exe=python_exe,
        cwd=cwd,
        preset=preset,
        target=target,
        timerange=str(training_window.get("timerange") or ""),
        epochs=loop_epochs_text,
        random_state=args.random_state or None,
        env=env,
    )
    candidate_params = filter_best_params(best_epoch, target)
    if not candidate_params:
        return None
    current_snapshot = normalize_strategy_snapshot(current_snapshot, str(preset.get("strategy_class") or ""))
    candidate_snapshot = normalize_strategy_snapshot(merge_params_into_snapshot(current_snapshot, candidate_params), str(preset.get("strategy_class") or ""))
    changes = describe_changes(candidate_params, current_snapshot)
    loop_dir = handoff_dir / f"loop_{loop_index:05d}"
    champion_file = loop_dir / "champion.json"
    challenger_file = loop_dir / "challenger.json"
    handoff_file = loop_dir / "handoff.json"
    save_json(champion_file, current_snapshot)
    save_json(challenger_file, candidate_snapshot)
    handoff = HyperoptHandoff(
        loop_index=loop_index,
        target_name=target_name,
        target=target,
        training_window=training_window,
        validation_windows=validation_windows,
        base_snapshot=deepcopy(current_snapshot),
        candidate_snapshot=deepcopy(candidate_snapshot),
        candidate_params=candidate_params,
        changes=changes,
        hyperopt_file=hyperopt_file,
        hyperopt_requested_epochs=int(loop_epochs_text),
        hyperopt_epoch_count=epoch_count,
        handoff_file=handoff_file,
        champion_file=champion_file,
        challenger_file=challenger_file,
    )
    _write_handoff(handoff)
    return handoff


def describe_changes(candidate_params: dict[str, dict[str, Any]], snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    from .explorer_commands import describe_param_changes

    return describe_param_changes(candidate_params, _snapshot_values(snapshot))


def summarize_and_write(
    *,
    args: argparse.Namespace,
    run_id: str,
    metadata_file: Path,
    audit_file: Path,
    audit: dict[str, Any],
    strategy_file: Path,
    strategy_class: str,
    catalog: dict[str, Any],
    state: dict[str, Any],
    handoff: HyperoptHandoff,
    comparison: dict[str, Any],
    validation: ValidationResult | None,
    error: str = "",
) -> dict[str, Any]:
    summary = latest_summary(
        run_id=run_id,
        loop_index=handoff.loop_index,
        loop_total=args.max_loops,
        strategy_file=strategy_file,
        strategy_class=strategy_class,
        target=handoff.target,
        target_selection=args.target_selection,
        training_windows=[compact_window(handoff.training_window)],
        validation_windows=[compact_window(window) for window in handoff.validation_windows],
        comparison=comparison,
        params_changed_count=sum(1 for item in handoff.changes if item.get("changed")),
        catalog=catalog,
        state=state,
    )
    if error:
        summary["error"] = error
    write_latest_summary(metadata_file, summary)
    backtests = []
    if validation is not None:
        backtests = validation.champion_records + validation.challenger_records
    audit["loops"].append(
        {
            "loop_index": handoff.loop_index,
            "pipeline_mode": "split_venv",
            "handoff_file": str(handoff.handoff_file),
            "target": handoff.target,
            "training_window": compact_window(handoff.training_window),
            "validation_windows": [compact_window(window) for window in handoff.validation_windows],
            "hyperopt_file": str(handoff.hyperopt_file),
            "hyperopt_requested_epochs": handoff.hyperopt_requested_epochs,
            "hyperopt_epoch_count": handoff.hyperopt_epoch_count,
            "changes": handoff.changes,
            "comparison": comparison,
            "summary": summary,
            "backtests": backtests,
            "error": error,
        }
    )
    write_audit(audit_file, audit)
    return summary


def complete_pending(
    *,
    pending: PendingBacktest,
    args: argparse.Namespace,
    run_id: str,
    metadata_file: Path,
    audit_file: Path,
    audit: dict[str, Any],
    strategy_file: Path,
    strategy_class: str,
    strategy_param_file: Path,
    catalog: dict[str, Any],
    state: dict[str, Any],
    current_snapshot: dict[str, Any],
) -> dict[str, Any]:
    handoff = pending.handoff
    try:
        validation = pending.future.result()
        comparison = compare(aggregate(validation.champion_scored), aggregate(validation.challenger_scored))
        if comparison["accepted"]:
            current_snapshot = handoff.candidate_snapshot
            save_json(strategy_param_file, current_snapshot)
        else:
            save_json(strategy_param_file, current_snapshot)
        score_delta = (comparison.get("delta") or {}).get("final_score")
        update_usage_counts(state, args.target_type, handoff.target_name, args.search_breadth, bool(comparison["accepted"]), score_delta)
        save_json(Path(args.state_file), state)
        summary = summarize_and_write(
            args=args,
            run_id=run_id,
            metadata_file=metadata_file,
            audit_file=audit_file,
            audit=audit,
            strategy_file=strategy_file,
            strategy_class=strategy_class,
            catalog=catalog,
            state=state,
            handoff=handoff,
            comparison=comparison,
            validation=validation,
        )
        print_score_table(summary)
        print_validation_table(summary)
        print(f"\nDecision: {summary['decision_code']} | Guard: {summary['guard']} | Changed params: {summary['params_changed_count']}")
        emit_status(
            "loop_decision",
            {
                "run_id": run_id,
                "loop_index": handoff.loop_index,
                "decision_code": summary["decision_code"],
                "guard": summary["guard"],
                "accepted": summary["accepted"],
                "target_label": handoff.target["target_label"],
                "search_breadth": args.search_breadth,
                "params_changed_count": summary["params_changed_count"],
                "final_score_delta": (summary.get("delta") or {}).get("final_score"),
            },
        )
    except Exception as exc:
        save_json(strategy_param_file, current_snapshot)
        update_usage_counts(state, args.target_type, handoff.target_name, args.search_breadth, False, None)
        save_json(Path(args.state_file), state)
        comparison = rejected_comparison("REJECTED_BACKTEST_FAILED")
        summary = summarize_and_write(
            args=args,
            run_id=run_id,
            metadata_file=metadata_file,
            audit_file=audit_file,
            audit=audit,
            strategy_file=strategy_file,
            strategy_class=strategy_class,
            catalog=catalog,
            state=state,
            handoff=handoff,
            comparison=comparison,
            validation=None,
            error=str(exc),
        )
        emit_status("loop_error", {"run_id": run_id, "loop_index": handoff.loop_index, "error": str(exc)})
        print_score_table(summary)
        print_validation_table(summary)
        print(f"Explorer split-venv loop failed: {exc}")
    return current_snapshot


def reject_no_change_handoff(
    *,
    args: argparse.Namespace,
    run_id: str,
    metadata_file: Path,
    audit_file: Path,
    audit: dict[str, Any],
    strategy_file: Path,
    strategy_class: str,
    catalog: dict[str, Any],
    state: dict[str, Any],
    handoff: HyperoptHandoff,
) -> None:
    comparison = rejected_comparison("REJECTED_SCORE_NOT_IMPROVED")
    update_usage_counts(state, args.target_type, handoff.target_name, args.search_breadth, False, None)
    save_json(Path(args.state_file), state)
    summary = summarize_and_write(
        args=args,
        run_id=run_id,
        metadata_file=metadata_file,
        audit_file=audit_file,
        audit=audit,
        strategy_file=strategy_file,
        strategy_class=strategy_class,
        catalog=catalog,
        state=state,
        handoff=handoff,
        comparison=comparison,
        validation=None,
        error="Hyperopt returned no parameter changes; challenger was not validated.",
    )
    emit_status(
        "loop_decision",
        {
            "run_id": run_id,
            "loop_index": handoff.loop_index,
            "decision_code": summary["decision_code"],
            "guard": summary["guard"],
            "accepted": False,
            "target_label": handoff.target["target_label"],
            "search_breadth": args.search_breadth,
            "params_changed_count": 0,
            "final_score_delta": None,
        },
    )
    print_score_table(summary)
    print_validation_table(summary)
    print("Hyperopt returned no parameter changes; loop rejected before challenger validation.")


def _loop_epochs(args: argparse.Namespace, preset: dict[str, Any], target: dict[str, Any], manual_epochs_text: str, auto_epochs_cap: int) -> str:
    if args.auto_epochs:
        value = max(1, int(len(target.get("resolved_params") or [])) * 20)
        if auto_epochs_cap > 0:
            value = min(value, auto_epochs_cap)
        return str(value)
    return manual_epochs_text


def _print_loop_start(args: argparse.Namespace, run_id: str, loop_index: int, loop_total_display: str, target: dict[str, Any], training_window: dict[str, Any], validation_windows: list[dict[str, Any]], loop_epochs_text: str) -> None:
    print("\n" + "=" * 80)
    print(f"Explorer split-venv loop {loop_index}/{loop_total_display}")
    print(f"Target: {target['target_label']} | breadth={args.search_breadth} | params={len(target['resolved_params'])}")
    print(f"Hyperopt epochs: {loop_epochs_text}{' (auto)' if args.auto_epochs else ''}")
    print(f"Training window: {window_label(training_window)} {training_window.get('timerange')}")
    print("Validation windows: " + ", ".join(window_label(window) for window in validation_windows))
    emit_status(
        "loop_started",
        {
            "run_id": run_id,
            "loop_index": loop_index,
            "loop_total": args.max_loops,
            "target_type": args.target_type,
            "target_name": target["target_name"],
            "target_label": target["target_label"],
            "search_breadth": args.search_breadth,
            "resolved_param_count": len(target["resolved_params"]),
            "epochs": int(loop_epochs_text),
            "training_window": compact_window(training_window),
            "validation_window_count": len(validation_windows),
            "pipeline_mode": "split_venv",
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_loops < 0:
        raise SystemExit("--max-loops must be >= 0. Use 0 for infinite loop mode.")

    presets = load_json(Path(args.preset_file), {})
    if args.preset not in presets:
        raise SystemExit(f"Preset not found: {args.preset}")
    preset = deepcopy(presets[args.preset])
    manual_epochs_text = str(args.epochs or preset.get("hyperopt_epochs") or "100").strip() or "100"
    if not args.auto_epochs:
        try:
            manual_epochs_value = int(manual_epochs_text)
        except ValueError as exc:
            raise SystemExit(f"--epochs must be an integer when auto epochs is disabled: {manual_epochs_text}") from exc
        if manual_epochs_value < 1:
            raise SystemExit("--epochs must be >= 1 when auto epochs is disabled.")
        preset["hyperopt_epochs"] = str(manual_epochs_value)
        manual_epochs_text = str(manual_epochs_value)
    auto_epochs_cap = int(str(args.auto_epochs_cap or "0").strip() or "0")
    if auto_epochs_cap < 0:
        raise SystemExit("--auto-epochs-cap must be >= 0. Use 0 or blank for no limit.")
    if args.random_state:
        preset["hyperopt_random_state"] = str(args.random_state)

    project_root = str(preset.get("project_root") or "").strip()
    if not project_root:
        raise SystemExit("Preset must include project_root.")
    cwd = Path(project_root).expanduser().resolve()
    strategy_file = Path(str(preset.get("strategy_file") or "")).expanduser()
    if not strategy_file:
        raise SystemExit("Preset must include strategy_file.")
    if not strategy_file.is_absolute():
        strategy_file = (cwd / strategy_file).resolve()
    strategy_class = str(preset.get("strategy_class") or "").strip()
    if not strategy_class:
        raise SystemExit("Preset must include strategy_class.")
    strategy_param_file = Path(args.strategy_param_file).expanduser() if args.strategy_param_file else strategy_file.with_suffix(".json")
    if not strategy_param_file.is_absolute():
        strategy_param_file = (cwd / strategy_param_file).resolve()

    verify_backtest_python(args.backtest_python_exe, cwd)

    all_windows = load_window_manifest(args.market_windows_file)
    training_windows = resolve_windows(all_windows, args.training_windows_json, label="training")
    validation_windows = resolve_windows(all_windows, args.validation_windows_json, label="validation")
    rng = random.Random(args.sampling_seed)
    python_exe = str(preset.get("python_exe") or sys.executable)
    env = build_child_env(os.environ.copy(), cwd)
    state_file = Path(args.state_file)
    state = load_json(state_file, deepcopy(DEFAULT_STATE))
    if not isinstance(state, dict):
        state = deepcopy(DEFAULT_STATE)
    catalog = load_catalog(strategy_file, strategy_class)
    strategy_spaces = strategy_parameter_spaces(strategy_file, strategy_class)
    param_file_exists = strategy_param_file.exists()
    loaded_snapshot = load_json(strategy_param_file, {})
    if not isinstance(loaded_snapshot, dict):
        raise SystemExit(f"Strategy param JSON is not an object: {strategy_param_file}")
    if param_file_exists and not _is_strategy_snapshot(loaded_snapshot):
        raise SystemExit(f"Existing strategy param JSON is not a valid Freqtrade parameter file: {strategy_param_file}")
    current_snapshot = normalize_strategy_snapshot(loaded_snapshot, strategy_class)
    if not param_file_exists or not _same_snapshot(loaded_snapshot, current_snapshot):
        save_json(strategy_param_file, current_snapshot)

    run_id = datetime.now().strftime("explorer_pipeline_%Y%m%dT%H%M%S")
    metadata_file = Path(args.metadata_file)
    audit_file = metadata_file.parent / "runs" / f"{run_id}.json"
    handoff_dir = Path(args.handoff_dir).expanduser().resolve() / run_id
    handoff_dir.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {
        "schema_version": 2,
        "pipeline_mode": "split_venv",
        "run_id": run_id,
        "started_at": datetime.now().astimezone().isoformat(),
        "strategy_file": str(strategy_file.resolve()),
        "strategy_class": strategy_class,
        "backtest_python_exe": str(Path(args.backtest_python_exe).expanduser()),
        "handoff_dir": str(handoff_dir),
        "args": vars(args),
        "loops": [],
    }

    loop_total_display = "infinite" if args.max_loops == 0 else str(args.max_loops)
    next_loop_index = 1
    pending: PendingBacktest | None = None

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="explorer-backtest") as executor:
        while True:
            if args.max_loops > 0 and next_loop_index > args.max_loops:
                if pending is not None:
                    current_snapshot = complete_pending(
                        pending=pending,
                        args=args,
                        run_id=run_id,
                        metadata_file=metadata_file,
                        audit_file=audit_file,
                        audit=audit,
                        strategy_file=strategy_file,
                        strategy_class=strategy_class,
                        strategy_param_file=strategy_param_file,
                        catalog=catalog,
                        state=state,
                        current_snapshot=current_snapshot,
                    )
                break

            target_name = choose_target(catalog, args.target_type, args.target_selection, args.target_name, args.search_breadth, state, rng)
            training_window = rng.choice(training_windows)
            try:
                target = resolve_params(catalog, args.target_type, target_name, args.search_breadth, strategy_spaces)
            except Exception as exc:
                comparison = rejected_comparison("REJECTED_NO_TUNABLE_PARAMS")
                update_usage_counts(state, args.target_type, target_name, args.search_breadth, False, None)
                save_json(state_file, state)
                print(f"Explorer split-venv loop rejected: {exc}")
                next_loop_index += 1
                continue

            loop_epochs_text = _loop_epochs(args, preset, target, manual_epochs_text, auto_epochs_cap)
            _print_loop_start(args, run_id, next_loop_index, loop_total_display, target, training_window, validation_windows, loop_epochs_text)
            handoff = build_handoff(
                args=args,
                loop_index=next_loop_index,
                cwd=cwd,
                preset=preset,
                python_exe=python_exe,
                env=env,
                catalog=catalog,
                strategy_spaces=strategy_spaces,
                current_snapshot=deepcopy(current_snapshot),
                target_name=target_name,
                target=target,
                training_window=training_window,
                validation_windows=validation_windows,
                loop_epochs_text=loop_epochs_text,
                handoff_dir=handoff_dir,
            )
            if handoff is None:
                update_usage_counts(state, args.target_type, target_name, args.search_breadth, False, None)
                save_json(state_file, state)
                print("No tunable params were returned for the target; loop rejected.")
                next_loop_index += 1
                continue
            if not any(bool(item.get("changed")) for item in handoff.changes):
                if pending is not None:
                    current_snapshot = complete_pending(
                        pending=pending,
                        args=args,
                        run_id=run_id,
                        metadata_file=metadata_file,
                        audit_file=audit_file,
                        audit=audit,
                        strategy_file=strategy_file,
                        strategy_class=strategy_class,
                        strategy_param_file=strategy_param_file,
                        catalog=catalog,
                        state=state,
                        current_snapshot=current_snapshot,
                    )
                    pending = None
                reject_no_change_handoff(
                    args=args,
                    run_id=run_id,
                    metadata_file=metadata_file,
                    audit_file=audit_file,
                    audit=audit,
                    strategy_file=strategy_file,
                    strategy_class=strategy_class,
                    catalog=catalog,
                    state=state,
                    handoff=handoff,
                )
                next_loop_index += 1
                continue

            if pending is not None:
                current_snapshot = complete_pending(
                    pending=pending,
                    args=args,
                    run_id=run_id,
                    metadata_file=metadata_file,
                    audit_file=audit_file,
                    audit=audit,
                    strategy_file=strategy_file,
                    strategy_class=strategy_class,
                    strategy_param_file=strategy_param_file,
                    catalog=catalog,
                    state=state,
                    current_snapshot=current_snapshot,
                )
                if not _same_snapshot(handoff.base_snapshot, current_snapshot):
                    audit.setdefault("discarded_hyperopts", []).append(
                        {
                            "loop_index": handoff.loop_index,
                            "reason": "base_snapshot_changed_after_previous_decision",
                            "hyperopt_file": str(handoff.hyperopt_file),
                            "handoff_file": str(handoff.handoff_file),
                            "discarded_at": datetime.now().astimezone().isoformat(),
                        }
                    )
                    write_audit(audit_file, audit)
                    print(f"Discarding stale split-venv hyperopt for loop {handoff.loop_index}; rerunning on updated params.")
                    handoff = build_handoff(
                        args=args,
                        loop_index=next_loop_index,
                        cwd=cwd,
                        preset=preset,
                        python_exe=python_exe,
                        env=env,
                        catalog=catalog,
                        strategy_spaces=strategy_spaces,
                        current_snapshot=deepcopy(current_snapshot),
                        target_name=target_name,
                        target=target,
                        training_window=training_window,
                        validation_windows=validation_windows,
                        loop_epochs_text=loop_epochs_text,
                        handoff_dir=handoff_dir,
                    )
                    if handoff is None:
                        update_usage_counts(state, args.target_type, target_name, args.search_breadth, False, None)
                        save_json(state_file, state)
                        next_loop_index += 1
                        pending = None
                        continue
                    if not any(bool(item.get("changed")) for item in handoff.changes):
                        reject_no_change_handoff(
                            args=args,
                            run_id=run_id,
                            metadata_file=metadata_file,
                            audit_file=audit_file,
                            audit=audit,
                            strategy_file=strategy_file,
                            strategy_class=strategy_class,
                            catalog=catalog,
                            state=state,
                            handoff=handoff,
                        )
                        next_loop_index += 1
                        pending = None
                        continue

            future = executor.submit(
                run_validation_pair,
                handoff=handoff,
                backtest_python_exe=str(Path(args.backtest_python_exe).expanduser()),
                cwd=cwd,
                preset=deepcopy(preset),
                source_strategy_file=strategy_file,
                strategy_class=strategy_class,
                handoff_dir=handoff_dir,
            )
            pending = PendingBacktest(handoff=handoff, future=future)
            next_loop_index += 1

    audit["finished_at"] = datetime.now().astimezone().isoformat()
    write_audit(audit_file, audit)
    save_json(state_file, state)
    save_json(strategy_param_file, current_snapshot)
    print(f"\nExplorer latest summary JSON: {metadata_file}")
    print(f"Explorer audit JSON: {audit_file}")
    print(f"Explorer state JSON: {state_file}")
    print(f"Explorer split-venv handoff dir: {handoff_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

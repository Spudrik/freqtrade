#!/usr/bin/env python3
"""Simplified Explorer runner.

Normal workflow:
family/mode target -> targeted/open breadth -> HyperOpt training window -> validation window(s)
-> ExplorerAcceptanceScoreV1 -> apply or reject.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import itertools
import json
from pathlib import Path
import random
import sys
from typing import Any

from .explorer_catalog import load_catalog
from .explorer_commands import (
    build_child_env,
    current_strategy_values,
    describe_param_changes,
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
from .explorer_scoring import aggregate, compare, score_window
from .explorer_targets import choose_target, resolve_params, update_usage_counts
from .explorer_windows import compact_window, load_window_manifest, resolve_windows, window_label


DEFAULT_STATE = {
    "family_usage_counts": {},
    "mode_usage_counts": {},
    "family_open_usage_counts": {},
    "mode_open_usage_counts": {},
    "family_accept_counts": {},
    "mode_accept_counts": {},
    "last_run_by_target": {},
    "last_score_delta_by_target": {},
}


def _empty_score() -> dict[str, Any]:
    return {
        "objective_total": 0.0,
        "max_drawdown_pct": 0.0,
        "drawdown_penalty_total": 0.0,
        "final_score": 0.0,
        "profit_total": 0.0,
        "trade_count": 0,
        "losing_window_count": 0,
        "winrate": 0.0,
        "hard_guard": "PASS",
        "windows": [],
    }


def _missing_score() -> dict[str, Any]:
    score = _empty_score()
    for key in (
        "objective_total",
        "max_drawdown_pct",
        "drawdown_penalty_total",
        "final_score",
        "profit_total",
        "trade_count",
        "losing_window_count",
        "winrate",
    ):
        score[key] = None
    return score


def _score_payload(scored_windows: list[Any] | None) -> dict[str, Any]:
    if scored_windows is None:
        return _empty_score()
    if not scored_windows:
        return _missing_score()
    return asdict(aggregate(scored_windows))


def _comparison_delta(champion: dict[str, Any], challenger: dict[str, Any]) -> dict[str, float | int | None]:
    fields = (
        "objective_total",
        "max_drawdown_pct",
        "drawdown_penalty_total",
        "final_score",
        "profit_total",
        "trade_count",
        "losing_window_count",
        "winrate",
    )
    values: dict[str, float | int | None] = {}
    for field in fields:
        champ_value = champion.get(field)
        chall_value = challenger.get(field)
        if isinstance(champ_value, (int, float)) and isinstance(chall_value, (int, float)):
            values[field] = chall_value - champ_value
        else:
            values[field] = None
    return values


def rejected_comparison(
    decision_code: str,
    *,
    guard: str = "PASS",
    champion_scored: list[Any] | None = None,
    challenger_scored: list[Any] | None = None,
) -> dict[str, Any]:
    champion = _score_payload(champion_scored)
    challenger = _score_payload(challenger_scored)
    challenger["hard_guard"] = guard
    return {
        "accepted": False,
        "decision_code": decision_code,
        "guard": guard,
        "champion": champion,
        "challenger": challenger,
        "delta": _comparison_delta(champion, challenger),
    }


def _json_arg(value: str) -> list[Any]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not isinstance(parsed, list):
        raise argparse.ArgumentTypeError("value must be a JSON list")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simplified family/mode Explorer runner.")
    parser.add_argument("--preset", required=True, help="Launcher preset name to reuse.")
    parser.add_argument("--preset-file", required=True, help="Preset JSON file to read.")
    parser.add_argument("--market-windows-file", required=True, help="Market-window manifest JSON file.")
    parser.add_argument("--target-type", choices=["family", "mode"], required=True)
    parser.add_argument("--target-selection", choices=["random", "specific"], required=True)
    parser.add_argument("--target-name", default="", help="Required when --target-selection specific.")
    parser.add_argument("--search-breadth", choices=["targeted", "open"], default="targeted")
    parser.add_argument("--training-windows-json", type=_json_arg, required=True)
    parser.add_argument("--validation-windows-json", type=_json_arg, required=True)
    parser.add_argument("--max-loops", type=int, default=0)
    parser.add_argument("--epochs", default="")
    parser.add_argument("--auto-epochs", action="store_true", help="Set epochs per loop to 20x resolved param count.")
    parser.add_argument("--auto-epochs-cap", default="", help="Optional cap for auto epochs. 0/blank means no cap.")
    parser.add_argument("--random-state", default="")
    parser.add_argument("--sampling-seed", type=int, default=None)
    parser.add_argument("--backtest-workers", type=int, default=1, help="Accepted for UI compatibility; simplified runner currently validates serially.")
    parser.add_argument("--metadata-file", required=True)
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--strategy-param-file", default="")
    return parser.parse_args(argv)


def print_score_table(summary: dict[str, Any]) -> None:
    print("\nChampion vs Challenger")
    print(f"{'Metric':<22} {'Champion':>14} {'Challenger':>14} {'Delta':>14} {'Role':<12}")
    for row in summary.get("score_table") or []:
        print(
            f"{str(row.get('metric')):<22} "
            f"{_fmt(row.get('champion')):>14} "
            f"{_fmt(row.get('challenger')):>14} "
            f"{_fmt(row.get('delta')):>14} "
            f"{str(row.get('role')):<12}"
        )


def print_validation_table(summary: dict[str, Any]) -> None:
    print("\nValidation windows")
    print(f"{'Window':<34} {'Regime':<10} {'Champ score':>13} {'Chall score':>13} {'Delta':>13} {'Chall DD%':>10} {'Guard':<6}")
    for row in summary.get("validation_breakdown") or []:
        print(
            f"{str(row.get('window'))[:34]:<34} "
            f"{str(row.get('regime')):<10} "
            f"{_fmt(row.get('champion_score')):>13} "
            f"{_fmt(row.get('challenger_score')):>13} "
            f"{_fmt(row.get('delta_score')):>13} "
            f"{_fmt(row.get('challenger_max_drawdown_pct')):>10} "
            f"{str(row.get('guard')):<6}"
        )


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def backtest_snapshot(
    *,
    python_exe: str,
    cwd: Path,
    preset: dict[str, Any],
    strategy_param_file: Path,
    snapshot: dict[str, Any],
    validation_windows: list[dict[str, Any]],
    env: dict[str, str],
    role: str,
) -> tuple[list[Any], list[dict[str, Any]]]:
    save_json(strategy_param_file, snapshot)
    scored = []
    records: list[dict[str, Any]] = []
    for window in validation_windows:
        timerange = str(window.get("timerange") or "")
        print(f"Starting {role} validation backtest: {window_label(window)} {timerange}")
        metrics, result_file = run_backtest(
            python_exe=python_exe,
            cwd=cwd,
            preset=preset,
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
                "metrics": metrics,
                "score": scored_window.__dict__,
            }
        )
    return scored, records


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_loops not in {0, 1}:
        raise SystemExit("--max-loops must be 0 (infinite loop) or 1 (single loop).")

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
    auto_epochs_cap_text = str(args.auto_epochs_cap or "").strip()
    auto_epochs_cap = 0
    if auto_epochs_cap_text:
        try:
            auto_epochs_cap = int(auto_epochs_cap_text)
        except ValueError as exc:
            raise SystemExit("--auto-epochs-cap must be an integer. Use 0 or blank for no limit.") from exc
        if auto_epochs_cap < 0:
            raise SystemExit("--auto-epochs-cap must be >= 0. Use 0 or blank for no limit.")
    if args.random_state:
        preset["hyperopt_random_state"] = str(args.random_state)

    strategy_file = Path(str(preset.get("strategy_file") or "")).expanduser()
    if not strategy_file:
        raise SystemExit("Preset must include strategy_file.")
    project_root = str(preset.get("project_root") or "").strip()
    if not project_root:
        raise SystemExit("Preset must include project_root.")
    cwd = Path(project_root).expanduser()
    if not strategy_file.is_absolute():
        strategy_file = (cwd / strategy_file).resolve()
    strategy_class = str(preset.get("strategy_class") or "").strip()
    if not strategy_class:
        raise SystemExit("Preset must include strategy_class.")
    strategy_param_file = Path(args.strategy_param_file).expanduser() if args.strategy_param_file else strategy_file.with_suffix(".json")
    if not strategy_param_file.is_absolute():
        strategy_param_file = (cwd / strategy_param_file).resolve()

    all_windows = load_window_manifest(args.market_windows_file)
    training_windows = resolve_windows(all_windows, args.training_windows_json, label="training")
    validation_windows = resolve_windows(all_windows, args.validation_windows_json, label="validation")

    rng = random.Random(args.sampling_seed)
    python_exe = str(preset.get("python_exe") or sys.executable)
    env = build_child_env(__import__("os").environ.copy(), cwd)
    state_file = Path(args.state_file)
    state = load_json(state_file, deepcopy(DEFAULT_STATE))
    if not isinstance(state, dict):
        state = deepcopy(DEFAULT_STATE)

    catalog = load_catalog(strategy_file, strategy_class)
    strategy_spaces = strategy_parameter_spaces(strategy_file, strategy_class)
    current_snapshot = load_json(strategy_param_file, {})
    if not isinstance(current_snapshot, dict):
        raise SystemExit(f"Strategy param JSON is not an object: {strategy_param_file}")

    run_id = datetime.now().strftime("explorer_clean_%Y%m%dT%H%M%S")
    metadata_file = Path(args.metadata_file)
    audit_file = metadata_file.parent / "runs" / f"{run_id}.json"
    audit: dict[str, Any] = {
        "schema_version": 2,
        "run_id": run_id,
        "started_at": datetime.now().astimezone().isoformat(),
        "strategy_file": str(strategy_file.resolve()),
        "strategy_class": strategy_class,
        "args": vars(args),
        "loops": [],
    }

    loop_total_display = "∞" if args.max_loops == 0 else str(args.max_loops)
    for loop_index in itertools.count(1):
        if args.max_loops == 1 and loop_index > 1:
            break
        target_name = choose_target(
            catalog,
            args.target_type,
            args.target_selection,
            args.target_name,
            args.search_breadth,
            state,
            rng,
        )
        training_window = rng.choice(training_windows)
        try:
            target = resolve_params(catalog, args.target_type, target_name, args.search_breadth, strategy_spaces)
        except Exception as exc:
            target = {
                "target_type": args.target_type,
                "target_name": target_name,
                "target_label": f"{args.target_type}:{target_name}",
                "search_breadth": args.search_breadth,
                "primary_params": [],
                "resolved_params": [],
                "spaces": [],
                "support_families": [],
                "missing_open_support_families": [],
            }
            comparison = rejected_comparison("REJECTED_NO_TUNABLE_PARAMS")
            update_usage_counts(state, args.target_type, target_name, args.search_breadth, False, None)
            save_json(state_file, state)
            summary = latest_summary(
                run_id=run_id,
                loop_index=loop_index,
                loop_total=args.max_loops,
                strategy_file=strategy_file,
                strategy_class=strategy_class,
                target=target,
                target_selection=args.target_selection,
                training_windows=[compact_window(training_window)],
                validation_windows=[compact_window(window) for window in validation_windows],
                comparison=comparison,
                params_changed_count=0,
                catalog=catalog,
                state=state,
            )
            write_latest_summary(metadata_file, summary)
            audit["loops"].append(
                {
                    "loop_index": loop_index,
                    "target": target,
                    "training_window": compact_window(training_window),
                    "validation_windows": [compact_window(window) for window in validation_windows],
                    "error": str(exc),
                    "decision_code": "REJECTED_NO_TUNABLE_PARAMS",
                }
            )
            write_audit(audit_file, audit)
            emit_status(
                "loop_decision",
                {
                    "run_id": run_id,
                    "loop_index": loop_index,
                    "decision_code": "REJECTED_NO_TUNABLE_PARAMS",
                    "guard": "PASS",
                    "accepted": False,
                    "target_label": target["target_label"],
                    "search_breadth": args.search_breadth,
                    "params_changed_count": 0,
                    "final_score_delta": 0.0,
                },
            )
            print(f"Explorer loop rejected: {exc}")
            continue

        if args.auto_epochs:
            loop_epochs_value = max(1, int(len(target.get("resolved_params") or [])) * 20)
            if auto_epochs_cap > 0:
                loop_epochs_value = min(loop_epochs_value, auto_epochs_cap)
            loop_epochs_text = str(loop_epochs_value)
        else:
            loop_epochs_text = manual_epochs_text

        print("\n" + "=" * 80)
        print(f"Explorer loop {loop_index}/{loop_total_display}")
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
                "target_name": target_name,
                "target_label": target["target_label"],
                "search_breadth": args.search_breadth,
                "resolved_param_count": len(target["resolved_params"]),
                "epochs": int(loop_epochs_text),
                "training_window": compact_window(training_window),
                "validation_window_count": len(validation_windows),
            },
        )

        champion_snapshot = deepcopy(current_snapshot)
        hyperopt_file: Path | None = None
        epoch_count = 0
        changes: list[dict[str, Any]] = []
        champion_scored: list[Any] = []
        challenger_scored: list[Any] = []
        champion_records: list[dict[str, Any]] = []
        challenger_records: list[dict[str, Any]] = []
        try:
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
                update_usage_counts(state, args.target_type, target_name, args.search_breadth, False, None)
                save_json(state_file, state)
                comparison = rejected_comparison("REJECTED_NO_TUNABLE_PARAMS")
                summary = latest_summary(
                    run_id=run_id,
                    loop_index=loop_index,
                    loop_total=args.max_loops,
                    strategy_file=strategy_file,
                    strategy_class=strategy_class,
                    target=target,
                    target_selection=args.target_selection,
                    training_windows=[compact_window(training_window)],
                    validation_windows=[compact_window(window) for window in validation_windows],
                    comparison=comparison,
                    params_changed_count=0,
                    catalog=catalog,
                    state=state,
                )
                write_latest_summary(metadata_file, summary)
                audit["loops"].append(
                    {
                        "loop_index": loop_index,
                        "target": target,
                        "training_window": compact_window(training_window),
                        "validation_windows": [compact_window(window) for window in validation_windows],
                        "hyperopt_file": str(hyperopt_file),
                        "hyperopt_requested_epochs": int(loop_epochs_text),
                        "hyperopt_epoch_count": epoch_count,
                        "decision_code": "REJECTED_NO_TUNABLE_PARAMS",
                    }
                )
                write_audit(audit_file, audit)
                print("No tunable params were returned for the target; loop rejected.")
                continue
            candidate_snapshot = merge_params_into_snapshot(champion_snapshot, candidate_params)
            changes = describe_param_changes(candidate_params, current_strategy_values(strategy_param_file))

            champion_scored, champion_records = backtest_snapshot(
                python_exe=python_exe,
                cwd=cwd,
                preset=preset,
                strategy_param_file=strategy_param_file,
                snapshot=champion_snapshot,
                validation_windows=validation_windows,
                env=env,
                role="champion",
            )
            challenger_scored, challenger_records = backtest_snapshot(
                python_exe=python_exe,
                cwd=cwd,
                preset=preset,
                strategy_param_file=strategy_param_file,
                snapshot=candidate_snapshot,
                validation_windows=validation_windows,
                env=env,
                role="challenger",
            )
            comparison = compare(aggregate(champion_scored), aggregate(challenger_scored))
            if comparison["accepted"]:
                current_snapshot = candidate_snapshot
                save_json(strategy_param_file, current_snapshot)
            else:
                save_json(strategy_param_file, champion_snapshot)
            score_delta = (comparison.get("delta") or {}).get("final_score")
            update_usage_counts(state, args.target_type, target_name, args.search_breadth, bool(comparison["accepted"]), score_delta)
            save_json(state_file, state)

            summary = latest_summary(
                run_id=run_id,
                loop_index=loop_index,
                loop_total=args.max_loops,
                strategy_file=strategy_file,
                strategy_class=strategy_class,
                target=target,
                target_selection=args.target_selection,
                training_windows=[compact_window(training_window)],
                validation_windows=[compact_window(window) for window in validation_windows],
                comparison=comparison,
                params_changed_count=sum(1 for item in changes if item.get("changed")),
                catalog=catalog,
                state=state,
            )
            write_latest_summary(metadata_file, summary)
            print_score_table(summary)
            print_validation_table(summary)
            print(f"\nDecision: {summary['decision_code']} | Guard: {summary['guard']} | Changed params: {summary['params_changed_count']}")
            emit_status(
                "loop_decision",
                {
                    "run_id": run_id,
                    "loop_index": loop_index,
                    "decision_code": summary["decision_code"],
                    "guard": summary["guard"],
                    "accepted": summary["accepted"],
                    "target_label": target["target_label"],
                    "search_breadth": args.search_breadth,
                    "params_changed_count": summary["params_changed_count"],
                    "final_score_delta": (summary.get("delta") or {}).get("final_score"),
                },
            )
            audit["loops"].append(
                {
                    "loop_index": loop_index,
                    "target": target,
                    "training_window": compact_window(training_window),
                    "validation_windows": [compact_window(window) for window in validation_windows],
                    "hyperopt_file": str(hyperopt_file),
                    "hyperopt_requested_epochs": int(loop_epochs_text),
                    "hyperopt_epoch_count": epoch_count,
                    "changes": changes,
                    "comparison": comparison,
                    "summary": summary,
                    "backtests": champion_records + challenger_records,
                }
            )
            write_audit(audit_file, audit)
        except Exception as exc:
            save_json(strategy_param_file, champion_snapshot)
            update_usage_counts(state, args.target_type, target_name, args.search_breadth, False, None)
            save_json(state_file, state)
            comparison = rejected_comparison(
                "REJECTED_BACKTEST_FAILED",
                champion_scored=champion_scored,
                challenger_scored=challenger_scored,
            )
            summary = latest_summary(
                run_id=run_id,
                loop_index=loop_index,
                loop_total=args.max_loops,
                strategy_file=strategy_file,
                strategy_class=strategy_class,
                target=target,
                target_selection=args.target_selection,
                training_windows=[compact_window(training_window)],
                validation_windows=[compact_window(window) for window in validation_windows],
                comparison=comparison,
                params_changed_count=0,
                catalog=catalog,
                state=state,
            )
            summary["error"] = str(exc)
            write_latest_summary(metadata_file, summary)
            error_record = {
                "loop_index": loop_index,
                "target": target,
                "training_window": compact_window(training_window),
                "validation_windows": [compact_window(window) for window in validation_windows],
                "error": str(exc),
                "decision_code": "REJECTED_BACKTEST_FAILED",
                "hyperopt_file": str(hyperopt_file) if hyperopt_file is not None else "",
                "hyperopt_requested_epochs": int(loop_epochs_text),
                "hyperopt_epoch_count": epoch_count,
                "changes": changes,
                "comparison": comparison,
                "summary": summary,
                "backtests": champion_records + challenger_records,
            }
            audit["loops"].append(error_record)
            write_audit(audit_file, audit)
            emit_status("loop_error", error_record)
            print_score_table(summary)
            print_validation_table(summary)
            print(f"Explorer loop failed: {exc}")

    audit["finished_at"] = datetime.now().astimezone().isoformat()
    write_audit(audit_file, audit)
    save_json(state_file, state)
    save_json(strategy_param_file, current_snapshot)
    print(f"\nExplorer latest summary JSON: {metadata_file}")
    print(f"Explorer audit JSON: {audit_file}")
    print(f"Explorer state JSON: {state_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

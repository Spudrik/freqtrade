"""Command helpers that wrap existing Freqtrade Explorer support functions."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from .explorer_support import (
    PARAM_ENV,
    backtest_results_dir,
    best_epoch_from_file,
    build_backtest_args,
    build_child_env,
    build_hyperopt_args,
    current_strategy_values,
    describe_param_changes,
    filtered_params,
    hyperopt_results_dir,
    latest_result_file,
    load_json,
    merge_params_into_snapshot,
    parse_backtest_metrics,
    result_file_snapshot,
    run_command,
    save_json,
    segment_objective,
    strategy_parameter_spaces,
)


def run_hyperopt(
    *,
    python_exe: str,
    cwd: Path,
    preset: dict[str, Any],
    target: dict[str, Any],
    timerange: str,
    epochs: str | None,
    random_state: str | None,
    env: dict[str, str],
) -> tuple[dict[str, Any], Path, int]:
    group = {"spaces": target.get("spaces") or []}
    directory = hyperopt_results_dir(preset)
    snapshot = result_file_snapshot(directory, ("*.fthypt", "*.json"))
    command = [python_exe, "-u", "-m", "freqtrade", *build_hyperopt_args(preset, group, timerange, epochs, random_state)]
    child_env = dict(env)
    child_env[PARAM_ENV] = ",".join(sorted(target.get("resolved_params") or []))
    started_at = time.time()
    completed = run_command(command, cwd, child_env, dry_run=False, stream_output=True)
    if completed is None or completed.returncode != 0:
        raise RuntimeError(f"HyperOpt failed for {target.get('target_label')} on {timerange}.")
    result = latest_result_file(directory, ("*.fthypt", "*.json"), started_at, None, snapshot)
    if result is None:
        raise RuntimeError("HyperOpt completed but no result file was found.")
    best_epoch, epoch_count = best_epoch_from_file(result)
    return best_epoch, result, epoch_count


def run_backtest(
    *,
    python_exe: str,
    cwd: Path,
    preset: dict[str, Any],
    timerange: str,
    env: dict[str, str],
) -> tuple[dict[str, Any], Path]:
    directory = backtest_results_dir(preset)
    snapshot = result_file_snapshot(directory, ("*.zip", "*.json"))
    command = [python_exe, "-u", "-m", "freqtrade", *build_backtest_args(preset, timerange)]
    started_at = time.time()
    completed = run_command(command, cwd, env, dry_run=False, stream_output=True)
    if completed is None or completed.returncode != 0:
        raise RuntimeError(f"Backtest failed for {timerange}.")
    result = latest_result_file(directory, ("*.zip", "*.json"), started_at, None, snapshot)
    if result is None:
        raise RuntimeError("Backtest completed but no result file was found.")
    return parse_backtest_metrics(result), result


def filter_best_params(best_epoch: dict[str, Any], target: dict[str, Any]) -> dict[str, dict[str, Any]]:
    params = best_epoch.get("params") if isinstance(best_epoch.get("params"), dict) else {}
    if not params:
        params = best_epoch.get("params_details") if isinstance(best_epoch.get("params_details"), dict) else {}
    return filtered_params(params, set(target.get("resolved_params") or []))


def score_objective(metrics: dict[str, Any]) -> float:
    return float(segment_objective(metrics))

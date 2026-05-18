from __future__ import annotations

import argparse
import ast
import atexit
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
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
    metric_summary,
    parse_backtest_metrics,
    result_file_snapshot,
    run_command,
)
from explorer.explorer_targets import resolve_params
from explorer.explorer_windows import compact_window, load_window_manifest, normalize_window, resolve_windows, window_label

from .entry_sieve_lock import EntrySieveRunLock


BACKTEST_LANE_START_STAGGER_SECONDS = 1.0
MAX_BACKTEST_WORKERS = 20
RESERVED_SYSTEM_WORKERS = 2
BACKTEST_WAIT_STATUS_SECONDS = 60.0
BACKTEST_START_STAGGER_SECONDS = 15.0
BACKTEST_START_RAM_LIMIT_PERCENT = 80.0
SPEED_RUN_EPOCH_CAP = 120
FULL_CYCLE_VALIDATION_WINDOW = "full_cycle_2020_2026"
TIMEFRAME_SECONDS = {
    "5m": 5 * 60,
    "15m": 15 * 60,
    "30m": 30 * 60,
    "1h": 60 * 60,
    "2h": 2 * 60 * 60,
    "4h": 4 * 60 * 60,
    "8h": 8 * 60 * 60,
    "12h": 12 * 60 * 60,
    "1d": 24 * 60 * 60,
    "3d": 3 * 24 * 60 * 60,
}
MIN_TRAINING_CANDLES = {
    "1h": 1000,
    "4h": 600,
    "8h": 450,
    "1d": 500,
    "3d": 250,
}
TARGET_TRAINING_CANDLES = {
    "1h": 3000,
    "4h": 1200,
    "8h": 900,
    "1d": 700,
    "3d": 350,
}
RESULT_METADATA_KEYS = (
    "strategy_batch",
    "strategy_batch_label",
    "strategy_filter",
    "speed_run_mode",
    "speed_pair_count",
    "auto_window_mode",
    "auto_window_count",
    "target_sweep_enabled",
)
ACTIVE_ENTRY_SIEVE_LOCK: EntrySieveRunLock | None = None


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


def _listish(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    text = str(value or "").replace(";", ",").replace("|", ",")
    return [item.strip().lower() for item in text.split(",") if item.strip()]


def _literal_or_name(node: ast.AST, constants: dict[str, Any] | None = None) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and constants is not None:
        return constants.get(node.id)
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _strategy_constants(strategy: dict[str, str]) -> dict[str, Any]:
    path = Path(str(strategy.get("strategy_file") or ""))
    if not path.exists():
        return {}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except Exception:
        return {}
    constants: dict[str, Any] = {}
    wanted = {"ENTRY_MODE", "ENTRY_TAG", "SIDE", "TIMEFRAME"}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        value = _literal_or_name(node.value, constants)
        if value is None:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                constants[target.id] = value
    class_name = str(strategy.get("strategy_class") or "")
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or (class_name and node.name != class_name):
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "timeframe" for target in stmt.targets):
                continue
            value = _literal_or_name(stmt.value, constants)
            if value is not None:
                constants["CLASS_TIMEFRAME"] = value
        if class_name:
            break
    return constants


def _strategy_timeframe(strategy: dict[str, str], constants: dict[str, Any]) -> str:
    for key in ("TIMEFRAME", "CLASS_TIMEFRAME"):
        value = str(constants.get(key) or "").strip().lower()
        if value in TIMEFRAME_SECONDS:
            return value
    path = Path(str(strategy.get("strategy_file") or ""))
    tokens = str(path.stem or strategy.get("name") or "").lower().replace("-", "_").split("_")
    for token in reversed(tokens):
        if token in TIMEFRAME_SECONDS:
            return token
    return "1h"


def _strategy_profile(strategy: dict[str, str]) -> dict[str, Any]:
    constants = _strategy_constants(strategy)
    timeframe = _strategy_timeframe(strategy, constants)
    path = Path(str(strategy.get("strategy_file") or ""))
    parts = [
        strategy.get("name"),
        path.stem,
        strategy.get("strategy_class"),
        strategy.get("core_behavior"),
        constants.get("ENTRY_MODE"),
        constants.get("ENTRY_TAG"),
    ]
    text = "_".join(str(part or "") for part in parts).lower().replace("-", "_").replace(" ", "_")
    pattern_types: list[str] = []
    family = "generic"
    patterns = [
        ("inverse_head_shoulders", "reversal"),
        ("head_shoulders", "reversal"),
        ("double_bottom", "reversal"),
        ("double_top", "reversal"),
        ("triple_bottom", "multi_peak"),
        ("triple_top", "multi_peak"),
        ("wolfe", "wolfe"),
        ("pennant", "continuation"),
        ("flag", "continuation"),
        ("ascending_channel", "geometry"),
        ("descending_channel", "geometry"),
        ("rectangle", "geometry"),
        ("triangle", "geometry"),
        ("wedge", "geometry"),
        ("compression", "geometry"),
        ("channel", "geometry"),
    ]
    for pattern, pattern_family in patterns:
        if pattern in text:
            pattern_types.append(pattern)
            family = pattern_family
    if not pattern_types:
        if "continuation" in text:
            family = "continuation"
        elif "reversal" in text or "choch" in text or "capitulation" in text:
            family = "reversal"
        elif "geometry" in text:
            family = "geometry"
        elif "vp_" in text or "volume_profile" in text:
            family = "volume_profile"
        elif "relative_strength" in text:
            family = "relative_strength"
        elif "tlv2" in text or "trendline" in text:
            family = "trendline"
        elif "prior_" in text or "equal_high" in text or "equal_low" in text:
            family = "prior_levels"
        elif "avwap" in text:
            family = "avwap"
    return {
        "timeframe": timeframe,
        "family": family,
        "pattern_types": pattern_types,
        "text": text,
    }


def _load_auto_windows(path: str | Path) -> list[dict[str, Any]]:
    auto_path = Path(path)
    if not auto_path.exists():
        raise ValueError(f"Entry Sieve auto window manifest not found: {auto_path}")
    payload = load_json(auto_path, {})
    raw_windows: list[Any] = []
    if isinstance(payload, dict):
        for key in ("auto_windows", "market_windows", "windows"):
            value = payload.get(key)
            if isinstance(value, list):
                raw_windows.extend(value)
    elif isinstance(payload, list):
        raw_windows = payload
    windows = [normalize_window(window) for window in raw_windows if isinstance(window, dict)]
    if not windows:
        raise ValueError(f"Entry Sieve auto window manifest has no windows: {auto_path}")
    return windows


def _parse_timerange(window: dict[str, Any]) -> tuple[datetime, datetime] | None:
    text = str(window.get("timerange") or "").strip()
    if "-" not in text:
        return None
    start_text, end_text = text.split("-", 1)
    try:
        start = datetime.strptime(start_text[:8], "%Y%m%d")
        end = datetime.strptime(end_text[:8], "%Y%m%d")
    except ValueError:
        return None
    if end <= start:
        return None
    return start, end


def _approx_candles(window: dict[str, Any], timeframe: str) -> int:
    parsed = _parse_timerange(window)
    seconds = TIMEFRAME_SECONDS.get(str(timeframe).lower())
    if parsed is None or not seconds:
        return 0
    start, end = parsed
    return max(0, int((end - start).total_seconds() // seconds))


def _window_timeframe_match(window: dict[str, Any], timeframe: str) -> bool:
    timeframes = _listish(window.get("timeframes") or window.get("timeframe"))
    return not timeframes or "all" in timeframes or str(timeframe).lower() in timeframes


def _auto_window_count(job: dict[str, Any]) -> int:
    try:
        count = int(str(job.get("auto_window_count") or "2").strip())
    except (TypeError, ValueError):
        count = 2
    return min(3, max(1, count))


def _window_score(window: dict[str, Any], profile: dict[str, Any]) -> float:
    timeframe = str(profile.get("timeframe") or "1h").lower()
    if not _window_timeframe_match(window, timeframe):
        return -1_000_000.0
    candles = _approx_candles(window, timeframe)
    minimum = int(MIN_TRAINING_CANDLES.get(timeframe, 500))
    if candles and candles < minimum:
        return -1_000_000.0

    score = 0.0
    try:
        score += float(window.get("priority") or 0.0)
    except (TypeError, ValueError):
        pass
    category = str(window.get("category") or window.get("segment_type") or "").lower()
    window_families = set(_listish(window.get("pattern_families") or window.get("pattern_family") or window.get("families")))
    window_types = set(_listish(window.get("pattern_types") or window.get("pattern_type") or window.get("patterns")))
    profile_family = str(profile.get("family") or "generic").lower()
    profile_types = {str(item).lower() for item in profile.get("pattern_types") or []}
    family_match = bool(profile_family and profile_family in window_families)
    type_match = bool(profile_types and profile_types.intersection(window_types))

    if _listish(window.get("timeframes") or window.get("timeframe")):
        score += 200.0
    if type_match:
        score += 1000.0
    if family_match:
        score += 500.0
    if profile_family in {"continuation", "geometry", "reversal", "multi_peak", "wolfe"} and category.startswith("pattern") and (family_match or type_match):
        score += 120.0
    if profile_family in {"continuation", "geometry", "reversal", "multi_peak", "wolfe"} and category.startswith("pattern") and not (family_match or type_match):
        score -= 250.0
    if profile_family not in {"continuation", "geometry", "reversal", "multi_peak", "wolfe"} and category in {"generic", "regime", "market_state", "full_cycle"}:
        score += 250.0
    if "generic" in window_families or category == "generic":
        score += 80.0

    target = int(TARGET_TRAINING_CANDLES.get(timeframe, minimum * 2))
    if candles:
        score += min(150.0, (candles / max(minimum, 1)) * 25.0)
        if candles > target:
            score -= min(160.0, ((candles - target) / max(target, 1)) * 50.0)
    return score


def _select_auto_windows(
    strategy: dict[str, str],
    auto_windows: list[dict[str, Any]],
    market_windows: list[dict[str, Any]],
    job: dict[str, Any],
) -> list[dict[str, Any]]:
    profile = _strategy_profile(strategy)
    count = _auto_window_count(job)
    candidates = [*auto_windows, *market_windows]
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, window in enumerate(candidates):
        normalized = normalize_window(window)
        score = _window_score(normalized, profile)
        if score <= -999_999:
            continue
        scored.append((score, index, normalized))
    scored.sort(key=lambda item: (-item[0], item[1]))

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, _, window in scored:
        key = str(window.get("name") or window.get("timerange") or "")
        if not key or key in seen:
            continue
        selected.append(window)
        seen.add(key)
        if len(selected) >= count:
            return selected

    strategy_name = strategy.get("name") or strategy.get("strategy_class") or strategy.get("strategy_file") or "unknown"
    timeframe = str(profile.get("timeframe") or "1h")
    raise ValueError(f"Entry Sieve could not assign {count} auto training window(s) for {strategy_name} on {timeframe}.")


def _manual_training_plan(strategies: list[dict[str, str]], training_windows: list[dict[str, Any]]) -> list[tuple[dict[str, str], dict[str, Any]]]:
    return [(strategy, training_window) for strategy in strategies for training_window in training_windows]


def _auto_training_plan(
    strategies: list[dict[str, str]],
    market_windows: list[dict[str, Any]],
    job: dict[str, Any],
) -> list[tuple[dict[str, str], dict[str, Any]]]:
    auto_windows = _load_auto_windows(job.get("auto_windows_file") or "")
    plan: list[tuple[dict[str, str], dict[str, Any]]] = []
    for strategy in strategies:
        for training_window in _select_auto_windows(strategy, auto_windows, market_windows, job):
            plan.append((strategy, training_window))
    return plan


def _build_training_plan(
    strategies: list[dict[str, str]],
    market_windows: list[dict[str, Any]],
    job: dict[str, Any],
) -> tuple[list[tuple[dict[str, str], dict[str, Any]]], int]:
    if bool(job.get("auto_window_mode")):
        count = _auto_window_count(job)
        return _auto_training_plan(strategies, market_windows, job), count
    training_windows = resolve_windows(market_windows, job.get("training_windows") or [], label="training")
    return _manual_training_plan(strategies, training_windows), len(training_windows)


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


def _build_preset(base_preset: dict[str, Any], strategy_file: Path, strategy_class: str, runtime_dir: Path, timeframe: str = "") -> dict[str, Any]:
    preset = deepcopy(base_preset)
    preset["strategy_file"] = str(strategy_file)
    preset["strategy_class"] = strategy_class
    if timeframe:
        preset["timeframe"] = str(timeframe)
    preset["hyperopt_spaces"] = "buy"
    preset["backtest_export"] = "trades"
    preset["backtest_directory"] = str(runtime_dir / "backtests")
    return preset


def _job_result_metadata(job: dict[str, Any]) -> dict[str, Any]:
    batch_id = str(job.get("strategy_batch") or "all").strip() or "all"
    return {
        "strategy_batch": batch_id,
        "strategy_batch_label": str(job.get("strategy_batch_label") or batch_id),
        "strategy_filter": str(job.get("strategy_filter") or ""),
        "speed_run_mode": bool(job.get("speed_run_mode")),
        "speed_pair_count": str(job.get("speed_pair_count") or ""),
        "auto_window_mode": bool(job.get("auto_window_mode")),
        "auto_window_count": str(job.get("auto_window_count") or ""),
        "target_sweep_enabled": bool(job.get("target_sweep_enabled")),
    }


def _annotate_strategies_with_job_metadata(strategies: list[dict[str, str]], job: dict[str, Any]) -> list[dict[str, str]]:
    metadata = _job_result_metadata(job)
    annotated: list[dict[str, str]] = []
    for strategy in strategies:
        row = dict(strategy)
        row.update(metadata)
        annotated.append(row)
    return annotated


def _pair_tokens(value: Any) -> list[str]:
    text = str(value or "").replace(";", "\n").replace(",", "\n")
    return [token.strip() for token in text.split() if token.strip()]


def _speed_pair_count(job: dict[str, Any]) -> int:
    try:
        count = int(str(job.get("speed_pair_count") or "5").strip())
    except (TypeError, ValueError):
        count = 5
    return max(1, count)


def _speed_limited_preset(base_preset: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    preset = deepcopy(base_preset)
    if not bool(job.get("speed_run_mode")):
        return preset
    if str(preset.get("pair_mode") or "").strip().lower() != "manual":
        return preset
    pairs = _pair_tokens(preset.get("pairs"))
    if not pairs:
        return preset
    preset["pairs"] = "\n".join(pairs[: _speed_pair_count(job)])
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
    lane_index: int | None = None


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
    user_data_dir = original_strategy_dir.parent
    extra_paths = [str(original_strategy_dir), str(user_data_dir), str(user_data_dir / "Indicators")]
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
    summary = metric_summary(metrics or {}) if metrics else {}
    objective = None
    if summary:
        try:
            objective = score_objective(summary)
        except Exception:
            objective = None
    return {
        "job_id": job_id,
        "finished_at": datetime.now().astimezone().isoformat(),
        "strategy": strategy.get("name") or Path(strategy.get("strategy_file", "")).stem,
        "strategy_class": strategy.get("strategy_class", ""),
        "strategy_file": strategy.get("strategy_file", ""),
        "strategy_batch": str(strategy.get("strategy_batch") or "all"),
        "strategy_batch_label": str(strategy.get("strategy_batch_label") or strategy.get("strategy_batch") or "all"),
        "strategy_filter": str(strategy.get("strategy_filter") or ""),
        "speed_run_mode": bool(strategy.get("speed_run_mode")),
        "speed_pair_count": str(strategy.get("speed_pair_count") or ""),
        "auto_window_mode": bool(strategy.get("auto_window_mode")),
        "auto_window_count": str(strategy.get("auto_window_count") or ""),
        "target_sweep_enabled": bool(strategy.get("target_sweep_enabled")),
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
        "profit_total_abs": _metric(summary, "profit_total_abs"),
        "profit_total": _metric(summary, "profit_total", "profit_total_pct"),
        "profit_total_pct": _metric(summary, "profit_total_pct"),
        "trade_count": _metric(summary, "total_trades", "trade_count"),
        "winrate": _metric(summary, "winrate"),
        "wins": _metric(summary, "wins"),
        "draws": _metric(summary, "draws"),
        "losses": _metric(summary, "losses"),
        "profit_factor": _metric(summary, "profit_factor"),
        "max_drawdown_pct": _metric(summary, "max_relative_drawdown", "max_drawdown_account"),
        "final_balance": _metric(summary, "final_balance"),
        "market_change": _metric(summary, "market_change"),
        "hyperopt_file": str(hyperopt_file) if hyperopt_file is not None else "",
        "backtest_file": str(backtest_file) if backtest_file is not None else "",
        "params_file": str(params_file) if params_file is not None else "",
        "error": error,
    }


def _result_file(runtime_dir: Path, job_id: str) -> Path:
    return runtime_dir / "results" / f"{_safe_name(job_id)}.jsonl"


def _result_summary_file(runtime_dir: Path, job_id: str) -> Path:
    return runtime_dir / "results" / f"{_safe_name(job_id)}.summary.json"


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _load_result_summary(runtime_dir: Path, job_id: str) -> dict[str, Any]:
    summary_file = _result_summary_file(runtime_dir, job_id)
    payload = load_json(summary_file, {}) if summary_file.exists() else {}
    return payload if isinstance(payload, dict) else {}


def _write_result_summary(
    runtime_dir: Path,
    job_id: str,
    *,
    status: str,
    phase: str,
    updated_at: str,
    row_count: int | None = None,
    created_at: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    results_file = _result_file(runtime_dir, job_id)
    summary_file = _result_summary_file(runtime_dir, job_id)
    payload = _load_result_summary(runtime_dir, job_id)
    if row_count is None:
        try:
            row_count = int(payload.get("row_count") or _count_jsonl_rows(results_file))
        except (TypeError, ValueError):
            row_count = _count_jsonl_rows(results_file)
    summary = {
        "schema_version": 3,
        "storage": "jsonl",
        "job_id": job_id,
        "created_at": str(payload.get("created_at") or created_at or updated_at),
        "updated_at": updated_at,
        "status": status,
        "phase": phase,
        "row_count": int(row_count),
        "path": str(results_file),
        "summary_path": str(summary_file),
    }
    for key in RESULT_METADATA_KEYS:
        if key in payload:
            summary[key] = payload.get(key)
    if metadata:
        for key in RESULT_METADATA_KEYS:
            if key in metadata:
                summary[key] = metadata.get(key)
    save_json(summary_file, summary)
    save_json(
        runtime_dir / "latest.json",
        {
            "job_id": job_id,
            "path": str(results_file),
            "summary_path": str(summary_file),
            "updated_at": updated_at,
            "status": status,
            "phase": phase,
            **{key: summary[key] for key in RESULT_METADATA_KEYS if key in summary},
        },
    )


def _append_result(runtime_dir: Path, row: dict[str, Any]) -> None:
    job_id = str(row.get("job_id") or "unknown")
    results_file = _result_file(runtime_dir, job_id)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row.pop("metrics", None)
    with results_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=False, default=str, separators=(",", ":")) + "\n")
    payload = _load_result_summary(runtime_dir, job_id)
    try:
        row_count = int(payload.get("row_count") or 0) + 1
    except (TypeError, ValueError):
        row_count = _count_jsonl_rows(results_file)
    updated_at = datetime.now().astimezone().isoformat()
    _write_result_summary(
        runtime_dir,
        job_id,
        status=str(payload.get("status") or "running"),
        phase=str(payload.get("phase") or "backtest"),
        updated_at=updated_at,
        row_count=row_count,
        metadata={key: row.get(key) for key in RESULT_METADATA_KEYS if key in row},
    )


def _initialize_result_batch(runtime_dir: Path, job: dict[str, Any]) -> None:
    _result_file(runtime_dir, str(job.get("job_id") or "unknown")).parent.mkdir(parents=True, exist_ok=True)


def _update_result_batch_status(runtime_dir: Path, job_id: str, status_payload: dict[str, Any]) -> None:
    results_file = _result_file(runtime_dir, job_id)
    summary_file = _result_summary_file(runtime_dir, job_id)
    if not results_file.exists() and not summary_file.exists():
        return
    if results_file.exists() and _count_jsonl_rows(results_file) == 0:
        return
    updated_at = str(status_payload.get("updated_at") or datetime.now().astimezone().isoformat())
    _write_result_summary(
        runtime_dir,
        job_id,
        status=str(status_payload.get("status") or "running"),
        phase=str(status_payload.get("phase") or ""),
        updated_at=updated_at,
        metadata={key: status_payload.get(key) for key in RESULT_METADATA_KEYS if key in status_payload},
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
    if ACTIVE_ENTRY_SIEVE_LOCK is not None:
        ACTIVE_ENTRY_SIEVE_LOCK.heartbeat(
            status=payload.get("status"),
            phase=payload.get("phase"),
            run_index=payload.get("run_index"),
            current_strategy=payload.get("current_strategy"),
            message=payload.get("message"),
        )


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


def _run_backtest_batch_on_lane(batch: PendingBacktests, python_exe: str, lane_index: int) -> list[dict[str, Any]]:
    print(f"Entry Sieve backtest lane {lane_index}: {batch.strategy.get('name') or batch.strategy.get('strategy_class')} | {python_exe}")
    return _run_backtest_batch(batch, python_exe)


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
    run_index: int | None = None,
    total_hyperopts: int | None = None,
    completed_hyperopts: int | None = None,
    current_strategy: str = "",
    current_training_window: str = "",
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


def _finish_one_pending(
    runtime_dir: Path,
    pending_batches: list[PendingBacktests],
    *,
    job_id: str,
    completed_backtests: int,
    total_backtests: int,
    run_index: int | None = None,
    total_hyperopts: int | None = None,
    completed_hyperopts: int | None = None,
    current_strategy: str = "",
    current_training_window: str = "",
) -> int:
    if not pending_batches:
        return 0
    completed = [batch for batch in pending_batches if batch.future is not None and batch.future.done()]
    if not completed:
        futures = [batch.future for batch in pending_batches if batch.future is not None]
        while futures:
            _write_run_status(
                runtime_dir,
                job_id,
                status="running",
                phase="backtest_wait",
                run_index=run_index,
                total_hyperopts=total_hyperopts,
                completed_hyperopts=completed_hyperopts,
                completed_backtests=completed_backtests,
                total_backtests=total_backtests,
                current_strategy=current_strategy,
                current_training_window=current_training_window,
                queued_backtest_batches=len(pending_batches),
                running_backtest_batches=len(pending_batches),
                message=f"Waiting for backtest lane ({len(pending_batches)} running, {completed_backtests}/{total_backtests} complete)",
            )
            done, _ = wait(futures, timeout=BACKTEST_WAIT_STATUS_SECONDS, return_when=FIRST_COMPLETED)
            if not done:
                continue
            completed = [batch for batch in pending_batches if batch.future in done]
            break
    batch = completed[0] if completed else pending_batches[0]
    pending_batches.remove(batch)
    return _finish_pending(
        runtime_dir,
        batch,
        job_id=job_id,
        completed_backtests=completed_backtests,
        total_backtests=total_backtests,
        run_index=run_index,
        total_hyperopts=total_hyperopts,
        completed_hyperopts=completed_hyperopts,
        current_strategy=current_strategy,
        current_training_window=current_training_window,
    )


def _finish_ready_backtests(
    runtime_dir: Path,
    running_batches: list[PendingBacktests],
    *,
    job_id: str,
    completed_backtests: int,
    total_backtests: int,
    run_index: int | None = None,
    total_hyperopts: int | None = None,
    completed_hyperopts: int | None = None,
    current_strategy: str = "",
    current_training_window: str = "",
) -> int:
    written = 0
    ready = [batch for batch in running_batches if batch.future is not None and batch.future.done()]
    for batch in ready:
        running_batches.remove(batch)
        written += _finish_pending(
            runtime_dir,
            batch,
            job_id=job_id,
            completed_backtests=completed_backtests + written,
            total_backtests=total_backtests,
            run_index=run_index,
            total_hyperopts=total_hyperopts,
            completed_hyperopts=completed_hyperopts,
            current_strategy=current_strategy,
            current_training_window=current_training_window,
        )
    return written


def _system_memory_percent() -> float | None:
    try:
        import psutil

        return float(psutil.virtual_memory().percent)
    except ImportError:
        pass
    except Exception:
        return None
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        memory_status = MEMORYSTATUSEX()
        memory_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status)):
            return float(memory_status.dwMemoryLoad)
    except Exception:
        return None
    return None


def _backtest_start_block_reason(last_start_at: float | None) -> str:
    memory_percent = _system_memory_percent()
    if memory_percent is not None and memory_percent >= BACKTEST_START_RAM_LIMIT_PERCENT:
        return f"RAM {memory_percent:.1f}% >= {BACKTEST_START_RAM_LIMIT_PERCENT:.0f}%"
    if last_start_at:
        remaining_seconds = BACKTEST_START_STAGGER_SECONDS - (time.monotonic() - last_start_at)
        if remaining_seconds > 0:
            return f"Waiting {remaining_seconds:.0f}s before starting next backtest"
    return ""


def _next_available_lane_index(
    running_batches: list[PendingBacktests],
    *,
    lane_count: int,
    lane_cursor: int,
) -> int | None:
    occupied = {batch.lane_index for batch in running_batches if batch.lane_index is not None}
    for offset in range(lane_count):
        lane_index = (lane_cursor + offset) % lane_count
        if lane_index not in occupied:
            return lane_index
    return None


def _write_backtest_queue_status(
    runtime_dir: Path,
    *,
    job_id: str,
    phase: str,
    run_index: int | None,
    total_hyperopts: int | None,
    completed_hyperopts: int | None,
    completed_backtests: int,
    total_backtests: int,
    current_strategy: str,
    current_training_window: str,
    waiting_batches: list[PendingBacktests],
    running_batches: list[PendingBacktests],
    lane_limit: int,
    message: str,
) -> None:
    _write_run_status(
        runtime_dir,
        job_id,
        status="running",
        phase=phase,
        run_index=run_index,
        total_hyperopts=total_hyperopts,
        completed_hyperopts=completed_hyperopts,
        completed_backtests=completed_backtests,
        total_backtests=total_backtests,
        current_strategy=current_strategy,
        current_training_window=current_training_window,
        queued_backtest_batches=len(waiting_batches) + len(running_batches),
        waiting_backtest_batches=len(waiting_batches),
        running_backtest_batches=len(running_batches),
        backtest_worker_limit=lane_limit,
        message=message,
    )


def _start_available_backtests(
    runtime_dir: Path,
    waiting_batches: list[PendingBacktests],
    running_batches: list[PendingBacktests],
    *,
    executor: ThreadPoolExecutor,
    backtest_lanes: list[str],
    lane_limit: int,
    lane_cursor: int,
    last_start_at: float | None,
    job_id: str,
    completed_backtests: int,
    total_backtests: int,
    run_index: int | None,
    total_hyperopts: int | None,
    completed_hyperopts: int | None,
    current_strategy: str,
    current_training_window: str,
    status_phase: str,
) -> tuple[int, float | None, int, str]:
    if not waiting_batches:
        return lane_cursor, last_start_at, 0, ""
    lane_count = len(backtest_lanes)
    if lane_count <= 0:
        return lane_cursor, last_start_at, 0, "No backtest lanes configured"
    active_limit = max(1, min(lane_limit, lane_count))
    started = 0
    block_reason = ""
    while waiting_batches and len(running_batches) < active_limit:
        lane_index = _next_available_lane_index(running_batches, lane_count=lane_count, lane_cursor=lane_cursor)
        if lane_index is None:
            block_reason = "All backtest lanes are busy"
            break
        block_reason = _backtest_start_block_reason(last_start_at)
        if block_reason:
            break
        batch = waiting_batches.pop(0)
        batch.lane_index = lane_index
        batch.future = executor.submit(
            _run_backtest_batch_on_lane,
            batch,
            backtest_lanes[lane_index],
            lane_index + 1,
        )
        running_batches.append(batch)
        lane_cursor = (lane_index + 1) % lane_count
        last_start_at = time.monotonic()
        started += 1
    if block_reason:
        _write_backtest_queue_status(
            runtime_dir,
            job_id=job_id,
            phase=status_phase,
            run_index=run_index,
            total_hyperopts=total_hyperopts,
            completed_hyperopts=completed_hyperopts,
            completed_backtests=completed_backtests,
            total_backtests=total_backtests,
            current_strategy=current_strategy,
            current_training_window=current_training_window,
            waiting_batches=waiting_batches,
            running_batches=running_batches,
            lane_limit=active_limit,
            message=block_reason,
        )
    return lane_cursor, last_start_at, started, block_reason


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
    strategy_timeframe = str(_strategy_profile(strategy).get("timeframe") or "")
    preset = _build_preset(base_preset, runtime_strategy_file, strategy_class, runtime_dir, strategy_timeframe)
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
    else:
        try:
            value = int(str(job.get("epochs") or "200").strip())
        except (TypeError, ValueError):
            value = 200
    if bool(job.get("speed_run_mode")):
        cap_text = str(job.get("auto_epochs_cap") or SPEED_RUN_EPOCH_CAP).strip()
        try:
            cap = int(cap_text)
        except (TypeError, ValueError):
            cap = SPEED_RUN_EPOCH_CAP
        if cap > 0:
            value = min(value, cap)
    return str(max(1, value))


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


def _split_python_exes(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").replace(";", ",").replace("\n", ",")
    return [item.strip() for item in text.split(",") if item.strip()]


def _dedupe_python_exes(paths: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for raw_path in paths:
        text = str(raw_path or "").strip()
        if not text:
            continue
        try:
            key = str(Path(text).expanduser().resolve()).lower()
        except OSError:
            key = str(Path(text).expanduser()).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def _worker_count(value: Any, *, default: int) -> int:
    try:
        count = int(str(value).strip())
    except (TypeError, ValueError):
        count = default
    return min(MAX_BACKTEST_WORKERS, max(1, count))


def _hyperopt_job_count(preset: dict[str, Any]) -> int:
    try:
        count = int(str(preset.get("hyperopt_jobs") or "").strip())
    except (TypeError, ValueError):
        count = 1
    if count < 1:
        cpu_total = max(1, int(os.cpu_count() or 1))
        count = max(1, cpu_total + 1 + count)
    return max(1, count)


def _final_backtest_worker_limit(
    *,
    backtest_lane_count: int,
    base_preset: dict[str, Any],
    split_venv_pipeline: bool,
) -> int:
    if not split_venv_pipeline:
        return 1
    return _active_worker_capacity(backtest_lane_count)


def _active_worker_capacity(backtest_lane_count: int) -> int:
    cpu_total = max(1, int(os.cpu_count() or 1))
    return max(1, min(backtest_lane_count, cpu_total - RESERVED_SYSTEM_WORKERS))


def _pipelined_backtest_worker_limit(
    *,
    backtest_lane_count: int,
    base_preset: dict[str, Any],
    split_venv_pipeline: bool,
) -> int:
    if not split_venv_pipeline:
        return 1
    active_capacity = _active_worker_capacity(backtest_lane_count)
    hyperopt_workers = _hyperopt_job_count(base_preset)
    return max(1, min(backtest_lane_count, active_capacity - hyperopt_workers))


def _effective_backtest_worker_count(
    requested_count: int,
    *,
    split_venv_pipeline: bool,
) -> int:
    if not split_venv_pipeline:
        return 1
    return requested_count


def _backtest_lanes(
    *,
    base_python_exe: str,
    backtest_python_exe: str,
    backtest_python_exes: list[str],
    split_venv_pipeline: bool,
    backtest_worker_count: int,
) -> list[str]:
    if split_venv_pipeline:
        candidates = _dedupe_python_exes([
            str(backtest_python_exe or ""),
            *backtest_python_exes,
        ])
    else:
        candidates = _dedupe_python_exes([str(backtest_python_exe or base_python_exe or sys.executable)])
    return candidates[: max(1, min(backtest_worker_count, len(candidates)))]


def _verify_python_lanes(lanes: list[str]) -> None:
    if not lanes:
        raise SystemExit("Entry Sieve split-venv mode has no configured backtest worker Python executables.")
    missing = [str(Path(lane).expanduser()) for lane in lanes if not Path(lane).expanduser().exists()]
    if missing:
        raise SystemExit("Entry Sieve configured Python executable does not exist: " + ", ".join(missing))


def _run_backtest_lane(
    *,
    lane_index: int,
    python_exe: str,
    work_items: list[tuple[PendingBacktests, BacktestTask]],
    result_queue: Queue[dict[str, Any]],
) -> int:
    completed = 0
    total = len(work_items)
    if lane_index > 1:
        time.sleep((lane_index - 1) * BACKTEST_LANE_START_STAGGER_SECONDS)
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
    auto_window_mode = bool(job.get("auto_window_mode"))
    validation_selection = [job.get("auto_validation_window") or FULL_CYCLE_VALIDATION_WINDOW] if auto_window_mode else job.get("validation_windows") or []
    validation_windows = resolve_windows(all_windows, validation_selection, label="validation")
    strategies = _annotate_strategies_with_job_metadata([strategy for strategy in job.get("strategies") or [] if isinstance(strategy, dict)], job)
    training_plan, configured_training_window_count = _build_training_plan(strategies, all_windows, job)
    base_preset = _speed_limited_preset(presets[preset_name], job)
    hyperopt_jobs = str(job.get("hyperopt_jobs") or "").strip()
    if hyperopt_jobs:
        base_preset["hyperopt_jobs"] = hyperopt_jobs
    job_id = str(job.get("job_id") or job_file.stem)
    global ACTIVE_ENTRY_SIEVE_LOCK
    ACTIVE_ENTRY_SIEVE_LOCK = EntrySieveRunLock(runtime_dir, job_id=job_id, owner="entry_sieve_runner")
    ACTIVE_ENTRY_SIEVE_LOCK.__enter__()
    atexit.register(ACTIVE_ENTRY_SIEVE_LOCK.__exit__, None, None, None)
    _initialize_result_batch(runtime_dir, job)
    split_venv_pipeline = bool(job.get("split_venv_pipeline"))
    target_sweep_enabled = bool(job.get("target_sweep_enabled"))
    base_python_exe = str(base_preset.get("python_exe") or job.get("python_exe") or sys.executable)
    if split_venv_pipeline:
        backtest_python_exe = str(job.get("backtest_python_exe") or "")
    else:
        backtest_python_exe = str(job.get("backtest_python_exe") or base_preset.get("python_exe") or sys.executable)
    backtest_python_exes = _split_python_exes(job.get("backtest_python_exes"))
    requested_backtest_worker_count = _worker_count(job.get("backtest_worker_count"), default=2 if split_venv_pipeline else 1)
    backtest_worker_count = _effective_backtest_worker_count(
        requested_backtest_worker_count,
        split_venv_pipeline=split_venv_pipeline,
    )
    backtest_lanes = _backtest_lanes(
        base_python_exe=base_python_exe,
        backtest_python_exe=backtest_python_exe,
        backtest_python_exes=backtest_python_exes,
        split_venv_pipeline=split_venv_pipeline,
        backtest_worker_count=backtest_worker_count,
    )
    _verify_python_lanes(backtest_lanes)
    state_file = runtime_dir / "state.json"
    state = explorer_load_json(state_file, {})
    if not isinstance(state, dict):
        state = {}

    total_runs = len(training_plan)
    target_pair_count = len(_target_pairs(job))
    total_backtests = total_runs * len(validation_windows) * target_pair_count
    print(f"Entry Sieve job: {job_id}")
    if auto_window_mode:
        print(f"Strategies: {len(strategies)} | auto windows/file: {configured_training_window_count} | validation windows: {len(validation_windows)} | runs: {total_runs}")
    else:
        print(f"Strategies: {len(strategies)} | training windows: {configured_training_window_count} | validation windows: {len(validation_windows)} | runs: {total_runs}")
    if split_venv_pipeline:
        print(f"Split-venv backtest lanes ({len(backtest_lanes)}/{requested_backtest_worker_count} requested, {backtest_worker_count} effective):")
        for lane_index, lane in enumerate(backtest_lanes, start=1):
            print(f"  lane {lane_index}: {lane}")
        if not target_sweep_enabled:
            print(
                "Non-sweep backtest scheduler: "
                f"{_pipelined_backtest_worker_limit(backtest_lane_count=len(backtest_lanes), base_preset=base_preset, split_venv_pipeline=split_venv_pipeline)} lane(s) during hyperopt, "
                f"{_final_backtest_worker_limit(backtest_lane_count=len(backtest_lanes), base_preset=base_preset, split_venv_pipeline=split_venv_pipeline)} lane(s) during final drain"
            )
    if bool(job.get("speed_run_mode")):
        print(f"Speed run mode enabled: first {_speed_pair_count(job)} manual pair(s), 1 auto window, epochs capped at {job.get('auto_epochs_cap') or SPEED_RUN_EPOCH_CAP}, target sweep disabled")
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
        training_window_count=configured_training_window_count,
        validation_window_count=len(validation_windows),
        target_pair_count=target_pair_count,
        auto_window_mode=auto_window_mode,
        message="Entry Sieve run started",
    )

    run_index = 0
    pending_batches: list[PendingBacktests] = []
    final_completed_backtests = 0
    if target_sweep_enabled:
        for strategy, training_window in training_plan:
            strategy_name = strategy.get("name") or Path(str(strategy.get("strategy_file") or "")).stem
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
            lanes=backtest_lanes,
            total_backtests=total_backtests,
        )
    else:
        waiting_batches: list[PendingBacktests] = []
        running_batches: list[PendingBacktests] = []
        completed_backtests = 0
        lane_cursor = 0
        last_backtest_start_at: float | None = None
        hyperopt_lane_limit = _pipelined_backtest_worker_limit(
            backtest_lane_count=len(backtest_lanes),
            base_preset=base_preset,
            split_venv_pipeline=split_venv_pipeline,
        )
        drain_lane_limit = _final_backtest_worker_limit(
            backtest_lane_count=len(backtest_lanes),
            base_preset=base_preset,
            split_venv_pipeline=split_venv_pipeline,
        )
        with ThreadPoolExecutor(max_workers=max(1, len(backtest_lanes)), thread_name_prefix="entry-sieve-backtest") as executor:
            for strategy, training_window in training_plan:
                strategy_name = strategy.get("name") or Path(str(strategy.get("strategy_file") or "")).stem
                run_index += 1
                print(f"\nEntry Sieve {run_index}/{total_runs}: {strategy_name} | train={window_label(training_window)}")
                if split_venv_pipeline:
                    completed_backtests += _finish_ready_backtests(
                        runtime_dir,
                        running_batches,
                        job_id=job_id,
                        completed_backtests=completed_backtests,
                        total_backtests=total_backtests,
                        run_index=run_index,
                        total_hyperopts=total_runs,
                        completed_hyperopts=run_index - 1,
                        current_strategy=strategy_name,
                        current_training_window=window_label(training_window),
                    )
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
                        waiting_batches.append(batch)
                        completed_backtests += _finish_ready_backtests(
                            runtime_dir,
                            running_batches,
                            job_id=job_id,
                            completed_backtests=completed_backtests,
                            total_backtests=total_backtests,
                            run_index=run_index,
                            total_hyperopts=total_runs,
                            completed_hyperopts=run_index,
                            current_strategy=strategy_name,
                            current_training_window=window_label(training_window),
                        )
                        lane_cursor, last_backtest_start_at, _, _ = _start_available_backtests(
                            runtime_dir,
                            waiting_batches,
                            running_batches,
                            executor=executor,
                            backtest_lanes=backtest_lanes,
                            lane_limit=hyperopt_lane_limit,
                            lane_cursor=lane_cursor,
                            last_start_at=last_backtest_start_at,
                            job_id=job_id,
                            completed_backtests=completed_backtests,
                            total_backtests=total_backtests,
                            run_index=run_index,
                            total_hyperopts=total_runs,
                            completed_hyperopts=run_index,
                            current_strategy=strategy_name,
                            current_training_window=window_label(training_window),
                            status_phase="backtest_wait",
                        )
                    else:
                        rows = _run_backtest_batch(batch, backtest_lanes[0])
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
                        queued_backtest_batches=len(waiting_batches) + len(running_batches) if split_venv_pipeline else 0,
                        waiting_backtest_batches=len(waiting_batches) if split_venv_pipeline else 0,
                        running_backtest_batches=len(running_batches) if split_venv_pipeline else 0,
                        backtest_worker_limit=hyperopt_lane_limit if split_venv_pipeline else 1,
                        message=f"Run queued {run_index}/{total_runs}: {strategy_name}" if split_venv_pipeline else f"Run complete {run_index}/{total_runs}: {strategy_name}",
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
            while waiting_batches or running_batches:
                completed_backtests += _finish_ready_backtests(
                    runtime_dir,
                    running_batches,
                    job_id=job_id,
                    completed_backtests=completed_backtests,
                    total_backtests=total_backtests,
                    run_index=run_index,
                    total_hyperopts=total_runs,
                    completed_hyperopts=total_runs,
                    current_strategy="final_backtest_drain",
                    current_training_window="",
                )
                lane_cursor, last_backtest_start_at, started, block_reason = _start_available_backtests(
                    runtime_dir,
                    waiting_batches,
                    running_batches,
                    executor=executor,
                    backtest_lanes=backtest_lanes,
                    lane_limit=drain_lane_limit,
                    lane_cursor=lane_cursor,
                    last_start_at=last_backtest_start_at,
                    job_id=job_id,
                    completed_backtests=completed_backtests,
                    total_backtests=total_backtests,
                    run_index=run_index,
                    total_hyperopts=total_runs,
                    completed_hyperopts=total_runs,
                    current_strategy="final_backtest_drain",
                    current_training_window="",
                    status_phase="backtest_wait",
                )
                if started:
                    continue
                if running_batches and waiting_batches:
                    _write_backtest_queue_status(
                        runtime_dir,
                        job_id=job_id,
                        phase="backtest_wait",
                        run_index=run_index,
                        total_hyperopts=total_runs,
                        completed_hyperopts=total_runs,
                        completed_backtests=completed_backtests,
                        total_backtests=total_backtests,
                        current_strategy="final_backtest_drain",
                        current_training_window="",
                        waiting_batches=waiting_batches,
                        running_batches=running_batches,
                        lane_limit=drain_lane_limit,
                        message=block_reason or "Waiting to start more backtests",
                    )
                    time.sleep(5.0)
                    continue
                if running_batches:
                    completed_backtests += _finish_one_pending(
                        runtime_dir,
                        running_batches,
                        job_id=job_id,
                        completed_backtests=completed_backtests,
                        total_backtests=total_backtests,
                        run_index=run_index,
                        total_hyperopts=total_runs,
                        completed_hyperopts=total_runs,
                        current_strategy="final_backtest_drain",
                        current_training_window="",
                    )
                    continue
                if waiting_batches:
                    _write_backtest_queue_status(
                        runtime_dir,
                        job_id=job_id,
                        phase="backtest_wait",
                        run_index=run_index,
                        total_hyperopts=total_runs,
                        completed_hyperopts=total_runs,
                        completed_backtests=completed_backtests,
                        total_backtests=total_backtests,
                        current_strategy="final_backtest_drain",
                        current_training_window="",
                        waiting_batches=waiting_batches,
                        running_batches=running_batches,
                        lane_limit=drain_lane_limit,
                        message=block_reason or "Waiting to start backtests",
                    )
                    time.sleep(5.0)
            if split_venv_pipeline:
                completed_backtests += _finish_ready_backtests(
                    runtime_dir,
                    running_batches,
                    job_id=job_id,
                    completed_backtests=completed_backtests,
                    total_backtests=total_backtests,
                    run_index=run_index,
                    total_hyperopts=total_runs,
                    completed_hyperopts=total_runs,
                    current_strategy="final_backtest_drain",
                    current_training_window="",
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
    print(f"\nEntry Sieve results: {_result_file(runtime_dir, job_id)}")
    if ACTIVE_ENTRY_SIEVE_LOCK is not None:
        ACTIVE_ENTRY_SIEVE_LOCK.__exit__(None, None, None)
        ACTIVE_ENTRY_SIEVE_LOCK = None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

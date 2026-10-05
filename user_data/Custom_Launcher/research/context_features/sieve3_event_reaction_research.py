from __future__ import annotations

import argparse
import faulthandler
import gc
import importlib.util
import json
import math
import os
import re
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

_EARLY_STALL_TRACE_SECONDS = int(
    os.environ.get("SIEVE3_DIAGNOSTIC_STALL_TRACE_SECONDS", "0") or "0"
)
if _EARLY_STALL_TRACE_SECONDS > 0:
    faulthandler.enable()
    faulthandler.dump_traceback_later(
        _EARLY_STALL_TRACE_SECONDS,
        repeat=True,
    )

import numpy as np
import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.strategies.sieve3_event_reaction_targets import (  # noqa: E402
    _atr,
    _future_extreme,
    build_sieve3_reaction_targets,
)


USER_DATA_DIR = REPO_ROOT / "user_data"
DEFAULT_HANDOVER = (
    REPO_ROOT
    / "ai_guidance_docs"
    / "03_status"
    / "sieve3_v2_freqai_resolved_params_handover_20260807.md"
)
DEFAULT_CONFIG = (
    USER_DATA_DIR / "configs" / "config_sieve3_event_reaction_freqai.example.json"
)
DEFAULT_OUTPUT = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "sieve3_event_reaction"
    / "round1"
)
BASELINE_PAIRS = ("BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT")
PROJECT_TOP10_PAIRS = (
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "BNB/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "ADA/USDT:USDT",
    "DOGE/USDT:USDT",
    "TRX/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
)
DEFAULT_WINDOWS = {
    "fresh_to_round1": (
        pd.Timestamp("2023-01-01", tz="UTC"),
        pd.Timestamp("2024-04-01", tz="UTC"),
    ),
    "round1_development_exposed": (
        pd.Timestamp("2024-04-01", tz="UTC"),
        pd.Timestamp("2025-04-01", tz="UTC"),
    ),
    "round1_validation_exposed": (
        pd.Timestamp("2025-04-01", tz="UTC"),
        pd.Timestamp("2026-04-01", tz="UTC"),
    ),
    "round1_late_exposed": (
        pd.Timestamp("2026-04-01", tz="UTC"),
        pd.Timestamp("2026-06-29", tz="UTC"),
    ),
}
DEFAULT_DIRECT_HORIZONS = (1, 2, 4, 8, 12, 24, 48)
ARCHIVED_ENTRY_SIEVE_TOOLS = (
    USER_DATA_DIR
    / "strategies"
    / "Archive"
    / "non_sieve3_top_level_strategies_archived_20260612"
    / "entry_sieve_tools.py"
)


@dataclass
class ExitSolution:
    family: str
    selected_values: str
    trades: int
    win_pct: float
    profit_pct: float
    drawdown_pct: float
    profit_factor: float
    strategy_path: Path | None
    params_path: Path | None
    backtest_path: Path | None
    sparse: bool


@dataclass
class EntryCandidate:
    entry_id: str
    side: str = ""
    timeframe: str = ""
    source_path: Path | None = None
    source_class: str = ""
    locked_values: dict[str, Any] = field(default_factory=dict)
    exits: list[ExitSolution] = field(default_factory=list)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def pair_token(pair: str) -> str:
    base = pair.strip().upper().split("/")[0].split(":")[0]
    if not base or not base.replace("-", "").isalnum():
        raise ValueError(f"Unsupported pair: {pair!r}")
    return base.lower()


def research_log_path(output_dir: Path) -> Path:
    return output_dir / "research_log.md"


def parse_pairs(raw: str) -> tuple[str, ...]:
    pairs = tuple(
        token.strip()
        for token in raw.replace(";", ",").split(",")
        if token.strip()
    )
    if not pairs:
        raise ValueError("At least one pair is required")
    missing = [str(ohlcv_path(pair)) for pair in pairs if not ohlcv_path(pair).exists()]
    if missing:
        raise FileNotFoundError(f"Missing 1h OHLCV files: {missing}")
    return pairs


def parse_windows(path: Path | None) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    if path is None:
        return dict(DEFAULT_WINDOWS)
    payload = json.loads(path.read_text(encoding="utf-8"))
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for name, bounds in payload.items():
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"Window {name!r} must contain [start, end]")
        start = pd.to_datetime(bounds[0], utc=True)
        end = pd.to_datetime(bounds[1], utc=True)
        if start >= end:
            raise ValueError(f"Window {name!r} has start >= end")
        windows[str(name)] = (start, end)
    if not windows:
        raise ValueError("At least one analysis window is required")
    return windows


def parse_horizons(raw: str) -> tuple[int, ...]:
    horizons = tuple(sorted({int(token.strip()) for token in raw.split(",") if token.strip()}))
    invalid = [horizon for horizon in horizons if horizon not in DEFAULT_DIRECT_HORIZONS]
    if invalid or not horizons:
        raise ValueError(
            f"Direct horizons must be selected from {DEFAULT_DIRECT_HORIZONS}; invalid={invalid}"
        )
    return horizons


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = "# Sieve3 Event-Reaction Round-One Research Log\n\n" if not path.exists() else ""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{prefix}- {now_iso()} — {message}\n")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_parquet(path: Path, frame: DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def parse_scalar(raw: str) -> Any:
    value = raw.strip()
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"none", "null"}:
        return None
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)", value):
            return float(value)
    except ValueError:
        pass
    return value


def parse_parameter_string(raw: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for token in raw.split(";"):
        token = token.strip()
        if not token or "=" not in token:
            continue
        name, value = token.split("=", 1)
        result[name.strip()] = parse_scalar(value)
    return result


def resolve_doc_link(handover: Path, raw: str | None) -> Path | None:
    if not raw:
        return None
    if re.match(r"^[A-Za-z]:[\\/]", raw):
        return Path(raw)
    candidate = Path(raw)
    repo_relative = (REPO_ROOT / candidate).resolve()
    if repo_relative.exists() or candidate.parts[:1] in {("user_data",), ("ai_guidance_docs",)}:
        return repo_relative
    return (handover.parent / candidate).resolve()


def parse_percent(raw: str) -> float:
    cleaned = raw.strip().replace("%", "").replace("+", "")
    return float(cleaned) if cleaned else math.nan


def parse_handover(path: Path) -> list[EntryCandidate]:
    text = path.read_text(encoding="utf-8")
    candidates: list[EntryCandidate] = []
    current: EntryCandidate | None = None
    in_solution_table = False
    for line in text.splitlines():
        heading = re.match(r"^### `([^`]+)`", line)
        if heading:
            current = EntryCandidate(entry_id=heading.group(1))
            candidates.append(current)
            in_solution_table = False
            continue
        if current is None:
            continue
        side_tf = re.match(r"^- Side/timeframe: `([^`]+)` / `([^`]+)`", line)
        if side_tf:
            current.side, current.timeframe = side_tf.groups()
            continue
        source = re.match(r"^- Entry source: `(.+\.py):([A-Za-z0-9_]+)`", line)
        if source:
            raw_path, current.source_class = source.groups()
            current.source_path = resolve_doc_link(path, raw_path)
            continue
        locked = re.match(r"^- Locked entry values used by active/shared paths: `(.*)`\.$", line)
        if locked:
            current.locked_values = parse_parameter_string(locked.group(1))
            continue
        if line.startswith("| Exit solution |"):
            in_solution_table = True
            continue
        if in_solution_table and line.startswith("|---"):
            continue
        if in_solution_table and line.startswith("|"):
            parts = [part.strip() for part in line.strip().strip("|").split("|")]
            if len(parts) < 8 or not parts[0].startswith("`"):
                continue
            evidence = parts[7]
            strategy_match = re.search(r"\[strategy\]\(([^)]+)\)", evidence)
            params_match = re.search(r"\[params\]\(([^)]+)\)", evidence)
            backtest_match = re.search(r"\[backtest\]\(([^)]+)\)", evidence)
            current.exits.append(
                ExitSolution(
                    family=parts[0].strip("`"),
                    selected_values=parts[1].strip("`"),
                    trades=int(parts[2]),
                    win_pct=parse_percent(parts[3]),
                    profit_pct=parse_percent(parts[4]),
                    drawdown_pct=parse_percent(parts[5]),
                    profit_factor=float(parts[6]),
                    strategy_path=resolve_doc_link(path, strategy_match.group(1) if strategy_match else None),
                    params_path=resolve_doc_link(path, params_match.group(1) if params_match else None),
                    backtest_path=resolve_doc_link(path, backtest_match.group(1) if backtest_match else None),
                    sparse="sparse" in evidence.lower(),
                )
            )
            continue
        if line.startswith("## "):
            in_solution_table = False
    return [candidate for candidate in candidates if candidate.source_class]


def parameter_value(strategy: Any, name: str) -> Any:
    value = getattr(strategy, name, None)
    return getattr(value, "value", value)


def values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return bool(actual) is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return str(actual) == str(expected)


def parameter_mismatches(strategy: Any, expected: dict[str, Any]) -> list[dict[str, Any]]:
    mismatches = []
    for name, expected_value in expected.items():
        if not hasattr(strategy, name):
            mismatches.append({"parameter": name, "expected": expected_value, "actual": "missing"})
            continue
        actual = parameter_value(strategy, name)
        if not values_equal(actual, expected_value):
            mismatches.append({"parameter": name, "expected": expected_value, "actual": actual})
    return mismatches


def apply_locked_parameters(strategy: Any, locked_values: dict[str, Any]) -> int:
    """Apply the handover's promoted entry surface, then let the caller verify it."""
    applied = 0
    missing: list[str] = []
    for name, selected_value in locked_values.items():
        parameter = getattr(strategy, name, None)
        if parameter is None or not hasattr(parameter, "value"):
            missing.append(name)
            continue
        parameter.value = selected_value
        applied += 1
    if missing:
        raise ValueError(f"locked parameters missing from strategy: {missing[:8]}")
    return applied


def register_archived_entry_sieve_tools() -> bool:
    """Make the exact archived helper available to historical source strategies."""
    module_name = "user_data.strategies.entry_sieve_tools"
    if module_name in sys.modules:
        loaded_from = Path(str(getattr(sys.modules[module_name], "__file__", "")))
        return loaded_from.resolve() == ARCHIVED_ENTRY_SIEVE_TOOLS.resolve()
    active_path = USER_DATA_DIR / "strategies" / "entry_sieve_tools.py"
    source = active_path if active_path.exists() else ARCHIVED_ENTRY_SIEVE_TOOLS
    if not source.exists():
        raise FileNotFoundError(source)
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return source == ARCHIVED_ENTRY_SIEVE_TOOLS


def timeframe_hours(timeframe: str) -> int:
    match = re.fullmatch(r"(\d+)([mhd])", timeframe.strip().lower())
    if match is None:
        raise ValueError(f"Unsupported timeframe {timeframe!r}")
    value = int(match.group(1))
    unit = match.group(2)
    if unit == "m":
        if 60 % value:
            raise ValueError(f"Timeframe {timeframe!r} cannot align exactly to 1h")
        return max(1, value // 60)
    return value if unit == "h" else value * 24


def strategy_arguments(
    candidate: EntryCandidate,
    *,
    config: Path,
    timerange: str,
    pairs: tuple[str, ...],
) -> dict[str, Any]:
    from freqtrade.commands.arguments import Arguments

    raw = [
        "backtesting",
        "--config",
        str(config),
        "--datadir",
        str(USER_DATA_DIR / "data" / "binance"),
        "--userdir",
        str(USER_DATA_DIR),
        "--strategy-path",
        str(candidate.source_path.parent if candidate.source_path else USER_DATA_DIR / "strategies"),
        "--strategy",
        candidate.source_class,
        "--timeframe",
        candidate.timeframe,
        "--timerange",
        timerange,
        "--export",
        "none",
        "-p",
        *pairs,
    ]
    return Arguments(raw).get_parsed_arg()


def materialize_embedded_sources(
    candidates: list[EntryCandidate], *, output_dir: Path
) -> None:
    """Materialize handover ``archive.zip!member.py`` sources for Freqtrade's resolver."""
    source_dir = output_dir / "cache" / "source_strategies"
    for candidate in candidates:
        reference = str(candidate.source_path or "")
        if "!" not in reference:
            continue
        archive_raw, member_raw = reference.split("!", 1)
        archive = Path(archive_raw)
        member = member_raw.replace("\\", "/").lstrip("/")
        if not archive.exists():
            raise FileNotFoundError(archive)
        if Path(member).name != member or not member.lower().endswith(".py"):
            raise ValueError(f"Unsafe embedded strategy member: {member_raw!r}")
        source_dir.mkdir(parents=True, exist_ok=True)
        destination = source_dir / f"{candidate.entry_id}.py"
        with zipfile.ZipFile(archive) as bundle:
            payload = bundle.read(member)
        if not destination.exists() or destination.read_bytes() != payload:
            temporary = destination.with_suffix(".py.tmp")
            temporary.write_bytes(payload)
            temporary.replace(destination)
        candidate.source_path = destination


def extract_candidate_events(
    candidate: EntryCandidate,
    *,
    config_path: Path,
    timerange: str,
    pairs: tuple[str, ...],
) -> tuple[DataFrame, dict[str, Any]]:
    from freqtrade.commands.optimize_commands import setup_optimize_configuration
    from freqtrade.enums import RunMode
    from freqtrade.optimize.backtesting import Backtesting

    if candidate.source_path is None or not candidate.source_path.exists():
        raise FileNotFoundError(candidate.source_path)
    used_archived_entry_sieve_tools = register_archived_entry_sieve_tools()
    config = setup_optimize_configuration(
        strategy_arguments(candidate, config=config_path, timerange=timerange, pairs=pairs),
        RunMode.BACKTEST,
    )
    # This phase only reconstructs exact Sieve trigger surfaces.  Model training
    # is a later, separate phase and must not alter startup requirements here.
    config.setdefault("freqai", {})["enabled"] = False
    backtesting = Backtesting(config)
    try:
        backtesting._set_strategy(backtesting.strategylist[0])
        strategy = backtesting.strategy
        if strategy.__class__.__name__ != candidate.source_class:
            raise ValueError(
                f"Resolved class {strategy.__class__.__name__} != {candidate.source_class}"
            )
        locked_parameters_applied = apply_locked_parameters(strategy, candidate.locked_values)
        mismatches = parameter_mismatches(strategy, candidate.locked_values)
        if mismatches:
            raise ValueError(f"locked parameter mismatch: {mismatches[:8]}")
        data, _ = backtesting.load_bt_data()
        processed = strategy.advise_all_indicators(data)
        start_raw, end_raw = timerange.split("-", 1)
        start = pd.Timestamp(datetime.strptime(start_raw, "%Y%m%d"), tz="UTC")
        end = pd.Timestamp(datetime.strptime(end_raw, "%Y%m%d"), tz="UTC")
        rows: list[DataFrame] = []
        counts: dict[str, dict[str, int]] = {}
        signal_column = "enter_long" if candidate.side == "long" else "enter_short"
        for pair, frame in processed.items():
            signals = strategy.advise_entry(frame.copy(), {"pair": pair})
            raw_active = pd.to_numeric(
                signals.get(signal_column, pd.Series(0.0, index=signals.index)),
                errors="coerce",
            ).fillna(0.0).gt(0.0)
            dates = pd.to_datetime(signals["date"], utc=True, errors="coerce")
            decision_times = dates + pd.Timedelta(
                hours=timeframe_hours(candidate.timeframe)
            )
            in_window = decision_times.ge(start) & decision_times.lt(end)
            raw_onset = raw_active & ~raw_active.shift(1, fill_value=False)
            active = raw_active & in_window
            onset = raw_onset & in_window
            selected = signals.loc[active, ["date"]].copy()
            selected["date"] = dates.loc[active]
            selected["decision_time"] = decision_times.loc[active]
            selected["pair"] = pair
            selected["entry_id"] = candidate.entry_id
            selected["side"] = candidate.side
            selected["timeframe"] = candidate.timeframe
            selected["is_onset"] = onset.loc[active].astype(bool).to_numpy()
            selected["source_path"] = str(candidate.source_path)
            selected["source_class"] = candidate.source_class
            rows.append(selected)
            counts[pair] = {
                "active": int(active.sum()),
                "onset": int(onset.sum()),
            }
        events = pd.concat(rows, ignore_index=True) if rows else DataFrame()
        return events, {
            "entry_id": candidate.entry_id,
            "status": "completed",
            "side": candidate.side,
            "timeframe": candidate.timeframe,
            "used_archived_entry_sieve_tools": used_archived_entry_sieve_tools,
            "locked_parameters_applied": locked_parameters_applied,
            "locked_parameters_checked": len(candidate.locked_values),
            "counts": counts,
            "finished_at": now_iso(),
        }
    finally:
        try:
            backtesting.exchange.close()
        finally:
            Backtesting.cleanup()
        del backtesting
        gc.collect()


def family_for_entry(entry_id: str) -> str:
    lowered = entry_id.lower()
    for token, family in (
        ("pivot", "pivot"),
        ("tlv2", "tlv2"),
        ("vp_", "volume_profile"),
        ("vah", "volume_profile"),
        ("val", "volume_profile"),
        ("hvn", "volume_profile"),
        ("bos", "market_structure"),
        ("choch", "market_structure"),
        ("pattern", "pattern"),
        ("wolfe", "pattern"),
        ("triangle", "pattern"),
        ("wedge", "pattern"),
        ("pennant", "pattern"),
        ("crash", "liquidity_risk"),
        ("liquidity", "liquidity_risk"),
        ("prior_month", "rolling_extreme"),
        ("prior_high", "rolling_extreme"),
        ("prior_low", "rolling_extreme"),
        ("mtf_std", "standard_mtf"),
        ("mtf", "multi_timeframe"),
    ):
        if token in lowered:
            return family
    return "other"


def ohlcv_path(pair: str) -> Path:
    token = pair.replace("/", "_").replace(":", "_")
    return USER_DATA_DIR / "data" / "binance" / "futures" / f"{token}-1h-futures.feather"


def build_wide_event_caches(
    events: DataFrame,
    candidates: list[EntryCandidate],
    *,
    pairs: tuple[str, ...],
    cache_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata = {candidate.entry_id: candidate for candidate in candidates}
    all_entry_ids = [candidate.entry_id for candidate in candidates]
    all_families = sorted({family_for_entry(entry_id) for entry_id in all_entry_ids})
    audit: dict[str, Any] = {"pairs": {}, "generated_at": now_iso()}
    for pair in pairs:
        raw = pd.read_feather(ohlcv_path(pair), columns=["date"])
        dates = pd.to_datetime(raw["date"], utc=True, errors="coerce")
        decision_time = dates + pd.Timedelta(hours=1)
        base = DataFrame({"decision_time": decision_time})
        base = base[
            base["decision_time"].ge(start) & base["decision_time"].lt(end)
        ].dropna().drop_duplicates("decision_time").sort_values("decision_time")
        base["coverage_present"] = 1
        pair_events = events[events["pair"].eq(pair)].copy()
        active = pair_events.pivot_table(
            index="decision_time", columns="entry_id", values="date", aggfunc="size", fill_value=0
        )
        onset = pair_events[pair_events["is_onset"]].pivot_table(
            index="decision_time", columns="entry_id", values="date", aggfunc="size", fill_value=0
        )
        for entry_id in all_entry_ids:
            base[f"entry_active__{entry_id}"] = (
                base["decision_time"].map(active[entry_id]) if entry_id in active else 0
            )
            base[f"entry_onset__{entry_id}"] = (
                base["decision_time"].map(onset[entry_id]) if entry_id in onset else 0
            )
        active_columns = [f"entry_active__{entry_id}" for entry_id in all_entry_ids]
        onset_columns = [f"entry_onset__{entry_id}" for entry_id in all_entry_ids]
        base[active_columns + onset_columns] = base[active_columns + onset_columns].fillna(0.0).astype("int8")
        long_ids = [entry_id for entry_id in all_entry_ids if metadata[entry_id].side == "long"]
        short_ids = [entry_id for entry_id in all_entry_ids if metadata[entry_id].side == "short"]
        base["summary__active_count"] = base[active_columns].sum(axis=1).astype("int16")
        base["summary__onset_count"] = base[onset_columns].sum(axis=1).astype("int16")
        base["summary__long_onset_count"] = base[[f"entry_onset__{x}" for x in long_ids]].sum(axis=1).astype("int16")
        base["summary__short_onset_count"] = base[[f"entry_onset__{x}" for x in short_ids]].sum(axis=1).astype("int16")
        base["summary__opposing_onset_conflict"] = (
            base["summary__long_onset_count"].gt(0) & base["summary__short_onset_count"].gt(0)
        ).astype("int8")
        base["summary__same_side_onset_overlap"] = np.maximum(
            base["summary__long_onset_count"], base["summary__short_onset_count"]
        ).sub(1).clip(lower=0).astype("int16")
        for family in all_families:
            ids = [entry_id for entry_id in all_entry_ids if family_for_entry(entry_id) == family]
            base[f"summary__family_{family}_onset_count"] = base[
                [f"entry_onset__{entry_id}" for entry_id in ids]
            ].sum(axis=1).astype("int16")
        output = cache_dir / f"{pair_token(pair)}_sieve3_events_1h.parquet"
        atomic_parquet(output, base)
        missing_event_times = int((~pair_events["decision_time"].isin(base["decision_time"])).sum())
        audit["pairs"][pair] = {
            "path": str(output),
            "rows": int(len(base)),
            "start": base["decision_time"].min().isoformat(),
            "end": base["decision_time"].max().isoformat(),
            "duplicate_decision_times": int(base["decision_time"].duplicated().sum()),
            "active_events": int(base["summary__active_count"].sum()),
            "onset_events": int(base["summary__onset_count"].sum()),
            "opposing_conflicts": int(base["summary__opposing_onset_conflict"].sum()),
            "missing_event_times": missing_event_times,
        }
    return audit


def build_entry_catalogue(
    candidates: list[EntryCandidate],
    *,
    config_path: Path,
    output_dir: Path,
    timerange: str,
    pairs: tuple[str, ...],
    selected_ids: set[str] | None,
    max_entries: int | None,
) -> dict[str, Any]:
    cache_dir = output_dir / "cache"
    catalogue_config_path = cache_dir / "catalogue_config.json"
    catalogue_config = json.loads(config_path.read_text(encoding="utf-8"))
    catalogue_config.setdefault("freqai", {})["enabled"] = False
    atomic_json(catalogue_config_path, catalogue_config)
    progress_path = output_dir / "entry_catalogue_progress.json"
    partial_path = cache_dir / "entry_events_long.partial.parquet"
    log_path = research_log_path(output_dir)
    progress: dict[str, Any] = {
        "started_at": now_iso(),
        "timerange": timerange,
        "completed": [],
        "errors": [],
    }
    events = DataFrame()
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if partial_path.exists():
        events = pd.read_parquet(partial_path)
    completed = {str(item["entry_id"]) for item in progress.get("completed", [])}
    parked = {
        str(item["entry_id"])
        for item in progress.get("errors", [])
        if str(item.get("status")) == "parked"
    }
    work = [candidate for candidate in candidates if selected_ids is None or candidate.entry_id in selected_ids]
    if max_entries is not None:
        work = work[:max_entries]
    materialize_embedded_sources(work, output_dir=output_dir)
    append_log(
        log_path,
        f"Entry catalogue started/resumed for {len(work)} frozen entries; baseline is raw trigger candles from the exact locked strategy defaults, independent of trade occupancy.",
    )
    milestones = {max(1, math.ceil(len(work) * fraction)) for fraction in (0.25, 0.50, 0.75, 1.0)}
    for index, candidate in enumerate(work, start=1):
        if candidate.entry_id in completed or candidate.entry_id in parked:
            continue
        progress["errors"] = [
            item
            for item in progress.get("errors", [])
            if str(item.get("entry_id")) != candidate.entry_id
        ]
        try:
            extracted, result = extract_candidate_events(
                candidate,
                config_path=catalogue_config_path,
                timerange=timerange,
                pairs=pairs,
            )
            events = pd.concat([events, extracted], ignore_index=True)
            progress.setdefault("completed", []).append(result)
            atomic_parquet(partial_path, events)
        except Exception as exc:
            is_parked = isinstance(exc, pd.errors.MergeError)
            error = {
                "entry_id": candidate.entry_id,
                "status": "parked" if is_parked else "error",
                "error": f"{type(exc).__name__}: {exc}",
                "failed_at": now_iso(),
            }
            progress.setdefault("errors", []).append(error)
            disposition = "Parked" if is_parked else "Failed"
            append_log(log_path, f"{disposition} entry `{candidate.entry_id}` during catalogue extraction: {error['error']}")
        progress["updated_at"] = now_iso()
        atomic_json(progress_path, progress)
        if index in milestones:
            print(
                json.dumps(
                    {
                        "phase": "entry_catalogue",
                        "processed": index,
                        "total": len(work),
                        "completed": len(progress.get("completed", [])),
                        "errors": len(progress.get("errors", [])),
                    }
                ),
                flush=True,
            )
    if events.empty:
        raise ValueError("No raw entry events were extracted")
    start_raw, end_raw = timerange.split("-", 1)
    start = pd.Timestamp(datetime.strptime(start_raw, "%Y%m%d"), tz="UTC")
    end = pd.Timestamp(datetime.strptime(end_raw, "%Y%m%d"), tz="UTC")
    events["decision_time"] = pd.to_datetime(events["decision_time"], utc=True, errors="coerce")
    invalid_decision_time_rows = int(events["decision_time"].isna().sum())
    if invalid_decision_time_rows:
        raise ValueError(
            f"Entry catalogue contains {invalid_decision_time_rows} invalid decision timestamps"
        )
    in_window = events["decision_time"].ge(start) & events["decision_time"].lt(end)
    excluded_out_of_window_event_rows = int((~in_window).sum())
    events = (
        events.loc[in_window]
        .drop_duplicates(["pair", "decision_time", "entry_id"], keep="last")
        .sort_values(["pair", "decision_time", "entry_id"])
        .reset_index(drop=True)
    )
    final_events_path = cache_dir / "entry_events_long.parquet"
    atomic_parquet(partial_path, events)
    atomic_parquet(final_events_path, events)
    cache_audit = build_wide_event_caches(
        events,
        work,
        pairs=pairs,
        cache_dir=cache_dir,
        start=pd.Timestamp(datetime.strptime(start_raw, "%Y%m%d"), tz="UTC"),
        end=pd.Timestamp(datetime.strptime(end_raw, "%Y%m%d"), tz="UTC"),
    )
    progress["finished_at"] = now_iso()
    progress["events_path"] = str(final_events_path)
    progress["active_event_rows"] = int(len(events))
    progress["onset_event_rows"] = int(events["is_onset"].sum())
    progress["excluded_out_of_window_event_rows"] = excluded_out_of_window_event_rows
    progress["invalid_decision_time_rows"] = invalid_decision_time_rows
    progress["cache_audit"] = cache_audit
    atomic_json(progress_path, progress)
    append_log(
        log_path,
        f"Entry catalogue completed with {len(events)} active event rows and {int(events['is_onset'].sum())} de-duplicated event onsets; errors={len(progress.get('errors', []))}. Audit: `{progress_path}`.",
    )
    return progress


def build_exit_catalogue(
    candidates: list[EntryCandidate], *, output_dir: Path
) -> dict[str, Any]:
    from freqtrade.data.btanalysis import load_backtest_data

    rows: list[DataFrame] = []
    errors: list[dict[str, str]] = []
    log_path = research_log_path(output_dir)
    solutions = [(candidate, solution) for candidate in candidates for solution in candidate.exits]
    milestones = {max(1, math.ceil(len(solutions) * fraction)) for fraction in (0.25, 0.50, 0.75, 1.0)}
    for index, (candidate, solution) in enumerate(solutions, start=1):
        try:
            if solution.backtest_path is None or not solution.backtest_path.exists():
                raise FileNotFoundError(solution.backtest_path)
            trades = load_backtest_data(solution.backtest_path)
            if trades.empty:
                continue
            selected = trades.copy()
            selected["entry_id"] = candidate.entry_id
            selected["entry_side"] = candidate.side
            selected["entry_timeframe"] = candidate.timeframe
            selected["exit_family"] = solution.family
            selected["source_validation_profit_pct"] = solution.profit_pct
            selected["source_validation_drawdown_pct"] = solution.drawdown_pct
            selected["source_validation_profit_factor"] = solution.profit_factor
            selected["source_backtest_path"] = str(solution.backtest_path)
            selected["source_sparse"] = solution.sparse
            rows.append(selected)
        except Exception as exc:
            errors.append(
                {
                    "entry_id": candidate.entry_id,
                    "exit_family": solution.family,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if index in milestones:
            print(
                json.dumps(
                    {
                        "phase": "exit_catalogue",
                        "processed": index,
                        "total": len(solutions),
                        "errors": len(errors),
                    }
                ),
                flush=True,
            )
    exits = pd.concat(rows, ignore_index=True) if rows else DataFrame()
    if not exits.empty:
        exits["open_date"] = pd.to_datetime(exits["open_date"], utc=True, errors="coerce")
        exits["close_date"] = pd.to_datetime(exits["close_date"], utc=True, errors="coerce")
        exits["decision_time"] = exits["close_date"].dt.floor("h")
        output_path = output_dir / "cache" / "exit_events_long.parquet"
        atomic_parquet(output_path, exits)
    else:
        output_path = output_dir / "cache" / "exit_events_long.parquet"
    audit = {
        "generated_at": now_iso(),
        "solutions": len(solutions),
        "loaded_solution_frames": len(rows),
        "trade_exit_rows": int(len(exits)),
        "errors": errors,
        "path": str(output_path),
    }
    atomic_json(output_dir / "exit_catalogue_audit.json", audit)
    append_log(
        log_path,
        f"Exit catalogue read {len(rows)}/{len(solutions)} resolved backtests and recovered {len(exits)} executed exit rows; errors={len(errors)}. These are open-trade-conditioned events, unlike the raw entry catalogue.",
    )
    return audit


def actual_frame(pair: str, start: pd.Timestamp, end: pd.Timestamp) -> DataFrame:
    raw = pd.read_feather(ohlcv_path(pair))
    raw["date"] = pd.to_datetime(raw["date"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date")
    labelled = build_sieve3_reaction_targets(raw)
    labelled["decision_time"] = labelled["date"] + pd.Timedelta(hours=1)
    labelled["pair"] = pair
    close = pd.to_numeric(labelled["close"], errors="coerce").replace(0.0, np.nan)
    high = pd.to_numeric(labelled["high"], errors="coerce")
    low = pd.to_numeric(labelled["low"], errors="coerce")
    volume = pd.to_numeric(labelled["volume"], errors="coerce").clip(lower=0.0)
    atr = _atr(labelled)
    labelled["atr"] = atr
    labelled["atr_pct"] = atr / close
    labelled["pre_return_6h"] = close.pct_change(6)
    labelled["pre_return_24h"] = close.pct_change(24)
    labelled["pre_volume_ratio_24h"] = (
        volume
        / volume.shift(1).rolling(24, min_periods=12).mean().replace(0.0, np.nan)
    ).clip(0.0, 20.0)
    candle_range = (high - low).replace(0.0, np.nan)
    close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
    signed_volume = close_location.fillna(0.0) * volume.fillna(0.0)
    for lookback in (6, 24):
        labelled[f"pre_pressure_{lookback}h"] = (
            signed_volume.rolling(lookback, min_periods=max(3, lookback // 2)).sum()
            / volume.rolling(lookback, min_periods=max(3, lookback // 2)).sum().replace(0.0, np.nan)
        )
    prior_close = close.shift(1)
    true_range = pd.concat(
        ((high - low).abs(), (high - prior_close).abs(), (low - prior_close).abs()),
        axis=1,
    ).max(axis=1)
    labelled["pre_volatility_24h"] = (
        true_range.rolling(24, min_periods=12).mean() / close
    )
    prior_high = high.shift(1).rolling(168, min_periods=84).max()
    prior_low = low.shift(1).rolling(168, min_periods=84).min()
    labelled["pre_range_position_168h"] = (
        (close - prior_low) / (prior_high - prior_low).replace(0.0, np.nan)
    )
    return labelled[
        labelled["decision_time"].ge(start - pd.Timedelta(days=2))
        & labelled["decision_time"].lt(end + pd.Timedelta(days=2))
    ].copy()


def oriented_outcomes(frame: DataFrame, side: str, horizon: int) -> DataFrame:
    sign = 1.0 if side == "long" else -1.0
    result = DataFrame(index=frame.index)
    def numeric_column(name: str) -> Series:
        if name not in frame:
            return pd.Series(np.nan, index=frame.index, dtype="float64")
        return pd.to_numeric(frame[name], errors="coerce")

    result["signed_return"] = pd.to_numeric(frame[f"&-future_return_{horizon}h"], errors="coerce") * sign
    if side == "long":
        result["best_move_if_held_atr"] = pd.to_numeric(frame[f"&-future_upside_{horizon}h_atr"], errors="coerce")
        result["worst_move_if_held_atr"] = pd.to_numeric(frame[f"&-future_downside_{horizon}h_atr"], errors="coerce")
        result["best_move_peak_step"] = numeric_column(
            f"&-future_upside_peak_step_{horizon}h"
        )
        result["worst_move_peak_step"] = numeric_column(
            f"&-future_downside_peak_step_{horizon}h"
        )
        result["target_before_invalidation"] = numeric_column(
            f"&-up_before_down_1atr_{horizon}h"
        )
    else:
        result["best_move_if_held_atr"] = pd.to_numeric(frame[f"&-future_downside_{horizon}h_atr"], errors="coerce")
        result["worst_move_if_held_atr"] = pd.to_numeric(frame[f"&-future_upside_{horizon}h_atr"], errors="coerce")
        result["best_move_peak_step"] = numeric_column(
            f"&-future_downside_peak_step_{horizon}h"
        )
        result["worst_move_peak_step"] = numeric_column(
            f"&-future_upside_peak_step_{horizon}h"
        )
        raw_touch = numeric_column(f"&-up_before_down_1atr_{horizon}h")
        result["target_before_invalidation"] = 1.0 - raw_touch
    result["first_1atr_touch_step"] = numeric_column(
        f"&-first_1atr_touch_step_{horizon}h"
    )
    result["path_balance_atr"] = result["best_move_if_held_atr"] - result["worst_move_if_held_atr"]
    result["reaction_magnitude_atr"] = result["best_move_if_held_atr"] + result["worst_move_if_held_atr"]
    result["volume_ratio"] = pd.to_numeric(frame[f"&-future_volume_ratio_{horizon}h"], errors="coerce")
    result["pressure_alignment"] = pd.to_numeric(frame[f"&-future_pressure_{horizon}h"], errors="coerce") * sign
    result["volatility_ratio"] = pd.to_numeric(frame[f"&-future_volatility_ratio_{horizon}h"], errors="coerce")
    return result


def nearest_neighbor_candidates(
    pool_values: np.ndarray,
    event_values: np.ndarray,
    neighbor_count: int,
    *,
    chunk_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact Euclidean neighbours without importing scikit-learn."""
    if neighbor_count < 1 or neighbor_count > len(pool_values):
        raise ValueError("neighbor_count must be within the available pool")
    distance_chunks: list[np.ndarray] = []
    index_chunks: list[np.ndarray] = []
    for start in range(0, len(event_values), chunk_size):
        chunk = event_values[start : start + chunk_size]
        deltas = chunk[:, np.newaxis, :] - pool_values[np.newaxis, :, :]
        squared = np.sum(deltas * deltas, axis=2)
        nearest = np.argpartition(
            squared,
            kth=neighbor_count - 1,
            axis=1,
        )[:, :neighbor_count]
        nearest_squared = np.take_along_axis(squared, nearest, axis=1)
        order = np.argsort(nearest_squared, axis=1, kind="stable")
        nearest = np.take_along_axis(nearest, order, axis=1)
        nearest_squared = np.take_along_axis(nearest_squared, order, axis=1)
        index_chunks.append(nearest)
        distance_chunks.append(np.sqrt(nearest_squared))
    return (
        np.vstack(distance_chunks),
        np.vstack(index_chunks),
    )


def matched_controls(
    scoped: DataFrame, event_mask: pd.Series, *, rng: np.random.Generator
) -> DataFrame:
    """Match ordinary rows to the full visible pre-event OHLCV state."""
    event_rows = scoped[event_mask].copy()
    ordinary = scoped[~event_mask & scoped["summary__onset_count"].eq(0)].copy()
    if event_rows.empty or ordinary.empty:
        return DataFrame(columns=scoped.columns)
    _ = rng
    features = (
        "atr_pct",
        "pre_return_6h",
        "pre_return_24h",
        "pre_volume_ratio_24h",
        "pre_pressure_6h",
        "pre_pressure_24h",
        "pre_volatility_24h",
        "pre_range_position_168h",
    )
    event_rows["match_quarter"] = (
        event_rows["decision_time"].dt.year.astype(str)
        + "Q"
        + event_rows["decision_time"].dt.quarter.astype(str)
    )
    ordinary["match_quarter"] = (
        ordinary["decision_time"].dt.year.astype(str)
        + "Q"
        + ordinary["decision_time"].dt.quarter.astype(str)
    )
    samples: list[DataFrame] = []
    for (pair, quarter), group in event_rows.groupby(
        ["pair", "match_quarter"], dropna=False
    ):
        pool = ordinary[
            ordinary["pair"].eq(pair)
            & ordinary["match_quarter"].eq(quarter)
        ]
        pool_kind = "same_pair_quarter"
        if len(pool) < 20:
            pool = ordinary[ordinary["pair"].eq(pair)]
            pool_kind = "same_pair_window"
        if pool.empty:
            continue
        combined = pd.concat(
            [pool.loc[:, list(features)], group.loc[:, list(features)]],
            ignore_index=True,
        )
        numeric = combined.apply(pd.to_numeric, errors="coerce")
        medians = numeric.median(axis=0)
        filled = numeric.fillna(medians).fillna(0.0)
        scale = (filled.quantile(0.75) - filled.quantile(0.25)).replace(0.0, 1.0)
        standardized = (filled - medians.fillna(0.0)) / scale
        pool_values = standardized.iloc[: len(pool)].to_numpy(dtype="float64")
        event_values = standardized.iloc[len(pool) :].to_numpy(dtype="float64")
        neighbor_count = min(len(pool), max(20, min(200, len(group) * 2)))
        distances, indices = nearest_neighbor_candidates(
            pool_values,
            event_values,
            neighbor_count,
        )
        pool_times = pd.to_datetime(pool["decision_time"], utc=True, errors="coerce").reset_index(drop=True)
        event_times = pd.to_datetime(group["decision_time"], utc=True, errors="coerce").reset_index(drop=True)
        selected_positions: list[int] = []
        selected_distances: list[float] = []
        separations: list[float] = []
        reused_flags: list[bool] = []
        used_positions: set[int] = set()
        for row_index, candidates in enumerate(indices):
            candidate_separations = [
                abs(
                    (
                        pool_times.iloc[int(candidate_position)]
                        - event_times.iloc[row_index]
                    ).total_seconds()
                )
                / 3600.0
                for candidate_position in candidates
            ]
            chosen_rank = 0
            for require_unused, require_separated in (
                (True, True),
                (True, False),
                (False, True),
                (False, False),
            ):
                eligible = [
                    rank
                    for rank, candidate_position in enumerate(candidates)
                    if (not require_unused or int(candidate_position) not in used_positions)
                    and (not require_separated or candidate_separations[rank] > 72.0)
                ]
                if eligible:
                    chosen_rank = eligible[0]
                    break
            chosen = int(candidates[chosen_rank])
            selected_positions.append(chosen)
            selected_distances.append(float(distances[row_index, chosen_rank]))
            separations.append(float(candidate_separations[chosen_rank]))
            reused_flags.append(chosen in used_positions)
            used_positions.add(chosen)
        matched = pool.iloc[selected_positions].copy().reset_index(drop=True)
        matched["control_match_distance"] = selected_distances
        matched["control_match_time_separation_h"] = separations
        matched["control_match_pool"] = pool_kind
        matched["control_match_reused"] = reused_flags
        samples.append(matched)
    return pd.concat(samples, ignore_index=True) if samples else DataFrame(columns=scoped.columns)


def summarize_sample(
    sample: DataFrame,
    *,
    side: str,
    horizon: int,
    sample_name: str,
) -> dict[str, Any]:
    outcomes = oriented_outcomes(sample, side, horizon)
    result: dict[str, Any] = {"sample": sample_name, "rows": int(len(sample))}
    for column in outcomes:
        values = pd.to_numeric(outcomes[column], errors="coerce").dropna()
        result[f"{column}_mean"] = float(values.mean()) if len(values) else None
        result[f"{column}_median"] = float(values.median()) if len(values) else None
        result[f"{column}_rows"] = int(len(values))
    return result


def build_exit_counterfactuals(
    exits: DataFrame,
    actual_lookup: DataFrame,
    *,
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    horizons: tuple[int, ...],
    earlier_offsets: tuple[int, ...] = (1, 2, 4, 8),
) -> tuple[DataFrame, DataFrame]:
    """Compare an executed exit with observable earlier and later market paths.

    Price-relative counterfactuals use the executed exit price as their common
    base. This keeps partial exits and fee accounting from being mistaken for a
    pure timing effect.
    """
    lookup_columns = [
        "pair",
        "date",
        "high",
        "low",
        "close",
    ]
    available = [column for column in lookup_columns if column in actual_lookup]
    market = (
        actual_lookup.loc[:, available]
        .rename(columns={"date": "exit_candle_time"})
        .drop_duplicates(["pair", "exit_candle_time"], keep="last")
        .sort_values(["pair", "exit_candle_time"], kind="stable")
        .copy()
    )
    for horizon in horizons:
        market[f"future_close_raw_{horizon}h"] = market.groupby(
            "pair", sort=False, observed=True
        )["close"].shift(-horizon)
        market[f"future_high_raw_{horizon}h"] = market.groupby(
            "pair", sort=False, observed=True
        )["high"].transform(lambda values: _future_extreme(values, horizon, "max"))
        market[f"future_low_raw_{horizon}h"] = market.groupby(
            "pair", sort=False, observed=True
        )["low"].transform(lambda values: _future_extreme(values, horizon, "min"))
    work = exits.copy()
    work["decision_time"] = pd.to_datetime(work["decision_time"], utc=True, errors="coerce")
    work["open_date"] = pd.to_datetime(work["open_date"], utc=True, errors="coerce")
    work["exit_candle_time"] = work["decision_time"]
    work = work.merge(
        market,
        on=["pair", "exit_candle_time"],
        how="inner",
        validate="many_to_one",
    )
    work["window"] = "outside"
    for name, (start, end) in windows.items():
        work.loc[
            work["decision_time"].ge(start) & work["decision_time"].lt(end),
            "window",
        ] = name
    work = work[~work["window"].eq("outside")].copy()
    if work.empty:
        return DataFrame(), DataFrame()

    work["side_sign"] = np.where(
        work["entry_side"].astype(str).str.lower().eq("short"), -1.0, 1.0
    )
    exit_price = pd.to_numeric(work["close_rate"], errors="coerce").replace(0.0, np.nan)
    market_price = pd.to_numeric(work["close"], errors="coerce").replace(0.0, np.nan)
    work["exit_price_vs_exit_candle_close_delta"] = work["side_sign"] * (
        exit_price / market_price - 1.0
    )

    price_lookup = market.loc[:, ["pair", "exit_candle_time", "close"]].copy()
    for offset in earlier_offsets:
        earlier = price_lookup.rename(columns={"close": f"market_price_{offset}h_before"})
        earlier["exit_candle_time"] += pd.Timedelta(hours=offset)
        work = work.merge(
            earlier,
            on=["pair", "exit_candle_time"],
            how="left",
            validate="many_to_one",
        )
        valid = work["exit_candle_time"].sub(pd.Timedelta(hours=offset)).ge(work["open_date"])
        earlier_price = pd.to_numeric(
            work[f"market_price_{offset}h_before"], errors="coerce"
        ).where(valid)
        work[f"earlier_exit_advantage_{offset}h"] = work["side_sign"] * (
            earlier_price / exit_price - 1.0
        )

    event_frames: list[DataFrame] = []
    for horizon in horizons:
        required = (
            f"future_close_raw_{horizon}h",
            f"future_high_raw_{horizon}h",
            f"future_low_raw_{horizon}h",
        )
        if any(column not in work for column in required):
            continue
        terminal_price = pd.to_numeric(work[required[0]], errors="coerce")
        future_high = pd.to_numeric(work[required[1]], errors="coerce")
        future_low = pd.to_numeric(work[required[2]], errors="coerce")
        long_mask = work["side_sign"].gt(0.0)
        best_price = future_high.where(long_mask, future_low)
        worst_price = future_low.where(long_mask, future_high)
        event = work.copy()
        event["horizon_hours"] = int(horizon)
        event["hold_instead_delta"] = event["side_sign"] * (
            terminal_price / exit_price - 1.0
        )
        best_delta = event["side_sign"] * (best_price / exit_price - 1.0)
        worst_delta = event["side_sign"] * (worst_price / exit_price - 1.0)
        event["missed_additional_profit"] = best_delta.clip(lower=0.0)
        event["avoided_loss_after_exit"] = (-worst_delta).clip(lower=0.0)
        event["net_exit_regret"] = (
            event["missed_additional_profit"]
            - event["avoided_loss_after_exit"]
        )
        event["best_post_exit_delta"] = best_delta
        event["worst_post_exit_delta"] = worst_delta
        event_frames.append(event)
    events = pd.concat(event_frames, ignore_index=True) if event_frames else DataFrame()
    if events.empty:
        return events, DataFrame()

    metric_columns = [
        "profit_ratio",
        "exit_price_vs_exit_candle_close_delta",
        "hold_instead_delta",
        "missed_additional_profit",
        "avoided_loss_after_exit",
        "net_exit_regret",
        "best_post_exit_delta",
        "worst_post_exit_delta",
        *[f"earlier_exit_advantage_{offset}h" for offset in earlier_offsets],
    ]
    summary_rows: list[dict[str, Any]] = []
    group_columns = ["entry_id", "exit_family", "pair", "entry_side", "window", "horizon_hours"]
    for keys, group in events.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys, strict=True))
        row["exit_rows"] = int(len(group))
        row["unique_trade_paths"] = int(
            group[["pair", "open_date", "entry_id"]].drop_duplicates().shape[0]
        )
        row["unique_source_backtests"] = int(group["source_backtest_path"].nunique())
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            row[f"{column}_rows"] = int(len(values))
            row[f"{column}_mean"] = float(values.mean()) if len(values) else None
            row[f"{column}_median"] = float(values.median()) if len(values) else None
            row[f"{column}_positive_share"] = float(values.gt(0.0).mean()) if len(values) else None
        summary_rows.append(row)
    event_columns = [
        "entry_id",
        "exit_family",
        "pair",
        "entry_side",
        "entry_timeframe",
        "window",
        "horizon_hours",
        "open_date",
        "decision_time",
        "exit_candle_time",
        "open_rate",
        "close_rate",
        "profit_ratio",
        "exit_reason",
        "source_backtest_path",
        "source_sparse",
        "exit_price_vs_exit_candle_close_delta",
        "hold_instead_delta",
        "missed_additional_profit",
        "avoided_loss_after_exit",
        "net_exit_regret",
        "best_post_exit_delta",
        "worst_post_exit_delta",
        *[f"earlier_exit_advantage_{offset}h" for offset in earlier_offsets],
    ]
    return events.loc[:, [column for column in event_columns if column in events]], DataFrame(summary_rows)


def summarize_direct_portability(entry_results: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Describe cross-pair/window consistency without turning it into promotion."""
    if entry_results.empty or "pair_scope" not in entry_results:
        return DataFrame(), DataFrame()
    metrics = (
        "signed_return",
        "path_balance_atr",
        "reaction_magnitude_atr",
        "volume_ratio",
        "pressure_alignment",
        "volatility_ratio",
        "target_before_invalidation",
    )
    scoped = entry_results[~entry_results["pair_scope"].eq("pooled")].copy()
    records: list[dict[str, Any]] = []
    for metric in metrics:
        control_column = f"event_minus_control_{metric}_mean"
        placebo_column = f"event_minus_placebo_{metric}_mean"
        precursor_column = f"event_minus_precursor_{metric}_mean"
        if control_column not in scoped:
            continue
        work = scoped[
            [
                "entry_id",
                "family",
                "side",
                "entry_timeframe",
                "window",
                "pair_scope",
                "horizon_hours",
                "native_horizon_multiple",
                "event_rows",
                control_column,
                placebo_column,
                precursor_column,
            ]
        ].copy()
        work["metric"] = metric
        work = work.rename(
            columns={
                control_column: "delta_vs_matched",
                placebo_column: "delta_vs_placebo",
                precursor_column: "delta_vs_precursor",
            }
        )
        records.extend(work.to_dict("records"))
    long = DataFrame(records)
    if long.empty:
        return DataFrame(), DataFrame()

    def aggregate(grouping: list[str]) -> DataFrame:
        rows: list[dict[str, Any]] = []
        for keys, group in long.groupby(grouping, dropna=False, observed=True):
            row = dict(zip(grouping, keys, strict=True))
            row["pair_window_tests"] = int(len(group))
            row["distinct_pairs"] = int(group["pair_scope"].nunique())
            row["distinct_windows"] = int(group["window"].nunique())
            row["summed_event_rows"] = int(
                pd.to_numeric(group["event_rows"], errors="coerce").fillna(0).sum()
            )
            for label in ("matched", "placebo", "precursor"):
                values = pd.to_numeric(group[f"delta_vs_{label}"], errors="coerce").dropna()
                row[f"{label}_tests"] = int(len(values))
                row[f"{label}_median_delta"] = float(values.median()) if len(values) else None
                row[f"{label}_positive_share"] = float(values.gt(0.0).mean()) if len(values) else None
            matched_share = row.get("matched_positive_share")
            placebo_share = row.get("placebo_positive_share")
            if row["distinct_pairs"] < 2 or row["pair_window_tests"] < 3:
                shape = "insufficient_or_isolated"
            elif matched_share is not None and placebo_share is not None and matched_share >= 0.65 and placebo_share >= 0.65:
                shape = "broad_positive_lead"
            elif matched_share is not None and placebo_share is not None and matched_share <= 0.35 and placebo_share <= 0.35:
                shape = "broad_negative_lead"
            else:
                shape = "mixed_or_context_dependent"
            precursor_share = row.get("precursor_positive_share")
            if precursor_share is None:
                precursor_relation = "unavailable"
            elif precursor_share >= 0.65:
                precursor_relation = "trigger_stronger_than_precursor"
            elif precursor_share <= 0.35:
                precursor_relation = "precursor_stronger_than_trigger"
            else:
                precursor_relation = "mixed"
            row["descriptive_evidence_shape"] = shape
            row["precursor_relation"] = precursor_relation
            row["isolation_warning"] = row["distinct_pairs"] < 3
            row["single_window_warning"] = row["distinct_windows"] < 2
            row["interpretation"] = "descriptive lead only; not a pass/fail or promotion decision"
            rows.append(row)
        return DataFrame(rows)

    entry_summary = aggregate(
        [
            "entry_id",
            "family",
            "side",
            "entry_timeframe",
            "horizon_hours",
            "native_horizon_multiple",
            "metric",
        ]
    )
    family_summary = aggregate(["family", "side", "horizon_hours", "metric"])
    return entry_summary, family_summary


def summarize_entry_coverage(
    events: DataFrame,
    candidates: list[EntryCandidate],
    *,
    pairs: tuple[str, ...],
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> DataFrame:
    work = events.copy()
    work["decision_time"] = pd.to_datetime(work["decision_time"], utc=True, errors="coerce")
    work["window"] = "outside"
    for name, (start, end) in windows.items():
        work.loc[
            work["decision_time"].ge(start) & work["decision_time"].lt(end),
            "window",
        ] = name
    work = work[work["pair"].isin(pairs) & ~work["window"].eq("outside")]
    counts = (
        work.groupby(["entry_id", "pair", "window"], observed=True)
        .agg(
            active_event_rows=("is_onset", "size"),
            onset_event_rows=("is_onset", "sum"),
        )
        .reset_index()
    )
    grid = pd.MultiIndex.from_product(
        [[candidate.entry_id for candidate in candidates], pairs, list(windows)],
        names=["entry_id", "pair", "window"],
    ).to_frame(index=False)
    result = grid.merge(counts, on=["entry_id", "pair", "window"], how="left")
    result[["active_event_rows", "onset_event_rows"]] = result[
        ["active_event_rows", "onset_event_rows"]
    ].fillna(0).astype(int)
    metadata = {
        candidate.entry_id: (candidate.side, candidate.timeframe)
        for candidate in candidates
    }
    result["side"] = result["entry_id"].map(lambda value: metadata[value][0])
    result["entry_timeframe"] = result["entry_id"].map(lambda value: metadata[value][1])
    result["family"] = result["entry_id"].map(family_for_entry)
    result["evidence_size"] = pd.cut(
        result["onset_event_rows"],
        bins=[-1, 0, 9, 29, np.inf],
        labels=["zero", "very_sparse_1_to_9", "thin_10_to_29", "30_plus"],
    ).astype(str)
    result["interpretation"] = "sample-size description only; not an edge or promotion decision"
    return result


def run_direct_baselines(
    candidates: list[EntryCandidate], *, output_dir: Path,
    pairs: tuple[str, ...],
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    horizons: tuple[int, ...],
    include_entries: bool = True,
    include_exits: bool = True,
) -> dict[str, Any]:
    cache_dir = output_dir / "cache"
    caches = (
        {
            pair: pd.read_parquet(cache_dir / f"{pair_token(pair)}_sieve3_events_1h.parquet")
            for pair in pairs
        }
        if include_entries
        else {}
    )
    overall_start = min(start for start, _ in windows.values())
    overall_end = max(end for _, end in windows.values())
    frames = []
    for pair in pairs:
        actual = actual_frame(pair, overall_start, overall_end)
        if include_entries:
            cache = caches[pair].copy()
            cache["decision_time"] = pd.to_datetime(cache["decision_time"], utc=True, errors="coerce")
            actual = actual.merge(cache, on="decision_time", how="inner", validate="one_to_one")
        frames.append(actual)
    combined = pd.concat(frames, ignore_index=True)
    coverage = DataFrame()
    coverage_path = output_dir / "entry_event_coverage_summary.csv"
    events_long_path = cache_dir / "entry_events_long.parquet"
    if include_entries and events_long_path.exists():
        coverage = summarize_entry_coverage(
            pd.read_parquet(events_long_path),
            candidates,
            pairs=pairs,
            windows=windows,
        )
        coverage.to_csv(coverage_path, index=False)
    rng = np.random.default_rng(42)
    rows: list[dict[str, Any]] = []
    metadata = (
        {candidate.entry_id: candidate for candidate in candidates}
        if include_entries
        else {}
    )
    scope_index = 0
    for window_name, (start, end) in windows.items():
        window = combined[
            combined["decision_time"].ge(start) & combined["decision_time"].lt(end)
        ].copy()
        for entry_id, candidate in metadata.items():
            event_column = f"entry_onset__{entry_id}"
            if event_column not in window:
                continue
            for pair_scope in ("pooled", *pairs):
                scoped = window if pair_scope == "pooled" else window[window["pair"].eq(pair_scope)]
                event_mask = scoped[event_column].eq(1)
                events = scoped[event_mask].copy()
                if events.empty:
                    continue
                scope_index += 1
                print(
                    json.dumps(
                        {
                            "phase": "direct_entry_scope_start",
                            "scope_index": scope_index,
                            "window": window_name,
                            "entry_id": entry_id,
                            "pair_scope": pair_scope,
                            "event_rows": int(len(events)),
                        }
                    ),
                    flush=True,
                )
                controls = matched_controls(scoped, event_mask, rng=rng)
                print(
                    json.dumps(
                        {
                            "phase": "direct_entry_scope_matched",
                            "scope_index": scope_index,
                            "window": window_name,
                            "entry_id": entry_id,
                            "pair_scope": pair_scope,
                            "control_rows": int(len(controls)),
                        }
                    ),
                    flush=True,
                )
                shifted_keys = events[["pair", "decision_time"]].copy()
                shifted_keys["decision_time"] += pd.Timedelta(hours=168)
                placebo = shifted_keys.merge(
                    scoped,
                    on=["pair", "decision_time"],
                    how="inner",
                )
                placebo = placebo[placebo["summary__onset_count"].eq(0)]
                precursor_keys = events[["pair", "decision_time"]].copy()
                precursor_keys["decision_time"] -= pd.Timedelta(
                    hours=timeframe_hours(candidate.timeframe)
                )
                precursor = precursor_keys.merge(
                    scoped,
                    on=["pair", "decision_time"],
                    how="inner",
                )
                precursor = precursor[precursor[event_column].eq(0)]
                for horizon in horizons:
                    event_summary = summarize_sample(
                        events, side=candidate.side, horizon=horizon, sample_name="event"
                    )
                    control_summary = summarize_sample(
                        controls, side=candidate.side, horizon=horizon, sample_name="matched_control"
                    )
                    placebo_summary = summarize_sample(
                        placebo, side=candidate.side, horizon=horizon, sample_name="shifted_168h_placebo"
                    )
                    precursor_summary = summarize_sample(
                        precursor,
                        side=candidate.side,
                        horizon=horizon,
                        sample_name="native_timeframe_precursor",
                    )
                    base = {
                        "entry_id": entry_id,
                        "family": family_for_entry(entry_id),
                        "side": candidate.side,
                        "entry_timeframe": candidate.timeframe,
                        "window": window_name,
                        "pair_scope": pair_scope,
                        "horizon_hours": horizon,
                        "native_timeframe_hours": timeframe_hours(candidate.timeframe),
                        "native_horizon_multiple": horizon / timeframe_hours(candidate.timeframe),
                    }
                    merged = {**base, **{f"event_{k}": v for k, v in event_summary.items() if k != "sample"}}
                    merged.update({f"control_{k}": v for k, v in control_summary.items() if k != "sample"})
                    merged.update({f"placebo_{k}": v for k, v in placebo_summary.items() if k != "sample"})
                    merged.update({f"precursor_{k}": v for k, v in precursor_summary.items() if k != "sample"})
                    if not controls.empty:
                        match_distances = pd.to_numeric(
                            controls.get("control_match_distance"), errors="coerce"
                        )
                        time_separations = pd.to_numeric(
                            controls.get("control_match_time_separation_h"), errors="coerce"
                        )
                        merged["control_match_distance_mean"] = float(match_distances.mean())
                        merged["control_match_distance_median"] = float(match_distances.median())
                        merged["control_match_time_separation_h_mean"] = float(
                            time_separations.mean()
                        )
                        merged["control_match_time_separation_h_median"] = float(
                            time_separations.median()
                        )
                        merged["control_match_same_pair_quarter_rows"] = int(
                            controls["control_match_pool"].eq("same_pair_quarter").sum()
                        )
                        merged["control_match_same_pair_window_rows"] = int(
                            controls["control_match_pool"].eq("same_pair_window").sum()
                        )
                        merged["control_match_reused_rows"] = int(
                            controls["control_match_reused"].fillna(False).sum()
                        )
                        merged["control_match_reused_share"] = float(
                            controls["control_match_reused"].fillna(False).mean()
                        )
                        merged["control_unique_decision_times"] = int(
                            controls["decision_time"].nunique()
                        )
                    else:
                        merged["control_match_distance_mean"] = None
                        merged["control_match_distance_median"] = None
                        merged["control_match_time_separation_h_mean"] = None
                        merged["control_match_time_separation_h_median"] = None
                        merged["control_match_same_pair_quarter_rows"] = 0
                        merged["control_match_same_pair_window_rows"] = 0
                        merged["control_match_reused_rows"] = 0
                        merged["control_match_reused_share"] = None
                        merged["control_unique_decision_times"] = 0
                    for metric in (
                        "signed_return_mean",
                        "path_balance_atr_mean",
                        "reaction_magnitude_atr_mean",
                        "volume_ratio_mean",
                        "pressure_alignment_mean",
                        "volatility_ratio_mean",
                        "target_before_invalidation_mean",
                    ):
                        event_value = merged.get(f"event_{metric}")
                        control_value = merged.get(f"control_{metric}")
                        placebo_value = merged.get(f"placebo_{metric}")
                        precursor_value = merged.get(f"precursor_{metric}")
                        merged[f"event_minus_control_{metric}"] = (
                            float(event_value) - float(control_value)
                            if event_value is not None and control_value is not None
                            else None
                        )
                        merged[f"event_minus_placebo_{metric}"] = (
                            float(event_value) - float(placebo_value)
                            if event_value is not None and placebo_value is not None
                            else None
                        )
                        merged[f"event_minus_precursor_{metric}"] = (
                            float(event_value) - float(precursor_value)
                            if event_value is not None and precursor_value is not None
                            else None
                        )
                    rows.append(merged)
    entry_results = DataFrame(rows)
    entry_path = output_dir / "direct_entry_reaction_results.csv"
    entry_portability_path = output_dir / "direct_entry_portability_summary.csv"
    family_portability_path = output_dir / "direct_family_portability_summary.csv"
    if include_entries:
        entry_results.to_csv(entry_path, index=False)
        entry_portability, family_portability = summarize_direct_portability(entry_results)
        entry_portability.to_csv(entry_portability_path, index=False)
        family_portability.to_csv(family_portability_path, index=False)
        print(
            json.dumps(
                {
                    "phase": "direct_entry_outputs_written",
                    "scopes": scope_index,
                    "result_rows": int(len(entry_results)),
                }
            ),
            flush=True,
        )
    else:
        entry_results = pd.read_csv(entry_path) if entry_path.exists() else DataFrame()
        entry_portability = (
            pd.read_csv(entry_portability_path)
            if entry_portability_path.exists()
            else DataFrame()
        )
        family_portability = (
            pd.read_csv(family_portability_path)
            if family_portability_path.exists()
            else DataFrame()
        )
        coverage = pd.read_csv(coverage_path) if coverage_path.exists() else DataFrame()
        print(
            json.dumps(
                {
                    "phase": "direct_entry_outputs_preserved",
                    "result_rows": int(len(entry_results)),
                }
            ),
            flush=True,
        )

    exit_path = cache_dir / "exit_events_long.parquet"
    exit_rows: list[dict[str, Any]] = []
    exit_counterfactual_events = DataFrame()
    exit_counterfactual_summary = DataFrame()
    if include_exits and exit_path.exists():
        print(json.dumps({"phase": "direct_exit_analysis_started"}), flush=True)
        exits = pd.read_parquet(exit_path)
        exits["decision_time"] = pd.to_datetime(exits["decision_time"], utc=True, errors="coerce")
        actual_lookup = combined.drop(columns=[column for column in combined if column.startswith("entry_") or column.startswith("summary__")])
        exit_actual_lookup = actual_lookup.drop(columns=["decision_time"]).rename(
            columns={"date": "decision_time"}
        )
        merged_exits = exits.merge(
            exit_actual_lookup,
            on=["pair", "decision_time"],
            how="inner",
        )
        for (entry_id, exit_family, pair), group in merged_exits.groupby(
            ["entry_id", "exit_family", "pair"], dropna=False
        ):
            side = str(group["entry_side"].iloc[0])
            for horizon in horizons:
                summary = summarize_sample(group, side=side, horizon=horizon, sample_name="executed_exit")
                best = summary.get("best_move_if_held_atr_mean")
                worst = summary.get("worst_move_if_held_atr_mean")
                exit_rows.append(
                    {
                        "entry_id": entry_id,
                        "exit_family": exit_family,
                        "pair": pair,
                        "side": side,
                        "horizon_hours": horizon,
                        **{key: value for key, value in summary.items() if key != "sample"},
                        "exit_reversal_advantage_atr_mean": (
                            float(worst) - float(best)
                            if best is not None and worst is not None
                            else None
                        ),
                    }
                )
        exit_counterfactual_events, exit_counterfactual_summary = build_exit_counterfactuals(
            exits,
            actual_lookup,
            windows=windows,
            horizons=horizons,
        )
    exit_results_path = output_dir / "direct_exit_reaction_results.csv"
    exit_counterfactual_events_path = output_dir / "exit_counterfactual_events.parquet"
    exit_counterfactual_summary_path = output_dir / "exit_counterfactual_summary.csv"
    exit_results = DataFrame(exit_rows)
    if include_exits:
        exit_results.to_csv(exit_results_path, index=False)
        if not exit_counterfactual_events.empty:
            atomic_parquet(exit_counterfactual_events_path, exit_counterfactual_events)
        exit_counterfactual_summary.to_csv(exit_counterfactual_summary_path, index=False)
        print(
            json.dumps(
                {
                    "phase": "direct_exit_outputs_written",
                    "reaction_rows": int(len(exit_results)),
                    "counterfactual_rows": int(len(exit_counterfactual_summary)),
                }
            ),
            flush=True,
        )
    else:
        exit_results = (
            pd.read_csv(exit_results_path) if exit_results_path.exists() else DataFrame()
        )
        exit_counterfactual_events = (
            pd.read_parquet(exit_counterfactual_events_path)
            if exit_counterfactual_events_path.exists()
            else DataFrame()
        )
        exit_counterfactual_summary = (
            pd.read_csv(exit_counterfactual_summary_path)
            if exit_counterfactual_summary_path.exists()
            else DataFrame()
        )
        print(
            json.dumps(
                {
                    "phase": "direct_exit_outputs_preserved",
                    "reaction_rows": int(len(exit_results)),
                    "counterfactual_rows": int(len(exit_counterfactual_summary)),
                }
            ),
            flush=True,
        )
    exit_counterfactual_meta_path = output_dir / "exit_counterfactual_definitions.json"
    atomic_json(
        exit_counterfactual_meta_path,
        {
            "generated_at": now_iso(),
            "common_price_base": "executed close_rate; price timing is separated from partial-exit and fee accounting",
            "timestamp_rule": "close_date is joined to the OHLCV candle with that opening timestamp. Post-exit outcomes begin with the following complete candle, so no unknown remainder of the exit candle is treated as observable after-exit evidence.",
            "labels": {
                "earlier_exit_advantage_Nh": "Positive means exiting N hours earlier had a better price for the trade direction than the executed exit; negative means waiting to the executed exit improved the price.",
                "hold_instead_delta": "Signed difference between holding to the horizon and exiting when executed. Positive means holding ended better; negative means the exit avoided a worse terminal outcome.",
                "missed_additional_profit": "Best additional favorable price movement available after the exit within the horizon, floored at zero. It is opportunity, not proof the trade could capture the exact peak.",
                "avoided_loss_after_exit": "Worst adverse price movement after the exit within the horizon, expressed as a positive loss magnitude floored at zero. It shows downside the exit removed.",
                "net_exit_regret": "Missed additional profit minus avoided loss after exit. Positive means the exit gave up more favorable opportunity than adverse movement it avoided; negative means the avoided downside was larger. It is a compact balance, not a replacement for the two path components.",
                "best_post_exit_delta": "Best signed movement after exit, retained without flooring so paths wholly worse than the exit remain visible.",
                "worst_post_exit_delta": "Worst signed movement after exit, retained without flooring so paths wholly better than the exit remain visible.",
                "profit_ratio": "Freqtrade's realised trade result including the resolved exit path; it can include fees and partial exits and is not treated as a pure timing measure.",
                "exit_price_vs_exit_candle_close_delta": "Signed difference between the executed intrabar exit price and the exit candle's final close. It is a timestamp/fill diagnostic, not an after-exit outcome."
            },
            "examples": {
                "earlier_exit_advantage_4h": "A long exits at 100 after the market price was 103 four hours earlier: +3%, so the earlier exit had a 3% price advantage.",
                "hold_instead_delta": "A short exits at 100 and price is 96 at the horizon: +4%, so holding instead ended 4% better.",
                "missed_additional_profit": "A long exits at 100 and reaches 106 before the horizon: 6% additional favorable opportunity was missed.",
                "avoided_loss_after_exit": "A long exits at 100 and later falls to 92: the exit avoided up to 8% adverse movement.",
                "net_exit_regret": "If missed additional profit is 6% and avoided loss is 8%, net exit regret is -2%: the measured downside protection exceeded the forgone continuation by 2 percentage points.",
                "both_can_be_true": "After a long exits at 100, price first reaches 106 and later falls to 92. Missed additional profit is 6% and avoided loss is 8%; keep both components even though their net exit regret is -2%, because they describe different parts of the path."
            },
            "dependence_limit": "Exit solutions reuse entries, pairs, windows, and sometimes trade paths. Rows are not independent evidence; compare solution families and unique trade-path counts before interpreting consistency.",
            "event_rows": int(len(exit_counterfactual_events)),
            "summary_rows": int(len(exit_counterfactual_summary)),
            "events_path": str(exit_counterfactual_events_path),
            "summary_path": str(exit_counterfactual_summary_path),
        },
    )
    audit = {
        "generated_at": now_iso(),
        "entry_result_rows": int(len(entry_results)),
        "exit_result_rows": int(len(exit_results)),
        "entry_results": str(entry_path),
        "entry_coverage_rows": int(len(coverage)),
        "entry_coverage": str(coverage_path),
        "entry_portability_rows": int(len(entry_portability)),
        "family_portability_rows": int(len(family_portability)),
        "entry_portability": str(entry_portability_path),
        "family_portability": str(family_portability_path),
        "exit_results": str(exit_results_path),
        "exit_counterfactual_event_rows": int(len(exit_counterfactual_events)),
        "exit_counterfactual_summary_rows": int(len(exit_counterfactual_summary)),
        "exit_counterfactual_events": str(exit_counterfactual_events_path),
        "exit_counterfactual_summary": str(exit_counterfactual_summary_path),
        "exit_counterfactual_definitions": str(exit_counterfactual_meta_path),
        "pairs": list(pairs),
        "windows": {
            name: [start.isoformat(), end.isoformat()]
            for name, (start, end) in windows.items()
        },
        "horizons_hours": list(horizons),
        "controls": (
            "nearest visible-state matched ordinary candles within pair/quarter, "
            "168h shifted placebos, and one-native-timeframe precursor controls"
        ),
        "interpretation_limit": "association only; Sieve candidates were preselected by Hyperopt",
        "included_entry_analysis": bool(not entry_results.empty),
        "included_exit_analysis": bool(not exit_results.empty),
        "refreshed_entry_analysis": bool(include_entries),
        "refreshed_exit_analysis": bool(include_exits),
        "preserved_entry_analysis": bool(not include_entries and not entry_results.empty),
        "preserved_exit_analysis": bool(not include_exits and not exit_results.empty),
    }
    atomic_json(output_dir / "direct_baseline_audit.json", audit)
    append_log(
        research_log_path(output_dir),
        f"Direct baselines completed: {len(entry_results)} entry comparison rows, {len(exit_results)} open-trade-conditioned exit reaction rows, and {len(exit_counterfactual_summary)} exit counterfactual summaries across horizons {list(horizons)}. Controls are nearest visible-state matches, 168h shifted placebos, and native-timeframe precursors.",
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build timestamp-safe raw Sieve3 entry/exit event catalogues and direct reaction baselines."
    )
    parser.add_argument("--handover", type=Path, default=DEFAULT_HANDOVER)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timerange", default="20221201-20260629")
    parser.add_argument("--pairs", default=",".join(BASELINE_PAIRS))
    parser.add_argument("--windows-json", type=Path)
    parser.add_argument(
        "--direct-horizons",
        default=",".join(str(horizon) for horizon in DEFAULT_DIRECT_HORIZONS),
    )
    parser.add_argument("--entry-ids", default="")
    parser.add_argument("--max-entries", type=int)
    parser.add_argument("--skip-entries", action="store_true")
    parser.add_argument("--skip-exits", action="store_true")
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--skip-direct-entries", action="store_true")
    parser.add_argument("--skip-direct-exits", action="store_true")
    parser.add_argument("--diagnostic-stall-trace-seconds", type=int, default=0)
    args = parser.parse_args()
    if (
        args.diagnostic_stall_trace_seconds > 0
        and _EARLY_STALL_TRACE_SECONDS <= 0
    ):
        faulthandler.enable()
        faulthandler.dump_traceback_later(
            args.diagnostic_stall_trace_seconds,
            repeat=True,
        )

    pairs = parse_pairs(args.pairs)
    windows = parse_windows(args.windows_json)
    horizons = parse_horizons(args.direct_horizons)

    candidates = parse_handover(args.handover)
    if not candidates:
        raise ValueError(f"No entry candidates parsed from {args.handover}")
    selected_ids = {
        token.strip() for token in args.entry_ids.replace(";", ",").split(",") if token.strip()
    } or None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "catalogue_run_summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_started_at = now_iso()
    summary.update({
        "started_at": summary.get("started_at", run_started_at),
        "last_started_at": run_started_at,
        "handover": str(args.handover),
        "parsed_entries": len(candidates),
        "parsed_exit_solutions": sum(len(candidate.exits) for candidate in candidates),
        "timerange": args.timerange,
        "pairs": list(pairs),
        "windows": {
            name: [start.isoformat(), end.isoformat()]
            for name, (start, end) in windows.items()
        },
        "direct_horizons_hours": list(horizons),
    })
    if not args.skip_entries:
        summary["entries"] = build_entry_catalogue(
            candidates,
            config_path=args.config,
            output_dir=args.output_dir,
            timerange=args.timerange,
            pairs=pairs,
            selected_ids=selected_ids,
            max_entries=args.max_entries,
        )
    if not args.skip_exits:
        summary["exits"] = build_exit_catalogue(candidates, output_dir=args.output_dir)
    elif "exits" not in summary:
        exit_audit_path = args.output_dir / "exit_catalogue_audit.json"
        if exit_audit_path.exists():
            summary["exits"] = json.loads(exit_audit_path.read_text(encoding="utf-8"))
    if not args.skip_direct:
        summary["direct"] = run_direct_baselines(
            candidates,
            output_dir=args.output_dir,
            pairs=pairs,
            windows=windows,
            horizons=horizons,
            include_entries=not args.skip_direct_entries,
            include_exits=not args.skip_direct_exits,
        )
    summary["finished_at"] = now_iso()
    atomic_json(summary_path, summary)
    if args.diagnostic_stall_trace_seconds > 0 or _EARLY_STALL_TRACE_SECONDS > 0:
        faulthandler.cancel_dump_traceback_later()
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

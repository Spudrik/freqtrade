from __future__ import annotations

# Fix numerical-library thread counts before importing numpy/pandas.
# ruff: noqa: E402
import os


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import json
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    LARGE_ARTIFACT_ROOT,
    OUTPUT_ROOT,
    atomic_write_json,
    atomic_write_parquet,
    episode_start_mask,
    load_manifest,
    load_ohlcv,
    normalize_dates,
    numeric_array,
    ohlcv_path,
    prepare_base_market_frame,
    sha256_file,
    timeframe_delta,
    utc_now,
    validate_worker_count,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_localization import (  # noqa: E501
    pair_stem,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_anchored_vwap import (  # noqa: E501
    stable_json_sha256,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_localization import (  # noqa: E501
    eligible_period_events,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_lvn_attribution import (  # noqa: E501
    select_pairs,
)


FROZEN_BATCH = OUTPUT_ROOT / "generation2_review" / "g3_frozen_branch_batch.json"
REPORT_ROOT = OUTPUT_ROOT / "generation3_branches" / "g3f_generic_levels"
ARTIFACT_ROOT = LARGE_ARTIFACT_ROOT / "generation3_branches" / "g3f_generic_levels"
OUTPUT_SCHEMA_VERSION = 2
SOURCE_TIMEFRAMES = ("1h", "4h", "8h", "1d")
SMA_WINDOWS = (20, 50, 200)
EMA_WINDOWS = (12, 26, 50)
BOLLINGER_WINDOW = 20
ZONE_HALF_WIDTH_ATR = 0.10
ZONE_MINIMUM_PRICE_FRACTION = 0.0005
EPISODE_COOLDOWN_HOURS = 6
OUTCOME_EMBARGO_HOURS = 24
MIN_PAIR_PERIOD_EVENTS = 5
MIN_COHORT_PERIOD_EVENTS = 50
MIN_COHORT_PERIOD_COINS = 5
GENERIC_SCOPES = (
    "isolated_generic_level",
    "same_timeframe_generic_cluster",
    "cross_timeframe_generic_cluster",
)
GENERIC_RELATIONSHIPS = (
    "isolated_generic_level",
    "same_timeframe_same_family",
    "same_timeframe_cross_family",
    "cross_timeframe_same_family",
    "cross_timeframe_cross_family",
)
GENERIC_DEPENDENCY_RELATIONSHIPS = (
    "isolated_generic_level",
    "same_dependency_group",
    "cross_dependency_group",
)


@dataclass(frozen=True)
class GenericLevel:
    family: str
    name: str
    source_timeframe: str
    level: np.ndarray
    valid: np.ndarray
    source_available: Series
    source_open: Series
    source_age_hours: np.ndarray
    lookback_bars: np.ndarray

    @property
    def key(self) -> str:
        return f"{self.source_timeframe}|{self.family}|{self.name}"


@dataclass(frozen=True)
class PairTask:
    pair: str
    manifest_path: str
    run_id: str
    request_sha256: str
    overwrite: bool
    source_contracts: tuple[tuple[str, str], ...]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 3F outcome-blind coverage preflight for explicit calendar, "
            "SMA, EMA, and Bollinger price levels."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cohort", choices=("large", "meme"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    validate_worker_count(args.workers, manifest=manifest)
    validate_frozen_branch(args.cohort, manifest)
    pairs = select_pairs(manifest, args.pairs)
    contracts = source_contracts(pairs)
    request = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "frozen_batch_sha256": sha256_file(FROZEN_BATCH),
        "cohort": args.cohort,
        "pairs": list(pairs),
        "source_contracts": contracts,
        "base_timeframe": "1h",
        "level_families": {
            "previous_completed_week": ["high", "low", "midpoint"],
            "previous_completed_month": ["high", "low", "midpoint"],
            "simple_moving_average": list(SMA_WINDOWS),
            "exponential_moving_average": list(EMA_WINDOWS),
            "bollinger_20": ["upper", "middle", "lower"],
        },
        "indicator_source_timeframes": list(SOURCE_TIMEFRAMES),
        "causality": (
            "Only completed source candles are aligned by their availability timestamp. "
            "Previous week/month levels require a complete UTC calendar period and become "
            "available at the next period open."
        ),
        "zone": (
            f"max({ZONE_HALF_WIDTH_ATR:.2f} base ATR, "
            f"{ZONE_MINIMUM_PRICE_FRACTION:.4%} of level price)"
        ),
        "episode": (
            f"first contact after {EPISODE_COOLDOWN_HOURS} non-contact one-hour candles, "
            "or after the dynamic level moves more than its current half-width"
        ),
        "cluster_scopes": list(GENERIC_SCOPES),
        "cluster_relationships": list(GENERIC_RELATIONSHIPS),
        "cluster_dependency_relationships": list(GENERIC_DEPENDENCY_RELATIONSHIPS),
        "formula_alias_rule": (
            "SMA(20) and the Bollinger(20) middle line on the same source timeframe "
            "remain named separately as anchors but do not count as two cluster components."
        ),
        "coverage_gate": {
            "minimum_pair_period_events": MIN_PAIR_PERIOD_EVENTS,
            "minimum_cohort_period_events": MIN_COHORT_PERIOD_EVENTS,
            "minimum_coins": MIN_COHORT_PERIOD_COINS,
            "must_repeat_in_both_validation_periods": True,
        },
        "coverage_only": True,
        "reaction_outcomes_loaded": False,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    request_sha256 = stable_json_sha256(request)
    run_dir = REPORT_ROOT / args.run_id
    artifact_dir = ARTIFACT_ROOT / args.run_id
    pair_dir = artifact_dir / "pair_geometry"
    run_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    record_path = run_dir / "g3f_preflight_run_record.json"
    if record_path.is_file() and not args.overwrite:
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        if existing.get("request_sha256") != request_sha256:
            raise ValueError("Run ID already exists with an incompatible request contract.")
    record = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "status": "running",
        "started_at_utc": utc_now(),
        "orchestrator_pid": os.getpid(),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "request_sha256": request_sha256,
        "request_contract": request,
        "baseline": (
            "Earlier broad feature tests mixed generic indicators together. G3F asks "
            "whether each simple visible price reference has enough repeated contacts for "
            "an explicit localization test."
        ),
        "hypothesis": (
            "At least one named calendar, SMA, EMA, or Bollinger level has adequate "
            "multi-coin contact coverage in the same single/cluster scope in both "
            "validation periods."
        ),
        "pass_fail": (
            "Open outcomes only for an exact level/timeframe/scope cell with at least "
            "five coins, 50 episodes, and at least five episodes per contributing coin in "
            "both validations. Do not pool names or tune periods to rescue coverage."
        ),
        "workers": args.workers,
        "coverage_only": True,
        "reaction_outcomes_loaded": False,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(record, record_path)
    tasks = [
        PairTask(
            pair=pair,
            manifest_path=str(manifest_path),
            run_id=args.run_id,
            request_sha256=request_sha256,
            overwrite=args.overwrite,
            source_contracts=tuple(
                (str(row["path"]), str(row["sha256"])) for row in contracts if row["pair"] == pair
            ),
        )
        for pair in pairs
    ]
    try:
        results = run_tasks(tasks, workers=args.workers)
        inventory = DataFrame(results)
        atomic_write_parquet(inventory, run_dir / "g3f_pair_inventory.parquet")
        failures = inventory.loc[inventory["status"].eq("failed")]
        if not failures.empty:
            raise RuntimeError(f"{len(failures)} pair task(s) failed; inspect inventory.")
        geometry = combine_pair_outputs(
            pair_dir=pair_dir,
            pairs=pairs,
            request_sha256=request_sha256,
        )
        coverage = geometry_coverage(geometry, manifest=manifest)
        support = comparison_support(geometry, manifest=manifest)
        atomic_write_parquet(coverage, run_dir / "g3f_geometry_coverage.parquet")
        atomic_write_parquet(support, run_dir / "g3f_comparison_support.parquet")
        integrity = integrity_record(geometry=geometry, support=support, results=results)
        atomic_write_json(integrity, run_dir / "g3f_preflight_integrity.json")
        if not integrity["passed"]:
            raise RuntimeError("G3F preflight integrity validation failed.")
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "pair_tasks": results,
                "generic_level_series": int(inventory["generic_level_series"].sum()),
                "geometry_event_rows": len(geometry),
                "coverage_rows": len(coverage),
                "comparison_support_rows": len(support),
                "supported_both_validation_cells": supported_cell_count(support),
                "integrity": str(run_dir / "g3f_preflight_integrity.json"),
                "bulky_artifacts": str(artifact_dir),
            }
        )
        atomic_write_json(record, record_path)
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "failed_at_utc": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_write_json(record, record_path)
        raise
    return 0


def validate_frozen_branch(cohort: str, manifest: dict[str, Any]) -> None:
    frozen = json.loads(FROZEN_BATCH.read_text(encoding="utf-8"))
    branches = {branch["id"] for branch in frozen["branches"]}
    if frozen.get("status") != "frozen_before_generation3_reaction_outcomes":
        raise ValueError("The frozen Generation 3 batch is not ready.")
    if "g3f_explicit_generic_calculated_levels" not in branches:
        raise ValueError("The frozen Generation 3F branch is unavailable.")
    boundary = frozen["research_boundary"]
    if boundary.get("direction_prediction") is not False:
        raise ValueError("Generation 3F must keep direction disabled.")
    if boundary.get("profit_optimization") is not False:
        raise ValueError("Generation 3F must keep profit optimization disabled.")
    scope_key = "large_coin_scope" if cohort == "large" else "frozen_meme_scope"
    allowed = set(frozen["common_scope"][scope_key])
    requested = set(manifest["data"]["pairs"])
    if requested != allowed:
        raise ValueError(
            f"The {cohort} manifest does not match the frozen cohort: "
            f"missing={sorted(allowed - requested)}, extra={sorted(requested - allowed)}"
        )


def source_contracts(pairs: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        for timeframe in SOURCE_TIMEFRAMES:
            path = ohlcv_path(pair, timeframe)
            rows.append(
                {
                    "pair": pair,
                    "timeframe": timeframe,
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
            )
    return rows


def run_tasks(tasks: Sequence[PairTask], *, workers: int) -> list[dict[str, Any]]:
    if workers == 1:
        return [safe_build_pair(task) for task in tasks]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(safe_build_pair, task): task for task in tasks}
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda row: row["pair"])


def safe_build_pair(task: PairTask) -> dict[str, Any]:
    try:
        return build_pair(task)
    except Exception as exc:
        return {"pair": task.pair, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def build_pair(task: PairTask) -> dict[str, Any]:
    started = time.perf_counter()
    output_path = ARTIFACT_ROOT / task.run_id / "pair_geometry" / f"{pair_stem(task.pair)}.parquet"
    if output_path.is_file() and not task.overwrite:
        existing = validate_pair_output(
            output_path,
            pair=task.pair,
            request_sha256=task.request_sha256,
        )
        return {
            "pair": task.pair,
            "status": "existing",
            "generic_level_series": int(existing["level_key"].nunique()),
            "geometry_events": len(existing),
            "seconds": round(time.perf_counter() - started, 3),
        }
    for path_string, expected_sha256 in task.source_contracts:
        path = Path(path_string)
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"Frozen OHLCV source changed: {path}")
    manifest = load_manifest(Path(task.manifest_path))
    base = prepare_base_market_frame(task.pair, manifest)
    levels = explicit_generic_levels(pair=task.pair, base=base)
    geometry = contact_geometry(pair=task.pair, base=base, levels=levels)
    geometry = eligible_period_events(geometry, manifest, embargo_hours=OUTCOME_EMBARGO_HOURS)
    if geometry.empty:
        raise ValueError(f"No eligible G3F contact geometry for {task.pair}.")
    geometry["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    geometry["run_request_sha256"] = task.request_sha256
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(geometry, output_path)
    return {
        "pair": task.pair,
        "status": "completed",
        "source_rows": len(base),
        "generic_level_series": len(levels),
        "geometry_events": len(geometry),
        "seconds": round(time.perf_counter() - started, 3),
    }


def aligned_source_frame(base_dates: Series, source: DataFrame) -> DataFrame:
    left = DataFrame({"event_time": normalize_dates(base_dates)}).sort_values("event_time")
    right = source.copy()
    right["available_at"] = normalize_dates(right["available_at"])
    if "source_open" in right:
        right["source_open"] = normalize_dates(right["source_open"])
    right = right.sort_values("available_at", kind="stable").drop_duplicates(
        "available_at", keep="last"
    )
    return pd.merge_asof(
        left,
        right,
        left_on="event_time",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )


def indicator_source_levels(*, pair: str, base: DataFrame, timeframe: str) -> list[GenericLevel]:
    source = load_ohlcv(ohlcv_path(pair, timeframe)).sort_values("date").drop_duplicates("date")
    source = source.reset_index(drop=True)
    close = pd.to_numeric(source["close"], errors="coerce")
    calculated: dict[str, Series] = {}
    metadata: dict[str, tuple[str, int]] = {}
    for window in SMA_WINDOWS:
        name = f"sma_{window}"
        calculated[name] = close.rolling(window, min_periods=window).mean()
        metadata[name] = ("simple_moving_average", window)
    for window in EMA_WINDOWS:
        name = f"ema_{window}"
        calculated[name] = close.ewm(span=window, adjust=False, min_periods=window).mean()
        metadata[name] = ("exponential_moving_average", window)
    middle = close.rolling(BOLLINGER_WINDOW, min_periods=BOLLINGER_WINDOW).mean()
    deviation = close.rolling(BOLLINGER_WINDOW, min_periods=BOLLINGER_WINDOW).std(ddof=0)
    calculated["bollinger_20_upper"] = middle + 2.0 * deviation
    calculated["bollinger_20_middle"] = middle
    calculated["bollinger_20_lower"] = middle - 2.0 * deviation
    for name in ("bollinger_20_upper", "bollinger_20_middle", "bollinger_20_lower"):
        metadata[name] = ("bollinger_band", BOLLINGER_WINDOW)
    source_frame = DataFrame(
        {
            "source_open": normalize_dates(source["date"]),
            "available_at": normalize_dates(source["date"]) + timeframe_delta(timeframe),
            **calculated,
        }
    )
    aligned = aligned_source_frame(base["date"], source_frame)
    event_time = normalize_dates(aligned["event_time"])
    available = normalize_dates(aligned["available_at"])
    age = (event_time - available).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    levels: list[GenericLevel] = []
    for name, (family, lookback) in metadata.items():
        values = pd.to_numeric(aligned[name], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values) & (values > 0.0) & available.notna().to_numpy()
        levels.append(
            GenericLevel(
                family=family,
                name=name,
                source_timeframe=timeframe,
                level=values,
                valid=valid,
                source_available=available.copy(),
                source_open=normalize_dates(aligned["source_open"]),
                source_age_hours=age.copy(),
                lookback_bars=np.full(len(base), lookback, dtype=np.int16),
            )
        )
    return levels


def complete_calendar_periods(source: DataFrame, *, period: str) -> DataFrame:
    dates = normalize_dates(source["date"])
    if period == "week":
        starts = dates.dt.floor("D") - pd.to_timedelta(dates.dt.weekday, unit="D")
        available = starts + pd.Timedelta(days=7)
        family = "previous_completed_week"
    elif period == "month":
        naive = dates.dt.tz_localize(None)
        starts = naive.dt.to_period("M").dt.start_time.dt.tz_localize("UTC")
        available = (naive.dt.to_period("M") + 1).dt.start_time.dt.tz_localize("UTC")
        family = "previous_completed_month"
    else:
        raise ValueError(f"Unsupported calendar period: {period}")
    frame = source.copy()
    frame["period_start"] = starts
    frame["available_at"] = available
    grouped = (
        frame.groupby(["period_start", "available_at"], observed=True)
        .agg(
            period_high=("high", "max"),
            period_low=("low", "min"),
            candle_count=("date", "size"),
            first_open=("date", "min"),
            last_open=("date", "max"),
        )
        .reset_index()
    )
    expected = (
        (grouped["available_at"] - grouped["period_start"]).dt.total_seconds() / 3600.0
    ).astype(int)
    complete = (
        grouped["candle_count"].eq(expected)
        & normalize_dates(grouped["first_open"]).eq(normalize_dates(grouped["period_start"]))
        & (normalize_dates(grouped["last_open"]) + pd.Timedelta(hours=1)).eq(
            normalize_dates(grouped["available_at"])
        )
    )
    grouped = grouped.loc[complete].reset_index(drop=True)
    grouped["period_midpoint"] = (grouped["period_high"] + grouped["period_low"]) / 2.0
    grouped["family"] = family
    grouped["lookback_bars"] = expected.loc[complete].to_numpy(dtype=np.int16)
    return grouped


def calendar_levels(*, pair: str, base: DataFrame) -> list[GenericLevel]:
    source = load_ohlcv(ohlcv_path(pair, "1h")).sort_values("date").drop_duplicates("date")
    source = source.reset_index(drop=True)
    output: list[GenericLevel] = []
    for period, timeframe in (("week", "1w"), ("month", "1M")):
        completed = complete_calendar_periods(source, period=period)
        source_frame = completed[
            [
                "period_start",
                "available_at",
                "period_high",
                "period_low",
                "period_midpoint",
                "lookback_bars",
                "family",
            ]
        ].rename(columns={"period_start": "source_open"})
        aligned = aligned_source_frame(base["date"], source_frame)
        event_time = normalize_dates(aligned["event_time"])
        available = normalize_dates(aligned["available_at"])
        age = (event_time - available).dt.total_seconds().to_numpy(dtype=float) / 3600.0
        family_values = aligned["family"].dropna().astype(str).unique()
        if len(family_values) != 1:
            raise ValueError(f"Calendar alignment lost its family for {pair} {period}.")
        for column, suffix in (
            ("period_high", "high"),
            ("period_low", "low"),
            ("period_midpoint", "midpoint"),
        ):
            values = pd.to_numeric(aligned[column], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(values) & (values > 0.0) & available.notna().to_numpy()
            output.append(
                GenericLevel(
                    family=family_values[0],
                    name=f"{family_values[0]}_{suffix}",
                    source_timeframe=timeframe,
                    level=values,
                    valid=valid,
                    source_available=available.copy(),
                    source_open=normalize_dates(aligned["source_open"]),
                    source_age_hours=age.copy(),
                    lookback_bars=pd.to_numeric(aligned["lookback_bars"], errors="coerce")
                    .fillna(0)
                    .to_numpy(dtype=np.int16),
                )
            )
    return output


def explicit_generic_levels(*, pair: str, base: DataFrame) -> list[GenericLevel]:
    levels = calendar_levels(pair=pair, base=base)
    for timeframe in SOURCE_TIMEFRAMES:
        levels.extend(indicator_source_levels(pair=pair, base=base, timeframe=timeframe))
    keys = [level.key for level in levels]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate explicit generic level keys for {pair}.")
    return levels


def classify_scope(other_count: int, cross_timeframe_count: int) -> str:
    if cross_timeframe_count > 0:
        return "cross_timeframe_generic_cluster"
    if other_count > 0:
        return "same_timeframe_generic_cluster"
    return "isolated_generic_level"


def classify_relationship(
    *,
    other_count: int,
    cross_timeframe_count: int,
    cross_family_count: int,
) -> str:
    if other_count == 0:
        return "isolated_generic_level"
    if cross_timeframe_count > 0 and cross_family_count > 0:
        return "cross_timeframe_cross_family"
    if cross_timeframe_count > 0:
        return "cross_timeframe_same_family"
    if cross_family_count > 0:
        return "same_timeframe_cross_family"
    return "same_timeframe_same_family"


def generic_dependency_group(item: GenericLevel) -> str:
    if item.family in {
        "simple_moving_average",
        "exponential_moving_average",
        "bollinger_band",
    }:
        return "price_average_family"
    if item.family in {"previous_completed_week", "previous_completed_month"}:
        return "prior_calendar_structure"
    return item.family


def formula_alias_group(item: GenericLevel) -> str | None:
    if item.name in {"sma_20", "bollinger_20_middle"}:
        return f"{item.source_timeframe}|sma20_bollinger20_middle"
    return None


def classify_dependency_relationship(
    *,
    other_count: int,
    cross_dependency_group_count: int,
) -> str:
    if other_count == 0:
        return "isolated_generic_level"
    if cross_dependency_group_count > 0:
        return "cross_dependency_group"
    return "same_dependency_group"


def contact_geometry(*, pair: str, base: DataFrame, levels: Sequence[GenericLevel]) -> DataFrame:
    base_atr = numeric_array(base["base_atr"])
    high = numeric_array(base["high"])
    low = numeric_array(base["low"])
    pre_close = numeric_array(base["pre_close"])
    dates = normalize_dates(base["date"])
    level_matrix = np.column_stack([level.level for level in levels])
    valid_matrix = np.column_stack([level.valid for level in levels])
    width_matrix = np.maximum(
        ZONE_HALF_WIDTH_ATR * base_atr[:, None],
        np.abs(level_matrix) * ZONE_MINIMUM_PRICE_FRACTION,
    )
    keys = np.asarray([level.key for level in levels], dtype=object)
    timeframes = np.asarray([level.source_timeframe for level in levels], dtype=object)
    families = np.asarray([level.family for level in levels], dtype=object)
    dependencies = np.asarray([generic_dependency_group(level) for level in levels], dtype=object)
    alias_groups = np.asarray([formula_alias_group(level) or "" for level in levels], dtype=object)
    rows: list[dict[str, Any]] = []
    for source_index, item in enumerate(levels):
        values = item.level
        widths = width_matrix[:, source_index]
        valid = item.valid & np.isfinite(widths) & (widths > 0.0)
        contact = valid & (high >= values - widths) & (low <= values + widths)
        starts = episode_start_mask(
            contact,
            values,
            widths,
            cooldown=EPISODE_COOLDOWN_HOURS,
        )
        starts[max(len(starts) - OUTCOME_EMBARGO_HOURS, 0) :] = False
        indexes = np.flatnonzero(starts)
        for index in indexes:
            local_level = level_matrix[index]
            local_width = width_matrix[index]
            overlap = (
                valid_matrix[index]
                & np.isfinite(local_level)
                & np.isfinite(local_width)
                & (np.abs(local_level - values[index]) <= local_width + widths[index])
            )
            overlap[source_index] = False
            alias_overlap = np.zeros(len(levels), dtype=bool)
            if alias_groups[source_index]:
                alias_overlap = overlap & (alias_groups == alias_groups[source_index])
                overlap[alias_overlap] = False
            alias_indexes = np.flatnonzero(alias_overlap)
            overlap_indexes = np.flatnonzero(overlap)
            other_count = len(overlap_indexes)
            cross_timeframe = int((timeframes[overlap_indexes] != item.source_timeframe).sum())
            same_timeframe = other_count - cross_timeframe
            cross_family = int((families[overlap_indexes] != item.family).sum())
            same_family = other_count - cross_family
            cross_dependency = int(
                (dependencies[overlap_indexes] != dependencies[source_index]).sum()
            )
            lower_bound = values[index] - widths[index]
            upper_bound = values[index] + widths[index]
            if pre_close[index] < lower_bound:
                approach = "from_below"
            elif pre_close[index] > upper_bound:
                approach = "from_above"
            else:
                approach = "already_inside_or_unclear"
            rows.append(
                {
                    "pair": pair,
                    "period": base["period"].iloc[index],
                    "event_time": dates.iloc[index],
                    "base_index": int(index),
                    "level_family": item.family,
                    "level_name": item.name,
                    "level_key": item.key,
                    "source_timeframe": item.source_timeframe,
                    "generic_scope": classify_scope(other_count, cross_timeframe),
                    "cluster_relationship": classify_relationship(
                        other_count=other_count,
                        cross_timeframe_count=cross_timeframe,
                        cross_family_count=cross_family,
                    ),
                    "cluster_dependency_relationship": classify_dependency_relationship(
                        other_count=other_count,
                        cross_dependency_group_count=cross_dependency,
                    ),
                    "approach_state": approach,
                    "level_price": float(values[index]),
                    "zone_half_width": float(widths[index]),
                    "zone_half_width_atr": float(widths[index] / base_atr[index]),
                    "pre_distance_atr": float(
                        abs(values[index] - pre_close[index]) / base_atr[index]
                    ),
                    "source_available_at": item.source_available.iloc[index],
                    "source_open": item.source_open.iloc[index],
                    "source_age_hours": float(item.source_age_hours[index]),
                    "source_lookback_bars": int(item.lookback_bars[index]),
                    "overlap_other_generic_level_count": other_count,
                    "overlap_same_timeframe_level_count": same_timeframe,
                    "overlap_cross_timeframe_level_count": cross_timeframe,
                    "overlap_same_family_level_count": same_family,
                    "overlap_cross_family_level_count": cross_family,
                    "overlap_cross_dependency_group_level_count": cross_dependency,
                    "overlap_level_keys": ";".join(keys[overlap_indexes].astype(str)),
                    "overlap_level_names": ";".join(
                        sorted({levels[position].name for position in overlap_indexes})
                    ),
                    "overlap_level_families": ";".join(
                        sorted({levels[position].family for position in overlap_indexes})
                    ),
                    "overlap_level_timeframes": ";".join(
                        sorted({levels[position].source_timeframe for position in overlap_indexes})
                    ),
                    "overlap_dependency_groups": ";".join(
                        sorted({dependencies[position] for position in overlap_indexes})
                    ),
                    "mathematical_alias_level_count": len(alias_indexes),
                    "mathematical_alias_level_keys": ";".join(keys[alias_indexes].astype(str)),
                    "reaction_outcomes_loaded": False,
                    "direction_prediction": False,
                    "profit_optimization": False,
                }
            )
    return DataFrame(rows)


def period_roles(manifest: dict[str, Any]) -> dict[str, str]:
    return {
        str(period["id"]): str(period["role"])
        for period in manifest["data"]["chronological_periods"]
    }


def geometry_coverage(geometry: DataFrame, *, manifest: dict[str, Any]) -> DataFrame:
    output = (
        geometry.groupby(
            [
                "pair",
                "period",
                "level_family",
                "level_name",
                "source_timeframe",
                "generic_scope",
                "cluster_relationship",
                "cluster_dependency_relationship",
            ],
            observed=True,
        )
        .agg(
            contact_events=("event_time", "size"),
            unique_event_days=("event_time", lambda values: values.dt.floor("1D").nunique()),
            median_source_age_hours=("source_age_hours", "median"),
            median_zone_half_width_atr=("zone_half_width_atr", "median"),
            median_other_level_count=("overlap_other_generic_level_count", "median"),
            median_cross_timeframe_count=("overlap_cross_timeframe_level_count", "median"),
        )
        .reset_index()
    )
    output["pair_period_eligible"] = output["contact_events"].ge(MIN_PAIR_PERIOD_EVENTS)
    output["period_role"] = output["period"].map(period_roles(manifest))
    output["direction_prediction"] = False
    output["profit_optimization"] = False
    return output


def comparison_support(geometry: DataFrame, *, manifest: dict[str, Any]) -> DataFrame:
    coverage = geometry_coverage(geometry, manifest=manifest)
    eligible = coverage.loc[coverage["pair_period_eligible"]].copy()
    cell_columns = [
        "period",
        "level_family",
        "level_name",
        "source_timeframe",
        "generic_scope",
        "cluster_relationship",
        "cluster_dependency_relationship",
    ]
    output = (
        eligible.groupby(cell_columns, observed=True)
        .agg(
            eligible_contact_events=("contact_events", "sum"),
            eligible_coins=("pair", "nunique"),
            eligible_coin_names=("pair", lambda values: ";".join(sorted(set(values)))),
        )
        .reset_index()
    )
    output["period_role"] = output["period"].map(period_roles(manifest))
    output["period_cell_supported"] = output["eligible_contact_events"].ge(
        MIN_COHORT_PERIOD_EVENTS
    ) & output["eligible_coins"].ge(MIN_COHORT_PERIOD_COINS)
    validation = output["period_role"].eq("chronological_internal_validation")
    identity = [
        "level_family",
        "level_name",
        "source_timeframe",
        "generic_scope",
        "cluster_relationship",
        "cluster_dependency_relationship",
    ]
    repeated = (
        output.loc[validation]
        .groupby(identity, observed=True)
        .agg(
            validation_periods=("period", "nunique"),
            supported_validation_periods=("period_cell_supported", "sum"),
        )
        .reset_index()
    )
    repeated["both_validation_periods_supported"] = repeated["validation_periods"].eq(2) & repeated[
        "supported_validation_periods"
    ].eq(2)
    output = output.merge(repeated, on=identity, how="left", validate="many_to_one")
    output["both_validation_periods_supported"] = output[
        "both_validation_periods_supported"
    ].fillna(False)
    output["direction_prediction"] = False
    output["profit_optimization"] = False
    return output


def supported_cell_count(support: DataFrame) -> int:
    identity = [
        "level_family",
        "level_name",
        "source_timeframe",
        "generic_scope",
        "cluster_relationship",
        "cluster_dependency_relationship",
    ]
    return int(
        support.loc[support["both_validation_periods_supported"], identity]
        .drop_duplicates()
        .shape[0]
    )


def integrity_record(
    *, geometry: DataFrame, support: DataFrame, results: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    failures = [row for row in results if row["status"] == "failed"]
    event_time = pd.to_datetime(geometry["event_time"], utc=True, errors="coerce")
    available = pd.to_datetime(geometry["source_available_at"], utc=True, errors="coerce")
    future_source_violations = int((available > event_time).sum())
    invalid_scopes = sorted(set(geometry["generic_scope"].astype(str)) - set(GENERIC_SCOPES))
    invalid_relationships = sorted(
        set(geometry["cluster_relationship"].astype(str)) - set(GENERIC_RELATIONSHIPS)
    )
    invalid_dependency_relationships = sorted(
        set(geometry["cluster_dependency_relationship"].astype(str))
        - set(GENERIC_DEPENDENCY_RELATIONSHIPS)
    )
    direction_violations = int(geometry["direction_prediction"].ne(False).sum())
    profit_violations = int(geometry["profit_optimization"].ne(False).sum())
    outcome_violations = int(geometry["reaction_outcomes_loaded"].ne(False).sum())
    duplicate_rows = int(geometry.duplicated(["pair", "level_key", "event_time"]).sum())
    return {
        "created_at_utc": utc_now(),
        "passed": bool(
            not failures
            and not geometry.empty
            and future_source_violations == 0
            and not invalid_scopes
            and not invalid_relationships
            and not invalid_dependency_relationships
            and direction_violations == 0
            and profit_violations == 0
            and outcome_violations == 0
            and duplicate_rows == 0
        ),
        "failures": failures,
        "geometry_event_rows": len(geometry),
        "level_families": sorted(geometry["level_family"].astype(str).unique()),
        "source_timeframes": sorted(geometry["source_timeframe"].astype(str).unique()),
        "generic_scopes": sorted(geometry["generic_scope"].astype(str).unique()),
        "future_source_violations": future_source_violations,
        "invalid_scopes": invalid_scopes,
        "cluster_relationships": sorted(geometry["cluster_relationship"].astype(str).unique()),
        "invalid_cluster_relationships": invalid_relationships,
        "cluster_dependency_relationships": sorted(
            geometry["cluster_dependency_relationship"].astype(str).unique()
        ),
        "invalid_cluster_dependency_relationships": invalid_dependency_relationships,
        "mathematical_alias_event_rows": int(
            geometry["mathematical_alias_level_count"].gt(0).sum()
        ),
        "duplicate_pair_level_event_rows": duplicate_rows,
        "reaction_outcome_violations": outcome_violations,
        "direction_prediction_violations": direction_violations,
        "profit_optimization_violations": profit_violations,
        "supported_both_validation_cells": supported_cell_count(support),
        "coverage_only": True,
        "reaction_outcomes_loaded": False,
        "direction_prediction": False,
        "profit_optimization": False,
    }


def validate_pair_output(path: Path, *, pair: str, request_sha256: str) -> DataFrame:
    frame = pd.read_parquet(path)
    if frame.empty or set(frame["pair"].astype(str)) != {pair}:
        raise ValueError(f"Existing output has an incompatible pair: {path}")
    if set(frame["output_schema_version"].astype(int)) != {OUTPUT_SCHEMA_VERSION}:
        raise ValueError(f"Existing output has an incompatible schema: {path}")
    if set(frame["run_request_sha256"].astype(str)) != {request_sha256}:
        raise ValueError(f"Existing output belongs to another request: {path}")
    return frame


def combine_pair_outputs(*, pair_dir: Path, pairs: Sequence[str], request_sha256: str) -> DataFrame:
    return pd.concat(
        [
            validate_pair_output(
                pair_dir / f"{pair_stem(pair)}.parquet",
                pair=pair,
                request_sha256=request_sha256,
            )
            for pair in pairs
        ],
        ignore_index=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())

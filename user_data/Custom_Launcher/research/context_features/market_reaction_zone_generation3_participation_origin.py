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
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    LARGE_ARTIFACT_ROOT,
    OUTPUT_ROOT,
    atomic_write_json,
    atomic_write_parquet,
    load_manifest,
    normalize_dates,
    numeric_array,
    prepare_base_market_frame,
    sha256_file,
    utc_now,
    validate_worker_count,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_localization import (  # noqa: E501
    pair_stem,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_anchored_vwap import (  # noqa: E501
    stable_json_sha256,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_density_reaction import (  # noqa: E501
    attach_structure_overlaps,
    build_density_surfaces,
    density_matrices,
    reference_matrices,
    selected_reference_levels,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_localization import (  # noqa: E501
    eligible_period_events,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_lvn_attribution import (  # noqa: E501
    select_pairs,
    source_contracts,
)


FROZEN_BATCH = OUTPUT_ROOT / "generation2_review" / "g3_frozen_branch_batch.json"
REPORT_ROOT = OUTPUT_ROOT / "generation3_branches" / "g3e_participation_origin"
ARTIFACT_ROOT = LARGE_ARTIFACT_ROOT / "generation3_branches" / "g3e_participation_origin"
OUTPUT_SCHEMA_VERSION = 1
ORIGIN_LOOKBACK_HOURS = 168
ORIGIN_MINIMUM_HISTORY = 134
ORIGIN_MEMORY_HOURS = 168
OUTSIDE_INTERVAL_HOURS = 6
OUTCOME_EMBARGO_HOURS = 24
VOLUME_QUANTILE = 0.90
RANGE_QUANTILE = 0.75
CONTEXT_QUANTILE = 0.75
MIN_ZONE_HALF_WIDTH_ATR = 0.10
MAX_ZONE_HALF_WIDTH_ATR = 0.50
NEAR_MISS_WIDTH_MULTIPLIER = 2.0
MIN_COHORT_PERIOD_EVENTS = 50
MIN_COHORT_PERIOD_COINS = 5
EVENT_KINDS = ("contact", "near_miss")
ORIGIN_SCOPES = (
    "isolated_origin",
    "origin_plus_density",
    "origin_plus_reference",
    "origin_plus_density_and_reference",
)


@dataclass(frozen=True)
class PairTask:
    pair: str
    manifest_path: str
    run_id: str
    request_sha256: str
    overwrite: bool
    cache_contracts: tuple[tuple[str, str], ...]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 3E coverage-only construction of causal participation-change "
            "origin zones and later revisit/near-miss events."
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
    contracts = source_contracts(
        cohort=args.cohort,
        manifest=manifest,
        manifest_path=manifest_path,
        pairs=pairs,
    )
    request = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "frozen_batch_sha256": sha256_file(FROZEN_BATCH),
        "cohort": args.cohort,
        "pairs": list(pairs),
        "reference_cache_contracts": contracts,
        "construction": {
            "base_timeframe": "1h",
            "prior_distribution_lookback_hours": ORIGIN_LOOKBACK_HOURS,
            "minimum_prior_candles": ORIGIN_MINIMUM_HISTORY,
            "abrupt_volume_condition": (
                f"completed-candle volume >= prior-only rolling {VOLUME_QUANTILE:.0%} quantile"
            ),
            "abrupt_range_condition": (
                "completed-candle high-low range divided by prior ATR >= prior-only rolling "
                f"{RANGE_QUANTILE:.0%} quantile"
            ),
            "pressure_and_volatility_context": (
                f"record whether absolute body pressure and prior ATR exceed their prior "
                f"{CONTEXT_QUANTILE:.0%} quantiles; do not use them to select the origin"
            ),
            "zone_centre": "midpoint of the completed candle real body",
            "zone_half_width": (
                "half the completed real body, bounded between "
                f"{MIN_ZONE_HALF_WIDTH_ATR:.2f} and {MAX_ZONE_HALF_WIDTH_ATR:.2f} prior ATR"
            ),
            "available_from": "next one-hour candle after the origin candle completes",
            "memory_hours": ORIGIN_MEMORY_HOURS,
            "revisit_rule": (
                f"price must first spend {OUTSIDE_INTERVAL_HOURS} complete candles outside; "
                "the first later encounter is a contact or clean two-width near miss"
            ),
        },
        "geometry_scopes": list(ORIGIN_SCOPES),
        "existing_level_context": (
            "causal density zones plus aligned 1h/4h/8h/1d VP nodes, prior highs/lows, "
            "Bollinger boundaries, round numbers, POC and prior POC"
        ),
        "coverage_gate": {
            "minimum_events_per_kind_scope_period": MIN_COHORT_PERIOD_EVENTS,
            "minimum_coins_per_kind_scope_period": MIN_COHORT_PERIOD_COINS,
            "contact_and_near_miss_must_share_scope": True,
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
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    record_path = run_dir / "g3e_preflight_run_record.json"
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
            "Calendar and swing anchored VWAP did not localize reaction. G3D instead left "
            "a participation/fresh-arrival clue, motivating a distinct causal origin level."
        ),
        "hypothesis": (
            "A bounded price area where an abrupt completed-candle volume-and-range change "
            "began has enough later revisits and clean near misses for a fair reaction test."
        ),
        "pass_fail": (
            "Open outcomes only for an origin scope with at least 50 contact revisits and "
            "50 clean near misses from at least five coins in both validation periods. "
            "Do not tune the fixed construction to rescue sparse coverage."
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
            cache_contracts=tuple(
                (str(row["cache_path"]), str(row["current_cache_sha256"]))
                for row in contracts
                if row["pair"] == pair
            ),
        )
        for pair in pairs
    ]
    try:
        results = run_tasks(tasks, workers=args.workers)
        inventory = DataFrame(results)
        atomic_write_parquet(inventory, run_dir / "g3e_pair_inventory.parquet")
        failures = inventory.loc[inventory["status"].eq("failed")]
        if not failures.empty:
            raise RuntimeError(
                f"{len(failures)} pair task(s) failed; inspect g3e_pair_inventory.parquet."
            )
        geometry = combine_pair_outputs(
            directory=artifact_dir / "pair_geometry",
            pairs=pairs,
            request_sha256=request_sha256,
        )
        coverage = geometry_coverage(geometry, manifest=manifest)
        support = comparison_support(geometry, manifest=manifest)
        atomic_write_parquet(coverage, run_dir / "g3e_geometry_coverage.parquet")
        atomic_write_parquet(support, run_dir / "g3e_comparison_support.parquet")
        integrity = integrity_record(geometry=geometry, support=support, results=results)
        atomic_write_json(integrity, run_dir / "g3e_preflight_integrity.json")
        if not integrity["passed"]:
            raise RuntimeError("G3E preflight integrity validation failed.")
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "pair_tasks": results,
                "origin_candidates": int(inventory["origin_candidates"].sum()),
                "geometry_event_rows": len(geometry),
                "coverage_rows": len(coverage),
                "comparison_support_rows": len(support),
                "supported_both_validation_scopes": supported_scope_count(support),
                "integrity": str(run_dir / "g3e_preflight_integrity.json"),
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
    if "g3e_causal_participation_change_origin_zone" not in branches:
        raise ValueError("The frozen Generation 3E branch is unavailable.")
    boundary = frozen["research_boundary"]
    if boundary.get("direction_prediction") is not False:
        raise ValueError("Generation 3E must keep direction disabled.")
    if boundary.get("profit_optimization") is not False:
        raise ValueError("Generation 3E must keep profit optimization disabled.")
    scope_key = "large_coin_scope" if cohort == "large" else "frozen_meme_scope"
    allowed = set(frozen["common_scope"][scope_key])
    requested = set(manifest["data"]["pairs"])
    if requested != allowed:
        raise ValueError(
            f"The {cohort} manifest does not match the frozen cohort: "
            f"missing={sorted(allowed - requested)}, extra={sorted(requested - allowed)}"
        )


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
            "origin_candidates": int(existing["origin_index"].nunique()),
            "geometry_events": len(existing),
            "seconds": round(time.perf_counter() - started, 3),
        }
    for cache_path, expected_sha256 in task.cache_contracts:
        if sha256_file(Path(cache_path)) != expected_sha256:
            raise ValueError(f"Frozen reference cache changed before pair build: {cache_path}")
    manifest_path = Path(task.manifest_path)
    manifest = load_manifest(manifest_path)
    base = prepare_base_market_frame(task.pair, manifest)
    origins = participation_origins(base)
    events = origin_revisit_events(pair=task.pair, base=base, origins=origins)
    events = eligible_period_events(events, manifest, embargo_hours=OUTCOME_EMBARGO_HOURS)
    if events.empty:
        raise ValueError(f"No eligible G3E revisit geometry for {task.pair}.")
    surfaces = build_density_surfaces(base)
    references = selected_reference_levels(
        pair=task.pair,
        base=base,
        manifest_path=manifest_path,
    )
    density_levels, density_widths, density_surface_keys = density_matrices(surfaces)
    reference_levels, reference_widths, reference_groups = reference_matrices(
        references,
        base_atr=numeric_array(base["base_atr"]),
    )
    events = attach_structure_overlaps(
        events,
        source_surface_key=None,
        density_levels=density_levels,
        density_widths=density_widths,
        density_surface_keys=density_surface_keys,
        reference_levels=reference_levels,
        reference_widths=reference_widths,
        reference_groups=reference_groups,
    )
    events["origin_scope"] = classify_origin_scope(events)
    keep_columns = [
        "pair",
        "period",
        "event_time",
        "base_index",
        "event_kind",
        "approach_state",
        "origin_scope",
        "origin_index",
        "origin_time",
        "origin_available_at",
        "origin_age_hours",
        "level_price",
        "zone_half_width",
        "zone_half_width_atr",
        "base_atr",
        "pre_distance_atr",
        "origin_volume_vs_prior_median",
        "origin_volume_vs_q90",
        "origin_range_atr",
        "origin_range_vs_q75",
        "origin_abs_pressure",
        "origin_pressure_above_q75",
        "origin_atr_above_q75",
        "origin_impulse_sign",
        "overlap_density_zone_count",
        "overlap_other_density_surface_count",
        "overlap_reference_level_count",
        "overlap_reference_group_count",
        "direction_prediction",
        "profit_optimization",
    ]
    events = events.loc[:, keep_columns].copy()
    events["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    events["run_request_sha256"] = task.request_sha256
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(events, output_path)
    return {
        "pair": task.pair,
        "status": "completed",
        "source_rows": len(base),
        "origin_candidates": len(origins),
        "geometry_events": len(events),
        "contact_events": int(events["event_kind"].eq("contact").sum()),
        "near_miss_events": int(events["event_kind"].eq("near_miss").sum()),
        "seconds": round(time.perf_counter() - started, 3),
    }


def prior_rolling_quantile(
    values: np.ndarray,
    quantile: float,
) -> np.ndarray:
    return (
        pd.Series(values)
        .shift(1)
        .rolling(ORIGIN_LOOKBACK_HOURS, min_periods=ORIGIN_MINIMUM_HISTORY)
        .quantile(quantile)
        .to_numpy(dtype=float)
    )


def participation_origins(base: DataFrame) -> DataFrame:
    open_price = numeric_array(base["open"])
    high = numeric_array(base["high"])
    low = numeric_array(base["low"])
    close = numeric_array(base["close"])
    volume = numeric_array(base["volume"])
    atr = numeric_array(base["base_atr"])
    candle_range = high - low
    range_atr = np.divide(
        candle_range,
        atr,
        out=np.full(len(base), np.nan),
        where=np.isfinite(atr) & (atr > 0.0),
    )
    abs_pressure = np.divide(
        np.abs(close - open_price),
        candle_range,
        out=np.zeros(len(base)),
        where=np.isfinite(candle_range) & (candle_range > 0.0),
    )
    volume_q90 = prior_rolling_quantile(volume, VOLUME_QUANTILE)
    volume_median = prior_rolling_quantile(volume, 0.50)
    range_q75 = prior_rolling_quantile(range_atr, RANGE_QUANTILE)
    pressure_q75 = prior_rolling_quantile(abs_pressure, CONTEXT_QUANTILE)
    atr_q75 = prior_rolling_quantile(atr, CONTEXT_QUANTILE)
    selected = (
        np.isfinite(volume)
        & np.isfinite(volume_q90)
        & (volume >= volume_q90)
        & np.isfinite(range_atr)
        & np.isfinite(range_q75)
        & (range_atr >= range_q75)
        & np.isfinite(open_price)
        & np.isfinite(close)
        & np.isfinite(atr)
        & (atr > 0.0)
    )
    indexes = np.flatnonzero(selected)
    if not len(indexes):
        return DataFrame()
    body_half_width = np.abs(close[indexes] - open_price[indexes]) / 2.0
    width = np.clip(
        body_half_width,
        MIN_ZONE_HALF_WIDTH_ATR * atr[indexes],
        MAX_ZONE_HALF_WIDTH_ATR * atr[indexes],
    )
    dates = normalize_dates(base["date"])
    return DataFrame(
        {
            "origin_index": indexes.astype(np.int64),
            "origin_time": dates.iloc[indexes].reset_index(drop=True),
            "origin_available_at": (
                dates.iloc[indexes].reset_index(drop=True) + pd.Timedelta(hours=1)
            ),
            "level_price": (open_price[indexes] + close[indexes]) / 2.0,
            "zone_half_width": width,
            "zone_half_width_atr": width / atr[indexes],
            "origin_volume_vs_prior_median": volume[indexes] / volume_median[indexes],
            "origin_volume_vs_q90": volume[indexes] / volume_q90[indexes],
            "origin_range_atr": range_atr[indexes],
            "origin_range_vs_q75": range_atr[indexes] / range_q75[indexes],
            "origin_abs_pressure": abs_pressure[indexes],
            "origin_pressure_above_q75": abs_pressure[indexes] >= pressure_q75[indexes],
            "origin_atr_above_q75": atr[indexes] >= atr_q75[indexes],
            "origin_impulse_sign": np.sign(close[indexes] - open_price[indexes]).astype(np.int8),
        }
    )


def first_revisit(
    *,
    origin_index: int,
    level: float,
    half_width: float,
    high: np.ndarray,
    low: np.ndarray,
) -> tuple[int, str] | None:
    outside_run = 0
    start = origin_index + 1
    stop = min(origin_index + ORIGIN_MEMORY_HOURS + 1, len(high))
    for index in range(start, stop):
        contact = high[index] >= level - half_width and low[index] <= level + half_width
        if outside_run >= OUTSIDE_INTERVAL_HOURS:
            if contact:
                return index, "contact"
            expanded = (
                high[index] >= level - NEAR_MISS_WIDTH_MULTIPLIER * half_width
                and low[index] <= level + NEAR_MISS_WIDTH_MULTIPLIER * half_width
            )
            if expanded:
                return index, "near_miss"
        if contact:
            outside_run = 0
        else:
            outside_run += 1
    return None


def origin_revisit_events(
    *,
    pair: str,
    base: DataFrame,
    origins: DataFrame,
) -> DataFrame:
    if origins.empty:
        return DataFrame()
    high = numeric_array(base["high"])
    low = numeric_array(base["low"])
    pre_close = numeric_array(base["pre_close"])
    atr = numeric_array(base["base_atr"])
    dates = normalize_dates(base["date"])
    rows: list[dict[str, Any]] = []
    for origin in origins.itertuples(index=False):
        revisit = first_revisit(
            origin_index=int(origin.origin_index),
            level=float(origin.level_price),
            half_width=float(origin.zone_half_width),
            high=high,
            low=low,
        )
        if revisit is None:
            continue
        index, event_kind = revisit
        if index >= len(base) or not np.isfinite(atr[index]) or atr[index] <= 0.0:
            continue
        lower = float(origin.level_price - origin.zone_half_width)
        upper = float(origin.level_price + origin.zone_half_width)
        if pre_close[index] < lower:
            approach = "from_below"
        elif pre_close[index] > upper:
            approach = "from_above"
        else:
            approach = "already_inside_or_unclear"
        rows.append(
            {
                "pair": pair,
                "period": base["period"].iloc[index],
                "event_time": dates.iloc[index],
                "base_index": int(index),
                "event_kind": event_kind,
                "approach_state": approach,
                "origin_index": int(origin.origin_index),
                "origin_time": origin.origin_time,
                "origin_available_at": origin.origin_available_at,
                "origin_age_hours": int(index - origin.origin_index),
                "level_price": float(origin.level_price),
                "zone_half_width": float(origin.zone_half_width),
                "zone_half_width_atr": float(origin.zone_half_width / atr[index]),
                "base_atr": float(atr[index]),
                "pre_distance_atr": float(abs(origin.level_price - pre_close[index]) / atr[index]),
                "origin_volume_vs_prior_median": float(origin.origin_volume_vs_prior_median),
                "origin_volume_vs_q90": float(origin.origin_volume_vs_q90),
                "origin_range_atr": float(origin.origin_range_atr),
                "origin_range_vs_q75": float(origin.origin_range_vs_q75),
                "origin_abs_pressure": float(origin.origin_abs_pressure),
                "origin_pressure_above_q75": bool(origin.origin_pressure_above_q75),
                "origin_atr_above_q75": bool(origin.origin_atr_above_q75),
                "origin_impulse_sign": int(origin.origin_impulse_sign),
                "direction_prediction": False,
                "profit_optimization": False,
            }
        )
    if not rows:
        return DataFrame()
    events = DataFrame(rows)
    event_order = events["event_kind"].map({"contact": 0, "near_miss": 1})
    events["event_kind_order"] = event_order
    return (
        events.sort_values(
            ["event_time", "event_kind_order", "pre_distance_atr", "origin_age_hours"],
            kind="stable",
        )
        .drop_duplicates("event_time", keep="first")
        .drop(columns="event_kind_order")
        .reset_index(drop=True)
    )


def classify_origin_scope(events: DataFrame) -> np.ndarray:
    density = events["overlap_density_zone_count"].to_numpy(dtype=int) > 0
    reference = events["overlap_reference_level_count"].to_numpy(dtype=int) > 0
    return np.select(
        [density & reference, density, reference],
        [
            "origin_plus_density_and_reference",
            "origin_plus_density",
            "origin_plus_reference",
        ],
        default="isolated_origin",
    )


def period_roles(manifest: dict[str, Any]) -> dict[str, str]:
    return {
        str(period["id"]): str(period["role"])
        for period in manifest["data"]["chronological_periods"]
    }


def geometry_coverage(geometry: DataFrame, *, manifest: dict[str, Any]) -> DataFrame:
    output = (
        geometry.groupby(["pair", "period", "origin_scope", "event_kind"], observed=True)
        .agg(
            events=("event_time", "size"),
            unique_event_days=("event_time", lambda values: values.dt.floor("1D").nunique()),
            median_origin_age_hours=("origin_age_hours", "median"),
            median_zone_half_width_atr=("zone_half_width_atr", "median"),
            median_origin_volume_vs_prior=("origin_volume_vs_prior_median", "median"),
            median_origin_range_atr=("origin_range_atr", "median"),
            pressure_high_fraction=("origin_pressure_above_q75", "mean"),
            volatility_high_fraction=("origin_atr_above_q75", "mean"),
        )
        .reset_index()
    )
    output["period_role"] = output["period"].map(period_roles(manifest))
    output["direction_prediction"] = False
    output["profit_optimization"] = False
    return output


def comparison_support(geometry: DataFrame, *, manifest: dict[str, Any]) -> DataFrame:
    roles = period_roles(manifest)
    rows: list[dict[str, Any]] = []
    for (period, scope), group in geometry.groupby(["period", "origin_scope"], observed=True):
        row: dict[str, Any] = {
            "period": period,
            "period_role": roles[str(period)],
            "origin_scope": scope,
        }
        for event_kind in EVENT_KINDS:
            selected = group.loc[group["event_kind"].eq(event_kind)]
            row[f"{event_kind}_events"] = len(selected)
            row[f"{event_kind}_coins"] = int(selected["pair"].nunique())
            row[f"{event_kind}_gate_passed"] = (
                len(selected) >= MIN_COHORT_PERIOD_EVENTS
                and int(selected["pair"].nunique()) >= MIN_COHORT_PERIOD_COINS
            )
        row["period_scope_supported"] = bool(
            row["contact_gate_passed"] and row["near_miss_gate_passed"]
        )
        row["direction_prediction"] = False
        row["profit_optimization"] = False
        rows.append(row)
    output = DataFrame(rows)
    validation = output["period_role"].eq("chronological_internal_validation")
    repeated = (
        output.loc[validation]
        .groupby("origin_scope", observed=True)
        .agg(
            validation_periods=("period", "nunique"),
            supported_validation_periods=("period_scope_supported", "sum"),
        )
        .reset_index()
    )
    repeated["both_validation_periods_supported"] = repeated["validation_periods"].eq(2) & repeated[
        "supported_validation_periods"
    ].eq(2)
    output = output.merge(repeated, on="origin_scope", how="left", validate="many_to_one")
    output["both_validation_periods_supported"] = output[
        "both_validation_periods_supported"
    ].fillna(False)
    return output


def supported_scope_count(support: DataFrame) -> int:
    return int(support.loc[support["both_validation_periods_supported"], "origin_scope"].nunique())


def integrity_record(
    *, geometry: DataFrame, support: DataFrame, results: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    failures = [row for row in results if row["status"] == "failed"]
    future_origin_violations = int(
        (
            pd.to_datetime(geometry["origin_available_at"], utc=True)
            > pd.to_datetime(geometry["event_time"], utc=True)
        ).sum()
    )
    age = pd.to_numeric(geometry["origin_age_hours"], errors="coerce")
    age_violations = int(((age <= OUTSIDE_INTERVAL_HOURS) | (age > ORIGIN_MEMORY_HOURS)).sum())
    invalid_scopes = sorted(set(geometry["origin_scope"]) - set(ORIGIN_SCOPES))
    invalid_kinds = sorted(set(geometry["event_kind"]) - set(EVENT_KINDS))
    direction_violations = int(geometry["direction_prediction"].ne(False).sum())
    profit_violations = int(geometry["profit_optimization"].ne(False).sum())
    duplicate_rows = int(geometry.duplicated(["pair", "event_time"]).sum())
    return {
        "created_at_utc": utc_now(),
        "passed": bool(
            not failures
            and not geometry.empty
            and future_origin_violations == 0
            and age_violations == 0
            and not invalid_scopes
            and not invalid_kinds
            and direction_violations == 0
            and profit_violations == 0
            and duplicate_rows == 0
        ),
        "failures": failures,
        "geometry_event_rows": len(geometry),
        "periods": sorted(geometry["period"].astype(str).unique()),
        "origin_scopes": sorted(geometry["origin_scope"].astype(str).unique()),
        "event_kinds": sorted(geometry["event_kind"].astype(str).unique()),
        "future_origin_violations": future_origin_violations,
        "origin_age_violations": age_violations,
        "invalid_scopes": invalid_scopes,
        "invalid_event_kinds": invalid_kinds,
        "duplicate_pair_event_rows": duplicate_rows,
        "direction_prediction_violations": direction_violations,
        "profit_optimization_violations": profit_violations,
        "supported_both_validation_scopes": supported_scope_count(support),
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


def combine_pair_outputs(
    *, directory: Path, pairs: Sequence[str], request_sha256: str
) -> DataFrame:
    return pd.concat(
        [
            validate_pair_output(
                directory / f"{pair_stem(pair)}.parquet",
                pair=pair,
                request_sha256=request_sha256,
            )
            for pair in pairs
        ],
        ignore_index=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())

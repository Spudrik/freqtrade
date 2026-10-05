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
    episode_start_mask,
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
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_density import (  # noqa: E501
    DensityZoneSet,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_density_reaction import (  # noqa: E501
    FIXED_BIN_WIDTH_ATR,
    DensitySurface,
    attach_structure_overlaps,
    build_density_surfaces,
    classify_actual_scope,
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
REPORT_ROOT = OUTPUT_ROOT / "generation3_branches" / "g3d_density_arrival"
ARTIFACT_ROOT = LARGE_ARTIFACT_ROOT / "generation3_branches" / "g3d_density_arrival"
OUTPUT_SCHEMA_VERSION = 1
OUTSIDE_INTERVAL_HOURS = 6
OUTCOME_EMBARGO_HOURS = 24
NEAR_MISS_WIDTH_MULTIPLIER = 2.0
MIN_COHORT_PERIOD_EVENTS = 50
MIN_COHORT_PERIOD_COINS = 5
EVENT_STATES = (
    "first_arrival_after_outside_interval",
    "near_miss",
    "already_inside",
    "repeat_contact",
)
OCCUPANCY_STATES = ("already_inside", "repeat_contact")

GEOMETRY_COLUMNS = (
    "pair",
    "period",
    "event_time",
    "base_index",
    "density_family",
    "history_hours",
    "level_name",
    "zone_rank",
    "event_state",
    "event_kind",
    "approach_state",
    "actual_scope",
    "level_price",
    "zone_half_width",
    "zone_half_width_atr",
    "base_atr",
    "pre_distance_atr",
    "zone_support_fraction",
    "prior_contact_candles",
    "prior_zone_continuity_candles",
    "outside_interval_hours",
    "overlap_density_zone_count",
    "overlap_other_density_zone_count",
    "overlap_other_density_surface_count",
    "overlap_reference_level_count",
    "overlap_reference_group_count",
    "direction_prediction",
    "profit_optimization",
)


@dataclass(frozen=True)
class PairTask:
    pair: str
    cohort: str
    manifest_path: str
    run_id: str
    overwrite: bool
    request_sha256: str
    cache_contracts: tuple[tuple[str, str], ...]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 3D coverage-only preflight separating fresh causal density-zone "
            "arrivals from occupancy, repeat contacts, and clean near misses."
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
        "source_contracts": contracts,
        "surfaces": [
            "confirmed_swing_density_168h",
            "confirmed_swing_density_720h",
            "repeated_close_density_168h",
            "repeated_close_density_720h",
        ],
        "density_bin_width_atr": FIXED_BIN_WIDTH_ATR,
        "event_states": list(EVENT_STATES),
        "event_definition": {
            "outside_interval_hours": OUTSIDE_INTERVAL_HOURS,
            "zone_continuity": (
                "At every one of the six completed pre-event candles, at least one zone "
                "from the same causal surface must overlap the event zone."
            ),
            "first_arrival_after_outside_interval": (
                "The event candle contacts the zone, the previous close is outside on an "
                "explicit side, and none of the six completed pre-event candle ranges "
                "contacted the event zone."
            ),
            "already_inside": (
                "At an episode-start timestamp, the previous completed close is inside "
                "the continuously available event zone."
            ),
            "repeat_contact": (
                "At an episode-start timestamp, the previous close is outside but at "
                "least one of the six completed pre-event candles contacted the event zone."
            ),
            "near_miss": (
                "After the same clean outside interval, the candle enters two zone half-widths "
                "but does not contact any zone on that density surface."
            ),
            "unstable_or_new_zone": "Excluded rather than labelled as a reaction.",
        },
        "scopes_kept_separate": [
            "single_density_zone",
            "density_cluster",
            "density_plus_reference_cluster",
        ],
        "coverage_gate": {
            "minimum_events_per_state_cell": MIN_COHORT_PERIOD_EVENTS,
            "minimum_coins_per_state_cell": MIN_COHORT_PERIOD_COINS,
            "primary_cell": (
                "fresh arrival, near miss, and combined already-inside/repeat occupancy "
                "must each pass in the same surface/scope/validation period"
            ),
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
    record_path = run_dir / "g3d_preflight_run_record.json"
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
            "G2E density-zone crossing results were dominated by broad-zone occupancy and "
            "mixed truly-inside events with zones that moved around price."
        ),
        "hypothesis": (
            "A continuously known density area receives enough clean first arrivals, near "
            "misses, and occupied/repeated observations for a fair fresh-reaction test."
        ),
        "pass_fail": (
            "Open outcomes only for a surface and scope where fresh arrival, near miss, and "
            "combined occupancy each supply at least 50 independent episode timestamps from "
            "at least five coins in both validation periods. Otherwise defer or park that cell."
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
            cohort=args.cohort,
            manifest_path=str(manifest_path),
            run_id=args.run_id,
            overwrite=args.overwrite,
            request_sha256=request_sha256,
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
        atomic_write_parquet(inventory, run_dir / "g3d_pair_inventory.parquet")
        failures = inventory.loc[inventory["status"].eq("failed")]
        if not failures.empty:
            raise RuntimeError(
                f"{len(failures)} pair task(s) failed; inspect g3d_pair_inventory.parquet."
            )
        geometry = combine_pair_outputs(
            directory=artifact_dir / "pair_geometry",
            pairs=pairs,
            request_sha256=request_sha256,
        )
        coverage = geometry_coverage(geometry, manifest=manifest)
        support = comparison_support(geometry, manifest=manifest)
        atomic_write_parquet(coverage, run_dir / "g3d_state_coverage.parquet")
        atomic_write_parquet(support, run_dir / "g3d_comparison_support.parquet")
        integrity = integrity_record(geometry=geometry, support=support, results=results)
        atomic_write_json(integrity, run_dir / "g3d_preflight_integrity.json")
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "pair_tasks": results,
                "geometry_event_rows": len(geometry),
                "coverage_rows": len(coverage),
                "comparison_support_rows": len(support),
                "supported_both_validation_cells": supported_both_validation_cell_count(support),
                "integrity": str(run_dir / "g3d_preflight_integrity.json"),
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
    if "g3d_density_first_arrival_vs_occupancy" not in branches:
        raise ValueError("The frozen Generation 3D branch is unavailable.")
    boundary = frozen["research_boundary"]
    if boundary.get("direction_prediction") is not False:
        raise ValueError("Generation 3D must keep direction disabled.")
    if boundary.get("profit_optimization") is not False:
        raise ValueError("Generation 3D must keep profit optimization disabled.")
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
        return {
            "pair": task.pair,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


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
            "geometry_events": len(existing),
            "seconds": round(time.perf_counter() - started, 3),
        }
    for cache_path, expected_sha256 in task.cache_contracts:
        if sha256_file(Path(cache_path)) != expected_sha256:
            raise ValueError(f"Frozen reference cache changed before pair build: {cache_path}")

    manifest_path = Path(task.manifest_path)
    manifest = load_manifest(manifest_path)
    base = prepare_base_market_frame(task.pair, manifest)
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
    frames: list[DataFrame] = []
    diagnostic_counts: dict[str, int] = {}
    for surface in surfaces:
        events, diagnostics = surface_event_geometry(
            pair=task.pair,
            base=base,
            surface=surface,
        )
        for key, value in diagnostics.items():
            diagnostic_counts[key] = diagnostic_counts.get(key, 0) + value
        events = eligible_period_events(events, manifest, embargo_hours=OUTCOME_EMBARGO_HOURS)
        if events.empty:
            continue
        events = attach_structure_overlaps(
            events,
            source_surface_key=surface.key,
            density_levels=density_levels,
            density_widths=density_widths,
            density_surface_keys=density_surface_keys,
            reference_levels=reference_levels,
            reference_widths=reference_widths,
            reference_groups=reference_groups,
        )
        events["actual_scope"] = classify_actual_scope(events)
        frames.append(events)
    if not frames:
        raise ValueError(f"No eligible G3D event geometry for {task.pair}.")
    geometry = pd.concat(frames, ignore_index=True)
    geometry = geometry.loc[:, list(GEOMETRY_COLUMNS)].copy()
    geometry["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    geometry["run_request_sha256"] = task.request_sha256
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(geometry, output_path)
    result: dict[str, Any] = {
        "pair": task.pair,
        "status": "completed",
        "source_rows": len(base),
        "density_surfaces": len(surfaces),
        "geometry_events": len(geometry),
        "seconds": round(time.perf_counter() - started, 3),
    }
    result.update({f"diagnostic__{key}": value for key, value in diagnostic_counts.items()})
    for state in EVENT_STATES:
        result[f"state__{state}"] = int(geometry["event_state"].eq(state).sum())
    return result


def classify_density_event(
    *,
    index: int,
    level: float,
    half_width: float,
    event_kind: str,
    zones: DensityZoneSet,
    high: np.ndarray,
    low: np.ndarray,
    pre_close: np.ndarray,
) -> tuple[str, str, int, int]:
    if index < OUTSIDE_INTERVAL_HOURS:
        return "insufficient_prior_context", "unclear", 0, index
    if not (
        np.isfinite(level)
        and np.isfinite(half_width)
        and half_width > 0.0
        and np.isfinite(pre_close[index])
    ):
        return "invalid_geometry", "unclear", 0, 0
    lower = level - half_width
    upper = level + half_width
    history = slice(index - OUTSIDE_INTERVAL_HOURS, index)
    prior_zone_overlap = (
        zones.valid[history]
        & (zones.levels[history] - zones.half_widths[history] <= upper)
        & (zones.levels[history] + zones.half_widths[history] >= lower)
    )
    continuity_rows = int(prior_zone_overlap.any(axis=1).sum())
    if continuity_rows != OUTSIDE_INTERVAL_HOURS:
        return "unstable_or_new_zone", "unclear", 0, continuity_rows
    prior_contacts = (high[history] >= lower) & (low[history] <= upper)
    prior_contact_count = int(prior_contacts.sum())
    previous_close = float(pre_close[index])
    if previous_close < lower:
        approach = "from_below"
    elif previous_close > upper:
        approach = "from_above"
    else:
        approach = "already_inside_or_unclear"
    if event_kind == "near_miss":
        if approach == "already_inside_or_unclear" or prior_contact_count:
            return "near_miss_not_clean", approach, prior_contact_count, continuity_rows
        return "near_miss", approach, prior_contact_count, continuity_rows
    if event_kind != "contact":
        raise ValueError(f"Unknown density event kind: {event_kind}")
    if approach == "already_inside_or_unclear":
        return "already_inside", approach, prior_contact_count, continuity_rows
    if prior_contact_count == 0:
        return (
            "first_arrival_after_outside_interval",
            approach,
            prior_contact_count,
            continuity_rows,
        )
    return "repeat_contact", approach, prior_contact_count, continuity_rows


def surface_event_geometry(
    *,
    pair: str,
    base: DataFrame,
    surface: DensitySurface,
) -> tuple[DataFrame, dict[str, int]]:
    high = numeric_array(base["high"])
    low = numeric_array(base["low"])
    pre_close = numeric_array(base["pre_close"])
    atr = numeric_array(base["base_atr"])
    dates = normalize_dates(base["date"])
    any_surface_contact = (
        surface.zones.valid
        & (high[:, None] >= surface.zones.levels - surface.zones.half_widths)
        & (low[:, None] <= surface.zones.levels + surface.zones.half_widths)
    ).any(axis=1)
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, int] = {}
    for rank in range(surface.zones.levels.shape[1]):
        level = surface.zones.levels[:, rank]
        width = surface.zones.half_widths[:, rank]
        valid = surface.zones.valid[:, rank] & np.isfinite(atr) & (atr > 0.0)
        contact = valid & (high >= level - width) & (low <= level + width)
        near_miss = (
            valid
            & ~any_surface_contact
            & (high >= level - NEAR_MISS_WIDTH_MULTIPLIER * width)
            & (low <= level + NEAR_MISS_WIDTH_MULTIPLIER * width)
        )
        candidates = (
            (
                "contact",
                episode_start_mask(contact, level, width, cooldown=OUTSIDE_INTERVAL_HOURS),
            ),
            (
                "near_miss",
                episode_start_mask(
                    near_miss,
                    level,
                    NEAR_MISS_WIDTH_MULTIPLIER * width,
                    cooldown=OUTSIDE_INTERVAL_HOURS,
                ),
            ),
        )
        for event_kind, mask in candidates:
            for index in np.flatnonzero(mask):
                state, approach, prior_contacts, continuity = classify_density_event(
                    index=int(index),
                    level=float(level[index]),
                    half_width=float(width[index]),
                    event_kind=event_kind,
                    zones=surface.zones,
                    high=high,
                    low=low,
                    pre_close=pre_close,
                )
                diagnostics[state] = diagnostics.get(state, 0) + 1
                if state not in EVENT_STATES:
                    continue
                source_count = surface.zones.source_counts[index]
                support = surface.zones.support_counts[index, rank]
                support_fraction = (
                    float(support / source_count)
                    if np.isfinite(source_count) and source_count > 0.0
                    else np.nan
                )
                rows.append(
                    {
                        "pair": pair,
                        "period": base["period"].iloc[index],
                        "event_time": dates.iloc[index],
                        "base_index": int(index),
                        "density_family": surface.family,
                        "history_hours": int(surface.history_hours),
                        "level_name": surface.key,
                        "zone_rank": int(rank),
                        "event_state": state,
                        "event_kind": event_kind,
                        "approach_state": approach,
                        "level_price": float(level[index]),
                        "zone_half_width": float(width[index]),
                        "zone_half_width_atr": float(width[index] / atr[index]),
                        "base_atr": float(atr[index]),
                        "pre_distance_atr": float(
                            abs(level[index] - pre_close[index]) / atr[index]
                        ),
                        "zone_support_fraction": support_fraction,
                        "prior_contact_candles": prior_contacts,
                        "prior_zone_continuity_candles": continuity,
                        "outside_interval_hours": OUTSIDE_INTERVAL_HOURS,
                        "direction_prediction": False,
                        "profit_optimization": False,
                    }
                )
    if not rows:
        return DataFrame(), diagnostics
    frame = DataFrame(rows)
    frame["event_kind_order"] = frame["event_kind"].map({"contact": 0, "near_miss": 1})
    frame = (
        frame.sort_values(
            ["event_time", "event_kind_order", "pre_distance_atr", "zone_rank"],
            kind="stable",
        )
        .drop_duplicates("event_time", keep="first")
        .drop(columns="event_kind_order")
        .reset_index(drop=True)
    )
    return frame, diagnostics


def period_roles(manifest: dict[str, Any]) -> dict[str, str]:
    return {
        str(period["id"]): str(period["role"])
        for period in manifest["data"]["chronological_periods"]
    }


def geometry_coverage(geometry: DataFrame, *, manifest: dict[str, Any]) -> DataFrame:
    keys = [
        "period",
        "density_family",
        "history_hours",
        "actual_scope",
        "event_state",
    ]
    output = (
        geometry.groupby(keys, observed=True)
        .agg(
            events=("event_time", "size"),
            coins=("pair", "nunique"),
            unique_event_days=("event_time", lambda values: values.dt.floor("1D").nunique()),
            median_zone_half_width_atr=("zone_half_width_atr", "median"),
            median_prior_contact_candles=("prior_contact_candles", "median"),
            density_cluster_fraction=(
                "overlap_other_density_zone_count",
                lambda values: values.gt(0).mean(),
            ),
            reference_cluster_fraction=(
                "overlap_reference_level_count",
                lambda values: values.gt(0).mean(),
            ),
        )
        .reset_index()
    )
    output["period_role"] = output["period"].map(period_roles(manifest))
    output["state_coverage_gate_passed"] = output["events"].ge(MIN_COHORT_PERIOD_EVENTS) & output[
        "coins"
    ].ge(MIN_COHORT_PERIOD_COINS)
    output["direction_prediction"] = False
    output["profit_optimization"] = False
    return output


def comparison_support(geometry: DataFrame, *, manifest: dict[str, Any]) -> DataFrame:
    keys = ["period", "density_family", "history_hours", "actual_scope"]
    rows: list[dict[str, Any]] = []
    roles = period_roles(manifest)
    for key, group in geometry.groupby(keys, observed=True):
        row = dict(zip(keys, key, strict=True))
        row["period_role"] = roles[str(row["period"])]
        for label, states in {
            "fresh": ("first_arrival_after_outside_interval",),
            "near_miss": ("near_miss",),
            "already_inside": ("already_inside",),
            "repeat_contact": ("repeat_contact",),
            "combined_occupancy": OCCUPANCY_STATES,
        }.items():
            selected = group.loc[group["event_state"].isin(states)]
            row[f"{label}_events"] = len(selected)
            row[f"{label}_coins"] = int(selected["pair"].nunique())
            row[f"{label}_gate_passed"] = (
                len(selected) >= MIN_COHORT_PERIOD_EVENTS
                and int(selected["pair"].nunique()) >= MIN_COHORT_PERIOD_COINS
            )
        row["primary_cell_supported"] = bool(
            row["fresh_gate_passed"]
            and row["near_miss_gate_passed"]
            and row["combined_occupancy_gate_passed"]
        )
        row["direction_prediction"] = False
        row["profit_optimization"] = False
        rows.append(row)
    output = DataFrame(rows)
    validation = output["period_role"].eq("chronological_internal_validation")
    repeated_keys = ["density_family", "history_hours", "actual_scope"]
    repeated = (
        output.loc[validation]
        .groupby(repeated_keys, observed=True)
        .agg(
            validation_periods=("period", "nunique"),
            supported_validation_periods=("primary_cell_supported", "sum"),
        )
        .reset_index()
    )
    repeated["both_validation_periods_supported"] = repeated["validation_periods"].eq(2) & repeated[
        "supported_validation_periods"
    ].eq(2)
    output = output.merge(repeated, on=repeated_keys, how="left", validate="many_to_one")
    output["both_validation_periods_supported"] = output[
        "both_validation_periods_supported"
    ].fillna(False)
    return output


def integrity_record(
    *,
    geometry: DataFrame,
    support: DataFrame,
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    failures = [row for row in results if row["status"] == "failed"]
    forbidden_outcome_columns = sorted(
        column
        for column in geometry
        if column.startswith(("actual__", "control__", "delta__"))
        or column
        in {
            "contact_range_ratio",
            "contact_volume_ratio",
            "contact_pressure_change",
            "abs_excursion_atr_h1",
            "abs_excursion_atr_h4",
            "abs_excursion_atr_h24",
        }
    )
    invalid_states = sorted(set(geometry["event_state"]) - set(EVENT_STATES))
    direction_violations = int(geometry["direction_prediction"].ne(False).sum())
    profit_violations = int(geometry["profit_optimization"].ne(False).sum())
    duplicate_rows = int(
        geometry.duplicated(["pair", "density_family", "history_hours", "event_time"]).sum()
    )
    return {
        "created_at_utc": utc_now(),
        "passed": bool(
            not failures
            and not geometry.empty
            and not forbidden_outcome_columns
            and not invalid_states
            and direction_violations == 0
            and profit_violations == 0
            and duplicate_rows == 0
        ),
        "failures": failures,
        "geometry_event_rows": len(geometry),
        "periods": sorted(geometry["period"].astype(str).unique()),
        "event_states": sorted(geometry["event_state"].astype(str).unique()),
        "forbidden_outcome_columns": forbidden_outcome_columns,
        "invalid_event_states": invalid_states,
        "duplicate_pair_surface_event_rows": duplicate_rows,
        "direction_prediction_violations": direction_violations,
        "profit_optimization_violations": profit_violations,
        "supported_both_validation_cells": supported_both_validation_cell_count(support),
        "coverage_only": True,
        "reaction_outcomes_loaded": False,
        "direction_prediction": False,
        "profit_optimization": False,
    }


def supported_both_validation_cell_count(support: DataFrame) -> int:
    cell_columns = ["density_family", "history_hours", "actual_scope"]
    return len(
        support.loc[support["both_validation_periods_supported"], cell_columns]
        .drop_duplicates()
        .index
    )


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
    *,
    directory: Path,
    pairs: Sequence[str],
    request_sha256: str,
) -> DataFrame:
    frames = []
    for pair in pairs:
        path = directory / f"{pair_stem(pair)}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(validate_pair_output(path, pair=pair, request_sha256=request_sha256))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    raise SystemExit(main())

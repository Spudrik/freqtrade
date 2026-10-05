from __future__ import annotations

# Fix numerical-library thread counts before importing numpy/pandas. Pair-level
# parallelism is controlled explicitly by the CLI worker limit.
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
    atomic_write_json,
    atomic_write_parquet,
    level_cache_path,
    load_manifest,
    manifest_storage_paths,
    matched_random_time_events,
    numeric_array,
    prepare_base_market_frame,
    sha256_file,
    utc_now,
    validate_worker_count,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_localization import (  # noqa: E501
    causal_local_state,
    pair_stem,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_anchored_vwap import (  # noqa: E501
    stable_json_sha256,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_density_reaction import (  # noqa: E501
    CORE_STATE_FEATURES,
    DENSITY_STATE_FEATURES,
    VP_CONTROL_COLUMNS,
    DensitySurface,
    add_global_structure_state,
    attach_structure_overlaps,
    build_density_surfaces,
    density_matrices,
    matrix_union_coverage,
    reference_matrices,
    surface_spec,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_density_reaction import (  # noqa: E501
    HISTORY_HOURS as G2E_HISTORY_HOURS,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_density_arrival import (  # noqa: E501
    eligible_period_events as g3d_eligible_period_events,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_density_arrival import (  # noqa: E501
    surface_event_geometry,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_lvn_attribution import (  # noqa: E501
    aligned_selected_levels_from_verified_prefix,
)


OUTPUT_SCHEMA_VERSION = 1
OUTPUT_ROOT = (
    REPO_ROOT
    / "user_data"
    / "research_news_data"
    / "context_features"
    / "market_reaction_zones"
)
LARGE_OUTPUT_ROOT = Path(
    "D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones"
)
REPORT_ROOT = (
    OUTPUT_ROOT
    / "generation5_branches"
    / "g5d_fresh_arrival_path_state_decomposition"
)
ARTIFACT_ROOT = (
    LARGE_OUTPUT_ROOT
    / "generation5_branches"
    / "g5d_fresh_arrival_path_state_decomposition"
)
FROZEN_BATCH = OUTPUT_ROOT / "generation4_review" / "g5_frozen_branch_batch.json"
NORMAL_MANIFEST = OUTPUT_ROOT / "generation0_manifest.json"
MEME_MANIFEST = OUTPUT_ROOT / "generation2_shared" / "g2_meme_reaction_manifest.json"

SOURCE_SPECS: dict[str, dict[str, Path | str]] = {
    "normal": {
        "manifest": NORMAL_MANIFEST,
        "geometry_dir": (
            LARGE_OUTPUT_ROOT
            / "generation3_branches"
            / "g3d_density_arrival"
            / "g3d_density_preflight_normal10_full_20260814b"
            / "pair_geometry"
        ),
        "geometry_record": (
            OUTPUT_ROOT
            / "generation3_branches"
            / "g3d_density_arrival"
            / "g3d_density_preflight_normal10_full_20260814b"
            / "g3d_preflight_run_record.json"
        ),
        "context": (
            OUTPUT_ROOT
            / "generation2_branches"
            / "g2e_density_zones"
            / "g2e_density_reaction_normal10_full_20260814a"
            / "g2e_causal_market_context.parquet"
        ),
    },
    "meme": {
        "manifest": MEME_MANIFEST,
        "geometry_dir": (
            LARGE_OUTPUT_ROOT
            / "generation3_branches"
            / "g3d_density_arrival"
            / "g3d_density_preflight_meme_full_20260814b"
            / "pair_geometry"
        ),
        "geometry_record": (
            OUTPUT_ROOT
            / "generation3_branches"
            / "g3d_density_arrival"
            / "g3d_density_preflight_meme_full_20260814b"
            / "g3d_preflight_run_record.json"
        ),
        "context": (
            OUTPUT_ROOT
            / "generation2_branches"
            / "g2e_density_zones"
            / "g2e_density_reaction_meme_full_20260814a"
            / "g2e_causal_market_context.parquet"
        ),
    },
}

DENSITY_FAMILY = "confirmed_swing_price_density"
HISTORY_HOURS = 168
FRESH_STATE = "first_arrival_after_outside_interval"
PATH_LOOKBACK_HOURS = 168
NEAR_MISS_LOOKBACK_HOURS = 48
NEAR_MISS_WIDTH_MULTIPLIER = 2.0
PATH_WINDOWS = (6, 12, 24, 48)
MINIMUM_EVENT_SEPARATION_HOURS = 4
DEVELOPMENT_MIN_EVENTS = 60
DEVELOPMENT_MIN_COINS = 5
VALIDATION_MIN_BAND_EVENTS = 30
VALIDATION_MIN_BAND_COINS = 5
LOW_QUANTILE = 1.0 / 3.0
HIGH_QUANTILE = 2.0 / 3.0

SCOPE_MAP: dict[str, dict[str, str]] = {
    "normal_isolated_168h_confirmed_swing_density": {
        "cohort": "normal",
        "actual_scope": "single_density_zone",
        "market_group": "normal_alt",
    },
    "normal_168h_confirmed_swing_density_cluster": {
        "cohort": "normal",
        "actual_scope": "density_cluster",
        "market_group": "normal_alt",
    },
    "meme_168h_confirmed_swing_density_cluster": {
        "cohort": "meme",
        "actual_scope": "density_cluster",
        "market_group": "meme",
    },
}

PATH_FEATURES = (
    "continuous_time_outside_zone_before_arrival",
    *(f"starting_distance_in_prior_atr_{hours}h" for hours in PATH_WINDOWS),
    *(f"distance_contraction_in_prior_atr_{hours}h" for hours in PATH_WINDOWS),
    *(f"approach_speed_in_prior_atr_per_hour_{hours}h" for hours in PATH_WINDOWS),
    "number_of_prior_clean_near_misses_48h",
    "time_since_zone_became_causally_available_hours",
)
STATE_COLUMNS = (
    *CORE_STATE_FEATURES,
    *DENSITY_STATE_FEATURES,
    "state_g5d_zone_half_width_atr",
    "state_g5d_zone_support_fraction",
    "state_g5d_other_density_zone_count",
    "state_g5d_reference_level_count",
)
FORBIDDEN_SELECTION_COLUMN_TOKENS = (
    "contact_volume_ratio",
    "contact_range_ratio",
    "abs_excursion",
    "away_excursion",
    "through_excursion",
    "future",
    "target",
    "profit",
    "reaction",
)


@dataclass(frozen=True)
class PairTask:
    cohort: str
    pair: str
    manifest_path: str
    manifest_sha256: str
    geometry_path: str
    geometry_sha256: str
    context_path: str
    context_sha256: str
    run_id: str
    request_sha256: str
    overwrite: bool


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 5D outcome-blind freeze of causal arrival-path geometry, "
            "development-only bands, and validation common support."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    normal_manifest = load_manifest(NORMAL_MANIFEST)
    meme_manifest = load_manifest(MEME_MANIFEST)
    validate_worker_count(args.workers, manifest=normal_manifest)
    validate_worker_count(args.workers, manifest=meme_manifest)
    branch = validate_frozen_branch()
    tasks, source_contracts = build_tasks(
        run_id=args.run_id,
        overwrite=args.overwrite,
    )
    request = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_batch_path": str(FROZEN_BATCH.resolve()),
        "frozen_batch_sha256": sha256_file(FROZEN_BATCH),
        "frozen_branch": branch,
        "source_contracts": source_contracts,
        "pairs": {
            "normal": list(normal_manifest["data"]["pairs"]),
            "meme": list(meme_manifest["data"]["pairs"]),
        },
        "fixed_surface": {
            "density_family": DENSITY_FAMILY,
            "history_hours": HISTORY_HOURS,
            "scope_map": SCOPE_MAP,
        },
        "path_geometry": {
            "maximum_lookback_hours": PATH_LOOKBACK_HOURS,
            "distance_windows_hours": list(PATH_WINDOWS),
            "near_miss_lookback_hours": NEAR_MISS_LOOKBACK_HOURS,
            "near_miss_width_multiplier": NEAR_MISS_WIDTH_MULTIPLIER,
            "path_features": list(PATH_FEATURES),
            "distance_definition": (
                "Distance from a completed candle close to the nearest edge of the "
                "fixed event zone, divided by the ATR known before contact."
            ),
        },
        "synthetic_boundary": (
            "Reuse the canonical matched-random pseudo-boundary selector. The selector "
            "may calculate reaction columns internally, but this freeze projects them "
            "out before any band, support, or eligibility decision."
        ),
        "appended_ohlcv_handling": (
            "Before using an older frozen causal cache against a now-longer OHLCV "
            "file, compare every required historical 1h OHLCV candle with the source "
            "columns stored in that cache. A changed value or missing historical "
            "timestamp fails; only an unchanged prefix plus later appended rows passes."
        ),
        "causal_shuffle": (
            "Reuse the pre-existing one-third confirmed-swing residual rotation and "
            "exclude shuffled contacts overlapping any actual density or reference level."
        ),
        "development_band_freeze": {
            "low_quantile": LOW_QUANTILE,
            "high_quantile": HIGH_QUANTILE,
            "minimum_events": DEVELOPMENT_MIN_EVENTS,
            "minimum_coins": DEVELOPMENT_MIN_COINS,
            "threshold_source": "actual fresh arrivals in development only",
            "outcome_values_used": False,
        },
        "declared_review_thresholds_not_native_scores": {
            "minimum_independent_events_per_validation_band": (
                VALIDATION_MIN_BAND_EVENTS
            ),
            "minimum_coins_per_validation_band": VALIDATION_MIN_BAND_COINS,
            "minimum_event_separation_hours": MINIMUM_EVENT_SEPARATION_HOURS,
        },
        "reaction_outcomes_loaded_for_selection": False,
        "reaction_outcomes_opened": False,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    request_sha256 = stable_json_sha256(request)
    tasks = [
        PairTask(**{**task.__dict__, "request_sha256": request_sha256}) for task in tasks
    ]
    run_dir = REPORT_ROOT / args.run_id
    artifact_dir = ARTIFACT_ROOT / args.run_id
    pair_dir = artifact_dir / "pair_path_surfaces"
    run_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    record_path = run_dir / "g5d_preflight_run_record.json"
    if record_path.is_file() and not args.overwrite:
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        if existing.get("request_sha256") != request_sha256:
            raise ValueError("Run ID already exists with a different G5D freeze contract.")
        if existing.get("status") in {
            "completed_ready_for_outcomes",
            "parked_insufficient_path_common_support",
        }:
            print(json.dumps(existing, indent=2, sort_keys=True))
            return 0

    record: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "status": "running",
        "started_at_utc": utc_now(),
        "orchestrator_pid": os.getpid(),
        "workers": args.workers,
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "request_sha256": request_sha256,
        "request_contract": request,
        "baseline": (
            "G4B retained fresh-arrival activity but did not attribute most cluster "
            "responses to the exact confirmed-swing density price."
        ),
        "hypothesis": branch["hypothesis"],
        "pass_fail": (
            "Open outcomes only for development-frozen path bands that retain at "
            "least 30 independent events and five coins in both the low and high "
            "band, for actual and no-level synthetic fresh arrivals, in both "
            "chronological validation periods."
        ),
        "reaction_outcomes_opened": False,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(record, record_path)
    try:
        results = run_pair_tasks(tasks, workers=args.workers)
        inventory = DataFrame(results)
        atomic_write_parquet(inventory, run_dir / "g5d_pair_inventory.parquet")
        failures = inventory.loc[inventory["status"].eq("failed")]
        if not failures.empty:
            raise ValueError(
                f"{len(failures)} G5D pair task(s) failed; inspect the pair inventory."
            )
        surfaces = combine_pair_surfaces(
            pair_dir=pair_dir,
            tasks=tasks,
            request_sha256=request_sha256,
        )
        independent = select_independent_events(surfaces)
        forbidden = forbidden_selection_columns(independent.columns)
        if forbidden:
            raise ValueError(f"G5D selection surface contains outcome columns: {forbidden}")
        thresholds = freeze_development_thresholds(independent)
        support = validation_support(independent, thresholds)
        decisions = path_route_decisions(support)
        opened = bool(decisions["generic_outcome_phase_eligible"].eq(True).any())
        status = (
            "completed_ready_for_outcomes"
            if opened
            else "parked_insufficient_path_common_support"
        )
        atomic_write_parquet(
            thresholds,
            run_dir / "g5d_development_path_thresholds.parquet",
        )
        atomic_write_parquet(
            support,
            run_dir / "g5d_validation_path_support.parquet",
        )
        atomic_write_parquet(
            decisions,
            run_dir / "g5d_path_route_decisions.parquet",
        )
        integrity = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "passed": not forbidden,
            "pair_tasks": len(tasks),
            "failed_pair_tasks": 0,
            "selection_rows_before_independence": len(surfaces),
            "selection_rows_after_independence": len(independent),
            "selection_columns": list(independent.columns),
            "forbidden_selection_columns": forbidden,
            "reaction_outcome_columns_admitted_to_selection": 0,
            "threshold_rows": len(thresholds),
            "support_rows": len(support),
            "route_decision_rows": len(decisions),
            "generic_routes_opened": int(
                decisions["generic_outcome_phase_eligible"].eq(True).sum()
            ),
            "exact_location_routes_opened": int(
                decisions["exact_location_outcome_phase_eligible"].eq(True).sum()
            ),
            "reaction_outcomes_opened": False,
            "direction_prediction": False,
            "profit_optimization": False,
        }
        atomic_write_json(integrity, run_dir / "g5d_preflight_integrity.json")
        record.update(
            {
                "status": status,
                "completed_at_utc": utc_now(),
                "pair_tasks": results,
                "bulky_artifacts": str(artifact_dir),
                "selection_rows_before_independence": len(surfaces),
                "selection_rows_after_independence": len(independent),
                "threshold_rows": len(thresholds),
                "support_rows": len(support),
                "route_decisions": decisions.to_dict(orient="records"),
                "reaction_outcomes_opened": False,
                "outcome_phase_authorized_by_support": opened,
                "plain_language_result": plain_language_result(decisions),
            }
        )
        atomic_write_json(record, record_path)
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "completed_at_utc": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_write_json(record, record_path)
        raise


def validate_frozen_branch() -> dict[str, Any]:
    payload = json.loads(FROZEN_BATCH.read_text(encoding="utf-8"))
    branch = next(
        (
            item
            for item in payload.get("branches", [])
            if item.get("id") == "g5d_fresh_arrival_path_state_decomposition"
        ),
        None,
    )
    if branch is None:
        raise ValueError("The frozen Generation 5 batch does not contain G5D.")
    if payload.get("generation") != 5 or payload.get("branch_layer") != 5:
        raise ValueError("G5D requires the frozen Generation 5 / branch-layer 5 contract.")
    if branch.get("status") != "frozen_next_batch":
        raise ValueError("The G5D branch is not frozen for execution.")
    return branch


def build_tasks(*, run_id: str, overwrite: bool) -> tuple[list[PairTask], list[dict[str, Any]]]:
    tasks: list[PairTask] = []
    contracts: list[dict[str, Any]] = []
    for cohort, spec in SOURCE_SPECS.items():
        manifest_path = Path(spec["manifest"])
        geometry_record = Path(spec["geometry_record"])
        context_path = Path(spec["context"])
        manifest = load_manifest(manifest_path)
        record = json.loads(geometry_record.read_text(encoding="utf-8"))
        if record.get("status") != "completed":
            raise ValueError(f"The {cohort} G3D geometry source is not complete.")
        if record.get("reaction_outcomes_loaded") is not False:
            raise ValueError(f"The {cohort} G3D geometry source is not outcome-blind.")
        for source_path in (manifest_path, geometry_record, context_path):
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            contracts.append(
                {
                    "cohort": cohort,
                    "kind": (
                        "manifest"
                        if source_path == manifest_path
                        else "geometry_record"
                        if source_path == geometry_record
                        else "causal_market_context"
                    ),
                    "path": str(source_path.resolve()),
                    "sha256": sha256_file(source_path),
                }
            )
        for pair in manifest["data"]["pairs"]:
            geometry_path = Path(spec["geometry_dir"]) / f"{pair_stem(pair)}.parquet"
            if not geometry_path.is_file():
                raise FileNotFoundError(geometry_path)
            geometry_sha256 = sha256_file(geometry_path)
            contracts.append(
                {
                    "cohort": cohort,
                    "kind": "pair_geometry",
                    "pair": pair,
                    "path": str(geometry_path),
                    "sha256": geometry_sha256,
                }
            )
            tasks.append(
                PairTask(
                    cohort=cohort,
                    pair=pair,
                    manifest_path=str(manifest_path.resolve()),
                    manifest_sha256=sha256_file(manifest_path),
                    geometry_path=str(geometry_path),
                    geometry_sha256=geometry_sha256,
                    context_path=str(context_path.resolve()),
                    context_sha256=sha256_file(context_path),
                    run_id=run_id,
                    request_sha256="pending",
                    overwrite=overwrite,
                )
            )
    return tasks, contracts


def run_pair_tasks(tasks: Sequence[PairTask], *, workers: int) -> list[dict[str, Any]]:
    if workers == 1:
        return [safe_build_pair(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(safe_build_pair, task): task for task in tasks}
        results = [future.result() for future in as_completed(future_map)]
    return sorted(results, key=lambda row: (row["cohort"], row["pair"]))


def safe_build_pair(task: PairTask) -> dict[str, Any]:
    try:
        return build_pair(task)
    except Exception as exc:
        return {
            "cohort": task.cohort,
            "pair": task.pair,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_pair(task: PairTask) -> dict[str, Any]:
    started = time.perf_counter()
    output_path = (
        ARTIFACT_ROOT
        / task.run_id
        / "pair_path_surfaces"
        / f"{task.cohort}__{pair_stem(task.pair)}.parquet"
    )
    if output_path.is_file() and not task.overwrite:
        existing = validate_pair_output(output_path, task=task)
        return pair_result(task, existing, "existing", started)
    for path_value, expected in (
        (task.manifest_path, task.manifest_sha256),
        (task.geometry_path, task.geometry_sha256),
        (task.context_path, task.context_sha256),
    ):
        if sha256_file(Path(path_value)) != expected:
            raise ValueError(f"Frozen G5D source changed before pair build: {path_value}")

    manifest_path = Path(task.manifest_path)
    manifest = load_manifest(manifest_path)
    base = prepare_base_market_frame(task.pair, manifest)
    context = pd.read_parquet(task.context_path)
    state = causal_local_state(base).merge(
        context,
        on="date",
        how="left",
        validate="one_to_one",
    )
    surfaces = build_density_surfaces(base)
    surface = next(
        (
            item
            for item in surfaces
            if item.family == DENSITY_FAMILY and item.history_hours == HISTORY_HOURS
        ),
        None,
    )
    if surface is None or surface.shuffled_zones is None:
        raise ValueError(f"The fixed G5D density surface is unavailable for {task.pair}.")
    geometry = pd.read_parquet(task.geometry_path)
    geometry = geometry.loc[
        geometry["density_family"].astype(str).eq(DENSITY_FAMILY)
        & pd.to_numeric(geometry["history_hours"], errors="coerce").eq(HISTORY_HOURS)
    ].copy()
    geometry["scope_id"] = geometry["actual_scope"].map(
        scope_ids_for_cohort(task.cohort)
    )
    geometry = geometry.loc[geometry["scope_id"].notna()].copy()
    if geometry.empty:
        raise ValueError(f"No fixed G5D actual geometry for {task.pair}.")
    verify_frozen_ohlcv_prefix(
        pair=task.pair,
        base=base,
        manifest=manifest,
        through_utc=pd.Timestamp(geometry["event_time"].max()),
    )
    references = aligned_selected_levels_from_verified_prefix(
        pair=task.pair,
        base=base,
        manifest_path=manifest_path,
        timeframes=("1h",),
    )
    wanted_reference_columns = {
        *{column for columns in VP_CONTROL_COLUMNS.values() for column in columns},
        *(
            f"generic_rolling_{side}_{history}"
            for side in ("high", "low")
            for history in G2E_HISTORY_HOURS
        ),
    }
    references = [
        item for item in references if item.spec.column in wanted_reference_columns
    ]
    density_levels, density_widths, density_surface_keys = density_matrices(surfaces)
    reference_levels, reference_widths, reference_groups = reference_matrices(
        references,
        base_atr=numeric_array(base["base_atr"]),
    )
    add_global_structure_state(
        state,
        base=base,
        surfaces=surfaces,
        density_levels=density_levels,
        density_widths=density_widths,
        density_surface_keys=density_surface_keys,
        reference_levels=reference_levels,
    )
    state["state_surface_active_zone_count"] = surface.zones.valid.sum(axis=1).astype(float)
    state["state_surface_zone_coverage_4atr"] = matrix_union_coverage(
        level_matrix=surface.zones.levels,
        width_matrix=surface.zones.half_widths,
        base_atr=numeric_array(base["base_atr"]),
        pre_close=numeric_array(base["pre_close"]),
    )

    geometry["boundary_kind"] = "actual_density_zone"
    actual = attach_event_context_and_paths(
        geometry,
        base=base,
        state=state,
        cohort=task.cohort,
        zone_surface=surface,
    )

    pseudo_frames: list[DataFrame] = []
    for scope_id, fresh in actual.loc[actual["event_state"].eq(FRESH_STATE)].groupby(
        "scope_id",
        observed=True,
    ):
        pseudo = synthetic_no_level_events(
            fresh,
            pair=task.pair,
            base=base,
            surface=surface,
            density_levels=density_levels,
            density_widths=density_widths,
            density_surface_keys=density_surface_keys,
            reference_levels=reference_levels,
            reference_widths=reference_widths,
            reference_groups=reference_groups,
        )
        if pseudo.empty:
            continue
        pseudo["scope_id"] = scope_id
        pseudo["boundary_kind"] = "same_width_same_approach_no_level"
        pseudo = attach_event_context_and_paths(
            pseudo,
            base=base,
            state=state,
            cohort=task.cohort,
            zone_surface=None,
        )
        pseudo = pseudo.loc[pseudo["event_state"].eq(FRESH_STATE)].copy()
        pseudo_frames.append(pseudo)

    shuffled_surface = DensitySurface(
        family=DENSITY_FAMILY,
        history_hours=HISTORY_HOURS,
        zones=surface.shuffled_zones,
    )
    shuffled, _ = surface_event_geometry(
        pair=task.pair,
        base=base,
        surface=shuffled_surface,
    )
    shuffled = g3d_eligible_period_events(shuffled, manifest, embargo_hours=24)
    if not shuffled.empty:
        shuffled = attach_structure_overlaps(
            shuffled,
            source_surface_key=None,
            density_levels=density_levels,
            density_widths=density_widths,
            density_surface_keys=density_surface_keys,
            reference_levels=reference_levels,
            reference_widths=reference_widths,
            reference_groups=reference_groups,
        )
        shuffled = shuffled.loc[
            shuffled["event_state"].eq(FRESH_STATE)
            & ~shuffled["overlap_any_real_level"].astype(bool)
        ].copy()
        shuffled["scope_id"] = "shared_causal_shuffle_control"
        shuffled["boundary_kind"] = "causal_shuffled_location"
        shuffled = attach_event_context_and_paths(
            shuffled,
            base=base,
            state=state,
            cohort=task.cohort,
            zone_surface=shuffled_surface,
        )

    frames = [actual, *pseudo_frames]
    if not shuffled.empty:
        frames.append(shuffled)
    output = pd.concat(frames, ignore_index=True, sort=False)
    output = project_selection_columns(output)
    output["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    output["run_request_sha256"] = task.request_sha256
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(output, output_path)
    return pair_result(task, output, "completed", started)


def scope_ids_for_cohort(cohort: str) -> dict[str, str]:
    return {
        values["actual_scope"]: scope_id
        for scope_id, values in SCOPE_MAP.items()
        if values["cohort"] == cohort
    }


def verify_frozen_ohlcv_prefix(
    *,
    pair: str,
    base: DataFrame,
    manifest: dict[str, Any],
    through_utc: pd.Timestamp,
) -> None:
    storage = manifest_storage_paths(manifest)
    cache_path = level_cache_path(
        pair,
        "1h",
        ("core", "generic"),
        cache_dir=storage.cache_dir,
    )
    frozen = pd.read_parquet(
        cache_path,
        columns=[
            "source_open",
            "source_open_price",
            "source_high",
            "source_low",
            "source_close",
            "source_volume",
        ],
    ).rename(
        columns={
            "source_open": "date",
            "source_open_price": "frozen_open",
            "source_high": "frozen_high",
            "source_low": "frozen_low",
            "source_close": "frozen_close",
            "source_volume": "frozen_volume",
        }
    )
    frozen["date"] = pd.to_datetime(frozen["date"], utc=True, errors="raise")
    current = base.loc[
        pd.to_datetime(base["date"], utc=True, errors="raise") <= pd.Timestamp(through_utc),
        ["date", "open", "high", "low", "close", "volume"],
    ].copy()
    current["date"] = pd.to_datetime(current["date"], utc=True, errors="raise")
    comparison = current.merge(frozen, on="date", how="left", validate="one_to_one")
    if comparison.empty or comparison["frozen_close"].isna().any():
        missing = int(comparison["frozen_close"].isna().sum())
        raise ValueError(
            f"Frozen 1h cache does not cover {missing} required historical rows for {pair}."
        )
    mismatches = 0
    maximum_difference = 0.0
    for column in ("open", "high", "low", "close", "volume"):
        current_values = pd.to_numeric(comparison[column], errors="coerce").to_numpy(float)
        frozen_values = pd.to_numeric(
            comparison[f"frozen_{column}"], errors="coerce"
        ).to_numpy(float)
        finite = np.isfinite(current_values) & np.isfinite(frozen_values)
        scale = np.maximum(1.0, np.maximum(np.abs(current_values), np.abs(frozen_values)))
        different = finite & (np.abs(current_values - frozen_values) > 1e-10 * scale)
        different |= np.isfinite(current_values) ^ np.isfinite(frozen_values)
        mismatches += int(different.sum())
        if finite.any():
            maximum_difference = max(
                maximum_difference,
                float(np.max(np.abs(current_values[finite] - frozen_values[finite]))),
            )
    if mismatches:
        raise ValueError(
            f"Current OHLCV changed inside the frozen G5D prefix for {pair}: "
            f"{mismatches} field mismatches, maximum absolute difference "
            f"{maximum_difference:.12g}."
        )


def synthetic_no_level_events(
    actual: DataFrame,
    *,
    pair: str,
    base: DataFrame,
    surface: DensitySurface,
    density_levels: np.ndarray,
    density_widths: np.ndarray,
    density_surface_keys: np.ndarray,
    reference_levels: np.ndarray,
    reference_widths: np.ndarray,
    reference_groups: np.ndarray,
) -> DataFrame:
    if actual.empty:
        return DataFrame()
    matched_input = actual.copy()
    indexes = matched_input["base_index"].to_numpy(dtype=np.int64)
    matched_input["volatility_band"] = base["volatility_band"].iloc[indexes].to_numpy()
    matched_input["contact_range_band"] = (
        base["contact_range_band"].iloc[indexes].to_numpy()
    )
    # The canonical selector needs path arrays only because its public helper also
    # constructs outcomes. NaN arrays preserve the exact geometry selection without
    # admitting a future value to this preflight.
    null_paths = {
        name: np.full((len(base), 1), np.nan, dtype=np.float64)
        for name in ("high", "low", "close", "volume", "pressure")
    }
    pseudo = matched_random_time_events(
        merged=base,
        paths=null_paths,
        actual=matched_input,
        spec=surface_spec(surface, suffix="g5d_path_control"),
        pair=pair,
        timeframe="1h_causal_density",
        zone_method="same_width_same_approach_no_level",
        horizons=(1,),
    )
    if pseudo.empty:
        return pseudo
    pseudo = attach_structure_overlaps(
        pseudo,
        source_surface_key=None,
        density_levels=density_levels,
        density_widths=density_widths,
        density_surface_keys=density_surface_keys,
        reference_levels=reference_levels,
        reference_widths=reference_widths,
        reference_groups=reference_groups,
    )
    pseudo = pseudo.loc[~pseudo["overlap_any_real_level"].astype(bool)].copy()
    pseudo["event_state"] = classify_fixed_boundary_states(pseudo, base=base)
    pseudo["event_kind"] = "contact"
    pseudo["zone_rank"] = -1
    pseudo["zone_support_fraction"] = np.nan
    pseudo["prior_contact_candles"] = np.nan
    pseudo["prior_zone_continuity_candles"] = np.nan
    pseudo["outside_interval_hours"] = 12
    pseudo["actual_scope"] = "synthetic_no_real_level"
    return pseudo


def classify_fixed_boundary_states(events: DataFrame, *, base: DataFrame) -> pd.Series:
    high = numeric_array(base["high"])
    low = numeric_array(base["low"])
    pre_close = numeric_array(base["pre_close"])
    states: list[str] = []
    for row in events.itertuples(index=False):
        index = int(row.base_index)
        level = float(row.level_price)
        width = float(row.zone_half_width)
        if index < 12 or not np.isfinite(level) or not np.isfinite(width):
            states.append("insufficient_prior_context")
            continue
        lower = level - width
        upper = level + width
        history = slice(index - 12, index)
        contacts = (high[history] >= lower) & (low[history] <= upper)
        if lower <= pre_close[index] <= upper:
            states.append("already_inside")
        elif not contacts.any():
            states.append(FRESH_STATE)
        else:
            states.append("repeat_contact")
    return pd.Series(states, index=events.index, dtype="string")


def attach_event_context_and_paths(
    events: DataFrame,
    *,
    base: DataFrame,
    state: DataFrame,
    cohort: str,
    zone_surface: DensitySurface | None,
) -> DataFrame:
    if events.empty:
        return events.copy()
    output = events.copy().reset_index(drop=True)
    indexes = output["base_index"].to_numpy(dtype=np.int64)
    if np.any(indexes < 0) or np.any(indexes >= len(base)):
        raise ValueError("G5D event index lies outside its base market frame.")
    for column in (*CORE_STATE_FEATURES, *DENSITY_STATE_FEATURES):
        output[column] = state[column].iloc[indexes].to_numpy()
    output["state_g5d_zone_half_width_atr"] = pd.to_numeric(
        output["zone_half_width_atr"], errors="coerce"
    )
    output["state_g5d_zone_support_fraction"] = pd.to_numeric(
        output.get("zone_support_fraction", np.nan), errors="coerce"
    )
    output["state_g5d_other_density_zone_count"] = pd.to_numeric(
        output.get("overlap_other_density_zone_count", np.nan), errors="coerce"
    )
    output["state_g5d_reference_level_count"] = pd.to_numeric(
        output.get("overlap_reference_level_count", np.nan), errors="coerce"
    )
    output["cohort"] = cohort
    output["market_group"] = np.where(
        output["pair"].astype(str).str.startswith("BTC/"),
        "btc_descriptive",
        "meme" if cohort == "meme" else "normal_alt",
    )
    return attach_path_features(output, base=base, zone_surface=zone_surface)


def attach_path_features(
    events: DataFrame,
    *,
    base: DataFrame,
    zone_surface: DensitySurface | None,
) -> DataFrame:
    output = events.copy()
    high = numeric_array(base["high"])
    low = numeric_array(base["low"])
    close = numeric_array(base["close"])
    outside_values: list[float] = []
    near_miss_values: list[float] = []
    zone_age_values: list[float] = []
    starts: dict[int, list[float]] = {hours: [] for hours in PATH_WINDOWS}
    contractions: dict[int, list[float]] = {hours: [] for hours in PATH_WINDOWS}
    speeds: dict[int, list[float]] = {hours: [] for hours in PATH_WINDOWS}
    for row in output.itertuples(index=False):
        index = int(row.base_index)
        level = float(row.level_price)
        width = float(row.zone_half_width)
        event_atr = float(row.base_atr)
        if not (np.isfinite(level) and np.isfinite(width) and event_atr > 0.0):
            outside_values.append(np.nan)
            near_miss_values.append(np.nan)
            zone_age_values.append(np.nan)
            for hours in PATH_WINDOWS:
                starts[hours].append(np.nan)
                contractions[hours].append(np.nan)
                speeds[hours].append(np.nan)
            continue
        lower = level - width
        upper = level + width
        count = 0
        for position in range(index - 1, max(-1, index - PATH_LOOKBACK_HOURS - 1), -1):
            if position < 0 or (high[position] >= lower and low[position] <= upper):
                break
            count += 1
        outside_values.append(float(count))
        near_miss_values.append(
            float(
                prior_near_miss_episode_count(
                    index=index,
                    level=level,
                    width=width,
                    high=high,
                    low=low,
                )
            )
        )
        end_distance = boundary_distance(close[index - 1], level, width, event_atr)
        for hours in PATH_WINDOWS:
            if index < hours or not np.isfinite(close[index - hours]):
                start = np.nan
                contraction = np.nan
            else:
                start = boundary_distance(
                    close[index - hours],
                    level,
                    width,
                    event_atr,
                )
                contraction = start - end_distance
            starts[hours].append(start)
            contractions[hours].append(contraction)
            speeds[hours].append(contraction / hours if np.isfinite(contraction) else np.nan)
        zone_age_values.append(
            causal_zone_age_hours(
                index=index,
                level=level,
                width=width,
                zone_surface=zone_surface,
            )
        )
    output["continuous_time_outside_zone_before_arrival"] = outside_values
    for hours in PATH_WINDOWS:
        output[f"starting_distance_in_prior_atr_{hours}h"] = starts[hours]
        output[f"distance_contraction_in_prior_atr_{hours}h"] = contractions[hours]
        output[f"approach_speed_in_prior_atr_per_hour_{hours}h"] = speeds[hours]
    output["number_of_prior_clean_near_misses_48h"] = near_miss_values
    output["time_since_zone_became_causally_available_hours"] = zone_age_values
    return output


def boundary_distance(close: float, level: float, width: float, atr: float) -> float:
    if not all(np.isfinite(value) for value in (close, level, width, atr)) or atr <= 0.0:
        return np.nan
    return max(abs(close - level) - width, 0.0) / atr


def prior_near_miss_episode_count(
    *,
    index: int,
    level: float,
    width: float,
    high: np.ndarray,
    low: np.ndarray,
) -> int:
    start = max(0, index - NEAR_MISS_LOOKBACK_HOURS)
    if start >= index:
        return 0
    lower = level - width
    upper = level + width
    expanded_lower = level - NEAR_MISS_WIDTH_MULTIPLIER * width
    expanded_upper = level + NEAR_MISS_WIDTH_MULTIPLIER * width
    contact = (high[start:index] >= lower) & (low[start:index] <= upper)
    expanded = (high[start:index] >= expanded_lower) & (low[start:index] <= expanded_upper)
    near = expanded & ~contact
    if not near.any():
        return 0
    starts = near & ~np.r_[False, near[:-1]]
    return int(starts.sum())


def causal_zone_age_hours(
    *,
    index: int,
    level: float,
    width: float,
    zone_surface: DensitySurface | None,
) -> float:
    if zone_surface is None:
        return np.nan
    count = 0
    lower = level - width
    upper = level + width
    for position in range(index, -1, -1):
        valid = zone_surface.zones.valid[position]
        local_level = zone_surface.zones.levels[position]
        local_width = zone_surface.zones.half_widths[position]
        overlapping = (
            valid
            & np.isfinite(local_level)
            & np.isfinite(local_width)
            & (local_level - local_width <= upper)
            & (local_level + local_width >= lower)
        )
        if not overlapping.any():
            break
        count += 1
    return float(count)


def project_selection_columns(frame: DataFrame) -> DataFrame:
    fixed = [
        "pair",
        "cohort",
        "market_group",
        "scope_id",
        "boundary_kind",
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
        "overlap_other_density_zone_count",
        "overlap_other_density_surface_count",
        "overlap_reference_level_count",
        *PATH_FEATURES,
        *STATE_COLUMNS,
    ]
    columns = [column for column in dict.fromkeys(fixed) if column in frame.columns]
    output = frame.loc[:, columns].copy()
    forbidden = forbidden_selection_columns(output.columns)
    if forbidden:
        raise ValueError(f"Outcome columns reached G5D selection projection: {forbidden}")
    return output


def forbidden_selection_columns(columns: Sequence[str]) -> list[str]:
    return sorted(
        column
        for column in columns
        if any(token in column.lower() for token in FORBIDDEN_SELECTION_COLUMN_TOKENS)
        and column not in {"direction_prediction", "profit_optimization"}
    )


def pair_result(
    task: PairTask,
    frame: DataFrame,
    status: str,
    started: float,
) -> dict[str, Any]:
    return {
        "cohort": task.cohort,
        "pair": task.pair,
        "status": status,
        "rows": len(frame),
        "actual_rows": int(frame["boundary_kind"].eq("actual_density_zone").sum()),
        "pseudo_rows": int(
            frame["boundary_kind"].eq("same_width_same_approach_no_level").sum()
        ),
        "shuffled_rows": int(
            frame["boundary_kind"].eq("causal_shuffled_location").sum()
        ),
        "seconds": round(time.perf_counter() - started, 3),
    }


def validate_pair_output(path: Path, *, task: PairTask) -> DataFrame:
    frame = pd.read_parquet(path)
    if not frame["pair"].astype(str).eq(task.pair).all():
        raise ValueError(f"Existing G5D pair file has the wrong pair: {path}")
    if not frame["cohort"].astype(str).eq(task.cohort).all():
        raise ValueError(f"Existing G5D pair file has the wrong cohort: {path}")
    if not frame["output_schema_version"].eq(OUTPUT_SCHEMA_VERSION).all():
        raise ValueError(f"Existing G5D pair file has the wrong schema: {path}")
    if not frame["run_request_sha256"].astype(str).eq(task.request_sha256).all():
        raise ValueError(f"Existing G5D pair file has the wrong request hash: {path}")
    forbidden = forbidden_selection_columns(frame.columns)
    if forbidden:
        raise ValueError(f"Existing G5D pair file contains outcome columns: {forbidden}")
    return frame


def combine_pair_surfaces(
    *,
    pair_dir: Path,
    tasks: Sequence[PairTask],
    request_sha256: str,
) -> DataFrame:
    frames: list[DataFrame] = []
    for task in tasks:
        path = pair_dir / f"{task.cohort}__{pair_stem(task.pair)}.parquet"
        frame = pd.read_parquet(path)
        if not frame["run_request_sha256"].astype(str).eq(request_sha256).all():
            raise ValueError(f"G5D pair artifact request hash mismatch: {path}")
        frames.append(frame)
    if not frames:
        raise ValueError("No G5D pair surfaces were produced.")
    return pd.concat(frames, ignore_index=True, sort=False)


def select_independent_events(frame: DataFrame) -> DataFrame:
    if frame.empty:
        return frame.copy()
    group_columns = [
        "cohort",
        "pair",
        "scope_id",
        "boundary_kind",
        "event_state",
    ]
    ordered = frame.copy()
    ordered["event_time"] = pd.to_datetime(ordered["event_time"], utc=True, errors="raise")
    ordered = ordered.sort_values(
        [*group_columns, "event_time", "pre_distance_atr", "base_index"],
        kind="stable",
    ).drop_duplicates([*group_columns, "event_time"], keep="first")
    selected: list[DataFrame] = []
    gap = pd.Timedelta(hours=MINIMUM_EVENT_SEPARATION_HOURS)
    for _, group in ordered.groupby(group_columns, observed=True, sort=False):
        keep: list[Any] = []
        last_time: pd.Timestamp | None = None
        for row in group.itertuples():
            event_time = pd.Timestamp(row.event_time)
            if last_time is None or event_time - last_time >= gap:
                keep.append(row.Index)
                last_time = event_time
        if keep:
            selected.append(ordered.loc[keep])
    if not selected:
        return ordered.iloc[0:0].copy()
    output = pd.concat(selected, ignore_index=True, sort=False)
    output["independence_hours"] = MINIMUM_EVENT_SEPARATION_HOURS
    return output


def freeze_development_thresholds(frame: DataFrame) -> DataFrame:
    roles = combined_period_roles()
    development = frame.loc[
        frame["period"].map(roles).eq("development")
        & frame["boundary_kind"].eq("actual_density_zone")
        & frame["event_state"].eq(FRESH_STATE)
    ].copy()
    rows: list[dict[str, Any]] = []
    for scope_id, scope in SCOPE_MAP.items():
        selected = development.loc[
            development["scope_id"].eq(scope_id)
            & development["market_group"].eq(scope["market_group"])
        ]
        for feature in PATH_FEATURES:
            values = pd.to_numeric(selected[feature], errors="coerce")
            valid = selected.loc[values.notna()].copy()
            values = values.loc[values.notna()]
            events = len(valid)
            coins = int(valid["pair"].nunique())
            eligible = events >= DEVELOPMENT_MIN_EVENTS and coins >= DEVELOPMENT_MIN_COINS
            low = float(values.quantile(LOW_QUANTILE)) if eligible else np.nan
            high = float(values.quantile(HIGH_QUANTILE)) if eligible else np.nan
            distinct = bool(eligible and np.isfinite(low) and np.isfinite(high) and low < high)
            rows.append(
                {
                    "scope_id": scope_id,
                    "cohort": scope["cohort"],
                    "market_group": scope["market_group"],
                    "path_feature": feature,
                    "development_events": events,
                    "development_coins": coins,
                    "low_threshold": low,
                    "high_threshold": high,
                    "threshold_eligible": distinct,
                    "outcome_values_used": False,
                    "direction_prediction": False,
                    "profit_optimization": False,
                }
            )
    return DataFrame(rows)


def combined_period_roles() -> dict[str, str]:
    output: dict[str, str] = {}
    for manifest_path in (NORMAL_MANIFEST, MEME_MANIFEST):
        manifest = load_manifest(manifest_path)
        output.update(
            {
                str(period["id"]): str(period["role"])
                for period in manifest["data"]["chronological_periods"]
            }
        )
    return output


def validation_periods_for_scope(scope_id: str) -> tuple[str, ...]:
    manifest_path = NORMAL_MANIFEST if SCOPE_MAP[scope_id]["cohort"] == "normal" else MEME_MANIFEST
    manifest = load_manifest(manifest_path)
    return tuple(
        str(period["id"])
        for period in manifest["data"]["chronological_periods"]
        if period["role"] == "chronological_internal_validation"
    )


def validation_support(frame: DataFrame, thresholds: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds.loc[thresholds["threshold_eligible"].eq(True)].itertuples(
        index=False
    ):
        for period in validation_periods_for_scope(threshold.scope_id):
            for boundary_kind in (
                "actual_density_zone",
                "same_width_same_approach_no_level",
                "causal_shuffled_location",
            ):
                scope_value = (
                    "shared_causal_shuffle_control"
                    if boundary_kind == "causal_shuffled_location"
                    else threshold.scope_id
                )
                selected = frame.loc[
                    frame["cohort"].eq(threshold.cohort)
                    & frame["market_group"].eq(threshold.market_group)
                    & frame["scope_id"].eq(scope_value)
                    & frame["boundary_kind"].eq(boundary_kind)
                    & frame["event_state"].eq(FRESH_STATE)
                    & frame["period"].eq(period)
                ].copy()
                values = pd.to_numeric(selected[threshold.path_feature], errors="coerce")
                low = selected.loc[values.le(threshold.low_threshold)]
                high = selected.loc[values.ge(threshold.high_threshold)]
                low_events = len(low)
                high_events = len(high)
                low_coins = int(low["pair"].nunique())
                high_coins = int(high["pair"].nunique())
                passed = (
                    low_events >= VALIDATION_MIN_BAND_EVENTS
                    and high_events >= VALIDATION_MIN_BAND_EVENTS
                    and low_coins >= VALIDATION_MIN_BAND_COINS
                    and high_coins >= VALIDATION_MIN_BAND_COINS
                )
                rows.append(
                    {
                        "scope_id": threshold.scope_id,
                        "cohort": threshold.cohort,
                        "market_group": threshold.market_group,
                        "path_feature": threshold.path_feature,
                        "period": period,
                        "boundary_kind": boundary_kind,
                        "low_threshold": threshold.low_threshold,
                        "high_threshold": threshold.high_threshold,
                        "low_band_events": low_events,
                        "high_band_events": high_events,
                        "low_band_coins": low_coins,
                        "high_band_coins": high_coins,
                        "support_passed": bool(passed),
                        "outcome_values_used": False,
                        "direction_prediction": False,
                        "profit_optimization": False,
                    }
                )
    return DataFrame(rows)


def path_route_decisions(support: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for (scope_id, feature), group in support.groupby(
        ["scope_id", "path_feature"], observed=True
    ):
        periods = validation_periods_for_scope(str(scope_id))
        passed: dict[str, bool] = {}
        for boundary in (
            "actual_density_zone",
            "same_width_same_approach_no_level",
            "causal_shuffled_location",
        ):
            selected = group.loc[group["boundary_kind"].eq(boundary)]
            passed[boundary] = bool(
                set(selected.loc[selected["support_passed"].eq(True), "period"]) == set(periods)
            )
        generic = passed["actual_density_zone"] and passed[
            "same_width_same_approach_no_level"
        ]
        exact = generic and passed["causal_shuffled_location"]
        rows.append(
            {
                "scope_id": scope_id,
                "path_feature": feature,
                "actual_both_periods_supported": passed["actual_density_zone"],
                "synthetic_no_level_both_periods_supported": passed[
                    "same_width_same_approach_no_level"
                ],
                "causal_shuffle_both_periods_supported": passed[
                    "causal_shuffled_location"
                ],
                "generic_outcome_phase_eligible": generic,
                "exact_location_outcome_phase_eligible": exact,
                "reaction_outcomes_opened": False,
                "direction_prediction": False,
                "profit_optimization": False,
            }
        )
    return DataFrame(rows)


def plain_language_result(decisions: DataFrame) -> str:
    generic = decisions.loc[decisions["generic_outcome_phase_eligible"].eq(True)]
    exact = decisions.loc[decisions["exact_location_outcome_phase_eligible"].eq(True)]
    if generic.empty:
        return (
            "No development-frozen path feature retained enough actual and no-level "
            "synthetic arrivals in both validation periods. G5D is parked without "
            "opening reaction outcomes."
        )
    return (
        f"{len(generic)} scope/path cells have fair actual-versus-synthetic support "
        f"for the reaction-outcome phase. {len(exact)} also have fair shuffled-location "
        "support; only those cells may test an exact density-location residual."
    )


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import os


# ruff: noqa: E402
# Fix numerical-library thread counts before importing numpy/pandas.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import json
import sys
import time
from bisect import bisect_left, insort
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
    DEFAULT_MANIFEST,
    LARGE_ARTIFACT_ROOT,
    OUTPUT_ROOT,
    LevelSpec,
    atomic_write_json,
    atomic_write_parquet,
    extract_episode_events,
    future_path_matrices,
    load_manifest,
    matched_random_time_events,
    normalize_dates,
    numeric_array,
    prepare_base_market_frame,
    sha256_file,
    utc_now,
    validate_worker_count,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_localization import (  # noqa: E501
    AlignedLevel,
    aligned_selected_levels,
    causal_local_state,
    causal_market_context,
    nearest_state_pairs,
    outcome_metadata,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_anchored_vwap import (  # noqa: E501
    DESCRIPTIVE_OUTCOMES,
    MAX_MEDIAN_STATE_SMD,
    MAX_STATE_SMD,
    MIN_COHORT_PERIOD_COINS,
    MIN_COHORT_PERIOD_EPISODES,
    MIN_PAIR_PERIOD_EPISODES,
    OUTCOMES,
    OUTCOMES_BY_RESPONSE_WINDOW,
    PRIMARY_OUTCOMES,
    SEPARATION_HOURS_BY_RESPONSE_WINDOW,
    attach_state_and_absolute_outcomes,
    balance_quality,
    matched_pair_balance,
    pair_stem,
    stable_json_sha256,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_density import (  # noqa: E501
    HISTORY_HOURS,
    MAX_ZONES,
    SWING_MINIMUM_POINTS,
    ZONE_FAMILIES,
    DensityZoneSet,
    SwingPoints,
    causal_density_zones,
    confirmed_swing_points,
    connected_density_candidates,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_localization import (  # noqa: E501
    eligible_period_events,
)


FROZEN_BATCH = OUTPUT_ROOT / "generation1_review" / "g2_frozen_branch_batch.json"
REPORT_ROOT = OUTPUT_ROOT / "generation2_branches" / "g2e_density_zones"
ARTIFACT_ROOT = (
    LARGE_ARTIFACT_ROOT / "generation2_branches" / "g2e_density_zones"
)
DEPENDENCY_FILES = {
    name: Path(__file__).with_name(filename)
    for name, filename in {
        "generation0_event_helpers": "market_reaction_zone_generation0.py",
        "generation1_matching_helpers": "market_reaction_zone_generation1_localization.py",
        "generation2_summary_helpers": "market_reaction_zone_generation2_anchored_vwap.py",
        "generation2_density_construction": "market_reaction_zone_generation2_density.py",
        "generation2_period_helpers": "market_reaction_zone_generation2_localization.py",
    }.items()
}
OUTPUT_SCHEMA_VERSION = 1
FIXED_BIN_WIDTH_ATR = 0.25
SHIFT_MULTIPLIERS_ATR = (-1.0, 1.0)
REFERENCE_ZONE_HALF_WIDTH_ATR = 0.10
SHUFFLED_RESIDUAL_ROTATION_FRACTION = 1 / 3

CORE_STATE_FEATURES = (
    "state_local_return_1h",
    "state_local_return_4h",
    "state_local_return_24h",
    "state_local_atr_pct",
    "state_local_range_ratio",
    "state_local_volume_ratio",
    "state_local_pressure_6h",
    "state_local_rsi14",
    "state_local_bb_position",
    "state_local_bb_width_atr",
    "state_local_macd_hist_atr",
    "state_local_ema50_gap_atr",
    "state_btc_return_24h",
    "state_top10_breadth",
    "state_top10_mean_abs_return",
    "state_top10_return_dispersion",
)
DENSITY_STATE_FEATURES = (
    "state_surface_active_zone_count",
    "state_surface_zone_coverage_4atr",
    "state_all_density_active_zone_count",
    "state_density_surface_count_near_price_2atr",
    "state_all_density_zone_coverage_4atr",
    "state_reference_level_count_near_price_2atr",
)
STATE_FEATURES = (*CORE_STATE_FEATURES, *DENSITY_STATE_FEATURES)

VP_CONTROL_COLUMNS = {
    "current_vp_hvn": {"vp_hvn_above", "vp_hvn_below"},
    "current_vp_lvn": {"vp_lvn_above", "vp_lvn_below"},
    "current_vp_poc": {"vp_poc"},
}
CONTROL_NAMES = (
    "current_vp_hvn",
    "current_vp_lvn",
    "current_vp_poc",
    "simple_prior_high_same_history",
    "simple_prior_low_same_history",
    "shift_-1atr",
    "shift_+1atr",
    "shuffled_swing_residual_assignment",
    "random_eligible_zone",
    "same_state_no_level",
    "same_density_coverage_no_level",
)
ACTUAL_SCOPES = (
    "single_density_zone",
    "density_cluster",
    "density_plus_reference_cluster",
)


@dataclass(frozen=True)
class PairTask:
    pair: str
    manifest_path: str
    context_path: str
    run_id: str
    request_sha256: str
    overwrite: bool


@dataclass
class DensitySurface:
    family: str
    history_hours: int
    zones: DensityZoneSet
    shuffled_zones: DensityZoneSet | None = None

    @property
    def key(self) -> str:
        return f"{self.family}__{self.history_hours}h"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 2E causal density-zone reaction test. It compares "
            "direction-neutral market behaviour at fixed density zones and controls."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--technical-smoke", action="store_true")
    args = parser.parse_args(argv)
    validate_worker_count(args.workers)
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    if manifest["data"]["base_timeframe"] != "1h":
        raise ValueError("G2E reaction testing is fixed to a one-hour base timeframe.")
    pairs = select_pairs(manifest, args.pairs)
    compact_dir = REPORT_ROOT / args.run_id
    bulky_dir = ARTIFACT_ROOT / args.run_id
    pair_dir = bulky_dir / "pair_matches"
    compact_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)

    request = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "dependency_sha256": {
            name: sha256_file(path) for name, path in DEPENDENCY_FILES.items()
        },
        "manifest_sha256": sha256_file(manifest_path),
        "frozen_batch_sha256": sha256_file(FROZEN_BATCH),
        "pairs": pairs,
        "base_timeframe": "1h",
        "families": list(ZONE_FAMILIES),
        "history_hours": list(HISTORY_HOURS),
        "bin_width_atr": FIXED_BIN_WIDTH_ATR,
        "maximum_zones_per_surface": MAX_ZONES,
        "actual_scopes": list(ACTUAL_SCOPES),
        "controls": list(CONTROL_NAMES),
        "shift_multipliers_atr": list(SHIFT_MULTIPLIERS_ATR),
        "shuffled_swing_control": (
            "Rotate confirmed-swing residuals by one third inside each completed "
            "causal history, then rebuild price points from their mismatched pivot "
            "baselines and ATR values."
        ),
        "state_features": list(STATE_FEATURES),
        "primary_outcomes": list(PRIMARY_OUTCOMES),
        "descriptive_outcomes": list(DESCRIPTIVE_OUTCOMES),
        "outcomes_by_response_window": OUTCOMES_BY_RESPONSE_WINDOW,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    request_sha256 = stable_json_sha256(request)
    record_path = compact_dir / "g2e_reaction_run_record.json"
    record: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "status": "running",
        "started_at_utc": utc_now(),
        "orchestrator_pid": os.getpid(),
        "workers": args.workers,
        "technical_smoke_not_evidence": bool(args.technical_smoke),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "request": request,
        "request_sha256": request_sha256,
        "baseline": (
            "The same one-hour market states at current Volume Profile nodes, "
            "same-history prior highs/lows, shifted or shuffled artificial zones, "
            "and matched times without a real level."
        ),
        "hypothesis": (
            "At least one fixed density family and history marks repeated absolute "
            "movement, range, volume, pressure-change, dwell, or crossing behaviour "
            "beyond every applicable control."
        ),
        "pass_rule": (
            "A named family/history/scope must retain a trader-readable response "
            "against every applicable control in at least two chronological validation "
            "periods, at least five coins, fifty independent event pairs, acceptable "
            "state balance, and leave-one-coin-out stability."
        ),
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(record, record_path)
    try:
        validate_frozen_branch()
        context = causal_market_context(manifest)
        context_path = compact_dir / "g2e_causal_market_context.parquet"
        atomic_write_parquet(context, context_path)
        tasks = [
            PairTask(
                pair=pair,
                manifest_path=str(manifest_path),
                context_path=str(context_path),
                run_id=args.run_id,
                request_sha256=request_sha256,
                overwrite=args.overwrite,
            )
            for pair in pairs
        ]
        results = run_tasks(tasks, workers=args.workers)
        inventory = DataFrame(results)
        atomic_write_parquet(inventory, compact_dir / "g2e_reaction_pair_inventory.parquet")
        failures = [row for row in results if row.get("status") == "failed"]
        if failures:
            raise ValueError(
                f"{len(failures)} G2E pair task(s) failed; inspect the pair inventory."
            )
        matches = combine_pair_outputs(
            pair_dir=pair_dir,
            pairs=pairs,
            request_sha256=request_sha256,
        )
        independent = purge_density_matches(matches)
        independent_path = bulky_dir / "g2e_independent_pairs.parquet"
        atomic_write_parquet(independent, independent_path)
        pair_period = pair_period_results(independent)
        cohort_period = cohort_period_results(pair_period, independent)
        leave_one_out = leave_one_coin_out_results(pair_period)
        atomic_write_parquet(pair_period, compact_dir / "g2e_pair_period_results.parquet")
        atomic_write_parquet(cohort_period, compact_dir / "g2e_cohort_period_results.parquet")
        atomic_write_parquet(
            leave_one_out,
            compact_dir / "g2e_leave_one_coin_out.parquet",
        )
        integrity = integrity_record(
            results=results,
            matches=matches,
            independent=independent,
            technical_smoke=args.technical_smoke,
            requested_pairs=pairs,
        )
        atomic_write_json(integrity, compact_dir / "g2e_reaction_integrity.json")
        if not integrity["passed"]:
            raise ValueError("G2E reaction integrity checks failed; inspect the integrity file.")
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "pair_tasks": results,
                "matched_comparisons_before_overlap_purge": len(matches),
                "independent_comparisons": len(independent),
                "pair_period_rows": len(pair_period),
                "cohort_period_rows": len(cohort_period),
                "leave_one_coin_out_rows": len(leave_one_out),
                "independent_pairs_path": str(independent_path),
            }
        )
        atomic_write_json(record, record_path)
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


def validate_frozen_branch() -> None:
    payload = json.loads(FROZEN_BATCH.read_text(encoding="utf-8"))
    branches = {item.get("id") for item in payload.get("branches", [])}
    if "g2e_causal_swing_price_density_zones" not in branches:
        raise ValueError("The frozen Generation 2 batch does not contain G2E.")


def select_pairs(manifest: dict[str, Any], requested: str) -> list[str]:
    allowed = list(manifest["data"]["pairs"])
    if requested.strip().lower() == "all":
        return allowed
    lookup: dict[str, str] = {}
    for pair in allowed:
        lookup[pair.upper()] = pair
        lookup[pair.split("/", maxsplit=1)[0].upper()] = pair
    selected: list[str] = []
    for item in requested.split(","):
        key = item.strip().upper()
        if key not in lookup:
            raise ValueError(f"Pair {item!r} is not declared by the manifest.")
        if lookup[key] not in selected:
            selected.append(lookup[key])
    if not selected:
        raise ValueError("No G2E reaction pairs were selected.")
    return selected


def run_tasks(tasks: Sequence[PairTask], *, workers: int) -> list[dict[str, Any]]:
    if workers == 1:
        return [safe_build_pair(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(safe_build_pair, task): task for task in tasks}
        results = [future.result() for future in as_completed(future_map)]
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
    output_path = (
        ARTIFACT_ROOT
        / task.run_id
        / "pair_matches"
        / f"{pair_stem(task.pair)}.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file() and not task.overwrite:
        existing = validate_pair_output(
            output_path,
            pair=task.pair,
            request_sha256=task.request_sha256,
        )
        return {
            "pair": task.pair,
            "status": "existing",
            "matched_comparisons": len(existing),
            "seconds": round(time.perf_counter() - started, 3),
        }

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
    add_global_structure_state(
        state,
        base=base,
        surfaces=surfaces,
        density_levels=density_levels,
        density_widths=density_widths,
        density_surface_keys=density_surface_keys,
        reference_levels=reference_levels,
    )
    paths = future_path_matrices(base, max_horizon=24)
    rows: list[dict[str, Any]] = []
    actual_event_count = 0
    control_event_counts: dict[str, int] = {}

    for surface in surfaces:
        surface_state = state.copy()
        surface_state["state_surface_active_zone_count"] = (
            surface.zones.valid.sum(axis=1).astype(float)
        )
        surface_state["state_surface_zone_coverage_4atr"] = matrix_union_coverage(
            level_matrix=surface.zones.levels,
            width_matrix=surface.zones.half_widths,
            base_atr=numeric_array(base["base_atr"]),
            pre_close=numeric_array(base["pre_close"]),
        )
        actual = density_event_pool(
            surface=surface,
            zones=surface.zones,
            pair=task.pair,
            base=base,
            paths=paths,
            manifest=manifest,
            control="actual_density_zone",
        )
        actual = prepare_events(
            actual,
            state=surface_state,
            source_surface_key=surface.key,
            density_levels=density_levels,
            density_widths=density_widths,
            density_surface_keys=density_surface_keys,
            reference_levels=reference_levels,
            reference_widths=reference_widths,
            reference_groups=reference_groups,
        )
        if actual.empty:
            continue
        actual["actual_scope"] = classify_actual_scope(actual)
        actual_event_count += len(actual)
        controls = control_event_pools(
            surface=surface,
            actual=actual,
            pair=task.pair,
            base=base,
            paths=paths,
            manifest=manifest,
            state=surface_state,
            references=references,
            density_levels=density_levels,
            density_widths=density_widths,
            density_surface_keys=density_surface_keys,
            reference_levels=reference_levels,
            reference_widths=reference_widths,
            reference_groups=reference_groups,
        )
        for control_name, (control_events, state_features) in controls.items():
            control_event_counts[control_name] = (
                control_event_counts.get(control_name, 0) + len(control_events)
            )
            for actual_scope, scoped_actual in actual.groupby(
                "actual_scope",
                observed=True,
                sort=True,
            ):
                rows.extend(
                    matched_outcome_rows(
                        actual=scoped_actual.reset_index(drop=True),
                        control=control_events,
                        pair=task.pair,
                        surface=surface,
                        actual_scope=str(actual_scope),
                        control_name=control_name,
                        state_features=state_features,
                    )
                )

    output = DataFrame(rows)
    if output.empty:
        raise ValueError(f"No matchable G2E outcome pairs for {task.pair}.")
    output["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    output["run_request_sha256"] = task.request_sha256
    atomic_write_parquet(output, output_path)
    return {
        "pair": task.pair,
        "status": "completed",
        "source_rows": len(base),
        "density_surfaces": len(surfaces),
        "actual_contact_episodes": actual_event_count,
        "control_event_counts": control_event_counts,
        "matched_comparisons": len(output),
        "families": sorted(output["density_family"].unique().tolist()),
        "histories": sorted(output["history_hours"].astype(int).unique().tolist()),
        "actual_scopes": sorted(output["actual_scope"].unique().tolist()),
        "controls": sorted(output["control"].unique().tolist()),
        "seconds": round(time.perf_counter() - started, 3),
    }


def build_density_surfaces(base: DataFrame) -> list[DensitySurface]:
    swings = confirmed_swing_points(base)
    surfaces: list[DensitySurface] = []
    for family in ZONE_FAMILIES:
        for history_hours in HISTORY_HOURS:
            zones = causal_density_zones(
                base,
                family=family,
                history_hours=history_hours,
                bin_width_atr=FIXED_BIN_WIDTH_ATR,
                swings=swings,
            )
            shuffled = (
                causal_shuffled_swing_zones(
                    base,
                    history_hours=history_hours,
                    bin_width_atr=FIXED_BIN_WIDTH_ATR,
                    swings=swings,
                )
                if family == "confirmed_swing_price_density"
                else None
            )
            surfaces.append(
                DensitySurface(
                    family=family,
                    history_hours=history_hours,
                    zones=zones,
                    shuffled_zones=shuffled,
                )
            )
    return surfaces


def causal_shuffled_swing_zones(
    frame: DataFrame,
    *,
    history_hours: int,
    bin_width_atr: float,
    swings: SwingPoints,
) -> DensityZoneSet:
    dates = normalize_dates(frame["date"])
    date_ns = dates.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    current_atr = numeric_array(frame["base_atr"])
    current_reference = numeric_array(frame["pre_close"])
    pivot_baseline = numeric_array(frame["pre_close"])[swings.pivot_positions]
    pivot_atr = numeric_array(frame["base_atr"])[swings.pivot_positions]
    residual = np.full(len(swings.prices), np.nan, dtype=np.float64)
    residual_valid = (
        np.isfinite(swings.prices)
        & np.isfinite(pivot_baseline)
        & np.isfinite(pivot_atr)
        & (pivot_atr > 0.0)
    )
    residual[residual_valid] = (
        swings.prices[residual_valid] - pivot_baseline[residual_valid]
    ) / pivot_atr[residual_valid]
    levels = np.full((len(frame), MAX_ZONES), np.nan, dtype=np.float64)
    half_widths = np.full_like(levels, np.nan)
    support_counts = np.full_like(levels, np.nan)
    source_counts = np.full(len(frame), np.nan, dtype=np.float64)
    history_ns = int(pd.Timedelta(hours=history_hours).value)
    swing_dates_ns = date_ns[swings.pivot_positions]
    minimum = SWING_MINIMUM_POINTS[history_hours]
    for position in range(len(frame)):
        if not np.isfinite(current_atr[position]) or current_atr[position] <= 0.0:
            continue
        history_start_ns = date_ns[position] - history_ns
        if date_ns[0] > history_start_ns:
            continue
        selected = (
            (swings.available_positions <= position)
            & (swing_dates_ns >= history_start_ns)
            & (swing_dates_ns < date_ns[position])
            & residual_valid
        )
        indexes = np.flatnonzero(selected)
        if len(indexes) < minimum:
            continue
        selected_residual = residual[indexes]
        rotation = max(
            1,
            int(np.floor(len(indexes) * SHUFFLED_RESIDUAL_ROTATION_FRACTION)),
        )
        reconstructed = (
            pivot_baseline[indexes]
            + np.roll(selected_residual, rotation) * pivot_atr[indexes]
        )
        reconstructed = reconstructed[
            np.isfinite(reconstructed) & (reconstructed > 0.0)
        ]
        if len(reconstructed) < minimum:
            continue
        source_counts[position] = float(len(reconstructed))
        candidates = connected_density_candidates(
            reconstructed,
            atr=float(current_atr[position]),
            reference=float(current_reference[position]),
            family="confirmed_swing_price_density",
            history_hours=history_hours,
            bin_width_atr=bin_width_atr,
        )
        for rank, candidate in enumerate(candidates[:MAX_ZONES]):
            levels[position, rank] = candidate[0]
            half_widths[position, rank] = candidate[1]
            support_counts[position, rank] = candidate[2]
    valid = np.isfinite(levels) & np.isfinite(half_widths) & (levels > 0.0)
    return DensityZoneSet(
        family="confirmed_swing_price_density_shuffled_residuals",
        history_hours=history_hours,
        bin_width_atr=bin_width_atr,
        levels=levels,
        half_widths=half_widths,
        valid=valid,
        support_counts=support_counts,
        source_counts=source_counts,
    )


def selected_reference_levels(
    *,
    pair: str,
    base: DataFrame,
    manifest_path: Path,
) -> list[AlignedLevel]:
    selected = aligned_selected_levels(
        pair=pair,
        base=base,
        manifest_path=manifest_path,
        timeframes=("1h",),
    )
    wanted = {
        *{column for columns in VP_CONTROL_COLUMNS.values() for column in columns},
        *(
            f"generic_rolling_{side}_{history}"
            for side in ("high", "low")
            for history in HISTORY_HOURS
        ),
    }
    return [item for item in selected if item.spec.column in wanted]


def density_matrices(
    surfaces: Sequence[DensitySurface],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    levels = np.column_stack([surface.zones.levels for surface in surfaces])
    widths = np.column_stack([surface.zones.half_widths for surface in surfaces])
    keys = np.asarray(
        [surface.key for surface in surfaces for _ in range(MAX_ZONES)],
        dtype=object,
    )
    return levels, widths, keys


def reference_matrices(
    references: Sequence[AlignedLevel],
    *,
    base_atr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not references:
        empty = np.empty((len(base_atr), 0), dtype=np.float64)
        return empty, empty.copy(), np.empty(0, dtype=object)
    levels = np.column_stack(
        [np.where(item.valid, item.level, np.nan) for item in references]
    )
    widths = np.repeat(
        (REFERENCE_ZONE_HALF_WIDTH_ATR * base_atr)[:, None],
        len(references),
        axis=1,
    )
    groups = np.asarray(
        [reference_group(item.spec.column) for item in references],
        dtype=object,
    )
    return levels, widths, groups


def reference_group(column: str) -> str:
    for name, columns in VP_CONTROL_COLUMNS.items():
        if column in columns:
            return name
    if column.startswith("generic_rolling_high_"):
        return "simple_prior_high"
    if column.startswith("generic_rolling_low_"):
        return "simple_prior_low"
    raise ValueError(f"Unsupported G2E reference level column: {column}")


def add_global_structure_state(
    state: DataFrame,
    *,
    base: DataFrame,
    surfaces: Sequence[DensitySurface],
    density_levels: np.ndarray,
    density_widths: np.ndarray,
    density_surface_keys: np.ndarray,
    reference_levels: np.ndarray,
) -> None:
    base_atr = numeric_array(base["base_atr"])
    pre_close = numeric_array(base["pre_close"])
    valid_density = np.isfinite(density_levels) & np.isfinite(density_widths)
    state["state_all_density_active_zone_count"] = valid_density.sum(axis=1).astype(float)
    near = (
        valid_density
        & np.isfinite(base_atr[:, None])
        & (base_atr[:, None] > 0.0)
        & (np.abs(density_levels - pre_close[:, None]) <= 2.0 * base_atr[:, None])
    )
    surface_count = np.zeros(len(base), dtype=float)
    for key in sorted(set(density_surface_keys)):
        surface_count += near[:, density_surface_keys == key].any(axis=1).astype(float)
    state["state_density_surface_count_near_price_2atr"] = surface_count
    state["state_all_density_zone_coverage_4atr"] = matrix_union_coverage(
        level_matrix=density_levels,
        width_matrix=density_widths,
        base_atr=base_atr,
        pre_close=pre_close,
    )
    reference_valid = np.isfinite(reference_levels)
    reference_near = (
        reference_valid
        & np.isfinite(base_atr[:, None])
        & (base_atr[:, None] > 0.0)
        & (np.abs(reference_levels - pre_close[:, None]) <= 2.0 * base_atr[:, None])
    )
    state["state_reference_level_count_near_price_2atr"] = reference_near.sum(
        axis=1
    ).astype(float)


def matrix_union_coverage(
    *,
    level_matrix: np.ndarray,
    width_matrix: np.ndarray,
    base_atr: np.ndarray,
    pre_close: np.ndarray,
) -> np.ndarray:
    output = np.full(len(base_atr), np.nan, dtype=np.float64)
    for row in range(len(base_atr)):
        atr = base_atr[row]
        centre = pre_close[row]
        if not np.isfinite(atr) or atr <= 0.0 or not np.isfinite(centre):
            continue
        valid = np.isfinite(level_matrix[row]) & np.isfinite(width_matrix[row])
        intervals: list[tuple[float, float]] = []
        lower_bound = centre - 2.0 * atr
        upper_bound = centre + 2.0 * atr
        for level, width in zip(
            level_matrix[row, valid],
            width_matrix[row, valid],
            strict=True,
        ):
            lower = max(lower_bound, float(level - width))
            upper = min(upper_bound, float(level + width))
            if lower < upper:
                intervals.append((lower, upper))
        if not intervals:
            output[row] = 0.0
            continue
        intervals.sort()
        covered = 0.0
        active_lower, active_upper = intervals[0]
        for lower, upper in intervals[1:]:
            if lower <= active_upper:
                active_upper = max(active_upper, upper)
            else:
                covered += active_upper - active_lower
                active_lower, active_upper = lower, upper
        covered += active_upper - active_lower
        output[row] = covered / (4.0 * atr)
    return output


def surface_spec(surface: DensitySurface, *, suffix: str) -> LevelSpec:
    return LevelSpec(
        name=f"{surface.family}_{surface.history_hours}h_{suffix}",
        family=surface.family,
        batch="g2e",
        column=f"g2e_{surface.key}_{suffix}",
    )


def density_event_pool(
    *,
    surface: DensitySurface,
    zones: DensityZoneSet,
    pair: str,
    base: DataFrame,
    paths: dict[str, np.ndarray],
    manifest: dict[str, Any],
    control: str,
    shift_atr: float = 0.0,
) -> DataFrame:
    base_atr = numeric_array(base["base_atr"])
    dates = normalize_dates(base["date"])
    source_open = dates - pd.Timedelta(hours=surface.history_hours)
    frames: list[DataFrame] = []
    for rank in range(MAX_ZONES):
        level = zones.levels[:, rank] + shift_atr * base_atr
        valid = zones.valid[:, rank] & np.isfinite(level) & (level > 0.0)
        events = extract_episode_events(
            merged=base,
            paths=paths,
            spec=surface_spec(surface, suffix=f"rank{rank}"),
            pair=pair,
            timeframe="1h_causal_density",
            level=level,
            valid=valid,
            half_width=zones.half_widths[:, rank],
            source_available=dates,
            source_open=source_open,
            control=control,
            zone_method="native_connected_density_zone",
            horizons=(1, 4, 24),
            event_kind="contact",
        )
        if events.empty:
            continue
        indexes = events["base_index"].to_numpy(dtype=np.int64)
        source_count = zones.source_counts[indexes]
        support = zones.support_counts[indexes, rank]
        support_fraction = np.full(len(events), np.nan, dtype=np.float64)
        supported = np.isfinite(source_count) & (source_count > 0.0)
        support_fraction[supported] = support[supported] / source_count[supported]
        events["zone_rank"] = rank
        events["zone_support_count"] = support
        events["zone_source_count"] = source_count
        events["zone_support_fraction"] = support_fraction
        events["reference_level_column"] = None
        frames.append(events)
    if not frames:
        return DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined.sort_values(
            ["event_time", "pre_distance_atr", "zone_rank"],
            kind="stable",
        )
        .drop_duplicates("event_time", keep="first")
        .reset_index(drop=True)
    )
    return eligible_period_events(combined, manifest, embargo_hours=24)


def reference_event_pool(
    *,
    references: Sequence[AlignedLevel],
    surface: DensitySurface,
    pair: str,
    base: DataFrame,
    paths: dict[str, np.ndarray],
    manifest: dict[str, Any],
    control: str,
) -> DataFrame:
    if not references:
        return DataFrame()
    width = surface_reference_width(surface, numeric_array(base["base_atr"]))
    frames: list[DataFrame] = []
    for item in references:
        events = extract_episode_events(
            merged=base,
            paths=paths,
            spec=item.spec,
            pair=pair,
            timeframe="1h_causal_reference",
            level=item.level,
            valid=item.valid,
            half_width=width,
            source_available=item.source_available,
            source_open=item.source_open,
            control=control,
            zone_method="density_width_matched_reference",
            horizons=(1, 4, 24),
            event_kind="contact",
        )
        if events.empty:
            continue
        events["zone_rank"] = -1
        events["zone_support_count"] = np.nan
        events["zone_source_count"] = np.nan
        events["zone_support_fraction"] = np.nan
        events["reference_level_column"] = item.spec.column
        frames.append(events)
    if not frames:
        return DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined.sort_values(
            ["event_time", "pre_distance_atr", "reference_level_column"],
            kind="stable",
        )
        .drop_duplicates("event_time", keep="first")
        .reset_index(drop=True)
    )
    return eligible_period_events(combined, manifest, embargo_hours=24)


def surface_reference_width(surface: DensitySurface, base_atr: np.ndarray) -> np.ndarray:
    widths = DataFrame(surface.zones.half_widths).median(axis=1, skipna=True).to_numpy(
        dtype=float
    )
    fallback = REFERENCE_ZONE_HALF_WIDTH_ATR * base_atr
    valid = np.isfinite(widths) & (widths > 0.0)
    return np.where(valid, widths, fallback)


def prepare_events(
    events: DataFrame,
    *,
    state: DataFrame,
    source_surface_key: str | None,
    density_levels: np.ndarray,
    density_widths: np.ndarray,
    density_surface_keys: np.ndarray,
    reference_levels: np.ndarray,
    reference_widths: np.ndarray,
    reference_groups: np.ndarray,
) -> DataFrame:
    if events.empty:
        return events.copy()
    output = attach_state_and_absolute_outcomes(events, state)
    return attach_structure_overlaps(
        output,
        source_surface_key=source_surface_key,
        density_levels=density_levels,
        density_widths=density_widths,
        density_surface_keys=density_surface_keys,
        reference_levels=reference_levels,
        reference_widths=reference_widths,
        reference_groups=reference_groups,
    )


def attach_structure_overlaps(
    events: DataFrame,
    *,
    source_surface_key: str | None,
    density_levels: np.ndarray,
    density_widths: np.ndarray,
    density_surface_keys: np.ndarray,
    reference_levels: np.ndarray,
    reference_widths: np.ndarray,
    reference_groups: np.ndarray,
) -> DataFrame:
    if events.empty:
        return events.copy()
    output = events.copy()
    indexes = output["base_index"].to_numpy(dtype=np.int64)
    event_level = numeric_array(output["level_price"])
    event_width = numeric_array(output["zone_half_width"])
    local_density = density_levels[indexes]
    local_density_width = density_widths[indexes]
    density_overlap = np.isfinite(local_density) & np.isfinite(local_density_width) & (
        np.abs(local_density - event_level[:, None])
        <= local_density_width + event_width[:, None]
    )
    output["overlap_any_density_zone"] = density_overlap.any(axis=1)
    output["overlap_density_zone_count"] = density_overlap.sum(axis=1).astype(np.int16)
    same_surface_count = np.zeros(len(output), dtype=np.int16)
    other_surface_count = np.zeros(len(output), dtype=np.int16)
    for key in sorted(set(density_surface_keys)):
        count = density_overlap[:, density_surface_keys == key].sum(axis=1).astype(
            np.int16
        )
        if source_surface_key is not None and key == source_surface_key:
            same_surface_count = count
        else:
            other_surface_count += (count > 0).astype(np.int16)
    output["overlap_source_surface_zone_count"] = same_surface_count
    output["overlap_other_density_surface_count"] = other_surface_count
    if source_surface_key is None:
        other_density_zone_count = density_overlap.sum(axis=1)
    else:
        source_mask = density_surface_keys == source_surface_key
        other_density_zone_count = density_overlap[:, ~source_mask].sum(axis=1)
        other_density_zone_count += np.maximum(
            density_overlap[:, source_mask].sum(axis=1) - 1,
            0,
        )
    output["overlap_other_density_zone_count"] = other_density_zone_count.astype(
        np.int16
    )

    local_reference = reference_levels[indexes]
    local_reference_width = reference_widths[indexes]
    reference_overlap = np.isfinite(local_reference) & np.isfinite(
        local_reference_width
    ) & (
        np.abs(local_reference - event_level[:, None])
        <= local_reference_width + event_width[:, None]
    )
    output["overlap_any_reference_level"] = reference_overlap.any(axis=1)
    output["overlap_reference_level_count"] = reference_overlap.sum(axis=1).astype(
        np.int16
    )
    group_count = np.zeros(len(output), dtype=np.int16)
    for group in sorted(set(reference_groups)):
        group_count += reference_overlap[:, reference_groups == group].any(axis=1).astype(
            np.int16
        )
    output["overlap_reference_group_count"] = group_count
    output["overlap_any_real_level"] = (
        output["overlap_any_density_zone"] | output["overlap_any_reference_level"]
    )
    return output


def classify_actual_scope(actual: DataFrame) -> np.ndarray:
    reference = actual["overlap_any_reference_level"].to_numpy(dtype=bool)
    density_cluster = actual["overlap_other_density_zone_count"].to_numpy(dtype=int) > 0
    return np.select(
        [reference, density_cluster],
        ["density_plus_reference_cluster", "density_cluster"],
        default="single_density_zone",
    )


def control_event_pools(
    *,
    surface: DensitySurface,
    actual: DataFrame,
    pair: str,
    base: DataFrame,
    paths: dict[str, np.ndarray],
    manifest: dict[str, Any],
    state: DataFrame,
    references: Sequence[AlignedLevel],
    density_levels: np.ndarray,
    density_widths: np.ndarray,
    density_surface_keys: np.ndarray,
    reference_levels: np.ndarray,
    reference_widths: np.ndarray,
    reference_groups: np.ndarray,
) -> dict[str, tuple[DataFrame, tuple[str, ...]]]:
    controls: dict[str, tuple[DataFrame, tuple[str, ...]]] = {}

    def prepared(events: DataFrame) -> DataFrame:
        return prepare_events(
            events,
            state=state,
            source_surface_key=None,
            density_levels=density_levels,
            density_widths=density_widths,
            density_surface_keys=density_surface_keys,
            reference_levels=reference_levels,
            reference_widths=reference_widths,
            reference_groups=reference_groups,
        )

    for shift in SHIFT_MULTIPLIERS_ATR:
        name = f"shift_{shift:+g}atr"
        events = density_event_pool(
            surface=surface,
            zones=surface.zones,
            pair=pair,
            base=base,
            paths=paths,
            manifest=manifest,
            control=name,
            shift_atr=shift,
        )
        events = prepared(events)
        controls[name] = (
            without_density_overlap(events),
            STATE_FEATURES,
        )

    if surface.shuffled_zones is not None:
        events = density_event_pool(
            surface=surface,
            zones=surface.shuffled_zones,
            pair=pair,
            base=base,
            paths=paths,
            manifest=manifest,
            control="shuffled_swing_residual_assignment",
        )
        events = prepared(events)
        controls["shuffled_swing_residual_assignment"] = (
            without_density_overlap(events),
            STATE_FEATURES,
        )

    for control_name, columns in VP_CONTROL_COLUMNS.items():
        selected = [item for item in references if item.spec.column in columns]
        events = reference_event_pool(
            references=selected,
            surface=surface,
            pair=pair,
            base=base,
            paths=paths,
            manifest=manifest,
            control=control_name,
        )
        events = prepared(events)
        controls[control_name] = (
            without_density_overlap(events),
            STATE_FEATURES,
        )

    for side in ("high", "low"):
        control_name = f"simple_prior_{side}_same_history"
        column = f"generic_rolling_{side}_{surface.history_hours}"
        selected = [item for item in references if item.spec.column == column]
        events = reference_event_pool(
            references=selected,
            surface=surface,
            pair=pair,
            base=base,
            paths=paths,
            manifest=manifest,
            control=control_name,
        )
        events = prepared(events)
        controls[control_name] = (
            without_density_overlap(events),
            STATE_FEATURES,
        )

    random_events = matched_random_time_events(
        merged=base,
        paths=paths,
        actual=actual,
        spec=surface_spec(surface, suffix="random"),
        pair=pair,
        timeframe="1h_causal_density",
        zone_method="native_connected_density_zone",
        horizons=(1, 4, 24),
    )
    random_events = eligible_period_events(random_events, manifest, embargo_hours=24)
    if not random_events.empty:
        random_events["zone_rank"] = -1
        random_events["zone_support_count"] = np.nan
        random_events["zone_source_count"] = np.nan
        random_events["zone_support_fraction"] = np.nan
        random_events["reference_level_column"] = None
    random_events = prepared(random_events)
    controls["random_eligible_zone"] = (random_events, CORE_STATE_FEATURES)
    if random_events.empty:
        no_level = random_events.copy()
    else:
        no_level = random_events.loc[~random_events["overlap_any_real_level"]].reset_index(
            drop=True
        )
    controls["same_state_no_level"] = (no_level, CORE_STATE_FEATURES)
    controls["same_density_coverage_no_level"] = (no_level, STATE_FEATURES)
    return controls


def without_density_overlap(events: DataFrame) -> DataFrame:
    if events.empty:
        return events.copy()
    if "overlap_any_density_zone" not in events:
        raise ValueError("Prepared G2E control events lack density-overlap state.")
    return events.loc[~events["overlap_any_density_zone"]].reset_index(drop=True)


def matched_outcome_rows(
    *,
    actual: DataFrame,
    control: DataFrame,
    pair: str,
    surface: DensitySurface,
    actual_scope: str,
    control_name: str,
    state_features: Sequence[str],
) -> list[dict[str, Any]]:
    if actual.empty or control.empty:
        return []
    required = {
        "event_time",
        "period",
        "approach_state",
        "pre_distance_atr",
        "overlap_density_zone_count",
        "overlap_reference_level_count",
        *STATE_FEATURES,
        *OUTCOMES,
    }
    for label, frame in (("actual", actual), ("control", control)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{label} G2E event pool lacks columns: {missing}")
    rows: list[dict[str, Any]] = []
    for period in sorted(set(actual["period"]).intersection(control["period"])):
        for approach in (
            "from_below",
            "from_above",
            "already_inside_or_unclear",
        ):
            left = actual.loc[
                actual["period"].eq(period)
                & actual["approach_state"].eq(approach)
            ].reset_index(drop=True)
            right = control.loc[
                control["period"].eq(period)
                & control["approach_state"].eq(approach)
            ].reset_index(drop=True)
            if left.empty or right.empty:
                continue
            for response_window, outcomes in OUTCOMES_BY_RESPONSE_WINDOW.items():
                independence_hours = SEPARATION_HOURS_BY_RESPONSE_WINDOW[response_window]
                pairs, audit = nearest_state_pairs(
                    left,
                    right,
                    state_columns=state_features,
                    pre_distance_atr_caliper=0.10,
                    minimum_event_separation_hours=independence_hours,
                )
                for left_position, right_position, distance in pairs:
                    left_row = left.iloc[left_position]
                    right_row = right.iloc[right_position]
                    separation = abs(
                        (
                            pd.Timestamp(left_row["event_time"])
                            - pd.Timestamp(right_row["event_time"])
                        ).total_seconds()
                        / 3600.0
                    )
                    row: dict[str, Any] = {
                        "pair": pair,
                        "route_id": "g2e_causal_swing_price_density_zones",
                        "density_family": surface.family,
                        "history_hours": surface.history_hours,
                        "level_name": surface.key,
                        "actual_scope": actual_scope,
                        "control": control_name,
                        "period": str(period),
                        "approach_state": approach,
                        "response_window": response_window,
                        "actual_event_time": pd.Timestamp(left_row["event_time"]),
                        "control_event_time": pd.Timestamp(right_row["event_time"]),
                        "actual_base_index": int(left_row["base_index"]),
                        "control_base_index": int(right_row["base_index"]),
                        "actual_source_available_at": left_row.get(
                            "source_available_at"
                        ),
                        "control_source_available_at": right_row.get(
                            "source_available_at"
                        ),
                        "actual_source_open": left_row.get("source_open"),
                        "control_source_open": right_row.get("source_open"),
                        "actual_zone_rank": int(left_row.get("zone_rank", -1)),
                        "actual_zone_support_fraction": float(
                            left_row.get("zone_support_fraction", np.nan)
                        ),
                        "control_reference_level_column": right_row.get(
                            "reference_level_column"
                        ),
                        "match_distance": float(distance),
                        "pre_distance_atr_abs_difference": abs(
                            float(left_row["pre_distance_atr"])
                            - float(right_row["pre_distance_atr"])
                        ),
                        "event_separation_hours": separation,
                        "actual_overlap_density_zone_count": int(
                            left_row["overlap_density_zone_count"]
                        ),
                        "control_overlap_density_zone_count": int(
                            right_row["overlap_density_zone_count"]
                        ),
                        "actual_overlap_other_density_surface_count": int(
                            left_row["overlap_other_density_surface_count"]
                        ),
                        "control_overlap_other_density_surface_count": int(
                            right_row["overlap_other_density_surface_count"]
                        ),
                        "actual_overlap_reference_level_count": int(
                            left_row["overlap_reference_level_count"]
                        ),
                        "control_overlap_reference_level_count": int(
                            right_row["overlap_reference_level_count"]
                        ),
                        "matching_state_features": ";".join(state_features),
                        "eligible_actual_events": int(audit["eligible_actual"]),
                        "eligible_control_events": int(audit["eligible_control"]),
                        "geometry_eligible_actual_events": int(
                            audit["geometry_eligible_actual"]
                        ),
                        "state_matchable_actual_events": int(
                            audit["state_matchable_actual"]
                        ),
                        "direction_prediction": False,
                        "profit_optimization": False,
                    }
                    for outcome in outcomes:
                        actual_value = float(left_row[outcome])
                        control_value = float(right_row[outcome])
                        row[f"actual__{outcome}"] = actual_value
                        row[f"control__{outcome}"] = control_value
                        row[f"delta__{outcome}"] = actual_value - control_value
                    for feature in STATE_FEATURES:
                        row[f"actual_state__{feature}"] = float(left_row[feature])
                        row[f"control_state__{feature}"] = float(right_row[feature])
                    rows.append(row)
    return rows


def purge_density_matches(matches: DataFrame) -> DataFrame:
    group_columns = (
        "route_id",
        "density_family",
        "history_hours",
        "actual_scope",
        "control",
        "pair",
        "period",
        "approach_state",
        "response_window",
    )
    if matches.empty:
        return DataFrame(
            columns=[
                *matches.columns,
                "independent_selection_order",
                "independence_hours",
            ]
        )
    observed = set(matches["response_window"].dropna().astype(str))
    missing = observed.difference(SEPARATION_HOURS_BY_RESPONSE_WINDOW)
    if missing:
        raise ValueError(
            f"Missing G2E independence horizon for response windows: {sorted(missing)}"
        )

    # The shared reference performs the same stable priority sort separately in
    # thousands of small groups. One global stable sort preserves that within-group
    # order, while a sorted timestamp index makes the identical greedy gap check fast.
    priority_columns = (
        "match_distance",
        "pre_distance_atr_abs_difference",
        "actual_event_time",
        "control_event_time",
        "level_name",
        "approach_state",
    )
    ordered = matches.sort_values(
        [*group_columns, *priority_columns],
        kind="stable",
    ).drop_duplicates(
        [*group_columns, "actual_event_time", "control_event_time"],
        keep="first",
    )
    selected: list[DataFrame] = []
    for key, group in ordered.groupby(
        list(group_columns),
        observed=True,
        sort=False,
    ):
        response_window = str(key[-1])
        hours = SEPARATION_HOURS_BY_RESPONSE_WINDOW[response_window]
        gap_ns = int(pd.Timedelta(hours=hours).value)
        indexes = group.index.to_numpy()
        left_times = pd.to_datetime(
            group["actual_event_time"], utc=True, errors="raise"
        ).astype("int64").to_numpy()
        right_times = pd.to_datetime(
            group["control_event_time"], utc=True, errors="raise"
        ).astype("int64").to_numpy()
        used_times: list[int] = []
        retained_indexes: list[Any] = []
        for index, left_time, right_time in zip(
            indexes,
            left_times,
            right_times,
            strict=True,
        ):
            left_position = bisect_left(used_times, left_time - gap_ns)
            right_position = bisect_left(used_times, right_time - gap_ns)
            left_overlaps = (
                left_position < len(used_times)
                and used_times[left_position] <= left_time + gap_ns
            )
            right_overlaps = (
                right_position < len(used_times)
                and used_times[right_position] <= right_time + gap_ns
            )
            if left_overlaps or right_overlaps:
                continue
            retained_indexes.append(index)
            insort(used_times, int(left_time))
            insort(used_times, int(right_time))
        if retained_indexes:
            retained = ordered.loc[retained_indexes].copy()
            retained["independent_selection_order"] = np.arange(
                1,
                len(retained) + 1,
                dtype=np.int32,
            )
            retained["independence_hours"] = hours
            selected.append(retained)
    if not selected:
        return DataFrame(
            columns=[
                *matches.columns,
                "independent_selection_order",
                "independence_hours",
            ]
        )
    return pd.concat(selected, ignore_index=True)


def validate_pair_output(
    path: Path,
    *,
    pair: str,
    request_sha256: str,
) -> DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "pair",
        "output_schema_version",
        "run_request_sha256",
        "density_family",
        "history_hours",
        "actual_scope",
        "control",
        "response_window",
        "actual_event_time",
        "control_event_time",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Existing G2E pair output lacks columns {missing}: {path}")
    if frame.empty or set(frame["pair"].astype(str)) != {pair}:
        raise ValueError(f"Existing G2E pair output has the wrong pair: {path}")
    if set(frame["output_schema_version"].astype(int)) != {OUTPUT_SCHEMA_VERSION}:
        raise ValueError(f"Existing G2E pair output has the wrong schema: {path}")
    if set(frame["run_request_sha256"].astype(str)) != {request_sha256}:
        raise ValueError(f"Existing G2E pair output has the wrong request hash: {path}")
    return frame


def combine_pair_outputs(
    *,
    pair_dir: Path,
    pairs: Sequence[str],
    request_sha256: str,
) -> DataFrame:
    frames = [
        validate_pair_output(
            pair_dir / f"{pair_stem(pair)}.parquet",
            pair=pair,
            request_sha256=request_sha256,
        )
        for pair in pairs
    ]
    return pd.concat(frames, ignore_index=True)


def pair_period_results(independent_pairs: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = [
        "route_id",
        "density_family",
        "history_hours",
        "actual_scope",
        "control",
        "pair",
        "period",
        "response_window",
    ]
    for key, group in independent_pairs.groupby(
        group_columns,
        observed=True,
        dropna=False,
        sort=False,
    ):
        response_window = str(key[7])
        balance = declared_matching_balance(group)
        for outcome in OUTCOMES_BY_RESPONSE_WINDOW[response_window]:
            delta = pd.to_numeric(group[f"delta__{outcome}"], errors="coerce")
            actual = pd.to_numeric(group[f"actual__{outcome}"], errors="coerce")
            control = pd.to_numeric(group[f"control__{outcome}"], errors="coerce")
            valid = delta.notna() & actual.notna() & control.notna()
            if not valid.any():
                continue
            rows.append(
                {
                    "route_id": key[0],
                    "density_family": key[1],
                    "history_hours": int(key[2]),
                    "actual_scope": key[3],
                    "control": key[4],
                    "pair": key[5],
                    "period": key[6],
                    "response_window": response_window,
                    "outcome": outcome,
                    "metric_family": outcome_metadata(outcome)[0],
                    "horizon_hours": outcome_metadata(outcome)[1],
                    "primary_outcome": outcome in PRIMARY_OUTCOMES,
                    "independent_event_pairs": int(valid.sum()),
                    "pair_period_coverage_eligible": int(valid.sum())
                    >= MIN_PAIR_PERIOD_EPISODES,
                    "actual_mean": float(actual.loc[valid].mean()),
                    "control_mean": float(control.loc[valid].mean()),
                    "delta_mean": float(delta.loc[valid].mean()),
                    "delta_median": float(delta.loc[valid].median()),
                    "positive_fraction": float(delta.loc[valid].gt(0.0).mean()),
                    "match_distance_median": float(group["match_distance"].median()),
                    "pre_distance_atr_abs_difference_max": float(
                        group["pre_distance_atr_abs_difference"].max()
                    ),
                    "event_separation_hours_min": float(
                        group["event_separation_hours"].min()
                    ),
                    "max_absolute_state_smd": balance["max_absolute_smd"],
                    "median_absolute_state_smd": balance["median_absolute_smd"],
                    "state_features_scored": balance["features_scored"],
                    "balance_quality": balance_quality(
                        balance["max_absolute_smd"],
                        balance["median_absolute_smd"],
                    ),
                    "direction_prediction": False,
                    "profit_optimization": False,
                }
            )
    return DataFrame(rows)


def cohort_period_results(
    pair_period: DataFrame,
    independent_pairs: DataFrame,
) -> DataFrame:
    if pair_period.empty:
        return DataFrame()
    eligible = pair_period.loc[pair_period["pair_period_coverage_eligible"]].copy()
    rows: list[dict[str, Any]] = []
    event_group_columns = [
        "route_id",
        "density_family",
        "history_hours",
        "actual_scope",
        "control",
        "period",
        "response_window",
    ]
    event_groups = {
        key: group
        for key, group in independent_pairs.groupby(
            event_group_columns,
            observed=True,
            dropna=False,
            sort=False,
        )
    }
    matched_event_cache: dict[tuple[Any, ...], DataFrame] = {}
    balance_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    group_columns = [*event_group_columns, "outcome"]
    for key, group in eligible.groupby(
        group_columns,
        observed=True,
        dropna=False,
        sort=False,
    ):
        coins = sorted(group["pair"].astype(str).unique())
        outcome = str(key[7])
        event_key = tuple(key[:7])
        cache_key = (*event_key, *coins)
        if cache_key not in matched_event_cache:
            if event_key not in event_groups:
                raise ValueError(
                    f"G2E cohort summary cannot resolve event group {event_key}."
                )
            source_events = event_groups[event_key]
            matched_event_cache[cache_key] = source_events.loc[
                source_events["pair"].isin(coins)
            ].copy()
            balance_cache[cache_key] = declared_matching_balance(
                matched_event_cache[cache_key]
            )
        events = matched_event_cache[cache_key]
        valid_events = (
            pd.to_numeric(events[f"actual__{outcome}"], errors="coerce").notna()
            & pd.to_numeric(events[f"control__{outcome}"], errors="coerce").notna()
            & pd.to_numeric(events[f"delta__{outcome}"], errors="coerce").notna()
        )
        outcome_events = events.loc[valid_events]
        balance = balance_cache[cache_key]
        event_delta = pd.to_numeric(
            outcome_events[f"delta__{outcome}"],
            errors="coerce",
        )
        coin_delta = pd.to_numeric(group["delta_mean"], errors="coerce")
        event_count = int(group["independent_event_pairs"].sum())
        coverage = (
            len(coins) >= MIN_COHORT_PERIOD_COINS
            and event_count >= MIN_COHORT_PERIOD_EPISODES
        )
        usable_balance = (
            np.isfinite(balance["max_absolute_smd"])
            and np.isfinite(balance["median_absolute_smd"])
            and balance["max_absolute_smd"] <= MAX_STATE_SMD
            and balance["median_absolute_smd"] <= MAX_MEDIAN_STATE_SMD
        )
        rows.append(
            {
                "route_id": key[0],
                "density_family": key[1],
                "history_hours": int(key[2]),
                "actual_scope": key[3],
                "control": key[4],
                "period": key[5],
                "response_window": key[6],
                "outcome": outcome,
                "metric_family": outcome_metadata(outcome)[0],
                "horizon_hours": outcome_metadata(outcome)[1],
                "primary_outcome": outcome in PRIMARY_OUTCOMES,
                "eligible_coin_count": len(coins),
                "eligible_coins": ";".join(coins),
                "independent_event_pairs": event_count,
                "coverage_gate_passed": coverage,
                "state_balance_usable": usable_balance,
                "evidence_eligible": coverage and usable_balance,
                "equal_coin_delta_median": float(coin_delta.median()),
                "equal_coin_delta_q25": float(coin_delta.quantile(0.25)),
                "equal_coin_delta_q75": float(coin_delta.quantile(0.75)),
                "equal_coin_positive_fraction": float(coin_delta.gt(0.0).mean()),
                "pooled_event_delta_mean": float(event_delta.mean()),
                "pooled_event_delta_median": float(event_delta.median()),
                "pooled_event_positive_fraction": float(event_delta.gt(0.0).mean()),
                "max_absolute_state_smd": balance["max_absolute_smd"],
                "median_absolute_state_smd": balance["median_absolute_smd"],
                "state_features_scored": balance["features_scored"],
                "balance_quality": balance_quality(
                    balance["max_absolute_smd"],
                    balance["median_absolute_smd"],
                ),
                "direction_prediction": False,
                "profit_optimization": False,
            }
        )
    return DataFrame(rows)


def declared_matching_balance(frame: DataFrame) -> dict[str, Any]:
    declarations = frame["matching_state_features"].dropna().astype(str).unique()
    if len(declarations) != 1:
        raise ValueError(
            "A G2E comparison group must contain one declared matching-feature list; "
            f"found {sorted(declarations.tolist())}."
        )
    features = tuple(item for item in declarations[0].split(";") if item)
    if not features:
        raise ValueError("A G2E comparison group declared no matching features.")
    missing = sorted(
        column
        for feature in features
        for column in (f"actual_state__{feature}", f"control_state__{feature}")
        if column not in frame
    )
    if missing:
        raise ValueError(f"G2E comparison group lacks declared state columns: {missing}")
    return matched_pair_balance(frame, features)


def leave_one_coin_out_results(pair_period: DataFrame) -> DataFrame:
    if pair_period.empty:
        return DataFrame()
    eligible = pair_period.loc[pair_period["pair_period_coverage_eligible"]].copy()
    rows: list[dict[str, Any]] = []
    group_columns = [
        "route_id",
        "density_family",
        "history_hours",
        "actual_scope",
        "control",
        "period",
        "response_window",
        "outcome",
    ]
    for key, group in eligible.groupby(
        group_columns,
        observed=True,
        dropna=False,
        sort=False,
    ):
        ordered = group.sort_values("pair", kind="stable").reset_index(drop=True)
        available_pairs = ordered["pair"].astype(str).to_numpy()
        if len(available_pairs) < 2:
            continue
        delta_values = pd.to_numeric(
            ordered["delta_mean"], errors="coerce"
        ).to_numpy(dtype=float)
        event_counts = pd.to_numeric(
            ordered["independent_event_pairs"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        total_events = float(event_counts.sum())
        for omitted_index, omitted in enumerate(available_pairs):
            retained_delta = np.delete(delta_values, omitted_index)
            retained_delta = retained_delta[np.isfinite(retained_delta)]
            remaining_coins = len(available_pairs) - 1
            remaining_events = int(total_events - event_counts[omitted_index])
            rows.append(
                {
                    "route_id": key[0],
                    "density_family": key[1],
                    "history_hours": int(key[2]),
                    "actual_scope": key[3],
                    "control": key[4],
                    "period": key[5],
                    "response_window": key[6],
                    "outcome": key[7],
                    "omitted_pair": omitted,
                    "remaining_coin_count": remaining_coins,
                    "remaining_independent_event_pairs": remaining_events,
                    "coverage_gate_passed": remaining_coins
                    >= MIN_COHORT_PERIOD_COINS
                    and remaining_events >= MIN_COHORT_PERIOD_EPISODES,
                    "equal_coin_delta_median": (
                        float(np.median(retained_delta))
                        if len(retained_delta)
                        else np.nan
                    ),
                    "equal_coin_positive_fraction": (
                        float(np.mean(retained_delta > 0.0))
                        if len(retained_delta)
                        else np.nan
                    ),
                }
            )
    return DataFrame(rows)


def integrity_record(
    *,
    results: Sequence[dict[str, Any]],
    matches: DataFrame,
    independent: DataFrame,
    technical_smoke: bool,
    requested_pairs: Sequence[str],
) -> dict[str, Any]:
    failures = [row for row in results if row.get("status") == "failed"]
    future_violations = 0
    source_open_violations = 0
    for prefix in ("actual", "control"):
        available = pd.to_datetime(
            matches[f"{prefix}_source_available_at"],
            utc=True,
            errors="coerce",
        )
        source_open = pd.to_datetime(
            matches[f"{prefix}_source_open"],
            utc=True,
            errors="coerce",
        )
        event_time = pd.to_datetime(
            matches[f"{prefix}_event_time"],
            utc=True,
            errors="coerce",
        )
        future_violations += int((available.notna() & (available > event_time)).sum())
        source_open_violations += int(
            (source_open.notna() & (source_open >= event_time)).sum()
        )
    separation = pd.to_numeric(
        matches["event_separation_hours"], errors="coerce"
    )
    required_separation = matches["response_window"].map(
        SEPARATION_HOURS_BY_RESPONSE_WINDOW
    )
    separation_violations = int((separation < required_separation).sum())
    caliper = pd.to_numeric(
        matches["pre_distance_atr_abs_difference"], errors="coerce"
    )
    direction_values = set(matches["direction_prediction"].dropna().astype(bool))
    profit_values = set(matches["profit_optimization"].dropna().astype(bool))
    present_pairs = sorted(matches["pair"].astype(str).unique())
    present_controls = sorted(matches["control"].astype(str).unique())
    missing_controls = sorted(set(CONTROL_NAMES).difference(present_controls))
    duplicate_columns = [
        "route_id",
        "density_family",
        "history_hours",
        "actual_scope",
        "control",
        "pair",
        "period",
        "approach_state",
        "response_window",
        "actual_event_time",
        "control_event_time",
    ]
    duplicate_independent_rows = int(independent.duplicated(duplicate_columns).sum())
    pair_set_ok = set(present_pairs) == set(requested_pairs)
    passed = (
        not failures
        and not matches.empty
        and not independent.empty
        and future_violations == 0
        and source_open_violations == 0
        and separation_violations == 0
        and (caliper.dropna().le(0.1000001).all())
        and direction_values in (set(), {False})
        and profit_values in (set(), {False})
        and duplicate_independent_rows == 0
        and pair_set_ok
    )
    return {
        "created_at_utc": utc_now(),
        "passed": bool(passed),
        "technical_smoke_not_evidence": bool(technical_smoke),
        "pair_failures": failures,
        "requested_pairs": list(requested_pairs),
        "present_pairs": present_pairs,
        "pair_set_ok": pair_set_ok,
        "matched_comparisons_before_overlap_purge": len(matches),
        "independent_comparisons": len(independent),
        "future_source_availability_violations": future_violations,
        "source_open_not_before_event_violations": source_open_violations,
        "event_separation_violations": separation_violations,
        "pre_distance_atr_abs_difference_max": (
            float(caliper.max()) if caliper.notna().any() else np.nan
        ),
        "duplicate_independent_rows": duplicate_independent_rows,
        "families_present": sorted(matches["density_family"].astype(str).unique()),
        "histories_present": sorted(
            matches["history_hours"].astype(int).unique().tolist()
        ),
        "actual_scopes_present": sorted(matches["actual_scope"].astype(str).unique()),
        "controls_present": present_controls,
        "missing_applicable_or_covered_controls": missing_controls,
        "direction_prediction": False,
        "profit_optimization": False,
    }


if __name__ == "__main__":
    raise SystemExit(main())

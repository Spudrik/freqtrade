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
    LevelSpec,
    atomic_write_json,
    atomic_write_parquet,
    event_frame,
    future_path_matrices,
    load_manifest,
    matched_random_time_events,
    numeric_array,
    prepare_base_market_frame,
    sha256_file,
    utc_now,
    validate_worker_count,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_localization import (  # noqa: E501
    causal_local_state,
    nearest_state_pairs,
    pair_stem,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_anchored_vwap import (  # noqa: E501
    OUTCOMES_BY_RESPONSE_WINDOW,
    SEPARATION_HOURS_BY_RESPONSE_WINDOW,
    attach_state_and_absolute_outcomes,
    stable_json_sha256,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_density_reaction import (  # noqa: E501
    CORE_STATE_FEATURES,
    cohort_period_results,
    leave_one_coin_out_results,
    pair_period_results,
    purge_density_matches,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_density_reaction import (  # noqa: E501
    REPORT_ROOT as G2E_REPORT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_localization import (  # noqa: E501
    eligible_period_events,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_generic_levels import (  # noqa: E501
    ARTIFACT_ROOT as G3F_PREFLIGHT_ARTIFACT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_generic_levels import (  # noqa: E501
    GENERIC_DEPENDENCY_RELATIONSHIPS,
    GENERIC_RELATIONSHIPS,
    GenericLevel,
    explicit_generic_levels,
    formula_alias_group,
    validate_frozen_branch,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_generic_levels import (  # noqa: E501
    REPORT_ROOT as G3F_PREFLIGHT_REPORT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_lvn_attribution import (  # noqa: E501
    select_pairs,
)


FROZEN_BATCH = OUTPUT_ROOT / "generation2_review" / "g3_frozen_branch_batch.json"
REPORT_ROOT = OUTPUT_ROOT / "generation3_branches" / "g3f_generic_levels_reaction"
ARTIFACT_ROOT = LARGE_ARTIFACT_ROOT / "generation3_branches" / "g3f_generic_levels_reaction"
OUTPUT_SCHEMA_VERSION = 2
MAX_HORIZON_HOURS = 24
NO_LEVEL_CONTROL = "same_state_same_density_no_level"
COMPONENT_CONTROL = "component_ablation_isolated_anchor"
CONTROL_NAMES = (NO_LEVEL_CONTROL, COMPONENT_CONTROL)
IDENTITY_COLUMNS = (
    "level_family",
    "level_name",
    "source_timeframe",
    "generic_scope",
    "cluster_relationship",
    "cluster_dependency_relationship",
)
GENERIC_DENSITY_STATE_FEATURES = (
    "state_g3f_zone_half_width_atr",
    "state_g3f_active_level_count_near_price_2atr",
    "state_g3f_level_zone_coverage_4atr",
)
GENERIC_SOURCE_STATE_FEATURES = ("state_g3f_source_age_fraction",)
NO_LEVEL_STATE_FEATURES = (*CORE_STATE_FEATURES, *GENERIC_DENSITY_STATE_FEATURES)
COMPONENT_STATE_FEATURES = (
    *CORE_STATE_FEATURES,
    *GENERIC_DENSITY_STATE_FEATURES,
    *GENERIC_SOURCE_STATE_FEATURES,
)


@dataclass(frozen=True)
class PairTask:
    pair: str
    manifest_path: str
    context_path: str
    context_sha256: str
    geometry_path: str
    geometry_sha256: str
    supported_cells: tuple[tuple[str, str, str, str, str, str], ...]
    run_id: str
    request_sha256: str
    overwrite: bool


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 3F direction-neutral direct comparison of explicit named "
            "generic levels with matched no-level times and isolated-anchor ablations."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cohort", choices=("large", "meme"), required=True)
    parser.add_argument("--preflight-run-id", required=True)
    parser.add_argument("--g2e-run-id", required=True)
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

    preflight_dir = G3F_PREFLIGHT_REPORT_ROOT / args.preflight_run_id
    preflight_artifact_dir = G3F_PREFLIGHT_ARTIFACT_ROOT / args.preflight_run_id
    preflight_record_path = preflight_dir / "g3f_preflight_run_record.json"
    preflight_integrity_path = preflight_dir / "g3f_preflight_integrity.json"
    support_path = preflight_dir / "g3f_comparison_support.parquet"
    g2e_dir = G2E_REPORT_ROOT / args.g2e_run_id
    g2e_record_path = g2e_dir / "g2e_reaction_run_record.json"
    g2e_integrity_path = g2e_dir / "g2e_reaction_integrity.json"
    context_path = g2e_dir / "g2e_causal_market_context.parquet"
    supported_cells = validate_sources_and_supported_cells(
        cohort=args.cohort,
        preflight_record_path=preflight_record_path,
        preflight_integrity_path=preflight_integrity_path,
        support_path=support_path,
        g2e_record_path=g2e_record_path,
        g2e_integrity_path=g2e_integrity_path,
        context_path=context_path,
    )
    context_sha256 = sha256_file(context_path)
    pair_contracts = []
    for pair in pairs:
        geometry_path = preflight_artifact_dir / "pair_geometry" / f"{pair_stem(pair)}.parquet"
        pair_contracts.append(
            {
                "pair": pair,
                "geometry_path": str(geometry_path),
                "geometry_sha256": sha256_file(geometry_path),
            }
        )

    request = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "frozen_batch_sha256": sha256_file(FROZEN_BATCH),
        "cohort": args.cohort,
        "pairs": list(pairs),
        "preflight_run_id": args.preflight_run_id,
        "preflight_record_sha256": sha256_file(preflight_record_path),
        "preflight_integrity_sha256": sha256_file(preflight_integrity_path),
        "preflight_support_sha256": sha256_file(support_path),
        "g2e_context_run_id": args.g2e_run_id,
        "context_path": str(context_path),
        "context_sha256": context_sha256,
        "pair_geometry_contracts": pair_contracts,
        "supported_exact_cells_opened": [list(cell) for cell in supported_cells],
        "controls": {
            NO_LEVEL_CONTROL: (
                "Pseudo-zone contacts on the same coin and period, with comparable "
                "approach geometry, OHLCV/technical/broad-market state and generic-level "
                "density, but no real generic level overlapping the pseudo-zone."
            ),
            COMPONENT_CONTROL: (
                "The same exact named anchor level contacted while isolated, compared "
                "with its named same-timeframe or cross-timeframe relationship."
            ),
        },
        "matching_state_features": {
            NO_LEVEL_CONTROL: list(NO_LEVEL_STATE_FEATURES),
            COMPONENT_CONTROL: list(COMPONENT_STATE_FEATURES),
        },
        "outcomes": (
            "contact-candle range and volume plus absolute movement, range, volume, "
            "absolute pressure change, dwell and crossings over 1h, 4h and 24h"
        ),
        "reaction_outcomes_loaded": True,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    request_sha256 = stable_json_sha256(request)
    run_dir = REPORT_ROOT / args.run_id
    artifact_dir = ARTIFACT_ROOT / args.run_id
    pair_dir = artifact_dir / "pair_independent_matches"
    run_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    record_path = run_dir / "g3f_reaction_run_record.json"
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
            "The outcome-blind preflight found adequate contact coverage for exact named "
            "level/timeframe/relationship cells in both validation periods."
        ),
        "hypothesis": (
            "At least one exact named generic level marks repeated direction-neutral "
            "market behaviour beyond a matched pseudo-zone with no generic level, or a "
            "named convergence relationship adds behaviour beyond the same anchor alone."
        ),
        "pass_fail": (
            "A lead requires at least five coins and 50 independent pairs, usable "
            "pre-event state balance, the same effect sign in both validations, and "
            "leave-one-coin-out sign stability. No-level or component evidence alone is "
            "not full attribution until the remaining frozen controls are tested."
        ),
        "workers": args.workers,
        "reaction_outcomes_loaded": True,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(record, record_path)
    tasks = [
        PairTask(
            pair=contract["pair"],
            manifest_path=str(manifest_path),
            context_path=str(context_path),
            context_sha256=context_sha256,
            geometry_path=contract["geometry_path"],
            geometry_sha256=contract["geometry_sha256"],
            supported_cells=supported_cells,
            run_id=args.run_id,
            request_sha256=request_sha256,
            overwrite=args.overwrite,
        )
        for contract in pair_contracts
    ]
    try:
        results = run_tasks(tasks, workers=args.workers)
        inventory = DataFrame(results)
        atomic_write_parquet(inventory, run_dir / "g3f_reaction_pair_inventory.parquet")
        failures = inventory.loc[inventory["status"].eq("failed")]
        if not failures.empty:
            raise RuntimeError(f"{len(failures)} pair task(s) failed; inspect inventory.")
        independent = combine_pair_outputs(
            pair_dir=pair_dir,
            pairs=pairs,
            request_sha256=request_sha256,
        )
        pair_period = pair_period_results(independent)
        cohort_period = cohort_period_results(pair_period, independent)
        leave_one_out = leave_one_coin_out_results(pair_period)
        atomic_write_parquet(pair_period, run_dir / "g3f_pair_period_results.parquet")
        atomic_write_parquet(cohort_period, run_dir / "g3f_cohort_period_results.parquet")
        atomic_write_parquet(leave_one_out, run_dir / "g3f_leave_one_coin_out.parquet")
        integrity = integrity_record(
            results=results,
            independent=independent,
            requested_pairs=pairs,
        )
        atomic_write_json(integrity, run_dir / "g3f_reaction_integrity.json")
        if not integrity["passed"]:
            raise RuntimeError("G3F direct reaction integrity validation failed.")
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "pair_tasks": results,
                "independent_comparisons": len(independent),
                "pair_period_result_rows": len(pair_period),
                "cohort_period_result_rows": len(cohort_period),
                "leave_one_coin_out_rows": len(leave_one_out),
                "evidence_eligible_cohort_rows": int(
                    cohort_period["evidence_eligible"].fillna(False).sum()
                ),
                "integrity": str(run_dir / "g3f_reaction_integrity.json"),
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


def validate_sources_and_supported_cells(
    *,
    cohort: str,
    preflight_record_path: Path,
    preflight_integrity_path: Path,
    support_path: Path,
    g2e_record_path: Path,
    g2e_integrity_path: Path,
    context_path: Path,
) -> tuple[tuple[str, str, str, str, str, str], ...]:
    preflight_record = json.loads(preflight_record_path.read_text(encoding="utf-8"))
    preflight_integrity = json.loads(preflight_integrity_path.read_text(encoding="utf-8"))
    g2e_record = json.loads(g2e_record_path.read_text(encoding="utf-8"))
    g2e_integrity = json.loads(g2e_integrity_path.read_text(encoding="utf-8"))
    if preflight_record.get("status") != "completed":
        raise ValueError("G3F preflight is not completed.")
    if preflight_record.get("request_contract", {}).get("cohort") != cohort:
        raise ValueError("G3F preflight cohort does not match this reaction run.")
    if preflight_integrity.get("passed") is not True:
        raise ValueError("G3F preflight integrity did not pass.")
    if preflight_integrity.get("reaction_outcomes_loaded") is not False:
        raise ValueError("G3F preflight unexpectedly opened reaction outcomes.")
    if g2e_record.get("status") != "completed" or g2e_integrity.get("passed") is not True:
        raise ValueError("The causal market-context source is not a completed valid run.")
    if not context_path.is_file():
        raise ValueError(f"Missing causal market-context table: {context_path}")
    for source in (preflight_record, preflight_integrity, g2e_record, g2e_integrity):
        if source.get("direction_prediction") is not False:
            raise ValueError("A source violates direction_prediction=False.")
        if source.get("profit_optimization") is not False:
            raise ValueError("A source violates profit_optimization=False.")
    support = pd.read_parquet(support_path)
    missing = sorted(set(IDENTITY_COLUMNS).difference(support.columns))
    if missing:
        raise ValueError(f"G3F support table lacks explicit relationship columns: {missing}")
    opened = support.loc[support["both_validation_periods_supported"].fillna(False)]
    cells = tuple(
        sorted(
            {tuple(str(row[column]) for column in IDENTITY_COLUMNS) for _, row in opened.iterrows()}
        )
    )
    if not cells:
        raise ValueError("G3F preflight exposed no exact cell supported in both validations.")
    return cells


def filter_supported_geometry(
    geometry: DataFrame,
    supported_cells: Sequence[tuple[str, str, str, str, str, str]],
) -> DataFrame:
    allowed = set(supported_cells)
    identities = zip(
        *(geometry[column].astype(str).to_numpy() for column in IDENTITY_COLUMNS),
        strict=True,
    )
    mask = np.fromiter((identity in allowed for identity in identities), dtype=bool)
    return geometry.loc[mask].reset_index(drop=True)


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


def build_pair(task: PairTask) -> dict[str, Any]:  # noqa: C901 - explicit control cells
    started = time.perf_counter()
    output_path = (
        ARTIFACT_ROOT / task.run_id / "pair_independent_matches" / f"{pair_stem(task.pair)}.parquet"
    )
    if output_path.is_file() and not task.overwrite:
        existing = validate_pair_output(
            output_path,
            pair=task.pair,
            request_sha256=task.request_sha256,
        )
        return {
            "pair": task.pair,
            "status": "existing",
            "independent_comparisons": len(existing),
            "seconds": round(time.perf_counter() - started, 3),
        }
    geometry_path = Path(task.geometry_path)
    context_path = Path(task.context_path)
    if sha256_file(geometry_path) != task.geometry_sha256:
        raise ValueError(f"G3F preflight geometry changed: {geometry_path}")
    if sha256_file(context_path) != task.context_sha256:
        raise ValueError(f"G2E causal market context changed: {context_path}")
    geometry = filter_supported_geometry(
        pd.read_parquet(geometry_path),
        task.supported_cells,
    )
    if geometry.empty:
        raise ValueError(f"No supported G3F geometry for {task.pair}.")
    manifest = load_manifest(Path(task.manifest_path))
    base = prepare_base_market_frame(task.pair, manifest)
    context = pd.read_parquet(context_path)
    state = causal_local_state(base).merge(context, on="date", how="left", validate="one_to_one")
    paths = future_path_matrices(base, max_horizon=MAX_HORIZON_HOURS)
    levels = explicit_generic_levels(pair=task.pair, base=base)
    level_matrix, width_matrix, valid_matrix = generic_level_matrices(base, levels)
    density_valid_matrix = formula_unique_valid_matrix(levels, valid_matrix)
    density_count, density_coverage = generic_density_context(
        base=base,
        level_matrix=level_matrix,
        width_matrix=width_matrix,
        valid_matrix=density_valid_matrix,
    )
    events = attach_generic_event_outcomes(
        pair=task.pair,
        base=base,
        state=state,
        paths=paths,
        geometry=geometry,
        density_count=density_count,
        density_coverage=density_coverage,
    )
    rows: list[dict[str, Any]] = []
    no_level_cells = 0
    component_cells = 0
    route_columns = [
        "level_family",
        "level_name",
        "source_timeframe",
        "generic_scope",
        "cluster_relationship",
        "cluster_dependency_relationship",
    ]
    for _, actual in events.groupby(route_columns, observed=True, sort=False):
        actual = actual.reset_index(drop=True)
        control = no_level_control_pool(
            actual=actual,
            pair=task.pair,
            base=base,
            paths=paths,
            manifest=manifest,
            state=state,
            level_matrix=level_matrix,
            width_matrix=width_matrix,
            valid_matrix=valid_matrix,
            density_count=density_count,
            density_coverage=density_coverage,
        )
        if control.empty:
            continue
        matched = matched_outcome_rows(
            actual=actual,
            control=control,
            pair=task.pair,
            control_name=NO_LEVEL_CONTROL,
            state_features=NO_LEVEL_STATE_FEATURES,
        )
        if matched:
            no_level_cells += 1
            rows.extend(matched)

    supported = set(task.supported_cells)
    anchor_columns = ["level_family", "level_name", "source_timeframe"]
    for anchor, anchor_events in events.groupby(anchor_columns, observed=True, sort=False):
        family, name, timeframe = (str(value) for value in anchor)
        isolated_key = (
            family,
            name,
            timeframe,
            "isolated_generic_level",
            "isolated_generic_level",
            "isolated_generic_level",
        )
        if isolated_key not in supported:
            continue
        isolated = anchor_events.loc[
            anchor_events["cluster_relationship"].eq("isolated_generic_level")
            & anchor_events["cluster_dependency_relationship"].eq("isolated_generic_level")
        ].reset_index(drop=True)
        if isolated.empty:
            continue
        for key in sorted(supported):
            (
                cell_family,
                cell_name,
                cell_timeframe,
                scope,
                relationship,
                dependency_relationship,
            ) = key
            if (cell_family, cell_name, cell_timeframe) != (family, name, timeframe):
                continue
            if relationship == "isolated_generic_level":
                continue
            if scope != relationship_scope(relationship):
                continue
            clustered = anchor_events.loc[
                anchor_events["cluster_relationship"].eq(relationship)
                & anchor_events["cluster_dependency_relationship"].eq(dependency_relationship)
            ].reset_index(drop=True)
            matched = matched_outcome_rows(
                actual=clustered,
                control=isolated,
                pair=task.pair,
                control_name=COMPONENT_CONTROL,
                state_features=COMPONENT_STATE_FEATURES,
            )
            if matched:
                component_cells += 1
                rows.extend(matched)
    matches = DataFrame(rows)
    if matches.empty:
        raise ValueError(f"No matchable G3F direct comparisons for {task.pair}.")
    independent = purge_density_matches(matches)
    if independent.empty:
        raise ValueError(f"No independent G3F direct comparisons for {task.pair}.")
    independent["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    independent["run_request_sha256"] = task.request_sha256
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(independent, output_path)
    return {
        "pair": task.pair,
        "status": "completed",
        "supported_geometry_events": len(geometry),
        "outcome_event_rows": len(events),
        "no_level_cells_with_matches": no_level_cells,
        "component_cells_with_matches": component_cells,
        "matched_comparisons_before_overlap_purge": len(matches),
        "independent_comparisons": len(independent),
        "seconds": round(time.perf_counter() - started, 3),
    }


def relationship_scope(relationship: str) -> str:
    if relationship == "isolated_generic_level":
        return "isolated_generic_level"
    if relationship.startswith("same_timeframe_"):
        return "same_timeframe_generic_cluster"
    if relationship.startswith("cross_timeframe_"):
        return "cross_timeframe_generic_cluster"
    raise ValueError(f"Unknown G3F relationship: {relationship}")


def relationship_claim(relationship: str, dependency_relationship: str) -> str:
    return f"{relationship}__{dependency_relationship}"


def generic_level_matrices(
    base: DataFrame,
    levels: Sequence[GenericLevel],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_atr = numeric_array(base["base_atr"])
    level_matrix = np.column_stack([item.level for item in levels])
    valid_matrix = np.column_stack([item.valid for item in levels])
    width_matrix = np.maximum(
        0.10 * base_atr[:, None],
        np.abs(level_matrix) * 0.0005,
    )
    valid_matrix &= np.isfinite(level_matrix) & np.isfinite(width_matrix) & (width_matrix > 0.0)
    return level_matrix, width_matrix, valid_matrix


def formula_unique_valid_matrix(
    levels: Sequence[GenericLevel],
    valid_matrix: np.ndarray,
) -> np.ndarray:
    output = valid_matrix.copy()
    retained_alias_groups: set[str] = set()
    for index, item in enumerate(levels):
        alias = formula_alias_group(item)
        if alias is None:
            continue
        if alias in retained_alias_groups:
            output[:, index] = False
        else:
            retained_alias_groups.add(alias)
    return output


def generic_density_context(
    *,
    base: DataFrame,
    level_matrix: np.ndarray,
    width_matrix: np.ndarray,
    valid_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pre_close = numeric_array(base["pre_close"])
    base_atr = numeric_array(base["base_atr"])
    distance = np.abs(level_matrix - pre_close[:, None])
    near = valid_matrix & (distance <= 2.0 * base_atr[:, None])
    count = near.sum(axis=1).astype(np.int16)
    within_four = valid_matrix & (distance <= 4.0 * base_atr[:, None])
    numerator = np.where(within_four, 2.0 * width_matrix, 0.0).sum(axis=1)
    denominator = 8.0 * base_atr
    coverage = np.full(len(base), np.nan, dtype=float)
    usable = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0.0)
    coverage[usable] = numerator[usable] / denominator[usable]
    return count, coverage


def nominal_history_hours(frame: DataFrame) -> np.ndarray:
    timeframe = frame["source_timeframe"].astype(str).to_numpy()
    family = frame["level_family"].astype(str).to_numpy()
    lookback = pd.to_numeric(frame["source_lookback_bars"], errors="coerce").to_numpy(dtype=float)
    multiplier = np.select(
        [
            timeframe == "1h",
            timeframe == "4h",
            timeframe == "8h",
            timeframe == "1d",
        ],
        [1.0, 4.0, 8.0, 24.0],
        default=np.nan,
    )
    output = lookback * multiplier
    output = np.where(family == "previous_completed_week", 168.0, output)
    output = np.where(family == "previous_completed_month", 720.0, output)
    if not (np.isfinite(output) & (output > 0.0)).all():
        raise ValueError("G3F could not derive nominal history hours for every event.")
    return output.astype(np.int32)


def route_ids(frame: DataFrame) -> np.ndarray:
    return (
        "g3f|"
        + frame["source_timeframe"].astype(str)
        + "|"
        + frame["level_family"].astype(str)
        + "|"
        + frame["level_name"].astype(str)
    ).to_numpy(dtype=object)


def attach_generic_state(
    events: DataFrame,
    *,
    state: DataFrame,
    density_count: np.ndarray,
    density_coverage: np.ndarray,
) -> DataFrame:
    output = attach_state_and_absolute_outcomes(events, state)
    indexes = output["base_index"].to_numpy(dtype=np.int64)
    output["state_g3f_zone_half_width_atr"] = pd.to_numeric(
        output["zone_half_width_atr"], errors="coerce"
    )
    output["state_g3f_active_level_count_near_price_2atr"] = density_count[indexes]
    output["state_g3f_level_zone_coverage_4atr"] = density_coverage[indexes]
    if "source_age_hours" in output and "history_hours" in output:
        age = pd.to_numeric(output["source_age_hours"], errors="coerce")
        history = pd.to_numeric(output["history_hours"], errors="coerce")
        output["state_g3f_source_age_fraction"] = age / history
    else:
        output["state_g3f_source_age_fraction"] = np.nan
    return output


def attach_generic_event_outcomes(
    *,
    pair: str,
    base: DataFrame,
    state: DataFrame,
    paths: dict[str, np.ndarray],
    geometry: DataFrame,
    density_count: np.ndarray,
    density_coverage: np.ndarray,
) -> DataFrame:
    source = geometry.sort_values(
        [
            "base_index",
            "level_key",
            "cluster_relationship",
            "cluster_dependency_relationship",
        ],
        kind="stable",
    ).reset_index(drop=True)
    indexes = source["base_index"].to_numpy(dtype=np.int64)
    spec = LevelSpec(
        name="explicit_named_generic_level",
        family="explicit_generic_level",
        batch="g3f",
        column="g3f_explicit_named_generic_level",
    )
    events = event_frame(
        merged=base,
        paths=paths,
        indexes=indexes,
        levels=source["level_price"].to_numpy(dtype=float),
        widths=source["zone_half_width"].to_numpy(dtype=float),
        spec=spec,
        pair=pair,
        timeframe="1h_causal_explicit_generic_level",
        control="actual",
        zone_method="explicit_generic_level_zone",
        source_available=pd.to_datetime(source["source_available_at"], utc=True),
        source_open=pd.to_datetime(source["source_open"], utc=True),
        horizons=(1, 4, 24),
        match_tier=None,
    )
    if not events["base_index"].equals(source["base_index"]):
        raise ValueError("G3F outcome reconstruction changed event order.")
    expected_time = pd.to_datetime(source["event_time"], utc=True).reset_index(drop=True)
    if not pd.to_datetime(events["event_time"], utc=True).equals(expected_time):
        raise ValueError("G3F outcome reconstruction changed event timestamps.")
    if not np.array_equal(
        events["approach_state"].astype(str).to_numpy(),
        source["approach_state"].astype(str).to_numpy(),
    ):
        raise ValueError("G3F outcome reconstruction changed the causal approach side.")
    copy_columns = (
        "level_family",
        "level_name",
        "level_key",
        "source_timeframe",
        "generic_scope",
        "cluster_relationship",
        "cluster_dependency_relationship",
        "source_age_hours",
        "source_lookback_bars",
        "overlap_other_generic_level_count",
        "overlap_same_timeframe_level_count",
        "overlap_cross_timeframe_level_count",
        "overlap_same_family_level_count",
        "overlap_cross_family_level_count",
        "overlap_cross_dependency_group_level_count",
        "overlap_level_keys",
        "overlap_level_names",
        "overlap_level_families",
        "overlap_level_timeframes",
        "overlap_dependency_groups",
        "mathematical_alias_level_count",
        "mathematical_alias_level_keys",
    )
    for column in copy_columns:
        events[column] = source[column].to_numpy()
    events["route_id"] = route_ids(events)
    events["history_hours"] = nominal_history_hours(events)
    events["event_kind"] = "contact"
    return attach_generic_state(
        events,
        state=state,
        density_count=density_count,
        density_coverage=density_coverage,
    )


def real_generic_overlap_count(
    events: DataFrame,
    *,
    level_matrix: np.ndarray,
    width_matrix: np.ndarray,
    valid_matrix: np.ndarray,
) -> np.ndarray:
    if events.empty:
        return np.asarray([], dtype=np.int16)
    indexes = events["base_index"].to_numpy(dtype=np.int64)
    event_level = pd.to_numeric(events["level_price"], errors="coerce").to_numpy(dtype=float)
    event_width = pd.to_numeric(events["zone_half_width"], errors="coerce").to_numpy(dtype=float)
    overlap = valid_matrix[indexes] & (
        np.abs(level_matrix[indexes] - event_level[:, None])
        <= width_matrix[indexes] + event_width[:, None]
    )
    return overlap.sum(axis=1).astype(np.int16)


def no_level_control_pool(
    *,
    actual: DataFrame,
    pair: str,
    base: DataFrame,
    paths: dict[str, np.ndarray],
    manifest: dict[str, Any],
    state: DataFrame,
    level_matrix: np.ndarray,
    width_matrix: np.ndarray,
    valid_matrix: np.ndarray,
    density_count: np.ndarray,
    density_coverage: np.ndarray,
) -> DataFrame:
    first = actual.iloc[0]
    spec = LevelSpec(
        name=str(first["level_name"]),
        family=str(first["level_family"]),
        batch="g3f",
        column=f"g3f_random_{first['level_name']}",
    )
    control = matched_random_time_events(
        merged=base,
        paths=paths,
        actual=actual,
        spec=spec,
        pair=pair,
        timeframe=str(first["source_timeframe"]),
        zone_method="matched_pseudo_zone_without_generic_level",
        horizons=(1, 4, 24),
    )
    control = eligible_period_events(control, manifest, embargo_hours=MAX_HORIZON_HOURS)
    if control.empty:
        return control
    overlap_count = real_generic_overlap_count(
        control,
        level_matrix=level_matrix,
        width_matrix=width_matrix,
        valid_matrix=valid_matrix,
    )
    control["real_generic_overlap_count"] = overlap_count
    control = control.loc[control["real_generic_overlap_count"].eq(0)].reset_index(drop=True)
    if control.empty:
        return control
    control["level_family"] = str(first["level_family"])
    control["level_name"] = str(first["level_name"])
    control["level_key"] = str(first["level_key"])
    control["source_timeframe"] = str(first["source_timeframe"])
    control["generic_scope"] = "no_generic_level"
    control["cluster_relationship"] = "no_generic_level"
    control["cluster_dependency_relationship"] = "no_generic_level"
    control["route_id"] = str(first["route_id"])
    control["history_hours"] = int(first["history_hours"])
    control["event_kind"] = "pseudo_contact"
    control["source_age_hours"] = np.nan
    return attach_generic_state(
        control,
        state=state,
        density_count=density_count,
        density_coverage=density_coverage,
    )


def matched_outcome_rows(
    *,
    actual: DataFrame,
    control: DataFrame,
    pair: str,
    control_name: str,
    state_features: Sequence[str],
) -> list[dict[str, Any]]:
    if actual.empty or control.empty:
        return []
    if control_name not in CONTROL_NAMES:
        raise ValueError(f"Unknown G3F control: {control_name}")
    rows: list[dict[str, Any]] = []
    periods = sorted(set(actual["period"].astype(str)).intersection(control["period"].astype(str)))
    for period in periods:
        for approach in ("from_below", "from_above", "already_inside_or_unclear"):
            left = actual.loc[
                actual["period"].astype(str).eq(period) & actual["approach_state"].eq(approach)
            ].reset_index(drop=True)
            right = control.loc[
                control["period"].astype(str).eq(period) & control["approach_state"].eq(approach)
            ].reset_index(drop=True)
            if left.empty or right.empty:
                continue
            for response_window, outcomes in OUTCOMES_BY_RESPONSE_WINDOW.items():
                separation_hours = SEPARATION_HOURS_BY_RESPONSE_WINDOW[response_window]
                pairs, audit = nearest_state_pairs(
                    left,
                    right,
                    state_columns=state_features,
                    pre_distance_atr_caliper=0.10,
                    minimum_event_separation_hours=separation_hours,
                )
                for left_position, right_position, distance in pairs:
                    actual_row = left.iloc[left_position]
                    control_row = right.iloc[right_position]
                    event_separation = abs(
                        (
                            pd.Timestamp(actual_row["event_time"])
                            - pd.Timestamp(control_row["event_time"])
                        ).total_seconds()
                        / 3600.0
                    )
                    row: dict[str, Any] = {
                        "pair": pair,
                        "route_id": str(actual_row["route_id"]),
                        "density_family": str(actual_row["level_family"]),
                        "history_hours": int(actual_row["history_hours"]),
                        "level_name": str(actual_row["level_name"]),
                        "source_timeframe": str(actual_row["source_timeframe"]),
                        "actual_scope": relationship_claim(
                            str(actual_row["cluster_relationship"]),
                            str(actual_row["cluster_dependency_relationship"]),
                        ),
                        "actual_generic_scope": str(actual_row["generic_scope"]),
                        "actual_cluster_relationship": str(actual_row["cluster_relationship"]),
                        "actual_cluster_dependency_relationship": str(
                            actual_row["cluster_dependency_relationship"]
                        ),
                        "control": control_name,
                        "period": period,
                        "approach_state": approach,
                        "response_window": response_window,
                        "actual_event_kind": str(actual_row["event_kind"]),
                        "control_event_kind": str(control_row["event_kind"]),
                        "actual_event_time": actual_row["event_time"],
                        "control_event_time": control_row["event_time"],
                        "actual_base_index": int(actual_row["base_index"]),
                        "control_base_index": int(control_row["base_index"]),
                        "actual_source_available_at": actual_row["source_available_at"],
                        "control_source_available_at": control_row["source_available_at"],
                        "actual_source_open": actual_row["source_open"],
                        "control_source_open": control_row["source_open"],
                        "actual_zone_rank": -1,
                        "actual_zone_support_fraction": np.nan,
                        "match_distance": float(distance),
                        "pre_distance_atr_abs_difference": abs(
                            float(actual_row["pre_distance_atr"])
                            - float(control_row["pre_distance_atr"])
                        ),
                        "event_separation_hours": event_separation,
                        "actual_overlap_other_generic_level_count": int(
                            actual_row["overlap_other_generic_level_count"]
                        ),
                        "actual_overlap_level_keys": str(actual_row["overlap_level_keys"]),
                        "actual_overlap_level_names": str(actual_row["overlap_level_names"]),
                        "actual_overlap_level_families": str(actual_row["overlap_level_families"]),
                        "actual_overlap_level_timeframes": str(
                            actual_row["overlap_level_timeframes"]
                        ),
                        "actual_overlap_dependency_groups": str(
                            actual_row["overlap_dependency_groups"]
                        ),
                        "actual_mathematical_alias_level_count": int(
                            actual_row["mathematical_alias_level_count"]
                        ),
                        "actual_mathematical_alias_level_keys": str(
                            actual_row["mathematical_alias_level_keys"]
                        ),
                        "matching_state_features": ";".join(state_features),
                        "eligible_actual_events": int(audit["eligible_actual"]),
                        "eligible_control_events": int(audit["eligible_control"]),
                        "geometry_eligible_actual_events": int(audit["geometry_eligible_actual"]),
                        "state_matchable_actual_events": int(audit["state_matchable_actual"]),
                        "direction_prediction": False,
                        "profit_optimization": False,
                    }
                    for outcome in outcomes:
                        actual_value = float(actual_row[outcome])
                        control_value = float(control_row[outcome])
                        row[f"actual__{outcome}"] = actual_value
                        row[f"control__{outcome}"] = control_value
                        row[f"delta__{outcome}"] = actual_value - control_value
                    for feature in state_features:
                        row[f"actual_state__{feature}"] = float(actual_row[feature])
                        row[f"control_state__{feature}"] = float(control_row[feature])
                    rows.append(row)
    return rows


def integrity_record(
    *,
    results: Sequence[dict[str, Any]],
    independent: DataFrame,
    requested_pairs: Sequence[str],
) -> dict[str, Any]:
    failures = [row for row in results if row["status"] == "failed"]
    observed_controls = set(independent["control"].astype(str))
    invalid_controls = sorted(observed_controls.difference(CONTROL_NAMES))
    missing_pairs = sorted(set(requested_pairs).difference(independent["pair"].astype(str)))
    separation = pd.to_numeric(independent["event_separation_hours"], errors="coerce")
    required = independent["response_window"].map(SEPARATION_HOURS_BY_RESPONSE_WINDOW)
    separation_violations = int((separation <= required).sum())
    geometry_difference = pd.to_numeric(
        independent["pre_distance_atr_abs_difference"], errors="coerce"
    )
    actual_availability = pd.to_datetime(
        independent["actual_source_available_at"], utc=True, errors="coerce"
    )
    control_availability = pd.to_datetime(
        independent["control_source_available_at"], utc=True, errors="coerce"
    )
    actual_times = pd.to_datetime(independent["actual_event_time"], utc=True, errors="coerce")
    control_times = pd.to_datetime(independent["control_event_time"], utc=True, errors="coerce")
    future_source_violations = int(
        (actual_availability > actual_times).sum() + (control_availability > control_times).sum()
    )
    direction_violations = int(independent["direction_prediction"].ne(False).sum())
    profit_violations = int(independent["profit_optimization"].ne(False).sum())
    invalid_relationships = sorted(
        set(independent["actual_cluster_relationship"].astype(str)).difference(
            GENERIC_RELATIONSHIPS
        )
    )
    invalid_dependency_relationships = sorted(
        set(independent["actual_cluster_dependency_relationship"].astype(str)).difference(
            GENERIC_DEPENDENCY_RELATIONSHIPS
        )
    )
    return {
        "created_at_utc": utc_now(),
        "passed": bool(
            not failures
            and not independent.empty
            and not invalid_controls
            and not missing_pairs
            and separation_violations == 0
            and geometry_difference.le(0.1000001).all()
            and future_source_violations == 0
            and direction_violations == 0
            and profit_violations == 0
            and not invalid_relationships
            and not invalid_dependency_relationships
        ),
        "failures": failures,
        "independent_comparisons": len(independent),
        "requested_pairs": list(requested_pairs),
        "observed_pairs": sorted(independent["pair"].astype(str).unique()),
        "missing_pairs": missing_pairs,
        "controls_present": sorted(observed_controls),
        "invalid_controls": invalid_controls,
        "relationships_present": sorted(
            independent["actual_cluster_relationship"].astype(str).unique()
        ),
        "invalid_relationships": invalid_relationships,
        "dependency_relationships_present": sorted(
            independent["actual_cluster_dependency_relationship"].astype(str).unique()
        ),
        "invalid_dependency_relationships": invalid_dependency_relationships,
        "event_separation_violations": separation_violations,
        "pre_distance_atr_abs_difference_max": float(geometry_difference.max()),
        "future_source_violations": future_source_violations,
        "direction_prediction_violations": direction_violations,
        "profit_optimization_violations": profit_violations,
        "reaction_outcomes_loaded": True,
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
    *,
    pair_dir: Path,
    pairs: Sequence[str],
    request_sha256: str,
) -> DataFrame:
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

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
    future_path_matrices,
    load_manifest,
    normalize_dates,
    numeric_array,
    prepare_base_market_frame,
    sha256_file,
    shift_array,
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
    stable_json_sha256,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_density_reaction import (  # noqa: E501
    REPORT_ROOT as G2E_REPORT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_density_reaction import (  # noqa: E501
    cohort_period_results,
    leave_one_coin_out_results,
    pair_period_results,
    purge_density_matches,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_generic_levels import (  # noqa: E501
    ARTIFACT_ROOT as G3F_PREFLIGHT_ARTIFACT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_generic_levels import (  # noqa: E501
    EPISODE_COOLDOWN_HOURS,
    GENERIC_DEPENDENCY_RELATIONSHIPS,
    GENERIC_RELATIONSHIPS,
    OUTCOME_EMBARGO_HOURS,
    ZONE_HALF_WIDTH_ATR,
    ZONE_MINIMUM_PRICE_FRACTION,
    GenericLevel,
    classify_dependency_relationship,
    classify_relationship,
    classify_scope,
    explicit_generic_levels,
    formula_alias_group,
    generic_dependency_group,
    validate_frozen_branch,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_generic_levels import (  # noqa: E501
    REPORT_ROOT as G3F_PREFLIGHT_REPORT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_generic_levels_reaction import (  # noqa: E501
    COMPONENT_STATE_FEATURES,
    attach_generic_event_outcomes,
    formula_unique_valid_matrix,
    generic_density_context,
    generic_level_matrices,
    relationship_claim,
    relationship_scope,
    validate_sources_and_supported_cells,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_generic_levels_review import (  # noqa: E501
    PLATFORM_PAIRS,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_generic_levels_review import (  # noqa: E501
    REPORT_ROOT as G3F_REVIEW_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_lvn_attribution import (  # noqa: E501
    select_pairs,
)


FROZEN_BATCH = OUTPUT_ROOT / "generation2_review" / "g3_frozen_branch_batch.json"
REPORT_ROOT = OUTPUT_ROOT / "generation3_branches" / "g3f_generic_levels_artificial"
ARTIFACT_ROOT = LARGE_ARTIFACT_ROOT / "generation3_branches" / "g3f_generic_levels_artificial"
OUTPUT_SCHEMA_VERSION = 1
MAX_HORIZON_HOURS = 24
SHIFT_CONTROLS = (-1.0, 1.0)
STALE_CONTROL = "stale_same_nominal_history"
SHUFFLED_CONTROL = "shuffled_indicator_identity"
PRIOR_HIGH_CONTROL = "simple_prior_high_same_history"
PRIOR_LOW_CONTROL = "simple_prior_low_same_history"
CONTROL_NAMES = (
    "shift_-1atr",
    "shift_+1atr",
    STALE_CONTROL,
    SHUFFLED_CONTROL,
    PRIOR_HIGH_CONTROL,
    PRIOR_LOW_CONTROL,
)
SOURCE_AGE_FEATURE = "state_g3f_source_age_fraction"
CONTROL_STATE_FEATURES = {
    "shift_-1atr": COMPONENT_STATE_FEATURES,
    "shift_+1atr": COMPONENT_STATE_FEATURES,
    SHUFFLED_CONTROL: COMPONENT_STATE_FEATURES,
    STALE_CONTROL: tuple(
        feature for feature in COMPONENT_STATE_FEATURES if feature != SOURCE_AGE_FEATURE
    ),
    PRIOR_HIGH_CONTROL: tuple(
        feature for feature in COMPONENT_STATE_FEATURES if feature != SOURCE_AGE_FEATURE
    ),
    PRIOR_LOW_CONTROL: tuple(
        feature for feature in COMPONENT_STATE_FEATURES if feature != SOURCE_AGE_FEATURE
    ),
}
IDENTITY_ROTATIONS = {
    "simple_moving_average": ("sma_20", "sma_50", "sma_200"),
    "exponential_moving_average": ("ema_12", "ema_26", "ema_50"),
    "bollinger_band": (
        "bollinger_20_lower",
        "bollinger_20_middle",
        "bollinger_20_upper",
    ),
    "previous_completed_week": (
        "previous_completed_week_low",
        "previous_completed_week_midpoint",
        "previous_completed_week_high",
    ),
    "previous_completed_month": (
        "previous_completed_month_low",
        "previous_completed_month_midpoint",
        "previous_completed_month_high",
    ),
}


@dataclass(frozen=True)
class CandidateCell:
    family: str
    name: str
    source_timeframe: str
    relationship: str
    dependency_relationship: str
    history_hours: int

    @property
    def route_id(self) -> str:
        return f"g3f|{self.source_timeframe}|{self.family}|{self.name}"

    @property
    def actual_scope(self) -> str:
        return relationship_claim(self.relationship, self.dependency_relationship)

    @property
    def target_key(self) -> str:
        return f"{self.source_timeframe}|{self.family}|{self.name}"


@dataclass(frozen=True)
class ArtificialLevel:
    control: str
    item: GenericLevel
    excluded_real_keys: tuple[str, ...]
    donor_key: str | None
    construction: str


@dataclass(frozen=True)
class PairTask:
    pair: str
    manifest_path: str
    context_path: str
    context_sha256: str
    geometry_path: str
    geometry_sha256: str
    candidate_cells: tuple[CandidateCell, ...]
    run_id: str
    request_sha256: str
    overwrite: bool


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 3F direction-neutral attribution of frozen named generic-level "
            "leads against shifted, stale, identity-shuffled and simple-prior controls."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cohort", choices=("large", "meme"), required=True)
    parser.add_argument("--preflight-run-id", required=True)
    parser.add_argument("--g2e-run-id", required=True)
    parser.add_argument("--review-run-id", required=True)
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
    review_dir = G3F_REVIEW_ROOT / args.review_run_id
    review_record_path = review_dir / "g3f_review_run_record.json"
    review_integrity_path = review_dir / "g3f_review_integrity.json"
    candidate_path = review_dir / "g3f_candidate_cells.parquet"
    candidate_cells = validated_candidate_cells(
        cohort=args.cohort,
        review_record_path=review_record_path,
        review_integrity_path=review_integrity_path,
        candidate_path=candidate_path,
        supported_cells=supported_cells,
    )
    context_sha256 = sha256_file(context_path)
    pair_contracts: list[dict[str, Any]] = []
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
        "review_run_id": args.review_run_id,
        "review_record_sha256": sha256_file(review_record_path),
        "review_integrity_sha256": sha256_file(review_integrity_path),
        "candidate_cells_sha256": sha256_file(candidate_path),
        "candidate_structural_cells": [cell.__dict__ for cell in candidate_cells],
        "pair_geometry_contracts": pair_contracts,
        "controls": {
            "shift_-1atr": "Move the named level down by exactly one causal ATR.",
            "shift_+1atr": "Move the named level up by exactly one causal ATR.",
            STALE_CONTROL: (
                "Use the same named level value from one complete nominal indicator "
                "history earlier."
            ),
            SHUFFLED_CONTROL: (
                "Use a fixed different member of the same family and source timeframe "
                "while retaining the tested level identity."
            ),
            PRIOR_HIGH_CONTROL: (
                "Use the highest completed one-hour high over the same nominal history."
            ),
            PRIOR_LOW_CONTROL: (
                "Use the lowest completed one-hour low over the same nominal history."
            ),
        },
        "identity_rotations": IDENTITY_ROTATIONS,
        "target_contamination_rule": (
            "Discard an artificial contact if its zone overlaps the current real target "
            "or a mathematical alias of that target. A shuffled donor and its aliases "
            "are excluded from cluster membership because they supply the placebo price."
        ),
        "relationship_rule": (
            "The artificial contact must reproduce the frozen same/cross-timeframe, "
            "same/cross-family and same/cross-dependency relationship of the real cell."
        ),
        "matching_state_features": {
            name: list(features) for name, features in CONTROL_STATE_FEATURES.items()
        },
        "source_age_rule": (
            "Source age is matched for shifted and identity-shuffled controls. It is "
            "deliberately omitted for stale and continuously refreshed prior-extreme "
            "controls because age/freshness is part of those placebo constructions."
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
    record_path = run_dir / "g3f_artificial_run_record.json"
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
            "The frozen direct review retained 50 distinct structural cells across the "
            "normal, meme and predeclared platform scopes, before artificial controls."
        ),
        "hypothesis": (
            "At least one frozen named level or exact convergence cell repeats the same "
            "direction-neutral response beyond plausible fake locations."
        ),
        "pass_fail": (
            "A final lead must retain coverage, state balance, two-period sign agreement, "
            "the declared cohort-member agreement and leave-one-coin-out stability against "
            "every applicable direct and artificial control. Missing fair control coverage "
            "is insufficient evidence, not a pass."
        ),
        "workers": args.workers,
        "reaction_outcomes_loaded": True,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(record, record_path)
    tasks = [
        PairTask(
            pair=row["pair"],
            manifest_path=str(manifest_path),
            context_path=str(context_path),
            context_sha256=context_sha256,
            geometry_path=row["geometry_path"],
            geometry_sha256=row["geometry_sha256"],
            candidate_cells=candidate_cells,
            run_id=args.run_id,
            request_sha256=request_sha256,
            overwrite=args.overwrite,
        )
        for row in pair_contracts
    ]
    try:
        results = run_tasks(tasks, workers=args.workers)
        inventory = DataFrame(results)
        atomic_write_parquet(inventory, run_dir / "g3f_artificial_pair_inventory.parquet")
        failures = inventory.loc[inventory["status"].eq("failed")]
        if not failures.empty:
            raise RuntimeError(f"{len(failures)} pair task(s) failed; inspect inventory.")
        independent = combine_pair_outputs(
            pair_dir=pair_dir,
            pairs=[row["pair"] for row in results if row["status"] in {"completed", "existing"}],
            request_sha256=request_sha256,
        )
        pair_period = pair_period_results(independent)
        cohort_period = cohort_period_results(pair_period, independent)
        leave_one_out = leave_one_coin_out_results(pair_period)
        atomic_write_parquet(pair_period, run_dir / "g3f_pair_period_results.parquet")
        atomic_write_parquet(cohort_period, run_dir / "g3f_cohort_period_results.parquet")
        atomic_write_parquet(leave_one_out, run_dir / "g3f_leave_one_coin_out.parquet")
        platform_counts: dict[str, int] = {}
        if args.cohort == "large":
            platform_independent = independent.loc[
                independent["pair"].astype(str).isin(PLATFORM_PAIRS)
            ].reset_index(drop=True)
            platform_pair = pair_period_results(platform_independent)
            platform_cohort = cohort_period_results(platform_pair, platform_independent)
            platform_loo = leave_one_coin_out_results(platform_pair)
            atomic_write_parquet(platform_pair, run_dir / "g3f_platform_pair_period.parquet")
            atomic_write_parquet(platform_cohort, run_dir / "g3f_platform_cohort_period.parquet")
            atomic_write_parquet(platform_loo, run_dir / "g3f_platform_leave_one_out.parquet")
            platform_counts = {
                "platform_pair_period_rows": len(platform_pair),
                "platform_cohort_period_rows": len(platform_cohort),
                "platform_leave_one_out_rows": len(platform_loo),
            }
        integrity = integrity_record(
            results=results,
            independent=independent,
            requested_pairs=pairs,
        )
        atomic_write_json(integrity, run_dir / "g3f_artificial_integrity.json")
        if not integrity["passed"]:
            raise RuntimeError("G3F artificial-control integrity validation failed.")
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "pair_tasks": results,
                "independent_comparisons": len(independent),
                "pair_period_rows": len(pair_period),
                "cohort_period_rows": len(cohort_period),
                "leave_one_coin_out_rows": len(leave_one_out),
                "evidence_eligible_rows": int(
                    cohort_period.get("evidence_eligible", Series(dtype=bool)).fillna(False).sum()
                ),
                **platform_counts,
                "integrity": str(run_dir / "g3f_artificial_integrity.json"),
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


def validated_candidate_cells(
    *,
    cohort: str,
    review_record_path: Path,
    review_integrity_path: Path,
    candidate_path: Path,
    supported_cells: Sequence[tuple[str, str, str, str, str, str]],
) -> tuple[CandidateCell, ...]:
    record = json.loads(review_record_path.read_text(encoding="utf-8"))
    integrity = json.loads(review_integrity_path.read_text(encoding="utf-8"))
    if record.get("status") != "completed" or integrity.get("passed") is not True:
        raise ValueError("The G3F direct review is not a completed valid source.")
    for source in (record, integrity):
        if source.get("direction_prediction") is not False:
            raise ValueError("The G3F review violates direction_prediction=False.")
        if source.get("profit_optimization") is not False:
            raise ValueError("The G3F review violates profit_optimization=False.")
    candidates = pd.read_parquet(candidate_path)
    wanted_scopes = {"meme_top10"} if cohort == "meme" else {"normal_top10", "platform6"}
    candidates = candidates.loc[candidates["scope"].astype(str).isin(wanted_scopes)].copy()
    if candidates.empty:
        raise ValueError(f"The G3F review has no frozen candidates for cohort {cohort}.")
    cells: set[CandidateCell] = set()
    supported = set(supported_cells)
    for _, row in candidates.iterrows():
        route = str(row["route_id"]).split("|")
        if len(route) != 4 or route[0] != "g3f":
            raise ValueError(f"Malformed G3F route ID: {row['route_id']}")
        timeframe, family, name = route[1:]
        relationship, dependency = str(row["actual_scope"]).split("__", maxsplit=1)
        generic_scope = relationship_scope(relationship)
        source_key = (family, name, timeframe, generic_scope, relationship, dependency)
        if source_key not in supported:
            raise ValueError(f"Frozen candidate is absent from preflight support: {source_key}")
        cells.add(
            CandidateCell(
                family=family,
                name=name,
                source_timeframe=timeframe,
                relationship=relationship,
                dependency_relationship=dependency,
                history_hours=int(row["history_hours"]),
            )
        )
    return tuple(
        sorted(
            cells,
            key=lambda cell: (
                cell.source_timeframe,
                cell.family,
                cell.name,
                cell.relationship,
                cell.dependency_relationship,
            ),
        )
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


def build_pair(task: PairTask) -> dict[str, Any]:  # noqa: C901 - explicit control ladder
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
    geometry = filter_candidate_geometry(pd.read_parquet(geometry_path), task.candidate_cells)
    if geometry.empty:
        return {
            "pair": task.pair,
            "status": "no_matches",
            "reason": "no frozen candidate contact geometry for this pair",
            "seconds": round(time.perf_counter() - started, 3),
        }
    manifest = load_manifest(Path(task.manifest_path))
    base = prepare_base_market_frame(task.pair, manifest)
    context = pd.read_parquet(context_path)
    state = causal_local_state(base).merge(context, on="date", how="left", validate="one_to_one")
    paths = future_path_matrices(base, max_horizon=MAX_HORIZON_HOURS)
    levels = explicit_generic_levels(pair=task.pair, base=base)
    level_by_key = {item.key: item for item in levels}
    level_matrix, width_matrix, valid_matrix = generic_level_matrices(base, levels)
    density_valid = formula_unique_valid_matrix(levels, valid_matrix)
    density_count, density_coverage = generic_density_context(
        base=base,
        level_matrix=level_matrix,
        width_matrix=width_matrix,
        valid_matrix=density_valid,
    )
    actual = attach_generic_event_outcomes(
        pair=task.pair,
        base=base,
        state=state,
        paths=paths,
        geometry=geometry,
        density_count=density_count,
        density_coverage=density_coverage,
    )
    actual["event_kind"] = "real_generic_contact"
    rows: list[dict[str, Any]] = []
    raw_control_starts = 0
    target_contamination_drops = 0
    relationship_kept_events = 0
    control_event_counts: dict[str, int] = {name: 0 for name in CONTROL_NAMES}
    cell_match_counts: dict[str, int] = {}
    by_target: dict[str, list[CandidateCell]] = {}
    for cell in task.candidate_cells:
        by_target.setdefault(cell.target_key, []).append(cell)
    for target_key, cells in sorted(by_target.items()):
        target = level_by_key.get(target_key)
        if target is None:
            raise ValueError(f"Cannot resolve frozen G3F target {target_key} for {task.pair}.")
        definitions = artificial_level_definitions(base=base, target=target, levels=levels)
        for definition in definitions:
            control_geometry, audit = artificial_contact_geometry(
                pair=task.pair,
                base=base,
                levels=levels,
                artificial=definition,
            )
            raw_control_starts += audit["raw_episode_starts"]
            target_contamination_drops += audit["target_contamination_drops"]
            if control_geometry.empty:
                continue
            control_events = attach_generic_event_outcomes(
                pair=task.pair,
                base=base,
                state=state,
                paths=paths,
                geometry=control_geometry,
                density_count=density_count,
                density_coverage=density_coverage,
            )
            control_events["event_kind"] = f"artificial_{definition.control}_contact"
            control_events["artificial_control"] = definition.control
            control_events["artificial_donor_key"] = definition.donor_key
            control_events["artificial_construction"] = definition.construction
            control_events["artificial_excluded_real_keys"] = ";".join(
                definition.excluded_real_keys
            )
            control_event_counts[definition.control] += len(control_events)
            for cell in cells:
                left = actual.loc[
                    actual["route_id"].eq(cell.route_id)
                    & actual["cluster_relationship"].eq(cell.relationship)
                    & actual["cluster_dependency_relationship"].eq(
                        cell.dependency_relationship
                    )
                ].reset_index(drop=True)
                right = control_events.loc[
                    control_events["cluster_relationship"].eq(cell.relationship)
                    & control_events["cluster_dependency_relationship"].eq(
                        cell.dependency_relationship
                    )
                ].reset_index(drop=True)
                relationship_kept_events += len(right)
                matched = artificial_match_rows(
                    actual=left,
                    control=right,
                    pair=task.pair,
                    cell=cell,
                    control_name=definition.control,
                    state_features=CONTROL_STATE_FEATURES[definition.control],
                )
                if matched:
                    key = f"{cell.route_id}|{cell.actual_scope}|{definition.control}"
                    cell_match_counts[key] = cell_match_counts.get(key, 0) + len(matched)
                    rows.extend(matched)
    matches = DataFrame(rows)
    if matches.empty:
        return {
            "pair": task.pair,
            "status": "no_matches",
            "reason": "candidate cells had no state-matchable artificial controls",
            "actual_contact_events": len(actual),
            "raw_artificial_episode_starts": raw_control_starts,
            "target_contamination_drops": target_contamination_drops,
            "seconds": round(time.perf_counter() - started, 3),
        }
    independent = purge_density_matches(matches)
    if independent.empty:
        return {
            "pair": task.pair,
            "status": "no_matches",
            "reason": "all artificial comparisons overlapped after independence purge",
            "seconds": round(time.perf_counter() - started, 3),
        }
    independent["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    independent["run_request_sha256"] = task.request_sha256
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(independent, output_path)
    return {
        "pair": task.pair,
        "status": "completed",
        "candidate_cells": len(task.candidate_cells),
        "actual_contact_events": len(actual),
        "raw_artificial_episode_starts": raw_control_starts,
        "target_contamination_drops": target_contamination_drops,
        "relationship_kept_control_events": relationship_kept_events,
        "control_event_counts": control_event_counts,
        "cells_with_matches": len(cell_match_counts),
        "matched_comparisons_before_overlap_purge": len(matches),
        "independent_comparisons": len(independent),
        "seconds": round(time.perf_counter() - started, 3),
    }


def filter_candidate_geometry(
    geometry: DataFrame,
    cells: Sequence[CandidateCell],
) -> DataFrame:
    allowed = {
        (
            cell.family,
            cell.name,
            cell.source_timeframe,
            cell.relationship,
            cell.dependency_relationship,
        )
        for cell in cells
    }
    identities = zip(
        geometry["level_family"].astype(str),
        geometry["level_name"].astype(str),
        geometry["source_timeframe"].astype(str),
        geometry["cluster_relationship"].astype(str),
        geometry["cluster_dependency_relationship"].astype(str),
        strict=True,
    )
    mask = np.fromiter((identity in allowed for identity in identities), dtype=bool)
    return geometry.loc[mask].reset_index(drop=True)


def shuffled_donor_name(family: str, name: str) -> str:
    rotation = IDENTITY_ROTATIONS.get(family)
    if rotation is None or name not in rotation:
        raise ValueError(f"No fixed shuffled-identity rotation for {family}|{name}.")
    return rotation[(rotation.index(name) + 1) % len(rotation)]


def equivalent_real_keys(item: GenericLevel, levels: Sequence[GenericLevel]) -> tuple[str, ...]:
    alias = formula_alias_group(item)
    keys = {
        candidate.key
        for candidate in levels
        if candidate.key == item.key
        or (alias is not None and formula_alias_group(candidate) == alias)
    }
    return tuple(sorted(keys))


def nominal_history_hours(item: GenericLevel) -> int:
    if item.family == "previous_completed_week":
        return 168
    if item.family == "previous_completed_month":
        return 720
    multiplier = {"1h": 1, "4h": 4, "8h": 8, "1d": 24}.get(item.source_timeframe)
    if multiplier is None:
        raise ValueError(f"Unsupported generic source timeframe: {item.source_timeframe}")
    lookbacks = np.unique(item.lookback_bars[item.lookback_bars > 0])
    if len(lookbacks) != 1:
        raise ValueError(f"Expected one lookback for generic level {item.key}; got {lookbacks}.")
    return int(lookbacks[0]) * multiplier


def artificial_level_definitions(
    *,
    base: DataFrame,
    target: GenericLevel,
    levels: Sequence[GenericLevel],
) -> tuple[ArtificialLevel, ...]:
    atr = numeric_array(base["base_atr"])
    dates = normalize_dates(base["date"])
    history = nominal_history_hours(target)
    target_exclusions = equivalent_real_keys(target, levels)
    output: list[ArtificialLevel] = []
    for shift in SHIFT_CONTROLS:
        values = target.level + shift * atr
        valid = target.valid & np.isfinite(values) & (values > 0.0)
        control = f"shift_{shift:+g}atr"
        output.append(
            ArtificialLevel(
                control=control,
                item=GenericLevel(
                    family=target.family,
                    name=target.name,
                    source_timeframe=target.source_timeframe,
                    level=values,
                    valid=valid,
                    source_available=target.source_available.copy(),
                    source_open=target.source_open.copy(),
                    source_age_hours=target.source_age_hours.copy(),
                    lookback_bars=target.lookback_bars.copy(),
                ),
                excluded_real_keys=target_exclusions,
                donor_key=None,
                construction=f"target level shifted by {shift:+g} causal ATR",
            )
        )
    stale_values = shift_array(target.level, history)
    stale_valid = shift_array(target.valid.astype(float), history) == 1.0
    stale_available = target.source_available.shift(history)
    stale_open = target.source_open.shift(history)
    stale_age = (
        dates - normalize_dates(stale_available)
    ).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    stale_valid &= np.isfinite(stale_values) & (stale_values > 0.0) & stale_available.notna()
    output.append(
        ArtificialLevel(
            control=STALE_CONTROL,
            item=GenericLevel(
                family=target.family,
                name=target.name,
                source_timeframe=target.source_timeframe,
                level=stale_values,
                valid=stale_valid,
                source_available=normalize_dates(stale_available),
                source_open=normalize_dates(stale_open),
                source_age_hours=stale_age,
                lookback_bars=target.lookback_bars.copy(),
            ),
            excluded_real_keys=target_exclusions,
            donor_key=None,
            construction=f"target value delayed by its {history}-hour nominal history",
        )
    )
    donor_name = shuffled_donor_name(target.family, target.name)
    donors = [
        item
        for item in levels
        if item.family == target.family
        and item.name == donor_name
        and item.source_timeframe == target.source_timeframe
    ]
    if len(donors) != 1:
        raise ValueError(f"Expected one shuffled donor for {target.key}; found {len(donors)}.")
    donor = donors[0]
    output.append(
        ArtificialLevel(
            control=SHUFFLED_CONTROL,
            item=GenericLevel(
                family=target.family,
                name=target.name,
                source_timeframe=target.source_timeframe,
                level=donor.level.copy(),
                valid=donor.valid.copy(),
                source_available=donor.source_available.copy(),
                source_open=donor.source_open.copy(),
                source_age_hours=donor.source_age_hours.copy(),
                lookback_bars=target.lookback_bars.copy(),
            ),
            excluded_real_keys=tuple(
                sorted(set(target_exclusions).union(equivalent_real_keys(donor, levels)))
            ),
            donor_key=donor.key,
            construction=f"{target.key} identity assigned to fixed donor {donor.key}",
        )
    )
    high = pd.to_numeric(base["high"], errors="coerce")
    low = pd.to_numeric(base["low"], errors="coerce")
    prior_sources = (
        (PRIOR_HIGH_CONTROL, high.shift(1).rolling(history, min_periods=history).max()),
        (PRIOR_LOW_CONTROL, low.shift(1).rolling(history, min_periods=history).min()),
    )
    for control, source in prior_sources:
        values = source.to_numpy(dtype=float)
        valid = np.isfinite(values) & (values > 0.0)
        output.append(
            ArtificialLevel(
                control=control,
                item=GenericLevel(
                    family=target.family,
                    name=target.name,
                    source_timeframe=target.source_timeframe,
                    level=values,
                    valid=valid,
                    source_available=dates.copy(),
                    source_open=dates - pd.Timedelta(hours=history),
                    source_age_hours=np.zeros(len(base), dtype=float),
                    lookback_bars=target.lookback_bars.copy(),
                ),
                excluded_real_keys=target_exclusions,
                donor_key=None,
                construction=(
                    f"completed one-hour {'high' if control == PRIOR_HIGH_CONTROL else 'low'} "
                    f"over the same {history}-hour nominal history"
                ),
            )
        )
    return tuple(output)


def artificial_contact_geometry(
    *,
    pair: str,
    base: DataFrame,
    levels: Sequence[GenericLevel],
    artificial: ArtificialLevel,
) -> tuple[DataFrame, dict[str, int]]:
    item = artificial.item
    base_atr = numeric_array(base["base_atr"])
    high = numeric_array(base["high"])
    low = numeric_array(base["low"])
    pre_close = numeric_array(base["pre_close"])
    dates = normalize_dates(base["date"])
    real_level, real_width, real_valid = generic_level_matrices(base, levels)
    width = np.maximum(
        ZONE_HALF_WIDTH_ATR * base_atr,
        np.abs(item.level) * ZONE_MINIMUM_PRICE_FRACTION,
    )
    valid = item.valid & np.isfinite(width) & (width > 0.0)
    contact = valid & (high >= item.level - width) & (low <= item.level + width)
    starts = episode_start_mask(contact, item.level, width, cooldown=EPISODE_COOLDOWN_HOURS)
    starts[max(len(starts) - OUTCOME_EMBARGO_HOURS, 0) :] = False
    indexes = np.flatnonzero(starts)
    keys = np.asarray([level.key for level in levels], dtype=object)
    timeframes = np.asarray([level.source_timeframe for level in levels], dtype=object)
    families = np.asarray([level.family for level in levels], dtype=object)
    dependencies = np.asarray([generic_dependency_group(level) for level in levels], dtype=object)
    target_keys = set(equivalent_real_keys(item, levels))
    excluded_keys = set(artificial.excluded_real_keys)
    target_mask = np.asarray([key in target_keys for key in keys], dtype=bool)
    excluded_mask = np.asarray([key in excluded_keys for key in keys], dtype=bool)
    rows: list[dict[str, Any]] = []
    contamination_drops = 0
    for index in indexes:
        overlap = real_valid[index] & (
            np.abs(real_level[index] - item.level[index]) <= real_width[index] + width[index]
        )
        if overlap[target_mask].any():
            contamination_drops += 1
            continue
        overlap[excluded_mask] = False
        overlap_indexes = np.flatnonzero(overlap)
        other_count = len(overlap_indexes)
        cross_timeframe = int((timeframes[overlap_indexes] != item.source_timeframe).sum())
        same_timeframe = other_count - cross_timeframe
        cross_family = int((families[overlap_indexes] != item.family).sum())
        same_family = other_count - cross_family
        source_dependency = generic_dependency_group(item)
        cross_dependency = int((dependencies[overlap_indexes] != source_dependency).sum())
        lower_bound = item.level[index] - width[index]
        upper_bound = item.level[index] + width[index]
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
                "level_price": float(item.level[index]),
                "zone_half_width": float(width[index]),
                "zone_half_width_atr": float(width[index] / base_atr[index]),
                "pre_distance_atr": float(
                    abs(item.level[index] - pre_close[index]) / base_atr[index]
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
                "mathematical_alias_level_count": 0,
                "mathematical_alias_level_keys": "",
                "reaction_outcomes_loaded": False,
                "direction_prediction": False,
                "profit_optimization": False,
            }
        )
    return DataFrame(rows), {
        "raw_episode_starts": len(indexes),
        "target_contamination_drops": contamination_drops,
        "retained_events": len(rows),
    }


def artificial_match_rows(
    *,
    actual: DataFrame,
    control: DataFrame,
    pair: str,
    cell: CandidateCell,
    control_name: str,
    state_features: Sequence[str],
) -> list[dict[str, Any]]:
    if actual.empty or control.empty:
        return []
    if control_name not in CONTROL_NAMES:
        raise ValueError(f"Unknown G3F artificial control: {control_name}")
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
                        "route_id": cell.route_id,
                        "density_family": cell.family,
                        "history_hours": cell.history_hours,
                        "level_name": cell.name,
                        "source_timeframe": cell.source_timeframe,
                        "actual_scope": cell.actual_scope,
                        "actual_generic_scope": str(actual_row["generic_scope"]),
                        "actual_cluster_relationship": cell.relationship,
                        "actual_cluster_dependency_relationship": cell.dependency_relationship,
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
                        "actual_source_age_hours": float(actual_row["source_age_hours"]),
                        "control_source_age_hours": float(control_row["source_age_hours"]),
                        "actual_overlap_other_generic_level_count": int(
                            actual_row["overlap_other_generic_level_count"]
                        ),
                        "control_overlap_other_generic_level_count": int(
                            control_row["overlap_other_generic_level_count"]
                        ),
                        "actual_overlap_level_keys": str(actual_row["overlap_level_keys"]),
                        "control_overlap_level_keys": str(control_row["overlap_level_keys"]),
                        "actual_overlap_level_names": str(actual_row["overlap_level_names"]),
                        "control_overlap_level_names": str(control_row["overlap_level_names"]),
                        "actual_overlap_level_families": str(
                            actual_row["overlap_level_families"]
                        ),
                        "control_overlap_level_families": str(
                            control_row["overlap_level_families"]
                        ),
                        "actual_overlap_level_timeframes": str(
                            actual_row["overlap_level_timeframes"]
                        ),
                        "control_overlap_level_timeframes": str(
                            control_row["overlap_level_timeframes"]
                        ),
                        "actual_overlap_dependency_groups": str(
                            actual_row["overlap_dependency_groups"]
                        ),
                        "control_overlap_dependency_groups": str(
                            control_row["overlap_dependency_groups"]
                        ),
                        "control_artificial_donor_key": control_row.get(
                            "artificial_donor_key"
                        ),
                        "control_artificial_construction": str(
                            control_row["artificial_construction"]
                        ),
                        "control_excluded_real_keys": str(
                            control_row["artificial_excluded_real_keys"]
                        ),
                        "matching_state_features": ";".join(state_features),
                        "eligible_actual_events": int(audit["eligible_actual"]),
                        "eligible_control_events": int(audit["eligible_control"]),
                        "geometry_eligible_actual_events": int(
                            audit["geometry_eligible_actual"]
                        ),
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
    invalid_relationships = sorted(
        set(independent["actual_cluster_relationship"].astype(str)).difference(
            GENERIC_RELATIONSHIPS
        )
    )
    invalid_dependencies = sorted(
        set(independent["actual_cluster_dependency_relationship"].astype(str)).difference(
            GENERIC_DEPENDENCY_RELATIONSHIPS
        )
    )
    observed_pairs = sorted(independent["pair"].astype(str).unique())
    return {
        "created_at_utc": utc_now(),
        "passed": bool(
            not failures
            and not independent.empty
            and not invalid_controls
            and separation_violations == 0
            and geometry_difference.le(0.1000001).all()
            and future_source_violations == 0
            and not invalid_relationships
            and not invalid_dependencies
            and independent["direction_prediction"].eq(False).all()
            and independent["profit_optimization"].eq(False).all()
        ),
        "failures": failures,
        "requested_pairs": list(requested_pairs),
        "observed_pairs": observed_pairs,
        "pairs_without_comparisons": sorted(set(requested_pairs).difference(observed_pairs)),
        "independent_comparisons": len(independent),
        "controls_present": sorted(observed_controls),
        "controls_without_comparisons": sorted(set(CONTROL_NAMES).difference(observed_controls)),
        "invalid_controls": invalid_controls,
        "event_separation_violations": separation_violations,
        "pre_distance_atr_abs_difference_max": float(geometry_difference.max()),
        "future_source_violations": future_source_violations,
        "invalid_relationships": invalid_relationships,
        "invalid_dependency_relationships": invalid_dependencies,
        "reaction_outcomes_loaded": True,
        "direction_prediction": False,
        "profit_optimization": False,
    }


def validate_pair_output(path: Path, *, pair: str, request_sha256: str) -> DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "pair",
        "output_schema_version",
        "run_request_sha256",
        "route_id",
        "actual_scope",
        "control",
        "response_window",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Existing G3F artificial output lacks columns {missing}: {path}")
    if frame.empty or set(frame["pair"].astype(str)) != {pair}:
        raise ValueError(f"Existing G3F artificial output has the wrong pair: {path}")
    if set(frame["output_schema_version"].astype(int)) != {OUTPUT_SCHEMA_VERSION}:
        raise ValueError(f"Existing G3F artificial output has the wrong schema: {path}")
    if set(frame["run_request_sha256"].astype(str)) != {request_sha256}:
        raise ValueError(f"Existing G3F artificial output has the wrong request: {path}")
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
    if not frames:
        raise ValueError("No G3F pair produced an artificial-control comparison.")
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    raise SystemExit(main())

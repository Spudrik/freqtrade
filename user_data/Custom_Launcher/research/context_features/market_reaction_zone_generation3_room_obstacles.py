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

from user_data.Custom_Launcher.research.context_features.market_reaction_zone_cluster_generation0 import (  # noqa: E501
    component_dependency_group,
    component_interpretation,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    LARGE_ARTIFACT_ROOT,
    OUTPUT_ROOT,
    atomic_write_json,
    atomic_write_parquet,
    load_manifest,
    prepare_base_market_frame,
    sha256_file,
    utc_now,
    validate_worker_count,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_localization import (  # noqa: E501
    AlignedLevel,
    pair_stem,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_volume_profile_roles import (  # noqa: E501
    assign_attribute_tertiles,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_localization import (  # noqa: E501
    eligible_period_events,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_lvn_attribution import (  # noqa: E501
    QUESTION as LVN_QUESTION,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_lvn_attribution import (  # noqa: E501
    aligned_selected_levels_from_verified_prefix,
    event_source,
    prefix_equivalence_audit,
    select_pairs,
    source_contracts,
    stable_json_sha256,
)


FROZEN_BATCH = OUTPUT_ROOT / "generation2_review" / "g3_frozen_branch_batch.json"
REPORT_ROOT = OUTPUT_ROOT / "generation3_branches" / "g3b_room_obstacles"
ARTIFACT_ROOT = LARGE_ARTIFACT_ROOT / "generation3_branches" / "g3b_room_obstacles"
OUTPUT_SCHEMA_VERSION = 1
OUTCOME_EMBARGO_HOURS = 8
CLOSE_OBSTACLE_GAP_ATR = 0.50
OPEN_ROOM_GAP_ATR = 1.50
MIN_COHORT_PERIOD_EVENTS = 50
MIN_COHORT_PERIOD_COINS = 5

EVENT_COLUMNS = (
    "pair",
    "source_timeframe",
    "level_name",
    "zone_method",
    "period",
    "approach_state",
    "base_index",
    "event_time",
    "level_price",
    "level_score",
    "base_atr",
    "zone_half_width",
    "pre_distance_atr",
    "contact_range_ratio",
    "contact_volume_ratio",
    "contact_pressure_change",
)

GEOMETRY_COLUMNS = (
    "pair",
    "period",
    "event_time",
    "base_index",
    "level_name",
    "approach_state",
    "through_side",
    "level_price",
    "level_score",
    "zone_half_width",
    "base_atr",
    "nearest_higher_tf_identity",
    "nearest_higher_tf_family",
    "nearest_higher_tf_name",
    "nearest_higher_tf_timeframe",
    "nearest_higher_tf_role",
    "nearest_center_distance_atr",
    "nearest_zone_gap_atr",
    "room_class",
    "geometry_contrast",
    "ahead_higher_tf_level_count",
    "close_higher_tf_level_count",
    "close_dependency_group_count",
    "close_activity_level_count",
    "close_acceptance_level_count",
    "close_4h_level_count",
    "close_8h_level_count",
    "close_1d_level_count",
    "multiple_close_independent_groups",
    "close_role_conflict",
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
    cache_sha256_by_timeframe: tuple[tuple[str, str], ...]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 3B coverage-only preflight for approach-relative higher-timeframe "
            "room and obstacles around unusually thin one-hour LVNs. Reaction outcomes "
            "are deliberately not loaded into the geometry artifact."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cohort", choices=("large", "meme"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    validate_worker_count(args.workers, manifest=manifest)
    validate_frozen_branch(args.cohort, manifest)
    pairs = select_pairs(manifest, args.pairs)
    contracts = source_contracts(
        cohort=args.cohort,
        manifest=manifest,
        manifest_path=args.manifest,
        pairs=pairs,
    )
    request = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "frozen_batch_sha256": sha256_file(FROZEN_BATCH),
        "cohort": args.cohort,
        "pairs": list(pairs),
        "source_contracts": contracts,
        "target": {
            "source_timeframe": "1h",
            "level_names": list(LVN_QUESTION.profile.level_names),
            "zone": LVN_QUESTION.zone,
            "thinness_tier": "upper within-period/level/approach tertile",
            "approaches": ["from_below", "from_above"],
        },
        "higher_timeframe_inputs": {
            "timeframes": ["4h", "8h", "1d"],
            "selected_families": (
                "rolling prior high/low, Bollinger boundary, round number, LVN, HVN, "
                "POC and prior POC"
            ),
            "distance": "approach-relative centre and non-overlapping zone-edge gap in prior ATR",
            "role": ["activity_transit", "acceptance_stickiness"],
            "dependency_groups": True,
        },
        "pre_outcome_geometry": {
            "overlapping_ahead_zone": "nearest zone gap <= 0 ATR",
            "close_obstacle": (f"0 < nearest zone gap <= {CLOSE_OBSTACLE_GAP_ATR:.2f} prior ATR"),
            "intermediate_room": (
                f"{CLOSE_OBSTACLE_GAP_ATR:.2f} < gap <= {OPEN_ROOM_GAP_ATR:.2f} prior ATR"
            ),
            "open_room": (
                f"nearest gap > {OPEN_ROOM_GAP_ATR:.2f} prior ATR or no valid ahead level"
            ),
            "primary_contrast": "overlap/close obstacle versus open room",
        },
        "coverage_gate": {
            "minimum_events_per_contrast_period": MIN_COHORT_PERIOD_EVENTS,
            "minimum_coins_per_contrast_period": MIN_COHORT_PERIOD_COINS,
            "must_cover_both_validation_periods": True,
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
    record_path = run_dir / "g3b_preflight_run_record.json"
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
            "G3A showed that unusually thin one-hour LVNs are usually embedded in a "
            "multi-timeframe cluster, but did not distinguish room ahead from an obstacle."
        ),
        "hypothesis": (
            "Approach-relative open room and a nearby higher-timeframe obstacle have "
            "enough repeated common support for a fair direction-neutral reaction test."
        ),
        "pass_fail": (
            "Proceed only where both obstacle and open-room cells have at least 50 events "
            "from at least five coins in both validation periods; otherwise revise or park "
            "the unsupported contrast before reading reaction outcomes."
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
            manifest_path=str(args.manifest.resolve()),
            run_id=args.run_id,
            overwrite=args.overwrite,
            request_sha256=request_sha256,
            cache_sha256_by_timeframe=tuple(
                (str(row["timeframe"]), str(row["current_cache_sha256"]))
                for row in contracts
                if row["pair"] == pair
            ),
        )
        for pair in pairs
    ]
    try:
        results = run_tasks(tasks, workers=args.workers)
        inventory = DataFrame(results)
        atomic_write_parquet(inventory, run_dir / "g3b_pair_inventory.parquet")
        failures = inventory.loc[inventory["status"].eq("failed")]
        if not failures.empty:
            raise RuntimeError(
                f"{len(failures)} pair task(s) failed; inspect g3b_pair_inventory.parquet."
            )
        prefix = combine_pair_outputs(
            directory=run_dir / "pair_prefix_audits",
            pairs=pairs,
            request_sha256=request_sha256,
        )
        geometry = combine_pair_outputs(
            directory=artifact_dir / "pair_geometry",
            pairs=pairs,
            request_sha256=request_sha256,
        )
        coverage = geometry_coverage(geometry)
        contrast = contrast_coverage(geometry)
        atomic_write_parquet(prefix, run_dir / "g3b_prefix_equivalence_audit.parquet")
        atomic_write_parquet(coverage, run_dir / "g3b_geometry_coverage.parquet")
        atomic_write_parquet(contrast, run_dir / "g3b_contrast_coverage.parquet")
        integrity = integrity_record(
            prefix=prefix,
            geometry=geometry,
            contrast=contrast,
            results=results,
        )
        atomic_write_json(integrity, run_dir / "g3b_preflight_integrity.json")
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "pair_tasks": results,
                "prefix_audit_rows": len(prefix),
                "geometry_event_rows": len(geometry),
                "geometry_coverage_rows": len(coverage),
                "contrast_coverage_rows": len(contrast),
                "supported_validation_contrast_rows": int(
                    contrast["contrast_coverage_gate_passed"].sum()
                ),
                "integrity": str(run_dir / "g3b_preflight_integrity.json"),
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
    if "g3b_higher_timeframe_room_and_obstacles" not in branches:
        raise ValueError("The frozen Generation 3B branch is unavailable.")
    boundary = frozen["research_boundary"]
    if boundary.get("direction_prediction") is not False:
        raise ValueError("Generation 3B must keep direction disabled.")
    if boundary.get("profit_optimization") is not False:
        raise ValueError("Generation 3B must keep profit optimization disabled.")
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
    run_dir = REPORT_ROOT / task.run_id
    artifact_dir = ARTIFACT_ROOT / task.run_id
    prefix_path = run_dir / "pair_prefix_audits" / f"{pair_stem(task.pair)}.parquet"
    geometry_path = artifact_dir / "pair_geometry" / f"{pair_stem(task.pair)}.parquet"
    if prefix_path.is_file() and geometry_path.is_file() and not task.overwrite:
        validate_pair_output(prefix_path, pair=task.pair, request_sha256=task.request_sha256)
        geometry = validate_pair_output(
            geometry_path,
            pair=task.pair,
            request_sha256=task.request_sha256,
        )
        return {
            "pair": task.pair,
            "status": "existing",
            "geometry_events": len(geometry),
            "seconds": round(time.perf_counter() - started, 3),
        }

    manifest_path = Path(task.manifest_path)
    manifest = load_manifest(manifest_path)
    event_path, _, _ = event_source(
        cohort=task.cohort,
        manifest=manifest,
        manifest_path=manifest_path,
        pair=task.pair,
        timeframe="1h",
    )
    events = pd.read_parquet(
        event_path,
        filters=[
            ("control", "==", "actual"),
            ("level_name", "in", list(LVN_QUESTION.profile.level_names)),
            ("zone_method", "==", LVN_QUESTION.zone),
        ],
        columns=list(EVENT_COLUMNS),
    )
    events = eligible_period_events(events, manifest, embargo_hours=OUTCOME_EMBARGO_HOURS)
    if events.empty:
        raise ValueError(f"No eligible one-hour LVN events for {task.pair}.")
    base = prepare_base_market_frame(task.pair, manifest)
    prefix = prefix_equivalence_audit(
        pair=task.pair,
        cohort=task.cohort,
        manifest=manifest,
        manifest_path=manifest_path,
        base=base,
        lvn_events=events,
        expected_cache_sha256=dict(task.cache_sha256_by_timeframe),
    )
    if not prefix["passed"].all():
        raise ValueError(f"Historical prefix equivalence failed for {task.pair}.")
    aligned = aligned_selected_levels_from_verified_prefix(
        pair=task.pair,
        base=base,
        manifest_path=manifest_path,
        timeframes=tuple(manifest["data"]["source_timeframes"]),
    )
    tiered = assign_attribute_tertiles(
        events,
        attribute_column=LVN_QUESTION.profile.attribute_column,
        assignment="actual",
        seed_key=f"{task.pair}|g3b_room_obstacle_preflight",
    )
    unusually_thin = tiered.loc[
        tiered["attribute_tier"].eq("high")
        & tiered["approach_state"].isin(("from_below", "from_above"))
    ].copy()
    geometry = attach_room_geometry(unusually_thin, aligned=aligned)
    if geometry.empty:
        raise ValueError(f"No explicit-approach thin-LVN geometry for {task.pair}.")
    geometry = geometry.loc[:, list(GEOMETRY_COLUMNS)].copy()
    for frame in (prefix, geometry):
        frame["output_schema_version"] = OUTPUT_SCHEMA_VERSION
        frame["run_request_sha256"] = task.request_sha256
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(prefix, prefix_path)
    atomic_write_parquet(geometry, geometry_path)
    return {
        "pair": task.pair,
        "status": "completed",
        "eligible_lvn_events": len(events),
        "unusually_thin_explicit_approach_events": len(unusually_thin),
        "geometry_events": len(geometry),
        "prefix_audit_rows": len(prefix),
        "seconds": round(time.perf_counter() - started, 3),
    }


def through_signed_distance(
    approach_state: str,
    *,
    target: float,
    candidate: float,
    atr: float,
) -> float:
    if approach_state == "from_below":
        direction = 1.0
    elif approach_state == "from_above":
        direction = -1.0
    else:
        raise ValueError(f"Unsupported approach state: {approach_state}")
    return float(direction * (candidate - target) / atr)


def classify_room(nearest_gap_atr: float | None) -> tuple[str, str]:
    if nearest_gap_atr is None or not np.isfinite(nearest_gap_atr):
        return "no_ahead_higher_tf_level", "open_room"
    if nearest_gap_atr <= 0.0:
        return "overlapping_ahead_zone", "close_obstacle"
    if nearest_gap_atr <= CLOSE_OBSTACLE_GAP_ATR:
        return "close_obstacle_gap_le_0_5atr", "close_obstacle"
    if nearest_gap_atr <= OPEN_ROOM_GAP_ATR:
        return "intermediate_gap_0_5_to_1_5atr", "intermediate_room"
    return "open_room_gap_gt_1_5atr", "open_room"


def attach_room_geometry(events: DataFrame, *, aligned: Sequence[AlignedLevel]) -> DataFrame:
    if events.empty:
        return DataFrame(columns=GEOMETRY_COLUMNS)
    targets: dict[str, AlignedLevel] = {}
    for level_name in LVN_QUESTION.profile.level_names:
        selected = [
            item for item in aligned if item.timeframe == "1h" and item.spec.name == level_name
        ]
        if len(selected) != 1:
            raise ValueError(f"Expected one 1h {level_name} target; found {len(selected)}.")
        targets[level_name] = selected[0]
    higher = [item for item in aligned if item.timeframe in {"4h", "8h", "1d"}]
    rows: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        base_index = int(event["base_index"])
        target = float(event["level_price"])
        atr = float(event["base_atr"])
        if not np.isfinite(atr) or atr <= 0.0:
            continue
        target_item = targets[str(event["level_name"])]
        current_target = float(target_item.level[base_index])
        if not np.isfinite(current_target) or abs(current_target - target) > max(
            1e-10, abs(target) * 1e-10
        ):
            raise ValueError(f"Current target differs from frozen LVN at row {base_index}.")
        target_width_atr = float(event["zone_half_width"]) / atr
        candidates: list[dict[str, Any]] = []
        for item in higher:
            if not bool(item.valid[base_index]):
                continue
            level = float(item.level[base_index])
            if not np.isfinite(level) or level <= 0.0:
                continue
            centre = through_signed_distance(
                str(event["approach_state"]),
                target=target,
                candidate=level,
                atr=atr,
            )
            if centre <= 0.0:
                continue
            peer_width_atr = max(0.5 * atr, abs(level) * 0.0005) / atr
            role = component_interpretation(item.spec)
            if role not in {"activity_transit", "acceptance_stickiness"}:
                continue
            candidates.append(
                {
                    "identity": (
                        f"{item.timeframe}:{item.spec.family}:{item.spec.name}:"
                        f"{item.spec.representation}"
                    ),
                    "family": item.spec.family,
                    "name": item.spec.name,
                    "timeframe": item.timeframe,
                    "role": role,
                    "dependency_group": component_dependency_group(item.spec),
                    "centre": centre,
                    "gap": centre - target_width_atr - peer_width_atr,
                }
            )
        nearest = min(candidates, key=lambda row: row["centre"]) if candidates else None
        nearest_gap = float(nearest["gap"]) if nearest else None
        room_class, contrast = classify_room(nearest_gap)
        close = [row for row in candidates if row["gap"] <= CLOSE_OBSTACLE_GAP_ATR]
        dependency_groups = {str(row["dependency_group"]) for row in close}
        roles = {str(row["role"]) for row in close}
        rows.append(
            {
                "pair": event["pair"],
                "period": event["period"],
                "event_time": event["event_time"],
                "base_index": base_index,
                "level_name": event["level_name"],
                "approach_state": event["approach_state"],
                "through_side": (
                    "above_level" if event["approach_state"] == "from_below" else "below_level"
                ),
                "level_price": target,
                "level_score": event["level_score"],
                "zone_half_width": event["zone_half_width"],
                "base_atr": atr,
                "nearest_higher_tf_identity": nearest["identity"] if nearest else None,
                "nearest_higher_tf_family": nearest["family"] if nearest else None,
                "nearest_higher_tf_name": nearest["name"] if nearest else None,
                "nearest_higher_tf_timeframe": nearest["timeframe"] if nearest else None,
                "nearest_higher_tf_role": nearest["role"] if nearest else None,
                "nearest_center_distance_atr": nearest["centre"] if nearest else np.nan,
                "nearest_zone_gap_atr": nearest_gap if nearest_gap is not None else np.nan,
                "room_class": room_class,
                "geometry_contrast": contrast,
                "ahead_higher_tf_level_count": len(candidates),
                "close_higher_tf_level_count": len(close),
                "close_dependency_group_count": len(dependency_groups),
                "close_activity_level_count": sum(
                    row["role"] == "activity_transit" for row in close
                ),
                "close_acceptance_level_count": sum(
                    row["role"] == "acceptance_stickiness" for row in close
                ),
                "close_4h_level_count": sum(row["timeframe"] == "4h" for row in close),
                "close_8h_level_count": sum(row["timeframe"] == "8h" for row in close),
                "close_1d_level_count": sum(row["timeframe"] == "1d" for row in close),
                "multiple_close_independent_groups": len(dependency_groups) >= 2,
                "close_role_conflict": {
                    "activity_transit",
                    "acceptance_stickiness",
                }.issubset(roles),
                "direction_prediction": False,
                "profit_optimization": False,
            }
        )
    return DataFrame(rows)


def geometry_coverage(geometry: DataFrame) -> DataFrame:
    group_columns = [
        "pair",
        "period",
        "approach_state",
        "geometry_contrast",
        "room_class",
        "nearest_higher_tf_timeframe",
        "nearest_higher_tf_role",
    ]
    working = geometry.copy()
    working["event_day"] = pd.to_datetime(working["event_time"], utc=True).dt.floor("1D")
    return (
        working.groupby(group_columns, observed=True, dropna=False)
        .agg(
            events=("event_time", "size"),
            unique_event_days=("event_day", "nunique"),
            nearest_center_distance_atr_median=("nearest_center_distance_atr", "median"),
            nearest_zone_gap_atr_median=("nearest_zone_gap_atr", "median"),
            close_level_count_median=("close_higher_tf_level_count", "median"),
            close_dependency_group_count_median=("close_dependency_group_count", "median"),
            role_conflict_fraction=("close_role_conflict", "mean"),
            daily_close_fraction=("close_1d_level_count", lambda values: values.gt(0).mean()),
        )
        .reset_index()
    )


def contrast_coverage(geometry: DataFrame) -> DataFrame:
    grouped = (
        geometry.groupby(["period", "geometry_contrast"], observed=True)
        .agg(
            events=("event_time", "size"),
            coins=("pair", "nunique"),
            from_below_events=("approach_state", lambda values: values.eq("from_below").sum()),
            from_above_events=("approach_state", lambda values: values.eq("from_above").sum()),
            daily_close_fraction=("close_1d_level_count", lambda values: values.gt(0).mean()),
            multiple_group_fraction=("multiple_close_independent_groups", "mean"),
            role_conflict_fraction=("close_role_conflict", "mean"),
        )
        .reset_index()
    )
    grouped["cell_coverage_gate_passed"] = grouped["events"].ge(MIN_COHORT_PERIOD_EVENTS) & grouped[
        "coins"
    ].ge(MIN_COHORT_PERIOD_COINS)
    support = grouped.loc[grouped["geometry_contrast"].isin(("close_obstacle", "open_room"))].pivot(
        index="period", columns="geometry_contrast", values="cell_coverage_gate_passed"
    )
    supported_periods = set(
        support.index[support.fillna(False).all(axis=1)]
        if {"close_obstacle", "open_room"}.issubset(support.columns)
        else []
    )
    grouped["contrast_coverage_gate_passed"] = grouped["period"].isin(supported_periods)
    grouped["direction_prediction"] = False
    grouped["profit_optimization"] = False
    return grouped


def integrity_record(
    *,
    prefix: DataFrame,
    geometry: DataFrame,
    contrast: DataFrame,
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    failures = [row for row in results if row["status"] == "failed"]
    forbidden_outcome_columns = sorted(
        column
        for column in geometry
        if column.startswith(
            (
                "away_excursion_",
                "through_excursion_",
                "volume_ratio_h",
                "dwell_fraction_h",
                "crossings_h",
                "time_to_",
            )
        )
        or column in {"contact_volume_ratio", "contact_pressure_change", "contact_range_ratio"}
    )
    direction_violations = int(geometry["direction_prediction"].ne(False).sum())
    profit_violations = int(geometry["profit_optimization"].ne(False).sum())
    return {
        "created_at_utc": utc_now(),
        "passed": (
            not failures
            and bool(prefix["passed"].all())
            and not geometry.empty
            and not forbidden_outcome_columns
            and direction_violations == 0
            and profit_violations == 0
        ),
        "failures": failures,
        "prefix_audit_rows": len(prefix),
        "prefix_audit_failures": int(prefix["passed"].ne(True).sum()),
        "geometry_event_rows": len(geometry),
        "periods": sorted(geometry["period"].astype(str).unique()),
        "room_classes": sorted(geometry["room_class"].astype(str).unique()),
        "geometry_contrasts": sorted(geometry["geometry_contrast"].astype(str).unique()),
        "forbidden_outcome_columns": forbidden_outcome_columns,
        "supported_validation_contrast_periods": sorted(
            contrast.loc[
                contrast["contrast_coverage_gate_passed"]
                & contrast["period"].astype(str).str.contains("validation"),
                "period",
            ]
            .astype(str)
            .unique()
        ),
        "direction_prediction_violations": direction_violations,
        "profit_optimization_violations": profit_violations,
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

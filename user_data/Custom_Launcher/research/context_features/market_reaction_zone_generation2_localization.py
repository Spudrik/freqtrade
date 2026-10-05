from __future__ import annotations

# Fix numerical-library thread counts before importing numpy/pandas/sklearn.
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
import hashlib
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
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    DEFAULT_MANIFEST,
    OUTPUT_ROOT,
    atomic_write_json,
    atomic_write_parquet,
    extract_episode_events,
    future_path_matrices,
    load_manifest,
    matched_random_time_events,
    numeric_array,
    prepare_base_market_frame,
    sha256_file,
    utc_now,
    validate_worker_count,
    zone_half_width,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_localization import (  # noqa: E501
    LOCAL_MATCH_FEATURES,
    AlignedLevel,
    aligned_selected_levels,
    attach_control_geometry,
    attach_overlap,
    attach_state,
    balance_diagnostics,
    cache_contracts,
    causal_local_state,
    causal_market_context,
    control_definitions,
    level_role,
    nearest_state_pairs,
    matched_cell_rows,
    outcome_metadata,
    outcome_values,
    peer_level_index,
    purge_overlapping_event_pairs_by_key,
    role_oriented_delta,
)


FROZEN_BATCH = OUTPUT_ROOT / "generation1_review" / "g2_frozen_branch_batch.json"
REPORT_ROOT = OUTPUT_ROOT / "generation2_branches" / "g2ab_localization"
MIN_CELL_EVENTS = 20
MAX_STATE_SMD = 0.50
MAX_MEDIAN_STATE_SMD = 0.20
OUTPUT_SCHEMA_VERSION = 1
MIN_PAIR_PERIOD_EPISODES = 5
MIN_COHORT_PERIOD_EPISODES = 50
MIN_COHORT_PERIOD_COINS = 5

LOCALIZATION_MATCH_FEATURES = (*LOCAL_MATCH_FEATURES, "zone_half_width_atr")


@dataclass(frozen=True)
class Target:
    route_id: str
    timeframe: str
    level_columns: tuple[str, ...]
    zone: str
    outcomes: tuple[str, ...]


TARGETS = (
    Target(
        "g2a_round_cluster",
        "8h",
        ("generic_round_nearest",),
        "tight_base_atr",
        (
            "contact_range_ratio",
            "contact_volume_ratio",
            "abs_excursion_atr_h1",
            "volume_ratio_h4",
            "crossings_h1",
        ),
    ),
    Target(
        "g2b_hvn_strength",
        "4h",
        ("vp_hvn_above", "vp_hvn_below"),
        "tight_base_atr",
        ("volume_ratio_h2", "range_ratio_h4", "crossings_h4"),
    ),
    Target(
        "g2b_lvn_thinness",
        "1h",
        ("vp_lvn_above", "vp_lvn_below"),
        "wide_base_atr",
        ("contact_volume_ratio", "dwell_fraction_h4", "volume_ratio_h48"),
    ),
    Target(
        "g2b_poc_evidence",
        "1h",
        ("vp_poc",),
        "tight_base_atr",
        (
            "volume_ratio_h24",
            "range_ratio_h24",
            "volume_ratio_h48",
            "range_ratio_h48",
        ),
    ),
)

TARGET_HORIZONS = {
    target.route_id: tuple(
        sorted({outcome_metadata(outcome)[1] for outcome in target.outcomes if outcome_metadata(outcome)[1]})
    )
    for target in TARGETS
}
INDEPENDENCE_HOURS_BY_ROUTE = {
    route_id: max(horizons) for route_id, horizons in TARGET_HORIZONS.items()
}


@dataclass(frozen=True)
class PairTask:
    pair: str
    manifest_path: str
    run_id: str
    overwrite: bool
    request_sha256: str


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Focused Generation 2 location controls for the frozen round-number and "
            "Volume Profile questions. This does not predict direction or optimize profit."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    validate_worker_count(args.workers, manifest=manifest)
    validate_frozen_branches()
    pairs = select_pairs(manifest, args.pairs)
    timeframes = tuple(manifest["data"]["source_timeframes"])
    contracts = cache_contracts(
        pairs=pairs,
        timeframes=timeframes,
        manifest_path=args.manifest,
    )
    request = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "generation1_helper_sha256": sha256_file(
            Path(__file__).with_name("market_reaction_zone_generation1_localization.py")
        ),
        "manifest_sha256": sha256_file(args.manifest),
        "frozen_batch_sha256": sha256_file(FROZEN_BATCH),
        "pairs": list(pairs),
        "targets": [target.__dict__ for target in TARGETS],
        "cache_contracts": contracts,
        "period_end_embargo_hours": max(manifest["reaction_definition"]["horizons_hours"]),
        "independent_pair_pooling": {
            "greedy_overlap_purge_hours_by_route": INDEPENDENCE_HOURS_BY_ROUTE,
            "minimum_pair_period_episodes": MIN_PAIR_PERIOD_EPISODES,
            "minimum_cohort_period_episodes": MIN_COHORT_PERIOD_EPISODES,
            "minimum_cohort_period_coins": MIN_COHORT_PERIOD_COINS,
            "cohort_effect_weights_coins_equally": True,
        },
        "direction_prediction": False,
        "profit_optimization": False,
    }
    request_sha256 = stable_json_sha256(request)
    run_dir = REPORT_ROOT / args.run_id
    pair_dir = run_dir / "pair_cells"
    run_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    record_path = run_dir / "g2ab_localization_run_record.json"
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
        "baseline": "The frozen G1 localization results on the original large-coin cohort.",
        "hypothesis": (
            "The four selected current locations retain non-directional reaction differences "
            "after stale, causally shuffled, symmetric shifted, market-state, distance, and "
            "independent-cluster controls on the frozen meme cohort."
        ),
        "controls": [
            "current isolated from an independent level",
            "current member of an independent level cluster",
            "cluster member matched directly to isolated current-level contact",
            "168-hour stale location",
            "causal location shuffle",
            "symmetric shifts at 0.25, 0.5, 1.0, and 2.0 ATR",
            "continuous pre-contact local, source, BTC, cohort, density, and distance matching",
            "same live level approached to within two zone widths but not contacted",
            "same-width matched random-time pseudo-level with no real level overlap",
            "outcome-horizon-specific non-overlap and end-of-period embargo",
        ],
        "workers": args.workers,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(record, record_path)
    tasks = [
        PairTask(
            pair=pair,
            manifest_path=str(args.manifest.resolve()),
            run_id=args.run_id,
            overwrite=args.overwrite,
            request_sha256=request_sha256,
        )
        for pair in pairs
    ]
    try:
        results = run_tasks(tasks, workers=args.workers)
        inventory = DataFrame(results)
        atomic_write_parquet(inventory, run_dir / "g2ab_pair_inventory.parquet")
        failures = inventory.loc[inventory["status"].eq("failed")]
        if not failures.empty:
            raise RuntimeError(
                f"{len(failures)} pair task(s) failed; inspect g2ab_pair_inventory.parquet."
            )
        cells = combine_pair_outputs(
            pair_dir=pair_dir,
            pairs=pairs,
            request_sha256=request_sha256,
        )
        atomic_write_parquet(cells, run_dir / "g2ab_localization_cells.parquet")
        screen = repeatability_screen(cells)
        atomic_write_parquet(screen, run_dir / "g2ab_repeatability_screen.parquet")
        matched_pairs = combine_pair_outputs(
            pair_dir=run_dir / "pair_event_matches",
            pairs=pairs,
            request_sha256=request_sha256,
        )
        independent_pairs = purge_overlapping_event_pairs_by_key(
            matched_pairs,
            separation_hours_by_key=INDEPENDENCE_HOURS_BY_ROUTE,
            key_column="route_id",
            group_columns=(
                "pair",
                "route_id",
                "actual_scope",
                "control",
                "period",
            ),
            left_time_column="actual_event_time",
            right_time_column="control_event_time",
        )
        atomic_write_parquet(
            independent_pairs,
            run_dir / "g2ab_independent_event_pairs.parquet",
        )
        pair_period = independent_pair_period_results(independent_pairs)
        atomic_write_parquet(pair_period, run_dir / "g2ab_pair_period_results.parquet")
        cohort_period = independent_cohort_period_results(pair_period, independent_pairs)
        atomic_write_parquet(cohort_period, run_dir / "g2ab_cohort_period_results.parquet")
        leave_one_out = independent_leave_one_coin_out(pair_period)
        atomic_write_parquet(leave_one_out, run_dir / "g2ab_leave_one_coin_out.parquet")
        integrity = integrity_record(cells, results, independent_pairs=independent_pairs)
        atomic_write_json(integrity, run_dir / "g2ab_integrity.json")
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "pair_tasks": results,
                "cell_rows": len(cells),
                "repeatability_rows": len(screen),
                "matched_event_pairs_before_overlap_purge": len(matched_pairs),
                "independent_event_pairs": len(independent_pairs),
                "pair_period_rows": len(pair_period),
                "cohort_period_rows": len(cohort_period),
                "leave_one_coin_out_rows": len(leave_one_out),
                "integrity": str(run_dir / "g2ab_integrity.json"),
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


def validate_frozen_branches() -> None:
    frozen = json.loads(FROZEN_BATCH.read_text(encoding="utf-8"))
    branches = {branch["id"] for branch in frozen["branches"]}
    required = {
        "g2a_8h_round_number_cluster_localization",
        "g2b_volume_profile_attribute_role_confirmation",
    }
    if frozen.get("status") != "frozen_before_generation2_reaction_outcomes":
        raise ValueError("The frozen Generation 2 batch is not ready for outcome work.")
    if not required.issubset(branches):
        raise ValueError("The frozen Generation 2A/2B branches are unavailable.")
    boundary = frozen["research_boundary"]
    if boundary.get("direction_prediction") is not False:
        raise ValueError("Generation 2 localization must keep direction disabled.")
    if boundary.get("profit_optimization") is not False:
        raise ValueError("Generation 2 localization must keep profit optimization disabled.")


def select_pairs(manifest: dict[str, Any], requested: str) -> tuple[str, ...]:
    allowed = tuple(manifest["data"]["pairs"])
    if requested.strip().lower() == "all":
        return allowed
    selected = tuple(value.strip() for value in requested.split(",") if value.strip())
    unknown = sorted(set(selected).difference(allowed))
    if not selected or unknown:
        raise ValueError(f"Invalid pair selection: selected={selected}, unknown={unknown}")
    return selected


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def build_pair(task: PairTask) -> dict[str, Any]:  # noqa: C901
    started = time.perf_counter()
    pair_dir = REPORT_ROOT / task.run_id / "pair_cells"
    output_path = pair_dir / f"{pair_stem(task.pair)}.parquet"
    event_match_path = (
        REPORT_ROOT / task.run_id / "pair_event_matches" / f"{pair_stem(task.pair)}.parquet"
    )
    if output_path.is_file() and not task.overwrite:
        if not event_match_path.is_file():
            raise FileNotFoundError(
                f"Existing cell output has no matched-event companion: {event_match_path}"
            )
        existing = validate_pair_output(
            output_path,
            pair=task.pair,
            request_sha256=task.request_sha256,
        )
        return {
            "pair": task.pair,
            "status": "existing",
            "cell_rows": len(existing),
            "seconds": round(time.perf_counter() - started, 3),
        }
    manifest_path = Path(task.manifest_path)
    manifest = load_manifest(manifest_path)
    base = prepare_base_market_frame(task.pair, manifest)
    state = causal_local_state(base).merge(
        causal_market_context(manifest),
        on="date",
        how="left",
        validate="one_to_one",
    )
    horizons = tuple(int(value) for value in manifest["reaction_definition"]["horizons_hours"])
    paths = future_path_matrices(base, max(horizons))
    aligned = aligned_selected_levels(
        pair=task.pair,
        base=base,
        manifest_path=manifest_path,
        timeframes=tuple(manifest["data"]["source_timeframes"]),
    )
    if not aligned:
        raise ValueError(f"No causal selected levels were available for {task.pair}.")
    real_matrix = np.column_stack([item.level for item in aligned])
    base_atr = numeric_array(base["base_atr"])
    pre_close = numeric_array(base["pre_close"])
    state["state_selected_level_density_2atr"] = np.sum(
        np.isfinite(real_matrix)
        & (np.abs(real_matrix - pre_close[:, None]) <= 2.0 * base_atr[:, None]),
        axis=1,
    ).astype(float)
    output_rows: list[dict[str, Any]] = []
    event_pair_rows: list[dict[str, Any]] = []
    nan_width = np.full(len(base), np.nan, dtype=np.float64)
    target_count = 0
    for source_index, item in enumerate(aligned):
        target = target_for(item)
        if target is None:
            continue
        target_count += 1
        target_horizons = TARGET_HORIZONS[target.route_id]
        target_independence_hours = INDEPENDENCE_HOURS_BY_ROUTE[target.route_id]
        peer = aligned[peer_level_index(source_index, len(aligned), item)]
        controls = control_definitions(
            item=item,
            peer=peer,
            pre_close=pre_close,
            base_atr=base_atr,
        )
        event_state = state.copy()
        for column in item.source_state:
            event_state[column] = numeric_array(item.source_state[column])
        actual_width = zone_half_width(target.zone, item.level, base_atr, nan_width)
        actual = extract_episode_events(
            merged=base,
            paths=paths,
            spec=item.spec,
            pair=task.pair,
            timeframe=item.timeframe,
            level=item.level,
            valid=item.valid,
            half_width=actual_width,
            source_available=item.source_available,
            source_open=item.source_open,
            control="actual",
            zone_method=target.zone,
            horizons=horizons,
            event_kind="contact",
        )
        actual = eligible_period_events(
            actual,
            manifest,
            embargo_hours=target_independence_hours,
        )
        if actual.empty:
            continue
        actual = attach_state(actual, event_state)
        actual = attach_overlap(
            actual,
            real_matrix=real_matrix,
            base_atr=base_atr,
            zone=target.zone,
            source_index=source_index,
        )
        actual = attach_independent_overlap(
            actual,
            aligned_levels=aligned,
            real_matrix=real_matrix,
            base_atr=base_atr,
            zone=target.zone,
            source_index=source_index,
        )
        actual["actual_scope"] = np.where(
            actual["independent_other_level_count"].gt(0),
            "independent_cluster_member",
            "isolated_from_independent_level",
        )
        near_miss = extract_episode_events(
            merged=base,
            paths=paths,
            spec=item.spec,
            pair=task.pair,
            timeframe=item.timeframe,
            level=item.level,
            valid=item.valid,
            half_width=actual_width,
            source_available=item.source_available,
            source_open=item.source_open,
            control="near_miss_same_level",
            zone_method=target.zone,
            horizons=target_horizons,
            event_kind="near_miss",
        )
        near_miss = eligible_period_events(
            near_miss,
            manifest,
            embargo_hours=target_independence_hours,
        )
        if not near_miss.empty:
            near_miss = attach_state(near_miss, event_state)
            near_miss = attach_overlap(
                near_miss,
                real_matrix=real_matrix,
                base_atr=base_atr,
                zone=target.zone,
                source_index=source_index,
            )
            near_miss = attach_independent_overlap(
                near_miss,
                aligned_levels=aligned,
                real_matrix=real_matrix,
                base_atr=base_atr,
                zone=target.zone,
                source_index=source_index,
            )
            near_miss["actual_scope"] = np.where(
                near_miss["independent_other_level_count"].gt(0),
                "independent_cluster_member",
                "isolated_from_independent_level",
            )
            for scope, scoped_actual in actual.groupby("actual_scope", observed=True):
                scoped_near_miss = near_miss.loc[near_miss["actual_scope"].eq(scope)].copy()
                if scoped_near_miss.empty:
                    continue
                rows = matched_cell_rows(
                    actual=scoped_actual.copy(),
                    control=scoped_near_miss,
                    actual_scope=str(scope),
                    control_name="near_miss_same_level",
                    control_scope="same_live_level_approached_but_not_contacted",
                    control_geometry="same_level_near_miss",
                    control_events_before_overlap=scoped_near_miss,
                    role=level_role(item.spec),
                    state_columns=LOCALIZATION_MATCH_FEATURES,
                    horizons=target_horizons,
                )
                output_rows.extend(tag_rows(rows, target.route_id))
                event_pair_rows.extend(
                    matched_localization_event_pairs(
                        actual=scoped_actual,
                        control=scoped_near_miss,
                        target=target,
                        actual_scope=str(scope),
                        control_name="near_miss_same_level",
                        control_scope="same_live_level_approached_but_not_contacted",
                        control_geometry="same_level_near_miss",
                        role=level_role(item.spec),
                        state_features=LOCALIZATION_MATCH_FEATURES,
                        independence_hours=target_independence_hours,
                    )
                )
        random_time = matched_random_time_events(
            merged=base,
            paths=paths,
            actual=actual,
            spec=item.spec,
            pair=task.pair,
            timeframe=item.timeframe,
            zone_method=target.zone,
            horizons=target_horizons,
        )
        random_time = eligible_period_events(
            random_time,
            manifest,
            embargo_hours=target_independence_hours,
        )
        if not random_time.empty:
            random_time = attach_state(random_time, event_state)
            random_time = attach_overlap(
                random_time,
                real_matrix=real_matrix,
                base_atr=base_atr,
                zone=target.zone,
                source_index=source_index,
            )
            random_time = random_time.loc[~random_time["overlap_any_real_level"]].copy()
            for scope, scoped_actual in actual.groupby("actual_scope", observed=True):
                if random_time.empty:
                    continue
                rows = matched_cell_rows(
                    actual=scoped_actual.copy(),
                    control=random_time.copy(),
                    actual_scope=str(scope),
                    control_name="matched_no_level",
                    control_scope="matched_random_time_without_selected_real_level",
                    control_geometry="random_time_pseudo_level",
                    control_events_before_overlap=random_time,
                    role=level_role(item.spec),
                    state_columns=LOCALIZATION_MATCH_FEATURES,
                    horizons=target_horizons,
                )
                output_rows.extend(tag_rows(rows, target.route_id))
                event_pair_rows.extend(
                    matched_localization_event_pairs(
                        actual=scoped_actual,
                        control=random_time,
                        target=target,
                        actual_scope=str(scope),
                        control_name="matched_no_level",
                        control_scope="matched_random_time_without_selected_real_level",
                        control_geometry="random_time_pseudo_level",
                        role=level_role(item.spec),
                        state_features=LOCALIZATION_MATCH_FEATURES,
                        independence_hours=target_independence_hours,
                    )
                )
        cluster = actual.loc[actual["actual_scope"].eq("independent_cluster_member")].copy()
        isolated = actual.loc[actual["actual_scope"].eq("isolated_from_independent_level")].copy()
        if not cluster.empty and not isolated.empty:
            rows = matched_cell_rows(
                actual=cluster,
                control=isolated,
                actual_scope="independent_cluster_member",
                control_name="isolated_current_level",
                control_scope="actual_level_without_independent_cluster_member",
                control_geometry="not_applicable",
                control_events_before_overlap=isolated,
                role=level_role(item.spec),
                state_columns=LOCALIZATION_MATCH_FEATURES,
                horizons=target_horizons,
            )
            output_rows.extend(tag_rows(rows, target.route_id))
            event_pair_rows.extend(
                matched_localization_event_pairs(
                    actual=cluster,
                    control=isolated,
                    target=target,
                    actual_scope="independent_cluster_member",
                    control_name="isolated_current_level",
                    control_scope="actual_level_without_independent_cluster_member",
                    control_geometry="not_applicable",
                    role=level_role(item.spec),
                    state_features=LOCALIZATION_MATCH_FEATURES,
                    independence_hours=target_independence_hours,
                )
            )
        for control_name, definition in controls.items():
            width = zone_half_width(target.zone, definition["level"], base_atr, nan_width)
            events = extract_episode_events(
                merged=base,
                paths=paths,
                spec=item.spec,
                pair=task.pair,
                timeframe=item.timeframe,
                level=definition["level"],
                valid=definition["valid"],
                half_width=width,
                source_available=definition["source_available"],
                source_open=definition["source_open"],
                control=control_name,
                zone_method=target.zone,
                horizons=horizons,
                event_kind="contact",
            )
            events = eligible_period_events(
                events,
                manifest,
                embargo_hours=target_independence_hours,
            )
            if events.empty:
                continue
            events = attach_state(events, event_state)
            events = attach_overlap(
                events,
                real_matrix=real_matrix,
                base_atr=base_atr,
                zone=target.zone,
                source_index=source_index,
            )
            events = attach_control_geometry(
                events,
                control_name=control_name,
                source_level=item.level,
                pre_close=pre_close,
                base_atr=base_atr,
            )
            clean = events.loc[~events["overlap_any_real_level"]].copy()
            if clean.empty:
                continue
            for scope, scoped_actual in actual.groupby("actual_scope", observed=True):
                for geometry, scoped_control in clean.groupby("control_geometry", observed=True):
                    before = events.loc[events["control_geometry"].eq(geometry)]
                    rows = matched_cell_rows(
                        actual=scoped_actual.copy(),
                        control=scoped_control.copy(),
                        actual_scope=str(scope),
                        control_name=control_name,
                        control_scope="no_selected_real_level_overlap",
                        control_geometry=str(geometry),
                        control_events_before_overlap=before,
                        role=level_role(item.spec),
                        state_columns=LOCALIZATION_MATCH_FEATURES,
                        horizons=target_horizons,
                    )
                    output_rows.extend(tag_rows(rows, target.route_id))
                    event_pair_rows.extend(
                        matched_localization_event_pairs(
                            actual=scoped_actual,
                            control=scoped_control,
                            target=target,
                            actual_scope=str(scope),
                            control_name=control_name,
                            control_scope="no_selected_real_level_overlap",
                            control_geometry=str(geometry),
                            role=level_role(item.spec),
                            state_features=LOCALIZATION_MATCH_FEATURES,
                            independence_hours=target_independence_hours,
                        )
                    )
    output = DataFrame(output_rows)
    event_pairs = DataFrame(event_pair_rows)
    if output.empty or event_pairs.empty:
        raise ValueError(f"No matchable Generation 2 localization cells for {task.pair}.")
    output["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    output["run_request_sha256"] = task.request_sha256
    event_pairs["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    event_pairs["run_request_sha256"] = task.request_sha256
    output_path.parent.mkdir(parents=True, exist_ok=True)
    event_match_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(output, output_path)
    atomic_write_parquet(event_pairs, event_match_path)
    return {
        "pair": task.pair,
        "status": "completed",
        "target_level_series": target_count,
        "cell_rows": len(output),
        "matched_event_pairs_before_overlap_purge": len(event_pairs),
        "seconds": round(time.perf_counter() - started, 3),
    }


def matched_localization_event_pairs(
    *,
    actual: DataFrame,
    control: DataFrame,
    target: Target,
    actual_scope: str,
    control_name: str,
    control_scope: str,
    control_geometry: str,
    role: str,
    state_features: Sequence[str],
    independence_hours: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    periods = sorted(set(actual["period"]).intersection(control["period"]))
    for period in periods:
        for approach in (
            "from_below",
            "from_above",
            "already_inside_or_unclear",
        ):
            left = actual.loc[actual["period"].eq(period) & actual["approach_state"].eq(approach)]
            right = control.loc[
                control["period"].eq(period) & control["approach_state"].eq(approach)
            ]
            pairs, _ = nearest_state_pairs(
                left,
                right,
                state_columns=state_features,
                pre_distance_atr_caliper=0.10,
                minimum_event_separation_hours=independence_hours,
            )
            if not pairs:
                continue
            actual_outcomes = {
                outcome: outcome_values(left, outcome) for outcome in target.outcomes
            }
            control_outcomes = {
                outcome: outcome_values(right, outcome) for outcome in target.outcomes
            }
            for actual_position, control_position, distance in pairs:
                actual_row = left.iloc[actual_position]
                control_row = right.iloc[control_position]
                row: dict[str, Any] = {
                    "pair": str(actual_row["pair"]),
                    "route_id": target.route_id,
                    "source_timeframe": str(actual_row["source_timeframe"]),
                    "level_name": str(actual_row["level_name"]),
                    "zone_method": str(actual_row["zone_method"]),
                    "period": str(period),
                    "approach_state": approach,
                    "actual_scope": actual_scope,
                    "control": control_name,
                    "control_scope": control_scope,
                    "control_geometry": control_geometry,
                    "actual_base_index": int(actual_row["base_index"]),
                    "control_base_index": int(control_row["base_index"]),
                    "actual_event_time": pd.Timestamp(actual_row["event_time"]),
                    "control_event_time": pd.Timestamp(control_row["event_time"]),
                    "actual_pre_distance_atr": float(actual_row["pre_distance_atr"]),
                    "control_pre_distance_atr": float(control_row["pre_distance_atr"]),
                    "pre_distance_atr_abs_difference": abs(
                        float(actual_row["pre_distance_atr"])
                        - float(control_row["pre_distance_atr"])
                    ),
                    "event_separation_hours": abs(
                        (
                            pd.Timestamp(actual_row["event_time"])
                            - pd.Timestamp(control_row["event_time"])
                        ).total_seconds()
                        / 3600.0
                    ),
                    "match_distance": float(distance),
                    "level_role": role,
                    "direction_prediction": False,
                    "profit_optimization": False,
                }
                for feature in state_features:
                    row[f"actual_state__{feature}"] = float(actual_row[feature])
                    row[f"control_state__{feature}"] = float(control_row[feature])
                for outcome in target.outcomes:
                    actual_value = float(actual_outcomes[outcome][actual_position])
                    control_value = float(control_outcomes[outcome][control_position])
                    delta = actual_value - control_value
                    metric, _ = outcome_metadata(outcome)
                    row[f"delta__{outcome}"] = delta
                    row[f"role_delta__{outcome}"] = role_oriented_delta(
                        role,
                        metric,
                        delta,
                    )
                rows.append(row)
    return rows


def target_for(item: AlignedLevel) -> Target | None:
    for target in TARGETS:
        if item.timeframe == target.timeframe and item.spec.column in target.level_columns:
            return target
    return None


def eligible_period_events(
    events: DataFrame,
    manifest: dict[str, Any],
    *,
    embargo_hours: int,
) -> DataFrame:
    if events.empty:
        return events.copy()
    times = pd.to_datetime(events["event_time"], utc=True)
    eligible = np.zeros(len(events), dtype=bool)
    for period in manifest["data"]["chronological_periods"]:
        start = pd.Timestamp(period["start_utc"])
        end = pd.Timestamp(period["end_utc_exclusive"]) - pd.Timedelta(hours=embargo_hours)
        eligible |= (times >= start) & (times < end)
    return events.loc[eligible].copy()


def attach_independent_overlap(
    events: DataFrame,
    *,
    aligned_levels: Sequence[AlignedLevel],
    real_matrix: np.ndarray,
    base_atr: np.ndarray,
    zone: str,
    source_index: int,
) -> DataFrame:
    output = events.copy()
    indexes = output["base_index"].to_numpy(dtype=np.int64)
    target_group = component_dependency_group(aligned_levels[source_index].spec)
    peer_indexes = np.asarray(
        [
            index
            for index, item in enumerate(aligned_levels)
            if index != source_index and component_dependency_group(item.spec) != target_group
        ],
        dtype=np.int64,
    )
    if not len(peer_indexes):
        output["independent_other_level_count"] = 0
        return output
    peers = real_matrix[indexes][:, peer_indexes]
    atr = base_atr[indexes]
    multiplier = {
        "tight_base_atr": 0.10,
        "standard_base_atr": 0.25,
        "wide_base_atr": 0.50,
    }[zone]
    peer_width = np.maximum(multiplier * atr[:, None], np.abs(peers) * 0.0005)
    target_level = numeric_array(output["level_price"])
    target_width = numeric_array(output["zone_half_width"])
    overlaps = np.isfinite(peers) & (
        np.abs(peers - target_level[:, None]) <= peer_width + target_width[:, None]
    )
    output["independent_other_level_count"] = overlaps.sum(axis=1).astype(np.int16)
    return output


def tag_rows(rows: Sequence[dict[str, Any]], route_id: str) -> list[dict[str, Any]]:
    return [{**row, "route_id": route_id} for row in rows]


def validate_pair_output(
    path: Path,
    *,
    pair: str,
    request_sha256: str,
) -> DataFrame:
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
    frames = []
    for pair in pairs:
        path = pair_dir / f"{pair_stem(pair)}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(validate_pair_output(path, pair=pair, request_sha256=request_sha256))
    return pd.concat(frames, ignore_index=True)


def independent_pair_period_results(independent_pairs: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    state_features = localization_state_features(independent_pairs)
    targets = {target.route_id: target for target in TARGETS}
    group_columns = ["route_id", "actual_scope", "control", "pair", "period"]
    for key, group in independent_pairs.groupby(group_columns, observed=True, sort=False):
        target = targets[str(key[0])]
        balance = localization_pair_balance(group, state_features)
        for outcome in target.outcomes:
            raw = pd.to_numeric(group[f"delta__{outcome}"], errors="coerce")
            role = pd.to_numeric(group[f"role_delta__{outcome}"], errors="coerce")
            valid = raw.notna() & role.notna()
            if not valid.any():
                continue
            rows.append(
                {
                    "route_id": key[0],
                    "actual_scope": key[1],
                    "control": key[2],
                    "pair": key[3],
                    "period": key[4],
                    "outcome": outcome,
                    "metric_family": outcome_metadata(outcome)[0],
                    "horizon_hours": outcome_metadata(outcome)[1],
                    "independent_event_pairs": int(valid.sum()),
                    "pair_period_coverage_eligible": int(valid.sum()) >= MIN_PAIR_PERIOD_EPISODES,
                    "raw_delta_mean": float(raw.loc[valid].mean()),
                    "raw_delta_median": float(raw.loc[valid].median()),
                    "raw_positive_fraction": float(raw.loc[valid].gt(0.0).mean()),
                    "role_oriented_delta_mean": float(role.loc[valid].mean()),
                    "role_oriented_delta_median": float(role.loc[valid].median()),
                    "role_positive_fraction": float(role.loc[valid].gt(0.0).mean()),
                    "match_distance_median": float(group["match_distance"].median()),
                    "pre_distance_atr_abs_difference_max": float(
                        group["pre_distance_atr_abs_difference"].max()
                    ),
                    "max_absolute_state_smd": balance["max_absolute_smd"],
                    "median_absolute_state_smd": balance["median_absolute_smd"],
                    "state_features_scored": balance["features_scored"],
                    "direction_prediction": False,
                    "profit_optimization": False,
                }
            )
    return DataFrame(rows)


def localization_state_features(frame: DataFrame) -> tuple[str, ...]:
    return tuple(
        sorted(
            column.removeprefix("actual_state__")
            for column in frame
            if column.startswith("actual_state__")
        )
    )


def localization_pair_balance(
    frame: DataFrame,
    state_features: Sequence[str],
) -> dict[str, Any]:
    actual = DataFrame(
        {feature: frame[f"actual_state__{feature}"].to_numpy() for feature in state_features}
    )
    control = DataFrame(
        {feature: frame[f"control_state__{feature}"].to_numpy() for feature in state_features}
    )
    return balance_diagnostics(actual, control, state_features)


def independent_cohort_period_results(
    pair_period: DataFrame,
    independent_pairs: DataFrame,
) -> DataFrame:
    eligible = pair_period.loc[pair_period["pair_period_coverage_eligible"]].copy()
    rows: list[dict[str, Any]] = []
    state_features = localization_state_features(independent_pairs)
    group_columns = ["route_id", "actual_scope", "control", "period", "outcome"]
    for key, group in eligible.groupby(group_columns, observed=True, sort=False):
        pairs = sorted(group["pair"].astype(str).unique().tolist())
        events = independent_pairs.loc[
            independent_pairs["route_id"].eq(key[0])
            & independent_pairs["actual_scope"].eq(key[1])
            & independent_pairs["control"].eq(key[2])
            & independent_pairs["period"].eq(key[3])
            & independent_pairs["pair"].isin(pairs)
        ]
        balance = localization_pair_balance(events, state_features)
        delta = group["role_oriented_delta_mean"]
        event_count = int(group["independent_event_pairs"].sum())
        coverage = (
            len(pairs) >= MIN_COHORT_PERIOD_COINS and event_count >= MIN_COHORT_PERIOD_EPISODES
        )
        rows.append(
            {
                "route_id": key[0],
                "actual_scope": key[1],
                "control": key[2],
                "period": key[3],
                "outcome": key[4],
                "metric_family": outcome_metadata(str(key[4]))[0],
                "horizon_hours": outcome_metadata(str(key[4]))[1],
                "coin_count": len(pairs),
                "coins": ";".join(pairs),
                "independent_event_pairs": event_count,
                "coverage_gate_passed": coverage,
                "pair_delta_median": float(delta.median()),
                "pair_positive_fraction": float(delta.gt(0.0).mean()),
                "pair_delta_q25": float(delta.quantile(0.25)),
                "pair_delta_q75": float(delta.quantile(0.75)),
                "max_absolute_state_smd": balance["max_absolute_smd"],
                "median_absolute_state_smd": balance["median_absolute_smd"],
                "state_features_scored": balance["features_scored"],
                "direction_prediction": False,
                "profit_optimization": False,
            }
        )
    return DataFrame(rows)


def independent_leave_one_coin_out(pair_period: DataFrame) -> DataFrame:
    eligible = pair_period.loc[pair_period["pair_period_coverage_eligible"]].copy()
    rows: list[dict[str, Any]] = []
    group_columns = ["route_id", "actual_scope", "control", "period", "outcome"]
    for key, group in eligible.groupby(group_columns, observed=True, sort=False):
        for omitted in sorted(group["pair"].astype(str).unique()):
            retained = group.loc[~group["pair"].eq(omitted)]
            if retained.empty:
                continue
            delta = retained["role_oriented_delta_mean"]
            rows.append(
                {
                    "route_id": key[0],
                    "actual_scope": key[1],
                    "control": key[2],
                    "period": key[3],
                    "outcome": key[4],
                    "omitted_pair": omitted,
                    "remaining_coin_count": int(retained["pair"].nunique()),
                    "remaining_independent_event_pairs": int(
                        retained["independent_event_pairs"].sum()
                    ),
                    "pair_delta_median": float(delta.median()),
                    "pair_positive_fraction": float(delta.gt(0.0).mean()),
                }
            )
    return DataFrame(rows)


def repeatability_screen(cells: DataFrame) -> DataFrame:
    eligible = cells.loc[
        cells["matched_events"].ge(MIN_CELL_EVENTS)
        & cells["max_absolute_state_smd"].le(MAX_STATE_SMD)
        & cells["median_absolute_state_smd"].le(MAX_MEDIAN_STATE_SMD)
    ].copy()
    if eligible.empty:
        return DataFrame()
    delta_columns = [column for column in eligible if column.startswith("delta_mean__")]
    identifiers = [
        "pair",
        "period",
        "route_id",
        "source_timeframe",
        "level_name",
        "zone_method",
        "actual_scope",
        "control",
        "control_scope",
        "control_geometry",
        "matched_events",
        "max_absolute_state_smd",
        "median_absolute_state_smd",
    ]
    long = eligible.melt(
        id_vars=identifiers,
        value_vars=delta_columns,
        var_name="outcome",
        value_name="delta",
    ).dropna(subset=["delta"])
    long["outcome"] = long["outcome"].str.removeprefix("delta_mean__")
    metadata = long["outcome"].apply(outcome_metadata)
    long["metric_family"] = [item[0] for item in metadata]
    long["horizon_hours"] = [item[1] for item in metadata]
    long["positive"] = long["delta"].gt(0.0)
    keys = [
        "route_id",
        "source_timeframe",
        "level_name",
        "zone_method",
        "actual_scope",
        "control",
        "control_scope",
        "control_geometry",
        "metric_family",
        "horizon_hours",
    ]
    return (
        long.groupby(keys, observed=True, dropna=False, sort=False)
        .agg(
            pair_period_cells=("pair", "size"),
            pair_count=("pair", "nunique"),
            period_count=("period", "nunique"),
            matched_events=("matched_events", "sum"),
            delta_median=("delta", "median"),
            positive_fraction=("positive", "mean"),
            max_state_smd=("max_absolute_state_smd", "max"),
        )
        .reset_index()
    )


def integrity_record(
    cells: DataFrame,
    results: Sequence[dict[str, Any]],
    *,
    independent_pairs: DataFrame,
) -> dict[str, Any]:
    eligible = cells.loc[cells["matched_events"].ge(MIN_CELL_EVENTS)]
    return {
        "created_at_utc": utc_now(),
        "pairs": len(results),
        "pair_failures": sum(row["status"] == "failed" for row in results),
        "cell_rows": len(cells),
        "cells_with_at_least_20_matches": len(eligible),
        "usable_cells": int(
            (
                eligible["max_absolute_state_smd"].le(MAX_STATE_SMD)
                & eligible["median_absolute_state_smd"].le(MAX_MEDIAN_STATE_SMD)
            ).sum()
        ),
        "matched_event_pairs_before_overlap_purge": int(
            sum(row.get("matched_event_pairs_before_overlap_purge", 0) for row in results)
        ),
        "independent_event_pairs_after_horizon_purge": len(independent_pairs),
        "independence_hours_by_route": INDEPENDENCE_HOURS_BY_ROUTE,
        "independent_pairs_by_route_scope_control": {
            f"{route}|{scope}|{control}": int(len(group))
            for (route, scope, control), group in independent_pairs.groupby(
                ["route_id", "actual_scope", "control"], observed=True
            )
        },
        "routes_present": sorted(cells["route_id"].unique().tolist()),
        "periods_present": sorted(cells["period"].unique().tolist()),
        "pre_distance_atr_abs_difference_max": float(
            cells["pre_distance_atr_abs_difference_max"].max()
        ),
        "event_separation_hours_min": float(cells["event_separation_hours_min"].min()),
        "period_end_embargo_hours": 48,
        "direction_prediction": False,
        "profit_optimization": False,
    }


def pair_stem(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")


if __name__ == "__main__":
    raise SystemExit(main())

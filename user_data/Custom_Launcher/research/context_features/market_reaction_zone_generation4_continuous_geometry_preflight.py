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
    pair_stem,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_anchored_vwap import (  # noqa: E501
    stable_json_sha256,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_density_arrival import (  # noqa: E501
    period_roles,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_generic_levels import (  # noqa: E501
    SOURCE_TIMEFRAMES,
    indicator_source_levels,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_lvn_attribution import (  # noqa: E501
    aligned_selected_levels_from_verified_prefix,
    select_pairs,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_room_obstacles import (  # noqa: E501
    ARTIFACT_ROOT as G3B_ARTIFACT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_room_obstacles import (  # noqa: E501
    REPORT_ROOT as G3B_REPORT_ROOT,
)


OUTPUT_SCHEMA_VERSION = 1
FROZEN_BATCH = OUTPUT_ROOT / "generation3_review" / "g4_frozen_branch_batch.json"
G3F_REPORT_ROOT = OUTPUT_ROOT / "generation3_branches" / "g3f_generic_levels"
REPORT_ROOT = (
    OUTPUT_ROOT
    / "generation4_branches"
    / "g4c_continuous_active_region_geometry"
)
ARTIFACT_ROOT = (
    LARGE_ARTIFACT_ROOT
    / "generation4_branches"
    / "g4c_continuous_active_region_geometry"
)

LOCAL_INTERVAL_HALF_WIDTH_ATR = 2.0
CONGESTION_THRESHOLDS_ATR = (0.5, 1.0, 2.0)
ROLE_OBSTACLE_THRESHOLD_ATR = 1.0
PRICE_DISTRIBUTION_HISTORY_HOURS = 168
BOLLINGER_WINDOW_HOURS = 20
MIN_BAND_EVENTS = 50
MIN_BAND_COINS = 5

CANDIDATE_FEATURES: dict[str, dict[str, Any]] = {
    "forward_room_atr": {
        "plain_language": (
            "Empty approach-relative ATR distance from the thin LVN edge to the nearest "
            "causal 4h, 8h, or 1d zone; no zone ahead is ordered above every finite gap."
        ),
        "expected_order": "more room",
        "missing_policy": "no_ahead_level_is_high",
    },
    "independent_groups_within_0_5atr": {
        "plain_language": "Independent higher-timeframe mechanisms no more than 0.5 ATR ahead.",
        "expected_order": "more congestion",
        "missing_policy": "exclude",
    },
    "independent_groups_within_1_0atr": {
        "plain_language": "Independent higher-timeframe mechanisms no more than 1.0 ATR ahead.",
        "expected_order": "more congestion",
        "missing_policy": "exclude",
    },
    "independent_groups_within_2_0atr": {
        "plain_language": "Independent higher-timeframe mechanisms no more than 2.0 ATR ahead.",
        "expected_order": "more congestion",
        "missing_policy": "exclude",
    },
    "higher_tf_zone_coverage_fraction_4atr": {
        "plain_language": (
            "Fraction of the four-ATR interval around the thin LVN covered by causal "
            "higher-timeframe zones."
        ),
        "expected_order": "more occupied price space",
        "missing_policy": "exclude",
    },
    "same_role_groups_within_1atr": {
        "plain_language": (
            "Independent activity/transit mechanisms within one ATR ahead, the same role as an LVN."
        ),
        "expected_order": "more same-role congestion",
        "missing_policy": "exclude",
    },
    "opposite_role_groups_within_1atr": {
        "plain_language": (
            "Independent acceptance/stickiness mechanisms within one ATR ahead, the opposite role."
        ),
        "expected_order": "more role conflict",
        "missing_policy": "exclude",
    },
    "price_distribution_stretch_168h": {
        "plain_language": (
            "Distance from the middle of the causal 168-hour rolling price-rank distribution."
        ),
        "expected_order": "closer to a distribution extreme",
        "missing_policy": "exclude",
    },
    "bollinger_stretch_sigma_1h": {
        "plain_language": (
            "Absolute distance of the previous completed close from its causal 20-hour "
            "average, measured in rolling standard deviations."
        ),
        "expected_order": "farther outside the average region",
        "missing_policy": "exclude",
    },
    "ma_bundle_min_distance_atr": {
        "plain_language": (
            "Smallest ATR distance from the previous completed close to any causal SMA "
            "or EMA in the 1h, 4h, 8h, and 1d moving-average bundle."
        ),
        "expected_order": "farther from the MA bundle",
        "missing_policy": "exclude",
    },
}


@dataclass(frozen=True)
class PairTask:
    pair: str
    cohort: str
    manifest_path: str
    g3b_geometry_path: str
    g3b_geometry_sha256: str
    run_id: str
    request_sha256: str
    overwrite: bool


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 4C outcome-blind continuous room, congestion, distribution-"
            "stretch, and MA-bundle geometry preflight."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cohort", choices=("large", "meme"), required=True)
    parser.add_argument("--g3b-run-id", required=True)
    parser.add_argument("--g3f-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frozen-batch", type=Path, default=FROZEN_BATCH)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    validate_worker_count(args.workers, manifest=manifest)
    pairs = select_pairs(manifest, args.pairs)
    frozen_batch_path = args.frozen_batch.resolve()
    frozen_batch = json.loads(frozen_batch_path.read_text(encoding="utf-8"))
    branch = validate_frozen_branch(frozen_batch)
    contracts = validate_sources(
        cohort=args.cohort,
        pairs=pairs,
        g3b_run_id=args.g3b_run_id,
        g3f_run_id=args.g3f_run_id,
    )
    request = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "frozen_batch_path": str(frozen_batch_path),
        "frozen_batch_sha256": sha256_file(frozen_batch_path),
        "frozen_branch": branch,
        "cohort": args.cohort,
        "pairs": list(pairs),
        "g3b_run_id": args.g3b_run_id,
        "g3f_run_id": args.g3f_run_id,
        "source_contracts": contracts,
        "event_anchor": (
            "The unchanged G3B upper-thinness-tertile one-hour LVN contacts with an "
            "explicit approach from above or below."
        ),
        "candidate_features": CANDIDATE_FEATURES,
        "local_interval_half_width_atr": LOCAL_INTERVAL_HALF_WIDTH_ATR,
        "congestion_thresholds_atr": list(CONGESTION_THRESHOLDS_ATR),
        "role_obstacle_threshold_atr": ROLE_OBSTACLE_THRESHOLD_ATR,
        "band_method": (
            "Freeze pooled development 1/3 and 2/3 quantiles separately for normal "
            "altcoins and memes. Apply normal-alt edges to descriptive BTC. Tied edges "
            "remain unsupported rather than being moved."
        ),
        "coverage_gate": {
            "minimum_events_per_band_period": MIN_BAND_EVENTS,
            "minimum_coins_per_band_period": MIN_BAND_COINS,
            "required_bands": ["low", "middle", "high"],
            "must_repeat_in_both_validation_periods": True,
        },
        "next_test_if_supported": (
            "Only supported features may open direction-neutral outcomes, using OHLCV/"
            "broad-state matching, total-density controls, within-state geometry "
            "permutation, same-state no-level controls, and named-level identity ablation."
        ),
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
    record_path = run_dir / "g4c_preflight_run_record.json"
    if record_path.is_file() and not args.overwrite:
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        if existing.get("request_sha256") != request_sha256:
            raise ValueError("Run ID already exists with an incompatible request contract.")
        if existing.get("status") == "completed":
            print(json.dumps(existing, indent=2, sort_keys=True))
            return 0

    record: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "status": "running",
        "started_at_utc": utc_now(),
        "orchestrator_pid": os.getpid(),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "request_sha256": request_sha256,
        "request_contract": request,
        "baseline": (
            "G3B could not compare exact named obstacles fairly, while G3C and G3F "
            "suggested broader active-region, congestion, and stretch representations."
        ),
        "hypothesis": (
            "At least one causal continuous room, independent-congestion, or stretch "
            "value has enough repeated chronological breadth for a monotonic reaction test."
        ),
        "pass_fail": (
            "Proceed only for a predeclared feature whose development-frozen low, middle, "
            "and high bands each contain at least fifty events from five coins in both "
            "validation periods. Tied bands or insufficient cells are parked before outcomes."
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
            g3b_geometry_path=str(
                G3B_ARTIFACT_ROOT
                / args.g3b_run_id
                / "pair_geometry"
                / f"{pair_stem(pair)}.parquet"
            ),
            g3b_geometry_sha256=contract_for_pair(contracts, pair)["g3b_geometry_sha256"],
            run_id=args.run_id,
            request_sha256=request_sha256,
            overwrite=args.overwrite,
        )
        for pair in pairs
    ]
    try:
        results = run_tasks(tasks, workers=args.workers)
        inventory = DataFrame(results)
        atomic_write_parquet(inventory, run_dir / "g4c_pair_inventory.parquet")
        failures = inventory.loc[inventory["status"].eq("failed")]
        if not failures.empty:
            raise RuntimeError(
                f"{len(failures)} pair feature task(s) failed; inspect g4c_pair_inventory.parquet."
            )
        geometry = combine_pair_outputs(
            pair_dir=pair_dir,
            pairs=pairs,
            request_sha256=request_sha256,
        )
        roles = period_roles(manifest)
        geometry["period_role"] = geometry["period"].map(roles).fillna("unassigned")
        geometry["analysis_scope"] = analysis_scope(geometry, cohort=args.cohort)
        edges = freeze_development_edges(geometry, cohort=args.cohort)
        banded = apply_frozen_bands(geometry, edges)
        coverage = band_coverage(banded)
        support = feature_support(coverage, edges)
        integrity = integrity_record(
            geometry=geometry,
            banded=banded,
            edges=edges,
            coverage=coverage,
            support=support,
            requested_pairs=pairs,
            results=results,
        )
        if not integrity["passed"]:
            raise RuntimeError("G4C continuous-geometry preflight integrity failed.")

        combined_path = artifact_dir / "g4c_continuous_geometry_features.parquet"
        banded_path = artifact_dir / "g4c_banded_geometry_features.parquet"
        atomic_write_parquet(geometry, combined_path)
        atomic_write_parquet(banded, banded_path)
        atomic_write_parquet(edges, run_dir / "g4c_development_band_edges.parquet")
        atomic_write_parquet(coverage, run_dir / "g4c_band_coverage.parquet")
        atomic_write_parquet(support, run_dir / "g4c_feature_support.parquet")
        atomic_write_json(integrity, run_dir / "g4c_preflight_integrity.json")

        supported = support.loc[support["both_validation_periods_supported"]]
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "geometry_event_rows": len(geometry),
                "banded_rows": len(banded),
                "supported_feature_scope_cells": len(supported),
                "supported_feature_scopes": supported.to_dict(orient="records"),
                "preflight_decision": (
                    "reaction_test_may_open_for_supported_cells"
                    if not supported.empty
                    else "parked_no_three_band_validation_support"
                ),
                "integrity": str(run_dir / "g4c_preflight_integrity.json"),
                "compact_outputs": str(run_dir),
                "bulky_outputs": str(artifact_dir),
                "reaction_outcomes_loaded": False,
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

    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def validate_frozen_branch(frozen_batch: dict[str, Any]) -> dict[str, Any]:
    matches = [
        branch
        for branch in frozen_batch.get("branches", [])
        if branch.get("id") == "g4c_continuous_active_region_geometry"
    ]
    if len(matches) != 1:
        raise ValueError("The frozen Generation 4 batch lacks exactly one G4C branch.")
    branch = matches[0]
    if branch.get("status") != "frozen_next_batch":
        raise ValueError("G4C is not frozen for execution.")
    if int(branch.get("iteration_cap", 0)) != 4:
        raise ValueError("G4C iteration cap changed from the frozen value of four.")
    return branch


def validate_sources(  # noqa: C901 - source contracts remain explicit and auditable
    *, cohort: str, pairs: Sequence[str], g3b_run_id: str, g3f_run_id: str
) -> list[dict[str, Any]]:
    g3b_dir = G3B_REPORT_ROOT / g3b_run_id
    g3f_dir = G3F_REPORT_ROOT / g3f_run_id
    g3b_record_path = g3b_dir / "g3b_preflight_run_record.json"
    g3b_integrity_path = g3b_dir / "g3b_preflight_integrity.json"
    g3f_record_path = g3f_dir / "g3f_preflight_run_record.json"
    g3f_integrity_path = g3f_dir / "g3f_preflight_integrity.json"
    for path in (
        g3b_record_path,
        g3b_integrity_path,
        g3f_record_path,
        g3f_integrity_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    g3b_record = json.loads(g3b_record_path.read_text(encoding="utf-8"))
    g3b_integrity = json.loads(g3b_integrity_path.read_text(encoding="utf-8"))
    g3f_record = json.loads(g3f_record_path.read_text(encoding="utf-8"))
    g3f_integrity = json.loads(g3f_integrity_path.read_text(encoding="utf-8"))
    for label, record, integrity in (
        ("G3B", g3b_record, g3b_integrity),
        ("G3F", g3f_record, g3f_integrity),
    ):
        request = record.get("request_contract", {})
        if record.get("status") != "completed" or not integrity.get("passed"):
            raise ValueError(f"{label} source is not complete and clean.")
        if request.get("cohort") != cohort:
            raise ValueError(f"{label} source cohort mismatch.")
        if not set(pairs).issubset(request.get("pairs", [])):
            raise ValueError(f"{label} source lacks a requested pair.")
        if request.get("direction_prediction") is not False:
            raise ValueError(f"{label} source crossed the direction boundary.")
        if request.get("profit_optimization") is not False:
            raise ValueError(f"{label} source crossed the profit boundary.")

    g3b_sources = g3b_record["request_contract"]["source_contracts"]
    g3f_sources = g3f_record["request_contract"]["source_contracts"]
    contracts: list[dict[str, Any]] = [
        {
            "stage": "g3b_record",
            "path": str(g3b_record_path),
            "sha256": sha256_file(g3b_record_path),
        },
        {
            "stage": "g3b_integrity",
            "path": str(g3b_integrity_path),
            "sha256": sha256_file(g3b_integrity_path),
        },
        {
            "stage": "g3f_record",
            "path": str(g3f_record_path),
            "sha256": sha256_file(g3f_record_path),
        },
        {
            "stage": "g3f_integrity",
            "path": str(g3f_integrity_path),
            "sha256": sha256_file(g3f_integrity_path),
        },
    ]
    for pair in pairs:
        geometry_path = (
            G3B_ARTIFACT_ROOT / g3b_run_id / "pair_geometry" / f"{pair_stem(pair)}.parquet"
        )
        if not geometry_path.is_file():
            raise FileNotFoundError(geometry_path)
        pair_contract: dict[str, Any] = {
            "stage": "pair_sources",
            "pair": pair,
            "g3b_geometry_path": str(geometry_path),
            "g3b_geometry_sha256": sha256_file(geometry_path),
            "generation0_cache_contracts": [],
            "ohlcv_contracts": [],
        }
        for row in g3b_sources:
            if row["pair"] != pair:
                continue
            cache_path = Path(row["cache_path"])
            current_hash = sha256_file(cache_path)
            if current_hash != row["current_cache_sha256"]:
                raise ValueError(f"Frozen G3B cache changed: {cache_path}")
            pair_contract["generation0_cache_contracts"].append(
                {
                    "timeframe": row["timeframe"],
                    "path": str(cache_path),
                    "sha256": current_hash,
                }
            )
        for row in g3f_sources:
            if row["pair"] != pair:
                continue
            ohlcv_path = Path(row["path"])
            current_hash = sha256_file(ohlcv_path)
            if current_hash != row["sha256"]:
                raise ValueError(f"Frozen G3F OHLCV changed: {ohlcv_path}")
            pair_contract["ohlcv_contracts"].append(
                {
                    "timeframe": row["timeframe"],
                    "path": str(ohlcv_path),
                    "sha256": current_hash,
                }
            )
        if len(pair_contract["generation0_cache_contracts"]) != 4:
            raise ValueError(f"Incomplete G3B cache contract for {pair}.")
        if len(pair_contract["ohlcv_contracts"]) != 4:
            raise ValueError(f"Incomplete G3F OHLCV contract for {pair}.")
        contracts.append(pair_contract)
    return contracts


def contract_for_pair(contracts: Sequence[dict[str, Any]], pair: str) -> dict[str, Any]:
    matches = [
        row
        for row in contracts
        if row.get("stage") == "pair_sources" and row["pair"] == pair
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one source contract for {pair}; found {len(matches)}.")
    return matches[0]


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
            "geometry_events": len(existing),
            "seconds": round(time.perf_counter() - started, 3),
        }
    geometry_path = Path(task.g3b_geometry_path)
    if sha256_file(geometry_path) != task.g3b_geometry_sha256:
        raise ValueError(f"Frozen G3B geometry changed: {geometry_path}")
    geometry = pd.read_parquet(geometry_path)
    manifest_path = Path(task.manifest_path)
    manifest = load_manifest(manifest_path)
    base = prepare_base_market_frame(task.pair, manifest)
    aligned = aligned_selected_levels_from_verified_prefix(
        pair=task.pair,
        base=base,
        manifest_path=manifest_path,
        timeframes=tuple(manifest["data"]["source_timeframes"]),
    )
    moving_averages = []
    for timeframe in SOURCE_TIMEFRAMES:
        moving_averages.extend(
            level
            for level in indicator_source_levels(
                pair=task.pair,
                base=base,
                timeframe=timeframe,
            )
            if level.family in {"simple_moving_average", "exponential_moving_average"}
        )
    features = continuous_geometry_features(
        geometry,
        base=base,
        aligned=aligned,
        moving_averages=moving_averages,
    )
    if features.empty:
        raise ValueError(f"No continuous G4C geometry rows for {task.pair}.")
    features["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    features["run_request_sha256"] = task.request_sha256
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(features, output_path)
    return {
        "pair": task.pair,
        "status": "completed",
        "geometry_events": len(features),
        "no_ahead_level_events": int(features["no_ahead_higher_tf_level"].sum()),
        "seconds": round(time.perf_counter() - started, 3),
    }


def continuous_geometry_features(  # noqa: C901 - geometry definitions stay explicit
    geometry: DataFrame,
    *,
    base: DataFrame,
    aligned: Sequence[Any],
    moving_averages: Sequence[Any],
) -> DataFrame:
    if geometry.empty:
        return DataFrame()
    output = geometry.copy().sort_values("event_time", kind="stable").reset_index(drop=True)
    indexes = output["base_index"].to_numpy(dtype=np.int64)
    base_dates = pd.to_datetime(base["date"], utc=True)
    event_dates = pd.to_datetime(output["event_time"], utc=True)
    expected_dates = base_dates.iloc[indexes].reset_index(drop=True)
    if not event_dates.reset_index(drop=True).equals(expected_dates):
        raise ValueError("G3B event timestamps no longer match their causal base rows.")

    pre_close = pd.to_numeric(base["pre_close"], errors="coerce")
    distribution_percentile = (
        pre_close.rolling(
            PRICE_DISTRIBUTION_HISTORY_HOURS,
            min_periods=PRICE_DISTRIBUTION_HISTORY_HOURS,
        )
        .rank(method="average", pct=True)
        .to_numpy(dtype=float)
    )
    bb_middle = pre_close.rolling(
        BOLLINGER_WINDOW_HOURS,
        min_periods=BOLLINGER_WINDOW_HOURS,
    ).mean()
    bb_deviation = pre_close.rolling(
        BOLLINGER_WINDOW_HOURS,
        min_periods=BOLLINGER_WINDOW_HOURS,
    ).std(ddof=0)
    bb_stretch = (
        (pre_close - bb_middle).abs() / bb_deviation.replace(0.0, np.nan)
    ).to_numpy(dtype=float)

    higher = [
        item
        for item in aligned
        if item.timeframe in {"4h", "8h", "1d"}
        and component_interpretation(item.spec)
        in {"activity_transit", "acceptance_stickiness"}
    ]
    rows: list[dict[str, Any]] = []
    for position, event in output.iterrows():
        base_index = int(event["base_index"])
        atr = float(event["base_atr"])
        target = float(event["level_price"])
        target_width_atr = float(event["zone_half_width"]) / atr
        direction = 1.0 if event["approach_state"] == "from_below" else -1.0
        group_min_gap: dict[str, float] = {}
        group_role_min_gap: dict[tuple[str, str], float] = {}
        intervals: list[tuple[float, float]] = []
        total_level_count_2atr = 0
        all_side_groups_2atr: set[str] = set()
        nearest_centre = np.inf
        nearest_gap = np.nan
        future_source_violations = 0
        for item in higher:
            if not bool(item.valid[base_index]):
                continue
            level = float(item.level[base_index])
            if not np.isfinite(level) or level <= 0.0:
                continue
            available = pd.Timestamp(item.source_available.iloc[base_index])
            if pd.notna(available) and available > event_dates.iloc[position]:
                future_source_violations += 1
            width = max(0.5 * atr, abs(level) * 0.0005)
            width_atr = width / atr
            absolute_distance_atr = abs(level - target) / atr
            dependency = str(component_dependency_group(item.spec))
            role = str(component_interpretation(item.spec))
            if absolute_distance_atr <= LOCAL_INTERVAL_HALF_WIDTH_ATR + width_atr:
                intervals.append((level - width, level + width))
            if absolute_distance_atr <= 2.0:
                total_level_count_2atr += 1
                all_side_groups_2atr.add(dependency)
            centre = direction * (level - target) / atr
            if centre <= 0.0:
                continue
            gap = centre - target_width_atr - width_atr
            group_min_gap[dependency] = min(group_min_gap.get(dependency, np.inf), gap)
            key = (dependency, role)
            group_role_min_gap[key] = min(group_role_min_gap.get(key, np.inf), gap)
            if centre < nearest_centre:
                nearest_centre = centre
                nearest_gap = gap
        if future_source_violations:
            raise ValueError("A future higher-timeframe level entered G4C geometry.")
        frozen_gap = float(event["nearest_zone_gap_atr"])
        if np.isfinite(frozen_gap) != np.isfinite(nearest_gap):
            raise ValueError("G4C recomputation changed whether a level existed ahead.")
        if np.isfinite(frozen_gap) and abs(frozen_gap - nearest_gap) > 1e-9:
            raise ValueError("G4C recomputation changed the frozen nearest-zone gap.")

        ma_distances: list[float] = []
        causal_ma_count = 0
        price = float(pre_close.iloc[base_index])
        for level in moving_averages:
            if not bool(level.valid[base_index]):
                continue
            value = float(level.level[base_index])
            available = pd.Timestamp(level.source_available.iloc[base_index])
            if pd.notna(available) and available > event_dates.iloc[position]:
                raise ValueError("A future moving-average value entered G4C geometry.")
            if np.isfinite(value) and value > 0.0:
                causal_ma_count += 1
                ma_distances.append(abs(price - value) / atr)

        percentile = distribution_percentile[base_index]
        row = event.to_dict()
        row.update(
            {
                "no_ahead_higher_tf_level": not np.isfinite(nearest_gap),
                "forward_room_atr": nearest_gap,
                "independent_groups_within_0_5atr": count_groups_within(
                    group_min_gap, 0.5
                ),
                "independent_groups_within_1_0atr": count_groups_within(
                    group_min_gap, 1.0
                ),
                "independent_groups_within_2_0atr": count_groups_within(
                    group_min_gap, 2.0
                ),
                "same_role_groups_within_1atr": count_role_groups_within(
                    group_role_min_gap,
                    role="activity_transit",
                    threshold=ROLE_OBSTACLE_THRESHOLD_ATR,
                ),
                "opposite_role_groups_within_1atr": count_role_groups_within(
                    group_role_min_gap,
                    role="acceptance_stickiness",
                    threshold=ROLE_OBSTACLE_THRESHOLD_ATR,
                ),
                "higher_tf_zone_coverage_fraction_4atr": interval_coverage_fraction(
                    intervals,
                    lower=target - LOCAL_INTERVAL_HALF_WIDTH_ATR * atr,
                    upper=target + LOCAL_INTERVAL_HALF_WIDTH_ATR * atr,
                ),
                "total_higher_tf_level_count_within_2atr": total_level_count_2atr,
                "total_higher_tf_dependency_group_count_within_2atr": len(
                    all_side_groups_2atr
                ),
                "price_distribution_percentile_168h": percentile,
                "price_distribution_stretch_168h": (
                    abs(2.0 * percentile - 1.0) if np.isfinite(percentile) else np.nan
                ),
                "bollinger_stretch_sigma_1h": bb_stretch[base_index],
                "ma_bundle_min_distance_atr": (
                    min(ma_distances) if ma_distances else np.nan
                ),
                "causal_ma_bundle_level_count": causal_ma_count,
                "reaction_outcomes_loaded": False,
                "direction_prediction": False,
                "profit_optimization": False,
            }
        )
        rows.append(row)
    return DataFrame(rows)


def count_groups_within(group_min_gap: dict[str, float], threshold: float) -> int:
    return sum(gap <= threshold for gap in group_min_gap.values())


def count_role_groups_within(
    group_role_min_gap: dict[tuple[str, str], float],
    *,
    role: str,
    threshold: float,
) -> int:
    groups = {
        dependency
        for (dependency, candidate_role), gap in group_role_min_gap.items()
        if candidate_role == role and gap <= threshold
    }
    return len(groups)


def interval_coverage_fraction(
    intervals: Sequence[tuple[float, float]], *, lower: float, upper: float
) -> float:
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return np.nan
    clipped = sorted(
        (max(lower, left), min(upper, right))
        for left, right in intervals
        if np.isfinite(left) and np.isfinite(right) and right > lower and left < upper
    )
    if not clipped:
        return 0.0
    covered = 0.0
    active_left, active_right = clipped[0]
    for left, right in clipped[1:]:
        if left <= active_right:
            active_right = max(active_right, right)
        else:
            covered += active_right - active_left
            active_left, active_right = left, right
    covered += active_right - active_left
    return float(covered / (upper - lower))


def analysis_scope(frame: DataFrame, *, cohort: str) -> Series:
    if cohort == "meme":
        return Series("memes", index=frame.index, dtype="object")
    return Series(
        np.where(frame["pair"].astype(str).eq("BTC/USDT:USDT"), "btc", "normal_alts"),
        index=frame.index,
        dtype="object",
    )


def freeze_development_edges(geometry: DataFrame, *, cohort: str) -> DataFrame:
    edge_scope = "memes" if cohort == "meme" else "normal_alts"
    development = geometry.loc[
        geometry["period_role"].eq("development")
        & geometry["analysis_scope"].eq(edge_scope)
    ]
    if development.empty:
        raise ValueError(f"No {edge_scope} development rows for G4C band freezing.")
    rows: list[dict[str, Any]] = []
    for feature, spec in CANDIDATE_FEATURES.items():
        values = pd.to_numeric(development[feature], errors="coerce")
        finite = values.loc[np.isfinite(values)]
        if finite.empty:
            low_edge = np.nan
            high_edge = np.nan
            distinct = False
        else:
            low_edge, high_edge = np.quantile(finite.to_numpy(dtype=float), [1 / 3, 2 / 3])
            distinct = bool(
                np.isfinite(low_edge)
                and np.isfinite(high_edge)
                and low_edge < high_edge
            )
        rows.append(
            {
                "edge_scope": edge_scope,
                "feature": feature,
                "plain_language": spec["plain_language"],
                "expected_order": spec["expected_order"],
                "missing_policy": spec["missing_policy"],
                "development_rows": len(development),
                "finite_development_rows": len(finite),
                "development_coins": int(development["pair"].nunique()),
                "low_upper_edge": low_edge,
                "middle_upper_edge": high_edge,
                "distinct_three_band_edges": distinct,
                "reaction_outcomes_loaded": False,
                "direction_prediction": False,
                "profit_optimization": False,
            }
        )
    return DataFrame(rows)


def apply_frozen_bands(geometry: DataFrame, edges: DataFrame) -> DataFrame:
    frames: list[DataFrame] = []
    edge_by_feature = edges.set_index("feature")
    for feature in CANDIDATE_FEATURES:
        edge = edge_by_feature.loc[feature]
        selected = geometry.copy()
        selected["feature"] = feature
        selected["feature_value"] = pd.to_numeric(selected[feature], errors="coerce")
        selected["feature_band"] = assign_frozen_band(
            selected["feature_value"],
            low_upper=float(edge["low_upper_edge"]),
            middle_upper=float(edge["middle_upper_edge"]),
            distinct=bool(edge["distinct_three_band_edges"]),
            no_ahead=selected["no_ahead_higher_tf_level"],
            missing_policy=str(edge["missing_policy"]),
        )
        frames.append(selected)
    return pd.concat(frames, ignore_index=True)


def assign_frozen_band(
    values: Series,
    *,
    low_upper: float,
    middle_upper: float,
    distinct: bool,
    no_ahead: Series | None = None,
    missing_policy: str = "exclude",
) -> Series:
    output = Series(pd.NA, index=values.index, dtype="object")
    if not distinct:
        return output
    numeric = pd.to_numeric(values, errors="coerce")
    finite = np.isfinite(numeric)
    output.loc[finite & numeric.le(low_upper)] = "low"
    output.loc[finite & numeric.gt(low_upper) & numeric.le(middle_upper)] = "middle"
    output.loc[finite & numeric.gt(middle_upper)] = "high"
    if missing_policy == "no_ahead_level_is_high" and no_ahead is not None:
        output.loc[no_ahead.astype(bool)] = "high"
    return output


def band_coverage(banded: DataFrame) -> DataFrame:
    selected = banded.loc[banded["feature_band"].notna()].copy()
    grouped = (
        selected.groupby(
            ["analysis_scope", "period", "period_role", "feature", "feature_band"],
            observed=True,
        )
        .agg(
            events=("event_time", "size"),
            coins=("pair", "nunique"),
            unique_event_days=(
                "event_time",
                lambda values: pd.to_datetime(values, utc=True).dt.date.nunique(),
            ),
            feature_value_median=("feature_value", "median"),
            no_ahead_fraction=("no_ahead_higher_tf_level", "mean"),
        )
        .reset_index()
    )
    grouped["band_coverage_gate_passed"] = (
        grouped["events"].ge(MIN_BAND_EVENTS) & grouped["coins"].ge(MIN_BAND_COINS)
    )
    grouped["direction_prediction"] = False
    grouped["profit_optimization"] = False
    return grouped


def feature_support(coverage: DataFrame, edges: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for scope in sorted(set(coverage["analysis_scope"]) - {"btc"}):
        periods = (
            coverage.loc[
                coverage["analysis_scope"].eq(scope)
                & coverage["period_role"].eq("chronological_internal_validation"),
                "period",
            ]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        for _, edge in edges.iterrows():
            feature = str(edge["feature"])
            cells = coverage.loc[
                coverage["analysis_scope"].eq(scope)
                & coverage["period"].isin(periods)
                & coverage["feature"].eq(feature)
            ]
            period_pass = []
            for period in periods:
                period_cells = cells.loc[cells["period"].eq(period)]
                band_pass = {
                    band: bool(
                        period_cells.loc[
                            period_cells["feature_band"].eq(band),
                            "band_coverage_gate_passed",
                        ].all()
                    )
                    and bool(len(period_cells.loc[period_cells["feature_band"].eq(band)]))
                    for band in ("low", "middle", "high")
                }
                period_pass.append(all(band_pass.values()))
            rows.append(
                {
                    "analysis_scope": scope,
                    "feature": feature,
                    "validation_periods": ";".join(periods),
                    "validation_period_count": len(periods),
                    "distinct_three_band_edges": bool(edge["distinct_three_band_edges"]),
                    "both_validation_periods_supported": (
                        len(periods) == 2
                        and bool(edge["distinct_three_band_edges"])
                        and all(period_pass)
                    ),
                    "direction_prediction": False,
                    "profit_optimization": False,
                }
            )
    return DataFrame(rows)


def integrity_record(
    *,
    geometry: DataFrame,
    banded: DataFrame,
    edges: DataFrame,
    coverage: DataFrame,
    support: DataFrame,
    requested_pairs: Sequence[str],
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    failures = [row for row in results if row.get("status") == "failed"]
    missing_pairs = sorted(set(requested_pairs) - set(geometry["pair"].astype(str)))
    direction_violations = int(geometry["direction_prediction"].ne(False).sum())
    profit_violations = int(geometry["profit_optimization"].ne(False).sum())
    outcome_violations = int(geometry["reaction_outcomes_loaded"].ne(False).sum())
    band_multiplier_valid = len(banded) == len(geometry) * len(CANDIDATE_FEATURES)
    feature_set_valid = set(edges["feature"].astype(str)) == set(CANDIDATE_FEATURES)
    coverage_feature_set_valid = set(coverage["feature"].astype(str)).issubset(
        CANDIDATE_FEATURES
    )
    return {
        "passed": bool(
            not failures
            and not missing_pairs
            and not geometry.empty
            and direction_violations == 0
            and profit_violations == 0
            and outcome_violations == 0
            and band_multiplier_valid
            and feature_set_valid
            and coverage_feature_set_valid
            and not support.empty
        ),
        "pair_failures": failures,
        "missing_pairs": missing_pairs,
        "geometry_rows": len(geometry),
        "banded_rows": len(banded),
        "edge_rows": len(edges),
        "coverage_rows": len(coverage),
        "support_rows": len(support),
        "direction_violations": direction_violations,
        "profit_violations": profit_violations,
        "reaction_outcome_violations": outcome_violations,
        "band_multiplier_valid": band_multiplier_valid,
        "feature_set_valid": feature_set_valid,
        "coverage_feature_set_valid": coverage_feature_set_valid,
        "coverage_only": True,
        "reaction_outcomes_loaded": False,
    }


def validate_pair_output(path: Path, *, pair: str, request_sha256: str) -> DataFrame:
    frame = pd.read_parquet(path)
    if set(frame["pair"].astype(str)) != {pair}:
        raise ValueError(f"Existing G4C pair output has the wrong pair: {path}")
    if set(frame["output_schema_version"].astype(int)) != {OUTPUT_SCHEMA_VERSION}:
        raise ValueError(f"Existing G4C pair output has the wrong schema: {path}")
    if set(frame["run_request_sha256"].astype(str)) != {request_sha256}:
        raise ValueError(f"Existing G4C pair output has the wrong request hash: {path}")
    return frame


def combine_pair_outputs(
    *, pair_dir: Path, pairs: Sequence[str], request_sha256: str
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


if __name__ == "__main__":
    raise SystemExit(main())

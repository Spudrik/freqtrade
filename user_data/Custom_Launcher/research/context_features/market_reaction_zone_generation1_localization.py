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
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
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
    DEFAULT_MANIFEST,
    LARGE_ARTIFACT_ROOT,
    OUTPUT_ROOT,
    LevelSpec,
    atomic_write_json,
    atomic_write_parquet,
    bool_array,
    extract_episode_events,
    future_path_matrices,
    level_cache_path,
    level_specs,
    load_manifest,
    manifest_storage_paths,
    normalize_dates,
    numeric_array,
    numeric_series,
    prepare_base_market_frame,
    resolved_level_values,
    sha256_file,
    shift_array,
    stable_hash_int,
    utc_now,
    validate_cache_metadata,
    validate_worker_count,
    wilder_atr,
    zone_half_width,
)


REPORT_ROOT = OUTPUT_ROOT / "generation1_localization"
ARTIFACT_ROOT = LARGE_ARTIFACT_ROOT / "generation1_localization"
FROZEN_BATCH = OUTPUT_ROOT / "generation0_review" / "g1_frozen_branch_batch.json"
ZONE_METHODS = ("tight_base_atr", "standard_base_atr", "wide_base_atr")
SHIFT_MULTIPLIERS = (-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0)
MAX_CONTROL_REUSE = 3
MATCH_RMS_CALIPER = 1.0
PRE_DISTANCE_ATR_CALIPER = 0.10
MIN_CELL_EVENTS = 20
OUTPUT_SCHEMA_VERSION = 2
RUN_SCHEMA_VERSION = 2
CONTROL_SCOPE = "no_real_level_overlap"
ACTUAL_SCOPES = ("isolated_level", "level_cluster_member")

SELECTED_LEVEL_COLUMNS = {
    "generic_rolling_high_24",
    "generic_rolling_low_24",
    "generic_rolling_high_168",
    "generic_rolling_low_168",
    "generic_rolling_high_720",
    "generic_rolling_low_720",
    "generic_bb20_upper",
    "generic_bb20_lower",
    "generic_round_nearest",
    "vp_lvn_above",
    "vp_lvn_below",
    "vp_hvn_above",
    "vp_hvn_below",
    "vp_poc",
    "vp_prior_poc",
}

ACTIVITY_LEVELS = {
    "rolling_high_24",
    "rolling_low_24",
    "rolling_high_168",
    "rolling_low_168",
    "rolling_high_720",
    "rolling_low_720",
    "bb20_upper",
    "bb20_lower",
    "round_nearest",
    "lvn_above",
    "lvn_below",
}
ACCEPTANCE_LEVELS = {"hvn_above", "hvn_below", "poc", "prior_poc"}

LOCAL_MATCH_FEATURES = (
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
    "state_source_return_1",
    "state_source_atr_pct",
    "state_source_volume_ratio",
    "state_source_rsi14",
    "state_btc_return_24h",
    "state_top10_breadth",
    "state_top10_mean_abs_return",
    "state_top10_return_dispersion",
    "state_selected_level_density_2atr",
)


@dataclass(frozen=True)
class PairTask:
    pair: str
    manifest_path: str
    run_id: str
    timeframes: tuple[str, ...]
    zones: tuple[str, ...]
    max_rows: int | None
    overwrite: bool
    request_sha256: str


@dataclass
class AlignedLevel:
    timeframe: str
    spec: LevelSpec
    level: np.ndarray
    valid: np.ndarray
    source_available: Series
    source_open: Series
    source_state: DataFrame


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 1A exact-state localization test. It compares causal current "
            "level contacts with stale, causally shuffled, and symmetric shifted locations "
            "without predicting direction or optimizing profit."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--timeframes", default="all")
    parser.add_argument("--zones", default=",".join(ZONE_METHODS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    validate_worker_count(args.workers)
    if args.max_rows is not None and args.max_rows < 1000:
        raise ValueError("--max-rows must be at least 1000 for state matching and future paths.")
    manifest = load_manifest(args.manifest)
    pairs = select_values(args.pairs, tuple(manifest["data"]["pairs"]), "pair")
    timeframes = select_values(
        args.timeframes, tuple(manifest["data"]["source_timeframes"]), "timeframe"
    )
    zones = select_values(args.zones, ZONE_METHODS, "zone")
    validate_frozen_branch()
    horizons = tuple(int(value) for value in manifest["reaction_definition"]["horizons_hours"])
    cache_contract = cache_contracts(
        pairs=pairs,
        timeframes=timeframes,
        manifest_path=args.manifest,
    )
    request = request_contract(
        manifest_path=args.manifest,
        pairs=pairs,
        timeframes=timeframes,
        zones=zones,
        horizons=horizons,
        max_rows=args.max_rows,
        cache_contract=cache_contract,
    )
    request_sha256 = stable_json_sha256(request)

    compact_dir = REPORT_ROOT / args.run_id
    bulky_dir = ARTIFACT_ROOT / args.run_id
    compact_dir.mkdir(parents=True, exist_ok=True)
    bulky_dir.mkdir(parents=True, exist_ok=True)
    record_path = compact_dir / "g1a_run_record.json"
    resumed_from_status: str | None = None
    if record_path.is_file() and not args.overwrite:
        existing_record = json.loads(record_path.read_text(encoding="utf-8"))
        validate_existing_run_record(existing_record, request_sha256=request_sha256)
        resumed_from_status = str(existing_record.get("status", "unknown"))
    record = run_record(
        args=args,
        manifest=manifest,
        pairs=pairs,
        timeframes=timeframes,
        zones=zones,
        compact_dir=compact_dir,
        bulky_dir=bulky_dir,
        request=request,
        request_sha256=request_sha256,
        resumed_from_status=resumed_from_status,
    )
    atomic_write_json(record, record_path)
    tasks = [
        PairTask(
            pair=pair,
            manifest_path=str(args.manifest.resolve()),
            run_id=args.run_id,
            timeframes=timeframes,
            zones=zones,
            max_rows=args.max_rows,
            overwrite=args.overwrite,
            request_sha256=request_sha256,
        )
        for pair in pairs
    ]
    try:
        results = run_tasks(tasks, workers=args.workers)
        inventory = DataFrame(results)
        atomic_write_parquet(inventory, compact_dir / "g1a_pair_inventory.parquet")
        failures = inventory.loc[inventory["status"].eq("failed")]
        if not failures.empty:
            raise RuntimeError(
                f"{len(failures)} G1A pair task(s) failed; inspect g1a_pair_inventory.parquet"
            )
        detail = combine_pair_outputs(
            args.run_id,
            pairs,
            request_sha256=request_sha256,
        )
        detail_path = bulky_dir / "pair_cells"
        screen = repeatability_screen(detail, manifest)
        screen_path = compact_dir / "g1a_repeatability_screen.parquet"
        atomic_write_parquet(screen, screen_path)
        integrity = integrity_record(detail, technical_smoke=args.max_rows is not None)
        integrity_path = compact_dir / "g1a_integrity.json"
        atomic_write_json(integrity, integrity_path)
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "pair_tasks": results,
                "matched_cell_rows": len(detail),
                "repeatability_rows": len(screen),
                "matched_cells": str(detail_path),
                "repeatability_screen": str(screen_path),
                "integrity": str(integrity_path),
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


def select_values(requested: str, allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    if requested.strip().lower() == "all":
        return allowed
    selected = tuple(value.strip() for value in requested.split(",") if value.strip())
    unknown = sorted(set(selected).difference(allowed))
    if not selected or unknown:
        raise ValueError(f"Invalid {label} selection: selected={selected}, unknown={unknown}")
    return selected


def validate_frozen_branch() -> None:
    if not FROZEN_BATCH.is_file():
        raise FileNotFoundError(FROZEN_BATCH)
    frozen = json.loads(FROZEN_BATCH.read_text(encoding="utf-8"))
    branches = {branch["id"]: branch for branch in frozen["branches"]}
    branch = branches.get("g1a_exact_state_localization")
    if branch is None or frozen.get("status") != "frozen_ready_for_execution":
        raise ValueError("The frozen Generation 1A branch is unavailable or no longer ready.")
    if frozen["common_scope"].get("direction_prediction") is not False:
        raise ValueError("Generation 1A must keep direction prediction disabled.")
    if frozen["common_scope"].get("profit_optimization") is not False:
        raise ValueError("Generation 1A must keep profit optimization disabled.")


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def cache_contracts(
    *,
    pairs: Sequence[str],
    timeframes: Sequence[str],
    manifest_path: Path,
) -> list[dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    storage = manifest_storage_paths(manifest)
    contracts: list[dict[str, Any]] = []
    for pair in pairs:
        for timeframe in timeframes:
            cache_path = level_cache_path(
                pair,
                timeframe,
                ("core", "generic"),
                cache_dir=storage.cache_dir,
            )
            metadata = validate_cache_metadata(cache_path, manifest_path)
            metadata_path = cache_path.with_suffix(".meta.json")
            stat = cache_path.stat()
            contracts.append(
                {
                    "pair": pair,
                    "timeframe": timeframe,
                    "cache": str(cache_path.resolve()),
                    "cache_bytes": int(stat.st_size),
                    "cache_modified_ns": int(stat.st_mtime_ns),
                    "cache_metadata_sha256": sha256_file(metadata_path),
                    "source_sha256": str(metadata["source_sha256"]),
                    "indicator_sha256": dict(metadata["indicator_sha256"]),
                }
            )
    return contracts


def request_contract(
    *,
    manifest_path: Path,
    pairs: Sequence[str],
    timeframes: Sequence[str],
    zones: Sequence[str],
    horizons: Sequence[int],
    max_rows: int | None,
    cache_contract: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    generation0_path = Path(__file__).with_name("market_reaction_zone_generation0.py")
    return {
        "run_schema_version": RUN_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "generation0_helper_sha256": sha256_file(generation0_path),
        "manifest_sha256": sha256_file(manifest_path),
        "frozen_batch_sha256": sha256_file(FROZEN_BATCH),
        "pairs": list(pairs),
        "timeframes": list(timeframes),
        "zones": list(zones),
        "horizons_hours": list(horizons),
        "max_rows": max_rows,
        "matching": {
            "state_features": list(LOCAL_MATCH_FEATURES),
            "state_rms_caliper": MATCH_RMS_CALIPER,
            "pre_distance_atr_caliper": PRE_DISTANCE_ATR_CALIPER,
            "maximum_control_reuse": MAX_CONTROL_REUSE,
            "minimum_event_separation_hours": max(horizons),
        },
        "cache_contract": list(cache_contract),
        "direction_prediction": False,
        "profit_optimization": False,
    }


def validate_existing_run_record(record: dict[str, Any], *, request_sha256: str) -> None:
    existing = record.get("request_sha256")
    if existing != request_sha256:
        raise ValueError(
            "Run ID already exists with incompatible settings, code, schema, or cache "
            "contract. Use a new --run-id or explicitly pass --overwrite."
        )


def run_record(
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    pairs: Sequence[str],
    timeframes: Sequence[str],
    zones: Sequence[str],
    compact_dir: Path,
    bulky_dir: Path,
    request: dict[str, Any],
    request_sha256: str,
    resumed_from_status: str | None,
) -> dict[str, Any]:
    record = {
        "schema_version": RUN_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "status": "running",
        "started_at_utc": utc_now(),
        "orchestrator_pid": os.getpid(),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "frozen_branch": "g1a_exact_state_localization",
        "frozen_batch": str(FROZEN_BATCH),
        "frozen_batch_sha256": sha256_file(FROZEN_BATCH),
        "request_sha256": request_sha256,
        "request_contract": request,
        "baseline": "Generation 0B direct atlas and Generation 0C market-state result.",
        "hypothesis": (
            "Selected current causal levels retain a localized non-directional reaction after "
            "continuous pre-contact local, source-timeframe, BTC, and top-ten state is balanced."
        ),
        "pass_interpretation": (
            "Current locations retain coherent role-aligned effects against both sides of the "
            "symmetric distance ladder and against stale and artificial causal locations in "
            "repeated periods and either broad markets or a defensible multi-coin cohort."
        ),
        "park_interpretation": (
            "Pre-contact state, another real level at the control location, or the artificial "
            "and shifted locations explain the measured response."
        ),
        "controls": [
            "stale_168h",
            "causal_location_shuffle",
            *[f"shift_{value:+g}atr" for value in SHIFT_MULTIPLIERS],
        ],
        "control_contamination_rule": (
            "Evidence controls must overlap no selected real level. Actual contacts are split "
            "into isolated-level and level-cluster-member scopes."
        ),
        "matching": {
            "features": list(LOCAL_MATCH_FEATURES),
            "exact_strata": [
                "pair",
                "timeframe",
                "level",
                "zone",
                "actual_scope",
                "control_geometry",
                "period",
                "approach",
            ],
            "robust_scaling": "pooled median and interquartile range inside each exact stratum",
            "distance": "root-mean-square standardized Euclidean distance",
            "caliper": MATCH_RMS_CALIPER,
            "pre_distance_atr_caliper": PRE_DISTANCE_ATR_CALIPER,
            "pre_distance_is_hard_gate_not_soft_feature": True,
            "minimum_event_separation_hours": max(
                manifest["reaction_definition"]["horizons_hours"]
            ),
            "maximum_control_reuse": MAX_CONTROL_REUSE,
            "minimum_scored_cell_events": MIN_CELL_EVENTS,
            "balance_thresholds_are_reporting_lenses_not_rejection_rules": True,
            "contact_candle_state_used_for_matching": False,
            "contact_candle_reported_as_outcome": True,
            "period_wide_future_quantile_bands_used_for_matching": False,
        },
        "pairs": list(pairs),
        "timeframes": list(timeframes),
        "zones": list(zones),
        "horizons_hours": list(manifest["reaction_definition"]["horizons_hours"]),
        "max_rows": args.max_rows,
        "technical_smoke_not_evidence": args.max_rows is not None,
        "workers": args.workers,
        "storage": {
            "compact": str(compact_dir),
            "bulky": str(bulky_dir),
            "raw_control_event_rows_retained": False,
            "reason": (
                "Controls are generated and matched one at a time; only auditable matched-cell "
                "summaries are retained to avoid unnecessary daily/event duplication."
            ),
        },
        "direction_prediction": False,
        "profit_optimization": False,
    }
    if resumed_from_status is not None:
        record["resumed_from_status"] = resumed_from_status
    return record


def run_tasks(tasks: Sequence[PairTask], *, workers: int) -> list[dict[str, Any]]:
    if workers == 1:
        return [safe_build_pair(task) for task in tasks]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(safe_build_pair, task): task for task in tasks}
        for future in as_completed(future_map):
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


def build_pair(task: PairTask) -> dict[str, Any]:  # noqa: C901 - explicit control ladder
    started = time.perf_counter()
    output_dir = ARTIFACT_ROOT / task.run_id / "pair_cells"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pair_stem(task.pair)}.parquet"
    if output_path.is_file() and not task.overwrite:
        existing = validate_pair_output_contract(
            output_path,
            pair=task.pair,
            request_sha256=task.request_sha256,
        )
        return {
            "pair": task.pair,
            "status": "existing",
            "matched_cells": len(existing),
            "path": str(output_path),
            "seconds": round(time.perf_counter() - started, 3),
        }

    manifest_path = Path(task.manifest_path)
    manifest = load_manifest(manifest_path)
    base = prepare_base_market_frame(task.pair, manifest)
    state = causal_local_state(base)
    market = causal_market_context(manifest)
    state = state.merge(market, on="date", how="left", validate="one_to_one")
    if task.max_rows is not None:
        keep = np.arange(max(0, len(base) - task.max_rows), len(base))
        base = base.iloc[keep].reset_index(drop=True)
        state = state.iloc[keep].reset_index(drop=True)
    paths = future_path_matrices(base, max(manifest["reaction_definition"]["horizons_hours"]))
    aligned_levels = aligned_selected_levels(
        pair=task.pair,
        base=base,
        manifest_path=manifest_path,
        timeframes=task.timeframes,
    )
    if not aligned_levels:
        raise ValueError(f"No selected G1A levels were available for {task.pair}.")
    real_matrix = np.column_stack([item.level for item in aligned_levels])
    base_atr = numeric_array(base["base_atr"])
    pre_close = numeric_array(base["pre_close"])
    density = np.sum(
        np.isfinite(real_matrix)
        & (np.abs(real_matrix - pre_close[:, None]) <= 2.0 * base_atr[:, None]),
        axis=1,
    )
    state["state_selected_level_density_2atr"] = density.astype(float)
    horizons = tuple(int(value) for value in manifest["reaction_definition"]["horizons_hours"])
    output_rows: list[dict[str, Any]] = []
    nan_width = np.full(len(base), np.nan, dtype=np.float64)

    for source_index, item in enumerate(aligned_levels):
        if item.timeframe not in task.timeframes:
            continue
        peer_index = peer_level_index(source_index, len(aligned_levels), item)
        peer = aligned_levels[peer_index]
        controls = control_definitions(
            item=item,
            peer=peer,
            pre_close=pre_close,
            base_atr=base_atr,
        )
        event_state = state.copy()
        for column in item.source_state:
            event_state[column] = numeric_array(item.source_state[column])
        for zone in task.zones:
            actual_width = zone_half_width(zone, item.level, base_atr, nan_width)
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
                zone_method=zone,
                horizons=horizons,
                event_kind="contact",
            )
            if actual.empty:
                continue
            actual = attach_state(actual, event_state)
            actual = attach_overlap(
                actual,
                real_matrix=real_matrix,
                base_atr=base_atr,
                zone=zone,
                source_index=source_index,
            )
            actual["actual_scope"] = np.where(
                actual["overlap_other_real_level_count"].eq(0),
                "isolated_level",
                "level_cluster_member",
            )
            for control_name, control in controls.items():
                width = zone_half_width(zone, control["level"], base_atr, nan_width)
                events = extract_episode_events(
                    merged=base,
                    paths=paths,
                    spec=item.spec,
                    pair=task.pair,
                    timeframe=item.timeframe,
                    level=control["level"],
                    valid=control["valid"],
                    half_width=width,
                    source_available=control["source_available"],
                    source_open=control["source_open"],
                    control=control_name,
                    zone_method=zone,
                    horizons=horizons,
                    event_kind="contact",
                )
                if events.empty:
                    continue
                events = attach_state(events, event_state)
                events = attach_overlap(
                    events,
                    real_matrix=real_matrix,
                    base_atr=base_atr,
                    zone=zone,
                    source_index=source_index,
                )
                events = attach_control_geometry(
                    events,
                    control_name=control_name,
                    source_level=item.level,
                    pre_close=pre_close,
                    base_atr=base_atr,
                )
                clean_controls = events.loc[~events["overlap_any_real_level"]].copy()
                if clean_controls.empty:
                    continue
                for actual_scope in ACTUAL_SCOPES:
                    scoped_actual = actual.loc[actual["actual_scope"].eq(actual_scope)].copy()
                    if scoped_actual.empty:
                        continue
                    for control_geometry, scoped_control in clean_controls.groupby(
                        "control_geometry", observed=True, sort=True
                    ):
                        all_geometry_events = events.loc[
                            events["control_geometry"].eq(control_geometry)
                        ]
                        output_rows.extend(
                            matched_cell_rows(
                                actual=scoped_actual,
                                control=scoped_control.copy(),
                                actual_scope=actual_scope,
                                control_name=control_name,
                                control_scope=CONTROL_SCOPE,
                                control_geometry=str(control_geometry),
                                control_events_before_overlap=all_geometry_events.copy(),
                                role=level_role(item.spec),
                                state_columns=LOCAL_MATCH_FEATURES,
                                horizons=horizons,
                            )
                        )

    output = DataFrame(output_rows)
    if output.empty:
        raise ValueError(f"No matchable G1A cells were produced for {task.pair}.")
    output["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    output["run_request_sha256"] = task.request_sha256
    atomic_write_parquet(output, output_path)
    return {
        "pair": task.pair,
        "status": "completed",
        "matched_cells": len(output),
        "levels": len(aligned_levels),
        "path": str(output_path),
        "seconds": round(time.perf_counter() - started, 3),
    }


def validate_pair_output_contract(path: Path, *, pair: str, request_sha256: str) -> DataFrame:
    required = [
        "pair",
        "output_schema_version",
        "run_request_sha256",
        "actual_scope",
        "control_scope",
        "control_geometry",
        "pre_distance_atr_abs_difference_median",
        "event_separation_hours_min",
    ]
    try:
        contract = pd.read_parquet(path, columns=required)
    except Exception as exc:
        raise ValueError(
            f"Existing pair output is missing the repaired G1A contract: {path}"
        ) from exc
    if contract.empty:
        raise ValueError(f"Existing pair output is empty: {path}")
    if set(contract["pair"].astype(str)) != {pair}:
        raise ValueError(f"Existing pair output contains an incompatible pair: {path}")
    if set(contract["output_schema_version"].astype(int)) != {OUTPUT_SCHEMA_VERSION}:
        raise ValueError(f"Existing pair output has an incompatible schema: {path}")
    if set(contract["run_request_sha256"].astype(str)) != {request_sha256}:
        raise ValueError(f"Existing pair output belongs to a different run request: {path}")
    return contract


def aligned_selected_levels(
    *,
    pair: str,
    base: DataFrame,
    manifest_path: Path,
    timeframes: Sequence[str],
) -> list[AlignedLevel]:
    manifest = load_manifest(manifest_path)
    storage = manifest_storage_paths(manifest)
    selected: list[AlignedLevel] = []
    for timeframe in timeframes:
        cache_path = level_cache_path(
            pair,
            timeframe,
            ("core", "generic"),
            cache_dir=storage.cache_dir,
        )
        validate_cache_metadata(cache_path, manifest_path)
        cache = pd.read_parquet(cache_path).sort_values("available_at").reset_index(drop=True)
        cache["available_at"] = normalize_dates(cache["available_at"])
        cache["source_open"] = normalize_dates(cache["source_open"])
        source_state = causal_source_state(cache)
        cache = pd.concat([cache, source_state], axis=1)
        aligned = pd.merge_asof(
            base[["date"]].sort_values("date"),
            cache,
            left_on="date",
            right_on="available_at",
            direction="backward",
            allow_exact_matches=True,
        )
        causal = aligned["available_at"].notna()
        if (aligned.loc[causal, "available_at"] > aligned.loc[causal, "date"]).any():
            raise AssertionError(f"Future level row admitted for {pair} {timeframe}.")
        source_columns = [column for column in aligned if column.startswith("state_source_")]
        aligned_source_state = aligned[source_columns].reset_index(drop=True).copy()
        for spec in level_specs(cache, ("g0b1", "g0b2")):
            if spec.column not in SELECTED_LEVEL_COLUMNS:
                continue
            level = resolved_level_values(aligned, spec, timeframe)
            valid = np.isfinite(level) & (level > 0.0)
            if spec.active_columns:
                active = np.zeros(len(aligned), dtype=bool)
                for column in spec.active_columns:
                    if column in aligned:
                        active |= bool_array(aligned[column])
                valid &= active
            selected.append(
                AlignedLevel(
                    timeframe=timeframe,
                    spec=spec,
                    level=level,
                    valid=valid,
                    source_available=normalize_dates(aligned["available_at"]),
                    source_open=normalize_dates(aligned["source_open"]),
                    source_state=aligned_source_state,
                )
            )
    return selected


def causal_local_state(base: DataFrame) -> DataFrame:
    close = numeric_series(base["close"])
    high = numeric_series(base["high"])
    low = numeric_series(base["low"])
    volume = numeric_series(base["volume"]).clip(lower=0.0)
    pre_close = close.shift(1)
    atr = numeric_series(base["base_atr"]).replace(0.0, np.nan)
    candle_range = high - low
    pressure = (2.0 * close - high - low).div(candle_range.replace(0.0, np.nan))
    previous_range = candle_range.shift(1)
    previous_volume = volume.shift(1)
    range_baseline = candle_range.shift(2).rolling(24, min_periods=12).median()
    volume_baseline = volume.shift(2).rolling(24, min_periods=12).median()
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rsi = 1.0 - 1.0 / (1.0 + gain.div(loss.replace(0.0, np.nan)))
    mid = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std(ddof=0)
    ema12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
    ema26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, min_periods=9, adjust=False).mean()
    ema50 = close.ewm(span=50, min_periods=50, adjust=False).mean()
    signed_volume = pressure.fillna(0.0) * volume
    result = DataFrame({"date": base["date"]})
    for horizon in (1, 4, 24):
        result[f"state_local_return_{horizon}h"] = close.pct_change(horizon).shift(1)
    result["state_local_atr_pct"] = atr.div(pre_close.abs())
    result["state_local_range_ratio"] = previous_range.div(range_baseline.replace(0.0, np.nan))
    result["state_local_volume_ratio"] = previous_volume.div(volume_baseline.replace(0.0, np.nan))
    result["state_local_pressure_6h"] = (
        signed_volume.rolling(6, min_periods=3)
        .sum()
        .div(volume.rolling(6, min_periods=3).sum().replace(0.0, np.nan))
        .shift(1)
    )
    result["state_local_rsi14"] = rsi.shift(1)
    result["state_local_bb_position"] = pre_close.sub(mid.shift(1)).div(
        (2.0 * std.shift(1)).replace(0.0, np.nan)
    )
    result["state_local_bb_width_atr"] = (4.0 * std.shift(1)).div(atr)
    result["state_local_macd_hist_atr"] = (macd - macd_signal).shift(1).div(atr)
    result["state_local_ema50_gap_atr"] = pre_close.sub(ema50.shift(1)).div(atr)
    return result.replace([np.inf, -np.inf], np.nan)


def causal_source_state(cache: DataFrame) -> DataFrame:
    close = numeric_series(cache["source_close"])
    high = numeric_series(cache["source_high"])
    low = numeric_series(cache["source_low"])
    volume = numeric_series(cache["source_volume"]).clip(lower=0.0)
    atr = numeric_series(cache["source_atr_14"]).replace(0.0, np.nan)
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rsi = 1.0 - 1.0 / (1.0 + gain.div(loss.replace(0.0, np.nan)))
    result = DataFrame(index=cache.index)
    result["state_source_return_1"] = close.pct_change()
    result["state_source_atr_pct"] = atr.div(close.abs())
    result["state_source_volume_ratio"] = volume.div(
        volume.shift(1).rolling(24, min_periods=12).median().replace(0.0, np.nan)
    )
    result["state_source_rsi14"] = rsi
    result["state_source_range_atr"] = (high - low).div(atr)
    return result.replace([np.inf, -np.inf], np.nan)


def causal_market_context(manifest: dict[str, Any]) -> DataFrame:
    cohort_pairs = list(manifest["data"]["pairs"])
    btc = "BTC/USDT:USDT"
    context_pairs = list(cohort_pairs)
    if btc not in context_pairs:
        context_pairs.append(btc)
    frames: dict[str, DataFrame] = {}
    for pair in context_pairs:
        raw = prepare_base_market_frame(pair, manifest)
        close = numeric_series(raw["close"])
        volume = numeric_series(raw["volume"]).clip(lower=0.0)
        available = DataFrame(
            {
                "date": raw["date"] + pd.Timedelta(hours=1),
                f"ret__{pair}": close.pct_change(),
                f"ret24__{pair}": close.pct_change(24),
                f"atrpct__{pair}": wilder_atr(raw, 14).div(close.abs()),
                f"vol__{pair}": volume.div(
                    volume.shift(1).rolling(24, min_periods=12).median().replace(0.0, np.nan)
                ),
            }
        )
        frames[pair] = available
    merged: DataFrame | None = None
    for frame in frames.values():
        merged = frame if merged is None else merged.merge(frame, on="date", how="outer")
    if merged is None:
        return DataFrame(columns=["date"])
    ret_columns = [f"ret__{pair}" for pair in cohort_pairs]
    vol_columns = [f"vol__{pair}" for pair in cohort_pairs]
    returns = merged[ret_columns]
    output = DataFrame({"date": merged["date"]})
    output["state_btc_return_1h"] = merged[f"ret__{btc}"]
    output["state_btc_return_24h"] = merged[f"ret24__{btc}"]
    output["state_btc_atr_pct"] = merged[f"atrpct__{btc}"]
    output["state_top10_breadth"] = (
        returns.gt(0.0).sum(axis=1).div(returns.notna().sum(axis=1).replace(0.0, np.nan))
    )
    output["state_top10_mean_abs_return"] = returns.abs().mean(axis=1)
    output["state_top10_return_dispersion"] = returns.std(axis=1, ddof=0)
    output["state_top10_common_volume"] = merged[vol_columns].median(axis=1)
    return output.sort_values("date").drop_duplicates("date", keep="last")


def peer_level_index(source_index: int, count: int, item: AlignedLevel) -> int:
    if count < 2:
        raise ValueError("Causal location shuffle needs at least two selected levels.")
    offset = 1 + stable_hash_int(f"{item.timeframe}|{item.spec.column}") % (count - 1)
    return (source_index + offset) % count


def control_definitions(
    *, item: AlignedLevel, peer: AlignedLevel, pre_close: np.ndarray, base_atr: np.ndarray
) -> dict[str, dict[str, Any]]:
    controls: dict[str, dict[str, Any]] = {}
    stale_level = shift_array(item.level, 168)
    controls["stale_168h"] = {
        "level": stale_level,
        "valid": shift_array(item.valid.astype(float), 168) == 1.0,
        "source_available": item.source_available.shift(168),
        "source_open": item.source_open.shift(168),
    }
    shuffled_level = 2.0 * pre_close - peer.level
    controls["causal_location_shuffle"] = {
        "level": shuffled_level,
        "valid": item.valid & peer.valid & np.isfinite(shuffled_level) & (shuffled_level > 0.0),
        "source_available": peer.source_available,
        "source_open": peer.source_open,
    }
    for multiplier in SHIFT_MULTIPLIERS:
        shifted = item.level + multiplier * base_atr
        name = f"shift_{multiplier:+g}atr"
        controls[name] = {
            "level": shifted,
            "valid": item.valid & np.isfinite(shifted) & (shifted > 0.0),
            "source_available": item.source_available,
            "source_open": item.source_open,
        }
    return controls


def attach_state(events: DataFrame, state: DataFrame) -> DataFrame:
    indexes = events["base_index"].to_numpy(dtype=np.int64)
    state_columns = [column for column in state if column.startswith("state_")]
    output = events.copy()
    for column in state_columns:
        output[column] = numeric_array(state[column])[indexes]
    return output


def attach_overlap(
    events: DataFrame,
    *,
    real_matrix: np.ndarray,
    base_atr: np.ndarray,
    zone: str,
    source_index: int,
) -> DataFrame:
    output = events.copy()
    indexes = output["base_index"].to_numpy(dtype=np.int64)
    control_level = numeric_array(output["level_price"])
    control_width = numeric_array(output["zone_half_width"])
    real = real_matrix[indexes]
    atr = base_atr[indexes]
    multiplier = {
        "tight_base_atr": 0.10,
        "standard_base_atr": 0.25,
        "wide_base_atr": 0.50,
    }[zone]
    real_width = np.maximum(multiplier * atr[:, None], np.abs(real) * 0.0005)
    overlap = np.isfinite(real) & (
        np.abs(real - control_level[:, None]) <= real_width + control_width[:, None]
    )
    source_overlap = overlap[:, source_index]
    counts = overlap.sum(axis=1)
    output["overlap_any_real_level"] = counts > 0
    output["overlap_real_level_count"] = counts.astype(np.int16)
    output["overlap_source_current_level"] = source_overlap
    output["overlap_other_real_level_count"] = (counts - source_overlap.astype(int)).astype(
        np.int16
    )
    return output


def attach_control_geometry(
    events: DataFrame,
    *,
    control_name: str,
    source_level: np.ndarray,
    pre_close: np.ndarray,
    base_atr: np.ndarray,
) -> DataFrame:
    output = events.copy()
    output["control_geometry"] = "not_applicable"
    output["control_shift_atr"] = np.nan
    if not control_name.startswith("shift_"):
        return output
    indexes = output["base_index"].to_numpy(dtype=np.int64)
    atr = base_atr[indexes]
    source_offset = (source_level[indexes] - pre_close[indexes]) / atr
    control_offset = (numeric_array(output["level_price"]) - pre_close[indexes]) / atr
    crossed = (
        np.isfinite(source_offset)
        & np.isfinite(control_offset)
        & (np.sign(source_offset) != 0)
        & (np.sign(control_offset) != 0)
        & (np.sign(source_offset) != np.sign(control_offset))
    )
    distance_change = np.abs(control_offset) - np.abs(source_offset)
    geometry = np.select(
        [
            crossed,
            distance_change > 1e-12,
            distance_change < -1e-12,
        ],
        ["crossed_price", "outward", "inward"],
        default="same_distance",
    )
    output["control_geometry"] = geometry
    output["control_shift_atr"] = float(control_name.removeprefix("shift_").removesuffix("atr"))
    return output


def matched_cell_rows(
    *,
    actual: DataFrame,
    control: DataFrame,
    actual_scope: str,
    control_name: str,
    control_scope: str,
    control_geometry: str,
    control_events_before_overlap: DataFrame,
    role: str,
    state_columns: Sequence[str],
    horizons: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    periods = sorted(set(actual["period"]).intersection(control["period"]))
    for period in periods:
        actual_period = actual.loc[actual["period"].eq(period)].copy()
        control_period = control.loc[control["period"].eq(period)].copy()
        matched_actual: list[int] = []
        matched_control: list[int] = []
        distances: list[float] = []
        approach_counts: dict[str, int] = {}
        eligible_actual = 0
        eligible_control = 0
        geometry_eligible_actual = 0
        state_matchable_actual = 0
        for approach in (
            "from_below",
            "from_above",
            "already_inside_or_unclear",
        ):
            left = actual_period.loc[actual_period["approach_state"].eq(approach)]
            right = control_period.loc[control_period["approach_state"].eq(approach)]
            pairs, audit = nearest_state_pairs(
                left,
                right,
                state_columns=state_columns,
                pre_distance_atr_caliper=PRE_DISTANCE_ATR_CALIPER,
                minimum_event_separation_hours=max(horizons),
            )
            eligible_actual += audit["eligible_actual"]
            eligible_control += audit["eligible_control"]
            geometry_eligible_actual += audit["geometry_eligible_actual"]
            state_matchable_actual += audit["state_matchable_actual"]
            approach_counts[approach] = len(pairs)
            if pairs:
                matched_actual.extend(left.index[pair[0]] for pair in pairs)
                matched_control.extend(right.index[pair[1]] for pair in pairs)
                distances.extend(pair[2] for pair in pairs)
        if not matched_actual:
            continue
        left = actual.loc[matched_actual].reset_index(drop=True)
        right = control.loc[matched_control].reset_index(drop=True)
        balance = balance_diagnostics(left, right, state_columns)
        pre_distance_difference = np.abs(
            numeric_array(left["pre_distance_atr"]) - numeric_array(right["pre_distance_atr"])
        )
        separation_hours = (
            (
                pd.to_datetime(left["event_time"], utc=True)
                - pd.to_datetime(right["event_time"], utc=True)
            )
            .abs()
            .dt.total_seconds()
            .div(3600.0)
            .to_numpy(dtype=float)
        )
        row: dict[str, Any] = {
            "pair": str(left["pair"].iloc[0]),
            "source_timeframe": str(left["source_timeframe"].iloc[0]),
            "level_family": str(left["level_family"].iloc[0]),
            "level_name": str(left["level_name"].iloc[0]),
            "level_column": str(left["level_column"].iloc[0]),
            "zone_method": str(left["zone_method"].iloc[0]),
            "period": str(period),
            "level_role": role,
            "actual_scope": actual_scope,
            "control": control_name,
            "control_scope": control_scope,
            "control_geometry": control_geometry,
            "actual_events": len(actual_period),
            "control_events": len(control_period),
            "control_events_before_overlap": len(
                control_events_before_overlap.loc[
                    control_events_before_overlap["period"].eq(period)
                ]
            ),
            "eligible_actual_events": eligible_actual,
            "eligible_control_events": eligible_control,
            "geometry_eligible_actual_events": geometry_eligible_actual,
            "state_matchable_actual_events": state_matchable_actual,
            "matched_events": len(left),
            "match_coverage": len(left) / eligible_actual if eligible_actual else 0.0,
            "unique_control_events": len(set(matched_control)),
            "maximum_control_reuse_observed": max_reuse(matched_control),
            "match_distance_median": float(np.median(distances)),
            "match_distance_q90": float(np.quantile(distances, 0.90)),
            "max_absolute_state_smd": balance["max_absolute_smd"],
            "median_absolute_state_smd": balance["median_absolute_smd"],
            "state_features_scored": balance["features_scored"],
            "pre_distance_atr_caliper": PRE_DISTANCE_ATR_CALIPER,
            "actual_pre_distance_atr_median": float(
                np.median(numeric_array(left["pre_distance_atr"]))
            ),
            "control_pre_distance_atr_median": float(
                np.median(numeric_array(right["pre_distance_atr"]))
            ),
            "pre_distance_atr_abs_difference_median": float(np.median(pre_distance_difference)),
            "pre_distance_atr_abs_difference_max": float(np.max(pre_distance_difference)),
            "minimum_event_separation_hours": max(horizons),
            "event_separation_hours_min": float(np.min(separation_hours)),
            "event_separation_hours_median": float(np.median(separation_hours)),
            "matched_from_below": approach_counts.get("from_below", 0),
            "matched_from_above": approach_counts.get("from_above", 0),
            "matched_inside_or_unclear": approach_counts.get("already_inside_or_unclear", 0),
            "control_overlap_any_real_fraction": float(right["overlap_any_real_level"].mean()),
            "control_overlap_other_real_fraction": float(
                right["overlap_other_real_level_count"].gt(0).mean()
            ),
            "control_overlap_source_real_fraction": float(
                right["overlap_source_current_level"].mean()
            ),
            "actual_overlap_other_real_fraction": float(
                left["overlap_other_real_level_count"].gt(0).mean()
            ),
            "actual_overlap_other_real_count_median": float(
                left["overlap_other_real_level_count"].median()
            ),
            "actual_contact_close_distance_atr_median": float(
                left["contact_close_distance_atr"].median()
            ),
            "control_contact_close_distance_atr_median": float(
                right["contact_close_distance_atr"].median()
            ),
            "direction_prediction": False,
            "profit_optimization": False,
        }
        for column in outcome_columns(horizons):
            left_values = outcome_values(left, column)
            right_values = outcome_values(right, column)
            valid = np.isfinite(left_values) & np.isfinite(right_values)
            if not valid.any():
                continue
            row[f"scored__{column}"] = int(valid.sum())
            row[f"actual_mean__{column}"] = float(np.mean(left_values[valid]))
            row[f"control_mean__{column}"] = float(np.mean(right_values[valid]))
            row[f"delta_mean__{column}"] = float(np.mean(left_values[valid] - right_values[valid]))
            row[f"delta_median__{column}"] = float(
                np.median(left_values[valid] - right_values[valid])
            )
        rows.append(row)
    return rows


def nearest_state_pairs(
    actual: DataFrame,
    control: DataFrame,
    *,
    state_columns: Sequence[str],
    pre_distance_atr_caliper: float = PRE_DISTANCE_ATR_CALIPER,
    minimum_event_separation_hours: int = 48,
) -> tuple[list[tuple[int, int, float]], dict[str, int]]:
    empty_audit = {
        "eligible_actual": 0,
        "eligible_control": 0,
        "geometry_eligible_actual": 0,
        "state_matchable_actual": 0,
    }
    if actual.empty or control.empty:
        return [], empty_audit
    required_geometry = {"pre_distance_atr", "event_time"}
    for label, frame in (("actual", actual), ("control", control)):
        missing = sorted(required_geometry.difference(frame.columns))
        if missing:
            raise ValueError(f"{label} events lack required matching geometry: {missing}")
    if pre_distance_atr_caliper <= 0.0:
        raise ValueError("pre_distance_atr_caliper must be positive")
    if minimum_event_separation_hours < 0:
        raise ValueError("minimum_event_separation_hours cannot be negative")
    left = actual[list(state_columns)].apply(pd.to_numeric, errors="coerce")
    right = control[list(state_columns)].apply(pd.to_numeric, errors="coerce")
    left_pre_distance = numeric_array(actual["pre_distance_atr"])
    right_pre_distance = numeric_array(control["pre_distance_atr"])
    left_times = pd.to_datetime(actual["event_time"], utc=True, errors="coerce")
    right_times = pd.to_datetime(control["event_time"], utc=True, errors="coerce")
    left_valid = (
        np.isfinite(left.to_numpy(dtype=float)).all(axis=1)
        & np.isfinite(left_pre_distance)
        & left_times.notna().to_numpy()
    )
    right_valid = (
        np.isfinite(right.to_numpy(dtype=float)).all(axis=1)
        & np.isfinite(right_pre_distance)
        & right_times.notna().to_numpy()
    )
    left_positions = np.flatnonzero(left_valid)
    right_positions = np.flatnonzero(right_valid)
    audit = {
        "eligible_actual": len(left_positions),
        "eligible_control": len(right_positions),
        "geometry_eligible_actual": 0,
        "state_matchable_actual": 0,
    }
    if not len(left_positions) or not len(right_positions):
        return [], audit
    left_values = left.iloc[left_positions].to_numpy(dtype=float)
    right_values = right.iloc[right_positions].to_numpy(dtype=float)
    pooled = np.vstack([left_values, right_values])
    centre = np.nanmedian(pooled, axis=0)
    q25 = np.nanquantile(pooled, 0.25, axis=0)
    q75 = np.nanquantile(pooled, 0.75, axis=0)
    scale = q75 - q25
    fallback = np.nanstd(pooled, axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, fallback)
    usable = np.isfinite(scale) & (scale > 1e-12)
    if usable.sum() < 4:
        return [], audit
    left_scaled = (left_values[:, usable] - centre[usable]) / scale[usable]
    right_scaled = (right_values[:, usable] - centre[usable]) / scale[usable]
    left_pre = left_pre_distance[left_positions]
    right_pre = right_pre_distance[right_positions]
    left_ns = left_times.to_numpy(dtype="datetime64[ns]").astype("int64")[left_positions]
    right_ns = right_times.to_numpy(dtype="datetime64[ns]").astype("int64")[right_positions]
    required_gap_ns = int(pd.Timedelta(hours=minimum_event_separation_hours).value)
    right_pre_order = np.argsort(right_pre, kind="stable")
    sorted_right_pre = right_pre[right_pre_order]
    candidate_lists: list[tuple[int, np.ndarray, np.ndarray]] = []
    geometry_eligible = 0
    state_matchable = 0
    for left_position, distance_atr in enumerate(left_pre):
        lower = np.searchsorted(
            sorted_right_pre,
            distance_atr - pre_distance_atr_caliper,
            side="left",
        )
        upper = np.searchsorted(
            sorted_right_pre,
            distance_atr + pre_distance_atr_caliper,
            side="right",
        )
        candidates = right_pre_order[lower:upper]
        if len(candidates):
            separated = np.abs(right_ns[candidates] - left_ns[left_position]) > required_gap_ns
            candidates = candidates[separated]
        if not len(candidates):
            continue
        geometry_eligible += 1
        state_distances = np.sqrt(
            np.mean(
                np.square(right_scaled[candidates] - left_scaled[left_position]),
                axis=1,
            )
        )
        within_state = np.isfinite(state_distances) & (state_distances <= MATCH_RMS_CALIPER)
        candidates = candidates[within_state]
        state_distances = state_distances[within_state]
        if not len(candidates):
            continue
        state_matchable += 1
        rank = np.argsort(state_distances, kind="stable")
        candidate_lists.append((left_position, candidates[rank], state_distances[rank]))
    audit["geometry_eligible_actual"] = geometry_eligible
    audit["state_matchable_actual"] = state_matchable
    order = np.argsort(
        np.asarray([distances[0] for _, _, distances in candidate_lists]),
        kind="stable",
    )
    reuse = np.zeros(len(right_scaled), dtype=np.int16)
    selected: list[tuple[int, int, float]] = []
    for candidate_position in order:
        left_position, candidates, distances = candidate_lists[int(candidate_position)]
        for right_position, distance in zip(candidates, distances, strict=False):
            right_position = int(right_position)
            if reuse[right_position] >= MAX_CONTROL_REUSE:
                continue
            reuse[right_position] += 1
            selected.append(
                (
                    int(left_positions[left_position]),
                    int(right_positions[right_position]),
                    float(distance),
                )
            )
            break
    return selected, audit


def balance_diagnostics(
    actual: DataFrame, control: DataFrame, state_columns: Sequence[str]
) -> dict[str, Any]:
    values: list[float] = []
    for column in state_columns:
        left = numeric_array(actual[column])
        right = numeric_array(control[column])
        valid = np.isfinite(left) & np.isfinite(right)
        if valid.sum() < 3:
            continue
        pooled = np.sqrt((np.var(left[valid]) + np.var(right[valid])) / 2.0)
        if not np.isfinite(pooled) or pooled <= 1e-12:
            continue
        values.append(float((np.mean(left[valid]) - np.mean(right[valid])) / pooled))
    absolute = np.abs(values)
    return {
        "features_scored": len(values),
        "max_absolute_smd": float(np.max(absolute)) if len(absolute) else np.nan,
        "median_absolute_smd": float(np.median(absolute)) if len(absolute) else np.nan,
    }


def purge_overlapping_event_pairs(
    pairs: DataFrame,
    *,
    separation_hours: int,
    group_columns: Sequence[str],
    left_time_column: str,
    right_time_column: str,
) -> DataFrame:
    """Greedily retain the closest matches whose two future paths do not overlap."""
    if pairs.empty:
        return pairs.copy()
    required_gap = pd.Timedelta(hours=separation_hours)
    selected_frames = []
    for _, group in pairs.groupby(list(group_columns), observed=True, sort=False):
        candidates = group.sort_values(
            [
                "match_distance",
                "pre_distance_atr_abs_difference",
                left_time_column,
                right_time_column,
                "level_name",
                "approach_state",
            ],
            kind="stable",
        ).drop_duplicates(
            [left_time_column, right_time_column],
            keep="first",
        )
        retained_indexes: list[Any] = []
        used_times: list[pd.Timestamp] = []
        for index, row in candidates.iterrows():
            left_time = pd.Timestamp(row[left_time_column])
            right_time = pd.Timestamp(row[right_time_column])
            if any(
                abs(left_time - used) <= required_gap or abs(right_time - used) <= required_gap
                for used in used_times
            ):
                continue
            retained_indexes.append(index)
            used_times.extend((left_time, right_time))
        if retained_indexes:
            retained = candidates.loc[retained_indexes].copy()
            retained["independent_selection_order"] = np.arange(
                1, len(retained) + 1, dtype=np.int32
            )
            retained["independence_hours"] = separation_hours
            selected_frames.append(retained)
    if not selected_frames:
        return DataFrame(
            columns=[*pairs.columns, "independent_selection_order", "independence_hours"]
        )
    return pd.concat(selected_frames, ignore_index=True)


def purge_overlapping_event_pairs_by_key(
    pairs: DataFrame,
    *,
    separation_hours_by_key: Mapping[str, int],
    key_column: str,
    group_columns: Sequence[str],
    left_time_column: str,
    right_time_column: str,
) -> DataFrame:
    """Apply the response-horizon overlap purge separately to each routed question."""
    if pairs.empty:
        return purge_overlapping_event_pairs(
            pairs,
            separation_hours=0,
            group_columns=group_columns,
            left_time_column=left_time_column,
            right_time_column=right_time_column,
        )
    observed = set(pairs[key_column].dropna().astype(str))
    missing = observed.difference(separation_hours_by_key)
    if missing:
        raise ValueError(
            f"Missing independence horizon for {key_column} values: {sorted(missing)}"
        )
    selected = []
    for key, hours in separation_hours_by_key.items():
        subset = pairs.loc[pairs[key_column].astype(str).eq(key)]
        if subset.empty:
            continue
        selected.append(
            purge_overlapping_event_pairs(
                subset,
                separation_hours=int(hours),
                group_columns=group_columns,
                left_time_column=left_time_column,
                right_time_column=right_time_column,
            )
        )
    if not selected:
        return DataFrame(
            columns=[*pairs.columns, "independent_selection_order", "independence_hours"]
        )
    return pd.concat(selected, ignore_index=True)


def outcome_columns(horizons: Sequence[int]) -> list[str]:
    columns: list[str] = [
        "contact_range_ratio",
        "contact_volume_ratio",
        "contact_pressure_change_abs",
    ]
    for horizon in horizons:
        columns.extend(
            (
                f"abs_excursion_atr_h{horizon}",
                f"close_abs_displacement_atr_h{horizon}",
                f"range_ratio_h{horizon}",
                f"volume_ratio_h{horizon}",
                f"pressure_change_abs_h{horizon}",
                f"dwell_fraction_h{horizon}",
                f"crossings_h{horizon}",
            )
        )
    return columns


def outcome_values(frame: DataFrame, column: str) -> np.ndarray:
    if column == "contact_pressure_change_abs":
        return np.abs(numeric_array(frame["contact_pressure_change"]))
    if column.startswith("pressure_change_abs_h"):
        horizon = column.removeprefix("pressure_change_abs_h")
        return np.abs(numeric_array(frame[f"pressure_change_h{horizon}"]))
    return numeric_array(frame[column])


def max_reuse(indexes: Sequence[int]) -> int:
    if not indexes:
        return 0
    return int(pd.Series(indexes).value_counts().max())


def level_role(spec: LevelSpec) -> str:
    if spec.name in ACTIVITY_LEVELS:
        return "activity_transit"
    if spec.name in ACCEPTANCE_LEVELS:
        return "acceptance_stickiness"
    return "unclassified"


def combine_pair_outputs(run_id: str, pairs: Sequence[str], *, request_sha256: str) -> DataFrame:
    root = ARTIFACT_ROOT / run_id / "pair_cells"
    frames = []
    for pair in pairs:
        path = root / f"{pair_stem(pair)}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        validate_pair_output_contract(
            path,
            pair=pair,
            request_sha256=request_sha256,
        )
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True) if frames else DataFrame()


def repeatability_screen(detail: DataFrame, manifest: dict[str, Any]) -> DataFrame:
    if detail.empty:
        return DataFrame()
    cohorts = {
        pair: cohort
        for cohort, pairs in manifest["reporting_groups"]["coin_cohorts"].items()
        for pair in pairs
    }
    scopes = []
    primary = detail.copy()
    primary["market_scope"] = "frozen_cohort:" + primary["pair"].map(cohorts)
    scopes.append(primary)
    broad = detail.copy()
    broad["market_scope"] = "all_top10"
    scopes.append(broad)
    non_btc = detail.loc[~detail["pair"].eq("BTC/USDT:USDT")].copy()
    non_btc["market_scope"] = "all_non_btc"
    scopes.append(non_btc)
    working = pd.concat(scopes, ignore_index=True)
    working = working.loc[working["matched_events"].ge(MIN_CELL_EVENTS)].copy()
    working["balance_quality"] = np.select(
        [
            working["max_absolute_state_smd"].le(0.25)
            & working["median_absolute_state_smd"].le(0.10),
            working["max_absolute_state_smd"].le(0.50)
            & working["median_absolute_state_smd"].le(0.20),
        ],
        ["strong", "usable"],
        default="weak_or_sparse",
    )
    delta_columns = [column for column in working if column.startswith("delta_mean__")]
    identifiers = [
        "pair",
        "period",
        "market_scope",
        "source_timeframe",
        "level_family",
        "level_name",
        "zone_method",
        "level_role",
        "actual_scope",
        "control",
        "control_scope",
        "control_geometry",
        "matched_events",
        "match_coverage",
        "max_absolute_state_smd",
        "balance_quality",
    ]
    long = working.melt(
        id_vars=identifiers,
        value_vars=delta_columns,
        var_name="outcome",
        value_name="delta_mean",
    )
    long["outcome"] = long["outcome"].str.removeprefix("delta_mean__")
    long[["metric_family", "horizon_hours"]] = long["outcome"].apply(
        lambda value: pd.Series(outcome_metadata(value))
    )
    long["role_oriented_delta"] = [
        role_oriented_delta(role, metric, value)
        for role, metric, value in zip(
            long["level_role"], long["metric_family"], long["delta_mean"], strict=False
        )
    ]
    long["role_oriented_positive"] = long["role_oriented_delta"].gt(0.0).astype(float)
    keys = [
        "market_scope",
        "source_timeframe",
        "level_family",
        "level_name",
        "zone_method",
        "level_role",
        "actual_scope",
        "control",
        "control_scope",
        "control_geometry",
        "balance_quality",
        "metric_family",
        "horizon_hours",
    ]
    grouped = long.groupby(keys, observed=True, dropna=False, sort=False)
    return grouped.agg(
        pair_period_cells=("pair", "size"),
        pair_count=("pair", "nunique"),
        period_count=("period", "nunique"),
        matched_events=("matched_events", "sum"),
        match_coverage_median=("match_coverage", "median"),
        max_absolute_state_smd_max=("max_absolute_state_smd", "max"),
        delta_mean_median=("delta_mean", "median"),
        role_oriented_delta_median=("role_oriented_delta", "median"),
        role_oriented_positive_fraction=("role_oriented_positive", "mean"),
    ).reset_index()


def outcome_metadata(column: str) -> tuple[str, int]:
    contact = {
        "contact_range_ratio": "contact_range_activity",
        "contact_volume_ratio": "contact_volume_activity",
        "contact_pressure_change_abs": "contact_pressure_change_magnitude",
    }
    if column in contact:
        return contact[column], 0
    for prefix, family in (
        ("abs_excursion_atr_h", "absolute_excursion"),
        ("close_abs_displacement_atr_h", "close_displacement"),
        ("range_ratio_h", "range_activity"),
        ("volume_ratio_h", "volume_activity"),
        ("pressure_change_abs_h", "pressure_change_magnitude"),
        ("dwell_fraction_h", "dwell"),
        ("crossings_h", "crossings"),
    ):
        if column.startswith(prefix):
            return family, int(column.removeprefix(prefix))
    raise ValueError(f"Unknown outcome column: {column}")


def role_oriented_delta(role: str, metric: str, value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    if role == "activity_transit":
        orientation = -1.0 if metric == "dwell" else 1.0
    elif role == "acceptance_stickiness":
        orientation = 1.0 if metric in {"dwell", "crossings"} else -1.0
    else:
        return np.nan
    return float(value * orientation)


def integrity_record(detail: DataFrame, *, technical_smoke: bool) -> dict[str, Any]:
    clean_any = detail.loc[detail["control_scope"].eq("no_real_level_overlap")]
    return {
        "created_at_utc": utc_now(),
        "technical_smoke_not_evidence": technical_smoke,
        "matched_cells": len(detail),
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "no_real_level_overlap_cells": len(clean_any),
        "actual_scopes": sorted(detail["actual_scope"].unique().tolist()) if len(detail) else [],
        "control_geometries": sorted(detail["control_geometry"].unique().tolist())
        if len(detail)
        else [],
        "pairs": int(detail["pair"].nunique()) if len(detail) else 0,
        "timeframes": sorted(detail["source_timeframe"].unique().tolist()) if len(detail) else [],
        "controls": sorted(detail["control"].unique().tolist()) if len(detail) else [],
        "median_match_coverage": float(detail["match_coverage"].median()) if len(detail) else 0.0,
        "cells_with_max_absolute_smd_le_0_25_fraction": float(
            detail["max_absolute_state_smd"].le(0.25).mean()
        )
        if len(detail)
        else 0.0,
        "no_real_level_scope_overlap_any_real_fraction_max": float(
            clean_any["control_overlap_any_real_fraction"].max()
        )
        if len(clean_any)
        else np.nan,
        "pre_distance_atr_abs_difference_max": float(
            detail["pre_distance_atr_abs_difference_max"].max()
        )
        if len(detail)
        else np.nan,
        "event_separation_hours_min": float(detail["event_separation_hours_min"].min())
        if len(detail)
        else np.nan,
        "contact_candle_state_used_for_matching": False,
        "contact_candle_reported_as_outcome": True,
        "future_period_quantile_bands_used_for_matching": False,
        "raw_control_event_rows_retained": False,
        "direction_prediction": False,
        "profit_optimization": False,
    }


def pair_stem(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")


if __name__ == "__main__":
    raise SystemExit(main())

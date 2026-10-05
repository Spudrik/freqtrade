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
    "VECLIB_MAXIMUM_THREADS",
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

from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    DEFAULT_MANIFEST,
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
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_cluster_increment import (  # noqa: E501
    ARTIFACT_ROOT as G1C_PREFLIGHT_ARTIFACT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_cluster_increment import (  # noqa: E501
    REPORT_ROOT as G1C_PREFLIGHT_REPORT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_localization import (  # noqa: E501
    balance_diagnostics,
    causal_local_state,
    causal_market_context,
    nearest_state_pairs,
    outcome_columns,
    outcome_metadata,
    outcome_values,
)


REPORT_ROOT = OUTPUT_ROOT / "generation1_cluster_outcomes"
ARTIFACT_ROOT = LARGE_ARTIFACT_ROOT / "generation1_cluster_outcomes"
FROZEN_BATCH = OUTPUT_ROOT / "generation0_review" / "g1_frozen_branch_batch.json"
DEFAULT_SOURCE_RUN_ID = "g1c_contact_preflight_full_20260813b_dependency"
ALLOWED_SCALES = ("tight", "standard")
ALLOWED_REPRESENTATIONS = ("projected", "held_sensitivity")
OUTPUT_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1
MIN_CELL_EVENTS = 20

G1C_STATE_FEATURES = (
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
    "state_cluster_density_2atr",
    "state_cluster_width_atr",
    "state_active_level_count",
)


@dataclass(frozen=True)
class PairTask:
    pair: str
    manifest_path: str
    source_run_id: str
    run_id: str
    scales: tuple[str, ...]
    representations: tuple[str, ...]
    max_events: int | None
    overwrite: bool
    request_sha256: str


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 1C separate-episode anchor outcome comparison. Independent "
            "cross-timeframe cluster contacts are compared with causally matched episodes "
            "where each contacted component appeared alone. Direction and profit are out "
            "of scope."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--scales", default=",".join(ALLOWED_SCALES))
    parser.add_argument(
        "--representations", default=",".join(ALLOWED_REPRESENTATIONS)
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    validate_worker_count(args.workers)
    if args.max_events is not None and args.max_events < 200:
        raise ValueError("--max-events must be at least 200.")
    manifest = load_manifest(args.manifest)
    pairs = select_values(args.pairs, tuple(manifest["data"]["pairs"]), "pair")
    scales = select_values(args.scales, ALLOWED_SCALES, "cluster scale")
    representations = select_values(
        args.representations,
        ALLOWED_REPRESENTATIONS,
        "representation mode",
    )
    validate_frozen_branch()
    source_record = validate_source_run(
        source_run_id=args.source_run_id,
        pairs=pairs,
    )
    request = request_contract(
        manifest_path=args.manifest,
        source_run_id=args.source_run_id,
        source_record=source_record,
        pairs=pairs,
        scales=scales,
        representations=representations,
        max_events=args.max_events,
    )
    request_sha256 = stable_json_sha256(request)

    compact_dir = REPORT_ROOT / args.run_id
    bulky_dir = ARTIFACT_ROOT / args.run_id
    compact_dir.mkdir(parents=True, exist_ok=True)
    bulky_dir.mkdir(parents=True, exist_ok=True)
    record_path = compact_dir / "g1c_outcome_run_record.json"
    if record_path.is_file() and not args.overwrite:
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        if existing.get("request_sha256") != request_sha256:
            raise ValueError(
                "Run ID already exists with incompatible code, inputs, or settings."
            )
    record = {
        "schema_version": RUN_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "status": "running",
        "started_at_utc": utc_now(),
        "orchestrator_pid": os.getpid(),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "frozen_branch": "g1c_same_episode_independent_cluster_increment",
        "source_run_id": args.source_run_id,
        "request_contract": request,
        "request_sha256": request_sha256,
        "pairs": list(pairs),
        "scales": list(scales),
        "representations": list(representations),
        "technical_smoke_not_evidence": args.max_events is not None,
        "hypothesis": (
            "A directly contacted cross-timeframe cluster containing at least two "
            "conservatively independent dependency groups changes non-directional market "
            "behaviour beyond separate episodes where each component was contacted alone."
        ),
        "matching": {
            "exact_strata": [
                "pair",
                "representation_mode",
                "cluster_scale",
                "anchor_component_key",
                "period",
                "approach_state",
            ],
            "continuous_state": list(G1C_STATE_FEATURES),
            "anchor_pre_distance_atr_caliper": 0.10,
            "minimum_event_separation_hours": max(
                manifest["reaction_definition"]["horizons_hours"]
            ),
            "maximum_control_reuse": 3,
        },
        "canonical_episode_rule": (
            "Inside each pair, representation, scale, and future market path, retain the "
            "eligible cluster with the fewest contacted components, then the narrowest "
            "cluster, then the stable episode key. This prevents nested copies from "
            "becoming independent outcomes."
        ),
        "all_anchor_rule": (
            "An episode is fully comparable only when every directly contacted component "
            "has a separate matched anchor-only episode. For each outcome the minimum "
            "multi-minus-anchor delta asks whether the cluster exceeded every anchor."
        ),
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(record, record_path)
    tasks = [
        PairTask(
            pair=pair,
            manifest_path=str(args.manifest.resolve()),
            source_run_id=args.source_run_id,
            run_id=args.run_id,
            scales=scales,
            representations=representations,
            max_events=args.max_events,
            overwrite=args.overwrite,
            request_sha256=request_sha256,
        )
        for pair in pairs
    ]
    try:
        results = run_pair_tasks(tasks, workers=args.workers)
        inventory = DataFrame(results)
        atomic_write_parquet(inventory, compact_dir / "g1c_outcome_pair_inventory.parquet")
        failures = inventory.loc[inventory["status"].eq("failed")]
        if not failures.empty:
            raise RuntimeError(
                f"{len(failures)} pair task(s) failed; inspect the pair inventory."
            )
        matches = combine_pair_outputs(
            args.run_id,
            pairs,
            "matches",
            request_sha256=request_sha256,
        )
        cells = combine_pair_outputs(
            args.run_id,
            pairs,
            "cells",
            request_sha256=request_sha256,
        )
        strict = combine_pair_outputs(
            args.run_id,
            pairs,
            "strict_episodes",
            request_sha256=request_sha256,
        )
        screen = cluster_repeatability_screen(strict, manifest)
        pattern_screen = recurring_pattern_screen(strict, manifest)
        atomic_write_parquet(
            cells,
            compact_dir / "g1c_anchor_cell_summary.parquet",
        )
        atomic_write_parquet(
            screen,
            compact_dir / "g1c_cluster_repeatability_screen.parquet",
        )
        atomic_write_parquet(
            pattern_screen,
            compact_dir / "g1c_recurring_pattern_screen.parquet",
        )
        integrity = integrity_record(
            results=results,
            matches=matches,
            cells=cells,
            strict=strict,
            technical_smoke=args.max_events is not None,
        )
        integrity_path = compact_dir / "g1c_outcome_integrity.json"
        atomic_write_json(integrity, integrity_path)
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "pair_tasks": results,
                "matched_anchor_rows": len(matches),
                "anchor_cell_rows": len(cells),
                "strict_episode_rows": len(strict),
                "cluster_repeatability_rows": len(screen),
                "recurring_pattern_rows": len(pattern_screen),
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


def select_values(
    requested: str, allowed: tuple[str, ...], label: str
) -> tuple[str, ...]:
    if requested.strip().lower() == "all":
        return allowed
    selected = tuple(value.strip() for value in requested.split(",") if value.strip())
    unknown = sorted(set(selected).difference(allowed))
    if not selected or unknown:
        raise ValueError(f"Invalid {label} selection: selected={selected}, unknown={unknown}")
    return selected


def validate_frozen_branch() -> None:
    frozen = json.loads(FROZEN_BATCH.read_text(encoding="utf-8"))
    branches = {branch["id"]: branch for branch in frozen["branches"]}
    if (
        frozen.get("status") != "frozen_ready_for_execution"
        or "g1c_same_episode_independent_cluster_increment" not in branches
    ):
        raise ValueError("The frozen Generation 1C branch is unavailable.")
    if frozen["common_scope"].get("direction_prediction") is not False:
        raise ValueError("Generation 1C must keep direction prediction disabled.")
    if frozen["common_scope"].get("profit_optimization") is not False:
        raise ValueError("Generation 1C must keep profit optimization disabled.")


def validate_source_run(
    *, source_run_id: str, pairs: Sequence[str]
) -> dict[str, Any]:
    path = G1C_PREFLIGHT_REPORT_ROOT / source_run_id / "g1c_run_record.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "completed":
        raise ValueError("The G1C contact preflight source run is incomplete.")
    if not set(pairs).issubset(record.get("pairs", [])):
        raise ValueError("The G1C contact preflight lacks a requested pair.")
    if record.get("direction_prediction") is not False:
        raise ValueError("The G1C contact preflight crossed the direction boundary.")
    return record


def preflight_pair_paths(source_run_id: str, pair: str) -> tuple[Path, Path]:
    root = G1C_PREFLIGHT_ARTIFACT_ROOT / source_run_id / "pair_contacts"
    stem = pair_stem(pair)
    return root / f"{stem}.parquet", root / f"{stem}.anchors.parquet"


def request_contract(
    *,
    manifest_path: Path,
    source_run_id: str,
    source_record: dict[str, Any],
    pairs: Sequence[str],
    scales: Sequence[str],
    representations: Sequence[str],
    max_events: int | None,
) -> dict[str, Any]:
    inputs = []
    for pair in pairs:
        for path in preflight_pair_paths(source_run_id, pair):
            if not path.is_file():
                raise FileNotFoundError(path)
            stat = path.stat()
            inputs.append(
                {
                    "path": str(path.resolve()),
                    "bytes": int(stat.st_size),
                    "modified_ns": int(stat.st_mtime_ns),
                    "sha256": sha256_file(path),
                }
            )
    return {
        "run_schema_version": RUN_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "frozen_batch_sha256": sha256_file(FROZEN_BATCH),
        "source_run_id": source_run_id,
        "source_request_sha256": source_record.get("request_sha256"),
        "source_inputs": inputs,
        "pairs": list(pairs),
        "scales": list(scales),
        "representations": list(representations),
        "max_events": max_events,
        "state_features": list(G1C_STATE_FEATURES),
        "direction_prediction": False,
        "profit_optimization": False,
    }


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_pair_tasks(
    tasks: Sequence[PairTask], *, workers: int
) -> list[dict[str, Any]]:
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


def pair_output_path(run_id: str, pair: str, kind: str) -> Path:
    return ARTIFACT_ROOT / run_id / f"pair_{kind}" / f"{pair_stem(pair)}.parquet"


def build_pair(task: PairTask) -> dict[str, Any]:  # noqa: C901
    started = time.perf_counter()
    output_paths = {
        kind: pair_output_path(task.run_id, task.pair, kind)
        for kind in ("matches", "cells", "strict_episodes")
    }
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    if all(path.is_file() for path in output_paths.values()) and not task.overwrite:
        for path in output_paths.values():
            validate_pair_output(
                path,
                pair=task.pair,
                request_sha256=task.request_sha256,
            )
        return {
            "pair": task.pair,
            "status": "existing",
            "matched_anchor_rows": parquet_row_count(output_paths["matches"]),
            "seconds": round(time.perf_counter() - started, 3),
        }

    manifest = load_manifest(Path(task.manifest_path))
    horizons = tuple(int(value) for value in manifest["reaction_definition"]["horizons_hours"])
    contact_path, anchor_path = preflight_pair_paths(task.source_run_id, task.pair)
    anchors = pd.read_parquet(anchor_path)
    contacts = pd.read_parquet(contact_path)
    anchors = anchors.loc[
        anchors["cluster_scale"].isin(task.scales)
        & anchors["representation_mode"].isin(task.representations)
    ].copy()
    contacts = contacts.loc[
        contacts["cluster_scale"].isin(task.scales)
        & contacts["representation_mode"].isin(task.representations)
    ].copy()
    outcome_source_columns = raw_outcome_columns(horizons)
    contact_metadata = [
        "contacted_component_keys",
        "contacted_component_dependency_groups",
        "contacted_component_timeframes",
        "contacted_interpretation_class",
    ]
    contact_outcomes = contacts[
        ["cluster_contact_episode_key", *contact_metadata, *outcome_source_columns]
    ].copy()
    anchors = anchors.merge(
        contact_outcomes,
        on="cluster_contact_episode_key",
        how="left",
        validate="many_to_one",
    )
    base = prepare_base_market_frame(task.pair, manifest)
    state = causal_local_state(base)
    state = state.merge(
        causal_market_context(manifest),
        on="date",
        how="left",
        validate="one_to_one",
    )
    indexes = anchors["base_index"].to_numpy(dtype=np.int64)
    for column in state:
        if column.startswith("state_"):
            anchors[column] = state[column].to_numpy(dtype=float)[indexes]
    anchors["state_cluster_density_2atr"] = anchors["local_level_density_2atr"]
    anchors["state_cluster_width_atr"] = anchors["cluster_width_atr"]
    anchors["state_active_level_count"] = anchors["active_level_count"]
    anchors["pre_distance_atr"] = anchors["anchor_pre_distance_atr"]

    multi = anchors.loc[
        anchors["anchor_contact_kind"].eq("independent_multi_component_contact")
        & anchors["contacted_independent_dependency_groups"].eq(True)
        & anchors["contact_structure_class"].eq(
            "different_family_cross_timeframe"
        )
    ].copy()
    single = anchors.loc[
        anchors["anchor_contact_kind"].eq("single_component_contact")
    ].copy()
    multi = canonical_multi_anchors(multi)
    single = canonical_single_anchors(single)
    if task.max_events is not None:
        multi = bounded_sample(multi, task.max_events)
        single = bounded_sample(single, task.max_events)
    match_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    group_keys = [
        "representation_mode",
        "cluster_scale",
        "anchor_component_key",
        "period",
        "approach_state",
    ]
    single_groups = {
        key: frame for key, frame in single.groupby(group_keys, observed=True, sort=False)
    }
    for key, actual in multi.groupby(group_keys, observed=True, sort=False):
        control = single_groups.get(key)
        if control is None or control.empty:
            continue
        pairs, audit = nearest_state_pairs(
            actual,
            control,
            state_columns=G1C_STATE_FEATURES,
            pre_distance_atr_caliper=0.10,
            minimum_event_separation_hours=max(horizons),
        )
        if not pairs:
            continue
        left = actual.iloc[[pair[0] for pair in pairs]].reset_index(drop=True)
        right = control.iloc[[pair[1] for pair in pairs]].reset_index(drop=True)
        cell_rows.append(
            anchor_cell_row(
                left,
                right,
                group_key=key,
                audit=audit,
                distances=[pair[2] for pair in pairs],
                horizons=horizons,
            )
        )
        for position, distance in enumerate(pair[2] for pair in pairs):
            match_rows.append(
                anchor_match_row(
                    left.iloc[position],
                    right.iloc[position],
                    match_distance=float(distance),
                    horizons=horizons,
                )
            )
    if not match_rows:
        raise ValueError(f"No separate anchor-only matches were produced for {task.pair}.")
    matches = DataFrame(match_rows)
    cells = DataFrame(cell_rows)
    strict = strict_episode_rows(matches, horizons=horizons)
    for frame in (matches, cells, strict):
        frame["output_schema_version"] = OUTPUT_SCHEMA_VERSION
        frame["run_request_sha256"] = task.request_sha256
    atomic_write_parquet(matches, output_paths["matches"])
    atomic_write_parquet(cells, output_paths["cells"])
    atomic_write_parquet(strict, output_paths["strict_episodes"])
    return {
        "pair": task.pair,
        "status": "completed",
        "multi_anchor_candidates": len(multi),
        "single_anchor_candidates": len(single),
        "matched_anchor_rows": len(matches),
        "anchor_cells": len(cells),
        "strict_episode_rows": len(strict),
        "fully_matched_cluster_episodes": int(strict["all_anchors_matched"].sum()),
        "seconds": round(time.perf_counter() - started, 3),
    }


def raw_outcome_columns(horizons: Sequence[int]) -> list[str]:
    columns = ["contact_range_ratio", "contact_volume_ratio", "contact_pressure_change"]
    for horizon in horizons:
        columns.extend(
            [
                f"abs_excursion_atr_h{horizon}",
                f"close_abs_displacement_atr_h{horizon}",
                f"range_ratio_h{horizon}",
                f"volume_ratio_h{horizon}",
                f"pressure_change_h{horizon}",
                f"dwell_fraction_h{horizon}",
                f"crossings_h{horizon}",
            ]
        )
    return columns


def canonical_multi_anchors(frame: DataFrame) -> DataFrame:
    if frame.empty:
        return frame
    episode_keys = [
        "representation_mode",
        "cluster_scale",
        "future_path_key",
        "cluster_contact_episode_key",
    ]
    episodes = frame.drop_duplicates(episode_keys).sort_values(
        [
            "representation_mode",
            "cluster_scale",
            "future_path_key",
            "contacted_component_count",
            "cluster_width_atr",
            "cluster_contact_episode_key",
        ],
        kind="stable",
    )
    chosen = episodes.drop_duplicates(
        ["representation_mode", "cluster_scale", "future_path_key"],
        keep="first",
    )["cluster_contact_episode_key"]
    return frame.loc[frame["cluster_contact_episode_key"].isin(chosen)].copy()


def canonical_single_anchors(frame: DataFrame) -> DataFrame:
    if frame.empty:
        return frame
    return (
        frame.sort_values(
            [
                "representation_mode",
                "cluster_scale",
                "anchor_component_key",
                "future_path_key",
                "cluster_width_atr",
                "cluster_contact_episode_key",
            ],
            kind="stable",
        )
        .drop_duplicates(
            [
                "representation_mode",
                "cluster_scale",
                "anchor_component_key",
                "future_path_key",
            ],
            keep="first",
        )
        .copy()
    )


def bounded_sample(frame: DataFrame, maximum: int) -> DataFrame:
    if len(frame) <= maximum:
        return frame
    positions = np.linspace(0, len(frame) - 1, num=maximum, dtype=np.int64)
    return frame.iloc[np.unique(positions)].copy()


def anchor_match_row(
    actual: pd.Series,
    control: pd.Series,
    *,
    match_distance: float,
    horizons: Sequence[int],
) -> dict[str, Any]:
    separation = abs(
        (pd.Timestamp(actual["event_time"]) - pd.Timestamp(control["event_time"]))
        .total_seconds()
        / 3600.0
    )
    row: dict[str, Any] = {
        "pair": actual["pair"],
        "representation_mode": actual["representation_mode"],
        "cluster_scale": actual["cluster_scale"],
        "period": actual["period"],
        "approach_state": actual["approach_state"],
        "multi_future_path_key": actual["future_path_key"],
        "multi_cluster_episode_key": actual["cluster_contact_episode_key"],
        "multi_event_time": actual["event_time"],
        "multi_base_index": int(actual["base_index"]),
        "control_future_path_key": control["future_path_key"],
        "control_cluster_episode_key": control["cluster_contact_episode_key"],
        "control_event_time": control["event_time"],
        "control_base_index": int(control["base_index"]),
        "anchor_component_key": actual["anchor_component_key"],
        "anchor_source_key": actual["anchor_source_key"],
        "anchor_family": actual["anchor_family"],
        "anchor_mechanism": actual["anchor_mechanism"],
        "anchor_dependency_group": actual["anchor_dependency_group"],
        "anchor_timeframe": actual["anchor_timeframe"],
        "anchor_interpretation": actual["anchor_interpretation"],
        "contacted_component_signature": actual["contacted_component_signature"],
        "contacted_component_count": int(actual["contacted_component_count"]),
        "contacted_dependency_group_count": int(
            actual["contacted_dependency_group_count"]
        ),
        "contacted_component_keys": actual["contacted_component_keys"],
        "contacted_component_dependency_groups": actual[
            "contacted_component_dependency_groups"
        ],
        "contacted_component_timeframes": actual["contacted_component_timeframes"],
        "contacted_interpretation_class": actual[
            "contacted_interpretation_class"
        ],
        "match_distance": match_distance,
        "pre_distance_atr_abs_difference": abs(
            float(actual["pre_distance_atr"]) - float(control["pre_distance_atr"])
        ),
        "event_separation_hours": separation,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    actual_frame = actual.to_frame().T
    control_frame = control.to_frame().T
    for column in outcome_columns(horizons):
        left = outcome_values(actual_frame, column)[0]
        right = outcome_values(control_frame, column)[0]
        row[f"multi__{column}"] = left
        row[f"anchor_only__{column}"] = right
        row[f"delta__{column}"] = left - right
    return row


def anchor_cell_row(
    actual: DataFrame,
    control: DataFrame,
    *,
    group_key: tuple[Any, ...],
    audit: dict[str, int],
    distances: Sequence[float],
    horizons: Sequence[int],
) -> dict[str, Any]:
    representation, scale, anchor, period, approach = group_key
    balance = balance_diagnostics(actual, control, G1C_STATE_FEATURES)
    pre_difference = np.abs(
        actual["pre_distance_atr"].to_numpy(dtype=float)
        - control["pre_distance_atr"].to_numpy(dtype=float)
    )
    row: dict[str, Any] = {
        "pair": actual["pair"].iloc[0],
        "representation_mode": representation,
        "cluster_scale": scale,
        "anchor_component_key": anchor,
        "anchor_family": actual["anchor_family"].iloc[0],
        "anchor_dependency_group": actual["anchor_dependency_group"].iloc[0],
        "anchor_timeframe": actual["anchor_timeframe"].iloc[0],
        "anchor_interpretation": actual["anchor_interpretation"].iloc[0],
        "period": period,
        "approach_state": approach,
        "eligible_multi_events": audit["eligible_actual"],
        "eligible_single_events": audit["eligible_control"],
        "geometry_eligible_multi_events": audit["geometry_eligible_actual"],
        "state_matchable_multi_events": audit["state_matchable_actual"],
        "matched_events": len(actual),
        "match_distance_median": float(np.median(distances)),
        "match_distance_q90": float(np.quantile(distances, 0.90)),
        "max_absolute_state_smd": balance["max_absolute_smd"],
        "median_absolute_state_smd": balance["median_absolute_smd"],
        "state_features_scored": balance["features_scored"],
        "pre_distance_atr_abs_difference_median": float(
            np.median(pre_difference)
        ),
        "pre_distance_atr_abs_difference_max": float(np.max(pre_difference)),
        "direction_prediction": False,
        "profit_optimization": False,
    }
    for column in outcome_columns(horizons):
        left = outcome_values(actual, column)
        right = outcome_values(control, column)
        valid = np.isfinite(left) & np.isfinite(right)
        if not valid.any():
            continue
        row[f"scored__{column}"] = int(valid.sum())
        row[f"multi_mean__{column}"] = float(np.mean(left[valid]))
        row[f"anchor_only_mean__{column}"] = float(np.mean(right[valid]))
        row[f"delta_mean__{column}"] = float(np.mean(left[valid] - right[valid]))
        row[f"delta_median__{column}"] = float(
            np.median(left[valid] - right[valid])
        )
    return row


def strict_episode_rows(matches: DataFrame, *, horizons: Sequence[int]) -> DataFrame:
    keys = [
        "pair",
        "representation_mode",
        "cluster_scale",
        "period",
        "multi_future_path_key",
        "multi_cluster_episode_key",
        "multi_event_time",
        "multi_base_index",
        "contacted_component_signature",
        "contacted_component_count",
        "contacted_dependency_group_count",
        "contacted_component_keys",
        "contacted_component_dependency_groups",
        "contacted_component_timeframes",
        "contacted_interpretation_class",
    ]
    grouped = matches.groupby(keys, observed=True, sort=False)
    base = grouped.agg(
        matched_anchor_count=("anchor_component_key", "nunique"),
        maximum_anchor_match_distance=("match_distance", "max"),
        maximum_pre_distance_atr_difference=(
            "pre_distance_atr_abs_difference",
            "max",
        ),
        minimum_event_separation_hours=("event_separation_hours", "min"),
    )
    outcome_statistics: dict[str, pd.Series] = {}
    for column in outcome_columns(horizons):
        delta = f"delta__{column}"
        outcome_statistics[f"minimum_delta_across_anchors__{column}"] = grouped[
            delta
        ].min()
        outcome_statistics[f"median_delta_across_anchors__{column}"] = grouped[
            delta
        ].median()
        outcome_statistics[f"maximum_delta_across_anchors__{column}"] = grouped[
            delta
        ].max()
    result = base.join(DataFrame(outcome_statistics)).reset_index()
    return result.assign(
        all_anchors_matched=result["matched_anchor_count"].eq(
            result["contacted_component_count"]
        ),
        direction_prediction=False,
        profit_optimization=False,
    )


def cluster_repeatability_screen(strict: DataFrame, manifest: dict[str, Any]) -> DataFrame:
    return _strict_screen(strict, manifest, include_signature=False)


def recurring_pattern_screen(strict: DataFrame, manifest: dict[str, Any]) -> DataFrame:
    return _strict_screen(strict, manifest, include_signature=True)


def _strict_screen(
    strict: DataFrame,
    manifest: dict[str, Any],
    *,
    include_signature: bool,
) -> DataFrame:
    working = strict.loc[strict["all_anchors_matched"]].copy()
    if working.empty:
        return DataFrame()
    cohorts = {
        pair: cohort
        for cohort, pairs in manifest["reporting_groups"]["coin_cohorts"].items()
        for pair in pairs
    }
    scopes = []
    cohort = working.copy()
    cohort["market_scope"] = "frozen_cohort:" + cohort["pair"].map(cohorts)
    scopes.append(cohort)
    broad = working.copy()
    broad["market_scope"] = "all_top10"
    scopes.append(broad)
    non_btc = working.loc[~working["pair"].eq("BTC/USDT:USDT")].copy()
    non_btc["market_scope"] = "all_non_btc"
    scopes.append(non_btc)
    working = pd.concat(scopes, ignore_index=True)
    value_columns = [
        column
        for column in working
        if column.startswith("minimum_delta_across_anchors__")
    ]
    identifiers = [
        "pair",
        "period",
        "market_scope",
        "representation_mode",
        "cluster_scale",
        "contacted_component_count",
        "contacted_dependency_group_count",
        "contacted_component_dependency_groups",
        "contacted_component_timeframes",
        "contacted_interpretation_class",
        "multi_future_path_key",
    ]
    if include_signature:
        identifiers.extend(
            ["contacted_component_signature", "contacted_component_keys"]
        )
    long = working.melt(
        id_vars=identifiers,
        value_vars=value_columns,
        var_name="outcome",
        value_name="minimum_delta_across_anchors",
    )
    long["outcome"] = long["outcome"].str.removeprefix(
        "minimum_delta_across_anchors__"
    )
    metadata = long["outcome"].apply(outcome_metadata)
    long["metric_family"] = [value[0] for value in metadata]
    long["horizon_hours"] = [value[1] for value in metadata]
    long["greater_than_every_anchor"] = long["minimum_delta_across_anchors"].gt(0.0)
    keys = [
        "market_scope",
        "representation_mode",
        "cluster_scale",
        "contacted_component_count",
        "contacted_dependency_group_count",
        "contacted_component_dependency_groups",
        "contacted_component_timeframes",
        "contacted_interpretation_class",
        "metric_family",
        "horizon_hours",
    ]
    if include_signature:
        keys.extend(["contacted_component_signature", "contacted_component_keys"])
    return (
        long.groupby(keys, observed=True, dropna=False, sort=False)
        .agg(
            unique_future_paths=("multi_future_path_key", "nunique"),
            pair_count=("pair", "nunique"),
            period_count=("period", "nunique"),
            minimum_delta_median=("minimum_delta_across_anchors", "median"),
            greater_than_every_anchor_fraction=("greater_than_every_anchor", "mean"),
        )
        .reset_index()
    )


def combine_pair_outputs(
    run_id: str,
    pairs: Sequence[str],
    kind: str,
    *,
    request_sha256: str,
) -> DataFrame:
    frames = []
    for pair in pairs:
        path = pair_output_path(run_id, pair, kind)
        if not path.is_file():
            raise FileNotFoundError(path)
        validate_pair_output(path, pair=pair, request_sha256=request_sha256)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True) if frames else DataFrame()


def validate_pair_output(
    path: Path, *, pair: str, request_sha256: str
) -> None:
    try:
        frame = pd.read_parquet(
            path,
            columns=["pair", "output_schema_version", "run_request_sha256"],
        )
    except Exception as exc:
        raise ValueError(f"Existing G1C outcome lacks a valid contract: {path}") from exc
    if frame.empty or set(frame["pair"].astype(str)) != {pair}:
        raise ValueError(f"Existing G1C outcome has an incompatible pair: {path}")
    if set(frame["output_schema_version"].astype(int)) != {OUTPUT_SCHEMA_VERSION}:
        raise ValueError(f"Existing G1C outcome has an incompatible schema: {path}")
    if set(frame["run_request_sha256"].astype(str)) != {request_sha256}:
        raise ValueError(f"Existing G1C outcome belongs to another request: {path}")


def parquet_row_count(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def integrity_record(
    *,
    results: Sequence[dict[str, Any]],
    matches: DataFrame,
    cells: DataFrame,
    strict: DataFrame,
    technical_smoke: bool,
) -> dict[str, Any]:
    fully = strict.loc[strict["all_anchors_matched"]]
    return {
        "created_at_utc": utc_now(),
        "technical_smoke_not_evidence": technical_smoke,
        "pairs": len(results),
        "matched_anchor_rows": len(matches),
        "anchor_cell_rows": len(cells),
        "strict_episode_rows": len(strict),
        "fully_matched_cluster_episodes": len(fully),
        "fully_matched_fraction": len(fully) / len(strict) if len(strict) else 0.0,
        "unique_fully_matched_future_paths": int(
            fully["multi_future_path_key"].nunique()
        )
        if len(fully)
        else 0,
        "pre_distance_atr_abs_difference_max": float(
            matches["pre_distance_atr_abs_difference"].max()
        )
        if len(matches)
        else np.nan,
        "event_separation_hours_min": float(matches["event_separation_hours"].min())
        if len(matches)
        else np.nan,
        "cells_with_at_least_20_matches": int(cells["matched_events"].ge(20).sum()),
        "cells_at_least_20_with_max_state_smd_le_0_50": int(
            (cells["matched_events"].ge(20) & cells["max_absolute_state_smd"].le(0.50)).sum()
        ),
        "nested_cluster_copies_counted_as_independent_outcomes": False,
        "direction_prediction": False,
        "profit_optimization": False,
    }


def pair_stem(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")


if __name__ == "__main__":
    raise SystemExit(main())

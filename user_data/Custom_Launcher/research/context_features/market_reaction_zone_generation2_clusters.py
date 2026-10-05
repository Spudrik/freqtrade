from __future__ import annotations

# Fix numerical-library thread counts before importing numpy/pandas.
# ruff: noqa: E402, E501
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

from user_data.Custom_Launcher.research.context_features.market_reaction_zone_cluster_generation0 import (
    cluster_storage_roots,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    DEFAULT_MANIFEST,
    LARGE_ARTIFACT_ROOT,
    OUTPUT_ROOT,
    LevelSpec,
    atomic_write_json,
    atomic_write_parquet,
    episode_start_mask,
    event_frame,
    future_path_matrices,
    load_manifest,
    normalize_dates,
    numeric_array,
    prepare_base_market_frame,
    sha256_file,
    utc_now,
    validate_worker_count,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_localization import (
    AlignedLevel,
    aligned_selected_levels,
    balance_diagnostics,
    causal_local_state,
    causal_market_context,
    nearest_state_pairs,
    purge_overlapping_event_pairs,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_localization import (
    eligible_period_events,
)


FROZEN_BATCH = OUTPUT_ROOT / "generation1_review" / "g2_frozen_branch_batch.json"
REPORT_ROOT = OUTPUT_ROOT / "generation2_branches" / "g2c_mirrored_clusters"
ARTIFACT_ROOT = (
    LARGE_ARTIFACT_ROOT / "generation2_branches" / "g2c_mirrored_clusters"
)
DEFAULT_SOURCE_RUN_ID = "g0d_main_20260812_v1"
OUTPUT_SCHEMA_VERSION = 1
CLUSTER_HALF_WIDTH_ATR = 0.10
SHIFT_MULTIPLIERS = (-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0)
INDEPENDENCE_HOURS = 4
MIN_PAIR_PERIOD_EPISODES = 5
MIN_COHORT_PERIOD_EPISODES = 50
MIN_COHORT_PERIOD_COINS = 5
OUTCOMES = (
    "contact_range_ratio",
    "contact_volume_ratio",
    "abs_excursion_atr_h4",
    "range_ratio_h4",
    "volume_ratio_h4",
)
STATE_FEATURES = (
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
class MirroredPattern:
    id: str
    extreme_timeframe: str
    extreme_column: str
    band_timeframe: str
    band_column: str

    @property
    def extreme_key_fragment(self) -> str:
        return f"{self.extreme_timeframe}|generic_prior_range|"

    @property
    def band_key_fragment(self) -> str:
        return f"{self.band_timeframe}|generic_bollinger|"


PATTERNS = (
    MirroredPattern(
        "rolling_low_24_plus_bb20_lower",
        "1h",
        "generic_rolling_low_24",
        "4h",
        "generic_bb20_lower",
    ),
    MirroredPattern(
        "rolling_high_24_plus_bb20_upper",
        "1h",
        "generic_rolling_high_24",
        "4h",
        "generic_bb20_upper",
    ),
)


@dataclass(frozen=True)
class PairTask:
    pair: str
    manifest_path: str
    source_event_path: str
    context_path: str
    run_id: str
    request_sha256: str
    overwrite: bool


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 2C mirrored rolling-extreme plus Bollinger cluster test. "
            "It measures non-directional activity and does not optimize profit."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--technical-smoke", action="store_true")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    validate_worker_count(args.workers, manifest=manifest)
    validate_frozen_branch()
    pairs = select_pairs(manifest, args.pairs)
    source_report_root, source_artifact_root = cluster_storage_roots(manifest)
    source_record_path = (
        source_report_root / args.source_run_id / "g0d_cluster_run_record.json"
    )
    source_integrity_path = source_report_root / args.source_run_id / "g0d_integrity.json"
    validate_source_atlas(
        record_path=source_record_path,
        integrity_path=source_integrity_path,
        technical_smoke=args.technical_smoke,
    )
    source_event_paths = {
        pair: source_artifact_root
        / args.source_run_id
        / "pair_events"
        / f"{pair_stem(pair)}.parquet"
        for pair in pairs
    }
    missing = [str(path) for path in source_event_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source cluster event files: {missing}")

    compact_dir = REPORT_ROOT / args.run_id
    bulky_dir = ARTIFACT_ROOT / args.run_id
    pair_dir = bulky_dir / "pair_matches"
    compact_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    context_path = compact_dir / "g2c_causal_market_context.parquet"
    if args.overwrite or not context_path.is_file():
        atomic_write_parquet(causal_market_context(manifest), context_path)
    request = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "frozen_batch_sha256": sha256_file(FROZEN_BATCH),
        "source_record_sha256": sha256_file(source_record_path),
        "source_integrity_sha256": sha256_file(source_integrity_path),
        "source_run_id": args.source_run_id,
        "source_event_sha256": {
            pair: sha256_file(path) for pair, path in source_event_paths.items()
        },
        "pairs": pairs,
        "patterns": [pattern.__dict__ for pattern in PATTERNS],
        "outcomes": list(OUTCOMES),
        "state_features": list(STATE_FEATURES),
        "cluster_half_width_atr": CLUSTER_HALF_WIDTH_ATR,
        "symmetric_shift_multipliers": list(SHIFT_MULTIPLIERS),
        "independence_hours": INDEPENDENCE_HOURS,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    request_sha256 = stable_json_sha256(request)
    record_path = compact_dir / "g2c_run_record.json"
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
        "request": request,
        "request_sha256": request_sha256,
        "baseline": (
            "The sparse large-coin Generation-1 observation that the mirrored 1h "
            "rolling-24 extremes plus matching 4h outer Bollinger boundaries added "
            "contact range and volume beyond separately contacted components."
        ),
        "hypothesis": (
            "Simultaneous contact with the recent 1h extreme and matching 4h volatility "
            "boundary produces more non-directional activity than one-component, "
            "other-membership, artificial, shifted, or shared-mechanism clusters."
        ),
        "technical_smoke_not_evidence": args.technical_smoke,
        "workers": args.workers,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(record, record_path)
    tasks = [
        PairTask(
            pair=pair,
            manifest_path=str(args.manifest.resolve()),
            source_event_path=str(source_event_paths[pair]),
            context_path=str(context_path),
            run_id=args.run_id,
            request_sha256=request_sha256,
            overwrite=args.overwrite,
        )
        for pair in pairs
    ]
    try:
        results = run_tasks(tasks, workers=args.workers)
        inventory = DataFrame(results)
        atomic_write_parquet(inventory, compact_dir / "g2c_pair_inventory.parquet")
        failures = inventory.loc[inventory["status"].eq("failed")]
        if not failures.empty:
            raise RuntimeError(
                f"{len(failures)} G2C pair task(s) failed; inspect the pair inventory."
            )
        matches = combine_pair_outputs(
            pair_dir=pair_dir,
            pairs=pairs,
            request_sha256=request_sha256,
        )
        independent = purge_overlapping_event_pairs(
            matches,
            separation_hours=INDEPENDENCE_HOURS,
            group_columns=("pair", "route_id", "pattern_id", "control", "period"),
            left_time_column="actual_event_time",
            right_time_column="control_event_time",
        )
        atomic_write_parquet(independent, compact_dir / "g2c_independent_pairs.parquet")
        pair_period = pair_period_results(independent)
        cohort_period = cohort_period_results(pair_period, independent)
        leave_one_out = leave_one_coin_out_results(pair_period)
        atomic_write_parquet(pair_period, compact_dir / "g2c_pair_period_results.parquet")
        atomic_write_parquet(cohort_period, compact_dir / "g2c_cohort_period_results.parquet")
        atomic_write_parquet(leave_one_out, compact_dir / "g2c_leave_one_coin_out.parquet")
        integrity = integrity_record(
            results=results,
            matches=matches,
            independent=independent,
            technical_smoke=args.technical_smoke,
        )
        atomic_write_json(integrity, compact_dir / "g2c_integrity.json")
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "pair_tasks": results,
                "matched_pairs_before_overlap_purge": len(matches),
                "independent_pairs": len(independent),
                "pair_period_rows": len(pair_period),
                "cohort_period_rows": len(cohort_period),
                "leave_one_coin_out_rows": len(leave_one_out),
                "integrity": str(compact_dir / "g2c_integrity.json"),
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


def validate_frozen_branch() -> None:
    frozen = json.loads(FROZEN_BATCH.read_text(encoding="utf-8"))
    branches = {branch["id"] for branch in frozen["branches"]}
    if "g2c_mirrored_extreme_bollinger_cluster" not in branches:
        raise ValueError("The frozen Generation 2C branch is unavailable.")
    boundary = frozen["research_boundary"]
    if boundary.get("direction_prediction") is not False:
        raise ValueError("Generation 2C must keep direction prediction disabled.")
    if boundary.get("profit_optimization") is not False:
        raise ValueError("Generation 2C must keep profit optimization disabled.")


def validate_source_atlas(
    *, record_path: Path, integrity_path: Path, technical_smoke: bool
) -> None:
    if not record_path.is_file() or not integrity_path.is_file():
        raise FileNotFoundError(
            f"Source cluster atlas record or integrity file is missing: "
            f"{record_path}, {integrity_path}"
        )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if record.get("status") != "completed" or integrity.get("passed") is not True:
        raise ValueError("Source cluster atlas is not a completed integrity-passing run.")
    source_is_smoke = bool(integrity.get("technical_smoke_not_evidence"))
    if source_is_smoke and not technical_smoke:
        raise ValueError(
            "A technical-smoke source atlas cannot support an evidence run."
        )
    modes = set(record.get("representation_modes", ()))
    scales = set(record.get("cluster_scales", ()))
    if "projected" not in modes or "tight" not in scales:
        raise ValueError("Source atlas lacks the frozen projected/tight cluster surface.")


def select_pairs(manifest: dict[str, Any], requested: str) -> list[str]:
    allowed = list(manifest["data"]["pairs"])
    if requested.strip().lower() == "all":
        return allowed
    lookup: dict[str, str] = {}
    for pair in allowed:
        lookup[pair.upper()] = pair
        lookup[pair.split("/", maxsplit=1)[0].upper()] = pair
    selected = []
    for value in (part.strip() for part in requested.split(",")):
        if not value:
            continue
        key = value.upper()
        if key not in lookup:
            raise ValueError(f"Pair {value!r} is outside the frozen manifest.")
        if lookup[key] not in selected:
            selected.append(lookup[key])
    if not selected:
        raise ValueError("No pairs selected.")
    return selected


def stable_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_tasks(tasks: Sequence[PairTask], *, workers: int) -> list[dict[str, Any]]:
    if workers == 1:
        return [safe_build_pair(task) for task in tasks]
    results = []
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


def build_pair(task: PairTask) -> dict[str, Any]:
    started = time.perf_counter()
    output_path = ARTIFACT_ROOT / task.run_id / "pair_matches" / f"{pair_stem(task.pair)}.parquet"
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
            "matched_pairs": len(existing),
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
    aligned = aligned_selected_levels(
        pair=task.pair,
        base=base,
        manifest_path=manifest_path,
        timeframes=tuple(manifest["data"]["source_timeframes"]),
    )
    if not aligned:
        raise ValueError(f"No aligned levels for {task.pair}.")
    valid_matrix = np.column_stack([item.valid for item in aligned])
    level_matrix = np.column_stack(
        [np.where(item.valid, item.level, np.nan) for item in aligned]
    )
    base_atr = numeric_array(base["base_atr"])
    pre_close = numeric_array(base["pre_close"])
    state["state_active_level_count"] = valid_matrix.sum(axis=1).astype(float)
    state["state_cluster_density_2atr"] = np.sum(
        valid_matrix
        & np.isfinite(level_matrix)
        & (np.abs(level_matrix - pre_close[:, None]) <= 2.0 * base_atr[:, None]),
        axis=1,
    ).astype(float)

    atlas = pd.read_parquet(task.source_event_path)
    allowed_periods = {period["id"] for period in manifest["data"]["chronological_periods"]}
    atlas = atlas.loc[
        atlas["representation_mode"].eq("projected")
        & atlas["cluster_scale"].eq("tight")
        & atlas["period"].isin(allowed_periods)
    ].copy()
    atlas = attach_event_state(atlas, state)
    atlas["state_cluster_width_atr"] = pd.to_numeric(
        atlas["cluster_width_atr"], errors="coerce"
    )
    actual_atlas = atlas.loc[atlas["control"].eq("actual_cluster")].copy()
    match_rows: list[dict[str, Any]] = []

    for pattern in PATTERNS:
        generated = generate_pattern_events(
            pair=task.pair,
            base=base,
            manifest=manifest,
            aligned=aligned,
            level_matrix=level_matrix,
            state=state,
            pattern=pattern,
        )
        actual = generated["simultaneous_contact"]
        if actual.empty:
            continue
        for control_name in ("extreme_only_contact", "bollinger_only_contact"):
            match_rows.extend(
                nearest_match_rows(
                    actual=actual,
                    control=generated[control_name],
                    pair=task.pair,
                    route_id="g2c_mirrored_cluster",
                    pattern_id=pattern.id,
                    control_name=control_name,
                    outcomes=OUTCOMES,
                )
            )
        for multiplier in SHIFT_MULTIPLIERS:
            control_name = f"symmetric_shift_{multiplier:+g}atr"
            match_rows.extend(
                nearest_match_rows(
                    actual=actual,
                    control=generated[control_name],
                    pair=task.pair,
                    route_id="g2c_mirrored_cluster",
                    pattern_id=pattern.id,
                    control_name=control_name,
                    outcomes=OUTCOMES,
                )
            )
        target_mask = exact_contacted_pattern_mask(actual_atlas, pattern)
        target_ids = set(
            actual_atlas.loc[target_mask, "cluster_event_id"].dropna().astype(str)
        )
        artificial = atlas.loc[
            atlas["control"].eq("matched_random_cluster")
            & atlas["control_pair_id"].fillna("").astype(str).isin(target_ids)
        ].copy()
        match_rows.extend(
            nearest_match_rows(
                actual=actual,
                control=canonical_control_pool(artificial),
                pair=task.pair,
                route_id="g2c_mirrored_cluster",
                pattern_id=pattern.id,
                control_name="matched_artificial_cluster",
                outcomes=OUTCOMES,
            )
        )
        independent_contact = contacted_independent_mask(actual_atlas)
        other = actual_atlas.loc[
            actual_atlas["multiple_components_contacted"].eq(True)
            & independent_contact
            & ~exact_contacted_pattern_mask(actual_atlas, pattern)
        ].copy()
        match_rows.extend(
            nearest_match_rows(
                actual=actual,
                control=canonical_control_pool(other),
                pair=task.pair,
                route_id="g2c_mirrored_cluster",
                pattern_id=pattern.id,
                control_name="matched_other_component_membership",
                outcomes=OUTCOMES,
            )
        )
        shared = actual_atlas.loc[
            actual_atlas["multiple_components_contacted"].eq(True)
            & shared_mechanism_mask(actual_atlas)
        ].copy()
        match_rows.extend(
            nearest_match_rows(
                actual=actual,
                control=canonical_control_pool(shared),
                pair=task.pair,
                route_id="g2c_mirrored_cluster",
                pattern_id=pattern.id,
                control_name="shared_mechanism_cluster",
                outcomes=OUTCOMES,
            )
        )

    round_key = "8h|generic_round_number|round_nearest|settled"
    round_contact = contacted_component_contains(actual_atlas, round_key)
    round_actual = actual_atlas.loc[
        round_contact
        & actual_atlas["multiple_components_contacted"].eq(True)
        & contacted_independent_mask(actual_atlas)
    ].copy()
    any_round_contact = contacted_component_contains(
        actual_atlas,
        "|generic_round_number|",
    )
    round_free = actual_atlas.loc[
        ~any_round_contact
        & actual_atlas["multiple_components_contacted"].eq(True)
        & contacted_independent_mask(actual_atlas)
    ].copy()
    if not round_actual.empty and not round_free.empty:
        round_actual["comparison_bucket"] = round_actual[
            "contacted_component_count_bucket"
        ].astype(str)
        round_free["comparison_bucket"] = round_free[
            "contacted_component_count_bucket"
        ].astype(str)
        match_rows.extend(
            nearest_match_rows(
                actual=canonical_control_pool(round_actual),
                control=canonical_control_pool(round_free),
                pair=task.pair,
                route_id="g2a_round_cluster",
                pattern_id="8h_round_cluster",
                control_name="cluster_without_round_number",
                outcomes=(
                    "contact_range_ratio",
                    "contact_volume_ratio",
                    "abs_excursion_atr_h1",
                    "volume_ratio_h4",
                    "crossings_h1",
                ),
                extra_group_columns=("comparison_bucket",),
            )
        )

    output = DataFrame(match_rows)
    if output.empty:
        raise ValueError(f"No G2C matches were produced for {task.pair}.")
    output["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    output["run_request_sha256"] = task.request_sha256
    atomic_write_parquet(output, output_path)
    return {
        "pair": task.pair,
        "status": "completed",
        "matched_pairs": len(output),
        "routes": sorted(output["route_id"].unique().tolist()),
        "patterns": sorted(output["pattern_id"].unique().tolist()),
        "controls": sorted(output["control"].unique().tolist()),
        "seconds": round(time.perf_counter() - started, 3),
    }


def find_level(
    aligned: Sequence[AlignedLevel], *, timeframe: str, column: str
) -> AlignedLevel:
    matches = [
        item
        for item in aligned
        if item.timeframe == timeframe and item.spec.column == column
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one aligned level for {timeframe} {column}; found {len(matches)}."
        )
    return matches[0]


def generate_pattern_events(
    *,
    pair: str,
    base: DataFrame,
    manifest: dict[str, Any],
    aligned: Sequence[AlignedLevel],
    level_matrix: np.ndarray,
    state: DataFrame,
    pattern: MirroredPattern,
) -> dict[str, DataFrame]:
    extreme = find_level(
        aligned,
        timeframe=pattern.extreme_timeframe,
        column=pattern.extreme_column,
    )
    band = find_level(
        aligned,
        timeframe=pattern.band_timeframe,
        column=pattern.band_column,
    )
    atr = numeric_array(base["base_atr"])
    high = numeric_array(base["high"])
    low = numeric_array(base["low"])
    extreme_level = np.where(extreme.valid, extreme.level, np.nan)
    band_level = np.where(band.valid, band.level, np.nan)
    extreme_width = np.maximum(
        CLUSTER_HALF_WIDTH_ATR * atr,
        np.abs(extreme_level) * 0.0005,
    )
    band_width = np.maximum(
        CLUSTER_HALF_WIDTH_ATR * atr,
        np.abs(band_level) * 0.0005,
    )
    valid = (
        np.isfinite(extreme_level)
        & np.isfinite(band_level)
        & np.isfinite(atr)
        & (atr > 0.0)
    )
    connected = valid & (
        np.abs(extreme_level - band_level) <= extreme_width + band_width
    )
    extreme_contact = (
        (high >= extreme_level - extreme_width)
        & (low <= extreme_level + extreme_width)
    )
    band_contact = (
        (high >= band_level - band_width) & (low <= band_level + band_width)
    )
    lower = np.minimum(extreme_level - extreme_width, band_level - band_width)
    upper = np.maximum(extreme_level + extreme_width, band_level + band_width)
    center = (lower + upper) / 2.0
    half_width = (upper - lower) / 2.0
    source_available = pd.concat(
        [extreme.source_available, band.source_available], axis=1
    ).max(axis=1)
    source_open = pd.concat([extreme.source_open, band.source_open], axis=1).max(axis=1)
    paths = future_path_matrices(base, max_horizon=4)
    spec = LevelSpec(
        name=pattern.id,
        family="mirrored_extreme_bollinger_cluster",
        batch="g2c",
        column=pattern.id,
    )

    output = {
        "simultaneous_contact": cluster_event_set(
            condition=connected & extreme_contact & band_contact,
            level=center,
            half_width=half_width,
            pair=pair,
            base=base,
            paths=paths,
            spec=spec,
            control="actual_pattern_contact",
            state=state,
            source_available=source_available,
            source_open=source_open,
            manifest=manifest,
        ),
        "extreme_only_contact": cluster_event_set(
            condition=connected & extreme_contact & ~band_contact,
            level=center,
            half_width=half_width,
            pair=pair,
            base=base,
            paths=paths,
            spec=spec,
            control="extreme_only_contact",
            state=state,
            source_available=source_available,
            source_open=source_open,
            manifest=manifest,
        ),
        "bollinger_only_contact": cluster_event_set(
            condition=connected & ~extreme_contact & band_contact,
            level=center,
            half_width=half_width,
            pair=pair,
            base=base,
            paths=paths,
            spec=spec,
            control="bollinger_only_contact",
            state=state,
            source_available=source_available,
            source_open=source_open,
            manifest=manifest,
        ),
    }

    real_width = np.maximum(
        CLUSTER_HALF_WIDTH_ATR * atr[:, None],
        np.abs(level_matrix) * 0.0005,
    )
    for multiplier in SHIFT_MULTIPLIERS:
        shifted_extreme = extreme_level + multiplier * atr
        shifted_band = band_level + multiplier * atr
        shifted_extreme_width = np.maximum(
            CLUSTER_HALF_WIDTH_ATR * atr,
            np.abs(shifted_extreme) * 0.0005,
        )
        shifted_band_width = np.maximum(
            CLUSTER_HALF_WIDTH_ATR * atr,
            np.abs(shifted_band) * 0.0005,
        )
        shifted_connected = valid & (
            np.abs(shifted_extreme - shifted_band)
            <= shifted_extreme_width + shifted_band_width
        )
        shifted_lower = np.minimum(
            shifted_extreme - shifted_extreme_width,
            shifted_band - shifted_band_width,
        )
        shifted_upper = np.maximum(
            shifted_extreme + shifted_extreme_width,
            shifted_band + shifted_band_width,
        )
        shifted_center = (shifted_lower + shifted_upper) / 2.0
        shifted_half_width = (shifted_upper - shifted_lower) / 2.0
        shifted_contact = (
            (high >= shifted_extreme - shifted_extreme_width)
            & (low <= shifted_extreme + shifted_extreme_width)
            & (high >= shifted_band - shifted_band_width)
            & (low <= shifted_band + shifted_band_width)
        )
        overlaps_real = np.any(
            np.isfinite(level_matrix)
            & (level_matrix + real_width >= shifted_lower[:, None])
            & (level_matrix - real_width <= shifted_upper[:, None]),
            axis=1,
        )
        name = f"symmetric_shift_{multiplier:+g}atr"
        output[name] = cluster_event_set(
            condition=shifted_connected & shifted_contact & ~overlaps_real,
            level=shifted_center,
            half_width=shifted_half_width,
            pair=pair,
            base=base,
            paths=paths,
            spec=spec,
            control=name,
            state=state,
            source_available=source_available,
            source_open=source_open,
            manifest=manifest,
        )
    return output


def cluster_event_set(
    *,
    condition: np.ndarray,
    level: np.ndarray,
    half_width: np.ndarray,
    pair: str,
    base: DataFrame,
    paths: dict[str, np.ndarray],
    spec: LevelSpec,
    control: str,
    state: DataFrame,
    source_available: pd.Series,
    source_open: pd.Series,
    manifest: dict[str, Any],
) -> DataFrame:
    starts = episode_start_mask(
        np.asarray(condition, dtype=bool),
        level,
        half_width,
        cooldown=INDEPENDENCE_HOURS,
    )
    indexes = np.flatnonzero(starts)
    if not len(indexes):
        return DataFrame()
    events = event_frame(
        merged=base,
        paths=paths,
        indexes=indexes,
        levels=level[indexes],
        widths=half_width[indexes],
        spec=spec,
        pair=pair,
        timeframe="1h+4h",
        control=control,
        zone_method="tight_connected_cluster",
        source_available=source_available.iloc[indexes].reset_index(drop=True),
        source_open=source_open.iloc[indexes].reset_index(drop=True),
        horizons=(1, 4),
        match_tier=None,
    )
    events = eligible_period_events(
        events,
        manifest,
        embargo_hours=INDEPENDENCE_HOURS,
    )
    events = attach_event_state(events, state)
    if not events.empty:
        events["state_cluster_width_atr"] = (
            2.0 * numeric_array(events["zone_half_width_atr"])
        )
    return events


def attach_event_state(events: DataFrame, state: DataFrame) -> DataFrame:
    if events.empty:
        return events.copy()
    output = events.drop(
        columns=[column for column in STATE_FEATURES if column in events],
        errors="ignore",
    ).copy()
    output["event_time"] = normalize_dates(output["event_time"])
    available_state = state.copy()
    available_state["date"] = normalize_dates(available_state["date"])
    return output.merge(
        available_state,
        left_on="event_time",
        right_on="date",
        how="left",
        validate="many_to_one",
    ).drop(columns="date")


def exact_contacted_pattern_mask(
    events: DataFrame, pattern: MirroredPattern
) -> pd.Series:
    if events.empty:
        return pd.Series(False, index=events.index, dtype=bool)
    extreme_name = pattern.extreme_column.removeprefix("generic_")
    band_name = pattern.band_column.removeprefix("generic_")
    required = {
        f"{pattern.extreme_timeframe}|generic_prior_range|{extreme_name}|settled",
        f"{pattern.band_timeframe}|generic_bollinger|{band_name}|settled",
    }
    if "contacted_component_keys" in events:
        return events["contacted_component_keys"].fillna("").astype(str).map(
            lambda value: required.issubset(set(value.split(";")))
        )
    if "component_keys" not in events or "all_cluster_components_contacted" not in events:
        raise ValueError(
            "Source atlas lacks both exact contacted identities and the conservative "
            "all-components-contacted fallback."
        )
    membership = events["component_keys"].fillna("").astype(str).map(
        lambda value: required.issubset(set(value.split(";")))
    )
    return membership & events["all_cluster_components_contacted"].eq(True)


def contacted_component_contains(events: DataFrame, key_fragment: str) -> pd.Series:
    if "contacted_component_keys" in events:
        return events["contacted_component_keys"].fillna("").astype(str).str.contains(
            key_fragment,
            regex=False,
        )
    if "component_keys" not in events or "all_cluster_components_contacted" not in events:
        raise ValueError(
            "Source atlas cannot establish whether the requested component was contacted."
        )
    return (
        events["component_keys"].fillna("").astype(str).str.contains(
            key_fragment,
            regex=False,
        )
        & events["all_cluster_components_contacted"].eq(True)
    )


def contacted_independent_mask(events: DataFrame) -> pd.Series:
    if "contacted_independent_dependency_groups" in events:
        return events["contacted_independent_dependency_groups"].eq(True)
    required = {"all_cluster_components_contacted", "independent_mechanism_cluster"}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(
            f"Source atlas cannot establish independent contacted components: {missing}"
        )
    return events["all_cluster_components_contacted"].eq(True) & events[
        "independent_mechanism_cluster"
    ].eq(True)


def shared_mechanism_mask(events: DataFrame) -> pd.Series:
    if "same_dependency_group_reference" in events:
        return events["same_dependency_group_reference"].eq(True)
    if "same_mechanism_reference" in events:
        return events["same_mechanism_reference"].eq(True)
    raise ValueError("Source atlas lacks a shared-mechanism reference field.")


def canonical_control_pool(events: DataFrame) -> DataFrame:
    if events.empty:
        return events.copy()
    output = events.copy()
    output["level_name"] = output.get("level_name", "cluster").fillna("cluster")
    output["approach_state"] = output["approach_state"].fillna(
        "already_inside_or_unclear"
    )
    output["state_cluster_width_atr"] = pd.to_numeric(
        output.get("state_cluster_width_atr", output.get("cluster_width_atr")),
        errors="coerce",
    )
    return output


def nearest_match_rows(
    *,
    actual: DataFrame,
    control: DataFrame,
    pair: str,
    route_id: str,
    pattern_id: str,
    control_name: str,
    outcomes: Sequence[str],
    extra_group_columns: Sequence[str] = (),
) -> list[dict[str, Any]]:
    if actual.empty or control.empty:
        return []
    required = {
        "event_time",
        "period",
        "approach_state",
        "pre_distance_atr",
        *STATE_FEATURES,
        *outcomes,
        *extra_group_columns,
    }
    for label, frame in (("actual", actual), ("control", control)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{label} G2C event pool lacks columns: {missing}")
    rows: list[dict[str, Any]] = []
    group_columns = ("period", "approach_state", *extra_group_columns)
    right_groups = {
        key if isinstance(key, tuple) else (key,): group.reset_index(drop=True)
        for key, group in control.groupby(
            list(group_columns), observed=True, dropna=False, sort=False
        )
    }
    for key, left in actual.groupby(
        list(group_columns), observed=True, dropna=False, sort=False
    ):
        key_tuple = key if isinstance(key, tuple) else (key,)
        right = right_groups.get(key_tuple)
        if right is None or right.empty:
            continue
        left = left.reset_index(drop=True)
        pairs, audit = nearest_state_pairs(
            left,
            right,
            state_columns=STATE_FEATURES,
            pre_distance_atr_caliper=0.10,
            minimum_event_separation_hours=INDEPENDENCE_HOURS,
        )
        for left_position, right_position, distance in pairs:
            left_row = left.iloc[left_position]
            right_row = right.iloc[right_position]
            row: dict[str, Any] = {
                "pair": pair,
                "route_id": route_id,
                "pattern_id": pattern_id,
                "control": control_name,
                "period": str(left_row["period"]),
                "approach_state": str(left_row["approach_state"]),
                "level_name": pattern_id,
                "actual_event_time": pd.Timestamp(left_row["event_time"]),
                "control_event_time": pd.Timestamp(right_row["event_time"]),
                "actual_base_index": int(left_row.get("base_index", -1)),
                "control_base_index": int(right_row.get("base_index", -1)),
                "match_distance": float(distance),
                "pre_distance_atr_abs_difference": float(
                    abs(
                        float(left_row["pre_distance_atr"])
                        - float(right_row["pre_distance_atr"])
                    )
                ),
                "actual_pre_distance_atr": float(left_row["pre_distance_atr"]),
                "control_pre_distance_atr": float(right_row["pre_distance_atr"]),
                "actual_source_available_at": left_row.get("source_available_at"),
                "control_source_available_at": right_row.get("source_available_at"),
                "eligible_actual_events": int(audit["eligible_actual"]),
                "eligible_control_events": int(audit["eligible_control"]),
                "geometry_eligible_actual_events": int(
                    audit["geometry_eligible_actual"]
                ),
                "state_matchable_actual_events": int(audit["state_matchable_actual"]),
                "direction_prediction": False,
                "profit_optimization": False,
            }
            for column in extra_group_columns:
                row[column] = left_row[column]
            for feature in STATE_FEATURES:
                row[f"actual_state__{feature}"] = float(left_row[feature])
                row[f"control_state__{feature}"] = float(right_row[feature])
            for outcome in outcomes:
                actual_value = pd.to_numeric(
                    pd.Series([left_row[outcome]]), errors="coerce"
                ).iloc[0]
                control_value = pd.to_numeric(
                    pd.Series([right_row[outcome]]), errors="coerce"
                ).iloc[0]
                row[f"actual__{outcome}"] = float(actual_value)
                row[f"control__{outcome}"] = float(control_value)
                row[f"delta__{outcome}"] = float(actual_value - control_value)
            rows.append(row)
    return rows


def validate_pair_output(
    path: Path, *, pair: str, request_sha256: str
) -> DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "pair",
        "output_schema_version",
        "run_request_sha256",
        "route_id",
        "pattern_id",
        "control",
        "actual_event_time",
        "control_event_time",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Existing G2C pair output lacks columns {missing}: {path}")
    if frame.empty or set(frame["pair"].astype(str)) != {pair}:
        raise ValueError(f"Existing G2C pair output has the wrong pair: {path}")
    if set(frame["output_schema_version"].astype(int)) != {OUTPUT_SCHEMA_VERSION}:
        raise ValueError(f"Existing G2C pair output has the wrong schema: {path}")
    if set(frame["run_request_sha256"].astype(str)) != {request_sha256}:
        raise ValueError(f"Existing G2C pair output has the wrong request hash: {path}")
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
    if not frames:
        return DataFrame()
    return pd.concat(frames, ignore_index=True)


def matched_state_features(frame: DataFrame) -> tuple[str, ...]:
    return tuple(
        feature
        for feature in STATE_FEATURES
        if f"actual_state__{feature}" in frame
        and f"control_state__{feature}" in frame
    )


def matched_pair_balance(
    frame: DataFrame, state_features: Sequence[str]
) -> dict[str, Any]:
    actual = DataFrame(
        {
            feature: pd.to_numeric(
                frame[f"actual_state__{feature}"], errors="coerce"
            ).to_numpy()
            for feature in state_features
        }
    )
    control = DataFrame(
        {
            feature: pd.to_numeric(
                frame[f"control_state__{feature}"], errors="coerce"
            ).to_numpy()
            for feature in state_features
        }
    )
    return balance_diagnostics(actual, control, state_features)


def outcome_metadata(outcome: str) -> tuple[str, int]:
    if "volume" in outcome:
        metric = "volume_activity"
    elif "range" in outcome:
        metric = "candle_range_activity"
    elif "excursion" in outcome:
        metric = "price_path_magnitude"
    elif "crossings" in outcome:
        metric = "level_crossing_activity"
    else:
        metric = "other_non_directional_reaction"
    horizon = 0
    if "_h" in outcome:
        suffix = outcome.rsplit("_h", maxsplit=1)[1]
        if suffix.isdigit():
            horizon = int(suffix)
    return metric, horizon


def pair_period_results(independent_pairs: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    if independent_pairs.empty:
        return DataFrame()
    state_features = matched_state_features(independent_pairs)
    group_columns = ["route_id", "pattern_id", "control", "pair", "period"]
    outcome_columns = sorted(
        column.removeprefix("delta__")
        for column in independent_pairs
        if column.startswith("delta__")
    )
    for key, group in independent_pairs.groupby(
        group_columns, observed=True, dropna=False, sort=False
    ):
        balance = matched_pair_balance(group, state_features)
        for outcome in outcome_columns:
            delta_column = f"delta__{outcome}"
            actual_column = f"actual__{outcome}"
            control_column = f"control__{outcome}"
            if delta_column not in group:
                continue
            delta = pd.to_numeric(group[delta_column], errors="coerce")
            actual = pd.to_numeric(group[actual_column], errors="coerce")
            control = pd.to_numeric(group[control_column], errors="coerce")
            valid = delta.notna() & actual.notna() & control.notna()
            if not valid.any():
                continue
            metric, horizon = outcome_metadata(outcome)
            rows.append(
                {
                    "route_id": key[0],
                    "pattern_id": key[1],
                    "control": key[2],
                    "pair": key[3],
                    "period": key[4],
                    "outcome": outcome,
                    "metric_family": metric,
                    "horizon_hours": horizon,
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
                    "max_absolute_state_smd": balance["max_absolute_smd"],
                    "median_absolute_state_smd": balance["median_absolute_smd"],
                    "state_features_scored": balance["features_scored"],
                    "direction_prediction": False,
                    "profit_optimization": False,
                }
            )
    return DataFrame(rows)


def cohort_period_results(
    pair_period: DataFrame, independent_pairs: DataFrame
) -> DataFrame:
    if pair_period.empty:
        return DataFrame()
    eligible = pair_period.loc[pair_period["pair_period_coverage_eligible"]].copy()
    rows: list[dict[str, Any]] = []
    group_columns = ["route_id", "pattern_id", "control", "period", "outcome"]
    state_features = matched_state_features(independent_pairs)
    for key, group in eligible.groupby(
        group_columns, observed=True, dropna=False, sort=False
    ):
        coins = sorted(group["pair"].astype(str).unique())
        mask = (
            independent_pairs["route_id"].eq(key[0])
            & independent_pairs["pattern_id"].eq(key[1])
            & independent_pairs["control"].eq(key[2])
            & independent_pairs["period"].eq(key[3])
            & independent_pairs["pair"].isin(coins)
        )
        events = independent_pairs.loc[mask]
        delta = pd.to_numeric(events.get(f"delta__{key[4]}"), errors="coerce")
        balance = matched_pair_balance(events, state_features)
        event_count = int(group["independent_event_pairs"].sum())
        rows.append(
            {
                "route_id": key[0],
                "pattern_id": key[1],
                "control": key[2],
                "period": key[3],
                "outcome": key[4],
                "metric_family": outcome_metadata(str(key[4]))[0],
                "horizon_hours": outcome_metadata(str(key[4]))[1],
                "eligible_coin_count": len(coins),
                "eligible_coins": ";".join(coins),
                "independent_event_pairs": event_count,
                "coverage_gate_passed": len(coins) >= MIN_COHORT_PERIOD_COINS
                and event_count >= MIN_COHORT_PERIOD_EPISODES,
                "equal_coin_delta_median": float(group["delta_mean"].median()),
                "equal_coin_positive_fraction": float(
                    group["delta_mean"].gt(0.0).mean()
                ),
                "pooled_event_delta_mean": float(delta.mean()),
                "pooled_event_delta_median": float(delta.median()),
                "pooled_event_positive_fraction": float(delta.gt(0.0).mean()),
                "max_absolute_state_smd": balance["max_absolute_smd"],
                "median_absolute_state_smd": balance["median_absolute_smd"],
                "state_features_scored": balance["features_scored"],
                "direction_prediction": False,
                "profit_optimization": False,
            }
        )
    return DataFrame(rows)


def leave_one_coin_out_results(pair_period: DataFrame) -> DataFrame:
    if pair_period.empty:
        return DataFrame()
    eligible = pair_period.loc[pair_period["pair_period_coverage_eligible"]].copy()
    rows: list[dict[str, Any]] = []
    group_columns = ["route_id", "pattern_id", "control", "period", "outcome"]
    for key, group in eligible.groupby(
        group_columns, observed=True, dropna=False, sort=False
    ):
        for omitted in sorted(group["pair"].astype(str).unique()):
            retained = group.loc[~group["pair"].eq(omitted)]
            if retained.empty:
                continue
            event_count = int(retained["independent_event_pairs"].sum())
            rows.append(
                {
                    "route_id": key[0],
                    "pattern_id": key[1],
                    "control": key[2],
                    "period": key[3],
                    "outcome": key[4],
                    "omitted_pair": omitted,
                    "remaining_coin_count": int(retained["pair"].nunique()),
                    "remaining_independent_event_pairs": event_count,
                    "coverage_gate_passed": int(retained["pair"].nunique())
                    >= MIN_COHORT_PERIOD_COINS
                    and event_count >= MIN_COHORT_PERIOD_EPISODES,
                    "equal_coin_delta_median": float(retained["delta_mean"].median()),
                    "equal_coin_positive_fraction": float(
                        retained["delta_mean"].gt(0.0).mean()
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
) -> dict[str, Any]:
    failures = [row for row in results if row.get("status") == "failed"]
    future_violations = 0
    if not matches.empty:
        for prefix in ("actual", "control"):
            available = pd.to_datetime(
                matches[f"{prefix}_source_available_at"], utc=True, errors="coerce"
            )
            event_time = pd.to_datetime(
                matches[f"{prefix}_event_time"], utc=True, errors="coerce"
            )
            future_violations += int((available.notna() & (available > event_time)).sum())
    separation = pd.Series(dtype=float)
    if not matches.empty:
        separation = (
            (
                pd.to_datetime(matches["actual_event_time"], utc=True)
                - pd.to_datetime(matches["control_event_time"], utc=True)
            )
            .abs()
            .dt.total_seconds()
            .div(3600.0)
        )
    observed_patterns = (
        sorted(matches["pattern_id"].dropna().astype(str).unique())
        if not matches.empty
        else []
    )
    expected_patterns = [pattern.id for pattern in PATTERNS]
    return {
        "created_at_utc": utc_now(),
        "passed": not failures and future_violations == 0 and not independent.empty,
        "technical_smoke_not_evidence": technical_smoke,
        "pairs_requested": len(results),
        "pair_failures": failures,
        "matched_pairs_before_overlap_purge": len(matches),
        "independent_pairs_after_overlap_purge": len(independent),
        "future_source_availability_violations": future_violations,
        "actual_control_event_separation_hours_min": (
            float(separation.min()) if not separation.empty else np.nan
        ),
        "controls_present": (
            sorted(matches["control"].dropna().astype(str).unique())
            if not matches.empty
            else []
        ),
        "routes_present": (
            sorted(matches["route_id"].dropna().astype(str).unique())
            if not matches.empty
            else []
        ),
        "expected_mirrored_patterns": expected_patterns,
        "observed_patterns": observed_patterns,
        "mirrored_patterns_without_matched_support": sorted(
            set(expected_patterns).difference(observed_patterns)
        ),
        "independence_hours": INDEPENDENCE_HOURS,
        "pre_distance_atr_caliper": 0.10,
        "state_features": list(STATE_FEATURES),
        "missing_pattern_support_is_an_evidence_limit_not_a_data_integrity_failure": True,
        "direction_prediction": False,
        "profit_optimization": False,
    }


def pair_stem(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")


if __name__ == "__main__":
    raise SystemExit(main())

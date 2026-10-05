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
    causal_local_state,
    causal_market_context,
    nearest_state_pairs,
    outcome_metadata,
    purge_overlapping_event_pairs_by_key,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_localization import (  # noqa: E501
    eligible_period_events,
)


FROZEN_BATCH = OUTPUT_ROOT / "generation1_review" / "g2_frozen_branch_batch.json"
REPORT_ROOT = OUTPUT_ROOT / "generation2_branches" / "g2d_anchored_vwap"
ARTIFACT_ROOT = (
    LARGE_ARTIFACT_ROOT / "generation2_branches" / "g2d_anchored_vwap"
)
OUTPUT_SCHEMA_VERSION = 2
SWING_LEFT_BARS = 3
SWING_RIGHT_BARS = 3
ZONE_HALF_WIDTH_ATR = 0.10
PRICE_SHIFTS_ATR = (-1.0, -0.5, 0.5, 1.0)
DISPERSION_MULTIPLIERS = (0, -1, 1, -2, 2)
MIN_PAIR_PERIOD_EPISODES = 5
MIN_COHORT_PERIOD_COINS = 5
MIN_COHORT_PERIOD_EPISODES = 50
MAX_STATE_SMD = 0.50
MAX_MEDIAN_STATE_SMD = 0.20
PRIMARY_OUTCOMES = (
    "contact_range_ratio",
    "contact_volume_ratio",
    "abs_excursion_atr_h4",
    "range_ratio_h4",
    "volume_ratio_h4",
    "pressure_change_abs_h4",
    "dwell_fraction_h4",
    "crossings_h4",
)
DESCRIPTIVE_OUTCOMES = (
    "abs_excursion_atr_h1",
    "range_ratio_h1",
    "volume_ratio_h1",
    "pressure_change_abs_h1",
    "dwell_fraction_h1",
    "crossings_h1",
    "abs_excursion_atr_h24",
    "range_ratio_h24",
    "volume_ratio_h24",
    "pressure_change_abs_h24",
    "dwell_fraction_h24",
    "crossings_h24",
)
OUTCOMES = (*PRIMARY_OUTCOMES, *DESCRIPTIVE_OUTCOMES)
INDEPENDENCE_HOURS_BY_OUTCOME = {
    outcome: (
        24
        if outcome.endswith("_h24")
        else 4
        if outcome.startswith("contact_") or outcome.endswith("_h4")
        else 1
    )
    for outcome in OUTCOMES
}
OUTCOMES_BY_RESPONSE_WINDOW = {
    f"h{hours}": tuple(
        outcome
        for outcome in OUTCOMES
        if INDEPENDENCE_HOURS_BY_OUTCOME[outcome] == hours
    )
    for hours in (1, 4, 24)
}
SEPARATION_HOURS_BY_RESPONSE_WINDOW = {
    response_window: int(response_window.removeprefix("h"))
    for response_window in OUTCOMES_BY_RESPONSE_WINDOW
}
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
    "state_anchor_age_hours",
    "state_anchor_average_volume_ratio",
    "state_active_anchor_level_count",
    "state_anchor_family_count_near_price_2atr",
    "state_anchor_zone_coverage_4atr",
)


@dataclass(frozen=True)
class AnchorStats:
    family: str
    start_index: np.ndarray
    source_open: Series
    age: np.ndarray
    average_volume_ratio: np.ndarray
    vwap: np.ndarray
    weighted_std: np.ndarray
    simple_mean: np.ndarray
    simple_std: np.ndarray
    same_horizon_ema: np.ndarray
    delay_hours: int


@dataclass(frozen=True)
class AnchorLevel:
    family: str
    name: str
    multiplier: int
    level: np.ndarray
    sma_control: np.ndarray
    ema_control: np.ndarray
    valid: np.ndarray
    source_open: Series
    age: np.ndarray
    average_volume_ratio: np.ndarray
    delay_hours: int


@dataclass(frozen=True)
class PairTask:
    pair: str
    manifest_path: str
    context_path: str
    run_id: str
    request_sha256: str
    overwrite: bool


@dataclass(frozen=True)
class SummarySource:
    run_id: str
    record_path: Path
    integrity_path: Path
    independent_pairs_path: Path
    request_sha256: str
    independent_comparisons: int
    file_hashes: dict[str, str]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 2D causal anchored-VWAP zone test. It measures "
            "non-directional reaction behaviour and does not optimize profit."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--technical-smoke", action="store_true")
    parser.add_argument(
        "--summary-source-run-id",
        help=(
            "Rebuild only G2D summaries from a completed run's preserved "
            "independent-pairs table. Pair evidence is not rebuilt or copied."
        ),
    )
    args = parser.parse_args(argv)

    validate_frozen_branch()
    if args.summary_source_run_id:
        if args.technical_smoke:
            raise ValueError("Summary-only rebuilds cannot be technical smokes.")
        if args.pairs.strip().lower() != "all":
            raise ValueError("Summary-only rebuilds use every pair in the source run.")
        if args.workers != 1:
            raise ValueError("Summary-only rebuilds are single-process calculations.")
        return rebuild_completed_summaries(
            run_id=args.run_id,
            source_run_id=args.summary_source_run_id,
            overwrite=args.overwrite,
        )

    manifest = load_manifest(args.manifest)
    validate_worker_count(args.workers, manifest=manifest)
    pairs = select_pairs(manifest, args.pairs)
    compact_dir = REPORT_ROOT / args.run_id
    bulky_dir = ARTIFACT_ROOT / args.run_id
    pair_dir = bulky_dir / "pair_matches"
    compact_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    context_path = compact_dir / "g2d_causal_market_context.parquet"
    if args.overwrite or not context_path.is_file():
        atomic_write_parquet(causal_market_context(manifest), context_path)

    request = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "frozen_batch_sha256": sha256_file(FROZEN_BATCH),
        "pairs": pairs,
        "anchor_families": [
            "utc_daily_reset",
            "utc_weekly_reset",
            "confirmed_swing_high",
            "confirmed_swing_low",
        ],
        "dispersion_multipliers": list(DISPERSION_MULTIPLIERS),
        "swing_confirmation": {
            "left_bars": SWING_LEFT_BARS,
            "right_bars": SWING_RIGHT_BARS,
        },
        "zone_half_width_atr": ZONE_HALF_WIDTH_ATR,
        "price_shifts_atr": list(PRICE_SHIFTS_ATR),
        "controls": [
            "ordinary_sma_same_anchor_horizon",
            "ordinary_ema_same_anchor_horizon",
            "time_shifted_anchor",
            "price_shifted_zone",
            "random_eligible_zone",
            "same_state_no_level",
            "same_density_and_zone_coverage_matching",
        ],
        "primary_outcomes": list(PRIMARY_OUTCOMES),
        "descriptive_outcomes": list(DESCRIPTIVE_OUTCOMES),
        "independence_hours_by_outcome": INDEPENDENCE_HOURS_BY_OUTCOME,
        "outcomes_by_response_window": OUTCOMES_BY_RESPONSE_WINDOW,
        "state_features": list(STATE_FEATURES),
        "direction_prediction": False,
        "profit_optimization": False,
    }
    request_sha256 = stable_json_sha256(request)
    record_path = compact_dir / "g2d_run_record.json"
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
            "Ordinary arithmetic and exponential price averages with the same "
            "anchor history, plus delayed, shifted, random, and no-level locations."
        ),
        "hypothesis": (
            "At least one fixed causal volume-weighted anchor family marks a repeated "
            "non-directional activity or acceptance response beyond every control."
        ),
        "pass_rule": (
            "A named family and band must retain the same trader-readable response "
            "against every required control in at least two validation periods, with "
            "at least five eligible coins, fifty independent comparisons, acceptable "
            "state balance, and no one-coin dependence."
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
        atomic_write_parquet(inventory, compact_dir / "g2d_pair_inventory.parquet")
        failures = inventory.loc[inventory["status"].eq("failed")]
        if not failures.empty:
            raise RuntimeError(
                f"{len(failures)} G2D pair task(s) failed; inspect the pair inventory."
            )
        matches = combine_pair_outputs(
            pair_dir=pair_dir,
            pairs=pairs,
            request_sha256=request_sha256,
        )
        independent = purge_overlapping_event_pairs_by_key(
            matches,
            separation_hours_by_key=SEPARATION_HOURS_BY_RESPONSE_WINDOW,
            key_column="response_window",
            group_columns=(
                "pair",
                "anchor_family",
                "level_name",
                "anchor_scope",
                "control",
                "period",
                "response_window",
            ),
            left_time_column="actual_event_time",
            right_time_column="control_event_time",
        )
        atomic_write_parquet(independent, compact_dir / "g2d_independent_pairs.parquet")
        pair_period = pair_period_results(independent)
        cohort_period = cohort_period_results(pair_period, independent)
        leave_one_out = leave_one_coin_out_results(pair_period)
        atomic_write_parquet(pair_period, compact_dir / "g2d_pair_period_results.parquet")
        atomic_write_parquet(
            cohort_period,
            compact_dir / "g2d_cohort_period_results.parquet",
        )
        atomic_write_parquet(
            leave_one_out,
            compact_dir / "g2d_leave_one_coin_out.parquet",
        )
        integrity = integrity_record(
            results=results,
            matches=matches,
            independent=independent,
            technical_smoke=args.technical_smoke,
        )
        atomic_write_json(integrity, compact_dir / "g2d_integrity.json")
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
                "integrity": str(compact_dir / "g2d_integrity.json"),
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
    if "g2d_causal_anchored_vwap_zones" not in branches:
        raise ValueError("The frozen Generation 2D branch is unavailable.")
    boundary = frozen["research_boundary"]
    if boundary.get("direction_prediction") is not False:
        raise ValueError("Generation 2D must keep direction prediction disabled.")
    if boundary.get("profit_optimization") is not False:
        raise ValueError("Generation 2D must keep profit optimization disabled.")


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


def validated_summary_source(source_run_id: str) -> SummarySource:
    source_dir = REPORT_ROOT / source_run_id
    record_path = source_dir / "g2d_run_record.json"
    integrity_path = source_dir / "g2d_integrity.json"
    independent_pairs_path = source_dir / "g2d_independent_pairs.parquet"
    required_paths = (record_path, integrity_path, independent_pairs_path)
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            f"G2D summary source is incomplete; missing {missing_paths}."
        )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    request_sha256, independent_comparisons = validate_summary_source_record(
        record, source_run_id=source_run_id
    )
    validate_summary_source_integrity(
        integrity, expected_rows=independent_comparisons
    )
    return SummarySource(
        run_id=source_run_id,
        record_path=record_path,
        integrity_path=integrity_path,
        independent_pairs_path=independent_pairs_path,
        request_sha256=request_sha256,
        independent_comparisons=independent_comparisons,
        file_hashes={
            "run_record_sha256": sha256_file(record_path),
            "integrity_sha256": sha256_file(integrity_path),
            "independent_pairs_sha256": sha256_file(independent_pairs_path),
        },
    )


def validate_summary_source_record(
    record: dict[str, Any], *, source_run_id: str
) -> tuple[str, int]:
    if record.get("run_id") != source_run_id:
        raise ValueError("G2D summary source record has the wrong run ID.")
    if record.get("status") != "completed":
        raise ValueError("G2D summaries require a completed source run.")
    if record.get("technical_smoke_not_evidence") is not False:
        raise ValueError("G2D summaries cannot be rebuilt from a technical smoke.")
    if record.get("direction_prediction") is not False:
        raise ValueError("G2D summary source enabled direction prediction.")
    if record.get("profit_optimization") is not False:
        raise ValueError("G2D summary source enabled profit optimization.")
    request = record.get("request")
    if not isinstance(request, dict):
        raise ValueError("G2D summary source lacks its request contract.")
    request_sha256 = str(record.get("request_sha256", ""))
    if stable_json_sha256(request) != request_sha256:
        raise ValueError("G2D summary source request hash is inconsistent.")
    independent_comparisons = int(record.get("independent_comparisons", -1))
    if independent_comparisons < 1:
        raise ValueError("G2D summary source has no independent comparisons.")
    return request_sha256, independent_comparisons


def validate_summary_source_integrity(
    integrity: dict[str, Any], *, expected_rows: int
) -> None:
    if integrity.get("passed") is not True:
        raise ValueError("G2D summaries require a source run with passed integrity.")
    if integrity.get("technical_smoke_not_evidence") is not False:
        raise ValueError("G2D source integrity identifies a technical smoke.")
    if integrity.get("direction_prediction") is not False:
        raise ValueError("G2D source integrity enabled direction prediction.")
    if integrity.get("profit_optimization") is not False:
        raise ValueError("G2D source integrity enabled profit optimization.")
    observed_rows = int(
        integrity.get("independent_comparisons_after_horizon_purge", -1)
    )
    if observed_rows != expected_rows:
        raise ValueError("G2D source record and integrity row counts disagree.")


def rebuild_completed_summaries(
    *, run_id: str, source_run_id: str, overwrite: bool
) -> int:
    if run_id == source_run_id:
        raise ValueError("A summary rebuild must use a new run ID.")
    source = validated_summary_source(source_run_id)
    request = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "mode": "summary_only_declared_matching_balance",
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_batch_sha256": sha256_file(FROZEN_BATCH),
        "source_run_id": source_run_id,
        "source_request_sha256": source.request_sha256,
        "source_files": source.file_hashes,
        "source_independent_comparisons": source.independent_comparisons,
        "balance_feature_source": "matching_state_features",
        "direction_prediction": False,
        "profit_optimization": False,
    }
    request_sha256 = stable_json_sha256(request)
    compact_dir = REPORT_ROOT / run_id
    if compact_dir.is_dir() and any(compact_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Summary output run already exists; use a new run ID: {compact_dir}"
        )
    compact_dir.mkdir(parents=True, exist_ok=True)
    record_path = compact_dir / "g2d_run_record.json"
    record = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "running",
        "mode": "summary_only_declared_matching_balance",
        "started_at_utc": utc_now(),
        "orchestrator_pid": os.getpid(),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "request": request,
        "request_sha256": request_sha256,
        "summary_source_run_id": source_run_id,
        "summary_source_independent_pairs": str(source.independent_pairs_path),
        "technical_smoke_not_evidence": False,
        "workers": 1,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(record, record_path)
    try:
        independent = pd.read_parquet(source.independent_pairs_path)
        validate_summary_source_frame(
            independent,
            expected_rows=source.independent_comparisons,
            source_request_sha256=source.request_sha256,
        )
        pair_period = pair_period_results(independent)
        cohort_period = cohort_period_results(pair_period, independent)
        leave_one_out = leave_one_coin_out_results(pair_period)
        output_paths = {
            "pair_period": compact_dir / "g2d_pair_period_results.parquet",
            "cohort_period": compact_dir / "g2d_cohort_period_results.parquet",
            "leave_one_coin_out": compact_dir / "g2d_leave_one_coin_out.parquet",
        }
        atomic_write_parquet(pair_period, output_paths["pair_period"])
        atomic_write_parquet(cohort_period, output_paths["cohort_period"])
        atomic_write_parquet(leave_one_out, output_paths["leave_one_coin_out"])
        integrity = {
            "created_at_utc": utc_now(),
            "passed": True,
            "mode": "summary_only_declared_matching_balance",
            "source_run_id": source_run_id,
            "source_integrity_passed": True,
            "source_files": source.file_hashes,
            "source_independent_comparisons": source.independent_comparisons,
            "balance_feature_source": "matching_state_features",
            "pair_period_rows": len(pair_period),
            "cohort_period_rows": len(cohort_period),
            "leave_one_coin_out_rows": len(leave_one_out),
            "output_files": {
                name: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for name, path in output_paths.items()
            },
            "direction_prediction": False,
            "profit_optimization": False,
            "technical_smoke_not_evidence": False,
        }
        integrity_path = compact_dir / "g2d_integrity.json"
        atomic_write_json(integrity, integrity_path)
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "source_independent_comparisons": source.independent_comparisons,
                "pair_period_rows": len(pair_period),
                "cohort_period_rows": len(cohort_period),
                "leave_one_coin_out_rows": len(leave_one_out),
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


def validate_summary_source_frame(
    frame: DataFrame, *, expected_rows: int, source_request_sha256: str
) -> None:
    required = {
        "output_schema_version",
        "run_request_sha256",
        "matching_state_features",
        "direction_prediction",
        "profit_optimization",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"G2D summary source lacks required columns: {missing}")
    if len(frame) != expected_rows:
        raise ValueError(
            "G2D summary source row count differs from its completed run record: "
            f"{len(frame)} != {expected_rows}."
        )
    if set(frame["output_schema_version"].dropna().astype(int)) != {
        OUTPUT_SCHEMA_VERSION
    }:
        raise ValueError("G2D summary source has the wrong output schema.")
    if set(frame["run_request_sha256"].dropna().astype(str)) != {
        source_request_sha256
    }:
        raise ValueError("G2D summary source has the wrong request hash.")
    for column in ("direction_prediction", "profit_optimization"):
        values = set(frame[column].dropna().tolist())
        if len(frame[column].dropna()) != len(frame) or values != {False}:
            raise ValueError(f"G2D summary source violates {column}=False.")


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

    manifest = load_manifest(Path(task.manifest_path))
    base = prepare_base_market_frame(task.pair, manifest)
    context = pd.read_parquet(task.context_path)
    state = causal_local_state(base).merge(
        context,
        on="date",
        how="left",
        validate="one_to_one",
    )
    anchor_levels = causal_anchor_levels(base)
    if not anchor_levels:
        raise ValueError(f"No causal anchored-VWAP levels for {task.pair}.")
    base_atr = numeric_array(base["base_atr"])
    pre_close = numeric_array(base["pre_close"])
    level_matrix = np.column_stack(
        [np.where(item.valid, item.level, np.nan) for item in anchor_levels]
    )
    family_matrix = np.asarray([item.family for item in anchor_levels], dtype=object)
    valid_matrix = np.isfinite(level_matrix)
    state["state_active_anchor_level_count"] = valid_matrix.sum(axis=1).astype(float)
    near = (
        valid_matrix
        & np.isfinite(base_atr[:, None])
        & (base_atr[:, None] > 0.0)
        & (np.abs(level_matrix - pre_close[:, None]) <= 2.0 * base_atr[:, None])
    )
    family_near = np.zeros(len(base), dtype=float)
    for family in sorted(set(family_matrix)):
        family_near += near[:, family_matrix == family].any(axis=1).astype(float)
    state["state_anchor_family_count_near_price_2atr"] = family_near
    state["state_anchor_zone_coverage_4atr"] = anchor_zone_coverage(
        level_matrix=level_matrix,
        base_atr=base_atr,
        pre_close=pre_close,
    )
    paths = future_path_matrices(base, max_horizon=24)
    match_rows: list[dict[str, Any]] = []
    actual_event_count = 0
    control_event_counts: dict[str, int] = {}

    for source_index, anchor in enumerate(anchor_levels):
        anchor_rows, anchor_actual_count, anchor_control_counts = evaluate_anchor_level(
            anchor=anchor,
            source_index=source_index,
            pair=task.pair,
            base=base,
            state=state,
            paths=paths,
            manifest=manifest,
            base_atr=base_atr,
            level_matrix=level_matrix,
            family_matrix=family_matrix,
        )
        match_rows.extend(anchor_rows)
        actual_event_count += anchor_actual_count
        for control_name, count in anchor_control_counts.items():
            control_event_counts[control_name] = (
                control_event_counts.get(control_name, 0) + count
            )

    output = DataFrame(match_rows)
    if output.empty:
        raise ValueError(f"No matchable G2D outcome pairs for {task.pair}.")
    output["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    output["run_request_sha256"] = task.request_sha256
    atomic_write_parquet(output, output_path)
    return {
        "pair": task.pair,
        "status": "completed",
        "anchor_level_series": len(anchor_levels),
        "actual_contact_episodes": actual_event_count,
        "control_event_counts": control_event_counts,
        "matched_comparisons": len(output),
        "anchor_families": sorted(output["anchor_family"].unique().tolist()),
        "controls": sorted(output["control"].unique().tolist()),
        "seconds": round(time.perf_counter() - started, 3),
    }


def evaluate_anchor_level(
    *,
    anchor: AnchorLevel,
    source_index: int,
    pair: str,
    base: DataFrame,
    state: DataFrame,
    paths: dict[str, np.ndarray],
    manifest: dict[str, Any],
    base_atr: np.ndarray,
    level_matrix: np.ndarray,
    family_matrix: np.ndarray,
) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
    spec = LevelSpec(
        name=anchor.name,
        family="causal_anchored_vwap",
        batch="g2d",
        column=anchor.name,
    )
    width = np.maximum(
        ZONE_HALF_WIDTH_ATR * base_atr,
        np.abs(anchor.level) * 0.0005,
    )
    actual = extract_episode_events(
        merged=base,
        paths=paths,
        spec=spec,
        pair=pair,
        timeframe="1h_causal_anchor",
        level=anchor.level,
        valid=anchor.valid,
        half_width=width,
        source_available=normalize_dates(base["date"]),
        source_open=anchor.source_open,
        control="actual_anchored_vwap",
        zone_method="tight_base_atr",
        horizons=(1, 4, 24),
        event_kind="contact",
    )
    actual = eligible_period_events(actual, manifest, embargo_hours=24)
    if actual.empty:
        return [], 0, {}
    anchor_state = state.copy()
    anchor_state["state_anchor_age_hours"] = anchor.age
    anchor_state["state_anchor_average_volume_ratio"] = anchor.average_volume_ratio
    actual = attach_state_and_absolute_outcomes(actual, anchor_state)
    actual = attach_anchor_overlap(
        actual,
        level_matrix=level_matrix,
        family_matrix=family_matrix,
        source_index=source_index,
        base_atr=base_atr,
    )
    actual["anchor_scope"] = np.where(
        actual["overlap_other_anchor_family_count"].gt(0),
        "cross_family_anchor_cluster",
        "isolated_anchor_family",
    )
    rows: list[dict[str, Any]] = []
    control_counts: dict[str, int] = {}
    controls = anchored_level_controls(anchor=anchor, base=base, base_atr=base_atr)
    for control_name, definition in controls.items():
        control_events = deterministic_control_events(
            definition=definition,
            control_name=control_name,
            spec=spec,
            pair=pair,
            base=base,
            paths=paths,
            manifest=manifest,
            anchor_state=anchor_state,
            level_matrix=level_matrix,
            family_matrix=family_matrix,
            source_index=source_index,
            base_atr=base_atr,
        )
        append_control_matches(
            rows=rows,
            control_counts=control_counts,
            actual=actual,
            control_events=control_events,
            pair=pair,
            anchor=anchor,
            control_name=control_name,
            state_features=STATE_FEATURES,
        )

    random_events = matched_random_time_events(
        merged=base,
        paths=paths,
        actual=actual,
        spec=spec,
        pair=pair,
        timeframe="1h_causal_anchor",
        zone_method="tight_base_atr",
        horizons=(1, 4, 24),
    )
    random_events = eligible_period_events(random_events, manifest, embargo_hours=24)
    random_events = prepare_control_events(
        random_events,
        anchor_state=anchor_state,
        level_matrix=level_matrix,
        family_matrix=family_matrix,
        source_index=source_index,
        base_atr=base_atr,
    )
    no_level = random_events.loc[~random_events["overlap_any_anchor_level"]].copy()
    for control_name, control_events, match_features in (
        ("random_eligible_zone", random_events, STATE_FEATURES[:16]),
        ("same_state_no_level", no_level, STATE_FEATURES[:18]),
        ("same_density_coverage_no_level", no_level, STATE_FEATURES),
    ):
        append_control_matches(
            rows=rows,
            control_counts=control_counts,
            actual=actual,
            control_events=control_events,
            pair=pair,
            anchor=anchor,
            control_name=control_name,
            state_features=match_features,
        )
    return rows, len(actual), control_counts


def deterministic_control_events(
    *,
    definition: dict[str, Any],
    control_name: str,
    spec: LevelSpec,
    pair: str,
    base: DataFrame,
    paths: dict[str, np.ndarray],
    manifest: dict[str, Any],
    anchor_state: DataFrame,
    level_matrix: np.ndarray,
    family_matrix: np.ndarray,
    source_index: int,
    base_atr: np.ndarray,
) -> DataFrame:
    control_width = np.maximum(
        ZONE_HALF_WIDTH_ATR * base_atr,
        np.abs(definition["level"]) * 0.0005,
    )
    events = extract_episode_events(
        merged=base,
        paths=paths,
        spec=spec,
        pair=pair,
        timeframe="1h_causal_anchor",
        level=definition["level"],
        valid=definition["valid"],
        half_width=control_width,
        source_available=definition["source_available"],
        source_open=definition["source_open"],
        control=control_name,
        zone_method="tight_base_atr",
        horizons=(1, 4, 24),
        event_kind="contact",
    )
    events = eligible_period_events(events, manifest, embargo_hours=24)
    events = prepare_control_events(
        events,
        anchor_state=anchor_state,
        level_matrix=level_matrix,
        family_matrix=family_matrix,
        source_index=source_index,
        base_atr=base_atr,
    )
    if control_name.startswith(("time_shifted", "price_shifted")):
        events = events.loc[~events["overlap_any_anchor_level"]].copy()
    return events


def prepare_control_events(
    events: DataFrame,
    *,
    anchor_state: DataFrame,
    level_matrix: np.ndarray,
    family_matrix: np.ndarray,
    source_index: int,
    base_atr: np.ndarray,
) -> DataFrame:
    if events.empty:
        return events.copy()
    events = attach_state_and_absolute_outcomes(events, anchor_state)
    return attach_anchor_overlap(
        events,
        level_matrix=level_matrix,
        family_matrix=family_matrix,
        source_index=source_index,
        base_atr=base_atr,
    )


def append_control_matches(
    *,
    rows: list[dict[str, Any]],
    control_counts: dict[str, int],
    actual: DataFrame,
    control_events: DataFrame,
    pair: str,
    anchor: AnchorLevel,
    control_name: str,
    state_features: Sequence[str],
) -> None:
    if control_events.empty:
        return
    control_counts[control_name] = control_counts.get(control_name, 0) + len(
        control_events
    )
    for scope, scoped_actual in actual.groupby("anchor_scope", observed=True):
        rows.extend(
            matched_outcome_rows(
                actual=scoped_actual,
                control=control_events,
                pair=pair,
                anchor=anchor,
                anchor_scope=str(scope),
                control_name=control_name,
                state_features=state_features,
            )
        )


def causal_anchor_levels(base: DataFrame) -> list[AnchorLevel]:
    stats = [
        reset_anchor_stats(base, family="utc_daily_reset", frequency="D", delay_hours=24),
        reset_anchor_stats(
            base,
            family="utc_weekly_reset",
            frequency="W-MON",
            delay_hours=168,
        ),
        swing_anchor_stats(base, side="high"),
        swing_anchor_stats(base, side="low"),
    ]
    levels: list[AnchorLevel] = []
    for item in stats:
        for multiplier in DISPERSION_MULTIPLIERS:
            suffix = (
                "center"
                if multiplier == 0
                else f"{'plus' if multiplier > 0 else 'minus'}{abs(multiplier)}sd"
            )
            level = item.vwap + multiplier * item.weighted_std
            sma = item.simple_mean + multiplier * item.simple_std
            ema = item.same_horizon_ema + multiplier * item.simple_std
            valid = (
                np.isfinite(level)
                & np.isfinite(sma)
                & np.isfinite(ema)
                & (level > 0.0)
                & (item.age >= 2.0)
            )
            levels.append(
                AnchorLevel(
                    family=item.family,
                    name=f"{item.family}_{suffix}",
                    multiplier=multiplier,
                    level=level,
                    sma_control=sma,
                    ema_control=ema,
                    valid=valid,
                    source_open=item.source_open,
                    age=item.age,
                    average_volume_ratio=item.average_volume_ratio,
                    delay_hours=item.delay_hours,
                )
            )
    return levels


def reset_anchor_stats(
    base: DataFrame,
    *,
    family: str,
    frequency: str,
    delay_hours: int,
) -> AnchorStats:
    dates = normalize_dates(base["date"])
    if frequency == "D":
        keys = dates.dt.floor("D")
    elif frequency == "W-MON":
        keys = dates.dt.floor("D") - pd.to_timedelta(dates.dt.weekday, unit="D")
    else:
        raise ValueError(f"Unsupported causal reset frequency: {frequency}")
    positions = np.arange(len(base), dtype=np.int64)
    start_index = (
        pd.Series(positions).groupby(keys, sort=False).transform("min").to_numpy(dtype=np.int64)
    )
    return anchored_stats_from_start(
        base,
        family=family,
        start_index=start_index,
        delay_hours=delay_hours,
    )


def swing_anchor_stats(base: DataFrame, *, side: str) -> AnchorStats:
    high = numeric_array(base["high"])
    low = numeric_array(base["low"])
    length = len(base)
    reset = np.full(length, -1, dtype=np.int64)
    for pivot in range(SWING_LEFT_BARS, length - SWING_RIGHT_BARS):
        confirmation_event = pivot + SWING_RIGHT_BARS + 1
        if confirmation_event >= length:
            continue
        if side == "high":
            peers = np.concatenate(
                (
                    high[pivot - SWING_LEFT_BARS : pivot],
                    high[pivot + 1 : pivot + SWING_RIGHT_BARS + 1],
                )
            )
            confirmed = np.isfinite(high[pivot]) and np.isfinite(peers).all() and (
                high[pivot] > np.max(peers)
            )
        elif side == "low":
            peers = np.concatenate(
                (
                    low[pivot - SWING_LEFT_BARS : pivot],
                    low[pivot + 1 : pivot + SWING_RIGHT_BARS + 1],
                )
            )
            confirmed = np.isfinite(low[pivot]) and np.isfinite(peers).all() and (
                low[pivot] < np.min(peers)
            )
        else:
            raise ValueError(f"Unsupported swing side: {side}")
        if confirmed:
            reset[confirmation_event] = pivot
    start_index = np.full(length, -1, dtype=np.int64)
    current = -1
    for position in range(length):
        if reset[position] >= 0:
            current = int(reset[position])
        start_index[position] = current
    return anchored_stats_from_start(
        base,
        family=f"confirmed_swing_{side}",
        start_index=start_index,
        delay_hours=168,
    )


def anchored_stats_from_start(
    base: DataFrame,
    *,
    family: str,
    start_index: np.ndarray,
    delay_hours: int,
) -> AnchorStats:
    high = numeric_array(base["high"])
    low = numeric_array(base["low"])
    close = numeric_array(base["close"])
    volume = numeric_array(base["volume"])
    typical = (high + low + close) / 3.0
    valid_price = np.isfinite(typical)
    valid_weight = valid_price & np.isfinite(volume) & (volume > 0.0)
    price = np.where(valid_price, typical, 0.0)
    weight = np.where(valid_weight, volume, 0.0)
    prefix_weight = np.concatenate(([0.0], np.cumsum(weight)))
    prefix_weighted_price = np.concatenate(([0.0], np.cumsum(weight * price)))
    prefix_weighted_square = np.concatenate(([0.0], np.cumsum(weight * price * price)))
    prefix_price = np.concatenate(([0.0], np.cumsum(price)))
    prefix_square = np.concatenate(([0.0], np.cumsum(price * price)))
    prefix_count = np.concatenate(([0.0], np.cumsum(valid_price.astype(float))))
    rows = np.arange(len(base), dtype=np.int64)
    valid_start = (start_index >= 0) & (start_index < rows)
    safe_start = np.where(valid_start, start_index, rows)
    total_weight = prefix_weight[rows] - prefix_weight[safe_start]
    weighted_sum = prefix_weighted_price[rows] - prefix_weighted_price[safe_start]
    weighted_square = prefix_weighted_square[rows] - prefix_weighted_square[safe_start]
    count = prefix_count[rows] - prefix_count[safe_start]
    simple_sum = prefix_price[rows] - prefix_price[safe_start]
    simple_square = prefix_square[rows] - prefix_square[safe_start]
    vwap = safe_divide(weighted_sum, total_weight)
    weighted_variance = safe_divide(weighted_square, total_weight) - vwap * vwap
    weighted_std = np.sqrt(np.maximum(weighted_variance, 0.0))
    simple_mean = safe_divide(simple_sum, count)
    simple_variance = safe_divide(simple_square, count) - simple_mean * simple_mean
    simple_std = np.sqrt(np.maximum(simple_variance, 0.0))
    age = np.where(valid_start, rows - start_index, np.nan).astype(float)
    pre_volume = numeric_array(base["pre_volume_median_24"])
    average_volume_ratio = safe_divide(safe_divide(total_weight, count), pre_volume)
    same_horizon_ema = exact_anchor_horizon_ema(typical, start_index)
    dates = normalize_dates(base["date"])
    source_values = pd.Series(pd.NaT, index=base.index, dtype="datetime64[ns, UTC]")
    source_values.loc[valid_start] = dates.iloc[start_index[valid_start]].to_numpy()
    for array in (
        vwap,
        weighted_std,
        simple_mean,
        simple_std,
        same_horizon_ema,
        average_volume_ratio,
    ):
        array[~valid_start] = np.nan
    return AnchorStats(
        family=family,
        start_index=start_index,
        source_open=source_values,
        age=age,
        average_volume_ratio=average_volume_ratio,
        vwap=vwap,
        weighted_std=weighted_std,
        simple_mean=simple_mean,
        simple_std=simple_std,
        same_horizon_ema=same_horizon_ema,
        delay_hours=delay_hours,
    )


def exact_anchor_horizon_ema(values: np.ndarray, start_index: np.ndarray) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype=np.float64)
    for position, start in enumerate(start_index):
        start = int(start)
        if start < 0 or start >= position:
            continue
        segment = values[start:position]
        valid = np.isfinite(segment)
        count = int(valid.sum())
        if count < 2:
            continue
        selected = segment[valid]
        alpha = 2.0 / (count + 1.0)
        weights = np.power(1.0 - alpha, np.arange(count - 1, -1, -1))
        output[position] = float(np.dot(selected, weights) / weights.sum())
    return output


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.full(len(numerator), np.nan, dtype=np.float64)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (np.abs(denominator) > 1e-12)
    )
    output[valid] = numerator[valid] / denominator[valid]
    return output


def shift_numeric(values: np.ndarray, hours: int) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype=np.float64)
    if hours < len(values):
        output[hours:] = values[:-hours]
    return output


def anchored_level_controls(
    *, anchor: AnchorLevel, base: DataFrame, base_atr: np.ndarray
) -> dict[str, dict[str, Any]]:
    dates = normalize_dates(base["date"])
    delayed_level = shift_numeric(anchor.level, anchor.delay_hours)
    delayed_valid = shift_numeric(anchor.valid.astype(float), anchor.delay_hours) == 1.0
    delayed_open = anchor.source_open.shift(anchor.delay_hours)
    delayed_available = dates - pd.Timedelta(hours=anchor.delay_hours)
    controls: dict[str, dict[str, Any]] = {
        "ordinary_sma_same_anchor_horizon": {
            "level": anchor.sma_control,
            "valid": anchor.valid & np.isfinite(anchor.sma_control),
            "source_available": dates,
            "source_open": anchor.source_open,
        },
        "ordinary_ema_same_anchor_horizon": {
            "level": anchor.ema_control,
            "valid": anchor.valid & np.isfinite(anchor.ema_control),
            "source_available": dates,
            "source_open": anchor.source_open,
        },
        f"time_shifted_anchor_{anchor.delay_hours}h": {
            "level": delayed_level,
            "valid": delayed_valid & np.isfinite(delayed_level) & (delayed_level > 0.0),
            "source_available": delayed_available,
            "source_open": delayed_open,
        },
    }
    for multiplier in PRICE_SHIFTS_ATR:
        shifted = anchor.level + multiplier * base_atr
        controls[f"price_shifted_{multiplier:+g}atr"] = {
            "level": shifted,
            "valid": anchor.valid & np.isfinite(shifted) & (shifted > 0.0),
            "source_available": dates,
            "source_open": anchor.source_open,
        }
    return controls


def anchor_zone_coverage(
    *, level_matrix: np.ndarray, base_atr: np.ndarray, pre_close: np.ndarray
) -> np.ndarray:
    output = np.full(len(level_matrix), np.nan, dtype=np.float64)
    for row_index, levels in enumerate(level_matrix):
        atr = float(base_atr[row_index])
        center = float(pre_close[row_index])
        if not np.isfinite(atr) or atr <= 0.0 or not np.isfinite(center):
            continue
        window_lower = center - 2.0 * atr
        window_upper = center + 2.0 * atr
        finite = levels[np.isfinite(levels)]
        intervals = []
        for level in finite:
            width = max(ZONE_HALF_WIDTH_ATR * atr, abs(float(level)) * 0.0005)
            lower = max(window_lower, float(level) - width)
            upper = min(window_upper, float(level) + width)
            if lower < upper:
                intervals.append((lower, upper))
        if not intervals:
            output[row_index] = 0.0
            continue
        intervals.sort()
        covered = 0.0
        current_lower, current_upper = intervals[0]
        for lower, upper in intervals[1:]:
            if lower <= current_upper:
                current_upper = max(current_upper, upper)
            else:
                covered += current_upper - current_lower
                current_lower, current_upper = lower, upper
        covered += current_upper - current_lower
        output[row_index] = covered / (4.0 * atr)
    return output


def attach_state_and_absolute_outcomes(
    events: DataFrame, state: DataFrame
) -> DataFrame:
    if events.empty:
        return events.copy()
    indexes = events["base_index"].to_numpy(dtype=np.int64)
    output = events.copy()
    for column in (name for name in state if name.startswith("state_")):
        output[column] = numeric_array(state[column])[indexes]
    output["contact_pressure_change_abs"] = pd.to_numeric(
        output["contact_pressure_change"], errors="coerce"
    ).abs()
    for horizon in (1, 4, 24):
        output[f"pressure_change_abs_h{horizon}"] = pd.to_numeric(
            output[f"pressure_change_h{horizon}"], errors="coerce"
        ).abs()
    return output


def attach_anchor_overlap(
    events: DataFrame,
    *,
    level_matrix: np.ndarray,
    family_matrix: np.ndarray,
    source_index: int,
    base_atr: np.ndarray,
) -> DataFrame:
    if events.empty:
        return events.copy()
    output = events.copy()
    indexes = output["base_index"].to_numpy(dtype=np.int64)
    control_level = numeric_array(output["level_price"])
    control_width = numeric_array(output["zone_half_width"])
    levels = level_matrix[indexes]
    atr = base_atr[indexes]
    widths = np.maximum(
        ZONE_HALF_WIDTH_ATR * atr[:, None],
        np.abs(levels) * 0.0005,
    )
    overlap = np.isfinite(levels) & (
        np.abs(levels - control_level[:, None])
        <= widths + control_width[:, None]
    )
    output["overlap_any_anchor_level"] = overlap.any(axis=1)
    output["overlap_anchor_level_count"] = overlap.sum(axis=1).astype(np.int16)
    output["overlap_source_current_level"] = overlap[:, source_index]
    source_family = family_matrix[source_index]
    other_family_count = np.zeros(len(output), dtype=np.int16)
    for family in sorted(set(family_matrix)):
        if family == source_family:
            continue
        other_family_count += overlap[:, family_matrix == family].any(axis=1).astype(
            np.int16
        )
    output["overlap_other_anchor_family_count"] = other_family_count
    return output


def outcome_array(frame: DataFrame, outcome: str) -> np.ndarray:
    if outcome in frame:
        return pd.to_numeric(frame[outcome], errors="coerce").to_numpy(dtype=float)
    raise ValueError(f"G2D event frame lacks requested outcome: {outcome}")


def matched_outcome_rows(
    *,
    actual: DataFrame,
    control: DataFrame,
    pair: str,
    anchor: AnchorLevel,
    anchor_scope: str,
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
        "overlap_anchor_level_count",
        "overlap_other_anchor_family_count",
        *STATE_FEATURES,
        *OUTCOMES,
    }
    for label, frame in (("actual", actual), ("control", control)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{label} G2D event pool lacks columns: {missing}")
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
                independence_hours = SEPARATION_HOURS_BY_RESPONSE_WINDOW[
                    response_window
                ]
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
                        "route_id": "g2d_causal_anchored_vwap_zones",
                        "anchor_family": anchor.family,
                        "level_name": anchor.name,
                        "dispersion_multiplier": anchor.multiplier,
                        "anchor_scope": anchor_scope,
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
                        "match_distance": float(distance),
                        "pre_distance_atr_abs_difference": abs(
                            float(left_row["pre_distance_atr"])
                            - float(right_row["pre_distance_atr"])
                        ),
                        "event_separation_hours": separation,
                        "actual_overlap_anchor_level_count": int(
                            left_row["overlap_anchor_level_count"]
                        ),
                        "control_overlap_anchor_level_count": int(
                            right_row["overlap_anchor_level_count"]
                        ),
                        "actual_overlap_other_anchor_family_count": int(
                            left_row["overlap_other_anchor_family_count"]
                        ),
                        "control_overlap_other_anchor_family_count": int(
                            right_row["overlap_other_anchor_family_count"]
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


def validate_pair_output(
    path: Path, *, pair: str, request_sha256: str
) -> DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "pair",
        "output_schema_version",
        "run_request_sha256",
        "anchor_family",
        "level_name",
        "anchor_scope",
        "control",
        "response_window",
        "actual_event_time",
        "control_event_time",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Existing G2D pair output lacks columns {missing}: {path}")
    if frame.empty or set(frame["pair"].astype(str)) != {pair}:
        raise ValueError(f"Existing G2D pair output has the wrong pair: {path}")
    if set(frame["output_schema_version"].astype(int)) != {OUTPUT_SCHEMA_VERSION}:
        raise ValueError(f"Existing G2D pair output has the wrong schema: {path}")
    if set(frame["run_request_sha256"].astype(str)) != {request_sha256}:
        raise ValueError(f"Existing G2D pair output has the wrong request hash: {path}")
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


def matched_pair_balance(
    frame: DataFrame, state_features: Sequence[str] = STATE_FEATURES
) -> dict[str, Any]:
    if frame.empty:
        return {
            "features_scored": 0,
            "max_absolute_smd": np.nan,
            "median_absolute_smd": np.nan,
        }
    actual = frame[
        [f"actual_state__{feature}" for feature in state_features]
    ].to_numpy(dtype=float)
    control = frame[
        [f"control_state__{feature}" for feature in state_features]
    ].to_numpy(dtype=float)
    valid = np.isfinite(actual) & np.isfinite(control)
    values: list[float] = []
    for position in range(len(state_features)):
        selected = valid[:, position]
        if selected.sum() < 3:
            continue
        left = actual[selected, position]
        right = control[selected, position]
        pooled = np.sqrt((np.var(left) + np.var(right)) / 2.0)
        if not np.isfinite(pooled) or pooled <= 1e-12:
            continue
        values.append(float((np.mean(left) - np.mean(right)) / pooled))
    absolute = np.abs(values)
    return {
        "features_scored": len(values),
        "max_absolute_smd": float(np.max(absolute)) if len(absolute) else np.nan,
        "median_absolute_smd": (
            float(np.median(absolute)) if len(absolute) else np.nan
        ),
    }


def declared_matching_balance(frame: DataFrame) -> dict[str, Any]:
    declarations = frame["matching_state_features"].dropna().astype(str).unique()
    if len(declarations) != 1:
        raise ValueError(
            "A G2D comparison group must contain one declared matching-feature list; "
            f"found {sorted(declarations.tolist())}."
        )
    features = tuple(item for item in declarations[0].split(";") if item)
    if not features:
        raise ValueError("A G2D comparison group declared no matching features.")
    missing = sorted(
        column
        for feature in features
        for column in (f"actual_state__{feature}", f"control_state__{feature}")
        if column not in frame
    )
    if missing:
        raise ValueError(f"G2D comparison group lacks declared state columns: {missing}")
    return matched_pair_balance(frame, features)


def balance_quality(maximum: float, median: float) -> str:
    if np.isfinite(maximum) and np.isfinite(median):
        if maximum <= 0.25 and median <= 0.10:
            return "good"
        if maximum <= MAX_STATE_SMD and median <= MAX_MEDIAN_STATE_SMD:
            return "usable_with_caution"
    return "poor"


def pair_period_results(independent_pairs: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = [
        "route_id",
        "anchor_family",
        "level_name",
        "dispersion_multiplier",
        "anchor_scope",
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
        response_window = str(key[8])
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
                    "anchor_family": key[1],
                    "level_name": key[2],
                    "dispersion_multiplier": key[3],
                    "anchor_scope": key[4],
                    "control": key[5],
                    "pair": key[6],
                    "period": key[7],
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
                    "match_distance_median": float(
                        group["match_distance"].median()
                    ),
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
    pair_period: DataFrame, independent_pairs: DataFrame
) -> DataFrame:
    if pair_period.empty:
        return DataFrame()
    eligible = pair_period.loc[pair_period["pair_period_coverage_eligible"]].copy()
    rows: list[dict[str, Any]] = []
    event_group_columns = [
        "route_id",
        "anchor_family",
        "level_name",
        "dispersion_multiplier",
        "anchor_scope",
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
    group_columns = [
        "route_id",
        "anchor_family",
        "level_name",
        "dispersion_multiplier",
        "anchor_scope",
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
        coins = sorted(group["pair"].astype(str).unique())
        outcome = str(key[8])
        event_key = tuple(key[:8])
        cache_key = (*event_key, *coins)
        if cache_key not in matched_event_cache:
            if event_key not in event_groups:
                raise ValueError(
                    f"G2D cohort summary cannot resolve event group {event_key}."
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
            outcome_events[f"delta__{outcome}"], errors="coerce"
        )
        coin_delta = pd.to_numeric(group["delta_mean"], errors="coerce")
        event_count = int(group["independent_event_pairs"].sum())
        coverage = (
            len(coins) >= MIN_COHORT_PERIOD_COINS
            and event_count >= MIN_COHORT_PERIOD_EPISODES
        )
        usable_balance = (
            balance["max_absolute_smd"] <= MAX_STATE_SMD
            and balance["median_absolute_smd"] <= MAX_MEDIAN_STATE_SMD
        )
        rows.append(
            {
                "route_id": key[0],
                "anchor_family": key[1],
                "level_name": key[2],
                "dispersion_multiplier": key[3],
                "anchor_scope": key[4],
                "control": key[5],
                "period": key[6],
                "response_window": key[7],
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


def leave_one_coin_out_results(pair_period: DataFrame) -> DataFrame:
    if pair_period.empty:
        return DataFrame()
    eligible = pair_period.loc[pair_period["pair_period_coverage_eligible"]].copy()
    rows: list[dict[str, Any]] = []
    group_columns = [
        "route_id",
        "anchor_family",
        "level_name",
        "dispersion_multiplier",
        "anchor_scope",
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
            retained_coin_count = len(available_pairs) - 1
            event_count = int(total_events - event_counts[omitted_index])
            rows.append(
                {
                    "route_id": key[0],
                    "anchor_family": key[1],
                    "level_name": key[2],
                    "dispersion_multiplier": key[3],
                    "anchor_scope": key[4],
                    "control": key[5],
                    "period": key[6],
                    "response_window": key[7],
                    "outcome": key[8],
                    "omitted_pair": omitted,
                    "remaining_coin_count": retained_coin_count,
                    "remaining_independent_event_pairs": event_count,
                    "coverage_gate_passed": retained_coin_count
                    >= MIN_COHORT_PERIOD_COINS
                    and event_count >= MIN_COHORT_PERIOD_EPISODES,
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
    expected_families = {
        "utc_daily_reset",
        "utc_weekly_reset",
        "confirmed_swing_high",
        "confirmed_swing_low",
    }
    expected_control_prefixes = {
        "ordinary_sma_same_anchor_horizon",
        "ordinary_ema_same_anchor_horizon",
        "time_shifted_anchor",
        "price_shifted",
        "random_eligible_zone",
        "same_state_no_level",
        "same_density_coverage_no_level",
    }
    observed_controls = set(matches["control"].dropna().astype(str))
    missing_control_types = sorted(
        prefix
        for prefix in expected_control_prefixes
        if not any(control.startswith(prefix) for control in observed_controls)
    )
    observed_families = set(matches["anchor_family"].dropna().astype(str))
    response_counts = independent["response_window"].value_counts().to_dict()
    outcome_counts = {
        outcome: int(response_counts.get(response_window, 0))
        for response_window, outcomes in OUTCOMES_BY_RESPONSE_WINDOW.items()
        for outcome in outcomes
    }
    return {
        "created_at_utc": utc_now(),
        "passed": (
            not failures
            and future_violations == 0
            and source_open_violations == 0
            and not independent.empty
            and not expected_families.difference(observed_families)
        ),
        "technical_smoke_not_evidence": technical_smoke,
        "pairs_requested": len(results),
        "pair_failures": failures,
        "matched_comparisons_before_overlap_purge": len(matches),
        "independent_comparisons_after_horizon_purge": len(independent),
        "independent_comparisons_by_response_window": {
            str(key): int(value) for key, value in response_counts.items()
        },
        "independent_comparisons_available_per_outcome": outcome_counts,
        "future_source_availability_violations": future_violations,
        "source_open_at_or_after_contact_violations": source_open_violations,
        "expected_anchor_families": sorted(expected_families),
        "observed_anchor_families": sorted(observed_families),
        "missing_anchor_families": sorted(
            expected_families.difference(observed_families)
        ),
        "observed_controls": sorted(observed_controls),
        "missing_control_types": missing_control_types,
        "missing_controls_are_coverage_limits_not_automatic_integrity_failures": True,
        "swing_confirmation_left_bars": SWING_LEFT_BARS,
        "swing_confirmation_right_bars": SWING_RIGHT_BARS,
        "current_contact_candle_in_anchor_calculation": False,
        "direction_prediction": False,
        "profit_optimization": False,
    }


def pair_stem(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")


if __name__ == "__main__":
    raise SystemExit(main())

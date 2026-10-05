from __future__ import annotations

# Importing pandas/numpy only after the thread environment is fixed is intentional.
# ruff: noqa: E402
import os


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pandas import DataFrame, Series
from scipy.stats import t as student_t


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    DEFAULT_MANIFEST,
    LARGE_ARTIFACT_ROOT,
    OUTPUT_ROOT,
    AtlasTask,
    atlas_event_path,
    atlas_metadata_path,
    atlas_summary_path,
    atomic_write_json,
    atomic_write_parquet,
    bool_array,
    cooldown_start_mask,
    episode_start_mask,
    level_cache_path,
    level_specs,
    load_manifest,
    normalize_dates,
    numeric_array,
    parse_choice_csv,
    parse_families,
    prepare_base_market_frame,
    resolved_level_values,
    resolved_native_width,
    select_pairs,
    select_timeframes,
    sha256_file,
    validate_cache_metadata,
    zone_half_width,
)


REVIEW_DIR = OUTPUT_ROOT / "generation0_review"
LARGE_REVIEW_DIR = LARGE_ARTIFACT_ROOT / "generation0_review"
COHORT_BY_PERIOD_PATH = LARGE_REVIEW_DIR / "g0b_cohort_by_period.parquet"
REPEATABILITY_PATH = LARGE_REVIEW_DIR / "g0b_repeatability.parquet"
CALIBRATION_CELL_DIR = LARGE_REVIEW_DIR / "attribute_calibration_cells"
CALIBRATION_REPEATABILITY_PATH = LARGE_REVIEW_DIR / "g0b_attribute_calibration.parquet"
DEFAULT_CACHE_FAMILIES = ("core", "generic")
DEFAULT_LEVEL_BATCHES = ("g0b1", "g0b2")
DEFAULT_ZONE_METHODS = (
    "tight_base_atr",
    "standard_base_atr",
    "wide_base_atr",
    "native_width",
)
DEFAULT_CONTROLS = (
    "actual",
    "stale_72h",
    "stale_168h",
    "price_shift",
    "near_miss",
    "matched_random_time",
)

TIME_TO_COLUMNS = (
    "time_to_abs_0_5atr",
    "time_to_abs_1_0atr",
    "time_to_away_0_5atr",
    "time_to_away_1_0atr",
    "time_to_through_0_5atr",
    "time_to_through_1_0atr",
)

CALIBRATION_CONTACT_TARGETS = {
    "contact_range_ratio": "contact_range_vs_precontact",
    "contact_volume_ratio": "contact_volume_vs_precontact",
    "contact_pressure_change": "contact_pressure_change",
}

CALIBRATION_HORIZON_TARGETS = {
    "abs_excursion_atr": "absolute_price_excursion_atr",
    "away_excursion_atr": "away_from_approach_excursion_atr",
    "through_excursion_atr": "through_level_excursion_atr",
    "close_abs_displacement_atr": "close_distance_from_level_atr",
    "range_ratio": "future_range_vs_precontact",
    "volume_ratio": "future_volume_vs_precontact",
    "pressure_change": "future_pressure_change",
    "dwell_fraction": "zone_dwell_fraction",
    "crossings": "level_crossing_count",
}

CALIBRATION_CELL_COLUMNS = (
    "pair",
    "source_timeframe",
    "batch",
    "level_family",
    "level_name",
    "representation",
    "zone_method",
    "period",
    "approach_state",
)

KEY_COLUMNS = (
    "pair",
    "source_timeframe",
    "batch",
    "level_family",
    "level_name",
    "representation",
    "zone_method",
    "period",
    "approach_state",
    "horizon_hours",
)

METRIC_COLUMNS = (
    "contact_range_ratio_median",
    "contact_volume_ratio_median",
    "contact_pressure_change_median",
    "contact_close_distance_atr_median",
    "abs_excursion_mean",
    "abs_excursion_median",
    "away_excursion_mean",
    "away_excursion_median",
    "through_excursion_mean",
    "through_excursion_median",
    "close_abs_displacement_mean",
    "close_abs_displacement_median",
    "range_ratio_mean",
    "range_ratio_median",
    "volume_ratio_mean",
    "volume_ratio_median",
    "pressure_change_mean",
    "pressure_change_median",
    "dwell_fraction_mean",
    "dwell_fraction_median",
    "crossings_mean",
    "crossings_median",
)

PRIMARY_DELTA_COLUMNS = (
    "contact_range_ratio_median_delta",
    "contact_volume_ratio_median_delta",
    "contact_pressure_change_median_delta",
    "abs_excursion_mean_delta",
    "away_excursion_mean_delta",
    "through_excursion_mean_delta",
    "close_abs_displacement_mean_delta",
    "abs_excursion_median_delta",
    "range_ratio_mean_delta",
    "volume_ratio_mean_delta",
    "pressure_change_mean_delta",
    "dwell_fraction_mean_delta",
    "crossings_mean_delta",
)


def configure_review_scope(name: str) -> None:
    global REVIEW_DIR, LARGE_REVIEW_DIR, COHORT_BY_PERIOD_PATH, REPEATABILITY_PATH
    global CALIBRATION_CELL_DIR, CALIBRATION_REPEATABILITY_PATH
    normalized = name.strip().lower()
    if not normalized or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in normalized
    ):
        raise ValueError(f"Invalid review name: {name!r}")
    directory_suffix = "" if normalized == "g0b" else f"_{normalized}"
    REVIEW_DIR = OUTPUT_ROOT / f"generation0_review{directory_suffix}"
    LARGE_REVIEW_DIR = LARGE_ARTIFACT_ROOT / f"generation0_review{directory_suffix}"
    COHORT_BY_PERIOD_PATH = LARGE_REVIEW_DIR / "g0b_cohort_by_period.parquet"
    REPEATABILITY_PATH = LARGE_REVIEW_DIR / "g0b_repeatability.parquet"
    CALIBRATION_CELL_DIR = LARGE_REVIEW_DIR / "attribute_calibration_cells"
    CALIBRATION_REPEATABILITY_PATH = LARGE_REVIEW_DIR / "g0b_attribute_calibration.parquet"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and pool Generation 0 reaction-zone summaries without converting "
            "them into directional or profit targets."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--timeframes", default="all")
    parser.add_argument("--cache-families", default="core,generic")
    parser.add_argument("--level-batches", default="g0b1,g0b2")
    parser.add_argument(
        "--zones",
        default="tight_base_atr,standard_base_atr,wide_base_atr,native_width",
    )
    parser.add_argument(
        "--controls",
        default="actual,stale_72h,stale_168h,price_shift,near_miss,matched_random_time",
    )
    parser.add_argument("--review-name", default="g0b")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "command",
        choices=("audit", "coverage", "compare", "calibration", "all"),
        nargs="?",
        default="all",
    )
    args = parser.parse_args(argv)
    configure_review_scope(args.review_name)
    manifest = load_manifest(args.manifest)
    pairs = select_pairs(manifest, args.pairs)
    timeframes = select_timeframes(manifest, args.timeframes)
    cache_families = parse_families(args.cache_families)
    level_batches = parse_choice_csv(args.level_batches, {"g0b1", "g0b2", "g0b3"}, "level batches")
    zone_methods = parse_choice_csv(
        args.zones,
        {"tight_base_atr", "standard_base_atr", "wide_base_atr", "native_width"},
        "zone methods",
    )
    controls = parse_choice_csv(
        args.controls,
        {
            "actual",
            "stale_72h",
            "stale_168h",
            "price_shift",
            "near_miss",
            "matched_random_time",
        },
        "controls",
    )
    tasks = task_matrix(
        pairs,
        timeframes,
        args.manifest,
        cache_families=cache_families,
        level_batches=level_batches,
        zone_methods=zone_methods,
        controls=controls,
    )

    if args.command == "coverage":
        coverage = build_level_coverage(tasks, manifest, manifest_path=args.manifest)
        print(json.dumps(coverage, indent=2, sort_keys=True))
        return 0

    integrity = audit_outputs(
        tasks,
        manifest_path=args.manifest,
        allow_incomplete=args.allow_incomplete,
    )
    if args.command == "audit":
        print(json.dumps(integrity["summary"], indent=2, sort_keys=True))
        return 1 if integrity["summary"]["invalid_tasks"] else 0
    if integrity["summary"]["invalid_tasks"] and not args.allow_incomplete:
        print(json.dumps(integrity["summary"], indent=2, sort_keys=True))
        return 1

    if args.command == "calibration":
        calibration = build_attribute_calibration(tasks, manifest)
        print(json.dumps(calibration, indent=2, sort_keys=True))
        return 0

    coverage = (
        build_level_coverage(tasks, manifest, manifest_path=args.manifest)
        if args.command == "all"
        else {}
    )
    comparison = build_comparisons(tasks, manifest)
    print(
        json.dumps(
            {
                **integrity["summary"],
                **comparison,
                **coverage,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def task_matrix(
    pairs: Sequence[str],
    timeframes: Sequence[str],
    manifest_path: Path,
    *,
    cache_families: tuple[str, ...] = DEFAULT_CACHE_FAMILIES,
    level_batches: tuple[str, ...] = DEFAULT_LEVEL_BATCHES,
    zone_methods: tuple[str, ...] = DEFAULT_ZONE_METHODS,
    controls: tuple[str, ...] = DEFAULT_CONTROLS,
) -> list[AtlasTask]:
    return [
        AtlasTask(
            pair=pair,
            timeframe=timeframe,
            cache_families=cache_families,
            level_batches=level_batches,
            zone_methods=zone_methods,
            controls=controls,
            manifest_path=str(manifest_path),
            overwrite=False,
        )
        for pair in pairs
        for timeframe in timeframes
    ]


def audit_outputs(  # noqa: C901 - every integrity failure remains explicit
    tasks: Sequence[AtlasTask], *, manifest_path: Path, allow_incomplete: bool
) -> dict[str, Any]:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    manifest_hash = sha256_file(manifest_path)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        event_path = atlas_event_path(task)
        summary_path = atlas_summary_path(task)
        metadata_path = atlas_metadata_path(task)
        problems: list[str] = []
        if not event_path.is_file():
            problems.append("missing_event_output")
        if not summary_path.is_file():
            problems.append("missing_detailed_summary")
        if not metadata_path.is_file():
            problems.append("missing_metadata")
        metadata: dict[str, Any] = {}
        if metadata_path.is_file():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            if metadata.get("manifest_sha256") != manifest_hash:
                problems.append("manifest_hash_mismatch")
            if metadata.get("direction_prediction") is not False:
                problems.append("direction_boundary_missing")
            if metadata.get("profit_optimization") is not False:
                problems.append("profit_boundary_missing")
            if tuple(metadata.get("controls", ())) != task.controls:
                problems.append("control_surface_mismatch")
            if tuple(metadata.get("zone_methods", ())) != task.zone_methods:
                problems.append("zone_surface_mismatch")
            if Path(metadata.get("event_output", "")) != event_path:
                problems.append("event_path_mismatch")
            if Path(metadata.get("summary_output", "")) != summary_path:
                problems.append("summary_path_mismatch")

        event_rows = parquet_rows(event_path) if event_path.is_file() else None
        summary_rows = parquet_rows(summary_path) if summary_path.is_file() else None
        if metadata:
            if event_rows != metadata.get("stored_event_rows"):
                problems.append("event_row_count_mismatch")
            if summary_rows != metadata.get("summary_rows"):
                problems.append("summary_row_count_mismatch")
        rows.append(
            {
                "pair": task.pair,
                "timeframe": task.timeframe,
                "event_path": str(event_path),
                "summary_path": str(summary_path),
                "metadata_path": str(metadata_path),
                "event_rows": event_rows,
                "summary_rows": summary_rows,
                "valid": not problems,
                "problems": problems,
            }
        )
    invalid = [row for row in rows if not row["valid"]]
    report = {
        "created_at_utc": utc_now(),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "allow_incomplete": allow_incomplete,
        "summary": {
            "expected_tasks": len(tasks),
            "valid_tasks": len(tasks) - len(invalid),
            "invalid_tasks": len(invalid),
            "stored_event_rows": int(sum(row["event_rows"] or 0 for row in rows if row["valid"])),
            "detailed_summary_rows": int(
                sum(row["summary_rows"] or 0 for row in rows if row["valid"])
            ),
        },
        "tasks": rows,
    }
    atomic_write_json(report, REVIEW_DIR / "g0b_integrity.json")
    return report


def build_level_coverage(  # noqa: C901 - level, zone, and union exposure stay auditable
    tasks: Sequence[AtlasTask],
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
) -> dict[str, Any]:
    """Measure level exposure before interpreting any reaction comparison."""
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    completed_tasks = 0
    max_horizon = max(int(value) for value in manifest["reaction_definition"]["horizons_hours"])
    frozen_periods = tuple(period["id"] for period in manifest["data"]["chronological_periods"])

    for task in tasks:
        cache_path = level_cache_path(task.pair, task.timeframe, task.cache_families)
        if not cache_path.is_file():
            continue
        validate_cache_metadata(cache_path, manifest_path)
        base = prepare_base_market_frame(task.pair, manifest).sort_values("date")
        cache = pd.read_parquet(cache_path).sort_values("available_at").reset_index(drop=True)
        cache["available_at"] = normalize_dates(cache["available_at"])
        cache["source_open"] = normalize_dates(cache["source_open"])
        merged = pd.merge_asof(
            base,
            cache,
            left_on="date",
            right_on="available_at",
            direction="backward",
            allow_exact_matches=True,
        )
        available = merged["available_at"].notna()
        if (merged.loc[available, "available_at"] > merged.loc[available, "date"]).any():
            raise AssertionError("Causal as-of merge admitted a future source row.")

        specs = level_specs(cache, task.level_batches)
        base_atr = numeric_array(merged["base_atr"])
        high = numeric_array(merged["high"])
        low = numeric_array(merged["low"])
        period_masks = coverage_period_masks(merged, frozen_periods)
        family_unions: dict[tuple[str, str, str], dict[str, Any]] = {}
        all_unions: dict[str, dict[str, Any]] = {}

        for spec in specs:
            level = resolved_level_values(merged, spec, task.timeframe)
            valid = np.isfinite(level) & (level > 0.0)
            if spec.active_columns:
                active = np.zeros(len(merged), dtype=bool)
                for column in spec.active_columns:
                    if column in merged.columns:
                        active |= bool_array(merged[column])
                valid &= active
            valid &= np.isfinite(base_atr) & (base_atr > 0.0)
            native_width = resolved_native_width(merged, spec)

            for zone_method in task.zone_methods:
                half_width = zone_half_width(zone_method, level, base_atr, native_width)
                eligible = valid & np.isfinite(half_width) & (half_width > 0.0)
                contact = eligible & (high >= level - half_width) & (low <= level + half_width)
                starts = episode_start_mask(contact, level, half_width, cooldown=6)
                if max_horizon:
                    starts[max(len(starts) - max_horizon, 0) :] = False
                width_atr = np.divide(
                    half_width,
                    base_atr,
                    out=np.full(len(merged), np.nan, dtype=np.float64),
                    where=eligible,
                )
                for period, period_mask in period_masks:
                    rows.append(
                        coverage_row(
                            task=task,
                            scope="level_spec",
                            batch=spec.batch,
                            level_family=spec.family,
                            level_name=spec.name,
                            representation=spec.representation,
                            zone_method=zone_method,
                            period=period,
                            period_mask=period_mask,
                            eligible=eligible,
                            contact=contact,
                            starts=starts,
                            available_count=eligible.astype(np.int16),
                            contact_count=contact.astype(np.int16),
                            member_level_specs=1,
                            width_atr=width_atr,
                        )
                    )

                family_key = (spec.batch, spec.family, zone_method)
                family = family_unions.setdefault(
                    family_key,
                    coverage_accumulator(len(merged)),
                )
                family["available_count"] += eligible.astype(np.int16)
                family["contact_count"] += contact.astype(np.int16)
                family["members"].add((spec.name, spec.representation))

                all_levels = all_unions.setdefault(
                    zone_method,
                    coverage_accumulator(len(merged)),
                )
                all_levels["available_count"] += eligible.astype(np.int16)
                all_levels["contact_count"] += contact.astype(np.int16)
                all_levels["members"].add((spec.batch, spec.family, spec.name, spec.representation))

        for (batch, family_name, zone_method), accumulator in family_unions.items():
            rows.extend(
                union_coverage_rows(
                    task=task,
                    scope="family_union",
                    batch=batch,
                    level_family=family_name,
                    level_name="family_union",
                    zone_method=zone_method,
                    period_masks=period_masks,
                    accumulator=accumulator,
                    max_horizon=max_horizon,
                )
            )
        for zone_method, accumulator in all_unions.items():
            rows.extend(
                union_coverage_rows(
                    task=task,
                    scope="all_levels_union",
                    batch="g0b1+g0b2",
                    level_family="all_level_families",
                    level_name="all_levels_union",
                    zone_method=zone_method,
                    period_masks=period_masks,
                    accumulator=accumulator,
                    max_horizon=max_horizon,
                )
            )
        completed_tasks += 1

    coverage = DataFrame(rows)
    coverage_path = REVIEW_DIR / "g0b_level_coverage.parquet"
    atomic_write_parquet(coverage, coverage_path)
    record = {
        "created_at_utc": utc_now(),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "completed_tasks": completed_tasks,
        "coverage_rows": len(coverage),
        "direction_prediction": False,
        "profit_optimization": False,
        "output": str(coverage_path),
    }
    atomic_write_json(record, REVIEW_DIR / "g0b_level_coverage_record.json")
    return record


def coverage_period_masks(
    frame: DataFrame, frozen_periods: Sequence[str]
) -> list[tuple[str, np.ndarray]]:
    period = frame["period"].astype(str)
    masks = [
        (
            "all_frozen_periods",
            period.isin(frozen_periods).to_numpy(dtype=bool),
        )
    ]
    masks.extend(
        (period_id, period.eq(period_id).to_numpy(dtype=bool)) for period_id in frozen_periods
    )
    return masks


def coverage_accumulator(rows: int) -> dict[str, Any]:
    return {
        "available_count": np.zeros(rows, dtype=np.int16),
        "contact_count": np.zeros(rows, dtype=np.int16),
        "members": set(),
    }


def union_coverage_rows(
    *,
    task: AtlasTask,
    scope: str,
    batch: str,
    level_family: str,
    level_name: str,
    zone_method: str,
    period_masks: Sequence[tuple[str, np.ndarray]],
    accumulator: dict[str, Any],
    max_horizon: int,
) -> list[dict[str, Any]]:
    available_count = accumulator["available_count"]
    contact_count = accumulator["contact_count"]
    eligible = available_count > 0
    contact = contact_count > 0
    starts = cooldown_start_mask(contact, cooldown=6)
    if max_horizon:
        starts[max(len(starts) - max_horizon, 0) :] = False
    return [
        coverage_row(
            task=task,
            scope=scope,
            batch=batch,
            level_family=level_family,
            level_name=level_name,
            representation="mixed",
            zone_method=zone_method,
            period=period,
            period_mask=period_mask,
            eligible=eligible,
            contact=contact,
            starts=starts,
            available_count=available_count,
            contact_count=contact_count,
            member_level_specs=len(accumulator["members"]),
            width_atr=None,
        )
        for period, period_mask in period_masks
    ]


def coverage_row(
    *,
    task: AtlasTask,
    scope: str,
    batch: str,
    level_family: str,
    level_name: str,
    representation: str,
    zone_method: str,
    period: str,
    period_mask: np.ndarray,
    eligible: np.ndarray,
    contact: np.ndarray,
    starts: np.ndarray,
    available_count: np.ndarray,
    contact_count: np.ndarray,
    member_level_specs: int,
    width_atr: np.ndarray | None,
) -> dict[str, Any]:
    total_bars = int(period_mask.sum())
    eligible_mask = period_mask & eligible
    contact_mask = period_mask & contact
    eligible_bars = int(eligible_mask.sum())
    contact_bars = int(contact_mask.sum())
    episodes = int((period_mask & starts).sum())
    width_values = (
        width_atr[eligible_mask] if width_atr is not None else np.empty(0, dtype=np.float64)
    )
    width_values = width_values[np.isfinite(width_values)]
    available_slots = available_count[period_mask]
    contacting_slots = contact_count[contact_mask]
    return {
        "scope": scope,
        "pair": task.pair,
        "source_timeframe": task.timeframe,
        "batch": batch,
        "level_family": level_family,
        "level_name": level_name,
        "representation": representation,
        "zone_method": zone_method,
        "period": period,
        "member_level_specs": member_level_specs,
        "total_base_bars": total_bars,
        "eligible_level_bars": eligible_bars,
        "eligible_level_fraction": safe_ratio(eligible_bars, total_bars),
        "contact_bars": contact_bars,
        "contact_bar_fraction_of_all": safe_ratio(contact_bars, total_bars),
        "contact_bar_fraction_of_eligible": safe_ratio(contact_bars, eligible_bars),
        "episodes": episodes,
        "episodes_per_1000_base_bars": 1000.0 * safe_ratio(episodes, total_bars),
        "episodes_per_1000_eligible_bars": 1000.0 * safe_ratio(episodes, eligible_bars),
        "median_zone_half_width_atr": (
            float(np.median(width_values)) if len(width_values) else np.nan
        ),
        "mean_available_level_slots_per_base_bar": (
            float(available_slots.mean()) if len(available_slots) else np.nan
        ),
        "max_available_level_slots": (int(available_slots.max()) if len(available_slots) else 0),
        "mean_contacting_level_slots_per_contact_bar": (
            float(contacting_slots.mean()) if len(contacting_slots) else np.nan
        ),
        "max_contacting_level_slots": (int(contacting_slots.max()) if len(contacting_slots) else 0),
        "direction_prediction": False,
        "profit_optimization": False,
    }


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else np.nan


def build_cohort_coverage(tasks: Sequence[AtlasTask], manifest: dict[str, Any]) -> DataFrame:
    coverage_path = REVIEW_DIR / "g0b_level_coverage.parquet"
    if not coverage_path.is_file():
        return DataFrame()
    selected_pairs = {task.pair for task in tasks}
    selected_timeframes = {task.timeframe for task in tasks}
    repeatability_periods = {
        period["id"]
        for period in manifest["data"]["chronological_periods"]
        if period["role"] != "diagnostic_only_not_confirmation"
    }
    coverage = pd.read_parquet(coverage_path)
    coverage = coverage.loc[
        coverage["scope"].eq("level_spec")
        & coverage["pair"].isin(selected_pairs)
        & coverage["source_timeframe"].isin(selected_timeframes)
        & coverage["period"].isin(repeatability_periods)
    ].copy()
    if coverage.empty:
        return DataFrame()
    expanded = add_cohort_lenses(coverage, manifest)
    keys = [
        "cohort",
        "source_timeframe",
        "batch",
        "level_family",
        "level_name",
        "representation",
        "zone_method",
    ]
    grouped = expanded.groupby(keys, sort=False, dropna=False, observed=True)
    result = grouped.agg(
        coverage_cells=("pair", "size"),
        coverage_pair_count=("pair", "nunique"),
        coverage_period_count=("period", "nunique"),
        total_base_bars=("total_base_bars", "sum"),
        eligible_level_bars=("eligible_level_bars", "sum"),
        contact_bars=("contact_bars", "sum"),
        contact_episodes=("episodes", "sum"),
        median_cell_contact_bar_fraction=("contact_bar_fraction_of_all", "median"),
        maximum_cell_contact_bar_fraction=("contact_bar_fraction_of_all", "max"),
    ).reset_index()
    result["eligible_level_fraction"] = result["eligible_level_bars"].div(
        result["total_base_bars"].replace(0.0, np.nan)
    )
    result["contact_bar_fraction_of_all"] = result["contact_bars"].div(
        result["total_base_bars"].replace(0.0, np.nan)
    )
    result["contact_bar_fraction_of_eligible"] = result["contact_bars"].div(
        result["eligible_level_bars"].replace(0.0, np.nan)
    )
    result["episodes_per_1000_base_bars"] = 1000.0 * result["contact_episodes"].div(
        result["total_base_bars"].replace(0.0, np.nan)
    )

    pair_keys = [*keys, "pair"]
    per_pair = (
        expanded.groupby(pair_keys, sort=False, dropna=False, observed=True)["contact_bars"]
        .sum()
        .rename("pair_contact_bars")
        .reset_index()
    )
    dominance = (
        per_pair.groupby(keys, sort=False, dropna=False, observed=True)["pair_contact_bars"]
        .max()
        .rename("largest_pair_contact_bars")
        .reset_index()
    )
    result = result.merge(dominance, on=keys, how="left", validate="one_to_one")
    result["largest_pair_contact_fraction"] = result["largest_pair_contact_bars"].div(
        result["contact_bars"].replace(0.0, np.nan)
    )
    return result


def build_timing_review(tasks: Sequence[AtlasTask], manifest: dict[str, Any]) -> dict[str, int]:
    pair_dir = LARGE_REVIEW_DIR / "pair_timing"
    pair_dir.mkdir(parents=True, exist_ok=True)
    repeatability_periods = {
        period["id"]
        for period in manifest["data"]["chronological_periods"]
        if period["role"] != "diagnostic_only_not_confirmation"
    }
    pair_frames: list[DataFrame] = []
    completed_tasks = 0
    pair_rows = 0
    for task in tasks:
        event_path = atlas_event_path(task)
        if not event_path.is_file():
            continue
        columns = [
            "pair",
            "source_timeframe",
            "batch",
            "level_family",
            "level_name",
            "representation",
            "control",
            "zone_method",
            "period",
            "approach_state",
            *TIME_TO_COLUMNS,
        ]
        available_columns = set(pq.ParquetFile(event_path).schema.names)
        if not set(columns).issubset(available_columns):
            completed_tasks += 1
            continue
        events = pd.read_parquet(event_path, columns=columns)
        events = events.loc[
            events["period"].isin(repeatability_periods)
            & events["control"].isin(("actual", "matched_random_time"))
        ].copy()
        if events.empty:
            completed_tasks += 1
            continue
        all_approaches = events.copy()
        all_approaches["approach_state"] = "all_approaches"
        events = pd.concat([events, all_approaches], ignore_index=True)
        pair_comparison = timing_comparison_from_events(events)
        output_path = pair_dir / f"{event_path.stem}-timing.parquet"
        atomic_write_parquet(pair_comparison, output_path)
        pair_frames.append(pair_comparison)
        pair_rows += len(pair_comparison)
        completed_tasks += 1

    pair_timing = pd.concat(pair_frames, ignore_index=True) if pair_frames else DataFrame()
    repeatability = (
        aggregate_timing_comparisons(add_cohort_lenses(pair_timing, manifest))
        if not pair_timing.empty
        else DataFrame()
    )
    atomic_write_parquet(repeatability, REVIEW_DIR / "g0b_timing_repeatability.parquet")
    return {
        "timing_completed_tasks": completed_tasks,
        "pair_timing_comparison_rows": pair_rows,
        "timing_repeatability_rows": len(repeatability),
    }


def timing_comparison_from_events(events: DataFrame) -> DataFrame:
    key_columns = [
        "pair",
        "source_timeframe",
        "batch",
        "level_family",
        "level_name",
        "representation",
        "zone_method",
        "period",
        "approach_state",
    ]
    group_columns = [*key_columns, "control"]
    grouped = events.groupby(group_columns, sort=False, dropna=False, observed=True)
    counts = grouped.size().rename("event_count")
    frames: list[DataFrame] = []
    for column in TIME_TO_COLUMNS:
        reached = events.assign(_reached=events[column].notna().astype(float))
        reach_rate = (
            reached.groupby(group_columns, sort=False, dropna=False, observed=True)["_reached"]
            .mean()
            .rename("reach_rate")
        )
        median = grouped[column].median().rename("median_time_hours_when_reached")
        summary = pd.concat([counts, reach_rate, median], axis=1).reset_index()
        summary["reaction_threshold"] = column.removeprefix("time_to_")
        frames.append(summary)
    long = pd.concat(frames, ignore_index=True)
    comparison_keys = [*key_columns, "reaction_threshold"]
    actual = (
        long.loc[long["control"] == "actual"]
        .drop(columns="control")
        .rename(
            columns={
                "event_count": "actual_n",
                "reach_rate": "actual_reach_rate",
                "median_time_hours_when_reached": "actual_median_time_hours_when_reached",
            }
        )
    )
    matched = (
        long.loc[long["control"] == "matched_random_time"]
        .drop(columns="control")
        .rename(
            columns={
                "event_count": "control_n",
                "reach_rate": "control_reach_rate",
                "median_time_hours_when_reached": "control_median_time_hours_when_reached",
            }
        )
    )
    comparison = actual.merge(
        matched,
        on=comparison_keys,
        how="left",
        validate="one_to_one",
    )
    comparison["reach_rate_delta"] = (
        comparison["actual_reach_rate"] - comparison["control_reach_rate"]
    )
    comparison["conditional_median_time_delta_hours"] = (
        comparison["actual_median_time_hours_when_reached"]
        - comparison["control_median_time_hours_when_reached"]
    )
    return comparison


def aggregate_timing_comparisons(frame: DataFrame) -> DataFrame:
    valid = frame.loc[frame["actual_n"].notna() & frame["control_n"].notna()].copy()
    if valid.empty:
        return DataFrame()
    keys = [
        "cohort",
        "source_timeframe",
        "batch",
        "level_family",
        "level_name",
        "representation",
        "zone_method",
        "approach_state",
        "reaction_threshold",
    ]
    valid["actual_reached"] = valid["actual_reach_rate"] * valid["actual_n"]
    valid["control_reached"] = valid["control_reach_rate"] * valid["control_n"]
    valid["reach_rate_delta_weighted"] = valid["reach_rate_delta"] * valid["actual_n"]
    valid["reach_rate_delta_positive"] = (valid["reach_rate_delta"] > 0.0).astype(float)
    valid["faster_when_reached"] = (valid["conditional_median_time_delta_hours"] < 0.0).astype(
        float
    )
    grouped = valid.groupby(keys, sort=False, dropna=False, observed=True)
    result = grouped.agg(
        cell_count=("pair", "size"),
        pair_count=("pair", "nunique"),
        period_count=("period", "nunique"),
        actual_events=("actual_n", "sum"),
        control_events=("control_n", "sum"),
        actual_reached=("actual_reached", "sum"),
        control_reached=("control_reached", "sum"),
        reach_rate_delta_median=("reach_rate_delta", "median"),
        reach_rate_delta_positive_fraction=("reach_rate_delta_positive", "mean"),
        conditional_median_time_delta_hours_median=(
            "conditional_median_time_delta_hours",
            "median",
        ),
        faster_when_reached_fraction=("faster_when_reached", "mean"),
        reach_rate_delta_weighted_sum=("reach_rate_delta_weighted", "sum"),
    ).reset_index()
    result["actual_reach_rate"] = result["actual_reached"].div(
        result["actual_events"].replace(0.0, np.nan)
    )
    result["control_reach_rate"] = result["control_reached"].div(
        result["control_events"].replace(0.0, np.nan)
    )
    result["reach_rate_delta_weighted_mean"] = result["reach_rate_delta_weighted_sum"].div(
        result["actual_events"].replace(0.0, np.nan)
    )
    result.drop(columns="reach_rate_delta_weighted_sum", inplace=True)
    return result


def calibration_target_columns(
    available_columns: set[str], horizons: Sequence[int]
) -> list[tuple[str, str, int]]:
    targets = [
        (column, name, 0)
        for column, name in CALIBRATION_CONTACT_TARGETS.items()
        if column in available_columns
    ]
    for horizon in horizons:
        for prefix, name in CALIBRATION_HORIZON_TARGETS.items():
            column = f"{prefix}_h{horizon}"
            if column in available_columns:
                targets.append((column, name, int(horizon)))
    return targets


def calibratable_attribute_columns(available_columns: set[str]) -> list[str]:
    attributes: list[str] = []
    if "level_score" in available_columns:
        attributes.append("level_score")
    for column in sorted(available_columns):
        if not column.startswith("attr_"):
            continue
        if any(
            token in column
            for token in (
                "value_area_width_pct",
                "pivot_count",
                "width_atr",
                "confirmation_tier",
            )
        ):
            attributes.append(column)
    return attributes


def deterministic_state_shuffle(
    frame: DataFrame,
    *,
    value_column: str,
    strata_columns: Sequence[str],
) -> np.ndarray:
    """Permute one attribute inside comparable market-state cells deterministically."""
    if frame.empty:
        return np.array([], dtype=float)
    token_columns = [column for column in ("event_time", "base_index") if column in frame]
    if not token_columns:
        token = np.arange(len(frame), dtype=np.uint64)
    else:
        token = pd.util.hash_pandas_object(
            frame[token_columns], index=False
        ).to_numpy(dtype=np.uint64)
    salt = np.uint64(
        int(
            pd.util.hash_pandas_object(
                pd.Series([value_column], dtype="string"), index=False
            ).iloc[0]
        )
    )
    source_key = token ^ salt
    destination_key = np.multiply(
        token ^ np.uint64(0xD1B54A32D192ED03),
        np.uint64(0x9E3779B97F4A7C15),
        dtype=np.uint64,
    ) ^ (salt >> np.uint64(1))

    source = frame[[*strata_columns, value_column]].copy()
    source["_source_key"] = source_key
    source.sort_values([*strata_columns, "_source_key"], inplace=True, kind="mergesort")
    source["_within_stratum"] = source.groupby(
        list(strata_columns), sort=False, dropna=False, observed=True
    ).cumcount()
    source.rename(columns={value_column: "_shuffled_attribute"}, inplace=True)

    destination = frame[list(strata_columns)].copy()
    destination["_destination_key"] = destination_key
    destination["_original_position"] = np.arange(len(frame), dtype=np.int64)
    destination.sort_values(
        [*strata_columns, "_destination_key"], inplace=True, kind="mergesort"
    )
    destination["_within_stratum"] = destination.groupby(
        list(strata_columns), sort=False, dropna=False, observed=True
    ).cumcount()
    mapped = destination.merge(
        source[[*strata_columns, "_within_stratum", "_shuffled_attribute"]],
        on=[*strata_columns, "_within_stratum"],
        how="left",
        validate="one_to_one",
    )
    mapped.sort_values("_original_position", inplace=True)
    return pd.to_numeric(mapped["_shuffled_attribute"], errors="coerce").to_numpy()


def rank_correlation_from_sums(
    frame: DataFrame,
    *,
    x_sum: str,
    x_square_sum: str,
    xy_sum: str,
) -> Series:
    count = frame["event_count"].astype(float)
    y_sum = frame["_outcome_rank_sum"]
    y_square_sum = frame["_outcome_rank_square_sum"]
    covariance = frame[xy_sum] - frame[x_sum] * y_sum / count
    x_variance = frame[x_square_sum] - frame[x_sum].pow(2) / count
    y_variance = y_square_sum - y_sum.pow(2) / count
    denominator = np.sqrt(np.maximum(x_variance * y_variance, 0.0))
    result = covariance.div(denominator.replace(0.0, np.nan))
    result[(count < 3.0) | (frame["attribute_unique_values"] < 3.0)] = np.nan
    return result.clip(-1.0, 1.0)


def nominal_spearman_pvalue(correlation: Series, count: Series) -> np.ndarray:
    result = np.full(len(correlation), np.nan, dtype=float)
    values = correlation.to_numpy(dtype=float)
    observations = count.to_numpy(dtype=float)
    valid = np.isfinite(values) & (observations > 2.0)
    if not valid.any():
        return result
    clipped = np.clip(values[valid], -0.999999999999, 0.999999999999)
    statistic = clipped * np.sqrt((observations[valid] - 2.0) / (1.0 - clipped**2))
    result[valid] = 2.0 * student_t.sf(np.abs(statistic), observations[valid] - 2.0)
    return result


def attribute_target_calibration(
    frame: DataFrame,
    *,
    attribute_column: str,
    target_column: str,
    target_name: str,
    horizon_hours: int,
) -> DataFrame:
    working = frame.loc[
        np.isfinite(pd.to_numeric(frame[target_column], errors="coerce"))
    ].copy()
    if working.empty:
        return DataFrame()
    group_columns = list(CALIBRATION_CELL_COLUMNS)
    working["_outcome_rank"] = working.groupby(
        group_columns, sort=False, dropna=False, observed=True
    )[target_column].rank(method="average", pct=True)
    working["_score_outcome"] = working["_score_rank"] * working["_outcome_rank"]
    working["_shuffle_outcome"] = working["_shuffle_rank"] * working["_outcome_rank"]
    working["_score_square"] = working["_score_rank"].pow(2)
    working["_shuffle_square"] = working["_shuffle_rank"].pow(2)
    working["_outcome_square"] = working["_outcome_rank"].pow(2)
    grouped = working.groupby(group_columns, sort=False, dropna=False, observed=True)
    result = grouped.agg(
        event_count=(target_column, "size"),
        attribute_unique_values=(attribute_column, "nunique"),
        attribute_min=(attribute_column, "min"),
        attribute_median=(attribute_column, "median"),
        attribute_max=(attribute_column, "max"),
        outcome_median=(target_column, "median"),
        _score_rank_sum=("_score_rank", "sum"),
        _score_rank_square_sum=("_score_square", "sum"),
        _score_outcome_sum=("_score_outcome", "sum"),
        _shuffle_rank_sum=("_shuffle_rank", "sum"),
        _shuffle_rank_square_sum=("_shuffle_square", "sum"),
        _shuffle_outcome_sum=("_shuffle_outcome", "sum"),
        _outcome_rank_sum=("_outcome_rank", "sum"),
        _outcome_rank_square_sum=("_outcome_square", "sum"),
    )
    result["observed_spearman"] = rank_correlation_from_sums(
        result,
        x_sum="_score_rank_sum",
        x_square_sum="_score_rank_square_sum",
        xy_sum="_score_outcome_sum",
    )
    result["shuffled_spearman"] = rank_correlation_from_sums(
        result,
        x_sum="_shuffle_rank_sum",
        x_square_sum="_shuffle_rank_square_sum",
        xy_sum="_shuffle_outcome_sum",
    )

    def bucket_summary(mask: Series, prefix: str) -> DataFrame:
        return (
            working.loc[mask]
            .groupby(group_columns, sort=False, dropna=False, observed=True)[target_column]
            .agg(**{f"{prefix}_count": "size", f"{prefix}_mean": "mean"})
        )

    result = result.join(bucket_summary(working["_score_rank"] <= 0.20, "bottom_score"))
    result = result.join(bucket_summary(working["_score_rank"] >= 0.80, "top_score"))
    result = result.join(
        bucket_summary(working["_shuffle_rank"] <= 0.20, "bottom_shuffled_score")
    )
    result = result.join(
        bucket_summary(working["_shuffle_rank"] >= 0.80, "top_shuffled_score")
    )
    result.reset_index(inplace=True)
    result["observed_spearman_pvalue_nominal"] = nominal_spearman_pvalue(
        result["observed_spearman"], result["event_count"]
    )
    result["shuffled_spearman_pvalue_nominal"] = nominal_spearman_pvalue(
        result["shuffled_spearman"], result["event_count"]
    )
    result["spearman_lift_vs_shuffle"] = (
        result["observed_spearman"] - result["shuffled_spearman"]
    )
    result["absolute_spearman_lift_vs_shuffle"] = (
        result["observed_spearman"].abs() - result["shuffled_spearman"].abs()
    )
    result["top_bottom_difference"] = (
        result["top_score_mean"] - result["bottom_score_mean"]
    )
    result["shuffled_top_bottom_difference"] = (
        result["top_shuffled_score_mean"] - result["bottom_shuffled_score_mean"]
    )
    result["top_bottom_lift_vs_shuffle"] = (
        result["top_bottom_difference"] - result["shuffled_top_bottom_difference"]
    )
    result["attribute_name"] = attribute_column
    result["target_name"] = target_name
    result["target_column"] = target_column
    result["horizon_hours"] = int(horizon_hours)
    result.drop(
        columns=[column for column in result.columns if column.startswith("_")],
        inplace=True,
    )
    return result


def attribute_calibration_cells(
    events: DataFrame,
    *,
    attribute_columns: Sequence[str],
    targets: Sequence[tuple[str, str, int]],
) -> DataFrame:
    actual = events.loc[events["control"] == "actual"].copy()
    if actual.empty:
        return DataFrame()
    pooled = actual.copy()
    pooled["approach_state"] = "all_approaches"
    actual = pd.concat([actual, pooled], ignore_index=True)
    state_columns = [
        *CALIBRATION_CELL_COLUMNS,
        "volatility_band",
        "contact_range_band",
    ]
    outputs: list[DataFrame] = []
    for attribute_column in attribute_columns:
        if attribute_column not in actual:
            continue
        values = pd.to_numeric(actual[attribute_column], errors="coerce")
        working = actual.loc[np.isfinite(values)].copy()
        if working.empty:
            continue
        working[attribute_column] = pd.to_numeric(
            working[attribute_column], errors="coerce"
        )
        working["_shuffled_attribute"] = deterministic_state_shuffle(
            working,
            value_column=attribute_column,
            strata_columns=state_columns,
        )
        cell_columns = list(CALIBRATION_CELL_COLUMNS)
        working["_score_rank"] = working.groupby(
            cell_columns, sort=False, dropna=False, observed=True
        )[attribute_column].rank(method="average", pct=True)
        working["_shuffle_rank"] = working.groupby(
            cell_columns, sort=False, dropna=False, observed=True
        )["_shuffled_attribute"].rank(method="average", pct=True)
        for target_column, target_name, horizon_hours in targets:
            output = attribute_target_calibration(
                working,
                attribute_column=attribute_column,
                target_column=target_column,
                target_name=target_name,
                horizon_hours=horizon_hours,
            )
            if not output.empty:
                outputs.append(output)
    return pd.concat(outputs, ignore_index=True) if outputs else DataFrame()


def aggregate_attribute_calibration(frame: DataFrame) -> DataFrame:
    if frame.empty:
        return DataFrame()
    keys = [
        "cohort",
        "source_timeframe",
        "batch",
        "level_family",
        "level_name",
        "representation",
        "zone_method",
        "approach_state",
        "attribute_name",
        "target_name",
        "horizon_hours",
    ]
    working = frame.copy()
    for source, destination in (
        ("observed_spearman", "observed_spearman_positive"),
        ("spearman_lift_vs_shuffle", "spearman_lift_positive"),
        ("absolute_spearman_lift_vs_shuffle", "absolute_spearman_lift_positive"),
        ("top_bottom_difference", "top_bottom_positive"),
        ("top_bottom_lift_vs_shuffle", "top_bottom_lift_positive"),
    ):
        working[destination] = (working[source] > 0.0).astype(float)
    working["nominal_pvalue_below_0_05"] = (
        working["observed_spearman_pvalue_nominal"] < 0.05
    ).astype(float)
    grouped = working.groupby(keys, sort=False, dropna=False, observed=True)
    result = grouped.agg(
        cell_count=("pair", "size"),
        pair_count=("pair", "nunique"),
        period_count=("period", "nunique"),
        event_count=("event_count", "sum"),
        largest_cell_events=("event_count", "max"),
        attribute_unique_values_median=("attribute_unique_values", "median"),
        observed_spearman_median=("observed_spearman", "median"),
        observed_spearman_positive_fraction=("observed_spearman_positive", "mean"),
        shuffled_spearman_median=("shuffled_spearman", "median"),
        spearman_lift_vs_shuffle_median=("spearman_lift_vs_shuffle", "median"),
        spearman_lift_positive_fraction=("spearman_lift_positive", "mean"),
        absolute_spearman_lift_vs_shuffle_median=(
            "absolute_spearman_lift_vs_shuffle",
            "median",
        ),
        absolute_spearman_lift_positive_fraction=(
            "absolute_spearman_lift_positive",
            "mean",
        ),
        top_bottom_difference_median=("top_bottom_difference", "median"),
        top_bottom_positive_fraction=("top_bottom_positive", "mean"),
        shuffled_top_bottom_difference_median=(
            "shuffled_top_bottom_difference",
            "median",
        ),
        top_bottom_lift_vs_shuffle_median=("top_bottom_lift_vs_shuffle", "median"),
        top_bottom_lift_positive_fraction=("top_bottom_lift_positive", "mean"),
        nominal_pvalue_below_0_05_fraction=("nominal_pvalue_below_0_05", "mean"),
    ).reset_index()
    result["largest_cell_event_fraction"] = result["largest_cell_events"].div(
        result["event_count"].replace(0.0, np.nan)
    )
    return result


def build_attribute_calibration_screen(repeatability: DataFrame) -> DataFrame:
    if repeatability.empty:
        return DataFrame()
    keys = [
        "cohort",
        "batch",
        "level_family",
        "level_name",
        "representation",
        "approach_state",
        "attribute_name",
        "target_name",
    ]
    working = repeatability.copy()
    working["configuration_spearman_positive"] = (
        working["observed_spearman_median"] > 0.0
    ).astype(float)
    working["configuration_absolute_lift_positive"] = (
        working["absolute_spearman_lift_vs_shuffle_median"] > 0.0
    ).astype(float)
    working["configuration_top_bottom_positive"] = (
        working["top_bottom_difference_median"] > 0.0
    ).astype(float)
    grouped = working.groupby(keys, sort=False, dropna=False, observed=True)
    screen = grouped.agg(
        configuration_rows=("target_name", "size"),
        source_timeframe_count=("source_timeframe", "nunique"),
        zone_method_count=("zone_method", "nunique"),
        horizon_count=("horizon_hours", "nunique"),
        pair_count_max=("pair_count", "max"),
        period_count_max=("period_count", "max"),
        event_count_median=("event_count", "median"),
        largest_cell_event_fraction_max=("largest_cell_event_fraction", "max"),
        observed_spearman_median=("observed_spearman_median", "median"),
        observed_spearman_positive_configuration_fraction=(
            "configuration_spearman_positive",
            "mean",
        ),
        shuffled_spearman_median=("shuffled_spearman_median", "median"),
        absolute_spearman_lift_vs_shuffle_median=(
            "absolute_spearman_lift_vs_shuffle_median",
            "median",
        ),
        absolute_lift_positive_configuration_fraction=(
            "configuration_absolute_lift_positive",
            "mean",
        ),
        top_bottom_difference_median=("top_bottom_difference_median", "median"),
        top_bottom_positive_configuration_fraction=(
            "configuration_top_bottom_positive",
            "mean",
        ),
        top_bottom_lift_vs_shuffle_median=(
            "top_bottom_lift_vs_shuffle_median",
            "median",
        ),
        nominal_pvalue_below_0_05_fraction_median=(
            "nominal_pvalue_below_0_05_fraction",
            "median",
        ),
    ).reset_index()
    screen.sort_values(
        [
            "absolute_lift_positive_configuration_fraction",
            "absolute_spearman_lift_vs_shuffle_median",
            "event_count_median",
        ],
        ascending=False,
        inplace=True,
        ignore_index=True,
    )
    return screen


def build_attribute_calibration(
    tasks: Sequence[AtlasTask], manifest: dict[str, Any]
) -> dict[str, Any]:
    CALIBRATION_CELL_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    repeatability_periods = {
        period["id"]
        for period in manifest["data"]["chronological_periods"]
        if period["role"] != "diagnostic_only_not_confirmation"
    }
    period_ids = {period["id"] for period in manifest["data"]["chronological_periods"]}
    horizons = tuple(manifest["reaction_definition"]["horizons_hours"])
    expanded_frames: list[DataFrame] = []
    completed_tasks = 0
    cell_rows = 0
    attributes_seen: set[str] = set()
    for task in tasks:
        event_path = atlas_event_path(task)
        if not event_path.is_file():
            continue
        available = set(pq.ParquetFile(event_path).schema.names)
        attributes = calibratable_attribute_columns(available)
        targets = calibration_target_columns(available, horizons)
        required = {
            *CALIBRATION_CELL_COLUMNS,
            "control",
            "event_time",
            "base_index",
            "volatility_band",
            "contact_range_band",
            *attributes,
            *(column for column, _, _ in targets),
        }
        events = pd.read_parquet(event_path, columns=sorted(required))
        events = events.loc[events["period"].isin(period_ids)].copy()
        cells = attribute_calibration_cells(
            events,
            attribute_columns=attributes,
            targets=targets,
        )
        output_path = CALIBRATION_CELL_DIR / f"{event_path.stem}-attribute-calibration.parquet"
        atomic_write_parquet(cells, output_path)
        completed_tasks += 1
        cell_rows += len(cells)
        attributes_seen.update(attributes)
        if not cells.empty:
            repeatable = cells.loc[cells["period"].isin(repeatability_periods)].copy()
            if not repeatable.empty:
                expanded_frames.append(add_cohort_lenses(repeatable, manifest))

    expanded = pd.concat(expanded_frames, ignore_index=True) if expanded_frames else DataFrame()
    repeatability = aggregate_attribute_calibration(expanded)
    screen = build_attribute_calibration_screen(repeatability)
    atomic_write_parquet(repeatability, CALIBRATION_REPEATABILITY_PATH)
    screen_path = REVIEW_DIR / "g0b_attribute_calibration_screen.parquet"
    atomic_write_parquet(screen, screen_path)
    record = {
        "created_at_utc": utc_now(),
        "completed_tasks": completed_tasks,
        "cell_rows": cell_rows,
        "repeatability_rows": len(repeatability),
        "screen_rows": len(screen),
        "attributes": sorted(attributes_seen),
        "repeatability_periods": sorted(repeatability_periods),
        "control": (
            "Deterministic attribute permutation inside identical pair, timeframe, "
            "level, zone, period, approach, volatility-band, and contact-range-band cells."
        ),
        "nominal_pvalue_is_promotion_rule": False,
        "direction_prediction": False,
        "profit_optimization": False,
        "outputs": {
            "per_task_cells": str(CALIBRATION_CELL_DIR),
            "repeatability": str(CALIBRATION_REPEATABILITY_PATH),
            "screen": str(screen_path),
        },
    }
    atomic_write_json(record, REVIEW_DIR / "g0b_attribute_calibration_record.json")
    return record


def build_comparisons(tasks: Sequence[AtlasTask], manifest: dict[str, Any]) -> dict[str, Any]:
    comparison_dir = LARGE_REVIEW_DIR / "pair_comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    compact_frames: list[DataFrame] = []
    comparison_rows = 0
    completed_tasks = 0
    period_ids = {period["id"] for period in manifest["data"]["chronological_periods"]}
    for task in tasks:
        summary_path = atlas_summary_path(task)
        metadata_path = atlas_metadata_path(task)
        if not summary_path.is_file() or not metadata_path.is_file():
            continue
        summary = pd.read_parquet(summary_path)
        summary = summary.loc[summary["period"].isin(period_ids)].copy()
        if summary.empty:
            completed_tasks += 1
            continue
        comparison = comparison_from_summary(summary)
        comparison_path = comparison_dir / (
            summary_path.stem.removesuffix("-summary") + "-comparisons.parquet"
        )
        atomic_write_parquet(comparison, comparison_path)
        comparison_rows += len(comparison)
        completed_tasks += 1
        compact_frames.append(
            comparison[
                [
                    *KEY_COLUMNS,
                    "control",
                    "actual_n",
                    "control_n",
                    "actual_unique_days",
                    "actual_largest_day_fraction",
                    *PRIMARY_DELTA_COLUMNS,
                ]
            ].copy()
        )

    compact = pd.concat(compact_frames, ignore_index=True) if compact_frames else DataFrame()
    expanded = add_cohort_lenses(compact, manifest) if not compact.empty else compact
    by_period_keys = [
        "cohort",
        "source_timeframe",
        "batch",
        "level_family",
        "level_name",
        "representation",
        "zone_method",
        "period",
        "approach_state",
        "horizon_hours",
        "control",
    ]
    repeatability_keys = [column for column in by_period_keys if column != "period"]
    by_period = aggregate_comparisons(expanded, by_period_keys)
    repeatability_periods = {
        period["id"]
        for period in manifest["data"]["chronological_periods"]
        if period["role"] != "diagnostic_only_not_confirmation"
    }
    repeatability = aggregate_comparisons(
        expanded.loc[expanded["period"].isin(repeatability_periods)],
        repeatability_keys,
    )
    screen = build_control_screen(repeatability)
    cohort_coverage = build_cohort_coverage(tasks, manifest)
    if not cohort_coverage.empty:
        coverage_keys = [
            "cohort",
            "source_timeframe",
            "batch",
            "level_family",
            "level_name",
            "representation",
            "zone_method",
        ]
        screen = screen.merge(
            cohort_coverage,
            on=coverage_keys,
            how="left",
            validate="many_to_one",
        )
    timing = build_timing_review(tasks, manifest)

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(by_period, COHORT_BY_PERIOD_PATH)
    atomic_write_parquet(repeatability, REPEATABILITY_PATH)
    atomic_write_parquet(screen, REVIEW_DIR / "g0b_control_screen.parquet")
    atomic_write_parquet(cohort_coverage, REVIEW_DIR / "g0b_cohort_coverage.parquet")
    record = {
        "created_at_utc": utc_now(),
        "completed_tasks": completed_tasks,
        "pair_comparison_rows": comparison_rows,
        "cohort_by_period_rows": len(by_period),
        "repeatability_rows": len(repeatability),
        "control_screen_rows": len(screen),
        "cohort_coverage_rows": len(cohort_coverage),
        **timing,
        "repeatability_periods": sorted(repeatability_periods),
        "direction_prediction": False,
        "profit_optimization": False,
        "outputs": {
            "pair_comparisons": str(comparison_dir),
            "cohort_by_period": str(COHORT_BY_PERIOD_PATH),
            "repeatability": str(REPEATABILITY_PATH),
            "control_screen": str(REVIEW_DIR / "g0b_control_screen.parquet"),
            "cohort_coverage": str(REVIEW_DIR / "g0b_cohort_coverage.parquet"),
            "timing_repeatability": str(REVIEW_DIR / "g0b_timing_repeatability.parquet"),
        },
    }
    atomic_write_json(record, REVIEW_DIR / "g0b_review_record.json")
    return record


def comparison_from_summary(summary: DataFrame) -> DataFrame:
    required = {*KEY_COLUMNS, "control", "event_count", *METRIC_COLUMNS}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"Detailed summary is missing columns: {missing}")
    actual = summary.loc[summary["control"] == "actual"].copy()
    actual_columns = [
        *KEY_COLUMNS,
        "event_count",
        "unique_event_days",
        "largest_day_event_fraction",
        *METRIC_COLUMNS,
    ]
    actual = actual[actual_columns].rename(
        columns={
            "event_count": "actual_n",
            "unique_event_days": "actual_unique_days",
            "largest_day_event_fraction": "actual_largest_day_fraction",
            **{column: f"actual_{column}" for column in METRIC_COLUMNS},
        }
    )
    frames: list[DataFrame] = []
    for control in DEFAULT_CONTROLS:
        if control == "actual":
            continue
        control_frame = summary.loc[summary["control"] == control].copy()
        control_frame = control_frame[[*KEY_COLUMNS, "event_count", *METRIC_COLUMNS]].rename(
            columns={
                "event_count": "control_n",
                **{column: f"control_{column}" for column in METRIC_COLUMNS},
            }
        )
        merged = actual.merge(
            control_frame,
            on=list(KEY_COLUMNS),
            how="left",
            validate="one_to_one",
        )
        merged["control"] = control
        for column in METRIC_COLUMNS:
            merged[f"{column}_delta"] = merged[f"actual_{column}"] - merged[f"control_{column}"]
        frames.append(merged)
    return pd.concat(frames, ignore_index=True) if frames else DataFrame()


def add_cohort_lenses(frame: DataFrame, manifest: dict[str, Any]) -> DataFrame:
    cohorts = manifest["reporting_groups"]["coin_cohorts"]
    primary: dict[str, str] = {}
    for cohort, pairs in cohorts.items():
        for pair in pairs:
            if pair in primary:
                raise ValueError(f"Pair {pair} appears in multiple primary cohorts.")
            primary[pair] = cohort
    missing = sorted(set(frame["pair"]).difference(primary))
    if missing:
        raise ValueError(f"Pairs lack a primary cohort: {missing}")
    primary_frame = frame.copy()
    primary_frame["cohort"] = primary_frame["pair"].map(primary)
    all_pairs = frame.copy()
    all_pairs["cohort"] = "all_pairs"
    non_btc = frame.loc[frame["pair"] != "BTC/USDT:USDT"].copy()
    non_btc["cohort"] = "all_non_btc"
    return pd.concat([primary_frame, all_pairs, non_btc], ignore_index=True)


def aggregate_comparisons(frame: DataFrame, group_columns: Sequence[str]) -> DataFrame:
    working = frame.loc[frame["actual_n"].notna() & frame["control_n"].notna()].copy()
    if working.empty:
        return DataFrame(columns=list(group_columns))
    working["matched_fraction"] = working["control_n"].div(working["actual_n"].replace(0.0, np.nan))
    for column in PRIMARY_DELTA_COLUMNS:
        working[f"{column}_positive"] = (working[column] > 0.0).astype(float)
        working[f"{column}_weighted"] = working[column] * working["actual_n"]
    grouped = working.groupby(list(group_columns), sort=False, dropna=False, observed=True)
    result = grouped.agg(
        cell_count=("pair", "size"),
        pair_count=("pair", "nunique"),
        period_count=("period", "nunique") if "period" in working else ("pair", "size"),
        actual_events=("actual_n", "sum"),
        control_events=("control_n", "sum"),
        median_matched_fraction=("matched_fraction", "median"),
        largest_cell_events=("actual_n", "max"),
        largest_day_fraction_median=("actual_largest_day_fraction", "median"),
    )
    result["largest_cell_event_fraction"] = result["largest_cell_events"].div(
        result["actual_events"].replace(0.0, np.nan)
    )
    for column in PRIMARY_DELTA_COLUMNS:
        result[f"{column}_median"] = grouped[column].median()
        result[f"{column}_mean"] = grouped[column].mean()
        result[f"{column}_positive_fraction"] = grouped[f"{column}_positive"].mean()
        weighted_sum = grouped[f"{column}_weighted"].sum(min_count=1)
        result[f"{column}_weighted_mean"] = weighted_sum.div(
            result["actual_events"].replace(0.0, np.nan)
        )
    return result.reset_index()


def build_control_screen(repeatability: DataFrame) -> DataFrame:
    index_columns = [
        "cohort",
        "source_timeframe",
        "batch",
        "level_family",
        "level_name",
        "representation",
        "zone_method",
        "approach_state",
        "horizon_hours",
    ]
    value_columns = [
        "cell_count",
        "pair_count",
        "period_count",
        "actual_events",
        "largest_cell_event_fraction",
        *[f"{column}_median" for column in PRIMARY_DELTA_COLUMNS],
        *[f"{column}_positive_fraction" for column in PRIMARY_DELTA_COLUMNS],
    ]
    wide = repeatability.pivot_table(
        index=index_columns,
        columns="control",
        values=value_columns,
        aggfunc="first",
        observed=True,
    )
    wide.columns = [f"{metric}__{control}" for metric, control in wide.columns]
    wide = wide.reset_index()
    for delta_column in PRIMARY_DELTA_COLUMNS:
        metric = delta_column.removesuffix("_delta")
        columns = [
            f"{delta_column}_median__{control}"
            for control in DEFAULT_CONTROLS
            if control != "actual" and f"{delta_column}_median__{control}" in wide.columns
        ]
        available_column = f"available_controls__{metric}"
        higher_column = f"controls_where_actual_is_higher__{metric}"
        lower_column = f"controls_where_actual_is_lower__{metric}"
        sign_column = f"control_sign_summary__{metric}"
        wide[available_column] = wide[columns].notna().sum(axis=1)
        wide[higher_column] = (wide[columns] > 0.0).sum(axis=1)
        wide[lower_column] = (wide[columns] < 0.0).sum(axis=1)
        wide[sign_column] = np.select(
            [
                wide[available_column] == 0,
                wide[higher_column] == wide[available_column],
                wide[lower_column] == wide[available_column],
            ],
            ["unavailable", "higher_than_all_available", "lower_than_all_available"],
            default="mixed_or_tied",
        )
    abs_columns = [
        f"abs_excursion_mean_delta_median__{control}"
        for control in DEFAULT_CONTROLS
        if control != "actual" and f"abs_excursion_mean_delta_median__{control}" in wide.columns
    ]
    range_columns = [
        f"range_ratio_mean_delta_median__{control}"
        for control in DEFAULT_CONTROLS
        if control != "actual" and f"range_ratio_mean_delta_median__{control}" in wide.columns
    ]
    wide["available_control_count"] = wide[abs_columns].notna().sum(axis=1)
    wide["controls_where_actual_has_higher_absolute_activity"] = (wide[abs_columns] > 0.0).sum(
        axis=1
    )
    wide["controls_where_actual_has_higher_range_activity"] = (wide[range_columns] > 0.0).sum(
        axis=1
    )
    matched_pair_count = "pair_count__matched_random_time"
    matched_cell_count = "cell_count__matched_random_time"
    wide["sample_scope_note"] = np.select(
        [
            wide.get(matched_pair_count, 0) >= 3,
            wide.get(matched_cell_count, 0) >= 3,
        ],
        ["multi_pair", "multi_period_or_cell"],
        default="narrow_or_sparse",
    )
    sort_columns = [
        column
        for column in (
            "controls_where_actual_has_higher_absolute_activity",
            "abs_excursion_mean_delta_positive_fraction__matched_random_time",
            "actual_events__matched_random_time",
        )
        if column in wide.columns
    ]
    if sort_columns:
        wide.sort_values(sort_columns, ascending=False, inplace=True, ignore_index=True)
    return wide


def parquet_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())

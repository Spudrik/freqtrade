from __future__ import annotations

import os


# ruff: noqa: E402
# Keep each pair worker to one numerical-library thread.
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
    OUTPUT_ROOT,
    atomic_write_json,
    atomic_write_parquet,
    load_manifest,
    load_ohlcv,
    normalize_dates,
    numeric_array,
    ohlcv_path,
    sha256_file,
    utc_now,
    validate_worker_count,
    wilder_atr,
)


FROZEN_BATCH = OUTPUT_ROOT / "generation1_review" / "g2_frozen_branch_batch.json"
REPORT_ROOT = OUTPUT_ROOT / "generation2_branches" / "g2e_density_zones"
OUTPUT_SCHEMA_VERSION = 1
HISTORY_HOURS = (168, 720)
CANDIDATE_BIN_WIDTHS_ATR = (0.20, 0.25, 1.0 / 3.0)
MAX_ZONES = 3
SWING_LEFT_BARS = 3
SWING_RIGHT_BARS = 3
CLOSE_HISTORY_MINIMUM_FRACTION = 0.80
SWING_MINIMUM_POINTS = {168: 6, 720: 15}
ZONE_FAMILIES = ("confirmed_swing_price_density", "repeated_close_density")


@dataclass(frozen=True)
class SwingPoints:
    pivot_positions: np.ndarray
    available_positions: np.ndarray
    prices: np.ndarray


@dataclass(frozen=True)
class DensityZoneSet:
    family: str
    history_hours: int
    bin_width_atr: float
    levels: np.ndarray
    half_widths: np.ndarray
    valid: np.ndarray
    support_counts: np.ndarray
    source_counts: np.ndarray


@dataclass(frozen=True)
class CoverageTask:
    pair: str
    manifest_path: str
    candidate_widths: tuple[float, ...]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 2E coverage-only preflight for causal swing-price and "
            "repeated-close density zones. It does not calculate reaction outcomes."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--candidate-widths-atr",
        default=",".join(f"{value:.12g}" for value in CANDIDATE_BIN_WIDTHS_ATR),
    )
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    validate_worker_count(args.workers, manifest=manifest)
    validate_frozen_branch(manifest)
    pairs = select_pairs(manifest, args.pairs)
    widths = parse_candidate_widths(args.candidate_widths_atr)
    run_dir = REPORT_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    request = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "frozen_batch_sha256": sha256_file(FROZEN_BATCH),
        "pairs": pairs,
        "candidate_bin_widths_atr": list(widths),
        "history_hours": list(HISTORY_HOURS),
        "families": list(ZONE_FAMILIES),
        "maximum_zones_per_family_history": MAX_ZONES,
        "confirmed_swing_left_bars": SWING_LEFT_BARS,
        "confirmed_swing_right_bars": SWING_RIGHT_BARS,
        "connected_bin_rule": "selected integer bins must be directly adjacent",
        "development_period_only": True,
        "reaction_outcomes_calculated": False,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    request_sha256 = stable_json_sha256(request)
    record_path = run_dir / "g2e_coverage_run_record.json"
    record = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "status": "running",
        "started_at_utc": utc_now(),
        "orchestrator_pid": os.getpid(),
        "request": request,
        "request_sha256": request_sha256,
        "baseline": (
            "Coverage-only construction comparison across three fixed ATR bin widths; "
            "no market-reaction value is available to the selector."
        ),
        "selection_rule": (
            "Prefer the middle 0.25 ATR width if it leaves repeated zones available "
            "without covering most of the nearby chart. Use another candidate only "
            "when the middle width fails the predeclared coverage diagnostics."
        ),
        "workers": args.workers,
        "reaction_outcomes_calculated": False,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(record, record_path)
    tasks = [
        CoverageTask(
            pair=pair,
            manifest_path=str(args.manifest.resolve()),
            candidate_widths=widths,
        )
        for pair in pairs
    ]
    try:
        results = run_coverage_tasks(tasks, workers=args.workers)
        failures = [row for row in results if row.get("status") == "failed"]
        inventory = DataFrame(
            [
                {
                    key: value
                    for key, value in result.items()
                    if key != "coverage_rows"
                }
                for result in results
            ]
        )
        atomic_write_parquet(inventory, run_dir / "g2e_coverage_pair_inventory.parquet")
        if failures:
            raise RuntimeError(
                f"{len(failures)} G2E coverage task(s) failed; inspect the inventory."
            )
        coverage = DataFrame(
            [row for result in results for row in result["coverage_rows"]]
        )
        if coverage.empty:
            raise ValueError("G2E coverage preflight produced no rows.")
        atomic_write_parquet(coverage, run_dir / "g2e_coverage_preflight.parquet")
        summary = summarize_coverage(coverage)
        atomic_write_parquet(summary, run_dir / "g2e_coverage_cohort_summary.parquet")
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "pair_tasks": results,
                "coverage_rows": len(coverage),
                "cohort_summary_rows": len(summary),
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


def validate_frozen_branch(manifest: dict[str, Any]) -> None:
    frozen = json.loads(FROZEN_BATCH.read_text(encoding="utf-8"))
    branches = {branch["id"] for branch in frozen["branches"]}
    if "g2e_causal_swing_price_density_zones" not in branches:
        raise ValueError("The frozen Generation 2E branch is unavailable.")
    boundary = frozen["research_boundary"]
    if boundary.get("direction_prediction") is not False:
        raise ValueError("Generation 2E must keep direction prediction disabled.")
    if boundary.get("profit_optimization") is not False:
        raise ValueError("Generation 2E must keep profit optimization disabled.")
    if manifest["data"]["base_timeframe"] != "1h":
        raise ValueError("Generation 2E coverage is fixed to one-hour base data.")


def select_pairs(manifest: dict[str, Any], requested: str) -> list[str]:
    allowed = list(manifest["data"]["pairs"])
    if requested.strip().lower() == "all":
        return allowed
    lookup: dict[str, str] = {}
    for pair in allowed:
        lookup[pair.upper()] = pair
        lookup[pair.split("/", maxsplit=1)[0].upper()] = pair
    selected: list[str] = []
    for value in (part.strip() for part in requested.split(",")):
        if not value:
            continue
        key = value.upper()
        if key not in lookup:
            raise ValueError(f"Pair {value!r} is outside the frozen manifest.")
        if lookup[key] not in selected:
            selected.append(lookup[key])
    if not selected:
        raise ValueError("No G2E coverage pairs were selected.")
    return selected


def parse_candidate_widths(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values or any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("Candidate ATR bin widths must be finite and positive.")
    if len(set(values)) != len(values):
        raise ValueError("Candidate ATR bin widths must be unique.")
    return values


def stable_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_coverage_tasks(
    tasks: Sequence[CoverageTask], *, workers: int
) -> list[dict[str, Any]]:
    if workers == 1:
        return [safe_coverage_pair(task) for task in tasks]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(safe_coverage_pair, task): task for task in tasks}
        for future in as_completed(future_map):
            results.append(future.result())
    return sorted(results, key=lambda row: row["pair"])


def safe_coverage_pair(task: CoverageTask) -> dict[str, Any]:
    try:
        return coverage_pair(task)
    except Exception as exc:
        return {
            "pair": task.pair,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def coverage_pair(task: CoverageTask) -> dict[str, Any]:
    started = time.perf_counter()
    manifest = load_manifest(Path(task.manifest_path))
    frame = density_source_frame(task.pair, manifest)
    swings = confirmed_swing_points(frame)
    development = development_mask(frame, manifest)
    coverage_rows: list[dict[str, Any]] = []
    for bin_width in task.candidate_widths:
        for history_hours in HISTORY_HOURS:
            for family in ZONE_FAMILIES:
                zones = causal_density_zones(
                    frame,
                    family=family,
                    history_hours=history_hours,
                    bin_width_atr=bin_width,
                    swings=swings,
                )
                coverage_rows.append(
                    coverage_record(
                        pair=task.pair,
                        frame=frame,
                        zones=zones,
                        development=development,
                    )
                )
    return {
        "pair": task.pair,
        "status": "completed",
        "coverage_rows": coverage_rows,
        "source_rows": len(frame),
        "confirmed_swing_points": len(swings.prices),
        "seconds": round(time.perf_counter() - started, 3),
    }


def density_source_frame(pair: str, manifest: dict[str, Any]) -> DataFrame:
    path = ohlcv_path(pair, manifest["data"]["base_timeframe"])
    frame = load_ohlcv(path)
    frame = frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    frame["date"] = normalize_dates(frame["date"])
    frame["base_atr"] = wilder_atr(frame, 14).shift(1)
    frame["pre_close"] = pd.to_numeric(frame["close"], errors="coerce").shift(1)
    end = manifest_analysis_end_exclusive(manifest)
    frame = frame.loc[frame["date"] < end].reset_index(drop=True)
    return frame


def manifest_analysis_end_exclusive(manifest: dict[str, Any]) -> pd.Timestamp:
    """Resolve the explicit end of the manifest's declared analysis periods."""
    data = manifest["data"]
    explicit = data.get("analysis_end_utc_exclusive")
    if explicit is not None:
        return pd.Timestamp(explicit)

    period_ends = [
        period.get("end_utc_exclusive")
        for period in data.get("chronological_periods", [])
        if period.get("end_utc_exclusive") is not None
    ]
    if not period_ends:
        raise ValueError(
            "Manifest must declare analysis_end_utc_exclusive or at least one "
            "chronological period end."
        )
    return max(pd.Timestamp(value) for value in period_ends)


def development_mask(frame: DataFrame, manifest: dict[str, Any]) -> np.ndarray:
    development = next(
        period
        for period in manifest["data"]["chronological_periods"]
        if period["role"] == "development"
    )
    dates = normalize_dates(frame["date"])
    return (
        (dates >= pd.Timestamp(development["start_utc"]))
        & (dates < pd.Timestamp(development["end_utc_exclusive"]))
    ).to_numpy()


def confirmed_swing_points(frame: DataFrame) -> SwingPoints:
    high = numeric_array(frame["high"])
    low = numeric_array(frame["low"])
    pivot_positions: list[int] = []
    available_positions: list[int] = []
    prices: list[float] = []
    for pivot in range(SWING_LEFT_BARS, len(frame) - SWING_RIGHT_BARS):
        confirmation = pivot + SWING_RIGHT_BARS + 1
        if confirmation >= len(frame):
            continue
        high_peers = np.concatenate(
            (
                high[pivot - SWING_LEFT_BARS : pivot],
                high[pivot + 1 : pivot + SWING_RIGHT_BARS + 1],
            )
        )
        low_peers = np.concatenate(
            (
                low[pivot - SWING_LEFT_BARS : pivot],
                low[pivot + 1 : pivot + SWING_RIGHT_BARS + 1],
            )
        )
        if (
            np.isfinite(high[pivot])
            and np.isfinite(high_peers).all()
            and high[pivot] > np.max(high_peers)
        ):
            pivot_positions.append(pivot)
            available_positions.append(confirmation)
            prices.append(float(high[pivot]))
        if (
            np.isfinite(low[pivot])
            and np.isfinite(low_peers).all()
            and low[pivot] < np.min(low_peers)
        ):
            pivot_positions.append(pivot)
            available_positions.append(confirmation)
            prices.append(float(low[pivot]))
    order = np.argsort(available_positions, kind="stable")
    return SwingPoints(
        pivot_positions=np.asarray(pivot_positions, dtype=np.int64)[order],
        available_positions=np.asarray(available_positions, dtype=np.int64)[order],
        prices=np.asarray(prices, dtype=np.float64)[order],
    )


def causal_density_zones(
    frame: DataFrame,
    *,
    family: str,
    history_hours: int,
    bin_width_atr: float,
    swings: SwingPoints | None = None,
) -> DensityZoneSet:
    if family not in ZONE_FAMILIES:
        raise ValueError(f"Unsupported density family: {family}")
    if history_hours not in HISTORY_HOURS:
        raise ValueError(f"Unsupported density history: {history_hours}")
    if bin_width_atr <= 0.0:
        raise ValueError("Density bin width must be positive.")
    dates = normalize_dates(frame["date"])
    date_ns = dates.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    atr = numeric_array(frame["base_atr"])
    pre_close = numeric_array(frame["pre_close"])
    close = numeric_array(frame["close"])
    if swings is None:
        swings = confirmed_swing_points(frame)
    levels = np.full((len(frame), MAX_ZONES), np.nan, dtype=np.float64)
    half_widths = np.full_like(levels, np.nan)
    support_counts = np.full_like(levels, np.nan)
    source_counts = np.full(len(frame), np.nan, dtype=np.float64)
    history_ns = int(pd.Timedelta(hours=history_hours).value)
    swing_dates_ns = date_ns[swings.pivot_positions]
    for position in range(len(frame)):
        if not np.isfinite(atr[position]) or atr[position] <= 0.0:
            continue
        history_start_ns = date_ns[position] - history_ns
        if date_ns[0] > history_start_ns:
            continue
        if family == "repeated_close_density":
            start = int(np.searchsorted(date_ns, history_start_ns, side="left"))
            source = close[start:position]
            source = source[np.isfinite(source) & (source > 0.0)]
            minimum = int(np.ceil(CLOSE_HISTORY_MINIMUM_FRACTION * history_hours))
        else:
            selected = (
                (swings.available_positions <= position)
                & (swing_dates_ns >= history_start_ns)
                & (swing_dates_ns < date_ns[position])
            )
            source = swings.prices[selected]
            source = source[np.isfinite(source) & (source > 0.0)]
            minimum = SWING_MINIMUM_POINTS[history_hours]
        if len(source) < minimum:
            continue
        source_counts[position] = float(len(source))
        candidates = connected_density_candidates(
            source,
            atr=float(atr[position]),
            reference=float(pre_close[position]),
            family=family,
            history_hours=history_hours,
            bin_width_atr=bin_width_atr,
        )
        for rank, candidate in enumerate(candidates[:MAX_ZONES]):
            levels[position, rank] = candidate[0]
            half_widths[position, rank] = candidate[1]
            support_counts[position, rank] = candidate[2]
    valid = np.isfinite(levels) & np.isfinite(half_widths) & (levels > 0.0)
    return DensityZoneSet(
        family=family,
        history_hours=history_hours,
        bin_width_atr=bin_width_atr,
        levels=levels,
        half_widths=half_widths,
        valid=valid,
        support_counts=support_counts,
        source_counts=source_counts,
    )


def connected_density_candidates(
    source: np.ndarray,
    *,
    atr: float,
    reference: float,
    family: str,
    history_hours: int,
    bin_width_atr: float,
) -> list[tuple[float, float, float, float]]:
    bin_size = bin_width_atr * atr
    if not np.isfinite(bin_size) or bin_size <= 0.0:
        return []
    bin_indexes = np.floor(source / bin_size).astype(np.int64)
    unique_bins, counts = np.unique(bin_indexes, return_counts=True)
    if not len(unique_bins):
        return []
    quantile_count = int(np.quantile(counts, 0.75, method="higher"))
    fixed_minimum = (
        2
        if family == "confirmed_swing_price_density"
        else max(3, int(np.ceil(history_hours * 0.01)))
    )
    threshold = max(fixed_minimum, quantile_count)
    selected_bins = unique_bins[counts >= threshold]
    if not len(selected_bins):
        return []
    groups: list[list[int]] = [[int(selected_bins[0])]]
    for bin_index in selected_bins[1:]:
        integer = int(bin_index)
        if integer == groups[-1][-1] + 1:
            groups[-1].append(integer)
        else:
            groups.append([integer])
    candidates: list[tuple[float, float, float, float]] = []
    for group in groups:
        included = np.isin(bin_indexes, group)
        prices = source[included]
        if not len(prices):
            continue
        lower = group[0] * bin_size
        upper = (group[-1] + 1) * bin_size
        centre = float(np.mean(prices))
        half_width = max(
            centre - lower,
            upper - centre,
            0.10 * atr,
        )
        support = float(len(prices))
        density = support / (len(group) * len(source))
        distance = abs(centre - reference) / atr if np.isfinite(reference) else np.inf
        candidates.append((centre, half_width, support, density - 1e-12 * distance))
    return sorted(candidates, key=lambda row: (-row[3], -row[2], row[1], row[0]))


def coverage_record(
    *,
    pair: str,
    frame: DataFrame,
    zones: DensityZoneSet,
    development: np.ndarray,
) -> dict[str, Any]:
    atr = numeric_array(frame["base_atr"])
    pre_close = numeric_array(frame["pre_close"])
    eligible = development & np.isfinite(atr) & (atr > 0.0) & np.isfinite(pre_close)
    any_zone = zones.valid.any(axis=1)
    selected = eligible & any_zone
    active_count = zones.valid.sum(axis=1).astype(float)
    nearest = np.full(len(frame), np.nan, dtype=np.float64)
    width_values = np.full_like(zones.half_widths, np.nan)
    support_ratio = np.full_like(zones.support_counts, np.nan)
    coverage = np.full(len(frame), np.nan, dtype=np.float64)
    valid_rows = np.flatnonzero(selected)
    for position in valid_rows:
        valid = zones.valid[position]
        levels = zones.levels[position, valid]
        widths = zones.half_widths[position, valid]
        nearest[position] = float(
            np.min(np.abs(levels - pre_close[position]) / atr[position])
        )
        width_values[position, valid] = widths / atr[position]
        if np.isfinite(zones.source_counts[position]) and zones.source_counts[position] > 0:
            support_ratio[position, valid] = (
                zones.support_counts[position, valid] / zones.source_counts[position]
            )
        coverage[position] = local_union_coverage(
            levels=levels,
            half_widths=widths,
            centre=float(pre_close[position]),
            atr=float(atr[position]),
        )
    eligible_count = int(eligible.sum())
    selected_count = int(selected.sum())
    widths = width_values[selected]
    widths = widths[np.isfinite(widths)]
    supports = support_ratio[selected]
    supports = supports[np.isfinite(supports)]
    return {
        "pair": pair,
        "family": zones.family,
        "history_hours": zones.history_hours,
        "bin_width_atr": zones.bin_width_atr,
        "development_eligible_rows": eligible_count,
        "development_rows_with_zone": selected_count,
        "zone_availability_fraction": (
            selected_count / eligible_count if eligible_count else np.nan
        ),
        "active_zone_count_median": safe_quantile(active_count[selected], 0.50),
        "active_zone_count_p90": safe_quantile(active_count[selected], 0.90),
        "nearest_zone_distance_atr_median": safe_quantile(nearest[selected], 0.50),
        "nearest_zone_distance_atr_p90": safe_quantile(nearest[selected], 0.90),
        "nearest_zone_within_2atr_fraction": (
            float(np.mean(nearest[selected] <= 2.0)) if selected_count else np.nan
        ),
        "native_half_width_atr_median": safe_quantile(widths, 0.50),
        "native_half_width_atr_p90": safe_quantile(widths, 0.90),
        "local_chart_coverage_4atr_median": safe_quantile(coverage[selected], 0.50),
        "local_chart_coverage_4atr_p90": safe_quantile(coverage[selected], 0.90),
        "zone_support_fraction_median": safe_quantile(supports, 0.50),
        "reaction_outcomes_calculated": False,
        "direction_prediction": False,
        "profit_optimization": False,
    }


def local_union_coverage(
    *, levels: np.ndarray, half_widths: np.ndarray, centre: float, atr: float
) -> float:
    lower_bound = centre - 2.0 * atr
    upper_bound = centre + 2.0 * atr
    intervals: list[tuple[float, float]] = []
    for level, half_width in zip(levels, half_widths, strict=True):
        lower = max(lower_bound, float(level - half_width))
        upper = min(upper_bound, float(level + half_width))
        if lower < upper:
            intervals.append((lower, upper))
    if not intervals:
        return 0.0
    intervals.sort()
    covered = 0.0
    active_lower, active_upper = intervals[0]
    for lower, upper in intervals[1:]:
        if lower <= active_upper:
            active_upper = max(active_upper, upper)
        else:
            covered += active_upper - active_lower
            active_lower, active_upper = lower, upper
    covered += active_upper - active_lower
    return covered / (4.0 * atr)


def safe_quantile(values: np.ndarray, quantile: float) -> float:
    selected = np.asarray(values, dtype=float)
    selected = selected[np.isfinite(selected)]
    return float(np.quantile(selected, quantile)) if len(selected) else np.nan


def summarize_coverage(coverage: DataFrame) -> DataFrame:
    metric_columns = [
        "zone_availability_fraction",
        "active_zone_count_median",
        "active_zone_count_p90",
        "nearest_zone_distance_atr_median",
        "nearest_zone_distance_atr_p90",
        "nearest_zone_within_2atr_fraction",
        "native_half_width_atr_median",
        "native_half_width_atr_p90",
        "local_chart_coverage_4atr_median",
        "local_chart_coverage_4atr_p90",
        "zone_support_fraction_median",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in coverage.groupby(
        ["family", "history_hours", "bin_width_atr"],
        observed=True,
        sort=False,
    ):
        row: dict[str, Any] = {
            "family": key[0],
            "history_hours": key[1],
            "bin_width_atr": key[2],
            "coin_count": int(group["pair"].nunique()),
            "coins": ";".join(sorted(group["pair"].astype(str).unique())),
            "reaction_outcomes_calculated": False,
            "direction_prediction": False,
            "profit_optimization": False,
        }
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"equal_coin_median__{metric}"] = float(values.median())
            row[f"equal_coin_min__{metric}"] = float(values.min())
            row[f"equal_coin_max__{metric}"] = float(values.max())
        rows.append(row)
    return DataFrame(rows)


if __name__ == "__main__":
    raise SystemExit(main())

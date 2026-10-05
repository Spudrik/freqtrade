"""Run the frozen China NBS nominal-clock unsigned-activity test."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_broad_relevance_direct as shared,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_central_bank_activity_direct as central_activity,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_layer2_breadth_freeze as layer2_sources,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_regional_nominal_clock_freeze as source_freeze,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_treasury_refunding_activity_direct as treasury_activity,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
OUTPUT_ROOT = source_freeze.OUTPUT_ROOT
CONTRACT_PATH = OUTPUT_ROOT / "regional_nominal_clock_market_analysis_contract.json"
CONTRACT_SHA256 = "6ca9ce1d421e27254171f901ef07dd0b3bdcc07f0be37187a1d98add29a58d02"
DIRECT_ROOT = OUTPUT_ROOT / "direct_review_20260912a"
CONTROL_PATH = DIRECT_ROOT / "regional_nominal_control_map.csv"
EXTERNAL_PATH = DIRECT_ROOT / "regional_external_collision_anchors.csv"
COVERAGE_PATH = DIRECT_ROOT / "regional_market_coverage.csv"
WINDOW_COVERAGE_PATH = DIRECT_ROOT / "regional_window_coverage.csv"
METRICS_PATH = DIRECT_ROOT / "regional_candidate_metrics.parquet"
SCORES_PATH = DIRECT_ROOT / "regional_candidate_scores.parquet"
SUMMARY_PATH = DIRECT_ROOT / "regional_activity_cell_summary.csv"
ROUTES_PATH = DIRECT_ROOT / "regional_activity_route_decisions.csv"
NULL_PATH = DIRECT_ROOT / "regional_activity_global_null.parquet"
REPORT_PATH = DIRECT_ROOT / "regional_activity_plain_review.md"
RESULT_PATH = DIRECT_ROOT / "regional_activity_result.json"

ASSETS = ("BTC/USDT:USDT", "ETH/USDT:USDT")
HORIZONS = (60, 240)
OFFSETS = (-30, 0, 30)
PARTITIONS = ("development_2021_2023", "internal_validation_2024_2025")
VIEWS = ("all_events", "collision_clean")
FAMILIES = tuple(source_freeze.FAMILY_SPECS)
CONTROL_COUNT = 12
CANDIDATE_COUNT = CONTROL_COUNT + 1
CONTROL_SEARCH_WEEKS = 60
COLLISION_HOURS = 4
METRIC_COLUMNS = ("absolute_movement", "full_range", "volume")
SCORE_COLUMNS = tuple(f"candidate_score_{slot:02d}" for slot in range(CANDIDATE_COUNT))
PERMUTATIONS = 2000
PERMUTATION_SEED = 20260912
DIRECT_MEDIAN = {0: 1.20, -30: 1.10, 30: 1.10}
DIRECT_RATE = {0: 0.55, -30: 0.50, 30: 0.50}
MINIMUM_EVENTS = {
    "development_2021_2023": 12,
    "internal_validation_2024_2025": 8,
}


@dataclass(frozen=True)
class CellSpec:
    """One frozen period/view/offset cell inside a searched route."""

    row_indices: np.ndarray
    minimum_events: int


@dataclass(frozen=True)
class RouteSpec:
    """All linked cells for one family, asset, and horizon."""

    key: tuple[str, str, int]
    cells_by_offset: Mapping[int, tuple[CellSpec, ...]]


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def validate_artifact_record(record: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(record["path"]))
    if not path.is_file() or g0.sha256_file(path) != str(record["sha256"]):
        raise ValueError(f"Frozen artifact changed: {label} -> {path}")
    return path


def load_contract_and_sources() -> tuple[dict[str, Any], DataFrame]:
    if g0.sha256_file(CONTRACT_PATH) != CONTRACT_SHA256:
        raise ValueError("Regional market-analysis contract changed after outcome-blind freeze")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen_outcome_blind_regional_nominal_clock_market_analysis":
        raise ValueError("Regional market-analysis contract is not frozen")
    if contract.get("market_outcomes_opened_before_contract"):
        raise ValueError("Regional market-analysis contract is not outcome-blind")
    if contract.get("profit_used") or contract.get("direction_tested"):
        raise ValueError("Regional contract permits a prohibited market outcome")
    if contract.get("eurostat_included"):
        raise ValueError("Eurostat must remain excluded from this market test")
    source_paths = {
        name: validate_artifact_record(record, label=name)
        for name, record in contract["source_contracts"].items()
    }
    result = json.loads(source_paths["source_result"].read_text(encoding="utf-8"))
    freeze = json.loads(source_paths["source_freeze"].read_text(encoding="utf-8"))
    if result.get("status") != "completed_regional_nominal_clock_source_freeze":
        raise ValueError("Regional source result is not terminal")
    if freeze.get("status") != "frozen_regional_nominal_clock_batch_before_market_outcomes":
        raise ValueError("Regional source definitions are not frozen")
    if result.get("outcomes_read") or freeze.get("outcomes_read"):
        raise ValueError("Regional source freeze unexpectedly opened outcomes")
    catalogue = pd.read_csv(source_paths["source_catalogue"])
    catalogue["anchor_utc"] = pd.to_datetime(catalogue["anchor_utc"], utc=True)
    if set(catalogue["event_family"]) != set(FAMILIES):
        raise ValueError("Regional source families differ from the frozen search")
    if not catalogue["nominal_clock_sensitivity_eligible"].astype(bool).all():
        raise ValueError("An ineligible nominal clock entered the market test")
    if catalogue["event_id"].duplicated().any() or len(catalogue) != 175:
        raise ValueError("Regional source catalogue is incomplete or duplicated")
    return contract, catalogue


def _source_artifacts(module: Any, catalogue_path: Path) -> dict[str, Any]:
    return {
        "catalogue": artifact(catalogue_path),
        "freeze": artifact(module.FREEZE_PATH),
        "result": artifact(module.RESULT_PATH),
    }


def load_external_anchors() -> tuple[DataFrame, list[dict[str, Any]]]:
    """Load source-only clocks already validated by existing event helpers."""

    rows: list[dict[str, Any]] = []
    source_artifacts: list[dict[str, Any]] = []

    common, common_artifacts = central_activity.load_external_anchors()
    rows.extend({"anchor_utc": value, "source_group": "common_scheduled"} for value in common)
    source_artifacts.extend(common_artifacts)

    _, central = central_activity.load_frozen_inputs()
    rows.extend(
        {"anchor_utc": value, "source_group": "scheduled_central_bank"}
        for value in central["available_at_utc"]
    )
    source_artifacts.append(
        {
            "source": "scheduled_central_bank",
            **_source_artifacts(central_activity.frozen, central_activity.CATALOG_PATH),
        }
    )

    _, treasury = treasury_activity.load_frozen_inputs()
    rows.extend(
        {"anchor_utc": value, "source_group": "treasury_refunding"}
        for value in treasury["anchor_utc"]
    )
    source_artifacts.append(
        {
            "source": "treasury_refunding",
            **_source_artifacts(treasury_activity.frozen, treasury_activity.CATALOG_PATH),
        }
    )

    layer2_specs = (
        (
            "uk_ons_cpi",
            layer2_sources.uk_cpi,
            layer2_sources.uk_cpi.CATALOGUE_PATH,
            "completed_uk_ons_cpi_catalogue",
            "frozen_uk_ons_cpi_catalogue_before_market_outcomes",
            "anchor_utc",
        ),
        (
            "japan_tankan",
            layer2_sources.tankan,
            layer2_sources.tankan.CATALOGUE_PATH,
            "completed_japan_tankan_catalogue",
            "frozen_japan_tankan_catalogue_before_market_outcomes",
            "anchor_utc",
        ),
        (
            "esma_euro_signed_rating",
            layer2_sources.esma,
            layer2_sources.esma.CATALOG_PATH,
            "completed_esma_sovereign_rating_catalogue",
            "frozen_esma_sovereign_ratings_before_market_outcomes",
            "available_at_utc",
        ),
        (
            "sec_hyperscaler",
            layer2_sources.sec,
            layer2_sources.sec.CATALOGUE_PATH,
            "completed_sec_hyperscaler_8k_catalogue",
            "frozen_sec_hyperscaler_8k_catalogue_before_market_outcomes",
            "acceptance_datetime_utc",
        ),
    )
    for source, module, catalogue_path, result_status, freeze_status, anchor_column in layer2_specs:
        _, _, frame = layer2_sources.validate_source(
            result_path=module.RESULT_PATH,
            freeze_path=module.FREEZE_PATH,
            catalogue_path=catalogue_path,
            result_status=result_status,
            freeze_status=freeze_status,
        )
        if source == "esma_euro_signed_rating":
            frame = frame.loc[
                frame["jurisdiction_group"].eq("selected_euro_area")
                & ~frame["source_action_sign"].eq("unsigned_or_lifecycle_action")
            ]
        rows.extend(
            {"anchor_utc": value, "source_group": source}
            for value in pd.to_datetime(frame[anchor_column], utc=True)
        )
        source_artifacts.append(
            {
                "source": source,
                **_source_artifacts(module, catalogue_path),
            }
        )

    external = DataFrame.from_records(rows)
    external["anchor_utc"] = pd.to_datetime(external["anchor_utc"], utc=True)
    external = external.drop_duplicates(["anchor_utc", "source_group"]).sort_values(
        ["anchor_utc", "source_group"], kind="stable"
    )
    return external.reset_index(drop=True), source_artifacts


def _within_collision(anchor: pd.Timestamp, blocked_ns: np.ndarray) -> bool:
    if not len(blocked_ns):
        return False
    distance = np.abs(blocked_ns - pd.Timestamp(anchor).value)
    return bool(np.min(distance) <= pd.Timedelta(hours=COLLISION_HOURS).value)


def _timestamp_ns(values: Sequence[pd.Timestamp]) -> np.ndarray:
    """Normalize pandas 3 microsecond arrays to Timestamp nanoseconds."""

    return np.asarray([pd.Timestamp(value).value for value in values], dtype=np.int64)


def _any_offset_collision(anchor: pd.Timestamp, blocked_ns: np.ndarray) -> bool:
    return any(
        _within_collision(anchor + pd.Timedelta(minutes=offset), blocked_ns)
        for offset in OFFSETS
    )


def add_event_collision_flags(catalogue: DataFrame, external: DataFrame) -> DataFrame:
    output = catalogue.copy()
    china_ns = _timestamp_ns(output["anchor_utc"].tolist())
    external_ns = _timestamp_ns(external["anchor_utc"].tolist())
    flags: list[bool] = []
    for position, anchor in enumerate(output["anchor_utc"]):
        other_china = np.delete(china_ns, position)
        blocked = np.concatenate([other_china, external_ns])
        flags.append(_any_offset_collision(pd.Timestamp(anchor), blocked))
    output["other_frozen_event_within_4h"] = flags
    return output


def build_control_map(catalogue: DataFrame, external: DataFrame) -> DataFrame:
    blocked_ns = np.concatenate(
        [
            _timestamp_ns(catalogue["anchor_utc"].tolist()),
            _timestamp_ns(external["anchor_utc"].tolist()),
        ]
    )
    rows: list[dict[str, Any]] = []
    for event in catalogue.itertuples(index=False):
        rank = 0
        for weeks in range(1, CONTROL_SEARCH_WEEKS + 1):
            candidate = pd.Timestamp(event.anchor_utc) - pd.Timedelta(weeks=weeks)
            if _any_offset_collision(candidate, blocked_ns):
                continue
            rank += 1
            rows.append(
                {
                    "event_id": str(event.event_id),
                    "event_family": str(event.event_family),
                    "event_anchor_utc": pd.Timestamp(event.anchor_utc),
                    "control_anchor_utc": candidate,
                    "control_rank": rank,
                    "weeks_before_event": weeks,
                    "control_type": "prior_same_weekday_and_utc_clock",
                }
            )
            if rank == CONTROL_COUNT:
                break
        if rank != CONTROL_COUNT:
            raise ValueError(f"Regional event has only {rank} clean controls: {event.event_id}")
    controls = DataFrame.from_records(rows)
    if controls.duplicated(["event_id", "control_rank"]).any():
        raise ValueError("Regional control slots are duplicated")
    return controls


def candidate_anchor_rows(catalogue: DataFrame, controls: DataFrame) -> DataFrame:
    control_groups = {
        str(event_id): group.sort_values("control_rank", kind="stable")
        for event_id, group in controls.groupby("event_id", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for event in catalogue.itertuples(index=False):
        anchors = [pd.Timestamp(event.anchor_utc)]
        anchors.extend(control_groups[str(event.event_id)]["control_anchor_utc"].tolist())
        if len(anchors) != CANDIDATE_COUNT:
            raise ValueError(f"Regional matched set is incomplete: {event.event_id}")
        for slot, base_anchor in enumerate(anchors):
            for offset in OFFSETS:
                rows.append(
                    {
                        "event_id": str(event.event_id),
                        "event_family": str(event.event_family),
                        "whole_event_partition": str(event.whole_event_partition),
                        "collision_clean": not bool(event.other_frozen_event_within_4h),
                        "candidate_slot": slot,
                        "candidate_kind": "event" if slot == 0 else "control",
                        "base_anchor_utc": pd.Timestamp(base_anchor),
                        "offset_minutes": offset,
                        "analysis_anchor_utc": pd.Timestamp(base_anchor)
                        + pd.Timedelta(minutes=offset),
                    }
                )
    return DataFrame.from_records(rows)


def load_market_frames() -> tuple[dict[str, DataFrame], DataFrame]:
    frames: dict[str, DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for pair in ASSETS:
        path = g0.ohlcv_path(pair, "1m")
        frame = g0.load_ohlcv(path).sort_values("date", kind="stable")
        duplicates = int(frame["date"].duplicated().sum())
        frame = frame.drop_duplicates("date").reset_index(drop=True)
        differences = frame["date"].diff().dt.total_seconds().div(60)
        rows.append(
            {
                "pair": pair,
                "timeframe": "1m",
                "path": str(path.resolve()),
                "sha256": g0.sha256_file(path),
                "rows": len(frame),
                "first_candle_utc": frame["date"].min(),
                "last_candle_utc": frame["date"].max(),
                "duplicate_timestamps_removed": duplicates,
                "gaps_over_one_minute": int(differences.gt(1).sum()),
                "maximum_gap_minutes": float(differences.max()),
            }
        )
        frames[pair] = frame
    return frames, DataFrame.from_records(rows)


def extract_candidate_metrics(
    candidate_anchors: DataFrame, frames: Mapping[str, DataFrame]
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for pair, frame in frames.items():
        positions = shared.position_by_date(frame)
        for candidate in candidate_anchors.itertuples(index=False):
            anchor = pd.Timestamp(candidate.analysis_anchor_utc)
            position = positions.get(anchor)
            for horizon in HORIZONS:
                if position is None:
                    status = "missing_anchor_candle"
                    metrics = None
                else:
                    status = central_activity.minute_window_status(frame, position, horizon)
                    metrics = (
                        central_activity.contiguous_window_metrics(frame, position, horizon)
                        if status == "usable"
                        else None
                    )
                row = {
                    "event_id": candidate.event_id,
                    "event_family": candidate.event_family,
                    "whole_event_partition": candidate.whole_event_partition,
                    "collision_clean": candidate.collision_clean,
                    "candidate_slot": int(candidate.candidate_slot),
                    "candidate_kind": candidate.candidate_kind,
                    "base_anchor_utc": candidate.base_anchor_utc,
                    "offset_minutes": int(candidate.offset_minutes),
                    "analysis_anchor_utc": anchor,
                    "pair": pair,
                    "horizon_minutes": horizon,
                    "window_status": status,
                    "absolute_movement": np.nan,
                    "full_range": np.nan,
                    "volume": np.nan,
                }
                if metrics is not None:
                    row.update(
                        {
                            "absolute_movement": float(metrics["abs_return"]),
                            "full_range": float(metrics["range"]),
                            "volume": float(metrics["volume"]),
                        }
                    )
                rows.append(row)
    return DataFrame.from_records(rows)


def symmetric_candidate_scores(metrics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(metrics, dtype=float)
    if values.shape != (CANDIDATE_COUNT, len(METRIC_COLUMNS)):
        raise ValueError("Regional candidate metrics must have shape (13, 3)")
    scores = np.full(CANDIDATE_COUNT, np.nan, dtype=float)
    ratios = np.full_like(values, np.nan)
    if not np.isfinite(values).all() or np.any(values < 0):
        return scores, ratios
    for candidate in range(CANDIDATE_COUNT):
        controls = np.delete(values, candidate, axis=0)
        baselines = np.median(controls, axis=0)
        if not np.isfinite(baselines).all() or np.any(baselines <= 0):
            continue
        ratios[candidate] = values[candidate] / baselines
        scores[candidate] = float(np.median(ratios[candidate]))
    return scores, ratios


def score_candidate_sets(metrics: DataFrame) -> tuple[DataFrame, DataFrame]:
    output = metrics.copy()
    for column in ("absolute_movement_ratio", "full_range_ratio", "volume_ratio"):
        output[column] = np.nan
    output["activity_score"] = np.nan
    records: list[dict[str, Any]] = []
    group_columns = ["event_id", "pair", "horizon_minutes", "offset_minutes"]
    for key, group in output.groupby(group_columns, sort=True):
        ordered = group.sort_values("candidate_slot", kind="stable")
        if ordered["candidate_slot"].tolist() != list(range(CANDIDATE_COUNT)):
            raise ValueError(f"Regional candidate slots changed: {key}")
        scores, ratios = symmetric_candidate_scores(ordered.loc[:, METRIC_COLUMNS].to_numpy())
        output.loc[ordered.index, "activity_score"] = scores
        output.loc[ordered.index, "absolute_movement_ratio"] = ratios[:, 0]
        output.loc[ordered.index, "full_range_ratio"] = ratios[:, 1]
        output.loc[ordered.index, "volume_ratio"] = ratios[:, 2]
        first = ordered.iloc[0]
        record = {
            "event_id": str(first.event_id),
            "event_family": str(first.event_family),
            "whole_event_partition": str(first.whole_event_partition),
            "collision_clean": bool(first.collision_clean),
            "pair": str(first.pair),
            "horizon_minutes": int(first.horizon_minutes),
            "offset_minutes": int(first.offset_minutes),
        }
        record.update(dict(zip(SCORE_COLUMNS, scores, strict=True)))
        records.append(record)
    return output, DataFrame.from_records(records)


def studentized_log_activity(values: Sequence[float], *, minimum_events: int) -> float:
    scores = np.asarray(values, dtype=float)
    scores = scores[np.isfinite(scores) & (scores > 0)]
    if len(scores) < minimum_events or len(scores) < 2:
        return np.nan
    logs = np.log(scores)
    standard_error = float(np.std(logs, ddof=1) / np.sqrt(len(logs)))
    if not np.isfinite(standard_error) or standard_error <= np.finfo(float).eps:
        return np.nan
    return float(np.mean(logs) / standard_error)


def summarize_cells(scores: DataFrame, score_column: str = SCORE_COLUMNS[0]) -> DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = [
        "event_family",
        "pair",
        "horizon_minutes",
        "offset_minutes",
        "whole_event_partition",
    ]
    for key, group in scores.groupby(group_columns, sort=True):
        family, pair, horizon, offset, partition = key
        for view in VIEWS:
            selected = group if view == "all_events" else group.loc[group["collision_clean"]]
            values = pd.to_numeric(selected[score_column], errors="coerce")
            values = values[np.isfinite(values)]
            support = len(values)
            required = MINIMUM_EVENTS[str(partition)]
            median = float(values.median()) if support else np.nan
            rate = float(values.gt(1.0).mean()) if support else np.nan
            rows.append(
                {
                    "event_family": family,
                    "pair": pair,
                    "horizon_minutes": int(horizon),
                    "offset_minutes": int(offset),
                    "whole_event_partition": partition,
                    "view": view,
                    "whole_events": support,
                    "minimum_whole_events": required,
                    "median_activity_score": median,
                    "above_control_rate": rate,
                    "studentized_log_activity": studentized_log_activity(
                        values, minimum_events=required
                    ),
                    "direct_cell_pass": bool(
                        support >= required
                        and np.isfinite(median)
                        and median >= DIRECT_MEDIAN[int(offset)]
                        and np.isfinite(rate)
                        and rate >= DIRECT_RATE[int(offset)]
                    ),
                }
            )
    return DataFrame.from_records(rows)


def _offset_direct_pass(cells: DataFrame, offset: int, *, strict: bool = False) -> bool:
    selected = cells.loc[cells["offset_minutes"].eq(offset)]
    if len(selected) != len(PARTITIONS) * len(VIEWS):
        return False
    median_minimum = DIRECT_MEDIAN[0] if strict else DIRECT_MEDIAN[offset]
    rate_minimum = DIRECT_RATE[0] if strict else DIRECT_RATE[offset]
    return bool(
        selected["whole_events"].ge(selected["minimum_whole_events"]).all()
        and selected["median_activity_score"].ge(median_minimum).all()
        and selected["above_control_rate"].ge(rate_minimum).all()
    )


def _weakest_offset_strength(cells: DataFrame, offset: int) -> float:
    values = cells.loc[
        cells["offset_minutes"].eq(offset), "studentized_log_activity"
    ].to_numpy(dtype=float)
    if len(values) != len(PARTITIONS) * len(VIEWS) or not np.isfinite(values).all():
        return np.nan
    return float(np.min(values))


def observed_route_decisions(cells: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    route_columns = ["event_family", "pair", "horizon_minutes"]
    for key, route in cells.groupby(route_columns, sort=True):
        family, pair, horizon = key
        nominal_pass = _offset_direct_pass(route, 0)
        adjacent_passes = {
            offset: _offset_direct_pass(route, offset) for offset in (-30, 30)
        }
        strengths = {offset: _weakest_offset_strength(route, offset) for offset in OFFSETS}
        finite_adjacent = {
            offset: strength
            for offset, strength in strengths.items()
            if offset != 0 and np.isfinite(strength)
        }
        chosen_adjacent = (
            max(finite_adjacent, key=finite_adjacent.get) if finite_adjacent else None
        )
        route_strength = (
            min(strengths[0], finite_adjacent[chosen_adjacent])
            if chosen_adjacent is not None and np.isfinite(strengths[0])
            else np.nan
        )
        shifted_only = bool(
            not nominal_pass
            and any(_offset_direct_pass(route, offset, strict=True) for offset in (-30, 30))
        )
        rows.append(
            {
                "event_family": family,
                "pair": pair,
                "horizon_minutes": int(horizon),
                "nominal_direct_gate_pass": nominal_pass,
                "minus_30_adjacent_gate_pass": adjacent_passes[-30],
                "plus_30_adjacent_gate_pass": adjacent_passes[30],
                "chosen_adjacent_offset_minutes": chosen_adjacent,
                "shifted_only_timing_mismatch": shifted_only,
                "all_direct_gates_pass": bool(
                    nominal_pass and any(adjacent_passes.values())
                ),
                "nominal_weakest_studentized_log_activity": strengths[0],
                "minus_30_weakest_studentized_log_activity": strengths[-30],
                "plus_30_weakest_studentized_log_activity": strengths[30],
                "observed_route_strength": route_strength,
            }
        )
    return DataFrame.from_records(rows)


def build_route_specs(scores: DataFrame) -> list[RouteSpec]:
    specs: list[RouteSpec] = []
    for key, route in scores.groupby(
        ["event_family", "pair", "horizon_minutes"], sort=True
    ):
        cells: dict[int, tuple[CellSpec, ...]] = {}
        for offset in OFFSETS:
            offset_cells: list[CellSpec] = []
            for partition in PARTITIONS:
                for view in VIEWS:
                    mask = route["offset_minutes"].eq(offset) & route[
                        "whole_event_partition"
                    ].eq(partition)
                    if view == "collision_clean":
                        mask &= route["collision_clean"]
                    offset_cells.append(
                        CellSpec(
                            row_indices=route.index[mask].to_numpy(dtype=int),
                            minimum_events=MINIMUM_EVENTS[partition],
                        )
                    )
            cells[offset] = tuple(offset_cells)
        specs.append(RouteSpec((str(key[0]), str(key[1]), int(key[2])), cells))
    return specs


def route_strength_from_scores(selected_scores: np.ndarray, spec: RouteSpec) -> float:
    offset_strengths: dict[int, float] = {}
    for offset, cells in spec.cells_by_offset.items():
        legs = [
            studentized_log_activity(
                selected_scores[cell.row_indices], minimum_events=cell.minimum_events
            )
            for cell in cells
        ]
        offset_strengths[offset] = (
            float(np.min(legs)) if np.isfinite(legs).all() else np.nan
        )
    adjacent = [offset_strengths[offset] for offset in (-30, 30)]
    adjacent = [value for value in adjacent if np.isfinite(value)]
    if not np.isfinite(offset_strengths[0]) or not adjacent:
        return np.nan
    return float(min(offset_strengths[0], max(adjacent)))


def event_codes(scores: DataFrame) -> tuple[np.ndarray, int]:
    events = sorted(scores["event_id"].astype(str).unique())
    lookup = {event: index for index, event in enumerate(events)}
    return scores["event_id"].astype(str).map(lookup).to_numpy(dtype=int), len(events)


def select_permuted_scores(
    candidate_matrix: np.ndarray, codes: np.ndarray, choices: np.ndarray
) -> np.ndarray:
    if len(candidate_matrix) != len(codes):
        raise ValueError("Regional candidate matrix and event-code rows differ")
    return candidate_matrix[np.arange(len(candidate_matrix)), choices[codes]]


def run_global_randomization(
    scores: DataFrame,
    specs: Sequence[RouteSpec],
    *,
    iterations: int = PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
) -> DataFrame:
    matrix = scores.loc[:, SCORE_COLUMNS].to_numpy(dtype=float)
    codes, event_count = event_codes(scores)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for iteration in range(iterations):
        choices = rng.integers(0, CANDIDATE_COUNT, size=event_count)
        selected = select_permuted_scores(matrix, codes, choices)
        strengths = [route_strength_from_scores(selected, spec) for spec in specs]
        finite = [value for value in strengths if np.isfinite(value)]
        rows.append(
            {
                "permutation": iteration + 1,
                "global_max_route_strength": max(finite) if finite else np.nan,
            }
        )
    return DataFrame.from_records(rows)


def add_familywide_decisions(routes: DataFrame, null: DataFrame) -> DataFrame:
    output = routes.copy()
    maxima = pd.to_numeric(null["global_max_route_strength"], errors="coerce").to_numpy()
    maxima = maxima[np.isfinite(maxima)]
    probabilities: list[float] = []
    exceedances: list[int | float] = []
    verdicts: list[str] = []
    for row in output.itertuples(index=False):
        if not bool(row.all_direct_gates_pass) or not np.isfinite(row.observed_route_strength):
            probabilities.append(np.nan)
            exceedances.append(np.nan)
            verdicts.append(
                "shifted_only_timing_mismatch"
                if bool(row.shifted_only_timing_mismatch)
                else "direct_gates_not_met"
            )
            continue
        count = int(np.sum(maxima >= float(row.observed_route_strength)))
        probability = float((count + 1) / (len(maxima) + 1))
        probabilities.append(probability)
        exceedances.append(count)
        verdicts.append(
            "retained_unsigned_activity_association"
            if probability <= 0.05
            else "not_retained_after_global_chance_control"
        )
    output["null_maxima_at_least_observed"] = exceedances
    output["finite_global_permutations"] = len(maxima)
    output["familywide_probability"] = probabilities
    output["familywide_gate_pass"] = (
        output["all_direct_gates_pass"]
        & output["familywide_probability"].le(0.05)
    )
    output["verdict"] = verdicts
    output["interpretation_limit"] = (
        "Unsigned activity association only; no direction, causation, profit, or trading rule."
    )
    return output


def window_coverage(metrics: DataFrame) -> DataFrame:
    return (
        metrics.groupby(
            ["pair", "horizon_minutes", "offset_minutes", "window_status"],
            dropna=False,
        )
        .size()
        .rename("candidate_windows")
        .reset_index()
    )


def render_report(
    routes: DataFrame, metrics: DataFrame, catalogue: DataFrame, controls: DataFrame
) -> str:
    retained = routes.loc[routes["familywide_gate_pass"]]
    mismatches = routes.loc[routes["shifted_only_timing_mismatch"]]
    usable = int(metrics["window_status"].eq("usable").sum())
    return "\n".join(
        [
            "# China NBS Nominal-Clock Activity Review",
            "",
            f"- Frozen whole events: `{catalogue['event_id'].nunique()}`",
            f"- Frozen controls: `{len(controls)}` (`12` per event)",
            f"- Candidate market windows: `{len(metrics)}`; usable: `{usable}`",
            f"- Complete searched routes: `{len(routes)}`",
            f"- Routes passing direct gates and global chance control: `{len(retained)}`",
            f"- Shifted-only timing mismatches: `{len(mismatches)}`",
            f"- Linked whole-event randomizations: `{PERMUTATIONS}`",
            "- Eurostat opened: **No**",
            "- Direction tested: **No**",
            "- Profit used: **No**",
            "",
            "Each event and control was scored symmetrically against the other 12 members "
            "of its frozen matched set. One reassigned candidate slot was shared across both "
            "assets, both horizons, and all clock offsets. The chance comparison used the "
            "largest route strength anywhere in the complete 3 x 2 x 2 search.",
            "",
            "This is an unsigned activity association test at uncertain nominal clocks. It "
            "does not establish direction, causation, profit, an entry, or an exit.",
            "",
        ]
    )


def validate_existing_result(result: Mapping[str, Any]) -> None:
    if result.get("status") != "completed_regional_nominal_clock_activity_test":
        raise ValueError("Existing regional activity result is invalid")
    required = {
        "controls",
        "external_collisions",
        "market_coverage",
        "window_coverage",
        "candidate_metrics",
        "candidate_scores",
        "cell_summary",
        "route_decisions",
        "global_null",
        "report",
        "analysis_script",
        "analysis_contract",
    }
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != required:
        raise ValueError("Existing regional activity result has an incomplete artifact set")
    for name, record in artifacts.items():
        validate_artifact_record(record, label=name)


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        validate_existing_result(result)
        return result
    output_paths = (
        CONTROL_PATH,
        EXTERNAL_PATH,
        COVERAGE_PATH,
        WINDOW_COVERAGE_PATH,
        METRICS_PATH,
        SCORES_PATH,
        SUMMARY_PATH,
        ROUTES_PATH,
        NULL_PATH,
        REPORT_PATH,
    )
    if not overwrite:
        partial = [str(path) for path in output_paths if path.exists()]
        if partial:
            raise FileExistsError(
                "Partial regional activity outputs exist without a result record: "
                + ", ".join(partial)
            )

    contract, catalogue = load_contract_and_sources()
    external, external_sources = load_external_anchors()
    catalogue = add_event_collision_flags(catalogue, external)
    controls = build_control_map(catalogue, external)
    anchors = candidate_anchor_rows(catalogue, controls)

    frames, coverage = load_market_frames()
    metrics = extract_candidate_metrics(anchors, frames)
    metrics, scores = score_candidate_sets(metrics)
    summary = summarize_cells(scores)
    routes = observed_route_decisions(summary)
    specs = build_route_specs(scores)
    null = run_global_randomization(scores, specs)
    routes = add_familywide_decisions(routes, null)
    windows = window_coverage(metrics)

    DIRECT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(controls, CONTROL_PATH)
    g0.atomic_write_csv(external, EXTERNAL_PATH)
    g0.atomic_write_csv(coverage, COVERAGE_PATH)
    g0.atomic_write_csv(windows, WINDOW_COVERAGE_PATH)
    g0.atomic_write_parquet(metrics, METRICS_PATH)
    g0.atomic_write_parquet(scores, SCORES_PATH)
    g0.atomic_write_csv(summary, SUMMARY_PATH)
    g0.atomic_write_csv(routes, ROUTES_PATH)
    g0.atomic_write_parquet(null, NULL_PATH)
    temporary_report = REPORT_PATH.with_suffix(f".md.{os.getpid()}.tmp")
    temporary_report.write_text(
        render_report(routes, metrics, catalogue, controls),
        encoding="utf-8",
        newline="\n",
    )
    temporary_report.replace(REPORT_PATH)
    result = {
        "schema_version": 1,
        "status": "completed_regional_nominal_clock_activity_test",
        "created_at_utc": g0.utc_now(),
        "profit_used": False,
        "direction_tested": False,
        "causal_claim_made": False,
        "eurostat_opened": False,
        "whole_events": int(catalogue["event_id"].nunique()),
        "collision_clean_events": int((~catalogue["other_frozen_event_within_4h"]).sum()),
        "controls": len(controls),
        "candidate_windows": len(metrics),
        "usable_candidate_windows": int(metrics["window_status"].eq("usable").sum()),
        "searched_routes": len(routes),
        "direct_gate_routes": int(routes["all_direct_gates_pass"].sum()),
        "familywide_retained_routes": int(routes["familywide_gate_pass"].sum()),
        "shifted_only_timing_mismatches": int(routes["shifted_only_timing_mismatch"].sum()),
        "permutations": PERMUTATIONS,
        "permutation_seed": PERMUTATION_SEED,
        "source_contracts": contract["source_contracts"],
        "external_source_artifacts": external_sources,
        "artifacts": {
            "controls": artifact(CONTROL_PATH),
            "external_collisions": artifact(EXTERNAL_PATH),
            "market_coverage": artifact(COVERAGE_PATH),
            "window_coverage": artifact(WINDOW_COVERAGE_PATH),
            "candidate_metrics": artifact(METRICS_PATH),
            "candidate_scores": artifact(SCORES_PATH),
            "cell_summary": artifact(SUMMARY_PATH),
            "route_decisions": artifact(ROUTES_PATH),
            "global_null": artifact(NULL_PATH),
            "report": artifact(REPORT_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
            "analysis_contract": artifact(CONTRACT_PATH),
        },
    }
    g0.atomic_write_json(result, RESULT_PATH)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "ready_not_executed",
                    "contract_sha256": CONTRACT_SHA256,
                    "searched_routes": len(FAMILIES) * len(ASSETS) * len(HORIZONS),
                    "permutations": PERMUTATIONS,
                    "eurostat_will_be_opened": False,
                    "direction_will_be_tested": False,
                    "profit_will_be_used": False,
                },
                indent=2,
            )
        )
        return os.EX_OK
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate the frozen SEC and central-bank activity leads family-wide."""

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
    market_event_central_bank_activity_direct as central_bank,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_layer2_source_activity_direct as source_activity,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_layer2_validation_freeze as validation_freeze,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
OUTPUT_ROOT = validation_freeze.OUTPUT_ROOT / "activity_family_randomization_20260908a"
SCORE_PATH = OUTPUT_ROOT / "activity_candidate_scores.parquet"
OBSERVED_PATH = OUTPUT_ROOT / "activity_observed_route_statistics.csv"
NULL_PATH = OUTPUT_ROOT / "activity_family_null_maxima.parquet"
DECISION_PATH = OUTPUT_ROOT / "activity_family_decisions.csv"
REPORT_PATH = OUTPUT_ROOT / "activity_family_plain_review.md"
RESULT_PATH = OUTPUT_ROOT / "activity_family_result.json"

CANDIDATE_COUNT = shared.CONTROL_COUNT + 1
SCORE_COLUMNS = tuple(f"candidate_score_{index:02d}" for index in range(CANDIDATE_COUNT))
METRIC_COLUMNS = ("abs_return", "range", "volume")


@dataclass(frozen=True)
class CellSpec:
    """Precomputed rows and threshold support for one route/period/view cell."""

    row_indices: np.ndarray
    minimum_events: int


@dataclass(frozen=True)
class RouteSpec:
    """All required cells belonging to one searched route."""

    key: tuple[str, ...]
    cells: tuple[CellSpec, ...]


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def load_validation_contract() -> dict[str, Any]:
    result = json.loads(validation_freeze.RESULT_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(validation_freeze.FREEZE_PATH.read_text(encoding="utf-8"))
    if result.get("status") != "completed_layer2_validation_outcome_blind_freeze":
        raise ValueError("Layer 2 validation freeze result is not terminal")
    if freeze.get("status") != "frozen_layer2_validation_before_new_validation_outcomes":
        raise ValueError("Layer 2 validation definitions are not frozen")
    if freeze.get("new_validation_outcomes_read") or freeze.get("profit_used"):
        raise ValueError("Layer 2 validation freeze is not outcome-blind")
    if result["artifacts"]["freeze"]["sha256"] != g0.sha256_file(
        validation_freeze.FREEZE_PATH
    ):
        raise ValueError("Layer 2 validation freeze changed after it was recorded")
    for name, recorded in freeze["source_contracts"].items():
        path = Path(recorded["path"])
        if not path.is_file() or g0.sha256_file(path) != recorded["sha256"]:
            raise ValueError(f"Frozen source contract changed: {name} -> {path}")
    required = {
        "sec_activity_full_family_randomization",
        "boj_activity_full_family_randomization",
    }
    available = {str(route["branch_id"]) for route in freeze["routes"]}
    if not required.issubset(available):
        raise ValueError("Frozen activity-family validation routes are missing")
    return freeze


def candidate_activity_scores(metrics: np.ndarray) -> np.ndarray:
    """Rotate the event label within one event plus its matched controls."""

    values = np.asarray(metrics, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(METRIC_COLUMNS):
        raise ValueError("Candidate metrics must have shape (candidate, 3)")
    finite = np.isfinite(values).all(axis=1)
    scores = np.full(len(values), np.nan, dtype=float)
    for candidate in range(len(values)):
        control_mask = finite.copy()
        control_mask[candidate] = False
        if not finite[candidate] or int(control_mask.sum()) < shared.MINIMUM_CONTROLS:
            continue
        baselines = np.median(values[control_mask], axis=0)
        if not np.isfinite(baselines).all() or np.any(baselines <= 0):
            continue
        scores[candidate] = float(np.median(values[candidate] / baselines))
    return scores


def metrics_for_anchors(
    frame: DataFrame,
    positions: Mapping[pd.Timestamp, int],
    anchors: Sequence[pd.Timestamp | None],
    horizon: int,
) -> np.ndarray:
    values = np.full((len(anchors), len(METRIC_COLUMNS)), np.nan, dtype=float)
    for index, anchor in enumerate(anchors):
        if anchor is None or pd.isna(anchor):
            continue
        position = positions.get(pd.Timestamp(anchor))
        if position is None:
            continue
        metrics = central_bank.contiguous_window_metrics(frame, position, horizon)
        if metrics is not None:
            values[index] = [float(metrics[column]) for column in METRIC_COLUMNS]
    return values


def load_market_frames() -> dict[str, tuple[DataFrame, dict[pd.Timestamp, int]]]:
    loaded: dict[str, tuple[DataFrame, dict[pd.Timestamp, int]]] = {}
    for pair in central_bank.ASSETS:
        frame = (
            g0.load_ohlcv(g0.ohlcv_path(pair, "1m"))
            .sort_values("date", kind="stable")
            .drop_duplicates("date")
            .reset_index(drop=True)
        )
        loaded[pair] = (frame, shared.position_by_date(frame))
    return loaded


def sec_candidate_rows(
    frames: Mapping[str, tuple[DataFrame, Mapping[pd.Timestamp, int]]]
) -> DataFrame:
    _, events, controls = source_activity.load_frozen_inputs()
    parent = pd.read_parquet(source_activity.OUTCOME_PATH)
    event_by_id = events.set_index("event_id", drop=False)
    controls_by_event = {
        str(event_id): [
            pd.Timestamp(value)
            for value in group.sort_values("control_rank")["control_anchor_utc"]
        ]
        for event_id, group in controls.groupby("event_id", sort=False)
    }
    cache: dict[tuple[str, str, int], np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for row in parent.itertuples(index=False):
        event = event_by_id.loc[str(row.event_id)]
        anchors = [
            pd.Timestamp(event["decision_utc"]),
            *controls_by_event.get(str(row.event_id), []),
        ]
        if len(anchors) != CANDIDATE_COUNT:
            raise ValueError(f"SEC matched set is not complete: {row.event_id}")
        key = (str(row.base_event_id), str(row.pair), int(row.horizon_minutes))
        if key not in cache:
            frame, positions = frames[str(row.pair)]
            metrics = metrics_for_anchors(
                frame, positions, anchors, int(row.horizon_minutes)
            )
            cache[key] = candidate_activity_scores(metrics)
        scores = cache[key]
        record = {
            "domain": "sec_layer2_sources",
            "whole_event_id": str(row.base_event_id),
            "event_id": str(row.event_id),
            "route_family": str(row.event_source),
            "pair": str(row.pair),
            "horizon_minutes": int(row.horizon_minutes),
            "whole_event_partition": str(row.whole_event_partition),
            "collision_clean": not bool(row.other_frozen_event_within_4h),
            "parent_activity_score": float(row.activity_score),
        }
        record.update(dict(zip(SCORE_COLUMNS, scores, strict=True)))
        records.append(record)
    return DataFrame.from_records(records)


def central_control_anchors(
    frame: DataFrame,
    positions: Mapping[pd.Timestamp, int],
    *,
    anchor: pd.Timestamp,
    horizon: int,
    blocked_anchors: Sequence[pd.Timestamp],
) -> list[pd.Timestamp]:
    """Mirror the parent control search while retaining each selected timestamp."""

    controls: list[pd.Timestamp] = []
    for weeks in range(1, shared.CONTROL_SEARCH_WEEKS + 1):
        control_anchor = anchor - pd.Timedelta(weeks=weeks)
        if (
            shared.minimum_event_distance_hours(blocked_anchors, control_anchor)
            <= central_bank.COLLISION_HOURS
        ):
            continue
        position = positions.get(control_anchor)
        if position is None:
            continue
        if central_bank.contiguous_window_metrics(frame, position, horizon) is not None:
            controls.append(control_anchor)
        if len(controls) >= shared.CONTROL_COUNT:
            break
    return controls


def central_bank_candidate_rows(
    frames: Mapping[str, tuple[DataFrame, Mapping[pd.Timestamp, int]]]
) -> DataFrame:
    _, catalogue = central_bank.load_frozen_inputs()
    external, _ = central_bank.load_external_anchors()
    catalogue = central_bank.add_collision_flags(catalogue, external)
    parent = pd.read_parquet(central_bank.OUTCOME_PATH)
    event_by_id = catalogue.set_index("event_id", drop=False)
    blocked = [*catalogue["available_at_utc"].tolist(), *external]
    slot_contract: dict[str, tuple[pd.Timestamp, ...]] = {}
    records: list[dict[str, Any]] = []
    for row in parent.itertuples(index=False):
        event = event_by_id.loc[str(row.event_id)]
        anchor = pd.Timestamp(event["available_at_utc"])
        frame, positions = frames[str(row.pair)]
        controls = central_control_anchors(
            frame,
            positions,
            anchor=anchor,
            horizon=int(row.horizon_minutes),
            blocked_anchors=blocked,
        )
        if len(controls) == shared.CONTROL_COUNT:
            signature = tuple(controls)
            previous = slot_contract.setdefault(str(row.event_id), signature)
            if signature != previous:
                raise ValueError(
                    "Central-bank matched candidate slots differ across related rows: "
                    f"{row.event_id}"
                )
        padded: list[pd.Timestamp | None] = [anchor, *controls]
        padded.extend([None] * (CANDIDATE_COUNT - len(padded)))
        metrics = metrics_for_anchors(
            frame, positions, padded[:CANDIDATE_COUNT], int(row.horizon_minutes)
        )
        scores = candidate_activity_scores(metrics)
        record = {
            "domain": "scheduled_central_banks",
            "whole_event_id": str(row.event_id),
            "event_id": str(row.event_id),
            "route_family": str(row.bank),
            "pair": str(row.pair),
            "horizon_minutes": int(row.horizon_minutes),
            "whole_event_partition": str(row.whole_event_partition),
            "collision_clean": not bool(row.other_major_scheduled_event_within_4h),
            "parent_activity_score": float(row.activity_score),
        }
        record.update(dict(zip(SCORE_COLUMNS, scores, strict=True)))
        records.append(record)
    return DataFrame.from_records(records)


def compare_parent_scores(rows: DataFrame) -> dict[str, int | float]:
    """Verify old eligible rows and quantify mechanically added market coverage."""

    parent = pd.to_numeric(rows["parent_activity_score"], errors="coerce").to_numpy()
    reproduced = pd.to_numeric(rows[SCORE_COLUMNS[0]], errors="coerce").to_numpy()
    comparable = np.isfinite(parent) & np.isfinite(reproduced)
    if not comparable.any():
        raise ValueError("No parent activity scores could be reproduced")
    parent_only = np.isfinite(parent) & ~np.isfinite(reproduced)
    newly_eligible = ~np.isfinite(parent) & np.isfinite(reproduced)
    if parent_only.any():
        raise ValueError("Current market data lost rows that were eligible in the parent test")
    absolute_error = np.abs(parent[comparable] - reproduced[comparable])
    return {
        "comparable_rows": int(comparable.sum()),
        "newly_eligible_rows": int(newly_eligible.sum()),
        "parent_finite_current_missing_rows": int(parent_only.sum()),
        "existing_rows_with_changed_score": int(np.sum(absolute_error > 1e-10)),
        "max_absolute_error_on_comparable_rows": float(np.max(absolute_error)),
    }


def build_route_specs(
    rows: DataFrame,
    *,
    domain: str,
    partitions: Sequence[str],
    variants: Sequence[str],
    minimum_events: Mapping[tuple[str, str], int],
) -> list[RouteSpec]:
    selected = rows.loc[rows["domain"].eq(domain)].reset_index(drop=True)
    specs: list[RouteSpec] = []
    for keys, route in selected.groupby(
        ["route_family", "pair", "horizon_minutes"], sort=True
    ):
        family, pair, horizon = keys
        cells: list[CellSpec] = []
        for variant in variants:
            for partition in partitions:
                mask = route["whole_event_partition"].eq(partition)
                if variant.startswith("exclude_"):
                    mask &= route["collision_clean"]
                original_indices = route.index[mask].to_numpy(dtype=int)
                unique_events = route.loc[mask, "whole_event_id"].nunique()
                if unique_events != len(original_indices):
                    raise ValueError("A route cell contains duplicate whole events")
                required = int(minimum_events[(str(family), str(partition))])
                cells.append(CellSpec(original_indices, required))
        specs.append(
            RouteSpec((str(family), str(pair), str(int(horizon))), tuple(cells))
        )
    return specs


def route_statistic(
    scores: np.ndarray,
    cells: Sequence[CellSpec],
    *,
    minimum_score: float,
    minimum_rate: float,
) -> float:
    """Return the weakest normalized gate across every required route cell."""

    margins: list[float] = []
    for cell in cells:
        values = np.asarray(scores[cell.row_indices], dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < cell.minimum_events:
            return np.nan
        margins.extend(
            [
                float(np.median(values)) / minimum_score,
                float(np.mean(values > 1.0)) / minimum_rate,
            ]
        )
    return float(min(margins)) if margins else np.nan


def route_statistics(
    scores: np.ndarray,
    specs: Sequence[RouteSpec],
    *,
    minimum_score: float,
    minimum_rate: float,
) -> dict[tuple[str, ...], float]:
    return {
        spec.key: route_statistic(
            scores,
            spec.cells,
            minimum_score=minimum_score,
            minimum_rate=minimum_rate,
        )
        for spec in specs
    }


def sec_connected_family_statistics(
    statistics: Mapping[tuple[str, ...], float],
) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for (family, _pair, horizon), statistic in statistics.items():
        grouped.setdefault((family, horizon), []).append(float(statistic))
    output: dict[tuple[str, str], float] = {}
    for key, values in grouped.items():
        finite = np.asarray(values, dtype=float)
        output[key] = (
            float(np.min(finite))
            if len(finite) == len(central_bank.ASSETS) and np.isfinite(finite).all()
            else np.nan
        )
    return output


def selected_scores(
    matrix: np.ndarray, event_codes: np.ndarray, choices: np.ndarray
) -> np.ndarray:
    if len(matrix) != len(event_codes):
        raise ValueError("Candidate matrix and event-code rows differ")
    return matrix[np.arange(len(matrix)), choices[event_codes]]


def event_codes(rows: DataFrame) -> tuple[np.ndarray, int]:
    events = sorted(rows["whole_event_id"].astype(str).unique())
    lookup = {event: index for index, event in enumerate(events)}
    codes = rows["whole_event_id"].astype(str).map(lookup).to_numpy(dtype=int)
    return codes, len(events)


def observed_route_table(
    sec_stats: Mapping[tuple[str, ...], float],
    bank_stats: Mapping[tuple[str, ...], float],
) -> DataFrame:
    records = [
        {
            "domain": "sec_layer2_sources",
            "route_family": key[0],
            "pair": key[1],
            "horizon_minutes": int(key[2]),
            "gate_margin_statistic": value,
            "passes_unchanged_gates": bool(np.isfinite(value) and value >= 1.0),
        }
        for key, value in sec_stats.items()
    ]
    records.extend(
        {
            "domain": "scheduled_central_banks",
            "route_family": key[0],
            "pair": key[1],
            "horizon_minutes": int(key[2]),
            "gate_margin_statistic": value,
            "passes_unchanged_gates": bool(np.isfinite(value) and value >= 1.0),
        }
        for key, value in bank_stats.items()
    )
    return DataFrame.from_records(records)


def run_randomization(
    sec_rows: DataFrame,
    bank_rows: DataFrame,
    sec_specs: Sequence[RouteSpec],
    bank_specs: Sequence[RouteSpec],
    *,
    sec_minimum_score: float,
    sec_minimum_rate: float,
    bank_minimum_score: float,
    bank_minimum_rate: float,
    iterations: int,
    seed: int,
) -> DataFrame:
    sec_matrix = sec_rows.loc[:, SCORE_COLUMNS].to_numpy(dtype=float)
    bank_matrix = bank_rows.loc[:, SCORE_COLUMNS].to_numpy(dtype=float)
    sec_codes, sec_event_count = event_codes(sec_rows)
    bank_codes, bank_event_count = event_codes(bank_rows)
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for permutation in range(iterations):
        sec_choices = rng.integers(0, CANDIDATE_COUNT, size=sec_event_count)
        bank_choices = rng.integers(0, CANDIDATE_COUNT, size=bank_event_count)
        sec_scores = selected_scores(sec_matrix, sec_codes, sec_choices)
        bank_scores = selected_scores(bank_matrix, bank_codes, bank_choices)
        sec_routes = route_statistics(
            sec_scores,
            sec_specs,
            minimum_score=sec_minimum_score,
            minimum_rate=sec_minimum_rate,
        )
        sec_families = sec_connected_family_statistics(sec_routes)
        bank_routes = route_statistics(
            bank_scores,
            bank_specs,
            minimum_score=bank_minimum_score,
            minimum_rate=bank_minimum_rate,
        )
        sec_finite = [value for value in sec_families.values() if np.isfinite(value)]
        bank_finite = [value for value in bank_routes.values() if np.isfinite(value)]
        records.append(
            {
                "permutation": permutation + 1,
                "sec_max_connected_family_statistic": (
                    max(sec_finite) if sec_finite else np.nan
                ),
                "boj_max_searched_route_statistic": (
                    max(bank_finite) if bank_finite else np.nan
                ),
            }
        )
    return DataFrame.from_records(records)


def empirical_p_value(observed: float, null_values: Sequence[float]) -> tuple[int, int, float]:
    values = np.asarray(null_values, dtype=float)
    values = values[np.isfinite(values)]
    if not np.isfinite(observed) or not len(values):
        return 0, len(values), np.nan
    exceedances = int(np.sum(values >= observed))
    return exceedances, len(values), float((exceedances + 1) / (len(values) + 1))


def decision_table(
    sec_observed: float,
    bank_observed: float,
    null: DataFrame,
) -> DataFrame:
    records: list[dict[str, Any]] = []
    definitions = (
        (
            "sec_activity_full_family_randomization",
            "SEC earnings-like filing activity, connected BTC/ETH 15-minute family",
            sec_observed,
            "sec_max_connected_family_statistic",
        ),
        (
            "boj_activity_full_family_randomization",
            "Bank of Japan decision activity, ETH 60-minute route",
            bank_observed,
            "boj_max_searched_route_statistic",
        ),
    )
    for branch_id, plain_name, observed, null_column in definitions:
        exceedances, finite, p_value = empirical_p_value(observed, null[null_column])
        original_gates = bool(np.isfinite(observed) and observed >= 1.0)
        family_pass = bool(original_gates and np.isfinite(p_value) and p_value <= 0.05)
        records.append(
            {
                "branch_id": branch_id,
                "plain_name": plain_name,
                "observed_gate_margin_statistic": observed,
                "unchanged_original_gates_pass": original_gates,
                "null_maximum_column": null_column,
                "finite_permutations": finite,
                "null_maxima_at_least_observed": exceedances,
                "familywise_p_value": p_value,
                "familywise_gate_pass": family_pass,
                "verdict": (
                    "retained_after_familywide_randomization"
                    if family_pass
                    else "not_retained_after_familywide_randomization"
                ),
                "interpretation_limit": (
                    "Activity association only; no direction, causation, profit, or trading rule."
                ),
            }
        )
    return DataFrame.from_records(records)


def render_report(
    decisions: DataFrame,
    null: DataFrame,
    comparisons: Mapping[str, Mapping[str, int | float]],
) -> str:
    rows = [
        "# Layer 2 Activity-Family Validation",
        "",
        "This checks whether the two provisional activity findings remain unusual after "
        "accounting for every related route searched in their original batches.",
        "",
        "| Question | Original gates | Family-adjusted probability | Decision |",
        "|---|---:|---:|---|",
    ]
    for row in decisions.itertuples(index=False):
        probability = (
            f"{row.familywise_p_value:.3%}"
            if np.isfinite(row.familywise_p_value)
            else "unavailable"
        )
        rows.append(
            f"| {row.plain_name} | "
            f"{'pass' if row.unchanged_original_gates_pass else 'fail'} | "
            f"{probability} | {row.verdict} |"
        )
    rows.extend(
        [
            "",
            f"- Whole-event matched-time randomizations: `{len(null)}`.",
            "- BTC/ETH and overlapping horizons kept the same shuffled event choice.",
            "- SEC nested all-filing and earnings-only rows also kept the same whole-event choice.",
            "- The SEC null compared every connected BTC/ETH source/window family; the "
            "central-bank null compared all 24 bank/market/window routes.",
            "- Largest SEC parent-score reconstruction error on unchanged eligible rows: "
            f"`{comparisons['sec']['max_absolute_error_on_comparable_rows']:.3g}`.",
            "- Largest central-bank reconstruction error on unchanged eligible rows: "
            f"`{comparisons['central_bank']['max_absolute_error_on_comparable_rows']:.3g}`.",
            "- Newly usable central-bank rows from later mechanical OHLCV backfill: "
            f"`{comparisons['central_bank']['newly_eligible_rows']}`. The original nominated "
            "route and all gates stayed fixed; the added rows were included symmetrically in "
            "the observed and shuffled tests.",
            "- Previously usable central-bank rows whose matched-control score changed after "
            "that backfill: "
            f"`{comparisons['central_bank']['existing_rows_with_changed_score']}`. This occurs "
            "because newly restored nearer control weeks replace more distant control weeks.",
            "- This is an activity test only. It says nothing about up/down direction or profit.",
            "",
        ]
    )
    return "\n".join(rows)


def validate_existing_result(result: Mapping[str, Any]) -> None:
    if result.get("status") != "completed_layer2_activity_family_validation":
        raise ValueError("Existing Layer 2 activity-family result is invalid")
    for recorded in result.get("artifacts", {}).values():
        path = Path(recorded["path"])
        if not path.is_file() or g0.sha256_file(path) != recorded["sha256"]:
            raise ValueError(f"Existing validation artifact changed: {path}")


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        validate_existing_result(result)
        return result

    validation = load_validation_contract()
    iterations = int(validation_freeze.PERMUTATION_ITERATIONS)
    seed = int(validation["permutation_seed"])
    frames = load_market_frames()
    sec_rows = sec_candidate_rows(frames)
    bank_rows = central_bank_candidate_rows(frames)
    comparisons = {
        "sec": compare_parent_scores(sec_rows),
        "central_bank": compare_parent_scores(bank_rows),
    }
    unexplained_changes = {
        name: comparison
        for name, comparison in comparisons.items()
        if float(comparison["max_absolute_error_on_comparable_rows"]) > 1e-10
        and int(comparison["newly_eligible_rows"]) == 0
    }
    if unexplained_changes:
        raise ValueError(
            "Parent activity scores changed without added coverage: "
            f"{unexplained_changes}"
        )

    sec_freeze, _, _ = source_activity.load_frozen_inputs()
    bank_freeze, _ = central_bank.load_frozen_inputs()
    sec_minimum_events = {
        (str(family), str(partition)): int(required)
        for family, config in sec_freeze["activity_routes"].items()
        for partition, required in config["minimum_events_by_partition"].items()
    }
    bank_minimum_events = {
        (bank, partition): int(bank_freeze["market_test"]["minimum_events_per_partition"])
        for bank in ("ECB", "BoE", "BoJ")
        for partition in central_bank.PARTITIONS
    }
    sec_specs = build_route_specs(
        sec_rows,
        domain="sec_layer2_sources",
        partitions=source_activity.PARTITIONS,
        variants=source_activity.VARIANTS,
        minimum_events=sec_minimum_events,
    )
    bank_specs = build_route_specs(
        bank_rows,
        domain="scheduled_central_banks",
        partitions=central_bank.PARTITIONS,
        variants=central_bank.VARIANTS,
        minimum_events=bank_minimum_events,
    )
    sec_scores = sec_rows.loc[:, SCORE_COLUMNS[0]].to_numpy(dtype=float)
    bank_scores = bank_rows.loc[:, SCORE_COLUMNS[0]].to_numpy(dtype=float)
    sec_observed_routes = route_statistics(
        sec_scores,
        sec_specs,
        minimum_score=float(sec_freeze["activity_test"]["minimum_median_activity_score"]),
        minimum_rate=float(sec_freeze["activity_test"]["minimum_above_control_rate"]),
    )
    bank_observed_routes = route_statistics(
        bank_scores,
        bank_specs,
        minimum_score=float(bank_freeze["market_test"]["minimum_median_activity_score"]),
        minimum_rate=float(bank_freeze["market_test"]["minimum_above_control_rate"]),
    )
    sec_families = sec_connected_family_statistics(sec_observed_routes)
    sec_observed = sec_families[("sec_hyperscaler_earnings", "15")]
    bank_observed = bank_observed_routes[("BoJ", "ETH/USDT:USDT", "60")]
    null = run_randomization(
        sec_rows,
        bank_rows,
        sec_specs,
        bank_specs,
        sec_minimum_score=float(sec_freeze["activity_test"]["minimum_median_activity_score"]),
        sec_minimum_rate=float(sec_freeze["activity_test"]["minimum_above_control_rate"]),
        bank_minimum_score=float(bank_freeze["market_test"]["minimum_median_activity_score"]),
        bank_minimum_rate=float(bank_freeze["market_test"]["minimum_above_control_rate"]),
        iterations=iterations,
        seed=seed,
    )
    observed = observed_route_table(sec_observed_routes, bank_observed_routes)
    decisions = decision_table(sec_observed, bank_observed, null)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([sec_rows, bank_rows], ignore_index=True)
    g0.atomic_write_parquet(combined, SCORE_PATH)
    g0.atomic_write_csv(observed, OBSERVED_PATH)
    g0.atomic_write_parquet(null, NULL_PATH)
    g0.atomic_write_csv(decisions, DECISION_PATH)
    REPORT_PATH.write_text(
        render_report(decisions, null, comparisons), encoding="utf-8", newline="\n"
    )
    result = {
        "schema_version": 1,
        "status": "completed_layer2_activity_family_validation",
        "created_at_utc": g0.utc_now(),
        "profit_used": False,
        "direction_tested": False,
        "validation_routes_completed": [
            "sec_activity_full_family_randomization",
            "boj_activity_full_family_randomization",
        ],
        "permutations": iterations,
        "seed": seed,
        "sec_whole_events": int(sec_rows["whole_event_id"].nunique()),
        "central_bank_whole_events": int(bank_rows["whole_event_id"].nunique()),
        "parent_score_comparison": comparisons,
        "decisions": decisions.to_dict(orient="records"),
        "source_contracts": {
            "validation_freeze": artifact(validation_freeze.FREEZE_PATH),
            "validation_freeze_result": artifact(validation_freeze.RESULT_PATH),
        },
        "artifacts": {
            "candidate_scores": artifact(SCORE_PATH),
            "observed_routes": artifact(OBSERVED_PATH),
            "null_maxima": artifact(NULL_PATH),
            "decisions": artifact(DECISION_PATH),
            "report": artifact(REPORT_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
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
                    "validation_routes": [
                        "sec_activity_full_family_randomization",
                        "boj_activity_full_family_randomization",
                    ],
                    "permutations": validation_freeze.PERMUTATION_ITERATIONS,
                    "profit_will_be_used": False,
                    "direction_will_be_tested": False,
                },
                indent=2,
            )
        )
        return os.EX_OK
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())

"""Familywise post-screen validation for the event-confirmation breadth batch.

The parent breadth review has already opened and screened market outcomes.  This
module therefore freezes that completed result first, then (only after the
freeze) runs a whole matched-set randomization.  The randomization is a ranking
diagnostic for the already-screened 544-cell universe; it is not a new
promotion rule.
"""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_confirmation_breadth_direct as direct,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
DIRECT_OUTPUT_ROOT = direct.OUTPUT_ROOT
OUTPUT_ROOT = DIRECT_OUTPUT_ROOT
DIRECT_RESULT_PATH = DIRECT_OUTPUT_ROOT / "event_confirmation_breadth_direct_result.json"
CELL_SUMMARY_PATH = DIRECT_OUTPUT_ROOT / "confirmation_direction_cell_summary.csv"
RESPONSE_PARQUET_PATH = (
    Path(r"D:\FreqTradeStuffLargeData")
    / "research_outputs"
    / "event_hierarchy"
    / "event_confirmation_breadth_20260905a"
    / "event_confirmation_response_rows.parquet"
)

FREEZE_PATH = OUTPUT_ROOT / "event_confirmation_breadth_familywise_validation_freeze.json"
RESULT_PATH = OUTPUT_ROOT / "event_confirmation_breadth_familywise_validation_result.json"
PER_CELL_PATH = OUTPUT_ROOT / "event_confirmation_breadth_familywise_validation_per_cell.csv"
REPORT_PATH = OUTPUT_ROOT / "event_confirmation_breadth_familywise_validation_report.md"

CELL_COLUMNS = (
    "analysis_group",
    "route_id",
    "leader_type",
    "response_scope",
    "total_horizon_hours",
)
RESPONSE_CELL_COLUMNS = (
    "event_family",
    "event_source",
    "leader_type",
    "scope",
    "horizon_hours",
)
RANK_COUNT = 13
RANKS = tuple(range(RANK_COUNT))
CONTROL_RANKS = tuple(range(1, RANK_COUNT))
PERMUTATION_SEED = 20260905
PERMUTATION_ITERATIONS = 2000
EXPECTED_CELL_COUNT = 544
EXPECTED_PASSING_CELL_COUNT = 27
MINIMUM_CONFIRMED_EVENTS = 10
MINIMUM_EVENTS_PER_PARTITION = 3
PAIR_RATE_FLOOR = 0.55
PARTITION_RATE_FLOOR = 0.50

GROUP_ROUTE_IDS: dict[str, tuple[str, ...]] = {
    "recent_media": (
        "live_news_activity",
        "live_web_activity",
        "live_news_or_web_union",
        "live_news_and_web_overlap",
    ),
    "historical_confluence": (
        "two_plus_families",
        "three_plus_families",
        "two_plus_source_groups",
    ),
    "historical_cross_asset": (
        "market_fear",
        "technology_equities",
        "broad_us_dollar",
    ),
}
GROUPS = (*GROUP_ROUTE_IDS, "global")
PARTITIONS_BY_GROUP: dict[str, tuple[str, ...]] = {
    "recent_media": (
        "development_2026_06_01_to_07_15",
        "validation_2026_07_16_to_08_30",
    ),
    "historical_confluence": (
        "development_2021_2023",
        "internal_validation_2024_2025",
    ),
    "historical_cross_asset": (
        "development_2021_2023",
        "internal_validation_2024_2025",
    ),
}


@dataclass(frozen=True)
class PreparedData:
    """Dense matched-set representation used by observed and null statistics."""

    cells: DataFrame
    event_ids: tuple[str, ...]
    event_partitions: np.ndarray
    present: np.ndarray
    usable: np.ndarray
    confirmed: np.ndarray
    success: np.ndarray


@dataclass(frozen=True)
class MetricArrays:
    """Per-cell metrics for one common event-level rank assignment."""

    selected_present: np.ndarray
    selected_usable_count: np.ndarray
    confirmed_selected_count: np.ndarray
    confirmed_control_event_sets: np.ndarray
    pair_rate: np.ndarray
    event_only_rate: np.ndarray
    confirmation_only_rate: np.ndarray
    statistic: np.ndarray
    partition_counts: dict[str, np.ndarray]
    partition_rates: dict[str, np.ndarray]


def artifact(path: Path) -> dict[str, Any]:
    """Return the path and content hash used by the freeze/result contracts."""

    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def _truthy(values: Series) -> Series:
    """Parse the boolean spellings used by parquet and CSV artifacts."""

    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    lowered = values.astype("string").str.strip().str.lower()
    return lowered.isin(("true", "1", "yes", "y"))


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.full(numerator.shape, np.nan, dtype=float)
    np.divide(numerator, denominator, out=output, where=denominator > 0)
    return output


def _partition_suffix(partition: str) -> str:
    return str(partition).replace("internal_", "").replace("_", "-")


def _cell_key(values: Sequence[Any]) -> tuple[str, str, str, str, int]:
    if len(values) != len(CELL_COLUMNS):
        raise ValueError("A cell key must contain exactly five fields.")
    group, route, leader, scope, horizon = values
    try:
        parsed_horizon = int(horizon)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid cell horizon: {horizon!r}") from exc
    return (str(group), str(route), str(leader), str(scope), parsed_horizon)


def _summary_keys(summary: DataFrame) -> list[tuple[str, str, str, str, int]]:
    missing = [column for column in CELL_COLUMNS if column not in summary.columns]
    if missing:
        raise ValueError(f"Cell summary is missing key columns: {missing}")
    keys = [
        _cell_key(values)
        for values in summary.loc[:, list(CELL_COLUMNS)].itertuples(
            index=False, name=None
        )
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("Cell summary contains duplicate cell keys.")
    return keys


def _response_keys(response: DataFrame) -> list[tuple[str, str, str, str, int]]:
    missing = [column for column in RESPONSE_CELL_COLUMNS if column not in response.columns]
    if missing:
        raise ValueError(f"Response rows are missing cell key columns: {missing}")
    return [
        _cell_key(values)
        for values in response.loc[:, list(RESPONSE_CELL_COLUMNS)].itertuples(
            index=False, name=None
        )
    ]


def _canonical_cell_universe(keys: Sequence[tuple[Any, ...]]) -> list[str]:
    return [json.dumps(list(key), separators=(",", ":")) for key in sorted(keys)]


def _cell_universe_hash(keys: Sequence[tuple[Any, ...]]) -> str:
    digest = hashlib.sha256()
    for key in _canonical_cell_universe(keys):
        digest.update(key.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_direct_result(direct_result: Mapping[str, Any]) -> None:
    if direct_result.get("status") != "completed_event_confirmation_breadth_direct":
        raise ValueError("Confirmation breadth direct result is not terminal.")
    if not bool(direct_result.get("outcomes_opened")):
        raise ValueError("Post-screen validation requires opened parent outcomes.")
    if int(direct_result.get("cell_count", -1)) != EXPECTED_CELL_COUNT:
        raise ValueError(
            "Confirmation breadth direct result does not contain the exact 544-cell universe."
        )


def _load_direct_and_summary() -> tuple[dict[str, Any], DataFrame, list[tuple[str, ...]]]:
    if not DIRECT_RESULT_PATH.is_file():
        raise FileNotFoundError(f"Missing direct result: {DIRECT_RESULT_PATH}")
    if not CELL_SUMMARY_PATH.is_file():
        raise FileNotFoundError(f"Missing direct cell summary: {CELL_SUMMARY_PATH}")
    direct_result = json.loads(DIRECT_RESULT_PATH.read_text(encoding="utf-8"))
    _validate_direct_result(direct_result)
    summary = pd.read_csv(CELL_SUMMARY_PATH)
    keys = _summary_keys(summary)
    if len(keys) != EXPECTED_CELL_COUNT:
        raise ValueError(
            f"Cell summary contains {len(keys)} cells; expected {EXPECTED_CELL_COUNT}."
        )
    expected_passing = int(_truthy(summary["meets_conditional_direction_rule"]).sum())
    if expected_passing != EXPECTED_PASSING_CELL_COUNT:
        raise ValueError(
            f"Cell summary contains {expected_passing} direct passes; expected "
            f"{EXPECTED_PASSING_CELL_COUNT}."
        )
    return direct_result, summary, keys


def build_freeze_plan() -> dict[str, Any]:
    """Build the terminal pre-randomization plan without reading outcome rows."""

    direct_result, summary, keys = _load_direct_and_summary()
    if not RESPONSE_PARQUET_PATH.is_file():
        raise FileNotFoundError(f"Missing response parquet: {RESPONSE_PARQUET_PATH}")
    input_artifacts = {
        "direct_result": artifact(DIRECT_RESULT_PATH),
        "cell_summary": artifact(CELL_SUMMARY_PATH),
        "response_parquet": artifact(RESPONSE_PARQUET_PATH),
    }
    routes = sorted({key[1] for key in keys})
    passing_count = int(_truthy(summary["meets_conditional_direction_rule"]).sum())
    cell_universe = _canonical_cell_universe(keys)
    return {
        "schema_version": 1,
        "objective": "objective_02b_market_reaction_zone_discovery",
        "status": "frozen_post_screen_before_familywise_randomization",
        "created_at_utc": g0.utc_now(),
        "analysis_script": artifact(ANALYSIS_PATH),
        "outcomes_already_opened": True,
        "parent_direct_result": input_artifacts["direct_result"],
        "input_artifacts": input_artifacts,
        "artifacts": input_artifacts,
        "cell_universe": {
            "count": EXPECTED_CELL_COUNT,
            "passing_count": EXPECTED_PASSING_CELL_COUNT,
            "key_columns": list(CELL_COLUMNS),
            "keys": cell_universe,
            "sha256": _cell_universe_hash(keys),
        },
        "cell_count": EXPECTED_CELL_COUNT,
        "passing_cell_count": passing_count,
        "routes": routes,
        "groups": {name: list(route_ids) for name, route_ids in GROUP_ROUTE_IDS.items()}
        | {"global": routes},
        "randomization": {
            "unit": "whole_matched_event_set",
            "rank_values": list(RANKS),
            "actual_rank": 0,
            "control_ranks": list(CONTROL_RANKS),
            "same_choice_across_cells": True,
            "seed": PERMUTATION_SEED,
            "permutations": PERMUTATION_ITERATIONS,
            "missing_selected_row": "unusable_abstain",
        },
        "eligibility": {
            "minimum_confirmed_selected_events": MINIMUM_CONFIRMED_EVENTS,
            "minimum_confirmed_control_event_sets": MINIMUM_CONFIRMED_EVENTS,
            "minimum_confirmed_selected_events_per_partition": MINIMUM_EVENTS_PER_PARTITION,
            "pair_rate_floor": PAIR_RATE_FLOOR,
            "partition_rate_floor": PARTITION_RATE_FLOOR,
            "rotated_q75_applied": False,
        },
        "diagnostic_boundary": {
            "purpose": "post_screen_familywise_robustness_and_ranking_diagnostic",
            "is_new_promotion_rule": False,
            "validation_partition_previously_screened": True,
            "promotion_statement": (
                "Familywise probabilities describe the already-screened cell ranking; "
                "they do not promote a route, entry, exit, or trading rule."
            ),
        },
        "direct_result_status": direct_result.get("status"),
    }


def _plan_inputs(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    inputs = plan.get("input_artifacts")
    if isinstance(inputs, Mapping):
        return inputs
    inputs = plan.get("artifacts")
    if isinstance(inputs, Mapping):
        return inputs
    raise ValueError("Familywise freeze has no input artifact hashes.")


def _validate_plan_hashes(plan: Mapping[str, Any]) -> None:
    if plan.get("status") != "frozen_post_screen_before_familywise_randomization":
        raise ValueError("Familywise validation freeze is not terminal.")
    if plan.get("outcomes_already_opened") is not True:
        raise ValueError("Familywise validation freeze must record opened outcomes.")
    if int(plan.get("cell_count", -1)) != EXPECTED_CELL_COUNT:
        raise ValueError("Familywise freeze does not preserve the exact 544-cell universe.")
    inputs = _plan_inputs(plan)
    expected_paths = {
        "direct_result": DIRECT_RESULT_PATH,
        "cell_summary": CELL_SUMMARY_PATH,
        "response_parquet": RESPONSE_PARQUET_PATH,
    }
    for name, expected_path in expected_paths.items():
        metadata = inputs.get(name)
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Familywise freeze is missing {name} hash.")
        path = Path(str(metadata.get("path", expected_path)))
        if not path.is_file():
            raise FileNotFoundError(f"Frozen {name} is missing: {path}")
        if str(metadata.get("sha256")) != g0.sha256_file(path):
            raise ValueError(f"Frozen {name} hash changed: {path}")
        if name == "direct_result" and path.resolve() != expected_path.resolve():
            raise ValueError("Frozen direct-result path does not match configured input.")
        if name == "cell_summary" and path.resolve() != expected_path.resolve():
            raise ValueError("Frozen cell-summary path does not match configured input.")
        if name == "response_parquet" and path.resolve() != expected_path.resolve():
            raise ValueError("Frozen response-parquet path does not match configured input.")


def freeze_plan(*, overwrite: bool = False) -> dict[str, Any]:
    """Write the pre-randomization plan; this function never randomizes."""

    if FREEZE_PATH.is_file() and not overwrite:
        plan = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        _validate_plan_hashes(plan)
        return plan
    plan = build_freeze_plan()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(plan, FREEZE_PATH)
    return plan


def _normalise_response(response: DataFrame) -> DataFrame:
    required = {
        "event_id",
        "whole_event_partition",
        "sample_type",
        "control_rank",
        "confirmed",
        "response_usable",
        "direction_usable",
        "continued",
        *RESPONSE_CELL_COLUMNS,
    }
    missing = sorted(required.difference(response.columns))
    if missing:
        raise ValueError(f"Response rows are missing required columns: {missing}")
    output = response.copy()
    output["event_id"] = output["event_id"].astype(str)
    output["sample_type"] = output["sample_type"].astype(str).str.lower()
    ranks = pd.to_numeric(output["control_rank"], errors="coerce")
    if ranks.isna().any() or (ranks % 1).ne(0).any():
        raise ValueError("Response rows contain a non-integral control rank.")
    output["rank"] = ranks.astype(int)
    bad_event_rank = output.loc[
        output["sample_type"].eq("event") & output["rank"].ne(0)
    ]
    bad_control_rank = output.loc[
        output["sample_type"].eq("control")
        & ~output["rank"].isin(CONTROL_RANKS)
    ]
    if not bad_event_rank.empty or not bad_control_rank.empty:
        raise ValueError("Response rows do not follow rank0-event/ranks1-12-control contract.")
    output["cell_key"] = _response_keys(output)
    output["usable_flag"] = (
        _truthy(output["response_usable"])
        & _truthy(output["direction_usable"])
        & output["continued"].notna()
    ).to_numpy(dtype=bool)
    output["confirmed_flag"] = _truthy(output["confirmed"]).to_numpy(dtype=bool)
    output["success_flag"] = _truthy(output["continued"]).to_numpy(dtype=bool)
    duplicate_columns = ["cell_key", "event_id", "rank"]
    if output.duplicated(duplicate_columns).any():
        raise ValueError("Response rows contain duplicate cell/event/rank observations.")
    return output


def prepare_data(summary: DataFrame, response: DataFrame) -> PreparedData:
    """Build the dense cell x event x rank matrix used by all statistics."""

    summary = summary.copy()
    summary_keys = _summary_keys(summary)
    response = _normalise_response(response)
    response_keys = set(response["cell_key"])
    summary_key_set = set(summary_keys)
    if response_keys != summary_key_set:
        missing = sorted(summary_key_set.difference(response_keys))
        extra = sorted(response_keys.difference(summary_key_set))
        raise ValueError(
            "Response rows and the exact 544-cell universe differ: "
            f"missing={len(missing)}, extra={len(extra)}."
        )

    cell_index = {key: index for index, key in enumerate(summary_keys)}
    event_ids = tuple(sorted(response["event_id"].unique()))
    event_index = {event_id: index for index, event_id in enumerate(event_ids)}
    cell_count = len(summary_keys)
    event_count = len(event_ids)
    present = np.zeros((cell_count, event_count, RANK_COUNT), dtype=bool)
    usable = np.zeros_like(present)
    confirmed = np.zeros_like(present)
    success = np.zeros_like(present)
    event_partitions: list[str | None] = [None] * event_count
    for row in response.itertuples(index=False):
        cell = cell_index[row.cell_key]
        event = event_index[row.event_id]
        rank = int(row.rank)
        partition = str(row.whole_event_partition)
        prior_partition = event_partitions[event]
        if prior_partition is not None and prior_partition != partition:
            raise ValueError(f"Event {row.event_id} has multiple whole-event partitions.")
        event_partitions[event] = partition
        present[cell, event, rank] = True
        usable[cell, event, rank] = bool(row.usable_flag)
        confirmed[cell, event, rank] = bool(row.confirmed_flag)
        success[cell, event, rank] = bool(row.success_flag)
    if any(partition is None for partition in event_partitions):
        raise ValueError("At least one event has no partition metadata.")
    return PreparedData(
        cells=summary.reset_index(drop=True),
        event_ids=event_ids,
        event_partitions=np.asarray(event_partitions, dtype=str),
        present=present,
        usable=usable,
        confirmed=confirmed,
        success=success,
    )


def _normalise_choices(
    event_ids: Sequence[str], choices: Mapping[str, int] | Sequence[int] | None
) -> np.ndarray:
    if choices is None:
        values = np.zeros(len(event_ids), dtype=int)
    elif isinstance(choices, Mapping):
        try:
            values = np.asarray([choices[event_id] for event_id in event_ids], dtype=int)
        except KeyError as exc:
            raise ValueError(f"No pseudo-event rank was provided for {exc.args[0]}") from exc
    else:
        values = np.asarray(choices, dtype=int)
        if values.shape != (len(event_ids),):
            raise ValueError("Pseudo-event choices must contain one rank per event_id.")
    if np.any(~np.isin(values, RANKS)):
        raise ValueError("Pseudo-event ranks must be integers from 0 through 12.")
    return values


def _expected_partitions(
    group: str,
    event_partitions: np.ndarray,
    partitions_by_group: Mapping[str, Sequence[str]] | None,
) -> tuple[str, ...]:
    configured = PARTITIONS_BY_GROUP if partitions_by_group is None else partitions_by_group
    if group in configured:
        return tuple(str(partition) for partition in configured[group])
    observed = tuple(sorted({str(value) for value in event_partitions}))
    return observed


def _group_masks(cells: DataFrame) -> dict[str, np.ndarray]:
    routes = cells["route_id"].astype(str)
    masks = {
        name: routes.isin(route_ids).to_numpy(dtype=bool)
        for name, route_ids in GROUP_ROUTE_IDS.items()
    }
    masks["global"] = np.ones(len(cells), dtype=bool)
    unknown = sorted(set(routes) - {route for ids in GROUP_ROUTE_IDS.values() for route in ids})
    if unknown:
        raise ValueError(f"Cell summary contains routes without a familywise group: {unknown}")
    return masks


def metric_arrays(
    prepared: PreparedData,
    choices: Mapping[str, int] | Sequence[int] | None = None,
    *,
    partitions_by_group: Mapping[str, Sequence[str]] | None = None,
) -> MetricArrays:
    """Calculate the three rates and eligibility ingredients for one assignment."""

    selected_ranks = _normalise_choices(prepared.event_ids, choices)
    cell_positions = np.arange(len(prepared.cells))
    event_positions = np.arange(len(prepared.event_ids))
    selected_present = prepared.present[:, event_positions, selected_ranks]
    selected_usable = prepared.usable[:, event_positions, selected_ranks]
    selected_confirmed = (
        prepared.confirmed[:, event_positions, selected_ranks] & selected_usable
    )
    selected_success = prepared.success[:, event_positions, selected_ranks]
    selected_confirmed_success = selected_confirmed & selected_success

    confirmed_usable = prepared.confirmed & prepared.usable
    all_confirmed_counts = confirmed_usable.sum(axis=2).astype(int)
    all_confirmed_success = (confirmed_usable & prepared.success).sum(axis=2).astype(int)
    nonselected_counts = all_confirmed_counts - selected_confirmed.astype(int)
    nonselected_success = all_confirmed_success - selected_confirmed_success.astype(int)
    # A missing selected row abstains for the complete matched event set, so it
    # cannot silently contribute only its remaining controls to confirmation_only.
    nonselected_counts = np.where(selected_present, nonselected_counts, 0)
    nonselected_success = np.where(selected_present, nonselected_success, 0)
    per_event_control_rate = _safe_divide(
        nonselected_success.astype(float), nonselected_counts.astype(float)
    )
    control_event_mask = nonselected_counts > 0
    confirmation_only_rate = np.full(len(prepared.cells), np.nan, dtype=float)
    for cell in cell_positions:
        values = per_event_control_rate[cell, control_event_mask[cell]]
        if values.size:
            confirmation_only_rate[cell] = float(np.nanmean(values))

    pair_count = selected_confirmed.sum(axis=1).astype(int)
    pair_success = selected_confirmed_success.sum(axis=1).astype(float)
    pair_rate = _safe_divide(pair_success, pair_count.astype(float))
    event_count = selected_usable.sum(axis=1).astype(int)
    event_success = (selected_usable & selected_success).sum(axis=1).astype(float)
    event_only_rate = _safe_divide(event_success, event_count.astype(float))
    statistic = pair_rate - np.maximum(event_only_rate, confirmation_only_rate)

    partition_counts: dict[str, np.ndarray] = {}
    partition_rates: dict[str, np.ndarray] = {}
    all_partitions = sorted({str(value) for value in prepared.event_partitions})
    for partition in all_partitions:
        event_mask = prepared.event_partitions == partition
        selected_partition = selected_confirmed & event_mask[None, :]
        selected_partition_success = selected_confirmed_success & event_mask[None, :]
        counts = selected_partition.sum(axis=1).astype(int)
        rates = _safe_divide(
            selected_partition_success.sum(axis=1).astype(float), counts.astype(float)
        )
        partition_counts[partition] = counts
        partition_rates[partition] = rates

    return MetricArrays(
        selected_present=selected_present,
        selected_usable_count=event_count,
        confirmed_selected_count=pair_count,
        confirmed_control_event_sets=control_event_mask.sum(axis=1).astype(int),
        pair_rate=pair_rate,
        event_only_rate=event_only_rate,
        confirmation_only_rate=confirmation_only_rate,
        statistic=statistic,
        partition_counts=partition_counts,
        partition_rates=partition_rates,
    )


def _eligible_array(
    prepared: PreparedData,
    metrics: MetricArrays,
    *,
    partitions_by_group: Mapping[str, Sequence[str]] | None = None,
) -> np.ndarray:
    masks = _group_masks(prepared.cells)
    eligible = (
        (metrics.confirmed_selected_count >= MINIMUM_CONFIRMED_EVENTS)
        & (metrics.confirmed_control_event_sets >= MINIMUM_CONFIRMED_EVENTS)
        & (metrics.pair_rate >= PAIR_RATE_FLOOR)
    )
    for group, cell_mask in masks.items():
        if group == "global":
            continue
        partitions = _expected_partitions(
            group, prepared.event_partitions, partitions_by_group
        )
        for partition in partitions:
            counts = metrics.partition_counts.get(partition)
            rates = metrics.partition_rates.get(partition)
            if counts is None or rates is None:
                eligible[cell_mask] = False
                continue
            eligible[cell_mask] &= counts[cell_mask] >= MINIMUM_EVENTS_PER_PARTITION
            eligible[cell_mask] &= rates[cell_mask] >= PARTITION_RATE_FLOOR
    eligible &= np.isfinite(metrics.statistic)
    return eligible


def cell_statistics(
    prepared: PreparedData,
    choices: Mapping[str, int] | Sequence[int] | None = None,
    *,
    partitions_by_group: Mapping[str, Sequence[str]] | None = None,
) -> DataFrame:
    """Return readable rates/statistic/eligibility for one common assignment."""

    metrics = metric_arrays(
        prepared, choices, partitions_by_group=partitions_by_group
    )
    eligible = _eligible_array(
        prepared, metrics, partitions_by_group=partitions_by_group
    )
    records: list[dict[str, Any]] = []
    for index, row in prepared.cells.iterrows():
        record = {
            **{column: row[column] for column in CELL_COLUMNS},
            "selected_usable_event_count": int(metrics.selected_usable_count[index]),
            "confirmed_selected_event_count": int(
                metrics.confirmed_selected_count[index]
            ),
            "confirmed_control_event_count": int(
                metrics.confirmed_control_event_sets[index]
            ),
            "pair_rate": float(metrics.pair_rate[index]),
            "event_only_rate": float(metrics.event_only_rate[index]),
            "confirmation_only_rate": float(metrics.confirmation_only_rate[index]),
            "pair_continuation_rate": float(metrics.pair_rate[index]),
            "event_only_continuation_rate": float(metrics.event_only_rate[index]),
            "confirmation_only_continuation_rate": float(
                metrics.confirmation_only_rate[index]
            ),
            "statistic": float(metrics.statistic[index]),
            "basic_eligible": bool(eligible[index]),
        }
        for partition, counts in metrics.partition_counts.items():
            suffix = _partition_suffix(partition)
            record[f"{suffix}_count"] = int(counts[index])
            record[f"{suffix}_rate"] = float(metrics.partition_rates[partition][index])
        records.append(record)
    return DataFrame.from_records(records)


def randomization_maxima(
    prepared: PreparedData,
    *,
    iterations: int = PERMUTATION_ITERATIONS,
    seed: int = PERMUTATION_SEED,
    partitions_by_group: Mapping[str, Sequence[str]] | None = None,
) -> DataFrame:
    """Run whole matched-set rank randomization and retain group maxima only."""

    if iterations <= 0:
        raise ValueError("The randomization requires at least one permutation.")
    rng = np.random.default_rng(seed)
    masks = _group_masks(prepared.cells)
    maxima = {group: np.full(iterations, np.nan, dtype=float) for group in GROUPS}
    for index in range(iterations):
        choices = rng.integers(0, RANK_COUNT, size=len(prepared.event_ids))
        metrics = metric_arrays(
            prepared, choices, partitions_by_group=partitions_by_group
        )
        eligible = _eligible_array(
            prepared, metrics, partitions_by_group=partitions_by_group
        )
        for group, group_mask in masks.items():
            values = metrics.statistic[group_mask & eligible]
            if values.size:
                maxima[group][index] = float(np.max(values))
    return DataFrame(
        {
            "permutation": np.arange(1, iterations + 1, dtype=int),
            **{group: maxima[group] for group in GROUPS},
        }
    )


# Explicit alias for callers/tests that use the task wording.
whole_event_randomization = randomization_maxima


def wilson_interval(successes: int, total: int, *, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for the add-one familywise probability."""

    if total <= 0 or successes < 0 or successes > total:
        return math.nan, math.nan
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def _familywise_comparison(
    observed: float, null_values: Series | np.ndarray
) -> tuple[int, float, float, float]:
    values = np.asarray(null_values, dtype=float)
    finite = values[np.isfinite(values)]
    if not math.isfinite(observed):
        return 0, math.nan, math.nan, math.nan
    exceedances = int(np.count_nonzero(finite >= observed))
    total = len(values) + 1
    numerator = exceedances + 1
    probability = numerator / total
    lower, upper = wilson_interval(numerator, total)
    return exceedances, probability, lower, upper


def _observed_result_rows(
    summary: DataFrame,
    observed: DataFrame,
    maxima: DataFrame,
) -> DataFrame:
    passing = summary.loc[
        _truthy(summary["meets_conditional_direction_rule"])
    ].copy()
    if len(passing) != EXPECTED_PASSING_CELL_COUNT:
        raise ValueError(
            f"Expected the original 27 passing cells, found {len(passing)}."
        )
    observed_join = observed.set_index(list(CELL_COLUMNS))
    null_join = maxima.drop(columns="permutation")
    output: list[dict[str, Any]] = []
    for row in passing.itertuples(index=False):
        key = _cell_key([getattr(row, column) for column in CELL_COLUMNS])
        key_values = dict(zip(CELL_COLUMNS, key, strict=True))
        observed_row = observed_join.loc[key]
        route_id = str(key_values["route_id"])
        group = next(
            name
            for name, routes in GROUP_ROUTE_IDS.items()
            if route_id in routes
        )
        group_values = null_join[group].to_numpy(dtype=float)
        global_values = null_join["global"].to_numpy(dtype=float)
        group_exceed, group_p, group_low, group_high = _familywise_comparison(
            _finite_float(observed_row["statistic"]), group_values
        )
        global_exceed, global_p, global_low, global_high = _familywise_comparison(
            _finite_float(observed_row["statistic"]), global_values
        )
        result = {
            **key_values,
            "original_direct_pass": True,
            "observed_pair_rate": _finite_float(observed_row["pair_rate"]),
            "observed_event_only_rate": _finite_float(observed_row["event_only_rate"]),
            "observed_confirmation_only_rate": _finite_float(
                observed_row["confirmation_only_rate"]
            ),
            "observed_statistic": _finite_float(observed_row["statistic"]),
            "observed_basic_eligible": bool(observed_row["basic_eligible"]),
            "familywise_group": group,
            "group_null_max_exceedances": group_exceed,
            "group_familywise_p_value": group_p,
            "group_familywise_wilson_lower": group_low,
            "group_familywise_wilson_upper": group_high,
            "global_null_max_exceedances": global_exceed,
            "global_familywise_p_value": global_p,
            "global_familywise_wilson_lower": global_low,
            "global_familywise_wilson_upper": global_high,
            # Short aliases make the per-cell artifact convenient to inspect.
            "familywise_p_value": group_p,
            "familywise_wilson_lower": group_low,
            "familywise_wilson_upper": group_high,
            "direct_meets_65_target": bool(
                _truthy(pd.Series([getattr(row, "meets_65_target", False)]))[0]
            ),
        }
        output.append(result)
    return DataFrame.from_records(output)


def _json_number(value: Any) -> float | None:
    number = _finite_float(value)
    return number if math.isfinite(number) else None


def _json_maxima(maxima: DataFrame) -> dict[str, list[float | None]]:
    return {
        group: [_json_number(value) for value in maxima[group].to_numpy(dtype=float)]
        for group in GROUPS
    }


def report_text(per_cell: DataFrame, maxima: DataFrame) -> str:
    group_survivors = int((per_cell["group_familywise_p_value"] <= 0.05).sum())
    global_survivors = int((per_cell["global_familywise_p_value"] <= 0.05).sum())
    lines = [
        "# Event-confirmation breadth familywise validation",
        "",
        (
            "This post-screen ranking diagnostic reran the completed 544-cell breadth "
            "universe under whole matched-event-set rank randomization. Each event_id "
            "received one common rank for every leader, scope, and horizon cell."
        ),
        "",
        f"- Original direct passing cells: `{len(per_cell)}`",
        f"- Null permutations: `{len(maxima)}`; seed: `{PERMUTATION_SEED}`",
        "- Statistic: `pair_rate - max(event_only_rate, confirmation_only_rate)`",
        "- Rotated q75 was not applied inside the randomization.",
        f"- Passes after the route-family search penalty: `{group_survivors}`.",
        f"- Passes after the complete 544-cell search penalty: `{global_survivors}`.",
        "",
        "| Family | Finite null maxima | Null maximum median | Null maximum 95th percentile |",
        "|---|---:|---:|---:|",
    ]
    for group in GROUPS:
        values = maxima[group].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        median = float(np.median(finite)) if finite.size else math.nan
        percentile = float(np.quantile(finite, 0.95)) if finite.size else math.nan
        median_text = f"{median:.4f}" if math.isfinite(median) else "n/a"
        percentile_text = f"{percentile:.4f}" if math.isfinite(percentile) else "n/a"
        lines.append(
            f"| `{group}` | {len(finite)} | {median_text} | {percentile_text} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "The validation partition was previously screened by the parent direct "
                "review. These familywise probabilities are therefore a robustness and "
                "ranking diagnostic for the already-opened result, not an independent "
                "confirmation and not a new promotion or trading rule."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    """Validate the frozen parent artifacts and execute the fixed 2,000 draws."""

    if not FREEZE_PATH.is_file():
        raise FileNotFoundError(
            f"Familywise validation is not frozen; run --freeze first: {FREEZE_PATH}"
        )
    plan = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    _validate_plan_hashes(plan)
    direct_result, summary, summary_keys = _load_direct_and_summary()
    response = pd.read_parquet(RESPONSE_PARQUET_PATH)
    prepared = prepare_data(summary, response)
    if len(prepared.cells) != EXPECTED_CELL_COUNT:
        raise ValueError("Prepared response matrix does not preserve 544 cells.")
    maxima = randomization_maxima(
        prepared, iterations=PERMUTATION_ITERATIONS, seed=PERMUTATION_SEED
    )
    observed = cell_statistics(prepared, choices=np.zeros(len(prepared.event_ids), dtype=int))
    per_cell = _observed_result_rows(summary, observed, maxima)

    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if result.get("status") != "completed_post_screen_familywise_validation":
            raise ValueError("Existing familywise validation result is not terminal.")
        return result

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(per_cell, PER_CELL_PATH)
    REPORT_PATH.write_text(report_text(per_cell, maxima), encoding="utf-8", newline="\n")
    result = {
        "schema_version": 1,
        "status": "completed_post_screen_familywise_validation",
        "created_at_utc": g0.utc_now(),
        "analysis_script": artifact(ANALYSIS_PATH),
        "freeze_contract": artifact(FREEZE_PATH),
        "parent_direct_result": artifact(DIRECT_RESULT_PATH),
        "input_artifacts": {
            "direct_result": artifact(DIRECT_RESULT_PATH),
            "cell_summary": artifact(CELL_SUMMARY_PATH),
            "response_parquet": artifact(RESPONSE_PARQUET_PATH),
        },
        "cell_universe": {
            "count": len(summary_keys),
            "passing_count": len(per_cell),
            "key_columns": list(CELL_COLUMNS),
            "sha256": _cell_universe_hash(summary_keys),
        },
        "randomization": {
            "unit": "whole_matched_event_set",
            "rank_values": list(RANKS),
            "actual_rank": 0,
            "control_ranks": list(CONTROL_RANKS),
            "same_choice_across_cells": True,
            "seed": PERMUTATION_SEED,
            "permutations": PERMUTATION_ITERATIONS,
            "missing_selected_row": "unusable_abstain",
        },
        "eligibility": {
            "minimum_confirmed_selected_events": MINIMUM_CONFIRMED_EVENTS,
            "minimum_confirmed_control_event_sets": MINIMUM_CONFIRMED_EVENTS,
            "minimum_confirmed_selected_events_per_partition": MINIMUM_EVENTS_PER_PARTITION,
            "pair_rate_floor": PAIR_RATE_FLOOR,
            "partition_rate_floor": PARTITION_RATE_FLOOR,
            "rotated_q75_applied": False,
        },
        "null_maxima": _json_maxima(maxima),
        "familywise_p_le_0_05_count": int(
            (per_cell["group_familywise_p_value"] <= 0.05).sum()
        ),
        "global_familywise_p_le_0_05_count": int(
            (per_cell["global_familywise_p_value"] <= 0.05).sum()
        ),
        "minimum_familywise_p_value": float(
            per_cell["group_familywise_p_value"].min()
        ),
        "minimum_global_familywise_p_value": float(
            per_cell["global_familywise_p_value"].min()
        ),
        "diagnostic_boundary": {
            "purpose": "post_screen_familywise_robustness_and_ranking_diagnostic",
            "is_new_promotion_rule": False,
            "validation_partition_previously_screened": True,
            "promotion_statement": (
                "Familywise probabilities are not a new promotion rule and do not "
                "promote a route, entry, exit, or trading strategy."
            ),
        },
        "artifacts": {
            "per_cell": artifact(PER_CELL_PATH),
            "report": artifact(REPORT_PATH),
        },
        "parent_result_status": direct_result.get("status"),
    }
    g0.atomic_write_json(result, RESULT_PATH)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.freeze:
        print(json.dumps(freeze_plan(overwrite=args.overwrite), indent=2))
        return 0
    if args.execute:
        print(json.dumps(execute(overwrite=args.overwrite), indent=2))
        return 0
    print(
        json.dumps(
            {
                "status": "ready_not_executed",
                "next": "run --freeze before --execute",
                "freeze_path": str(FREEZE_PATH),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

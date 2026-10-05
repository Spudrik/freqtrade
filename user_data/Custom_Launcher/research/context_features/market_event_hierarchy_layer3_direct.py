"""Run the frozen Layer 3 event-hierarchy pairwise comparisons."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import math
import sys
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_layer2_direct as layer2,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_layer2_freeze as layer2_freeze,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_layer3_freeze as frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
FREEZE_RESULT = frozen.OUTPUT_ROOT / "layer3_freeze_result.json"
FREEZE_PATH = frozen.OUTPUT_ROOT / "layer3_freeze.json"
ACTIVE_PAIRS_PATH = frozen.OUTPUT_ROOT / "layer3_active_pair_matrix.csv"
PARKED_PAIRS_PATH = frozen.OUTPUT_ROOT / "layer3_parked_pair_matrix.csv"
LAYER2_FREEZE_PATH = layer2_freeze.OUTPUT_ROOT / "layer2_freeze.json"
OUTPUT_ROOT = frozen.OUTPUT_ROOT / "direct_review_20260904a"
DETAIL_ROOT = (
    Path(r"D:\FreqTradeStuffLargeData")
    / "research_outputs"
    / "event_hierarchy"
    / "layer3_pairwise_20260904a"
)

PRIMARY_CONTROL = "matched_prior_state"
MINIMUM_EVENTS = 10
ROTATION_COUNT = 19
PARTITIONS = ("development_2021_2023", "internal_validation_2024_2025")
HORIZON_ORDER = (1, 2, 4, 8, 24)


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def load_frozen_inputs() -> tuple[dict[str, Any], DataFrame, DataFrame, dict[str, Any]]:
    result = json.loads(FREEZE_RESULT.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if result.get("status") != "completed_layer3_pairwise_freeze":
        raise ValueError("Layer 3 freeze result is not terminal.")
    if freeze.get("status") != "frozen_before_layer3_pair_outcomes":
        raise ValueError("Layer 3 questions were not frozen before outcomes.")
    if result.get("new_pair_outcomes_read"):
        raise ValueError("Layer 3 freeze says pair outcomes were already opened.")
    for name, path in (
        ("freeze", FREEZE_PATH),
        ("active_pairs", ACTIVE_PAIRS_PATH),
        ("parked_pairs", PARKED_PAIRS_PATH),
    ):
        if result["artifacts"][name]["sha256"] != g0.sha256_file(path):
            raise ValueError(f"Frozen Layer 3 artifact changed: {path}")
    parent_path = Path(freeze["parent_layer"]["result"]["path"])
    if freeze["parent_layer"]["result"]["sha256"] != g0.sha256_file(parent_path):
        raise ValueError("Layer 2 parent result changed after the Layer 3 freeze.")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    for item in freeze["source_contracts"].values():
        path = Path(item["path"])
        if item["sha256"] != g0.sha256_file(path):
            raise ValueError(f"Layer 3 source contract changed: {path}")
    return (
        freeze,
        pd.read_csv(ACTIVE_PAIRS_PATH),
        pd.read_csv(PARKED_PAIRS_PATH),
        parent,
    )


def read_parent_details(parent: dict[str, Any]) -> tuple[DataFrame, DataFrame]:
    metrics = pd.read_parquet(parent["detail_artifacts"]["pair_metrics"]["path"])
    background = pd.read_parquet(
        parent["detail_artifacts"]["background_details"]["path"]
    )
    metrics["sample_anchor_utc"] = pd.to_datetime(metrics["sample_anchor_utc"], utc=True)
    anchors = metrics[["sample_id", "sample_anchor_utc"]].drop_duplicates("sample_id")
    background = background.merge(
        anchors, on="sample_id", how="left", validate="many_to_one"
    )
    if background["sample_anchor_utc"].isna().any():
        raise ValueError("A Layer 2 background row has no matching sample timestamp.")
    return metrics, background


def _safe_rate(values: Series) -> float:
    if values.empty:
        return math.nan
    return float(pd.to_numeric(values, errors="coerce").mean())


def _quantile(values: list[float], q: float) -> float:
    usable = [value for value in values if math.isfinite(value)]
    return float(np.quantile(usable, q)) if usable else math.nan


def _rotate_columns(
    frame: DataFrame, columns: Sequence[str], grouping: Sequence[str], shift: int
) -> DataFrame:
    output = frame.sort_values("sample_anchor_utc", kind="stable").reset_index(drop=True)
    group_key: str | list[str] = grouping[0] if len(grouping) == 1 else list(grouping)
    for _, indexes in output.groupby(group_key, dropna=False, sort=False).groups.items():
        positions = np.asarray(list(indexes), dtype=int)
        if len(positions) < 2:
            continue
        values = output.loc[positions, list(columns)].to_numpy(copy=True)
        step = shift % len(positions)
        if step:
            output.loc[positions, list(columns)] = np.roll(values, step, axis=0)
    return output


def summarize_background_pairs(background: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    for source in frozen.EVENT_SOURCES:
        source_rows = background.loc[background["event_source"].eq(source)].copy()
        event = source_rows.loc[
            source_rows["sample_type"].eq("event")
            & source_rows["positive_initial_move"]
        ].copy()
        controls = source_rows.loc[
            source_rows["sample_type"].eq("control")
            & source_rows["control_type"].eq(PRIMARY_CONTROL)
            & source_rows["positive_initial_move"]
        ].copy()
        event_negative = event.loc[event["background"].eq("negative_30d_background")]
        event_positive = event.loc[event["background"].eq("positive_30d_background")]
        control_negative = controls.loc[
            controls["background"].eq("negative_30d_background")
        ]
        control_by_event = control_negative.groupby("event_id", sort=False)[
            "positive_move_faded_by_half"
        ].mean()
        pair_rate = _safe_rate(event_negative["positive_move_faded_by_half"])
        event_only_rate = _safe_rate(event["positive_move_faded_by_half"])
        background_only_rate = _safe_rate(control_by_event)
        actual_gap = pair_rate - _safe_rate(
            event_positive["positive_move_faded_by_half"]
        )
        partition_gaps: dict[str, float] = {}
        partition_minimums: dict[str, int] = {}
        for partition in PARTITIONS:
            part = event.loc[event["whole_event_partition"].eq(partition)]
            negative = part.loc[part["background"].eq("negative_30d_background")]
            positive = part.loc[part["background"].eq("positive_30d_background")]
            partition_minimums[partition] = min(len(negative), len(positive))
            partition_gaps[partition] = _safe_rate(
                negative["positive_move_faded_by_half"]
            ) - _safe_rate(positive["positive_move_faded_by_half"])
        shuffled_gaps: list[float] = []
        for shift in range(1, ROTATION_COUNT + 1):
            shuffled = _rotate_columns(
                event,
                ["background"],
                ["whole_event_partition"],
                shift,
            )
            negative = shuffled.loc[
                shuffled["background"].eq("negative_30d_background")
            ]
            positive = shuffled.loc[
                shuffled["background"].eq("positive_30d_background")
            ]
            shuffled_gaps.append(
                _safe_rate(negative["positive_move_faded_by_half"])
                - _safe_rate(positive["positive_move_faded_by_half"])
            )
        shuffle_q75 = _quantile(shuffled_gaps, 0.75)
        passes = (
            len(event_negative) >= MINIMUM_EVENTS
            and math.isfinite(background_only_rate)
            and control_by_event.size >= MINIMUM_EVENTS
            and pair_rate >= 0.55
            and pair_rate - event_only_rate >= 0.10
            and pair_rate - background_only_rate >= 0.05
            and all(partition_minimums[item] >= 3 for item in PARTITIONS)
            and all(partition_gaps[item] > 0 for item in PARTITIONS)
            and math.isfinite(shuffle_q75)
            and actual_gap > shuffle_q75
        )
        records.append(
            {
                "pair_id": f"background_event_confirmation__{source}",
                "pair_family": "background_plus_event_confirmation",
                "event_source": source,
                "negative_background_event_count": len(event_negative),
                "positive_background_event_count": len(event_positive),
                "background_only_control_event_count": int(control_by_event.size),
                "pair_fade_rate": pair_rate,
                "event_only_fade_rate": event_only_rate,
                "background_only_fade_rate": background_only_rate,
                "pair_minus_event_only": pair_rate - event_only_rate,
                "pair_minus_background_only": pair_rate - background_only_rate,
                "negative_minus_positive_event_gap": actual_gap,
                "development_gap": partition_gaps[PARTITIONS[0]],
                "validation_gap": partition_gaps[PARTITIONS[1]],
                "shuffled_gap_q75": shuffle_q75,
                "beats_shuffled": bool(
                    math.isfinite(shuffle_q75) and actual_gap > shuffle_q75
                ),
                "meets_pair_rule": bool(passes),
                "meets_65_target": bool(passes and pair_rate >= 0.65),
            }
        )
    return DataFrame.from_records(records)


def build_broad_response_rows(
    metrics: DataFrame, layer2_contract: dict[str, Any]
) -> DataFrame:
    confirmation_pairs = [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        *layer2_contract["cohorts"]["established_alts"],
    ]
    identifiers = [
        "sample_id",
        "event_id",
        "event_source",
        "whole_event_partition",
        "sample_type",
        "control_type",
        "control_rank",
        "sample_anchor_utc",
    ]
    first = metrics.loc[
        metrics["horizon_hours"].eq(1) & metrics["pair"].isin(confirmation_pairs)
    ].copy()
    first["positive"] = first["signed_return"].gt(0)
    first["negative"] = first["signed_return"].lt(0)
    confirmation = first.groupby(identifiers, dropna=False, sort=False).agg(
        confirmation_asset_count=("activity_score", "count"),
        median_first_hour_activity=("activity_score", "median"),
        median_first_hour_return=("signed_return", "median"),
        positive_fraction=("positive", "mean"),
        negative_fraction=("negative", "mean"),
    ).reset_index()
    confirmation["directional_agreement"] = confirmation[
        ["positive_fraction", "negative_fraction"]
    ].max(axis=1)
    confirmation["initial_direction"] = np.sign(
        confirmation["median_first_hour_return"]
    ).astype(int)
    confirmation["broad_confirmation"] = (
        confirmation["confirmation_asset_count"].ge(6)
        & confirmation["directional_agreement"].ge(0.60)
        & confirmation["median_first_hour_activity"].ge(1.25)
        & confirmation["initial_direction"].ne(0)
    )

    one_return = metrics.loc[metrics["horizon_hours"].eq(1), [
        "sample_id",
        "pair",
        "signed_return",
    ]].rename(columns={"signed_return": "return_1h"})
    later = metrics.loc[metrics["horizon_hours"].isin(frozen.BROAD_HORIZONS)].merge(
        one_return,
        on=["sample_id", "pair"],
        how="left",
        validate="many_to_one",
    )
    later["post_first_hour_return"] = (
        later["signed_return"].add(1).div(later["return_1h"].add(1)).sub(1)
    )
    mapping = layer2.cohort_map(layer2_contract)
    later["scope"] = later["pair"].map(mapping)
    later = later.loc[later["scope"].isin(frozen.BROAD_SCOPES)]
    response = later.groupby(
        [*identifiers, "horizon_hours", "scope"], dropna=False, sort=False
    ).agg(
        response_asset_count=("post_first_hour_return", "count"),
        median_post_first_hour_return=("post_first_hour_return", "median"),
    ).reset_index()
    minimum_assets = {"btc": 1, "eth": 1, "established_alts": 4, "memes": 3}
    response["minimum_response_assets"] = response["scope"].map(minimum_assets)
    response["response_usable"] = (
        response["response_asset_count"].ge(response["minimum_response_assets"])
        & response["median_post_first_hour_return"].notna()
    )
    output = response.merge(
        confirmation[
            [
                "sample_id",
                "confirmation_asset_count",
                "median_first_hour_activity",
                "median_first_hour_return",
                "directional_agreement",
                "initial_direction",
                "broad_confirmation",
            ]
        ],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    output["continued"] = (
        output["median_post_first_hour_return"] * output["initial_direction"]
    ).gt(0)
    return output


def _collapsed_control_rate(frame: DataFrame, value_column: str) -> tuple[int, float]:
    collapsed = frame.groupby("event_id", sort=False)[value_column].mean()
    return int(collapsed.size), _safe_rate(collapsed)


def summarize_broad_response_pairs(rows: DataFrame, active: DataFrame) -> DataFrame:
    candidates = active.loc[
        active["pair_family"].eq("event_plus_first_broad_response")
    ]
    records: list[dict[str, Any]] = []
    for candidate in candidates.itertuples(index=False):
        cell = rows.loc[
            rows["event_source"].eq(candidate.event_source)
            & rows["scope"].eq(candidate.scope)
            & rows["horizon_hours"].eq(candidate.horizon_hours)
            & rows["response_usable"]
        ].copy()
        event = cell.loc[cell["sample_type"].eq("event")]
        pair = event.loc[event["broad_confirmation"]]
        controls = cell.loc[
            cell["sample_type"].eq("control")
            & cell["control_type"].eq(PRIMARY_CONTROL)
            & cell["broad_confirmation"]
        ]
        control_count, response_only_rate = _collapsed_control_rate(controls, "continued")
        pair_rate = _safe_rate(pair["continued"])
        event_only_rate = _safe_rate(event["continued"])
        partition_rates: dict[str, float] = {}
        partition_counts: dict[str, int] = {}
        for partition in PARTITIONS:
            part = pair.loc[pair["whole_event_partition"].eq(partition)]
            partition_counts[partition] = len(part)
            partition_rates[partition] = _safe_rate(part["continued"])
        shuffled_rates: list[float] = []
        for shift in range(1, ROTATION_COUNT + 1):
            shuffled = _rotate_columns(
                event,
                ["broad_confirmation", "initial_direction"],
                ["whole_event_partition"],
                shift,
            )
            selected = shuffled.loc[shuffled["broad_confirmation"].astype(bool)]
            shuffled_rates.append(
                _safe_rate(
                    (
                        selected["median_post_first_hour_return"]
                        * pd.to_numeric(selected["initial_direction"])
                    ).gt(0)
                )
            )
        shuffle_q75 = _quantile(shuffled_rates, 0.75)
        passes = (
            len(pair) >= MINIMUM_EVENTS
            and control_count >= MINIMUM_EVENTS
            and pair_rate >= 0.55
            and pair_rate - event_only_rate >= 0.05
            and pair_rate - response_only_rate >= 0.05
            and all(partition_counts[item] >= 3 for item in PARTITIONS)
            and all(partition_rates[item] >= 0.50 for item in PARTITIONS)
            and math.isfinite(shuffle_q75)
            and pair_rate > shuffle_q75
        )
        records.append(
            {
                "pair_id": candidate.pair_id,
                "pair_family": candidate.pair_family,
                "event_source": candidate.event_source,
                "scope": candidate.scope,
                "horizon_hours": int(candidate.horizon_hours),
                "confirmed_event_count": len(pair),
                "confirmed_control_event_count": control_count,
                "pair_continuation_rate": pair_rate,
                "event_only_continuation_rate": event_only_rate,
                "response_only_continuation_rate": response_only_rate,
                "pair_minus_event_only": pair_rate - event_only_rate,
                "pair_minus_response_only": pair_rate - response_only_rate,
                "development_rate": partition_rates[PARTITIONS[0]],
                "validation_rate": partition_rates[PARTITIONS[1]],
                "shuffled_rate_q75": shuffle_q75,
                "beats_shuffled": bool(
                    math.isfinite(shuffle_q75) and pair_rate > shuffle_q75
                ),
                "meets_pair_rule": bool(passes),
                "meets_65_target": bool(passes and pair_rate >= 0.65),
            }
        )
    return DataFrame.from_records(records)


def _single_state_stats(
    frame: DataFrame, state_column: str, target_state: str, far_state: str
) -> dict[str, float | int]:
    event = frame.loc[frame["sample_type"].eq("event")].copy()
    controls = frame.loc[
        frame["sample_type"].eq("control")
        & frame["control_type"].eq(PRIMARY_CONTROL)
    ].copy()
    control_baseline = controls.groupby(["event_id", "pair"], sort=False)[
        "activity_score"
    ].median().rename("control_baseline").reset_index()
    event = event.merge(
        control_baseline, on=["event_id", "pair"], how="left", validate="many_to_one"
    )
    event["paired_success"] = event["activity_score"].gt(event["control_baseline"])

    def calculate(partition: str | None) -> tuple[float, float, float]:
        event_part = event
        control_part = controls
        if partition is not None:
            event_part = event_part.loc[event_part["whole_event_partition"].eq(partition)]
            control_part = control_part.loc[
                control_part["whole_event_partition"].eq(partition)
            ]
        event_target = event_part.loc[event_part[state_column].eq(target_state)]
        event_far = event_part.loc[event_part[state_column].eq(far_state)]
        control_target = control_part.loc[control_part[state_column].eq(target_state)]
        control_far = control_part.loc[control_part[state_column].eq(far_state)]
        event_gap = (
            event_target["activity_score"].median()
            - event_far["activity_score"].median()
        )
        control_gap = (
            control_target["activity_score"].median()
            - control_far["activity_score"].median()
        )
        return float(event_gap), float(control_gap), float(event_gap - control_gap)

    event_target = event.loc[event[state_column].eq(target_state)]
    event_far = event.loc[event[state_column].eq(far_state)]
    control_target = controls.loc[controls[state_column].eq(target_state)]
    control_far = controls.loc[controls[state_column].eq(far_state)]
    event_gap, control_gap, increment = calculate(None)
    development_increment = calculate(PARTITIONS[0])[2]
    validation_increment = calculate(PARTITIONS[1])[2]
    return {
        "target_event_count": len(event_target),
        "far_event_count": len(event_far),
        "target_control_count": len(control_target),
        "far_control_count": len(control_far),
        "paired_success_rate": _safe_rate(event_target["paired_success"]),
        "event_target_minus_far": event_gap,
        "control_target_minus_far": control_gap,
        "pair_increment_beyond_components": increment,
        "development_increment": development_increment,
        "validation_increment": validation_increment,
    }


def _cohort_state_stats(
    frame: DataFrame, state_column: str, target_state: str, far_state: str
) -> dict[str, float | int]:
    identifiers = [
        "sample_id",
        "event_id",
        "whole_event_partition",
        "sample_type",
        "control_type",
        "control_rank",
    ]
    grouped = frame.groupby(
        [*identifiers, state_column], dropna=False, sort=False
    )["activity_score"].median().unstack(state_column)
    if target_state not in grouped or far_state not in grouped:
        return {
            "target_event_count": 0,
            "far_event_count": 0,
            "target_control_count": 0,
            "far_control_count": 0,
            "paired_success_rate": math.nan,
            "event_target_minus_far": math.nan,
            "control_target_minus_far": math.nan,
            "pair_increment_beyond_components": math.nan,
            "development_increment": math.nan,
            "validation_increment": math.nan,
        }
    gaps = grouped[[target_state, far_state]].dropna().copy()
    gaps["gap"] = gaps[target_state] - gaps[far_state]
    gaps = gaps.reset_index()
    event = gaps.loc[gaps["sample_type"].eq("event")]
    controls = gaps.loc[
        gaps["sample_type"].eq("control")
        & gaps["control_type"].eq(PRIMARY_CONTROL)
    ]
    control_by_event = controls.groupby("event_id", sort=False)["gap"].median()
    event_gap = float(event["gap"].median())
    control_gap = float(control_by_event.median())

    def partition_increment(partition: str) -> float:
        event_part = event.loc[event["whole_event_partition"].eq(partition), "gap"]
        control_part = controls.loc[
            controls["whole_event_partition"].eq(partition)
        ].groupby("event_id", sort=False)["gap"].median()
        return float(event_part.median() - control_part.median())

    return {
        "target_event_count": len(event),
        "far_event_count": len(event),
        "target_control_count": int(control_by_event.size),
        "far_control_count": int(control_by_event.size),
        "paired_success_rate": _safe_rate(event["gap"].gt(0)),
        "event_target_minus_far": event_gap,
        "control_target_minus_far": control_gap,
        "pair_increment_beyond_components": event_gap - control_gap,
        "development_increment": partition_increment(PARTITIONS[0]),
        "validation_increment": partition_increment(PARTITIONS[1]),
    }


def _state_stats(
    frame: DataFrame,
    scope: str,
    state_column: str,
    target_state: str,
    far_state: str,
) -> dict[str, float | int]:
    if scope in {"btc", "eth"}:
        return _single_state_stats(frame, state_column, target_state, far_state)
    return _cohort_state_stats(frame, state_column, target_state, far_state)


def _shuffle_state_increment_q75(
    frame: DataFrame,
    scope: str,
    state_column: str,
    target_state: str,
    far_state: str,
) -> float:
    increments: list[float] = []
    for shift in range(1, ROTATION_COUNT + 1):
        shuffled = _rotate_columns(
            frame,
            [state_column],
            [
                "pair",
                "whole_event_partition",
                "sample_type",
                "control_type",
                "control_rank",
            ],
            shift,
        )
        value = _state_stats(
            shuffled, scope, state_column, target_state, far_state
        )["pair_increment_beyond_components"]
        increments.append(float(value))
    return _quantile(increments, 0.75)


def _scope_pairs(scope: str, contract: dict[str, Any]) -> list[str]:
    if scope == "btc":
        return ["BTC/USDT:USDT"]
    if scope == "eth":
        return ["ETH/USDT:USDT"]
    if scope == "established_alts":
        return list(contract["cohorts"]["established_alts"])
    if scope == "memes":
        return list(contract["cohorts"]["top_ten_traded_memes"])
    raise ValueError(f"Unknown scope: {scope}")


def _state_pair_pass(stats: dict[str, float | int], shuffle_q75: float) -> bool:
    return bool(
        int(stats["target_event_count"]) >= MINIMUM_EVENTS
        and int(stats["far_event_count"]) >= MINIMUM_EVENTS
        and int(stats["target_control_count"]) >= MINIMUM_EVENTS
        and int(stats["far_control_count"]) >= MINIMUM_EVENTS
        and float(stats["paired_success_rate"]) >= 0.55
        and float(stats["event_target_minus_far"]) > 0
        and float(stats["pair_increment_beyond_components"]) > 0
        and float(stats["development_increment"]) > 0
        and float(stats["validation_increment"]) > 0
        and math.isfinite(shuffle_q75)
        and float(stats["pair_increment_beyond_components"]) > shuffle_q75
    )


def summarize_local_state_pairs(
    metrics: DataFrame, active: DataFrame, contract: dict[str, Any]
) -> DataFrame:
    records: list[dict[str, Any]] = []
    candidates = active.loc[
        active["pair_family"].eq("event_plus_local_technical_state")
    ]
    for candidate in candidates.itertuples(index=False):
        cell = metrics.loc[
            metrics["event_source"].eq(candidate.event_source)
            & metrics["horizon_hours"].eq(candidate.horizon_hours)
            & metrics["pair"].isin(_scope_pairs(candidate.scope, contract))
            & (
                metrics["sample_type"].eq("event")
                | metrics["control_type"].eq(PRIMARY_CONTROL)
            )
        ].copy()
        stats = _state_stats(
            cell,
            candidate.scope,
            "local_level_state",
            candidate.state,
            "far_from_frozen_levels",
        )
        shuffle_q75 = _shuffle_state_increment_q75(
            cell,
            candidate.scope,
            "local_level_state",
            candidate.state,
            "far_from_frozen_levels",
        )
        passes = _state_pair_pass(stats, shuffle_q75)
        records.append(
            {
                "pair_id": candidate.pair_id,
                "pair_family": candidate.pair_family,
                "event_source": candidate.event_source,
                "scope": candidate.scope,
                "horizon_hours": int(candidate.horizon_hours),
                "state": candidate.state,
                **stats,
                "shuffled_increment_q75": shuffle_q75,
                "beats_shuffled": bool(
                    math.isfinite(shuffle_q75)
                    and float(stats["pair_increment_beyond_components"]) > shuffle_q75
                ),
                "meets_pair_rule": passes,
                "meets_65_target": bool(
                    passes and float(stats["paired_success_rate"]) >= 0.65
                ),
            }
        )
    return DataFrame.from_records(records)


def add_prior_range_position(metrics: DataFrame) -> DataFrame:
    pieces: list[DataFrame] = []
    for pair in ("BTC/USDT:USDT", "ETH/USDT:USDT"):
        frame = g0.load_ohlcv(g0.ohlcv_path(pair, "1h"))
        frame = frame.sort_values("date", kind="stable").drop_duplicates("date")
        prior_high = frame["high"].shift(1).rolling(720, min_periods=360).max()
        prior_low = frame["low"].shift(1).rolling(720, min_periods=360).min()
        width = prior_high - prior_low
        position = frame["open"].sub(prior_low).div(width.where(width.gt(0)))
        state = np.where(
            position.le(0.20) | position.ge(0.80),
            "range_edge",
            "range_middle",
        )
        state = Series(state, index=frame.index, dtype="string").where(position.notna())
        pieces.append(
            DataFrame(
                {
                    "sample_anchor_utc": frame["date"],
                    "pair": pair,
                    "prior_30d_range_position": position,
                    "prior_30d_range_state": state,
                }
            )
        )
    range_state = pd.concat(pieces, ignore_index=True)
    return metrics.merge(
        range_state,
        on=["sample_anchor_utc", "pair"],
        how="left",
        validate="many_to_one",
    )


def summarize_range_position_pairs(
    metrics: DataFrame, active: DataFrame, contract: dict[str, Any]
) -> DataFrame:
    records: list[dict[str, Any]] = []
    candidates = active.loc[
        active["pair_family"].eq("event_plus_prior_30d_range_position")
    ]
    for candidate in candidates.itertuples(index=False):
        cell = metrics.loc[
            metrics["event_source"].eq(candidate.event_source)
            & metrics["horizon_hours"].eq(candidate.horizon_hours)
            & metrics["pair"].isin(_scope_pairs(candidate.scope, contract))
            & (
                metrics["sample_type"].eq("event")
                | metrics["control_type"].eq(PRIMARY_CONTROL)
            )
        ].copy()
        stats = _state_stats(
            cell,
            candidate.scope,
            "prior_30d_range_state",
            "range_edge",
            "range_middle",
        )
        shuffle_q75 = _shuffle_state_increment_q75(
            cell,
            candidate.scope,
            "prior_30d_range_state",
            "range_edge",
            "range_middle",
        )
        passes = _state_pair_pass(stats, shuffle_q75)
        records.append(
            {
                "pair_id": candidate.pair_id,
                "pair_family": candidate.pair_family,
                "event_source": candidate.event_source,
                "scope": candidate.scope,
                "horizon_hours": int(candidate.horizon_hours),
                "state": candidate.state,
                **stats,
                "shuffled_increment_q75": shuffle_q75,
                "beats_shuffled": bool(
                    math.isfinite(shuffle_q75)
                    and float(stats["pair_increment_beyond_components"]) > shuffle_q75
                ),
                "meets_pair_rule": passes,
                "meets_65_target": bool(
                    passes and float(stats["paired_success_rate"]) >= 0.65
                ),
            }
        )
    return DataFrame.from_records(records)


def _repeated_cell_ids(summary: DataFrame) -> set[str]:
    passed = summary.loc[summary["meets_pair_rule"].astype(bool)].copy()
    repeated: set[str] = set()
    if passed.empty:
        return repeated
    group_columns = ["event_source", "scope"]
    if "state" in passed:
        group_columns.append("state")
    horizon_index = {value: index for index, value in enumerate(HORIZON_ORDER)}
    for _, group in passed.groupby(group_columns, dropna=False, sort=False):
        ordered = sorted(
            ((horizon_index[int(row.horizon_hours)], row.pair_id) for row in group.itertuples()),
            key=lambda item: item[0],
        )
        for left, right in pairwise(ordered):
            if right[0] - left[0] == 1:
                repeated.update([left[1], right[1]])
    cross_columns = ["scope", "horizon_hours"]
    if "state" in passed:
        cross_columns.append("state")
    for _, group in passed.groupby(cross_columns, dropna=False, sort=False):
        if group["event_source"].nunique() >= 2:
            repeated.update(group["pair_id"].tolist())
    return repeated


def route_decisions(
    background: DataFrame,
    broad: DataFrame,
    local: DataFrame,
    range_position: DataFrame,
    parked: DataFrame,
) -> tuple[DataFrame, set[str]]:
    records: list[dict[str, Any]] = []
    eligible_ids: set[str] = set()

    background_passed = background.loc[background["meets_pair_rule"].astype(bool)]
    background_repeats = background_passed["event_source"].nunique() >= 2
    if background_repeats:
        eligible_ids.update(background_passed["pair_id"])
    records.append(
        {
            "pair_family": "background_plus_event_confirmation",
            "verdict": (
                "retained_pairwise_lead"
                if background_repeats
                else "parked_or_provisional_no_cross_source_repeat"
            ),
            "passing_cell_count": len(background_passed),
            "layer4_eligible_cell_count": len(background_passed) if background_repeats else 0,
            "plain_result": (
                "The same background-conditioned fade relationship passed for both event sources."
                if background_repeats
                else "The background pair did not repeat across both event sources."
            ),
        }
    )

    for family, summary, label in (
        (
            "event_plus_first_broad_response",
            broad,
            "event plus first-hour broad response",
        ),
        ("event_plus_local_technical_state", local, "event plus calculated local state"),
        (
            "event_plus_prior_30d_range_position",
            range_position,
            "event plus prior-month range position",
        ),
    ):
        passed = summary.loc[summary["meets_pair_rule"].astype(bool)]
        repeated = _repeated_cell_ids(summary)
        eligible_ids.update(repeated)
        if repeated:
            verdict = "retained_pairwise_lead"
            plain = f"{len(repeated)} repeated {label} cells are eligible for Layer 4."
        elif len(passed):
            verdict = "provisional_cells_parked_no_repetition"
            plain = f"{len(passed)} isolated cells passed, but none repeated as required."
        else:
            verdict = "parked_no_incremental_pair_lead"
            plain = "No cell beat both components, both periods, and the shuffled control."
        records.append(
            {
                "pair_family": family,
                "verdict": verdict,
                "passing_cell_count": len(passed),
                "layer4_eligible_cell_count": len(repeated),
                "plain_result": plain,
            }
        )

    for row in parked.itertuples(index=False):
        records.append(
            {
                "pair_family": row.pair_family,
                "verdict": row.status,
                "passing_cell_count": 0,
                "layer4_eligible_cell_count": 0,
                "plain_result": row.reason,
            }
        )
    return DataFrame.from_records(records), eligible_ids


def render_report(decisions: DataFrame, result: dict[str, Any]) -> str:
    lines = [
        "# Event Hierarchy Layer 3 - Pairwise Review",
        "",
        f"- Frozen pair cells tested: `{result['active_pair_count']}`",
        f"- Pair families reviewed or honestly parked: `{result['route_count']}`",
        "- Profit or trade-return target used: **No**",
        "- Event direction guessed from unsigned news: **No**",
        "",
        "| Pairwise question | Decision | Plain result |",
        "| --- | --- | --- |",
    ]
    for row in decisions.itertuples(index=False):
        lines.append(
            f"| `{row.pair_family}` | `{row.verdict}` | {row.plain_result} |"
        )
    lines.extend(
        [
            "",
            "Only repeated pair cells enter the Layer 4 candidate set. "
            "Isolated passes remain parked.",
            "These are conditional research leads, not trade rules or untouched confirmation.",
            "",
        ]
    )
    return "\n".join(lines)


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    result_path = OUTPUT_ROOT / "layer3_direct_result.json"
    if result_path.is_file() and not overwrite:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed_event_hierarchy_layer3_pairwise_review":
            raise ValueError("Existing Layer 3 direct result is not terminal.")
        return result
    _freeze, active, parked, parent = load_frozen_inputs()
    metrics, background_rows = read_parent_details(parent)
    layer2_contract = json.loads(LAYER2_FREEZE_PATH.read_text(encoding="utf-8"))

    background = summarize_background_pairs(background_rows)
    broad_rows = build_broad_response_rows(metrics, layer2_contract)
    broad = summarize_broad_response_pairs(broad_rows, active)
    local = summarize_local_state_pairs(metrics, active, layer2_contract)
    range_metrics = add_prior_range_position(metrics)
    range_position = summarize_range_position_pairs(
        range_metrics, active, layer2_contract
    )
    decisions, eligible_ids = route_decisions(
        background, broad, local, range_position, parked
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    DETAIL_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {
        "background": OUTPUT_ROOT / "background_pair_summary.csv",
        "broad": OUTPUT_ROOT / "broad_confirmation_pair_summary.csv",
        "local": OUTPUT_ROOT / "local_state_pair_summary.csv",
        "range": OUTPUT_ROOT / "range_position_pair_summary.csv",
        "decisions": OUTPUT_ROOT / "layer3_route_decisions.csv",
        "eligible": OUTPUT_ROOT / "layer4_candidate_cells.csv",
        "report": OUTPUT_ROOT / "layer3_plain_review.md",
        "broad_details": DETAIL_ROOT / "broad_confirmation_rows.parquet",
    }
    g0.atomic_write_csv(background, paths["background"])
    g0.atomic_write_csv(broad, paths["broad"])
    g0.atomic_write_csv(local, paths["local"])
    g0.atomic_write_csv(range_position, paths["range"])
    g0.atomic_write_csv(decisions, paths["decisions"])
    combined = pd.concat(
        [
            background,
            broad,
            local,
            range_position,
        ],
        ignore_index=True,
        sort=False,
    )
    eligible = combined.loc[combined["pair_id"].isin(eligible_ids)].copy()
    g0.atomic_write_csv(eligible, paths["eligible"])
    g0.atomic_write_parquet(broad_rows, paths["broad_details"])

    retained = decisions["verdict"].eq("retained_pairwise_lead")
    result = {
        "schema_version": 1,
        "status": "completed_event_hierarchy_layer3_pairwise_review",
        "created_at_utc": g0.utc_now(),
        "active_pair_count": len(active),
        "route_count": len(decisions),
        "retained_route_count": int(retained.sum()),
        "parked_or_provisional_route_count": int((~retained).sum()),
        "layer4_candidate_cell_count": len(eligible),
        "profit_used": False,
        "event_direction_without_surprise_used": False,
        "freeze_contract": artifact(FREEZE_PATH),
        "analysis_script": artifact(ANALYSIS_PATH),
    }
    paths["report"].write_text(
        render_report(decisions, result), encoding="utf-8", newline="\n"
    )
    result["summary_artifacts"] = {
        name: artifact(path)
        for name, path in paths.items()
        if name != "broad_details"
    }
    result["detail_artifacts"] = {
        "broad_confirmation_rows": artifact(paths["broad_details"])
    }
    g0.atomic_write_json(result, result_path)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "ready_not_executed"}, indent=2))
        return 0
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Test frozen macro expectation surprises against short BTC and ETH direction."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_macro_expectation_direction_freeze as frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_scheduled_macro_direction_direct as prior_direct,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
OUTPUT_ROOT = frozen.OUTPUT_ROOT / "direct_review_20260911a"
DETAIL_ROOT = OUTPUT_ROOT / "details"
OUTCOME_PATH = DETAIL_ROOT / "macro_expectation_direction_rows.parquet"
SUMMARY_PATH = OUTPUT_ROOT / "macro_expectation_direction_summary.csv"
DECISION_PATH = OUTPUT_ROOT / "macro_expectation_direction_decisions.csv"
VALIDATION_PATH = OUTPUT_ROOT / "macro_expectation_direction_validation.csv"
NULL_PATH = DETAIL_ROOT / "macro_expectation_direction_null_statistics.parquet"
COVERAGE_PATH = OUTPUT_ROOT / "macro_expectation_direction_market_coverage.csv"
REPORT_PATH = OUTPUT_ROOT / "macro_expectation_direction_plain_review.md"
RESULT_PATH = OUTPUT_ROOT / "macro_expectation_direction_result.json"


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def load_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], DataFrame, DataFrame]:
    result = json.loads(frozen.RESULT_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(frozen.FREEZE_PATH.read_text(encoding="utf-8"))
    if result.get("status") != "completed_macro_expectation_direction_freeze":
        raise ValueError("Macro-expectation direction freeze is not terminal")
    if freeze.get("status") != (
        "frozen_macro_expectation_direction_before_crypto_outcomes"
    ):
        raise ValueError("Macro-expectation direction definitions were not frozen")
    if result.get("crypto_outcomes_read") or freeze.get("crypto_outcomes_read"):
        raise ValueError("Direction freeze unexpectedly contains crypto outcomes")
    for name, path in (
        ("catalog", frozen.CATALOG_PATH),
        ("counts", frozen.COUNTS_PATH),
        ("freeze", frozen.FREEZE_PATH),
    ):
        if result["artifacts"][name]["sha256"] != g0.sha256_file(path):
            raise ValueError(f"Frozen direction artifact changed: {path}")
    catalog = pd.read_csv(frozen.CATALOG_PATH)
    catalog["anchor_utc"] = pd.to_datetime(catalog["anchor_utc"], utc=True)
    catalog["historical_direction_test_eligible"] = catalog[
        "historical_direction_test_eligible"
    ].map(frozen.actual_frozen.strict_bool)
    counts = pd.read_csv(frozen.COUNTS_PATH)
    return result, freeze, catalog, counts


def direction(value: Any) -> int:
    return prior_direct.direction(value)


def blocked_macro_anchors(catalog: DataFrame) -> list[pd.Timestamp]:
    anchors = prior_direct.blocked_scheduled_anchors()
    anchors.extend(catalog["anchor_utc"].tolist())
    return sorted(set(anchors))


def matched_ordinary_returns(
    frame: DataFrame,
    positions: Mapping[pd.Timestamp, int],
    *,
    anchor: pd.Timestamp,
    horizon: int,
    blocked_anchors: Sequence[pd.Timestamp],
) -> list[float]:
    returns: list[float] = []
    for weeks in range(1, frozen.CONTROL_WEEKS + 1):
        control_anchor = anchor - pd.Timedelta(weeks=weeks)
        if (
            prior_direct.minimum_event_distance_hours(blocked_anchors, control_anchor)
            <= frozen.CONTROL_EVENT_EXCLUSION_HOURS
        ):
            continue
        position = positions.get(control_anchor)
        if position is None:
            continue
        value, reason = prior_direct.future_return(frame, position, horizon)
        if reason is None and direction(value):
            returns.append(value)
        if len(returns) >= frozen.CONTROL_COUNT:
            break
    return returns


def extract_raw_rows(
    catalog: DataFrame,
    *,
    blocked_anchors: Sequence[pd.Timestamp] | None = None,
) -> tuple[DataFrame, DataFrame]:
    selected = catalog[catalog["historical_direction_test_eligible"]].copy()
    blocked = (
        list(blocked_anchors)
        if blocked_anchors is not None
        else blocked_macro_anchors(catalog)
    )
    records: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for pair in frozen.ASSETS:
        path = g0.ohlcv_path(pair, "1m")
        frame = g0.load_ohlcv(path).sort_values("date", kind="stable")
        frame = frame.drop_duplicates("date").reset_index(drop=True)
        positions = prior_direct.position_by_date(frame)
        coverage.append(
            {
                "pair": pair,
                "timeframe": "1m",
                "path": str(path.resolve()),
                "sha256": g0.sha256_file(path),
                "first_candle_utc": frame["date"].min(),
                "last_candle_utc": frame["date"].max(),
                "rows": len(frame),
            }
        )
        for event in selected.itertuples(index=False):
            anchor = pd.Timestamp(event.anchor_utc)
            position = positions.get(anchor)
            for horizon in frozen.HORIZONS_MINUTES:
                if position is None:
                    response_return = np.nan
                    abstention_reason = "missing_anchor_candle"
                    pre_return = np.nan
                    controls: list[float] = []
                else:
                    response_return, abstention_reason = prior_direct.future_return(
                        frame, position, horizon
                    )
                    pre_return = prior_direct.pre_event_return(frame, position, horizon)
                    controls = matched_ordinary_returns(
                        frame,
                        positions,
                        anchor=anchor,
                        horizon=horizon,
                        blocked_anchors=blocked,
                    )
                actual_direction = direction(response_return)
                if abstention_reason is None and actual_direction == 0:
                    abstention_reason = "zero_future_return"
                ordinary_up_rate = (
                    float(np.mean([direction(value) == 1 for value in controls]))
                    if len(controls) >= frozen.MINIMUM_CONTROLS
                    else np.nan
                )
                records.append(
                    {
                        "route_id": event.route_id,
                        "family": event.family,
                        "expectation_episode_id": event.expectation_episode_id,
                        "source_block": event.source_block,
                        "official_event_id": event.official_event_id,
                        "anchor_utc": anchor,
                        "pair": pair,
                        "horizon_minutes": int(horizon),
                        "predicted_base_direction": int(
                            event.predicted_base_direction
                        ),
                        "orientation_policy": event.orientation_policy,
                        "previous_release_proxy_direction": int(
                            event.previous_release_proxy_direction
                        ),
                        "response_return": response_return,
                        "actual_direction": actual_direction,
                        "pre_event_return": pre_return,
                        "pretrend_direction": direction(pre_return),
                        "ordinary_control_count": len(controls),
                        "ordinary_up_rate": ordinary_up_rate,
                        "abstention_reason": abstention_reason,
                    }
                )
    rows = DataFrame.from_records(records)
    return rows, DataFrame.from_records(coverage)


def _rate(values: Series) -> float:
    valid = values.dropna()
    return float(valid.astype(bool).mean()) if len(valid) else np.nan


def orientation_choices(rows: DataFrame) -> dict[str, dict[str, Any]]:
    choices: dict[str, dict[str, Any]] = {}
    for route_id, group in rows.groupby("route_id", sort=True):
        policy = str(group["orientation_policy"].iloc[0])
        if policy != "learn_rate_or_growth_on_development_only":
            choices[str(route_id)] = {
                "factor": 1,
                "label": "fixed_inverse_rate_pressure",
                "development_whole_events": 0,
                "development_base_accuracy": None,
            }
            continue
        development = group[
            group["source_block"].eq(frozen.PARTITIONS[0])
            & group["actual_direction"].isin([-1, 1])
        ]
        event_votes: list[bool] = []
        for _, event in development.groupby("expectation_episode_id", sort=False):
            actual_consensus = direction(event["actual_direction"].sum())
            if actual_consensus == 0:
                continue
            predicted = int(event["predicted_base_direction"].iloc[0])
            event_votes.append(predicted == actual_consensus)
        if not event_votes:
            raise ValueError(f"No development events can orient route {route_id}")
        accuracy = float(np.mean(event_votes))
        factor = 1 if accuracy >= 0.5 else -1
        choices[str(route_id)] = {
            "factor": factor,
            "label": "rate_pressure" if factor == 1 else "growth_risk_appetite",
            "development_whole_events": len(event_votes),
            "development_base_accuracy": accuracy,
        }
    return choices


def _circular_previous(values: Series) -> Series:
    if values.empty:
        return values
    return Series(np.roll(values.to_numpy(copy=True), 1), index=values.index)


def add_orientation_and_controls(
    rows: DataFrame,
) -> tuple[DataFrame, dict[str, dict[str, Any]]]:
    result = rows.copy()
    choices = orientation_choices(result)
    factors = {route: int(value["factor"]) for route, value in choices.items()}
    labels = {route: str(value["label"]) for route, value in choices.items()}
    result["orientation_factor"] = result["route_id"].map(factors).astype(int)
    result["chosen_orientation"] = result["route_id"].map(labels)
    result["predicted_direction"] = (
        result["predicted_base_direction"] * result["orientation_factor"]
    ).astype(int)
    result["oriented_previous_proxy_direction"] = (
        result["previous_release_proxy_direction"] * result["orientation_factor"]
    ).astype(int)
    result["direction_hit"] = [
        prior_direct._hit(predicted, actual)
        for predicted, actual in zip(
            result["predicted_direction"], result["actual_direction"], strict=True
        )
    ]
    result["pretrend_hit"] = [
        prior_direct._hit(predicted, actual)
        for predicted, actual in zip(
            result["pretrend_direction"], result["actual_direction"], strict=True
        )
    ]
    result["previous_proxy_hit"] = [
        prior_direct._hit(predicted, actual)
        for predicted, actual in zip(
            result["oriented_previous_proxy_direction"],
            result["actual_direction"],
            strict=True,
        )
    ]
    result["ordinary_control_accuracy"] = np.where(
        result["predicted_direction"].eq(1),
        result["ordinary_up_rate"],
        1.0 - result["ordinary_up_rate"],
    )

    result = result.sort_values(
        [
            "route_id",
            "pair",
            "horizon_minutes",
            "source_block",
            "anchor_utc",
        ],
        kind="stable",
    )
    rotate_keys = ["route_id", "pair", "horizon_minutes", "source_block"]
    result["rotated_predicted_direction"] = result.groupby(
        rotate_keys, sort=False
    )["predicted_direction"].transform(_circular_previous)
    result["rotated_hit"] = [
        prior_direct._hit(predicted, actual)
        for predicted, actual in zip(
            result["rotated_predicted_direction"],
            result["actual_direction"],
            strict=True,
        )
    ]

    development = result[
        result["source_block"].eq(frozen.PARTITIONS[0])
        & result["actual_direction"].isin([-1, 1])
    ]
    majority = (
        development.groupby(
            ["route_id", "pair", "horizon_minutes"], sort=False
        )["actual_direction"]
        .sum()
        .map(direction)
        .rename("development_majority_direction")
        .reset_index()
    )
    result = result.merge(
        majority,
        on=["route_id", "pair", "horizon_minutes"],
        how="left",
        validate="many_to_one",
    )
    result["development_majority_direction"] = result[
        "development_majority_direction"
    ].fillna(0).astype(int)
    result["development_majority_hit"] = [
        prior_direct._hit(predicted, actual)
        for predicted, actual in zip(
            result["development_majority_direction"],
            result["actual_direction"],
            strict=True,
        )
    ]
    return (
        result.sort_values(
            ["anchor_utc", "route_id", "pair", "horizon_minutes"], kind="stable"
        ).reset_index(drop=True),
        choices,
    )


def summarize_group(group: DataFrame) -> dict[str, Any]:
    usable = group[group["direction_hit"].notna()]
    ordinary = usable[usable["ordinary_control_accuracy"].notna()]
    event_accuracy = _rate(ordinary["direction_hit"])
    ordinary_accuracy = (
        float(ordinary["ordinary_control_accuracy"].mean())
        if len(ordinary)
        else np.nan
    )
    record = {
        "events_total": int(group["expectation_episode_id"].nunique()),
        "events_usable": int(usable["expectation_episode_id"].nunique()),
        "events_with_ordinary_controls": int(
            ordinary["expectation_episode_id"].nunique()
        ),
        "accuracy": _rate(usable["direction_hit"]),
        "ordinary_paired_event_accuracy": event_accuracy,
        "ordinary_period_accuracy": ordinary_accuracy,
        "ordinary_paired_lift": (
            event_accuracy - ordinary_accuracy
            if np.isfinite(event_accuracy) and np.isfinite(ordinary_accuracy)
            else np.nan
        ),
        "median_response_return": (
            float(usable["response_return"].median()) if len(usable) else np.nan
        ),
    }
    controls = {
        "development_majority": "development_majority_hit",
        "pretrend": "pretrend_hit",
        "previous_proxy": "previous_proxy_hit",
        "rotated_sign": "rotated_hit",
    }
    for label, column in controls.items():
        paired = usable[usable[column].notna()]
        main_accuracy = _rate(paired["direction_hit"])
        control_accuracy = _rate(paired[column])
        record[f"{label}_paired_events"] = int(
            paired["expectation_episode_id"].nunique()
        )
        record[f"main_on_{label}_events_accuracy"] = main_accuracy
        record[f"{label}_accuracy"] = control_accuracy
        record[f"{label}_paired_lift"] = (
            main_accuracy - control_accuracy
            if np.isfinite(main_accuracy) and np.isfinite(control_accuracy)
            else np.nan
        )
    additional = usable[usable["previous_proxy_hit"].isna()]
    record["events_additional_to_previous_proxy"] = int(
        additional["expectation_episode_id"].nunique()
    )
    record["accuracy_additional_to_previous_proxy"] = _rate(
        additional["direction_hit"]
    )
    return record


def direction_summaries(rows: DataFrame) -> DataFrame:
    keys = ["route_id", "family", "pair", "horizon_minutes", "chosen_orientation"]
    records: list[dict[str, Any]] = []
    for key, group in rows.groupby(keys, sort=True, dropna=False):
        base = dict(zip(keys, key, strict=True))
        for partition, partition_rows in group.groupby("source_block", sort=False):
            records.append(
                {
                    **base,
                    "partition": partition,
                    **summarize_group(partition_rows),
                }
            )
        records.append(
            {
                **base,
                "partition": "development_plus_later",
                **summarize_group(group),
            }
        )
    return DataFrame.from_records(records)


def _finite_max(values: Iterable[Any]) -> float:
    finite = [float(value) for value in values if pd.notna(value) and np.isfinite(value)]
    return max(finite) if finite else np.nan


def route_decisions(summary: DataFrame) -> DataFrame:
    keys = ["route_id", "family", "pair", "horizon_minutes", "chosen_orientation"]
    records: list[dict[str, Any]] = []
    for key, group in summary.groupby(keys, sort=True, dropna=False):
        base = dict(zip(keys, key, strict=True))
        by_partition = group.set_index("partition")
        development = by_partition.loc[frozen.PARTITIONS[0]]
        later = by_partition.loc[frozen.PARTITIONS[1]]
        combined = by_partition.loc["development_plus_later"]
        null_control_lifts = [
            float(combined[column])
            for column in (
                "development_majority_paired_lift",
                "pretrend_paired_lift",
                "rotated_sign_paired_lift",
            )
            if pd.notna(combined[column]) and np.isfinite(combined[column])
        ]
        minimum_null_control_lift = (
            min(null_control_lifts) if null_control_lifts else np.nan
        )
        coverage_ok = bool(
            int(development["events_usable"])
            >= frozen.MINIMUM_EVENTS_PER_PARTITION
            and int(later["events_usable"]) >= frozen.MINIMUM_EVENTS_PER_PARTITION
            and int(development["events_with_ordinary_controls"])
            >= frozen.MINIMUM_EVENTS_PER_PARTITION
            and int(later["events_with_ordinary_controls"])
            >= frozen.MINIMUM_EVENTS_PER_PARTITION
        )
        accuracy_ok = bool(
            float(combined["accuracy"]) >= frozen.MINIMUM_OVERALL_ACCURACY
            and float(later["accuracy"]) >= frozen.MINIMUM_VALIDATION_ACCURACY
        )
        null_control_ok = bool(
            np.isfinite(minimum_null_control_lift)
            and minimum_null_control_lift >= frozen.MINIMUM_CONTROL_LIFT
        )
        ordinary_control_ok = bool(
            pd.notna(combined["ordinary_paired_lift"])
            and float(combined["ordinary_paired_lift"])
            >= frozen.MINIMUM_CONTROL_LIFT
        )
        candidate = bool(
            coverage_ok and accuracy_ok and null_control_ok and ordinary_control_ok
        )
        records.append(
            {
                **base,
                "development_events": int(development["events_usable"]),
                "later_events": int(later["events_usable"]),
                "development_accuracy": float(development["accuracy"]),
                "later_accuracy": float(later["accuracy"]),
                "overall_accuracy": float(combined["accuracy"]),
                "strong_target_reached": bool(
                    float(combined["accuracy"]) >= frozen.STRONG_ACCURACY_TARGET
                ),
                "minimum_paired_lift_over_null_controls": minimum_null_control_lift,
                "previous_proxy_paired_events": int(
                    combined["previous_proxy_paired_events"]
                ),
                "main_on_previous_proxy_events_accuracy": float(
                    combined["main_on_previous_proxy_events_accuracy"]
                ),
                "previous_proxy_accuracy": float(
                    combined["previous_proxy_accuracy"]
                ),
                "previous_proxy_paired_lift": float(
                    combined["previous_proxy_paired_lift"]
                ),
                "events_additional_to_previous_proxy": int(
                    combined["events_additional_to_previous_proxy"]
                ),
                "accuracy_additional_to_previous_proxy": float(
                    combined["accuracy_additional_to_previous_proxy"]
                ),
                "ordinary_paired_event_accuracy": float(
                    combined["ordinary_paired_event_accuracy"]
                ),
                "ordinary_period_accuracy": float(
                    combined["ordinary_period_accuracy"]
                ),
                "ordinary_paired_lift": float(combined["ordinary_paired_lift"]),
                "raw_candidate": candidate,
                "raw_decision": (
                    "provisional_requires_whole_batch_search_adjustment"
                    if candidate
                    else "coverage_limited"
                    if not coverage_ok
                    else "not_retained_by_frozen_controls"
                ),
            }
        )
    return DataFrame.from_records(records)


def observed_statistics(decisions: DataFrame) -> dict[str, int]:
    candidates = decisions[decisions["raw_candidate"]]
    per_route = candidates.groupby("route_id").size()
    shared: list[int] = []
    for _, group in candidates.groupby("route_id", sort=False):
        btc = set(
            group.loc[group["pair"].eq(frozen.ASSETS[0]), "horizon_minutes"]
        )
        eth = set(
            group.loc[group["pair"].eq(frozen.ASSETS[1]), "horizon_minutes"]
        )
        shared.append(len(btc & eth))
    return {
        "candidate_cells": len(candidates),
        "max_candidate_cells_one_route": int(per_route.max()) if len(per_route) else 0,
        "max_shared_horizons_one_route": max(shared, default=0),
    }


OUTCOME_COLUMNS = (
    "response_return",
    "actual_direction",
    "pre_event_return",
    "pretrend_direction",
    "ordinary_control_count",
    "ordinary_up_rate",
    "abstention_reason",
)


def permute_whole_event_outcomes(
    raw_rows: DataFrame, rng: np.random.Generator
) -> DataFrame:
    outcome_keys = [
        "family",
        "source_block",
        "expectation_episode_id",
        "pair",
        "horizon_minutes",
    ]
    outcomes = raw_rows[[*outcome_keys, *OUTCOME_COLUMNS]].drop_duplicates()
    if outcomes.duplicated(outcome_keys).any():
        raise ValueError("A whole event has conflicting duplicated market outcomes")

    mapping: dict[tuple[str, str, str], str] = {}
    episodes = raw_rows[
        ["family", "source_block", "expectation_episode_id"]
    ].drop_duplicates()
    for (family, partition), group in episodes.groupby(
        ["family", "source_block"], sort=False
    ):
        source = group["expectation_episode_id"].astype(str).to_numpy()
        destination = rng.permutation(source)
        mapping.update(
            {
                (str(family), str(partition), str(src)): str(dst)
                for src, dst in zip(source, destination, strict=True)
            }
        )

    shuffled = raw_rows.drop(columns=list(OUTCOME_COLUMNS)).copy()
    shuffled["destination_episode_id"] = [
        mapping[(str(family), str(partition), str(episode))]
        for family, partition, episode in zip(
            shuffled["family"],
            shuffled["source_block"],
            shuffled["expectation_episode_id"],
            strict=True,
        )
    ]
    lookup = outcomes.rename(
        columns={"expectation_episode_id": "destination_episode_id"}
    )
    shuffled = shuffled.merge(
        lookup,
        on=[
            "family",
            "source_block",
            "destination_episode_id",
            "pair",
            "horizon_minutes",
        ],
        how="left",
        validate="many_to_one",
    )
    return shuffled.drop(columns=["destination_episode_id"])


def null_distribution(
    raw_rows: DataFrame,
    *,
    iterations: int,
    seed: int,
) -> DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, int]] = []
    for iteration in range(iterations):
        permuted = permute_whole_event_outcomes(raw_rows, rng)
        scored, _ = add_orientation_and_controls(permuted)
        decisions = route_decisions(direction_summaries(scored))
        records.append({"iteration": iteration, **observed_statistics(decisions)})
    return DataFrame.from_records(records)


def route_validation(decisions: DataFrame, null: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    for route_id, group in decisions.groupby("route_id", sort=True):
        candidates = group[group["raw_candidate"]]
        btc = set(
            candidates.loc[
                candidates["pair"].eq(frozen.ASSETS[0]), "horizon_minutes"
            ]
        )
        eth = set(
            candidates.loc[
                candidates["pair"].eq(frozen.ASSETS[1]), "horizon_minutes"
            ]
        )
        candidate_cells = len(candidates)
        shared_horizons = len(btc & eth)
        cell_probability = (
            int(null["max_candidate_cells_one_route"].ge(candidate_cells).sum()) + 1
        ) / (len(null) + 1)
        shared_probability = (
            int(null["max_shared_horizons_one_route"].ge(shared_horizons).sum()) + 1
        ) / (len(null) + 1)
        retained = bool(
            candidate_cells > 0
            and shared_horizons > 0
            and cell_probability <= 0.05
            and shared_probability <= 0.05
        )
        records.append(
            {
                "route_id": route_id,
                "family": str(group["family"].iloc[0]),
                "chosen_orientation": str(group["chosen_orientation"].iloc[0]),
                "raw_candidate_cells": candidate_cells,
                "shared_btc_eth_horizons": shared_horizons,
                "cell_count_familywise_probability": cell_probability,
                "shared_horizon_familywise_probability": shared_probability,
                "decision": (
                    "retained_historical_lead_requires_future_confirmation"
                    if retained
                    else "not_retained_after_whole_batch_controls"
                ),
                "independent_confirmation": False,
                "profit_used": False,
            }
        )
    return DataFrame.from_records(records)


def render_report(
    counts: DataFrame,
    decisions: DataFrame,
    validation: DataFrame,
    choices: Mapping[str, Mapping[str, Any]],
) -> str:
    lines = [
        "# Macro Expectation-Surprise Direction Review",
        "",
        "This test asks whether the released figure being above or below the market's "
        "pre-release expectation helped with BTC/ETH direction over 5-60 minutes. All "
        "pass/fail controls use the exact same announcements as the signal.",
        "",
        "## Source boundary",
        "",
    ]
    for row in counts.itertuples(index=False):
        lines.append(
            f"- `{row.route_id}`: `{row.source_decision}` "
            f"(older `{row.development_events}`, later `{row.later_events}`)"
        )
    lines.extend(["", "## Jobs channel choice", ""])
    for route, choice in choices.items():
        if route.startswith("jobs_"):
            lines.append(
                f"- `{route}`: `{choice['label']}` selected only from the older block "
                f"using `{choice['development_whole_events']}` whole announcements."
            )
    lines.extend(["", "## Results", ""])
    for row in validation.itertuples(index=False):
        lines.append(
            f"- `{row.route_id}`: `{row.decision}`; raw cells "
            f"`{row.raw_candidate_cells}`, shared BTC/ETH windows "
            f"`{row.shared_btc_eth_horizons}`."
        )
    best = decisions.sort_values("overall_accuracy", ascending=False).head(5)
    lines.extend(["", "## Strongest descriptive cells", ""])
    for row in best.itertuples(index=False):
        lines.append(
            f"- `{row.route_id}` / `{row.pair}` / `{row.horizon_minutes}m`: "
            f"older `{row.development_accuracy:.1%}`, later `{row.later_accuracy:.1%}`, "
            f"overall `{row.overall_accuracy:.1%}`, raw decision `{row.raw_decision}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A failure here does not mean the announcement did nothing; event-driven "
            "activity was tested separately. A retained result is one historical lead, "
            "not a universal rule, profit result, or independent future confirmation.",
            "",
        ]
    )
    return "\n".join(lines)


def execute(
    *,
    overwrite: bool = False,
    iterations: int = frozen.PERMUTATION_ITERATIONS,
    seed: int = frozen.PERMUTATION_SEED,
) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if result.get("status") != "completed_macro_expectation_direction_review":
            raise ValueError("Existing direction review is invalid")
        return result

    freeze_result, freeze, catalog, counts = load_frozen_inputs()
    if iterations != int(freeze["search_adjustment"]["iterations"]):
        raise ValueError("Iteration count must match the frozen batch")
    if seed != int(freeze["search_adjustment"]["seed"]):
        raise ValueError("Seed must match the frozen batch")
    raw_rows, coverage = extract_raw_rows(catalog)
    rows, choices = add_orientation_and_controls(raw_rows)
    summary = direction_summaries(rows)
    decisions = route_decisions(summary)
    null = null_distribution(raw_rows, iterations=iterations, seed=seed)
    validation = route_validation(decisions, null)
    report = render_report(counts, decisions, validation, choices)

    DETAIL_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(rows, OUTCOME_PATH)
    g0.atomic_write_csv(summary, SUMMARY_PATH)
    g0.atomic_write_csv(decisions, DECISION_PATH)
    g0.atomic_write_csv(validation, VALIDATION_PATH)
    g0.atomic_write_parquet(null, NULL_PATH)
    g0.atomic_write_csv(coverage, COVERAGE_PATH)
    REPORT_PATH.write_text(report, encoding="utf-8", newline="\n")
    retained = validation[
        validation["decision"].eq(
            "retained_historical_lead_requires_future_confirmation"
        )
    ]["route_id"].tolist()
    result = {
        "schema_version": 1,
        "status": "completed_macro_expectation_direction_review",
        "created_at_utc": g0.utc_now(),
        "crypto_outcomes_read": True,
        "profit_used": False,
        "independent_confirmation": False,
        "paired_control_method_repair": True,
        "raw_candidate_cells": int(decisions["raw_candidate"].sum()),
        "retained_routes": retained,
        "orientation_choices": choices,
        "frozen_input": freeze_result,
        "artifacts": {
            "direction_rows": artifact(OUTCOME_PATH),
            "summary": artifact(SUMMARY_PATH),
            "decisions": artifact(DECISION_PATH),
            "validation": artifact(VALIDATION_PATH),
            "null_statistics": artifact(NULL_PATH),
            "coverage": artifact(COVERAGE_PATH),
            "plain_review": artifact(REPORT_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }
    g0.atomic_write_json(result, RESULT_PATH)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--iterations", type=int, default=frozen.PERMUTATION_ITERATIONS)
    parser.add_argument("--seed", type=int, default=frozen.PERMUTATION_SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "ready_not_executed",
                    "crypto_outcomes_will_be_read": True,
                    "assets": list(frozen.ASSETS),
                    "horizons_minutes": list(frozen.HORIZONS_MINUTES),
                },
                indent=2,
            )
        )
        return os.EX_OK
    print(
        json.dumps(
            execute(
                overwrite=args.overwrite,
                iterations=args.iterations,
                seed=args.seed,
            ),
            indent=2,
        )
    )
    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())

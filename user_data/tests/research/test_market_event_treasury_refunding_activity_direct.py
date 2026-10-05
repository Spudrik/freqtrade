# ruff: noqa: S101

"""Focused tests for the Treasury refunding market-activity evaluator."""

from __future__ import annotations

import json

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_event_treasury_refunding_activity_direct as direct,
)


def frozen_rules() -> dict[str, object]:
    return {
        "market_test": {
            "minimum_events_by_partition": {
                "development_2021_2023": 2,
                "internal_validation_2024_2025": 1,
            },
            "minimum_median_activity_score": 1.2,
            "minimum_above_control_rate": 0.55,
        }
    }


def outcome_rows(*, score: float = 1.3, clean: bool = True) -> pd.DataFrame:
    records = []
    counts = {"development_2021_2023": 2, "internal_validation_2024_2025": 1}
    for partition in direct.PARTITIONS:
        for number in range(counts[partition]):
            records.append(
                {
                    "event_id": f"{partition}-{number}",
                    "pair": "BTC/USDT:USDT",
                    "horizon_minutes": 15,
                    "whole_event_partition": partition,
                    "activity_score": score,
                    "above_control_median": score > 1.0,
                    "abs_return_ratio": score,
                    "range_ratio": score,
                    "volume_ratio": score,
                    "other_major_scheduled_event_within_4h": not clean,
                }
            )
    return pd.DataFrame.from_records(records)


def test_collision_flags_use_external_events() -> None:
    catalogue = pd.DataFrame(
        {
            "anchor_utc": pd.to_datetime(
                ["2024-01-01T12:00Z", "2024-01-08T12:00Z"], utc=True
            )
        }
    )
    clean = direct.add_collision_flags(catalogue, [])
    assert not clean["other_major_scheduled_event_within_4h"].any()
    collision = direct.add_collision_flags(
        catalogue, [pd.Timestamp("2024-01-01T15:00Z")]
    )
    assert collision["other_major_scheduled_event_within_4h"].tolist() == [True, False]


def test_route_requires_both_partitions_and_both_variants() -> None:
    summary = direct.summarize_outcomes(outcome_rows(), frozen_rules())
    decisions = direct.classify_routes(summary, frozen_rules())
    selected = decisions.loc[
        decisions["pair"].eq("BTC/USDT:USDT")
        & decisions["horizon_minutes"].eq(15)
    ].iloc[0]
    assert selected["verdict"] == "retained_exploratory_activity_lead"


def test_all_only_pass_is_not_retained() -> None:
    clean = outcome_rows(score=0.8, clean=True)
    collision_one = outcome_rows(score=2.0, clean=False).assign(
        event_id=lambda frame: "collision-one-" + frame["event_id"]
    )
    collision_two = outcome_rows(score=2.0, clean=False).assign(
        event_id=lambda frame: "collision-two-" + frame["event_id"]
    )
    summary = direct.summarize_outcomes(
        pd.concat([clean, collision_one, collision_two], ignore_index=True),
        frozen_rules(),
    )
    decisions = direct.classify_routes(summary, frozen_rules())
    selected = decisions.loc[
        decisions["pair"].eq("BTC/USDT:USDT")
        & decisions["horizon_minutes"].eq(15)
    ].iloc[0]
    assert selected["verdict"] == "overlap_dependent_not_retained"


def test_partition_specific_minimums_are_enforced() -> None:
    rules = frozen_rules()
    rules["market_test"]["minimum_events_by_partition"][  # type: ignore[index]
        "internal_validation_2024_2025"
    ] = 2
    summary = direct.summarize_outcomes(outcome_rows(), rules)
    decisions = direct.classify_routes(summary, rules)
    selected = decisions.loc[
        decisions["pair"].eq("BTC/USDT:USDT")
        & decisions["horizon_minutes"].eq(15)
    ].iloc[0]
    assert selected["verdict"] == "coverage_limited"


def test_no_execute_reports_activity_only(capsys) -> None:  # type: ignore[no-untyped-def]
    assert direct.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ready_not_executed"
    assert result["profit_will_be_used"] is False
    assert result["direction_will_be_tested"] is False

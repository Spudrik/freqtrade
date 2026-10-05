# ruff: noqa: S101

"""Focused tests for the central-bank market-activity evaluator."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_event_central_bank_activity_direct as direct,
)


def frozen_rules() -> dict[str, object]:
    return {
        "market_test": {
            "minimum_events_per_partition": 2,
            "minimum_median_activity_score": 1.2,
            "minimum_above_control_rate": 0.55,
        }
    }


def outcome_rows(*, score: float = 1.3, clean: bool = True) -> pd.DataFrame:
    records = []
    for partition in direct.PARTITIONS:
        for number in range(2):
            records.append(
                {
                    "event_id": f"{partition}-{number}",
                    "bank": "ECB",
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


def test_collision_flags_ignore_the_event_itself() -> None:
    catalogue = pd.DataFrame(
        {
            "available_at_utc": pd.to_datetime(
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


def test_minute_window_rejects_timestamp_gaps() -> None:
    continuous = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC"),
            "open": np.ones(5),
            "high": np.ones(5),
            "low": np.ones(5),
            "close": np.ones(5),
            "volume": np.ones(5),
        }
    )
    assert direct.minute_window_status(continuous, 0, 5) == "usable"
    assert direct.contiguous_window_metrics(continuous, 0, 5) is not None
    gapped = continuous.copy()
    gapped.loc[3, "date"] = pd.Timestamp("2024-01-01T00:10:00Z")
    assert direct.minute_window_status(gapped, 0, 5) == "timestamp_gap"
    assert direct.contiguous_window_metrics(gapped, 0, 5) is None


def test_route_requires_both_partitions_and_both_variants() -> None:
    rows = outcome_rows()
    summary = direct.summarize_outcomes(rows, frozen_rules())
    decisions = direct.classify_routes(summary, frozen_rules())
    selected = decisions.loc[
        decisions["bank"].eq("ECB")
        & decisions["pair"].eq("BTC/USDT:USDT")
        & decisions["horizon_minutes"].eq(15)
    ].iloc[0]
    assert selected["verdict"] == "retained_repeatable_activity_lead"


def test_all_only_pass_is_not_retained() -> None:
    collision_one = outcome_rows(score=2.0, clean=False).assign(
        event_id=lambda frame: "collision-one-" + frame["event_id"]
    )
    collision_two = outcome_rows(score=2.0, clean=False).assign(
        event_id=lambda frame: "collision-two-" + frame["event_id"]
    )
    rows = pd.concat(
        [
            outcome_rows(clean=True),
            collision_one,
            collision_two,
        ],
        ignore_index=True,
    )
    rows.loc[
        ~rows["other_major_scheduled_event_within_4h"],
        [
            "activity_score",
            "above_control_median",
            "abs_return_ratio",
            "range_ratio",
            "volume_ratio",
        ],
    ] = [0.8, False, 0.8, 0.8, 0.8]
    summary = direct.summarize_outcomes(rows, frozen_rules())
    decisions = direct.classify_routes(summary, frozen_rules())
    selected = decisions.loc[
        decisions["bank"].eq("ECB")
        & decisions["pair"].eq("BTC/USDT:USDT")
        & decisions["horizon_minutes"].eq(15)
    ].iloc[0]
    assert selected["verdict"] == "overlap_dependent_not_retained"


def test_no_execute_reports_activity_only(capsys) -> None:  # type: ignore[no-untyped-def]
    assert direct.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ready_not_executed"
    assert result["profit_will_be_used"] is False
    assert result["direction_will_be_tested"] is False

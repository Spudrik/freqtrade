# ruff: noqa: S101

"""Tests for the frozen multi-source event activity runner."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_event_layer2_source_activity_direct as direct,
)


def synthetic_freeze() -> dict[str, Any]:
    return {
        "activity_test": {
            "assets": ["BTC/USDT:USDT"],
            "minimum_median_activity_score": 1.2,
            "minimum_above_control_rate": 0.55,
        },
        "activity_routes": {
            "test_source": {
                "plain_name": "Test source",
                "horizons_minutes": [15],
                "minimum_events_by_partition": {
                    "development_2021_2023": 2,
                    "internal_validation_2024_2025": 2,
                },
                "timestamp_limit": "test",
            }
        },
    }


def synthetic_outcomes(*, collision_fail: bool = False) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for partition in direct.PARTITIONS:
        for index in range(2):
            rows.append(
                {
                    "event_id": f"{partition}_{index}",
                    "base_event_id": f"{partition}_{index}",
                    "event_source": "test_source",
                    "pair": "BTC/USDT:USDT",
                    "horizon_minutes": 15,
                    "whole_event_partition": partition,
                    "activity_score": 1.4,
                    "above_control_median": True,
                    "abs_return_ratio": 1.3,
                    "range_ratio": 1.4,
                    "volume_ratio": 1.5,
                    "other_frozen_event_within_4h": collision_fail and index == 1,
                }
            )
    return pd.DataFrame(rows)


def test_complete_cross_period_route_is_retained_pending_shuffle() -> None:
    freeze = synthetic_freeze()
    summary = direct.summarize_outcomes(synthetic_outcomes(), freeze)
    decisions = direct.classify_routes(summary, freeze)

    assert len(summary) == 4
    assert decisions.iloc[0]["verdict"] == (
        "retained_exploratory_activity_lead_pending_family_shuffle"
    )


def test_collision_dependent_route_is_not_retained() -> None:
    freeze = synthetic_freeze()
    summary = direct.summarize_outcomes(
        synthetic_outcomes(collision_fail=True), freeze
    )
    decisions = direct.classify_routes(summary, freeze)

    assert decisions.iloc[0]["verdict"] == "coverage_limited"


def test_response_rejects_timestamp_gap() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-01-01T00:00:00Z", "2025-01-01T00:02:00Z"], utc=True
            ),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [10.0, 11.0],
        }
    )
    status, response = direct.response_for_horizon(frame, 0, 2)

    assert status == "timestamp_gap"
    assert response is None


def test_no_execute_declares_no_profit_or_direction(capsys: Any) -> None:
    assert direct.main([]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["status"] == "ready_not_executed"
    assert result["outcomes_will_be_read"] is True
    assert result["profit_will_be_used"] is False
    assert result["direction_will_be_tested"] is False

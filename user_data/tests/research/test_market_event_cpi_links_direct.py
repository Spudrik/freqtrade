# ruff: noqa: S101

"""Tests for frozen CPI leader, follower, and meme link analysis."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_event_cpi_links_direct as direct,
)


def test_subwindow_metrics_uses_only_requested_future_minutes() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=10, freq="1min", tz="UTC"),
            "open": np.arange(100.0, 110.0),
            "high": np.arange(101.0, 111.0),
            "low": np.arange(99.0, 109.0),
            "close": np.arange(100.5, 110.5),
            "volume": np.ones(10),
        }
    )
    positions = direct.shared.position_by_date(frame)
    status, metrics = direct.subwindow_metrics(
        frame,
        positions,
        anchor=pd.Timestamp("2025-01-01T00:00:00Z"),
        start_minute=5,
        end_minute=10,
    )

    assert status == "usable"
    assert metrics is not None
    assert np.isclose(metrics["return"], 109.5 / 105.0 - 1)
    assert metrics["volume"] == 5.0


def test_subwindow_continuity_accepts_millisecond_datetime_storage() -> None:
    dates = pd.Series(
        pd.date_range("2025-01-01", periods=5, freq="1min", tz="UTC")
    ).astype("datetime64[ms, UTC]")
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": np.full(5, 100.0),
            "high": np.full(5, 101.0),
            "low": np.full(5, 99.0),
            "close": np.full(5, 100.5),
            "volume": np.ones(5),
        }
    )
    positions = direct.shared.position_by_date(frame)
    status, metrics = direct.subwindow_metrics(
        frame,
        positions,
        anchor=pd.Timestamp("2025-01-01T00:00:00Z"),
        start_minute=0,
        end_minute=5,
    )

    assert status == "usable"
    assert metrics is not None


def test_leader_call_requires_above_median_first_move() -> None:
    samples = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "sample_id": "event",
                "sample_type": "event",
            },
            *(
                {
                    "event_id": "e1",
                    "sample_id": f"control_{index}",
                    "sample_type": "control",
                }
                for index in range(6)
            ),
        ]
    )
    lookup: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    lookup[("event", "BTC/USDT:USDT", 0, 5)] = {
        "return": 0.02,
        "abs_return": 0.02,
        "range": 0.02,
        "volume": 1.0,
        "eligibility_status": "usable",
    }
    for index in range(6):
        lookup[(f"control_{index}", "BTC/USDT:USDT", 0, 5)] = {
            "return": 0.01,
            "abs_return": 0.01,
            "range": 0.01,
            "volume": 1.0,
            "eligibility_status": "usable",
        }
    issued, sign, ratio = direct.leader_call(
        samples,
        lookup,
        event_id="e1",
        sample_id="event",
        leader="btc_first_move",
        window=5,
    )

    assert issued
    assert sign == 1.0
    assert ratio == 2.0


def test_transmission_summary_requires_control_and_self_baseline() -> None:
    rows: list[dict[str, Any]] = []
    for partition in direct.frozen.PARTITIONS:
        for index in range(5):
            rows.append(
                {
                    "event_id": f"{partition}_{index}",
                    "sample_type": "event",
                    "whole_event_partition": partition,
                    "leader": "btc_first_move",
                    "leader_window_minutes": 5,
                    "follower_end_minutes": 30,
                    "follower": "established_group_median",
                    "call_issued": True,
                    "direction_success": index < 4,
                    "own_initial_direction_success": index < 3,
                }
            )
    for index in range(20):
        rows.append(
            {
                "event_id": f"control_{index}",
                "sample_type": "control",
                "whole_event_partition": "development_2021_2023",
                "leader": "btc_first_move",
                "leader_window_minutes": 5,
                "follower_end_minutes": 30,
                "follower": "established_group_median",
                "call_issued": True,
                "direction_success": index < 10,
                "own_initial_direction_success": index < 10,
            }
        )
    freeze = {
        "cpi_immediate_leader_transmission": {
            "minimum_issued_whole_events": 10,
            "minimum_overall_direction_rate": 0.55,
            "minimum_each_partition_rate": 0.50,
            "minimum_control_uplift": 0.05,
        }
    }
    summary = direct.transmission_summary(pd.DataFrame(rows), freeze)

    assert summary.iloc[0]["event_direction_rate"] == 0.8
    assert summary.iloc[0]["control_direction_rate"] == 0.5
    assert bool(summary.iloc[0]["meets_frozen_rule"])


def test_no_execute_keeps_direction_inside_post_event_scope(capsys: Any) -> None:
    assert direct.main([]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["status"] == "ready_not_executed"
    assert result["profit_will_be_used"] is False
    assert result["direction_scope"] == "post_cpi_first_move_only"


def test_meme_windows_include_full_reaction_windows() -> None:
    assert {(0, 15), (0, 30), (0, 60)}.issubset(set(direct.MEME_WINDOWS))

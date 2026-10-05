# ruff: noqa: S101

"""Outcome-blind tests for the next event Layer 2 breadth freeze."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_event_layer2_breadth_freeze as frozen,
)


def test_fixed_window_clusters_do_not_chain_past_four_hours() -> None:
    frame = pd.DataFrame(
        {
            "stamp": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T03:00:00Z",
                    "2025-01-01T05:00:00Z",
                ],
                utc=True,
            )
        }
    )
    clustered = frozen.fixed_window_clusters(
        frame, anchor_column="stamp", prefix="test", window_minutes=240
    )

    assert clustered["market_window_group_id"].tolist() == [
        "test_0001",
        "test_0001",
        "test_0002",
    ]
    assert clustered["cluster_anchor_utc"].tolist() == [
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-01T05:00:00Z"),
    ]


def test_decision_minute_never_uses_prepublication_seconds() -> None:
    row = frozen._base_row(
        event_id="test",
        source_family="sec_hyperscaler_all",
        anchor=pd.Timestamp("2025-01-01T20:01:23Z"),
        partition="internal_validation_2024_2025",
        timestamp_quality="test",
    )

    assert row["decision_utc"] == pd.Timestamp("2025-01-01T20:02:00Z")
    assert row["decision_delay_seconds"] == 37


def test_controls_exclude_nearby_frozen_events() -> None:
    events = pd.DataFrame(
        [
            frozen._base_row(
                event_id="event_a",
                source_family="uk_ons_cpi",
                anchor=pd.Timestamp("2025-06-18T06:00:00Z"),
                partition="internal_validation_2024_2025",
                timestamp_quality="test",
            )
        ]
    )
    first_control = pd.Timestamp("2025-06-11T06:00:00Z")
    controls = frozen.build_control_map(events, [first_control])

    assert first_control not in set(controls["control_anchor_utc"])
    assert controls.iloc[0]["control_rank"] == 1
    assert controls.iloc[0]["control_anchor_utc"] == pd.Timestamp(
        "2025-06-04T06:00:00Z"
    )
    assert len(controls) == frozen.CONTROL_COUNT


def test_no_execute_declares_all_frozen_siblings(capsys: Any) -> None:
    assert frozen.main([]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["status"] == "ready_not_executed"
    assert result["outcomes_will_be_read"] is False
    assert set(result["activity_routes"]) == set(frozen.ACTIVITY_ROUTES)
    assert result["separate_links"] == [
        "cpi_immediate_leader_transmission",
        "cpi_meme_response",
    ]

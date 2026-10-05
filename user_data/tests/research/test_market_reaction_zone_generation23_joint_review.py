# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_joint_review as g23j,
)


def test_generation24_queue_preserves_five_route_breadth() -> None:
    queue = g23j.generation24_queue()
    active = [item for item in queue if item["status"].startswith("active_")]
    parked = [item for item in queue if item["status"].startswith("parked_")]

    assert len(active) == 5
    assert len(parked) == 5
    assert len({item["route_family"] for item in active}) == 5
    assert any("FreqAI" in item["plain_question"] for item in active)
    assert any("VWAP" in item["plain_question"] for item in active)


def test_direct_summary_does_not_promote_incomplete_near_miss() -> None:
    result = {
        "artifacts": {
            "decisions": {"path": "decisions.csv"},
            "scores": {"path": "scores.csv"},
        }
    }
    decisions = pd.DataFrame(
        {
            "complete_control_period_ladder": [False],
            "status": ["not_retained"],
        }
    )
    scores = pd.DataFrame(
        {
            "scope_kind": ["level"],
            "scope_value": ["candidate"],
            "metric": ["crossing_count"],
            "horizon_hours": [4],
            "market_scope": ["btc_separate"],
            "point_period_pass": [True],
            "strict_period_pass": [True],
            "equal_coin_difference": [0.2],
            "positive_coin_fraction": [1.0],
            "bootstrap_lower_95": [0.1],
        }
    )

    original = g23j.pd.read_csv
    g23j.pd.read_csv = lambda path: decisions if path == "decisions.csv" else scores
    try:
        summary = g23j.direct_summary(result, expected_controls=5)
    finally:
        g23j.pd.read_csv = original

    assert summary["strict_rows"] == 0
    assert summary["point_rows"] == 0
    assert summary["complete_ladder_rows"] == 0

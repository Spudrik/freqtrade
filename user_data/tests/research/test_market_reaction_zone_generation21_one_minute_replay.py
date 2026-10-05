# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation21_one_minute_replay as g21e,
)


def test_registry_reports_every_fixed_method_distance_and_horizon() -> None:
    assert len(g21e.FIXED_CALLS) == 6
    assert g21e.DISTANCE_THRESHOLDS_ATR == (0.5, 1.0)
    assert g21e.HORIZON_MINUTES == (15, 60, 240)
    assert g21e.MINIMUM_EPISODES == 12
    assert g21e.MAXIMUM_EPISODES == 18
    assert g21e.CONTROLS == (
        "naive_rejection",
        "majority_path",
        "matched_no_level_episode",
    )


def test_prior_episode_neighbourhood_is_excluded_by_pair() -> None:
    candidates = pd.DataFrame(
        {
            "pair": ["BTC", "BTC", "ETH"],
            "event_time": pd.to_datetime(
                ["2026-01-02T00:00Z", "2026-01-04T00:00Z", "2026-01-02T00:00Z"]
            ),
        }
    )
    prior = pd.DataFrame(
        {
            "pair": ["BTC"],
            "event_time": pd.to_datetime(["2026-01-02T12:00Z"]),
        }
    )

    result = g21e.exclude_prior_episode_neighbourhoods(candidates, prior)

    assert list(result["pair"]) == ["BTC", "ETH"]
    assert result.iloc[0]["event_time"] == pd.Timestamp("2026-01-04T00:00Z")


def test_no_level_outcome_uses_event_open_as_neutral_anchor() -> None:
    minute = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01T00:00Z", periods=300, freq="min"),
            "open": np.full(300, 100.0),
            "high": np.full(300, 100.1),
            "low": np.full(300, 99.9),
            "close": np.full(300, 100.0),
            "volume": np.ones(300),
        }
    )
    minute.loc[121:130, "close"] = 101.1
    episode = pd.Series(
        {
            "episode_id": "control-g21e-001",
            "episode_kind": "matched_no_level",
            "matched_to_episode_id": "g21e-001",
            "pair": "BTC/USDT:USDT",
            "period": "p1",
            "event_time": pd.Timestamp("2026-01-01T02:00Z"),
            "level_price": 999.0,
            "base_atr": 1.0,
            "approach_state": "from_below",
        }
    )

    rows = g21e.episode_outcomes(episode, minute, pd.Timestamp(episode["event_time"]))
    selected = next(
        row
        for row in rows
        if row["distance_threshold_atr"] == 1.0 and row["horizon_minutes"] == 15
    )

    assert selected["reaction"]
    assert selected["first_direction_numeric"] == 1


def test_naive_rejection_is_opposite_the_approach_direction() -> None:
    outcomes = pd.DataFrame(
        {
            "episode_id": ["a", "b"],
            "episode_kind": ["actual_level", "actual_level"],
            "distance_threshold_atr": [0.5, 0.5],
            "horizon_minutes": [15, 15],
            "first_direction_numeric": [-1, 1],
        }
    )
    sample = pd.DataFrame(
        {
            "episode_id": ["a", "b"],
            "approach_state": ["from_below", "from_above"],
        }
    )

    calls = g21e.control_calls(outcomes, sample)
    rejection = calls.loc[calls["method"].eq("naive_rejection")]

    assert rejection["call_numeric"].tolist() == [-1, 1]


def test_two_control_methods_can_score_against_one_episode_outcome() -> None:
    outcomes = pd.DataFrame(
        {
            "episode_id": ["a", "control-a"],
            "episode_kind": ["actual_level", "matched_no_level"],
            "distance_threshold_atr": [0.5, 0.5],
            "horizon_minutes": [15, 15],
            "reaction": [True, True],
            "first_direction_numeric": [-1, 1],
        }
    )
    candidates = pd.DataFrame(
        {
            "episode_id": ["a", "control-a"],
            "episode_kind": ["actual_level", "matched_no_level"],
            "method": ["approach_trend_15m", "approach_trend_15m"],
            "method_kind": ["causal_candidate", "causal_candidate"],
            "call_numeric": [-1, 1],
            "issued": [True, True],
        }
    )
    controls = pd.DataFrame(
        {
            "episode_id": ["a", "a"],
            "episode_kind": ["actual_level", "actual_level"],
            "distance_threshold_atr": [0.5, 0.5],
            "horizon_minutes": [15, 15],
            "method": ["naive_rejection", "majority_path"],
            "method_kind": ["control", "control"],
            "call_numeric": [-1, 1],
            "issued": [True, True],
        }
    )

    scored = g21e.score_methods(candidates, controls, outcomes)

    assert set(scored.loc[scored["method_kind"].eq("control"), "method"]) == {
        "naive_rejection",
        "majority_path",
    }

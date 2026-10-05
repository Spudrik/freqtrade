# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation20_one_minute_replay as g20m,
)


def test_fixed_call_set_and_controls_remain_bounded() -> None:
    assert len(g20m.FIXED_CALLS) == 6
    assert g20m.CONTROLS == (
        "majority_path",
        "simple_approach_trend",
        "matched_no_level_episode",
    )
    assert g20m.MINIMUM_EPISODES == 6
    assert g20m.MAXIMUM_EPISODES == 12


def test_pressure_call_uses_signed_volume_weighted_location() -> None:
    frame = pd.DataFrame(
        {
            "high": [11.0, 11.0],
            "low": [9.0, 9.0],
            "close": [10.8, 10.6],
            "volume": [2.0, 1.0],
        }
    )

    assert g20m.pressure_call(frame) == 1


def test_momentum_vote_abstains_on_balanced_votes() -> None:
    frame = pd.DataFrame(
        {
            "close": np.tile([1.0, 2.0], 30),
            "volume": np.ones(60),
        }
    )

    assert g20m.momentum_vote(frame) in {-1, 0, 1}


def test_finite_sign_has_explicit_abstention() -> None:
    assert g20m.finite_sign(np.nan) == 0
    assert g20m.finite_sign(0.0) == 0
    assert g20m.finite_sign(0.1) == 1
    assert g20m.finite_sign(-0.1) == -1


def test_no_level_control_does_not_require_a_contact_minute(monkeypatch) -> None:
    minute = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01T00:00Z", periods=121, freq="min"),
            "open": np.ones(121),
            "high": np.full(121, 1.1),
            "low": np.full(121, 0.9),
            "close": np.ones(121),
            "volume": np.ones(121),
        }
    )
    episode = pd.Series(
        {
            "episode_id": "control-test",
            "episode_kind": "matched_no_level",
            "matched_to_episode_id": "test",
            "pair": "BTC/USDT:USDT",
            "period": "p1",
            "event_time": pd.Timestamp("2026-01-01T01:00Z"),
            "level_price": 2.0,
            "zone_half_width": 0.1,
            "approach_state": "from_below",
        }
    )

    monkeypatch.setattr(
        g20m,
        "first_contact_time",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()),
    )
    outcome, _ = g20m.episode_calls_and_outcome(episode, minute)

    assert outcome["contact_time"] == pd.Timestamp("2026-01-01T01:00Z")

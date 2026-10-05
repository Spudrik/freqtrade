# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation21_structural_levels as g21d,
)


def synthetic_base() -> pd.DataFrame:
    rows = 190
    close = np.linspace(100.0, 101.0, rows)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC"),
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.tile(np.array([9.0, 10.0, 11.0]), rows // 3 + 1)[:rows],
            "base_atr": 1.0,
        }
    )
    frame["pre_close"] = frame["close"].shift(1)
    frame.loc[175, ["open", "high", "low", "close", "volume"]] = [
        100.0,
        103.0,
        99.0,
        102.0,
        50.0,
    ]
    return frame


def test_abnormal_candle_only_becomes_available_on_following_row() -> None:
    base = synthetic_base()
    surfaces = g21d.coordinate_surfaces(base, "BTC/USDT:USDT")

    assert surfaces["actual_anchor"][175] != 175
    assert surfaces["actual_anchor"][176] == 175
    assert surfaces["actual"]["event_open"][176] == 100.0
    assert surfaces["actual"]["event_close"][176] == 102.0
    assert surfaces["actual"]["event_body_midpoint"][176] == 101.0
    assert surfaces["actual"]["event_high"][176] == 103.0
    assert surfaces["actual"]["event_low"][176] == 99.0
    assert surfaces["actual"]["event_typical_price"][176] == 101.33333333333333


def test_random_active_anchors_are_strictly_prior_and_bounded() -> None:
    base = synthetic_base()
    surfaces = g21d.coordinate_surfaces(base, "BTC/USDT:USDT")
    anchors = surfaces["random_anchor"]
    rows = np.arange(len(base))
    valid = anchors >= 0

    assert np.all(anchors[valid] < rows[valid])
    assert np.all(rows[valid] - anchors[valid] <= 168)
    assert not np.any(surfaces["abnormal"][anchors[valid]])


def test_structural_registry_matches_frozen_question() -> None:
    assert len(g21d.LEVELS) == 6
    assert set(g21d.CONTROLS) == {
        "matched_random_time",
        "random_recent_high_activity_candle",
        "near_miss",
        "stale_72h",
        "price_shift",
    }
    assert g21d.ZONE_HALF_WIDTH_ATR == 0.25

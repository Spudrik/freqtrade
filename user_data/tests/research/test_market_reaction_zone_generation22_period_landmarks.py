# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_period_landmarks as g22a,
)


def hourly_base(days: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=24 * days, freq="1h", tz="UTC")
    day = np.repeat(np.arange(days, dtype=float), 24)
    within = np.tile(np.arange(24, dtype=float), days)
    close = 100.0 + day * 10.0 + within / 10.0
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.ones(len(dates)),
        }
    )


def test_previous_day_uses_only_the_fully_completed_prior_day() -> None:
    base = hourly_base()
    surfaces = g22a.completed_period_surfaces(
        base, "BTC/USDT:USDT", "previous_utc_day"
    )
    row = 2 * 24

    assert surfaces["actual"]["period_open"][row] == base["open"].iloc[24]
    assert surfaces["actual"]["period_close"][row] == base["close"].iloc[47]
    assert surfaces["actual"]["period_high"][row] == base["high"].iloc[24:48].max()
    assert surfaces["actual_source_open"].iloc[row] == pd.Timestamp("2026-01-02T00:00Z")


def test_current_period_future_values_cannot_change_previous_period_surface() -> None:
    base = hourly_base()
    row = 2 * 24
    before = g22a.completed_period_surfaces(
        base, "BTC/USDT:USDT", "previous_utc_day"
    )["actual"]["period_high"][row]
    changed = base.copy()
    changed.loc[row + 1 :, "high"] = 1_000_000.0
    after = g22a.completed_period_surfaces(
        changed, "BTC/USDT:USDT", "previous_utc_day"
    )["actual"]["period_high"][row]

    assert before == after


def test_landmark_registry_has_three_periods_six_coordinates_and_five_controls() -> None:
    assert len(g22a.PERIODS) == 3
    assert len(g22a.COORDINATES) == 6
    assert len(g22a.CONTROLS) == 5

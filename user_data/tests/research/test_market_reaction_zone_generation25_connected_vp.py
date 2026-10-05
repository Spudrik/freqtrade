# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation25_connected_volume_profile as g25v,
)


def _base(rows: int = 130) -> pd.DataFrame:
    price = 100.0 + np.sin(np.arange(rows) / 8.0) * 4.0
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-07-15", periods=rows, freq="h", tz="UTC"),
            "open": price - 0.2,
            "high": price + 1.0,
            "low": price - 1.0,
            "close": price + 0.2,
            "volume": 100.0 + (np.arange(rows) % 9) * 10.0,
            "pre_close": pd.Series(price + 0.2).shift(1),
            "base_atr": 2.0,
            "period": "g18_normal_early",
        }
    )


def test_component_bounds_returns_nearest_connected_run() -> None:
    mask = np.array([False, True, True, False, True, False])
    low = np.arange(6, dtype=float)
    high = low + 1.0
    assert g25v._component_bounds(mask, low, high, 4.2) == (4.0, 5.0, 1)
    assert g25v._component_bounds(mask, low, high, 2.5) == (1.0, 3.0, 2)


def test_profile_geometry_excludes_current_candle() -> None:
    base = _base()
    index = np.array([110], dtype=np.int64)
    original = g25v._profile_block(base, index)
    changed = base.copy()
    changed.loc[110, ["high", "low", "volume"]] = [1000.0, 1.0, 1e12]
    perturbed = g25v._profile_block(changed, index)
    for name in original:
        assert np.allclose(original[name][index], perturbed[name][index], equal_nan=True)


def test_near_miss_does_not_overlap_actual_contact() -> None:
    base = _base()
    lower = np.full(len(base), 100.0)
    upper = np.full(len(base), 101.0)
    bins = np.full(len(base), 2.0)
    actual = g25v.zone_events(
        base,
        pair="BTC/USDT:USDT",
        cohort="normal",
        zone_kind="connected_hvn_area",
        lower=lower,
        upper=upper,
        bin_count=bins,
        control="actual",
    )
    near = g25v.zone_events(
        base,
        pair="BTC/USDT:USDT",
        cohort="normal",
        zone_kind="connected_hvn_area",
        lower=lower,
        upper=upper,
        bin_count=bins,
        control="near_miss",
        event_kind="near_miss",
    )
    assert set(actual["event_time"]).isdisjoint(set(near["event_time"]))

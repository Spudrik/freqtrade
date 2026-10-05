# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_trend_channels as g22c,
)


def test_causal_channel_extrapolates_linear_log_price_one_step() -> None:
    indexes = np.arange(200, dtype=float)
    close = pd.Series(np.exp(2.0 + 0.001 * indexes))

    channel = g22c.causal_channel(close, 72)

    assert np.isclose(channel["trend_centre"][100], close.iloc[100], rtol=1e-10)
    assert np.isclose(
        channel["upper_1p5_residual_sd"][100], close.iloc[100], rtol=1e-8
    )
    assert np.isclose(channel["slope"][100], 0.001, rtol=1e-10)


def test_current_and_future_close_cannot_change_current_channel() -> None:
    close = pd.Series(np.exp(2.0 + 0.001 * np.arange(200, dtype=float)))
    before = g22c.causal_channel(close, 72)["trend_centre"][100]
    changed = close.copy()
    changed.iloc[100:] *= 10.0
    after = g22c.causal_channel(changed, 72)["trend_centre"][100]

    assert before == after


def test_trend_registry_is_fixed_and_bounded() -> None:
    assert g22c.LOOKBACKS == (72, 168, 720)
    assert len(g22c.COORDINATES) == 3
    assert len(g22c.CONTROLS) == 5

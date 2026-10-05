# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_cross_asset_context as g22d,
)


def pair_frame(multiplier: float) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=240, freq="1h", tz="UTC")
    close = 100.0 + multiplier * np.arange(240, dtype=float) * 0.1
    return pd.DataFrame(
        {
            "close": close,
            "volume": 10.0 + np.sin(np.arange(240, dtype=float)),
            "base_atr": np.ones(240),
        },
        index=dates,
    )


def test_cross_asset_features_use_only_completed_rows() -> None:
    frames = {
        "BTC/USDT:USDT": pair_frame(1.0),
        "ETH/USDT:USDT": pair_frame(0.8),
        "SOL/USDT:USDT": pair_frame(1.2),
    }
    before = g22d.causal_cross_asset_context(frames)
    changed = {name: frame.copy() for name, frame in frames.items()}
    for frame in changed.values():
        frame.iloc[200:, frame.columns.get_loc("close")] *= 10.0
    after = g22d.causal_cross_asset_context(changed)

    column = "equal_weight_median_absolute_return_over_atr_w4"
    assert before.loc[200, column] == after.loc[200, column]


def test_context_state_registry_is_mutually_exclusive() -> None:
    frames = {
        "BTC/USDT:USDT": pair_frame(1.0),
        "ETH/USDT:USDT": pair_frame(0.8),
        "SOL/USDT:USDT": pair_frame(1.2),
    }
    context = g22d.causal_cross_asset_context(frames)
    # Give the synthetic history a calibration/holdout boundary it can satisfy.
    original = g22d.CALIBRATION_END_EXCLUSIVE
    g22d.CALIBRATION_END_EXCLUSIVE = pd.Timestamp("2026-01-08T00:00Z")
    try:
        labelled, _ = g22d.add_context_states(context)
    finally:
        g22d.CALIBRATION_END_EXCLUSIVE = original

    for window in g22d.CONTEXT_WINDOWS:
        assert labelled[f"context_state_w{window}"].notna().all()


def test_context_requires_same_state_no_level_control() -> None:
    assert "same_state_no_level_time" in g22d.CONTROL_COMPARISONS
    assert len(g22d.CONTROL_COMPARISONS) == 6
    assert len(g22d.SELECTED_LEVELS) == 16

# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation21_market_state as g21s,
)


def test_causal_features_do_not_use_current_candle_volume() -> None:
    rows = 80
    base = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC"),
            "open": np.full(rows, 10.0),
            "high": np.full(rows, 11.0),
            "low": np.full(rows, 9.0),
            "close": np.linspace(10.0, 12.0, rows),
            "volume": np.arange(1.0, rows + 1.0),
        }
    )
    changed = base.copy()
    changed.loc[60, "volume"] = 1_000_000.0

    original = g21s.causal_features(base)
    perturbed = g21s.causal_features(changed)

    assert original.loc[60, "relative_volume"] == perturbed.loc[60, "relative_volume"]
    assert original.loc[61, "relative_volume"] != perturbed.loc[61, "relative_volume"]


def test_timeframe_hours_are_explicit() -> None:
    assert g21s.timeframe_hours("1h") == 1
    assert g21s.timeframe_hours("4h") == 4
    assert g21s.timeframe_hours("8h") == 8


def test_market_state_has_three_distinct_blocks_and_four_controls() -> None:
    assert len(g21s.BLOCKS) == 3
    assert len(g21s.CONTROLS) == 4

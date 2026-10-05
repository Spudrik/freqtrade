# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_generic_indicator_context as g24i,
)


def synthetic_base(rows: int = 900) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC")
    close = 100.0 + np.linspace(0.0, 15.0, rows) + 2.0 * np.sin(np.arange(rows) / 17.0)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.1,
            "high": close + 0.6,
            "low": close - 0.6,
            "close": close,
            "volume": 1000.0 + 50.0 * np.cos(np.arange(rows) / 9.0),
        }
    )


def test_generic_indicator_values_are_finite_after_warmup() -> None:
    values = g24i.indicator_values(synthetic_base())

    assert tuple(values.columns) == g24i.INDICATORS
    assert values.iloc[-100:].notna().all().all()


def test_informative_alignment_uses_only_previous_completed_bucket() -> None:
    base = synthetic_base()
    original = g24i.completed_timeframe_features(base, "4h")
    changed = base.copy()
    changed.loc[len(changed) - 1, ["open", "high", "low", "close", "volume"]] = 1e9
    revised = g24i.completed_timeframe_features(changed, "4h")

    pd.testing.assert_frame_equal(original.iloc[:-4], revised.iloc[:-4])


def test_indicator_route_allows_only_same_indicator_timeframe_agreement() -> None:
    frozen = g24i.load_freeze()

    assert frozen["status"] == "frozen_before_generation24_outcomes"
    assert g24i.AGREEMENTS == ("all_low", "all_high")
    assert g24i.CONTROLS[-1] == "same_state_no_level_time"

# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_price_distribution as g23p,
)


def synthetic_base() -> pd.DataFrame:
    values = np.arange(1.0, 31.0)
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=30, freq="1h", tz="UTC"),
            "high": values + 1.0,
            "low": values - 1.0,
            "close": values,
        }
    )


def test_distribution_surface_uses_completed_candles_only() -> None:
    surfaces = g23p.actual_surfaces(synthetic_base())
    median_24 = next(
        item
        for item in surfaces
        if item["lookback_hours"] == 24 and item["quantile"] == 0.5
    )

    assert np.isnan(median_24["level"][23])
    assert median_24["level"][24] == 12.5


def test_future_changes_do_not_rewrite_prior_distribution_levels() -> None:
    base = synthetic_base()
    original = g23p.actual_surfaces(base)
    changed = base.copy()
    changed.loc[29, ["high", "low", "close"]] = 1_000_000.0
    revised = g23p.actual_surfaces(changed)

    for before, after in zip(original, revised, strict=True):
        np.testing.assert_allclose(
            before["level"][:29], after["level"][:29], equal_nan=True
        )


def test_distribution_registry_matches_frozen_route() -> None:
    frozen = g23p.load_freeze()

    assert frozen["status"] == "frozen_before_generation23_outcomes"
    assert len(g23p.LOOKBACKS) == 4
    assert len(g23p.QUANTILES) == 5

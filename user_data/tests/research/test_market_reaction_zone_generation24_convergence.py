# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_round_distribution_convergence as g24x,
)


def synthetic_base(rows: int = 800) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC")
    close = 100.0 + np.linspace(0.0, 20.0, rows) + np.sin(np.arange(rows) / 12.0)
    return pd.DataFrame(
        {
            "date": dates,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "pre_close": pd.Series(close).shift(1),
            "base_atr": np.full(rows, 1.0),
        }
    )


def test_convergence_registry_covers_every_frozen_combination() -> None:
    surfaces = g24x.combination_surfaces(synthetic_base(), "TEST/USDT:USDT")

    assert len(surfaces) == 3 * 4 * 5
    assert len({g24x.identity(surface) for surface in surfaces}) == len(surfaces)
    assert set(g24x.COMPONENT_CONTROLS).issubset(g24x.CONTROLS)


def test_future_candle_change_does_not_rewrite_prior_convergence_levels() -> None:
    base = synthetic_base()
    original = g24x.combination_surfaces(base, "TEST/USDT:USDT")
    changed = base.copy()
    changed.loc[len(changed) - 1, ["high", "low", "close", "pre_close"]] = 10_000.0
    revised = g24x.combination_surfaces(changed, "TEST/USDT:USDT")

    for before, after in zip(original, revised, strict=True):
        np.testing.assert_allclose(before["level"][:-1], after["level"][:-1], equal_nan=True)


def test_combined_source_uses_later_component_timestamp() -> None:
    left = pd.Series(pd.to_datetime(["2026-01-01T00:00:00Z", None]))
    right = pd.Series(pd.to_datetime(["2026-01-01T01:00:00Z", "2026-01-01T02:00:00Z"]))

    combined = g24x.combined_source_open(left, right)

    assert combined.iloc[0] == pd.Timestamp("2026-01-01T01:00:00Z")
    assert combined.iloc[1] == pd.Timestamp("2026-01-01T02:00:00Z")


def test_convergence_route_matches_frozen_batch() -> None:
    frozen = g24x.load_freeze()

    assert frozen["status"] == "frozen_before_generation24_outcomes"
    assert len(g24x.CONTROLS) == 8

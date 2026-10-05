# ruff: noqa: S101

from __future__ import annotations

import numpy as np

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation20_change_point_avwap as g20v,
)


def test_anchored_statistics_respect_per_row_completed_intervals() -> None:
    price = np.array([10.0, 20.0, 30.0, 40.0])
    volume = np.ones(4)
    starts = np.array([0, 0, 1, 2])
    ends = np.array([0, 1, 2, 3])

    mean, sigma = g20v.anchored_statistics(price, volume, starts, ends)

    assert mean.tolist() == [10.0, 15.0, 25.0, 35.0]
    assert sigma[0] == 0.0
    assert np.allclose(sigma[1:], 5.0)


def test_invalid_anchor_interval_remains_missing() -> None:
    result = g20v.interval_sum(
        np.cumsum(np.array([1.0, 2.0, 3.0])),
        np.array([-1, 1]),
        np.array([0, 2]),
    )

    assert np.isnan(result[0])
    assert result[1] == 5.0


def test_change_point_registry_preserves_five_controls() -> None:
    assert set(g20v.LEVELS) == {
        "anchored_vwap",
        "anchored_vwap_plus_1p5_sigma",
        "anchored_vwap_minus_1p5_sigma",
    }
    assert len(g20v.CONTROLS) == 5
    assert g20v.ZONE_HALF_WIDTH_ATR == 0.25

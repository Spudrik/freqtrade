# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_multitimeframe_convergence as g22b,
)


def hourly_base(hours: int = 24 * 70) -> pd.DataFrame:
    dates = pd.date_range("2026-01-05", periods=hours, freq="1h", tz="UTC")
    close = 100.0 + np.arange(hours, dtype=float) * 0.01
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.ones(hours),
        }
    )


def test_four_hour_levels_use_the_previous_completed_four_hour_bar() -> None:
    base = hourly_base()
    surfaces = g22b.mapped_timeframe_levels(base, "4h", 4)
    high = next(item for item in surfaces if item["level_name"].endswith("__high"))
    row = 8

    assert high["level"][row] == base["high"].iloc[4:8].max()
    assert high["source_open"].iloc[row] == pd.Timestamp("2026-01-05T04:00Z")


def test_current_source_bar_cannot_change_current_mapped_level() -> None:
    base = hourly_base()
    before = next(
        item
        for item in g22b.mapped_timeframe_levels(base, "4h", 4)
        if item["level_name"].endswith("__high")
    )["level"][8]
    changed = base.copy()
    changed.loc[8:11, "high"] = 1_000_000.0
    after = next(
        item
        for item in g22b.mapped_timeframe_levels(changed, "4h", 4)
        if item["level_name"].endswith("__high")
    )["level"][8]

    assert before == after


def test_cluster_annotation_counts_independent_timeframes() -> None:
    events = pd.DataFrame({"base_index": [0], "level_price": [100.0]})
    surfaces = [
        {
            "level": np.array([100.0]),
            "timeframe_hours": 1,
            "family": "a",
        },
        {
            "level": np.array([100.2]),
            "timeframe_hours": 24,
            "family": "b",
        },
        {
            "level": np.array([102.0]),
            "timeframe_hours": 168,
            "family": "c",
        },
    ]

    result = g22b.annotate_clusters(events, surfaces, np.array([1.0]))

    assert result.loc[0, "geometry_state"] == "two_timeframe_cluster"
    assert result.loc[0, "highest_timeframe"] == "1d"
    assert result.loc[0, "source_timeframes_present"] == "1h|1d"


def test_multitimeframe_registry_keeps_singles_and_clusters() -> None:
    assert tuple(g22b.TIMEFRAMES) == ("1h", "4h", "1d", "1w")
    assert g22b.GEOMETRIES[0] == "isolated_single_timeframe"
    assert len(g22b.CONTROLS) == 5

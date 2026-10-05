# ruff: noqa: S101

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from pandas import DataFrame

from user_data.Custom_Launcher.research.context_features import (
    market_event_freqai_breadth_cache as cache,
)


def test_outcome_surface_uses_current_and_future_candles_only() -> None:
    dates = pd.date_range("2024-01-01", periods=220, freq="h", tz="UTC")
    frame = DataFrame(
        {
            "date": dates,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 10.0,
        }
    )
    frame.loc[180:183, "high"] = [102.0, 103.0, 104.0, 105.0]
    frame.loc[180:183, "low"] = [98.0, 97.0, 96.0, 95.0]
    frame.loc[180:183, "close"] = [101.0, 102.0, 103.0, 104.0]
    frame.loc[180:183, "volume"] = 20.0
    result = cache.outcome_surface(frame)
    row = result.iloc[180]
    assert np.isclose(row["raw_range_atr_h4"], 5.0)
    assert np.isclose(row["raw_volume_ratio_h4"], 2.0)
    assert np.isclose(row["raw_close_return_atr_h4"], 2.0)
    assert np.isclose(row["raw_excursion_balance_atr_h4"], 0.0)
    assert result.iloc[-1]["&-meb_log_range_atr_h8"] != result.iloc[-1][
        "&-meb_log_range_atr_h8"
    ]


def test_event_context_never_counts_a_future_event() -> None:
    samples = DataFrame(
        {
            "sample_kind": ["actual_event"],
            "model_anchor_utc": pd.to_datetime(["2024-01-02T12:00:00Z"], utc=True),
            "event_families_json": [json.dumps(["us_cpi"])],
            "event_kinds_json": [json.dumps(["scheduled_us_macro"])],
            "parent_exact_clock_fraction": [1.0],
            "parent_scheduled_fraction": [1.0],
            "source_sign_available_count": [1],
            "source_sign_primary_mean": [-1.0],
            "source_sign_secondary_mean": [-1.0],
            "crypto_relation_sign_available_count": [1],
            "crypto_relation_sign_mean": [1.0],
            "current_event_count": [1],
            "current_family_count": [1],
            "current_kind_count": [1],
        }
    )
    events = DataFrame(
        {
            "model_anchor_utc": pd.to_datetime(
                ["2024-01-02T11:00:00Z", "2024-01-02T14:00:00Z"], utc=True
            ),
            "event_family": ["fomc_policy_decision", "uk_cpi"],
            "source_sign_primary": [np.nan, 1.0],
        }
    )
    result = cache.event_sample_context(samples, events).iloc[0]
    assert result["event_confluence__events_known_within_24h"] == 1.0
    assert result["event_confluence__families_known_within_24h"] == 1.0


def test_level_context_keeps_single_and_cluster_surfaces_distinct() -> None:
    parent = DataFrame(
        {
            "independent_level_family_count_within_0p5atr": [0.0, 1.0, 3.0],
            "nearest_level_distance_atr": [np.nan, 0.2, 0.1],
            "source_role_fixed_encoding": [np.nan, 2.0, 3.0],
            "source_timeframe_fixed_encoding": [np.nan, 1.0, 2.0],
            "pre_crossing_count_4h": [np.nan, 2.0, 4.0],
        }
    )
    result = cache._level_context(parent)
    assert result["single_level__present"].tolist() == [0.0, 1.0, 0.0]
    assert result["cluster__present"].tolist() == [0.0, 0.0, 1.0]
    assert result.loc[1, "single_level__nearest_distance_atr"] == 0.2
    assert result.loc[2, "cluster__independent_family_count"] == 3.0

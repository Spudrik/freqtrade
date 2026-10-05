# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_freqai_cache as g22e,
)


def synthetic_base(rows: int = 400) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC")
    close = pd.Series(100.0 + np.linspace(0.0, 8.0, rows) + np.sin(np.arange(rows) / 8.0))
    high = close + 1.0
    low = close - 1.0
    volume = pd.Series(1000.0 + (np.arange(rows) % 24) * 5.0)
    candle_range = high - low
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.2,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "base_atr": 2.0,
            "pre_close": close.shift(1),
            "pre_range_median_24": candle_range.shift(1).rolling(24, min_periods=12).median(),
            "pre_volume_median_24": volume.shift(1).rolling(24, min_periods=12).median(),
            "pre_pressure_mean_24": 0.0,
            "candle_pressure": 0.0,
        }
    )


def synthetic_context(dates: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "btc_absolute_return_over_atr_w4": 0.4,
            "equal_weight_median_absolute_return_over_atr_w4": 0.3,
            "cross_coin_return_dispersion_w4": 0.2,
        }
    )


def test_registry_has_exact_low_dimensional_profiles_and_controls() -> None:
    registry = g22e.build_registry()
    assert len(g22e.FEATURE_COLUMNS) == 12
    assert len(registry["profiles"]) == 8
    assert len(registry["comparisons"]) == 8
    assert {
        item["control_type"]
        for item in registry["comparisons"]
        if item["cohort"] == "normal"
    } == {
        "constant_training_median",
        "level_geometry_only_model",
        "market_state_only_model",
        "within_pair_time_shuffled_training_labels",
    }
    assert not any("profit" in target or "direction" in target for target in g22e.TARGETS)


def test_feature_row_uses_only_information_known_before_that_hour() -> None:
    base = synthetic_base()
    context = synthetic_context(base["date"])
    original, _ = g22e.feature_surface(base, context, "BTC/USDT:USDT")
    changed = base.copy()
    changed.loc[300, ["open", "high", "low", "close", "volume"]] = [50, 300, 10, 250, 999999]
    revised, _ = g22e.feature_surface(changed, context, "BTC/USDT:USDT")

    pd.testing.assert_series_equal(
        original.loc[300, list(g22e.FEATURE_COLUMNS)],
        revised.loc[300, list(g22e.FEATURE_COLUMNS)],
    )


def test_targets_start_after_feature_timestamp_and_stay_unsigned() -> None:
    base = synthetic_base(80)
    support = pd.DataFrame(
        {
            "date": base["date"],
            "nearest_level_price": 100.0,
            "zone_half_width": 0.5,
            f"ready__{g22e.READY_BLOCK}": True,
            "shuffled_source_date": base["date"] - pd.Timedelta(hours=168),
        }
    )
    original = g22e.target_frame(base, support, "normal")
    changed = base.copy()
    changed.loc[11, ["high", "low", "close", "volume"]] = [150.0, 50.0, 125.0, 5000.0]
    revised = g22e.target_frame(changed, support, "normal")

    target = "&-g22_maximum_absolute_excursion_atr_h1"
    assert original.loc[10, target] != revised.loc[10, target]
    assert original.loc[11, target] == revised.loc[11, target]

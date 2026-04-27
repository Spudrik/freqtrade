from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy

from entry_sieve_tools import apply_explicit_hyperopt_surface, entry_sieve_minimal_roi, entry_sieve_stoploss


def tagged_parameter(param: Any, *tags: str) -> Any:
    family_map = {
        "entry": "entries",
        "entries": "entries",
        "structure": "entries",
        "volume": "entries",
        "entry_enables": "entries",
        "entry_confirmation": "entries",
        "entry_structure": "entries",
        "trend_filter": "entries",
        "daily_structure": "entries",
        "hourly_execution": "entries",
        "line_quality": "entries",
        "target_space": "entries",
        "trendline_projection": "entries",
        "breakout_long": "entries",
        "exit": "exits",
        "exits": "exits",
        "exit_profile": "exits",
        "exit_peel": "exits",
        "exit_target": "exits",
        "entry_family_exit": "exits",
        "stoploss": "exits",
        "adjust_position": "adjust_position",
        "add_enables": "adjust_position",
        "add_rules": "adjust_position",
        "capital": "stake",
        "stake": "stake",
        "risk": "risk",
    }
    family: str | None = None
    mode: str | None = None
    for raw_tag in tags:
        tag = str(raw_tag or "").strip()
        if not tag or ":" not in tag:
            continue
        key, value = tag.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key == "family":
            mapped = family_map.get(value.lower())
            if mapped and family is None:
                family = mapped
        elif key == "mode" and value and mode is None:
            mode = value

    if family is None:
        if mode and mode.startswith("exit_"):
            family = "exits"
        elif mode and mode.startswith("adjust_"):
            family = "adjust_position"
        elif mode and mode.startswith("stake_"):
            family = "stake"
        elif mode and mode.startswith("risk_"):
            family = "risk"
        else:
            family = "entries"

    if mode is None:
        mode_defaults = {
            "entries": "entry_core",
            "exits": "exit_core",
            "adjust_position": "adjust_core",
            "stake": "stake_core",
            "risk": "risk_core",
        }
        mode = mode_defaults.get(family, "entry_core")

    setattr(param, "batch_tags", (f"family:{family}", f"mode:{mode}"))
    return param


class CodexRegimePullback(IStrategy):
    """
    Trend-regime pullback strategy.

    Hypothesis:
    - Only trade in a directional EMA regime.
    - Enter when price pulls back toward the faster trend mean, momentum resets,
      then closes back in the trend direction on acceptable volume.
    """

    INTERFACE_VERSION = 3

    timeframe = "1h"
    startup_candle_count = 240
    process_only_new_candles = True
    can_short = True

    minimal_roi = entry_sieve_minimal_roi(0.02)
    stoploss = entry_sieve_stoploss(-0.02)
    use_exit_signal = False
    use_custom_stoploss = False
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    ema_fast_period = tagged_parameter(
        IntParameter(18, 36, default=24, space="buy", optimize=True, load=True),
        "family:entry",
        "mode:entry_regime_pullback",
    )
    ema_slow_period = tagged_parameter(
        IntParameter(72, 144, default=96, space="buy", optimize=True, load=True),
        "family:entry",
        "mode:entry_regime_pullback",
    )
    rsi_pullback_low = tagged_parameter(
        IntParameter(36, 48, default=42, space="buy", optimize=True, load=True),
        "family:entry",
        "mode:entry_regime_pullback",
    )
    rsi_pullback_high = tagged_parameter(
        IntParameter(52, 64, default=58, space="buy", optimize=True, load=True),
        "family:entry",
        "mode:entry_regime_pullback",
    )
    pullback_atr_max = tagged_parameter(
        DecimalParameter(0.10, 1.20, decimals=2, default=0.55, space="buy", optimize=True, load=True),
        "family:entry",
        "mode:entry_regime_pullback",
    )
    volume_ratio_min = tagged_parameter(
        DecimalParameter(0.60, 1.60, decimals=2, default=0.85, space="buy", optimize=True, load=True),
        "family:entry",
        "mode:entry_regime_pullback",
    )

    @staticmethod
    def _rsi(close: Series, period: int = 14) -> Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        rs = gain / loss.replace(0.0, np.nan)
        return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

    @staticmethod
    def _atr(dataframe: DataFrame, period: int = 14) -> Series:
        prev_close = dataframe["close"].shift(1)
        tr = pd.concat(
            [
                dataframe["high"] - dataframe["low"],
                (dataframe["high"] - prev_close).abs(),
                (dataframe["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        fast = int(self.ema_fast_period.value)
        slow = int(self.ema_slow_period.value)
        dataframe["codex_ema_fast"] = dataframe["close"].ewm(span=fast, adjust=False, min_periods=fast).mean()
        dataframe["codex_ema_slow"] = dataframe["close"].ewm(span=slow, adjust=False, min_periods=slow).mean()
        dataframe["codex_rsi"] = self._rsi(dataframe["close"], 14)
        dataframe["codex_atr"] = self._atr(dataframe, 14)
        dataframe["codex_volume_ratio"] = dataframe["volume"] / dataframe["volume"].rolling(48, min_periods=24).mean()
        dataframe["codex_pullback_dist_atr"] = (
            (dataframe["close"] - dataframe["codex_ema_fast"]).abs() / dataframe["codex_atr"].replace(0.0, np.nan)
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None

        long_regime = dataframe["codex_ema_fast"] > dataframe["codex_ema_slow"]
        short_regime = dataframe["codex_ema_fast"] < dataframe["codex_ema_slow"]
        volume_ok = dataframe["codex_volume_ratio"] >= float(self.volume_ratio_min.value)
        near_mean = dataframe["codex_pullback_dist_atr"] <= float(self.pullback_atr_max.value)

        long_trigger = (
            long_regime
            & near_mean
            & volume_ok
            & dataframe["codex_rsi"].between(float(self.rsi_pullback_low.value), 55.0)
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["close"] > dataframe["codex_ema_fast"])
        )
        short_trigger = (
            short_regime
            & near_mean
            & volume_ok
            & dataframe["codex_rsi"].between(45.0, float(self.rsi_pullback_high.value))
            & (dataframe["close"] < dataframe["open"])
            & (dataframe["close"] < dataframe["codex_ema_fast"])
        )

        dataframe.loc[long_trigger.fillna(False), "enter_long"] = 1
        dataframe.loc[long_trigger.fillna(False), "enter_tag"] = "codex_regime_pullback_long"
        dataframe.loc[short_trigger.fillna(False), "enter_short"] = 1
        dataframe.loc[short_trigger.fillna(False), "enter_tag"] = "codex_regime_pullback_short"
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe


apply_explicit_hyperopt_surface(CodexRegimePullback)

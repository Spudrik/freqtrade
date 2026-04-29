from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy


HYPEROPT_PARAM_ENV = "HYBRID_RECOVERY_HYPEROPT_PARAMS"
ENTRY_SIEVE_TAKE_PROFIT_ENV = "ENTRY_SIEVE_TAKE_PROFIT_PCT"
ENTRY_SIEVE_STOPLOSS_ENV = "ENTRY_SIEVE_STOPLOSS_PCT"


def split_hyperopt_tokens(value: str | None) -> set[str]:
    if not value:
        return set()
    normalized = value.replace(";", ",").replace("|", ",").replace(" ", ",")
    return {token.strip() for token in normalized.split(",") if token.strip()}


def is_parameter_object(value: Any) -> bool:
    return bool(value is not None and value.__class__.__name__.endswith("Parameter"))


def apply_explicit_hyperopt_surface(strategy_cls: type) -> None:
    selected = split_hyperopt_tokens(os.environ.get(HYPEROPT_PARAM_ENV))
    if not selected:
        return
    for name in dir(strategy_cls):
        value = getattr(strategy_cls, name, None)
        if is_parameter_object(value):
            value.optimize = str(name) in selected


def _pct_env(name: str, default_ratio: float) -> float:
    raw = str(os.environ.get(name) or "").strip()
    if not raw:
        return float(default_ratio)
    try:
        return max(0.0, float(raw)) / 100.0
    except ValueError:
        return float(default_ratio)


def entry_sieve_minimal_roi(default: float = 0.02) -> dict[str, float]:
    return {"0": _pct_env(ENTRY_SIEVE_TAKE_PROFIT_ENV, default)}


def entry_sieve_stoploss(default: float = -0.02) -> float:
    return -_pct_env(ENTRY_SIEVE_STOPLOSS_ENV, abs(default))



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


class Sieve1CapitulationReclaimLong(IStrategy):
    """
    Overshoot reclaim strategy.

    Hypothesis:
    - Extreme distance from a slow mean plus abnormal volume often marks forced
      selling/buying.
    - Entry waits for a reclaim candle instead of trying to catch the first spike.
    """

    INTERFACE_VERSION = 3

    timeframe = "1h"
    startup_candle_count = 260
    process_only_new_candles = True
    can_short = False

    minimal_roi = entry_sieve_minimal_roi(0.02)
    stoploss = entry_sieve_stoploss(-0.02)
    use_exit_signal = False
    use_custom_stoploss = False
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    mean_period = tagged_parameter(
        IntParameter(96, 240, default=144, space="buy", optimize=True, load=True),
        "family:entry",
        "mode:entry_capitulation_reclaim_long",
    )
    overshoot_atr_min = tagged_parameter(
        DecimalParameter(1.20, 4.00, decimals=2, default=2.10, space="buy", optimize=True, load=True),
        "family:entry",
        "mode:entry_capitulation_reclaim_long",
    )
    reclaim_lookback = tagged_parameter(
        IntParameter(3, 12, default=6, space="buy", optimize=True, load=True),
        "family:entry",
        "mode:entry_capitulation_reclaim_long",
    )
    volume_ratio_min = tagged_parameter(
        DecimalParameter(1.00, 4.00, decimals=2, default=1.60, space="buy", optimize=True, load=True),
        "family:entry",
        "mode:entry_capitulation_reclaim_long",
    )
    wick_fraction_min = tagged_parameter(
        DecimalParameter(0.20, 0.70, decimals=2, default=0.35, space="buy", optimize=True, load=True),
        "family:entry",
        "mode:entry_capitulation_reclaim_long",
    )

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
        mean_period = int(self.mean_period.value)
        dataframe["mean"] = dataframe["close"].ewm(span=mean_period, adjust=False, min_periods=mean_period).mean()
        dataframe["atr"] = self._atr(dataframe, 14)
        dataframe["dist_atr"] = (dataframe["close"] - dataframe["mean"]) / dataframe["atr"].replace(0.0, np.nan)
        dataframe["volume_ratio"] = dataframe["volume"] / dataframe["volume"].rolling(72, min_periods=24).mean()
        candle_range = (dataframe["high"] - dataframe["low"]).replace(0.0, np.nan)
        dataframe["lower_wick_fraction"] = (dataframe[["open", "close"]].min(axis=1) - dataframe["low"]) / candle_range
        dataframe["upper_wick_fraction"] = (dataframe["high"] - dataframe[["open", "close"]].max(axis=1)) / candle_range
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None

        lookback = int(self.reclaim_lookback.value)
        overshoot = float(self.overshoot_atr_min.value)
        volume_ok = dataframe["volume_ratio"] >= float(self.volume_ratio_min.value)

        long_capitulation = (
            (dataframe["dist_atr"] <= -overshoot)
            & volume_ok
            & (dataframe["lower_wick_fraction"] >= float(self.wick_fraction_min.value))
        )
        short_capitulation = (
            (dataframe["dist_atr"] >= overshoot)
            & volume_ok
            & (dataframe["upper_wick_fraction"] >= float(self.wick_fraction_min.value))
        )
        recent_long_capitulation = long_capitulation.rolling(lookback, min_periods=1).max().shift(1).fillna(False).astype(bool)
        recent_short_capitulation = short_capitulation.rolling(lookback, min_periods=1).max().shift(1).fillna(False).astype(bool)

        long_reclaim = (
            recent_long_capitulation
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["close"] > dataframe["high"].shift(1))
            & (dataframe["dist_atr"] > dataframe["dist_atr"].shift(1))
        )
        short_reclaim = (
            recent_short_capitulation
            & (dataframe["close"] < dataframe["open"])
            & (dataframe["close"] < dataframe["low"].shift(1))
            & (dataframe["dist_atr"] < dataframe["dist_atr"].shift(1))
        )

        dataframe.loc[long_reclaim.fillna(False), "enter_long"] = 1
        dataframe.loc[long_reclaim.fillna(False), "enter_tag"] = "capitulation_reclaim_long"
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe


apply_explicit_hyperopt_surface(Sieve1CapitulationReclaimLong)




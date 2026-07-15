from __future__ import annotations
# SIEVE2_ARCHIVE_NOTE: Parked after 20260612T211448 retry: zero trades at TP/SL 3/3; no usable signal in the validation window.

SIEVE_STAGE = "sieve2"
NOVEL_IDEA = True
SOURCE_STRATEGY = "novel"
SOURCE_RESULT_BATCH = "manual_20260612"
RESEARCH_PATH = "volatility_4h"
UPDATE_HYPOTHESIS = "Volatility contraction after a weak bounce may precede a short breakdown."

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, IStrategy
from user_data.strategies.entry_sieve_tools import entry_sieve_minimal_roi, entry_sieve_stoploss

SIDE = "short"
ENTRY_MODE = "volatility_contraction_breakdown"
ENTRY_TAG = "volatility_contraction_breakdown_short_4h"


def tagged_parameter(param: Any) -> Any:
    setattr(param, "batch_tags", ("family:entries", "mode:sieve2_pivot"))
    return param


def _safe_div(numer: Series, denom: Series) -> Series:
    denom = denom.replace(0.0, np.nan)
    return numer / denom


def _rsi(close: Series, length: int = 14) -> Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / float(length), adjust=False, min_periods=max(2, length // 2)).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / float(length), adjust=False, min_periods=max(2, length // 2)).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(frame: DataFrame, length: int = 14) -> Series:
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=max(2, length // 2)).mean()


class Sieve2VolatilityContractionBreakdownShort4h(IStrategy):
    timeframe = "4h"
    can_short = True
    startup_candle_count = 320
    minimal_roi = entry_sieve_minimal_roi(0.03)
    stoploss = entry_sieve_stoploss(-0.03)
    process_only_new_candles = True
    use_exit_signal = False
    INTERFACE_VERSION = 3

    sweep_lookback = tagged_parameter(IntParameter(8, 56, default=24, space="buy", optimize=True, load=True))
    compression_lookback = tagged_parameter(IntParameter(3, 18, default=8, space="buy", optimize=True, load=True))
    trend_window = tagged_parameter(IntParameter(12, 80, default=32, space="buy", optimize=True, load=True))
    sweep_buffer_pct = tagged_parameter(DecimalParameter(0.001, 0.03, default=0.006, decimals=3, space="buy", optimize=True, load=True))
    range_contraction_max = tagged_parameter(DecimalParameter(0.03, 0.50, default=0.18, decimals=2, space="buy", optimize=True, load=True))
    volume_ratio_min = tagged_parameter(DecimalParameter(0.70, 4.0, default=1.10, decimals=2, space="buy", optimize=True, load=True))
    wick_ratio_min = tagged_parameter(DecimalParameter(0.20, 0.90, default=0.45, decimals=2, space="buy", optimize=True, load=True))
    expansion_ratio_min = tagged_parameter(DecimalParameter(0.80, 3.50, default=1.20, decimals=2, space="buy", optimize=True, load=True))
    trend_bias_min = tagged_parameter(DecimalParameter(0.00, 0.20, default=0.02, decimals=3, space="buy", optimize=True, load=True))
    reclaim_buffer_pct = tagged_parameter(DecimalParameter(0.001, 0.03, default=0.006, decimals=3, space="buy", optimize=True, load=True))

    def populate_indicators(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:
        close = pd.to_numeric(dataframe["close"], errors="coerce")
        high = pd.to_numeric(dataframe["high"], errors="coerce")
        low = pd.to_numeric(dataframe["low"], errors="coerce")
        open_ = pd.to_numeric(dataframe["open"], errors="coerce")
        volume = pd.to_numeric(dataframe["volume"], errors="coerce").fillna(0.0)
        look = int(self.sweep_lookback.value)
        comp = int(self.compression_lookback.value)
        trend = int(self.trend_window.value)
        rolling_high = high.rolling(look, min_periods=max(2, look // 2)).max()
        rolling_low = low.rolling(look, min_periods=max(2, look // 2)).min()
        comp_high = high.rolling(comp, min_periods=max(2, comp // 2)).max()
        comp_low = low.rolling(comp, min_periods=max(2, comp // 2)).min()
        comp_mean = close.rolling(comp, min_periods=max(2, comp // 2)).mean()
        trend_mean = close.rolling(trend, min_periods=max(3, trend // 2)).mean()
        dataframe["rsi"] = _rsi(close, 14)
        dataframe["atr"] = _atr(dataframe, 14)
        dataframe["rolling_high"] = rolling_high
        dataframe["rolling_low"] = rolling_low
        dataframe["compression_high"] = comp_high
        dataframe["compression_low"] = comp_low
        dataframe["compression_mid"] = (comp_high + comp_low) / 2.0
        dataframe["range_ratio"] = _safe_div(comp_high - comp_low, comp_mean).fillna(0.0)
        dataframe["volume_ratio"] = _safe_div(volume, volume.rolling(comp, min_periods=max(2, comp // 2)).mean()).fillna(0.0)
        dataframe["body_ratio"] = _safe_div((close - open_).abs(), high - low).fillna(0.0)
        dataframe["lower_wick_ratio"] = _safe_div(np.minimum(open_, close) - low, high - low).fillna(0.0)
        dataframe["upper_wick_ratio"] = _safe_div(high - np.maximum(open_, close), high - low).fillna(0.0)
        dataframe["trend_mean"] = trend_mean
        dataframe["trend_bias"] = _safe_div(close, trend_mean) - 1.0
        dataframe["close_prev"] = close.shift(1)
        return dataframe

    def _entry_condition(self, dataframe: DataFrame) -> Series:
        close = pd.to_numeric(dataframe["close"], errors="coerce")
        open_ = pd.to_numeric(dataframe["open"], errors="coerce")
        high = pd.to_numeric(dataframe["high"], errors="coerce")
        low = pd.to_numeric(dataframe["low"], errors="coerce")
        rolling_high = pd.to_numeric(dataframe["rolling_high"], errors="coerce")
        rolling_low = pd.to_numeric(dataframe["rolling_low"], errors="coerce")
        compression_high = pd.to_numeric(dataframe["compression_high"], errors="coerce")
        compression_low = pd.to_numeric(dataframe["compression_low"], errors="coerce")
        compression_mid = pd.to_numeric(dataframe["compression_mid"], errors="coerce")
        volume_ratio = pd.to_numeric(dataframe["volume_ratio"], errors="coerce")
        range_ratio = pd.to_numeric(dataframe["range_ratio"], errors="coerce")
        lower_wick = pd.to_numeric(dataframe["lower_wick_ratio"], errors="coerce")
        upper_wick = pd.to_numeric(dataframe["upper_wick_ratio"], errors="coerce")
        body_ratio = pd.to_numeric(dataframe["body_ratio"], errors="coerce")
        trend_mean = pd.to_numeric(dataframe["trend_mean"], errors="coerce")
        trend_bias = pd.to_numeric(dataframe["trend_bias"], errors="coerce")
        sweep_ok = low.lt(rolling_low * (1.0 - float(self.sweep_buffer_pct.value)))
        sweep_high_ok = high.gt(rolling_high * (1.0 + float(self.sweep_buffer_pct.value)))
        reclaim_long = close.gt(rolling_low)
        reject_short = close.lt(rolling_high)
        volume_ok = volume_ratio.ge(float(self.volume_ratio_min.value))
        wick_ok_long = lower_wick.ge(float(self.wick_ratio_min.value))
        wick_ok_short = upper_wick.ge(float(self.wick_ratio_min.value))
        compress_ok = range_ratio.le(float(self.range_contraction_max.value))
        expansion_ok = range_ratio.ge(float(self.expansion_ratio_min.value))

        if ENTRY_MODE == "equal_lows_sweep":
            equal_level = rolling_low.sub(rolling_low.shift(1)).abs().div(close.abs()).le(float(self.reclaim_buffer_pct.value) * 2.0)
            return sweep_ok & equal_level & reclaim_long & wick_ok_long & volume_ok & close.gt(open_)
        if ENTRY_MODE == "equal_highs_sweep":
            equal_level = rolling_high.sub(rolling_high.shift(1)).abs().div(close.abs()).le(float(self.reclaim_buffer_pct.value) * 2.0)
            return sweep_high_ok & equal_level & reject_short & wick_ok_short & volume_ok & close.lt(open_)
        if ENTRY_MODE == "prior_low_reclaim":
            return sweep_ok & reclaim_long & wick_ok_long & volume_ok
        if ENTRY_MODE == "capitulation_absorption":
            return expansion_ok & wick_ok_long & volume_ok & close.gt(open_) & body_ratio.le(0.60)
        if ENTRY_MODE == "capitulation_distribution":
            return expansion_ok & wick_ok_short & volume_ok & close.lt(open_) & body_ratio.le(0.60)
        if ENTRY_MODE == "volatility_expansion_breakout":
            return compress_ok & close.gt(compression_high * (1.0 + float(self.reclaim_buffer_pct.value))) & volume_ok
        if ENTRY_MODE == "volatility_contraction_breakdown":
            return compress_ok & close.lt(compression_low * (1.0 - float(self.reclaim_buffer_pct.value))) & volume_ok
        if ENTRY_MODE == "regime_pullback_reclaim":
            return trend_bias.ge(float(self.trend_bias_min.value)) & close.lt(trend_mean) & close.gt(trend_mean * (1.0 - float(self.reclaim_buffer_pct.value))) & volume_ok
        return pd.Series(False, index=dataframe.index)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:
        condition = self._entry_condition(dataframe)
        if SIDE == "long":
            dataframe.loc[condition, ["enter_long", "enter_tag"]] = (1, ENTRY_TAG)
        else:
            dataframe.loc[condition, ["enter_short", "enter_tag"]] = (1, ENTRY_TAG)
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:
        return dataframe

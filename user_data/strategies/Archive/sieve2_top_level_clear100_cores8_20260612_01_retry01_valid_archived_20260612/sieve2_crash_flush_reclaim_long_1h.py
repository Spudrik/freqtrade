from __future__ import annotations

SIEVE_STAGE = "sieve2"
NOVEL_IDEA = True
SOURCE_STRATEGY = "novel"
SOURCE_RESULT_BATCH = "manual_20260612"
RESEARCH_PATH = "crash_exhaustion_1h"
UPDATE_HYPOTHESIS = "Capitulation flushes that wick and reclaim should be cleaner than raw fades."

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, IStrategy
from user_data.strategies.entry_sieve_tools import entry_sieve_minimal_roi, entry_sieve_stoploss

SIDE = "long"
ENTRY_MODE = "flush_reclaim"
ENTRY_TAG = "crash_flush_reclaim_long_1h"

def tagged_parameter(param):
    setattr(param, "batch_tags", ("family:entries", f"mode:{ENTRY_MODE}"))
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


class Sieve2CrashFlushReclaimLong1h(IStrategy):
    timeframe = "1h"
    can_short = False
    startup_candle_count = 320
    minimal_roi = entry_sieve_minimal_roi(0.03)
    stoploss = entry_sieve_stoploss(-0.03)
    process_only_new_candles = True
    use_exit_signal = False
    INTERFACE_VERSION = 3

    crash_lookback = tagged_parameter(IntParameter(8, 56, default=24, space="buy", optimize=True, load=True))
    compression_lookback = tagged_parameter(IntParameter(3, 18, default=8, space="buy", optimize=True, load=True))
    drop_pct_min = tagged_parameter(DecimalParameter(0.04, 0.35, default=0.12, decimals=2, space="buy", optimize=True, load=True))
    compression_width_max = tagged_parameter(DecimalParameter(0.03, 0.50, default=0.18, decimals=2, space="buy", optimize=True, load=True))
    volume_ratio_min = tagged_parameter(DecimalParameter(0.70, 3.50, default=1.10, decimals=2, space="buy", optimize=True, load=True))
    price_spike_pct_min = tagged_parameter(DecimalParameter(0.01, 0.20, default=0.04, decimals=3, space="buy", optimize=True, load=True))
    rsi_max = tagged_parameter(IntParameter(5, 30, default=10, space="buy", optimize=True, load=True))
    reclaim_buffer_pct = tagged_parameter(DecimalParameter(0.001, 0.03, default=0.006, decimals=3, space="buy", optimize=True, load=True))
    rsi_gate_mode = tagged_parameter(CategoricalParameter(["off", "soft", "hard"], default="soft", space="buy", optimize=True, load=True))

    def populate_indicators(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:
        close = pd.to_numeric(dataframe["close"], errors="coerce")
        high = pd.to_numeric(dataframe["high"], errors="coerce")
        low = pd.to_numeric(dataframe["low"], errors="coerce")
        open_ = pd.to_numeric(dataframe["open"], errors="coerce")
        volume = pd.to_numeric(dataframe["volume"], errors="coerce").fillna(0.0)
        look = int(self.crash_lookback.value)
        comp = int(self.compression_lookback.value)
        roll_high = close.rolling(look, min_periods=max(2, look // 2)).max()
        roll_low = close.rolling(look, min_periods=max(2, look // 2)).min()
        comp_high = high.rolling(comp, min_periods=max(2, comp // 2)).max()
        comp_low = low.rolling(comp, min_periods=max(2, comp // 2)).min()
        comp_mean = close.rolling(comp, min_periods=max(2, comp // 2)).mean()
        dataframe["rsi"] = _rsi(close, 14)
        dataframe["atr"] = _atr(dataframe, 14)
        dataframe["drop_pct"] = _safe_div(roll_high - close, roll_high).fillna(0.0)
        dataframe["compression_width"] = _safe_div(comp_high - comp_low, comp_mean).fillna(0.0)
        dataframe["compression_high"] = comp_high
        dataframe["compression_low"] = comp_low
        dataframe["compression_mid"] = (comp_high + comp_low) / 2.0
        dataframe["rolling_low"] = roll_low
        dataframe["volume_ratio"] = _safe_div(volume, volume.rolling(comp, min_periods=max(2, comp // 2)).mean()).fillna(0.0)
        dataframe["price_spike"] = _safe_div((close - close.shift(1)).abs(), close.shift(1).abs()).fillna(0.0)
        dataframe["body_ratio"] = _safe_div((close - open_).abs(), high - low).fillna(0.0)
        dataframe["lower_wick_ratio"] = _safe_div(np.minimum(open_, close) - low, high - low).fillna(0.0)
        dataframe["upper_wick_ratio"] = _safe_div(high - np.maximum(open_, close), high - low).fillna(0.0)
        dataframe["trend_mean"] = close.rolling(max(look * 2, comp * 2), min_periods=max(3, look)).mean()
        return dataframe

    def _rsi_ok(self, dataframe: DataFrame) -> Series:
        rsi = pd.to_numeric(dataframe["rsi"], errors="coerce")
        gate = str(self.rsi_gate_mode.value)
        if gate == "off":
            return pd.Series(True, index=dataframe.index)
        if gate == "hard":
            return rsi.le(float(self.rsi_max.value) - 2.0) & pd.to_numeric(dataframe["price_spike"], errors="coerce").ge(float(self.price_spike_pct_min.value))
        return rsi.le(float(self.rsi_max.value))

    def _entry_condition(self, dataframe: DataFrame) -> Series:
        drop_ok = pd.to_numeric(dataframe["drop_pct"], errors="coerce").ge(float(self.drop_pct_min.value))
        compress_ok = pd.to_numeric(dataframe["compression_width"], errors="coerce").le(float(self.compression_width_max.value))
        volume_ok = pd.to_numeric(dataframe["volume_ratio"], errors="coerce").ge(float(self.volume_ratio_min.value))
        spike_ok = volume_ok | pd.to_numeric(dataframe["price_spike"], errors="coerce").ge(float(self.price_spike_pct_min.value))
        close = pd.to_numeric(dataframe["close"], errors="coerce")
        open_ = pd.to_numeric(dataframe["open"], errors="coerce")
        high = pd.to_numeric(dataframe["high"], errors="coerce")
        low = pd.to_numeric(dataframe["low"], errors="coerce")
        reclaim = close.gt(pd.to_numeric(dataframe["compression_high"], errors="coerce") * (1.0 + float(self.reclaim_buffer_pct.value)))
        compression_mid = pd.to_numeric(dataframe["compression_mid"], errors="coerce")
        rolling_low = pd.to_numeric(dataframe["rolling_low"], errors="coerce")
        trend_mean = pd.to_numeric(dataframe["trend_mean"], errors="coerce")
        rsi_ok = self._rsi_ok(dataframe)

        if ENTRY_MODE == "drop_compress_reclaim":
            return drop_ok & compress_ok & reclaim & spike_ok & (close.gt(open_))
        if ENTRY_MODE == "flush_reclaim":
            return drop_ok & pd.to_numeric(dataframe["lower_wick_ratio"], errors="coerce").ge(0.55) & close.gt(compression_mid) & spike_ok
        if ENTRY_MODE == "rsi_extreme_reclaim":
            return drop_ok & rsi_ok & close.gt(compression_mid) & spike_ok
        if ENTRY_MODE == "liquidity_sweep_snapback":
            swept = low.lt(rolling_low * (1.0 - float(self.reclaim_buffer_pct.value)))
            return swept & close.gt(rolling_low) & pd.to_numeric(dataframe["lower_wick_ratio"], errors="coerce").ge(0.45) & spike_ok
        if ENTRY_MODE == "regime_pullback_reclaim":
            return drop_ok & close.lt(trend_mean) & close.gt(trend_mean * (1.0 - float(self.reclaim_buffer_pct.value))) & spike_ok
        if ENTRY_MODE == "volatility_squeeze_reversal":
            return compress_ok & reclaim & spike_ok
        if ENTRY_MODE == "double_flush_reversal":
            prior_low = rolling_low.shift(1)
            prior_prior_low = rolling_low.shift(2)
            first_flush = low.lt(prior_low * (1.0 - float(self.reclaim_buffer_pct.value)))
            second_flush = low.shift(1).lt(prior_prior_low * (1.0 - float(self.reclaim_buffer_pct.value)))
            return first_flush & second_flush & close.gt(compression_mid) & spike_ok
        return pd.Series(False, index=dataframe.index)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:
        condition = self._entry_condition(dataframe)
        dataframe.loc[condition, ["enter_long", "enter_tag"]] = (1, ENTRY_TAG)
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:
        return dataframe


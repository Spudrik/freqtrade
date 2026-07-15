from __future__ import annotations

SIEVE_STAGE = "sieve2"
NOVEL_IDEA = True
SOURCE_STRATEGY = "novel"
SOURCE_RESULT_BATCH = "manual_20260612"
RESEARCH_PATH = "mtf_std_1h"
UPDATE_HYPOTHESIS = "Daily MACD positive bias with 4h squeeze and 1h breakout may improve standard momentum entries."

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, IStrategy, merge_informative_pair
from user_data.strategies.entry_sieve_tools import entry_sieve_minimal_roi, entry_sieve_stoploss

SIDE = "long"
ENTRY_MODE = "daily_macd_volume_breakout"
ENTRY_TAG = "mtf_std_daily_macd_volume_breakout_long_1h"

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


def _ema(close: Series, length: int) -> Series:
    return close.ewm(span=length, adjust=False, min_periods=max(2, length // 2)).mean()


def _macd_hist(close: Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Series:
    macd = _ema(close, fast) - _ema(close, slow)
    sig = macd.ewm(span=signal, adjust=False, min_periods=max(2, signal // 2)).mean()
    return macd - sig


def _bb_width(close: Series, length: int, mult: float = 2.0) -> Series:
    mid = close.rolling(length, min_periods=max(2, length // 2)).mean()
    std = close.rolling(length, min_periods=max(2, length // 2)).std()
    upper = mid + mult * std
    lower = mid - mult * std
    return _safe_div(upper - lower, mid)


class Sieve2MtfStdDailyMacdVolumeBreakoutLong1h(IStrategy):
    timeframe = "1h"
    can_short = False
    startup_candle_count = 400
    minimal_roi = entry_sieve_minimal_roi(0.03)
    stoploss = entry_sieve_stoploss(-0.03)
    process_only_new_candles = True
    use_exit_signal = False
    INTERFACE_VERSION = 3

    ema_fast_len = tagged_parameter(IntParameter(8, 48, default=20, space="buy", optimize=True, load=True))
    ema_slow_len = tagged_parameter(IntParameter(20, 240, default=50, space="buy", optimize=True, load=True))
    bb_len = tagged_parameter(IntParameter(10, 60, default=20, space="buy", optimize=True, load=True))
    bb_width_max = tagged_parameter(DecimalParameter(0.02, 0.35, default=0.12, decimals=2, space="buy", optimize=True, load=True))
    volume_ratio_min = tagged_parameter(DecimalParameter(0.70, 3.50, default=1.10, decimals=2, space="buy", optimize=True, load=True))
    retest_buffer_pct = tagged_parameter(DecimalParameter(0.001, 0.03, default=0.006, decimals=3, space="buy", optimize=True, load=True))
    rsi_len = tagged_parameter(IntParameter(7, 28, default=14, space="buy", optimize=True, load=True))
    rsi_long_min = tagged_parameter(IntParameter(40, 70, default=52, space="buy", optimize=True, load=True))
    rsi_short_max = tagged_parameter(IntParameter(30, 60, default=48, space="buy", optimize=True, load=True))
    use_daily_trend = tagged_parameter(CategoricalParameter([False, True], default=True, space="buy", optimize=True, load=True))
    use_h4_compression = tagged_parameter(CategoricalParameter([False, True], default=True, space="buy", optimize=True, load=True))
    use_volume_filter = tagged_parameter(CategoricalParameter([False, True], default=True, space="buy", optimize=True, load=True))
    use_retest = tagged_parameter(CategoricalParameter([False, True], default=True, space="buy", optimize=True, load=True))
    use_momentum_filter = tagged_parameter(CategoricalParameter([False, True], default=False, space="buy", optimize=True, load=True))

    def informative_pairs(self):
        dp = getattr(self, "dp", None)
        if dp is None:
            return []
        try:
            pairs = self.dp.current_whitelist()
        except Exception:
            pairs = []
        return [(pair, "4h") for pair in pairs] + [(pair, "1d") for pair in pairs]

    def _features(self, frame: DataFrame, tf: str) -> DataFrame:
        close = pd.to_numeric(frame["close"], errors="coerce")
        high = pd.to_numeric(frame["high"], errors="coerce")
        low = pd.to_numeric(frame["low"], errors="coerce")
        volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
        ema_fast = int(self.ema_fast_len.value)
        ema_slow = int(self.ema_slow_len.value)
        bb_len = int(self.bb_len.value)
        rsi_len = int(self.rsi_len.value)
        frame[f"ema_fast_{tf}"] = _ema(close, ema_fast)
        frame[f"ema_slow_{tf}"] = _ema(close, ema_slow)
        frame[f"rsi_{tf}"] = _rsi(close, rsi_len)
        frame[f"bb_width_{tf}"] = _bb_width(close, bb_len, 2.0).fillna(0.0)
        frame[f"macd_hist_{tf}"] = _macd_hist(close).fillna(0.0)
        frame[f"volume_ratio_{tf}"] = _safe_div(volume, volume.rolling(20, min_periods=5).mean()).fillna(0.0)
        frame[f"atr_ratio_{tf}"] = _safe_div((high - low).rolling(14, min_periods=5).mean(), close).fillna(0.0)
        frame[f"prior_high_{tf}"] = high.shift(1).rolling(20, min_periods=5).max()
        frame[f"prior_low_{tf}"] = low.shift(1).rolling(20, min_periods=5).min()
        return frame

    def populate_indicators(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:
        frame = dataframe.copy()
        dp = getattr(self, "dp", None)
        if dp is not None and metadata and metadata.get("pair"):
            for tf in ("4h", "1d"):
                informative = dp.get_pair_dataframe(pair=metadata["pair"], timeframe=tf)
                if informative is not None and not informative.empty:
                    frame = merge_informative_pair(frame, self._features(informative.copy(), tf), self.timeframe, tf, ffill=True)
                    frame = frame.rename(columns=lambda col, tf=tf: col.replace(f"_{tf}_{tf}", f"_{tf}"))
        return self._features(frame, "1h")

    def _daily_trend_ok(self, frame: DataFrame, bullish: bool) -> Series:
        fast = pd.to_numeric(frame["ema_fast_1d"], errors="coerce")
        slow = pd.to_numeric(frame["ema_slow_1d"], errors="coerce")
        if not bool(self.use_daily_trend.value):
            return pd.Series(True, index=frame.index)
        return fast.gt(slow) if bullish else fast.lt(slow)

    def _h4_compression_ok(self, frame: DataFrame) -> Series:
        if not bool(self.use_h4_compression.value):
            return pd.Series(True, index=frame.index)
        return pd.to_numeric(frame["bb_width_4h"], errors="coerce").le(float(self.bb_width_max.value))

    def _volume_ok(self, frame: DataFrame) -> Series:
        if not bool(self.use_volume_filter.value):
            return pd.Series(True, index=frame.index)
        return pd.to_numeric(frame["volume_ratio_1h"], errors="coerce").ge(float(self.volume_ratio_min.value))

    def _momentum_ok(self, frame: DataFrame, bullish: bool) -> Series:
        if not bool(self.use_momentum_filter.value):
            return pd.Series(True, index=frame.index)
        if bullish:
            return pd.to_numeric(frame["rsi_1d"], errors="coerce").ge(float(self.rsi_long_min.value)) | pd.to_numeric(frame["macd_hist_1d"], errors="coerce").gt(0.0)
        return pd.to_numeric(frame["rsi_1d"], errors="coerce").le(float(self.rsi_short_max.value)) | pd.to_numeric(frame["macd_hist_1d"], errors="coerce").lt(0.0)

    def _entry_condition(self, frame: DataFrame) -> Series:
        close = pd.to_numeric(frame["close"], errors="coerce")
        high = pd.to_numeric(frame["high"], errors="coerce")
        low = pd.to_numeric(frame["low"], errors="coerce")
        ema_fast_1h = pd.to_numeric(frame["ema_fast_1h"], errors="coerce")
        ema_slow_1h = pd.to_numeric(frame["ema_slow_1h"], errors="coerce")
        ema_fast_4h = pd.to_numeric(frame["ema_fast_4h"], errors="coerce")
        ema_slow_4h = pd.to_numeric(frame["ema_slow_4h"], errors="coerce")
        prior_high_1d = pd.to_numeric(frame["prior_high_1d"], errors="coerce")
        prior_low_1d = pd.to_numeric(frame["prior_low_1d"], errors="coerce")
        rsi_1d = pd.to_numeric(frame["rsi_1d"], errors="coerce")
        rsi_1h = pd.to_numeric(frame["rsi_1h"], errors="coerce")
        rsi_4h = pd.to_numeric(frame["rsi_4h"], errors="coerce")
        volume_ok = self._volume_ok(frame)
        h4_ok = self._h4_compression_ok(frame)
        long_trend_ok = self._daily_trend_ok(frame, True)
        short_trend_ok = self._daily_trend_ok(frame, False)
        bull_momentum = self._momentum_ok(frame, True)
        bear_momentum = self._momentum_ok(frame, False)
        retest = float(self.retest_buffer_pct.value)
        breakout_1h = close.gt(high.shift(1).rolling(20, min_periods=5).max())
        breakdown_1h = close.lt(low.shift(1).rolling(20, min_periods=5).min())
        touch_fast = low.lt(ema_fast_1h * (1.0 + retest))
        reject_fast = high.gt(ema_fast_1h * (1.0 - retest))
        reclaim_fast = close.gt(ema_fast_1h)
        reject_fast_short = close.lt(ema_fast_1h)

        if ENTRY_MODE == "daily_ema_bb_volume_breakout":
            return long_trend_ok & h4_ok & volume_ok & close.gt(ema_fast_1h)
        if ENTRY_MODE == "daily_ema_bb_retest":
            return long_trend_ok & h4_ok & volume_ok & bool(self.use_retest.value) & touch_fast & reclaim_fast
        if ENTRY_MODE == "daily_macd_volume_breakout":
            return long_trend_ok & h4_ok & volume_ok & pd.to_numeric(frame["macd_hist_1d"], errors="coerce").gt(0.0) & breakout_1h
        if ENTRY_MODE == "daily_rsi_pullback_reclaim":
            return long_trend_ok & volume_ok & rsi_1d.ge(float(self.rsi_long_min.value)) & rsi_4h.ge(45.0) & touch_fast & reclaim_fast
        if ENTRY_MODE == "daily_prior_high_breakout":
            return long_trend_ok & h4_ok & volume_ok & close.gt(prior_high_1d * (1.0 + retest))
        if ENTRY_MODE == "daily_h4_trend_pullback_reclaim":
            return long_trend_ok & pd.to_numeric(frame["ema_fast_4h"], errors="coerce").gt(ema_slow_4h) & volume_ok & touch_fast & reclaim_fast
        if ENTRY_MODE == "daily_ema_bb_volume_breakdown":
            return short_trend_ok & h4_ok & volume_ok & close.lt(ema_fast_1h)
        if ENTRY_MODE == "daily_ema_bb_reject_short":
            return short_trend_ok & h4_ok & volume_ok & bool(self.use_retest.value) & reject_fast & reject_fast_short
        if ENTRY_MODE == "daily_macd_volume_breakdown":
            return short_trend_ok & h4_ok & volume_ok & pd.to_numeric(frame["macd_hist_1d"], errors="coerce").lt(0.0) & breakdown_1h
        if ENTRY_MODE == "daily_rsi_pullback_reject_short":
            return short_trend_ok & volume_ok & rsi_1d.le(float(self.rsi_short_max.value)) & rsi_4h.le(55.0) & reject_fast & reject_fast_short
        if ENTRY_MODE == "daily_prior_low_breakdown_short":
            return short_trend_ok & h4_ok & volume_ok & close.lt(prior_low_1d * (1.0 - retest))
        if ENTRY_MODE == "daily_h4_trend_pullback_reject_short":
            return short_trend_ok & pd.to_numeric(frame["ema_fast_4h"], errors="coerce").lt(ema_slow_4h) & volume_ok & reject_fast & reject_fast_short
        return pd.Series(False, index=frame.index)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:
        condition = self._entry_condition(dataframe)
        if SIDE == "long":
            dataframe.loc[condition, ["enter_long", "enter_tag"]] = (1, ENTRY_TAG)
        else:
            dataframe.loc[condition, ["enter_short", "enter_tag"]] = (1, ENTRY_TAG)
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:
        return dataframe

from __future__ import annotations

from datetime import datetime
import os
from typing import Any

import pandas as pd
from pandas import DataFrame, Series
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, IStrategy


LEVEL_LOOKBACK_CHOICES = [48, 96, 168, 336, 720]
LOCAL_LOOKBACK_CHOICES = [6, 12, 24, 48]
VOLUME_WINDOW_CHOICES = [12, 24, 48, 72]


def _pct_env(name: str, default_ratio: float) -> float:
    raw = str(os.environ.get(name) or "").strip()
    if not raw:
        return float(default_ratio)
    try:
        return max(0.0, float(raw)) / 100.0
    except ValueError:
        return float(default_ratio)


def _tag(param: Any, mode: str) -> Any:
    setattr(param, "batch_tags", ("family:entries", f"mode:{mode}"))
    return param


class TestPivotLongTrendPullback(IStrategy):
    INTERFACE_VERSION = 3
    can_short = False
    timeframe = "1h"
    startup_candle_count = 720
    process_only_new_candles = True

    minimal_roi = {"0": _pct_env("ENTRY_SIEVE_TAKE_PROFIT_PCT", 0.02)}
    stoploss = -_pct_env("ENTRY_SIEVE_STOPLOSS_PCT", 0.02)
    use_exit_signal = False
    use_custom_stoploss = False
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    ENTRY_TAG = "long_trend_pullback"
    ENTRY_SIDE = "long"
    ENTRY_KIND = "long_trend_pullback"
    MODE = "entry_long_trend_pullback"

    level_lookback = _tag(CategoricalParameter(LEVEL_LOOKBACK_CHOICES, default=168, space="buy", optimize=True, load=True), MODE)
    local_lookback = _tag(CategoricalParameter(LOCAL_LOOKBACK_CHOICES, default=24, space="buy", optimize=True, load=True), MODE)
    volume_window = _tag(CategoricalParameter(VOLUME_WINDOW_CHOICES, default=24, space="buy", optimize=True, load=True), MODE)
    confirm_bars = _tag(IntParameter(3, 48, default=12, space="buy", optimize=True, load=True), MODE)
    zone_pct = _tag(DecimalParameter(0.002, 0.040, decimals=3, default=0.012, space="buy", optimize=True, load=True), MODE)
    trigger_buffer_pct = _tag(DecimalParameter(0.000, 0.020, decimals=3, default=0.003, space="buy", optimize=True, load=True), MODE)
    volume_ratio_min = _tag(DecimalParameter(0.8, 2.5, decimals=1, default=1.1, space="buy", optimize=True, load=True), MODE)

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        close = pd.to_numeric(dataframe["close"], errors="coerce")
        high = pd.to_numeric(dataframe["high"], errors="coerce")
        low = pd.to_numeric(dataframe["low"], errors="coerce")
        volume = pd.to_numeric(dataframe["volume"], errors="coerce")
        dataframe["sieve_prev_close"] = close.shift(1)
        for lookback in LEVEL_LOOKBACK_CHOICES:
            min_periods = max(2, int(lookback) // 4)
            dataframe[f"sieve_resistance_{lookback}"] = high.shift(1).rolling(int(lookback), min_periods=min_periods).max()
            dataframe[f"sieve_support_{lookback}"] = low.shift(1).rolling(int(lookback), min_periods=min_periods).min()
        for lookback in LOCAL_LOOKBACK_CHOICES:
            dataframe[f"sieve_local_high_{lookback}"] = high.shift(1).rolling(int(lookback), min_periods=2).max()
            dataframe[f"sieve_local_low_{lookback}"] = low.shift(1).rolling(int(lookback), min_periods=2).min()
        for window in VOLUME_WINDOW_CHOICES:
            avg_volume = volume.shift(1).rolling(int(window), min_periods=2).mean().replace(0.0, pd.NA)
            dataframe[f"sieve_volume_ratio_{window}"] = volume / avg_volume
        dataframe["sieve_ema_fast"] = close.ewm(span=24, adjust=False, min_periods=12).mean()
        dataframe["sieve_ema_slow"] = close.ewm(span=96, adjust=False, min_periods=48).mean()
        dataframe["sieve_trend_up"] = dataframe["sieve_ema_fast"] > dataframe["sieve_ema_slow"]
        dataframe["sieve_trend_down"] = dataframe["sieve_ema_fast"] < dataframe["sieve_ema_slow"]
        return dataframe

    @staticmethod
    def _bool(mask: Series, index: pd.Index) -> Series:
        return pd.Series(mask, index=index).fillna(False).astype(bool)

    @staticmethod
    def _recent(mask: Series, bars: int) -> Series:
        return mask.fillna(False).astype("int8").shift(1).rolling(max(1, int(bars)), min_periods=1).max().gt(0)

    def _entry_mask(self, dataframe: DataFrame) -> Series:
        level = int(self.level_lookback.value)
        local = int(self.local_lookback.value)
        window = int(self.volume_window.value)
        zone = float(self.zone_pct.value)
        trigger = float(self.trigger_buffer_pct.value)
        volume_min = float(self.volume_ratio_min.value)
        confirm = int(self.confirm_bars.value)

        close = pd.to_numeric(dataframe["close"], errors="coerce")
        high = pd.to_numeric(dataframe["high"], errors="coerce")
        low = pd.to_numeric(dataframe["low"], errors="coerce")
        prev_close = pd.to_numeric(dataframe["sieve_prev_close"], errors="coerce")
        resistance = pd.to_numeric(dataframe[f"sieve_resistance_{level}"], errors="coerce")
        support = pd.to_numeric(dataframe[f"sieve_support_{level}"], errors="coerce")
        local_high = pd.to_numeric(dataframe[f"sieve_local_high_{local}"], errors="coerce")
        local_low = pd.to_numeric(dataframe[f"sieve_local_low_{local}"], errors="coerce")
        volume_ratio = pd.to_numeric(dataframe[f"sieve_volume_ratio_{window}"], errors="coerce")
        ema_fast = pd.to_numeric(dataframe["sieve_ema_fast"], errors="coerce")
        trend_up = pd.Series(dataframe["sieve_trend_up"], index=dataframe.index).fillna(False).astype(bool)
        trend_down = pd.Series(dataframe["sieve_trend_down"], index=dataframe.index).fillna(False).astype(bool)
        volume_ok = volume_ratio.ge(volume_min) | volume_ratio.isna()

        kind = self.ENTRY_KIND
        if kind == "long_res_break":
            line = resistance * (1.0 + trigger)
            mask = close.gt(line) & prev_close.le(line) & local_high.ge(resistance * (1.0 - zone)) & volume_ok
        elif kind == "long_sup_hold":
            mask = low.le(support * (1.0 + zone)) & local_low.le(support * (1.0 + zone)) & close.gt(support * (1.0 + trigger)) & volume_ok
        elif kind == "long_sup_reclaim":
            mask = low.lt(support * (1.0 - trigger)) & local_low.le(support * (1.0 + zone)) & close.gt(support * (1.0 + trigger)) & volume_ok
        elif kind == "long_res_retest_hold":
            break_mask = close.gt(resistance * (1.0 + trigger))
            mask = self._recent(break_mask, confirm) & low.le(resistance * (1.0 + zone)) & local_low.le(resistance * (1.0 + zone)) & close.ge(resistance) & volume_ok
        elif kind == "short_res_fail":
            mask = high.ge(resistance * (1.0 - zone)) & local_high.ge(resistance * (1.0 - zone)) & close.lt(resistance * (1.0 - trigger)) & volume_ok
        elif kind == "short_res_reclaim":
            mask = high.gt(resistance * (1.0 + trigger)) & local_high.ge(resistance * (1.0 - zone)) & close.lt(resistance * (1.0 - trigger)) & volume_ok
        elif kind == "short_sup_break":
            line = support * (1.0 - trigger)
            mask = close.lt(line) & prev_close.ge(line) & local_low.le(support * (1.0 + zone)) & volume_ok
        elif kind == "short_sup_retest_reject":
            break_mask = close.lt(support * (1.0 - trigger))
            mask = self._recent(break_mask, confirm) & high.ge(support * (1.0 - zone)) & local_high.ge(support * (1.0 - zone)) & close.le(support) & volume_ok
        elif kind == "long_trend_pullback":
            mask = trend_up & low.le(ema_fast * (1.0 + zone)) & close.gt(ema_fast * (1.0 + trigger)) & volume_ok
        elif kind == "short_trend_pullback":
            mask = trend_down & high.ge(ema_fast * (1.0 - zone)) & close.lt(ema_fast * (1.0 - trigger)) & volume_ok
        else:
            mask = pd.Series(False, index=dataframe.index)
        return self._bool(mask, dataframe.index)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""
        mask = self._entry_mask(dataframe)
        dataframe[f"plot_{self.ENTRY_TAG}"] = mask.astype(float)
        if self.ENTRY_SIDE == "short":
            dataframe.loc[mask, "enter_short"] = 1
        else:
            dataframe.loc[mask, "enter_long"] = 1
        dataframe.loc[mask, "enter_tag"] = self.ENTRY_TAG
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = ""
        return dataframe
"""Signal-candle and recent-swing invalidations for the frozen 8h bounce.

Exit thesis: compare exact OHLC lows known when the entry fills, separately
from the channel lower-rail provider.  No ATR or percentage buffer is invented.
Target control: fixed 6% favorable move.  Active sell parameter:
``invalidation_plan``.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2


ENTRY_MODE = "entry_geometry_ascending_channel_lower_bounce_long_8h"


TIMEFRAME = "8h"
SOURCE_STRATEGY = "sieve3_exit_fixed_tp_sl_from_complete_pattern_geometry_ascending_channel_lower_bounce_long_8h.py"
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume:pattern_geometry_8h_2020_23"
ENTRY_LOCK_STATUS = "executable_source_defaults_frozen_historical_promoted_snapshots_conflict"

STOP_STATE_KEY = "s3v2_ascending_channel_8h_signal_swing_stop"

INVALIDATION_PLANS: dict[str, int] = {
    "signal_candle_low": 1,
    "two_candle_swing_low": 2,
    "three_candle_swing_low": 3,
    "six_candle_swing_low": 6,
    "twelve_candle_swing_low": 12,
}




def _num(frame: DataFrame, column: str, default: float | Series=...) -> Series:
    if column not in frame.columns:
        if default is ...:
            raise KeyError(f'missing required column: {column!r}')
        return pd.to_numeric(
            pd.Series(default, index=frame.index), errors='coerce'
        ).replace([np.inf, -np.inf], np.nan)
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        raise KeyError(f'missing required column: {column!r}')
    return pd.Series(frame[column], index=frame.index).astype('boolean').fillna(False).astype(bool)


EXIT_FAMILY = 'signal_swing_invalidation'
PRIMARY_TRIGGER = 'bullish 8h bounce close above pg2_ascending_channel_lower'
PRIMARY_GUARD = 'locked volume/pressure guards and channel score/width qualification'
PRIMARY_TARGET = 'fixed 6% favorable move'
PRIMARY_INVALIDATION = 'entry-frozen signal low or selected recent swing low'
TARGET_PROVIDER = PRIMARY_TARGET
INVALIDATION_PROVIDER = PRIMARY_INVALIDATION
ENTRY_SOURCE_STAGE = 'sieve2'
RESEARCH_PATH = 'sieve3_exit_signal_swing_invalidation'
EXIT_HYPOTHESIS = 'Use the entry signal low or a short entry-frozen swing low as initial invalidation.'
ACTIVE_SELL_PARAMS = ('invalidation_plan',)
SIEVE_STAGE = 'sieve3'
SIDE = 'long'
ENTRY_TAG = 'geometry_ascending_channel_lower_bounce_long_8h'

class Sieve3V2SignalSwingInvalidationFromCompletePatternGeometryAscendingChannelLowerBounceLong8H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 180
    process_only_new_candles = True
    can_short = False
    minimal_roi = {"0": 100.0}
    stoploss = -0.99
    use_exit_signal = True
    exit_profit_only = False
    use_custom_stoploss = True
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    
    
    
    
    
    
    
    
    
    
    
    
    
    use_volume_guard = True
    volume_guard_window = 12
    volume_ratio_min = 1.0
    use_pressure_guard = True
    pressure_window = 24
    pressure_min = 0.1
    # Legacy fixed-off entry parameter: use_accumulation_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_body_direction_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_close_direction_guard=False (BooleanParameter, buy space).
    min_pattern_bars = 12
    max_pattern_bars = 72
    compression_max_width_atr = 2.0
    squeeze_active_width_atr = 2.0
    local_narrowing_min_ratio = 0.1
    min_line_score = 0.55
    min_containment = 0.88
    max_recent_touch_age_bars = 12
    channel_min_pattern_bars = 18
    channel_max_pattern_bars = 96
    channel_min_quality = 0.82
    channel_min_containment = 0.68
    channel_near_boundary_atr_mult = 0.7
    channel_breakout_atr_mult = 0.35
    channel_lifecycle_confirm_break_bars = 2
    score_min = 0.65
    width_atr_max = 6.0
    rail_buffer_pct = 0.004

    invalidation_plan = CategoricalParameter(tuple(INVALIDATION_PLANS), default="signal_candle_low", space="sell", optimize=True, load=True)
    invalidation_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    ACTIVE_SELL_PARAMS = ('invalidation_plan',)

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe = add_pattern_geometry_v2(
            dataframe, timeframe=self.timeframe, output_slots=1,
            include_triangle_patterns=False, include_wedge_patterns=False, include_compression_patterns=False,
            include_rectangle_patterns=False, include_ascending_channel_patterns=True, include_descending_channel_patterns=False,
            min_pattern_bars=int(self.min_pattern_bars), max_pattern_bars=int(self.max_pattern_bars),
            compression_max_width_atr=float(self.compression_max_width_atr), squeeze_active_width_atr=float(self.squeeze_active_width_atr),
            local_narrowing_min_ratio=float(self.local_narrowing_min_ratio), min_line_score=float(self.min_line_score),
            min_containment=float(self.min_containment), max_recent_touch_age_bars=int(self.max_recent_touch_age_bars),
            channel_min_pattern_bars=int(self.channel_min_pattern_bars), channel_max_pattern_bars=int(self.channel_max_pattern_bars),
            channel_min_quality=float(self.channel_min_quality), channel_min_containment=float(self.channel_min_containment),
            channel_near_boundary_atr_mult=float(self.channel_near_boundary_atr_mult), channel_breakout_atr_mult=float(self.channel_breakout_atr_mult),
            channel_lifecycle_confirm_break_bars=int(self.channel_lifecycle_confirm_break_bars), output_prefix="pg2",
        )
        
        return dataframe

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype="bool")
        close = _num(dataframe, "close")
        if bool(self.use_volume_guard):
            volume = _num(dataframe, "volume").clip(lower=0.0)
            window = int(self.volume_guard_window)
            baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
            guard &= volume.ge(baseline.mul(float(self.volume_ratio_min)))
        if bool(self.use_pressure_guard):
            open_ = _num(dataframe, "open")
            high = _num(dataframe, "high")
            low = _num(dataframe, "low")
            volume = _num(dataframe, "volume").clip(lower=0.0).fillna(0.0)
            candle_range = (high - low).replace(0.0, np.nan)
            body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
            close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
            pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
            directional_volume = (pressure * volume).fillna(0.0)
        if bool(self.use_pressure_guard):
            window = int(self.pressure_window)
            pressure_baseline = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
            pressure_ratio = directional_volume.rolling(window, min_periods=max(2, window // 3)).sum() / pressure_baseline
            guard &= pressure_ratio.le(-float(self.pressure_min)) if SIDE == "short" else pressure_ratio.ge(float(self.pressure_min))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        missing = sorted(set(("pg2_ascending_channel_lower", "pg2_ascending_channel_width_atr")).difference(dataframe.columns))
        if missing:
            raise KeyError(f"{type(self).__name__} entry dataframe is missing required source columns: {missing}")
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        lower = _num(dataframe, "pg2_ascending_channel_lower", np.nan)
        condition = _bool(dataframe, "pg2_ascending_channel_pattern_present") & _num(dataframe, "pg2_ascending_channel_indicator_score").ge(float(self.score_min)) & _num(dataframe, "pg2_ascending_channel_width_atr", np.nan).le(float(self.width_atr_max)) & _num(dataframe, "low").le(lower.mul(1.0 + float(self.rail_buffer_pct))) & _num(dataframe, "close").gt(lower) & _num(dataframe, "close").gt(_num(dataframe, "open"))
        condition &= self._common_guards(dataframe)
        
        valid = condition.fillna(False) & dataframe["volume"].gt(0.0) & dataframe["close"].notna()
        dataframe.loc[valid, "enter_long"] = 1
        dataframe.loc[valid, "enter_tag"] = ENTRY_TAG
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe

    @staticmethod
    def _utc(value: Any) -> pd.Timestamp:
        stamp = pd.Timestamp(value)
        return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")

    def _closed_rows(self, pair: str, current_time: datetime) -> DataFrame:
        if self.dp is None:
            raise RuntimeError("signal/swing invalidation requires Freqtrade's data provider")
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        close_times = pd.to_datetime(dataframe["date"], utc=True, errors="raise") + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        return dataframe.loc[close_times.le(self._utc(current_time))].sort_values("date")

    def order_filled(self, pair: str, trade: Any, order: Any, current_time: datetime, **kwargs: Any) -> None:
        _ = current_time, kwargs
        if order.ft_order_side != trade.entry_side or trade.get_custom_data(key=STOP_STATE_KEY) is not None:
            return
        closed = self._closed_rows(pair, order.order_date_utc)
        lookback = INVALIDATION_PLANS[str(self.invalidation_plan.value)]
        if len(closed) < lookback:
            raise RuntimeError("insufficient closed 8h candles for selected swing invalidation")
        signal = closed.iloc[-1]
        if not bool(signal["pg2_ascending_channel_pattern_present"]):
            raise RuntimeError("entry order does not map to an ascending-channel signal candle")
        stop_price = float(_num(closed.tail(lookback), "low").min())
        entry_price = float(order.safe_price)
        if not math.isfinite(stop_price) or stop_price <= 0.0 or stop_price >= entry_price:
            raise RuntimeError("entry-frozen signal/swing stop must be positive and below entry")
        trade.set_custom_data(key=STOP_STATE_KEY, value={"plan": str(self.invalidation_plan.value), "stop_price": stop_price})

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        _ = pair, trade, current_time, current_rate, kwargs
        return "s3v2_signal_swing_fixed_target_6" if current_profit >= 0.06 else None

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float:
        _ = pair, current_time, current_profit, after_fill, kwargs
        state = trade.get_custom_data(key=STOP_STATE_KEY)
        if not isinstance(state, dict) or "stop_price" not in state:
            raise RuntimeError("entry-frozen signal/swing invalidation is missing")
        return stoploss_from_absolute(float(state["stop_price"]), current_rate=current_rate, is_short=False, leverage=float(trade.leverage or 1.0))

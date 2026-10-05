"""Projected lower-channel invalidation for the frozen 8h bounce entry.

Exit thesis: invalidate against the exact slot-1 lower rail captured on the
closed signal candle and projected forward with that rail's emitted slope.
This file does not test signal-candle or generic swing lows.
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
ENTRY_TAG = "geometry_ascending_channel_lower_bounce_long_8h"
SIDE = "long"
TIMEFRAME = "8h"
SOURCE_STRATEGY = "sieve3_exit_fixed_tp_sl_from_complete_pattern_geometry_ascending_channel_lower_bounce_long_8h.py"
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume:pattern_geometry_8h_2020_23"
ENTRY_LOCK_STATUS = "executable_source_defaults_frozen_historical_promoted_snapshots_conflict"
EXIT_HYPOTHESIS = 'Compare projected lower-rail stops with closed-candle lower-rail failures.'
GEOMETRY_STATE_KEY = "s3v2_ascending_channel_8h_lower_invalidation"

INVALIDATION_PLANS: dict[str, tuple[str, float, int]] = {
    "projected_lower_stop_exact": ("stop", 0.0, 0),
    "projected_lower_stop_tenth_width": ("stop", 0.10, 0),
    "projected_lower_close_one": ("close", 0.0, 1),
    "projected_lower_close_two": ("close", 0.0, 2),
    "projected_lower_close_two_tenth_width": ("close", 0.10, 2),
}




def _num(frame: DataFrame, column: str, _default: float | Series=...) -> Series:
    if column in frame.columns:
        value = frame[column]
    elif _default is not ...:
        value = pd.Series(_default, index=frame.index)
    else:
        raise KeyError(column)
    return pd.to_numeric(value, errors='coerce').replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)


SIEVE_STAGE = 'sieve3'
ENTRY_SOURCE_STAGE = 'sieve2'
RESEARCH_PATH = 'sieve3_exit_channel_lower_invalidation'
ACTIVE_SELL_PARAMS = ('invalidation_plan',)
class Sieve3V2ChannelLowerInvalidationFromCompletePatternGeometryAscendingChannelLowerBounceLong8H(IStrategy):
    SIEVE_STAGE = 'sieve3'
    ENTRY_SOURCE_STAGE = 'sieve2'
    RESEARCH_PATH = 'sieve3_exit_channel_lower_invalidation'
    EXIT_HYPOTHESIS = 'Compare projected lower-rail stops with closed-candle lower-rail failures.'
    ACTIVE_SELL_PARAMS = ('invalidation_plan',)
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

    # Inactive buy declaration: use_volume_guard = BooleanParameter(default=True, space="buy"); fixed-on entry gate makes its guard unconditional.
    volume_guard_window = 12
    volume_ratio_min = 1.0
    # Inactive buy declaration: use_pressure_guard = BooleanParameter(default=True, space="buy"); fixed-on entry gate makes its guard unconditional.
    pressure_window = 24
    pressure_min = 0.1
    # Inactive buy declaration: use_accumulation_guard = BooleanParameter(default=False, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: use_body_direction_guard = BooleanParameter(default=False, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: use_close_direction_guard = BooleanParameter(default=False, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
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

    invalidation_plan = CategoricalParameter(tuple(INVALIDATION_PLANS), default="projected_lower_stop_exact", space="sell", optimize=True, load=True)
    invalidation_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')

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
        volume = _num(dataframe, "volume").clip(lower=0.0)
        window = int(self.volume_guard_window)
        baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
        guard &= volume.ge(baseline.mul(float(self.volume_ratio_min)))
        open_ = _num(dataframe, "open")
        high = _num(dataframe, "high")
        low = _num(dataframe, "low")
        volume = _num(dataframe, "volume").clip(lower=0.0).fillna(0.0)
        candle_range = (high - low).replace(0.0, np.nan)
        body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
        close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
        pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
        directional_volume = (pressure * volume).fillna(0.0)
        window = int(self.pressure_window)
        pressure_baseline = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
        pressure_ratio = directional_volume.rolling(window, min_periods=max(2, window // 3)).sum() / pressure_baseline
        guard &= pressure_ratio.le(-float(self.pressure_min)) if SIDE == "short" else pressure_ratio.ge(float(self.pressure_min))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        missing = sorted(set(('pg2_ascending_channel_lower', 'pg2_ascending_channel_width_atr')).difference(dataframe.columns))
        if missing:
            raise KeyError(f'{type(self).__name__} entry dataframe is missing required source columns: {missing}')
        _ = metadata
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = None
        lower = _num(dataframe, 'pg2_ascending_channel_lower', np.nan)
        condition = _bool(dataframe, 'pg2_ascending_channel_pattern_present') & _num(dataframe, 'pg2_ascending_channel_indicator_score').ge(float(self.score_min)) & _num(dataframe, 'pg2_ascending_channel_width_atr', np.nan).le(float(self.width_atr_max)) & _num(dataframe, 'low').le(lower.mul(1.0 + float(self.rail_buffer_pct))) & _num(dataframe, 'close').gt(lower) & _num(dataframe, 'close').gt(_num(dataframe, 'open'))
        condition &= self._common_guards(dataframe)
        valid = condition.fillna(False) & dataframe['volume'].gt(0.0) & dataframe['close'].notna()
        dataframe.loc[valid, 'enter_long'] = 1
        dataframe.loc[valid, 'enter_tag'] = ENTRY_TAG
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
            raise RuntimeError("lower-channel invalidation requires Freqtrade's data provider")
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        close_times = pd.to_datetime(dataframe["date"], utc=True, errors="raise") + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        return dataframe.loc[close_times.le(self._utc(current_time))].sort_values("date")

    def _signal_row(self, pair: str, order_time: datetime) -> Series:
        closed = self._closed_rows(pair, order_time)
        if closed.empty:
            raise RuntimeError("entry order has no closed 8h signal candle")
        row = closed.iloc[-1]
        required = ("pg2_slot_1_active", "pg2_slot_1_family", "pg2_slot_1_upper", "pg2_slot_1_lower", "pg2_slot_1_upper_slope", "pg2_slot_1_lower_slope")
        if any(column not in row.index for column in required):
            raise RuntimeError("signal candle is missing exact slot-1 channel geometry")
        if not bool(row["pg2_slot_1_active"]) or int(row["pg2_slot_1_family"]) != 5:
            raise RuntimeError("entry order does not map to an active slot-1 ascending channel")
        return row

    @staticmethod
    def _projected_rails(state: dict[str, Any], steps: int) -> tuple[float, float]:
        upper = float(state["upper"]) + float(state["upper_slope"]) * steps
        lower = float(state["lower"]) + float(state["lower_slope"]) * steps
        if not np.isfinite([upper, lower]).all() or upper <= lower:
            raise RuntimeError("projected ascending-channel rails are invalid")
        return upper, lower

    def order_filled(self, pair: str, trade: Any, order: Any, current_time: datetime, **kwargs: Any) -> None:
        _ = current_time, kwargs
        if order.ft_order_side != trade.entry_side or trade.get_custom_data(key=GEOMETRY_STATE_KEY) is not None:
            return
        row = self._signal_row(pair, order.order_date_utc)
        signal_close = self._utc(row["date"]) + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        state = {
            "plan": str(self.invalidation_plan.value),
            "signal_close": signal_close.isoformat(),
            "upper": float(row["pg2_slot_1_upper"]),
            "lower": float(row["pg2_slot_1_lower"]),
            "upper_slope": float(row["pg2_slot_1_upper_slope"]),
            "lower_slope": float(row["pg2_slot_1_lower_slope"]),
            "tightest_stop": None,
        }
        mode, buffer_fraction, _ = INVALIDATION_PLANS[state["plan"]]
        if mode == "stop":
            upper, lower = self._projected_rails(state, 1)
            candidate = lower - (upper - lower) * buffer_fraction
            if not math.isfinite(candidate) or candidate <= 0.0 or candidate >= float(order.safe_price):
                raise RuntimeError("projected lower-channel stop must be positive and below entry")
            state["tightest_stop"] = candidate
        trade.set_custom_data(key=GEOMETRY_STATE_KEY, value=state)

    def _state(self, trade: Any) -> dict[str, Any]:
        state = trade.get_custom_data(key=GEOMETRY_STATE_KEY)
        if not isinstance(state, dict):
            raise RuntimeError("entry-frozen lower-channel geometry is missing")
        return state

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        _ = current_rate, kwargs
        state = self._state(trade)
        mode, buffer_fraction, confirmations = INVALIDATION_PLANS[str(state["plan"])]
        if mode == "close":
            signal_close = self._utc(state["signal_close"])
            closed = self._closed_rows(pair, current_time)
            close_times = pd.to_datetime(closed["date"], utc=True, errors="raise") + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
            post_entry = closed.loc[close_times.gt(signal_close)].tail(confirmations)
            if len(post_entry) == confirmations:
                failures = []
                for _, row in post_entry.iterrows():
                    row_close = self._utc(row["date"]) + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
                    steps = max(1, int((row_close - signal_close).total_seconds() // (timeframe_to_minutes(self.timeframe) * 60)))
                    upper, lower = self._projected_rails(state, steps)
                    failures.append(float(row["close"]) < lower - (upper - lower) * buffer_fraction)
                if all(failures):
                    return f"s3v2_channel_lower_{state['plan']}"
        if current_profit >= 0.06:
            return "s3v2_channel_lower_fixed_target_6"
        return None

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float:
        _ = pair, current_profit, after_fill, kwargs
        state = self._state(trade)
        mode, buffer_fraction, _ = INVALIDATION_PLANS[str(state["plan"])]
        if mode == "close":
            stop_price = float(trade.open_rate) * 0.95
        else:
            signal_close = self._utc(state["signal_close"])
            elapsed = max(0.0, (self._utc(current_time) - signal_close).total_seconds())
            steps = int(elapsed // (timeframe_to_minutes(self.timeframe) * 60)) + 1
            upper, lower = self._projected_rails(state, steps)
            candidate = lower - (upper - lower) * buffer_fraction
            previous = float(state["tightest_stop"])
            stop_price = max(previous, candidate)
            if stop_price > previous and stop_price < current_rate:
                state["tightest_stop"] = stop_price
                trade.set_custom_data(key=GEOMETRY_STATE_KEY, value=state)
            elif stop_price >= current_rate:
                stop_price = previous
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=False, leverage=float(trade.leverage or 1.0))

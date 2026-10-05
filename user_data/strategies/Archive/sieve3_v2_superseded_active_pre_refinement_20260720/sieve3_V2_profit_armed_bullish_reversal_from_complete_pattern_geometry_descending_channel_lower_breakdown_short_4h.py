from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from user_data.Indicators.market_state import add_market_state

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2

ENTRY_MODE = "entry_geometry_descending_channel_lower_breakdown_short_4h"
ENTRY_TAG, SIDE, TIMEFRAME = "geometry_descending_channel_lower_breakdown_short_4h", "short", "4h"

SOURCE_STRATEGY = "user_data/strategies/sieve3_exit_fixed_tp_sl_from_complete_pattern_geometry_descending_channel_lower_breakdown_short_4h.py:Sieve3ExitFixedTpSlFromCompletePatternGeometryDescendingChannelLowerBreakdownShort4H"
SOURCE_RESULT_BATCH = "sieve3_exit_branch_generation_20260612"
RESEARCH_PATH = 'sieve3_exit_profit_armed_bullish_reversal'
ENTRY_SOURCE_STAGE = 'sieve2'

ENTRY_LOCK_STATUS = "unanimous_current_executable_defaults_locked; historical promoted snapshots conflict"
ENTRY_LOCK_SHA256 = "8b900a491a755ddc11adb21c8bd307b92af2b3848ae48a9e06766233dc8673b9"


def _tag(parameter: Any) -> Any:
    setattr(parameter, "batch_tags", ("family:entries", f"mode:{ENTRY_MODE}"))
    return parameter


def _num(frame: DataFrame, column: str, default: float=...) -> Series:
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


def _cross_below(value: Series, level: Series) -> Series:
    return value.lt(level) & value.shift(1).ge(level.shift(1))


EXIT_FAMILY = 'profit_armed_bullish_reversal'
PRIMARY_TRIGGER = '4h close cross below the descending-channel lower rail with locked score and width qualification'
PRIMARY_GUARD = 'enabled volume/pressure guards'
PRIMARY_TARGET = 'selected open-profit arm'
PRIMARY_INVALIDATION = 'selected bullish weakening or reversal evidence; fixed 5% adverse stop'
TARGET_PROVIDER = PRIMARY_TARGET
INVALIDATION_PROVIDER = PRIMARY_INVALIDATION
EXIT_HYPOTHESIS = 'Once the short is profitable, exit on explicit bullish weakening or reversal in 4h OHLCV evidence.'
ACTIVE_SELL_PARAMS = ('reversal_plan',)
SIEVE_STAGE = 'sieve3'
SIDE = 'short'
ENTRY_TAG = 'geometry_descending_channel_lower_breakdown_short_4h'

class Sieve3V2ProfitArmedBullishReversalFromCompletePatternGeometryDescendingChannelLowerBreakdownShort4H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe, startup_candle_count, process_only_new_candles, can_short = TIMEFRAME, 180, True, True
    minimal_roi, stoploss = {"0": 100.0}, -0.12
    use_exit_signal, use_custom_stoploss, position_adjustment_enable = True, True, False
    trailing_stop, ignore_roi_if_entry_signal = False, False

    use_volume_guard = True
    volume_guard_window = 48
    volume_ratio_min = 0.8
    use_pressure_guard = True
    pressure_window = 24
    pressure_min = 0.15
    # Legacy fixed-off entry parameter: use_accumulation_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_body_direction_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_close_direction_guard=False (BooleanParameter, buy space).
    min_pattern_bars = 12
    max_pattern_bars = 72
    compression_max_width_atr = 2.0
    squeeze_active_width_atr = 2.0
    local_narrowing_min_ratio = 0.1
    min_line_score = 0.5
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

    reversal_plan = CategoricalParameter(("arm_1_engulf", "arm_2_engulf", "arm_3_engulf", "arm_1_two_bull", "arm_2_two_bull", "arm_3_two_bull", "arm_2_three_higher", "arm_3_three_higher", "arm_1_pressure_zero", "arm_2_pressure_zero", "arm_2_pressure_05", "arm_3_pressure_05", "arm_2_lower_low_reject", "arm_3_lower_low_reject"), default="arm_2_two_bull", space="sell", optimize=True, load=True)
    reversal_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    ACTIVE_SELL_PARAMS = ('reversal_plan',)
    REVERSAL_PLANS = {
        "arm_1_engulf": (0.01, "engulf", 0.0), "arm_2_engulf": (0.02, "engulf", 0.0), "arm_3_engulf": (0.03, "engulf", 0.0),
        "arm_1_two_bull": (0.01, "two_bull", 0.0), "arm_2_two_bull": (0.02, "two_bull", 0.0), "arm_3_two_bull": (0.03, "two_bull", 0.0),
        "arm_2_three_higher": (0.02, "three_higher", 0.0), "arm_3_three_higher": (0.03, "three_higher", 0.0),
        "arm_1_pressure_zero": (0.01, "pressure", 0.0), "arm_2_pressure_zero": (0.02, "pressure", 0.0),
        "arm_2_pressure_05": (0.02, "pressure", 0.05), "arm_3_pressure_05": (0.03, "pressure", 0.05),
        "arm_2_lower_low_reject": (0.02, "lower_low_reject", 0.0), "arm_3_lower_low_reject": (0.03, "lower_low_reject", 0.0),
    }

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = add_pattern_geometry_v2(dataframe, timeframe=self.timeframe, output_slots=1, include_triangle_patterns=False, include_wedge_patterns=False, include_compression_patterns=False, include_rectangle_patterns=False, include_ascending_channel_patterns=False, include_descending_channel_patterns=True, min_pattern_bars=int(self.min_pattern_bars), max_pattern_bars=int(self.max_pattern_bars), compression_max_width_atr=float(self.compression_max_width_atr), squeeze_active_width_atr=float(self.squeeze_active_width_atr), local_narrowing_min_ratio=float(self.local_narrowing_min_ratio), min_line_score=float(self.min_line_score), min_containment=float(self.min_containment), max_recent_touch_age_bars=int(self.max_recent_touch_age_bars), channel_min_pattern_bars=int(self.channel_min_pattern_bars), channel_max_pattern_bars=int(self.channel_max_pattern_bars), channel_min_quality=float(self.channel_min_quality), channel_min_containment=float(self.channel_min_containment), channel_near_boundary_atr_mult=float(self.channel_near_boundary_atr_mult), channel_breakout_atr_mult=float(self.channel_breakout_atr_mult), channel_lifecycle_confirm_break_bars=int(self.channel_lifecycle_confirm_break_bars), output_prefix="pg2")
        dataframe = add_market_state(dataframe, window=24, prefix="s2m")
        return dataframe

    def _common_guards(self, dataframe: DataFrame) -> Series:
        close, volume = _num(dataframe, "close"), _num(dataframe, "volume").clip(lower=0.0)
        window = int(self.volume_guard_window)
        guard = volume.ge(volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan).mul(float(self.volume_ratio_min)))
        open_, high, low = (_num(dataframe, name) for name in ("open", "high", "low"))
        candle_range = (high - low).replace(0.0, np.nan)
        pressure = ((((close - open_) / candle_range).clip(-1.0, 1.0).fillna(0.0) + ((((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)).fillna(0.0)) / 2.0).clip(-1.0, 1.0)
        window = int(self.pressure_window)
        baseline = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
        guard &= (pressure * volume.fillna(0.0)).rolling(window, min_periods=max(2, window // 3)).sum().div(baseline).le(-float(self.pressure_min))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        missing = sorted(set(("pg2_descending_channel_lower", "pg2_descending_channel_width_atr")).difference(dataframe.columns))
        if missing:
            raise KeyError(f"{type(self).__name__} entry dataframe is missing required source columns: {missing}")
        dataframe["enter_long"], dataframe["enter_short"], dataframe["enter_tag"] = 0, 0, None
        lower = _num(dataframe, "pg2_descending_channel_lower", np.nan).mul(1.0 - float(self.rail_buffer_pct))
        condition = _bool(dataframe, "pg2_descending_channel_pattern_present") & _num(dataframe, "pg2_descending_channel_indicator_score").ge(float(self.score_min)) & _num(dataframe, "pg2_descending_channel_width_atr", np.nan).le(float(self.width_atr_max)) & _cross_below(_num(dataframe, "close"), lower)
        condition &= self._common_guards(dataframe)
        valid = condition.fillna(False) & dataframe["volume"].gt(0.0) & dataframe["close"].notna()
        dataframe.loc[valid, "enter_short"] = 1
        dataframe.loc[valid, "enter_tag"] = ENTRY_TAG
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"], dataframe["exit_short"], dataframe["exit_tag"] = 0, 0, None
        return dataframe

    def _post_entry(self, pair: str, trade: Any, current_time: datetime) -> DataFrame:
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        result = frame.copy()
        result["date"] = pd.to_datetime(result["date"], utc=True, errors="raise")
        entered = pd.Timestamp(getattr(trade, "date_entry_fill_utc", None) or trade.open_date_utc)
        entered = entered.tz_localize("UTC") if entered.tzinfo is None else entered.tz_convert("UTC")
        cutoff = pd.Timestamp(current_time)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        close_times = result["date"] + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        return result.loc[result["date"].ge(entered) & close_times.le(cutoff)].sort_values("date")

    def _bullish_reversal(self, frame: DataFrame, mode: str, threshold: float) -> bool:
        if len(frame) < 2:
            return False
        latest, previous = frame.iloc[-1], frame.iloc[-2]
        if mode == "engulf":
            return bool(latest["close"] > latest["open"] and previous["close"] < previous["open"] and latest["close"] >= previous["open"] and latest["open"] <= previous["close"])
        if mode == "two_bull":
            tail = frame.tail(2)
            return bool((tail["close"] > tail["open"]).all() and tail["close"].is_monotonic_increasing)
        if mode == "three_higher":
            return len(frame) >= 3 and bool(frame["close"].tail(3).diff().dropna().gt(0.0).all())
        if mode == "lower_low_reject":
            return bool(latest["low"] < previous["low"] and latest["close"] > previous["close"] and latest["close"] > latest["open"])
        return bool(float(latest["s2m_pressure_ratio"]) >= threshold)

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | bool | None:
        _ = current_rate, kwargs
        arm, mode, threshold = self.REVERSAL_PLANS[str(self.reversal_plan.value)]
        if current_profit < arm:
            return None
        return f"profit_armed_bullish_{mode}" if self._bullish_reversal(self._post_entry(pair, trade, current_time), mode, threshold) else None

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float | None:
        _ = pair, current_time, current_profit, after_fill, kwargs
        return stoploss_from_absolute(float(trade.open_rate) * 1.05, current_rate=current_rate, is_short=True, leverage=float(getattr(trade, "leverage", 1.0) or 1.0))


__all__ = ["Sieve3V2ProfitArmedBullishReversalFromCompletePatternGeometryDescendingChannelLowerBreakdownShort4H"]

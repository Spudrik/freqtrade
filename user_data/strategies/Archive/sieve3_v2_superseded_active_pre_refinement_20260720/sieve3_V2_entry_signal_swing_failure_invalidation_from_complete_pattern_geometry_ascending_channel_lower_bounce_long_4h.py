"""Entry-signal swing failure invalidations for the 4h ascending-channel bounce.

Source entry: the current executable Sieve3 fixed-TP/SL representative.
Primary trigger: bullish lower-rail bounce candle. Primary guard: source volume/pressure guards.
Target provider: none. Invalidation provider: the entry signal candle's observed low.
Active Hyperopt parameter: swing_failure_plan. No branch-local inactive parameters.
This stays separate from lower-rail failure because candle swing risk is not the fitted channel line.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import (
    CategoricalParameter,
    IStrategy,
)
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2

ENTRY_MODE = "entry_geometry_ascending_channel_lower_bounce_long_4h"
ENTRY_TAG = "geometry_ascending_channel_lower_bounce_long_4h"
SIDE = "long"
TIMEFRAME = "4h"

SIEVE_STAGE = 'sieve3'
SOURCE_STRATEGY = (
    "user_data/strategies/"
    "sieve3_exit_fixed_tp_sl_from_complete_pattern_geometry_ascending_channel_lower_bounce_long_4h.py:"
    "Sieve3ExitFixedTpSlFromCompletePatternGeometryAscendingChannelLowerBounceLong4H"
)
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume"
RESEARCH_PATH = 'sieve3_exit_entry_signal_swing_failure_invalidation'
ENTRY_SOURCE_STAGE = 'sieve2'
ENTRY_LOCK_STATUS = "current executable defaults locked; promoted parameter artifact not found"
EXIT_HYPOTHESIS = "Exit when later price action invalidates the bounce candle's exact swing low."

SWING_FAILURE_PLANS = (
    "one_close_below_signal_low",
    "one_close_below_signal_low_0_25",
    "two_closes_below_signal_low",
    "wick_below_signal_low_bearish_close",
)




def tagged_exit_parameter(param: Any) -> Any:
    setattr(param, "batch_tags", ("family:exits", "mode:sieve3_exit"))
    return param


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


class Sieve3V2EntrySignalSwingFailureInvalidationFromCompletePatternGeometryAscendingChannelLowerBounceLong4H(
    IStrategy
):
    SIEVE_STAGE = 'sieve3'
    ENTRY_SOURCE_STAGE = 'sieve2'
    RESEARCH_PATH = 'sieve3_exit_entry_signal_swing_failure_invalidation'
    EXIT_HYPOTHESIS = "Exit when later price action invalidates the bounce candle's exact swing low."
    ACTIVE_SELL_PARAMS = ('swing_failure_plan',)
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 180
    process_only_new_candles = True
    can_short = False

    minimal_roi = {"0": 100.0}
    stoploss = -0.03
    use_exit_signal = True
    use_custom_stoploss = False
    position_adjustment_enable = False
    trailing_stop = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False


    # Legacy fixed-on entry parameter: use_volume_guard=True; guard is now unconditional.
    volume_guard_window = 48
    volume_ratio_min = 1.0
    # Legacy fixed-on entry parameter: use_pressure_guard=True; guard is now unconditional.
    pressure_window = 12
    pressure_min = 0.05
    # Legacy fixed-off entry parameters: accumulation/body/close-direction guards were unreachable.

    min_pattern_bars = 12
    max_pattern_bars = 72
    compression_max_width_atr = 2.0
    squeeze_active_width_atr = 2.0
    local_narrowing_min_ratio = 0.1
    min_line_score = 0.55
    min_containment = 0.9
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

    swing_failure_plan = tagged_exit_parameter(
        CategoricalParameter(
            SWING_FAILURE_PLANS,
            default="one_close_below_signal_low",
            space="sell",
            optimize=True,
            load=True,
        )
    )

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

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe = add_pattern_geometry_v2(
            dataframe,
            timeframe=self.timeframe,
            output_slots=1,
            include_triangle_patterns=False,
            include_wedge_patterns=False,
            include_compression_patterns=False,
            include_rectangle_patterns=False,
            include_ascending_channel_patterns=True,
            include_descending_channel_patterns=False,
            min_pattern_bars=int(self.min_pattern_bars),
            max_pattern_bars=int(self.max_pattern_bars),
            compression_max_width_atr=float(self.compression_max_width_atr),
            squeeze_active_width_atr=float(self.squeeze_active_width_atr),
            local_narrowing_min_ratio=float(self.local_narrowing_min_ratio),
            min_line_score=float(self.min_line_score),
            min_containment=float(self.min_containment),
            max_recent_touch_age_bars=int(self.max_recent_touch_age_bars),
            channel_min_pattern_bars=int(self.channel_min_pattern_bars),
            channel_max_pattern_bars=int(self.channel_max_pattern_bars),
            channel_min_quality=float(self.channel_min_quality),
            channel_min_containment=float(self.channel_min_containment),
            channel_near_boundary_atr_mult=float(self.channel_near_boundary_atr_mult),
            channel_breakout_atr_mult=float(self.channel_breakout_atr_mult),
            channel_lifecycle_confirm_break_bars=int(self.channel_lifecycle_confirm_break_bars),
            output_prefix="pg2",
        )
        return dataframe

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype="bool")
        close = _num(dataframe, "close")
        volume = _num(dataframe, "volume").clip(lower=0.0)
        window = int(self.volume_guard_window)
        baseline = volume.shift(1).rolling(
            window, min_periods=max(2, window // 3)
        ).mean().replace(0.0, np.nan)
        guard &= volume.ge(baseline.mul(float(self.volume_ratio_min)))
        open_ = _num(dataframe, "open")
        high = _num(dataframe, "high")
        low = _num(dataframe, "low")
        candle_range = (high - low).replace(0.0, np.nan)
        body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
        close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
        pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
        directional_volume = (pressure * volume.fillna(0.0)).fillna(0.0)
        window = int(self.pressure_window)
        pressure_baseline = volume.rolling(
            window, min_periods=max(2, window // 3)
        ).sum().replace(0.0, np.nan)
        pressure_ratio = directional_volume.rolling(
            window, min_periods=max(2, window // 3)
        ).sum() / pressure_baseline
        guard &= pressure_ratio.ge(float(self.pressure_min))
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
        timestamp = pd.Timestamp(value)
        return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")

    def _closed_frame(self, pair: str, current_time: datetime) -> DataFrame:
        if getattr(self, "dp", None) is None:
            raise RuntimeError("signal-swing invalidation requires Freqtrade's data provider")
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        required = {"date", "open", "low", "close"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"signal-swing invalidation missing OHLCV columns: {missing}")
        closes_at = pd.to_datetime(frame["date"], utc=True, errors="raise") + pd.to_timedelta(
            timeframe_to_minutes(self.timeframe), unit="m"
        )
        return frame.loc[closes_at.le(self._utc(current_time))].sort_values("date")

    def _entry_signal_low(self, pair: str, trade: Any) -> float:
        entry_frame = self._closed_frame(pair, trade.open_date_utc)
        if entry_frame.empty:
            raise RuntimeError("signal-swing invalidation has no closed entry-signal candle")
        signal_low = float(entry_frame.iloc[-1]["low"])
        if not math.isfinite(signal_low) or signal_low <= 0.0:
            raise ValueError("entry signal candle low is invalid")
        return signal_low

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | None:
        _ = current_rate, current_profit, kwargs
        frame = self._closed_frame(pair, current_time)
        if frame.empty:
            return None
        signal_low = self._entry_signal_low(pair, trade)
        plan = str(self.swing_failure_plan.value)
        threshold = signal_low * (0.9975 if plan.endswith("_0_25") else 1.0)
        latest = frame.iloc[-1]
        one_close_failed = float(latest["close"]) < threshold
        two_closes_failed = len(frame) >= 2 and bool(_num(frame.tail(2), "close").lt(signal_low).all())
        wick_bearish_failed = (
            float(latest["low"]) < signal_low
            and float(latest["close"]) < float(latest["open"])
        )
        should_exit = (
            (plan.startswith("one_close") and one_close_failed)
            or (plan == "two_closes_below_signal_low" and two_closes_failed)
            or (plan == "wick_below_signal_low_bearish_close" and wick_bearish_failed)
        )
        return f"entry_signal_swing_failure_{plan}" if should_exit else None


ACTIVE_SELL_PARAMS = ('swing_failure_plan',)

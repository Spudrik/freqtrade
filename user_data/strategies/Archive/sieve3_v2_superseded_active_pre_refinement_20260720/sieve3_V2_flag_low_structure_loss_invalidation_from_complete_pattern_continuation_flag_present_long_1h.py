"""Flag-low structure invalidation for the promoted bullish 1h flag entry.

Source strategy: sieve2_continuation_flag_present_long_1h
Exit family: flag-low structure loss
Primary trigger: bullish pat_flag_pattern_present at the promoted score floor
Primary guard: promoted volume and bullish-pressure guards
Target provider: none
Invalidation provider: entry-frozen pat_flag_lower
Active Hyperopt parameters: exit_plan
Branch-local parameters: none; each category binds touch/close confirmation and buffer
Split rationale: losing the flag low breaks the setup structure without requiring a prior breakout.
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
from user_data.Indicators.pattern_continuation import add_pattern_continuation

ENTRY_MODE = "entry_continuation_flag_present_long_1h"
ENTRY_TAG = "continuation_flag_present_long_1h"
SIDE = "long"
TIMEFRAME = "1h"
SIEVE_STAGE = "sieve3"
SOURCE_STRATEGY = "sieve2_continuation_flag_present_long_1h"
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume:pattern_continuation_1h_2021_22"
RESEARCH_PATH = "sieve3_exit_flag_low_structure_loss_invalidation"
ENTRY_SOURCE_STAGE = "sieve2"
EXIT_HYPOTHESIS = "Compare an executable flag-low stop with close-confirmed structural failure below the frozen lower rail."
ENTRY_LOCK_STATUS = "authoritative exported buy params over archived executable defaults"

# kind, consecutive closes, close buffer
FLAG_LOW_PLANS: dict[str, tuple[str, int, float]] = {
    "flag_low_touch_stop": ("touch", 0, 0.0),
    "flag_low_close_loss": ("close", 1, 0.0),
    "flag_low_two_close_loss": ("close", 2, 0.0),
    "flag_low_close_loss_buffer_0_25": ("close", 1, 0.0025),
    "flag_low_two_close_loss_buffer_0_25": ("close", 2, 0.0025),
    "flag_low_close_loss_buffer_0_5": ("close", 1, 0.005),
}




def _num(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)



def _bool(frame: DataFrame, column: str) -> Series:
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)



ACTIVE_SELL_PARAMS = ("exit_plan",)



class Sieve3V2FlagLowStructureLossInvalidationFromCompletePatternContinuationFlagPresentLong1H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 160
    process_only_new_candles = True
    can_short = False
    ACTIVE_SELL_PARAMS = ("exit_plan",)
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
    volume_ratio_min = 1.3
    use_pressure_guard = True
    pressure_window = 48
    pressure_min = 0.1
    # Legacy fixed-off entry parameter: use_accumulation_guard=False; its branch is unreachable.
    # Legacy fixed-off entry parameters: use_body_direction_guard=False and
    # use_close_direction_guard=False; their directional branches are unreachable.
    score_min = 0.55
    rail_buffer_pct = 0.0
    exit_plan = CategoricalParameter(tuple(FLAG_LOW_PLANS), default="flag_low_close_loss", space="sell", optimize=True, load=True)
    exit_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = add_pattern_continuation(dataframe, timeframe=self.timeframe, min_flag_quality=float(self.score_min), min_pennant_quality=float(self.score_min))
        return dataframe

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype="bool")
        close = _num(dataframe, "close")
        if self.use_volume_guard:
            volume = _num(dataframe, "volume").clip(lower=0.0)
            window = int(self.volume_guard_window)
            baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
            guard &= volume.ge(baseline.mul(float(self.volume_ratio_min)))
        if self.use_pressure_guard:
            open_ = _num(dataframe, "open")
            high = _num(dataframe, "high")
            low = _num(dataframe, "low")
            volume = _num(dataframe, "volume").clip(lower=0.0).fillna(0.0)
            candle_range = (high - low).replace(0.0, np.nan)
            body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
            close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
            pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
            directional_volume = (pressure * volume).fillna(0.0)
        if self.use_pressure_guard:
            window = int(self.pressure_window)
            pressure_baseline = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
            pressure_ratio = directional_volume.rolling(window, min_periods=max(2, window // 3)).sum() / pressure_baseline
            guard &= pressure_ratio.ge(float(self.pressure_min))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        condition = _bool(dataframe, "pat_flag_pattern_present") & _num(dataframe, "pat_flag_direction").ge(1) & _num(dataframe, "pat_flag_indicator_score").ge(float(self.score_min))
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
        timestamp = pd.Timestamp(value)
        return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")

    def _closed_frame(self, pair: str, current_time: datetime) -> DataFrame:
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
        close_times = dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        return frame.loc[close_times.le(self._utc(current_time))].sort_values("date").copy()

    def _flag_low(self, pair: str, trade: Any) -> float:
        frame = self._closed_frame(pair, trade.open_date_utc)
        if frame.empty:
            raise RuntimeError("No closed flag signal candle is available at trade entry.")
        level = float(frame.iloc[-1]["pat_flag_lower"])
        if not math.isfinite(level) or level <= 0.0 or level >= float(trade.open_rate):
            raise RuntimeError("The flag signal candle has no directional lower-rail invalidation.")
        return level

    def _post_entry_frame(self, pair: str, trade: Any, current_time: datetime) -> DataFrame:
        frame = self._closed_frame(pair, current_time)
        dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
        filled_at = self._utc(
            getattr(trade, "date_entry_fill_utc", None) or trade.open_date_utc
        )
        return frame.loc[dates.ge(filled_at)]

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        _ = current_rate, current_profit, kwargs
        kind, confirmations, buffer = FLAG_LOW_PLANS[str(self.exit_plan.value)]
        if kind == "touch":
            return None
        frame = self._post_entry_frame(pair, trade, current_time)
        if len(frame) < confirmations:
            return None
        threshold = self._flag_low(pair, trade) * (1.0 - buffer)
        lost = _num(frame, "close").tail(confirmations).lt(threshold).all()
        return "s3v2_flag_low_structure_lost" if bool(lost) else None

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float:
        _ = current_time, current_profit, after_fill, kwargs
        kind, _, _ = FLAG_LOW_PLANS[str(self.exit_plan.value)]
        stop_price = self._flag_low(pair, trade) if kind == "touch" else float(trade.open_rate) * 0.92
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=False, leverage=float(trade.leverage or 1.0))

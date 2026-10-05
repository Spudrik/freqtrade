"""Profit-armed bearish weakening and opposite-flag exits for the bullish 1h flag entry.

Source strategy: sieve2_continuation_flag_present_long_1h
Exit family: profit-armed bearish weakening/reversal
Primary trigger: bullish pat_flag_pattern_present at the promoted score floor
Primary guard: promoted volume and bullish-pressure guards
Target provider: selected open-profit gate
Invalidation provider: bearish candles or a later bearish pat_flag source event
Active Hyperopt parameters: exit_plan
Branch-local parameters: none; plan names bind profit gate and bearish evidence
Split rationale: every mode asks when bearish evidence is strong enough to bank an existing profit.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, IStrategy
from user_data.Indicators.pattern_continuation import add_pattern_continuation

ENTRY_MODE = "entry_continuation_flag_present_long_1h"


TIMEFRAME = "1h"


SOURCE_STRATEGY = "sieve2_continuation_flag_present_long_1h"
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume:pattern_continuation_1h_2021_22"
RESEARCH_PATH = 'sieve3_exit_profit_armed_bearish_weakening_reversal'
ENTRY_SOURCE_STAGE = 'sieve2'

ENTRY_LOCK_STATUS = "authoritative exported buy params over archived executable defaults"

PROFIT_ARMED_PLANS: dict[str, tuple[float, str]] = {
    "bear_close_after_2pct": (0.02, "bear_close"),
    "two_bear_closes_after_1_5pct": (0.015, "two_bear_closes"),
    "bear_2of3_after_1pct": (0.01, "bear_2of3"),
    "bear_2of3_after_2pct": (0.02, "bear_2of3"),
    "opposite_flag_after_any_profit": (0.0, "opposite_flag"),
    "opposite_flag_after_1pct": (0.01, "opposite_flag"),
    "opposite_flag_bear_close_after_0_5pct": (0.005, "opposite_flag_bear_close"),
    "opposite_flag_or_bear_2of3_after_2pct": (0.02, "opposite_flag_or_bear_2of3"),
}




def _num(frame: DataFrame, column: str, default: float | Series=0.0) -> Series:
    if column not in frame.columns:
        raise KeyError(f'missing required column: {column!r}')
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        raise KeyError(f'missing required column: {column!r}')
    return pd.Series(frame[column], index=frame.index).astype('boolean').fillna(False).astype(bool)


EXIT_FAMILY = 'profit_armed_bearish_weakening_reversal'
PRIMARY_TRIGGER = 'bullish pat_flag_pattern_present at the locked score floor'
PRIMARY_GUARD = 'locked volume and bullish-pressure guards'
PRIMARY_TARGET = 'selected open-profit gate'
PRIMARY_INVALIDATION = 'bearish closes or a later bearish pat_flag source event'
TARGET_PROVIDER = PRIMARY_TARGET
INVALIDATION_PROVIDER = PRIMARY_INVALIDATION
EXIT_HYPOTHESIS = 'Bank profit only after bearish candle weakness or a later opposite flag challenges the bullish continuation.'
ACTIVE_SELL_PARAMS = ('exit_plan',)
SIEVE_STAGE = 'sieve3'
SIDE = 'long'
ENTRY_TAG = 'continuation_flag_present_long_1h'

class Sieve3V2ProfitArmedBearishWeakeningReversalFromCompletePatternContinuationFlagPresentLong1H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 160
    process_only_new_candles = True
    can_short = False
    minimal_roi = {"0": 100.0}
    stoploss = -0.03
    use_exit_signal = True
    exit_profit_only = False
    use_custom_stoploss = False
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
    # Legacy fixed-off entry parameter: use_accumulation_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_body_direction_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_close_direction_guard=False (BooleanParameter, buy space).
    score_min = 0.55
    rail_buffer_pct = 0.0

    exit_plan = CategoricalParameter(tuple(PROFIT_ARMED_PLANS), default="opposite_flag_or_bear_2of3_after_2pct", space="sell", optimize=True, load=True)
    exit_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    ACTIVE_SELL_PARAMS = ('exit_plan',)

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

    def _post_entry_frame(self, pair: str, trade: Any, current_time: datetime) -> DataFrame:
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
        close_times = dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        closed = frame.loc[close_times.le(self._utc(current_time))].copy()
        closed_dates = pd.to_datetime(closed["date"], utc=True, errors="raise")
        entered = self._utc(getattr(trade, "date_entry_fill_utc", None) or trade.open_date_utc)
        return closed.loc[closed_dates.ge(entered)].sort_values("date")

    def _bearish_evidence(self, frame: DataFrame, evidence: str) -> bool:
        if frame.empty:
            return False
        bearish = _num(frame, "close").lt(_num(frame, "open"))
        opposite_flag = bool(
            _bool(frame, "pat_flag_pattern_present").iloc[-1]
            and _num(frame, "pat_flag_direction").iloc[-1] <= -1
            and _num(frame, "pat_flag_indicator_score").iloc[-1] >= float(self.score_min)
        )
        one_bear = bool(bearish.iloc[-1])
        two_bear = len(bearish) >= 2 and bool(bearish.tail(2).all())
        two_of_three = len(bearish) >= 3 and int(bearish.tail(3).sum()) >= 2
        if evidence == "bear_close":
            return one_bear
        if evidence == "two_bear_closes":
            return two_bear
        if evidence == "bear_2of3":
            return two_of_three
        if evidence == "opposite_flag":
            return opposite_flag
        if evidence == "opposite_flag_bear_close":
            return opposite_flag and one_bear
        return opposite_flag or two_of_three

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        _ = current_rate, kwargs
        profit_gate, evidence = PROFIT_ARMED_PLANS[str(self.exit_plan.value)]
        if current_profit < profit_gate:
            return None
        frame = self._post_entry_frame(pair, trade, current_time)
        return f"s3v2_profit_armed_{evidence}" if self._bearish_evidence(frame, evidence) else None

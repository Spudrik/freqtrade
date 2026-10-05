"""Profit-armed bullish structure reversal exit for the bearish 1h BOS entry.

Source strategy: sieve2_bos_bear_continuation_short_1h
Exit family: profit-armed source weakening/reversal
Primary trigger: ms_bos_to_bear
Primary guard: promoted volume and bearish-pressure guards
Target provider: none; profits run until opposing market structure appears
Invalidation provider: fixed 3% initial stop
Active Hyperopt parameters: exit_plan
Branch-local parameters: none; each category binds signal, confirmation, and profit gate
Split rationale: all modes ask when a profitable bearish BOS should yield to bullish structure.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, IStrategy
from user_data.Indicators.pattern_bos_choch import add_bos_choch


ENTRY_MODE = "entry_bos_bear_continuation_short_1h"


TIMEFRAME = "1h"


SOURCE_STRATEGY = "sieve2_bos_bear_continuation_short_1h"
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume:pattern_continuation_1h_2021_22"
RESEARCH_PATH = 'sieve3_exit_profit_armed_bullish_structure_reversal'
ENTRY_SOURCE_STAGE = 'sieve2'


REVERSAL_PLANS: dict[str, tuple[str, str, float]] = {
    "choch_event_profit_0_5": ("choch", "event", 0.005),
    "choch_event_profit_1": ("choch", "event", 0.01),
    "choch_event_profit_2": ("choch", "event", 0.02),
    "choch_event_profit_3": ("choch", "event", 0.03),
    "bos_event_profit_0_5": ("bos", "event", 0.005),
    "bos_event_profit_1": ("bos", "event", 0.01),
    "bos_event_profit_2": ("bos", "event", 0.02),
    "bos_event_profit_3": ("bos", "event", 0.03),
    "either_event_profit_1": ("either", "event", 0.01),
    "either_event_profit_2": ("either", "event", 0.02),
    "either_event_profit_3": ("either", "event", 0.03),
    "choch_then_bull_close_profit_1": ("choch", "then_bull_close", 0.01),
    "choch_then_bull_close_profit_2": ("choch", "then_bull_close", 0.02),
    "either_then_bull_close_profit_1": ("either", "then_bull_close", 0.01),
    "either_then_bull_close_profit_2": ("either", "then_bull_close", 0.02),
    "bull_state_2_closes_profit_1": ("state", "state_2", 0.01),
    "bull_state_2_closes_profit_2": ("state", "state_2", 0.02),
    "bull_state_2_closes_profit_3": ("state", "state_2", 0.03),
}




def _num(frame: DataFrame, column: str, default: float | Series=0.0) -> Series:
    if column not in frame.columns:
        raise KeyError(f'missing required column: {column!r}')
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        raise KeyError(f'missing required column: {column!r}')
    return pd.Series(frame[column], index=frame.index).astype('boolean').fillna(False).astype(bool)


EXIT_FAMILY = 'profit_armed_bullish_structure_reversal'
PRIMARY_TRIGGER = 'confirmed 1h ms_bos_to_bear'
PRIMARY_GUARD = 'locked volume and bearish-pressure guards'
PRIMARY_TARGET = 'selected open-profit gate'
PRIMARY_INVALIDATION = 'bullish CHoCH, BOS, or positive ms_state confirmation'
TARGET_PROVIDER = PRIMARY_TARGET
INVALIDATION_PROVIDER = PRIMARY_INVALIDATION
EXIT_HYPOTHESIS = 'Once profitable, bullish CHoCH/BOS evidence may identify when bearish continuation has weakened.'
ACTIVE_SELL_PARAMS = ('exit_plan',)
SIEVE_STAGE = 'sieve3'
SIDE = 'short'
ENTRY_TAG = 'bos_bear_continuation_short_1h'

class Sieve3V2ProfitArmedBullishStructureReversalFromBosBearContinuationShort1H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 160
    process_only_new_candles = True
    can_short = True
    minimal_roi = {"0": 100.0}
    stoploss = -0.03
    use_exit_signal = True
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    
    
    
    
    
    
    
    
    
    
    
    
    
    use_volume_guard = True
    volume_guard_window = 24
    volume_ratio_min = 1.6
    use_pressure_guard = True
    pressure_window = 48
    pressure_min = 0.2
    # Legacy fixed-off entry parameter: use_accumulation_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_body_direction_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_close_direction_guard=False (BooleanParameter, buy space).
    strength = 3
    min_prominence_atr = 0.35
    min_pivot_spacing_bars = 2
    max_pivot_age_bars = 96
    breakout_buffer_atr = 0.3
    # Legacy fixed-off entry parameter: use_state_guard=False (BooleanParameter, buy space).

    exit_plan = CategoricalParameter(tuple(REVERSAL_PLANS), default="either_event_profit_2", space="sell", optimize=True, load=True)
    exit_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    ACTIVE_SELL_PARAMS = ('exit_plan',)

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = add_bos_choch(dataframe, strength=int(self.strength), min_prominence_atr=float(self.min_prominence_atr), min_pivot_spacing_bars=int(self.min_pivot_spacing_bars), max_pivot_age_bars=int(self.max_pivot_age_bars), breakout_buffer_atr=float(self.breakout_buffer_atr), prefix="ms")
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
            guard &= pressure_ratio.le(-float(self.pressure_min))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        condition = _bool(dataframe, "ms_bos_to_bear")
        condition &= self._common_guards(dataframe)
        
        valid = condition.fillna(False) & dataframe["volume"].gt(0.0) & dataframe["close"].notna()
        dataframe.loc[valid, "enter_short"] = 1
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

    def _closed_trade_frame(self, pair: str, trade: Any, current_time: datetime) -> DataFrame:
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
        close_times = dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        closed = frame.loc[close_times.le(self._utc(current_time))].copy()
        trade_open = self._utc(getattr(trade, "date_entry_fill_utc", None) or trade.open_date_utc)
        closed_dates = pd.to_datetime(closed["date"], utc=True, errors="raise")
        return closed.loc[closed_dates.ge(trade_open)].sort_values("date")

    @staticmethod
    def _signal_matches(frame: DataFrame, signal: str, confirmation: str) -> bool:
        if len(frame) == 0:
            return False
        choch = _bool(frame, "ms_choch_to_bull")
        bos = _bool(frame, "ms_bos_to_bull")
        event = choch if signal == "choch" else bos if signal == "bos" else choch | bos
        if confirmation == "event":
            return bool(event.iloc[-1])
        if confirmation == "then_bull_close":
            recent = frame.tail(3)
            recent_event = event.tail(3)
            latest_bullish = float(recent["close"].iloc[-1]) > float(recent["open"].iloc[-1])
            return bool(recent_event.any()) and latest_bullish
        if confirmation == "state_2":
            return len(frame) >= 2 and bool(_num(frame, "ms_state").tail(2).gt(0.0).all())
        return False

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        _ = current_rate, kwargs
        signal, confirmation, profit_gate = REVERSAL_PLANS[str(self.exit_plan.value)]
        if current_profit < profit_gate:
            return None
        frame = self._closed_trade_frame(pair, trade, current_time)
        if self._signal_matches(frame, signal, confirmation):
            return "s3v2_profit_armed_bullish_structure"
        return None

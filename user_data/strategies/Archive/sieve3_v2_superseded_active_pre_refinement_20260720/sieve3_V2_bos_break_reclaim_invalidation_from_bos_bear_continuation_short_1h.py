"""Broken-support reclaim invalidation for the promoted bearish 1h BOS entry.

Source strategy: sieve2_bos_bear_continuation_short_1h
Exit family: entry-specific close-based invalidation
Primary trigger: ms_bos_to_bear
Primary guard: promoted volume and bearish-pressure guards
Target provider: fixed 3% control target
Invalidation provider: closes reclaiming the entry-candle ms_break_level
Active Hyperopt parameters: exit_plan
Branch-local parameters: none; each category binds level buffer, close count, and structure confirmation
Split rationale: all modes define failure as acceptance back above the broken BOS support.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, IStrategy
from user_data.Indicators.pattern_bos_choch import add_bos_choch

ENTRY_MODE = "entry_bos_bear_continuation_short_1h"
ENTRY_TAG = "bos_bear_continuation_short_1h"
SIDE = "short"
TIMEFRAME = "1h"

SIEVE_STAGE = 'sieve3'
SOURCE_STRATEGY = "sieve2_bos_bear_continuation_short_1h"
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume:pattern_continuation_1h_2021_22"
RESEARCH_PATH = 'sieve3_exit_bos_break_reclaim_invalidation'
ENTRY_SOURCE_STAGE = 'sieve2'
EXIT_HYPOTHESIS = 'Acceptance back above the broken support may invalidate the bearish BOS before a distant stop.'

RECLAIM_PLANS: dict[str, tuple[float, int, str]] = {
    "exact_reclaim_1_close": (0.0, 1, "none"),
    "exact_reclaim_2_closes": (0.0, 2, "none"),
    "exact_reclaim_3_closes": (0.0, 3, "none"),
    "buffer_0_25atr_reclaim_1_close": (0.25, 1, "none"),
    "buffer_0_25atr_reclaim_2_closes": (0.25, 2, "none"),
    "buffer_0_25atr_reclaim_3_closes": (0.25, 3, "none"),
    "buffer_0_5atr_reclaim_1_close": (0.50, 1, "none"),
    "buffer_0_5atr_reclaim_2_closes": (0.50, 2, "none"),
    "buffer_0_5atr_reclaim_3_closes": (0.50, 3, "none"),
    "exact_reclaim_1_close_plus_bull_choch": (0.0, 1, "choch"),
    "exact_reclaim_1_close_plus_bull_bos": (0.0, 1, "bos"),
    "exact_reclaim_2_closes_plus_bull_structure": (0.0, 2, "either"),
    "buffer_0_25atr_reclaim_1_plus_bull_choch": (0.25, 1, "choch"),
    "buffer_0_25atr_reclaim_2_plus_bull_structure": (0.25, 2, "either"),
    "buffer_0_5atr_reclaim_1_plus_bull_structure": (0.50, 1, "either"),
}




def _num(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)


ACTIVE_SELL_PARAMS = ('exit_plan',)
class Sieve3V2BosBreakReclaimInvalidationFromBosBearContinuationShort1H(IStrategy):
    SIEVE_STAGE = 'sieve3'
    ENTRY_SOURCE_STAGE = 'sieve2'
    RESEARCH_PATH = 'sieve3_exit_bos_break_reclaim_invalidation'
    EXIT_HYPOTHESIS = 'Acceptance back above the broken support may invalidate the bearish BOS before a distant stop.'
    ACTIVE_SELL_PARAMS = ('exit_plan',)
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 160
    process_only_new_candles = True
    can_short = True
    minimal_roi = {"0": 0.03}
    stoploss = -0.99
    use_exit_signal = True
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    # Legacy fixed-on entry parameters: volume and pressure guards are unconditional.
    volume_guard_window = 24
    volume_ratio_min = 1.6
    pressure_window = 48
    pressure_min = 0.2
    # Legacy fixed-off entry parameters: accumulation/body/close-direction guards were unreachable.
    strength = 3
    min_prominence_atr = 0.35
    min_pivot_spacing_bars = 2
    max_pivot_age_bars = 96
    breakout_buffer_atr = 0.3
    # Legacy fixed-off entry parameter: use_state_guard=False made the state branch unreachable.

    exit_plan = CategoricalParameter(tuple(RECLAIM_PLANS), default="exact_reclaim_2_closes", space="sell", optimize=True, load=True)
    exit_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = add_bos_choch(dataframe, strength=int(self.strength), min_prominence_atr=float(self.min_prominence_atr), min_pivot_spacing_bars=int(self.min_pivot_spacing_bars), max_pivot_age_bars=int(self.max_pivot_age_bars), breakout_buffer_atr=float(self.breakout_buffer_atr), include_diagnostics=True, prefix="ms")
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
        candle_range = (high - low).replace(0.0, np.nan)
        body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
        close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
        pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
        directional_volume = (pressure * volume.fillna(0.0)).fillna(0.0)
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

    def _closed_frame(self, pair: str, current_time: datetime) -> DataFrame:
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
        close_times = dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        return frame.loc[close_times.le(self._utc(current_time))].sort_values("date").copy()

    def _entry_reclaim_level(self, pair: str, trade: Any) -> tuple[float, float]:
        frame = self._closed_frame(pair, trade.open_date_utc)
        if frame.empty:
            raise RuntimeError("No closed BOS signal candle is available at trade entry.")
        row = frame.iloc[-1]
        break_level = float(row["ms_break_level"])
        atr = float(row["ms_atr"])
        if not math.isfinite(break_level) or not math.isfinite(atr) or atr <= 0.0:
            raise RuntimeError("The BOS signal candle does not contain a valid break level and ATR.")
        return break_level, atr

    def _post_entry_frame(self, pair: str, trade: Any, current_time: datetime) -> DataFrame:
        frame = self._closed_frame(pair, current_time)
        dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
        return frame.loc[dates.ge(self._utc(trade.open_date_utc))]

    @staticmethod
    def _confirmation_matches(frame: DataFrame, confirmation: str) -> bool:
        recent = frame.tail(3)
        choch = bool(_bool(recent, "ms_choch_to_bull").any())
        bos = bool(_bool(recent, "ms_bos_to_bull").any())
        if confirmation == "choch":
            return choch
        if confirmation == "bos":
            return bos
        if confirmation == "either":
            return choch or bos
        return True

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        _ = current_rate, current_profit, kwargs
        atr_buffer, close_count, confirmation = RECLAIM_PLANS[str(self.exit_plan.value)]
        break_level, atr = self._entry_reclaim_level(pair, trade)
        frame = self._post_entry_frame(pair, trade, current_time)
        reclaim = _num(frame, "close").gt(break_level + atr * atr_buffer)
        if len(reclaim) < close_count or not bool(reclaim.tail(close_count).all()):
            return None
        if not self._confirmation_matches(frame, confirmation):
            return None
        return "s3v2_bos_break_reclaim_invalidation"

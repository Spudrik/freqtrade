"""Swing-high invalidation stops for the promoted bearish 1h BOS entry.

Source strategy: sieve2_bos_bear_continuation_short_1h
Exit family: entry-specific price-stop invalidation
Primary trigger: ms_bos_to_bear
Primary guard: promoted volume and bearish-pressure guards
Target provider: fixed 3% control target
Invalidation provider: entry-candle ms_invalidation_level / active swing high
Active Hyperopt parameters: exit_plan
Branch-local parameters: none; each category binds ATR or percentage clearance beyond the swing
Split rationale: all modes place the initial stop beyond the same frozen structural swing.
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
from user_data.Indicators.pattern_bos_choch import add_bos_choch

ENTRY_MODE = "entry_bos_bear_continuation_short_1h"
ENTRY_TAG = "bos_bear_continuation_short_1h"
SIDE = "short"
TIMEFRAME = "1h"

SIEVE_STAGE = 'sieve3'
SOURCE_STRATEGY = "sieve2_bos_bear_continuation_short_1h"
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume:pattern_continuation_1h_2021_22"
RESEARCH_PATH = 'sieve3_exit_bos_swing_high_invalidation'
ENTRY_SOURCE_STAGE = 'sieve2'
EXIT_HYPOTHESIS = 'The active swing high that defines the bearish BOS should provide its coherent initial stop.'

SWING_STOP_PLANS: dict[str, tuple[str, float]] = {
    "swing_exact": ("atr", 0.0),
    "swing_plus_0_25atr": ("atr", 0.25),
    "swing_plus_0_5atr": ("atr", 0.50),
    "swing_plus_0_75atr": ("atr", 0.75),
    "swing_plus_1atr": ("atr", 1.00),
    "swing_plus_1_5atr": ("atr", 1.50),
    "swing_plus_2atr": ("atr", 2.00),
    "swing_plus_3atr": ("atr", 3.00),
    "swing_plus_0_25pct": ("pct", 0.0025),
    "swing_plus_0_5pct": ("pct", 0.005),
    "swing_plus_1pct": ("pct", 0.01),
    "swing_plus_1_5pct": ("pct", 0.015),
}




def _num(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)


ACTIVE_SELL_PARAMS = ('exit_plan',)
class Sieve3V2BosSwingHighInvalidationFromBosBearContinuationShort1H(IStrategy):
    SIEVE_STAGE = 'sieve3'
    ENTRY_SOURCE_STAGE = 'sieve2'
    RESEARCH_PATH = 'sieve3_exit_bos_swing_high_invalidation'
    EXIT_HYPOTHESIS = 'The active swing high that defines the bearish BOS should provide its coherent initial stop.'
    ACTIVE_SELL_PARAMS = ('exit_plan',)
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 160
    process_only_new_candles = True
    can_short = True
    minimal_roi = {"0": 0.03}
    stoploss = -0.99
    use_exit_signal = False
    use_custom_stoploss = True
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
    # use_accumulation_guard = False (locked off; gated entry branch removed).
    # use_body_direction_guard = False (locked off; gated entry branch removed).
    # use_close_direction_guard = False (locked off; gated entry branch removed).
    strength = 3
    min_prominence_atr = 0.35
    min_pivot_spacing_bars = 2
    max_pivot_age_bars = 96
    breakout_buffer_atr = 0.3
    use_state_guard = False

    exit_plan = CategoricalParameter(tuple(SWING_STOP_PLANS), default="swing_exact", space="sell", optimize=True, load=True)
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
        if bool(self.use_state_guard):
            condition &= _num(dataframe, "ms_state").le(0)
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

    def _entry_swing(self, pair: str, trade: Any) -> tuple[float, float]:
        frame = self._closed_frame(pair, trade.open_date_utc)
        if frame.empty:
            raise RuntimeError("No closed BOS signal candle is available at trade entry.")
        row = frame.iloc[-1]
        invalidation = float(row["ms_invalidation_level"])
        atr = float(row["ms_atr"])
        if not math.isfinite(invalidation) or not math.isfinite(atr) or atr <= 0.0:
            raise RuntimeError("The BOS signal candle does not contain a valid swing-high invalidation and ATR.")
        if invalidation <= float(trade.open_rate):
            raise RuntimeError("The short BOS swing-high invalidation is not above the trade entry.")
        return invalidation, atr

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float:
        _ = current_time, current_profit, after_fill, kwargs
        clearance_mode, clearance = SWING_STOP_PLANS[str(self.exit_plan.value)]
        invalidation, atr = self._entry_swing(pair, trade)
        stop_price = (
            invalidation + atr * clearance
            if clearance_mode == "atr"
            else invalidation * (1.0 + clearance)
        )
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=True, leverage=float(trade.leverage or 1.0))

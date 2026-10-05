"""BOS measured-move target exits for the promoted bearish 1h BOS entry.

Source strategy: sieve2_bos_bear_continuation_short_1h
Exit family: source-structure measured-move target
Primary trigger: ms_bos_to_bear
Primary guard: promoted volume and bearish-pressure guards
Target provider: entry-candle break level projected by the break-to-swing-high range
Invalidation provider: entry-candle active swing high
Active Hyperopt parameters: exit_plan
Branch-local parameters: none; each category binds projection and target confirmation
Split rationale: every mode exits the full short at a downside BOS projection.
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
RESEARCH_PATH = 'sieve3_exit_bos_measured_move_target'
ENTRY_SOURCE_STAGE = 'sieve2'
EXIT_HYPOTHESIS = 'The bearish BOS range may project coherent downside objectives for full profit-taking.'

MEASURED_TARGET_PLANS: dict[str, tuple[float, str]] = {
    "half_touch": (0.50, "touch"),
    "three_quarter_touch": (0.75, "touch"),
    "full_touch": (1.00, "touch"),
    "one_and_quarter_touch": (1.25, "touch"),
    "one_and_half_touch": (1.50, "touch"),
    "double_touch": (2.00, "touch"),
    "half_then_bull_close": (0.50, "bull_close"),
    "three_quarter_then_bull_close": (0.75, "bull_close"),
    "full_then_bull_close": (1.00, "bull_close"),
    "one_and_quarter_then_bull_close": (1.25, "bull_close"),
    "half_then_bull_2of3": (0.50, "bull_2of3"),
    "three_quarter_then_bull_2of3": (0.75, "bull_2of3"),
    "full_then_bull_2of3": (1.00, "bull_2of3"),
    "one_and_quarter_then_bull_2of3": (1.25, "bull_2of3"),
}




def _num(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)


ACTIVE_SELL_PARAMS = ('exit_plan',)
class Sieve3V2BosMeasuredMoveTargetFromBosBearContinuationShort1H(IStrategy):
    SIEVE_STAGE = 'sieve3'
    ENTRY_SOURCE_STAGE = 'sieve2'
    RESEARCH_PATH = 'sieve3_exit_bos_measured_move_target'
    EXIT_HYPOTHESIS = 'The bearish BOS range may project coherent downside objectives for full profit-taking.'
    ACTIVE_SELL_PARAMS = ('exit_plan',)
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 160
    process_only_new_candles = True
    can_short = True
    minimal_roi = {"0": 100.0}
    stoploss = -0.99
    use_exit_signal = True
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

    exit_plan = CategoricalParameter(tuple(MEASURED_TARGET_PLANS), default="full_touch", space="sell", optimize=True, load=True)
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

    def _entry_structure(self, pair: str, trade: Any) -> tuple[float, float]:
        frame = self._closed_frame(pair, trade.open_date_utc)
        if frame.empty:
            raise RuntimeError("No closed BOS signal candle is available at trade entry.")
        row = frame.iloc[-1]
        break_level = float(row["ms_break_level"])
        invalidation = float(row["ms_invalidation_level"])
        if not math.isfinite(break_level) or not math.isfinite(invalidation) or invalidation <= break_level:
            raise RuntimeError("The BOS signal candle does not contain a valid break-to-invalidation range.")
        return break_level, invalidation

    def _post_entry_frame(self, pair: str, trade: Any, current_time: datetime) -> DataFrame:
        frame = self._closed_frame(pair, current_time)
        dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
        return frame.loc[dates.ge(self._utc(trade.open_date_utc))]

    @staticmethod
    def _target_confirmed(frame: DataFrame, target: float, confirmation: str) -> bool:
        touched = _num(frame, "low").le(target)
        positions = np.flatnonzero(touched.to_numpy())
        if len(positions) == 0:
            return False
        if confirmation == "touch":
            return True
        post_touch = frame.iloc[int(positions[0]):]
        bullish = _num(post_touch, "close").gt(_num(post_touch, "open"))
        if confirmation == "bull_close":
            return bool(bullish.iloc[-1])
        return len(bullish) >= 3 and int(bullish.tail(3).sum()) >= 2

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        _ = current_rate, current_profit, kwargs
        multiplier, confirmation = MEASURED_TARGET_PLANS[str(self.exit_plan.value)]
        break_level, invalidation = self._entry_structure(pair, trade)
        target = break_level - (invalidation - break_level) * multiplier
        if target >= float(trade.open_rate):
            return None
        frame = self._post_entry_frame(pair, trade, current_time)
        if self._target_confirmed(frame, target, confirmation):
            return "s3v2_bos_measured_move_target"
        return None

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float:
        _ = current_time, current_profit, after_fill, kwargs
        _, invalidation = self._entry_structure(pair, trade)
        return stoploss_from_absolute(invalidation, current_rate=current_rate, is_short=True, leverage=float(trade.leverage or 1.0))

"""Flag-width and pole-projection target zones for the bullish 1h flag entry.

Source strategy: sieve2_continuation_flag_present_long_1h
Exit family: continuation-pattern target-zone behavior
Primary trigger: bullish pat_flag_pattern_present at the promoted score floor
Primary guard: promoted volume and bullish-pressure guards
Target provider: entry-frozen flag rails and pat_impulse_up_pct
Invalidation provider: fixed 3% control stop; source invalidations are isolated elsewhere
Active Hyperopt parameters: exit_plan
Branch-local parameters: none; each category binds projection and zone confirmation
Split rationale: all modes compare full exits at targets derived from the same entry-candle flag geometry.
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
from user_data.Indicators.pattern_continuation import add_pattern_continuation

ENTRY_MODE = "entry_continuation_flag_present_long_1h"
ENTRY_GEOMETRY_STATE_KEY = "s3v2_flag_target_entry_geometry"


TIMEFRAME = "1h"


SOURCE_STRATEGY = "sieve2_continuation_flag_present_long_1h"
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume:pattern_continuation_1h_2021_22"
RESEARCH_PATH = 'sieve3_exit_pattern_target_zone'
ENTRY_SOURCE_STAGE = 'sieve2'

ENTRY_LOCK_STATUS = "authoritative exported buy params over archived executable defaults"

TARGET_ZONE_PLANS: dict[str, tuple[str, str]] = {
    "flag_width_touch_full": ("flag_width", "touch"),
    "flag_width_rejection1_full": ("flag_width", "rejection1"),
    "flag_width_rejection2of3_full": ("flag_width", "rejection2of3"),
    "half_pole_touch_full": ("half_pole", "touch"),
    "half_pole_rejection1_full": ("half_pole", "rejection1"),
    "half_pole_rejection2of3_full": ("half_pole", "rejection2of3"),
    "full_pole_touch_full": ("full_pole", "touch"),
    "full_pole_rejection1_full": ("full_pole", "rejection1"),
    "full_pole_rejection2of3_full": ("full_pole", "rejection2of3"),
}




def _num(frame: DataFrame, column: str, default: float | Series=0.0) -> Series:
    if column not in frame.columns:
        raise KeyError(f'missing required column: {column!r}')
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        raise KeyError(f'missing required column: {column!r}')
    return pd.Series(frame[column], index=frame.index).astype('boolean').fillna(False).astype(bool)


EXIT_FAMILY = 'pattern_target_zone'
PRIMARY_TRIGGER = 'bullish pat_flag_pattern_present at the locked score floor'
PRIMARY_GUARD = 'locked volume and bullish-pressure guards'
PRIMARY_TARGET = 'entry-frozen flag-width, half-pole, or full-pole objective'
PRIMARY_INVALIDATION = 'fixed 3% control stop'
TARGET_PROVIDER = PRIMARY_TARGET
INVALIDATION_PROVIDER = PRIMARY_INVALIDATION
EXIT_HYPOTHESIS = 'Exit at a flag-width, half-pole, or full-pole objective, either on touch or bearish rejection in the target zone.'
ACTIVE_SELL_PARAMS = ('exit_plan',)
SIEVE_STAGE = 'sieve3'
SIDE = 'long'
ENTRY_TAG = 'continuation_flag_present_long_1h'

class Sieve3V2PatternTargetZoneFromCompletePatternContinuationFlagPresentLong1H(IStrategy):
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

    exit_plan = CategoricalParameter(tuple(TARGET_ZONE_PLANS), default="half_pole_rejection1_full", space="sell", optimize=True, load=True)
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

    def _closed_frame(self, pair: str, current_time: datetime) -> DataFrame:
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
        close_times = dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        return frame.loc[close_times.le(self._utc(current_time))].sort_values("date").copy()

    def _freeze_entry_geometry(self, pair: str, trade: Any, cutoff_time: datetime) -> dict[str, float]:
        frame = self._closed_frame(pair, cutoff_time)
        if frame.empty:
            raise RuntimeError("No closed flag signal candle is available at trade entry.")
        row = frame.iloc[-1]
        upper = float(row["pat_flag_upper"])
        lower = float(row["pat_flag_lower"])
        pole_pct = float(row["pat_impulse_up_pct"])
        if not math.isfinite(upper) or not math.isfinite(lower) or not math.isfinite(pole_pct):
            raise RuntimeError("The flag signal candle has incomplete rail or pole geometry.")
        if lower <= 0.0 or upper <= lower or pole_pct <= 0.0:
            raise RuntimeError("The flag signal candle has invalid bullish geometry.")
        state = {
            "flag_width": upper + (upper - lower),
            "half_pole": upper * (1.0 + 0.5 * pole_pct),
            "full_pole": upper * (1.0 + pole_pct),
        }
        trade.set_custom_data(key=ENTRY_GEOMETRY_STATE_KEY, value=state)
        return state

    def _entry_geometry(self, pair: str, trade: Any) -> dict[str, float]:
        state = trade.get_custom_data(key=ENTRY_GEOMETRY_STATE_KEY)
        if state is None:
            cutoff = getattr(trade, "date_entry_fill_utc", None) or trade.open_date_utc
            return self._freeze_entry_geometry(pair, trade, cutoff)
        if not isinstance(state, dict):
            raise RuntimeError("The flag entry geometry snapshot is invalid.")
        return {name: float(state[name]) for name in ("flag_width", "half_pole", "full_pole")}

    def order_filled(self, pair: str, trade: Any, order: Any, current_time: datetime, **kwargs: Any) -> None:
        _ = kwargs
        if getattr(order, "ft_order_side", None) == getattr(trade, "entry_side", None) and trade.get_custom_data(key=ENTRY_GEOMETRY_STATE_KEY) is None:
            self._freeze_entry_geometry(pair, trade, current_time)

    def _post_entry_frame(self, pair: str, trade: Any, current_time: datetime) -> DataFrame:
        frame = self._closed_frame(pair, current_time)
        dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
        entered = self._utc(getattr(trade, "date_entry_fill_utc", None) or trade.open_date_utc)
        return frame.loc[dates.ge(entered)]

    @staticmethod
    def _target_confirmed(frame: DataFrame, target: float, confirmation: str) -> bool:
        if frame.empty:
            return False
        zone_floor = target * 0.9975
        touched = _num(frame, "high").ge(zone_floor)
        positions = np.flatnonzero(touched.to_numpy())
        if len(positions) == 0:
            return False
        if confirmation == "touch":
            return True
        post_touch = frame.iloc[int(positions[0]):]
        bearish = _num(post_touch, "close").lt(_num(post_touch, "open"))
        rejected = _num(post_touch, "close").lt(target)
        if confirmation == "rejection1":
            return bool(bearish.iloc[-1] and rejected.iloc[-1])
        return len(bearish) >= 3 and int((bearish & rejected).tail(3).sum()) >= 2

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        _ = current_rate, current_profit, kwargs
        target_name, confirmation = TARGET_ZONE_PLANS[str(self.exit_plan.value)]
        target = self._entry_geometry(pair, trade)[target_name]
        if target <= float(trade.open_rate):
            return None
        frame = self._post_entry_frame(pair, trade, current_time)
        return f"s3v2_{target_name}_{confirmation}" if self._target_confirmed(frame, target, confirmation) else None

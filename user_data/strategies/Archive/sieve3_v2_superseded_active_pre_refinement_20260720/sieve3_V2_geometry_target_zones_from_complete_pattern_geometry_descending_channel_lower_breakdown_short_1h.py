from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


from freqtrade.exchange import timeframe_to_minutes  # noqa: E402
from freqtrade.strategy import CategoricalParameter, IStrategy  # noqa: E402
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2  # noqa: E402

ENTRY_MODE = "entry_geometry_descending_channel_lower_breakdown_short_1h"
ENTRY_TAG = "geometry_descending_channel_lower_breakdown_short_1h"
SIDE = "short"
TIMEFRAME = "1h"




def _num(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)



def _bool(frame: DataFrame, column: str) -> Series:
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)



def _cross_below(value: Series, level: Series) -> Series:
    return value.lt(level) & value.shift(1).ge(level.shift(1))


SIEVE_STAGE = "sieve3"
ENTRY_SOURCE_STAGE = "sieve2"
RESEARCH_PATH = "sieve3_exit_geometry_target_zones"
SOURCE_RESULT_BATCH = "current_executable_defaults_historical_promoted_snapshots_conflict"
SOURCE_STRATEGY = "sieve3_exit_fixed_tp_sl_from_complete_pattern_geometry_descending_channel_lower_breakdown_short_1h.py:Sieve3ExitFixedTpSlFromCompletePatternGeometryDescendingChannelLowerBreakdownShort1H"
EXIT_HYPOTHESIS = "Use only source-supported descending-channel width projections below the broken lower rail."


class Sieve3V2GeometryTargetZonesFromCompletePatternGeometryDescendingChannelLowerBreakdownShort1H(IStrategy):
    """Full exit at an entry-frozen lower-rail extension measured in channel widths."""

    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 180
    process_only_new_candles = True
    can_short = True
    max_entry_position_adjustment = 0
    SIEVE_STAGE = "sieve3"
    SOURCE_STRATEGY = "sieve3_exit_fixed_tp_sl_from_complete_pattern_geometry_descending_channel_lower_breakdown_short_1h.py:Sieve3ExitFixedTpSlFromCompletePatternGeometryDescendingChannelLowerBreakdownShort1H"
    SOURCE_RESULT_BATCH = "current_executable_defaults_historical_promoted_snapshots_conflict"
    RESEARCH_PATH = "sieve3_exit_geometry_target_zones"
    ENTRY_SOURCE_STAGE = "sieve2"
    EXIT_HYPOTHESIS = "Use only source-supported descending-channel width projections below the broken lower rail."
    TARGET_PROVIDER = "entry-frozen pg2 descending-channel lower rail minus channel-width extension"
    INVALIDATION_PROVIDER = "fixed 4% hard risk cap"
    ACTIVE_SELL_PARAMS = ("target_zone_plan",)

    # Legacy inactive entry parameters: the selected entry never calls _common_guards,
    # so its volume, pressure, accumulation, body-direction, and close-direction controls are unreachable.
    # Values were use_volume_guard=True, volume_guard_window=48, volume_ratio_min=1.0,
    # use_pressure_guard=True, pressure_window=24, pressure_min=0.2,
    # use_accumulation_guard=False, use_body_direction_guard=False, use_close_direction_guard=False.
    min_pattern_bars = 12
    max_pattern_bars = 72
    compression_max_width_atr = 2.0
    squeeze_active_width_atr = 2.0
    local_narrowing_min_ratio = 0.1
    min_line_score = 0.5
    min_containment = 0.7
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

    target_zone_plan = CategoricalParameter(["lower_extension_half_width", "measured_move_one_width", "lower_extension_one_and_half_widths"], default="measured_move_one_width", space="sell", optimize=True, load=True)
    target_zone_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    _MULTIPLIERS = {"lower_extension_half_width": 0.5, "measured_move_one_width": 1.0, "lower_extension_one_and_half_widths": 1.5}
    _STATE_KEY = "sieve3_v2_desc_channel_geometry_target"
    minimal_roi = {"0": 100.0}
    stoploss = -0.04
    use_exit_signal = True
    position_adjustment_enable = False
    trailing_stop = False

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]: return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe = add_pattern_geometry_v2(dataframe, timeframe=self.timeframe, output_slots=1, include_triangle_patterns=False, include_wedge_patterns=False, include_compression_patterns=False, include_rectangle_patterns=False, include_ascending_channel_patterns=False, include_descending_channel_patterns=True, min_pattern_bars=int(self.min_pattern_bars), max_pattern_bars=int(self.max_pattern_bars), compression_max_width_atr=float(self.compression_max_width_atr), squeeze_active_width_atr=float(self.squeeze_active_width_atr), local_narrowing_min_ratio=float(self.local_narrowing_min_ratio), min_line_score=float(self.min_line_score), min_containment=float(self.min_containment), max_recent_touch_age_bars=int(self.max_recent_touch_age_bars), channel_min_pattern_bars=int(self.channel_min_pattern_bars), channel_max_pattern_bars=int(self.channel_max_pattern_bars), channel_min_quality=float(self.channel_min_quality), channel_min_containment=float(self.channel_min_containment), channel_near_boundary_atr_mult=float(self.channel_near_boundary_atr_mult), channel_breakout_atr_mult=float(self.channel_breakout_atr_mult), channel_lifecycle_confirm_break_bars=int(self.channel_lifecycle_confirm_break_bars), output_prefix="pg2")
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata; dataframe["enter_long"] = 0; dataframe["enter_short"] = 0; dataframe["enter_tag"] = None
        lower = _num(dataframe, "pg2_descending_channel_lower").mul(1.0 - float(self.rail_buffer_pct))
        condition = _bool(dataframe, "pg2_descending_channel_pattern_present") & _num(dataframe, "pg2_descending_channel_indicator_score").ge(float(self.score_min)) & _num(dataframe, "pg2_descending_channel_width_atr").le(float(self.width_atr_max)) & _cross_below(_num(dataframe, "close"), lower)
        valid = condition.fillna(False) & dataframe["volume"].gt(0.0) & dataframe["close"].notna(); dataframe.loc[valid, "enter_short"] = 1; dataframe.loc[valid, "enter_tag"] = ENTRY_TAG
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata; dataframe["exit_long"] = 0; dataframe["exit_short"] = 0; dataframe["exit_tag"] = None; return dataframe

    def _snapshot(self, pair: str, trade: Any, current_time: datetime) -> dict[str, float]:
        state = trade.get_custom_data(key=self._STATE_KEY)
        if state is not None: return dict(state)
        if self.dp is None: raise RuntimeError("data provider required for geometry target snapshot")
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        required = {"date", "pg2_descending_channel_upper", "pg2_descending_channel_lower"}; missing = sorted(required - set(frame.columns))
        if missing: raise KeyError(f"missing exact geometry target columns: {missing}")
        dates = pd.to_datetime(frame["date"], utc=True, errors="raise") + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        freeze_time = pd.Timestamp(getattr(trade, "open_date_utc", None) or current_time)
        if freeze_time.tzinfo is None: freeze_time = freeze_time.tz_localize("UTC")
        closed = frame.loc[dates.le(freeze_time)]
        if closed.empty: raise RuntimeError("no closed entry candle for geometry target")
        row = closed.iloc[-1]; upper = float(row["pg2_descending_channel_upper"]); lower = float(row["pg2_descending_channel_lower"]); width = upper - lower
        if not np.isfinite([upper, lower, width]).all() or width <= 0.0: raise ValueError("invalid entry-frozen descending-channel geometry")
        state = {"upper": upper, "lower": lower, "width": width}; trade.set_custom_data(key=self._STATE_KEY, value=state); return state

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        _ = current_profit, kwargs
        state = self._snapshot(pair, trade, current_time); multiplier = self._MULTIPLIERS[str(self.target_zone_plan.value)]; target = state["lower"] - state["width"] * multiplier
        if target >= float(trade.open_rate) * 0.999: return None
        return f"geometry_target_{self.target_zone_plan.value}" if current_rate <= target else None

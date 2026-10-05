"""Exit when the long breakout loses its frozen entry-candle upper rail."""
from __future__ import annotations

from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2

ENTRY_TAG = "geometry_triangle_squeeze_breakout_long_1h"
def _num(frame: DataFrame, column: str, _default: float | Series=...) -> Series:
    if column in frame.columns:
        value = frame[column]
    elif _default is not ...:
        value = pd.Series(_default, index=frame.index)
    else:
        raise KeyError(column)
    return pd.to_numeric(value, errors='coerce').replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    return frame[column].astype("boolean").fillna(False).astype(bool)


SIEVE_STAGE = 'sieve3'
ENTRY_SOURCE_STAGE = 'sieve2'
RESEARCH_PATH = 'sieve3_exit_breakout_upper_boundary_loss_invalidation'
EXIT_HYPOTHESIS = 'Exit when price loses the triangle breakout boundary, comparing confirmation plans.'
ACTIVE_SELL_PARAMS = ('failure_plan',)
class Sieve3V2BreakoutUpperBoundaryLossInvalidationFromCompletePatternGeometryTriangleSqueezeBreakoutLong1H(IStrategy):
    SIEVE_STAGE = 'sieve3'
    ENTRY_SOURCE_STAGE = 'sieve2'
    RESEARCH_PATH = 'sieve3_exit_breakout_upper_boundary_loss_invalidation'
    EXIT_HYPOTHESIS = 'Exit when price loses the triangle breakout boundary, comparing confirmation plans.'
    INTERFACE_VERSION = 3
    timeframe = "1h"; startup_candle_count = 180; process_only_new_candles = True; can_short = False
    position_adjustment_enable = False; max_entry_position_adjustment = 0; use_custom_stoploss = True; use_exit_signal = True
    exit_profit_only = False; ignore_roi_if_entry_signal = False; trailing_stop = False; minimal_roi = {"0": 100.0}; stoploss = -0.99

    use_volume_guard = True
    volume_guard_window = 48
    volume_ratio_min = 1.6
    use_pressure_guard = True
    pressure_window = 12
    pressure_min = 0.35
    # Legacy fixed-off entry parameter: use_accumulation_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_body_direction_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_close_direction_guard=False (BooleanParameter, buy space).
    min_pattern_bars = 12
    max_pattern_bars = 72
    compression_max_width_atr = 2.0
    squeeze_active_width_atr = 2.0
    min_line_score = 0.5
    min_containment = 0.88

    failure_plan = CategoricalParameter(
        [
            "close1_below_upper", "close2_below_upper", "close3_below_upper",
            "bear1_below_upper", "bear2_below_upper", "close1_below_upper025",
            "close2_below_upper025", "close3_below_upper025", "bear1_below_upper025",
            "bear2_below_upper025", "close1_below_upper050", "close2_below_upper050",
            "close3_below_upper050", "bear1_below_upper050", "bear2_below_upper050",
        ], default="close2_below_upper", space="sell", optimize=True, load=True,
    )
    failure_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    LOCK_STATUS = "current executable defaults preserved; historical promoted snapshots conflict"
    ACTIVE_SELL_PARAMS = ('failure_plan',)
    _GEOMETRY_KEY = "triangle_upper_loss_entry_geometry"

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float: return 1.0
    def informative_pairs(self) -> list[tuple[str, str]]: return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = add_pattern_geometry_v2(dataframe, timeframe=self.timeframe, output_slots=1, include_triangle_patterns=True, include_wedge_patterns=False, include_compression_patterns=False, include_rectangle_patterns=False, include_ascending_channel_patterns=False, include_descending_channel_patterns=False, min_pattern_bars=int(self.min_pattern_bars), max_pattern_bars=int(self.max_pattern_bars), compression_max_width_atr=float(self.compression_max_width_atr), squeeze_active_width_atr=float(self.squeeze_active_width_atr), min_line_score=float(self.min_line_score), min_containment=float(self.min_containment), output_prefix="pg2")
        return dataframe


    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        missing = sorted(set(('pg2_triangle_upper',)).difference(dataframe.columns))
        if missing:
            raise KeyError(f'{type(self).__name__} entry dataframe is missing required source columns: {missing}')
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = None
        condition = _bool(dataframe, 'pg2_triangle_pattern_present') & _bool(dataframe, 'pg2_triangle_squeeze_active') & _num(dataframe, 'pg2_triangle_indicator_score').ge(float(self.min_line_score)) & _num(dataframe, 'pg2_triangle_direction').ge(0) & _num(dataframe, 'close').gt(_num(dataframe, 'pg2_triangle_upper', np.nan))
        valid = condition.fillna(False) & dataframe['volume'].gt(0.0) & dataframe['close'].notna()
        dataframe.loc[valid, 'enter_long'] = 1
        dataframe.loc[valid, 'enter_tag'] = ENTRY_TAG
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0; dataframe["exit_short"] = 0; dataframe["exit_tag"] = None; return dataframe

    def _entry_geometry(self, pair: str, trade: Any) -> dict[str, float]:
        saved = trade.get_custom_data(key=self._GEOMETRY_KEY)
        if saved is not None: return saved
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe); dates = pd.to_datetime(frame["date"], utc=True); opened = pd.Timestamp(trade.open_date_utc); opened = opened.tz_localize("UTC") if opened.tzinfo is None else opened.tz_convert("UTC"); row = frame.loc[dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m") <= opened].sort_values("date").iloc[-1]
        upper = float(row["pg2_triangle_upper"]); lower = float(row["pg2_triangle_lower"])
        if not np.isfinite([upper, lower]).all() or not 0.0 < lower < upper or lower >= float(trade.open_rate): raise ValueError("entry triangle rails are unavailable or incoherent")
        saved = {"upper": upper, "lower": lower}; trade.set_custom_data(key=self._GEOMETRY_KEY, value=saved); return saved

    def _closed(self, pair: str, current_time: datetime) -> DataFrame:
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe); dates = pd.to_datetime(frame["date"], utc=True); cutoff = pd.Timestamp(current_time); cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        return frame.loc[dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m") <= cutoff].sort_values("date")

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        plan = str(self.failure_plan.value); count = int(plan[5]) if plan.startswith("close") else int(plan[4]); buffer = 0.005 if plan.endswith("050") else 0.0025 if plan.endswith("025") else 0.0; threshold = self._entry_geometry(pair, trade)["upper"] * (1.0 - buffer); frame = self._closed(pair, current_time)
        if len(frame) < count: return None
        failed = _num(frame, "close").tail(count).lt(threshold)
        if plan.startswith("bear"): failed &= _num(frame, "close").tail(count).lt(_num(frame, "open").tail(count))
        return "breakout_upper_boundary_lost" if bool(failed.all()) else None

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float:
        return stoploss_from_absolute(self._entry_geometry(pair, trade)["lower"], current_rate=current_rate, is_short=False, leverage=float(getattr(trade, "leverage", 1.0) or 1.0))

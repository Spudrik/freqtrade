"""Profit-armed bearish weakening/reversal exits for the triangle breakout."""
from __future__ import annotations

from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import BooleanParameter, CategoricalParameter, IntParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2




def _num(frame: DataFrame, column: str, default: float | Series=...) -> Series:
    if column not in frame.columns:
        if default is ...:
            raise KeyError(f'missing required column: {column!r}')
        return pd.to_numeric(
            pd.Series(default, index=frame.index), errors='coerce'
        ).replace([np.inf, -np.inf], np.nan)
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        raise KeyError(f"missing required column: {column!r}")
    return frame[column].astype("boolean").fillna(False).astype(bool)


EXIT_FAMILY = 'profit_armed_bearish_weakening_reversal'
PRIMARY_TRIGGER = '1h close above the triangle upper rail with squeeze, score, and direction confirmation'
PRIMARY_GUARD = 'triangle presence, squeeze, score floor, and non-negative direction'
PRIMARY_TARGET = 'selected open-profit gate'
PRIMARY_INVALIDATION = 'bearish lower-close evidence with optional score, squeeze, direction, and previous-low confirmation'
TARGET_PROVIDER = PRIMARY_TARGET
INVALIDATION_PROVIDER = PRIMARY_INVALIDATION
ENTRY_SOURCE_STAGE = 'sieve2'
SOURCE_STRATEGY = 'complete_pattern_geometry_triangle_squeeze_breakout_long_1h'
SOURCE_RESULT_BATCH = 'historical_promotion_lineage_unknown'
RESEARCH_PATH = 'sieve3_exit_profit_armed_bearish_weakening_reversal'
EXIT_HYPOTHESIS = 'After profit is armed, bearish closed-candle weakness should exit a triangle breakout when source strength also weakens where selected.'
ACTIVE_SELL_PARAMS = ('reversal_plan', 'require_below_previous_low', 'hard_stop_percent')
SIEVE_STAGE = 'sieve3'
SIDE = 'long'
ENTRY_TAG = 'geometry_triangle_squeeze_breakout_long_1h'

class Sieve3V2ProfitArmedBearishWeakeningReversalFromCompletePatternGeometryTriangleSqueezeBreakoutLong1H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    startup_candle_count = 180
    process_only_new_candles = True
    can_short = False
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    use_custom_stoploss = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    trailing_stop = False
    minimal_roi = {"0": 100.0}
    stoploss = -0.99

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

    reversal_plan = CategoricalParameter(
        [
            "p05_bear_candle", "p10_bear_candle", "p15_bear_candle", "p20_bear_candle",
            "p30_bear_candle", "p10_two_lower_closes", "p15_two_lower_closes",
            "p20_two_lower_closes", "p30_two_lower_closes", "p10_score_weak_bear",
            "p15_score_weak_bear", "p20_score_weak_bear", "p30_score_weak_bear",
            "p10_bear_direction", "p20_bear_direction", "p30_bear_direction",
        ],
        default="p15_score_weak_bear", space="sell", optimize=True, load=True,
    )
    reversal_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    require_below_previous_low = BooleanParameter(default=False, space="sell", optimize=True, load=True)
    require_below_previous_low.batch_tags = ('family:exits', 'mode:sieve3_exit')
    hard_stop_percent = IntParameter(2, 8, default=3, space="sell", optimize=True, load=True)
    hard_stop_percent.batch_tags = ('family:exits', 'mode:sieve3_exit')

    LOCK_STATUS = "current executable defaults preserved; historical promoted snapshots conflict"
    ACTIVE_SELL_PARAMS = ("reversal_plan", "require_below_previous_low", "hard_stop_percent")

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]: return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = add_pattern_geometry_v2(dataframe, timeframe=self.timeframe, output_slots=1, include_triangle_patterns=True, include_wedge_patterns=False, include_compression_patterns=False, include_rectangle_patterns=False, include_ascending_channel_patterns=False, include_descending_channel_patterns=False, min_pattern_bars=int(self.min_pattern_bars), max_pattern_bars=int(self.max_pattern_bars), compression_max_width_atr=float(self.compression_max_width_atr), squeeze_active_width_atr=float(self.squeeze_active_width_atr), min_line_score=float(self.min_line_score), min_containment=float(self.min_containment), output_prefix="pg2")
        return dataframe

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype="bool"); close = _num(dataframe, "close")
        if bool(self.use_volume_guard):
            volume = _num(dataframe, "volume").clip(lower=0.0); window = int(self.volume_guard_window)
            baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
            guard &= volume.ge(baseline.mul(float(self.volume_ratio_min)))
        if bool(self.use_pressure_guard):
            open_ = _num(dataframe, "open"); high = _num(dataframe, "high"); low = _num(dataframe, "low"); volume = _num(dataframe, "volume").clip(lower=0.0).fillna(0.0)
            candle_range = (high - low).replace(0.0, np.nan)
            pressure = ((((close - open_) / candle_range).clip(-1.0, 1.0).fillna(0.0) + (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0).fillna(0.0)) / 2.0).clip(-1.0, 1.0); directional_volume = (pressure * volume).fillna(0.0)
        if bool(self.use_pressure_guard):
            window = int(self.pressure_window); denominator = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
            guard &= directional_volume.rolling(window, min_periods=max(2, window // 3)).sum().div(denominator).ge(float(self.pressure_min))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        missing = sorted(set(("pg2_triangle_upper",)).difference(dataframe.columns))
        if missing:
            raise KeyError(f"{type(self).__name__} entry dataframe is missing required source columns: {missing}")
        dataframe["enter_long"] = 0; dataframe["enter_short"] = 0; dataframe["enter_tag"] = None
        condition = _bool(dataframe, "pg2_triangle_pattern_present") & _bool(dataframe, "pg2_triangle_squeeze_active") & _num(dataframe, "pg2_triangle_indicator_score").ge(float(self.min_line_score)) & _num(dataframe, "pg2_triangle_direction").ge(0) & _num(dataframe, "close").gt(_num(dataframe, "pg2_triangle_upper", np.nan))
        valid = condition.fillna(False) & dataframe["volume"].gt(0.0) & dataframe["close"].notna(); dataframe.loc[valid, "enter_long"] = 1; dataframe.loc[valid, "enter_tag"] = ENTRY_TAG
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0; dataframe["exit_short"] = 0; dataframe["exit_tag"] = None; return dataframe

    def _closed(self, pair: str, current_time: datetime, trade: Any) -> DataFrame:
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        dates = pd.to_datetime(frame["date"], utc=True)
        cutoff = pd.Timestamp(current_time)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        close_times = dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        entered = pd.Timestamp(getattr(trade, "date_entry_fill_utc", None) or trade.open_date_utc)
        entered = entered.tz_localize("UTC") if entered.tzinfo is None else entered.tz_convert("UTC")
        return frame.loc[dates.ge(entered) & close_times.le(cutoff)].sort_values("date").tail(3)

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        frame = self._closed(pair, current_time, trade)
        if len(frame) < 2: return None
        plan = str(self.reversal_plan.value); gate = float(plan[1:3]) / 1000.0
        if current_profit < gate: return None
        close = _num(frame, "close"); open_ = _num(frame, "open")
        bearish = bool(close.iloc[-1] < open_.iloc[-1] and close.iloc[-1] < close.iloc[-2])
        if "two_lower_closes" in plan: bearish = bearish and len(frame) >= 3 and bool(close.iloc[-2] < close.iloc[-3])
        elif "score_weak_bear" in plan:
            score = _num(frame, "pg2_triangle_indicator_score")
            bearish = bearish and bool(score.iloc[-1] < score.iloc[-2] or not _bool(frame, "pg2_triangle_squeeze_active").iloc[-1])
        elif "bear_direction" in plan:
            bearish = bearish and bool(_num(frame, "pg2_triangle_direction").iloc[-1] < 0)
        if bool(self.require_below_previous_low.value): bearish = bearish and bool(close.iloc[-1] < _num(frame, "low").iloc[-2])
        return "profit_armed_bearish_reversal" if bearish else None

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float:
        hard_stop = int(self.hard_stop_percent.value) * 0.01
        stop_price = float(trade.open_rate) * (1.0 - hard_stop)
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=False, leverage=float(getattr(trade, "leverage", 1.0) or 1.0))

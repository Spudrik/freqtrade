from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import BooleanParameter, CategoricalParameter, IntParameter, IStrategy
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2  # noqa: E402
  # noqa: E402

ENTRY_MODE = "entry_geometry_descending_channel_lower_breakdown_short_1h"


TIMEFRAME = "1h"




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
        raise KeyError(f'missing required column: {column!r}')
    return pd.Series(frame[column], index=frame.index).astype('boolean').fillna(False).astype(bool)


def _cross_below(value: Series, level: Series) -> Series:
    return value.lt(level) & value.shift(1).ge(level.shift(1))


EXIT_FAMILY = 'profit_armed_bullish_weakening_reversal'
PRIMARY_TRIGGER = '1h close cross below the descending-channel lower rail with locked score and width qualification'
PRIMARY_GUARD = 'locked volume and bearish-pressure guards'
PRIMARY_TARGET = 'selected open-profit gate'
PRIMARY_INVALIDATION = 'bullish OHLC reversal confirmed by descending-channel score decay or direction loss'
TARGET_PROVIDER = PRIMARY_TARGET
INVALIDATION_PROVIDER = PRIMARY_INVALIDATION
ENTRY_SOURCE_STAGE = 'sieve2'
SOURCE_STRATEGY = 'sieve3_exit_fixed_tp_sl_from_complete_pattern_geometry_descending_channel_lower_breakdown_short_1h.py:Sieve3ExitFixedTpSlFromCompletePatternGeometryDescendingChannelLowerBreakdownShort1H'
SOURCE_RESULT_BATCH = 'current_executable_defaults_historical_promoted_snapshots_conflict'
RESEARCH_PATH = 'sieve3_exit_profit_armed_bullish_weakening_reversal'
EXIT_HYPOTHESIS = 'Once profit is available, close on source-score decay plus a bullish reversal candle.'
ACTIVE_SELL_PARAMS = ('profit_gate_step', 'weakening_plan', 'require_previous_high_break')
SIEVE_STAGE = 'sieve3'
SIDE = 'short'
ENTRY_TAG = 'geometry_descending_channel_lower_breakdown_short_1h'

class Sieve3V2ProfitArmedBullishWeakeningReversalFromCompletePatternGeometryDescendingChannelLowerBreakdownShort1H(IStrategy):
    """Exit a profitable short only when bearish geometry weakens with bullish reversal evidence."""

    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 180
    process_only_new_candles = True
    can_short = True
    max_entry_position_adjustment = 0
    SIEVE_STAGE = "sieve3"
    SOURCE_STRATEGY = "sieve3_exit_fixed_tp_sl_from_complete_pattern_geometry_descending_channel_lower_breakdown_short_1h.py:Sieve3ExitFixedTpSlFromCompletePatternGeometryDescendingChannelLowerBreakdownShort1H"
    SOURCE_RESULT_BATCH = "current_executable_defaults_historical_promoted_snapshots_conflict"
    RESEARCH_PATH = 'sieve3_exit_profit_armed_bullish_weakening_reversal'
    ENTRY_SOURCE_STAGE = 'sieve2'
    EXIT_HYPOTHESIS = "Once profit is available, close on source-score decay plus a bullish reversal candle."
    TARGET_PROVIDER = "profit gate"
    INVALIDATION_PROVIDER = "bullish OHLC reversal confirmed by descending-channel score decay"
    ACTIVE_SELL_PARAMS = ("profit_gate_step", "weakening_plan", "require_previous_high_break")

    
    
    
    
    
    
    
    
    
    
    
    
    
    use_volume_guard = True
    volume_guard_window = 48
    volume_ratio_min = 1.0
    use_pressure_guard = True
    pressure_window = 24
    pressure_min = 0.2
    # Legacy fixed-off entry parameter: use_accumulation_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_body_direction_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_close_direction_guard=False (BooleanParameter, buy space).
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

    profit_gate_step = IntParameter(1, 4, default=2, space="sell", optimize=True, load=True)
    profit_gate_step.batch_tags = ('family:exits', 'mode:sieve3_exit')
    weakening_plan = CategoricalParameter(["score_drop_50pct", "score_drop_25pct", "score_below_locked_min", "direction_no_longer_bearish"], default="score_drop_25pct", space="sell", optimize=True, load=True)
    weakening_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    require_previous_high_break = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    require_previous_high_break.batch_tags = ('family:exits', 'mode:sieve3_exit')

    minimal_roi = {"0": 100.0}
    stoploss = -0.04
    use_exit_signal = True
    position_adjustment_enable = False
    trailing_stop = False

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe = add_pattern_geometry_v2(dataframe, timeframe=self.timeframe, output_slots=1, include_triangle_patterns=False, include_wedge_patterns=False, include_compression_patterns=False, include_rectangle_patterns=False, include_ascending_channel_patterns=False, include_descending_channel_patterns=True, min_pattern_bars=int(self.min_pattern_bars), max_pattern_bars=int(self.max_pattern_bars), compression_max_width_atr=float(self.compression_max_width_atr), squeeze_active_width_atr=float(self.squeeze_active_width_atr), local_narrowing_min_ratio=float(self.local_narrowing_min_ratio), min_line_score=float(self.min_line_score), min_containment=float(self.min_containment), max_recent_touch_age_bars=int(self.max_recent_touch_age_bars), channel_min_pattern_bars=int(self.channel_min_pattern_bars), channel_max_pattern_bars=int(self.channel_max_pattern_bars), channel_min_quality=float(self.channel_min_quality), channel_min_containment=float(self.channel_min_containment), channel_near_boundary_atr_mult=float(self.channel_near_boundary_atr_mult), channel_breakout_atr_mult=float(self.channel_breakout_atr_mult), channel_lifecycle_confirm_break_bars=int(self.channel_lifecycle_confirm_break_bars), output_prefix="pg2")
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
            open_ = _num(dataframe, "open"); high = _num(dataframe, "high"); low = _num(dataframe, "low")
            volume = _num(dataframe, "volume").clip(lower=0.0).fillna(0.0)
            candle_range = (high - low).replace(0.0, np.nan)
            pressure = ((((close - open_) / candle_range).clip(-1.0, 1.0).fillna(0.0) + (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0).fillna(0.0)) / 2.0).clip(-1.0, 1.0)
            directional_volume = (pressure * volume).fillna(0.0)
        if bool(self.use_pressure_guard):
            window = int(self.pressure_window)
            denominator = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
            guard &= (directional_volume.rolling(window, min_periods=max(2, window // 3)).sum() / denominator).le(-float(self.pressure_min))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        missing = sorted(set(("pg2_descending_channel_lower", "pg2_descending_channel_width_atr")).difference(dataframe.columns))
        if missing:
            raise KeyError(f"{type(self).__name__} entry dataframe is missing required source columns: {missing}")
        dataframe["enter_long"] = 0; dataframe["enter_short"] = 0; dataframe["enter_tag"] = None
        lower = _num(dataframe, "pg2_descending_channel_lower", np.nan).mul(1.0 - float(self.rail_buffer_pct))
        condition = _bool(dataframe, "pg2_descending_channel_pattern_present") & _num(dataframe, "pg2_descending_channel_indicator_score").ge(float(self.score_min)) & _num(dataframe, "pg2_descending_channel_width_atr", np.nan).le(float(self.width_atr_max)) & _cross_below(_num(dataframe, "close"), lower)
        condition &= self._common_guards(dataframe)
        
        valid = condition.fillna(False) & dataframe["volume"].gt(0.0) & dataframe["close"].notna()
        dataframe.loc[valid, "enter_short"] = 1; dataframe.loc[valid, "enter_tag"] = ENTRY_TAG
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0; dataframe["exit_short"] = 0; dataframe["exit_tag"] = None
        return dataframe

    def _latest_two(self, pair: str, trade: Any, current_time: datetime) -> DataFrame:
        if self.dp is None:
            raise RuntimeError("data provider required for bullish weakening exit")
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        required = {"date", "open", "high", "close", "pg2_descending_channel_indicator_score", "pg2_descending_channel_direction"}
        missing = sorted(required - set(frame.columns))
        if missing: raise KeyError(f"missing exact bullish-weakening columns: {missing}")
        dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
        close_times = dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        cutoff = pd.Timestamp(current_time)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        entered = pd.Timestamp(getattr(trade, "date_entry_fill_utc", None) or trade.open_date_utc)
        entered = entered.tz_localize("UTC") if entered.tzinfo is None else entered.tz_convert("UTC")
        return frame.loc[dates.ge(entered) & close_times.le(cutoff)].sort_values("date").tail(2)

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        _ = current_rate, kwargs
        profit_gate = int(self.profit_gate_step.value) * 0.01
        if current_profit < profit_gate: return None
        rows = self._latest_two(pair, trade, current_time)
        if len(rows) < 2: return None
        previous, last = rows.iloc[0], rows.iloc[1]
        bullish = float(last["close"]) > float(last["open"])
        if bool(self.require_previous_high_break.value): bullish &= float(last["close"]) > float(previous["high"])
        if not bullish: return None
        current_score = float(last["pg2_descending_channel_indicator_score"])
        previous_score = float(previous["pg2_descending_channel_indicator_score"])
        plan = str(self.weakening_plan.value)
        weakened = (current_score <= previous_score * 0.5 if plan == "score_drop_50pct" else current_score <= previous_score * 0.75 if plan == "score_drop_25pct" else current_score < float(self.score_min) if plan == "score_below_locked_min" else float(last["pg2_descending_channel_direction"]) >= 0.0)
        return f"profit_armed_bullish_weakening_{plan}" if weakened else None

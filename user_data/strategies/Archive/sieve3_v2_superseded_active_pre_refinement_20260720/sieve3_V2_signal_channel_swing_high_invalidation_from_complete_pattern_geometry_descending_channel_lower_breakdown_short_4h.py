from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2


ENTRY_MODE = "entry_geometry_descending_channel_lower_breakdown_short_4h"
ENTRY_TAG, SIDE, TIMEFRAME = "geometry_descending_channel_lower_breakdown_short_4h", "short", "4h"

SOURCE_STRATEGY = "user_data/strategies/sieve3_exit_fixed_tp_sl_from_complete_pattern_geometry_descending_channel_lower_breakdown_short_4h.py:Sieve3ExitFixedTpSlFromCompletePatternGeometryDescendingChannelLowerBreakdownShort4H"
SOURCE_RESULT_BATCH = "sieve3_exit_branch_generation_20260612"
RESEARCH_PATH = 'sieve3_exit_signal_channel_swing_high_invalidation'
ENTRY_SOURCE_STAGE = 'sieve2'

ENTRY_LOCK_STATUS = "unanimous_current_executable_defaults_locked; historical promoted snapshots conflict"
ENTRY_LOCK_SHA256 = "8b900a491a755ddc11adb21c8bd307b92af2b3848ae48a9e06766233dc8673b9"
ENTRY_GEOMETRY_STATE_KEY = "s3v2_descending_channel_short_swing_highs"


def _num(frame: DataFrame, column: str, default: float=...) -> Series:
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
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)


def _cross_below(value: Series, level: Series) -> Series:
    return value.lt(level) & value.shift(1).ge(level.shift(1))


EXIT_FAMILY = 'signal_channel_swing_high_invalidation'
PRIMARY_TRIGGER = '4h close cross below the descending-channel lower rail with locked score and width qualification'
PRIMARY_GUARD = 'enabled volume/pressure guards'
PRIMARY_TARGET = 'none'
PRIMARY_INVALIDATION = 'entry signal high, channel upper rail, or causal pre-signal swing high; fixed 8% adverse stop'
TARGET_PROVIDER = PRIMARY_TARGET
INVALIDATION_PROVIDER = PRIMARY_INVALIDATION
EXIT_HYPOTHESIS = 'Invalidate on 4h close failure above the signal high, entry channel upper rail, or causal pre-signal swing high.'
ACTIVE_SELL_PARAMS = ('failure_plan',)
SIEVE_STAGE = 'sieve3'
SIDE = 'short'
ENTRY_TAG = 'geometry_descending_channel_lower_breakdown_short_4h'

class Sieve3V2SignalChannelSwingHighInvalidationFromCompletePatternGeometryDescendingChannelLowerBreakdownShort4H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe, startup_candle_count, process_only_new_candles, can_short = TIMEFRAME, 180, True, True
    minimal_roi, stoploss = {"0": 100.0}, -0.12
    use_exit_signal, use_custom_stoploss, position_adjustment_enable = True, True, False
    trailing_stop, ignore_roi_if_entry_signal = False, False

    
    
    
    
    
    
    
    
    
    
    
    
    
    use_volume_guard = True
    volume_guard_window = 48
    volume_ratio_min = 0.8
    use_pressure_guard = True
    pressure_window = 24
    pressure_min = 0.15
    # Legacy fixed-off entry parameter: use_accumulation_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_body_direction_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_close_direction_guard=False (BooleanParameter, buy space).
    min_pattern_bars = 12
    max_pattern_bars = 72
    compression_max_width_atr = 2.0
    squeeze_active_width_atr = 2.0
    local_narrowing_min_ratio = 0.1
    min_line_score = 0.5
    min_containment = 0.88
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

    failure_plan = CategoricalParameter(("signal_high_1", "signal_high_2", "signal_high_2_buffer_25", "channel_upper_1", "channel_upper_2", "channel_upper_2_buffer_25", "prior_18_1", "prior_18_2", "prior_36_1", "prior_36_2", "prior_72_1", "prior_72_2"), default="channel_upper_2", space="sell", optimize=True, load=True)
    failure_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    ACTIVE_SELL_PARAMS = ('failure_plan',)
    FAILURE_PLANS = {
        "signal_high_1": ("signal_high", 1, 0.0), "signal_high_2": ("signal_high", 2, 0.0), "signal_high_2_buffer_25": ("signal_high", 2, 0.0025),
        "channel_upper_1": ("channel_upper", 1, 0.0), "channel_upper_2": ("channel_upper", 2, 0.0), "channel_upper_2_buffer_25": ("channel_upper", 2, 0.0025),
        "prior_18_1": ("prior_18", 1, 0.0), "prior_18_2": ("prior_18", 2, 0.0), "prior_36_1": ("prior_36", 1, 0.0),
        "prior_36_2": ("prior_36", 2, 0.0), "prior_72_1": ("prior_72", 1, 0.0), "prior_72_2": ("prior_72", 2, 0.0),
    }

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = add_pattern_geometry_v2(dataframe, timeframe=self.timeframe, output_slots=1, include_triangle_patterns=False, include_wedge_patterns=False, include_compression_patterns=False, include_rectangle_patterns=False, include_ascending_channel_patterns=False, include_descending_channel_patterns=True, min_pattern_bars=int(self.min_pattern_bars), max_pattern_bars=int(self.max_pattern_bars), compression_max_width_atr=float(self.compression_max_width_atr), squeeze_active_width_atr=float(self.squeeze_active_width_atr), local_narrowing_min_ratio=float(self.local_narrowing_min_ratio), min_line_score=float(self.min_line_score), min_containment=float(self.min_containment), max_recent_touch_age_bars=int(self.max_recent_touch_age_bars), channel_min_pattern_bars=int(self.channel_min_pattern_bars), channel_max_pattern_bars=int(self.channel_max_pattern_bars), channel_min_quality=float(self.channel_min_quality), channel_min_containment=float(self.channel_min_containment), channel_near_boundary_atr_mult=float(self.channel_near_boundary_atr_mult), channel_breakout_atr_mult=float(self.channel_breakout_atr_mult), channel_lifecycle_confirm_break_bars=int(self.channel_lifecycle_confirm_break_bars), output_prefix="pg2")
        return dataframe

    def _common_guards(self, dataframe: DataFrame) -> Series:
        close, volume = _num(dataframe, "close"), _num(dataframe, "volume").clip(lower=0.0)
        window = int(self.volume_guard_window)
        guard = volume.ge(volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan).mul(float(self.volume_ratio_min)))
        open_, high, low = (_num(dataframe, name) for name in ("open", "high", "low"))
        candle_range = (high - low).replace(0.0, np.nan)
        pressure = ((((close - open_) / candle_range).clip(-1.0, 1.0).fillna(0.0) + ((((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)).fillna(0.0)) / 2.0).clip(-1.0, 1.0)
        window = int(self.pressure_window)
        baseline = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
        guard &= (pressure * volume.fillna(0.0)).rolling(window, min_periods=max(2, window // 3)).sum().div(baseline).le(-float(self.pressure_min))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        missing = sorted(set(("pg2_descending_channel_lower", "pg2_descending_channel_width_atr")).difference(dataframe.columns))
        if missing:
            raise KeyError(f"{type(self).__name__} entry dataframe is missing required source columns: {missing}")
        dataframe["enter_long"], dataframe["enter_short"], dataframe["enter_tag"] = 0, 0, None
        lower = _num(dataframe, "pg2_descending_channel_lower", np.nan).mul(1.0 - float(self.rail_buffer_pct))
        condition = _bool(dataframe, "pg2_descending_channel_pattern_present") & _num(dataframe, "pg2_descending_channel_indicator_score").ge(float(self.score_min)) & _num(dataframe, "pg2_descending_channel_width_atr", np.nan).le(float(self.width_atr_max)) & _cross_below(_num(dataframe, "close"), lower)
        condition &= self._common_guards(dataframe)
        
        valid = condition.fillna(False) & dataframe["volume"].gt(0.0) & dataframe["close"].notna()
        dataframe.loc[valid, "enter_short"] = 1
        dataframe.loc[valid, "enter_tag"] = ENTRY_TAG
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"], dataframe["exit_short"], dataframe["exit_tag"] = 0, 0, None
        return dataframe

    def _freeze_entry_levels(self, pair: str, trade: Any, cutoff_time: datetime) -> dict[str, float]:
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if frame is None or frame.empty:
            raise RuntimeError("swing-high invalidation requires an analyzed entry dataframe")
        missing = sorted({"date", "high", "pg2_descending_channel_upper"} - set(frame.columns))
        if missing:
            raise KeyError(f"swing-high entry snapshot is missing columns: {missing}")
        result = frame.copy()
        result["date"] = pd.to_datetime(result["date"], utc=True, errors="raise")
        cutoff = pd.Timestamp(cutoff_time)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        close_times = result["date"] + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        signal = result.loc[close_times.le(cutoff)]
        if signal.empty:
            raise RuntimeError("entry signal candle is unavailable for swing-high invalidation")
        row, history = signal.iloc[-1], signal.iloc[:-1]
        levels = {"signal_high": float(row["high"]), "channel_upper": float(row["pg2_descending_channel_upper"])}
        for lookback in (18, 36, 72):
            levels[f"prior_{lookback}"] = float(_num(history.tail(lookback), "high").max())
        invalid = sorted(name for name, value in levels.items() if not np.isfinite(value) or value <= 0.0)
        if invalid:
            raise RuntimeError(f"entry signal has unusable swing-high levels: {invalid}")
        trade.set_custom_data(key=ENTRY_GEOMETRY_STATE_KEY, value=levels)
        return levels

    def _entry_levels(self, pair: str, trade: Any) -> dict[str, float]:
        state = trade.get_custom_data(key=ENTRY_GEOMETRY_STATE_KEY)
        if state is None:
            cutoff = getattr(trade, "date_entry_fill_utc", None) or trade.open_date_utc
            return self._freeze_entry_levels(pair, trade, cutoff)
        if not isinstance(state, dict):
            raise RuntimeError("swing-high entry geometry snapshot is invalid")
        return {name: float(value) for name, value in state.items()}

    def order_filled(self, pair: str, trade: Any, order: Any, current_time: datetime, **kwargs: Any) -> None:
        _ = kwargs
        if getattr(order, "ft_order_side", None) == getattr(trade, "entry_side", None) and trade.get_custom_data(key=ENTRY_GEOMETRY_STATE_KEY) is None:
            self._freeze_entry_levels(pair, trade, current_time)

    def _context(self, pair: str, trade: Any, current_time: datetime) -> tuple[DataFrame, dict[str, float]]:
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        result = frame.copy()
        result["date"] = pd.to_datetime(result["date"], utc=True, errors="raise")
        entered = pd.Timestamp(getattr(trade, "date_entry_fill_utc", None) or trade.open_date_utc)
        entered = entered.tz_localize("UTC") if entered.tzinfo is None else entered.tz_convert("UTC")
        cutoff = pd.Timestamp(current_time)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        close_times = result["date"] + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        post_entry = result.loc[result["date"].ge(entered) & close_times.le(cutoff)].sort_values("date")
        levels = self._entry_levels(pair, trade)
        return post_entry, levels

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | bool | None:
        source, confirmations, buffer = self.FAILURE_PLANS[str(self.failure_plan.value)]
        post_entry, levels = self._context(pair, trade, current_time)
        level = levels[source]
        if not np.isfinite(level) or level <= float(trade.open_rate):
            return None
        closes = _num(post_entry, "close").tail(confirmations)
        return f"failed_above_{source}_{confirmations}x4h" if len(closes) == confirmations and bool(closes.gt(level * (1.0 + buffer)).all()) else None

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float | None:
        return stoploss_from_absolute(float(trade.open_rate) * 1.08, current_rate=current_rate, is_short=True, leverage=float(getattr(trade, "leverage", 1.0) or 1.0))


__all__ = ["Sieve3V2SignalChannelSwingHighInvalidationFromCompletePatternGeometryDescendingChannelLowerBreakdownShort4H"]

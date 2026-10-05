from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.complex_volume_profile import add_volume_profile
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2


ENTRY_MODE = "entry_geometry_descending_channel_lower_breakdown_short_4h"
ENTRY_TAG = "geometry_descending_channel_lower_breakdown_short_4h"
SIDE, TIMEFRAME = "short", "4h"
SIEVE_STAGE = "sieve3"
SOURCE_STRATEGY = "user_data/strategies/sieve3_exit_fixed_tp_sl_from_complete_pattern_geometry_descending_channel_lower_breakdown_short_4h.py:Sieve3ExitFixedTpSlFromCompletePatternGeometryDescendingChannelLowerBreakdownShort4H"
SOURCE_RESULT_BATCH = "sieve3_exit_branch_generation_20260612"
RESEARCH_PATH = "sieve3_exit_measured_channel_target_zones"
ENTRY_SOURCE_STAGE = "sieve2"
EXIT_HYPOTHESIS = "Exit in downside zones projected from the entry-frozen descending-channel width."
ENTRY_LOCK_STATUS = "unanimous_current_executable_defaults_locked; historical promoted snapshots conflict"
ENTRY_LOCK_SHA256 = "8b900a491a755ddc11adb21c8bd307b92af2b3848ae48a9e06766233dc8673b9"
VP_MODES = ("score_or_context", "node_confirm", "value_area_confirm", "breakout_acceptance", "rejection_confirm", "poc_hvn_reject", "prior_level_confirm")
MARKET_MODES = ("pressure_or_trend", "pressure_and_trend", "directional_pressure", "trend_state", "avoid_adverse_pressure", "avoid_chop")


def _tag(parameter: Any) -> Any:
    setattr(parameter, "batch_tags", ("family:entries", f"mode:{ENTRY_MODE}"))
    return parameter


def _num(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)



def _bool(frame: DataFrame, column: str) -> Series:
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)



def _cross_below(value: Series, level: Series) -> Series:
    return value.lt(level) & value.shift(1).ge(level.shift(1))


ACTIVE_SELL_PARAMS = ("target_zone_plan",)



class Sieve3V2MeasuredChannelTargetZonesFromCompletePatternGeometryDescendingChannelLowerBreakdownShort4H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe, startup_candle_count, process_only_new_candles, can_short = TIMEFRAME, 180, True, True
    ACTIVE_SELL_PARAMS = ("target_zone_plan",)
    minimal_roi, stoploss = {"0": 100.0}, -0.12
    use_exit_signal, use_custom_stoploss, position_adjustment_enable = True, True, False
    trailing_stop, ignore_roi_if_entry_signal = False, False

    # Legacy fixed-off entry parameter: use_sieve2_vp_guard=False (CategoricalParameter, buy space).
    # Legacy inert entry parameter: sieve2_vp_guard_mode='score_or_context' (fixed entry mode/gate).
    sieve2_vp_window = 96
    sieve2_vp_bins = 36
    # Legacy inert entry parameter: sieve2_vp_score_min=0.25 (fixed entry mode/gate).
    # Legacy inert entry parameter: sieve2_vp_context_min=0.28 (fixed entry mode/gate).
    # Legacy fixed-off entry parameter: use_sieve2_market_guard=False (CategoricalParameter, buy space).
    # Legacy inert entry parameter: sieve2_market_guard_mode='pressure_or_trend' (fixed entry mode/gate).
    sieve2_market_window = 24
    # Legacy inert entry parameter: sieve2_market_pressure_min=0.07 (fixed entry mode/gate).
    # Legacy inert entry parameter: sieve2_market_trend_min=0.25 (fixed entry mode/gate).
    # Legacy inert entry parameter: sieve2_rs_benchmark_pair='BTC/USDT:USDT' (fixed entry mode/gate).
    # Legacy inert entry parameter: sieve2_rs_score_min=0.45 (fixed entry mode/gate).
    # Legacy inert entry parameter: use_volume_guard=True (fixed entry mode/gate).
    volume_guard_window = 48
    volume_ratio_min = 0.8
    # Legacy inert entry parameter: use_pressure_guard=True (fixed entry mode/gate).
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

    target_zone_plan = CategoricalParameter(
        ("half_touch", "half_10pct_zone", "three_quarter_touch", "three_quarter_10pct_zone", "one_width_touch", "one_width_10pct_zone", "one_width_20pct_zone", "one_quarter_touch", "one_quarter_10pct_zone", "one_half_touch", "one_half_10pct_zone", "two_width_touch"),
        default="one_width_10pct_zone", space="sell", optimize=True, load=True,
    )
    target_zone_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    TARGET_ZONE_PLANS = {
        "half_touch": (0.50, 0.00, 0.04), "half_10pct_zone": (0.50, 0.10, 0.04),
        "three_quarter_touch": (0.75, 0.00, 0.04), "three_quarter_10pct_zone": (0.75, 0.10, 0.04),
        "one_width_touch": (1.00, 0.00, 0.05), "one_width_10pct_zone": (1.00, 0.10, 0.05),
        "one_width_20pct_zone": (1.00, 0.20, 0.05), "one_quarter_touch": (1.25, 0.00, 0.06),
        "one_quarter_10pct_zone": (1.25, 0.10, 0.06), "one_half_touch": (1.50, 0.00, 0.06),
        "one_half_10pct_zone": (1.50, 0.10, 0.06), "two_width_touch": (2.00, 0.00, 0.08),
    }

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe = add_pattern_geometry_v2(dataframe, timeframe=self.timeframe, output_slots=1, include_triangle_patterns=False, include_wedge_patterns=False, include_compression_patterns=False, include_rectangle_patterns=False, include_ascending_channel_patterns=False, include_descending_channel_patterns=True, min_pattern_bars=int(self.min_pattern_bars), max_pattern_bars=int(self.max_pattern_bars), compression_max_width_atr=float(self.compression_max_width_atr), squeeze_active_width_atr=float(self.squeeze_active_width_atr), local_narrowing_min_ratio=float(self.local_narrowing_min_ratio), min_line_score=float(self.min_line_score), min_containment=float(self.min_containment), max_recent_touch_age_bars=int(self.max_recent_touch_age_bars), channel_min_pattern_bars=int(self.channel_min_pattern_bars), channel_max_pattern_bars=int(self.channel_max_pattern_bars), channel_min_quality=float(self.channel_min_quality), channel_min_containment=float(self.channel_min_containment), channel_near_boundary_atr_mult=float(self.channel_near_boundary_atr_mult), channel_breakout_atr_mult=float(self.channel_breakout_atr_mult), channel_lifecycle_confirm_break_bars=int(self.channel_lifecycle_confirm_break_bars), output_prefix="pg2")
        return add_volume_profile(dataframe, window=int(self.sieve2_vp_window), bins=int(self.sieve2_vp_bins), value_area_pct=0.70, price_source="hlc3", smooth_bins=3, pressure_delta_min=0.05, node_near_pct=0.01, prefix="s2vp")

    def _common_guards(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, "close")
        volume = _num(dataframe, "volume").clip(lower=0.0)
        volume_window = int(self.volume_guard_window)
        guard = volume.ge(volume.shift(1).rolling(volume_window, min_periods=max(2, volume_window // 3)).mean().replace(0.0, np.nan).mul(float(self.volume_ratio_min)))
        open_, high, low = (_num(dataframe, name) for name in ("open", "high", "low"))
        candle_range = (high - low).replace(0.0, np.nan)
        pressure = ((((close - open_) / candle_range).clip(-1.0, 1.0).fillna(0.0) + ((((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)).fillna(0.0)) / 2.0).clip(-1.0, 1.0)
        pressure_window = int(self.pressure_window)
        pressure_baseline = volume.rolling(pressure_window, min_periods=max(2, pressure_window // 3)).sum().replace(0.0, np.nan)
        guard &= (pressure * volume.fillna(0.0)).rolling(pressure_window, min_periods=max(2, pressure_window // 3)).sum().div(pressure_baseline).le(-float(self.pressure_min))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe['enter_long'], dataframe['enter_short'], dataframe['enter_tag'] = (0, 0, None)
        lower = _num(dataframe, 'pg2_descending_channel_lower').mul(1.0 - float(self.rail_buffer_pct))
        condition = _bool(dataframe, 'pg2_descending_channel_pattern_present') & _num(dataframe, 'pg2_descending_channel_indicator_score').ge(float(self.score_min)) & _num(dataframe, 'pg2_descending_channel_width_atr').le(float(self.width_atr_max)) & _cross_below(_num(dataframe, 'close'), lower)
        condition &= self._common_guards(dataframe)
        valid = condition.fillna(False) & dataframe['volume'].gt(0.0) & dataframe['close'].notna()
        dataframe.loc[valid, 'enter_short'] = 1
        dataframe.loc[valid, 'enter_tag'] = ENTRY_TAG
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"], dataframe["exit_short"], dataframe["exit_tag"] = 0, 0, None
        return dataframe

    def _frame(self, pair: str) -> DataFrame:
        if getattr(self, "dp", None) is None:
            raise RuntimeError("measured channel exits require the analyzed dataframe")
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if frame is None or frame.empty:
            raise RuntimeError("measured channel exits require a non-empty analyzed dataframe")
        result = frame.copy()
        result["date"] = pd.to_datetime(result["date"], utc=True, errors="raise")
        return result.sort_values("date")

    def _entry_geometry(self, pair: str, trade: Any) -> tuple[float, float]:
        frame = self._frame(pair)
        opened = pd.Timestamp(trade.open_date_utc)
        if opened.tzinfo is None:
            opened = opened.tz_localize("UTC")
        entry_candle = pd.Timestamp(timeframe_to_prev_date(self.timeframe, opened.to_pydatetime()))
        signal_rows = frame.loc[frame["date"].lt(entry_candle)]
        if signal_rows.empty:
            raise RuntimeError("entry signal candle is unavailable for channel target reconstruction")
        row = signal_rows.iloc[-1]
        lower, upper = float(row["pg2_descending_channel_lower"]), float(row["pg2_descending_channel_upper"])
        if not np.isfinite([lower, upper]).all() or upper <= lower:
            raise RuntimeError("entry signal has no valid descending-channel rails")
        return lower, upper - lower

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | bool | None:
        _ = current_time, current_profit, kwargs
        multiplier, zone_fraction, _ = self.TARGET_ZONE_PLANS[str(self.target_zone_plan.value)]
        lower, width = self._entry_geometry(pair, trade)
        target = lower - width * multiplier
        trigger = target + width * zone_fraction
        if trigger >= float(trade.open_rate):
            return None
        return f"measured_channel_zone_{self.target_zone_plan.value}" if current_rate <= trigger else None

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float | None:
        _ = pair, current_time, current_profit, after_fill, kwargs
        _, _, stop_ratio = self.TARGET_ZONE_PLANS[str(self.target_zone_plan.value)]
        return stoploss_from_absolute(float(trade.open_rate) * (1.0 + stop_ratio), current_rate=current_rate, is_short=True, leverage=float(getattr(trade, "leverage", 1.0) or 1.0))


__all__ = ["Sieve3V2MeasuredChannelTargetZonesFromCompletePatternGeometryDescendingChannelLowerBreakdownShort4H"]

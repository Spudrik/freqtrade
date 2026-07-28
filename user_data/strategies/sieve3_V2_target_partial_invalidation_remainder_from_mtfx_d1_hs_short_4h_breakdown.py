from __future__ import annotations
import math
import pandas as pd
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from typing import Any
from pandas import DataFrame, Series
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, stoploss_from_absolute, IntParameter
import numpy as np
from freqtrade.strategy import IStrategy
from user_data.Indicators.complex_trendline_projection_v2 import add_trendline_projection_v2
from user_data.Indicators.complex_volume_profile import add_volume_profile
from user_data.Indicators.pattern_bos_choch import add_bos_choch
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2
from user_data.Indicators.pattern_reversal import add_pattern_reversal
NOVEL_IDEA = True
ENTRY_MODE = 'entry_sieve2_mtfx_d1_hs_short_4h_breakdown'
ENTRY_TAG = 'sieve2_mtfx_d1_hs_short_4h_breakdown'
ENTRY_SOURCE_STAGE = "sieve2"
SIDE = 'short'
TIMEFRAME = '4h'
CONTEXT_TIMEFRAME = '1d'
HTF_CONCEPT = 'head_shoulders_short'
LTF_TRIGGER = 'breakdown'


def _num(frame: DataFrame, column: str, default: float | Series=...) -> Series:
    if column not in frame.columns:
        if default is ...:
            raise KeyError(f"missing required column: {column!r}")
        return pd.to_numeric(
            pd.Series(default, index=frame.index), errors='coerce'
        ).replace([np.inf, -np.inf], np.nan)
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)

def _bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        raise KeyError(f"missing required column: {column!r}")
    return pd.Series(frame[column], index=frame.index).astype('boolean').fillna(False).astype(bool)

def _cross_above(value: Series, level: Series) -> Series:
    return value.gt(level) & value.shift(1).le(level.shift(1))

def _cross_below(value: Series, level: Series) -> Series:
    return value.lt(level) & value.shift(1).ge(level.shift(1))
SIEVE_STAGE = "sieve3"
SOURCE_ENTRY_STAGE = 'sieve2'
SOURCE_ENTRY_CLASS = 'Sieve3ExitFixedTpSlFromMtfxD1HsShort4HBreakdown'
SOURCE_STRATEGY = 'C:/FreqTradeStuff/user_data/strategies/sieve3_exit_fixed_tp_sl_from_mtfx_d1_hs_short_4h_breakdown.py:Sieve3ExitFixedTpSlFromMtfxD1HsShort4HBreakdown'
LINEAGE_STATUS = 'verified'
EXIT_FAMILY = 'target_partial_invalidation_remainder'
EXIT_THEORY = 'target_partial_invalidation_remainder'
EXIT_HYPOTHESIS = 'Take an optional target partial, then manage the remainder against entry-specific invalidation and stop movement.'
PRIMARY_TRIGGER = 'provider=D1 head-and-shoulders neckline with 4h breakdown execution'
PRIMARY_GUARD = 'provider=selected HTF pattern context plus locked LTF volume, pressure, and optional Sieve2 guards'
TARGET_PROVIDER = 'head-and-shoulders measured target'
INVALIDATION_PROVIDER = 'head-and-shoulders neckline'
ACTIVE_SELL_PARAMS = ('exit_plan', 'target_band_quarter_percent', 'partial_five_percent_units', 'stop_after_partial', 'invalidation_band_quarter_percent', 'invalidation_confirmations')

RESEARCH_PATH = 'sieve3_exit_target_partial_invalidation_remainder'
SOURCE_RESULT_BATCH = "historical_promotion_lineage_unknown"

class Sieve3V2TargetPartialInvalidationRemainderFromMtfxD1HsShort4HBreakdown(IStrategy):
    """Sieve2 structural MTF expansion: HTF concept plus LTF structural execution."""
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 240
    process_only_new_candles = True
    can_short = True
    max_entry_position_adjustment = 0
    score_min = 0.72
    htf_recent_bars = 2
    level_buffer_pct = 0.004
    local_window = 24
    use_ltf_volume_guard = True
    volume_ratio_min = 1.6
    # Legacy inactive entry parameter: use_ltf_pressure_guard=False (enable gate is fixed off).
    # Legacy inactive entry parameter: pressure_min=0.07 (its enable gate is fixed off).

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = (pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs)
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        if not self.dp:
            return []
        return [(pair, CONTEXT_TIMEFRAME) for pair in self.dp.current_whitelist()]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self._add_ltf_features(dataframe)
        dataframe = self._merge_htf_context(dataframe, metadata or {})
        return dataframe

    def _add_ltf_features(self, dataframe: DataFrame) -> DataFrame:
        frame = dataframe.copy()
        frame = add_bos_choch(frame, strength=3, min_prominence_atr=0.25, min_pivot_spacing_bars=1, max_pivot_age_bars=96, breakout_buffer_atr=0.1, prefix='ltfms')
        close = _num(frame, 'close')
        open_ = _num(frame, 'open')
        high = _num(frame, 'high')
        low = _num(frame, 'low')
        volume = _num(frame, 'volume').clip(lower=0.0).fillna(0.0)
        candle_range = (high - low).replace(0.0, np.nan)
        pressure = (((close - open_) / candle_range).clip(-1.0, 1.0).fillna(0.0) + ((close - low) / candle_range * 2.0 - 1.0).clip(-1.0, 1.0).fillna(0.0)) / 2.0
        window = int(self.local_window)
        baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
        frame['mtfx_ltf_pressure'] = pressure
        frame['mtfx_ltf_volume_ratio'] = volume / baseline
        frame['mtfx_ltf_near_level_recent'] = False
        return frame

    def _merge_htf_context(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        frame = dataframe.copy()
        frame['mtfx_htf_context'] = False
        frame['mtfx_htf_level'] = np.nan
        frame['sieve3_exit_target_level'] = np.nan
        frame['sieve3_exit_invalidation_level'] = np.nan
        if not self.dp:
            return frame
        pair = metadata.get('pair')
        if not pair:
            return frame
        informative = self.dp.get_pair_dataframe(pair=pair, timeframe=CONTEXT_TIMEFRAME)
        if informative is None or informative.empty or 'date' not in informative.columns:
            return frame
        informative = self._add_htf_features(informative.copy())
        informative = informative[['date', 'mtfx_htf_context', 'mtfx_htf_level', 'sieve3_exit_target_level', 'sieve3_exit_invalidation_level']].copy().sort_values('date')
        informative['date_merge'] = pd.to_datetime(informative['date'], utc=True).dt.tz_convert(None) + pd.to_timedelta(timeframe_to_minutes(CONTEXT_TIMEFRAME), unit='m')
        base = frame.drop(columns=['mtfx_htf_context', 'mtfx_htf_level', 'sieve3_exit_target_level', 'sieve3_exit_invalidation_level'], errors='ignore').reset_index().rename(columns={'index': '__row_index'}).sort_values('date')
        base['__date_merge'] = pd.to_datetime(base['date'], utc=True).dt.tz_convert(None)
        merged = pd.merge_asof(base, informative.drop(columns=['date']).sort_values('date_merge'), left_on='__date_merge', right_on='date_merge', direction='backward')
        merged = merged.set_index('__row_index').reindex(frame.index)
        frame['mtfx_htf_context'] = pd.Series(merged['mtfx_htf_context'], index=frame.index).astype('boolean').fillna(False).astype(bool)
        frame['mtfx_htf_level'] = pd.to_numeric(merged['mtfx_htf_level'], errors='coerce')
        frame['sieve3_exit_target_level'] = pd.to_numeric(merged['sieve3_exit_target_level'], errors='coerce')
        frame['sieve3_exit_invalidation_level'] = pd.to_numeric(merged['sieve3_exit_invalidation_level'], errors='coerce')
        return frame

    def _add_htf_features(self, dataframe: DataFrame) -> DataFrame:
        frame = dataframe.copy()
        frame = add_pattern_reversal(frame, timeframe=CONTEXT_TIMEFRAME, min_double_quality=float(self.score_min))
        frame = add_pattern_geometry_v2(frame, timeframe=CONTEXT_TIMEFRAME, output_slots=1, include_triangle_patterns=False, include_wedge_patterns=True, include_compression_patterns=False, include_rectangle_patterns=True, include_ascending_channel_patterns=False, include_descending_channel_patterns=False, min_line_score=float(self.score_min), output_prefix='pg2')
        frame = add_volume_profile(frame, window=96, bins=48, value_area_pct=0.7, price_source='hlc3', smooth_bins=3, pressure_delta_min=0.05, node_near_pct=float(self.level_buffer_pct), prefix='htfvp')
        frame = add_trendline_projection_v2(frame, timeframe=CONTEXT_TIMEFRAME, pivot_strength=3, min_output_line_score=float(self.score_min), raw_line_output_count=1, output_prefix='tlv2')
        frame = add_bos_choch(frame, strength=3, min_prominence_atr=0.25, min_pivot_spacing_bars=1, max_pivot_age_bars=160, breakout_buffer_atr=0.1, prefix='htfms')
        frame = self._add_local_htf_levels(frame)
        context, level = self._htf_context_and_level(frame)
        recent = int(self.htf_recent_bars)
        frame['mtfx_htf_context'] = context.fillna(False).astype('int8').rolling(recent, min_periods=1).max().gt(0)
        frame['mtfx_htf_level'] = level
        frame = self._sieve3_add_exit_levels(frame)
        return frame

    def _sieve3_add_exit_levels(self, dataframe: DataFrame) -> DataFrame:
        frame = dataframe.copy()
        close = _num(frame, 'close')
        target = _num(frame, 'pat_head_shoulders_target_level', np.nan)
        invalidation = _num(frame, 'pat_head_shoulders_confirmation_level', np.nan)
        recent = max(1, int(self.htf_recent_bars))
        if recent > 1:
            target = target.ffill(limit=recent - 1)
            invalidation = invalidation.ffill(limit=recent - 1)
        if SIDE == 'long':
            target = target.where(target.gt(close.mul(1.001)))
            invalidation = invalidation.where(invalidation.lt(close))
        else:
            target = target.where(target.lt(close.mul(0.999)))
            invalidation = invalidation.where(invalidation.gt(close))
        context = _bool(frame, 'mtfx_htf_context')
        frame['sieve3_exit_target_level'] = target.where(context)
        frame['sieve3_exit_invalidation_level'] = invalidation.where(context)
        return frame

    def _add_local_htf_levels(self, dataframe: DataFrame) -> DataFrame:
        frame = dataframe.copy()
        close = _num(frame, 'close')
        open_ = _num(frame, 'open')
        high = _num(frame, 'high')
        low = _num(frame, 'low')
        _num(frame, 'volume').clip(lower=0.0).fillna(0.0)
        lookback = 48
        prior_high = high.shift(1)
        prior_low = low.shift(1)
        frame['mtfx_prior_high'] = prior_high.rolling(lookback, min_periods=12).max()
        frame['mtfx_prior_low'] = prior_low.rolling(lookback, min_periods=12).min()
        tol = float(self.level_buffer_pct)
        frame['mtfx_equal_high'] = frame['mtfx_prior_high'].where(prior_high.sub(frame['mtfx_prior_high']).abs().le(frame['mtfx_prior_high'].abs().mul(tol)).rolling(lookback, min_periods=12).sum().ge(2))
        frame['mtfx_equal_low'] = frame['mtfx_prior_low'].where(prior_low.sub(frame['mtfx_prior_low']).abs().le(frame['mtfx_prior_low'].abs().mul(tol)).rolling(lookback, min_periods=12).sum().ge(2))
        body_high = pd.concat([open_, close], axis=1).max(axis=1)
        body_low = pd.concat([open_, close], axis=1).min(axis=1)
        avg_range = (high - low).rolling(20, min_periods=5).mean()
        bull_impulse = close.gt(open_) & (close - open_).gt(avg_range.mul(0.8))
        bear_impulse = close.lt(open_) & (open_ - close).gt(avg_range.mul(0.8))
        frame['mtfx_demand_zone_low'] = low.where(bull_impulse).ffill().shift(1)
        frame['mtfx_demand_zone_high'] = body_low.where(bull_impulse).ffill().shift(1)
        frame['mtfx_supply_zone_low'] = body_high.where(bear_impulse).ffill().shift(1)
        frame['mtfx_supply_zone_high'] = high.where(bear_impulse).ffill().shift(1)
        low_reset = prior_low.le(prior_low.rolling(72, min_periods=12).min())
        high_reset = prior_high.ge(prior_high.rolling(72, min_periods=12).max())
        frame['mtfx_avwap_from_low'] = self._anchored_vwap(frame, low_reset)
        frame['mtfx_avwap_from_high'] = self._anchored_vwap(frame, high_reset)
        return frame

    def _anchored_vwap(self, dataframe: DataFrame, reset: Series) -> Series:
        close = _num(dataframe, 'close')
        high = _num(dataframe, 'high')
        low = _num(dataframe, 'low')
        volume = _num(dataframe, 'volume').clip(lower=0.0).fillna(0.0)
        typical = (high + low + close) / 3.0
        group = reset.fillna(False).astype(int).cumsum()
        pv = (typical * volume).groupby(group).cumsum()
        vv = volume.groupby(group).cumsum().replace(0.0, np.nan)
        return pv / vv

    def _htf_context_and_level(self, dataframe: DataFrame) -> tuple[Series, Series]:
        close = _num(dataframe, 'close')
        high = _num(dataframe, 'high')
        low = _num(dataframe, 'low')
        score_min = float(self.score_min)
        near = float(self.level_buffer_pct)
        false = pd.Series(False, index=dataframe.index, dtype='bool')
        level = pd.Series(np.nan, index=dataframe.index, dtype='float64')
        concept = HTF_CONCEPT
        pattern_map = {'head_shoulders_short': ('pat_head_shoulders', 'short'), 'inverse_head_shoulders_long': ('pat_inverse_head_shoulders', 'long'), 'double_bottom_long': ('pat_double_bottom', 'long'), 'double_top_short': ('pat_double_top', 'short'), 'triple_bottom_long': ('pat_triple_bottom', 'long'), 'triple_top_short': ('pat_triple_top', 'short')}
        if concept in pattern_map:
            prefix, direction = pattern_map[concept]
            level = _num(dataframe, f'{prefix}_confirmation_level', np.nan)
            context = _bool(dataframe, f'{prefix}_pattern_confirmed') & _num(dataframe, f'{prefix}_indicator_score').ge(score_min)
            context &= close.le(level.mul(1.0 + near)) if direction == 'short' else close.ge(level.mul(1.0 - near))
            return (context, level)
        if concept in {'rectangle_long', 'rectangle_short'}:
            level = _num(dataframe, 'pg2_rectangle_upper', np.nan) if SIDE == 'long' else _num(dataframe, 'pg2_rectangle_lower', np.nan)
            context = _bool(dataframe, 'pg2_rectangle_pattern_present') & _num(dataframe, 'pg2_rectangle_indicator_score').ge(score_min)
            return (context, level)
        if concept in {'wedge_long', 'wedge_short'}:
            level = _num(dataframe, 'pg2_wedge_upper', np.nan) if SIDE == 'long' else _num(dataframe, 'pg2_wedge_lower', np.nan)
            direction_ok = _num(dataframe, 'pg2_wedge_direction').ge(0) if SIDE == 'long' else _num(dataframe, 'pg2_wedge_direction').le(0)
            context = _bool(dataframe, 'pg2_wedge_pattern_present') & _bool(dataframe, 'pg2_wedge_squeeze_active') & direction_ok & _num(dataframe, 'pg2_wedge_indicator_score').ge(score_min)
            return (context, level)
        if concept == 'tlv2_res_break_long':
            level = _num(dataframe, 'tlv2_resistance_line_rank0', np.nan)
            return (_num(dataframe, 'tlv2_resistance_score_rank0').ge(score_min) & close.ge(level.mul(1.0 - near)), level)
        if concept == 'tlv2_sup_reclaim_long':
            level = _num(dataframe, 'tlv2_support_line_rank0', np.nan)
            return (_num(dataframe, 'tlv2_support_score_rank0').ge(score_min) & close.ge(level.mul(1.0 - near)), level)
        if concept == 'tlv2_sup_break_short':
            level = _num(dataframe, 'tlv2_support_line_rank0', np.nan)
            return (_num(dataframe, 'tlv2_support_score_rank0').ge(score_min) & close.le(level.mul(1.0 + near)), level)
        if concept == 'vp_poc_reclaim_long':
            level = _num(dataframe, 'htfvp_prior_poc', np.nan)
            return (_num(dataframe, 'htfvp_context_score_bull').ge(0.2) & close.ge(level.mul(1.0 - near)), level)
        if concept == 'vp_vah_reject_short':
            level = _num(dataframe, 'htfvp_prior_vah', np.nan)
            return (_num(dataframe, 'htfvp_context_score_bear').ge(0.2) & close.le(level.mul(1.0 + near)), level)
        if concept == 'vp_val_reclaim_long':
            level = _num(dataframe, 'htfvp_prior_val', np.nan)
            return (_num(dataframe, 'htfvp_context_score_bull').ge(0.2) & close.ge(level.mul(1.0 - near)), level)
        if concept == 'vp_poc_reject_short':
            level = _num(dataframe, 'htfvp_prior_poc', np.nan)
            return (_num(dataframe, 'htfvp_context_score_bear').ge(0.2) & close.le(level.mul(1.0 + near)), level)
        if concept == 'vp_lvn_traverse_long':
            level = _num(dataframe, 'htfvp_lvn_above', np.nan)
            return (level.notna(), level)
        if concept == 'vp_lvn_traverse_short':
            level = _num(dataframe, 'htfvp_lvn_below', np.nan)
            return (level.notna(), level)
        if concept == 'liq_equal_lows_long':
            level = _num(dataframe, 'mtfx_equal_low', np.nan)
            return (low.le(level.mul(1.0 - near)) & close.gt(level), level)
        if concept == 'liq_equal_highs_short':
            level = _num(dataframe, 'mtfx_equal_high', np.nan)
            return (high.ge(level.mul(1.0 + near)) & close.lt(level), level)
        if concept == 'liq_prior_low_long':
            level = _num(dataframe, 'mtfx_prior_low', np.nan)
            return (low.le(level.mul(1.0 - near)) & close.gt(level), level)
        if concept == 'liq_prior_high_short':
            level = _num(dataframe, 'mtfx_prior_high', np.nan)
            return (high.ge(level.mul(1.0 + near)) & close.lt(level), level)
        if concept == 'demand_zone_long':
            level = _num(dataframe, 'mtfx_demand_zone_high', np.nan)
            return (close.ge(level.mul(1.0 - near)), level)
        if concept == 'supply_zone_short':
            level = _num(dataframe, 'mtfx_supply_zone_low', np.nan)
            return (close.le(level.mul(1.0 + near)), level)
        if concept == 'avwap_reclaim_long':
            level = _num(dataframe, 'mtfx_avwap_from_low', np.nan)
            return (close.ge(level.mul(1.0 - near)), level)
        if concept == 'avwap_reject_short':
            level = _num(dataframe, 'mtfx_avwap_from_high', np.nan)
            return (close.le(level.mul(1.0 + near)), level)
        return (false, level)

    def _near_level_recent(self, dataframe: DataFrame, level: Series) -> Series:
        near = float(self.level_buffer_pct)
        window = int(self.local_window)
        low = _num(dataframe, 'low')
        high = _num(dataframe, 'high')
        touched = low.le(level.mul(1.0 + near)) & high.ge(level.mul(1.0 - near))
        return touched.fillna(False).astype('int8').rolling(window, min_periods=1).max().gt(0)

    def _ltf_trigger(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        open_ = _num(dataframe, 'open')
        high = _num(dataframe, 'high')
        low = _num(dataframe, 'low')
        level = _num(dataframe, 'mtfx_htf_level', np.nan)
        near = float(self.level_buffer_pct)
        trigger = LTF_TRIGGER
        if trigger == 'breakout':
            return _cross_above(close, level.mul(1.0 + near))
        if trigger == 'breakdown':
            return _cross_below(close, level.mul(1.0 - near))
        if trigger in {'retest', 'reclaim'}:
            return low.le(level.mul(1.0 + near)) & close.gt(level) & close.gt(open_)
        if trigger == 'reject':
            return high.ge(level.mul(1.0 - near)) & close.lt(level) & close.lt(open_)
        if trigger == 'bos':
            event = _bool(dataframe, 'ltfms_bos_to_bull') if SIDE == 'long' else _bool(dataframe, 'ltfms_bos_to_bear')
            return event & self._near_level_recent(dataframe, level)
        if trigger == 'choch':
            event = _bool(dataframe, 'ltfms_choch_to_bull') if SIDE == 'long' else _bool(dataframe, 'ltfms_choch_to_bear')
            return event & self._near_level_recent(dataframe, level)
        return pd.Series(False, index=dataframe.index, dtype='bool')

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = None
        condition = _bool(dataframe, 'mtfx_htf_context') & self._ltf_trigger(dataframe)
        if bool(self.use_ltf_volume_guard):
            condition &= _num(dataframe, 'mtfx_ltf_volume_ratio').ge(float(self.volume_ratio_min))
        valid = condition.fillna(False) & dataframe['volume'].gt(0.0) & dataframe['close'].notna()
        if SIDE == 'short':
            dataframe.loc[valid, 'enter_short'] = 1
        else:
            dataframe.loc[valid, 'enter_long'] = 1
        dataframe.loc[valid, 'enter_tag'] = ENTRY_TAG
        return dataframe
    SOURCE_ENTRY_STEM = 'mtfx_d1_hs_short_4h_breakdown'
    FOCUSED_EXIT_CONTRACT = 'target_partial_invalidation_remainder'
    FOCUSED_SOURCE_PROFILE = {'side': 'short', 'min_level_distance': 0.001, 'targets': {'target_1': {'provider': 'head-and-shoulders measured target', 'short': {'level': 'sieve3_exit_target_level', 'available': None}}}, 'invalidation': {'provider': 'head-and-shoulders neckline', 'mode': 'level', 'short': {'level': 'sieve3_exit_invalidation_level', 'available': None}}}
    FOCUSED_EXIT_PLANS = {'touch_partial': {'contract': 'target_partial_invalidation_remainder', 'name': 'touch_partial', 'hard_stop_ratio': 0.03, 'max_hold_candles': 336}, 'close_partial': {'contract': 'target_partial_invalidation_remainder', 'name': 'close_partial', 'hard_stop_ratio': 0.03, 'max_hold_candles': 336}, 'reversal_1_partial': {'contract': 'target_partial_invalidation_remainder', 'name': 'reversal_1_partial', 'hard_stop_ratio': 0.03, 'max_hold_candles': 336}, 'reversal_2_of_3_partial': {'contract': 'target_partial_invalidation_remainder', 'name': 'reversal_2_of_3_partial', 'hard_stop_ratio': 0.03, 'max_hold_candles': 336}}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'high', 'low', 'open', 'sieve3_exit_invalidation_level', 'sieve3_exit_target_level')
    FOCUSED_STATE_VERSION = 2
    FOCUSED_STATE_KEY = 'sieve3_v2_focused:Sieve3V2TargetPartialInvalidationRemainderFromMtfxD1HsShort4HBreakdown:target_partial_invalidation_remainder'
    LOCKED_BUY_PARAMS = {'use_sieve2_vp_guard': False, 'sieve2_vp_guard_mode': 'score_or_context', 'sieve2_vp_window': 96, 'sieve2_vp_bins': 36, 'sieve2_vp_score_min': 0.25, 'sieve2_vp_context_min': 0.28, 'use_sieve2_market_guard': False, 'sieve2_market_guard_mode': 'pressure_or_trend', 'sieve2_market_window': 24, 'sieve2_market_pressure_min': 0.07, 'sieve2_market_trend_min': 0.25, 'sieve2_rs_benchmark_pair': 'BTC/USDT:USDT', 'sieve2_rs_score_min': 0.45, 'score_min': 0.72, 'htf_recent_bars': 2, 'level_buffer_pct': 0.004, 'local_window': 24, 'use_ltf_volume_guard': True, 'volume_ratio_min': 1.6, 'use_ltf_pressure_guard': False, 'pressure_min': 0.07}
    ACTIVE_SELL_PARAMS = ('exit_plan', 'target_band_quarter_percent', 'partial_five_percent_units', 'stop_after_partial', 'invalidation_band_quarter_percent', 'invalidation_confirmations')
    position_adjustment_enable = True
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    minimal_roi = {'0': 100.0}
    stoploss = -0.99
    exit_plan = CategoricalParameter(('touch_partial', 'close_partial', 'reversal_1_partial', 'reversal_2_of_3_partial'), default='touch_partial', space='sell', optimize=True, load=True)
    exit_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    target_band_quarter_percent = IntParameter(0, 12, default=2, space='sell', optimize=True, load=True)
    target_band_quarter_percent.batch_tags = ('family:exits', 'mode:sieve3_exit')
    partial_five_percent_units = IntParameter(1, 8, default=5, space='sell', optimize=True, load=True)
    partial_five_percent_units.batch_tags = ('family:exits', 'mode:sieve3_exit')
    stop_after_partial = CategoricalParameter(('unchanged', 'entry'), default='entry', space='sell', optimize=True, load=True)
    stop_after_partial.batch_tags = ('family:exits', 'mode:sieve3_exit')
    invalidation_band_quarter_percent = IntParameter(0, 4, default=1, space='sell', optimize=True, load=True)
    invalidation_band_quarter_percent.batch_tags = ('family:exits', 'mode:sieve3_exit')
    invalidation_confirmations = IntParameter(1, 3, default=1, space='sell', optimize=True, load=True)
    invalidation_confirmations.batch_tags = ('family:exits', 'mode:sieve3_exit')

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        dataframe['exit_tag'] = None
        return dataframe

    @staticmethod
    def _focused_utc(value: Any) -> pd.Timestamp | None:
        if value is None:
            return None
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize('UTC')
        return timestamp.tz_convert('UTC')

    @staticmethod
    def _focused_float(value: Any) -> float | None:
        if value is None or bool(pd.isna(value)):
            return None
        number = float(value)
        return number if math.isfinite(number) else None

    @staticmethod
    def _focused_number(frame: DataFrame, column: str) -> Series:
        return pd.to_numeric(frame[column], errors='coerce')


    def _focused_close_times(self, frame: DataFrame) -> Series:
        dates = pd.to_datetime(frame['date'], utc=True, errors='raise')
        return dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit='m')

    def _focused_closed_frame(self, frame: DataFrame, current_time: Any) -> DataFrame:
        if frame is None or frame.empty:
            raise RuntimeError('focused exit runtime requires a non-empty analyzed dataframe')
        if 'date' not in frame.columns:
            raise KeyError('focused exit runtime requires the dataframe date column')
        now = self._focused_utc(current_time)
        if now is None:
            raise ValueError('current_time is required for closed-candle evaluation')
        close_times = self._focused_close_times(frame)
        closed = frame.loc[close_times.le(now)].copy()
        return closed.sort_values('date').drop_duplicates('date', keep='last')

    def _focused_analyzed_frame(self, pair: str, current_time: Any) -> DataFrame:
        _ = current_time
        if getattr(self, 'dp', None) is None:
            raise RuntimeError("focused exit runtime requires Freqtrade's data provider")
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if frame is None or frame.empty:
            raise RuntimeError('focused exit runtime requires a non-empty analyzed dataframe')
        if 'date' not in frame.columns:
            raise KeyError('focused exit runtime requires the dataframe date column')
        return frame

    def _focused_require_columns(self, frame: DataFrame) -> None:
        missing = sorted(set(self.FOCUSED_REQUIRED_COLUMNS) - set(frame.columns))
        if missing:
            raise KeyError(f'focused source profile is missing exact columns: {missing}')

    def _focused_side(self, trade: Any) -> str:
        side = 'short' if bool(getattr(trade, 'is_short', False)) else 'long'
        declared = str(self.FOCUSED_SOURCE_PROFILE['side'])
        if declared != 'both' and declared != side:
            raise ValueError(f'trade side {side!r} violates exact profile side {declared!r}')
        return side

    def _focused_plan(self, state: Mapping[str, Any] | None = None) -> dict[str, Any]:
        name = str((state or {}).get('plan') or self.exit_plan.value)
        if name not in self.FOCUSED_EXIT_PLANS:
            raise ValueError(f'unknown focused exit plan: {name}')
        plan = dict(self.FOCUSED_EXIT_PLANS[name])
        plan['target_confirmation'] = {'touch_partial': 'touch', 'close_partial': 'close', 'reversal_1_partial': 'reversal_1', 'reversal_2_of_3_partial': 'reversal_2_of_3'}[name]
        plan['target_band'] = float(self.target_band_quarter_percent.value) * 0.0025
        plan['partial_fractions'] = (float(self.partial_five_percent_units.value) * 0.05,)
        plan['stop_after_partial'] = str(self.stop_after_partial.value)
        plan['invalidation_band'] = float(self.invalidation_band_quarter_percent.value) * 0.0025
        plan['invalidation_confirmations'] = int(self.invalidation_confirmations.value)
        return plan

    @staticmethod
    def _focused_decision(action: str='hold', tag: str | None=None, stage: str | None=None, fraction: float=0.0) -> dict[str, Any]:
        return {'action': action, 'tag': tag, 'stage': stage, 'fraction': float(fraction)}

    def _focused_state(self, trade: Any) -> dict[str, Any] | None:
        cache_attr = f'_sieve3_v2_state_cache_{self.FOCUSED_STATE_KEY}'
        state = getattr(trade, cache_attr, None)
        if state is None:
            state = trade.get_custom_data(key=self.FOCUSED_STATE_KEY)
        if state is None:
            return None
        if not isinstance(state, dict):
            raise ValueError('focused exit trade state must be a mapping')
        if state.get('version') != self.FOCUSED_STATE_VERSION:
            raise ValueError('focused exit trade state version mismatch')
        if state.get('contract') != self.FOCUSED_EXIT_CONTRACT:
            raise ValueError('focused exit trade state contract mismatch')
        if getattr(trade, cache_attr, None) is None:
            setattr(trade, cache_attr, deepcopy(state))
        return deepcopy(state)

    def _focused_save_state(self, trade: Any, state: Mapping[str, Any]) -> None:
        cache_attr = f'_sieve3_v2_state_cache_{self.FOCUSED_STATE_KEY}'
        snapshot = deepcopy(dict(state))
        setattr(trade, cache_attr, snapshot)
        trade.set_custom_data(key=self.FOCUSED_STATE_KEY, value=deepcopy(snapshot))

    def _focused_side_binding(self, role: str, side: str, slot: str | None=None) -> Mapping[str, Any] | None:
        if role == 'target':
            targets = self.FOCUSED_SOURCE_PROFILE.get('targets', {})
            provider = targets.get(str(slot))
        else:
            provider = self.FOCUSED_SOURCE_PROFILE.get(role)
        if not isinstance(provider, Mapping):
            return None
        binding = provider.get(side)
        return binding if isinstance(binding, Mapping) else None

    def _focused_frozen_level(self, row: Series, binding: Mapping[str, Any] | None, side: str, entry_rate: float, role: str) -> float | None:
        if binding is None:
            return None
        available_column = binding.get('available')
        if available_column:
            availability = row[str(available_column)]
            if bool(pd.isna(availability)) or not bool(availability):
                return None
        level_column = binding.get('level')
        if not level_column:
            return None
        level = self._focused_float(row[str(level_column)])
        if level is None or level <= 0.0:
            return None
        if role == 'invalidation':
            return level
        if role != 'target':
            raise ValueError(f'unsupported frozen level role: {role}')
        minimum = float(self.FOCUSED_SOURCE_PROFILE['min_level_distance'])
        valid = level < entry_rate * (1.0 - minimum) if side == 'short' else level > entry_rate * (1.0 + minimum)
        return level if valid else None

    def _focused_new_state(self, pair: str, trade: Any, current_time: Any, order: Any | None=None) -> dict[str, Any] | None:
        side = self._focused_side(trade)
        order_rate = self._focused_float(getattr(order, 'safe_price', None)) if order else None
        entry_rate = order_rate or self._focused_float(getattr(trade, 'open_rate', None))
        if entry_rate is None or entry_rate <= 0.0:
            raise ValueError('focused exit runtime requires a positive entry rate')
        freeze_time = (getattr(order, 'order_date_utc', None) if order is not None else None) or (getattr(order, 'order_date', None) if order is not None else None) or getattr(trade, 'open_date_utc', None) or getattr(trade, 'date_entry_fill_utc', None) or current_time
        frame = self._focused_closed_frame(self._focused_analyzed_frame(pair, freeze_time), freeze_time)
        if frame.empty:
            return None
        self._focused_require_columns(frame)
        row = frame.iloc[-1]
        levels: dict[str, float | None] = {}
        target_specs = self.FOCUSED_SOURCE_PROFILE.get('targets', {})
        if isinstance(target_specs, Mapping):
            for slot in target_specs:
                levels[str(slot)] = self._focused_frozen_level(row, self._focused_side_binding('target', side, str(slot)), side, entry_rate, 'target')
        invalidation_binding = self._focused_side_binding('invalidation', side)
        levels['invalidation'] = self._focused_frozen_level(row, invalidation_binding, side, entry_rate, 'invalidation')
        invalidation_level = levels.get('invalidation')
        for slot in ('target_1', 'target_2'):
            target_level = levels.get(slot)
            if target_level is not None and invalidation_level is not None and math.isclose(target_level, invalidation_level, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f'resolved {slot} equals the source invalidation level')
        first = levels.get('target_1')
        second = levels.get('target_2')
        if first is not None and second is not None:
            ordered = second < first if side == 'short' else second > first
            if not ordered:
                raise ValueError('entry-frozen target_2 must be beyond target_1')
        selected_plan = str(self.exit_plan.value)
        plan = self._focused_plan({'plan': selected_plan})
        fill_time = (getattr(order, 'order_filled_utc', None) if order is not None else None) or getattr(trade, 'date_entry_fill_utc', None) or getattr(trade, 'open_date_utc', None) or current_time
        fill_timestamp = self._focused_utc(fill_time)
        snapshot_date = self._focused_utc(row['date'])
        stages = {f'stage_{index + 1}': {'status': 'ready', 'fraction': float(fraction), 'tag': f'focused_partial_stage_{index + 1}', 'requested_at': None, 'filled_at': None, 'target_stake': None, 'credited_stake': 0.0} for index, fraction in enumerate(plan.get('partial_fractions', ()))}
        return {'version': self.FOCUSED_STATE_VERSION, 'contract': self.FOCUSED_EXIT_CONTRACT, 'plan': selected_plan, 'side': side, 'phase': 'pre_target', 'entry_rate': float(entry_rate), 'entry_filled_at': fill_timestamp.isoformat() if fill_timestamp is not None else None, 'entry_snapshot_candle': snapshot_date.isoformat() if snapshot_date is not None else None, 'levels': levels, 'target_states': {slot: {'status': 'available' if value is not None else 'unavailable', 'touched_at': None} for slot, value in levels.items() if slot.startswith('target_')}, 'invalidation_seen': False, 'guard_state': 'unknown', 'trigger_state': 'unknown', 'stages': stages, 'initial_stake': self._focused_float(getattr(trade, 'stake_amount', None)), 'favorable_rate': float(entry_rate), 'stop_price': None, 'terminal_pending': False, 'terminal_tag': None}

    def _focused_ensure_state(self, pair: str, trade: Any, current_time: Any, order: Any | None=None) -> dict[str, Any] | None:
        state = self._focused_state(trade)
        if state is None:
            state = self._focused_new_state(pair, trade, current_time, order)
            if state is None:
                return None
            self._focused_save_state(trade, state)
        return state

    def _focused_post_entry(self, frame: DataFrame, state: Mapping[str, Any]) -> DataFrame:
        cursor = self._focused_utc(state.get('last_processed_candle') or state.get('entry_snapshot_candle'))
        if cursor is None:
            raise ValueError('focused exit state has no processed-candle cursor')
        filled_at = self._focused_utc(state.get('entry_filled_at'))
        if filled_at is None:
            raise ValueError('focused exit state has no entry fill timestamp')
        latest = self._focused_utc(frame['date'].iat[-1])
        if latest is None or latest <= cursor:
            return frame.iloc[0:0]
        fill_cursor = filled_at - pd.Timedelta(minutes=timeframe_to_minutes(self.timeframe))
        fill_start = int(frame['date'].searchsorted(fill_cursor, side='right'))
        start = max(
            int(frame['date'].searchsorted(cursor, side='right')),
            fill_start,
        )
        return frame.iloc[start:]

    def _focused_update_favorable(self, post_entry: DataFrame, state: dict[str, Any]) -> None:
        entry_rate = float(state['entry_rate'])
        favorable = float(state.get('favorable_rate') or entry_rate)
        if not post_entry.empty:
            column = 'low' if state['side'] == 'short' else 'high'
            values = self._focused_number(post_entry, column)
            observed = self._focused_float(values.min() if state['side'] == 'short' else values.max())
            if observed is not None:
                favorable = min(favorable, observed) if state['side'] == 'short' else max(favorable, observed)
        state['favorable_rate'] = favorable

    @staticmethod
    def _focused_favorable_move(state: Mapping[str, Any]) -> float:
        entry_rate = float(state['entry_rate'])
        favorable = float(state.get('favorable_rate') or entry_rate)
        if state['side'] == 'short':
            return max(0.0, 1.0 - favorable / entry_rate)
        return max(0.0, favorable / entry_rate - 1.0)

    @staticmethod
    def _focused_stage_filled(state: Mapping[str, Any], stage: str) -> bool:
        stages = state.get('stages', {})
        return isinstance(stages, Mapping) and (stages.get(stage) or {}).get('status') == 'filled'

    @staticmethod
    def _focused_requested_stage(state: Mapping[str, Any]) -> str | None:
        stages = state.get('stages', {})
        if not isinstance(stages, Mapping):
            return None
        for name, stage in stages.items():
            if isinstance(stage, Mapping) and stage.get('status') == 'requested':
                return str(name)
        return None

    @staticmethod
    def _focused_tighter_stop(side: str, current: float, candidate: float) -> float:
        return min(current, candidate) if side == 'short' else max(current, candidate)

    @staticmethod
    def _focused_hard_stop_price(plan: Mapping[str, Any], state: Mapping[str, Any]) -> float:
        entry_rate = float(state['entry_rate'])
        ratio = float(plan['hard_stop_ratio'])
        return entry_rate * (1.0 + ratio if state['side'] == 'short' else 1.0 - ratio)

    def _focused_hard_stop_breached(self, plan: Mapping[str, Any], state: Mapping[str, Any], current_rate: float) -> bool:
        stop_price = self._focused_hard_stop_price(plan, state)
        return current_rate >= stop_price if state['side'] == 'short' else current_rate <= stop_price




    def _focused_context(self, pair: str, trade: Any, current_time: Any, current_rate: float, current_profit: float) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        state = self._focused_ensure_state(pair, trade, current_time)
        if state is None:
            return (self._focused_plan(), {}, {}, self._focused_decision())
        before = repr(state)
        plan = self._focused_plan(state)

        stages = state.get('stages', {})
        if isinstance(stages, Mapping):
            requested = self._focused_requested_stage(state)
            filled = sum(1 for stage in stages.values() if isinstance(stage, Mapping) and stage.get('status') == 'filled')
            native_filled = int(getattr(trade, 'nr_of_successful_exits', 0) or 0)
            if requested is not None or native_filled > filled:
                self._focused_sync_partial_stages(trade, state, current_time)

        now = self._focused_utc(current_time)
        filled_at = self._focused_utc(state.get('entry_filled_at'))
        if now is not None and filled_at is not None:
            timeframe_seconds = max(60, int(timeframe_to_minutes(self.timeframe)) * 60)
            age_candles = max(0, int((now - filled_at).total_seconds() // timeframe_seconds))
        else:
            age_candles = int(state.get('age_candles') or 0)

        side = str(state['side'])
        entry_rate = float(state['entry_rate'])
        favorable = self._focused_float(getattr(trade, 'min_rate' if side == 'short' else 'max_rate', None)) or entry_rate
        adverse = self._focused_float(getattr(trade, 'max_rate' if side == 'short' else 'min_rate', None)) or entry_rate
        favorable_move = max(0.0, 1.0 - favorable / entry_rate) if side == 'short' else max(0.0, favorable / entry_rate - 1.0)

        levels = state.get('levels', {})
        target = self._focused_float(levels.get('target_1')) if isinstance(levels, Mapping) else None
        invalidation = self._focused_float(levels.get('invalidation')) if isinstance(levels, Mapping) else None
        target_state = (state.get('target_states') or {}).get('target_1', {})
        target_status = str(target_state.get('status') or 'unavailable') if isinstance(target_state, Mapping) else 'unavailable'
        target_done = target_status in {'accepted', 'rejected', 'confirmed'}
        stage_1 = stages.get('stage_1', {}) if isinstance(stages, Mapping) else {}
        partial_done = isinstance(stage_1, Mapping) and stage_1.get('status') == 'filled'

        target_near = False
        if target is not None and not target_done and not partial_done:
            threshold = target * (1.0 + float(plan.get('target_band') or 0.0)) if side == 'short' else target * (1.0 - float(plan.get('target_band') or 0.0))
            target_near = target_status == 'touched' or (favorable <= threshold if side == 'short' else favorable >= threshold)

        invalidation_near = False
        if invalidation is not None and not state.get('invalidation_seen'):
            threshold = invalidation * (1.0 + float(plan.get('invalidation_band') or 0.0)) if side == 'short' else invalidation * (1.0 - float(plan.get('invalidation_band') or 0.0))
            invalidation_near = int(state.get('invalidation_streak') or 0) > 0 or (adverse >= threshold if side == 'short' else adverse <= threshold)

        if target_near or invalidation_near:
            frame = self._focused_analyzed_frame(pair, current_time)
            self._focused_require_columns(frame)
            post_entry = self._focused_post_entry(frame, state)
            events = self._focused_contract_events(post_entry, state, plan, current_profit)
            if not post_entry.empty:
                processed = self._focused_utc(post_entry['date'].iat[-1])
                state['last_processed_candle'] = processed.isoformat() if processed is not None else None
        else:
            events = {'invalidation': bool(state.get('invalidation_seen')), 'target_1': target_done}

        events['age_candles'] = age_candles
        events['favorable_move'] = favorable_move
        if self._focused_hard_stop_breached(plan, state, float(current_rate)):
            decision = self._focused_decision('full', 'focused_hard_stop')
        else:
            decision = self._focused_contract_decision(state, plan, events, current_profit)
        if decision['action'] == 'hold' and age_candles >= int(plan['max_hold_candles']):
            decision = self._focused_decision('full', 'focused_max_hold')
        if state.get('terminal_pending') and self._focused_requested_stage(state) is None:
            decision = self._focused_decision('full', str(state.get('terminal_tag') or 'focused_deferred_full'))
        if repr(state) != before:
            self._focused_save_state(trade, state)
        return (plan, state, events, decision)

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | bool | None:
        _ = kwargs
        _, state, _, decision = self._focused_context(pair, trade, current_time, current_rate, current_profit)
        if self._focused_requested_stage(state) is not None:
            if decision['action'] == 'full':
                state['terminal_pending'] = True
                state['terminal_tag'] = decision['tag']
                self._focused_save_state(trade, state)
            return None
        return str(decision['tag']) if decision['action'] == 'full' else None

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float | None:
        _ = (pair, current_time, current_profit, after_fill, kwargs)
        if bool(getattr(trade, 'has_open_orders', False)):
            return None
        plan = self._focused_plan()
        entry_rate = self._focused_float(getattr(trade, 'open_rate', None))
        if entry_rate is None or entry_rate <= 0.0:
            raise ValueError('focused exit runtime requires a positive trade open rate')
        is_short = bool(getattr(trade, 'is_short', False))
        stop_price = entry_rate * (1.0 + float(plan['hard_stop_ratio']) if is_short else 1.0 - float(plan['hard_stop_ratio']))
        exits = int(getattr(trade, 'nr_of_successful_exits', 0) or 0)
        if exits >= 1 and str(plan['stop_after_partial']) == 'entry':
            stop_price = entry_rate
        state = self._focused_state(trade)
        if state is not None and (state.get('invalidation_seen') or state.get('terminal_pending')):
            return None
        if (is_short and stop_price <= current_rate) or (not is_short and stop_price >= current_rate):
            return None
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=is_short, leverage=float(getattr(trade, 'leverage', 1.0) or 1.0))

    def _focused_target_event(self, post_entry: DataFrame, state: dict[str, Any], plan: Mapping[str, Any], slot: str) -> bool:
        target = self._focused_float((state.get('levels') or {}).get(slot))
        target_states = state.setdefault('target_states', {})
        target_state = target_states.setdefault(slot, {'status': 'unavailable', 'touched_at': None})
        if target_state.get('status') in {'accepted', 'rejected', 'confirmed'}:
            return True
        if target is None:
            target_state['status'] = 'unavailable'
            return False
        if post_entry.empty:
            return False
        band = float(plan.get('target_band') or 0.0)
        close_times = self._focused_close_times(post_entry)
        confirmation = str(plan.get('target_confirmation') or 'touch')
        touched_at = self._focused_utc(target_state.get('touched_at'))
        if confirmation not in {'touch', 'close', 'reversal_1', 'reversal_2_of_3'}:
            raise ValueError(f'unsupported target confirmation: {confirmation}')
        window = [bool(value) for value in target_state.get('reversal_window', [])][-2:]
        for position in range(len(post_entry)):
            row = post_entry.iloc[position]
            candle_close = self._focused_utc(close_times.iloc[position])
            low = self._focused_float(row['low'])
            high = self._focused_float(row['high'])
            touched = low is not None and low <= target * (1.0 + band) if state['side'] == 'short' else high is not None and high >= target * (1.0 - band)
            if touched_at is None and touched:
                touched_at = candle_close
                target_state['touched_at'] = touched_at.isoformat() if touched_at is not None else None
                target_state['status'] = 'touched'
                state['phase'] = f'{slot}_zone'
                if confirmation == 'touch':
                    target_state['status'] = 'confirmed'
                    return True
            if touched_at is None or candle_close is None:
                continue
            close = self._focused_float(row['close'])
            if confirmation == 'close':
                accepted = close is not None and (close <= target * (1.0 + band) if state['side'] == 'short' else close >= target * (1.0 - band))
                if candle_close >= touched_at and accepted:
                    target_state['status'] = 'accepted'
                    return True
                continue
            if candle_close <= touched_at:
                continue
            open_ = self._focused_float(row['open'])
            reversal = close is not None and open_ is not None and (close > open_ and close > target * (1.0 - band) if state['side'] == 'short' else close < open_ and close < target * (1.0 + band))
            window = (window + [bool(reversal)])[-3:]
            target_state['reversal_window'] = window
            confirmed = bool(reversal) if confirmation == 'reversal_1' else len(window) >= 3 and sum(window) >= 2
            if confirmed:
                target_state['status'] = 'rejected'
                return True
        if touched_at is None:
            target_state['status'] = 'available'
        return False

    def _focused_invalidation_event(self, post_entry: DataFrame, state: dict[str, Any], plan: Mapping[str, Any]) -> bool:
        if state.get('invalidation_seen'):
            return True
        if post_entry.empty:
            return False
        side = str(state['side'])
        band = float(plan.get('invalidation_band') or 0.0)
        level = self._focused_float((state.get('levels') or {}).get('invalidation'))
        close = self._focused_number(post_entry, 'close')
        if level is None:
            level_condition = pd.Series(False, index=post_entry.index, dtype='bool')
        elif side == 'short':
            level_condition = close.gt(level * (1.0 + band))
        else:
            level_condition = close.lt(level * (1.0 - band))
        confirmations = int(plan.get('invalidation_confirmations') or 1)
        streak = int(state.get('invalidation_streak') or 0)
        for value in level_condition.fillna(False).tolist():
            streak = streak + 1 if bool(value) else 0
            state['invalidation_streak'] = streak
            if streak >= confirmations:
                state['invalidation_seen'] = True
                state['phase'] = 'invalidated'
                return True
        return False

    def _focused_contract_events(self, post_entry: DataFrame, state: dict[str, Any], plan: Mapping[str, Any], current_profit: float) -> dict[str, Any]:
        _ = current_profit
        invalidation = self._focused_invalidation_event(post_entry, state, plan)
        target = self._focused_target_event(post_entry, state, plan, 'target_1')
        return {'invalidation': invalidation, 'target_1': target}

    def _focused_contract_decision(self, state: dict[str, Any], plan: Mapping[str, Any], events: Mapping[str, Any], current_profit: float) -> dict[str, Any]:
        _ = (plan, current_profit)
        if events.get('invalidation'):
            return self._focused_decision('full', 'focused_source_invalidation')
        stage = state['stages']['stage_1']
        if events.get('target_1') and stage['status'] == 'ready':
            return self._focused_decision('partial', str(stage['tag']), 'stage_1', float(stage['fraction']))
        return self._focused_decision()

    def _focused_breakeven_price(self, state: Mapping[str, Any]) -> float:
        entry_rate = self._focused_float(state.get('entry_rate'))
        if entry_rate is None or entry_rate <= 0.0:
            raise ValueError('focused exit state requires a positive entry rate for breakeven')
        return entry_rate


    def _focused_stop_overlay(self, plan: Mapping[str, Any], state: Mapping[str, Any], current_profit: float) -> float:
        _ = current_profit
        stop_price = self._focused_hard_stop_price(plan, state)
        if str(plan['stop_after_partial']) == 'entry' and self._focused_stage_filled(state, 'stage_1'):
            stop_price = self._focused_tighter_stop(str(state['side']), stop_price, self._focused_breakeven_price(state))
        return stop_price

    def order_filled(self, pair: str, trade: Any, order: Any, current_time: datetime, **kwargs: Any) -> None:
        _ = kwargs
        if getattr(order, 'ft_order_side', None) != getattr(trade, 'entry_side', None):
            return None
        if self._focused_state(trade) is not None:
            return None
        state = self._focused_new_state(pair, trade, current_time, order)
        if state is not None:
            self._focused_save_state(trade, state)
        return None

    def _focused_sync_partial_stages(self, trade: Any, state: dict[str, Any], current_time: Any) -> None:
        stages = state.get('stages', {})
        if not isinstance(stages, Mapping):
            return
        exit_side = getattr(trade, 'exit_side', None)
        orders = tuple(trade.select_filled_or_open_orders())
        now = self._focused_utc(current_time)
        for name, candidate in stages.items():
            if not isinstance(candidate, dict):
                continue
            stage = candidate
            tag = str(stage.get('tag') or '')
            matching = [
                order
                for order in orders
                if getattr(order, 'ft_order_side', None) == exit_side
                and str(getattr(order, 'ft_order_tag', None) or '') == tag
            ]
            credited = sum(
                max(0.0, float(getattr(order, 'stake_amount_filled', 0.0) or 0.0))
                for order in matching
            )
            stage['credited_stake'] = credited
            target = float(stage.get('target_stake') or 0.0)
            tolerance = max(1e-9, target * 1e-9)
            if any(bool(getattr(order, 'ft_is_open', False)) for order in matching):
                stage['status'] = 'requested'
                state['phase'] = f'{name}_pending'
                continue
            if target > 0.0 and credited >= target - tolerance:
                stage['status'] = 'filled'
                filled_times = [
                    self._focused_utc(getattr(order, 'order_filled_utc', None))
                    for order in matching
                ]
                latest_fill = max((value for value in filled_times if value is not None), default=None)
                stage['filled_at'] = latest_fill.isoformat() if latest_fill is not None else stage.get('filled_at')
                stage['requested_at'] = None
                state['phase'] = 'remainder' if name == 'stage_1' else 'runner'
                continue
            requested_at = self._focused_utc(stage.get('requested_at'))
            if stage.get('status') == 'requested' and requested_at is not None and requested_at == now:
                continue
            if stage.get('status') != 'filled':
                stage['status'] = 'ready'
                stage['requested_at'] = None
                stage['filled_at'] = None
                state['phase'] = f'{name}_retry' if matching else state.get('phase', 'pre_target')

    def adjust_trade_position(self, trade: Any, current_time: datetime, current_rate: float, current_profit: float, min_stake: float | None, max_stake: float, current_entry_rate: float, current_exit_rate: float, current_entry_profit: float, current_exit_profit: float, **kwargs: Any) -> float | None | tuple[float | None, str | None]:
        _ = (max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, kwargs)
        _, state, _, decision = self._focused_context(str(getattr(trade, 'pair', '')), trade, current_time, current_rate, current_profit)
        if not state:
            return None
        if trade.has_open_orders:
            return None
        if decision['action'] != 'partial' or not decision.get('stage'):
            return None
        stage_name = str(decision['stage'])
        stage = state['stages'][stage_name]
        if stage['status'] != 'ready':
            return None
        current_stake = self._focused_float(getattr(trade, 'stake_amount', None))
        initial_stake = self._focused_float(state.get('initial_stake'))
        if current_stake is None or initial_stake is None or current_stake <= 0.0:
            return None
        target_stake = self._focused_float(stage.get('target_stake'))
        if target_stake is None:
            target_stake = min(current_stake, initial_stake * float(decision['fraction']))
            if min_stake is not None and 0.0 < current_stake - target_stake < float(min_stake):
                target_stake = current_stake - float(min_stake)
            if target_stake <= 0.0 or target_stake >= current_stake:
                return None
            stage['target_stake'] = target_stake
        credited_stake = max(0.0, float(stage.get('credited_stake') or 0.0))
        remaining_stake = max(0.0, target_stake - credited_stake)
        tolerance = max(1e-9, target_stake * 1e-9)
        if remaining_stake <= tolerance:
            stage['status'] = 'filled'
            state['phase'] = 'remainder' if stage_name == 'stage_1' else 'runner'
            self._focused_save_state(trade, state)
            return None
        request = min(current_stake, remaining_stake)
        if request <= 0.0 or request >= current_stake:
            return None
        stage['status'] = 'requested'
        stage['requested_at'] = self._focused_utc(current_time).isoformat()
        state['phase'] = f'{stage_name}_pending'
        self._focused_save_state(trade, state)
        return (-request, str(stage['tag']))

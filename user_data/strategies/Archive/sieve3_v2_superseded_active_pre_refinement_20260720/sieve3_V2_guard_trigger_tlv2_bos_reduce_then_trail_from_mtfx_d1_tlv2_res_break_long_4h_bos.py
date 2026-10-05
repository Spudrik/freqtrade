from __future__ import annotations
import math
import pandas as pd
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from pandas import DataFrame, Series
from freqtrade.constants import NON_OPEN_EXCHANGE_STATES
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.strategy import BooleanParameter, CategoricalParameter, stoploss_from_absolute, IntParameter
import numpy as np
from freqtrade.strategy import IStrategy
from user_data.Indicators.market_state import add_market_state
from user_data.Indicators.complex_trendline_projection_v2 import add_trendline_projection_v2
from user_data.Indicators.complex_volume_profile import add_volume_profile
from user_data.Indicators.pattern_bos_choch import add_bos_choch
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2
from user_data.Indicators.pattern_reversal import add_pattern_reversal
SIEVE2_VP_GUARD_MODES = [
    "score_or_context",
    "node_confirm",
    "value_area_confirm",
    "breakout_acceptance",
    "rejection_confirm",
    "poc_hvn_reject",
    "prior_level_confirm",
]
SIEVE2_MARKET_GUARD_MODES = [
    "pressure_or_trend",
    "pressure_and_trend",
    "directional_pressure",
    "trend_state",
    "avoid_adverse_pressure",
    "avoid_chop",
]


def add_sieve2_guard_indicators(strategy: Any, dataframe: pd.DataFrame, metadata: dict | None) -> pd.DataFrame:
    frame = dataframe.copy()
    frame = add_market_state(frame, window=int(_sieve2_param(strategy, "sieve2_market_window", 24)), prefix="s2m")
    frame = add_volume_profile(
        frame,
        window=int(_sieve2_param(strategy, "sieve2_vp_window", 96)),
        bins=int(_sieve2_param(strategy, "sieve2_vp_bins", 36)),
        value_area_pct=0.70,
        price_source="hlc3",
        smooth_bins=3,
        pressure_delta_min=0.05,
        node_near_pct=0.01,
        prefix="s2vp",
    )
    return frame


def apply_sieve2_optional_guards(
    strategy: Any, dataframe: pd.DataFrame, condition: pd.Series, side: str
) -> pd.Series:
    guarded = pd.Series(condition, index=dataframe.index).fillna(False).astype(bool)
    normalized_side = "short" if str(side or "").lower() == "short" else "long"
    always_on = _sieve2_enabled(getattr(strategy, "SIEVE2_ALWAYS_ON_GUARDS", False))
    if always_on or _sieve2_enabled(_sieve2_param(strategy, "use_sieve2_vp_guard", False)):
        guarded &= _sieve2_vp_guard(
            dataframe,
            normalized_side,
            str(_sieve2_param(strategy, "sieve2_vp_guard_mode", "score_or_context")),
            float(_sieve2_param(strategy, "sieve2_vp_score_min", 0.25)),
            float(_sieve2_param(strategy, "sieve2_vp_context_min", 0.28)),
        )
    if always_on or _sieve2_enabled(_sieve2_param(strategy, "use_sieve2_market_guard", False)):
        guarded &= _sieve2_market_guard(
            dataframe,
            normalized_side,
            str(_sieve2_param(strategy, "sieve2_market_guard_mode", "pressure_or_trend")),
            float(_sieve2_param(strategy, "sieve2_market_pressure_min", 0.07)),
            float(_sieve2_param(strategy, "sieve2_market_trend_min", 0.25)),
            float(_sieve2_param(strategy, "sieve2_rs_score_min", 0.45)),
        )
    return guarded.fillna(False)


def _sieve2_vp_guard(
    frame: pd.DataFrame, side: str, mode: str, score_min: float, context_min: float
) -> pd.Series:
    close = _sieve2_num(frame, "close")
    high = _sieve2_num(frame, "high")
    low = _sieve2_num(frame, "low")
    score = _sieve2_num(frame, f"s2vp_score_{side}")
    other = _sieve2_num(frame, "s2vp_score_short" if side == "long" else "s2vp_score_long")
    context = _sieve2_num(
        frame, "s2vp_context_score_bull" if side == "long" else "s2vp_context_score_bear"
    )
    in_value = _sieve2_bool(frame, "s2vp_in_value_area")
    above_value = _sieve2_bool(frame, "s2vp_above_value_area")
    below_value = _sieve2_bool(frame, "s2vp_below_value_area")
    node_entry = _sieve2_bool(frame, f"s2vp_node_entry_{side}")
    node_hold = _sieve2_bool(frame, f"s2vp_node_hold_{side}")
    prior_vah = _sieve2_num(frame, "s2vp_prior_vah")
    prior_val = _sieve2_num(frame, "s2vp_prior_val")
    base_score = score.ge(score_min) & score.gt(other)
    base_context = context.ge(context_min)
    if mode == "node_confirm":
        return (node_entry | node_hold | (base_score & base_context)).fillna(False)
    if mode == "value_area_confirm":
        value_side = above_value if side == "long" else below_value
        return (value_side | (in_value & base_context)).fillna(False)
    if mode == "breakout_acceptance":
        accepted_beyond_value = close.gt(prior_vah) if side == "long" else close.lt(prior_val)
        return (accepted_beyond_value & (base_score | base_context | node_entry | node_hold)).fillna(False)
    if mode == "rejection_confirm":
        rejection = (
            (low.le(prior_val) & close.gt(prior_val) & close.le(prior_vah))
            if side == "long"
            else (high.ge(prior_vah) & close.lt(prior_vah) & close.ge(prior_val))
        )
        return (rejection & (base_score | base_context | node_entry | node_hold)).fillna(False)
    if mode == "poc_hvn_reject":
        return (node_entry | (base_score & base_context)).fillna(False)
    if mode == "prior_level_confirm":
        level_confirm = close.gt(prior_vah) if side == "long" else close.lt(prior_val)
        return (level_confirm & (base_score | base_context)).fillna(False)
    return (base_score | base_context).fillna(False)


def _sieve2_market_guard(
    frame: pd.DataFrame,
    side: str,
    mode: str,
    pressure_min: float,
    trend_min: float,
    rs_score_min: float,
) -> pd.Series:
    pressure = _sieve2_num(frame, "s2m_pressure_ratio")
    trend = _sieve2_num(frame, "s2m_trend_z")
    directional_pressure = pressure.ge(pressure_min) if side == "long" else pressure.le(-pressure_min)
    trend_state = trend.ge(trend_min) if side == "long" else trend.le(-trend_min)
    avoids_adverse_pressure = pressure.ge(-pressure_min) if side == "long" else pressure.le(pressure_min)
    avoids_adverse_trend = trend.ge(-trend_min) if side == "long" else trend.le(trend_min)
    if mode == "pressure_and_trend":
        return (directional_pressure & trend_state).fillna(False)
    if mode == "directional_pressure":
        return directional_pressure.fillna(False)
    if mode == "trend_state":
        return trend_state.fillna(False)
    if mode == "avoid_adverse_pressure":
        return (avoids_adverse_pressure & avoids_adverse_trend).fillna(False)
    if mode == "avoid_chop":
        return (pressure.abs().ge(pressure_min) | trend.abs().ge(trend_min)).fillna(False)
    return (directional_pressure | trend_state).fillna(False)


def _sieve2_param(strategy: Any, name: str, default: Any) -> Any:
    return getattr(strategy, name, default)


def _sieve2_enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _sieve2_num(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _sieve2_bool(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)
NOVEL_IDEA = True
ENTRY_MODE = 'entry_sieve2_mtfx_d1_tlv2_res_break_long_4h_bos'
ENTRY_TAG = 'sieve2_mtfx_d1_tlv2_res_break_long_4h_bos'
ENTRY_SOURCE_STAGE = "sieve2"
SIDE = 'long'
TIMEFRAME = '4h'
CONTEXT_TIMEFRAME = '1d'
HTF_CONCEPT = 'tlv2_res_break_long'
LTF_TRIGGER = 'bos'


def _num(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    return pd.Series(frame[column], index=frame.index).astype('boolean').fillna(False).astype(bool)


def _cross_above(value: Series, level: Series) -> Series:
    return value.gt(level) & value.shift(1).le(level.shift(1))

def _cross_below(value: Series, level: Series) -> Series:
    return value.lt(level) & value.shift(1).ge(level.shift(1))
LOCKED_BUY_SOURCE = 'unanimous current effective defaults; promotion lock unverified'
LOCKED_BUY_DEFAULTS_SHA256 = '721625dc8e6694ce13356b8e75932a755af67335287749f12d162512a6154bb9'
SOURCE_ENTRY_SIGNATURE_SHA256 = '422e9c83d1f65e9952375da495429ec321ad642487e5a7d48185b9eed9c708ad'
SECONDARY_GUARD = 'locked Sieve2 VP and market guards; 4h volume and pressure guards are off'
SIEVE_STAGE = "sieve3"
SOURCE_ENTRY_STAGE = 'sieve2'
SOURCE_ENTRY_CLASS = 'Sieve3ExitFixedTpSlFromMtfxD1Tlv2ResBreakLong4HBos'
SOURCE_STRATEGY = 'user_data/strategies/sieve3_exit_fixed_tp_sl_from_mtfx_d1_tlv2_res_break_long_4h_bos.py:Sieve3ExitFixedTpSlFromMtfxD1Tlv2ResBreakLong4HBos'
LINEAGE_STATUS = 'sieve2_baseline_hint_only_current_surface_unverified'
EXIT_FAMILY = 'guard_trigger_reduce_then_trail'
EXIT_THEORY = 'tlv2_context_bos_reduce_then_trail'
EXIT_HYPOTHESIS = 'Loss of closed-D1 TLV2 rank-0 resistance score/context or weakening of 4h bullish BOS near the closed-D1 TLV2 rank-0 resistance break reduces once and trails the remainder; the paired opposite BOS invalidates fully.'
PRIMARY_TRIGGER = '4h bullish BOS near the closed-D1 TLV2 rank-0 resistance break'
PRIMARY_GUARD = 'closed-D1 TLV2 rank-0 resistance score/context'
TARGET_PROVIDER = 'none'
INVALIDATION_PROVIDER = 'paired opposite structure invalidation'
ACTIVE_SELL_PARAMS = ('exit_plan', 'breakeven_after_partial', 'hard_stop_percent', 'max_hold_scale')

RESEARCH_PATH = "sieve3_exit_guard_trigger_tlv2_bos_reduce_then_trail"
SOURCE_RESULT_BATCH = "20260612T071622_entry_sieve2_top_level_clear100_cores8_20260612_01_retry01"

class Sieve3V2GuardTriggerTlv2BosReduceThenTrailFromMtfxD1Tlv2ResBreakLong4hBos(IStrategy):
    SIEVE2_FUNDAMENTAL_REWORK = '20260605_nonprofitable_loosen_sparse'
    REWORK_HYPOTHESIS = 'Entry-only rerun after non-profitable revised-guard result: loosen_sparse; widen guard thresholds and adjust entry-strength search space.'
    SIEVE2_ALWAYS_ON_GUARDS = True
    'Sieve2 structural MTF expansion: HTF concept plus LTF structural execution.'
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 240
    process_only_new_candles = True
    can_short = False
    max_entry_position_adjustment = 0
    use_sieve2_vp_guard = True
    sieve2_vp_guard_mode = 'score_or_context'
    sieve2_vp_window = 96
    sieve2_vp_bins = 36
    sieve2_vp_score_min = 0.15
    sieve2_vp_context_min = 0.15
    use_sieve2_market_guard = True
    sieve2_market_guard_mode = 'pressure_or_trend'
    sieve2_market_window = 24
    sieve2_market_pressure_min = 0.03
    sieve2_market_trend_min = 0.15
    # Legacy inactive entry parameter: sieve2_rs_benchmark_pair='BTC/USDT:USDT';
    # the localized market guard never reads a benchmark pair.
    sieve2_rs_score_min = 0.3
    score_min = 0.72
    htf_recent_bars = 2
    level_buffer_pct = 0.002
    local_window = 12
    # Legacy fixed-off entry parameters: use_ltf_volume_guard=False, volume_ratio_min=1.0,
    # use_ltf_pressure_guard=False, and pressure_min=0.25; both LTF guard branches are unreachable.

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
        dataframe = add_sieve2_guard_indicators(self, dataframe, metadata)
        dataframe['mtfx_exit_guard_opposed'] = ~_bool(dataframe, 'mtfx_htf_context')
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
        if not self.dp:
            return frame
        pair = metadata.get('pair')
        if not pair:
            return frame
        informative = self.dp.get_pair_dataframe(pair=pair, timeframe=CONTEXT_TIMEFRAME)
        if informative is None or informative.empty or 'date' not in informative.columns:
            return frame
        informative = self._add_htf_features(informative.copy())
        informative = informative[['date', 'mtfx_htf_context', 'mtfx_htf_level']].copy().sort_values('date')
        informative['date_merge'] = pd.to_datetime(informative['date'], utc=True).dt.tz_convert(None) + pd.to_timedelta(timeframe_to_minutes(CONTEXT_TIMEFRAME), unit='m')
        base = frame.drop(columns=['mtfx_htf_context', 'mtfx_htf_level'], errors='ignore').reset_index().rename(columns={'index': '__row_index'}).sort_values('date')
        base['__date_merge'] = pd.to_datetime(base['date'], utc=True).dt.tz_convert(None)
        merged = pd.merge_asof(base, informative.drop(columns=['date']).sort_values('date_merge'), left_on='__date_merge', right_on='date_merge', direction='backward')
        merged = merged.set_index('__row_index').reindex(frame.index)
        frame['mtfx_htf_context'] = pd.Series(merged['mtfx_htf_context'], index=frame.index).astype('boolean').fillna(False).astype(bool)
        frame['mtfx_htf_level'] = pd.to_numeric(merged['mtfx_htf_level'], errors='coerce')
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
            level = _num(dataframe, f'{prefix}_confirmation_level')
            context = _bool(dataframe, f'{prefix}_pattern_confirmed') & _num(dataframe, f'{prefix}_indicator_score').ge(score_min)
            context &= close.le(level.mul(1.0 + near)) if direction == 'short' else close.ge(level.mul(1.0 - near))
            return (context, level)
        if concept in {'rectangle_long', 'rectangle_short'}:
            level = _num(dataframe, 'pg2_rectangle_upper') if SIDE == 'long' else _num(dataframe, 'pg2_rectangle_lower')
            context = _bool(dataframe, 'pg2_rectangle_pattern_present') & _num(dataframe, 'pg2_rectangle_indicator_score').ge(score_min)
            return (context, level)
        if concept in {'wedge_long', 'wedge_short'}:
            level = _num(dataframe, 'pg2_wedge_upper') if SIDE == 'long' else _num(dataframe, 'pg2_wedge_lower')
            direction_ok = _num(dataframe, 'pg2_wedge_direction').ge(0) if SIDE == 'long' else _num(dataframe, 'pg2_wedge_direction').le(0)
            context = _bool(dataframe, 'pg2_wedge_pattern_present') & _bool(dataframe, 'pg2_wedge_squeeze_active') & direction_ok & _num(dataframe, 'pg2_wedge_indicator_score').ge(score_min)
            return (context, level)
        if concept == 'tlv2_res_break_long':
            level = _num(dataframe, 'tlv2_resistance_line_rank0')
            return (_num(dataframe, 'tlv2_resistance_score_rank0').ge(score_min) & close.ge(level.mul(1.0 - near)), level)
        if concept == 'tlv2_sup_reclaim_long':
            level = _num(dataframe, 'tlv2_support_line_rank0')
            return (_num(dataframe, 'tlv2_support_score_rank0').ge(score_min) & close.ge(level.mul(1.0 - near)), level)
        if concept == 'tlv2_sup_break_short':
            level = _num(dataframe, 'tlv2_support_line_rank0')
            return (_num(dataframe, 'tlv2_support_score_rank0').ge(score_min) & close.le(level.mul(1.0 + near)), level)
        if concept == 'vp_poc_reclaim_long':
            level = _num(dataframe, 'htfvp_prior_poc')
            return (_num(dataframe, 'htfvp_context_score_bull').ge(0.2) & close.ge(level.mul(1.0 - near)), level)
        if concept == 'vp_vah_reject_short':
            level = _num(dataframe, 'htfvp_prior_vah')
            return (_num(dataframe, 'htfvp_context_score_bear').ge(0.2) & close.le(level.mul(1.0 + near)), level)
        if concept == 'vp_val_reclaim_long':
            level = _num(dataframe, 'htfvp_prior_val')
            return (_num(dataframe, 'htfvp_context_score_bull').ge(0.2) & close.ge(level.mul(1.0 - near)), level)
        if concept == 'vp_poc_reject_short':
            level = _num(dataframe, 'htfvp_prior_poc')
            return (_num(dataframe, 'htfvp_context_score_bear').ge(0.2) & close.le(level.mul(1.0 + near)), level)
        if concept == 'vp_lvn_traverse_long':
            level = _num(dataframe, 'htfvp_lvn_above')
            return (level.notna(), level)
        if concept == 'vp_lvn_traverse_short':
            level = _num(dataframe, 'htfvp_lvn_below')
            return (level.notna(), level)
        if concept == 'liq_equal_lows_long':
            level = _num(dataframe, 'mtfx_equal_low')
            return (low.le(level.mul(1.0 - near)) & close.gt(level), level)
        if concept == 'liq_equal_highs_short':
            level = _num(dataframe, 'mtfx_equal_high')
            return (high.ge(level.mul(1.0 + near)) & close.lt(level), level)
        if concept == 'liq_prior_low_long':
            level = _num(dataframe, 'mtfx_prior_low')
            return (low.le(level.mul(1.0 - near)) & close.gt(level), level)
        if concept == 'liq_prior_high_short':
            level = _num(dataframe, 'mtfx_prior_high')
            return (high.ge(level.mul(1.0 + near)) & close.lt(level), level)
        if concept == 'demand_zone_long':
            level = _num(dataframe, 'mtfx_demand_zone_high')
            return (close.ge(level.mul(1.0 - near)), level)
        if concept == 'supply_zone_short':
            level = _num(dataframe, 'mtfx_supply_zone_low')
            return (close.le(level.mul(1.0 + near)), level)
        if concept == 'avwap_reclaim_long':
            level = _num(dataframe, 'mtfx_avwap_from_low')
            return (close.ge(level.mul(1.0 - near)), level)
        if concept == 'avwap_reject_short':
            level = _num(dataframe, 'mtfx_avwap_from_high')
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
        level = _num(dataframe, 'mtfx_htf_level')
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
        condition = apply_sieve2_optional_guards(self, dataframe, condition, SIDE)
        valid = condition.fillna(False) & dataframe['volume'].gt(0.0) & dataframe['close'].notna()
        if SIDE == 'short':
            dataframe.loc[valid, 'enter_short'] = 1
        else:
            dataframe.loc[valid, 'enter_long'] = 1
        dataframe.loc[valid, 'enter_tag'] = ENTRY_TAG
        return dataframe
    SOURCE_ENTRY_STEM = 'mtfx_d1_tlv2_res_break_long_4h_bos'
    FOCUSED_EXIT_CONTRACT = 'guard_trigger_reduce_then_trail'
    FOCUSED_SOURCE_PROFILE = {'side': 'long', 'min_level_distance': 0.001, 'guard': {'provider': 'closed-D1 TLV2 rank-0 resistance score/context', 'long': {'aligned': 'mtfx_htf_context', 'opposed': 'mtfx_exit_guard_opposed'}}, 'trigger': {'provider': '4h bullish BOS near the closed-D1 TLV2 rank-0 resistance break', 'long': {'intact': None, 'weakening': 'ltfms_choch_to_bear', 'invalidated': 'ltfms_bos_to_bear'}}}
    FOCUSED_EXIT_PLANS = {'guard1_trigger1_reduce25_trail1': {'contract': 'guard_trigger_reduce_then_trail', 'name': 'guard1_trigger1_reduce25_trail1', 'action_sequence': ('trigger_invalidation_full', 'guard_or_trigger_weakening', 'partial_stage_1', 'percent_trail_remainder', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'partial_fractions': (0.25,), 'guard_confirmations': 1, 'trigger_confirmations': 1, 'trail_ratio': 0.01}, 'guard2_trigger1_reduce33_trail1_5': {'contract': 'guard_trigger_reduce_then_trail', 'name': 'guard2_trigger1_reduce33_trail1_5', 'action_sequence': ('trigger_invalidation_full', 'guard_or_trigger_weakening', 'partial_stage_1', 'percent_trail_remainder', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'partial_fractions': (0.33,), 'guard_confirmations': 2, 'trigger_confirmations': 1, 'trail_ratio': 0.015}, 'guard2_trigger2_reduce50_trail2': {'contract': 'guard_trigger_reduce_then_trail', 'name': 'guard2_trigger2_reduce50_trail2', 'action_sequence': ('trigger_invalidation_full', 'guard_or_trigger_weakening', 'partial_stage_1', 'percent_trail_remainder', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'partial_fractions': (0.5,), 'guard_confirmations': 2, 'trigger_confirmations': 2, 'trail_ratio': 0.02}}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'high', 'low', 'ltfms_bos_to_bear', 'ltfms_choch_to_bear', 'mtfx_exit_guard_opposed', 'mtfx_htf_context', 'open')
    FOCUSED_STATE_VERSION = 1
    FOCUSED_STATE_KEY = 'sieve3_v2_focused:Sieve3V2GuardTriggerTlv2BosReduceThenTrailFromMtfxD1Tlv2ResBreakLong4hBos:guard_trigger_reduce_then_trail'
    LOCKED_BUY_PARAMS = {'use_sieve2_vp_guard': True, 'sieve2_vp_guard_mode': 'score_or_context', 'sieve2_vp_window': 96, 'sieve2_vp_bins': 36, 'sieve2_vp_score_min': 0.15, 'sieve2_vp_context_min': 0.15, 'use_sieve2_market_guard': True, 'sieve2_market_guard_mode': 'pressure_or_trend', 'sieve2_market_window': 24, 'sieve2_market_pressure_min': 0.03, 'sieve2_market_trend_min': 0.15, 'sieve2_rs_benchmark_pair': 'BTC/USDT:USDT', 'sieve2_rs_score_min': 0.3, 'score_min': 0.72, 'htf_recent_bars': 2, 'level_buffer_pct': 0.002, 'local_window': 12, 'use_ltf_volume_guard': False, 'volume_ratio_min': 1.0, 'use_ltf_pressure_guard': False, 'pressure_min': 0.25}
    ACTIVE_SELL_PARAMS = ('exit_plan', 'breakeven_after_partial', 'hard_stop_percent', 'max_hold_scale')
    position_adjustment_enable = True
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    minimal_roi = {'0': 100.0}
    stoploss = -0.99
    exit_plan = CategoricalParameter(('guard1_trigger1_reduce25_trail1', 'guard2_trigger1_reduce33_trail1_5', 'guard2_trigger2_reduce50_trail2'), default='guard1_trigger1_reduce25_trail1', space='sell', optimize=True, load=True)
    exit_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    hard_stop_percent = IntParameter(2, 6, default=3, space='sell', optimize=True, load=True)
    hard_stop_percent.batch_tags = ('family:exits', 'mode:sieve3_exit')
    max_hold_scale = IntParameter(1, 4, default=2, space='sell', optimize=True, load=True)
    max_hold_scale.batch_tags = ('family:exits', 'mode:sieve3_exit')
    breakeven_after_partial = BooleanParameter(default=False, space='sell', optimize=True, load=True)
    breakeven_after_partial.batch_tags = ('family:exits', 'mode:sieve3_exit')

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

    @staticmethod
    def _focused_boolean(frame: DataFrame, column: str) -> Series:
        return frame[column].fillna(False).astype(bool)

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
        if getattr(self, 'dp', None) is None:
            raise RuntimeError("focused exit runtime requires Freqtrade's data provider")
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        return self._focused_closed_frame(frame, current_time)

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
        plan['hard_stop_ratio'] = float(self.hard_stop_percent.value) * 0.01
        plan['max_hold_candles'] = max(1, int(round(float(plan.get('max_hold_candles', 336)) * float(self.max_hold_scale.value) / 2.0)))
        return plan

    @staticmethod
    def _focused_decision(action: str='hold', tag: str | None=None, stage: str | None=None, fraction: float=0.0) -> dict[str, Any]:
        return {'action': action, 'tag': tag, 'stage': stage, 'fraction': float(fraction)}

    def _focused_state(self, trade: Any) -> dict[str, Any] | None:
        state = trade.get_custom_data(key=self.FOCUSED_STATE_KEY)
        if state is None:
            return None
        if not isinstance(state, dict):
            raise ValueError('focused exit trade state must be a mapping')
        if state.get('version') != self.FOCUSED_STATE_VERSION:
            raise ValueError('focused exit trade state version mismatch')
        if state.get('contract') != self.FOCUSED_EXIT_CONTRACT:
            raise ValueError('focused exit trade state contract mismatch')
        return dict(state)

    def _focused_save_state(self, trade: Any, state: Mapping[str, Any]) -> None:
        trade.set_custom_data(key=self.FOCUSED_STATE_KEY, value=dict(state))

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
        frame = self._focused_analyzed_frame(pair, freeze_time)
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
        stages = {
            f'stage_{index + 1}': {
                'status': 'ready',
                'fraction': float(fraction),
                'tag': f'focused_partial_stage_{index + 1}',
                'target_stake': None,
                'requested_stake': None,
                'credited_stake': 0.0,
                'credited_order_stakes': {},
                'requested_at': None,
                'filled_at': None,
                'order_id': None,
                'resolved_order_ids': [],
            }
            for index, fraction in enumerate(plan.get('partial_fractions', ()))
        }
        return {'version': self.FOCUSED_STATE_VERSION, 'contract': self.FOCUSED_EXIT_CONTRACT, 'plan': selected_plan, 'side': side, 'phase': 'pre_target', 'entry_rate': float(entry_rate), 'entry_filled_at': fill_timestamp.isoformat() if fill_timestamp is not None else None, 'entry_snapshot_candle': snapshot_date.isoformat() if snapshot_date is not None else None, 'levels': levels, 'target_states': {slot: {'status': 'available' if value is not None else 'unavailable', 'touched_at': None} for slot, value in levels.items() if slot.startswith('target_')}, 'invalidation_seen': False, 'guard_state': 'unknown', 'trigger_state': 'unknown', 'stages': stages, 'initial_stake': self._focused_float(getattr(trade, 'stake_amount', None)), 'favorable_rate': float(entry_rate), 'stop_price': None, 'terminal_pending': False, 'terminal_tag': None}

    def _focused_ensure_state(self, pair: str, trade: Any, current_time: Any, order: Any | None=None) -> dict[str, Any] | None:
        state = self._focused_state(trade)
        if state is None:
            state = self._focused_new_state(pair, trade, current_time, order)
            if state is None:
                return None
            self._focused_save_state(trade, state)
        return state

    def _focused_post_entry(
        self,
        frame: DataFrame,
        state: Mapping[str, Any],
        current_time: Any,
    ) -> DataFrame:
        filled_at = self._focused_utc(state.get("entry_filled_at"))
        if filled_at is None:
            raise ValueError("focused exit state has no entry fill timestamp")
        candle_opens = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        close_times = self._focused_close_times(frame)
        now = self._focused_utc(current_time)
        if now is None:
            raise ValueError("focused exit runtime requires current_time")
        eligible = candle_opens.ge(filled_at) & close_times.le(now)
        return frame.loc[eligible].copy()


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
    def _focused_last_n(condition: Series, count: int) -> bool:
        needed = max(1, int(count))
        return len(condition) >= needed and bool(condition.tail(needed).fillna(False).all())

    @staticmethod
    def _focused_distinct_periods(frame: DataFrame, condition: Series, timeframe: str) -> Series:
        minutes = int(timeframe_to_minutes(timeframe))
        if minutes <= 0:
            raise ValueError('invalidation evidence timeframe must be at least one minute')
        dates = pd.to_datetime(frame['date'], utc=True, errors='raise')
        period_ids = dates.map(lambda value: timeframe_to_prev_date(timeframe, value.to_pydatetime()))
        return condition.fillna(False).groupby(period_ids, sort=True).any()

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


    @staticmethod
    def _focused_trail_price(state: Mapping[str, Any], ratio: float) -> float:
        favorable = float(state.get('favorable_rate') or state['entry_rate'])
        return favorable * (1.0 + ratio if state['side'] == 'short' else 1.0 - ratio)

    def _focused_context(self, pair: str, trade: Any, current_time: Any, current_rate: float, current_profit: float) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        state = self._focused_ensure_state(pair, trade, current_time)
        if state is None:
            return (self._focused_plan(), {}, {}, self._focused_decision())
        before = repr(state)
        if state.get('stages'):
            self._focused_reconcile_partial_orders(trade, state, current_time)
        plan = self._focused_plan(state)
        frame = self._focused_analyzed_frame(pair, current_time)
        self._focused_require_columns(frame)
        post_entry = self._focused_post_entry(frame, state, current_time)
        self._focused_update_favorable(post_entry, state)
        events = self._focused_contract_events(post_entry, state, plan, current_profit)
        events['age_candles'] = len(post_entry)
        events['favorable_move'] = self._focused_favorable_move(state)
        if self._focused_hard_stop_breached(plan, state, float(current_rate)):
            decision = self._focused_decision('full', 'focused_hard_stop')
        else:
            decision = self._focused_contract_decision(state, plan, events, current_profit)
        if decision['action'] == 'hold' and events['age_candles'] >= int(plan['max_hold_candles']):
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
        _ = (after_fill, kwargs)
        plan, state, _, decision = self._focused_context(pair, trade, current_time, current_rate, current_profit)
        if not state:
            return None
        if decision['action'] == 'full' or self._focused_requested_stage(state) is not None:
            return None
        desired = self._focused_stop_overlay(plan, state, current_profit)
        persisted = self._focused_float(state.get('stop_price'))
        stop_price = desired if persisted is None else self._focused_tighter_stop(str(state['side']), persisted, desired)
        if self.breakeven_after_partial.value and self._focused_stage_filled(state, 'stage_1'):
            stop_price = self._focused_tighter_stop(str(state['side']), stop_price, float(trade.open_rate))
        if state['side'] == 'short' and stop_price <= current_rate or (state['side'] == 'long' and stop_price >= current_rate):
            return None
        if persisted is None or not math.isclose(stop_price, persisted, rel_tol=0.0, abs_tol=1e-12):
            state['stop_price'] = stop_price
            self._focused_save_state(trade, state)
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=state['side'] == 'short', leverage=float(getattr(trade, 'leverage', 1.0) or 1.0))

    def _focused_guard_trigger_events(self, post_entry: DataFrame, state: dict[str, Any], plan: Mapping[str, Any]) -> dict[str, bool]:
        if post_entry.empty:
            return {'trigger_invalidated': False, 'trigger_weakening': False, 'guard_opposed': False}
        side = str(state['side'])
        guard = self.FOCUSED_SOURCE_PROFILE['guard'][side]
        trigger = self.FOCUSED_SOURCE_PROFILE['trigger'][side]
        guard_count = int(plan.get('guard_confirmations') or 1)
        trigger_count = int(plan.get('trigger_confirmations') or 1)
        opposed = self._focused_last_n(self._focused_boolean(post_entry, str(guard['opposed'])), guard_count)
        weakening = self._focused_last_n(self._focused_boolean(post_entry, str(trigger['weakening'])), trigger_count)
        invalidated = self._focused_last_n(self._focused_boolean(post_entry, str(trigger['invalidated'])), trigger_count)
        aligned_column = guard.get('aligned')
        aligned = bool(aligned_column and self._focused_last_n(self._focused_boolean(post_entry, str(aligned_column)), guard_count))
        state['guard_state'] = 'opposed' if opposed else 'aligned' if aligned else 'neutral'
        state['trigger_state'] = 'invalidated' if invalidated else 'weakening' if weakening else 'intact'
        return {'trigger_invalidated': invalidated, 'trigger_weakening': weakening, 'guard_opposed': opposed}

    def _focused_contract_events(self, post_entry: DataFrame, state: dict[str, Any], plan: Mapping[str, Any], current_profit: float) -> dict[str, Any]:
        _ = current_profit
        return self._focused_guard_trigger_events(post_entry, state, plan)

    def _focused_contract_decision(self, state: dict[str, Any], plan: Mapping[str, Any], events: Mapping[str, Any], current_profit: float) -> dict[str, Any]:
        _ = (plan, current_profit)
        if events.get('trigger_invalidated'):
            return self._focused_decision('full', 'focused_trigger_invalidation')
        stage = state['stages']['stage_1']
        if stage['status'] == 'ready' and (events.get('guard_opposed') or events.get('trigger_weakening')):
            return self._focused_decision('partial', str(stage['tag']), 'stage_1', float(stage['fraction']))
        return self._focused_decision()

    def _focused_stop_overlay(self, plan: Mapping[str, Any], state: Mapping[str, Any], current_profit: float) -> float:
        stop_price = self._focused_hard_stop_price(plan, state)
        if self._focused_stage_filled(state, 'stage_1'):
            candidate = self._focused_trail_price(state, float(plan['trail_ratio']))
            stop_price = self._focused_tighter_stop(str(state['side']), stop_price, candidate)
        return stop_price

    def order_filled(self, pair: str, trade: Any, order: Any, current_time: datetime, **kwargs: Any) -> None:
        _ = kwargs
        if getattr(order, 'ft_order_side', None) == getattr(trade, 'entry_side', None) and self._focused_state(trade) is None:
            state = self._focused_new_state(pair, trade, current_time, order)
            if state is None:
                return None
            self._focused_save_state(trade, state)
            return None
        if getattr(order, 'ft_order_side', None) != getattr(trade, 'exit_side', None):
            return None
        tag = str(getattr(order, 'ft_order_tag', None) or '')
        state = self._focused_state(trade)
        if state is None:
            return None
        matched = next(((name, stage) for name, stage in state.get('stages', {}).items() if isinstance(stage, Mapping) and stage.get('tag') == tag), None)
        if matched is None:
            return None
        name, stage = matched
        if stage.get('status') == 'filled':
            return None
        before = repr(state)
        self._focused_apply_partial_order(trade, state, str(name), stage, order, current_time)
        if repr(state) != before:
            self._focused_save_state(trade, state)
        return None

    @staticmethod
    def _focused_order_snapshot(order: Any) -> tuple[str, bool, float]:
        status = str(getattr(order, 'status', None) or '').casefold()
        is_open = bool(getattr(order, 'ft_is_open', False)) and status not in NON_OPEN_EXCHANGE_STATES
        filled = float(getattr(order, 'safe_filled', 0.0) or 0.0)
        return (status, is_open, filled)

    @staticmethod
    def _focused_add_resolved_order(stage: dict[str, Any], order: Any) -> None:
        order_id = str(getattr(order, 'order_id', None) or '')
        if not order_id:
            return
        resolved = stage.setdefault('resolved_order_ids', [])
        if order_id not in resolved:
            resolved.append(order_id)

    def _focused_reset_stage_for_retry(self, state: dict[str, Any], name: str, stage: dict[str, Any], orders: tuple[Any, ...]=()) -> None:
        for order in orders:
            self._focused_add_resolved_order(stage, order)
        active_order_id = str(stage.get('order_id') or '')
        resolved = stage.setdefault('resolved_order_ids', [])
        if active_order_id and active_order_id not in resolved:
            resolved.append(active_order_id)
        stage['status'] = 'ready'
        stage['requested_at'] = None
        stage['requested_stake'] = None
        stage['filled_at'] = None
        stage['order_id'] = None
        state['phase'] = f'{name}_retry'

    def _focused_apply_partial_order(self, trade: Any, state: dict[str, Any], name: str, stage: dict[str, Any], order: Any, current_time: Any) -> None:
        _, is_open, filled = self._focused_order_snapshot(order)
        order_id = str(getattr(order, 'order_id', None) or '')
        credit_key = order_id or '|'.join(
            str(value or '')
            for value in (
                getattr(order, 'ft_order_tag', None),
                getattr(order, 'order_date_utc', None) or getattr(order, 'order_date', None),
                getattr(order, 'safe_amount', None),
                getattr(order, 'safe_price', None),
            )
        )
        fill_price = float(getattr(order, 'safe_price', 0.0) or 0.0)
        leverage = float(getattr(trade, 'leverage', 1.0) or 1.0)
        credited_for_order = filled * fill_price / leverage if fill_price > 0.0 else 0.0
        credits = stage.setdefault('credited_order_stakes', {})
        previous_credit = float(credits.get(credit_key) or 0.0)
        if credited_for_order > previous_credit:
            stage['credited_stake'] = float(stage.get('credited_stake') or 0.0) + credited_for_order - previous_credit
            credits[credit_key] = credited_for_order
        requested_amount = float(getattr(order, 'safe_amount', 0.0) or 0.0)
        fully_filled = requested_amount > 0.0 and filled >= requested_amount - max(1e-12, requested_amount * 1e-9)
        target_stake = self._focused_float(stage.get('target_stake'))
        credited_stake = float(stage.get('credited_stake') or 0.0)
        target_filled = target_stake is not None and credited_stake >= target_stake - 1e-8
        if is_open and not (fully_filled or target_filled):
            stage['status'] = 'requested'
            stage['order_id'] = order_id or None
            state['phase'] = f'{name}_pending'
            return
        if fully_filled or target_filled:
            stage['status'] = 'filled'
            stage['filled_at'] = self._focused_utc(getattr(order, 'order_filled_utc', None) or current_time).isoformat()
            stage['order_id'] = order_id or None
            state['phase'] = 'remainder' if name == 'stage_1' else 'runner'
            return
        self._focused_reset_stage_for_retry(state, name, stage, (order,))

    def _focused_reconcile_partial_orders(self, trade: Any, state: dict[str, Any], current_time: Any) -> None:
        orders = tuple(getattr(trade, 'orders', ()) or ())
        exit_side = getattr(trade, 'exit_side', None)
        now = self._focused_utc(current_time)
        for name, candidate in state.get('stages', {}).items():
            if not isinstance(candidate, dict) or candidate.get('status') != 'requested':
                continue
            stage = candidate
            tag = str(stage.get('tag') or '')
            resolved = {str(value) for value in stage.setdefault('resolved_order_ids', [])}
            matching = [order for order in orders if getattr(order, 'ft_order_side', None) == exit_side and str(getattr(order, 'ft_order_tag', None) or '') == tag and (str(getattr(order, 'order_id', None) or '') not in resolved)]
            filled = next((order for order in reversed(matching) if not self._focused_order_snapshot(order)[1] and self._focused_order_snapshot(order)[2] > 0.0), None)
            if filled is not None:
                self._focused_apply_partial_order(trade, state, str(name), stage, filled, current_time)
                continue
            open_order = next((order for order in reversed(matching) if self._focused_order_snapshot(order)[1]), None)
            if open_order is not None:
                for order in matching:
                    if order is not open_order:
                        self._focused_add_resolved_order(stage, order)
                self._focused_apply_partial_order(trade, state, str(name), stage, open_order, current_time)
                continue
            if matching:
                self._focused_reset_stage_for_retry(state, str(name), stage, tuple(matching))
                continue
            requested_at = self._focused_utc(stage.get('requested_at'))
            if requested_at is not None and now is not None and (requested_at == now):
                continue
            self._focused_reset_stage_for_retry(state, str(name), stage)

    def adjust_trade_position(self, trade: Any, current_time: datetime, current_rate: float, current_profit: float, min_stake: float | None, max_stake: float, current_entry_rate: float, current_exit_rate: float, current_entry_profit: float, current_exit_profit: float, **kwargs: Any) -> float | None | tuple[float | None, str | None]:
        _ = (max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, kwargs)
        _, state, _, decision = self._focused_context(str(getattr(trade, 'pair', '')), trade, current_time, current_rate, current_profit)
        if not state:
            return None
        if bool(getattr(trade, 'open_orders', ())):
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
        credited_stake = float(stage.get('credited_stake') or 0.0)
        new_target = target_stake is None
        if new_target:
            request = min(current_stake, initial_stake * float(decision['fraction']))
        else:
            request = min(current_stake, max(0.0, target_stake - credited_stake))
        if min_stake is not None:
            minimum = float(min_stake)
            if request < minimum:
                return None
            if 0.0 < current_stake - request < minimum:
                request = current_stake - minimum
            if request < minimum:
                return None
        if request <= 0.0 or request >= current_stake:
            return None
        if new_target:
            stage['target_stake'] = request
        stage['status'] = 'requested'
        stage['requested_at'] = self._focused_utc(current_time).isoformat()
        stage['requested_stake'] = request
        stage['order_id'] = None
        state['phase'] = f'{stage_name}_pending'
        self._focused_save_state(trade, state)
        return (-request, str(stage['tag']))

from __future__ import annotations
import math
import pandas as pd
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from pandas import DataFrame, Series
from user_data.Indicators.market_state import add_market_state
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.strategy import CategoricalParameter, stoploss_from_absolute, IntParameter
ENTRY_SOURCE_STAGE = 'sieve2'
import numpy as np
from freqtrade.strategy import IStrategy
from user_data.Indicators.complex_volume_profile import add_volume_profile
from user_data.Indicators.pattern_bos_choch import add_bos_choch
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

def add_sieve2_guard_indicators(strategy: Any, dataframe: DataFrame, metadata: dict | None) -> DataFrame:
    frame = dataframe.copy()
    frame = add_market_state(frame, window=int(_param(strategy, "sieve2_market_window", 24)), prefix="s2m")
    frame = add_volume_profile(
        frame,
        window=int(_param(strategy, "sieve2_vp_window", 96)),
        bins=int(_param(strategy, "sieve2_vp_bins", 36)),
        value_area_pct=0.70,
        price_source="hlc3",
        smooth_bins=3,
        pressure_delta_min=0.05,
        node_near_pct=0.01,
        prefix="s2vp",
    )
    return frame

def apply_sieve2_optional_guards(strategy: Any, dataframe: DataFrame, condition: Series, side: str) -> Series:
    guarded = pd.Series(condition, index=dataframe.index).fillna(False).astype(bool)
    normalized_side = "short" if str(side or "").lower() == "short" else "long"
    always_on = _enabled(getattr(strategy, "SIEVE2_ALWAYS_ON_GUARDS", False))
    if always_on or _enabled(_param(strategy, "use_sieve2_vp_guard", False)):
        guarded &= _vp_guard(
            dataframe,
            normalized_side,
            str(_param(strategy, "sieve2_vp_guard_mode", "score_or_context")),
            float(_param(strategy, "sieve2_vp_score_min", 0.25)),
            float(_param(strategy, "sieve2_vp_context_min", 0.28)),
        )
    if always_on or _enabled(_param(strategy, "use_sieve2_market_guard", False)):
        guarded &= _market_guard(
            dataframe,
            normalized_side,
            str(_param(strategy, "sieve2_market_guard_mode", "pressure_or_trend")),
            float(_param(strategy, "sieve2_market_pressure_min", 0.07)),
            float(_param(strategy, "sieve2_market_trend_min", 0.25)),
            float(_param(strategy, "sieve2_rs_score_min", 0.45)),
        )
    return guarded.fillna(False)

def _vp_guard(frame: DataFrame, side: str, mode: str, score_min: float, context_min: float) -> Series:
    close = _num(frame, "close")
    high = _num(frame, "high")
    low = _num(frame, "low")
    score = _num(frame, f"s2vp_score_{side}")
    other = _num(frame, "s2vp_score_short" if side == "long" else "s2vp_score_long")
    context = _num(frame, "s2vp_context_score_bull" if side == "long" else "s2vp_context_score_bear")
    in_value = _bool(frame, "s2vp_in_value_area")
    above_value = _bool(frame, "s2vp_above_value_area")
    below_value = _bool(frame, "s2vp_below_value_area")
    node_entry = _bool(frame, f"s2vp_node_entry_{side}")
    node_hold = _bool(frame, f"s2vp_node_hold_{side}")
    prior_vah = _num(frame, "s2vp_prior_vah")
    prior_val = _num(frame, "s2vp_prior_val")
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

def _market_guard(
    frame: DataFrame,
    side: str,
    mode: str,
    pressure_min: float,
    trend_min: float,
    rs_score_min: float,
) -> Series:
    pressure = _num(frame, "s2m_pressure_ratio")
    trend = _num(frame, "s2m_trend_z")
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

def _param(strategy: Any, name: str, default: Any) -> Any:
    value = getattr(strategy, name, default)
    return getattr(value, "value", value)

def _enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
ENTRY_MODE = 'entry_overtrade_mtf_h4_supply_breakout_long_1h_local_break_vp_market_guard'
ENTRY_TAG = 'overtrade_mtf_h4_supply_breakout_long_1h_local_break_vp_market_guard'
UPDATE_HYPOTHESIS = 'High-trade entry may contain a real directional edge; require stronger volume-profile/market context so HTF or broader context carries confidence and the original trigger supplies execution.'
SIDE = 'long'
TIMEFRAME = '1h'
CONTEXT_TIMEFRAME = '4h'
HTF_BEHAVIOR = 'supply_break_long'
LTF_TRIGGER = 'local_break'


def _num(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        raise KeyError(f"required numeric column is missing: {column!r}")
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)

def _bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        raise KeyError(f"required boolean column is missing: {column!r}")
    return pd.Series(frame[column], index=frame.index).astype('boolean').fillna(False).astype(bool)


SIEVE_STAGE = 'sieve3'
SOURCE_RESULT_BATCH = 'unknown_not_recorded'
RESEARCH_PATH = 'sieve3_exit_source_invalidation'
SOURCE_ENTRY_CLASS = 'Sieve3ExitFromOvertradeMtfH4SupplyBreakoutLong1hLocalBreakVpMarketGuard'
SOURCE_STRATEGY = 'user_data/strategies/sieve3_exit_from_overtrade_mtf_h4_supply_breakout_long_1h_local_break_vp_market_guard.py:Sieve3ExitFromOvertradeMtfH4SupplyBreakoutLong1hLocalBreakVpMarketGuard'
LINEAGE_STATUS = 'verified'
EXIT_FAMILY = 'source_invalidation'
EXIT_THEORY = 'source_invalidation'
EXIT_HYPOTHESIS = 'A closed-candle loss of the frozen H4 supply-breakout boundary invalidates the long local-break thesis.'
PRIMARY_TRIGGER = 'none'
PRIMARY_GUARD = 'none'
TARGET_PROVIDER = 'none'
PRIMARY_TARGET = TARGET_PROVIDER
INVALIDATION_PROVIDER = 'provider=h4_supply_breakout_failure;mode=level;long.level=mtf_htf_level'
PRIMARY_INVALIDATION = INVALIDATION_PROVIDER
ACTIVE_SELL_PARAMS = ('invalidation_confirmations', 'invalidation_band_quarter_percent', 'hard_stop_percent', 'max_hold_scale')

class Sieve3V2SourceInvalidationFromOvertradeMtfH4SupplyBreakoutLong1hLocalBreakVpMarketGuard(IStrategy):
    SIEVE2_ALWAYS_ON_GUARDS = True
    'MTF Sieve2 probe: 4h supply break; 1h local breakout.'
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 420
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
    # Legacy inactive entry parameter: sieve2_rs_benchmark_pair='BTC/USDT:USDT' (fixed guard mode does not consume it).
    sieve2_rs_score_min = 0.3
    context_lookback = 48
    context_recent_bars = 5
    local_lookback = 12
    retest_buffer_pct = 0.004
    breakout_buffer_pct = 0.0
    # Legacy inactive entry parameter: volume_ratio_min=1.5 (its enable gate is fixed off).
    pressure_min = 0.03
    vp_score_min = 0.5
    vp_context_min = 0.18
    # Legacy inactive entry parameter: use_ltf_volume_guard=False (enable gate is fixed off).
    # Legacy inactive entry parameter: use_ltf_pressure_guard=False (enable gate is fixed off).
    # Legacy inactive entry parameter: use_htf_turn_guard=False (enable gate is fixed off).

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = (pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs)
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        if not getattr(self, 'dp', None):
            return []
        return [(pair, CONTEXT_TIMEFRAME) for pair in self.dp.current_whitelist()]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self._add_ltf_features(dataframe)
        dataframe = self._merge_htf_context(dataframe, metadata or {})
        dataframe = add_sieve2_guard_indicators(self, dataframe, metadata)
        return dataframe

    def _add_ltf_features(self, dataframe: DataFrame) -> DataFrame:
        frame = dataframe.copy()
        close = _num(frame, 'close')
        open_ = _num(frame, 'open')
        high = _num(frame, 'high')
        low = _num(frame, 'low')
        volume = _num(frame, 'volume').clip(lower=0.0).fillna(0.0)
        window = int(self.local_lookback)
        candle_range = (high - low).replace(0.0, np.nan)
        close_location = ((close - low) / candle_range * 2.0 - 1.0).clip(-1.0, 1.0)
        body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
        pressure = ((close_location.fillna(0.0) + body_pressure.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
        volume_baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
        frame['mtf_ltf_pressure'] = pressure
        frame['mtf_ltf_volume_ratio'] = volume / volume_baseline
        frame['mtf_ltf_ema'] = close.ewm(span=window, min_periods=max(2, window // 3), adjust=False).mean()
        frame['mtf_ltf_local_high'] = high.shift(1).rolling(window, min_periods=max(2, window // 3)).max()
        frame['mtf_ltf_local_low'] = low.shift(1).rolling(window, min_periods=max(2, window // 3)).min()
        return frame

    def _merge_htf_context(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        frame = dataframe.copy()
        for column, default in (('mtf_htf_context', False), ('mtf_htf_turn_bad', True)):
            frame[column] = default
        frame['mtf_htf_level'] = np.nan
        if 'date' not in frame.columns or not getattr(self, 'dp', None):
            return frame
        pair = str((metadata or {}).get('pair') or '')
        if not pair:
            return frame
        informative = self.dp.get_pair_dataframe(pair=pair, timeframe=CONTEXT_TIMEFRAME)
        if informative is None or informative.empty or 'date' not in informative.columns:
            return frame
        informative = self._add_htf_context(informative.copy())
        keep = ['date', 'mtf_htf_context', 'mtf_htf_level', 'mtf_htf_turn_bad']
        informative = informative[[column for column in keep if column in informative.columns]].copy().sort_values('date')
        informative['date_merge'] = pd.to_datetime(informative['date'], utc=True).dt.tz_convert(None) + pd.to_timedelta(timeframe_to_minutes(CONTEXT_TIMEFRAME), unit='m')
        base = frame.drop(columns=['mtf_htf_context', 'mtf_htf_level', 'mtf_htf_turn_bad'], errors='ignore').reset_index().rename(columns={'index': '__row_index'}).sort_values('date')
        base['__date_merge'] = pd.to_datetime(base['date'], utc=True).dt.tz_convert(None)
        merged = pd.merge_asof(base, informative.drop(columns=['date']).sort_values('date_merge'), left_on='__date_merge', right_on='date_merge', direction='backward')
        merged = merged.sort_values('__row_index').drop(columns=['__row_index', '__date_merge', 'date_merge'], errors='ignore')
        merged.index = dataframe.index
        merged['mtf_htf_context'] = pd.Series(merged['mtf_htf_context'], index=merged.index).astype('boolean').fillna(False).astype(bool)
        merged['mtf_htf_turn_bad'] = pd.Series(merged['mtf_htf_turn_bad'], index=merged.index).astype('boolean').fillna(True).astype(bool)
        return merged

    def _add_htf_context(self, informative: DataFrame) -> DataFrame:
        frame = informative.copy()
        lookback = int(self.context_lookback)
        recent = int(self.context_recent_bars)
        buffer = float(self.breakout_buffer_pct)
        pressure_min = float(self.pressure_min)
        close = _num(frame, 'close')
        open_ = _num(frame, 'open')
        high = _num(frame, 'high')
        low = _num(frame, 'low')
        volume = _num(frame, 'volume').clip(lower=0.0).fillna(0.0)
        candle_range = (high - low).replace(0.0, np.nan)
        pressure = (((close - open_) / candle_range).clip(-1.0, 1.0).fillna(0.0) + ((close - low) / candle_range * 2.0 - 1.0).clip(-1.0, 1.0).fillna(0.0)) / 2.0
        volume_sum = volume.rolling(lookback, min_periods=max(3, lookback // 4)).sum().replace(0.0, np.nan)
        pressure_ratio = (pressure * volume).rolling(lookback, min_periods=max(3, lookback // 4)).sum() / volume_sum
        prior_high = high.shift(1).rolling(lookback, min_periods=max(3, lookback // 4)).max()
        prior_low = low.shift(1).rolling(lookback, min_periods=max(3, lookback // 4)).min()
        trend_ema = close.ewm(span=max(3, lookback // 3), min_periods=max(3, lookback // 6), adjust=False).mean()
        frame = add_volume_profile(frame, window=min(max(lookback, 24), 168), bins=48, value_area_pct=0.7, price_source='hlc3', smooth_bins=3, pressure_delta_min=0.05, node_near_pct=0.01, prefix='htfvp')
        frame = add_bos_choch(frame, strength=3, min_prominence_atr=0.35, min_pivot_spacing_bars=2, max_pivot_age_bars=max(24, min(lookback * 2, 160)), breakout_buffer_atr=0.15, prefix='htfms')
        vp_score_long = _num(frame, 'htfvp_score_long')
        vp_score_short = _num(frame, 'htfvp_score_short')
        vp_ctx_long = _num(frame, 'htfvp_context_score_bull')
        vp_ctx_short = _num(frame, 'htfvp_context_score_bear')
        prior_vah = _num(frame, 'htfvp_prior_vah')
        prior_val = _num(frame, 'htfvp_prior_val')
        node_long = _bool(frame, 'htfvp_node_entry_long') | _bool(frame, 'htfvp_node_hold_long')
        node_short = _bool(frame, 'htfvp_node_entry_short') | _bool(frame, 'htfvp_node_hold_short')
        vp_long_ok = (vp_score_long.ge(float(self.vp_score_min)) | vp_ctx_long.ge(float(self.vp_context_min))) & vp_score_long.ge(vp_score_short)
        vp_short_ok = (vp_score_short.ge(float(self.vp_score_min)) | vp_ctx_short.ge(float(self.vp_context_min))) & vp_score_short.ge(vp_score_long)
        behavior = HTF_BEHAVIOR
        if behavior in {'res_break_long', 'prior_high_break_long', 'supply_break_long'}:
            context = close.gt(prior_high.mul(1.0 + buffer))
            level = prior_high
        elif behavior == 'vp_value_accept_long':
            context = close.gt(prior_vah) & vp_long_ok
            level = prior_vah
        elif behavior == 'vp_node_break_long':
            context = (node_long | vp_long_ok) & close.gt(prior_vah)
            level = prior_vah.fillna(prior_high)
        elif behavior == 'support_reclaim_long':
            context = low.lt(prior_low.mul(1.0 - buffer)) & close.gt(prior_low) & pressure_ratio.ge(-pressure_min)
            level = prior_low
        elif behavior == 'bos_bull_long':
            context = _bool(frame, 'htfms_bos_to_bull') | _num(frame, 'htfms_state').ge(0.0) & close.gt(trend_ema) & pressure_ratio.ge(-pressure_min)
            level = prior_high.fillna(trend_ema)
        elif behavior in {'prior_low_break_short', 'support_break_short'}:
            context = close.lt(prior_low.mul(1.0 - buffer))
            level = prior_low
        elif behavior == 'vp_value_break_short':
            context = close.lt(prior_val) & vp_short_ok
            level = prior_val
        elif behavior in {'vp_vah_reject_short', 'vp_node_reject_short'}:
            context = high.ge(prior_vah) & close.lt(prior_vah) & (vp_short_ok | node_short)
            level = prior_vah
        elif behavior == 'res_reject_short':
            context = high.gt(prior_high.mul(1.0 - buffer)) & close.lt(prior_high) & pressure_ratio.le(pressure_min)
            level = prior_high
        elif behavior == 'bos_bear_short':
            context = _bool(frame, 'htfms_bos_to_bear') | _num(frame, 'htfms_state').le(0.0) & close.lt(trend_ema) & pressure_ratio.le(pressure_min)
            level = prior_low.fillna(trend_ema)
        else:
            context = pd.Series(False, index=frame.index)
            level = pd.Series(np.nan, index=frame.index)
        frame['mtf_htf_context'] = context.fillna(False).astype('int8').rolling(recent, min_periods=1).max().gt(0)
        frame['mtf_htf_level'] = level
        if SIDE == 'short':
            frame['mtf_htf_turn_bad'] = (close.gt(trend_ema) & pressure_ratio.ge(pressure_min)).fillna(False)
        else:
            frame['mtf_htf_turn_bad'] = (close.lt(trend_ema) & pressure_ratio.le(-pressure_min)).fillna(False)
        return frame

    def _ltf_trigger(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        open_ = _num(dataframe, 'open')
        high = _num(dataframe, 'high')
        low = _num(dataframe, 'low')
        level = _num(dataframe, 'mtf_htf_level')
        ema = _num(dataframe, 'mtf_ltf_ema')
        local_high = _num(dataframe, 'mtf_ltf_local_high')
        local_low = _num(dataframe, 'mtf_ltf_local_low')
        retest_buffer = float(self.retest_buffer_pct)
        breakout_buffer = float(self.breakout_buffer_pct)
        pressure = _num(dataframe, 'mtf_ltf_pressure')
        if LTF_TRIGGER == 'local_break':
            return close.lt(local_low.mul(1.0 - breakout_buffer)) if SIDE == 'short' else close.gt(local_high.mul(1.0 + breakout_buffer))
        if LTF_TRIGGER == 'pullback_reclaim':
            return high.ge(ema.mul(1.0 - retest_buffer)) & close.lt(ema) & close.lt(open_) if SIDE == 'short' else low.le(ema.mul(1.0 + retest_buffer)) & close.gt(ema) & close.gt(open_)
        if LTF_TRIGGER == 'momentum_confirm':
            return close.lt(close.shift(1)) & pressure.le(-float(self.pressure_min)) if SIDE == 'short' else close.gt(close.shift(1)) & pressure.ge(float(self.pressure_min))
        if LTF_TRIGGER == 'rejection_reclaim':
            return high.ge(level.mul(1.0 - retest_buffer)) & close.lt(level) & close.lt(open_) if SIDE == 'short' else low.le(level.mul(1.0 + retest_buffer)) & close.gt(level) & close.gt(open_)
        return high.ge(level.mul(1.0 - retest_buffer)) & close.lt(level) & close.lt(open_) if SIDE == 'short' else low.le(level.mul(1.0 + retest_buffer)) & close.gt(level) & close.gt(open_)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = None
        condition = _bool(dataframe, 'mtf_htf_context') & self._ltf_trigger(dataframe)
        condition = apply_sieve2_optional_guards(self, dataframe, condition, SIDE)
        valid = condition.fillna(False) & dataframe['volume'].gt(0.0) & dataframe['close'].notna()
        if SIDE == 'short':
            dataframe.loc[valid, 'enter_short'] = 1
        else:
            dataframe.loc[valid, 'enter_long'] = 1
        dataframe.loc[valid, 'enter_tag'] = ENTRY_TAG
        return dataframe
    SOURCE_ENTRY_STEM = 'overtrade_mtf_h4_supply_breakout_long_1h_local_break_vp_market_guard'
    FOCUSED_EXIT_CONTRACT = 'source_invalidation'
    FOCUSED_SOURCE_PROFILE = {'side': 'long', 'min_level_distance': 0.001, 'invalidation': {'provider': 'h4_supply_breakout_failure', 'mode': 'level', 'long': {'level': 'mtf_htf_level', 'available': None}}}
    FOCUSED_EXIT_PLANS = {'source_invalidation_close1': {'contract': 'source_invalidation', 'name': 'source_invalidation_close1', 'action_sequence': ('source_invalidation_confirmation', 'invalidation_full', 'hard_stop', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'invalidation_confirmations': 1, 'invalidation_band': 0.0025}, 'source_invalidation_close2': {'contract': 'source_invalidation', 'name': 'source_invalidation_close2', 'action_sequence': ('source_invalidation_confirmation', 'invalidation_full', 'hard_stop', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'invalidation_confirmations': 2, 'invalidation_band': 0.0025}, 'source_invalidation_close2_band_0_5': {'contract': 'source_invalidation', 'name': 'source_invalidation_close2_band_0_5', 'action_sequence': ('source_invalidation_confirmation', 'invalidation_full', 'hard_stop', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'invalidation_confirmations': 2, 'invalidation_band': 0.005}}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'high', 'low', 'mtf_htf_level', 'open')
    FOCUSED_STATE_VERSION = 1
    FOCUSED_STATE_KEY = 'sieve3_v2_focused:Sieve3V2SourceInvalidationFromOvertradeMtfH4SupplyBreakoutLong1hLocalBreakVpMarketGuard:source_invalidation'
    LOCKED_BUY_PARAMS = {'use_sieve2_vp_guard': True, 'sieve2_vp_guard_mode': 'score_or_context', 'sieve2_vp_window': 96, 'sieve2_vp_bins': 36, 'sieve2_vp_score_min': 0.15, 'sieve2_vp_context_min': 0.15, 'use_sieve2_market_guard': True, 'sieve2_market_guard_mode': 'pressure_or_trend', 'sieve2_market_window': 24, 'sieve2_market_pressure_min': 0.03, 'sieve2_market_trend_min': 0.15, 'sieve2_rs_benchmark_pair': 'BTC/USDT:USDT', 'sieve2_rs_score_min': 0.3, 'context_lookback': 48, 'context_recent_bars': 5, 'local_lookback': 12, 'retest_buffer_pct': 0.004, 'breakout_buffer_pct': 0.0, 'volume_ratio_min': 1.5, 'pressure_min': 0.03, 'vp_score_min': 0.5, 'vp_context_min': 0.18, 'use_ltf_volume_guard': False, 'use_ltf_pressure_guard': False, 'use_htf_turn_guard': False}
    ACTIVE_SELL_PARAMS = ('invalidation_confirmations', 'invalidation_band_quarter_percent', 'hard_stop_percent', 'max_hold_scale')
    position_adjustment_enable = False
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    minimal_roi = {'0': 100.0}
    stoploss = -0.99
    exit_plan = CategoricalParameter(('source_invalidation_close1', 'source_invalidation_close2', 'source_invalidation_close2_band_0_5'), default='source_invalidation_close1', space='sell', optimize=False, load=True)
    exit_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    invalidation_confirmations = IntParameter(1, 3, default=1, space='sell', optimize=True, load=True)
    invalidation_confirmations.batch_tags = ('family:exits', 'mode:sieve3_exit')
    invalidation_band_quarter_percent = IntParameter(0, 6, default=1, space='sell', optimize=True, load=True)
    invalidation_band_quarter_percent.batch_tags = ('family:exits', 'mode:sieve3_exit')
    hard_stop_percent = IntParameter(2, 6, default=3, space='sell', optimize=True, load=True)
    hard_stop_percent.batch_tags = ('family:exits', 'mode:sieve3_exit')
    max_hold_scale = IntParameter(1, 4, default=2, space='sell', optimize=True, load=True)
    max_hold_scale.batch_tags = ('family:exits', 'mode:sieve3_exit')

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
        values = frame[column]
        if bool(values.isna().any()):
            raise ValueError(f'required boolean evidence column contains NaN: {column}')
        try:
            return values.astype('boolean').astype(bool)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'required boolean evidence column is malformed: {column}') from exc

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
        plan['invalidation_confirmations'] = int(self.invalidation_confirmations.value)
        plan['invalidation_band'] = float(self.invalidation_band_quarter_percent.value) * 0.0025
        plan['hard_stop_ratio'] = float(self.hard_stop_percent.value) * 0.01
        plan['max_hold_candles'] = max(1, int(round(float(plan.get('max_hold_candles', 336)) * float(self.max_hold_scale.value) / 2.0)))
        return plan

    @staticmethod
    def _focused_decision(action: str='hold', tag: str | None=None) -> dict[str, Any]:
        return {'action': action, 'tag': tag}

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
            if bool(pd.isna(availability)):
                if role == 'invalidation':
                    raise ValueError(f'required invalidation availability is NaN: {available_column}')
                return None
            if not bool(availability):
                if role == 'invalidation':
                    raise ValueError(f'required invalidation level is unavailable: {available_column}')
                return None
        level_column = binding.get('level')
        if not level_column:
            if role == 'invalidation':
                raise ValueError('required invalidation profile has no level column')
            return None
        level = self._focused_float(row[str(level_column)])
        if level is None or level <= 0.0:
            if role == 'invalidation':
                raise ValueError(f'required invalidation level must be positive and finite: {level_column}')
            return None
        if role == 'invalidation':
            valid = level > entry_rate if side == 'short' else level < entry_rate
            if not valid:
                raise ValueError(f'required invalidation level is on the wrong side of entry: {level_column}')
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
        invalidation_provider = self.FOCUSED_SOURCE_PROFILE.get('invalidation')
        if not isinstance(invalidation_provider, Mapping):
            raise ValueError('source invalidation profile must be a mapping')
        invalidation_mode = str(invalidation_provider.get('mode') or '')
        if invalidation_mode == 'evidence':
            levels['invalidation'] = None
        elif invalidation_mode in {'level', 'level_and_evidence', 'level_or_evidence'}:
            invalidation_binding = self._focused_side_binding('invalidation', side)
            levels['invalidation'] = self._focused_frozen_level(row, invalidation_binding, side, entry_rate, 'invalidation')
        else:
            raise ValueError(f'unsupported invalidation mode: {invalidation_mode}')
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
        fill_time = getattr(trade, 'date_entry_fill_utc', None) or getattr(trade, 'open_date_utc', None)
        if fill_time is None:
            raise ValueError('focused exit runtime requires an entry fill timestamp')
        fill_timestamp = self._focused_utc(fill_time)
        snapshot_date = self._focused_utc(row['date'])
        return {'version': self.FOCUSED_STATE_VERSION, 'contract': self.FOCUSED_EXIT_CONTRACT, 'plan': selected_plan, 'side': side, 'phase': 'pre_target', 'entry_rate': float(entry_rate), 'entry_filled_at': fill_timestamp.isoformat() if fill_timestamp is not None else None, 'entry_snapshot_candle': snapshot_date.isoformat() if snapshot_date is not None else None, 'levels': levels, 'target_states': {slot: {'status': 'available' if value is not None else 'unavailable', 'touched_at': None} for slot, value in levels.items() if slot.startswith('target_')}, 'invalidation_seen': False, 'stop_price': None}

    def _focused_ensure_state(self, pair: str, trade: Any, current_time: Any, order: Any | None=None) -> dict[str, Any] | None:
        state = self._focused_state(trade)
        if state is None:
            state = self._focused_new_state(pair, trade, current_time, order)
            if state is None:
                return None
            self._focused_save_state(trade, state)
        return state

    def _focused_post_entry(self, frame: DataFrame, state: Mapping[str, Any], current_time: Any) -> DataFrame:
        filled_at = self._focused_utc(state.get('entry_filled_at'))
        if filled_at is None:
            raise ValueError('focused exit state has no entry fill timestamp')
        now = self._focused_utc(current_time)
        if now is None:
            raise ValueError('current_time is required for post-entry evaluation')
        dates = pd.to_datetime(frame['date'], utc=True, errors='raise')
        close_times = self._focused_close_times(frame)
        return frame.loc[dates.ge(filled_at) & close_times.le(now)]


    @staticmethod

    @staticmethod
    def _focused_any_n(condition: Series, count: int) -> bool:
        needed = max(1, int(count))
        values = condition.fillna(False).astype(bool)
        if len(values) < needed:
            return False
        confirmed = values.rolling(window=needed, min_periods=needed).sum().ge(needed)
        return bool(confirmed.any())
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
        frame = self._focused_analyzed_frame(pair, current_time)
        self._focused_require_columns(frame)
        post_entry = self._focused_post_entry(frame, state, current_time)
        events = self._focused_contract_events(post_entry, state, plan, current_profit)
        events['age_candles'] = len(post_entry)
        if self._focused_hard_stop_breached(plan, state, float(current_rate)):
            decision = self._focused_decision('full', 'focused_hard_stop')
        else:
            decision = self._focused_contract_decision(state, plan, events, current_profit)
        if decision['action'] == 'hold' and events['age_candles'] >= int(plan['max_hold_candles']):
            decision = self._focused_decision('full', 'focused_max_hold')
        if repr(state) != before:
            self._focused_save_state(trade, state)
        return (plan, state, events, decision)

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | bool | None:
        _ = kwargs
        _, _, _, decision = self._focused_context(pair, trade, current_time, current_rate, current_profit)
        return str(decision['tag']) if decision['action'] == 'full' else None

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float | None:
        _ = (after_fill, kwargs)
        plan, state, _, decision = self._focused_context(pair, trade, current_time, current_rate, current_profit)
        if not state:
            return None
        if decision['action'] == 'full':
            return None
        desired = self._focused_stop_overlay(plan, state, current_profit)
        persisted = self._focused_float(state.get('stop_price'))
        stop_price = desired if persisted is None else self._focused_tighter_stop(str(state['side']), persisted, desired)
        if state['side'] == 'short' and stop_price <= current_rate or (state['side'] == 'long' and stop_price >= current_rate):
            return None
        if persisted is None or not math.isclose(stop_price, persisted, rel_tol=0.0, abs_tol=1e-12):
            state['stop_price'] = stop_price
            self._focused_save_state(trade, state)
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=state['side'] == 'short', leverage=float(getattr(trade, 'leverage', 1.0) or 1.0))

    def _focused_invalidation_event(self, post_entry: DataFrame, state: dict[str, Any], plan: Mapping[str, Any]) -> bool:
        if state.get('invalidation_seen'):
            return True
        side = str(state['side'])
        provider = self.FOCUSED_SOURCE_PROFILE['invalidation']
        if not isinstance(provider, Mapping):
            raise ValueError('source invalidation profile must be a mapping')
        binding = provider[side]
        mode = str(provider.get('mode') or '')
        requires_level = mode in {'level', 'level_and_evidence', 'level_or_evidence'}
        if not requires_level and mode != 'evidence':
            raise ValueError(f'unsupported invalidation mode: {mode}')
        level = self._focused_float((state.get('levels') or {}).get('invalidation'))
        if requires_level and (level is None or level <= 0.0):
            level_column = binding.get('level')
            raise ValueError(f'active trade state requires a positive finite invalidation level: {level_column}')
        if post_entry.empty:
            return False
        band = float(plan.get('invalidation_band') or 0.0)
        close = self._focused_number(post_entry, 'close')
        if level is None:
            level_condition = pd.Series(False, index=post_entry.index, dtype='bool')
        elif side == 'short':
            level_condition = close.gt(level * (1.0 + band))
        else:
            level_condition = close.lt(level * (1.0 - band))
        evidence_column = binding.get('evidence')
        if evidence_column:
            available_column = binding.get('available')
            if available_column:
                availability = self._focused_boolean(post_entry, str(available_column))
                if not bool(availability.all()):
                    raise ValueError(f'required invalidation evidence is unavailable: {available_column}')
            evidence_condition = self._focused_boolean(post_entry, str(evidence_column))
        else:
            evidence_condition = pd.Series(False, index=post_entry.index, dtype='bool')
        confirmations = int(plan.get('invalidation_confirmations') or 1)
        if mode == 'level':
            confirmed = self._focused_any_n(level_condition, confirmations)
        elif mode == 'evidence':
            event_condition = evidence_condition
        elif mode == 'level_and_evidence':
            event_condition = level_condition & evidence_condition
        elif mode == 'level_or_evidence':
            event_condition = evidence_condition
        else:
            raise ValueError(f'unsupported invalidation mode: {mode}')
        if mode != 'level':
            evidence_timeframe = provider.get('timeframe')
            if evidence_timeframe is not None:
                event_condition = self._focused_distinct_periods(post_entry, event_condition, str(evidence_timeframe))
            event_seen = bool(event_condition.fillna(False).any())
            confirmed = self._focused_any_n(level_condition, confirmations) or event_seen if mode == 'level_or_evidence' else event_seen
        if confirmed:
            state['invalidation_seen'] = True
            state['phase'] = 'invalidated'
        return confirmed

    def _focused_contract_events(self, post_entry: DataFrame, state: dict[str, Any], plan: Mapping[str, Any], current_profit: float) -> dict[str, Any]:
        _ = current_profit
        return {'invalidation': self._focused_invalidation_event(post_entry, state, plan)}

    def _focused_contract_decision(self, state: dict[str, Any], plan: Mapping[str, Any], events: Mapping[str, Any], current_profit: float) -> dict[str, Any]:
        _ = (state, plan, current_profit)
        if events.get('invalidation'):
            return self._focused_decision('full', 'focused_source_invalidation')
        return self._focused_decision()

    def _focused_stop_overlay(self, plan: Mapping[str, Any], state: Mapping[str, Any], current_profit: float) -> float:
        stop_price = self._focused_hard_stop_price(plan, state)
        return stop_price

    def order_filled(self, pair: str, trade: Any, order: Any, current_time: datetime, **kwargs: Any) -> None:
        _ = kwargs
        if getattr(order, 'ft_order_side', None) == getattr(trade, 'entry_side', None) and self._focused_state(trade) is None:
            state = self._focused_new_state(pair, trade, current_time, order)
            if state is None:
                return None
            self._focused_save_state(trade, state)
            return None
        return None

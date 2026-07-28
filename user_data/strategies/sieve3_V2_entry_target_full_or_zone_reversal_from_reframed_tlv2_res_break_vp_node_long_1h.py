from __future__ import annotations
import math
import pandas as pd
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from pandas import DataFrame, Series
from user_data.Indicators.market_state import add_market_state
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, stoploss_from_absolute, IntParameter
ENTRY_SOURCE_STAGE = 'sieve2'
import numpy as np
from freqtrade.strategy import IStrategy
from user_data.Indicators.complex_trendline_projection_v2 import add_trendline_projection_v2
from user_data.Indicators.complex_volume_profile import add_volume_profile
from user_data.Indicators.pattern_continuation import add_pattern_continuation
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2
from user_data.Indicators.pattern_wolfe_waves import add_pattern_wolfe_waves
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


def _sieve2_vp_guard(frame: pd.DataFrame, side: str, mode: str, score_min: float, context_min: float) -> pd.Series:
    score = _sieve2_num(frame, f"s2vp_score_{side}")
    other = _sieve2_num(frame, "s2vp_score_short" if side == "long" else "s2vp_score_long")
    context = _sieve2_num(frame, "s2vp_context_score_bull" if side == "long" else "s2vp_context_score_bear")
    base_score = score.ge(score_min) & score.gt(other)
    base_context = context.ge(context_min)
    if mode == "node_confirm":
        node_entry = _sieve2_bool(frame, f"s2vp_node_entry_{side}")
        node_hold = _sieve2_bool(frame, f"s2vp_node_hold_{side}")
        return (node_entry | node_hold | (base_score & base_context)).fillna(False)
    if mode == "value_area_confirm":
        in_value = _sieve2_bool(frame, "s2vp_in_value_area")
        value_side = _sieve2_bool(frame, "s2vp_above_value_area" if side == "long" else "s2vp_below_value_area")
        return (value_side | (in_value & base_context)).fillna(False)
    if mode in {"breakout_acceptance", "rejection_confirm"}:
        close = _sieve2_num(frame, "close")
        prior_vah = _sieve2_num(frame, "s2vp_prior_vah")
        prior_val = _sieve2_num(frame, "s2vp_prior_val")
        node_entry = _sieve2_bool(frame, f"s2vp_node_entry_{side}")
        node_hold = _sieve2_bool(frame, f"s2vp_node_hold_{side}")
        if mode == "breakout_acceptance":
            accepted = close.gt(prior_vah) if side == "long" else close.lt(prior_val)
            return (accepted & (base_score | base_context | node_entry | node_hold)).fillna(False)
        high = _sieve2_num(frame, "high")
        low = _sieve2_num(frame, "low")
        rejection = low.le(prior_val) & close.gt(prior_val) & close.le(prior_vah) if side == "long" else high.ge(prior_vah) & close.lt(prior_vah) & close.ge(prior_val)
        return (rejection & (base_score | base_context | node_entry | node_hold)).fillna(False)
    if mode == "poc_hvn_reject":
        node_entry = _sieve2_bool(frame, f"s2vp_node_entry_{side}")
        return (node_entry | (base_score & base_context)).fillna(False)
    if mode == "prior_level_confirm":
        close = _sieve2_num(frame, "close")
        level = _sieve2_num(frame, "s2vp_prior_vah" if side == "long" else "s2vp_prior_val")
        confirmed = close.gt(level) if side == "long" else close.lt(level)
        return (confirmed & (base_score | base_context)).fillna(False)
    return (base_score | base_context).fillna(False)


def _sieve2_market_guard(frame: pd.DataFrame, side: str, mode: str, pressure_min: float, trend_min: float, rs_score_min: float) -> pd.Series:
    _ = rs_score_min
    if mode == "directional_pressure":
        pressure = _sieve2_num(frame, "s2m_pressure_ratio")
        directional = pressure.ge(pressure_min) if side == "long" else pressure.le(-pressure_min)
        return directional.fillna(False)
    if mode == "trend_state":
        trend = _sieve2_num(frame, "s2m_trend_z")
        state = trend.ge(trend_min) if side == "long" else trend.le(-trend_min)
        return state.fillna(False)
    pressure = _sieve2_num(frame, "s2m_pressure_ratio")
    trend = _sieve2_num(frame, "s2m_trend_z")
    directional = pressure.ge(pressure_min) if side == "long" else pressure.le(-pressure_min)
    trend_state = trend.ge(trend_min) if side == "long" else trend.le(-trend_min)
    avoids_pressure = pressure.ge(-pressure_min) if side == "long" else pressure.le(pressure_min)
    avoids_trend = trend.ge(-trend_min) if side == "long" else trend.le(trend_min)
    if mode == "pressure_and_trend": return (directional & trend_state).fillna(False)
    if mode == "avoid_adverse_pressure": return (avoids_pressure & avoids_trend).fillna(False)
    if mode == "avoid_chop": return (pressure.abs().ge(pressure_min) | trend.abs().ge(trend_min)).fillna(False)
    return (directional | trend_state).fillna(False)


def _sieve2_param(strategy: Any, name: str, default: Any) -> Any:
    value = getattr(strategy, name, default)
    return getattr(value, "value", value)


def _sieve2_enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _sieve2_num(frame: pd.DataFrame, column: str, default: float | pd.Series=0.0) -> pd.Series:
    if column not in frame.columns:
        raise KeyError(f'missing required column: {column!r}')
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)


def _sieve2_bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise KeyError(f'missing required column: {column!r}')
    return pd.Series(frame[column], index=frame.index).astype('boolean').fillna(False).astype(bool)
NOVEL_IDEA = False
ENTRY_MODE = 'entry_sieve2_reframed_tlv2_res_break_vp_node_long_1h'


TIMEFRAME = '1h'
CONTEXT_TIMEFRAME = ''
CONCEPT_FAMILY = 'tlv2_vp'
CONCEPT = 'res_break_vp_node'
LTF_TRIGGER = 'breakout'


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

def _cross_above(value: Series, level: Series) -> Series:
    return value.gt(level) & value.shift(1).le(level.shift(1))

def _cross_below(value: Series, level: Series) -> Series:
    return value.lt(level) & value.shift(1).ge(level.shift(1))

SOURCE_ENTRY_CLASS = 'Sieve3ExitFromReframedTlv2ResBreakVpNodeLong1h'
SOURCE_STRATEGY = 'C:/FreqTradeStuff/user_data/strategies/sieve3_exit_from_reframed_tlv2_res_break_vp_node_long_1h.py:Sieve3ExitFromReframedTlv2ResBreakVpNodeLong1h'
LINEAGE_STATUS = 'verified'

EXIT_THEORY = 'entry_target_full_or_zone_reversal'







EXIT_FAMILY = 'entry_target_full_or_zone_reversal'
PRIMARY_TRIGGER = 'provider=TLV2 resistance breakout with VP node or score confirmation'
PRIMARY_GUARD = 'provider=active TLV2 resistance score and distance plus VP node or score context and locked guards'
PRIMARY_TARGET = 'named fixed partial and remainder objectives'
PRIMARY_INVALIDATION = 'named fixed hard stop'
TARGET_PROVIDER = 'nearest forward entry-family VP/TLV2 obstacle'
INVALIDATION_PROVIDER = 'nearest crossed-side entry-family VP/TLV2 boundary'
SOURCE_RESULT_BATCH = 'historical_promotion_lineage_unknown'
RESEARCH_PATH = 'sieve3_exit_entry_target_full_or_zone_reversal'
EXIT_HYPOTHESIS = 'Entry-known target tests touch/close full exit versus target-zone reversal confirmation.'
ACTIVE_SELL_PARAMS = ('exit_plan', 'target_band_quarter_percent', 'reversal_confirmations', 'invalidation_band_quarter_percent', 'invalidation_confirmations')
SIEVE_STAGE = 'sieve3'
SIDE = 'long'
ENTRY_TAG = 'sieve2_reframed_tlv2_res_break_vp_node_long_1h'

class Sieve3V2EntryTargetFullOrZoneReversalFromReframedTlv2ResBreakVpNodeLong1H(IStrategy):
    SIEVE2_ALWAYS_ON_GUARDS = True
    'Sieve2 reframed missed concept probe.'
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 420
    process_only_new_candles = True
    can_short = SIDE == 'short'
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
    sieve2_rs_benchmark_pair = 'BTC/USDT:USDT'
    sieve2_rs_score_min = 0.3
    # Legacy fixed-off entry parameter: use_volume_guard=False (BooleanParameter, buy space).
    volume_guard_window = 12
    volume_ratio_min = 0.8
    # Legacy fixed-off entry parameter: use_pressure_guard=False (BooleanParameter, buy space).
    pressure_window = 24
    pressure_min = 0.1
    # Legacy fixed-off entry parameter: use_body_direction_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_close_direction_guard=False (BooleanParameter, buy space).
    vp_window = 96
    vp_bins = 48
    vp_score_min = 0.35
    vp_context_min = 0.28
    level_buffer_pct = 0.004
    context_recent_bars = 2
    liquidity_lookback = 96
    min_pattern_score = 0.55
    pivot_strength = 2
    min_line_score = 0.4
    max_distance_atr = 10.0
    min_active_bars = 8
    capitulation_move_pct = 0.04

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = (pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs)
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        if not CONTEXT_TIMEFRAME or not getattr(self, 'dp', None):
            return []
        return [(pair, CONTEXT_TIMEFRAME) for pair in self.dp.current_whitelist()]

    def _sieve3_entry_populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self._add_base_features(dataframe.copy())
        if CONTEXT_TIMEFRAME:
            dataframe = self._merge_htf_context(dataframe, metadata or {})
        dataframe = add_sieve2_guard_indicators(self, dataframe, metadata)
        return dataframe

    def _add_base_features(self, dataframe: DataFrame) -> DataFrame:
        if CONCEPT_FAMILY in {'vp_location', 'tlv2_vp', 'capitulation_vp'}:
            dataframe = self._add_vp(dataframe)
        if CONCEPT_FAMILY == 'tlv2_vp':
            dataframe = self._add_tlv2(dataframe)
        if CONCEPT_FAMILY == 'liquidity':
            dataframe = self._add_liquidity(dataframe)
        if CONCEPT_FAMILY == 'capitulation_vp':
            dataframe = self._add_capitulation(dataframe)
        return dataframe

    def _add_vp(self, dataframe: DataFrame) -> DataFrame:
        return add_volume_profile(dataframe, window=int(self.vp_window), bins=int(self.vp_bins), value_area_pct=0.7, price_source='hlc3', smooth_bins=3, pressure_delta_min=0.05, node_near_pct=float(self.level_buffer_pct), prefix='vp')

    def _add_tlv2(self, dataframe: DataFrame) -> DataFrame:
        return add_trendline_projection_v2(dataframe, timeframe=self.timeframe, pivot_strength=int(self.pivot_strength), raw_line_output_count=1, min_output_line_score=float(self.min_line_score), min_output_active_bars=int(self.min_active_bars), max_active_line_distance_atr_mult=float(self.max_distance_atr), proximity_rank_weight=0.05, output_prefix='tlv2')

    def _add_liquidity(self, dataframe: DataFrame) -> DataFrame:
        close = _num(dataframe, 'close')
        high = _num(dataframe, 'high')
        low = _num(dataframe, 'low')
        lookback = int(self.liquidity_lookback)
        tol = float(self.level_buffer_pct)
        prior_high = high.shift(1).rolling(lookback, min_periods=max(4, lookback // 4)).max()
        prior_low = low.shift(1).rolling(lookback, min_periods=max(4, lookback // 4)).min()
        equal_high = high.shift(1).rolling(lookback, min_periods=max(4, lookback // 4)).quantile(0.9)
        equal_low = low.shift(1).rolling(lookback, min_periods=max(4, lookback // 4)).quantile(0.1)
        dataframe['liq_sweep_reclaim_long'] = low.lt(prior_low.mul(1.0 - tol)) & close.gt(prior_low)
        dataframe['liq_sweep_reject_short'] = high.gt(prior_high.mul(1.0 + tol)) & close.lt(prior_high)
        dataframe['liq_equal_reclaim_long'] = low.lt(equal_low.mul(1.0 - tol)) & close.gt(equal_low)
        dataframe['liq_equal_reject_short'] = high.gt(equal_high.mul(1.0 + tol)) & close.lt(equal_high)
        return dataframe

    def _add_capitulation(self, dataframe: DataFrame) -> DataFrame:
        close = _num(dataframe, 'close')
        open_ = _num(dataframe, 'open')
        volume = _num(dataframe, 'volume').clip(lower=0.0)
        window = int(self.liquidity_lookback)
        move = (close - open_) / close.replace(0.0, np.nan)
        volume_base = volume.shift(1).rolling(window, min_periods=max(4, window // 4)).mean().replace(0.0, np.nan)
        dataframe['capitulation_down'] = move.le(-float(self.capitulation_move_pct)) & volume.ge(volume_base.mul(float(self.volume_ratio_min)))
        dataframe['capitulation_up'] = move.ge(float(self.capitulation_move_pct)) & volume.ge(volume_base.mul(float(self.volume_ratio_min)))
        dataframe['recent_capitulation_down'] = dataframe['capitulation_down'].astype('int8').rolling(int(self.context_recent_bars), min_periods=1).max().gt(0)
        dataframe['recent_capitulation_up'] = dataframe['capitulation_up'].astype('int8').rolling(int(self.context_recent_bars), min_periods=1).max().gt(0)
        return dataframe

    def _merge_htf_context(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        frame = dataframe.copy()
        frame['mtf_context'] = False
        frame['mtf_level'] = np.nan
        if 'date' not in frame.columns or not getattr(self, 'dp', None):
            return frame
        pair = str((metadata or {}).get('pair') or '')
        if not pair:
            return frame
        informative = self.dp.get_pair_dataframe(pair=pair, timeframe=CONTEXT_TIMEFRAME)
        if informative is None or informative.empty or 'date' not in informative.columns:
            return frame
        informative = self._add_htf_features(informative.copy())
        informative = informative[['date', 'mtf_context', 'mtf_level']].copy().sort_values('date')
        informative['date_merge'] = pd.to_datetime(informative['date'], utc=True).dt.tz_convert(None) + pd.to_timedelta(timeframe_to_minutes(CONTEXT_TIMEFRAME), unit='m')
        base = frame.drop(columns=['mtf_context', 'mtf_level'], errors='ignore').reset_index().rename(columns={'index': '__row_index'}).sort_values('date')
        base['__date_merge'] = pd.to_datetime(base['date'], utc=True).dt.tz_convert(None)
        merged = pd.merge_asof(base, informative.drop(columns=['date']).sort_values('date_merge'), left_on='__date_merge', right_on='date_merge', direction='backward')
        merged = merged.sort_values('__row_index').drop(columns=['__row_index', '__date_merge', 'date_merge'], errors='ignore')
        merged.index = dataframe.index
        merged['mtf_context'] = pd.Series(merged['mtf_context'], index=merged.index).astype('boolean').fillna(False).astype(bool)
        return merged

    def _add_htf_features(self, dataframe: DataFrame) -> DataFrame:
        if CONCEPT_FAMILY == 'mtf_continuation':
            dataframe = add_pattern_continuation(dataframe, timeframe=CONTEXT_TIMEFRAME, min_flag_quality=float(self.min_pattern_score))
            prefix = 'pat_flag' if CONCEPT == 'flag' else 'pat_pennant'
            context = _bool(dataframe, f'{prefix}_pattern_confirmed') | _bool(dataframe, f'{prefix}_pattern_present')
            dataframe['mtf_context'] = context & _num(dataframe, f'{prefix}_indicator_score').ge(float(self.min_pattern_score))
            dataframe['mtf_level'] = _num(dataframe, f'{prefix}_confirmation_level', np.nan)
            return dataframe
        if CONCEPT_FAMILY == 'mtf_geometry':
            dataframe = add_pattern_geometry_v2(dataframe, timeframe=CONTEXT_TIMEFRAME, output_slots=1, include_triangle_patterns=CONCEPT == 'triangle', include_wedge_patterns=CONCEPT == 'wedge', include_compression_patterns=False, include_rectangle_patterns=CONCEPT == 'rectangle', include_ascending_channel_patterns=False, include_descending_channel_patterns=False, min_line_score=float(self.min_line_score), output_prefix='pg2')
            prefix = f'pg2_{CONCEPT}'
            dataframe['mtf_context'] = _bool(dataframe, f'{prefix}_pattern_present') & _num(dataframe, f'{prefix}_indicator_score').ge(float(self.min_line_score))
            dataframe['mtf_level'] = _num(dataframe, f'{prefix}_lower', np.nan) if SIDE == 'short' else _num(dataframe, f'{prefix}_upper', np.nan)
            return dataframe
        if CONCEPT_FAMILY == 'mtf_wolfe':
            dataframe = add_pattern_wolfe_waves(dataframe, min_wave_quality=float(self.min_pattern_score))
            prefix = 'pww_bullish' if SIDE == 'long' else 'pww_bearish'
            dataframe['mtf_context'] = (_bool(dataframe, f'{prefix}_pattern_confirmed') | _bool(dataframe, f'{prefix}_pattern_present')) & _num(dataframe, f'{prefix}_indicator_score').ge(float(self.min_pattern_score))
            dataframe['mtf_level'] = _num(dataframe, f'{prefix}_confirmation_level', np.nan)
            return dataframe
        dataframe['mtf_context'] = False
        dataframe['mtf_level'] = np.nan
        return dataframe

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype='bool')
        _num(dataframe, 'close')
        return guard.fillna(False)

    def _directional_pressure(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        open_ = _num(dataframe, 'open')
        high = _num(dataframe, 'high')
        low = _num(dataframe, 'low')
        volume = _num(dataframe, 'volume').clip(lower=0.0).fillna(0.0)
        candle_range = (high - low).replace(0.0, np.nan)
        body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
        close_location = ((close - low) / candle_range * 2.0 - 1.0).clip(-1.0, 1.0)
        pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
        window = int(self.pressure_window)
        baseline = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
        return (pressure * volume).rolling(window, min_periods=max(2, window // 3)).sum() / baseline

    def _vp_trigger(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        open_ = _num(dataframe, 'open')
        high = _num(dataframe, 'high')
        low = _num(dataframe, 'low')
        near = float(self.level_buffer_pct)
        long_ok = _num(dataframe, 'vp_score_long').ge(float(self.vp_score_min)) | _num(dataframe, 'vp_context_score_bull').ge(float(self.vp_context_min))
        short_ok = _num(dataframe, 'vp_score_short').ge(float(self.vp_score_min)) | _num(dataframe, 'vp_context_score_bear').ge(float(self.vp_context_min))
        poc = _num(dataframe, 'vp_prior_poc', np.nan)
        if CONCEPT == 'vp_poc_reclaim_long':
            return low.le(poc.mul(1.0 + near)) & close.gt(poc) & close.gt(open_) & long_ok
        if CONCEPT == 'vp_poc_reject_short':
            return high.ge(poc.mul(1.0 - near)) & close.lt(poc) & close.lt(open_) & short_ok
        if CONCEPT == 'vp_hvn_reclaim_long':
            return _bool(dataframe, 'vp_hvn_below_reclaim') & long_ok
        if CONCEPT == 'vp_hvn_reject_short':
            return _bool(dataframe, 'vp_hvn_above_reject') & short_ok
        if CONCEPT == 'vp_lvn_accept_long':
            return _bool(dataframe, 'vp_lvn_accept_long') & long_ok
        if CONCEPT == 'vp_lvn_accept_short':
            return _bool(dataframe, 'vp_lvn_accept_short') & short_ok
        if CONCEPT == 'vp_lvn_fast_traverse_long':
            return _bool(dataframe, 'vp_lvn_fast_traverse_long') & long_ok
        if CONCEPT == 'vp_lvn_fast_traverse_short':
            return _bool(dataframe, 'vp_lvn_fast_traverse_short') & short_ok
        return pd.Series(False, index=dataframe.index, dtype='bool')

    def _mtf_trigger(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        open_ = _num(dataframe, 'open')
        high = _num(dataframe, 'high')
        low = _num(dataframe, 'low')
        level = _num(dataframe, 'mtf_level', np.nan)
        near = float(self.level_buffer_pct)
        if LTF_TRIGGER == 'retest':
            trigger = high.ge(level.mul(1.0 - near)) & close.lt(level) & close.lt(open_) if SIDE == 'short' else low.le(level.mul(1.0 + near)) & close.gt(level) & close.gt(open_)
        else:
            trigger = _cross_below(close, level.mul(1.0 - near)) if SIDE == 'short' else _cross_above(close, level.mul(1.0 + near))
        return _bool(dataframe, 'mtf_context') & trigger

    def _liquidity_trigger(self, dataframe: DataFrame) -> Series:
        if CONCEPT == 'equal_lows_reclaim':
            return _bool(dataframe, 'liq_equal_reclaim_long')
        if CONCEPT == 'equal_highs_reject':
            return _bool(dataframe, 'liq_equal_reject_short')
        if CONCEPT == 'prior_low_reclaim':
            return _bool(dataframe, 'liq_sweep_reclaim_long')
        if CONCEPT == 'prior_high_reject':
            return _bool(dataframe, 'liq_sweep_reject_short')
        return pd.Series(False, index=dataframe.index, dtype='bool')

    def _tlv2_vp_trigger(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        open_ = _num(dataframe, 'open')
        high = _num(dataframe, 'high')
        low = _num(dataframe, 'low')
        near = float(self.level_buffer_pct)
        if SIDE == 'short':
            line = _num(dataframe, 'tlv2_support_line_rank0', np.nan)
            score = _num(dataframe, 'tlv2_support_score_rank0')
            distance = _num(dataframe, 'tlv2_support_distance_atr_rank0', np.nan)
            vp_ok = _bool(dataframe, 'vp_node_entry_short') | _bool(dataframe, 'vp_node_hold_short') | _num(dataframe, 'vp_score_short').ge(float(self.vp_score_min))
            breakout = _cross_below(close, line.mul(1.0 - near))
            retest = high.ge(line.mul(1.0 - near)) & close.lt(line) & close.lt(open_)
        else:
            line = _num(dataframe, 'tlv2_resistance_line_rank0', np.nan)
            score = _num(dataframe, 'tlv2_resistance_score_rank0')
            distance = _num(dataframe, 'tlv2_resistance_distance_atr_rank0', np.nan)
            vp_ok = _bool(dataframe, 'vp_node_entry_long') | _bool(dataframe, 'vp_node_hold_long') | _num(dataframe, 'vp_score_long').ge(float(self.vp_score_min))
            breakout = _cross_above(close, line.mul(1.0 + near))
            retest = low.le(line.mul(1.0 + near)) & close.gt(line) & close.gt(open_)
        active = score.ge(float(self.min_line_score)) & distance.le(float(self.max_distance_atr))
        return active & vp_ok & (retest if LTF_TRIGGER == 'retest' else breakout)

    def _capitulation_trigger(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        open_ = _num(dataframe, 'open')
        poc = _num(dataframe, 'vp_prior_poc', np.nan)
        val = _num(dataframe, 'vp_prior_val', np.nan)
        vah = _num(dataframe, 'vp_prior_vah', np.nan)
        near = float(self.level_buffer_pct)
        if SIDE == 'short':
            return _bool(dataframe, 'recent_capitulation_up') & (close.lt(vah) & close.lt(open_) | _cross_below(close, poc.mul(1.0 - near))) & (_num(dataframe, 'vp_score_short').ge(float(self.vp_score_min)) | _num(dataframe, 'vp_context_score_bear').ge(float(self.vp_context_min)))
        return _bool(dataframe, 'recent_capitulation_down') & (close.gt(val) & close.gt(open_) | _cross_above(close, poc.mul(1.0 + near))) & (_num(dataframe, 'vp_score_long').ge(float(self.vp_score_min)) | _num(dataframe, 'vp_context_score_bull').ge(float(self.vp_context_min)))

    def _entry_condition(self, dataframe: DataFrame) -> Series:
        if CONCEPT_FAMILY == 'vp_location':
            return self._vp_trigger(dataframe)
        if CONCEPT_FAMILY in {'mtf_continuation', 'mtf_geometry', 'mtf_wolfe'}:
            return self._mtf_trigger(dataframe)
        if CONCEPT_FAMILY == 'liquidity':
            return self._liquidity_trigger(dataframe)
        if CONCEPT_FAMILY == 'tlv2_vp':
            return self._tlv2_vp_trigger(dataframe)
        if CONCEPT_FAMILY == 'capitulation_vp':
            return self._capitulation_trigger(dataframe)
        return pd.Series(False, index=dataframe.index, dtype='bool')


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self._sieve3_entry_populate_indicators(dataframe, metadata)
        close = pd.to_numeric(dataframe['close'], errors='coerce')
        candidate_columns = ('vp_poc', 'vp_vah', 'vp_val', 'vp_prior_poc', 'vp_prior_vah', 'vp_prior_val', 'vp_hvn_above', 'vp_hvn_below', 'vp_lvn_above', 'vp_lvn_below', 'tlv2_support_line_rank0', 'tlv2_resistance_line_rank0', 'tlv2_support_line_rank1', 'tlv2_resistance_line_rank1', 'tlv2_support_line_rank2', 'tlv2_resistance_line_rank2')
        candidates = [pd.to_numeric(dataframe[column], errors='coerce') for column in candidate_columns if column in dataframe.columns]
        if candidates:
            levels = pd.concat(candidates, axis=1)
            if SIDE == 'short':
                dataframe['sieve3_exit_target_level'] = levels.where(levels.lt(close.mul(0.999))).max(axis=1)
                dataframe['sieve3_exit_invalidation_level'] = levels.where(levels.gt(close)).min(axis=1)
            else:
                dataframe['sieve3_exit_target_level'] = levels.where(levels.gt(close.mul(1.001))).min(axis=1)
                dataframe['sieve3_exit_invalidation_level'] = levels.where(levels.lt(close)).max(axis=1)
        else:
            dataframe['sieve3_exit_target_level'] = np.nan
            dataframe['sieve3_exit_invalidation_level'] = np.nan
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = None
        condition = self._entry_condition(dataframe)
        condition &= self._common_guards(dataframe)
        condition = apply_sieve2_optional_guards(self, dataframe, condition, SIDE)
        valid = condition.fillna(False) & dataframe['volume'].gt(0.0) & dataframe['close'].notna()
        if SIDE == 'short':
            dataframe.loc[valid, 'enter_short'] = 1
        else:
            dataframe.loc[valid, 'enter_long'] = 1
        dataframe.loc[valid, 'enter_tag'] = ENTRY_TAG
        return dataframe
    SOURCE_ENTRY_STEM = 'reframed_tlv2_res_break_vp_node_long_1h'
    FOCUSED_EXIT_CONTRACT = 'entry_target_full_or_zone_reversal'
    FOCUSED_SOURCE_PROFILE = {'side': 'long', 'min_level_distance': 0.001, 'targets': {'target_1': {'provider': 'nearest forward entry-family VP/TLV2 obstacle', 'long': {'level': 'sieve3_exit_target_level', 'available': None}}}, 'invalidation': {'provider': 'nearest crossed-side entry-family VP/TLV2 boundary', 'mode': 'level', 'long': {'level': 'sieve3_exit_invalidation_level', 'available': None}}}
    FOCUSED_EXIT_PLANS = {'touch_full': {'contract': 'entry_target_full_or_zone_reversal', 'name': 'touch_full', 'hard_stop_ratio': 0.03, 'max_hold_candles': 336}, 'close_full': {'contract': 'entry_target_full_or_zone_reversal', 'name': 'close_full', 'hard_stop_ratio': 0.03, 'max_hold_candles': 336}, 'zone_reversal': {'contract': 'entry_target_full_or_zone_reversal', 'name': 'zone_reversal', 'hard_stop_ratio': 0.03, 'max_hold_candles': 336}}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'high', 'low', 'open', 'sieve3_exit_invalidation_level', 'sieve3_exit_target_level')
    FOCUSED_STATE_VERSION = 1
    FOCUSED_STATE_KEY = 'sieve3_v2_focused:Sieve3V2EntryTargetFullOrZoneReversalFromReframedTlv2ResBreakVpNodeLong1H:entry_target_full_or_zone_reversal'
    LOCKED_BUY_PARAMS = {'use_sieve2_vp_guard': True, 'sieve2_vp_guard_mode': 'score_or_context', 'sieve2_vp_window': 96, 'sieve2_vp_bins': 36, 'sieve2_vp_score_min': 0.15, 'sieve2_vp_context_min': 0.15, 'use_sieve2_market_guard': True, 'sieve2_market_guard_mode': 'pressure_or_trend', 'sieve2_market_window': 24, 'sieve2_market_pressure_min': 0.03, 'sieve2_market_trend_min': 0.15, 'sieve2_rs_benchmark_pair': 'BTC/USDT:USDT', 'sieve2_rs_score_min': 0.3, 'use_volume_guard': False, 'volume_guard_window': 12, 'volume_ratio_min': 0.8, 'use_pressure_guard': False, 'pressure_window': 24, 'pressure_min': 0.1, 'use_body_direction_guard': False, 'use_close_direction_guard': False, 'vp_window': 96, 'vp_bins': 48, 'vp_score_min': 0.35, 'vp_context_min': 0.28, 'level_buffer_pct': 0.004, 'context_recent_bars': 2, 'liquidity_lookback': 96, 'min_pattern_score': 0.55, 'pivot_strength': 2, 'min_line_score': 0.4, 'max_distance_atr': 10.0, 'min_active_bars': 8, 'capitulation_move_pct': 0.04}
    ACTIVE_SELL_PARAMS = ('exit_plan', 'target_band_quarter_percent', 'reversal_confirmations', 'invalidation_band_quarter_percent', 'invalidation_confirmations')
    position_adjustment_enable = False
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    minimal_roi = {'0': 100.0}
    stoploss = -0.99
    exit_plan = CategoricalParameter(('touch_full', 'close_full', 'zone_reversal'), default='zone_reversal', space='sell', optimize=True, load=True)
    exit_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    target_band_quarter_percent = IntParameter(0, 12, default=2, space='sell', optimize=True, load=True)
    target_band_quarter_percent.batch_tags = ('family:exits', 'mode:sieve3_exit')
    reversal_confirmations = IntParameter(1, 3, default=1, space='sell', optimize=True, load=True)
    reversal_confirmations.batch_tags = ('family:exits', 'mode:sieve3_exit')
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
        plan['target_confirmation'] = {'touch_full': 'touch', 'close_full': 'close', 'zone_reversal': f'reversal_{int(self.reversal_confirmations.value)}'}[name]
        plan['target_band'] = float(self.target_band_quarter_percent.value) * 0.0025
        plan['invalidation_band'] = float(self.invalidation_band_quarter_percent.value) * 0.0025
        plan['invalidation_confirmations'] = int(self.invalidation_confirmations.value)
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
        fill_time = (getattr(order, 'order_filled_utc', None) if order is not None else None) or getattr(trade, 'date_entry_fill_utc', None) or getattr(trade, 'open_date_utc', None) or current_time
        fill_timestamp = self._focused_utc(fill_time)
        snapshot_date = self._focused_utc(row['date'])
        return {'version': self.FOCUSED_STATE_VERSION, 'contract': self.FOCUSED_EXIT_CONTRACT, 'plan': selected_plan, 'side': side, 'phase': 'pre_target', 'entry_rate': float(entry_rate), 'entry_filled_at': fill_timestamp.isoformat() if fill_timestamp is not None else None, 'entry_snapshot_candle': snapshot_date.isoformat() if snapshot_date is not None else None, 'levels': levels, 'target_states': {slot: {'status': 'available' if value is not None else 'unavailable', 'touched_at': None} for slot, value in levels.items() if slot.startswith('target_')}, 'invalidation_seen': False, 'guard_state': 'unknown', 'trigger_state': 'unknown', 'initial_stake': self._focused_float(getattr(trade, 'stake_amount', None)), 'favorable_rate': float(entry_rate), 'stop_price': None, 'terminal_pending': False, 'terminal_tag': None}

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
        state = self._focused_state(trade)
        if state is not None:
            target_state = (state.get('target_states') or {}).get('target_1', {})
            target_done = isinstance(target_state, Mapping) and target_state.get('status') in {'accepted', 'rejected', 'confirmed'}
            if state.get('invalidation_seen') or target_done or state.get('terminal_pending'):
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
        if not confirmation.startswith('reversal_'):
            if confirmation not in {'touch', 'close'}:
                raise ValueError(f'unsupported target confirmation: {confirmation}')
            confirmations = 0
        else:
            confirmations = int(confirmation.rsplit('_', 1)[1])
        streak = int(target_state.get('reversal_streak') or 0)
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
            streak = streak + 1 if reversal else 0
            target_state['reversal_streak'] = streak
            if streak >= confirmations:
                target_state['status'] = 'rejected'
                return True
        if touched_at is None:
            target_state['status'] = 'available'
        return False

    def _focused_invalidation_event(self, post_entry: DataFrame, state: dict[str, Any], plan: Mapping[str, Any]) -> bool:
        if state.get('invalidation_seen') or post_entry.empty:
            return bool(state.get('invalidation_seen'))
        level = self._focused_float((state.get('levels') or {}).get('invalidation'))
        if level is None:
            return False
        close = self._focused_number(post_entry, 'close')
        band = float(plan['invalidation_band'])
        breached = close.gt(level * (1.0 + band)) if state['side'] == 'short' else close.lt(level * (1.0 - band))
        streak = int(state.get('invalidation_streak') or 0)
        confirmations = int(plan['invalidation_confirmations'])
        for value in breached.fillna(False).tolist():
            streak = streak + 1 if bool(value) else 0
            state['invalidation_streak'] = streak
            if streak >= confirmations:
                state['invalidation_seen'] = True
                state['phase'] = 'invalidated'
                return True
        return False

    def _focused_contract_events(self, post_entry: DataFrame, state: dict[str, Any], plan: Mapping[str, Any], current_profit: float) -> dict[str, Any]:
        _ = current_profit
        return {'invalidation': self._focused_invalidation_event(post_entry, state, plan), 'target_1': self._focused_target_event(post_entry, state, plan, 'target_1')}

    def _focused_contract_decision(self, state: dict[str, Any], plan: Mapping[str, Any], events: Mapping[str, Any], current_profit: float) -> dict[str, Any]:
        _ = (state, plan, current_profit)
        if events.get('invalidation'):
            return self._focused_decision('full', 'focused_source_invalidation')
        if events.get('target_1'):
            return self._focused_decision('full', 'focused_entry_target')
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

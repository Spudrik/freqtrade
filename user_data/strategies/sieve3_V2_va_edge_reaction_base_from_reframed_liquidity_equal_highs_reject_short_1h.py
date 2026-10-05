from __future__ import annotations
from freqtrade.strategy import informative
import math
import pandas as pd
from collections.abc import Mapping
from copy import deepcopy
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
NOVEL_IDEA = True
ENTRY_MODE = 'entry_sieve2_reframed_liquidity_equal_highs_reject_short_1h'


TIMEFRAME = '1h'
CONTEXT_TIMEFRAME = ''
CONCEPT_FAMILY = 'liquidity'
CONCEPT = 'equal_highs_reject'
LTF_TRIGGER = 'reject'


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

SOURCE_ENTRY_CLASS = 'Sieve3ExitFromReframedLiquidityEqualHighsRejectShort1h'
SOURCE_STRATEGY = 'C:/FreqTradeStuff/user_data/strategies/sieve3_exit_from_reframed_liquidity_equal_highs_reject_short_1h.py:Sieve3ExitFromReframedLiquidityEqualHighsRejectShort1h'
LINEAGE_STATUS = 'verified'

EXIT_THEORY = 'va_edge_reaction_base'







EXIT_FAMILY = 'va_edge_reaction_base'
PRIMARY_TRIGGER = 'provider=liq_equal_reject_short equal-high sweep rejection event'
PRIMARY_GUARD = 'provider=locked common and Sieve2 optional guards'
PRIMARY_TARGET = 'named fixed partial and remainder objectives'
PRIMARY_INVALIDATION = 'named fixed hard stop'
TARGET_PROVIDER = 'closed-candle actual-fill-relative deterministic named level zone'
INVALIDATION_PROVIDER = 'coherent source invalidation when available, otherwise preserved 3% hard stop'
SOURCE_RESULT_BATCH = 'historical_promotion_lineage_unknown'
RESEARCH_PATH = 'sieve3_exit_final_generic_levels'
EXIT_HYPOTHESIS = 'Isolate va_edge on the base timeframe scope and exit only after directional weakening confirms around that named level zone.'
ACTIVE_SELL_PARAMS = ('zone_width_quarter_percent', 'weakening_signal', 'confirmation_profile')
SIEVE_STAGE = 'sieve3'
SIDE = 'short'
ENTRY_TAG = 'sieve2_reframed_liquidity_equal_highs_reject_short_1h'

class Sieve3V2VaEdgeReactionBaseFromReframedLiquidityEqualHighsRejectShort1H(IStrategy):
    SIEVE2_FUNDAMENTAL_REWORK = '20260605_nonprofitable_loosen_sparse'
    REWORK_HYPOTHESIS = 'Entry-only rerun after non-profitable revised-guard result: loosen_sparse; widen guard thresholds and adjust entry-strength search space.'
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
    use_volume_guard = True
    volume_guard_window = 12
    volume_ratio_min = 1.3
    # Legacy fixed-off entry parameter: use_pressure_guard=False (BooleanParameter, buy space).
    pressure_window = 12
    pressure_min = 0.15
    # Legacy fixed-off entry parameter: use_body_direction_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_close_direction_guard=False (BooleanParameter, buy space).
    vp_window = 96
    vp_bins = 48
    vp_score_min = 0.25
    vp_context_min = 0.28
    level_buffer_pct = 0.012
    context_recent_bars = 3
    liquidity_lookback = 24
    min_pattern_score = 0.72
    pivot_strength = 2
    min_line_score = 0.6
    max_distance_atr = 10.0
    min_active_bars = 8
    capitulation_move_pct = 0.06

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
        if bool(self.use_volume_guard):
            volume = _num(dataframe, 'volume').clip(lower=0.0)
            window = int(self.volume_guard_window)
            baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
            guard &= volume.ge(baseline.mul(float(self.volume_ratio_min)))
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

    def _generic_level_frame(self, dataframe: DataFrame, _timeframe: str) -> DataFrame:
        return add_volume_profile(dataframe, prefix='gsvp')
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        frame = self._sieve3_entry_populate_indicators(dataframe, metadata)
    
        if True:
            frame = self._generic_level_frame(frame, self.timeframe)
        return frame

    SOURCE_ENTRY_STEM = 'reframed_liquidity_equal_highs_reject_short_1h'
    FOCUSED_EXIT_CONTRACT = 'va_edge_reaction_base'
    FOCUSED_SOURCE_PROFILE = {'side': 'short', 'min_level_distance': 0.001}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'high', 'low', 'open')
    FOCUSED_STATE_VERSION = 1
    FOCUSED_STATE_KEY = 'sieve3_v2_generic:Sieve3V2VaEdgeReactionBaseFromReframedLiquidityEqualHighsRejectShort1H:va_edge_reaction_base'
    LOCKED_BUY_PARAMS = {'use_sieve2_vp_guard': True, 'sieve2_vp_guard_mode': 'score_or_context', 'sieve2_vp_window': 96, 'sieve2_vp_bins': 36, 'sieve2_vp_score_min': 0.15, 'sieve2_vp_context_min': 0.15, 'use_sieve2_market_guard': True, 'sieve2_market_guard_mode': 'pressure_or_trend', 'sieve2_market_window': 24, 'sieve2_market_pressure_min': 0.03, 'sieve2_market_trend_min': 0.15, 'sieve2_rs_benchmark_pair': 'BTC/USDT:USDT', 'sieve2_rs_score_min': 0.3, 'use_volume_guard': True, 'volume_guard_window': 12, 'volume_ratio_min': 1.3, 'use_pressure_guard': False, 'pressure_window': 12, 'pressure_min': 0.15, 'use_body_direction_guard': False, 'use_close_direction_guard': False, 'vp_window': 96, 'vp_bins': 48, 'vp_score_min': 0.25, 'vp_context_min': 0.28, 'level_buffer_pct': 0.012, 'context_recent_bars': 3, 'liquidity_lookback': 24, 'min_pattern_score': 0.72, 'pivot_strength': 2, 'min_line_score': 0.6, 'max_distance_atr': 10.0, 'min_active_bars': 8, 'capitulation_move_pct': 0.06}
    ACTIVE_SELL_PARAMS = ('zone_width_quarter_percent', 'weakening_signal', 'confirmation_profile')
    GENERIC_MODE = 'simple'
    GENERIC_LEVEL_FAMILY = 'va_edge'
    GENERIC_TIMEFRAME_SCOPE = 'base'
    GENERIC_AVAILABLE_TIMEFRAMES = ('1h', '4h', '8h', '1d', '3d')
    GENERIC_HVN_MIN_STRENGTH = 0.5
    GENERIC_MIN_LEVEL_DISTANCE = 0.001
    GENERIC_HARD_STOP_RATIO = 0.03
    position_adjustment_enable = False
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    minimal_roi = {'0': 100.0}
    stoploss = -0.99
    zone_width_quarter_percent = IntParameter(1, 12, default=4, space='sell', optimize=True, load=True)
    zone_width_quarter_percent.batch_tags = ('family:exits', 'mode:sieve3_exit')
    weakening_signal = CategoricalParameter(('opposite_body', 'close_momentum', 'directional_pressure', 'zone_rejection'), default='close_momentum', space='sell', optimize=True, load=True)
    weakening_signal.batch_tags = ('family:exits', 'mode:sieve3_exit')
    confirmation_profile = CategoricalParameter(('1_of_1', '2_of_2', '2_of_3', '3_of_3'), default='2_of_3', space='sell', optimize=True, load=True)
    confirmation_profile.batch_tags = ('family:exits', 'mode:sieve3_exit')

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        dataframe['exit_tag'] = None
        return dataframe

    @staticmethod
    def _generic_utc(value: Any) -> pd.Timestamp | None:
        if value is None:
            return None
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize('UTC')
        return timestamp.tz_convert('UTC')

    @staticmethod
    def _generic_float(value: Any) -> float | None:
        if value is None or bool(pd.isna(value)):
            return None
        number = float(value)
        return number if math.isfinite(number) else None

    def _generic_closed_rows(self, frame: DataFrame, current_time: Any, cursor: Any=None) -> DataFrame:
        if frame is None or frame.empty or 'date' not in frame.columns:
            raise RuntimeError('generic level exit requires a dated analyzed dataframe')
        dates = frame['date']
        if not pd.api.types.is_datetime64_any_dtype(dates.dtype):
            raise TypeError('generic level exit requires Freqtrade datetime values in the date column')
        now = self._generic_utc(current_time)
        if now is None:
            raise ValueError('current_time is required for closed-candle evaluation')
        cutoff = now - pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit='m')
        normalized_cursor = self._generic_utc(cursor)
        timezone = dates.dt.tz
        if timezone is None:
            cutoff = cutoff.tz_localize(None)
            normalized_cursor = normalized_cursor.tz_localize(None) if normalized_cursor is not None else None
        else:
            cutoff = cutoff.tz_convert(timezone)
            normalized_cursor = normalized_cursor.tz_convert(timezone) if normalized_cursor is not None else None
        start = int(dates.searchsorted(normalized_cursor, side='right')) if normalized_cursor is not None else 0
        stop = int(dates.searchsorted(cutoff, side='right'))
        return frame.iloc[min(start, stop):stop]

    def _generic_analyzed_frame(self, pair: str) -> DataFrame:
        if getattr(self, 'dp', None) is None:
            raise RuntimeError("generic level exit requires Freqtrade's data provider")
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if frame is None or frame.empty:
            raise RuntimeError('generic level exit requires a non-empty analyzed dataframe')
        missing = sorted(set(self.FOCUSED_REQUIRED_COLUMNS) - set(frame.columns))
        if missing:
            raise KeyError(f'generic level source is missing exact columns: {missing}')
        return frame

    def _generic_side(self, trade: Any) -> str:
        side = 'short' if bool(getattr(trade, 'is_short', False)) else 'long'
        declared = str(self.FOCUSED_SOURCE_PROFILE['side'])
        if declared != 'both' and declared != side:
            raise ValueError(f'trade side {side!r} violates source side {declared!r}')
        return side

    def _generic_state(self, trade: Any) -> dict[str, Any] | None:
        cache_attr = f'_sieve3_generic_state_{self.FOCUSED_STATE_KEY}'
        state = getattr(trade, cache_attr, None)
        if state is None:
            state = trade.get_custom_data(key=self.FOCUSED_STATE_KEY)
        if state is None:
            return None
        if not isinstance(state, dict):
            raise ValueError('generic level trade state must be a mapping')
        if state.get('version') != self.FOCUSED_STATE_VERSION:
            raise ValueError('generic level trade state version mismatch')
        if state.get('contract') != self.FOCUSED_EXIT_CONTRACT:
            raise ValueError('generic level trade state contract mismatch')
        if getattr(trade, cache_attr, None) is None:
            setattr(trade, cache_attr, deepcopy(state))
        return deepcopy(state)

    def _generic_save_state(self, trade: Any, state: Mapping[str, Any]) -> None:
        cache_attr = f'_sieve3_generic_state_{self.FOCUSED_STATE_KEY}'
        snapshot = deepcopy(dict(state))
        setattr(trade, cache_attr, snapshot)
        trade.set_custom_data(key=self.FOCUSED_STATE_KEY, value=deepcopy(snapshot))

    def _generic_source_invalidation(self, row: Series, side: str) -> float | None:
        provider = self.FOCUSED_SOURCE_PROFILE.get('invalidation')
        if not isinstance(provider, Mapping):
            return None
        binding = provider.get(side)
        if not isinstance(binding, Mapping):
            return None
        available = binding.get('available')
        if available:
            value = row.get(str(available))
            if bool(pd.isna(value)) or not bool(value):
                return None
        column = binding.get('level')
        return self._generic_float(row.get(str(column))) if column else None

    def _generic_column(self, base: str, timeframe: str, btc: bool=False) -> str:
        if btc:
            return f'gsbtc_{base}_{timeframe}'
        return base if timeframe == self.timeframe else f'gsl_{base}_{timeframe}'

    def _generic_value(self, row: Series, base: str, timeframe: str, btc: bool=False) -> float | None:
        column = self._generic_column(base, timeframe, btc)
        if column not in row.index:
            raise KeyError(f'generic level column is missing: {column}')
        return self._generic_float(row[column])

    def _generic_candidate(self, row: Series, side: str, timeframe: str, family: str, btc: bool=False) -> dict[str, Any] | None:
        if family == 'va_edge':
            base = 'gsvp_prior_val' if side == 'short' else 'gsvp_prior_vah'
            level = self._generic_value(row, base, timeframe, btc)
            metadata: dict[str, Any] = {}
            score = 3

        elif family == 'hvn_strength':
            base = 'gsvp_hvn_below' if side == 'short' else 'gsvp_hvn_above'
            level = self._generic_value(row, base, timeframe, btc)
            strength = self._generic_value(row, f'{base}_strength', timeframe, btc)
            threshold = self.GENERIC_HVN_MIN_STRENGTH
            if self.GENERIC_MODE == 'simple':
                threshold = float(self.hvn_strength_tenth_units.value) * 0.1
            if strength is None or strength < threshold:
                return None
            metadata = {'strength': strength}
            score = 3 + int(strength >= 0.85)
        else:
            raise ValueError(f'unsupported primitive family: {family}')
        if level is None or level <= 0.0:
            return None
        return {'level': level, 'family': family, 'timeframe': timeframe, 'score': score, 'metadata': metadata}


    @staticmethod
    def _generic_forward(candidates: list[dict[str, Any]], side: str, reference: float) -> list[dict[str, Any]]:
        minimum = 0.001
        if side == 'short':
            valid = [item for item in candidates if float(item['level']) < reference * (1.0 - minimum)]
        else:
            valid = [item for item in candidates if float(item['level']) > reference * (1.0 + minimum)]
        return sorted(valid, key=lambda item: abs(float(item['level']) - reference))

    def _generic_primitives(self, row: Series, side: str, timeframe: str, reference: float, btc: bool=False) -> list[dict[str, Any]]:
        candidates = [
            candidate
            for family in ('va_edge', 'hvn_strength')
            for candidate in [self._generic_candidate(row, side, timeframe, family, btc)]
            if candidate is not None
        ]
        return self._generic_forward(candidates, side, reference)

    def _generic_btc_strength(self, row: Series, side: str, band: float) -> int:
        if self.GENERIC_MODE == 'simple':
            return 0
        near: list[dict[str, Any]] = []
        for timeframe in self.GENERIC_AVAILABLE_TIMEFRAMES:
            close = self._generic_value(row, 'close', timeframe, True)
            if close is None or close <= 0.0:
                continue
            for candidate in self._generic_primitives(row, side, timeframe, close, True):
                if abs(float(candidate['level']) / close - 1.0) <= band:
                    near.append(candidate)
        if not near:
            return 0
        families = {str(item['family']) for item in near}
        timeframes = {str(item['timeframe']) for item in near}
        return 2 if len(families) > 1 or len(timeframes) > 1 else 1

    def _generic_scored_zones(self, row: Series, side: str, reference: float, pair: str) -> list[dict[str, Any]]:
        band = float(self.zone_width_quarter_percent.value) * 0.0025
        primitives = [
            candidate
            for timeframe in self.GENERIC_AVAILABLE_TIMEFRAMES
            for candidate in self._generic_primitives(row, side, timeframe, reference)
        ]
        retained = primitives
        zones: list[dict[str, Any]] = []
        for item in retained:
            zones.append(
                {
                    'center': float(item['level']),
                    'lower': float(item['level']) * (1.0 - band),
                    'upper': float(item['level']) * (1.0 + band),
                    'score': int(item['score']),
                    'families': [str(item['family'])],
                    'timeframes': [str(item['timeframe'])],
                    'members': [item],
                }
            )
        seen: set[tuple[str, ...]] = set()
        for anchor in primitives:
            anchor_level = float(anchor['level'])
            members = [item for item in primitives if abs(float(item['level']) / anchor_level - 1.0) <= band]
            key = tuple(sorted(f"{item['family']}:{item['timeframe']}:{float(item['level']):.10g}" for item in members))
            if len(members) < 2 or key in seen:
                continue
            seen.add(key)
            levels = sorted(float(item['level']) for item in members)
            families = {str(item['family']) for item in members}
            timeframes = {str(item['timeframe']) for item in members}
            score = max(int(item['score']) for item in members)
            score += int(len(families) > 1)
            if len(timeframes) > 1:
                score += (
                    int(self.mtf_alignment_bonus.value)
                    if self.GENERIC_MODE == 'scored_full'
                    else 1
                )
            zones.append(
                {
                    'center': float(np.median(levels)),
                    'lower': min(levels) * (1.0 - band),
                    'upper': max(levels) * (1.0 + band),
                    'score': score,
                    'families': sorted(families),
                    'timeframes': sorted(timeframes, key=timeframe_to_minutes),
                    'members': members,
                }
            )
        if not str(pair).upper().startswith('BTC/'):
            btc_strength = self._generic_btc_strength(row, side, band)
            configured = (
                int(self.btc_same_side_bonus.value)
                if self.GENERIC_MODE == 'scored_full'
                else 1
            )
            for zone in zones:
                zone['score'] += min(configured, btc_strength)
                zone['btc_context_strength'] = btc_strength
        minimum_score = (
            int(self.minimum_level_score.value)
            if self.GENERIC_MODE == 'scored_full'
            else 4
        )
        return [zone for zone in zones if int(zone['score']) >= minimum_score]

    def _generic_simple_zones(self, row: Series, side: str, reference: float) -> list[dict[str, Any]]:
        timeframe = self.timeframe if self.GENERIC_TIMEFRAME_SCOPE == 'base' else str(self.target_timeframe.value)
        candidate = self._generic_candidate(row, side, timeframe, self.GENERIC_LEVEL_FAMILY)
        candidates = [candidate] if candidate is not None else []
        band = float(self.zone_width_quarter_percent.value) * 0.0025
        return [
            {
                'center': float(item['level']),
                'lower': float(item['level']) * (1.0 - band),
                'upper': float(item['level']) * (1.0 + band),
                'score': int(item['score']),
                'families': [str(item['family'])],
                'timeframes': [str(item['timeframe'])],
                'members': [item],
            }
            for item in self._generic_forward(candidates, side, reference)
        ]

    def _generic_select_zone(self, row: Series, side: str, reference: float, pair: str, passed: list[float]) -> dict[str, Any] | None:
        zones = self._generic_simple_zones(row, side, reference) if self.GENERIC_MODE == 'simple' else self._generic_scored_zones(row, side, reference, pair)
        band = float(self.zone_width_quarter_percent.value) * 0.0025
        zones = [zone for zone in zones if not any(abs(float(zone['center']) / old - 1.0) <= band for old in passed if old > 0.0)]
        if not zones:
            return None
        if self.GENERIC_MODE == 'simple':
            return min(zones, key=lambda zone: abs(float(zone['center']) - reference))
        mode = str(self.level_order_mode.value) if self.GENERIC_MODE == 'scored_full' else 'highest_score'
        if mode == 'nearest_qualified':
            return min(zones, key=lambda zone: abs(float(zone['center']) - reference))
        if mode == 'highest_score':
            return min(zones, key=lambda zone: (-int(zone['score']), abs(float(zone['center']) - reference)))
        if mode == 'higher_tf_priority':
            return min(zones, key=lambda zone: (-max(timeframe_to_minutes(tf) for tf in zone['timeframes']), -int(zone['score']), abs(float(zone['center']) - reference)))
        raise ValueError(f'unsupported level order mode: {mode}')

    def _generic_new_state(self, pair: str, trade: Any, current_time: Any) -> dict[str, Any]:
        side = self._generic_side(trade)
        entry_rate = self._generic_float(getattr(trade, 'open_rate', None))
        if entry_rate is None or entry_rate <= 0.0:
            raise ValueError('generic level exit requires a positive actual fill rate')
        filled_at = self._generic_utc(getattr(trade, 'date_entry_fill_utc', None) or getattr(trade, 'open_date_utc', None) or current_time)
        frame = self._generic_closed_rows(self._generic_analyzed_frame(pair), filled_at)
        row = frame.iloc[-1] if not frame.empty else None
        zone = self._generic_select_zone(row, side, entry_rate, pair, []) if row is not None else None
        invalidation = self._generic_source_invalidation(row, side) if row is not None else None
        cursor = self._generic_utc(row['date']).isoformat() if row is not None else None
        return {
            'version': self.FOCUSED_STATE_VERSION,
            'contract': self.FOCUSED_EXIT_CONTRACT,
            'side': side,
            'entry_rate': entry_rate,
            'entry_filled_at': filled_at.isoformat() if filled_at is not None else None,
            'last_processed_candle': cursor,
            'last_callback_bucket': None,
            'zone': zone,
            'zone_touched': False,
            'reaction_history': [],
            'passed_centers': [],
            'protected_levels': [],
            'source_invalidation': invalidation,
            'stop_price': None,
            'terminal_tag': None,
            'pending_partial': None,
            'completed_partials': int(getattr(trade, 'nr_of_successful_exits', 0) or 0),
            'initial_stake': self._generic_float(getattr(trade, 'stake_amount', None)),
            'previous_close': entry_rate,
        }

    def _generic_ensure_state(self, pair: str, trade: Any, current_time: Any) -> dict[str, Any]:
        state = self._generic_state(trade)
        if state is None:
            state = self._generic_new_state(pair, trade, current_time)
            self._generic_save_state(trade, state)
        return state

    @staticmethod
    def _generic_pressure(row: Series) -> float:
        open_ = float(row['open'])
        high = float(row['high'])
        low = float(row['low'])
        close = float(row['close'])
        spread = max(high - low, 1e-12)
        body = (close - open_) / spread
        location = (close - low) / spread * 2.0 - 1.0
        return max(-1.0, min(1.0, (body + location) / 2.0))

    def _generic_weakening(self, row: Series, state: Mapping[str, Any]) -> tuple[bool, bool]:
        side = str(state['side'])
        zone = state['zone']
        open_ = float(row['open'])
        close = float(row['close'])
        previous = float(state.get('previous_close') or state['entry_rate'])
        opposite_body = close > open_ if side == 'short' else close < open_
        close_momentum = close > previous if side == 'short' else close < previous
        pressure = self._generic_pressure(row)
        directional_pressure = pressure > 0.0 if side == 'short' else pressure < 0.0
        zone_rejection = close > float(zone['upper']) if side == 'short' else close < float(zone['lower'])
        if self.GENERIC_MODE == 'simple':
            selected = str(self.weakening_signal.value)
            values = {
                'opposite_body': opposite_body,
                'close_momentum': close_momentum,
                'directional_pressure': directional_pressure,
                'zone_rejection': zone_rejection,
            }
            return bool(values[selected]), zone_rejection
        return sum((opposite_body, close_momentum, directional_pressure)) >= 2, zone_rejection

    def _generic_reaction_confirmed(self, row: Series, state: dict[str, Any]) -> bool:
        weakening, rejection = self._generic_weakening(row, state)
        if self.GENERIC_MODE == 'scored_full' and str(self.reaction_profile.value) == 'rejection_close':
            return rejection
        profile = str(self.confirmation_profile.value) if self.GENERIC_MODE == 'simple' else str(self.reaction_profile.value)
        required, window = {
            '1_of_1': (1, 1),
            '2_of_2': (2, 2),
            '2_of_3': (2, 3),
            '3_of_3': (3, 3),
        }[profile]
        history = list(state.get('reaction_history') or [])
        history.append(bool(weakening))
        state['reaction_history'] = history[-3:]
        recent = history[-window:]
        return len(recent) == window and sum(bool(value) for value in recent) >= required

    def _generic_ratchet(self, state: dict[str, Any], level: float) -> None:
        if self.GENERIC_MODE != 'scored_partial':
            return
        protected = list(state.get('protected_levels') or [])
        protected.append(float(level))
        state['protected_levels'] = protected
        lag = int(self.ratchet_lag_levels.value)
        if lag <= 0:
            return
        candidate = float(state['entry_rate']) if len(protected) <= lag else protected[len(protected) - lag - 1]
        current = self._generic_float(state.get('stop_price'))
        state['stop_price'] = candidate if current is None else min(current, candidate) if state['side'] == 'short' else max(current, candidate)

    def _generic_reaction_action(self, state: dict[str, Any]) -> None:
        zone = state.get('zone')
        if not isinstance(zone, Mapping):
            return
        center = float(zone['center'])
        if self.GENERIC_MODE != 'scored_partial':
            state['terminal_tag'] = f"generic_{self.GENERIC_LEVEL_FAMILY}_reaction_full"
            return
        configured = [float(self.partial_1_five_percent_units.value) * 0.05]
        second = int(self.partial_2_five_percent_units.value)
        if second > 0:
            configured.append(float(second) * 0.05)
        completed = int(state.get('completed_partials') or 0)
        if completed < len(configured):
            state['pending_partial'] = {
                'fraction': configured[completed],
                'tag': f'generic_level_partial_{completed + 1}',
                'requested': False,
                'ratchet_level': center,
            }
        else:
            state['terminal_tag'] = 'generic_level_progression_remainder'
        passed = list(state.get('passed_centers') or [])
        passed.append(center)
        state['passed_centers'] = passed
        state['zone'] = None
        state['zone_touched'] = False
        state['reaction_history'] = []

    def _generic_process_row(self, row: Series, pair: str, state: dict[str, Any]) -> None:
        side = str(state['side'])
        close = float(row['close'])
        invalidation = self._generic_float(state.get('source_invalidation'))
        if invalidation is not None and ((side == 'short' and close >= invalidation) or (side == 'long' and close <= invalidation)):
            state['terminal_tag'] = 'generic_source_invalidation'
            return
        if state.get('zone') is None:
            state['zone'] = self._generic_select_zone(row, side, close, pair, list(state.get('passed_centers') or []))
            state['zone_touched'] = False
            state['reaction_history'] = []
        zone = state.get('zone')
        if not isinstance(zone, Mapping):
            state['previous_close'] = close
            return
        high = float(row['high'])
        low = float(row['low'])
        touched = low <= float(zone['upper']) if side == 'short' else high >= float(zone['lower'])
        clean_pass = close < float(zone['lower']) if side == 'short' else close > float(zone['upper'])
        if clean_pass:
            center = float(zone['center'])
            passed = list(state.get('passed_centers') or [])
            passed.append(center)
            state['passed_centers'] = passed
            self._generic_ratchet(state, center)
            state['zone'] = self._generic_select_zone(row, side, close, pair, passed)
            state['zone_touched'] = False
            state['reaction_history'] = []
        elif touched or bool(state.get('zone_touched')):
            state['zone_touched'] = True
            if self._generic_reaction_confirmed(row, state):
                self._generic_reaction_action(state)
        state['previous_close'] = close

    def _generic_sync_partial(self, trade: Any, state: dict[str, Any]) -> bool:
        native = int(getattr(trade, 'nr_of_successful_exits', 0) or 0)
        completed = int(state.get('completed_partials') or 0)
        pending = state.get('pending_partial')
        if native > completed:
            if isinstance(pending, Mapping):
                ratchet_level = self._generic_float(pending.get('ratchet_level'))
                if ratchet_level is not None:
                    self._generic_ratchet(state, ratchet_level)
            state['completed_partials'] = native
            state['pending_partial'] = None
            return True
        if isinstance(pending, Mapping) and bool(pending.get('requested')) and native < completed:
            raise ValueError('native partial-exit count moved backwards')
        return False

    def _generic_context(self, pair: str, trade: Any, current_time: Any) -> dict[str, Any]:
        state = self._generic_ensure_state(pair, trade, current_time)
        if self._generic_sync_partial(trade, state):
            self._generic_save_state(trade, state)
        if state.get('pending_partial'):
            self._generic_save_state(trade, state)
            return state
        bucket = self._generic_utc(current_time)
        bucket_key = bucket.floor(f'{max(1, timeframe_to_minutes(self.timeframe))}min').isoformat() if bucket is not None else None
        if state.get('last_callback_bucket') == bucket_key:
            return state
        state['last_callback_bucket'] = bucket_key
        if state.get('terminal_tag'):
            self._generic_save_state(trade, state)
            return state
        cursor = self._generic_utc(state.get('last_processed_candle'))
        rows = self._generic_closed_rows(self._generic_analyzed_frame(pair), current_time, cursor)
        for _, row in rows.iterrows():
            self._generic_process_row(row, pair, state)
            state['last_processed_candle'] = self._generic_utc(row['date']).isoformat()
            if state.get('terminal_tag') or state.get('pending_partial'):
                break
        self._generic_save_state(trade, state)
        return state

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | bool | None:
        _ = (current_rate, current_profit, kwargs)
        state = self._generic_context(pair, trade, current_time)
        return str(state['terminal_tag']) if state.get('terminal_tag') else None

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float | None:
        _ = (current_profit, after_fill, kwargs)
        state = self._generic_context(pair, trade, current_time)
        entry = float(state['entry_rate'])
        hard = entry * (1.0 + self.GENERIC_HARD_STOP_RATIO if state['side'] == 'short' else 1.0 - self.GENERIC_HARD_STOP_RATIO)
        stop = hard
        invalidation = self._generic_float(state.get('source_invalidation'))
        if invalidation is not None:
            stop = min(stop, invalidation) if state['side'] == 'short' else max(stop, invalidation)
        ratchet = self._generic_float(state.get('stop_price'))
        if ratchet is not None:
            stop = min(stop, ratchet) if state['side'] == 'short' else max(stop, ratchet)
        return stoploss_from_absolute(stop, current_rate, is_short=state['side'] == 'short', leverage=float(getattr(trade, 'leverage', 1.0) or 1.0))

    def adjust_trade_position(self, trade: Any, current_time: datetime, current_rate: float, current_profit: float, min_stake: float | None, max_stake: float, current_entry_rate: float, current_exit_rate: float, current_entry_profit: float, current_exit_profit: float, **kwargs: Any) -> float | None | tuple[float | None, str | None]:
        _ = (current_rate, current_profit, min_stake, max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, kwargs)
        if self.GENERIC_MODE != 'scored_partial':
            return None
        state = self._generic_context(trade.pair, trade, current_time)
        pending = state.get('pending_partial')
        if not isinstance(pending, Mapping) or bool(pending.get('requested')):
            return None
        initial = self._generic_float(state.get('initial_stake')) or self._generic_float(getattr(trade, 'stake_amount', None))
        current = self._generic_float(getattr(trade, 'stake_amount', None))
        if initial is None or current is None or current <= 0.0:
            return None
        amount = min(current, initial * float(pending['fraction']))
        if amount <= 0.0:
            return None
        pending = dict(pending)
        pending['requested'] = True
        state['pending_partial'] = pending
        self._generic_save_state(trade, state)
        return -amount, str(pending['tag'])

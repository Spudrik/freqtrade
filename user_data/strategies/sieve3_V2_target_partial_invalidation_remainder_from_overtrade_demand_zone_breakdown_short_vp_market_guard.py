from __future__ import annotations
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
from user_data.Indicators.complex_volume_profile import add_volume_profile
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
    if always_on:
        guarded &= _sieve2_vp_guard(
            dataframe,
            normalized_side,
            str(_sieve2_param(strategy, "sieve2_vp_guard_mode", "score_or_context")),
            float(_sieve2_param(strategy, "sieve2_vp_score_min", 0.25)),
            float(_sieve2_param(strategy, "sieve2_vp_context_min", 0.28)),
        )
    if always_on:
        guarded &= _sieve2_market_guard(
            dataframe,
            normalized_side,
            str(_sieve2_param(strategy, "sieve2_market_guard_mode", "pressure_or_trend")),
            float(_sieve2_param(strategy, "sieve2_market_pressure_min", 0.07)),
            float(_sieve2_param(strategy, "sieve2_market_trend_min", 0.25)),
            float(0.3),
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
ENTRY_MODE = 'entry_overtrade_demand_zone_breakdown_short_vp_market_guard'

UPDATE_HYPOTHESIS = 'High-trade entry may contain a real directional edge; require stronger volume-profile/market context so HTF or broader context carries confidence and the original trigger supplies execution.'

CONCEPT = 'demand_breakdown'
PERIOD_KIND = 'fixed'
GUARD_MODE_CHOICES = ['direction', 'score', 'context', 'score_or_context', 'balance']
PERIOD_CHOICES = ['day', 'week', 'month']
PRICE_SOURCE_CHOICES = ['close', 'hl2', 'hlc3', 'ohlc4']
VP_COLUMNS = ['score_long', 'score_short', 'context_score_bull', 'context_score_bear', 'context_score_balance', 'market_context']


def _num(frame: DataFrame, column: str, default: float | Series=...) -> Series:
    if column not in frame.columns:
        if default is ...:
            raise KeyError(f'missing required column: {column!r}')
        return pd.to_numeric(
            pd.Series(default, index=frame.index), errors='coerce'
        ).replace([np.inf, -np.inf], np.nan)
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)

SOURCE_ENTRY_STAGE = 'sieve2'
SOURCE_ENTRY_CLASS = 'Sieve3ExitFromOvertradeDemandZoneBreakdownShortVpMarketGuard'
SOURCE_STRATEGY = 'user_data/strategies/sieve3_exit_from_overtrade_demand_zone_breakdown_short_vp_market_guard.py:Sieve3ExitFromOvertradeDemandZoneBreakdownShortVpMarketGuard'
LINEAGE_STATUS = 'verified'

EXIT_THEORY = 'target_partial_invalidation_remainder'







EXIT_FAMILY = 'target_partial_invalidation_remainder'
PRIMARY_TRIGGER = 'closed-candle breakdown through demand_zone_low with the selected zone-age and execution guards'
PRIMARY_GUARD = 'selected Sieve2 VP and market guard context; VP is contextual and not a target'
PRIMARY_TARGET = 'entry-frozen nearest forward VP/TLV2/structural obstacle'
PRIMARY_INVALIDATION = 'demand_zone_breakdown_reclaim'
TARGET_PROVIDER = 'nearest_forward_vp_tlv2_or_structure_obstacle'
INVALIDATION_PROVIDER = 'demand_zone_breakdown_reclaim'
SOURCE_RESULT_BATCH = 'historical_promotion_lineage_unknown'
RESEARCH_PATH = 'sieve3_exit_target_partial_invalidation_remainder'
EXIT_HYPOTHESIS = 'An entry-frozen forward structural target should support full or staged exits while the crossed entry level remains the source-specific invalidation.'
ACTIVE_SELL_PARAMS = ('target_band_quarter_percent', 'partial_five_percent_units', 'invalidation_band_quarter_percent', 'invalidation_confirmations', 'stop_after_partial')
SIEVE_STAGE = 'sieve3'
SIDE = 'short'
ENTRY_TAG = 'overtrade_demand_zone_breakdown_short_vp_market_guard'

class Sieve3V2TargetPartialInvalidationRemainderFromOvertradeDemandZoneBreakdownShortVpMarketGuard(IStrategy):
    SIEVE2_ALWAYS_ON_GUARDS = True
    '\n    Sieve2 entry concept: demand_zone_breakdown_short.\n\n    This is entry-quality research only. It has no custom exit, no DCA, and no\n    position-management hooks. Optional guards are hyperoptable inside this\n    standalone strategy so Explorer can test whether they improve the trigger.\n    '
    INTERFACE_VERSION = 3
    timeframe = '1h'
    startup_candle_count = 336
    process_only_new_candles = True
    can_short = True
    max_entry_position_adjustment = 0
    # Legacy inactive entry parameter: use_sieve2_vp_guard=True (guards are fixed always-on).
    sieve2_vp_guard_mode = 'score_or_context'
    sieve2_vp_window = 96
    sieve2_vp_bins = 36
    sieve2_vp_score_min = 0.15
    sieve2_vp_context_min = 0.15
    # Legacy inactive entry parameter: use_sieve2_market_guard=True (guards are fixed always-on).
    sieve2_market_guard_mode = 'pressure_or_trend'
    sieve2_market_window = 24
    sieve2_market_pressure_min = 0.03
    sieve2_market_trend_min = 0.15
    # Legacy inactive entry parameter: sieve2_rs_benchmark_pair='BTC/USDT:USDT' (fixed guard mode does not consume it).
    # Legacy inactive entry parameter: sieve2_rs_score_min=0.3 (fixed guard mode does not consume it).
    breakout_buffer_pct = 0.003
    reclaim_buffer_pct = 0.006
    sweep_buffer_pct = 0.006
    zone_near_pct = 0.01
    rolling_level_lookback = 72
    equal_level_lookback = 72
    equal_level_tolerance_pct = 0.006
    equal_level_min_touches = 2
    confluence_period = 'day'
    avwap_anchor_lookback = 120
    avwap_band_mult = 1.25
    zone_impulse_window = 36
    zone_impulse_atr_min = 0.8
    zone_body_fraction_min = 0.55
    zone_volume_ratio_min = 1.1
    zone_max_age_bars = 168
    # Legacy inactive entry parameter: use_volume_guard=False (enable gate is fixed off).
    volume_window = 12
    volume_ratio_min = 0.8
    pressure_min = 0.35
    # Legacy inactive entry parameter: use_close_direction_guard=False (enable gate is fixed off).
    vp_window = 96
    vp_bins = 48
    vp_value_area_pct = 0.7
    vp_price_source = 'hlc3'
    vp_smooth_bins = 3
    vp_hvn_threshold = 0.7
    vp_lvn_threshold = 0.35
    vp_pressure_delta_min = 0.05
    vp_node_near_pct = 0.01
    vp_volume_percentile_min = 0.55
    vp_score_window = 48
    vp_fast_traverse_atr_mult = 1.2
    vp_entry_score_margin = 0.02
    # Legacy inactive entry parameter: use_vp_1h_guard=False (enable gate is fixed off).
    vp_guard_mode = 'score_or_context'
    vp_score_min = 0.25
    vp_context_min = 0.28
    use_vp_4h_guard = True
    vp_4h_window = 48
    vp_4h_bins = 36
    # Legacy inactive entry parameter: use_vp_1d_guard=False (enable gate is fixed off).
    vp_1d_window = 30
    vp_1d_bins = 36

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = (pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs)
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        if not getattr(self, 'dp', None):
            return []
        pairs = self.dp.current_whitelist()
        return [(pair, '4h') for pair in pairs] + [(pair, '1d') for pair in pairs]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self._add_volume_pressure(dataframe)
        dataframe = self._add_prior_period_levels(dataframe)
        dataframe = self._add_rolling_levels(dataframe)
        dataframe = self._add_avwap(dataframe)
        dataframe = self._add_supply_demand(dataframe)
        dataframe = self._add_liquidity_levels(dataframe)
        dataframe = self._add_volume_profile(dataframe, 'vp', int(self.vp_window), int(self.vp_bins))
        dataframe = self._merge_informative_vp(dataframe, metadata, '4h', 'vp4h', int(self.vp_4h_window), int(self.vp_4h_bins))
        dataframe = self._merge_informative_vp(dataframe, metadata, '1d', 'vp1d', int(self.vp_1d_window), int(self.vp_1d_bins))
        dataframe = add_sieve2_guard_indicators(self, dataframe, metadata)
        dataframe = self._add_sieve3_forward_target(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = None
        condition = self._entry_condition(dataframe)
        if bool(self.use_vp_4h_guard):
            condition &= self._vp_guard(dataframe, 'vp4h', SIDE, str(self.vp_guard_mode), float(self.vp_score_min), float(self.vp_context_min))
        condition = apply_sieve2_optional_guards(self, dataframe, condition, getattr(self, 'ENTRY_SIDE', 'short'))
        valid = condition.fillna(False) & dataframe['volume'].gt(0.0) & dataframe['close'].notna()
        dataframe.loc[valid, 'enter_short'] = 1
        dataframe.loc[valid, 'enter_tag'] = ENTRY_TAG
        return dataframe

    @staticmethod
    def _atr(dataframe: DataFrame, period: int=14) -> Series:
        high = _num(dataframe, 'high')
        low = _num(dataframe, 'low')
        close = _num(dataframe, 'close')
        previous_close = close.shift(1)
        true_range = pd.concat([high.sub(low), high.sub(previous_close).abs(), low.sub(previous_close).abs()], axis=1).max(axis=1)
        return true_range.rolling(period, min_periods=max(2, period // 2)).mean()

    @staticmethod
    def _bars_since(signal: Series) -> Series:
        clean = pd.Series(signal, index=signal.index).fillna(False).astype(bool)
        positions = pd.Series(np.arange(len(clean), dtype='float64'), index=clean.index)
        last_hit = positions.where(clean).ffill()
        return positions.sub(last_hit).fillna(9999.0)

    @staticmethod
    def _period_key(dates: Series, period: str) -> Series:
        if period == 'week':
            iso = dates.dt.isocalendar()
            return iso['year'].astype('string').str.cat(iso['week'].astype('string').str.zfill(2), sep='-')
        if period == 'month':
            return dates.dt.strftime('%Y-%m')
        return dates.dt.strftime('%Y-%m-%d')

    def _selected_period(self) -> str:
        if PERIOD_KIND == 'select':
            return str(self.confluence_period)
        if PERIOD_KIND in PERIOD_CHOICES:
            return PERIOD_KIND
        return 'day'

    def _add_volume_pressure(self, dataframe: DataFrame) -> DataFrame:
        volume_window = int(self.volume_window)
        volume_mean = _num(dataframe, 'volume').rolling(volume_window, min_periods=max(2, volume_window // 3)).mean()
        candle_range = _num(dataframe, 'high').sub(_num(dataframe, 'low')).replace(0.0, np.nan)
        close_location = _num(dataframe, 'close').sub(_num(dataframe, 'low')).div(candle_range).clip(0.0, 1.0)
        dataframe['entry_close_location'] = close_location
        dataframe['entry_volume_ratio'] = _num(dataframe, 'volume').div(volume_mean.replace(0.0, np.nan))
        dataframe['entry_pressure'] = close_location.sub(0.5).mul(2.0)
        return dataframe

    def _add_prior_period_levels(self, dataframe: DataFrame) -> DataFrame:
        dates = pd.to_datetime(dataframe['date'], utc=True, errors='coerce')
        high = _num(dataframe, 'high')
        low = _num(dataframe, 'low')
        for period in PERIOD_CHOICES:
            key = self._period_key(dates, period)
            grouped = pd.DataFrame({'period': key, 'high': high, 'low': low}).groupby('period', sort=True).agg(period_high=('high', 'max'), period_low=('low', 'min'))
            grouped['prior_high'] = grouped['period_high'].shift(1)
            grouped['prior_low'] = grouped['period_low'].shift(1)
            dataframe[f'prior_{period}_high'] = key.map(grouped['prior_high']).astype('float64')
            dataframe[f'prior_{period}_low'] = key.map(grouped['prior_low']).astype('float64')
        return dataframe

    def _add_rolling_levels(self, dataframe: DataFrame) -> DataFrame:
        lookback = int(self.rolling_level_lookback)
        min_periods = max(4, lookback // 4)
        dataframe['rolling_resistance'] = _num(dataframe, 'high').shift(1).rolling(lookback, min_periods=min_periods).max()
        dataframe['rolling_support'] = _num(dataframe, 'low').shift(1).rolling(lookback, min_periods=min_periods).min()
        return dataframe

    def _add_avwap(self, dataframe: DataFrame) -> DataFrame:
        lookback = int(self.avwap_anchor_lookback)
        low = _num(dataframe, 'low')
        high = _num(dataframe, 'high')
        prior_low = low.shift(1)
        prior_high = high.shift(1)
        low_reset = prior_low.le(prior_low.rolling(lookback, min_periods=2).min())
        high_reset = prior_high.ge(prior_high.rolling(lookback, min_periods=2).max())
        dataframe['avwap_from_low'] = self._anchored_vwap_series(dataframe, low_reset)
        dataframe['avwap_from_high'] = self._anchored_vwap_series(dataframe, high_reset)
        typical = _num(dataframe, 'high').add(_num(dataframe, 'low')).add(_num(dataframe, 'close')).div(3.0)
        deviation = typical.rolling(lookback, min_periods=max(5, lookback // 5)).std()
        band_mult = float(self.avwap_band_mult)
        dataframe['avwap_lower_band'] = dataframe['avwap_from_low'].sub(deviation.mul(band_mult))
        dataframe['avwap_upper_band'] = dataframe['avwap_from_high'].add(deviation.mul(band_mult))
        return dataframe

    @staticmethod
    def _anchored_vwap_series(dataframe: DataFrame, reset: Series) -> Series:
        typical = _num(dataframe, 'high').add(_num(dataframe, 'low')).add(_num(dataframe, 'close')).div(3.0)
        volume = _num(dataframe, 'volume').clip(lower=0.0)
        groups = reset.fillna(False).astype(bool).cumsum()
        volume_sum = volume.groupby(groups).cumsum().replace(0.0, np.nan)
        price_volume_sum = typical.mul(volume).groupby(groups).cumsum()
        return price_volume_sum.div(volume_sum)

    def _add_supply_demand(self, dataframe: DataFrame) -> DataFrame:
        window = int(self.zone_impulse_window)
        open_ = _num(dataframe, 'open')
        high = _num(dataframe, 'high')
        low = _num(dataframe, 'low')
        close = _num(dataframe, 'close')
        body = close.sub(open_).abs()
        candle_range = high.sub(low).replace(0.0, np.nan)
        body_fraction = body.div(candle_range)
        atr = self._atr(dataframe)
        volume_ratio = _num(dataframe, 'volume').div(_num(dataframe, 'volume').rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan))
        impulse_distance = atr.mul(float(self.zone_impulse_atr_min))
        quality = body_fraction.ge(float(self.zone_body_fraction_min)) & volume_ratio.ge(float(self.zone_volume_ratio_min))
        bull_impulse = close.gt(open_) & close.sub(open_).ge(impulse_distance) & quality
        bear_impulse = close.lt(open_) & open_.sub(close).ge(impulse_distance) & quality
        body_low = pd.concat([open_, close], axis=1).min(axis=1)
        body_high = pd.concat([open_, close], axis=1).max(axis=1)
        dataframe['demand_zone_low'] = low.where(bull_impulse).ffill().shift(1)
        dataframe['demand_zone_high'] = body_low.where(bull_impulse).ffill().shift(1)
        dataframe['supply_zone_low'] = body_high.where(bear_impulse).ffill().shift(1)
        dataframe['supply_zone_high'] = high.where(bear_impulse).ffill().shift(1)
        dataframe['demand_zone_age'] = self._bars_since(bull_impulse).shift(1)
        dataframe['supply_zone_age'] = self._bars_since(bear_impulse).shift(1)
        return dataframe

    def _add_liquidity_levels(self, dataframe: DataFrame) -> DataFrame:
        lookback = int(self.equal_level_lookback)
        min_touches = int(self.equal_level_min_touches)
        tolerance = float(self.equal_level_tolerance_pct)
        min_periods = max(4, lookback // 4)
        prior_high = _num(dataframe, 'high').shift(1)
        prior_low = _num(dataframe, 'low').shift(1)
        high_level = prior_high.rolling(lookback, min_periods=min_periods).max()
        low_level = prior_low.rolling(lookback, min_periods=min_periods).min()
        high_touches = prior_high.sub(high_level).abs().le(high_level.abs().mul(tolerance)).rolling(lookback, min_periods=min_periods).sum()
        low_touches = prior_low.sub(low_level).abs().le(low_level.abs().mul(tolerance)).rolling(lookback, min_periods=min_periods).sum()
        dataframe['equal_high_level'] = high_level.where(high_touches.ge(min_touches))
        dataframe['equal_low_level'] = low_level.where(low_touches.ge(min_touches))
        dataframe['range_resistance'] = high_level
        dataframe['range_support'] = low_level
        return dataframe

    def _add_volume_profile(self, dataframe: DataFrame, prefix: str, window: int, bins: int) -> DataFrame:
        return add_volume_profile(dataframe, window=window, bins=bins, value_area_pct=float(self.vp_value_area_pct), price_source=str(self.vp_price_source), smooth_bins=int(self.vp_smooth_bins), hvn_threshold=float(self.vp_hvn_threshold), lvn_threshold=float(self.vp_lvn_threshold), pressure_delta_min=float(self.vp_pressure_delta_min), node_near_pct=float(self.vp_node_near_pct), volume_percentile_min=float(self.vp_volume_percentile_min), score_window=int(self.vp_score_window), fast_traverse_atr_mult=float(self.vp_fast_traverse_atr_mult), entry_score_margin=float(self.vp_entry_score_margin), prefix=prefix)

    def _entry_condition(self, dataframe: DataFrame) -> Series:
        if CONCEPT == 'prior_high_breakout':
            return self._prior_high_breakout(dataframe)
        if CONCEPT == 'prior_low_breakdown':
            return self._prior_low_breakdown(dataframe)
        if CONCEPT == 'avwap_reclaim':
            return self._avwap_reclaim(dataframe)
        if CONCEPT == 'avwap_reject':
            return self._avwap_reject(dataframe)
        if CONCEPT == 'avwap_lower_band_reclaim':
            return self._avwap_lower_band_reclaim(dataframe)
        if CONCEPT == 'avwap_upper_band_reject':
            return self._avwap_upper_band_reject(dataframe)
        if CONCEPT == 'demand_reclaim':
            return self._demand_reclaim(dataframe)
        if CONCEPT == 'supply_reject':
            return self._supply_reject(dataframe)
        if CONCEPT == 'demand_breakdown':
            return self._demand_breakdown(dataframe)
        if CONCEPT == 'supply_breakout':
            return self._supply_breakout(dataframe)
        if CONCEPT == 'equal_lows_sweep_reclaim':
            return self._equal_lows_sweep_reclaim(dataframe)
        if CONCEPT == 'equal_highs_sweep_reject':
            return self._equal_highs_sweep_reject(dataframe)
        if CONCEPT == 'range_low_sweep_reclaim':
            return self._range_low_sweep_reclaim(dataframe)
        if CONCEPT == 'range_high_sweep_reject':
            return self._range_high_sweep_reject(dataframe)
        if CONCEPT == 'confluence_prior_vp_breakout':
            return self._prior_high_breakout(dataframe) & self._vp_bull_ok(dataframe)
        if CONCEPT == 'confluence_prior_vp_breakdown':
            return self._prior_low_breakdown(dataframe) & self._vp_bear_ok(dataframe)
        if CONCEPT == 'confluence_prior_avwap_vp_breakout':
            return self._prior_high_breakout(dataframe) & self._avwap_trend_bull(dataframe) & self._vp_bull_ok(dataframe)
        if CONCEPT == 'confluence_prior_avwap_vp_breakdown':
            return self._prior_low_breakdown(dataframe) & self._avwap_trend_bear(dataframe) & self._vp_bear_ok(dataframe)
        if CONCEPT == 'confluence_demand_vp_reclaim':
            return self._demand_reclaim(dataframe) & self._vp_bull_ok(dataframe)
        if CONCEPT == 'confluence_supply_vp_reject':
            return self._supply_reject(dataframe) & self._vp_bear_ok(dataframe)
        if CONCEPT == 'confluence_liquidity_vp_reclaim':
            return self._equal_lows_sweep_reclaim(dataframe) & self._vp_bull_ok(dataframe)
        if CONCEPT == 'confluence_liquidity_vp_reject':
            return self._equal_highs_sweep_reject(dataframe) & self._vp_bear_ok(dataframe)
        if CONCEPT == 'confluence_four_way_breakout':
            return self._prior_high_breakout(dataframe) & self._rolling_resistance_breakout(dataframe) & self._avwap_trend_bull(dataframe) & self._vp_bull_ok(dataframe)
        if CONCEPT == 'confluence_four_way_breakdown':
            return self._prior_low_breakdown(dataframe) & self._rolling_support_breakdown(dataframe) & self._avwap_trend_bear(dataframe) & self._vp_bear_ok(dataframe)
        return pd.Series(False, index=dataframe.index, dtype='bool')

    def _prior_level(self, dataframe: DataFrame, field: str) -> Series:
        period = self._selected_period()
        return _num(dataframe, f'prior_{period}_{field}', np.nan)

    def _prior_high_breakout(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        level = self._prior_level(dataframe, 'high')
        trigger = level.mul(1.0 + float(self.breakout_buffer_pct))
        return close.ge(trigger) & close.shift(1).lt(trigger.shift(1))

    def _prior_low_breakdown(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        level = self._prior_level(dataframe, 'low')
        trigger = level.mul(1.0 - float(self.breakout_buffer_pct))
        return close.le(trigger) & close.shift(1).gt(trigger.shift(1))

    def _rolling_resistance_breakout(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        level = _num(dataframe, 'rolling_resistance', np.nan)
        trigger = level.mul(1.0 + float(self.breakout_buffer_pct))
        return close.ge(trigger) & close.shift(1).lt(trigger.shift(1))

    def _rolling_support_breakdown(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        level = _num(dataframe, 'rolling_support', np.nan)
        trigger = level.mul(1.0 - float(self.breakout_buffer_pct))
        return close.le(trigger) & close.shift(1).gt(trigger.shift(1))

    def _avwap_reclaim(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        avwap = _num(dataframe, 'avwap_from_low', np.nan)
        trigger = avwap.mul(1.0 + float(self.reclaim_buffer_pct))
        return close.ge(trigger) & close.shift(1).lt(avwap.shift(1)) & _num(dataframe, 'low').le(avwap.mul(1.0 + float(self.zone_near_pct)))

    def _avwap_reject(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        avwap = _num(dataframe, 'avwap_from_high', np.nan)
        trigger = avwap.mul(1.0 - float(self.reclaim_buffer_pct))
        return close.le(trigger) & close.shift(1).gt(avwap.shift(1)) & _num(dataframe, 'high').ge(avwap.mul(1.0 - float(self.zone_near_pct)))

    def _avwap_lower_band_reclaim(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        band = _num(dataframe, 'avwap_lower_band', np.nan)
        return _num(dataframe, 'low').le(band.mul(1.0 + float(self.sweep_buffer_pct))) & close.ge(band.mul(1.0 + float(self.reclaim_buffer_pct))) & close.gt(_num(dataframe, 'open'))

    def _avwap_upper_band_reject(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        band = _num(dataframe, 'avwap_upper_band', np.nan)
        return _num(dataframe, 'high').ge(band.mul(1.0 - float(self.sweep_buffer_pct))) & close.le(band.mul(1.0 - float(self.reclaim_buffer_pct))) & close.lt(_num(dataframe, 'open'))

    def _demand_reclaim(self, dataframe: DataFrame) -> Series:
        low = _num(dataframe, 'low')
        close = _num(dataframe, 'close')
        zone_high = _num(dataframe, 'demand_zone_high', np.nan)
        age_ok = _num(dataframe, 'demand_zone_age', 9999.0).le(float(self.zone_max_age_bars))
        return age_ok & low.le(zone_high.mul(1.0 + float(self.zone_near_pct))) & close.ge(zone_high.mul(1.0 + float(self.reclaim_buffer_pct))) & close.gt(_num(dataframe, 'open'))

    def _supply_reject(self, dataframe: DataFrame) -> Series:
        high = _num(dataframe, 'high')
        close = _num(dataframe, 'close')
        zone_low = _num(dataframe, 'supply_zone_low', np.nan)
        age_ok = _num(dataframe, 'supply_zone_age', 9999.0).le(float(self.zone_max_age_bars))
        return age_ok & high.ge(zone_low.mul(1.0 - float(self.zone_near_pct))) & close.le(zone_low.mul(1.0 - float(self.reclaim_buffer_pct))) & close.lt(_num(dataframe, 'open'))

    def _demand_breakdown(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        zone_low = _num(dataframe, 'demand_zone_low', np.nan)
        trigger = zone_low.mul(1.0 - float(self.breakout_buffer_pct))
        age_ok = _num(dataframe, 'demand_zone_age', 9999.0).le(float(self.zone_max_age_bars))
        return age_ok & close.le(trigger) & close.shift(1).gt(trigger.shift(1))

    def _supply_breakout(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        zone_high = _num(dataframe, 'supply_zone_high', np.nan)
        trigger = zone_high.mul(1.0 + float(self.breakout_buffer_pct))
        age_ok = _num(dataframe, 'supply_zone_age', 9999.0).le(float(self.zone_max_age_bars))
        return age_ok & close.ge(trigger) & close.shift(1).lt(trigger.shift(1))

    def _equal_lows_sweep_reclaim(self, dataframe: DataFrame) -> Series:
        level = _num(dataframe, 'equal_low_level', np.nan)
        return _num(dataframe, 'low').le(level.mul(1.0 - float(self.sweep_buffer_pct))) & _num(dataframe, 'close').ge(level.mul(1.0 + float(self.reclaim_buffer_pct))) & _num(dataframe, 'close').gt(_num(dataframe, 'open'))

    def _equal_highs_sweep_reject(self, dataframe: DataFrame) -> Series:
        level = _num(dataframe, 'equal_high_level', np.nan)
        return _num(dataframe, 'high').ge(level.mul(1.0 + float(self.sweep_buffer_pct))) & _num(dataframe, 'close').le(level.mul(1.0 - float(self.reclaim_buffer_pct))) & _num(dataframe, 'close').lt(_num(dataframe, 'open'))

    def _range_low_sweep_reclaim(self, dataframe: DataFrame) -> Series:
        level = _num(dataframe, 'range_support', np.nan)
        return _num(dataframe, 'low').le(level.mul(1.0 - float(self.sweep_buffer_pct))) & _num(dataframe, 'close').ge(level.mul(1.0 + float(self.reclaim_buffer_pct))) & _num(dataframe, 'close').gt(_num(dataframe, 'open'))

    def _range_high_sweep_reject(self, dataframe: DataFrame) -> Series:
        level = _num(dataframe, 'range_resistance', np.nan)
        return _num(dataframe, 'high').ge(level.mul(1.0 + float(self.sweep_buffer_pct))) & _num(dataframe, 'close').le(level.mul(1.0 - float(self.reclaim_buffer_pct))) & _num(dataframe, 'close').lt(_num(dataframe, 'open'))

    def _avwap_trend_bull(self, dataframe: DataFrame) -> Series:
        return _num(dataframe, 'close').ge(_num(dataframe, 'avwap_from_low', np.nan).mul(1.0 + float(self.reclaim_buffer_pct)))

    def _avwap_trend_bear(self, dataframe: DataFrame) -> Series:
        return _num(dataframe, 'close').le(_num(dataframe, 'avwap_from_high', np.nan).mul(1.0 - float(self.reclaim_buffer_pct)))

    def _vp_bull_ok(self, dataframe: DataFrame) -> Series:
        return self._vp_guard(dataframe, 'vp', 'long', str(self.vp_guard_mode), float(self.vp_score_min), float(self.vp_context_min))

    def _vp_bear_ok(self, dataframe: DataFrame) -> Series:
        return self._vp_guard(dataframe, 'vp', 'short', str(self.vp_guard_mode), float(self.vp_score_min), float(self.vp_context_min))

    def _volume_guard(self, dataframe: DataFrame) -> Series:
        ratio_ok = _num(dataframe, 'entry_volume_ratio').ge(float(self.volume_ratio_min))
        pressure = _num(dataframe, 'entry_pressure')
        if SIDE == 'long':
            return ratio_ok & pressure.ge(float(self.pressure_min))
        return ratio_ok & pressure.le(-float(self.pressure_min))

    def _close_direction_guard(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        if SIDE == 'long':
            return close.gt(close.shift(1))
        return close.lt(close.shift(1))

    def _merge_informative_vp(self, dataframe: DataFrame, metadata: dict, timeframe: str, prefix: str, window: int, bins: int) -> DataFrame:
        if not getattr(self, 'dp', None) or 'date' not in dataframe.columns:
            return dataframe
        pair = metadata.get('pair') if metadata else None
        if not pair:
            return dataframe
        informative = self.dp.get_pair_dataframe(pair=pair, timeframe=timeframe)
        if informative is None or informative.empty or 'date' not in informative.columns:
            return dataframe
        informative = self._add_volume_profile(informative.copy(), prefix, window, bins)
        merge_columns = [f'{prefix}_{name}' for name in VP_COLUMNS if f'{prefix}_{name}' in informative.columns]
        if not merge_columns:
            return dataframe
        minutes = timeframe_to_minutes(timeframe)
        inf = informative[['date', *merge_columns]].copy().sort_values('date')
        inf['date_merge'] = inf['date'] + pd.to_timedelta(minutes, unit='m')
        base = dataframe.reset_index().rename(columns={'index': '__row_index'}).sort_values('date')
        merged = pd.merge_asof(base, inf[['date_merge', *merge_columns]].sort_values('date_merge'), left_on='date', right_on='date_merge', direction='backward').sort_values('__row_index')
        for column in merge_columns:
            dataframe[column] = merged[column].to_numpy()
        return dataframe

    @staticmethod
    def _score_guard(dataframe: DataFrame, prefix: str, side: str, score_min: float) -> Series:
        score = _num(dataframe, f'{prefix}_score_{side}')
        opposite = _num(dataframe, f"{prefix}_score_{('short' if side == 'long' else 'long')}")
        return score.ge(score_min) & score.ge(opposite)

    @staticmethod
    def _context_guard(dataframe: DataFrame, prefix: str, side: str, context_min: float) -> Series:
        if side == 'long':
            context = _num(dataframe, f'{prefix}_context_score_bull')
            opposite = _num(dataframe, f'{prefix}_context_score_bear')
            market_ok = _num(dataframe, f'{prefix}_market_context').ge(0)
        else:
            context = _num(dataframe, f'{prefix}_context_score_bear')
            opposite = _num(dataframe, f'{prefix}_context_score_bull')
            market_ok = _num(dataframe, f'{prefix}_market_context').le(0)
        return context.ge(context_min) & context.ge(opposite) & market_ok

    def _vp_guard(self, dataframe: DataFrame, prefix: str, side: str, mode: str, score_min: float, context_min: float) -> Series:
        if mode == "score": return self._score_guard(dataframe, prefix, side, score_min)
        if mode == "context": return self._context_guard(dataframe, prefix, side, context_min)
        if mode == "score_or_context": return self._score_guard(dataframe, prefix, side, score_min) | self._context_guard(dataframe, prefix, side, context_min)
        if mode == "balance": return _num(dataframe, f"{prefix}_context_score_balance").ge(context_min)
        direction = _num(dataframe, f"{prefix}_market_context")
        return direction.ge(0) if side == "long" else direction.le(0)

    @staticmethod
    def _add_sieve3_forward_target(dataframe: DataFrame) -> DataFrame:
        candidate_columns = ('vp_hvn_below', 'vp_val', 'vp_poc')
        missing = [column for column in candidate_columns if column not in dataframe.columns]
        if len(missing) == len(candidate_columns):
            raise KeyError(f'no configured forward-target columns are present: {missing}')
        close = pd.to_numeric(dataframe['close'], errors='coerce')
        levels = pd.concat(
            [pd.to_numeric(dataframe[column], errors='coerce') for column in candidate_columns if column in dataframe.columns],
            axis=1,
        )
        invalidation = pd.to_numeric(dataframe['demand_zone_low'], errors='coerce')
        same_as_invalidation = pd.DataFrame(
            np.isclose(
                levels.to_numpy(dtype=float),
                invalidation.to_numpy(dtype=float)[:, None],
                rtol=0.0,
                atol=1e-12,
                equal_nan=False,
            ),
            index=levels.index,
            columns=levels.columns,
        )
        levels = levels.mask(same_as_invalidation)
        forward = levels.where(levels.lt(close.mul(0.999), axis='index'), np.nan)
        dataframe['sieve3_exit_target_level'] = forward.max(axis=1, skipna=True)
        return dataframe

    SOURCE_ENTRY_STEM = 'overtrade_demand_zone_breakdown_short_vp_market_guard'
    FOCUSED_EXIT_CONTRACT = 'target_partial_invalidation_remainder'
    FOCUSED_SOURCE_PROFILE = {'side': 'short', 'min_level_distance': 0.001, 'targets': {'target_1': {'provider': 'nearest_forward_vp_tlv2_or_structure_obstacle', 'short': {'level': 'sieve3_exit_target_level', 'available': None}}}, 'invalidation': {'provider': 'demand_zone_breakdown_reclaim', 'mode': 'level', 'short': {'level': 'demand_zone_low', 'available': None}}}
    FOCUSED_EXIT_PLANS = {'touch_partial': {'contract': 'target_partial_invalidation_remainder', 'name': 'touch_partial', 'hard_stop_ratio': 0.03, 'max_hold_candles': 336}, 'close_partial': {'contract': 'target_partial_invalidation_remainder', 'name': 'close_partial', 'hard_stop_ratio': 0.03, 'max_hold_candles': 336}, 'reversal_1_partial': {'contract': 'target_partial_invalidation_remainder', 'name': 'reversal_1_partial', 'hard_stop_ratio': 0.03, 'max_hold_candles': 336}, 'reversal_2_of_3_partial': {'contract': 'target_partial_invalidation_remainder', 'name': 'reversal_2_of_3_partial', 'hard_stop_ratio': 0.03, 'max_hold_candles': 336}}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'demand_zone_low', 'high', 'low', 'open', 'sieve3_exit_target_level')
    FOCUSED_STATE_VERSION = 2
    FOCUSED_STATE_KEY = 'sieve3_v2_focused:Sieve3V2TargetPartialInvalidationRemainderFromOvertradeDemandZoneBreakdownShortVpMarketGuard:target_partial_invalidation_remainder'
    LOCKED_BUY_PARAMS = {'use_sieve2_vp_guard': True, 'sieve2_vp_guard_mode': 'score_or_context', 'sieve2_vp_window': 96, 'sieve2_vp_bins': 36, 'sieve2_vp_score_min': 0.15, 'sieve2_vp_context_min': 0.15, 'use_sieve2_market_guard': True, 'sieve2_market_guard_mode': 'pressure_or_trend', 'sieve2_market_window': 24, 'sieve2_market_pressure_min': 0.03, 'sieve2_market_trend_min': 0.15, 'sieve2_rs_benchmark_pair': 'BTC/USDT:USDT', 'sieve2_rs_score_min': 0.3, 'breakout_buffer_pct': 0.003, 'reclaim_buffer_pct': 0.006, 'sweep_buffer_pct': 0.006, 'zone_near_pct': 0.01, 'rolling_level_lookback': 72, 'equal_level_lookback': 72, 'equal_level_tolerance_pct': 0.006, 'equal_level_min_touches': 2, 'confluence_period': 'day', 'avwap_anchor_lookback': 120, 'avwap_band_mult': 1.25, 'zone_impulse_window': 36, 'zone_impulse_atr_min': 0.8, 'zone_body_fraction_min': 0.55, 'zone_volume_ratio_min': 1.1, 'zone_max_age_bars': 168, 'use_volume_guard': False, 'volume_window': 12, 'volume_ratio_min': 0.8, 'pressure_min': 0.35, 'use_close_direction_guard': False, 'vp_window': 96, 'vp_bins': 48, 'vp_value_area_pct': 0.7, 'vp_price_source': 'hlc3', 'vp_smooth_bins': 3, 'vp_hvn_threshold': 0.7, 'vp_lvn_threshold': 0.35, 'vp_pressure_delta_min': 0.05, 'vp_node_near_pct': 0.01, 'vp_volume_percentile_min': 0.55, 'vp_score_window': 48, 'vp_fast_traverse_atr_mult': 1.2, 'vp_entry_score_margin': 0.02, 'use_vp_1h_guard': False, 'vp_guard_mode': 'score_or_context', 'vp_score_min': 0.25, 'vp_context_min': 0.28, 'use_vp_4h_guard': True, 'vp_4h_window': 48, 'vp_4h_bins': 36, 'use_vp_1d_guard': False, 'vp_1d_window': 30, 'vp_1d_bins': 36}
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

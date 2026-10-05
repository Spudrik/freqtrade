from __future__ import annotations
from freqtrade.strategy import informative
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
from user_data.Indicators.complex_volume_profile import add_volume_profile
ENTRY_MODE = 'entry_prior_month_high_breakout_long'

ENTRY_SOURCE_STAGE = 'sieve2'

CONCEPT = 'prior_high_breakout'
PERIOD_KIND = 'month'
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
SOURCE_ENTRY_CLASS = 'Sieve3ExitBreakevenFromPriorMonthHighBreakoutLong'
SOURCE_STRATEGY = 'user_data/strategies/sieve3_exit_breakeven_from_prior_month_high_breakout_long.py:Sieve3ExitBreakevenFromPriorMonthHighBreakoutLong'
LINEAGE_STATUS = 'verified'

EXIT_THEORY = 'va_edge_reaction_base'







EXIT_FAMILY = 'va_edge_reaction_base'
PRIMARY_TRIGGER = 'close crosses above prior_month_high * (1 + breakout_buffer_pct)'
PRIMARY_GUARD = 'entry_volume_ratio >= volume_ratio_min and entry_pressure >= pressure_min'
PRIMARY_TARGET = 'named fixed partial and remainder objectives'
PRIMARY_INVALIDATION = 'named fixed hard stop'
TARGET_PROVIDER = 'closed-candle actual-fill-relative deterministic named level zone'
INVALIDATION_PROVIDER = 'coherent source invalidation when available, otherwise preserved 3% hard stop'
SOURCE_RESULT_BATCH = 'historical_promotion_lineage_unknown'
RESEARCH_PATH = 'sieve3_exit_final_generic_levels'
EXIT_HYPOTHESIS = 'Isolate va_edge on the base timeframe scope and exit only after directional weakening confirms around that named level zone.'
ACTIVE_SELL_PARAMS = ('zone_width_quarter_percent', 'weakening_signal', 'confirmation_profile')
SIEVE_STAGE = 'sieve3'
SIDE = 'long'
ENTRY_TAG = 'prior_month_high_breakout_long'

class Sieve3V2VaEdgeReactionBaseFromPriorMonthHighBreakoutLong(IStrategy):
    """

    Sieve2 entry concept: prior_month_high_breakout_long.



    This is entry-quality research only. It has no custom exit, no DCA, and no

    position-management hooks. Optional guards are hyperoptable inside this

    standalone strategy so Explorer can test whether they improve the trigger.

    """
    INTERFACE_VERSION = 3
    timeframe = '1h'
    startup_candle_count = 336
    process_only_new_candles = True
    can_short = False
    max_entry_position_adjustment = 0
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
    zone_max_age_bars = 72
    use_volume_guard = True
    volume_window = 24
    volume_ratio_min = 1.3
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
    # Legacy inactive entry parameter: use_vp_4h_guard=False (enable gate is fixed off).
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

    def _sieve3_entry_populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self._add_volume_pressure(dataframe)
        dataframe = self._add_prior_period_levels(dataframe)
        dataframe = self._add_rolling_levels(dataframe)
        dataframe = self._add_avwap(dataframe)
        dataframe = self._add_supply_demand(dataframe)
        dataframe = self._add_liquidity_levels(dataframe)
        dataframe = self._add_volume_profile(dataframe, 'vp', int(self.vp_window), int(self.vp_bins))
        dataframe = self._merge_informative_vp(dataframe, metadata, '4h', 'vp4h', int(self.vp_4h_window), int(self.vp_4h_bins))
        dataframe = self._merge_informative_vp(dataframe, metadata, '1d', 'vp1d', int(self.vp_1d_window), int(self.vp_1d_bins))
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = None
        condition = self._entry_condition(dataframe)
        if bool(self.use_volume_guard):
            condition &= self._volume_guard(dataframe)
        valid = condition.fillna(False) & dataframe['volume'].gt(0.0) & dataframe['close'].notna()
        dataframe.loc[valid, 'enter_long'] = 1
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

    def _generic_level_frame(self, dataframe: DataFrame, _timeframe: str) -> DataFrame:
        return add_volume_profile(dataframe, prefix='gsvp')
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        frame = self._sieve3_entry_populate_indicators(dataframe, metadata)
    
        if True:
            frame = self._generic_level_frame(frame, self.timeframe)
        return frame

    SOURCE_ENTRY_STEM = 'prior_month_high_breakout_long'
    FOCUSED_EXIT_CONTRACT = 'va_edge_reaction_base'
    FOCUSED_SOURCE_PROFILE = {'side': 'long', 'min_level_distance': 0.001}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'high', 'low', 'open')
    FOCUSED_STATE_VERSION = 1
    FOCUSED_STATE_KEY = 'sieve3_v2_generic:Sieve3V2VaEdgeReactionBaseFromPriorMonthHighBreakoutLong:va_edge_reaction_base'
    LOCKED_BUY_PARAMS = {'use_sieve2_vp_guard': False, 'sieve2_vp_guard_mode': 'score_or_context', 'sieve2_vp_window': 96, 'sieve2_vp_bins': 36, 'sieve2_vp_score_min': 0.25, 'sieve2_vp_context_min': 0.28, 'use_sieve2_market_guard': False, 'sieve2_market_guard_mode': 'pressure_or_trend', 'sieve2_market_window': 24, 'sieve2_market_pressure_min': 0.07, 'sieve2_market_trend_min': 0.25, 'sieve2_rs_benchmark_pair': 'BTC/USDT:USDT', 'sieve2_rs_score_min': 0.45, 'breakout_buffer_pct': 0.003, 'reclaim_buffer_pct': 0.006, 'sweep_buffer_pct': 0.006, 'zone_near_pct': 0.01, 'rolling_level_lookback': 72, 'equal_level_lookback': 72, 'equal_level_tolerance_pct': 0.006, 'equal_level_min_touches': 2, 'confluence_period': 'day', 'avwap_anchor_lookback': 120, 'avwap_band_mult': 1.25, 'zone_impulse_window': 36, 'zone_impulse_atr_min': 0.8, 'zone_body_fraction_min': 0.55, 'zone_volume_ratio_min': 1.1, 'zone_max_age_bars': 72, 'use_volume_guard': True, 'volume_window': 24, 'volume_ratio_min': 1.3, 'pressure_min': 0.35, 'use_close_direction_guard': False, 'vp_window': 96, 'vp_bins': 48, 'vp_value_area_pct': 0.7, 'vp_price_source': 'hlc3', 'vp_smooth_bins': 3, 'vp_hvn_threshold': 0.7, 'vp_lvn_threshold': 0.35, 'vp_pressure_delta_min': 0.05, 'vp_node_near_pct': 0.01, 'vp_volume_percentile_min': 0.55, 'vp_score_window': 48, 'vp_fast_traverse_atr_mult': 1.2, 'vp_entry_score_margin': 0.02, 'use_vp_1h_guard': False, 'vp_guard_mode': 'score_or_context', 'vp_score_min': 0.25, 'vp_context_min': 0.28, 'use_vp_4h_guard': False, 'vp_4h_window': 48, 'vp_4h_bins': 36, 'use_vp_1d_guard': False, 'vp_1d_window': 30, 'vp_1d_bins': 36}
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

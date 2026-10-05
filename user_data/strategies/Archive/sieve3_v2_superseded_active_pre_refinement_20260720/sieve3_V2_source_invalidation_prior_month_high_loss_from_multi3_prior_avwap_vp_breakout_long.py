from __future__ import annotations
import math
import pandas as pd
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from pandas import DataFrame, Series
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.strategy import CategoricalParameter, stoploss_from_absolute, IntParameter
LOCKED_BUY_SOURCE = 'verified migration-ledger effective buy lock'
SOURCE_ENTRY_SIGNATURE_SHA256 = '0f95ee55961912ab66a49446e2d7a288629c11cd09e68a965eb26ff104184d8c'
SECONDARY_GUARD = 'locked volume-ratio guard and optional Sieve2 guards'
import numpy as np
from freqtrade.strategy import IStrategy
from user_data.Indicators.complex_volume_profile import add_volume_profile

ENTRY_MODE = 'entry_multi3_prior_avwap_vp_breakout_long'
ENTRY_TAG = 'multi3_prior_avwap_vp_breakout_long'
ENTRY_SOURCE_STAGE = 'sieve2'
SIDE = 'long'
CONCEPT = 'multi3_prior_avwap_vp_breakout'
PERIOD_KIND = 'select'
GUARD_MODE_CHOICES = ['direction', 'score', 'context', 'score_or_context', 'balance']
PERIOD_CHOICES = ['day', 'week', 'month']
PRICE_SOURCE_CHOICES = ['close', 'hl2', 'hlc3', 'ohlc4']
VP_COLUMNS = ['score_long', 'score_short', 'context_score_bull', 'context_score_bear', 'context_score_balance', 'market_context']


def _num(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        raise KeyError(f"required numeric column is missing: {column!r}")
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
SIEVE_STAGE = 'sieve3'
SOURCE_ENTRY_CLASS = 'Sieve3ExitBreakevenFromMulti3PriorAvwapVpBreakoutLong'
SOURCE_STRATEGY = 'user_data/strategies/sieve3_exit_breakeven_from_multi3_prior_avwap_vp_breakout_long.py:Sieve3ExitBreakevenFromMulti3PriorAvwapVpBreakoutLong'
SOURCE_RESULT_BATCH = '20260521T012740_entry_all_resume'
RESEARCH_PATH = 'sieve3_exit_source_invalidation'
LINEAGE_STATUS = 'verified'
EXIT_FAMILY = 'source_invalidation'
EXIT_THEORY = 'source_invalidation_prior_month_high_loss'
EXIT_HYPOTHESIS = 'A confirmed close through the entry-frozen prior_month_high_loss boundary invalidates the source entry thesis and closes the position.'
PRIMARY_TRIGGER = '1h close cross above locked multi_period=month prior high'
PRIMARY_GUARD = 'close above avwap_from_low plus locked bullish VP guard'
TARGET_PROVIDER = 'none'
PRIMARY_TARGET = 'none'
INVALIDATION_PROVIDER = 'provider=crossed_locked_month_high_loss;mode=level;long.level=prior_month_high'
PRIMARY_INVALIDATION = 'provider=crossed_locked_month_high_loss;mode=level;long.level=prior_month_high'
ACTIVE_SELL_PARAMS = ('invalidation_confirmations', 'invalidation_band_quarter_percent', 'hard_stop_percent', 'max_hold_scale')

class Sieve3V2SourceInvalidationPriorMonthHighLossFromMulti3PriorAvwapVpBreakoutLong(IStrategy):
    """

    Sieve2 entry concept: multi3_prior_avwap_vp_breakout_long.



    This is entry-quality research only. It has no custom exit, no DCA, and no

    position-management hooks. Optional guards are hyperoptable inside this

    standalone strategy so Explorer can test whether they improve the trigger.

    """
    INTERFACE_VERSION = 3
    timeframe = '1h'
    startup_candle_count = 336
    process_only_new_candles = True
    can_short = False
    use_sieve2_vp_guard = False
    sieve2_vp_guard_mode = 'score_or_context'
    sieve2_vp_window = 96
    sieve2_vp_bins = 36
    sieve2_vp_score_min = 0.25
    sieve2_vp_context_min = 0.28
    use_sieve2_market_guard = False
    sieve2_market_guard_mode = 'pressure_or_trend'
    sieve2_market_window = 24
    sieve2_market_pressure_min = 0.07
    sieve2_market_trend_min = 0.25
    sieve2_rs_benchmark_pair = 'BTC/USDT:USDT'
    sieve2_rs_score_min = 0.45
    use_close_direction_guard = False
    use_vp_1h_guard = False
    use_vp_4h_guard = False
    use_vp_1d_guard = False
    max_entry_position_adjustment = 0
    breakout_buffer_pct = 0.003
    reclaim_buffer_pct = 0.0
    sweep_buffer_pct = 0.006
    zone_near_pct = 0.01
    rolling_level_lookback = 72
    equal_level_lookback = 72
    equal_level_tolerance_pct = 0.006
    equal_level_min_touches = 2
    multi_period = 'month'
    avwap_anchor_lookback = 240
    avwap_band_mult = 1.25
    zone_impulse_window = 36
    zone_impulse_atr_min = 0.8
    zone_body_fraction_min = 0.55
    zone_volume_ratio_min = 1.1
    zone_max_age_bars = 72
    use_volume_guard = True
    volume_window = 24
    volume_ratio_min = 0.8
    pressure_min = 0.05
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
    vp_guard_mode = 'context'
    vp_score_min = 0.25
    vp_context_min = 0.6
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
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = None
        condition = self._entry_condition(dataframe)
        if bool(self.use_volume_guard):
            condition &= self._volume_guard(dataframe)
        if bool(self.use_close_direction_guard):
            condition &= self._close_direction_guard(dataframe)
        if bool(self.use_vp_1h_guard):
            condition &= self._vp_guard(dataframe, 'vp', SIDE, str(self.vp_guard_mode), float(self.vp_score_min), float(self.vp_context_min))
        if bool(self.use_vp_4h_guard):
            condition &= self._vp_guard(dataframe, 'vp4h', SIDE, str(self.vp_guard_mode), float(self.vp_score_min), float(self.vp_context_min))
        if bool(self.use_vp_1d_guard):
            condition &= self._vp_guard(dataframe, 'vp1d', SIDE, str(self.vp_guard_mode), float(self.vp_score_min), float(self.vp_context_min))
        condition = pd.Series(condition, index=dataframe.index).fillna(False).astype(bool).fillna(False)
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
            return str(self.multi_period)
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
        if CONCEPT == 'multi2_prior_vp_breakout':
            return self._prior_high_breakout(dataframe) & self._vp_bull_ok(dataframe)
        if CONCEPT == 'multi2_prior_vp_breakdown':
            return self._prior_low_breakdown(dataframe) & self._vp_bear_ok(dataframe)
        if CONCEPT == 'multi3_prior_avwap_vp_breakout':
            return self._prior_high_breakout(dataframe) & self._avwap_trend_bull(dataframe) & self._vp_bull_ok(dataframe)
        if CONCEPT == 'multi3_prior_avwap_vp_breakdown':
            return self._prior_low_breakdown(dataframe) & self._avwap_trend_bear(dataframe) & self._vp_bear_ok(dataframe)
        if CONCEPT == 'multi2_demand_vp_reclaim':
            return self._demand_reclaim(dataframe) & self._vp_bull_ok(dataframe)
        if CONCEPT == 'multi2_supply_vp_reject':
            return self._supply_reject(dataframe) & self._vp_bear_ok(dataframe)
        if CONCEPT == 'multi2_liquidity_vp_reclaim':
            return self._equal_lows_sweep_reclaim(dataframe) & self._vp_bull_ok(dataframe)
        if CONCEPT == 'multi2_liquidity_vp_reject':
            return self._equal_highs_sweep_reject(dataframe) & self._vp_bear_ok(dataframe)
        if CONCEPT == 'multi4_prior_range_avwap_vp_breakout':
            return self._prior_high_breakout(dataframe) & self._rolling_resistance_breakout(dataframe) & self._avwap_trend_bull(dataframe) & self._vp_bull_ok(dataframe)
        if CONCEPT == 'multi4_prior_range_avwap_vp_breakdown':
            return self._prior_low_breakdown(dataframe) & self._rolling_support_breakdown(dataframe) & self._avwap_trend_bear(dataframe) & self._vp_bear_ok(dataframe)
        return pd.Series(False, index=dataframe.index, dtype='bool')

    def _prior_level(self, dataframe: DataFrame, field: str) -> Series:
        period = self._selected_period()
        return _num(dataframe, f'prior_{period}_{field}')

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
        level = _num(dataframe, 'rolling_resistance')
        trigger = level.mul(1.0 + float(self.breakout_buffer_pct))
        return close.ge(trigger) & close.shift(1).lt(trigger.shift(1))

    def _rolling_support_breakdown(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        level = _num(dataframe, 'rolling_support')
        trigger = level.mul(1.0 - float(self.breakout_buffer_pct))
        return close.le(trigger) & close.shift(1).gt(trigger.shift(1))

    def _avwap_reclaim(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        avwap = _num(dataframe, 'avwap_from_low')
        trigger = avwap.mul(1.0 + float(self.reclaim_buffer_pct))
        return close.ge(trigger) & close.shift(1).lt(avwap.shift(1)) & _num(dataframe, 'low').le(avwap.mul(1.0 + float(self.zone_near_pct)))

    def _avwap_reject(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        avwap = _num(dataframe, 'avwap_from_high')
        trigger = avwap.mul(1.0 - float(self.reclaim_buffer_pct))
        return close.le(trigger) & close.shift(1).gt(avwap.shift(1)) & _num(dataframe, 'high').ge(avwap.mul(1.0 - float(self.zone_near_pct)))

    def _avwap_lower_band_reclaim(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        band = _num(dataframe, 'avwap_lower_band')
        return _num(dataframe, 'low').le(band.mul(1.0 + float(self.sweep_buffer_pct))) & close.ge(band.mul(1.0 + float(self.reclaim_buffer_pct))) & close.gt(_num(dataframe, 'open'))

    def _avwap_upper_band_reject(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        band = _num(dataframe, 'avwap_upper_band')
        return _num(dataframe, 'high').ge(band.mul(1.0 - float(self.sweep_buffer_pct))) & close.le(band.mul(1.0 - float(self.reclaim_buffer_pct))) & close.lt(_num(dataframe, 'open'))

    def _demand_reclaim(self, dataframe: DataFrame) -> Series:
        low = _num(dataframe, 'low')
        close = _num(dataframe, 'close')
        zone_high = _num(dataframe, 'demand_zone_high')
        age_ok = _num(dataframe, 'demand_zone_age').le(float(self.zone_max_age_bars))
        return age_ok & low.le(zone_high.mul(1.0 + float(self.zone_near_pct))) & close.ge(zone_high.mul(1.0 + float(self.reclaim_buffer_pct))) & close.gt(_num(dataframe, 'open'))

    def _supply_reject(self, dataframe: DataFrame) -> Series:
        high = _num(dataframe, 'high')
        close = _num(dataframe, 'close')
        zone_low = _num(dataframe, 'supply_zone_low')
        age_ok = _num(dataframe, 'supply_zone_age').le(float(self.zone_max_age_bars))
        return age_ok & high.ge(zone_low.mul(1.0 - float(self.zone_near_pct))) & close.le(zone_low.mul(1.0 - float(self.reclaim_buffer_pct))) & close.lt(_num(dataframe, 'open'))

    def _demand_breakdown(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        zone_low = _num(dataframe, 'demand_zone_low')
        trigger = zone_low.mul(1.0 - float(self.breakout_buffer_pct))
        age_ok = _num(dataframe, 'demand_zone_age').le(float(self.zone_max_age_bars))
        return age_ok & close.le(trigger) & close.shift(1).gt(trigger.shift(1))

    def _supply_breakout(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        zone_high = _num(dataframe, 'supply_zone_high')
        trigger = zone_high.mul(1.0 + float(self.breakout_buffer_pct))
        age_ok = _num(dataframe, 'supply_zone_age').le(float(self.zone_max_age_bars))
        return age_ok & close.ge(trigger) & close.shift(1).lt(trigger.shift(1))

    def _equal_lows_sweep_reclaim(self, dataframe: DataFrame) -> Series:
        level = _num(dataframe, 'equal_low_level')
        return _num(dataframe, 'low').le(level.mul(1.0 - float(self.sweep_buffer_pct))) & _num(dataframe, 'close').ge(level.mul(1.0 + float(self.reclaim_buffer_pct))) & _num(dataframe, 'close').gt(_num(dataframe, 'open'))

    def _equal_highs_sweep_reject(self, dataframe: DataFrame) -> Series:
        level = _num(dataframe, 'equal_high_level')
        return _num(dataframe, 'high').ge(level.mul(1.0 + float(self.sweep_buffer_pct))) & _num(dataframe, 'close').le(level.mul(1.0 - float(self.reclaim_buffer_pct))) & _num(dataframe, 'close').lt(_num(dataframe, 'open'))

    def _range_low_sweep_reclaim(self, dataframe: DataFrame) -> Series:
        level = _num(dataframe, 'range_support')
        return _num(dataframe, 'low').le(level.mul(1.0 - float(self.sweep_buffer_pct))) & _num(dataframe, 'close').ge(level.mul(1.0 + float(self.reclaim_buffer_pct))) & _num(dataframe, 'close').gt(_num(dataframe, 'open'))

    def _range_high_sweep_reject(self, dataframe: DataFrame) -> Series:
        level = _num(dataframe, 'range_resistance')
        return _num(dataframe, 'high').ge(level.mul(1.0 + float(self.sweep_buffer_pct))) & _num(dataframe, 'close').le(level.mul(1.0 - float(self.reclaim_buffer_pct))) & _num(dataframe, 'close').lt(_num(dataframe, 'open'))

    def _avwap_trend_bull(self, dataframe: DataFrame) -> Series:
        return _num(dataframe, 'close').ge(_num(dataframe, 'avwap_from_low').mul(1.0 + float(self.reclaim_buffer_pct)))

    def _avwap_trend_bear(self, dataframe: DataFrame) -> Series:
        return _num(dataframe, 'close').le(_num(dataframe, 'avwap_from_high').mul(1.0 - float(self.reclaim_buffer_pct)))

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
        score_ok = self._score_guard(dataframe, prefix, side, score_min)
        context_ok = self._context_guard(dataframe, prefix, side, context_min)
        balance_ok = _num(dataframe, f'{prefix}_context_score_balance').ge(context_min)
        if mode == 'score':
            return score_ok
        if mode == 'context':
            return context_ok
        if mode == 'score_or_context':
            return score_ok | context_ok
        if mode == 'balance':
            return balance_ok
        if side == 'long':
            return _num(dataframe, f'{prefix}_market_context').ge(0)
        return _num(dataframe, f'{prefix}_market_context').le(0)
    SOURCE_ENTRY_STEM = 'multi3_prior_avwap_vp_breakout_long'
    FOCUSED_EXIT_CONTRACT = 'source_invalidation'
    FOCUSED_SOURCE_PROFILE = {'side': 'long', 'min_level_distance': 0.001, 'invalidation': {'provider': 'crossed_locked_month_high_loss', 'mode': 'level', 'long': {'level': 'prior_month_high', 'available': None}}}
    FOCUSED_EXIT_PLANS = {'source_invalidation_close1': {'contract': 'source_invalidation', 'name': 'source_invalidation_close1', 'action_sequence': ('source_invalidation_confirmation', 'invalidation_full', 'hard_stop', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'invalidation_confirmations': 1, 'invalidation_band': 0.0025}, 'source_invalidation_close2': {'contract': 'source_invalidation', 'name': 'source_invalidation_close2', 'action_sequence': ('source_invalidation_confirmation', 'invalidation_full', 'hard_stop', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'invalidation_confirmations': 2, 'invalidation_band': 0.0025}, 'source_invalidation_close2_band_0_5': {'contract': 'source_invalidation', 'name': 'source_invalidation_close2_band_0_5', 'action_sequence': ('source_invalidation_confirmation', 'invalidation_full', 'hard_stop', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'invalidation_confirmations': 2, 'invalidation_band': 0.005}}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'high', 'low', 'open', 'prior_month_high')
    FOCUSED_STATE_VERSION = 1
    FOCUSED_STATE_KEY = 'sieve3_v2_focused:Sieve3V2SourceInvalidationPriorMonthHighLossFromMulti3PriorAvwapVpBreakoutLong:source_invalidation'
    LOCKED_BUY_PARAMS = {'use_sieve2_vp_guard': False, 'sieve2_vp_guard_mode': 'score_or_context', 'sieve2_vp_window': 96, 'sieve2_vp_bins': 36, 'sieve2_vp_score_min': 0.25, 'sieve2_vp_context_min': 0.28, 'use_sieve2_market_guard': False, 'sieve2_market_guard_mode': 'pressure_or_trend', 'sieve2_market_window': 24, 'sieve2_market_pressure_min': 0.07, 'sieve2_market_trend_min': 0.25, 'sieve2_rs_benchmark_pair': 'BTC/USDT:USDT', 'sieve2_rs_score_min': 0.45, 'breakout_buffer_pct': 0.003, 'reclaim_buffer_pct': 0.0, 'sweep_buffer_pct': 0.006, 'zone_near_pct': 0.01, 'rolling_level_lookback': 72, 'equal_level_lookback': 72, 'equal_level_tolerance_pct': 0.006, 'equal_level_min_touches': 2, 'multi_period': 'month', 'avwap_anchor_lookback': 240, 'avwap_band_mult': 1.25, 'zone_impulse_window': 36, 'zone_impulse_atr_min': 0.8, 'zone_body_fraction_min': 0.55, 'zone_volume_ratio_min': 1.1, 'zone_max_age_bars': 72, 'use_volume_guard': True, 'volume_window': 24, 'volume_ratio_min': 0.8, 'pressure_min': 0.05, 'use_close_direction_guard': False, 'vp_window': 96, 'vp_bins': 48, 'vp_value_area_pct': 0.7, 'vp_price_source': 'hlc3', 'vp_smooth_bins': 3, 'vp_hvn_threshold': 0.7, 'vp_lvn_threshold': 0.35, 'vp_pressure_delta_min': 0.05, 'vp_node_near_pct': 0.01, 'vp_volume_percentile_min': 0.55, 'vp_score_window': 48, 'vp_fast_traverse_atr_mult': 1.2, 'vp_entry_score_margin': 0.02, 'use_vp_1h_guard': False, 'vp_guard_mode': 'context', 'vp_score_min': 0.25, 'vp_context_min': 0.6, 'use_vp_4h_guard': False, 'vp_4h_window': 48, 'vp_4h_bins': 36, 'use_vp_1d_guard': False, 'vp_1d_window': 30, 'vp_1d_bins': 36}
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
            if role == 'invalidation':
                raise ValueError('required invalidation profile has no side binding')
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
            invalidation_mode = str(self.FOCUSED_SOURCE_PROFILE.get('invalidation', {}).get('mode'))
            if role == 'invalidation' and invalidation_mode != 'evidence':
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
        if post_entry.empty:
            return False
        side = str(state['side'])
        provider = self.FOCUSED_SOURCE_PROFILE['invalidation']
        binding = provider[side]
        mode = str(provider['mode'])
        band = float(plan.get('invalidation_band') or 0.0)
        level = self._focused_float((state.get('levels') or {}).get('invalidation'))
        close = self._focused_number(post_entry, 'close')
        if mode in {'level', 'level_and_evidence', 'level_or_evidence'} and level is None:
            level_column = binding.get('level')
            raise ValueError(f'active trade state has no finite required invalidation level: {level_column}')
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


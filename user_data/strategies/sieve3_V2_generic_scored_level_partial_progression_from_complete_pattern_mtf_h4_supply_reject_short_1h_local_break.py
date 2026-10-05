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
from user_data.Indicators.pattern_bos_choch import add_bos_choch
ENTRY_MODE = 'entry_sieve2_mtf_h4_supply_reject_short_1h_local_break'
ENTRY_TAG = 'sieve2_mtf_h4_supply_reject_short_1h_local_break'
SIDE = 'short'
TIMEFRAME = '1h'
CONTEXT_TIMEFRAME = '4h'
HTF_BEHAVIOR = 'res_reject_short'
LTF_TRIGGER = 'local_break'


def _num(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    return pd.Series(frame[column], index=frame.index).astype('boolean').fillna(False).astype(bool)



SOURCE_ENTRY_STEM = 'complete_pattern_mtf_h4_supply_reject_short_1h_local_break'
LOCKED_BUY_SOURCE = 'verified tested-snapshot effective buy lock'
SOURCE_ENTRY_SIGNATURE_SHA256 = '51736107e19e6d89fbc1911220a6cf4decc5c26332a50e26102ec2bd585eaaa8'
SECONDARY_GUARD = 'locked selected source and Sieve2 optional guards'
SIEVE_STAGE = "sieve3"
SOURCE_ENTRY_STAGE = 'sieve2'
SOURCE_ENTRY_CLASS = 'Sieve2MtfH4SupplyRejectShort1hLocalBreak'
SOURCE_STRATEGY = 'D:\\FreqTradeStuffLargeData\\sieve_runtime\\entry_sieve\\backtests\\sieve2_mtf_h4_supply_reject_short_1h_local_break__auto_generic_1h_2020_q2_q3\\full_cycle_2020_2026\\tp_3_sl_2\\backtest-result-2026-05-25_09-27-47.zip!backtest-result-2026-05-25_09-27-47_Sieve2MtfH4SupplyRejectShort1hLocalBreak.py:Sieve2MtfH4SupplyRejectShort1hLocalBreak'
LINEAGE_STATUS = 'verified'
EXIT_FAMILY = 'generic_scored_level_partial_progression'
EXIT_THEORY = 'generic_scored_level_partial_progression'
EXIT_HYPOTHESIS = 'Use the deterministic retained level-zone score, advance through clean passes, take one or two coarse partials on confirmed reactions, and optionally ratchet protection behind progress.'
PRIMARY_TRIGGER = 'shifted 4h supply rejection plus 1h local-low break'
PRIMARY_GUARD = 'exact selected entry guards with frozen buy defaults'
TARGET_PROVIDER = 'closed-candle actual-fill-relative deterministic named level zone'
INVALIDATION_PROVIDER = 'coherent source invalidation when available, otherwise preserved 3% hard stop'
ACTIVE_SELL_PARAMS = ('zone_width_quarter_percent', 'partial_1_five_percent_units', 'partial_2_five_percent_units', 'ratchet_lag_levels', 'reaction_profile')

ENTRY_SOURCE_STAGE = "sieve2"
RESEARCH_PATH = 'sieve3_exit_final_generic_levels'
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume:457"

class Sieve3V2GenericScoredLevelPartialProgressionFromCompletePatternMtfH4SupplyRejectShort1HLocalBreak(IStrategy):
    """MTF Sieve2 probe: 4h supply/resistance rejection; 1h local breakdown."""
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 420
    process_only_new_candles = True
    can_short = True
    max_entry_position_adjustment = 0
    context_lookback = 24
    context_recent_bars = 2
    local_lookback = 48
    retest_buffer_pct = 0.012
    breakout_buffer_pct = 0.008
    volume_ratio_min = 0.8
    pressure_min = 0.07
    vp_score_min = 0.25
    vp_context_min = 0.38
    # Inactive buy declaration: use_ltf_volume_guard = BooleanParameter(default=True, space="buy"); fixed-on entry gate makes its guard unconditional.
    # Inactive buy declaration: use_ltf_pressure_guard = BooleanParameter(default=True, space="buy"); fixed-on entry gate makes its guard unconditional.
    # Inactive buy declaration: use_htf_turn_guard = BooleanParameter(default=True, space="buy"); fixed-on entry gate makes its guard unconditional.

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = (pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs)
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        if not getattr(self, 'dp', None):
            return []
        return [(pair, CONTEXT_TIMEFRAME) for pair in self.dp.current_whitelist()]

    def _sieve3_entry_populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self._add_ltf_features(dataframe)
        dataframe = self._merge_htf_context(dataframe, metadata or {})
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
        frame['mtf_htf_turn_bad'] = (close.gt(trend_ema) & pressure_ratio.ge(pressure_min)).fillna(False)
        return frame

    def _ltf_trigger(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        _num(dataframe, 'open')
        _num(dataframe, 'high')
        _num(dataframe, 'low')
        _num(dataframe, 'mtf_htf_level')
        _num(dataframe, 'mtf_ltf_ema')
        local_high = _num(dataframe, 'mtf_ltf_local_high')
        local_low = _num(dataframe, 'mtf_ltf_local_low')
        float(self.retest_buffer_pct)
        breakout_buffer = float(self.breakout_buffer_pct)
        _num(dataframe, 'mtf_ltf_pressure')
        return close.lt(local_low.mul(1.0 - breakout_buffer)) if SIDE == 'short' else close.gt(local_high.mul(1.0 + breakout_buffer))

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = None
        condition = _bool(dataframe, 'mtf_htf_context') & self._ltf_trigger(dataframe)
        condition &= _num(dataframe, 'mtf_ltf_volume_ratio').ge(float(self.volume_ratio_min))
        pressure = _num(dataframe, 'mtf_ltf_pressure')
        condition &= pressure.le(-float(self.pressure_min)) if SIDE == 'short' else pressure.ge(float(self.pressure_min))
        condition &= ~_bool(dataframe, 'mtf_htf_turn_bad')
        valid = condition.fillna(False) & dataframe['volume'].gt(0.0) & dataframe['close'].notna()
        dataframe.loc[valid, 'enter_short'] = 1
        dataframe.loc[valid, 'enter_tag'] = ENTRY_TAG
        return dataframe

    @informative('4h', fmt='gsl_{column}_{timeframe}')
    def populate_generic_levels_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        return self._generic_level_frame(dataframe, '4h')

    @informative('8h', fmt='gsl_{column}_{timeframe}')
    def populate_generic_levels_8h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        return self._generic_level_frame(dataframe, '8h')

    @informative('1d', fmt='gsl_{column}_{timeframe}')
    def populate_generic_levels_1d_(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        return self._generic_level_frame(dataframe, '1d')

    @informative('3d', fmt='gsl_{column}_{timeframe}')
    def populate_generic_levels_3d_(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        return self._generic_level_frame(dataframe, '3d')

    @informative('1h', 'BTC/USDT:USDT', fmt='gsbtc_{column}_{timeframe}')
    def populate_generic_btc_levels_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        return self._generic_level_frame(dataframe, '1h')

    @informative('4h', 'BTC/USDT:USDT', fmt='gsbtc_{column}_{timeframe}')
    def populate_generic_btc_levels_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        return self._generic_level_frame(dataframe, '4h')

    @informative('8h', 'BTC/USDT:USDT', fmt='gsbtc_{column}_{timeframe}')
    def populate_generic_btc_levels_8h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        return self._generic_level_frame(dataframe, '8h')

    @informative('1d', 'BTC/USDT:USDT', fmt='gsbtc_{column}_{timeframe}')
    def populate_generic_btc_levels_1d_(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        return self._generic_level_frame(dataframe, '1d')

    @informative('3d', 'BTC/USDT:USDT', fmt='gsbtc_{column}_{timeframe}')
    def populate_generic_btc_levels_3d_(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        return self._generic_level_frame(dataframe, '3d')

    def _generic_level_frame(self, dataframe: DataFrame, _timeframe: str) -> DataFrame:
        return add_volume_profile(dataframe, prefix='gsvp')
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        frame = self._sieve3_entry_populate_indicators(dataframe, metadata)
    
        if True:
            frame = self._generic_level_frame(frame, self.timeframe)
        return frame

    SOURCE_ENTRY_STEM = 'complete_pattern_mtf_h4_supply_reject_short_1h_local_break'
    FOCUSED_EXIT_CONTRACT = 'generic_scored_level_partial_progression'
    FOCUSED_SOURCE_PROFILE = {'side': 'short', 'min_level_distance': 0.001}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'high', 'low', 'open')
    FOCUSED_STATE_VERSION = 1
    FOCUSED_STATE_KEY = 'sieve3_v2_generic:Sieve3V2GenericScoredLevelPartialProgressionFromCompletePatternMtfH4SupplyRejectShort1HLocalBreak:generic_scored_level_partial_progression'
    LOCKED_BUY_PARAMS = {'use_sieve2_vp_guard': False, 'sieve2_vp_guard_mode': 'score_or_context', 'sieve2_vp_window': 96, 'sieve2_vp_bins': 36, 'sieve2_vp_score_min': 0.25, 'sieve2_vp_context_min': 0.28, 'use_sieve2_market_guard': False, 'sieve2_market_guard_mode': 'pressure_or_trend', 'sieve2_market_window': 24, 'sieve2_market_pressure_min': 0.07, 'sieve2_market_trend_min': 0.25, 'sieve2_rs_benchmark_pair': 'BTC/USDT:USDT', 'sieve2_rs_score_min': 0.45, 'context_lookback': 24, 'context_recent_bars': 2, 'local_lookback': 48, 'retest_buffer_pct': 0.012, 'breakout_buffer_pct': 0.008, 'volume_ratio_min': 0.8, 'pressure_min': 0.07, 'vp_score_min': 0.25, 'vp_context_min': 0.38, 'use_ltf_volume_guard': True, 'use_ltf_pressure_guard': True, 'use_htf_turn_guard': True}
    ACTIVE_SELL_PARAMS = ('zone_width_quarter_percent', 'partial_1_five_percent_units', 'partial_2_five_percent_units', 'ratchet_lag_levels', 'reaction_profile')
    GENERIC_MODE = 'scored_partial'
    GENERIC_LEVEL_FAMILY = 'all'
    GENERIC_TIMEFRAME_SCOPE = 'all'
    GENERIC_AVAILABLE_TIMEFRAMES = ('1h', '4h', '8h', '1d', '3d')
    GENERIC_HVN_MIN_STRENGTH = 0.5
    GENERIC_MIN_LEVEL_DISTANCE = 0.001
    GENERIC_HARD_STOP_RATIO = 0.03
    position_adjustment_enable = True
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    minimal_roi = {'0': 100.0}
    stoploss = -0.99
    zone_width_quarter_percent = IntParameter(1, 12, default=4, space='sell', optimize=True, load=True)
    zone_width_quarter_percent.batch_tags = ('family:exits', 'mode:sieve3_exit')
    partial_1_five_percent_units = IntParameter(1, 8, default=5, space='sell', optimize=True, load=True)
    partial_1_five_percent_units.batch_tags = ('family:exits', 'mode:sieve3_exit')
    partial_2_five_percent_units = IntParameter(0, 8, default=3, space='sell', optimize=True, load=True)
    partial_2_five_percent_units.batch_tags = ('family:exits', 'mode:sieve3_exit')
    ratchet_lag_levels = IntParameter(0, 2, default=1, space='sell', optimize=True, load=True)
    ratchet_lag_levels.batch_tags = ('family:exits', 'mode:sieve3_exit')
    reaction_profile = CategoricalParameter(('2_of_2', '2_of_3', '3_of_3'), default='2_of_3', space='sell', optimize=True, load=True)
    reaction_profile.batch_tags = ('family:exits', 'mode:sieve3_exit')

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

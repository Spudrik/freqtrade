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
from user_data.Indicators.complex_volume_profile import add_volume_profile
from user_data.Indicators.complex_trendline_projection_v2 import add_trendline_projection_v2


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

def _cross_above(series: Series, level: Series) -> Series:
    return series.gt(level) & series.shift(1).le(level.shift(1))

def _cross_below(series: Series, level: Series) -> Series:
    return series.lt(level) & series.shift(1).ge(level.shift(1))
ENTRY_MODE = 'entry_multi2_vp_prior_week_high_break_vp_bullctx_long'
ENTRY_TAG = 'multi2_vp_prior_week_high_break_vp_bullctx_long'
ENTRY_SOURCE_STAGE = "sieve2"
SIDE = 'long'
TIMEFRAME = '1h'
VP_CONFIRM = 'bull_context'
PRIOR_KIND = 'week_high_breakout'
SIEVE_STAGE = "sieve3"
SOURCE_ENTRY_STAGE = 'sieve2'
SOURCE_ENTRY_CLASS = 'Sieve3ExitBreakevenFromMulti2VpPriorWeekHighBreakVpBullctxLong'
SOURCE_STRATEGY = 'user_data/strategies/sieve3_exit_breakeven_from_multi2_vp_prior_week_high_break_vp_bullctx_long.py:Sieve3ExitBreakevenFromMulti2VpPriorWeekHighBreakVpBullctxLong'
LINEAGE_STATUS = 'verified'
EXIT_FAMILY = 'target_quality_stage1_tlv2_quality'
EXIT_THEORY = 'target_quality_stage1_tlv2_quality'
EXIT_HYPOTHESIS = 'Stage One compares actual-fill-relative TLV2 line filtered by score, pivots, and age while preserving the source entry and invalidation behavior.'
PRIMARY_TRIGGER = 'none'
PRIMARY_GUARD = 'none'
TARGET_PROVIDER = 'actual-fill-relative TLV2 line filtered by score, pivots, and age'
INVALIDATION_PROVIDER = 'provider=crossed_prior_week_high_loss;mode=level;long.level=prior_week_high'
ACTIVE_SELL_PARAMS = ('target_action', 'target_band_quarter_percent', 'tlv2_score_tenths', 'tlv2_min_pivots', 'tlv2_max_age_mult')

RESEARCH_PATH = 'sieve3_v2_target_quality_stage1'
SOURCE_RESULT_BATCH = "historical_promotion_lineage_unknown"


PRIMARY_TARGET = 'actual-fill-relative TLV2 line filtered by score, pivots, and age'

class Sieve3V2TqTlv2QualityFromMulti2VpPriorWeekHighBreakVpBullctxLong(IStrategy):
    """Sieve2 multi2 confluence probe."""
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 220
    process_only_new_candles = True
    can_short = False
    max_entry_position_adjustment = 0
    use_volume_guard = True
    volume_guard_window = 12
    volume_ratio_min = 1.0
    use_pressure_guard = True
    pressure_window = 12
    pressure_min = 0.35
    # Legacy inactive entry parameter: use_accumulation_guard=False (enable gate is fixed off).
    # Legacy inactive entry parameter: use_body_direction_guard=False (enable gate is fixed off).
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
    vp_score_min = 0.4
    vp_context_min = 0.2
    vp_level_buffer_pct = 0.003
    level_buffer_pct = 0.006
    reclaim_buffer_pct = 0.0
    level_lookback = 96
    equal_level_tolerance_pct = 0.006
    equal_level_min_touches = 3

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = (pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs)
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype='bool')
        close = _num(dataframe, 'close')
        if bool(self.use_volume_guard):
            volume = _num(dataframe, 'volume').clip(lower=0.0)
            window = int(self.volume_guard_window)
            baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
            guard &= volume.ge(baseline.mul(float(self.volume_ratio_min)))
        if bool(self.use_pressure_guard):
            open_ = _num(dataframe, 'open')
            high = _num(dataframe, 'high')
            low = _num(dataframe, 'low')
            volume = _num(dataframe, 'volume').clip(lower=0.0).fillna(0.0)
            candle_range = (high - low).replace(0.0, np.nan)
            body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
            close_location = ((close - low) / candle_range * 2.0 - 1.0).clip(-1.0, 1.0)
            pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
            directional_volume = (pressure * volume).fillna(0.0)
        if bool(self.use_pressure_guard):
            window = int(self.pressure_window)
            pressure_baseline = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
            pressure_ratio = directional_volume.rolling(window, min_periods=max(2, window // 3)).sum() / pressure_baseline
            guard &= pressure_ratio.le(-float(self.pressure_min)) if SIDE == 'short' else pressure_ratio.ge(float(self.pressure_min))
        return guard.fillna(False)

    def _add_vp(self, dataframe: DataFrame) -> DataFrame:
        return add_volume_profile(dataframe, window=int(self.vp_window), bins=int(self.vp_bins), value_area_pct=float(self.vp_value_area_pct), price_source=str(self.vp_price_source), smooth_bins=int(self.vp_smooth_bins), hvn_threshold=float(self.vp_hvn_threshold), lvn_threshold=float(self.vp_lvn_threshold), pressure_delta_min=float(self.vp_pressure_delta_min), node_near_pct=float(self.vp_node_near_pct), volume_percentile_min=float(self.vp_volume_percentile_min), score_window=int(self.vp_score_window), fast_traverse_atr_mult=float(self.vp_fast_traverse_atr_mult), entry_score_margin=float(self.vp_entry_score_margin), prefix='vp')

    def _vp_confirm(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        score_long = _num(dataframe, 'vp_score_long')
        score_short = _num(dataframe, 'vp_score_short')
        bull_context = _num(dataframe, 'vp_context_score_bull')
        bear_context = _num(dataframe, 'vp_context_score_bear')
        balance = _num(dataframe, 'vp_context_score_balance')
        market = _num(dataframe, 'vp_market_context')
        score_min = float(self.vp_score_min)
        context_min = float(self.vp_context_min)
        near = float(self.vp_level_buffer_pct)
        if VP_CONFIRM == 'bull_context':
            return score_long.ge(score_min) & bull_context.ge(context_min) & bull_context.ge(bear_context) & market.ge(0)
        if VP_CONFIRM == 'bear_context':
            return score_short.ge(score_min) & bear_context.ge(context_min) & bear_context.ge(bull_context) & market.le(0)
        if VP_CONFIRM == 'val_reclaim':
            val = _num(dataframe, 'vp_val', np.nan)
            return close.ge(val.mul(1.0 - near)) & (score_long.ge(score_min) | _bool(dataframe, 'vp_entry_trigger_long') | balance.ge(context_min))
        if VP_CONFIRM == 'vah_reject':
            vah = _num(dataframe, 'vp_vah', np.nan)
            return close.le(vah.mul(1.0 + near)) & (score_short.ge(score_min) | _bool(dataframe, 'vp_entry_trigger_short') | balance.ge(context_min))
        if VP_CONFIRM == 'node_entry':
            return _bool(dataframe, 'vp_node_entry_short') | score_short.ge(score_min) if SIDE == 'short' else _bool(dataframe, 'vp_node_entry_long') | score_long.ge(score_min)
        return pd.Series(True, index=dataframe.index, dtype='bool')

    def _date_series(self, dataframe: DataFrame) -> Series:
        if 'date' in dataframe.columns:
            return pd.to_datetime(dataframe['date'], utc=True, errors='coerce')
        return pd.Series(pd.to_datetime(dataframe.index, utc=True, errors='coerce'), index=dataframe.index)

    def _period_key(self, dates: Series, period: str) -> Series:
        if period == 'week':
            iso = dates.dt.isocalendar()
            return iso['year'].astype('string').str.cat(iso['week'].astype('string').str.zfill(2), sep='-')
        if period == 'month':
            return dates.dt.strftime('%Y-%m')
        return dates.dt.strftime('%Y-%m-%d')

    def _add_prior_levels(self, dataframe: DataFrame) -> DataFrame:
        dates = self._date_series(dataframe)
        high = _num(dataframe, 'high')
        low = _num(dataframe, 'low')
        for period in ('day', 'week', 'month'):
            key = self._period_key(dates, period)
            levels = pd.DataFrame({'key': key, 'high': high, 'low': low}, index=dataframe.index).groupby('key', sort=False).agg({'high': 'max', 'low': 'min'})
            levels[f'prior_{period}_high'] = levels['high'].shift(1)
            levels[f'prior_{period}_low'] = levels['low'].shift(1)
            dataframe[f'prior_{period}_high'] = key.map(levels[f'prior_{period}_high'])
            dataframe[f'prior_{period}_low'] = key.map(levels[f'prior_{period}_low'])
        lookback = int(self.level_lookback)
        min_periods = max(8, lookback // 4)
        dataframe['rolling_high'] = high.shift(1).rolling(lookback, min_periods=min_periods).max()
        dataframe['rolling_low'] = low.shift(1).rolling(lookback, min_periods=min_periods).min()
        tol = float(self.equal_level_tolerance_pct)
        dataframe['equal_high_touches'] = high.shift(1).rolling(lookback, min_periods=min_periods).apply(lambda values: float(np.sum(values >= np.nanmax(values) * (1.0 - tol))) if np.isfinite(np.nanmax(values)) else 0.0, raw=True)
        dataframe['equal_low_touches'] = low.shift(1).rolling(lookback, min_periods=min_periods).apply(lambda values: float(np.sum(values <= np.nanmin(values) * (1.0 + tol))) if np.isfinite(np.nanmin(values)) else 0.0, raw=True)
        return dataframe

    def _prior_trigger(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        open_ = _num(dataframe, 'open')
        high = _num(dataframe, 'high')
        low = _num(dataframe, 'low')
        buffer = float(self.level_buffer_pct)
        reclaim = float(self.reclaim_buffer_pct)
        min_touches = int(self.equal_level_min_touches)
        if PRIOR_KIND.endswith('_high_breakout'):
            period = PRIOR_KIND.split('_', 1)[0]
            level = _num(dataframe, f'prior_{period}_high', np.nan)
            return _cross_above(close, level.mul(1.0 + buffer)) & close.gt(open_)
        if PRIOR_KIND.endswith('_low_breakdown'):
            period = PRIOR_KIND.split('_', 1)[0]
            level = _num(dataframe, f'prior_{period}_low', np.nan)
            return _cross_below(close, level.mul(1.0 - buffer)) & close.lt(open_)
        if PRIOR_KIND == 'range_low_reclaim':
            level = _num(dataframe, 'rolling_low', np.nan)
            return low.lt(level.mul(1.0 - buffer)) & close.gt(level.mul(1.0 + reclaim)) & close.gt(open_)
        if PRIOR_KIND == 'range_high_reject':
            level = _num(dataframe, 'rolling_high', np.nan)
            return high.gt(level.mul(1.0 + buffer)) & close.lt(level.mul(1.0 - reclaim)) & close.lt(open_)
        if PRIOR_KIND == 'equal_lows_reclaim':
            level = _num(dataframe, 'rolling_low', np.nan)
            touches = _num(dataframe, 'equal_low_touches')
            return touches.ge(min_touches) & low.lt(level.mul(1.0 - buffer)) & close.gt(level.mul(1.0 + reclaim)) & close.gt(open_)
        if PRIOR_KIND == 'equal_highs_reject':
            level = _num(dataframe, 'rolling_high', np.nan)
            touches = _num(dataframe, 'equal_high_touches')
            return touches.ge(min_touches) & high.gt(level.mul(1.0 + buffer)) & close.lt(level.mul(1.0 - reclaim)) & close.lt(open_)
        return pd.Series(False, index=dataframe.index, dtype='bool')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe = self._add_prior_levels(dataframe)
        dataframe = self._add_vp(dataframe)
        dataframe = add_trendline_projection_v2(
            dataframe,
            timeframe=self.timeframe,
            raw_line_output_count=3,
            output_prefix='tqtlv2',
        )
        dataframe['sieve3_tq_row_index'] = np.arange(len(dataframe), dtype='int64')
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = None
        condition = self._prior_trigger(dataframe) & self._vp_confirm(dataframe)
        condition &= self._common_guards(dataframe)
        valid = condition.fillna(False) & dataframe['volume'].gt(0.0) & dataframe['close'].notna()
        if SIDE == 'short':
            dataframe.loc[valid, 'enter_short'] = 1
        else:
            dataframe.loc[valid, 'enter_long'] = 1
        dataframe.loc[valid, 'enter_tag'] = ENTRY_TAG
        return dataframe
    SOURCE_ENTRY_STEM = 'multi2_vp_prior_week_high_break_vp_bullctx_long'
    FOCUSED_EXIT_CONTRACT = 'target_quality_stage1_tlv2_quality'
    FOCUSED_SOURCE_PROFILE = {'side': 'long', 'min_level_distance': 0.001, 'targets': {'target_1': {'provider': 'actual-fill-relative TLV2 line filtered by score, pivots, and age'}}, 'invalidation': {'provider': 'crossed_prior_week_high_loss', 'mode': 'level', 'long': {'level': 'prior_week_high', 'available': None}}}
    FOCUSED_EXIT_PLANS = {'touch_full': {'contract': 'target_quality_stage1_tlv2_quality', 'name': 'touch_full', 'hard_stop_ratio': 0.03, 'max_hold_candles': 336}, 'close_full': {'contract': 'target_quality_stage1_tlv2_quality', 'name': 'close_full', 'hard_stop_ratio': 0.03, 'max_hold_candles': 336}}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'high', 'low', 'open', 'prior_week_high', 'tqtlv2_support_line_rank0', 'tqtlv2_support_score_rank0', 'tqtlv2_support_pivot_count_rank0', 'tqtlv2_support_last_confirm_index_rank0', 'tqtlv2_resistance_line_rank0', 'tqtlv2_resistance_score_rank0', 'tqtlv2_resistance_pivot_count_rank0', 'tqtlv2_resistance_last_confirm_index_rank0', 'tqtlv2_support_line_rank1', 'tqtlv2_support_score_rank1', 'tqtlv2_support_pivot_count_rank1', 'tqtlv2_support_last_confirm_index_rank1', 'tqtlv2_resistance_line_rank1', 'tqtlv2_resistance_score_rank1', 'tqtlv2_resistance_pivot_count_rank1', 'tqtlv2_resistance_last_confirm_index_rank1', 'tqtlv2_support_line_rank2', 'tqtlv2_support_score_rank2', 'tqtlv2_support_pivot_count_rank2', 'tqtlv2_support_last_confirm_index_rank2', 'tqtlv2_resistance_line_rank2', 'tqtlv2_resistance_score_rank2', 'tqtlv2_resistance_pivot_count_rank2', 'tqtlv2_resistance_last_confirm_index_rank2', 'sieve3_tq_row_index')
    FOCUSED_STATE_VERSION = 1
    FOCUSED_STATE_KEY = 'sieve3_v2_tq:Sieve3V2TqTlv2QualityFromMulti2VpPriorWeekHighBreakVpBullctxLong:target_quality_stage1_tlv2_quality'
    LOCKED_BUY_PARAMS = {'use_sieve2_vp_guard': False, 'sieve2_vp_guard_mode': 'score_or_context', 'sieve2_vp_window': 96, 'sieve2_vp_bins': 36, 'sieve2_vp_score_min': 0.25, 'sieve2_vp_context_min': 0.28, 'use_sieve2_market_guard': False, 'sieve2_market_guard_mode': 'pressure_or_trend', 'sieve2_market_window': 24, 'sieve2_market_pressure_min': 0.07, 'sieve2_market_trend_min': 0.25, 'sieve2_rs_benchmark_pair': 'BTC/USDT:USDT', 'sieve2_rs_score_min': 0.45, 'use_volume_guard': True, 'volume_guard_window': 12, 'volume_ratio_min': 1.0, 'use_pressure_guard': True, 'pressure_window': 12, 'pressure_min': 0.35, 'use_accumulation_guard': False, 'use_body_direction_guard': False, 'use_close_direction_guard': False, 'vp_window': 96, 'vp_bins': 48, 'vp_value_area_pct': 0.7, 'vp_price_source': 'hlc3', 'vp_smooth_bins': 3, 'vp_hvn_threshold': 0.7, 'vp_lvn_threshold': 0.35, 'vp_pressure_delta_min': 0.05, 'vp_node_near_pct': 0.01, 'vp_volume_percentile_min': 0.55, 'vp_score_window': 48, 'vp_fast_traverse_atr_mult': 1.2, 'vp_entry_score_margin': 0.02, 'vp_score_min': 0.4, 'vp_context_min': 0.2, 'vp_level_buffer_pct': 0.003, 'level_buffer_pct': 0.006, 'reclaim_buffer_pct': 0.0, 'level_lookback': 96, 'equal_level_tolerance_pct': 0.006, 'equal_level_min_touches': 3}
    ACTIVE_SELL_PARAMS = ('target_action', 'target_band_quarter_percent', 'tlv2_score_tenths', 'tlv2_min_pivots', 'tlv2_max_age_mult')
    position_adjustment_enable = False
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    minimal_roi = {'0': 100.0}
    stoploss = -0.99
    target_action = CategoricalParameter(
        ('touch_full', 'close_full'),
        default='close_full',
        space='sell',
        optimize=True,
        load=True,
    )
    target_action.batch_tags = ('family:exits', 'mode:sieve3_exit')
    target_band_quarter_percent = IntParameter(
        0, 12, default=2, space='sell', optimize=True, load=True
    )
    target_band_quarter_percent.batch_tags = ('family:exits', 'mode:sieve3_exit')
    TARGET_QUALITY_AGE_STEP_BARS = 8
    tlv2_score_tenths = IntParameter(
        5, 9, default=5, space='sell', optimize=True, load=True
    )
    tlv2_score_tenths.batch_tags = ('family:exits', 'mode:sieve3_exit')
    tlv2_min_pivots = IntParameter(
        3, 5, default=3, space='sell', optimize=True, load=True
    )
    tlv2_min_pivots.batch_tags = ('family:exits', 'mode:sieve3_exit')
    tlv2_max_age_mult = IntParameter(
        1, 8, default=4, space='sell', optimize=True, load=True
    )
    tlv2_max_age_mult.batch_tags = ('family:exits', 'mode:sieve3_exit')

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
        name = str((state or {}).get('plan') or self.target_action.value)
        if name not in self.FOCUSED_EXIT_PLANS:
            raise ValueError(f'unknown focused exit plan: {name}')
        plan = dict(self.FOCUSED_EXIT_PLANS[name])
        plan['target_confirmation'] = {
            'touch_full': 'touch',
            'close_full': 'close',
        }[name]
        plan['target_band'] = float(self.target_band_quarter_percent.value) * 0.0025
        plan['invalidation_band'] = 0.0025
        plan['invalidation_confirmations'] = 1
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

    def _stage1_candidate(
        self,
        row: Series,
        name: str,
        level_column: str,
        metadata_columns: Mapping[str, str] | None = None,
    ) -> dict[str, Any] | None:
        level = self._focused_float(row[level_column])
        if level is None or level <= 0.0:
            return None
        metadata: dict[str, float] = {}
        for label, column in (metadata_columns or {}).items():
            value = self._focused_float(row[column])
            if value is not None:
                metadata[str(label)] = value
        if 'last_confirm_index' in metadata:
            row_index = self._focused_float(row['sieve3_tq_row_index'])
            if row_index is not None:
                metadata['age_bars'] = max(
                    0.0,
                    row_index - metadata['last_confirm_index'],
                )
        return {'name': name, 'level': float(level), 'metadata': metadata}

    def _stage1_forward_candidates(
        self,
        row: Series,
        candidates: list[dict[str, Any]],
        side: str,
        entry_rate: float,
    ) -> list[dict[str, Any]]:
        minimum = float(self.FOCUSED_SOURCE_PROFILE['min_level_distance'])
        if side == 'short':
            valid = [
                candidate
                for candidate in candidates
                if float(candidate['level']) < entry_rate * (1.0 - minimum)
            ]
        else:
            valid = [
                candidate
                for candidate in candidates
                if float(candidate['level']) > entry_rate * (1.0 + minimum)
            ]
        invalidation_binding = self._focused_side_binding('invalidation', side)
        invalidation_level = self._focused_frozen_level(
            row,
            invalidation_binding,
            side,
            entry_rate,
            'invalidation',
        )
        if invalidation_level is not None:
            valid = [
                candidate
                for candidate in valid
                if not math.isclose(
                    float(candidate['level']),
                    invalidation_level,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ]
        return sorted(
            valid,
            key=lambda candidate: abs(float(candidate['level']) - entry_rate),
        )

    def _stage1_select_target(
        self,
        row: Series,
        side: str,
        entry_rate: float,
    ) -> tuple[float | None, dict[str, Any]]:
        line_side = 'support' if side == 'short' else 'resistance'
        candidates: list[dict[str, Any]] = []
        for rank in range(3):
            candidate = self._stage1_candidate(
                row,
                f'tqtlv2_{line_side}_rank{rank}',
                f'tqtlv2_{line_side}_line_rank{rank}',
                {
                    'score': f'tqtlv2_{line_side}_score_rank{rank}',
                    'pivot_count': f'tqtlv2_{line_side}_pivot_count_rank{rank}',
                    'last_confirm_index': (
                        f'tqtlv2_{line_side}_last_confirm_index_rank{rank}'
                    ),
                },
            )
            if candidate is not None:
                candidates.append(candidate)
        candidates = self._stage1_forward_candidates(row, candidates, side, entry_rate)
        minimum_score = float(self.tlv2_score_tenths.value) * 0.1
        minimum_pivots = int(self.tlv2_min_pivots.value)
        maximum_age = (
            int(self.tlv2_max_age_mult.value)
            * int(self.TARGET_QUALITY_AGE_STEP_BARS)
        )
        qualified = [
            candidate
            for candidate in candidates
            if float(candidate['metadata'].get('score', 0.0)) >= minimum_score
            and float(candidate['metadata'].get('pivot_count', 0.0)) >= minimum_pivots
            and float(candidate['metadata'].get('age_bars', float('inf')))
            <= maximum_age
        ]
        thresholds = {
            'minimum_score': minimum_score,
            'minimum_pivots': minimum_pivots,
            'maximum_age_bars': maximum_age,
        }
        if not qualified:
            return None, {
                'variant': 'tlv2_quality',
                'status': 'quality_rejected' if candidates else 'unavailable',
                'candidate_count': len(candidates),
                **thresholds,
            }
        selected = qualified[0]
        return float(selected['level']), {
            'variant': 'tlv2_quality',
            'status': 'selected',
            'candidate': str(selected['name']),
            'candidate_count': len(candidates),
            'qualified_count': len(qualified),
            'metadata': dict(selected['metadata']),
            **thresholds,
        }

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
        target_level, target_metadata = self._stage1_select_target(
            row,
            side,
            entry_rate,
        )
        levels: dict[str, float | None] = {'target_1': target_level}
        invalidation_binding = self._focused_side_binding(
            'invalidation',
            side,
        )
        levels['invalidation'] = self._focused_frozen_level(
            row,
            invalidation_binding,
            side,
            entry_rate,
            'invalidation',
        )
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
        selected_plan = str(self.target_action.value)
        fill_time = (getattr(order, 'order_filled_utc', None) if order is not None else None) or getattr(trade, 'date_entry_fill_utc', None) or getattr(trade, 'open_date_utc', None) or current_time
        fill_timestamp = self._focused_utc(fill_time)
        snapshot_date = self._focused_utc(row['date'])
        return {'version': self.FOCUSED_STATE_VERSION, 'contract': self.FOCUSED_EXIT_CONTRACT, 'plan': selected_plan, 'side': side, 'phase': 'pre_target', 'entry_rate': float(entry_rate), 'entry_filled_at': fill_timestamp.isoformat() if fill_timestamp is not None else None, 'entry_snapshot_candle': snapshot_date.isoformat() if snapshot_date is not None else None, 'levels': levels, 'target_metadata': target_metadata, 'target_states': {slot: {'status': 'available' if value is not None else 'unavailable', 'touched_at': None} for slot, value in levels.items() if slot.startswith('target_')}, 'invalidation_seen': False, 'guard_state': 'unknown', 'trigger_state': 'unknown', 'initial_stake': self._focused_float(getattr(trade, 'stake_amount', None)), 'favorable_rate': float(entry_rate), 'stop_price': None, 'terminal_pending': False, 'terminal_tag': None}

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

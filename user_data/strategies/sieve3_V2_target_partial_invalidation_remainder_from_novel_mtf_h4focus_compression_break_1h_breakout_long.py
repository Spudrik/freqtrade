from __future__ import annotations
import math
import pandas as pd
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from pandas import DataFrame, Series
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, stoploss_from_absolute, IntParameter
NOVEL_IDEA = True
UPDATE_HYPOTHESIS = 'Entry-only H4 compression break context with 1h breakout execution; tests volatility release continuation.'
import numpy as np
from freqtrade.strategy import IStrategy, merge_informative_pair
SIDE = 'long'

CONTEXT_MODE = 'h4_compression_break'
EXECUTION_MODE = 'breakout'
SIEVE_STAGE = "sieve3"
SOURCE_ENTRY_STAGE = 'sieve2'
SOURCE_ENTRY_CLASS = 'Sieve3ExitNovelMtfH4focusCompressionBreak1hBreakoutLong'
SOURCE_STRATEGY = 'user_data/strategies/sieve3_exit_novel_mtf_h4focus_compression_break_1h_breakout_long.py:Sieve3ExitNovelMtfH4focusCompressionBreak1hBreakoutLong'
LINEAGE_STATUS = 'verified'
EXIT_FAMILY = 'target_partial_invalidation_remainder'
EXIT_THEORY = 'target_partial_invalidation_remainder'
EXIT_HYPOTHESIS = 'Take an optional target partial, then manage the remainder against entry-specific invalidation and stop movement.'
PRIMARY_TRIGGER = '1h bullish breakout through the shifted prior_high_1h execution boundary with locked candle-body follow-through'
PRIMARY_GUARD = 'closed 4h compression break above prior_high_4h with the locked H4 confirmation; volume confirmation is disabled'
TARGET_PROVIDER = 'compression-width measured move'
INVALIDATION_PROVIDER = 'broken breakout/retest boundary'
ACTIVE_SELL_PARAMS = ('exit_plan', 'target_band_quarter_percent', 'partial_five_percent_units', 'stop_after_partial', 'invalidation_band_quarter_percent', 'invalidation_confirmations')

ENTRY_SOURCE_STAGE = "sieve2"
RESEARCH_PATH = 'sieve3_exit_target_partial_invalidation_remainder'
SOURCE_RESULT_BATCH = "historical_promotion_lineage_unknown"

class Sieve3V2TargetPartialInvalidationRemainderFromNovelMtfH4focusCompressionBreak1HBreakoutLong(IStrategy):
    timeframe = '1h'
    can_short = False
    startup_candle_count = 260
    process_only_new_candles = True
    max_entry_position_adjustment = 0
    INTERFACE_VERSION = 3
    context_lookback = 10
    h4_lookback = 44
    exec_lookback = 12
    body_ratio_min = 0.58
    range_ratio_min = 2.48
    volume_ratio_min = 1.67
    # Legacy inert entry parameter: retest_tolerance=0.012 (fixed entry mode/gate).
    close_follow_min = 0.35
    # Legacy inert entry parameter: require_h4_confirm=True (fixed entry mode/gate).
    # Legacy fixed-off entry parameter: require_volume_confirm=False (CategoricalParameter, buy space).

    def informative_pairs(self):
        dp = getattr(self, 'dp', None)
        if dp is None:
            return []
        pairs = self.dp.current_whitelist()
        return [(pair, '4h') for pair in pairs] + [(pair, '1d') for pair in pairs]

    def _sieve3_entry_populate_indicators(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        dataframe = dataframe.copy()
        dp = getattr(self, 'dp', None)
        if dp is not None and metadata and metadata.get('pair'):
            for tf in ('4h', '1d'):
                informative = dp.get_pair_dataframe(pair=metadata['pair'], timeframe=tf)
                if informative is not None and (not informative.empty):
                    dataframe = merge_informative_pair(dataframe, self._features(informative.copy(), tf), self.timeframe, tf, ffill=True)
        return self._features(dataframe, '1h')

    def _features(self, frame: DataFrame, tf: str) -> DataFrame:
        close = pd.to_numeric(frame['close'], errors='coerce')
        open_ = pd.to_numeric(frame['open'], errors='coerce')
        high = pd.to_numeric(frame['high'], errors='coerce')
        low = pd.to_numeric(frame['low'], errors='coerce')
        volume = pd.to_numeric(frame['volume'], errors='coerce').fillna(0.0)
        look = int(self.context_lookback if tf == '1d' else self.h4_lookback if tf == '4h' else self.exec_lookback)
        rng = (high - low).replace(0, np.nan)
        body = close - open_
        frame[f'body_ratio_{tf}'] = (body.abs() / rng).replace([np.inf, -np.inf], np.nan)
        frame[f'prior_high_{tf}'] = high.shift(1).rolling(look, min_periods=max(2, look // 3)).max()
        frame[f'prior_low_{tf}'] = low.shift(1).rolling(look, min_periods=max(2, look // 3)).min()
        frame[f'range_mid_{tf}'] = (frame[f'prior_high_{tf}'] + frame[f'prior_low_{tf}']) / 2.0
        frame[f'range_ratio_{tf}'] = rng / rng.rolling(look, min_periods=max(2, look // 3)).mean().replace(0, np.nan)
        frame[f'volume_ratio_{tf}'] = volume / volume.rolling(look, min_periods=max(2, look // 3)).mean().replace(0, np.nan)
        frame[f'hh_{tf}'] = high.gt(high.shift(1).rolling(look, min_periods=max(2, look // 3)).max())
        frame[f'll_{tf}'] = low.lt(low.shift(1).rolling(look, min_periods=max(2, look // 3)).min())
        frame[f'hl_{tf}'] = low.gt(low.shift(1).rolling(max(2, look // 2), min_periods=2).min())
        frame[f'lh_{tf}'] = high.lt(high.shift(1).rolling(max(2, look // 2), min_periods=2).max())
        return frame

    def _context(self, dataframe: DataFrame) -> Series:
        c1 = self._num(dataframe, 'close_1d')
        o1 = self._num(dataframe, 'open_1d')
        h1 = self._num(dataframe, 'high_1d')
        l1 = self._num(dataframe, 'low_1d')
        c4 = self._num(dataframe, 'close_4h')
        o4 = self._num(dataframe, 'open_4h')
        h4 = self._num(dataframe, 'high_4h')
        l4 = self._num(dataframe, 'low_4h')
        bull1 = c1.gt(o1) & self._num(dataframe, 'body_ratio_1d').ge(float(self.body_ratio_min))
        bear1 = c1.lt(o1) & self._num(dataframe, 'body_ratio_1d').ge(float(self.body_ratio_min))
        bull4 = c4.gt(o4)
        bear4 = c4.lt(o4)
        vol_ok = self._num(dataframe, 'volume_ratio_1d').ge(float(self.volume_ratio_min)) | (not self._bool_param(False))
        h4_long_ok = bull4 | (not self._bool_param(True))
        h4_short_ok = bear4 | (not self._bool_param(True))
        mode = CONTEXT_MODE
        if mode == 'prior_high_break':
            ctx = c1.gt(self._num(dataframe, 'prior_high_1d')) & bull1
        elif mode == 'failed_low_reclaim':
            ctx = l1.lt(self._num(dataframe, 'prior_low_1d')) & c1.gt(self._num(dataframe, 'prior_low_1d')) & bull1
        elif mode == 'trend_hhhl':
            ctx = self._bool(dataframe, 'hh_1d') & self._bool(dataframe, 'hl_1d') & c1.gt(self._num(dataframe, 'range_mid_1d'))
        elif mode == 'range_expansion':
            ctx = bull1 & self._num(dataframe, 'range_ratio_1d').ge(float(self.range_ratio_min))
        elif mode == 'midline_reclaim':
            ctx = c1.gt(self._num(dataframe, 'range_mid_1d')) & c1.shift(1).le(self._num(dataframe, 'range_mid_1d'))
        elif mode == 'compression_break':
            ctx = c1.gt(self._num(dataframe, 'prior_high_1d')) & self._num(dataframe, 'range_ratio_1d').le(float(self.range_ratio_min))
        elif mode == 'volume_break':
            ctx = c1.gt(self._num(dataframe, 'prior_high_1d')) & self._num(dataframe, 'volume_ratio_1d').ge(float(self.volume_ratio_min))
        elif mode == 'h4_prior_high_break':
            ctx = c4.gt(self._num(dataframe, 'prior_high_4h')) & bull4
        elif mode == 'h4_hl_reclaim':
            ctx = self._bool(dataframe, 'hl_4h') & c4.gt(self._num(dataframe, 'range_mid_4h'))
        elif mode == 'h4_failed_low_reclaim':
            ctx = l4.lt(self._num(dataframe, 'prior_low_4h')) & c4.gt(self._num(dataframe, 'prior_low_4h'))
        elif mode == 'h4_range_expansion':
            ctx = bull4 & self._num(dataframe, 'range_ratio_4h').ge(float(self.range_ratio_min))
        elif mode == 'h4_compression_break':
            ctx = c4.gt(self._num(dataframe, 'prior_high_4h')) & self._num(dataframe, 'range_ratio_4h').le(float(self.range_ratio_min))
        elif mode == 'dual_break':
            ctx = c1.gt(self._num(dataframe, 'prior_high_1d')) & c4.gt(self._num(dataframe, 'prior_high_4h'))
        elif mode == 'support_h4_break':
            ctx = c1.gt(self._num(dataframe, 'range_mid_1d')) & c4.gt(self._num(dataframe, 'prior_high_4h'))
        elif mode == 'pullback_h4_reclaim':
            ctx = c1.gt(self._num(dataframe, 'range_mid_1d')) & self._bool(dataframe, 'hl_4h')
        elif mode == 'prior_low_break':
            ctx = c1.lt(self._num(dataframe, 'prior_low_1d')) & bear1
        elif mode == 'failed_high_reject':
            ctx = h1.gt(self._num(dataframe, 'prior_high_1d')) & c1.lt(self._num(dataframe, 'prior_high_1d')) & bear1
        elif mode == 'trend_lhll':
            ctx = self._bool(dataframe, 'll_1d') & self._bool(dataframe, 'lh_1d') & c1.lt(self._num(dataframe, 'range_mid_1d'))
        elif mode == 'bear_range_expansion':
            ctx = bear1 & self._num(dataframe, 'range_ratio_1d').ge(float(self.range_ratio_min))
        elif mode == 'midline_reject':
            ctx = c1.lt(self._num(dataframe, 'range_mid_1d')) & c1.shift(1).ge(self._num(dataframe, 'range_mid_1d'))
        elif mode == 'compression_breakdown':
            ctx = c1.lt(self._num(dataframe, 'prior_low_1d')) & self._num(dataframe, 'range_ratio_1d').le(float(self.range_ratio_min))
        elif mode == 'volume_breakdown':
            ctx = c1.lt(self._num(dataframe, 'prior_low_1d')) & self._num(dataframe, 'volume_ratio_1d').ge(float(self.volume_ratio_min))
        elif mode == 'h4_prior_low_break':
            ctx = c4.lt(self._num(dataframe, 'prior_low_4h')) & bear4
        elif mode == 'h4_lh_reject':
            ctx = self._bool(dataframe, 'lh_4h') & c4.lt(self._num(dataframe, 'range_mid_4h'))
        elif mode == 'h4_failed_high_reject':
            ctx = h4.gt(self._num(dataframe, 'prior_high_4h')) & c4.lt(self._num(dataframe, 'prior_high_4h'))
        elif mode == 'h4_bear_range_expansion':
            ctx = bear4 & self._num(dataframe, 'range_ratio_4h').ge(float(self.range_ratio_min))
        elif mode == 'h4_compression_breakdown':
            ctx = c4.lt(self._num(dataframe, 'prior_low_4h')) & self._num(dataframe, 'range_ratio_4h').le(float(self.range_ratio_min))
        elif mode == 'dual_breakdown':
            ctx = c1.lt(self._num(dataframe, 'prior_low_1d')) & c4.lt(self._num(dataframe, 'prior_low_4h'))
        elif mode == 'resistance_h4_break':
            ctx = c1.lt(self._num(dataframe, 'range_mid_1d')) & c4.lt(self._num(dataframe, 'prior_low_4h'))
        elif mode == 'bounce_h4_reject':
            ctx = c1.lt(self._num(dataframe, 'range_mid_1d')) & self._bool(dataframe, 'lh_4h')
        else:
            ctx = pd.Series(False, index=dataframe.index)
        return (ctx & (h4_long_ok if SIDE == 'long' else h4_short_ok) & vol_ok).fillna(False)

    def _execution(self, dataframe: DataFrame) -> Series:
        close = self._num(dataframe, 'close')
        open_ = self._num(dataframe, 'open')
        ph = self._num(dataframe, 'prior_high_1h')
        follow = float(self.close_follow_min)
        bull = close.gt(open_) & self._num(dataframe, 'body_ratio_1h').ge(follow)
        exe = close.gt(ph) & bull
        return exe.fillna(False)


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self._sieve3_entry_populate_indicators(dataframe, metadata)
        close = self._num(dataframe, 'close')
        upper = self._num(dataframe, 'prior_high_4h')
        lower = self._num(dataframe, 'prior_low_4h')
        width = (upper - lower).clip(lower=0.0)
        target = upper + width
        invalidation = self._num(dataframe, 'prior_high_1h')
        dataframe['sieve3_exit_target_level'] = target.where(target.gt(close.mul(1.001)))
        dataframe['sieve3_exit_invalidation_level'] = invalidation.where(invalidation.lt(close))
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        condition = self._context(dataframe) & self._execution(dataframe)
        dataframe.loc[condition, ['enter_long', 'enter_tag']] = (1, f'{CONTEXT_MODE}_1h_{EXECUTION_MODE}_long')
        return dataframe

    @staticmethod
    def _column_alias(frame: DataFrame, column: str) -> str | None:
        if column in frame.columns:
            return column
        for suffix in ('_1d', '_4h'):
            if column.endswith(suffix):
                merged = f'{column}{suffix}'
                if merged in frame.columns:
                    return merged
        return None

    @staticmethod
    def _num(frame: DataFrame, column: str, default: float=0.0) -> Series:
        resolved = Sieve3V2TargetPartialInvalidationRemainderFromNovelMtfH4focusCompressionBreak1HBreakoutLong._column_alias(frame, column)
        if resolved is None:
            if column in frame.columns:
                resolved = column
            else:
                for suffix in ('_1d', '_4h'):
                    if column.endswith(suffix) and f'{column}{suffix}' in frame.columns:
                        resolved = f'{column}{suffix}'
                        break
        if resolved is None:
            raise KeyError(f"missing required column: {column!r}")
        return pd.to_numeric(frame[resolved], errors='coerce').replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _bool(frame: DataFrame, column: str) -> Series:
        resolved = column if column in frame.columns else None
        if resolved is None:
            for suffix in ('_1d', '_4h'):
                if column.endswith(suffix) and f'{column}{suffix}' in frame.columns:
                    resolved = f'{column}{suffix}'
                    break
        if resolved is None:
            raise KeyError(f"missing required column: {column!r}")
        return pd.Series(frame[resolved], index=frame.index).astype('boolean').fillna(False).astype(bool)

    @staticmethod
    def _bool_param(param: Any) -> bool:
        value = getattr(param, 'value', param)
        if isinstance(value, str):
            return value.lower() in {'1', 'true', 'yes', 'on'}
        return bool(value)
    SOURCE_ENTRY_STEM = 'novel_mtf_h4focus_compression_break_1h_breakout_long'
    FOCUSED_EXIT_CONTRACT = 'target_partial_invalidation_remainder'
    FOCUSED_SOURCE_PROFILE = {'side': 'long', 'min_level_distance': 0.001, 'targets': {'target_1': {'provider': 'compression-width measured move', 'long': {'level': 'sieve3_exit_target_level', 'available': None}}}, 'invalidation': {'provider': 'broken breakout/retest boundary', 'mode': 'level', 'long': {'level': 'sieve3_exit_invalidation_level', 'available': None}}}
    FOCUSED_EXIT_PLANS = {'touch_partial': {'contract': 'target_partial_invalidation_remainder', 'name': 'touch_partial', 'hard_stop_ratio': 0.03, 'max_hold_candles': 336}, 'close_partial': {'contract': 'target_partial_invalidation_remainder', 'name': 'close_partial', 'hard_stop_ratio': 0.03, 'max_hold_candles': 336}, 'reversal_1_partial': {'contract': 'target_partial_invalidation_remainder', 'name': 'reversal_1_partial', 'hard_stop_ratio': 0.03, 'max_hold_candles': 336}, 'reversal_2_of_3_partial': {'contract': 'target_partial_invalidation_remainder', 'name': 'reversal_2_of_3_partial', 'hard_stop_ratio': 0.03, 'max_hold_candles': 336}}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'high', 'low', 'open', 'sieve3_exit_invalidation_level', 'sieve3_exit_target_level')
    FOCUSED_STATE_VERSION = 2
    FOCUSED_STATE_KEY = 'sieve3_v2_focused:Sieve3V2TargetPartialInvalidationRemainderFromNovelMtfH4focusCompressionBreak1HBreakoutLong:target_partial_invalidation_remainder'
    LOCKED_BUY_PARAMS = {'context_lookback': 10, 'h4_lookback': 44, 'exec_lookback': 12, 'body_ratio_min': 0.58, 'range_ratio_min': 2.48, 'volume_ratio_min': 1.67, 'retest_tolerance': 0.012, 'close_follow_min': 0.35, 'require_h4_confirm': True, 'require_volume_confirm': False}
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

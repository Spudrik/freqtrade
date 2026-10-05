from __future__ import annotations
import math
import pandas as pd
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from pandas import DataFrame, Series
from freqtrade.constants import NON_OPEN_EXCHANGE_STATES
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, stoploss_from_absolute, IntParameter
import numpy as np
from freqtrade.strategy import IStrategy
from user_data.Indicators.pattern_reversal import add_pattern_reversal

ENTRY_MODE = 'entry_reversal_double_bottom_present_long_4h'
ENTRY_TAG = 'reversal_double_bottom_present_long_4h'
ENTRY_SOURCE_STAGE = 'sieve2'
SIDE = 'long'
TIMEFRAME = '4h'


def _num(frame: DataFrame, column: str, _default: float | Series | object = ...) -> Series:
    if column not in frame.columns:
        if _default is ...:
            raise KeyError(f"required numeric column is missing: {column!r}")
        if isinstance(_default, Series):
            return pd.to_numeric(_default, errors="coerce").reindex(frame.index)
        return pd.Series(float(_default), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)

def _bool(frame: DataFrame, column: str) -> Series:
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)




SIEVE_STAGE = 'sieve3'
SOURCE_ENTRY_STAGE = 'sieve3'
SOURCE_ENTRY_CLASS = 'Sieve3ExitFixedTpSlFromCompletePatternReversalDoubleBottomPresentLong4H'
SOURCE_STRATEGY = 'user_data/strategies/sieve3_exit_fixed_tp_sl_from_complete_pattern_reversal_double_bottom_present_long_4h.py:Sieve3ExitFixedTpSlFromCompletePatternReversalDoubleBottomPresentLong4H'
LINEAGE_STATUS = 'parked_legacy_executable_surface_defaults_frozen; promotion_lock_unverified'
EXIT_FAMILY = 'target_partial_invalidation_remainder'
EXIT_THEORY = 'target_partial_invalidation_remainder_double_bottom_projection_neckline'
EXIT_HYPOTHESIS = 'The double-bottom projection takes a partial while a failure through the frozen neckline invalidates the remainder.'
PRIMARY_TRIGGER = 'pat_double_bottom_pattern_present with score >= 0.82 and close > pat_double_bottom_confirmation_level plus 0.5%'
PRIMARY_GUARD = 'selected 12-candle volume >= 1.3x baseline and 12-candle bullish pressure >= 0.35; optional Sieve2 guards disabled'
TARGET_PROVIDER = 'target_1[provider=entry-frozen double-bottom measured projection;long.level=pat_double_bottom_target_level]'
INVALIDATION_PROVIDER = 'provider=entry-frozen double-bottom confirmation neckline;mode=level;long.level=pat_double_bottom_confirmation_level'
ACTIVE_SELL_PARAMS = ('exit_plan', 'hard_stop_percent', 'max_hold_scale')



RESEARCH_PATH = 'sieve3_exit_target_partial_invalidation_remainder'
SOURCE_RESULT_BATCH = 'historical_promotion_lineage_unknown'


class Sieve3V2TargetPartialInvalidationRemainderDoubleBottomProjectionNecklineFromCompletePatternReversalDoubleBottomPresentLong4H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 160
    process_only_new_candles = True
    can_short = False
    max_entry_position_adjustment = 0
    
    
    
    
    
    
    
    
    
    
    
    
    
    use_volume_guard = True
    volume_guard_window = 12
    volume_ratio_min = 1.3
    use_pressure_guard = True
    pressure_window = 12
    pressure_min = 0.35
    use_accumulation_guard = False
    use_body_direction_guard = False
    use_close_direction_guard = False
    score_min = 0.82
    confirmation_buffer_pct = 0.005

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = (pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs)
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe = add_pattern_reversal(dataframe, timeframe=self.timeframe, min_double_quality=float(self.score_min), min_head_shoulders_quality=float(self.score_min))
        
        return dataframe

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype='bool')
        close = _num(dataframe, 'close')
        if bool(self.use_volume_guard):
            volume = _num(dataframe, 'volume').clip(lower=0.0)
            window = int(self.volume_guard_window)
            baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
            guard &= volume.ge(baseline.mul(float(self.volume_ratio_min)))
        if bool(self.use_pressure_guard) or bool(self.use_accumulation_guard):
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
        if bool(self.use_accumulation_guard):
            window = int(self.pressure_window)
            accumulation = directional_volume.rolling(window, min_periods=max(2, window // 3)).sum()
            guard &= accumulation.le(0.0) if SIDE == 'short' else accumulation.ge(0.0)
        if bool(self.use_body_direction_guard):
            open_ = _num(dataframe, 'open')
            guard &= close.lt(open_) if SIDE == 'short' else close.gt(open_)
        if bool(self.use_close_direction_guard):
            guard &= close.lt(close.shift(1)) if SIDE == 'short' else close.gt(close.shift(1))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = None
        level = _num(dataframe, 'pat_double_bottom_confirmation_level').mul(1.0 + float(self.confirmation_buffer_pct))
        condition = _bool(dataframe, 'pat_double_bottom_pattern_present') & _num(dataframe, 'pat_double_bottom_indicator_score').ge(float(self.score_min)) & _num(dataframe, 'close').gt(level)
        condition &= self._common_guards(dataframe)
        condition = pd.Series(condition, index=dataframe.index).fillna(False).astype(bool).fillna(False)
        valid = condition.fillna(False) & dataframe['volume'].gt(0.0) & dataframe['close'].notna()
        dataframe.loc[valid, 'enter_long'] = 1
        dataframe.loc[valid, 'enter_tag'] = ENTRY_TAG
        return dataframe
    SOURCE_ENTRY_STEM = 'complete_pattern_reversal_double_bottom_present_long_4h'
    FOCUSED_EXIT_CONTRACT = 'target_partial_invalidation_remainder'
    FOCUSED_SOURCE_PROFILE = {'side': 'long', 'min_level_distance': 0.001, 'targets': {'target_1': {'provider': 'entry-frozen double-bottom measured projection', 'long': {'level': 'pat_double_bottom_target_level', 'available': None}}}, 'invalidation': {'provider': 'entry-frozen double-bottom confirmation neckline', 'mode': 'level', 'long': {'level': 'pat_double_bottom_confirmation_level', 'available': None}}}
    FOCUSED_EXIT_PLANS = {'target_touch_p25_be_until_invalidation': {'contract': 'target_partial_invalidation_remainder', 'name': 'target_touch_p25_be_until_invalidation', 'action_sequence': ('source_invalidation_full', 'source_target_confirmation', 'partial_stage_1', 'breakeven_remainder', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'partial_fractions': (0.25,), 'target_confirmation': 'touch', 'target_band': 0.0025, 'invalidation_confirmations': 1, 'invalidation_band': 0.0025, 'breakeven_offset': 0.0}, 'target_close_p33_be_until_invalidation': {'contract': 'target_partial_invalidation_remainder', 'name': 'target_close_p33_be_until_invalidation', 'action_sequence': ('source_invalidation_full', 'source_target_confirmation', 'partial_stage_1', 'breakeven_remainder', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'partial_fractions': (0.33,), 'target_confirmation': 'close', 'target_band': 0.0025, 'invalidation_confirmations': 2, 'invalidation_band': 0.0025, 'breakeven_offset': 0.0}, 'target_reversal_p50_be_until_invalidation': {'contract': 'target_partial_invalidation_remainder', 'name': 'target_reversal_p50_be_until_invalidation', 'action_sequence': ('source_invalidation_full', 'source_target_confirmation', 'partial_stage_1', 'breakeven_remainder', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'partial_fractions': (0.5,), 'target_confirmation': 'reversal_1', 'target_band': 0.005, 'invalidation_confirmations': 2, 'invalidation_band': 0.005, 'breakeven_offset': 0.0}}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'high', 'low', 'open', 'pat_double_bottom_confirmation_level', 'pat_double_bottom_target_level')
    FOCUSED_STATE_VERSION = 2
    FOCUSED_STATE_KEY = 'sieve3_v2_focused:Sieve3V2TargetPartialInvalidationRemainderDoubleBottomProjectionNecklineFromCompletePatternReversalDoubleBottomPresentLong4H:target_partial_invalidation_remainder'
    LOCKED_BUY_PARAMS = {'use_sieve2_vp_guard': False, 'sieve2_vp_guard_mode': 'score_or_context', 'sieve2_vp_window': 96, 'sieve2_vp_bins': 36, 'sieve2_vp_score_min': 0.25, 'sieve2_vp_context_min': 0.28, 'use_sieve2_market_guard': False, 'sieve2_market_guard_mode': 'pressure_or_trend', 'sieve2_market_window': 24, 'sieve2_market_pressure_min': 0.07, 'sieve2_market_trend_min': 0.25, 'sieve2_rs_benchmark_pair': 'BTC/USDT:USDT', 'sieve2_rs_score_min': 0.45, 'use_volume_guard': True, 'volume_guard_window': 12, 'volume_ratio_min': 1.3, 'use_pressure_guard': True, 'pressure_window': 12, 'pressure_min': 0.35, 'use_accumulation_guard': False, 'use_body_direction_guard': False, 'use_close_direction_guard': False, 'score_min': 0.82, 'confirmation_buffer_pct': 0.005}
    ACTIVE_SELL_PARAMS = ('exit_plan', 'hard_stop_percent', 'max_hold_scale')
    position_adjustment_enable = True
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    minimal_roi = {'0': 100.0}
    stoploss = -0.99
    exit_plan = CategoricalParameter(('target_touch_p25_be_until_invalidation', 'target_close_p33_be_until_invalidation', 'target_reversal_p50_be_until_invalidation'), default='target_touch_p25_be_until_invalidation', space='sell', optimize=True, load=True)
    exit_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
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
        if valid:
            return level
        if str(level_column) != 'pat_double_bottom_target_level':
            return None
        origin = self._focused_float(row['pat_double_bottom_confirmation_level'])
        if origin is None:
            return None
        extension = abs(level - origin)
        if extension <= entry_rate * minimum:
            return None
        threshold = entry_rate * (1.0 - minimum if side == 'short' else 1.0 + minimum)
        while level >= threshold if side == 'short' else level <= threshold:
            level = level - extension if side == 'short' else level + extension
        return level

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
        stages = {f'stage_{index + 1}': {'status': 'ready', 'fraction': float(fraction), 'tag': f'focused_partial_stage_{index + 1}', 'requested_at': None, 'filled_at': None, 'order_id': None, 'target_stake': None, 'credited_stake': 0.0, 'request_stake': None, 'resolved_order_ids': []} for index, fraction in enumerate(plan.get('partial_fractions', ()))}
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
        filled_at = self._focused_utc(state.get('entry_filled_at'))
        if filled_at is None:
            raise ValueError('focused exit state has no entry fill timestamp')
        return frame.loc[self._focused_close_times(frame).gt(filled_at)]

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
        if state.get('stages'):
            self._focused_reconcile_partial_orders(trade, state, current_time)
        plan = self._focused_plan(state)
        frame = self._focused_analyzed_frame(pair, current_time)
        self._focused_require_columns(frame)
        post_entry = self._focused_post_entry(frame, state)
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
        if state['side'] == 'short' and stop_price <= current_rate or (state['side'] == 'long' and stop_price >= current_rate):
            return None
        if persisted is None or not math.isclose(stop_price, persisted, rel_tol=0.0, abs_tol=1e-12):
            state['stop_price'] = stop_price
            self._focused_save_state(trade, state)
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=state['side'] == 'short', leverage=float(getattr(trade, 'leverage', 1.0) or 1.0))

    def _focused_target_event(self, post_entry: DataFrame, state: dict[str, Any], plan: Mapping[str, Any], slot: str) -> bool:
        target = self._focused_float((state.get('levels') or {}).get(slot))
        target_states = state.setdefault('target_states', {})
        target_state = target_states.setdefault(slot, {'status': 'unavailable', 'touched_at': None})
        if target is None or post_entry.empty:
            target_state['status'] = 'unavailable'
            return False
        if target_state.get('status') in {'accepted', 'rejected', 'confirmed'}:
            return True
        band = float(plan.get('target_band') or 0.0)
        close_times = self._focused_close_times(post_entry)
        if state['side'] == 'short':
            touched = self._focused_number(post_entry, 'low').le(target * (1.0 + band))
        else:
            touched = self._focused_number(post_entry, 'high').ge(target * (1.0 - band))
        touched_at = self._focused_utc(target_state.get('touched_at'))
        if touched_at is None:
            positions = [index for index, value in enumerate(touched.fillna(False).tolist()) if value]
            if not positions:
                target_state['status'] = 'available'
                return False
            touched_at = self._focused_utc(close_times.iloc[positions[0]])
            target_state['touched_at'] = touched_at.isoformat() if touched_at is not None else None
            target_state['status'] = 'touched'
            state['phase'] = f'{slot}_zone'
        confirmation = str(plan.get('target_confirmation') or 'touch')
        if confirmation == 'touch':
            target_state['status'] = 'confirmed'
            return True
        after_touch = post_entry.loc[close_times.ge(touched_at) if confirmation == 'close' else close_times.gt(touched_at)]
        if after_touch.empty:
            return False
        close = self._focused_number(after_touch, 'close')
        open_ = self._focused_number(after_touch, 'open')
        if confirmation == 'close':
            accepted = close.le(target * (1.0 + band)) if state['side'] == 'short' else close.ge(target * (1.0 - band))
            if bool(accepted.fillna(False).any()):
                target_state['status'] = 'accepted'
                return True
            return False
        reversal = close.gt(open_) & close.gt(target * (1.0 - band)) if state['side'] == 'short' else close.lt(open_) & close.lt(target * (1.0 + band))
        if confirmation == 'reversal_1':
            confirmed = self._focused_last_n(reversal, 1)
        elif confirmation == 'reversal_2_of_3':
            confirmed = len(reversal) >= 3 and int(reversal.tail(3).fillna(False).sum()) >= 2
        else:
            raise ValueError(f'unsupported target confirmation: {confirmation}')
        if confirmed:
            target_state['status'] = 'rejected'
        return confirmed

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
        confirmed = self._focused_last_n(level_condition, confirmations)
        if confirmed:
            state['invalidation_seen'] = True
            state['phase'] = 'invalidated'
        return confirmed

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
        stop_price = self._focused_hard_stop_price(plan, state)
        if self._focused_stage_filled(state, 'stage_1'):
            candidate = self._focused_breakeven_price(state)
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
        self._focused_apply_partial_order(state, str(name), stage, order, current_time)
        if repr(state) != before:
            self._focused_save_state(trade, state)
        return None

    @staticmethod
    def _focused_order_key(order: Any) -> str:
        order_key = str(getattr(order, 'order_id', None) or getattr(order, 'id', None) or '')
        if not order_key:
            raise ValueError('focused partial order requires a persistent identity')
        return order_key

    @staticmethod
    def _focused_order_snapshot(order: Any) -> tuple[str, bool, float, bool]:
        status = str(getattr(order, 'status', None) or '').casefold()
        is_open = bool(getattr(order, 'ft_is_open', False)) and status not in NON_OPEN_EXCHANGE_STATES
        filled_amount = float(getattr(order, 'safe_filled', 0.0) or 0.0)
        requested_amount = float(getattr(order, 'safe_amount', 0.0) or 0.0)
        filled_stake = max(0.0, float(getattr(order, 'stake_amount_filled', 0.0) or 0.0))
        tolerance = max(1e-12, requested_amount * 1e-9)
        fully_filled = requested_amount > 0.0 and filled_amount >= requested_amount - tolerance
        return (status, is_open, filled_stake, fully_filled)

    def _focused_add_resolved_order(self, stage: dict[str, Any], order: Any) -> None:
        order_key = self._focused_order_key(order)
        resolved = stage.setdefault('resolved_order_ids', [])
        if order_key not in resolved:
            resolved.append(order_key)

    def _focused_reset_stage_for_retry(self, state: dict[str, Any], name: str, stage: dict[str, Any], orders: tuple[Any, ...]=()) -> None:
        for order in orders:
            self._focused_add_resolved_order(stage, order)
        active_order_id = str(stage.get('order_id') or '')
        resolved = stage.setdefault('resolved_order_ids', [])
        if active_order_id and active_order_id not in resolved:
            resolved.append(active_order_id)
        stage['status'] = 'ready'
        stage['requested_at'] = None
        stage['filled_at'] = None
        stage['order_id'] = None
        stage['request_stake'] = None
        state['phase'] = f'{name}_retry'

    def _focused_apply_partial_order(self, state: dict[str, Any], name: str, stage: dict[str, Any], order: Any, current_time: Any) -> None:
        order_key = self._focused_order_key(order)
        if order_key in {str(value) for value in stage.setdefault('resolved_order_ids', [])}:
            return
        _, is_open, filled_stake, fully_filled = self._focused_order_snapshot(order)
        if is_open:
            stage['status'] = 'requested'
            stage['order_id'] = order_key
            state['phase'] = f'{name}_pending'
            return
        target_stake = float(stage.get('target_stake') or 0.0)
        credited_stake = float(stage.get('credited_stake') or 0.0)
        if filled_stake > 0.0:
            credited_stake += filled_stake
            stage['credited_stake'] = min(target_stake, credited_stake) if target_stake > 0.0 else credited_stake
        self._focused_add_resolved_order(stage, order)
        tolerance = max(1e-9, target_stake * 1e-9)
        target_reached = target_stake > 0.0 and float(stage.get('credited_stake') or 0.0) >= target_stake - tolerance
        if fully_filled or target_reached:
            stage['status'] = 'filled'
            stage['filled_at'] = self._focused_utc(getattr(order, 'order_filled_utc', None) or current_time).isoformat()
            stage['order_id'] = order_key
            stage['request_stake'] = None
            state['phase'] = 'remainder' if name == 'stage_1' else 'runner'
            return
        self._focused_reset_stage_for_retry(state, name, stage)

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
            matching = [
                order
                for order in orders
                if getattr(order, 'ft_order_side', None) == exit_side
                and str(getattr(order, 'ft_order_tag', None) or '') == tag
                and self._focused_order_key(order) not in resolved
            ]
            terminal_orders = [order for order in matching if not self._focused_order_snapshot(order)[1]]
            for order in terminal_orders:
                self._focused_apply_partial_order(state, str(name), stage, order, current_time)
                if stage.get('status') == 'filled':
                    break
            if stage.get('status') == 'filled':
                continue
            open_order = next((order for order in reversed(matching) if self._focused_order_snapshot(order)[1]), None)
            if open_order is not None:
                self._focused_apply_partial_order(state, str(name), stage, open_order, current_time)
                continue
            if terminal_orders:
                continue
            requested_at = self._focused_utc(stage.get('requested_at'))
            if requested_at is not None and now is not None and requested_at == now:
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
            stage['request_stake'] = None
            state['phase'] = 'remainder' if stage_name == 'stage_1' else 'runner'
            self._focused_save_state(trade, state)
            return None
        request = min(current_stake, remaining_stake)
        if request <= 0.0 or request >= current_stake:
            return None
        stage['status'] = 'requested'
        stage['requested_at'] = self._focused_utc(current_time).isoformat()
        stage['order_id'] = None
        stage['request_stake'] = request
        state['phase'] = f'{stage_name}_pending'
        self._focused_save_state(trade, state)
        return (-request, str(stage['tag']))

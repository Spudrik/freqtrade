from __future__ import annotations
import math
import pandas as pd
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from pandas import DataFrame, Series
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
EXIT_FAMILY = 'time_progress_failure'
EXIT_THEORY = 'time_progress_failure'
EXIT_HYPOTHESIS = 'The pat_double_bottom_pattern_present with score >= 0.82 and close > pat_double_bottom_confirmation_level plus 0.5% thesis should make minimum favorable progress within its selected candle budget or close as failed follow-through.'
PRIMARY_TRIGGER = 'pat_double_bottom_pattern_present with score >= 0.82 and close > pat_double_bottom_confirmation_level plus 0.5%'
PRIMARY_GUARD = 'selected 12-candle volume >= 1.3x baseline and 12-candle bullish pressure >= 0.35; optional Sieve2 guards disabled'
TARGET_PROVIDER = 'exit_plan minimum favorable progress threshold by candle budget'
INVALIDATION_PROVIDER = 'insufficient favorable progress at the selected candle budget or fixed 3% hard stop'
ACTIVE_SELL_PARAMS = ('progress_window_step', 'min_progress_half_percent', 'hard_stop_percent', 'max_hold_scale')


SOURCE_RESULT_BATCH = 'historical_promotion_lineage_unknown'
RESEARCH_PATH = 'sieve3_exit_time_progress_failure'

class Sieve3V2TimeProgressFailureFromCompletePatternReversalDoubleBottomPresentLong4H(IStrategy):
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
    FOCUSED_EXIT_CONTRACT = 'time_progress_failure'
    FOCUSED_SOURCE_PROFILE = {'side': 'long', 'min_level_distance': 0.001}
    FOCUSED_EXIT_PLANS = {'progress_12_min_1pct': {'contract': 'time_progress_failure', 'name': 'progress_12_min_1pct', 'action_sequence': ('progress_check', 'failure_full', 'hard_stop', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 96, 'progress_candles': 12, 'min_progress_ratio': 0.01}, 'progress_24_min_1_5pct': {'contract': 'time_progress_failure', 'name': 'progress_24_min_1_5pct', 'action_sequence': ('progress_check', 'failure_full', 'hard_stop', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 168, 'progress_candles': 24, 'min_progress_ratio': 0.015}, 'progress_48_min_2pct': {'contract': 'time_progress_failure', 'name': 'progress_48_min_2pct', 'action_sequence': ('progress_check', 'failure_full', 'hard_stop', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'progress_candles': 48, 'min_progress_ratio': 0.02}}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'high', 'low', 'open')
    FOCUSED_STATE_VERSION = 1
    FOCUSED_STATE_KEY = 'sieve3_v2_focused:Sieve3V2TimeProgressFailureFromCompletePatternReversalDoubleBottomPresentLong4H:time_progress_failure'
    LOCKED_BUY_PARAMS = {'use_sieve2_vp_guard': False, 'sieve2_vp_guard_mode': 'score_or_context', 'sieve2_vp_window': 96, 'sieve2_vp_bins': 36, 'sieve2_vp_score_min': 0.25, 'sieve2_vp_context_min': 0.28, 'use_sieve2_market_guard': False, 'sieve2_market_guard_mode': 'pressure_or_trend', 'sieve2_market_window': 24, 'sieve2_market_pressure_min': 0.07, 'sieve2_market_trend_min': 0.25, 'sieve2_rs_benchmark_pair': 'BTC/USDT:USDT', 'sieve2_rs_score_min': 0.45, 'use_volume_guard': True, 'volume_guard_window': 12, 'volume_ratio_min': 1.3, 'use_pressure_guard': True, 'pressure_window': 12, 'pressure_min': 0.35, 'use_accumulation_guard': False, 'use_body_direction_guard': False, 'use_close_direction_guard': False, 'score_min': 0.82, 'confirmation_buffer_pct': 0.005}
    ACTIVE_SELL_PARAMS = ('progress_window_step', 'min_progress_half_percent', 'hard_stop_percent', 'max_hold_scale')
    position_adjustment_enable = False
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    minimal_roi = {'0': 100.0}
    stoploss = -0.99
    exit_plan = CategoricalParameter(('progress_12_min_1pct', 'progress_24_min_1_5pct', 'progress_48_min_2pct'), default='progress_12_min_1pct', space='sell', optimize=False, load=True)
    exit_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    progress_window_step = IntParameter(1, 4, default=1, space='sell', optimize=True, load=True)
    progress_window_step.batch_tags = ('family:exits', 'mode:sieve3_exit')
    min_progress_half_percent = IntParameter(1, 6, default=2, space='sell', optimize=True, load=True)
    min_progress_half_percent.batch_tags = ('family:exits', 'mode:sieve3_exit')
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
        progress_candles = 12 * int(self.progress_window_step.value)
        plan['progress_candles'] = progress_candles
        plan['min_progress_ratio'] = float(self.min_progress_half_percent.value) * 0.005
        plan['hard_stop_ratio'] = float(self.hard_stop_percent.value) * 0.01
        scaled_hold = int(round(float(plan.get('max_hold_candles', 96)) * float(self.max_hold_scale.value) / 2.0))
        plan['max_hold_candles'] = max(progress_candles + 1, scaled_hold)
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
        fill_time = (getattr(order, 'order_filled_utc', None) if order is not None else None) or getattr(trade, 'date_entry_fill_utc', None) or getattr(trade, 'open_date_utc', None) or current_time
        fill_timestamp = self._focused_utc(fill_time)
        snapshot_date = self._focused_utc(row['date'])
        return {'version': self.FOCUSED_STATE_VERSION, 'contract': self.FOCUSED_EXIT_CONTRACT, 'plan': selected_plan, 'side': side, 'phase': 'pre_target', 'entry_rate': float(entry_rate), 'entry_filled_at': fill_timestamp.isoformat() if fill_timestamp is not None else None, 'entry_snapshot_candle': snapshot_date.isoformat() if snapshot_date is not None else None, 'levels': levels, 'target_states': {slot: {'status': 'available' if value is not None else 'unavailable', 'touched_at': None} for slot, value in levels.items() if slot.startswith('target_')}, 'invalidation_seen': False, 'guard_state': 'unknown', 'trigger_state': 'unknown', 'initial_stake': self._focused_float(getattr(trade, 'stake_amount', None)), 'favorable_rate': float(entry_rate), 'stop_price': None, }

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

    def _focused_contract_events(self, post_entry: DataFrame, state: dict[str, Any], plan: Mapping[str, Any], current_profit: float) -> dict[str, Any]:
        _ = (self, post_entry, state, plan, current_profit)
        return {}

    def _focused_contract_decision(self, state: dict[str, Any], plan: Mapping[str, Any], events: Mapping[str, Any], current_profit: float) -> dict[str, Any]:
        _ = (state, current_profit)
        if int(events['age_candles']) >= int(plan['progress_candles']) and float(events['favorable_move']) < float(plan['min_progress_ratio']):
            return self._focused_decision('full', 'focused_progress_failure')
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

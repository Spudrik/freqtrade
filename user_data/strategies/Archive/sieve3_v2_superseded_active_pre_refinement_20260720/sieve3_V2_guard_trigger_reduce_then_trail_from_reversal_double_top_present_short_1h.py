from __future__ import annotations
import math
import pandas as pd
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from pandas import DataFrame, Series
from freqtrade.constants import NON_OPEN_EXCHANGE_STATES
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.strategy import CategoricalParameter, stoploss_from_absolute, IntParameter
import numpy as np
from freqtrade.strategy import IStrategy
from user_data.Indicators.pattern_reversal import add_pattern_reversal
ENTRY_MODE = 'entry_reversal_double_top_present_short_1h'
ENTRY_TAG = 'reversal_double_top_present_short_1h'
ENTRY_SOURCE_STAGE = "sieve2"
SIDE = 'short'
TIMEFRAME = '1h'


def _num(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    return pd.Series(frame[column], index=frame.index).astype('boolean').fillna(False).astype(bool)


def _cross_above(value: Series, level: Series) -> Series:
    return value.gt(level) & value.shift(1).le(level.shift(1))

def _cross_below(value: Series, level: Series) -> Series:
    return value.lt(level) & value.shift(1).ge(level.shift(1))
SIEVE_STAGE = "sieve3"
SOURCE_ENTRY_STAGE = 'sieve3'
SOURCE_ENTRY_CLASS = 'Sieve3ExitFixedTpSlFromReversalDoubleTopPresentShort1H'
SOURCE_STRATEGY = 'user_data/strategies/sieve3_exit_fixed_tp_sl_from_reversal_double_top_present_short_1h.py:Sieve3ExitFixedTpSlFromReversalDoubleTopPresentShort1H'
LINEAGE_STATUS = 'parked_legacy_executable_surface_defaults_frozen; promotion_lock_unverified'
EXIT_FAMILY = 'guard_trigger_reduce_then_trail'
EXIT_THEORY = 'guard_trigger_reduce_then_trail'
EXIT_HYPOTHESIS = 'Selected guard opposition reduces the pat_double_top_pattern_present with score >= 0.82 and close < pat_double_top_confirmation_level position, while exact trigger invalidation exits before trailing the remainder.'
PRIMARY_TRIGGER = 'pat_double_top_pattern_present with score >= 0.82 and close < pat_double_top_confirmation_level'
PRIMARY_GUARD = 'selected 48-candle volume >= 0.8x baseline; pressure and optional Sieve2 guards disabled'
TARGET_PROVIDER = 'no price target; source event reduces exposure then percent-trails the remainder'
INVALIDATION_PROVIDER = 's3v2_trigger_invalidated derived only from the selected source trigger'
ACTIVE_SELL_PARAMS = ('exit_plan', 'hard_stop_percent', 'max_hold_scale')

RESEARCH_PATH = "sieve3_exit_guard_trigger_reduce_then_trail"
SOURCE_RESULT_BATCH = "historical_promotion_lineage_unknown"


class Sieve3V2GuardTriggerReduceThenTrailFromReversalDoubleTopPresentShort1H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 160
    process_only_new_candles = True
    can_short = True
    max_entry_position_adjustment = 0
    use_volume_guard = True
    volume_guard_window = 48
    volume_ratio_min = 0.8
    # Legacy fixed-off entry parameters: use_pressure_guard=False, pressure_window=12,
    # pressure_min=0.05, and use_accumulation_guard=False; their pressure branch is unreachable.
    # Legacy fixed-off entry parameters: use_body_direction_guard=False and
    # use_close_direction_guard=False; their directional branches are unreachable.
    score_min = 0.82
    confirmation_buffer_pct = 0.0

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = (pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs)
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe = add_pattern_reversal(dataframe, timeframe=self.timeframe, min_double_quality=float(self.score_min), min_head_shoulders_quality=float(self.score_min))
        selected_guard = self._common_guards(dataframe)
        selected_state = _bool(dataframe, 'pat_double_top_pattern_present')
        score_ok = _num(dataframe, 'pat_double_top_indicator_score').ge(float(self.score_min))
        level = _num(dataframe, 'pat_double_top_confirmation_level')
        close = _num(dataframe, 'close')
        dataframe['s3v2_guard_aligned'] = selected_guard.fillna(False)
        dataframe['s3v2_guard_opposed'] = (~selected_guard).fillna(False)
        dataframe['s3v2_trigger_weakening'] = (~(selected_state & score_ok)).fillna(False)
        dataframe['s3v2_trigger_invalidated'] = close.gt(level).fillna(False)
        return dataframe

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype='bool')
        if self.use_volume_guard:
            volume = _num(dataframe, 'volume').clip(lower=0.0)
            window = int(self.volume_guard_window)
            baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
            guard &= volume.ge(baseline.mul(float(self.volume_ratio_min)))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = None
        level = _num(dataframe, 'pat_double_top_confirmation_level').mul(1.0 - float(self.confirmation_buffer_pct))
        condition = _bool(dataframe, 'pat_double_top_pattern_present') & _num(dataframe, 'pat_double_top_indicator_score').ge(float(self.score_min)) & _num(dataframe, 'close').lt(level)
        condition &= self._common_guards(dataframe)
        valid = condition.fillna(False) & dataframe['volume'].gt(0.0) & dataframe['close'].notna()
        dataframe.loc[valid, 'enter_short'] = 1
        dataframe.loc[valid, 'enter_tag'] = ENTRY_TAG
        return dataframe
    SOURCE_ENTRY_STEM = 'reversal_double_top_present_short_1h'
    FOCUSED_EXIT_CONTRACT = 'guard_trigger_reduce_then_trail'
    FOCUSED_SOURCE_PROFILE = {'side': 'short', 'min_level_distance': 0.001, 'guard': {'provider': 'selected entry guards for reversal_double_top_present_short_1h', 'short': {'aligned': 's3v2_guard_aligned', 'opposed': 's3v2_guard_opposed'}}, 'trigger': {'provider': 'pat_double_top_pattern_present with score >= 0.82 and close < pat_double_top_confirmation_level', 'short': {'intact': None, 'weakening': 's3v2_trigger_weakening', 'invalidated': 's3v2_trigger_invalidated'}}}
    FOCUSED_EXIT_PLANS = {'guard1_trigger1_reduce25_trail1': {'contract': 'guard_trigger_reduce_then_trail', 'name': 'guard1_trigger1_reduce25_trail1', 'action_sequence': ('trigger_invalidation_full', 'guard_or_trigger_weakening', 'partial_stage_1', 'percent_trail_remainder', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'partial_fractions': (0.25,), 'guard_confirmations': 1, 'trigger_confirmations': 1, 'trail_ratio': 0.01}, 'guard2_trigger1_reduce33_trail1_5': {'contract': 'guard_trigger_reduce_then_trail', 'name': 'guard2_trigger1_reduce33_trail1_5', 'action_sequence': ('trigger_invalidation_full', 'guard_or_trigger_weakening', 'partial_stage_1', 'percent_trail_remainder', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'partial_fractions': (0.33,), 'guard_confirmations': 2, 'trigger_confirmations': 1, 'trail_ratio': 0.015}, 'guard2_trigger2_reduce50_trail2': {'contract': 'guard_trigger_reduce_then_trail', 'name': 'guard2_trigger2_reduce50_trail2', 'action_sequence': ('trigger_invalidation_full', 'guard_or_trigger_weakening', 'partial_stage_1', 'percent_trail_remainder', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'partial_fractions': (0.5,), 'guard_confirmations': 2, 'trigger_confirmations': 2, 'trail_ratio': 0.02}}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'high', 'low', 'open', 's3v2_guard_aligned', 's3v2_guard_opposed', 's3v2_trigger_invalidated', 's3v2_trigger_weakening')
    FOCUSED_STATE_VERSION = 1
    FOCUSED_STATE_KEY = 'sieve3_v2_focused:Sieve3V2GuardTriggerReduceThenTrailFromReversalDoubleTopPresentShort1H:guard_trigger_reduce_then_trail'
    LOCKED_BUY_PARAMS = {'use_sieve2_vp_guard': False, 'sieve2_vp_guard_mode': 'score_or_context', 'sieve2_vp_window': 96, 'sieve2_vp_bins': 36, 'sieve2_vp_score_min': 0.25, 'sieve2_vp_context_min': 0.28, 'use_sieve2_market_guard': False, 'sieve2_market_guard_mode': 'pressure_or_trend', 'sieve2_market_window': 24, 'sieve2_market_pressure_min': 0.07, 'sieve2_market_trend_min': 0.25, 'sieve2_rs_benchmark_pair': 'BTC/USDT:USDT', 'sieve2_rs_score_min': 0.45, 'use_volume_guard': True, 'volume_guard_window': 48, 'volume_ratio_min': 0.8, 'use_pressure_guard': False, 'pressure_window': 12, 'pressure_min': 0.05, 'use_accumulation_guard': False, 'use_body_direction_guard': False, 'use_close_direction_guard': False, 'score_min': 0.82, 'confirmation_buffer_pct': 0.0}
    ACTIVE_SELL_PARAMS = ('exit_plan', 'hard_stop_percent', 'max_hold_scale')
    position_adjustment_enable = True
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    minimal_roi = {'0': 100.0}
    stoploss = -0.99
    exit_plan = CategoricalParameter(('guard1_trigger1_reduce25_trail1', 'guard2_trigger1_reduce33_trail1_5', 'guard2_trigger2_reduce50_trail2'), default='guard1_trigger1_reduce25_trail1', space='sell', optimize=True, load=True)
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

    @staticmethod
    def _focused_boolean(frame: DataFrame, column: str) -> Series:
        return frame[column].fillna(False).astype(bool)

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
        plan = self._focused_plan({'plan': selected_plan})
        fill_time = (getattr(order, 'order_filled_utc', None) if order is not None else None) or getattr(trade, 'date_entry_fill_utc', None) or getattr(trade, 'open_date_utc', None) or current_time
        fill_timestamp = self._focused_utc(fill_time)
        snapshot_date = self._focused_utc(row['date'])
        stages = {
            f'stage_{index + 1}': {
                'status': 'ready',
                'fraction': float(fraction),
                'tag': f'focused_partial_stage_{index + 1}',
                'target_stake': None,
                'requested_stake': None,
                'credited_stake': 0.0,
                'credited_order_stakes': {},
                'requested_at': None,
                'filled_at': None,
                'order_id': None,
                'resolved_order_ids': [],
            }
            for index, fraction in enumerate(plan.get('partial_fractions', ()))
        }
        return {'version': self.FOCUSED_STATE_VERSION, 'contract': self.FOCUSED_EXIT_CONTRACT, 'plan': selected_plan, 'side': side, 'phase': 'pre_target', 'entry_rate': float(entry_rate), 'entry_filled_at': fill_timestamp.isoformat() if fill_timestamp is not None else None, 'entry_snapshot_candle': snapshot_date.isoformat() if snapshot_date is not None else None, 'levels': levels, 'target_states': {slot: {'status': 'available' if value is not None else 'unavailable', 'touched_at': None} for slot, value in levels.items() if slot.startswith('target_')}, 'invalidation_seen': False, 'guard_state': 'unknown', 'trigger_state': 'unknown', 'stages': stages, 'initial_stake': self._focused_float(getattr(trade, 'stake_amount', None)), 'favorable_rate': float(entry_rate), 'stop_price': None, 'terminal_pending': False, 'terminal_tag': None}

    def _focused_ensure_state(self, pair: str, trade: Any, current_time: Any, order: Any | None=None) -> dict[str, Any] | None:
        state = self._focused_state(trade)
        if state is None:
            state = self._focused_new_state(pair, trade, current_time, order)
            if state is None:
                return None
            self._focused_save_state(trade, state)
        return state

    def _focused_post_entry(
        self,
        frame: DataFrame,
        state: Mapping[str, Any],
        current_time: Any,
    ) -> DataFrame:
        filled_at = self._focused_utc(state.get("entry_filled_at"))
        if filled_at is None:
            raise ValueError("focused exit state has no entry fill timestamp")
        candle_opens = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        close_times = self._focused_close_times(frame)
        now = self._focused_utc(current_time)
        if now is None:
            raise ValueError("focused exit runtime requires current_time")
        eligible = candle_opens.ge(filled_at) & close_times.le(now)
        return frame.loc[eligible].copy()


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
    def _focused_distinct_periods(frame: DataFrame, condition: Series, timeframe: str) -> Series:
        minutes = int(timeframe_to_minutes(timeframe))
        if minutes <= 0:
            raise ValueError('invalidation evidence timeframe must be at least one minute')
        dates = pd.to_datetime(frame['date'], utc=True, errors='raise')
        period_ids = dates.map(lambda value: timeframe_to_prev_date(timeframe, value.to_pydatetime()))
        return condition.fillna(False).groupby(period_ids, sort=True).any()

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


    @staticmethod
    def _focused_trail_price(state: Mapping[str, Any], ratio: float) -> float:
        favorable = float(state.get('favorable_rate') or state['entry_rate'])
        return favorable * (1.0 + ratio if state['side'] == 'short' else 1.0 - ratio)

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
        post_entry = self._focused_post_entry(frame, state, current_time)
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

    def _focused_guard_trigger_events(self, post_entry: DataFrame, state: dict[str, Any], plan: Mapping[str, Any]) -> dict[str, bool]:
        if post_entry.empty:
            return {'trigger_invalidated': False, 'trigger_weakening': False, 'guard_opposed': False}
        side = str(state['side'])
        guard = self.FOCUSED_SOURCE_PROFILE['guard'][side]
        trigger = self.FOCUSED_SOURCE_PROFILE['trigger'][side]
        guard_count = int(plan.get('guard_confirmations') or 1)
        trigger_count = int(plan.get('trigger_confirmations') or 1)
        opposed = self._focused_last_n(self._focused_boolean(post_entry, str(guard['opposed'])), guard_count)
        weakening = self._focused_last_n(self._focused_boolean(post_entry, str(trigger['weakening'])), trigger_count)
        invalidated = self._focused_last_n(self._focused_boolean(post_entry, str(trigger['invalidated'])), trigger_count)
        aligned_column = guard.get('aligned')
        aligned = bool(aligned_column and self._focused_last_n(self._focused_boolean(post_entry, str(aligned_column)), guard_count))
        state['guard_state'] = 'opposed' if opposed else 'aligned' if aligned else 'neutral'
        state['trigger_state'] = 'invalidated' if invalidated else 'weakening' if weakening else 'intact'
        return {'trigger_invalidated': invalidated, 'trigger_weakening': weakening, 'guard_opposed': opposed}

    def _focused_contract_events(self, post_entry: DataFrame, state: dict[str, Any], plan: Mapping[str, Any], current_profit: float) -> dict[str, Any]:
        _ = current_profit
        return self._focused_guard_trigger_events(post_entry, state, plan)

    def _focused_contract_decision(self, state: dict[str, Any], plan: Mapping[str, Any], events: Mapping[str, Any], current_profit: float) -> dict[str, Any]:
        _ = (plan, current_profit)
        if events.get('trigger_invalidated'):
            return self._focused_decision('full', 'focused_trigger_invalidation')
        stage = state['stages']['stage_1']
        if stage['status'] == 'ready' and (events.get('guard_opposed') or events.get('trigger_weakening')):
            return self._focused_decision('partial', str(stage['tag']), 'stage_1', float(stage['fraction']))
        return self._focused_decision()

    def _focused_stop_overlay(self, plan: Mapping[str, Any], state: Mapping[str, Any], current_profit: float) -> float:
        stop_price = self._focused_hard_stop_price(plan, state)
        if self._focused_stage_filled(state, 'stage_1'):
            candidate = self._focused_trail_price(state, float(plan['trail_ratio']))
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
        self._focused_apply_partial_order(trade, state, str(name), stage, order, current_time)
        if repr(state) != before:
            self._focused_save_state(trade, state)
        return None

    @staticmethod
    def _focused_order_snapshot(order: Any) -> tuple[str, bool, float]:
        status = str(getattr(order, 'status', None) or '').casefold()
        is_open = bool(getattr(order, 'ft_is_open', False)) and status not in NON_OPEN_EXCHANGE_STATES
        filled = float(getattr(order, 'safe_filled', 0.0) or 0.0)
        return (status, is_open, filled)

    @staticmethod
    def _focused_add_resolved_order(stage: dict[str, Any], order: Any) -> None:
        order_id = str(getattr(order, 'order_id', None) or '')
        if not order_id:
            return
        resolved = stage.setdefault('resolved_order_ids', [])
        if order_id not in resolved:
            resolved.append(order_id)

    def _focused_reset_stage_for_retry(self, state: dict[str, Any], name: str, stage: dict[str, Any], orders: tuple[Any, ...]=()) -> None:
        for order in orders:
            self._focused_add_resolved_order(stage, order)
        active_order_id = str(stage.get('order_id') or '')
        resolved = stage.setdefault('resolved_order_ids', [])
        if active_order_id and active_order_id not in resolved:
            resolved.append(active_order_id)
        stage['status'] = 'ready'
        stage['requested_at'] = None
        stage['requested_stake'] = None
        stage['filled_at'] = None
        stage['order_id'] = None
        state['phase'] = f'{name}_retry'

    def _focused_apply_partial_order(self, trade: Any, state: dict[str, Any], name: str, stage: dict[str, Any], order: Any, current_time: Any) -> None:
        _, is_open, filled = self._focused_order_snapshot(order)
        order_id = str(getattr(order, 'order_id', None) or '')
        credit_key = order_id or '|'.join(
            str(value or '')
            for value in (
                getattr(order, 'ft_order_tag', None),
                getattr(order, 'order_date_utc', None) or getattr(order, 'order_date', None),
                getattr(order, 'safe_amount', None),
                getattr(order, 'safe_price', None),
            )
        )
        fill_price = float(getattr(order, 'safe_price', 0.0) or 0.0)
        leverage = float(getattr(trade, 'leverage', 1.0) or 1.0)
        credited_for_order = filled * fill_price / leverage if fill_price > 0.0 else 0.0
        credits = stage.setdefault('credited_order_stakes', {})
        previous_credit = float(credits.get(credit_key) or 0.0)
        if credited_for_order > previous_credit:
            stage['credited_stake'] = float(stage.get('credited_stake') or 0.0) + credited_for_order - previous_credit
            credits[credit_key] = credited_for_order
        requested_amount = float(getattr(order, 'safe_amount', 0.0) or 0.0)
        fully_filled = requested_amount > 0.0 and filled >= requested_amount - max(1e-12, requested_amount * 1e-9)
        target_stake = self._focused_float(stage.get('target_stake'))
        credited_stake = float(stage.get('credited_stake') or 0.0)
        target_filled = target_stake is not None and credited_stake >= target_stake - 1e-8
        if is_open and not (fully_filled or target_filled):
            stage['status'] = 'requested'
            stage['order_id'] = order_id or None
            state['phase'] = f'{name}_pending'
            return
        if fully_filled or target_filled:
            stage['status'] = 'filled'
            stage['filled_at'] = self._focused_utc(getattr(order, 'order_filled_utc', None) or current_time).isoformat()
            stage['order_id'] = order_id or None
            state['phase'] = 'remainder' if name == 'stage_1' else 'runner'
            return
        self._focused_reset_stage_for_retry(state, name, stage, (order,))

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
            matching = [order for order in orders if getattr(order, 'ft_order_side', None) == exit_side and str(getattr(order, 'ft_order_tag', None) or '') == tag and (str(getattr(order, 'order_id', None) or '') not in resolved)]
            filled = next((order for order in reversed(matching) if not self._focused_order_snapshot(order)[1] and self._focused_order_snapshot(order)[2] > 0.0), None)
            if filled is not None:
                self._focused_apply_partial_order(trade, state, str(name), stage, filled, current_time)
                continue
            open_order = next((order for order in reversed(matching) if self._focused_order_snapshot(order)[1]), None)
            if open_order is not None:
                for order in matching:
                    if order is not open_order:
                        self._focused_add_resolved_order(stage, order)
                self._focused_apply_partial_order(trade, state, str(name), stage, open_order, current_time)
                continue
            if matching:
                self._focused_reset_stage_for_retry(state, str(name), stage, tuple(matching))
                continue
            requested_at = self._focused_utc(stage.get('requested_at'))
            if requested_at is not None and now is not None and (requested_at == now):
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
        credited_stake = float(stage.get('credited_stake') or 0.0)
        new_target = target_stake is None
        if new_target:
            request = min(current_stake, initial_stake * float(decision['fraction']))
        else:
            request = min(current_stake, max(0.0, target_stake - credited_stake))
        if min_stake is not None:
            minimum = float(min_stake)
            if request < minimum:
                return None
            if 0.0 < current_stake - request < minimum:
                request = current_stake - minimum
            if request < minimum:
                return None
        if request <= 0.0 or request >= current_stake:
            return None
        if new_target:
            stage['target_stake'] = request
        stage['status'] = 'requested'
        stage['requested_at'] = self._focused_utc(current_time).isoformat()
        stage['requested_stake'] = request
        stage['order_id'] = None
        state['phase'] = f'{stage_name}_pending'
        self._focused_save_state(trade, state)
        return (-request, str(stage['tag']))

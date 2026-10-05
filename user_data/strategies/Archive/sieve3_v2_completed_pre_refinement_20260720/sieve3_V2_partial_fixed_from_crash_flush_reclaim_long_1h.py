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
NOVEL_IDEA = True
UPDATE_HYPOTHESIS = 'Capitulation flushes that wick and reclaim should be cleaner than raw fades.'
import numpy as np
from freqtrade.strategy import IStrategy
SIDE = 'long'
ENTRY_MODE = 'flush_reclaim'
ENTRY_TAG = 'crash_flush_reclaim_long_1h'
ENTRY_SOURCE_STAGE = "sieve2"


def _safe_div(numer: Series, denom: Series) -> Series:
    denom = denom.replace(0.0, np.nan)
    return numer / denom

def _rsi(close: Series, length: int=14) -> Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / float(length), adjust=False, min_periods=max(2, length // 2)).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / float(length), adjust=False, min_periods=max(2, length // 2)).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)

def _atr(frame: DataFrame, length: int=14) -> Series:
    high = pd.to_numeric(frame['high'], errors='coerce')
    low = pd.to_numeric(frame['low'], errors='coerce')
    close = pd.to_numeric(frame['close'], errors='coerce')
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=max(2, length // 2)).mean()
SIEVE_STAGE = "sieve3"
SOURCE_ENTRY_STAGE = 'sieve2'
SOURCE_ENTRY_CLASS = 'Sieve3ExitFixedTpSlFromCrashFlushReclaimLong1H'
SOURCE_STRATEGY = 'user_data/strategies/sieve3_exit_fixed_tp_sl_from_crash_flush_reclaim_long_1h.py:Sieve3ExitFixedTpSlFromCrashFlushReclaimLong1H'
LINEAGE_STATUS = 'verified'
EXIT_FAMILY = 'partial_fixed'
EXIT_THEORY = 'partial_fixed'
EXIT_HYPOTHESIS = 'A named favorable threshold realizes one idempotent partial before a fixed remainder objective.'
PRIMARY_TRIGGER = 'drop_pct >= 0.13 with lower_wick_ratio >= 0.55 and close > compression_mid'
PRIMARY_GUARD = 'volume_ratio >= 1.02 or price_spike >= 0.042'
TARGET_PROVIDER = 'none'
INVALIDATION_PROVIDER = 'none'
ACTIVE_SELL_PARAMS = ('partial_trigger_half_percent', 'partial_fraction_mode', 'final_target_offset_percent', 'hard_stop_percent')

RESEARCH_PATH = "sieve3_exit_partial_fixed"
SOURCE_RESULT_BATCH = "historical_promotion_lineage_unknown"


class Sieve3V2PartialFixedFromCrashFlushReclaimLong1H(IStrategy):
    timeframe = '1h'
    can_short = False
    startup_candle_count = 320
    process_only_new_candles = True
    max_entry_position_adjustment = 0
    INTERFACE_VERSION = 3
    crash_lookback = 56
    compression_lookback = 5
    drop_pct_min = 0.13
    # Legacy inert entry parameter: compression_width_max=0.44 (fixed entry mode/gate).
    volume_ratio_min = 1.02
    price_spike_pct_min = 0.042
    # Legacy inert entry parameter: rsi_max=15 (fixed entry mode/gate).
    # Legacy inert entry parameter: reclaim_buffer_pct=0.009 (fixed entry mode/gate).
    # Legacy inert entry parameter: rsi_gate_mode='soft' (fixed entry mode/gate).

    def populate_indicators(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:
        close = pd.to_numeric(dataframe['close'], errors='coerce')
        high = pd.to_numeric(dataframe['high'], errors='coerce')
        low = pd.to_numeric(dataframe['low'], errors='coerce')
        open_ = pd.to_numeric(dataframe['open'], errors='coerce')
        volume = pd.to_numeric(dataframe['volume'], errors='coerce').fillna(0.0)
        look = int(self.crash_lookback)
        comp = int(self.compression_lookback)
        roll_high = close.rolling(look, min_periods=max(2, look // 2)).max()
        roll_low = close.rolling(look, min_periods=max(2, look // 2)).min()
        comp_high = high.rolling(comp, min_periods=max(2, comp // 2)).max()
        comp_low = low.rolling(comp, min_periods=max(2, comp // 2)).min()
        comp_mean = close.rolling(comp, min_periods=max(2, comp // 2)).mean()
        dataframe['rsi'] = _rsi(close, 14)
        dataframe['atr'] = _atr(dataframe, 14)
        dataframe['drop_pct'] = _safe_div(roll_high - close, roll_high).fillna(0.0)
        dataframe['compression_width'] = _safe_div(comp_high - comp_low, comp_mean).fillna(0.0)
        dataframe['compression_high'] = comp_high
        dataframe['compression_low'] = comp_low
        dataframe['compression_mid'] = (comp_high + comp_low) / 2.0
        dataframe['rolling_low'] = roll_low
        dataframe['volume_ratio'] = _safe_div(volume, volume.rolling(comp, min_periods=max(2, comp // 2)).mean()).fillna(0.0)
        dataframe['price_spike'] = _safe_div((close - close.shift(1)).abs(), close.shift(1).abs()).fillna(0.0)
        dataframe['body_ratio'] = _safe_div((close - open_).abs(), high - low).fillna(0.0)
        dataframe['lower_wick_ratio'] = _safe_div(np.minimum(open_, close) - low, high - low).fillna(0.0)
        dataframe['upper_wick_ratio'] = _safe_div(high - np.maximum(open_, close), high - low).fillna(0.0)
        dataframe['trend_mean'] = close.rolling(max(look * 2, comp * 2), min_periods=max(3, look)).mean()
        return dataframe

    

    def _entry_condition(self, dataframe: DataFrame) -> Series:
        drop_ok = pd.to_numeric(dataframe['drop_pct'], errors='coerce').ge(float(self.drop_pct_min))
        volume_ok = pd.to_numeric(dataframe['volume_ratio'], errors='coerce').ge(float(self.volume_ratio_min))
        spike_ok = volume_ok | pd.to_numeric(dataframe['price_spike'], errors='coerce').ge(float(self.price_spike_pct_min))
        close = pd.to_numeric(dataframe['close'], errors='coerce')
        compression_mid = pd.to_numeric(dataframe['compression_mid'], errors='coerce')
        return drop_ok & pd.to_numeric(dataframe['lower_wick_ratio'], errors='coerce').ge(0.55) & close.gt(compression_mid) & spike_ok

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:
        condition = self._entry_condition(dataframe)
        dataframe.loc[condition, ['enter_long', 'enter_tag']] = (1, ENTRY_TAG)
        return dataframe
    SOURCE_ENTRY_STEM = 'crash_flush_reclaim_long_1h'
    FOCUSED_EXIT_CONTRACT = 'partial_fixed'
    FOCUSED_SOURCE_PROFILE = {'side': 'long', 'min_level_distance': 0.0}
    FOCUSED_EXIT_PLANS = {'p25_at_1_5_then_fixed_4': {'contract': 'partial_fixed', 'name': 'p25_at_1_5_then_fixed_4', 'action_sequence': ('profit_partial_stage_1', 'fixed_remainder_full', 'hard_stop', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'profit_trigger': 0.015, 'partial_fractions': (0.25,), 'fixed_target_ratio': 0.04}, 'p33_at_2_then_fixed_5': {'contract': 'partial_fixed', 'name': 'p33_at_2_then_fixed_5', 'action_sequence': ('profit_partial_stage_1', 'fixed_remainder_full', 'hard_stop', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'profit_trigger': 0.02, 'partial_fractions': (0.33,), 'fixed_target_ratio': 0.05}, 'p50_at_3_then_fixed_6': {'contract': 'partial_fixed', 'name': 'p50_at_3_then_fixed_6', 'action_sequence': ('profit_partial_stage_1', 'fixed_remainder_full', 'hard_stop', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'profit_trigger': 0.03, 'partial_fractions': (0.5,), 'fixed_target_ratio': 0.06}}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'high', 'low', 'open')
    FOCUSED_STATE_VERSION = 1
    FOCUSED_STATE_KEY = 'sieve3_v2_focused:Sieve3V2PartialFixedFromCrashFlushReclaimLong1H:partial_fixed'
    LOCKED_BUY_PARAMS = {'crash_lookback': 56, 'compression_lookback': 5, 'drop_pct_min': 0.13, 'compression_width_max': 0.44, 'volume_ratio_min': 1.02, 'price_spike_pct_min': 0.042, 'rsi_max': 15, 'reclaim_buffer_pct': 0.009, 'rsi_gate_mode': 'soft'}
    ACTIVE_SELL_PARAMS = ('partial_trigger_half_percent', 'partial_fraction_mode', 'final_target_offset_percent', 'hard_stop_percent')
    position_adjustment_enable = True
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    minimal_roi = {'0': 100.0}
    stoploss = -0.99
    exit_plan = CategoricalParameter(('p25_at_1_5_then_fixed_4', 'p33_at_2_then_fixed_5', 'p50_at_3_then_fixed_6'), default='p25_at_1_5_then_fixed_4', space='sell', optimize=False, load=True)
    exit_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    partial_trigger_half_percent = IntParameter(3, 8, default=3, space='sell', optimize=True, load=True)
    partial_trigger_half_percent.batch_tags = ('family:exits', 'mode:sieve3_exit')
    partial_fraction_mode = CategoricalParameter(('p25', 'p33', 'p50'), default='p25', space='sell', optimize=True, load=True)
    partial_fraction_mode.batch_tags = ('family:exits', 'mode:sieve3_exit')
    final_target_offset_percent = IntParameter(2, 6, default=2, space='sell', optimize=True, load=True)
    final_target_offset_percent.batch_tags = ('family:exits', 'mode:sieve3_exit')
    hard_stop_percent = IntParameter(2, 6, default=3, space='sell', optimize=True, load=True)
    hard_stop_percent.batch_tags = ('family:exits', 'mode:sieve3_exit')

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
        trigger = float(self.partial_trigger_half_percent.value) * 0.005
        fraction = {'p25': 0.25, 'p33': 0.33, 'p50': 0.50}[str(self.partial_fraction_mode.value)]
        plan['profit_trigger'] = trigger
        plan['partial_fractions'] = (fraction,)
        plan['fixed_target_ratio'] = trigger + float(self.final_target_offset_percent.value) * 0.01
        plan['hard_stop_ratio'] = float(self.hard_stop_percent.value) * 0.01
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
        stages = {f'stage_{index + 1}': {'status': 'ready', 'fraction': float(fraction), 'tag': f'focused_partial_stage_{index + 1}', 'requested_at': None, 'requested_stake': None, 'target_stake': None, 'credited_stake': 0.0, 'order_credits': {}, 'filled_at': None, 'order_id': None, 'resolved_order_ids': []} for index, fraction in enumerate(plan.get('partial_fractions', ()))}
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

    def _focused_contract_events(self, post_entry: DataFrame, state: dict[str, Any], plan: Mapping[str, Any], current_profit: float) -> dict[str, Any]:
        _ = (self, post_entry, state, plan, current_profit)
        return {}

    def _focused_contract_decision(self, state: dict[str, Any], plan: Mapping[str, Any], events: Mapping[str, Any], current_profit: float) -> dict[str, Any]:
        _ = events
        stage = state['stages']['stage_1']
        if stage['status'] == 'ready' and current_profit >= float(plan['profit_trigger']):
            return self._focused_decision('partial', str(stage['tag']), 'stage_1', float(stage['fraction']))
        if stage['status'] == 'filled' and current_profit >= float(plan['fixed_target_ratio']):
            return self._focused_decision('full', 'focused_fixed_remainder')
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
    def _focused_order_snapshot(order: Any) -> tuple[str, bool, float, bool]:
        status = str(getattr(order, 'status', None) or '').casefold()
        is_open = bool(getattr(order, 'ft_is_open', False)) and status not in NON_OPEN_EXCHANGE_STATES
        filled_stake = float(getattr(order, 'stake_amount_filled', 0.0) or 0.0)
        filled_amount = float(getattr(order, 'safe_filled', 0.0) or 0.0)
        requested_amount = float(getattr(order, 'safe_amount', 0.0) or 0.0)
        fully_filled = requested_amount > 0.0 and filled_amount >= requested_amount - max(1e-12, requested_amount * 1e-9)
        return (status, is_open, filled_stake, fully_filled)

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

    @staticmethod
    def _focused_credit_partial_order(stage: dict[str, Any], order: Any, filled_stake: float) -> None:
        order_id = str(getattr(order, 'order_id', None) or '')
        if not order_id:
            raise ValueError('focused partial reconciliation requires an order id')
        credits = stage.setdefault('order_credits', {})
        previous = float(credits.get(order_id) or 0.0)
        credited = max(previous, float(filled_stake))
        if credited > previous:
            stage['credited_stake'] = float(stage.get('credited_stake') or 0.0) + credited - previous
            credits[order_id] = credited

    def _focused_apply_partial_order(self, state: dict[str, Any], name: str, stage: dict[str, Any], order: Any, current_time: Any) -> None:
        _, is_open, filled_stake, fully_filled = self._focused_order_snapshot(order)
        self._focused_credit_partial_order(stage, order, filled_stake)
        stage['order_id'] = str(getattr(order, 'order_id', None) or '') or None
        if is_open:
            stage['status'] = 'requested'
            state['phase'] = f'{name}_pending'
            return
        target = float(stage.get('target_stake') or 0.0)
        credited = float(stage.get('credited_stake') or 0.0)
        tolerance = max(1e-8, target * 0.005)
        if fully_filled or (target > 0.0 and credited >= target - tolerance):
            self._focused_add_resolved_order(stage, order)
            stage['status'] = 'filled'
            stage['requested_at'] = None
            stage['requested_stake'] = None
            stage['filled_at'] = self._focused_utc(getattr(order, 'order_filled_utc', None) or current_time).isoformat()
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
                self._focused_apply_partial_order(state, str(name), stage, filled, current_time)
                continue
            open_order = next((order for order in reversed(matching) if self._focused_order_snapshot(order)[1]), None)
            if open_order is not None:
                for order in matching:
                    if order is not open_order:
                        self._focused_add_resolved_order(stage, order)
                self._focused_apply_partial_order(state, str(name), stage, open_order, current_time)
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
        target = self._focused_float(stage.get('target_stake'))
        credited = float(stage.get('credited_stake') or 0.0)
        if target is None:
            request = min(current_stake, initial_stake * float(decision['fraction']))
        else:
            remaining = max(0.0, target - credited)
            if remaining <= max(1e-8, target * 0.005):
                stage['status'] = 'filled'
                stage['filled_at'] = self._focused_utc(current_time).isoformat()
                state['phase'] = 'remainder' if stage_name == 'stage_1' else 'runner'
                self._focused_save_state(trade, state)
                return None
            request = min(current_stake, remaining)
        if min_stake is not None and 0.0 < current_stake - request < float(min_stake):
            request = current_stake - float(min_stake)
        if request <= 0.0 or request >= current_stake:
            return None
        if target is None:
            stage['target_stake'] = request
        stage['status'] = 'requested'
        stage['requested_at'] = self._focused_utc(current_time).isoformat()
        stage['requested_stake'] = request
        stage['order_id'] = None
        state['phase'] = f'{stage_name}_pending'
        self._focused_save_state(trade, state)
        return (-request, str(stage['tag']))

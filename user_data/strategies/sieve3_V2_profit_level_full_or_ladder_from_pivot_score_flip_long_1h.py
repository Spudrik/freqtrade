from __future__ import annotations
import math
import pandas as pd
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from pandas import DataFrame, Series
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, stoploss_from_absolute, IntParameter
SECONDARY_GUARD = 'none'
SOURCE_ENTRY_SIGNATURE_SHA256 = '11267681243e7c71c9adc6c364999a84005143c23f4205dd3cc848c7d123518d'
LOCKED_BUY_SOURCE = 'verified effective lock from user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve/params/Sieve2PivotScoreFlipLong1h__auto_generic_1h_2020_q2_q3.json via user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve/results/20260612T211448_entry_sieve2_top_level_remaining120_cores8_20260612_failed15_retry01.jsonl:11'
SOURCE_SIEVE2_STRATEGY = 'sieve2_pivot_score_flip_long_1h'
SIEVE3_PROMOTION_NOTE = 'Promoted from Sieve2 survivor: 8589 trades, 56.43% winrate, -60.56% profit, 68.62% max DD at TP/SL 3/3; high-winrate overtrader kept for Sieve3 guard tightening'
NOVEL_IDEA = True
UPDATE_HYPOTHESIS = 'When pivot-low score overtakes pivot-high score, the structure may favor long rotation.'
import numpy as np
from freqtrade.strategy import IStrategy
from user_data.Indicators.pivot_foundation import build_clean_pivot_source

ENTRY_MODE = 'score_flip_long'



def _safe_div(numer: Series, denom: Series) -> Series:
    denom = denom.replace(0.0, np.nan)
    return numer / denom

def _atr(frame: DataFrame, length: int=14) -> Series:
    high = pd.to_numeric(frame['high'], errors='coerce')
    low = pd.to_numeric(frame['low'], errors='coerce')
    close = pd.to_numeric(frame['close'], errors='coerce')
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=max(2, length // 2)).mean()

SOURCE_ENTRY_STAGE = 'sieve2'
SOURCE_ENTRY_CLASS = 'Sieve3PivotScoreFlipLong1h'
SOURCE_STRATEGY = 'user_data/strategies/sieve3_pivot_score_flip_long_1h.py:Sieve3PivotScoreFlipLong1h'
LINEAGE_STATUS = 'verified'

EXIT_THEORY = 'profit_level_full_or_ladder'







EXIT_FAMILY = 'profit_level_full_or_ladder'
PRIMARY_TRIGGER = 'latest pivot-low score exceeds pivot-high score and meets the locked minimum'
PRIMARY_GUARD = 'close remains above the latest confirmed pivot low'
PRIMARY_TARGET = 'named fixed partial and remainder objectives'
PRIMARY_INVALIDATION = 'named fixed hard stop'
TARGET_PROVIDER = PRIMARY_TARGET
INVALIDATION_PROVIDER = PRIMARY_INVALIDATION
ENTRY_SOURCE_STAGE = 'sieve2'
SOURCE_RESULT_BATCH = 'historical_promotion_lineage_unknown'
RESEARCH_PATH = 'sieve3_exit_profit_level_full_or_ladder'
EXIT_HYPOTHESIS = 'Hyperopt chooses a full first profit exit or one partial followed by a complete second-target exit, with optional stop movement to entry.'
ACTIVE_SELL_PARAMS = ('exit_steps', 'target_1_half_percent_units', 'target_2_gap_half_percent_units', 'partial_1_five_percent_units', 'stop_after_target_1', 'hard_stop_percent')
SIEVE_STAGE = 'sieve3'
SIDE = 'long'
ENTRY_TAG = 'pivot_score_flip_long_1h'

class Sieve3V2ProfitLevelFullOrLadderFromPivotScoreFlipLong1h(IStrategy):
    timeframe = '1h'
    can_short = False
    startup_candle_count = 360
    process_only_new_candles = True
    INTERFACE_VERSION = 3
    pivot_strength = 2
    min_prominence_atr = 0.43
    min_pivot_spacing_bars = 4
    min_pivot_distance_atr = 0.72
    # Legacy inert entry parameter: reclaim_buffer_pct=0.011 (fixed entry mode/gate).
    score_min = 0.48
    # Legacy inert entry parameter: range_atr_mult=2.46 (fixed entry mode/gate).
    # Legacy inert entry parameter: freshness_max_bars=57 (fixed entry mode/gate).

    def populate_indicators(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:
        frame = dataframe.copy()
        frame['atr'] = _atr(frame, 14)
        frame['bar_index'] = pd.Series(np.arange(len(frame), dtype='float64'), index=frame.index)
        pivots = build_clean_pivot_source(body_high=pd.to_numeric(frame['high'], errors='coerce'), body_low=pd.to_numeric(frame['low'], errors='coerce'), atr=pd.to_numeric(frame['atr'], errors='coerce'), bar_index=frame['bar_index'], strength=int(self.pivot_strength), min_prominence_atr=float(self.min_prominence_atr), min_prominence_pct=0.0, min_pivot_spacing_bars=int(self.min_pivot_spacing_bars), min_pivot_distance_atr=float(self.min_pivot_distance_atr), min_pivot_distance_pct=0.0)
        frame = frame.join(pd.DataFrame(pivots))
        frame['pivot_high_last'] = frame['pivot_high'].ffill()
        frame['pivot_low_last'] = frame['pivot_low'].ffill()
        frame['pivot_high_prev'] = frame['pivot_high'].where(frame['pivot_high'].notna()).shift(1).ffill()
        frame['pivot_low_prev'] = frame['pivot_low'].where(frame['pivot_low'].notna()).shift(1).ffill()
        frame['pivot_mid'] = (frame['pivot_high_last'] + frame['pivot_low_last']) / 2.0
        frame['pivot_range'] = (frame['pivot_high_last'] - frame['pivot_low_last']).abs()
        frame['pivot_age_high'] = (frame['bar_index'] - frame['pivot_high_available_index'].ffill()).fillna(999.0)
        frame['pivot_age_low'] = (frame['bar_index'] - frame['pivot_low_available_index'].ffill()).fillna(999.0)
        frame['volume_ratio'] = _safe_div(pd.to_numeric(frame['volume'], errors='coerce').fillna(0.0), pd.to_numeric(frame['volume'], errors='coerce').rolling(20, min_periods=5).mean()).fillna(0.0)
        return frame

    def _entry_condition(self, frame: DataFrame) -> Series:
        close = pd.to_numeric(frame['close'], errors='coerce')
        pivot_low = pd.to_numeric(frame['pivot_low_last'], errors='coerce')
        score_min = float(self.score_min)
        pivot_high_score = pd.to_numeric(frame['pivot_high_score'], errors='coerce').ffill()
        pivot_low_score = pd.to_numeric(frame['pivot_low_score'], errors='coerce').ffill()
        return pivot_low_score.gt(pivot_high_score) & pivot_low_score.ge(score_min) & close.gt(pivot_low)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:
        condition = self._entry_condition(dataframe)
        dataframe.loc[condition, ['enter_long', 'enter_tag']] = (1, ENTRY_TAG)
        return dataframe
    SOURCE_ENTRY_STEM = 'pivot_score_flip_long_1h'
    FOCUSED_EXIT_CONTRACT = 'profit_level_full_or_ladder'
    FOCUSED_SOURCE_PROFILE = {'side': 'long', 'min_level_distance': 0.001}
    FOCUSED_EXIT_PLANS = {'profit_level_full_or_ladder': {'contract': 'profit_level_full_or_ladder', 'name': 'profit_level_full_or_ladder', 'action_sequence': ('profit_target_1', 'optional_partial', 'profit_target_2_full', 'optional_stop_to_entry', 'hard_stop', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336}}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'high', 'low', 'open')
    FOCUSED_STATE_VERSION = 1
    FOCUSED_STATE_KEY = 'sieve3_v2_focused:Sieve3V2ProfitLevelFullOrLadderFromPivotScoreFlipLong1h:profit_level_full_or_ladder'
    LOCKED_BUY_PARAMS = {'pivot_strength': 2, 'min_prominence_atr': 0.43, 'min_pivot_spacing_bars': 4, 'min_pivot_distance_atr': 0.72, 'reclaim_buffer_pct': 0.011, 'score_min': 0.48, 'range_atr_mult': 2.46, 'freshness_max_bars': 57}
    ACTIVE_SELL_PARAMS = ('exit_steps', 'target_1_half_percent_units', 'target_2_gap_half_percent_units', 'partial_1_five_percent_units', 'stop_after_target_1', 'hard_stop_percent')
    position_adjustment_enable = True
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    minimal_roi = {'0': 100.0}
    stoploss = -0.99
    exit_plan = 'profit_level_full_or_ladder'
    exit_steps = IntParameter(1, 2, default=2, space='sell', optimize=True, load=True)
    exit_steps.batch_tags = ('family:exits', 'mode:sieve3_exit')
    target_1_half_percent_units = IntParameter(2, 10, default=4, space='sell', optimize=True, load=True)
    target_1_half_percent_units.batch_tags = ('family:exits', 'mode:sieve3_exit')
    target_2_gap_half_percent_units = IntParameter(1, 6, default=4, space='sell', optimize=True, load=True)
    target_2_gap_half_percent_units.batch_tags = ('family:exits', 'mode:sieve3_exit')
    partial_1_five_percent_units = IntParameter(1, 8, default=5, space='sell', optimize=True, load=True)
    partial_1_five_percent_units.batch_tags = ('family:exits', 'mode:sieve3_exit')
    stop_after_target_1 = CategoricalParameter(('unchanged', 'entry'), default='entry', space='sell', optimize=True, load=True)
    stop_after_target_1.batch_tags = ('family:exits', 'mode:sieve3_exit')
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
        name = str((state or {}).get('plan') or self.exit_plan)
        if name not in self.FOCUSED_EXIT_PLANS:
            raise ValueError(f'unknown focused exit plan: {name}')
        plan = dict(self.FOCUSED_EXIT_PLANS[name])
        target_1 = float(self.target_1_half_percent_units.value) * 0.005
        plan['exit_steps'] = int(self.exit_steps.value)
        plan['target_1_ratio'] = target_1
        plan['target_2_ratio'] = target_1 + float(self.target_2_gap_half_percent_units.value) * 0.005
        plan['partial_fractions'] = (float(self.partial_1_five_percent_units.value) * 0.05,)
        plan['stop_after_target_1'] = str(self.stop_after_target_1.value)
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
        selected_plan = str(self.exit_plan)
        plan = self._focused_plan({'plan': selected_plan})
        fill_time = (getattr(order, 'order_filled_utc', None) if order is not None else None) or getattr(trade, 'date_entry_fill_utc', None) or getattr(trade, 'open_date_utc', None) or current_time
        fill_timestamp = self._focused_utc(fill_time)
        snapshot_date = self._focused_utc(row['date'])
        stages = {f'stage_{index + 1}': {'status': 'ready', 'fraction': float(fraction), 'tag': f'focused_partial_stage_{index + 1}', 'requested_at': None, 'target_stake': None, 'credited_stake': 0.0, 'filled_at': None} for index, fraction in enumerate(plan.get('partial_fractions', ()))}
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
        fill_start = int(frame['date'].searchsorted(filled_at, side='left'))
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
        favorable_move = max(0.0, 1.0 - favorable / entry_rate) if side == 'short' else max(0.0, favorable / entry_rate - 1.0)
        events = {'age_candles': age_candles, 'favorable_move': favorable_move}

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
        if exits >= 1 and str(plan['stop_after_target_1']) == 'entry':
            stop_price = entry_rate
        if (is_short and stop_price <= current_rate) or (not is_short and stop_price >= current_rate):
            return None
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=is_short, leverage=float(getattr(trade, 'leverage', 1.0) or 1.0))


    def _focused_contract_decision(self, state: dict[str, Any], plan: Mapping[str, Any], events: Mapping[str, Any], current_profit: float) -> dict[str, Any]:
        _ = events
        stage = state['stages']['stage_1']
        if current_profit >= float(plan['target_1_ratio']):
            if int(plan['exit_steps']) == 1:
                return self._focused_decision('full', 'focused_profit_target_1_full')
            if stage['status'] == 'ready':
                return self._focused_decision('partial', str(stage['tag']), 'stage_1', float(stage['fraction']))
        if stage['status'] == 'filled' and current_profit >= float(plan['target_2_ratio']):
            return self._focused_decision('full', 'focused_profit_target_2_full')
        return self._focused_decision()

    def _focused_breakeven_price(self, state: Mapping[str, Any]) -> float:
        entry_rate = self._focused_float(state.get('entry_rate'))
        if entry_rate is None or entry_rate <= 0.0:
            raise ValueError('focused exit state requires a positive entry rate')
        return entry_rate

    def _focused_stop_overlay(self, plan: Mapping[str, Any], state: Mapping[str, Any], current_profit: float) -> float:
        _ = current_profit
        stop_price = self._focused_hard_stop_price(plan, state)
        if str(plan['stop_after_target_1']) == 'entry' and self._focused_stage_filled(state, 'stage_1'):
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
        state['phase'] = f'{stage_name}_pending'
        self._focused_save_state(trade, state)
        return (-request, str(stage['tag']))

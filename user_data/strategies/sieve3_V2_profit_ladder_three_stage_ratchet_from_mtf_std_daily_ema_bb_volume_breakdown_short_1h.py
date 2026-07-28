from __future__ import annotations
import math
import pandas as pd
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from typing import Any
from pandas import DataFrame, Series
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import stoploss_from_absolute, IntParameter
NOVEL_IDEA = True
UPDATE_HYPOTHESIS = 'Daily downtrend plus 4h compression and 1h volume should favor breakdown continuation.'
import numpy as np
from freqtrade.strategy import IStrategy, merge_informative_pair
SIDE = 'short'
ENTRY_MODE = 'daily_ema_bb_volume_breakdown'
ENTRY_TAG = 'mtf_std_daily_ema_bb_volume_breakdown_short_1h'
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

def _ema(close: Series, length: int) -> Series:
    return close.ewm(span=length, adjust=False, min_periods=max(2, length // 2)).mean()

def _macd_hist(close: Series, fast: int=12, slow: int=26, signal: int=9) -> Series:
    macd = _ema(close, fast) - _ema(close, slow)
    sig = macd.ewm(span=signal, adjust=False, min_periods=max(2, signal // 2)).mean()
    return macd - sig

def _bb_width(close: Series, length: int, mult: float=2.0) -> Series:
    mid = close.rolling(length, min_periods=max(2, length // 2)).mean()
    std = close.rolling(length, min_periods=max(2, length // 2)).std()
    upper = mid + mult * std
    lower = mid - mult * std
    return _safe_div(upper - lower, mid)
SIEVE_STAGE = "sieve3"
SOURCE_ENTRY_STAGE = 'sieve2'
SOURCE_ENTRY_CLASS = 'Sieve3ExitBreakevenFromMtfStdDailyEmaBbVolumeBreakdownShort1H'
SOURCE_STRATEGY = 'user_data/strategies/sieve3_exit_breakeven_from_mtf_std_daily_ema_bb_volume_breakdown_short_1h.py:Sieve3ExitBreakevenFromMtfStdDailyEmaBbVolumeBreakdownShort1H'
LINEAGE_STATUS = 'verified'
EXIT_FAMILY = 'profit_ladder_three_stage_ratchet'
EXIT_THEORY = 'profit_ladder_three_stage_ratchet'
EXIT_HYPOTHESIS = 'Three ordered arbitrary profit targets take two independently sized partials and fully exit the remainder; the stop ratchets to entry and then target one after filled partials.'
PRIMARY_TRIGGER = '1h fast-EMA breakdown state: close < ema_fast_1h'
PRIMARY_GUARD = 'Locked active guards: bb_width_4h <= bb_width_max and volume_ratio_1h >= volume_ratio_min; daily trend is disabled'
TARGET_PROVIDER = 'fixed entry-relative profit objective selected by exit_plan'
INVALIDATION_PROVIDER = 'fixed entry-relative hard stop selected by exit_plan'
ACTIVE_SELL_PARAMS = ('target_1_percent', 'target_gap_half_percent_units', 'partial_1_five_percent_units', 'partial_2_five_percent_units', 'hard_stop_percent')

RESEARCH_PATH = "sieve3_exit_profit_ladder_three_stage_ratchet"
SOURCE_RESULT_BATCH = "historical_promotion_lineage_unknown"


class Sieve3V2ProfitLadderThreeStageRatchetFromMtfStdDailyEmaBbVolumeBreakdownShort1H(IStrategy):
    timeframe = '1h'
    can_short = True
    startup_candle_count = 400
    process_only_new_candles = True
    max_entry_position_adjustment = 0
    INTERFACE_VERSION = 3
    ema_fast_len = 42
    ema_slow_len = 121
    bb_len = 27
    bb_width_max = 0.08
    volume_ratio_min = 1.84
    # Legacy inert entry parameter: retest_buffer_pct=0.002 (fixed entry mode/gate).
    rsi_len = 25
    # Legacy inert entry parameter: rsi_long_min=49 (fixed entry mode/gate).
    # Legacy inert entry parameter: rsi_short_max=38 (fixed entry mode/gate).
    # Legacy fixed-off entry parameter: use_daily_trend=False (CategoricalParameter, buy space).
    # Legacy inert entry parameter: use_h4_compression=True (fixed entry mode/gate).
    # Legacy inert entry parameter: use_volume_filter=True (fixed entry mode/gate).
    # Legacy fixed-off entry parameter: use_retest=False (CategoricalParameter, buy space).
    # Legacy fixed-off entry parameter: use_momentum_filter=False (CategoricalParameter, buy space).

    def informative_pairs(self):
        dp = getattr(self, 'dp', None)
        if dp is None:
            return []
        pairs = self.dp.current_whitelist()
        return [(pair, '4h') for pair in pairs] + [(pair, '1d') for pair in pairs]

    def _features(self, frame: DataFrame, tf: str) -> DataFrame:
        close = pd.to_numeric(frame['close'], errors='coerce')
        high = pd.to_numeric(frame['high'], errors='coerce')
        low = pd.to_numeric(frame['low'], errors='coerce')
        volume = pd.to_numeric(frame['volume'], errors='coerce').fillna(0.0)
        ema_fast = int(self.ema_fast_len)
        ema_slow = int(self.ema_slow_len)
        bb_len = int(self.bb_len)
        rsi_len = int(self.rsi_len)
        frame[f'ema_fast_{tf}'] = _ema(close, ema_fast)
        frame[f'ema_slow_{tf}'] = _ema(close, ema_slow)
        frame[f'rsi_{tf}'] = _rsi(close, rsi_len)
        frame[f'bb_width_{tf}'] = _bb_width(close, bb_len, 2.0).fillna(0.0)
        frame[f'macd_hist_{tf}'] = _macd_hist(close).fillna(0.0)
        frame[f'volume_ratio_{tf}'] = _safe_div(volume, volume.rolling(20, min_periods=5).mean()).fillna(0.0)
        frame[f'atr_ratio_{tf}'] = _safe_div((high - low).rolling(14, min_periods=5).mean(), close).fillna(0.0)
        frame[f'prior_high_{tf}'] = high.shift(1).rolling(20, min_periods=5).max()
        frame[f'prior_low_{tf}'] = low.shift(1).rolling(20, min_periods=5).min()
        return frame

    def populate_indicators(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:
        frame = dataframe.copy()
        dp = getattr(self, 'dp', None)
        if dp is not None and metadata and metadata.get('pair'):
            for tf in ('4h', '1d'):
                informative = dp.get_pair_dataframe(pair=metadata['pair'], timeframe=tf)
                if informative is not None and (not informative.empty):
                    frame = merge_informative_pair(frame, self._features(informative.copy(), tf), self.timeframe, tf, ffill=True)
                    frame = frame.rename(columns=lambda col, tf=tf: col.replace(f'_{tf}_{tf}', f'_{tf}'))
        return self._features(frame, '1h')

    def _daily_trend_ok(self, frame: DataFrame, bullish: bool) -> Series:
        return pd.Series(True, index=frame.index)

    def _h4_compression_ok(self, frame: DataFrame) -> Series:
        return pd.to_numeric(frame['bb_width_4h'], errors='coerce').le(float(self.bb_width_max))

    def _volume_ok(self, frame: DataFrame) -> Series:
        return pd.to_numeric(frame['volume_ratio_1h'], errors='coerce').ge(float(self.volume_ratio_min))



    def _entry_condition(self, frame: DataFrame) -> Series:
        close = pd.to_numeric(frame['close'], errors='coerce')
        ema_fast_1h = pd.to_numeric(frame['ema_fast_1h'], errors='coerce')
        volume_ok = self._volume_ok(frame)
        h4_ok = self._h4_compression_ok(frame)
        short_trend_ok = self._daily_trend_ok(frame, False)
        return short_trend_ok & h4_ok & volume_ok & close.lt(ema_fast_1h)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:
        condition = self._entry_condition(dataframe)
        dataframe.loc[condition, ['enter_short', 'enter_tag']] = (1, ENTRY_TAG)
        return dataframe
    SOURCE_ENTRY_STEM = 'mtf_std_daily_ema_bb_volume_breakdown_short_1h'
    FOCUSED_EXIT_CONTRACT = 'profit_ladder_three_stage_ratchet'
    FOCUSED_SOURCE_PROFILE = {'side': 'short', 'min_level_distance': 0.001}
    FOCUSED_EXIT_PLANS = {'profit_ladder_three_stage_ratchet': {'contract': 'profit_ladder_three_stage_ratchet', 'name': 'profit_ladder_three_stage_ratchet', 'action_sequence': ('profit_target_1_partial', 'profit_target_2_partial', 'profit_target_3_full', 'stop_ratchet', 'hard_stop', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336}}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'high', 'low', 'open')
    FOCUSED_STATE_VERSION = 1
    FOCUSED_STATE_KEY = 'sieve3_v2_focused:Sieve3V2ProfitLadderThreeStageRatchetFromMtfStdDailyEmaBbVolumeBreakdownShort1H:profit_ladder_three_stage_ratchet'
    LOCKED_BUY_PARAMS = {'ema_fast_len': 42, 'ema_slow_len': 121, 'bb_len': 27, 'bb_width_max': 0.08, 'volume_ratio_min': 1.84, 'retest_buffer_pct': 0.002, 'rsi_len': 25, 'rsi_long_min': 49, 'rsi_short_max': 38, 'use_daily_trend': False, 'use_h4_compression': True, 'use_volume_filter': True, 'use_retest': False, 'use_momentum_filter': False}
    ACTIVE_SELL_PARAMS = ('target_1_percent', 'target_gap_half_percent_units', 'partial_1_five_percent_units', 'partial_2_five_percent_units', 'hard_stop_percent')
    position_adjustment_enable = True
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    minimal_roi = {'0': 100.0}
    stoploss = -0.99
    exit_plan = 'profit_ladder_three_stage_ratchet'
    target_1_percent = IntParameter(1, 5, default=2, space='sell', optimize=True, load=True)
    target_1_percent.batch_tags = ('family:exits', 'mode:sieve3_exit')
    target_gap_half_percent_units = IntParameter(1, 6, default=4, space='sell', optimize=True, load=True)
    target_gap_half_percent_units.batch_tags = ('family:exits', 'mode:sieve3_exit')
    partial_1_five_percent_units = IntParameter(1, 8, default=5, space='sell', optimize=True, load=True)
    partial_1_five_percent_units.batch_tags = ('family:exits', 'mode:sieve3_exit')
    partial_2_five_percent_units = IntParameter(1, 8, default=5, space='sell', optimize=True, load=True)
    partial_2_five_percent_units.batch_tags = ('family:exits', 'mode:sieve3_exit')
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
        target_1 = float(self.target_1_percent.value) * 0.01
        gap = float(self.target_gap_half_percent_units.value) * 0.005
        plan['target_1_ratio'] = target_1
        plan['target_2_ratio'] = target_1 + gap
        plan['target_3_ratio'] = target_1 + gap * 2.0
        plan['partial_fractions'] = (float(self.partial_1_five_percent_units.value) * 0.05, float(self.partial_2_five_percent_units.value) * 0.05)
        plan['hard_stop_ratio'] = float(self.hard_stop_percent.value) * 0.01
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
        _ = pair
        side = self._focused_side(trade)
        order_rate = self._focused_float(getattr(order, 'safe_price', None)) if order else None
        entry_rate = order_rate or self._focused_float(getattr(trade, 'open_rate', None))
        if entry_rate is None or entry_rate <= 0.0:
            raise ValueError('focused exit runtime requires a positive entry rate')
        freeze_time = (getattr(order, 'order_date_utc', None) if order is not None else None) or (getattr(order, 'order_date', None) if order is not None else None) or getattr(trade, 'open_date_utc', None) or getattr(trade, 'date_entry_fill_utc', None) or current_time
        freeze_timestamp = self._focused_utc(freeze_time)
        timeframe_minutes = max(1, int(timeframe_to_minutes(self.timeframe)))
        snapshot_date = (freeze_timestamp - pd.Timedelta(minutes=timeframe_minutes)).floor(f'{timeframe_minutes}min') if freeze_timestamp is not None else None
        levels: dict[str, float | None] = {'invalidation': None}
        selected_plan = str(self.exit_plan)
        plan = self._focused_plan({'plan': selected_plan})
        fill_time = (getattr(order, 'order_filled_utc', None) if order is not None else None) or getattr(trade, 'date_entry_fill_utc', None) or getattr(trade, 'open_date_utc', None) or current_time
        fill_timestamp = self._focused_utc(fill_time)
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
        if exits >= 1:
            stop_price = entry_rate
        if exits >= 2:
            target_1 = float(plan['target_1_ratio'])
            stop_price = entry_rate * (1.0 - target_1 if is_short else 1.0 + target_1)
        if (is_short and stop_price <= current_rate) or (not is_short and stop_price >= current_rate):
            return None
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=is_short, leverage=float(getattr(trade, 'leverage', 1.0) or 1.0))

    def _focused_contract_events(self, post_entry: DataFrame, state: dict[str, Any], plan: Mapping[str, Any], current_profit: float) -> dict[str, Any]:
        _ = (self, post_entry, state, plan, current_profit)
        return {}

    def _focused_contract_decision(self, state: dict[str, Any], plan: Mapping[str, Any], events: Mapping[str, Any], current_profit: float) -> dict[str, Any]:
        _ = events
        stage_1, stage_2 = state['stages']['stage_1'], state['stages']['stage_2']
        if stage_1['status'] == 'ready' and current_profit >= float(plan['target_1_ratio']):
            return self._focused_decision('partial', str(stage_1['tag']), 'stage_1', float(stage_1['fraction']))
        if stage_1['status'] == 'filled' and stage_2['status'] == 'ready' and current_profit >= float(plan['target_2_ratio']):
            return self._focused_decision('partial', str(stage_2['tag']), 'stage_2', float(stage_2['fraction']))
        if stage_2['status'] == 'filled' and current_profit >= float(plan['target_3_ratio']):
            return self._focused_decision('full', 'focused_profit_target_3_full')
        return self._focused_decision()

    def _focused_breakeven_price(self, state: Mapping[str, Any]) -> float:
        entry_rate = self._focused_float(state.get('entry_rate'))
        if entry_rate is None or entry_rate <= 0.0:
            raise ValueError('focused exit state requires a positive entry rate')
        return entry_rate

    def _focused_target_price(self, state: Mapping[str, Any], ratio: float) -> float:
        entry_rate = self._focused_breakeven_price(state)
        return entry_rate * (1.0 - ratio) if state['side'] == 'short' else entry_rate * (1.0 + ratio)

    def _focused_stop_overlay(self, plan: Mapping[str, Any], state: Mapping[str, Any], current_profit: float) -> float:
        _ = current_profit
        stop_price = self._focused_hard_stop_price(plan, state)
        if self._focused_stage_filled(state, 'stage_1'):
            stop_price = self._focused_tighter_stop(str(state['side']), stop_price, self._focused_breakeven_price(state))
        if self._focused_stage_filled(state, 'stage_2'):
            stop_price = self._focused_tighter_stop(str(state['side']), stop_price, self._focused_target_price(state, float(plan['target_1_ratio'])))
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
        if bool(getattr(trade, 'has_open_orders', False)):
            return None
        plan = self._focused_plan()
        completed = int(getattr(trade, 'nr_of_successful_exits', 0) or 0)
        if completed >= 2:
            return None
        next_target = float(plan['target_1_ratio'] if completed == 0 else plan['target_2_ratio'])
        if current_profit < next_target:
            return None
        _, state, _, decision = self._focused_context(str(getattr(trade, 'pair', '')), trade, current_time, current_rate, current_profit)
        if not state:
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

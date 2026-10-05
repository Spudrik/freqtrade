from __future__ import annotations
import math
import pandas as pd
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from pandas import DataFrame, Series
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.strategy import CategoricalParameter, stoploss_from_absolute, IntParameter
ENTRY_SOURCE_STAGE = 'sieve2'
RESEARCH_PATH = 'sieve3_exit_source_invalidation'
SOURCE_RESULT_BATCH = 'historical_promotion_lineage_unknown'
SOURCE_STRATEGY = 'C:/FreqTradeStuff/user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve/backtests/sieve2_loosen_geometry_triangle_squeeze_breakdown_short_1h_broader_trigger__pattern_geometry_1h_2023_24_compression/full_cycle_2020_2026/tp_3_sl_3/backtest-result-2026-06-12_01-03-51.zip!backtest-result-2026-06-12_01-03-51_Sieve2LoosenGeometryTriangleSqueezeBreakdownShort1hBroaderTrigger.py:Sieve2LoosenGeometryTriangleSqueezeBreakdownShort1hBroaderTrigger'
EXIT_FAMILY = 'source_invalidation'

SECONDARY_GUARD = 'none'
SOURCE_ENTRY_SIGNATURE_SHA256 = '3b70a8e1a7222e01d6e4aa07108af0ab41887ff4387c541a102eb16c128d3a55'
LOCKED_BUY_SOURCE = 'verified effective lock from C:/FreqTradeStuff/user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve/params/Sieve2LoosenGeometryTriangleSqueezeBreakdownShort1hBroaderTrigger__pattern_geometry_1h_2023_24_compression.json via C:/FreqTradeStuff/user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve/results/20260611T234019_entry_sieve_top_level_backlog6_cores14_20260611_01.jsonl:35'
import numpy as np
from freqtrade.strategy import IStrategy
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2

ENTRY_MODE = 'entry_loosen_geometry_triangle_squeeze_breakdown_short_1h_broader_trigger'
ENTRY_TAG = 'loosen_geometry_triangle_squeeze_breakdown_short_1h_broader_trigger'
UPDATE_HYPOTHESIS = 'High-win sparse entry may be too restrictive; broaden structural timing/quality thresholds while preserving the original entry idea.'
SIDE = 'short'
TIMEFRAME = '1h'


def _num(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        raise KeyError(f"required numeric column is missing: {column!r}")
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)

def _bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        raise KeyError(f"required boolean column is missing: {column!r}")
    return pd.Series(frame[column], index=frame.index).astype('boolean').fillna(False).astype(bool)
SIEVE_STAGE = 'sieve3'
SOURCE_ENTRY_STAGE = 'sieve2'
SOURCE_ENTRY_CLASS = 'Sieve2LoosenGeometryTriangleSqueezeBreakdownShort1hBroaderTrigger'
SOURCE_STRATEGY = 'C:/FreqTradeStuff/user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve/backtests/sieve2_loosen_geometry_triangle_squeeze_breakdown_short_1h_broader_trigger__pattern_geometry_1h_2023_24_compression/full_cycle_2020_2026/tp_3_sl_3/backtest-result-2026-06-12_01-03-51.zip!backtest-result-2026-06-12_01-03-51_Sieve2LoosenGeometryTriangleSqueezeBreakdownShort1hBroaderTrigger.py:Sieve2LoosenGeometryTriangleSqueezeBreakdownShort1hBroaderTrigger'
LINEAGE_STATUS = 'verified'
EXIT_FAMILY = 'source_invalidation'
EXIT_THEORY = 'source_invalidation_triangle_lower_breakdown_failure'
EXIT_HYPOTHESIS = 'A confirmed close through the entry-frozen triangle_lower_breakdown_boundary_reclaim boundary invalidates the pattern thesis and closes the position.'
PRIMARY_TRIGGER = 'close below the emitted triangle lower rail while the selected squeeze and non-positive direction remain active'
PRIMARY_GUARD = 'triangle presence, squeeze state, score, direction, locked common candle guards, and optional Sieve2 VP/market guards'
TARGET_PROVIDER = 'none'
INVALIDATION_PROVIDER = 'provider=triangle_lower_breakdown_boundary_reclaim;mode=level;short.level=pg2_triangle_lower'
ACTIVE_SELL_PARAMS = ('invalidation_confirmations', 'invalidation_band_quarter_percent', 'hard_stop_percent', 'max_hold_scale')


class Sieve3V2SourceInvalidationTriangleLowerBreakdownFailureFromLoosenGeometryTriangleSqueezeBreakdownShort1hBroaderTrigger(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 180
    process_only_new_candles = True
    can_short = True
    max_entry_position_adjustment = 0
    
    
    
    
    
    
    
    
    
    
    
    
    
    use_volume_guard = True
    volume_guard_window = 24
    volume_ratio_min = 1.0
    use_pressure_guard = True
    pressure_window = 24
    pressure_min = 0.2
    use_accumulation_guard = False
    use_body_direction_guard = False
    use_close_direction_guard = False
    min_pattern_bars = 12
    max_pattern_bars = 72
    compression_max_width_atr = 3.0
    squeeze_active_width_atr = 2.8
    min_line_score = 0.5
    min_containment = 0.88

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = (pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs)
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe = add_pattern_geometry_v2(dataframe, timeframe=self.timeframe, output_slots=1, include_triangle_patterns=True, include_wedge_patterns=False, include_compression_patterns=False, include_rectangle_patterns=False, include_ascending_channel_patterns=False, include_descending_channel_patterns=False, min_pattern_bars=int(self.min_pattern_bars), max_pattern_bars=int(self.max_pattern_bars), compression_max_width_atr=float(self.compression_max_width_atr), squeeze_active_width_atr=float(self.squeeze_active_width_atr), min_line_score=float(self.min_line_score), min_containment=float(self.min_containment), output_prefix='pg2')
        
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
        condition = _bool(dataframe, 'pg2_triangle_pattern_present') & _bool(dataframe, 'pg2_triangle_squeeze_active') & _num(dataframe, 'pg2_triangle_indicator_score').ge(float(self.min_line_score)) & _num(dataframe, 'pg2_triangle_direction').le(0) & _num(dataframe, 'close').lt(_num(dataframe, 'pg2_triangle_lower'))
        condition &= self._common_guards(dataframe)
        
        valid = condition.fillna(False) & dataframe['volume'].gt(0.0) & dataframe['close'].notna()
        dataframe.loc[valid, 'enter_short'] = 1
        dataframe.loc[valid, 'enter_tag'] = ENTRY_TAG
        return dataframe
    FOCUSED_EXIT_CONTRACT = 'source_invalidation'
    FOCUSED_SOURCE_PROFILE = {'side': 'short', 'min_level_distance': 0.001, 'invalidation': {'provider': 'triangle_lower_breakdown_boundary_reclaim', 'mode': 'level', 'short': {'level': 'pg2_triangle_lower', 'available': None}}}
    FOCUSED_EXIT_PLANS = {'source_invalidation_close1': {'contract': 'source_invalidation', 'name': 'source_invalidation_close1', 'action_sequence': ('source_invalidation_confirmation', 'invalidation_full', 'hard_stop', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'invalidation_confirmations': 1, 'invalidation_band': 0.0025}, 'source_invalidation_close2': {'contract': 'source_invalidation', 'name': 'source_invalidation_close2', 'action_sequence': ('source_invalidation_confirmation', 'invalidation_full', 'hard_stop', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'invalidation_confirmations': 2, 'invalidation_band': 0.0025}, 'source_invalidation_close2_band_0_5': {'contract': 'source_invalidation', 'name': 'source_invalidation_close2_band_0_5', 'action_sequence': ('source_invalidation_confirmation', 'invalidation_full', 'hard_stop', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'invalidation_confirmations': 2, 'invalidation_band': 0.005}}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'high', 'low', 'open', 'pg2_triangle_lower')
    FOCUSED_STATE_VERSION = 1
    FOCUSED_STATE_KEY = 'sieve3_v2_focused:Sieve3V2SourceInvalidationTriangleLowerBreakdownFailureFromLoosenGeometryTriangleSqueezeBreakdownShort1hBroaderTrigger:source_invalidation'
    LOCKED_BUY_PARAMS = {'use_sieve2_vp_guard': False, 'sieve2_vp_guard_mode': 'score_or_context', 'sieve2_vp_window': 96, 'sieve2_vp_bins': 36, 'sieve2_vp_score_min': 0.25, 'sieve2_vp_context_min': 0.28, 'use_sieve2_market_guard': False, 'sieve2_market_guard_mode': 'pressure_or_trend', 'sieve2_market_window': 24, 'sieve2_market_pressure_min': 0.07, 'sieve2_market_trend_min': 0.25, 'sieve2_rs_benchmark_pair': 'BTC/USDT:USDT', 'sieve2_rs_score_min': 0.45, 'use_volume_guard': True, 'volume_guard_window': 24, 'volume_ratio_min': 1.0, 'use_pressure_guard': True, 'pressure_window': 24, 'pressure_min': 0.2, 'use_accumulation_guard': False, 'use_body_direction_guard': False, 'use_close_direction_guard': False, 'min_pattern_bars': 12, 'max_pattern_bars': 72, 'compression_max_width_atr': 3.0, 'squeeze_active_width_atr': 2.8, 'min_line_score': 0.5, 'min_containment': 0.88}
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
            if role == 'invalidation':
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
        if level is None:
            level_condition = pd.Series(False, index=post_entry.index, dtype='bool')
        elif side == 'short':
            level_condition = close.gt(level * (1.0 + band))
        else:
            level_condition = close.lt(level * (1.0 - band))
        evidence_column = binding.get('evidence')
        evidence_condition = self._focused_boolean(post_entry, str(evidence_column)) if evidence_column else pd.Series(False, index=post_entry.index, dtype='bool')
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
SOURCE_STRATEGY = 'C:/FreqTradeStuff/user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve/backtests/sieve2_loosen_geometry_triangle_squeeze_breakdown_short_1h_broader_trigger__pattern_geometry_1h_2023_24_compression/full_cycle_2020_2026/tp_3_sl_3/backtest-result-2026-06-12_01-03-51.zip!backtest-result-2026-06-12_01-03-51_Sieve2LoosenGeometryTriangleSqueezeBreakdownShort1hBroaderTrigger.py:Sieve2LoosenGeometryTriangleSqueezeBreakdownShort1hBroaderTrigger'
EXIT_FAMILY = 'source_invalidation'

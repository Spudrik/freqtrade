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
NOVEL_IDEA = True
UPDATE_HYPOTHESIS = 'Entry-only novel Sieve3 probe: 1d+4h aligned break as higher-timeframe context, with 1h bos execution.'
import numpy as np
from freqtrade.strategy import IStrategy, merge_informative_pair
SIDE = 'long'

CONTEXT_MODE = 'dual_break'
EXECUTION_MODE = 'bos'
SIEVE_STAGE = 'sieve3'
SOURCE_RESULT_BATCH = 'unknown_not_recorded'
RESEARCH_PATH = 'sieve3_exit_source_invalidation'
SOURCE_ENTRY_CLASS = 'Sieve3ExitNovelMtfD1H4DualBreak1hBosLong'
SOURCE_STRATEGY = 'user_data/strategies/sieve3_exit_novel_mtf_d1_h4_dual_break_1h_bos_long.py:Sieve3ExitNovelMtfD1H4DualBreak1hBosLong'
LINEAGE_STATUS = 'verified_selected_defaults_from_accepted_20260606T100454; historical_candidate_path_stale'
EXIT_FAMILY = 'source_invalidation'
EXIT_THEORY = 'source_invalidation_d1_break_failure'
EXIT_HYPOTHESIS = 'Loss of the entry-frozen D1 prior-high level invalidates one half of the dual-break context.'
PRIMARY_TRIGGER = '1h BOS long: close > prior_high_1h and hl_1h'
PRIMARY_GUARD = 'D1/H4 prior-high dual break plus selected D1 volume_ratio_1d >= 1.46; H4 candle confirmation disabled'
TARGET_PROVIDER = 'none'
PRIMARY_TARGET = TARGET_PROVIDER
INVALIDATION_PROVIDER = 'provider=frozen D1 prior-high break level;mode=level;long.level=prior_high_1d_1d'
PRIMARY_INVALIDATION = INVALIDATION_PROVIDER
ACTIVE_SELL_PARAMS = ('invalidation_confirmations', 'invalidation_band_quarter_percent', 'hard_stop_percent', 'max_hold_scale')

class Sieve3V2SourceInvalidationD1BreakFailureFromNovelMtfD1H4DualBreak1HBosLong(IStrategy):
    timeframe = '1h'
    can_short = False
    startup_candle_count = 260
    process_only_new_candles = True
    INTERFACE_VERSION = 3
    context_lookback = 38
    h4_lookback = 14
    exec_lookback = 24
    body_ratio_min = 0.38
    range_ratio_min = 2.1
    volume_ratio_min = 1.46
    # Legacy inert entry parameter: retest_tolerance=0.021 (fixed entry mode/gate).
    # Legacy inert entry parameter: close_follow_min=0.11 (fixed entry mode/gate).
    # Legacy fixed-off entry parameter: require_h4_confirm=False (CategoricalParameter, buy space).
    # Legacy inert entry parameter: require_volume_confirm=True (fixed entry mode/gate).

    def informative_pairs(self):
        dp = getattr(self, 'dp', None)
        if dp is None:
            return []
        pairs = self.dp.current_whitelist()
        return [(pair, '4h') for pair in pairs] + [(pair, '1d') for pair in pairs]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
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
        vol_ok = self._num(dataframe, 'volume_ratio_1d').ge(float(self.volume_ratio_min)) | (not self._bool_param(True))
        h4_long_ok = bull4 | (not self._bool_param(False))
        h4_short_ok = bear4 | (not self._bool_param(False))
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
        ph = self._num(dataframe, 'prior_high_1h')
        exe = close.gt(ph) & self._bool(dataframe, 'hl_1h')
        return exe.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        condition = self._context(dataframe) & self._execution(dataframe)
        dataframe.loc[condition, ['enter_long', 'enter_tag']] = (1, self.ENTRY_TAG)
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
    def _num(frame: DataFrame, column: str, default: float | None = None) -> Series:
        resolved = Sieve3V2SourceInvalidationD1BreakFailureFromNovelMtfD1H4DualBreak1HBosLong._column_alias(frame, column)
        if resolved is None:
            if column in frame.columns:
                resolved = column
            else:
                for suffix in ('_1d', '_4h'):
                    if column.endswith(suffix) and f'{column}{suffix}' in frame.columns:
                        resolved = f'{column}{suffix}'
                        break
        if resolved is None:
            if default is None:
                raise KeyError(f"required numeric column is missing: {column!r}")
            return pd.Series(default, index=frame.index, dtype='float64')
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
            raise KeyError(f"required boolean column is missing: {column!r}")
        return pd.Series(frame[resolved], index=frame.index).astype('boolean').fillna(False).astype(bool)

    @staticmethod
    def _bool_param(param: Any) -> bool:
        value = getattr(param, 'value', param)
        if isinstance(value, str):
            return value.lower() in {'1', 'true', 'yes', 'on'}
        return bool(value)
    SOURCE_ENTRY_STEM = 'novel_mtf_d1_h4_dual_break_1h_bos_long'
    ENTRY_TAG = 'novel_mtf_d1_h4_dual_break_1h_bos_long'
    FOCUSED_EXIT_CONTRACT = 'source_invalidation'
    FOCUSED_SOURCE_PROFILE = {'side': 'long', 'min_level_distance': 0.001, 'invalidation': {'provider': 'frozen D1 prior-high break level', 'mode': 'level', 'long': {'level': 'prior_high_1d_1d', 'available': None}}}
    FOCUSED_EXIT_PLANS = {'source_invalidation_close1': {'contract': 'source_invalidation', 'name': 'source_invalidation_close1', 'action_sequence': ('source_invalidation_confirmation', 'invalidation_full', 'hard_stop', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'invalidation_confirmations': 1, 'invalidation_band': 0.0025}, 'source_invalidation_close2': {'contract': 'source_invalidation', 'name': 'source_invalidation_close2', 'action_sequence': ('source_invalidation_confirmation', 'invalidation_full', 'hard_stop', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'invalidation_confirmations': 2, 'invalidation_band': 0.0025}, 'source_invalidation_close2_band_0_5': {'contract': 'source_invalidation', 'name': 'source_invalidation_close2_band_0_5', 'action_sequence': ('source_invalidation_confirmation', 'invalidation_full', 'hard_stop', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'invalidation_confirmations': 2, 'invalidation_band': 0.005}}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'high', 'low', 'open', 'prior_high_1d_1d')
    FOCUSED_STATE_VERSION = 1
    FOCUSED_STATE_KEY = 'sieve3_v2_focused:Sieve3V2SourceInvalidationD1BreakFailureFromNovelMtfD1H4DualBreak1HBosLong:source_invalidation'
    LOCKED_BUY_PARAMS = {'context_lookback': 38, 'h4_lookback': 14, 'exec_lookback': 24, 'body_ratio_min': 0.38, 'range_ratio_min': 2.1, 'volume_ratio_min': 1.46, 'retest_tolerance': 0.021, 'close_follow_min': 0.11, 'require_h4_confirm': False, 'require_volume_confirm': True}
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
        values = frame[column]
        if bool(values.isna().any()):
            raise ValueError(f'required boolean evidence column contains NaN: {column}')
        try:
            return values.astype('boolean').astype(bool)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'required boolean evidence column is malformed: {column}') from exc

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
        invalidation_provider = self.FOCUSED_SOURCE_PROFILE.get('invalidation')
        if not isinstance(invalidation_provider, Mapping):
            raise ValueError('source invalidation profile must be a mapping')
        invalidation_mode = str(invalidation_provider.get('mode') or '')
        if invalidation_mode == 'evidence':
            levels['invalidation'] = None
        elif invalidation_mode in {'level', 'level_and_evidence', 'level_or_evidence'}:
            invalidation_binding = self._focused_side_binding('invalidation', side)
            levels['invalidation'] = self._focused_frozen_level(row, invalidation_binding, side, entry_rate, 'invalidation')
        else:
            raise ValueError(f'unsupported invalidation mode: {invalidation_mode}')
        invalidation_level = levels.get('invalidation')
        if invalidation_mode != 'evidence':
            if invalidation_level is None:
                raise ValueError('focused source invalidation requires a finite positive entry-frozen level')
            usable = invalidation_level > entry_rate if side == 'short' else invalidation_level < entry_rate
            if not usable:
                raise ValueError('focused source invalidation level must be strictly protective-side of entry')
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
        return {'version': self.FOCUSED_STATE_VERSION, 'contract': self.FOCUSED_EXIT_CONTRACT, 'plan': selected_plan, 'side': side, 'phase': 'pre_target', 'entry_rate': float(entry_rate), 'entry_filled_at': fill_timestamp.isoformat() if fill_timestamp is not None else None, 'entry_snapshot_candle': snapshot_date.isoformat() if snapshot_date is not None else None, 'levels': levels, 'target_states': {slot: {'status': 'available' if value is not None else 'unavailable', 'touched_at': None} for slot, value in levels.items() if slot.startswith('target_')}, 'invalidation_seen': False, 'guard_state': 'unknown', 'trigger_state': 'unknown', 'favorable_rate': float(entry_rate), 'stop_price': None}

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
        candle_opens = pd.to_datetime(frame['date'], utc=True, errors='raise')
        return frame.loc[candle_opens.ge(filled_at)]

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

    def _focused_invalidation_event(self, post_entry: DataFrame, state: dict[str, Any], plan: Mapping[str, Any]) -> bool:
        if state.get('invalidation_seen'):
            return True
        side = str(state['side'])
        provider = self.FOCUSED_SOURCE_PROFILE['invalidation']
        if not isinstance(provider, Mapping):
            raise ValueError('source invalidation profile must be a mapping')
        binding = provider[side]
        mode = str(provider.get('mode') or '')
        requires_level = mode in {'level', 'level_and_evidence', 'level_or_evidence'}
        if not requires_level and mode != 'evidence':
            raise ValueError(f'unsupported invalidation mode: {mode}')
        level = self._focused_float((state.get('levels') or {}).get('invalidation'))
        if requires_level and (level is None or level <= 0.0):
            level_column = binding.get('level')
            raise ValueError(f'active trade state requires a positive finite invalidation level: {level_column}')
        if post_entry.empty:
            return False
        band = float(plan.get('invalidation_band') or 0.0)
        close = self._focused_number(post_entry, 'close')
        if level is None:
            level_condition = pd.Series(False, index=post_entry.index, dtype='bool')
        elif side == 'short':
            level_condition = close.gt(level * (1.0 + band))
        else:
            level_condition = close.lt(level * (1.0 - band))
        evidence_column = binding.get('evidence')
        if evidence_column:
            available_column = binding.get('available')
            if available_column:
                availability = self._focused_boolean(post_entry, str(available_column))
                if not bool(availability.all()):
                    raise ValueError(f'required invalidation evidence is unavailable: {available_column}')
            evidence_condition = self._focused_boolean(post_entry, str(evidence_column))
        else:
            evidence_condition = pd.Series(False, index=post_entry.index, dtype='bool')
        confirmations = int(plan.get('invalidation_confirmations') or 1)
        if mode == 'level':
            confirmed = self._focused_last_n(level_condition, confirmations)
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
            confirmed = self._focused_last_n(level_condition, confirmations) or event_seen if mode == 'level_or_evidence' else event_seen
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

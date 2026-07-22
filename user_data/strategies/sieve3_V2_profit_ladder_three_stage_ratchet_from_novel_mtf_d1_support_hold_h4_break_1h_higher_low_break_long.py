from __future__ import annotations
import math
import pandas as pd
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from pandas import DataFrame, Series
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import stoploss_from_absolute, IntParameter
ENTRY_SOURCE_STAGE = "sieve2"
NOVEL_IDEA = True
UPDATE_HYPOTHESIS = 'Entry-only novel Sieve3 probe: 1d support hold plus 4h break as higher-timeframe context, with 1h higher low break execution.'
import numpy as np
from freqtrade.strategy import IStrategy, merge_informative_pair
SIDE = 'long'

CONTEXT_MODE = 'support_h4_break'
EXECUTION_MODE = 'higher_low_break'
SIEVE_STAGE = "sieve3"
SOURCE_ENTRY_STAGE = 'sieve2'
SOURCE_ENTRY_CLASS = 'Sieve3ExitNovelMtfD1SupportHoldH4Break1hHigherLowBreakLong'
SOURCE_STRATEGY = 'user_data/strategies/sieve3_exit_novel_mtf_d1_support_hold_h4_break_1h_higher_low_break_long.py:Sieve3ExitNovelMtfD1SupportHoldH4Break1hHigherLowBreakLong'
LINEAGE_STATUS = 'verified/domain_extension_required'
EXIT_FAMILY = 'profit_ladder_three_stage_ratchet'
EXIT_THEORY = 'profit_ladder_three_stage_ratchet'
EXIT_HYPOTHESIS = 'Three ordered arbitrary profit targets take two independently sized partials and fully exit the remainder; the stop ratchets to entry and then target one after filled partials.'
PRIMARY_TRIGGER = 'hl_1h confirmed with close > range_mid_1h, bullish candle, and body_ratio_1h >= 0.29'
PRIMARY_GUARD = 'close_1d > range_mid_1d and close_4h > prior_high_4h with a bullish 4h candle'
TARGET_PROVIDER = 'none'
INVALIDATION_PROVIDER = 'none'
ACTIVE_SELL_PARAMS = ('target_1_percent', 'target_gap_half_percent_units', 'partial_1_five_percent_units', 'partial_2_five_percent_units', 'hard_stop_percent')

RESEARCH_PATH = "sieve3_exit_profit_ladder_three_stage_ratchet"
SOURCE_RESULT_BATCH = "historical_promotion_lineage_unknown"


class Sieve3V2ProfitLadderThreeStageRatchetFromNovelMtfD1SupportHoldH4Break1HHigherLowBreakLong(IStrategy):
    timeframe = '1h'
    can_short = False
    startup_candle_count = 260
    process_only_new_candles = True
    INTERFACE_VERSION = 3
    context_lookback = 22
    h4_lookback = 23
    exec_lookback = 4
    body_ratio_min = 0.47
    range_ratio_min = 1.96
    volume_ratio_min = 1.71
    # Legacy inert entry parameter: retest_tolerance=0.037 (fixed entry mode/gate).
    close_follow_min = 0.29
    # Legacy inert entry parameter: require_h4_confirm=True (fixed entry mode/gate).
    # Legacy fixed-off entry parameter: require_volume_confirm=False (CategoricalParameter, buy space).

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
        mid = self._num(dataframe, 'range_mid_1h')
        follow = float(self.close_follow_min)
        bull = close.gt(open_) & self._num(dataframe, 'body_ratio_1h').ge(follow)
        exe = self._bool(dataframe, 'hl_1h') & close.gt(mid) & bull
        return exe.fillna(False)

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
        resolved = Sieve3V2ProfitLadderThreeStageRatchetFromNovelMtfD1SupportHoldH4Break1HHigherLowBreakLong._column_alias(frame, column)
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
    SOURCE_ENTRY_STEM = 'novel_mtf_d1_support_hold_h4_break_1h_higher_low_break_long'
    ENTRY_TAG = 'novel_mtf_d1_support_hold_h4_break_1h_higher_low_break_long'
    FOCUSED_EXIT_CONTRACT = 'profit_ladder_three_stage_ratchet'
    FOCUSED_SOURCE_PROFILE = {'side': 'long', 'min_level_distance': 0.001}
    FOCUSED_EXIT_PLANS = {'profit_ladder_three_stage_ratchet': {'contract': 'profit_ladder_three_stage_ratchet', 'name': 'profit_ladder_three_stage_ratchet', 'action_sequence': ('profit_target_1_partial', 'profit_target_2_partial', 'profit_target_3_full', 'stop_ratchet', 'hard_stop', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336}}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'high', 'low', 'open')
    FOCUSED_STATE_VERSION = 1
    FOCUSED_STATE_KEY = 'sieve3_v2_focused:Sieve3V2ProfitLadderThreeStageRatchetFromNovelMtfD1SupportHoldH4Break1HHigherLowBreakLong:profit_ladder_three_stage_ratchet'
    LOCKED_BUY_PARAMS = {'context_lookback': 22, 'h4_lookback': 23, 'exec_lookback': 4, 'body_ratio_min': 0.47, 'range_ratio_min': 1.96, 'volume_ratio_min': 1.71, 'retest_tolerance': 0.037, 'close_follow_min': 0.29, 'require_h4_confirm': True, 'require_volume_confirm': False}
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
        filled_at = self._focused_utc(state.get('entry_filled_at'))
        if filled_at is None:
            raise ValueError('focused exit state has no entry fill timestamp')
        dates = pd.to_datetime(frame['date'], utc=True, errors='raise')
        return frame.loc[dates.ge(filled_at)]

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
        if state.get('stages'):
            self._focused_sync_partial_stages(trade, state, current_time)
        plan = self._focused_plan(state)
        frame = self._focused_analyzed_frame(pair, current_time)
        self._focused_require_columns(frame)
        post_entry = self._focused_post_entry(frame, state)
        self._focused_update_favorable(post_entry, state)
        events = {}
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

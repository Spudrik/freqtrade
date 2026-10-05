from __future__ import annotations
from freqtrade.strategy import informative
from user_data.Indicators.complex_volume_profile import add_volume_profile
import math
import pandas as pd
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from typing import Any
from pandas import DataFrame, Series
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, stoploss_from_absolute, IntParameter
ENTRY_SOURCE_STAGE = "sieve2"
NOVEL_IDEA = True
UPDATE_HYPOTHESIS = 'Entry-only novel Sieve3 probe: 1d+4h aligned break as higher-timeframe context, with 1h bos execution.'
import numpy as np
from freqtrade.strategy import IStrategy, merge_informative_pair
SIDE = 'long'

CONTEXT_MODE = 'dual_break'
EXECUTION_MODE = 'bos'
SIEVE_STAGE = "sieve3"
SOURCE_ENTRY_STAGE = 'sieve3'
SOURCE_ENTRY_CLASS = 'Sieve3ExitNovelMtfD1H4DualBreak1hBosLong'
SOURCE_STRATEGY = 'user_data/strategies/sieve3_exit_novel_mtf_d1_h4_dual_break_1h_bos_long.py:Sieve3ExitNovelMtfD1H4DualBreak1hBosLong'
LINEAGE_STATUS = 'verified_selected_defaults_from_accepted_20260606T100454; historical_candidate_path_stale'
EXIT_FAMILY = 'generic_scored_level_partial_progression'
EXIT_THEORY = 'generic_scored_level_partial_progression'
EXIT_HYPOTHESIS = 'Use the deterministic retained level-zone score, advance through clean passes, take one or two coarse partials on confirmed reactions, and optionally ratchet protection behind progress.'
PRIMARY_TRIGGER = '1h BOS long: close > prior_high_1h and hl_1h'
PRIMARY_GUARD = 'D1/H4 prior-high dual break plus selected D1 volume_ratio_1d >= 1.46; H4 candle confirmation disabled'
TARGET_PROVIDER = 'closed-candle actual-fill-relative deterministic named level zone'
INVALIDATION_PROVIDER = 'coherent source invalidation when available, otherwise preserved 3% hard stop'
ACTIVE_SELL_PARAMS = ('zone_width_quarter_percent', 'partial_1_five_percent_units', 'partial_2_five_percent_units', 'ratchet_lag_levels', 'reaction_profile')

RESEARCH_PATH = 'sieve3_exit_final_generic_levels'
SOURCE_RESULT_BATCH = "historical_promotion_lineage_unknown"


class Sieve3V2GenericScoredLevelPartialProgressionFromNovelMtfD1H4DualBreak1HBosLong(IStrategy):
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

    @classmethod
    def _num(cls, frame: DataFrame, column: str, default: float=0.0) -> Series:
        resolved = cls._column_alias(frame, column)
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

    @informative('4h', fmt='gsl_{column}_{timeframe}')
    def populate_generic_levels_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        return self._generic_level_frame(dataframe, '4h')

    @informative('8h', fmt='gsl_{column}_{timeframe}')
    def populate_generic_levels_8h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        return self._generic_level_frame(dataframe, '8h')

    @informative('1d', fmt='gsl_{column}_{timeframe}')
    def populate_generic_levels_1d_(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        return self._generic_level_frame(dataframe, '1d')

    @informative('3d', fmt='gsl_{column}_{timeframe}')
    def populate_generic_levels_3d_(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        return self._generic_level_frame(dataframe, '3d')

    @informative('1h', 'BTC/USDT:USDT', fmt='gsbtc_{column}_{timeframe}')
    def populate_generic_btc_levels_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        return self._generic_level_frame(dataframe, '1h')

    @informative('4h', 'BTC/USDT:USDT', fmt='gsbtc_{column}_{timeframe}')
    def populate_generic_btc_levels_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        return self._generic_level_frame(dataframe, '4h')

    @informative('8h', 'BTC/USDT:USDT', fmt='gsbtc_{column}_{timeframe}')
    def populate_generic_btc_levels_8h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        return self._generic_level_frame(dataframe, '8h')

    @informative('1d', 'BTC/USDT:USDT', fmt='gsbtc_{column}_{timeframe}')
    def populate_generic_btc_levels_1d_(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        return self._generic_level_frame(dataframe, '1d')

    @informative('3d', 'BTC/USDT:USDT', fmt='gsbtc_{column}_{timeframe}')
    def populate_generic_btc_levels_3d_(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        return self._generic_level_frame(dataframe, '3d')

    def _generic_level_frame(self, dataframe: DataFrame, _timeframe: str) -> DataFrame:
        return add_volume_profile(dataframe, prefix='gsvp')
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        frame = self._sieve3_entry_populate_indicators(dataframe, metadata)
    
        if True:
            frame = self._generic_level_frame(frame, self.timeframe)
        return frame

    SOURCE_ENTRY_STEM = 'novel_mtf_d1_h4_dual_break_1h_bos_long'
    FOCUSED_EXIT_CONTRACT = 'generic_scored_level_partial_progression'
    FOCUSED_SOURCE_PROFILE = {'side': 'long', 'min_level_distance': 0.001}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'high', 'low', 'open')
    FOCUSED_STATE_VERSION = 1
    FOCUSED_STATE_KEY = 'sieve3_v2_generic:Sieve3V2GenericScoredLevelPartialProgressionFromNovelMtfD1H4DualBreak1HBosLong:generic_scored_level_partial_progression'
    LOCKED_BUY_PARAMS = {'context_lookback': 38, 'h4_lookback': 14, 'exec_lookback': 24, 'body_ratio_min': 0.38, 'range_ratio_min': 2.1, 'volume_ratio_min': 1.46, 'retest_tolerance': 0.021, 'close_follow_min': 0.11, 'require_h4_confirm': False, 'require_volume_confirm': True}
    ACTIVE_SELL_PARAMS = ('zone_width_quarter_percent', 'partial_1_five_percent_units', 'partial_2_five_percent_units', 'ratchet_lag_levels', 'reaction_profile')
    GENERIC_MODE = 'scored_partial'
    GENERIC_LEVEL_FAMILY = 'all'
    GENERIC_TIMEFRAME_SCOPE = 'all'
    GENERIC_AVAILABLE_TIMEFRAMES = ('1h', '4h', '8h', '1d', '3d')
    GENERIC_HVN_MIN_STRENGTH = 0.5
    GENERIC_MIN_LEVEL_DISTANCE = 0.001
    GENERIC_HARD_STOP_RATIO = 0.03
    position_adjustment_enable = True
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    minimal_roi = {'0': 100.0}
    stoploss = -0.99
    zone_width_quarter_percent = IntParameter(1, 12, default=4, space='sell', optimize=True, load=True)
    zone_width_quarter_percent.batch_tags = ('family:exits', 'mode:sieve3_exit')
    partial_1_five_percent_units = IntParameter(1, 8, default=5, space='sell', optimize=True, load=True)
    partial_1_five_percent_units.batch_tags = ('family:exits', 'mode:sieve3_exit')
    partial_2_five_percent_units = IntParameter(0, 8, default=3, space='sell', optimize=True, load=True)
    partial_2_five_percent_units.batch_tags = ('family:exits', 'mode:sieve3_exit')
    ratchet_lag_levels = IntParameter(0, 2, default=1, space='sell', optimize=True, load=True)
    ratchet_lag_levels.batch_tags = ('family:exits', 'mode:sieve3_exit')
    reaction_profile = CategoricalParameter(('2_of_2', '2_of_3', '3_of_3'), default='2_of_3', space='sell', optimize=True, load=True)
    reaction_profile.batch_tags = ('family:exits', 'mode:sieve3_exit')

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        dataframe['exit_tag'] = None
        return dataframe

    @staticmethod
    def _generic_utc(value: Any) -> pd.Timestamp | None:
        if value is None:
            return None
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize('UTC')
        return timestamp.tz_convert('UTC')

    @staticmethod
    def _generic_float(value: Any) -> float | None:
        if value is None or bool(pd.isna(value)):
            return None
        number = float(value)
        return number if math.isfinite(number) else None

    def _generic_closed_rows(self, frame: DataFrame, current_time: Any, cursor: Any=None) -> DataFrame:
        if frame is None or frame.empty or 'date' not in frame.columns:
            raise RuntimeError('generic level exit requires a dated analyzed dataframe')
        dates = frame['date']
        if not pd.api.types.is_datetime64_any_dtype(dates.dtype):
            raise TypeError('generic level exit requires Freqtrade datetime values in the date column')
        now = self._generic_utc(current_time)
        if now is None:
            raise ValueError('current_time is required for closed-candle evaluation')
        cutoff = now - pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit='m')
        normalized_cursor = self._generic_utc(cursor)
        timezone = dates.dt.tz
        if timezone is None:
            cutoff = cutoff.tz_localize(None)
            normalized_cursor = normalized_cursor.tz_localize(None) if normalized_cursor is not None else None
        else:
            cutoff = cutoff.tz_convert(timezone)
            normalized_cursor = normalized_cursor.tz_convert(timezone) if normalized_cursor is not None else None
        start = int(dates.searchsorted(normalized_cursor, side='right')) if normalized_cursor is not None else 0
        stop = int(dates.searchsorted(cutoff, side='right'))
        return frame.iloc[min(start, stop):stop]

    def _generic_analyzed_frame(self, pair: str) -> DataFrame:
        if getattr(self, 'dp', None) is None:
            raise RuntimeError("generic level exit requires Freqtrade's data provider")
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if frame is None or frame.empty:
            raise RuntimeError('generic level exit requires a non-empty analyzed dataframe')
        missing = sorted(set(self.FOCUSED_REQUIRED_COLUMNS) - set(frame.columns))
        if missing:
            raise KeyError(f'generic level source is missing exact columns: {missing}')
        return frame

    def _generic_side(self, trade: Any) -> str:
        side = 'short' if bool(getattr(trade, 'is_short', False)) else 'long'
        declared = str(self.FOCUSED_SOURCE_PROFILE['side'])
        if declared != 'both' and declared != side:
            raise ValueError(f'trade side {side!r} violates source side {declared!r}')
        return side

    def _generic_state(self, trade: Any) -> dict[str, Any] | None:
        cache_attr = f'_sieve3_generic_state_{self.FOCUSED_STATE_KEY}'
        state = getattr(trade, cache_attr, None)
        if state is None:
            state = trade.get_custom_data(key=self.FOCUSED_STATE_KEY)
        if state is None:
            return None
        if not isinstance(state, dict):
            raise ValueError('generic level trade state must be a mapping')
        if state.get('version') != self.FOCUSED_STATE_VERSION:
            raise ValueError('generic level trade state version mismatch')
        if state.get('contract') != self.FOCUSED_EXIT_CONTRACT:
            raise ValueError('generic level trade state contract mismatch')
        if getattr(trade, cache_attr, None) is None:
            setattr(trade, cache_attr, deepcopy(state))
        return deepcopy(state)

    def _generic_save_state(self, trade: Any, state: Mapping[str, Any]) -> None:
        cache_attr = f'_sieve3_generic_state_{self.FOCUSED_STATE_KEY}'
        snapshot = deepcopy(dict(state))
        setattr(trade, cache_attr, snapshot)
        trade.set_custom_data(key=self.FOCUSED_STATE_KEY, value=deepcopy(snapshot))

    def _generic_source_invalidation(self, row: Series, side: str) -> float | None:
        provider = self.FOCUSED_SOURCE_PROFILE.get('invalidation')
        if not isinstance(provider, Mapping):
            return None
        binding = provider.get(side)
        if not isinstance(binding, Mapping):
            return None
        available = binding.get('available')
        if available:
            value = row.get(str(available))
            if bool(pd.isna(value)) or not bool(value):
                return None
        column = binding.get('level')
        return self._generic_float(row.get(str(column))) if column else None

    def _generic_column(self, base: str, timeframe: str, btc: bool=False) -> str:
        if btc:
            return f'gsbtc_{base}_{timeframe}'
        return base if timeframe == self.timeframe else f'gsl_{base}_{timeframe}'

    def _generic_value(self, row: Series, base: str, timeframe: str, btc: bool=False) -> float | None:
        column = self._generic_column(base, timeframe, btc)
        if column not in row.index:
            raise KeyError(f'generic level column is missing: {column}')
        return self._generic_float(row[column])

    def _generic_candidate(self, row: Series, side: str, timeframe: str, family: str, btc: bool=False) -> dict[str, Any] | None:
        if family == 'va_edge':
            base = 'gsvp_prior_val' if side == 'short' else 'gsvp_prior_vah'
            level = self._generic_value(row, base, timeframe, btc)
            metadata: dict[str, Any] = {}
            score = 3

        elif family == 'hvn_strength':
            base = 'gsvp_hvn_below' if side == 'short' else 'gsvp_hvn_above'
            level = self._generic_value(row, base, timeframe, btc)
            strength = self._generic_value(row, f'{base}_strength', timeframe, btc)
            threshold = self.GENERIC_HVN_MIN_STRENGTH
            if self.GENERIC_MODE == 'simple':
                threshold = float(self.hvn_strength_tenth_units.value) * 0.1
            if strength is None or strength < threshold:
                return None
            metadata = {'strength': strength}
            score = 3 + int(strength >= 0.85)
        else:
            raise ValueError(f'unsupported primitive family: {family}')
        if level is None or level <= 0.0:
            return None
        return {'level': level, 'family': family, 'timeframe': timeframe, 'score': score, 'metadata': metadata}


    @staticmethod
    def _generic_forward(candidates: list[dict[str, Any]], side: str, reference: float) -> list[dict[str, Any]]:
        minimum = 0.001
        if side == 'short':
            valid = [item for item in candidates if float(item['level']) < reference * (1.0 - minimum)]
        else:
            valid = [item for item in candidates if float(item['level']) > reference * (1.0 + minimum)]
        return sorted(valid, key=lambda item: abs(float(item['level']) - reference))

    def _generic_primitives(self, row: Series, side: str, timeframe: str, reference: float, btc: bool=False) -> list[dict[str, Any]]:
        candidates = [
            candidate
            for family in ('va_edge', 'hvn_strength')
            for candidate in [self._generic_candidate(row, side, timeframe, family, btc)]
            if candidate is not None
        ]
        return self._generic_forward(candidates, side, reference)

    def _generic_btc_strength(self, row: Series, side: str, band: float) -> int:
        if self.GENERIC_MODE == 'simple':
            return 0
        near: list[dict[str, Any]] = []
        for timeframe in self.GENERIC_AVAILABLE_TIMEFRAMES:
            close = self._generic_value(row, 'close', timeframe, True)
            if close is None or close <= 0.0:
                continue
            for candidate in self._generic_primitives(row, side, timeframe, close, True):
                if abs(float(candidate['level']) / close - 1.0) <= band:
                    near.append(candidate)
        if not near:
            return 0
        families = {str(item['family']) for item in near}
        timeframes = {str(item['timeframe']) for item in near}
        return 2 if len(families) > 1 or len(timeframes) > 1 else 1

    def _generic_scored_zones(self, row: Series, side: str, reference: float, pair: str) -> list[dict[str, Any]]:
        band = float(self.zone_width_quarter_percent.value) * 0.0025
        primitives = [
            candidate
            for timeframe in self.GENERIC_AVAILABLE_TIMEFRAMES
            for candidate in self._generic_primitives(row, side, timeframe, reference)
        ]
        retained = primitives
        zones: list[dict[str, Any]] = []
        for item in retained:
            zones.append(
                {
                    'center': float(item['level']),
                    'lower': float(item['level']) * (1.0 - band),
                    'upper': float(item['level']) * (1.0 + band),
                    'score': int(item['score']),
                    'families': [str(item['family'])],
                    'timeframes': [str(item['timeframe'])],
                    'members': [item],
                }
            )
        seen: set[tuple[str, ...]] = set()
        for anchor in primitives:
            anchor_level = float(anchor['level'])
            members = [item for item in primitives if abs(float(item['level']) / anchor_level - 1.0) <= band]
            key = tuple(sorted(f"{item['family']}:{item['timeframe']}:{float(item['level']):.10g}" for item in members))
            if len(members) < 2 or key in seen:
                continue
            seen.add(key)
            levels = sorted(float(item['level']) for item in members)
            families = {str(item['family']) for item in members}
            timeframes = {str(item['timeframe']) for item in members}
            score = max(int(item['score']) for item in members)
            score += int(len(families) > 1)
            if len(timeframes) > 1:
                score += (
                    int(self.mtf_alignment_bonus.value)
                    if self.GENERIC_MODE == 'scored_full'
                    else 1
                )
            zones.append(
                {
                    'center': float(np.median(levels)),
                    'lower': min(levels) * (1.0 - band),
                    'upper': max(levels) * (1.0 + band),
                    'score': score,
                    'families': sorted(families),
                    'timeframes': sorted(timeframes, key=timeframe_to_minutes),
                    'members': members,
                }
            )
        if not str(pair).upper().startswith('BTC/'):
            btc_strength = self._generic_btc_strength(row, side, band)
            configured = (
                int(self.btc_same_side_bonus.value)
                if self.GENERIC_MODE == 'scored_full'
                else 1
            )
            for zone in zones:
                zone['score'] += min(configured, btc_strength)
                zone['btc_context_strength'] = btc_strength
        minimum_score = (
            int(self.minimum_level_score.value)
            if self.GENERIC_MODE == 'scored_full'
            else 4
        )
        return [zone for zone in zones if int(zone['score']) >= minimum_score]

    def _generic_simple_zones(self, row: Series, side: str, reference: float) -> list[dict[str, Any]]:
        timeframe = self.timeframe if self.GENERIC_TIMEFRAME_SCOPE == 'base' else str(self.target_timeframe.value)
        candidate = self._generic_candidate(row, side, timeframe, self.GENERIC_LEVEL_FAMILY)
        candidates = [candidate] if candidate is not None else []
        band = float(self.zone_width_quarter_percent.value) * 0.0025
        return [
            {
                'center': float(item['level']),
                'lower': float(item['level']) * (1.0 - band),
                'upper': float(item['level']) * (1.0 + band),
                'score': int(item['score']),
                'families': [str(item['family'])],
                'timeframes': [str(item['timeframe'])],
                'members': [item],
            }
            for item in self._generic_forward(candidates, side, reference)
        ]

    def _generic_select_zone(self, row: Series, side: str, reference: float, pair: str, passed: list[float]) -> dict[str, Any] | None:
        zones = self._generic_simple_zones(row, side, reference) if self.GENERIC_MODE == 'simple' else self._generic_scored_zones(row, side, reference, pair)
        band = float(self.zone_width_quarter_percent.value) * 0.0025
        zones = [zone for zone in zones if not any(abs(float(zone['center']) / old - 1.0) <= band for old in passed if old > 0.0)]
        if not zones:
            return None
        if self.GENERIC_MODE == 'simple':
            return min(zones, key=lambda zone: abs(float(zone['center']) - reference))
        mode = str(self.level_order_mode.value) if self.GENERIC_MODE == 'scored_full' else 'highest_score'
        if mode == 'nearest_qualified':
            return min(zones, key=lambda zone: abs(float(zone['center']) - reference))
        if mode == 'highest_score':
            return min(zones, key=lambda zone: (-int(zone['score']), abs(float(zone['center']) - reference)))
        if mode == 'higher_tf_priority':
            return min(zones, key=lambda zone: (-max(timeframe_to_minutes(tf) for tf in zone['timeframes']), -int(zone['score']), abs(float(zone['center']) - reference)))
        raise ValueError(f'unsupported level order mode: {mode}')

    def _generic_new_state(self, pair: str, trade: Any, current_time: Any) -> dict[str, Any]:
        side = self._generic_side(trade)
        entry_rate = self._generic_float(getattr(trade, 'open_rate', None))
        if entry_rate is None or entry_rate <= 0.0:
            raise ValueError('generic level exit requires a positive actual fill rate')
        filled_at = self._generic_utc(getattr(trade, 'date_entry_fill_utc', None) or getattr(trade, 'open_date_utc', None) or current_time)
        frame = self._generic_closed_rows(self._generic_analyzed_frame(pair), filled_at)
        row = frame.iloc[-1] if not frame.empty else None
        zone = self._generic_select_zone(row, side, entry_rate, pair, []) if row is not None else None
        invalidation = self._generic_source_invalidation(row, side) if row is not None else None
        cursor = self._generic_utc(row['date']).isoformat() if row is not None else None
        return {
            'version': self.FOCUSED_STATE_VERSION,
            'contract': self.FOCUSED_EXIT_CONTRACT,
            'side': side,
            'entry_rate': entry_rate,
            'entry_filled_at': filled_at.isoformat() if filled_at is not None else None,
            'last_processed_candle': cursor,
            'last_callback_bucket': None,
            'zone': zone,
            'zone_touched': False,
            'reaction_history': [],
            'passed_centers': [],
            'protected_levels': [],
            'source_invalidation': invalidation,
            'stop_price': None,
            'terminal_tag': None,
            'pending_partial': None,
            'completed_partials': int(getattr(trade, 'nr_of_successful_exits', 0) or 0),
            'initial_stake': self._generic_float(getattr(trade, 'stake_amount', None)),
            'previous_close': entry_rate,
        }

    def _generic_ensure_state(self, pair: str, trade: Any, current_time: Any) -> dict[str, Any]:
        state = self._generic_state(trade)
        if state is None:
            state = self._generic_new_state(pair, trade, current_time)
            self._generic_save_state(trade, state)
        return state

    @staticmethod
    def _generic_pressure(row: Series) -> float:
        open_ = float(row['open'])
        high = float(row['high'])
        low = float(row['low'])
        close = float(row['close'])
        spread = max(high - low, 1e-12)
        body = (close - open_) / spread
        location = (close - low) / spread * 2.0 - 1.0
        return max(-1.0, min(1.0, (body + location) / 2.0))

    def _generic_weakening(self, row: Series, state: Mapping[str, Any]) -> tuple[bool, bool]:
        side = str(state['side'])
        zone = state['zone']
        open_ = float(row['open'])
        close = float(row['close'])
        previous = float(state.get('previous_close') or state['entry_rate'])
        opposite_body = close > open_ if side == 'short' else close < open_
        close_momentum = close > previous if side == 'short' else close < previous
        pressure = self._generic_pressure(row)
        directional_pressure = pressure > 0.0 if side == 'short' else pressure < 0.0
        zone_rejection = close > float(zone['upper']) if side == 'short' else close < float(zone['lower'])
        if self.GENERIC_MODE == 'simple':
            selected = str(self.weakening_signal.value)
            values = {
                'opposite_body': opposite_body,
                'close_momentum': close_momentum,
                'directional_pressure': directional_pressure,
                'zone_rejection': zone_rejection,
            }
            return bool(values[selected]), zone_rejection
        return sum((opposite_body, close_momentum, directional_pressure)) >= 2, zone_rejection

    def _generic_reaction_confirmed(self, row: Series, state: dict[str, Any]) -> bool:
        weakening, rejection = self._generic_weakening(row, state)
        if self.GENERIC_MODE == 'scored_full' and str(self.reaction_profile.value) == 'rejection_close':
            return rejection
        profile = str(self.confirmation_profile.value) if self.GENERIC_MODE == 'simple' else str(self.reaction_profile.value)
        required, window = {
            '1_of_1': (1, 1),
            '2_of_2': (2, 2),
            '2_of_3': (2, 3),
            '3_of_3': (3, 3),
        }[profile]
        history = list(state.get('reaction_history') or [])
        history.append(bool(weakening))
        state['reaction_history'] = history[-3:]
        recent = history[-window:]
        return len(recent) == window and sum(bool(value) for value in recent) >= required

    def _generic_ratchet(self, state: dict[str, Any], level: float) -> None:
        if self.GENERIC_MODE != 'scored_partial':
            return
        protected = list(state.get('protected_levels') or [])
        protected.append(float(level))
        state['protected_levels'] = protected
        lag = int(self.ratchet_lag_levels.value)
        if lag <= 0:
            return
        candidate = float(state['entry_rate']) if len(protected) <= lag else protected[len(protected) - lag - 1]
        current = self._generic_float(state.get('stop_price'))
        state['stop_price'] = candidate if current is None else min(current, candidate) if state['side'] == 'short' else max(current, candidate)

    def _generic_reaction_action(self, state: dict[str, Any]) -> None:
        zone = state.get('zone')
        if not isinstance(zone, Mapping):
            return
        center = float(zone['center'])
        if self.GENERIC_MODE != 'scored_partial':
            state['terminal_tag'] = f"generic_{self.GENERIC_LEVEL_FAMILY}_reaction_full"
            return
        configured = [float(self.partial_1_five_percent_units.value) * 0.05]
        second = int(self.partial_2_five_percent_units.value)
        if second > 0:
            configured.append(float(second) * 0.05)
        completed = int(state.get('completed_partials') or 0)
        if completed < len(configured):
            state['pending_partial'] = {
                'fraction': configured[completed],
                'tag': f'generic_level_partial_{completed + 1}',
                'requested': False,
                'ratchet_level': center,
            }
        else:
            state['terminal_tag'] = 'generic_level_progression_remainder'
        passed = list(state.get('passed_centers') or [])
        passed.append(center)
        state['passed_centers'] = passed
        state['zone'] = None
        state['zone_touched'] = False
        state['reaction_history'] = []

    def _generic_process_row(self, row: Series, pair: str, state: dict[str, Any]) -> None:
        side = str(state['side'])
        close = float(row['close'])
        invalidation = self._generic_float(state.get('source_invalidation'))
        if invalidation is not None and ((side == 'short' and close >= invalidation) or (side == 'long' and close <= invalidation)):
            state['terminal_tag'] = 'generic_source_invalidation'
            return
        if state.get('zone') is None:
            state['zone'] = self._generic_select_zone(row, side, close, pair, list(state.get('passed_centers') or []))
            state['zone_touched'] = False
            state['reaction_history'] = []
        zone = state.get('zone')
        if not isinstance(zone, Mapping):
            state['previous_close'] = close
            return
        high = float(row['high'])
        low = float(row['low'])
        touched = low <= float(zone['upper']) if side == 'short' else high >= float(zone['lower'])
        clean_pass = close < float(zone['lower']) if side == 'short' else close > float(zone['upper'])
        if clean_pass:
            center = float(zone['center'])
            passed = list(state.get('passed_centers') or [])
            passed.append(center)
            state['passed_centers'] = passed
            self._generic_ratchet(state, center)
            state['zone'] = self._generic_select_zone(row, side, close, pair, passed)
            state['zone_touched'] = False
            state['reaction_history'] = []
        elif touched or bool(state.get('zone_touched')):
            state['zone_touched'] = True
            if self._generic_reaction_confirmed(row, state):
                self._generic_reaction_action(state)
        state['previous_close'] = close

    def _generic_sync_partial(self, trade: Any, state: dict[str, Any]) -> bool:
        native = int(getattr(trade, 'nr_of_successful_exits', 0) or 0)
        completed = int(state.get('completed_partials') or 0)
        pending = state.get('pending_partial')
        if native > completed:
            if isinstance(pending, Mapping):
                ratchet_level = self._generic_float(pending.get('ratchet_level'))
                if ratchet_level is not None:
                    self._generic_ratchet(state, ratchet_level)
            state['completed_partials'] = native
            state['pending_partial'] = None
            return True
        if isinstance(pending, Mapping) and bool(pending.get('requested')) and native < completed:
            raise ValueError('native partial-exit count moved backwards')
        return False

    def _generic_context(self, pair: str, trade: Any, current_time: Any) -> dict[str, Any]:
        state = self._generic_ensure_state(pair, trade, current_time)
        if self._generic_sync_partial(trade, state):
            self._generic_save_state(trade, state)
        if state.get('pending_partial'):
            self._generic_save_state(trade, state)
            return state
        bucket = self._generic_utc(current_time)
        bucket_key = bucket.floor(f'{max(1, timeframe_to_minutes(self.timeframe))}min').isoformat() if bucket is not None else None
        if state.get('last_callback_bucket') == bucket_key:
            return state
        state['last_callback_bucket'] = bucket_key
        if state.get('terminal_tag'):
            self._generic_save_state(trade, state)
            return state
        cursor = self._generic_utc(state.get('last_processed_candle'))
        rows = self._generic_closed_rows(self._generic_analyzed_frame(pair), current_time, cursor)
        for _, row in rows.iterrows():
            self._generic_process_row(row, pair, state)
            state['last_processed_candle'] = self._generic_utc(row['date']).isoformat()
            if state.get('terminal_tag') or state.get('pending_partial'):
                break
        self._generic_save_state(trade, state)
        return state

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | bool | None:
        _ = (current_rate, current_profit, kwargs)
        state = self._generic_context(pair, trade, current_time)
        return str(state['terminal_tag']) if state.get('terminal_tag') else None

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float | None:
        _ = (current_profit, after_fill, kwargs)
        state = self._generic_context(pair, trade, current_time)
        entry = float(state['entry_rate'])
        hard = entry * (1.0 + self.GENERIC_HARD_STOP_RATIO if state['side'] == 'short' else 1.0 - self.GENERIC_HARD_STOP_RATIO)
        stop = hard
        invalidation = self._generic_float(state.get('source_invalidation'))
        if invalidation is not None:
            stop = min(stop, invalidation) if state['side'] == 'short' else max(stop, invalidation)
        ratchet = self._generic_float(state.get('stop_price'))
        if ratchet is not None:
            stop = min(stop, ratchet) if state['side'] == 'short' else max(stop, ratchet)
        return stoploss_from_absolute(stop, current_rate, is_short=state['side'] == 'short', leverage=float(getattr(trade, 'leverage', 1.0) or 1.0))

    def adjust_trade_position(self, trade: Any, current_time: datetime, current_rate: float, current_profit: float, min_stake: float | None, max_stake: float, current_entry_rate: float, current_exit_rate: float, current_entry_profit: float, current_exit_profit: float, **kwargs: Any) -> float | None | tuple[float | None, str | None]:
        _ = (current_rate, current_profit, min_stake, max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, kwargs)
        if self.GENERIC_MODE != 'scored_partial':
            return None
        state = self._generic_context(trade.pair, trade, current_time)
        pending = state.get('pending_partial')
        if not isinstance(pending, Mapping) or bool(pending.get('requested')):
            return None
        initial = self._generic_float(state.get('initial_stake')) or self._generic_float(getattr(trade, 'stake_amount', None))
        current = self._generic_float(getattr(trade, 'stake_amount', None))
        if initial is None or current is None or current <= 0.0:
            return None
        amount = min(current, initial * float(pending['fraction']))
        if amount <= 0.0:
            return None
        pending = dict(pending)
        pending['requested'] = True
        state['pending_partial'] = pending
        self._generic_save_state(trade, state)
        return -amount, str(pending['tag'])

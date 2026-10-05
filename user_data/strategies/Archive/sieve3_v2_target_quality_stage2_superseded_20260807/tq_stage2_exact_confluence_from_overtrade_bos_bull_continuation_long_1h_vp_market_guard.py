from __future__ import annotations
import math
import pandas as pd
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from typing import Any
from pandas import DataFrame, Series
from user_data.Indicators.market_state import add_market_state
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import informative, CategoricalParameter, stoploss_from_absolute, IntParameter
import numpy as np
from freqtrade.strategy import IStrategy
from user_data.Indicators.pattern_bos_choch import add_bos_choch
from user_data.Indicators.complex_volume_profile import add_volume_profile
from user_data.Indicators.complex_trendline_projection_v2 import add_trendline_projection_v2

SIEVE2_VP_GUARD_MODES = [
    "score_or_context",
    "node_confirm",
    "value_area_confirm",
    "breakout_acceptance",
    "rejection_confirm",
    "poc_hvn_reject",
    "prior_level_confirm",
]
SIEVE2_MARKET_GUARD_MODES = [
    "pressure_or_trend",
    "pressure_and_trend",
    "directional_pressure",
    "trend_state",
    "avoid_adverse_pressure",
    "avoid_chop",
]


def add_sieve2_guard_indicators(strategy: Any, dataframe: pd.DataFrame, metadata: dict | None) -> pd.DataFrame:
    frame = dataframe.copy()
    frame = add_market_state(frame, window=int(_sieve2_param(strategy, "sieve2_market_window", 24)), prefix="s2m")
    frame = add_volume_profile(
        frame,
        window=int(_sieve2_param(strategy, "sieve2_vp_window", 96)),
        bins=int(_sieve2_param(strategy, "sieve2_vp_bins", 36)),
        value_area_pct=0.70,
        price_source="hlc3",
        smooth_bins=3,
        pressure_delta_min=0.05,
        node_near_pct=0.01,
        prefix="s2vp",
    )
    return frame


def apply_sieve2_optional_guards(
    strategy: Any, dataframe: pd.DataFrame, condition: pd.Series, side: str
) -> pd.Series:
    guarded = pd.Series(condition, index=dataframe.index).fillna(False).astype(bool)
    normalized_side = "short" if str(side or "").lower() == "short" else "long"
    always_on = _sieve2_enabled(getattr(strategy, "SIEVE2_ALWAYS_ON_GUARDS", False))
    if always_on or _sieve2_enabled(_sieve2_param(strategy, "use_sieve2_vp_guard", False)):
        guarded &= _sieve2_vp_guard(
            dataframe,
            normalized_side,
            str(_sieve2_param(strategy, "sieve2_vp_guard_mode", "score_or_context")),
            float(_sieve2_param(strategy, "sieve2_vp_score_min", 0.25)),
            float(_sieve2_param(strategy, "sieve2_vp_context_min", 0.28)),
        )
    if always_on or _sieve2_enabled(_sieve2_param(strategy, "use_sieve2_market_guard", False)):
        guarded &= _sieve2_market_guard(
            dataframe,
            normalized_side,
            str(_sieve2_param(strategy, "sieve2_market_guard_mode", "pressure_or_trend")),
            float(_sieve2_param(strategy, "sieve2_market_pressure_min", 0.07)),
            float(_sieve2_param(strategy, "sieve2_market_trend_min", 0.25)),
            float(0.45),
        )
    return guarded.fillna(False)


def _sieve2_vp_guard(frame: pd.DataFrame, side: str, mode: str, score_min: float, context_min: float) -> pd.Series:
    score = _sieve2_num(frame, f"s2vp_score_{side}")
    other = _sieve2_num(frame, "s2vp_score_short" if side == "long" else "s2vp_score_long")
    context = _sieve2_num(frame, "s2vp_context_score_bull" if side == "long" else "s2vp_context_score_bear")
    base_score = score.ge(score_min) & score.gt(other)
    base_context = context.ge(context_min)
    if mode == "node_confirm":
        node_entry = _sieve2_bool(frame, f"s2vp_node_entry_{side}")
        node_hold = _sieve2_bool(frame, f"s2vp_node_hold_{side}")
        return (node_entry | node_hold | (base_score & base_context)).fillna(False)
    if mode == "value_area_confirm":
        in_value = _sieve2_bool(frame, "s2vp_in_value_area")
        value_side = _sieve2_bool(frame, "s2vp_above_value_area" if side == "long" else "s2vp_below_value_area")
        return (value_side | (in_value & base_context)).fillna(False)
    if mode in {"breakout_acceptance", "rejection_confirm"}:
        close = _sieve2_num(frame, "close")
        prior_vah = _sieve2_num(frame, "s2vp_prior_vah")
        prior_val = _sieve2_num(frame, "s2vp_prior_val")
        node_entry = _sieve2_bool(frame, f"s2vp_node_entry_{side}")
        node_hold = _sieve2_bool(frame, f"s2vp_node_hold_{side}")
        if mode == "breakout_acceptance":
            accepted = close.gt(prior_vah) if side == "long" else close.lt(prior_val)
            return (accepted & (base_score | base_context | node_entry | node_hold)).fillna(False)
        high = _sieve2_num(frame, "high")
        low = _sieve2_num(frame, "low")
        rejection = low.le(prior_val) & close.gt(prior_val) & close.le(prior_vah) if side == "long" else high.ge(prior_vah) & close.lt(prior_vah) & close.ge(prior_val)
        return (rejection & (base_score | base_context | node_entry | node_hold)).fillna(False)
    if mode == "poc_hvn_reject":
        node_entry = _sieve2_bool(frame, f"s2vp_node_entry_{side}")
        return (node_entry | (base_score & base_context)).fillna(False)
    if mode == "prior_level_confirm":
        close = _sieve2_num(frame, "close")
        level = _sieve2_num(frame, "s2vp_prior_vah" if side == "long" else "s2vp_prior_val")
        confirmed = close.gt(level) if side == "long" else close.lt(level)
        return (confirmed & (base_score | base_context)).fillna(False)
    return (base_score | base_context).fillna(False)


def _sieve2_market_guard(frame: pd.DataFrame, side: str, mode: str, pressure_min: float, trend_min: float, rs_score_min: float) -> pd.Series:
    _ = rs_score_min
    if mode == "directional_pressure":
        pressure = _sieve2_num(frame, "s2m_pressure_ratio")
        directional = pressure.ge(pressure_min) if side == "long" else pressure.le(-pressure_min)
        return directional.fillna(False)
    if mode == "trend_state":
        trend = _sieve2_num(frame, "s2m_trend_z")
        state = trend.ge(trend_min) if side == "long" else trend.le(-trend_min)
        return state.fillna(False)
    pressure = _sieve2_num(frame, "s2m_pressure_ratio")
    trend = _sieve2_num(frame, "s2m_trend_z")
    directional = pressure.ge(pressure_min) if side == "long" else pressure.le(-pressure_min)
    trend_state = trend.ge(trend_min) if side == "long" else trend.le(-trend_min)
    avoids_pressure = pressure.ge(-pressure_min) if side == "long" else pressure.le(pressure_min)
    avoids_trend = trend.ge(-trend_min) if side == "long" else trend.le(trend_min)
    if mode == "pressure_and_trend": return (directional & trend_state).fillna(False)
    if mode == "avoid_adverse_pressure": return (avoids_pressure & avoids_trend).fillna(False)
    if mode == "avoid_chop": return (pressure.abs().ge(pressure_min) | trend.abs().ge(trend_min)).fillna(False)
    return (directional | trend_state).fillna(False)


def _sieve2_param(strategy: Any, name: str, default: Any) -> Any:
    value = getattr(strategy, name, default)
    return getattr(value, "value", value)


def _sieve2_enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _sieve2_num(frame: pd.DataFrame, column: str, default: float | pd.Series=0.0) -> pd.Series:
    if column not in frame.columns:
        raise KeyError(f'missing required column: {column!r}')
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)


def _sieve2_bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise KeyError(f'missing required column: {column!r}')
    return pd.Series(frame[column], index=frame.index).astype('boolean').fillna(False).astype(bool)
ENTRY_MODE = 'entry_overtrade_bos_bull_continuation_long_1h_vp_market_guard'

UPDATE_HYPOTHESIS = 'High-trade entry may contain a real directional edge; require stronger volume-profile/market context so HTF or broader context carries confidence and the original trigger supplies execution.'

TIMEFRAME = '1h'


def _num(frame: DataFrame, column: str, default: float | Series=0.0) -> Series:
    if column not in frame.columns:
        raise KeyError(f'missing required column: {column!r}')
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)

def _bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        raise KeyError(f'missing required column: {column!r}')
    return pd.Series(frame[column], index=frame.index).astype('boolean').fillna(False).astype(bool)

SOURCE_ENTRY_STAGE = 'sieve2'
SOURCE_ENTRY_CLASS = 'Sieve3ExitOvertradeBOSBullContinuationLong1hVPMarketGuard'
SOURCE_STRATEGY = 'user_data/strategies/sieve3_exit_from_overtrade_bos_bull_continuation_long_1h_vp_market_guard.py:Sieve3ExitOvertradeBOSBullContinuationLong1hVPMarketGuard'
LINEAGE_STATUS = 'verified'








ENTRY_SOURCE_STAGE = 'sieve2'
SOURCE_RESULT_BATCH = 'historical_promotion_lineage_unknown'
SIEVE_STAGE = 'target_quality_stage2'
SIDE = 'long'
ENTRY_TAG = 'overtrade_bos_bull_continuation_long_1h_vp_market_guard'

EXIT_FAMILY = 'target_quality_stage2_exact_confluence'
EXIT_THEORY = 'specific_level_family_composition'
EXIT_HYPOTHESIS = 'Compare explicit pairs, triples, and one four-family composition at a selected timeframe instead of treating all overlap as equivalent.'
PRIMARY_TRIGGER = 'entry-frozen nearest forward VP obstacle'
PRIMARY_GUARD = 'target touch/close or reversal confirmation inside the target zone'
PRIMARY_TARGET = 'actual-fill-relative exact family composition'
PRIMARY_INVALIDATION = 'none; source entry exposes no coherent frozen invalidation level'
TARGET_PROVIDER = 'actual-fill-relative exact family composition'
INVALIDATION_PROVIDER = 'none'
ACTIVE_SELL_PARAMS = ('confluence_composition', 'confluence_timeframe', 'confluence_band_quarter_percent', 'target_action', 'target_band_quarter_percent')
RESEARCH_PATH = 'target_quality_stage2'

STAGE2_INFORMATIVE_TIMEFRAMES = ('4h', '1d')
STAGE2_HVN_MIN_STRENGTH = 0.5
STAGE2_CONFLUENCE_COUNT = 3
STAGE2_CONFLUENCE_BAND = 0.0025
STAGE2_TLV2_MIN_SCORE = 0.5
STAGE2_TLV2_MIN_PIVOTS = 3
STAGE2_TLV2_MAX_AGE_BARS = 48
STAGE2_CONFLUENCE_TIMEFRAMES = ('1h', '4h', '1d')
STAGE2_EXACT_COMPOSITIONS = {
    'va_hvn': ('va_edge', 'hvn_strength'),
    'va_prior': ('va_edge', 'prior_poc'),
    'hvn_prior': ('hvn_strength', 'prior_poc'),
    'va_hvn_prior': ('va_edge', 'hvn_strength', 'prior_poc'),
    'va_hvn_tlv2': ('va_edge', 'hvn_strength', 'tlv2_quality'),
    'va_prior_tlv2': ('va_edge', 'prior_poc', 'tlv2_quality'),
    'hvn_prior_tlv2': ('hvn_strength', 'prior_poc', 'tlv2_quality'),
    'va_hvn_prior_tlv2': (
        'va_edge',
        'hvn_strength',
        'prior_poc',
        'tlv2_quality',
    ),
}

STAGE2_TARGET_COLUMN_BASES = (
    'tqvp_vah',
    'tqvp_val',
    'tqvp_prior_poc',
    'tqvp_hvn_above',
    'tqvp_hvn_below',
    'tqvp_hvn_above_strength',
    'tqvp_hvn_below_strength',
    'sieve3_tq_row_index',
    *tuple(
        f'tqtlv2_{line_side}_{field}_rank{rank}'
        for rank in range(3)
        for line_side in ('support', 'resistance')
        for field in ('line', 'score', 'pivot_count', 'last_confirm_index')
    ),
)
STAGE2_INFORMATIVE_REQUIRED_COLUMNS = tuple(
    f'mtf_{column}_{timeframe}'
    for timeframe in STAGE2_INFORMATIVE_TIMEFRAMES
    for column in STAGE2_TARGET_COLUMN_BASES
)

class TqStage2ExactConfluenceFromOvertradeBosBullContinuationLong1HVpMarketGuard(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 160
    process_only_new_candles = True
    can_short = False
    max_entry_position_adjustment = 0
    use_sieve2_vp_guard = True
    sieve2_vp_guard_mode = 'score_or_context'
    sieve2_vp_window = 96
    sieve2_vp_bins = 36
    sieve2_vp_score_min = 0.25
    sieve2_vp_context_min = 0.28
    use_sieve2_market_guard = True
    sieve2_market_guard_mode = 'pressure_or_trend'
    sieve2_market_window = 24
    sieve2_market_pressure_min = 0.07
    sieve2_market_trend_min = 0.25
    # Legacy inactive entry parameter: sieve2_rs_benchmark_pair='BTC/USDT:USDT' (fixed guard mode does not consume it).
    # Legacy inactive entry parameter: sieve2_rs_score_min=0.45 (fixed guard mode does not consume it).
    # Legacy inactive entry parameter: use_volume_guard=False (enable gate is fixed off).
    # Legacy inactive entry parameter: volume_guard_window=48 (its enable gate is fixed off).
    # Legacy inactive entry parameter: volume_ratio_min=1.3 (its enable gate is fixed off).
    use_pressure_guard = True
    pressure_window = 12
    pressure_min = 0.2
    # Legacy inactive entry parameter: use_accumulation_guard=False (enable gate is fixed off).
    # Legacy inactive entry parameter: use_body_direction_guard=False (enable gate is fixed off).
    # Legacy inactive entry parameter: use_close_direction_guard=False (enable gate is fixed off).
    strength = 3
    min_prominence_atr = 0.35
    min_pivot_spacing_bars = 2
    max_pivot_age_bars = 96
    breakout_buffer_atr = 0.3
    use_state_guard = True

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = (pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs)
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def _sieve3_entry_populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe = add_bos_choch(dataframe, strength=int(self.strength), min_prominence_atr=float(self.min_prominence_atr), min_pivot_spacing_bars=int(self.min_pivot_spacing_bars), max_pivot_age_bars=int(self.max_pivot_age_bars), breakout_buffer_atr=float(self.breakout_buffer_atr), prefix='ms')
        dataframe = add_sieve2_guard_indicators(self, dataframe, metadata)
        return dataframe

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype='bool')
        close = _num(dataframe, 'close')
        if bool(self.use_pressure_guard):
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
        return guard.fillna(False)


    @staticmethod
    def _sieve3_forward_target(dataframe: DataFrame, columns: tuple[str, ...], side: str) -> Series:
        missing = [column for column in columns if column not in dataframe.columns]
        if missing:
            raise KeyError(f'missing target columns: {missing}')
        close = pd.to_numeric(dataframe['close'], errors='coerce').replace([np.inf, -np.inf], np.nan)
        candidates = pd.concat([
            pd.to_numeric(dataframe[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
            for column in columns
        ], axis=1)
        if side == 'short':
            return candidates.where(candidates.lt(close, axis=0)).max(axis=1, skipna=True)
        return candidates.where(candidates.gt(close, axis=0)).min(axis=1, skipna=True)

    @informative('4h', fmt='mtf_{column}_{timeframe}')
    def populate_stage2_targets_4h(
        self,
        dataframe: DataFrame,
        metadata: dict,
    ) -> DataFrame:
        _ = metadata
        dataframe = add_volume_profile(dataframe, prefix='tqvp')
        dataframe = add_trendline_projection_v2(
            dataframe,
            timeframe='4h',
            raw_line_output_count=3,
            output_prefix='tqtlv2',
        )
        dataframe['sieve3_tq_row_index'] = np.arange(len(dataframe), dtype='int64')
        return dataframe

    @informative('1d', fmt='mtf_{column}_{timeframe}')
    def populate_stage2_targets_1d(
        self,
        dataframe: DataFrame,
        metadata: dict,
    ) -> DataFrame:
        _ = metadata
        dataframe = add_volume_profile(dataframe, prefix='tqvp')
        dataframe = add_trendline_projection_v2(
            dataframe,
            timeframe='1d',
            raw_line_output_count=3,
            output_prefix='tqtlv2',
        )
        dataframe['sieve3_tq_row_index'] = np.arange(len(dataframe), dtype='int64')
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self._sieve3_entry_populate_indicators(dataframe, metadata)
        dataframe['sieve3_exit_target_level'] = self._sieve3_forward_target(
            dataframe,
            ('s2vp_hvn_above', 's2vp_lvn_above', 's2vp_prior_poc', 's2vp_prior_vah'),
            SIDE,
        )
        dataframe = add_volume_profile(dataframe, prefix='tqvp')
        dataframe = add_trendline_projection_v2(
            dataframe,
            timeframe=self.timeframe,
            raw_line_output_count=3,
            output_prefix='tqtlv2',
        )
        dataframe['sieve3_tq_row_index'] = np.arange(len(dataframe), dtype='int64')
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = None
        condition = _bool(dataframe, 'ms_bos_to_bull')
        if bool(self.use_state_guard):
            condition &= _num(dataframe, 'ms_state').ge(0)
        condition &= self._common_guards(dataframe)
        condition = apply_sieve2_optional_guards(self, dataframe, condition, getattr(self, 'ENTRY_SIDE', 'long'))
        valid = condition.fillna(False) & dataframe['volume'].gt(0.0) & dataframe['close'].notna()
        dataframe.loc[valid, 'enter_long'] = 1
        dataframe.loc[valid, 'enter_tag'] = ENTRY_TAG
        return dataframe
    SOURCE_ENTRY_STEM = 'overtrade_bos_bull_continuation_long_1h_vp_market_guard'
    FOCUSED_EXIT_CONTRACT = 'target_quality_stage2_exact_confluence'
    FOCUSED_SOURCE_PROFILE = {'side': 'long', 'min_level_distance': 0.001, 'targets': {'target_1': {'provider': 'actual-fill-relative exact family composition'}}}
    FOCUSED_EXIT_PLANS = {'touch_full': {'contract': 'target_quality_stage2_exact_confluence', 'name': 'touch_full', 'hard_stop_ratio': 0.03, 'max_hold_candles': 336}, 'close_full': {'contract': 'target_quality_stage2_exact_confluence', 'name': 'close_full', 'hard_stop_ratio': 0.03, 'max_hold_candles': 336}}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'high', 'low', 'open', 'tqvp_poc', 'tqvp_vah', 'tqvp_val', 'tqvp_prior_poc', 'tqvp_prior_vah', 'tqvp_prior_val', 'tqvp_hvn_above', 'tqvp_hvn_below', 'tqvp_lvn_above', 'tqvp_lvn_below', 'tqvp_hvn_above_strength', 'tqvp_hvn_below_strength', 'tqvp_lvn_above_thinness', 'tqvp_lvn_below_thinness', 'tqtlv2_support_line_rank0', 'tqtlv2_support_score_rank0', 'tqtlv2_support_pivot_count_rank0', 'tqtlv2_support_last_confirm_index_rank0', 'tqtlv2_resistance_line_rank0', 'tqtlv2_resistance_score_rank0', 'tqtlv2_resistance_pivot_count_rank0', 'tqtlv2_resistance_last_confirm_index_rank0', 'tqtlv2_support_line_rank1', 'tqtlv2_support_score_rank1', 'tqtlv2_support_pivot_count_rank1', 'tqtlv2_support_last_confirm_index_rank1', 'tqtlv2_resistance_line_rank1', 'tqtlv2_resistance_score_rank1', 'tqtlv2_resistance_pivot_count_rank1', 'tqtlv2_resistance_last_confirm_index_rank1', 'tqtlv2_support_line_rank2', 'tqtlv2_support_score_rank2', 'tqtlv2_support_pivot_count_rank2', 'tqtlv2_support_last_confirm_index_rank2', 'tqtlv2_resistance_line_rank2', 'tqtlv2_resistance_score_rank2', 'tqtlv2_resistance_pivot_count_rank2', 'tqtlv2_resistance_last_confirm_index_rank2', 'sieve3_tq_row_index') + STAGE2_INFORMATIVE_REQUIRED_COLUMNS
    FOCUSED_STATE_VERSION = 1
    FOCUSED_STATE_KEY = 'sieve3_v2_tq:TqStage2ExactConfluenceFromOvertradeBosBullContinuationLong1HVpMarketGuard:target_quality_stage2_exact_confluence'
    LOCKED_BUY_PARAMS = {'use_sieve2_vp_guard': True, 'sieve2_vp_guard_mode': 'score_or_context', 'sieve2_vp_window': 96, 'sieve2_vp_bins': 36, 'sieve2_vp_score_min': 0.25, 'sieve2_vp_context_min': 0.28, 'use_sieve2_market_guard': True, 'sieve2_market_guard_mode': 'pressure_or_trend', 'sieve2_market_window': 24, 'sieve2_market_pressure_min': 0.07, 'sieve2_market_trend_min': 0.25, 'sieve2_rs_benchmark_pair': 'BTC/USDT:USDT', 'sieve2_rs_score_min': 0.45, 'use_volume_guard': False, 'volume_guard_window': 48, 'volume_ratio_min': 1.3, 'use_pressure_guard': True, 'pressure_window': 12, 'pressure_min': 0.2, 'use_accumulation_guard': False, 'use_body_direction_guard': False, 'use_close_direction_guard': False, 'strength': 3, 'min_prominence_atr': 0.35, 'min_pivot_spacing_bars': 2, 'max_pivot_age_bars': 96, 'breakout_buffer_atr': 0.3, 'use_state_guard': True}
    ACTIVE_SELL_PARAMS = ('confluence_composition', 'confluence_timeframe', 'confluence_band_quarter_percent', 'target_action', 'target_band_quarter_percent')
    position_adjustment_enable = False
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    minimal_roi = {'0': 100.0}
    stoploss = -0.99
    confluence_composition = CategoricalParameter(
        tuple(STAGE2_EXACT_COMPOSITIONS),
        default='va_hvn_prior',
        space='sell',
        optimize=True,
        load=True,
    )
    confluence_composition.batch_tags = ('family:exits', 'mode:target_context')
    confluence_timeframe = CategoricalParameter(
        STAGE2_CONFLUENCE_TIMEFRAMES,
        default='1h',
        space='sell',
        optimize=True,
        load=True,
    )
    confluence_timeframe.batch_tags = ('family:exits', 'mode:target_context')
    confluence_band_quarter_percent = IntParameter(
        1, 12, default=4, space='sell', optimize=True, load=True
    )
    confluence_band_quarter_percent.batch_tags = ('family:exits', 'mode:target_context')
    target_action = CategoricalParameter(
        ('touch_full', 'close_full'),
        default='close_full',
        space='sell',
        optimize=True,
        load=True,
    )
    target_action.batch_tags = ('family:exits', 'mode:target_context')
    target_band_quarter_percent = IntParameter(
        0, 12, default=2, space='sell', optimize=True, load=True
    )
    target_band_quarter_percent.batch_tags = ('family:exits', 'mode:target_context')

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
        name = str((state or {}).get('plan') or self.target_action.value)
        if name not in self.FOCUSED_EXIT_PLANS:
            raise ValueError(f'unknown focused exit plan: {name}')
        plan = dict(self.FOCUSED_EXIT_PLANS[name])
        plan['target_confirmation'] = {
            'touch_full': 'touch',
            'close_full': 'close',
        }[name]
        plan['target_band'] = float(self.target_band_quarter_percent.value) * 0.0025
        plan['invalidation_band'] = 0.0025
        plan['invalidation_confirmations'] = 1
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

    def _stage2_column(self, column: str, timeframe: str) -> str:
        return column if timeframe == self.timeframe else f'mtf_{column}_{timeframe}'

    def _stage2_candidate(
        self,
        row: Series,
        name: str,
        level_column: str,
        timeframe: str,
        family: str,
        metadata_columns: Mapping[str, str] | None = None,
    ) -> dict[str, Any] | None:
        level = self._focused_float(row[self._stage2_column(level_column, timeframe)])
        if level is None or level <= 0.0:
            return None
        metadata: dict[str, Any] = {
            'family': family,
            'timeframe': timeframe,
        }
        for label, column in (metadata_columns or {}).items():
            value = self._focused_float(row[self._stage2_column(column, timeframe)])
            if value is not None:
                metadata[str(label)] = value
        if 'last_confirm_index' in metadata:
            row_index = self._focused_float(
                row[self._stage2_column('sieve3_tq_row_index', timeframe)]
            )
            if row_index is not None:
                metadata['age_bars'] = max(
                    0.0,
                    row_index - float(metadata['last_confirm_index']),
                )
        return {
            'name': name,
            'level': float(level),
            'family': family,
            'timeframe': timeframe,
            'metadata': metadata,
        }

    def _stage2_forward_candidates(
        self,
        row: Series,
        candidates: list[dict[str, Any]],
        side: str,
        entry_rate: float,
    ) -> list[dict[str, Any]]:
        minimum = float(self.FOCUSED_SOURCE_PROFILE['min_level_distance'])
        if side == 'short':
            valid = [
                candidate
                for candidate in candidates
                if float(candidate['level']) < entry_rate * (1.0 - minimum)
            ]
        else:
            valid = [
                candidate
                for candidate in candidates
                if float(candidate['level']) > entry_rate * (1.0 + minimum)
            ]
        invalidation_binding = self._focused_side_binding('invalidation', side)
        invalidation_level = self._focused_frozen_level(
            row,
            invalidation_binding,
            side,
            entry_rate,
            'invalidation',
        )
        if invalidation_level is not None:
            valid = [
                candidate
                for candidate in valid
                if not math.isclose(
                    float(candidate['level']),
                    invalidation_level,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ]
        return sorted(
            valid,
            key=lambda candidate: abs(float(candidate['level']) - entry_rate),
        )

    def _stage2_primitive_candidates(
        self,
        row: Series,
        timeframe: str,
        side: str,
    ) -> list[dict[str, Any]]:
        va_column = 'tqvp_val' if side == 'short' else 'tqvp_vah'
        hvn_column = 'tqvp_hvn_below' if side == 'short' else 'tqvp_hvn_above'
        hvn_strength_column = f'{hvn_column}_strength'
        line_side = 'support' if side == 'short' else 'resistance'
        candidates: list[dict[str, Any]] = []
        for name, column, family, metadata in (
            (f'va_edge_{timeframe}', va_column, 'va_edge', {}),
            (f'prior_poc_{timeframe}', 'tqvp_prior_poc', 'prior_poc', {}),
            (
                f'hvn_strength_{timeframe}',
                hvn_column,
                'hvn_strength',
                {'strength': hvn_strength_column},
            ),
        ):
            candidate = self._stage2_candidate(
                row,
                name,
                column,
                timeframe,
                family,
                metadata,
            )
            if candidate is not None:
                candidates.append(candidate)
        candidates = [
            candidate
            for candidate in candidates
            if candidate['family'] != 'hvn_strength'
            or float(candidate['metadata'].get('strength', 0.0))
            >= STAGE2_HVN_MIN_STRENGTH
        ]
        for rank in range(3):
            candidate = self._stage2_candidate(
                row,
                f'tlv2_{line_side}_rank{rank}_{timeframe}',
                f'tqtlv2_{line_side}_line_rank{rank}',
                timeframe,
                'tlv2_quality',
                {
                    'score': f'tqtlv2_{line_side}_score_rank{rank}',
                    'pivot_count': f'tqtlv2_{line_side}_pivot_count_rank{rank}',
                    'last_confirm_index': (
                        f'tqtlv2_{line_side}_last_confirm_index_rank{rank}'
                    ),
                },
            )
            if candidate is None:
                continue
            metadata = candidate['metadata']
            if (
                float(metadata.get('score', 0.0)) >= STAGE2_TLV2_MIN_SCORE
                and int(metadata.get('pivot_count', 0)) >= STAGE2_TLV2_MIN_PIVOTS
                and float(metadata.get('age_bars', float('inf')))
                <= STAGE2_TLV2_MAX_AGE_BARS
            ):
                candidates.append(candidate)
        return candidates

    def _stage2_exact_confluence_candidate(
        self,
        row: Series,
        timeframe: str,
        side: str,
        entry_rate: float,
        composition_name: str,
    ) -> dict[str, Any] | None:
        required_families = STAGE2_EXACT_COMPOSITIONS[composition_name]
        tolerance = float(self.confluence_band_quarter_percent.value) * 0.0025
        candidates = self._stage2_forward_candidates(
            row,
            self._stage2_primitive_candidates(row, timeframe, side),
            side,
            entry_rate,
        )
        anchors = [
            candidate
            for candidate in candidates
            if candidate['family'] in required_families
        ]
        for anchor in anchors:
            anchor_level = float(anchor['level'])
            clustered = [
                candidate
                for candidate in candidates
                if candidate['family'] in required_families
                and abs(float(candidate['level']) / anchor_level - 1.0) <= tolerance
            ]
            distinct: dict[str, dict[str, Any]] = {}
            for candidate in clustered:
                family = str(candidate['family'])
                previous = distinct.get(family)
                if previous is None or abs(float(candidate['level']) - anchor_level) < abs(
                    float(previous['level']) - anchor_level
                ):
                    distinct[family] = candidate
            if any(family not in distinct for family in required_families):
                continue
            members = [distinct[family] for family in required_families]
            levels = sorted(float(candidate['level']) for candidate in members)
            center = float(levels[len(levels) // 2])
            return {
                'name': f'{composition_name}_{timeframe}',
                'level': center,
                'family': 'exact_confluence',
                'timeframe': timeframe,
                'metadata': {
                    'composition': composition_name,
                    'timeframe': timeframe,
                    'tolerance': tolerance,
                    'members': [
                        {
                            'candidate': str(candidate['name']),
                            'level': float(candidate['level']),
                            'metadata': dict(candidate['metadata']),
                        }
                        for candidate in members
                    ],
                },
            }
        return None

    def _stage2_select_target(
        self,
        row: Series,
        side: str,
        entry_rate: float,
    ) -> tuple[float | None, dict[str, Any]]:
        timeframe = str(self.confluence_timeframe.value)
        composition = str(self.confluence_composition.value)
        selected = self._stage2_exact_confluence_candidate(
            row,
            timeframe,
            side,
            entry_rate,
            composition,
        )
        if selected is None:
            return None, {
                'variant': 'exact_confluence',
                'status': 'unavailable',
                'composition': composition,
                'target_timeframe': timeframe,
                'tolerance': float(self.confluence_band_quarter_percent.value) * 0.0025,
            }
        return float(selected['level']), {
            'variant': 'exact_confluence',
            'status': 'selected',
            'composition': composition,
            'target_timeframe': timeframe,
            'candidate': str(selected['name']),
            'metadata': dict(selected['metadata']),
        }

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
        target_level, target_metadata = self._stage2_select_target(
            row,
            side,
            entry_rate,
        )
        levels: dict[str, float | None] = {'target_1': target_level}
        invalidation_binding = self._focused_side_binding(
            'invalidation',
            side,
        )
        levels['invalidation'] = self._focused_frozen_level(
            row,
            invalidation_binding,
            side,
            entry_rate,
            'invalidation',
        )
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
        selected_plan = str(self.target_action.value)
        fill_time = (getattr(order, 'order_filled_utc', None) if order is not None else None) or getattr(trade, 'date_entry_fill_utc', None) or getattr(trade, 'open_date_utc', None) or current_time
        fill_timestamp = self._focused_utc(fill_time)
        snapshot_date = self._focused_utc(row['date'])
        return {'version': self.FOCUSED_STATE_VERSION, 'contract': self.FOCUSED_EXIT_CONTRACT, 'plan': selected_plan, 'side': side, 'phase': 'pre_target', 'entry_rate': float(entry_rate), 'entry_filled_at': fill_timestamp.isoformat() if fill_timestamp is not None else None, 'entry_snapshot_candle': snapshot_date.isoformat() if snapshot_date is not None else None, 'levels': levels, 'target_metadata': target_metadata, 'target_states': {slot: {'status': 'available' if value is not None else 'unavailable', 'touched_at': None} for slot, value in levels.items() if slot.startswith('target_')}, 'invalidation_seen': False, 'guard_state': 'unknown', 'trigger_state': 'unknown', 'initial_stake': self._focused_float(getattr(trade, 'stake_amount', None)), 'favorable_rate': float(entry_rate), 'stop_price': None, 'terminal_pending': False, 'terminal_tag': None}

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
        fill_cursor = filled_at - pd.Timedelta(minutes=timeframe_to_minutes(self.timeframe))
        fill_start = int(frame['date'].searchsorted(fill_cursor, side='right'))
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
        adverse = self._focused_float(getattr(trade, 'max_rate' if side == 'short' else 'min_rate', None)) or entry_rate
        favorable_move = max(0.0, 1.0 - favorable / entry_rate) if side == 'short' else max(0.0, favorable / entry_rate - 1.0)

        levels = state.get('levels', {})
        target = self._focused_float(levels.get('target_1')) if isinstance(levels, Mapping) else None
        invalidation = self._focused_float(levels.get('invalidation')) if isinstance(levels, Mapping) else None
        target_state = (state.get('target_states') or {}).get('target_1', {})
        target_status = str(target_state.get('status') or 'unavailable') if isinstance(target_state, Mapping) else 'unavailable'
        target_done = target_status in {'accepted', 'rejected', 'confirmed'}
        stage_1 = stages.get('stage_1', {}) if isinstance(stages, Mapping) else {}
        partial_done = isinstance(stage_1, Mapping) and stage_1.get('status') == 'filled'

        target_near = False
        if target is not None and not target_done and not partial_done:
            threshold = target * (1.0 + float(plan.get('target_band') or 0.0)) if side == 'short' else target * (1.0 - float(plan.get('target_band') or 0.0))
            target_near = target_status == 'touched' or (favorable <= threshold if side == 'short' else favorable >= threshold)

        invalidation_near = False
        if invalidation is not None and not state.get('invalidation_seen'):
            threshold = invalidation * (1.0 + float(plan.get('invalidation_band') or 0.0)) if side == 'short' else invalidation * (1.0 - float(plan.get('invalidation_band') or 0.0))
            invalidation_near = int(state.get('invalidation_streak') or 0) > 0 or (adverse >= threshold if side == 'short' else adverse <= threshold)

        if target_near or invalidation_near:
            frame = self._focused_analyzed_frame(pair, current_time)
            self._focused_require_columns(frame)
            post_entry = self._focused_post_entry(frame, state)
            events = self._focused_contract_events(post_entry, state, plan, current_profit)
            if not post_entry.empty:
                processed = self._focused_utc(post_entry['date'].iat[-1])
                state['last_processed_candle'] = processed.isoformat() if processed is not None else None
        else:
            events = {'invalidation': bool(state.get('invalidation_seen')), 'target_1': target_done}

        events['age_candles'] = age_candles
        events['favorable_move'] = favorable_move
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
        state = self._focused_state(trade)
        if state is not None:
            target_state = (state.get('target_states') or {}).get('target_1', {})
            target_done = isinstance(target_state, Mapping) and target_state.get('status') in {'accepted', 'rejected', 'confirmed'}
            if state.get('invalidation_seen') or target_done or state.get('terminal_pending'):
                return None
        if (is_short and stop_price <= current_rate) or (not is_short and stop_price >= current_rate):
            return None
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=is_short, leverage=float(getattr(trade, 'leverage', 1.0) or 1.0))

    def _focused_target_event(self, post_entry: DataFrame, state: dict[str, Any], plan: Mapping[str, Any], slot: str) -> bool:
        target = self._focused_float((state.get('levels') or {}).get(slot))
        target_states = state.setdefault('target_states', {})
        target_state = target_states.setdefault(slot, {'status': 'unavailable', 'touched_at': None})
        if target_state.get('status') in {'accepted', 'rejected', 'confirmed'}:
            return True
        if target is None:
            target_state['status'] = 'unavailable'
            return False
        if post_entry.empty:
            return False
        band = float(plan.get('target_band') or 0.0)
        close_times = self._focused_close_times(post_entry)
        confirmation = str(plan.get('target_confirmation') or 'touch')
        touched_at = self._focused_utc(target_state.get('touched_at'))
        if not confirmation.startswith('reversal_'):
            if confirmation not in {'touch', 'close'}:
                raise ValueError(f'unsupported target confirmation: {confirmation}')
            confirmations = 0
        else:
            confirmations = int(confirmation.rsplit('_', 1)[1])
        streak = int(target_state.get('reversal_streak') or 0)
        for position in range(len(post_entry)):
            row = post_entry.iloc[position]
            candle_close = self._focused_utc(close_times.iloc[position])
            low = self._focused_float(row['low'])
            high = self._focused_float(row['high'])
            touched = low is not None and low <= target * (1.0 + band) if state['side'] == 'short' else high is not None and high >= target * (1.0 - band)
            if touched_at is None and touched:
                touched_at = candle_close
                target_state['touched_at'] = touched_at.isoformat() if touched_at is not None else None
                target_state['status'] = 'touched'
                state['phase'] = f'{slot}_zone'
                if confirmation == 'touch':
                    target_state['status'] = 'confirmed'
                    return True
            if touched_at is None or candle_close is None:
                continue
            close = self._focused_float(row['close'])
            if confirmation == 'close':
                accepted = close is not None and (close <= target * (1.0 + band) if state['side'] == 'short' else close >= target * (1.0 - band))
                if candle_close >= touched_at and accepted:
                    target_state['status'] = 'accepted'
                    return True
                continue
            if candle_close <= touched_at:
                continue
            open_ = self._focused_float(row['open'])
            reversal = close is not None and open_ is not None and (close > open_ and close > target * (1.0 - band) if state['side'] == 'short' else close < open_ and close < target * (1.0 + band))
            streak = streak + 1 if reversal else 0
            target_state['reversal_streak'] = streak
            if streak >= confirmations:
                target_state['status'] = 'rejected'
                return True
        if touched_at is None:
            target_state['status'] = 'available'
        return False

    def _focused_invalidation_event(self, post_entry: DataFrame, state: dict[str, Any], plan: Mapping[str, Any]) -> bool:
        if state.get('invalidation_seen') or post_entry.empty:
            return bool(state.get('invalidation_seen'))
        level = self._focused_float((state.get('levels') or {}).get('invalidation'))
        if level is None:
            return False
        close = self._focused_number(post_entry, 'close')
        band = float(plan['invalidation_band'])
        breached = close.gt(level * (1.0 + band)) if state['side'] == 'short' else close.lt(level * (1.0 - band))
        streak = int(state.get('invalidation_streak') or 0)
        confirmations = int(plan['invalidation_confirmations'])
        for value in breached.fillna(False).tolist():
            streak = streak + 1 if bool(value) else 0
            state['invalidation_streak'] = streak
            if streak >= confirmations:
                state['invalidation_seen'] = True
                state['phase'] = 'invalidated'
                return True
        return False

    def _focused_contract_events(self, post_entry: DataFrame, state: dict[str, Any], plan: Mapping[str, Any], current_profit: float) -> dict[str, Any]:
        _ = current_profit
        return {'invalidation': self._focused_invalidation_event(post_entry, state, plan), 'target_1': self._focused_target_event(post_entry, state, plan, 'target_1')}

    def _focused_contract_decision(self, state: dict[str, Any], plan: Mapping[str, Any], events: Mapping[str, Any], current_profit: float) -> dict[str, Any]:
        _ = (state, plan, current_profit)
        if events.get('invalidation'):
            return self._focused_decision('full', 'focused_source_invalidation')
        if events.get('target_1'):
            return self._focused_decision('full', 'focused_entry_target')
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

from __future__ import annotations
from freqtrade.strategy import informative
import math
import pandas as pd
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from typing import Any
from pandas import DataFrame, Series
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, stoploss_from_absolute, IntParameter
LOCKED_BUY_SOURCE = 'verified migration-ledger effective buy lock'
SOURCE_ENTRY_SIGNATURE_SHA256 = '2ac69702c25ba148ca30e81c7c0a461fad7807160d75dfd7db4bd0905576e1f2'
SECONDARY_GUARD = 'H4 VP-or-pressure context and the locked optional volume guard'
import numpy as np
from freqtrade.strategy import IStrategy
from user_data.Indicators.market_state import add_market_state
from user_data.Indicators.complex_volume_profile import add_volume_profile
from user_data.Indicators.pattern_bos_choch import add_bos_choch
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
            float(_sieve2_param(strategy, "sieve2_rs_score_min", 0.45)),
        )
    return guarded.fillna(False)


def _sieve2_vp_guard(
    frame: pd.DataFrame, side: str, mode: str, score_min: float, context_min: float
) -> pd.Series:
    close = _sieve2_num(frame, "close")
    high = _sieve2_num(frame, "high")
    low = _sieve2_num(frame, "low")
    score = _sieve2_num(frame, f"s2vp_score_{side}")
    other = _sieve2_num(frame, "s2vp_score_short" if side == "long" else "s2vp_score_long")
    context = _sieve2_num(
        frame, "s2vp_context_score_bull" if side == "long" else "s2vp_context_score_bear"
    )
    in_value = _sieve2_bool(frame, "s2vp_in_value_area")
    above_value = _sieve2_bool(frame, "s2vp_above_value_area")
    below_value = _sieve2_bool(frame, "s2vp_below_value_area")
    node_entry = _sieve2_bool(frame, f"s2vp_node_entry_{side}")
    node_hold = _sieve2_bool(frame, f"s2vp_node_hold_{side}")
    prior_vah = _sieve2_num(frame, "s2vp_prior_vah")
    prior_val = _sieve2_num(frame, "s2vp_prior_val")
    base_score = score.ge(score_min) & score.gt(other)
    base_context = context.ge(context_min)
    if mode == "node_confirm":
        return (node_entry | node_hold | (base_score & base_context)).fillna(False)
    if mode == "value_area_confirm":
        value_side = above_value if side == "long" else below_value
        return (value_side | (in_value & base_context)).fillna(False)
    if mode == "breakout_acceptance":
        accepted_beyond_value = close.gt(prior_vah) if side == "long" else close.lt(prior_val)
        return (accepted_beyond_value & (base_score | base_context | node_entry | node_hold)).fillna(False)
    if mode == "rejection_confirm":
        rejection = (
            (low.le(prior_val) & close.gt(prior_val) & close.le(prior_vah))
            if side == "long"
            else (high.ge(prior_vah) & close.lt(prior_vah) & close.ge(prior_val))
        )
        return (rejection & (base_score | base_context | node_entry | node_hold)).fillna(False)
    if mode == "poc_hvn_reject":
        return (node_entry | (base_score & base_context)).fillna(False)
    if mode == "prior_level_confirm":
        level_confirm = close.gt(prior_vah) if side == "long" else close.lt(prior_val)
        return (level_confirm & (base_score | base_context)).fillna(False)
    return (base_score | base_context).fillna(False)


def _sieve2_market_guard(
    frame: pd.DataFrame,
    side: str,
    mode: str,
    pressure_min: float,
    trend_min: float,
    rs_score_min: float,
) -> pd.Series:
    pressure = _sieve2_num(frame, "s2m_pressure_ratio")
    trend = _sieve2_num(frame, "s2m_trend_z")
    directional_pressure = pressure.ge(pressure_min) if side == "long" else pressure.le(-pressure_min)
    trend_state = trend.ge(trend_min) if side == "long" else trend.le(-trend_min)
    avoids_adverse_pressure = pressure.ge(-pressure_min) if side == "long" else pressure.le(pressure_min)
    avoids_adverse_trend = trend.ge(-trend_min) if side == "long" else trend.le(trend_min)
    if mode == "pressure_and_trend":
        return (directional_pressure & trend_state).fillna(False)
    if mode == "directional_pressure":
        return directional_pressure.fillna(False)
    if mode == "trend_state":
        return trend_state.fillna(False)
    if mode == "avoid_adverse_pressure":
        return (avoids_adverse_pressure & avoids_adverse_trend).fillna(False)
    if mode == "avoid_chop":
        return (pressure.abs().ge(pressure_min) | trend.abs().ge(trend_min)).fillna(False)
    return (directional_pressure | trend_state).fillna(False)


def _sieve2_param(strategy: Any, name: str, default: Any) -> Any:
    value = getattr(strategy, name, default)
    return getattr(value, "value", value)


def _sieve2_enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _sieve2_num(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _sieve2_bool(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)
ENTRY_MODE = 'entry_mtf_confluence_d1_resistance_reject_4h_short'
ENTRY_TAG = 'mtf_confluence_d1_resistance_reject_4h_short'
ENTRY_SOURCE_STAGE = "sieve2"
UPDATE_HYPOTHESIS = 'D1 resistance/VP rejection context with 4H structural execution. Uses flexible count-based gates so hyperopt can test two-gate and three-gate confluence rather than one rigid chain.'
SIDE = 'short'
CONTEXT_TIMEFRAME = '1d'
TIMEFRAME = '4h'


def _num(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    return pd.Series(frame[column], index=frame.index).astype('boolean').fillna(False).astype(bool)



def _cross_below(value: Series, level: Series) -> Series:
    return value.lt(level) & value.shift(1).ge(level.shift(1))

def _date_naive(value: Series) -> Series:
    return pd.to_datetime(value, utc=True, errors='coerce').dt.tz_convert(None)
SIEVE_STAGE = "sieve3"
SOURCE_ENTRY_STAGE = 'sieve2'
SOURCE_ENTRY_CLASS = 'Sieve3ExitFixedTpSlFromMtfConfluenceD1ResistanceReject4HShort'
SOURCE_STRATEGY = 'user_data/strategies/sieve3_exit_fixed_tp_sl_from_mtf_confluence_d1_resistance_reject_4h_short.py:Sieve3ExitFixedTpSlFromMtfConfluenceD1ResistanceReject4HShort'
LINEAGE_STATUS = 'verified'
EXIT_FAMILY = 'va_edge_reaction_base'
EXIT_THEORY = 'va_edge_reaction_base'
EXIT_HYPOTHESIS = 'Isolate va_edge on the base timeframe scope and exit only after directional weakening confirms around that named level zone.'
PRIMARY_TRIGGER = 'H4 local break-or-failed-retest with ltfms_bos_to_bear/bear state in the locked count-gate closure'
PRIMARY_GUARD = 'D1 prior resistance/rejection gate with D1 BOS/VP and locked optional VP/market guards'
TARGET_PROVIDER = 'closed-candle actual-fill-relative deterministic named level zone'
INVALIDATION_PROVIDER = 'coherent source invalidation when available, otherwise preserved 3% hard stop'
ACTIVE_SELL_PARAMS = ('zone_width_quarter_percent', 'weakening_signal', 'confirmation_profile')

RESEARCH_PATH = 'sieve3_exit_final_generic_levels'
SOURCE_RESULT_BATCH = "historical_promotion_lineage_unknown"

class Sieve3V2VaEdgeReactionBaseFromMtfConfluenceD1ResistanceReject4HShort(IStrategy):
    SIEVE2_FUNDAMENTAL_REWORK = '20260605_nonprofitable_tighten_overtrade'
    REWORK_HYPOTHESIS = 'Entry-only rerun after non-profitable revised-guard result: tighten_overtrade; widen guard thresholds and adjust entry-strength search space.'
    SIEVE2_ALWAYS_ON_GUARDS = True
    'Flexible MTF confluence probe: HTF confidence, LTF execution.'
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 520
    process_only_new_candles = True
    can_short = True
    max_entry_position_adjustment = 0
    # Inactive buy declaration: use_sieve2_vp_guard = BooleanParameter(default=True, space="buy"); fixed-on entry gate makes its guard unconditional.
    # Inactive buy declaration: sieve2_vp_guard_mode = CategoricalParameter(default='score_or_context', space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: sieve2_vp_window = CategoricalParameter(default=96, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: sieve2_vp_bins = CategoricalParameter(default=36, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: sieve2_vp_score_min = CategoricalParameter(default=0.15, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: sieve2_vp_context_min = CategoricalParameter(default=0.15, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: use_sieve2_market_guard = BooleanParameter(default=True, space="buy"); fixed-on entry gate makes its guard unconditional.
    # Inactive buy declaration: sieve2_market_guard_mode = CategoricalParameter(default='pressure_or_trend', space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: sieve2_market_window = CategoricalParameter(default=24, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: sieve2_market_pressure_min = CategoricalParameter(default=0.03, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: sieve2_market_trend_min = CategoricalParameter(default=0.15, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: sieve2_rs_benchmark_pair = CategoricalParameter(default='BTC/USDT:USDT', space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: sieve2_rs_score_min = CategoricalParameter(default=0.3, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    min_htf_gates = 3
    min_ltf_gates = 2
    min_total_gates = 4
    htf_lookback = 72
    ltf_lookback = 48
    recent_htf_bars = 3
    level_buffer_pct = 0.002
    retest_buffer_pct = 0.012
    vp_score_min = 0.35
    vp_context_min = 0.28
    ltf_volume_ratio_min = 1.5
    ltf_pressure_min = 0.18
    # Inactive buy declaration: use_ltf_volume_guard = BooleanParameter(default=False, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: use_ltf_pressure_guard = BooleanParameter(default=True, space="buy"); fixed-on entry gate makes its guard unconditional.

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = (pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs)
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        if not getattr(self, 'dp', None):
            return []
        return [(pair, CONTEXT_TIMEFRAME) for pair in self.dp.current_whitelist()]

    def _sieve3_entry_populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self._add_ltf_features(dataframe)
        dataframe = self._merge_htf_context(dataframe, metadata or {})
        dataframe = add_sieve2_guard_indicators(self, dataframe, metadata)
        return dataframe

    def _add_ltf_features(self, dataframe: DataFrame) -> DataFrame:
        frame = dataframe.copy()
        lookback = int(self.ltf_lookback)
        close = _num(frame, 'close')
        open_ = _num(frame, 'open')
        high = _num(frame, 'high')
        low = _num(frame, 'low')
        volume = _num(frame, 'volume').clip(lower=0.0).fillna(0.0)
        candle_range = (high - low).replace(0.0, np.nan)
        body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
        close_location = ((close - low) / candle_range * 2.0 - 1.0).clip(-1.0, 1.0)
        pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
        baseline = volume.shift(1).rolling(lookback, min_periods=max(2, lookback // 3)).mean().replace(0.0, np.nan)
        frame['mtf_ltf_pressure'] = pressure
        frame['mtf_ltf_volume_ratio'] = volume / baseline
        frame['mtf_ltf_local_high'] = high.shift(1).rolling(lookback, min_periods=max(2, lookback // 3)).max()
        frame['mtf_ltf_local_low'] = low.shift(1).rolling(lookback, min_periods=max(2, lookback // 3)).min()
        frame = add_volume_profile(frame, window=max(48, min(168, lookback * 4)), bins=48, value_area_pct=0.7, price_source='hlc3', smooth_bins=3, pressure_delta_min=0.05, node_near_pct=0.01, prefix='ltfvp')
        frame = add_bos_choch(frame, strength=3, min_prominence_atr=0.35, min_pivot_spacing_bars=2, max_pivot_age_bars=max(24, min(lookback * 6, 160)), breakout_buffer_atr=0.15, prefix='ltfms')
        return frame

    def _merge_htf_context(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        frame = dataframe.copy()
        frame['mtf_htf_vp_gate'] = False
        frame['mtf_htf_structure_gate'] = False
        frame['mtf_htf_prior_gate'] = False
        if 'date' not in frame.columns or not getattr(self, 'dp', None):
            return frame
        pair = str((metadata or {}).get('pair') or '')
        if not pair:
            return frame
        informative = self.dp.get_pair_dataframe(pair=pair, timeframe=CONTEXT_TIMEFRAME)
        if informative is None or informative.empty or 'date' not in informative.columns:
            return frame
        informative = self._add_htf_features(informative.copy())
        keep = ['date', 'mtf_htf_vp_gate', 'mtf_htf_structure_gate', 'mtf_htf_prior_gate']
        informative = informative[keep].copy().sort_values('date')
        informative['date_merge'] = _date_naive(informative['date']) + pd.to_timedelta(timeframe_to_minutes(CONTEXT_TIMEFRAME), unit='m')
        base = frame.drop(columns=['mtf_htf_vp_gate', 'mtf_htf_structure_gate', 'mtf_htf_prior_gate'], errors='ignore').reset_index().rename(columns={'index': '__row_index'}).sort_values('date')
        base['__date_merge'] = _date_naive(base['date'])
        merged = pd.merge_asof(base, informative.drop(columns=['date']).sort_values('date_merge'), left_on='__date_merge', right_on='date_merge', direction='backward')
        merged = merged.sort_values('__row_index').drop(columns=['__row_index', '__date_merge', 'date_merge'], errors='ignore')
        merged.index = dataframe.index
        for column in ('mtf_htf_vp_gate', 'mtf_htf_structure_gate', 'mtf_htf_prior_gate'):
            merged[column] = pd.Series(merged[column], index=merged.index).astype('boolean').fillna(False).astype(bool)
        return merged

    def _add_htf_features(self, informative: DataFrame) -> DataFrame:
        frame = informative.copy()
        lookback = int(self.htf_lookback)
        buffer = float(self.level_buffer_pct)
        close = _num(frame, 'close')
        _num(frame, 'open')
        high = _num(frame, 'high')
        low = _num(frame, 'low')
        prior_high = high.shift(1).rolling(lookback, min_periods=max(3, lookback // 4)).max()
        prior_low = low.shift(1).rolling(lookback, min_periods=max(3, lookback // 4)).min()
        ema = close.ewm(span=max(3, lookback // 3), min_periods=max(3, lookback // 6), adjust=False).mean()
        frame = add_volume_profile(frame, window=max(24, min(168, lookback)), bins=48, value_area_pct=0.7, price_source='hlc3', smooth_bins=3, pressure_delta_min=0.05, node_near_pct=0.01, prefix='htfvp')
        frame = add_bos_choch(frame, strength=3, min_prominence_atr=0.35, min_pivot_spacing_bars=2, max_pivot_age_bars=max(24, min(lookback * 2, 160)), breakout_buffer_atr=0.15, prefix='htfms')
        score_long = _num(frame, 'htfvp_score_long')
        score_short = _num(frame, 'htfvp_score_short')
        _num(frame, 'htfvp_context_score_bull')
        ctx_short = _num(frame, 'htfvp_context_score_bear')
        _num(frame, 'htfvp_prior_vah')
        prior_val = _num(frame, 'htfvp_prior_val')
        vp_gate = ((score_short.ge(float(self.vp_score_min)) | ctx_short.ge(float(self.vp_context_min)) | close.lt(prior_val)) & score_short.ge(score_long)).fillna(False)
        structure_gate = (_bool(frame, 'htfms_bos_to_bear') | _num(frame, 'htfms_state').le(0.0) & close.lt(ema)).fillna(False)
        prior_gate = (close.lt(prior_low.mul(1.0 - buffer)) | high.gt(prior_high.mul(1.0 + buffer)) & close.lt(prior_high)).fillna(False)
        recent = int(self.recent_htf_bars)
        frame['mtf_htf_vp_gate'] = vp_gate.rolling(recent, min_periods=1).max().fillna(0).astype(bool)
        frame['mtf_htf_structure_gate'] = structure_gate.rolling(recent, min_periods=1).max().fillna(0).astype(bool)
        frame['mtf_htf_prior_gate'] = prior_gate.rolling(recent, min_periods=1).max().fillna(0).astype(bool)
        return frame

    def _gate_count(self, gates: list[Series], index: pd.Index) -> Series:
        total = pd.Series(0, index=index, dtype='int64')
        for gate in gates:
            total = total + pd.Series(gate, index=index).fillna(False).astype(bool).astype('int64')
        return total

    def _ltf_gates(self, dataframe: DataFrame) -> list[Series]:
        close = _num(dataframe, 'close')
        high = _num(dataframe, 'high')
        _num(dataframe, 'low')
        buffer = float(self.level_buffer_pct)
        retest = float(self.retest_buffer_pct)
        _num(dataframe, 'mtf_ltf_local_high')
        local_low = _num(dataframe, 'mtf_ltf_local_low')
        pressure = _num(dataframe, 'mtf_ltf_pressure')
        volume_ratio = _num(dataframe, 'mtf_ltf_volume_ratio')
        local_break_or_retest = _cross_below(close, local_low.mul(1.0 - buffer)) | high.ge(local_low.mul(1.0 - retest)) & close.lt(local_low)
        structure = _bool(dataframe, 'ltfms_bos_to_bear') | _num(dataframe, 'ltfms_state').le(0.0)
        vp_or_pressure = _bool(dataframe, 'ltfvp_node_entry_short') | _bool(dataframe, 'ltfvp_node_hold_short') | _num(dataframe, 'ltfvp_score_short').ge(float(self.vp_score_min)) | pressure.le(-float(self.ltf_pressure_min))
        volume_ratio.ge(float(self.ltf_volume_ratio_min))
        context_gate = vp_or_pressure
        context_gate &= pressure.le(-float(self.ltf_pressure_min)) if SIDE == 'short' else pressure.ge(float(self.ltf_pressure_min))
        return [local_break_or_retest.fillna(False), structure.fillna(False), context_gate.fillna(False)]

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = None
        htf_gates = [_bool(dataframe, 'mtf_htf_vp_gate'), _bool(dataframe, 'mtf_htf_structure_gate'), _bool(dataframe, 'mtf_htf_prior_gate')]
        ltf_gates = self._ltf_gates(dataframe)
        htf_count = self._gate_count(htf_gates, dataframe.index)
        ltf_count = self._gate_count(ltf_gates, dataframe.index)
        total_count = htf_count + ltf_count
        condition = htf_count.ge(int(self.min_htf_gates)) & ltf_count.ge(int(self.min_ltf_gates)) & total_count.ge(int(self.min_total_gates))
        condition = apply_sieve2_optional_guards(self, dataframe, condition, SIDE)
        valid = condition.fillna(False) & dataframe['volume'].gt(0.0) & dataframe['close'].notna()
        dataframe.loc[valid, 'enter_short'] = 1
        dataframe.loc[valid, 'enter_tag'] = ENTRY_TAG
        return dataframe

    def _generic_level_frame(self, dataframe: DataFrame, _timeframe: str) -> DataFrame:
        return add_volume_profile(dataframe, prefix='gsvp')
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        frame = self._sieve3_entry_populate_indicators(dataframe, metadata)
    
        if True:
            frame = self._generic_level_frame(frame, self.timeframe)
        return frame

    SOURCE_ENTRY_STEM = 'mtf_confluence_d1_resistance_reject_4h_short'
    FOCUSED_EXIT_CONTRACT = 'va_edge_reaction_base'
    FOCUSED_SOURCE_PROFILE = {'side': 'short', 'min_level_distance': 0.001}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'high', 'low', 'open')
    FOCUSED_STATE_VERSION = 1
    FOCUSED_STATE_KEY = 'sieve3_v2_generic:Sieve3V2VaEdgeReactionBaseFromMtfConfluenceD1ResistanceReject4HShort:va_edge_reaction_base'
    LOCKED_BUY_PARAMS = {'use_sieve2_vp_guard': True, 'sieve2_vp_guard_mode': 'score_or_context', 'sieve2_vp_window': 96, 'sieve2_vp_bins': 36, 'sieve2_vp_score_min': 0.15, 'sieve2_vp_context_min': 0.15, 'use_sieve2_market_guard': True, 'sieve2_market_guard_mode': 'pressure_or_trend', 'sieve2_market_window': 24, 'sieve2_market_pressure_min': 0.03, 'sieve2_market_trend_min': 0.15, 'sieve2_rs_benchmark_pair': 'BTC/USDT:USDT', 'sieve2_rs_score_min': 0.3, 'min_htf_gates': 3, 'min_ltf_gates': 2, 'min_total_gates': 4, 'htf_lookback': 72, 'ltf_lookback': 48, 'recent_htf_bars': 3, 'level_buffer_pct': 0.002, 'retest_buffer_pct': 0.012, 'vp_score_min': 0.35, 'vp_context_min': 0.28, 'ltf_volume_ratio_min': 1.5, 'ltf_pressure_min': 0.18, 'use_ltf_volume_guard': False, 'use_ltf_pressure_guard': True}
    ACTIVE_SELL_PARAMS = ('zone_width_quarter_percent', 'weakening_signal', 'confirmation_profile')
    GENERIC_MODE = 'simple'
    GENERIC_LEVEL_FAMILY = 'va_edge'
    GENERIC_TIMEFRAME_SCOPE = 'base'
    GENERIC_AVAILABLE_TIMEFRAMES = ('4h', '8h', '1d', '3d')
    GENERIC_HVN_MIN_STRENGTH = 0.5
    GENERIC_MIN_LEVEL_DISTANCE = 0.001
    GENERIC_HARD_STOP_RATIO = 0.03
    position_adjustment_enable = False
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    minimal_roi = {'0': 100.0}
    stoploss = -0.99
    zone_width_quarter_percent = IntParameter(1, 12, default=4, space='sell', optimize=True, load=True)
    zone_width_quarter_percent.batch_tags = ('family:exits', 'mode:sieve3_exit')
    weakening_signal = CategoricalParameter(('opposite_body', 'close_momentum', 'directional_pressure', 'zone_rejection'), default='close_momentum', space='sell', optimize=True, load=True)
    weakening_signal.batch_tags = ('family:exits', 'mode:sieve3_exit')
    confirmation_profile = CategoricalParameter(('1_of_1', '2_of_2', '2_of_3', '3_of_3'), default='2_of_3', space='sell', optimize=True, load=True)
    confirmation_profile.batch_tags = ('family:exits', 'mode:sieve3_exit')

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

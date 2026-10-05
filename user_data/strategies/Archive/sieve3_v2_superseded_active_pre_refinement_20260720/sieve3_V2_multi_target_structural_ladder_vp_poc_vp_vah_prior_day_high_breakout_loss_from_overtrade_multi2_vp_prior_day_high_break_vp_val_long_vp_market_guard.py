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
LOCKED_BUY_SOURCE = 'verified tested-snapshot effective buy lock'
SOURCE_ENTRY_SIGNATURE_SHA256 = '5b539892a1f47466b3a669c2bd64f50819c2a75dcfbf46b9d6edbd1c7e4850af'
SECONDARY_GUARD = 'locked volume/pressure settings and verified Sieve2 VP/market guards'
import numpy as np
from freqtrade.strategy import IStrategy
from user_data.Indicators.market_state import add_market_state
from user_data.Indicators.complex_volume_profile import add_volume_profile
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


def _num(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    return pd.Series(frame[column], index=frame.index).astype('boolean').fillna(False).astype(bool)


def _cross_above(series: Series, level: Series) -> Series:
    return series.gt(level) & series.shift(1).le(level.shift(1))

def _cross_below(series: Series, level: Series) -> Series:
    return series.lt(level) & series.shift(1).ge(level.shift(1))
ENTRY_MODE = 'entry_overtrade_multi2_vp_prior_day_high_break_vp_val_long_vp_market_guard'
ENTRY_TAG = 'overtrade_multi2_vp_prior_day_high_break_vp_val_long_vp_market_guard'
UPDATE_HYPOTHESIS = 'High-trade entry may contain a real directional edge; require stronger volume-profile/market context so HTF or broader context carries confidence and the original trigger supplies execution.'
SIDE = 'long'
TIMEFRAME = '1h'
VP_CONFIRM = 'val_reclaim'
PRIOR_KIND = 'day_high_breakout'
SOURCE_ENTRY_STEM = 'overtrade_multi2_vp_prior_day_high_break_vp_val_long_vp_market_guard'
SIEVE_STAGE = "sieve3"
SOURCE_ENTRY_STAGE = 'sieve2'
SOURCE_ENTRY_CLASS = 'Sieve2OvertradeMulti2VPPriorDayHighBreakVPVALLongVPMarketGuard'
SOURCE_STRATEGY = 'D:\\FreqTradeStuffLargeData\\sieve_runtime\\entry_sieve\\backtests\\sieve2_overtrade_multi2_vp_prior_day_high_break_vp_val_long_vp_market_guard__auto_generic_1h_2020_q2_q3\\full_cycle_2020_2026\\tp_3_sl_3\\backtest-result-2026-06-12_17-34-55.zip!backtest-result-2026-06-12_17-34-55_Sieve2OvertradeMulti2VPPriorDayHighBreakVPVALLongVPMarketGuard.py:Sieve2OvertradeMulti2VPPriorDayHighBreakVPVALLongVPMarketGuard'
LINEAGE_STATUS = 'verified'
EXIT_FAMILY = 'multi_target_structural_ladder'
EXIT_THEORY = 'multi_target_structural_ladder_vp_poc_vp_vah_prior_day_high_breakout_loss'
EXIT_HYPOTHESIS = 'Ordered entry-frozen vp_poc then vp_vah objectives realize two partials while prior_day_high failure remains terminal.'
PRIMARY_TRIGGER = '1h close cross above prior_day_high with the locked breakout buffer'
PRIMARY_GUARD = 'vp_val reclaim with vp_score_long, vp_entry_trigger_long, or balance context'
TARGET_PROVIDER = 'target_1:vp_val_reclaim_to_point_of_control_forward_objective:vp_poc; target_2:vp_value_area_high_second_objective:vp_vah'
INVALIDATION_PROVIDER = 'prior_day_high_breakout_loss:prior_day_high'
ACTIVE_SELL_PARAMS = ('exit_plan', 'hard_stop_percent', 'max_hold_scale')

ENTRY_SOURCE_STAGE = "sieve2"
RESEARCH_PATH = "sieve3_exit_multi_target_structural_ladder_vp_poc_vp_vah_prior_day_high_breakout_loss"
SOURCE_RESULT_BATCH = "20260612T143424_entry_sieve2_top_level_remaining120_cores8_20260612:59"

class Sieve3V2MultiTargetStructuralLadderVpPocVpVahPriorDayHighBreakoutLossFromOvertradeMulti2VpPriorDayHighBreakVpValLongVpMarketGuard(IStrategy):
    """Sieve2 multi2 confluence probe."""
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 220
    process_only_new_candles = True
    can_short = False
    max_entry_position_adjustment = 0
    # Inactive buy declaration: use_sieve2_vp_guard = BooleanParameter(default=True, space="buy"); fixed-on entry gate makes its guard unconditional.
    # Inactive buy declaration: sieve2_vp_guard_mode = CategoricalParameter(default='score_or_context', space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: sieve2_vp_window = CategoricalParameter(default=96, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: sieve2_vp_bins = CategoricalParameter(default=36, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: sieve2_vp_score_min = CategoricalParameter(default=0.25, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: sieve2_vp_context_min = CategoricalParameter(default=0.28, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: use_sieve2_market_guard = CategoricalParameter(default=True, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: sieve2_market_guard_mode = CategoricalParameter(default='pressure_or_trend', space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: sieve2_market_window = CategoricalParameter(default=24, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: sieve2_market_pressure_min = CategoricalParameter(default=0.07, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: sieve2_market_trend_min = CategoricalParameter(default=0.25, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: sieve2_rs_benchmark_pair = CategoricalParameter(default='BTC/USDT:USDT', space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: sieve2_rs_score_min = CategoricalParameter(default=0.45, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: use_volume_guard = BooleanParameter(default=True, space="buy"); fixed-on entry gate makes its guard unconditional.
    volume_guard_window = 24
    volume_ratio_min = 1.3
    # Inactive buy declaration: use_pressure_guard = BooleanParameter(default=True, space="buy"); fixed-on entry gate makes its guard unconditional.
    pressure_window = 12
    pressure_min = 0.05
    # Inactive buy declaration: use_accumulation_guard = BooleanParameter(default=False, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: use_body_direction_guard = BooleanParameter(default=False, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: use_close_direction_guard = BooleanParameter(default=False, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    vp_window = 96
    vp_bins = 48
    vp_value_area_pct = 0.7
    vp_price_source = 'hlc3'
    vp_smooth_bins = 3
    vp_hvn_threshold = 0.7
    vp_lvn_threshold = 0.35
    vp_pressure_delta_min = 0.05
    vp_node_near_pct = 0.01
    vp_volume_percentile_min = 0.55
    vp_score_window = 48
    vp_fast_traverse_atr_mult = 1.2
    vp_entry_score_margin = 0.02
    vp_score_min = 0.25
    vp_context_min = 0.45
    vp_level_buffer_pct = 0.01
    level_buffer_pct = 0.012
    reclaim_buffer_pct = 0.006
    level_lookback = 96
    equal_level_tolerance_pct = 0.003
    equal_level_min_touches = 3

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = (pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs)
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype='bool')
        close = _num(dataframe, 'close')
        volume = _num(dataframe, 'volume').clip(lower=0.0)
        window = int(self.volume_guard_window)
        baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
        guard &= volume.ge(baseline.mul(float(self.volume_ratio_min)))
        open_ = _num(dataframe, 'open')
        high = _num(dataframe, 'high')
        low = _num(dataframe, 'low')
        volume = _num(dataframe, 'volume').clip(lower=0.0).fillna(0.0)
        candle_range = (high - low).replace(0.0, np.nan)
        body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
        close_location = ((close - low) / candle_range * 2.0 - 1.0).clip(-1.0, 1.0)
        pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
        directional_volume = (pressure * volume).fillna(0.0)
        window = int(self.pressure_window)
        pressure_baseline = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
        pressure_ratio = directional_volume.rolling(window, min_periods=max(2, window // 3)).sum() / pressure_baseline
        guard &= pressure_ratio.le(-float(self.pressure_min)) if SIDE == 'short' else pressure_ratio.ge(float(self.pressure_min))
        return guard.fillna(False)

    def _add_vp(self, dataframe: DataFrame) -> DataFrame:
        return add_volume_profile(dataframe, window=int(self.vp_window), bins=int(self.vp_bins), value_area_pct=float(self.vp_value_area_pct), price_source=str(self.vp_price_source), smooth_bins=int(self.vp_smooth_bins), hvn_threshold=float(self.vp_hvn_threshold), lvn_threshold=float(self.vp_lvn_threshold), pressure_delta_min=float(self.vp_pressure_delta_min), node_near_pct=float(self.vp_node_near_pct), volume_percentile_min=float(self.vp_volume_percentile_min), score_window=int(self.vp_score_window), fast_traverse_atr_mult=float(self.vp_fast_traverse_atr_mult), entry_score_margin=float(self.vp_entry_score_margin), prefix='vp')

    def _vp_confirm(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        score_long = _num(dataframe, 'vp_score_long')
        _num(dataframe, 'vp_score_short')
        _num(dataframe, 'vp_context_score_bull')
        _num(dataframe, 'vp_context_score_bear')
        balance = _num(dataframe, 'vp_context_score_balance')
        _num(dataframe, 'vp_market_context')
        score_min = float(self.vp_score_min)
        context_min = float(self.vp_context_min)
        near = float(self.vp_level_buffer_pct)
        val = _num(dataframe, 'vp_val')
        return close.ge(val.mul(1.0 - near)) & (score_long.ge(score_min) | _bool(dataframe, 'vp_entry_trigger_long') | balance.ge(context_min))

    def _date_series(self, dataframe: DataFrame) -> Series:
        if 'date' in dataframe.columns:
            return pd.to_datetime(dataframe['date'], utc=True, errors='coerce')
        return pd.Series(pd.to_datetime(dataframe.index, utc=True, errors='coerce'), index=dataframe.index)

    def _period_key(self, dates: Series, period: str) -> Series:
        if period == 'week':
            iso = dates.dt.isocalendar()
            return iso['year'].astype('string').str.cat(iso['week'].astype('string').str.zfill(2), sep='-')
        if period == 'month':
            return dates.dt.strftime('%Y-%m')
        return dates.dt.strftime('%Y-%m-%d')

    def _add_prior_levels(self, dataframe: DataFrame) -> DataFrame:
        dates = self._date_series(dataframe)
        high = _num(dataframe, 'high')
        low = _num(dataframe, 'low')
        for period in ('day', 'week', 'month'):
            key = self._period_key(dates, period)
            levels = pd.DataFrame({'key': key, 'high': high, 'low': low}, index=dataframe.index).groupby('key', sort=False).agg({'high': 'max', 'low': 'min'})
            levels[f'prior_{period}_high'] = levels['high'].shift(1)
            levels[f'prior_{period}_low'] = levels['low'].shift(1)
            dataframe[f'prior_{period}_high'] = key.map(levels[f'prior_{period}_high'])
            dataframe[f'prior_{period}_low'] = key.map(levels[f'prior_{period}_low'])
        lookback = int(self.level_lookback)
        min_periods = max(8, lookback // 4)
        dataframe['rolling_high'] = high.shift(1).rolling(lookback, min_periods=min_periods).max()
        dataframe['rolling_low'] = low.shift(1).rolling(lookback, min_periods=min_periods).min()
        tol = float(self.equal_level_tolerance_pct)
        dataframe['equal_high_touches'] = high.shift(1).rolling(lookback, min_periods=min_periods).apply(lambda values: float(np.sum(values >= np.nanmax(values) * (1.0 - tol))) if np.isfinite(np.nanmax(values)) else 0.0, raw=True)
        dataframe['equal_low_touches'] = low.shift(1).rolling(lookback, min_periods=min_periods).apply(lambda values: float(np.sum(values <= np.nanmin(values) * (1.0 + tol))) if np.isfinite(np.nanmin(values)) else 0.0, raw=True)
        return dataframe

    def _prior_trigger(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, 'close')
        open_ = _num(dataframe, 'open')
        _num(dataframe, 'high')
        _num(dataframe, 'low')
        buffer = float(self.level_buffer_pct)
        float(self.reclaim_buffer_pct)
        int(self.equal_level_min_touches)
        if PRIOR_KIND.endswith('_high_breakout'):
            period = PRIOR_KIND.split('_', 1)[0]
            level = _num(dataframe, f'prior_{period}_high')
            return _cross_above(close, level.mul(1.0 + buffer)) & close.gt(open_)
        if PRIOR_KIND.endswith('_low_breakdown'):
            period = PRIOR_KIND.split('_', 1)[0]
            level = _num(dataframe, f'prior_{period}_low')
            return _cross_below(close, level.mul(1.0 - buffer)) & close.lt(open_)
        return pd.Series(False, index=dataframe.index, dtype='bool')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe = self._add_prior_levels(dataframe)
        dataframe = self._add_vp(dataframe)
        dataframe = add_sieve2_guard_indicators(self, dataframe, metadata)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = None
        condition = self._prior_trigger(dataframe) & self._vp_confirm(dataframe)
        condition &= self._common_guards(dataframe)
        condition = apply_sieve2_optional_guards(self, dataframe, condition, SIDE)
        valid = condition.fillna(False) & dataframe['volume'].gt(0.0) & dataframe['close'].notna()
        dataframe.loc[valid, 'enter_long'] = 1
        dataframe.loc[valid, 'enter_tag'] = ENTRY_TAG
        return dataframe
    FOCUSED_EXIT_CONTRACT = 'multi_target_structural_ladder'
    FOCUSED_SOURCE_PROFILE = {'side': 'long', 'min_level_distance': 0.001, 'targets': {'target_1': {'provider': 'vp_val_reclaim_to_point_of_control_forward_objective', 'long': {'level': 'vp_poc', 'available': None}}, 'target_2': {'provider': 'vp_value_area_high_second_objective', 'long': {'level': 'vp_vah', 'available': None}}}, 'invalidation': {'provider': 'prior_day_high_breakout_loss', 'mode': 'level', 'long': {'level': 'prior_day_high', 'available': None}}}
    FOCUSED_EXIT_PLANS = {'targets_touch_p25_p25_lock_trail': {'contract': 'multi_target_structural_ladder', 'name': 'targets_touch_p25_p25_lock_trail', 'action_sequence': ('source_invalidation_full', 'target_1_partial_stage_1', 'target_2_partial_stage_2', 'target_1_lock', 'percent_trail_remainder', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'partial_fractions': (0.25, 0.25), 'target_confirmation': 'touch', 'target_band': 0.0025, 'invalidation_confirmations': 1, 'invalidation_band': 0.0025, 'trail_ratio': 0.015}, 'targets_close_p33_p25_lock_trail': {'contract': 'multi_target_structural_ladder', 'name': 'targets_close_p33_p25_lock_trail', 'action_sequence': ('source_invalidation_full', 'target_1_partial_stage_1', 'target_2_partial_stage_2', 'target_1_lock', 'percent_trail_remainder', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'partial_fractions': (0.33, 0.25), 'target_confirmation': 'close', 'target_band': 0.0025, 'invalidation_confirmations': 2, 'invalidation_band': 0.0025, 'trail_ratio': 0.015}, 'targets_reversal_p33_p33_lock_trail': {'contract': 'multi_target_structural_ladder', 'name': 'targets_reversal_p33_p33_lock_trail', 'action_sequence': ('source_invalidation_full', 'target_1_partial_stage_1', 'target_2_partial_stage_2', 'target_1_lock', 'percent_trail_remainder', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336, 'partial_fractions': (0.33, 0.33), 'target_confirmation': 'reversal_1', 'target_band': 0.005, 'invalidation_confirmations': 2, 'invalidation_band': 0.005, 'trail_ratio': 0.015}}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'high', 'low', 'open', 'prior_day_high', 'vp_poc', 'vp_vah')
    FOCUSED_STATE_VERSION = 1
    FOCUSED_STATE_KEY = 'sieve3_v2_focused:Sieve3V2MultiTargetStructuralLadderVpPocVpVahPriorDayHighBreakoutLossFromOvertradeMulti2VpPriorDayHighBreakVpValLongVpMarketGuard:multi_target_structural_ladder'
    LOCKED_BUY_PARAMS = {'use_sieve2_vp_guard': True, 'sieve2_vp_guard_mode': 'score_or_context', 'sieve2_vp_window': 96, 'sieve2_vp_bins': 36, 'sieve2_vp_score_min': 0.25, 'sieve2_vp_context_min': 0.28, 'use_sieve2_market_guard': True, 'sieve2_market_guard_mode': 'pressure_or_trend', 'sieve2_market_window': 24, 'sieve2_market_pressure_min': 0.07, 'sieve2_market_trend_min': 0.25, 'sieve2_rs_benchmark_pair': 'BTC/USDT:USDT', 'sieve2_rs_score_min': 0.45, 'use_volume_guard': True, 'volume_guard_window': 24, 'volume_ratio_min': 1.3, 'use_pressure_guard': True, 'pressure_window': 12, 'pressure_min': 0.05, 'use_accumulation_guard': False, 'use_body_direction_guard': False, 'use_close_direction_guard': False, 'vp_window': 96, 'vp_bins': 48, 'vp_value_area_pct': 0.7, 'vp_price_source': 'hlc3', 'vp_smooth_bins': 3, 'vp_hvn_threshold': 0.7, 'vp_lvn_threshold': 0.35, 'vp_pressure_delta_min': 0.05, 'vp_node_near_pct': 0.01, 'vp_volume_percentile_min': 0.55, 'vp_score_window': 48, 'vp_fast_traverse_atr_mult': 1.2, 'vp_entry_score_margin': 0.02, 'vp_score_min': 0.25, 'vp_context_min': 0.45, 'vp_level_buffer_pct': 0.01, 'level_buffer_pct': 0.012, 'reclaim_buffer_pct': 0.006, 'level_lookback': 96, 'equal_level_tolerance_pct': 0.003, 'equal_level_min_touches': 3}
    ACTIVE_SELL_PARAMS = ('exit_plan', 'hard_stop_percent', 'max_hold_scale')
    position_adjustment_enable = True
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    minimal_roi = {'0': 100.0}
    stoploss = -0.99
    exit_plan = CategoricalParameter(('targets_touch_p25_p25_lock_trail', 'targets_close_p33_p25_lock_trail', 'targets_reversal_p33_p33_lock_trail'), default='targets_touch_p25_p25_lock_trail', space='sell', optimize=True, load=True)
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
        required_targets = ('target_1', 'target_2')
        unusable_targets = [slot for slot in required_targets if levels.get(slot) is None]
        if unusable_targets:
            raise ValueError(
                f"{selected_plan} requires usable entry-frozen targets: {unusable_targets}"
            )
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

    def _focused_post_entry(self, frame: DataFrame, state: Mapping[str, Any]) -> DataFrame:
        filled_at = self._focused_utc(state.get('entry_filled_at'))
        if filled_at is None:
            raise ValueError('focused exit state has no entry fill timestamp')
        candle_open = self._focused_utc(
            timeframe_to_prev_date(self.timeframe, filled_at.to_pydatetime())
        )
        if candle_open is None:
            raise ValueError('focused exit runtime could not resolve the entry candle')
        dates = pd.to_datetime(frame['date'], utc=True, errors='raise')
        eligible = dates.ge(candle_open) if filled_at == candle_open else dates.gt(candle_open)
        return frame.loc[eligible]

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
    def _focused_breakeven_price(state: Mapping[str, Any], offset: float) -> float:
        entry_rate = float(state['entry_rate'])
        return entry_rate * (1.0 - offset if state['side'] == 'short' else 1.0 + offset)

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

    def _focused_target_event(self, post_entry: DataFrame, state: dict[str, Any], plan: Mapping[str, Any], slot: str) -> bool:
        target = self._focused_float((state.get('levels') or {}).get(slot))
        target_states = state.setdefault('target_states', {})
        target_state = target_states.setdefault(slot, {'status': 'unavailable', 'touched_at': None})
        if target is None or post_entry.empty:
            target_state['status'] = 'unavailable'
            return False
        if target_state.get('status') in {'accepted', 'rejected', 'confirmed'}:
            return True
        band = float(plan.get('target_band') or 0.0)
        close_times = self._focused_close_times(post_entry)
        if state['side'] == 'short':
            touched = self._focused_number(post_entry, 'low').le(target * (1.0 + band))
        else:
            touched = self._focused_number(post_entry, 'high').ge(target * (1.0 - band))
        touched_at = self._focused_utc(target_state.get('touched_at'))
        if touched_at is None:
            positions = [index for index, value in enumerate(touched.fillna(False).tolist()) if value]
            if not positions:
                target_state['status'] = 'available'
                return False
            touched_at = self._focused_utc(close_times.iloc[positions[0]])
            target_state['touched_at'] = touched_at.isoformat() if touched_at is not None else None
            target_state['status'] = 'touched'
            state['phase'] = f'{slot}_zone'
        confirmation = str(plan.get('target_confirmation') or 'touch')
        if confirmation == 'touch':
            target_state['status'] = 'confirmed'
            return True
        after_touch = post_entry.loc[close_times.ge(touched_at) if confirmation == 'close' else close_times.gt(touched_at)]
        if after_touch.empty:
            return False
        close = self._focused_number(after_touch, 'close')
        open_ = self._focused_number(after_touch, 'open')
        if confirmation == 'close':
            accepted = close.le(target * (1.0 + band)) if state['side'] == 'short' else close.ge(target * (1.0 - band))
            if bool(accepted.fillna(False).any()):
                target_state['status'] = 'accepted'
                return True
            return False
        reversal = close.gt(open_) & close.gt(target * (1.0 - band)) if state['side'] == 'short' else close.lt(open_) & close.lt(target * (1.0 + band))
        if confirmation == 'reversal_1':
            confirmed = self._focused_last_n(reversal, 1)
        elif confirmation == 'reversal_2_of_3':
            confirmed = len(reversal) >= 3 and int(reversal.tail(3).fillna(False).sum()) >= 2
        else:
            raise ValueError(f'unsupported target confirmation: {confirmation}')
        if confirmed:
            target_state['status'] = 'rejected'
        return confirmed

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
        invalidation = self._focused_invalidation_event(post_entry, state, plan)
        first = self._focused_target_event(post_entry, state, plan, 'target_1')
        second = bool(self._focused_stage_filled(state, 'stage_1')) and self._focused_target_event(post_entry, state, plan, 'target_2')
        return {'invalidation': invalidation, 'target_1': first, 'target_2': second}

    def _focused_contract_decision(self, state: dict[str, Any], plan: Mapping[str, Any], events: Mapping[str, Any], current_profit: float) -> dict[str, Any]:
        _ = (plan, current_profit)
        if events.get('invalidation'):
            return self._focused_decision('full', 'focused_source_invalidation')
        first = state['stages']['stage_1']
        second = state['stages']['stage_2']
        if events.get('target_1') and first['status'] == 'ready':
            return self._focused_decision('partial', str(first['tag']), 'stage_1', float(first['fraction']))
        if events.get('target_2') and first['status'] == 'filled' and (second['status'] == 'ready'):
            return self._focused_decision('partial', str(second['tag']), 'stage_2', float(second['fraction']))
        return self._focused_decision()

    def _focused_stop_overlay(self, plan: Mapping[str, Any], state: Mapping[str, Any], current_profit: float) -> float:
        stop_price = self._focused_hard_stop_price(plan, state)
        if self._focused_stage_filled(state, 'stage_1'):
            candidate = self._focused_breakeven_price(state, 0.0)
            stop_price = self._focused_tighter_stop(str(state['side']), stop_price, candidate)
        if self._focused_stage_filled(state, 'stage_2'):
            target_lock = self._focused_float((state.get('levels') or {}).get('target_1'))
            if target_lock is not None:
                stop_price = self._focused_tighter_stop(str(state['side']), stop_price, target_lock)
            trail = self._focused_trail_price(state, float(plan['trail_ratio']))
            stop_price = self._focused_tighter_stop(str(state['side']), stop_price, trail)
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

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
ENTRY_SOURCE_STAGE = "sieve2"
UPDATE_HYPOTHESIS = 'High-win sparse entry may be too restrictive; broaden structural timing/quality thresholds while preserving the original entry idea.'
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
LEVEL_LOOKBACK_CHOICES = [48, 96, 168, 336]
LOCAL_LOOKBACK_CHOICES = [6, 12, 24, 48]
VOLUME_WINDOW_CHOICES = [12, 24, 48, 72]
D1_LEVEL_LOOKBACK_CHOICES = [20, 50, 100, 200]
D1_VOLUME_WINDOW_CHOICES = [14, 28, 56]
ENABLE_CHOICES = ['off', 'on']
D1_STRUCTURE_MODE_CHOICES = ['off', 'near', 'aligned']
GUARD_MODE_CHOICES = ['direction', 'score', 'context', 'score_or_context', 'balance']
PRICE_SOURCE_CHOICES = ['close', 'hl2', 'hlc3', 'ohlc4']
VP_COLUMNS = ['score_long', 'score_short', 'context_score_bull', 'context_score_bear', 'context_score_balance', 'market_context']

def _tag(param: Any, mode: str) -> Any:
    setattr(param, 'batch_tags', ('family:entries', f'mode:{mode}'))
    return param
SIEVE_STAGE = "sieve3"
SOURCE_ENTRY_STAGE = 'sieve2'
SOURCE_ENTRY_CLASS = 'Sieve3ExitFromLoosenLadderLongSupHoldBroaderTrigger'
SOURCE_STRATEGY = 'user_data/strategies/sieve3_exit_from_loosen_ladder_long_sup_hold_broader_trigger.py:Sieve3ExitFromLoosenLadderLongSupHoldBroaderTrigger'
LINEAGE_STATUS = 'executable_source_verified; historical_parent_path_missing'
EXIT_FAMILY = 'va_edge_reaction_base'
EXIT_THEORY = 'va_edge_reaction_base'
EXIT_HYPOTHESIS = 'Isolate va_edge on the base timeframe scope and exit only after directional weakening confirms around that named level zone.'
PRIMARY_TRIGGER = 'none'
PRIMARY_GUARD = 'none'
TARGET_PROVIDER = 'closed-candle actual-fill-relative deterministic named level zone'
INVALIDATION_PROVIDER = 'coherent source invalidation when available, otherwise preserved 3% hard stop'
ACTIVE_SELL_PARAMS = ('zone_width_quarter_percent', 'weakening_signal', 'confirmation_profile')

RESEARCH_PATH = 'sieve3_exit_final_generic_levels'
SOURCE_RESULT_BATCH = "historical_promotion_lineage_unknown"


class Sieve3V2VaEdgeReactionBaseFromLoosenLadderLongSupHoldBroaderTrigger(IStrategy):
    SIEVE2_FUNDAMENTAL_REWORK = '20260605_nonprofitable_loosen_sparse'
    REWORK_HYPOTHESIS = 'Entry-only rerun after non-profitable revised-guard result: loosen_sparse; widen guard thresholds and adjust entry-strength search space.'
    SIEVE2_ALWAYS_ON_GUARDS = True
    INTERFACE_VERSION = 3
    can_short = False
    timeframe = '1h'
    startup_candle_count = 336
    process_only_new_candles = True
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
    ENTRY_TAG = 'long_sup_hold'
    ENTRY_SIDE = 'long'
    ENTRY_KIND = 'long_sup_hold'
    MODE = 'entry_long_sup_hold'
    CONFIRMATION_PROFILE = 'long_support_hold'
    BREATHING_PROFILE = 'none'
    level_lookback = 168
    local_lookback = 48
    d1_level_lookback = 50
    volume_window = 24
    d1_volume_window = 28
    confirm_bars = 3
    zone_pct = 0.012
    trigger_buffer_pct = 0.003
    # Inactive buy declaration: h1_rvol_enable = CategoricalParameter(default='on', space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: h1_pressure_enable = CategoricalParameter(default='on', space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: h1_event_enable = CategoricalParameter(default='on', space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: d1_rvol_enable = CategoricalParameter(default='on', space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: d1_pressure_enable = CategoricalParameter(default='on', space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    d1_structure_mode = 'off'
    h1_rvol_min = 1.1
    h1_pressure_min = 0.3
    d1_rvol_min = 1.0
    d1_pressure_min = 0.2
    # Inactive buy declaration: use_vp_4h_guard = CategoricalParameter(default='off', space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: vp_4h_guard_mode = CategoricalParameter(default='score_or_context', space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    vp_4h_window = 48
    vp_4h_bins = 36
    # Inactive buy declaration: vp_4h_score_min = DecimalParameter(default=0.25, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: vp_4h_context_min = DecimalParameter(default=0.28, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: use_vp_1d_guard = CategoricalParameter(default='off', space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: vp_1d_guard_mode = CategoricalParameter(default='context', space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    vp_1d_window = 30
    vp_1d_bins = 36
    # Inactive buy declaration: vp_1d_score_min = DecimalParameter(default=0.25, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: vp_1d_context_min = DecimalParameter(default=0.28, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    vp_guard_value_area_pct = 0.7
    vp_guard_price_source = 'hlc3'
    vp_guard_node_near_pct = 0.01
    vp_guard_pressure_delta_min = 0.05

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = (pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs)
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        if not getattr(self, 'dp', None):
            return []
        whitelist = self.dp.current_whitelist()
        return [(pair, '1d') for pair in whitelist] + [(pair, '4h') for pair in whitelist]

    @staticmethod
    def _num(value: Series) -> Series:
        return pd.to_numeric(value, errors='coerce')

    @staticmethod
    def _safe_div(numerator: Series, denominator: Series) -> Series:
        return numerator / denominator.replace(0.0, np.nan)

    @staticmethod
    def _zscore(value: Series, window: int) -> Series:
        mean = value.rolling(int(window), min_periods=max(2, int(window) // 4)).mean()
        std = value.rolling(int(window), min_periods=max(2, int(window) // 4)).std(ddof=0).replace(0.0, np.nan)
        return (value - mean) / std

    @staticmethod
    def _enabled(value: Any) -> bool:
        return str(value or '').lower() == 'on'

    @staticmethod
    def _bool(mask: Series, index: pd.Index) -> Series:
        return pd.Series(mask, index=index).fillna(False).astype(bool)

    @staticmethod
    def _recent(mask: Series, bars: int) -> Series:
        return mask.fillna(False).astype('int8').shift(1).rolling(max(1, int(bars)), min_periods=1).max().gt(0)

    def _add_volume_pressure(self, dataframe: DataFrame, prefix: str, volume_windows: list[int]) -> DataFrame:
        close = self._num(dataframe['close'])
        open_ = self._num(dataframe['open'])
        high = self._num(dataframe['high'])
        low = self._num(dataframe['low'])
        volume = self._num(dataframe['volume']).clip(lower=0.0).fillna(0.0)
        candle_range = (high - low).clip(lower=0.0)
        close_location = (self._safe_div(close - low, candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
        body_pressure = self._safe_div(close - open_, candle_range).clip(-1.0, 1.0)
        delta_pressure = ((close_location.fillna(0.0) + body_pressure.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
        delta = volume * delta_pressure
        dataframe[f'{prefix}_close_location'] = close_location
        dataframe[f'{prefix}_delta_pressure'] = delta_pressure
        dataframe[f'{prefix}_delta_zscore'] = self._zscore(delta, 96 if prefix == 'h1' else 56)
        dataframe[f'{prefix}_cvd'] = delta.cumsum()
        pressure_window = 48 if prefix == 'h1' else 28
        dataframe[f'{prefix}_cvd_trend'] = delta.rolling(pressure_window, min_periods=max(4, pressure_window // 4)).sum()
        for window in volume_windows:
            avg_volume = volume.shift(1).rolling(int(window), min_periods=max(2, int(window) // 4)).mean().replace(0.0, np.nan)
            dataframe[f'{prefix}_rvol_{window}'] = volume / avg_volume
        return dataframe

    def _daily_indicators(self, daily: DataFrame) -> DataFrame:
        frame = daily.copy()
        close = self._num(frame['close'])
        high = self._num(frame['high'])
        low = self._num(frame['low'])
        frame = self._add_volume_pressure(frame, 'd1', D1_VOLUME_WINDOW_CHOICES)
        for lookback in D1_LEVEL_LOOKBACK_CHOICES:
            min_periods = max(5, int(lookback) // 4)
            frame[f'd1_resistance_{lookback}'] = high.shift(1).rolling(int(lookback), min_periods=min_periods).max()
            frame[f'd1_support_{lookback}'] = low.shift(1).rolling(int(lookback), min_periods=min_periods).min()
        frame['d1_trend_up'] = close.gt(close.shift(1))
        frame['d1_trend_down'] = close.lt(close.shift(1))
        return frame

    def _merge_daily_context(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not getattr(self, 'dp', None) or 'date' not in dataframe.columns:
            return dataframe
        daily = self.dp.get_pair_dataframe(pair=metadata.get('pair', ''), timeframe='1d')
        if daily is None or daily.empty or 'date' not in daily.columns:
            return dataframe
        daily = self._daily_indicators(daily)
        keep = ['date', 'd1_close_location', 'd1_delta_zscore', 'd1_cvd_trend', 'd1_trend_up', 'd1_trend_down']
        for window in D1_VOLUME_WINDOW_CHOICES:
            keep.append(f'd1_rvol_{window}')
        for lookback in D1_LEVEL_LOOKBACK_CHOICES:
            keep.extend([f'd1_resistance_{lookback}', f'd1_support_{lookback}'])
        informative = daily[[column for column in keep if column in daily.columns]].copy()
        informative['_sieve_d1_key'] = (pd.to_datetime(informative['date'], utc=True).dt.tz_convert(None) + pd.Timedelta(days=1)).astype('datetime64[ns]')
        informative = informative.drop(columns=['date']).sort_values('_sieve_d1_key')
        base = dataframe.copy()
        base['_sieve_order'] = np.arange(len(base))
        base['_sieve_h1_key'] = pd.to_datetime(base['date'], utc=True).dt.tz_convert(None).astype('datetime64[ns]')
        merged = pd.merge_asof(base.sort_values('_sieve_h1_key'), informative, left_on='_sieve_h1_key', right_on='_sieve_d1_key', direction='backward')
        merged = merged.sort_values('_sieve_order').drop(columns=['_sieve_order', '_sieve_h1_key', '_sieve_d1_key'], errors='ignore')
        merged.index = dataframe.index
        return merged

    def _sieve3_entry_populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        close = self._num(dataframe['close'])
        high = self._num(dataframe['high'])
        low = self._num(dataframe['low'])
        dataframe['sieve_prev_close'] = close.shift(1)
        for lookback in LEVEL_LOOKBACK_CHOICES:
            min_periods = max(2, int(lookback) // 4)
            dataframe[f'sieve_resistance_{lookback}'] = high.shift(1).rolling(int(lookback), min_periods=min_periods).max()
            dataframe[f'sieve_support_{lookback}'] = low.shift(1).rolling(int(lookback), min_periods=min_periods).min()
        for lookback in LOCAL_LOOKBACK_CHOICES:
            dataframe[f'sieve_local_high_{lookback}'] = high.shift(1).rolling(int(lookback), min_periods=2).max()
            dataframe[f'sieve_local_low_{lookback}'] = low.shift(1).rolling(int(lookback), min_periods=2).min()
        dataframe = self._add_volume_pressure(dataframe, 'h1', VOLUME_WINDOW_CHOICES)
        dataframe['sieve_trend_up'] = close.gt(close.shift(1))
        dataframe['sieve_trend_down'] = close.lt(close.shift(1))
        dataframe = self._merge_daily_context(dataframe, metadata)
        dataframe = self._merge_informative_vp(dataframe, metadata, '4h', 'vp4h', int(self.vp_4h_window), int(self.vp_4h_bins))
        dataframe = self._merge_informative_vp(dataframe, metadata, '1d', 'vp1d', int(self.vp_1d_window), int(self.vp_1d_bins))
        dataframe = add_sieve2_guard_indicators(self, dataframe, metadata)
        return dataframe

    def _side_volume_guard(self, dataframe: DataFrame, side: str, local: int, d1_window: int) -> Series:
        index = dataframe.index
        h1_rvol = self._num(dataframe[f'h1_rvol_{int(self.volume_window)}'])
        h1_delta = self._num(dataframe['h1_delta_zscore'])
        h1_cvd = self._num(dataframe['h1_cvd_trend'])
        h1_loc = self._num(dataframe['h1_close_location'])
        d1_rvol = self._num(dataframe[f'd1_rvol_{d1_window}'])
        d1_delta = self._num(dataframe['d1_delta_zscore'])
        d1_cvd = self._num(dataframe['d1_cvd_trend'])
        if side == 'long':
            h1_pressure = h1_delta.ge(float(self.h1_pressure_min)) & h1_cvd.ge(0.0) & h1_loc.ge(-0.15)
            d1_pressure = d1_delta.ge(float(self.d1_pressure_min)) | d1_cvd.ge(0.0)
            event = self._h1_event_guard(dataframe, 'long', local)
        else:
            h1_pressure = h1_delta.le(-float(self.h1_pressure_min)) & h1_cvd.le(0.0) & h1_loc.le(0.15)
            d1_pressure = d1_delta.le(-float(self.d1_pressure_min)) | d1_cvd.le(0.0)
            event = self._h1_event_guard(dataframe, 'short', local)
        guard = pd.Series(True, index=index, dtype='bool')
        guard &= h1_rvol.ge(float(self.h1_rvol_min)).fillna(False)
        guard &= h1_pressure.fillna(False)
        guard &= event.fillna(False)
        guard &= d1_rvol.ge(float(self.d1_rvol_min)).fillna(False)
        guard &= d1_pressure.fillna(False)
        return guard

    def _h1_event_guard(self, dataframe: DataFrame, side: str, local: int) -> Series:
        kind = self.ENTRY_KIND
        close = self._num(dataframe['close'])
        high = self._num(dataframe['high'])
        low = self._num(dataframe['low'])
        local_high = self._num(dataframe[f'sieve_local_high_{local}'])
        local_low = self._num(dataframe[f'sieve_local_low_{local}'])
        close_loc = self._num(dataframe['h1_close_location'])
        trigger = float(self.trigger_buffer_pct)
        long_breakout = close.gt(local_high * (1.0 + trigger)) & close_loc.ge(0.2)
        short_breakout = close.lt(local_low * (1.0 - trigger)) & close_loc.le(-0.2)
        long_reclaim = low.lt(local_low * (1.0 - trigger)) & close.gt(local_low) & close_loc.ge(0.0)
        short_reject = high.gt(local_high * (1.0 + trigger)) & close.lt(local_high) & close_loc.le(0.0)
        if side == 'long':
            if 'break' in kind or 'retest' in kind:
                return long_breakout | self._recent(long_breakout, int(self.confirm_bars))
            return long_reclaim | close_loc.ge(0.15)
        if 'break' in kind or 'retest' in kind:
            return short_breakout | self._recent(short_breakout, int(self.confirm_bars))
        return short_reject | close_loc.le(-0.15)

    def _daily_structure_guard(self, dataframe: DataFrame, side: str, d1_level: int, zone: float, trigger: float) -> Series:
        mode = str(self.d1_structure_mode)
        if mode == 'off':
            return pd.Series(True, index=dataframe.index, dtype='bool')
        close = self._num(dataframe['close'])
        high = self._num(dataframe['high'])
        low = self._num(dataframe['low'])
        d1_resistance = self._num(dataframe[f'd1_resistance_{d1_level}'])
        d1_support = self._num(dataframe[f'd1_support_{d1_level}'])
        wide_zone = max(zone * 2.0, trigger)
        kind = self.ENTRY_KIND
        if kind in {'long_res_break', 'long_res_retest_hold'}:
            near = close.ge(d1_resistance * (1.0 - wide_zone))
            aligned = close.gt(d1_resistance * (1.0 + trigger))
        elif kind in {'long_sup_hold', 'long_sup_reclaim', 'long_trend_pullback'}:
            near = low.le(d1_support * (1.0 + wide_zone))
            aligned = close.gt(d1_support * (1.0 + trigger))
        elif kind in {'short_res_fail', 'short_res_reclaim', 'short_trend_pullback'}:
            near = high.ge(d1_resistance * (1.0 - wide_zone))
            aligned = close.lt(d1_resistance * (1.0 - trigger))
        else:
            near = close.le(d1_support * (1.0 + wide_zone))
            aligned = close.lt(d1_support * (1.0 - trigger))
        return (aligned if mode == 'aligned' else near).fillna(False)

    def _effective_zone(self, zone: float) -> float:
        if str(getattr(self, 'BREATHING_PROFILE', 'none')) == 'loose_support_reclaim':
            return min(zone * 1.5, 0.06)
        return zone

    def _effective_trigger(self, trigger: float) -> float:
        if str(getattr(self, 'BREATHING_PROFILE', 'none')) == 'loose_support_reclaim':
            return max(trigger * 0.6, 0.0)
        if str(getattr(self, 'CONFIRMATION_PROFILE', 'none')) != 'none':
            return min(trigger * 1.25, 0.03)
        return trigger

    def _effective_confirm(self, confirm: int) -> int:
        if str(getattr(self, 'BREATHING_PROFILE', 'none')) == 'loose_support_reclaim':
            return min(max(1, int(confirm * 1.5)), 72)
        if str(getattr(self, 'CONFIRMATION_PROFILE', 'none')) != 'none':
            return min(max(1, int(confirm)), 12)
        return confirm

    def _adaptive_confirmation_guard(self, dataframe: DataFrame, side: str, local: int, level: int, zone: float, trigger: float) -> Series:
        profile = str(getattr(self, 'CONFIRMATION_PROFILE', 'none'))
        if profile == 'none':
            return pd.Series(True, index=dataframe.index, dtype='bool')
        close = self._num(dataframe['close'])
        prev_close = self._num(dataframe['sieve_prev_close'])
        high = self._num(dataframe['high'])
        low = self._num(dataframe['low'])
        resistance = self._num(dataframe[f'sieve_resistance_{level}'])
        support = self._num(dataframe[f'sieve_support_{level}'])
        local_high = self._num(dataframe[f'sieve_local_high_{local}'])
        local_low = self._num(dataframe[f'sieve_local_low_{local}'])
        close_loc = self._num(dataframe['h1_close_location'])
        h1_rvol = self._num(dataframe[f'h1_rvol_{int(self.volume_window)}'])
        h1_delta = self._num(dataframe['h1_delta_zscore'])
        h1_cvd = self._num(dataframe['h1_cvd_trend'])
        trend_up = pd.Series(dataframe['sieve_trend_up'], index=dataframe.index).fillna(False).astype(bool)
        trend_down = pd.Series(dataframe['sieve_trend_down'], index=dataframe.index).fillna(False).astype(bool)
        has_d1_up = 'd1_trend_up' in dataframe.columns
        has_d1_down = 'd1_trend_down' in dataframe.columns
        d1_up = pd.Series(dataframe['d1_trend_up'], index=dataframe.index).fillna(True).astype(bool)
        d1_down = pd.Series(dataframe['d1_trend_down'], index=dataframe.index).fillna(True).astype(bool)
        d1_not_up = ~d1_up if has_d1_up else pd.Series(True, index=dataframe.index, dtype='bool')
        d1_not_down = ~d1_down if has_d1_down else pd.Series(True, index=dataframe.index, dtype='bool')
        min_rvol = max(float(self.h1_rvol_min), 1.2)
        min_pressure = max(float(self.h1_pressure_min), 0.3)
        long_pressure = h1_rvol.ge(min_rvol) & h1_delta.ge(min_pressure) & h1_cvd.ge(0.0) & close_loc.ge(0.25)
        short_pressure = h1_rvol.ge(min_rvol) & h1_delta.le(-min_pressure) & h1_cvd.le(0.0) & close_loc.le(-0.25)
        long_breakout = close.gt(local_high * (1.0 + trigger)) & close.gt(resistance * (1.0 + trigger)) & close_loc.ge(0.35)
        short_breakdown = close.lt(local_low * (1.0 - trigger)) & close.lt(support * (1.0 - trigger)) & close_loc.le(-0.35)
        long_support_response = low.le(support * (1.0 + zone)) & close.gt(support * (1.0 + trigger)) & close_loc.ge(0.25)
        short_resistance_response = high.ge(resistance * (1.0 - zone)) & close.lt(resistance * (1.0 - trigger)) & close_loc.le(-0.25)
        if profile == 'long_resistance_break':
            return (long_pressure & long_breakout & trend_up & d1_up).fillna(False)
        if profile == 'long_resistance_retest':
            retest_hold = low.le(resistance * (1.0 + zone)) & close.gt(resistance * (1.0 + trigger * 0.5)) & close_loc.ge(0.25)
            return (long_pressure & retest_hold & trend_up & d1_up).fillna(False)
        if profile == 'long_support_hold':
            return (long_pressure & long_support_response & ~trend_down & d1_not_down).fillna(False)
        if profile == 'long_trend_pullback':
            trend_reclaim = close.gt(prev_close) & close_loc.ge(0.25)
            return (long_pressure & trend_reclaim & trend_up & d1_not_down).fillna(False)
        if profile == 'short_resistance_fail':
            return (short_pressure & short_resistance_response & ~trend_up & d1_not_up).fillna(False)
        if profile == 'short_support_break':
            return (short_pressure & short_breakdown & trend_down & d1_down).fillna(False)
        if profile == 'short_support_retest':
            retest_reject = high.ge(support * (1.0 - zone)) & close.lt(support * (1.0 - trigger * 0.5)) & close_loc.le(-0.25)
            return (short_pressure & retest_reject & trend_down & d1_down).fillna(False)
        return (long_pressure if side == 'long' else short_pressure).fillna(False)

    def _merge_informative_vp(self, dataframe: DataFrame, metadata: dict, timeframe: str, prefix: str, window: int, bins: int) -> DataFrame:
        if 'date' not in dataframe.columns or not getattr(self, 'dp', None):
            return dataframe
        pair = str(metadata.get('pair') or '')
        if not pair:
            return dataframe
        informative = self.dp.get_pair_dataframe(pair=pair, timeframe=timeframe)
        if informative is None or informative.empty or 'date' not in informative.columns:
            return dataframe
        informative = add_volume_profile(informative.copy(), window=window, bins=bins, value_area_pct=float(self.vp_guard_value_area_pct), price_source=str(self.vp_guard_price_source), pressure_delta_min=float(self.vp_guard_pressure_delta_min), node_near_pct=float(self.vp_guard_node_near_pct), prefix=prefix)
        merge_columns = [f'{prefix}_{name}' for name in VP_COLUMNS if f'{prefix}_{name}' in informative.columns]
        if not merge_columns:
            return dataframe
        minutes = timeframe_to_minutes(timeframe)
        inf = informative[['date', *merge_columns]].copy().sort_values('date')
        inf['date_merge'] = inf['date'] + pd.to_timedelta(minutes, unit='m')
        base = dataframe.reset_index().rename(columns={'index': '__row_index'}).sort_values('date')
        merged = pd.merge_asof(base, inf[['date_merge', *merge_columns]].sort_values('date_merge'), left_on='date', right_on='date_merge', direction='backward').sort_values('__row_index')
        for column in merge_columns:
            dataframe[column] = merged[column].to_numpy()
        return dataframe

    def _score_guard(self, dataframe: DataFrame, prefix: str, side: str, score_min: float) -> Series:
        score = self._num(dataframe[f'{prefix}_score_{side}'])
        opposite_side = 'short' if side == 'long' else 'long'
        opposite = self._num(dataframe[f'{prefix}_score_{opposite_side}'])
        return score.ge(score_min) & score.ge(opposite)

    def _context_guard(self, dataframe: DataFrame, prefix: str, side: str, context_min: float) -> Series:
        if side == 'long':
            context = self._num(dataframe[f'{prefix}_context_score_bull'])
            opposite = self._num(dataframe[f'{prefix}_context_score_bear'])
            market_ok = self._num(dataframe[f'{prefix}_market_context']).ge(0)
        else:
            context = self._num(dataframe[f'{prefix}_context_score_bear'])
            opposite = self._num(dataframe[f'{prefix}_context_score_bull'])
            market_ok = self._num(dataframe[f'{prefix}_market_context']).le(0)
        return context.ge(context_min) & context.ge(opposite) & market_ok

    def _vp_guard(self, dataframe: DataFrame, prefix: str, side: str, mode: str, score_min: float, context_min: float) -> Series:
        score_ok = self._score_guard(dataframe, prefix, side, score_min)
        context_ok = self._context_guard(dataframe, prefix, side, context_min)
        balance = self._num(dataframe[f'{prefix}_context_score_balance'])
        if mode == 'score':
            return score_ok
        if mode == 'context':
            return context_ok
        if mode == 'score_or_context':
            return score_ok | context_ok
        if mode == 'balance':
            return balance.ge(context_min)
        market = self._num(dataframe[f'{prefix}_market_context'])
        return market.ge(0) if side == 'long' else market.le(0)

    def _entry_mask(self, dataframe: DataFrame) -> Series:
        level = int(self.level_lookback)
        local = int(self.local_lookback)
        d1_level = int(self.d1_level_lookback)
        d1_window = int(self.d1_volume_window)
        zone = self._effective_zone(float(self.zone_pct))
        trigger = self._effective_trigger(float(self.trigger_buffer_pct))
        confirm = self._effective_confirm(int(self.confirm_bars))
        close = self._num(dataframe['close'])
        high = self._num(dataframe['high'])
        low = self._num(dataframe['low'])
        prev_close = self._num(dataframe['sieve_prev_close'])
        resistance = self._num(dataframe[f'sieve_resistance_{level}'])
        support = self._num(dataframe[f'sieve_support_{level}'])
        local_high = self._num(dataframe[f'sieve_local_high_{local}'])
        local_low = self._num(dataframe[f'sieve_local_low_{local}'])
        trend_up = pd.Series(dataframe['sieve_trend_up'], index=dataframe.index).fillna(False).astype(bool)
        trend_down = pd.Series(dataframe['sieve_trend_down'], index=dataframe.index).fillna(False).astype(bool)
        kind = self.ENTRY_KIND
        if kind == 'long_res_break':
            line = resistance * (1.0 + trigger)
            structure = close.gt(line) & prev_close.le(line) & local_high.ge(resistance * (1.0 - zone))
            side = 'long'
        elif kind == 'long_sup_hold':
            structure = low.le(support * (1.0 + zone)) & local_low.le(support * (1.0 + zone)) & close.gt(support * (1.0 + trigger))
            side = 'long'
        elif kind == 'long_sup_reclaim':
            structure = low.lt(support * (1.0 - trigger)) & local_low.le(support * (1.0 + zone)) & close.gt(support * (1.0 + trigger))
            side = 'long'
        elif kind == 'long_res_retest_hold':
            break_mask = close.gt(resistance * (1.0 + trigger))
            structure = self._recent(break_mask, confirm) & low.le(resistance * (1.0 + zone)) & local_low.le(resistance * (1.0 + zone)) & close.ge(resistance)
            side = 'long'
        elif kind == 'short_res_fail':
            structure = high.ge(resistance * (1.0 - zone)) & local_high.ge(resistance * (1.0 - zone)) & close.lt(resistance * (1.0 - trigger))
            side = 'short'
        elif kind == 'short_res_reclaim':
            structure = high.gt(resistance * (1.0 + trigger)) & local_high.ge(resistance * (1.0 - zone)) & close.lt(resistance * (1.0 - trigger))
            side = 'short'
        elif kind == 'short_sup_break':
            line = support * (1.0 - trigger)
            structure = close.lt(line) & prev_close.ge(line) & local_low.le(support * (1.0 + zone))
            side = 'short'
        elif kind == 'short_sup_retest_reject':
            break_mask = close.lt(support * (1.0 - trigger))
            structure = self._recent(break_mask, confirm) & high.ge(support * (1.0 - zone)) & local_high.ge(support * (1.0 - zone)) & close.le(support)
            side = 'short'
        elif kind == 'long_trend_pullback':
            structure = trend_up & low.le(support * (1.0 + zone)) & close.gt(support * (1.0 + trigger)) & close.gt(prev_close)
            side = 'long'
        elif kind == 'short_trend_pullback':
            structure = trend_down & high.ge(resistance * (1.0 - zone)) & close.lt(resistance * (1.0 - trigger)) & close.lt(prev_close)
            side = 'short'
        else:
            structure = pd.Series(False, index=dataframe.index)
            side = self.ENTRY_SIDE
        confirmation = self._adaptive_confirmation_guard(dataframe, side, local, level, zone, trigger)
        mask = structure & confirmation & self._side_volume_guard(dataframe, side, local, d1_window) & self._daily_structure_guard(dataframe, side, d1_level, zone, trigger)
        return self._bool(mask, dataframe.index)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = ''
        mask = self._entry_mask(dataframe)
        mask = apply_sieve2_optional_guards(self, dataframe, mask, self.ENTRY_SIDE)
        dataframe[f'plot_{self.ENTRY_TAG}'] = mask.astype(float)
        dataframe.loc[mask, 'enter_long'] = 1
        dataframe.loc[mask, 'enter_tag'] = self.ENTRY_TAG
        return dataframe

    def _generic_level_frame(self, dataframe: DataFrame, _timeframe: str) -> DataFrame:
        return add_volume_profile(dataframe, prefix='gsvp')
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        frame = self._sieve3_entry_populate_indicators(dataframe, metadata)
    
        if True:
            frame = self._generic_level_frame(frame, self.timeframe)
        return frame

    SOURCE_ENTRY_STEM = 'loosen_ladder_long_sup_hold_broader_trigger'
    FOCUSED_EXIT_CONTRACT = 'va_edge_reaction_base'
    FOCUSED_SOURCE_PROFILE = {'side': 'long', 'min_level_distance': 0.001}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'high', 'low', 'open')
    FOCUSED_STATE_VERSION = 1
    FOCUSED_STATE_KEY = 'sieve3_v2_generic:Sieve3V2VaEdgeReactionBaseFromLoosenLadderLongSupHoldBroaderTrigger:va_edge_reaction_base'
    LOCKED_BUY_PARAMS = {'use_sieve2_vp_guard': True, 'sieve2_vp_guard_mode': 'score_or_context', 'sieve2_vp_window': 96, 'sieve2_vp_bins': 36, 'sieve2_vp_score_min': 0.15, 'sieve2_vp_context_min': 0.15, 'use_sieve2_market_guard': True, 'sieve2_market_guard_mode': 'pressure_or_trend', 'sieve2_market_window': 24, 'sieve2_market_pressure_min': 0.03, 'sieve2_market_trend_min': 0.15, 'sieve2_rs_benchmark_pair': 'BTC/USDT:USDT', 'sieve2_rs_score_min': 0.3, 'level_lookback': 168, 'local_lookback': 48, 'd1_level_lookback': 50, 'volume_window': 24, 'd1_volume_window': 28, 'confirm_bars': 3, 'zone_pct': 0.012, 'trigger_buffer_pct': 0.003, 'h1_rvol_enable': 'on', 'h1_pressure_enable': 'on', 'h1_event_enable': 'on', 'd1_rvol_enable': 'on', 'd1_pressure_enable': 'on', 'd1_structure_mode': 'off', 'h1_rvol_min': 1.1, 'h1_pressure_min': 0.3, 'd1_rvol_min': 1.0, 'd1_pressure_min': 0.2, 'use_vp_4h_guard': 'off', 'vp_4h_guard_mode': 'score_or_context', 'vp_4h_window': 48, 'vp_4h_bins': 36, 'vp_4h_score_min': 0.25, 'vp_4h_context_min': 0.28, 'use_vp_1d_guard': 'off', 'vp_1d_guard_mode': 'context', 'vp_1d_window': 30, 'vp_1d_bins': 36, 'vp_1d_score_min': 0.25, 'vp_1d_context_min': 0.28, 'vp_guard_value_area_pct': 0.7, 'vp_guard_price_source': 'hlc3', 'vp_guard_node_near_pct': 0.01, 'vp_guard_pressure_delta_min': 0.05}
    ACTIVE_SELL_PARAMS = ('zone_width_quarter_percent', 'weakening_signal', 'confirmation_profile')
    GENERIC_MODE = 'simple'
    GENERIC_LEVEL_FAMILY = 'va_edge'
    GENERIC_TIMEFRAME_SCOPE = 'base'
    GENERIC_AVAILABLE_TIMEFRAMES = ('1h', '4h', '8h', '1d', '3d')
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

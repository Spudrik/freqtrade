from __future__ import annotations
import math
import pandas as pd
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from pandas import DataFrame, Series
from user_data.Indicators.market_state import add_market_state
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, stoploss_from_absolute, IntParameter
LOCKED_BUY_SOURCE = 'verified tested-snapshot effective buy lock'
SOURCE_ENTRY_SIGNATURE_SHA256 = '06a2256aba445bbf73814ca7229acc6b2e33c52c18368c71452102919b8a68e2'
SECONDARY_GUARD = 'H1 event, VP4h, VP1d, and D1 structure switches are locked off and remain inactive'
UPDATE_HYPOTHESIS = 'High-trade entry may contain a real directional edge; require stronger volume-profile/market context so HTF or broader context carries confidence and the original trigger supplies execution.'
import numpy as np
from freqtrade.strategy import IStrategy
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
    if always_on:
        guarded &= _sieve2_vp_guard(
            dataframe,
            normalized_side,
            str(_sieve2_param(strategy, "sieve2_vp_guard_mode", "score_or_context")),
            float(_sieve2_param(strategy, "sieve2_vp_score_min", 0.25)),
            float(_sieve2_param(strategy, "sieve2_vp_context_min", 0.28)),
        )
    if always_on:
        guarded &= _sieve2_market_guard(
            dataframe,
            normalized_side,
            str(_sieve2_param(strategy, "sieve2_market_guard_mode", "pressure_or_trend")),
            float(_sieve2_param(strategy, "sieve2_market_pressure_min", 0.07)),
            float(_sieve2_param(strategy, "sieve2_market_trend_min", 0.25)),
            float(0.3),
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
SOURCE_ENTRY_STEM = 'overtrade_pivot_long_resistance_breakout_vp_market_guard'

SOURCE_ENTRY_STAGE = 'sieve2'
SOURCE_ENTRY_CLASS = 'Sieve2OvertradePivotLongResistanceBreakoutVPMarketGuard'
SOURCE_STRATEGY = 'D:\\FreqTradeStuffLargeData\\sieve_runtime\\entry_sieve\\backtests\\sieve2_overtrade_pivot_long_resistance_breakout_vp_market_guard__auto_generic_1h_2020_q2_q3\\full_cycle_2020_2026\\tp_3_sl_3\\backtest-result-2026-06-12_17-46-50.zip!backtest-result-2026-06-12_17-46-50_Sieve2OvertradePivotLongResistanceBreakoutVPMarketGuard.py:Sieve2OvertradePivotLongResistanceBreakoutVPMarketGuard'
LINEAGE_STATUS = 'verified'

EXIT_THEORY = 'entry_target_full_or_zone_reversal'







EXIT_FAMILY = 'entry_target_full_or_zone_reversal'
PRIMARY_TRIGGER = '1h close cross above locked sieve_resistance_336 with local-high confirmation'
PRIMARY_GUARD = 'locked D1 relative-volume and pressure confirmation plus verified Sieve2 VP/market guards'
PRIMARY_TARGET = 'entry-frozen nearest forward VP/TLV2/structural obstacle'
PRIMARY_INVALIDATION = 'crossed_resistance_336_loss'
TARGET_PROVIDER = 'nearest_forward_vp_tlv2_or_structure_obstacle'
INVALIDATION_PROVIDER = 'crossed_resistance_336_loss'
ENTRY_SOURCE_STAGE = 'sieve2'
SOURCE_RESULT_BATCH = 'historical_promotion_lineage_unknown'
RESEARCH_PATH = 'sieve3_exit_entry_target_full_or_zone_reversal'
EXIT_HYPOTHESIS = 'An entry-frozen forward structural target should support full or staged exits while the crossed entry level remains the source-specific invalidation.'
ACTIVE_SELL_PARAMS = ('exit_plan', 'target_band_quarter_percent', 'reversal_confirmations', 'invalidation_band_quarter_percent', 'invalidation_confirmations')
SIEVE_STAGE = 'sieve3'
SIDE = 'long'
ENTRY_TAG = 'long_resistance_breakout'

class Sieve3V2EntryTargetFullOrZoneReversalFromOvertradePivotLongResistanceBreakoutVpMarketGuard(IStrategy):
    SIEVE2_FUNDAMENTAL_REWORK = '20260605_nonprofitable_tighten_overtrade'
    REWORK_HYPOTHESIS = 'Entry-only rerun after non-profitable revised-guard result: tighten_overtrade; widen guard thresholds and adjust entry-strength search space.'
    SIEVE2_ALWAYS_ON_GUARDS = True
    INTERFACE_VERSION = 3
    can_short = False
    timeframe = '1h'
    startup_candle_count = 336
    process_only_new_candles = True
    max_entry_position_adjustment = 0
    # Legacy inactive entry parameter: use_sieve2_vp_guard=True (guards are fixed always-on).
    sieve2_vp_guard_mode = 'score_or_context'
    sieve2_vp_window = 96
    sieve2_vp_bins = 36
    sieve2_vp_score_min = 0.15
    sieve2_vp_context_min = 0.15
    # Legacy inactive entry parameter: use_sieve2_market_guard=True (guards are fixed always-on).
    sieve2_market_guard_mode = 'pressure_or_trend'
    sieve2_market_window = 24
    sieve2_market_pressure_min = 0.03
    sieve2_market_trend_min = 0.15
    # Legacy inactive entry parameter: sieve2_rs_benchmark_pair='BTC/USDT:USDT' (fixed guard mode does not consume it).
    # Legacy inactive entry parameter: sieve2_rs_score_min=0.3 (fixed guard mode does not consume it).
    ENTRY_TAG = 'long_resistance_breakout'
    ENTRY_SIDE = 'long'
    ENTRY_KIND = 'long_res_break'
    MODE = 'entry_long_resistance_breakout'
    CONFIRMATION_PROFILE = 'long_resistance_break'
    BREATHING_PROFILE = 'none'
    level_lookback = 336
    local_lookback = 24
    # Legacy inactive entry parameter: d1_level_lookback=50 (daily structure mode is fixed off).
    volume_window = 24
    d1_volume_window = 28
    confirm_bars = 12
    zone_pct = 0.012
    trigger_buffer_pct = 0.0
    # Legacy inactive entry parameter: h1_rvol_enable='off' (enable gate is fixed off).
    # Legacy inactive entry parameter: h1_pressure_enable='off' (enable gate is fixed off).
    # Legacy inactive entry parameter: h1_event_enable='off' (enable gate is fixed off).
    d1_rvol_enable = 'on'
    d1_pressure_enable = 'on'
    # Legacy inactive entry parameter: d1_structure_mode='off' (fixed mode bypasses daily structure).
    h1_rvol_min = 1.1
    h1_pressure_min = 0.3
    d1_rvol_min = 1.0
    d1_pressure_min = 0.2
    # Legacy inactive entry parameters: use_vp_4h_guard='off', vp_4h_guard_mode='score_or_context',
    # vp_4h_window=48, vp_4h_bins=36, vp_4h_score_min=0.25, vp_4h_context_min=0.28 (4h VP gate is fixed off).
    # Legacy inactive entry parameters: use_vp_1d_guard='off', vp_1d_guard_mode='context',
    # vp_1d_window=30, vp_1d_bins=36, vp_1d_score_min=0.25, vp_1d_context_min=0.28 (1d VP gate is fixed off).
    # Legacy inactive entry parameters: vp_guard_value_area_pct=0.7, vp_guard_price_source='hlc3',
    # vp_guard_node_near_pct=0.01, vp_guard_pressure_delta_min=0.05 (both VP gates are fixed off).

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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
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
        dataframe = add_sieve2_guard_indicators(self, dataframe, metadata)
        dataframe = self._add_sieve3_forward_target(dataframe)
        return dataframe

    def _side_volume_guard(self, dataframe: DataFrame, side: str, local: int, d1_window: int) -> Series:
        index = dataframe.index
        self._num(dataframe[f'h1_rvol_{int(self.volume_window)}'])
        h1_delta = self._num(dataframe['h1_delta_zscore'])
        h1_cvd = self._num(dataframe['h1_cvd_trend'])
        h1_loc = self._num(dataframe['h1_close_location'])
        d1_rvol = self._num(dataframe[f'd1_rvol_{d1_window}'])
        d1_delta = self._num(dataframe['d1_delta_zscore'])
        d1_cvd = self._num(dataframe['d1_cvd_trend'])
        if side == 'long':
            h1_delta.ge(float(self.h1_pressure_min)) & h1_cvd.ge(0.0) & h1_loc.ge(-0.15)
            d1_pressure = d1_delta.ge(float(self.d1_pressure_min)) | d1_cvd.ge(0.0)
            self._h1_event_guard(dataframe, 'long', local)
        else:
            h1_delta.le(-float(self.h1_pressure_min)) & h1_cvd.le(0.0) & h1_loc.le(0.15)
            d1_pressure = d1_delta.le(-float(self.d1_pressure_min)) | d1_cvd.le(0.0)
            self._h1_event_guard(dataframe, 'short', local)
        guard = pd.Series(True, index=index, dtype='bool')
        if self._enabled(self.d1_rvol_enable):
            guard &= d1_rvol.ge(float(self.d1_rvol_min)).fillna(False)
        if self._enabled(self.d1_pressure_enable):
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
        return pd.Series(True, index=dataframe.index, dtype='bool')

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

    def _entry_mask(self, dataframe: DataFrame) -> Series:
        level = int(self.level_lookback)
        local = int(self.local_lookback)
        d1_level = 50
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
        mask = apply_sieve2_optional_guards(self, dataframe, mask, getattr(self, 'ENTRY_SIDE', 'long'))
        dataframe[f'plot_{self.ENTRY_TAG}'] = mask.astype(float)
        dataframe.loc[mask, 'enter_long'] = 1
        dataframe.loc[mask, 'enter_tag'] = self.ENTRY_TAG
        return dataframe

    @staticmethod
    def _add_sieve3_forward_target(dataframe: DataFrame) -> DataFrame:
        candidate_columns = ('s2vp_hvn_above', 's2vp_vah', 's2vp_poc', 'd1_resistance_50')
        missing = [column for column in candidate_columns if column not in dataframe.columns]
        if len(missing) == len(candidate_columns):
            raise KeyError(f'no configured forward-target columns are present: {missing}')
        close = pd.to_numeric(dataframe['close'], errors='coerce')
        levels = pd.concat(
            [pd.to_numeric(dataframe[column], errors='coerce') for column in candidate_columns if column in dataframe.columns],
            axis=1,
        )
        invalidation = pd.to_numeric(dataframe['sieve_resistance_336'], errors='coerce')
        same_as_invalidation = pd.DataFrame(
            np.isclose(
                levels.to_numpy(dtype=float),
                invalidation.to_numpy(dtype=float)[:, None],
                rtol=0.0,
                atol=1e-12,
                equal_nan=False,
            ),
            index=levels.index,
            columns=levels.columns,
        )
        levels = levels.mask(same_as_invalidation)
        forward = levels.where(levels.gt(close.mul(1.001), axis='index'), np.nan)
        dataframe['sieve3_exit_target_level'] = forward.min(axis=1, skipna=True)
        return dataframe

    SOURCE_ENTRY_STEM = 'overtrade_pivot_long_resistance_breakout_vp_market_guard'
    FOCUSED_EXIT_CONTRACT = 'entry_target_full_or_zone_reversal'
    FOCUSED_SOURCE_PROFILE = {'side': 'long', 'min_level_distance': 0.001, 'targets': {'target_1': {'provider': 'nearest_forward_vp_tlv2_or_structure_obstacle', 'long': {'level': 'sieve3_exit_target_level', 'available': None}}}, 'invalidation': {'provider': 'crossed_resistance_336_loss', 'mode': 'level', 'long': {'level': 'sieve_resistance_336', 'available': None}}}
    FOCUSED_EXIT_PLANS = {'touch_full': {'contract': 'entry_target_full_or_zone_reversal', 'name': 'touch_full', 'hard_stop_ratio': 0.03, 'max_hold_candles': 336}, 'close_full': {'contract': 'entry_target_full_or_zone_reversal', 'name': 'close_full', 'hard_stop_ratio': 0.03, 'max_hold_candles': 336}, 'zone_reversal': {'contract': 'entry_target_full_or_zone_reversal', 'name': 'zone_reversal', 'hard_stop_ratio': 0.03, 'max_hold_candles': 336}}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'high', 'low', 'open', 'sieve3_exit_target_level', 'sieve_resistance_336')
    FOCUSED_STATE_VERSION = 1
    FOCUSED_STATE_KEY = 'sieve3_v2_focused:Sieve3V2EntryTargetFullOrZoneReversalFromOvertradePivotLongResistanceBreakoutVpMarketGuard:entry_target_full_or_zone_reversal'
    LOCKED_BUY_PARAMS = {'use_sieve2_vp_guard': True, 'sieve2_vp_guard_mode': 'score_or_context', 'sieve2_vp_window': 96, 'sieve2_vp_bins': 36, 'sieve2_vp_score_min': 0.15, 'sieve2_vp_context_min': 0.15, 'use_sieve2_market_guard': True, 'sieve2_market_guard_mode': 'pressure_or_trend', 'sieve2_market_window': 24, 'sieve2_market_pressure_min': 0.03, 'sieve2_market_trend_min': 0.15, 'sieve2_rs_benchmark_pair': 'BTC/USDT:USDT', 'sieve2_rs_score_min': 0.3, 'level_lookback': 336, 'local_lookback': 24, 'd1_level_lookback': 50, 'volume_window': 24, 'd1_volume_window': 28, 'confirm_bars': 12, 'zone_pct': 0.012, 'trigger_buffer_pct': 0.0, 'h1_rvol_enable': 'off', 'h1_pressure_enable': 'off', 'h1_event_enable': 'off', 'd1_rvol_enable': 'on', 'd1_pressure_enable': 'on', 'd1_structure_mode': 'off', 'h1_rvol_min': 1.1, 'h1_pressure_min': 0.3, 'd1_rvol_min': 1.0, 'd1_pressure_min': 0.2, 'use_vp_4h_guard': 'off', 'vp_4h_guard_mode': 'score_or_context', 'vp_4h_window': 48, 'vp_4h_bins': 36, 'vp_4h_score_min': 0.25, 'vp_4h_context_min': 0.28, 'use_vp_1d_guard': 'off', 'vp_1d_guard_mode': 'context', 'vp_1d_window': 30, 'vp_1d_bins': 36, 'vp_1d_score_min': 0.25, 'vp_1d_context_min': 0.28, 'vp_guard_value_area_pct': 0.7, 'vp_guard_price_source': 'hlc3', 'vp_guard_node_near_pct': 0.01, 'vp_guard_pressure_delta_min': 0.05}
    ACTIVE_SELL_PARAMS = ('exit_plan', 'target_band_quarter_percent', 'reversal_confirmations', 'invalidation_band_quarter_percent', 'invalidation_confirmations')
    position_adjustment_enable = False
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    minimal_roi = {'0': 100.0}
    stoploss = -0.99
    exit_plan = CategoricalParameter(('touch_full', 'close_full', 'zone_reversal'), default='zone_reversal', space='sell', optimize=True, load=True)
    exit_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    target_band_quarter_percent = IntParameter(0, 12, default=2, space='sell', optimize=True, load=True)
    target_band_quarter_percent.batch_tags = ('family:exits', 'mode:sieve3_exit')
    reversal_confirmations = IntParameter(1, 3, default=1, space='sell', optimize=True, load=True)
    reversal_confirmations.batch_tags = ('family:exits', 'mode:sieve3_exit')
    invalidation_band_quarter_percent = IntParameter(0, 4, default=1, space='sell', optimize=True, load=True)
    invalidation_band_quarter_percent.batch_tags = ('family:exits', 'mode:sieve3_exit')
    invalidation_confirmations = IntParameter(1, 3, default=1, space='sell', optimize=True, load=True)
    invalidation_confirmations.batch_tags = ('family:exits', 'mode:sieve3_exit')

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
        name = str((state or {}).get('plan') or self.exit_plan.value)
        if name not in self.FOCUSED_EXIT_PLANS:
            raise ValueError(f'unknown focused exit plan: {name}')
        plan = dict(self.FOCUSED_EXIT_PLANS[name])
        plan['target_confirmation'] = {'touch_full': 'touch', 'close_full': 'close', 'zone_reversal': f'reversal_{int(self.reversal_confirmations.value)}'}[name]
        plan['target_band'] = float(self.target_band_quarter_percent.value) * 0.0025
        plan['invalidation_band'] = float(self.invalidation_band_quarter_percent.value) * 0.0025
        plan['invalidation_confirmations'] = int(self.invalidation_confirmations.value)
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
        selected_plan = str(self.exit_plan.value)
        fill_time = (getattr(order, 'order_filled_utc', None) if order is not None else None) or getattr(trade, 'date_entry_fill_utc', None) or getattr(trade, 'open_date_utc', None) or current_time
        fill_timestamp = self._focused_utc(fill_time)
        snapshot_date = self._focused_utc(row['date'])
        return {'version': self.FOCUSED_STATE_VERSION, 'contract': self.FOCUSED_EXIT_CONTRACT, 'plan': selected_plan, 'side': side, 'phase': 'pre_target', 'entry_rate': float(entry_rate), 'entry_filled_at': fill_timestamp.isoformat() if fill_timestamp is not None else None, 'entry_snapshot_candle': snapshot_date.isoformat() if snapshot_date is not None else None, 'levels': levels, 'target_states': {slot: {'status': 'available' if value is not None else 'unavailable', 'touched_at': None} for slot, value in levels.items() if slot.startswith('target_')}, 'invalidation_seen': False, 'guard_state': 'unknown', 'trigger_state': 'unknown', 'initial_stake': self._focused_float(getattr(trade, 'stake_amount', None)), 'favorable_rate': float(entry_rate), 'stop_price': None, 'terminal_pending': False, 'terminal_tag': None}

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

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
SOURCE_ENTRY_CLASS = 'Sieve3ExitFromLoosenPivotLongTrendPullbackBroaderTrigger'
SOURCE_STRATEGY = 'user_data/strategies/sieve3_exit_from_loosen_pivot_long_trend_pullback_broader_trigger.py:Sieve3ExitFromLoosenPivotLongTrendPullbackBroaderTrigger'
LINEAGE_STATUS = 'executable_source_verified; historical_parent_path_missing'
EXIT_FAMILY = 'profit_ladder_three_stage_no_ratchet'
EXIT_THEORY = 'profit_ladder_three_stage_no_ratchet'
EXIT_HYPOTHESIS = 'Three ordered arbitrary profit targets take two independently sized partials and fully exit the remainder; the original protective stop remains unchanged until the final exit.'
PRIMARY_TRIGGER = 'none'
PRIMARY_GUARD = 'none'
TARGET_PROVIDER = 'none'
INVALIDATION_PROVIDER = 'none'
ACTIVE_SELL_PARAMS = ('target_1_percent', 'target_gap_half_percent_units', 'partial_1_five_percent_units', 'partial_2_five_percent_units', 'hard_stop_percent')

RESEARCH_PATH = "sieve3_exit_profit_ladder_three_stage_no_ratchet"
SOURCE_RESULT_BATCH = "historical_promotion_lineage_unknown"


class Sieve3V2ProfitLadderThreeStageNoRatchetFromLoosenPivotLongTrendPullbackBroaderTrigger(IStrategy):
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
    ENTRY_TAG = 'long_trend_pullback'
    ENTRY_SIDE = 'long'
    ENTRY_KIND = 'long_trend_pullback'
    MODE = 'entry_long_trend_pullback'
    CONFIRMATION_PROFILE = 'long_trend_pullback'
    BREATHING_PROFILE = 'none'
    level_lookback = 96
    local_lookback = 24
    d1_level_lookback = 50
    volume_window = 24
    d1_volume_window = 28
    confirm_bars = 3
    zone_pct = 0.024
    trigger_buffer_pct = 0.003
    # Inactive buy declaration: h1_rvol_enable = CategoricalParameter(default='on', space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: h1_pressure_enable = CategoricalParameter(default='off', space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
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
            h1_delta.ge(float(self.h1_pressure_min)) & h1_cvd.ge(0.0) & h1_loc.ge(-0.15)
            d1_pressure = d1_delta.ge(float(self.d1_pressure_min)) | d1_cvd.ge(0.0)
            event = self._h1_event_guard(dataframe, 'long', local)
        else:
            h1_delta.le(-float(self.h1_pressure_min)) & h1_cvd.le(0.0) & h1_loc.le(0.15)
            d1_pressure = d1_delta.le(-float(self.d1_pressure_min)) | d1_cvd.le(0.0)
            event = self._h1_event_guard(dataframe, 'short', local)
        guard = pd.Series(True, index=index, dtype='bool')
        guard &= h1_rvol.ge(float(self.h1_rvol_min)).fillna(False)
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
    SOURCE_ENTRY_STEM = 'loosen_pivot_long_trend_pullback_broader_trigger'
    FOCUSED_EXIT_CONTRACT = 'profit_ladder_three_stage_no_ratchet'
    FOCUSED_SOURCE_PROFILE = {'side': 'long', 'min_level_distance': 0.001}
    FOCUSED_EXIT_PLANS = {'profit_ladder_three_stage_no_ratchet': {'contract': 'profit_ladder_three_stage_no_ratchet', 'name': 'profit_ladder_three_stage_no_ratchet', 'action_sequence': ('profit_target_1_partial', 'profit_target_2_partial', 'profit_target_3_full', 'hard_stop', 'max_hold'), 'hard_stop_ratio': 0.03, 'max_hold_candles': 336}}
    FOCUSED_REQUIRED_COLUMNS = ('close', 'date', 'high', 'low', 'open')
    FOCUSED_STATE_VERSION = 1
    FOCUSED_STATE_KEY = 'sieve3_v2_focused:Sieve3V2ProfitLadderThreeStageNoRatchetFromLoosenPivotLongTrendPullbackBroaderTrigger:profit_ladder_three_stage_no_ratchet'
    LOCKED_BUY_PARAMS = {'use_sieve2_vp_guard': True, 'sieve2_vp_guard_mode': 'score_or_context', 'sieve2_vp_window': 96, 'sieve2_vp_bins': 36, 'sieve2_vp_score_min': 0.15, 'sieve2_vp_context_min': 0.15, 'use_sieve2_market_guard': True, 'sieve2_market_guard_mode': 'pressure_or_trend', 'sieve2_market_window': 24, 'sieve2_market_pressure_min': 0.03, 'sieve2_market_trend_min': 0.15, 'sieve2_rs_benchmark_pair': 'BTC/USDT:USDT', 'sieve2_rs_score_min': 0.3, 'level_lookback': 96, 'local_lookback': 24, 'd1_level_lookback': 50, 'volume_window': 24, 'd1_volume_window': 28, 'confirm_bars': 3, 'zone_pct': 0.024, 'trigger_buffer_pct': 0.003, 'h1_rvol_enable': 'on', 'h1_pressure_enable': 'off', 'h1_event_enable': 'on', 'd1_rvol_enable': 'on', 'd1_pressure_enable': 'on', 'd1_structure_mode': 'off', 'h1_rvol_min': 1.1, 'h1_pressure_min': 0.3, 'd1_rvol_min': 1.0, 'd1_pressure_min': 0.2, 'use_vp_4h_guard': 'off', 'vp_4h_guard_mode': 'score_or_context', 'vp_4h_window': 48, 'vp_4h_bins': 36, 'vp_4h_score_min': 0.25, 'vp_4h_context_min': 0.28, 'use_vp_1d_guard': 'off', 'vp_1d_guard_mode': 'context', 'vp_1d_window': 30, 'vp_1d_bins': 36, 'vp_1d_score_min': 0.25, 'vp_1d_context_min': 0.28, 'vp_guard_value_area_pct': 0.7, 'vp_guard_price_source': 'hlc3', 'vp_guard_node_near_pct': 0.01, 'vp_guard_pressure_delta_min': 0.05}
    ACTIVE_SELL_PARAMS = ('target_1_percent', 'target_gap_half_percent_units', 'partial_1_five_percent_units', 'partial_2_five_percent_units', 'hard_stop_percent')
    position_adjustment_enable = True
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    minimal_roi = {'0': 100.0}
    stoploss = -0.99
    exit_plan = 'profit_ladder_three_stage_no_ratchet'
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

    def _focused_stop_overlay(self, plan: Mapping[str, Any], state: Mapping[str, Any], current_profit: float) -> float:
        _ = current_profit
        return self._focused_hard_stop_price(plan, state)

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

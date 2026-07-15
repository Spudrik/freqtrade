from __future__ import annotations

EXIT_RESEARCH_PATH = "broad_exit_sweep"

UPDATE_HYPOTHESIS = "High-win sparse entry may be too restrictive; broaden structural timing/quality thresholds while preserving the original entry idea."



# Updated Sieve2 rescue clone from sieve2_ladder_long_sup_hold; entry-only research variant.



from datetime import datetime

import os

from typing import Any



import numpy as np

import pandas as pd

from pandas import DataFrame, Series

from freqtrade.exchange import timeframe_to_minutes

from freqtrade.strategy import BooleanParameter, CategoricalParameter, DecimalParameter, IntParameter, IStrategy, stoploss_from_absolute

from user_data.Indicators.complex_volume_profile import add_volume_profile





from user_data.strategies.sieve_guard_helpers import (

    SIEVE2_MARKET_GUARD_MODES,

    SIEVE2_VP_GUARD_MODES,

    add_sieve2_guard_indicators,

    apply_sieve2_optional_guards,

)



LEVEL_LOOKBACK_CHOICES = [48, 96, 168, 336]

LOCAL_LOOKBACK_CHOICES = [6, 12, 24, 48]

VOLUME_WINDOW_CHOICES = [12, 24, 48, 72]

D1_LEVEL_LOOKBACK_CHOICES = [20, 50, 100, 200]

D1_VOLUME_WINDOW_CHOICES = [14, 28, 56]

ENABLE_CHOICES = ["off", "on"]

D1_STRUCTURE_MODE_CHOICES = ["off", "near", "aligned"]

GUARD_MODE_CHOICES = ["direction", "score", "context", "score_or_context", "balance"]

PRICE_SOURCE_CHOICES = ["close", "hl2", "hlc3", "ohlc4"]

VP_COLUMNS = [

    "score_long",

    "score_short",

    "context_score_bull",

    "context_score_bear",

    "context_score_balance",

    "market_context",

]







HYPEROPT_PARAM_ENV = "HYBRID_RECOVERY_HYPEROPT_PARAMS"





def split_hyperopt_tokens(value: str | None) -> set[str]:

    if not value:

        return set()

    normalized = value.replace(";", ",").replace("|", ",").replace(" ", ",")

    return {token.strip() for token in normalized.split(",") if token.strip()}





def is_parameter_object(value: Any) -> bool:

    return bool(value is not None and value.__class__.__name__.endswith("Parameter"))





def apply_explicit_hyperopt_surface(strategy_cls: type) -> None:

    selected = split_hyperopt_tokens(os.environ.get(HYPEROPT_PARAM_ENV))

    if not selected:

        return

    for name in dir(strategy_cls):

        value = getattr(strategy_cls, name, None)

        if is_parameter_object(value):

            value.optimize = str(name) in selected



def _pct_env(name: str, default_ratio: float) -> float:

    raw = str(os.environ.get(name) or "").strip()

    if not raw:

        return float(default_ratio)

    try:

        return max(0.0, float(raw)) / 100.0

    except ValueError:

        return float(default_ratio)





def _tag(param: Any, mode: str) -> Any:

    setattr(param, "batch_tags", ("family:entries", f"mode:{mode}"))

    return param







def tagged_exit_parameter(param):

    setattr(param, "batch_tags", ("family:exits", "mode:sieve3_exit"))

    return param





def s3_active_exit_parameters(family: str, mode_name: str = "exit_path_mode") -> tuple[str, ...]:
    _ = family
    return (
        mode_name,
        "fixed_tp_pct",
        "fixed_sl_pct",
        "partial_1_profit",
        "partial_2_profit",
        "partial_1_fraction",
        "partial_2_fraction",
        "breakeven_trigger",
        "breakeven_offset",
        "trailing_activation",
        "trailing_distance",
        "indicator_min_profit",
        "indicator_max_profit",
        "indicator_near_pct",
        "indicator_stop_buffer",
        "structural_min_profit",
        "structural_max_profit",
        "structural_proximity_pct",
        "structural_stop_buffer",
        "guard_tighten_buffer",
        "guard_profit_floor",
        "time_stop_candles",
        "time_stop_min_profit",
    )


def apply_s3_branch_surface(strategy_cls: type) -> None:
    family = str(getattr(strategy_cls, "S3_BRANCH_FAMILY", getattr(strategy_cls, "EXIT_FAMILY", ""))).lower()
    mode_name = "exit_path_mode" if hasattr(strategy_cls, "exit_path_mode") else "exit_mode"
    active = set(s3_active_exit_parameters(family, mode_name))
    for name in (
        "exit_path_mode", "exit_mode", "fixed_tp_pct", "fixed_sl_pct", "partial_1_profit", "partial_2_profit",
        "partial_1_fraction", "partial_2_fraction", "breakeven_trigger", "breakeven_offset",
        "trailing_activation", "trailing_distance", "indicator_min_profit", "indicator_max_profit",
        "indicator_near_pct", "indicator_stop_buffer", "structural_min_profit", "structural_max_profit",
        "structural_proximity_pct", "structural_stop_buffer", "guard_tighten_buffer", "guard_profit_floor",
        "time_stop_candles", "time_stop_min_profit",
    ):
        value = getattr(strategy_cls, name, None)
        if is_parameter_object(value):
            value.optimize = name in active

class Sieve3ExitGuardFailureFromLoosenLadderLongSupHoldBroaderTrigger(IStrategy):

    SIEVE2_FUNDAMENTAL_REWORK = "20260605_nonprofitable_loosen_sparse"

    SOURCE_RESULT_BATCH = "20260603T101611_entry_sieve2_guard_revised_weak_unrun"

    REWORK_HYPOTHESIS = "Entry-only rerun after non-profitable revised-guard result: loosen_sparse; widen guard thresholds and adjust entry-strength search space."

    SIEVE2_ALWAYS_ON_GUARDS = True

    RESEARCH_PATH = "guard_revised_weak_unrun"

    INTERFACE_VERSION = 3

    can_short = False

    timeframe = "1h"

    startup_candle_count = 336

    process_only_new_candles = True



    minimal_roi = {"0": 100.0}

    stoploss = -0.99

    use_exit_signal = False

    use_custom_stoploss = True

    position_adjustment_enable = True

    max_entry_position_adjustment = 0

    trailing_stop = False

    ignore_roi_if_entry_signal = False



    use_sieve2_vp_guard = BooleanParameter(default=True, space="buy", optimize=False, load=True)

    sieve2_vp_guard_mode = CategoricalParameter(SIEVE2_VP_GUARD_MODES, default="score_or_context", space="buy", optimize=True, load=True)

    sieve2_vp_window = CategoricalParameter([48, 96, 168], default=96, space="buy", optimize=False, load=True)

    sieve2_vp_bins = CategoricalParameter([24, 36, 48], default=36, space="buy", optimize=False, load=True)

    sieve2_vp_score_min = CategoricalParameter([-20.1, -10, -1, -0.25, 0, 0.05, 0.15, 0.25, 0.35, 0.5, 0.75, 1, 2, 10], default=0.15, space="buy", optimize=True, load=True)

    sieve2_vp_context_min = CategoricalParameter([-20.1, -10, -1, -0.25, 0, 0.05, 0.15, 0.28, 0.4, 0.6, 0.85, 1, 2, 10], default=0.15, space="buy", optimize=True, load=True)

    use_sieve2_market_guard = BooleanParameter(default=True, space="buy", optimize=False, load=True)

    sieve2_market_guard_mode = CategoricalParameter(SIEVE2_MARKET_GUARD_MODES, default="pressure_or_trend", space="buy", optimize=True, load=True)

    sieve2_market_window = CategoricalParameter([12, 24, 48, 96], default=24, space="buy", optimize=False, load=True)

    sieve2_market_pressure_min = CategoricalParameter([-20.1, -10, -1, -0.25, 0, 0.03, 0.07, 0.12, 0.18, 0.25, 0.4, 0.75, 1, 10], default=0.03, space="buy", optimize=True, load=True)

    sieve2_market_trend_min = CategoricalParameter([-20.1, -10, -3, -1, -0.25, 0, 0.15, 0.25, 0.5, 0.9, 1.5, 3, 10], default=0.15, space="buy", optimize=True, load=True)

    sieve2_rs_benchmark_pair = CategoricalParameter(["BTC/USDT:USDT", "ETH/USDT:USDT"], default="BTC/USDT:USDT", space="buy", optimize=False, load=True)

    sieve2_rs_score_min = CategoricalParameter([-2.1, -1, 0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.1, 2], default=0.30, space="buy", optimize=True, load=True)



    ENTRY_TAG = "long_sup_hold"

    ENTRY_SIDE = "long"

    ENTRY_KIND = "long_sup_hold"

    MODE = "entry_long_sup_hold"

    CONFIRMATION_PROFILE = "long_support_hold"

    BREATHING_PROFILE = "none"



    level_lookback = _tag(CategoricalParameter([96, 168, 336], default=168, space="buy", optimize=True, load=True), MODE)

    local_lookback = _tag(CategoricalParameter([12, 24, 48], default=48, space="buy", optimize=True, load=True), MODE)

    d1_level_lookback = _tag(CategoricalParameter(D1_LEVEL_LOOKBACK_CHOICES, default=50, space="buy", optimize=False, load=True), MODE)

    volume_window = _tag(CategoricalParameter([12, 24, 48], default=24, space="buy", optimize=False, load=True), MODE)

    d1_volume_window = _tag(CategoricalParameter(D1_VOLUME_WINDOW_CHOICES, default=28, space="buy", optimize=False, load=True), MODE)

    confirm_bars = _tag(CategoricalParameter([3, 6, 12], default=3, space="buy", optimize=True, load=True), MODE)

    zone_pct = _tag(CategoricalParameter([0.006, 0.012, 0.024], default=0.012, space="buy", optimize=True, load=True), MODE)

    trigger_buffer_pct = _tag(CategoricalParameter([0.0, 0.003, 0.006, 0.01], default=0.003, space="buy", optimize=True, load=True), MODE)

    h1_rvol_enable = _tag(CategoricalParameter(['off', 'on'], default='on', space="buy", optimize=True, load=True), MODE)

    h1_pressure_enable = _tag(CategoricalParameter(['off', 'on'], default='on', space="buy", optimize=True, load=True), MODE)

    h1_event_enable = _tag(CategoricalParameter(['off', 'on'], default='on', space="buy", optimize=True, load=True), MODE)

    d1_rvol_enable = _tag(CategoricalParameter(ENABLE_CHOICES, default="on", space="buy", optimize=False, load=True), MODE)

    d1_pressure_enable = _tag(CategoricalParameter(ENABLE_CHOICES, default="on", space="buy", optimize=False, load=True), MODE)

    d1_structure_mode = _tag(CategoricalParameter(['off', 'near', 'aligned'], default='off', space="buy", optimize=True, load=True), MODE)

    h1_rvol_min = _tag(CategoricalParameter([1.0, 1.1, 1.2, 1.6], default=1.1, space="buy", optimize=False, load=True), MODE)

    h1_pressure_min = _tag(CategoricalParameter([0.1, 0.3, 0.6], default=0.3, space="buy", optimize=False, load=True), MODE)

    d1_rvol_min = _tag(DecimalParameter(0.7, 2.2, decimals=1, default=1.0, space="buy", optimize=False, load=True), MODE)

    d1_pressure_min = _tag(DecimalParameter(0.0, 1.2, decimals=1, default=0.2, space="buy", optimize=False, load=True), MODE)



    use_vp_4h_guard = _tag(CategoricalParameter(ENABLE_CHOICES, default="off", space="buy", optimize=False, load=True), MODE)

    vp_4h_guard_mode = _tag(CategoricalParameter(GUARD_MODE_CHOICES, default="score_or_context", space="buy", optimize=False, load=True), MODE)

    vp_4h_window = _tag(CategoricalParameter([12, 24, 48, 72, 96], default=48, space="buy", optimize=False, load=True), MODE)

    vp_4h_bins = _tag(CategoricalParameter([16, 24, 36, 48, 64], default=36, space="buy", optimize=False, load=True), MODE)

    vp_4h_score_min = _tag(DecimalParameter(0.00, 1.00, decimals=2, default=0.25, space="buy", optimize=False, load=True), MODE)

    vp_4h_context_min = _tag(DecimalParameter(0.00, 1.00, decimals=2, default=0.28, space="buy", optimize=False, load=True), MODE)

    use_vp_1d_guard = _tag(CategoricalParameter(ENABLE_CHOICES, default="off", space="buy", optimize=False, load=True), MODE)

    vp_1d_guard_mode = _tag(CategoricalParameter(GUARD_MODE_CHOICES, default="context", space="buy", optimize=False, load=True), MODE)

    vp_1d_window = _tag(CategoricalParameter([10, 20, 30, 45, 60, 84], default=30, space="buy", optimize=False, load=True), MODE)

    vp_1d_bins = _tag(CategoricalParameter([16, 24, 36, 48, 64], default=36, space="buy", optimize=False, load=True), MODE)

    vp_1d_score_min = _tag(DecimalParameter(0.00, 1.00, decimals=2, default=0.25, space="buy", optimize=False, load=True), MODE)

    vp_1d_context_min = _tag(DecimalParameter(0.00, 1.00, decimals=2, default=0.28, space="buy", optimize=False, load=True), MODE)

    vp_guard_value_area_pct = _tag(DecimalParameter(0.55, 0.85, decimals=2, default=0.70, space="buy", optimize=False, load=True), MODE)

    vp_guard_price_source = _tag(CategoricalParameter(PRICE_SOURCE_CHOICES, default="hlc3", space="buy", optimize=False, load=True), MODE)

    vp_guard_node_near_pct = _tag(DecimalParameter(0.002, 0.030, decimals=3, default=0.010, space="buy", optimize=False, load=True), MODE)

    vp_guard_pressure_delta_min = _tag(DecimalParameter(0.00, 0.35, decimals=2, default=0.05, space="buy", optimize=False, load=True), MODE)



    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:

        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs

        return 1.0



    def informative_pairs(self) -> list[tuple[str, str]]:

        if not getattr(self, "dp", None):

            return []

        whitelist = self.dp.current_whitelist()
        return [(pair, "1d") for pair in whitelist] + [(pair, "4h") for pair in whitelist]



    @staticmethod

    def _num(value: Series) -> Series:

        return pd.to_numeric(value, errors="coerce")



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

        return str(value or "").lower() == "on"



    @staticmethod

    def _bool(mask: Series, index: pd.Index) -> Series:

        return pd.Series(mask, index=index).fillna(False).astype(bool)



    @staticmethod

    def _recent(mask: Series, bars: int) -> Series:

        return mask.fillna(False).astype("int8").shift(1).rolling(max(1, int(bars)), min_periods=1).max().gt(0)



    def _add_volume_pressure(self, dataframe: DataFrame, prefix: str, volume_windows: list[int]) -> DataFrame:

        close = self._num(dataframe["close"])

        open_ = self._num(dataframe["open"])

        high = self._num(dataframe["high"])

        low = self._num(dataframe["low"])

        volume = self._num(dataframe["volume"]).clip(lower=0.0).fillna(0.0)

        candle_range = (high - low).clip(lower=0.0)

        close_location = ((self._safe_div(close - low, candle_range) * 2.0) - 1.0).clip(-1.0, 1.0)

        body_pressure = self._safe_div(close - open_, candle_range).clip(-1.0, 1.0)

        delta_pressure = ((close_location.fillna(0.0) + body_pressure.fillna(0.0)) / 2.0).clip(-1.0, 1.0)

        delta = volume * delta_pressure

        dataframe[f"{prefix}_close_location"] = close_location

        dataframe[f"{prefix}_delta_pressure"] = delta_pressure

        dataframe[f"{prefix}_delta_zscore"] = self._zscore(delta, 96 if prefix == "h1" else 56)

        dataframe[f"{prefix}_cvd"] = delta.cumsum()

        pressure_window = 48 if prefix == "h1" else 28

        dataframe[f"{prefix}_cvd_trend"] = delta.rolling(pressure_window, min_periods=max(4, pressure_window // 4)).sum()

        for window in volume_windows:

            avg_volume = volume.shift(1).rolling(int(window), min_periods=max(2, int(window) // 4)).mean().replace(0.0, np.nan)

            dataframe[f"{prefix}_rvol_{window}"] = volume / avg_volume

        return dataframe



    def _daily_indicators(self, daily: DataFrame) -> DataFrame:

        frame = daily.copy()

        close = self._num(frame["close"])

        high = self._num(frame["high"])

        low = self._num(frame["low"])

        frame = self._add_volume_pressure(frame, "d1", D1_VOLUME_WINDOW_CHOICES)

        for lookback in D1_LEVEL_LOOKBACK_CHOICES:

            min_periods = max(5, int(lookback) // 4)

            frame[f"d1_resistance_{lookback}"] = high.shift(1).rolling(int(lookback), min_periods=min_periods).max()

            frame[f"d1_support_{lookback}"] = low.shift(1).rolling(int(lookback), min_periods=min_periods).min()

        frame["d1_trend_up"] = close.gt(close.shift(1))

        frame["d1_trend_down"] = close.lt(close.shift(1))

        return frame



    def _merge_daily_context(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if not getattr(self, "dp", None) or "date" not in dataframe.columns:

            return dataframe

        daily = self.dp.get_pair_dataframe(pair=metadata.get("pair", ""), timeframe="1d")

        if daily is None or daily.empty or "date" not in daily.columns:

            return dataframe

        daily = self._daily_indicators(daily)

        keep = ["date", "d1_close_location", "d1_delta_zscore", "d1_cvd_trend", "d1_trend_up", "d1_trend_down"]

        for window in D1_VOLUME_WINDOW_CHOICES:

            keep.append(f"d1_rvol_{window}")

        for lookback in D1_LEVEL_LOOKBACK_CHOICES:

            keep.extend([f"d1_resistance_{lookback}", f"d1_support_{lookback}"])

        informative = daily[[column for column in keep if column in daily.columns]].copy()

        informative["_sieve_d1_key"] = pd.to_datetime(informative["date"], utc=True).dt.tz_convert(None) + pd.Timedelta(days=1)

        informative = informative.drop(columns=["date"]).sort_values("_sieve_d1_key")

        base = dataframe.copy()

        base["_sieve_order"] = np.arange(len(base))

        base["_sieve_h1_key"] = pd.to_datetime(base["date"], utc=True).dt.tz_convert(None)

        merged = pd.merge_asof(base.sort_values("_sieve_h1_key"), informative, left_on="_sieve_h1_key", right_on="_sieve_d1_key", direction="backward")

        merged = merged.sort_values("_sieve_order").drop(columns=["_sieve_order", "_sieve_h1_key", "_sieve_d1_key"], errors="ignore")

        merged.index = dataframe.index

        return merged



    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        close = self._num(dataframe["close"])

        high = self._num(dataframe["high"])

        low = self._num(dataframe["low"])

        dataframe["sieve_prev_close"] = close.shift(1)

        for lookback in LEVEL_LOOKBACK_CHOICES:

            min_periods = max(2, int(lookback) // 4)

            dataframe[f"sieve_resistance_{lookback}"] = high.shift(1).rolling(int(lookback), min_periods=min_periods).max()

            dataframe[f"sieve_support_{lookback}"] = low.shift(1).rolling(int(lookback), min_periods=min_periods).min()

        for lookback in LOCAL_LOOKBACK_CHOICES:

            dataframe[f"sieve_local_high_{lookback}"] = high.shift(1).rolling(int(lookback), min_periods=2).max()

            dataframe[f"sieve_local_low_{lookback}"] = low.shift(1).rolling(int(lookback), min_periods=2).min()

        dataframe = self._add_volume_pressure(dataframe, "h1", VOLUME_WINDOW_CHOICES)

        dataframe["sieve_trend_up"] = close.gt(close.shift(1))

        dataframe["sieve_trend_down"] = close.lt(close.shift(1))

        dataframe = self._merge_daily_context(dataframe, metadata)

        dataframe = self._merge_informative_vp(dataframe, metadata, "4h", "vp4h", int(self.vp_4h_window.value), int(self.vp_4h_bins.value))

        dataframe = self._merge_informative_vp(dataframe, metadata, "1d", "vp1d", int(self.vp_1d_window.value), int(self.vp_1d_bins.value))

        dataframe = add_sieve2_guard_indicators(self, dataframe, metadata)

        return dataframe



    def _side_volume_guard(self, dataframe: DataFrame, side: str, local: int, d1_window: int) -> Series:

        index = dataframe.index

        h1_rvol = self._num(dataframe[f"h1_rvol_{int(self.volume_window.value)}"])

        h1_delta = self._num(dataframe["h1_delta_zscore"])

        h1_cvd = self._num(dataframe["h1_cvd_trend"])

        h1_loc = self._num(dataframe["h1_close_location"])

        d1_rvol = self._num(dataframe.get(f"d1_rvol_{d1_window}", pd.Series(index=index, dtype="float64")))

        d1_delta = self._num(dataframe.get("d1_delta_zscore", pd.Series(index=index, dtype="float64")))

        d1_cvd = self._num(dataframe.get("d1_cvd_trend", pd.Series(index=index, dtype="float64")))

        if side == "long":

            h1_pressure = h1_delta.ge(float(self.h1_pressure_min.value)) & h1_cvd.ge(0.0) & h1_loc.ge(-0.15)

            d1_pressure = d1_delta.ge(float(self.d1_pressure_min.value)) | d1_cvd.ge(0.0)

            event = self._h1_event_guard(dataframe, "long", local)

        else:

            h1_pressure = h1_delta.le(-float(self.h1_pressure_min.value)) & h1_cvd.le(0.0) & h1_loc.le(0.15)

            d1_pressure = d1_delta.le(-float(self.d1_pressure_min.value)) | d1_cvd.le(0.0)

            event = self._h1_event_guard(dataframe, "short", local)

        guard = pd.Series(True, index=index, dtype="bool")

        if self._enabled(self.h1_rvol_enable.value):

            guard &= h1_rvol.ge(float(self.h1_rvol_min.value)).fillna(False)

        if self._enabled(self.h1_pressure_enable.value):

            guard &= h1_pressure.fillna(False)

        if self._enabled(self.h1_event_enable.value):

            guard &= event.fillna(False)

        if self._enabled(self.d1_rvol_enable.value):

            guard &= d1_rvol.ge(float(self.d1_rvol_min.value)).fillna(False)

        if self._enabled(self.d1_pressure_enable.value):

            guard &= d1_pressure.fillna(False)

        return guard



    def _h1_event_guard(self, dataframe: DataFrame, side: str, local: int) -> Series:

        kind = self.ENTRY_KIND

        close = self._num(dataframe["close"])

        high = self._num(dataframe["high"])

        low = self._num(dataframe["low"])

        local_high = self._num(dataframe[f"sieve_local_high_{local}"])

        local_low = self._num(dataframe[f"sieve_local_low_{local}"])

        close_loc = self._num(dataframe["h1_close_location"])

        trigger = float(self.trigger_buffer_pct.value)

        long_breakout = close.gt(local_high * (1.0 + trigger)) & close_loc.ge(0.20)

        short_breakout = close.lt(local_low * (1.0 - trigger)) & close_loc.le(-0.20)

        long_reclaim = low.lt(local_low * (1.0 - trigger)) & close.gt(local_low) & close_loc.ge(0.0)

        short_reject = high.gt(local_high * (1.0 + trigger)) & close.lt(local_high) & close_loc.le(0.0)

        if side == "long":

            if "break" in kind or "retest" in kind:

                return long_breakout | self._recent(long_breakout, int(self.confirm_bars.value))

            return long_reclaim | close_loc.ge(0.15)

        if "break" in kind or "retest" in kind:

            return short_breakout | self._recent(short_breakout, int(self.confirm_bars.value))

        return short_reject | close_loc.le(-0.15)



    def _daily_structure_guard(self, dataframe: DataFrame, side: str, d1_level: int, zone: float, trigger: float) -> Series:

        mode = str(self.d1_structure_mode.value)

        if mode == "off":

            return pd.Series(True, index=dataframe.index, dtype="bool")

        close = self._num(dataframe["close"])

        high = self._num(dataframe["high"])

        low = self._num(dataframe["low"])

        d1_resistance = self._num(dataframe.get(f"d1_resistance_{d1_level}", pd.Series(index=dataframe.index, dtype="float64")))

        d1_support = self._num(dataframe.get(f"d1_support_{d1_level}", pd.Series(index=dataframe.index, dtype="float64")))

        wide_zone = max(zone * 2.0, trigger)

        kind = self.ENTRY_KIND

        if kind in {"long_res_break", "long_res_retest_hold"}:

            near = close.ge(d1_resistance * (1.0 - wide_zone))

            aligned = close.gt(d1_resistance * (1.0 + trigger))

        elif kind in {"long_sup_hold", "long_sup_reclaim", "long_trend_pullback"}:

            near = low.le(d1_support * (1.0 + wide_zone))

            aligned = close.gt(d1_support * (1.0 + trigger))

        elif kind in {"short_res_fail", "short_res_reclaim", "short_trend_pullback"}:

            near = high.ge(d1_resistance * (1.0 - wide_zone))

            aligned = close.lt(d1_resistance * (1.0 - trigger))

        else:

            near = close.le(d1_support * (1.0 + wide_zone))

            aligned = close.lt(d1_support * (1.0 - trigger))

        return (aligned if mode == "aligned" else near).fillna(False)



    def _effective_zone(self, zone: float) -> float:

        if str(getattr(self, "BREATHING_PROFILE", "none")) == "loose_support_reclaim":

            return min(zone * 1.50, 0.060)

        return zone



    def _effective_trigger(self, trigger: float) -> float:

        if str(getattr(self, "BREATHING_PROFILE", "none")) == "loose_support_reclaim":

            return max(trigger * 0.60, 0.0)

        if str(getattr(self, "CONFIRMATION_PROFILE", "none")) != "none":

            return min(trigger * 1.25, 0.030)

        return trigger



    def _effective_confirm(self, confirm: int) -> int:

        if str(getattr(self, "BREATHING_PROFILE", "none")) == "loose_support_reclaim":

            return min(max(1, int(confirm * 1.50)), 72)

        if str(getattr(self, "CONFIRMATION_PROFILE", "none")) != "none":

            return min(max(1, int(confirm)), 12)

        return confirm



    def _adaptive_confirmation_guard(self, dataframe: DataFrame, side: str, local: int, level: int, zone: float, trigger: float) -> Series:

        profile = str(getattr(self, "CONFIRMATION_PROFILE", "none"))

        if profile == "none":

            return pd.Series(True, index=dataframe.index, dtype="bool")



        close = self._num(dataframe["close"])

        prev_close = self._num(dataframe["sieve_prev_close"])

        high = self._num(dataframe["high"])

        low = self._num(dataframe["low"])

        resistance = self._num(dataframe[f"sieve_resistance_{level}"])

        support = self._num(dataframe[f"sieve_support_{level}"])

        local_high = self._num(dataframe[f"sieve_local_high_{local}"])

        local_low = self._num(dataframe[f"sieve_local_low_{local}"])

        close_loc = self._num(dataframe["h1_close_location"])

        h1_rvol = self._num(dataframe[f"h1_rvol_{int(self.volume_window.value)}"])

        h1_delta = self._num(dataframe["h1_delta_zscore"])

        h1_cvd = self._num(dataframe["h1_cvd_trend"])

        trend_up = pd.Series(dataframe["sieve_trend_up"], index=dataframe.index).fillna(False).astype(bool)

        trend_down = pd.Series(dataframe["sieve_trend_down"], index=dataframe.index).fillna(False).astype(bool)

        has_d1_up = "d1_trend_up" in dataframe.columns

        has_d1_down = "d1_trend_down" in dataframe.columns

        d1_up = pd.Series(dataframe.get("d1_trend_up", True), index=dataframe.index).fillna(True).astype(bool)

        d1_down = pd.Series(dataframe.get("d1_trend_down", True), index=dataframe.index).fillna(True).astype(bool)

        d1_not_up = (~d1_up) if has_d1_up else pd.Series(True, index=dataframe.index, dtype="bool")

        d1_not_down = (~d1_down) if has_d1_down else pd.Series(True, index=dataframe.index, dtype="bool")



        min_rvol = max(float(self.h1_rvol_min.value), 1.20)

        min_pressure = max(float(self.h1_pressure_min.value), 0.30)

        long_pressure = h1_rvol.ge(min_rvol) & h1_delta.ge(min_pressure) & h1_cvd.ge(0.0) & close_loc.ge(0.25)

        short_pressure = h1_rvol.ge(min_rvol) & h1_delta.le(-min_pressure) & h1_cvd.le(0.0) & close_loc.le(-0.25)



        long_breakout = close.gt(local_high * (1.0 + trigger)) & close.gt(resistance * (1.0 + trigger)) & close_loc.ge(0.35)

        short_breakdown = close.lt(local_low * (1.0 - trigger)) & close.lt(support * (1.0 - trigger)) & close_loc.le(-0.35)

        long_support_response = low.le(support * (1.0 + zone)) & close.gt(support * (1.0 + trigger)) & close_loc.ge(0.25)

        short_resistance_response = high.ge(resistance * (1.0 - zone)) & close.lt(resistance * (1.0 - trigger)) & close_loc.le(-0.25)



        if profile == "long_resistance_break":

            return (long_pressure & long_breakout & trend_up & d1_up).fillna(False)

        if profile == "long_resistance_retest":

            retest_hold = low.le(resistance * (1.0 + zone)) & close.gt(resistance * (1.0 + trigger * 0.50)) & close_loc.ge(0.25)

            return (long_pressure & retest_hold & trend_up & d1_up).fillna(False)

        if profile == "long_support_hold":

            return (long_pressure & long_support_response & ~trend_down & d1_not_down).fillna(False)

        if profile == "long_trend_pullback":

            trend_reclaim = close.gt(prev_close) & close_loc.ge(0.25)

            return (long_pressure & trend_reclaim & trend_up & d1_not_down).fillna(False)

        if profile == "short_resistance_fail":

            return (short_pressure & short_resistance_response & ~trend_up & d1_not_up).fillna(False)

        if profile == "short_support_break":

            return (short_pressure & short_breakdown & trend_down & d1_down).fillna(False)

        if profile == "short_support_retest":

            retest_reject = high.ge(support * (1.0 - zone)) & close.lt(support * (1.0 - trigger * 0.50)) & close_loc.le(-0.25)

            return (short_pressure & retest_reject & trend_down & d1_down).fillna(False)

        return (long_pressure if side == "long" else short_pressure).fillna(False)





    def _merge_informative_vp(self, dataframe: DataFrame, metadata: dict, timeframe: str, prefix: str, window: int, bins: int) -> DataFrame:

        if "date" not in dataframe.columns or not getattr(self, "dp", None):

            return dataframe

        pair = str(metadata.get("pair") or "")

        if not pair:

            return dataframe

        informative = self.dp.get_pair_dataframe(pair=pair, timeframe=timeframe)

        if informative is None or informative.empty or "date" not in informative.columns:

            return dataframe

        informative = add_volume_profile(

            informative.copy(),

            window=window,

            bins=bins,

            value_area_pct=float(self.vp_guard_value_area_pct.value),

            price_source=str(self.vp_guard_price_source.value),

            pressure_delta_min=float(self.vp_guard_pressure_delta_min.value),

            node_near_pct=float(self.vp_guard_node_near_pct.value),

            prefix=prefix,

        )

        merge_columns = [f"{prefix}_{name}" for name in VP_COLUMNS if f"{prefix}_{name}" in informative.columns]

        if not merge_columns:

            return dataframe

        minutes = timeframe_to_minutes(timeframe)

        inf = informative[["date", *merge_columns]].copy().sort_values("date")

        inf["date_merge"] = inf["date"] + pd.to_timedelta(minutes, unit="m")

        base = dataframe.reset_index().rename(columns={"index": "__row_index"}).sort_values("date")

        merged = pd.merge_asof(

            base,

            inf[["date_merge", *merge_columns]].sort_values("date_merge"),

            left_on="date",

            right_on="date_merge",

            direction="backward",

        ).sort_values("__row_index")

        for column in merge_columns:

            dataframe[column] = merged[column].to_numpy()

        return dataframe



    def _score_guard(self, dataframe: DataFrame, prefix: str, side: str, score_min: float) -> Series:

        score = self._num(dataframe.get(f"{prefix}_score_{side}", pd.Series(index=dataframe.index, dtype="float64")))

        opposite_side = "short" if side == "long" else "long"

        opposite = self._num(dataframe.get(f"{prefix}_score_{opposite_side}", pd.Series(index=dataframe.index, dtype="float64")))

        return score.ge(score_min) & score.ge(opposite)



    def _context_guard(self, dataframe: DataFrame, prefix: str, side: str, context_min: float) -> Series:

        if side == "long":

            context = self._num(dataframe.get(f"{prefix}_context_score_bull", pd.Series(index=dataframe.index, dtype="float64")))

            opposite = self._num(dataframe.get(f"{prefix}_context_score_bear", pd.Series(index=dataframe.index, dtype="float64")))

            market_ok = self._num(dataframe.get(f"{prefix}_market_context", pd.Series(index=dataframe.index, dtype="float64"))).ge(0)

        else:

            context = self._num(dataframe.get(f"{prefix}_context_score_bear", pd.Series(index=dataframe.index, dtype="float64")))

            opposite = self._num(dataframe.get(f"{prefix}_context_score_bull", pd.Series(index=dataframe.index, dtype="float64")))

            market_ok = self._num(dataframe.get(f"{prefix}_market_context", pd.Series(index=dataframe.index, dtype="float64"))).le(0)

        return context.ge(context_min) & context.ge(opposite) & market_ok



    def _vp_guard(self, dataframe: DataFrame, prefix: str, side: str, mode: str, score_min: float, context_min: float) -> Series:

        score_ok = self._score_guard(dataframe, prefix, side, score_min)

        context_ok = self._context_guard(dataframe, prefix, side, context_min)

        balance = self._num(dataframe.get(f"{prefix}_context_score_balance", pd.Series(index=dataframe.index, dtype="float64")))

        if mode == "score":

            return score_ok

        if mode == "context":

            return context_ok

        if mode == "score_or_context":

            return score_ok | context_ok

        if mode == "balance":

            return balance.ge(context_min)

        market = self._num(dataframe.get(f"{prefix}_market_context", pd.Series(index=dataframe.index, dtype="float64")))

        return market.ge(0) if side == "long" else market.le(0)



    def _entry_mask(self, dataframe: DataFrame) -> Series:

        level = int(self.level_lookback.value)

        local = int(self.local_lookback.value)

        d1_level = int(self.d1_level_lookback.value)

        d1_window = int(self.d1_volume_window.value)

        zone = self._effective_zone(float(self.zone_pct.value))

        trigger = self._effective_trigger(float(self.trigger_buffer_pct.value))

        confirm = self._effective_confirm(int(self.confirm_bars.value))

        close = self._num(dataframe["close"])

        high = self._num(dataframe["high"])

        low = self._num(dataframe["low"])

        prev_close = self._num(dataframe["sieve_prev_close"])

        resistance = self._num(dataframe[f"sieve_resistance_{level}"])

        support = self._num(dataframe[f"sieve_support_{level}"])

        local_high = self._num(dataframe[f"sieve_local_high_{local}"])

        local_low = self._num(dataframe[f"sieve_local_low_{local}"])

        trend_up = pd.Series(dataframe["sieve_trend_up"], index=dataframe.index).fillna(False).astype(bool)

        trend_down = pd.Series(dataframe["sieve_trend_down"], index=dataframe.index).fillna(False).astype(bool)

        kind = self.ENTRY_KIND

        if kind == "long_res_break":

            line = resistance * (1.0 + trigger)

            structure = close.gt(line) & prev_close.le(line) & local_high.ge(resistance * (1.0 - zone))

            side = "long"

        elif kind == "long_sup_hold":

            structure = low.le(support * (1.0 + zone)) & local_low.le(support * (1.0 + zone)) & close.gt(support * (1.0 + trigger))

            side = "long"

        elif kind == "long_sup_reclaim":

            structure = low.lt(support * (1.0 - trigger)) & local_low.le(support * (1.0 + zone)) & close.gt(support * (1.0 + trigger))

            side = "long"

        elif kind == "long_res_retest_hold":

            break_mask = close.gt(resistance * (1.0 + trigger))

            structure = self._recent(break_mask, confirm) & low.le(resistance * (1.0 + zone)) & local_low.le(resistance * (1.0 + zone)) & close.ge(resistance)

            side = "long"

        elif kind == "short_res_fail":

            structure = high.ge(resistance * (1.0 - zone)) & local_high.ge(resistance * (1.0 - zone)) & close.lt(resistance * (1.0 - trigger))

            side = "short"

        elif kind == "short_res_reclaim":

            structure = high.gt(resistance * (1.0 + trigger)) & local_high.ge(resistance * (1.0 - zone)) & close.lt(resistance * (1.0 - trigger))

            side = "short"

        elif kind == "short_sup_break":

            line = support * (1.0 - trigger)

            structure = close.lt(line) & prev_close.ge(line) & local_low.le(support * (1.0 + zone))

            side = "short"

        elif kind == "short_sup_retest_reject":

            break_mask = close.lt(support * (1.0 - trigger))

            structure = self._recent(break_mask, confirm) & high.ge(support * (1.0 - zone)) & local_high.ge(support * (1.0 - zone)) & close.le(support)

            side = "short"

        elif kind == "long_trend_pullback":

            structure = trend_up & low.le(support * (1.0 + zone)) & close.gt(support * (1.0 + trigger)) & close.gt(prev_close)

            side = "long"

        elif kind == "short_trend_pullback":

            structure = trend_down & high.ge(resistance * (1.0 - zone)) & close.lt(resistance * (1.0 - trigger)) & close.lt(prev_close)

            side = "short"

        else:

            structure = pd.Series(False, index=dataframe.index)

            side = self.ENTRY_SIDE

        confirmation = self._adaptive_confirmation_guard(dataframe, side, local, level, zone, trigger)

        mask = structure & confirmation & self._side_volume_guard(dataframe, side, local, d1_window) & self._daily_structure_guard(dataframe, side, d1_level, zone, trigger)

        if self._enabled(self.use_vp_4h_guard.value):

            mask &= self._vp_guard(dataframe, "vp4h", side, str(self.vp_4h_guard_mode.value), float(self.vp_4h_score_min.value), float(self.vp_4h_context_min.value))

        if self._enabled(self.use_vp_1d_guard.value):

            mask &= self._vp_guard(dataframe, "vp1d", side, str(self.vp_1d_guard_mode.value), float(self.vp_1d_score_min.value), float(self.vp_1d_context_min.value))

        return self._bool(mask, dataframe.index)



    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        _ = metadata

        dataframe["enter_long"] = 0

        dataframe["enter_short"] = 0

        dataframe["enter_tag"] = ""

        mask = self._entry_mask(dataframe)

        mask = apply_sieve2_optional_guards(self, dataframe, mask, getattr(self, "ENTRY_SIDE", globals().get("SIDE", "long")))

        dataframe[f"plot_{self.ENTRY_TAG}"] = mask.astype(float)

        dataframe.loc[mask, "enter_long"] = 1

        dataframe.loc[mask, "enter_tag"] = self.ENTRY_TAG

        return dataframe



    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        _ = metadata

        dataframe["exit_long"] = 0

        dataframe["exit_short"] = 0

        dataframe["exit_tag"] = ""

        return dataframe



    S3_BRANCH_FAMILY = "guard_failure"

    EXIT_FAMILY = "guard_failure"

    EXIT_HYPOTHESIS = 'Exit or tighten when key entry guards degrade after entry.'

    SOURCE_ENTRY_STEM = "loosen_ladder_long_sup_hold_broader_trigger"

    SOURCE_ENTRY_CLASS = "Sieve3ExitLoosenLadderLongSupHoldBroaderTrigger"

    PRIMARY_TRIGGER = SOURCE_ENTRY_STEM

    PRIMARY_GUARD = "source_entry_guards_plus_sieve2_optional_guards"

    TARGET_PROVIDER = "explicit_s3_target_provider_order"

    INVALIDATION_PROVIDER = "source_trigger_state_plus_explicit_target_provider"

    EXIT_PARAMETER_VOCABULARY = (

        "exit_path_mode", "fixed_tp_pct", "fixed_sl_pct", "partial_1_profit", "partial_2_profit",

        "partial_1_fraction", "partial_2_fraction", "breakeven_trigger", "breakeven_offset",

        "trailing_activation", "trailing_distance", "indicator_min_profit", "indicator_max_profit",

        "indicator_near_pct", "indicator_stop_buffer", "guard_tighten_buffer", "guard_profit_floor",

        "time_stop_candles", "time_stop_min_profit",

    )

    ACTIVE_EXIT_PARAMETERS = s3_active_exit_parameters(S3_BRANCH_FAMILY, "exit_path_mode")

    BRANCH_SPLIT_RATIONALE = "Standalone source-family branch; active sell params are constrained by apply_s3_branch_surface."

    S3_TARGET_PROVIDER_COLUMNS = {

        "bos_choch": {

            "long": ("ms_pivot_high", "ms_bullish_break_level", "ms_break_level"),

            "short": ("ms_pivot_low", "ms_bearish_break_level", "ms_break_level"),

        },

        "pivot": {

            "long": ("pivot_high", "ms_pivot_high", "tlv2_pivot_high"),

            "short": ("pivot_low", "ms_pivot_low", "tlv2_pivot_low"),

        },

        "vp": {

            "long": ("vp_poc", "vp_vah", "vp_hvn", "vp_lvn", "vp_node", "vp_value_area_high"),

            "short": ("vp_poc", "vp_val", "vp_hvn", "vp_lvn", "vp_node", "vp_value_area_low"),

        },

        "tlv2": {

            "long": ("tlv2_resistance_line_rank1", "tlv2_resistance_line_rank2", "tlv2_pivot_high"),

            "short": ("tlv2_support_line_rank1", "tlv2_support_line_rank2", "tlv2_pivot_low"),

        },

        "geometry": {

            "long": ("pg2_slot_0_upper", "pg2_best_upper", "pg2_upper"),

            "short": ("pg2_slot_0_lower", "pg2_best_lower", "pg2_lower"),

        },

        "prior_level": {

            "long": ("prior_high", "prior_day_high", "prior_week_high", "prior_month_high"),

            "short": ("prior_low", "prior_day_low", "prior_week_low", "prior_month_low"),

        },

        "avwap": {

            "long": ("avwap", "anchored_vwap", "prior_avwap"),

            "short": ("avwap", "anchored_vwap", "prior_avwap"),

        },

    }



    exit_path_mode = tagged_exit_parameter(CategoricalParameter([S3_BRANCH_FAMILY, "fixed", "partial", "breakeven", "trailing", "indicator_target", "guard_tighten", "trigger_invalidation", "time_stagnation", "mixed_partial_trail"], default=S3_BRANCH_FAMILY, space="sell", optimize=True, load=True))

    fixed_tp_pct = tagged_exit_parameter(CategoricalParameter([0.015, 0.020, 0.030, 0.040, 0.050, 0.080, 0.120], default=0.030, space="sell", optimize=False, load=True))

    fixed_sl_pct = tagged_exit_parameter(CategoricalParameter([0.015, 0.020, 0.030, 0.040, 0.060, 0.080], default=0.030, space="sell", optimize=True, load=True))

    partial_1_profit = tagged_exit_parameter(CategoricalParameter([0.010, 0.015, 0.020, 0.030, 0.050], default=0.020, space="sell", optimize=False, load=True))

    partial_2_profit = tagged_exit_parameter(CategoricalParameter([0.030, 0.050, 0.080, 0.120], default=0.060, space="sell", optimize=False, load=True))

    partial_1_fraction = tagged_exit_parameter(CategoricalParameter([0.25, 0.33, 0.50, 0.66], default=0.33, space="sell", optimize=False, load=True))

    partial_2_fraction = tagged_exit_parameter(CategoricalParameter([0.25, 0.33, 0.50], default=0.33, space="sell", optimize=False, load=True))

    breakeven_trigger = tagged_exit_parameter(CategoricalParameter([0.010, 0.015, 0.020, 0.030, 0.040], default=0.020, space="sell", optimize=False, load=True))

    breakeven_offset = tagged_exit_parameter(CategoricalParameter([0.000, 0.001, 0.0025, 0.005], default=0.001, space="sell", optimize=False, load=True))

    trailing_activation = tagged_exit_parameter(CategoricalParameter([0.015, 0.020, 0.030, 0.050, 0.080], default=0.030, space="sell", optimize=False, load=True))

    trailing_distance = tagged_exit_parameter(CategoricalParameter([0.005, 0.010, 0.015, 0.025, 0.040], default=0.015, space="sell", optimize=False, load=True))

    indicator_min_profit = tagged_exit_parameter(CategoricalParameter([0.000, 0.010, 0.015, 0.020, 0.030, 0.050], default=0.015, space="sell", optimize=False, load=True))

    indicator_max_profit = tagged_exit_parameter(CategoricalParameter([0.040, 0.060, 0.080, 0.120, 0.180], default=0.120, space="sell", optimize=False, load=True))

    indicator_near_pct = tagged_exit_parameter(CategoricalParameter([0.002, 0.004, 0.006, 0.010, 0.015, 0.025], default=0.006, space="sell", optimize=False, load=True))

    indicator_stop_buffer = tagged_exit_parameter(CategoricalParameter([0.005, 0.010, 0.015, 0.025, 0.040], default=0.015, space="sell", optimize=False, load=True))

    guard_tighten_buffer = tagged_exit_parameter(CategoricalParameter([0.005, 0.010, 0.015, 0.025, 0.040], default=0.015, space="sell", optimize=False, load=True))

    guard_profit_floor = tagged_exit_parameter(CategoricalParameter([0.000, 0.005, 0.010, 0.020, 0.030], default=0.005, space="sell", optimize=False, load=True))

    time_stop_candles = tagged_exit_parameter(CategoricalParameter([6, 12, 24, 48, 96, 168], default=48, space="sell", optimize=False, load=True))

    time_stop_min_profit = tagged_exit_parameter(CategoricalParameter([-0.010, 0.000, 0.005, 0.010, 0.020], default=0.005, space="sell", optimize=False, load=True))



    S3_GUARD_FAILURE_COLUMNS = {

        "long": {

            "positive": (

                "s2vp_score_short", "s2vp_context_score_bear", "vp_score_short", "vp_context_score_bear",

                "htfvp_score_short", "htfvp_context_score_bear",

            ),

            "negative": ("s2m_pressure_ratio", "s2m_trend_z", "vp_market_context", "htfvp_market_context"),

        },

        "short": {

            "positive": (

                "s2vp_score_long", "s2vp_context_score_bull", "vp_score_long", "vp_context_score_bull",

                "htfvp_score_long", "htfvp_context_score_bull", "s2m_pressure_ratio", "s2m_trend_z",

                "vp_market_context", "htfvp_market_context",

            ),

            "negative": ("s2vp_score_short", "s2vp_context_score_bear", "vp_score_short", "vp_context_score_bear"),

        },

    }



    def _s3_latest_candle(self, pair):

        if not getattr(self, "dp", None):

            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if dataframe is None or dataframe.empty:

            return None

        return dataframe.iloc[-1]



    def _s3_target_provider_order(self):

        source = str(getattr(self, "SOURCE_ENTRY_STEM", "")).lower()

        order = []

        if any(token in source for token in ("vp", "poc", "vah", "val", "hvn", "lvn", "node")):

            order.append("vp")

        if any(token in source for token in ("tlv2", "trendline", "support", "resistance", "sup_", "res_")):

            order.append("tlv2")

        if any(token in source for token in ("geometry", "wedge", "triangle", "rectangle", "channel")):

            order.append("geometry")

        if any(token in source for token in ("pivot", "bos", "choch", "higher_low", "lower_high")):

            order.append("pivot")

        if any(token in source for token in ("bos", "choch")):

            order.append("bos_choch")

        if any(token in source for token in ("prior", "equal_high", "equal_low", "liquidity")):

            order.append("prior_level")

        if "avwap" in source or "vwap" in source:

            order.append("avwap")

        return tuple(dict.fromkeys(order))



    def _s3_directional_profit(self, trade, target_rate):
        open_rate = float(getattr(trade, "open_rate", 0.0) or 0.0)
        target_rate = float(target_rate or 0.0)
        if not np.isfinite(open_rate) or not np.isfinite(target_rate) or open_rate <= 0.0 or target_rate <= 0.0:
            return None
        if bool(getattr(trade, "is_short", False)):
            return (open_rate - target_rate) / open_rate
        return (target_rate - open_rate) / open_rate
    def _s3_indicator_target(self, pair, trade, current_rate, nearest=True):

        last = self._s3_latest_candle(pair)

        if last is None:

            return None

        current_rate = float(current_rate or 0.0)

        if current_rate <= 0.0:

            return None

        side_key = "short" if bool(getattr(trade, "is_short", False)) else "long"

        candidates = []

        for provider_index, provider in enumerate(self._s3_target_provider_order()):

            provider_columns = self.S3_TARGET_PROVIDER_COLUMNS.get(provider, {})

            for column in provider_columns.get(side_key, ()): 

                if column not in last.index:

                    continue

                value = pd.to_numeric(pd.Series([last[column]]), errors="coerce").iloc[0]

                if not np.isfinite(value) or float(value) <= 0.0:

                    continue

                target_profit = self._s3_directional_profit(trade, float(value))

                if target_profit is None:

                    continue

                if float(self.indicator_min_profit.value) <= target_profit <= float(self.indicator_max_profit.value):

                    near = abs(float(value) - current_rate) / current_rate

                    rank_profit = target_profit if nearest else -target_profit

                    candidates.append((provider_index, rank_profit, near, float(value), str(column), provider))

        if not candidates:

            return None

        provider_index, target_profit, near, value, column, provider = sorted(candidates, key=lambda item: (item[0], item[1], item[2]))[0]

        return {"profit": abs(target_profit), "near": near, "value": value, "column": column, "provider": provider, "priority": provider_index}



    def _s3_guard_failure(self, pair, trade, current_profit):

        last = self._s3_latest_candle(pair)

        if last is None:

            return False

        side_key = "short" if bool(getattr(trade, "is_short", False)) else "long"

        rules = self.S3_GUARD_FAILURE_COLUMNS.get(side_key, {})

        score = 0

        for col in rules.get("positive", ()):

            if col not in last.index:

                continue

            value = pd.to_numeric(pd.Series([last[col]]), errors="coerce").iloc[0]

            if np.isfinite(value) and value > 0:

                score += 1

        for col in rules.get("negative", ()):

            if col not in last.index:

                continue

            value = pd.to_numeric(pd.Series([last[col]]), errors="coerce").iloc[0]

            if np.isfinite(value) and value < 0:

                score += 1

        return score >= 1 and float(current_profit) >= float(self.guard_profit_floor.value)



    def _s3_trigger_invalidated(self, pair, trade, current_rate, current_profit):

        last = self._s3_latest_candle(pair)

        if last is None:

            return False

        is_short = bool(getattr(trade, "is_short", False))

        close = pd.to_numeric(pd.Series([last.get("close", current_rate)]), errors="coerce").iloc[0]

        if not np.isfinite(close):

            close = float(current_rate or 0.0)

        target = self._s3_indicator_target(pair, trade, current_rate, nearest=True)

        if target is None:

            return self._s3_guard_failure(pair, trade, current_profit)

        level = float(target["value"])

        if not np.isfinite(level) or level <= 0.0 or close <= 0.0:

            return False

        if is_short:

            return close > level * (1.0 + float(self._s3_param_value("indicator_near_pct", 0.006))) and float(current_profit) >= float(self.guard_profit_floor.value)

        return close < level * (1.0 - float(self._s3_param_value("indicator_near_pct", 0.006))) and float(current_profit) >= float(self.guard_profit_floor.value)



    def _s3_trade_age_candles(self, trade, current_time):

        minutes = timeframe_to_minutes(self.timeframe)

        opened = getattr(trade, "open_date_utc", None) or getattr(trade, "open_date", None)

        if opened is None or current_time is None or minutes <= 0:

            return 0

        return int(max(0.0, (current_time - opened).total_seconds()) // (60 * minutes))



    def _s3_param_value(self, name, default):

        param = getattr(self, name, None)

        return getattr(param, "value", default)



    def _s3_profit_bucket(self, current_profit):

        profit = float(current_profit or 0.0)

        stop_ref = float(self._s3_param_value("fixed_sl_pct", 0.03) or 0.03)

        if profit <= -max(stop_ref * 0.5, 0.005):

            return "loss_beyond_half_stop"

        if profit < -0.001:

            return "small_loss"

        if profit < 0.005:

            return "flat"

        if profit < 0.010:

            return "profit_0_5"

        if profit < 0.020:

            return "profit_1"

        if profit < 0.030:

            return "profit_2"

        return "profit_3_plus"



    def _s3_entry_family(self):

        source = str(getattr(self, "SOURCE_ENTRY_STEM", "")).lower()

        if any(token in source for token in ("vp", "poc", "vah", "val", "hvn", "lvn", "node")):

            return "vp"

        if any(token in source for token in ("tlv2", "trendline", "support", "resistance", "sup_", "res_")):

            return "tlv2"

        if any(token in source for token in ("bos", "choch", "higher_low", "lower_high")):

            return "bos_choch"

        if any(token in source for token in ("pivot", "prior", "equal_high", "equal_low", "liquidity")):

            return "pivot_prior_liquidity"

        if any(token in source for token in ("geometry", "pattern", "wedge", "triangle", "rectangle", "channel", "reversal", "continuation", "wolfe")):

            return "pattern"

        if any(token in source for token in ("crash", "flush", "capitulation", "reclaim")):

            return "crash"

        if any(token in source for token in ("mtf", "mtfx", "h4", "d1", "3d")):

            return "mtf"

        return "source_specific"



    def _s3_trade_state(self, pair, trade, current_time, current_rate, current_profit):

        target = self._s3_indicator_target(pair, trade, current_rate, nearest=True)

        guard_failed = self._s3_guard_failure(pair, trade, current_profit)

        trigger_failed = self._s3_trigger_invalidated(pair, trade, current_rate, current_profit)

        exits_done = int(getattr(trade, "nr_of_successful_exits", 0) or 0)

        age_candles = self._s3_trade_age_candles(trade, current_time) if current_time is not None else 0

        if exits_done <= 0:

            partial_state = "no_partial"

        elif exits_done == 1:

            partial_state = "first_partial_done"

        else:

            partial_state = "second_partial_done"

        if age_candles >= int(self._s3_param_value("time_stop_candles", 48)):

            time_state = "stale"

        elif age_candles >= max(1, int(self._s3_param_value("time_stop_candles", 48)) // 2):

            time_state = "normal"

        else:

            time_state = "early"

        return {

            "entry_side": "short" if bool(getattr(trade, "is_short", False)) else "long",

            "current_profit": float(current_profit or 0.0),

            "profit_bucket": self._s3_profit_bucket(current_profit),

            "entry_family": self._s3_entry_family(),

            "target_source": target["provider"] if target is not None else "fixed",

            "nearest_target_distance": target["near"] if target is not None else None,

            "target_touched": bool(target is not None and target["near"] <= float(self._s3_param_value("indicator_near_pct", 0.006))),

            "target": target,

            "guard_state": "opposite" if guard_failed else "aligned_or_neutral",

            "trigger_state": "invalidated" if trigger_failed else "intact",

            "partial_state": partial_state,

            "time_state": time_state,

            "age_candles": age_candles,

        }



    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, after_fill=False, **kwargs):

        _ = current_time, after_fill, kwargs

        mode = str(self.exit_path_mode.value)

        stop_profit = -float(self.fixed_sl_pct.value)

        state = self._s3_trade_state(pair, trade, current_time, current_rate, current_profit)

        indicator_hit = state["target"]

        guard_failed = state["guard_state"] == "opposite"

        trigger_failed = state["trigger_state"] == "invalidated"

        if ("be" in mode or "indicator" in mode) and float(current_profit) >= float(self.breakeven_trigger.value):

            stop_profit = max(stop_profit, float(self.breakeven_offset.value))

        if indicator_hit is not None and ("indicator" in mode or "structural" in mode or "target" in mode):

            stop_profit = max(stop_profit, float(current_profit) - float(self.indicator_stop_buffer.value))

        if (guard_failed or trigger_failed) and any(token in mode for token in ("guard", "trigger", "stop", "tighten")):

            stop_profit = max(stop_profit, float(current_profit) - float(self.guard_tighten_buffer.value))

        if "trail" in mode and float(current_profit) >= float(self.trailing_activation.value):

            stop_profit = max(stop_profit, float(current_profit) - float(self.trailing_distance.value))

        open_rate = float(getattr(trade, "open_rate", 0.0) or current_rate or 0.0)

        current_rate = float(current_rate or 0.0)

        if open_rate <= 0.0 or current_rate <= 0.0:

            return -float(self.fixed_sl_pct.value)

        stop_price = open_rate * (1.0 + stop_profit) if not bool(getattr(trade, "is_short", False)) else open_rate * (1.0 - stop_profit)

        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=bool(getattr(trade, "is_short", False)), leverage=float(getattr(trade, "leverage", 1.0) or 1.0))



    def adjust_trade_position(self, trade, current_time, current_rate, current_profit, min_stake, max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, **kwargs):

        _ = current_time, min_stake, max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, kwargs

        if bool(getattr(trade, "has_open_orders", False)):

            return None

        mode = str(self.exit_path_mode.value)

        if not any(token in mode for token in ("partial", "ladder")):

            return None

        exits_done = int(getattr(trade, "nr_of_successful_exits", 0) or 0)

        stake = float(getattr(trade, "stake_amount", 0.0) or 0.0)

        if stake <= 0.0:

            return None

        state = self._s3_trade_state(getattr(trade, "pair", ""), trade, current_time, current_rate, current_profit)

        indicator_hit = state["target"]

        guard_failed = state["guard_state"] == "opposite"

        trigger_failed = state["trigger_state"] == "invalidated"

        first_hit = float(current_profit) >= float(self.partial_1_profit.value)

        if "indicator" in mode and indicator_hit is not None and indicator_hit["near"] <= float(self._s3_param_value("indicator_near_pct", 0.006)):

            first_hit = True

        if any(token in mode for token in ("guard", "trigger")) and (guard_failed or trigger_failed):

            first_hit = first_hit or float(current_profit) >= float(self.guard_profit_floor.value)

        if exits_done == 0 and first_hit:

            return -(stake * float(self.partial_1_fraction.value)), f"s3_{self.S3_BRANCH_FAMILY}_partial_1"

        second_hit = float(current_profit) >= float(self.partial_2_profit.value)

        if exits_done == 1 and ("ladder" in mode or "partial" in mode) and second_hit:

            return -(stake * float(self.partial_2_fraction.value)), f"s3_{self.S3_BRANCH_FAMILY}_partial_2"

        return None



    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):

        _ = kwargs

        mode = str(self.exit_path_mode.value)

        state = self._s3_trade_state(pair, trade, current_time, current_rate, current_profit)

        indicator_hit = state["target"]

        guard_failed = state["guard_state"] == "opposite"

        trigger_failed = state["trigger_state"] == "invalidated"

        exits_done = int(getattr(trade, "nr_of_successful_exits", 0) or 0)

        if indicator_hit is not None and any(token in mode for token in ("indicator", "target", "structural")):

            near_enough = indicator_hit["near"] <= float(self._s3_param_value("indicator_near_pct", 0.006)) or float(current_profit) >= float(self.indicator_min_profit.value)

            if near_enough and ("partial" not in mode or exits_done >= 1):

                return f"s3_{self.S3_BRANCH_FAMILY}_indicator_{indicator_hit['column']}"

        if guard_failed and "guard" in mode and ("partial" not in mode or exits_done >= 1):

            return f"s3_{self.S3_BRANCH_FAMILY}_guard_exit"

        if trigger_failed and "trigger" in mode and ("partial" not in mode or exits_done >= 1):

            return f"s3_{self.S3_BRANCH_FAMILY}_trigger_exit"

        if float(current_profit) >= float(self.fixed_tp_pct.value) and not any(token in mode for token in ("trail", "indicator", "target")):

            return f"s3_{self.S3_BRANCH_FAMILY}_fixed_target"

        if float(current_profit) <= -float(self.fixed_sl_pct.value):

            return f"s3_{self.S3_BRANCH_FAMILY}_fixed_stop"

        if state["time_state"] == "stale" and float(current_profit) >= float(self.time_stop_min_profit.value):

            return f"s3_{self.S3_BRANCH_FAMILY}_time_exit"

        return None





apply_s3_branch_surface(Sieve3ExitGuardFailureFromLoosenLadderLongSupHoldBroaderTrigger)

apply_explicit_hyperopt_surface(Sieve3ExitGuardFailureFromLoosenLadderLongSupHoldBroaderTrigger)

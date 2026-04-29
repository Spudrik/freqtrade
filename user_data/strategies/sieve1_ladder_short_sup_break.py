from __future__ import annotations

from datetime import datetime
import os
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, IStrategy
from user_data.Indicators.complex_volume_profile import add_volume_profile


LEVEL_LOOKBACK_CHOICES = [48, 96, 168, 336, 720]
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


class Sieve1LadderShortSupBreak(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "1h"
    startup_candle_count = 1200
    process_only_new_candles = True

    minimal_roi = {"0": _pct_env("ENTRY_SIEVE_TAKE_PROFIT_PCT", 0.02)}
    stoploss = -_pct_env("ENTRY_SIEVE_STOPLOSS_PCT", 0.02)
    use_exit_signal = False
    use_custom_stoploss = False
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    ENTRY_TAG = "short_sup_break"
    ENTRY_SIDE = "short"
    ENTRY_KIND = "short_sup_break"
    MODE = "entry_short_sup_break"
    CONFIRMATION_PROFILE = "short_support_break"
    BREATHING_PROFILE = "none"

    level_lookback = _tag(CategoricalParameter(LEVEL_LOOKBACK_CHOICES, default=168, space="buy", optimize=True, load=True), MODE)
    local_lookback = _tag(CategoricalParameter(LOCAL_LOOKBACK_CHOICES, default=24, space="buy", optimize=True, load=True), MODE)
    d1_level_lookback = _tag(CategoricalParameter(D1_LEVEL_LOOKBACK_CHOICES, default=50, space="buy", optimize=True, load=True), MODE)
    volume_window = _tag(CategoricalParameter(VOLUME_WINDOW_CHOICES, default=24, space="buy", optimize=True, load=True), MODE)
    d1_volume_window = _tag(CategoricalParameter(D1_VOLUME_WINDOW_CHOICES, default=28, space="buy", optimize=True, load=True), MODE)
    confirm_bars = _tag(IntParameter(3, 48, default=12, space="buy", optimize=True, load=True), MODE)
    zone_pct = _tag(DecimalParameter(0.002, 0.040, decimals=3, default=0.012, space="buy", optimize=True, load=True), MODE)
    trigger_buffer_pct = _tag(DecimalParameter(0.000, 0.020, decimals=3, default=0.003, space="buy", optimize=True, load=True), MODE)
    h1_rvol_enable = _tag(CategoricalParameter(ENABLE_CHOICES, default="on", space="buy", optimize=True, load=True), MODE)
    h1_pressure_enable = _tag(CategoricalParameter(ENABLE_CHOICES, default="on", space="buy", optimize=True, load=True), MODE)
    h1_event_enable = _tag(CategoricalParameter(ENABLE_CHOICES, default="on", space="buy", optimize=True, load=True), MODE)
    d1_rvol_enable = _tag(CategoricalParameter(ENABLE_CHOICES, default="on", space="buy", optimize=True, load=True), MODE)
    d1_pressure_enable = _tag(CategoricalParameter(ENABLE_CHOICES, default="on", space="buy", optimize=True, load=True), MODE)
    d1_structure_mode = _tag(CategoricalParameter(D1_STRUCTURE_MODE_CHOICES, default="near", space="buy", optimize=True, load=True), MODE)
    h1_rvol_min = _tag(DecimalParameter(0.8, 2.5, decimals=1, default=1.1, space="buy", optimize=True, load=True), MODE)
    h1_pressure_min = _tag(DecimalParameter(0.0, 1.5, decimals=1, default=0.3, space="buy", optimize=True, load=True), MODE)
    d1_rvol_min = _tag(DecimalParameter(0.7, 2.2, decimals=1, default=1.0, space="buy", optimize=True, load=True), MODE)
    d1_pressure_min = _tag(DecimalParameter(0.0, 1.2, decimals=1, default=0.2, space="buy", optimize=True, load=True), MODE)

    use_vp_4h_guard = _tag(CategoricalParameter(ENABLE_CHOICES, default="off", space="buy", optimize=True, load=True), MODE)
    vp_4h_guard_mode = _tag(CategoricalParameter(GUARD_MODE_CHOICES, default="score_or_context", space="buy", optimize=True, load=True), MODE)
    vp_4h_window = _tag(CategoricalParameter([12, 24, 48, 72, 96], default=48, space="buy", optimize=True, load=True), MODE)
    vp_4h_bins = _tag(CategoricalParameter([16, 24, 36, 48, 64], default=36, space="buy", optimize=True, load=True), MODE)
    vp_4h_score_min = _tag(DecimalParameter(0.00, 1.00, decimals=2, default=0.25, space="buy", optimize=True, load=True), MODE)
    vp_4h_context_min = _tag(DecimalParameter(0.00, 1.00, decimals=2, default=0.28, space="buy", optimize=True, load=True), MODE)
    use_vp_1d_guard = _tag(CategoricalParameter(ENABLE_CHOICES, default="off", space="buy", optimize=True, load=True), MODE)
    vp_1d_guard_mode = _tag(CategoricalParameter(GUARD_MODE_CHOICES, default="context", space="buy", optimize=True, load=True), MODE)
    vp_1d_window = _tag(CategoricalParameter([10, 20, 30, 45, 60, 84], default=30, space="buy", optimize=True, load=True), MODE)
    vp_1d_bins = _tag(CategoricalParameter([16, 24, 36, 48, 64], default=36, space="buy", optimize=True, load=True), MODE)
    vp_1d_score_min = _tag(DecimalParameter(0.00, 1.00, decimals=2, default=0.25, space="buy", optimize=True, load=True), MODE)
    vp_1d_context_min = _tag(DecimalParameter(0.00, 1.00, decimals=2, default=0.28, space="buy", optimize=True, load=True), MODE)
    vp_guard_value_area_pct = _tag(DecimalParameter(0.55, 0.85, decimals=2, default=0.70, space="buy", optimize=True, load=True), MODE)
    vp_guard_price_source = _tag(CategoricalParameter(PRICE_SOURCE_CHOICES, default="hlc3", space="buy", optimize=True, load=True), MODE)
    vp_guard_node_near_pct = _tag(DecimalParameter(0.002, 0.030, decimals=3, default=0.010, space="buy", optimize=True, load=True), MODE)
    vp_guard_pressure_delta_min = _tag(DecimalParameter(0.00, 0.35, decimals=2, default=0.05, space="buy", optimize=True, load=True), MODE)

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        if not self.dp:
            return []
        try:
            whitelist = self.dp.current_whitelist()
            return [(pair, "1d") for pair in whitelist] + [(pair, "4h") for pair in whitelist]
        except Exception:
            return []

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
        dataframe[f"{prefix}_cvd_fast"] = dataframe[f"{prefix}_cvd"].ewm(span=20, adjust=False, min_periods=5).mean()
        dataframe[f"{prefix}_cvd_slow"] = dataframe[f"{prefix}_cvd"].ewm(span=48, adjust=False, min_periods=12).mean()
        dataframe[f"{prefix}_cvd_trend"] = dataframe[f"{prefix}_cvd_fast"] - dataframe[f"{prefix}_cvd_slow"]
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
        frame["d1_ema_fast"] = close.ewm(span=20, adjust=False, min_periods=10).mean()
        frame["d1_ema_slow"] = close.ewm(span=60, adjust=False, min_periods=30).mean()
        frame["d1_trend_up"] = frame["d1_ema_fast"] > frame["d1_ema_slow"]
        frame["d1_trend_down"] = frame["d1_ema_fast"] < frame["d1_ema_slow"]
        return frame

    def _merge_daily_context(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp or "date" not in dataframe.columns:
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
        dataframe["sieve_ema_fast"] = close.ewm(span=24, adjust=False, min_periods=12).mean()
        dataframe["sieve_ema_slow"] = close.ewm(span=96, adjust=False, min_periods=48).mean()
        dataframe["sieve_trend_up"] = dataframe["sieve_ema_fast"] > dataframe["sieve_ema_slow"]
        dataframe["sieve_trend_down"] = dataframe["sieve_ema_fast"] < dataframe["sieve_ema_slow"]
        dataframe = self._merge_daily_context(dataframe, metadata)
        dataframe = self._merge_informative_vp(dataframe, metadata, "4h", "vp4h", int(self.vp_4h_window.value), int(self.vp_4h_bins.value))
        dataframe = self._merge_informative_vp(dataframe, metadata, "1d", "vp1d", int(self.vp_1d_window.value), int(self.vp_1d_bins.value))
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
        high = self._num(dataframe["high"])
        low = self._num(dataframe["low"])
        resistance = self._num(dataframe[f"sieve_resistance_{level}"])
        support = self._num(dataframe[f"sieve_support_{level}"])
        local_high = self._num(dataframe[f"sieve_local_high_{local}"])
        local_low = self._num(dataframe[f"sieve_local_low_{local}"])
        ema_fast = self._num(dataframe["sieve_ema_fast"])
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
            trend_reclaim = close.gt(ema_fast * (1.0 + trigger * 0.50)) & close_loc.ge(0.25)
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
        ema_fast = self._num(dataframe["sieve_ema_fast"])
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
            structure = trend_up & low.le(ema_fast * (1.0 + zone)) & low.le(support * (1.0 + zone)) & close.gt(ema_fast * (1.0 + trigger))
            side = "long"
        elif kind == "short_trend_pullback":
            structure = trend_down & high.ge(ema_fast * (1.0 - zone)) & high.ge(resistance * (1.0 - zone)) & close.lt(ema_fast * (1.0 - trigger))
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
        dataframe[f"plot_{self.ENTRY_TAG}"] = mask.astype(float)
        dataframe.loc[mask, "enter_short"] = 1
        dataframe.loc[mask, "enter_tag"] = self.ENTRY_TAG
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = ""
        return dataframe



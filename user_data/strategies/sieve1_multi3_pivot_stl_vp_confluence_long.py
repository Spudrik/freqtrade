from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.strategy import BooleanParameter, CategoricalParameter, DecimalParameter, IntParameter, IStrategy

from user_data.Indicators.complex_pivot_structure import add_pivot_structure
from user_data.Indicators.complex_structural_trendlines import add_structural_trendlines
from user_data.Indicators.complex_volume_profile import add_volume_profile


HYPEROPT_PARAM_ENV = "HYBRID_RECOVERY_HYPEROPT_PARAMS"
ENTRY_SIEVE_TAKE_PROFIT_ENV = "ENTRY_SIEVE_TAKE_PROFIT_PCT"
ENTRY_SIEVE_STOPLOSS_ENV = "ENTRY_SIEVE_STOPLOSS_PCT"
ENTRY_MODE = "entry_multi3_pivot_stl_vp_confluence_long"
ENTRY_TAG = "multi3_pivot_stl_vp_confluence_long"
SIDE = "long"

PIVOT_BATCH_CHOICES = ["any", "break_reclaim", "trend_break", "compression_range", "score_context"]
STL_BATCH_CHOICES = ["suggested", "line_event", "touch_reclaim_reject", "triangle", "score_context"]
VP_BATCH_CHOICES = ["trigger", "node", "value_area", "lvn_hvn", "score_context"]
PRICE_SOURCE_CHOICES = ["close", "hl2", "hlc3", "ohlc4"]


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


def entry_sieve_minimal_roi(default: float = 0.02) -> dict[str, float]:
    return {"0": _pct_env(ENTRY_SIEVE_TAKE_PROFIT_ENV, default)}


def entry_sieve_stoploss(default: float = -0.02) -> float:
    return -_pct_env(ENTRY_SIEVE_STOPLOSS_ENV, abs(default))


def tagged_parameter(param: Any) -> Any:
    setattr(param, "batch_tags", ("family:entries", f"mode:{ENTRY_MODE}"))
    return param


def _num(frame: DataFrame, column: str, default: float | Series = 0.0) -> Series:
    if column not in frame.columns:
        if isinstance(default, Series):
            return pd.to_numeric(default, errors="coerce")
        return pd.Series(float(default), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype="bool")
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)


def _or_columns(frame: DataFrame, columns: list[str]) -> Series:
    result = pd.Series(False, index=frame.index, dtype="bool")
    for column in columns:
        result |= _bool(frame, column)
    return result


class Sieve1Multi3PivotStlVpConfluenceLong(IStrategy):
    """
    Sieve1 multi3 probe: pivot structure + structural trendlines + volume profile.

    This is not a finished strategy. It is a broad confluence search surface that
    lets hyperopt select evidence batches behind enables. If a useful combination
    appears in Sieve1, extract it into a narrower dedicated strategy for refinement.
    """

    INTERFACE_VERSION = 3

    timeframe = "1h"
    startup_candle_count = 1800
    process_only_new_candles = True
    can_short = False

    minimal_roi = entry_sieve_minimal_roi(0.02)
    stoploss = entry_sieve_stoploss(-0.02)
    use_exit_signal = False
    use_custom_stoploss = False
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    min_confluence_hits = tagged_parameter(IntParameter(2, 3, default=2, space="buy", optimize=True, load=True))
    use_pivot_batch = tagged_parameter(BooleanParameter(default=True, space="buy", optimize=True, load=True))
    use_stl_batch = tagged_parameter(BooleanParameter(default=True, space="buy", optimize=True, load=True))
    use_vp_batch = tagged_parameter(BooleanParameter(default=True, space="buy", optimize=True, load=True))

    pivot_batch = tagged_parameter(CategoricalParameter(PIVOT_BATCH_CHOICES, default="any", space="buy", optimize=True, load=True))
    pivot_strength = tagged_parameter(IntParameter(3, 13, default=5, space="buy", optimize=True, load=True))
    pivot_atr_period = tagged_parameter(IntParameter(7, 28, default=14, space="buy", optimize=True, load=True))
    pivot_min_prominence_atr = tagged_parameter(DecimalParameter(0.10, 1.50, decimals=2, default=0.35, space="buy", optimize=True, load=True))
    pivot_zone_pct = tagged_parameter(DecimalParameter(0.001, 0.020, decimals=3, default=0.003, space="buy", optimize=True, load=True))
    pivot_breakout_buffer_pct = tagged_parameter(DecimalParameter(0.000, 0.020, decimals=3, default=0.002, space="buy", optimize=True, load=True))
    pivot_score_window = tagged_parameter(IntParameter(6, 48, default=12, space="buy", optimize=True, load=True))
    pivot_score_min = tagged_parameter(DecimalParameter(0.00, 1.00, decimals=2, default=0.25, space="buy", optimize=True, load=True))
    pivot_context_min = tagged_parameter(DecimalParameter(0.00, 1.00, decimals=2, default=0.30, space="buy", optimize=True, load=True))
    use_pivot_target_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=True, load=True))
    pivot_target_distance_min = tagged_parameter(DecimalParameter(0.000, 0.080, decimals=3, default=0.006, space="buy", optimize=True, load=True))

    stl_batch = tagged_parameter(CategoricalParameter(STL_BATCH_CHOICES, default="suggested", space="buy", optimize=True, load=True))
    stl_strength = tagged_parameter(IntParameter(13, 34, default=21, space="buy", optimize=True, load=True))
    stl_min_touch_count = tagged_parameter(IntParameter(2, 6, default=3, space="buy", optimize=True, load=True))
    stl_min_anchor_span_bars = tagged_parameter(IntParameter(24, 144, default=48, space="buy", optimize=True, load=True))
    stl_min_line_span_bars = tagged_parameter(IntParameter(48, 300, default=120, space="buy", optimize=True, load=True))
    stl_min_pivot_prominence_atr = tagged_parameter(DecimalParameter(0.50, 4.00, decimals=2, default=2.00, space="buy", optimize=True, load=True))
    stl_min_respect_ratio = tagged_parameter(DecimalParameter(0.60, 1.00, decimals=2, default=0.90, space="buy", optimize=True, load=True))
    stl_breakout_buffer_pct = tagged_parameter(DecimalParameter(0.000, 0.020, decimals=3, default=0.003, space="buy", optimize=True, load=True))
    stl_score_window = tagged_parameter(IntParameter(24, 180, default=96, space="buy", optimize=True, load=True))
    stl_score_min = tagged_parameter(DecimalParameter(0.00, 1.00, decimals=2, default=0.28, space="buy", optimize=True, load=True))
    stl_context_min = tagged_parameter(IntParameter(0, 2, default=1, space="buy", optimize=True, load=True))
    stl_triangle_min = tagged_parameter(DecimalParameter(0.00, 1.00, decimals=2, default=0.35, space="buy", optimize=True, load=True))

    vp_batch = tagged_parameter(CategoricalParameter(VP_BATCH_CHOICES, default="trigger", space="buy", optimize=True, load=True))
    vp_window = tagged_parameter(IntParameter(24, 168, default=96, space="buy", optimize=True, load=True))
    vp_bins = tagged_parameter(IntParameter(24, 72, default=48, space="buy", optimize=True, load=True))
    vp_value_area_pct = tagged_parameter(DecimalParameter(0.55, 0.85, decimals=2, default=0.70, space="buy", optimize=True, load=True))
    vp_price_source = tagged_parameter(CategoricalParameter(PRICE_SOURCE_CHOICES, default="hlc3", space="buy", optimize=True, load=True))
    vp_smooth_bins = tagged_parameter(IntParameter(1, 6, default=3, space="buy", optimize=True, load=True))
    vp_hvn_threshold = tagged_parameter(DecimalParameter(0.50, 0.90, decimals=2, default=0.70, space="buy", optimize=True, load=True))
    vp_lvn_threshold = tagged_parameter(DecimalParameter(0.10, 0.55, decimals=2, default=0.35, space="buy", optimize=True, load=True))
    vp_pressure_delta_min = tagged_parameter(DecimalParameter(0.00, 0.35, decimals=2, default=0.05, space="buy", optimize=True, load=True))
    vp_node_near_pct = tagged_parameter(DecimalParameter(0.002, 0.030, decimals=3, default=0.010, space="buy", optimize=True, load=True))
    vp_volume_percentile_min = tagged_parameter(DecimalParameter(0.00, 0.90, decimals=2, default=0.55, space="buy", optimize=True, load=True))
    vp_score_window = tagged_parameter(IntParameter(12, 120, default=48, space="buy", optimize=True, load=True))
    vp_entry_score_margin = tagged_parameter(DecimalParameter(0.00, 0.20, decimals=2, default=0.02, space="buy", optimize=True, load=True))
    vp_score_min = tagged_parameter(DecimalParameter(0.00, 1.00, decimals=2, default=0.25, space="buy", optimize=True, load=True))
    vp_context_min = tagged_parameter(DecimalParameter(0.00, 1.00, decimals=2, default=0.28, space="buy", optimize=True, load=True))

    use_volume_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=True, load=True))
    volume_window = tagged_parameter(IntParameter(6, 96, default=24, space="buy", optimize=True, load=True))
    volume_ratio_min = tagged_parameter(DecimalParameter(0.00, 5.00, decimals=2, default=0.80, space="buy", optimize=True, load=True))
    pressure_min = tagged_parameter(DecimalParameter(0.00, 1.00, decimals=2, default=0.05, space="buy", optimize=True, load=True))

    use_trend_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=True, load=True))
    trend_fast_period = tagged_parameter(IntParameter(3, 72, default=21, space="buy", optimize=True, load=True))
    trend_slow_period = tagged_parameter(IntParameter(24, 240, default=96, space="buy", optimize=True, load=True))

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe = self._add_volume_pressure(dataframe)
        dataframe = self._add_trend(dataframe)

        pivot_strength = int(self.pivot_strength.value)
        stl_strength = int(self.stl_strength.value)
        shared_strengths = tuple(sorted({3, 5, 8, 13, 21, 34, pivot_strength, stl_strength}))
        dataframe = add_pivot_structure(
            dataframe,
            strength=pivot_strength,
            strengths=shared_strengths,
            atr_period=int(self.pivot_atr_period.value),
            min_prominence_atr=float(self.pivot_min_prominence_atr.value),
            zone_pct=float(self.pivot_zone_pct.value),
            breakout_buffer_pct=float(self.pivot_breakout_buffer_pct.value),
            structure_score_window=int(self.pivot_score_window.value),
            min_target_distance_pct=float(self.pivot_target_distance_min.value),
            structural_strength=max(21, stl_strength),
            prefix="pa",
        )
        dataframe = add_structural_trendlines(
            dataframe,
            strength=stl_strength,
            strengths=tuple(sorted({13, 21, 34, stl_strength})),
            pivot_prefix="pa",
            output_prefix="stl",
            missing_pivot_mode="raise",
            min_touch_count=int(self.stl_min_touch_count.value),
            min_anchor_span_bars=int(self.stl_min_anchor_span_bars.value),
            min_line_span_bars=int(self.stl_min_line_span_bars.value),
            min_pivot_prominence_atr=float(self.stl_min_pivot_prominence_atr.value),
            min_respect_ratio=float(self.stl_min_respect_ratio.value),
            breakout_buffer_pct=float(self.stl_breakout_buffer_pct.value),
            score_window=int(self.stl_score_window.value),
        )
        dataframe = add_volume_profile(
            dataframe,
            window=int(self.vp_window.value),
            bins=int(self.vp_bins.value),
            value_area_pct=float(self.vp_value_area_pct.value),
            price_source=str(self.vp_price_source.value),
            smooth_bins=int(self.vp_smooth_bins.value),
            hvn_threshold=float(self.vp_hvn_threshold.value),
            lvn_threshold=float(self.vp_lvn_threshold.value),
            pressure_delta_min=float(self.vp_pressure_delta_min.value),
            node_near_pct=float(self.vp_node_near_pct.value),
            volume_percentile_min=float(self.vp_volume_percentile_min.value),
            score_window=int(self.vp_score_window.value),
            entry_score_margin=float(self.vp_entry_score_margin.value),
            prefix="vp",
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None

        pivot_enabled = bool(self.use_pivot_batch.value)
        stl_enabled = bool(self.use_stl_batch.value)
        vp_enabled = bool(self.use_vp_batch.value)
        pivot_hit = self._pivot_signal(dataframe) if pivot_enabled else pd.Series(False, index=dataframe.index, dtype="bool")
        stl_hit = self._stl_signal(dataframe) if stl_enabled else pd.Series(False, index=dataframe.index, dtype="bool")
        vp_hit = self._vp_signal(dataframe) if vp_enabled else pd.Series(False, index=dataframe.index, dtype="bool")

        enabled_count = int(pivot_enabled) + int(stl_enabled) + int(vp_enabled)
        hit_count = pivot_hit.astype("int8") + stl_hit.astype("int8") + vp_hit.astype("int8")
        minimum_hits = int(self.min_confluence_hits.value)
        condition = hit_count.ge(minimum_hits) & pd.Series(enabled_count >= minimum_hits, index=dataframe.index)
        if bool(self.use_volume_guard.value):
            condition &= self._volume_guard(dataframe)
        if bool(self.use_trend_guard.value):
            condition &= self._trend_guard(dataframe)

        valid = condition.fillna(False) & dataframe["volume"].gt(0.0) & dataframe["close"].notna()
        tag = f"{ENTRY_TAG}|p:{self.pivot_batch.value}|s:{self.stl_batch.value}|v:{self.vp_batch.value}|m:{minimum_hits}"
        dataframe.loc[valid, "enter_long"] = 1
        dataframe.loc[valid, "enter_tag"] = tag[:255]
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe

    def _add_volume_pressure(self, dataframe: DataFrame) -> DataFrame:
        window = int(self.volume_window.value)
        volume_mean = _num(dataframe, "volume").rolling(window, min_periods=max(2, window // 3)).mean()
        candle_range = _num(dataframe, "high").sub(_num(dataframe, "low")).replace(0.0, np.nan)
        close_location = _num(dataframe, "close").sub(_num(dataframe, "low")).div(candle_range).clip(0.0, 1.0)
        dataframe["entry_volume_ratio"] = _num(dataframe, "volume").div(volume_mean.replace(0.0, np.nan))
        dataframe["entry_pressure"] = close_location.sub(0.5).mul(2.0)
        return dataframe

    def _add_trend(self, dataframe: DataFrame) -> DataFrame:
        close = _num(dataframe, "close")
        dataframe["entry_ema_fast"] = close.ewm(span=int(self.trend_fast_period.value), adjust=False, min_periods=2).mean()
        dataframe["entry_ema_slow"] = close.ewm(span=int(self.trend_slow_period.value), adjust=False, min_periods=2).mean()
        return dataframe

    def _pivot_signal(self, dataframe: DataFrame) -> Series:
        mode = str(self.pivot_batch.value)
        if mode == "break_reclaim":
            signal = _or_columns(dataframe, ["pa_entry_resistance_breakout_long", "pa_entry_support_reclaim_long"])
        elif mode == "trend_break":
            signal = _or_columns(dataframe, ["pa_entry_bullish_trend_flip_breakout_long", "pa_entry_bullish_trend_aligned_breakout_long"])
        elif mode == "compression_range":
            signal = _or_columns(dataframe, ["pa_entry_compression_breakout_long", "pa_entry_range_support_long"])
        elif mode == "score_context":
            signal = self._score_context_signal(dataframe, "pa", "long", float(self.pivot_score_min.value), float(self.pivot_context_min.value))
        else:
            signal = _bool(dataframe, "pa_entry_any_long")
        if bool(self.use_pivot_target_guard.value):
            signal &= _num(dataframe, "pa_long_target_distance_pct").ge(float(self.pivot_target_distance_min.value))
        return signal

    def _stl_signal(self, dataframe: DataFrame) -> Series:
        mode = str(self.stl_batch.value)
        if mode == "line_event":
            return _or_columns(dataframe, ["stl_entry_resistance_breakout_long", "stl_entry_support_reclaim_long"])
        if mode == "touch_reclaim_reject":
            return _bool(dataframe, "stl_support_reclaim")
        if mode == "triangle":
            return _bool(dataframe, "stl_entry_triangle_breakout_long") | _num(dataframe, "stl_triangle_score").ge(float(self.stl_triangle_min.value))
        if mode == "score_context":
            return self._stl_score_context_signal(dataframe, "long")
        return _bool(dataframe, "stl_suggested_entry_long")

    def _vp_signal(self, dataframe: DataFrame) -> Series:
        mode = str(self.vp_batch.value)
        if mode == "node":
            return _bool(dataframe, "vp_node_entry_long")
        if mode == "value_area":
            return _or_columns(dataframe, ["vp_vah_breakout_with_pressure", "vp_lower_rejection_with_pressure"])
        if mode == "lvn_hvn":
            return _or_columns(dataframe, ["vp_hvn_below_reclaim", "vp_lvn_below_reject_long", "vp_lvn_accept_long", "vp_lvn_fast_traverse_long"])
        if mode == "score_context":
            return self._score_context_signal(dataframe, "vp", "long", float(self.vp_score_min.value), float(self.vp_context_min.value))
        return _bool(dataframe, "vp_entry_trigger_long")

    def _volume_guard(self, dataframe: DataFrame) -> Series:
        return _num(dataframe, "entry_volume_ratio").ge(float(self.volume_ratio_min.value)) & _num(dataframe, "entry_pressure").ge(float(self.pressure_min.value))

    def _trend_guard(self, dataframe: DataFrame) -> Series:
        return _num(dataframe, "entry_ema_fast").ge(_num(dataframe, "entry_ema_slow"))

    @staticmethod
    def _score_context_signal(dataframe: DataFrame, prefix: str, side: str, score_min: float, context_min: float) -> Series:
        score = _num(dataframe, f"{prefix}_score_{side}")
        opposite = _num(dataframe, f"{prefix}_score_{'short' if side == 'long' else 'long'}")
        if side == "long":
            context = _num(dataframe, f"{prefix}_context_score_bull")
            opposite_context = _num(dataframe, f"{prefix}_context_score_bear")
            market_ok = _num(dataframe, f"{prefix}_market_context").ge(0)
        else:
            context = _num(dataframe, f"{prefix}_context_score_bear")
            opposite_context = _num(dataframe, f"{prefix}_context_score_bull")
            market_ok = _num(dataframe, f"{prefix}_market_context").le(0)
        return score.ge(score_min) & score.ge(opposite) & context.ge(context_min) & context.ge(opposite_context) & market_ok

    def _stl_score_context_signal(self, dataframe: DataFrame, side: str) -> Series:
        score = _num(dataframe, f"stl_score_{side}")
        opposite = _num(dataframe, f"stl_score_{'short' if side == 'long' else 'long'}")
        context = _num(dataframe, "stl_market_context")
        if side == "long":
            context_ok = context.ge(float(self.stl_context_min.value))
        else:
            context_ok = context.le(-float(self.stl_context_min.value))
        return score.ge(float(self.stl_score_min.value)) & score.ge(opposite) & context_ok


apply_explicit_hyperopt_surface(Sieve1Multi3PivotStlVpConfluenceLong)

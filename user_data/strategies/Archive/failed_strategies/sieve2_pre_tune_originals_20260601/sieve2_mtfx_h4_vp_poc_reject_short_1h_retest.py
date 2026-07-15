from __future__ import annotations
# Archived pre-tune copy from 2026-06-01 review. Active top-level file remains in Sieve2;
# reason: tune/fix candidate for entry-only rerun after high-performer 4/2 validation.

# Held out because this top-level Sieve2 file was not present in the reviewed full Sieve2 result batch.

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

_INDICATOR_DIR = Path(__file__).resolve().parents[1] / "Indicators"
if str(_INDICATOR_DIR) not in sys.path:
    sys.path.insert(0, str(_INDICATOR_DIR))

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import BooleanParameter, CategoricalParameter, IStrategy
from user_data.Indicators.complex_trendline_projection_v2 import add_trendline_projection_v2
from user_data.Indicators.complex_volume_profile import add_volume_profile
from user_data.Indicators.pattern_bos_choch import add_bos_choch
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2
from user_data.Indicators.pattern_reversal import add_pattern_reversal
from user_data.strategies.sieve_guard_helpers import (
    SIEVE2_MARKET_GUARD_MODES,
    SIEVE2_VP_GUARD_MODES,
    add_sieve2_guard_indicators,
    apply_sieve2_optional_guards,
)

SIEVE_STAGE = "sieve2"
SOURCE_STRATEGY = "new_structural_mtf_expansion"
SOURCE_RESULT_BATCH = "manual_gap_review"
RESEARCH_PATH = "sieve2_mtf_structural_expansion"
NOVEL_IDEA = True

HYPEROPT_PARAM_ENV = "HYBRID_RECOVERY_HYPEROPT_PARAMS"
ENTRY_SIEVE_TAKE_PROFIT_ENV = "ENTRY_SIEVE_TAKE_PROFIT_PCT"
ENTRY_SIEVE_STOPLOSS_ENV = "ENTRY_SIEVE_STOPLOSS_PCT"
ENTRY_MODE = "entry_sieve2_mtfx_h4_vp_poc_reject_short_1h_retest"
ENTRY_TAG = "sieve2_mtfx_h4_vp_poc_reject_short_1h_retest"
SIDE = "short"
TIMEFRAME = "1h"
CONTEXT_TIMEFRAME = "4h"
HTF_CONCEPT = "vp_poc_reject_short"
LTF_TRIGGER = "retest"


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


def _cross_above(value: Series, level: Series) -> Series:
    return value.gt(level) & value.shift(1).le(level.shift(1))


def _cross_below(value: Series, level: Series) -> Series:
    return value.lt(level) & value.shift(1).ge(level.shift(1))


class Sieve2MtfxH4VpPocRejectShort1hRetest(IStrategy):
    """Sieve2 structural MTF expansion: HTF concept plus LTF structural execution."""

    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 240
    process_only_new_candles = True
    can_short = True

    minimal_roi = {"0": _pct_env(ENTRY_SIEVE_TAKE_PROFIT_ENV, 0.02)}
    stoploss = -_pct_env(ENTRY_SIEVE_STOPLOSS_ENV, 0.02)
    use_exit_signal = False
    use_custom_stoploss = False
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    use_sieve2_vp_guard = CategoricalParameter([False, True], default=False, space="buy", optimize=True, load=True)
    sieve2_vp_guard_mode = CategoricalParameter(SIEVE2_VP_GUARD_MODES, default="score_or_context", space="buy", optimize=True, load=True)
    sieve2_vp_window = CategoricalParameter([48, 96, 168], default=96, space="buy", optimize=False, load=True)
    sieve2_vp_bins = CategoricalParameter([24, 36, 48], default=36, space="buy", optimize=False, load=True)
    sieve2_vp_score_min = CategoricalParameter([0.15, 0.25, 0.35, 0.50], default=0.25, space="buy", optimize=True, load=True)
    sieve2_vp_context_min = CategoricalParameter([0.18, 0.28, 0.38, 0.50], default=0.28, space="buy", optimize=True, load=True)
    use_sieve2_market_guard = CategoricalParameter([False, True], default=False, space="buy", optimize=True, load=True)
    sieve2_market_guard_mode = CategoricalParameter(SIEVE2_MARKET_GUARD_MODES, default="pressure_or_trend", space="buy", optimize=True, load=True)
    sieve2_market_window = CategoricalParameter([12, 24, 48, 96], default=24, space="buy", optimize=False, load=True)
    sieve2_market_pressure_min = CategoricalParameter([0.03, 0.07, 0.12, 0.18, 0.25], default=0.07, space="buy", optimize=True, load=True)
    sieve2_market_trend_min = CategoricalParameter([0.0, 0.25, 0.50, 0.80], default=0.25, space="buy", optimize=True, load=True)
    sieve2_rs_benchmark_pair = CategoricalParameter(["BTC/USDT:USDT", "ETH/USDT:USDT"], default="BTC/USDT:USDT", space="buy", optimize=False, load=True)
    sieve2_rs_score_min = CategoricalParameter([0.35, 0.45, 0.55, 0.65], default=0.45, space="buy", optimize=True, load=True)

    score_min = tagged_parameter(CategoricalParameter([0.50, 0.60, 0.72, 0.82], default=0.60, space="buy", optimize=True, load=True))
    htf_recent_bars = tagged_parameter(CategoricalParameter([1, 2, 3, 5], default=3, space="buy", optimize=True, load=True))
    level_buffer_pct = tagged_parameter(CategoricalParameter([0.002, 0.004, 0.008, 0.012, 0.016], default=0.006, space="buy", optimize=True, load=True))
    local_window = tagged_parameter(CategoricalParameter([6, 12, 24], default=12, space="buy", optimize=True, load=True))
    use_ltf_volume_guard = tagged_parameter(BooleanParameter(default=True, space="buy", optimize=True, load=True))
    volume_ratio_min = tagged_parameter(CategoricalParameter([0.8, 1.0, 1.3, 1.6], default=1.0, space="buy", optimize=True, load=True))
    use_ltf_pressure_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=True, load=True))
    pressure_min = tagged_parameter(CategoricalParameter([0.03, 0.07, 0.12, 0.18, 0.25], default=0.07, space="buy", optimize=True, load=True))

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        if not self.dp:
            return []
        return [(pair, CONTEXT_TIMEFRAME) for pair in self.dp.current_whitelist()]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self._add_ltf_features(dataframe)
        dataframe = self._merge_htf_context(dataframe, metadata or {})
        dataframe = add_sieve2_guard_indicators(self, dataframe, metadata)
        return dataframe

    def _add_ltf_features(self, dataframe: DataFrame) -> DataFrame:
        frame = dataframe.copy()
        frame = add_bos_choch(frame, strength=3, min_prominence_atr=0.25, min_pivot_spacing_bars=1, max_pivot_age_bars=96, breakout_buffer_atr=0.10, prefix="ltfms")
        close = _num(frame, "close")
        open_ = _num(frame, "open")
        high = _num(frame, "high")
        low = _num(frame, "low")
        volume = _num(frame, "volume").clip(lower=0.0).fillna(0.0)
        candle_range = (high - low).replace(0.0, np.nan)
        pressure = (((close - open_) / candle_range).clip(-1.0, 1.0).fillna(0.0) + (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0).fillna(0.0)) / 2.0
        window = int(self.local_window.value)
        baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
        frame["mtfx_ltf_pressure"] = pressure
        frame["mtfx_ltf_volume_ratio"] = volume / baseline
        frame["mtfx_ltf_near_level_recent"] = False
        return frame

    def _merge_htf_context(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        frame = dataframe.copy()
        frame["mtfx_htf_context"] = False
        frame["mtfx_htf_level"] = np.nan
        if not self.dp:
            return frame
        pair = metadata.get("pair")
        if not pair:
            return frame
        informative = self.dp.get_pair_dataframe(pair=pair, timeframe=CONTEXT_TIMEFRAME)
        if informative is None or informative.empty or "date" not in informative.columns:
            return frame
        informative = self._add_htf_features(informative.copy())
        informative = informative[["date", "mtfx_htf_context", "mtfx_htf_level"]].copy().sort_values("date")
        informative["date_merge"] = pd.to_datetime(informative["date"], utc=True).dt.tz_convert(None) + pd.to_timedelta(timeframe_to_minutes(CONTEXT_TIMEFRAME), unit="m")
        base = frame.drop(columns=["mtfx_htf_context", "mtfx_htf_level"], errors="ignore").reset_index().rename(columns={"index": "__row_index"}).sort_values("date")
        base["__date_merge"] = pd.to_datetime(base["date"], utc=True).dt.tz_convert(None)
        merged = pd.merge_asof(base, informative.drop(columns=["date"]).sort_values("date_merge"), left_on="__date_merge", right_on="date_merge", direction="backward")
        merged = merged.set_index("__row_index").reindex(frame.index)
        frame["mtfx_htf_context"] = pd.Series(merged["mtfx_htf_context"], index=frame.index).astype("boolean").fillna(False).astype(bool)
        frame["mtfx_htf_level"] = pd.to_numeric(merged["mtfx_htf_level"], errors="coerce")
        return frame

    def _add_htf_features(self, dataframe: DataFrame) -> DataFrame:
        frame = dataframe.copy()
        frame = add_pattern_reversal(frame, timeframe=CONTEXT_TIMEFRAME, min_double_quality=float(self.score_min.value))
        frame = add_pattern_geometry_v2(
            frame,
            timeframe=CONTEXT_TIMEFRAME,
            output_slots=1,
            include_triangle_patterns=False,
            include_wedge_patterns=True,
            include_compression_patterns=False,
            include_rectangle_patterns=True,
            include_ascending_channel_patterns=False,
            include_descending_channel_patterns=False,
            min_line_score=float(self.score_min.value),
            output_prefix="pg2",
        )
        frame = add_volume_profile(frame, window=96, bins=48, value_area_pct=0.70, price_source="hlc3", smooth_bins=3, pressure_delta_min=0.05, node_near_pct=float(self.level_buffer_pct.value), prefix="htfvp")
        frame = add_trendline_projection_v2(frame, timeframe=CONTEXT_TIMEFRAME, pivot_strength=3, min_output_line_score=float(self.score_min.value), raw_line_output_count=1, output_prefix="tlv2")
        frame = add_bos_choch(frame, strength=3, min_prominence_atr=0.25, min_pivot_spacing_bars=1, max_pivot_age_bars=160, breakout_buffer_atr=0.10, prefix="htfms")
        frame = self._add_local_htf_levels(frame)
        context, level = self._htf_context_and_level(frame)
        recent = int(self.htf_recent_bars.value)
        frame["mtfx_htf_context"] = context.fillna(False).astype("int8").rolling(recent, min_periods=1).max().gt(0)
        frame["mtfx_htf_level"] = level
        return frame

    def _add_local_htf_levels(self, dataframe: DataFrame) -> DataFrame:
        frame = dataframe.copy()
        close = _num(frame, "close")
        open_ = _num(frame, "open")
        high = _num(frame, "high")
        low = _num(frame, "low")
        volume = _num(frame, "volume").clip(lower=0.0).fillna(0.0)
        lookback = 48
        prior_high = high.shift(1)
        prior_low = low.shift(1)
        frame["mtfx_prior_high"] = prior_high.rolling(lookback, min_periods=12).max()
        frame["mtfx_prior_low"] = prior_low.rolling(lookback, min_periods=12).min()
        tol = float(self.level_buffer_pct.value)
        frame["mtfx_equal_high"] = frame["mtfx_prior_high"].where(prior_high.sub(frame["mtfx_prior_high"]).abs().le(frame["mtfx_prior_high"].abs().mul(tol)).rolling(lookback, min_periods=12).sum().ge(2))
        frame["mtfx_equal_low"] = frame["mtfx_prior_low"].where(prior_low.sub(frame["mtfx_prior_low"]).abs().le(frame["mtfx_prior_low"].abs().mul(tol)).rolling(lookback, min_periods=12).sum().ge(2))
        body_high = pd.concat([open_, close], axis=1).max(axis=1)
        body_low = pd.concat([open_, close], axis=1).min(axis=1)
        avg_range = (high - low).rolling(20, min_periods=5).mean()
        bull_impulse = close.gt(open_) & (close - open_).gt(avg_range.mul(0.8))
        bear_impulse = close.lt(open_) & (open_ - close).gt(avg_range.mul(0.8))
        frame["mtfx_demand_zone_low"] = low.where(bull_impulse).ffill().shift(1)
        frame["mtfx_demand_zone_high"] = body_low.where(bull_impulse).ffill().shift(1)
        frame["mtfx_supply_zone_low"] = body_high.where(bear_impulse).ffill().shift(1)
        frame["mtfx_supply_zone_high"] = high.where(bear_impulse).ffill().shift(1)
        low_reset = prior_low.le(prior_low.rolling(72, min_periods=12).min())
        high_reset = prior_high.ge(prior_high.rolling(72, min_periods=12).max())
        frame["mtfx_avwap_from_low"] = self._anchored_vwap(frame, low_reset)
        frame["mtfx_avwap_from_high"] = self._anchored_vwap(frame, high_reset)
        return frame

    def _anchored_vwap(self, dataframe: DataFrame, reset: Series) -> Series:
        close = _num(dataframe, "close")
        high = _num(dataframe, "high")
        low = _num(dataframe, "low")
        volume = _num(dataframe, "volume").clip(lower=0.0).fillna(0.0)
        typical = (high + low + close) / 3.0
        group = reset.fillna(False).astype(int).cumsum()
        pv = (typical * volume).groupby(group).cumsum()
        vv = volume.groupby(group).cumsum().replace(0.0, np.nan)
        return pv / vv

    def _htf_context_and_level(self, dataframe: DataFrame) -> tuple[Series, Series]:
        close = _num(dataframe, "close")
        high = _num(dataframe, "high")
        low = _num(dataframe, "low")
        score_min = float(self.score_min.value)
        near = float(self.level_buffer_pct.value)
        false = pd.Series(False, index=dataframe.index, dtype="bool")
        level = pd.Series(np.nan, index=dataframe.index, dtype="float64")
        concept = HTF_CONCEPT

        pattern_map = {
            "head_shoulders_short": ("pat_head_shoulders", "short"),
            "inverse_head_shoulders_long": ("pat_inverse_head_shoulders", "long"),
            "double_bottom_long": ("pat_double_bottom", "long"),
            "double_top_short": ("pat_double_top", "short"),
            "triple_bottom_long": ("pat_triple_bottom", "long"),
            "triple_top_short": ("pat_triple_top", "short"),
        }
        if concept in pattern_map:
            prefix, direction = pattern_map[concept]
            level = _num(dataframe, f"{prefix}_confirmation_level", np.nan)
            context = _bool(dataframe, f"{prefix}_pattern_confirmed") & _num(dataframe, f"{prefix}_indicator_score").ge(score_min)
            context &= close.le(level.mul(1.0 + near)) if direction == "short" else close.ge(level.mul(1.0 - near))
            return context, level

        if concept in {"rectangle_long", "rectangle_short"}:
            level = _num(dataframe, "pg2_rectangle_upper", np.nan) if SIDE == "long" else _num(dataframe, "pg2_rectangle_lower", np.nan)
            context = _bool(dataframe, "pg2_rectangle_pattern_present") & _num(dataframe, "pg2_rectangle_indicator_score").ge(score_min)
            return context, level
        if concept in {"wedge_long", "wedge_short"}:
            level = _num(dataframe, "pg2_wedge_upper", np.nan) if SIDE == "long" else _num(dataframe, "pg2_wedge_lower", np.nan)
            direction_ok = _num(dataframe, "pg2_wedge_direction").ge(0) if SIDE == "long" else _num(dataframe, "pg2_wedge_direction").le(0)
            context = _bool(dataframe, "pg2_wedge_pattern_present") & _bool(dataframe, "pg2_wedge_squeeze_active") & direction_ok & _num(dataframe, "pg2_wedge_indicator_score").ge(score_min)
            return context, level

        if concept == "tlv2_res_break_long":
            level = _num(dataframe, "tlv2_resistance_line_rank0", np.nan)
            return _num(dataframe, "tlv2_resistance_score_rank0").ge(score_min) & close.ge(level.mul(1.0 - near)), level
        if concept == "tlv2_sup_reclaim_long":
            level = _num(dataframe, "tlv2_support_line_rank0", np.nan)
            return _num(dataframe, "tlv2_support_score_rank0").ge(score_min) & close.ge(level.mul(1.0 - near)), level
        if concept == "tlv2_sup_break_short":
            level = _num(dataframe, "tlv2_support_line_rank0", np.nan)
            return _num(dataframe, "tlv2_support_score_rank0").ge(score_min) & close.le(level.mul(1.0 + near)), level

        if concept == "vp_poc_reclaim_long":
            level = _num(dataframe, "htfvp_prior_poc", np.nan)
            return _num(dataframe, "htfvp_context_score_bull").ge(0.20) & close.ge(level.mul(1.0 - near)), level
        if concept == "vp_vah_reject_short":
            level = _num(dataframe, "htfvp_prior_vah", np.nan)
            return _num(dataframe, "htfvp_context_score_bear").ge(0.20) & close.le(level.mul(1.0 + near)), level
        if concept == "vp_val_reclaim_long":
            level = _num(dataframe, "htfvp_prior_val", np.nan)
            return _num(dataframe, "htfvp_context_score_bull").ge(0.20) & close.ge(level.mul(1.0 - near)), level
        if concept == "vp_poc_reject_short":
            level = _num(dataframe, "htfvp_prior_poc", np.nan)
            return _num(dataframe, "htfvp_context_score_bear").ge(0.20) & close.le(level.mul(1.0 + near)), level
        if concept == "vp_lvn_traverse_long":
            level = _num(dataframe, "htfvp_lvn_above", np.nan)
            return level.notna(), level
        if concept == "vp_lvn_traverse_short":
            level = _num(dataframe, "htfvp_lvn_below", np.nan)
            return level.notna(), level

        if concept == "liq_equal_lows_long":
            level = _num(dataframe, "mtfx_equal_low", np.nan)
            return low.le(level.mul(1.0 - near)) & close.gt(level), level
        if concept == "liq_equal_highs_short":
            level = _num(dataframe, "mtfx_equal_high", np.nan)
            return high.ge(level.mul(1.0 + near)) & close.lt(level), level
        if concept == "liq_prior_low_long":
            level = _num(dataframe, "mtfx_prior_low", np.nan)
            return low.le(level.mul(1.0 - near)) & close.gt(level), level
        if concept == "liq_prior_high_short":
            level = _num(dataframe, "mtfx_prior_high", np.nan)
            return high.ge(level.mul(1.0 + near)) & close.lt(level), level

        if concept == "demand_zone_long":
            level = _num(dataframe, "mtfx_demand_zone_high", np.nan)
            return close.ge(level.mul(1.0 - near)), level
        if concept == "supply_zone_short":
            level = _num(dataframe, "mtfx_supply_zone_low", np.nan)
            return close.le(level.mul(1.0 + near)), level
        if concept == "avwap_reclaim_long":
            level = _num(dataframe, "mtfx_avwap_from_low", np.nan)
            return close.ge(level.mul(1.0 - near)), level
        if concept == "avwap_reject_short":
            level = _num(dataframe, "mtfx_avwap_from_high", np.nan)
            return close.le(level.mul(1.0 + near)), level
        return false, level

    def _near_level_recent(self, dataframe: DataFrame, level: Series) -> Series:
        near = float(self.level_buffer_pct.value)
        window = int(self.local_window.value)
        low = _num(dataframe, "low")
        high = _num(dataframe, "high")
        touched = low.le(level.mul(1.0 + near)) & high.ge(level.mul(1.0 - near))
        return touched.fillna(False).astype("int8").rolling(window, min_periods=1).max().gt(0)

    def _ltf_trigger(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, "close")
        open_ = _num(dataframe, "open")
        high = _num(dataframe, "high")
        low = _num(dataframe, "low")
        level = _num(dataframe, "mtfx_htf_level", np.nan)
        near = float(self.level_buffer_pct.value)
        trigger = LTF_TRIGGER
        if trigger == "breakout":
            return _cross_above(close, level.mul(1.0 + near))
        if trigger == "breakdown":
            return _cross_below(close, level.mul(1.0 - near))
        if trigger in {"retest", "reclaim"}:
            return low.le(level.mul(1.0 + near)) & close.gt(level) & close.gt(open_)
        if trigger == "reject":
            return high.ge(level.mul(1.0 - near)) & close.lt(level) & close.lt(open_)
        if trigger == "bos":
            event = _bool(dataframe, "ltfms_bos_to_bull") if SIDE == "long" else _bool(dataframe, "ltfms_bos_to_bear")
            return event & self._near_level_recent(dataframe, level)
        if trigger == "choch":
            event = _bool(dataframe, "ltfms_choch_to_bull") if SIDE == "long" else _bool(dataframe, "ltfms_choch_to_bear")
            return event & self._near_level_recent(dataframe, level)
        return pd.Series(False, index=dataframe.index, dtype="bool")

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        condition = _bool(dataframe, "mtfx_htf_context") & self._ltf_trigger(dataframe)
        if bool(self.use_ltf_volume_guard.value):
            condition &= _num(dataframe, "mtfx_ltf_volume_ratio").ge(float(self.volume_ratio_min.value))
        if bool(self.use_ltf_pressure_guard.value):
            pressure = _num(dataframe, "mtfx_ltf_pressure")
            condition &= pressure.le(-float(self.pressure_min.value)) if SIDE == "short" else pressure.ge(float(self.pressure_min.value))
        condition = apply_sieve2_optional_guards(self, dataframe, condition, SIDE)
        valid = condition.fillna(False) & dataframe["volume"].gt(0.0) & dataframe["close"].notna()
        if SIDE == "short":
            dataframe.loc[valid, "enter_short"] = 1
        else:
            dataframe.loc[valid, "enter_long"] = 1
        dataframe.loc[valid, "enter_tag"] = ENTRY_TAG
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe


apply_explicit_hyperopt_surface(Sieve2MtfxH4VpPocRejectShort1hRetest)

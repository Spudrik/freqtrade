from __future__ import annotations



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

from freqtrade.strategy import BooleanParameter, CategoricalParameter, IStrategy, stoploss_from_absolute, DecimalParameter, IntParameter

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



NOVEL_IDEA = True



HYPEROPT_PARAM_ENV = "HYBRID_RECOVERY_HYPEROPT_PARAMS"

ENTRY_SIEVE_TAKE_PROFIT_ENV = "ENTRY_SIEVE_TAKE_PROFIT_PCT"

ENTRY_SIEVE_STOPLOSS_ENV = "ENTRY_SIEVE_STOPLOSS_PCT"

ENTRY_MODE = "entry_sieve2_mtfx_h4_tlv2_sup_break_short_1h_choch"

ENTRY_TAG = "sieve2_mtfx_h4_tlv2_sup_break_short_1h_choch"

SIEVE_STAGE = "sieve3"

SOURCE_STRATEGY = "sieve3/sieve2_mtfx_h4_tlv2_sup_break_short_1h_choch.py:Sieve2MtfxH4Tlv2SupBreakShort1hChoch"

SOURCE_RESULT_BATCH = "sieve3_exit_branch_generation_20260612"

RESEARCH_PATH = "sieve3_exit_multi_partial_structural_ladder"

ENTRY_SOURCE_STAGE = "sieve2_or_sieve3_candidate"

EXIT_HYPOTHESIS = 'Multi-partial ladder with structural target and guard-aware remainder management.'

SIDE = "short"

TIMEFRAME = "1h"

CONTEXT_TIMEFRAME = "4h"

HTF_CONCEPT = "tlv2_sup_break_short"

LTF_TRIGGER = "choch"





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

class Sieve3ExitMultiPartialStructuralLadderFromMtfxH4Tlv2SupBreakShort1HChoch(IStrategy):

    SIEVE2_FUNDAMENTAL_REWORK = "20260605_nonprofitable_loosen_sparse"

    SOURCE_RESULT_BATCH = "20260603T101611_entry_sieve2_guard_revised_weak_unrun"

    REWORK_HYPOTHESIS = "Entry-only rerun after non-profitable revised-guard result: loosen_sparse; widen guard thresholds and adjust entry-strength search space."

    SIEVE2_ALWAYS_ON_GUARDS = True

    RESEARCH_PATH = "guard_revised_weak_unrun"

    """Sieve2 structural MTF expansion: HTF concept plus LTF structural execution."""



    INTERFACE_VERSION = 3

    timeframe = TIMEFRAME

    startup_candle_count = 240

    process_only_new_candles = True

    can_short = True



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



    score_min = tagged_parameter(CategoricalParameter([0.50, 0.60, 0.72, 0.82], default=0.5, space="buy", optimize=True, load=True))

    htf_recent_bars = tagged_parameter(CategoricalParameter([1, 2, 3, 5], default=3, space="buy", optimize=True, load=True))

    level_buffer_pct = tagged_parameter(CategoricalParameter([0.002, 0.004, 0.008, 0.012, 0.016], default=0.004, space="buy", optimize=True, load=True))

    local_window = tagged_parameter(CategoricalParameter([6, 12, 24], default=6, space="buy", optimize=True, load=True))

    use_ltf_volume_guard = tagged_parameter(BooleanParameter(default=True, space="buy", optimize=True, load=True))

    volume_ratio_min = tagged_parameter(CategoricalParameter([0.8, 1.0, 1.3, 1.6], default=1.3, space="buy", optimize=True, load=True))

    use_ltf_pressure_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=True, load=True))

    pressure_min = tagged_parameter(CategoricalParameter([0.03, 0.07, 0.12, 0.18, 0.25], default=0.25, space="buy", optimize=True, load=True))



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





    S3_BRANCH_FAMILY = "multi_partial_structural_ladder"

    EXIT_FAMILY = "multi_partial_structural_ladder"

    EXIT_HYPOTHESIS = 'Multi-partial ladder with structural target and guard-aware remainder management.'

    SOURCE_ENTRY_STEM = "mtfx_h4_tlv2_sup_break_short_1h_choch"

    SOURCE_ENTRY_CLASS = "Sieve2MtfxH4Tlv2SupBreakShort1hChoch"

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





apply_s3_branch_surface(Sieve3ExitMultiPartialStructuralLadderFromMtfxH4Tlv2SupBreakShort1HChoch)

apply_explicit_hyperopt_surface(Sieve3ExitMultiPartialStructuralLadderFromMtfxH4Tlv2SupBreakShort1HChoch)

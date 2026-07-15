from __future__ import annotations
EXIT_HYPOTHESIS = "Broad Sieve3 exit sweep added in place to existing distinct clone."
EXIT_RESEARCH_PATH = "broad_exit_sweep"
ENTRY_SOURCE_STAGE = "sieve3_candidate"
# Sieve3 candidate cloned from `sieve2_reframed_tlv2_support_break_retest_vp_node_short_1h` after clean revised-guard Sieve2 run.
# Source result: 20260603T101611_entry_sieve2_guard_revised_weak_unrun; trades=169; winrate=56.2%; profit_abs=57.741; drawdown=4.09%.
# Entry-only candidate; do not add exit logic until entry sieves are finalised.
SIEVE_STAGE = "sieve3"
SOURCE_RESULT_BATCH = "sieve3_exit_in_place_20260612"

# Held out because this top-level Sieve2 file was not present in the reviewed full Sieve2 result batch.

import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import BooleanParameter, CategoricalParameter, IStrategy
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, stoploss_from_absolute
from user_data.Indicators.complex_trendline_projection_v2 import add_trendline_projection_v2
from user_data.Indicators.complex_volume_profile import add_volume_profile
from user_data.Indicators.pattern_continuation import add_pattern_continuation
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2
from user_data.Indicators.pattern_wolfe_waves import add_pattern_wolfe_waves
from user_data.strategies.sieve_guard_helpers import (
    SIEVE2_MARKET_GUARD_MODES,
    SIEVE2_VP_GUARD_MODES,
    add_sieve2_guard_indicators,
    apply_sieve2_optional_guards,
)

HYPEROPT_PARAM_ENV = "HYBRID_RECOVERY_HYPEROPT_PARAMS"
ENTRY_SIEVE_TAKE_PROFIT_ENV = "ENTRY_SIEVE_TAKE_PROFIT_PCT"
ENTRY_SIEVE_STOPLOSS_ENV = "ENTRY_SIEVE_STOPLOSS_PCT"
SIEVE_STAGE = "sieve3"
SOURCE_STRATEGY = "C:/FreqTradeStuff/user_data/strategies/sieve3_candidates/from_20260603_revised_guard_profitable/sieve3_from_reframed_tlv2_support_break_retest_vp_node_short_1h.py:Sieve3FromReframedTlv2SupportBreakRetestVpNodeShort1h"
SOURCE_RESULT_BATCH = "historical_entry_sieve_results"
RESEARCH_PATH = "sieve3_exit_in_place_broad_sweep"
NOVEL_IDEA = False
ENTRY_MODE = "entry_sieve2_reframed_tlv2_support_break_retest_vp_node_short_1h"
ENTRY_TAG = "sieve2_reframed_tlv2_support_break_retest_vp_node_short_1h"
SIDE = "short"
TIMEFRAME = "1h"
CONTEXT_TIMEFRAME = ""
CONCEPT_FAMILY = "tlv2_vp"
CONCEPT = "support_break_vp_node"
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

class Sieve3ExitFromReframedTlv2SupportBreakRetestVpNodeShort1h(IStrategy):
    SIEVE2_ALWAYS_ON_GUARDS = True
    RESEARCH_PATH = "guard_revised_weak_unrun"
    """Sieve2 reframed missed concept probe."""

    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 420
    process_only_new_candles = True
    can_short = SIDE == "short"

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
    sieve2_vp_score_min = CategoricalParameter([-1.0, 0.05, 0.15, 0.25, 0.35, 0.50, 0.75, 1.0], default=0.15, space="buy", optimize=True, load=True)
    sieve2_vp_context_min = CategoricalParameter([-1.0, 0.05, 0.15, 0.28, 0.40, 0.60, 0.85, 1.0], default=0.15, space="buy", optimize=True, load=True)
    use_sieve2_market_guard = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    sieve2_market_guard_mode = CategoricalParameter(SIEVE2_MARKET_GUARD_MODES, default="pressure_or_trend", space="buy", optimize=True, load=True)
    sieve2_market_window = CategoricalParameter([12, 24, 48, 96], default=24, space="buy", optimize=False, load=True)
    sieve2_market_pressure_min = CategoricalParameter([-1.0, 0.0, 0.03, 0.07, 0.12, 0.18, 0.25, 0.40, 1.0], default=0.03, space="buy", optimize=True, load=True)
    sieve2_market_trend_min = CategoricalParameter([-10.0, -1.0, 0.0, 0.15, 0.25, 0.50, 0.90, 1.50, 3.0], default=0.15, space="buy", optimize=True, load=True)
    sieve2_rs_benchmark_pair = CategoricalParameter(["BTC/USDT:USDT", "ETH/USDT:USDT"], default="BTC/USDT:USDT", space="buy", optimize=False, load=True)
    sieve2_rs_score_min = CategoricalParameter([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.10], default=0.30, space="buy", optimize=True, load=True)

    use_volume_guard = tagged_parameter(BooleanParameter(default=True, space="buy", optimize=True, load=True))
    volume_guard_window = tagged_parameter(CategoricalParameter([12, 24, 48], default=24, space="buy", optimize=True, load=True))
    volume_ratio_min = tagged_parameter(CategoricalParameter([0.8, 1.0, 1.3, 1.6], default=0.8, space="buy", optimize=True, load=True))
    use_pressure_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=True, load=True))
    pressure_window = tagged_parameter(CategoricalParameter([12, 24, 48], default=24, space="buy", optimize=True, load=True))
    pressure_min = tagged_parameter(CategoricalParameter([0.05, 0.10, 0.15, 0.20, 0.35], default=0.35, space="buy", optimize=True, load=True))
    use_body_direction_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=False, load=True))
    use_close_direction_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=False, load=True))
    vp_window = tagged_parameter(CategoricalParameter([48, 96, 144], default=96, space="buy", optimize=False, load=True))
    vp_bins = tagged_parameter(CategoricalParameter([36, 48, 72], default=48, space="buy", optimize=False, load=True))
    vp_score_min = tagged_parameter(CategoricalParameter([0.15, 0.25, 0.35, 0.50], default=0.15, space="buy", optimize=True, load=True))
    vp_context_min = tagged_parameter(CategoricalParameter([0.18, 0.28, 0.38, 0.50], default=0.18, space="buy", optimize=True, load=True))
    level_buffer_pct = tagged_parameter(CategoricalParameter([0.002, 0.004, 0.008, 0.012, 0.016], default=0.012, space="buy", optimize=True, load=True))
    context_recent_bars = tagged_parameter(CategoricalParameter([1, 2, 3, 5], default=1, space="buy", optimize=True, load=True))
    liquidity_lookback = tagged_parameter(CategoricalParameter([24, 48, 96, 168], default=24, space="buy", optimize=True, load=True))
    min_pattern_score = tagged_parameter(CategoricalParameter([0.55, 0.72, 0.82], default=0.72, space="buy", optimize=True, load=True))
    pivot_strength = tagged_parameter(CategoricalParameter([2, 3, 4], default=2, space="buy", optimize=False, load=True))
    min_line_score = tagged_parameter(CategoricalParameter([0.4, 0.5, 0.6], default=0.4, space="buy", optimize=True, load=True))
    max_distance_atr = tagged_parameter(CategoricalParameter([3.0, 6.0, 10.0], default=6.0, space="buy", optimize=True, load=True))
    min_active_bars = tagged_parameter(CategoricalParameter([4, 8, 16], default=8, space="buy", optimize=False, load=True))
    capitulation_move_pct = tagged_parameter(CategoricalParameter([0.025, 0.04, 0.06, 0.08], default=0.06, space="buy", optimize=True, load=True))

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        if not CONTEXT_TIMEFRAME or not getattr(self, "dp", None):
            return []
        return [(pair, CONTEXT_TIMEFRAME) for pair in self.dp.current_whitelist()]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self._add_base_features(dataframe.copy())
        if CONTEXT_TIMEFRAME:
            dataframe = self._merge_htf_context(dataframe, metadata or {})
        dataframe = add_sieve2_guard_indicators(self, dataframe, metadata)
        return dataframe

    def _add_base_features(self, dataframe: DataFrame) -> DataFrame:
        if CONCEPT_FAMILY in {"vp_location", "tlv2_vp", "capitulation_vp"}:
            dataframe = self._add_vp(dataframe)
        if CONCEPT_FAMILY == "tlv2_vp":
            dataframe = self._add_tlv2(dataframe)
        if CONCEPT_FAMILY == "liquidity":
            dataframe = self._add_liquidity(dataframe)
        if CONCEPT_FAMILY == "capitulation_vp":
            dataframe = self._add_capitulation(dataframe)
        return dataframe

    def _add_vp(self, dataframe: DataFrame) -> DataFrame:
        return add_volume_profile(dataframe, window=int(self.vp_window.value), bins=int(self.vp_bins.value), value_area_pct=0.70, price_source="hlc3", smooth_bins=3, pressure_delta_min=0.05, node_near_pct=float(self.level_buffer_pct.value), prefix="vp")

    def _add_tlv2(self, dataframe: DataFrame) -> DataFrame:
        return add_trendline_projection_v2(dataframe, timeframe=self.timeframe, pivot_strength=int(self.pivot_strength.value), raw_line_output_count=1, min_output_line_score=float(self.min_line_score.value), min_output_active_bars=int(self.min_active_bars.value), max_active_line_distance_atr_mult=float(self.max_distance_atr.value), proximity_rank_weight=0.05, output_prefix="tlv2")

    def _add_liquidity(self, dataframe: DataFrame) -> DataFrame:
        close = _num(dataframe, "close")
        high = _num(dataframe, "high")
        low = _num(dataframe, "low")
        lookback = int(self.liquidity_lookback.value)
        tol = float(self.level_buffer_pct.value)
        prior_high = high.shift(1).rolling(lookback, min_periods=max(4, lookback // 4)).max()
        prior_low = low.shift(1).rolling(lookback, min_periods=max(4, lookback // 4)).min()
        equal_high = high.shift(1).rolling(lookback, min_periods=max(4, lookback // 4)).quantile(0.90)
        equal_low = low.shift(1).rolling(lookback, min_periods=max(4, lookback // 4)).quantile(0.10)
        dataframe["liq_sweep_reclaim_long"] = low.lt(prior_low.mul(1.0 - tol)) & close.gt(prior_low)
        dataframe["liq_sweep_reject_short"] = high.gt(prior_high.mul(1.0 + tol)) & close.lt(prior_high)
        dataframe["liq_equal_reclaim_long"] = low.lt(equal_low.mul(1.0 - tol)) & close.gt(equal_low)
        dataframe["liq_equal_reject_short"] = high.gt(equal_high.mul(1.0 + tol)) & close.lt(equal_high)
        return dataframe

    def _add_capitulation(self, dataframe: DataFrame) -> DataFrame:
        close = _num(dataframe, "close")
        open_ = _num(dataframe, "open")
        volume = _num(dataframe, "volume").clip(lower=0.0)
        window = int(self.liquidity_lookback.value)
        move = (close - open_) / close.replace(0.0, np.nan)
        volume_base = volume.shift(1).rolling(window, min_periods=max(4, window // 4)).mean().replace(0.0, np.nan)
        dataframe["capitulation_down"] = move.le(-float(self.capitulation_move_pct.value)) & volume.ge(volume_base.mul(float(self.volume_ratio_min.value)))
        dataframe["capitulation_up"] = move.ge(float(self.capitulation_move_pct.value)) & volume.ge(volume_base.mul(float(self.volume_ratio_min.value)))
        dataframe["recent_capitulation_down"] = dataframe["capitulation_down"].astype("int8").rolling(int(self.context_recent_bars.value), min_periods=1).max().gt(0)
        dataframe["recent_capitulation_up"] = dataframe["capitulation_up"].astype("int8").rolling(int(self.context_recent_bars.value), min_periods=1).max().gt(0)
        return dataframe

    def _merge_htf_context(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        frame = dataframe.copy()
        frame["mtf_context"] = False
        frame["mtf_level"] = np.nan
        if "date" not in frame.columns or not getattr(self, "dp", None):
            return frame
        pair = str((metadata or {}).get("pair") or "")
        if not pair:
            return frame
        informative = self.dp.get_pair_dataframe(pair=pair, timeframe=CONTEXT_TIMEFRAME)
        if informative is None or informative.empty or "date" not in informative.columns:
            return frame
        informative = self._add_htf_features(informative.copy())
        informative = informative[["date", "mtf_context", "mtf_level"]].copy().sort_values("date")
        informative["date_merge"] = pd.to_datetime(informative["date"], utc=True).dt.tz_convert(None) + pd.to_timedelta(timeframe_to_minutes(CONTEXT_TIMEFRAME), unit="m")
        base = frame.drop(columns=["mtf_context", "mtf_level"], errors="ignore").reset_index().rename(columns={"index": "__row_index"}).sort_values("date")
        base["__date_merge"] = pd.to_datetime(base["date"], utc=True).dt.tz_convert(None)
        merged = pd.merge_asof(base, informative.drop(columns=["date"]).sort_values("date_merge"), left_on="__date_merge", right_on="date_merge", direction="backward")
        merged = merged.sort_values("__row_index").drop(columns=["__row_index", "__date_merge", "date_merge"], errors="ignore")
        merged.index = dataframe.index
        merged["mtf_context"] = pd.Series(merged["mtf_context"], index=merged.index).astype("boolean").fillna(False).astype(bool)
        return merged

    def _add_htf_features(self, dataframe: DataFrame) -> DataFrame:
        if CONCEPT_FAMILY == "mtf_continuation":
            dataframe = add_pattern_continuation(dataframe, timeframe=CONTEXT_TIMEFRAME, min_flag_quality=float(self.min_pattern_score.value))
            prefix = "pat_flag" if CONCEPT == "flag" else "pat_pennant"
            context = _bool(dataframe, f"{prefix}_pattern_confirmed") | _bool(dataframe, f"{prefix}_pattern_present")
            dataframe["mtf_context"] = context & _num(dataframe, f"{prefix}_indicator_score").ge(float(self.min_pattern_score.value))
            dataframe["mtf_level"] = _num(dataframe, f"{prefix}_confirmation_level", np.nan)
            return dataframe
        if CONCEPT_FAMILY == "mtf_geometry":
            dataframe = add_pattern_geometry_v2(dataframe, timeframe=CONTEXT_TIMEFRAME, output_slots=1, include_triangle_patterns=CONCEPT == "triangle", include_wedge_patterns=CONCEPT == "wedge", include_compression_patterns=False, include_rectangle_patterns=CONCEPT == "rectangle", include_ascending_channel_patterns=False, include_descending_channel_patterns=False, min_line_score=float(self.min_line_score.value), output_prefix="pg2")
            prefix = f"pg2_{CONCEPT}"
            dataframe["mtf_context"] = _bool(dataframe, f"{prefix}_pattern_present") & _num(dataframe, f"{prefix}_indicator_score").ge(float(self.min_line_score.value))
            dataframe["mtf_level"] = _num(dataframe, f"{prefix}_lower", np.nan) if SIDE == "short" else _num(dataframe, f"{prefix}_upper", np.nan)
            return dataframe
        if CONCEPT_FAMILY == "mtf_wolfe":
            dataframe = add_pattern_wolfe_waves(dataframe, min_wave_quality=float(self.min_pattern_score.value))
            prefix = "pww_bullish" if SIDE == "long" else "pww_bearish"
            dataframe["mtf_context"] = (_bool(dataframe, f"{prefix}_pattern_confirmed") | _bool(dataframe, f"{prefix}_pattern_present")) & _num(dataframe, f"{prefix}_indicator_score").ge(float(self.min_pattern_score.value))
            dataframe["mtf_level"] = _num(dataframe, f"{prefix}_confirmation_level", np.nan)
            return dataframe
        dataframe["mtf_context"] = False
        dataframe["mtf_level"] = np.nan
        return dataframe

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype="bool")
        close = _num(dataframe, "close")
        if bool(self.use_volume_guard.value):
            volume = _num(dataframe, "volume").clip(lower=0.0)
            window = int(self.volume_guard_window.value)
            baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
            guard &= volume.ge(baseline.mul(float(self.volume_ratio_min.value)))
        if bool(self.use_pressure_guard.value):
            pressure = self._directional_pressure(dataframe)
            guard &= pressure.le(-float(self.pressure_min.value)) if SIDE == "short" else pressure.ge(float(self.pressure_min.value))
        if bool(self.use_body_direction_guard.value):
            open_ = _num(dataframe, "open")
            guard &= close.lt(open_) if SIDE == "short" else close.gt(open_)
        if bool(self.use_close_direction_guard.value):
            guard &= close.lt(close.shift(1)) if SIDE == "short" else close.gt(close.shift(1))
        return guard.fillna(False)

    def _directional_pressure(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, "close")
        open_ = _num(dataframe, "open")
        high = _num(dataframe, "high")
        low = _num(dataframe, "low")
        volume = _num(dataframe, "volume").clip(lower=0.0).fillna(0.0)
        candle_range = (high - low).replace(0.0, np.nan)
        body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
        close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
        pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
        window = int(self.pressure_window.value)
        baseline = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
        return (pressure * volume).rolling(window, min_periods=max(2, window // 3)).sum() / baseline

    def _vp_trigger(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, "close")
        open_ = _num(dataframe, "open")
        high = _num(dataframe, "high")
        low = _num(dataframe, "low")
        near = float(self.level_buffer_pct.value)
        long_ok = _num(dataframe, "vp_score_long").ge(float(self.vp_score_min.value)) | _num(dataframe, "vp_context_score_bull").ge(float(self.vp_context_min.value))
        short_ok = _num(dataframe, "vp_score_short").ge(float(self.vp_score_min.value)) | _num(dataframe, "vp_context_score_bear").ge(float(self.vp_context_min.value))
        poc = _num(dataframe, "vp_prior_poc", np.nan)
        if CONCEPT == "vp_poc_reclaim_long":
            return low.le(poc.mul(1.0 + near)) & close.gt(poc) & close.gt(open_) & long_ok
        if CONCEPT == "vp_poc_reject_short":
            return high.ge(poc.mul(1.0 - near)) & close.lt(poc) & close.lt(open_) & short_ok
        if CONCEPT == "vp_hvn_reclaim_long":
            return _bool(dataframe, "vp_hvn_below_reclaim") & long_ok
        if CONCEPT == "vp_hvn_reject_short":
            return _bool(dataframe, "vp_hvn_above_reject") & short_ok
        if CONCEPT == "vp_lvn_accept_long":
            return _bool(dataframe, "vp_lvn_accept_long") & long_ok
        if CONCEPT == "vp_lvn_accept_short":
            return _bool(dataframe, "vp_lvn_accept_short") & short_ok
        if CONCEPT == "vp_lvn_fast_traverse_long":
            return _bool(dataframe, "vp_lvn_fast_traverse_long") & long_ok
        if CONCEPT == "vp_lvn_fast_traverse_short":
            return _bool(dataframe, "vp_lvn_fast_traverse_short") & short_ok
        return pd.Series(False, index=dataframe.index, dtype="bool")

    def _mtf_trigger(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, "close")
        open_ = _num(dataframe, "open")
        high = _num(dataframe, "high")
        low = _num(dataframe, "low")
        level = _num(dataframe, "mtf_level", np.nan)
        near = float(self.level_buffer_pct.value)
        if LTF_TRIGGER == "retest":
            trigger = high.ge(level.mul(1.0 - near)) & close.lt(level) & close.lt(open_) if SIDE == "short" else low.le(level.mul(1.0 + near)) & close.gt(level) & close.gt(open_)
        else:
            trigger = _cross_below(close, level.mul(1.0 - near)) if SIDE == "short" else _cross_above(close, level.mul(1.0 + near))
        return _bool(dataframe, "mtf_context") & trigger

    def _liquidity_trigger(self, dataframe: DataFrame) -> Series:
        if CONCEPT == "equal_lows_reclaim":
            return _bool(dataframe, "liq_equal_reclaim_long")
        if CONCEPT == "equal_highs_reject":
            return _bool(dataframe, "liq_equal_reject_short")
        if CONCEPT == "prior_low_reclaim":
            return _bool(dataframe, "liq_sweep_reclaim_long")
        if CONCEPT == "prior_high_reject":
            return _bool(dataframe, "liq_sweep_reject_short")
        return pd.Series(False, index=dataframe.index, dtype="bool")

    def _tlv2_vp_trigger(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, "close")
        open_ = _num(dataframe, "open")
        high = _num(dataframe, "high")
        low = _num(dataframe, "low")
        near = float(self.level_buffer_pct.value)
        if SIDE == "short":
            line = _num(dataframe, "tlv2_support_line_rank0", np.nan)
            score = _num(dataframe, "tlv2_support_score_rank0")
            distance = _num(dataframe, "tlv2_support_distance_atr_rank0", np.nan)
            vp_ok = _bool(dataframe, "vp_node_entry_short") | _bool(dataframe, "vp_node_hold_short") | _num(dataframe, "vp_score_short").ge(float(self.vp_score_min.value))
            breakout = _cross_below(close, line.mul(1.0 - near))
            retest = high.ge(line.mul(1.0 - near)) & close.lt(line) & close.lt(open_)
        else:
            line = _num(dataframe, "tlv2_resistance_line_rank0", np.nan)
            score = _num(dataframe, "tlv2_resistance_score_rank0")
            distance = _num(dataframe, "tlv2_resistance_distance_atr_rank0", np.nan)
            vp_ok = _bool(dataframe, "vp_node_entry_long") | _bool(dataframe, "vp_node_hold_long") | _num(dataframe, "vp_score_long").ge(float(self.vp_score_min.value))
            breakout = _cross_above(close, line.mul(1.0 + near))
            retest = low.le(line.mul(1.0 + near)) & close.gt(line) & close.gt(open_)
        active = score.ge(float(self.min_line_score.value)) & distance.le(float(self.max_distance_atr.value))
        return active & vp_ok & (retest if LTF_TRIGGER == "retest" else breakout)

    def _capitulation_trigger(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, "close")
        open_ = _num(dataframe, "open")
        poc = _num(dataframe, "vp_prior_poc", np.nan)
        val = _num(dataframe, "vp_prior_val", np.nan)
        vah = _num(dataframe, "vp_prior_vah", np.nan)
        near = float(self.level_buffer_pct.value)
        if SIDE == "short":
            return _bool(dataframe, "recent_capitulation_up") & ((close.lt(vah) & close.lt(open_)) | _cross_below(close, poc.mul(1.0 - near))) & (_num(dataframe, "vp_score_short").ge(float(self.vp_score_min.value)) | _num(dataframe, "vp_context_score_bear").ge(float(self.vp_context_min.value)))
        return _bool(dataframe, "recent_capitulation_down") & ((close.gt(val) & close.gt(open_)) | _cross_above(close, poc.mul(1.0 + near))) & (_num(dataframe, "vp_score_long").ge(float(self.vp_score_min.value)) | _num(dataframe, "vp_context_score_bull").ge(float(self.vp_context_min.value)))

    def _entry_condition(self, dataframe: DataFrame) -> Series:
        if CONCEPT_FAMILY == "vp_location":
            return self._vp_trigger(dataframe)
        if CONCEPT_FAMILY in {"mtf_continuation", "mtf_geometry", "mtf_wolfe"}:
            return self._mtf_trigger(dataframe)
        if CONCEPT_FAMILY == "liquidity":
            return self._liquidity_trigger(dataframe)
        if CONCEPT_FAMILY == "tlv2_vp":
            return self._tlv2_vp_trigger(dataframe)
        if CONCEPT_FAMILY == "capitulation_vp":
            return self._capitulation_trigger(dataframe)
        return pd.Series(False, index=dataframe.index, dtype="bool")

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        condition = self._entry_condition(dataframe)
        condition &= self._common_guards(dataframe)
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
    EXIT_FAMILY = "broad_exit_sweep"
    EXIT_HYPOTHESIS = "Broad Sieve3 exit sweep: fixed exits, partial exits, breakeven shifts, trailing, structural targets, and time stops against the unchanged entry clone."
    SOURCE_ENTRY_STEM = "reframed_tlv2_support_break_retest_vp_node_short_1h"
    S3_BRANCH_FAMILY = EXIT_FAMILY
    PRIMARY_TRIGGER = SOURCE_ENTRY_STEM
    PRIMARY_GUARD = "source_entry_guards_plus_sieve2_optional_guards"
    TARGET_PROVIDER = "explicit_s3_target_provider_order"
    INVALIDATION_PROVIDER = "source_trigger_state_plus_explicit_target_provider"
    EXIT_PARAMETER_VOCABULARY = (
        "exit_mode", "fixed_tp_pct", "fixed_sl_pct", "partial_1_profit", "partial_2_profit",
        "partial_1_fraction", "partial_2_fraction", "breakeven_trigger", "breakeven_offset",
        "trailing_activation", "trailing_distance", "structural_min_profit", "structural_max_profit",
        "structural_proximity_pct", "structural_stop_buffer", "time_stop_candles", "time_stop_min_profit",
    )
    ACTIVE_EXIT_PARAMETERS = s3_active_exit_parameters(S3_BRANCH_FAMILY, "exit_mode")
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

    exit_mode = tagged_exit_parameter(CategoricalParameter(["fixed", "fixed_wide", "partial", "partial_ladder", "be_only", "fixed_be", "trail_only", "partial_trail", "partial_be_trail", "structural", "structural_partial", "structural_trail", "time_profit"], default="fixed", space="sell", optimize=True, load=True))
    fixed_tp_pct = tagged_exit_parameter(CategoricalParameter([0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.060, 0.080, 0.100, 0.120, 0.150], default=0.030, space="sell", optimize=True, load=True))
    fixed_sl_pct = tagged_exit_parameter(CategoricalParameter([0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.060, 0.080], default=0.030, space="sell", optimize=True, load=True))
    partial_1_profit = tagged_exit_parameter(CategoricalParameter([0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050], default=0.020, space="sell", optimize=True, load=True))
    partial_2_profit = tagged_exit_parameter(CategoricalParameter([0.030, 0.040, 0.050, 0.060, 0.080, 0.100, 0.120], default=0.060, space="sell", optimize=True, load=True))
    partial_1_fraction = tagged_exit_parameter(CategoricalParameter([0.20, 0.25, 0.33, 0.50, 0.66, 0.75], default=0.33, space="sell", optimize=True, load=True))
    partial_2_fraction = tagged_exit_parameter(CategoricalParameter([0.20, 0.25, 0.33, 0.50], default=0.33, space="sell", optimize=True, load=True))
    breakeven_trigger = tagged_exit_parameter(CategoricalParameter([0.010, 0.015, 0.020, 0.025, 0.030, 0.040], default=0.020, space="sell", optimize=True, load=True))
    breakeven_offset = tagged_exit_parameter(CategoricalParameter([0.000, 0.001, 0.0025, 0.005], default=0.001, space="sell", optimize=True, load=True))
    trailing_activation = tagged_exit_parameter(CategoricalParameter([0.015, 0.020, 0.030, 0.040, 0.050, 0.080], default=0.030, space="sell", optimize=True, load=True))
    trailing_distance = tagged_exit_parameter(CategoricalParameter([0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050], default=0.015, space="sell", optimize=True, load=True))
    structural_min_profit = tagged_exit_parameter(CategoricalParameter([0.000, 0.010, 0.015, 0.020, 0.030, 0.050, 0.080], default=0.015, space="sell", optimize=True, load=True))
    structural_max_profit = tagged_exit_parameter(CategoricalParameter([0.040, 0.060, 0.080, 0.100, 0.120, 0.160, 0.220, 0.300], default=0.120, space="sell", optimize=True, load=True))
    structural_proximity_pct = tagged_exit_parameter(CategoricalParameter([0.000, 0.003, 0.006, 0.010, 0.015, 0.020, 0.030], default=0.006, space="sell", optimize=True, load=True))
    structural_stop_buffer = tagged_exit_parameter(CategoricalParameter([0.005, 0.010, 0.015, 0.020, 0.030, 0.045, 0.060], default=0.015, space="sell", optimize=True, load=True))
    time_stop_candles = tagged_exit_parameter(CategoricalParameter([6, 12, 24, 48, 96, 168], default=48, space="sell", optimize=True, load=True))
    time_stop_min_profit = tagged_exit_parameter(CategoricalParameter([-0.020, 0.000, 0.005, 0.010, 0.020], default=0.005, space="sell", optimize=True, load=True))

    def _s3_target_provider_order(self):
        tokens = " ".join(str(value).lower() for value in (getattr(self, "SOURCE_ENTRY_STEM", ""), globals().get("CONCEPT_FAMILY", ""), globals().get("CONCEPT", ""), globals().get("LTF_TRIGGER", ""), globals().get("CONTEXT_MODE", ""), globals().get("EXECUTION_MODE", "")))
        providers = []
        if any(token in tokens for token in ("vp", "poc", "vah", "val", "hvn", "lvn", "node")):
            providers.append("vp")
        if any(token in tokens for token in ("tlv2", "trendline", "support", "resistance", "sup", "res")):
            providers.append("tlv2")
        if any(token in tokens for token in ("geometry", "wedge", "triangle", "rectangle", "channel")):
            providers.append("geometry")
        if any(token in tokens for token in ("pivot", "higher_low", "lower_high")):
            providers.append("pivot")
        if any(token in tokens for token in ("bos", "choch")):
            providers.extend(("bos_choch", "pivot"))
        if any(token in tokens for token in ("prior", "equal_high", "equal_low", "liquidity", "liq", "demand", "supply")):
            providers.append("prior_level")
        if any(token in tokens for token in ("avwap", "vwap")):
            providers.append("avwap")
        ordered = []
        for provider in providers:
            if provider not in ordered:
                ordered.append(provider)
        return tuple(ordered)

    def _sieve3_latest_candle(self, pair):
        if not getattr(self, "dp", None):
            return None
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None
        return dataframe.iloc[-1]

    def _sieve3_structural_target(self, pair, trade, current_rate):
        last = self._sieve3_latest_candle(pair)
        if last is None:
            return None
        is_short = bool(getattr(trade, "is_short", False))
        open_rate = float(getattr(trade, "open_rate", 0.0) or 0.0)
        current_rate = float(current_rate or 0.0)
        if open_rate <= 0.0 or current_rate <= 0.0:
            return None
        candidates = []
        side_key = "short" if is_short else "long"
        for provider in self._s3_target_provider_order():
            for col in self.S3_TARGET_PROVIDER_COLUMNS.get(provider, {}).get(side_key, ()):
                if col not in last.index:
                    continue
                value = pd.to_numeric(pd.Series([last[col]]), errors="coerce").iloc[0]
                if not np.isfinite(value) or value <= 0.0:
                    continue
                if (not is_short and value <= open_rate) or (is_short and value >= open_rate):
                    continue
                target_profit = ((value - open_rate) / open_rate) if not is_short else ((open_rate - value) / open_rate)
                if float(self.structural_min_profit.value) <= target_profit <= float(self.structural_max_profit.value):
                    candidates.append((target_profit, abs(value - current_rate) / current_rate))
        return sorted(candidates, key=lambda item: (item[0], item[1]))[0] if candidates else None

    def _sieve3_trade_age_candles(self, trade, current_time):
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
        structural_target = self._sieve3_structural_target(pair, trade, current_rate)
        exits_done = int(getattr(trade, "nr_of_successful_exits", 0) or 0)
        age_candles = self._sieve3_trade_age_candles(trade, current_time) if current_time is not None else 0
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
            "target_source": "explicit_structural_provider" if structural_target is not None else "fixed",
            "nearest_target_distance": structural_target[1] if structural_target is not None else None,
            "target_touched": bool(structural_target is not None and structural_target[1] <= float(self._s3_param_value("structural_proximity_pct", 0.006))),
            "target": structural_target,
            "guard_state": "aligned_or_neutral",
            "trigger_state": "intact",
            "partial_state": partial_state,
            "time_state": time_state,
            "age_candles": age_candles,
        }

    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, after_fill=False, **kwargs):
        _ = current_time, after_fill, kwargs
        mode = str(self.exit_mode.value)
        state = self._s3_trade_state(pair, trade, current_time, current_rate, current_profit)
        stop_profit = -float(self.fixed_sl_pct.value)
        if "be" in mode and float(current_profit) >= float(self.breakeven_trigger.value):
            stop_profit = max(stop_profit, float(self.breakeven_offset.value))
        if "trail" in mode and float(current_profit) >= float(self.trailing_activation.value):
            stop_profit = max(stop_profit, float(current_profit) - float(self.trailing_distance.value))
        if "structural" in mode and state["target"] is not None:
            stop_profit = max(stop_profit, float(current_profit) - float(self.structural_stop_buffer.value))
        open_rate = float(getattr(trade, "open_rate", 0.0) or current_rate or 0.0)
        if open_rate <= 0.0 or current_rate <= 0.0:
            return -float(self.fixed_sl_pct.value)
        stop_price = open_rate * (1.0 + stop_profit) if not bool(getattr(trade, "is_short", False)) else open_rate * (1.0 - stop_profit)
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=bool(getattr(trade, "is_short", False)), leverage=float(getattr(trade, "leverage", 1.0) or 1.0))

    def adjust_trade_position(self, trade, current_time, current_rate, current_profit, min_stake, max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, **kwargs):
        _ = current_time, current_rate, min_stake, max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, kwargs
        if not bool(self.position_adjustment_enable) or bool(getattr(trade, "has_open_orders", False)):
            return None
        mode = str(self.exit_mode.value)
        if "partial" not in mode:
            return None
        exits_done = int(getattr(trade, "nr_of_successful_exits", 0) or 0)
        stake = float(getattr(trade, "stake_amount", 0.0) or 0.0)
        if exits_done == 0 and stake > 0.0 and float(current_profit) >= float(self.partial_1_profit.value):
            return -(stake * float(self.partial_1_fraction.value)), "s3_partial_1"
        if exits_done == 1 and stake > 0.0 and "ladder" in mode and float(current_profit) >= float(self.partial_2_profit.value):
            return -(stake * float(self.partial_2_fraction.value)), "s3_partial_2"
        return None

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        _ = kwargs
        mode = str(self.exit_mode.value)
        state = self._s3_trade_state(pair, trade, current_time, current_rate, current_profit)
        structural_hit = state["target"]
        if "structural" in mode and structural_hit is not None and float(current_profit) >= float(self.structural_min_profit.value):
            if "partial" not in mode or int(getattr(trade, "nr_of_successful_exits", 0) or 0) >= 1:
                return "s3_structural_target"
        if float(current_profit) >= float(self.fixed_tp_pct.value) and "trail" not in mode:
            return "s3_fixed_target"
        if float(current_profit) <= -float(self.fixed_sl_pct.value):
            return "s3_fixed_stop"
        if state["time_state"] == "stale" and float(current_profit) >= float(self.time_stop_min_profit.value):
            return "s3_time_profit_exit"
        return None


apply_s3_branch_surface(Sieve3ExitFromReframedTlv2SupportBreakRetestVpNodeShort1h)
apply_explicit_hyperopt_surface(Sieve3ExitFromReframedTlv2SupportBreakRetestVpNodeShort1h)
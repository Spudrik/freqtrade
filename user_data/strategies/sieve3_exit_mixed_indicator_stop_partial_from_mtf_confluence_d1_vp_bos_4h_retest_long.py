from __future__ import annotations

EXIT_RESEARCH_PATH = "broad_exit_sweep"



# New Sieve2 MTF confluence strategy; entry-only research variant.



import os

from datetime import datetime

from typing import Any



import numpy as np

import pandas as pd

from pandas import DataFrame, Series



from freqtrade.exchange import timeframe_to_minutes

from freqtrade.strategy import BooleanParameter, CategoricalParameter, IStrategy, DecimalParameter, IntParameter, stoploss_from_absolute

from user_data.Indicators.complex_volume_profile import add_volume_profile

from user_data.Indicators.pattern_bos_choch import add_bos_choch

from user_data.strategies.sieve_guard_helpers import (

    SIEVE2_MARKET_GUARD_MODES,

    SIEVE2_VP_GUARD_MODES,

    add_sieve2_guard_indicators,

    apply_sieve2_optional_guards,

)



HYPEROPT_PARAM_ENV = "HYBRID_RECOVERY_HYPEROPT_PARAMS"

ENTRY_SIEVE_TAKE_PROFIT_ENV = "ENTRY_SIEVE_TAKE_PROFIT_PCT"

ENTRY_SIEVE_STOPLOSS_ENV = "ENTRY_SIEVE_STOPLOSS_PCT"

ENTRY_MODE = "entry_mtf_confluence_d1_vp_bos_4h_retest_long"

ENTRY_TAG = "mtf_confluence_d1_vp_bos_4h_retest_long"

SIEVE_STAGE = "sieve3"

SOURCE_STRATEGY = "sieve3/sieve3_exit_mtf_confluence_d1_vp_bos_4h_retest_long.py:Sieve3ExitMTFConfluenceD1VPBOS4hRetestLong"

SOURCE_RESULT_BATCH = "sieve3_exit_branch_generation_20260612"

RESEARCH_PATH = "sieve3_exit_mixed_indicator_stop_partial"

ENTRY_SOURCE_STAGE = "sieve2_or_sieve3_candidate"

EXIT_HYPOTHESIS = 'Mixed branch: indicator/level touch shifts stop and optionally takes partial profit.'

UPDATE_HYPOTHESIS = "D1 VP node reclaim plus BOS context, executed on 4H retest. Gates are mode-specific and count-based, so hyperopt can test two-gate and three-plus-gate confluence."

SIDE = "long"

CONTEXT_TIMEFRAME = "1d"

TIMEFRAME = "4h"

HTF_VP_FOCUS = "node_reclaim"

HTF_STRUCTURE_FOCUS = "bos"

HTF_LEVEL_FOCUS = "prior_break"

LTF_EXECUTION_FOCUS = "break_retest"





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





def _date_naive(value: Series) -> Series:

    return pd.to_datetime(value, utc=True, errors="coerce").dt.tz_convert(None)







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

class Sieve3ExitMixedIndicatorStopPartialFromMtfConfluenceD1VpBos4HRetestLong(IStrategy):

    """Mode-specific MTF confluence probe: HTF confidence, LTF execution."""



    INTERFACE_VERSION = 3

    timeframe = TIMEFRAME

    startup_candle_count = 520

    process_only_new_candles = True

    can_short = False



    minimal_roi = {"0": 100.0}

    stoploss = -0.99

    use_exit_signal = False

    use_custom_stoploss = True

    position_adjustment_enable = True

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



    min_htf_gates = tagged_parameter(CategoricalParameter([1, 2, 3], default=3, space="buy", optimize=True, load=True))

    min_ltf_gates = tagged_parameter(CategoricalParameter([1, 2], default=2, space="buy", optimize=True, load=True))

    min_total_gates = tagged_parameter(CategoricalParameter([2, 3, 4, 5], default=2, space="buy", optimize=True, load=True))

    htf_lookback = tagged_parameter(CategoricalParameter([24, 48, 72, 120], default=72, space="buy", optimize=True, load=True))

    ltf_lookback = tagged_parameter(CategoricalParameter([6, 12, 24, 48], default=24, space="buy", optimize=True, load=True))

    recent_htf_bars = tagged_parameter(CategoricalParameter([1, 2, 3, 5], default=2, space="buy", optimize=True, load=True))

    level_buffer_pct = tagged_parameter(CategoricalParameter([0.0, 0.002, 0.004, 0.008], default=0.004, space="buy", optimize=True, load=True))

    retest_buffer_pct = tagged_parameter(CategoricalParameter([0.002, 0.004, 0.008, 0.012], default=0.012, space="buy", optimize=True, load=True))

    vp_score_min = tagged_parameter(CategoricalParameter([0.15, 0.25, 0.35, 0.50], default=0.25, space="buy", optimize=True, load=True))

    vp_context_min = tagged_parameter(CategoricalParameter([0.18, 0.28, 0.38, 0.50], default=0.5, space="buy", optimize=True, load=True))

    ltf_volume_ratio_min = tagged_parameter(CategoricalParameter([0.8, 1.0, 1.2, 1.5], default=1.5, space="buy", optimize=True, load=True))

    ltf_pressure_min = tagged_parameter(CategoricalParameter([0.03, 0.07, 0.12, 0.18], default=0.18, space="buy", optimize=True, load=True))

    use_ltf_volume_guard = tagged_parameter(BooleanParameter(default=True, space="buy", optimize=True, load=True))

    use_ltf_pressure_guard = tagged_parameter(BooleanParameter(default=True, space="buy", optimize=True, load=True))



    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:

        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs

        return 1.0



    def informative_pairs(self) -> list[tuple[str, str]]:

        if not getattr(self, "dp", None):

            return []

        return [(pair, CONTEXT_TIMEFRAME) for pair in self.dp.current_whitelist()]



    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe = self._add_ltf_features(dataframe)

        dataframe = self._merge_htf_context(dataframe, metadata or {})

        dataframe = add_sieve2_guard_indicators(self, dataframe, metadata)

        return dataframe



    def _add_ltf_features(self, dataframe: DataFrame) -> DataFrame:

        frame = dataframe.copy()

        lookback = int(self.ltf_lookback.value)

        close = _num(frame, "close")

        open_ = _num(frame, "open")

        high = _num(frame, "high")

        low = _num(frame, "low")

        volume = _num(frame, "volume").clip(lower=0.0).fillna(0.0)

        candle_range = (high - low).replace(0.0, np.nan)

        body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)

        close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)

        pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)

        baseline = volume.shift(1).rolling(lookback, min_periods=max(2, lookback // 3)).mean().replace(0.0, np.nan)

        frame["mtf_ltf_pressure"] = pressure

        frame["mtf_ltf_volume_ratio"] = volume / baseline

        frame["mtf_ltf_local_high"] = high.shift(1).rolling(lookback, min_periods=max(2, lookback // 3)).max()

        frame["mtf_ltf_local_low"] = low.shift(1).rolling(lookback, min_periods=max(2, lookback // 3)).min()

        frame = add_volume_profile(frame, window=max(48, min(168, lookback * 4)), bins=48, value_area_pct=0.70, price_source="hlc3", smooth_bins=3, pressure_delta_min=0.05, node_near_pct=0.01, prefix="ltfvp")

        frame = add_bos_choch(frame, strength=3, min_prominence_atr=0.35, min_pivot_spacing_bars=2, max_pivot_age_bars=max(24, min(lookback * 6, 160)), breakout_buffer_atr=0.15, prefix="ltfms")

        return frame



    def _merge_htf_context(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        frame = dataframe.copy()

        frame["mtf_htf_vp_gate"] = False

        frame["mtf_htf_structure_gate"] = False

        frame["mtf_htf_level_gate"] = False

        if "date" not in frame.columns or not getattr(self, "dp", None):

            return frame

        pair = str((metadata or {}).get("pair") or "")

        if not pair:

            return frame

        informative = self.dp.get_pair_dataframe(pair=pair, timeframe=CONTEXT_TIMEFRAME)

        if informative is None or informative.empty or "date" not in informative.columns:

            return frame

        informative = self._add_htf_features(informative.copy())

        keep = ["date", "mtf_htf_vp_gate", "mtf_htf_structure_gate", "mtf_htf_level_gate"]

        informative = informative[keep].copy().sort_values("date")

        informative["date_merge"] = _date_naive(informative["date"]) + pd.to_timedelta(timeframe_to_minutes(CONTEXT_TIMEFRAME), unit="m")

        base = frame.drop(columns=["mtf_htf_vp_gate", "mtf_htf_structure_gate", "mtf_htf_level_gate"], errors="ignore").reset_index().rename(columns={"index": "__row_index"}).sort_values("date")

        base["__date_merge"] = _date_naive(base["date"])

        merged = pd.merge_asof(base, informative.drop(columns=["date"]).sort_values("date_merge"), left_on="__date_merge", right_on="date_merge", direction="backward")

        merged = merged.sort_values("__row_index").drop(columns=["__row_index", "__date_merge", "date_merge"], errors="ignore")

        merged.index = dataframe.index

        for column in ("mtf_htf_vp_gate", "mtf_htf_structure_gate", "mtf_htf_level_gate"):

            merged[column] = pd.Series(merged[column], index=merged.index).astype("boolean").fillna(False).astype(bool)

        return merged



    def _add_htf_features(self, informative: DataFrame) -> DataFrame:

        frame = informative.copy()

        lookback = int(self.htf_lookback.value)

        buffer = float(self.level_buffer_pct.value)

        close = _num(frame, "close")

        open_ = _num(frame, "open")

        high = _num(frame, "high")

        low = _num(frame, "low")

        prior_high = high.shift(1).rolling(lookback, min_periods=max(3, lookback // 4)).max()

        prior_low = low.shift(1).rolling(lookback, min_periods=max(3, lookback // 4)).min()

        ema = close.ewm(span=max(3, lookback // 3), min_periods=max(3, lookback // 6), adjust=False).mean()

        frame = add_volume_profile(frame, window=max(24, min(168, lookback)), bins=48, value_area_pct=0.70, price_source="hlc3", smooth_bins=3, pressure_delta_min=0.05, node_near_pct=0.01, prefix="htfvp")

        frame = add_bos_choch(frame, strength=3, min_prominence_atr=0.35, min_pivot_spacing_bars=2, max_pivot_age_bars=max(24, min(lookback * 2, 160)), breakout_buffer_atr=0.15, prefix="htfms")

        score_long = _num(frame, "htfvp_score_long")

        score_short = _num(frame, "htfvp_score_short")

        ctx_long = _num(frame, "htfvp_context_score_bull")

        ctx_short = _num(frame, "htfvp_context_score_bear")

        prior_vah = _num(frame, "htfvp_prior_vah", np.nan)

        prior_val = _num(frame, "htfvp_prior_val", np.nan)

        prior_poc = _num(frame, "htfvp_prior_poc", np.nan)

        node_long = _bool(frame, "htfvp_node_entry_long") | _bool(frame, "htfvp_node_hold_long")

        node_short = _bool(frame, "htfvp_node_entry_short") | _bool(frame, "htfvp_node_hold_short")

        if SIDE == "short":

            vp_base = (score_short.ge(float(self.vp_score_min.value)) | ctx_short.ge(float(self.vp_context_min.value))) & score_short.ge(score_long)

            if HTF_VP_FOCUS == "value_accept":

                vp_gate = vp_base & close.lt(prior_val)

            elif HTF_VP_FOCUS == "node_reject":

                vp_gate = vp_base | node_short

            elif HTF_VP_FOCUS == "poc_reject":

                vp_gate = vp_base & close.lt(prior_poc.fillna(prior_val))

            elif HTF_VP_FOCUS == "pullback_value":

                vp_gate = vp_base & (high.ge(prior_vah) | _bool(frame, "htfvp_in_value_area"))

            else:

                vp_gate = vp_base

            if HTF_STRUCTURE_FOCUS == "bos":

                structure_gate = _bool(frame, "htfms_bos_to_bear")

            elif HTF_STRUCTURE_FOCUS == "trend":

                structure_gate = _num(frame, "htfms_state").le(0.0) & close.lt(ema)

            else:

                structure_gate = _bool(frame, "htfms_bos_to_bear") | (_num(frame, "htfms_state").le(0.0) & close.lt(ema))

            if HTF_LEVEL_FOCUS == "prior_break":

                level_gate = close.lt(prior_low.mul(1.0 - buffer))

            elif HTF_LEVEL_FOCUS == "sweep_reject":

                level_gate = high.gt(prior_high.mul(1.0 + buffer)) & close.lt(prior_high)

            elif HTF_LEVEL_FOCUS == "resistance_fail":

                level_gate = high.ge(prior_high.mul(1.0 - buffer)) & close.lt(open_)

            else:

                level_gate = close.lt(prior_low.mul(1.0 - buffer)) | (high.gt(prior_high.mul(1.0 + buffer)) & close.lt(prior_high))

        else:

            vp_base = (score_long.ge(float(self.vp_score_min.value)) | ctx_long.ge(float(self.vp_context_min.value))) & score_long.ge(score_short)

            if HTF_VP_FOCUS == "value_accept":

                vp_gate = vp_base & close.gt(prior_vah)

            elif HTF_VP_FOCUS == "node_reclaim":

                vp_gate = vp_base | node_long

            elif HTF_VP_FOCUS == "poc_reclaim":

                vp_gate = vp_base & close.gt(prior_poc.fillna(prior_vah))

            elif HTF_VP_FOCUS == "pullback_value":

                vp_gate = vp_base & (low.le(prior_val) | _bool(frame, "htfvp_in_value_area"))

            else:

                vp_gate = vp_base

            if HTF_STRUCTURE_FOCUS == "bos":

                structure_gate = _bool(frame, "htfms_bos_to_bull")

            elif HTF_STRUCTURE_FOCUS == "trend":

                structure_gate = _num(frame, "htfms_state").ge(0.0) & close.gt(ema)

            else:

                structure_gate = _bool(frame, "htfms_bos_to_bull") | (_num(frame, "htfms_state").ge(0.0) & close.gt(ema))

            if HTF_LEVEL_FOCUS == "prior_break":

                level_gate = close.gt(prior_high.mul(1.0 + buffer))

            elif HTF_LEVEL_FOCUS == "sweep_reclaim":

                level_gate = low.lt(prior_low.mul(1.0 - buffer)) & close.gt(prior_low)

            elif HTF_LEVEL_FOCUS == "support_hold":

                level_gate = low.le(prior_low.mul(1.0 + buffer)) & close.gt(open_)

            else:

                level_gate = close.gt(prior_high.mul(1.0 + buffer)) | (low.lt(prior_low.mul(1.0 - buffer)) & close.gt(prior_low))

        recent = int(self.recent_htf_bars.value)

        frame["mtf_htf_vp_gate"] = vp_gate.fillna(False).rolling(recent, min_periods=1).max().fillna(0).astype(bool)

        frame["mtf_htf_structure_gate"] = structure_gate.fillna(False).rolling(recent, min_periods=1).max().fillna(0).astype(bool)

        frame["mtf_htf_level_gate"] = level_gate.fillna(False).rolling(recent, min_periods=1).max().fillna(0).astype(bool)

        return frame



    def _gate_count(self, gates: list[Series], index: pd.Index) -> Series:

        total = pd.Series(0, index=index, dtype="int64")

        for gate in gates:

            total = total + pd.Series(gate, index=index).fillna(False).astype(bool).astype("int64")

        return total



    def _ltf_gates(self, dataframe: DataFrame) -> list[Series]:

        close = _num(dataframe, "close")

        high = _num(dataframe, "high")

        low = _num(dataframe, "low")

        buffer = float(self.level_buffer_pct.value)

        retest = float(self.retest_buffer_pct.value)

        local_high = _num(dataframe, "mtf_ltf_local_high", np.nan)

        local_low = _num(dataframe, "mtf_ltf_local_low", np.nan)

        pressure = _num(dataframe, "mtf_ltf_pressure")

        volume_ratio = _num(dataframe, "mtf_ltf_volume_ratio")

        if SIDE == "short":

            if LTF_EXECUTION_FOCUS == "failed_reclaim":

                location = high.ge(local_low.mul(1.0 - retest)) & close.lt(local_low)

            elif LTF_EXECUTION_FOCUS == "vp_node":

                location = _bool(dataframe, "ltfvp_node_entry_short") | _bool(dataframe, "ltfvp_node_hold_short")

            else:

                location = _cross_below(close, local_low.mul(1.0 - buffer)) | (high.ge(local_low.mul(1.0 - retest)) & close.lt(local_low))

            structure = _bool(dataframe, "ltfms_bos_to_bear") | _num(dataframe, "ltfms_state").le(0.0)

            context = _bool(dataframe, "ltfvp_node_entry_short") | _bool(dataframe, "ltfvp_node_hold_short") | _num(dataframe, "ltfvp_score_short").ge(float(self.vp_score_min.value)) | pressure.le(-float(self.ltf_pressure_min.value))

            pressure_ok = pressure.le(-float(self.ltf_pressure_min.value))

        else:

            if LTF_EXECUTION_FOCUS == "failed_break_reclaim":

                location = low.le(local_high.mul(1.0 + retest)) & close.gt(local_high)

            elif LTF_EXECUTION_FOCUS == "vp_node":

                location = _bool(dataframe, "ltfvp_node_entry_long") | _bool(dataframe, "ltfvp_node_hold_long")

            else:

                location = _cross_above(close, local_high.mul(1.0 + buffer)) | (low.le(local_high.mul(1.0 + retest)) & close.gt(local_high))

            structure = _bool(dataframe, "ltfms_bos_to_bull") | _num(dataframe, "ltfms_state").ge(0.0)

            context = _bool(dataframe, "ltfvp_node_entry_long") | _bool(dataframe, "ltfvp_node_hold_long") | _num(dataframe, "ltfvp_score_long").ge(float(self.vp_score_min.value)) | pressure.ge(float(self.ltf_pressure_min.value))

            pressure_ok = pressure.ge(float(self.ltf_pressure_min.value))

        if LTF_EXECUTION_FOCUS == "local_bos":

            location = structure

        elif LTF_EXECUTION_FOCUS == "local_break":

            location = _cross_below(close, local_low.mul(1.0 - buffer)) if SIDE == "short" else _cross_above(close, local_high.mul(1.0 + buffer))

        if bool(self.use_ltf_volume_guard.value):

            context &= volume_ratio.ge(float(self.ltf_volume_ratio_min.value))

        if bool(self.use_ltf_pressure_guard.value):

            context &= pressure_ok

        return [location.fillna(False), structure.fillna(False), context.fillna(False)]



    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        _ = metadata

        dataframe["enter_long"] = 0

        dataframe["enter_short"] = 0

        dataframe["enter_tag"] = None

        htf_gates = [_bool(dataframe, "mtf_htf_vp_gate"), _bool(dataframe, "mtf_htf_structure_gate"), _bool(dataframe, "mtf_htf_level_gate")]

        ltf_gates = self._ltf_gates(dataframe)

        htf_count = self._gate_count(htf_gates, dataframe.index)

        ltf_count = self._gate_count(ltf_gates, dataframe.index)

        total_count = htf_count + ltf_count

        condition = htf_count.ge(int(self.min_htf_gates.value)) & ltf_count.ge(int(self.min_ltf_gates.value)) & total_count.ge(int(self.min_total_gates.value))

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



    S3_BRANCH_FAMILY = "mixed_indicator_stop_partial"

    EXIT_FAMILY = "mixed_indicator_stop_partial"

    EXIT_HYPOTHESIS = 'Mixed branch: indicator/level touch shifts stop and optionally takes partial profit.'

    SOURCE_ENTRY_STEM = "mtf_confluence_d1_vp_bos_4h_retest_long"

    SOURCE_ENTRY_CLASS = "Sieve3ExitMTFConfluenceD1VPBOS4hRetestLong"

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





apply_s3_branch_surface(Sieve3ExitMixedIndicatorStopPartialFromMtfConfluenceD1VpBos4HRetestLong)

apply_explicit_hyperopt_surface(Sieve3ExitMixedIndicatorStopPartialFromMtfConfluenceD1VpBos4HRetestLong)

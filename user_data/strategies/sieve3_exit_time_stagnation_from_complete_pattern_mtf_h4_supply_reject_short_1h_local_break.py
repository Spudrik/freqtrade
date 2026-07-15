from __future__ import annotations

EXIT_RESEARCH_PATH = "broad_exit_sweep"



# Parked as complete-pattern candidate: clean/high-win low-volume result shape; not active Sieve2 rescue.



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

ENTRY_MODE = "entry_sieve2_mtf_h4_supply_reject_short_1h_local_break"

ENTRY_TAG = "sieve2_mtf_h4_supply_reject_short_1h_local_break"

SIEVE_STAGE = "sieve3"

SOURCE_STRATEGY = "sieve3/sieve3_exit_complete_pattern_mtf_h4_supply_reject_short_1h_local_break.py:Sieve3ExitCompletePatternMtfH4SupplyRejectShort1hLocalBreak"

SOURCE_RESULT_BATCH = "sieve3_exit_branch_generation_20260612"

RESEARCH_PATH = "sieve3_exit_time_stagnation"

ENTRY_SOURCE_STAGE = "sieve2_or_sieve3_candidate"

EXIT_HYPOTHESIS = 'Time and stagnation exit sweep for trades failing to progress.'

SIDE = "short"

TIMEFRAME = "1h"

CONTEXT_TIMEFRAME = "4h"

HTF_BEHAVIOR = "res_reject_short"

LTF_TRIGGER = "local_break"





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

class Sieve3ExitTimeStagnationFromCompletePatternMtfH4SupplyRejectShort1HLocalBreak(IStrategy):

    """MTF Sieve2 probe: 4h supply/resistance rejection; 1h local breakdown."""



    INTERFACE_VERSION = 3

    timeframe = TIMEFRAME

    startup_candle_count = 420

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



    context_lookback = tagged_parameter(CategoricalParameter([24, 48, 72, 120], default=24, space="buy", optimize=True, load=True))

    context_recent_bars = tagged_parameter(CategoricalParameter([1, 2, 3, 5], default=2, space="buy", optimize=True, load=True))

    local_lookback = tagged_parameter(CategoricalParameter([6, 12, 24, 48], default=48, space="buy", optimize=True, load=True))

    retest_buffer_pct = tagged_parameter(CategoricalParameter([0.002, 0.004, 0.008, 0.012], default=0.012, space="buy", optimize=True, load=True))

    breakout_buffer_pct = tagged_parameter(CategoricalParameter([0.0, 0.002, 0.004, 0.008], default=0.008, space="buy", optimize=True, load=True))

    volume_ratio_min = tagged_parameter(CategoricalParameter([0.8, 1.0, 1.2, 1.5], default=0.8, space="buy", optimize=True, load=True))

    pressure_min = tagged_parameter(CategoricalParameter([0.03, 0.07, 0.12, 0.18], default=0.07, space="buy", optimize=True, load=True))

    vp_score_min = tagged_parameter(CategoricalParameter([0.15, 0.25, 0.35, 0.50], default=0.25, space="buy", optimize=True, load=True))

    vp_context_min = tagged_parameter(CategoricalParameter([0.18, 0.28, 0.38, 0.50], default=0.38, space="buy", optimize=True, load=True))

    use_ltf_volume_guard = tagged_parameter(BooleanParameter(default=True, space="buy", optimize=True, load=True))

    use_ltf_pressure_guard = tagged_parameter(BooleanParameter(default=True, space="buy", optimize=True, load=True))

    use_htf_turn_guard = tagged_parameter(BooleanParameter(default=True, space="buy", optimize=True, load=True))



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

        close = _num(frame, "close")

        open_ = _num(frame, "open")

        high = _num(frame, "high")

        low = _num(frame, "low")

        volume = _num(frame, "volume").clip(lower=0.0).fillna(0.0)

        window = int(self.local_lookback.value)

        candle_range = (high - low).replace(0.0, np.nan)

        close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)

        body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)

        pressure = ((close_location.fillna(0.0) + body_pressure.fillna(0.0)) / 2.0).clip(-1.0, 1.0)

        volume_baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)

        frame["mtf_ltf_pressure"] = pressure

        frame["mtf_ltf_volume_ratio"] = volume / volume_baseline

        frame["mtf_ltf_ema"] = close.ewm(span=window, min_periods=max(2, window // 3), adjust=False).mean()

        frame["mtf_ltf_local_high"] = high.shift(1).rolling(window, min_periods=max(2, window // 3)).max()

        frame["mtf_ltf_local_low"] = low.shift(1).rolling(window, min_periods=max(2, window // 3)).min()

        return frame



    def _merge_htf_context(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        frame = dataframe.copy()

        for column, default in (("mtf_htf_context", False), ("mtf_htf_turn_bad", True)):

            frame[column] = default

        frame["mtf_htf_level"] = np.nan

        if "date" not in frame.columns or not getattr(self, "dp", None):

            return frame

        pair = str((metadata or {}).get("pair") or "")

        if not pair:

            return frame

        informative = self.dp.get_pair_dataframe(pair=pair, timeframe=CONTEXT_TIMEFRAME)

        if informative is None or informative.empty or "date" not in informative.columns:

            return frame

        informative = self._add_htf_context(informative.copy())

        keep = ["date", "mtf_htf_context", "mtf_htf_level", "mtf_htf_turn_bad"]

        informative = informative[[column for column in keep if column in informative.columns]].copy().sort_values("date")

        informative["date_merge"] = pd.to_datetime(informative["date"], utc=True).dt.tz_convert(None) + pd.to_timedelta(timeframe_to_minutes(CONTEXT_TIMEFRAME), unit="m")

        base = frame.drop(columns=["mtf_htf_context", "mtf_htf_level", "mtf_htf_turn_bad"], errors="ignore").reset_index().rename(columns={"index": "__row_index"}).sort_values("date")

        base["__date_merge"] = pd.to_datetime(base["date"], utc=True).dt.tz_convert(None)

        merged = pd.merge_asof(base, informative.drop(columns=["date"]).sort_values("date_merge"), left_on="__date_merge", right_on="date_merge", direction="backward")

        merged = merged.sort_values("__row_index").drop(columns=["__row_index", "__date_merge", "date_merge"], errors="ignore")

        merged.index = dataframe.index

        merged["mtf_htf_context"] = pd.Series(merged["mtf_htf_context"], index=merged.index).astype("boolean").fillna(False).astype(bool)

        merged["mtf_htf_turn_bad"] = pd.Series(merged["mtf_htf_turn_bad"], index=merged.index).astype("boolean").fillna(True).astype(bool)

        return merged



    def _add_htf_context(self, informative: DataFrame) -> DataFrame:

        frame = informative.copy()

        lookback = int(self.context_lookback.value)

        recent = int(self.context_recent_bars.value)

        buffer = float(self.breakout_buffer_pct.value)

        pressure_min = float(self.pressure_min.value)

        close = _num(frame, "close")

        open_ = _num(frame, "open")

        high = _num(frame, "high")

        low = _num(frame, "low")

        volume = _num(frame, "volume").clip(lower=0.0).fillna(0.0)

        candle_range = (high - low).replace(0.0, np.nan)

        pressure = (((close - open_) / candle_range).clip(-1.0, 1.0).fillna(0.0) + (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0).fillna(0.0)) / 2.0

        volume_sum = volume.rolling(lookback, min_periods=max(3, lookback // 4)).sum().replace(0.0, np.nan)

        pressure_ratio = (pressure * volume).rolling(lookback, min_periods=max(3, lookback // 4)).sum() / volume_sum

        prior_high = high.shift(1).rolling(lookback, min_periods=max(3, lookback // 4)).max()

        prior_low = low.shift(1).rolling(lookback, min_periods=max(3, lookback // 4)).min()

        trend_ema = close.ewm(span=max(3, lookback // 3), min_periods=max(3, lookback // 6), adjust=False).mean()

        frame = add_volume_profile(frame, window=min(max(lookback, 24), 168), bins=48, value_area_pct=0.70, price_source="hlc3", smooth_bins=3, pressure_delta_min=0.05, node_near_pct=0.01, prefix="htfvp")

        frame = add_bos_choch(frame, strength=3, min_prominence_atr=0.35, min_pivot_spacing_bars=2, max_pivot_age_bars=max(24, min(lookback * 2, 160)), breakout_buffer_atr=0.15, prefix="htfms")

        vp_score_long = _num(frame, "htfvp_score_long")

        vp_score_short = _num(frame, "htfvp_score_short")

        vp_ctx_long = _num(frame, "htfvp_context_score_bull")

        vp_ctx_short = _num(frame, "htfvp_context_score_bear")

        prior_vah = _num(frame, "htfvp_prior_vah", np.nan)

        prior_val = _num(frame, "htfvp_prior_val", np.nan)

        node_long = _bool(frame, "htfvp_node_entry_long") | _bool(frame, "htfvp_node_hold_long")

        node_short = _bool(frame, "htfvp_node_entry_short") | _bool(frame, "htfvp_node_hold_short")

        vp_long_ok = (vp_score_long.ge(float(self.vp_score_min.value)) | vp_ctx_long.ge(float(self.vp_context_min.value))) & vp_score_long.ge(vp_score_short)

        vp_short_ok = (vp_score_short.ge(float(self.vp_score_min.value)) | vp_ctx_short.ge(float(self.vp_context_min.value))) & vp_score_short.ge(vp_score_long)

        behavior = HTF_BEHAVIOR

        if behavior in {"res_break_long", "prior_high_break_long", "supply_break_long"}:

            context = close.gt(prior_high.mul(1.0 + buffer))

            level = prior_high

        elif behavior == "vp_value_accept_long":

            context = close.gt(prior_vah) & vp_long_ok

            level = prior_vah

        elif behavior == "vp_node_break_long":

            context = (node_long | vp_long_ok) & close.gt(prior_vah)

            level = prior_vah.fillna(prior_high)

        elif behavior == "support_reclaim_long":

            context = low.lt(prior_low.mul(1.0 - buffer)) & close.gt(prior_low) & pressure_ratio.ge(-pressure_min)

            level = prior_low

        elif behavior == "bos_bull_long":

            context = _bool(frame, "htfms_bos_to_bull") | (_num(frame, "htfms_state").ge(0.0) & close.gt(trend_ema) & pressure_ratio.ge(-pressure_min))

            level = prior_high.fillna(trend_ema)

        elif behavior in {"prior_low_break_short", "support_break_short"}:

            context = close.lt(prior_low.mul(1.0 - buffer))

            level = prior_low

        elif behavior == "vp_value_break_short":

            context = close.lt(prior_val) & vp_short_ok

            level = prior_val

        elif behavior in {"vp_vah_reject_short", "vp_node_reject_short"}:

            context = high.ge(prior_vah) & close.lt(prior_vah) & (vp_short_ok | node_short)

            level = prior_vah

        elif behavior == "res_reject_short":

            context = high.gt(prior_high.mul(1.0 - buffer)) & close.lt(prior_high) & pressure_ratio.le(pressure_min)

            level = prior_high

        elif behavior == "bos_bear_short":

            context = _bool(frame, "htfms_bos_to_bear") | (_num(frame, "htfms_state").le(0.0) & close.lt(trend_ema) & pressure_ratio.le(pressure_min))

            level = prior_low.fillna(trend_ema)

        else:

            context = pd.Series(False, index=frame.index)

            level = pd.Series(np.nan, index=frame.index)

        frame["mtf_htf_context"] = context.fillna(False).astype("int8").rolling(recent, min_periods=1).max().gt(0)

        frame["mtf_htf_level"] = level

        if SIDE == "short":

            frame["mtf_htf_turn_bad"] = (close.gt(trend_ema) & pressure_ratio.ge(pressure_min)).fillna(False)

        else:

            frame["mtf_htf_turn_bad"] = (close.lt(trend_ema) & pressure_ratio.le(-pressure_min)).fillna(False)

        return frame



    def _ltf_trigger(self, dataframe: DataFrame) -> Series:

        close = _num(dataframe, "close")

        open_ = _num(dataframe, "open")

        high = _num(dataframe, "high")

        low = _num(dataframe, "low")

        level = _num(dataframe, "mtf_htf_level", np.nan)

        ema = _num(dataframe, "mtf_ltf_ema", np.nan)

        local_high = _num(dataframe, "mtf_ltf_local_high", np.nan)

        local_low = _num(dataframe, "mtf_ltf_local_low", np.nan)

        retest_buffer = float(self.retest_buffer_pct.value)

        breakout_buffer = float(self.breakout_buffer_pct.value)

        pressure = _num(dataframe, "mtf_ltf_pressure")

        if LTF_TRIGGER == "local_break":

            return close.lt(local_low.mul(1.0 - breakout_buffer)) if SIDE == "short" else close.gt(local_high.mul(1.0 + breakout_buffer))

        if LTF_TRIGGER == "pullback_reclaim":

            return (high.ge(ema.mul(1.0 - retest_buffer)) & close.lt(ema) & close.lt(open_)) if SIDE == "short" else (low.le(ema.mul(1.0 + retest_buffer)) & close.gt(ema) & close.gt(open_))

        if LTF_TRIGGER == "momentum_confirm":

            return (close.lt(close.shift(1)) & pressure.le(-float(self.pressure_min.value))) if SIDE == "short" else (close.gt(close.shift(1)) & pressure.ge(float(self.pressure_min.value)))

        if LTF_TRIGGER == "rejection_reclaim":

            return (high.ge(level.mul(1.0 - retest_buffer)) & close.lt(level) & close.lt(open_)) if SIDE == "short" else (low.le(level.mul(1.0 + retest_buffer)) & close.gt(level) & close.gt(open_))

        return (high.ge(level.mul(1.0 - retest_buffer)) & close.lt(level) & close.lt(open_)) if SIDE == "short" else (low.le(level.mul(1.0 + retest_buffer)) & close.gt(level) & close.gt(open_))



    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        _ = metadata

        dataframe["enter_long"] = 0

        dataframe["enter_short"] = 0

        dataframe["enter_tag"] = None

        condition = _bool(dataframe, "mtf_htf_context") & self._ltf_trigger(dataframe)

        if bool(self.use_ltf_volume_guard.value):

            condition &= _num(dataframe, "mtf_ltf_volume_ratio").ge(float(self.volume_ratio_min.value))

        if bool(self.use_ltf_pressure_guard.value):

            pressure = _num(dataframe, "mtf_ltf_pressure")

            condition &= pressure.le(-float(self.pressure_min.value)) if SIDE == "short" else pressure.ge(float(self.pressure_min.value))

        if bool(self.use_htf_turn_guard.value):

            condition &= ~_bool(dataframe, "mtf_htf_turn_bad")

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



    S3_BRANCH_FAMILY = "time_stagnation"

    EXIT_FAMILY = "time_stagnation"

    EXIT_HYPOTHESIS = 'Time and stagnation exit sweep for trades failing to progress.'

    SOURCE_ENTRY_STEM = "complete_pattern_mtf_h4_supply_reject_short_1h_local_break"

    SOURCE_ENTRY_CLASS = "Sieve3ExitCompletePatternMtfH4SupplyRejectShort1hLocalBreak"

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





apply_s3_branch_surface(Sieve3ExitTimeStagnationFromCompletePatternMtfH4SupplyRejectShort1HLocalBreak)

apply_explicit_hyperopt_surface(Sieve3ExitTimeStagnationFromCompletePatternMtfH4SupplyRejectShort1HLocalBreak)

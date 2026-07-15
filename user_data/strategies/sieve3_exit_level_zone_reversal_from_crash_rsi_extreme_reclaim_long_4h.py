from __future__ import annotations

import os

from freqtrade.exchange import timeframe_to_minutes



NOVEL_IDEA = True

UPDATE_HYPOTHESIS = "4h crash pressure with oversold RSI should produce stronger rebound windows."



import numpy as np

import pandas as pd

from pandas import DataFrame, Series

from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, IStrategy, stoploss_from_absolute, BooleanParameter



SIDE = "long"

ENTRY_MODE = "rsi_extreme_reclaim"

ENTRY_TAG = "crash_rsi_extreme_reclaim_long_4h"

SIEVE_STAGE = "sieve3"

SOURCE_STRATEGY = "sieve3/sieve2_crash_rsi_extreme_reclaim_long_4h.py:Sieve2CrashRsiExtremeReclaimLong4h"

SOURCE_RESULT_BATCH = "sieve3_exit_branch_generation_20260612"

RESEARCH_PATH = "sieve3_exit_level_zone_reversal"

ENTRY_SOURCE_STAGE = "sieve2_or_sieve3_candidate"

EXIT_HYPOTHESIS = 'Target-zone exit sweep using prior swing, VP, or indicator levels with bands, offsets, reversal confirmation, and selectable trigger timeframe.'



def tagged_parameter(param):

    setattr(param, "batch_tags", ("family:entries", f"mode:{ENTRY_MODE}"))

    return param





def _safe_div(numer: Series, denom: Series) -> Series:

    denom = denom.replace(0.0, np.nan)

    return numer / denom



def _num(frame: DataFrame, column: str, default: float | Series = 0.0) -> Series:
    if column not in frame.columns:
        if isinstance(default, Series):
            return pd.to_numeric(default, errors="coerce")
        return pd.Series(float(default), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)





def _rsi(close: Series, length: int = 14) -> Series:

    delta = close.diff()

    gain = delta.clip(lower=0.0).ewm(alpha=1 / float(length), adjust=False, min_periods=max(2, length // 2)).mean()

    loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / float(length), adjust=False, min_periods=max(2, length // 2)).mean()

    rs = gain / loss.replace(0.0, np.nan)

    return 100.0 - (100.0 / (1.0 + rs))





def _atr(frame: DataFrame, length: int = 14) -> Series:

    high = pd.to_numeric(frame["high"], errors="coerce")

    low = pd.to_numeric(frame["low"], errors="coerce")

    close = pd.to_numeric(frame["close"], errors="coerce")

    prev_close = close.shift(1)

    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

    return tr.rolling(length, min_periods=max(2, length // 2)).mean()







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





def tagged_exit_parameter(param):

    setattr(param, "batch_tags", ("family:exits", "mode:sieve3_exit"))

    return param





def s3_active_exit_parameters(family: str, mode_name: str = "exit_path_mode") -> tuple[str, ...]:
    _ = family, mode_name
    return (
        "fixed_sl_pct",
        "level_source_mode",
        "s3_level_lookback",
        "level_band_pct",
        "level_touch_offset_pct",
        "level_trigger_timeframe",
        "level_confirmation_mode",
        "level_action_mode",
        "profit_gate_mode",
        "partial_1_fraction",
        "breakeven_offset",
        "trailing_distance",
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
        "time_stop_candles", "time_stop_min_profit", "level_source_mode", "s3_level_lookback", "level_band_pct", "level_touch_offset_pct", "level_trigger_timeframe", "level_confirmation_mode", "level_action_mode", "profit_gate_mode",
    ):
        value = getattr(strategy_cls, name, None)
        if is_parameter_object(value):
            value.optimize = name in active

class Sieve3ExitLevelZoneReversalFromCrashRsiExtremeReclaimLong4H(IStrategy):

    timeframe = "4h"

    can_short = False

    startup_candle_count = 320

    minimal_roi = {"0": 100.0}

    stoploss = -0.99

    process_only_new_candles = True

    use_exit_signal = False

    use_custom_stoploss = True

    position_adjustment_enable = True

    max_entry_position_adjustment = 0

    INTERFACE_VERSION = 3



    crash_lookback = tagged_parameter(IntParameter(8, 56, default=14, space="buy", optimize=True, load=True))

    compression_lookback = tagged_parameter(IntParameter(3, 18, default=6, space="buy", optimize=True, load=True))

    drop_pct_min = tagged_parameter(DecimalParameter(0.04, 0.35, default=0.11, decimals=2, space="buy", optimize=True, load=True))

    compression_width_max = tagged_parameter(DecimalParameter(0.03, 0.50, default=0.39, decimals=2, space="buy", optimize=True, load=True))

    volume_ratio_min = tagged_parameter(DecimalParameter(0.70, 3.50, default=1.21, decimals=2, space="buy", optimize=True, load=True))

    price_spike_pct_min = tagged_parameter(DecimalParameter(0.01, 0.20, default=0.178, decimals=3, space="buy", optimize=True, load=True))

    rsi_max = tagged_parameter(IntParameter(5, 30, default=28, space="buy", optimize=True, load=True))

    reclaim_buffer_pct = tagged_parameter(DecimalParameter(0.001, 0.03, default=0.029, decimals=3, space="buy", optimize=True, load=True))

    rsi_gate_mode = tagged_parameter(CategoricalParameter(["off", "soft", "hard"], default='soft', space="buy", optimize=True, load=True))



    def populate_indicators(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:

        close = pd.to_numeric(dataframe["close"], errors="coerce")

        high = pd.to_numeric(dataframe["high"], errors="coerce")

        low = pd.to_numeric(dataframe["low"], errors="coerce")

        open_ = pd.to_numeric(dataframe["open"], errors="coerce")

        volume = pd.to_numeric(dataframe["volume"], errors="coerce").fillna(0.0)

        look = int(self.crash_lookback.value)

        comp = int(self.compression_lookback.value)

        roll_high = close.rolling(look, min_periods=max(2, look // 2)).max()

        roll_low = close.rolling(look, min_periods=max(2, look // 2)).min()

        comp_high = high.rolling(comp, min_periods=max(2, comp // 2)).max()

        comp_low = low.rolling(comp, min_periods=max(2, comp // 2)).min()

        comp_mean = close.rolling(comp, min_periods=max(2, comp // 2)).mean()

        dataframe["rsi"] = _rsi(close, 14)

        dataframe["atr"] = _atr(dataframe, 14)

        dataframe["drop_pct"] = _safe_div(roll_high - close, roll_high).fillna(0.0)

        dataframe["compression_width"] = _safe_div(comp_high - comp_low, comp_mean).fillna(0.0)

        dataframe["compression_high"] = comp_high

        dataframe["compression_low"] = comp_low

        dataframe["compression_mid"] = (comp_high + comp_low) / 2.0

        dataframe["rolling_low"] = roll_low

        dataframe["volume_ratio"] = _safe_div(volume, volume.rolling(comp, min_periods=max(2, comp // 2)).mean()).fillna(0.0)

        dataframe["price_spike"] = _safe_div((close - close.shift(1)).abs(), close.shift(1).abs()).fillna(0.0)

        dataframe["body_ratio"] = _safe_div((close - open_).abs(), high - low).fillna(0.0)

        dataframe["lower_wick_ratio"] = _safe_div(np.minimum(open_, close) - low, high - low).fillna(0.0)

        dataframe["upper_wick_ratio"] = _safe_div(high - np.maximum(open_, close), high - low).fillna(0.0)

        dataframe["trend_mean"] = close.rolling(max(look * 2, comp * 2), min_periods=max(3, look)).mean()

        dataframe = self._s3_lzr_add_levels(dataframe)

        return dataframe



    def _rsi_ok(self, dataframe: DataFrame) -> Series:

        rsi = pd.to_numeric(dataframe["rsi"], errors="coerce")

        gate = str(self.rsi_gate_mode.value)

        if gate == "off":

            return pd.Series(True, index=dataframe.index)

        if gate == "hard":

            return rsi.le(float(self.rsi_max.value) - 2.0) & pd.to_numeric(dataframe["price_spike"], errors="coerce").ge(float(self.price_spike_pct_min.value))

        return rsi.le(float(self.rsi_max.value))



    def _entry_condition(self, dataframe: DataFrame) -> Series:

        drop_ok = pd.to_numeric(dataframe["drop_pct"], errors="coerce").ge(float(self.drop_pct_min.value))

        compress_ok = pd.to_numeric(dataframe["compression_width"], errors="coerce").le(float(self.compression_width_max.value))

        volume_ok = pd.to_numeric(dataframe["volume_ratio"], errors="coerce").ge(float(self.volume_ratio_min.value))

        spike_ok = volume_ok | pd.to_numeric(dataframe["price_spike"], errors="coerce").ge(float(self.price_spike_pct_min.value))

        close = pd.to_numeric(dataframe["close"], errors="coerce")

        open_ = pd.to_numeric(dataframe["open"], errors="coerce")

        high = pd.to_numeric(dataframe["high"], errors="coerce")

        low = pd.to_numeric(dataframe["low"], errors="coerce")

        reclaim = close.gt(pd.to_numeric(dataframe["compression_high"], errors="coerce") * (1.0 + float(self.reclaim_buffer_pct.value)))

        compression_mid = pd.to_numeric(dataframe["compression_mid"], errors="coerce")

        rolling_low = pd.to_numeric(dataframe["rolling_low"], errors="coerce")

        trend_mean = pd.to_numeric(dataframe["trend_mean"], errors="coerce")

        rsi_ok = self._rsi_ok(dataframe)



        if ENTRY_MODE == "drop_compress_reclaim":

            return drop_ok & compress_ok & reclaim & spike_ok & (close.gt(open_))

        if ENTRY_MODE == "flush_reclaim":

            return drop_ok & pd.to_numeric(dataframe["lower_wick_ratio"], errors="coerce").ge(0.55) & close.gt(compression_mid) & spike_ok

        if ENTRY_MODE == "rsi_extreme_reclaim":

            return drop_ok & rsi_ok & close.gt(compression_mid) & spike_ok

        if ENTRY_MODE == "liquidity_sweep_snapback":

            swept = low.lt(rolling_low * (1.0 - float(self.reclaim_buffer_pct.value)))

            return swept & close.gt(rolling_low) & pd.to_numeric(dataframe["lower_wick_ratio"], errors="coerce").ge(0.45) & spike_ok

        if ENTRY_MODE == "regime_pullback_reclaim":

            return drop_ok & close.lt(trend_mean) & close.gt(trend_mean * (1.0 - float(self.reclaim_buffer_pct.value))) & spike_ok

        if ENTRY_MODE == "volatility_squeeze_reversal":

            return compress_ok & reclaim & spike_ok

        if ENTRY_MODE == "double_flush_reversal":

            prior_low = rolling_low.shift(1)

            prior_prior_low = rolling_low.shift(2)

            first_flush = low.lt(prior_low * (1.0 - float(self.reclaim_buffer_pct.value)))

            second_flush = low.shift(1).lt(prior_prior_low * (1.0 - float(self.reclaim_buffer_pct.value)))

            return first_flush & second_flush & close.gt(compression_mid) & spike_ok

        return pd.Series(False, index=dataframe.index)



    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:

        condition = self._entry_condition(dataframe)

        dataframe.loc[condition, ["enter_long", "enter_tag"]] = (1, ENTRY_TAG)

        return dataframe



    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:

        return dataframe



    S3_BRANCH_FAMILY = "level_zone_reversal"

    EXIT_FAMILY = "level_zone_reversal"

    EXIT_HYPOTHESIS = 'Target-zone exit sweep using prior swing, VP, or indicator levels with bands, offsets, reversal confirmation, and selectable trigger timeframe.'

    SOURCE_ENTRY_STEM = "crash_rsi_extreme_reclaim_long_4h"

    SOURCE_ENTRY_CLASS = "Sieve2CrashRsiExtremeReclaimLong4h"

    PRIMARY_TRIGGER = SOURCE_ENTRY_STEM

    PRIMARY_GUARD = "source_entry_guards_plus_sieve2_optional_guards"

    TARGET_PROVIDER = "prior_swing_or_vp_or_indicator_level_zone"

    INVALIDATION_PROVIDER = "level_zone_rejection_or_fixed_stop"

    EXIT_PARAMETER_VOCABULARY = (

        "exit_path_mode", "fixed_tp_pct", "fixed_sl_pct", "partial_1_profit", "partial_2_profit",

        "partial_1_fraction", "partial_2_fraction", "breakeven_trigger", "breakeven_offset",

        "trailing_activation", "trailing_distance", "indicator_min_profit", "indicator_max_profit",

        "indicator_near_pct", "indicator_stop_buffer", "guard_tighten_buffer", "guard_profit_floor",

        "time_stop_candles", "time_stop_min_profit", "level_source_mode", "s3_level_lookback", "level_band_pct", "level_touch_offset_pct", "level_trigger_timeframe", "level_confirmation_mode", "level_action_mode", "profit_gate_mode",

    )

    ACTIVE_EXIT_PARAMETERS = s3_active_exit_parameters(S3_BRANCH_FAMILY, "exit_path_mode")

    BRANCH_SPLIT_RATIONALE = "Standalone source-family branch for target-zone management; provider/action/confirmation choices are compact categorical modes to avoid a broad Cartesian grid."

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



    S3_LZR_LOOKBACKS = (24, 48, 96, 168, 336)
    S3_LZR_VP_TARGET_COLUMNS = {
        "long": (
            "vp_poc", "vp_vah", "vp_hvn", "vp_lvn", "vp_node", "vp_value_area_high",
            "ltfvp_poc", "ltfvp_vah", "ltfvp_hvn", "ltfvp_lvn", "htfvp_poc", "htfvp_vah",
            "s2vp_poc", "s2vp_vah",
        ),
        "short": (
            "vp_poc", "vp_val", "vp_hvn", "vp_lvn", "vp_node", "vp_value_area_low",
            "ltfvp_poc", "ltfvp_val", "ltfvp_hvn", "ltfvp_lvn", "htfvp_poc", "htfvp_val",
            "s2vp_poc", "s2vp_val",
        ),
    }

    exit_path_mode = tagged_exit_parameter(CategoricalParameter([S3_BRANCH_FAMILY, "level_zone_reversal_control"], default=S3_BRANCH_FAMILY, space="sell", optimize=False, load=True))
    fixed_tp_pct = tagged_exit_parameter(CategoricalParameter([0.030, 0.050], default=0.030, space="sell", optimize=False, load=True))
    fixed_sl_pct = tagged_exit_parameter(CategoricalParameter([0.015, 0.020, 0.030, 0.040, 0.060, 0.080], default=0.030, space="sell", optimize=True, load=True))
    partial_1_profit = tagged_exit_parameter(CategoricalParameter([0.010, 0.020], default=0.010, space="sell", optimize=False, load=True))
    partial_2_profit = tagged_exit_parameter(CategoricalParameter([0.030, 0.050], default=0.030, space="sell", optimize=False, load=True))
    partial_1_fraction = tagged_exit_parameter(CategoricalParameter([0.25, 0.33, 0.50], default=0.33, space="sell", optimize=True, load=True))
    partial_2_fraction = tagged_exit_parameter(CategoricalParameter([0.25, 0.33], default=0.25, space="sell", optimize=False, load=True))
    breakeven_trigger = tagged_exit_parameter(CategoricalParameter([0.010, 0.020], default=0.010, space="sell", optimize=False, load=True))
    breakeven_offset = tagged_exit_parameter(CategoricalParameter([0.000, 0.001, 0.0025, 0.005], default=0.001, space="sell", optimize=True, load=True))
    trailing_activation = tagged_exit_parameter(CategoricalParameter([0.010, 0.020], default=0.010, space="sell", optimize=False, load=True))
    trailing_distance = tagged_exit_parameter(CategoricalParameter([0.005, 0.010, 0.015, 0.025], default=0.010, space="sell", optimize=True, load=True))
    indicator_min_profit = tagged_exit_parameter(CategoricalParameter([0.000, 0.005], default=0.000, space="sell", optimize=False, load=True))
    indicator_max_profit = tagged_exit_parameter(CategoricalParameter([0.120, 0.180], default=0.180, space="sell", optimize=False, load=True))
    indicator_near_pct = tagged_exit_parameter(CategoricalParameter([0.010, 0.020], default=0.010, space="sell", optimize=False, load=True))
    indicator_stop_buffer = tagged_exit_parameter(CategoricalParameter([0.010, 0.020], default=0.010, space="sell", optimize=False, load=True))
    guard_tighten_buffer = tagged_exit_parameter(CategoricalParameter([0.010, 0.020], default=0.010, space="sell", optimize=False, load=True))
    guard_profit_floor = tagged_exit_parameter(CategoricalParameter([0.000, 0.005], default=0.000, space="sell", optimize=False, load=True))
    time_stop_candles = tagged_exit_parameter(CategoricalParameter([48, 96], default=48, space="sell", optimize=False, load=True))
    time_stop_min_profit = tagged_exit_parameter(CategoricalParameter([0.000, 0.005], default=0.000, space="sell", optimize=False, load=True))
    level_source_mode = tagged_exit_parameter(CategoricalParameter(["prior_swing", "vp_level", "indicator_level", "nearest_any"], default="nearest_any", space="sell", optimize=True, load=True))
    s3_level_lookback = tagged_exit_parameter(CategoricalParameter([24, 48, 96, 168, 336], default=96, space="sell", optimize=True, load=True))
    level_band_pct = tagged_exit_parameter(CategoricalParameter([0.005, 0.010, 0.015, 0.020, 0.030], default=0.015, space="sell", optimize=True, load=True))
    level_touch_offset_pct = tagged_exit_parameter(CategoricalParameter([0.000, 0.010, 0.020, 0.030], default=0.010, space="sell", optimize=True, load=True))
    level_trigger_timeframe = tagged_exit_parameter(CategoricalParameter(["trade_tf", "trigger_1h", "trigger_4h", "trigger_1d", "source_context"], default="trade_tf", space="sell", optimize=True, load=True))
    level_confirmation_mode = tagged_exit_parameter(CategoricalParameter(["touch", "one_reversal", "two_reversal", "three_reversal", "two_opposite", "three_opposite", "close_reject"], default="touch", space="sell", optimize=True, load=True))
    level_action_mode = tagged_exit_parameter(CategoricalParameter(["full_exit", "partial_33_then_be", "partial_50_then_be", "tighten_stop", "partial_33_then_trail", "full_after_partial", "target_or_fixed_3pct"], default="partial_33_then_be", space="sell", optimize=True, load=True))
    profit_gate_mode = tagged_exit_parameter(CategoricalParameter(["any_profit", "profit_0_5pct", "profit_1pct", "profit_2pct", "profit_3pct"], default="profit_0_5pct", space="sell", optimize=True, load=True))


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



    def _s3_lzr_add_levels(self, dataframe: DataFrame) -> DataFrame:
        frame = dataframe.copy()
        high = _num(frame, "high")
        low = _num(frame, "low")
        for lookback in self.S3_LZR_LOOKBACKS:
            min_periods = max(2, min(int(lookback), max(4, int(lookback) // 4)))
            frame[f"s3_lzr_prior_high_{lookback}"] = high.shift(1).rolling(int(lookback), min_periods=min_periods).max()
            frame[f"s3_lzr_prior_low_{lookback}"] = low.shift(1).rolling(int(lookback), min_periods=min_periods).min()
        return frame

    def _s3_lzr_profit_gate_met(self, current_profit: float) -> bool:
        mode = str(self.profit_gate_mode.value)
        thresholds = {
            "any_profit": 0.0,
            "profit_0_5pct": 0.005,
            "profit_1pct": 0.010,
            "profit_2pct": 0.020,
            "profit_3pct": 0.030,
        }
        return float(current_profit or 0.0) >= float(thresholds.get(mode, 0.0))

    def _s3_lzr_recent_frame(self, pair: str, current_time, timeframe_mode: str) -> DataFrame | None:
        if not getattr(self, "dp", None):
            return None
        selected_tf = self.timeframe
        if timeframe_mode == "trigger_1h":
            selected_tf = "1h"
        elif timeframe_mode == "trigger_4h":
            selected_tf = "4h"
        elif timeframe_mode == "trigger_1d":
            selected_tf = "1d"
        elif timeframe_mode == "source_context":
            selected_tf = str(globals().get("CONTEXT_TIMEFRAME", self.timeframe) or self.timeframe)
        if selected_tf == self.timeframe:
            frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        else:
            frame = self.dp.get_pair_dataframe(pair=pair, timeframe=selected_tf)
        if frame is None or frame.empty:
            return None
        frame = frame.copy().sort_values("date") if "date" in frame.columns else frame.copy()
        if current_time is not None and "date" in frame.columns:
            dates = pd.to_datetime(frame["date"], utc=True, errors="coerce")
            cutoff = pd.Timestamp(current_time)
            if cutoff.tzinfo is None:
                cutoff = cutoff.tz_localize("UTC")
            else:
                cutoff = cutoff.tz_convert("UTC")
            frame = frame.loc[dates <= cutoff]
        return frame.tail(8) if not frame.empty else None

    def _s3_lzr_add_candidate(self, candidates: list[dict[str, Any]], trade, current_rate: float, level: Any, provider: str, column: str) -> None:
        try:
            value = float(level)
        except (TypeError, ValueError):
            return
        if not np.isfinite(value) or value <= 0.0 or current_rate <= 0.0:
            return
        target_profit = self._s3_directional_profit(trade, value)
        if target_profit is None or target_profit < 0.0:
            return
        max_profit = float(self._s3_param_value("indicator_max_profit", 0.18) or 0.18)
        if target_profit > max_profit:
            return
        near = abs(value - current_rate) / current_rate
        candidates.append({"value": value, "provider": provider, "column": column, "profit": abs(target_profit), "near": near})

    def _s3_lzr_target_candidates(self, pair: str, trade, current_rate: float) -> list[dict[str, Any]]:
        last = self._s3_latest_candle(pair)
        if last is None:
            return []
        side_key = "short" if bool(getattr(trade, "is_short", False)) else "long"
        mode = str(self.level_source_mode.value)
        lookback = int(self.s3_level_lookback.value)
        candidates: list[dict[str, Any]] = []
        if mode in {"prior_swing", "nearest_any"}:
            column = f"s3_lzr_prior_low_{lookback}" if side_key == "short" else f"s3_lzr_prior_high_{lookback}"
            self._s3_lzr_add_candidate(candidates, trade, current_rate, last.get(column), "prior_swing", column)
        if mode in {"vp_level", "nearest_any"}:
            for column in self.S3_LZR_VP_TARGET_COLUMNS.get(side_key, ()): 
                if column in last.index:
                    self._s3_lzr_add_candidate(candidates, trade, current_rate, last.get(column), "vp_level", column)
        if mode in {"indicator_level", "nearest_any"}:
            target = self._s3_indicator_target(pair, trade, current_rate, nearest=True)
            if target is not None:
                self._s3_lzr_add_candidate(candidates, trade, current_rate, target.get("value"), str(target.get("provider") or "indicator_level"), str(target.get("column") or "indicator_target"))
        return candidates

    def _s3_lzr_target(self, pair: str, trade, current_rate: float) -> dict[str, Any] | None:
        candidates = self._s3_lzr_target_candidates(pair, trade, float(current_rate or 0.0))
        if not candidates:
            return None
        return sorted(candidates, key=lambda row: (row["near"], row["profit"], row["provider"]))[0]

    def _s3_lzr_confirmation(self, pair: str, trade, current_time, current_rate: float, target: dict[str, Any]) -> dict[str, Any]:
        frame = self._s3_lzr_recent_frame(pair, current_time, str(self.level_trigger_timeframe.value))
        if frame is None or frame.empty:
            return {"triggered": False, "reason": "missing_trigger_frame"}
        level = float(target["value"])
        zone = float(self.level_band_pct.value)
        offset = float(self.level_touch_offset_pct.value)
        is_short = bool(getattr(trade, "is_short", False))
        high = pd.to_numeric(frame.get("high"), errors="coerce")
        low = pd.to_numeric(frame.get("low"), errors="coerce")
        close = pd.to_numeric(frame.get("close"), errors="coerce")
        open_ = pd.to_numeric(frame.get("open"), errors="coerce")
        if high.empty or low.empty or close.empty or open_.empty:
            return {"triggered": False, "reason": "missing_ohlc"}
        latest_high = float(high.iloc[-1])
        latest_low = float(low.iloc[-1])
        latest_close = float(close.iloc[-1])
        if not all(np.isfinite(value) for value in (level, latest_high, latest_low, latest_close)):
            return {"triggered": False, "reason": "bad_level"}
        if is_short:
            touched = latest_low <= level * (1.0 + offset)
            in_zone = latest_low <= level * (1.0 + zone) and latest_high >= level * (1.0 - zone)
            candle_opposite = close.gt(open_)
            close_opposite = close.diff().gt(0)
            close_reject = touched and latest_close >= level * (1.0 + max(zone * 0.5, 0.001))
        else:
            touched = latest_high >= level * (1.0 - offset)
            in_zone = latest_high >= level * (1.0 - zone) and latest_low <= level * (1.0 + zone)
            candle_opposite = close.lt(open_)
            close_opposite = close.diff().lt(0)
            close_reject = touched and latest_close <= level * (1.0 - max(zone * 0.5, 0.001))
        mode = str(self.level_confirmation_mode.value)
        candle_opposite = candle_opposite.fillna(False)
        close_opposite = close_opposite.fillna(False)
        def last_n(series: Series, n: int) -> bool:
            if len(series) < n:
                return False
            return bool(series.tail(n).all())
        if mode == "touch":
            triggered = bool(touched or in_zone)
        elif mode == "one_reversal":
            triggered = bool((touched or in_zone) and last_n(candle_opposite, 1))
        elif mode == "two_reversal":
            triggered = bool((touched or in_zone) and last_n(candle_opposite, 2))
        elif mode == "three_reversal":
            triggered = bool((touched or in_zone) and last_n(candle_opposite, 3))
        elif mode == "two_opposite":
            triggered = bool((touched or in_zone) and last_n(close_opposite, 2))
        elif mode == "three_opposite":
            triggered = bool((touched or in_zone) and last_n(close_opposite, 3))
        elif mode == "close_reject":
            triggered = bool(close_reject)
        else:
            triggered = False
        return {"triggered": triggered, "reason": mode, "touched": bool(touched), "in_zone": bool(in_zone), "level": level}

    def _s3_lzr_state(self, pair: str, trade, current_time, current_rate: float, current_profit: float) -> dict[str, Any]:
        target = self._s3_lzr_target(pair, trade, current_rate)
        if target is None:
            return {"target": None, "triggered": False, "profit_gate": False, "reason": "no_target"}
        confirmation = self._s3_lzr_confirmation(pair, trade, current_time, current_rate, target)
        profit_gate = self._s3_lzr_profit_gate_met(float(current_profit or 0.0))
        return {"target": target, "triggered": bool(confirmation.get("triggered")), "profit_gate": profit_gate, "reason": confirmation.get("reason"), "confirmation": confirmation}

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
        _ = after_fill, kwargs
        stop_profit = -float(self.fixed_sl_pct.value)
        state = self._s3_lzr_state(pair, trade, current_time, current_rate, current_profit)
        action = str(self.level_action_mode.value)
        if state["triggered"] and state["profit_gate"]:
            if any(token in action for token in ("be", "tighten", "trail")):
                stop_profit = max(stop_profit, float(self.breakeven_offset.value))
            if "tighten" in action:
                stop_profit = max(stop_profit, float(current_profit) - float(self.indicator_stop_buffer.value))
            if "trail" in action:
                stop_profit = max(stop_profit, float(current_profit) - float(self.trailing_distance.value))
        open_rate = float(getattr(trade, "open_rate", 0.0) or current_rate or 0.0)
        current_rate = float(current_rate or 0.0)
        if open_rate <= 0.0 or current_rate <= 0.0:
            return -float(self.fixed_sl_pct.value)
        stop_price = open_rate * (1.0 + stop_profit) if not bool(getattr(trade, "is_short", False)) else open_rate * (1.0 - stop_profit)
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=bool(getattr(trade, "is_short", False)), leverage=float(getattr(trade, "leverage", 1.0) or 1.0))

    def adjust_trade_position(self, trade, current_time, current_rate, current_profit, min_stake, max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, **kwargs):
        _ = min_stake, max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, kwargs
        if bool(getattr(trade, "has_open_orders", False)):
            return None
        action = str(self.level_action_mode.value)
        if "partial" not in action:
            return None
        exits_done = int(getattr(trade, "nr_of_successful_exits", 0) or 0)
        if exits_done > 0:
            return None
        state = self._s3_lzr_state(getattr(trade, "pair", ""), trade, current_time, current_rate, current_profit)
        if not (state["triggered"] and state["profit_gate"]):
            return None
        stake = float(getattr(trade, "stake_amount", 0.0) or 0.0)
        if stake <= 0.0:
            return None
        fraction = 0.50 if "partial_50" in action else float(self.partial_1_fraction.value)
        return -(stake * fraction), f"s3_{self.S3_BRANCH_FAMILY}_{action}"

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        _ = kwargs
        action = str(self.level_action_mode.value)
        state = self._s3_lzr_state(pair, trade, current_time, current_rate, current_profit)
        exits_done = int(getattr(trade, "nr_of_successful_exits", 0) or 0)
        if state["triggered"] and state["profit_gate"]:
            target = state["target"] or {}
            tag = f"s3_{self.S3_BRANCH_FAMILY}_{target.get('provider', 'level')}_{state.get('reason', 'zone')}"
            if action in {"full_exit", "target_or_fixed_3pct"}:
                return tag
            if action == "full_after_partial" and exits_done >= 1:
                return tag
            if action == "tighten_stop" and float(current_profit or 0.0) >= 0.030:
                return tag
        if action == "target_or_fixed_3pct" and float(current_profit or 0.0) >= 0.030:
            return f"s3_{self.S3_BRANCH_FAMILY}_fallback_fixed_3pct"
        if float(current_profit or 0.0) <= -float(self.fixed_sl_pct.value):
            return f"s3_{self.S3_BRANCH_FAMILY}_fixed_stop"
        return None



apply_s3_branch_surface(Sieve3ExitLevelZoneReversalFromCrashRsiExtremeReclaimLong4H)

apply_explicit_hyperopt_surface(Sieve3ExitLevelZoneReversalFromCrashRsiExtremeReclaimLong4H)

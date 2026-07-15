from __future__ import annotations

from freqtrade.exchange import timeframe_to_minutes

NOVEL_IDEA = True

UPDATE_HYPOTHESIS = "Entry-only novel Sieve2 probe: d1 range midpoint reject as higher-timeframe context, with 1h reject execution."



import os

from typing import Any

import numpy as np

import pandas as pd

from pandas import DataFrame, Series

from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, IStrategy, merge_informative_pair, stoploss_from_absolute, BooleanParameter



SIEVE_STAGE = "sieve3"

SOURCE_STRATEGY = "sieve3/sieve2_novel_mtf_d1_midline_reject_1h_reject_short.py:Sieve2NovelMtfD1MidlineReject1hRejectShort"

SOURCE_RESULT_BATCH = "sieve3_exit_branch_generation_20260612"

RESEARCH_PATH = "sieve3_exit_level_zone_reversal"

ENTRY_SOURCE_STAGE = "sieve2_or_sieve3_candidate"

EXIT_HYPOTHESIS = 'Target-zone exit sweep using prior swing, VP, or indicator levels with bands, offsets, reversal confirmation, and selectable trigger timeframe.'

SIDE = "short"

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





def tagged_parameter(param: Any) -> Any:

    setattr(param, "batch_tags", ("family:entries", "mode:sieve2_novel_mtf"))

    return param





CONTEXT_MODE = "midline_reject"

EXECUTION_MODE = "reject"





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

class Sieve3ExitLevelZoneReversalFromNovelMtfD1MidlineReject1HRejectShort(IStrategy):

    timeframe = "1h"

    can_short = True

    startup_candle_count = 260

    minimal_roi = {"0": 100.0}

    stoploss = -0.99

    process_only_new_candles = True

    use_exit_signal = False

    use_custom_stoploss = True

    position_adjustment_enable = True

    max_entry_position_adjustment = 0

    INTERFACE_VERSION = 3



    context_lookback = tagged_parameter(IntParameter(8, 40, default=30, space="buy", optimize=True, load=True))

    h4_lookback = tagged_parameter(IntParameter(6, 36, default=31, space="buy", optimize=True, load=True))

    exec_lookback = tagged_parameter(IntParameter(4, 24, default=18, space="buy", optimize=True, load=True))

    body_ratio_min = tagged_parameter(DecimalParameter(0.10, 0.85, default=0.31, decimals=2, space="buy", optimize=True, load=True))

    range_ratio_min = tagged_parameter(DecimalParameter(0.55, 2.80, default=1.77, decimals=2, space="buy", optimize=True, load=True))

    volume_ratio_min = tagged_parameter(DecimalParameter(0.50, 2.80, default=2.66, decimals=2, space="buy", optimize=True, load=True))

    retest_tolerance = tagged_parameter(DecimalParameter(0.001, 0.030, default=0.001, decimals=3, space="buy", optimize=True, load=True))

    close_follow_min = tagged_parameter(DecimalParameter(0.00, 0.65, default=0.44, decimals=2, space="buy", optimize=True, load=True))

    require_h4_confirm = tagged_parameter(CategoricalParameter([False, True], default=True, space="buy", optimize=True, load=True))

    require_volume_confirm = tagged_parameter(CategoricalParameter([False, True], default=False, space="buy", optimize=True, load=True))



    def informative_pairs(self):

        dp = getattr(self, "dp", None)

        if dp is None:

            return []

        pairs = self.dp.current_whitelist()

        return [(pair, "4h") for pair in pairs] + [(pair, "1d") for pair in pairs]



    def populate_indicators(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:

        dataframe = dataframe.copy()

        dp = getattr(self, "dp", None)

        if dp is not None and metadata and metadata.get("pair"):

            for tf in ("4h", "1d"):

                informative = dp.get_pair_dataframe(pair=metadata["pair"], timeframe=tf)

                if informative is not None and not informative.empty:

                    dataframe = merge_informative_pair(dataframe, self._features(informative.copy(), tf), self.timeframe, tf, ffill=True)

        return self._s3_lzr_add_levels(self._features(dataframe, "1h"))



    def _features(self, frame: DataFrame, tf: str) -> DataFrame:

        close = pd.to_numeric(frame["close"], errors="coerce")

        open_ = pd.to_numeric(frame["open"], errors="coerce")

        high = pd.to_numeric(frame["high"], errors="coerce")

        low = pd.to_numeric(frame["low"], errors="coerce")

        volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)

        look = int(self.context_lookback.value if tf == "1d" else self.h4_lookback.value if tf == "4h" else self.exec_lookback.value)

        rng = (high - low).replace(0, np.nan)

        body = close - open_

        frame[f"body_ratio_{tf}"] = (body.abs() / rng).replace([np.inf, -np.inf], np.nan)

        frame[f"prior_high_{tf}"] = high.shift(1).rolling(look, min_periods=max(2, look // 3)).max()

        frame[f"prior_low_{tf}"] = low.shift(1).rolling(look, min_periods=max(2, look // 3)).min()

        frame[f"range_mid_{tf}"] = (frame[f"prior_high_{tf}"] + frame[f"prior_low_{tf}"]) / 2.0

        frame[f"range_ratio_{tf}"] = rng / rng.rolling(look, min_periods=max(2, look // 3)).mean().replace(0, np.nan)

        frame[f"volume_ratio_{tf}"] = volume / volume.rolling(look, min_periods=max(2, look // 3)).mean().replace(0, np.nan)

        frame[f"hh_{tf}"] = high.gt(high.shift(1).rolling(look, min_periods=max(2, look // 3)).max())

        frame[f"ll_{tf}"] = low.lt(low.shift(1).rolling(look, min_periods=max(2, look // 3)).min())

        frame[f"hl_{tf}"] = low.gt(low.shift(1).rolling(max(2, look // 2), min_periods=2).min())

        frame[f"lh_{tf}"] = high.lt(high.shift(1).rolling(max(2, look // 2), min_periods=2).max())

        return frame



    def _context(self, dataframe: DataFrame) -> Series:

        c1 = self._num(dataframe, "close_1d"); o1 = self._num(dataframe, "open_1d"); h1 = self._num(dataframe, "high_1d"); l1 = self._num(dataframe, "low_1d")

        c4 = self._num(dataframe, "close_4h"); o4 = self._num(dataframe, "open_4h"); h4 = self._num(dataframe, "high_4h"); l4 = self._num(dataframe, "low_4h")

        bull1 = c1.gt(o1) & self._num(dataframe, "body_ratio_1d").ge(float(self.body_ratio_min.value))

        bear1 = c1.lt(o1) & self._num(dataframe, "body_ratio_1d").ge(float(self.body_ratio_min.value))

        bull4 = c4.gt(o4); bear4 = c4.lt(o4)

        vol_ok = self._num(dataframe, "volume_ratio_1d").ge(float(self.volume_ratio_min.value)) | ~self._bool_param(self.require_volume_confirm)

        h4_long_ok = bull4 | ~self._bool_param(self.require_h4_confirm); h4_short_ok = bear4 | ~self._bool_param(self.require_h4_confirm)

        mode = CONTEXT_MODE

        if mode == "prior_high_break": ctx = c1.gt(self._num(dataframe,"prior_high_1d")) & bull1

        elif mode == "failed_low_reclaim": ctx = l1.lt(self._num(dataframe,"prior_low_1d")) & c1.gt(self._num(dataframe,"prior_low_1d")) & bull1

        elif mode == "trend_hhhl": ctx = self._bool(dataframe,"hh_1d") & self._bool(dataframe,"hl_1d") & c1.gt(self._num(dataframe,"range_mid_1d"))

        elif mode == "range_expansion": ctx = bull1 & self._num(dataframe,"range_ratio_1d").ge(float(self.range_ratio_min.value))

        elif mode == "midline_reclaim": ctx = c1.gt(self._num(dataframe,"range_mid_1d")) & c1.shift(1).le(self._num(dataframe,"range_mid_1d"))

        elif mode == "compression_break": ctx = c1.gt(self._num(dataframe,"prior_high_1d")) & self._num(dataframe,"range_ratio_1d").le(float(self.range_ratio_min.value))

        elif mode == "volume_break": ctx = c1.gt(self._num(dataframe,"prior_high_1d")) & self._num(dataframe,"volume_ratio_1d").ge(float(self.volume_ratio_min.value))

        elif mode == "h4_prior_high_break": ctx = c4.gt(self._num(dataframe,"prior_high_4h")) & bull4

        elif mode == "h4_hl_reclaim": ctx = self._bool(dataframe,"hl_4h") & c4.gt(self._num(dataframe,"range_mid_4h"))

        elif mode == "h4_failed_low_reclaim": ctx = l4.lt(self._num(dataframe,"prior_low_4h")) & c4.gt(self._num(dataframe,"prior_low_4h"))

        elif mode == "h4_range_expansion": ctx = bull4 & self._num(dataframe,"range_ratio_4h").ge(float(self.range_ratio_min.value))

        elif mode == "h4_compression_break": ctx = c4.gt(self._num(dataframe,"prior_high_4h")) & self._num(dataframe,"range_ratio_4h").le(float(self.range_ratio_min.value))

        elif mode == "dual_break": ctx = c1.gt(self._num(dataframe,"prior_high_1d")) & c4.gt(self._num(dataframe,"prior_high_4h"))

        elif mode == "support_h4_break": ctx = c1.gt(self._num(dataframe,"range_mid_1d")) & c4.gt(self._num(dataframe,"prior_high_4h"))

        elif mode == "pullback_h4_reclaim": ctx = c1.gt(self._num(dataframe,"range_mid_1d")) & self._bool(dataframe,"hl_4h")

        elif mode == "prior_low_break": ctx = c1.lt(self._num(dataframe,"prior_low_1d")) & bear1

        elif mode == "failed_high_reject": ctx = h1.gt(self._num(dataframe,"prior_high_1d")) & c1.lt(self._num(dataframe,"prior_high_1d")) & bear1

        elif mode == "trend_lhll": ctx = self._bool(dataframe,"ll_1d") & self._bool(dataframe,"lh_1d") & c1.lt(self._num(dataframe,"range_mid_1d"))

        elif mode == "bear_range_expansion": ctx = bear1 & self._num(dataframe,"range_ratio_1d").ge(float(self.range_ratio_min.value))

        elif mode == "midline_reject": ctx = c1.lt(self._num(dataframe,"range_mid_1d")) & c1.shift(1).ge(self._num(dataframe,"range_mid_1d"))

        elif mode == "compression_breakdown": ctx = c1.lt(self._num(dataframe,"prior_low_1d")) & self._num(dataframe,"range_ratio_1d").le(float(self.range_ratio_min.value))

        elif mode == "volume_breakdown": ctx = c1.lt(self._num(dataframe,"prior_low_1d")) & self._num(dataframe,"volume_ratio_1d").ge(float(self.volume_ratio_min.value))

        elif mode == "h4_prior_low_break": ctx = c4.lt(self._num(dataframe,"prior_low_4h")) & bear4

        elif mode == "h4_lh_reject": ctx = self._bool(dataframe,"lh_4h") & c4.lt(self._num(dataframe,"range_mid_4h"))

        elif mode == "h4_failed_high_reject": ctx = h4.gt(self._num(dataframe,"prior_high_4h")) & c4.lt(self._num(dataframe,"prior_high_4h"))

        elif mode == "h4_bear_range_expansion": ctx = bear4 & self._num(dataframe,"range_ratio_4h").ge(float(self.range_ratio_min.value))

        elif mode == "h4_compression_breakdown": ctx = c4.lt(self._num(dataframe,"prior_low_4h")) & self._num(dataframe,"range_ratio_4h").le(float(self.range_ratio_min.value))

        elif mode == "dual_breakdown": ctx = c1.lt(self._num(dataframe,"prior_low_1d")) & c4.lt(self._num(dataframe,"prior_low_4h"))

        elif mode == "resistance_h4_break": ctx = c1.lt(self._num(dataframe,"range_mid_1d")) & c4.lt(self._num(dataframe,"prior_low_4h"))

        elif mode == "bounce_h4_reject": ctx = c1.lt(self._num(dataframe,"range_mid_1d")) & self._bool(dataframe,"lh_4h")

        else: ctx = pd.Series(False, index=dataframe.index)

        return (ctx & (h4_long_ok if SIDE == "long" else h4_short_ok) & vol_ok).fillna(False)



    def _execution(self, dataframe: DataFrame) -> Series:

        close = self._num(dataframe,"close"); open_ = self._num(dataframe,"open"); high = self._num(dataframe,"high"); low = self._num(dataframe,"low")

        ph = self._num(dataframe,"prior_high_1h"); pl = self._num(dataframe,"prior_low_1h"); mid = self._num(dataframe,"range_mid_1h")

        tol = float(self.retest_tolerance.value); follow = float(self.close_follow_min.value)

        bull = close.gt(open_) & self._num(dataframe,"body_ratio_1h").ge(follow)

        bear = close.lt(open_) & self._num(dataframe,"body_ratio_1h").ge(follow)

        if EXECUTION_MODE == "breakout": exe = close.gt(ph) & bull

        elif EXECUTION_MODE == "breakdown": exe = close.lt(pl) & bear

        elif EXECUTION_MODE == "retest" and SIDE == "long": exe = low.le(ph*(1+tol)) & close.gt(ph) & bull

        elif EXECUTION_MODE == "retest" and SIDE == "short": exe = high.ge(pl*(1-tol)) & close.lt(pl) & bear

        elif EXECUTION_MODE == "bos" and SIDE == "long": exe = close.gt(ph) & self._bool(dataframe,"hl_1h")

        elif EXECUTION_MODE == "bos" and SIDE == "short": exe = close.lt(pl) & self._bool(dataframe,"lh_1h")

        elif EXECUTION_MODE == "higher_low_break": exe = self._bool(dataframe,"hl_1h") & close.gt(mid) & bull

        elif EXECUTION_MODE == "lower_high_break": exe = self._bool(dataframe,"lh_1h") & close.lt(mid) & bear

        elif EXECUTION_MODE == "reclaim": exe = low.lt(mid) & close.gt(mid) & bull

        elif EXECUTION_MODE == "reject": exe = high.gt(mid) & close.lt(mid) & bear

        else: exe = pd.Series(False, index=dataframe.index)

        return exe.fillna(False)



    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:

        condition = self._context(dataframe) & self._execution(dataframe)

        if SIDE == "long": dataframe.loc[condition, ["enter_long", "enter_tag"]] = (1, f"{CONTEXT_MODE}_1h_{EXECUTION_MODE}_long")

        else: dataframe.loc[condition, ["enter_short", "enter_tag"]] = (1, f"{CONTEXT_MODE}_1h_{EXECUTION_MODE}_short")

        return dataframe



    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:

        return dataframe



    @staticmethod

    def _column_alias(frame: DataFrame, column: str) -> str | None:

        if column in frame.columns:

            return column

        for suffix in ("_1d", "_4h"):

            if column.endswith(suffix):

                merged = f"{column}{suffix}"

                if merged in frame.columns:

                    return merged

        return None



    @staticmethod

    def _num(frame: DataFrame, column: str, default: float = 0.0) -> Series:

        resolved = Sieve2NovelMtfBaseMixin._column_alias(frame, column) if "Sieve2NovelMtfBaseMixin" in globals() else None

        if resolved is None:

            if column in frame.columns:

                resolved = column

            else:

                for suffix in ("_1d", "_4h"):

                    if column.endswith(suffix) and f"{column}{suffix}" in frame.columns:

                        resolved = f"{column}{suffix}"

                        break

        if resolved is None:

            return pd.Series(default, index=frame.index, dtype="float64")

        return pd.to_numeric(frame[resolved], errors="coerce").replace([np.inf, -np.inf], np.nan)



    @staticmethod

    def _bool(frame: DataFrame, column: str) -> Series:

        resolved = column if column in frame.columns else None

        if resolved is None:

            for suffix in ("_1d", "_4h"):

                if column.endswith(suffix) and f"{column}{suffix}" in frame.columns:

                    resolved = f"{column}{suffix}"

                    break

        if resolved is None:

            return pd.Series(False, index=frame.index, dtype="bool")

        return pd.Series(frame[resolved], index=frame.index).astype("boolean").fillna(False).astype(bool)



    @staticmethod

    def _bool_param(param: Any) -> bool:

        value = getattr(param, "value", param)

        if isinstance(value, str): return value.lower() in {"1", "true", "yes", "on"}

        return bool(value)



    S3_BRANCH_FAMILY = "level_zone_reversal"

    EXIT_FAMILY = "level_zone_reversal"

    EXIT_HYPOTHESIS = 'Target-zone exit sweep using prior swing, VP, or indicator levels with bands, offsets, reversal confirmation, and selectable trigger timeframe.'

    SOURCE_ENTRY_STEM = "novel_mtf_d1_midline_reject_1h_reject_short"

    ENTRY_TAG = "novel_mtf_d1_midline_reject_1h_reject_short"

    SOURCE_ENTRY_CLASS = "Sieve2NovelMtfD1MidlineReject1hRejectShort"

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
        high = self._num(frame, "high")
        low = self._num(frame, "low")
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



apply_s3_branch_surface(Sieve3ExitLevelZoneReversalFromNovelMtfD1MidlineReject1HRejectShort)

apply_explicit_hyperopt_surface(Sieve3ExitLevelZoneReversalFromNovelMtfD1MidlineReject1HRejectShort)

# SIEVE3_PROMOTION_NOTE: Accepted from 20260606T100454 batch03: 862 trades, 53.71% WR, +141.891 USDT, 6.69% DD at TP/SL 3/3. Strongest batch03 result; H4 compression break with 1h retest entry is logically useful.
from __future__ import annotations
EXIT_HYPOTHESIS = "Broad Sieve3 exit sweep added in place to existing distinct clone."
EXIT_RESEARCH_PATH = "broad_exit_sweep"
ENTRY_SOURCE_STAGE = "sieve3_candidate"
from freqtrade.exchange import timeframe_to_minutes
SIEVE_STAGE = "sieve3"
NOVEL_IDEA = True
SOURCE_STRATEGY = "C:/FreqTradeStuff/user_data/strategies/sieve3_candidates/from_20260606_novel_mtf/sieve3_novel_mtf_h4_compression_break_1h_retest_long.py:Sieve3NovelMtfH4CompressionBreak1hRetestLong"
SOURCE_RESULT_BATCH = "sieve3_exit_in_place_20260612"
RESEARCH_PATH = "sieve3_exit_in_place_broad_sweep"
UPDATE_HYPOTHESIS = "Entry-only novel Sieve3 probe: 4h compression break as higher-timeframe context, with 1h retest execution."

import os
from typing import Any
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, IStrategy, merge_informative_pair
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, stoploss_from_absolute

SIDE = "long"
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
    setattr(param, "batch_tags", ("family:entries", "mode:sieve3_novel_mtf"))
    return param


CONTEXT_MODE = "h4_compression_break"
EXECUTION_MODE = "retest"


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

class Sieve3ExitNovelMtfH4CompressionBreak1hRetestLong(IStrategy):
    trailing_stop = False
    position_adjustment_enable = True
    use_custom_stoploss = True
    timeframe = "1h"
    can_short = True
    startup_candle_count = 260
    minimal_roi = {"0": 100.0}
    stoploss = -0.99
    process_only_new_candles = True
    use_exit_signal = False
    INTERFACE_VERSION = 3

    context_lookback = tagged_parameter(IntParameter(8, 40, default=12, space="buy", optimize=True, load=True))
    h4_lookback = tagged_parameter(IntParameter(6, 36, default=23, space="buy", optimize=True, load=True))
    exec_lookback = tagged_parameter(IntParameter(4, 24, default=21, space="buy", optimize=True, load=True))
    body_ratio_min = tagged_parameter(DecimalParameter(0.10, 0.85, default=0.32, decimals=2, space="buy", optimize=True, load=True))
    range_ratio_min = tagged_parameter(DecimalParameter(0.55, 2.80, default=1.96, decimals=2, space="buy", optimize=True, load=True))
    volume_ratio_min = tagged_parameter(DecimalParameter(0.50, 2.80, default=1.71, decimals=2, space="buy", optimize=True, load=True))
    retest_tolerance = tagged_parameter(DecimalParameter(0.001, 0.030, default=0.002, decimals=3, space="buy", optimize=True, load=True))
    close_follow_min = tagged_parameter(DecimalParameter(0.00, 0.65, default=0.29, decimals=2, space="buy", optimize=True, load=True))
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
        return self._features(dataframe, "1h")

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
        resolved = Sieve3NovelMtfBaseMixin._column_alias(frame, column) if "Sieve3NovelMtfBaseMixin" in globals() else None
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
    EXIT_FAMILY = "broad_exit_sweep"
    EXIT_HYPOTHESIS = "Broad Sieve3 exit sweep: fixed exits, partial exits, breakeven shifts, trailing, structural targets, and time stops against the unchanged entry clone."
    SOURCE_ENTRY_STEM = "novel_mtf_h4_compression_break_1h_retest_long"
    ENTRY_TAG = "novel_mtf_h4_compression_break_1h_retest_long"
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

apply_s3_branch_surface(Sieve3ExitNovelMtfH4CompressionBreak1hRetestLong)
apply_explicit_hyperopt_surface(Sieve3ExitNovelMtfH4CompressionBreak1hRetestLong)
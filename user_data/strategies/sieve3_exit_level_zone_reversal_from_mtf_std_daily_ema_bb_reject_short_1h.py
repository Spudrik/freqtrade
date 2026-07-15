from __future__ import annotations

import os

from freqtrade.exchange import timeframe_to_minutes



NOVEL_IDEA = True

UPDATE_HYPOTHESIS = "Daily downtrend with 4h compression and 1h EMA rejection may catch cleaner short continuation entries."



import numpy as np

import pandas as pd

from pandas import DataFrame, Series

from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, IStrategy, merge_informative_pair, stoploss_from_absolute, BooleanParameter



SIDE = "short"

ENTRY_MODE = "daily_ema_bb_reject_short"

ENTRY_TAG = "mtf_std_daily_ema_bb_reject_short_1h"

SIEVE_STAGE = "sieve3"

SOURCE_STRATEGY = "sieve3/sieve2_mtf_std_daily_ema_bb_reject_short_1h.py:Sieve2MtfStdDailyEmaBbRejectShort1h"

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





def _rsi(close: Series, length: int = 14) -> Series:

    delta = close.diff()

    gain = delta.clip(lower=0.0).ewm(alpha=1 / float(length), adjust=False, min_periods=max(2, length // 2)).mean()

    loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / float(length), adjust=False, min_periods=max(2, length // 2)).mean()

    rs = gain / loss.replace(0.0, np.nan)

    return 100.0 - (100.0 / (1.0 + rs))





def _ema(close: Series, length: int) -> Series:

    return close.ewm(span=length, adjust=False, min_periods=max(2, length // 2)).mean()





def _macd_hist(close: Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Series:

    macd = _ema(close, fast) - _ema(close, slow)

    sig = macd.ewm(span=signal, adjust=False, min_periods=max(2, signal // 2)).mean()

    return macd - sig





def _bb_width(close: Series, length: int, mult: float = 2.0) -> Series:

    mid = close.rolling(length, min_periods=max(2, length // 2)).mean()

    std = close.rolling(length, min_periods=max(2, length // 2)).std()

    upper = mid + mult * std

    lower = mid - mult * std

    return _safe_div(upper - lower, mid)







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


def _num(frame: DataFrame, column: str, default: float | Series = 0.0) -> Series:
    if column not in frame.columns:
        if isinstance(default, Series):
            return pd.to_numeric(default, errors="coerce")
        return pd.Series(float(default), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


class Sieve3ExitLevelZoneReversalFromMtfStdDailyEmaBbRejectShort1H(IStrategy):

    timeframe = "1h"

    can_short = True

    startup_candle_count = 400

    minimal_roi = {"0": 100.0}

    stoploss = -0.99

    process_only_new_candles = True

    use_exit_signal = False

    use_custom_stoploss = True

    position_adjustment_enable = True

    max_entry_position_adjustment = 0

    INTERFACE_VERSION = 3



    ema_fast_len = tagged_parameter(IntParameter(8, 48, default=28, space="buy", optimize=True, load=True))

    ema_slow_len = tagged_parameter(IntParameter(20, 240, default=98, space="buy", optimize=True, load=True))

    bb_len = tagged_parameter(IntParameter(10, 60, default=10, space="buy", optimize=True, load=True))

    bb_width_max = tagged_parameter(DecimalParameter(0.02, 0.35, default=0.1, decimals=2, space="buy", optimize=True, load=True))

    volume_ratio_min = tagged_parameter(DecimalParameter(0.70, 3.50, default=3.32, decimals=2, space="buy", optimize=True, load=True))

    retest_buffer_pct = tagged_parameter(DecimalParameter(0.001, 0.03, default=0.005, decimals=3, space="buy", optimize=True, load=True))

    rsi_len = tagged_parameter(IntParameter(7, 28, default=10, space="buy", optimize=True, load=True))

    rsi_long_min = tagged_parameter(IntParameter(40, 70, default=63, space="buy", optimize=True, load=True))

    rsi_short_max = tagged_parameter(IntParameter(30, 60, default=47, space="buy", optimize=True, load=True))

    use_daily_trend = tagged_parameter(CategoricalParameter([False, True], default=False, space="buy", optimize=True, load=True))

    use_h4_compression = tagged_parameter(CategoricalParameter([False, True], default=False, space="buy", optimize=True, load=True))

    use_volume_filter = tagged_parameter(CategoricalParameter([False, True], default=True, space="buy", optimize=True, load=True))

    use_retest = tagged_parameter(CategoricalParameter([False, True], default=True, space="buy", optimize=True, load=True))

    use_momentum_filter = tagged_parameter(CategoricalParameter([False, True], default=False, space="buy", optimize=True, load=True))



    def informative_pairs(self):

        dp = getattr(self, "dp", None)

        if dp is None:

            return []

        pairs = self.dp.current_whitelist()

        return [(pair, "4h") for pair in pairs] + [(pair, "1d") for pair in pairs]



    def _features(self, frame: DataFrame, tf: str) -> DataFrame:

        close = pd.to_numeric(frame["close"], errors="coerce")

        high = pd.to_numeric(frame["high"], errors="coerce")

        low = pd.to_numeric(frame["low"], errors="coerce")

        volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)

        ema_fast = int(self.ema_fast_len.value)

        ema_slow = int(self.ema_slow_len.value)

        bb_len = int(self.bb_len.value)

        rsi_len = int(self.rsi_len.value)

        frame[f"ema_fast_{tf}"] = _ema(close, ema_fast)

        frame[f"ema_slow_{tf}"] = _ema(close, ema_slow)

        frame[f"rsi_{tf}"] = _rsi(close, rsi_len)

        frame[f"bb_width_{tf}"] = _bb_width(close, bb_len, 2.0).fillna(0.0)

        frame[f"macd_hist_{tf}"] = _macd_hist(close).fillna(0.0)

        frame[f"volume_ratio_{tf}"] = _safe_div(volume, volume.rolling(20, min_periods=5).mean()).fillna(0.0)

        frame[f"atr_ratio_{tf}"] = _safe_div((high - low).rolling(14, min_periods=5).mean(), close).fillna(0.0)

        frame[f"prior_high_{tf}"] = high.shift(1).rolling(20, min_periods=5).max()

        frame[f"prior_low_{tf}"] = low.shift(1).rolling(20, min_periods=5).min()

        return frame



    def populate_indicators(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:

        frame = dataframe.copy()

        dp = getattr(self, "dp", None)

        if dp is not None and metadata and metadata.get("pair"):

            for tf in ("4h", "1d"):

                informative = dp.get_pair_dataframe(pair=metadata["pair"], timeframe=tf)

                if informative is not None and not informative.empty:

                    frame = merge_informative_pair(frame, self._features(informative.copy(), tf), self.timeframe, tf, ffill=True)

                    frame = frame.rename(columns=lambda col, tf=tf: col.replace(f"_{tf}_{tf}", f"_{tf}"))

        return self._s3_lzr_add_levels(self._features(frame, "1h"))



    def _daily_trend_ok(self, frame: DataFrame, bullish: bool) -> Series:

        fast = pd.to_numeric(frame["ema_fast_1d"], errors="coerce")

        slow = pd.to_numeric(frame["ema_slow_1d"], errors="coerce")

        if not bool(self.use_daily_trend.value):

            return pd.Series(True, index=frame.index)

        return fast.gt(slow) if bullish else fast.lt(slow)



    def _h4_compression_ok(self, frame: DataFrame) -> Series:

        if not bool(self.use_h4_compression.value):

            return pd.Series(True, index=frame.index)

        return pd.to_numeric(frame["bb_width_4h"], errors="coerce").le(float(self.bb_width_max.value))



    def _volume_ok(self, frame: DataFrame) -> Series:

        if not bool(self.use_volume_filter.value):

            return pd.Series(True, index=frame.index)

        return pd.to_numeric(frame["volume_ratio_1h"], errors="coerce").ge(float(self.volume_ratio_min.value))



    def _momentum_ok(self, frame: DataFrame, bullish: bool) -> Series:

        if not bool(self.use_momentum_filter.value):

            return pd.Series(True, index=frame.index)

        if bullish:

            return pd.to_numeric(frame["rsi_1d"], errors="coerce").ge(float(self.rsi_long_min.value)) | pd.to_numeric(frame["macd_hist_1d"], errors="coerce").gt(0.0)

        return pd.to_numeric(frame["rsi_1d"], errors="coerce").le(float(self.rsi_short_max.value)) | pd.to_numeric(frame["macd_hist_1d"], errors="coerce").lt(0.0)



    def _entry_condition(self, frame: DataFrame) -> Series:

        close = pd.to_numeric(frame["close"], errors="coerce")

        high = pd.to_numeric(frame["high"], errors="coerce")

        low = pd.to_numeric(frame["low"], errors="coerce")

        ema_fast_1h = pd.to_numeric(frame["ema_fast_1h"], errors="coerce")

        ema_slow_1h = pd.to_numeric(frame["ema_slow_1h"], errors="coerce")

        ema_fast_4h = pd.to_numeric(frame["ema_fast_4h"], errors="coerce")

        ema_slow_4h = pd.to_numeric(frame["ema_slow_4h"], errors="coerce")

        prior_high_1d = pd.to_numeric(frame["prior_high_1d"], errors="coerce")

        prior_low_1d = pd.to_numeric(frame["prior_low_1d"], errors="coerce")

        rsi_1d = pd.to_numeric(frame["rsi_1d"], errors="coerce")

        rsi_1h = pd.to_numeric(frame["rsi_1h"], errors="coerce")

        rsi_4h = pd.to_numeric(frame["rsi_4h"], errors="coerce")

        volume_ok = self._volume_ok(frame)

        h4_ok = self._h4_compression_ok(frame)

        long_trend_ok = self._daily_trend_ok(frame, True)

        short_trend_ok = self._daily_trend_ok(frame, False)

        bull_momentum = self._momentum_ok(frame, True)

        bear_momentum = self._momentum_ok(frame, False)

        retest = float(self.retest_buffer_pct.value)

        breakout_1h = close.gt(high.shift(1).rolling(20, min_periods=5).max())

        breakdown_1h = close.lt(low.shift(1).rolling(20, min_periods=5).min())

        touch_fast = low.lt(ema_fast_1h * (1.0 + retest))

        reject_fast = high.gt(ema_fast_1h * (1.0 - retest))

        reclaim_fast = close.gt(ema_fast_1h)

        reject_fast_short = close.lt(ema_fast_1h)



        if ENTRY_MODE == "daily_ema_bb_volume_breakout":

            return long_trend_ok & h4_ok & volume_ok & close.gt(ema_fast_1h)

        if ENTRY_MODE == "daily_ema_bb_retest":

            return long_trend_ok & h4_ok & volume_ok & bool(self.use_retest.value) & touch_fast & reclaim_fast

        if ENTRY_MODE == "daily_macd_volume_breakout":

            return long_trend_ok & h4_ok & volume_ok & pd.to_numeric(frame["macd_hist_1d"], errors="coerce").gt(0.0) & breakout_1h

        if ENTRY_MODE == "daily_rsi_pullback_reclaim":

            return long_trend_ok & volume_ok & rsi_1d.ge(float(self.rsi_long_min.value)) & rsi_4h.ge(45.0) & touch_fast & reclaim_fast

        if ENTRY_MODE == "daily_prior_high_breakout":

            return long_trend_ok & h4_ok & volume_ok & close.gt(prior_high_1d * (1.0 + retest))

        if ENTRY_MODE == "daily_h4_trend_pullback_reclaim":

            return long_trend_ok & pd.to_numeric(frame["ema_fast_4h"], errors="coerce").gt(ema_slow_4h) & volume_ok & touch_fast & reclaim_fast

        if ENTRY_MODE == "daily_ema_bb_volume_breakdown":

            return short_trend_ok & h4_ok & volume_ok & close.lt(ema_fast_1h)

        if ENTRY_MODE == "daily_ema_bb_reject_short":

            return short_trend_ok & h4_ok & volume_ok & bool(self.use_retest.value) & reject_fast & reject_fast_short

        if ENTRY_MODE == "daily_macd_volume_breakdown":

            return short_trend_ok & h4_ok & volume_ok & pd.to_numeric(frame["macd_hist_1d"], errors="coerce").lt(0.0) & breakdown_1h

        if ENTRY_MODE == "daily_rsi_pullback_reject_short":

            return short_trend_ok & volume_ok & rsi_1d.le(float(self.rsi_short_max.value)) & rsi_4h.le(55.0) & reject_fast & reject_fast_short

        if ENTRY_MODE == "daily_prior_low_breakdown_short":

            return short_trend_ok & h4_ok & volume_ok & close.lt(prior_low_1d * (1.0 - retest))

        if ENTRY_MODE == "daily_h4_trend_pullback_reject_short":

            return short_trend_ok & pd.to_numeric(frame["ema_fast_4h"], errors="coerce").lt(ema_slow_4h) & volume_ok & reject_fast & reject_fast_short

        return pd.Series(False, index=frame.index)



    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:

        condition = self._entry_condition(dataframe)

        if SIDE == "long":

            dataframe.loc[condition, ["enter_long", "enter_tag"]] = (1, ENTRY_TAG)

        else:

            dataframe.loc[condition, ["enter_short", "enter_tag"]] = (1, ENTRY_TAG)

        return dataframe



    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:

        return dataframe



    S3_BRANCH_FAMILY = "level_zone_reversal"

    EXIT_FAMILY = "level_zone_reversal"

    EXIT_HYPOTHESIS = 'Target-zone exit sweep using prior swing, VP, or indicator levels with bands, offsets, reversal confirmation, and selectable trigger timeframe.'

    SOURCE_ENTRY_STEM = "mtf_std_daily_ema_bb_reject_short_1h"

    SOURCE_ENTRY_CLASS = "Sieve2MtfStdDailyEmaBbRejectShort1h"

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



apply_s3_branch_surface(Sieve3ExitLevelZoneReversalFromMtfStdDailyEmaBbRejectShort1H)

apply_explicit_hyperopt_surface(Sieve3ExitLevelZoneReversalFromMtfStdDailyEmaBbRejectShort1H)

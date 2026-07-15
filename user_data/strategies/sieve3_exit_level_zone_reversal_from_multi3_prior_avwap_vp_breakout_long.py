from __future__ import annotations



# Parked as Sieve3 candidate: cleaner Sieve2 performer for later undefined refinement.



import os

from datetime import datetime

from typing import Any



import numpy as np

import pandas as pd

from pandas import DataFrame, Series



from freqtrade.exchange import timeframe_to_minutes

from freqtrade.strategy import BooleanParameter, CategoricalParameter, DecimalParameter, IntParameter, IStrategy, stoploss_from_absolute



from user_data.Indicators.complex_volume_profile import add_volume_profile





from user_data.strategies.sieve_guard_helpers import (

    SIEVE2_MARKET_GUARD_MODES,

    SIEVE2_VP_GUARD_MODES,

    add_sieve2_guard_indicators,

    apply_sieve2_optional_guards,

)



HYPEROPT_PARAM_ENV = "HYBRID_RECOVERY_HYPEROPT_PARAMS"

ENTRY_SIEVE_TAKE_PROFIT_ENV = "ENTRY_SIEVE_TAKE_PROFIT_PCT"

ENTRY_SIEVE_STOPLOSS_ENV = "ENTRY_SIEVE_STOPLOSS_PCT"

ENTRY_MODE = "entry_multi3_prior_avwap_vp_breakout_long"

ENTRY_TAG = "multi3_prior_avwap_vp_breakout_long"

SIEVE_STAGE = "sieve3"

SOURCE_STRATEGY = "sieve3/sieve3_exit_multi3_prior_avwap_vp_breakout_long.py:Sieve3ExitMulti3PriorAvwapVpBreakoutLong"

SOURCE_RESULT_BATCH = "sieve3_exit_branch_generation_20260612"

RESEARCH_PATH = "sieve3_exit_level_zone_reversal"

ENTRY_SOURCE_STAGE = "sieve2_or_sieve3_candidate"

EXIT_HYPOTHESIS = 'Target-zone exit sweep using prior swing, VP, or indicator levels with bands, offsets, reversal confirmation, and selectable trigger timeframe.'

SIDE = "long"

CONCEPT = "multi3_prior_avwap_vp_breakout"

PERIOD_KIND = "select"

GUARD_MODE_CHOICES = ["direction", "score", "context", "score_or_context", "balance"]

PERIOD_CHOICES = ["day", "week", "month"]

PRICE_SOURCE_CHOICES = ["close", "hl2", "hlc3", "ohlc4"]

VP_COLUMNS = [

    "score_long",

    "score_short",

    "context_score_bull",

    "context_score_bear",

    "context_score_balance",

    "market_context",

]





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

class Sieve3ExitLevelZoneReversalFromMulti3PriorAvwapVpBreakoutLong(IStrategy):

    """

    Sieve2 entry concept: multi3_prior_avwap_vp_breakout_long.



    This is entry-quality research only. It has no custom exit, no DCA, and no

    position-management hooks. Optional guards are hyperoptable inside this

    standalone strategy so Explorer can test whether they improve the trigger.

    """



    INTERFACE_VERSION = 3



    timeframe = "1h"

    startup_candle_count = 336

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



    breakout_buffer_pct = tagged_parameter(CategoricalParameter([0.0, 0.003, 0.004, 0.006, 0.012], default=0.003, space="buy", optimize=True, load=True))

    reclaim_buffer_pct = tagged_parameter(CategoricalParameter([0.0, 0.006, 0.012, 0.02], default=0.0, space="buy", optimize=True, load=True))

    sweep_buffer_pct = tagged_parameter(DecimalParameter(0.000, 0.040, decimals=3, default=0.006, space="buy", optimize=False, load=True))

    zone_near_pct = tagged_parameter(DecimalParameter(0.000, 0.035, decimals=3, default=0.010, space="buy", optimize=False, load=True))

    rolling_level_lookback = tagged_parameter(IntParameter(12, 240, default=72, space="buy", optimize=False, load=True))

    equal_level_lookback = tagged_parameter(IntParameter(12, 240, default=72, space="buy", optimize=False, load=True))

    equal_level_tolerance_pct = tagged_parameter(DecimalParameter(0.001, 0.025, decimals=3, default=0.006, space="buy", optimize=False, load=True))

    equal_level_min_touches = tagged_parameter(IntParameter(2, 6, default=2, space="buy", optimize=False, load=True))

    multi_period = tagged_parameter(CategoricalParameter(['day', 'week', 'month'], default='month', space="buy", optimize=True, load=True))



    avwap_anchor_lookback = tagged_parameter(CategoricalParameter([72, 120, 240], default=240, space="buy", optimize=True, load=True))

    avwap_band_mult = tagged_parameter(DecimalParameter(0.25, 3.00, decimals=2, default=1.25, space="buy", optimize=False, load=True))



    zone_impulse_window = tagged_parameter(IntParameter(12, 96, default=36, space="buy", optimize=False, load=True))

    zone_impulse_atr_min = tagged_parameter(DecimalParameter(0.20, 3.00, decimals=2, default=0.80, space="buy", optimize=False, load=True))

    zone_body_fraction_min = tagged_parameter(DecimalParameter(0.30, 0.90, decimals=2, default=0.55, space="buy", optimize=False, load=True))

    zone_volume_ratio_min = tagged_parameter(DecimalParameter(0.00, 3.00, decimals=2, default=1.10, space="buy", optimize=False, load=True))

    zone_max_age_bars = tagged_parameter(IntParameter(6, 240, default=72, space="buy", optimize=False, load=True))



    use_volume_guard = tagged_parameter(BooleanParameter(default=True, space="buy", optimize=True, load=True))

    volume_window = tagged_parameter(CategoricalParameter([12, 24, 48], default=24, space="buy", optimize=False, load=True))

    volume_ratio_min = tagged_parameter(CategoricalParameter([0.8, 1.0, 1.3, 1.6], default=0.8, space="buy", optimize=True, load=True))

    pressure_min = tagged_parameter(CategoricalParameter([0.05, 0.1, 0.2, 0.35], default=0.05, space="buy", optimize=False, load=True))



    use_close_direction_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=False, load=True))



    vp_window = tagged_parameter(IntParameter(24, 168, default=96, space="buy", optimize=False, load=True))

    vp_bins = tagged_parameter(IntParameter(24, 72, default=48, space="buy", optimize=False, load=True))

    vp_value_area_pct = tagged_parameter(DecimalParameter(0.55, 0.85, decimals=2, default=0.70, space="buy", optimize=False, load=True))

    vp_price_source = tagged_parameter(CategoricalParameter(PRICE_SOURCE_CHOICES, default="hlc3", space="buy", optimize=False, load=True))

    vp_smooth_bins = tagged_parameter(IntParameter(1, 6, default=3, space="buy", optimize=False, load=True))

    vp_hvn_threshold = tagged_parameter(DecimalParameter(0.50, 0.90, decimals=2, default=0.70, space="buy", optimize=False, load=True))

    vp_lvn_threshold = tagged_parameter(DecimalParameter(0.10, 0.55, decimals=2, default=0.35, space="buy", optimize=False, load=True))

    vp_pressure_delta_min = tagged_parameter(DecimalParameter(0.00, 0.35, decimals=2, default=0.05, space="buy", optimize=False, load=True))

    vp_node_near_pct = tagged_parameter(DecimalParameter(0.002, 0.030, decimals=3, default=0.010, space="buy", optimize=False, load=True))

    vp_volume_percentile_min = tagged_parameter(DecimalParameter(0.00, 0.90, decimals=2, default=0.55, space="buy", optimize=False, load=True))

    vp_score_window = tagged_parameter(IntParameter(12, 120, default=48, space="buy", optimize=False, load=True))

    vp_fast_traverse_atr_mult = tagged_parameter(DecimalParameter(0.50, 2.80, decimals=2, default=1.20, space="buy", optimize=False, load=True))

    vp_entry_score_margin = tagged_parameter(DecimalParameter(0.00, 0.20, decimals=2, default=0.02, space="buy", optimize=False, load=True))



    use_vp_1h_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=False, load=True))

    vp_guard_mode = tagged_parameter(CategoricalParameter(['score', 'context', 'score_or_context'], default='context', space="buy", optimize=True, load=True))

    vp_score_min = tagged_parameter(CategoricalParameter([0.15, 0.25, 0.4, 0.6], default=0.25, space="buy", optimize=True, load=True))

    vp_context_min = tagged_parameter(CategoricalParameter([0.2, 0.28, 0.45, 0.6], default=0.6, space="buy", optimize=True, load=True))



    use_vp_4h_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=False, load=True))

    vp_4h_window = tagged_parameter(IntParameter(12, 96, default=48, space="buy", optimize=False, load=True))

    vp_4h_bins = tagged_parameter(IntParameter(16, 64, default=36, space="buy", optimize=False, load=True))



    use_vp_1d_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=False, load=True))

    vp_1d_window = tagged_parameter(IntParameter(10, 84, default=30, space="buy", optimize=False, load=True))

    vp_1d_bins = tagged_parameter(IntParameter(16, 64, default=36, space="buy", optimize=False, load=True))



    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:

        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs

        return 1.0



    def informative_pairs(self) -> list[tuple[str, str]]:

        if not getattr(self, "dp", None):

            return []

        pairs = self.dp.current_whitelist()

        return [(pair, "4h") for pair in pairs] + [(pair, "1d") for pair in pairs]



    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe = self._add_volume_pressure(dataframe)

        dataframe = self._add_prior_period_levels(dataframe)

        dataframe = self._add_rolling_levels(dataframe)

        dataframe = self._add_avwap(dataframe)

        dataframe = self._add_supply_demand(dataframe)

        dataframe = self._add_liquidity_levels(dataframe)

        dataframe = self._add_volume_profile(dataframe, "vp", int(self.vp_window.value), int(self.vp_bins.value))

        dataframe = self._merge_informative_vp(dataframe, metadata, "4h", "vp4h", int(self.vp_4h_window.value), int(self.vp_4h_bins.value))

        dataframe = self._merge_informative_vp(dataframe, metadata, "1d", "vp1d", int(self.vp_1d_window.value), int(self.vp_1d_bins.value))

        dataframe = add_sieve2_guard_indicators(self, dataframe, metadata)
        dataframe = self._s3_lzr_add_levels(dataframe)

        return dataframe



    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        _ = metadata

        dataframe["enter_long"] = 0

        dataframe["enter_short"] = 0

        dataframe["enter_tag"] = None



        condition = self._entry_condition(dataframe)

        if bool(self.use_volume_guard.value):

            condition &= self._volume_guard(dataframe)

        if bool(self.use_close_direction_guard.value):

            condition &= self._close_direction_guard(dataframe)

        if bool(self.use_vp_1h_guard.value):

            condition &= self._vp_guard(dataframe, "vp", SIDE, str(self.vp_guard_mode.value), float(self.vp_score_min.value), float(self.vp_context_min.value))

        if bool(self.use_vp_4h_guard.value):

            condition &= self._vp_guard(dataframe, "vp4h", SIDE, str(self.vp_guard_mode.value), float(self.vp_score_min.value), float(self.vp_context_min.value))

        if bool(self.use_vp_1d_guard.value):

            condition &= self._vp_guard(dataframe, "vp1d", SIDE, str(self.vp_guard_mode.value), float(self.vp_score_min.value), float(self.vp_context_min.value))



        condition = apply_sieve2_optional_guards(self, dataframe, condition, getattr(self, "ENTRY_SIDE", globals().get("SIDE", "long")))

        valid = condition.fillna(False) & dataframe["volume"].gt(0.0) & dataframe["close"].notna()

        dataframe.loc[valid, "enter_long"] = 1

        dataframe.loc[valid, "enter_tag"] = ENTRY_TAG

        return dataframe



    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        _ = metadata

        dataframe["exit_long"] = 0

        dataframe["exit_short"] = 0

        dataframe["exit_tag"] = None

        return dataframe



    @staticmethod

    def _atr(dataframe: DataFrame, period: int = 14) -> Series:

        high = _num(dataframe, "high")

        low = _num(dataframe, "low")

        close = _num(dataframe, "close")

        previous_close = close.shift(1)

        true_range = pd.concat(

            [

                high.sub(low),

                high.sub(previous_close).abs(),

                low.sub(previous_close).abs(),

            ],

            axis=1,

        ).max(axis=1)

        return true_range.rolling(period, min_periods=max(2, period // 2)).mean()



    @staticmethod

    def _bars_since(signal: Series) -> Series:

        clean = pd.Series(signal, index=signal.index).fillna(False).astype(bool)

        positions = pd.Series(np.arange(len(clean), dtype="float64"), index=clean.index)

        last_hit = positions.where(clean).ffill()

        return positions.sub(last_hit).fillna(9999.0)



    @staticmethod

    def _period_key(dates: Series, period: str) -> Series:

        if period == "week":

            iso = dates.dt.isocalendar()

            return iso["year"].astype("string").str.cat(iso["week"].astype("string").str.zfill(2), sep="-")

        if period == "month":

            return dates.dt.strftime("%Y-%m")

        return dates.dt.strftime("%Y-%m-%d")



    def _selected_period(self) -> str:

        if PERIOD_KIND == "select":

            return str(self.multi_period.value)

        if PERIOD_KIND in PERIOD_CHOICES:

            return PERIOD_KIND

        return "day"



    def _add_volume_pressure(self, dataframe: DataFrame) -> DataFrame:

        volume_window = int(self.volume_window.value)

        volume_mean = _num(dataframe, "volume").rolling(volume_window, min_periods=max(2, volume_window // 3)).mean()

        candle_range = _num(dataframe, "high").sub(_num(dataframe, "low")).replace(0.0, np.nan)

        close_location = _num(dataframe, "close").sub(_num(dataframe, "low")).div(candle_range).clip(0.0, 1.0)

        dataframe["entry_close_location"] = close_location

        dataframe["entry_volume_ratio"] = _num(dataframe, "volume").div(volume_mean.replace(0.0, np.nan))

        dataframe["entry_pressure"] = close_location.sub(0.5).mul(2.0)

        return dataframe





    def _add_prior_period_levels(self, dataframe: DataFrame) -> DataFrame:

        dates = pd.to_datetime(dataframe["date"], utc=True, errors="coerce")

        high = _num(dataframe, "high")

        low = _num(dataframe, "low")

        for period in PERIOD_CHOICES:

            key = self._period_key(dates, period)

            grouped = pd.DataFrame({"period": key, "high": high, "low": low}).groupby("period", sort=True).agg(period_high=("high", "max"), period_low=("low", "min"))

            grouped["prior_high"] = grouped["period_high"].shift(1)

            grouped["prior_low"] = grouped["period_low"].shift(1)

            dataframe[f"prior_{period}_high"] = key.map(grouped["prior_high"]).astype("float64")

            dataframe[f"prior_{period}_low"] = key.map(grouped["prior_low"]).astype("float64")

        return dataframe



    def _add_rolling_levels(self, dataframe: DataFrame) -> DataFrame:

        lookback = int(self.rolling_level_lookback.value)

        min_periods = max(4, lookback // 4)

        dataframe["rolling_resistance"] = _num(dataframe, "high").shift(1).rolling(lookback, min_periods=min_periods).max()

        dataframe["rolling_support"] = _num(dataframe, "low").shift(1).rolling(lookback, min_periods=min_periods).min()

        return dataframe



    def _add_avwap(self, dataframe: DataFrame) -> DataFrame:

        lookback = int(self.avwap_anchor_lookback.value)

        low = _num(dataframe, "low")

        high = _num(dataframe, "high")

        prior_low = low.shift(1)

        prior_high = high.shift(1)

        low_reset = prior_low.le(prior_low.rolling(lookback, min_periods=2).min())

        high_reset = prior_high.ge(prior_high.rolling(lookback, min_periods=2).max())

        dataframe["avwap_from_low"] = self._anchored_vwap_series(dataframe, low_reset)

        dataframe["avwap_from_high"] = self._anchored_vwap_series(dataframe, high_reset)

        typical = _num(dataframe, "high").add(_num(dataframe, "low")).add(_num(dataframe, "close")).div(3.0)

        deviation = typical.rolling(lookback, min_periods=max(5, lookback // 5)).std()

        band_mult = float(self.avwap_band_mult.value)

        dataframe["avwap_lower_band"] = dataframe["avwap_from_low"].sub(deviation.mul(band_mult))

        dataframe["avwap_upper_band"] = dataframe["avwap_from_high"].add(deviation.mul(band_mult))

        return dataframe



    @staticmethod

    def _anchored_vwap_series(dataframe: DataFrame, reset: Series) -> Series:

        typical = _num(dataframe, "high").add(_num(dataframe, "low")).add(_num(dataframe, "close")).div(3.0)

        volume = _num(dataframe, "volume").clip(lower=0.0)

        groups = reset.fillna(False).astype(bool).cumsum()

        volume_sum = volume.groupby(groups).cumsum().replace(0.0, np.nan)

        price_volume_sum = typical.mul(volume).groupby(groups).cumsum()

        return price_volume_sum.div(volume_sum)



    def _add_supply_demand(self, dataframe: DataFrame) -> DataFrame:

        window = int(self.zone_impulse_window.value)

        open_ = _num(dataframe, "open")

        high = _num(dataframe, "high")

        low = _num(dataframe, "low")

        close = _num(dataframe, "close")

        body = close.sub(open_).abs()

        candle_range = high.sub(low).replace(0.0, np.nan)

        body_fraction = body.div(candle_range)

        atr = self._atr(dataframe)

        volume_ratio = _num(dataframe, "volume").div(_num(dataframe, "volume").rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan))

        impulse_distance = atr.mul(float(self.zone_impulse_atr_min.value))

        quality = body_fraction.ge(float(self.zone_body_fraction_min.value)) & volume_ratio.ge(float(self.zone_volume_ratio_min.value))

        bull_impulse = close.gt(open_) & close.sub(open_).ge(impulse_distance) & quality

        bear_impulse = close.lt(open_) & open_.sub(close).ge(impulse_distance) & quality

        body_low = pd.concat([open_, close], axis=1).min(axis=1)

        body_high = pd.concat([open_, close], axis=1).max(axis=1)

        dataframe["demand_zone_low"] = low.where(bull_impulse).ffill().shift(1)

        dataframe["demand_zone_high"] = body_low.where(bull_impulse).ffill().shift(1)

        dataframe["supply_zone_low"] = body_high.where(bear_impulse).ffill().shift(1)

        dataframe["supply_zone_high"] = high.where(bear_impulse).ffill().shift(1)

        dataframe["demand_zone_age"] = self._bars_since(bull_impulse).shift(1)

        dataframe["supply_zone_age"] = self._bars_since(bear_impulse).shift(1)

        return dataframe



    def _add_liquidity_levels(self, dataframe: DataFrame) -> DataFrame:

        lookback = int(self.equal_level_lookback.value)

        min_touches = int(self.equal_level_min_touches.value)

        tolerance = float(self.equal_level_tolerance_pct.value)

        min_periods = max(4, lookback // 4)

        prior_high = _num(dataframe, "high").shift(1)

        prior_low = _num(dataframe, "low").shift(1)

        high_level = prior_high.rolling(lookback, min_periods=min_periods).max()

        low_level = prior_low.rolling(lookback, min_periods=min_periods).min()

        high_touches = prior_high.sub(high_level).abs().le(high_level.abs().mul(tolerance)).rolling(lookback, min_periods=min_periods).sum()

        low_touches = prior_low.sub(low_level).abs().le(low_level.abs().mul(tolerance)).rolling(lookback, min_periods=min_periods).sum()

        dataframe["equal_high_level"] = high_level.where(high_touches.ge(min_touches))

        dataframe["equal_low_level"] = low_level.where(low_touches.ge(min_touches))

        dataframe["range_resistance"] = high_level

        dataframe["range_support"] = low_level

        return dataframe



    def _add_volume_profile(self, dataframe: DataFrame, prefix: str, window: int, bins: int) -> DataFrame:

        return add_volume_profile(

            dataframe,

            window=window,

            bins=bins,

            value_area_pct=float(self.vp_value_area_pct.value),

            price_source=str(self.vp_price_source.value),

            smooth_bins=int(self.vp_smooth_bins.value),

            hvn_threshold=float(self.vp_hvn_threshold.value),

            lvn_threshold=float(self.vp_lvn_threshold.value),

            pressure_delta_min=float(self.vp_pressure_delta_min.value),

            node_near_pct=float(self.vp_node_near_pct.value),

            volume_percentile_min=float(self.vp_volume_percentile_min.value),

            score_window=int(self.vp_score_window.value),

            fast_traverse_atr_mult=float(self.vp_fast_traverse_atr_mult.value),

            entry_score_margin=float(self.vp_entry_score_margin.value),

            prefix=prefix,

        )



    def _entry_condition(self, dataframe: DataFrame) -> Series:

        if CONCEPT == "prior_high_breakout":

            return self._prior_high_breakout(dataframe)

        if CONCEPT == "prior_low_breakdown":

            return self._prior_low_breakdown(dataframe)

        if CONCEPT == "avwap_reclaim":

            return self._avwap_reclaim(dataframe)

        if CONCEPT == "avwap_reject":

            return self._avwap_reject(dataframe)

        if CONCEPT == "avwap_lower_band_reclaim":

            return self._avwap_lower_band_reclaim(dataframe)

        if CONCEPT == "avwap_upper_band_reject":

            return self._avwap_upper_band_reject(dataframe)

        if CONCEPT == "demand_reclaim":

            return self._demand_reclaim(dataframe)

        if CONCEPT == "supply_reject":

            return self._supply_reject(dataframe)

        if CONCEPT == "demand_breakdown":

            return self._demand_breakdown(dataframe)

        if CONCEPT == "supply_breakout":

            return self._supply_breakout(dataframe)

        if CONCEPT == "equal_lows_sweep_reclaim":

            return self._equal_lows_sweep_reclaim(dataframe)

        if CONCEPT == "equal_highs_sweep_reject":

            return self._equal_highs_sweep_reject(dataframe)

        if CONCEPT == "range_low_sweep_reclaim":

            return self._range_low_sweep_reclaim(dataframe)

        if CONCEPT == "range_high_sweep_reject":

            return self._range_high_sweep_reject(dataframe)

        if CONCEPT == "multi2_prior_vp_breakout":

            return self._prior_high_breakout(dataframe) & self._vp_bull_ok(dataframe)

        if CONCEPT == "multi2_prior_vp_breakdown":

            return self._prior_low_breakdown(dataframe) & self._vp_bear_ok(dataframe)

        if CONCEPT == "multi3_prior_avwap_vp_breakout":

            return self._prior_high_breakout(dataframe) & self._avwap_trend_bull(dataframe) & self._vp_bull_ok(dataframe)

        if CONCEPT == "multi3_prior_avwap_vp_breakdown":

            return self._prior_low_breakdown(dataframe) & self._avwap_trend_bear(dataframe) & self._vp_bear_ok(dataframe)

        if CONCEPT == "multi2_demand_vp_reclaim":

            return self._demand_reclaim(dataframe) & self._vp_bull_ok(dataframe)

        if CONCEPT == "multi2_supply_vp_reject":

            return self._supply_reject(dataframe) & self._vp_bear_ok(dataframe)

        if CONCEPT == "multi2_liquidity_vp_reclaim":

            return self._equal_lows_sweep_reclaim(dataframe) & self._vp_bull_ok(dataframe)

        if CONCEPT == "multi2_liquidity_vp_reject":

            return self._equal_highs_sweep_reject(dataframe) & self._vp_bear_ok(dataframe)

        if CONCEPT == "multi4_prior_range_avwap_vp_breakout":

            return self._prior_high_breakout(dataframe) & self._rolling_resistance_breakout(dataframe) & self._avwap_trend_bull(dataframe) & self._vp_bull_ok(dataframe)

        if CONCEPT == "multi4_prior_range_avwap_vp_breakdown":

            return self._prior_low_breakdown(dataframe) & self._rolling_support_breakdown(dataframe) & self._avwap_trend_bear(dataframe) & self._vp_bear_ok(dataframe)

        return pd.Series(False, index=dataframe.index, dtype="bool")



    def _prior_level(self, dataframe: DataFrame, field: str) -> Series:

        period = self._selected_period()

        return _num(dataframe, f"prior_{period}_{field}", np.nan)



    def _prior_high_breakout(self, dataframe: DataFrame) -> Series:

        close = _num(dataframe, "close")

        level = self._prior_level(dataframe, "high")

        trigger = level.mul(1.0 + float(self.breakout_buffer_pct.value))

        return close.ge(trigger) & close.shift(1).lt(trigger.shift(1))



    def _prior_low_breakdown(self, dataframe: DataFrame) -> Series:

        close = _num(dataframe, "close")

        level = self._prior_level(dataframe, "low")

        trigger = level.mul(1.0 - float(self.breakout_buffer_pct.value))

        return close.le(trigger) & close.shift(1).gt(trigger.shift(1))



    def _rolling_resistance_breakout(self, dataframe: DataFrame) -> Series:

        close = _num(dataframe, "close")

        level = _num(dataframe, "rolling_resistance", np.nan)

        trigger = level.mul(1.0 + float(self.breakout_buffer_pct.value))

        return close.ge(trigger) & close.shift(1).lt(trigger.shift(1))



    def _rolling_support_breakdown(self, dataframe: DataFrame) -> Series:

        close = _num(dataframe, "close")

        level = _num(dataframe, "rolling_support", np.nan)

        trigger = level.mul(1.0 - float(self.breakout_buffer_pct.value))

        return close.le(trigger) & close.shift(1).gt(trigger.shift(1))



    def _avwap_reclaim(self, dataframe: DataFrame) -> Series:

        close = _num(dataframe, "close")

        avwap = _num(dataframe, "avwap_from_low", np.nan)

        trigger = avwap.mul(1.0 + float(self.reclaim_buffer_pct.value))

        return close.ge(trigger) & close.shift(1).lt(avwap.shift(1)) & _num(dataframe, "low").le(avwap.mul(1.0 + float(self.zone_near_pct.value)))



    def _avwap_reject(self, dataframe: DataFrame) -> Series:

        close = _num(dataframe, "close")

        avwap = _num(dataframe, "avwap_from_high", np.nan)

        trigger = avwap.mul(1.0 - float(self.reclaim_buffer_pct.value))

        return close.le(trigger) & close.shift(1).gt(avwap.shift(1)) & _num(dataframe, "high").ge(avwap.mul(1.0 - float(self.zone_near_pct.value)))



    def _avwap_lower_band_reclaim(self, dataframe: DataFrame) -> Series:

        close = _num(dataframe, "close")

        band = _num(dataframe, "avwap_lower_band", np.nan)

        return _num(dataframe, "low").le(band.mul(1.0 + float(self.sweep_buffer_pct.value))) & close.ge(band.mul(1.0 + float(self.reclaim_buffer_pct.value))) & close.gt(_num(dataframe, "open"))



    def _avwap_upper_band_reject(self, dataframe: DataFrame) -> Series:

        close = _num(dataframe, "close")

        band = _num(dataframe, "avwap_upper_band", np.nan)

        return _num(dataframe, "high").ge(band.mul(1.0 - float(self.sweep_buffer_pct.value))) & close.le(band.mul(1.0 - float(self.reclaim_buffer_pct.value))) & close.lt(_num(dataframe, "open"))



    def _demand_reclaim(self, dataframe: DataFrame) -> Series:

        low = _num(dataframe, "low")

        close = _num(dataframe, "close")

        zone_high = _num(dataframe, "demand_zone_high", np.nan)

        age_ok = _num(dataframe, "demand_zone_age", 9999.0).le(float(self.zone_max_age_bars.value))

        return age_ok & low.le(zone_high.mul(1.0 + float(self.zone_near_pct.value))) & close.ge(zone_high.mul(1.0 + float(self.reclaim_buffer_pct.value))) & close.gt(_num(dataframe, "open"))



    def _supply_reject(self, dataframe: DataFrame) -> Series:

        high = _num(dataframe, "high")

        close = _num(dataframe, "close")

        zone_low = _num(dataframe, "supply_zone_low", np.nan)

        age_ok = _num(dataframe, "supply_zone_age", 9999.0).le(float(self.zone_max_age_bars.value))

        return age_ok & high.ge(zone_low.mul(1.0 - float(self.zone_near_pct.value))) & close.le(zone_low.mul(1.0 - float(self.reclaim_buffer_pct.value))) & close.lt(_num(dataframe, "open"))



    def _demand_breakdown(self, dataframe: DataFrame) -> Series:

        close = _num(dataframe, "close")

        zone_low = _num(dataframe, "demand_zone_low", np.nan)

        trigger = zone_low.mul(1.0 - float(self.breakout_buffer_pct.value))

        age_ok = _num(dataframe, "demand_zone_age", 9999.0).le(float(self.zone_max_age_bars.value))

        return age_ok & close.le(trigger) & close.shift(1).gt(trigger.shift(1))



    def _supply_breakout(self, dataframe: DataFrame) -> Series:

        close = _num(dataframe, "close")

        zone_high = _num(dataframe, "supply_zone_high", np.nan)

        trigger = zone_high.mul(1.0 + float(self.breakout_buffer_pct.value))

        age_ok = _num(dataframe, "supply_zone_age", 9999.0).le(float(self.zone_max_age_bars.value))

        return age_ok & close.ge(trigger) & close.shift(1).lt(trigger.shift(1))



    def _equal_lows_sweep_reclaim(self, dataframe: DataFrame) -> Series:

        level = _num(dataframe, "equal_low_level", np.nan)

        return _num(dataframe, "low").le(level.mul(1.0 - float(self.sweep_buffer_pct.value))) & _num(dataframe, "close").ge(level.mul(1.0 + float(self.reclaim_buffer_pct.value))) & _num(dataframe, "close").gt(_num(dataframe, "open"))



    def _equal_highs_sweep_reject(self, dataframe: DataFrame) -> Series:

        level = _num(dataframe, "equal_high_level", np.nan)

        return _num(dataframe, "high").ge(level.mul(1.0 + float(self.sweep_buffer_pct.value))) & _num(dataframe, "close").le(level.mul(1.0 - float(self.reclaim_buffer_pct.value))) & _num(dataframe, "close").lt(_num(dataframe, "open"))



    def _range_low_sweep_reclaim(self, dataframe: DataFrame) -> Series:

        level = _num(dataframe, "range_support", np.nan)

        return _num(dataframe, "low").le(level.mul(1.0 - float(self.sweep_buffer_pct.value))) & _num(dataframe, "close").ge(level.mul(1.0 + float(self.reclaim_buffer_pct.value))) & _num(dataframe, "close").gt(_num(dataframe, "open"))



    def _range_high_sweep_reject(self, dataframe: DataFrame) -> Series:

        level = _num(dataframe, "range_resistance", np.nan)

        return _num(dataframe, "high").ge(level.mul(1.0 + float(self.sweep_buffer_pct.value))) & _num(dataframe, "close").le(level.mul(1.0 - float(self.reclaim_buffer_pct.value))) & _num(dataframe, "close").lt(_num(dataframe, "open"))



    def _avwap_trend_bull(self, dataframe: DataFrame) -> Series:

        return _num(dataframe, "close").ge(_num(dataframe, "avwap_from_low", np.nan).mul(1.0 + float(self.reclaim_buffer_pct.value)))



    def _avwap_trend_bear(self, dataframe: DataFrame) -> Series:

        return _num(dataframe, "close").le(_num(dataframe, "avwap_from_high", np.nan).mul(1.0 - float(self.reclaim_buffer_pct.value)))



    def _vp_bull_ok(self, dataframe: DataFrame) -> Series:

        return self._vp_guard(dataframe, "vp", "long", str(self.vp_guard_mode.value), float(self.vp_score_min.value), float(self.vp_context_min.value))



    def _vp_bear_ok(self, dataframe: DataFrame) -> Series:

        return self._vp_guard(dataframe, "vp", "short", str(self.vp_guard_mode.value), float(self.vp_score_min.value), float(self.vp_context_min.value))



    def _volume_guard(self, dataframe: DataFrame) -> Series:

        ratio_ok = _num(dataframe, "entry_volume_ratio").ge(float(self.volume_ratio_min.value))

        pressure = _num(dataframe, "entry_pressure")

        if SIDE == "long":

            return ratio_ok & pressure.ge(float(self.pressure_min.value))

        return ratio_ok & pressure.le(-float(self.pressure_min.value))



    def _close_direction_guard(self, dataframe: DataFrame) -> Series:

        close = _num(dataframe, "close")

        if SIDE == "long":

            return close.gt(close.shift(1))

        return close.lt(close.shift(1))



    def _merge_informative_vp(self, dataframe: DataFrame, metadata: dict, timeframe: str, prefix: str, window: int, bins: int) -> DataFrame:

        if not getattr(self, "dp", None) or "date" not in dataframe.columns:

            return dataframe

        pair = metadata.get("pair") if metadata else None

        if not pair:

            return dataframe

        informative = self.dp.get_pair_dataframe(pair=pair, timeframe=timeframe)

        if informative is None or informative.empty or "date" not in informative.columns:

            return dataframe

        informative = self._add_volume_profile(informative.copy(), prefix, window, bins)

        merge_columns = [f"{prefix}_{name}" for name in VP_COLUMNS if f"{prefix}_{name}" in informative.columns]

        if not merge_columns:

            return dataframe

        minutes = timeframe_to_minutes(timeframe)

        inf = informative[["date", *merge_columns]].copy().sort_values("date")

        inf["date_merge"] = inf["date"] + pd.to_timedelta(minutes, unit="m")

        base = dataframe.reset_index().rename(columns={"index": "__row_index"}).sort_values("date")

        merged = pd.merge_asof(

            base,

            inf[["date_merge", *merge_columns]].sort_values("date_merge"),

            left_on="date",

            right_on="date_merge",

            direction="backward",

        ).sort_values("__row_index")

        for column in merge_columns:

            dataframe[column] = merged[column].to_numpy()

        return dataframe



    @staticmethod

    def _score_guard(dataframe: DataFrame, prefix: str, side: str, score_min: float) -> Series:

        score = _num(dataframe, f"{prefix}_score_{side}")

        opposite = _num(dataframe, f"{prefix}_score_{'short' if side == 'long' else 'long'}")

        return score.ge(score_min) & score.ge(opposite)



    @staticmethod

    def _context_guard(dataframe: DataFrame, prefix: str, side: str, context_min: float) -> Series:

        if side == "long":

            context = _num(dataframe, f"{prefix}_context_score_bull")

            opposite = _num(dataframe, f"{prefix}_context_score_bear")

            market_ok = _num(dataframe, f"{prefix}_market_context").ge(0)

        else:

            context = _num(dataframe, f"{prefix}_context_score_bear")

            opposite = _num(dataframe, f"{prefix}_context_score_bull")

            market_ok = _num(dataframe, f"{prefix}_market_context").le(0)

        return context.ge(context_min) & context.ge(opposite) & market_ok



    def _vp_guard(self, dataframe: DataFrame, prefix: str, side: str, mode: str, score_min: float, context_min: float) -> Series:

        score_ok = self._score_guard(dataframe, prefix, side, score_min)

        context_ok = self._context_guard(dataframe, prefix, side, context_min)

        balance_ok = _num(dataframe, f"{prefix}_context_score_balance").ge(context_min)

        if mode == "score":

            return score_ok

        if mode == "context":

            return context_ok

        if mode == "score_or_context":

            return score_ok | context_ok

        if mode == "balance":

            return balance_ok

        if side == "long":

            return _num(dataframe, f"{prefix}_market_context").ge(0)

        return _num(dataframe, f"{prefix}_market_context").le(0)





    S3_BRANCH_FAMILY = "level_zone_reversal"

    EXIT_FAMILY = "level_zone_reversal"

    EXIT_HYPOTHESIS = 'Target-zone exit sweep using prior swing, VP, or indicator levels with bands, offsets, reversal confirmation, and selectable trigger timeframe.'

    SOURCE_ENTRY_STEM = "multi3_prior_avwap_vp_breakout_long"

    SOURCE_ENTRY_CLASS = "Sieve3ExitMulti3PriorAvwapVpBreakoutLong"

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



apply_s3_branch_surface(Sieve3ExitLevelZoneReversalFromMulti3PriorAvwapVpBreakoutLong)

apply_explicit_hyperopt_surface(Sieve3ExitLevelZoneReversalFromMulti3PriorAvwapVpBreakoutLong)

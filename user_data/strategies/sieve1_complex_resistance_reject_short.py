from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import BooleanParameter, CategoricalParameter, DecimalParameter, IntParameter, IStrategy

from user_data.Indicators.complex_pattern_structure import add_pattern_structure
from user_data.Indicators.complex_pivot_structure import add_pivot_structure
from user_data.Indicators.complex_relative_strength import add_relative_strength
from user_data.Indicators.complex_trendline_projection import add_trendline_projection
from user_data.Indicators.complex_volatility_cycles import add_volatility_cycles
from user_data.Indicators.complex_volume_indicators import add_complex_volume_indicators
from user_data.Indicators.complex_volume_profile import add_volume_profile

HYPEROPT_PARAM_ENV = "HYBRID_RECOVERY_HYPEROPT_PARAMS"
ENTRY_SIEVE_TAKE_PROFIT_ENV = "ENTRY_SIEVE_TAKE_PROFIT_PCT"
ENTRY_SIEVE_STOPLOSS_ENV = "ENTRY_SIEVE_STOPLOSS_PCT"


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


ENTRY_MODE = "entry_complex_resistance_reject_short"
ENTRY_TAG = "complex_res_reject_short"
SIDE = "short"
CORE_BEHAVIOR = "resistance rejection after stop-run"
GUARD_MODE_CHOICES = ["direction", "score", "context", "score_or_context", "balance"]
PRICE_SOURCE_CHOICES = ["close", "hl2", "hlc3", "ohlc4"]
VP_COLUMNS = [
    "score_long",
    "score_short",
    "context_score_bull",
    "context_score_bear",
    "context_score_balance",
    "market_context",
]



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


class Sieve1ComplexResistanceRejectShort(IStrategy):
    """
    Probe above resistance and close back below it with bearish pressure/upthrust evidence.

    Single-entry Entry Sieve research strategy.
    Objective: test whether resistance rejection after stop-run has standalone edge under fixed +2%/-2% exits.
    """

    INTERFACE_VERSION = 3

    timeframe = "1h"
    startup_candle_count = 360
    process_only_new_candles = True
    can_short = True

    minimal_roi = entry_sieve_minimal_roi(0.02)
    stoploss = entry_sieve_stoploss(-0.02)
    use_exit_signal = False
    use_custom_stoploss = False
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    pivot_strength = tagged_parameter(CategoricalParameter([3, 5, 8], default=5, space="buy", optimize=True, load=True))
    pivot_score_min = tagged_parameter(DecimalParameter(0.05, 0.75, decimals=2, default=0.25, space="buy", optimize=True, load=True))
    trendline_score_min = tagged_parameter(DecimalParameter(0.05, 0.75, decimals=2, default=0.25, space="buy", optimize=True, load=True))
    pattern_score_min = tagged_parameter(DecimalParameter(0.05, 0.75, decimals=2, default=0.25, space="buy", optimize=True, load=True))
    volatility_score_min = tagged_parameter(DecimalParameter(0.05, 0.75, decimals=2, default=0.25, space="buy", optimize=True, load=True))
    compression_score_min = tagged_parameter(DecimalParameter(0.05, 0.80, decimals=2, default=0.35, space="buy", optimize=True, load=True))
    volume_score_min = tagged_parameter(DecimalParameter(0.05, 0.80, decimals=2, default=0.25, space="buy", optimize=True, load=True))
    volume_rvol_min = tagged_parameter(DecimalParameter(0.80, 1.95, decimals=2, default=1.20, space="buy", optimize=True, load=True))
    volume_pressure_min = tagged_parameter(DecimalParameter(0.00, 1.50, decimals=2, default=0.25, space="buy", optimize=True, load=True))
    profile_score_min = tagged_parameter(DecimalParameter(0.05, 0.80, decimals=2, default=0.25, space="buy", optimize=True, load=True))
    rs_score_min = tagged_parameter(DecimalParameter(0.05, 0.80, decimals=2, default=0.25, space="buy", optimize=True, load=True))
    breakout_buffer_pct = tagged_parameter(DecimalParameter(0.000, 0.010, decimals=3, default=0.002, space="buy", optimize=True, load=True))
    recent_window = tagged_parameter(IntParameter(24, 144, default=72, space="buy", optimize=True, load=True))
    volume_sweep_window = tagged_parameter(IntParameter(12, 48, default=24, space="buy", optimize=True, load=True))
    profile_window = tagged_parameter(IntParameter(48, 144, default=96, space="buy", optimize=True, load=True))
    profile_bins = tagged_parameter(IntParameter(24, 64, default=40, space="buy", optimize=True, load=True))
    use_relative_strength = tagged_parameter(BooleanParameter(default=True, space="buy", optimize=True, load=True))
    use_volume_score = tagged_parameter(BooleanParameter(default=True, space="buy", optimize=True, load=True))
    use_profile_score = tagged_parameter(BooleanParameter(default=True, space="buy", optimize=True, load=True))

    profile_value_area_pct = tagged_parameter(DecimalParameter(0.55, 0.85, decimals=2, default=0.70, space="buy", optimize=True, load=True))
    profile_price_source = tagged_parameter(CategoricalParameter(PRICE_SOURCE_CHOICES, default="hlc3", space="buy", optimize=True, load=True))
    profile_smooth_bins = tagged_parameter(IntParameter(1, 6, default=3, space="buy", optimize=True, load=True))
    profile_hvn_threshold = tagged_parameter(DecimalParameter(0.50, 0.90, decimals=2, default=0.70, space="buy", optimize=True, load=True))
    profile_lvn_threshold = tagged_parameter(DecimalParameter(0.10, 0.55, decimals=2, default=0.35, space="buy", optimize=True, load=True))
    profile_node_near_pct = tagged_parameter(DecimalParameter(0.002, 0.030, decimals=3, default=0.010, space="buy", optimize=True, load=True))
    profile_node_hvn_strength_min = tagged_parameter(DecimalParameter(0.40, 0.95, decimals=2, default=0.70, space="buy", optimize=True, load=True))
    profile_node_lvn_thinness_min = tagged_parameter(DecimalParameter(0.25, 0.95, decimals=2, default=0.55, space="buy", optimize=True, load=True))
    profile_volume_percentile_min = tagged_parameter(DecimalParameter(0.00, 0.90, decimals=2, default=0.55, space="buy", optimize=True, load=True))
    profile_poc_migration_window = tagged_parameter(IntParameter(4, 36, default=12, space="buy", optimize=True, load=True))
    profile_score_window = tagged_parameter(IntParameter(12, 120, default=48, space="buy", optimize=True, load=True))
    profile_fast_traverse_atr_mult = tagged_parameter(DecimalParameter(0.50, 2.80, decimals=2, default=1.20, space="buy", optimize=True, load=True))
    profile_entry_score_margin = tagged_parameter(DecimalParameter(0.00, 0.20, decimals=2, default=0.02, space="buy", optimize=True, load=True))
    use_vp_4h_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=True, load=True))
    vp_4h_guard_mode = tagged_parameter(CategoricalParameter(GUARD_MODE_CHOICES, default="score_or_context", space="buy", optimize=True, load=True))
    vp_4h_window = tagged_parameter(IntParameter(12, 96, default=48, space="buy", optimize=True, load=True))
    vp_4h_bins = tagged_parameter(IntParameter(16, 64, default=36, space="buy", optimize=True, load=True))
    vp_4h_score_min = tagged_parameter(DecimalParameter(0.00, 1.00, decimals=2, default=0.25, space="buy", optimize=True, load=True))
    vp_4h_context_min = tagged_parameter(DecimalParameter(0.00, 1.00, decimals=2, default=0.28, space="buy", optimize=True, load=True))
    use_vp_1d_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=True, load=True))
    vp_1d_guard_mode = tagged_parameter(CategoricalParameter(GUARD_MODE_CHOICES, default="context", space="buy", optimize=True, load=True))
    vp_1d_window = tagged_parameter(IntParameter(10, 84, default=30, space="buy", optimize=True, load=True))
    vp_1d_bins = tagged_parameter(IntParameter(16, 64, default=36, space="buy", optimize=True, load=True))
    vp_1d_score_min = tagged_parameter(DecimalParameter(0.00, 1.00, decimals=2, default=0.25, space="buy", optimize=True, load=True))
    vp_1d_context_min = tagged_parameter(DecimalParameter(0.00, 1.00, decimals=2, default=0.28, space="buy", optimize=True, load=True))

    def informative_pairs(self) -> list[tuple[str, str]]:
        quote = str(self.config.get("stake_currency") or "USDT")
        trading_mode = str(self.config.get("trading_mode") or "")
        suffix = f":{quote}" if trading_mode == "futures" else ""
        pairs = [(f"BTC/{quote}{suffix}", self.timeframe)]
        if getattr(self, "dp", None):
            try:
                whitelist = self.dp.current_whitelist()
                pairs.extend((pair, "4h") for pair in whitelist)
                pairs.extend((pair, "1d") for pair in whitelist)
            except Exception:
                pass
        return pairs

    def _benchmark_dataframe(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        benchmark = pd.DataFrame({"benchmark_close": dataframe["close"]}, index=dataframe.index)
        if not getattr(self, "dp", None):
            return benchmark
        quote = str(self.config.get("stake_currency") or "USDT")
        trading_mode = str(self.config.get("trading_mode") or "")
        suffix = f":{quote}" if trading_mode == "futures" else ""
        btc_pair = f"BTC/{quote}{suffix}"
        try:
            btc = self.dp.get_pair_dataframe(pair=btc_pair, timeframe=self.timeframe)
        except Exception:
            return benchmark
        if btc is None or btc.empty or "date" not in btc.columns or "date" not in dataframe.columns:
            return benchmark
        btc_frame = btc[["date", "close"]].rename(columns={"close": "benchmark_close"}).sort_values("date")
        merged = pd.merge_asof(
            dataframe[["date"]].sort_values("date"),
            btc_frame,
            on="date",
            direction="backward",
        ).set_index(dataframe.index)
        return merged[["benchmark_close"]].ffill()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        strength = int(self.pivot_strength.value)
        strengths = (3, 5, 8)
        buffer = float(self.breakout_buffer_pct.value)

        dataframe = add_pivot_structure(
            dataframe,
            strength=strength,
            strengths=strengths,
            breakout_buffer_pct=buffer,
            prefix="pa",
        )
        dataframe = add_trendline_projection(
            dataframe,
            strength=strength,
            strengths=strengths,
            pivot_prefix="pa",
            output_prefix="tl",
            missing_pivot_mode="raise",
            breakout_buffer_pct=buffer,
        )
        dataframe = add_pattern_structure(
            dataframe,
            pivot_prefix="pa",
            trendline_prefix="tl",
            output_prefix="pat",
            breakout_buffer_pct=buffer,
        )
        dataframe = add_volatility_cycles(
            dataframe,
            trendline_prefix="tl",
            output_prefix="vc",
            expansion_volume_rvol=float(self.volume_rvol_min.value),
        )
        dataframe = add_complex_volume_indicators(
            dataframe,
            sweep_window=int(self.volume_sweep_window.value),
            expansion_rvol=float(self.volume_rvol_min.value),
            pressure_zscore_min=float(self.volume_pressure_min.value),
            prefix="vol",
        )
        dataframe = add_volume_profile(
            dataframe,
            window=int(self.profile_window.value),
            bins=int(self.profile_bins.value),
            value_area_pct=float(self.profile_value_area_pct.value),
            price_source=str(self.profile_price_source.value),
            smooth_bins=int(self.profile_smooth_bins.value),
            hvn_threshold=float(self.profile_hvn_threshold.value),
            lvn_threshold=float(self.profile_lvn_threshold.value),
            pressure_delta_min=max(0.01, float(self.volume_pressure_min.value) / 10.0, float(self.profile_node_near_pct.value) / 5.0),
            node_near_pct=float(self.profile_node_near_pct.value),
            node_hvn_strength_min=float(self.profile_node_hvn_strength_min.value),
            node_lvn_thinness_min=float(self.profile_node_lvn_thinness_min.value),
            volume_percentile_min=float(self.profile_volume_percentile_min.value),
            poc_migration_window=int(self.profile_poc_migration_window.value),
            score_window=int(self.profile_score_window.value),
            fast_traverse_atr_mult=float(self.profile_fast_traverse_atr_mult.value),
            entry_score_margin=float(self.profile_entry_score_margin.value),
            prefix="vp",
        )
        dataframe = add_relative_strength(
            dataframe,
            self._benchmark_dataframe(dataframe, metadata),
            benchmark_close="benchmark_close",
            min_outperformance=0.0,
            prefix="rs",
        )
        dataframe = self._merge_informative_vp(dataframe, metadata, "4h", "vp4h", int(self.vp_4h_window.value), int(self.vp_4h_bins.value))
        dataframe = self._merge_informative_vp(dataframe, metadata, "1d", "vp1d", int(self.vp_1d_window.value), int(self.vp_1d_bins.value))

        window = int(self.recent_window.value)
        dataframe["recent_high"] = dataframe["high"].rolling(window, min_periods=max(12, window // 3)).max().shift(1)
        dataframe["recent_low"] = dataframe["low"].rolling(window, min_periods=max(12, window // 3)).min().shift(1)
        resistance_parts = pd.concat([_num(dataframe, "pa_resistance_line"), _num(dataframe, "tl_resistance_line"), dataframe["recent_high"]], axis=1)
        support_parts = pd.concat([_num(dataframe, "pa_support_line"), _num(dataframe, "tl_support_line"), dataframe["recent_low"]], axis=1)
        dataframe["resistance_ref"] = resistance_parts.max(axis=1)
        dataframe["support_ref"] = support_parts.min(axis=1)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        buffer = float(self.breakout_buffer_pct.value)


        swept_resistance = dataframe["high"] >= dataframe["resistance_ref"] * (1.0 + buffer)
        rejected_resistance = dataframe["close"] < dataframe["resistance_ref"] * (1.0 - buffer)
        condition = (
            swept_resistance
            & rejected_resistance
            & (_num(dataframe, "pa_score_short") >= float(self.pivot_score_min.value))
            & (_num(dataframe, "tl_score_short") >= float(self.trendline_score_min.value))
            & (_bool(dataframe, "vol_liq_stoprun_short") | _bool(dataframe, "vol_evr_upthrust") | _bool(dataframe, "vp_upper_rejection_with_pressure"))
        )

        if bool(self.use_volume_score.value):
            volume_score = _num(dataframe, "vol_score_long" if SIDE == "long" else "vol_score_short")
            condition &= volume_score >= float(self.volume_score_min.value)
        if bool(self.use_profile_score.value):
            profile_score = _num(dataframe, "vp_score_long" if SIDE == "long" else "vp_score_short")
            condition &= profile_score >= float(self.profile_score_min.value)
        if bool(self.use_relative_strength.value):
            rs_score = _num(dataframe, "rs_score_long" if SIDE == "long" else "rs_score_short")
            condition &= rs_score >= float(self.rs_score_min.value)

        if bool(self.use_vp_4h_guard.value):
            condition &= self._vp_guard(dataframe, "vp4h", SIDE, str(self.vp_4h_guard_mode.value), float(self.vp_4h_score_min.value), float(self.vp_4h_context_min.value))
        if bool(self.use_vp_1d_guard.value):
            condition &= self._vp_guard(dataframe, "vp1d", SIDE, str(self.vp_1d_guard_mode.value), float(self.vp_1d_score_min.value), float(self.vp_1d_context_min.value))

        valid = condition.fillna(False) & dataframe["volume"].gt(0.0) & dataframe["close"].notna()
        dataframe.loc[valid, "enter_short"] = 1
        dataframe.loc[valid, "enter_tag"] = ENTRY_TAG
        return dataframe


    def _merge_informative_vp(self, dataframe: DataFrame, metadata: dict, timeframe: str, prefix: str, window: int, bins: int) -> DataFrame:
        if "date" not in dataframe.columns or not getattr(self, "dp", None):
            return dataframe
        pair = str(metadata.get("pair") or "")
        if not pair:
            return dataframe
        informative = self.dp.get_pair_dataframe(pair=pair, timeframe=timeframe)
        if informative is None or informative.empty or "date" not in informative.columns:
            return dataframe
        informative = add_volume_profile(
            informative.copy(),
            window=window,
            bins=bins,
            value_area_pct=float(self.profile_value_area_pct.value),
            price_source=str(self.profile_price_source.value),
            smooth_bins=int(self.profile_smooth_bins.value),
            hvn_threshold=float(self.profile_hvn_threshold.value),
            lvn_threshold=float(self.profile_lvn_threshold.value),
            pressure_delta_min=max(0.01, float(self.volume_pressure_min.value) / 10.0, float(self.profile_node_near_pct.value) / 5.0),
            node_near_pct=float(self.profile_node_near_pct.value),
            node_hvn_strength_min=float(self.profile_node_hvn_strength_min.value),
            node_lvn_thinness_min=float(self.profile_node_lvn_thinness_min.value),
            volume_percentile_min=float(self.profile_volume_percentile_min.value),
            poc_migration_window=int(self.profile_poc_migration_window.value),
            score_window=int(self.profile_score_window.value),
            fast_traverse_atr_mult=float(self.profile_fast_traverse_atr_mult.value),
            entry_score_margin=float(self.profile_entry_score_margin.value),
            prefix=prefix,
        )
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

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe


apply_explicit_hyperopt_surface(Sieve1ComplexResistanceRejectShort)



from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.strategy import BooleanParameter, CategoricalParameter, DecimalParameter, IntParameter, IStrategy

from entry_sieve_tools import apply_explicit_hyperopt_surface, entry_sieve_minimal_roi, entry_sieve_stoploss
from user_data.Indicators.complex_pattern_structure import add_pattern_structure
from user_data.Indicators.complex_pivot_structure import add_pivot_structure
from user_data.Indicators.complex_relative_strength import add_relative_strength
from user_data.Indicators.complex_trendline_projection import add_trendline_projection
from user_data.Indicators.complex_volatility_cycles import add_volatility_cycles
from user_data.Indicators.complex_volume_indicators import add_complex_volume_indicators
from user_data.Indicators.complex_volume_profile import add_volume_profile


ENTRY_MODE = "entry_complex_support_reclaim_long"
ENTRY_TAG = "codex_complex_sup_reclaim_long"
SIDE = "long"
CORE_BEHAVIOR = "support reclaim after liquidity sweep"


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


class CodexComplexSupportReclaimLong(IStrategy):
    """
    Sweep or undercut support, then reclaim it with absorption/profile support pressure.

    Single-entry Entry Sieve research strategy.
    Objective: test whether support reclaim after liquidity sweep has standalone edge under fixed +2%/-2% exits.
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
    volume_rvol_min = tagged_parameter(DecimalParameter(0.80, 3.00, decimals=2, default=1.20, space="buy", optimize=True, load=True))
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

    def informative_pairs(self) -> list[tuple[str, str]]:
        quote = str(self.config.get("stake_currency") or "USDT")
        trading_mode = str(self.config.get("trading_mode") or "")
        suffix = f":{quote}" if trading_mode == "futures" else ""
        return [(f"BTC/{quote}{suffix}", self.timeframe)]

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
            pressure_delta_min=max(0.01, float(self.volume_pressure_min.value) / 10.0),
            volume_percentile_min=0.50,
            prefix="vp",
        )
        dataframe = add_relative_strength(
            dataframe,
            self._benchmark_dataframe(dataframe, metadata),
            benchmark_close="benchmark_close",
            min_outperformance=0.0,
            prefix="rs",
        )

        window = int(self.recent_window.value)
        dataframe["codex_recent_high"] = dataframe["high"].rolling(window, min_periods=max(12, window // 3)).max().shift(1)
        dataframe["codex_recent_low"] = dataframe["low"].rolling(window, min_periods=max(12, window // 3)).min().shift(1)
        resistance_parts = pd.concat([_num(dataframe, "pa_resistance_line"), _num(dataframe, "tl_resistance_line"), dataframe["codex_recent_high"]], axis=1)
        support_parts = pd.concat([_num(dataframe, "pa_support_line"), _num(dataframe, "tl_support_line"), dataframe["codex_recent_low"]], axis=1)
        dataframe["codex_resistance_ref"] = resistance_parts.max(axis=1)
        dataframe["codex_support_ref"] = support_parts.min(axis=1)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        buffer = float(self.breakout_buffer_pct.value)


        swept_support = dataframe["low"] <= dataframe["codex_support_ref"] * (1.0 - buffer)
        reclaimed_support = dataframe["close"] > dataframe["codex_support_ref"] * (1.0 + buffer)
        condition = (
            swept_support
            & reclaimed_support
            & (_num(dataframe, "pa_score_long") >= float(self.pivot_score_min.value))
            & (_num(dataframe, "tl_score_long") >= float(self.trendline_score_min.value))
            & (_bool(dataframe, "vol_liq_stoprun_long") | _bool(dataframe, "vol_evr_spring") | _bool(dataframe, "vp_lower_rejection_with_pressure"))
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

        valid = condition.fillna(False) & dataframe["volume"].gt(0.0) & dataframe["close"].notna()
        if SIDE == "long":
            dataframe.loc[valid, "enter_long"] = 1
        else:
            dataframe.loc[valid, "enter_short"] = 1
        dataframe.loc[valid, "enter_tag"] = ENTRY_TAG
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe


apply_explicit_hyperopt_surface(CodexComplexSupportReclaimLong)

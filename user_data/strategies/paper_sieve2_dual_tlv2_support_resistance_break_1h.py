"""Paper-only dual-side probe of the selected Sieve2 TLV2 breakout.

The saved Sieve2 support-break short remains the tested side. The symmetric
resistance-break long is experimental; their combinations are not validated.
Selection: entry_sieve/results/20260521T012740_entry_all_resume.jsonl:1004;
the matching saved backtest used TP/SL 2%/2%.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.enums import RunMode
from freqtrade.strategy import IStrategy

_INDICATOR_DIR = Path(__file__).resolve().parents[1] / "Indicators"
if str(_INDICATOR_DIR) not in sys.path:
    sys.path.insert(0, str(_INDICATOR_DIR))

from user_data.Indicators.complex_trendline_projection_v2 import add_trendline_projection_v2
from user_data.strategies.sieve_guard_helpers import (
    add_sieve2_guard_indicators,
    apply_sieve2_optional_guards,
)


TIMEFRAME = "1h"
SAVED_ENTRY_PARAMS = {
    # Selected Sieve2 parameter JSON. Inactive saved parameters are retained too.
    "use_volume_guard": False,
    "volume_guard_window": 48,
    "volume_ratio_min": 1.0,
    "use_pressure_guard": True,
    "pressure_window": 12,
    "pressure_min": 0.15,
    # Remaining entry/indicator settings are the exact source defaults.
    "use_accumulation_guard": False,
    "use_body_direction_guard": False,
    "use_close_direction_guard": False,
    "pivot_strength": 2,
    "min_output_line_score": 0.50,
    "min_output_active_bars": 8,
    "max_active_line_distance_atr_mult": 6.0,
    "proximity_rank_weight": 0.05,
    "proximity_rank_distance_cap_atr": 6.0,
    "min_anchor_bars": 10,
    "max_anchor_bars": 50,
    "max_slope_atr_per_bar": 0.35,
    "max_projection_bars": 50,
    "score_min": 0.50,
    "distance_atr_max": 0.85,
    "line_buffer_pct": 0.003,
    "line_slope_min_pct": 0.001,
    "use_sieve2_vp_guard": False,
    "sieve2_vp_guard_mode": "score_or_context",
    "sieve2_vp_window": 96,
    "sieve2_vp_bins": 36,
    "sieve2_vp_score_min": 0.25,
    "sieve2_vp_context_min": 0.28,
    "use_sieve2_market_guard": False,
    "sieve2_market_guard_mode": "pressure_or_trend",
    "sieve2_market_window": 24,
    "sieve2_market_pressure_min": 0.07,
    "sieve2_market_trend_min": 0.25,
    "sieve2_rs_benchmark_pair": "BTC/USDT:USDT",
    "sieve2_rs_score_min": 0.45,
}
SAVED_EXIT_PARAMS = {"minimal_roi": {"0": 0.02}, "stoploss": -0.02}


def _num(frame: DataFrame, column: str, default: float | Series = 0.0) -> Series:
    if column not in frame.columns:
        if isinstance(default, Series):
            return pd.to_numeric(default, errors="coerce")
        return pd.Series(float(default), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _cross_above(value: Series, level: Series) -> Series:
    return value.gt(level) & value.shift(1).le(level.shift(1))


def _cross_below(value: Series, level: Series) -> Series:
    return value.lt(level) & value.shift(1).ge(level.shift(1))


def _normalized_roi(roi: dict[Any, Any]) -> dict[int, float]:
    return {int(duration): float(value) for duration, value in roi.items()}


class PaperSieve2DualTLV2SupportResistanceBreak1h(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 160
    process_only_new_candles = True
    can_short = True

    minimal_roi = dict(SAVED_EXIT_PARAMS["minimal_roi"])
    stoploss = SAVED_EXIT_PARAMS["stoploss"]
    use_exit_signal = False
    use_custom_stoploss = False
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    for _name, _value in SAVED_ENTRY_PARAMS.items():
        locals()[_name] = _value
    del _name, _value

    def bot_start(self, **kwargs: Any) -> None:
        _ = kwargs
        if self.config.get("dry_run") is not True or self.config.get("runmode") != RunMode.DRY_RUN:
            raise RuntimeError("PaperSieve2DualTLV2SupportResistanceBreak1h is dry-run only")
        expected = {
            "timeframe": TIMEFRAME,
            "startup_candle_count": 160,
            "process_only_new_candles": True,
            "stoploss": SAVED_EXIT_PARAMS["stoploss"],
            "use_exit_signal": False,
            "use_custom_stoploss": False,
            "position_adjustment_enable": False,
            "max_entry_position_adjustment": 0,
            "trailing_stop": False,
            "ignore_roi_if_entry_signal": False,
            "can_short": True,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise RuntimeError(f"Frozen Sieve2 paper setting drifted: {name}")
        if _normalized_roi(self.minimal_roi) != _normalized_roi(SAVED_EXIT_PARAMS["minimal_roi"]):
            raise RuntimeError("Frozen Sieve2 paper setting drifted: minimal_roi")
        for name, value in SAVED_ENTRY_PARAMS.items():
            if getattr(self, name) != value:
                raise RuntimeError(f"Frozen Sieve2 entry setting drifted: {name}")

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float:
        _ = (pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs)
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = add_trendline_projection_v2(
            dataframe,
            timeframe=self.timeframe,
            pivot_strength=self.pivot_strength,
            min_output_line_score=self.min_output_line_score,
            min_output_active_bars=self.min_output_active_bars,
            max_active_line_distance_atr_mult=self.max_active_line_distance_atr_mult,
            proximity_rank_weight=self.proximity_rank_weight,
            proximity_rank_distance_cap_atr=self.proximity_rank_distance_cap_atr,
            min_anchor_bars=self.min_anchor_bars,
            max_anchor_bars=self.max_anchor_bars,
            max_slope_atr_per_bar=self.max_slope_atr_per_bar,
            max_projection_bars=self.max_projection_bars,
            raw_line_output_count=1,
            output_prefix="tlv2",
        )
        return add_sieve2_guard_indicators(self, dataframe, metadata)

    def _common_guards(self, dataframe: DataFrame, side: str) -> Series:
        close = _num(dataframe, "close")
        guard = pd.Series(True, index=dataframe.index, dtype="bool")
        if self.use_volume_guard:
            volume = _num(dataframe, "volume").clip(lower=0.0)
            window = self.volume_guard_window
            baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
            guard &= volume.ge(baseline.mul(self.volume_ratio_min))
        if self.use_pressure_guard:
            open_ = _num(dataframe, "open")
            high = _num(dataframe, "high")
            low = _num(dataframe, "low")
            volume = _num(dataframe, "volume").clip(lower=0.0).fillna(0.0)
            candle_range = (high - low).replace(0.0, np.nan)
            body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
            close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
            pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
            directional_volume = (pressure * volume).fillna(0.0)
            window = self.pressure_window
            baseline = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
            pressure_ratio = directional_volume.rolling(window, min_periods=max(2, window // 3)).sum() / baseline
            guard &= pressure_ratio.le(-self.pressure_min) if side == "short" else pressure_ratio.ge(self.pressure_min)
        if self.use_accumulation_guard:
            open_ = _num(dataframe, "open")
            high = _num(dataframe, "high")
            low = _num(dataframe, "low")
            volume = _num(dataframe, "volume").clip(lower=0.0).fillna(0.0)
            candle_range = (high - low).replace(0.0, np.nan)
            body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
            close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
            directional_volume = (((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0) * volume).fillna(0.0)
            accumulation = directional_volume.rolling(self.pressure_window, min_periods=max(2, self.pressure_window // 3)).sum()
            guard &= accumulation.le(0.0) if side == "short" else accumulation.ge(0.0)
        if self.use_body_direction_guard:
            open_ = _num(dataframe, "open")
            guard &= close.lt(open_) if side == "short" else close.gt(open_)
        if self.use_close_direction_guard:
            guard &= close.lt(close.shift(1)) if side == "short" else close.gt(close.shift(1))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None

        close = _num(dataframe, "close")
        resistance = _num(dataframe, "tlv2_resistance_line_rank0", np.nan).mul(1.0 + self.line_buffer_pct)
        support = _num(dataframe, "tlv2_support_line_rank0", np.nan).mul(1.0 - self.line_buffer_pct)
        score_min = self.score_min
        long_condition = (
            _num(dataframe, "tlv2_resistance_score_rank0").ge(score_min)
            & _cross_above(close, resistance)
        )
        short_condition = (
            _num(dataframe, "tlv2_support_score_rank0").ge(score_min)
            & _cross_below(close, support)
        )
        long_condition &= self._common_guards(dataframe, "long")
        short_condition &= self._common_guards(dataframe, "short")
        long_condition = apply_sieve2_optional_guards(self, dataframe, long_condition, "long")
        short_condition = apply_sieve2_optional_guards(self, dataframe, short_condition, "short")

        long_condition &= dataframe["volume"].gt(0.0) & close.notna()
        short_condition &= dataframe["volume"].gt(0.0) & close.notna()
        both = long_condition & short_condition
        long_condition &= ~both
        short_condition &= ~both

        dataframe.loc[long_condition, "enter_long"] = 1
        dataframe.loc[long_condition, "enter_tag"] = "experimental_symmetric_resistance_breakout_long_1h"
        dataframe.loc[short_condition, "enter_short"] = 1
        dataframe.loc[short_condition, "enter_tag"] = "tested_sieve2_tlv2_support_breakdown_short_1h"
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe

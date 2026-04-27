from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.strategy import CategoricalParameter, IStrategy, merge_informative_pair

from user_data.Indicators.complex_pivot_structure import PivotStructureConfig, add_pivot_structure
from user_data.Indicators.complex_trendline_projection import TrendlineProjectionConfig, add_trendline_projection
from user_data.Indicators.complex_volume_indicators import ComplexVolumeConfig, add_complex_volume_indicators


def tagged_parameter(param: Any, *tags: str) -> Any:
    """Attach simple family/mode metadata for Explorer/catalog tools."""
    family_map = {
        "entry": "entries",
        "entries": "entries",
        "structure": "entries",
        "volume": "entries",
        "volume_pressure": "entries",
        "entry_enables": "entries",
        "entry_confirmation": "entries",
        "entry_structure": "entries",
        "trend_filter": "entries",
        "daily_structure": "entries",
        "hourly_execution": "entries",
        "line_quality": "entries",
        "target_space": "entries",
        "trendline_projection": "entries",
        "breakout_long": "entries",
        "exit": "exits",
        "exits": "exits",
        "exit_profile": "exits",
        "exit_peel": "exits",
        "exit_target": "exits",
        "entry_family_exit": "exits",
        "stoploss": "exits",
        "adjust_position": "adjust_position",
        "add_enables": "adjust_position",
        "add_rules": "adjust_position",
        "capital": "stake",
        "stake": "stake",
        "risk": "risk",
    }
    family: str | None = None
    mode: str | None = None
    for raw_tag in tags:
        tag = str(raw_tag or "").strip()
        if not tag or ":" not in tag:
            continue
        key, value = tag.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key == "family":
            mapped = family_map.get(value.lower())
            if mapped and family is None:
                family = mapped
        elif key == "mode" and value and mode is None:
            mode = value

    if family is None:
        if mode and mode.startswith("exit_"):
            family = "exits"
        elif mode and mode.startswith("adjust_"):
            family = "adjust_position"
        elif mode and mode.startswith("stake_"):
            family = "stake"
        elif mode and mode.startswith("risk_"):
            family = "risk"
        else:
            family = "entries"

    if mode is None:
        mode_defaults = {
            "entries": "entry_core",
            "exits": "exit_core",
            "adjust_position": "adjust_core",
            "stake": "stake_core",
            "risk": "risk_core",
        }
        mode = mode_defaults.get(family, "entry_core")

    setattr(param, "batch_tags", (f"family:{family}", f"mode:{mode}"))
    return param


class PivotTrendlineMTFResearchStrategy(IStrategy):
    """1h execution strategy using completed 1d pivot/trendline structure.

    Design intent:
    - 1d pivot anchors and projected trendlines define the major map.
    - 1h pivot/trendline evidence times entries around that map.
    - Complex OHLCV volume evidence confirms pressure; no RSI/MACD/EMA gates.
    - No leverage override and no position adjustment; this isolates entry quality.
    """

    INTERFACE_VERSION = 3

    timeframe = "1h"
    informative_timeframe = "1d"
    can_short = True
    startup_candle_count = 600
    process_only_new_candles = True

    position_adjustment_enable = False
    minimal_roi = {"0": 0.04}
    stoploss = -0.10
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    trailing_stop = False
    use_custom_stoploss = False

    PIVOT_STRENGTH_CHOICES = [3, 5, 8, 13]
    PROJECTION_HORIZON_CHOICES = [1, 3, 6, 12]
    EVENT_WINDOW_CHOICES = [12, 24, 48, 96]
    ENTRY_MODE_CHOICES = ["support_reclaim", "resistance_reject", "resistance_breakout", "support_breakdown", "trend_pullback", "all"]
    ENTRY_PARAM_SPECS = {
        "daily_level_source": (["trendline", "pivot", "active_swing", "nearest"], "nearest", "daily_structure"),
        "daily_context_mode": (["level_required", "trend_or_level", "trend_required"], "trend_or_level", "daily_structure"),
        "daily_pivot_strength": (PIVOT_STRENGTH_CHOICES, 5, "daily_structure"),
        "hourly_pivot_strength": (PIVOT_STRENGTH_CHOICES, 5, "hourly_execution"),
        "projection_horizon": (PROJECTION_HORIZON_CHOICES, 3, "trendline_projection"),
        "event_window": (EVENT_WINDOW_CHOICES, 48, "line_quality"),
        "entry_zone_pct": ([0.003, 0.006, 0.010, 0.020, 0.040], 0.010, "entry_structure"),
        "breakout_buffer_pct": ([0.000, 0.001, 0.002, 0.004, 0.008], 0.002, "entry_structure"),
        "min_daily_context_score": ([0, 1, 2, 3, 4], 2, "daily_structure"),
        "min_hourly_execution_score": ([1, 2, 3, 4, 5], 3, "hourly_execution"),
        "min_volume_score": ([0, 1, 2, 3, 4], 2, "volume_pressure"),
        "min_line_respect_ratio": ([0.0, 0.25, 0.50, 0.67, 0.80], 0.50, "line_quality"),
        "max_line_violation_count": ([0, 1, 2, 3, 999], 2, "line_quality"),
        "min_target_distance_pct": ([0.000, 0.004, 0.008, 0.015, 0.030], 0.008, "target_space"),
        "level_confluence_max_pct": ([0.005, 0.010, 0.020, 0.040, 999.0], 0.020, "daily_structure"),
        "volume_rvol_min": ([0.0, 0.8, 1.0, 1.25, 1.5, 2.0], 1.0, "volume_pressure"),
        "volume_delta_abs_min": ([0.0, 0.15, 0.25, 0.50, 0.75], 0.25, "volume_pressure"),
    }

    entry_mode = tagged_parameter(
        CategoricalParameter(
            ENTRY_MODE_CHOICES,
            default="all",
            space="buy",
            optimize=True,
            load=True,
        ),
        "family:entry_structure",
        "mode:pivot_trendline_active_entry_mode",
    )

    exit_mode = tagged_parameter(
        CategoricalParameter(["opposite_daily_level", "daily_invalidation", "both"], default="both", space="sell", optimize=True, load=True),
        "family:exit_profile",
        "mode:pivot_trendline_basic_exit",
    )
    exit_buffer_pct = tagged_parameter(
        CategoricalParameter([0.000, 0.002, 0.004, 0.008, 0.015], default=0.004, space="sell", optimize=True, load=True),
        "family:exit_profile",
        "mode:pivot_trendline_basic_exit",
    )

    plot_config = {
        "main_plot": {
            "dbg_d1_support_plot": {},
            "dbg_d1_resistance_plot": {},
            "dbg_d1_support_level": {},
            "dbg_d1_resistance_level": {},
            "dbg_h1_support_plot": {},
            "dbg_h1_resistance_plot": {},
        },
        "subplots": {
            "Daily Context": {
                "dbg_long_daily_context_score": {},
                "dbg_short_daily_context_score": {},
                "dbg_d1_trend_bias": {},
                "dbg_d1_support_respect": {},
                "dbg_d1_resistance_respect": {},
            },
            "Hourly Execution": {
                "dbg_long_hourly_score": {},
                "dbg_short_hourly_score": {},
                "dbg_long_volume_score": {},
                "dbg_short_volume_score": {},
            },
            "Entries": {
                "dbg_long_support_reclaim": {},
                "dbg_short_resistance_reject": {},
                "dbg_long_resistance_breakout": {},
                "dbg_short_support_breakdown": {},
                "dbg_long_trend_pullback": {},
                "dbg_short_trend_pullback": {},
            },
        },
    }

    def informative_pairs(self) -> list[tuple[str, str]]:
        if not self.dp:
            return []
        return [(pair, self.informative_timeframe) for pair in self.dp.current_whitelist()]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self._add_hourly_evidence(dataframe)
        if not self.dp:
            return dataframe
        informative = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.informative_timeframe).copy()
        informative = self._add_daily_evidence(informative)
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe, ffill=True)
        dataframe = self._normalize_merged_booleans(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""

        entry_params = self._active_entry_params()
        mode = str(entry_params["mode"])
        s1d = int(entry_params["daily_pivot_strength"])
        s1h = int(entry_params["hourly_pivot_strength"])
        h = int(entry_params["projection_horizon"])
        w = int(entry_params["event_window"])
        zone_pct = float(entry_params["entry_zone_pct"])
        breakout_buffer = float(entry_params["breakout_buffer_pct"])

        close = self._num(dataframe, "close")
        high = self._num(dataframe, "high")
        low = self._num(dataframe, "low")
        volume_present = self._num(dataframe, "volume").gt(0.0)

        d1_support, d1_resistance = self._daily_levels(dataframe, s1d, h, entry_params)
        d1_support_plot = self._series(dataframe, self._d1(f"d1tl_support_proj_plot_{h}_{s1d}"))
        d1_resistance_plot = self._series(dataframe, self._d1(f"d1tl_resistance_proj_plot_{h}_{s1d}"))
        d1_trend_bias = self._series(dataframe, self._d1(f"d1tl_trend_bias_{s1d}"), 0.0)
        d1_support_respect = self._series(dataframe, self._d1(f"d1tl_support_respect_ratio_{w}_{s1d}"), 0.0)
        d1_resistance_respect = self._series(dataframe, self._d1(f"d1tl_resistance_respect_ratio_{w}_{s1d}"), 0.0)
        d1_support_violations = self._series(dataframe, self._d1(f"d1tl_support_violation_count_{w}_{s1d}"), 0.0)
        d1_resistance_violations = self._series(dataframe, self._d1(f"d1tl_resistance_violation_count_{w}_{s1d}"), 0.0)
        d1_long_target = self._series(dataframe, self._d1(f"d1tl_long_target_distance_pct_{h}_{s1d}"), 0.0)
        d1_short_target = self._series(dataframe, self._d1(f"d1tl_short_target_distance_pct_{h}_{s1d}"), 0.0)

        near_support = self._touch_level(high, low, d1_support, zone_pct)
        near_resistance = self._touch_level(high, low, d1_resistance, zone_pct)
        long_breaks_resistance = close.gt(d1_resistance * (1.0 + breakout_buffer))
        short_breaks_support = close.lt(d1_support * (1.0 - breakout_buffer))

        support_confluence = self._support_confluence(dataframe, d1_support, s1d, h, entry_params)
        resistance_confluence = self._resistance_confluence(dataframe, d1_resistance, s1d, h, entry_params)
        daily_long_score = self._score_sum(
            d1_support.notna(),
            near_support | long_breaks_resistance,
            d1_trend_bias.ge(0.0),
            d1_support_respect.ge(float(entry_params["min_line_respect_ratio"])),
            d1_support_violations.le(float(entry_params["max_line_violation_count"])),
            d1_long_target.ge(float(entry_params["min_target_distance_pct"])),
            support_confluence,
        )
        daily_short_score = self._score_sum(
            d1_resistance.notna(),
            near_resistance | short_breaks_support,
            d1_trend_bias.le(0.0),
            d1_resistance_respect.ge(float(entry_params["min_line_respect_ratio"])),
            d1_resistance_violations.le(float(entry_params["max_line_violation_count"])),
            d1_short_target.ge(float(entry_params["min_target_distance_pct"])),
            resistance_confluence,
        )
        daily_long_ok, daily_short_ok = self._daily_context_ok(daily_long_score, daily_short_score, d1_trend_bias, entry_params)

        h1_long_score, h1_short_score = self._hourly_execution_scores(
            dataframe,
            s1h,
            h,
            w,
            d1_support,
            d1_resistance,
            zone_pct,
            breakout_buffer,
            entry_params,
        )
        vol_long_score, vol_short_score = self._volume_scores(dataframe, entry_params)

        min_hourly = int(entry_params["min_hourly_execution_score"])
        min_volume = int(entry_params["min_volume_score"])
        long_evidence_ok = daily_long_ok & h1_long_score.ge(min_hourly) & vol_long_score.ge(min_volume) & volume_present
        short_evidence_ok = daily_short_ok & h1_short_score.ge(min_hourly) & vol_short_score.ge(min_volume) & volume_present

        h1_support_reclaim = self._bool(dataframe, f"h1tl_support_reclaim_{s1h}")
        h1_resistance_reject = self._bool(dataframe, f"h1tl_resistance_reject_{s1h}")
        h1_resistance_breakout = self._bool(dataframe, f"h1tl_resistance_breakout_{h}_{s1h}") | self._bool(dataframe, f"h1pa_ms_bullish_break_{s1h}")
        h1_support_breakdown = self._bool(dataframe, f"h1tl_support_breakdown_{h}_{s1h}") | self._bool(dataframe, f"h1pa_ms_bearish_break_{s1h}")
        h1_bull_pullback = near_support & self._bool(dataframe, f"h1pa_ms_higher_low_{s1h}") & self._series(dataframe, f"h1tl_trend_bias_{s1h}", 0.0).ge(0.0)
        h1_bear_pullback = near_resistance & self._bool(dataframe, f"h1pa_ms_lower_high_{s1h}") & self._series(dataframe, f"h1tl_trend_bias_{s1h}", 0.0).le(0.0)

        long_support_reclaim = long_evidence_ok & near_support & h1_support_reclaim & close.gt(d1_support)
        short_resistance_reject = short_evidence_ok & near_resistance & h1_resistance_reject & close.lt(d1_resistance)
        long_resistance_breakout = long_evidence_ok & long_breaks_resistance & h1_resistance_breakout
        short_support_breakdown = short_evidence_ok & short_breaks_support & h1_support_breakdown
        long_trend_pullback = long_evidence_ok & h1_bull_pullback
        short_trend_pullback = short_evidence_ok & h1_bear_pullback

        false = pd.Series(False, index=dataframe.index, dtype="bool")
        long_condition = false.copy()
        short_condition = false.copy()
        use_support_reclaim = mode in {"support_reclaim", "all"}
        use_resistance_reject = mode in {"resistance_reject", "all"}
        use_resistance_breakout = mode in {"resistance_breakout", "all"}
        use_support_breakdown = mode in {"support_breakdown", "all"}
        use_trend_pullback = mode in {"trend_pullback", "all"}
        if use_support_reclaim:
            long_condition |= long_support_reclaim
        if use_resistance_reject:
            short_condition |= short_resistance_reject
        if use_resistance_breakout:
            long_condition |= long_resistance_breakout
        if use_support_breakdown:
            short_condition |= short_support_breakdown
        if use_trend_pullback:
            long_condition |= long_trend_pullback
            short_condition |= short_trend_pullback
        short_condition &= ~long_condition

        self._write_debug_columns(
            dataframe,
            d1_support=d1_support,
            d1_resistance=d1_resistance,
            d1_support_plot=d1_support_plot,
            d1_resistance_plot=d1_resistance_plot,
            d1_trend_bias=d1_trend_bias,
            d1_support_respect=d1_support_respect,
            d1_resistance_respect=d1_resistance_respect,
            daily_long_score=daily_long_score,
            daily_short_score=daily_short_score,
            h1_long_score=h1_long_score,
            h1_short_score=h1_short_score,
            vol_long_score=vol_long_score,
            vol_short_score=vol_short_score,
            long_support_reclaim=long_support_reclaim,
            short_resistance_reject=short_resistance_reject,
            long_resistance_breakout=long_resistance_breakout,
            short_support_breakdown=short_support_breakdown,
            long_trend_pullback=long_trend_pullback,
            short_trend_pullback=short_trend_pullback,
            s1h=s1h,
            h=h,
        )

        dataframe.loc[long_condition.fillna(False), "enter_long"] = 1
        dataframe.loc[short_condition.fillna(False), "enter_short"] = 1
        def set_entry_tag(mask: Series, tag: str) -> None:
            tagged = mask.fillna(False) & dataframe["enter_tag"].eq("")
            dataframe.loc[tagged, "enter_tag"] = tag

        if use_support_reclaim:
            set_entry_tag(long_support_reclaim & long_condition, "long_support_reclaim")
        if use_resistance_breakout:
            set_entry_tag(long_resistance_breakout & long_condition, "long_resistance_breakout")
        if use_trend_pullback:
            set_entry_tag(long_trend_pullback & long_condition, "long_trend_pullback")
        if use_resistance_reject:
            set_entry_tag(short_resistance_reject & short_condition, "short_resistance_reject")
        if use_support_breakdown:
            set_entry_tag(short_support_breakdown & short_condition, "short_support_breakdown")
        if use_trend_pullback:
            set_entry_tag(short_trend_pullback & short_condition, "short_trend_pullback")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = ""

        entry_params = self._active_entry_params()
        s = int(entry_params["daily_pivot_strength"])
        h = int(entry_params["projection_horizon"])
        support, resistance = self._daily_levels(dataframe, s, h, entry_params)
        close = self._num(dataframe, "close")
        buffer = float(self.exit_buffer_pct.value)
        mode = str(self.exit_mode.value)
        use_target = mode in {"opposite_daily_level", "both"}
        use_break = mode in {"daily_invalidation", "both"}

        long_target = close.ge(resistance * (1.0 - buffer))
        short_target = close.le(support * (1.0 + buffer))
        long_invalid = close.lt(support * (1.0 - buffer))
        short_invalid = close.gt(resistance * (1.0 + buffer))
        exit_long = (long_target if use_target else False) | (long_invalid if use_break else False)
        exit_short = (short_target if use_target else False) | (short_invalid if use_break else False)

        dataframe["dbg_exit_long_target"] = self._bool_int(long_target)
        dataframe["dbg_exit_short_target"] = self._bool_int(short_target)
        dataframe["dbg_exit_long_invalid"] = self._bool_int(long_invalid)
        dataframe["dbg_exit_short_invalid"] = self._bool_int(short_invalid)
        dataframe.loc[pd.Series(exit_long, index=dataframe.index).fillna(False), "exit_long"] = 1
        dataframe.loc[pd.Series(exit_short, index=dataframe.index).fillna(False), "exit_short"] = 1
        dataframe.loc[long_target.fillna(False), "exit_tag"] = "long_opposite_daily_level"
        dataframe.loc[short_target.fillna(False), "exit_tag"] = "short_opposite_daily_level"
        dataframe.loc[long_invalid.fillna(False), "exit_tag"] = "long_daily_invalidated"
        dataframe.loc[short_invalid.fillna(False), "exit_tag"] = "short_daily_invalidated"
        return dataframe

    def _add_daily_evidence(self, dataframe: DataFrame) -> DataFrame:
        dataframe = add_pivot_structure(
            dataframe,
            PivotStructureConfig(
                strength=5,
                strengths=tuple(self.PIVOT_STRENGTH_CHOICES),
                atr_period=14,
                min_prominence_atr=0.35,
                zone_atr_mult=0.35,
                zone_pct=0.003,
                min_channel_width_pct=0.003,
                min_target_distance_pct=0.002,
                breakout_buffer_pct=0.001,
                prefix="d1pa",
            ),
        )
        return add_trendline_projection(
            dataframe,
            TrendlineProjectionConfig(
                strength=5,
                strengths=tuple(self.PIVOT_STRENGTH_CHOICES),
                horizons=tuple(self.PROJECTION_HORIZON_CHOICES),
                event_windows=tuple(self.EVENT_WINDOW_CHOICES),
                pivot_prefix="d1pa",
                output_prefix="d1tl",
                missing_pivot_mode="raise",
                near_zone_atr_mult=0.35,
                near_zone_pct=0.004,
                breakout_buffer_pct=0.001,
                min_anchor_span_bars=3,
                max_anchor_age_bars=240,
                min_anchor_prominence_atr=0.15,
                max_line_slope_pct_per_bar=0.12,
                max_projection_distance_pct=1.5,
                plot_break_on_line_change_pct=0.06,
            ),
        )

    def _add_hourly_evidence(self, dataframe: DataFrame) -> DataFrame:
        dataframe = add_pivot_structure(
            dataframe,
            PivotStructureConfig(
                strength=5,
                strengths=tuple(self.PIVOT_STRENGTH_CHOICES),
                atr_period=14,
                min_prominence_atr=0.30,
                zone_atr_mult=0.35,
                zone_pct=0.003,
                min_channel_width_pct=0.002,
                min_target_distance_pct=0.001,
                breakout_buffer_pct=0.001,
                prefix="h1pa",
            ),
        )
        dataframe = add_trendline_projection(
            dataframe,
            TrendlineProjectionConfig(
                strength=5,
                strengths=tuple(self.PIVOT_STRENGTH_CHOICES),
                horizons=tuple(self.PROJECTION_HORIZON_CHOICES),
                event_windows=tuple(self.EVENT_WINDOW_CHOICES),
                pivot_prefix="h1pa",
                output_prefix="h1tl",
                missing_pivot_mode="raise",
                near_zone_atr_mult=0.35,
                near_zone_pct=0.004,
                breakout_buffer_pct=0.001,
                min_anchor_span_bars=4,
                max_anchor_age_bars=180,
                min_anchor_prominence_atr=0.10,
                max_line_slope_pct_per_bar=0.08,
                max_projection_distance_pct=1.5,
                plot_break_on_line_change_pct=0.06,
            ),
        )
        return add_complex_volume_indicators(
            dataframe,
            ComplexVolumeConfig(short_window=12, medium_window=48, long_window=96, divergence_window=48, sweep_window=24, vwap_window=48, prefix="vol"),
        )

    def _active_entry_params(self) -> dict[str, Any]:
        mode = str(self.entry_mode.value)
        params: dict[str, Any] = {"mode": mode}
        for name in self.ENTRY_PARAM_SPECS:
            params[name] = getattr(self, f"{mode}_{name}").value
        return params

    def _daily_levels(self, dataframe: DataFrame, strength: int, horizon: int, entry_params: dict[str, Any]) -> tuple[Series, Series]:
        close = self._num(dataframe, "close")
        source = str(entry_params["daily_level_source"])
        trend_sup = self._series(dataframe, self._d1(f"d1tl_support_proj_{horizon}_{strength}"))
        trend_res = self._series(dataframe, self._d1(f"d1tl_resistance_proj_{horizon}_{strength}"))
        pivot_sup = self._series(dataframe, self._d1(f"d1pa_last_pivot_low_{strength}"))
        pivot_res = self._series(dataframe, self._d1(f"d1pa_last_pivot_high_{strength}"))
        active_sup = self._series(dataframe, self._d1(f"d1pa_ms_active_swing_low_{strength}"))
        active_res = self._series(dataframe, self._d1(f"d1pa_ms_active_swing_high_{strength}"))
        if source == "trendline":
            return trend_sup, trend_res
        if source == "pivot":
            return pivot_sup, pivot_res
        if source == "active_swing":
            return active_sup, active_res
        return self._nearest_level(close, [trend_sup, pivot_sup, active_sup]), self._nearest_level(close, [trend_res, pivot_res, active_res])

    def _support_confluence(self, dataframe: DataFrame, selected_support: Series, strength: int, horizon: int, entry_params: dict[str, Any]) -> Series:
        close = self._num(dataframe, "close")
        candidates = [
            self._series(dataframe, self._d1(f"d1tl_support_proj_{horizon}_{strength}")),
            self._series(dataframe, self._d1(f"d1pa_last_pivot_low_{strength}")),
            self._series(dataframe, self._d1(f"d1pa_ms_active_swing_low_{strength}")),
        ]
        return self._has_confluence(selected_support, candidates, close, entry_params)

    def _resistance_confluence(self, dataframe: DataFrame, selected_resistance: Series, strength: int, horizon: int, entry_params: dict[str, Any]) -> Series:
        close = self._num(dataframe, "close")
        candidates = [
            self._series(dataframe, self._d1(f"d1tl_resistance_proj_{horizon}_{strength}")),
            self._series(dataframe, self._d1(f"d1pa_last_pivot_high_{strength}")),
            self._series(dataframe, self._d1(f"d1pa_ms_active_swing_high_{strength}")),
        ]
        return self._has_confluence(selected_resistance, candidates, close, entry_params)

    def _has_confluence(self, selected: Series, candidates: list[Series], close: Series, entry_params: dict[str, Any]) -> Series:
        limit = float(entry_params["level_confluence_max_pct"])
        if limit >= 100.0:
            return pd.Series(True, index=selected.index, dtype="bool")
        checks = [((selected - c).abs() / close).le(limit) & selected.notna() & c.notna() for c in candidates]
        result = pd.Series(False, index=selected.index, dtype="bool")
        for check in checks:
            result |= check.fillna(False)
        return result

    def _daily_context_ok(self, long_score: Series, short_score: Series, trend_bias: Series, entry_params: dict[str, Any]) -> tuple[Series, Series]:
        min_score = int(entry_params["min_daily_context_score"])
        mode = str(entry_params["daily_context_mode"])
        score_long = long_score.ge(min_score)
        score_short = short_score.ge(min_score)
        trend_long = trend_bias.ge(0.0)
        trend_short = trend_bias.le(0.0)
        if mode == "trend_required":
            return score_long & trend_long, score_short & trend_short
        if mode == "trend_or_level":
            return score_long | trend_long, score_short | trend_short
        return score_long, score_short

    def _hourly_execution_scores(
        self,
        dataframe: DataFrame,
        strength: int,
        horizon: int,
        window: int,
        d1_support: Series,
        d1_resistance: Series,
        zone_pct: float,
        breakout_buffer: float,
        entry_params: dict[str, Any],
    ) -> tuple[Series, Series]:
        close = self._num(dataframe, "close")
        high = self._num(dataframe, "high")
        low = self._num(dataframe, "low")
        close_loc = self._series(dataframe, "vol_close_location", 0.0)
        h1_support_respect = self._series(dataframe, f"h1tl_support_respect_ratio_{window}_{strength}", 0.0)
        h1_resistance_respect = self._series(dataframe, f"h1tl_resistance_respect_ratio_{window}_{strength}", 0.0)
        h1_support_violations = self._series(dataframe, f"h1tl_support_violation_count_{window}_{strength}", 0.0)
        h1_resistance_violations = self._series(dataframe, f"h1tl_resistance_violation_count_{window}_{strength}", 0.0)
        near_support = self._touch_level(high, low, d1_support, zone_pct)
        near_resistance = self._touch_level(high, low, d1_resistance, zone_pct)
        long_break = close.gt(d1_resistance * (1.0 + breakout_buffer))
        short_break = close.lt(d1_support * (1.0 - breakout_buffer))
        long_score = self._score_sum(
            near_support | long_break,
            self._bool(dataframe, f"h1tl_support_reclaim_{strength}") | self._bool(dataframe, f"h1tl_resistance_breakout_{horizon}_{strength}"),
            self._bool(dataframe, f"h1pa_ms_bullish_break_{strength}") | self._bool(dataframe, f"h1pa_ms_bullish_bos_{strength}") | self._bool(dataframe, f"h1pa_ms_bullish_choch_{strength}"),
            h1_support_respect.ge(float(entry_params["min_line_respect_ratio"])),
            h1_support_violations.le(float(entry_params["max_line_violation_count"])),
            close_loc.ge(0.15),
        )
        short_score = self._score_sum(
            near_resistance | short_break,
            self._bool(dataframe, f"h1tl_resistance_reject_{strength}") | self._bool(dataframe, f"h1tl_support_breakdown_{horizon}_{strength}"),
            self._bool(dataframe, f"h1pa_ms_bearish_break_{strength}") | self._bool(dataframe, f"h1pa_ms_bearish_bos_{strength}") | self._bool(dataframe, f"h1pa_ms_bearish_choch_{strength}"),
            h1_resistance_respect.ge(float(entry_params["min_line_respect_ratio"])),
            h1_resistance_violations.le(float(entry_params["max_line_violation_count"])),
            close_loc.le(-0.15),
        )
        return long_score, short_score

    def _volume_scores(self, dataframe: DataFrame, entry_params: dict[str, Any]) -> tuple[Series, Series]:
        rvol = self._series(dataframe, "vol_rvol", 0.0)
        delta_z = self._series(dataframe, "vol_delta_zscore", 0.0)
        rvol_ok = rvol.ge(float(entry_params["volume_rvol_min"]))
        delta_min = float(entry_params["volume_delta_abs_min"])
        long_score = self._score_sum(
            rvol_ok,
            delta_z.ge(delta_min),
            self._bool(dataframe, "vol_cvd_trend_confirm_long"),
            self._bool(dataframe, "vol_liq_sweep_low_reclaim_long") | self._bool(dataframe, "vol_liq_stoprun_long"),
            self._bool(dataframe, "vol_avwap_reclaim_long") | self._bool(dataframe, "vol_avwap_mean_reversion_long"),
            self._bool(dataframe, "vol_evr_bull_absorption") | self._bool(dataframe, "vol_vol_breakout_confirm_long"),
        )
        short_score = self._score_sum(
            rvol_ok,
            delta_z.le(-delta_min),
            self._bool(dataframe, "vol_cvd_trend_confirm_short"),
            self._bool(dataframe, "vol_liq_sweep_high_reject_short") | self._bool(dataframe, "vol_liq_stoprun_short"),
            self._bool(dataframe, "vol_avwap_reject_short") | self._bool(dataframe, "vol_avwap_mean_reversion_short"),
            self._bool(dataframe, "vol_evr_bear_absorption") | self._bool(dataframe, "vol_vol_breakout_confirm_short"),
        )
        return long_score, short_score

    def _write_debug_columns(self, dataframe: DataFrame, **values: Any) -> None:
        dataframe["dbg_d1_support_level"] = values["d1_support"]
        dataframe["dbg_d1_resistance_level"] = values["d1_resistance"]
        dataframe["dbg_d1_support_plot"] = values["d1_support_plot"]
        dataframe["dbg_d1_resistance_plot"] = values["d1_resistance_plot"]
        dataframe["dbg_h1_support_plot"] = self._series(dataframe, f"h1tl_support_proj_plot_{values['h']}_{values['s1h']}")
        dataframe["dbg_h1_resistance_plot"] = self._series(dataframe, f"h1tl_resistance_proj_plot_{values['h']}_{values['s1h']}")
        dataframe["dbg_d1_trend_bias"] = values["d1_trend_bias"]
        dataframe["dbg_d1_support_respect"] = values["d1_support_respect"]
        dataframe["dbg_d1_resistance_respect"] = values["d1_resistance_respect"]
        dataframe["dbg_long_daily_context_score"] = values["daily_long_score"]
        dataframe["dbg_short_daily_context_score"] = values["daily_short_score"]
        dataframe["dbg_long_hourly_score"] = values["h1_long_score"]
        dataframe["dbg_short_hourly_score"] = values["h1_short_score"]
        dataframe["dbg_long_volume_score"] = values["vol_long_score"]
        dataframe["dbg_short_volume_score"] = values["vol_short_score"]
        dataframe["dbg_long_support_reclaim"] = self._bool_int(values["long_support_reclaim"])
        dataframe["dbg_short_resistance_reject"] = self._bool_int(values["short_resistance_reject"])
        dataframe["dbg_long_resistance_breakout"] = self._bool_int(values["long_resistance_breakout"])
        dataframe["dbg_short_support_breakdown"] = self._bool_int(values["short_support_breakdown"])
        dataframe["dbg_long_trend_pullback"] = self._bool_int(values["long_trend_pullback"])
        dataframe["dbg_short_trend_pullback"] = self._bool_int(values["short_trend_pullback"])

    @staticmethod
    def _d1(name: str) -> str:
        return f"{name}_1d"

    @staticmethod
    def _score_sum(*conditions: Series) -> Series:
        result: Series | None = None
        for condition in conditions:
            clean = pd.Series(condition, index=condition.index).fillna(False).astype("int8")
            result = clean.astype("int16") if result is None else result + clean.astype("int16")
        if result is None:
            raise ValueError("_score_sum requires at least one condition")
        return result.astype("int16")

    @staticmethod
    def _touch_level(high: Series, low: Series, level: Series, zone_pct: float) -> Series:
        """True when the candle high/low range overlaps a percentage zone around level."""
        level_clean = pd.to_numeric(level, errors="coerce")
        high_clean = pd.to_numeric(high, errors="coerce")
        low_clean = pd.to_numeric(low, errors="coerce")
        zone = float(zone_pct)
        return (
            level_clean.notna()
            & high_clean.ge(level_clean * (1.0 - zone))
            & low_clean.le(level_clean * (1.0 + zone))
        )

    @staticmethod
    def _nearest_level(close: Series, candidates: list[Series]) -> Series:
        """Return the nearest non-null candidate level to close for each row."""
        if not candidates:
            return pd.Series(np.nan, index=close.index, dtype="float64")

        frame = pd.concat([pd.to_numeric(candidate, errors="coerce") for candidate in candidates], axis=1)
        if frame.empty:
            return pd.Series(np.nan, index=close.index, dtype="float64")

        close_clean = pd.to_numeric(close, errors="coerce").replace(0.0, np.nan)
        dist = frame.sub(close_clean, axis=0).abs().div(close_clean.abs(), axis=0)
        valid_rows = dist.notna().any(axis=1).to_numpy()

        out_values = np.full(len(frame), np.nan, dtype="float64")
        if valid_rows.any():
            dist_values = dist.fillna(np.inf).to_numpy(dtype="float64", copy=False)
            frame_values = frame.to_numpy(dtype="float64", copy=False)
            row_positions = np.arange(len(frame_values))
            best_positions = np.argmin(dist_values, axis=1)
            out_values[valid_rows] = frame_values[row_positions[valid_rows], best_positions[valid_rows]]

        return pd.Series(out_values, index=close.index, dtype="float64")


    @staticmethod
    def _num(dataframe: DataFrame, column: str) -> Series:
        if column not in dataframe.columns:
            return pd.Series(np.nan, index=dataframe.index, dtype="float64")
        return pd.to_numeric(dataframe[column], errors="coerce")

    @staticmethod
    def _series(dataframe: DataFrame, column: str, default: float = float("nan")) -> Series:
        if column in dataframe.columns:
            return pd.to_numeric(dataframe[column], errors="coerce")
        return pd.Series(default, index=dataframe.index, dtype="float64")

    @staticmethod
    def _bool(dataframe: DataFrame, column: str) -> Series:
        if column not in dataframe.columns:
            return pd.Series(False, index=dataframe.index, dtype="bool")
        series = pd.Series(dataframe[column], index=dataframe.index)
        if pd.api.types.is_bool_dtype(series):
            return series.fillna(False).astype(bool)
        return series.astype("boolean").fillna(False).astype(bool)

    @staticmethod
    def _bool_int(series: Series) -> Series:
        if pd.api.types.is_bool_dtype(series):
            return series.fillna(False).astype("int8")
        return pd.Series(series, index=series.index).astype("boolean").fillna(False).astype("int8")

    @staticmethod
    def _normalize_merged_booleans(dataframe: DataFrame) -> DataFrame:
        markers = ("_valid_", "_touch_", "_reject_", "_reclaim_", "_break_", "_breakout_", "_breakdown_", "_violation_", "_near_")
        frame = dataframe.copy()
        for column in frame.columns:
            name = str(column)
            if not name.endswith("_1d") or not any(marker in name for marker in markers):
                continue
            series = frame[column]
            if pd.api.types.is_bool_dtype(series):
                frame[column] = series.fillna(False).astype(bool)
            else:
                non_null = series.dropna()
                if non_null.empty or set(non_null.unique()).issubset({True, False, 0, 1}):
                    frame[column] = series.astype("boolean").fillna(False).astype(bool)
        return frame


for _entry_mode in PivotTrendlineMTFResearchStrategy.ENTRY_MODE_CHOICES:
    for _param_name, (_choices, _default, _family) in PivotTrendlineMTFResearchStrategy.ENTRY_PARAM_SPECS.items():
        setattr(
            PivotTrendlineMTFResearchStrategy,
            f"{_entry_mode}_{_param_name}",
            tagged_parameter(
                CategoricalParameter(_choices, default=_default, space="buy", optimize=True, load=True),
                f"family:{_family}",
                f"mode:pivot_trendline_{_entry_mode}",
            ),
        )

del _entry_mode, _param_name, _choices, _default, _family

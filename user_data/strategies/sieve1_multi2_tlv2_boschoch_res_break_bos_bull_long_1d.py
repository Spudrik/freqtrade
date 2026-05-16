from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.strategy import BooleanParameter, CategoricalParameter, IStrategy
from user_data.Indicators.complex_volume_profile import add_volume_profile
from user_data.Indicators.complex_trendline_projection_v2 import add_trendline_projection_v2
from user_data.Indicators.pattern_bos_choch import add_bos_choch

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


def _cross_above(series: Series, level: Series) -> Series:
    return series.gt(level) & series.shift(1).le(level.shift(1))


def _cross_below(series: Series, level: Series) -> Series:
    return series.lt(level) & series.shift(1).ge(level.shift(1))

ENTRY_MODE = "entry_multi2_tlv2_boschoch_res_break_bos_bull_long_1d"
ENTRY_TAG = "multi2_tlv2_boschoch_res_break_bos_bull_long_1d"
SIDE = "long"
TIMEFRAME = "1d"
TLV2_KIND = "resistance_breakout"
TLV2_LINE = "resistance"
MS_EVENT = "bos_bull"

class Sieve1Multi2Tlv2BoschochResBreakBosBullLong1d(IStrategy):
    """Sieve1 multi2 confluence probe."""

    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 220
    process_only_new_candles = True
    can_short = False

    minimal_roi = {"0": _pct_env(ENTRY_SIEVE_TAKE_PROFIT_ENV, 0.02)}
    stoploss = -_pct_env(ENTRY_SIEVE_STOPLOSS_ENV, 0.02)
    use_exit_signal = False
    use_custom_stoploss = False
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    use_volume_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=True, load=True))
    volume_guard_window = tagged_parameter(CategoricalParameter([12, 24, 48], default=24, space="buy", optimize=True, load=True))
    volume_ratio_min = tagged_parameter(CategoricalParameter([0.8, 1.0, 1.3, 1.6], default=1.0, space="buy", optimize=True, load=True))
    use_pressure_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=True, load=True))
    pressure_window = tagged_parameter(CategoricalParameter([12, 24, 48], default=24, space="buy", optimize=True, load=True))
    pressure_min = tagged_parameter(CategoricalParameter([0.05, 0.1, 0.15, 0.2, 0.35], default=0.15, space="buy", optimize=True, load=True))
    use_accumulation_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=False, load=True))
    use_body_direction_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=False, load=True))
    use_close_direction_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=False, load=True))

    pivot_strength = tagged_parameter(CategoricalParameter([2, 3, 4], default=2, space="buy", optimize=False, load=True))
    min_line_score = tagged_parameter(CategoricalParameter([0.4, 0.5, 0.6], default=0.5, space="buy", optimize=True, load=True))
    min_active_bars = tagged_parameter(CategoricalParameter([4, 8, 16], default=8, space="buy", optimize=False, load=True))
    max_distance_atr = tagged_parameter(CategoricalParameter([3.0, 6.0, 10.0], default=6.0, space="buy", optimize=True, load=True))
    proximity_rank_weight = tagged_parameter(CategoricalParameter([0.0, 0.05, 0.1], default=0.05, space="buy", optimize=False, load=True))
    line_buffer_pct = tagged_parameter(CategoricalParameter([0.0, 0.003, 0.006, 0.012], default=0.003, space="buy", optimize=True, load=True))
    line_slope_min_pct = tagged_parameter(CategoricalParameter([0.0, 0.0005, 0.0015], default=0.0005, space="buy", optimize=True, load=True))

    strength = tagged_parameter(CategoricalParameter([2, 3, 5], default=3, space="buy", optimize=False, load=True))
    min_prominence_atr = tagged_parameter(CategoricalParameter([0.25, 0.35, 0.6], default=0.35, space="buy", optimize=False, load=True))
    min_pivot_spacing_bars = tagged_parameter(CategoricalParameter([1, 2, 4], default=2, space="buy", optimize=False, load=True))
    max_pivot_age_bars = tagged_parameter(CategoricalParameter([48, 96, 160], default=96, space="buy", optimize=False, load=True))
    breakout_buffer_atr = tagged_parameter(CategoricalParameter([0.0, 0.15, 0.3], default=0.15, space="buy", optimize=True, load=True))
    use_state_guard = tagged_parameter(BooleanParameter(default=False, space="buy", optimize=True, load=True))

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype="bool")
        close = _num(dataframe, "close")
        if bool(self.use_volume_guard.value):
            volume = _num(dataframe, "volume").clip(lower=0.0)
            window = int(self.volume_guard_window.value)
            baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
            guard &= volume.ge(baseline.mul(float(self.volume_ratio_min.value)))
        if bool(self.use_pressure_guard.value) or bool(self.use_accumulation_guard.value):
            open_ = _num(dataframe, "open")
            high = _num(dataframe, "high")
            low = _num(dataframe, "low")
            volume = _num(dataframe, "volume").clip(lower=0.0).fillna(0.0)
            candle_range = (high - low).replace(0.0, np.nan)
            body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
            close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
            pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
            directional_volume = (pressure * volume).fillna(0.0)
        if bool(self.use_pressure_guard.value):
            window = int(self.pressure_window.value)
            pressure_baseline = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
            pressure_ratio = directional_volume.rolling(window, min_periods=max(2, window // 3)).sum() / pressure_baseline
            guard &= pressure_ratio.le(-float(self.pressure_min.value)) if SIDE == "short" else pressure_ratio.ge(float(self.pressure_min.value))
        if bool(self.use_accumulation_guard.value):
            window = int(self.pressure_window.value)
            accumulation = directional_volume.rolling(window, min_periods=max(2, window // 3)).sum()
            guard &= accumulation.le(0.0) if SIDE == "short" else accumulation.ge(0.0)
        if bool(self.use_body_direction_guard.value):
            open_ = _num(dataframe, "open")
            guard &= close.lt(open_) if SIDE == "short" else close.gt(open_)
        if bool(self.use_close_direction_guard.value):
            guard &= close.lt(close.shift(1)) if SIDE == "short" else close.gt(close.shift(1))
        return guard.fillna(False)

    def _add_vp(self, dataframe: DataFrame) -> DataFrame:
        return add_volume_profile(
            dataframe,
            window=int(self.vp_window.value),
            bins=int(self.vp_bins.value),
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
            prefix="vp",
        )

    def _vp_confirm(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, "close")
        score_long = _num(dataframe, "vp_score_long")
        score_short = _num(dataframe, "vp_score_short")
        bull_context = _num(dataframe, "vp_context_score_bull")
        bear_context = _num(dataframe, "vp_context_score_bear")
        balance = _num(dataframe, "vp_context_score_balance")
        market = _num(dataframe, "vp_market_context")
        score_min = float(self.vp_score_min.value)
        context_min = float(self.vp_context_min.value)
        near = float(self.vp_level_buffer_pct.value)
        if VP_CONFIRM == "bull_context":
            return score_long.ge(score_min) & bull_context.ge(context_min) & bull_context.ge(bear_context) & market.ge(0)
        if VP_CONFIRM == "bear_context":
            return score_short.ge(score_min) & bear_context.ge(context_min) & bear_context.ge(bull_context) & market.le(0)
        if VP_CONFIRM == "val_reclaim":
            val = _num(dataframe, "vp_val", np.nan)
            return close.ge(val.mul(1.0 - near)) & (score_long.ge(score_min) | _bool(dataframe, "vp_entry_trigger_long") | balance.ge(context_min))
        if VP_CONFIRM == "vah_reject":
            vah = _num(dataframe, "vp_vah", np.nan)
            return close.le(vah.mul(1.0 + near)) & (score_short.ge(score_min) | _bool(dataframe, "vp_entry_trigger_short") | balance.ge(context_min))
        if VP_CONFIRM == "node_entry":
            return (_bool(dataframe, "vp_node_entry_short") | score_short.ge(score_min)) if SIDE == "short" else (_bool(dataframe, "vp_node_entry_long") | score_long.ge(score_min))
        return pd.Series(True, index=dataframe.index, dtype="bool")

    def _add_tlv2(self, dataframe: DataFrame) -> DataFrame:
        return add_trendline_projection_v2(
            dataframe,
            timeframe=self.timeframe,
            pivot_strength=int(self.pivot_strength.value),
            raw_line_output_count=1,
            min_output_line_score=float(self.min_line_score.value),
            min_output_active_bars=int(self.min_active_bars.value),
            max_active_line_distance_atr_mult=float(self.max_distance_atr.value),
            proximity_rank_weight=float(self.proximity_rank_weight.value),
            output_prefix="tlv2",
        )

    def _tlv2_trigger(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, "close")
        open_ = _num(dataframe, "open")
        high = _num(dataframe, "high")
        low = _num(dataframe, "low")
        buffer = float(self.line_buffer_pct.value)
        slope_min = float(self.line_slope_min_pct.value)
        if TLV2_LINE == "support":
            line = _num(dataframe, "tlv2_support_line_rank0", np.nan)
            score = _num(dataframe, "tlv2_support_score_rank0")
            distance = _num(dataframe, "tlv2_support_distance_atr_rank0", np.nan)
        else:
            line = _num(dataframe, "tlv2_resistance_line_rank0", np.nan)
            score = _num(dataframe, "tlv2_resistance_score_rank0")
            distance = _num(dataframe, "tlv2_resistance_distance_atr_rank0", np.nan)
        active = score.ge(float(self.min_line_score.value)) & distance.le(float(self.max_distance_atr.value))
        if TLV2_KIND == "support_reclaim":
            return active & low.le(line.mul(1.0 + buffer)) & close.ge(line.mul(1.0 - buffer)) & close.gt(open_)
        if TLV2_KIND == "support_bounce":
            return active & low.le(line.mul(1.0 + buffer)) & close.gt(line) & close.gt(open_)
        if TLV2_KIND == "rising_support_ride":
            return active & line.pct_change(fill_method=None).gt(slope_min) & close.gt(line)
        if TLV2_KIND == "resistance_breakout":
            return active & _cross_above(close, line.mul(1.0 + buffer))
        if TLV2_KIND == "resistance_reject":
            return active & high.ge(line.mul(1.0 - buffer)) & close.le(line.mul(1.0 + buffer)) & close.lt(open_)
        if TLV2_KIND == "resistance_proximity_reject":
            return active & high.ge(line.mul(1.0 - buffer)) & close.lt(line) & close.lt(open_)
        if TLV2_KIND == "falling_resistance_ride":
            return active & line.pct_change(fill_method=None).lt(-slope_min) & close.lt(line)
        if TLV2_KIND == "support_breakdown":
            return active & _cross_below(close, line.mul(1.0 - buffer))
        return pd.Series(False, index=dataframe.index, dtype="bool")

    def _add_bos_choch(self, dataframe: DataFrame) -> DataFrame:
        return add_bos_choch(
            dataframe,
            strength=int(self.strength.value),
            min_prominence_atr=float(self.min_prominence_atr.value),
            min_pivot_spacing_bars=int(self.min_pivot_spacing_bars.value),
            max_pivot_age_bars=int(self.max_pivot_age_bars.value),
            breakout_buffer_atr=float(self.breakout_buffer_atr.value),
            prefix="ms",
        )

    def _ms_confirm(self, dataframe: DataFrame) -> Series:
        if MS_EVENT == "bos_bull":
            condition = _bool(dataframe, "ms_bos_to_bull")
            if bool(self.use_state_guard.value):
                condition &= _num(dataframe, "ms_state").ge(0)
            return condition
        if MS_EVENT == "choch_bull":
            condition = _bool(dataframe, "ms_choch_to_bull")
            if bool(self.use_state_guard.value):
                condition &= _num(dataframe, "ms_state").ge(0)
            return condition
        if MS_EVENT == "bos_bear":
            condition = _bool(dataframe, "ms_bos_to_bear")
            if bool(self.use_state_guard.value):
                condition &= _num(dataframe, "ms_state").le(0)
            return condition
        if MS_EVENT == "choch_bear":
            condition = _bool(dataframe, "ms_choch_to_bear")
            if bool(self.use_state_guard.value):
                condition &= _num(dataframe, "ms_state").le(0)
            return condition
        return pd.Series(False, index=dataframe.index, dtype="bool")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe = self._add_tlv2(dataframe)
        dataframe = self._add_bos_choch(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        condition = self._tlv2_trigger(dataframe) & self._ms_confirm(dataframe)
        condition &= self._common_guards(dataframe)
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


apply_explicit_hyperopt_surface(Sieve1Multi2Tlv2BoschochResBreakBosBullLong1d)

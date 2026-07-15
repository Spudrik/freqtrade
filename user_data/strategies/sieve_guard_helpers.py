from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from user_data.Indicators.complex_volume_profile import add_volume_profile


SIEVE2_VP_GUARD_MODES = [
    "score_or_context",
    "node_confirm",
    "value_area_confirm",
    "breakout_acceptance",
    "rejection_confirm",
    "poc_hvn_reject",
    "prior_level_confirm",
]
SIEVE2_MARKET_GUARD_MODES = [
    "pressure_or_trend",
    "pressure_and_trend",
    "directional_pressure",
    "trend_state",
    "avoid_adverse_pressure",
    "avoid_chop",
]


def add_sieve2_guard_indicators(strategy: Any, dataframe: DataFrame, metadata: dict | None) -> DataFrame:
    frame = dataframe.copy()
    frame = _add_market_state(frame, int(_param(strategy, "sieve2_market_window", 24)))
    frame = add_volume_profile(
        frame,
        window=int(_param(strategy, "sieve2_vp_window", 96)),
        bins=int(_param(strategy, "sieve2_vp_bins", 36)),
        value_area_pct=0.70,
        price_source="hlc3",
        smooth_bins=3,
        pressure_delta_min=0.05,
        node_near_pct=0.01,
        prefix="s2vp",
    )
    return frame


def apply_sieve2_optional_guards(strategy: Any, dataframe: DataFrame, condition: Series, side: str) -> Series:
    guarded = pd.Series(condition, index=dataframe.index).fillna(False).astype(bool)
    normalized_side = "short" if str(side or "").lower() == "short" else "long"
    always_on = _enabled(getattr(strategy, "SIEVE2_ALWAYS_ON_GUARDS", False))
    if always_on or _enabled(_param(strategy, "use_sieve2_vp_guard", False)):
        guarded &= _vp_guard(
            dataframe,
            normalized_side,
            str(_param(strategy, "sieve2_vp_guard_mode", "score_or_context")),
            float(_param(strategy, "sieve2_vp_score_min", 0.25)),
            float(_param(strategy, "sieve2_vp_context_min", 0.28)),
        )
    if always_on or _enabled(_param(strategy, "use_sieve2_market_guard", False)):
        guarded &= _market_guard(
            dataframe,
            normalized_side,
            str(_param(strategy, "sieve2_market_guard_mode", "pressure_or_trend")),
            float(_param(strategy, "sieve2_market_pressure_min", 0.07)),
            float(_param(strategy, "sieve2_market_trend_min", 0.25)),
            float(_param(strategy, "sieve2_rs_score_min", 0.45)),
        )
    return guarded.fillna(False)


def _add_market_state(frame: DataFrame, window: int) -> DataFrame:
    close = _num(frame, "close")
    open_ = _num(frame, "open")
    high = _num(frame, "high")
    low = _num(frame, "low")
    volume = _num(frame, "volume").clip(lower=0.0).fillna(0.0)
    candle_range = (high - low).replace(0.0, np.nan)
    body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
    close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
    pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
    directional_volume = (pressure * volume).fillna(0.0)
    baseline = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
    returns = close.pct_change(fill_method=None)
    window_return = close.pct_change(window, fill_method=None)
    volatility = returns.rolling(window, min_periods=max(3, window // 3)).std() * np.sqrt(float(window))
    frame["s2m_pressure_ratio"] = directional_volume.rolling(window, min_periods=max(2, window // 3)).sum() / baseline
    frame["s2m_trend_z"] = window_return / volatility.replace(0.0, np.nan)
    frame["s2m_close_location"] = close_location
    return frame


def _vp_guard(frame: DataFrame, side: str, mode: str, score_min: float, context_min: float) -> Series:
    close = _num(frame, "close")
    high = _num(frame, "high")
    low = _num(frame, "low")
    score = _num(frame, f"s2vp_score_{side}")
    other = _num(frame, "s2vp_score_short" if side == "long" else "s2vp_score_long")
    context = _num(frame, "s2vp_context_score_bull" if side == "long" else "s2vp_context_score_bear")
    in_value = _bool(frame, "s2vp_in_value_area")
    above_value = _bool(frame, "s2vp_above_value_area")
    below_value = _bool(frame, "s2vp_below_value_area")
    node_entry = _bool(frame, f"s2vp_node_entry_{side}")
    node_hold = _bool(frame, f"s2vp_node_hold_{side}")
    prior_vah = _num(frame, "s2vp_prior_vah", np.nan)
    prior_val = _num(frame, "s2vp_prior_val", np.nan)
    base_score = score.ge(score_min) & score.gt(other)
    base_context = context.ge(context_min)
    if mode == "node_confirm":
        return (node_entry | node_hold | (base_score & base_context)).fillna(False)
    if mode == "value_area_confirm":
        value_side = above_value if side == "long" else below_value
        return (value_side | (in_value & base_context)).fillna(False)
    if mode == "breakout_acceptance":
        accepted_beyond_value = close.gt(prior_vah) if side == "long" else close.lt(prior_val)
        return (accepted_beyond_value & (base_score | base_context | node_entry | node_hold)).fillna(False)
    if mode == "rejection_confirm":
        rejection = (
            (low.le(prior_val) & close.gt(prior_val) & close.le(prior_vah))
            if side == "long"
            else (high.ge(prior_vah) & close.lt(prior_vah) & close.ge(prior_val))
        )
        return (rejection & (base_score | base_context | node_entry | node_hold)).fillna(False)
    if mode == "poc_hvn_reject":
        return (node_entry | (base_score & base_context)).fillna(False)
    if mode == "prior_level_confirm":
        level_confirm = close.gt(prior_vah) if side == "long" else close.lt(prior_val)
        return (level_confirm & (base_score | base_context)).fillna(False)
    return (base_score | base_context).fillna(False)


def _market_guard(
    frame: DataFrame,
    side: str,
    mode: str,
    pressure_min: float,
    trend_min: float,
    rs_score_min: float,
) -> Series:
    pressure = _num(frame, "s2m_pressure_ratio")
    trend = _num(frame, "s2m_trend_z")
    directional_pressure = pressure.ge(pressure_min) if side == "long" else pressure.le(-pressure_min)
    trend_state = trend.ge(trend_min) if side == "long" else trend.le(-trend_min)
    avoids_adverse_pressure = pressure.ge(-pressure_min) if side == "long" else pressure.le(pressure_min)
    avoids_adverse_trend = trend.ge(-trend_min) if side == "long" else trend.le(trend_min)
    if mode == "pressure_and_trend":
        return (directional_pressure & trend_state).fillna(False)
    if mode == "directional_pressure":
        return directional_pressure.fillna(False)
    if mode == "trend_state":
        return trend_state.fillna(False)
    if mode == "avoid_adverse_pressure":
        return (avoids_adverse_pressure & avoids_adverse_trend).fillna(False)
    if mode == "avoid_chop":
        return (pressure.abs().ge(pressure_min) | trend.abs().ge(trend_min)).fillna(False)
    return (directional_pressure | trend_state).fillna(False)


def _param(strategy: Any, name: str, default: Any) -> Any:
    value = getattr(strategy, name, default)
    return getattr(value, "value", value)


def _enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


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

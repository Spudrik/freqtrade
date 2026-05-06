from __future__ import annotations

from typing import Any as PatternStructureConfig

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from pattern_common import (
    _clip_value,
    _dedupe_interval_level_events,
    _geometry_boundary_proof_columns,
    _line_fit_with_error,
    _pattern_geometry_arrays,
    _prior_pattern_move,
    _recent_confirmed_pattern_pivots,
)


def _channel_pattern_columns(frame: DataFrame, cfg: PatternStructureConfig) -> dict[str, Series]:
    """Detect sloping channel and broadening structures from confirmed pivots.

    This is pattern evidence, not a separate trendline-channel foundation. It
    uses recent confirmed pivot envelopes to say "price may be respecting this
    sloping channel" or "price may be expanding into a broadening formation".
    """

    p = cfg.output_prefix
    arrays = _channel_pattern_arrays(frame, cfg)
    ascending = _dedupe_channel_setup(frame, cfg, arrays, "ascending_channel_setup")
    descending = _dedupe_channel_setup(frame, cfg, arrays, "descending_channel_setup")
    broadening_top = _dedupe_channel_setup(frame, cfg, arrays, "broadening_top_setup_short")
    broadening_bottom = _dedupe_channel_setup(frame, cfg, arrays, "broadening_bottom_setup_long")
    columns = {
        f"{p}_ascending_channel_quality": pd.Series(arrays["ascending_channel_quality"], index=frame.index, dtype="float64").where(
            ascending, 0.0
        ),
        f"{p}_descending_channel_quality": pd.Series(arrays["descending_channel_quality"], index=frame.index, dtype="float64").where(
            descending, 0.0
        ),
        f"{p}_broadening_top_quality": pd.Series(arrays["broadening_top_quality"], index=frame.index, dtype="float64").where(
            broadening_top, 0.0
        ),
        f"{p}_broadening_bottom_quality": pd.Series(arrays["broadening_bottom_quality"], index=frame.index, dtype="float64").where(
            broadening_bottom, 0.0
        ),
        f"{p}_ascending_channel_setup": ascending,
        f"{p}_descending_channel_setup": descending,
        f"{p}_broadening_top_setup_short": broadening_top,
        f"{p}_broadening_bottom_setup_long": broadening_bottom,
        f"{p}_channel_pattern_upper": pd.Series(arrays["upper"], index=frame.index, dtype="float64"),
        f"{p}_channel_pattern_lower": pd.Series(arrays["lower"], index=frame.index, dtype="float64"),
        f"{p}_channel_pattern_position": pd.Series(arrays["position"], index=frame.index, dtype="float64"),
        f"{p}_channel_pattern_width_pct": pd.Series(arrays["width_pct"], index=frame.index, dtype="float64"),
        f"{p}_ascending_channel_go_long": pd.Series(arrays["ascending_channel_go_long"], index=frame.index, dtype="bool").fillna(False),
        f"{p}_ascending_channel_go_short": pd.Series(arrays["ascending_channel_go_short"], index=frame.index, dtype="bool").fillna(False),
        f"{p}_descending_channel_go_long": pd.Series(arrays["descending_channel_go_long"], index=frame.index, dtype="bool").fillna(False),
        f"{p}_descending_channel_go_short": pd.Series(arrays["descending_channel_go_short"], index=frame.index, dtype="bool").fillna(False),
        f"{p}_broadening_top_go_short": pd.Series(arrays["broadening_top_go_short"], index=frame.index, dtype="bool").fillna(False),
        f"{p}_broadening_bottom_go_long": pd.Series(arrays["broadening_bottom_go_long"], index=frame.index, dtype="bool").fillna(False),
        f"{p}_channel_pattern_breakout_up": pd.Series(arrays["breakout_up"], index=frame.index, dtype="bool").fillna(False),
        f"{p}_channel_pattern_breakdown_down": pd.Series(arrays["breakdown_down"], index=frame.index, dtype="bool").fillna(False),
        f"{p}_channel_pattern_exit_long": pd.Series(arrays["exit_long"], index=frame.index, dtype="bool").fillna(False),
        f"{p}_channel_pattern_exit_short": pd.Series(arrays["exit_short"], index=frame.index, dtype="bool").fillna(False),
    }
    if bool(getattr(cfg, "include_pattern_diagnostics", False)):
        proof_arrays = {
            "geometry_start_index": arrays["start_index"],
            "geometry_end_index": arrays["end_index"],
            "geometry_upper_start": arrays["upper_start"],
            "geometry_lower_start": arrays["lower_start"],
            "geometry_upper": arrays["upper"],
            "geometry_lower": arrays["lower"],
        }
        columns.update(
            {
                f"{p}_channel_pattern_slope_gap_pct_per_bar": pd.Series(
                    arrays["slope_gap_pct_per_bar"], index=frame.index, dtype="float64"
                ),
                f"{p}_channel_pattern_containment_ratio": pd.Series(
                    arrays["containment_ratio"], index=frame.index, dtype="float64"
                ),
                **_geometry_boundary_proof_columns(frame.index, f"{p}_ascending_channel", ascending, proof_arrays),
                **_geometry_boundary_proof_columns(frame.index, f"{p}_descending_channel", descending, proof_arrays),
                **_geometry_boundary_proof_columns(frame.index, f"{p}_broadening_top_short", broadening_top, proof_arrays),
                **_geometry_boundary_proof_columns(frame.index, f"{p}_broadening_bottom_long", broadening_bottom, proof_arrays),
            }
        )
    return columns


def _channel_pattern_arrays(frame: DataFrame, cfg: PatternStructureConfig) -> dict[str, np.ndarray]:
    close, body_high, body_low, high_pivot, high_index, low_pivot, low_index = _pattern_geometry_arrays(frame, cfg)
    high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype="float64")
    low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype="float64")
    rows = len(frame)
    out = _empty_channel_output(rows)
    window = int(cfg.channel_pattern_window)
    if bool(getattr(cfg, "include_broadening_patterns", False)):
        window = max(window, int(getattr(cfg, "broadening_pattern_window", window)))
    min_bars = int(cfg.min_channel_pattern_bars)
    min_side_pivots = int(cfg.channel_pattern_min_side_pivots)
    min_slope = float(cfg.min_channel_pattern_slope_pct_per_bar)
    slope_tolerance = float(cfg.channel_parallel_slope_tolerance_pct_per_bar)
    max_fit_error = float(cfg.max_channel_pattern_fit_error_pct)
    min_containment = float(cfg.min_channel_pattern_containment_ratio)
    boundary_tolerance_pct = float(cfg.channel_pattern_boundary_tolerance_pct)
    management_buffer_pct = float(cfg.channel_pattern_management_buffer_pct)

    for row in range(rows):
        if not np.isfinite(close[row]) or float(close[row]) == 0.0:
            continue
        start = max(0, row - window + 1)
        hx, hy, lx, ly = _recent_confirmed_pattern_pivots(row, start, high_pivot, high_index, low_pivot, low_index)
        if len(hx) < min_side_pivots or len(lx) < min_side_pivots:
            continue
        first_x = float(min(float(np.nanmin(hx)), float(np.nanmin(lx))))
        last_x = float(max(float(np.nanmax(hx)), float(np.nanmax(lx))))
        span = last_x - first_x
        if span < min_bars:
            continue
        broadening_candidate = _broadening_envelope_candidate(
            row,
            close,
            body_high,
            body_low,
            hx,
            hy,
            lx,
            ly,
            first_x,
            span,
            cfg,
        )
        if broadening_candidate is not None:
            _store_broadening_candidate(out, row, broadening_candidate)
        high_slope, high_error, high_intercept = _line_fit_with_error(hx, hy)
        low_slope, low_error, low_intercept = _line_fit_with_error(lx, ly)
        if not np.isfinite([high_slope, low_slope, high_error, low_error, high_intercept, low_intercept]).all():
            continue
        if high_error > max_fit_error or low_error > max_fit_error:
            continue
        upper_start = high_slope * first_x + high_intercept
        lower_start = low_slope * first_x + low_intercept
        upper = high_slope * float(row) + high_intercept
        lower = low_slope * float(row) + low_intercept
        if not np.isfinite([upper_start, lower_start, upper, lower]).all() or upper <= lower:
            continue
        start_width = upper_start - lower_start
        width = upper - lower
        reference = max(abs(float(close[row])), 1e-9)
        width_pct = width / reference
        if width_pct < float(cfg.channel_pattern_min_width_pct) or width_pct > float(cfg.channel_pattern_max_width_pct):
            continue
        segment = slice(int(max(0, first_x)), row + 1)
        x_values = np.arange(int(max(0, first_x)), row + 1, dtype="float64")
        upper_line = high_slope * x_values + high_intercept
        lower_line = low_slope * x_values + low_intercept
        tolerance = reference * boundary_tolerance_pct
        contained = (
            (body_high[segment] <= upper_line + tolerance)
            & (body_low[segment] >= lower_line - tolerance)
            & np.isfinite(body_high[segment])
            & np.isfinite(body_low[segment])
        )
        if contained.size == 0:
            continue
        containment_ratio = float(np.mean(contained))
        if containment_ratio < min_containment:
            continue
        high_slope_pct = high_slope / reference
        low_slope_pct = low_slope / reference
        slope_gap = abs(high_slope_pct - low_slope_pct)
        parallel_score = _clip_value(1.0 - slope_gap / max(slope_tolerance, 1e-9))
        stable_width_score = _clip_value(1.0 - abs(width - start_width) / max(width, start_width, 1e-9))
        touch_score = _clip_value((min(len(hx), len(lx)) - min_side_pivots + 1.0) / 4.0)
        fit_score = _clip_value(1.0 - ((high_error + low_error) / 2.0) / max(max_fit_error, 1e-9))
        containment_score = _clip_value((containment_ratio - min_containment) / max(1.0 - min_containment, 1e-9))
        quality = _clip_value(
            0.28 * parallel_score
            + 0.22 * stable_width_score
            + 0.18 * touch_score
            + 0.16 * fit_score
            + 0.16 * containment_score
        )
        position = _clip_value((float(close[row]) - lower) / max(width, 1e-9))
        buffer = reference * management_buffer_pct
        near_lower = bool(float(low[row]) <= lower + buffer and float(close[row]) >= lower - buffer)
        near_upper = bool(float(high[row]) >= upper - buffer and float(close[row]) <= upper + buffer)
        breakout_up = bool(float(close[row]) > upper + buffer)
        breakdown_down = bool(float(close[row]) < lower - buffer)
        rising_parallel = high_slope_pct > min_slope and low_slope_pct > min_slope and slope_gap <= slope_tolerance
        falling_parallel = high_slope_pct < -min_slope and low_slope_pct < -min_slope and slope_gap <= slope_tolerance
        width_expansion = (width - start_width) / max(start_width, 1e-9)
        diverging = (high_slope_pct - low_slope_pct) >= float(cfg.min_broadening_slope_gap_pct_per_bar)
        broadening_quality = _clip_value(
            0.26 * _clip_value(width_expansion / max(float(cfg.min_broadening_expansion_pct), 1e-9))
            + 0.22 * touch_score
            + 0.18 * fit_score
            + 0.18 * containment_score
            + 0.16 * _clip_value((high_slope_pct - low_slope_pct) / max(float(cfg.min_broadening_slope_gap_pct_per_bar) * 2.0, 1e-9))
        )
        prior_direction, prior_move_pct = _prior_pattern_move(close, first_x, span)
        broadening_top = (
            diverging
            and width_expansion >= float(cfg.min_broadening_expansion_pct)
            and prior_direction > 0
            and prior_move_pct >= float(cfg.min_broadening_prior_move_pct)
            and broadening_quality >= float(cfg.min_broadening_quality)
        )
        broadening_bottom = (
            diverging
            and width_expansion >= float(cfg.min_broadening_expansion_pct)
            and prior_direction < 0
            and prior_move_pct >= float(cfg.min_broadening_prior_move_pct)
            and broadening_quality >= float(cfg.min_broadening_quality)
        )
        ascending = rising_parallel and quality >= float(cfg.min_channel_pattern_quality)
        descending = falling_parallel and quality >= float(cfg.min_channel_pattern_quality)
        if not (ascending or descending or broadening_top or broadening_bottom):
            continue

        out["upper"][row] = upper
        out["lower"][row] = lower
        out["upper_start"][row] = upper_start
        out["lower_start"][row] = lower_start
        out["start_index"][row] = first_x
        out["end_index"][row] = float(row)
        out["position"][row] = position
        out["width_pct"][row] = width_pct
        out["slope_gap_pct_per_bar"][row] = slope_gap
        out["containment_ratio"][row] = containment_ratio
        out["breakout_up"][row] = breakout_up
        out["breakdown_down"][row] = breakdown_down
        out["exit_long"][row] = breakdown_down
        out["exit_short"][row] = breakout_up
        if ascending:
            out["ascending_channel_quality"][row] = quality
            out["ascending_channel_setup"][row] = True
            out["ascending_channel_go_long"][row] = near_lower
            out["ascending_channel_go_short"][row] = near_upper
        if descending:
            out["descending_channel_quality"][row] = quality
            out["descending_channel_setup"][row] = True
            out["descending_channel_go_long"][row] = near_lower
            out["descending_channel_go_short"][row] = near_upper
        if broadening_top:
            out["broadening_top_quality"][row] = broadening_quality
            out["broadening_top_setup_short"][row] = True
            out["broadening_top_go_short"][row] = near_upper
        if broadening_bottom:
            out["broadening_bottom_quality"][row] = broadening_quality
            out["broadening_bottom_setup_long"][row] = True
            out["broadening_bottom_go_long"][row] = near_lower
    return out


def _broadening_envelope_candidate(
    row: int,
    close: np.ndarray,
    body_high: np.ndarray,
    body_low: np.ndarray,
    high_x: np.ndarray,
    high_y: np.ndarray,
    low_x: np.ndarray,
    low_y: np.ndarray,
    first_x: float,
    span: float,
    cfg: PatternStructureConfig,
) -> dict[str, float] | None:
    """Detect broadening from expanding pivot envelopes, not channel containment.

    Broadening formations are not ordinary channels. The earlier implementation
    reused strict channel fit/containment gates, which filtered out long-form
    expanding structures before broadening logic could score them. This helper
    scores expansion directly: rising resistance envelope, falling support
    envelope, widening range, and enough pivot evidence.
    """

    reference = max(abs(float(close[row])), 1e-9)
    high_slope, high_error, high_intercept = _line_fit_with_error(high_x, high_y)
    low_slope, low_error, low_intercept = _line_fit_with_error(low_x, low_y)
    if not np.isfinite([high_slope, low_slope, high_error, low_error, high_intercept, low_intercept]).all():
        return None
    max_fit_error = float(cfg.max_channel_pattern_fit_error_pct) * 1.75
    if high_error > max_fit_error or low_error > max_fit_error:
        return None
    upper_start = high_slope * first_x + high_intercept
    lower_start = low_slope * first_x + low_intercept
    upper = high_slope * float(row) + high_intercept
    lower = low_slope * float(row) + low_intercept
    if not np.isfinite([upper_start, lower_start, upper, lower]).all() or upper <= lower or upper_start <= lower_start:
        return None
    start_width = upper_start - lower_start
    width = upper - lower
    width_pct = width / reference
    if width_pct < float(cfg.channel_pattern_min_width_pct) or width_pct > float(cfg.channel_pattern_max_width_pct) * 1.75:
        return None
    width_expansion = (width - start_width) / max(start_width, 1e-9)
    if width_expansion < float(cfg.min_broadening_expansion_pct):
        return None
    high_slope_pct = high_slope / reference
    low_slope_pct = low_slope / reference
    slope_gap = high_slope_pct - low_slope_pct
    if slope_gap < float(cfg.min_broadening_slope_gap_pct_per_bar):
        return None
    if not _pivot_envelope_expands(high_x, high_y, low_x, low_y):
        return None

    start = int(max(np.floor(first_x), 0))
    x_values = np.arange(start, row + 1, dtype="float64")
    upper_line = high_slope * x_values + high_intercept
    lower_line = low_slope * x_values + low_intercept
    tolerance = np.maximum(np.abs(close[start : row + 1]), 1e-9) * float(cfg.channel_pattern_boundary_tolerance_pct) * 2.0
    contained = (
        (body_high[start : row + 1] <= upper_line + tolerance)
        & (body_low[start : row + 1] >= lower_line - tolerance)
        & np.isfinite(body_high[start : row + 1])
        & np.isfinite(body_low[start : row + 1])
    )
    containment_ratio = float(np.nanmean(contained.astype("float64"))) if contained.size else 0.0
    min_containment = max(float(cfg.min_channel_pattern_containment_ratio) * 0.70, 0.35)
    if containment_ratio < min_containment:
        return None

    prior_direction, prior_move_pct = _prior_pattern_move(close, first_x, span)
    if prior_direction == 0 or prior_move_pct < float(cfg.min_broadening_prior_move_pct):
        return None
    expansion_score = _clip_value(width_expansion / max(float(cfg.min_broadening_expansion_pct) * 2.5, 1e-9))
    slope_score = _clip_value(slope_gap / max(float(cfg.min_broadening_slope_gap_pct_per_bar) * 3.0, 1e-9))
    touch_score = _clip_value((min(len(high_x), len(low_x)) - float(cfg.channel_pattern_min_side_pivots) + 1.0) / 5.0)
    fit_score = _clip_value(1.0 - ((high_error + low_error) / 2.0) / max(max_fit_error, 1e-9))
    containment_score = _clip_value((containment_ratio - min_containment) / max(1.0 - min_containment, 1e-9))
    quality = _clip_value(
        0.30 * expansion_score
        + 0.22 * slope_score
        + 0.18 * touch_score
        + 0.16 * fit_score
        + 0.14 * containment_score
    )
    if quality < float(cfg.min_broadening_quality):
        return None
    position = _clip_value((float(close[row]) - lower) / max(width, 1e-9))
    return {
        "side": 1.0 if prior_direction > 0 else -1.0,
        "quality": quality,
        "upper": float(upper),
        "lower": float(lower),
        "upper_start": float(upper_start),
        "lower_start": float(lower_start),
        "start_index": float(first_x),
        "end_index": float(row),
        "position": position,
        "width_pct": float(width_pct),
        "slope_gap_pct_per_bar": float(slope_gap),
        "containment_ratio": float(containment_ratio),
    }


def _pivot_envelope_expands(high_x: np.ndarray, high_y: np.ndarray, low_x: np.ndarray, low_y: np.ndarray) -> bool:
    if len(high_y) < 3 or len(low_y) < 3:
        return False
    high_order = np.argsort(high_x)
    low_order = np.argsort(low_x)
    high_prices = high_y[high_order]
    low_prices = low_y[low_order]
    split_high = max(len(high_prices) // 2, 1)
    split_low = max(len(low_prices) // 2, 1)
    early_high = float(np.nanmedian(high_prices[:split_high]))
    recent_high = float(np.nanmedian(high_prices[split_high:]))
    early_low = float(np.nanmedian(low_prices[:split_low]))
    recent_low = float(np.nanmedian(low_prices[split_low:]))
    return recent_high > early_high and recent_low < early_low


def _store_broadening_candidate(out: dict[str, np.ndarray], row: int, candidate: dict[str, float]) -> None:
    side = float(candidate["side"])
    if side > 0.0:
        out["broadening_top_quality"][row] = max(out["broadening_top_quality"][row], float(candidate["quality"]))
        out["broadening_top_setup_short"][row] = True
    else:
        out["broadening_bottom_quality"][row] = max(out["broadening_bottom_quality"][row], float(candidate["quality"]))
        out["broadening_bottom_setup_long"][row] = True
    for name in (
        "upper",
        "lower",
        "upper_start",
        "lower_start",
        "start_index",
        "end_index",
        "position",
        "width_pct",
        "slope_gap_pct_per_bar",
        "containment_ratio",
    ):
        out[name][row] = float(candidate[name])


def _empty_channel_output(rows: int) -> dict[str, np.ndarray]:
    out = {
        "ascending_channel_quality": np.zeros(rows, dtype="float64"),
        "descending_channel_quality": np.zeros(rows, dtype="float64"),
        "broadening_top_quality": np.zeros(rows, dtype="float64"),
        "broadening_bottom_quality": np.zeros(rows, dtype="float64"),
        "ascending_channel_setup": np.zeros(rows, dtype=bool),
        "descending_channel_setup": np.zeros(rows, dtype=bool),
        "broadening_top_setup_short": np.zeros(rows, dtype=bool),
        "broadening_bottom_setup_long": np.zeros(rows, dtype=bool),
        "ascending_channel_go_long": np.zeros(rows, dtype=bool),
        "ascending_channel_go_short": np.zeros(rows, dtype=bool),
        "descending_channel_go_long": np.zeros(rows, dtype=bool),
        "descending_channel_go_short": np.zeros(rows, dtype=bool),
        "broadening_top_go_short": np.zeros(rows, dtype=bool),
        "broadening_bottom_go_long": np.zeros(rows, dtype=bool),
        "breakout_up": np.zeros(rows, dtype=bool),
        "breakdown_down": np.zeros(rows, dtype=bool),
        "exit_long": np.zeros(rows, dtype=bool),
        "exit_short": np.zeros(rows, dtype=bool),
    }
    for name in (
        "upper",
        "lower",
        "upper_start",
        "lower_start",
        "start_index",
        "end_index",
        "position",
        "width_pct",
        "slope_gap_pct_per_bar",
        "containment_ratio",
    ):
        out[name] = np.full(rows, np.nan, dtype="float64")
    return out


def _dedupe_channel_setup(
    frame: DataFrame,
    cfg: PatternStructureConfig,
    arrays: dict[str, np.ndarray],
    mask_name: str,
) -> Series:
    return pd.Series(
        _dedupe_interval_level_events(
            arrays[mask_name],
            arrays["start_index"],
            arrays["end_index"],
            arrays["upper"],
            arrays["lower"],
            int(cfg.entry_cooldown_bars),
            float(cfg.double_duplicate_overlap_pct),
            float(cfg.double_duplicate_neckline_tolerance_pct),
        ),
        index=frame.index,
        dtype="bool",
    ).fillna(False)


__all__ = ["_channel_pattern_columns"]

from __future__ import annotations

from typing import Any as PatternStructureConfig

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from pattern_common import _clip_value, _pattern_geometry_arrays


def _rectangle_range_columns(frame: DataFrame, cfg: PatternStructureConfig) -> dict[str, Series]:
    """Detect horizontal rectangle/range structure from confirmed pivots.

    A rectangle is not a trade by itself. It says price appears boxed between
    repeated resistance and support. Advice columns are only emitted when the
    current candle interacts with the live box boundary.
    """

    p = cfg.output_prefix
    arrays = _rectangle_range_arrays(frame, cfg)
    setup = pd.Series(arrays["rectangle_setup"], index=frame.index, dtype="bool").fillna(False)
    return {
        f"{p}_rectangle_quality": pd.Series(arrays["rectangle_quality"], index=frame.index, dtype="float64").where(setup, 0.0),
        f"{p}_rectangle_setup": setup,
        f"{p}_rectangle_upper": pd.Series(arrays["rectangle_upper"], index=frame.index, dtype="float64"),
        f"{p}_rectangle_lower": pd.Series(arrays["rectangle_lower"], index=frame.index, dtype="float64"),
        f"{p}_rectangle_width_pct": pd.Series(arrays["rectangle_width_pct"], index=frame.index, dtype="float64"),
        f"{p}_rectangle_position": pd.Series(arrays["rectangle_position"], index=frame.index, dtype="float64"),
        f"{p}_rectangle_upper_touch_count": pd.Series(arrays["rectangle_upper_touch_count"], index=frame.index, dtype="float64"),
        f"{p}_rectangle_lower_touch_count": pd.Series(arrays["rectangle_lower_touch_count"], index=frame.index, dtype="float64"),
        f"{p}_rectangle_go_long": pd.Series(arrays["rectangle_go_long"], index=frame.index, dtype="bool").fillna(False),
        f"{p}_rectangle_go_short": pd.Series(arrays["rectangle_go_short"], index=frame.index, dtype="bool").fillna(False),
        f"{p}_rectangle_breakout_up": pd.Series(arrays["rectangle_breakout_up"], index=frame.index, dtype="bool").fillna(False),
        f"{p}_rectangle_breakdown_down": pd.Series(arrays["rectangle_breakdown_down"], index=frame.index, dtype="bool").fillna(False),
        f"{p}_rectangle_exit_long": pd.Series(arrays["rectangle_exit_long"], index=frame.index, dtype="bool").fillna(False),
        f"{p}_rectangle_exit_short": pd.Series(arrays["rectangle_exit_short"], index=frame.index, dtype="bool").fillna(False),
    }


def _rectangle_range_arrays(frame: DataFrame, cfg: PatternStructureConfig) -> dict[str, np.ndarray]:
    close, body_high, body_low, high_pivot, high_index, low_pivot, low_index = _pattern_geometry_arrays(frame, cfg)
    high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype="float64")
    low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype="float64")
    rows = len(frame)
    out = {
        "rectangle_quality": np.zeros(rows, dtype="float64"),
        "rectangle_setup": np.zeros(rows, dtype=bool),
        "rectangle_upper": np.full(rows, np.nan, dtype="float64"),
        "rectangle_lower": np.full(rows, np.nan, dtype="float64"),
        "rectangle_width_pct": np.full(rows, np.nan, dtype="float64"),
        "rectangle_position": np.full(rows, np.nan, dtype="float64"),
        "rectangle_upper_touch_count": np.zeros(rows, dtype="float64"),
        "rectangle_lower_touch_count": np.zeros(rows, dtype="float64"),
        "rectangle_go_long": np.zeros(rows, dtype=bool),
        "rectangle_go_short": np.zeros(rows, dtype=bool),
        "rectangle_breakout_up": np.zeros(rows, dtype=bool),
        "rectangle_breakdown_down": np.zeros(rows, dtype=bool),
        "rectangle_exit_long": np.zeros(rows, dtype=bool),
        "rectangle_exit_short": np.zeros(rows, dtype=bool),
    }
    window = int(cfg.rectangle_window)
    min_bars = int(cfg.min_rectangle_bars)
    min_touches = int(cfg.rectangle_min_side_touches)
    boundary_tolerance_pct = float(cfg.rectangle_boundary_tolerance_pct)
    management_buffer_pct = float(cfg.rectangle_management_buffer_pct)
    min_width_pct = float(cfg.rectangle_min_width_pct)
    max_width_pct = float(cfg.rectangle_max_width_pct)
    min_quality = float(cfg.min_rectangle_quality)

    for row in range(rows):
        reference = abs(float(close[row])) if np.isfinite(close[row]) else np.nan
        if not np.isfinite(reference) or reference <= 0.0:
            continue
        start = max(0, row - window + 1)
        high_mask = (
            np.isfinite(high_pivot)
            & np.isfinite(high_index)
            & (np.arange(rows) <= row)
            & (high_index >= float(start))
            & (high_index <= float(row))
        )
        low_mask = (
            np.isfinite(low_pivot)
            & np.isfinite(low_index)
            & (np.arange(rows) <= row)
            & (low_index >= float(start))
            & (low_index <= float(row))
        )
        if int(high_mask.sum()) < min_touches or int(low_mask.sum()) < min_touches:
            continue

        high_prices = high_pivot[high_mask]
        high_x = high_index[high_mask]
        low_prices = low_pivot[low_mask]
        low_x = low_index[low_mask]
        upper_seed = float(np.nanpercentile(high_prices, 70.0))
        lower_seed = float(np.nanpercentile(low_prices, 30.0))
        upper_cluster = high_prices >= upper_seed
        lower_cluster = low_prices <= lower_seed
        if int(upper_cluster.sum()) < min_touches or int(lower_cluster.sum()) < min_touches:
            continue

        upper = float(np.nanmedian(high_prices[upper_cluster]))
        lower = float(np.nanmedian(low_prices[lower_cluster]))
        if not np.isfinite([upper, lower]).all() or upper <= lower:
            continue
        width_pct = (upper - lower) / reference
        if width_pct < min_width_pct or width_pct > max_width_pct:
            continue

        tolerance = max(reference * boundary_tolerance_pct, (upper - lower) * 0.08)
        upper_touch = np.abs(high_prices - upper) <= tolerance
        lower_touch = np.abs(low_prices - lower) <= tolerance
        upper_count = int(upper_touch.sum())
        lower_count = int(lower_touch.sum())
        if upper_count < min_touches or lower_count < min_touches:
            continue

        first_touch = int(max(0.0, min(float(np.nanmin(high_x[upper_touch])), float(np.nanmin(low_x[lower_touch])))))
        if row - first_touch < min_bars:
            continue
        segment = slice(first_touch, row + 1)
        containment = (
            (body_high[segment] <= upper + tolerance)
            & (body_low[segment] >= lower - tolerance)
            & np.isfinite(body_high[segment])
            & np.isfinite(body_low[segment])
        )
        if containment.size == 0:
            continue
        containment_ratio = float(np.mean(containment))
        if containment_ratio < float(cfg.min_rectangle_containment_ratio):
            continue

        upper_dev = float(np.nanmean(np.abs(high_prices[upper_touch] - upper))) if upper_count else np.inf
        lower_dev = float(np.nanmean(np.abs(low_prices[lower_touch] - lower))) if lower_count else np.inf
        flatness_score = _clip_value(1.0 - ((upper_dev + lower_dev) / 2.0) / max(tolerance, 1e-9))
        touch_balance = min(upper_count, lower_count) / max(max(upper_count, lower_count), 1)
        touch_score = _clip_value((min(upper_count, lower_count) - min_touches + 1.0) / 3.0)
        width_mid = (min_width_pct + max_width_pct) / 2.0
        width_score = _clip_value(1.0 - abs(width_pct - width_mid) / max(width_mid, 1e-9))
        span_score = _clip_value((row - first_touch) / max(float(window), 1.0))
        quality = _clip_value(
            0.28 * flatness_score
            + 0.24 * containment_ratio
            + 0.18 * touch_score
            + 0.14 * touch_balance
            + 0.10 * width_score
            + 0.06 * span_score
        )
        if quality < min_quality:
            continue

        buffer = reference * management_buffer_pct
        position = _clip_value((float(close[row]) - lower) / max(upper - lower, 1e-9))
        go_long = bool(float(low[row]) <= lower + buffer and float(close[row]) >= lower - buffer)
        go_short = bool(float(high[row]) >= upper - buffer and float(close[row]) <= upper + buffer)
        breakout_up = bool(float(close[row]) > upper + buffer)
        breakdown_down = bool(float(close[row]) < lower - buffer)

        out["rectangle_quality"][row] = quality
        out["rectangle_setup"][row] = True
        out["rectangle_upper"][row] = upper
        out["rectangle_lower"][row] = lower
        out["rectangle_width_pct"][row] = width_pct
        out["rectangle_position"][row] = position
        out["rectangle_upper_touch_count"][row] = float(upper_count)
        out["rectangle_lower_touch_count"][row] = float(lower_count)
        out["rectangle_go_long"][row] = go_long
        out["rectangle_go_short"][row] = go_short
        out["rectangle_breakout_up"][row] = breakout_up
        out["rectangle_breakdown_down"][row] = breakdown_down
        out["rectangle_exit_long"][row] = breakdown_down
        out["rectangle_exit_short"][row] = breakout_up
    return out


__all__ = ["_rectangle_range_columns"]

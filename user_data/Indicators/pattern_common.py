from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


def _continuation_proof_columns(
    index: pd.Index,
    base: str,
    mask: Series,
    arrays: dict[str, np.ndarray],
    side: str,
) -> dict[str, Series]:
    """Expose the pole plus upper/lower consolidation boundaries for review.

    line1 is the impulse/pole. line2 is the upper consolidation boundary.
    line3 is the lower consolidation boundary. Values are only emitted on rows
    where the matching flag or pennant setup is active.
    """

    return _proof_line_columns(
        index,
        base,
        mask,
        {
            line_number: tuple(arrays[f"proof_line{line_number}_{field}_{side}"] for field in ("x1", "y1", "x2", "y2"))
            for line_number in (1, 2, 3)
        },
    )


def _geometry_boundary_proof_columns(
    index: pd.Index,
    base: str,
    mask: Series,
    arrays: dict[str, np.ndarray],
) -> dict[str, Series]:
    """Expose upper/lower fitted boundaries for two-line geometric patterns."""

    return _proof_line_columns(
        index,
        base,
        mask,
        {
            1: (
                arrays.get("geometry_upper_start_index", arrays["geometry_start_index"]),
                arrays.get("geometry_upper_anchor_start", arrays["geometry_upper_start"]),
                arrays["geometry_end_index"],
                arrays["geometry_upper"],
            ),
            2: (
                arrays.get("geometry_lower_start_index", arrays["geometry_start_index"]),
                arrays.get("geometry_lower_anchor_start", arrays["geometry_lower_start"]),
                arrays["geometry_end_index"],
                arrays["geometry_lower"],
            ),
        },
    )


def _proof_line_columns(
    index: pd.Index,
    base: str,
    mask: Series,
    lines: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, Series]:
    # TODO_DELETE_CHECK: proof-line columns are diagnostic evidence for visual
    # review and no-lookahead auditing. Before release/hyperopt scale-out, check
    # whether these columns should be gated or trimmed to reduce dataframe bloat.
    clean_mask = pd.Series(mask, index=index).fillna(False).astype("bool")
    columns: dict[str, Series] = {}
    for line_number, values in lines.items():
        for field, source in zip(("x1", "y1", "x2", "y2"), values):
            columns[f"{base}_line{line_number}_{field}"] = pd.Series(source, index=index, dtype="float64").where(clean_mask)
    return columns


def _pattern_geometry_arrays(
    frame: DataFrame,
    cfg: PatternStructureConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    close = _num(frame["close"]).replace(0.0, np.nan).to_numpy(dtype="float64")
    open_ = _num(frame["open"]).to_numpy(dtype="float64")
    body_high = np.fmax(open_, close)
    body_low = np.fmin(open_, close)
    pp = cfg.pivot_prefix
    high_pivot = _num(frame[f"{pp}_pivot_high"]).to_numpy(dtype="float64")
    low_pivot = _num(frame[f"{pp}_pivot_low"]).to_numpy(dtype="float64")
    fallback_index = np.arange(len(frame), dtype="float64")
    high_index = _num(frame.get(f"{pp}_pivot_high_index", pd.Series(fallback_index, index=frame.index))).to_numpy(dtype="float64")
    low_index = _num(frame.get(f"{pp}_pivot_low_index", pd.Series(fallback_index, index=frame.index))).to_numpy(dtype="float64")
    micro_high, micro_high_index, micro_low, micro_low_index = _confirmed_micro_pivots(
        body_high,
        body_low,
        int(cfg.pattern_pivot_strength),
    )
    high_pivot = _combine_sparse_pivots(high_pivot, high_index, micro_high, micro_high_index)
    low_pivot = _combine_sparse_pivots(low_pivot, low_index, micro_low, micro_low_index)
    high_index = _combine_sparse_indexes(high_pivot, high_index, micro_high_index)
    low_index = _combine_sparse_indexes(low_pivot, low_index, micro_low_index)
    return close, body_high, body_low, high_pivot, high_index, low_pivot, low_index


def _recent_confirmed_pattern_pivots(
    row: int,
    start: int,
    high_pivot: np.ndarray,
    high_index: np.ndarray,
    low_pivot: np.ndarray,
    low_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    high_prices = high_pivot[start : row + 1]
    high_x = high_index[start : row + 1]
    low_prices = low_pivot[start : row + 1]
    low_x = low_index[start : row + 1]
    high_mask = np.isfinite(high_prices) & np.isfinite(high_x) & (high_x >= float(start)) & (high_x <= float(row))
    low_mask = np.isfinite(low_prices) & np.isfinite(low_x) & (low_x >= float(start)) & (low_x <= float(row))
    return high_x[high_mask], high_prices[high_mask], low_x[low_mask], low_prices[low_mask]


def _line_fit_with_error(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return 0.0, np.inf, np.nan
    xv = x[valid].astype("float64")
    yv = y[valid].astype("float64")
    x_mean = float(np.mean(xv))
    y_mean = float(np.mean(yv))
    denominator = float(np.sum((xv - x_mean) ** 2))
    if denominator <= 0.0:
        return 0.0, np.inf, np.nan
    slope = float(np.sum((xv - x_mean) * (yv - y_mean)) / denominator)
    intercept = y_mean - slope * x_mean
    fitted = slope * xv + intercept
    fit_error = float(np.mean(np.abs(yv - fitted)) / max(np.mean(np.abs(yv)), 1e-9))
    return slope, fit_error, intercept


def _line_segment_from_points(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return np.nan, np.nan, np.nan, np.nan
    xv = x[valid].astype("float64")
    yv = y[valid].astype("float64")
    order = np.argsort(xv)
    xv = xv[order]
    yv = yv[order]
    x1 = float(xv[0])
    x2 = float(xv[-1])
    return x1, float(yv[0]), x2, float(yv[-1])


def _line_value_at(x: float, x1: float, y1: float, x2: float, y2: float) -> float:
    if not np.isfinite([x, x1, y1, x2, y2]).all() or x2 == x1:
        return np.nan
    slope = (y2 - y1) / (x2 - x1)
    return float(y1 + slope * (x - x1))


def _prior_pattern_move(close: np.ndarray, first_anchor: float, span: float) -> tuple[int, float]:
    anchor = int(max(min(first_anchor, len(close) - 1), 0))
    lookback = int(max(span, 3.0))
    prior = max(0, anchor - lookback)
    if not np.isfinite(close[anchor]) or not np.isfinite(close[prior]) or close[prior] == 0.0:
        return 0, 0.0
    move_pct = abs(float(close[anchor]) - float(close[prior])) / max(abs(float(close[prior])), 1e-9)
    return (1 if close[anchor] >= close[prior] else -1), float(move_pct)


def _confirmed_micro_pivots(
    body_high: np.ndarray,
    body_low: np.ndarray,
    strength: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = len(body_high)
    pivot_high = np.full(rows, np.nan, dtype="float64")
    pivot_low = np.full(rows, np.nan, dtype="float64")
    high_index = np.full(rows, np.nan, dtype="float64")
    low_index = np.full(rows, np.nan, dtype="float64")
    wing = max(int(strength), 1)
    window = wing * 2 + 1
    if rows < window:
        return pivot_high, high_index, pivot_low, low_index

    for confirm_row in range(window - 1, rows):
        anchor = confirm_row - wing
        start = anchor - wing
        stop = anchor + wing + 1
        high_window = body_high[start:stop]
        low_window = body_low[start:stop]
        high_value = body_high[anchor]
        low_value = body_low[anchor]
        if np.isfinite(high_value) and np.isfinite(high_window).all() and high_value >= float(np.nanmax(high_window)):
            pivot_high[confirm_row] = float(high_value)
            high_index[confirm_row] = float(anchor)
        if np.isfinite(low_value) and np.isfinite(low_window).all() and low_value <= float(np.nanmin(low_window)):
            pivot_low[confirm_row] = float(low_value)
            low_index[confirm_row] = float(anchor)
    return pivot_high, high_index, pivot_low, low_index


def _combine_sparse_pivots(
    foundation_price: np.ndarray,
    foundation_index: np.ndarray,
    micro_price: np.ndarray,
    micro_index: np.ndarray,
) -> np.ndarray:
    out = micro_price.copy()
    foundation_valid = np.isfinite(foundation_price) & np.isfinite(foundation_index)
    out[foundation_valid] = foundation_price[foundation_valid]
    return out


def _combine_sparse_indexes(
    combined_price: np.ndarray,
    foundation_index: np.ndarray,
    micro_index: np.ndarray,
) -> np.ndarray:
    out = micro_index.copy()
    foundation_valid = np.isfinite(foundation_index)
    out[foundation_valid] = foundation_index[foundation_valid]
    out[~np.isfinite(combined_price)] = np.nan
    return out


def _slope_from_points(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return 0.0
    xv = x[valid].astype("float64")
    yv = y[valid].astype("float64")
    x_mean = float(np.mean(xv))
    y_mean = float(np.mean(yv))
    denominator = float(np.sum((xv - x_mean) ** 2))
    if denominator <= 0.0:
        return 0.0
    return float(np.sum((xv - x_mean) * (yv - y_mean)) / denominator)


def _clip_value(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(max(value, 0.0), 1.0))


def _nanmax_pair(left: float, right: float) -> float:
    if np.isfinite(left) and np.isfinite(right):
        return float(max(left, right))
    if np.isfinite(left):
        return float(left)
    if np.isfinite(right):
        return float(right)
    return np.nan


def _num(series: Series) -> Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")

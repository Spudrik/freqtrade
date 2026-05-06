from __future__ import annotations

from typing import Any as PatternStructureConfig

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from pattern_common import (
    _clip_value,
    _dedupe_interval_level_events,
    _lifecycle_state_from_events,
    _pattern_geometry_arrays,
    _prior_pattern_move,
    _proof_line_columns,
)


def _triple_reversal_columns(frame: DataFrame, cfg: PatternStructureConfig) -> dict[str, Series]:
    """Detect triple top / triple bottom attention at the third confirmed pivot.

    A triple top is three resistance touches after an upward move. A triple
    bottom is three support touches after a downward move. The indicator emits
    attention at the third confirmed touch; the strategy decides whether to
    fade the level, wait for a neckline break, or ignore it.
    """

    p = cfg.output_prefix
    arrays = _triple_reversal_arrays(frame, cfg)
    top_raw = pd.Series(arrays["triple_top_setup_short"], index=frame.index, dtype="bool").fillna(False)
    bottom_raw = pd.Series(arrays["triple_bottom_setup_long"], index=frame.index, dtype="bool").fillna(False)
    top = pd.Series(
        _dedupe_interval_level_events(
            arrays["triple_top_setup_short"],
            arrays["triple_top_first_index"],
            arrays["triple_top_third_index"],
            arrays["triple_top_level"],
            arrays["triple_top_neckline"],
            int(cfg.entry_cooldown_bars),
            float(cfg.double_duplicate_overlap_pct),
            float(cfg.double_duplicate_neckline_tolerance_pct),
        ),
        index=frame.index,
        dtype="bool",
    ).fillna(False)
    bottom = pd.Series(
        _dedupe_interval_level_events(
            arrays["triple_bottom_setup_long"],
            arrays["triple_bottom_first_index"],
            arrays["triple_bottom_third_index"],
            arrays["triple_bottom_level"],
            arrays["triple_bottom_neckline"],
            int(cfg.entry_cooldown_bars),
            float(cfg.double_duplicate_overlap_pct),
            float(cfg.double_duplicate_neckline_tolerance_pct),
        ),
        index=frame.index,
        dtype="bool",
    ).fillna(False)
    columns = {
        f"{p}_triple_top_quality": pd.Series(arrays["triple_top_quality"], index=frame.index, dtype="float64").where(top, 0.0),
        f"{p}_triple_bottom_quality": pd.Series(arrays["triple_bottom_quality"], index=frame.index, dtype="float64").where(bottom, 0.0),
        f"{p}_triple_top_structure_quality": pd.Series(
            arrays["triple_top_quality"], index=frame.index, dtype="float64"
        ).where(top_raw, 0.0),
        f"{p}_triple_bottom_structure_quality": pd.Series(
            arrays["triple_bottom_quality"], index=frame.index, dtype="float64"
        ).where(bottom_raw, 0.0),
        f"{p}_triple_top_setup_short": top,
        f"{p}_triple_bottom_setup_long": bottom,
        f"{p}_triple_top_state": pd.Series(
            _lifecycle_state_from_events(
                top,
                int(cfg.pattern_lifecycle_mature_bars),
                int(cfg.pattern_lifecycle_stale_bars),
            ),
            index=frame.index,
            dtype="int8",
        ),
        f"{p}_triple_bottom_state": pd.Series(
            _lifecycle_state_from_events(
                bottom,
                int(cfg.pattern_lifecycle_mature_bars),
                int(cfg.pattern_lifecycle_stale_bars),
            ),
            index=frame.index,
            dtype="int8",
        ),
        f"{p}_triple_top_structure_short": top_raw,
        f"{p}_triple_bottom_structure_long": bottom_raw,
        f"{p}_triple_top_first_index": pd.Series(arrays["triple_top_first_index"], index=frame.index, dtype="float64"),
        f"{p}_triple_top_second_index": pd.Series(arrays["triple_top_second_index"], index=frame.index, dtype="float64"),
        f"{p}_triple_top_third_index": pd.Series(arrays["triple_top_third_index"], index=frame.index, dtype="float64"),
        f"{p}_triple_top_first_price": pd.Series(arrays["triple_top_first_price"], index=frame.index, dtype="float64"),
        f"{p}_triple_top_second_price": pd.Series(arrays["triple_top_second_price"], index=frame.index, dtype="float64"),
        f"{p}_triple_top_third_price": pd.Series(arrays["triple_top_third_price"], index=frame.index, dtype="float64"),
        f"{p}_triple_top_neckline": pd.Series(arrays["triple_top_neckline"], index=frame.index, dtype="float64"),
        f"{p}_triple_top_neckline_index": pd.Series(arrays["triple_top_neckline_index"], index=frame.index, dtype="float64"),
        f"{p}_triple_top_resistance": pd.Series(arrays["triple_top_level"], index=frame.index, dtype="float64"),
        f"{p}_triple_top_reaction_score": pd.Series(arrays["triple_top_reaction_score"], index=frame.index, dtype="float64").where(
            top_raw, 0.0
        ),
        f"{p}_triple_bottom_first_index": pd.Series(arrays["triple_bottom_first_index"], index=frame.index, dtype="float64"),
        f"{p}_triple_bottom_second_index": pd.Series(arrays["triple_bottom_second_index"], index=frame.index, dtype="float64"),
        f"{p}_triple_bottom_third_index": pd.Series(arrays["triple_bottom_third_index"], index=frame.index, dtype="float64"),
        f"{p}_triple_bottom_first_price": pd.Series(arrays["triple_bottom_first_price"], index=frame.index, dtype="float64"),
        f"{p}_triple_bottom_second_price": pd.Series(arrays["triple_bottom_second_price"], index=frame.index, dtype="float64"),
        f"{p}_triple_bottom_third_price": pd.Series(arrays["triple_bottom_third_price"], index=frame.index, dtype="float64"),
        f"{p}_triple_bottom_neckline": pd.Series(arrays["triple_bottom_neckline"], index=frame.index, dtype="float64"),
        f"{p}_triple_bottom_neckline_index": pd.Series(arrays["triple_bottom_neckline_index"], index=frame.index, dtype="float64"),
        f"{p}_triple_bottom_support": pd.Series(arrays["triple_bottom_level"], index=frame.index, dtype="float64"),
        f"{p}_triple_bottom_reaction_score": pd.Series(
            arrays["triple_bottom_reaction_score"], index=frame.index, dtype="float64"
        ).where(bottom_raw, 0.0),
    }
    if bool(getattr(cfg, "include_pattern_diagnostics", False)):
        columns.update(
            {
                **_proof_line_columns(
                    frame.index,
                    f"{p}_triple_top_short",
                    top,
                    {
                        1: (
                            arrays["triple_top_first_index"],
                            arrays["triple_top_first_price"],
                            arrays["triple_top_third_index"],
                            arrays["triple_top_third_price"],
                        ),
                        2: (
                            arrays["triple_top_first_index"],
                            arrays["triple_top_neckline"],
                            arrays["triple_top_third_index"],
                            arrays["triple_top_neckline"],
                        ),
                    },
                ),
                **_proof_line_columns(
                    frame.index,
                    f"{p}_triple_bottom_long",
                    bottom,
                    {
                        1: (
                            arrays["triple_bottom_first_index"],
                            arrays["triple_bottom_first_price"],
                            arrays["triple_bottom_third_index"],
                            arrays["triple_bottom_third_price"],
                        ),
                        2: (
                            arrays["triple_bottom_first_index"],
                            arrays["triple_bottom_neckline"],
                            arrays["triple_bottom_third_index"],
                            arrays["triple_bottom_neckline"],
                        ),
                    },
                ),
            }
        )
    return columns


def _triple_reversal_arrays(frame: DataFrame, cfg: PatternStructureConfig) -> dict[str, np.ndarray]:
    close, _, _, high_pivot, high_index, low_pivot, low_index = _pattern_geometry_arrays(frame, cfg)
    rows = len(frame)
    out = _empty_triple_output(rows)
    window = int(cfg.triple_pattern_window)
    min_bars = int(cfg.min_triple_pattern_bars)
    max_bars = int(cfg.max_triple_pattern_bars)
    min_spacing = int(cfg.min_triple_spacing_bars)
    max_candidates = int(cfg.triple_max_candidate_pivots)

    for row in range(rows):
        if not np.isfinite(close[row]) or float(close[row]) == 0.0:
            continue
        if np.isfinite(high_pivot[row]) and np.isfinite(high_index[row]):
            top = _score_triple_top(
                row,
                close,
                high_pivot,
                high_index,
                low_pivot,
                low_index,
                window,
                min_bars,
                max_bars,
                min_spacing,
                max_candidates,
                float(cfg.triple_peak_tolerance_pct),
                float(cfg.min_triple_neckline_depth_pct),
                float(cfg.min_triple_prior_move_pct),
                float(cfg.min_triple_reaction_score),
            )
            if top["quality"] >= float(cfg.min_triple_quality):
                _store_triple(out, row, "triple_top", top)
        if np.isfinite(low_pivot[row]) and np.isfinite(low_index[row]):
            bottom = _score_triple_bottom(
                row,
                close,
                high_pivot,
                high_index,
                low_pivot,
                low_index,
                window,
                min_bars,
                max_bars,
                min_spacing,
                max_candidates,
                float(cfg.triple_peak_tolerance_pct),
                float(cfg.min_triple_neckline_depth_pct),
                float(cfg.min_triple_prior_move_pct),
                float(cfg.min_triple_reaction_score),
            )
            if bottom["quality"] >= float(cfg.min_triple_quality):
                _store_triple(out, row, "triple_bottom", bottom)
    return out


def _empty_triple_output(rows: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in ("triple_top", "triple_bottom"):
        out[f"{name}_quality"] = np.zeros(rows, dtype="float64")
        out[f"{name}_setup_short" if name == "triple_top" else f"{name}_setup_long"] = np.zeros(rows, dtype=bool)
        for field in ("first_index", "second_index", "third_index", "first_price", "second_price", "third_price"):
            out[f"{name}_{field}"] = np.full(rows, np.nan, dtype="float64")
        out[f"{name}_neckline"] = np.full(rows, np.nan, dtype="float64")
        out[f"{name}_neckline_index"] = np.full(rows, np.nan, dtype="float64")
        out[f"{name}_level"] = np.full(rows, np.nan, dtype="float64")
        out[f"{name}_reaction_score"] = np.zeros(rows, dtype="float64")
    return out


def _score_triple_top(
    row: int,
    close: np.ndarray,
    high_pivot: np.ndarray,
    high_index: np.ndarray,
    low_pivot: np.ndarray,
    low_index: np.ndarray,
    window: int,
    min_bars: int,
    max_bars: int,
    min_spacing: int,
    max_candidates: int,
    tolerance_pct: float,
    min_depth: float,
    min_prior_move: float,
    min_reaction_score: float,
) -> dict[str, float]:
    third_x = float(high_index[row])
    third_y = float(high_pivot[row])
    valid = np.isfinite(high_pivot[:row]) & np.isfinite(high_index[:row])
    prior_rows = np.flatnonzero(valid & (high_index[:row] >= third_x - window) & (high_index[:row] < third_x - min_spacing))
    if prior_rows.size > max_candidates:
        prior_rows = prior_rows[-max_candidates:]
    best = _empty_triple_candidate(third_x, third_y)
    for second_row in prior_rows:
        second_x = float(high_index[second_row])
        if third_x - second_x < min_spacing:
            continue
        first_candidates = prior_rows[high_index[prior_rows] < second_x - min_spacing]
        for first_row in first_candidates:
            first_x = float(high_index[first_row])
            span = third_x - first_x
            if span < min_bars or span > max_bars:
                continue
            prices = np.asarray([float(high_pivot[first_row]), float(high_pivot[second_row]), third_y], dtype="float64")
            level = float(np.nanmedian(prices))
            similarity = _same_level_score(prices, level, tolerance_pct)
            if similarity <= 0.0:
                continue
            if _triple_top_overshoot_score(high_pivot, high_index, row, first_x, third_x, level, tolerance_pct) <= 0.0:
                continue
            neckline = _triple_top_neckline(low_pivot, low_index, row, first_x, second_x, third_x)
            if not np.isfinite([neckline["price"], neckline["index"]]).all():
                continue
            depth_pct = (level - neckline["price"]) / max(abs(float(close[row])), 1e-9)
            if depth_pct < min_depth:
                continue
            prior_direction, prior_move_pct = _prior_pattern_move(close, first_x, span)
            if prior_direction <= 0 or prior_move_pct < min_prior_move:
                continue
            reaction_pct = max((third_y - float(close[row])) / max(abs(float(close[row])), 1e-9), 0.0)
            reaction_score = _clip_value(reaction_pct / max(min_depth * 0.65, 1e-9))
            if reaction_score < min_reaction_score:
                continue
            quality = _triple_quality(similarity, depth_pct, min_depth, prior_move_pct, min_prior_move, reaction_score, span, min_bars, max_bars)
            if quality > best["quality"]:
                best = {
                    "quality": quality,
                    "first_index": first_x,
                    "second_index": second_x,
                    "third_index": third_x,
                    "first_price": float(prices[0]),
                    "second_price": float(prices[1]),
                    "third_price": third_y,
                    "neckline": float(neckline["price"]),
                    "neckline_index": float(neckline["index"]),
                    "level": level,
                    "reaction_score": reaction_score,
                }
    return best


def _score_triple_bottom(
    row: int,
    close: np.ndarray,
    high_pivot: np.ndarray,
    high_index: np.ndarray,
    low_pivot: np.ndarray,
    low_index: np.ndarray,
    window: int,
    min_bars: int,
    max_bars: int,
    min_spacing: int,
    max_candidates: int,
    tolerance_pct: float,
    min_depth: float,
    min_prior_move: float,
    min_reaction_score: float,
) -> dict[str, float]:
    third_x = float(low_index[row])
    third_y = float(low_pivot[row])
    valid = np.isfinite(low_pivot[:row]) & np.isfinite(low_index[:row])
    prior_rows = np.flatnonzero(valid & (low_index[:row] >= third_x - window) & (low_index[:row] < third_x - min_spacing))
    if prior_rows.size > max_candidates:
        prior_rows = prior_rows[-max_candidates:]
    best = _empty_triple_candidate(third_x, third_y)
    for second_row in prior_rows:
        second_x = float(low_index[second_row])
        if third_x - second_x < min_spacing:
            continue
        first_candidates = prior_rows[low_index[prior_rows] < second_x - min_spacing]
        for first_row in first_candidates:
            first_x = float(low_index[first_row])
            span = third_x - first_x
            if span < min_bars or span > max_bars:
                continue
            prices = np.asarray([float(low_pivot[first_row]), float(low_pivot[second_row]), third_y], dtype="float64")
            level = float(np.nanmedian(prices))
            similarity = _same_level_score(prices, level, tolerance_pct)
            if similarity <= 0.0:
                continue
            if _triple_bottom_undershoot_score(low_pivot, low_index, row, first_x, third_x, level, tolerance_pct) <= 0.0:
                continue
            neckline = _triple_bottom_neckline(high_pivot, high_index, row, first_x, second_x, third_x)
            if not np.isfinite([neckline["price"], neckline["index"]]).all():
                continue
            depth_pct = (neckline["price"] - level) / max(abs(float(close[row])), 1e-9)
            if depth_pct < min_depth:
                continue
            prior_direction, prior_move_pct = _prior_pattern_move(close, first_x, span)
            if prior_direction >= 0 or prior_move_pct < min_prior_move:
                continue
            reaction_pct = max((float(close[row]) - third_y) / max(abs(float(close[row])), 1e-9), 0.0)
            reaction_score = _clip_value(reaction_pct / max(min_depth * 0.65, 1e-9))
            if reaction_score < min_reaction_score:
                continue
            quality = _triple_quality(similarity, depth_pct, min_depth, prior_move_pct, min_prior_move, reaction_score, span, min_bars, max_bars)
            if quality > best["quality"]:
                best = {
                    "quality": quality,
                    "first_index": first_x,
                    "second_index": second_x,
                    "third_index": third_x,
                    "first_price": float(prices[0]),
                    "second_price": float(prices[1]),
                    "third_price": third_y,
                    "neckline": float(neckline["price"]),
                    "neckline_index": float(neckline["index"]),
                    "level": level,
                    "reaction_score": reaction_score,
                }
    return best


def _empty_triple_candidate(third_x: float, third_y: float) -> dict[str, float]:
    return {
        "quality": 0.0,
        "first_index": np.nan,
        "second_index": np.nan,
        "third_index": third_x,
        "first_price": np.nan,
        "second_price": np.nan,
        "third_price": third_y,
        "neckline": np.nan,
        "neckline_index": np.nan,
        "level": np.nan,
        "reaction_score": 0.0,
    }


def _same_level_score(prices: np.ndarray, level: float, tolerance_pct: float) -> float:
    reference = max(abs(float(level)), 1e-9)
    max_deviation = float(np.nanmax(np.abs(prices - float(level)))) / reference
    return _clip_value(1.0 - max_deviation / max(float(tolerance_pct), 1e-9))


def _triple_top_overshoot_score(
    high_pivot: np.ndarray,
    high_index: np.ndarray,
    row: int,
    first_x: float,
    third_x: float,
    level: float,
    tolerance_pct: float,
) -> float:
    confirmed = np.arange(len(high_pivot)) <= row
    between = confirmed & np.isfinite(high_pivot) & np.isfinite(high_index) & (high_index >= first_x) & (high_index <= third_x)
    if not between.any():
        return 1.0
    overshoot = max(float(np.nanmax(high_pivot[between])) - float(level), 0.0)
    return _clip_value(1.0 - overshoot / max(abs(float(level)) * float(tolerance_pct), 1e-9))


def _triple_bottom_undershoot_score(
    low_pivot: np.ndarray,
    low_index: np.ndarray,
    row: int,
    first_x: float,
    third_x: float,
    level: float,
    tolerance_pct: float,
) -> float:
    confirmed = np.arange(len(low_pivot)) <= row
    between = confirmed & np.isfinite(low_pivot) & np.isfinite(low_index) & (low_index >= first_x) & (low_index <= third_x)
    if not between.any():
        return 1.0
    undershoot = max(float(level) - float(np.nanmin(low_pivot[between])), 0.0)
    return _clip_value(1.0 - undershoot / max(abs(float(level)) * float(tolerance_pct), 1e-9))


def _triple_top_neckline(
    low_pivot: np.ndarray,
    low_index: np.ndarray,
    row: int,
    first_x: float,
    second_x: float,
    third_x: float,
) -> dict[str, float]:
    confirmed = np.arange(len(low_pivot)) <= row
    left = confirmed & np.isfinite(low_pivot) & np.isfinite(low_index) & (low_index > first_x) & (low_index < second_x)
    right = confirmed & np.isfinite(low_pivot) & np.isfinite(low_index) & (low_index > second_x) & (low_index < third_x)
    if not left.any() or not right.any():
        return {"price": np.nan, "index": np.nan}
    left_row = int(np.flatnonzero(left)[int(np.nanargmin(low_pivot[left]))])
    right_row = int(np.flatnonzero(right)[int(np.nanargmin(low_pivot[right]))])
    prices = np.asarray([float(low_pivot[left_row]), float(low_pivot[right_row])], dtype="float64")
    indexes = np.asarray([float(low_index[left_row]), float(low_index[right_row])], dtype="float64")
    return {"price": float(np.nanmedian(prices)), "index": float(np.nanmean(indexes))}


def _triple_bottom_neckline(
    high_pivot: np.ndarray,
    high_index: np.ndarray,
    row: int,
    first_x: float,
    second_x: float,
    third_x: float,
) -> dict[str, float]:
    confirmed = np.arange(len(high_pivot)) <= row
    left = confirmed & np.isfinite(high_pivot) & np.isfinite(high_index) & (high_index > first_x) & (high_index < second_x)
    right = confirmed & np.isfinite(high_pivot) & np.isfinite(high_index) & (high_index > second_x) & (high_index < third_x)
    if not left.any() or not right.any():
        return {"price": np.nan, "index": np.nan}
    left_row = int(np.flatnonzero(left)[int(np.nanargmax(high_pivot[left]))])
    right_row = int(np.flatnonzero(right)[int(np.nanargmax(high_pivot[right]))])
    prices = np.asarray([float(high_pivot[left_row]), float(high_pivot[right_row])], dtype="float64")
    indexes = np.asarray([float(high_index[left_row]), float(high_index[right_row])], dtype="float64")
    return {"price": float(np.nanmedian(prices)), "index": float(np.nanmean(indexes))}


def _triple_quality(
    similarity: float,
    depth_pct: float,
    min_depth: float,
    prior_move_pct: float,
    min_prior_move: float,
    reaction_score: float,
    span: float,
    min_bars: int,
    max_bars: int,
) -> float:
    span_mid = (float(min_bars) + float(max_bars)) / 2.0
    span_score = _clip_value(1.0 - abs(float(span) - span_mid) / max(span_mid, 1.0))
    depth_score = _clip_value(depth_pct / max(float(min_depth) * 2.2, 1e-9))
    prior_score = _clip_value(prior_move_pct / max(float(min_prior_move) * 2.2, 1e-9))
    return _clip_value(
        0.30 * float(similarity)
        + 0.22 * depth_score
        + 0.16 * prior_score
        + 0.12 * float(reaction_score)
        + 0.20 * span_score
    )


def _store_triple(out: dict[str, np.ndarray], row: int, name: str, candidate: dict[str, float]) -> None:
    out[f"{name}_quality"][row] = float(candidate["quality"])
    setup_key = f"{name}_setup_short" if name == "triple_top" else f"{name}_setup_long"
    out[setup_key][row] = True
    for field in ("first_index", "second_index", "third_index", "first_price", "second_price", "third_price"):
        out[f"{name}_{field}"][row] = float(candidate[field])
    out[f"{name}_neckline"][row] = float(candidate["neckline"])
    out[f"{name}_neckline_index"][row] = float(candidate["neckline_index"])
    out[f"{name}_level"][row] = float(candidate["level"])
    out[f"{name}_reaction_score"][row] = float(candidate["reaction_score"])


__all__ = ["_triple_reversal_columns"]

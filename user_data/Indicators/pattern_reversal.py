from __future__ import annotations

from typing import Any as PatternStructureConfig

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from pattern_common import (
    _clip_value,
    _lifecycle_state_from_events,
    _line_value_at,
    _pattern_geometry_arrays,
    _prior_pattern_move,
    _proof_line_columns,
)


def _double_reversal_columns(frame: DataFrame, cfg: PatternStructureConfig) -> dict[str, Series]:
    """Detect double top / double bottom attention at the second confirmed pivot."""

    p = cfg.output_prefix
    arrays = _double_reversal_arrays(frame, cfg)
    raw_top = pd.Series(arrays["double_top_structure_short"], index=frame.index, dtype="bool").fillna(False)
    raw_bottom = pd.Series(arrays["double_bottom_structure_long"], index=frame.index, dtype="bool").fillna(False)
    top = pd.Series(
        _dedupe_double_events(
            arrays["double_top_structure_short"],
            arrays["double_top_first_index"],
            arrays["double_top_second_index"],
            arrays["double_top_neckline"],
            int(cfg.entry_cooldown_bars),
            float(cfg.double_duplicate_overlap_pct),
            float(cfg.double_duplicate_neckline_tolerance_pct),
        ),
        index=frame.index,
        dtype="bool",
    )
    bottom = pd.Series(
        _dedupe_double_events(
            arrays["double_bottom_structure_long"],
            arrays["double_bottom_first_index"],
            arrays["double_bottom_second_index"],
            arrays["double_bottom_neckline"],
            int(cfg.entry_cooldown_bars),
            float(cfg.double_duplicate_overlap_pct),
            float(cfg.double_duplicate_neckline_tolerance_pct),
        ),
        index=frame.index,
        dtype="bool",
    )
    top_neckline_score = pd.Series(arrays["double_top_neckline_score"], index=frame.index, dtype="float64")
    bottom_neckline_score = pd.Series(arrays["double_bottom_neckline_score"], index=frame.index, dtype="float64")
    top_reaction_score = pd.Series(arrays["double_top_reaction_score"], index=frame.index, dtype="float64")
    bottom_reaction_score = pd.Series(arrays["double_bottom_reaction_score"], index=frame.index, dtype="float64")
    top_cleanliness_score = pd.Series(arrays["double_top_between_cleanliness_score"], index=frame.index, dtype="float64")
    bottom_cleanliness_score = pd.Series(arrays["double_bottom_between_cleanliness_score"], index=frame.index, dtype="float64")
    top_quality = pd.Series(arrays["double_top_quality"], index=frame.index, dtype="float64")
    bottom_quality = pd.Series(arrays["double_bottom_quality"], index=frame.index, dtype="float64")
    clean_top = (
        top
        & top_neckline_score.ge(0.55)
        & top_reaction_score.ge(float(cfg.min_double_reaction_score) + 0.20)
        & top_cleanliness_score.ge(float(cfg.min_double_between_cleanliness_score) + 0.20)
        & top_quality.ge(float(cfg.min_double_quality) + 0.06)
    ).fillna(False)
    clean_bottom = (
        bottom
        & bottom_neckline_score.ge(0.55)
        & bottom_reaction_score.ge(float(cfg.min_double_reaction_score) + 0.20)
        & bottom_cleanliness_score.ge(float(cfg.min_double_between_cleanliness_score) + 0.20)
        & bottom_quality.ge(float(cfg.min_double_quality) + 0.06)
    ).fillna(False)
    range_top = (top & ~clean_top).fillna(False)
    range_bottom = (bottom & ~clean_bottom).fillna(False)
    columns = {
        f"{p}_double_top_structure_quality": top_quality.where(raw_top, 0.0),
        f"{p}_double_bottom_structure_quality": bottom_quality.where(raw_bottom, 0.0),
        f"{p}_double_top_quality": top_quality.where(top, 0.0),
        f"{p}_double_bottom_quality": bottom_quality.where(bottom, 0.0),
        f"{p}_double_top_structure_short": raw_top,
        f"{p}_double_bottom_structure_long": raw_bottom,
        f"{p}_double_top_setup_short": top.fillna(False),
        f"{p}_double_bottom_setup_long": bottom.fillna(False),
        f"{p}_double_top_state": pd.Series(
            _lifecycle_state_from_events(
                top,
                int(cfg.pattern_lifecycle_mature_bars),
                int(cfg.pattern_lifecycle_stale_bars),
            ),
            index=frame.index,
            dtype="int8",
        ),
        f"{p}_double_bottom_state": pd.Series(
            _lifecycle_state_from_events(
                bottom,
                int(cfg.pattern_lifecycle_mature_bars),
                int(cfg.pattern_lifecycle_stale_bars),
            ),
            index=frame.index,
            dtype="int8",
        ),
        f"{p}_double_top_clean_short": clean_top,
        f"{p}_double_bottom_clean_long": clean_bottom,
        f"{p}_double_top_range_retest_short": range_top,
        f"{p}_double_bottom_range_retest_long": range_bottom,
        f"{p}_double_top_first_index": pd.Series(arrays["double_top_first_index"], index=frame.index, dtype="float64"),
        f"{p}_double_top_second_index": pd.Series(arrays["double_top_second_index"], index=frame.index, dtype="float64"),
        f"{p}_double_top_first_price": pd.Series(arrays["double_top_first_price"], index=frame.index, dtype="float64"),
        f"{p}_double_top_second_price": pd.Series(arrays["double_top_second_price"], index=frame.index, dtype="float64"),
        f"{p}_double_top_neckline_index": pd.Series(arrays["double_top_neckline_index"], index=frame.index, dtype="float64"),
        f"{p}_double_top_neckline": pd.Series(arrays["double_top_neckline"], index=frame.index, dtype="float64"),
        f"{p}_double_top_neckline_score": top_neckline_score.where(raw_top, 0.0),
        f"{p}_double_top_reaction_score": top_reaction_score.where(raw_top, 0.0),
        f"{p}_double_top_between_cleanliness_score": top_cleanliness_score.where(raw_top, 0.0),
        f"{p}_double_bottom_first_index": pd.Series(arrays["double_bottom_first_index"], index=frame.index, dtype="float64"),
        f"{p}_double_bottom_second_index": pd.Series(arrays["double_bottom_second_index"], index=frame.index, dtype="float64"),
        f"{p}_double_bottom_first_price": pd.Series(arrays["double_bottom_first_price"], index=frame.index, dtype="float64"),
        f"{p}_double_bottom_second_price": pd.Series(arrays["double_bottom_second_price"], index=frame.index, dtype="float64"),
        f"{p}_double_bottom_neckline_index": pd.Series(arrays["double_bottom_neckline_index"], index=frame.index, dtype="float64"),
        f"{p}_double_bottom_neckline": pd.Series(arrays["double_bottom_neckline"], index=frame.index, dtype="float64"),
        f"{p}_double_bottom_neckline_score": bottom_neckline_score.where(raw_bottom, 0.0),
        f"{p}_double_bottom_reaction_score": bottom_reaction_score.where(raw_bottom, 0.0),
        f"{p}_double_bottom_between_cleanliness_score": bottom_cleanliness_score.where(raw_bottom, 0.0),
    }
    if bool(getattr(cfg, "include_pattern_diagnostics", False)):
        columns.update(
            {
                **_proof_line_columns(
                    frame.index,
                    f"{p}_double_top_short",
                    top,
                    {
                        1: (
                            arrays["double_top_first_index"],
                            arrays["double_top_first_price"],
                            arrays["double_top_second_index"],
                            arrays["double_top_second_price"],
                        ),
                        2: (
                            arrays["double_top_first_index"],
                            arrays["double_top_neckline"],
                            arrays["double_top_second_index"],
                            arrays["double_top_neckline"],
                        ),
                    },
                ),
                **_proof_line_columns(
                    frame.index,
                    f"{p}_double_bottom_long",
                    bottom,
                    {
                        1: (
                            arrays["double_bottom_first_index"],
                            arrays["double_bottom_first_price"],
                            arrays["double_bottom_second_index"],
                            arrays["double_bottom_second_price"],
                        ),
                        2: (
                            arrays["double_bottom_first_index"],
                            arrays["double_bottom_neckline"],
                            arrays["double_bottom_second_index"],
                            arrays["double_bottom_neckline"],
                        ),
                    },
                ),
            }
        )
    return columns


def _double_reversal_arrays(frame: DataFrame, cfg: PatternStructureConfig) -> dict[str, np.ndarray]:
    close, _, _, high_pivot, high_index, low_pivot, low_index = _pattern_geometry_arrays(frame, cfg)
    rows = len(frame)
    out = {
        "double_top_quality": np.zeros(rows, dtype="float64"),
        "double_bottom_quality": np.zeros(rows, dtype="float64"),
        "double_top_structure_short": np.zeros(rows, dtype=bool),
        "double_bottom_structure_long": np.zeros(rows, dtype=bool),
        "double_top_setup_short": np.zeros(rows, dtype=bool),
        "double_bottom_setup_long": np.zeros(rows, dtype=bool),
        "double_top_first_index": np.full(rows, np.nan, dtype="float64"),
        "double_top_second_index": np.full(rows, np.nan, dtype="float64"),
        "double_top_first_price": np.full(rows, np.nan, dtype="float64"),
        "double_top_second_price": np.full(rows, np.nan, dtype="float64"),
        "double_top_neckline_index": np.full(rows, np.nan, dtype="float64"),
        "double_top_neckline": np.full(rows, np.nan, dtype="float64"),
        "double_top_neckline_score": np.zeros(rows, dtype="float64"),
        "double_top_reaction_score": np.zeros(rows, dtype="float64"),
        "double_top_between_cleanliness_score": np.zeros(rows, dtype="float64"),
        "double_bottom_first_index": np.full(rows, np.nan, dtype="float64"),
        "double_bottom_second_index": np.full(rows, np.nan, dtype="float64"),
        "double_bottom_first_price": np.full(rows, np.nan, dtype="float64"),
        "double_bottom_second_price": np.full(rows, np.nan, dtype="float64"),
        "double_bottom_neckline_index": np.full(rows, np.nan, dtype="float64"),
        "double_bottom_neckline": np.full(rows, np.nan, dtype="float64"),
        "double_bottom_neckline_score": np.zeros(rows, dtype="float64"),
        "double_bottom_reaction_score": np.zeros(rows, dtype="float64"),
        "double_bottom_between_cleanliness_score": np.zeros(rows, dtype="float64"),
    }
    window = int(cfg.double_pattern_window)
    min_bars = int(cfg.min_double_pattern_bars)
    max_bars = int(cfg.max_double_pattern_bars)
    tolerance_pct = float(cfg.double_peak_tolerance_pct)
    min_depth = float(cfg.min_double_neckline_depth_pct)
    min_neckline_position = float(cfg.double_min_neckline_position)
    min_prior_move = float(cfg.min_double_prior_move_pct)

    for row in range(rows):
        if not np.isfinite(close[row]) or close[row] == 0.0:
            continue
        if np.isfinite(high_pivot[row]) and np.isfinite(high_index[row]):
            top = _score_double_top(
                row,
                close,
                high_pivot,
                high_index,
                low_pivot,
                low_index,
                window,
                min_bars,
                max_bars,
                tolerance_pct,
                min_depth,
                min_neckline_position,
                min_prior_move,
                float(cfg.min_double_reaction_score),
                float(cfg.min_double_between_cleanliness_score),
            )
            if top["quality"] >= float(cfg.min_double_quality):
                out["double_top_quality"][row] = float(top["quality"])
                out["double_top_structure_short"][row] = True
                out["double_top_setup_short"][row] = True
                out["double_top_first_index"][row] = float(top["first_index"])
                out["double_top_second_index"][row] = float(top["second_index"])
                out["double_top_first_price"][row] = float(top["first_price"])
                out["double_top_second_price"][row] = float(top["second_price"])
                out["double_top_neckline_index"][row] = float(top["neckline_index"])
                out["double_top_neckline"][row] = float(top["neckline"])
                out["double_top_neckline_score"][row] = float(top["neckline_score"])
                out["double_top_reaction_score"][row] = float(top["reaction_score"])
                out["double_top_between_cleanliness_score"][row] = float(top["between_cleanliness_score"])
        if np.isfinite(low_pivot[row]) and np.isfinite(low_index[row]):
            bottom = _score_double_bottom(
                row,
                close,
                high_pivot,
                high_index,
                low_pivot,
                low_index,
                window,
                min_bars,
                max_bars,
                tolerance_pct,
                min_depth,
                min_neckline_position,
                min_prior_move,
                float(cfg.min_double_reaction_score),
                float(cfg.min_double_between_cleanliness_score),
            )
            if bottom["quality"] >= float(cfg.min_double_quality):
                out["double_bottom_quality"][row] = float(bottom["quality"])
                out["double_bottom_structure_long"][row] = True
                out["double_bottom_setup_long"][row] = True
                out["double_bottom_first_index"][row] = float(bottom["first_index"])
                out["double_bottom_second_index"][row] = float(bottom["second_index"])
                out["double_bottom_first_price"][row] = float(bottom["first_price"])
                out["double_bottom_second_price"][row] = float(bottom["second_price"])
                out["double_bottom_neckline_index"][row] = float(bottom["neckline_index"])
                out["double_bottom_neckline"][row] = float(bottom["neckline"])
                out["double_bottom_neckline_score"][row] = float(bottom["neckline_score"])
                out["double_bottom_reaction_score"][row] = float(bottom["reaction_score"])
                out["double_bottom_between_cleanliness_score"][row] = float(bottom["between_cleanliness_score"])
    return out


def _score_double_top(
    row: int,
    close: np.ndarray,
    high_pivot: np.ndarray,
    high_index: np.ndarray,
    low_pivot: np.ndarray,
    low_index: np.ndarray,
    window: int,
    min_bars: int,
    max_bars: int,
    tolerance_pct: float,
    min_depth: float,
    min_neckline_position: float,
    min_prior_move: float,
    min_reaction_score: float,
    min_between_cleanliness_score: float,
) -> dict[str, float]:
    second_x = float(high_index[row])
    second_y = float(high_pivot[row])
    high_mask = np.isfinite(high_pivot[:row]) & np.isfinite(high_index[:row])
    candidates = np.flatnonzero(high_mask & (high_index[:row] < second_x - min_bars) & (high_index[:row] >= second_x - window))
    best = {
        "quality": 0.0,
        "first_index": np.nan,
        "second_index": second_x,
        "first_price": np.nan,
        "second_price": second_y,
        "neckline_index": np.nan,
        "neckline": np.nan,
        "neckline_score": 0.0,
        "reaction_score": 0.0,
        "between_cleanliness_score": 0.0,
    }
    for candidate in candidates:
        first_x = float(high_index[candidate])
        span = second_x - first_x
        if span < min_bars or span > max_bars:
            continue
        first_y = float(high_pivot[candidate])
        peak_ref = max((first_y + second_y) / 2.0, 1e-9)
        similarity = 1.0 - abs(second_y - first_y) / max(peak_ref * tolerance_pct, 1e-9)
        if similarity <= 0.0:
            continue
        between_cleanliness_score = _double_top_between_cleanliness_score(
            high_pivot,
            high_index,
            row,
            first_x,
            second_x,
            max(first_y, second_y),
            peak_ref,
            tolerance_pct,
        )
        if between_cleanliness_score < float(min_between_cleanliness_score):
            continue
        confirmed_rows = np.arange(len(low_pivot)) <= row
        between = confirmed_rows & np.isfinite(low_pivot) & np.isfinite(low_index) & (low_index > first_x) & (low_index < second_x)
        if not between.any():
            continue
        neckline_rows = np.flatnonzero(between)
        neckline_row = int(neckline_rows[int(np.nanargmin(low_pivot[neckline_rows]))])
        neckline = float(low_pivot[neckline_row])
        neckline_index = float(low_index[neckline_row])
        neckline_position = (neckline_index - first_x) / max(span, 1e-9)
        if neckline_position < min_neckline_position or neckline_position > 1.0 - min_neckline_position:
            continue
        neckline_score = _double_neckline_score(neckline_position, min_neckline_position)
        depth_pct = (peak_ref - neckline) / max(abs(close[row]), 1e-9)
        if depth_pct < min_depth:
            continue
        prior_direction, prior_move_pct = _prior_pattern_move(close, first_x, span)
        if prior_direction <= 0 or prior_move_pct < min_prior_move:
            continue
        turn_pct = max((second_y - float(close[row])) / max(abs(float(close[row])), 1e-9), 0.0)
        reaction_score = _clip_value(turn_pct / max(min_depth * 0.75, 1e-9))
        if reaction_score < float(min_reaction_score):
            continue
        quality = _double_quality(
            similarity,
            depth_pct,
            min_depth,
            prior_move_pct,
            min_prior_move,
            turn_pct,
            neckline_score,
            between_cleanliness_score,
            span,
            min_bars,
            max_bars,
        )
        if quality > best["quality"]:
            best = {
                "quality": quality,
                "first_index": first_x,
                "second_index": second_x,
                "first_price": first_y,
                "second_price": second_y,
                "neckline_index": neckline_index,
                "neckline": neckline,
                "neckline_score": neckline_score,
                "reaction_score": reaction_score,
                "between_cleanliness_score": between_cleanliness_score,
            }
    return best


def _score_double_bottom(
    row: int,
    close: np.ndarray,
    high_pivot: np.ndarray,
    high_index: np.ndarray,
    low_pivot: np.ndarray,
    low_index: np.ndarray,
    window: int,
    min_bars: int,
    max_bars: int,
    tolerance_pct: float,
    min_depth: float,
    min_neckline_position: float,
    min_prior_move: float,
    min_reaction_score: float,
    min_between_cleanliness_score: float,
) -> dict[str, float]:
    second_x = float(low_index[row])
    second_y = float(low_pivot[row])
    low_mask = np.isfinite(low_pivot[:row]) & np.isfinite(low_index[:row])
    candidates = np.flatnonzero(low_mask & (low_index[:row] < second_x - min_bars) & (low_index[:row] >= second_x - window))
    best = {
        "quality": 0.0,
        "first_index": np.nan,
        "second_index": second_x,
        "first_price": np.nan,
        "second_price": second_y,
        "neckline_index": np.nan,
        "neckline": np.nan,
        "neckline_score": 0.0,
        "reaction_score": 0.0,
        "between_cleanliness_score": 0.0,
    }
    for candidate in candidates:
        first_x = float(low_index[candidate])
        span = second_x - first_x
        if span < min_bars or span > max_bars:
            continue
        first_y = float(low_pivot[candidate])
        trough_ref = max((first_y + second_y) / 2.0, 1e-9)
        similarity = 1.0 - abs(second_y - first_y) / max(trough_ref * tolerance_pct, 1e-9)
        if similarity <= 0.0:
            continue
        between_cleanliness_score = _double_bottom_between_cleanliness_score(
            low_pivot,
            low_index,
            row,
            first_x,
            second_x,
            min(first_y, second_y),
            trough_ref,
            tolerance_pct,
        )
        if between_cleanliness_score < float(min_between_cleanliness_score):
            continue
        confirmed_rows = np.arange(len(high_pivot)) <= row
        between = confirmed_rows & np.isfinite(high_pivot) & np.isfinite(high_index) & (high_index > first_x) & (high_index < second_x)
        if not between.any():
            continue
        neckline_rows = np.flatnonzero(between)
        neckline_row = int(neckline_rows[int(np.nanargmax(high_pivot[neckline_rows]))])
        neckline = float(high_pivot[neckline_row])
        neckline_index = float(high_index[neckline_row])
        neckline_position = (neckline_index - first_x) / max(span, 1e-9)
        if neckline_position < min_neckline_position or neckline_position > 1.0 - min_neckline_position:
            continue
        neckline_score = _double_neckline_score(neckline_position, min_neckline_position)
        depth_pct = (neckline - trough_ref) / max(abs(close[row]), 1e-9)
        if depth_pct < min_depth:
            continue
        prior_direction, prior_move_pct = _prior_pattern_move(close, first_x, span)
        if prior_direction >= 0 or prior_move_pct < min_prior_move:
            continue
        turn_pct = max((float(close[row]) - second_y) / max(abs(float(close[row])), 1e-9), 0.0)
        reaction_score = _clip_value(turn_pct / max(min_depth * 0.75, 1e-9))
        if reaction_score < float(min_reaction_score):
            continue
        quality = _double_quality(
            similarity,
            depth_pct,
            min_depth,
            prior_move_pct,
            min_prior_move,
            turn_pct,
            neckline_score,
            between_cleanliness_score,
            span,
            min_bars,
            max_bars,
        )
        if quality > best["quality"]:
            best = {
                "quality": quality,
                "first_index": first_x,
                "second_index": second_x,
                "first_price": first_y,
                "second_price": second_y,
                "neckline_index": neckline_index,
                "neckline": neckline,
                "neckline_score": neckline_score,
                "reaction_score": reaction_score,
                "between_cleanliness_score": between_cleanliness_score,
            }
    return best


def _double_top_between_cleanliness_score(
    high_pivot: np.ndarray,
    high_index: np.ndarray,
    row: int,
    first_x: float,
    second_x: float,
    allowed_peak: float,
    peak_ref: float,
    tolerance_pct: float,
) -> float:
    confirmed_rows = np.arange(len(high_pivot)) <= int(row)
    between = confirmed_rows & np.isfinite(high_pivot) & np.isfinite(high_index) & (high_index > first_x) & (high_index < second_x)
    if not between.any():
        return 1.0
    max_between = float(np.nanmax(high_pivot[between]))
    overshoot = max(max_between - float(allowed_peak), 0.0)
    tolerance = max(abs(float(peak_ref)) * float(tolerance_pct) * 0.50, 1e-9)
    return _clip_value(1.0 - overshoot / tolerance)


def _double_bottom_between_cleanliness_score(
    low_pivot: np.ndarray,
    low_index: np.ndarray,
    row: int,
    first_x: float,
    second_x: float,
    allowed_trough: float,
    trough_ref: float,
    tolerance_pct: float,
) -> float:
    confirmed_rows = np.arange(len(low_pivot)) <= int(row)
    between = confirmed_rows & np.isfinite(low_pivot) & np.isfinite(low_index) & (low_index > first_x) & (low_index < second_x)
    if not between.any():
        return 1.0
    min_between = float(np.nanmin(low_pivot[between]))
    undershoot = max(float(allowed_trough) - min_between, 0.0)
    tolerance = max(abs(float(trough_ref)) * float(tolerance_pct) * 0.50, 1e-9)
    return _clip_value(1.0 - undershoot / tolerance)


def _double_quality(
    similarity: float,
    depth_pct: float,
    min_depth: float,
    prior_move_pct: float,
    min_prior_move: float,
    turn_pct: float,
    neckline_score: float,
    between_cleanliness_score: float,
    span: float,
    min_bars: int,
    max_bars: int,
) -> float:
    span_mid = (float(min_bars) + float(max_bars)) / 2.0
    span_score = _clip_value(1.0 - abs(float(span) - span_mid) / max(span_mid, 1.0))
    depth_score = _clip_value(depth_pct / max(min_depth * 2.5, 1e-9))
    prior_score = _clip_value(prior_move_pct / max(min_prior_move * 2.5, 1e-9))
    turn_score = _clip_value(turn_pct / max(min_depth * 0.75, 1e-9))
    return _clip_value(
        0.23 * similarity
        + 0.20 * depth_score
        + 0.14 * prior_score
        + 0.10 * span_score
        + 0.12 * turn_score
        + 0.09 * neckline_score
        + 0.12 * between_cleanliness_score
    )


def _double_neckline_score(position: float, min_position: float) -> float:
    edge_room = min(float(position), 1.0 - float(position))
    return _clip_value((edge_room - float(min_position)) / max(0.5 - float(min_position), 1e-9))


def _dedupe_double_events(
    mask: np.ndarray,
    first_index: np.ndarray,
    second_index: np.ndarray,
    neckline: np.ndarray,
    cooldown_bars: int,
    overlap_pct: float,
    neckline_tolerance_pct: float,
) -> np.ndarray:
    """Suppress repeated double-pattern attention without looking ahead.

    The first confirmed setup is kept. Later setups are dropped when their
    anchor span overlaps a prior kept setup and their neckline describes the
    same level. This prevents one broad top/bottom from emitting several
    strategy-facing attention rows while preserving raw structure diagnostics.
    """

    clean = np.asarray(mask, dtype=bool)
    selected = np.zeros(len(clean), dtype=bool)
    selected_rows: list[int] = []
    cooldown = max(int(cooldown_bars), 1)
    min_overlap = float(overlap_pct)
    neck_tolerance = float(neckline_tolerance_pct)
    for row in np.flatnonzero(clean):
        if not np.isfinite([first_index[row], second_index[row], neckline[row]]).all():
            continue
        duplicate = False
        for prior in reversed(selected_rows):
            if row - prior <= cooldown:
                duplicate = True
                break
            if _same_double_structure(
                first_index[row],
                second_index[row],
                neckline[row],
                first_index[prior],
                second_index[prior],
                neckline[prior],
                min_overlap,
                neck_tolerance,
            ):
                duplicate = True
                break
        if not duplicate:
            selected[row] = True
            selected_rows.append(int(row))
    return selected


def _same_double_structure(
    first_x: float,
    second_x: float,
    neckline: float,
    prior_first_x: float,
    prior_second_x: float,
    prior_neckline: float,
    min_overlap: float,
    neckline_tolerance_pct: float,
) -> bool:
    span = max(float(second_x) - float(first_x), 1.0)
    prior_span = max(float(prior_second_x) - float(prior_first_x), 1.0)
    overlap = min(float(second_x), float(prior_second_x)) - max(float(first_x), float(prior_first_x))
    overlap_ratio = overlap / max(min(span, prior_span), 1.0)
    neckline_ref = max(abs(float(neckline)), abs(float(prior_neckline)), 1e-9)
    neckline_close = abs(float(neckline) - float(prior_neckline)) / neckline_ref <= float(neckline_tolerance_pct)
    return overlap_ratio >= float(min_overlap) and neckline_close


def _head_shoulders_columns(frame: DataFrame, cfg: PatternStructureConfig) -> dict[str, Series]:
    """Detect head-and-shoulders / inverse head-and-shoulders at the right shoulder."""

    p = cfg.output_prefix
    arrays = _head_shoulders_arrays(frame, cfg)
    hs = pd.Series(
        _dedupe_head_shoulders_events(
            arrays["head_shoulders_setup_short"],
            arrays["head_shoulders_left_index"],
            arrays["head_shoulders_head_index"],
            arrays["head_shoulders_right_index"],
            arrays["head_shoulders_neckline_left"],
            arrays["head_shoulders_neckline_right"],
            int(cfg.entry_cooldown_bars),
            float(cfg.double_duplicate_overlap_pct),
            float(cfg.double_duplicate_neckline_tolerance_pct),
        ),
        index=frame.index,
        dtype="bool",
    )
    inverse = pd.Series(
        _dedupe_head_shoulders_events(
            arrays["inverse_head_shoulders_setup_long"],
            arrays["inverse_head_shoulders_left_index"],
            arrays["inverse_head_shoulders_head_index"],
            arrays["inverse_head_shoulders_right_index"],
            arrays["inverse_head_shoulders_neckline_left"],
            arrays["inverse_head_shoulders_neckline_right"],
            int(cfg.entry_cooldown_bars),
            float(cfg.double_duplicate_overlap_pct),
            float(cfg.double_duplicate_neckline_tolerance_pct),
        ),
        index=frame.index,
        dtype="bool",
    )
    hs = _dedupe_setup_series(
        hs,
        int(cfg.entry_cooldown_bars),
    )
    inverse = _dedupe_setup_series(
        inverse,
        int(cfg.entry_cooldown_bars),
    )
    columns = {
        f"{p}_head_shoulders_structure_quality": pd.Series(arrays["head_shoulders_quality"], index=frame.index, dtype="float64").where(
            pd.Series(arrays["head_shoulders_structure_short"], index=frame.index, dtype="bool").fillna(False),
            0.0,
        ),
        f"{p}_inverse_head_shoulders_structure_quality": pd.Series(
            arrays["inverse_head_shoulders_quality"], index=frame.index, dtype="float64"
        ).where(
            pd.Series(arrays["inverse_head_shoulders_structure_long"], index=frame.index, dtype="bool").fillna(False),
            0.0,
        ),
        f"{p}_head_shoulders_quality": pd.Series(arrays["head_shoulders_quality"], index=frame.index, dtype="float64").where(hs, 0.0),
        f"{p}_inverse_head_shoulders_quality": pd.Series(arrays["inverse_head_shoulders_quality"], index=frame.index, dtype="float64").where(inverse, 0.0),
        f"{p}_head_shoulders_shoulder_score": pd.Series(arrays["head_shoulders_shoulder_score"], index=frame.index, dtype="float64"),
        f"{p}_inverse_head_shoulders_shoulder_score": pd.Series(arrays["inverse_head_shoulders_shoulder_score"], index=frame.index, dtype="float64"),
        f"{p}_head_shoulders_head_score": pd.Series(arrays["head_shoulders_head_score"], index=frame.index, dtype="float64"),
        f"{p}_inverse_head_shoulders_head_score": pd.Series(arrays["inverse_head_shoulders_head_score"], index=frame.index, dtype="float64"),
        f"{p}_head_shoulders_time_balance_score": pd.Series(
            arrays["head_shoulders_time_balance_score"], index=frame.index, dtype="float64"
        ),
        f"{p}_inverse_head_shoulders_time_balance_score": pd.Series(
            arrays["inverse_head_shoulders_time_balance_score"], index=frame.index, dtype="float64"
        ),
        f"{p}_head_shoulders_head_position_score": pd.Series(
            arrays["head_shoulders_head_position_score"], index=frame.index, dtype="float64"
        ),
        f"{p}_inverse_head_shoulders_head_position_score": pd.Series(
            arrays["inverse_head_shoulders_head_position_score"], index=frame.index, dtype="float64"
        ),
        f"{p}_head_shoulders_neckline_score": pd.Series(arrays["head_shoulders_neckline_score"], index=frame.index, dtype="float64"),
        f"{p}_inverse_head_shoulders_neckline_score": pd.Series(arrays["inverse_head_shoulders_neckline_score"], index=frame.index, dtype="float64"),
        f"{p}_head_shoulders_neckline_position_score": pd.Series(
            arrays["head_shoulders_neckline_position_score"], index=frame.index, dtype="float64"
        ),
        f"{p}_inverse_head_shoulders_neckline_position_score": pd.Series(
            arrays["inverse_head_shoulders_neckline_position_score"], index=frame.index, dtype="float64"
        ),
        f"{p}_head_shoulders_neckline_clearance_score": pd.Series(
            arrays["head_shoulders_neckline_clearance_score"], index=frame.index, dtype="float64"
        ),
        f"{p}_inverse_head_shoulders_neckline_clearance_score": pd.Series(
            arrays["inverse_head_shoulders_neckline_clearance_score"], index=frame.index, dtype="float64"
        ),
        f"{p}_head_shoulders_neckline_body_respect_ratio": pd.Series(
            arrays["head_shoulders_neckline_body_respect_ratio"], index=frame.index, dtype="float64"
        ),
        f"{p}_inverse_head_shoulders_neckline_body_respect_ratio": pd.Series(
            arrays["inverse_head_shoulders_neckline_body_respect_ratio"], index=frame.index, dtype="float64"
        ),
        f"{p}_head_shoulders_reaction_score": pd.Series(arrays["head_shoulders_reaction_score"], index=frame.index, dtype="float64"),
        f"{p}_inverse_head_shoulders_reaction_score": pd.Series(arrays["inverse_head_shoulders_reaction_score"], index=frame.index, dtype="float64"),
        f"{p}_head_shoulders_structure_short": pd.Series(arrays["head_shoulders_structure_short"], index=frame.index, dtype="bool").fillna(False),
        f"{p}_inverse_head_shoulders_structure_long": pd.Series(
            arrays["inverse_head_shoulders_structure_long"], index=frame.index, dtype="bool"
        ).fillna(False),
        f"{p}_head_shoulders_setup_short": hs.fillna(False),
        f"{p}_inverse_head_shoulders_setup_long": inverse.fillna(False),
        f"{p}_head_shoulders_state": pd.Series(
            _lifecycle_state_from_events(
                hs,
                int(cfg.pattern_lifecycle_mature_bars),
                int(cfg.pattern_lifecycle_stale_bars),
            ),
            index=frame.index,
            dtype="int8",
        ),
        f"{p}_inverse_head_shoulders_state": pd.Series(
            _lifecycle_state_from_events(
                inverse,
                int(cfg.pattern_lifecycle_mature_bars),
                int(cfg.pattern_lifecycle_stale_bars),
            ),
            index=frame.index,
            dtype="int8",
        ),
        f"{p}_head_shoulders_left_index": pd.Series(arrays["head_shoulders_left_index"], index=frame.index, dtype="float64"),
        f"{p}_head_shoulders_head_index": pd.Series(arrays["head_shoulders_head_index"], index=frame.index, dtype="float64"),
        f"{p}_head_shoulders_right_index": pd.Series(arrays["head_shoulders_right_index"], index=frame.index, dtype="float64"),
        f"{p}_head_shoulders_left_price": pd.Series(arrays["head_shoulders_left_price"], index=frame.index, dtype="float64"),
        f"{p}_head_shoulders_head_price": pd.Series(arrays["head_shoulders_head_price"], index=frame.index, dtype="float64"),
        f"{p}_head_shoulders_right_price": pd.Series(arrays["head_shoulders_right_price"], index=frame.index, dtype="float64"),
        f"{p}_head_shoulders_neckline_left": pd.Series(arrays["head_shoulders_neckline_left"], index=frame.index, dtype="float64"),
        f"{p}_head_shoulders_neckline_right": pd.Series(arrays["head_shoulders_neckline_right"], index=frame.index, dtype="float64"),
        f"{p}_inverse_head_shoulders_left_index": pd.Series(arrays["inverse_head_shoulders_left_index"], index=frame.index, dtype="float64"),
        f"{p}_inverse_head_shoulders_head_index": pd.Series(arrays["inverse_head_shoulders_head_index"], index=frame.index, dtype="float64"),
        f"{p}_inverse_head_shoulders_right_index": pd.Series(arrays["inverse_head_shoulders_right_index"], index=frame.index, dtype="float64"),
        f"{p}_inverse_head_shoulders_left_price": pd.Series(arrays["inverse_head_shoulders_left_price"], index=frame.index, dtype="float64"),
        f"{p}_inverse_head_shoulders_head_price": pd.Series(arrays["inverse_head_shoulders_head_price"], index=frame.index, dtype="float64"),
        f"{p}_inverse_head_shoulders_right_price": pd.Series(arrays["inverse_head_shoulders_right_price"], index=frame.index, dtype="float64"),
        f"{p}_inverse_head_shoulders_neckline_left": pd.Series(arrays["inverse_head_shoulders_neckline_left"], index=frame.index, dtype="float64"),
        f"{p}_inverse_head_shoulders_neckline_right": pd.Series(arrays["inverse_head_shoulders_neckline_right"], index=frame.index, dtype="float64"),
    }
    if bool(getattr(cfg, "include_pattern_diagnostics", False)):
        columns.update(
            {
                **_proof_line_columns(
                    frame.index,
                    f"{p}_head_shoulders_short",
                    hs,
                    {
                        1: (
                            arrays["head_shoulders_left_index"],
                            arrays["head_shoulders_left_price"],
                            arrays["head_shoulders_head_index"],
                            arrays["head_shoulders_head_price"],
                        ),
                        2: (
                            arrays["head_shoulders_head_index"],
                            arrays["head_shoulders_head_price"],
                            arrays["head_shoulders_right_index"],
                            arrays["head_shoulders_right_price"],
                        ),
                        3: (
                            arrays["head_shoulders_left_index"],
                            arrays["head_shoulders_neckline_left"],
                            arrays["head_shoulders_right_index"],
                            arrays["head_shoulders_neckline_right"],
                        ),
                    },
                ),
                **_proof_line_columns(
                    frame.index,
                    f"{p}_inverse_head_shoulders_long",
                    inverse,
                    {
                        1: (
                            arrays["inverse_head_shoulders_left_index"],
                            arrays["inverse_head_shoulders_left_price"],
                            arrays["inverse_head_shoulders_head_index"],
                            arrays["inverse_head_shoulders_head_price"],
                        ),
                        2: (
                            arrays["inverse_head_shoulders_head_index"],
                            arrays["inverse_head_shoulders_head_price"],
                            arrays["inverse_head_shoulders_right_index"],
                            arrays["inverse_head_shoulders_right_price"],
                        ),
                        3: (
                            arrays["inverse_head_shoulders_left_index"],
                            arrays["inverse_head_shoulders_neckline_left"],
                            arrays["inverse_head_shoulders_right_index"],
                            arrays["inverse_head_shoulders_neckline_right"],
                        ),
                    },
                ),
            }
        )
    return columns


def _head_shoulders_arrays(frame: DataFrame, cfg: PatternStructureConfig) -> dict[str, np.ndarray]:
    close, body_high, body_low, high_pivot, high_index, low_pivot, low_index = _pattern_geometry_arrays(frame, cfg)
    rows = len(frame)
    names = (
        "head_shoulders",
        "inverse_head_shoulders",
    )
    out: dict[str, np.ndarray] = {}
    for name in names:
        out[f"{name}_quality"] = np.zeros(rows, dtype="float64")
        out[f"{name}_left_index"] = np.full(rows, np.nan, dtype="float64")
        out[f"{name}_head_index"] = np.full(rows, np.nan, dtype="float64")
        out[f"{name}_right_index"] = np.full(rows, np.nan, dtype="float64")
        out[f"{name}_left_price"] = np.full(rows, np.nan, dtype="float64")
        out[f"{name}_head_price"] = np.full(rows, np.nan, dtype="float64")
        out[f"{name}_right_price"] = np.full(rows, np.nan, dtype="float64")
        out[f"{name}_neckline_left"] = np.full(rows, np.nan, dtype="float64")
        out[f"{name}_neckline_right"] = np.full(rows, np.nan, dtype="float64")
        out[f"{name}_shoulder_score"] = np.zeros(rows, dtype="float64")
        out[f"{name}_head_score"] = np.zeros(rows, dtype="float64")
        out[f"{name}_time_balance_score"] = np.zeros(rows, dtype="float64")
        out[f"{name}_head_position_score"] = np.zeros(rows, dtype="float64")
        out[f"{name}_neckline_score"] = np.zeros(rows, dtype="float64")
        out[f"{name}_neckline_position_score"] = np.zeros(rows, dtype="float64")
        out[f"{name}_neckline_clearance_score"] = np.zeros(rows, dtype="float64")
        out[f"{name}_neckline_body_respect_ratio"] = np.zeros(rows, dtype="float64")
        out[f"{name}_reaction_score"] = np.zeros(rows, dtype="float64")
    out["head_shoulders_structure_short"] = np.zeros(rows, dtype=bool)
    out["inverse_head_shoulders_structure_long"] = np.zeros(rows, dtype=bool)
    out["head_shoulders_setup_short"] = np.zeros(rows, dtype=bool)
    out["inverse_head_shoulders_setup_long"] = np.zeros(rows, dtype=bool)

    for row in range(rows):
        if not np.isfinite(close[row]) or close[row] == 0.0:
            continue
        if np.isfinite(high_pivot[row]) and np.isfinite(high_index[row]):
            hs = _score_head_shoulders(
                row,
                close,
                body_high,
                body_low,
                high_pivot,
                high_index,
                low_pivot,
                low_index,
                cfg,
                inverse=False,
            )
            if hs["quality"] >= float(cfg.min_head_shoulders_quality):
                _store_head_shoulders(out, row, "head_shoulders", hs)
                out["head_shoulders_structure_short"][row] = True
        if np.isfinite(low_pivot[row]) and np.isfinite(low_index[row]):
            inv = _score_head_shoulders(
                row,
                close,
                body_high,
                body_low,
                high_pivot,
                high_index,
                low_pivot,
                low_index,
                cfg,
                inverse=True,
            )
            if inv["quality"] >= float(cfg.min_head_shoulders_quality):
                _store_head_shoulders(out, row, "inverse_head_shoulders", inv)
                out["inverse_head_shoulders_structure_long"][row] = True
    _activate_head_shoulders_setups(out, close, body_high, body_low, cfg, name="head_shoulders", inverse=False)
    _activate_head_shoulders_setups(out, close, body_high, body_low, cfg, name="inverse_head_shoulders", inverse=True)
    return out


def _score_head_shoulders(
    row: int,
    close: np.ndarray,
    body_high: np.ndarray,
    body_low: np.ndarray,
    high_pivot: np.ndarray,
    high_index: np.ndarray,
    low_pivot: np.ndarray,
    low_index: np.ndarray,
    cfg: PatternStructureConfig,
    *,
    inverse: bool,
) -> dict[str, float]:
    pivot = low_pivot if inverse else high_pivot
    pivot_index = low_index if inverse else high_index
    neckline_pivot = high_pivot if inverse else low_pivot
    neckline_index = high_index if inverse else low_index
    right_x = float(pivot_index[row])
    right_y = float(pivot[row])
    pivot_mask = np.isfinite(pivot[:row]) & np.isfinite(pivot_index[:row])
    recent = np.flatnonzero(pivot_mask & (pivot_index[:row] < right_x) & (pivot_index[:row] >= right_x - int(cfg.head_shoulders_window)))
    best = _empty_head_shoulders_candidate(right_x, right_y)
    if len(recent) < 2:
        return best

    min_bars = int(cfg.min_head_shoulders_bars)
    max_bars = int(cfg.max_head_shoulders_bars)
    for left_pos in range(max(0, len(recent) - 8), len(recent) - 1):
        left = recent[left_pos]
        for head_pos in range(left_pos + 1, len(recent)):
            head = recent[head_pos]
            left_x = float(pivot_index[left])
            head_x = float(pivot_index[head])
            span = right_x - left_x
            if span < min_bars or span > max_bars or head_x <= left_x or head_x >= right_x:
                continue
            left_time = head_x - left_x
            right_time = right_x - head_x
            time_balance_score = min(left_time, right_time) / max(left_time, right_time, 1e-9)
            if time_balance_score < float(cfg.min_head_shoulders_time_balance_score):
                continue
            head_position = (head_x - left_x) / max(span, 1e-9)
            head_position_score = _clip_value(1.0 - abs(head_position - 0.5) / 0.5)
            if head_position_score < float(cfg.min_head_shoulders_head_position_score):
                continue
            left_y = float(pivot[left])
            head_y = float(pivot[head])
            shoulder_ref = max((abs(left_y) + abs(right_y)) / 2.0, 1e-9)
            shoulder_similarity = 1.0 - abs(right_y - left_y) / max(shoulder_ref * float(cfg.shoulder_tolerance_pct), 1e-9)
            if shoulder_similarity < float(cfg.min_head_shoulders_shoulder_score):
                continue
            if inverse:
                prominence_pct = (min(left_y, right_y) - head_y) / max(abs(close[row]), 1e-9)
                prior_direction_required = -1
            else:
                prominence_pct = (head_y - max(left_y, right_y)) / max(abs(close[row]), 1e-9)
                prior_direction_required = 1
            if prominence_pct < float(cfg.min_head_prominence_pct):
                continue
            confirmed_rows = np.arange(len(neckline_pivot)) <= row
            left_neck_mask = (
                confirmed_rows
                & np.isfinite(neckline_pivot)
                & np.isfinite(neckline_index)
                & (neckline_index > left_x)
                & (neckline_index < head_x)
            )
            right_neck_mask = (
                confirmed_rows
                & np.isfinite(neckline_pivot)
                & np.isfinite(neckline_index)
                & (neckline_index > head_x)
                & (neckline_index < right_x)
            )
            if not left_neck_mask.any() or not right_neck_mask.any():
                continue
            if inverse:
                left_neck_rows = np.flatnonzero(left_neck_mask)
                right_neck_rows = np.flatnonzero(right_neck_mask)
                left_neck_row = int(left_neck_rows[int(np.nanargmax(neckline_pivot[left_neck_rows]))])
                right_neck_row = int(right_neck_rows[int(np.nanargmax(neckline_pivot[right_neck_rows]))])
                neckline_left = float(neckline_pivot[left_neck_row])
                neckline_right = float(neckline_pivot[right_neck_row])
                depth_pct = (min(neckline_left, neckline_right) - head_y) / max(abs(close[row]), 1e-9)
            else:
                left_neck_rows = np.flatnonzero(left_neck_mask)
                right_neck_rows = np.flatnonzero(right_neck_mask)
                left_neck_row = int(left_neck_rows[int(np.nanargmin(neckline_pivot[left_neck_rows]))])
                right_neck_row = int(right_neck_rows[int(np.nanargmin(neckline_pivot[right_neck_rows]))])
                neckline_left = float(neckline_pivot[left_neck_row])
                neckline_right = float(neckline_pivot[right_neck_row])
                depth_pct = (head_y - max(neckline_left, neckline_right)) / max(abs(close[row]), 1e-9)
            if depth_pct < float(cfg.min_head_shoulders_neckline_depth_pct):
                continue
            left_neck_x = float(neckline_index[left_neck_row])
            right_neck_x = float(neckline_index[right_neck_row])
            neckline_position_score = _head_shoulders_neckline_position_score(
                left_x,
                head_x,
                right_x,
                left_neck_x,
                right_neck_x,
            )
            if neckline_position_score < float(cfg.min_head_shoulders_neckline_position_score):
                continue
            neckline_score = _head_shoulders_neckline_score(
                neckline_left,
                neckline_right,
                left_x,
                right_x,
                float(close[row]),
                float(cfg.head_shoulders_max_neckline_slope_pct_per_bar),
            )
            if neckline_score < float(cfg.min_head_shoulders_neckline_score):
                continue
            neckline_clearance_score = _head_shoulders_neckline_clearance_score(
                left_y,
                head_y,
                right_y,
                neckline_left,
                neckline_right,
                left_x,
                head_x,
                right_x,
                float(close[row]),
                float(cfg.min_head_shoulders_neckline_depth_pct),
                inverse=inverse,
            )
            if neckline_clearance_score < float(cfg.min_head_shoulders_neckline_clearance_score):
                continue
            neckline_body_respect_ratio = _head_shoulders_neckline_body_respect_ratio(
                body_high,
                body_low,
                left_x,
                right_x,
                neckline_left,
                neckline_right,
                float(close[row]),
                float(cfg.head_shoulders_neckline_proximity_pct),
                inverse=inverse,
            )
            if neckline_body_respect_ratio < float(cfg.min_head_shoulders_neckline_body_respect_ratio):
                continue
            prior_direction, prior_move_pct = _prior_pattern_move(close, left_x, span)
            if prior_direction != prior_direction_required or prior_move_pct < float(cfg.min_head_shoulders_prior_move_pct):
                continue
            quality = _head_shoulders_quality(
                shoulder_similarity,
                prominence_pct,
                float(cfg.min_head_prominence_pct),
                depth_pct,
                float(cfg.min_head_shoulders_neckline_depth_pct),
                prior_move_pct,
                float(cfg.min_head_shoulders_prior_move_pct),
                neckline_score,
                time_balance_score,
                head_position_score,
                neckline_position_score,
                neckline_clearance_score,
                neckline_body_respect_ratio,
                span,
                min_bars,
                max_bars,
            )
            if quality > best["quality"]:
                best = {
                    "quality": quality,
                    "left_index": left_x,
                    "head_index": head_x,
                    "right_index": right_x,
                    "left_price": left_y,
                    "head_price": head_y,
                    "right_price": right_y,
                    "neckline_left": neckline_left,
                    "neckline_right": neckline_right,
                    "shoulder_score": shoulder_similarity,
                    "head_score": _clip_value(prominence_pct / max(float(cfg.min_head_prominence_pct) * 2.0, 1e-9)),
                    "time_balance_score": time_balance_score,
                    "head_position_score": head_position_score,
                    "neckline_score": neckline_score,
                    "neckline_position_score": neckline_position_score,
                    "neckline_clearance_score": neckline_clearance_score,
                    "neckline_body_respect_ratio": neckline_body_respect_ratio,
                    "reaction_score": 0.0,
                }
    return best


def _dedupe_setup_series(mask: Series, cooldown_bars: int) -> Series:
    clean = mask.fillna(False).astype("bool")
    cooldown = max(int(cooldown_bars), 1)
    if cooldown <= 1:
        return clean
    recent = clean.shift(1, fill_value=False).rolling(cooldown, min_periods=1).max().astype("bool")
    return clean & ~recent


def _dedupe_head_shoulders_events(
    mask: np.ndarray,
    left_index: np.ndarray,
    head_index: np.ndarray,
    right_index: np.ndarray,
    neckline_left: np.ndarray,
    neckline_right: np.ndarray,
    cooldown_bars: int,
    overlap_pct: float,
    neckline_tolerance_pct: float,
) -> np.ndarray:
    clean = np.asarray(mask, dtype=bool)
    selected = np.zeros(len(clean), dtype=bool)
    selected_rows: list[int] = []
    cooldown = max(int(cooldown_bars), 1)
    for row in np.flatnonzero(clean):
        if not np.isfinite([left_index[row], head_index[row], right_index[row], neckline_left[row], neckline_right[row]]).all():
            continue
        duplicate = False
        for prior in reversed(selected_rows):
            if row - prior <= cooldown:
                duplicate = True
                break
            if _same_head_shoulders_structure(
                left_index[row],
                head_index[row],
                right_index[row],
                neckline_left[row],
                neckline_right[row],
                left_index[prior],
                head_index[prior],
                right_index[prior],
                neckline_left[prior],
                neckline_right[prior],
                overlap_pct,
                neckline_tolerance_pct,
            ):
                duplicate = True
                break
        if not duplicate:
            selected[row] = True
            selected_rows.append(int(row))
    return selected


def _same_head_shoulders_structure(
    left_a: float,
    head_a: float,
    right_a: float,
    neckline_left_a: float,
    neckline_right_a: float,
    left_b: float,
    head_b: float,
    right_b: float,
    neckline_left_b: float,
    neckline_right_b: float,
    overlap_pct: float,
    neckline_tolerance_pct: float,
) -> bool:
    span_a = max(float(right_a) - float(left_a), 1e-9)
    span_b = max(float(right_b) - float(left_b), 1e-9)
    overlap = max(0.0, min(float(right_a), float(right_b)) - max(float(left_a), float(left_b)))
    if overlap / max(min(span_a, span_b), 1e-9) < float(overlap_pct):
        return False
    head_close = abs(float(head_a) - float(head_b)) <= max(span_a, span_b) * 0.20
    neckline_a = (float(neckline_left_a) + float(neckline_right_a)) / 2.0
    neckline_b = (float(neckline_left_b) + float(neckline_right_b)) / 2.0
    neckline_ref = max(abs(neckline_a), abs(neckline_b), 1e-9)
    neckline_close = abs(neckline_a - neckline_b) / neckline_ref <= float(neckline_tolerance_pct)
    return bool(head_close and neckline_close)


def _empty_head_shoulders_candidate(right_x: float, right_y: float) -> dict[str, float]:
    return {
        "quality": 0.0,
        "left_index": np.nan,
        "head_index": np.nan,
        "right_index": right_x,
        "left_price": np.nan,
        "head_price": np.nan,
        "right_price": right_y,
        "neckline_left": np.nan,
        "neckline_right": np.nan,
        "shoulder_score": 0.0,
        "head_score": 0.0,
        "time_balance_score": 0.0,
        "head_position_score": 0.0,
        "neckline_score": 0.0,
        "neckline_position_score": 0.0,
        "neckline_clearance_score": 0.0,
        "neckline_body_respect_ratio": 0.0,
        "reaction_score": 0.0,
    }


def _store_head_shoulders(out: dict[str, np.ndarray], row: int, name: str, candidate: dict[str, float]) -> None:
    for field in (
        "quality",
        "left_index",
        "head_index",
        "right_index",
        "left_price",
        "head_price",
        "right_price",
        "neckline_left",
        "neckline_right",
        "shoulder_score",
        "head_score",
        "time_balance_score",
        "head_position_score",
        "neckline_score",
        "neckline_position_score",
        "neckline_clearance_score",
        "neckline_body_respect_ratio",
        "reaction_score",
    ):
        out[f"{name}_{field}"][row] = float(candidate[field])


def _copy_head_shoulders_candidate(out: dict[str, np.ndarray], source_row: int, target_row: int, name: str) -> None:
    for field in (
        "quality",
        "left_index",
        "head_index",
        "right_index",
        "left_price",
        "head_price",
        "right_price",
        "neckline_left",
        "neckline_right",
        "shoulder_score",
        "head_score",
        "time_balance_score",
        "head_position_score",
        "neckline_score",
        "neckline_position_score",
        "neckline_clearance_score",
        "neckline_body_respect_ratio",
        "reaction_score",
    ):
        out[f"{name}_{field}"][target_row] = float(out[f"{name}_{field}"][source_row])


def _activate_head_shoulders_setups(
    out: dict[str, np.ndarray],
    close: np.ndarray,
    body_high: np.ndarray,
    body_low: np.ndarray,
    cfg: PatternStructureConfig,
    *,
    name: str,
    inverse: bool,
) -> None:
    """Emit H&S setup when the right shoulder is confirmed.

    Pattern detection should say "pay attention, this may be H&S", not wait
    for a neckline break. The confirmed right-shoulder row is the earliest
    no-lookahead point where the full shape exists. The old neckline-only setup
    path was too late and missed usable forming patterns; it remains as a
    fallback if the right-shoulder confirmation row has not yet reacted enough.
    """

    rows = len(close)
    structure_key = f"{name}_structure_long" if inverse else f"{name}_structure_short"
    setup_key = f"{name}_setup_long" if inverse else f"{name}_setup_short"
    monitor_bars = max(int(cfg.head_shoulders_setup_monitor_bars), 1)
    proximity_pct = float(cfg.head_shoulders_neckline_proximity_pct)
    structure_rows = np.flatnonzero(out[structure_key])
    for structure_row in structure_rows:
        left_x = float(out[f"{name}_left_index"][structure_row])
        right_x = float(out[f"{name}_right_index"][structure_row])
        neckline_left = float(out[f"{name}_neckline_left"][structure_row])
        neckline_right = float(out[f"{name}_neckline_right"][structure_row])
        right_price = float(out[f"{name}_right_price"][structure_row])
        if not np.isfinite([left_x, right_x, neckline_left, neckline_right, right_price]).all():
            continue
        start = int(structure_row)
        stop = min(rows - 1, start + monitor_bars)
        if np.isfinite(close[start]) and close[start] != 0.0:
            tolerance = abs(float(close[start])) * proximity_pct
            if inverse:
                reaction_score = _clip_value((float(close[start]) - right_price) / max(tolerance, 1e-9))
                if reaction_score >= float(cfg.min_head_shoulders_right_reaction_score):
                    out[setup_key][start] = True
                    out[f"{name}_reaction_score"][start] = reaction_score
                    continue
            else:
                reaction_score = _clip_value((right_price - float(close[start])) / max(tolerance, 1e-9))
                if reaction_score >= float(cfg.min_head_shoulders_right_reaction_score):
                    out[setup_key][start] = True
                    out[f"{name}_reaction_score"][start] = reaction_score
                    continue
        for row in range(start, stop + 1):
            neckline_now = _line_value_at(float(row), left_x, neckline_left, right_x, neckline_right)
            if not np.isfinite(neckline_now) or not np.isfinite(close[row]) or close[row] == 0.0:
                continue
            tolerance = abs(float(close[row])) * proximity_pct
            if inverse:
                approaching_neckline = float(body_high[row]) >= neckline_now - tolerance
                shoulder_bounce = float(close[row]) >= right_price
                reaction_score = _clip_value((float(close[row]) - right_price) / max(tolerance, 1e-9))
                if approaching_neckline and shoulder_bounce and reaction_score >= float(cfg.min_head_shoulders_right_reaction_score):
                    out[setup_key][row] = True
                    _copy_head_shoulders_candidate(out, int(structure_row), row, name)
                    out[f"{name}_reaction_score"][row] = reaction_score
                    break
            else:
                approaching_neckline = float(body_low[row]) <= neckline_now + tolerance
                shoulder_reject = float(close[row]) <= right_price
                reaction_score = _clip_value((right_price - float(close[row])) / max(tolerance, 1e-9))
                if approaching_neckline and shoulder_reject and reaction_score >= float(cfg.min_head_shoulders_right_reaction_score):
                    out[setup_key][row] = True
                    _copy_head_shoulders_candidate(out, int(structure_row), row, name)
                    out[f"{name}_reaction_score"][row] = reaction_score
                    break


def _head_shoulders_quality(
    shoulder_similarity: float,
    prominence_pct: float,
    min_prominence: float,
    depth_pct: float,
    min_depth: float,
    prior_move_pct: float,
    min_prior_move: float,
    neckline_score: float,
    time_balance_score: float,
    head_position_score: float,
    neckline_position_score: float,
    neckline_clearance_score: float,
    neckline_body_respect_ratio: float,
    span: float,
    min_bars: int,
    max_bars: int,
) -> float:
    span_mid = (float(min_bars) + float(max_bars)) / 2.0
    span_score = _clip_value(1.0 - abs(float(span) - span_mid) / max(span_mid, 1.0))
    prominence_score = _clip_value(prominence_pct / max(min_prominence * 2.0, 1e-9))
    depth_score = _clip_value(depth_pct / max(min_depth * 2.5, 1e-9))
    prior_score = _clip_value(prior_move_pct / max(min_prior_move * 2.0, 1e-9))
    return _clip_value(
        0.21 * shoulder_similarity
        + 0.21 * prominence_score
        + 0.16 * depth_score
        + 0.13 * prior_score
        + 0.06 * span_score
        + 0.04 * time_balance_score
        + 0.02 * head_position_score
        + 0.09 * neckline_score
        + 0.03 * neckline_position_score
        + 0.03 * neckline_clearance_score
        + 0.02 * neckline_body_respect_ratio
    )


def _head_shoulders_neckline_position_score(
    left_x: float,
    head_x: float,
    right_x: float,
    left_neck_x: float,
    right_neck_x: float,
) -> float:
    left_span = max(float(head_x) - float(left_x), 1e-9)
    right_span = max(float(right_x) - float(head_x), 1e-9)
    left_pos = (float(left_neck_x) - float(left_x)) / left_span
    right_pos = (float(right_neck_x) - float(head_x)) / right_span
    return min(_interval_midpoint_score(left_pos), _interval_midpoint_score(right_pos))


def _interval_midpoint_score(position: float) -> float:
    if position <= 0.0 or position >= 1.0:
        return 0.0
    return _clip_value(1.0 - abs(float(position) - 0.5) / 0.5)


def _head_shoulders_neckline_body_respect_ratio(
    body_high: np.ndarray,
    body_low: np.ndarray,
    left_x: float,
    right_x: float,
    neckline_left: float,
    neckline_right: float,
    reference_price: float,
    tolerance_pct: float,
    *,
    inverse: bool,
) -> float:
    start = max(int(np.floor(left_x)), 0)
    stop = min(int(np.ceil(right_x)), len(body_high) - 1)
    if stop <= start:
        return 0.0
    rows = np.arange(start, stop + 1)
    neckline = np.array(
        [_line_value_at(float(row), float(left_x), float(neckline_left), float(right_x), float(neckline_right)) for row in rows],
        dtype="float64",
    )
    valid = np.isfinite(neckline)
    if not valid.any():
        return 0.0
    tolerance = max(abs(float(reference_price)) * float(tolerance_pct), 1e-9)
    if inverse:
        respected = np.asarray(body_high[rows], dtype="float64") <= neckline + tolerance
    else:
        respected = np.asarray(body_low[rows], dtype="float64") >= neckline - tolerance
    return float(np.mean(respected[valid]))


def _head_shoulders_neckline_clearance_score(
    left_y: float,
    head_y: float,
    right_y: float,
    neckline_left: float,
    neckline_right: float,
    left_x: float,
    head_x: float,
    right_x: float,
    reference_price: float,
    min_depth: float,
    *,
    inverse: bool,
) -> float:
    neckline_at_head = _line_value_at(float(head_x), float(left_x), float(neckline_left), float(right_x), float(neckline_right))
    if not np.isfinite(neckline_at_head):
        return 0.0
    reference = max(abs(float(reference_price)), 1e-9)
    if inverse:
        clearances = (
            float(neckline_left) - float(left_y),
            float(neckline_at_head) - float(head_y),
            float(neckline_right) - float(right_y),
        )
    else:
        clearances = (
            float(left_y) - float(neckline_left),
            float(head_y) - float(neckline_at_head),
            float(right_y) - float(neckline_right),
        )
    min_clearance_pct = min(clearances) / reference
    return _clip_value(min_clearance_pct / max(float(min_depth) * 0.75, 1e-9))


def _head_shoulders_neckline_score(
    neckline_left: float,
    neckline_right: float,
    left_x: float,
    right_x: float,
    reference_price: float,
    max_slope_pct_per_bar: float,
) -> float:
    span = max(float(right_x) - float(left_x), 1.0)
    slope_pct = abs(float(neckline_right) - float(neckline_left)) / max(abs(float(reference_price)), 1e-9) / span
    return _clip_value(1.0 - slope_pct / max(float(max_slope_pct_per_bar), 1e-9))

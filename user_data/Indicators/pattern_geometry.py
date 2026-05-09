from __future__ import annotations

from typing import Any as PatternStructureConfig

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from pattern_common import (
    _clip_value,
    _pattern_geometry_arrays,
    _prior_pattern_move,
    _recent_confirmed_pattern_pivots,
)


_GEOMETRY_SLOT_FAMILIES = ("triangle", "wedge", "compression")
_GEOMETRY_SLOT_OUTPUT_FIELDS = (
    "active",
    "type",
    "upper",
    "lower",
    "close_to_upper_atr",
    "close_to_lower_atr",
    "width_atr",
    "age",
    "upper_touch_count",
    "lower_touch_count",
    "upper_fit_error_atr",
    "lower_fit_error_atr",
)
_GEOMETRY_SLOT_INTERNAL_FIELDS = (
    "event_row",
    "event_type",
    "upper_slope",
    "upper_intercept",
    "lower_slope",
    "lower_intercept",
    "start_x",
    "strength_touch_count",
    "strength_fit_error_atr",
    "strength_span",
)


def _triangle_wedge_columns(frame: DataFrame, sequence: dict[str, Series], cfg: PatternStructureConfig) -> dict[str, Series]:
    """Detect triangle and wedge geometry from confirmed pivot boundaries.

    TODO_REVIEW_BEFORE_ACCEPTANCE: Current visual results were accepted as a
    useful first pass, but this module grew through several complex refinement
    loops. Before calling another pattern family finished, remind the user that
    this geometry code still needs a simplicity/code-quality review and may need
    to be simplified like the double/triple peak detectors.

    This path is intentionally independent from the flag/pennant impulse
    detector and from TLV2 trendlines. It does not require breakout
    confirmation; it emits only current pattern evidence for strategy use.

    Output families are triangle, wedge, and compression. Each slot has a
    direction ``type`` where ``1`` is upward, ``-1`` is downward, ``0`` is
    neutral, and ``9`` is unknown.
    """

    p = cfg.output_prefix
    _ = sequence
    arrays = _triangle_wedge_arrays(frame, cfg)
    slot_count = _geometry_slot_count(cfg)
    columns: dict[str, Series] = {}
    for family in _GEOMETRY_SLOT_FAMILIES:
        for slot in range(1, slot_count + 1):
            for field in _GEOMETRY_SLOT_OUTPUT_FIELDS:
                dtype = "bool" if field == "active" else "int8" if field == "type" else "float64"
                series = pd.Series(arrays[f"{family}_slot_{slot}_{field}"], index=frame.index, dtype=dtype)
                columns[f"{p}_{family}_{slot}_{field}"] = series.fillna(False) if field == "active" else series
    return columns


def _geometry_slot_count(cfg: PatternStructureConfig) -> int:
    return min(max(int(getattr(cfg, "geometry_output_slots", 1)), 1), 4)


def _init_geometry_slot_arrays(out: dict[str, np.ndarray], rows: int, prefix: str, slot_count: int) -> None:
    for slot in range(1, slot_count + 1):
        for field in _GEOMETRY_SLOT_OUTPUT_FIELDS + _GEOMETRY_SLOT_INTERNAL_FIELDS:
            key = f"{prefix}_slot_{slot}_{field}"
            if field == "active":
                out[key] = np.zeros(rows, dtype=bool)
            elif field in {"type", "event_type"}:
                out[key] = np.zeros(rows, dtype="int8")
            else:
                out[key] = np.full(rows, np.nan, dtype="float64")


def _triangle_wedge_arrays(frame: DataFrame, cfg: PatternStructureConfig) -> dict[str, np.ndarray]:
    close, body_high, body_low, high_pivot, high_index, low_pivot, low_index = _pattern_geometry_arrays(frame, cfg)
    rows = len(frame)
    atr = _geometry_atr(frame)
    high_pivot, high_index, low_pivot, low_index = _reduce_geometry_pivots(
        high_pivot,
        high_index,
        low_pivot,
        low_index,
        atr,
        cfg,
    )
    out: dict[str, np.ndarray] = {}
    slot_count = _geometry_slot_count(cfg)
    for family in _GEOMETRY_SLOT_FAMILIES:
        _init_geometry_slot_arrays(out, rows, family, slot_count)
    window, min_bars = _effective_geometry_window(frame, cfg)
    min_span = max(int(min_bars), int(round(float(min_bars) * float(cfg.geometry_envelope_min_span_mult))))
    min_side_pivots = int(cfg.min_pattern_side_pivots)
    min_slope = float(cfg.min_boundary_slope_pct_per_bar)
    flat_tolerance = float(cfg.flat_boundary_slope_pct_per_bar)
    min_width = float(cfg.min_triangle_width_pct)
    max_width = float(cfg.max_triangle_width_pct)
    min_containment = float(cfg.min_geometry_containment_ratio)
    min_contraction = float(cfg.min_geometry_contraction_score)
    max_fit_error_atr = float(cfg.max_geometry_fit_error_atr) * float(cfg.geometry_line_fit_tolerance_mult)
    max_body_excursion = float(cfg.max_geometry_body_excursion_pct)
    min_side_switches = int(cfg.min_geometry_side_switches)
    eval_step = max(int(cfg.geometry_envelope_eval_step), 1)
    max_width_atr = float(cfg.geometry_envelope_max_width_atr)
    pivot_events = np.unique(
        np.concatenate(
            [
                high_index[np.isfinite(high_pivot) & np.isfinite(high_index)],
                low_index[np.isfinite(low_pivot) & np.isfinite(low_index)],
            ]
        ).astype("float64")
    )

    for row in range(min_span, rows, eval_step):
        if not np.isfinite(close[row]) or close[row] == 0.0:
            continue
        search_start = max(0, row - window + 1)
        all_high_x, all_high_y, all_low_x, all_low_y = _recent_confirmed_pattern_pivots(
            row,
            search_start,
            high_pivot,
            high_index,
            low_pivot,
            low_index,
        )
        if len(all_high_x) < min_side_pivots or len(all_low_x) < min_side_pivots:
            continue

        family_candidates: dict[str, list[dict[str, float]]] = {family: [] for family in _GEOMETRY_SLOT_FAMILIES}
        for start_anchor in _geometry_envelope_candidate_starts(pivot_events, row, min_span, window, cfg):
            high_mask = all_high_x >= start_anchor
            low_mask = all_low_x >= start_anchor
            high_x = all_high_x[high_mask]
            high_y = _geometry_body_values_at_pivots(high_x, body_high)
            low_x = all_low_x[low_mask]
            low_y = _geometry_body_values_at_pivots(low_x, body_low)
            if len(high_x) < min_side_pivots or len(low_x) < min_side_pivots:
                continue
            start_x = float(max(0.0, round(float(start_anchor))))
            span = float(row) - start_x
            if span < float(min_span):
                continue
            if _pivot_side_switches(high_x, low_x) < min_side_switches:
                continue

            reference = max(abs(float(close[row])), 1e-9)
            scan_start = int(start_x)
            scan_x = np.arange(scan_start, row + 1, dtype="float64")
            segment = slice(scan_start, row + 1)
            atr_reference = max(float(atr[row]), 1e-9)
            upper_options = _geometry_envelope_boundary_options(
                high_x,
                high_y,
                scan_x,
                body_high[segment],
                body_low[segment],
                atr_reference,
                span,
                cfg,
                side="upper",
            )
            lower_options = _geometry_envelope_boundary_options(
                low_x,
                low_y,
                scan_x,
                body_high[segment],
                body_low[segment],
                atr_reference,
                span,
                cfg,
                side="lower",
            )
            if not upper_options or not lower_options:
                continue

            for upper in upper_options:
                for lower in lower_options:
                    high_slope = float(upper["slope"])
                    high_intercept = float(upper["intercept"])
                    low_slope = float(lower["slope"])
                    low_intercept = float(lower["intercept"])
                    upper_now = high_slope * row + high_intercept
                    lower_now = low_slope * row + low_intercept
                    upper_start = high_slope * start_x + high_intercept
                    lower_start = low_slope * start_x + low_intercept
                    if (
                        not np.isfinite([upper_now, lower_now, upper_start, lower_start]).all()
                        or upper_now <= lower_now
                        or upper_start <= lower_start
                    ):
                        continue
                    width_pct = (upper_now - lower_now) / reference
                    width_atr = (upper_now - lower_now) / atr_reference
                    if width_pct < min_width or width_pct > max_width or width_atr > max_width_atr:
                        continue
                    start_width = max(upper_start - lower_start, 1e-9)
                    contraction_score = _clip_value(1.0 - (upper_now - lower_now) / start_width)
                    if contraction_score < min_contraction:
                        continue
                    containment_ratio, body_excursion_pct = _geometry_containment_metrics(
                        body_high,
                        body_low,
                        close,
                        start_x,
                        row,
                        high_slope,
                        high_intercept,
                        low_slope,
                        low_intercept,
                        float(cfg.geometry_boundary_touch_tolerance_pct),
                    )
                    if containment_ratio < min_containment or body_excursion_pct > max_body_excursion:
                        continue
                    boundary_respect_ratio, boundary_intrusion_pct = _geometry_boundary_respect_metrics(
                        body_high,
                        body_low,
                        close,
                        start_x,
                        row,
                        high_slope,
                        high_intercept,
                        low_slope,
                        low_intercept,
                        float(cfg.geometry_boundary_tolerance_pct),
                    )
                    if boundary_respect_ratio < float(cfg.min_geometry_boundary_respect_ratio):
                        continue
                    high_slope_pct = high_slope / reference
                    low_slope_pct = low_slope / reference
                    if low_slope_pct <= high_slope_pct:
                        continue
                    max_abs_slope_pct = max(abs(float(high_slope_pct)), abs(float(low_slope_pct)))
                    if max_abs_slope_pct > float(cfg.max_geometry_boundary_slope_pct_per_bar):
                        continue
                    upper_fit_error_atr = float(upper["fit_error_atr"])
                    lower_fit_error_atr = float(lower["fit_error_atr"])
                    if max(upper_fit_error_atr, lower_fit_error_atr) > max_fit_error_atr:
                        continue
                    min_touch_span_ratio = min(
                        float(upper["touch_span"]),
                        float(lower["touch_span"]),
                    ) / max(span, 1.0)
                    if min_touch_span_ratio < float(cfg.geometry_envelope_min_touch_span_ratio):
                        continue
                    recent_touch_score = min(
                        _clip_value(1.0 - (float(row) - float(upper["last_touch_x"])) / max(span, 1.0)),
                        _clip_value(1.0 - (float(row) - float(lower["last_touch_x"])) / max(span, 1.0)),
                    )
                    if recent_touch_score < float(cfg.min_geometry_recent_touch_score):
                        continue
                    family, pattern_direction = _geometry_envelope_family(
                        high_slope_pct,
                        low_slope_pct,
                        min_slope,
                        flat_tolerance,
                    )
                    if family == "compression" and (
                        min(float(upper["touch_count"]), float(lower["touch_count"]))
                        < float(cfg.min_compression_side_touches)
                        or boundary_intrusion_pct > float(cfg.max_compression_boundary_intrusion_pct)
                        or contraction_score < float(cfg.min_compression_contraction_score)
                    ):
                        continue
                    touch_balance = _clip_value(
                        1.0
                        - abs(float(upper["touch_count"]) - float(lower["touch_count"]))
                        / max(float(upper["touch_count"]) + float(lower["touch_count"]), 1.0)
                    )
                    touch_score = min((float(upper["touch_count"]) + float(lower["touch_count"])) / 8.0, 1.35)
                    fit_score = max(0.0, 1.0 - max(upper_fit_error_atr, lower_fit_error_atr) / max(max_fit_error_atr, 1e-9))
                    width_score = max(0.0, 1.0 - width_atr / max(max_width_atr, 1e-9))
                    score = (
                        _clip_value(span / max(float(window), 1.0)) * 1.55
                        + touch_score * 1.25
                        + fit_score * 1.15
                        + min(contraction_score / 0.45, 1.35) * 0.95
                        + touch_balance * 0.50
                        + width_score * 0.45
                        + min_touch_span_ratio * 0.55
                    )
                    if score < float(cfg.min_geometry_candidate_score):
                        continue
                    family_candidates[family].append(
                        {
                            "type": float(pattern_direction),
                            "upper_now": float(upper_now),
                            "lower_now": float(lower_now),
                            "upper_start": float(upper_start),
                            "lower_start": float(lower_start),
                            "upper_start_x": float(upper["anchor_x"]),
                            "lower_start_x": float(lower["anchor_x"]),
                            "upper_slope": high_slope,
                            "lower_slope": low_slope,
                            "start_x": float(start_x),
                            "span": float(span),
                            "score": float(score),
                            "upper_touch_count": float(upper["touch_count"]),
                            "lower_touch_count": float(lower["touch_count"]),
                            "upper_fit_error_atr": upper_fit_error_atr,
                            "lower_fit_error_atr": lower_fit_error_atr,
                        }
                    )

        for family, candidates in family_candidates.items():
            _store_geometry_identification_slots(
                out,
                row,
                candidates,
                body_high,
                body_low,
                close,
                cfg,
                slot_count,
                prefix=family,
            )
    _project_geometry_slots(out, frame, close, atr, cfg, slot_count)

    return out


def _geometry_envelope_candidate_starts(
    pivot_events: np.ndarray,
    row: int,
    min_span: int,
    window: int,
    cfg: PatternStructureConfig,
) -> np.ndarray:
    lower = float(max(0, int(row) - int(window) + 1))
    upper = float(int(row) - int(min_span))
    starts = np.unique(pivot_events[(pivot_events >= lower) & (pivot_events <= upper)].astype("float64"))
    start_count = max(int(cfg.geometry_envelope_start_count), 1)
    if len(starts) <= start_count:
        return starts
    keep = np.unique(np.linspace(0, len(starts) - 1, start_count, dtype=int))
    return starts[keep]


def _geometry_body_values_at_pivots(x: np.ndarray, body_values: np.ndarray) -> np.ndarray:
    indexes = np.rint(np.asarray(x, dtype="float64")).astype("int64", copy=False)
    out = np.full(len(indexes), np.nan, dtype="float64")
    valid = (indexes >= 0) & (indexes < len(body_values))
    out[valid] = np.asarray(body_values, dtype="float64")[indexes[valid]]
    return out


def _reduce_geometry_pivots(
    high_pivot: np.ndarray,
    high_index: np.ndarray,
    low_pivot: np.ndarray,
    low_index: np.ndarray,
    atr: np.ndarray,
    cfg: PatternStructureConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    high_items = _geometry_pivot_items(high_pivot, high_index)
    low_items = _geometry_pivot_items(low_pivot, low_index)
    high_items = _geometry_dominant_pivot_items(high_items, atr, cfg, side="high")
    low_items = _geometry_dominant_pivot_items(low_items, atr, cfg, side="low")
    high_items = _merge_nearby_geometry_pivots(high_items, atr, cfg, side="high")
    low_items = _merge_nearby_geometry_pivots(low_items, atr, cfg, side="low")
    high_out = np.full_like(high_pivot, np.nan, dtype="float64")
    high_idx_out = np.full_like(high_index, np.nan, dtype="float64")
    low_out = np.full_like(low_pivot, np.nan, dtype="float64")
    low_idx_out = np.full_like(low_index, np.nan, dtype="float64")
    for array_row, pivot_x, price in high_items:
        if 0 <= array_row < len(high_out):
            high_out[array_row] = price
            high_idx_out[array_row] = float(pivot_x)
    for array_row, pivot_x, price in low_items:
        if 0 <= array_row < len(low_out):
            low_out[array_row] = price
            low_idx_out[array_row] = float(pivot_x)
    return high_out, high_idx_out, low_out, low_idx_out


def _geometry_pivot_items(pivot: np.ndarray, index: np.ndarray) -> list[tuple[int, int, float]]:
    mask = np.isfinite(pivot) & np.isfinite(index)
    items = [
        (int(array_row), int(pivot_x), float(price))
        for array_row, pivot_x, price in zip(
            np.where(mask)[0],
            index[mask].astype("int64"),
            pivot[mask].astype("float64"),
        )
    ]
    items.sort(key=lambda item: item[1])
    return items


def _geometry_dominant_pivot_items(
    items: list[tuple[int, int, float]],
    atr: np.ndarray,
    cfg: PatternStructureConfig,
    *,
    side: str,
) -> list[tuple[int, int, float]]:
    dominance_bars = int(cfg.geometry_pivot_dominance_bars)
    if dominance_bars <= 0:
        return items
    kept: list[tuple[int, int, float]] = []
    for array_row, pivot_x, price in items:
        threshold = max(
            float(atr[min(max(array_row, 0), len(atr) - 1)]) * float(cfg.geometry_pivot_dominance_atr_mult),
            0.0,
        )
        left = [other_price for _, other_x, other_price in items if pivot_x - dominance_bars <= other_x < pivot_x]
        right = [other_price for _, other_x, other_price in items if pivot_x < other_x <= pivot_x + dominance_bars]
        if side == "high":
            dominated = bool(left and right and max(left) > price + threshold and max(right) > price + threshold)
        else:
            dominated = bool(left and right and min(left) < price - threshold and min(right) < price - threshold)
        if not dominated:
            kept.append((array_row, pivot_x, price))
    return kept


def _merge_nearby_geometry_pivots(
    items: list[tuple[int, int, float]],
    atr: np.ndarray,
    cfg: PatternStructureConfig,
    *,
    side: str,
) -> list[tuple[int, int, float]]:
    merge_bars = int(cfg.geometry_pivot_merge_bars)
    if merge_bars <= 0 or not items:
        return items
    groups: list[list[tuple[int, int, float]]] = []
    for item in items:
        array_row, pivot_x, price = item
        scale = max(float(atr[min(max(array_row, 0), len(atr) - 1)]) * float(cfg.geometry_pivot_merge_atr_mult), 1e-9)
        for group in groups:
            near_time = min(abs(pivot_x - other_x) for _, other_x, _ in group) <= merge_bars
            near_price = min(abs(price - group_price) for _, _, group_price in group) <= scale
            if near_time and near_price:
                group.append(item)
                break
        else:
            groups.append([item])
    if side == "high":
        merged = [max(group, key=lambda item: (item[2], -item[1])) for group in groups]
    else:
        merged = [min(group, key=lambda item: (item[2], item[1])) for group in groups]
    merged.sort(key=lambda item: item[1])
    return merged


def _geometry_reduced_boundary_points(
    x: np.ndarray,
    y: np.ndarray,
    cfg: PatternStructureConfig,
    *,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    max_points = max(int(cfg.geometry_envelope_max_side_pivots), 2)
    if len(x) <= max_points:
        return x, y
    extreme_order = np.argsort(y)
    if side == "upper":
        extreme_order = extreme_order[::-1]
    selected: list[int] = []

    def add_indexes(indexes: np.ndarray | list[int]) -> None:
        for index in indexes:
            value = int(index)
            if value not in selected:
                selected.append(value)

    add_indexes([0, len(x) - 1])
    add_indexes(extreme_order[: max(3, max_points // 2)])
    add_indexes(np.linspace(0, len(x) - 1, max_points, dtype=int))
    if len(selected) > max_points:
        mandatory = [idx for idx in [0, len(x) - 1] if idx in selected]
        rest = [idx for idx in selected if idx not in mandatory]
        selected = mandatory + rest[: max(max_points - len(mandatory), 0)]
    selected = sorted(selected)
    return x[selected], y[selected]


def _geometry_envelope_boundary_options(
    x: np.ndarray,
    y: np.ndarray,
    scan_x: np.ndarray,
    scan_body_high: np.ndarray,
    scan_body_low: np.ndarray,
    atr: float,
    span: float,
    cfg: PatternStructureConfig,
    *,
    side: str,
) -> list[dict[str, float]]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return []
    xv = x[valid].astype("float64")
    yv = y[valid].astype("float64")
    order = np.argsort(xv)
    xv = xv[order]
    yv = yv[order]
    xv, yv = _geometry_reduced_boundary_points(xv, yv, cfg, side=side)
    min_pair_span = max(float(span) * float(cfg.geometry_envelope_pair_min_span_mult), 2.0)
    touch_tolerance = max(float(atr) * float(cfg.geometry_envelope_touch_atr_mult), 1e-9)
    options: list[dict[str, float]] = []
    for left in range(len(xv) - 1):
        for right in range(left + 1, len(xv)):
            pair_span = float(xv[right] - xv[left])
            if pair_span < min_pair_span:
                continue
            slope = float((yv[right] - yv[left]) / max(pair_span, 1e-9))
            if side == "upper":
                intercept = float(np.nanmax(scan_body_high - slope * scan_x))
                pivot_gap = slope * xv + intercept - yv
                scan_gap = slope * scan_x + intercept - scan_body_high
                anchor_index = int(np.nanargmin(np.abs(scan_gap)))
            else:
                intercept = float(np.nanmin(scan_body_low - slope * scan_x))
                pivot_gap = yv - (slope * xv + intercept)
                scan_gap = scan_body_low - (slope * scan_x + intercept)
                anchor_index = int(np.nanargmin(np.abs(scan_gap)))
            if not np.isfinite(pivot_gap).all() or float(np.nanmin(pivot_gap)) < -1e-7:
                continue
            touch_mask = pivot_gap <= touch_tolerance
            touch_count = int(np.count_nonzero(touch_mask))
            if touch_count < int(cfg.min_pattern_side_pivots):
                continue
            touch_x = xv[touch_mask]
            touch_span = float(np.nanmax(touch_x) - np.nanmin(touch_x)) if len(touch_x) else 0.0
            last_touch_x = float(np.nanmax(touch_x)) if len(touch_x) else float(xv[right])
            fit_error_atr = float(np.nanmean(np.clip(pivot_gap, 0.0, None))) / max(float(atr), 1e-9)
            scan_gap_atr = float(np.nanmean(np.clip(scan_gap, 0.0, None))) / max(float(atr), 1e-9)
            stale_gap_bars = _geometry_longest_stale_proximity_gap(scan_gap, atr, cfg)
            stale_proximity_penalty = _geometry_stale_proximity_penalty(stale_gap_bars, cfg)
            score = (
                min(touch_count / 4.0, 1.5) * 1.35
                + min(touch_span / max(float(span), 1.0), 1.0) * 1.15
                + min(pair_span / max(float(span), 1.0), 1.0) * 0.55
                - fit_error_atr * 0.40
                - scan_gap_atr * 0.10
                - stale_proximity_penalty
            )
            options.append(
                {
                    "slope": slope,
                    "intercept": intercept,
                    "touch_count": float(touch_count),
                    "touch_span": touch_span,
                    "last_touch_x": last_touch_x,
                    "fit_error_atr": fit_error_atr,
                    "stale_proximity_bars": float(stale_gap_bars),
                    "stale_proximity_penalty": float(stale_proximity_penalty),
                    "score": float(score),
                    "anchor_x": float(scan_x[anchor_index]),
                }
            )
    options.sort(key=lambda item: item["score"], reverse=True)
    return options[: max(int(cfg.geometry_envelope_max_pair_options), 1)]


def _geometry_longest_stale_proximity_gap(scan_gap: np.ndarray, atr: float, cfg: PatternStructureConfig) -> int:
    threshold = max(float(atr) * float(cfg.geometry_envelope_proximity_atr_mult), 1e-9)
    gaps = np.asarray(scan_gap, dtype="float64")
    near = np.isfinite(gaps) & (gaps <= threshold)
    longest = 0
    current = 0
    for is_near in near:
        if bool(is_near):
            current = 0
            continue
        current += 1
        longest = max(longest, current)
    return int(longest)


def _geometry_stale_proximity_penalty(stale_gap_bars: int, cfg: PatternStructureConfig) -> float:
    overage = max(int(stale_gap_bars) - int(cfg.geometry_envelope_proximity_grace_bars), 0)
    if overage <= 0:
        return 0.0
    ramp = max(int(cfg.geometry_envelope_proximity_ramp_bars), 1)
    raw_penalty = np.exp(float(overage) / float(ramp)) - 1.0
    return float(cfg.geometry_envelope_proximity_penalty_weight) * min(float(raw_penalty), 12.0)


def _geometry_envelope_family(
    upper_slope_pct: float,
    lower_slope_pct: float,
    min_slope_pct: float,
    flat_tolerance_pct: float,
) -> tuple[str, int]:
    pattern_direction = _geometry_direction_type(upper_slope_pct, lower_slope_pct, min_slope_pct)
    upper_flat = abs(float(upper_slope_pct)) <= float(flat_tolerance_pct)
    lower_flat = abs(float(lower_slope_pct)) <= float(flat_tolerance_pct)
    upper_falling = float(upper_slope_pct) <= -float(min_slope_pct)
    upper_rising = float(upper_slope_pct) >= float(min_slope_pct)
    lower_falling = float(lower_slope_pct) <= -float(min_slope_pct)
    lower_rising = float(lower_slope_pct) >= float(min_slope_pct)
    if (upper_flat and lower_rising) or (upper_falling and lower_flat) or (upper_falling and lower_rising):
        return "triangle", pattern_direction
    if (upper_falling and lower_falling and upper_slope_pct < lower_slope_pct) or (
        upper_rising and lower_rising and upper_slope_pct < lower_slope_pct
    ):
        return "wedge", pattern_direction
    return "compression", pattern_direction


def _store_geometry_identification_slots(
    out: dict[str, np.ndarray],
    row: int,
    candidates: list[dict[str, float]],
    body_high: np.ndarray,
    body_low: np.ndarray,
    close: np.ndarray,
    cfg: PatternStructureConfig,
    slot_count: int,
    prefix: str = "geometry",
) -> None:
    if slot_count <= 0 or not candidates:
        return
    candidates = _merge_similar_geometry_candidates(candidates, close, row, cfg)
    selected: list[dict[str, float]] = []
    for candidate in sorted(candidates, key=_geometry_candidate_sort_key, reverse=True):
        if any(_same_identification_slot(candidate, existing, close, row, cfg) for existing in selected):
            continue
        selected.append(candidate)
        if len(selected) >= slot_count:
            break

    for offset, candidate in enumerate(selected, start=1):
        upper_slope = float(candidate["upper_slope"])
        lower_slope = float(candidate["lower_slope"])
        upper_intercept = float(candidate["upper_now"]) - upper_slope * float(row)
        lower_intercept = float(candidate["lower_now"]) - lower_slope * float(row)
        upper_intercept, lower_intercept = _snap_geometry_boundaries_outward(
            body_high,
            body_low,
            close,
            float(candidate["start_x"]),
            row,
            upper_slope,
            upper_intercept,
            lower_slope,
            lower_intercept,
            float(cfg.geometry_boundary_snap_max_pct),
        )
        out[f"{prefix}_slot_{offset}_event_row"][row] = float(row)
        out[f"{prefix}_slot_{offset}_event_type"][row] = int(candidate.get("type", 0.0))
        out[f"{prefix}_slot_{offset}_upper_slope"][row] = upper_slope
        out[f"{prefix}_slot_{offset}_upper_intercept"][row] = upper_intercept
        out[f"{prefix}_slot_{offset}_lower_slope"][row] = lower_slope
        out[f"{prefix}_slot_{offset}_lower_intercept"][row] = lower_intercept
        out[f"{prefix}_slot_{offset}_start_x"][row] = float(candidate["start_x"])
        out[f"{prefix}_slot_{offset}_upper_touch_count"][row] = float(candidate["upper_touch_count"])
        out[f"{prefix}_slot_{offset}_lower_touch_count"][row] = float(candidate["lower_touch_count"])
        out[f"{prefix}_slot_{offset}_upper_fit_error_atr"][row] = float(candidate["upper_fit_error_atr"])
        out[f"{prefix}_slot_{offset}_lower_fit_error_atr"][row] = float(candidate["lower_fit_error_atr"])
        out[f"{prefix}_slot_{offset}_strength_touch_count"][row] = min(
            float(candidate["upper_touch_count"]),
            float(candidate["lower_touch_count"]),
        )
        out[f"{prefix}_slot_{offset}_strength_fit_error_atr"][row] = max(
            float(candidate["upper_fit_error_atr"]),
            float(candidate["lower_fit_error_atr"]),
        )
        out[f"{prefix}_slot_{offset}_strength_span"][row] = float(candidate.get("span", 0.0))


def _geometry_candidate_sort_key(candidate: dict[str, float]) -> tuple[float, float, float, float]:
    side_touches = min(float(candidate["upper_touch_count"]), float(candidate["lower_touch_count"]))
    fit_error = max(float(candidate["upper_fit_error_atr"]), float(candidate["lower_fit_error_atr"]))
    return float(candidate.get("score", 0.0)), float(candidate.get("span", 0.0)), side_touches, -fit_error


def _merge_similar_geometry_candidates(
    candidates: list[dict[str, float]],
    close: np.ndarray,
    row: int,
    cfg: PatternStructureConfig,
) -> list[dict[str, float]]:
    merged: list[dict[str, float]] = []
    for candidate in sorted(candidates, key=_geometry_candidate_sort_key, reverse=True):
        replacement_index = -1
        for index, existing in enumerate(merged):
            if _same_or_extendable_geometry(candidate, existing, close, row, cfg):
                replacement_index = index
                break
        if replacement_index < 0:
            merged.append(candidate)
            continue
        existing = merged[replacement_index]
        if _geometry_candidate_sort_key(candidate) > _geometry_candidate_sort_key(existing):
            merged[replacement_index] = candidate
    return merged


def _same_or_extendable_geometry(
    candidate: dict[str, float],
    existing: dict[str, float],
    close: np.ndarray,
    row: int,
    cfg: PatternStructureConfig,
) -> bool:
    reference = max(abs(float(close[row])), 1e-9)
    span = max(float(candidate.get("span", 1.0)), float(existing.get("span", 1.0)), 1.0)
    upper_distance = abs(float(candidate["upper_now"]) - float(existing["upper_now"])) / reference
    lower_distance = abs(float(candidate["lower_now"]) - float(existing["lower_now"])) / reference
    upper_slope_distance = abs(float(candidate["upper_slope"]) - float(existing["upper_slope"])) * span / reference
    lower_slope_distance = abs(float(candidate["lower_slope"]) - float(existing["lower_slope"])) * span / reference
    start_overlap = abs(float(candidate["start_x"]) - float(existing["start_x"])) / span
    line_tolerance = float(cfg.geometry_slot_duplicate_price_tolerance_pct) * 1.5
    slope_tolerance = float(cfg.geometry_envelope_merge_slope_tolerance_pct)
    return (
        max(upper_distance, lower_distance) <= line_tolerance
        and max(upper_slope_distance, lower_slope_distance) <= slope_tolerance
        and start_overlap <= max(float(cfg.geometry_slot_duplicate_start_tolerance), 0.55)
    )


def _geometry_direction_type(upper_slope_pct: float, lower_slope_pct: float, min_slope_pct: float) -> int:
    if not np.isfinite([upper_slope_pct, lower_slope_pct]).all():
        return 9
    mid_slope = 0.5 * (float(upper_slope_pct) + float(lower_slope_pct))
    if mid_slope >= float(min_slope_pct):
        return 1
    if mid_slope <= -float(min_slope_pct):
        return -1
    return 0


def _project_geometry_slots(
    out: dict[str, np.ndarray],
    frame: DataFrame,
    close: np.ndarray,
    atr: np.ndarray,
    cfg: PatternStructureConfig,
    slot_count: int,
) -> None:
    rows = len(frame)
    max_age = int(cfg.geometry_management_max_age_bars)
    invalidation_atr = float(cfg.geometry_body_invalidation_atr)
    body_high = np.maximum(
        pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype="float64"),
        pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype="float64"),
    )
    body_low = np.minimum(
        pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype="float64"),
        pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype="float64"),
    )
    for family in _GEOMETRY_SLOT_FAMILIES:
        active_slots: list[dict[str, float] | None] = [None] * slot_count
        for row in range(rows):
            released_indexes: set[int] = set()
            for index, state in enumerate(active_slots):
                if state is None:
                    continue
                if row - int(state["event_row"]) > max_age:
                    active_slots[index] = None
                    released_indexes.add(index)
                    continue
                upper, lower = _geometry_state_lines(state, row)
                scale = max(float(atr[row]), 1e-9)
                if (
                    not np.isfinite(upper)
                    or not np.isfinite(lower)
                    or upper <= lower
                    or not np.isfinite(close[row])
                    or _geometry_projection_invalidated(upper, lower, body_high[row], body_low[row], scale, invalidation_atr)
                ):
                    active_slots[index] = None
                    released_indexes.add(index)

            replacement_allowed = all(state is not None for state in active_slots)
            candidates = []
            for slot in range(1, slot_count + 1):
                event_row = out[f"{family}_slot_{slot}_event_row"][row]
                if np.isfinite(event_row):
                    candidate = _geometry_event_candidate(out, family, slot, row)
                    if candidate is not None:
                        candidates.append(candidate)

            for candidate in sorted(candidates, key=_geometry_state_sort_key, reverse=True):
                upper, lower = _geometry_state_lines(candidate, row)
                scale = max(float(atr[row]), 1e-9)
                if (
                    not np.isfinite(upper)
                    or not np.isfinite(lower)
                    or upper <= lower
                    or not np.isfinite(close[row])
                    or _geometry_projection_invalidated(upper, lower, body_high[row], body_low[row], scale, invalidation_atr)
                ):
                    continue
                if any(
                    state is not None and _same_active_geometry_state(candidate, state, close, row, cfg)
                    for state in active_slots
                ):
                    matching_index = next(
                        idx
                        for idx, state in enumerate(active_slots)
                        if state is not None and _same_active_geometry_state(candidate, state, close, row, cfg)
                    )
                    if _geometry_state_sort_key(candidate) > _geometry_state_sort_key(active_slots[matching_index]):
                        active_slots[matching_index] = candidate
                    continue
                empty_index = next(
                    (idx for idx, state in enumerate(active_slots) if state is None and idx not in released_indexes),
                    -1,
                )
                if empty_index >= 0:
                    active_slots[empty_index] = candidate
                    continue
                if not replacement_allowed or any(state is None for state in active_slots):
                    continue
                weakest_index, weakest_state = min(
                    enumerate(active_slots),
                    key=lambda item: _geometry_state_sort_key(item[1]) if item[1] is not None else (-np.inf, -np.inf, -np.inf),
                )
                if weakest_state is not None and _geometry_state_sort_key(candidate) > _geometry_state_sort_key(weakest_state):
                    active_slots[weakest_index] = candidate

            for index, state in enumerate(active_slots, start=1):
                if state is None:
                    continue
                upper, lower = _geometry_state_lines(state, row)
                scale = max(float(atr[row]), 1e-9)
                out[f"{family}_slot_{index}_active"][row] = True
                out[f"{family}_slot_{index}_type"][row] = int(state["event_type"])
                out[f"{family}_slot_{index}_upper"][row] = upper
                out[f"{family}_slot_{index}_lower"][row] = lower
                out[f"{family}_slot_{index}_close_to_upper_atr"][row] = (upper - float(close[row])) / scale
                out[f"{family}_slot_{index}_close_to_lower_atr"][row] = (float(close[row]) - lower) / scale
                out[f"{family}_slot_{index}_width_atr"][row] = (upper - lower) / scale
                out[f"{family}_slot_{index}_age"][row] = float(row - int(state["event_row"]))
                out[f"{family}_slot_{index}_upper_touch_count"][row] = float(state["upper_touch_count"])
                out[f"{family}_slot_{index}_lower_touch_count"][row] = float(state["lower_touch_count"])
                out[f"{family}_slot_{index}_upper_fit_error_atr"][row] = float(state["upper_fit_error_atr"])
                out[f"{family}_slot_{index}_lower_fit_error_atr"][row] = float(state["lower_fit_error_atr"])


def _geometry_event_candidate(
    out: dict[str, np.ndarray],
    family: str,
    slot: int,
    row: int,
) -> dict[str, float] | None:
    values = {
        "event_row": float(out[f"{family}_slot_{slot}_event_row"][row]),
        "event_type": float(out[f"{family}_slot_{slot}_event_type"][row]),
        "upper_slope": float(out[f"{family}_slot_{slot}_upper_slope"][row]),
        "upper_intercept": float(out[f"{family}_slot_{slot}_upper_intercept"][row]),
        "lower_slope": float(out[f"{family}_slot_{slot}_lower_slope"][row]),
        "lower_intercept": float(out[f"{family}_slot_{slot}_lower_intercept"][row]),
        "start_x": float(out[f"{family}_slot_{slot}_start_x"][row]),
        "upper_touch_count": float(out[f"{family}_slot_{slot}_upper_touch_count"][row]),
        "lower_touch_count": float(out[f"{family}_slot_{slot}_lower_touch_count"][row]),
        "upper_fit_error_atr": float(out[f"{family}_slot_{slot}_upper_fit_error_atr"][row]),
        "lower_fit_error_atr": float(out[f"{family}_slot_{slot}_lower_fit_error_atr"][row]),
        "strength_touch_count": float(out[f"{family}_slot_{slot}_strength_touch_count"][row]),
        "strength_fit_error_atr": float(out[f"{family}_slot_{slot}_strength_fit_error_atr"][row]),
        "strength_span": float(out[f"{family}_slot_{slot}_strength_span"][row]),
    }
    required = [
        values["event_row"],
        values["upper_slope"],
        values["upper_intercept"],
        values["lower_slope"],
        values["lower_intercept"],
        values["start_x"],
    ]
    if not np.isfinite(required).all():
        return None
    return values


def _geometry_state_lines(state: dict[str, float], row: int) -> tuple[float, float]:
    x = float(row)
    upper = float(state["upper_slope"]) * x + float(state["upper_intercept"])
    lower = float(state["lower_slope"]) * x + float(state["lower_intercept"])
    return upper, lower


def _geometry_state_sort_key(state: dict[str, float]) -> tuple[float, float, float]:
    return (
        float(state["strength_touch_count"]),
        -float(state["strength_fit_error_atr"]),
        float(state["strength_span"]),
    )


def _same_active_geometry_state(
    candidate: dict[str, float],
    state: dict[str, float],
    close: np.ndarray,
    row: int,
    cfg: PatternStructureConfig,
) -> bool:
    reference = max(abs(float(close[row])), 1e-9)
    candidate_upper, candidate_lower = _geometry_state_lines(candidate, row)
    state_upper, state_lower = _geometry_state_lines(state, row)
    upper_distance = abs(candidate_upper - state_upper) / reference
    lower_distance = abs(candidate_lower - state_lower) / reference
    span = max(abs(float(candidate.get("strength_span", 1.0))), 1.0)
    start_distance = abs(float(candidate["start_x"]) - float(state["start_x"])) / span
    return (
        max(upper_distance, lower_distance) <= float(cfg.geometry_slot_duplicate_price_tolerance_pct)
        and start_distance <= max(float(cfg.geometry_slot_duplicate_start_tolerance), 0.45)
    )


def _geometry_projection_invalidated(
    upper: float,
    lower: float,
    high: float,
    low: float,
    atr: float,
    invalidation_atr: float,
) -> bool:
    if not np.isfinite([upper, lower, high, low, atr]).all():
        return True
    tolerance = max(float(atr) * float(invalidation_atr), 0.0)
    return float(high) > float(upper) + tolerance or float(low) < float(lower) - tolerance


def _geometry_atr(frame: DataFrame, period: int = 14) -> np.ndarray:
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(int(period), min_periods=1).mean().bfill().fillna(1e-9)
    return atr.clip(lower=1e-9).to_numpy(dtype="float64")


def _same_identification_slot(
    candidate: dict[str, float],
    existing: dict[str, float],
    close: np.ndarray,
    row: int,
    cfg: PatternStructureConfig,
) -> bool:
    reference = max(abs(float(close[row])), 1e-9)
    upper_distance = abs(float(candidate["upper_now"]) - float(existing["upper_now"])) / reference
    lower_distance = abs(float(candidate["lower_now"]) - float(existing["lower_now"])) / reference
    span = max(float(candidate.get("span", 1.0)), 1.0)
    start_distance = abs(float(candidate["start_x"]) - float(existing["start_x"])) / span
    return (
        max(upper_distance, lower_distance) <= float(cfg.geometry_slot_duplicate_price_tolerance_pct)
        and start_distance <= float(cfg.geometry_slot_duplicate_start_tolerance)
    )


def _effective_geometry_window(frame: DataFrame, cfg: PatternStructureConfig) -> tuple[int, int]:
    """Return triangle/wedge lookback bars after optional time normalization.

    The baseline config is still candle-count based. When
    geometry_window_time_scale_power is enabled, shorter candles can expand the
    lookback so a 1h dataframe can inspect a comparable calendar structure to
    the 4h baseline. The factor never shrinks higher timeframes because existing
    1d/3d behaviour was already visually useful in review.
    """

    base_window = int(cfg.triangle_window)
    base_min_bars = int(cfg.min_triangle_bars)
    power = float(getattr(cfg, "geometry_window_time_scale_power", 0.0))
    if power <= 0.0:
        return base_window, base_min_bars
    reference_seconds = max(float(getattr(cfg, "geometry_window_reference_seconds", 14400.0)), 1.0)
    timeframe_seconds = float(getattr(cfg, "geometry_timeframe_seconds", 0.0))
    if timeframe_seconds <= 0.0:
        timeframe_seconds = _infer_dataframe_seconds(frame)
    if timeframe_seconds <= 0.0:
        return base_window, base_min_bars
    max_mult = max(float(getattr(cfg, "geometry_window_max_mult", 4.0)), 1.0)
    scale = min(max((reference_seconds / timeframe_seconds) ** power, 1.0), max_mult)
    window = max(base_window, int(round(float(base_window) * scale)))
    min_bars = max(base_min_bars, int(round(float(base_min_bars) * scale)))
    return max(window, min_bars + 1), min_bars


def _infer_dataframe_seconds(frame: DataFrame) -> float:
    """Infer candle spacing from the first available date deltas."""

    if "date" not in frame:
        return 0.0
    dates = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    diffs = dates.diff().dt.total_seconds()
    positive = diffs[diffs > 0.0].dropna()
    if positive.empty:
        return 0.0
    return float(positive.iloc[: min(len(positive), 8)].median())


def _geometry_candidate_starts(
    high_x: np.ndarray,
    low_x: np.ndarray,
    row: int,
    min_bars: int,
    cfg: PatternStructureConfig,
) -> np.ndarray:
    events = np.concatenate([high_x[np.isfinite(high_x)], low_x[np.isfinite(low_x)]])
    if len(events) == 0:
        return np.array([], dtype="float64")
    events = np.unique(np.sort(events.astype("float64")))
    valid = events <= float(row - min_bars)
    starts = events[valid]
    early_count = int(cfg.geometry_candidate_early_start_count)
    middle_count = int(cfg.geometry_candidate_middle_start_count)
    recent_count = int(cfg.geometry_candidate_recent_start_count)
    sample_count = early_count + middle_count + recent_count
    if len(starts) <= sample_count:
        return starts
    early = starts[:early_count] if early_count else np.array([], dtype="float64")
    recent = starts[-recent_count:] if recent_count else np.array([], dtype="float64")
    middle = starts[early_count : len(starts) - recent_count if recent_count else len(starts)]
    if middle_count and len(middle):
        middle = middle[np.linspace(0, len(middle) - 1, min(middle_count, len(middle)), dtype=int)]
    else:
        middle = np.array([], dtype="float64")
    selected = np.concatenate([early, middle, recent])
    return np.unique(selected).astype("float64")


def _best_boundary_line(
    x: np.ndarray,
    y: np.ndarray,
    row: int,
    reference: float,
    touch_tolerance_pct: float,
    max_violation_pct: float,
    min_bars: int,
    min_span_mult: float,
    min_span_bars: int,
    fit_tolerance_mult: float,
    line_weights: tuple[float, float, float, float],
    *,
    side: str,
) -> dict[str, float] | None:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return None
    xv = x[valid].astype("float64")
    yv = y[valid].astype("float64")
    order = np.argsort(xv)
    xv = xv[order]
    yv = yv[order]
    tolerance = max(abs(float(reference)) * float(touch_tolerance_pct), 1e-9)
    max_violation = max(abs(float(reference)) * float(max_violation_pct), tolerance)
    best: dict[str, float] | None = None
    min_span = max(float(min_bars) * float(min_span_mult), float(min_span_bars))
    touch_weight, fit_weight, span_weight, recency_weight = line_weights
    weight_sum = max(touch_weight + fit_weight + span_weight + recency_weight, 1e-9)
    for left in range(len(xv) - 1):
        for right in range(left + 1, len(xv)):
            x1 = float(xv[left])
            x2 = float(xv[right])
            if x2 - x1 < min_span:
                continue
            y1 = float(yv[left])
            y2 = float(yv[right])
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            line = slope * xv + intercept
            if side == "upper":
                violation = np.maximum(yv - line, 0.0)
                distance = np.maximum(line - yv, 0.0)
            else:
                violation = np.maximum(line - yv, 0.0)
                distance = np.maximum(yv - line, 0.0)
            if float(np.nanmax(violation)) > max_violation:
                continue
            touch_count = int(np.sum(np.abs(yv - line) <= tolerance))
            if touch_count < 2:
                continue
            touch_x = xv[np.abs(yv - line) <= tolerance]
            last_touch_x = float(np.nanmax(touch_x)) if len(touch_x) else x2
            mean_distance = float(np.nanmean(distance))
            mean_distance_pct = mean_distance / max(abs(float(reference)), 1e-9)
            fit_score = _clip_value(1.0 - mean_distance_pct / max(float(touch_tolerance_pct) * fit_tolerance_mult, 1e-9))
            touch_score = _clip_value(float(touch_count) / max(float(len(xv)), 1.0))
            span_score = _clip_value((x2 - x1) / max(float(row) - float(xv[0]), 1.0))
            recency_score = _clip_value(1.0 - (float(row) - x2) / max(float(row) - float(xv[0]), 1.0))
            recent_touch_score = _clip_value(1.0 - (float(row) - last_touch_x) / max(float(row) - float(xv[0]), 1.0))
            score = (
                touch_weight * touch_score
                + fit_weight * fit_score
                + span_weight * span_score
                + recency_weight * recency_score
            ) / weight_sum
            if best is None or score > best["score"]:
                best = {
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "x1": x1,
                    "x2": x2,
                    "touch_count": float(touch_count),
                    "last_touch_x": float(last_touch_x),
                    "mean_distance": mean_distance,
                    "recent_touch_score": float(recent_touch_score),
                    "score": float(score),
                }
    return best


def _pivot_side_switches(high_x: np.ndarray, low_x: np.ndarray) -> int:
    events: list[tuple[float, str]] = [(float(x), "h") for x in high_x if np.isfinite(x)]
    events.extend((float(x), "l") for x in low_x if np.isfinite(x))
    events.sort(key=lambda item: item[0])
    switches = 0
    previous = ""
    for _, side in events:
        if previous and side != previous:
            switches += 1
        previous = side
    return switches


def _geometry_containment_metrics(
    body_high: np.ndarray,
    body_low: np.ndarray,
    close: np.ndarray,
    first_anchor: float,
    row: int,
    upper_slope: float,
    upper_intercept: float,
    lower_slope: float,
    lower_intercept: float,
    tolerance_pct: float,
) -> tuple[float, float]:
    start = int(max(np.floor(first_anchor), 0))
    end = int(min(row, len(close) - 1))
    if end <= start:
        return 0.0, np.inf
    x = np.arange(start, end + 1, dtype="float64")
    reference = np.maximum(np.abs(close[start : end + 1]), 1e-9)
    upper = upper_slope * x + upper_intercept
    lower = lower_slope * x + lower_intercept
    tolerance = reference * float(tolerance_pct)
    upper_excursion = np.maximum(body_high[start : end + 1] - upper, 0.0)
    lower_excursion = np.maximum(lower - body_low[start : end + 1], 0.0)
    inside = (upper_excursion <= tolerance) & (lower_excursion <= tolerance)
    max_excursion_pct = float(np.nanmax(np.maximum(upper_excursion, lower_excursion) / reference))
    return float(np.nanmean(inside.astype("float64"))), max_excursion_pct


def _geometry_boundary_respect_metrics(
    body_high: np.ndarray,
    body_low: np.ndarray,
    close: np.ndarray,
    first_anchor: float,
    row: int,
    upper_slope: float,
    upper_intercept: float,
    lower_slope: float,
    lower_intercept: float,
    tolerance_pct: float,
) -> tuple[float, float]:
    """Measure whether selected rails actually bound candle bodies.

    The broad compression label is experimental. It should not survive
    when a support rail is visibly cutting through bodies or a resistance rail
    sits below body highs. This is deliberately separate from the looser
    containment metric so the broad label can be tightened without changing the
    diagnostic proof-line shape itself.
    """

    start = int(max(np.floor(first_anchor), 0))
    end = int(min(row, len(close) - 1))
    if end <= start:
        return 0.0, np.inf
    x = np.arange(start, end + 1, dtype="float64")
    reference = np.maximum(np.abs(close[start : end + 1]), 1e-9)
    upper = upper_slope * x + upper_intercept
    lower = lower_slope * x + lower_intercept
    tolerance = reference * float(tolerance_pct)
    upper_intrusion = np.maximum(body_high[start : end + 1] - upper, 0.0)
    lower_intrusion = np.maximum(lower - body_low[start : end + 1], 0.0)
    intrusion = np.maximum(upper_intrusion, lower_intrusion)
    respected = intrusion <= tolerance
    max_intrusion_pct = float(np.nanmax(intrusion / reference))
    return float(np.nanmean(respected.astype("float64"))), max_intrusion_pct


def _snap_geometry_boundaries_outward(
    body_high: np.ndarray,
    body_low: np.ndarray,
    close: np.ndarray,
    first_anchor: float,
    row: int,
    upper_slope: float,
    upper_intercept: float,
    lower_slope: float,
    lower_intercept: float,
    max_snap_pct: float,
) -> tuple[float, float]:
    """Nudge fitted rails outward when they slightly cut candle bodies.

    This keeps the selected pivot geometry intact but prevents plotted support
    from sitting just above candle bodies, which made otherwise-valid patterns
    look wrong in visual review. The snap is capped so one noisy candle cannot
    massively widen the pattern.
    """

    start = int(max(np.floor(first_anchor), 0))
    end = int(min(row, len(close) - 1))
    if end <= start or float(max_snap_pct) <= 0.0:
        return float(upper_intercept), float(lower_intercept)
    x = np.arange(start, end + 1, dtype="float64")
    reference = np.maximum(np.abs(close[start : end + 1]), 1e-9)
    snap_cap = float(np.nanmedian(reference)) * float(max_snap_pct)
    upper_line = float(upper_slope) * x + float(upper_intercept)
    lower_line = float(lower_slope) * x + float(lower_intercept)
    upper_intrusion = np.maximum(body_high[start : end + 1] - upper_line, 0.0)
    lower_intrusion = np.maximum(lower_line - body_low[start : end + 1], 0.0)
    upper_shift = min(float(np.nanmax(upper_intrusion)), snap_cap)
    lower_shift = min(float(np.nanmax(lower_intrusion)), snap_cap)
    return float(upper_intercept) + upper_shift, float(lower_intercept) - lower_shift

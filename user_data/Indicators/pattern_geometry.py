from __future__ import annotations

from typing import Any as PatternStructureConfig

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from pattern_common import (
    _clip_value,
    _geometry_boundary_proof_columns,
    _pattern_geometry_arrays,
    _prior_pattern_move,
    _recent_confirmed_pattern_pivots,
)


_GEOMETRY_SLOT_FAMILIES = ("triangle", "wedge", "compression")
_GEOMETRY_SLOT_FIELDS = (
    "quality",
    "upper",
    "lower",
    "upper_start",
    "lower_start",
    "upper_start_index",
    "lower_start_index",
    "start_index",
    "end_index",
)


def _triangle_wedge_columns(frame: DataFrame, sequence: dict[str, Series], cfg: PatternStructureConfig) -> dict[str, Series]:
    """Detect triangle and wedge geometry from confirmed pivot boundaries.

    TODO_REVIEW_BEFORE_ACCEPTANCE: Current visual results were accepted as a
    useful first pass, but this module grew through several complex refinement
    loops. Before calling another pattern family finished, remind the user that
    this geometry code still needs a simplicity/code-quality review and may need
    to be simplified like the double/triple peak detectors.

    This path is intentionally independent from the flag/pennant impulse
    detector and from TLV2 trendlines so it can be split into its own module
    later. It does not require breakout confirmation; it emits attention when
    pivot-envelope boundaries look visually defensible and current price remains
    inside/near the pattern.

    Shape labels are mutually exclusive:
    - Ascending triangle: flat resistance plus rising support.
    - Descending triangle: falling resistance plus flat support.
    - Symmetric triangle: falling resistance plus rising support.
    - Falling wedge: both rails falling, with the upper rail falling faster.
    - Rising wedge: both rails rising, with the lower rail rising faster.
    - Compression: broad contraction that is not clean enough for a named
      triangle/wedge label.
    """

    p = cfg.output_prefix
    _ = sequence
    arrays = _triangle_wedge_arrays(frame, cfg)
    ascending_quality = pd.Series(arrays["triangle_ascending_quality"], index=frame.index, dtype="float64")
    descending_quality = pd.Series(arrays["triangle_descending_quality"], index=frame.index, dtype="float64")
    symmetric_quality = pd.Series(arrays["triangle_symmetric_quality"], index=frame.index, dtype="float64")
    falling_wedge_quality = pd.Series(arrays["wedge_falling_quality"], index=frame.index, dtype="float64")
    rising_wedge_quality = pd.Series(arrays["wedge_rising_quality"], index=frame.index, dtype="float64")
    compression_quality = pd.Series(arrays["compression_quality"], index=frame.index, dtype="float64")
    ascending_raw = pd.Series(arrays["triangle_ascending_setup_long"], index=frame.index, dtype="bool")
    descending_raw = pd.Series(arrays["triangle_descending_setup_short"], index=frame.index, dtype="bool")
    symmetric_long_raw = pd.Series(arrays["triangle_symmetric_setup_long"], index=frame.index, dtype="bool")
    symmetric_short_raw = pd.Series(arrays["triangle_symmetric_setup_short"], index=frame.index, dtype="bool")
    falling_wedge_raw = pd.Series(arrays["wedge_falling_setup_long"], index=frame.index, dtype="bool")
    rising_wedge_raw = pd.Series(arrays["wedge_rising_setup_short"], index=frame.index, dtype="bool")
    compression_long_raw = pd.Series(arrays["compression_setup_long"], index=frame.index, dtype="bool")
    compression_short_raw = pd.Series(arrays["compression_setup_short"], index=frame.index, dtype="bool")
    ascending = _dedupe_geometry_events(ascending_raw, int(cfg.entry_cooldown_bars))
    descending = _dedupe_geometry_events(descending_raw, int(cfg.entry_cooldown_bars))
    symmetric_long = _dedupe_geometry_events(symmetric_long_raw, int(cfg.entry_cooldown_bars))
    symmetric_short = _dedupe_geometry_events(symmetric_short_raw, int(cfg.entry_cooldown_bars))
    falling_wedge = _dedupe_geometry_events(falling_wedge_raw, int(cfg.entry_cooldown_bars))
    rising_wedge = _dedupe_geometry_events(rising_wedge_raw, int(cfg.entry_cooldown_bars))
    compression_long = _dedupe_geometry_events(compression_long_raw, int(cfg.entry_cooldown_bars))
    compression_short = _dedupe_geometry_events(compression_short_raw, int(cfg.entry_cooldown_bars))
    symmetric = symmetric_long | symmetric_short
    triangle_setup_long = (ascending | symmetric_long).fillna(False)
    triangle_setup_short = (descending | symmetric_short).fillna(False)
    triangle_quality_long = pd.concat(
        [ascending_quality.where(ascending, 0.0), symmetric_quality.where(symmetric_long, 0.0)],
        axis=1,
    ).max(axis=1)
    triangle_quality_short = pd.concat(
        [descending_quality.where(descending, 0.0), symmetric_quality.where(symmetric_short, 0.0)],
        axis=1,
    ).max(axis=1)
    wedge_quality_long = falling_wedge_quality.where(falling_wedge, 0.0)
    wedge_quality_short = rising_wedge_quality.where(rising_wedge, 0.0)
    management = _geometry_management_columns(frame, arrays, cfg)
    slot_count = min(max(int(getattr(cfg, "geometry_output_slots", 3)), 0), 3)

    columns = {
        f"{p}_triangle_ascending_quality": ascending_quality.where(ascending, 0.0),
        f"{p}_triangle_descending_quality": descending_quality.where(descending, 0.0),
        f"{p}_triangle_symmetric_quality": symmetric_quality.where(symmetric, 0.0),
        f"{p}_wedge_falling_quality": falling_wedge_quality.where(falling_wedge, 0.0),
        f"{p}_wedge_rising_quality": rising_wedge_quality.where(rising_wedge, 0.0),
        f"{p}_triangle_quality_long": triangle_quality_long,
        f"{p}_triangle_quality_short": triangle_quality_short,
        f"{p}_wedge_quality_long": wedge_quality_long,
        f"{p}_wedge_quality_short": wedge_quality_short,
        f"{p}_compression_quality_long": compression_quality.where(compression_long, 0.0),
        f"{p}_compression_quality_short": compression_quality.where(compression_short, 0.0),
        f"{p}_triangle_ascending_setup_long": ascending.fillna(False),
        f"{p}_triangle_descending_setup_short": descending.fillna(False),
        f"{p}_triangle_symmetric_setup": symmetric.fillna(False),
        f"{p}_wedge_falling_setup_long": falling_wedge.fillna(False),
        f"{p}_wedge_rising_setup_short": rising_wedge.fillna(False),
        f"{p}_triangle_setup_long": triangle_setup_long,
        f"{p}_triangle_setup_short": triangle_setup_short,
        f"{p}_wedge_setup_long": falling_wedge.fillna(False),
        f"{p}_wedge_setup_short": rising_wedge.fillna(False),
        f"{p}_compression_setup_long": compression_long.fillna(False),
        f"{p}_compression_setup_short": compression_short.fillna(False),
        f"{p}_geometry_active": pd.Series(management["active"], index=frame.index, dtype="bool").fillna(False),
        f"{p}_geometry_bias": pd.Series(management["bias"], index=frame.index, dtype="int8"),
        f"{p}_geometry_quality": pd.Series(management["quality"], index=frame.index, dtype="float64"),
        f"{p}_geometry_candidate_count": pd.Series(management["candidate_count"], index=frame.index, dtype="float64"),
        f"{p}_geometry_confirmation_count": pd.Series(
            management["confirmation_count"], index=frame.index, dtype="float64"
        ),
        f"{p}_geometry_upper": pd.Series(management["upper"], index=frame.index, dtype="float64"),
        f"{p}_geometry_lower": pd.Series(management["lower"], index=frame.index, dtype="float64"),
        f"{p}_geometry_width_pct": pd.Series(management["width_pct"], index=frame.index, dtype="float64"),
        f"{p}_geometry_stop_ref_long": pd.Series(management["stop_ref_long"], index=frame.index, dtype="float64"),
        f"{p}_geometry_stop_ref_short": pd.Series(management["stop_ref_short"], index=frame.index, dtype="float64"),
        f"{p}_geometry_target_ref_long": pd.Series(management["target_ref_long"], index=frame.index, dtype="float64"),
        f"{p}_geometry_target_ref_short": pd.Series(management["target_ref_short"], index=frame.index, dtype="float64"),
        f"{p}_geometry_long_bounce_support": pd.Series(
            management["long_bounce_support"], index=frame.index, dtype="bool"
        ).fillna(False),
        f"{p}_geometry_long_breakout": pd.Series(management["long_breakout"], index=frame.index, dtype="bool").fillna(False),
        f"{p}_geometry_go_long": pd.Series(management["go_long"], index=frame.index, dtype="bool").fillna(False),
        f"{p}_geometry_short_reject_resistance": pd.Series(
            management["short_reject_resistance"], index=frame.index, dtype="bool"
        ).fillna(False),
        f"{p}_geometry_short_breakdown": pd.Series(management["short_breakdown"], index=frame.index, dtype="bool").fillna(False),
        f"{p}_geometry_go_short": pd.Series(management["go_short"], index=frame.index, dtype="bool").fillna(False),
        f"{p}_geometry_breakout_up": pd.Series(management["breakout_up"], index=frame.index, dtype="bool").fillna(False),
        f"{p}_geometry_breakdown_down": pd.Series(management["breakdown_down"], index=frame.index, dtype="bool").fillna(False),
        f"{p}_geometry_invalid_long": pd.Series(management["invalid_long"], index=frame.index, dtype="bool").fillna(False),
        f"{p}_geometry_invalid_short": pd.Series(management["invalid_short"], index=frame.index, dtype="bool").fillna(False),
        f"{p}_geometry_exit_long": pd.Series(management["exit_long"], index=frame.index, dtype="bool").fillna(False),
        f"{p}_geometry_exit_short": pd.Series(management["exit_short"], index=frame.index, dtype="bool").fillna(False),
    }
    for slot in range(1, slot_count + 1):
        columns.update(
            {
                f"{p}_geometry_slot_{slot}_quality": pd.Series(
                    arrays[f"geometry_slot_{slot}_quality"], index=frame.index, dtype="float64"
                ),
                f"{p}_geometry_slot_{slot}_upper": pd.Series(
                    arrays[f"geometry_slot_{slot}_upper"], index=frame.index, dtype="float64"
                ),
                f"{p}_geometry_slot_{slot}_lower": pd.Series(
                    arrays[f"geometry_slot_{slot}_lower"], index=frame.index, dtype="float64"
                ),
                f"{p}_geometry_slot_{slot}_start_index": pd.Series(
                    arrays[f"geometry_slot_{slot}_start_index"], index=frame.index, dtype="float64"
                ),
                f"{p}_geometry_slot_{slot}_end_index": pd.Series(
                    arrays[f"geometry_slot_{slot}_end_index"], index=frame.index, dtype="float64"
                ),
            }
        )
    for family in _GEOMETRY_SLOT_FAMILIES:
        family_qualities = []
        for slot in range(1, slot_count + 1):
            slot_quality = pd.Series(arrays[f"{family}_slot_{slot}_quality"], index=frame.index, dtype="float64")
            family_qualities.append(slot_quality)
            columns.update(
                {
                    f"{p}_{family}_slot_{slot}_quality": slot_quality,
                    f"{p}_{family}_slot_{slot}_upper": pd.Series(
                        arrays[f"{family}_slot_{slot}_upper"], index=frame.index, dtype="float64"
                    ),
                    f"{p}_{family}_slot_{slot}_lower": pd.Series(
                        arrays[f"{family}_slot_{slot}_lower"], index=frame.index, dtype="float64"
                    ),
                    f"{p}_{family}_slot_{slot}_start_index": pd.Series(
                        arrays[f"{family}_slot_{slot}_start_index"], index=frame.index, dtype="float64"
                    ),
                    f"{p}_{family}_slot_{slot}_end_index": pd.Series(
                        arrays[f"{family}_slot_{slot}_end_index"], index=frame.index, dtype="float64"
                    ),
                }
            )
        family_quality = (
            pd.concat(family_qualities, axis=1).max(axis=1)
            if family_qualities
            else pd.Series(0.0, index=frame.index, dtype="float64")
        )
        columns[f"{p}_{family}_quality"] = family_quality
        columns[f"{p}_{family}_active"] = family_quality.gt(0.0).fillna(False)
    if bool(getattr(cfg, "include_pattern_diagnostics", False)):
        columns.update(
            {
                f"{p}_geometry_upper_start": pd.Series(arrays["geometry_upper_start"], index=frame.index, dtype="float64"),
                f"{p}_geometry_lower_start": pd.Series(arrays["geometry_lower_start"], index=frame.index, dtype="float64"),
                f"{p}_geometry_start_index": pd.Series(arrays["geometry_start_index"], index=frame.index, dtype="float64"),
                f"{p}_geometry_end_index": pd.Series(arrays["geometry_end_index"], index=frame.index, dtype="float64"),
                f"{p}_range_contraction_score": pd.Series(arrays["range_contraction_score"], index=frame.index, dtype="float64"),
                f"{p}_geometry_recent_touch_score": pd.Series(
                    arrays["geometry_recent_touch_score"], index=frame.index, dtype="float64"
                ),
                f"{p}_geometry_anchor_balance_score": pd.Series(
                    arrays["geometry_anchor_balance_score"], index=frame.index, dtype="float64"
                ),
                f"{p}_geometry_touch_balance_score": pd.Series(
                    arrays["geometry_touch_balance_score"], index=frame.index, dtype="float64"
                ),
                f"{p}_geometry_upper_touch_count": pd.Series(
                    arrays["geometry_upper_touch_count"], index=frame.index, dtype="float64"
                ),
                f"{p}_geometry_lower_touch_count": pd.Series(
                    arrays["geometry_lower_touch_count"], index=frame.index, dtype="float64"
                ),
                f"{p}_geometry_tlv2_source": pd.Series(arrays["geometry_tlv2_source"], index=frame.index, dtype="bool").fillna(False),
                f"{p}_geometry_source_count": pd.Series(arrays["geometry_source_count"], index=frame.index, dtype="float64"),
                f"{p}_geometry_experimental_confluence_score": pd.Series(
                    arrays["geometry_experimental_confluence_score"], index=frame.index, dtype="float64"
                ),
                f"{p}_geometry_experimental_confluence_bonus": pd.Series(
                    arrays["geometry_experimental_confluence_bonus"], index=frame.index, dtype="float64"
                ),
                **_geometry_boundary_proof_columns(frame.index, f"{p}_triangle_long", triangle_setup_long, arrays),
                **_geometry_boundary_proof_columns(frame.index, f"{p}_triangle_short", triangle_setup_short, arrays),
                **_geometry_boundary_proof_columns(frame.index, f"{p}_wedge_long", falling_wedge, arrays),
                **_geometry_boundary_proof_columns(frame.index, f"{p}_wedge_short", rising_wedge, arrays),
                **_geometry_boundary_proof_columns(frame.index, f"{p}_compression_long", compression_long, arrays),
                **_geometry_boundary_proof_columns(frame.index, f"{p}_compression_short", compression_short, arrays),
            }
        )
        for slot in range(1, slot_count + 1):
            slot_active = pd.Series(arrays[f"geometry_slot_{slot}_quality"], index=frame.index, dtype="float64").gt(0.0)
            columns.update(
                _geometry_boundary_proof_columns(
                    frame.index,
                    f"{p}_geometry_slot_{slot}",
                    slot_active,
                    {
                        "geometry_start_index": arrays[f"geometry_slot_{slot}_start_index"],
                        "geometry_upper_start_index": arrays[f"geometry_slot_{slot}_upper_start_index"],
                        "geometry_upper_start": arrays[f"geometry_slot_{slot}_upper_start"],
                        "geometry_upper_anchor_start": arrays[f"geometry_slot_{slot}_upper_start"],
                        "geometry_end_index": arrays[f"geometry_slot_{slot}_end_index"],
                        "geometry_upper": arrays[f"geometry_slot_{slot}_upper"],
                        "geometry_lower_start_index": arrays[f"geometry_slot_{slot}_lower_start_index"],
                        "geometry_lower_start": arrays[f"geometry_slot_{slot}_lower_start"],
                        "geometry_lower_anchor_start": arrays[f"geometry_slot_{slot}_lower_start"],
                        "geometry_lower": arrays[f"geometry_slot_{slot}_lower"],
                    },
                )
            )
        for family in _GEOMETRY_SLOT_FAMILIES:
            for slot in range(1, slot_count + 1):
                slot_active = pd.Series(
                    arrays[f"{family}_slot_{slot}_quality"], index=frame.index, dtype="float64"
                ).gt(0.0)
                columns.update(
                    _geometry_boundary_proof_columns(
                        frame.index,
                        f"{p}_{family}_slot_{slot}",
                        slot_active,
                        {
                            "geometry_start_index": arrays[f"{family}_slot_{slot}_start_index"],
                            "geometry_upper_start_index": arrays[f"{family}_slot_{slot}_upper_start_index"],
                            "geometry_upper_start": arrays[f"{family}_slot_{slot}_upper_start"],
                            "geometry_upper_anchor_start": arrays[f"{family}_slot_{slot}_upper_start"],
                            "geometry_end_index": arrays[f"{family}_slot_{slot}_end_index"],
                            "geometry_upper": arrays[f"{family}_slot_{slot}_upper"],
                            "geometry_lower_start_index": arrays[f"{family}_slot_{slot}_lower_start_index"],
                            "geometry_lower_start": arrays[f"{family}_slot_{slot}_lower_start"],
                            "geometry_lower_anchor_start": arrays[f"{family}_slot_{slot}_lower_start"],
                            "geometry_lower": arrays[f"{family}_slot_{slot}_lower"],
                        },
                    )
                )
    return columns


def _init_geometry_slot_arrays(out: dict[str, np.ndarray], rows: int, prefix: str, slot_count: int) -> None:
    for slot in range(1, slot_count + 1):
        for field in _GEOMETRY_SLOT_FIELDS:
            key = f"{prefix}_slot_{slot}_{field}"
            if field == "quality":
                out[key] = np.zeros(rows, dtype="float64")
            else:
                out[key] = np.full(rows, np.nan, dtype="float64")


def _triangle_wedge_arrays(frame: DataFrame, cfg: PatternStructureConfig) -> dict[str, np.ndarray]:
    close, body_high, body_low, high_pivot, high_index, low_pivot, low_index = _pattern_geometry_arrays(frame, cfg)
    rows = len(frame)
    out = {
        "triangle_ascending_quality": np.zeros(rows, dtype="float64"),
        "triangle_descending_quality": np.zeros(rows, dtype="float64"),
        "triangle_symmetric_quality": np.zeros(rows, dtype="float64"),
        "wedge_falling_quality": np.zeros(rows, dtype="float64"),
        "wedge_rising_quality": np.zeros(rows, dtype="float64"),
        "compression_quality": np.zeros(rows, dtype="float64"),
        "range_contraction_score": np.zeros(rows, dtype="float64"),
        "geometry_recent_touch_score": np.zeros(rows, dtype="float64"),
        "geometry_anchor_balance_score": np.zeros(rows, dtype="float64"),
        "geometry_touch_balance_score": np.zeros(rows, dtype="float64"),
        "geometry_tlv2_source": np.zeros(rows, dtype=bool),
        "geometry_source_count": np.zeros(rows, dtype="float64"),
        "geometry_experimental_confluence_score": np.zeros(rows, dtype="float64"),
        "geometry_experimental_confluence_bonus": np.zeros(rows, dtype="float64"),
        "geometry_bias": np.zeros(rows, dtype="int8"),
        "geometry_quality": np.zeros(rows, dtype="float64"),
        "geometry_candidate_count": np.zeros(rows, dtype="float64"),
        "geometry_confirmation_count": np.zeros(rows, dtype="float64"),
        "geometry_upper_touch_count": np.zeros(rows, dtype="float64"),
        "geometry_lower_touch_count": np.zeros(rows, dtype="float64"),
        "geometry_upper": np.full(rows, np.nan, dtype="float64"),
        "geometry_lower": np.full(rows, np.nan, dtype="float64"),
        "geometry_upper_start": np.full(rows, np.nan, dtype="float64"),
        "geometry_lower_start": np.full(rows, np.nan, dtype="float64"),
        "geometry_upper_anchor_start": np.full(rows, np.nan, dtype="float64"),
        "geometry_lower_anchor_start": np.full(rows, np.nan, dtype="float64"),
        "geometry_upper_start_index": np.full(rows, np.nan, dtype="float64"),
        "geometry_lower_start_index": np.full(rows, np.nan, dtype="float64"),
        "geometry_start_index": np.full(rows, np.nan, dtype="float64"),
        "geometry_end_index": np.full(rows, np.nan, dtype="float64"),
        "geometry_width_pct": np.full(rows, np.nan, dtype="float64"),
        "triangle_ascending_setup_long": np.zeros(rows, dtype=bool),
        "triangle_descending_setup_short": np.zeros(rows, dtype=bool),
        "triangle_symmetric_setup_long": np.zeros(rows, dtype=bool),
        "triangle_symmetric_setup_short": np.zeros(rows, dtype=bool),
        "wedge_falling_setup_long": np.zeros(rows, dtype=bool),
        "wedge_rising_setup_short": np.zeros(rows, dtype=bool),
        "compression_setup_long": np.zeros(rows, dtype=bool),
        "compression_setup_short": np.zeros(rows, dtype=bool),
    }
    slot_count = min(max(int(getattr(cfg, "geometry_output_slots", 3)), 0), 3)
    _init_geometry_slot_arrays(out, rows, "geometry", slot_count)
    for family in _GEOMETRY_SLOT_FAMILIES:
        _init_geometry_slot_arrays(out, rows, family, slot_count)
    window, min_bars = _effective_geometry_window(frame, cfg)
    min_side_pivots = int(cfg.min_pattern_side_pivots)
    min_slope = float(cfg.min_boundary_slope_pct_per_bar)
    flat_tolerance = float(cfg.flat_boundary_slope_pct_per_bar)
    max_fit_error = float(cfg.max_triangle_fit_error_pct)
    min_width = float(cfg.min_triangle_width_pct)
    max_width = float(cfg.max_triangle_width_pct)
    breakout_tolerance = float(cfg.triangle_breakout_tolerance_pct)
    min_shape_score = float(cfg.min_geometry_shape_score)
    min_containment = float(cfg.min_geometry_containment_ratio)
    max_body_excursion = float(cfg.max_geometry_body_excursion_pct)
    min_side_switches = int(cfg.min_geometry_side_switches)
    min_recent_touch = float(cfg.min_geometry_recent_touch_score)
    min_anchor_balance = float(cfg.min_geometry_anchor_balance_score)
    min_touch_balance = float(cfg.min_geometry_touch_balance_score)
    min_compression_quality = _compression_quality_threshold(cfg)
    min_identification_quality = float(getattr(cfg, "min_geometry_identification_quality", min_compression_quality))

    for row in range(rows):
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

        best: dict[str, float] | None = None
        identified_candidates: list[dict[str, float]] = []
        family_candidates: dict[str, list[dict[str, float]]] = {family: [] for family in _GEOMETRY_SLOT_FAMILIES}
        for start_anchor in _geometry_candidate_starts(all_high_x, all_low_x, row, min_bars):
            high_mask = all_high_x >= start_anchor
            low_mask = all_low_x >= start_anchor
            high_x = all_high_x[high_mask]
            high_y = all_high_y[high_mask]
            low_x = all_low_x[low_mask]
            low_y = all_low_y[low_mask]
            if len(high_x) < min_side_pivots or len(low_x) < min_side_pivots:
                continue
            first_anchor = float(min(np.nanmin(high_x), np.nanmin(low_x)))
            last_anchor = float(max(np.nanmax(high_x), np.nanmax(low_x)))
            span = last_anchor - first_anchor
            if span < float(min_bars):
                continue
            if row - last_anchor > max(float(min_bars), float(window) * 0.38):
                continue
            if _pivot_side_switches(high_x, low_x) < min_side_switches:
                continue
            prior_direction, prior_move_pct = _prior_pattern_move(close, first_anchor, span)
            if prior_move_pct < float(cfg.min_triangle_prior_move_pct):
                continue

            reference = max(abs(float(close[row])), 1e-9)
            touch_tolerance_pct = max(float(breakout_tolerance), 0.004)
            upper = _best_boundary_line(
                high_x,
                high_y,
                row,
                reference,
                touch_tolerance_pct,
                max_body_excursion,
                min_bars,
                side="upper",
            )
            lower = _best_boundary_line(
                low_x,
                low_y,
                row,
                reference,
                touch_tolerance_pct,
                max_body_excursion,
                min_bars,
                side="lower",
            )
            if upper is None or lower is None:
                continue
            recent_touch_score = min(float(upper["recent_touch_score"]), float(lower["recent_touch_score"]))
            if recent_touch_score < min_recent_touch:
                continue
            anchor_balance_score = _clip_value(1.0 - abs(float(upper["x1"]) - float(lower["x1"])) / max(span, 1.0))
            touch_balance_score = _clip_value(
                1.0 - abs(float(upper["last_touch_x"]) - float(lower["last_touch_x"])) / max(span, 1.0)
            )
            if anchor_balance_score < min_anchor_balance or touch_balance_score < min_touch_balance:
                continue

            high_slope = upper["slope"]
            high_intercept = upper["intercept"]
            low_slope = lower["slope"]
            low_intercept = lower["intercept"]
            upper_now = high_slope * row + high_intercept
            lower_now = low_slope * row + low_intercept
            if not np.isfinite(upper_now) or not np.isfinite(lower_now) or upper_now <= lower_now:
                continue
            width_pct = (upper_now - lower_now) / reference
            if width_pct < min_width or width_pct > max_width:
                continue
            tolerance = reference * breakout_tolerance
            if float(close[row]) > upper_now + tolerance or float(close[row]) < lower_now - tolerance:
                continue
            start_x = float(min(upper["x1"], lower["x1"]))
            upper_start = high_slope * start_x + high_intercept
            lower_start = low_slope * start_x + low_intercept
            upper_anchor_start = high_slope * float(upper["x1"]) + high_intercept
            lower_anchor_start = low_slope * float(lower["x1"]) + low_intercept
            if not np.isfinite(upper_start) or not np.isfinite(lower_start) or upper_start <= lower_start:
                continue
            start_width = max(upper_start - lower_start, 1e-9)
            contraction_score = _clip_value(1.0 - (upper_now - lower_now) / start_width)
            if contraction_score < max(0.12, min_shape_score * 0.35):
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
                touch_tolerance_pct,
            )
            if containment_ratio < min_containment or body_excursion_pct > max_body_excursion:
                continue
            containment_score = _clip_value((containment_ratio - min_containment) / max(1.0 - min_containment, 1e-9))
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
            max_abs_slope_pct = max(abs(float(high_slope_pct)), abs(float(low_slope_pct)))
            if max_abs_slope_pct > float(cfg.max_geometry_boundary_slope_pct_per_bar):
                continue
            if (
                float(cfg.aggressive_geometry_slope_pct_per_bar) > 0.0
                and max_abs_slope_pct > float(cfg.aggressive_geometry_slope_pct_per_bar)
                and min(float(upper["touch_count"]), float(lower["touch_count"]))
                < float(cfg.aggressive_geometry_min_side_touches)
            ):
                continue
            convergence_score = _clip_value((low_slope_pct - high_slope_pct) / max(min_slope * 8.0, 1e-9))
            touch_score = _clip_value(
                min(float(upper["touch_count"]), float(lower["touch_count"])) / max(float(min_side_pivots) + 1.0, 1.0)
            )
            line_score = 0.5 * float(upper["score"]) + 0.5 * float(lower["score"])
            width_score = _clip_value(1.0 - width_pct / max(max_width, 1e-9))
            span_score = _clip_value((span - float(min_bars)) / max(float(window - min_bars), 1.0))
            recency_score = _clip_value(1.0 - (row - last_anchor) / max(float(window), 1.0))
            base_quality = (
                0.18 * touch_score
                + 0.14 * line_score
                + 0.12 * width_score
                + 0.13 * contraction_score
                + 0.08 * span_score
                + 0.15 * containment_score
                + 0.06 * recency_score
                + 0.06 * recent_touch_score
                + 0.04 * anchor_balance_score
                + 0.04 * touch_balance_score
            )
            compression_quality = _clip_value(0.78 * base_quality + 0.22 * convergence_score)
            upper_flat = abs(high_slope_pct) <= flat_tolerance
            lower_flat = abs(low_slope_pct) <= flat_tolerance
            upper_falling = high_slope_pct <= -min_slope
            upper_rising = high_slope_pct >= min_slope
            lower_falling = low_slope_pct <= -min_slope
            lower_rising = low_slope_pct >= min_slope
            # These buckets are deliberately non-overlapping. If a shape is
            # ambiguous, it must survive as broad compression evidence
            # instead of claiming a triangle/wedge subtype.
            ascending_shape = 0.0
            if upper_flat and lower_rising:
                ascending_shape = _clip_value(1.0 - abs(high_slope_pct) / max(flat_tolerance, 1e-9)) * _clip_value(
                    low_slope_pct / max(min_slope * 4.0, 1e-9)
                )
            descending_shape = 0.0
            if upper_falling and lower_flat:
                descending_shape = _clip_value(1.0 - abs(low_slope_pct) / max(flat_tolerance, 1e-9)) * _clip_value(
                    -high_slope_pct / max(min_slope * 4.0, 1e-9)
                )
            symmetric_shape = 0.0
            if upper_falling and lower_rising:
                symmetric_shape = (
                    convergence_score
                    * _clip_value(-high_slope_pct / max(min_slope * 4.0, 1e-9))
                    * _clip_value(low_slope_pct / max(min_slope * 4.0, 1e-9))
                )
            falling_wedge_shape = 0.0
            if upper_falling and lower_falling and high_slope_pct < low_slope_pct:
                falling_wedge_shape = convergence_score * _clip_value(-high_slope_pct / max(min_slope * 4.0, 1e-9))
            rising_wedge_shape = 0.0
            if upper_rising and lower_rising and high_slope_pct < low_slope_pct:
                rising_wedge_shape = convergence_score * _clip_value(low_slope_pct / max(min_slope * 4.0, 1e-9))

            shapes = {
                "triangle_ascending": ascending_shape,
                "triangle_descending": descending_shape,
                "triangle_symmetric": symmetric_shape,
                "wedge_falling": falling_wedge_shape,
                "wedge_rising": rising_wedge_shape,
            }
            qualities = {name: _clip_value(0.74 * base_quality + 0.26 * shape_score) for name, shape_score in shapes.items()}
            best_name, best_quality = max(qualities.items(), key=lambda item: item[1])
            best_shape_score = float(shapes[best_name])
            identification_quality = _clip_value(0.82 * base_quality + 0.18 * convergence_score)
            slot_candidate = {
                "upper_now": float(upper_now),
                "lower_now": float(lower_now),
                "upper_start": float(upper_start),
                "lower_start": float(lower_start),
                "upper_start_x": float(upper["x1"]),
                "lower_start_x": float(lower["x1"]),
                "upper_slope": float(high_slope),
                "lower_slope": float(low_slope),
                "start_x": float(start_x),
                "span": float(span),
            }
            if identification_quality >= min_identification_quality:
                identified_candidates.append({"quality": float(identification_quality), **slot_candidate})
            named_valid = best_shape_score >= min_shape_score
            if best_name.startswith("triangle"):
                named_valid = named_valid and best_quality >= float(cfg.min_triangle_quality)
            if best_name.startswith("wedge"):
                named_valid = named_valid and best_quality >= float(cfg.min_wedge_quality)
            broad_valid = compression_quality >= min_compression_quality
            # The broad compression label is not a consolation prize for
            # a triangle/wedge that almost passed. If the shape already looks
            # named, keep pressure on the named thresholds instead of emitting a
            # vague fallback signal.
            broad_valid = broad_valid and best_shape_score < min_shape_score
            broad_valid = (
                broad_valid
                and min(float(upper["touch_count"]), float(lower["touch_count"]))
                >= float(cfg.min_compression_side_touches)
                and boundary_intrusion_pct <= float(cfg.max_compression_boundary_intrusion_pct)
                and anchor_balance_score >= float(cfg.min_compression_anchor_balance_score)
                and touch_balance_score >= float(cfg.min_compression_touch_balance_score)
                and contraction_score >= float(cfg.min_compression_contraction_score)
                and convergence_score >= float(cfg.min_compression_convergence_score)
            )
            if not named_valid and not broad_valid:
                continue
            if named_valid:
                named_family = "triangle" if best_name.startswith("triangle") else "wedge"
                family_candidates[named_family].append({"quality": float(best_quality), **slot_candidate})
            if broad_valid:
                family_candidates["compression"].append({"quality": float(compression_quality), **slot_candidate})
            candidate_quality = best_quality if named_valid else compression_quality
            candidate_name = best_name if named_valid else "compression"
            if best is None or candidate_quality > best["quality"]:
                best = {
                    "name": candidate_name,
                    "quality": candidate_quality,
                    "named_quality": best_quality if named_valid else 0.0,
                    "prior_direction": float(prior_direction),
                    "upper_now": float(upper_now),
                    "lower_now": float(lower_now),
                    "upper_start": float(upper_start),
                    "lower_start": float(lower_start),
                    "upper_start_x": float(upper["x1"]),
                    "lower_start_x": float(lower["x1"]),
                    "upper_anchor_start": float(upper_anchor_start),
                    "lower_anchor_start": float(lower_anchor_start),
                    "upper_slope": float(high_slope),
                    "lower_slope": float(low_slope),
                    "start_x": float(start_x),
                    "width_pct": float(width_pct),
                    "contraction_score": float(contraction_score),
                    "recent_touch_score": float(recent_touch_score),
                    "anchor_balance_score": float(anchor_balance_score),
                    "touch_balance_score": float(touch_balance_score),
                    "boundary_respect_ratio": float(boundary_respect_ratio),
                    "boundary_intrusion_pct": float(boundary_intrusion_pct),
                    "compression_quality": float(compression_quality),
                    "upper_touch_count": float(upper["touch_count"]),
                    "lower_touch_count": float(lower["touch_count"]),
                    "source": "pivot",
                    "source_count": 1.0,
                    "experimental_confluence_score": 0.0,
                    "experimental_confluence_bonus": 0.0,
                }

        _store_geometry_identification_slots(
            out,
            row,
            identified_candidates,
            body_high,
            body_low,
            close,
            cfg,
            slot_count,
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
        if best is None:
            continue

        display_upper_now = float(best["upper_now"])
        display_lower_now = float(best["lower_now"])
        display_upper_start = float(best["upper_start"])
        display_lower_start = float(best["lower_start"])
        display_upper_anchor_start = float(best.get("upper_anchor_start", best["upper_start"]))
        display_lower_anchor_start = float(best.get("lower_anchor_start", best["lower_start"]))
        upper_start_x = float(best.get("upper_start_x", best["start_x"]))
        lower_start_x = float(best.get("lower_start_x", best["start_x"]))
        upper_slope = float(best.get("upper_slope", np.nan))
        lower_slope = float(best.get("lower_slope", np.nan))
        if np.isfinite([upper_slope, lower_slope, display_upper_now, display_lower_now]).all():
            upper_intercept = display_upper_now - upper_slope * float(row)
            lower_intercept = display_lower_now - lower_slope * float(row)
            upper_intercept, lower_intercept = _snap_geometry_boundaries_outward(
                body_high,
                body_low,
                close,
                float(best["start_x"]),
                row,
                upper_slope,
                upper_intercept,
                lower_slope,
                lower_intercept,
                float(cfg.geometry_boundary_snap_max_pct),
            )
            display_upper_now = upper_slope * float(row) + upper_intercept
            display_lower_now = lower_slope * float(row) + lower_intercept
            display_upper_start = upper_slope * float(best["start_x"]) + upper_intercept
            display_lower_start = lower_slope * float(best["start_x"]) + lower_intercept
            display_upper_anchor_start = upper_slope * upper_start_x + upper_intercept
            display_lower_anchor_start = lower_slope * lower_start_x + lower_intercept
        display_width_pct = (display_upper_now - display_lower_now) / max(abs(float(close[row])), 1e-9)

        out["range_contraction_score"][row] = max(out["range_contraction_score"][row], best["contraction_score"])
        out["geometry_upper"][row] = display_upper_now
        out["geometry_lower"][row] = display_lower_now
        out["geometry_upper_start"][row] = display_upper_start
        out["geometry_lower_start"][row] = display_lower_start
        out["geometry_upper_anchor_start"][row] = display_upper_anchor_start
        out["geometry_lower_anchor_start"][row] = display_lower_anchor_start
        out["geometry_upper_start_index"][row] = upper_start_x
        out["geometry_lower_start_index"][row] = lower_start_x
        out["geometry_start_index"][row] = best["start_x"]
        out["geometry_end_index"][row] = float(row)
        out["geometry_width_pct"][row] = display_width_pct
        out["geometry_recent_touch_score"][row] = best["recent_touch_score"]
        out["geometry_anchor_balance_score"][row] = best["anchor_balance_score"]
        out["geometry_touch_balance_score"][row] = best["touch_balance_score"]
        out["geometry_tlv2_source"][row] = str(best.get("source", "")) == "tlv2"
        out["geometry_source_count"][row] = float(best.get("source_count", 1.0))
        out["geometry_experimental_confluence_score"][row] = float(best.get("experimental_confluence_score", 0.0))
        out["geometry_experimental_confluence_bonus"][row] = float(best.get("experimental_confluence_bonus", 0.0))
        out["geometry_quality"][row] = float(best["quality"])
        out["geometry_candidate_count"][row] = 1.0
        out["geometry_upper_touch_count"][row] = float(best.get("upper_touch_count", 0.0))
        out["geometry_lower_touch_count"][row] = float(best.get("lower_touch_count", 0.0))
        out["geometry_confirmation_count"][row] = float(
            best.get("upper_touch_count", 0.0) + best.get("lower_touch_count", 0.0)
        )

        best_name = str(best["name"])
        best_quality = float(best["quality"])
        compression_quality = float(best["compression_quality"])
        if best_name in {"triangle_ascending", "wedge_falling"}:
            out["geometry_bias"][row] = 1
        elif best_name in {"triangle_descending", "wedge_rising"}:
            out["geometry_bias"][row] = -1
        else:
            out["geometry_bias"][row] = 1 if best["prior_direction"] >= 0 else -1
        if best_name == "compression" and compression_quality >= min_compression_quality:
            out["compression_quality"][row] = compression_quality
            if best["prior_direction"] >= 0:
                out["compression_setup_long"][row] = True
            else:
                out["compression_setup_short"][row] = True
        named_quality = float(best.get("named_quality", best_quality))
        if best_name == "triangle_ascending":
            out["triangle_ascending_quality"][row] = named_quality
            out["triangle_ascending_setup_long"][row] = True
        elif best_name == "triangle_descending":
            out["triangle_descending_quality"][row] = named_quality
            out["triangle_descending_setup_short"][row] = True
        elif best_name == "triangle_symmetric":
            out["triangle_symmetric_quality"][row] = named_quality
            if best["prior_direction"] >= 0:
                out["triangle_symmetric_setup_long"][row] = True
            else:
                out["triangle_symmetric_setup_short"][row] = True
        elif best_name == "wedge_falling":
            out["wedge_falling_quality"][row] = named_quality
            out["wedge_falling_setup_long"][row] = True
        elif best_name == "wedge_rising":
            out["wedge_rising_quality"][row] = named_quality
            out["wedge_rising_setup_short"][row] = True

    if bool(getattr(cfg, "merge_geometry_patterns", True)):
        _merge_geometry_pattern_rows(out, frame, cfg)

    return out


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
    selected: list[dict[str, float]] = []
    for candidate in sorted(candidates, key=lambda item: float(item["quality"]), reverse=True):
        if any(_same_identification_slot(candidate, existing, close, row) for existing in selected):
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
        upper_now = upper_slope * float(row) + upper_intercept
        lower_now = lower_slope * float(row) + lower_intercept
        upper_start_x = float(candidate["upper_start_x"])
        lower_start_x = float(candidate["lower_start_x"])
        out[f"{prefix}_slot_{offset}_quality"][row] = float(candidate["quality"])
        out[f"{prefix}_slot_{offset}_upper"][row] = upper_now
        out[f"{prefix}_slot_{offset}_lower"][row] = lower_now
        out[f"{prefix}_slot_{offset}_upper_start"][row] = upper_slope * upper_start_x + upper_intercept
        out[f"{prefix}_slot_{offset}_lower_start"][row] = lower_slope * lower_start_x + lower_intercept
        out[f"{prefix}_slot_{offset}_upper_start_index"][row] = upper_start_x
        out[f"{prefix}_slot_{offset}_lower_start_index"][row] = lower_start_x
        out[f"{prefix}_slot_{offset}_start_index"][row] = float(candidate["start_x"])
        out[f"{prefix}_slot_{offset}_end_index"][row] = float(row)


def _same_identification_slot(
    candidate: dict[str, float],
    existing: dict[str, float],
    close: np.ndarray,
    row: int,
) -> bool:
    reference = max(abs(float(close[row])), 1e-9)
    upper_distance = abs(float(candidate["upper_now"]) - float(existing["upper_now"])) / reference
    lower_distance = abs(float(candidate["lower_now"]) - float(existing["lower_now"])) / reference
    span = max(float(candidate.get("span", 1.0)), 1.0)
    start_distance = abs(float(candidate["start_x"]) - float(existing["start_x"])) / span
    return max(upper_distance, lower_distance) <= 0.01 and start_distance <= 0.20


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


def _geometry_management_columns(
    frame: DataFrame,
    arrays: dict[str, np.ndarray],
    cfg: PatternStructureConfig,
) -> dict[str, np.ndarray]:
    """Turn raw geometry candidates into a compact strategy management packet.

    The pattern detector owns "where is this setup wrong?" evidence, while the
    strategy owns whether to enter, size, stop, or exit. This helper carries the
    latest valid geometry rails forward for a bounded no-lookahead window and
    marks boundary interaction states against the current candle.

    Cause columns describe what price did at the rails, such as bouncing from
    support or rejecting resistance. Advice columns use explicit names:
    ``go_long``, ``go_short``, ``exit_long``, and ``exit_short``. This keeps the
    dataframe readable in strategy files and avoids names like "reject short",
    which can be mistaken for an exit instruction.
    """

    rows = len(frame)
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype="float64")
    high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype="float64")
    low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype="float64")
    upper_source = arrays["geometry_upper"]
    lower_source = arrays["geometry_lower"]
    upper_start_source = arrays["geometry_upper_start"]
    lower_start_source = arrays["geometry_lower_start"]
    start_index_source = arrays["geometry_start_index"]
    width_source = arrays["geometry_width_pct"]
    bias_source = arrays["geometry_bias"]
    quality_source = arrays["geometry_quality"]
    candidate_count_source = arrays["geometry_candidate_count"]
    confirmation_count_source = arrays["geometry_confirmation_count"]
    max_age = int(cfg.geometry_management_max_age_bars)
    buffer_pct = float(cfg.geometry_management_buffer_pct)
    out = {
        "active": np.zeros(rows, dtype=bool),
        "bias": np.zeros(rows, dtype="int8"),
        "quality": np.zeros(rows, dtype="float64"),
        "candidate_count": np.zeros(rows, dtype="float64"),
        "confirmation_count": np.zeros(rows, dtype="float64"),
        "upper": np.full(rows, np.nan, dtype="float64"),
        "lower": np.full(rows, np.nan, dtype="float64"),
        "width_pct": np.full(rows, np.nan, dtype="float64"),
        "stop_ref_long": np.full(rows, np.nan, dtype="float64"),
        "stop_ref_short": np.full(rows, np.nan, dtype="float64"),
        "target_ref_long": np.full(rows, np.nan, dtype="float64"),
        "target_ref_short": np.full(rows, np.nan, dtype="float64"),
        "long_bounce_support": np.zeros(rows, dtype=bool),
        "long_breakout": np.zeros(rows, dtype=bool),
        "go_long": np.zeros(rows, dtype=bool),
        "short_reject_resistance": np.zeros(rows, dtype=bool),
        "short_breakdown": np.zeros(rows, dtype=bool),
        "go_short": np.zeros(rows, dtype=bool),
        "breakout_up": np.zeros(rows, dtype=bool),
        "breakdown_down": np.zeros(rows, dtype=bool),
        "invalid_long": np.zeros(rows, dtype=bool),
        "invalid_short": np.zeros(rows, dtype=bool),
        "exit_long": np.zeros(rows, dtype=bool),
        "exit_short": np.zeros(rows, dtype=bool),
    }
    active_upper = np.nan
    active_lower = np.nan
    active_upper_slope = 0.0
    active_upper_intercept = np.nan
    active_lower_slope = 0.0
    active_lower_intercept = np.nan
    active_width = np.nan
    active_bias = 0
    active_quality = 0.0
    active_candidate_count = 0.0
    active_confirmation_count = 0.0
    active_start = -1
    terminate_after_row = False
    for row in range(rows):
        if terminate_after_row:
            active_upper = np.nan
            active_lower = np.nan
            active_upper_slope = 0.0
            active_upper_intercept = np.nan
            active_lower_slope = 0.0
            active_lower_intercept = np.nan
            active_width = np.nan
            active_bias = 0
            active_quality = 0.0
            active_candidate_count = 0.0
            active_confirmation_count = 0.0
            active_start = -1
            terminate_after_row = False
        if (
            np.isfinite(upper_source[row])
            and np.isfinite(lower_source[row])
            and upper_source[row] > lower_source[row]
            and int(bias_source[row]) != 0
        ):
            source_start = float(start_index_source[row])
            source_upper = float(upper_source[row])
            source_lower = float(lower_source[row])
            source_upper_start = float(upper_start_source[row])
            source_lower_start = float(lower_start_source[row])
            if (
                np.isfinite(source_start)
                and row > source_start
                and np.isfinite(source_upper_start)
                and np.isfinite(source_lower_start)
            ):
                active_upper_slope = (source_upper - source_upper_start) / (float(row) - source_start)
                active_upper_intercept = source_upper_start - active_upper_slope * source_start
                active_lower_slope = (source_lower - source_lower_start) / (float(row) - source_start)
                active_lower_intercept = source_lower_start - active_lower_slope * source_start
            else:
                active_upper_slope = 0.0
                active_upper_intercept = source_upper
                active_lower_slope = 0.0
                active_lower_intercept = source_lower
            active_upper = source_upper
            active_lower = source_lower
            active_width = float(width_source[row]) if np.isfinite(width_source[row]) else np.nan
            active_bias = int(bias_source[row])
            active_quality = float(quality_source[row]) if np.isfinite(quality_source[row]) else 0.0
            active_candidate_count = (
                float(candidate_count_source[row]) if np.isfinite(candidate_count_source[row]) else 0.0
            )
            active_confirmation_count = (
                float(confirmation_count_source[row]) if np.isfinite(confirmation_count_source[row]) else 0.0
            )
            active_start = row
        if active_start < 0 or row - active_start > max_age:
            continue
        if not np.isfinite(close[row]) or close[row] == 0.0:
            continue
        if np.isfinite(active_upper_intercept) and np.isfinite(active_lower_intercept):
            active_upper = active_upper_slope * float(row) + active_upper_intercept
            active_lower = active_lower_slope * float(row) + active_lower_intercept
        if not np.isfinite(active_upper) or not np.isfinite(active_lower) or active_upper <= active_lower:
            continue
        active_width = (active_upper - active_lower) / max(abs(float(close[row])), 1e-9)
        buffer = abs(float(close[row])) * buffer_pct
        out["active"][row] = True
        out["bias"][row] = active_bias
        out["quality"][row] = active_quality
        out["candidate_count"][row] = active_candidate_count
        out["confirmation_count"][row] = active_confirmation_count
        out["upper"][row] = active_upper
        out["lower"][row] = active_lower
        out["width_pct"][row] = active_width
        if active_bias > 0:
            out["stop_ref_long"][row] = active_lower - buffer
            out["target_ref_long"][row] = active_upper
            long_bounce_support = float(low[row]) <= active_lower + buffer and float(close[row]) >= active_lower - buffer
            breakout_up = float(close[row]) > active_upper + buffer
            breakdown_down = float(close[row]) < active_lower - buffer
            out["long_bounce_support"][row] = long_bounce_support
            out["long_breakout"][row] = breakout_up
            out["go_long"][row] = long_bounce_support or breakout_up
            out["breakout_up"][row] = breakout_up
            out["breakdown_down"][row] = breakdown_down
            out["invalid_long"][row] = breakdown_down
            out["exit_long"][row] = breakdown_down
            terminate_after_row = breakout_up or breakdown_down
        elif active_bias < 0:
            out["stop_ref_short"][row] = active_upper + buffer
            out["target_ref_short"][row] = active_lower
            short_reject_resistance = float(high[row]) >= active_upper - buffer and float(close[row]) <= active_upper + buffer
            breakout_up = float(close[row]) > active_upper + buffer
            breakdown_down = float(close[row]) < active_lower - buffer
            out["short_reject_resistance"][row] = short_reject_resistance
            out["short_breakdown"][row] = breakdown_down
            out["go_short"][row] = short_reject_resistance or breakdown_down
            out["breakout_up"][row] = breakout_up
            out["breakdown_down"][row] = breakdown_down
            out["invalid_short"][row] = breakout_up
            out["exit_short"][row] = breakout_up
            terminate_after_row = breakout_up or breakdown_down
    return out


def _compression_quality_threshold(cfg: PatternStructureConfig) -> float:
    return float(cfg.min_compression_quality)


def _dedupe_geometry_events(mask: Series, cooldown_bars: int) -> Series:
    clean = mask.fillna(False).astype("bool")
    cooldown = max(int(cooldown_bars), 1)
    if cooldown <= 1:
        return clean
    recent = clean.shift(1, fill_value=False).rolling(cooldown, min_periods=1).max().astype("bool")
    return clean & ~recent


def _geometry_pattern_family(name: str) -> str:
    if name.startswith("triangle"):
        return "triangle"
    if name.startswith("wedge"):
        return "wedge"
    if name.startswith("compression"):
        return "compression"
    return "other"


def _geometry_price_scale(frame: DataFrame, window: int = 30) -> np.ndarray:
    height = (pd.to_numeric(frame["high"], errors="coerce") - pd.to_numeric(frame["low"], errors="coerce")).abs()
    body = (pd.to_numeric(frame["close"], errors="coerce") - pd.to_numeric(frame["open"], errors="coerce")).abs()
    scale = pd.concat(
        [height.rolling(window, min_periods=3).mean(), body.rolling(window, min_periods=3).mean() * 1.8],
        axis=1,
    ).max(axis=1)
    scale = scale.bfill().fillna(height.expanding(min_periods=1).mean()).fillna(1e-9)
    return scale.clip(lower=1e-9).to_numpy(dtype="float64")


def _geometry_row_name(out: dict[str, np.ndarray], row: int) -> tuple[str, float]:
    candidates = [
        ("triangle_ascending", out["triangle_ascending_quality"][row], out["triangle_ascending_setup_long"][row]),
        ("triangle_descending", out["triangle_descending_quality"][row], out["triangle_descending_setup_short"][row]),
        ("triangle_symmetric", out["triangle_symmetric_quality"][row], out["triangle_symmetric_setup_long"][row] or out["triangle_symmetric_setup_short"][row]),
        ("wedge_falling", out["wedge_falling_quality"][row], out["wedge_falling_setup_long"][row]),
        ("wedge_rising", out["wedge_rising_quality"][row], out["wedge_rising_setup_short"][row]),
        ("compression", out["compression_quality"][row], out["compression_setup_long"][row] or out["compression_setup_short"][row]),
    ]
    active = [(name, float(quality)) for name, quality, enabled in candidates if bool(enabled)]
    if not active:
        return "", 0.0
    return max(active, key=lambda item: item[1])


def _geometry_candidate_from_row(out: dict[str, np.ndarray], row: int, name: str, quality: float) -> dict[str, float] | None:
    upper = float(out["geometry_upper"][row])
    lower = float(out["geometry_lower"][row])
    upper_x1 = float(out["geometry_upper_start_index"][row])
    lower_x1 = float(out["geometry_lower_start_index"][row])
    upper_y1 = float(out["geometry_upper_anchor_start"][row])
    lower_y1 = float(out["geometry_lower_anchor_start"][row])
    if not np.isfinite([upper, lower, upper_x1, lower_x1, upper_y1, lower_y1]).all():
        return None
    if upper <= lower:
        return None
    return {
        "row": float(row),
        "name": name,
        "quality": float(quality),
        "family": _geometry_pattern_family(name),
        "start_x": float(out["geometry_start_index"][row]),
        "upper_x1": upper_x1,
        "upper_y1": upper_y1,
        "upper_x2": float(row),
        "upper_y2": upper,
        "lower_x1": lower_x1,
        "lower_y1": lower_y1,
        "lower_x2": float(row),
        "lower_y2": lower,
    }


def _geometry_line(candidate: dict[str, float], side: str) -> tuple[float, float]:
    x1 = float(candidate[f"{side}_x1"])
    x2 = float(candidate[f"{side}_x2"])
    y1 = float(candidate[f"{side}_y1"])
    y2 = float(candidate[f"{side}_y2"])
    if abs(x2 - x1) < 1e-9:
        return 0.0, y1
    slope = (y2 - y1) / (x2 - x1)
    return slope, y1 - slope * x1


def _geometry_candidate_distance(
    a: dict[str, float],
    b: dict[str, float],
    price_scale: np.ndarray,
) -> tuple[float, float]:
    left = max(float(a["start_x"]), float(b["start_x"]))
    right = min(float(a["row"]), float(b["row"]))
    if right <= left:
        left = min(float(a["row"]), float(b["row"])) - 3.0
        right = min(float(a["row"]), float(b["row"]))
    xs = np.linspace(left, right, 5)
    au_slope, au_intercept = _geometry_line(a, "upper")
    al_slope, al_intercept = _geometry_line(a, "lower")
    bu_slope, bu_intercept = _geometry_line(b, "upper")
    bl_slope, bl_intercept = _geometry_line(b, "lower")
    upper_dist = np.nanmean(np.abs((au_slope * xs + au_intercept) - (bu_slope * xs + bu_intercept)))
    lower_dist = np.nanmean(np.abs((al_slope * xs + al_intercept) - (bl_slope * xs + bl_intercept)))
    scale_row = int(np.clip(max(float(a["row"]), float(b["row"])), 0, len(price_scale) - 1))
    scale = max(float(price_scale[scale_row]), 1e-9)
    rail_distance_scale = max(float(upper_dist), float(lower_dist)) / scale
    slope_distance_scale = max(abs(au_slope - bu_slope), abs(al_slope - bl_slope)) * max(right - left, 1.0) / scale
    return rail_distance_scale, slope_distance_scale


def _clear_geometry_setup_row(out: dict[str, np.ndarray], row: int) -> None:
    out["triangle_ascending_setup_long"][row] = False
    out["triangle_descending_setup_short"][row] = False
    out["triangle_symmetric_setup_long"][row] = False
    out["triangle_symmetric_setup_short"][row] = False
    out["wedge_falling_setup_long"][row] = False
    out["wedge_rising_setup_short"][row] = False
    out["compression_setup_long"][row] = False
    out["compression_setup_short"][row] = False


def _merge_geometry_pattern_rows(out: dict[str, np.ndarray], frame: DataFrame, cfg: PatternStructureConfig) -> None:
    """Collapse repeated rolling-window geometry detections online.

    The detector intentionally uses a wider search window to avoid missing
    human-visible structures. That creates repeated rail-pair candidates across
    nearby candles. This pass only uses information available up to the current
    row: a later row may strengthen the active cluster, but it must not
    retroactively create an earlier signal.
    """

    price_scale = _geometry_price_scale(frame)
    row_gap = int(cfg.geometry_merge_row_gap)
    rail_limit = float(cfg.geometry_merge_rail_distance_body_mult)
    slope_limit = float(cfg.geometry_merge_slope_distance_body_mult)
    cross_family = bool(cfg.geometry_merge_cross_family)
    clusters: list[dict[str, object]] = []
    rows = len(out["geometry_upper"])
    for row in range(rows):
        name, quality = _geometry_row_name(out, row)
        if not name:
            continue
        candidate = _geometry_candidate_from_row(out, row, name, quality)
        if candidate is None:
            continue
        best_idx = -1
        best_distance = np.inf
        for idx, cluster in enumerate(clusters):
            family = str(cluster["family"])
            same_family = str(candidate["family"]) == family
            if not same_family and not cross_family:
                continue
            if float(candidate["row"]) - float(cluster["last_row"]) > float(row_gap):
                continue
            member_distances = [
                _geometry_candidate_distance(candidate, member, price_scale)
                for member in cluster["members"]  # type: ignore[index]
            ]
            rail_distance, slope_distance = min(member_distances, key=lambda pair: pair[0] + 0.45 * pair[1])
            if rail_distance > rail_limit or slope_distance > slope_limit:
                continue
            distance = rail_distance + 0.45 * slope_distance
            if distance < best_distance:
                best_distance = distance
                best_idx = idx
        if best_idx < 0:
            clusters.append(
                {
                    "family": candidate["family"],
                    "last_row": float(row),
                    "members": [candidate],
                    "touch_keys": {
                        ("u", round(float(candidate["upper_x1"]), 1)),
                        ("u", round(float(candidate["upper_x2"]), 1)),
                        ("l", round(float(candidate["lower_x1"]), 1)),
                        ("l", round(float(candidate["lower_x2"]), 1)),
                    },
                }
            )
            out["geometry_candidate_count"][row] = max(out["geometry_candidate_count"][row], 1.0)
            out["geometry_confirmation_count"][row] = max(
                out["geometry_confirmation_count"][row],
                out["geometry_upper_touch_count"][row] + out["geometry_lower_touch_count"][row],
            )
            continue

        cluster = clusters[best_idx]
        cluster["last_row"] = float(row)
        cluster["members"].append(candidate)  # type: ignore[index]
        touch_keys = cluster["touch_keys"]  # type: ignore[assignment]
        new_touch_keys = {
            ("u", round(float(candidate["upper_x1"]), 1)),
            ("u", round(float(candidate["upper_x2"]), 1)),
            ("l", round(float(candidate["lower_x1"]), 1)),
            ("l", round(float(candidate["lower_x2"]), 1)),
        }
        touch_keys.update(new_touch_keys)
        out["geometry_candidate_count"][row] = float(len(cluster["members"]))  # type: ignore[arg-type]
        out["geometry_confirmation_count"][row] = max(
            float(len(touch_keys)),
            out["geometry_upper_touch_count"][row] + out["geometry_lower_touch_count"][row],
        )
        _clear_geometry_setup_row(out, row)


def _geometry_candidate_starts(high_x: np.ndarray, low_x: np.ndarray, row: int, min_bars: int) -> np.ndarray:
    events = np.concatenate([high_x[np.isfinite(high_x)], low_x[np.isfinite(low_x)]])
    if len(events) == 0:
        return np.array([], dtype="float64")
    events = np.unique(np.sort(events.astype("float64")))
    valid = events <= float(row - min_bars)
    starts = events[valid]
    if len(starts) <= 12:
        return starts
    # Test early, middle, and recent starts. This keeps broad human-visible
    # structures in play while still bounding runtime for rolling detection.
    middle = starts[2:-4]
    if len(middle):
        middle_pick = middle[np.linspace(0, len(middle) - 1, min(6, len(middle)), dtype=int)]
        selected = np.unique(np.concatenate([starts[:2], middle_pick, starts[-4:]]))
    else:
        selected = np.unique(np.concatenate([starts[:2], starts[-4:]]))
    return selected.astype("float64")


def _best_boundary_line(
    x: np.ndarray,
    y: np.ndarray,
    row: int,
    reference: float,
    touch_tolerance_pct: float,
    max_violation_pct: float,
    min_bars: int,
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
    for left in range(len(xv) - 1):
        for right in range(left + 1, len(xv)):
            x1 = float(xv[left])
            x2 = float(xv[right])
            if x2 - x1 < max(float(min_bars) * 0.45, 2.0):
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
            mean_distance_pct = float(np.nanmean(distance) / max(abs(float(reference)), 1e-9))
            fit_score = _clip_value(1.0 - mean_distance_pct / max(float(touch_tolerance_pct) * 3.0, 1e-9))
            touch_score = _clip_value(float(touch_count) / max(float(len(xv)), 1.0))
            span_score = _clip_value((x2 - x1) / max(float(row) - float(xv[0]), 1.0))
            recency_score = _clip_value(1.0 - (float(row) - x2) / max(float(row) - float(xv[0]), 1.0))
            recent_touch_score = _clip_value(1.0 - (float(row) - last_touch_x) / max(float(row) - float(xv[0]), 1.0))
            score = 0.38 * touch_score + 0.30 * fit_score + 0.18 * span_score + 0.14 * recency_score
            if best is None or score > best["score"]:
                best = {
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "x1": x1,
                    "x2": x2,
                    "touch_count": float(touch_count),
                    "last_touch_x": float(last_touch_x),
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

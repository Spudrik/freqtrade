"""Pattern geometry v2 indicator.

This module owns the line-pair geometry family that is currently considered
production enough to use in strategy research:

- triangle, wedge, and generic compression come from paired TLV2 resistance
  and support lines. These are the strongest part of the indicator and are
  tuned mainly around 1h and 4h data.
- rectangle, ascending channel, and descending channel are emitted from the
  same geometry contract, but they are intended as context and avoidance
  signals first. A channel output can warn that price is near a rail or has
  broken a rail, but it should not be treated as an equally mature breakout
  setup until it has been validated further.

The key design choice is scale independence. Gates use ATR, candle spans,
line scores, containment, pivot counts, and slope measured in ATR-per-bar.
Price-percent filters are intentionally absent because they do not transfer
cleanly across coins and timeframes.

Example: a BTC 4h compressing pattern and a SOL 1h compressing pattern should
be judged by how narrow the current width is versus current ATR, how recently
both rails were touched, and whether candle bodies stay inside the rails. They
should not be judged by a fixed percent of price.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

try:
    from .complex_trendline_projection_v2 import (
        _base_inputs,
        _build_sequence_candidate_table,
        _resolve_config as _resolve_trendline_v2_config,
    )
except Exception:  # pragma: no cover - standalone review scripts import this module directly
    from complex_trendline_projection_v2 import (  # type: ignore[no-redef]
        _base_inputs,
        _build_sequence_candidate_table,
        _resolve_config as _resolve_trendline_v2_config,
    )


_FAMILY_CODE = {
    "triangle": 1,
    "wedge": 2,
    "compression": 3,
    "rectangle": 4,
    "ascending_channel": 5,
    "descending_channel": 6,
}
_COMPRESSIVE_FAMILY_CODES = {1, 2, 3}

# Slot fields describe an active pattern instance. Strategy code should treat
# each slot as one complete geometry object for that candle:
#
# - ``active`` says the slot is usable on this row.
# - ``family`` identifies the shape: 1 triangle, 2 wedge, 3 compression,
#   4 rectangle, 5 ascending channel, 6 descending channel.
# - ``direction`` is -1, 0, or +1 based on the midpoint slope. For example,
#   an ascending channel is +1, a rectangle is 0, and a descending channel is
#   -1. A triangle can still be neutral when the two rails converge without
#   a meaningful midpoint slope.
# - ``upper`` and ``lower`` are the current row's projected rails. ``*_start``
#   and ``*_slope`` allow plotting the same rails back to ``start_index``.
# - ``line_score``, ``containment``, ``width_atr``, and pivot counts are
#   diagnostics for downstream ranking. They are not separate entry signals.
_SLOT_FIELDS = (
    "active",
    "family",
    "direction",
    "upper",
    "lower",
    "upper_start",
    "lower_start",
    "upper_slope",
    "lower_slope",
    "start_index",
    "end_index",
    "line_score",
    "contraction",
    "containment",
    "width_atr",
    "upper_pivots",
    "lower_pivots",
)
_ROW_FIELDS = (
    "best_width_atr",
    "compression_flag",
    "best_position",
    "near_upper",
    "near_lower",
    "avoid_long",
    "avoid_short",
    "breakout_up",
    "breakdown_down",
)
_ROW_BOOL_FIELDS = {
    "compression_flag",
    "near_upper",
    "near_lower",
    "avoid_long",
    "avoid_short",
    "breakout_up",
    "breakdown_down",
}
_PROFILE_FIELDS = (
    "min_pattern_bars",
    "max_pattern_bars",
    "compression_max_width_atr",
    "local_narrowing_lookback_bars",
    "local_narrowing_min_ratio",
    "min_line_score",
    "max_recent_touch_age_bars",
    "channel_min_pattern_bars",
    "channel_max_pattern_bars",
    "channel_min_width_atr",
    "channel_max_width_atr",
    "channel_max_width_change_ratio",
    "channel_min_containment",
    "channel_min_quality",
    "channel_envelope_start_options",
    "channel_lifecycle_confirm_break_bars",
)

# Timeframe profiles are deliberately narrow. The indicator is most effective
# on 1h and 4h because those windows showed enough pivot density for line
# geometry to be meaningful. Higher timeframe defaults are retained so the
# function does not fail when called with 8h/1d/3d, but those outputs should be
# treated as context until separately validated.
#
# Profiles override only fields listed in ``_PROFILE_FIELDS``. Any caller
# override always wins. Example: ``add_pattern_geometry_v2(df, timeframe="1h",
# max_pattern_bars=96)`` uses the 1h defaults for the other profile fields,
# but keeps the explicit 96-bar cap.
_TIMEFRAME_PROFILES: dict[str, dict[str, object]] = {
    "1h": {
        "min_pattern_bars": 12,
        "max_pattern_bars": 72,
        "compression_max_width_atr": 2.0,
        "local_narrowing_lookback_bars": 8,
        "local_narrowing_min_ratio": 0.10,
        "min_line_score": 0.50,
        "max_recent_touch_age_bars": 12,
        "channel_min_pattern_bars": 36,
        "channel_max_pattern_bars": 168,
        "channel_min_width_atr": 1.0,
        "channel_max_width_atr": 8.0,
        "channel_max_width_change_ratio": 0.22,
        "channel_min_containment": 0.68,
        "channel_min_quality": 0.82,
        "channel_envelope_start_options": 10,
        "channel_lifecycle_confirm_break_bars": 2,
    },
    "4h": {
        "min_pattern_bars": 12,
        "max_pattern_bars": 72,
        "compression_max_width_atr": 2.0,
        "local_narrowing_lookback_bars": 8,
        "local_narrowing_min_ratio": 0.10,
        "min_line_score": 0.50,
        "max_recent_touch_age_bars": 12,
        "channel_min_pattern_bars": 18,
        "channel_max_pattern_bars": 96,
        "channel_min_width_atr": 1.0,
        "channel_max_width_atr": 8.0,
        "channel_max_width_change_ratio": 0.22,
        "channel_min_containment": 0.68,
        "channel_min_quality": 0.82,
        "channel_envelope_start_options": 10,
        "channel_lifecycle_confirm_break_bars": 2,
    },
    "8h": {
        "min_pattern_bars": 12,
        "max_pattern_bars": 72,
        "compression_max_width_atr": 2.0,
        "local_narrowing_lookback_bars": 8,
        "local_narrowing_min_ratio": 0.10,
        "min_line_score": 0.50,
        "max_recent_touch_age_bars": 12,
        "channel_min_pattern_bars": 14,
        "channel_max_pattern_bars": 72,
        "channel_min_width_atr": 0.75,
        "channel_max_width_atr": 8.5,
        "channel_max_width_change_ratio": 0.26,
        "channel_min_containment": 0.66,
        "channel_min_quality": 0.80,
        "channel_envelope_start_options": 14,
        "channel_lifecycle_confirm_break_bars": 2,
    },
    "1d": {
        "min_pattern_bars": 10,
        "max_pattern_bars": 60,
        "compression_max_width_atr": 2.4,
        "local_narrowing_lookback_bars": 6,
        "local_narrowing_min_ratio": 0.10,
        "min_line_score": 0.45,
        "max_recent_touch_age_bars": 10,
        "channel_min_pattern_bars": 10,
        "channel_max_pattern_bars": 120,
        "channel_min_width_atr": 0.50,
        "channel_max_width_atr": 9.0,
        "channel_max_width_change_ratio": 0.32,
        "channel_min_containment": 0.62,
        "channel_min_quality": 0.78,
        "channel_envelope_start_options": 22,
        "channel_lifecycle_confirm_break_bars": 2,
    },
    "3d": {
        "min_pattern_bars": 8,
        "max_pattern_bars": 45,
        "compression_max_width_atr": 2.8,
        "local_narrowing_lookback_bars": 4,
        "local_narrowing_min_ratio": 0.08,
        "min_line_score": 0.40,
        "max_recent_touch_age_bars": 24,
        "channel_min_pattern_bars": 8,
        "channel_max_pattern_bars": 160,
        "channel_min_width_atr": 0.35,
        "channel_max_width_atr": 10.0,
        "channel_max_width_change_ratio": 0.36,
        "channel_min_containment": 0.58,
        "channel_min_quality": 0.76,
        "channel_envelope_start_options": 30,
        "channel_lifecycle_confirm_break_bars": 2,
    },
}
_TIMEFRAME_ALIASES = {
    "1h": "1h",
    "60m": "1h",
    "4h": "4h",
    "240m": "4h",
    "8h": "8h",
    "480m": "8h",
    "1d": "1d",
    "1day": "1d",
    "3d": "3d",
    "3day": "3d",
}


@dataclass(frozen=True)
class PatternGeometryV2Config:
    """Public levers for geometry v2.

    The first group of levers controls the mature compression engine. The
    process is:

    1. TLV2 supplies already-filtered resistance and support lines.
    2. This file pairs active upper/lower lines.
    3. The pair is accepted only if the rails are close enough in ATR terms,
       locally narrowing, recently touched on both sides, and containing candle
       bodies well enough.

    Example compression case: an upper rail is drifting down, a lower rail is
    drifting up, current width is 1.6 ATR, both rails touched within the last
    few candles, and most candle bodies remain inside the rails. That should
    survive as triangle/wedge/compression depending on the rail slopes.

    The channel levers are intentionally kept in the same config because
    channels share the same output contract and rail math. They are less
    mature than compression. A good use case is blocking bad longs near the
    top of an established rectangle, or blocking bad shorts near the lower
    rail of a descending channel. A weaker use case is using channel breakout
    alone as an entry trigger.

    Important scope note: some channel levers apply only to the pivot-envelope
    channel builder, while TLV2-supplied line pairs are governed by TLV2 line
    quality plus the shared containment, touch, and family gates.
    """

    output_prefix: str = "pg2"
    output_slots: int = 4
    timeframe: str = "4h"
    pivot_strength: int = 2
    pair_start_mode: str = "old"
    min_pattern_bars: int = 12
    max_pattern_bars: int = 72
    compression_max_width_atr: float = 2.0
    local_narrowing_lookback_bars: int = 8
    local_narrowing_min_ratio: float = 0.10
    min_containment: float = 0.88
    min_line_score: float = 0.50
    compression_flag_atr_threshold: float = 2.0
    include_channel_patterns: bool = True
    channel_min_pattern_bars: int = 18
    channel_max_pattern_bars: int = 96
    channel_min_width_atr: float = 1.0
    channel_max_width_atr: float = 8.0
    channel_max_width_change_ratio: float = 0.22
    channel_parallel_tolerance_atr_per_bar: float = 0.018
    channel_flat_slope_atr_per_bar: float = 0.012
    channel_min_slope_atr_per_bar: float = 0.018
    channel_min_containment: float = 0.68
    channel_min_quality: float = 0.82
    channel_min_side_pivots: int = 3
    channel_min_rail_span_ratio: float = 0.45
    channel_min_side_zones: int = 2
    channel_envelope_start_options: int = 10
    channel_merge_enabled: bool = True
    channel_merge_min_span_overlap: float = 0.55
    channel_merge_max_boundary_distance_atr: float = 1.40
    channel_merge_max_width_diff_atr: float = 1.25
    channel_merge_max_slope_diff_atr_per_bar: float = 0.012
    channel_max_active_outputs: int = 1
    channel_lifecycle_enabled: bool = True
    channel_lifecycle_break_atr_mult: float = 0.35
    channel_lifecycle_confirm_break_bars: int = 2
    channel_near_boundary_atr_mult: float = 0.70
    channel_breakout_atr_mult: float = 0.35
    min_output_bars: int = 2
    containment_tolerance_atr_mult: float = 0.25
    touch_tolerance_atr_mult: float = 0.55
    max_recent_touch_age_bars: int = 12
    min_slope_atr_per_bar: float = 0.010
    flat_slope_atr_per_bar: float = 0.015
    min_total_pivots: int = 4
    impulse_min_span_bars: int = 16
    impulse_max_slope_atr_per_bar: float = 0.10


@dataclass(frozen=True)
class _LineCandidate:
    x_old: float
    x_new: float
    y_old: float
    y_new: float
    slope: float
    intercept: float
    projection_end: float
    active_start: float
    score: float
    pivot_count: float
    absorbed_pivot_count: float
    impulse_span: float
    impulse_slope_atr: float


@dataclass(frozen=True)
class _PivotEvents:
    confirmed_at: np.ndarray
    anchor_index: np.ndarray
    price: np.ndarray


def add_pattern_geometry_v2(
    dataframe: DataFrame,
    config: PatternGeometryV2Config | None = None,
    **overrides: object,
) -> DataFrame:
    """Append geometry v2 columns to an OHLCV frame.

    This function is the only public entry point in this file. Strategies call
    it with a timeframe, for example:

    ``add_pattern_geometry_v2(df, timeframe="4h")``

    The returned frame keeps the original OHLCV columns and appends ``pg2_*``
    columns. Per-slot columns describe concrete patterns. Row-level columns
    summarize the best active compression/channel state for simple strategy
    use. Existing ``pg2_*`` columns are removed first so repeated indicator
    calls do not leave stale outputs.
    """
    cfg = _resolve_config(config, overrides)
    _validate_config(cfg)
    _validate_dataframe(dataframe)

    arrays = _geometry_v2_arrays(dataframe, cfg)
    p = cfg.output_prefix
    columns: dict[str, Series] = {}
    for slot in range(1, int(cfg.output_slots) + 1):
        for field in _SLOT_FIELDS:
            key = f"{p}_slot_{slot}_{field}"
            value = arrays[f"slot_{slot}_{field}"]
            if field == "active":
                columns[key] = pd.Series(value, index=dataframe.index, dtype="bool").fillna(False)
            elif field in {"family", "direction"}:
                columns[key] = pd.Series(value, index=dataframe.index, dtype="int8")
            else:
                columns[key] = pd.Series(value, index=dataframe.index, dtype="float64")
    for field in _ROW_FIELDS:
        key = f"{p}_{field}"
        value = arrays[field]
        if field in _ROW_BOOL_FIELDS:
            columns[key] = pd.Series(value, index=dataframe.index, dtype="bool").fillna(False)
        else:
            columns[key] = pd.Series(value, index=dataframe.index, dtype="float64")

    source = dataframe.copy()
    existing = [col for col in source.columns if str(col).startswith(f"{p}_")]
    clean = source.drop(columns=existing).copy() if existing else source.copy()
    return pd.concat([clean, pd.DataFrame(columns, index=dataframe.index)], axis=1)


def _geometry_v2_arrays(frame: DataFrame, cfg: PatternGeometryV2Config) -> dict[str, np.ndarray]:
    rows = len(frame)
    out = _empty_slot_arrays(rows, int(cfg.output_slots))
    out["best_width_atr"] = np.full(rows, np.nan, dtype="float64")
    out["compression_flag"] = np.zeros(rows, dtype=bool)
    out["best_position"] = np.full(rows, np.nan, dtype="float64")
    for field in ("near_upper", "near_lower", "avoid_long", "avoid_short", "breakout_up", "breakdown_down"):
        out[field] = np.zeros(rows, dtype=bool)

    # TLV2 is the primary line supply. The geometry layer does not rediscover
    # mature trendlines from scratch for compression patterns; it consumes the
    # TLV2 candidate table and asks a narrower question: "Can one active upper
    # line and one active lower line form a tradable pattern right now?"
    #
    # Example: if TLV2 has a resistance line from highs at candles 20 and 42
    # and a support line from lows at candles 25 and 44, geometry v2 can start
    # evaluating the pair only once both lines are active on the current row.
    # That avoids drawing hypothetical future patterns before both rails have
    # been confirmed.
    tl_cfg = _resolve_trendline_v2_config(
        None,
        {
            "timeframe": str(cfg.timeframe),
            "pivot_strength": int(cfg.pivot_strength),
        },
    )
    base = _base_inputs(frame.copy(), tl_cfg)
    candidates = _build_sequence_candidate_table(base, tl_cfg)
    body_high = base["body_high"].to_numpy(dtype="float64")
    body_low = base["body_low"].to_numpy(dtype="float64")
    high = base["high"].to_numpy(dtype="float64")
    low = base["low"].to_numpy(dtype="float64")
    close = base["close"].replace(0.0, np.nan).to_numpy(dtype="float64")
    atr = base["atr"].clip(lower=1e-9).to_numpy(dtype="float64")
    high_pivot = base["pivot_high"].to_numpy(dtype="float64")
    high_index = base["pivot_high_index"].to_numpy(dtype="float64")
    low_pivot = base["pivot_low"].to_numpy(dtype="float64")
    low_index = base["pivot_low_index"].to_numpy(dtype="float64")
    high_events = _pivot_events(high_pivot, high_index)
    low_events = _pivot_events(low_pivot, low_index)
    if candidates.empty:
        resistance_lines: list[_LineCandidate] = []
        support_lines: list[_LineCandidate] = []
    else:
        resistance_lines = _line_candidates_from_frame(candidates, "resistance", atr)
        support_lines = _line_candidates_from_frame(candidates, "support", atr)

    channel_state: dict[str, float] | None = None
    expire_channel_after_row = False
    for row in range(rows):
        if expire_channel_after_row:
            channel_state = None
            expire_channel_after_row = False
        if not np.isfinite(close[row]):
            continue
        pattern_candidates: list[dict[str, float]] = []
        # Stage A: pair live TLV2 resistance/support lines. This is where the
        # mature triangle, wedge, and compression outputs usually come from.
        # The same pair can also classify as a channel if the rails are nearly
        # parallel rather than converging.
        if resistance_lines and support_lines:
            active_resistance = _active_lines_for_row(resistance_lines, row)
            active_support = _active_lines_for_row(support_lines, row)
            for upper in active_resistance:
                for lower_line in active_support:
                    candidate = _pair_lines_as_pattern(
                        row=row,
                        high=high,
                        low=low,
                        atr=atr,
                        body_high=body_high,
                        body_low=body_low,
                        upper=upper,
                        lower_line=lower_line,
                        cfg=cfg,
                    )
                    if candidate is not None:
                        pattern_candidates.append(candidate)
        # Stage B: add channel/rectangle candidates from a pivot envelope. This
        # was added because channels can be more naturally described as "price
        # stayed between two rails" than as "two TLV2 lines happened to pair".
        # It is intentionally selective and should not emit a box around every
        # sideways section.
        pattern_candidates.extend(
            _channel_envelope_candidates(
                row=row,
                high_events=high_events,
                low_events=low_events,
                high=high,
                low=low,
                atr=atr,
                body_high=body_high,
                body_low=body_low,
                cfg=cfg,
            )
        )
        pattern_candidates = _reduce_channel_candidates(pattern_candidates, atr, row, cfg)
        pattern_candidates, channel_state, expire_channel_after_row = _apply_channel_lifecycle(
            candidates=pattern_candidates,
            channel_state=channel_state,
            close=close,
            high=high,
            low=low,
            atr=atr,
            body_high=body_high,
            body_low=body_low,
            row=row,
            cfg=cfg,
        )

        if not pattern_candidates:
            continue

        selected = _select_distinct_patterns(pattern_candidates, atr, row, cfg)
        for slot, candidate in enumerate(selected[: int(cfg.output_slots)], start=1):
            out[f"slot_{slot}_active"][row] = True
            out[f"slot_{slot}_family"][row] = int(candidate["family_code"])
            out[f"slot_{slot}_direction"][row] = int(candidate["direction"])
            out[f"slot_{slot}_upper"][row] = float(candidate["upper"])
            out[f"slot_{slot}_lower"][row] = float(candidate["lower"])
            out[f"slot_{slot}_upper_start"][row] = float(candidate["upper_start"])
            out[f"slot_{slot}_lower_start"][row] = float(candidate["lower_start"])
            out[f"slot_{slot}_upper_slope"][row] = float(candidate["upper_slope"])
            out[f"slot_{slot}_lower_slope"][row] = float(candidate["lower_slope"])
            out[f"slot_{slot}_start_index"][row] = float(candidate["start_index"])
            out[f"slot_{slot}_end_index"][row] = float(candidate["end_index"])
            out[f"slot_{slot}_line_score"][row] = float(candidate["line_score"])
            out[f"slot_{slot}_contraction"][row] = float(candidate["contraction"])
            out[f"slot_{slot}_containment"][row] = float(candidate["containment"])
            out[f"slot_{slot}_width_atr"][row] = float(candidate["width_atr"])
            out[f"slot_{slot}_upper_pivots"][row] = float(candidate["upper_pivots"])
            out[f"slot_{slot}_lower_pivots"][row] = float(candidate["lower_pivots"])
    _suppress_short_output_segments(out, int(cfg.output_slots), int(cfg.min_output_bars))
    _update_row_outputs(out, close, atr, cfg, int(cfg.output_slots))
    return out


def _line_candidates_from_frame(candidates: DataFrame, side: str, atr: np.ndarray) -> list[_LineCandidate]:
    frame = candidates[candidates["side"].eq(side)]
    if frame.empty:
        return []
    lines: list[_LineCandidate] = []
    has_live_start = "live_start" in frame.columns
    for row in frame.itertuples(index=False):
        x_new = float(row.x_new)
        live_start = float(getattr(row, "live_start", x_new)) if has_live_start else x_new
        if not np.isfinite(live_start):
            live_start = x_new
        impulse_span, impulse_slope_atr = _line_impulse_from_anchors(
            x_old=float(row.x_old),
            x_new=x_new,
            y_old=float(row.y_old),
            y_new=float(row.y_new),
            atr=atr,
        )
        lines.append(
            _LineCandidate(
                x_old=float(row.x_old),
                x_new=x_new,
                y_old=float(row.y_old),
                y_new=float(row.y_new),
                slope=float(row.slope),
                intercept=float(row.intercept),
                projection_end=float(row.projection_end),
                active_start=live_start,
                score=float(row.score),
                pivot_count=float(getattr(row, "pivot_count", 2.0)),
                absorbed_pivot_count=float(getattr(row, "absorbed_pivot_count", getattr(row, "pivot_count", 2.0))),
                impulse_span=impulse_span,
                impulse_slope_atr=impulse_slope_atr,
            )
        )
    return lines


def _active_lines_for_row(lines: list[_LineCandidate], row: int) -> list[_LineCandidate]:
    value = float(row)
    return [line for line in lines if line.active_start <= value <= line.projection_end]


def _pivot_events(pivot: np.ndarray, pivot_index: np.ndarray) -> _PivotEvents:
    mask = np.isfinite(pivot) & np.isfinite(pivot_index)
    return _PivotEvents(
        confirmed_at=np.flatnonzero(mask).astype("int64"),
        anchor_index=pivot_index[mask].astype("float64"),
        price=pivot[mask].astype("float64"),
    )


def _pair_lines_as_pattern(
    *,
    row: int,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    body_high: np.ndarray,
    body_low: np.ndarray,
    upper: object,
    lower_line: object,
    cfg: PatternGeometryV2Config,
) -> dict[str, float] | None:
    """Evaluate one active upper/lower TLV2 pair as one geometry candidate.

    The pair is rejected early for structural failures that cannot be fixed by
    scoring later: inverted rails, too-short/too-long span, weak source lines,
    stale touches, post-impulse anchor lines, or candle-body containment that
    does not match the family. Only after those gates pass does the function
    classify the pair as triangle, wedge, compression, rectangle, or channel.

    Example accepted compression: the current rail width is below
    ``compression_max_width_atr``, the last few bars are narrower than the
    lookback width by ``local_narrowing_min_ratio``, and both rails have recent
    contact. Example rejected line-pair: two good TLV2 lines that are parallel
    and wide, but candle bodies repeatedly close outside the projected rails.
    """
    start_index = _pair_start_index(upper, lower_line, str(cfg.pair_start_mode))
    end_index = int(row)
    span = end_index - start_index
    min_allowed_span = min(int(cfg.min_pattern_bars), int(cfg.channel_min_pattern_bars))
    max_allowed_span = max(int(cfg.max_pattern_bars), int(cfg.channel_max_pattern_bars))
    if span < min_allowed_span or span > max_allowed_span:
        return None

    upper_slope = float(upper.slope)
    upper_intercept = float(upper.intercept)
    lower_slope = float(lower_line.slope)
    lower_intercept = float(lower_line.intercept)
    upper_start = upper_slope * float(start_index) + upper_intercept
    lower_start = lower_slope * float(start_index) + lower_intercept
    upper_now = upper_slope * float(row) + upper_intercept
    lower_now = lower_slope * float(row) + lower_intercept
    if (
        not np.isfinite([upper_start, lower_start, upper_now, lower_now]).all()
        or upper_start <= lower_start
        or upper_now <= lower_now
    ):
        return None

    start_width = upper_start - lower_start
    current_width = upper_now - lower_now
    width_atr = current_width / max(float(atr[row]), 1e-9)
    local_contraction = _local_narrowing_ratio(
        start_index=start_index,
        end_index=end_index,
        upper_slope=upper_slope,
        upper_intercept=upper_intercept,
        lower_slope=lower_slope,
        lower_intercept=lower_intercept,
        lookback_bars=int(cfg.local_narrowing_lookback_bars),
    )
    width_change_ratio = abs(current_width - start_width) / max(start_width, current_width, 1e-9)

    line_score = min(float(upper.score), float(lower_line.score))
    if line_score < float(cfg.min_line_score):
        return None
    upper_pivots = float(max(getattr(upper, "absorbed_pivot_count", getattr(upper, "pivot_count", 2.0)), 2.0))
    lower_pivots = float(max(getattr(lower_line, "absorbed_pivot_count", getattr(lower_line, "pivot_count", 2.0)), 2.0))
    if upper_pivots + lower_pivots < float(cfg.min_total_pivots):
        return None
    if _pair_fails_impulse_filter(upper, lower_line, atr, cfg):
        return None

    containment = _containment_ratio(
        body_high=body_high,
        body_low=body_low,
        atr=atr,
        start_index=start_index,
        end_index=end_index,
        upper_slope=upper_slope,
        upper_intercept=upper_intercept,
        lower_slope=lower_slope,
        lower_intercept=lower_intercept,
        tolerance_atr_mult=float(cfg.containment_tolerance_atr_mult),
    )

    touch_tolerance = float(atr[row]) * float(cfg.touch_tolerance_atr_mult)
    upper_touch_age = _recent_touch_age(
        side="upper",
        start_index=start_index,
        end_index=end_index,
        high=high,
        low=low,
        atr=atr,
        slope=upper_slope,
        intercept=upper_intercept,
        tolerance=max(touch_tolerance, 1e-9),
    )
    lower_touch_age = _recent_touch_age(
        side="lower",
        start_index=start_index,
        end_index=end_index,
        high=high,
        low=low,
        atr=atr,
        slope=lower_slope,
        intercept=lower_intercept,
        tolerance=max(touch_tolerance, 1e-9),
    )
    allowed_touch_age = min(int(cfg.max_recent_touch_age_bars), max(span // 2, 4))
    if upper_touch_age > allowed_touch_age or lower_touch_age > allowed_touch_age:
        return None

    span_atr = _atr_window_median(atr, start_index, end_index)
    slope_scale = max(span_atr, max(float(atr[row]), 1e-9), 1e-9)
    upper_slope_atr = upper_slope / slope_scale
    lower_slope_atr = lower_slope / slope_scale
    family = _classify_geometry_family(
        upper_slope_atr=upper_slope_atr,
        lower_slope_atr=lower_slope_atr,
        local_contraction=local_contraction,
        width_atr=width_atr,
        width_change_ratio=width_change_ratio,
        span=span,
        containment=containment,
        cfg=cfg,
    )
    if family is None:
        return None

    if family in {"triangle", "wedge", "compression"}:
        if span < int(cfg.min_pattern_bars) or span > int(cfg.max_pattern_bars):
            return None
        if containment < float(cfg.min_containment):
            return None
        shape_score = max(float(local_contraction), 0.0)
        contraction = float(local_contraction)
    else:
        if span < int(cfg.channel_min_pattern_bars) or span > int(cfg.channel_max_pattern_bars):
            return None
        if upper_pivots < float(cfg.channel_min_side_pivots) or lower_pivots < float(cfg.channel_min_side_pivots):
            return None
        if line_score < float(cfg.channel_min_quality):
            return None
        if containment < float(cfg.channel_min_containment):
            return None
        shape_score = max(0.0, 1.0 - width_change_ratio / max(float(cfg.channel_max_width_change_ratio), 1e-9))
        contraction = float(1.0 - current_width / max(start_width, 1e-9))

    if family == "rectangle":
        direction = 0
    elif family in {"ascending_channel", "descending_channel"}:
        direction = _direction_code(upper_slope_atr, lower_slope_atr, float(cfg.channel_min_slope_atr_per_bar))
    else:
        direction = _direction_code(upper_slope_atr, lower_slope_atr, float(cfg.min_slope_atr_per_bar))
    return {
        "family_code": float(_FAMILY_CODE[family]),
        "direction": float(direction),
        "upper": float(upper_now),
        "lower": float(lower_now),
        "upper_start": float(upper_start),
        "lower_start": float(lower_start),
        "start_index": float(start_index),
        "end_index": float(end_index),
        "upper_slope": float(upper_slope),
        "upper_intercept": float(upper_intercept),
        "lower_slope": float(lower_slope),
        "lower_intercept": float(lower_intercept),
        "line_score": float(line_score),
        "contraction": float(contraction),
        "containment": float(containment),
        "width_atr": float(width_atr),
        "shape_score": float(shape_score),
        "width_change_ratio": float(width_change_ratio),
        "upper_pivots": float(upper_pivots),
        "lower_pivots": float(lower_pivots),
        "upper_touch_age": float(upper_touch_age),
        "lower_touch_age": float(lower_touch_age),
        "span": float(span),
    }


def _channel_envelope_candidates(
    *,
    row: int,
    high_events: _PivotEvents,
    low_events: _PivotEvents,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    body_high: np.ndarray,
    body_low: np.ndarray,
    cfg: PatternGeometryV2Config,
) -> list[dict[str, float]]:
    """Build channel candidates directly from confirmed same-side pivots.

    This is a second channel source, not a replacement for TLV2 line pairs. It
    samples possible start pivots, fits one line through high pivots and one
    line through low pivots, then requires the two fitted rails to behave like
    a stable envelope.

    Example accepted rectangle: at least three high pivots and three low pivots
    span enough of the candidate window, both sides appear in multiple time
    zones, the fitted rails are nearly parallel, and most candle bodies remain
    between the rails. Example rejected human-looking range: price chops inside
    a horizontal box, but the confirmed high pivots are mostly internal highs
    rather than boundary touches; the upper fit drifts into the middle and
    fails width/containment/parallel checks.
    """
    if not bool(cfg.include_channel_patterns):
        return []

    candidates: list[dict[str, float]] = []
    for start_index in _channel_envelope_starts(row, high_events, low_events, cfg):
        span = int(row) - int(start_index)
        if span < int(cfg.channel_min_pattern_bars) or span > int(cfg.channel_max_pattern_bars):
            continue
        high_x, high_y = _confirmed_pivots_between(high_events, row, start_index, row)
        low_x, low_y = _confirmed_pivots_between(low_events, row, start_index, row)
        if len(high_x) < int(cfg.channel_min_side_pivots) or len(low_x) < int(cfg.channel_min_side_pivots):
            continue

        upper_fit = _fit_line(high_x, high_y)
        lower_fit = _fit_line(low_x, low_y)
        if upper_fit is None or lower_fit is None:
            continue
        upper_slope, upper_intercept = upper_fit
        lower_slope, lower_intercept = lower_fit
        upper_start = upper_slope * float(start_index) + upper_intercept
        lower_start = lower_slope * float(start_index) + lower_intercept
        upper_now = upper_slope * float(row) + upper_intercept
        lower_now = lower_slope * float(row) + lower_intercept
        if (
            not np.isfinite([upper_start, lower_start, upper_now, lower_now]).all()
            or upper_start <= lower_start
            or upper_now <= lower_now
        ):
            continue

        atr_scale = max(_atr_window_median(atr, start_index, row), float(atr[row]), 1e-9)
        upper_slope_atr = upper_slope / atr_scale
        lower_slope_atr = lower_slope / atr_scale
        family = _channel_pattern_family(upper_slope_atr, lower_slope_atr, cfg)
        if family is None:
            continue

        start_width = upper_start - lower_start
        current_width = upper_now - lower_now
        width_atr = current_width / max(float(atr[row]), 1e-9)
        if width_atr < float(cfg.channel_min_width_atr) or width_atr > float(cfg.channel_max_width_atr):
            continue
        width_change_ratio = abs(current_width - start_width) / max(start_width, current_width, 1e-9)
        if width_change_ratio > float(cfg.channel_max_width_change_ratio):
            continue

        high_span_ratio = _pivot_span_ratio(high_x, start_index, row)
        low_span_ratio = _pivot_span_ratio(low_x, start_index, row)
        if min(high_span_ratio, low_span_ratio) < float(cfg.channel_min_rail_span_ratio):
            continue
        high_zones = _side_zone_count(high_x, start_index, row)
        low_zones = _side_zone_count(low_x, start_index, row)
        if min(high_zones, low_zones) < int(cfg.channel_min_side_zones):
            continue

        containment = _containment_ratio(
            body_high=body_high,
            body_low=body_low,
            atr=atr,
            start_index=start_index,
            end_index=row,
            upper_slope=upper_slope,
            upper_intercept=upper_intercept,
            lower_slope=lower_slope,
            lower_intercept=lower_intercept,
            tolerance_atr_mult=float(cfg.containment_tolerance_atr_mult),
        )
        if containment < float(cfg.channel_min_containment):
            continue

        touch_tolerance = float(atr[row]) * float(cfg.touch_tolerance_atr_mult)
        upper_touch_age = _recent_touch_age(
            side="upper",
            start_index=start_index,
            end_index=row,
            high=high,
            low=low,
            atr=atr,
            slope=upper_slope,
            intercept=upper_intercept,
            tolerance=max(touch_tolerance, 1e-9),
        )
        lower_touch_age = _recent_touch_age(
            side="lower",
            start_index=start_index,
            end_index=row,
            high=high,
            low=low,
            atr=atr,
            slope=lower_slope,
            intercept=lower_intercept,
            tolerance=max(touch_tolerance, 1e-9),
        )
        allowed_touch_age = min(int(cfg.max_recent_touch_age_bars), max(span // 2, 4))
        if upper_touch_age > allowed_touch_age or lower_touch_age > allowed_touch_age:
            continue

        parallel_error = abs(lower_slope_atr - upper_slope_atr)
        width_stability = max(0.0, 1.0 - width_change_ratio / max(float(cfg.channel_max_width_change_ratio), 1e-9))
        parallel_score = max(0.0, 1.0 - parallel_error / max(float(cfg.channel_parallel_tolerance_atr_per_bar), 1e-9))
        span_score = min(high_span_ratio, low_span_ratio)
        zone_score = min(high_zones, low_zones) / 3.0
        touch_score = min(
            _line_touch_ratio(high_x, high_y, upper_slope, upper_intercept, atr_scale),
            _line_touch_ratio(low_x, low_y, lower_slope, lower_intercept, atr_scale),
        )
        quality = _clip01(
            0.30 * containment
            + 0.22 * parallel_score
            + 0.18 * width_stability
            + 0.14 * span_score
            + 0.10 * touch_score
            + 0.06 * zone_score
        )
        if quality < float(cfg.channel_min_quality):
            continue

        direction = 0 if family == "rectangle" else _direction_code(upper_slope_atr, lower_slope_atr, float(cfg.channel_min_slope_atr_per_bar))
        candidates.append(
            {
                "family_code": float(_FAMILY_CODE[family]),
                "direction": float(direction),
                "upper": float(upper_now),
                "lower": float(lower_now),
                "upper_start": float(upper_start),
                "lower_start": float(lower_start),
                "start_index": float(start_index),
                "end_index": float(row),
                "upper_slope": float(upper_slope),
                "upper_intercept": float(upper_intercept),
                "lower_slope": float(lower_slope),
                "lower_intercept": float(lower_intercept),
                "line_score": float(quality),
                "contraction": float(1.0 - current_width / max(start_width, 1e-9)),
                "containment": float(containment),
                "width_atr": float(width_atr),
                "shape_score": float(quality),
                "width_change_ratio": float(width_change_ratio),
                "upper_pivots": float(len(high_x)),
                "lower_pivots": float(len(low_x)),
                "upper_touch_age": float(upper_touch_age),
                "lower_touch_age": float(lower_touch_age),
                "span": float(span),
            }
        )
    return candidates


def _channel_envelope_starts(
    row: int,
    high_events: _PivotEvents,
    low_events: _PivotEvents,
    cfg: PatternGeometryV2Config,
) -> np.ndarray:
    min_start = int(row) - int(cfg.channel_max_pattern_bars)
    max_start = int(row) - int(cfg.channel_min_pattern_bars)
    if max_start <= 0:
        return np.asarray([], dtype="int64")
    event_mask_high = high_events.confirmed_at <= int(row)
    event_mask_low = low_events.confirmed_at <= int(row)
    events = np.unique(
        np.concatenate(
            [
                high_events.anchor_index[event_mask_high],
                low_events.anchor_index[event_mask_low],
            ]
        ).astype("int64")
    )
    eligible = events[(events >= max(0, min_start)) & (events <= max_start)]
    if eligible.size <= int(cfg.channel_envelope_start_options):
        return eligible
    positions = np.linspace(0, eligible.size - 1, int(cfg.channel_envelope_start_options)).round().astype("int64")
    return eligible[positions]


def _confirmed_pivots_between(
    events: _PivotEvents,
    row: int,
    start_index: int,
    end_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (
        (events.confirmed_at <= int(row))
        & (events.anchor_index >= float(start_index))
        & (events.anchor_index <= float(end_index))
    )
    x = events.anchor_index[mask]
    y = events.price[mask]
    return x, y


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    if len(x) < 2 or len(y) < 2:
        return None
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    x_float = x.astype("float64", copy=False)
    y_float = y.astype("float64", copy=False)
    count = float(len(x_float))
    sum_x = float(np.sum(x_float))
    sum_y = float(np.sum(y_float))
    denominator = count * float(np.dot(x_float, x_float)) - sum_x * sum_x
    if denominator <= 0.0:
        return None
    slope = float((count * float(np.dot(x_float, y_float)) - sum_x * sum_y) / denominator)
    intercept = float((sum_y - slope * sum_x) / count)
    if not np.isfinite([slope, intercept]).all():
        return None
    return float(slope), float(intercept)


def _atr_window_median(atr: np.ndarray, start_index: int, end_index: int) -> float:
    if end_index < start_index:
        return np.nan
    values = atr[max(int(start_index), 0) : int(end_index) + 1]
    if values.size == 0:
        return np.nan
    if np.isfinite(values).all():
        return float(np.median(values))
    return float(np.nanmedian(values))


def _channel_pattern_family(upper_slope_atr: float, lower_slope_atr: float, cfg: PatternGeometryV2Config) -> str | None:
    if not np.isfinite([upper_slope_atr, lower_slope_atr]).all():
        return None
    if abs(float(lower_slope_atr) - float(upper_slope_atr)) > float(cfg.channel_parallel_tolerance_atr_per_bar):
        return None
    mid_slope = 0.5 * (float(upper_slope_atr) + float(lower_slope_atr))
    if abs(mid_slope) <= float(cfg.channel_flat_slope_atr_per_bar):
        return "rectangle"
    if mid_slope >= float(cfg.channel_min_slope_atr_per_bar):
        return "ascending_channel"
    if mid_slope <= -float(cfg.channel_min_slope_atr_per_bar):
        return "descending_channel"
    return None


def _pivot_span_ratio(x: np.ndarray, start_index: int, end_index: int) -> float:
    span = max(float(end_index - start_index), 1.0)
    if len(x) < 2:
        return 0.0
    return float((np.nanmax(x) - np.nanmin(x)) / span)


def _side_zone_count(x: np.ndarray, start_index: int, end_index: int) -> int:
    span = max(float(end_index - start_index), 1.0)
    if len(x) == 0:
        return 0
    zones = np.floor(np.clip((x - float(start_index)) / span, 0.0, 0.999999) * 3.0).astype("int64")
    return int(len(np.unique(zones)))


def _line_touch_ratio(x: np.ndarray, y: np.ndarray, slope: float, intercept: float, atr_scale: float) -> float:
    if len(x) == 0:
        return 0.0
    distance = np.abs(y - (float(slope) * x + float(intercept)))
    return float(np.nanmean((distance <= max(float(atr_scale) * 0.75, 1e-9)).astype("float64")))


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(max(value, 0.0), 1.0))


def _containment_ratio(
    *,
    body_high: np.ndarray,
    body_low: np.ndarray,
    atr: np.ndarray,
    start_index: int,
    end_index: int,
    upper_slope: float,
    upper_intercept: float,
    lower_slope: float,
    lower_intercept: float,
    tolerance_atr_mult: float,
) -> float:
    if end_index <= start_index:
        return 0.0
    x = np.arange(start_index, end_index + 1, dtype="float64")
    tolerance = np.maximum(atr[start_index : end_index + 1] * float(tolerance_atr_mult), 1e-9)
    upper_line = upper_slope * x + upper_intercept
    lower_line = lower_slope * x + lower_intercept
    upper_intrusion = np.maximum(body_high[start_index : end_index + 1] - upper_line, 0.0)
    lower_intrusion = np.maximum(lower_line - body_low[start_index : end_index + 1], 0.0)
    respected = (upper_intrusion <= tolerance) & (lower_intrusion <= tolerance)
    return float(np.nanmean(respected.astype("float64")))


def _pair_start_index(upper: object, lower_line: object, mode: str) -> int:
    if mode == "x_new":
        return int(round(max(float(upper.x_new), float(lower_line.x_new))))
    if mode == "active_start":
        return int(round(max(float(upper.active_start), float(lower_line.active_start))))
    return int(round(max(float(upper.x_old), float(lower_line.x_old))))


def _local_narrowing_ratio(
    *,
    start_index: int,
    end_index: int,
    upper_slope: float,
    upper_intercept: float,
    lower_slope: float,
    lower_intercept: float,
    lookback_bars: int,
) -> float:
    if end_index <= start_index:
        return 0.0
    lookback_index = max(start_index, end_index - max(int(lookback_bars), 1))
    if lookback_index >= end_index:
        return 0.0
    prior_upper = upper_slope * float(lookback_index) + upper_intercept
    prior_lower = lower_slope * float(lookback_index) + lower_intercept
    current_upper = upper_slope * float(end_index) + upper_intercept
    current_lower = lower_slope * float(end_index) + lower_intercept
    prior_width = prior_upper - prior_lower
    current_width = current_upper - current_lower
    if not np.isfinite([prior_width, current_width]).all() or prior_width <= 0.0 or current_width <= 0.0:
        return 0.0
    return float(1.0 - current_width / max(prior_width, 1e-9))


def _pair_fails_impulse_filter(
    upper: object,
    lower_line: object,
    atr: np.ndarray,
    cfg: PatternGeometryV2Config,
) -> bool:
    return _line_fails_impulse_filter(upper, atr, cfg) or _line_fails_impulse_filter(lower_line, atr, cfg)


def _line_fails_impulse_filter(line: object, atr: np.ndarray, cfg: PatternGeometryV2Config) -> bool:
    metrics = _line_impulse_metrics(line, atr)
    if metrics is None:
        return False
    span, slope_atr = metrics
    if span < float(cfg.impulse_min_span_bars):
        return False
    return slope_atr >= float(cfg.impulse_max_slope_atr_per_bar)


def _line_impulse_metrics(line: object, atr: np.ndarray) -> tuple[float, float] | None:
    cached_span = getattr(line, "impulse_span", np.nan)
    cached_slope = getattr(line, "impulse_slope_atr", np.nan)
    if np.isfinite([cached_span, cached_slope]).all():
        return float(cached_span), float(cached_slope)
    x_old = int(round(float(getattr(line, "x_old"))))
    x_new = int(round(float(getattr(line, "x_new"))))
    y_old = _line_anchor_value(line, "x_old", "y_old")
    y_new = _line_anchor_value(line, "x_new", "y_new")
    span, slope_atr = _line_impulse_from_anchors(
        x_old=float(x_old),
        x_new=float(x_new),
        y_old=float(y_old),
        y_new=float(y_new),
        atr=atr,
    )
    if not np.isfinite([span, slope_atr]).all():
        return None
    return span, slope_atr


def _line_impulse_from_anchors(
    *,
    x_old: float,
    x_new: float,
    y_old: float,
    y_new: float,
    atr: np.ndarray,
) -> tuple[float, float]:
    old_index = int(round(float(x_old)))
    new_index = int(round(float(x_new)))
    span = float(new_index - old_index)
    if span <= 0.0:
        return np.nan, np.nan
    atr_scale = _atr_window_median(atr, x_old, x_new)
    atr_scale = max(atr_scale, 1e-9)
    slope_atr = abs((y_new - y_old) / span) / atr_scale
    return span, float(slope_atr)


def _line_anchor_value(line: object, x_field: str, y_field: str) -> float:
    y = getattr(line, y_field, np.nan)
    if np.isfinite(y):
        return float(y)
    x = float(getattr(line, x_field))
    return float(getattr(line, "slope")) * x + float(getattr(line, "intercept"))


def _recent_touch_age(
    *,
    side: str,
    start_index: int,
    end_index: int,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    slope: float,
    intercept: float,
    tolerance: float,
) -> int:
    if end_index <= start_index:
        return int(1e9)
    x = np.arange(start_index, end_index + 1, dtype="float64")
    line = slope * x + intercept
    if side == "upper":
        distance = np.abs(high[start_index : end_index + 1] - line)
    else:
        distance = np.abs(low[start_index : end_index + 1] - line)
    touches = np.isfinite(distance) & np.isfinite(atr[start_index : end_index + 1]) & (distance <= tolerance)
    if not np.any(touches):
        return int(1e9)
    last_touch = int(np.flatnonzero(touches)[-1]) + start_index
    return int(end_index - last_touch)


def _classify_geometry_family(
    *,
    upper_slope_atr: float,
    lower_slope_atr: float,
    local_contraction: float,
    width_atr: float,
    width_change_ratio: float,
    span: int,
    containment: float,
    cfg: PatternGeometryV2Config,
) -> str | None:
    # Classification is intentionally ordered from "compressing setup" to
    # "parallel channel". A pair that is already narrow and locally narrowing
    # should be treated as triangle/wedge/compression first, even if the rails
    # are close to parallel for a short stretch. Only non-compressive pairs are
    # allowed to fall through to rectangle/channel classification.
    #
    # Example: two upward-sloping rails whose width is shrinking are a wedge,
    # not an ascending channel. Two upward-sloping rails whose width stays
    # stable are an ascending channel. A flat upper rail with a rising lower
    # rail is a triangle when the local narrowing gate passes.
    if (
        width_atr <= float(cfg.compression_max_width_atr)
        and local_contraction >= float(cfg.local_narrowing_min_ratio)
        and containment >= float(cfg.min_containment)
    ):
        compression_family = _compressive_pattern_family(
            upper_slope_atr=upper_slope_atr,
            lower_slope_atr=lower_slope_atr,
            min_slope=float(cfg.min_slope_atr_per_bar),
            flat_slope=float(cfg.flat_slope_atr_per_bar),
        )
        if compression_family is not None:
            return compression_family

    if not bool(cfg.include_channel_patterns):
        return None
    if span < int(cfg.channel_min_pattern_bars):
        return None
    if span > int(cfg.channel_max_pattern_bars):
        return None
    if width_atr < float(cfg.channel_min_width_atr) or width_atr > float(cfg.channel_max_width_atr):
        return None
    if width_change_ratio > float(cfg.channel_max_width_change_ratio):
        return None
    if abs(float(lower_slope_atr) - float(upper_slope_atr)) > float(cfg.channel_parallel_tolerance_atr_per_bar):
        return None

    mid_slope = 0.5 * (float(upper_slope_atr) + float(lower_slope_atr))
    if abs(mid_slope) <= float(cfg.channel_flat_slope_atr_per_bar):
        return "rectangle"
    if mid_slope >= float(cfg.channel_min_slope_atr_per_bar):
        return "ascending_channel"
    if mid_slope <= -float(cfg.channel_min_slope_atr_per_bar):
        return "descending_channel"
    return None


def _compressive_pattern_family(
    *,
    upper_slope_atr: float,
    lower_slope_atr: float,
    min_slope: float,
    flat_slope: float,
) -> str | None:
    if not np.isfinite([upper_slope_atr, lower_slope_atr]).all():
        return None
    slope_gap = float(lower_slope_atr) - float(upper_slope_atr)
    if slope_gap < float(min_slope):
        return None

    upper_flat = abs(float(upper_slope_atr)) <= float(flat_slope)
    lower_flat = abs(float(lower_slope_atr)) <= float(flat_slope)
    upper_down = float(upper_slope_atr) <= -float(min_slope)
    upper_up = float(upper_slope_atr) >= float(min_slope)
    lower_down = float(lower_slope_atr) <= -float(min_slope)
    lower_up = float(lower_slope_atr) >= float(min_slope)

    if (upper_flat and lower_up) or (upper_down and lower_flat) or (upper_down and lower_up):
        return "triangle"
    if (upper_up and lower_up) or (upper_down and lower_down):
        return "wedge"
    return "compression"


def _direction_code(upper_slope: float, lower_slope: float, min_slope: float) -> int:
    mid = 0.5 * (float(upper_slope) + float(lower_slope))
    if mid >= float(min_slope):
        return 1
    if mid <= -float(min_slope):
        return -1
    return 0


def _select_distinct_patterns(
    candidates: list[dict[str, float]],
    atr: np.ndarray,
    row: int,
    cfg: PatternGeometryV2Config,
) -> list[dict[str, float]]:
    selected: list[dict[str, float]] = []
    channel_count = 0
    for candidate in sorted(candidates, key=_pattern_rank_key, reverse=True):
        is_channel = _is_channel_candidate(candidate)
        if is_channel and channel_count >= int(cfg.channel_max_active_outputs):
            continue
        if any(_same_pattern(candidate, existing, atr, row) for existing in selected):
            continue
        selected.append(candidate)
        if is_channel:
            channel_count += 1
    return selected


def _reduce_channel_candidates(
    candidates: list[dict[str, float]],
    atr: np.ndarray,
    row: int,
    cfg: PatternGeometryV2Config,
) -> list[dict[str, float]]:
    if not bool(cfg.channel_merge_enabled):
        return candidates
    non_channels = [candidate for candidate in candidates if not _is_channel_candidate(candidate)]
    channel_candidates = [candidate for candidate in candidates if _is_channel_candidate(candidate)]
    clusters: list[list[dict[str, float]]] = []
    for candidate in sorted(channel_candidates, key=_pattern_rank_key, reverse=True):
        for cluster in clusters:
            if _same_channel_cluster(candidate, cluster[0], atr, row, cfg):
                cluster.append(candidate)
                break
        else:
            clusters.append([candidate])
    reduced = [_channel_cluster_representative(cluster) for cluster in clusters]
    return non_channels + reduced


def _apply_channel_lifecycle(
    *,
    candidates: list[dict[str, float]],
    channel_state: dict[str, float] | None,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    body_high: np.ndarray,
    body_low: np.ndarray,
    row: int,
    cfg: PatternGeometryV2Config,
) -> tuple[list[dict[str, float]], dict[str, float] | None, bool]:
    """Keep one selected channel alive while it remains technically valid.

    Raw channel candidates can flicker because the best-fit envelope changes
    as new pivots confirm. The lifecycle state stabilizes output by carrying
    the selected channel forward until one of three things happens:

    - the channel grows beyond its allowed bar span,
    - the projected width leaves the allowed ATR range,
    - price closes beyond a rail by ``channel_lifecycle_break_atr_mult`` for
      ``channel_lifecycle_confirm_break_bars`` consecutive candles.

    Example: if a rectangle is selected on candle 100, candle 101 can keep the
    same rectangle even if no fresh candidate is rebuilt on that exact row.
    If candle 102 closes above the upper rail but candle 103 repairs back
    inside, the same channel remains active. If both candles close beyond the
    break distance, the state is expired after candle 103.
    """
    if not bool(cfg.channel_lifecycle_enabled):
        return candidates, channel_state, False

    non_channels = [candidate for candidate in candidates if not _is_channel_candidate(candidate)]
    channel_candidates = [candidate for candidate in candidates if _is_channel_candidate(candidate)]
    if channel_state is not None:
        projected = _project_channel_state(
            channel_state=channel_state,
            high=high,
            low=low,
            atr=atr,
            body_high=body_high,
            body_low=body_low,
            row=row,
            cfg=cfg,
        )
        if projected is not None:
            projected = _refresh_projected_channel(
                projected=projected,
                channel_candidates=channel_candidates,
                atr=atr,
                row=row,
                cfg=cfg,
            )
            break_run = _channel_break_run_after_row(channel_state, projected, close, atr, row, cfg)
            refreshed_state = _channel_state_from_candidate(projected)
            refreshed_state["break_run"] = float(break_run)
            channel_state = refreshed_state
            return non_channels + [projected], channel_state, break_run >= int(cfg.channel_lifecycle_confirm_break_bars)
        channel_state = None

    if not channel_candidates:
        return non_channels, None, False

    selected = max(channel_candidates, key=_pattern_rank_key)
    channel_state = _channel_state_from_candidate(selected)
    break_run = _channel_break_run_after_row(channel_state, selected, close, atr, row, cfg)
    channel_state["break_run"] = float(break_run)
    return non_channels + [selected], channel_state, break_run >= int(cfg.channel_lifecycle_confirm_break_bars)


def _channel_state_from_candidate(candidate: dict[str, float]) -> dict[str, float]:
    return {
        key: float(candidate[key])
        for key in (
            "family_code",
            "direction",
            "upper_start",
            "lower_start",
            "start_index",
            "upper_slope",
            "upper_intercept",
            "lower_slope",
            "lower_intercept",
            "line_score",
            "containment",
            "upper_pivots",
            "lower_pivots",
            "shape_score",
        )
        if key in candidate
    }


def _project_channel_state(
    *,
    channel_state: dict[str, float],
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    body_high: np.ndarray,
    body_low: np.ndarray,
    row: int,
    cfg: PatternGeometryV2Config,
) -> dict[str, float] | None:
    start_index = int(round(float(channel_state["start_index"])))
    span = int(row) - start_index
    if span < int(cfg.channel_min_pattern_bars) or span > int(cfg.channel_max_pattern_bars):
        return None

    upper_slope = float(channel_state["upper_slope"])
    lower_slope = float(channel_state["lower_slope"])
    upper_intercept = float(channel_state["upper_intercept"])
    lower_intercept = float(channel_state["lower_intercept"])
    upper_start = float(channel_state["upper_start"])
    lower_start = float(channel_state["lower_start"])
    upper_now = upper_slope * float(row) + upper_intercept
    lower_now = lower_slope * float(row) + lower_intercept
    if not np.isfinite([upper_now, lower_now, upper_start, lower_start]).all() or upper_now <= lower_now:
        return None

    current_width = upper_now - lower_now
    start_width = upper_start - lower_start
    width_atr = current_width / max(float(atr[row]), 1e-9)
    if width_atr < float(cfg.channel_min_width_atr) or width_atr > float(cfg.channel_max_width_atr):
        return None

    containment = _containment_ratio(
        body_high=body_high,
        body_low=body_low,
        atr=atr,
        start_index=start_index,
        end_index=row,
        upper_slope=upper_slope,
        upper_intercept=upper_intercept,
        lower_slope=lower_slope,
        lower_intercept=lower_intercept,
        tolerance_atr_mult=float(cfg.containment_tolerance_atr_mult),
    )

    touch_tolerance = float(atr[row]) * float(cfg.touch_tolerance_atr_mult)
    upper_touch_age = _recent_touch_age(
        side="upper",
        start_index=start_index,
        end_index=row,
        high=high,
        low=low,
        atr=atr,
        slope=upper_slope,
        intercept=upper_intercept,
        tolerance=max(touch_tolerance, 1e-9),
    )
    lower_touch_age = _recent_touch_age(
        side="lower",
        start_index=start_index,
        end_index=row,
        high=high,
        low=low,
        atr=atr,
        slope=lower_slope,
        intercept=lower_intercept,
        tolerance=max(touch_tolerance, 1e-9),
    )

    return {
        "family_code": float(channel_state["family_code"]),
        "direction": float(channel_state["direction"]),
        "upper": float(upper_now),
        "lower": float(lower_now),
        "upper_start": float(upper_start),
        "lower_start": float(lower_start),
        "start_index": float(start_index),
        "end_index": float(row),
        "upper_slope": float(upper_slope),
        "upper_intercept": float(upper_intercept),
        "lower_slope": float(lower_slope),
        "lower_intercept": float(lower_intercept),
        "line_score": float(channel_state.get("line_score", 0.0)),
        "contraction": float(1.0 - current_width / max(start_width, 1e-9)),
        "containment": float(max(containment, float(channel_state.get("containment", 0.0)))),
        "width_atr": float(width_atr),
        "shape_score": float(channel_state.get("shape_score", channel_state.get("line_score", 0.0))),
        "width_change_ratio": float(abs(current_width - start_width) / max(start_width, current_width, 1e-9)),
        "upper_pivots": float(channel_state.get("upper_pivots", 2.0)),
        "lower_pivots": float(channel_state.get("lower_pivots", 2.0)),
        "upper_touch_age": float(upper_touch_age),
        "lower_touch_age": float(lower_touch_age),
        "span": float(span),
    }


def _channel_breaks_after_row(
    candidate: dict[str, float],
    close: np.ndarray,
    atr: np.ndarray,
    row: int,
    cfg: PatternGeometryV2Config,
) -> bool:
    if not np.isfinite(close[row]):
        return False
    distance = max(float(atr[row]) * float(cfg.channel_lifecycle_break_atr_mult), 1e-9)
    return bool(
        float(close[row]) > float(candidate["upper"]) + distance
        or float(close[row]) < float(candidate["lower"]) - distance
    )


def _refresh_projected_channel(
    *,
    projected: dict[str, float],
    channel_candidates: list[dict[str, float]],
    atr: np.ndarray,
    row: int,
    cfg: PatternGeometryV2Config,
) -> dict[str, float]:
    compatible = [
        candidate
        for candidate in channel_candidates
        if _same_channel_cluster(candidate, projected, atr, row, cfg)
    ]
    if not compatible:
        return projected
    return _channel_cluster_representative([projected, *compatible])


def _channel_break_run_after_row(
    channel_state: dict[str, float],
    candidate: dict[str, float],
    close: np.ndarray,
    atr: np.ndarray,
    row: int,
    cfg: PatternGeometryV2Config,
) -> int:
    previous = int(round(float(channel_state.get("break_run", 0.0))))
    if _channel_breaks_after_row(candidate, close, atr, row, cfg):
        return previous + 1
    return 0


def _is_channel_candidate(candidate: dict[str, float]) -> bool:
    return int(candidate["family_code"]) not in _COMPRESSIVE_FAMILY_CODES


def _same_channel_cluster(
    candidate: dict[str, float],
    existing: dict[str, float],
    atr: np.ndarray,
    row: int,
    cfg: PatternGeometryV2Config,
) -> bool:
    if int(candidate["family_code"]) != int(existing["family_code"]):
        return False
    if int(candidate["direction"]) != int(existing["direction"]):
        return False
    overlap = _span_overlap_ratio(
        float(candidate["start_index"]),
        float(candidate["end_index"]),
        float(existing["start_index"]),
        float(existing["end_index"]),
    )
    if overlap < float(cfg.channel_merge_min_span_overlap):
        return False
    scale = max(float(atr[row]), 1e-9)
    upper_distance = abs(float(candidate["upper"]) - float(existing["upper"])) / scale
    lower_distance = abs(float(candidate["lower"]) - float(existing["lower"])) / scale
    if max(upper_distance, lower_distance) > float(cfg.channel_merge_max_boundary_distance_atr):
        return False
    width_diff = abs(float(candidate["width_atr"]) - float(existing["width_atr"]))
    if width_diff > float(cfg.channel_merge_max_width_diff_atr):
        return False
    slope_diff = max(
        abs(float(candidate["upper_slope"]) - float(existing["upper_slope"])) / scale,
        abs(float(candidate["lower_slope"]) - float(existing["lower_slope"])) / scale,
    )
    return slope_diff <= float(cfg.channel_merge_max_slope_diff_atr_per_bar)


def _span_overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    a_span = max(float(a_end) - float(a_start), 1.0)
    b_span = max(float(b_end) - float(b_start), 1.0)
    overlap = max(0.0, min(float(a_end), float(b_end)) - max(float(a_start), float(b_start)))
    return float(overlap / max(min(a_span, b_span), 1e-9))


def _channel_cluster_representative(cluster: list[dict[str, float]]) -> dict[str, float]:
    return max(cluster, key=_pattern_rank_key)


def _pattern_rank_key(candidate: dict[str, float]) -> tuple[float, float, float, float, float, float]:
    return (
        min(float(candidate["upper_pivots"]), float(candidate["lower_pivots"])),
        float(candidate["line_score"]),
        float(candidate["containment"]),
        float(candidate.get("shape_score", candidate["contraction"])),
        -max(float(candidate["upper_touch_age"]), float(candidate["lower_touch_age"])),
        -float(candidate["width_atr"]),
    )


def _same_pattern(
    candidate: dict[str, float],
    existing: dict[str, float],
    atr: np.ndarray,
    row: int,
) -> bool:
    scale = max(float(atr[row]), 1e-9)
    upper_distance = abs(float(candidate["upper"]) - float(existing["upper"])) / scale
    lower_distance = abs(float(candidate["lower"]) - float(existing["lower"])) / scale
    span = max(abs(float(candidate["end_index"]) - float(candidate["start_index"])), 1.0)
    start_distance = abs(float(candidate["start_index"]) - float(existing["start_index"])) / span
    return upper_distance <= 1.25 and lower_distance <= 1.25 and start_distance <= 0.25


def _empty_slot_arrays(rows: int, slot_count: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for slot in range(1, slot_count + 1):
        for field in _SLOT_FIELDS:
            key = f"slot_{slot}_{field}"
            if field == "active":
                out[key] = np.zeros(rows, dtype=bool)
            elif field in {"family", "direction"}:
                out[key] = np.zeros(rows, dtype="int8")
            else:
                out[key] = np.full(rows, np.nan, dtype="float64")
    return out


def _suppress_short_output_segments(out: dict[str, np.ndarray], slot_count: int, min_output_bars: int) -> None:
    if min_output_bars <= 1:
        return
    for slot in range(1, slot_count + 1):
        active = out[f"slot_{slot}_active"]
        family = out[f"slot_{slot}_family"]
        start_index = out[f"slot_{slot}_start_index"]
        segment_start: int | None = None
        for row in range(len(active) + 1):
            is_active = row < len(active) and bool(active[row])
            same_prev = False
            if row > 0 and is_active and segment_start is not None:
                same_prev = (
                    bool(active[row - 1])
                    and int(family[row]) == int(family[row - 1])
                    and abs(float(start_index[row]) - float(start_index[row - 1])) < 0.5
                )
            if segment_start is None:
                if is_active:
                    segment_start = row
                continue
            if is_active and same_prev:
                continue
            segment_end = row - 1
            if segment_end - segment_start + 1 < min_output_bars:
                _clear_slot_range(out, slot, segment_start, segment_end)
            segment_start = row if is_active else None


def _clear_slot_range(out: dict[str, np.ndarray], slot: int, start: int, end: int) -> None:
    if end < start:
        return
    for field in _SLOT_FIELDS:
        key = f"slot_{slot}_{field}"
        values = out[key]
        if field == "active":
            values[start : end + 1] = False
        elif field in {"family", "direction"}:
            values[start : end + 1] = 0
        else:
            values[start : end + 1] = np.nan


def _update_row_outputs(
    out: dict[str, np.ndarray],
    close: np.ndarray,
    atr: np.ndarray,
    cfg: PatternGeometryV2Config,
    slot_count: int,
) -> None:
    """Derive row-level helper flags from selected slot outputs.

    Slot columns are the authoritative geometry data. Row-level fields are a
    convenience layer for strategy research:

    - ``best_width_atr`` and ``compression_flag`` summarize the tightest active
      compressive pattern.
    - ``best_position`` maps current close inside the best channel: 0 is at
      the lower rail, 1 is at the upper rail, below 0 means breakdown, and
      above 1 means breakout.
    - ``near_upper`` and ``near_lower`` are avoidance/context flags. For
      example, a long system can avoid fresh longs when ``near_upper`` is true
      because reward-to-risk is likely poor near channel resistance.
    - ``breakout_up`` and ``breakdown_down`` are rail breach flags, not full
      trade recommendations.
    """
    threshold = float(cfg.compression_flag_atr_threshold)
    for row in range(len(atr)):
        compression_widths: list[float] = []
        best_channel: dict[str, float] | None = None
        scale = max(float(atr[row]), 1e-9)
        for slot in range(1, slot_count + 1):
            if not bool(out[f"slot_{slot}_active"][row]):
                continue
            upper = float(out[f"slot_{slot}_upper"][row])
            lower = float(out[f"slot_{slot}_lower"][row])
            if not np.isfinite([upper, lower]).all():
                continue
            family_code = int(out[f"slot_{slot}_family"][row])
            width = upper - lower
            width_atr = width / scale
            if family_code in _COMPRESSIVE_FAMILY_CODES:
                compression_widths.append(width_atr)
            else:
                line_score = float(out[f"slot_{slot}_line_score"][row])
                containment = float(out[f"slot_{slot}_containment"][row])
                candidate = {
                    "upper": upper,
                    "lower": lower,
                    "width": width,
                    "width_atr": width_atr,
                    "line_score": line_score,
                    "containment": containment,
                }
                if best_channel is None or _row_channel_rank_key(candidate) > _row_channel_rank_key(best_channel):
                    best_channel = candidate
        if compression_widths:
            best = float(np.nanmin(np.asarray(compression_widths, dtype="float64")))
            out["best_width_atr"][row] = best
            out["compression_flag"][row] = bool(best <= threshold)
        if best_channel is None or not np.isfinite(close[row]) or best_channel["width"] <= 0.0:
            continue
        position = (float(close[row]) - float(best_channel["lower"])) / max(float(best_channel["width"]), 1e-9)
        near_distance = scale * float(cfg.channel_near_boundary_atr_mult)
        breakout_distance = scale * float(cfg.channel_breakout_atr_mult)
        near_upper = bool(float(best_channel["upper"]) - float(close[row]) <= near_distance and float(close[row]) <= float(best_channel["upper"]) + breakout_distance)
        near_lower = bool(float(close[row]) - float(best_channel["lower"]) <= near_distance and float(close[row]) >= float(best_channel["lower"]) - breakout_distance)
        breakout_up = bool(float(close[row]) > float(best_channel["upper"]) + breakout_distance)
        breakdown_down = bool(float(close[row]) < float(best_channel["lower"]) - breakout_distance)
        out["best_position"][row] = float(position)
        out["near_upper"][row] = near_upper
        out["near_lower"][row] = near_lower
        out["breakout_up"][row] = breakout_up
        out["breakdown_down"][row] = breakdown_down
        out["avoid_long"][row] = bool(near_upper or breakdown_down)
        out["avoid_short"][row] = bool(near_lower or breakout_up)


def _row_channel_rank_key(candidate: dict[str, float]) -> tuple[float, float, float]:
    return (
        float(candidate["line_score"]),
        float(candidate["containment"]),
        -abs((float(candidate["width_atr"]) - 3.0)),
    )


def _resolve_config(config: PatternGeometryV2Config | None, overrides: dict[str, object]) -> PatternGeometryV2Config:
    base = config or PatternGeometryV2Config()
    valid = {field.name for field in fields(PatternGeometryV2Config)}
    unknown = sorted(set(overrides).difference(valid))
    if unknown:
        raise TypeError(f"Unknown geometry v2 config override(s): {', '.join(unknown)}")
    requested = replace(base, **{name: overrides[name] for name in overrides if name in valid})
    timeframe = _normalize_timeframe(str(requested.timeframe))
    requested = replace(requested, timeframe=timeframe)
    profile = _TIMEFRAME_PROFILES[timeframe]
    if config is None:
        profiled = replace(requested, **profile)
    else:
        default = PatternGeometryV2Config()
        profile_updates = {
            name: profile[name]
            for name in _PROFILE_FIELDS
            if name not in overrides and getattr(config, name) == getattr(default, name)
        }
        profiled = replace(requested, **profile_updates)
    if overrides:
        profiled = replace(profiled, **{name: overrides[name] for name in overrides if name in valid})
        profiled = replace(profiled, timeframe=_normalize_timeframe(str(profiled.timeframe)))
    return profiled


def _normalize_timeframe(value: str) -> str:
    key = str(value).strip().lower()
    if key not in _TIMEFRAME_ALIASES:
        allowed = ", ".join(sorted(_TIMEFRAME_PROFILES))
        raise ValueError(f"timeframe must be one of: {allowed}")
    return _TIMEFRAME_ALIASES[key]


def _validate_dataframe(frame: DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Geometry v2 requires OHLCV columns. Missing: {', '.join(missing)}")


def _validate_config(cfg: PatternGeometryV2Config) -> None:
    if not cfg.output_prefix:
        raise ValueError("output_prefix must be set")
    if int(cfg.output_slots) < 1:
        raise ValueError("output_slots must be at least 1")
    _normalize_timeframe(str(cfg.timeframe))
    if int(cfg.pivot_strength) < 1:
        raise ValueError("pivot_strength must be at least 1")
    if str(cfg.pair_start_mode) not in {"old", "x_new", "active_start"}:
        raise ValueError("pair_start_mode must be one of: old, x_new, active_start")
    if int(cfg.min_pattern_bars) < 4:
        raise ValueError("min_pattern_bars must be at least 4")
    if int(cfg.max_pattern_bars) <= int(cfg.min_pattern_bars):
        raise ValueError("max_pattern_bars must be greater than min_pattern_bars")
    if float(cfg.compression_max_width_atr) <= 0.0:
        raise ValueError("compression_max_width_atr must be positive")
    if int(cfg.local_narrowing_lookback_bars) < 1:
        raise ValueError("local_narrowing_lookback_bars must be at least 1")
    if not 0.0 <= float(cfg.local_narrowing_min_ratio) <= 1.0:
        raise ValueError("local_narrowing_min_ratio must be between 0 and 1")
    if not 0.0 <= float(cfg.min_containment) <= 1.0:
        raise ValueError("min_containment must be between 0 and 1")
    if not 0.0 <= float(cfg.min_line_score) <= 1.0:
        raise ValueError("min_line_score must be between 0 and 1")
    if float(cfg.compression_flag_atr_threshold) <= 0.0:
        raise ValueError("compression_flag_atr_threshold must be positive")
    if int(cfg.channel_min_pattern_bars) < 4:
        raise ValueError("channel_min_pattern_bars must be at least 4")
    if int(cfg.channel_max_pattern_bars) <= int(cfg.channel_min_pattern_bars):
        raise ValueError("channel_max_pattern_bars must be greater than channel_min_pattern_bars")
    if not 0.0 < float(cfg.channel_min_width_atr) < float(cfg.channel_max_width_atr):
        raise ValueError("channel_min_width_atr must be lower than channel_max_width_atr")
    if float(cfg.channel_max_width_change_ratio) < 0.0:
        raise ValueError("channel_max_width_change_ratio must be non-negative")
    if float(cfg.channel_parallel_tolerance_atr_per_bar) <= 0.0:
        raise ValueError("channel_parallel_tolerance_atr_per_bar must be positive")
    if float(cfg.channel_flat_slope_atr_per_bar) <= 0.0:
        raise ValueError("channel_flat_slope_atr_per_bar must be positive")
    if float(cfg.channel_min_slope_atr_per_bar) <= 0.0:
        raise ValueError("channel_min_slope_atr_per_bar must be positive")
    if not 0.0 <= float(cfg.channel_min_containment) <= 1.0:
        raise ValueError("channel_min_containment must be between 0 and 1")
    if not 0.0 <= float(cfg.channel_min_quality) <= 1.0:
        raise ValueError("channel_min_quality must be between 0 and 1")
    if int(cfg.channel_min_side_pivots) < 2:
        raise ValueError("channel_min_side_pivots must be at least 2")
    if not 0.0 <= float(cfg.channel_min_rail_span_ratio) <= 1.0:
        raise ValueError("channel_min_rail_span_ratio must be between 0 and 1")
    if not 1 <= int(cfg.channel_min_side_zones) <= 3:
        raise ValueError("channel_min_side_zones must be between 1 and 3")
    if int(cfg.channel_envelope_start_options) < 1:
        raise ValueError("channel_envelope_start_options must be at least 1")
    if not 0.0 <= float(cfg.channel_merge_min_span_overlap) <= 1.0:
        raise ValueError("channel_merge_min_span_overlap must be between 0 and 1")
    if float(cfg.channel_merge_max_boundary_distance_atr) < 0.0:
        raise ValueError("channel_merge_max_boundary_distance_atr must be non-negative")
    if float(cfg.channel_merge_max_width_diff_atr) < 0.0:
        raise ValueError("channel_merge_max_width_diff_atr must be non-negative")
    if float(cfg.channel_merge_max_slope_diff_atr_per_bar) < 0.0:
        raise ValueError("channel_merge_max_slope_diff_atr_per_bar must be non-negative")
    if int(cfg.channel_max_active_outputs) < 1:
        raise ValueError("channel_max_active_outputs must be at least 1")
    if float(cfg.channel_lifecycle_break_atr_mult) < 0.0:
        raise ValueError("channel_lifecycle_break_atr_mult must be non-negative")
    if int(cfg.channel_lifecycle_confirm_break_bars) < 1:
        raise ValueError("channel_lifecycle_confirm_break_bars must be at least 1")
    if float(cfg.channel_near_boundary_atr_mult) < 0.0:
        raise ValueError("channel_near_boundary_atr_mult must be non-negative")
    if float(cfg.channel_breakout_atr_mult) < 0.0:
        raise ValueError("channel_breakout_atr_mult must be non-negative")
    if int(cfg.min_output_bars) < 1:
        raise ValueError("min_output_bars must be at least 1")
    if float(cfg.containment_tolerance_atr_mult) < 0.0:
        raise ValueError("containment_tolerance_atr_mult must be non-negative")
    if float(cfg.touch_tolerance_atr_mult) <= 0.0:
        raise ValueError("touch_tolerance_atr_mult must be positive")
    if int(cfg.max_recent_touch_age_bars) < 1:
        raise ValueError("max_recent_touch_age_bars must be at least 1")
    if float(cfg.min_slope_atr_per_bar) <= 0.0:
        raise ValueError("min_slope_atr_per_bar must be positive")
    if float(cfg.flat_slope_atr_per_bar) <= 0.0:
        raise ValueError("flat_slope_atr_per_bar must be positive")
    if int(cfg.min_total_pivots) < 4:
        raise ValueError("min_total_pivots must be at least 4")
    if int(cfg.impulse_min_span_bars) < 1:
        raise ValueError("impulse_min_span_bars must be at least 1")
    if float(cfg.impulse_max_slope_atr_per_bar) <= 0.0:
        raise ValueError("impulse_max_slope_atr_per_bar must be positive")


__all__ = [
    "PatternGeometryV2Config",
    "add_pattern_geometry_v2",
]

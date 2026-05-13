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


_FAMILY_CODE = {"rectangle": 1, "ascending_channel": 2, "descending_channel": 3}
_SLOT_FIELDS = (
    "active",
    "near_upper",
    "near_lower",
    "breakout_up",
    "breakdown_down",
    "family",
    "direction",
    "upper",
    "lower",
    "start_index",
    "end_index",
    "position",
    "width_atr",
    "line_score",
    "containment",
    "parallel_error",
    "upper_pivots",
    "lower_pivots",
)
_SLOT_BOOL_FIELDS = {"active", "near_upper", "near_lower", "breakout_up", "breakdown_down"}
_ROW_FIELDS = (
    "best_position",
    "best_width_atr",
    "near_upper",
    "near_lower",
    "avoid_long",
    "avoid_short",
    "breakout_up",
    "breakdown_down",
)


@dataclass(frozen=True)
class PatternChannelConfig:
    """Lean TLV2-backed rectangle/channel context detector.

    The useful strategy surface is context, not a direct trade command:
    price location within the channel, avoid-long/avoid-short boundary flags,
    and breakout/breakdown flags. Validation focus is 1h and 4h.
    """

    output_prefix: str = "pch"
    output_slots: int = 3
    timeframe: str = "4h"
    pivot_strength: int = 2
    min_channel_bars: int = 12
    max_channel_bars: int = 96
    min_width_atr: float = 1.0
    max_width_atr: float = 8.0
    parallel_tolerance_atr_per_bar: float = 0.020
    flat_slope_atr_per_bar: float = 0.012
    min_channel_slope_atr_per_bar: float = 0.018
    max_width_change_ratio: float = 0.30
    min_line_score: float = 0.50
    min_total_pivots: int = 4
    min_containment: float = 0.90
    containment_tolerance_atr_mult: float = 0.25
    near_boundary_atr_mult: float = 0.70
    breakout_atr_mult: float = 0.35
    min_output_bars: int = 2


def add_pattern_channel(
    dataframe: DataFrame,
    config: PatternChannelConfig | None = None,
    **overrides: object,
) -> DataFrame:
    cfg = _resolve_config(config, overrides)
    _validate_config(cfg)
    _validate_dataframe(dataframe)

    arrays = _channel_arrays(dataframe, cfg)
    p = cfg.output_prefix
    columns: dict[str, Series] = {}
    for slot in range(1, int(cfg.output_slots) + 1):
        for field in _SLOT_FIELDS:
            key = f"{p}_slot_{slot}_{field}"
            value = arrays[f"slot_{slot}_{field}"]
            if field in _SLOT_BOOL_FIELDS:
                columns[key] = pd.Series(value, index=dataframe.index, dtype="bool").fillna(False)
            elif field in {"family", "direction"}:
                columns[key] = pd.Series(value, index=dataframe.index, dtype="int8")
            else:
                columns[key] = pd.Series(value, index=dataframe.index, dtype="float64")
    for field in _ROW_FIELDS:
        key = f"{p}_{field}"
        value = arrays[field]
        if field in {"near_upper", "near_lower", "avoid_long", "avoid_short", "breakout_up", "breakdown_down"}:
            columns[key] = pd.Series(value, index=dataframe.index, dtype="bool").fillna(False)
        else:
            columns[key] = pd.Series(value, index=dataframe.index, dtype="float64")

    source = dataframe.copy()
    existing = [col for col in source.columns if str(col).startswith(f"{p}_")]
    clean = source.drop(columns=existing).copy() if existing else source.copy()
    return pd.concat([clean, pd.DataFrame(columns, index=dataframe.index)], axis=1)


def _channel_arrays(frame: DataFrame, cfg: PatternChannelConfig) -> dict[str, np.ndarray]:
    rows = len(frame)
    out = _empty_arrays(rows, int(cfg.output_slots))
    tl_cfg = _resolve_trendline_v2_config(
        None,
        {
            "timeframe": str(cfg.timeframe),
            "pivot_strength": int(cfg.pivot_strength),
        },
    )
    base = _base_inputs(frame.copy(), tl_cfg)
    candidates = _build_sequence_candidate_table(base, tl_cfg)
    if candidates.empty:
        return out

    close = base["close"].replace(0.0, np.nan).to_numpy(dtype="float64")
    body_high = base["body_high"].to_numpy(dtype="float64")
    body_low = base["body_low"].to_numpy(dtype="float64")
    atr = base["atr"].clip(lower=1e-9).to_numpy(dtype="float64")
    resistance = candidates[candidates["side"].eq("resistance")].copy()
    support = candidates[candidates["side"].eq("support")].copy()
    if resistance.empty or support.empty:
        return out

    resistance["active_start"] = pd.to_numeric(
        resistance.get("live_start", resistance["x_new"]),
        errors="coerce",
    ).fillna(pd.to_numeric(resistance["x_new"], errors="coerce"))
    support["active_start"] = pd.to_numeric(
        support.get("live_start", support["x_new"]),
        errors="coerce",
    ).fillna(pd.to_numeric(support["x_new"], errors="coerce"))

    for row in range(rows):
        if not np.isfinite(close[row]):
            continue
        active_resistance = resistance[
            resistance["active_start"].le(float(row)) & resistance["projection_end"].ge(float(row))
        ]
        active_support = support[
            support["active_start"].le(float(row)) & support["projection_end"].ge(float(row))
        ]
        if active_resistance.empty or active_support.empty:
            continue

        channel_candidates: list[dict[str, float]] = []
        for upper in active_resistance.itertuples(index=False):
            for lower in active_support.itertuples(index=False):
                candidate = _pair_lines_as_channel(
                    row=row,
                    close=close,
                    body_high=body_high,
                    body_low=body_low,
                    atr=atr,
                    upper=upper,
                    lower_line=lower,
                    cfg=cfg,
                )
                if candidate is not None:
                    channel_candidates.append(candidate)
        if not channel_candidates:
            continue

        selected = _select_distinct_channels(channel_candidates, close, atr, row, cfg)
        for slot, candidate in enumerate(selected[: int(cfg.output_slots)], start=1):
            out[f"slot_{slot}_active"][row] = True
            out[f"slot_{slot}_near_upper"][row] = bool(candidate["near_upper"])
            out[f"slot_{slot}_near_lower"][row] = bool(candidate["near_lower"])
            out[f"slot_{slot}_breakout_up"][row] = bool(candidate["breakout_up"])
            out[f"slot_{slot}_breakdown_down"][row] = bool(candidate["breakdown_down"])
            out[f"slot_{slot}_family"][row] = int(candidate["family_code"])
            out[f"slot_{slot}_direction"][row] = int(candidate["direction"])
            out[f"slot_{slot}_upper"][row] = float(candidate["upper"])
            out[f"slot_{slot}_lower"][row] = float(candidate["lower"])
            out[f"slot_{slot}_start_index"][row] = float(candidate["start_index"])
            out[f"slot_{slot}_end_index"][row] = float(candidate["end_index"])
            out[f"slot_{slot}_position"][row] = float(candidate["position"])
            out[f"slot_{slot}_width_atr"][row] = float(candidate["width_atr"])
            out[f"slot_{slot}_line_score"][row] = float(candidate["line_score"])
            out[f"slot_{slot}_containment"][row] = float(candidate["containment"])
            out[f"slot_{slot}_parallel_error"][row] = float(candidate["parallel_error"])
            out[f"slot_{slot}_upper_pivots"][row] = float(candidate["upper_pivots"])
            out[f"slot_{slot}_lower_pivots"][row] = float(candidate["lower_pivots"])
    _suppress_short_segments(out, int(cfg.output_slots), int(cfg.min_output_bars))
    _update_row_fields(out, int(cfg.output_slots))
    return out


def _pair_lines_as_channel(
    *,
    row: int,
    close: np.ndarray,
    body_high: np.ndarray,
    body_low: np.ndarray,
    atr: np.ndarray,
    upper: object,
    lower_line: object,
    cfg: PatternChannelConfig,
) -> dict[str, float] | None:
    start_index = int(round(max(float(upper.x_old), float(lower_line.x_old))))
    end_index = int(row)
    span = end_index - start_index
    if span < int(cfg.min_channel_bars) or span > int(cfg.max_channel_bars):
        return None

    upper_slope = float(upper.slope)
    lower_slope = float(lower_line.slope)
    upper_intercept = float(upper.intercept)
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

    atr_scale = max(float(np.nanmedian(atr[start_index : end_index + 1])), float(atr[row]), 1e-9)
    upper_slope_atr = upper_slope / atr_scale
    lower_slope_atr = lower_slope / atr_scale
    parallel_error = abs(upper_slope_atr - lower_slope_atr)
    if parallel_error > float(cfg.parallel_tolerance_atr_per_bar):
        return None

    start_width = upper_start - lower_start
    current_width = upper_now - lower_now
    width_atr = current_width / max(float(atr[row]), 1e-9)
    if width_atr < float(cfg.min_width_atr) or width_atr > float(cfg.max_width_atr):
        return None
    width_change = abs(current_width - start_width) / max(start_width, current_width, 1e-9)
    if width_change > float(cfg.max_width_change_ratio):
        return None

    line_score = min(float(upper.score), float(lower_line.score))
    if line_score < float(cfg.min_line_score):
        return None
    upper_pivots = float(max(getattr(upper, "absorbed_pivot_count", getattr(upper, "pivot_count", 2.0)), 2.0))
    lower_pivots = float(max(getattr(lower_line, "absorbed_pivot_count", getattr(lower_line, "pivot_count", 2.0)), 2.0))
    if upper_pivots + lower_pivots < float(cfg.min_total_pivots):
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
    if containment < float(cfg.min_containment):
        return None

    family = _channel_family(
        upper_slope_atr,
        lower_slope_atr,
        flat_slope=float(cfg.flat_slope_atr_per_bar),
        min_slope=float(cfg.min_channel_slope_atr_per_bar),
    )
    if family is None:
        return None

    position = (float(close[row]) - lower_now) / max(current_width, 1e-9)
    near_distance = float(atr[row]) * float(cfg.near_boundary_atr_mult)
    breakout_distance = float(atr[row]) * float(cfg.breakout_atr_mult)
    near_upper = bool(upper_now - float(close[row]) <= near_distance and float(close[row]) <= upper_now + breakout_distance)
    near_lower = bool(float(close[row]) - lower_now <= near_distance and float(close[row]) >= lower_now - breakout_distance)
    breakout_up = bool(float(close[row]) > upper_now + breakout_distance)
    breakdown_down = bool(float(close[row]) < lower_now - breakout_distance)
    return {
        "family_code": float(_FAMILY_CODE[family]),
        "direction": float(_direction_code(family)),
        "upper": float(upper_now),
        "lower": float(lower_now),
        "start_index": float(start_index),
        "end_index": float(end_index),
        "position": float(position),
        "width_atr": float(width_atr),
        "line_score": float(line_score),
        "containment": float(containment),
        "parallel_error": float(parallel_error),
        "upper_pivots": float(upper_pivots),
        "lower_pivots": float(lower_pivots),
        "near_upper": float(near_upper),
        "near_lower": float(near_lower),
        "breakout_up": float(breakout_up),
        "breakdown_down": float(breakdown_down),
    }


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


def _channel_family(upper_slope_atr: float, lower_slope_atr: float, *, flat_slope: float, min_slope: float) -> str | None:
    mid = 0.5 * (float(upper_slope_atr) + float(lower_slope_atr))
    if abs(mid) <= float(flat_slope):
        return "rectangle"
    if mid >= float(min_slope):
        return "ascending_channel"
    if mid <= -float(min_slope):
        return "descending_channel"
    return None


def _direction_code(family: str) -> int:
    if family == "ascending_channel":
        return 1
    if family == "descending_channel":
        return -1
    return 0


def _select_distinct_channels(
    candidates: list[dict[str, float]],
    close: np.ndarray,
    atr: np.ndarray,
    row: int,
    cfg: PatternChannelConfig,
) -> list[dict[str, float]]:
    selected: list[dict[str, float]] = []
    for candidate in sorted(candidates, key=_rank_key, reverse=True):
        if any(_same_channel(candidate, existing, close, atr, row, cfg) for existing in selected):
            continue
        selected.append(candidate)
    return selected


def _rank_key(candidate: dict[str, float]) -> tuple[float, float, float, float, float]:
    return (
        min(float(candidate["upper_pivots"]), float(candidate["lower_pivots"])),
        float(candidate["line_score"]),
        float(candidate["containment"]),
        -float(candidate["parallel_error"]),
        -abs(float(candidate["position"]) - 0.5),
    )


def _same_channel(
    candidate: dict[str, float],
    existing: dict[str, float],
    close: np.ndarray,
    atr: np.ndarray,
    row: int,
    cfg: PatternChannelConfig,
) -> bool:
    span = max(abs(float(candidate["end_index"]) - float(candidate["start_index"])), 1.0)
    start_distance = abs(float(candidate["start_index"]) - float(existing["start_index"])) / span
    if start_distance > 0.20:
        return False
    tolerance = max(float(atr[row]) * float(cfg.near_boundary_atr_mult), abs(float(close[row])) * 0.0, 1e-9)
    upper_close = abs(float(candidate["upper"]) - float(existing["upper"])) <= tolerance
    lower_close = abs(float(candidate["lower"]) - float(existing["lower"])) <= tolerance
    return upper_close and lower_close


def _empty_arrays(rows: int, slot_count: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for slot in range(1, slot_count + 1):
        for field in _SLOT_FIELDS:
            key = f"slot_{slot}_{field}"
            if field in _SLOT_BOOL_FIELDS:
                out[key] = np.zeros(rows, dtype=bool)
            elif field in {"family", "direction"}:
                out[key] = np.zeros(rows, dtype="int8")
            else:
                out[key] = np.full(rows, np.nan, dtype="float64")
    out["best_position"] = np.full(rows, np.nan, dtype="float64")
    out["best_width_atr"] = np.full(rows, np.nan, dtype="float64")
    for field in ("near_upper", "near_lower", "avoid_long", "avoid_short", "breakout_up", "breakdown_down"):
        out[field] = np.zeros(rows, dtype=bool)
    return out


def _suppress_short_segments(out: dict[str, np.ndarray], slot_count: int, min_output_bars: int) -> None:
    if min_output_bars <= 1:
        return
    rows = len(out["slot_1_active"]) if slot_count else 0
    for slot in range(1, slot_count + 1):
        active = out[f"slot_{slot}_active"]
        family = out[f"slot_{slot}_family"]
        start_index = out[f"slot_{slot}_start_index"]
        segment_start: int | None = None
        for row in range(rows + 1):
            is_active = row < rows and bool(active[row])
            same_segment = (
                is_active
                and segment_start is not None
                and row > 0
                and int(family[row]) == int(family[row - 1])
                and abs(float(start_index[row]) - float(start_index[row - 1])) < 0.5
            )
            if is_active and segment_start is None:
                segment_start = row
                continue
            if same_segment:
                continue
            if segment_start is not None:
                segment_end = row - 1
                if segment_end - segment_start + 1 < int(min_output_bars):
                    _clear_slot_range(out, slot, segment_start, segment_end)
            segment_start = row if is_active else None


def _clear_slot_range(out: dict[str, np.ndarray], slot: int, start: int, end: int) -> None:
    if end < start:
        return
    for field in _SLOT_FIELDS:
        key = f"slot_{slot}_{field}"
        if field in _SLOT_BOOL_FIELDS:
            out[key][start : end + 1] = False
        elif field in {"family", "direction"}:
            out[key][start : end + 1] = 0
        else:
            out[key][start : end + 1] = np.nan


def _update_row_fields(out: dict[str, np.ndarray], slot_count: int) -> None:
    rows = len(out["slot_1_active"]) if slot_count else 0
    for row in range(rows):
        best: dict[str, float] | None = None
        for slot in range(1, slot_count + 1):
            if not bool(out[f"slot_{slot}_active"][row]):
                continue
            candidate = {
                "position": float(out[f"slot_{slot}_position"][row]),
                "width_atr": float(out[f"slot_{slot}_width_atr"][row]),
                "line_score": float(out[f"slot_{slot}_line_score"][row]),
                "containment": float(out[f"slot_{slot}_containment"][row]),
                "near_upper": float(out[f"slot_{slot}_near_upper"][row]),
                "near_lower": float(out[f"slot_{slot}_near_lower"][row]),
            }
            if best is None or _row_rank_key(candidate) > _row_rank_key(best):
                best = candidate
        if best is None:
            continue
        out["best_position"][row] = float(best["position"])
        out["best_width_atr"][row] = float(best["width_atr"])
        out["near_upper"][row] = bool(best["near_upper"])
        out["near_lower"][row] = bool(best["near_lower"])
        out["breakout_up"][row] = _any_slot_flag(out, "breakout_up", row, slot_count)
        out["breakdown_down"][row] = _any_slot_flag(out, "breakdown_down", row, slot_count)
        out["avoid_long"][row] = bool(best["near_upper"]) or bool(out["breakdown_down"][row])
        out["avoid_short"][row] = bool(best["near_lower"]) or bool(out["breakout_up"][row])


def _row_rank_key(candidate: dict[str, float]) -> tuple[float, float, float]:
    return (
        float(candidate["line_score"]),
        float(candidate["containment"]),
        -abs(float(candidate["position"]) - 0.5),
    )


def _any_slot_flag(out: dict[str, np.ndarray], name: str, row: int, slot_count: int) -> bool:
    return any(bool(out[f"slot_{slot}_active"][row]) and bool(out[f"slot_{slot}_{name}"][row]) for slot in range(1, slot_count + 1))


def _resolve_config(config: PatternChannelConfig | None, overrides: dict[str, object]) -> PatternChannelConfig:
    cfg = config or PatternChannelConfig()
    valid = {field.name for field in fields(PatternChannelConfig)}
    clean = {key: value for key, value in overrides.items() if value is not None}
    unknown = sorted(set(clean).difference(valid))
    if unknown:
        raise TypeError(f"Unknown channel config override(s): {', '.join(unknown)}")
    return replace(cfg, **clean) if clean else cfg


def _validate_dataframe(frame: DataFrame) -> None:
    missing = {"open", "high", "low", "close"}.difference(frame.columns)
    if missing:
        raise ValueError(f"dataframe missing required columns: {', '.join(sorted(missing))}")


def _validate_config(cfg: PatternChannelConfig) -> None:
    if int(cfg.output_slots) < 1:
        raise ValueError("output_slots must be positive")
    if int(cfg.pivot_strength) < 1:
        raise ValueError("pivot_strength must be positive")
    if int(cfg.min_channel_bars) < 2:
        raise ValueError("min_channel_bars must be at least 2")
    if int(cfg.max_channel_bars) < int(cfg.min_channel_bars):
        raise ValueError("max_channel_bars must be >= min_channel_bars")
    for name in (
        "min_width_atr",
        "max_width_atr",
        "parallel_tolerance_atr_per_bar",
        "flat_slope_atr_per_bar",
        "min_channel_slope_atr_per_bar",
        "min_line_score",
        "min_containment",
        "containment_tolerance_atr_mult",
        "near_boundary_atr_mult",
        "breakout_atr_mult",
    ):
        if float(getattr(cfg, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if float(cfg.max_width_atr) <= float(cfg.min_width_atr):
        raise ValueError("max_width_atr must be greater than min_width_atr")
    if float(cfg.max_width_change_ratio) < 0.0:
        raise ValueError("max_width_change_ratio must be non-negative")
    if int(cfg.min_total_pivots) < 4:
        raise ValueError("min_total_pivots must be at least 4")
    if int(cfg.min_output_bars) < 1:
        raise ValueError("min_output_bars must be positive")


__all__ = [
    "PatternChannelConfig",
    "add_pattern_channel",
]

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

try:
    from .complex_trendline_projection_v2 import (
        TrendlineProjectionV2Config,
        _base_inputs,
        _build_sequence_candidate_table,
    )
except Exception:  # pragma: no cover - standalone review scripts import this module directly
    from complex_trendline_projection_v2 import (  # type: ignore[no-redef]
        TrendlineProjectionV2Config,
        _base_inputs,
        _build_sequence_candidate_table,
    )


_FAMILY_CODE = {"triangle": 1, "wedge": 2, "compression": 3}
_SLOT_FIELDS = (
    "active",
    "family",
    "direction",
    "upper",
    "lower",
    "start_index",
    "end_index",
    "line_score",
    "contraction",
    "containment",
    "width_pct",
    "upper_pivots",
    "lower_pivots",
)


@dataclass(frozen=True)
class PatternGeometryV2Config:
    """Lean line-first triangle/wedge/compression detector.

    The detector uses validated support/resistance lines from TLV2, then pairs
    only currently-active lines into patterns. It does not try to invent
    patterns directly from arbitrary pivot envelopes.
    """

    output_prefix: str = "pg2"
    output_slots: int = 4
    pivot_strength: int = 2
    min_pattern_bars: int = 12
    max_pattern_bars: int = 72
    min_width_pct: float = 0.006
    max_width_pct: float = 0.12
    min_contraction: float = 0.08
    min_containment: float = 0.88
    min_line_score: float = 0.50
    boundary_tolerance_pct: float = 0.0035
    touch_tolerance_atr_mult: float = 0.55
    max_recent_touch_age_bars: int = 12
    min_slope_pct_per_bar: float = 0.00018
    flat_slope_pct_per_bar: float = 0.00030
    min_total_pivots: int = 4


def add_pattern_geometry_v2(
    dataframe: DataFrame,
    config: PatternGeometryV2Config | None = None,
    **overrides: object,
) -> DataFrame:
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

    source = dataframe.copy()
    existing = [col for col in source.columns if str(col).startswith(f"{p}_")]
    clean = source.drop(columns=existing).copy() if existing else source.copy()
    return pd.concat([clean, pd.DataFrame(columns, index=dataframe.index)], axis=1)


def _geometry_v2_arrays(frame: DataFrame, cfg: PatternGeometryV2Config) -> dict[str, np.ndarray]:
    rows = len(frame)
    out = _empty_slot_arrays(rows, int(cfg.output_slots))
    tl_cfg = TrendlineProjectionV2Config(pivot_strength=int(cfg.pivot_strength))
    base = _base_inputs(frame.copy(), tl_cfg)
    candidates = _build_sequence_candidate_table(base, tl_cfg)
    if candidates.empty:
        return out

    body_high = base["body_high"].to_numpy(dtype="float64")
    body_low = base["body_low"].to_numpy(dtype="float64")
    high = base["high"].to_numpy(dtype="float64")
    low = base["low"].to_numpy(dtype="float64")
    close = base["close"].replace(0.0, np.nan).to_numpy(dtype="float64")
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

        pattern_candidates: list[dict[str, float]] = []
        for upper in active_resistance.itertuples(index=False):
            for lower_line in active_support.itertuples(index=False):
                candidate = _pair_lines_as_pattern(
                    row=row,
                    close=close,
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

        if not pattern_candidates:
            continue

        selected = _select_distinct_patterns(pattern_candidates, close, atr, row, cfg)
        for slot, candidate in enumerate(selected[: int(cfg.output_slots)], start=1):
            out[f"slot_{slot}_active"][row] = True
            out[f"slot_{slot}_family"][row] = int(candidate["family_code"])
            out[f"slot_{slot}_direction"][row] = int(candidate["direction"])
            out[f"slot_{slot}_upper"][row] = float(candidate["upper"])
            out[f"slot_{slot}_lower"][row] = float(candidate["lower"])
            out[f"slot_{slot}_start_index"][row] = float(candidate["start_index"])
            out[f"slot_{slot}_end_index"][row] = float(candidate["end_index"])
            out[f"slot_{slot}_line_score"][row] = float(candidate["line_score"])
            out[f"slot_{slot}_contraction"][row] = float(candidate["contraction"])
            out[f"slot_{slot}_containment"][row] = float(candidate["containment"])
            out[f"slot_{slot}_width_pct"][row] = float(candidate["width_pct"])
            out[f"slot_{slot}_upper_pivots"][row] = float(candidate["upper_pivots"])
            out[f"slot_{slot}_lower_pivots"][row] = float(candidate["lower_pivots"])
    return out


def _pair_lines_as_pattern(
    *,
    row: int,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    body_high: np.ndarray,
    body_low: np.ndarray,
    upper: object,
    lower_line: object,
    cfg: PatternGeometryV2Config,
) -> dict[str, float] | None:
    start_index = int(round(max(float(upper.x_old), float(lower_line.x_old))))
    end_index = int(row)
    span = end_index - start_index
    if span < int(cfg.min_pattern_bars) or span > int(cfg.max_pattern_bars):
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

    reference = max(abs(float(close[row])), 1e-9)
    start_width = upper_start - lower_start
    current_width = upper_now - lower_now
    width_pct = current_width / reference
    contraction = 1.0 - current_width / max(start_width, 1e-9)
    if width_pct < float(cfg.min_width_pct) or width_pct > float(cfg.max_width_pct):
        return None
    if contraction < float(cfg.min_contraction):
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
        close=close,
        start_index=start_index,
        end_index=end_index,
        upper_slope=upper_slope,
        upper_intercept=upper_intercept,
        lower_slope=lower_slope,
        lower_intercept=lower_intercept,
        tolerance_pct=float(cfg.boundary_tolerance_pct),
    )
    if containment < float(cfg.min_containment):
        return None

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

    upper_slope_pct = upper_slope / reference
    lower_slope_pct = lower_slope / reference
    family = _pattern_family(
        upper_slope_pct=upper_slope_pct,
        lower_slope_pct=lower_slope_pct,
        min_slope=float(cfg.min_slope_pct_per_bar),
        flat_slope=float(cfg.flat_slope_pct_per_bar),
    )
    if family is None:
        return None

    direction = _direction_code(upper_slope_pct, lower_slope_pct, float(cfg.min_slope_pct_per_bar))
    return {
        "family_code": float(_FAMILY_CODE[family]),
        "direction": float(direction),
        "upper": float(upper_now),
        "lower": float(lower_now),
        "start_index": float(start_index),
        "end_index": float(end_index),
        "upper_slope": float(upper_slope),
        "upper_intercept": float(upper_intercept),
        "lower_slope": float(lower_slope),
        "lower_intercept": float(lower_intercept),
        "line_score": float(line_score),
        "contraction": float(contraction),
        "containment": float(containment),
        "width_pct": float(width_pct),
        "upper_pivots": float(upper_pivots),
        "lower_pivots": float(lower_pivots),
        "upper_touch_age": float(upper_touch_age),
        "lower_touch_age": float(lower_touch_age),
        "span": float(span),
    }


def _containment_ratio(
    *,
    body_high: np.ndarray,
    body_low: np.ndarray,
    close: np.ndarray,
    start_index: int,
    end_index: int,
    upper_slope: float,
    upper_intercept: float,
    lower_slope: float,
    lower_intercept: float,
    tolerance_pct: float,
) -> float:
    if end_index <= start_index:
        return 0.0
    x = np.arange(start_index, end_index + 1, dtype="float64")
    reference = np.maximum(np.abs(close[start_index : end_index + 1]), 1e-9)
    tolerance = reference * float(tolerance_pct)
    upper_line = upper_slope * x + upper_intercept
    lower_line = lower_slope * x + lower_intercept
    upper_intrusion = np.maximum(body_high[start_index : end_index + 1] - upper_line, 0.0)
    lower_intrusion = np.maximum(lower_line - body_low[start_index : end_index + 1], 0.0)
    respected = (upper_intrusion <= tolerance) & (lower_intrusion <= tolerance)
    return float(np.nanmean(respected.astype("float64")))


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


def _pattern_family(
    *,
    upper_slope_pct: float,
    lower_slope_pct: float,
    min_slope: float,
    flat_slope: float,
) -> str | None:
    if not np.isfinite([upper_slope_pct, lower_slope_pct]).all():
        return None
    slope_gap = float(lower_slope_pct) - float(upper_slope_pct)
    if slope_gap < float(min_slope):
        return None

    upper_flat = abs(float(upper_slope_pct)) <= float(flat_slope)
    lower_flat = abs(float(lower_slope_pct)) <= float(flat_slope)
    upper_down = float(upper_slope_pct) <= -float(min_slope)
    upper_up = float(upper_slope_pct) >= float(min_slope)
    lower_down = float(lower_slope_pct) <= -float(min_slope)
    lower_up = float(lower_slope_pct) >= float(min_slope)

    if (upper_flat and lower_up) or (upper_down and lower_flat) or (upper_down and lower_up):
        return "triangle"
    if (upper_up and lower_up) or (upper_down and lower_down):
        return "wedge"
    return "compression"


def _direction_code(upper_slope_pct: float, lower_slope_pct: float, min_slope: float) -> int:
    mid = 0.5 * (float(upper_slope_pct) + float(lower_slope_pct))
    if mid >= float(min_slope):
        return 1
    if mid <= -float(min_slope):
        return -1
    return 0


def _select_distinct_patterns(
    candidates: list[dict[str, float]],
    close: np.ndarray,
    atr: np.ndarray,
    row: int,
    cfg: PatternGeometryV2Config,
) -> list[dict[str, float]]:
    selected: list[dict[str, float]] = []
    for candidate in sorted(candidates, key=_pattern_rank_key, reverse=True):
        if any(_same_pattern(candidate, existing, close, atr, row, cfg) for existing in selected):
            continue
        selected.append(candidate)
    return selected


def _pattern_rank_key(candidate: dict[str, float]) -> tuple[float, float, float, float, float, float]:
    return (
        min(float(candidate["upper_pivots"]), float(candidate["lower_pivots"])),
        float(candidate["line_score"]),
        float(candidate["containment"]),
        float(candidate["contraction"]),
        -max(float(candidate["upper_touch_age"]), float(candidate["lower_touch_age"])),
        -float(candidate["width_pct"]),
    )


def _same_pattern(
    candidate: dict[str, float],
    existing: dict[str, float],
    close: np.ndarray,
    atr: np.ndarray,
    row: int,
    cfg: PatternGeometryV2Config,
) -> bool:
    scale = max(float(atr[row]), 1e-9)
    upper_distance = abs(float(candidate["upper"]) - float(existing["upper"])) / scale
    lower_distance = abs(float(candidate["lower"]) - float(existing["lower"])) / scale
    span = max(abs(float(candidate["end_index"]) - float(candidate["start_index"])), 1.0)
    start_distance = abs(float(candidate["start_index"]) - float(existing["start_index"])) / span
    family_match = int(candidate["family_code"]) == int(existing["family_code"])
    return family_match and upper_distance <= 1.25 and lower_distance <= 1.25 and start_distance <= 0.25


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


def _resolve_config(config: PatternGeometryV2Config | None, overrides: dict[str, object]) -> PatternGeometryV2Config:
    cfg = config or PatternGeometryV2Config()
    if not overrides:
        return cfg
    valid = {field.name for field in fields(PatternGeometryV2Config)}
    unknown = sorted(set(overrides).difference(valid))
    if unknown:
        raise TypeError(f"Unknown geometry v2 config override(s): {', '.join(unknown)}")
    return replace(cfg, **{name: overrides[name] for name in overrides if name in valid})


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
    if int(cfg.pivot_strength) < 1:
        raise ValueError("pivot_strength must be at least 1")
    if int(cfg.min_pattern_bars) < 4:
        raise ValueError("min_pattern_bars must be at least 4")
    if int(cfg.max_pattern_bars) <= int(cfg.min_pattern_bars):
        raise ValueError("max_pattern_bars must be greater than min_pattern_bars")
    if not 0.0 < float(cfg.min_width_pct) < float(cfg.max_width_pct):
        raise ValueError("min_width_pct must be lower than max_width_pct")
    if not 0.0 <= float(cfg.min_contraction) <= 1.0:
        raise ValueError("min_contraction must be between 0 and 1")
    if not 0.0 <= float(cfg.min_containment) <= 1.0:
        raise ValueError("min_containment must be between 0 and 1")
    if not 0.0 <= float(cfg.min_line_score) <= 1.0:
        raise ValueError("min_line_score must be between 0 and 1")
    if float(cfg.boundary_tolerance_pct) < 0.0:
        raise ValueError("boundary_tolerance_pct must be non-negative")
    if float(cfg.touch_tolerance_atr_mult) <= 0.0:
        raise ValueError("touch_tolerance_atr_mult must be positive")
    if int(cfg.max_recent_touch_age_bars) < 1:
        raise ValueError("max_recent_touch_age_bars must be at least 1")
    if float(cfg.min_slope_pct_per_bar) <= 0.0:
        raise ValueError("min_slope_pct_per_bar must be positive")
    if float(cfg.flat_slope_pct_per_bar) <= 0.0:
        raise ValueError("flat_slope_pct_per_bar must be positive")
    if int(cfg.min_total_pivots) < 4:
        raise ValueError("min_total_pivots must be at least 4")


__all__ = [
    "PatternGeometryV2Config",
    "add_pattern_geometry_v2",
]

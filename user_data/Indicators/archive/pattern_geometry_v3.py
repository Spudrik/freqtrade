from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

# PARKED: geometry v3 is intentionally not the active geometry path.
# TODO: Revisit only after v2 review stabilizes. Open items:
# - keep the A+B line cleanup candidate: span_slope + co-terminal outer suppression
# - decide whether internal amplitude is useful after better line reduction
# - define the next merge/reduction pass before adding pattern pairing
# - keep v3 output as lines-only until line quality is good enough

try:
    from .pivot_foundation import build_clean_pivot_source
except Exception:  # pragma: no cover - standalone review scripts import this module directly
    from pivot_foundation import build_clean_pivot_source  # type: ignore[no-redef]


@dataclass(frozen=True)
class PatternGeometryV3Config:
    output_prefix: str = "g3"
    output_slots: int = 3
    pivot_strength: int = 2
    min_pivot_prominence_atr: float = 0.35
    min_seed_span_bars: int = 10
    max_seed_span_bars: int = 100
    max_slope_atr_per_bar: float = 0.40
    volatility_scale_mode: str = "atr"
    volatility_body_mult: float = 1.8
    impulse_filter_mode: str = "off"
    impulse_min_span_bars: int = 16
    impulse_max_slope_atr_per_bar: float = 0.12
    impulse_max_travel_atr: float = 8.0
    impulse_max_score: float = 1.5
    internal_amplitude_mode: str = "off"
    internal_amplitude_max_atr: float = 8.0
    internal_amplitude_base_atr: float = 4.0
    internal_amplitude_span_growth_atr: float = 1.5
    internal_amplitude_reference_bars: int = 24
    internal_amplitude_max_dynamic_atr: float = 10.0
    shock_cooldown_bars: int = 0
    shock_tr_atr_threshold: float = 2.0
    shock_body_atr_threshold: float = 1.5
    anchor_break_max_run: int = 2
    anchor_break_max_depth_atr: float = 0.40
    anchor_break_edge_bars: int = 3
    max_joined_pivots: int = 6
    step4_reduction_sequence: str = "A"
    join_angle_tolerance: float = 0.15
    shared_subset_min_pivots: int = 2
    partial_merge_shared_pivots_min: int = 2
    partial_merge_angle_tolerance: float = 0.18
    partial_merge_use_atr_check: bool = False
    partial_merge_max_unmatched_pivot_distance_atr: float = 0.40
    co_terminal_outer_suppression: bool = False
    co_terminal_start_tolerance_bars: int = 6
    co_terminal_end_tolerance_bars: int = 8
    co_terminal_min_overlap_ratio: float = 0.65
    co_terminal_max_mean_gap_atr: float = 1.5
    final_selection_recheck_use_atr: bool = False
    final_selection_max_distance_atr: float = 0.40
    max_active_price_distance_atr: float = 2.0
    active_break_close_run: int = 2


def add_pattern_geometry_v3(
    dataframe: DataFrame,
    config: PatternGeometryV3Config | None = None,
    **overrides: object,
) -> DataFrame:
    cfg = _resolve_config(config, overrides)
    _validate_config(cfg)
    _validate_dataframe(dataframe)

    base = _base_inputs(dataframe, cfg)
    _, _, reduced = _geometry_v3_stage_tables_from_base(base, cfg)
    columns = _ranked_line_columns(base, reduced, cfg)

    source = dataframe.copy()
    p = cfg.output_prefix
    existing = [col for col in source.columns if str(col).startswith(f"{p}_")]
    clean = source.drop(columns=existing).copy() if existing else source.copy()
    return pd.concat([clean, pd.DataFrame(columns, index=dataframe.index)], axis=1)


def _geometry_v3_stage_tables(frame: DataFrame, cfg: PatternGeometryV3Config) -> tuple[dict[str, Series], DataFrame, DataFrame, DataFrame]:
    base = _base_inputs(frame, cfg)
    return (base, *_geometry_v3_stage_tables_from_base(base, cfg))


def _geometry_v3_stage_tables_from_base(
    base: dict[str, Series],
    cfg: PatternGeometryV3Config,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    raw = _raw_line_candidates(base, cfg)
    filtered = _filter_raw_lines(raw, base, cfg)
    reduced = _reduce_lines(filtered, base, cfg)
    if bool(cfg.co_terminal_outer_suppression):
        reduced = _suppress_co_terminal_outer_lines(reduced, base, cfg)
    if bool(cfg.final_selection_recheck_use_atr):
        reduced = _final_distance_recheck(reduced, base, float(cfg.final_selection_max_distance_atr))
    if reduced.empty:
        return raw, filtered, _empty_line_table()
    reduced = reduced.sort_values(["side", "pivot_count", "score", "span"], ascending=[True, False, False, False]).reset_index(drop=True)
    return raw, filtered, reduced


def _base_inputs(frame: DataFrame, cfg: PatternGeometryV3Config) -> dict[str, Series]:
    open_ = _num(frame, "open")
    close = _num(frame, "close")
    high = _num(frame, "high")
    low = _num(frame, "low")
    body_high = pd.concat([open_, close], axis=1).max(axis=1)
    body_low = pd.concat([open_, close], axis=1).min(axis=1)
    atr = _atr(frame, 14)
    bar_index = pd.Series(np.arange(len(frame), dtype="float64"), index=frame.index)
    true_range = _true_range(high, low, close)
    last_shock_index = _last_shock_index(
        open_=open_.to_numpy(dtype="float64"),
        close=close.to_numpy(dtype="float64"),
        true_range=true_range.to_numpy(dtype="float64"),
        atr=atr.to_numpy(dtype="float64"),
        tr_threshold=float(cfg.shock_tr_atr_threshold),
        body_threshold=float(cfg.shock_body_atr_threshold),
    )
    pivots = build_clean_pivot_source(
        body_high=body_high,
        body_low=body_low,
        atr=atr,
        bar_index=bar_index,
        strength=int(cfg.pivot_strength),
        min_prominence_atr=float(cfg.min_pivot_prominence_atr),
    )
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "body_high": body_high,
        "body_low": body_low,
        "atr": atr,
        "true_range": true_range,
        "bar_index": bar_index,
        "last_shock_index": pd.Series(last_shock_index, index=frame.index, dtype="float64"),
        **pivots,
    }


def _raw_line_candidates(base: dict[str, Series], cfg: PatternGeometryV3Config) -> DataFrame:
    rows = []
    for side in ("resistance", "support"):
        pivot_frame = _pivot_frame(base, side)
        if len(pivot_frame) < 2:
            continue
        prices = pivot_frame["price"].to_numpy(dtype="float64")
        indexes = pivot_frame["anchor_index"].to_numpy(dtype="float64")
        available = pivot_frame["available_index"].to_numpy(dtype="float64")
        prominences = pivot_frame["prominence"].to_numpy(dtype="float64")
        pivot_scores = pivot_frame["score"].fillna(0.0).to_numpy(dtype="float64")
        for left in range(len(pivot_frame) - 1):
            for right in range(left + 1, len(pivot_frame)):
                x_old = float(indexes[left])
                x_new = float(indexes[right])
                span = x_new - x_old
                if span <= 0.0:
                    continue
                y_old = float(prices[left])
                y_new = float(prices[right])
                slope = (y_new - y_old) / span
                intercept = y_old - slope * x_old
                atr_scale = _median_atr(base["atr"], int(round(x_old)), int(round(x_new)))
                slope_atr = abs(slope) / max(atr_scale, 1e-9)
                rows.append(
                    _line_row(
                        side=side,
                        pivots=(int(round(x_old)), int(round(x_new))),
                        prices=(y_old, y_new),
                        available=(float(available[left]), float(available[right])),
                        prominences=(float(prominences[left]), float(prominences[right])),
                        pivot_scores=(float(pivot_scores[left]), float(pivot_scores[right])),
                        slope=float(slope),
                        intercept=float(intercept),
                        slope_atr=float(slope_atr),
                        line_kind="raw",
                    )
                )
    if not rows:
        return _empty_line_table()
    return pd.DataFrame(rows)


def _pivot_frame(base: dict[str, Series], side: str) -> DataFrame:
    if side == "resistance":
        return pd.DataFrame(
            {
                "price": base["pivot_high"],
                "anchor_index": base["pivot_high_index"],
                "available_index": base["pivot_high_available_index"],
                "prominence": base["pivot_high_prominence"],
                "score": base["pivot_high_score"],
            }
        ).dropna(subset=["price", "anchor_index", "available_index", "prominence"])
    return pd.DataFrame(
        {
            "price": base["pivot_low"],
            "anchor_index": base["pivot_low_index"],
            "available_index": base["pivot_low_available_index"],
            "prominence": base["pivot_low_prominence"],
            "score": base["pivot_low_score"],
        }
    ).dropna(subset=["price", "anchor_index", "available_index", "prominence"])


def _line_row(
    *,
    side: str,
    pivots: tuple[int, ...],
    prices: tuple[float, ...],
    available: tuple[float, ...],
    prominences: tuple[float, ...],
    pivot_scores: tuple[float, ...],
    slope: float,
    intercept: float,
    slope_atr: float,
    line_kind: str,
) -> dict[str, float | str]:
    x_old = float(pivots[0])
    x_end = float(pivots[-1])
    y_old = float(prices[0])
    y_end = float(prices[-1])
    span = max(x_end - x_old, 0.0)
    pivot_count = float(len(pivots))
    mean_prominence = float(np.nanmean(np.asarray(prominences, dtype="float64")))
    mean_pivot_score = float(np.nanmean(np.asarray(pivot_scores, dtype="float64")))
    score = float(pivot_count + min(mean_prominence, 8.0) * 0.25 + mean_pivot_score)
    travel_atr = float(abs(slope_atr) * span)
    impulse_score = float(travel_atr / max(np.sqrt(max(span, 1.0)), 1e-9))
    return {
        "side": str(side),
        "x_old": x_old,
        "x_new": float(pivots[1]) if len(pivots) > 1 else x_old,
        "x_end": x_end,
        "y_old": y_old,
        "y_new": float(prices[1]) if len(prices) > 1 else y_old,
        "y_end": y_end,
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_atr_per_bar": float(slope_atr),
        "span": float(span),
        "travel_atr": travel_atr,
        "impulse_score": impulse_score,
        "pivot_count": pivot_count,
        "score": score,
        "prominence_mean": mean_prominence,
        "pivot_score_mean": mean_pivot_score,
        "pivot_path": "|".join(str(int(pivot)) for pivot in pivots),
        "price_path": "|".join(str(float(price)) for price in prices),
        "available_path": "|".join(str(float(value)) for value in available),
        "prominence_path": "|".join(str(float(value)) for value in prominences),
        "line_kind": str(line_kind),
        "live_start": float(max(available)),
    }


def _fails_impulse_filter(candidate: pd.Series | dict[str, float | str], cfg: PatternGeometryV3Config) -> bool:
    mode = str(cfg.impulse_filter_mode).lower()
    if mode == "off":
        return False
    span = float(candidate["span"])
    if span < float(cfg.impulse_min_span_bars):
        return False
    slope_atr = float(candidate["slope_atr_per_bar"])
    travel_atr = float(candidate["travel_atr"])
    impulse_score = float(candidate["impulse_score"])
    if mode == "span_slope":
        return slope_atr >= float(cfg.impulse_max_slope_atr_per_bar)
    if mode == "travel":
        return travel_atr >= float(cfg.impulse_max_travel_atr)
    if mode == "score":
        return impulse_score >= float(cfg.impulse_max_score)
    if mode == "hybrid":
        return (
            slope_atr >= float(cfg.impulse_max_slope_atr_per_bar)
            and travel_atr >= float(cfg.impulse_max_travel_atr)
        )
    raise ValueError("impulse_filter_mode must be one of: off, span_slope, travel, score, hybrid")


def _fails_internal_amplitude_filter(
    candidate: pd.Series | dict[str, float | str],
    base: dict[str, Series],
    cfg: PatternGeometryV3Config,
) -> bool:
    mode = str(cfg.internal_amplitude_mode).lower()
    if mode == "off":
        return False
    units = _internal_amplitude_units(candidate, base, cfg)
    if not np.isfinite(units):
        return True
    if mode == "fixed":
        return units > float(cfg.internal_amplitude_max_atr)
    if mode == "dynamic":
        return units > _dynamic_amplitude_limit(int(round(float(_row_value(candidate, "span")))), cfg)
    raise ValueError("internal_amplitude_mode must be one of: off, fixed, dynamic")


def _internal_amplitude_units(
    candidate: pd.Series | dict[str, float | str],
    base: dict[str, Series],
    cfg: PatternGeometryV3Config,
) -> float:
    start = int(round(float(_row_value(candidate, "x_old"))))
    end = int(round(float(_row_value(candidate, "x_end"))))
    if end < start:
        return np.nan
    high = base["high"].iloc[start : end + 1].to_numpy(dtype="float64")
    low = base["low"].iloc[start : end + 1].to_numpy(dtype="float64")
    amplitude = float(np.nanmax(high) - np.nanmin(low))
    scale = _span_volatility_scale(base, start, end, cfg)
    return amplitude / max(scale, 1e-9)


def _span_volatility_scale(base: dict[str, Series], start: int, end: int, cfg: PatternGeometryV3Config) -> float:
    mode = str(cfg.volatility_scale_mode).lower()
    atr = base["atr"].iloc[start : end + 1].to_numpy(dtype="float64")
    atr_scale = float(np.nanmedian(atr))
    if mode == "atr":
        return max(atr_scale, 1e-9)
    true_range = base["true_range"].iloc[start : end + 1].to_numpy(dtype="float64")
    body = (
        base["body_high"].iloc[start : end + 1].to_numpy(dtype="float64")
        - base["body_low"].iloc[start : end + 1].to_numpy(dtype="float64")
    )
    return max(
        atr_scale,
        float(np.nanmedian(true_range)),
        float(np.nanmedian(body)) * float(cfg.volatility_body_mult),
        1e-9,
    )


def _dynamic_amplitude_limit(span: int, cfg: PatternGeometryV3Config) -> float:
    reference = max(float(cfg.internal_amplitude_reference_bars), 1.0)
    allowed = float(cfg.internal_amplitude_base_atr) + float(cfg.internal_amplitude_span_growth_atr) * np.sqrt(
        max(float(span), 1.0) / reference
    )
    return float(min(allowed, float(cfg.internal_amplitude_max_dynamic_atr)))


def _filter_raw_lines(candidates: DataFrame, base: dict[str, Series], cfg: PatternGeometryV3Config) -> DataFrame:
    if candidates.empty:
        return _empty_line_table()

    kept_rows: list[pd.Series] = []
    for _, candidate in candidates.iterrows():
        span = float(candidate["span"])
        if span < float(cfg.min_seed_span_bars) or span > float(cfg.max_seed_span_bars):
            continue
        if float(candidate["slope_atr_per_bar"]) > float(cfg.max_slope_atr_per_bar):
            continue
        if _fails_impulse_filter(candidate, cfg):
            continue
        if _fails_internal_amplitude_filter(candidate, base, cfg):
            continue
        if not _anchor_span_is_clear(candidate, base, cfg):
            continue
        kept_rows.append(candidate.copy())
    if not kept_rows:
        return _empty_line_table()
    return pd.DataFrame(kept_rows).reset_index(drop=True)


def _anchor_span_is_clear(candidate: pd.Series, base: dict[str, Series], cfg: PatternGeometryV3Config) -> bool:
    x_old = int(round(float(candidate["x_old"])))
    x_new = int(round(float(candidate["x_new"])))
    start = max(x_old + 1, 0)
    end = min(x_new, len(base["close"]))
    if end <= start:
        return True

    xs = np.arange(start, end, dtype="float64")
    line = float(candidate["intercept"]) + float(candidate["slope"]) * xs
    if str(candidate["side"]) == "resistance":
        body = base["body_high"].iloc[start:end].to_numpy(dtype="float64")
        breach = body > line
        depth = body - line
    else:
        body = base["body_low"].iloc[start:end].to_numpy(dtype="float64")
        breach = body < line
        depth = line - body

    if not bool(np.any(breach)):
        return True

    breach_indexes = np.flatnonzero(breach)
    breach_x = xs[breach_indexes]
    near_anchor = (breach_x <= float(x_old + int(cfg.anchor_break_edge_bars))) | (
        breach_x >= float(x_new - int(cfg.anchor_break_edge_bars))
    )
    interior = breach_indexes[~near_anchor]
    if len(interior) == 0:
        return True

    interior_mask = np.zeros_like(breach, dtype=bool)
    interior_mask[interior] = True
    if _max_true_run(interior_mask) > int(cfg.anchor_break_max_run):
        return False

    atr = base["atr"].iloc[start:end].to_numpy(dtype="float64")
    depth_atr = depth[interior] / np.maximum(atr[interior], 1e-9)
    return bool(np.nanmax(depth_atr) <= float(cfg.anchor_break_max_depth_atr))


def _reduce_lines(candidates: DataFrame, base: dict[str, Series], cfg: PatternGeometryV3Config) -> DataFrame:
    if candidates.empty:
        return _empty_line_table()
    sequence = str(cfg.step4_reduction_sequence).upper()
    steps = {
        "A": (_join_lines, _delete_subset_lines, _merge_partial_lines),
        "B": (_delete_subset_lines, _join_lines, _merge_partial_lines),
        "C": (_join_lines, _merge_partial_lines, _delete_subset_lines),
    }.get(sequence)
    if steps is None:
        raise ValueError("step4_reduction_sequence must be one of: A, B, C")

    reduced = candidates.copy()
    for step in steps:
        reduced = step(reduced, base, cfg)
        if reduced.empty:
            break
    return reduced.reset_index(drop=True)


def _suppress_co_terminal_outer_lines(candidates: DataFrame, base: dict[str, Series], cfg: PatternGeometryV3Config) -> DataFrame:
    if candidates.empty:
        return _empty_line_table()

    remove: set[int] = set()
    for _, side_candidates in candidates.groupby("side", sort=False):
        records = list(side_candidates.iterrows())
        for left_pos, (left_idx, left) in enumerate(records):
            if int(left_idx) in remove:
                continue
            for right_idx, right in records[left_pos + 1 :]:
                if int(right_idx) in remove:
                    continue
                if not _co_terminal_pair(left, right, base, cfg):
                    continue
                outer = _outer_line_index(left_idx, left, right_idx, right, base)
                remove.add(int(outer))
    if not remove:
        return candidates.reset_index(drop=True)
    result = candidates.drop(index=list(remove), errors="ignore").reset_index(drop=True)
    return result if not result.empty else _empty_line_table()


def _co_terminal_pair(left: pd.Series, right: pd.Series, base: dict[str, Series], cfg: PatternGeometryV3Config) -> bool:
    if abs(float(left["x_old"]) - float(right["x_old"])) > float(cfg.co_terminal_start_tolerance_bars):
        return False
    if abs(float(left["x_end"]) - float(right["x_end"])) > float(cfg.co_terminal_end_tolerance_bars):
        return False
    overlap_start = max(int(round(float(left["x_old"]))), int(round(float(right["x_old"]))))
    overlap_end = min(int(round(float(left["x_end"]))), int(round(float(right["x_end"]))))
    if overlap_end <= overlap_start:
        return False
    min_span = max(min(float(left["span"]), float(right["span"])), 1.0)
    if (overlap_end - overlap_start) / min_span < float(cfg.co_terminal_min_overlap_ratio):
        return False
    return _mean_line_gap_atr(left, right, base, overlap_start, overlap_end) <= float(cfg.co_terminal_max_mean_gap_atr)


def _outer_line_index(
    left_idx: int,
    left: pd.Series,
    right_idx: int,
    right: pd.Series,
    base: dict[str, Series],
) -> int:
    overlap_start = max(int(round(float(left["x_old"]))), int(round(float(right["x_old"]))))
    overlap_end = min(int(round(float(left["x_end"]))), int(round(float(right["x_end"]))))
    left_mean = _mean_line_value(left, overlap_start, overlap_end)
    right_mean = _mean_line_value(right, overlap_start, overlap_end)
    side = str(left["side"])
    if side == "resistance":
        return int(left_idx if left_mean >= right_mean else right_idx)
    return int(left_idx if left_mean <= right_mean else right_idx)


def _mean_line_gap_atr(left: pd.Series, right: pd.Series, base: dict[str, Series], start: int, end: int) -> float:
    xs = np.arange(start, end + 1, dtype="float64")
    left_line = float(left["intercept"]) + float(left["slope"]) * xs
    right_line = float(right["intercept"]) + float(right["slope"]) * xs
    atr = base["atr"].iloc[start : end + 1].to_numpy(dtype="float64")
    gap = np.abs(left_line - right_line) / np.maximum(atr, 1e-9)
    return float(np.nanmean(gap))


def _mean_line_value(row: pd.Series, start: int, end: int) -> float:
    xs = np.arange(start, end + 1, dtype="float64")
    line = float(row["intercept"]) + float(row["slope"]) * xs
    return float(np.nanmean(line))


def _join_lines(candidates: DataFrame, base: dict[str, Series], cfg: PatternGeometryV3Config) -> DataFrame:
    if candidates.empty:
        return _empty_line_table()

    output_rows: list[dict[str, float | str]] = candidates.to_dict("records")
    absorbed_component_paths: set[tuple[str, str]] = set()
    for side, side_candidates in candidates.groupby("side", sort=False):
        by_old_anchor = {float(anchor): group for anchor, group in side_candidates.groupby("x_old")}
        frontier = side_candidates.to_dict("records")
        for _ in range(3, int(cfg.max_joined_pivots) + 1):
            next_frontier: list[dict[str, float | str]] = []
            seen_paths: set[str] = set()
            for path in frontier:
                path_pivots = _pivot_tuple(path["pivot_path"])
                path_prices = _value_tuple(path["price_path"])
                path_available = _value_tuple(path["available_path"])
                path_prominence = _value_tuple(path["prominence_path"])
                continuations = by_old_anchor.get(float(path_pivots[-1]))
                if continuations is None:
                    continue
                path_norm = _signed_norm_slope(path)
                for segment in continuations.itertuples(index=False):
                    next_pivot = int(round(float(segment.x_new)))
                    if next_pivot <= path_pivots[-1] or next_pivot in path_pivots:
                        continue
                    if not _angle_within_tolerance(path_norm, _signed_norm_slope(segment), float(cfg.join_angle_tolerance)):
                        continue
                    new_pivots = (*path_pivots, next_pivot)
                    new_prices = (*path_prices, float(segment.y_new))
                    new_available = (*path_available, max(float(segment.live_start), float(segment.x_new)))
                    new_prominence = (*path_prominence, float(segment.prominence_mean))
                    joined = _path_candidate(
                        side=str(side),
                        pivots=new_pivots,
                        prices=new_prices,
                        available=new_available,
                        prominences=new_prominence,
                        pivot_scores=tuple([float(segment.pivot_score_mean)] * len(new_pivots)),
                        base=base,
                        cfg=cfg,
                        line_kind="joined",
                    )
                    if joined is None:
                        continue
                    key = str(joined["pivot_path"])
                    if key in seen_paths:
                        continue
                    seen_paths.add(key)
                    absorbed_component_paths.add((str(side), str(path["pivot_path"])))
                    absorbed_component_paths.add((str(side), str(segment.pivot_path)))
                    next_frontier.append(joined)
            if not next_frontier:
                break
            output_rows.extend(next_frontier)
            frontier = next_frontier

    joined = pd.DataFrame(output_rows)
    if joined.empty:
        return _empty_line_table()
    if absorbed_component_paths:
        keep_mask = [
            (str(row["side"]), str(row["pivot_path"])) not in absorbed_component_paths
            for row in output_rows
        ]
        joined = pd.DataFrame([row for row, keep in zip(output_rows, keep_mask, strict=False) if keep])
    if joined.empty:
        return _empty_line_table()
    joined = joined.sort_values(["side", "pivot_count", "score", "span"], ascending=[True, False, False, False])
    joined = joined.drop_duplicates(["side", "pivot_path"], keep="first").reset_index(drop=True)
    return joined


def _delete_subset_lines(candidates: DataFrame, base: dict[str, Series], cfg: PatternGeometryV3Config) -> DataFrame:
    if candidates.empty:
        return _empty_line_table()

    kept_rows: list[pd.Series] = []
    for _, side_candidates in candidates.sort_values(["pivot_count", "score", "span"], ascending=False).groupby("side", sort=False):
        side_kept: list[pd.Series] = []
        for _, candidate in side_candidates.iterrows():
            candidate_pivots = _pivot_set(candidate["pivot_path"])
            absorbed = False
            for kept in side_kept:
                kept_pivots = _pivot_set(kept["pivot_path"])
                if len(candidate_pivots) < int(cfg.shared_subset_min_pivots):
                    continue
                if candidate_pivots.issubset(kept_pivots):
                    absorbed = True
                    break
            if not absorbed:
                side_kept.append(candidate.copy())
        kept_rows.extend(side_kept)
    if not kept_rows:
        return _empty_line_table()
    return pd.DataFrame(kept_rows).reset_index(drop=True)


def _merge_partial_lines(candidates: DataFrame, base: dict[str, Series], cfg: PatternGeometryV3Config) -> DataFrame:
    if candidates.empty:
        return _empty_line_table()

    out_rows: list[pd.Series] = []
    for _, side_candidates in candidates.sort_values(["pivot_count", "score", "span"], ascending=False).groupby("side", sort=False):
        used: set[int] = set()
        records = list(side_candidates.iterrows())
        for left_pos, (_, candidate) in enumerate(records):
            if left_pos in used:
                continue
            best = candidate.copy()
            best_pivots = _pivot_set(best["pivot_path"])
            for right_pos in range(left_pos + 1, len(records)):
                if right_pos in used:
                    continue
                _, other = records[right_pos]
                shared = best_pivots.intersection(_pivot_set(other["pivot_path"]))
                if len(shared) < int(cfg.partial_merge_shared_pivots_min):
                    continue
                if not _angle_within_tolerance(
                    _signed_norm_slope(best),
                    _signed_norm_slope(other),
                    float(cfg.partial_merge_angle_tolerance),
                ):
                    continue
                merged = _merge_two_lines(best, other, base, cfg)
                if merged is None:
                    continue
                best = merged
                best_pivots = _pivot_set(best["pivot_path"])
                used.add(right_pos)
            out_rows.append(best)
    if not out_rows:
        return _empty_line_table()
    return pd.DataFrame(out_rows).reset_index(drop=True)


def _merge_two_lines(
    left: pd.Series,
    right: pd.Series,
    base: dict[str, Series],
    cfg: PatternGeometryV3Config,
) -> pd.Series | None:
    path = _merged_path(left, right)
    merged = _path_candidate(
        side=str(left["side"]),
        pivots=tuple(path["pivots"]),
        prices=tuple(path["prices"]),
        available=tuple(path["available"]),
        prominences=tuple(path["prominences"]),
        pivot_scores=tuple(path["pivot_scores"]),
        base=base,
        cfg=cfg,
        line_kind="merged",
    )
    if merged is None:
        return None
    if bool(cfg.partial_merge_use_atr_check) and not _within_atr_distance(
        merged,
        base,
        float(cfg.partial_merge_max_unmatched_pivot_distance_atr),
    ):
        return None
    return pd.Series(merged)


def _merged_path(left: pd.Series, right: pd.Series) -> dict[str, list[float]]:
    points: dict[int, dict[str, float]] = {}
    for row in (left, right):
        pivots = _pivot_tuple(row["pivot_path"])
        prices = _value_tuple(row["price_path"])
        available = _value_tuple(row["available_path"])
        prominences = _value_tuple(row["prominence_path"])
        pivot_score = float(row.get("pivot_score_mean", 0.0))
        for pivot, price, avail, prominence in zip(pivots, prices, available, prominences, strict=False):
            points[int(pivot)] = {
                "price": float(price),
                "available": float(avail),
                "prominence": float(prominence),
                "pivot_score": float(pivot_score),
            }
    ordered = sorted(points.items())
    return {
        "pivots": [float(pivot) for pivot, _ in ordered],
        "prices": [point["price"] for _, point in ordered],
        "available": [point["available"] for _, point in ordered],
        "prominences": [point["prominence"] for _, point in ordered],
        "pivot_scores": [point["pivot_score"] for _, point in ordered],
    }


def _path_candidate(
    *,
    side: str,
    pivots: tuple[int | float, ...],
    prices: tuple[float, ...],
    available: tuple[float, ...],
    prominences: tuple[float, ...],
    pivot_scores: tuple[float, ...],
    base: dict[str, Series],
    cfg: PatternGeometryV3Config,
    line_kind: str,
) -> dict[str, float | str] | None:
    if len(pivots) < 2 or len(prices) != len(pivots):
        return None
    pivot_indexes = tuple(int(round(float(pivot))) for pivot in pivots)
    price_values = tuple(float(price) for price in prices)
    available_values = tuple(float(value) for value in available[: len(pivot_indexes)])
    prominence_values = tuple(float(value) for value in prominences[: len(pivot_indexes)])
    pivot_score_values = tuple(float(value) for value in pivot_scores[: len(pivot_indexes)])
    x_old = float(pivot_indexes[0])
    x_end = float(pivot_indexes[-1])
    span = x_end - x_old
    if span <= 0.0:
        return None
    slope, intercept = _fit_line_to_path(pivot_indexes, price_values)
    if not np.isfinite([slope, intercept]).all():
        return None
    atr_scale = _median_atr(base["atr"], int(round(x_old)), int(round(x_end)))
    slope_atr = abs(slope) / max(atr_scale, 1e-9)
    if slope_atr > float(cfg.max_slope_atr_per_bar):
        return None
    row = _line_row(
        side=side,
        pivots=tuple(int(value) for value in pivot_indexes),
        prices=price_values,
        available=available_values,
        prominences=prominence_values,
        pivot_scores=pivot_score_values,
        slope=float(slope),
        intercept=float(intercept),
        slope_atr=float(slope_atr),
        line_kind=line_kind,
    )
    if _fails_impulse_filter(row, cfg):
        return None
    if _fails_internal_amplitude_filter(row, base, cfg):
        return None
    return row


def _fit_line_to_path(pivots: tuple[int, ...], prices: tuple[float, ...]) -> tuple[float, float]:
    if len(pivots) < 2 or len(prices) != len(pivots):
        return np.nan, np.nan
    x = np.asarray(pivots, dtype="float64")
    y = np.asarray(prices, dtype="float64")
    if len(pivots) == 2:
        span = float(x[-1] - x[0])
        if span <= 0.0:
            return np.nan, np.nan
        slope = float((y[-1] - y[0]) / span)
        intercept = float(y[0] - slope * x[0])
        return slope, intercept
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _final_distance_recheck(candidates: DataFrame, base: dict[str, Series], max_distance_atr: float) -> DataFrame:
    if candidates.empty:
        return _empty_line_table()
    mask = candidates.apply(lambda row: _within_atr_distance(row, base, max_distance_atr), axis=1)
    result = candidates.loc[mask.fillna(False)].reset_index(drop=True)
    return result if not result.empty else _empty_line_table()


def _within_atr_distance(candidate: pd.Series | dict[str, float | str], base: dict[str, Series], max_distance_atr: float) -> bool:
    points = _pivot_points(candidate)
    if not points:
        return False
    slope = float(candidate["slope"])
    intercept = float(candidate["intercept"])
    atr = base["atr"].to_numpy(dtype="float64")
    for pivot, price in points:
        idx = int(round(float(pivot)))
        if idx < 0 or idx >= len(atr):
            return False
        line_price = intercept + slope * float(pivot)
        distance_atr = abs(float(price) - line_price) / max(float(atr[idx]), 1e-9)
        if distance_atr > float(max_distance_atr):
            return False
    return True


def _pivot_points(candidate: pd.Series | dict[str, float | str]) -> list[tuple[int, float]]:
    pivots = _pivot_tuple(candidate["pivot_path"])
    prices = _value_tuple(candidate["price_path"])
    if len(pivots) != len(prices):
        return []
    return [(int(pivot), float(price)) for pivot, price in zip(pivots, prices, strict=False)]


def _ranked_line_columns(base: dict[str, Series], reduced: DataFrame, cfg: PatternGeometryV3Config) -> dict[str, Series]:
    index = base["close"].index
    rows = len(index)
    prefix = str(cfg.output_prefix)
    slots = int(cfg.output_slots)
    outputs: dict[str, Series] = {}
    if reduced.empty or rows == 0:
        return _empty_ranked_line_columns(index, prefix, slots)

    x_values = np.arange(rows, dtype="float64")
    close = base["close"].to_numpy(dtype="float64")
    atr = base["atr"].clip(lower=1e-9).to_numpy(dtype="float64")
    for side in ("resistance", "support"):
        top = _empty_top_line_arrays(rows, slots)
        side_candidates = reduced[reduced["side"].eq(side)].sort_values(
            ["pivot_count", "score", "span"],
            ascending=[False, False, False],
        )
        for line_id, candidate in enumerate(side_candidates.itertuples(index=False), start=1):
            start = max(int(np.ceil(float(candidate.live_start))), 0)
            start = max(
                start,
                _shock_safe_start(
                    start=start,
                    anchor_start=int(round(float(candidate.x_old))),
                    last_shock_index=base["last_shock_index"].to_numpy(dtype="float64"),
                    cooldown_bars=int(cfg.shock_cooldown_bars),
                ),
            )
            end = rows - 1
            if end < start:
                continue
            xs = x_values[start : end + 1]
            line = float(candidate.intercept) + float(candidate.slope) * xs
            distance_atr = np.abs(close[start : end + 1] - line) / atr[start : end + 1]
            termination = _active_termination_index(
                side=side,
                close=close[start : end + 1],
                line=line,
                distance_atr=distance_atr,
                max_distance_atr=float(cfg.max_active_price_distance_atr),
                break_close_run=int(cfg.active_break_close_run),
            )
            if termination < 0:
                continue
            active_rows = np.arange(start, start + termination + 1, dtype=int)
            score = np.full(rows, -np.inf, dtype="float64")
            score[active_rows] = float(candidate.score)
            metrics: dict[str, np.ndarray] = {
                "line": np.full(rows, np.nan, dtype="float64"),
                "start_index": _masked_constant_metric(rows, float(candidate.x_old), active_rows),
                "end_index": _masked_constant_metric(rows, float(candidate.x_end), active_rows),
                "pivot_count": _masked_constant_metric(rows, float(candidate.pivot_count), active_rows),
                "score": _masked_constant_metric(rows, float(candidate.score), active_rows),
                "slope": _masked_constant_metric(rows, float(candidate.slope), active_rows),
                "line_id": _masked_constant_metric(rows, float(line_id), active_rows),
                "line_kind": np.full(rows, "", dtype=object),
            }
            metrics["line"][active_rows] = line[: termination + 1]
            metrics["line_kind"][active_rows] = str(candidate.line_kind)
            _insert_top_line_candidate(top, score, metrics)
        outputs.update(_top_line_arrays_to_columns(top, index, prefix, side))
    return outputs


def _constant_metric(rows: int, value: float, start: int, end: int) -> np.ndarray:
    out = np.full(rows, np.nan, dtype="float64")
    if np.isfinite(value) and end >= start:
        out[start : end + 1] = value
    return out


def _masked_constant_metric(rows: int, value: float, active_rows: np.ndarray) -> np.ndarray:
    out = np.full(rows, np.nan, dtype="float64")
    if np.isfinite(value) and active_rows.size:
        out[active_rows] = value
    return out


def _active_termination_index(
    *,
    side: str,
    close: np.ndarray,
    line: np.ndarray,
    distance_atr: np.ndarray,
    max_distance_atr: float,
    break_close_run: int,
) -> int:
    if len(line) == 0:
        return -1
    distance_breach = np.isfinite(distance_atr) & (distance_atr > float(max_distance_atr))
    distance_stop = _first_true_run_start(distance_breach, 1)
    if side == "resistance":
        wrong_side = np.isfinite(close) & np.isfinite(line) & (close > line)
    else:
        wrong_side = np.isfinite(close) & np.isfinite(line) & (close < line)
    break_stop = _first_true_run_start(wrong_side, max(int(break_close_run), 1))
    stops = [stop for stop in (distance_stop, break_stop) if stop is not None]
    if not stops:
        return len(line) - 1
    return int(min(stops) - 1)


def _shock_safe_start(
    *,
    start: int,
    anchor_start: int,
    last_shock_index: np.ndarray,
    cooldown_bars: int,
) -> int:
    cooldown = int(cooldown_bars)
    if cooldown <= 0 or start <= 0 or start >= len(last_shock_index):
        return start
    last = int(last_shock_index[start])
    if last < anchor_start:
        return start
    return max(start, last + cooldown)


def _first_true_run_start(mask: np.ndarray, run_length: int) -> int | None:
    needed = max(int(run_length), 1)
    run = 0
    for idx, value in enumerate(mask):
        if bool(value):
            run += 1
            if run >= needed:
                return idx - needed + 1
        else:
            run = 0
    return None


def _empty_top_line_arrays(rows: int, slots: int) -> dict[str, np.ndarray]:
    return {
        "score": np.full((rows, slots), -np.inf, dtype="float64"),
        "line": np.full((rows, slots), np.nan, dtype="float64"),
        "start_index": np.full((rows, slots), np.nan, dtype="float64"),
        "end_index": np.full((rows, slots), np.nan, dtype="float64"),
        "pivot_count": np.full((rows, slots), np.nan, dtype="float64"),
        "slope": np.full((rows, slots), np.nan, dtype="float64"),
        "line_id": np.full((rows, slots), np.nan, dtype="float64"),
        "line_kind": np.full((rows, slots), "", dtype=object),
    }


def _insert_top_line_candidate(top: dict[str, np.ndarray], score: np.ndarray, metrics: dict[str, np.ndarray]) -> None:
    inserted = np.zeros(score.shape[0], dtype=bool)
    slots = top["score"].shape[1]
    for rank in range(slots):
        mask = (~inserted) & (score > top["score"][:, rank])
        if not np.any(mask):
            continue
        if rank < slots - 1:
            for key, values in top.items():
                values[mask, rank + 1 :] = values[mask, rank:-1]
        top["score"][mask, rank] = score[mask]
        for key, candidate_values in metrics.items():
            top[key][mask, rank] = candidate_values[mask]
        inserted[mask] = True
        if np.all(inserted):
            break


def _top_line_arrays_to_columns(top: dict[str, np.ndarray], index: pd.Index, prefix: str, side: str) -> dict[str, Series]:
    out: dict[str, Series] = {}
    slots = top["score"].shape[1]
    for rank in range(slots):
        valid = np.isfinite(top["score"][:, rank]) & (top["score"][:, rank] > -np.inf)
        suffix = rank + 1
        out[f"{prefix}_{side}_line_rank{suffix}"] = pd.Series(
            np.where(valid, top["line"][:, rank], np.nan),
            index=index,
            dtype="float64",
        )
        out[f"{prefix}_{side}_start_index_rank{suffix}"] = pd.Series(
            np.where(valid, top["start_index"][:, rank], np.nan),
            index=index,
            dtype="float64",
        )
        out[f"{prefix}_{side}_end_index_rank{suffix}"] = pd.Series(
            np.where(valid, top["end_index"][:, rank], np.nan),
            index=index,
            dtype="float64",
        )
        out[f"{prefix}_{side}_pivot_count_rank{suffix}"] = pd.Series(
            np.where(valid, top["pivot_count"][:, rank], np.nan),
            index=index,
            dtype="float64",
        )
        out[f"{prefix}_{side}_score_rank{suffix}"] = pd.Series(
            np.where(valid, top["score"][:, rank], np.nan),
            index=index,
            dtype="float64",
        )
        out[f"{prefix}_{side}_slope_rank{suffix}"] = pd.Series(
            np.where(valid, top["slope"][:, rank], np.nan),
            index=index,
            dtype="float64",
        )
        line_kind = np.where(valid, top["line_kind"][:, rank], "")
        out[f"{prefix}_{side}_line_kind_rank{suffix}"] = pd.Series(line_kind, index=index, dtype="object")
    return out


def _empty_ranked_line_columns(index: pd.Index, prefix: str, slots: int) -> dict[str, Series]:
    out: dict[str, Series] = {}
    for side in ("resistance", "support"):
        for rank in range(1, slots + 1):
            out[f"{prefix}_{side}_line_rank{rank}"] = pd.Series(np.nan, index=index, dtype="float64")
            out[f"{prefix}_{side}_start_index_rank{rank}"] = pd.Series(np.nan, index=index, dtype="float64")
            out[f"{prefix}_{side}_end_index_rank{rank}"] = pd.Series(np.nan, index=index, dtype="float64")
            out[f"{prefix}_{side}_pivot_count_rank{rank}"] = pd.Series(np.nan, index=index, dtype="float64")
            out[f"{prefix}_{side}_score_rank{rank}"] = pd.Series(np.nan, index=index, dtype="float64")
            out[f"{prefix}_{side}_slope_rank{rank}"] = pd.Series(np.nan, index=index, dtype="float64")
            out[f"{prefix}_{side}_line_kind_rank{rank}"] = pd.Series("", index=index, dtype="object")
    return out


def _empty_line_table() -> DataFrame:
    return pd.DataFrame(
        columns=[
            "side",
            "x_old",
            "x_new",
            "x_end",
            "y_old",
            "y_new",
            "y_end",
            "slope",
            "intercept",
            "slope_atr_per_bar",
            "span",
            "travel_atr",
            "impulse_score",
            "pivot_count",
            "score",
            "prominence_mean",
            "pivot_score_mean",
            "pivot_path",
            "price_path",
            "available_path",
            "prominence_path",
            "line_kind",
            "live_start",
        ]
    )


def _signed_norm_slope(row: object) -> float:
    slope = float(_row_value(row, "slope"))
    slope_atr = float(_row_value(row, "slope_atr_per_bar"))
    return float(np.sign(slope) * abs(slope_atr))


def _row_value(row: object, field: str) -> object:
    if isinstance(row, pd.Series):
        return row[field]
    if isinstance(row, dict):
        return row[field]
    return getattr(row, field)


def _angle_within_tolerance(left: float, right: float, tolerance: float) -> bool:
    if not np.isfinite(left) or not np.isfinite(right):
        return False
    denominator = max(abs(left), abs(right), 1e-9)
    return abs(left - right) / denominator <= float(tolerance)


def _median_atr(atr: Series, start: int, end: int) -> float:
    left = max(int(start), 0)
    right = min(int(end), len(atr) - 1)
    if right < left:
        return 1e-9
    value = float(np.nanmedian(atr.iloc[left : right + 1].to_numpy(dtype="float64")))
    return max(value, 1e-9)


def _last_shock_index(
    *,
    open_: np.ndarray,
    close: np.ndarray,
    true_range: np.ndarray,
    atr: np.ndarray,
    tr_threshold: float,
    body_threshold: float,
) -> np.ndarray:
    body = np.abs(close - open_)
    shock = (true_range / np.maximum(atr, 1e-9) >= float(tr_threshold)) | (
        body / np.maximum(atr, 1e-9) >= float(body_threshold)
    )
    last = np.full(len(close), -1, dtype=int)
    seen = -1
    for idx, value in enumerate(shock):
        if bool(value):
            seen = idx
        last[idx] = seen
    return last


def _max_true_run(values: np.ndarray) -> int:
    best = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _pivot_tuple(path: object) -> tuple[int, ...]:
    return tuple(int(part) for part in str(path).split("|") if part != "")


def _value_tuple(path: object) -> tuple[float, ...]:
    return tuple(float(part) for part in str(path).split("|") if part != "")


def _pivot_set(path: object) -> set[int]:
    return {int(part) for part in str(path).split("|") if part != ""}


def _atr(frame: DataFrame, period: int) -> Series:
    high = _num(frame, "high")
    low = _num(frame, "low")
    close = _num(frame, "close")
    true_range = _true_range(high, low, close)
    return true_range.rolling(int(period), min_periods=max(2, int(period) // 2)).mean()


def _true_range(high: Series, low: Series, close: Series) -> Series:
    previous_close = close.shift(1)
    return pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1).astype("float64")


def _num(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _resolve_config(config: PatternGeometryV3Config | None, overrides: dict[str, object]) -> PatternGeometryV3Config:
    cfg = config or PatternGeometryV3Config()
    if not overrides:
        return cfg
    valid = {field.name for field in fields(PatternGeometryV3Config)}
    unknown = sorted(set(overrides).difference(valid))
    if unknown:
        raise TypeError(f"Unknown geometry v3 config override(s): {', '.join(unknown)}")
    return replace(cfg, **{name: overrides[name] for name in overrides if name in valid})


def _validate_dataframe(frame: DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Geometry v3 requires OHLCV columns. Missing: {', '.join(missing)}")


def _validate_config(cfg: PatternGeometryV3Config) -> None:
    if not cfg.output_prefix:
        raise ValueError("output_prefix must be set")
    if int(cfg.output_slots) < 1:
        raise ValueError("output_slots must be at least 1")
    if int(cfg.pivot_strength) < 1:
        raise ValueError("pivot_strength must be at least 1")
    if float(cfg.min_pivot_prominence_atr) < 0.0:
        raise ValueError("min_pivot_prominence_atr must be non-negative")
    if int(cfg.min_seed_span_bars) < 1:
        raise ValueError("min_seed_span_bars must be at least 1")
    if int(cfg.max_seed_span_bars) <= int(cfg.min_seed_span_bars):
        raise ValueError("max_seed_span_bars must be greater than min_seed_span_bars")
    if float(cfg.max_slope_atr_per_bar) <= 0.0:
        raise ValueError("max_slope_atr_per_bar must be positive")
    if str(cfg.volatility_scale_mode).lower() not in {"atr", "composite"}:
        raise ValueError("volatility_scale_mode must be one of: atr, composite")
    if float(cfg.volatility_body_mult) <= 0.0:
        raise ValueError("volatility_body_mult must be positive")
    if str(cfg.impulse_filter_mode).lower() not in {"off", "span_slope", "travel", "score", "hybrid"}:
        raise ValueError("impulse_filter_mode must be one of: off, span_slope, travel, score, hybrid")
    if int(cfg.impulse_min_span_bars) < 1:
        raise ValueError("impulse_min_span_bars must be at least 1")
    if float(cfg.impulse_max_slope_atr_per_bar) <= 0.0:
        raise ValueError("impulse_max_slope_atr_per_bar must be positive")
    if float(cfg.impulse_max_travel_atr) <= 0.0:
        raise ValueError("impulse_max_travel_atr must be positive")
    if float(cfg.impulse_max_score) <= 0.0:
        raise ValueError("impulse_max_score must be positive")
    if str(cfg.internal_amplitude_mode).lower() not in {"off", "fixed", "dynamic"}:
        raise ValueError("internal_amplitude_mode must be one of: off, fixed, dynamic")
    if float(cfg.internal_amplitude_max_atr) <= 0.0:
        raise ValueError("internal_amplitude_max_atr must be positive")
    if float(cfg.internal_amplitude_base_atr) <= 0.0:
        raise ValueError("internal_amplitude_base_atr must be positive")
    if float(cfg.internal_amplitude_span_growth_atr) < 0.0:
        raise ValueError("internal_amplitude_span_growth_atr must be non-negative")
    if int(cfg.internal_amplitude_reference_bars) < 1:
        raise ValueError("internal_amplitude_reference_bars must be at least 1")
    if float(cfg.internal_amplitude_max_dynamic_atr) <= 0.0:
        raise ValueError("internal_amplitude_max_dynamic_atr must be positive")
    if int(cfg.shock_cooldown_bars) < 0:
        raise ValueError("shock_cooldown_bars must be non-negative")
    if float(cfg.shock_tr_atr_threshold) <= 0.0:
        raise ValueError("shock_tr_atr_threshold must be positive")
    if float(cfg.shock_body_atr_threshold) <= 0.0:
        raise ValueError("shock_body_atr_threshold must be positive")
    if int(cfg.anchor_break_max_run) < 0:
        raise ValueError("anchor_break_max_run must be non-negative")
    if float(cfg.anchor_break_max_depth_atr) < 0.0:
        raise ValueError("anchor_break_max_depth_atr must be non-negative")
    if int(cfg.anchor_break_edge_bars) < 0:
        raise ValueError("anchor_break_edge_bars must be non-negative")
    if int(cfg.max_joined_pivots) < 2:
        raise ValueError("max_joined_pivots must be at least 2")
    if str(cfg.step4_reduction_sequence).upper() not in {"A", "B", "C"}:
        raise ValueError("step4_reduction_sequence must be one of: A, B, C")
    if float(cfg.join_angle_tolerance) <= 0.0:
        raise ValueError("join_angle_tolerance must be positive")
    if int(cfg.shared_subset_min_pivots) < 2:
        raise ValueError("shared_subset_min_pivots must be at least 2")
    if int(cfg.partial_merge_shared_pivots_min) < 2:
        raise ValueError("partial_merge_shared_pivots_min must be at least 2")
    if float(cfg.partial_merge_angle_tolerance) <= 0.0:
        raise ValueError("partial_merge_angle_tolerance must be positive")
    if float(cfg.partial_merge_max_unmatched_pivot_distance_atr) <= 0.0:
        raise ValueError("partial_merge_max_unmatched_pivot_distance_atr must be positive")
    if int(cfg.co_terminal_start_tolerance_bars) < 0:
        raise ValueError("co_terminal_start_tolerance_bars must be non-negative")
    if int(cfg.co_terminal_end_tolerance_bars) < 0:
        raise ValueError("co_terminal_end_tolerance_bars must be non-negative")
    if not 0.0 <= float(cfg.co_terminal_min_overlap_ratio) <= 1.0:
        raise ValueError("co_terminal_min_overlap_ratio must be between 0 and 1")
    if float(cfg.co_terminal_max_mean_gap_atr) <= 0.0:
        raise ValueError("co_terminal_max_mean_gap_atr must be positive")
    if float(cfg.final_selection_max_distance_atr) <= 0.0:
        raise ValueError("final_selection_max_distance_atr must be positive")
    if float(cfg.max_active_price_distance_atr) <= 0.0:
        raise ValueError("max_active_price_distance_atr must be positive")
    if int(cfg.active_break_close_run) < 1:
        raise ValueError("active_break_close_run must be at least 1")


__all__ = [
    "PatternGeometryV3Config",
    "add_pattern_geometry_v3",
    "_geometry_v3_stage_tables",
]

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pandas import Series

LineSide = Literal["high", "low"]


@dataclass(frozen=True)
class PivotLineEngineConfig:
    """Shared no-lookahead pivot-line ranking engine.

    The engine builds candidate support/resistance lines from already-confirmed
    pivot pairs, then shifts the fitted line to an envelope so resistance lines
    touch pivot highs and support lines touch pivot lows. Candidate quality is
    based on touches, pivot violations, span, age, slope sanity, and current
    distance from price.
    """

    candidate_pivot_count: int
    ranked_line_count: int
    min_touch_count: int
    min_anchor_span_bars: int
    min_line_span_bars: int
    max_line_age_bars: int
    min_pivot_prominence: float
    touch_tolerance_atr_mult: float
    touch_tolerance_pct: float
    fit_touch_tolerance_pct: float
    min_respect_ratio: float
    min_span_age_ratio: float
    max_slope_pct_per_bar: float
    max_projection_distance_pct: float
    max_active_line_distance_pct: float | None = None
    use_envelope_fit: bool = True
    include_rolling_fit: bool = False
    use_weighted_fit: bool = False
    projection_bars_after_last_touch: int | None = 50
    max_raw_slope_multiplier: float = 1.15
    merge_slope_tolerance_pct: float = 0.0008
    merge_level_tolerance_pct: float = 0.006
    merge_overlap_ratio_min: float = 0.20
    merge_cluster_score_weight: float = 0.30


def ranked_pivot_lines(
    *,
    close: Series,
    atr: Series,
    bar_index: Series,
    event_price: Series,
    event_index: Series,
    event_prominence: Series,
    side: LineSide,
    cfg: PivotLineEngineConfig,
) -> dict[str, Series]:
    slots = max(int(cfg.ranked_line_count), 1)
    pivot_count = max(int(cfg.candidate_pivot_count), int(cfg.min_touch_count), 3)
    allowed = event_price.notna() & event_prominence.fillna(0.0).ge(float(cfg.min_pivot_prominence))
    recent = _recent_pivot_events(
        event_price.where(allowed),
        event_index.where(allowed),
        event_prominence.where(allowed),
        close.index,
        pivot_count,
    )
    candidates: list[dict[str, Series]] = []
    if cfg.include_rolling_fit:
        candidates.append(
            _rolling_fit_candidate(
                close=close,
                atr=atr,
                bar_index=bar_index,
                event_price=event_price.where(allowed),
                event_index=event_index.where(allowed),
                event_prominence=event_prominence.where(allowed),
                recent=recent,
                side=side,
                cfg=cfg,
            )
        )
    for newer in range(pivot_count - 1):
        for older in range(newer + 1, pivot_count):
            candidates.append(
                _candidate_from_pair(
                    close=close,
                    atr=atr,
                    bar_index=bar_index,
                    recent=recent,
                    newer=newer,
                    older=older,
                    side=side,
                    cfg=cfg,
                )
            )
    return _rank_candidates(candidates, close.index, slots, cfg)


def _rolling_fit_candidate(
    *,
    close: Series,
    atr: Series,
    bar_index: Series,
    event_price: Series,
    event_index: Series,
    event_prominence: Series,
    recent: dict[str, list[Series]],
    side: LineSide,
    cfg: PivotLineEngineConfig,
) -> dict[str, Series]:
    weight = event_price.notna().astype("float64") * (1.0 + _clip01(event_prominence.fillna(0.0) / 5.0))
    clean_x = event_index.fillna(0.0)
    clean_y = event_price.fillna(0.0)
    window = max(int(cfg.max_line_age_bars), int(cfg.min_line_span_bars) * 2, int(cfg.min_anchor_span_bars) * 4)
    min_periods = max(2, int(cfg.min_touch_count))
    sum_w = weight.rolling(window, min_periods=min_periods).sum()
    sum_x = (weight * clean_x).rolling(window, min_periods=min_periods).sum()
    sum_y = (weight * clean_y).rolling(window, min_periods=min_periods).sum()
    sum_x2 = (weight * clean_x * clean_x).rolling(window, min_periods=min_periods).sum()
    sum_xy = (weight * clean_x * clean_y).rolling(window, min_periods=min_periods).sum()
    denominator = sum_w * sum_x2 - sum_x * sum_x
    slope = _safe_div(sum_w * sum_xy - sum_x * sum_y, denominator).where(denominator.abs().gt(1e-9))
    latest_price = event_price.ffill()
    latest_index = event_index.ffill()
    latest_prominence = event_prominence.ffill()
    intercept = latest_price - slope * latest_index
    line = (intercept + slope * bar_index).clip(lower=0.0)
    quality = _line_quality_counts(
        recent=recent,
        line_slope=slope,
        line_intercept=intercept,
        atr=atr,
        close=close,
        side=side,
        cfg=cfg,
    )
    touch_count = quality["touch_count"]
    violation_count = quality["violation_count"]
    earliest_touch_index = quality["earliest_touch_index"]
    latest_touch_index = quality["latest_touch_index"]
    line_span = latest_touch_index - earliest_touch_index
    age = bar_index - latest_touch_index
    prominence_mean = quality["prominence_mean"]
    slope_pct = (slope.abs() / close.abs()).replace([np.inf, -np.inf], np.nan)
    distance_pct = ((line - close).abs() / close.abs()).replace([np.inf, -np.inf], np.nan)
    current_tolerance = _tolerance(atr, close, cfg)
    side_ok = line.le(close + current_tolerance) if side == "low" else line.ge(close - current_tolerance)
    projection_ok = distance_pct.le(float(cfg.max_projection_distance_pct))
    if cfg.max_active_line_distance_pct is not None:
        projection_ok = projection_ok & distance_pct.le(float(cfg.max_active_line_distance_pct))
    respect_ratio = _safe_div(touch_count, touch_count + violation_count).fillna(0.0)
    total_line_age = bar_index - earliest_touch_index
    span_age_ratio = _safe_div(line_span, total_line_age).fillna(0.0)
    projection_age_ok = pd.Series(True, index=close.index)
    if cfg.projection_bars_after_last_touch is not None:
        projection_age_ok = age.le(float(cfg.projection_bars_after_last_touch))
    valid = (
        line.notna()
        & line.gt(0.0)
        & latest_price.notna()
        & latest_index.notna()
        & touch_count.ge(float(cfg.min_touch_count))
        & line_span.ge(float(cfg.min_line_span_bars))
        & age.le(float(cfg.max_line_age_bars))
        & projection_age_ok
        & respect_ratio.ge(float(cfg.min_respect_ratio))
        & span_age_ratio.ge(float(cfg.min_span_age_ratio))
        & slope_pct.le(float(cfg.max_slope_pct_per_bar))
        & projection_ok
        & side_ok
    ).fillna(False)
    id_scale = float(max(len(close) + int(cfg.max_line_age_bars) + 10, 10_000))
    line_id = ((latest_index - line_span).fillna(latest_index) * id_scale + latest_index + 0.5).where(valid)
    touch_score = _clip01(touch_count / max(float(cfg.min_touch_count) + 2.0, 1.0))
    span_score = _clip01(line_span / max(float(cfg.min_line_span_bars) * 4.0, 1.0))
    projection_age_limit = float(cfg.projection_bars_after_last_touch or cfg.max_line_age_bars)
    age_score = _clip01(1.0 - age / max(min(float(cfg.max_line_age_bars), projection_age_limit), 1.0))
    prominence_score = _clip01(prominence_mean / max(float(cfg.min_pivot_prominence) * 3.0, 1.0))
    distance_score = _clip01(1.0 - distance_pct / max(float(cfg.max_projection_distance_pct), 1e-9))
    slope_score = _clip01(1.0 - slope_pct / max(float(cfg.max_slope_pct_per_bar), 1e-9))
    score = _clip01(
        0.32 * touch_score
        + 0.24 * respect_ratio
        + 0.15 * span_score
        + 0.11 * prominence_score
        + 0.08 * distance_score
        + 0.06 * age_score
        + 0.04 * slope_score
    ).where(valid)
    return {
        "line": line.where(valid),
        "slope": slope.where(valid),
        "score": score,
        "touch_count": touch_count.where(valid),
        "violation_count": violation_count.where(valid),
        "slope_pct": slope_pct.where(valid),
        "age": age.where(valid),
        "span": line_span.where(valid),
        "prominence": latest_prominence.where(valid),
        "prominence_min": latest_prominence.where(valid),
        "prominence_mean": prominence_mean.where(valid),
        "respect_ratio": respect_ratio.where(valid),
        "span_age_ratio": span_age_ratio.where(valid),
        "line_id": line_id,
        "intercept": intercept.where(valid),
        "anchor_old_index": earliest_touch_index.where(valid),
        "anchor_new_index": latest_index.where(valid),
        "touch_start_index": earliest_touch_index.where(valid),
        "touch_end_index": latest_touch_index.where(valid),
    }


def _recent_pivot_events(
    event_price: Series,
    event_index: Series,
    event_prominence: Series,
    index: pd.Index,
    count: int,
) -> dict[str, list[Series]]:
    price_events = event_price.dropna()
    index_events = event_index.where(event_price.notna()).dropna()
    prominence_events = event_prominence.where(event_price.notna()).dropna()
    return {
        "price": [price_events.shift(offset).reindex(index).ffill() for offset in range(count)],
        "index": [index_events.shift(offset).reindex(index).ffill() for offset in range(count)],
        "prominence": [prominence_events.shift(offset).reindex(index).ffill() for offset in range(count)],
    }


def _candidate_from_pair(
    *,
    close: Series,
    atr: Series,
    bar_index: Series,
    recent: dict[str, list[Series]],
    newer: int,
    older: int,
    side: LineSide,
    cfg: PivotLineEngineConfig,
) -> dict[str, Series]:
    p_new = recent["price"][newer]
    p_old = recent["price"][older]
    x_new = recent["index"][newer]
    x_old = recent["index"][older]
    prom_new = recent["prominence"][newer]
    prom_old = recent["prominence"][older]
    span = x_new - x_old
    prominence_min = pd.concat([prom_new, prom_old], axis=1).min(axis=1)
    anchor_valid = (
        p_new.notna()
        & p_old.notna()
        & x_new.notna()
        & x_old.notna()
        & span.ge(float(cfg.min_anchor_span_bars))
        & prominence_min.ge(float(cfg.min_pivot_prominence))
    )
    raw_slope = _safe_div(p_new - p_old, span).where(anchor_valid)
    raw_slope_pct = (raw_slope.abs() / p_new.abs()).replace([np.inf, -np.inf], np.nan)
    anchor_valid = anchor_valid & raw_slope_pct.le(float(cfg.max_slope_pct_per_bar) * float(cfg.max_raw_slope_multiplier))
    raw_slope = raw_slope.where(anchor_valid)
    raw_intercept = p_new - raw_slope * x_new
    raw_tolerance_pct = max(float(cfg.fit_touch_tolerance_pct), float(cfg.touch_tolerance_pct))

    fit = _weighted_fit_for_nearby_pivots(
        recent=recent,
        raw_slope=raw_slope,
        raw_intercept=raw_intercept,
        raw_tolerance_pct=raw_tolerance_pct,
        anchor_valid=anchor_valid,
        close_index=close.index,
    )
    line_slope = raw_slope
    if cfg.use_weighted_fit:
        line_slope = fit["slope"].where(fit["touch_count"].ge(float(cfg.min_touch_count)), raw_slope)
    line_intercept = raw_intercept
    if cfg.use_envelope_fit:
        line_intercept = _envelope_intercept(
            recent=recent,
            line_slope=line_slope,
            raw_slope=raw_slope,
            raw_intercept=raw_intercept,
            side=side,
            cfg=cfg,
            close_index=close.index,
        ).where(fit["touch_count"].ge(float(cfg.min_touch_count)), raw_intercept)

    line = (line_intercept + line_slope * bar_index).clip(lower=0.0)
    quality = _line_quality_counts(
        recent=recent,
        line_slope=line_slope,
        line_intercept=line_intercept,
        atr=atr,
        close=close,
        side=side,
        cfg=cfg,
    )
    touch_count = quality["touch_count"]
    violation_count = quality["violation_count"]
    earliest_touch_index = quality["earliest_touch_index"]
    latest_touch_index = quality["latest_touch_index"]
    line_span = latest_touch_index - earliest_touch_index
    age = bar_index - latest_touch_index
    prominence_mean = quality["prominence_mean"]
    slope_pct = (line_slope.abs() / close.abs()).replace([np.inf, -np.inf], np.nan)
    distance_pct = ((line - close).abs() / close.abs()).replace([np.inf, -np.inf], np.nan)
    current_tolerance = _tolerance(atr, close, cfg)
    side_ok = line.le(close + current_tolerance) if side == "low" else line.ge(close - current_tolerance)
    projection_ok = distance_pct.le(float(cfg.max_projection_distance_pct))
    if cfg.max_active_line_distance_pct is not None:
        projection_ok = projection_ok & distance_pct.le(float(cfg.max_active_line_distance_pct))
    respect_ratio = _safe_div(touch_count, touch_count + violation_count).fillna(0.0)
    total_line_age = bar_index - earliest_touch_index
    span_age_ratio = _safe_div(line_span, total_line_age).fillna(0.0)
    projection_age_ok = pd.Series(True, index=close.index)
    if cfg.projection_bars_after_last_touch is not None:
        projection_age_ok = age.le(float(cfg.projection_bars_after_last_touch))
    valid = (
        line.notna()
        & line.gt(0.0)
        & touch_count.ge(float(cfg.min_touch_count))
        & line_span.ge(float(cfg.min_line_span_bars))
        & age.le(float(cfg.max_line_age_bars))
        & projection_age_ok
        & respect_ratio.ge(float(cfg.min_respect_ratio))
        & span_age_ratio.ge(float(cfg.min_span_age_ratio))
        & slope_pct.le(float(cfg.max_slope_pct_per_bar))
        & projection_ok
        & side_ok
    ).fillna(False)

    id_scale = float(max(len(close) + int(cfg.max_line_age_bars) + 10, 10_000))
    line_id = (x_old * id_scale + x_new).where(valid)
    touch_score = _clip01(touch_count / max(float(cfg.min_touch_count) + 2.0, 1.0))
    span_score = _clip01(line_span / max(float(cfg.min_line_span_bars) * 4.0, 1.0))
    projection_age_limit = float(cfg.projection_bars_after_last_touch or cfg.max_line_age_bars)
    age_score = _clip01(1.0 - age / max(min(float(cfg.max_line_age_bars), projection_age_limit), 1.0))
    prominence_score = _clip01(prominence_mean / max(float(cfg.min_pivot_prominence) * 3.0, 1.0))
    distance_score = _clip01(1.0 - distance_pct / max(float(cfg.max_projection_distance_pct), 1e-9))
    slope_score = _clip01(1.0 - slope_pct / max(float(cfg.max_slope_pct_per_bar), 1e-9))
    score = _clip01(
        0.30 * touch_score
        + 0.22 * respect_ratio
        + 0.16 * span_score
        + 0.12 * prominence_score
        + 0.08 * distance_score
        + 0.07 * age_score
        + 0.05 * slope_score
    ).where(valid)
    return {
        "line": line.where(valid),
        "slope": line_slope.where(valid),
        "score": score,
        "touch_count": touch_count.where(valid),
        "violation_count": violation_count.where(valid),
        "slope_pct": slope_pct.where(valid),
        "age": age.where(valid),
        "span": line_span.where(valid),
        "prominence": prominence_min.where(valid),
        "prominence_min": prominence_min.where(valid),
        "prominence_mean": prominence_mean.where(valid),
        "respect_ratio": respect_ratio.where(valid),
        "span_age_ratio": span_age_ratio.where(valid),
        "line_id": line_id,
        "intercept": line_intercept.where(valid),
        "anchor_old_index": x_old.where(valid),
        "anchor_new_index": x_new.where(valid),
        "touch_start_index": earliest_touch_index.where(valid),
        "touch_end_index": latest_touch_index.where(valid),
    }


def _weighted_fit_for_nearby_pivots(
    *,
    recent: dict[str, list[Series]],
    raw_slope: Series,
    raw_intercept: Series,
    raw_tolerance_pct: float,
    anchor_valid: Series,
    close_index: pd.Index,
) -> dict[str, Series]:
    zero = pd.Series(0.0, index=close_index, dtype="float64")
    sum_w = zero.copy()
    sum_x = zero.copy()
    sum_y = zero.copy()
    sum_x2 = zero.copy()
    sum_xy = zero.copy()
    touch_count = zero.copy()
    for price, pivot_index, prominence in zip(recent["price"], recent["index"], recent["prominence"], strict=False):
        raw_level = raw_intercept + raw_slope * pivot_index
        distance_pct = ((price - raw_level).abs() / price.abs().replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        touch = anchor_valid & price.notna() & pivot_index.notna() & distance_pct.le(raw_tolerance_pct)
        touch_f = touch.fillna(False).astype("float64")
        weight = touch_f * (1.0 + _clip01(prominence.fillna(0.0) / 5.0))
        clean_x = pivot_index.fillna(0.0)
        clean_y = price.fillna(0.0)
        sum_w = sum_w + weight
        sum_x = sum_x + weight * clean_x
        sum_y = sum_y + weight * clean_y
        sum_x2 = sum_x2 + weight * clean_x * clean_x
        sum_xy = sum_xy + weight * clean_x * clean_y
        touch_count = touch_count + touch_f
    variance = sum_x2 - _safe_div(sum_x * sum_x, sum_w)
    covariance = sum_xy - _safe_div(sum_x * sum_y, sum_w)
    slope = _safe_div(covariance, variance).where(variance.abs().gt(1e-9), raw_slope)
    return {"slope": slope, "touch_count": touch_count}


def _envelope_intercept(
    *,
    recent: dict[str, list[Series]],
    line_slope: Series,
    raw_slope: Series,
    raw_intercept: Series,
    side: LineSide,
    cfg: PivotLineEngineConfig,
    close_index: pd.Index,
) -> Series:
    if side == "high":
        envelope = pd.Series(-np.inf, index=close_index, dtype="float64")
    else:
        envelope = pd.Series(np.inf, index=close_index, dtype="float64")
    for price, pivot_index, _prominence in zip(recent["price"], recent["index"], recent["prominence"], strict=False):
        raw_level = raw_intercept + raw_slope * pivot_index
        distance_pct = ((price - raw_level).abs() / price.abs().replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        touch = price.notna() & pivot_index.notna() & distance_pct.le(max(float(cfg.fit_touch_tolerance_pct), float(cfg.touch_tolerance_pct)))
        candidate_intercept = (price - line_slope * pivot_index).where(touch)
        if side == "high":
            envelope = _series_max(envelope.replace(-np.inf, np.nan), candidate_intercept)
        else:
            envelope = _series_min(envelope.replace(np.inf, np.nan), candidate_intercept)
    return envelope.replace([np.inf, -np.inf], np.nan)


def _line_quality_counts(
    *,
    recent: dict[str, list[Series]],
    line_slope: Series,
    line_intercept: Series,
    atr: Series,
    close: Series,
    side: LineSide,
    cfg: PivotLineEngineConfig,
) -> dict[str, Series]:
    zero = pd.Series(0.0, index=close.index, dtype="float64")
    touch_count = zero.copy()
    violation_count = zero.copy()
    sum_w = zero.copy()
    sum_prominence = zero.copy()
    latest_touch_index = pd.Series(np.nan, index=close.index, dtype="float64")
    earliest_touch_index = pd.Series(np.nan, index=close.index, dtype="float64")
    for price, pivot_index, prominence in zip(recent["price"], recent["index"], recent["prominence"], strict=False):
        level_at_pivot = line_intercept + line_slope * pivot_index
        tolerance = _tolerance(atr, price, cfg)
        residual = price - level_at_pivot
        touch = price.notna() & pivot_index.notna() & residual.abs().le(tolerance)
        violation = (residual.lt(-tolerance) if side == "low" else residual.gt(tolerance)).fillna(False)
        touch_f = touch.fillna(False).astype("float64")
        weight = touch_f * (1.0 + _clip01(prominence.fillna(0.0) / 5.0))
        touch_count = touch_count + touch_f
        violation_count = violation_count + violation.astype("float64")
        sum_w = sum_w + weight
        sum_prominence = sum_prominence + weight * prominence.fillna(0.0)
        touch_index = pivot_index.where(touch)
        latest_touch_index = _series_max(latest_touch_index, touch_index)
        earliest_touch_index = _series_min(earliest_touch_index, touch_index)
    return {
        "touch_count": touch_count,
        "violation_count": violation_count,
        "prominence_mean": _safe_div(sum_prominence, sum_w),
        "latest_touch_index": latest_touch_index,
        "earliest_touch_index": earliest_touch_index,
    }


def _rank_candidates(
    candidates: list[dict[str, Series]],
    index: pd.Index,
    slots: int,
    cfg: PivotLineEngineConfig,
) -> dict[str, Series]:
    if not candidates:
        return _empty_ranked(index, slots)
    score_frame = pd.concat([candidate["score"] for candidate in candidates], axis=1)
    score_values = score_frame.astype("float64").to_numpy()
    score_values = np.where(np.isfinite(score_values), score_values, -np.inf)
    if score_values.shape[1] == 0:
        return _empty_ranked(index, slots)
    order = np.argsort(-score_values, axis=1)
    sorted_scores = np.take_along_axis(score_values, order, axis=1)
    row_index = np.arange(len(index))
    metrics = (
        "line",
        "slope",
        "touch_count",
        "violation_count",
        "slope_pct",
        "age",
        "span",
        "prominence",
        "prominence_min",
        "prominence_mean",
        "respect_ratio",
        "span_age_ratio",
        "line_id",
        "intercept",
        "anchor_old_index",
        "anchor_new_index",
        "touch_start_index",
        "touch_end_index",
    )
    metric_frames = {
        metric: pd.concat([candidate[metric] for candidate in candidates], axis=1).astype("float64").to_numpy()
        for metric in metrics
    }
    ranked_scores = _cluster_adjusted_scores(score_values, metric_frames, cfg)
    remaining_scores = ranked_scores.copy()
    out: dict[str, Series] = {}
    for rank in range(slots):
        if rank >= score_values.shape[1]:
            for metric in ("score", *metrics):
                out[f"{metric}_{rank}"] = pd.Series(np.nan, index=index, dtype="float64")
            out[f"valid_{rank}"] = pd.Series(False, index=index)
            continue
        chosen = np.argmax(remaining_scores, axis=1)
        chosen_score = remaining_scores[row_index, chosen]
        valid = np.isfinite(chosen_score) & (chosen_score > -np.inf)
        cluster_mask = _similar_candidates_for_rows(metric_frames, chosen, valid, cfg)
        out[f"score_{rank}"] = pd.Series(np.where(valid, chosen_score, np.nan), index=index, dtype="float64")
        out[f"valid_{rank}"] = pd.Series(valid, index=index)
        for metric, values in metric_frames.items():
            selected = _merged_cluster_metric(metric, values, score_values, cluster_mask, row_index, chosen, metric_frames)
            out[f"{metric}_{rank}"] = pd.Series(np.where(valid, selected, np.nan), index=index, dtype="float64")
        _suppress_similar_candidates(remaining_scores, cluster_mask)
    return out


def _cluster_adjusted_scores(
    score_values: np.ndarray,
    metric_frames: dict[str, np.ndarray],
    cfg: PivotLineEngineConfig,
) -> np.ndarray:
    candidate_count = score_values.shape[1]
    if candidate_count <= 1:
        return score_values
    finite = np.isfinite(score_values) & (score_values > -np.inf)
    line_values = metric_frames["line"]
    slope_values = metric_frames["slope"]
    anchor_old = metric_frames["anchor_old_index"]
    anchor_new = metric_frames["anchor_new_index"]
    member_count = np.zeros_like(score_values, dtype="float64")
    support_sum = np.zeros_like(score_values, dtype="float64")
    for left in range(candidate_count):
        for right in range(left, candidate_count):
            similar = _candidate_similarity(
                line_values[:, left],
                line_values[:, right],
                slope_values[:, left],
                slope_values[:, right],
                anchor_old[:, left],
                anchor_new[:, left],
                anchor_old[:, right],
                anchor_new[:, right],
                cfg,
            )
            if not np.any(similar):
                continue
            left_supported = similar & finite[:, right] & finite[:, left]
            right_supported = similar & finite[:, left] & finite[:, right]
            if np.any(left_supported):
                member_count[left_supported, left] += 1.0
                support_sum[left_supported, left] += np.where(np.isfinite(score_values[left_supported, right]), score_values[left_supported, right], 0.0)
            if right != left and np.any(right_supported):
                member_count[right_supported, right] += 1.0
                support_sum[right_supported, right] += np.where(np.isfinite(score_values[right_supported, left]), score_values[right_supported, left], 0.0)
    support_mean = np.divide(support_sum, member_count, out=np.zeros_like(support_sum), where=member_count > 0.0)
    cluster_bonus = np.clip((member_count - 1.0) / 4.0, 0.0, 1.0)
    adjusted = (
        score_values * (1.0 - float(cfg.merge_cluster_score_weight))
        + support_mean * float(cfg.merge_cluster_score_weight) * 0.65
        + cluster_bonus * float(cfg.merge_cluster_score_weight) * 0.35
    )
    adjusted = np.where(finite, np.clip(adjusted, 0.0, 1.0), -np.inf)
    return adjusted


def _similar_candidates_for_rows(
    metric_frames: dict[str, np.ndarray],
    chosen: np.ndarray,
    valid: np.ndarray,
    cfg: PivotLineEngineConfig,
) -> np.ndarray:
    row_index = np.arange(metric_frames["line"].shape[0])
    selected_line = metric_frames["line"][row_index, chosen]
    selected_slope = metric_frames["slope"][row_index, chosen]
    selected_old = metric_frames["anchor_old_index"][row_index, chosen]
    selected_new = metric_frames["anchor_new_index"][row_index, chosen]
    cluster_mask = np.zeros_like(metric_frames["line"], dtype=bool)
    for candidate in range(metric_frames["line"].shape[1]):
        similar = _candidate_similarity(
            selected_line,
            metric_frames["line"][:, candidate],
            selected_slope,
            metric_frames["slope"][:, candidate],
            selected_old,
            selected_new,
            metric_frames["anchor_old_index"][:, candidate],
            metric_frames["anchor_new_index"][:, candidate],
            cfg,
        )
        cluster_mask[:, candidate] = valid & similar
    return cluster_mask


def _suppress_similar_candidates(remaining_scores: np.ndarray, cluster_mask: np.ndarray) -> None:
    remaining_scores[cluster_mask] = -np.inf


def _merged_cluster_metric(
    metric: str,
    values: np.ndarray,
    score_values: np.ndarray,
    cluster_mask: np.ndarray,
    row_index: np.ndarray,
    chosen: np.ndarray,
    metric_frames: dict[str, np.ndarray],
) -> np.ndarray:
    selected = values[row_index, chosen]
    if metric in {"line", "line_id", "slope", "slope_pct", "intercept"}:
        # Keep the plotted/value line tied to a real pivot-to-pivot candidate.
        # Cluster support raises score, but it must not create a synthetic line
        # that drifts away from the candle-body anchors that produced it.
        return selected
    weighted_metrics = {
        "prominence",
        "prominence_min",
        "prominence_mean",
        "respect_ratio",
        "span_age_ratio",
    }
    if metric in weighted_metrics:
        merged = _weighted_cluster_average(values, score_values, cluster_mask)
        return np.where(np.isfinite(merged), merged, selected)
    if metric in {"touch_count", "violation_count", "span"}:
        masked = np.where(cluster_mask & np.isfinite(values), values, np.nan)
        merged = _nanmax_2d(masked)
        return np.where(np.isfinite(merged), merged, selected)
    if metric == "age":
        masked = np.where(cluster_mask & np.isfinite(values), values, np.nan)
        merged = _nanmin_2d(masked)
        return np.where(np.isfinite(merged), merged, selected)
    if metric in {"anchor_old_index", "touch_start_index"}:
        masked = np.where(cluster_mask & np.isfinite(values), values, np.nan)
        merged = _nanmin_2d(masked)
        return np.where(np.isfinite(merged), merged, selected)
    if metric in {"anchor_new_index", "touch_end_index"}:
        masked = np.where(cluster_mask & np.isfinite(values), values, np.nan)
        merged = _nanmax_2d(masked)
        return np.where(np.isfinite(merged), merged, selected)
    return selected


def _candidate_similarity(
    left_line: np.ndarray,
    right_line: np.ndarray,
    left_slope: np.ndarray,
    right_slope: np.ndarray,
    left_old: np.ndarray,
    left_new: np.ndarray,
    right_old: np.ndarray,
    right_new: np.ndarray,
    cfg: PivotLineEngineConfig,
) -> np.ndarray:
    level_reference = (np.abs(left_line) + np.abs(right_line)) / 2.0
    level_distance = np.abs(left_line - right_line) / np.where(level_reference > 0.0, level_reference, np.nan)
    slope_reference = level_reference
    slope_distance = np.abs(left_slope - right_slope) / np.where(slope_reference > 0.0, slope_reference, np.nan)
    shared_anchor = (
        _float_equal(left_old, right_old)
        | _float_equal(left_old, right_new)
        | _float_equal(left_new, right_old)
        | _float_equal(left_new, right_new)
    )
    overlap_start = np.maximum(np.minimum(left_old, left_new), np.minimum(right_old, right_new))
    overlap_end = np.minimum(np.maximum(left_old, left_new), np.maximum(right_old, right_new))
    left_span = np.abs(left_new - left_old)
    right_span = np.abs(right_new - right_old)
    overlap = np.maximum(overlap_end - overlap_start, 0.0)
    span_reference = np.maximum(np.minimum(left_span, right_span), 1.0)
    overlaps_enough = np.divide(overlap, span_reference, out=np.zeros_like(overlap), where=span_reference > 0.0) >= float(cfg.merge_overlap_ratio_min)
    angle_ok = slope_distance <= float(cfg.merge_slope_tolerance_pct)
    close_level = level_distance <= float(cfg.merge_level_tolerance_pct)
    shared_angle = shared_anchor & angle_ok & (level_distance <= float(cfg.merge_level_tolerance_pct) * 2.0)
    similar = angle_ok & ((close_level & overlaps_enough) | shared_angle)
    return np.isfinite(level_distance) & np.isfinite(slope_distance) & similar


def _float_equal(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.isfinite(left) & np.isfinite(right) & (np.abs(left - right) < 0.5)


def _weighted_cluster_average(values: np.ndarray, score_values: np.ndarray, cluster_mask: np.ndarray) -> np.ndarray:
    weights = np.where(cluster_mask & np.isfinite(values) & np.isfinite(score_values), score_values, 0.0)
    denominator = weights.sum(axis=1)
    numerator = np.where(np.isfinite(values), values, 0.0)
    return np.divide(
        (numerator * weights).sum(axis=1),
        denominator,
        out=np.full(values.shape[0], np.nan, dtype="float64"),
        where=denominator > 0.0,
    )


def _empty_ranked(index: pd.Index, slots: int) -> dict[str, Series]:
    metrics = (
        "line",
        "slope",
        "score",
        "touch_count",
        "violation_count",
        "slope_pct",
        "age",
        "span",
        "prominence",
        "prominence_min",
        "prominence_mean",
        "respect_ratio",
        "span_age_ratio",
        "line_id",
        "intercept",
        "anchor_old_index",
        "anchor_new_index",
        "touch_start_index",
        "touch_end_index",
    )
    out: dict[str, Series] = {}
    for rank in range(slots):
        for metric in metrics:
            out[f"{metric}_{rank}"] = pd.Series(np.nan, index=index, dtype="float64")
        out[f"valid_{rank}"] = pd.Series(False, index=index)
    return out


def _tolerance(atr: Series, price: Series, cfg: PivotLineEngineConfig) -> Series:
    return pd.concat(
        [
            atr * float(cfg.touch_tolerance_atr_mult),
            price.abs() * float(cfg.touch_tolerance_pct),
        ],
        axis=1,
    ).max(axis=1)


def _safe_div(numerator: Series, denominator: Series) -> Series:
    return numerator / denominator.replace(0.0, np.nan)


def _series_max(left: Series, right: Series) -> Series:
    return pd.Series(np.fmax(left.to_numpy(dtype="float64"), right.to_numpy(dtype="float64")), index=left.index)


def _series_min(left: Series, right: Series) -> Series:
    return pd.Series(np.fmin(left.to_numpy(dtype="float64"), right.to_numpy(dtype="float64")), index=left.index)


def _nanmax_2d(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    filled = np.where(finite, values, -np.inf)
    result = filled.max(axis=1)
    return np.where(np.isfinite(result), result, np.nan)


def _nanmin_2d(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    filled = np.where(finite, values, np.inf)
    result = filled.min(axis=1)
    return np.where(np.isfinite(result), result, np.nan)


def _clip01(series: Series) -> Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0).fillna(0.0)

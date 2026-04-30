from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

try:
    from .complex_pivot_structure import PivotStructureConfig, add_pivot_structure
    from .complex_line_engine import PivotLineEngineConfig, ranked_pivot_lines
except Exception:  # pragma: no cover - optional fallback for standalone notebooks
    PivotStructureConfig = None  # type: ignore[assignment]
    add_pivot_structure = None  # type: ignore[assignment]
    from complex_line_engine import PivotLineEngineConfig, ranked_pivot_lines  # type: ignore[no-redef]

MissingPivotMode = Literal["raise", "compute", "skip"]
TrendlineSide = Literal["high", "low"]


@dataclass(frozen=True)
class StructuralTrendlineConfig:
    """Sparse, no-lookahead structural trendlines from confirmed major pivots.

    This module is deliberately separate from ``complex_trendline_projection``.
    The local ``tl`` indicator is for tactical nearby channels. This ``stl``
    indicator is for larger human-style structure: descending resistance,
    ascending support, wedges, and triangle compression.

    Strategy-facing columns are only emitted after the relevant pivots are
    confirmed. Plot columns do not backfill historical anchors, so they remain
    safe to use in Freqtrade strategies.
    """

    strength: int = 21
    strengths: Sequence[int] = (13, 21)
    horizons: Sequence[int] = (1, 3, 6, 12, 24, 48)
    pivot_prefix: str = "pa"
    output_prefix: str = "stl"
    missing_pivot_mode: MissingPivotMode = "raise"
    pivot_config: object | None = None

    ranked_line_count: int = 3
    candidate_pivot_count: int = 10
    min_touch_count: int = 3
    min_anchor_span_bars: int = 48
    min_line_span_bars: int = 120
    max_line_age_bars: int = 700
    min_pivot_prominence_atr: float = 2.00
    touch_tolerance_atr_mult: float = 0.85
    touch_tolerance_pct: float = 0.004
    min_respect_ratio: float = 0.95
    min_span_age_ratio: float = 0.18
    max_slope_pct_per_bar: float = 0.0035
    max_projection_distance_pct: float = 0.25
    max_projection_bars_after_last_touch: int = 50
    breakout_buffer_pct: float = 0.003
    compression_window: int = 240
    compression_min_periods: int = 48
    score_window: int = 96
    plot_break_on_line_change_pct: float = 0.025
    plot_min_segment_bars: int = 12
    plot_max_distance_pct: float = 0.18


def add_structural_trendlines(
    dataframe: DataFrame,
    config: StructuralTrendlineConfig | None = None,
    *,
    strength: int | None = None,
    strengths: Sequence[int] | None = None,
    horizons: Sequence[int] | None = None,
    pivot_prefix: str | None = None,
    output_prefix: str | None = None,
    missing_pivot_mode: MissingPivotMode | None = None,
    pivot_config: object | None = None,
    ranked_line_count: int | None = None,
    candidate_pivot_count: int | None = None,
    min_touch_count: int | None = None,
    min_anchor_span_bars: int | None = None,
    min_line_span_bars: int | None = None,
    max_line_age_bars: int | None = None,
    min_pivot_prominence_atr: float | None = None,
    touch_tolerance_atr_mult: float | None = None,
    touch_tolerance_pct: float | None = None,
    min_respect_ratio: float | None = None,
    min_span_age_ratio: float | None = None,
    max_slope_pct_per_bar: float | None = None,
    max_projection_distance_pct: float | None = None,
    max_projection_bars_after_last_touch: int | None = None,
    breakout_buffer_pct: float | None = None,
    compression_window: int | None = None,
    compression_min_periods: int | None = None,
    score_window: int | None = None,
    plot_break_on_line_change_pct: float | None = None,
    plot_min_segment_bars: int | None = None,
    plot_max_distance_pct: float | None = None,
) -> DataFrame:
    """Append sparse structural trendline columns.

    The implementation loops over candidate pivot-pairs, but each candidate is
    evaluated vectorized over the full dataframe. There is no candle-by-candle
    row loop and no use of unconfirmed future pivots.
    """

    cfg = _resolve_config(
        config,
        strength=strength,
        strengths=strengths,
        horizons=horizons,
        pivot_prefix=pivot_prefix,
        output_prefix=output_prefix,
        missing_pivot_mode=missing_pivot_mode,
        pivot_config=pivot_config,
        ranked_line_count=ranked_line_count,
        candidate_pivot_count=candidate_pivot_count,
        min_touch_count=min_touch_count,
        min_anchor_span_bars=min_anchor_span_bars,
        min_line_span_bars=min_line_span_bars,
        max_line_age_bars=max_line_age_bars,
        min_pivot_prominence_atr=min_pivot_prominence_atr,
        touch_tolerance_atr_mult=touch_tolerance_atr_mult,
        touch_tolerance_pct=touch_tolerance_pct,
        min_respect_ratio=min_respect_ratio,
        min_span_age_ratio=min_span_age_ratio,
        max_slope_pct_per_bar=max_slope_pct_per_bar,
        max_projection_distance_pct=max_projection_distance_pct,
        max_projection_bars_after_last_touch=max_projection_bars_after_last_touch,
        breakout_buffer_pct=breakout_buffer_pct,
        compression_window=compression_window,
        compression_min_periods=compression_min_periods,
        score_window=score_window,
        plot_break_on_line_change_pct=plot_break_on_line_change_pct,
        plot_min_segment_bars=plot_min_segment_bars,
        plot_max_distance_pct=plot_max_distance_pct,
    )
    _validate_config(cfg)
    _validate_dataframe(dataframe)

    frame = _ensure_pivot_columns(dataframe.copy(), cfg)
    close = _num(frame, "close").replace(0.0, np.nan)
    high = _num(frame, "high")
    low = _num(frame, "low")
    open_ = _num(frame, "open")
    atr = _optional_num(frame, f"{cfg.pivot_prefix}_atr", _atr(frame, 14))
    bar_index = _optional_num(frame, f"{cfg.pivot_prefix}_bar_index", pd.Series(np.arange(len(frame), dtype="float64"), index=frame.index))
    zone_width = pd.concat([atr * float(cfg.touch_tolerance_atr_mult), close * float(cfg.touch_tolerance_pct)], axis=1).max(axis=1)

    p = cfg.output_prefix
    strengths_list = _sorted_strengths(cfg)
    horizons_list = _sorted_unique_ints(cfg.horizons)
    new_cols: dict[str, Series] = {f"{p}_zone_width": zone_width}
    for current_strength in strengths_list:
        new_cols.update(
            _columns_for_strength(
                frame=frame,
                close=close,
                high=high,
                low=low,
                open_=open_,
                atr=atr,
                bar_index=bar_index,
                zone_width=zone_width,
                strength=current_strength,
                horizons=horizons_list,
                cfg=cfg,
            )
        )

    selected = int(cfg.strength)
    if selected not in strengths_list:
        selected = strengths_list[0]
    new_cols.update(_active_alias_columns(new_cols, p, selected, horizons_list, int(cfg.ranked_line_count)))

    existing = [col for col in frame.columns if str(col).startswith(f"{p}_")]
    base = frame.drop(columns=existing).copy() if existing else frame.copy()
    return pd.concat([base, pd.DataFrame(new_cols, index=frame.index)], axis=1)


def _columns_for_strength(
    *,
    frame: DataFrame,
    close: Series,
    high: Series,
    low: Series,
    open_: Series,
    atr: Series,
    bar_index: Series,
    zone_width: Series,
    strength: int,
    horizons: list[int],
    cfg: StructuralTrendlineConfig,
) -> dict[str, Series]:
    p = cfg.output_prefix
    s = int(strength)
    resistance = _ranked_structural_lines(frame, close, atr, bar_index, strength=s, side="high", cfg=cfg)
    support = _ranked_structural_lines(frame, close, atr, bar_index, strength=s, side="low", cfg=cfg)

    res_line = resistance["line_0"]
    sup_line = support["line_0"]
    res_slope = resistance["slope_0"]
    sup_slope = support["slope_0"]
    res_valid = resistance["valid_0"]
    sup_valid = support["valid_0"]

    channel_width_pct = ((res_line - sup_line) / close).replace([np.inf, -np.inf], np.nan)
    channel_valid = (res_valid & sup_valid & res_line.gt(sup_line) & channel_width_pct.gt(0.0)).fillna(False)
    channel_mid = ((res_line + sup_line) / 2.0).where(channel_valid)
    channel_compression = channel_width_pct / channel_width_pct.rolling(
        int(cfg.compression_window),
        min_periods=int(cfg.compression_min_periods),
    ).median()
    descending_resistance = (res_valid & res_slope.lt(0.0)).fillna(False)
    ascending_support = (sup_valid & sup_slope.gt(0.0)).fillna(False)
    converging = channel_valid & descending_resistance & ascending_support & (res_slope < sup_slope)
    wedge_compression = _clip01(1.0 - channel_compression.fillna(1.0)).where(channel_valid, 0.0)
    triangle_score = _clip01(
        0.45 * converging.astype("float64")
        + 0.35 * wedge_compression
        + 0.20 * _clip01((resistance["span_0"].fillna(0.0) + support["span_0"].fillna(0.0)) / max(float(cfg.min_line_span_bars) * 4.0, 1.0))
    )

    resistance_touch = res_valid & _touches_line(high, low, res_line, zone_width)
    support_touch = sup_valid & _touches_line(high, low, sup_line, zone_width)
    resistance_reject = resistance_touch & close.le(res_line) & close.lt(open_)
    support_reclaim = support_touch & close.ge(sup_line) & close.gt(open_)
    resistance_violation = res_valid & close.gt(res_line + zone_width)
    support_violation = sup_valid & close.lt(sup_line - zone_width)

    trend_bias = pd.Series(
        np.select(
            [channel_valid & res_slope.gt(0.0) & sup_slope.gt(0.0), channel_valid & res_slope.lt(0.0) & sup_slope.lt(0.0)],
            [1.0, -1.0],
            default=0.0,
        ),
        index=frame.index,
        dtype="float64",
    )

    cols: dict[str, Series] = {
        f"{p}_resistance_line_{s}": res_line,
        f"{p}_support_line_{s}": sup_line,
        f"{p}_resistance_plot_{s}": _plot_safe_line(
            res_line.where(res_valid),
            cfg.plot_break_on_line_change_pct,
            cfg.plot_min_segment_bars,
            close,
            cfg.plot_max_distance_pct,
            resistance["line_id_0"],
        ),
        f"{p}_support_plot_{s}": _plot_safe_line(
            sup_line.where(sup_valid),
            cfg.plot_break_on_line_change_pct,
            cfg.plot_min_segment_bars,
            close,
            cfg.plot_max_distance_pct,
            support["line_id_0"],
        ),
        f"{p}_resistance_slope_{s}": res_slope,
        f"{p}_support_slope_{s}": sup_slope,
        f"{p}_resistance_slope_pct_{s}": resistance["slope_pct_0"],
        f"{p}_support_slope_pct_{s}": support["slope_pct_0"],
        f"{p}_resistance_confirmed_{s}": res_valid,
        f"{p}_support_confirmed_{s}": sup_valid,
        f"{p}_descending_resistance_{s}": descending_resistance,
        f"{p}_ascending_support_{s}": ascending_support,
        f"{p}_resistance_touch_count_{s}": resistance["touch_count_0"],
        f"{p}_support_touch_count_{s}": support["touch_count_0"],
        f"{p}_resistance_score_{s}": resistance["score_0"],
        f"{p}_support_score_{s}": support["score_0"],
        f"{p}_resistance_age_{s}": resistance["age_0"],
        f"{p}_support_age_{s}": support["age_0"],
        f"{p}_resistance_span_{s}": resistance["span_0"],
        f"{p}_support_span_{s}": support["span_0"],
        f"{p}_channel_valid_{s}": channel_valid,
        f"{p}_channel_mid_{s}": channel_mid,
        f"{p}_channel_mid_plot_{s}": _plot_safe_line(
            channel_mid.where(channel_valid),
            cfg.plot_break_on_line_change_pct,
            cfg.plot_min_segment_bars,
            close,
            cfg.plot_max_distance_pct,
            _combined_identity(resistance["line_id_0"], support["line_id_0"]),
        ),
        f"{p}_channel_width_pct_{s}": channel_width_pct.where(channel_valid),
        f"{p}_channel_compression_{s}": channel_compression.where(channel_valid),
        f"{p}_wedge_compression_{s}": wedge_compression,
        f"{p}_triangle_score_{s}": triangle_score,
        f"{p}_trend_bias_{s}": trend_bias,
        f"{p}_resistance_touch_{s}": resistance_touch,
        f"{p}_support_touch_{s}": support_touch,
        f"{p}_resistance_reject_{s}": resistance_reject,
        f"{p}_support_reclaim_{s}": support_reclaim,
        f"{p}_resistance_violation_{s}": resistance_violation,
        f"{p}_support_violation_{s}": support_violation,
    }

    for rank in range(int(cfg.ranked_line_count)):
        cols.update(
            {
                f"{p}_resistance_line_rank{rank}_{s}": resistance[f"line_{rank}"],
                f"{p}_support_line_rank{rank}_{s}": support[f"line_{rank}"],
                f"{p}_resistance_line_id_rank{rank}_{s}": resistance[f"line_id_{rank}"],
                f"{p}_support_line_id_rank{rank}_{s}": support[f"line_id_{rank}"],
                f"{p}_resistance_slope_rank{rank}_{s}": resistance[f"slope_{rank}"],
                f"{p}_support_slope_rank{rank}_{s}": support[f"slope_{rank}"],
                f"{p}_resistance_slope_pct_rank{rank}_{s}": resistance[f"slope_pct_{rank}"],
                f"{p}_support_slope_pct_rank{rank}_{s}": support[f"slope_pct_{rank}"],
                f"{p}_resistance_plot_rank{rank}_{s}": _plot_safe_line(
                    resistance[f"line_{rank}"].where(resistance[f"valid_{rank}"]),
                    cfg.plot_break_on_line_change_pct,
                    cfg.plot_min_segment_bars,
                    close,
                    cfg.plot_max_distance_pct,
                    resistance[f"line_id_{rank}"],
                ),
                f"{p}_support_plot_rank{rank}_{s}": _plot_safe_line(
                    support[f"line_{rank}"].where(support[f"valid_{rank}"]),
                    cfg.plot_break_on_line_change_pct,
                    cfg.plot_min_segment_bars,
                    close,
                    cfg.plot_max_distance_pct,
                    support[f"line_id_{rank}"],
                ),
                f"{p}_resistance_score_rank{rank}_{s}": resistance[f"score_{rank}"],
                f"{p}_support_score_rank{rank}_{s}": support[f"score_{rank}"],
                f"{p}_resistance_touch_count_rank{rank}_{s}": resistance[f"touch_count_{rank}"],
                f"{p}_support_touch_count_rank{rank}_{s}": support[f"touch_count_{rank}"],
                f"{p}_resistance_confirmed_rank{rank}_{s}": resistance[f"valid_{rank}"],
                f"{p}_support_confirmed_rank{rank}_{s}": support[f"valid_{rank}"],
                f"{p}_resistance_age_rank{rank}_{s}": resistance[f"age_{rank}"],
                f"{p}_support_age_rank{rank}_{s}": support[f"age_{rank}"],
                f"{p}_resistance_span_rank{rank}_{s}": resistance[f"span_{rank}"],
                f"{p}_support_span_rank{rank}_{s}": support[f"span_{rank}"],
                f"{p}_resistance_respect_ratio_rank{rank}_{s}": resistance[f"respect_ratio_{rank}"],
                f"{p}_support_respect_ratio_rank{rank}_{s}": support[f"respect_ratio_{rank}"],
                f"{p}_resistance_span_age_ratio_rank{rank}_{s}": resistance[f"span_age_ratio_{rank}"],
                f"{p}_support_span_age_ratio_rank{rank}_{s}": support[f"span_age_ratio_{rank}"],
                f"{p}_resistance_intercept_rank{rank}_{s}": resistance[f"intercept_{rank}"],
                f"{p}_support_intercept_rank{rank}_{s}": support[f"intercept_{rank}"],
                f"{p}_resistance_anchor_old_index_rank{rank}_{s}": resistance[f"anchor_old_index_{rank}"],
                f"{p}_support_anchor_old_index_rank{rank}_{s}": support[f"anchor_old_index_{rank}"],
                f"{p}_resistance_anchor_new_index_rank{rank}_{s}": resistance[f"anchor_new_index_{rank}"],
                f"{p}_support_anchor_new_index_rank{rank}_{s}": support[f"anchor_new_index_{rank}"],
                f"{p}_resistance_touch_start_index_rank{rank}_{s}": resistance[f"touch_start_index_{rank}"],
                f"{p}_support_touch_start_index_rank{rank}_{s}": support[f"touch_start_index_{rank}"],
                f"{p}_resistance_touch_end_index_rank{rank}_{s}": resistance[f"touch_end_index_{rank}"],
                f"{p}_support_touch_end_index_rank{rank}_{s}": support[f"touch_end_index_{rank}"],
            }
        )

    for horizon in horizons:
        h = int(horizon)
        res_proj = (res_line + res_slope * h).clip(lower=0.0)
        sup_proj = (sup_line + sup_slope * h).clip(lower=0.0)
        res_proj_valid = res_valid & _projection_sane(res_proj, close, cfg)
        sup_proj_valid = sup_valid & _projection_sane(sup_proj, close, cfg)
        cols[f"{p}_resistance_proj_{h}_{s}"] = res_proj.where(res_proj_valid)
        cols[f"{p}_support_proj_{h}_{s}"] = sup_proj.where(sup_proj_valid)
        cols[f"{p}_resistance_breakout_{h}_{s}"] = res_proj_valid & close.gt(res_proj * (1.0 + float(cfg.breakout_buffer_pct)))
        cols[f"{p}_support_breakdown_{h}_{s}"] = sup_proj_valid & close.lt(sup_proj * (1.0 - float(cfg.breakout_buffer_pct)))
        cols[f"{p}_long_target_distance_pct_{h}_{s}"] = ((res_proj - close) / close).where(res_proj_valid)
        cols[f"{p}_short_target_distance_pct_{h}_{s}"] = ((close - sup_proj) / close).where(sup_proj_valid)

    cols.update(
        _score_columns(
            prefix=p,
            strength=s,
            support_score=support["score_0"],
            resistance_score=resistance["score_0"],
            support_reclaim=support_reclaim,
            resistance_reject=resistance_reject,
            resistance_breakout=cols.get(f"{p}_resistance_breakout_1_{s}", pd.Series(False, index=frame.index)),
            support_breakdown=cols.get(f"{p}_support_breakdown_1_{s}", pd.Series(False, index=frame.index)),
            triangle_score=triangle_score,
            trend_bias=trend_bias,
            cfg=cfg,
        )
    )
    return cols


def _ranked_structural_lines(
    frame: DataFrame,
    close: Series,
    atr: Series,
    bar_index: Series,
    *,
    strength: int,
    side: TrendlineSide,
    cfg: StructuralTrendlineConfig,
) -> dict[str, Series]:
    pp = cfg.pivot_prefix
    pivot_event_price = _num(frame, f"{pp}_pivot_{side}_{strength}")
    event_price = _body_anchor_price(frame, side, strength).where(pivot_event_price.notna())
    event_prominence = _num(frame, f"{pp}_pivot_{side}_prominence_{strength}")
    event_index = (bar_index - float(strength)).where(pivot_event_price.notna())
    engine_cfg = PivotLineEngineConfig(
        candidate_pivot_count=int(cfg.candidate_pivot_count),
        ranked_line_count=int(cfg.ranked_line_count),
        min_touch_count=int(cfg.min_touch_count),
        min_anchor_span_bars=int(cfg.min_anchor_span_bars),
        min_line_span_bars=int(cfg.min_line_span_bars),
        max_line_age_bars=int(cfg.max_line_age_bars),
        min_pivot_prominence=float(cfg.min_pivot_prominence_atr),
        touch_tolerance_atr_mult=float(cfg.touch_tolerance_atr_mult),
        touch_tolerance_pct=float(cfg.touch_tolerance_pct),
        fit_touch_tolerance_pct=float(cfg.touch_tolerance_pct),
        min_respect_ratio=float(cfg.min_respect_ratio),
        min_span_age_ratio=float(cfg.min_span_age_ratio),
        max_slope_pct_per_bar=float(cfg.max_slope_pct_per_bar),
        max_projection_distance_pct=float(cfg.max_projection_distance_pct),
        max_active_line_distance_pct=None,
        projection_bars_after_last_touch=int(cfg.max_projection_bars_after_last_touch),
        use_envelope_fit=False,
        include_rolling_fit=False,
        use_weighted_fit=False,
    )
    return ranked_pivot_lines(
        close=close,
        atr=atr,
        bar_index=bar_index,
        event_price=event_price,
        event_index=event_index,
        event_prominence=event_prominence,
        side=side,
        cfg=engine_cfg,
    )


def _recent_pivot_events(event_price: Series, event_index: Series, event_prominence: Series, index: pd.Index, count: int) -> dict[str, list[Series]]:
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
    side: TrendlineSide,
    cfg: StructuralTrendlineConfig,
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
        & prominence_min.ge(float(cfg.min_pivot_prominence_atr))
    )
    raw_slope = ((p_new - p_old) / span.replace(0.0, np.nan)).where(anchor_valid)

    zero = pd.Series(0.0, index=close.index, dtype="float64")
    touch_count = zero.copy()
    violation_count = zero.copy()
    sum_w = zero.copy()
    sum_prom = zero.copy()
    latest_touch_index = pd.Series(np.nan, index=close.index, dtype="float64")
    earliest_touch_index = pd.Series(np.nan, index=close.index, dtype="float64")

    for price, pivot_index, prominence in zip(recent["price"], recent["index"], recent["prominence"], strict=False):
        projected_at_pivot = p_new + raw_slope * (pivot_index - x_new)
        tolerance = pd.concat([atr * float(cfg.touch_tolerance_atr_mult), price.abs() * float(cfg.touch_tolerance_pct)], axis=1).max(axis=1)
        residual = price - projected_at_pivot
        touch = anchor_valid & price.notna() & pivot_index.notna() & residual.abs().le(tolerance)
        violation = (residual.lt(-tolerance) if side == "low" else residual.gt(tolerance)).fillna(False)
        touch_f = touch.fillna(False).astype("float64")
        weight = touch_f * (1.0 + _clip01(prominence.fillna(0.0) / 5.0))
        touch_count = touch_count + touch_f
        violation_count = violation_count + violation.astype("float64")
        sum_w = sum_w + weight
        sum_prom = sum_prom + weight * prominence.fillna(0.0)
        touch_index = pivot_index.where(touch)
        latest_touch_index = pd.concat([latest_touch_index, touch_index], axis=1).max(axis=1)
        earliest_touch_index = pd.concat([earliest_touch_index, touch_index], axis=1).min(axis=1)

    line_slope = raw_slope
    line_intercept = p_new - line_slope * x_new
    line = (line_intercept + line_slope * bar_index).clip(lower=0.0)

    line_span = latest_touch_index - earliest_touch_index
    age = bar_index - latest_touch_index
    prominence_mean = _safe_div(sum_prom, sum_w)
    slope_pct = (line_slope.abs() / close).replace([np.inf, -np.inf], np.nan)
    distance_pct = ((line - close).abs() / close).replace([np.inf, -np.inf], np.nan)
    current_tolerance = pd.concat([atr * float(cfg.touch_tolerance_atr_mult), close.abs() * float(cfg.touch_tolerance_pct)], axis=1).max(axis=1)
    side_ok = line.le(close + current_tolerance) if side == "low" else line.ge(close - current_tolerance)
    respect_ratio = _safe_div(touch_count, touch_count + violation_count).fillna(0.0)
    total_line_age = bar_index - earliest_touch_index
    span_age_ratio = _safe_div(line_span, total_line_age).fillna(0.0)
    valid = (
        line.notna()
        & line.gt(0.0)
        & touch_count.ge(float(cfg.min_touch_count))
        & line_span.ge(float(cfg.min_line_span_bars))
        & age.le(float(cfg.max_line_age_bars))
        & respect_ratio.ge(float(cfg.min_respect_ratio))
        & span_age_ratio.ge(float(cfg.min_span_age_ratio))
        & slope_pct.le(float(cfg.max_slope_pct_per_bar))
        & distance_pct.le(float(cfg.max_projection_distance_pct))
        & side_ok
    ).fillna(False)
    id_scale = float(max(len(close) + int(cfg.max_line_age_bars) + 10, 10_000))
    line_id = (x_old * id_scale + x_new).where(valid)

    touch_score = _clip01(touch_count / max(float(cfg.min_touch_count) + 3.0, 1.0))
    span_score = _clip01(line_span / max(float(cfg.min_line_span_bars) * 5.0, 1.0))
    age_score = _clip01(1.0 - age / max(float(cfg.max_line_age_bars), 1.0))
    prominence_score = _clip01(prominence_mean / max(float(cfg.min_pivot_prominence_atr) * 4.0, 1.0))
    distance_score = _clip01(1.0 - distance_pct / max(float(cfg.max_projection_distance_pct), 1e-9))
    slope_score = _clip01(1.0 - slope_pct / max(float(cfg.max_slope_pct_per_bar), 1e-9))
    score = _clip01(
        0.30 * touch_score
        + 0.22 * span_score
        + 0.16 * prominence_score
        + 0.14 * respect_ratio
        + 0.08 * age_score
        + 0.06 * distance_score
        + 0.04 * slope_score
    ).where(valid)
    return {
        "line": line.where(valid),
        "slope": line_slope.where(valid),
        "score": score,
        "touch_count": touch_count.where(valid),
        "slope_pct": slope_pct.where(valid),
        "age": age.where(valid),
        "span": line_span.where(valid),
        "prominence": prominence_min.where(valid),
        "respect_ratio": respect_ratio.where(valid),
        "span_age_ratio": span_age_ratio.where(valid),
        "line_id": line_id,
        "intercept": line_intercept.where(valid),
        "anchor_old_index": x_old.where(valid),
        "anchor_new_index": x_new.where(valid),
        "touch_start_index": earliest_touch_index.where(valid),
        "touch_end_index": latest_touch_index.where(valid),
        "valid": valid,
    }


def _rank_candidates(candidates: list[dict[str, Series]], index: pd.Index, slots: int) -> dict[str, Series]:
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
    metric_frames = {
        metric: pd.concat([candidate[metric] for candidate in candidates], axis=1).astype("float64").to_numpy()
        for metric in (
            "line", "slope", "touch_count", "slope_pct", "age", "span", "prominence",
            "respect_ratio", "span_age_ratio", "line_id", "intercept", "anchor_old_index",
            "anchor_new_index", "touch_start_index", "touch_end_index",
        )
    }
    out: dict[str, Series] = {}
    for rank in range(slots):
        if rank >= score_values.shape[1]:
            for metric in (
                "line", "slope", "score", "touch_count", "slope_pct", "age", "span", "prominence",
                "respect_ratio", "span_age_ratio", "line_id", "intercept", "anchor_old_index",
                "anchor_new_index", "touch_start_index", "touch_end_index",
            ):
                out[f"{metric}_{rank}"] = pd.Series(np.nan, index=index, dtype="float64")
            out[f"valid_{rank}"] = pd.Series(False, index=index)
            continue
        chosen = order[:, rank]
        valid = np.isfinite(sorted_scores[:, rank]) & (sorted_scores[:, rank] > -np.inf)
        out[f"score_{rank}"] = pd.Series(np.where(valid, sorted_scores[:, rank], np.nan), index=index, dtype="float64")
        out[f"valid_{rank}"] = pd.Series(valid, index=index)
        for metric, values in metric_frames.items():
            selected = values[row_index, chosen]
            out[f"{metric}_{rank}"] = pd.Series(np.where(valid, selected, np.nan), index=index, dtype="float64")
    return out


def _empty_ranked(index: pd.Index, slots: int) -> dict[str, Series]:
    out: dict[str, Series] = {}
    for rank in range(slots):
        for metric in (
            "line", "slope", "score", "touch_count", "slope_pct", "age", "span", "prominence",
            "respect_ratio", "span_age_ratio", "line_id", "intercept", "anchor_old_index",
            "anchor_new_index", "touch_start_index", "touch_end_index",
        ):
            out[f"{metric}_{rank}"] = pd.Series(np.nan, index=index, dtype="float64")
        out[f"valid_{rank}"] = pd.Series(False, index=index)
    return out


def _score_columns(
    *,
    prefix: str,
    strength: int,
    support_score: Series,
    resistance_score: Series,
    support_reclaim: Series,
    resistance_reject: Series,
    resistance_breakout: Series,
    support_breakdown: Series,
    triangle_score: Series,
    trend_bias: Series,
    cfg: StructuralTrendlineConfig,
) -> dict[str, Series]:
    s = int(strength)
    w = int(cfg.score_window)
    resistance_breakout = pd.Series(resistance_breakout, index=support_score.index).fillna(False).astype("bool")
    support_breakdown = pd.Series(support_breakdown, index=support_score.index).fillna(False).astype("bool")
    support_reclaim = pd.Series(support_reclaim, index=support_score.index).fillna(False).astype("bool")
    resistance_reject = pd.Series(resistance_reject, index=support_score.index).fillna(False).astype("bool")
    support_quality = support_score.fillna(0.0)
    resistance_quality = resistance_score.fillna(0.0)
    triangle_quality = triangle_score.fillna(0.0)

    long_score = _clip01(
        0.30 * support_quality
        + 0.25 * _rolling_count(resistance_breakout, w).clip(0.0, 1.0)
        + 0.20 * _rolling_count(support_reclaim, w).clip(0.0, 1.0)
        + 0.15 * triangle_quality
        + 0.10 * trend_bias.gt(0.0).astype("float64")
    )
    short_score = _clip01(
        0.30 * resistance_quality
        + 0.25 * _rolling_count(support_breakdown, w).clip(0.0, 1.0)
        + 0.20 * _rolling_count(resistance_reject, w).clip(0.0, 1.0)
        + 0.15 * triangle_quality
        + 0.10 * trend_bias.lt(0.0).astype("float64")
    )
    abs_score = pd.concat([long_score, short_score], axis=1).max(axis=1)
    state = pd.Series(np.select([long_score.gt(short_score), short_score.gt(long_score)], [1.0, -1.0], default=0.0), index=long_score.index)
    margin = long_score - short_score
    recent_bull_event = _rolling_count(resistance_breakout | support_reclaim, w).ge(1.0)
    recent_bear_event = _rolling_count(support_breakdown | resistance_reject, w).ge(1.0)
    full_bull = (
        long_score.ge(0.46)
        & margin.ge(0.08)
        & (trend_bias.gt(0.0) | support_quality.ge(0.45) | recent_bull_event)
    )
    full_bear = (
        short_score.ge(0.46)
        & margin.le(-0.08)
        & (trend_bias.lt(0.0) | resistance_quality.ge(0.45) | recent_bear_event)
    )
    bullish_chop = ~full_bull & ~full_bear & margin.ge(0.04) & long_score.ge(0.28)
    bearish_chop = ~full_bull & ~full_bear & margin.le(-0.04) & short_score.ge(0.28)
    market_context = pd.Series(
        np.select([full_bull, full_bear, bullish_chop, bearish_chop], [2, -2, 1, -1], default=0),
        index=support_score.index,
        dtype="int8",
    )

    cooldown = max(3, int(cfg.score_window) // 8)
    entry_resistance_breakout_long = resistance_breakout & long_score.ge(short_score) & resistance_quality.ge(0.30)
    entry_support_reclaim_long = support_reclaim & long_score.ge(short_score - 0.03) & support_quality.ge(0.30)
    entry_triangle_breakout_long = resistance_breakout & triangle_quality.ge(0.35) & long_score.ge(short_score)
    entry_support_breakdown_short = support_breakdown & short_score.ge(long_score) & support_quality.ge(0.30)
    entry_resistance_reject_short = resistance_reject & short_score.ge(long_score - 0.03) & resistance_quality.ge(0.30)
    entry_triangle_breakdown_short = support_breakdown & triangle_quality.ge(0.35) & short_score.ge(long_score)

    hold_long = support_quality.ge(0.45) & long_score.ge(short_score - 0.05) & ~support_breakdown
    hold_short = resistance_quality.ge(0.45) & short_score.ge(long_score - 0.05) & ~resistance_breakout
    exit_long = resistance_reject | support_breakdown
    exit_short = support_reclaim | resistance_breakout
    suggested_entry_long = entry_resistance_breakout_long | entry_support_reclaim_long | entry_triangle_breakout_long
    suggested_entry_short = entry_support_breakdown_short | entry_resistance_reject_short | entry_triangle_breakdown_short
    return {
        f"{prefix}_market_context_{s}": market_context,
        f"{prefix}_entry_resistance_breakout_long_{s}": _dedupe_events(entry_resistance_breakout_long, cooldown),
        f"{prefix}_entry_support_reclaim_long_{s}": _dedupe_events(entry_support_reclaim_long, cooldown),
        f"{prefix}_entry_triangle_breakout_long_{s}": _dedupe_events(entry_triangle_breakout_long, cooldown),
        f"{prefix}_entry_support_breakdown_short_{s}": _dedupe_events(entry_support_breakdown_short, cooldown),
        f"{prefix}_entry_resistance_reject_short_{s}": _dedupe_events(entry_resistance_reject_short, cooldown),
        f"{prefix}_entry_triangle_breakdown_short_{s}": _dedupe_events(entry_triangle_breakdown_short, cooldown),
        f"{prefix}_hold_long_{s}": hold_long.fillna(False),
        f"{prefix}_hold_short_{s}": hold_short.fillna(False),
        f"{prefix}_exit_long_{s}": _dedupe_events(exit_long, cooldown),
        f"{prefix}_exit_short_{s}": _dedupe_events(exit_short, cooldown),
        f"{prefix}_suggested_entry_long_{s}": _dedupe_events(suggested_entry_long, cooldown),
        f"{prefix}_suggested_entry_short_{s}": _dedupe_events(suggested_entry_short, cooldown),
        f"{prefix}_score_long_{s}": long_score,
        f"{prefix}_score_short_{s}": short_score,
        f"{prefix}_score_abs_{s}": abs_score,
        f"{prefix}_state_{s}": state,
    }


def _touches_line(high: Series, low: Series, level: Series, zone_width: Series) -> Series:
    return level.notna() & high.ge(level - zone_width) & low.le(level + zone_width)


def _projection_sane(level: Series, close: Series, cfg: StructuralTrendlineConfig) -> Series:
    distance = ((level - close).abs() / close).replace([np.inf, -np.inf], np.nan)
    return level.gt(0.0) & distance.le(float(cfg.max_projection_distance_pct))


def _combined_identity(left: Series, right: Series) -> Series:
    left_id = pd.to_numeric(left, errors="coerce")
    right_id = pd.to_numeric(right, errors="coerce")
    max_id = pd.concat([left_id.abs(), right_id.abs()]).max()
    scale = float(max(max_id + 1.0, 1_000_000.0)) if pd.notna(max_id) else 1_000_000.0
    return left_id.fillna(-1.0) * scale + right_id.fillna(-1.0)


def _plot_safe_line(
    line: Series,
    break_pct: float,
    min_segment_bars: int,
    close: Series,
    max_distance_pct: float,
    identity: Series | None = None,
) -> Series:
    plotted = pd.to_numeric(line, errors="coerce").copy()
    close_num = pd.to_numeric(close, errors="coerce")
    too_far = ((plotted - close_num).abs() / close_num.abs().replace(0.0, np.nan)).gt(float(max_distance_pct))
    plotted = plotted.mask(too_far)
    if float(break_pct) >= 100.0:
        return plotted
    prior = plotted.shift(1)
    jump = ((plotted - prior).abs() / prior.abs().replace(0.0, np.nan)).gt(float(break_pct))
    starts_after_gap = plotted.notna() & prior.isna()
    candidate_changed = pd.Series(False, index=plotted.index)
    if identity is not None:
        line_id = pd.Series(identity, index=plotted.index)
        prior_id = line_id.shift(1)
        candidate_changed = plotted.notna() & prior.notna() & line_id.notna() & prior_id.notna() & line_id.ne(prior_id)
    broken = plotted.mask(jump | starts_after_gap | candidate_changed)
    min_bars = max(int(min_segment_bars), 1)
    if min_bars <= 1:
        return broken
    present = broken.notna()
    group_id = present.ne(present.shift(fill_value=False)).cumsum()
    segment_len = present.groupby(group_id).transform("sum")
    return broken.mask(present & segment_len.lt(min_bars))


def _rolling_count(mask: Series, window: int) -> Series:
    clean = pd.Series(mask, index=mask.index).fillna(False).astype("int8")
    cumulative = clean.cumsum()
    return (cumulative - cumulative.shift(int(window), fill_value=0)).astype("float64")


def _dedupe_events(mask: Series, cooldown_bars: int) -> Series:
    clean = pd.Series(mask, index=mask.index).fillna(False).astype("bool")
    prior_recent = _rolling_count(clean.shift(1, fill_value=False), max(int(cooldown_bars), 1))
    return (clean & prior_recent.eq(0.0)).fillna(False)


def _active_alias_columns(cols: dict[str, Series], prefix: str, strength: int, horizons: list[int], ranked_line_count: int) -> dict[str, Series]:
    aliases = (
        "resistance_line", "support_line", "resistance_plot", "support_plot", "resistance_slope", "support_slope",
        "resistance_slope_pct", "support_slope_pct", "resistance_confirmed", "support_confirmed",
        "descending_resistance", "ascending_support",
        "resistance_touch_count", "support_touch_count", "resistance_score", "support_score",
        "resistance_age", "support_age", "resistance_span", "support_span", "channel_valid", "channel_mid",
        "channel_mid_plot", "channel_width_pct", "channel_compression", "wedge_compression", "triangle_score",
        "trend_bias", "resistance_touch", "support_touch", "resistance_reject", "support_reclaim",
        "resistance_violation", "support_violation", "suggested_entry_long", "suggested_entry_short",
        "market_context", "entry_resistance_breakout_long", "entry_support_reclaim_long", "entry_triangle_breakout_long",
        "entry_support_breakdown_short", "entry_resistance_reject_short", "entry_triangle_breakdown_short",
        "hold_long", "hold_short", "exit_long", "exit_short",
        "score_long", "score_short", "score_abs", "state",
    )
    out: dict[str, Series] = {}
    for alias in aliases:
        src = f"{prefix}_{alias}_{strength}"
        if src in cols:
            out[f"{prefix}_{alias}"] = cols[src]
    for rank in range(max(int(ranked_line_count), 1)):
        for alias in (
            "resistance_line_rank", "support_line_rank", "resistance_plot_rank", "support_plot_rank",
            "resistance_line_id_rank", "support_line_id_rank", "resistance_slope_rank", "support_slope_rank",
            "resistance_slope_pct_rank", "support_slope_pct_rank", "resistance_score_rank", "support_score_rank",
            "resistance_touch_count_rank", "support_touch_count_rank",
            "resistance_confirmed_rank", "support_confirmed_rank", "resistance_age_rank", "support_age_rank",
            "resistance_span_rank", "support_span_rank", "resistance_respect_ratio_rank", "support_respect_ratio_rank",
            "resistance_span_age_ratio_rank", "support_span_age_ratio_rank", "resistance_intercept_rank", "support_intercept_rank",
            "resistance_anchor_old_index_rank", "support_anchor_old_index_rank",
            "resistance_anchor_new_index_rank", "support_anchor_new_index_rank",
            "resistance_touch_start_index_rank", "support_touch_start_index_rank",
            "resistance_touch_end_index_rank", "support_touch_end_index_rank",
        ):
            src = f"{prefix}_{alias}{rank}_{strength}"
            if src in cols:
                out[f"{prefix}_{alias}{rank}"] = cols[src]
    for horizon in horizons:
        for alias in (
            "resistance_proj", "support_proj", "resistance_breakout", "support_breakdown",
            "long_target_distance_pct", "short_target_distance_pct",
        ):
            src = f"{prefix}_{alias}_{horizon}_{strength}"
            if src in cols:
                out[f"{prefix}_{alias}_{horizon}"] = cols[src]
    return out


def _ensure_pivot_columns(frame: DataFrame, cfg: StructuralTrendlineConfig) -> DataFrame:
    required = [
        f"{cfg.pivot_prefix}_pivot_{side}_{strength}"
        for strength in _sorted_strengths(cfg)
        for side in ("high", "low")
    ]
    missing = [column for column in required if column not in frame.columns]
    if not missing:
        return frame
    if cfg.missing_pivot_mode == "skip":
        return frame
    if cfg.missing_pivot_mode == "compute":
        if add_pivot_structure is None:
            raise ImportError("missing_pivot_mode='compute' requires complex_pivot_structure.py")
        pivot_cfg = cfg.pivot_config
        if pivot_cfg is None:
            if PivotStructureConfig is None:
                raise ImportError("PivotStructureConfig unavailable")
            pivot_cfg = PivotStructureConfig(strength=cfg.strength, strengths=tuple(cfg.strengths), prefix=cfg.pivot_prefix)
        return add_pivot_structure(frame, pivot_cfg)
    raise KeyError(
        f"Missing pivot columns with prefix '{cfg.pivot_prefix}': {', '.join(missing[:6])}. "
        "Run add_pivot_structure with matching strengths first or use missing_pivot_mode='compute'."
    )


def _resolve_config(config: StructuralTrendlineConfig | None, **overrides: object) -> StructuralTrendlineConfig:
    base = config or StructuralTrendlineConfig()
    values = dict(base.__dict__)
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return StructuralTrendlineConfig(**values)


def _validate_config(cfg: StructuralTrendlineConfig) -> None:
    if int(cfg.strength) <= 0:
        raise ValueError("strength must be positive")
    if not _sorted_strengths(cfg):
        raise ValueError("strengths must contain at least one value")
    if cfg.ranked_line_count < 1:
        raise ValueError("ranked_line_count must be at least 1")
    if cfg.candidate_pivot_count < 3:
        raise ValueError("candidate_pivot_count must be at least 3")
    if cfg.min_touch_count < 2:
        raise ValueError("min_touch_count must be at least 2")
    if cfg.min_anchor_span_bars < 1 or cfg.min_line_span_bars < 1:
        raise ValueError("span settings must be positive")
    if cfg.max_line_age_bars < cfg.min_line_span_bars:
        raise ValueError("max_line_age_bars must be >= min_line_span_bars")
    if cfg.min_pivot_prominence_atr < 0.0:
        raise ValueError("min_pivot_prominence_atr must be non-negative")
    if cfg.touch_tolerance_atr_mult < 0.0 or cfg.touch_tolerance_pct < 0.0:
        raise ValueError("touch tolerance settings must be non-negative")
    if not 0.0 <= float(cfg.min_respect_ratio) <= 1.0:
        raise ValueError("min_respect_ratio must be between 0 and 1")
    if not 0.0 <= float(cfg.min_span_age_ratio) <= 1.0:
        raise ValueError("min_span_age_ratio must be between 0 and 1")
    if cfg.max_slope_pct_per_bar <= 0.0 or cfg.max_projection_distance_pct <= 0.0:
        raise ValueError("slope/projection settings must be positive")
    if cfg.max_projection_bars_after_last_touch < 1:
        raise ValueError("max_projection_bars_after_last_touch must be positive")
    if cfg.compression_window < 2:
        raise ValueError("compression_window must be at least 2")
    if cfg.compression_min_periods < 1 or cfg.compression_min_periods > cfg.compression_window:
        raise ValueError("compression_min_periods must be between 1 and compression_window")
    if cfg.score_window < 2:
        raise ValueError("score_window must be at least 2")
    if cfg.plot_min_segment_bars < 1:
        raise ValueError("plot_min_segment_bars must be at least 1")
    if cfg.plot_max_distance_pct <= 0.0:
        raise ValueError("plot_max_distance_pct must be positive")


def _validate_dataframe(dataframe: DataFrame) -> None:
    missing = [col for col in ("open", "high", "low", "close") if col not in dataframe.columns]
    if missing:
        raise KeyError(f"Missing required OHLC columns: {', '.join(missing)}")


def _sorted_unique_ints(values: Sequence[int]) -> list[int]:
    return sorted({int(v) for v in values if int(v) > 0})


def _sorted_strengths(cfg: StructuralTrendlineConfig) -> list[int]:
    values = {int(v) for v in cfg.strengths if int(v) > 0}
    values.add(int(cfg.strength))
    return sorted(values)


def _safe_div(numerator: Series, denominator: Series) -> Series:
    return numerator / denominator.replace(0.0, np.nan)


def _clip01(series: Series) -> Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0).fillna(0.0)


def _num(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _body_anchor_price(frame: DataFrame, side: str, strength: int) -> Series:
    body_high = pd.concat([_num(frame, "open"), _num(frame, "close")], axis=1).max(axis=1)
    body_low = pd.concat([_num(frame, "open"), _num(frame, "close")], axis=1).min(axis=1)
    body = body_high if side == "high" else body_low
    return body.shift(int(strength))


def _optional_num(frame: DataFrame, column: str, fallback: Series | float) -> Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    if isinstance(fallback, Series):
        return pd.Series(fallback, index=frame.index)
    return pd.Series(float(fallback), index=frame.index, dtype="float64")


def _atr(frame: DataFrame, period: int) -> Series:
    high = _num(frame, "high")
    low = _num(frame, "low")
    close = _num(frame, "close")
    prev_close = close.shift(1)
    true_range = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return true_range.rolling(int(period), min_periods=1).mean()

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

try:
    from .complex_pivot_structure import PivotStructureConfig, add_pivot_structure
except Exception:  # pragma: no cover - optional fallback for standalone notebooks
    PivotStructureConfig = None  # type: ignore[assignment]
    add_pivot_structure = None  # type: ignore[assignment]

MissingPivotMode = Literal["raise", "compute", "skip"]


@dataclass(frozen=True)
class TrendlineProjectionConfig:
    """Projected trendline evidence derived from confirmed pivot structure.

    The module emits evidence and normalized score columns for Freqtrade
    strategies. It does not create entry/exit signals, import volume/profile
    indicators, or apply hidden trading decisions. Strategy code should decide
    how to use the columns.

    Score columns are 0..1 confidence values:
    - ``*_score_long``: support/channel evidence favoring long setups.
    - ``*_score_short``: resistance/channel evidence favoring short setups.
    - ``*_score_abs``: max(long, short).
    - ``*_state``: -1, 0, or 1 directional state.

    Score meaning:
    Long scores rise when support lines are valid, respected, recently
    reclaimed, and the channel/breakout structure favors upside. Short scores
    rise when resistance lines are valid, respected, recently rejected, and the
    channel/breakdown structure favors downside. Compression contributes to both
    sides because compressed channels can precede directional expansion.

    Hyperopt-facing quality knobs include anchor span/age/prominence, slope
    sanity, channel width bounds, projection distance, touch count, respect
    ratio, and compression windows.
    """

    strength: int = 5
    strengths: Sequence[int] = (3, 5, 8, 13)
    horizons: Sequence[int] = (1, 3, 6, 12, 24)
    event_windows: Sequence[int] = (12, 24, 48, 96)
    pivot_prefix: str = "pa"
    output_prefix: str = "tl"
    missing_pivot_mode: MissingPivotMode = "raise"
    pivot_config: object | None = None

    near_zone_atr_mult: float = 0.35
    near_zone_pct: float = 0.004
    breakout_buffer_pct: float = 0.002

    # Quality guards. Use large values to make a guard effectively non-blocking.
    min_anchor_span_bars: int = 4
    max_anchor_age_bars: int = 180
    min_anchor_prominence_atr: float = 0.0
    max_line_slope_pct_per_bar: float = 0.08
    min_channel_width_pct: float = 0.001
    max_channel_width_pct: float = 0.80
    max_projection_distance_pct: float = 2.00
    compression_window: int = 48
    compression_min_periods: int = 12
    score_window: int = 48
    min_touch_count: int = 1
    min_respect_ratio: float = 0.50

    # Plot-only break. A large value keeps plot lines continuous. A smaller value
    # inserts NaNs on large line jumps so FreqUI does not draw misleading verticals.
    plot_break_on_line_change_pct: float = 0.08


def add_trendline_projection(
    dataframe: DataFrame,
    config: TrendlineProjectionConfig | None = None,
    *,
    strength: int | None = None,
    strengths: Sequence[int] | None = None,
    horizons: Sequence[int] | None = None,
    event_windows: Sequence[int] | None = None,
    pivot_prefix: str | None = None,
    output_prefix: str | None = None,
    missing_pivot_mode: MissingPivotMode | None = None,
    pivot_config: object | None = None,
    near_zone_atr_mult: float | None = None,
    near_zone_pct: float | None = None,
    breakout_buffer_pct: float | None = None,
    min_anchor_span_bars: int | None = None,
    max_anchor_age_bars: int | None = None,
    min_anchor_prominence_atr: float | None = None,
    max_line_slope_pct_per_bar: float | None = None,
    min_channel_width_pct: float | None = None,
    max_channel_width_pct: float | None = None,
    max_projection_distance_pct: float | None = None,
    compression_window: int | None = None,
    compression_min_periods: int | None = None,
    score_window: int | None = None,
    min_touch_count: int | None = None,
    min_respect_ratio: float | None = None,
    plot_break_on_line_change_pct: float | None = None,
) -> DataFrame:
    """Append vectorized trendline projection and quality evidence columns.

    The calculation uses confirmed pivot columns. It reconstructs anchor spans and
    slopes from the actual pivot events instead of taking ``line.diff()``, which
    would turn pivot-line regime changes into false slopes and vertical artefacts.

    Expected usage is to call ``add_pivot_structure`` first, then this function,
    or set ``missing_pivot_mode='compute'`` for convenience.
    """

    cfg = _resolve_config(
        config,
        strength=strength,
        strengths=strengths,
        horizons=horizons,
        event_windows=event_windows,
        pivot_prefix=pivot_prefix,
        output_prefix=output_prefix,
        missing_pivot_mode=missing_pivot_mode,
        pivot_config=pivot_config,
        near_zone_atr_mult=near_zone_atr_mult,
        near_zone_pct=near_zone_pct,
        breakout_buffer_pct=breakout_buffer_pct,
        min_anchor_span_bars=min_anchor_span_bars,
        max_anchor_age_bars=max_anchor_age_bars,
        min_anchor_prominence_atr=min_anchor_prominence_atr,
        max_line_slope_pct_per_bar=max_line_slope_pct_per_bar,
        min_channel_width_pct=min_channel_width_pct,
        max_channel_width_pct=max_channel_width_pct,
        max_projection_distance_pct=max_projection_distance_pct,
        compression_window=compression_window,
        compression_min_periods=compression_min_periods,
        score_window=score_window,
        min_touch_count=min_touch_count,
        min_respect_ratio=min_respect_ratio,
        plot_break_on_line_change_pct=plot_break_on_line_change_pct,
    )
    _validate_config(cfg)
    _validate_dataframe(dataframe)

    frame = dataframe.copy()
    frame = _ensure_pivot_columns(frame, cfg)

    close = _num(frame, "close").replace(0.0, np.nan)
    high = _num(frame, "high")
    low = _num(frame, "low")
    open_ = _num(frame, "open")
    atr = _optional_num(frame, f"{cfg.pivot_prefix}_atr", _atr(frame, 14))
    zone_width = pd.concat([atr * cfg.near_zone_atr_mult, close * cfg.near_zone_pct], axis=1).max(axis=1)

    strengths_list = _sorted_strengths(cfg)
    horizons_list = _sorted_unique_ints(cfg.horizons)
    windows_list = _sorted_unique_ints(cfg.event_windows)
    p = cfg.output_prefix

    new_cols: dict[str, Series] = {f"{p}_zone_width": zone_width}
    for s in strengths_list:
        new_cols.update(
            _columns_for_strength(
                frame=frame,
                close=close,
                high=high,
                low=low,
                open_=open_,
                zone_width=zone_width,
                strength=s,
                horizons=horizons_list,
                windows=windows_list,
                cfg=cfg,
            )
        )

    selected = int(cfg.strength)
    if selected not in strengths_list:
        selected = strengths_list[0]
    new_cols.update(_active_alias_columns(new_cols, p, selected, horizons_list, windows_list))

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
    zone_width: Series,
    strength: int,
    horizons: list[int],
    windows: list[int],
    cfg: TrendlineProjectionConfig,
) -> dict[str, Series]:
    pp = cfg.pivot_prefix
    p = cfg.output_prefix
    s = int(strength)
    bar_index = _optional_num(frame, f"{pp}_bar_index", pd.Series(np.arange(len(frame), dtype="float64"), index=frame.index))

    res_anchor = _anchor_state(frame, pp, s, "high", bar_index)
    sup_anchor = _anchor_state(frame, pp, s, "low", bar_index)

    resistance_line = _line_from_anchor(bar_index, res_anchor["last_price"], res_anchor["last_index"], res_anchor["slope"])
    support_line = _line_from_anchor(bar_index, sup_anchor["last_price"], sup_anchor["last_index"], sup_anchor["slope"])
    resistance_line = resistance_line.clip(lower=0.0)
    support_line = support_line.clip(lower=0.0)

    res_slope = res_anchor["slope"]
    sup_slope = sup_anchor["slope"]
    channel_width_pct = (resistance_line - support_line) / close
    channel_mid = (resistance_line + support_line) / 2.0
    channel_compression = channel_width_pct / channel_width_pct.rolling(
        int(cfg.compression_window),
        min_periods=int(cfg.compression_min_periods),
    ).median()

    res_line_valid = _line_validity(res_anchor, res_slope, resistance_line, close, cfg)
    sup_line_valid = _line_validity(sup_anchor, sup_slope, support_line, close, cfg)
    channel_valid = (
        res_line_valid
        & sup_line_valid
        & resistance_line.gt(support_line)
        & channel_width_pct.ge(cfg.min_channel_width_pct)
        & channel_width_pct.le(cfg.max_channel_width_pct)
    ).fillna(False)

    res_plot = _plot_safe_line(resistance_line.where(res_line_valid), cfg.plot_break_on_line_change_pct)
    sup_plot = _plot_safe_line(support_line.where(sup_line_valid), cfg.plot_break_on_line_change_pct)
    channel_mid_plot = _plot_safe_line(channel_mid.where(channel_valid), cfg.plot_break_on_line_change_pct)

    resistance_touch = res_line_valid & _touches_line(high, low, resistance_line, zone_width)
    support_touch = sup_line_valid & _touches_line(high, low, support_line, zone_width)
    resistance_violation = res_line_valid & close.gt(resistance_line + zone_width)
    support_violation = sup_line_valid & close.lt(support_line - zone_width)
    resistance_reject = resistance_touch & close.le(resistance_line) & close.lt(open_)
    support_reclaim = support_touch & close.ge(support_line) & close.gt(open_)

    trend_bias = pd.Series(
        np.select(
            [channel_valid & res_slope.gt(0.0) & sup_slope.gt(0.0), channel_valid & res_slope.lt(0.0) & sup_slope.lt(0.0)],
            [1.0, -1.0],
            default=0.0,
        ),
        index=frame.index,
    )

    parallel_error_pct = ((res_slope - sup_slope).abs() / close).where(channel_valid)
    slope_mean_pct = ((res_slope + sup_slope) / 2.0 / close).where(channel_valid)

    cols: dict[str, Series] = {
        f"{p}_resistance_line_{s}": resistance_line.where(res_line_valid),
        f"{p}_support_line_{s}": support_line.where(sup_line_valid),
        f"{p}_resistance_plot_{s}": res_plot,
        f"{p}_support_plot_{s}": sup_plot,
        f"{p}_channel_mid_{s}": channel_mid.where(channel_valid),
        f"{p}_channel_mid_plot_{s}": channel_mid_plot,
        f"{p}_resistance_slope_{s}": res_slope.where(res_line_valid),
        f"{p}_support_slope_{s}": sup_slope.where(sup_line_valid),
        f"{p}_resistance_slope_pct_{s}": (res_slope / close).where(res_line_valid),
        f"{p}_support_slope_pct_{s}": (sup_slope / close).where(sup_line_valid),
        f"{p}_slope_mean_pct_{s}": slope_mean_pct,
        f"{p}_parallel_error_pct_{s}": parallel_error_pct,
        f"{p}_channel_valid_{s}": channel_valid,
        f"{p}_channel_width_pct_{s}": channel_width_pct.where(channel_valid),
        f"{p}_channel_compression_{s}": channel_compression.where(channel_valid),
        f"{p}_trend_bias_{s}": trend_bias,
        f"{p}_resistance_line_valid_{s}": res_line_valid,
        f"{p}_support_line_valid_{s}": sup_line_valid,
        f"{p}_resistance_anchor_span_{s}": res_anchor["span"],
        f"{p}_support_anchor_span_{s}": sup_anchor["span"],
        f"{p}_resistance_age_{s}": res_anchor["age"],
        f"{p}_support_age_{s}": sup_anchor["age"],
        f"{p}_resistance_anchor_prominence_min_{s}": res_anchor["prominence_min"],
        f"{p}_support_anchor_prominence_min_{s}": sup_anchor["prominence_min"],
        f"{p}_last_pivot_high_{s}": res_anchor["last_price"],
        f"{p}_last_pivot_low_{s}": sup_anchor["last_price"],
        f"{p}_prev_pivot_high_{s}": res_anchor["prev_price"],
        f"{p}_prev_pivot_low_{s}": sup_anchor["prev_price"],
        f"{p}_resistance_touch_{s}": resistance_touch,
        f"{p}_support_touch_{s}": support_touch,
        f"{p}_resistance_reject_{s}": resistance_reject,
        f"{p}_support_reclaim_{s}": support_reclaim,
        f"{p}_resistance_violation_{s}": resistance_violation,
        f"{p}_support_violation_{s}": support_violation,
    }

    for h in horizons:
        hp = int(h)
        resistance_proj = (resistance_line + res_slope * hp).clip(lower=0.0)
        support_proj = (support_line + sup_slope * hp).clip(lower=0.0)
        res_proj_valid = res_line_valid & _projection_sane(resistance_proj, close, cfg)
        sup_proj_valid = sup_line_valid & _projection_sane(support_proj, close, cfg)
        channel_proj_valid = channel_valid & res_proj_valid & sup_proj_valid & resistance_proj.gt(support_proj)

        cols[f"{p}_resistance_proj_{hp}_{s}"] = resistance_proj.where(res_proj_valid)
        cols[f"{p}_support_proj_{hp}_{s}"] = support_proj.where(sup_proj_valid)
        cols[f"{p}_resistance_proj_plot_{hp}_{s}"] = _plot_safe_line(resistance_proj.where(res_proj_valid), cfg.plot_break_on_line_change_pct)
        cols[f"{p}_support_proj_plot_{hp}_{s}"] = _plot_safe_line(support_proj.where(sup_proj_valid), cfg.plot_break_on_line_change_pct)
        cols[f"{p}_channel_mid_proj_{hp}_{s}"] = ((resistance_proj + support_proj) / 2.0).where(channel_proj_valid)
        cols[f"{p}_channel_width_proj_pct_{hp}_{s}"] = ((resistance_proj - support_proj) / close).where(channel_proj_valid)
        cols[f"{p}_resistance_near_{hp}_{s}"] = res_proj_valid & _near_level(close, resistance_proj, zone_width)
        cols[f"{p}_support_near_{hp}_{s}"] = sup_proj_valid & _near_level(close, support_proj, zone_width)
        cols[f"{p}_long_target_distance_pct_{hp}_{s}"] = ((resistance_proj - close) / close).where(res_proj_valid)
        cols[f"{p}_short_target_distance_pct_{hp}_{s}"] = ((close - support_proj) / close).where(sup_proj_valid)
        cols[f"{p}_resistance_breakout_{hp}_{s}"] = res_proj_valid & close.gt(resistance_proj * (1.0 + cfg.breakout_buffer_pct))
        cols[f"{p}_support_breakdown_{hp}_{s}"] = sup_proj_valid & close.lt(support_proj * (1.0 - cfg.breakout_buffer_pct))

    for w in windows:
        wp = int(w)
        res_touch_count = _rolling_count(resistance_touch, wp)
        sup_touch_count = _rolling_count(support_touch, wp)
        res_violation_count = _rolling_count(resistance_violation, wp)
        sup_violation_count = _rolling_count(support_violation, wp)
        cols[f"{p}_resistance_touch_count_{wp}_{s}"] = res_touch_count
        cols[f"{p}_support_touch_count_{wp}_{s}"] = sup_touch_count
        cols[f"{p}_resistance_violation_count_{wp}_{s}"] = res_violation_count
        cols[f"{p}_support_violation_count_{wp}_{s}"] = sup_violation_count
        cols[f"{p}_resistance_respect_ratio_{wp}_{s}"] = _safe_div(res_touch_count, res_touch_count + res_violation_count)
        cols[f"{p}_support_respect_ratio_{wp}_{s}"] = _safe_div(sup_touch_count, sup_touch_count + sup_violation_count)

    score_cols = _score_columns(
        prefix=p,
        strength=s,
        trend_bias=trend_bias,
        channel_valid=channel_valid,
        channel_compression=channel_compression,
        support_reclaim=support_reclaim,
        resistance_reject=resistance_reject,
        resistance_violation=resistance_violation,
        support_violation=support_violation,
        resistance_breakout=cols.get(f"{p}_resistance_breakout_1_{s}", pd.Series(False, index=frame.index)),
        support_breakdown=cols.get(f"{p}_support_breakdown_1_{s}", pd.Series(False, index=frame.index)),
        support_touch=support_touch,
        resistance_touch=resistance_touch,
        cfg=cfg,
    )
    cols.update(score_cols)
    return cols


def _anchor_state(frame: DataFrame, prefix: str, strength: int, side: Literal["high", "low"], bar_index: Series) -> dict[str, Series]:
    event_price = _num(frame, f"{prefix}_pivot_{side}_{strength}")
    event_index = (bar_index - float(strength)).where(event_price.notna())
    event_prominence = _num(frame, f"{prefix}_pivot_{side}_prominence_{strength}")

    last_price = event_price.ffill()
    prev_price = event_price.dropna().shift(1).reindex(frame.index).ffill()
    last_index = event_index.ffill()
    prev_index = event_index.dropna().shift(1).reindex(frame.index).ffill()
    last_prom = event_prominence.ffill()
    prev_prom = event_prominence.dropna().shift(1).reindex(frame.index).ffill()
    span = last_index - prev_index
    valid_span = span.gt(0.0)
    slope = ((last_price - prev_price) / span).where(valid_span)
    age = bar_index - last_index
    prominence_min = pd.concat([last_prom, prev_prom], axis=1).min(axis=1)
    return {
        "last_price": last_price,
        "prev_price": prev_price,
        "last_index": last_index,
        "prev_index": prev_index,
        "span": span,
        "slope": slope,
        "age": age,
        "prominence_min": prominence_min,
    }


def _line_from_anchor(bar_index: Series, last_price: Series, last_index: Series, slope: Series) -> Series:
    return last_price + slope * (bar_index - last_index)


def _line_validity(anchor: dict[str, Series], slope: Series, line: Series, close: Series, cfg: TrendlineProjectionConfig) -> Series:
    slope_pct = (slope.abs() / close).replace([np.inf, -np.inf], np.nan)
    return (
        line.notna()
        & anchor["last_price"].notna()
        & anchor["prev_price"].notna()
        & anchor["span"].ge(float(cfg.min_anchor_span_bars))
        & anchor["age"].le(float(cfg.max_anchor_age_bars))
        & anchor["prominence_min"].fillna(0.0).ge(float(cfg.min_anchor_prominence_atr))
        & slope_pct.le(float(cfg.max_line_slope_pct_per_bar))
        & _projection_sane(line, close, cfg)
    ).fillna(False)


def _projection_sane(level: Series, close: Series, cfg: TrendlineProjectionConfig) -> Series:
    distance = ((level - close).abs() / close).replace([np.inf, -np.inf], np.nan)
    return level.gt(0.0) & distance.le(float(cfg.max_projection_distance_pct))


def _plot_safe_line(line: Series, break_pct: float) -> Series:
    plotted = pd.to_numeric(line, errors="coerce").copy()
    if float(break_pct) >= 100.0:
        return plotted
    prior = plotted.shift(1)
    jump = ((plotted - prior).abs() / prior.abs().replace(0.0, np.nan)).gt(float(break_pct))
    starts_after_gap = plotted.notna() & prior.isna()
    return plotted.mask(jump | starts_after_gap)


def _touches_line(high: Series, low: Series, level: Series, zone_width: Series) -> Series:
    return level.notna() & high.ge(level - zone_width) & low.le(level + zone_width)


def _near_level(price: Series, level: Series, zone_width: Series) -> Series:
    return level.notna() & price.ge(level - zone_width) & price.le(level + zone_width)


def _rolling_count(mask: Series, window: int) -> Series:
    clean = pd.Series(mask, index=mask.index).fillna(False).astype("int8")
    cumulative = clean.cumsum()
    return (cumulative - cumulative.shift(int(window), fill_value=0)).astype("float64")


def _score_columns(
    *,
    prefix: str,
    strength: int,
    trend_bias: Series,
    channel_valid: Series,
    channel_compression: Series,
    support_reclaim: Series,
    resistance_reject: Series,
    resistance_violation: Series,
    support_violation: Series,
    resistance_breakout: Series,
    support_breakdown: Series,
    support_touch: Series,
    resistance_touch: Series,
    cfg: TrendlineProjectionConfig,
) -> dict[str, Series]:
    s = int(strength)
    w = int(cfg.score_window)
    support_touches = _rolling_count(support_touch, w)
    resistance_touches = _rolling_count(resistance_touch, w)
    support_violations = _rolling_count(support_violation, w)
    resistance_violations = _rolling_count(resistance_violation, w)
    support_respect = _safe_div(support_touches, support_touches + support_violations).fillna(0.0)
    resistance_respect = _safe_div(resistance_touches, resistance_touches + resistance_violations).fillna(0.0)
    support_quality = _clip01(
        0.55 * _clip01(support_touches / max(float(cfg.min_touch_count), 1.0))
        + 0.45 * _clip01((support_respect - float(cfg.min_respect_ratio)) / max(1.0 - float(cfg.min_respect_ratio), 1e-9))
    )
    resistance_quality = _clip01(
        0.55 * _clip01(resistance_touches / max(float(cfg.min_touch_count), 1.0))
        + 0.45 * _clip01((resistance_respect - float(cfg.min_respect_ratio)) / max(1.0 - float(cfg.min_respect_ratio), 1e-9))
    )
    compression_quality = _clip01(1.0 - channel_compression.fillna(1.0))
    valid = channel_valid.astype("float64")

    long_score = _clip01(
        0.25 * trend_bias.gt(0.0).astype("float64")
        + 0.25 * support_quality
        + 0.20 * _rolling_count(support_reclaim, w).clip(0.0, 1.0)
        + 0.20 * _rolling_count(resistance_breakout, w).clip(0.0, 1.0)
        + 0.10 * compression_quality
    ) * valid
    short_score = _clip01(
        0.25 * trend_bias.lt(0.0).astype("float64")
        + 0.25 * resistance_quality
        + 0.20 * _rolling_count(resistance_reject, w).clip(0.0, 1.0)
        + 0.20 * _rolling_count(support_breakdown, w).clip(0.0, 1.0)
        + 0.10 * compression_quality
    ) * valid
    abs_score = pd.concat([long_score, short_score], axis=1).max(axis=1)
    score_state = pd.Series(
        np.select([long_score.gt(short_score), short_score.gt(long_score)], [1.0, -1.0], default=0.0),
        index=long_score.index,
    )
    return {
        f"{prefix}_score_long_{s}": long_score,
        f"{prefix}_score_short_{s}": short_score,
        f"{prefix}_score_abs_{s}": abs_score,
        f"{prefix}_state_{s}": score_state,
        f"{prefix}_support_quality_{s}": support_quality,
        f"{prefix}_resistance_quality_{s}": resistance_quality,
    }


def _safe_div(numerator: Series, denominator: Series) -> Series:
    return numerator / denominator.replace(0.0, np.nan)


def _clip01(series: Series) -> Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0).fillna(0.0)


def _active_alias_columns(cols: dict[str, Series], prefix: str, strength: int, horizons: list[int], windows: list[int]) -> dict[str, Series]:
    aliases = (
        "resistance_line", "support_line", "resistance_plot", "support_plot", "channel_mid", "channel_mid_plot",
        "resistance_slope", "support_slope", "resistance_slope_pct", "support_slope_pct", "slope_mean_pct",
        "parallel_error_pct", "channel_valid", "channel_width_pct", "channel_compression", "trend_bias",
        "resistance_line_valid", "support_line_valid", "resistance_anchor_span", "support_anchor_span",
        "resistance_age", "support_age", "resistance_anchor_prominence_min", "support_anchor_prominence_min",
        "last_pivot_high", "last_pivot_low", "prev_pivot_high", "prev_pivot_low",
        "resistance_touch", "support_touch", "resistance_reject", "support_reclaim",
        "resistance_violation", "support_violation",
        "score_long", "score_short", "score_abs", "state", "support_quality", "resistance_quality",
    )
    out: dict[str, Series] = {}
    for alias in aliases:
        src = f"{prefix}_{alias}_{strength}"
        if src in cols:
            out[f"{prefix}_{alias}"] = cols[src]
    for h in horizons:
        for alias in (
            "resistance_proj", "support_proj", "resistance_proj_plot", "support_proj_plot", "channel_mid_proj",
            "channel_width_proj_pct", "resistance_near", "support_near", "long_target_distance_pct",
            "short_target_distance_pct", "resistance_breakout", "support_breakdown",
        ):
            src = f"{prefix}_{alias}_{h}_{strength}"
            if src in cols:
                out[f"{prefix}_{alias}_{h}"] = cols[src]
    for w in windows:
        for alias in (
            "resistance_touch_count", "support_touch_count", "resistance_violation_count",
            "support_violation_count", "resistance_respect_ratio", "support_respect_ratio",
        ):
            src = f"{prefix}_{alias}_{w}_{strength}"
            if src in cols:
                out[f"{prefix}_{alias}_{w}"] = cols[src]
    return out


def _ensure_pivot_columns(frame: DataFrame, cfg: TrendlineProjectionConfig) -> DataFrame:
    required_any = f"{cfg.pivot_prefix}_pivot_high_{_sorted_unique_ints(cfg.strengths)[0]}"
    if required_any in frame.columns:
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
    raise KeyError(f"Missing pivot columns with prefix '{cfg.pivot_prefix}'. Run add_pivot_structure first or use missing_pivot_mode='compute'.")


def _resolve_config(config: TrendlineProjectionConfig | None, **overrides: object) -> TrendlineProjectionConfig:
    base = config or TrendlineProjectionConfig()
    values = dict(base.__dict__)
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return TrendlineProjectionConfig(**values)


def _validate_config(cfg: TrendlineProjectionConfig) -> None:
    if int(cfg.strength) <= 0:
        raise ValueError("strength must be positive")
    if not _sorted_strengths(cfg):
        raise ValueError("strengths must contain at least one value")
    if not _sorted_unique_ints(cfg.horizons):
        raise ValueError("horizons must contain at least one value")
    if not _sorted_unique_ints(cfg.event_windows):
        raise ValueError("event_windows must contain at least one value")
    if cfg.near_zone_atr_mult < 0.0 or cfg.near_zone_pct < 0.0:
        raise ValueError("zone settings must be non-negative")
    if cfg.max_line_slope_pct_per_bar <= 0.0:
        raise ValueError("max_line_slope_pct_per_bar must be positive")
    if cfg.compression_window < 2:
        raise ValueError("compression_window must be at least 2")
    if cfg.compression_min_periods < 1 or cfg.compression_min_periods > cfg.compression_window:
        raise ValueError("compression_min_periods must be between 1 and compression_window")
    if cfg.score_window < 2:
        raise ValueError("score_window must be at least 2")
    if cfg.min_touch_count < 1:
        raise ValueError("min_touch_count must be at least 1")
    if not 0.0 <= cfg.min_respect_ratio <= 1.0:
        raise ValueError("min_respect_ratio must be between 0.0 and 1.0")


def _validate_dataframe(dataframe: DataFrame) -> None:
    missing = [col for col in ("open", "high", "low", "close") if col not in dataframe.columns]
    if missing:
        raise KeyError(f"Missing required OHLC columns: {', '.join(missing)}")


def _sorted_unique_ints(values: Sequence[int]) -> list[int]:
    return sorted({int(v) for v in values if int(v) > 0})


def _sorted_strengths(cfg: TrendlineProjectionConfig) -> list[int]:
    values = {int(v) for v in cfg.strengths if int(v) > 0}
    values.add(int(cfg.strength))
    return sorted(values)


def _num(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


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

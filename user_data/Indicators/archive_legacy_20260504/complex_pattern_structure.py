from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


@dataclass(frozen=True)
class PatternStructureConfig:
    """Geometry-backed chart-pattern evidence for Freqtrade.

    This module consumes confirmed pivots plus tactical/structural trendlines.
    It keeps broad impulse/consolidation evidence for diagnostics, but named
    patterns such as flags, pennants, wedges, triangles, double tops/bottoms,
    and head-and-shoulders require ordered pivot geometry.

    Scores are normalized 0..1 evidence values only. Strategies own the final
    entry, exit, stake, and risk decisions.
    """

    impulse_window: int = 24
    consolidation_window: int = 18
    atr_period: int = 14
    impulse_atr_min: float = 2.0
    impulse_pct_min: float = 0.025
    impulse_close_location_min: float = 0.65
    max_retrace_pct: float = 0.62
    max_consolidation_extension_pct: float = 0.018
    max_consolidation_drift_atr: float = 1.25
    min_range_contraction: float = 0.15
    dry_volume_rvol_max: float = 0.90
    breakout_buffer_pct: float = 0.001
    breakout_zone_fraction: float = 0.25

    line_zone_atr_mult: float = 0.45
    line_zone_pct: float = 0.004
    min_line_score: float = 0.28
    min_line_touch_count: int = 2
    min_line_respect_ratio: float = 0.45
    line_respect_window: int = 48
    structural_line_score_margin: float = 0.04
    flat_slope_pct_per_bar: float = 0.00045
    parallel_slope_tolerance_pct: float = 0.00075
    min_converging_slope_gap_pct: float = 0.00025
    min_channel_compression: float = 0.22
    min_sequence_containment_ratio: float = 0.55
    min_sequence_touch_count: int = 5
    max_flag_forward_slope_pct: float = 0.00055
    max_wedge_parallel_error_pct: float = 0.0012

    pivot_similarity_pct: float = 0.018
    shoulder_similarity_pct: float = 0.035
    head_prominence_pct: float = 0.025
    max_neckline_slope_pct_per_bar: float = 0.012
    min_pivot_pattern_depth_pct: float = 0.025
    min_sequence_span_bars: int = 8
    max_sequence_span_bars: int = 240
    max_pivot_pattern_age_bars: int = 96

    pattern_memory_window: int = 36
    entry_cooldown_bars: int = 6
    pivot_prefix: str = "pa"
    trendline_prefix: str = "tl"
    structural_trendline_prefix: str = "stl"
    output_prefix: str = "pat"


def add_pattern_structure(
    dataframe: DataFrame,
    config: PatternStructureConfig | None = None,
    *,
    impulse_window: int | None = None,
    consolidation_window: int | None = None,
    atr_period: int | None = None,
    impulse_atr_min: float | None = None,
    impulse_pct_min: float | None = None,
    impulse_close_location_min: float | None = None,
    max_retrace_pct: float | None = None,
    max_consolidation_extension_pct: float | None = None,
    max_consolidation_drift_atr: float | None = None,
    min_range_contraction: float | None = None,
    dry_volume_rvol_max: float | None = None,
    breakout_buffer_pct: float | None = None,
    breakout_zone_fraction: float | None = None,
    line_zone_atr_mult: float | None = None,
    line_zone_pct: float | None = None,
    min_line_score: float | None = None,
    min_line_touch_count: int | None = None,
    min_line_respect_ratio: float | None = None,
    line_respect_window: int | None = None,
    structural_line_score_margin: float | None = None,
    flat_slope_pct_per_bar: float | None = None,
    parallel_slope_tolerance_pct: float | None = None,
    min_converging_slope_gap_pct: float | None = None,
    min_channel_compression: float | None = None,
    min_sequence_containment_ratio: float | None = None,
    min_sequence_touch_count: int | None = None,
    max_flag_forward_slope_pct: float | None = None,
    max_wedge_parallel_error_pct: float | None = None,
    pivot_similarity_pct: float | None = None,
    shoulder_similarity_pct: float | None = None,
    head_prominence_pct: float | None = None,
    max_neckline_slope_pct_per_bar: float | None = None,
    min_pivot_pattern_depth_pct: float | None = None,
    min_sequence_span_bars: int | None = None,
    max_sequence_span_bars: int | None = None,
    max_pivot_pattern_age_bars: int | None = None,
    pattern_memory_window: int | None = None,
    entry_cooldown_bars: int | None = None,
    pivot_prefix: str | None = None,
    trendline_prefix: str | None = None,
    structural_trendline_prefix: str | None = None,
    output_prefix: str | None = None,
) -> DataFrame:
    """Append pattern setup, breakout/breakdown, invalidation, and score columns."""

    cfg = _resolve_config(
        config,
        impulse_window=impulse_window,
        consolidation_window=consolidation_window,
        atr_period=atr_period,
        impulse_atr_min=impulse_atr_min,
        impulse_pct_min=impulse_pct_min,
        impulse_close_location_min=impulse_close_location_min,
        max_retrace_pct=max_retrace_pct,
        max_consolidation_extension_pct=max_consolidation_extension_pct,
        max_consolidation_drift_atr=max_consolidation_drift_atr,
        min_range_contraction=min_range_contraction,
        dry_volume_rvol_max=dry_volume_rvol_max,
        breakout_buffer_pct=breakout_buffer_pct,
        breakout_zone_fraction=breakout_zone_fraction,
        line_zone_atr_mult=line_zone_atr_mult,
        line_zone_pct=line_zone_pct,
        min_line_score=min_line_score,
        min_line_touch_count=min_line_touch_count,
        min_line_respect_ratio=min_line_respect_ratio,
        line_respect_window=line_respect_window,
        structural_line_score_margin=structural_line_score_margin,
        flat_slope_pct_per_bar=flat_slope_pct_per_bar,
        parallel_slope_tolerance_pct=parallel_slope_tolerance_pct,
        min_converging_slope_gap_pct=min_converging_slope_gap_pct,
        min_channel_compression=min_channel_compression,
        min_sequence_containment_ratio=min_sequence_containment_ratio,
        min_sequence_touch_count=min_sequence_touch_count,
        max_flag_forward_slope_pct=max_flag_forward_slope_pct,
        max_wedge_parallel_error_pct=max_wedge_parallel_error_pct,
        pivot_similarity_pct=pivot_similarity_pct,
        shoulder_similarity_pct=shoulder_similarity_pct,
        head_prominence_pct=head_prominence_pct,
        max_neckline_slope_pct_per_bar=max_neckline_slope_pct_per_bar,
        min_pivot_pattern_depth_pct=min_pivot_pattern_depth_pct,
        min_sequence_span_bars=min_sequence_span_bars,
        max_sequence_span_bars=max_sequence_span_bars,
        max_pivot_pattern_age_bars=max_pivot_pattern_age_bars,
        pattern_memory_window=pattern_memory_window,
        entry_cooldown_bars=entry_cooldown_bars,
        pivot_prefix=pivot_prefix,
        trendline_prefix=trendline_prefix,
        structural_trendline_prefix=structural_trendline_prefix,
        output_prefix=output_prefix,
    )
    _validate_config(cfg)
    _validate_dataframe(dataframe)

    frame = dataframe.copy()
    p = cfg.output_prefix
    close = _num(frame, "close").replace(0.0, np.nan)
    high = _num(frame, "high")
    low = _num(frame, "low")
    open_ = _num(frame, "open")
    volume = _num(frame, "volume").clip(lower=0.0).fillna(0.0)
    atr = _atr(frame, cfg.atr_period).replace(0.0, np.nan)
    bar_index = pd.Series(np.arange(len(frame), dtype="float64"), index=frame.index)
    zone_width = pd.concat([atr * cfg.line_zone_atr_mult, close.abs() * cfg.line_zone_pct], axis=1).max(axis=1)

    impulse = _impulse_consolidation_columns(frame, close, high, low, open_, volume, atr, cfg)
    lines = _selected_line_columns(frame, close, high, low, atr, zone_width, cfg)
    pivots = _pivot_history_columns(frame, bar_index, cfg)

    line_patterns = _line_pattern_columns(close, high, low, impulse, lines, pivots, zone_width, bar_index, cfg, p)
    pivot_patterns = _pivot_pattern_columns(close, high, low, pivots, zone_width, bar_index, cfg, p)

    setup_long_names = [name for name in (*line_patterns, *pivot_patterns) if name.endswith("_setup_long")]
    setup_short_names = [name for name in (*line_patterns, *pivot_patterns) if name.endswith("_setup_short")]
    trigger_long_names = [
        name
        for name in (*line_patterns, *pivot_patterns)
        if name.endswith("_breakout_long") and name != f"{p}_line_breakout_long"
    ]
    trigger_short_names = [
        name
        for name in (*line_patterns, *pivot_patterns)
        if name.endswith("_breakdown_short") and name != f"{p}_line_breakdown_short"
    ]
    invalid_long_names = [name for name in (*line_patterns, *pivot_patterns) if name.endswith("_invalid_long")]
    invalid_short_names = [name for name in (*line_patterns, *pivot_patterns) if name.endswith("_invalid_short")]
    quality_long_names = [name for name in (*line_patterns, *pivot_patterns) if name.endswith("_quality_long")]
    quality_short_names = [name for name in (*line_patterns, *pivot_patterns) if name.endswith("_quality_short")]

    setup_long = _any_columns(line_patterns | pivot_patterns, setup_long_names, frame.index)
    setup_short = _any_columns(line_patterns | pivot_patterns, setup_short_names, frame.index)
    trigger_long = _any_columns(line_patterns | pivot_patterns, trigger_long_names, frame.index)
    trigger_short = _any_columns(line_patterns | pivot_patterns, trigger_short_names, frame.index)
    invalid_long = _any_columns(line_patterns | pivot_patterns, invalid_long_names, frame.index)
    invalid_short = _any_columns(line_patterns | pivot_patterns, invalid_short_names, frame.index)
    pattern_quality_long = _max_columns(line_patterns | pivot_patterns, quality_long_names, frame.index)
    pattern_quality_short = _max_columns(line_patterns | pivot_patterns, quality_short_names, frame.index)

    recent_setup_long = _rolling_count(setup_long, cfg.pattern_memory_window).ge(1.0)
    recent_setup_short = _rolling_count(setup_short, cfg.pattern_memory_window).ge(1.0)
    broad_quality_long = impulse[f"{p}_consolidation_quality_long"]
    broad_quality_short = impulse[f"{p}_consolidation_quality_short"]

    score_long = _clip01(
        0.70 * pattern_quality_long
        + 0.15 * broad_quality_long
        + 0.10 * trigger_long.astype("float64")
        + 0.05 * recent_setup_long.astype("float64")
    )
    score_short = _clip01(
        0.70 * pattern_quality_short
        + 0.15 * broad_quality_short
        + 0.10 * trigger_short.astype("float64")
        + 0.05 * recent_setup_short.astype("float64")
    )
    abs_score = pd.concat([score_long, score_short], axis=1).max(axis=1)
    state = pd.Series(
        np.select([score_long.gt(score_short), score_short.gt(score_long)], [1.0, -1.0], default=0.0),
        index=frame.index,
    )
    margin = score_long - score_short
    full_bull = score_long.ge(0.52) & margin.ge(0.08) & (recent_setup_long | trigger_long)
    full_bear = score_short.ge(0.52) & margin.le(-0.08) & (recent_setup_short | trigger_short)
    bullish_chop = ~full_bull & ~full_bear & score_long.ge(0.34) & margin.ge(0.04) & recent_setup_long
    bearish_chop = ~full_bull & ~full_bear & score_short.ge(0.34) & margin.le(-0.04) & recent_setup_short
    market_context = pd.Series(
        np.select([full_bull, full_bear, bullish_chop, bearish_chop], [2, -2, 1, -1], default=0),
        index=frame.index,
        dtype="int8",
    )

    suggested_entry_long = _dedupe_events(trigger_long, cfg.entry_cooldown_bars)
    suggested_entry_short = _dedupe_events(trigger_short, cfg.entry_cooldown_bars)
    hold_long = recent_setup_long & score_long.ge(score_short - 0.05) & ~invalid_long & ~trigger_short
    hold_short = recent_setup_short & score_short.ge(score_long - 0.05) & ~invalid_short & ~trigger_long
    exit_long = _dedupe_events(invalid_long | trigger_short, cfg.entry_cooldown_bars)
    exit_short = _dedupe_events(invalid_short | trigger_long, cfg.entry_cooldown_bars)

    new_cols: dict[str, Series] = {}
    new_cols.update(impulse)
    new_cols.update(lines)
    new_cols.update(pivots)
    new_cols.update(line_patterns)
    new_cols.update(pivot_patterns)
    new_cols.update(
        {
            f"{p}_setup_any_long": setup_long.fillna(False),
            f"{p}_setup_any_short": setup_short.fillna(False),
            f"{p}_invalid_any_long": invalid_long.fillna(False),
            f"{p}_invalid_any_short": invalid_short.fillna(False),
            f"{p}_pattern_quality_long": pattern_quality_long,
            f"{p}_pattern_quality_short": pattern_quality_short,
            f"{p}_hold_long": hold_long.fillna(False),
            f"{p}_hold_short": hold_short.fillna(False),
            f"{p}_exit_long": exit_long,
            f"{p}_exit_short": exit_short,
            f"{p}_suggested_entry_long": suggested_entry_long,
            f"{p}_suggested_entry_short": suggested_entry_short,
            f"{p}_market_context": market_context,
            f"{p}_score_long": score_long,
            f"{p}_score_short": score_short,
            f"{p}_score_abs": abs_score,
            f"{p}_state": state,
        }
    )
    new_cols.update(_compatibility_columns(new_cols, p, cfg, close, high, low))

    existing = [col for col in frame.columns if str(col).startswith(f"{p}_")]
    base = frame.drop(columns=existing).copy() if existing else frame.copy()
    return pd.concat([base, pd.DataFrame(new_cols, index=frame.index)], axis=1)


def _impulse_consolidation_columns(
    frame: DataFrame,
    close: Series,
    high: Series,
    low: Series,
    open_: Series,
    volume: Series,
    atr: Series,
    cfg: PatternStructureConfig,
) -> dict[str, Series]:
    impulse_low = low.rolling(cfg.impulse_window, min_periods=2).min().shift(cfg.consolidation_window)
    impulse_high = high.rolling(cfg.impulse_window, min_periods=2).max().shift(cfg.consolidation_window)
    impulse_start_close = close.shift(cfg.impulse_window + cfg.consolidation_window)
    impulse_end_close = close.shift(cfg.consolidation_window)
    impulse_range = impulse_high - impulse_low
    impulse_close_location = _safe_div(impulse_end_close - impulse_low, impulse_range).clip(0.0, 1.0)
    impulse_net_move = impulse_end_close - impulse_start_close
    impulse_up_move = impulse_end_close - impulse_low
    impulse_down_move = impulse_high - impulse_end_close
    impulse_up_valid = (
        impulse_net_move.gt(0.0)
        & impulse_up_move.gt(0.0)
        & impulse_close_location.ge(cfg.impulse_close_location_min)
    )
    impulse_down_valid = (
        impulse_net_move.lt(0.0)
        & impulse_down_move.gt(0.0)
        & impulse_close_location.le(1.0 - cfg.impulse_close_location_min)
    )
    impulse_up_score = _clip01(
        0.45 * _safe_div(impulse_net_move, atr * cfg.impulse_atr_min)
        + 0.40 * _safe_div(impulse_net_move, close * cfg.impulse_pct_min)
        + 0.15 * impulse_close_location
    ).where(impulse_up_valid, 0.0)
    impulse_down_score = _clip01(
        0.45 * _safe_div(-impulse_net_move, atr * cfg.impulse_atr_min)
        + 0.40 * _safe_div(-impulse_net_move, close * cfg.impulse_pct_min)
        + 0.15 * (1.0 - impulse_close_location)
    ).where(impulse_down_valid, 0.0)

    cons_high = high.rolling(cfg.consolidation_window, min_periods=2).max()
    cons_low = low.rolling(cfg.consolidation_window, min_periods=2).min()
    cons_range = cons_high - cons_low
    prior_range = (
        high.rolling(cfg.impulse_window, min_periods=2).max()
        - low.rolling(cfg.impulse_window, min_periods=2).min()
    ).shift(cfg.consolidation_window)
    range_contraction = _clip01(1.0 - _safe_div(cons_range, prior_range))
    retrace_long = _safe_div(impulse_end_close - cons_low, impulse_net_move.abs())
    retrace_short = _safe_div(cons_high - impulse_end_close, impulse_net_move.abs())
    controlled_long = retrace_long.between(0.0, cfg.max_retrace_pct)
    controlled_short = retrace_short.between(0.0, cfg.max_retrace_pct)
    extension_long = _safe_div((cons_high - impulse_end_close).clip(lower=0.0), impulse_end_close)
    extension_short = _safe_div((impulse_end_close - cons_low).clip(lower=0.0), impulse_end_close)
    consolidation_drift_atr = _safe_div(close - impulse_end_close, atr)
    short_volume = volume.rolling(cfg.consolidation_window, min_periods=1).mean()
    long_volume = volume.rolling(cfg.impulse_window + cfg.consolidation_window, min_periods=2).mean()
    dry_volume = _safe_div(short_volume, long_volume) <= cfg.dry_volume_rvol_max
    contraction_ok = range_contraction >= cfg.min_range_contraction

    consolidation_setup_long = (
        impulse_up_valid
        & impulse_up_score.gt(0.5)
        & controlled_long
        & extension_long.le(cfg.max_consolidation_extension_pct)
        & consolidation_drift_atr.le(cfg.max_consolidation_drift_atr)
        & contraction_ok
        & dry_volume
    )
    consolidation_setup_short = (
        impulse_down_valid
        & impulse_down_score.gt(0.5)
        & controlled_short
        & extension_short.le(cfg.max_consolidation_extension_pct)
        & consolidation_drift_atr.ge(-cfg.max_consolidation_drift_atr)
        & contraction_ok
        & dry_volume
    )
    raw_breakout_long = close.gt(cons_high.shift(1) * (1.0 + cfg.breakout_buffer_pct)) & close.gt(open_)
    raw_breakdown_short = close.lt(cons_low.shift(1) * (1.0 - cfg.breakout_buffer_pct)) & close.lt(open_)
    recent_consolidation_long = _rolling_count(consolidation_setup_long, cfg.consolidation_window).ge(1.0)
    recent_consolidation_short = _rolling_count(consolidation_setup_short, cfg.consolidation_window).ge(1.0)
    consolidation_quality_long = _clip01(
        0.42 * impulse_up_score
        + 0.28 * range_contraction
        + 0.16 * dry_volume.astype("float64")
        + 0.14 * _clip01(1.0 - retrace_long / max(cfg.max_retrace_pct, 1e-9))
    ).where(consolidation_setup_long | recent_consolidation_long, 0.0)
    consolidation_quality_short = _clip01(
        0.42 * impulse_down_score
        + 0.28 * range_contraction
        + 0.16 * dry_volume.astype("float64")
        + 0.14 * _clip01(1.0 - retrace_short / max(cfg.max_retrace_pct, 1e-9))
    ).where(consolidation_setup_short | recent_consolidation_short, 0.0)
    return {
        "pat_impulse_up_score": impulse_up_score,
        "pat_impulse_down_score": impulse_down_score,
        "pat_impulse_direction": pd.Series(
            np.select([impulse_up_valid, impulse_down_valid], [1, -1], default=0),
            index=frame.index,
            dtype="int8",
        ),
        "pat_impulse_close_location": impulse_close_location,
        "pat_impulse_end_close": impulse_end_close,
        "pat_impulse_net_move": impulse_net_move,
        "pat_impulse_up_move": impulse_up_move.where(impulse_up_valid),
        "pat_impulse_down_move": impulse_down_move.where(impulse_down_valid),
        "pat_consolidation_high": cons_high,
        "pat_consolidation_low": cons_low,
        "pat_range_contraction_score": range_contraction,
        "pat_retrace_long": retrace_long.where(impulse_up_valid),
        "pat_retrace_short": retrace_short.where(impulse_down_valid),
        "pat_consolidation_extension_long": extension_long.where(impulse_up_valid),
        "pat_consolidation_extension_short": extension_short.where(impulse_down_valid),
        "pat_consolidation_drift_atr": consolidation_drift_atr,
        "pat_dry_volume": dry_volume,
        "pat_consolidation_setup_long": consolidation_setup_long.fillna(False),
        "pat_consolidation_setup_short": consolidation_setup_short.fillna(False),
        "pat_consolidation_breakout_long": (raw_breakout_long & recent_consolidation_long).fillna(False),
        "pat_consolidation_breakdown_short": (raw_breakdown_short & recent_consolidation_short).fillna(False),
        "pat_consolidation_quality_long": consolidation_quality_long,
        "pat_consolidation_quality_short": consolidation_quality_short,
    }


def _selected_line_columns(
    frame: DataFrame,
    close: Series,
    high: Series,
    low: Series,
    atr: Series,
    zone_width: Series,
    cfg: PatternStructureConfig,
) -> dict[str, Series]:
    tl = cfg.trendline_prefix
    stl = cfg.structural_trendline_prefix
    tl_valid = _bool(frame, f"{tl}_channel_valid")
    stl_valid = _bool(frame, f"{stl}_channel_valid")
    tl_score = pd.concat(
        [
            _num(frame, f"{tl}_support_quality"),
            _num(frame, f"{tl}_resistance_quality"),
            _num(frame, f"{tl}_score_abs"),
        ],
        axis=1,
    ).max(axis=1).fillna(0.0)
    stl_score = pd.concat(
        [
            _num(frame, f"{stl}_support_score"),
            _num(frame, f"{stl}_resistance_score"),
            _num(frame, f"{stl}_score_abs"),
        ],
        axis=1,
    ).max(axis=1).fillna(0.0)
    use_stl = stl_valid & (~tl_valid | stl_score.ge(tl_score + cfg.structural_line_score_margin))

    def choose(tactical_name: str, structural_name: str | None = None, default: float = np.nan) -> Series:
        structural = structural_name or tactical_name
        tactical_series = _num(frame, f"{tl}_{tactical_name}")
        structural_series = _num(frame, f"{stl}_{structural}")
        chosen = tactical_series.where(~use_stl, structural_series)
        return chosen.fillna(default) if np.isfinite(default) else chosen

    resistance_line = choose("resistance_line")
    support_line = choose("support_line")
    resistance_slope_pct = choose("resistance_slope_pct", default=0.0)
    support_slope_pct = choose("support_slope_pct", default=0.0)
    channel_width_pct = _safe_div(resistance_line - support_line, close)
    channel_compression_ratio = choose("channel_compression", default=1.0)
    channel_compression_score = _clip01(1.0 - channel_compression_ratio)

    resistance_quality = choose("resistance_quality", "resistance_score", default=0.0)
    support_quality = choose("support_quality", "support_score", default=0.0)
    resistance_rank_quality = choose("resistance_score_rank0", default=0.0)
    support_rank_quality = choose("support_score_rank0", default=0.0)
    resistance_quality = pd.concat([resistance_quality, resistance_rank_quality], axis=1).max(axis=1).fillna(0.0)
    support_quality = pd.concat([support_quality, support_rank_quality], axis=1).max(axis=1).fillna(0.0)
    line_quality = pd.concat([resistance_quality, support_quality], axis=1).min(axis=1).fillna(0.0)

    resistance_touches = choose("resistance_touch_count_rank0", default=0.0)
    support_touches = choose("support_touch_count_rank0", default=0.0)
    resistance_respect = choose(f"resistance_respect_ratio_{cfg.line_respect_window}", "resistance_respect_ratio_rank0", default=1.0)
    support_respect = choose(f"support_respect_ratio_{cfg.line_respect_window}", "support_respect_ratio_rank0", default=1.0)
    resistance_touch_now = _touches_line(high, low, resistance_line, zone_width)
    support_touch_now = _touches_line(high, low, support_line, zone_width)
    resistance_excursion = resistance_line.notna() & close.gt(resistance_line + zone_width)
    support_excursion = support_line.notna() & close.lt(support_line - zone_width)

    channel_valid = (
        (tl_valid | use_stl)
        & resistance_line.gt(support_line)
        & line_quality.ge(cfg.min_line_score)
        & resistance_touches.ge(float(cfg.min_line_touch_count))
        & support_touches.ge(float(cfg.min_line_touch_count))
        & resistance_respect.ge(cfg.min_line_respect_ratio)
        & support_respect.ge(cfg.min_line_respect_ratio)
    ).fillna(False)
    source = pd.Series(np.select([use_stl, tl_valid], [2, 1], default=0), index=frame.index, dtype="int8")
    return {
        "pat_line_source": source,
        "pat_line_zone_width": zone_width,
        "pat_resistance_line": resistance_line.where(channel_valid),
        "pat_support_line": support_line.where(channel_valid),
        "pat_resistance_slope_pct": resistance_slope_pct.where(channel_valid),
        "pat_support_slope_pct": support_slope_pct.where(channel_valid),
        "pat_channel_width_pct": channel_width_pct.where(channel_valid),
        "pat_channel_compression_score": channel_compression_score.where(channel_valid, 0.0),
        "pat_line_quality": line_quality.where(channel_valid, 0.0),
        "pat_resistance_quality": resistance_quality.where(channel_valid, 0.0),
        "pat_support_quality": support_quality.where(channel_valid, 0.0),
        "pat_resistance_touch_count": resistance_touches.where(channel_valid, 0.0),
        "pat_support_touch_count": support_touches.where(channel_valid, 0.0),
        "pat_resistance_respect_ratio": resistance_respect.where(channel_valid),
        "pat_support_respect_ratio": support_respect.where(channel_valid),
        "pat_resistance_touch": resistance_touch_now & channel_valid,
        "pat_support_touch": support_touch_now & channel_valid,
        "pat_resistance_excursion": resistance_excursion & channel_valid,
        "pat_support_excursion": support_excursion & channel_valid,
        "pat_channel_valid": channel_valid,
    }


def _line_pattern_columns(
    close: Series,
    high: Series,
    low: Series,
    impulse: dict[str, Series],
    lines: dict[str, Series],
    pivots: dict[str, Series],
    zone_width: Series,
    bar_index: Series,
    cfg: PatternStructureConfig,
    prefix: str,
) -> dict[str, Series]:
    resistance = lines["pat_resistance_line"]
    support = lines["pat_support_line"]
    res_slope = lines["pat_resistance_slope_pct"].fillna(0.0)
    sup_slope = lines["pat_support_slope_pct"].fillna(0.0)
    line_valid = lines["pat_channel_valid"].fillna(False)
    line_quality = lines["pat_line_quality"].fillna(0.0)
    compression = lines["pat_channel_compression_score"].fillna(0.0)
    breakout_width = _breakout_width(close, zone_width, cfg)
    touch_score = _clip01(
        (
            lines["pat_resistance_touch_count"].fillna(0.0)
            + lines["pat_support_touch_count"].fillna(0.0)
        )
        / max(float(cfg.min_line_touch_count) * 3.0, 1.0)
    )
    line_shape_quality = _clip01(0.55 * line_quality + 0.25 * touch_score + 0.20 * compression)
    recent_long_base = _rolling_count(impulse["pat_consolidation_setup_long"], cfg.pattern_memory_window).ge(1.0)
    recent_short_base = _rolling_count(impulse["pat_consolidation_setup_short"], cfg.pattern_memory_window).ge(1.0)
    res_breakout = _fresh_cross_above(close, resistance + breakout_width)
    sup_breakdown = _fresh_cross_below(close, support - breakout_width)

    h0 = pivots["pat_pivot_high_0"]
    h1 = pivots["pat_pivot_high_1"]
    h2 = pivots["pat_pivot_high_2"]
    h3 = pivots.get("pat_pivot_high_3", h2)
    h4 = pivots.get("pat_pivot_high_4", h3)
    hi0 = pivots["pat_pivot_high_index_0"]
    hi1 = pivots["pat_pivot_high_index_1"]
    hi2 = pivots["pat_pivot_high_index_2"]
    hi3 = pivots.get("pat_pivot_high_index_3", hi2)
    hi4 = pivots.get("pat_pivot_high_index_4", hi3)
    l0 = pivots["pat_pivot_low_0"]
    l1 = pivots["pat_pivot_low_1"]
    l2 = pivots["pat_pivot_low_2"]
    l3 = pivots.get("pat_pivot_low_3", l2)
    l4 = pivots.get("pat_pivot_low_4", l3)
    li0 = pivots["pat_pivot_low_index_0"]
    li1 = pivots["pat_pivot_low_index_1"]
    li2 = pivots["pat_pivot_low_index_2"]
    li3 = pivots.get("pat_pivot_low_index_3", li2)
    li4 = pivots.get("pat_pivot_low_index_4", li3)

    trend_tol = cfg.pivot_similarity_pct * 0.35
    flat_tol = cfg.pivot_similarity_pct * 1.5
    high_values = [h4, h3, h2, h1, h0]
    high_indexes = [hi4, hi3, hi2, hi1, hi0]
    low_values = [l4, l3, l2, l1, l0]
    low_indexes = [li4, li3, li2, li1, li0]
    high_lh_score = _ordered_sequence_score(high_values, "down", trend_tol)
    high_hh_score = _ordered_sequence_score(high_values, "up", trend_tol)
    high_flat_score = _flat_sequence_score(high_values, flat_tol)
    low_hl_score = _ordered_sequence_score(low_values, "up", trend_tol)
    low_ll_score = _ordered_sequence_score(low_values, "down", trend_tol)
    low_flat_score = _flat_sequence_score(low_values, flat_tol)
    high_lh = high_lh_score.ge(0.55)
    high_hh = high_hh_score.ge(0.55)
    high_flat = high_flat_score.ge(0.70)
    low_hl = low_hl_score.ge(0.55)
    low_ll = low_ll_score.ge(0.55)
    low_flat = low_flat_score.ge(0.70)

    oldest_high_index = hi4.combine_first(hi3).combine_first(hi2)
    oldest_low_index = li4.combine_first(li3).combine_first(li2)
    sequence_start = pd.concat([oldest_high_index, oldest_low_index], axis=1).min(axis=1)
    sequence_end = pd.concat([hi0, li0], axis=1).max(axis=1)
    sequence_span = sequence_end - sequence_start
    sequence_age = bar_index - sequence_end
    sequence_complete = (
        h0.notna()
        & h1.notna()
        & h2.notna()
        & l0.notna()
        & l1.notna()
        & l2.notna()
        & _ordered_indexes(high_indexes)
        & _ordered_indexes(low_indexes)
        & _between(sequence_span, cfg.min_sequence_span_bars, cfg.max_sequence_span_bars)
        & sequence_age.between(0.0, float(cfg.max_pivot_pattern_age_bars))
    )

    upper_fit = _pivot_sequence_envelope_line(high_indexes, high_values, bar_index, close, "high", cfg)
    lower_fit = _pivot_sequence_envelope_line(low_indexes, low_values, bar_index, close, "low", cfg)
    upper_line = upper_fit["line"].combine_first(_line_from_points(hi2, h2, hi0, h0, bar_index))
    lower_line = lower_fit["line"].combine_first(_line_from_points(li2, l2, li0, l0, bar_index))
    sequence_touch_count = (upper_fit["touch_count"] + lower_fit["touch_count"]).where(sequence_complete, 0.0)

    upper_slope = upper_fit["slope_pct"].combine_first(_safe_div(h0 - h2, (hi0 - hi2) * close.abs())).fillna(0.0)
    lower_slope = lower_fit["slope_pct"].combine_first(_safe_div(l0 - l2, (li0 - li2) * close.abs())).fillna(0.0)
    slope_gap = lower_slope - upper_slope
    mean_slope = (upper_slope + lower_slope) / 2.0
    start_width = (h2 - l2).abs()
    end_width = (h0 - l0).abs()
    sequence_compression = _clip01(1.0 - _safe_div(end_width, start_width))
    parallel_score = _clip01(1.0 - (upper_slope - lower_slope).abs() / max(cfg.parallel_slope_tolerance_pct, 1e-9))
    converging_score = _clip01(slope_gap / max(cfg.min_converging_slope_gap_pct * 3.0, 1e-9))
    contained = upper_line.gt(lower_line) & high.le(upper_line + zone_width) & low.ge(lower_line - zone_width)
    containment_ratio = (
        contained.astype("float64")
        .rolling(cfg.pattern_memory_window, min_periods=max(2, cfg.pattern_memory_window // 3))
        .mean()
        .shift(1)
        .fillna(0.0)
    )
    inside_now = close.le(upper_line + zone_width) & close.ge(lower_line - zone_width)
    sequence_valid = (
        sequence_complete
        & upper_line.gt(lower_line)
        & sequence_touch_count.ge(float(cfg.min_sequence_touch_count))
        & containment_ratio.ge(cfg.min_sequence_containment_ratio)
        & inside_now
    ).fillna(False)
    contracting_valid = sequence_valid & sequence_compression.ge(cfg.min_channel_compression)
    parallel_valid = sequence_valid & parallel_score.ge(0.55)
    flag_parallel_valid = sequence_valid & parallel_score.ge(0.25)
    recent_impulse_long = _rolling_count(impulse["pat_impulse_up_score"].gt(0.50), cfg.pattern_memory_window * 2).ge(1.0)
    recent_impulse_short = _rolling_count(impulse["pat_impulse_down_score"].gt(0.50), cfg.pattern_memory_window * 2).ge(1.0)
    seq_res_breakout = _fresh_cross_above(close, upper_line + breakout_width)
    seq_sup_breakdown = _fresh_cross_below(close, lower_line - breakout_width)

    flag_setup_long = (
        recent_impulse_long
        & flag_parallel_valid
        & high_lh
        & low_ll
        & mean_slope.le(cfg.max_flag_forward_slope_pct)
    )
    flag_setup_short = (
        recent_impulse_short
        & flag_parallel_valid
        & high_hh
        & low_hl
        & mean_slope.ge(-cfg.max_flag_forward_slope_pct)
    )
    symmetric_triangle_setup = contracting_valid & high_lh & low_hl & converging_score.ge(0.35)
    pennant_setup_long = impulse["pat_consolidation_setup_long"] & symmetric_triangle_setup
    pennant_setup_short = impulse["pat_consolidation_setup_short"] & symmetric_triangle_setup

    ascending_triangle_setup_long = contracting_valid & high_flat & low_hl
    descending_triangle_setup_short = contracting_valid & high_lh & low_flat
    rising_wedge_setup_short = (
        contracting_valid
        & high_hh
        & low_hl
        & upper_slope.gt(0.0)
        & lower_slope.gt(upper_slope + cfg.min_converging_slope_gap_pct)
    )
    falling_wedge_setup_long = (
        contracting_valid
        & high_lh
        & low_ll
        & upper_slope.lt(lower_slope - cfg.min_converging_slope_gap_pct)
    )

    setups = {
        "flag_setup_long": flag_setup_long,
        "flag_setup_short": flag_setup_short,
        "pennant_setup_long": pennant_setup_long,
        "pennant_setup_short": pennant_setup_short,
        "ascending_triangle_setup_long": ascending_triangle_setup_long,
        "descending_triangle_setup_short": descending_triangle_setup_short,
        "symmetric_triangle_setup_long": symmetric_triangle_setup,
        "symmetric_triangle_setup_short": symmetric_triangle_setup,
        "rising_wedge_setup_short": rising_wedge_setup_short,
        "falling_wedge_setup_long": falling_wedge_setup_long,
    }
    out: dict[str, Series] = {}
    for name, setup in setups.items():
        out[f"{prefix}_{name}"] = setup.fillna(False)

    long_setup_memory = {
        "flag": _rolling_count(flag_setup_long, cfg.pattern_memory_window).ge(1.0),
        "pennant": _rolling_count(pennant_setup_long, cfg.pattern_memory_window).ge(1.0),
        "ascending_triangle": _rolling_count(ascending_triangle_setup_long, cfg.pattern_memory_window).ge(1.0),
        "symmetric_triangle": _rolling_count(symmetric_triangle_setup, cfg.pattern_memory_window).ge(1.0),
        "falling_wedge": _rolling_count(falling_wedge_setup_long, cfg.pattern_memory_window).ge(1.0),
    }
    short_setup_memory = {
        "flag": _rolling_count(flag_setup_short, cfg.pattern_memory_window).ge(1.0),
        "pennant": _rolling_count(pennant_setup_short, cfg.pattern_memory_window).ge(1.0),
        "descending_triangle": _rolling_count(descending_triangle_setup_short, cfg.pattern_memory_window).ge(1.0),
        "symmetric_triangle": _rolling_count(symmetric_triangle_setup, cfg.pattern_memory_window).ge(1.0),
        "rising_wedge": _rolling_count(rising_wedge_setup_short, cfg.pattern_memory_window).ge(1.0),
    }

    def line_quality_for(setup: Series, base_bias: Series | None = None) -> Series:
        bias = base_bias if base_bias is not None else pd.Series(1.0, index=setup.index)
        sequence_quality = _clip01(
            0.30 * (sequence_touch_count / 6.0)
            + 0.30 * containment_ratio
            + 0.20 * sequence_compression
            + 0.10 * parallel_score
            + 0.10 * _span_score(sequence_span, cfg.min_sequence_span_bars, cfg.max_sequence_span_bars)
        )
        return _clip01(0.70 * sequence_quality + 0.15 * line_shape_quality + 0.15 * bias).where(setup, 0.0)

    quality_long = {
        "flag": line_quality_for(flag_setup_long, impulse["pat_impulse_up_score"]),
        "pennant": line_quality_for(pennant_setup_long, impulse["pat_consolidation_quality_long"]),
        "ascending_triangle": line_quality_for(ascending_triangle_setup_long),
        "symmetric_triangle": line_quality_for(symmetric_triangle_setup),
        "falling_wedge": line_quality_for(falling_wedge_setup_long),
    }
    quality_short = {
        "flag": line_quality_for(flag_setup_short, impulse["pat_impulse_down_score"]),
        "pennant": line_quality_for(pennant_setup_short, impulse["pat_consolidation_quality_short"]),
        "descending_triangle": line_quality_for(descending_triangle_setup_short),
        "symmetric_triangle": line_quality_for(symmetric_triangle_setup),
        "rising_wedge": line_quality_for(rising_wedge_setup_short),
    }

    for pattern, memory in long_setup_memory.items():
        out[f"{prefix}_{pattern}_breakout_long"] = _dedupe_events(seq_res_breakout & memory, cfg.entry_cooldown_bars)
        out[f"{prefix}_{pattern}_invalid_long"] = _dedupe_events(seq_sup_breakdown & memory, cfg.entry_cooldown_bars)
        out[f"{prefix}_{pattern}_quality_long"] = quality_long[pattern]
    for pattern, memory in short_setup_memory.items():
        out[f"{prefix}_{pattern}_breakdown_short"] = _dedupe_events(seq_sup_breakdown & memory, cfg.entry_cooldown_bars)
        out[f"{prefix}_{pattern}_invalid_short"] = _dedupe_events(seq_res_breakout & memory, cfg.entry_cooldown_bars)
        out[f"{prefix}_{pattern}_quality_short"] = quality_short[pattern]

    # Triangle failure breaks are useful evidence in the opposite direction.
    out[f"{prefix}_ascending_triangle_breakdown_short"] = _dedupe_events(
        seq_sup_breakdown & _rolling_count(ascending_triangle_setup_long, cfg.pattern_memory_window).ge(1.0),
        cfg.entry_cooldown_bars,
    )
    out[f"{prefix}_descending_triangle_breakout_long"] = _dedupe_events(
        seq_res_breakout & _rolling_count(descending_triangle_setup_short, cfg.pattern_memory_window).ge(1.0),
        cfg.entry_cooldown_bars,
    )
    out[f"{prefix}_sequence_upper_line"] = upper_line.where(sequence_valid | seq_res_breakout)
    out[f"{prefix}_sequence_lower_line"] = lower_line.where(sequence_valid | seq_sup_breakdown)
    out[f"{prefix}_sequence_high_direction"] = pd.Series(
        np.select([high_hh, high_lh, high_flat], [1, -1, 0], default=0),
        index=close.index,
        dtype="int8",
    )
    out[f"{prefix}_sequence_low_direction"] = pd.Series(
        np.select([low_hl, low_ll, low_flat], [1, -1, 0], default=0),
        index=close.index,
        dtype="int8",
    )
    out[f"{prefix}_sequence_touch_count"] = sequence_touch_count
    out[f"{prefix}_sequence_containment_ratio"] = containment_ratio.where(sequence_complete)
    out[f"{prefix}_sequence_compression_score"] = sequence_compression.where(sequence_complete, 0.0)
    out[f"{prefix}_sequence_parallel_score"] = parallel_score.where(sequence_complete, 0.0)
    out[f"{prefix}_sequence_converging_score"] = converging_score.where(sequence_complete, 0.0)
    out[f"{prefix}_line_breakout_long"] = _dedupe_events(res_breakout & (recent_long_base | line_valid), cfg.entry_cooldown_bars)
    out[f"{prefix}_line_breakdown_short"] = _dedupe_events(sup_breakdown & (recent_short_base | line_valid), cfg.entry_cooldown_bars)
    return out


def _pivot_history_columns(frame: DataFrame, bar_index: Series, cfg: PatternStructureConfig) -> dict[str, Series]:
    pp = cfg.pivot_prefix
    high_price = _num(frame, f"{pp}_pivot_high")
    low_price = _num(frame, f"{pp}_pivot_low")
    high_index = _num(frame, f"{pp}_pivot_high_index")
    low_index = _num(frame, f"{pp}_pivot_low_index")
    high_prom = _num(frame, f"{pp}_pivot_high_prominence_pct")
    low_prom = _num(frame, f"{pp}_pivot_low_prominence_pct")
    if high_price.isna().all():
        high_price = _num(frame, f"{pp}_structural_pivot_high")
        high_index = _num(frame, f"{pp}_structural_pivot_high_index")
        high_prom = _num(frame, f"{pp}_structural_pivot_high_prominence_pct")
    if low_price.isna().all():
        low_price = _num(frame, f"{pp}_structural_pivot_low")
        low_index = _num(frame, f"{pp}_structural_pivot_low_index")
        low_prom = _num(frame, f"{pp}_structural_pivot_low_prominence_pct")

    high_events = _event_history(high_price, high_index, high_prom, depth=5, index=frame.index)
    low_events = _event_history(low_price, low_index, low_prom, depth=5, index=frame.index)
    out: dict[str, Series] = {"pat_bar_index": bar_index}
    for lag in range(5):
        out[f"pat_pivot_high_{lag}"] = high_events[f"price_{lag}"]
        out[f"pat_pivot_high_index_{lag}"] = high_events[f"index_{lag}"]
        out[f"pat_pivot_high_prominence_{lag}"] = high_events[f"prominence_{lag}"]
        out[f"pat_pivot_low_{lag}"] = low_events[f"price_{lag}"]
        out[f"pat_pivot_low_index_{lag}"] = low_events[f"index_{lag}"]
        out[f"pat_pivot_low_prominence_{lag}"] = low_events[f"prominence_{lag}"]
    return out


def _pivot_pattern_columns(
    close: Series,
    high: Series,
    low: Series,
    pivots: dict[str, Series],
    zone_width: Series,
    bar_index: Series,
    cfg: PatternStructureConfig,
    prefix: str,
) -> dict[str, Series]:
    h0 = pivots["pat_pivot_high_0"]
    h1 = pivots["pat_pivot_high_1"]
    h2 = pivots["pat_pivot_high_2"]
    hi0 = pivots["pat_pivot_high_index_0"]
    hi1 = pivots["pat_pivot_high_index_1"]
    hi2 = pivots["pat_pivot_high_index_2"]
    hp0 = pivots["pat_pivot_high_prominence_0"].fillna(0.0)
    hp1 = pivots["pat_pivot_high_prominence_1"].fillna(0.0)
    hp2 = pivots["pat_pivot_high_prominence_2"].fillna(0.0)
    l0 = pivots["pat_pivot_low_0"]
    l1 = pivots["pat_pivot_low_1"]
    l2 = pivots["pat_pivot_low_2"]
    li0 = pivots["pat_pivot_low_index_0"]
    li1 = pivots["pat_pivot_low_index_1"]
    li2 = pivots["pat_pivot_low_index_2"]
    lp0 = pivots["pat_pivot_low_prominence_0"].fillna(0.0)
    lp1 = pivots["pat_pivot_low_prominence_1"].fillna(0.0)
    lp2 = pivots["pat_pivot_low_prominence_2"].fillna(0.0)
    breakout_width = _breakout_width(close, zone_width, cfg)

    double_top_similarity = _similarity_score(h0, h1, cfg.pivot_similarity_pct)
    double_top_span = hi0 - hi1
    double_top_neckline = l0
    double_top_age = bar_index - hi0
    double_top_depth = _safe_div(pd.concat([h0, h1], axis=1).min(axis=1) - double_top_neckline, close).clip(lower=0.0)
    double_top_setup = (
        double_top_similarity.gt(0.0)
        & _between(double_top_span, cfg.min_sequence_span_bars, cfg.max_sequence_span_bars)
        & double_top_age.between(0.0, float(cfg.max_pivot_pattern_age_bars))
        & double_top_depth.ge(cfg.min_pivot_pattern_depth_pct)
        & li0.gt(hi1)
        & li0.lt(hi0)
        & close.le(pd.concat([h0, h1], axis=1).max(axis=1) + zone_width)
    )
    double_top_quality = _clip01(
        0.40 * double_top_similarity
        + 0.25 * _span_score(double_top_span, cfg.min_sequence_span_bars, cfg.max_sequence_span_bars)
        + 0.20 * _clip01((hp0 + hp1) / max(cfg.pivot_similarity_pct * 4.0, 1e-9))
        + 0.15 * _clip01(double_top_depth / max(cfg.min_pivot_pattern_depth_pct * 2.0, 1e-9))
    ).where(double_top_setup, 0.0)
    double_top_memory = _rolling_count(double_top_setup, cfg.pattern_memory_window).ge(1.0)
    double_top_breakdown = _dedupe_events(_fresh_cross_below(close, double_top_neckline - breakout_width) & double_top_memory, cfg.entry_cooldown_bars)
    double_top_invalid = _dedupe_events(close.gt(pd.concat([h0, h1], axis=1).max(axis=1) + breakout_width) & double_top_memory, cfg.entry_cooldown_bars)

    double_bottom_similarity = _similarity_score(l0, l1, cfg.pivot_similarity_pct)
    double_bottom_span = li0 - li1
    double_bottom_neckline = h0
    double_bottom_age = bar_index - li0
    double_bottom_depth = _safe_div(double_bottom_neckline - pd.concat([l0, l1], axis=1).max(axis=1), close).clip(lower=0.0)
    double_bottom_setup = (
        double_bottom_similarity.gt(0.0)
        & _between(double_bottom_span, cfg.min_sequence_span_bars, cfg.max_sequence_span_bars)
        & double_bottom_age.between(0.0, float(cfg.max_pivot_pattern_age_bars))
        & double_bottom_depth.ge(cfg.min_pivot_pattern_depth_pct)
        & hi0.gt(li1)
        & hi0.lt(li0)
        & close.ge(pd.concat([l0, l1], axis=1).min(axis=1) - zone_width)
    )
    double_bottom_quality = _clip01(
        0.40 * double_bottom_similarity
        + 0.25 * _span_score(double_bottom_span, cfg.min_sequence_span_bars, cfg.max_sequence_span_bars)
        + 0.20 * _clip01((lp0 + lp1) / max(cfg.pivot_similarity_pct * 4.0, 1e-9))
        + 0.15 * _clip01(double_bottom_depth / max(cfg.min_pivot_pattern_depth_pct * 2.0, 1e-9))
    ).where(double_bottom_setup, 0.0)
    double_bottom_memory = _rolling_count(double_bottom_setup, cfg.pattern_memory_window).ge(1.0)
    double_bottom_breakout = _dedupe_events(_fresh_cross_above(close, double_bottom_neckline + breakout_width) & double_bottom_memory, cfg.entry_cooldown_bars)
    double_bottom_invalid = _dedupe_events(close.lt(pd.concat([l0, l1], axis=1).min(axis=1) - breakout_width) & double_bottom_memory, cfg.entry_cooldown_bars)

    hs_left = h2
    hs_head = h1
    hs_right = h0
    hs_left_idx = hi2
    hs_head_idx = hi1
    hs_right_idx = hi0
    hs_neckline = _line_from_points(li1, l1, li0, l0, bar_index)
    hs_sequence = hs_left_idx.lt(li1) & li1.lt(hs_head_idx) & hs_head_idx.lt(li0) & li0.lt(hs_right_idx)
    hs_shoulder_similarity = _similarity_score(hs_left, hs_right, cfg.shoulder_similarity_pct)
    hs_head_prominence = _safe_div(hs_head - pd.concat([hs_left, hs_right], axis=1).max(axis=1), hs_head)
    hs_age = bar_index - hs_right_idx
    hs_span = hs_right_idx - hs_left_idx
    hs_neckline_slope = _safe_div(l0 - l1, (li0 - li1) * close.abs()).abs()
    hs_depth = _safe_div(hs_head - hs_neckline, close).clip(lower=0.0)
    hs_setup = (
        hs_sequence
        & _between(hs_span, cfg.min_sequence_span_bars * 2, cfg.max_sequence_span_bars)
        & hs_neckline_slope.le(cfg.max_neckline_slope_pct_per_bar)
        & hs_shoulder_similarity.gt(0.0)
        & hs_head_prominence.ge(cfg.head_prominence_pct)
        & hs_age.between(0.0, float(cfg.max_pivot_pattern_age_bars))
        & hs_depth.ge(cfg.min_pivot_pattern_depth_pct)
    )
    hs_quality = _clip01(
        0.34 * hs_shoulder_similarity
        + 0.34 * _clip01(hs_head_prominence / max(cfg.head_prominence_pct * 3.0, 1e-9))
        + 0.18 * _span_score(hs_right_idx - hs_left_idx, cfg.min_sequence_span_bars * 2, cfg.max_sequence_span_bars)
        + 0.14 * _clip01((hp0 + hp1 + hp2) / max(cfg.head_prominence_pct * 6.0, 1e-9))
    ).where(hs_setup, 0.0)
    hs_memory = _rolling_count(hs_setup, cfg.pattern_memory_window).ge(1.0)
    hs_breakdown = _dedupe_events(_fresh_cross_below(close, hs_neckline - breakout_width) & hs_memory, cfg.entry_cooldown_bars)
    hs_invalid = _dedupe_events(close.gt(hs_head + breakout_width) & hs_memory, cfg.entry_cooldown_bars)

    ihs_left = l2
    ihs_head = l1
    ihs_right = l0
    ihs_left_idx = li2
    ihs_head_idx = li1
    ihs_right_idx = li0
    ihs_neckline = _line_from_points(hi1, h1, hi0, h0, bar_index)
    ihs_sequence = ihs_left_idx.lt(hi1) & hi1.lt(ihs_head_idx) & ihs_head_idx.lt(hi0) & hi0.lt(ihs_right_idx)
    ihs_shoulder_similarity = _similarity_score(ihs_left, ihs_right, cfg.shoulder_similarity_pct)
    ihs_head_prominence = _safe_div(pd.concat([ihs_left, ihs_right], axis=1).min(axis=1) - ihs_head, ihs_head.abs())
    ihs_age = bar_index - ihs_right_idx
    ihs_span = ihs_right_idx - ihs_left_idx
    ihs_neckline_slope = _safe_div(h0 - h1, (hi0 - hi1) * close.abs()).abs()
    ihs_depth = _safe_div(ihs_neckline - ihs_head, close).clip(lower=0.0)
    ihs_setup = (
        ihs_sequence
        & _between(ihs_span, cfg.min_sequence_span_bars * 2, cfg.max_sequence_span_bars)
        & ihs_neckline_slope.le(cfg.max_neckline_slope_pct_per_bar)
        & ihs_shoulder_similarity.gt(0.0)
        & ihs_head_prominence.ge(cfg.head_prominence_pct)
        & ihs_age.between(0.0, float(cfg.max_pivot_pattern_age_bars))
        & ihs_depth.ge(cfg.min_pivot_pattern_depth_pct)
    )
    ihs_quality = _clip01(
        0.34 * ihs_shoulder_similarity
        + 0.34 * _clip01(ihs_head_prominence / max(cfg.head_prominence_pct * 3.0, 1e-9))
        + 0.18 * _span_score(ihs_right_idx - ihs_left_idx, cfg.min_sequence_span_bars * 2, cfg.max_sequence_span_bars)
        + 0.14 * _clip01((lp0 + lp1 + lp2) / max(cfg.head_prominence_pct * 6.0, 1e-9))
    ).where(ihs_setup, 0.0)
    ihs_memory = _rolling_count(ihs_setup, cfg.pattern_memory_window).ge(1.0)
    ihs_breakout = _dedupe_events(_fresh_cross_above(close, ihs_neckline + breakout_width) & ihs_memory, cfg.entry_cooldown_bars)
    ihs_invalid = _dedupe_events(close.lt(ihs_head - breakout_width) & ihs_memory, cfg.entry_cooldown_bars)

    return {
        f"{prefix}_double_top_setup_short": double_top_setup.fillna(False),
        f"{prefix}_double_top_breakdown_short": double_top_breakdown,
        f"{prefix}_double_top_invalid_short": double_top_invalid,
        f"{prefix}_double_top_neckline": double_top_neckline.where(double_top_setup | double_top_memory),
        f"{prefix}_double_top_quality_short": double_top_quality,
        f"{prefix}_double_bottom_setup_long": double_bottom_setup.fillna(False),
        f"{prefix}_double_bottom_breakout_long": double_bottom_breakout,
        f"{prefix}_double_bottom_invalid_long": double_bottom_invalid,
        f"{prefix}_double_bottom_neckline": double_bottom_neckline.where(double_bottom_setup | double_bottom_memory),
        f"{prefix}_double_bottom_quality_long": double_bottom_quality,
        f"{prefix}_head_shoulders_setup_short": hs_setup.fillna(False),
        f"{prefix}_head_shoulders_breakdown_short": hs_breakdown,
        f"{prefix}_head_shoulders_invalid_short": hs_invalid,
        f"{prefix}_head_shoulders_neckline": hs_neckline.where(hs_setup | hs_memory),
        f"{prefix}_head_shoulders_quality_short": hs_quality,
        f"{prefix}_inverse_head_shoulders_setup_long": ihs_setup.fillna(False),
        f"{prefix}_inverse_head_shoulders_breakout_long": ihs_breakout,
        f"{prefix}_inverse_head_shoulders_invalid_long": ihs_invalid,
        f"{prefix}_inverse_head_shoulders_neckline": ihs_neckline.where(ihs_setup | ihs_memory),
        f"{prefix}_inverse_head_shoulders_quality_long": ihs_quality,
    }


def _compatibility_columns(new_cols: dict[str, Series], prefix: str, cfg: PatternStructureConfig, close: Series, high: Series, low: Series) -> dict[str, Series]:
    cooldown = cfg.entry_cooldown_bars
    flag_setup_long = new_cols[f"{prefix}_flag_setup_long"]
    flag_setup_short = new_cols[f"{prefix}_flag_setup_short"]
    pennant_setup_long = new_cols[f"{prefix}_pennant_setup_long"]
    pennant_setup_short = new_cols[f"{prefix}_pennant_setup_short"]
    flag_breakout_long = new_cols[f"{prefix}_flag_breakout_long"]
    flag_breakdown_short = new_cols[f"{prefix}_flag_breakdown_short"]
    pennant_breakout_long = new_cols[f"{prefix}_pennant_breakout_long"]
    pennant_breakdown_short = new_cols[f"{prefix}_pennant_breakdown_short"]
    breakout_long = _any_columns(
        new_cols,
        [name for name in new_cols if name.endswith("_breakout_long")],
        close.index,
    )
    breakdown_short = _any_columns(
        new_cols,
        [name for name in new_cols if name.endswith("_breakdown_short")],
        close.index,
    )
    consolidation_breakout_long = new_cols.get(
        f"{prefix}_consolidation_breakout_long",
        new_cols["pat_consolidation_breakout_long"],
    )
    consolidation_breakdown_short = new_cols.get(
        f"{prefix}_consolidation_breakdown_short",
        new_cols["pat_consolidation_breakdown_short"],
    )
    return {
        f"{prefix}_flag_state_long": flag_setup_long,
        f"{prefix}_flag_state_short": flag_setup_short,
        f"{prefix}_pennant_state_long": pennant_setup_long,
        f"{prefix}_pennant_state_short": pennant_setup_short,
        f"{prefix}_flag_long": _dedupe_events(flag_setup_long & ~flag_setup_long.shift(1, fill_value=False), cooldown),
        f"{prefix}_flag_short": _dedupe_events(flag_setup_short & ~flag_setup_short.shift(1, fill_value=False), cooldown),
        f"{prefix}_pennant_long": _dedupe_events(pennant_setup_long & ~pennant_setup_long.shift(1, fill_value=False), cooldown),
        f"{prefix}_pennant_short": _dedupe_events(pennant_setup_short & ~pennant_setup_short.shift(1, fill_value=False), cooldown),
        f"{prefix}_raw_breakout_long": new_cols[f"{prefix}_line_breakout_long"] | consolidation_breakout_long,
        f"{prefix}_raw_breakout_short": new_cols[f"{prefix}_line_breakdown_short"] | consolidation_breakdown_short,
        f"{prefix}_breakout_long": _dedupe_events(breakout_long, cooldown),
        f"{prefix}_breakout_short": _dedupe_events(breakdown_short, cooldown),
        f"{prefix}_entry_flag_breakout_long": flag_breakout_long,
        f"{prefix}_entry_pennant_breakout_long": pennant_breakout_long,
        f"{prefix}_entry_flag_breakdown_short": flag_breakdown_short,
        f"{prefix}_entry_pennant_breakdown_short": pennant_breakdown_short,
    }


def _event_history(price: Series, event_index: Series, prominence: Series, depth: int, index: pd.Index) -> dict[str, Series]:
    mask = price.notna()
    event_number = mask.astype("int64").cumsum()
    event_ids = np.arange(1, int(mask.sum()) + 1)
    price_lookup = pd.Series(price[mask].to_numpy(dtype="float64"), index=event_ids)
    index_lookup = pd.Series(event_index[mask].to_numpy(dtype="float64"), index=event_ids)
    prom_lookup = pd.Series(prominence[mask].to_numpy(dtype="float64"), index=event_ids)
    out: dict[str, Series] = {}
    for lag in range(depth):
        lookup_id = event_number - int(lag)
        out[f"price_{lag}"] = pd.Series(lookup_id.map(price_lookup), index=index, dtype="float64")
        out[f"index_{lag}"] = pd.Series(lookup_id.map(index_lookup), index=index, dtype="float64")
        out[f"prominence_{lag}"] = pd.Series(lookup_id.map(prom_lookup), index=index, dtype="float64")
    return out


def _resolve_config(config: PatternStructureConfig | None, **overrides: object) -> PatternStructureConfig:
    base = config or PatternStructureConfig()
    values = dict(base.__dict__)
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return PatternStructureConfig(**values)


def _validate_config(cfg: PatternStructureConfig) -> None:
    windows = [
        cfg.impulse_window,
        cfg.consolidation_window,
        cfg.atr_period,
        cfg.line_respect_window,
        cfg.pattern_memory_window,
        cfg.entry_cooldown_bars,
    ]
    if any(int(value) < 2 for value in windows):
        raise ValueError("windows must be at least 2")
    if cfg.impulse_atr_min <= 0.0 or cfg.impulse_pct_min <= 0.0:
        raise ValueError("impulse thresholds must be positive")
    if not 0.5 <= cfg.impulse_close_location_min < 1.0:
        raise ValueError("impulse_close_location_min must be between 0.5 and 1")
    if not 0.0 < cfg.max_retrace_pct <= 1.0:
        raise ValueError("max_retrace_pct must be between 0 and 1")
    if cfg.max_consolidation_extension_pct < 0.0 or cfg.max_consolidation_drift_atr < 0.0:
        raise ValueError("consolidation limits must be non-negative")
    if not 0.0 <= cfg.min_range_contraction <= 1.0:
        raise ValueError("min_range_contraction must be between 0 and 1")
    if cfg.dry_volume_rvol_max <= 0.0:
        raise ValueError("dry_volume_rvol_max must be positive")
    if cfg.breakout_buffer_pct < 0.0:
        raise ValueError("breakout_buffer_pct must be non-negative")
    if not 0.0 <= cfg.breakout_zone_fraction <= 1.0:
        raise ValueError("breakout_zone_fraction must be between 0 and 1")
    if cfg.line_zone_atr_mult < 0.0 or cfg.line_zone_pct < 0.0:
        raise ValueError("line zone settings must be non-negative")
    if cfg.min_line_touch_count < 1:
        raise ValueError("min_line_touch_count must be at least 1")
    for value in (
        cfg.min_line_score,
        cfg.min_line_respect_ratio,
        cfg.min_channel_compression,
        cfg.min_sequence_containment_ratio,
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("score/ratio thresholds must be between 0 and 1")
    if cfg.min_sequence_touch_count < 3:
        raise ValueError("min_sequence_touch_count must be at least 3")
    if cfg.min_sequence_span_bars < 1 or cfg.max_sequence_span_bars < cfg.min_sequence_span_bars:
        raise ValueError("invalid sequence span bounds")
    if cfg.max_pivot_pattern_age_bars < 2:
        raise ValueError("max_pivot_pattern_age_bars must be at least 2")
    if cfg.min_pivot_pattern_depth_pct < 0.0 or cfg.max_neckline_slope_pct_per_bar < 0.0:
        raise ValueError("pivot pattern thresholds must be non-negative")
    if not cfg.pivot_prefix or not cfg.trendline_prefix or not cfg.output_prefix:
        raise ValueError("prefix values must be non-empty")


def _validate_dataframe(dataframe: DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"DataFrame is missing OHLCV columns: {missing}")


def _atr(frame: DataFrame, period: int) -> Series:
    high = _num(frame, "high")
    low = _num(frame, "low")
    close = _num(frame, "close")
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(int(period), min_periods=1).mean()


def _breakout_width(close: Series, zone_width: Series, cfg: PatternStructureConfig) -> Series:
    zone_component = zone_width.fillna(0.0) * cfg.breakout_zone_fraction
    price_component = close.abs().fillna(0.0) * cfg.breakout_buffer_pct
    return pd.concat([zone_component, price_component], axis=1).max(axis=1)


def _line_value_at(x_old: Series, y_old: Series, x_new: Series, y_new: Series, x_target: Series) -> Series:
    slope = _safe_div(y_new - y_old, x_new - x_old)
    return (y_old + slope * (x_target - x_old)).where((x_new - x_old).abs().gt(0.0))


def _pivot_sequence_envelope_line(
    indexes: list[Series],
    prices: list[Series],
    bar_index: Series,
    close: Series,
    side: str,
    cfg: PatternStructureConfig,
) -> dict[str, Series]:
    weights = [price.notna().astype("float64") for price in prices]
    sum_w = sum(weights)
    sum_x = sum(weight * index.fillna(0.0) for weight, index in zip(weights, indexes, strict=False))
    sum_y = sum(weight * price.fillna(0.0) for weight, price in zip(weights, prices, strict=False))
    sum_x2 = sum(weight * index.fillna(0.0) * index.fillna(0.0) for weight, index in zip(weights, indexes, strict=False))
    sum_xy = sum(
        weight * index.fillna(0.0) * price.fillna(0.0)
        for weight, index, price in zip(weights, indexes, prices, strict=False)
    )
    denominator = sum_w * sum_x2 - sum_x * sum_x
    slope = _safe_div(sum_w * sum_xy - sum_x * sum_y, denominator).where(sum_w.ge(3.0) & denominator.abs().gt(1e-9))
    intercept_candidates = [
        (price - slope * index).where(price.notna() & index.notna())
        for index, price in zip(indexes, prices, strict=False)
    ]
    if side == "high":
        intercept = pd.concat(intercept_candidates, axis=1).max(axis=1)
    else:
        intercept = pd.concat(intercept_candidates, axis=1).min(axis=1)
    line = intercept + slope * bar_index
    touch_parts = []
    for index, price in zip(indexes, prices, strict=False):
        line_at_pivot = intercept + slope * index
        tolerance = price.abs() * max(cfg.pivot_similarity_pct, cfg.line_zone_pct)
        touch_parts.append((price.notna() & index.notna() & (price - line_at_pivot).abs().le(tolerance)).astype("float64"))
    touch_count = sum(touch_parts)
    return {
        "line": line.where(sum_w.ge(3.0)),
        "slope_pct": _safe_div(slope, close.abs()).where(sum_w.ge(3.0)),
        "touch_count": touch_count.where(sum_w.ge(3.0), 0.0),
    }


def _line_from_points(x_old: Series, y_old: Series, x_new: Series, y_new: Series, bar_index: Series) -> Series:
    slope = _safe_div(y_new - y_old, x_new - x_old)
    return (y_new + slope * (bar_index - x_new)).where((x_new - x_old).abs().gt(0.0))


def _fresh_cross_above(value: Series, level: Series) -> Series:
    return (value.gt(level) & value.shift(1).le(level.shift(1))).fillna(False)


def _fresh_cross_below(value: Series, level: Series) -> Series:
    return (value.lt(level) & value.shift(1).ge(level.shift(1))).fillna(False)


def _touches_line(high: Series, low: Series, level: Series, zone_width: Series) -> Series:
    return (level.notna() & high.ge(level - zone_width) & low.le(level + zone_width)).fillna(False)


def _similarity_score(a: Series, b: Series, tolerance_pct: float) -> Series:
    distance = (a - b).abs()
    reference = pd.concat([a.abs(), b.abs()], axis=1).mean(axis=1)
    return _clip01(1.0 - _safe_div(distance, reference * max(float(tolerance_pct), 1e-9)))


def _span_score(span: Series, minimum: int, maximum: int) -> Series:
    min_value = float(minimum)
    max_value = float(maximum)
    mid = (min_value + max_value) / 2.0
    half_range = max((max_value - min_value) / 2.0, 1.0)
    return _clip01(1.0 - (span - mid).abs() / half_range).where(span.between(min_value, max_value), 0.0)


def _sequence_up(old: Series, middle: Series, new: Series, tolerance: float) -> Series:
    return (middle.ge(old * (1.0 + tolerance)) & new.ge(middle * (1.0 + tolerance))).fillna(False)


def _sequence_down(old: Series, middle: Series, new: Series, tolerance: float) -> Series:
    return (middle.le(old * (1.0 - tolerance)) & new.le(middle * (1.0 - tolerance))).fillna(False)


def _sequence_flat(old: Series, middle: Series, new: Series, tolerance: float) -> Series:
    values = pd.concat([old, middle, new], axis=1)
    width = values.max(axis=1) - values.min(axis=1)
    base = values.max(axis=1).abs()
    return _safe_div(width, base).le(tolerance).fillna(False)


def _ordered_sequence_score(values: list[Series], direction: str, tolerance: float) -> Series:
    count = pd.Series(0.0, index=values[-1].index)
    hits = pd.Series(0.0, index=values[-1].index)
    for old, new in zip(values[:-1], values[1:], strict=False):
        pair = old.notna() & new.notna()
        if direction == "up":
            ok = new.ge(old * (1.0 + tolerance))
        else:
            ok = new.le(old * (1.0 - tolerance))
        count = count + pair.astype("float64")
        hits = hits + (pair & ok).astype("float64")
    return _safe_div(hits, count).where(count.ge(2.0), 0.0).fillna(0.0)


def _flat_sequence_score(values: list[Series], tolerance: float) -> Series:
    count = pd.Series(0.0, index=values[-1].index)
    hits = pd.Series(0.0, index=values[-1].index)
    for old, new in zip(values[:-1], values[1:], strict=False):
        pair = old.notna() & new.notna()
        distance = (new - old).abs()
        reference = pd.concat([old.abs(), new.abs()], axis=1).max(axis=1)
        ok = _safe_div(distance, reference).le(tolerance)
        count = count + pair.astype("float64")
        hits = hits + (pair & ok).astype("float64")
    return _safe_div(hits, count).where(count.ge(2.0), 0.0).fillna(0.0)


def _ordered_indexes(indexes: list[Series]) -> Series:
    ordered = pd.Series(True, index=indexes[-1].index)
    for old, new in zip(indexes[:-1], indexes[1:], strict=False):
        pair = old.notna() & new.notna()
        ordered = ordered & (~pair | old.lt(new))
    return ordered.fillna(False)


def _between(series: Series, minimum: int | float, maximum: int | float) -> Series:
    return series.ge(float(minimum)) & series.le(float(maximum))


def _safe_div(numerator: Series, denominator: Series) -> Series:
    denominator = denominator.where(denominator.abs() > 0.0, np.nan)
    return numerator / denominator


def _rolling_count(mask: Series, window: int) -> Series:
    clean = pd.Series(mask, index=mask.index).fillna(False).astype("int8")
    cumulative = clean.cumsum()
    return (cumulative - cumulative.shift(int(window), fill_value=0)).astype("float64")


def _dedupe_events(mask: Series, cooldown_bars: int) -> Series:
    clean = pd.Series(mask, index=mask.index).fillna(False).astype("bool")
    prior_recent = _rolling_count(clean.shift(1, fill_value=False), max(int(cooldown_bars), 1))
    return (clean & prior_recent.eq(0.0)).fillna(False)


def _any_columns(columns: dict[str, Series], names: list[str], index: pd.Index) -> Series:
    selected = [pd.Series(columns[name], index=index).fillna(False).astype("bool") for name in names if name in columns]
    if not selected:
        return pd.Series(False, index=index)
    return pd.concat(selected, axis=1).any(axis=1)


def _max_columns(columns: dict[str, Series], names: list[str], index: pd.Index) -> Series:
    selected = [pd.to_numeric(columns[name], errors="coerce") for name in names if name in columns]
    if not selected:
        return pd.Series(0.0, index=index)
    return pd.concat(selected, axis=1).max(axis=1).fillna(0.0)


def _clip01(series: Series) -> Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0).fillna(0.0)


def _bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[column].fillna(False).astype("bool")


def _num(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")

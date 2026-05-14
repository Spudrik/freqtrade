from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


@dataclass(frozen=True)
class RelativeStrengthConfig:
    """Relative strength evidence against BTC, ETH, or a market benchmark.

    Pass a benchmark dataframe or close series aligned to the strategy timeframe.
    Freqtrade strategies can source this from informative pairs. The module does
    not fetch data and does not make trading decisions. Scores are normalized
    0..1 for direct validation against forward returns.

    Score meaning:
    - ``*_score_long`` rises when the pair is outperforming the benchmark over
      short/medium/long windows and the relative-strength line is high in its
      own rolling range.
    - ``*_score_short`` rises when the pair is underperforming the benchmark.
      It is relative weakness, not a standalone short-entry recommendation.
    - ``*_score_abs`` is max(long, short). It measures benchmark-relative edge
      strength.
    - ``*_state`` is -1/0/1 directional lean and is not normalized.
    - ``*_go_long`` is ``1`` only when relative strength supports new long
      exposure and the benchmark/target regime is not falling.
    - ``*_go_short`` is ``1`` only when the benchmark is falling, the target is
      falling, and the target is weak relative to the benchmark.
    - ``*_long_caution`` is ``1`` when relative strength is good but the
      benchmark is falling. This is a warning, not a hard entry or exit.

    Typical benchmarks are BTC, ETH, or a market-index/informative pair. This is
    not confluence by itself; it is a base indicator for validating whether a
    pair's relative strength improves trade selection.
    """

    short_window: int = 12
    medium_window: int = 48
    long_window: int = 144
    min_outperformance: float = 0.0
    slope_scale: float = 0.03
    percentile_window: int = 240
    context_window: int = 24
    entry_cooldown_bars: int = 8
    entry_score_min: float = 0.45
    context_full_min: float = 0.55
    context_full_margin: float = 0.12
    context_soft_min: float = 0.35
    context_soft_margin: float = 0.06
    long_reference_min_z: float = 0.0
    long_target_min_z: float = 0.0
    short_reference_max_z: float = -0.35
    short_target_max_z: float = 0.0
    short_relative_max_z: float = -0.25
    caution_reference_max_z: float = -0.35
    benchmark_close: str = "close"
    prefix: str = "rs"


def add_relative_strength(
    dataframe: DataFrame,
    benchmark: DataFrame | Series,
    config: RelativeStrengthConfig | None = None,
    *,
    short_window: int | None = None,
    medium_window: int | None = None,
    long_window: int | None = None,
    min_outperformance: float | None = None,
    slope_scale: float | None = None,
    percentile_window: int | None = None,
    context_window: int | None = None,
    entry_cooldown_bars: int | None = None,
    entry_score_min: float | None = None,
    context_full_min: float | None = None,
    context_full_margin: float | None = None,
    context_soft_min: float | None = None,
    context_soft_margin: float | None = None,
    long_reference_min_z: float | None = None,
    long_target_min_z: float | None = None,
    short_reference_max_z: float | None = None,
    short_target_max_z: float | None = None,
    short_relative_max_z: float | None = None,
    caution_reference_max_z: float | None = None,
    benchmark_close: str | None = None,
    prefix: str | None = None,
) -> DataFrame:
    """Append benchmark-relative performance columns and normalized scores."""

    cfg = _resolve_config(
        config,
        short_window=short_window,
        medium_window=medium_window,
        long_window=long_window,
        min_outperformance=min_outperformance,
        slope_scale=slope_scale,
        percentile_window=percentile_window,
        context_window=context_window,
        entry_cooldown_bars=entry_cooldown_bars,
        entry_score_min=entry_score_min,
        context_full_min=context_full_min,
        context_full_margin=context_full_margin,
        context_soft_min=context_soft_min,
        context_soft_margin=context_soft_margin,
        long_reference_min_z=long_reference_min_z,
        long_target_min_z=long_target_min_z,
        short_reference_max_z=short_reference_max_z,
        short_target_max_z=short_target_max_z,
        short_relative_max_z=short_relative_max_z,
        caution_reference_max_z=caution_reference_max_z,
        benchmark_close=benchmark_close,
        prefix=prefix,
    )
    _validate_config(cfg)
    _validate_dataframe(dataframe)

    frame = dataframe.copy()
    p = cfg.prefix
    close = _num(frame, "close").replace(0.0, np.nan)
    bench_close = _benchmark_close(benchmark, cfg, frame.index).replace(0.0, np.nan)

    pair_ret_short = close.pct_change(cfg.short_window, fill_method=None)
    bench_ret_short = bench_close.pct_change(cfg.short_window, fill_method=None)
    pair_ret_medium = close.pct_change(cfg.medium_window, fill_method=None)
    bench_ret_medium = bench_close.pct_change(cfg.medium_window, fill_method=None)
    pair_ret_long = close.pct_change(cfg.long_window, fill_method=None)
    bench_ret_long = bench_close.pct_change(cfg.long_window, fill_method=None)

    rel_short = pair_ret_short - bench_ret_short
    rel_medium = pair_ret_medium - bench_ret_medium
    rel_long = pair_ret_long - bench_ret_long
    rs_line = close / bench_close
    rs_slope = rs_line.pct_change(cfg.medium_window, fill_method=None)
    rs_percentile = _rolling_rank(rs_line, cfg.percentile_window)
    target_trend_z = _trend_z(close, cfg.medium_window)
    reference_trend_z = _trend_z(bench_close, cfg.medium_window)
    relative_spread_z = _relative_spread_z(close, bench_close, cfg.medium_window)

    outperforming = rel_medium > cfg.min_outperformance
    underperforming = rel_medium < -cfg.min_outperformance
    long_score = _clip01(
        0.35 * _clip01(rel_short / max(cfg.slope_scale, 1e-9))
        + 0.35 * _clip01(rel_medium / max(cfg.slope_scale, 1e-9))
        + 0.15 * _clip01(rel_long / max(cfg.slope_scale * 2.0, 1e-9))
        + 0.15 * rs_percentile
    )
    short_score = _clip01(
        0.35 * _clip01(-rel_short / max(cfg.slope_scale, 1e-9))
        + 0.35 * _clip01(-rel_medium / max(cfg.slope_scale, 1e-9))
        + 0.15 * _clip01(-rel_long / max(cfg.slope_scale * 2.0, 1e-9))
        + 0.15 * (1.0 - rs_percentile)
    )
    abs_score = pd.concat([long_score, short_score], axis=1).max(axis=1)
    score_margin = long_score - short_score
    score_state = pd.Series(
        np.select([long_score.gt(short_score), short_score.gt(long_score)], [1.0, -1.0], default=0.0),
        index=frame.index,
    )
    context_long = long_score.ewm(span=cfg.context_window, min_periods=1, adjust=False).mean()
    context_short = short_score.ewm(span=cfg.context_window, min_periods=1, adjust=False).mean()
    context_margin = context_long - context_short
    full_bull = context_long.ge(cfg.context_full_min) & context_margin.ge(cfg.context_full_margin)
    full_bear = context_short.ge(cfg.context_full_min) & context_margin.le(-cfg.context_full_margin)
    bullish_chop = ~full_bull & ~full_bear & context_long.ge(cfg.context_soft_min) & context_margin.ge(cfg.context_soft_margin)
    bearish_chop = ~full_bull & ~full_bear & context_short.ge(cfg.context_soft_min) & context_margin.le(-cfg.context_soft_margin)
    market_context = pd.Series(
        np.select(
            [full_bull, bullish_chop, full_bear, bearish_chop],
            [2.0, 1.0, -2.0, -1.0],
            default=0.0,
        ),
        index=frame.index,
        dtype="float64",
    )

    cross_outperforming = outperforming & ~outperforming.shift(1, fill_value=False).astype(bool)
    cross_underperforming = underperforming & ~underperforming.shift(1, fill_value=False).astype(bool)
    improving_slope = rs_slope.gt(0.0) & rs_slope.gt(rs_slope.shift(1).fillna(0.0))
    weakening_slope = rs_slope.lt(0.0) & rs_slope.lt(rs_slope.shift(1).fillna(0.0))
    high_relative_range = rs_percentile.ge(0.65)
    low_relative_range = rs_percentile.le(0.35)

    entry_rotation_long = _dedupe_events(
        cross_outperforming & long_score.ge(cfg.entry_score_min * 0.75) & improving_slope,
        cfg.entry_cooldown_bars,
    )
    entry_persistent_strength_long = _dedupe_events(
        rel_short.gt(cfg.min_outperformance)
        & rel_medium.gt(cfg.min_outperformance)
        & high_relative_range
        & long_score.ge(cfg.entry_score_min)
        & score_margin.ge(0.05),
        cfg.entry_cooldown_bars,
    )
    weakness_rotation = _dedupe_events(
        cross_underperforming & short_score.ge(cfg.entry_score_min * 0.75) & weakening_slope,
        cfg.entry_cooldown_bars,
    )
    persistent_weakness = _dedupe_events(
        rel_short.lt(-cfg.min_outperformance)
        & rel_medium.lt(-cfg.min_outperformance)
        & low_relative_range
        & short_score.ge(cfg.entry_score_min)
        & score_margin.le(-0.05),
        cfg.entry_cooldown_bars,
    )
    base_long_signal = entry_rotation_long | entry_persistent_strength_long
    base_weakness_signal = weakness_rotation | persistent_weakness
    reference_ok_for_long = reference_trend_z.ge(cfg.long_reference_min_z)
    target_ok_for_long = target_trend_z.ge(cfg.long_target_min_z)
    reference_falling_for_short = reference_trend_z.le(cfg.short_reference_max_z)
    target_falling_for_short = target_trend_z.le(cfg.short_target_max_z)
    relative_weak_for_short = relative_spread_z.le(cfg.short_relative_max_z)
    go_long_signal = base_long_signal & reference_ok_for_long & target_ok_for_long
    go_short_signal = (
        base_weakness_signal
        & reference_falling_for_short
        & target_falling_for_short
        & relative_weak_for_short
    )
    long_caution_signal = base_long_signal & reference_trend_z.le(cfg.caution_reference_max_z)
    go_long = pd.Series(
        np.where(go_long_signal, 1.0, 0.0),
        index=frame.index,
        dtype="float64",
    )
    go_short = pd.Series(
        np.where(go_short_signal, 1.0, 0.0),
        index=frame.index,
        dtype="float64",
    )
    long_caution = pd.Series(
        np.where(long_caution_signal, 1.0, 0.0),
        index=frame.index,
        dtype="float64",
    )
    hold_long = market_context.gt(0.0) & rs_slope.ge(0.0) & ~cross_underperforming & reference_ok_for_long & target_ok_for_long
    hold_short = (
        market_context.lt(0.0)
        & rs_slope.le(0.0)
        & ~cross_outperforming
        & reference_falling_for_short
        & target_falling_for_short
        & relative_weak_for_short
    )
    exit_long = _dedupe_events(
        cross_underperforming | (short_score.gt(long_score + 0.10) & weakening_slope),
        cfg.entry_cooldown_bars,
    )
    exit_short = _dedupe_events(
        cross_outperforming | (long_score.gt(short_score + 0.10) & improving_slope),
        cfg.entry_cooldown_bars,
    )

    new_cols = {
        f"{p}_benchmark_close": bench_close,
        f"{p}_line": rs_line,
        f"{p}_slope": rs_slope,
        f"{p}_ret_short": rel_short,
        f"{p}_ret_medium": rel_medium,
        f"{p}_ret_long": rel_long,
        f"{p}_percentile": rs_percentile,
        f"{p}_target_trend_z": target_trend_z,
        f"{p}_reference_trend_z": reference_trend_z,
        f"{p}_relative_spread_z": relative_spread_z,
        f"{p}_outperforming": outperforming,
        f"{p}_underperforming": underperforming,
        f"{p}_market_context": market_context,
        f"{p}_entry_rotation_long": entry_rotation_long,
        f"{p}_entry_persistent_strength_long": entry_persistent_strength_long,
        f"{p}_weakness_rotation": weakness_rotation,
        f"{p}_persistent_weakness": persistent_weakness,
        f"{p}_go_long": go_long,
        f"{p}_go_short": go_short,
        f"{p}_long_caution": long_caution,
        f"{p}_hold_long": hold_long,
        f"{p}_hold_short": hold_short,
        f"{p}_exit_long": exit_long,
        f"{p}_exit_short": exit_short,
        f"{p}_score_long": long_score,
        f"{p}_score_short": short_score,
        f"{p}_score_abs": abs_score,
        f"{p}_state": score_state,
    }
    existing = [col for col in frame.columns if str(col).startswith(f"{p}_")]
    base = frame.drop(columns=existing).copy() if existing else frame.copy()
    return pd.concat([base, pd.DataFrame(new_cols, index=frame.index)], axis=1)


def _benchmark_close(benchmark: DataFrame | Series, cfg: RelativeStrengthConfig, index: pd.Index) -> Series:
    if isinstance(benchmark, Series):
        series = pd.to_numeric(benchmark, errors="coerce")
    else:
        if cfg.benchmark_close not in benchmark.columns:
            raise ValueError(f"benchmark is missing close column: {cfg.benchmark_close}")
        series = pd.to_numeric(benchmark[cfg.benchmark_close], errors="coerce")
    return series.reindex(index).ffill()


def _rolling_rank(series: Series, window: int) -> Series:
    current = series
    rolling_min = series.rolling(window, min_periods=2).min()
    rolling_max = series.rolling(window, min_periods=2).max()
    return _clip01((current - rolling_min) / (rolling_max - rolling_min).replace(0.0, np.nan))


def _trend_z(close: Series, window: int) -> Series:
    returns = close.pct_change(fill_method=None)
    window_return = close.pct_change(window, fill_method=None)
    volatility = returns.rolling(window, min_periods=max(3, window // 3)).std() * np.sqrt(float(window))
    return window_return / volatility.replace(0.0, np.nan)


def _relative_spread_z(close: Series, benchmark_close: Series, window: int) -> Series:
    spread_return = close.pct_change(fill_method=None) - benchmark_close.pct_change(fill_method=None)
    window_spread = close.pct_change(window, fill_method=None) - benchmark_close.pct_change(window, fill_method=None)
    spread_volatility = spread_return.rolling(window, min_periods=max(3, window // 3)).std() * np.sqrt(float(window))
    return window_spread / spread_volatility.replace(0.0, np.nan)


def _resolve_config(config: RelativeStrengthConfig | None, **overrides: object) -> RelativeStrengthConfig:
    base = config or RelativeStrengthConfig()
    values = dict(base.__dict__)
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return RelativeStrengthConfig(**values)


def _validate_config(cfg: RelativeStrengthConfig) -> None:
    if min(
        cfg.short_window,
        cfg.medium_window,
        cfg.long_window,
        cfg.percentile_window,
        cfg.context_window,
        cfg.entry_cooldown_bars,
    ) < 2:
        raise ValueError("windows must be at least 2")
    if not cfg.short_window < cfg.medium_window < cfg.long_window:
        raise ValueError("expected short_window < medium_window < long_window")
    if cfg.slope_scale <= 0.0:
        raise ValueError("slope_scale must be positive")
    bounded = [
        cfg.entry_score_min,
        cfg.context_full_min,
        cfg.context_full_margin,
        cfg.context_soft_min,
        cfg.context_soft_margin,
    ]
    if any(value < 0.0 or value > 1.0 for value in bounded):
        raise ValueError("score/context thresholds must be between 0.0 and 1.0")
    if not cfg.prefix:
        raise ValueError("prefix must not be empty")


def _validate_dataframe(dataframe: DataFrame) -> None:
    if "close" not in dataframe.columns:
        raise ValueError("DataFrame is missing close column")


def _clip01(series: Series) -> Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0).fillna(0.0)


def _dedupe_events(mask: Series, cooldown_bars: int) -> Series:
    event = pd.Series(mask, index=mask.index).astype("boolean").fillna(False).astype(bool)
    previous_count = event.astype("float64").shift(1).rolling(int(cooldown_bars), min_periods=1).sum().fillna(0.0)
    return event & previous_count.eq(0.0)


def _num(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")

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
    - ``*_score_abs`` is max(long, short). It measures benchmark-relative edge
      strength.
    - ``*_state`` is -1/0/1 directional lean and is not normalized.

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
        benchmark_close=benchmark_close,
        prefix=prefix,
    )
    _validate_config(cfg)
    _validate_dataframe(dataframe)

    frame = dataframe.copy()
    p = cfg.prefix
    close = _num(frame, "close").replace(0.0, np.nan)
    bench_close = _benchmark_close(benchmark, cfg, frame.index).replace(0.0, np.nan)

    pair_ret_short = close.pct_change(cfg.short_window)
    bench_ret_short = bench_close.pct_change(cfg.short_window)
    pair_ret_medium = close.pct_change(cfg.medium_window)
    bench_ret_medium = bench_close.pct_change(cfg.medium_window)
    pair_ret_long = close.pct_change(cfg.long_window)
    bench_ret_long = bench_close.pct_change(cfg.long_window)

    rel_short = pair_ret_short - bench_ret_short
    rel_medium = pair_ret_medium - bench_ret_medium
    rel_long = pair_ret_long - bench_ret_long
    rs_line = close / bench_close
    rs_slope = rs_line.pct_change(cfg.medium_window)
    rs_percentile = _rolling_rank(rs_line, cfg.percentile_window)

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
    score_state = pd.Series(
        np.select([long_score.gt(short_score), short_score.gt(long_score)], [1.0, -1.0], default=0.0),
        index=frame.index,
    )

    new_cols = {
        f"{p}_benchmark_close": bench_close,
        f"{p}_line": rs_line,
        f"{p}_slope": rs_slope,
        f"{p}_ret_short": rel_short,
        f"{p}_ret_medium": rel_medium,
        f"{p}_ret_long": rel_long,
        f"{p}_percentile": rs_percentile,
        f"{p}_outperforming": outperforming,
        f"{p}_underperforming": underperforming,
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


def _resolve_config(config: RelativeStrengthConfig | None, **overrides: object) -> RelativeStrengthConfig:
    base = config or RelativeStrengthConfig()
    values = dict(base.__dict__)
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return RelativeStrengthConfig(**values)


def _validate_config(cfg: RelativeStrengthConfig) -> None:
    if min(cfg.short_window, cfg.medium_window, cfg.long_window, cfg.percentile_window) < 2:
        raise ValueError("windows must be at least 2")
    if not cfg.short_window < cfg.medium_window < cfg.long_window:
        raise ValueError("expected short_window < medium_window < long_window")
    if cfg.slope_scale <= 0.0:
        raise ValueError("slope_scale must be positive")
    if not cfg.prefix:
        raise ValueError("prefix must not be empty")


def _validate_dataframe(dataframe: DataFrame) -> None:
    if "close" not in dataframe.columns:
        raise ValueError("DataFrame is missing close column")


def _clip01(series: Series) -> Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0).fillna(0.0)


def _num(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")

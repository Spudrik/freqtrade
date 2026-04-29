from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


@dataclass(frozen=True)
class PatternStructureConfig:
    """Vectorized flag/pennant-style pattern evidence for Freqtrade.

    The detector is intentionally built on structure that a strategy can already
    compute: pivots, trendlines, volume, and OHLCV. It looks for a strong impulse
    followed by a controlled, contracting consolidation. Scores are normalized
    0..1 and are evidence only, not final trading signals.

    Score meaning:
    - ``*_score_long`` rises when a bullish impulse is followed by controlled
      retracement, range contraction, dry volume, bullish pivot context, and a
      potential upside breakout.
    - ``*_score_short`` mirrors that for bearish impulses and downside breaks.
    - ``*_score_abs`` is max(long, short). It measures pattern quality.
    - ``*_state`` is -1/0/1 directional lean and is not normalized.

    Flags and pennants are treated as variants of pivot/trendline behavior:
    impulse first, then corrective channel or compression. Missing pivot or
    trendline columns are neutral so the module can still be validated on raw
    OHLCV.
    """

    impulse_window: int = 24
    consolidation_window: int = 18
    atr_period: int = 14
    impulse_atr_min: float = 2.0
    impulse_pct_min: float = 0.025
    max_retrace_pct: float = 0.62
    min_range_contraction: float = 0.15
    dry_volume_rvol_max: float = 0.90
    breakout_buffer_pct: float = 0.001
    pivot_prefix: str = "pa"
    trendline_prefix: str = "tl"
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
    max_retrace_pct: float | None = None,
    min_range_contraction: float | None = None,
    dry_volume_rvol_max: float | None = None,
    breakout_buffer_pct: float | None = None,
    pivot_prefix: str | None = None,
    trendline_prefix: str | None = None,
    output_prefix: str | None = None,
) -> DataFrame:
    """Append flag, pennant, and breakout-readiness columns.

    Expected Freqtrade usage is to call pivot/trendline modules first, then this
    module. Missing pivot/trendline columns are treated as neutral so the module
    can still be tested against raw OHLCV.

    Main outputs are impulse scores, contraction score, flag/pennant booleans,
    breakout-readiness booleans, and normalized long/short scores.
    """

    cfg = _resolve_config(
        config,
        impulse_window=impulse_window,
        consolidation_window=consolidation_window,
        atr_period=atr_period,
        impulse_atr_min=impulse_atr_min,
        impulse_pct_min=impulse_pct_min,
        max_retrace_pct=max_retrace_pct,
        min_range_contraction=min_range_contraction,
        dry_volume_rvol_max=dry_volume_rvol_max,
        breakout_buffer_pct=breakout_buffer_pct,
        pivot_prefix=pivot_prefix,
        trendline_prefix=trendline_prefix,
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

    impulse_low = low.rolling(cfg.impulse_window, min_periods=2).min().shift(cfg.consolidation_window)
    impulse_high = high.rolling(cfg.impulse_window, min_periods=2).max().shift(cfg.consolidation_window)
    impulse_end_close = close.shift(cfg.consolidation_window)
    impulse_up_move = impulse_end_close - impulse_low
    impulse_down_move = impulse_high - impulse_end_close
    impulse_up_valid = impulse_up_move.gt(0.0)
    impulse_down_valid = impulse_down_move.gt(0.0)
    impulse_up_score = _clip01(
        0.5 * _safe_div(impulse_up_move, atr * cfg.impulse_atr_min)
        + 0.5 * _safe_div(impulse_up_move, close * cfg.impulse_pct_min)
    ).where(impulse_up_valid, 0.0)
    impulse_down_score = _clip01(
        0.5 * _safe_div(impulse_down_move, atr * cfg.impulse_atr_min)
        + 0.5 * _safe_div(impulse_down_move, close * cfg.impulse_pct_min)
    ).where(impulse_down_valid, 0.0)

    cons_high = high.rolling(cfg.consolidation_window, min_periods=2).max()
    cons_low = low.rolling(cfg.consolidation_window, min_periods=2).min()
    cons_range = cons_high - cons_low
    prior_range = (high.rolling(cfg.impulse_window, min_periods=2).max() - low.rolling(cfg.impulse_window, min_periods=2).min()).shift(cfg.consolidation_window)
    range_contraction = _clip01(1.0 - _safe_div(cons_range, prior_range))
    contraction_ok = range_contraction >= cfg.min_range_contraction

    retrace_long = _safe_div(impulse_end_close - cons_low, impulse_up_move.abs())
    retrace_short = _safe_div(cons_high - impulse_end_close, impulse_down_move.abs())
    controlled_long = retrace_long.between(0.0, cfg.max_retrace_pct)
    controlled_short = retrace_short.between(0.0, cfg.max_retrace_pct)

    short_volume = volume.rolling(cfg.consolidation_window, min_periods=1).mean()
    long_volume = volume.rolling(cfg.impulse_window + cfg.consolidation_window, min_periods=2).mean()
    dry_volume = _safe_div(short_volume, long_volume) <= cfg.dry_volume_rvol_max

    tl_compression = _num(frame, f"{cfg.trendline_prefix}_channel_compression")
    tl_compression_score = _clip01(1.0 - tl_compression.fillna(1.0))
    pa_up_sequence = _num(frame, f"{cfg.pivot_prefix}_ms_up_sequence_score").fillna(0.0)
    pa_down_sequence = _num(frame, f"{cfg.pivot_prefix}_ms_down_sequence_score").fillna(0.0)

    flag_long = impulse_up_valid & impulse_up_score.gt(0.5) & controlled_long & contraction_ok & dry_volume
    flag_short = impulse_down_valid & impulse_down_score.gt(0.5) & controlled_short & contraction_ok & dry_volume
    pennant_long = flag_long & tl_compression_score.gt(0.35)
    pennant_short = flag_short & tl_compression_score.gt(0.35)
    raw_breakout_long = close.gt(cons_high.shift(1) * (1.0 + cfg.breakout_buffer_pct)) & close.gt(open_)
    raw_breakout_short = close.lt(cons_low.shift(1) * (1.0 - cfg.breakout_buffer_pct)) & close.lt(open_)
    recent_flag_long = _rolling_count(flag_long | pennant_long, cfg.consolidation_window).ge(1.0)
    recent_flag_short = _rolling_count(flag_short | pennant_short, cfg.consolidation_window).ge(1.0)
    breakout_long = raw_breakout_long & recent_flag_long
    breakout_short = raw_breakout_short & recent_flag_short

    long_score = _clip01(
        0.30 * impulse_up_score
        + 0.25 * range_contraction
        + 0.15 * dry_volume.astype("float64")
        + 0.15 * pa_up_sequence
        + 0.10 * breakout_long.astype("float64")
        + 0.05 * tl_compression_score
    )
    short_score = _clip01(
        0.30 * impulse_down_score
        + 0.25 * range_contraction
        + 0.15 * dry_volume.astype("float64")
        + 0.15 * pa_down_sequence
        + 0.10 * breakout_short.astype("float64")
        + 0.05 * tl_compression_score
    )
    abs_score = pd.concat([long_score, short_score], axis=1).max(axis=1)
    score_state = pd.Series(
        np.select([long_score.gt(short_score), short_score.gt(long_score)], [1.0, -1.0], default=0.0),
        index=frame.index,
    )
    margin = long_score - short_score
    full_bull = long_score.ge(0.56) & margin.ge(0.08) & (recent_flag_long | breakout_long)
    full_bear = short_score.ge(0.56) & margin.le(-0.08) & (recent_flag_short | breakout_short)
    bullish_chop = ~full_bull & ~full_bear & long_score.ge(0.40) & margin.ge(0.04) & recent_flag_long
    bearish_chop = ~full_bull & ~full_bear & short_score.ge(0.40) & margin.le(-0.04) & recent_flag_short
    market_context = pd.Series(
        np.select([full_bull, full_bear, bullish_chop, bearish_chop], [2, -2, 1, -1], default=0),
        index=frame.index,
        dtype="int8",
    )

    cooldown = max(3, int(cfg.consolidation_window) // 3)
    flag_event_long = _dedupe_events(flag_long & ~flag_long.shift(1, fill_value=False), cooldown)
    flag_event_short = _dedupe_events(flag_short & ~flag_short.shift(1, fill_value=False), cooldown)
    pennant_event_long = _dedupe_events(pennant_long & ~pennant_long.shift(1, fill_value=False), cooldown)
    pennant_event_short = _dedupe_events(pennant_short & ~pennant_short.shift(1, fill_value=False), cooldown)
    entry_flag_breakout_long = breakout_long & flag_long & long_score.ge(short_score)
    entry_pennant_breakout_long = breakout_long & pennant_long & long_score.ge(short_score)
    entry_flag_breakdown_short = breakout_short & flag_short & short_score.ge(long_score)
    entry_pennant_breakdown_short = breakout_short & pennant_short & short_score.ge(long_score)
    suggested_entry_long = entry_flag_breakout_long | entry_pennant_breakout_long
    suggested_entry_short = entry_flag_breakdown_short | entry_pennant_breakdown_short
    hold_long = recent_flag_long & long_score.ge(short_score - 0.05) & ~breakout_short
    hold_short = recent_flag_short & short_score.ge(long_score - 0.05) & ~breakout_long
    exit_long = breakout_short | flag_event_short
    exit_short = breakout_long | flag_event_long

    new_cols = {
        f"{p}_impulse_up_score": impulse_up_score,
        f"{p}_impulse_down_score": impulse_down_score,
        f"{p}_impulse_end_close": impulse_end_close,
        f"{p}_impulse_up_move": impulse_up_move.where(impulse_up_valid),
        f"{p}_impulse_down_move": impulse_down_move.where(impulse_down_valid),
        f"{p}_range_contraction_score": range_contraction,
        f"{p}_retrace_long": retrace_long.where(impulse_up_valid),
        f"{p}_retrace_short": retrace_short.where(impulse_down_valid),
        f"{p}_dry_volume": dry_volume,
        f"{p}_flag_state_long": flag_long,
        f"{p}_flag_state_short": flag_short,
        f"{p}_pennant_state_long": pennant_long,
        f"{p}_pennant_state_short": pennant_short,
        f"{p}_flag_long": flag_event_long,
        f"{p}_flag_short": flag_event_short,
        f"{p}_pennant_long": pennant_event_long,
        f"{p}_pennant_short": pennant_event_short,
        f"{p}_raw_breakout_long": raw_breakout_long,
        f"{p}_raw_breakout_short": raw_breakout_short,
        f"{p}_breakout_long": _dedupe_events(breakout_long, cooldown),
        f"{p}_breakout_short": _dedupe_events(breakout_short, cooldown),
        f"{p}_entry_flag_breakout_long": _dedupe_events(entry_flag_breakout_long, cooldown),
        f"{p}_entry_pennant_breakout_long": _dedupe_events(entry_pennant_breakout_long, cooldown),
        f"{p}_entry_flag_breakdown_short": _dedupe_events(entry_flag_breakdown_short, cooldown),
        f"{p}_entry_pennant_breakdown_short": _dedupe_events(entry_pennant_breakdown_short, cooldown),
        f"{p}_hold_long": hold_long.fillna(False),
        f"{p}_hold_short": hold_short.fillna(False),
        f"{p}_exit_long": _dedupe_events(exit_long, cooldown),
        f"{p}_exit_short": _dedupe_events(exit_short, cooldown),
        f"{p}_suggested_entry_long": _dedupe_events(suggested_entry_long, cooldown),
        f"{p}_suggested_entry_short": _dedupe_events(suggested_entry_short, cooldown),
        f"{p}_market_context": market_context,
        f"{p}_score_long": long_score,
        f"{p}_score_short": short_score,
        f"{p}_score_abs": abs_score,
        f"{p}_state": score_state,
    }
    existing = [col for col in frame.columns if str(col).startswith(f"{p}_")]
    base = frame.drop(columns=existing).copy() if existing else frame.copy()
    return pd.concat([base, pd.DataFrame(new_cols, index=frame.index)], axis=1)


def _resolve_config(config: PatternStructureConfig | None, **overrides: object) -> PatternStructureConfig:
    base = config or PatternStructureConfig()
    values = dict(base.__dict__)
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return PatternStructureConfig(**values)


def _validate_config(cfg: PatternStructureConfig) -> None:
    if min(cfg.impulse_window, cfg.consolidation_window, cfg.atr_period) < 2:
        raise ValueError("windows must be at least 2")
    if cfg.impulse_atr_min <= 0.0 or cfg.impulse_pct_min <= 0.0:
        raise ValueError("impulse thresholds must be positive")
    if not 0.0 < cfg.max_retrace_pct <= 1.0:
        raise ValueError("max_retrace_pct must be between 0 and 1")
    if not 0.0 <= cfg.min_range_contraction <= 1.0:
        raise ValueError("min_range_contraction must be between 0 and 1")
    if cfg.dry_volume_rvol_max <= 0.0:
        raise ValueError("dry_volume_rvol_max must be positive")
    if cfg.breakout_buffer_pct < 0.0:
        raise ValueError("breakout_buffer_pct must be non-negative")


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


def _clip01(series: Series) -> Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0).fillna(0.0)


def _num(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


@dataclass(frozen=True)
class VolatilityCycleConfig:
    """Compression, expansion, and exhaustion evidence for Freqtrade.

    Volatility tends to cycle from contraction to expansion and then exhaustion.
    This module keeps those phases separate so they can be validated before any
    confluence layer. All score columns are normalized to 0..1.

    Score meaning:
    - ``*_compression_score`` rises when ATR, range, and volume are contracting.
    - ``*_expansion_score`` rises when range and volume expand together.
    - ``*_exhaustion_score`` rises when large range, high relative volume, and
      extreme close location suggest a stretched move.
    - ``*_score_long`` rises when prior compression resolves upward or downside
      exhaustion suggests mean-reversion potential.
    - ``*_score_short`` mirrors that for downside expansion or upside exhaustion.
    - ``*_state`` is -1/0/1 directional lean and is not normalized.

    The module is deliberately independent of confluence. It should be validated
    by checking whether compression/expansion/exhaustion states have measurable
    forward-return behavior on each timeframe.
    """

    atr_period: int = 14
    short_window: int = 12
    long_window: int = 96
    volume_window: int = 48
    compression_atr_ratio: float = 0.75
    expansion_atr_ratio: float = 1.35
    dry_volume_rvol: float = 0.80
    expansion_volume_rvol: float = 1.25
    exhaustion_range_atr: float = 1.80
    exhaustion_volume_rvol: float = 2.00
    close_location_extreme: float = 0.55
    compression_release_lookback: int = 8
    entry_cooldown_bars: int = 8
    context_window: int = 24
    channel_prefix: str = "tlv2"
    channel_label: str = "local_channel"
    channel_rank: int = 0
    output_prefix: str = "vc"


def add_volatility_cycles(
    dataframe: DataFrame,
    config: VolatilityCycleConfig | None = None,
    *,
    atr_period: int | None = None,
    short_window: int | None = None,
    long_window: int | None = None,
    volume_window: int | None = None,
    compression_atr_ratio: float | None = None,
    expansion_atr_ratio: float | None = None,
    dry_volume_rvol: float | None = None,
    expansion_volume_rvol: float | None = None,
    exhaustion_range_atr: float | None = None,
    exhaustion_volume_rvol: float | None = None,
    close_location_extreme: float | None = None,
    compression_release_lookback: int | None = None,
    entry_cooldown_bars: int | None = None,
    context_window: int | None = None,
    channel_prefix: str | None = None,
    channel_label: str | None = None,
    channel_rank: int | None = None,
    output_prefix: str | None = None,
) -> DataFrame:
    """Append vectorized volatility-cycle columns and normalized scores."""

    cfg = _resolve_config(
        config,
        atr_period=atr_period,
        short_window=short_window,
        long_window=long_window,
        volume_window=volume_window,
        compression_atr_ratio=compression_atr_ratio,
        expansion_atr_ratio=expansion_atr_ratio,
        dry_volume_rvol=dry_volume_rvol,
        expansion_volume_rvol=expansion_volume_rvol,
        exhaustion_range_atr=exhaustion_range_atr,
        exhaustion_volume_rvol=exhaustion_volume_rvol,
        close_location_extreme=close_location_extreme,
        compression_release_lookback=compression_release_lookback,
        entry_cooldown_bars=entry_cooldown_bars,
        context_window=context_window,
        channel_prefix=channel_prefix,
        channel_label=channel_label,
        channel_rank=channel_rank,
        output_prefix=output_prefix,
    )
    _validate_config(cfg)
    _validate_dataframe(dataframe)

    frame = dataframe.copy()
    p = cfg.output_prefix
    open_ = _num(frame, "open")
    high = _num(frame, "high")
    low = _num(frame, "low")
    close = _num(frame, "close").replace(0.0, np.nan)
    volume = _num(frame, "volume").clip(lower=0.0).fillna(0.0)

    tr = _true_range(frame)
    atr = tr.rolling(cfg.atr_period, min_periods=1).mean()
    short_atr = tr.rolling(cfg.short_window, min_periods=1).mean()
    long_atr = tr.rolling(cfg.long_window, min_periods=2).median()
    atr_ratio = _safe_div(short_atr, long_atr)

    short_range = (high.rolling(cfg.short_window, min_periods=2).max() - low.rolling(cfg.short_window, min_periods=2).min())
    long_range = (high.rolling(cfg.long_window, min_periods=2).max() - low.rolling(cfg.long_window, min_periods=2).min())
    range_ratio = _safe_div(short_range, long_range)
    volume_mean = volume.rolling(cfg.volume_window, min_periods=1).mean()
    rvol = _safe_div(volume, volume_mean)

    candle_range = (high - low).clip(lower=0.0)
    close_location = ((_safe_div(close - low, candle_range) * 2.0) - 1.0).clip(-1.0, 1.0)
    body_direction = np.sign(close - open_)
    channel_base = f"{cfg.channel_prefix}_{cfg.channel_label}"
    channel_active = _bool(frame, f"{channel_base}_active_rank{int(cfg.channel_rank)}")
    channel_score = _num(frame, f"{channel_base}_score_rank{int(cfg.channel_rank)}")
    channel_shape = _num(frame, f"{channel_base}_shape_rank{int(cfg.channel_rank)}")
    channel_width_change = _num(frame, f"{channel_base}_width_change_pct_rank{int(cfg.channel_rank)}")

    atr_compression = _clip01((cfg.compression_atr_ratio - atr_ratio) / max(cfg.compression_atr_ratio, 1e-9))
    range_compression = _clip01(1.0 - range_ratio)
    volume_dry = _clip01((cfg.dry_volume_rvol - rvol) / max(cfg.dry_volume_rvol, 1e-9))
    channel_convergence_score = _clip01((-channel_width_change.fillna(0.0)) / 0.35)
    channel_compression_score = (
        _clip01(0.60 * channel_convergence_score + 0.40 * channel_score.fillna(0.0))
        .where(channel_active & channel_shape.eq(1.0), 0.0)
    )
    compression_score = _clip01(
        0.40 * atr_compression
        + 0.30 * range_compression
        + 0.20 * volume_dry
        + 0.10 * channel_compression_score
    )

    range_expansion = _clip01(
        (atr_ratio - 1.0) / max(cfg.expansion_atr_ratio - 1.0, 1e-9)
    )
    volume_expansion = _clip01(
        (rvol - 1.0) / max(cfg.expansion_volume_rvol - 1.0, 1e-9)
    )
    expansion_score = _clip01(0.60 * range_expansion + 0.40 * volume_expansion)
    range_atr = _safe_div(candle_range, atr)
    exhaustion_range_score = _clip01(
        (range_atr - 1.0) / max(cfg.exhaustion_range_atr - 1.0, 1e-9)
    )
    exhaustion_volume_score = _clip01(
        (rvol - 1.0) / max(cfg.exhaustion_volume_rvol - 1.0, 1e-9)
    )
    exhaustion_score = _clip01(
        0.50 * exhaustion_range_score
        + 0.30 * exhaustion_volume_score
        + 0.20 * close_location.abs().ge(cfg.close_location_extreme).astype("float64")
    )

    expansion_long = expansion_score.gt(0.55) & body_direction.gt(0.0) & close_location.gt(0.0)
    expansion_short = expansion_score.gt(0.55) & body_direction.lt(0.0) & close_location.lt(0.0)
    exhaustion_up = exhaustion_score.gt(0.65) & close_location.gt(cfg.close_location_extreme)
    exhaustion_down = exhaustion_score.gt(0.65) & close_location.lt(-cfg.close_location_extreme)

    recent_compression = (
        compression_score.shift(1)
        .rolling(cfg.compression_release_lookback, min_periods=1)
        .max()
        .fillna(0.0)
    )
    expansion_long_setup = recent_compression * expansion_long.astype("float64")
    expansion_short_setup = recent_compression * expansion_short.astype("float64")
    long_score = _clip01(
        0.60 * expansion_long_setup
        + 0.25 * expansion_long.astype("float64")
        + 0.15 * exhaustion_down.astype("float64")
    )
    short_score = _clip01(
        0.60 * expansion_short_setup
        + 0.25 * expansion_short.astype("float64")
        + 0.15 * exhaustion_up.astype("float64")
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
    full_bull = long_score.ge(0.35) & score_margin.ge(0.08) & (
        expansion_long_setup.ge(0.20) | exhaustion_down
    )
    full_bear = short_score.ge(0.35) & score_margin.le(-0.08) & (
        expansion_short_setup.ge(0.20) | exhaustion_up
    )
    bullish_chop = ~full_bull & ~full_bear & context_long.ge(0.16) & context_margin.ge(0.03)
    bearish_chop = ~full_bull & ~full_bear & context_short.ge(0.16) & context_margin.le(-0.03)
    market_context = pd.Series(
        np.select(
            [full_bull, bullish_chop, full_bear, bearish_chop],
            [2.0, 1.0, -2.0, -1.0],
            default=0.0,
        ),
        index=frame.index,
        dtype="float64",
    )
    entry_breakout_long = _dedupe_events(expansion_long_setup.ge(0.25), cfg.entry_cooldown_bars)
    entry_exhaustion_reversal_long = _dedupe_events(
        exhaustion_down & long_score.ge(short_score),
        cfg.entry_cooldown_bars,
    )
    entry_breakdown_short = _dedupe_events(expansion_short_setup.ge(0.25), cfg.entry_cooldown_bars)
    entry_exhaustion_reversal_short = _dedupe_events(
        exhaustion_up & short_score.ge(long_score),
        cfg.entry_cooldown_bars,
    )
    suggested_entry_long = entry_breakout_long | entry_exhaustion_reversal_long
    suggested_entry_short = entry_breakdown_short | entry_exhaustion_reversal_short
    hold_long = (market_context.gt(0.0) | expansion_long) & ~exhaustion_up
    hold_short = (market_context.lt(0.0) | expansion_short) & ~exhaustion_down
    exit_long = _dedupe_events(
        exhaustion_up | expansion_short_setup.ge(0.25) | short_score.gt(long_score + 0.12),
        cfg.entry_cooldown_bars,
    )
    exit_short = _dedupe_events(
        exhaustion_down | expansion_long_setup.ge(0.25) | long_score.gt(short_score + 0.12),
        cfg.entry_cooldown_bars,
    )
    compression_phase = compression_score.ge(0.35) & expansion_score.le(0.45)
    expansion_phase = expansion_score.ge(0.65)
    exhaustion_phase = exhaustion_score.ge(0.65)
    cycle_state = pd.Series(
        np.select(
            [exhaustion_phase, compression_phase, expansion_phase],
            [3.0, 1.0, 2.0],
            default=0.0,
        ),
        index=frame.index,
    )

    new_cols = {
        f"{p}_atr": atr,
        f"{p}_atr_ratio": atr_ratio,
        f"{p}_range_ratio": range_ratio,
        f"{p}_rvol": rvol,
        f"{p}_range_atr": range_atr,
        f"{p}_compression_score": compression_score,
        f"{p}_expansion_score": expansion_score,
        f"{p}_exhaustion_score": exhaustion_score,
        f"{p}_cycle_state": cycle_state,
        f"{p}_compression_phase": compression_phase,
        f"{p}_expansion_phase": expansion_phase,
        f"{p}_exhaustion_phase": exhaustion_phase,
        f"{p}_expansion_long": expansion_long,
        f"{p}_expansion_short": expansion_short,
        f"{p}_recent_compression": recent_compression,
        f"{p}_expansion_long_setup": expansion_long_setup,
        f"{p}_expansion_short_setup": expansion_short_setup,
        f"{p}_exhaustion_up": exhaustion_up,
        f"{p}_exhaustion_down": exhaustion_down,
        f"{p}_market_context": market_context,
        f"{p}_entry_breakout_long": entry_breakout_long,
        f"{p}_entry_exhaustion_reversal_long": entry_exhaustion_reversal_long,
        f"{p}_entry_breakdown_short": entry_breakdown_short,
        f"{p}_entry_exhaustion_reversal_short": entry_exhaustion_reversal_short,
        f"{p}_hold_long": hold_long,
        f"{p}_hold_short": hold_short,
        f"{p}_exit_long": exit_long,
        f"{p}_exit_short": exit_short,
        f"{p}_suggested_entry_long": suggested_entry_long,
        f"{p}_suggested_entry_short": suggested_entry_short,
        f"{p}_score_long": long_score,
        f"{p}_score_short": short_score,
        f"{p}_score_abs": abs_score,
        f"{p}_state": score_state,
    }
    existing = [col for col in frame.columns if str(col).startswith(f"{p}_")]
    base = frame.drop(columns=existing).copy() if existing else frame.copy()
    return pd.concat([base, pd.DataFrame(new_cols, index=frame.index)], axis=1)


def _resolve_config(config: VolatilityCycleConfig | None, **overrides: object) -> VolatilityCycleConfig:
    base = config or VolatilityCycleConfig()
    values = dict(base.__dict__)
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return VolatilityCycleConfig(**values)


def _validate_config(cfg: VolatilityCycleConfig) -> None:
    if min(
        cfg.atr_period,
        cfg.short_window,
        cfg.long_window,
        cfg.volume_window,
        cfg.compression_release_lookback,
        cfg.entry_cooldown_bars,
        cfg.context_window,
    ) < 2:
        raise ValueError("windows must be at least 2")
    if cfg.short_window >= cfg.long_window:
        raise ValueError("short_window must be smaller than long_window")
    positives = [
        cfg.compression_atr_ratio,
        cfg.expansion_atr_ratio,
        cfg.dry_volume_rvol,
        cfg.expansion_volume_rvol,
        cfg.exhaustion_range_atr,
        cfg.exhaustion_volume_rvol,
    ]
    if any(value <= 0.0 for value in positives):
        raise ValueError("cycle thresholds must be positive")
    if not 0.0 <= cfg.close_location_extreme <= 1.0:
        raise ValueError("close_location_extreme must be between 0.0 and 1.0")
    if int(cfg.channel_rank) < 0:
        raise ValueError("channel_rank must be non-negative")
    if not cfg.channel_prefix or not cfg.channel_label:
        raise ValueError("channel_prefix and channel_label must be set")


def _validate_dataframe(dataframe: DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"DataFrame is missing OHLCV columns: {missing}")


def _true_range(frame: DataFrame) -> Series:
    high = _num(frame, "high")
    low = _num(frame, "low")
    close = _num(frame, "close")
    prev_close = close.shift(1)
    return pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)


def _safe_div(numerator: Series, denominator: Series) -> Series:
    denominator = denominator.where(denominator.abs() > 0.0, np.nan)
    return numerator / denominator


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


def _bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype="bool")
    return pd.Series(frame[column], index=frame.index).fillna(False).astype("bool")

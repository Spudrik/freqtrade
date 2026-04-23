from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


@dataclass(frozen=True)
class PivotStructureConfig:
    """
    Settings for confirmed pivot, trendline, and target-range columns.

    Pivots are only emitted on the confirmation candle. For example, strength 5
    confirms that the candle from five bars ago was a pivot only after the five
    right-side candles have closed, so the current row never uses unknown future
    candles.
    """

    strength: int = 5
    strengths: Sequence[int] = (3, 5, 8, 13)
    atr_period: int = 14
    min_prominence_atr: float = 0.35
    zone_atr_mult: float = 0.35
    zone_pct: float = 0.003
    min_channel_width_pct: float = 0.003
    min_target_distance_pct: float = 0.002
    breakout_buffer_pct: float = 0.001
    prefix: str = "pa"


def add_pivot_structure(
    dataframe: DataFrame,
    config: PivotStructureConfig | None = None,
    *,
    strength: int | None = None,
    strengths: Sequence[int] | None = None,
    atr_period: int | None = None,
    min_prominence_atr: float | None = None,
    zone_atr_mult: float | None = None,
    zone_pct: float | None = None,
    min_channel_width_pct: float | None = None,
    min_target_distance_pct: float | None = None,
    breakout_buffer_pct: float | None = None,
    prefix: str | None = None,
) -> DataFrame:
    """
    Append no-lookahead pivot structure, trendline, and target-range columns.

    Per-strength columns are precomputed so strategies can choose an active
    strength later without recalculating indicators during hyperopt. Active
    aliases without a strength suffix are copied from ``config.strength``.
    """

    cfg = _resolve_config(
        config,
        strength=strength,
        strengths=strengths,
        atr_period=atr_period,
        min_prominence_atr=min_prominence_atr,
        zone_atr_mult=zone_atr_mult,
        zone_pct=zone_pct,
        min_channel_width_pct=min_channel_width_pct,
        min_target_distance_pct=min_target_distance_pct,
        breakout_buffer_pct=breakout_buffer_pct,
        prefix=prefix,
    )
    _validate_config(cfg)
    _validate_dataframe(dataframe)

    high = _num(dataframe["high"])
    low = _num(dataframe["low"])
    close = _num(dataframe["close"]).replace(0, np.nan)
    atr = _atr(dataframe, cfg.atr_period)
    p = cfg.prefix

    zone_width = pd.concat(
        [
            atr * float(cfg.zone_atr_mult),
            close * float(cfg.zone_pct),
        ],
        axis=1,
    ).max(axis=1)
    bar_index = pd.Series(np.arange(len(dataframe), dtype="float64"), index=dataframe.index)

    new_cols: dict[str, Series] = {
        f"{p}_atr": atr,
        f"{p}_zone_width": zone_width,
        f"{p}_bar_index": bar_index,
    }

    strengths_list = _sorted_strengths(cfg)
    for pivot_strength in strengths_list:
        columns = _pivot_columns(
            dataframe.index,
            bar_index,
            high,
            low,
            close,
            atr,
            zone_width,
            pivot_strength,
            float(cfg.min_prominence_atr),
            float(cfg.min_channel_width_pct),
            float(cfg.min_target_distance_pct),
            float(cfg.breakout_buffer_pct),
            p,
        )
        new_cols.update(columns)

    selected = int(cfg.strength)
    if selected not in strengths_list:
        selected = strengths_list[0]
    new_cols.update(_active_alias_columns(new_cols, p, selected))

    existing = [col for col in dataframe.columns if str(col).startswith(f"{p}_")]
    base = dataframe.drop(columns=existing).copy() if existing else dataframe.copy()
    features = pd.DataFrame(new_cols, index=dataframe.index)
    return pd.concat([base, features], axis=1)


def _pivot_columns(
    index: pd.Index,
    bar_index: Series,
    high: Series,
    low: Series,
    close: Series,
    atr: Series,
    zone_width: Series,
    strength: int,
    min_prominence_atr: float,
    min_channel_width_pct: float,
    min_target_distance_pct: float,
    breakout_buffer_pct: float,
    prefix: str,
) -> dict[str, Series]:
    left = int(strength)
    right = int(strength)
    window = left + right + 1

    candidate_high = high.shift(right)
    candidate_low = low.shift(right)
    candidate_atr = atr.shift(right).replace(0, np.nan)

    left_high = high.shift(right + 1).rolling(left, min_periods=left).max()
    right_high = high.rolling(right, min_periods=right).max()
    left_low = low.shift(right + 1).rolling(left, min_periods=left).min()
    right_low = low.rolling(right, min_periods=right).min()
    window_high = high.rolling(window, min_periods=window).max()
    window_low = low.rolling(window, min_periods=window).min()

    high_prominence = (candidate_high - window_low) / candidate_atr
    low_prominence = (window_high - candidate_low) / candidate_atr
    high_confirmed = (
        candidate_high.notna()
        & left_high.notna()
        & right_high.notna()
        & (candidate_high >= left_high)
        & (candidate_high > right_high)
        & (high_prominence >= min_prominence_atr)
    )
    low_confirmed = (
        candidate_low.notna()
        & left_low.notna()
        & right_low.notna()
        & (candidate_low <= left_low)
        & (candidate_low < right_low)
        & (low_prominence >= min_prominence_atr)
    )

    high_event_price = candidate_high.where(high_confirmed)
    low_event_price = candidate_low.where(low_confirmed)
    high_event_index = (bar_index - right).where(high_confirmed)
    low_event_index = (bar_index - right).where(low_confirmed)

    high_state = _last_two_events(high_event_price, high_event_index, index)
    low_state = _last_two_events(low_event_price, low_event_index, index)

    resistance_slope, resistance_line = _line_from_events(
        bar_index,
        high_state["last_price"],
        high_state["last_index"],
        high_state["prev_price"],
        high_state["prev_index"],
    )
    support_slope, support_line = _line_from_events(
        bar_index,
        low_state["last_price"],
        low_state["last_index"],
        low_state["prev_price"],
        low_state["prev_index"],
    )

    resistance_line = resistance_line.clip(lower=0.0)
    support_line = support_line.clip(lower=0.0)
    channel_width = resistance_line - support_line
    channel_width_pct = channel_width / close
    channel_valid = (
        resistance_line.notna()
        & support_line.notna()
        & channel_width.gt(0.0)
        & channel_width_pct.ge(min_channel_width_pct)
    )
    valid_resistance_line = resistance_line.where(channel_valid)
    valid_support_line = support_line.where(channel_valid)
    channel_mid = ((valid_resistance_line + valid_support_line) / 2.0).where(channel_valid)
    channel_width_pct = channel_width_pct.where(channel_valid)
    resistance_slope_pct = (resistance_slope / close).where(channel_valid)
    support_slope_pct = (support_slope / close).where(channel_valid)
    trend_bias = pd.Series(
        np.select(
            [
                channel_valid & (resistance_slope > 0) & (support_slope > 0),
                channel_valid & (resistance_slope < 0) & (support_slope < 0),
            ],
            [1.0, -1.0],
            default=0.0,
        ),
        index=index,
    )
    compression = channel_width_pct / channel_width_pct.rolling(48, min_periods=12).median()

    resistance_zone_upper = valid_resistance_line + zone_width
    resistance_zone_lower = valid_resistance_line - zone_width
    support_zone_upper = valid_support_line + zone_width
    support_zone_lower = valid_support_line - zone_width

    long_target_low = resistance_zone_lower
    long_target_high = resistance_zone_upper
    short_target_low = support_zone_lower
    short_target_high = support_zone_upper
    long_target_distance_pct = (long_target_low - close) / close
    short_target_distance_pct = (close - short_target_high) / close
    long_target_actionable = channel_valid & long_target_distance_pct.ge(min_target_distance_pct)
    short_target_actionable = channel_valid & short_target_distance_pct.ge(min_target_distance_pct)

    market_structure = _market_structure_columns(
        index,
        close,
        high_confirmed,
        low_confirmed,
        high_event_price,
        low_event_price,
        high_state,
        low_state,
        strength,
        breakout_buffer_pct,
        prefix,
    )

    s = int(strength)
    columns = {
        f"{prefix}_pivot_high_confirmed_{s}": high_confirmed.fillna(False),
        f"{prefix}_pivot_low_confirmed_{s}": low_confirmed.fillna(False),
        f"{prefix}_pivot_high_{s}": high_event_price,
        f"{prefix}_pivot_low_{s}": low_event_price,
        f"{prefix}_pivot_high_prominence_{s}": high_prominence.where(high_confirmed),
        f"{prefix}_pivot_low_prominence_{s}": low_prominence.where(low_confirmed),
        f"{prefix}_last_pivot_high_{s}": high_state["last_price"],
        f"{prefix}_last_pivot_low_{s}": low_state["last_price"],
        f"{prefix}_prev_pivot_high_{s}": high_state["prev_price"],
        f"{prefix}_prev_pivot_low_{s}": low_state["prev_price"],
        f"{prefix}_pivot_high_age_{s}": bar_index - high_state["last_index"],
        f"{prefix}_pivot_low_age_{s}": bar_index - low_state["last_index"],
        f"{prefix}_resistance_line_{s}": resistance_line,
        f"{prefix}_support_line_{s}": support_line,
        f"{prefix}_channel_valid_{s}": channel_valid.fillna(False),
        f"{prefix}_resistance_slope_pct_{s}": resistance_slope_pct,
        f"{prefix}_support_slope_pct_{s}": support_slope_pct,
        f"{prefix}_channel_mid_{s}": channel_mid,
        f"{prefix}_channel_width_pct_{s}": channel_width_pct,
        f"{prefix}_channel_compression_{s}": compression,
        f"{prefix}_trend_bias_{s}": trend_bias,
        f"{prefix}_resistance_zone_upper_{s}": resistance_zone_upper,
        f"{prefix}_resistance_zone_lower_{s}": resistance_zone_lower,
        f"{prefix}_support_zone_upper_{s}": support_zone_upper,
        f"{prefix}_support_zone_lower_{s}": support_zone_lower,
        f"{prefix}_long_target_low_{s}": long_target_low.where(long_target_actionable),
        f"{prefix}_long_target_high_{s}": long_target_high.where(long_target_actionable),
        f"{prefix}_short_target_low_{s}": short_target_low.where(short_target_actionable),
        f"{prefix}_short_target_high_{s}": short_target_high.where(short_target_actionable),
        f"{prefix}_long_target_distance_pct_{s}": long_target_distance_pct.where(long_target_actionable),
        f"{prefix}_short_target_distance_pct_{s}": short_target_distance_pct.where(short_target_actionable),
        f"{prefix}_long_target_actionable_{s}": long_target_actionable.fillna(False),
        f"{prefix}_short_target_actionable_{s}": short_target_actionable.fillna(False),
    }
    columns.update(market_structure)
    return columns


def _market_structure_columns(
    index: pd.Index,
    close: Series,
    high_confirmed: Series,
    low_confirmed: Series,
    high_event_price: Series,
    low_event_price: Series,
    high_state: dict[str, Series],
    low_state: dict[str, Series],
    strength: int,
    breakout_buffer_pct: float,
    prefix: str,
) -> dict[str, Series]:
    s = int(strength)
    prior_high = high_state["prev_price"]
    prior_low = low_state["prev_price"]

    higher_high = high_confirmed & high_event_price.gt(prior_high)
    lower_high = high_confirmed & high_event_price.lt(prior_high)
    higher_low = low_confirmed & low_event_price.gt(prior_low)
    lower_low = low_confirmed & low_event_price.lt(prior_low)

    high_class = pd.Series(
        np.select([higher_high, lower_high], [1.0, -1.0], default=np.nan),
        index=index,
    )
    low_class = pd.Series(
        np.select([higher_low, lower_low], [1.0, -1.0], default=np.nan),
        index=index,
    )
    last_high_class = high_class.ffill()
    last_low_class = low_class.ffill()
    swing_sequence_bias = pd.Series(
        np.select(
            [
                (last_high_class == 1.0) & (last_low_class == 1.0),
                (last_high_class == -1.0) & (last_low_class == -1.0),
            ],
            [1.0, -1.0],
            default=0.0,
        ),
        index=index,
    )

    active_swing_high = high_state["last_price"].shift(1)
    active_swing_low = low_state["last_price"].shift(1)
    prev_close = close.shift(1)
    bull_break_level = active_swing_high * (1.0 + breakout_buffer_pct)
    bear_break_level = active_swing_low * (1.0 - breakout_buffer_pct)
    bullish_break = (
        active_swing_high.notna()
        & close.gt(bull_break_level)
        & prev_close.le(bull_break_level)
    )
    bearish_break = (
        active_swing_low.notna()
        & close.lt(bear_break_level)
        & prev_close.ge(bear_break_level)
    )

    break_direction = pd.Series(
        np.select([bullish_break, bearish_break], [1.0, -1.0], default=np.nan),
        index=index,
    )
    prior_state = break_direction.ffill().shift(1).fillna(0.0)
    structure_state = break_direction.ffill().fillna(0.0)

    bullish_bos = bullish_break & prior_state.ge(0.0)
    bearish_bos = bearish_break & prior_state.le(0.0)
    bullish_choch = bullish_break & prior_state.lt(0.0)
    bearish_choch = bearish_break & prior_state.gt(0.0)
    transition = pd.Series(
        np.select(
            [bullish_bos, bullish_choch, bearish_bos, bearish_choch],
            [2.0, 1.0, -2.0, -1.0],
            default=0.0,
        ),
        index=index,
    )

    return {
        f"{prefix}_ms_higher_high_{s}": higher_high.fillna(False),
        f"{prefix}_ms_lower_high_{s}": lower_high.fillna(False),
        f"{prefix}_ms_higher_low_{s}": higher_low.fillna(False),
        f"{prefix}_ms_lower_low_{s}": lower_low.fillna(False),
        f"{prefix}_ms_high_class_{s}": high_class.fillna(0.0),
        f"{prefix}_ms_low_class_{s}": low_class.fillna(0.0),
        f"{prefix}_ms_last_high_class_{s}": last_high_class.fillna(0.0),
        f"{prefix}_ms_last_low_class_{s}": last_low_class.fillna(0.0),
        f"{prefix}_ms_swing_sequence_bias_{s}": swing_sequence_bias,
        f"{prefix}_ms_active_swing_high_{s}": active_swing_high,
        f"{prefix}_ms_active_swing_low_{s}": active_swing_low,
        f"{prefix}_ms_bullish_break_{s}": bullish_break.fillna(False),
        f"{prefix}_ms_bearish_break_{s}": bearish_break.fillna(False),
        f"{prefix}_ms_bullish_bos_{s}": bullish_bos.fillna(False),
        f"{prefix}_ms_bearish_bos_{s}": bearish_bos.fillna(False),
        f"{prefix}_ms_bullish_choch_{s}": bullish_choch.fillna(False),
        f"{prefix}_ms_bearish_choch_{s}": bearish_choch.fillna(False),
        f"{prefix}_ms_state_{s}": structure_state,
        f"{prefix}_ms_prior_state_{s}": prior_state,
        f"{prefix}_ms_transition_{s}": transition,
    }


def _last_two_events(event_price: Series, event_index: Series, index: pd.Index) -> dict[str, Series]:
    price_events = event_price.dropna()
    index_events = event_index.dropna()
    return {
        "last_price": event_price.ffill(),
        "last_index": event_index.ffill(),
        "prev_price": price_events.shift(1).reindex(index).ffill(),
        "prev_index": index_events.shift(1).reindex(index).ffill(),
    }


def _line_from_events(
    bar_index: Series,
    last_price: Series,
    last_index: Series,
    prev_price: Series,
    prev_index: Series,
) -> tuple[Series, Series]:
    span = last_index - prev_index
    valid = last_price.notna() & prev_price.notna() & span.gt(0)
    slope = ((last_price - prev_price) / span).where(valid)
    line = (last_price + slope * (bar_index - last_index)).where(valid)
    return slope, line


def _active_alias_columns(columns: dict[str, Series], prefix: str, strength: int) -> dict[str, Series]:
    aliases = (
        "pivot_high_confirmed",
        "pivot_low_confirmed",
        "pivot_high",
        "pivot_low",
        "pivot_high_prominence",
        "pivot_low_prominence",
        "last_pivot_high",
        "last_pivot_low",
        "prev_pivot_high",
        "prev_pivot_low",
        "pivot_high_age",
        "pivot_low_age",
        "resistance_line",
        "support_line",
        "channel_valid",
        "resistance_slope_pct",
        "support_slope_pct",
        "channel_mid",
        "channel_width_pct",
        "channel_compression",
        "trend_bias",
        "resistance_zone_upper",
        "resistance_zone_lower",
        "support_zone_upper",
        "support_zone_lower",
        "long_target_low",
        "long_target_high",
        "short_target_low",
        "short_target_high",
        "long_target_distance_pct",
        "short_target_distance_pct",
        "long_target_actionable",
        "short_target_actionable",
        "ms_higher_high",
        "ms_lower_high",
        "ms_higher_low",
        "ms_lower_low",
        "ms_high_class",
        "ms_low_class",
        "ms_last_high_class",
        "ms_last_low_class",
        "ms_swing_sequence_bias",
        "ms_active_swing_high",
        "ms_active_swing_low",
        "ms_bullish_break",
        "ms_bearish_break",
        "ms_bullish_bos",
        "ms_bearish_bos",
        "ms_bullish_choch",
        "ms_bearish_choch",
        "ms_state",
        "ms_prior_state",
        "ms_transition",
    )
    active: dict[str, Series] = {}
    for alias in aliases:
        source = f"{prefix}_{alias}_{strength}"
        if source in columns:
            active[f"{prefix}_{alias}"] = columns[source]
    return active


def _atr(frame: DataFrame, period: int) -> Series:
    high = _num(frame["high"])
    low = _num(frame["low"])
    close = _num(frame["close"])
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(int(period), min_periods=int(period)).mean()


def _resolve_config(config: PivotStructureConfig | None, **overrides: object) -> PivotStructureConfig:
    cfg = config or PivotStructureConfig()
    values = {
        "strength": cfg.strength,
        "strengths": cfg.strengths,
        "atr_period": cfg.atr_period,
        "min_prominence_atr": cfg.min_prominence_atr,
        "zone_atr_mult": cfg.zone_atr_mult,
        "zone_pct": cfg.zone_pct,
        "min_channel_width_pct": cfg.min_channel_width_pct,
        "min_target_distance_pct": cfg.min_target_distance_pct,
        "breakout_buffer_pct": cfg.breakout_buffer_pct,
        "prefix": cfg.prefix,
    }
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return PivotStructureConfig(**values)


def _validate_config(cfg: PivotStructureConfig) -> None:
    if cfg.strength < 1:
        raise ValueError("strength must be at least 1")
    if cfg.atr_period < 2:
        raise ValueError("atr_period must be at least 2")
    if cfg.min_prominence_atr < 0:
        raise ValueError("min_prominence_atr must be non-negative")
    if cfg.zone_atr_mult < 0:
        raise ValueError("zone_atr_mult must be non-negative")
    if cfg.zone_pct < 0:
        raise ValueError("zone_pct must be non-negative")
    if cfg.min_channel_width_pct < 0:
        raise ValueError("min_channel_width_pct must be non-negative")
    if cfg.min_target_distance_pct < 0:
        raise ValueError("min_target_distance_pct must be non-negative")
    if cfg.breakout_buffer_pct < 0:
        raise ValueError("breakout_buffer_pct must be non-negative")
    if not cfg.prefix:
        raise ValueError("prefix must not be empty")
    strengths = _sorted_strengths(cfg)
    if not strengths:
        raise ValueError("at least one strength is required")
    if any(value < 1 for value in strengths):
        raise ValueError("all strengths must be at least 1")


def _validate_dataframe(dataframe: DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"DataFrame is missing OHLCV columns: {missing}")


def _sorted_strengths(cfg: PivotStructureConfig) -> list[int]:
    values = {int(value) for value in cfg.strengths}
    values.add(int(cfg.strength))
    return sorted(values)


def _num(series: Series) -> Series:
    return pd.to_numeric(series, errors="coerce")

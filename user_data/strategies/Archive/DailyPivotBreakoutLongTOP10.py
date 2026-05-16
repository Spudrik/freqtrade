from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd
from pandas import DataFrame, Series

from freqtrade.persistence import Trade
from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IStrategy,
)

from entry_sieve_tools import apply_explicit_hyperopt_surface, entry_sieve_minimal_roi, entry_sieve_stoploss



# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------
# This research strategy keeps the pivot-structure and OHLCV-volume indicators
# in one file so the entry experiment is portable as a single strategy module.
# ---------------------------------------------------------------------------


from collections.abc import Sequence

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


class PivotStructureConfig:
    """Settings for confirmed pivot, trendline, and target-range columns."""

    def __init__(
        self,
        strength: int = 5,
        strengths: Sequence[int] = (3, 5, 8, 13),
        atr_period: int = 14,
        min_prominence_atr: float = 0.35,
        zone_atr_mult: float = 0.35,
        zone_pct: float = 0.003,
        min_channel_width_pct: float = 0.003,
        min_target_distance_pct: float = 0.002,
        breakout_buffer_pct: float = 0.001,
        prefix: str = "pa",
    ) -> None:
        self.strength = strength
        self.strengths = strengths
        self.atr_period = atr_period
        self.min_prominence_atr = min_prominence_atr
        self.zone_atr_mult = zone_atr_mult
        self.zone_pct = zone_pct
        self.min_channel_width_pct = min_channel_width_pct
        self.min_target_distance_pct = min_target_distance_pct
        self.breakout_buffer_pct = breakout_buffer_pct
        self.prefix = prefix

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

    cfg = _pivot_resolve_config(
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
    _pivot_validate_config(cfg)
    _pivot_validate_dataframe(dataframe)

    high = _pivot_num(dataframe["high"])
    low = _pivot_num(dataframe["low"])
    close = _pivot_num(dataframe["close"]).replace(0, np.nan)
    atr = _pivot_atr(dataframe, cfg.atr_period)
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
        f"{p}_pivot_atr": atr,
        f"{p}_zone_width": zone_width,
        f"{p}_bar_index": bar_index,
    }

    strengths_list = _pivot_sorted_strengths(cfg)
    for pivot_strength in strengths_list:
        columns = _pivot_pivot_columns(
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
    new_cols.update(_pivot_active_alias_columns(new_cols, p, selected))

    existing = [col for col in dataframe.columns if str(col).startswith(f"{p}_")]
    base = dataframe.drop(columns=existing).copy() if existing else dataframe.copy()
    features = pd.DataFrame(new_cols, index=dataframe.index)
    return pd.concat([base, features], axis=1)


def _pivot_pivot_columns(
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

    high_state = _pivot_last_two_events(high_event_price, high_event_index, index)
    low_state = _pivot_last_two_events(low_event_price, low_event_index, index)

    resistance_slope, resistance_line = _pivot_line_from_events(
        bar_index,
        high_state["last_price"],
        high_state["last_index"],
        high_state["prev_price"],
        high_state["prev_index"],
    )
    support_slope, support_line = _pivot_line_from_events(
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

    market_structure = _pivot_market_structure_columns(
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


def _pivot_market_structure_columns(
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


def _pivot_last_two_events(event_price: Series, event_index: Series, index: pd.Index) -> dict[str, Series]:
    price_events = event_price.dropna()
    index_events = event_index.dropna()
    return {
        "last_price": event_price.ffill(),
        "last_index": event_index.ffill(),
        "prev_price": price_events.shift(1).reindex(index).ffill(),
        "prev_index": index_events.shift(1).reindex(index).ffill(),
    }


def _pivot_line_from_events(
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


def _pivot_active_alias_columns(columns: dict[str, Series], prefix: str, strength: int) -> dict[str, Series]:
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


def _pivot_atr(frame: DataFrame, period: int) -> Series:
    high = _pivot_num(frame["high"])
    low = _pivot_num(frame["low"])
    close = _pivot_num(frame["close"])
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


def _pivot_resolve_config(config: PivotStructureConfig | None, **overrides: object) -> PivotStructureConfig:
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


def _pivot_validate_config(cfg: PivotStructureConfig) -> None:
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
    strengths = _pivot_sorted_strengths(cfg)
    if not strengths:
        raise ValueError("at least one strength is required")
    if any(value < 1 for value in strengths):
        raise ValueError("all strengths must be at least 1")


def _pivot_validate_dataframe(dataframe: DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"DataFrame is missing OHLCV columns: {missing}")


def _pivot_sorted_strengths(cfg: PivotStructureConfig) -> list[int]:
    values = {int(value) for value in cfg.strengths}
    values.add(int(cfg.strength))
    return sorted(values)


def _pivot_num(series: Series) -> Series:
    return pd.to_numeric(series, errors="coerce")




import numpy as np
import pandas as pd
from pandas import DataFrame, Series


class ComplexVolumeConfig:
    """Settings for the standalone complex volume indicator suite."""

    def __init__(
        self,
        short_window: int = 20,
        medium_window: int = 48,
        long_window: int = 96,
        divergence_window: int = 48,
        sweep_window: int = 24,
        vwap_window: int = 48,
        anchor_volume_zscore: float = 2.50,
        vwap_band_mult: float = 1.50,
        dry_rvol: float = 0.60,
        expansion_rvol: float = 1.35,
        climax_rvol: float = 2.25,
        absorption_zscore: float = 1.00,
        low_result_atr: float = 0.45,
        wide_result_atr: float = 1.35,
        trigger_delta_zscore: float = 0.25,
        trigger_score_min: int = 3,
        prefix: str = "vol",
    ) -> None:
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.divergence_window = divergence_window
        self.sweep_window = sweep_window
        self.vwap_window = vwap_window
        self.anchor_volume_zscore = anchor_volume_zscore
        self.vwap_band_mult = vwap_band_mult
        self.dry_rvol = dry_rvol
        self.expansion_rvol = expansion_rvol
        self.climax_rvol = climax_rvol
        self.absorption_zscore = absorption_zscore
        self.low_result_atr = low_result_atr
        self.wide_result_atr = wide_result_atr
        self.trigger_delta_zscore = trigger_delta_zscore
        self.trigger_score_min = trigger_score_min
        self.prefix = prefix

def add_complex_volume_indicators(
    dataframe: DataFrame,
    config: ComplexVolumeConfig | None = None,
    *,
    short_window: int | None = None,
    medium_window: int | None = None,
    long_window: int | None = None,
    divergence_window: int | None = None,
    sweep_window: int | None = None,
    vwap_window: int | None = None,
    anchor_volume_zscore: float | None = None,
    vwap_band_mult: float | None = None,
    dry_rvol: float | None = None,
    expansion_rvol: float | None = None,
    climax_rvol: float | None = None,
    absorption_zscore: float | None = None,
    low_result_atr: float | None = None,
    wide_result_atr: float | None = None,
    trigger_delta_zscore: float | None = None,
    trigger_score_min: int | None = None,
    prefix: str | None = None,
) -> DataFrame:
    """
    Append a complex OHLCV-only volume suite to a strategy DataFrame.

    The output is designed for Freqtrade strategies that want richer volume
    behavior without orderbook or tick data. Delta/CVD columns are therefore a
    proxy: they infer pressure from candle body, close location, and volume.

    Distilled action columns:
    - ``*_enter_long`` / ``*_enter_short``
    - ``*_exit_long`` / ``*_exit_short``
    - ``*_signal``: ``1.0`` long, ``-1.0`` short, ``0.0`` neutral
    - ``*_signal_strength`` plus side-specific trigger scores

    Example:
        dataframe = add_complex_volume_indicators(dataframe)
        dataframe.loc[dataframe["vol_enter_long"], "enter_long"] = 1
    """

    cfg = _vol_resolve_config(
        config,
        short_window=short_window,
        medium_window=medium_window,
        long_window=long_window,
        divergence_window=divergence_window,
        sweep_window=sweep_window,
        vwap_window=vwap_window,
        anchor_volume_zscore=anchor_volume_zscore,
        vwap_band_mult=vwap_band_mult,
        dry_rvol=dry_rvol,
        expansion_rvol=expansion_rvol,
        climax_rvol=climax_rvol,
        absorption_zscore=absorption_zscore,
        low_result_atr=low_result_atr,
        wide_result_atr=wide_result_atr,
        trigger_delta_zscore=trigger_delta_zscore,
        trigger_score_min=trigger_score_min,
        prefix=prefix,
    )
    _vol_validate_config(cfg)
    _vol_validate_dataframe(dataframe)

    frame = dataframe.copy()
    frame = _vol_add_base_volume_columns(frame, cfg)
    frame = _vol_add_cvd_columns(frame, cfg)
    frame = _vol_add_volume_regime_columns(frame, cfg)
    frame = _vol_add_effort_result_columns(frame, cfg)
    frame = _vol_add_liquidity_sweep_columns(frame, cfg)
    frame = _vol_add_vwap_columns(frame, cfg)
    frame = _vol_add_action_columns(frame, cfg)
    return frame


def _vol_add_base_volume_columns(frame: DataFrame, cfg: ComplexVolumeConfig) -> DataFrame:
    p = cfg.prefix
    open_ = _vol_num(frame, "open")
    high = _vol_num(frame, "high")
    low = _vol_num(frame, "low")
    close = _vol_num(frame, "close")
    volume = _vol_num(frame, "volume").clip(lower=0.0).fillna(0.0)
    prev_close = close.shift(1)

    candle_range = (high - low).clip(lower=0.0)
    true_range = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(span=cfg.medium_window, min_periods=1, adjust=False).mean()

    body = close - open_
    hlc3 = (high + low + close) / 3.0
    close_location = ((_vol_safe_div(close - low, candle_range) * 2.0) - 1.0).clip(
        lower=-1.0,
        upper=1.0,
    )
    body_pressure = _vol_safe_div(body, candle_range).clip(lower=-1.0, upper=1.0)

    # With only OHLCV, "delta" must be inferred. Combining body direction with
    # close location avoids treating every green candle with a poor close as
    # strong buying, and vice versa.
    delta_pressure = ((close_location.fillna(0.0) + body_pressure.fillna(0.0)) / 2.0).clip(
        lower=-1.0,
        upper=1.0,
    )

    frame[f"{p}_hlc3"] = hlc3
    frame[f"{p}_true_range"] = true_range
    frame[f"{p}_atr"] = atr
    frame[f"{p}_range_atr"] = _vol_safe_div(true_range, atr)
    frame[f"{p}_body_atr"] = _vol_safe_div(body.abs(), atr)
    frame[f"{p}_close_location"] = close_location
    frame[f"{p}_delta_pressure"] = delta_pressure
    return frame


def _vol_add_cvd_columns(frame: DataFrame, cfg: ComplexVolumeConfig) -> DataFrame:
    p = cfg.prefix
    close = _vol_num(frame, "close")
    volume = _vol_num(frame, "volume").clip(lower=0.0).fillna(0.0)
    delta_pressure = _vol_num(frame, f"{p}_delta_pressure").fillna(0.0)
    delta = volume * delta_pressure
    cvd = delta.cumsum()
    fast = cvd.ewm(span=cfg.short_window, min_periods=1, adjust=False).mean()
    slow = cvd.ewm(span=cfg.medium_window, min_periods=1, adjust=False).mean()
    delta_zscore = _vol_zscore(delta, cfg.long_window)

    prior_price_low = close.rolling(cfg.divergence_window, min_periods=2).min().shift(1)
    prior_price_high = close.rolling(cfg.divergence_window, min_periods=2).max().shift(1)
    prior_cvd_low = cvd.rolling(cfg.divergence_window, min_periods=2).min().shift(1)
    prior_cvd_high = cvd.rolling(cfg.divergence_window, min_periods=2).max().shift(1)

    bull_pressure = delta_zscore >= cfg.trigger_delta_zscore
    bear_pressure = delta_zscore <= -cfg.trigger_delta_zscore

    frame[f"{p}_delta"] = delta
    frame[f"{p}_delta_zscore"] = delta_zscore
    frame[f"{p}_cvd"] = cvd
    frame[f"{p}_cvd_ema_fast"] = fast
    frame[f"{p}_cvd_ema_slow"] = slow
    frame[f"{p}_cvd_trend"] = fast - slow
    frame[f"{p}_cvd_bull_divergence"] = close.lt(prior_price_low) & cvd.gt(prior_cvd_low)
    frame[f"{p}_cvd_bear_divergence"] = close.gt(prior_price_high) & cvd.lt(prior_cvd_high)
    frame[f"{p}_cvd_trend_confirm_long"] = fast.gt(slow) & bull_pressure
    frame[f"{p}_cvd_trend_confirm_short"] = fast.lt(slow) & bear_pressure
    return frame


def _vol_add_volume_regime_columns(frame: DataFrame, cfg: ComplexVolumeConfig) -> DataFrame:
    p = cfg.prefix
    open_ = _vol_num(frame, "open")
    close = _vol_num(frame, "close")
    volume = _vol_num(frame, "volume").clip(lower=0.0).fillna(0.0)
    volume_mean = volume.rolling(cfg.long_window, min_periods=1).mean()
    rvol = _vol_safe_div(volume, volume_mean)
    volume_zscore = _vol_zscore(volume, cfg.long_window)
    close_location = _vol_num(frame, f"{p}_close_location")

    dry = rvol <= cfg.dry_rvol
    expansion = rvol >= cfg.expansion_rvol
    climax = rvol >= cfg.climax_rvol
    capitulation = climax & close.lt(open_) & close_location.lt(-0.35)
    accumulation = climax & close.gt(open_) & close_location.gt(0.35)

    regime = np.select(
        [capitulation, dry, climax, expansion],
        [-2.0, -1.0, 2.0, 1.0],
        default=0.0,
    )

    frame[f"{p}_volume_mean"] = volume_mean
    frame[f"{p}_rvol"] = rvol
    frame[f"{p}_volume_zscore"] = volume_zscore
    frame[f"{p}_regime_code"] = regime
    frame[f"{p}_regime_dry"] = dry
    frame[f"{p}_regime_expansion"] = expansion
    frame[f"{p}_regime_climax"] = climax
    frame[f"{p}_regime_capitulation"] = capitulation
    frame[f"{p}_regime_accumulation"] = accumulation
    return frame


def _vol_add_effort_result_columns(frame: DataFrame, cfg: ComplexVolumeConfig) -> DataFrame:
    p = cfg.prefix
    high = _vol_num(frame, "high")
    low = _vol_num(frame, "low")
    close = _vol_num(frame, "close")
    prior_high = high.rolling(cfg.sweep_window, min_periods=2).max().shift(1)
    prior_low = low.rolling(cfg.sweep_window, min_periods=2).min().shift(1)
    volume_zscore = _vol_num(frame, f"{p}_volume_zscore")
    rvol = _vol_num(frame, f"{p}_rvol")
    range_atr = _vol_num(frame, f"{p}_range_atr")
    body_atr = _vol_num(frame, f"{p}_body_atr")
    close_location = _vol_num(frame, f"{p}_close_location")
    delta_zscore = _vol_num(frame, f"{p}_delta_zscore")

    high_effort = (volume_zscore >= cfg.absorption_zscore) | (rvol >= cfg.expansion_rvol)
    low_result = body_atr <= cfg.low_result_atr
    wide_result = range_atr >= cfg.wide_result_atr

    # Effort-vs-result separates "lots of volume went nowhere" from "lots of
    # volume moved price." That distinction is useful for absorption and
    # exhaustion filters.
    bull_absorption = high_effort & low_result & close_location.gt(0.15) & delta_zscore.gt(-0.25)
    bear_absorption = high_effort & low_result & close_location.lt(-0.15) & delta_zscore.lt(0.25)
    spring = low.lt(prior_low) & close.gt(prior_low) & high_effort & close_location.gt(-0.25)
    upthrust = high.gt(prior_high) & close.lt(prior_high) & high_effort & close_location.lt(0.25)
    exhaustion_up = _vol_bool(frame, f"{p}_regime_climax") & wide_result & close_location.gt(0.50)
    exhaustion_down = _vol_bool(frame, f"{p}_regime_climax") & wide_result & close_location.lt(-0.50)

    frame[f"{p}_effort_high"] = high_effort
    frame[f"{p}_result_low"] = low_result
    frame[f"{p}_result_wide"] = wide_result
    frame[f"{p}_evr_bull_absorption"] = bull_absorption
    frame[f"{p}_evr_bear_absorption"] = bear_absorption
    frame[f"{p}_evr_spring"] = spring
    frame[f"{p}_evr_upthrust"] = upthrust
    frame[f"{p}_evr_exhaustion_up"] = exhaustion_up
    frame[f"{p}_evr_exhaustion_down"] = exhaustion_down
    return frame


def _vol_add_liquidity_sweep_columns(frame: DataFrame, cfg: ComplexVolumeConfig) -> DataFrame:
    p = cfg.prefix
    high = _vol_num(frame, "high")
    low = _vol_num(frame, "low")
    close = _vol_num(frame, "close")
    delta_zscore = _vol_num(frame, f"{p}_delta_zscore")
    close_location = _vol_num(frame, f"{p}_close_location")
    prior_high = high.rolling(cfg.sweep_window, min_periods=2).max().shift(1)
    prior_low = low.rolling(cfg.sweep_window, min_periods=2).min().shift(1)
    volume_ok = _vol_bool(frame, f"{p}_regime_expansion") | _vol_bool(frame, f"{p}_regime_climax")

    sweep_low = low.lt(prior_low)
    sweep_high = high.gt(prior_high)
    low_reclaim = sweep_low & close.gt(prior_low) & close_location.gt(-0.20)
    high_reject = sweep_high & close.lt(prior_high) & close_location.lt(0.20)
    stoprun_long = low_reclaim & volume_ok & delta_zscore.gt(-0.25)
    stoprun_short = high_reject & volume_ok & delta_zscore.lt(0.25)

    breakout_long = close.gt(prior_high) & volume_ok & delta_zscore.gt(cfg.trigger_delta_zscore)
    breakout_short = close.lt(prior_low) & volume_ok & delta_zscore.lt(-cfg.trigger_delta_zscore)

    frame[f"{p}_prior_sweep_high"] = prior_high
    frame[f"{p}_prior_sweep_low"] = prior_low
    frame[f"{p}_liq_sweep_low"] = sweep_low
    frame[f"{p}_liq_sweep_high"] = sweep_high
    frame[f"{p}_liq_sweep_low_reclaim_long"] = low_reclaim & volume_ok
    frame[f"{p}_liq_sweep_high_reject_short"] = high_reject & volume_ok
    frame[f"{p}_liq_stoprun_long"] = stoprun_long
    frame[f"{p}_liq_stoprun_short"] = stoprun_short
    frame[f"{p}_vol_breakout_confirm_long"] = breakout_long
    frame[f"{p}_vol_breakout_confirm_short"] = breakout_short
    frame[f"{p}_vol_fakeout_risk_long"] = high_reject & volume_ok
    frame[f"{p}_vol_fakeout_risk_short"] = low_reclaim & volume_ok
    return frame


def _vol_add_vwap_columns(frame: DataFrame, cfg: ComplexVolumeConfig) -> DataFrame:
    p = cfg.prefix
    close = _vol_num(frame, "close")
    high = _vol_num(frame, "high")
    low = _vol_num(frame, "low")
    volume = _vol_num(frame, "volume").clip(lower=0.0).fillna(0.0)
    tp = _vol_num(frame, f"{p}_hlc3")
    pv = tp * volume
    pv2 = tp * tp * volume

    rolling_volume = volume.rolling(cfg.vwap_window, min_periods=1).sum()
    rolling_vwap = _vol_safe_div(pv.rolling(cfg.vwap_window, min_periods=1).sum(), rolling_volume)
    rolling_second = _vol_safe_div(pv2.rolling(cfg.vwap_window, min_periods=1).sum(), rolling_volume)
    rolling_std = np.sqrt((rolling_second - rolling_vwap * rolling_vwap).clip(lower=0.0))

    anchor_event = (
        _vol_num(frame, f"{p}_volume_zscore").ge(cfg.anchor_volume_zscore)
        | _vol_bool(frame, f"{p}_liq_stoprun_long")
        | _vol_bool(frame, f"{p}_liq_stoprun_short")
    )
    if len(anchor_event) > 0:
        anchor_event = anchor_event.copy()
        anchor_event.iloc[0] = True
    anchor_id = anchor_event.astype("int64").cumsum()
    anchored_volume = volume.groupby(anchor_id).cumsum()
    anchored_vwap = _vol_safe_div(pv.groupby(anchor_id).cumsum(), anchored_volume)
    anchored_second = _vol_safe_div(pv2.groupby(anchor_id).cumsum(), anchored_volume)
    anchored_std = np.sqrt((anchored_second - anchored_vwap * anchored_vwap).clip(lower=0.0))
    upper = anchored_vwap + (anchored_std * cfg.vwap_band_mult)
    lower = anchored_vwap - (anchored_std * cfg.vwap_band_mult)

    prev_close = close.shift(1)
    prev_avwap = anchored_vwap.shift(1)
    cross_above = close.gt(anchored_vwap) & prev_close.le(prev_avwap)
    cross_below = close.lt(anchored_vwap) & prev_close.ge(prev_avwap)
    bull_pressure = _vol_num(frame, f"{p}_delta_zscore").ge(cfg.trigger_delta_zscore)
    bear_pressure = _vol_num(frame, f"{p}_delta_zscore").le(-cfg.trigger_delta_zscore)

    # The anchored VWAP resets at unusually important volume events. This gives
    # a lightweight "market memory" line without hand-picking sessions.
    frame[f"{p}_rvwap"] = rolling_vwap
    frame[f"{p}_rvwap_upper"] = rolling_vwap + (rolling_std * cfg.vwap_band_mult)
    frame[f"{p}_rvwap_lower"] = rolling_vwap - (rolling_std * cfg.vwap_band_mult)
    frame[f"{p}_anchor_event"] = anchor_event
    frame[f"{p}_anchor_id"] = anchor_id.astype("float64")
    frame[f"{p}_avwap"] = anchored_vwap
    frame[f"{p}_avwap_upper"] = upper
    frame[f"{p}_avwap_lower"] = lower
    frame[f"{p}_avwap_reclaim_long"] = cross_above & bull_pressure
    frame[f"{p}_avwap_reject_short"] = cross_below & bear_pressure
    frame[f"{p}_avwap_upper_extension"] = high.ge(upper)
    frame[f"{p}_avwap_lower_extension"] = low.le(lower)
    frame[f"{p}_avwap_mean_reversion_long"] = low.le(lower) & close.gt(lower) & bull_pressure
    frame[f"{p}_avwap_mean_reversion_short"] = high.ge(upper) & close.lt(upper) & bear_pressure
    return frame


def _vol_add_action_columns(frame: DataFrame, cfg: ComplexVolumeConfig) -> DataFrame:
    p = cfg.prefix
    long_parts = {
        "cvd_bull_divergence": 2,
        "cvd_trend_confirm_long": 1,
        "evr_bull_absorption": 1,
        "evr_spring": 3,
        "liq_sweep_low_reclaim_long": 3,
        "liq_stoprun_long": 3,
        "vol_breakout_confirm_long": 3,
        "avwap_reclaim_long": 2,
        "avwap_mean_reversion_long": 2,
    }
    short_parts = {
        "cvd_bear_divergence": 2,
        "cvd_trend_confirm_short": 1,
        "evr_bear_absorption": 1,
        "evr_upthrust": 3,
        "liq_sweep_high_reject_short": 3,
        "liq_stoprun_short": 3,
        "vol_breakout_confirm_short": 3,
        "avwap_reject_short": 2,
        "avwap_mean_reversion_short": 2,
    }

    long_score = _vol_weighted_bool_score(frame, p, long_parts)
    short_score = _vol_weighted_bool_score(frame, p, short_parts)
    enter_long = long_score.ge(cfg.trigger_score_min) & long_score.gt(short_score)
    enter_short = short_score.ge(cfg.trigger_score_min) & short_score.gt(long_score)
    exit_long = (
        short_score.ge(cfg.trigger_score_min)
        | _vol_bool(frame, f"{p}_evr_exhaustion_up")
        | _vol_bool(frame, f"{p}_vol_fakeout_risk_long")
    )
    exit_short = (
        long_score.ge(cfg.trigger_score_min)
        | _vol_bool(frame, f"{p}_evr_exhaustion_down")
        | _vol_bool(frame, f"{p}_vol_fakeout_risk_short")
    )

    frame[f"{p}_long_trigger_score"] = long_score.astype("float64")
    frame[f"{p}_short_trigger_score"] = short_score.astype("float64")
    frame[f"{p}_enter_long"] = enter_long
    frame[f"{p}_enter_short"] = enter_short
    frame[f"{p}_exit_long"] = exit_long
    frame[f"{p}_exit_short"] = exit_short
    frame[f"{p}_signal"] = np.select([enter_long, enter_short], [1.0, -1.0], default=0.0)
    frame[f"{p}_signal_strength"] = (long_score - short_score).abs().astype("float64")
    return frame


def _vol_resolve_config(config: ComplexVolumeConfig | None, **overrides: object) -> ComplexVolumeConfig:
    cfg = config or ComplexVolumeConfig()
    values = {
        "short_window": cfg.short_window,
        "medium_window": cfg.medium_window,
        "long_window": cfg.long_window,
        "divergence_window": cfg.divergence_window,
        "sweep_window": cfg.sweep_window,
        "vwap_window": cfg.vwap_window,
        "anchor_volume_zscore": cfg.anchor_volume_zscore,
        "vwap_band_mult": cfg.vwap_band_mult,
        "dry_rvol": cfg.dry_rvol,
        "expansion_rvol": cfg.expansion_rvol,
        "climax_rvol": cfg.climax_rvol,
        "absorption_zscore": cfg.absorption_zscore,
        "low_result_atr": cfg.low_result_atr,
        "wide_result_atr": cfg.wide_result_atr,
        "trigger_delta_zscore": cfg.trigger_delta_zscore,
        "trigger_score_min": cfg.trigger_score_min,
        "prefix": cfg.prefix,
    }
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return ComplexVolumeConfig(**values)


def _vol_validate_config(cfg: ComplexVolumeConfig) -> None:
    window_values = [
        cfg.short_window,
        cfg.medium_window,
        cfg.long_window,
        cfg.divergence_window,
        cfg.sweep_window,
        cfg.vwap_window,
    ]
    if any(value < 2 for value in window_values):
        raise ValueError("all volume indicator windows must be at least 2")
    if cfg.vwap_band_mult <= 0.0:
        raise ValueError("vwap_band_mult must be positive")
    if cfg.dry_rvol <= 0.0 or cfg.expansion_rvol <= 0.0 or cfg.climax_rvol <= 0.0:
        raise ValueError("relative volume thresholds must be positive")
    if cfg.dry_rvol >= cfg.expansion_rvol or cfg.expansion_rvol >= cfg.climax_rvol:
        raise ValueError("expected dry_rvol < expansion_rvol < climax_rvol")
    if cfg.low_result_atr <= 0.0 or cfg.wide_result_atr <= 0.0:
        raise ValueError("effort/result ATR thresholds must be positive")
    if cfg.trigger_score_min < 1:
        raise ValueError("trigger_score_min must be at least 1")
    if not cfg.prefix:
        raise ValueError("prefix must not be empty")


def _vol_validate_dataframe(dataframe: DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"DataFrame is missing OHLCV columns: {missing}")


def _vol_weighted_bool_score(frame: DataFrame, prefix: str, parts: dict[str, int]) -> Series:
    score = pd.Series(0, index=frame.index, dtype="int64")
    for name, weight in parts.items():
        score += _vol_bool(frame, f"{prefix}_{name}").astype("int64") * int(weight)
    return score


def _vol_zscore(series: Series, window: int) -> Series:
    mean = series.rolling(window, min_periods=2).mean()
    std = series.rolling(window, min_periods=2).std(ddof=0)
    return _vol_safe_div(series - mean, std)


def _vol_safe_div(numerator: Series, denominator: Series) -> Series:
    denominator = denominator.where(denominator.abs() > 0.0, np.nan)
    return numerator / denominator


def _vol_num(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _vol_bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype="bool")
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)


logger = logging.getLogger(__name__)


def tagged_parameter(param: Any, *tags: str) -> Any:
    """Attach lightweight discovery metadata for Explorer/tag-catalog tooling."""
    family_map = {
        "entry": "entries",
        "entries": "entries",
        "structure": "entries",
        "volume": "entries",
        "volume_pressure": "entries",
        "entry_enables": "entries",
        "entry_confirmation": "entries",
        "entry_structure": "entries",
        "trend_filter": "entries",
        "daily_structure": "entries",
        "hourly_execution": "entries",
        "line_quality": "entries",
        "target_space": "entries",
        "trendline_projection": "entries",
        "breakout_long": "entries",
        "exit": "exits",
        "exits": "exits",
        "exit_profile": "exits",
        "exit_peel": "exits",
        "exit_target": "exits",
        "entry_family_exit": "exits",
        "stoploss": "exits",
        "adjust_position": "adjust_position",
        "add_enables": "adjust_position",
        "add_rules": "adjust_position",
        "capital": "stake",
        "stake": "stake",
        "risk": "risk",
    }
    mode_alias = {
        "pivot_breakout_long": "entry_core",
    }
    family: str | None = None
    mode: str | None = None
    for raw_tag in tags:
        tag = str(raw_tag or "").strip()
        if not tag or ":" not in tag:
            continue
        key, value = tag.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key == "family":
            mapped = family_map.get(value.lower())
            if mapped and family is None:
                family = mapped
        elif key == "mode" and value and mode is None:
            mode = value

    if family is None:
        if mode and mode.startswith("exit_"):
            family = "exits"
        elif mode and mode.startswith("adjust_"):
            family = "adjust_position"
        elif mode and mode.startswith("stake_"):
            family = "stake"
        elif mode and mode.startswith("risk_"):
            family = "risk"
        else:
            family = "entries"

    if mode is None:
        mode_defaults = {
            "entries": "entry_core",
            "exits": "exit_core",
            "adjust_position": "adjust_core",
            "stake": "stake_core",
            "risk": "risk_core",
        }
        mode = mode_defaults.get(family, "entry_core")
    else:
        mode = mode_alias.get(mode, mode)

    setattr(param, "batch_tags", (f"family:{family}", f"mode:{mode}"))
    return param


class DailyPivotBreakoutLongTOP10(IStrategy):
    """
    Minimal long-only research strategy for confirmed pivot breakout entries.

    Purpose
    -------
    This file isolates one question: does a long close above confirmed pivot /
    swing resistance produce enough follow-through to justify a simple long
    entry? Capital scaling, leverage optimisation, shorts, adds, peels, broad
    exit profiles, and volume profile logic are intentionally excluded.

    Indicator scope
    ---------------
    - Pivot-structure and complex-volume indicator helpers are embedded in this
      file to keep the research wrapper portable as a single strategy module.
    - Volume confirmation is always evaluated through loose-to-strict HyperOpt
      thresholds rather than a separate enable switch.
    - Pair/BTC trend filters are selectable by HyperOpt so their effect can be
      compared directly against the unfiltered breakout baseline.
    - Volume profile logic is deliberately not used in v2.

    Hyperopt note
    -------------
    Freqtrade commonly caches indicator columns during HyperOpt. This strategy
    therefore optimises thresholds that act on precomputed columns, while pivot
    construction settings stay fixed constants in v1. Pivot strength is safe to
    optimise because add_pivot_structure() precomputes all declared strengths and
    entry/exit logic selects the matching suffixed columns at runtime.
    """

    INTERFACE_VERSION = 3

    can_short = False
    timeframe = "1h"
    startup_candle_count = 600
    process_only_new_candles = True

    use_exit_signal = False
    use_custom_stoploss = False
    use_custom_roi = False
    minimal_roi = entry_sieve_minimal_roi(0.02)
    stoploss = entry_sieve_stoploss(-0.02)

    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    PIVOT_STRENGTH_CHOICES = [3, 5, 8, 13]

    # Fixed v1 indicator-construction settings. Keep these out of HyperOpt until
    # the research wrapper has proven the breakout concept with stable columns.
    PIVOT_ATR_PERIOD = 14
    PIVOT_MIN_PROMINENCE_ATR = 0.35
    PIVOT_ZONE_ATR_MULT = 0.35
    PIVOT_ZONE_PCT = 0.003
    PIVOT_MIN_CHANNEL_WIDTH_PCT = 0.003
    PIVOT_MIN_TARGET_DISTANCE_PCT = 0.002
    PIVOT_INTERNAL_BREAKOUT_BUFFER_PCT = 0.001

    # Volume diagnostics are fixed-width. Entry thresholds below decide whether
    # volume behaviour acts like a loose diagnostic or a strict confirmation.
    VOLUME_SHORT_WINDOW = 20
    VOLUME_MEDIUM_WINDOW = 48
    VOLUME_LONG_WINDOW = 96

    TREND_EMA_DAY_CHOICES = [50, 100, 200]
    BTC_REFERENCE_PAIR = "BTC/USDT:USDT"

    pivot_strength = tagged_parameter(
        CategoricalParameter(PIVOT_STRENGTH_CHOICES, default=5, space="buy", optimize=True, load=True),
        "family:breakout_long",
        "family:structure",
        "batch:pivot_breakout_long",
        "domain:entry",
        "domain:price_action",
        "action:seed_entry",
        "mode:pivot_breakout_long",
        "role:pivot_strength",
    )
    max_channel_compression = tagged_parameter(
        DecimalParameter(0.40, 1.20, decimals=2, default=0.90, space="buy", optimize=True, load=True),
        "family:breakout_long",
        "family:structure",
        "batch:pivot_breakout_long",
        "domain:entry",
        "domain:price_action",
        "role:compression_threshold",
    )
    breakout_buffer_pct = tagged_parameter(
        DecimalParameter(0.001, 0.015, decimals=3, default=0.003, space="buy", optimize=True, load=True),
        "family:breakout_long",
        "family:structure",
        "batch:pivot_breakout_long",
        "domain:entry",
        "domain:price_action",
        "role:breakout_buffer",
    )
    trend_filter_mode = tagged_parameter(
        CategoricalParameter(
            ["none", "pair_trend", "btc_trend", "pair_and_btc", "pair_or_btc"],
            default="none",
            space="buy",
            optimize=True,
            load=True,
        ),
        "family:breakout_long",
        "family:trend_filter",
        "batch:pivot_breakout_long",
        "domain:entry",
        "domain:regime",
        "role:filter_mode",
    )
    trend_ema_days = tagged_parameter(
        CategoricalParameter(TREND_EMA_DAY_CHOICES, default=100, space="buy", optimize=True, load=True),
        "family:breakout_long",
        "family:trend_filter",
        "batch:pivot_breakout_long",
        "domain:entry",
        "domain:regime",
        "role:trend_period",
    )
    min_rvol = tagged_parameter(
        DecimalParameter(0.1, 2.5, decimals=1, default=0.8, space="buy", optimize=True, load=True),
        "family:breakout_long",
        "family:volume",
        "batch:volume_confirmation",
        "domain:entry",
        "domain:volume",
        "role:threshold",
    )
    min_delta_zscore = tagged_parameter(
        DecimalParameter(-2.0, 1.5, decimals=1, default=-1.0, space="buy", optimize=True, load=True),
        "family:breakout_long",
        "family:volume",
        "batch:volume_confirmation",
        "domain:entry",
        "domain:volume",
        "role:threshold",
    )
    min_close_location = tagged_parameter(
        DecimalParameter(-1.0, 0.8, decimals=1, default=-1.0, space="buy", optimize=True, load=True),
        "family:breakout_long",
        "family:volume",
        "batch:volume_confirmation",
        "domain:entry",
        "domain:volume",
        "role:threshold",
    )
    stop_buffer_pct = tagged_parameter(
        DecimalParameter(0.003, 0.030, decimals=3, default=0.010, space="sell", optimize=True, load=True),
        "family:exit",
        "batch:pivot_breakout_long_exit",
        "domain:exit",
        "domain:price_action",
        "action:structural_exit",
        "mode:exit_structural_breakout",
        "role:stop_buffer",
    )

    plot_config = {
        "main_plot": {
            "pa_resistance_line": {"color": "#ff9900"},
            "pa_resistance_zone_upper": {"color": "#ffcc66"},
            "pa_resistance_zone_lower": {"color": "#cc6600"},
            "pivot_breakout_structural_stop": {"color": "#ff3333"},
        },
        "subplots": {
            "Pivot Structure": {
                "pa_channel_compression": {"color": "#33aaff"},
                "pa_ms_state": {"color": "#aa66ff"},
            },
            "Volume Diagnostics": {
                "vol_rvol": {"color": "#66cc66"},
                "vol_delta_zscore": {"color": "#cc66cc"},
                "pivot_breakout_volume_ok": {"color": "#00aa00"},
            },
            "Trend Filter": {
                "pivot_breakout_market_filter_ok": {"color": "#ffaa00"},
                "pair_trend_up": {"color": "#66aaff"},
                "btc_trend_up": {"color": "#ff66aa"},
            },
        },
    }

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        dataframe = add_pivot_structure(
            dataframe,
            PivotStructureConfig(
                strength=int(self.PIVOT_STRENGTH_CHOICES[0]),
                strengths=tuple(int(value) for value in self.PIVOT_STRENGTH_CHOICES),
                atr_period=int(self.PIVOT_ATR_PERIOD),
                min_prominence_atr=float(self.PIVOT_MIN_PROMINENCE_ATR),
                zone_atr_mult=float(self.PIVOT_ZONE_ATR_MULT),
                zone_pct=float(self.PIVOT_ZONE_PCT),
                min_channel_width_pct=float(self.PIVOT_MIN_CHANNEL_WIDTH_PCT),
                min_target_distance_pct=float(self.PIVOT_MIN_TARGET_DISTANCE_PCT),
                breakout_buffer_pct=float(self.PIVOT_INTERNAL_BREAKOUT_BUFFER_PCT),
                prefix="pa",
            ),
        )
        dataframe = add_complex_volume_indicators(
            dataframe,
            ComplexVolumeConfig(
                short_window=int(self.VOLUME_SHORT_WINDOW),
                medium_window=int(self.VOLUME_MEDIUM_WINDOW),
                long_window=int(self.VOLUME_LONG_WINDOW),
                divergence_window=int(self.VOLUME_MEDIUM_WINDOW),
                sweep_window=24,
                vwap_window=int(self.VOLUME_MEDIUM_WINDOW),
                prefix="vol",
            ),
        )
        dataframe = self._add_trend_filter_columns(dataframe, metadata)

        # Plot aliases use the default strength for FreqUI readability. Trading
        # logic below uses the suffixed columns selected by pivot_strength.
        default_strength = int(self.PIVOT_STRENGTH_CHOICES[1])
        for base in (
            "resistance_line",
            "resistance_zone_upper",
            "resistance_zone_lower",
            "channel_compression",
            "ms_state",
        ):
            source = f"pa_{base}_{default_strength}"
            if source in dataframe.columns:
                dataframe[f"pa_{base}"] = dataframe[source]

        dataframe["pivot_breakout_long_signal"] = False
        dataframe["pivot_breakout_volume_ok"] = False
        dataframe["pivot_breakout_market_filter_ok"] = False
        dataframe["pivot_breakout_structural_stop"] = pd.NA
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""
        dataframe["enter_short"] = 0

        signal = self._pivot_breakout_long_mask(dataframe)
        dataframe.loc[signal, "enter_long"] = 1
        dataframe.loc[signal, "enter_tag"] = "pivot_breakout_long"
        dataframe["pivot_breakout_long_signal"] = signal.astype(bool)
        dataframe["pivot_breakout_volume_ok"] = self._volume_confirmation_mask(dataframe).astype(bool)
        dataframe["pivot_breakout_market_filter_ok"] = self._market_filter_mask(dataframe).astype(bool)
        dataframe["pivot_breakout_structural_stop"] = self._structural_stop_level(dataframe)
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = ""
        dataframe["exit_short"] = 0

        structural_failure = self._structural_failure_mask(dataframe)
        dataframe.loc[structural_failure, "exit_long"] = 1
        dataframe.loc[structural_failure, "exit_tag"] = "pivot_breakout_structural_fail"
        dataframe["pivot_breakout_structural_stop"] = self._structural_stop_level(dataframe)
        return dataframe

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float:
        """Research wrapper is deliberately unlevered."""
        return 1.0

    # ------------------------------------------------------------------
    # Signal helpers
    # ------------------------------------------------------------------

    def _pivot_breakout_long_mask(self, dataframe: DataFrame) -> Series:
        strength = int(self.pivot_strength.value)
        close = self._num_col(dataframe, "close")
        active_high = self._num_col(dataframe, self._pa_col("ms_active_swing_high", strength))
        res_upper = self._num_col(dataframe, self._pa_col("resistance_zone_upper", strength))
        channel_valid = self._bool_col(dataframe, self._pa_col("channel_valid", strength))
        bullish_break = self._bool_col(dataframe, self._pa_col("ms_bullish_break", strength))
        compression = self._num_col(dataframe, self._pa_col("channel_compression", strength))
        buffer = float(self.breakout_buffer_pct.value)

        return self._all_conditions(
            dataframe,
            [
                channel_valid,
                active_high.notna(),
                res_upper.notna(),
                close > res_upper * (1.0 + buffer),
                close > active_high * (1.0 + buffer),
                bullish_break,
                compression.notna() & compression.le(float(self.max_channel_compression.value)),
                self._num_col(dataframe, "volume") > 0.0,
                self._volume_confirmation_mask(dataframe),
                self._market_filter_mask(dataframe),
            ],
        )

    def _volume_confirmation_mask(self, dataframe: DataFrame) -> Series:
        return self._all_conditions(
            dataframe,
            [
                self._num_col(dataframe, "vol_rvol") >= float(self.min_rvol.value),
                self._num_col(dataframe, "vol_delta_zscore") >= float(self.min_delta_zscore.value),
                self._num_col(dataframe, "vol_close_location") >= float(self.min_close_location.value),
            ],
        )

    def _market_filter_mask(self, dataframe: DataFrame) -> Series:
        mode = str(self.trend_filter_mode.value)
        days = int(self.trend_ema_days.value)
        pair_ok = self._bool_col(dataframe, f"pair_trend_up_{days}")
        btc_ok = self._bool_col(dataframe, f"btc_trend_up_{days}")
        if mode == "pair_trend":
            return pair_ok
        if mode == "btc_trend":
            return btc_ok
        if mode == "pair_and_btc":
            return pair_ok & btc_ok
        if mode == "pair_or_btc":
            return pair_ok | btc_ok
        return self._true_mask(dataframe)

    def _add_trend_filter_columns(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        close = self._num_col(dataframe, "close")
        for days in self.TREND_EMA_DAY_CHOICES:
            hours = max(1, int(days) * 24)
            ema = close.ewm(span=hours, min_periods=1, adjust=False).mean()
            dataframe[f"pair_trend_ema_{days}"] = ema
            dataframe[f"pair_trend_up_{days}"] = (close > ema).fillna(False)

        dataframe = self._add_btc_trend_columns(dataframe, metadata)
        selected_days = int(self.trend_ema_days.value) if hasattr(self.trend_ema_days, "value") else 100
        dataframe["pair_trend_up"] = self._bool_col(dataframe, f"pair_trend_up_{selected_days}").astype(float)
        dataframe["btc_trend_up"] = self._bool_col(dataframe, f"btc_trend_up_{selected_days}").astype(float)
        return dataframe

    def _add_btc_trend_columns(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        pair = str(metadata.get("pair") or "")
        if pair == self.BTC_REFERENCE_PAIR:
            for days in self.TREND_EMA_DAY_CHOICES:
                dataframe[f"btc_trend_up_{days}"] = self._bool_col(dataframe, f"pair_trend_up_{days}")
            return dataframe

        btc_frame = None
        try:
            if self.dp:
                btc_frame = self.dp.get_pair_dataframe(pair=self.BTC_REFERENCE_PAIR, timeframe=self.timeframe)
        except Exception as exc:
            logger.debug("Could not load BTC trend dataframe: %s", exc)

        if btc_frame is None or btc_frame.empty or "date" not in dataframe.columns or "date" not in btc_frame.columns:
            for days in self.TREND_EMA_DAY_CHOICES:
                dataframe[f"btc_trend_up_{days}"] = False
            return dataframe

        btc = btc_frame[["date", "close"]].copy()
        btc["date"] = pd.to_datetime(btc["date"], utc=True)
        btc_close = pd.to_numeric(btc["close"], errors="coerce")
        for days in self.TREND_EMA_DAY_CHOICES:
            hours = max(1, int(days) * 24)
            btc_ema = btc_close.ewm(span=hours, min_periods=1, adjust=False).mean()
            btc[f"btc_trend_up_{days}"] = (btc_close > btc_ema).fillna(False)

        left = dataframe.copy()
        left["date"] = pd.to_datetime(left["date"], utc=True)
        trend_cols = [f"btc_trend_up_{days}" for days in self.TREND_EMA_DAY_CHOICES]
        merged = pd.merge_asof(
            left.sort_values("date"),
            btc[["date", *trend_cols]].sort_values("date"),
            on="date",
            direction="backward",
        ).sort_index()
        for col in trend_cols:
            dataframe[col] = merged[col].fillna(False).astype(bool)
        return dataframe

    def _structural_failure_mask(self, dataframe: DataFrame) -> Series:
        close = self._num_col(dataframe, "close")
        stop_level = self._structural_stop_level(dataframe)
        return close.lt(stop_level).fillna(False).astype(bool)

    def _structural_stop_level(self, dataframe: DataFrame) -> Series:
        strength = int(self.pivot_strength.value)
        res_lower = self._num_col(dataframe, self._pa_col("resistance_zone_lower", strength))
        return res_lower * (1.0 - float(self.stop_buffer_pct.value))

    def _pa_col(self, base: str, strength: int | None = None) -> str:
        selected = int(strength if strength is not None else self.pivot_strength.value)
        return f"pa_{base}_{selected}"

    # ------------------------------------------------------------------
    # DataFrame helpers
    # ------------------------------------------------------------------

    def _false_mask(self, dataframe: DataFrame) -> Series:
        return pd.Series(False, index=dataframe.index, dtype="bool")

    def _true_mask(self, dataframe: DataFrame) -> Series:
        return pd.Series(True, index=dataframe.index, dtype="bool")

    def _bool_col(self, dataframe: DataFrame, column: str) -> Series:
        if column not in dataframe.columns:
            return self._false_mask(dataframe)
        return pd.Series(dataframe[column], index=dataframe.index).astype("boolean").fillna(False).astype(bool)

    def _num_col(self, dataframe: DataFrame, column: str) -> Series:
        if column not in dataframe.columns:
            return pd.Series(pd.NA, index=dataframe.index, dtype="Float64")
        return pd.to_numeric(dataframe[column], errors="coerce")

    def _all_conditions(self, dataframe: DataFrame, conditions: list[Series]) -> Series:
        if not conditions:
            return self._false_mask(dataframe)
        result = self._true_mask(dataframe)
        for condition in conditions:
            result &= pd.Series(condition, index=dataframe.index).fillna(False).astype(bool)
        return result


apply_explicit_hyperopt_surface(DailyPivotBreakoutLongTOP10)

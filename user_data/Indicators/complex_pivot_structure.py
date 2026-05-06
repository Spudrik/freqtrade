from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

try:
    from .pivot_foundation import build_clean_pivot_source
except Exception:  # pragma: no cover - optional fallback for standalone notebooks
    from pivot_foundation import build_clean_pivot_source  # type: ignore[no-redef]


@dataclass(frozen=True)
class PivotStructureConfig:
    """
    Settings for confirmed pivot and structure columns.

    Pivots are only emitted on the confirmation candle. For example, strength 5
    confirms that the candle from five bars ago was a pivot only after the five
    right-side candles have closed, so the current row never uses unknown future
    candles.

    Pivot detection uses open/close body highs and lows from the shared pivot
    foundation. Wick highs/lows are still used where this indicator checks
    actual candle interaction with structural zones.

    Freqtrade/hyperopt notes:
    - All fields are strategy-facing tuning knobs.
    - Per-strength outputs let a strategy hyperopt the active strength without
      recomputing indicators.
    - Score columns are normalized to 0..1 and are evidence only. They are not
      entry/exit signals.

    Score meaning:
    - ``*_score_long`` rises when confirmed structure shows higher-high /
      higher-low sequences and bullish trend-aligned/trend-flip breakouts.
    - ``*_score_short`` mirrors that for lower-high / lower-low structure and
      bearish breakouts.
    - ``*_score_abs`` is max(long, short). It measures structural clarity rather
      than direction.
    - ``*_state`` is -1/0/1 directional lean and is not normalized.

    Noise-control knobs:
    ``min_prominence_atr``, ``min_prominence_pct``,
    ``min_pivot_spacing_bars``, ``min_pivot_distance_atr``,
    ``min_pivot_distance_pct``, and ``max_pivot_age_bars`` are intended to stop
    tiny peaks from dominating the structure.

    Local vs structural split:
    - Local pivots use ``strength`` / ``strengths`` and are intended to describe
      tactical swing points. They can be noisy by design.
    - Structural pivots use ``structural_*`` settings and are intended to mark
      larger support/resistance memory. They use stricter prominence, spacing,
      and distance filters, then expose horizontal support/resistance zones.

    Channel detection is intentionally not part of this indicator. Channels are
    built from ranked trendline evidence in the dedicated channel indicator.
    """

    strength: int = 5
    strengths: Sequence[int] = (3, 5, 8, 13)
    atr_period: int = 14
    min_prominence_atr: float = 0.35
    min_prominence_pct: float = 0.0
    min_pivot_spacing_bars: int = 1
    min_pivot_distance_atr: float = 0.0
    min_pivot_distance_pct: float = 0.0
    max_pivot_age_bars: int = 240
    breakout_buffer_pct: float = 0.001
    structure_score_window: int = 12
    structural_strength: int = 21
    structural_min_prominence_atr: float = 1.25
    structural_min_prominence_pct: float = 0.012
    structural_min_pivot_spacing_bars: int = 8
    structural_min_pivot_distance_atr: float = 1.00
    structural_min_pivot_distance_pct: float = 0.010
    structural_max_age_bars: int = 720
    structural_zone_atr_mult: float = 0.75
    structural_zone_pct: float = 0.007
    prefix: str = "pa"


def add_pivot_structure(
    dataframe: DataFrame,
    config: PivotStructureConfig | None = None,
    *,
    strength: int | None = None,
    strengths: Sequence[int] | None = None,
    atr_period: int | None = None,
    min_prominence_atr: float | None = None,
    min_prominence_pct: float | None = None,
    min_pivot_spacing_bars: int | None = None,
    min_pivot_distance_atr: float | None = None,
    min_pivot_distance_pct: float | None = None,
    max_pivot_age_bars: int | None = None,
    breakout_buffer_pct: float | None = None,
    structure_score_window: int | None = None,
    structural_strength: int | None = None,
    structural_min_prominence_atr: float | None = None,
    structural_min_prominence_pct: float | None = None,
    structural_min_pivot_spacing_bars: int | None = None,
    structural_min_pivot_distance_atr: float | None = None,
    structural_min_pivot_distance_pct: float | None = None,
    structural_max_age_bars: int | None = None,
    structural_zone_atr_mult: float | None = None,
    structural_zone_pct: float | None = None,
    prefix: str | None = None,
) -> DataFrame:
    """
    Append no-lookahead pivot structure and structural-zone columns.

    Per-strength columns are precomputed so strategies can choose an active
    strength later without recalculating indicators during hyperopt. Active
    aliases without a strength suffix are copied from ``config.strength``.

    Main output groups:
    - confirmed local pivots and pivot prominence
    - confirmed structural pivots and horizontal structure zones
    - current/previous swing levels and swing ages
    - simple two-pivot local support/resistance line evidence
    - HH/HL/LH/LL, trend-aligned/trend-flip breakouts, and structure state columns
    - normalized ``*_score_long/short/abs`` and directional ``*_state``
    """

    cfg = _resolve_config(
        config,
        strength=strength,
        strengths=strengths,
        atr_period=atr_period,
        min_prominence_atr=min_prominence_atr,
        min_prominence_pct=min_prominence_pct,
        min_pivot_spacing_bars=min_pivot_spacing_bars,
        min_pivot_distance_atr=min_pivot_distance_atr,
        min_pivot_distance_pct=min_pivot_distance_pct,
        max_pivot_age_bars=max_pivot_age_bars,
        breakout_buffer_pct=breakout_buffer_pct,
        structure_score_window=structure_score_window,
        structural_strength=structural_strength,
        structural_min_prominence_atr=structural_min_prominence_atr,
        structural_min_prominence_pct=structural_min_prominence_pct,
        structural_min_pivot_spacing_bars=structural_min_pivot_spacing_bars,
        structural_min_pivot_distance_atr=structural_min_pivot_distance_atr,
        structural_min_pivot_distance_pct=structural_min_pivot_distance_pct,
        structural_max_age_bars=structural_max_age_bars,
        structural_zone_atr_mult=structural_zone_atr_mult,
        structural_zone_pct=structural_zone_pct,
        prefix=prefix,
    )
    _validate_config(cfg)
    _validate_dataframe(dataframe)

    open_ = _num(dataframe["open"])
    high = _num(dataframe["high"])
    low = _num(dataframe["low"])
    close = _num(dataframe["close"]).replace(0, np.nan)
    body_high = pd.concat([open_, close], axis=1).max(axis=1)
    body_low = pd.concat([open_, close], axis=1).min(axis=1)
    atr = _atr(dataframe, cfg.atr_period)
    p = cfg.prefix

    bar_index = pd.Series(np.arange(len(dataframe), dtype="float64"), index=dataframe.index)

    new_cols: dict[str, Series] = {
        f"{p}_atr": atr,
        f"{p}_bar_index": bar_index,
    }

    strengths_list = _sorted_strengths(cfg)
    for pivot_strength in strengths_list:
        columns = _pivot_columns(
            dataframe.index,
            bar_index,
            body_high,
            body_low,
            close,
            atr,
            pivot_strength,
            cfg,
            p,
        )
        new_cols.update(columns)

    new_cols.update(
        _structural_pivot_columns(
            dataframe.index,
            bar_index,
            body_high,
            body_low,
            high,
            low,
            close,
            atr,
            int(cfg.structural_strength),
            cfg,
            p,
        )
    )

    selected = int(cfg.strength)
    if selected not in strengths_list:
        selected = strengths_list[0]
    new_cols.update(_active_alias_columns(new_cols, p, selected))
    new_cols.update(_pivot_diagnostic_columns(new_cols, dataframe.index, cfg, p))

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
    strength: int,
    cfg: PivotStructureConfig,
    prefix: str,
) -> dict[str, Series]:
    pivots = build_clean_pivot_source(
        body_high=high,
        body_low=low,
        atr=atr,
        bar_index=bar_index,
        strength=int(strength),
        method="body",
        min_prominence_atr=float(cfg.min_prominence_atr),
        min_prominence_pct=float(cfg.min_prominence_pct),
        min_pivot_spacing_bars=int(cfg.min_pivot_spacing_bars),
        min_pivot_distance_atr=float(cfg.min_pivot_distance_atr),
        min_pivot_distance_pct=float(cfg.min_pivot_distance_pct),
    )
    high_confirmed = pivots["pivot_high_confirmed"]
    low_confirmed = pivots["pivot_low_confirmed"]
    high_event_price = pivots["pivot_high"]
    low_event_price = pivots["pivot_low"]
    high_event_index = pivots["pivot_high_index"]
    low_event_index = pivots["pivot_low_index"]
    high_event_prominence = pivots["pivot_high_prominence"]
    low_event_prominence = pivots["pivot_low_prominence"]
    high_event_prominence_pct = pivots["pivot_high_prominence_pct"]
    low_event_prominence_pct = pivots["pivot_low_prominence_pct"]
    high_event_score = pivots["pivot_high_score"]
    low_event_score = pivots["pivot_low_score"]

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
    high_age = bar_index - high_state["last_index"]
    low_age = bar_index - low_state["last_index"]
    high_anchor_span = high_state["last_index"] - high_state["prev_index"]
    low_anchor_span = low_state["last_index"] - low_state["prev_index"]
    min_line_span = max(float(strength) * 3.0, float(cfg.min_pivot_spacing_bars))
    max_slope_pct_per_bar = 0.03
    resistance_line_valid = (
        resistance_line.notna()
        & high_anchor_span.ge(min_line_span)
        & high_age.le(float(cfg.max_pivot_age_bars))
        & _safe_div(resistance_slope.abs(), close).le(max_slope_pct_per_bar)
    ).fillna(False)
    support_line_valid = (
        support_line.notna()
        & low_anchor_span.ge(min_line_span)
        & low_age.le(float(cfg.max_pivot_age_bars))
        & _safe_div(support_slope.abs(), close).le(max_slope_pct_per_bar)
    ).fillna(False)
    resistance_line = resistance_line.where(resistance_line_valid)
    support_line = support_line.where(support_line_valid)
    resistance_slope_pct = (resistance_slope / close).where(resistance_line_valid)
    support_slope_pct = (support_slope / close).where(support_line_valid)

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
        float(cfg.breakout_buffer_pct),
        int(cfg.structure_score_window),
        prefix,
    )

    score_columns = _score_columns(
        prefix,
        strength,
        market_structure,
        int(cfg.structure_score_window),
    )

    s = int(strength)
    columns = {
        f"{prefix}_pivot_high_confirmed_{s}": high_confirmed.fillna(False),
        f"{prefix}_pivot_low_confirmed_{s}": low_confirmed.fillna(False),
        f"{prefix}_pivot_high_{s}": high_event_price,
        f"{prefix}_pivot_low_{s}": low_event_price,
        f"{prefix}_pivot_high_prominence_{s}": high_event_prominence,
        f"{prefix}_pivot_low_prominence_{s}": low_event_prominence,
        f"{prefix}_pivot_high_prominence_pct_{s}": high_event_prominence_pct,
        f"{prefix}_pivot_low_prominence_pct_{s}": low_event_prominence_pct,
        f"{prefix}_pivot_high_score_{s}": high_event_score,
        f"{prefix}_pivot_low_score_{s}": low_event_score,
        f"{prefix}_pivot_high_index_{s}": high_event_index,
        f"{prefix}_pivot_low_index_{s}": low_event_index,
        f"{prefix}_last_pivot_high_{s}": high_state["last_price"],
        f"{prefix}_last_pivot_low_{s}": low_state["last_price"],
        f"{prefix}_prev_pivot_high_{s}": high_state["prev_price"],
        f"{prefix}_prev_pivot_low_{s}": low_state["prev_price"],
        f"{prefix}_pivot_high_age_{s}": high_age,
        f"{prefix}_pivot_low_age_{s}": low_age,
        f"{prefix}_resistance_line_{s}": resistance_line,
        f"{prefix}_support_line_{s}": support_line,
        f"{prefix}_resistance_line_valid_{s}": resistance_line_valid,
        f"{prefix}_support_line_valid_{s}": support_line_valid,
        f"{prefix}_resistance_anchor_span_{s}": high_anchor_span.where(resistance_line_valid),
        f"{prefix}_support_anchor_span_{s}": low_anchor_span.where(support_line_valid),
        f"{prefix}_resistance_slope_pct_{s}": resistance_slope_pct,
        f"{prefix}_support_slope_pct_{s}": support_slope_pct,
    }
    columns.update(market_structure)
    columns.update(score_columns)
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
    score_window: int,
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

    bullish_trend_aligned_breakout = bullish_break & prior_state.ge(0.0)
    bearish_trend_aligned_breakout = bearish_break & prior_state.le(0.0)
    bullish_trend_flip_breakout = bullish_break & prior_state.lt(0.0)
    bearish_trend_flip_breakout = bearish_break & prior_state.gt(0.0)
    transition = pd.Series(
        np.select(
            [
                bullish_trend_aligned_breakout,
                bullish_trend_flip_breakout,
                bearish_trend_aligned_breakout,
                bearish_trend_flip_breakout,
            ],
            [2.0, 1.0, -2.0, -1.0],
            default=0.0,
        ),
        index=index,
    )
    higher_high_count = _rolling_count(higher_high, score_window)
    higher_low_count = _rolling_count(higher_low, score_window)
    lower_high_count = _rolling_count(lower_high, score_window)
    lower_low_count = _rolling_count(lower_low, score_window)
    up_sequence_score = _clip01((higher_high_count + higher_low_count) / max(float(score_window), 1.0))
    down_sequence_score = _clip01((lower_high_count + lower_low_count) / max(float(score_window), 1.0))

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
        f"{prefix}_ms_bullish_breakout_{s}": bullish_break.fillna(False),
        f"{prefix}_ms_bearish_breakout_{s}": bearish_break.fillna(False),
        f"{prefix}_ms_bullish_trend_aligned_breakout_{s}": bullish_trend_aligned_breakout.fillna(False),
        f"{prefix}_ms_bearish_trend_aligned_breakout_{s}": bearish_trend_aligned_breakout.fillna(False),
        f"{prefix}_ms_bullish_trend_flip_breakout_{s}": bullish_trend_flip_breakout.fillna(False),
        f"{prefix}_ms_bearish_trend_flip_breakout_{s}": bearish_trend_flip_breakout.fillna(False),
        f"{prefix}_ms_bullish_break_{s}": bullish_break.fillna(False),
        f"{prefix}_ms_bearish_break_{s}": bearish_break.fillna(False),
        f"{prefix}_ms_bullish_continuation_break_{s}": bullish_trend_aligned_breakout.fillna(False),
        f"{prefix}_ms_bearish_continuation_break_{s}": bearish_trend_aligned_breakout.fillna(False),
        f"{prefix}_ms_bullish_reversal_break_{s}": bullish_trend_flip_breakout.fillna(False),
        f"{prefix}_ms_bearish_reversal_break_{s}": bearish_trend_flip_breakout.fillna(False),
        f"{prefix}_ms_bullish_bos_{s}": bullish_trend_aligned_breakout.fillna(False),
        f"{prefix}_ms_bearish_bos_{s}": bearish_trend_aligned_breakout.fillna(False),
        f"{prefix}_ms_bullish_choch_{s}": bullish_trend_flip_breakout.fillna(False),
        f"{prefix}_ms_bearish_choch_{s}": bearish_trend_flip_breakout.fillna(False),
        f"{prefix}_ms_state_{s}": structure_state,
        f"{prefix}_ms_prior_state_{s}": prior_state,
        f"{prefix}_ms_transition_{s}": transition,
        f"{prefix}_ms_higher_high_count_{s}": higher_high_count,
        f"{prefix}_ms_higher_low_count_{s}": higher_low_count,
        f"{prefix}_ms_lower_high_count_{s}": lower_high_count,
        f"{prefix}_ms_lower_low_count_{s}": lower_low_count,
        f"{prefix}_ms_up_sequence_score_{s}": up_sequence_score,
        f"{prefix}_ms_down_sequence_score_{s}": down_sequence_score,
    }


def _structural_pivot_columns(
    index: pd.Index,
    bar_index: Series,
    pivot_high_source: Series,
    pivot_low_source: Series,
    high: Series,
    low: Series,
    close: Series,
    atr: Series,
    strength: int,
    cfg: PivotStructureConfig,
    prefix: str,
) -> dict[str, Series]:
    pivots = build_clean_pivot_source(
        body_high=pivot_high_source,
        body_low=pivot_low_source,
        atr=atr,
        bar_index=bar_index,
        strength=int(strength),
        method="body",
        min_prominence_atr=float(cfg.structural_min_prominence_atr),
        min_prominence_pct=float(cfg.structural_min_prominence_pct),
        min_pivot_spacing_bars=int(cfg.structural_min_pivot_spacing_bars),
        min_pivot_distance_atr=float(cfg.structural_min_pivot_distance_atr),
        min_pivot_distance_pct=float(cfg.structural_min_pivot_distance_pct),
    )
    high_confirmed = pivots["pivot_high_confirmed"]
    low_confirmed = pivots["pivot_low_confirmed"]
    high_event_price = pivots["pivot_high"]
    low_event_price = pivots["pivot_low"]
    high_event_index = pivots["pivot_high_index"]
    low_event_index = pivots["pivot_low_index"]
    high_event_prominence = pivots["pivot_high_prominence"]
    low_event_prominence = pivots["pivot_low_prominence"]
    high_event_prominence_pct = pivots["pivot_high_prominence_pct"]
    low_event_prominence_pct = pivots["pivot_low_prominence_pct"]
    high_event_score = pivots["pivot_high_score"]
    low_event_score = pivots["pivot_low_score"]
    high_state = _last_two_events(high_event_price, high_event_index, index)
    low_state = _last_two_events(low_event_price, low_event_index, index)
    high_age = bar_index - high_state["last_index"]
    low_age = bar_index - low_state["last_index"]

    max_age = float(cfg.structural_max_age_bars)
    structural_resistance = high_state["last_price"].where(high_age.le(max_age))
    structural_support = low_state["last_price"].where(low_age.le(max_age))
    structural_zone_width = pd.concat(
        [
            atr * float(cfg.structural_zone_atr_mult),
            close.abs() * float(cfg.structural_zone_pct),
        ],
        axis=1,
    ).max(axis=1)
    resistance_upper = structural_resistance + structural_zone_width
    resistance_lower = structural_resistance - structural_zone_width
    support_upper = structural_support + structural_zone_width
    support_lower = structural_support - structural_zone_width

    active_resistance = structural_resistance.shift(1)
    active_resistance_upper = resistance_upper.shift(1)
    active_resistance_lower = resistance_lower.shift(1)
    active_support = structural_support.shift(1)
    active_support_upper = support_upper.shift(1)
    active_support_lower = support_lower.shift(1)
    prev_close = close.shift(1)

    structural_resistance_break = (
        active_resistance_upper.notna()
        & close.gt(active_resistance_upper)
        & prev_close.le(active_resistance_upper)
    )
    structural_support_break = (
        active_support_lower.notna()
        & close.lt(active_support_lower)
        & prev_close.ge(active_support_lower)
    )
    structural_resistance_reject = (
        active_resistance.notna()
        & high.ge(active_resistance_lower)
        & close.lt(active_resistance)
        & prev_close.lt(active_resistance_upper)
    )
    structural_support_reclaim = (
        active_support.notna()
        & low.le(active_support_upper)
        & close.gt(active_support)
        & prev_close.gt(active_support_lower)
    )
    structural_cooldown = max(3, min(int(strength) // 2, 12))
    structural_resistance_break = _dedupe_events(structural_resistance_break, structural_cooldown)
    structural_support_break = _dedupe_events(structural_support_break, structural_cooldown)
    structural_resistance_reject = _dedupe_events(structural_resistance_reject, structural_cooldown)
    structural_support_reclaim = _dedupe_events(structural_support_reclaim, structural_cooldown)

    higher_high = high_confirmed & high_event_price.gt(high_state["prev_price"])
    lower_high = high_confirmed & high_event_price.lt(high_state["prev_price"])
    higher_low = low_confirmed & low_event_price.gt(low_state["prev_price"])
    lower_low = low_confirmed & low_event_price.lt(low_state["prev_price"])
    last_high_class = pd.Series(
        np.select([higher_high, lower_high], [1.0, -1.0], default=np.nan),
        index=index,
    ).ffill()
    last_low_class = pd.Series(
        np.select([higher_low, lower_low], [1.0, -1.0], default=np.nan),
        index=index,
    ).ffill()
    structural_state = pd.Series(
        np.select(
            [
                last_high_class.eq(1.0) & last_low_class.eq(1.0),
                last_high_class.eq(-1.0) & last_low_class.eq(-1.0),
            ],
            [1.0, -1.0],
            default=0.0,
        ),
        index=index,
    )
    structural_range = structural_resistance - structural_support
    structural_range_pct = _safe_div(structural_range, close).where(structural_range.gt(0.0))
    structural_range_position = _clip01(_safe_div(close - structural_support, structural_range))
    near_structural_resistance = (
        active_resistance.notna()
        & high.ge(active_resistance_lower)
        & low.le(active_resistance_upper)
    )
    near_structural_support = (
        active_support.notna()
        & low.le(active_support_upper)
        & high.ge(active_support_lower)
    )

    return {
        f"{prefix}_structural_strength": pd.Series(float(strength), index=index),
        f"{prefix}_structural_pivot_high_confirmed": high_confirmed.fillna(False),
        f"{prefix}_structural_pivot_low_confirmed": low_confirmed.fillna(False),
        f"{prefix}_structural_pivot_high": high_event_price,
        f"{prefix}_structural_pivot_low": low_event_price,
        f"{prefix}_structural_pivot_high_index": high_event_index,
        f"{prefix}_structural_pivot_low_index": low_event_index,
        f"{prefix}_structural_pivot_high_prominence": high_event_prominence,
        f"{prefix}_structural_pivot_low_prominence": low_event_prominence,
        f"{prefix}_structural_pivot_high_prominence_pct": high_event_prominence_pct,
        f"{prefix}_structural_pivot_low_prominence_pct": low_event_prominence_pct,
        f"{prefix}_structural_pivot_high_score": high_event_score,
        f"{prefix}_structural_pivot_low_score": low_event_score,
        f"{prefix}_structural_resistance": structural_resistance,
        f"{prefix}_structural_support": structural_support,
        f"{prefix}_structural_resistance_zone_upper": resistance_upper,
        f"{prefix}_structural_resistance_zone_lower": resistance_lower,
        f"{prefix}_structural_support_zone_upper": support_upper,
        f"{prefix}_structural_support_zone_lower": support_lower,
        f"{prefix}_structural_pivot_high_age": high_age,
        f"{prefix}_structural_pivot_low_age": low_age,
        f"{prefix}_structural_range_pct": structural_range_pct,
        f"{prefix}_structural_range_position": structural_range_position,
        f"{prefix}_near_structural_resistance": near_structural_resistance.fillna(False),
        f"{prefix}_near_structural_support": near_structural_support.fillna(False),
        f"{prefix}_structural_resistance_break": structural_resistance_break.fillna(False),
        f"{prefix}_structural_support_break": structural_support_break.fillna(False),
        f"{prefix}_structural_resistance_reject": structural_resistance_reject.fillna(False),
        f"{prefix}_structural_support_reclaim": structural_support_reclaim.fillna(False),
        f"{prefix}_structural_higher_high": higher_high.fillna(False),
        f"{prefix}_structural_lower_high": lower_high.fillna(False),
        f"{prefix}_structural_higher_low": higher_low.fillna(False),
        f"{prefix}_structural_lower_low": lower_low.fillna(False),
        f"{prefix}_structural_state": structural_state,
    }


def _score_columns(
    prefix: str,
    strength: int,
    market_structure: dict[str, Series],
    score_window: int,
) -> dict[str, Series]:
    s = int(strength)
    up_sequence = market_structure[f"{prefix}_ms_up_sequence_score_{s}"]
    down_sequence = market_structure[f"{prefix}_ms_down_sequence_score_{s}"]
    state = market_structure[f"{prefix}_ms_state_{s}"]
    bullish_breaks = _rolling_count(market_structure[f"{prefix}_ms_bullish_breakout_{s}"], score_window)
    bearish_breaks = _rolling_count(market_structure[f"{prefix}_ms_bearish_breakout_{s}"], score_window)
    break_norm = max(float(score_window) / 3.0, 1.0)

    long_score = _clip01(
        0.45 * up_sequence
        + 0.35 * (state.gt(0.0).astype("float64"))
        + 0.20 * _clip01(bullish_breaks / break_norm)
    )
    short_score = _clip01(
        0.45 * down_sequence
        + 0.35 * (state.lt(0.0).astype("float64"))
        + 0.20 * _clip01(bearish_breaks / break_norm)
    )
    abs_score = pd.concat([long_score, short_score], axis=1).max(axis=1)
    score_state = pd.Series(
        np.select([long_score.gt(short_score), short_score.gt(long_score)], [1.0, -1.0], default=0.0),
        index=long_score.index,
    )
    suggested_entry_long = (
        market_structure[f"{prefix}_ms_bullish_breakout_{s}"]
        | market_structure[f"{prefix}_ms_bullish_trend_aligned_breakout_{s}"]
        | market_structure[f"{prefix}_ms_bullish_trend_flip_breakout_{s}"]
    ) & long_score.ge(short_score)
    suggested_entry_short = (
        market_structure[f"{prefix}_ms_bearish_breakout_{s}"]
        | market_structure[f"{prefix}_ms_bearish_trend_aligned_breakout_{s}"]
        | market_structure[f"{prefix}_ms_bearish_trend_flip_breakout_{s}"]
    ) & short_score.ge(long_score)
    return {
        f"{prefix}_suggested_entry_long_{s}": suggested_entry_long,
        f"{prefix}_suggested_entry_short_{s}": suggested_entry_short,
        f"{prefix}_score_long_{s}": long_score,
        f"{prefix}_score_short_{s}": short_score,
        f"{prefix}_score_abs_{s}": abs_score,
        f"{prefix}_state_{s}": score_state,
    }


def _pivot_diagnostic_columns(
    columns: dict[str, Series],
    index: pd.Index,
    cfg: PivotStructureConfig,
    prefix: str,
) -> dict[str, Series]:
    score_long = _column_or_default(columns, f"{prefix}_score_long", index, 0.0)
    score_short = _column_or_default(columns, f"{prefix}_score_short", index, 0.0)
    score_abs = _column_or_default(columns, f"{prefix}_score_abs", index, 0.0)
    up_sequence = _column_or_default(columns, f"{prefix}_ms_up_sequence_score", index, 0.0)
    down_sequence = _column_or_default(columns, f"{prefix}_ms_down_sequence_score", index, 0.0)
    structural_state = _column_or_default(columns, f"{prefix}_structural_state", index, 0.0)
    range_position = _column_or_default(columns, f"{prefix}_structural_range_position", index, 0.5)
    structural_resistance_break = _bool_column(columns, f"{prefix}_structural_resistance_break", index)
    structural_support_break = _bool_column(columns, f"{prefix}_structural_support_break", index)
    structural_resistance_reject = _bool_column(columns, f"{prefix}_structural_resistance_reject", index)
    structural_support_reclaim = _bool_column(columns, f"{prefix}_structural_support_reclaim", index)
    bullish_trend_aligned_breakout = _bool_column(columns, f"{prefix}_ms_bullish_trend_aligned_breakout", index)
    bearish_trend_aligned_breakout = _bool_column(columns, f"{prefix}_ms_bearish_trend_aligned_breakout", index)
    bullish_trend_flip_breakout = _bool_column(columns, f"{prefix}_ms_bullish_trend_flip_breakout", index)
    bearish_trend_flip_breakout = _bool_column(columns, f"{prefix}_ms_bearish_trend_flip_breakout", index)

    bull_events = (
        structural_resistance_break
        | structural_support_reclaim
        | bullish_trend_aligned_breakout
        | bullish_trend_flip_breakout
    )
    bear_events = (
        structural_support_break
        | structural_resistance_reject
        | bearish_trend_aligned_breakout
        | bearish_trend_flip_breakout
    )
    context_window = max(6, int(cfg.structure_score_window) * 2)
    recent_bull_structure_event = _rolling_count(structural_resistance_break | structural_support_reclaim, context_window).ge(1.0)
    recent_bear_structure_event = _rolling_count(structural_support_break | structural_resistance_reject, context_window).ge(1.0)
    bull_event_density = _clip01(_rolling_count(bull_events, context_window) / 4.0)
    bear_event_density = _clip01(_rolling_count(bear_events, context_window) / 4.0)
    balance_score = _clip01(
        0.40 * (1.0 - score_abs)
        + 0.34 * (1.0 - (range_position - 0.5).abs() * 2.0)
        + 0.26 * (structural_state.eq(0.0).astype("float64"))
    )
    bull_context_score = _clip01(
        0.34 * score_long
        + 0.23 * structural_state.gt(0.0).astype("float64")
        + 0.18 * bull_event_density
        + 0.15 * up_sequence
        + 0.10 * range_position.ge(0.55).astype("float64")
    )
    bear_context_score = _clip01(
        0.34 * score_short
        + 0.23 * structural_state.lt(0.0).astype("float64")
        + 0.18 * bear_event_density
        + 0.15 * down_sequence
        + 0.10 * range_position.le(0.45).astype("float64")
    )
    smooth = max(3, min(int(cfg.structure_score_window), 12))
    bull_context_score = _clip01(bull_context_score.rolling(smooth, min_periods=1).mean())
    bear_context_score = _clip01(bear_context_score.rolling(smooth, min_periods=1).mean())
    balance_score = _clip01(balance_score.rolling(smooth, min_periods=1).mean())

    strong_bull_structure = structural_state.gt(0.0) | recent_bull_structure_event
    strong_bear_structure = structural_state.lt(0.0) | recent_bear_structure_event
    trend_full_bull_raw = (
        bull_context_score.ge(0.50)
        & bull_context_score.gt(bear_context_score + 0.10)
        & strong_bull_structure
        & ~structural_state.lt(0.0)
        & balance_score.lt(0.64)
    )
    event_full_bull_raw = (
        recent_bull_structure_event
        & bull_context_score.ge(0.42)
        & bull_context_score.gt(bear_context_score + 0.06)
        & balance_score.lt(0.76)
    )
    trend_full_bear_raw = (
        bear_context_score.ge(0.50)
        & bear_context_score.gt(bull_context_score + 0.10)
        & strong_bear_structure
        & ~structural_state.gt(0.0)
        & balance_score.lt(0.64)
    )
    event_full_bear_raw = (
        recent_bear_structure_event
        & bear_context_score.ge(0.42)
        & bear_context_score.gt(bull_context_score + 0.06)
        & balance_score.lt(0.76)
    )
    full_bull_raw = trend_full_bull_raw | event_full_bull_raw
    full_bear_raw = trend_full_bear_raw | event_full_bear_raw
    full_confirm_window = max(2, min(3, int(cfg.structure_score_window) // 4))
    full_confirm_count = max(2.0, float(full_confirm_window) - 1.0)
    full_bull = _rolling_count(full_bull_raw, full_confirm_window).ge(full_confirm_count)
    full_bear = _rolling_count(full_bear_raw, full_confirm_window).ge(full_confirm_count)
    direction_margin = bull_context_score - bear_context_score
    bullish_chop = (
        ~full_bull
        & ~full_bear
        & direction_margin.ge(0.04)
        & bull_context_score.ge(0.28)
        & (balance_score.ge(0.24) | structural_state.ge(0.0) | recent_bull_structure_event)
        & ~strong_bear_structure
    )
    bearish_chop = (
        ~full_bull
        & ~full_bear
        & direction_margin.le(-0.04)
        & bear_context_score.ge(0.28)
        & (balance_score.ge(0.24) | structural_state.le(0.0) | recent_bear_structure_event)
        & ~strong_bull_structure
    )
    market_context = pd.Series(
        np.select([full_bull, full_bear, bullish_chop, bearish_chop], [2, -2, 1, -1], default=0),
        index=index,
        dtype="int8",
    )

    context_not_full_bear = market_context.gt(-2)
    context_not_full_bull = market_context.lt(2)
    context_bullish = market_context.ge(1)
    context_bearish = market_context.le(-1)
    score_long_ok = score_long.ge(score_short)
    score_short_ok = score_short.ge(score_long)

    long_breakout = structural_resistance_break & score_long_ok
    long_reclaim = structural_support_reclaim & context_not_full_bear
    long_trend_flip_breakout = (
        bullish_trend_flip_breakout
        & context_not_full_bear
        & range_position.le(0.65)
    )
    long_trend_aligned_breakout = (
        bullish_trend_aligned_breakout
        & context_bullish
        & score_long_ok
    )
    long_range_support = (
        _bool_column(columns, f"{prefix}_near_structural_support", index)
        & score_long.gt(score_short + 0.04)
        & structural_state.ge(0.0)
        & range_position.le(0.42)
    )

    short_breakdown = structural_support_break & score_short_ok
    short_reject = structural_resistance_reject & context_not_full_bull
    short_trend_flip_breakout = (
        bearish_trend_flip_breakout
        & context_not_full_bull
        & range_position.ge(0.35)
    )
    short_trend_aligned_breakout = (
        bearish_trend_aligned_breakout
        & context_bearish
        & score_short_ok
    )
    short_range_resistance = (
        _bool_column(columns, f"{prefix}_near_structural_resistance", index)
        & score_short.gt(score_long + 0.04)
        & structural_state.le(0.0)
        & range_position.ge(0.58)
    )

    cooldown = max(3, int(cfg.structure_score_window) // 2)
    primary_entries = {
        f"{prefix}_entry_resistance_breakout_long": _dedupe_events(long_breakout, cooldown),
        f"{prefix}_entry_support_reclaim_long": _dedupe_events(long_reclaim, cooldown),
        f"{prefix}_entry_bullish_trend_flip_breakout_long": _dedupe_events(long_trend_flip_breakout, cooldown),
        f"{prefix}_entry_bullish_trend_aligned_breakout_long": _dedupe_events(long_trend_aligned_breakout, cooldown),
        f"{prefix}_entry_range_support_long": _dedupe_events(long_range_support, cooldown),
        f"{prefix}_entry_support_breakdown_short": _dedupe_events(short_breakdown, cooldown),
        f"{prefix}_entry_resistance_reject_short": _dedupe_events(short_reject, cooldown),
        f"{prefix}_entry_bearish_trend_flip_breakout_short": _dedupe_events(short_trend_flip_breakout, cooldown),
        f"{prefix}_entry_bearish_trend_aligned_breakout_short": _dedupe_events(short_trend_aligned_breakout, cooldown),
        f"{prefix}_entry_range_resistance_short": _dedupe_events(short_range_resistance, cooldown),
    }
    legacy_entries = {
        f"{prefix}_entry_bullish_reversal_break_long": primary_entries[
            f"{prefix}_entry_bullish_trend_flip_breakout_long"
        ],
        f"{prefix}_entry_bullish_continuation_break_long": primary_entries[
            f"{prefix}_entry_bullish_trend_aligned_breakout_long"
        ],
        f"{prefix}_entry_bearish_reversal_break_short": primary_entries[
            f"{prefix}_entry_bearish_trend_flip_breakout_short"
        ],
        f"{prefix}_entry_bearish_continuation_break_short": primary_entries[
            f"{prefix}_entry_bearish_trend_aligned_breakout_short"
        ],
        f"{prefix}_entry_bullish_choch_reversal_long": primary_entries[
            f"{prefix}_entry_bullish_trend_flip_breakout_long"
        ],
        f"{prefix}_entry_bullish_bos_continuation_long": primary_entries[
            f"{prefix}_entry_bullish_trend_aligned_breakout_long"
        ],
        f"{prefix}_entry_bearish_choch_reversal_short": primary_entries[
            f"{prefix}_entry_bearish_trend_flip_breakout_short"
        ],
        f"{prefix}_entry_bearish_bos_continuation_short": primary_entries[
            f"{prefix}_entry_bearish_trend_aligned_breakout_short"
        ],
    }
    entries = {**primary_entries, **legacy_entries}
    entry_any_long = pd.concat(
        [series for name, series in primary_entries.items() if name.endswith("_long")],
        axis=1,
    ).any(axis=1)
    entry_any_short = pd.concat(
        [series for name, series in primary_entries.items() if name.endswith("_short")],
        axis=1,
    ).any(axis=1)

    return {
        f"{prefix}_context_score_bull": bull_context_score,
        f"{prefix}_context_score_bear": bear_context_score,
        f"{prefix}_context_score_balance": balance_score,
        f"{prefix}_market_context": market_context,
        **entries,
        f"{prefix}_entry_any_long": entry_any_long.fillna(False),
        f"{prefix}_entry_any_short": entry_any_short.fillna(False),
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


def _column_or_default(columns: dict[str, Series], column: str, index: pd.Index, default: float) -> Series:
    if column not in columns:
        return pd.Series(float(default), index=index, dtype="float64")
    return pd.to_numeric(columns[column], errors="coerce").reindex(index).fillna(float(default))


def _bool_column(columns: dict[str, Series], column: str, index: pd.Index) -> Series:
    if column not in columns:
        return pd.Series(False, index=index, dtype="bool")
    return pd.Series(columns[column], index=index).fillna(False).astype("bool")


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


def _rolling_count(mask: Series, window: int) -> Series:
    clean = pd.Series(mask, index=mask.index).fillna(False).astype("int8")
    cumulative = clean.cumsum()
    return (cumulative - cumulative.shift(int(window), fill_value=0)).astype("float64")


def _dedupe_events(mask: Series, cooldown_bars: int) -> Series:
    clean = pd.Series(mask, index=mask.index).fillna(False).astype("bool")
    prior_recent = _rolling_count(clean.shift(1, fill_value=False), max(int(cooldown_bars), 1))
    return (clean & prior_recent.eq(0.0)).fillna(False)


def _safe_div(numerator: Series, denominator: Series) -> Series:
    denominator = denominator.where(denominator.abs() > 0.0, np.nan)
    return numerator / denominator


def _clip01(series: Series) -> Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0).fillna(0.0)


def _active_alias_columns(columns: dict[str, Series], prefix: str, strength: int) -> dict[str, Series]:
    aliases = (
        "pivot_high_confirmed",
        "pivot_low_confirmed",
        "pivot_high",
        "pivot_low",
        "pivot_high_prominence",
        "pivot_low_prominence",
        "pivot_high_prominence_pct",
        "pivot_low_prominence_pct",
        "pivot_high_score",
        "pivot_low_score",
        "pivot_high_index",
        "pivot_low_index",
        "last_pivot_high",
        "last_pivot_low",
        "prev_pivot_high",
        "prev_pivot_low",
        "pivot_high_age",
        "pivot_low_age",
        "resistance_line",
        "support_line",
        "resistance_line_valid",
        "support_line_valid",
        "resistance_anchor_span",
        "support_anchor_span",
        "resistance_slope_pct",
        "support_slope_pct",
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
        "ms_bullish_breakout",
        "ms_bearish_breakout",
        "ms_bullish_trend_aligned_breakout",
        "ms_bearish_trend_aligned_breakout",
        "ms_bullish_trend_flip_breakout",
        "ms_bearish_trend_flip_breakout",
        "ms_bullish_break",
        "ms_bearish_break",
        "ms_bullish_continuation_break",
        "ms_bearish_continuation_break",
        "ms_bullish_reversal_break",
        "ms_bearish_reversal_break",
        "ms_bullish_bos",
        "ms_bearish_bos",
        "ms_bullish_choch",
        "ms_bearish_choch",
        "ms_state",
        "ms_prior_state",
        "ms_transition",
        "ms_higher_high_count",
        "ms_higher_low_count",
        "ms_lower_high_count",
        "ms_lower_low_count",
        "ms_up_sequence_score",
        "ms_down_sequence_score",
        "suggested_entry_long",
        "suggested_entry_short",
        "score_long",
        "score_short",
        "score_abs",
        "state",
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
        "min_prominence_pct": cfg.min_prominence_pct,
        "min_pivot_spacing_bars": cfg.min_pivot_spacing_bars,
        "min_pivot_distance_atr": cfg.min_pivot_distance_atr,
        "min_pivot_distance_pct": cfg.min_pivot_distance_pct,
        "max_pivot_age_bars": cfg.max_pivot_age_bars,
        "breakout_buffer_pct": cfg.breakout_buffer_pct,
        "structure_score_window": cfg.structure_score_window,
        "structural_strength": cfg.structural_strength,
        "structural_min_prominence_atr": cfg.structural_min_prominence_atr,
        "structural_min_prominence_pct": cfg.structural_min_prominence_pct,
        "structural_min_pivot_spacing_bars": cfg.structural_min_pivot_spacing_bars,
        "structural_min_pivot_distance_atr": cfg.structural_min_pivot_distance_atr,
        "structural_min_pivot_distance_pct": cfg.structural_min_pivot_distance_pct,
        "structural_max_age_bars": cfg.structural_max_age_bars,
        "structural_zone_atr_mult": cfg.structural_zone_atr_mult,
        "structural_zone_pct": cfg.structural_zone_pct,
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
    if cfg.min_prominence_pct < 0:
        raise ValueError("min_prominence_pct must be non-negative")
    if cfg.min_pivot_spacing_bars < 1:
        raise ValueError("min_pivot_spacing_bars must be at least 1")
    if cfg.min_pivot_distance_atr < 0:
        raise ValueError("min_pivot_distance_atr must be non-negative")
    if cfg.min_pivot_distance_pct < 0:
        raise ValueError("min_pivot_distance_pct must be non-negative")
    if cfg.max_pivot_age_bars < 1:
        raise ValueError("max_pivot_age_bars must be at least 1")
    if cfg.breakout_buffer_pct < 0:
        raise ValueError("breakout_buffer_pct must be non-negative")
    if cfg.structure_score_window < 2:
        raise ValueError("structure_score_window must be at least 2")
    if cfg.structural_strength < 1:
        raise ValueError("structural_strength must be at least 1")
    if cfg.structural_min_prominence_atr < 0:
        raise ValueError("structural_min_prominence_atr must be non-negative")
    if cfg.structural_min_prominence_pct < 0:
        raise ValueError("structural_min_prominence_pct must be non-negative")
    if cfg.structural_min_pivot_spacing_bars < 1:
        raise ValueError("structural_min_pivot_spacing_bars must be at least 1")
    if cfg.structural_min_pivot_distance_atr < 0:
        raise ValueError("structural_min_pivot_distance_atr must be non-negative")
    if cfg.structural_min_pivot_distance_pct < 0:
        raise ValueError("structural_min_pivot_distance_pct must be non-negative")
    if cfg.structural_max_age_bars < 1:
        raise ValueError("structural_max_age_bars must be at least 1")
    if cfg.structural_zone_atr_mult < 0:
        raise ValueError("structural_zone_atr_mult must be non-negative")
    if cfg.structural_zone_pct < 0:
        raise ValueError("structural_zone_pct must be non-negative")
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

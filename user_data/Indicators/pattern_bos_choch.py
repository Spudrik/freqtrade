from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

try:
    from .pivot_foundation import build_clean_pivot_source
except Exception:  # pragma: no cover - optional fallback for standalone notebooks
    from pivot_foundation import build_clean_pivot_source  # type: ignore[no-redef]


@dataclass(frozen=True)
class BosChochConfig:
    """Confirmed-pivot break-of-structure and change-of-character events.

    This indicator has one narrow job: use confirmed pivots to label structural
    breaks. It does not score the setup, draw trendlines/channels, infer ranges,
    or create support/resistance zones. Those are owned by the focused pattern,
    geometry, and trendline indicators.

    Event convention:
    - ``*_structure_event`` = ``2`` for BOS to bull.
    - ``*_structure_event`` = ``1`` for CHoCH to bull.
    - ``*_structure_event`` = ``-1`` for CHoCH to bear.
    - ``*_structure_event`` = ``-2`` for BOS to bear.
    - ``*_structure_event`` = ``0`` when no structural break occurred.

    BOS means price broke the active swing level in the same direction as the
    previous confirmed swing state. CHoCH means price broke against that state.
    """

    strength: int = 5
    atr_period: int = 14
    min_prominence_atr: float = 0.35
    min_pivot_spacing_bars: int = 1
    min_pivot_distance_atr: float = 0.0
    max_pivot_age_bars: int = 240
    breakout_buffer_atr: float = 0.15
    include_sequence: bool = False
    include_diagnostics: bool = False
    prefix: str = "ms"


def add_bos_choch(
    dataframe: DataFrame,
    config: BosChochConfig | None = None,
    *,
    strength: int | None = None,
    atr_period: int | None = None,
    min_prominence_atr: float | None = None,
    min_pivot_spacing_bars: int | None = None,
    min_pivot_distance_atr: float | None = None,
    max_pivot_age_bars: int | None = None,
    breakout_buffer_atr: float | None = None,
    include_sequence: bool | None = None,
    include_diagnostics: bool | None = None,
    prefix: str | None = None,
) -> DataFrame:
    """Append BOS/CHoCH columns to an OHLC dataframe.

    Pivots are no-lookahead. A strength-5 pivot anchored five bars ago is only
    visible after five right-side bars have closed. Diagnostic index columns are
    therefore available when plots need to distinguish anchor and confirmation.
    """

    cfg = _resolve_config(
        config,
        strength=strength,
        atr_period=atr_period,
        min_prominence_atr=min_prominence_atr,
        min_pivot_spacing_bars=min_pivot_spacing_bars,
        min_pivot_distance_atr=min_pivot_distance_atr,
        max_pivot_age_bars=max_pivot_age_bars,
        breakout_buffer_atr=breakout_buffer_atr,
        include_sequence=include_sequence,
        include_diagnostics=include_diagnostics,
        prefix=prefix,
    )
    _validate_config(cfg)
    _validate_dataframe(dataframe)

    open_ = _num(dataframe["open"])
    high = _num(dataframe["high"])
    low = _num(dataframe["low"])
    close = _num(dataframe["close"]).replace(0.0, np.nan)
    body_high = pd.concat([open_, close], axis=1).max(axis=1)
    body_low = pd.concat([open_, close], axis=1).min(axis=1)
    atr = _atr(dataframe, cfg.atr_period)
    bar_index = pd.Series(np.arange(len(dataframe), dtype="float64"), index=dataframe.index)
    p = cfg.prefix

    pivots = build_clean_pivot_source(
        body_high=body_high,
        body_low=body_low,
        atr=atr,
        bar_index=bar_index,
        strength=int(cfg.strength),
        min_prominence_atr=float(cfg.min_prominence_atr),
        min_prominence_pct=0.0,
        min_pivot_spacing_bars=int(cfg.min_pivot_spacing_bars),
        min_pivot_distance_atr=float(cfg.min_pivot_distance_atr),
        min_pivot_distance_pct=0.0,
    )

    high_confirmed = pivots["pivot_high_confirmed"].fillna(False)
    low_confirmed = pivots["pivot_low_confirmed"].fillna(False)
    high_state = _last_two_events(
        pivots["pivot_high"],
        pivots["pivot_high_index"],
        pivots["pivot_high_available_index"],
        dataframe.index,
    )
    low_state = _last_two_events(
        pivots["pivot_low"],
        pivots["pivot_low_index"],
        pivots["pivot_low_available_index"],
        dataframe.index,
    )

    max_age = float(cfg.max_pivot_age_bars)
    high_anchor_age = bar_index - high_state["last_index"]
    low_anchor_age = bar_index - low_state["last_index"]
    active_high = high_state["last_price"].where(high_anchor_age.le(max_age))
    active_low = low_state["last_price"].where(low_anchor_age.le(max_age))
    prior_high = high_state["prev_price"]
    prior_low = low_state["prev_price"]

    higher_high = (high_confirmed & pivots["pivot_high"].gt(prior_high)).fillna(False)
    lower_high = (high_confirmed & pivots["pivot_high"].lt(prior_high)).fillna(False)
    higher_low = (low_confirmed & pivots["pivot_low"].gt(prior_low)).fillna(False)
    lower_low = (low_confirmed & pivots["pivot_low"].lt(prior_low)).fillna(False)

    high_class = pd.Series(np.select([higher_high, lower_high], [1.0, -1.0], default=np.nan), index=dataframe.index)
    low_class = pd.Series(np.select([higher_low, lower_low], [1.0, -1.0], default=np.nan), index=dataframe.index)
    last_high_class = high_class.ffill()
    last_low_class = low_class.ffill()
    state = pd.Series(
        np.select(
            [
                last_high_class.eq(1.0) & last_low_class.eq(1.0),
                last_high_class.eq(-1.0) & last_low_class.eq(-1.0),
            ],
            [1.0, -1.0],
            default=0.0,
        ),
        index=dataframe.index,
        dtype="float64",
    )
    prior_state = state.replace(0.0, np.nan).ffill().shift(1).fillna(0.0)

    breakout_buffer = atr.fillna(0.0) * float(cfg.breakout_buffer_atr)
    bullish_break_level = active_high.shift(1) + breakout_buffer
    bearish_break_level = active_low.shift(1) - breakout_buffer
    prev_close = close.shift(1)
    bullish_break = (
        bullish_break_level.notna()
        & close.gt(bullish_break_level)
        & prev_close.le(bullish_break_level)
    ).fillna(False)
    bearish_break = (
        bearish_break_level.notna()
        & close.lt(bearish_break_level)
        & prev_close.ge(bearish_break_level)
    ).fillna(False)

    bos_to_bull = (bullish_break & prior_state.gt(0.0)).fillna(False)
    bos_to_bear = (bearish_break & prior_state.lt(0.0)).fillna(False)
    choch_to_bull = (bullish_break & prior_state.lt(0.0)).fillna(False)
    choch_to_bear = (bearish_break & prior_state.gt(0.0)).fillna(False)
    structure_event = pd.Series(
        np.select(
            [bos_to_bull, choch_to_bull, choch_to_bear, bos_to_bear],
            [2.0, 1.0, -1.0, -2.0],
            default=0.0,
        ),
        index=dataframe.index,
        dtype="float64",
    )
    break_level = pd.Series(
        np.select(
            [bullish_break, bearish_break],
            [bullish_break_level, bearish_break_level],
            default=np.nan,
        ),
        index=dataframe.index,
        dtype="float64",
    )
    invalidation_level = pd.Series(
        np.select(
            [state.gt(0.0), state.lt(0.0)],
            [active_low, active_high],
            default=np.nan,
        ),
        index=dataframe.index,
        dtype="float64",
    )

    new_cols: dict[str, Series] = {
        f"{p}_bos_to_bull": bos_to_bull,
        f"{p}_bos_to_bear": bos_to_bear,
        f"{p}_choch_to_bull": choch_to_bull,
        f"{p}_choch_to_bear": choch_to_bear,
        f"{p}_structure_event": structure_event,
        f"{p}_state": state,
    }
    if cfg.include_sequence:
        new_cols.update(
            {
                f"{p}_higher_high": higher_high,
                f"{p}_lower_high": lower_high,
                f"{p}_higher_low": higher_low,
                f"{p}_lower_low": lower_low,
            }
        )
    if cfg.include_diagnostics:
        new_cols.update(
            {
                f"{p}_atr": atr,
                f"{p}_bar_index": bar_index,
                f"{p}_pivot_high_confirmed": high_confirmed,
                f"{p}_pivot_low_confirmed": low_confirmed,
                f"{p}_pivot_high": pivots["pivot_high"],
                f"{p}_pivot_low": pivots["pivot_low"],
                f"{p}_pivot_high_index": pivots["pivot_high_index"],
                f"{p}_pivot_low_index": pivots["pivot_low_index"],
                f"{p}_pivot_high_available_index": pivots["pivot_high_available_index"],
                f"{p}_pivot_low_available_index": pivots["pivot_low_available_index"],
                f"{p}_prev_swing_high": prior_high,
                f"{p}_prev_swing_low": prior_low,
                f"{p}_last_swing_high": active_high,
                f"{p}_last_swing_low": active_low,
                f"{p}_break_level": break_level,
                f"{p}_invalidation_level": invalidation_level,
                f"{p}_bullish_break_level": bullish_break_level,
                f"{p}_bearish_break_level": bearish_break_level,
            }
        )

    existing = [col for col in dataframe.columns if str(col).startswith(f"{p}_")]
    base = dataframe.drop(columns=existing).copy() if existing else dataframe.copy()
    return pd.concat([base, pd.DataFrame(new_cols, index=dataframe.index)], axis=1)


def _last_two_events(event_price: Series, event_index: Series, available_index: Series, index: pd.Index) -> dict[str, Series]:
    price_events = event_price.dropna()
    index_events = event_index.dropna()
    available_events = available_index.dropna()
    return {
        "last_price": event_price.ffill(),
        "last_index": event_index.ffill(),
        "last_available_index": available_index.ffill(),
        "prev_price": price_events.shift(1).reindex(index).ffill(),
        "prev_index": index_events.shift(1).reindex(index).ffill(),
        "prev_available_index": available_events.shift(1).reindex(index).ffill(),
    }


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


def _resolve_config(config: BosChochConfig | None, **overrides: object) -> BosChochConfig:
    cfg = config or BosChochConfig()
    values = dict(cfg.__dict__)
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return BosChochConfig(**values)


def _validate_config(cfg: BosChochConfig) -> None:
    if cfg.strength < 1:
        raise ValueError("strength must be at least 1")
    if cfg.atr_period < 2:
        raise ValueError("atr_period must be at least 2")
    if cfg.min_prominence_atr < 0:
        raise ValueError("min_prominence_atr must be non-negative")
    if cfg.min_pivot_spacing_bars < 1:
        raise ValueError("min_pivot_spacing_bars must be at least 1")
    if cfg.min_pivot_distance_atr < 0:
        raise ValueError("min_pivot_distance_atr must be non-negative")
    if cfg.max_pivot_age_bars < 1:
        raise ValueError("max_pivot_age_bars must be at least 1")
    if cfg.breakout_buffer_atr < 0:
        raise ValueError("breakout_buffer_atr must be non-negative")
    if not cfg.prefix:
        raise ValueError("prefix must not be empty")


def _validate_dataframe(dataframe: DataFrame) -> None:
    required = {"open", "high", "low", "close"}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"DataFrame is missing OHLC columns: {missing}")


def _num(series: Series) -> Series:
    return pd.to_numeric(series, errors="coerce")


__all__ = ["BosChochConfig", "add_bos_choch"]

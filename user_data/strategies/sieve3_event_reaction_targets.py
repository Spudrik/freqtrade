from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


TARGET_HORIZONS = (1, 2, 4, 8, 12, 24, 48)
TOUCH_HORIZONS = (4, 8, 12, 24, 48)
EXIT_ROLE_HORIZONS = (1, 2, 4, 8, 24, 48)
EXIT_ROLE_SEQUENCE_HORIZONS = (4, 24, 48)


def _atr(dataframe: DataFrame, period: int = 14) -> Series:
    high = pd.to_numeric(dataframe["high"], errors="coerce")
    low = pd.to_numeric(dataframe["low"], errors="coerce")
    close = pd.to_numeric(dataframe["close"], errors="coerce")
    prior_close = close.shift(1)
    true_range = pd.concat(
        ((high - low).abs(), (high - prior_close).abs(), (low - prior_close).abs()),
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period, min_periods=period).mean()


def _future_extreme(series: Series, horizon: int, method: str) -> Series:
    future = pd.to_numeric(series, errors="coerce").shift(-1).iloc[::-1]
    rolling = future.rolling(horizon, min_periods=horizon)
    extreme = rolling.max() if method == "max" else rolling.min()
    return extreme.iloc[::-1]


def _future_mean(series: Series, horizon: int) -> Series:
    future = pd.to_numeric(series, errors="coerce").shift(-1).iloc[::-1]
    return future.rolling(horizon, min_periods=horizon).mean().iloc[::-1]


def _future_volume_pressure(dataframe: DataFrame, horizon: int) -> Series:
    high = pd.to_numeric(dataframe["high"], errors="coerce")
    low = pd.to_numeric(dataframe["low"], errors="coerce")
    close = pd.to_numeric(dataframe["close"], errors="coerce")
    volume = pd.to_numeric(dataframe["volume"], errors="coerce").clip(lower=0.0)
    candle_range = (high - low).replace(0.0, np.nan)
    pressure = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
    signed_volume = pressure.fillna(0.0) * volume.fillna(0.0)
    future_signed = (
        signed_volume.shift(-1)
        .iloc[::-1]
        .rolling(horizon, min_periods=horizon)
        .sum()
        .iloc[::-1]
    )
    future_volume = (
        volume.shift(-1)
        .iloc[::-1]
        .rolling(horizon, min_periods=horizon)
        .sum()
        .iloc[::-1]
    )
    return future_signed / future_volume.replace(0.0, np.nan)


def _first_one_atr_touch_details(
    dataframe: DataFrame, horizon: int, atr: Series
) -> tuple[Series, Series]:
    """Return first-touch direction and candles-to-touch for a one-ATR move."""
    close = pd.to_numeric(dataframe["close"], errors="coerce")
    high = pd.to_numeric(dataframe["high"], errors="coerce")
    low = pd.to_numeric(dataframe["low"], errors="coerce")
    upper = close + atr
    lower = close - atr
    up_hits = pd.concat(
        [high.shift(-step).ge(upper) for step in range(1, horizon + 1)], axis=1
    ).to_numpy(dtype=bool)
    down_hits = pd.concat(
        [low.shift(-step).le(lower) for step in range(1, horizon + 1)], axis=1
    ).to_numpy(dtype=bool)
    steps = np.arange(1, horizon + 1, dtype="int16")
    up_first = np.where(up_hits, steps, horizon + 1).min(axis=1)
    down_first = np.where(down_hits, steps, horizon + 1).min(axis=1)
    direction = np.full(len(dataframe), np.nan, dtype="float64")
    direction[up_first < down_first] = 1.0
    direction[down_first < up_first] = 0.0
    touch_step = np.minimum(up_first, down_first).astype("float64")
    touch_step[touch_step > horizon] = np.nan
    invalid = atr.isna().to_numpy() | close.isna().to_numpy()
    direction[invalid] = np.nan
    touch_step[invalid] = np.nan
    return (
        pd.Series(direction, index=dataframe.index),
        pd.Series(touch_step, index=dataframe.index),
    )


def _future_extreme_step(series: Series, horizon: int, mode: str) -> Series:
    """Return the extreme offset only when the complete future window exists."""
    future = pd.concat(
        [
            pd.to_numeric(series, errors="coerce").shift(-step)
            for step in range(1, horizon + 1)
        ],
        axis=1,
    ).to_numpy(dtype="float64")
    finite = np.isfinite(future)
    safe = np.where(
        finite,
        future,
        -np.inf if mode == "max" else np.inf,
    )
    offsets = (
        np.argmax(safe, axis=1) + 1
        if mode == "max"
        else np.argmin(safe, axis=1) + 1
    ).astype("float64")
    offsets[~finite.all(axis=1)] = np.nan
    return pd.Series(offsets, index=series.index)


def _strict_before_label(first_step: Series, second_step: Series) -> Series:
    """Return a binary order label only when separate candles establish order.

    Equal step numbers mean both extremes occurred within the same candle.  OHLCV
    does not reveal their intrabar order, so those rows must remain indeterminate
    instead of being assigned to the second-first class.
    """
    observed_order = (
        first_step.notna() & second_step.notna() & first_step.ne(second_step)
    )
    return first_step.lt(second_step).astype(float).where(observed_order)


def build_sieve3_reaction_targets(dataframe: DataFrame) -> DataFrame:
    """Build trader-readable path, activity, pressure, and sequence outcomes."""
    out = dataframe.copy()
    close = pd.to_numeric(out["close"], errors="coerce").replace(0.0, np.nan)
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    volume = pd.to_numeric(out["volume"], errors="coerce").clip(lower=0.0)
    atr = _atr(out)
    atr_ratio = (atr / close).replace(0.0, np.nan)
    prior_volume = volume.rolling(24, min_periods=12).mean().replace(0.0, np.nan)
    prior_close = close.shift(1)
    true_range = pd.concat(
        ((high - low).abs(), (high - prior_close).abs(), (low - prior_close).abs()),
        axis=1,
    ).max(axis=1)
    prior_range = true_range.rolling(24, min_periods=12).mean().replace(0.0, np.nan)

    for horizon in TARGET_HORIZONS:
        future_high = _future_extreme(high, horizon, "max")
        future_low = _future_extreme(low, horizon, "min")
        future_close = close.shift(-horizon)
        upside = (future_high / close) - 1.0
        downside = 1.0 - (future_low / close)
        out[f"&-future_return_{horizon}h"] = (future_close / close) - 1.0
        out[f"&-future_upside_{horizon}h_atr"] = (upside / atr_ratio).clip(0.0, 20.0)
        out[f"&-future_downside_{horizon}h_atr"] = (downside / atr_ratio).clip(0.0, 20.0)
        out[f"&-future_upside_peak_step_{horizon}h"] = _future_extreme_step(
            high, horizon, "max"
        )
        out[f"&-future_downside_peak_step_{horizon}h"] = _future_extreme_step(
            low, horizon, "min"
        )
        out[f"&-future_path_balance_{horizon}h_atr"] = (
            (upside - downside) / atr_ratio
        ).clip(-20.0, 20.0)
        out[f"&-future_volume_ratio_{horizon}h"] = (
            _future_mean(volume, horizon) / prior_volume
        ).clip(0.0, 20.0)
        out[f"&-future_pressure_{horizon}h"] = _future_volume_pressure(out, horizon)
        out[f"&-future_volatility_ratio_{horizon}h"] = (
            _future_mean(true_range, horizon) / prior_range
        ).clip(0.0, 20.0)
        if horizon in TOUCH_HORIZONS:
            direction, touch_step = _first_one_atr_touch_details(out, horizon, atr)
            out[f"&-up_before_down_1atr_{horizon}h"] = direction
            out[f"&-first_1atr_touch_step_{horizon}h"] = touch_step
            complete_future = pd.concat(
                [
                    high.shift(-step).notna() & low.shift(-step).notna()
                    for step in range(1, horizon + 1)
                ],
                axis=1,
            ).all(axis=1) & atr.notna() & close.notna()
            out[f"&-one_atr_touch_observed_{horizon}h"] = (
                touch_step.notna().astype(float).where(complete_future)
            )
            out[f"&-first_1atr_touch_step_censored_{horizon}h"] = (
                touch_step.fillna(float(horizon + 1)).where(complete_future)
            )
    return out


def build_sieve3_exit_role_targets(dataframe: DataFrame) -> DataFrame:
    """Build side-aware post-exit path components without collapsing regret.

    The source row is the completed candle containing the observed exit event.
    Every outcome begins with the next complete candle.  This conservative
    boundary avoids treating unknown intrabar ordering in stop/ROI/custom exits
    as post-exit information.
    """
    out = dataframe.copy()
    close = pd.to_numeric(out["close"], errors="coerce").replace(0.0, np.nan)
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")

    for horizon in EXIT_ROLE_HORIZONS:
        future_close = close.shift(-horizon)
        future_high = _future_extreme(high, horizon, "max")
        future_low = _future_extreme(low, horizon, "min")
        upside = (future_high / close) - 1.0
        downside = 1.0 - (future_low / close)
        terminal_return = (future_close / close) - 1.0
        upside_step = _future_extreme_step(high, horizon, "max")
        downside_step = _future_extreme_step(low, horizon, "min")

        for side, sign in (("long", 1.0), ("short", -1.0)):
            favourable = upside if sign > 0.0 else downside
            adverse = downside if sign > 0.0 else upside
            favourable_step = upside_step if sign > 0.0 else downside_step
            adverse_step = downside_step if sign > 0.0 else upside_step
            out[f"&-exit_{side}_hold_delta_{horizon}h"] = terminal_return * sign
            out[f"&-exit_{side}_missed_profit_{horizon}h"] = favourable.clip(
                lower=0.0
            )
            out[f"&-exit_{side}_avoided_loss_{horizon}h"] = adverse.clip(
                lower=0.0
            )
            out[f"&-exit_{side}_net_regret_{horizon}h"] = (
                favourable.clip(lower=0.0) - adverse.clip(lower=0.0)
            )
            if horizon in EXIT_ROLE_SEQUENCE_HORIZONS:
                out[f"&-exit_{side}_favourable_peak_step_{horizon}h"] = (
                    favourable_step
                )
                out[f"&-exit_{side}_adverse_peak_step_{horizon}h"] = adverse_step
                out[f"&-exit_{side}_favourable_before_adverse_{horizon}h"] = (
                    _strict_before_label(favourable_step, adverse_step)
                )
    return out

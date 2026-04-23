from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from pandas import DataFrame


PriceSource = Literal["close", "hl2", "hlc3", "ohlc4"]


@dataclass(frozen=True)
class VolumeProfileConfig:
    """Settings for the rolling volume profile indicator."""

    window: int = 96
    bins: int = 48
    value_area_pct: float = 0.70
    price_source: PriceSource = "hlc3"
    smooth_bins: int = 3
    hvn_threshold: float = 0.70
    lvn_threshold: float = 0.35
    chunk_size: int = 1024
    trigger_delta_min: float = 0.05
    trigger_node_near_pct: float = 0.01
    trigger_volume_percentile_min: float = 0.55
    prefix: str = "vp"


def add_volume_profile(
    dataframe: DataFrame,
    config: VolumeProfileConfig | None = None,
    *,
    window: int | None = None,
    bins: int | None = None,
    value_area_pct: float | None = None,
    price_source: PriceSource | None = None,
    smooth_bins: int | None = None,
    hvn_threshold: float | None = None,
    lvn_threshold: float | None = None,
    chunk_size: int | None = None,
    trigger_delta_min: float | None = None,
    trigger_node_near_pct: float | None = None,
    trigger_volume_percentile_min: float | None = None,
    prefix: str | None = None,
) -> DataFrame:
    """
    Append a complex rolling volume profile to an OHLCV DataFrame.

    The profile distributes each candle's volume across price bins by candle
    range overlap. Rolling windows are processed as NumPy blocks, so the slow
    path is chunked by block instead of looping candle by candle.

    Output columns use ``prefix`` and include:
    - ``*_poc``, ``*_vah``, ``*_val``: point of control and value area.
    - ``*_hvn_above/below`` and ``*_lvn_above/below``: nearest high/low volume
      nodes around the current close.
    - ``*_delta_ratio`` and ``*_poc_delta_ratio``: directional volume pressure.
    - ``*_entropy``, ``*_concentration``, ``*_skew``, ``*_kurtosis``: profile
      shape metrics.
    - boolean state columns for value-area acceptance, rejection, and breaks
      against the prior completed profile.
    - ``*_enter_long``, ``*_enter_short``, ``*_exit_long``, and
      ``*_exit_short`` distilled trigger columns for strategy actions.
    """

    cfg = _resolve_config(
        config,
        window=window,
        bins=bins,
        value_area_pct=value_area_pct,
        price_source=price_source,
        smooth_bins=smooth_bins,
        hvn_threshold=hvn_threshold,
        lvn_threshold=lvn_threshold,
        chunk_size=chunk_size,
        trigger_delta_min=trigger_delta_min,
        trigger_node_near_pct=trigger_node_near_pct,
        trigger_volume_percentile_min=trigger_volume_percentile_min,
        prefix=prefix,
    )
    _validate_config(cfg)
    _validate_dataframe(dataframe)

    frame = dataframe.copy()
    n_rows = len(frame)
    p = cfg.prefix

    float_columns = [
        "profile_low",
        "profile_high",
        "prior_poc",
        "prior_vah",
        "prior_val",
        "poc",
        "vah",
        "val",
        "mean_price",
        "value_area_width",
        "value_area_width_pct",
        "value_area_position",
        "distance_to_poc_pct",
        "distance_to_vah_pct",
        "distance_to_val_pct",
        "close_bin_volume_share",
        "close_bin_volume_percentile",
        "delta_ratio",
        "poc_delta_ratio",
        "entropy",
        "concentration",
        "skew",
        "kurtosis",
        "hvn_above",
        "hvn_below",
        "lvn_above",
        "lvn_below",
        "nearest_hvn_distance_pct",
        "nearest_lvn_distance_pct",
        "long_trigger_score",
        "short_trigger_score",
        "signal",
        "signal_strength",
    ]
    bool_columns = [
        "in_value_area",
        "above_value_area",
        "below_value_area",
        "upper_rejection",
        "lower_rejection",
        "vah_breakout",
        "val_breakdown",
        "trigger_long_breakout",
        "trigger_long_reclaim",
        "trigger_long_acceptance",
        "trigger_long_hvn_bounce",
        "trigger_short_breakdown",
        "trigger_short_reject",
        "trigger_short_acceptance",
        "trigger_short_hvn_reject",
        "enter_long",
        "enter_short",
        "exit_long",
        "exit_short",
    ]

    for name in float_columns:
        frame[f"{p}_{name}"] = np.nan
    for name in bool_columns:
        frame[f"{p}_{name}"] = False

    if n_rows < cfg.window:
        return frame

    open_arr = _float_array(frame["open"])
    high_arr = _float_array(frame["high"])
    low_arr = _float_array(frame["low"])
    close_arr = _float_array(frame["close"])
    volume_arr = np.nan_to_num(_float_array(frame["volume"]), nan=0.0, posinf=0.0, neginf=0.0)
    source_arr = _price_source(frame, cfg.price_source)

    high_windows = sliding_window_view(high_arr, cfg.window)
    low_windows = sliding_window_view(low_arr, cfg.window)
    volume_windows = sliding_window_view(volume_arr, cfg.window)
    source_windows = sliding_window_view(source_arr, cfg.window)
    close_windows = sliding_window_view(close_arr, cfg.window)
    open_windows = sliding_window_view(open_arr, cfg.window)

    valid_rows = n_rows - cfg.window + 1
    output_index = np.arange(cfg.window - 1, n_rows)

    results: dict[str, np.ndarray] = {
        name: np.full(valid_rows, np.nan, dtype="float64") for name in float_columns
    }
    bool_results: dict[str, np.ndarray] = {
        name: np.zeros(valid_rows, dtype="bool") for name in bool_columns
    }

    for start in range(0, valid_rows, cfg.chunk_size):
        stop = min(start + cfg.chunk_size, valid_rows)
        chunk = _profile_chunk(
            open_windows[start:stop],
            high_windows[start:stop],
            low_windows[start:stop],
            close_windows[start:stop],
            volume_windows[start:stop],
            source_windows[start:stop],
            high_arr[output_index[start:stop]],
            low_arr[output_index[start:stop]],
            close_arr[output_index[start:stop]],
            cfg,
        )
        for name, values in chunk.float_values.items():
            results[name][start:stop] = values
        for name, values in chunk.bool_values.items():
            bool_results[name][start:stop] = values

    for name, values in results.items():
        frame.iloc[output_index, frame.columns.get_loc(f"{p}_{name}")] = values
    for name, values in bool_results.items():
        frame.iloc[output_index, frame.columns.get_loc(f"{p}_{name}")] = values

    frame = _add_trigger_columns(frame, cfg)
    return frame


@dataclass(frozen=True)
class _ChunkResult:
    float_values: dict[str, np.ndarray]
    bool_values: dict[str, np.ndarray]


def _profile_chunk(
    open_windows: np.ndarray,
    high_windows: np.ndarray,
    low_windows: np.ndarray,
    close_windows: np.ndarray,
    volume_windows: np.ndarray,
    source_windows: np.ndarray,
    current_high: np.ndarray,
    current_low: np.ndarray,
    current_close: np.ndarray,
    cfg: VolumeProfileConfig,
) -> _ChunkResult:
    rows = len(high_windows)
    bin_numbers = np.arange(cfg.bins, dtype="float64")
    row_numbers = np.arange(rows)
    eps = np.finfo("float64").eps

    candle_low = np.nanmin(np.stack((low_windows, high_windows), axis=0), axis=0)
    candle_high = np.nanmax(np.stack((low_windows, high_windows), axis=0), axis=0)
    candle_range = candle_high - candle_low
    finite_candle = np.isfinite(candle_low) & np.isfinite(candle_high)

    profile_low = np.nanmin(candle_low, axis=1)
    profile_high = np.nanmax(candle_high, axis=1)
    span = profile_high - profile_low
    span = np.where(np.isfinite(span) & (span > eps), span, eps)
    bin_width = span / float(cfg.bins)

    bin_low = profile_low[:, None] + bin_numbers[None, :] * bin_width[:, None]
    bin_high = bin_low + bin_width[:, None]
    bin_centers = bin_low + (bin_width[:, None] * 0.5)

    directional = _directional_volume(
        open_windows, high_windows, low_windows, close_windows, volume_windows
    )
    profile = np.zeros((rows, cfg.bins), dtype="float64")
    delta_profile = np.zeros_like(profile)

    ranged = finite_candle & (candle_range > eps)
    if np.any(ranged):
        overlap = (
            np.minimum(candle_high[:, :, None], bin_high[:, None, :])
            - np.maximum(candle_low[:, :, None], bin_low[:, None, :])
        )
        weights = np.clip(overlap, 0.0, None) / np.where(ranged, candle_range, np.nan)[:, :, None]
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        profile += np.sum(weights * volume_windows[:, :, None], axis=1)
        delta_profile += np.sum(weights * directional[:, :, None], axis=1)

    flat_degenerate = np.isfinite(source_windows) & finite_candle & (candle_range <= eps)
    if np.any(flat_degenerate):
        degenerate_bin = np.floor(
            (source_windows - profile_low[:, None]) / bin_width[:, None]
        ).astype("float64")
        degenerate_bin = np.nan_to_num(degenerate_bin, nan=0.0, posinf=cfg.bins - 1, neginf=0.0)
        degenerate_bin = np.clip(degenerate_bin.astype("int64"), 0, cfg.bins - 1)
        row_idx = np.broadcast_to(row_numbers[:, None], degenerate_bin.shape)[flat_degenerate]
        bin_idx = degenerate_bin[flat_degenerate]
        np.add.at(profile, (row_idx, bin_idx), volume_windows[flat_degenerate])
        np.add.at(delta_profile, (row_idx, bin_idx), directional[flat_degenerate])

    total_volume = profile.sum(axis=1)
    valid_profile = total_volume > eps
    poc_idx = np.argmax(profile, axis=1)
    poc = bin_centers[row_numbers, poc_idx]
    poc_volume = profile[row_numbers, poc_idx]

    value_mask = _value_area_mask(profile, total_volume, cfg.value_area_pct, poc_idx)
    val_idx = np.where(value_mask, np.arange(cfg.bins)[None, :], cfg.bins).min(axis=1)
    vah_idx = np.where(value_mask, np.arange(cfg.bins)[None, :], -1).max(axis=1)
    val_idx = np.where(valid_profile, val_idx, 0)
    vah_idx = np.where(valid_profile, vah_idx, 0)
    val = bin_low[row_numbers, val_idx]
    vah = bin_high[row_numbers, vah_idx]

    volume_prob = np.divide(
        profile,
        total_volume[:, None],
        out=np.zeros_like(profile),
        where=total_volume[:, None] > eps,
    )
    mean_price = np.sum(volume_prob * bin_centers, axis=1)
    centered = bin_centers - mean_price[:, None]
    variance = np.sum(volume_prob * centered * centered, axis=1)
    std = np.sqrt(np.maximum(variance, eps))
    skew = np.sum(volume_prob * centered**3, axis=1) / std**3
    kurtosis = (np.sum(volume_prob * centered**4, axis=1) / std**4) - 3.0

    positive_prob = np.where(volume_prob > 0.0, volume_prob, 1.0)
    entropy = -np.sum(np.where(volume_prob > 0.0, volume_prob * np.log(positive_prob), 0.0), axis=1)
    entropy = entropy / np.log(float(cfg.bins))
    concentration = np.divide(
        poc_volume, total_volume, out=np.zeros(rows), where=total_volume > eps
    )

    current_bin = np.floor((current_close - profile_low) / bin_width).astype("float64")
    current_bin = np.nan_to_num(current_bin, nan=0.0, posinf=cfg.bins - 1, neginf=0.0)
    current_bin = np.clip(current_bin.astype("int64"), 0, cfg.bins - 1)
    current_bin_volume = profile[row_numbers, current_bin]
    close_bin_volume_share = np.divide(
        current_bin_volume,
        total_volume,
        out=np.zeros(rows),
        where=total_volume > eps,
    )
    close_bin_volume_percentile = (profile <= current_bin_volume[:, None]).sum(axis=1)
    close_bin_volume_percentile = close_bin_volume_percentile / float(cfg.bins)

    value_area_width = vah - val
    value_area_width_pct = _pct(value_area_width, current_close)
    value_area_position = np.divide(
        current_close - val,
        np.where(value_area_width > eps, value_area_width, np.nan),
    )

    profile_delta = delta_profile.sum(axis=1)
    delta_ratio = np.divide(
        profile_delta, total_volume, out=np.zeros(rows), where=total_volume > eps
    )
    poc_delta = delta_profile[row_numbers, poc_idx]
    poc_delta_ratio = np.divide(poc_delta, poc_volume, out=np.zeros(rows), where=poc_volume > eps)

    smooth_profile = _smooth_profile(profile, cfg.smooth_bins)
    hvn_mask = _hvn_mask(smooth_profile, poc_volume, cfg.hvn_threshold)
    lvn_mask = _lvn_mask(smooth_profile, total_volume, cfg.lvn_threshold)
    hvn_above, hvn_below = _nearest_nodes(bin_centers, current_close, hvn_mask)
    lvn_above, lvn_below = _nearest_nodes(bin_centers, current_close, lvn_mask)
    nearest_hvn_distance_pct = _nearest_distance_pct(current_close, hvn_above, hvn_below)
    nearest_lvn_distance_pct = _nearest_distance_pct(current_close, lvn_above, lvn_below)

    in_value_area = valid_profile & (current_close >= val) & (current_close <= vah)
    above_value_area = valid_profile & (current_close > vah)
    below_value_area = valid_profile & (current_close < val)
    upper_rejection = (
        valid_profile & (current_high > vah) & (current_close < vah) & (current_close >= val)
    )
    lower_rejection = (
        valid_profile & (current_low < val) & (current_close > val) & (current_close <= vah)
    )
    vah_breakout = valid_profile & (current_close > vah) & (current_low <= vah)
    val_breakdown = valid_profile & (current_close < val) & (current_high >= val)

    float_values = {
        "profile_low": _nan_invalid(profile_low, valid_profile),
        "profile_high": _nan_invalid(profile_high, valid_profile),
        "poc": _nan_invalid(poc, valid_profile),
        "vah": _nan_invalid(vah, valid_profile),
        "val": _nan_invalid(val, valid_profile),
        "mean_price": _nan_invalid(mean_price, valid_profile),
        "value_area_width": _nan_invalid(value_area_width, valid_profile),
        "value_area_width_pct": _nan_invalid(value_area_width_pct, valid_profile),
        "value_area_position": _nan_invalid(value_area_position, valid_profile),
        "distance_to_poc_pct": _nan_invalid(
            _pct(current_close - poc, current_close), valid_profile
        ),
        "distance_to_vah_pct": _nan_invalid(
            _pct(current_close - vah, current_close), valid_profile
        ),
        "distance_to_val_pct": _nan_invalid(
            _pct(current_close - val, current_close), valid_profile
        ),
        "close_bin_volume_share": _nan_invalid(close_bin_volume_share, valid_profile),
        "close_bin_volume_percentile": _nan_invalid(close_bin_volume_percentile, valid_profile),
        "delta_ratio": _nan_invalid(delta_ratio, valid_profile),
        "poc_delta_ratio": _nan_invalid(poc_delta_ratio, valid_profile),
        "entropy": _nan_invalid(entropy, valid_profile),
        "concentration": _nan_invalid(concentration, valid_profile),
        "skew": _nan_invalid(skew, valid_profile),
        "kurtosis": _nan_invalid(kurtosis, valid_profile),
        "hvn_above": _nan_invalid(hvn_above, valid_profile),
        "hvn_below": _nan_invalid(hvn_below, valid_profile),
        "lvn_above": _nan_invalid(lvn_above, valid_profile),
        "lvn_below": _nan_invalid(lvn_below, valid_profile),
        "nearest_hvn_distance_pct": _nan_invalid(nearest_hvn_distance_pct, valid_profile),
        "nearest_lvn_distance_pct": _nan_invalid(nearest_lvn_distance_pct, valid_profile),
    }
    bool_values = {
        "in_value_area": in_value_area,
        "above_value_area": above_value_area,
        "below_value_area": below_value_area,
        "upper_rejection": upper_rejection,
        "lower_rejection": lower_rejection,
        "vah_breakout": vah_breakout,
        "val_breakdown": val_breakdown,
    }
    return _ChunkResult(float_values=float_values, bool_values=bool_values)


def _resolve_config(
    config: VolumeProfileConfig | None,
    **overrides: object,
) -> VolumeProfileConfig:
    cfg = config or VolumeProfileConfig()
    values = {
        "window": cfg.window,
        "bins": cfg.bins,
        "value_area_pct": cfg.value_area_pct,
        "price_source": cfg.price_source,
        "smooth_bins": cfg.smooth_bins,
        "hvn_threshold": cfg.hvn_threshold,
        "lvn_threshold": cfg.lvn_threshold,
        "chunk_size": cfg.chunk_size,
        "trigger_delta_min": cfg.trigger_delta_min,
        "trigger_node_near_pct": cfg.trigger_node_near_pct,
        "trigger_volume_percentile_min": cfg.trigger_volume_percentile_min,
        "prefix": cfg.prefix,
    }
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return VolumeProfileConfig(**values)


def _validate_config(cfg: VolumeProfileConfig) -> None:
    if cfg.window < 2:
        raise ValueError("window must be at least 2")
    if cfg.bins < 4:
        raise ValueError("bins must be at least 4")
    if not 0.05 <= cfg.value_area_pct <= 0.99:
        raise ValueError("value_area_pct must be between 0.05 and 0.99")
    if cfg.smooth_bins < 1:
        raise ValueError("smooth_bins must be at least 1")
    if cfg.chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if not 0.0 <= cfg.trigger_delta_min <= 1.0:
        raise ValueError("trigger_delta_min must be between 0.0 and 1.0")
    if cfg.trigger_node_near_pct < 0.0:
        raise ValueError("trigger_node_near_pct must be non-negative")
    if not 0.0 <= cfg.trigger_volume_percentile_min <= 1.0:
        raise ValueError("trigger_volume_percentile_min must be between 0.0 and 1.0")
    if not cfg.prefix:
        raise ValueError("prefix must not be empty")


def _validate_dataframe(dataframe: DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"DataFrame is missing OHLCV columns: {missing}")


def _add_trigger_columns(frame: DataFrame, cfg: VolumeProfileConfig) -> DataFrame:
    p = cfg.prefix
    close = _numeric_series(frame, "close")
    high = _numeric_series(frame, "high")
    low = _numeric_series(frame, "low")
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    prior_poc = _numeric_series(frame, f"{p}_poc").shift(1)
    prior_vah = _numeric_series(frame, f"{p}_vah").shift(1)
    prior_val = _numeric_series(frame, f"{p}_val").shift(1)
    valid_prior = prior_poc.notna() & prior_vah.notna() & prior_val.notna()

    delta = _numeric_series(frame, f"{p}_delta_ratio").fillna(0.0)
    poc_delta = _numeric_series(frame, f"{p}_poc_delta_ratio").fillna(0.0)
    volume_percentile = _numeric_series(frame, f"{p}_close_bin_volume_percentile").fillna(0.0)
    hvn_above = _numeric_series(frame, f"{p}_hvn_above")
    hvn_below = _numeric_series(frame, f"{p}_hvn_below")

    bull_pressure = delta >= cfg.trigger_delta_min
    bear_pressure = delta <= -cfg.trigger_delta_min
    poc_bull = poc_delta >= 0.0
    poc_bear = poc_delta <= 0.0
    volume_ok = volume_percentile >= cfg.trigger_volume_percentile_min

    in_value_area = valid_prior & close.ge(prior_val) & close.le(prior_vah)
    above_value_area = valid_prior & close.gt(prior_vah)
    below_value_area = valid_prior & close.lt(prior_val)

    upper_rejection = (
        valid_prior & high.gt(prior_vah) & close.lt(prior_vah) & close.ge(prior_val)
    )
    lower_rejection = (
        valid_prior & low.lt(prior_val) & close.gt(prior_val) & close.le(prior_vah)
    )
    vah_breakout = valid_prior & close.gt(prior_vah) & (
        low.le(prior_vah) | prev_close.le(prior_vah)
    )
    val_breakdown = valid_prior & close.lt(prior_val) & (
        high.ge(prior_val) | prev_close.ge(prior_val)
    )

    near_hvn_below = (
        hvn_below.notna()
        & close.notna()
        & (_pct_series(close - hvn_below, close).abs() <= cfg.trigger_node_near_pct)
    )
    near_hvn_above = (
        hvn_above.notna()
        & close.notna()
        & (_pct_series(hvn_above - close, close).abs() <= cfg.trigger_node_near_pct)
    )

    trigger_long_breakout = vah_breakout & bull_pressure & (poc_bull | volume_ok)
    trigger_short_breakdown = val_breakdown & bear_pressure & (poc_bear | volume_ok)
    trigger_long_reclaim = lower_rejection & bull_pressure
    trigger_short_reject = upper_rejection & bear_pressure
    trigger_long_acceptance = (
        above_value_area & prev_close.gt(prior_vah) & bull_pressure & volume_ok
    )
    trigger_short_acceptance = (
        below_value_area & prev_close.lt(prior_val) & bear_pressure & volume_ok
    )
    trigger_long_hvn_bounce = (
        near_hvn_below
        & low.le(hvn_below)
        & prev_low.gt(hvn_below)
        & close.gt(hvn_below)
        & bull_pressure
    )
    trigger_short_hvn_reject = (
        near_hvn_above
        & high.ge(hvn_above)
        & prev_high.lt(hvn_above)
        & close.lt(hvn_above)
        & bear_pressure
    )

    long_base = (
        trigger_long_breakout
        | trigger_long_reclaim
        | trigger_long_acceptance
        | trigger_long_hvn_bounce
    )
    short_base = (
        trigger_short_breakdown
        | trigger_short_reject
        | trigger_short_acceptance
        | trigger_short_hvn_reject
    )
    long_score = (
        trigger_long_breakout.astype("int8") * 3
        + trigger_long_reclaim.astype("int8") * 2
        + trigger_long_acceptance.astype("int8") * 2
        + trigger_long_hvn_bounce.astype("int8")
        + (long_base & poc_bull).astype("int8")
        + (long_base & volume_ok).astype("int8")
    )
    short_score = (
        trigger_short_breakdown.astype("int8") * 3
        + trigger_short_reject.astype("int8") * 2
        + trigger_short_acceptance.astype("int8") * 2
        + trigger_short_hvn_reject.astype("int8")
        + (short_base & poc_bear).astype("int8")
        + (short_base & volume_ok).astype("int8")
    )
    enter_long = long_base & (long_score > short_score)
    enter_short = short_base & (short_score > long_score)
    exit_long = short_base | (valid_prior & close.lt(prior_poc) & bear_pressure)
    exit_short = long_base | (valid_prior & close.gt(prior_poc) & bull_pressure)

    frame[f"{p}_prior_poc"] = prior_poc
    frame[f"{p}_prior_vah"] = prior_vah
    frame[f"{p}_prior_val"] = prior_val
    frame[f"{p}_in_value_area"] = in_value_area
    frame[f"{p}_above_value_area"] = above_value_area
    frame[f"{p}_below_value_area"] = below_value_area
    frame[f"{p}_upper_rejection"] = upper_rejection
    frame[f"{p}_lower_rejection"] = lower_rejection
    frame[f"{p}_vah_breakout"] = vah_breakout
    frame[f"{p}_val_breakdown"] = val_breakdown
    frame[f"{p}_trigger_long_breakout"] = trigger_long_breakout
    frame[f"{p}_trigger_long_reclaim"] = trigger_long_reclaim
    frame[f"{p}_trigger_long_acceptance"] = trigger_long_acceptance
    frame[f"{p}_trigger_long_hvn_bounce"] = trigger_long_hvn_bounce
    frame[f"{p}_trigger_short_breakdown"] = trigger_short_breakdown
    frame[f"{p}_trigger_short_reject"] = trigger_short_reject
    frame[f"{p}_trigger_short_acceptance"] = trigger_short_acceptance
    frame[f"{p}_trigger_short_hvn_reject"] = trigger_short_hvn_reject
    frame[f"{p}_enter_long"] = enter_long
    frame[f"{p}_enter_short"] = enter_short
    frame[f"{p}_exit_long"] = exit_long
    frame[f"{p}_exit_short"] = exit_short
    frame[f"{p}_long_trigger_score"] = long_score.astype("float64")
    frame[f"{p}_short_trigger_score"] = short_score.astype("float64")
    frame[f"{p}_signal"] = np.select([enter_long, enter_short], [1.0, -1.0], default=0.0)
    frame[f"{p}_signal_strength"] = np.abs(long_score - short_score).astype("float64")
    return frame


def _numeric_series(frame: DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _float_array(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")


def _price_source(frame: DataFrame, price_source: PriceSource) -> np.ndarray:
    open_arr = _float_array(frame["open"])
    high_arr = _float_array(frame["high"])
    low_arr = _float_array(frame["low"])
    close_arr = _float_array(frame["close"])
    if price_source == "close":
        return close_arr
    if price_source == "hl2":
        return (high_arr + low_arr) / 2.0
    if price_source == "hlc3":
        return (high_arr + low_arr + close_arr) / 3.0
    if price_source == "ohlc4":
        return (open_arr + high_arr + low_arr + close_arr) / 4.0
    raise ValueError(f"Unsupported price_source: {price_source}")


def _directional_volume(
    open_windows: np.ndarray,
    high_windows: np.ndarray,
    low_windows: np.ndarray,
    close_windows: np.ndarray,
    volume_windows: np.ndarray,
) -> np.ndarray:
    candle_range = high_windows - low_windows
    body_pressure = np.divide(
        close_windows - open_windows,
        np.where(np.abs(candle_range) > 0.0, candle_range, np.nan),
    )
    body_pressure = np.clip(np.nan_to_num(body_pressure, nan=0.0), -1.0, 1.0)
    return volume_windows * body_pressure


def _value_area_mask(
    profile: np.ndarray,
    total_volume: np.ndarray,
    value_area_pct: float,
    poc_idx: np.ndarray,
) -> np.ndarray:
    rows, bins = profile.shape
    target = total_volume[:, None] * value_area_pct
    valid = total_volume > 0.0
    row_numbers = np.arange(rows)
    selected = np.zeros((rows, bins), dtype="bool")
    selected[row_numbers[valid], poc_idx[valid]] = True

    left = poc_idx.copy()
    right = poc_idx.copy()
    selected_volume = profile[row_numbers, poc_idx].copy()
    target = target[:, 0]

    for _ in range(bins - 1):
        need_more = valid & (selected_volume < target)
        if not np.any(need_more):
            break

        left_candidate = left - 1
        right_candidate = right + 1
        has_left = left_candidate >= 0
        has_right = right_candidate < bins
        left_idx = np.clip(left_candidate, 0, bins - 1)
        right_idx = np.clip(right_candidate, 0, bins - 1)
        left_volume = np.where(has_left, profile[row_numbers, left_idx], -np.inf)
        right_volume = np.where(has_right, profile[row_numbers, right_idx], -np.inf)

        take_left = need_more & has_left & (~has_right | (left_volume >= right_volume))
        take_right = need_more & has_right & ~take_left

        left = np.where(take_left, left_candidate, left)
        right = np.where(take_right, right_candidate, right)
        selected[row_numbers[take_left], left[take_left]] = True
        selected[row_numbers[take_right], right[take_right]] = True
        selected_volume += np.where(take_left, left_volume, 0.0)
        selected_volume += np.where(take_right, right_volume, 0.0)

    return selected


def _smooth_profile(profile: np.ndarray, smooth_bins: int) -> np.ndarray:
    width = int(smooth_bins)
    if width <= 1:
        return profile
    if width % 2 == 0:
        width += 1
    pad = width // 2
    padded = np.pad(profile, ((0, 0), (pad, pad)), mode="edge")
    cumulative = np.pad(np.cumsum(padded, axis=1), ((0, 0), (1, 0)), mode="constant")
    return (cumulative[:, width:] - cumulative[:, :-width]) / float(width)


def _hvn_mask(smooth_profile: np.ndarray, poc_volume: np.ndarray, threshold: float) -> np.ndarray:
    left = np.full_like(smooth_profile, -np.inf)
    right = np.full_like(smooth_profile, -np.inf)
    left[:, 1:] = smooth_profile[:, :-1]
    right[:, :-1] = smooth_profile[:, 1:]
    return (
        (smooth_profile >= left)
        & (smooth_profile > right)
        & (smooth_profile >= (poc_volume[:, None] * float(threshold)))
    )


def _lvn_mask(smooth_profile: np.ndarray, total_volume: np.ndarray, threshold: float) -> np.ndarray:
    left = np.full_like(smooth_profile, np.inf)
    right = np.full_like(smooth_profile, np.inf)
    left[:, 1:] = smooth_profile[:, :-1]
    right[:, :-1] = smooth_profile[:, 1:]
    mean_bin_volume = total_volume[:, None] / float(smooth_profile.shape[1])
    return (
        (smooth_profile <= left)
        & (smooth_profile < right)
        & (smooth_profile <= (mean_bin_volume * float(threshold)))
    )


def _nearest_nodes(
    bin_centers: np.ndarray,
    current_close: np.ndarray,
    node_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    above_distance = np.where(
        node_mask & (bin_centers > current_close[:, None]),
        bin_centers - current_close[:, None],
        np.inf,
    )
    below_distance = np.where(
        node_mask & (bin_centers < current_close[:, None]),
        current_close[:, None] - bin_centers,
        np.inf,
    )
    above_idx = np.argmin(above_distance, axis=1)
    below_idx = np.argmin(below_distance, axis=1)
    has_above = np.isfinite(above_distance[np.arange(len(current_close)), above_idx])
    has_below = np.isfinite(below_distance[np.arange(len(current_close)), below_idx])
    above = np.where(has_above, bin_centers[np.arange(len(current_close)), above_idx], np.nan)
    below = np.where(has_below, bin_centers[np.arange(len(current_close)), below_idx], np.nan)
    return above, below


def _nearest_distance_pct(
    current_close: np.ndarray,
    above: np.ndarray,
    below: np.ndarray,
) -> np.ndarray:
    above_distance = np.abs(_pct(above - current_close, current_close))
    below_distance = np.abs(_pct(current_close - below, current_close))
    distances = np.stack((above_distance, below_distance), axis=0)
    finite = np.isfinite(distances)
    return np.where(finite.any(axis=0), np.min(np.where(finite, distances, np.inf), axis=0), np.nan)


def _pct(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    denominator = np.where(np.abs(denominator) > 0.0, denominator, np.nan)
    return numerator / denominator


def _pct_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.where(denominator.abs() > 0.0, np.nan)
    return numerator / denominator


def _nan_invalid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.where(valid & np.isfinite(values), values, np.nan)

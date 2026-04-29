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
    """Settings for the rolling volume profile indicator.

    This module is intended for Freqtrade strategies and hyperopt. It builds a
    rolling price/volume distribution, then emits raw profile evidence plus
    normalized 0..1 score columns. The scores summarize profile behavior only;
    final entries/exits remain in the strategy.

    Score meaning:
    - ``*_score_long`` rises when value is migrating higher, price accepts above
      prior value, failed downside auctions reclaim VAL/HVN/LVN zones, or price
      traverses thin LVN liquidity upward with pressure.
    - ``*_score_short`` is the mirrored bearish evidence: lower POC migration,
      acceptance below value, failed upside auctions, HVN/LVN rejection, or thin
      LVN traversal downward.
    - ``*_score_abs`` is max(long, short). It is profile evidence strength, not
      direction.
    - ``*_state`` is -1/0/1 directional lean and is not a normalized score.

    Practical interpretation:
    POC migration measures where the market is accepting value over time. VAH
    and VAL events measure value-area acceptance or rejection. HVNs represent
    high-liquidity memory zones; LVNs represent thin areas where price often
    rejects sharply or moves through quickly. A high score means the current
    candle is interacting with those profile levels in a way worth testing
    against forward returns.

    Tunable groups:
    - Profile construction:
      ``window`` controls how many candles form each rolling profile.
      ``bins`` controls price-resolution of the volume histogram.
      ``value_area_pct`` controls how much profile volume is captured between
      VAL and VAH. ``price_source`` controls which candle price anchors the
      profile. ``smooth_bins`` reduces noisy one-bin HVN/LVN artifacts.
    - Node detection:
      ``hvn_threshold`` defines how strong a bin must be relative to POC to
      count as an HVN. ``lvn_threshold`` defines how low-volume a bin must be
      to count as an LVN. ``node_hvn_strength_min`` and
      ``node_lvn_thinness_min`` filter raw nearby nodes before they become
      actionable evidence. ``node_near_pct`` is the base distance tolerance
      around profile levels and nodes. ``node_hold_near_mult`` widens the
      node-distance tolerance only for hold evidence.
    - Pressure and participation:
      ``pressure_delta_min`` controls how much directional candle-volume
      pressure is required. ``volume_percentile_min`` controls whether the
      current close-bin has enough participation. ``fast_traverse_atr_mult``
      controls how large a candle must be, relative to ATR, to qualify as a
      fast LVN traverse.
    - Scoring and context:
      ``poc_migration_window`` controls how far back POC/value direction is
      compared. ``score_window`` controls recent event memory and de-duplication
      cooldown. ``entry_score_margin`` requires long/short score separation
      before a trigger is emitted. ``context_*`` values control when the
      smoothed profile context becomes full bull/bear or softer directional
      chop.
    """

    window: int = 96
    bins: int = 48
    value_area_pct: float = 0.70
    price_source: PriceSource = "hlc3"
    smooth_bins: int = 3
    hvn_threshold: float = 0.70
    lvn_threshold: float = 0.35
    chunk_size: int = 1024
    pressure_delta_min: float = 0.05
    node_near_pct: float = 0.01
    node_hvn_strength_min: float = 0.70
    node_lvn_thinness_min: float = 0.55
    node_hold_near_mult: float = 3.00
    volume_percentile_min: float = 0.55
    poc_migration_window: int = 12
    score_window: int = 48
    fast_traverse_atr_mult: float = 1.20
    entry_score_margin: float = 0.02
    context_full_min: float = 0.44
    context_full_margin: float = 0.06
    context_soft_min: float = 0.26
    context_soft_margin: float = 0.035
    context_balance_min: float = 0.42
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
    pressure_delta_min: float | None = None,
    node_near_pct: float | None = None,
    node_hvn_strength_min: float | None = None,
    node_lvn_thinness_min: float | None = None,
    node_hold_near_mult: float | None = None,
    volume_percentile_min: float | None = None,
    poc_migration_window: int | None = None,
    score_window: int | None = None,
    fast_traverse_atr_mult: float | None = None,
    entry_score_margin: float | None = None,
    context_full_min: float | None = None,
    context_full_margin: float | None = None,
    context_soft_min: float | None = None,
    context_soft_margin: float | None = None,
    context_balance_min: float | None = None,
    prefix: str | None = None,
) -> DataFrame:
    """
    Append a complex rolling volume profile to an OHLCV DataFrame.

    The profile distributes each candle's volume across price bins by candle
    range overlap. Rolling windows are processed as NumPy blocks, so the slow
    path is chunked by block instead of looping candle by candle.

    Output columns use ``prefix`` and include:
    - Profile levels:
      ``*_poc`` is the highest-volume price bin in the rolling profile.
      ``*_vah`` and ``*_val`` are value-area high/low. ``*_prior_*`` columns
      are shifted one candle and should be preferred for entry logic to avoid
      using the current candle's completed profile as its own trigger.
    - Raw node levels and diagnostics:
      ``*_hvn_above/below`` and ``*_lvn_above/below`` are nearest high/low
      volume nodes around the current close. ``*_hvn_*_strength`` and
      ``*_lvn_*_thinness`` describe node quality. Distance columns describe how
      far the current close is from those nodes.
    - Profile shape:
      ``*_entropy``, ``*_concentration``, ``*_skew``, and ``*_kurtosis`` are
      diagnostics for whether the profile is balanced, concentrated, or
      asymmetric. They are useful as guards, not direct entry triggers.
    - Pressure and value movement:
      ``*_delta_ratio`` estimates candle directional volume pressure.
      ``*_poc_delta_ratio`` estimates pressure around the POC bin.
      ``*_poc_migration_pct`` tracks POC movement over
      ``poc_migration_window``. ``*_value_direction_pct`` tracks movement of
      the VAH/VAL midpoint and is named as direction because that is the useful
      interpretation for strategies.
    - Event evidence:
      ``*_vah_breakout_with_pressure`` and ``*_val_breakdown_with_pressure``
      mark value-area breaks with pressure. ``*_lower_rejection_with_pressure``
      and ``*_upper_rejection_with_pressure`` mark failed auctions back into
      value. LVN/HVN accept, reject, reclaim, and fast-traverse columns expose
      the raw event components used by the trigger score.
    - Strategy-facing triggers:
      ``*_entry_trigger_long`` and ``*_entry_trigger_short`` are de-duplicated
      entry-trigger evidence. They are intentionally not final trade decisions;
      strategies should still apply their own timeframe, regime, stake, and
      risk logic.
    - Node action evidence:
      ``*_node_entry_long/short`` isolate trigger evidence that specifically
      comes from HVN/LVN interaction. ``*_node_hold_long/short`` says profile
      structure still supports an existing position in that direction.
      ``*_node_exit_long/short`` says opposing node/rejection evidence is strong
      enough that an existing position should consider exit or reduction. These
      are evidence flags only, not target or stop instructions.
    - Context and scores:
      ``*_score_long``, ``*_score_short``, and ``*_score_abs`` are normalized
      0..1 evidence scores. ``*_state`` is a simple -1/0/1 score lean.
      ``*_context_score_bull``, ``*_context_score_bear``, and
      ``*_context_score_balance`` are smoothed context components.
      ``*_market_context`` is the strategy-facing directional state:
      ``2`` bull, ``1`` bullish chop, ``0`` undefined, ``-1`` bearish chop,
      and ``-2`` bear.
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
        pressure_delta_min=pressure_delta_min,
        node_near_pct=node_near_pct,
        node_hvn_strength_min=node_hvn_strength_min,
        node_lvn_thinness_min=node_lvn_thinness_min,
        node_hold_near_mult=node_hold_near_mult,
        volume_percentile_min=volume_percentile_min,
        poc_migration_window=poc_migration_window,
        score_window=score_window,
        fast_traverse_atr_mult=fast_traverse_atr_mult,
        entry_score_margin=entry_score_margin,
        context_full_min=context_full_min,
        context_full_margin=context_full_margin,
        context_soft_min=context_soft_min,
        context_soft_margin=context_soft_margin,
        context_balance_min=context_balance_min,
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
        "close_bin_volume_rank",
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
        "hvn_above_strength",
        "hvn_below_strength",
        "lvn_above_thinness",
        "lvn_below_thinness",
        "hvn_above_distance_pct",
        "hvn_below_distance_pct",
        "lvn_above_distance_pct",
        "lvn_below_distance_pct",
        "nearest_hvn_distance_pct",
        "nearest_lvn_distance_pct",
    ]
    bool_columns = [
        "in_value_area",
        "above_value_area",
        "below_value_area",
        "upper_rejection",
        "lower_rejection",
        "vah_breakout",
        "val_breakdown",
        "bull_pressure",
        "bear_pressure",
        "poc_delta_bull",
        "poc_delta_bear",
        "close_bin_volume_ok",
        "near_hvn_below",
        "near_hvn_above",
        "hvn_below_reclaim",
        "hvn_above_reject",
        "vah_breakout_with_pressure",
        "val_breakdown_with_pressure",
        "lower_rejection_with_pressure",
        "upper_rejection_with_pressure",
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

    frame = _add_interaction_columns(frame, cfg)
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
    hvn_score = np.divide(
        smooth_profile,
        np.where(poc_volume[:, None] > eps, poc_volume[:, None], np.nan),
    )
    mean_bin_volume = np.divide(
        total_volume,
        float(cfg.bins),
        out=np.zeros(rows),
        where=total_volume > eps,
    )
    lvn_score = 1.0 - np.divide(
        smooth_profile,
        np.where(mean_bin_volume[:, None] > eps, mean_bin_volume[:, None], np.nan),
    )
    lvn_score = np.clip(np.nan_to_num(lvn_score, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    hvn_above, hvn_below, hvn_above_strength, hvn_below_strength = _nearest_nodes(
        bin_centers,
        current_close,
        hvn_mask,
        hvn_score,
    )
    lvn_above, lvn_below, lvn_above_thinness, lvn_below_thinness = _nearest_nodes(
        bin_centers,
        current_close,
        lvn_mask,
        lvn_score,
    )
    hvn_above_distance_pct = np.abs(_pct(hvn_above - current_close, current_close))
    hvn_below_distance_pct = np.abs(_pct(current_close - hvn_below, current_close))
    lvn_above_distance_pct = np.abs(_pct(lvn_above - current_close, current_close))
    lvn_below_distance_pct = np.abs(_pct(current_close - lvn_below, current_close))
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
        "close_bin_volume_rank": _nan_invalid(close_bin_volume_percentile, valid_profile),
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
        "hvn_above_strength": _nan_invalid(hvn_above_strength, valid_profile),
        "hvn_below_strength": _nan_invalid(hvn_below_strength, valid_profile),
        "lvn_above_thinness": _nan_invalid(lvn_above_thinness, valid_profile),
        "lvn_below_thinness": _nan_invalid(lvn_below_thinness, valid_profile),
        "hvn_above_distance_pct": _nan_invalid(hvn_above_distance_pct, valid_profile),
        "hvn_below_distance_pct": _nan_invalid(hvn_below_distance_pct, valid_profile),
        "lvn_above_distance_pct": _nan_invalid(lvn_above_distance_pct, valid_profile),
        "lvn_below_distance_pct": _nan_invalid(lvn_below_distance_pct, valid_profile),
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
        "pressure_delta_min": cfg.pressure_delta_min,
        "node_near_pct": cfg.node_near_pct,
        "node_hvn_strength_min": cfg.node_hvn_strength_min,
        "node_lvn_thinness_min": cfg.node_lvn_thinness_min,
        "node_hold_near_mult": cfg.node_hold_near_mult,
        "volume_percentile_min": cfg.volume_percentile_min,
        "poc_migration_window": cfg.poc_migration_window,
        "score_window": cfg.score_window,
        "fast_traverse_atr_mult": cfg.fast_traverse_atr_mult,
        "entry_score_margin": cfg.entry_score_margin,
        "context_full_min": cfg.context_full_min,
        "context_full_margin": cfg.context_full_margin,
        "context_soft_min": cfg.context_soft_min,
        "context_soft_margin": cfg.context_soft_margin,
        "context_balance_min": cfg.context_balance_min,
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
    if not 0.0 <= cfg.pressure_delta_min <= 1.0:
        raise ValueError("pressure_delta_min must be between 0.0 and 1.0")
    if cfg.node_near_pct < 0.0:
        raise ValueError("node_near_pct must be non-negative")
    if not 0.0 <= cfg.node_hvn_strength_min <= 1.0:
        raise ValueError("node_hvn_strength_min must be between 0.0 and 1.0")
    if not 0.0 <= cfg.node_lvn_thinness_min <= 1.0:
        raise ValueError("node_lvn_thinness_min must be between 0.0 and 1.0")
    if cfg.node_hold_near_mult <= 0.0:
        raise ValueError("node_hold_near_mult must be positive")
    if not 0.0 <= cfg.volume_percentile_min <= 1.0:
        raise ValueError("volume_percentile_min must be between 0.0 and 1.0")
    if cfg.poc_migration_window < 1:
        raise ValueError("poc_migration_window must be at least 1")
    if cfg.score_window < 2:
        raise ValueError("score_window must be at least 2")
    if cfg.fast_traverse_atr_mult <= 0.0:
        raise ValueError("fast_traverse_atr_mult must be positive")
    if cfg.entry_score_margin < 0.0:
        raise ValueError("entry_score_margin must be non-negative")
    if not 0.0 <= cfg.context_full_min <= 1.0:
        raise ValueError("context_full_min must be between 0.0 and 1.0")
    if cfg.context_full_margin < 0.0:
        raise ValueError("context_full_margin must be non-negative")
    if not 0.0 <= cfg.context_soft_min <= 1.0:
        raise ValueError("context_soft_min must be between 0.0 and 1.0")
    if cfg.context_soft_margin < 0.0:
        raise ValueError("context_soft_margin must be non-negative")
    if not 0.0 <= cfg.context_balance_min <= 1.0:
        raise ValueError("context_balance_min must be between 0.0 and 1.0")
    if not cfg.prefix:
        raise ValueError("prefix must not be empty")


def _validate_dataframe(dataframe: DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"DataFrame is missing OHLCV columns: {missing}")


def _add_interaction_columns(frame: DataFrame, cfg: VolumeProfileConfig) -> DataFrame:
    p = cfg.prefix
    close = _numeric_series(frame, "close")
    high = _numeric_series(frame, "high")
    low = _numeric_series(frame, "low")
    open_ = _numeric_series(frame, "open")
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    atr = _atr(frame, max(14, min(int(cfg.window), 96)))

    poc = _numeric_series(frame, f"{p}_poc")
    vah = _numeric_series(frame, f"{p}_vah")
    val = _numeric_series(frame, f"{p}_val")
    prior_poc = _numeric_series(frame, f"{p}_poc").shift(1)
    prior_vah = _numeric_series(frame, f"{p}_vah").shift(1)
    prior_val = _numeric_series(frame, f"{p}_val").shift(1)
    valid_prior = prior_poc.notna() & prior_vah.notna() & prior_val.notna()

    delta = _numeric_series(frame, f"{p}_delta_ratio").fillna(0.0)
    poc_delta = _numeric_series(frame, f"{p}_poc_delta_ratio").fillna(0.0)
    volume_percentile = _numeric_series(frame, f"{p}_close_bin_volume_percentile").fillna(0.0)
    hvn_above = _numeric_series(frame, f"{p}_hvn_above")
    hvn_below = _numeric_series(frame, f"{p}_hvn_below")
    lvn_above = _numeric_series(frame, f"{p}_lvn_above")
    lvn_below = _numeric_series(frame, f"{p}_lvn_below")
    hvn_above_strength = _numeric_series(frame, f"{p}_hvn_above_strength").fillna(0.0)
    hvn_below_strength = _numeric_series(frame, f"{p}_hvn_below_strength").fillna(0.0)
    lvn_above_thinness = _numeric_series(frame, f"{p}_lvn_above_thinness").fillna(0.0)
    lvn_below_thinness = _numeric_series(frame, f"{p}_lvn_below_thinness").fillna(0.0)
    prior_lvn_above = lvn_above.shift(1)
    prior_lvn_below = lvn_below.shift(1)
    prior_lvn_above_thinness = lvn_above_thinness.shift(1).fillna(0.0)
    prior_lvn_below_thinness = lvn_below_thinness.shift(1).fillna(0.0)

    bull_pressure = delta >= cfg.pressure_delta_min
    bear_pressure = delta <= -cfg.pressure_delta_min
    poc_bull = poc_delta >= 0.0
    poc_bear = poc_delta <= 0.0
    volume_ok = volume_percentile >= cfg.volume_percentile_min

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

    atr_pct = _pct_series(atr, close).abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    base_near_pct = pd.Series(float(cfg.node_near_pct), index=frame.index)
    level_tolerance_pct = pd.concat([base_near_pct, atr_pct * 0.75], axis=1).max(axis=1).clip(0.002, 0.035)
    node_tolerance_pct = pd.concat([base_near_pct, atr_pct * 0.60], axis=1).max(axis=1).clip(0.002, 0.025)
    entry_extension_pct = (level_tolerance_pct * 1.75).clip(0.004, 0.045)
    close_above_vah_pct = _pct_series(close - prior_vah, close)
    close_below_val_pct = _pct_series(prior_val - close, close)
    close_above_lvn_pct = _pct_series(close - prior_lvn_above, close)
    close_below_lvn_pct = _pct_series(prior_lvn_below - close, close)
    near_prior_poc = valid_prior & (_pct_series(close - prior_poc, close).abs() <= level_tolerance_pct)
    not_far_above_vah = valid_prior & close_above_vah_pct.le(entry_extension_pct)
    not_far_below_val = valid_prior & close_below_val_pct.le(entry_extension_pct)
    not_far_above_lvn = prior_lvn_above.notna() & close_above_lvn_pct.le(entry_extension_pct)
    not_far_below_lvn = prior_lvn_below.notna() & close_below_lvn_pct.le(entry_extension_pct)

    near_hvn_below = (
        hvn_below.notna()
        & close.notna()
        & (_pct_series(close - hvn_below, close).abs() <= node_tolerance_pct)
    )
    near_hvn_above = (
        hvn_above.notna()
        & close.notna()
        & (_pct_series(hvn_above - close, close).abs() <= node_tolerance_pct)
    )
    hvn_strength_min = max(float(cfg.hvn_threshold), float(cfg.node_hvn_strength_min))
    lvn_thinness_min = float(cfg.node_lvn_thinness_min)
    actionable_hvn_below = near_hvn_below & hvn_below_strength.ge(hvn_strength_min)
    actionable_hvn_above = near_hvn_above & hvn_above_strength.ge(hvn_strength_min)

    vah_breakout_with_pressure = vah_breakout & bull_pressure & (poc_bull | volume_ok) & not_far_above_vah
    val_breakdown_with_pressure = val_breakdown & bear_pressure & (poc_bear | volume_ok) & not_far_below_val
    lower_rejection_with_pressure = lower_rejection & bull_pressure
    upper_rejection_with_pressure = upper_rejection & bear_pressure
    hvn_below_reclaim = (
        actionable_hvn_below
        & low.le(hvn_below)
        & close.gt(hvn_below)
        & (prev_close.le(hvn_below) | prev_low.le(hvn_below) | near_hvn_below.shift(1, fill_value=False).astype("bool"))
        & bull_pressure
    )
    hvn_above_reject = (
        actionable_hvn_above
        & high.ge(hvn_above)
        & close.lt(hvn_above)
        & (prev_close.ge(hvn_above) | prev_high.ge(hvn_above) | near_hvn_above.shift(1, fill_value=False).astype("bool"))
        & bear_pressure
    )
    near_lvn_below = (
        lvn_below.notna()
        & close.notna()
        & (_pct_series(close - lvn_below, close).abs() <= node_tolerance_pct)
    )
    near_lvn_above = (
        lvn_above.notna()
        & close.notna()
        & (_pct_series(lvn_above - close, close).abs() <= node_tolerance_pct)
    )
    actionable_lvn_below = near_lvn_below & lvn_below_thinness.ge(lvn_thinness_min)
    actionable_lvn_above = near_lvn_above & lvn_above_thinness.ge(lvn_thinness_min)
    lvn_below_reject_long = (
        actionable_lvn_below
        & low.le(lvn_below)
        & close.gt(lvn_below)
        & (prev_close.le(lvn_below) | prev_low.le(lvn_below) | near_lvn_below.shift(1, fill_value=False).astype("bool"))
        & bull_pressure
    )
    lvn_above_reject_short = (
        actionable_lvn_above
        & high.ge(lvn_above)
        & close.lt(lvn_above)
        & (prev_close.ge(lvn_above) | prev_high.ge(lvn_above) | near_lvn_above.shift(1, fill_value=False).astype("bool"))
        & bear_pressure
    )
    lvn_accept_long = (
        prior_lvn_above.notna()
        & close.gt(prior_lvn_above)
        & prev_close.le(prior_lvn_above)
        & bull_pressure
        & not_far_above_lvn
        & prior_lvn_above_thinness.ge(lvn_thinness_min)
    )
    lvn_accept_short = (
        prior_lvn_below.notna()
        & close.lt(prior_lvn_below)
        & prev_close.ge(prior_lvn_below)
        & bear_pressure
        & not_far_below_lvn
        & prior_lvn_below_thinness.ge(lvn_thinness_min)
    )
    wide_range = _pct_series((high - low).abs(), atr).abs() >= float(cfg.fast_traverse_atr_mult)
    lvn_fast_traverse_long = (
        prior_lvn_above.notna()
        & low.lt(prior_lvn_above)
        & close.gt(prior_lvn_above)
        & wide_range
        & bull_pressure
        & not_far_above_lvn
        & prior_lvn_above_thinness.ge(lvn_thinness_min)
    )
    lvn_fast_traverse_short = (
        prior_lvn_below.notna()
        & high.gt(prior_lvn_below)
        & close.lt(prior_lvn_below)
        & wide_range
        & bear_pressure
        & not_far_below_lvn
        & prior_lvn_below_thinness.ge(lvn_thinness_min)
    )

    poc_migration_pct = _pct_series(poc - poc.shift(int(cfg.poc_migration_window)), close)
    value_mid = (vah + val) / 2.0
    value_direction_pct = _pct_series(value_mid - value_mid.shift(int(cfg.poc_migration_window)), close)
    poc_migration_long = _clip01(poc_migration_pct / max(cfg.node_near_pct * 4.0, 1e-9))
    poc_migration_short = _clip01(-poc_migration_pct / max(cfg.node_near_pct * 4.0, 1e-9))
    value_direction_long = _clip01(value_direction_pct / max(cfg.node_near_pct * 4.0, 1e-9))
    value_direction_short = _clip01(-value_direction_pct / max(cfg.node_near_pct * 4.0, 1e-9))
    value_direction_abs = pd.concat([value_direction_long, value_direction_short], axis=1).max(axis=1)
    score_long, score_short = _profile_scores(
        cfg,
        poc_migration_long,
        poc_migration_short,
        vah_breakout_with_pressure,
        val_breakdown_with_pressure,
        lower_rejection_with_pressure,
        upper_rejection_with_pressure,
        hvn_below_reclaim,
        hvn_above_reject,
        lvn_below_reject_long,
        lvn_above_reject_short,
        lvn_accept_long,
        lvn_accept_short,
        lvn_fast_traverse_long,
        lvn_fast_traverse_short,
        volume_ok,
    )
    score_abs = pd.concat([score_long, score_short], axis=1).max(axis=1)
    score_state = pd.Series(
        np.select([score_long.gt(score_short), score_short.gt(score_long)], [1.0, -1.0], default=0.0),
        index=frame.index,
    )
    raw_entry_trigger_long = (
        vah_breakout_with_pressure
        | lower_rejection_with_pressure
        | hvn_below_reclaim
        | lvn_below_reject_long
        | lvn_accept_long
        | lvn_fast_traverse_long
    ) & score_long.gt(score_short + cfg.entry_score_margin)
    raw_entry_trigger_short = (
        val_breakdown_with_pressure
        | upper_rejection_with_pressure
        | hvn_above_reject
        | lvn_above_reject_short
        | lvn_accept_short
        | lvn_fast_traverse_short
    ) & score_short.gt(score_long + cfg.entry_score_margin)
    cooldown_bars = max(3, min(int(cfg.score_window) // 6, 10))
    entry_trigger_long = _dedupe_events(raw_entry_trigger_long, cooldown_bars)
    entry_trigger_short = _dedupe_events(raw_entry_trigger_short, cooldown_bars)

    context_window = max(6, min(int(cfg.score_window) // 2, 18))
    long_event_density = _clip01(
        raw_entry_trigger_long.astype("float64").rolling(context_window, min_periods=1).sum() / 3.0
    )
    short_event_density = _clip01(
        raw_entry_trigger_short.astype("float64").rolling(context_window, min_periods=1).sum() / 3.0
    )
    inside_value_ratio = in_value_area.astype("float64").rolling(context_window, min_periods=1).mean()
    bull_acceptance_ratio = (
        above_value_area | (in_value_area & close.ge(prior_poc)) | lower_rejection_with_pressure
    ).astype("float64").rolling(context_window, min_periods=1).mean()
    bear_acceptance_ratio = (
        below_value_area | (in_value_area & close.le(prior_poc)) | upper_rejection_with_pressure
    ).astype("float64").rolling(context_window, min_periods=1).mean()
    pressure_abs = delta.abs().clip(0.0, 1.0)
    pressure_bull_score = _clip01(delta / max(float(cfg.pressure_delta_min) * 3.0, 1e-9))
    pressure_bear_score = _clip01(-delta / max(float(cfg.pressure_delta_min) * 3.0, 1e-9))
    poc_stability = _clip01(1.0 - (poc_migration_pct.abs() / max(float(cfg.node_near_pct) * 3.0, 1e-9)))
    bull_context_raw = _clip01(
        0.24 * score_long
        + 0.19 * poc_migration_long
        + 0.17 * value_direction_long
        + 0.17 * bull_acceptance_ratio
        + 0.13 * long_event_density
        + 0.10 * pressure_bull_score
    )
    bear_context_raw = _clip01(
        0.24 * score_short
        + 0.19 * poc_migration_short
        + 0.17 * value_direction_short
        + 0.17 * bear_acceptance_ratio
        + 0.13 * short_event_density
        + 0.10 * pressure_bear_score
    )
    balance_context_raw = _clip01(
        0.30 * inside_value_ratio
        + 0.24 * poc_stability
        + 0.18 * (1.0 - score_abs)
        + 0.16 * (1.0 - pressure_abs)
        + 0.12 * (1.0 - value_direction_abs)
    )
    context_smooth = max(3, min(int(cfg.score_window) // 4, 10))
    bull_context_score = _clip01(bull_context_raw.rolling(context_smooth, min_periods=1).mean())
    bear_context_score = _clip01(bear_context_raw.rolling(context_smooth, min_periods=1).mean())
    balance_context_score = _clip01(balance_context_raw.rolling(context_smooth, min_periods=1).mean())
    min_duration = max(2, min(context_smooth // 2, 4))
    full_bull_candidate = (
        valid_prior
        & bull_context_score.ge(cfg.context_full_min)
        & bull_context_score.gt(bear_context_score + cfg.context_full_margin)
        & bull_context_score.gt(balance_context_score)
    )
    full_bear_candidate = (
        valid_prior
        & bear_context_score.ge(cfg.context_full_min)
        & bear_context_score.gt(bull_context_score + cfg.context_full_margin)
        & bear_context_score.gt(balance_context_score)
    )
    full_bull_context = full_bull_candidate.astype("float64").rolling(
        min_duration,
        min_periods=min_duration,
    ).mean().ge(0.66)
    full_bear_context = (
        full_bear_candidate.astype("float64").rolling(min_duration, min_periods=min_duration).mean().ge(0.66)
        & ~full_bull_context
    )
    direction_margin = bull_context_score - bear_context_score
    bullish_chop_candidate = (
        valid_prior
        & ~full_bull_context
        & ~full_bear_context
        & direction_margin.ge(cfg.context_soft_margin)
        & bull_context_score.ge(cfg.context_soft_min)
        & balance_context_score.ge(cfg.context_balance_min)
    )
    bearish_chop_candidate = (
        valid_prior
        & ~full_bull_context
        & ~full_bear_context
        & direction_margin.le(-cfg.context_soft_margin)
        & bear_context_score.ge(cfg.context_soft_min)
        & balance_context_score.ge(cfg.context_balance_min)
    )
    bullish_chop_context = bullish_chop_candidate.astype("float64").rolling(
        min_duration,
        min_periods=min_duration,
    ).mean().ge(0.66)
    bearish_chop_context = (
        bearish_chop_candidate.astype("float64").rolling(min_duration, min_periods=min_duration).mean().ge(0.66)
        & ~bullish_chop_context
    )
    market_context = pd.Series(
        np.select(
            [full_bull_context, full_bear_context, bullish_chop_context, bearish_chop_context],
            [2, -2, 1, -1],
            default=0,
        ),
        index=frame.index,
        dtype="int8",
    )

    node_entry_long_raw = (
        hvn_below_reclaim
        | lvn_below_reject_long
        | lvn_accept_long
        | lvn_fast_traverse_long
    ) & score_long.gt(score_short + cfg.entry_score_margin)
    node_entry_short_raw = (
        hvn_above_reject
        | lvn_above_reject_short
        | lvn_accept_short
        | lvn_fast_traverse_short
    ) & score_short.gt(score_long + cfg.entry_score_margin)
    node_entry_long = _dedupe_events(node_entry_long_raw, cooldown_bars)
    node_entry_short = _dedupe_events(node_entry_short_raw, cooldown_bars)
    hvn_hold_long = (
        hvn_below.notna()
        & hvn_below_strength.ge(hvn_strength_min)
        & close.gt(hvn_below)
        & _pct_series(close - hvn_below, close).le(entry_extension_pct * cfg.node_hold_near_mult)
        & (score_long.ge(score_short) | market_context.ge(1))
    )
    hvn_hold_short = (
        hvn_above.notna()
        & hvn_above_strength.ge(hvn_strength_min)
        & close.lt(hvn_above)
        & _pct_series(hvn_above - close, close).le(entry_extension_pct * cfg.node_hold_near_mult)
        & (score_short.ge(score_long) | market_context.le(-1))
    )
    lvn_hold_long = (
        prior_lvn_above.notna()
        & prior_lvn_above_thinness.ge(lvn_thinness_min)
        & close.gt(prior_lvn_above)
        & close_above_lvn_pct.le(entry_extension_pct * cfg.node_hold_near_mult)
        & (score_long.gt(score_short) | market_context.ge(1))
    )
    lvn_hold_short = (
        prior_lvn_below.notna()
        & prior_lvn_below_thinness.ge(lvn_thinness_min)
        & close.lt(prior_lvn_below)
        & close_below_lvn_pct.le(entry_extension_pct * cfg.node_hold_near_mult)
        & (score_short.gt(score_long) | market_context.le(-1))
    )
    node_hold_long = hvn_hold_long | lvn_hold_long
    node_hold_short = hvn_hold_short | lvn_hold_short
    node_exit_long_raw = actionable_hvn_above & (
        hvn_above_reject
        | lvn_above_reject_short
        | upper_rejection_with_pressure
        | (bear_pressure & score_short.ge(score_long * 0.75))
    )
    node_exit_short_raw = actionable_hvn_below & (
        hvn_below_reclaim
        | lvn_below_reject_long
        | lower_rejection_with_pressure
        | (bull_pressure & score_long.ge(score_short * 0.75))
    )
    node_exit_long = _dedupe_events(node_exit_long_raw, cooldown_bars)
    node_exit_short = _dedupe_events(node_exit_short_raw, cooldown_bars)

    updates = {
        f"{p}_prior_poc": prior_poc,
        f"{p}_prior_vah": prior_vah,
        f"{p}_prior_val": prior_val,
        f"{p}_in_value_area": in_value_area,
        f"{p}_above_value_area": above_value_area,
        f"{p}_below_value_area": below_value_area,
        f"{p}_upper_rejection": upper_rejection,
        f"{p}_lower_rejection": lower_rejection,
        f"{p}_vah_breakout": vah_breakout,
        f"{p}_val_breakdown": val_breakdown,
        f"{p}_bull_pressure": bull_pressure,
        f"{p}_bear_pressure": bear_pressure,
        f"{p}_poc_delta_bull": poc_bull,
        f"{p}_poc_delta_bear": poc_bear,
        f"{p}_close_bin_volume_ok": volume_ok,
        f"{p}_near_hvn_below": near_hvn_below,
        f"{p}_near_hvn_above": near_hvn_above,
        f"{p}_actionable_hvn_below": actionable_hvn_below,
        f"{p}_actionable_hvn_above": actionable_hvn_above,
        f"{p}_near_lvn_below": near_lvn_below,
        f"{p}_near_lvn_above": near_lvn_above,
        f"{p}_actionable_lvn_below": actionable_lvn_below,
        f"{p}_actionable_lvn_above": actionable_lvn_above,
        f"{p}_hvn_below_reclaim": hvn_below_reclaim,
        f"{p}_hvn_above_reject": hvn_above_reject,
        f"{p}_lvn_below_reject_long": lvn_below_reject_long,
        f"{p}_lvn_above_reject_short": lvn_above_reject_short,
        f"{p}_lvn_accept_long": lvn_accept_long,
        f"{p}_lvn_accept_short": lvn_accept_short,
        f"{p}_lvn_fast_traverse_long": lvn_fast_traverse_long,
        f"{p}_lvn_fast_traverse_short": lvn_fast_traverse_short,
        f"{p}_poc_migration_pct": poc_migration_pct,
        f"{p}_value_direction_pct": value_direction_pct,
        f"{p}_poc_migration_score_long": poc_migration_long,
        f"{p}_poc_migration_score_short": poc_migration_short,
        f"{p}_value_direction_score_long": value_direction_long,
        f"{p}_value_direction_score_short": value_direction_short,
        f"{p}_vah_breakout_with_pressure": vah_breakout_with_pressure,
        f"{p}_val_breakdown_with_pressure": val_breakdown_with_pressure,
        f"{p}_lower_rejection_with_pressure": lower_rejection_with_pressure,
        f"{p}_upper_rejection_with_pressure": upper_rejection_with_pressure,
        f"{p}_entry_trigger_long": entry_trigger_long,
        f"{p}_entry_trigger_short": entry_trigger_short,
        f"{p}_context_score_bull": bull_context_score,
        f"{p}_context_score_bear": bear_context_score,
        f"{p}_context_score_balance": balance_context_score,
        f"{p}_market_context": market_context,
        f"{p}_node_entry_long": node_entry_long,
        f"{p}_node_entry_short": node_entry_short,
        f"{p}_node_hold_long": node_hold_long,
        f"{p}_node_hold_short": node_hold_short,
        f"{p}_node_exit_long": node_exit_long,
        f"{p}_node_exit_short": node_exit_short,
        f"{p}_score_long": score_long,
        f"{p}_score_short": score_short,
        f"{p}_score_abs": score_abs,
        f"{p}_state": score_state,
    }
    update_frame = pd.DataFrame(updates, index=frame.index)
    existing_update_columns = [column for column in update_frame.columns if column in frame.columns]
    frame = frame.drop(columns=existing_update_columns, errors="ignore")
    return pd.concat([frame, update_frame], axis=1)


def _profile_scores(
    cfg: VolumeProfileConfig,
    poc_migration_long: pd.Series,
    poc_migration_short: pd.Series,
    vah_breakout_with_pressure: pd.Series,
    val_breakdown_with_pressure: pd.Series,
    lower_rejection_with_pressure: pd.Series,
    upper_rejection_with_pressure: pd.Series,
    hvn_below_reclaim: pd.Series,
    hvn_above_reject: pd.Series,
    lvn_below_reject_long: pd.Series,
    lvn_above_reject_short: pd.Series,
    lvn_accept_long: pd.Series,
    lvn_accept_short: pd.Series,
    lvn_fast_traverse_long: pd.Series,
    lvn_fast_traverse_short: pd.Series,
    volume_ok: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    w = int(cfg.score_window)
    long_events = (
        1.10 * vah_breakout_with_pressure.astype("float64")
        + 1.00 * lower_rejection_with_pressure.astype("float64")
        + 0.90 * hvn_below_reclaim.astype("float64")
        + 0.90 * lvn_below_reject_long.astype("float64")
        + 0.80 * lvn_accept_long.astype("float64")
        + 0.80 * lvn_fast_traverse_long.astype("float64")
    )
    short_events = (
        1.10 * val_breakdown_with_pressure.astype("float64")
        + 1.00 * upper_rejection_with_pressure.astype("float64")
        + 0.90 * hvn_above_reject.astype("float64")
        + 0.90 * lvn_above_reject_short.astype("float64")
        + 0.80 * lvn_accept_short.astype("float64")
        + 0.80 * lvn_fast_traverse_short.astype("float64")
    )
    long_recent = _clip01(long_events.rolling(w, min_periods=1).sum() / 4.0)
    short_recent = _clip01(short_events.rolling(w, min_periods=1).sum() / 4.0)
    participation = volume_ok.astype("float64")
    long_score = _clip01(0.45 * long_recent + 0.35 * poc_migration_long + 0.20 * participation)
    short_score = _clip01(0.45 * short_recent + 0.35 * poc_migration_short + 0.20 * participation)
    return long_score, short_score


def _dedupe_events(events: pd.Series, cooldown_bars: int) -> pd.Series:
    flags = events.fillna(False).astype("bool")
    if cooldown_bars <= 0:
        return flags
    recent = flags.shift(1, fill_value=False).rolling(int(cooldown_bars), min_periods=1).max()
    return flags & ~recent.fillna(0.0).astype("bool")


def _atr(frame: DataFrame, period: int) -> pd.Series:
    high = _numeric_series(frame, "high")
    low = _numeric_series(frame, "low")
    close = _numeric_series(frame, "close")
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(int(period), min_periods=1).mean()


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
    node_score: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    if node_score is None:
        above_score = np.where(has_above, 1.0, np.nan)
        below_score = np.where(has_below, 1.0, np.nan)
    else:
        above_score = np.where(has_above, node_score[np.arange(len(current_close)), above_idx], np.nan)
        below_score = np.where(has_below, node_score[np.arange(len(current_close)), below_idx], np.nan)
    return above, below, above_score, below_score


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


def _clip01(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0).fillna(0.0)


def _nan_invalid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.where(valid & np.isfinite(values), values, np.nan)

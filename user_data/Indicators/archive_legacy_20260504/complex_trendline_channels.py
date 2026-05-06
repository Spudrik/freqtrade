from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


@dataclass(frozen=True)
class TrendlineChannelConfig:
    """Pair ranked trendline outputs into channel evidence columns.

    This module intentionally consumes trendline columns instead of generating
    pivots or lines itself. The split keeps trendlines as the shared foundation
    and lets pattern detectors use channels without duplicating line logic.

    Required source columns are produced by ``complex_trendline_projection_v2``:
    ``<source_prefix>_resistance_*_rankN`` and
    ``<source_prefix>_support_*_rankN``.

    ``channel_data_proximity_pct`` suppresses geometrically valid channels when
    the channel rails are not close to actual candle highs/lows. Both rails must
    have recent candle interaction, which keeps loose projected structures from
    being treated as active market context on the current timeframe.
    """

    source_prefix: str = "tlv2"
    output_prefix: str = "tlv2"
    output_label: str = "local_channel"
    source_line_count: int = 3
    channel_output_count: int = 3

    channel_min_overlap_bars: int = 10
    channel_slope_tolerance_pct: float = 0.45
    channel_parallel_width_change_pct: float = 0.35
    channel_min_convergence_pct: float = 0.12
    channel_min_width_pct: float = 0.004
    channel_max_width_pct: float = 0.24
    channel_preferred_width_pct: float = 0.045
    channel_data_proximity_pct: float = 0.010
    channel_data_lookback_bars: int = 36
    channel_min_recent_rail_touches: int = 1
    channel_touch_tolerance_pct: float = 0.0050
    channel_touch_atr_mult: float = 0.55
    channel_duplicate_overlap_pct: float = 0.60
    channel_duplicate_mid_width_mult: float = 0.90
    channel_duplicate_width_tolerance_pct: float = 0.70
    channel_breakout_grace_bars: int = 3


def add_trendline_channels(
    dataframe: DataFrame,
    config: TrendlineChannelConfig | None = None,
    **overrides: object,
) -> DataFrame:
    """Append ranked channel columns derived from existing trendline outputs."""

    cfg = _resolve_config(config, overrides)
    _validate_config(cfg)
    _validate_dataframe(dataframe, cfg)

    frame = dataframe.copy()
    base = _base_inputs(frame, cfg)
    line_columns = _line_columns(frame, cfg)
    new_cols = _channel_columns_from_ranked(line_columns, base, cfg)

    p = cfg.output_prefix
    label = cfg.output_label
    existing = [col for col in frame.columns if str(col).startswith(f"{p}_{label}_")]
    clean = frame.drop(columns=existing).copy() if existing else frame.copy()
    return pd.concat([clean, pd.DataFrame(new_cols, index=frame.index)], axis=1)


def _base_inputs(frame: DataFrame, cfg: TrendlineChannelConfig) -> dict[str, Series]:
    p = cfg.source_prefix
    close = _num(frame, "close")
    open_ = _num(frame, "open")
    high = _num(frame, "high")
    low = _num(frame, "low")
    body_high = _column_or_body(frame, f"{p}_body_high", pd.concat([open_, close], axis=1).max(axis=1))
    body_low = _column_or_body(frame, f"{p}_body_low", pd.concat([open_, close], axis=1).min(axis=1))
    atr = _column_or_body(frame, f"{p}_atr", _atr(frame, 14))
    bar_index = pd.Series(np.arange(len(frame), dtype="float64"), index=frame.index)
    return {
        "close": close,
        "high": high,
        "low": low,
        "body_high": body_high,
        "body_low": body_low,
        "atr": atr,
        "bar_index": bar_index,
    }


def _line_columns(frame: DataFrame, cfg: TrendlineChannelConfig) -> dict[str, Series]:
    required = _required_line_columns(cfg)
    return {column: pd.to_numeric(frame[column], errors="coerce") for column in required}


def _required_line_columns(cfg: TrendlineChannelConfig) -> list[str]:
    required: list[str] = []
    for side in ("resistance", "support"):
        for rank in range(int(cfg.source_line_count)):
            for metric in (
                "line",
                "slope",
                "score",
                "anchor_old_index",
                "anchor_new_index",
                "projection_end_index",
                "pivot_count",
                "line_id",
            ):
                required.append(f"{cfg.source_prefix}_{side}_{metric}_rank{rank}")
    return required


def _channel_columns_from_ranked(
    line_columns: dict[str, Series],
    base: dict[str, Series],
    cfg: TrendlineChannelConfig,
) -> dict[str, Series]:
    pack = _empty_channel_pack()
    count = int(cfg.source_line_count)
    for resistance_rank in range(count):
        for support_rank in range(count):
            candidate = _rolling_channel_from_ranked_pair(
                line_columns,
                base,
                cfg,
                resistance_rank=resistance_rank,
                support_rank=support_rank,
            )
            for key, value in candidate.items():
                pack[key].append(value)
    return _rank_channel_columns(pack, cfg, index=base["close"].index)


def _rolling_channel_from_ranked_pair(
    line_columns: dict[str, Series],
    base: dict[str, Series],
    cfg: TrendlineChannelConfig,
    *,
    resistance_rank: int,
    support_rank: int,
) -> dict[str, Series]:
    index = base["close"].index
    p = cfg.source_prefix

    def get(name: str) -> Series:
        return line_columns.get(name, pd.Series(np.nan, index=index, dtype="float64"))

    res = f"{p}_resistance"
    sup = f"{p}_support"
    upper = get(f"{res}_line_rank{resistance_rank}")
    lower = get(f"{sup}_line_rank{support_rank}")
    res_slope = get(f"{res}_slope_rank{resistance_rank}")
    sup_slope = get(f"{sup}_slope_rank{support_rank}")
    res_score = get(f"{res}_score_rank{resistance_rank}")
    sup_score = get(f"{sup}_score_rank{support_rank}")
    res_new = get(f"{res}_anchor_new_index_rank{resistance_rank}")
    sup_new = get(f"{sup}_anchor_new_index_rank{support_rank}")
    res_end = get(f"{res}_projection_end_index_rank{resistance_rank}")
    sup_end = get(f"{sup}_projection_end_index_rank{support_rank}")
    res_pivots = get(f"{res}_pivot_count_rank{resistance_rank}")
    sup_pivots = get(f"{sup}_pivot_count_rank{support_rank}")
    res_line_id = get(f"{res}_line_id_rank{resistance_rank}")
    sup_line_id = get(f"{sup}_line_id_rank{support_rank}")

    close = base["close"]
    bar_index = base["bar_index"]
    overlap_start = pd.concat([res_new, sup_new], axis=1).max(axis=1)
    overlap_end = pd.concat([res_end, sup_end], axis=1).min(axis=1)
    overlap_bars = overlap_end - overlap_start
    width = upper - lower
    width_pct = (width / close.abs().replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    slope_diff_pct = (
        (res_slope - sup_slope).abs()
        / pd.concat([res_slope.abs(), sup_slope.abs(), pd.Series(1e-9, index=index)], axis=1).max(axis=1)
    ).replace([np.inf, -np.inf], np.nan)

    span_start_width = _channel_width_at(overlap_start, res_slope, sup_slope, upper, lower, bar_index)
    span_end_width = _channel_width_at(overlap_end, res_slope, sup_slope, upper, lower, bar_index)
    width_change_pct = ((span_end_width - span_start_width) / span_start_width.abs().replace(0.0, np.nan)).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    parallel_score = _clip01(1.0 - width_change_pct.abs() / max(float(cfg.channel_parallel_width_change_pct), 1e-9))
    convergence_score = _clip01(
        (-width_change_pct - float(cfg.channel_min_convergence_pct)) / max(float(cfg.channel_parallel_width_change_pct), 1e-9)
    )
    relation_score = pd.concat([parallel_score, convergence_score], axis=1).max(axis=1)
    channel_shape = pd.Series(0.0, index=index, dtype="float64").where(convergence_score.le(parallel_score), 1.0)
    channel_shape = channel_shape.where(width_change_pct.le(float(cfg.channel_parallel_width_change_pct)), -1.0)

    line_score = pd.concat([res_score, sup_score], axis=1).mean(axis=1)
    overlap_score = _clip01(overlap_bars / max(float(cfg.channel_min_overlap_bars) * 3.0, 1.0))
    slope_score = _clip01(1.0 - slope_diff_pct / max(float(cfg.channel_slope_tolerance_pct), 1e-9))
    preferred_width = max(float(cfg.channel_preferred_width_pct), float(cfg.channel_min_width_pct))
    width_score = _clip01(1.0 - (width_pct - preferred_width).abs() / max(preferred_width, 1e-9))
    pivot_score = _clip01((res_pivots.fillna(2.0) + sup_pivots.fillna(2.0)) / 10.0)
    score = _clip01(
        0.24 * line_score
        + 0.18 * overlap_score
        + 0.18 * relation_score
        + 0.14 * slope_score
        + 0.10 * width_score
        + 0.16 * pivot_score
    )
    valid = (
        upper.notna()
        & lower.notna()
        & width.gt(0.0)
        & width_pct.ge(float(cfg.channel_min_width_pct))
        & width_pct.le(float(cfg.channel_max_width_pct))
        & span_start_width.gt(0.0)
        & span_end_width.gt(0.0)
        & overlap_bars.ge(float(cfg.channel_min_overlap_bars))
        & relation_score.gt(0.0)
        & bar_index.ge(overlap_start)
        & bar_index.le(overlap_end)
    ).fillna(False)
    tolerance = pd.concat(
        [
            close.abs() * float(cfg.channel_touch_tolerance_pct),
            base["atr"].fillna(0.0) * float(cfg.channel_touch_atr_mult),
        ],
        axis=1,
    ).max(axis=1)
    position = ((close - lower) / width.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    body_high = base["body_high"]
    body_low = base["body_low"]
    candle_high = base["high"]
    candle_low = base["low"]
    channel_data_distance = pd.Series(
        np.select(
            [
                body_low.gt(upper),
                body_high.lt(lower),
            ],
            [
                body_low - upper,
                lower - body_high,
            ],
            default=0.0,
        ),
        index=index,
        dtype="float64",
    ).where(valid)
    channel_data_distance_pct = (
        channel_data_distance / close.abs().replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    upper_distance = _distance_from_price_to_range(upper, candle_low, candle_high).where(valid)
    lower_distance = _distance_from_price_to_range(lower, candle_low, candle_high).where(valid)
    upper_distance_pct = (upper_distance / close.abs().replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    lower_distance_pct = (lower_distance / close.abs().replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    upper_near_data = upper_distance_pct.le(float(cfg.channel_data_proximity_pct)).fillna(False)
    lower_near_data = lower_distance_pct.le(float(cfg.channel_data_proximity_pct)).fillna(False)
    rail_touch_window = max(int(cfg.channel_data_lookback_bars), 1)
    upper_recent_touch_count = _rolling_count(upper_near_data.where(valid, False), rail_touch_window)
    lower_recent_touch_count = _rolling_count(lower_near_data.where(valid, False), rail_touch_window)
    near_data = (
        channel_data_distance_pct.le(float(cfg.channel_data_proximity_pct)).fillna(False)
        & upper_recent_touch_count.ge(float(cfg.channel_min_recent_rail_touches))
        & lower_recent_touch_count.ge(float(cfg.channel_min_recent_rail_touches))
    )
    inside = (body_high.le(upper + tolerance) & body_low.ge(lower - tolerance)).fillna(False)
    breakout_up = close.gt(upper + tolerance).fillna(False)
    breakdown_down = close.lt(lower - tolerance).fillna(False)
    outside = breakout_up | breakdown_down
    stale_break = _rolling_all_true(outside.where(valid, False), max(int(cfg.channel_breakout_grace_bars), 1))
    active = valid & near_data & ~stale_break
    active_quality = pd.Series(1.0, index=index, dtype="float64").where(inside, 0.72)
    active_quality = active_quality.where(~outside, 0.46)
    score = (score * active_quality).where(active)
    return {
        "upper": upper.where(active),
        "lower": lower.where(active),
        "mid": ((upper + lower) / 2.0).where(active),
        "width": width.where(active),
        "width_pct": width_pct.where(active),
        "data_distance_pct": channel_data_distance_pct.where(active),
        "upper_distance_pct": upper_distance_pct.where(active),
        "lower_distance_pct": lower_distance_pct.where(active),
        "position": position.where(active),
        "score": score,
        "slope_upper": res_slope.where(active),
        "slope_lower": sup_slope.where(active),
        "slope_diff_pct": slope_diff_pct.where(active),
        "width_change_pct": width_change_pct.where(active),
        "shape": channel_shape.where(active),
        "overlap_start_index": overlap_start.where(active),
        "overlap_end_index": overlap_end.where(active),
        "overlap_bars": overlap_bars.where(active),
        "resistance_slot": pd.Series(float(resistance_rank), index=index).where(active),
        "support_slot": pd.Series(float(support_rank), index=index).where(active),
        "resistance_line_id": res_line_id.where(active),
        "support_line_id": sup_line_id.where(active),
        "active": active.astype("float64").where(valid),
        "near_data": near_data.astype("float64").where(valid),
        "upper_near_data": upper_near_data.astype("float64").where(valid),
        "lower_near_data": lower_near_data.astype("float64").where(valid),
        "upper_recent_touch_count": upper_recent_touch_count.where(valid),
        "lower_recent_touch_count": lower_recent_touch_count.where(valid),
        "inside": inside.astype("float64").where(valid),
        "stale_break": stale_break.astype("float64").where(valid),
        "breakout_up": breakout_up.astype("float64").where(valid),
        "breakdown_down": breakdown_down.astype("float64").where(valid),
        "near_upper": (upper - close).abs().le(tolerance).astype("float64").where(active),
        "near_lower": (close - lower).abs().le(tolerance).astype("float64").where(active),
    }


def _channel_width_at(
    target_x: Series,
    res_slope: Series,
    sup_slope: Series,
    current_upper: Series,
    current_lower: Series,
    current_x: Series,
) -> Series:
    upper_at_x = current_upper + res_slope * (target_x - current_x)
    lower_at_x = current_lower + sup_slope * (target_x - current_x)
    return upper_at_x - lower_at_x


def _rank_channel_columns(
    pack: dict[str, list[Series]],
    cfg: TrendlineChannelConfig,
    *,
    index: pd.Index,
) -> dict[str, Series]:
    p = cfg.output_prefix
    label = cfg.output_label
    slots = int(cfg.channel_output_count)
    out: dict[str, Series] = {}
    if not pack["score"]:
        return _empty_channel_columns(index, p, slots, label=label)
    score_values = _series_matrix(pack["score"])
    score_values = np.where(np.isfinite(score_values), score_values, -np.inf)
    metric_values = {key: _series_matrix(pack[key]) for key in pack if key != "score"}
    remaining = score_values.copy()
    rows = np.arange(len(index))
    for rank in range(slots):
        chosen = np.argmax(remaining, axis=1)
        chosen_score = remaining[rows, chosen]
        valid = np.isfinite(chosen_score) & (chosen_score > -np.inf)
        out[f"{p}_{label}_score_rank{rank}"] = pd.Series(np.where(valid, chosen_score, np.nan), index=index, dtype="float64")
        for metric in _channel_metric_names():
            out[f"{p}_{label}_{metric}_rank{rank}"] = pd.Series(
                _select_metric(metric_values[metric], chosen, valid),
                index=index,
                dtype="float64",
            )
        duplicate = _rolling_channel_duplicate_mask(metric_values, chosen, valid, cfg)
        remaining[duplicate] = -np.inf
        remaining[rows, chosen] = -np.inf
    return out


def _rolling_channel_duplicate_mask(
    metric_values: dict[str, np.ndarray],
    chosen: np.ndarray,
    valid: np.ndarray,
    cfg: TrendlineChannelConfig,
) -> np.ndarray:
    """Row-wise duplicate suppression for strategy-facing ranked channels."""

    if "mid" not in metric_values or metric_values["mid"].size == 0:
        return np.zeros((len(valid), 0), dtype=bool)

    rows = np.arange(len(valid))
    start = metric_values["overlap_start_index"]
    end = metric_values["overlap_end_index"]
    mid = metric_values["mid"]
    width = metric_values["width"]

    chosen_start = start[rows, chosen][:, np.newaxis]
    chosen_end = end[rows, chosen][:, np.newaxis]
    chosen_mid = mid[rows, chosen][:, np.newaxis]
    chosen_width = width[rows, chosen][:, np.newaxis]

    overlap = np.minimum(end, chosen_end) - np.maximum(start, chosen_start)
    span = np.maximum(end - start, 1.0)
    chosen_span = np.maximum(chosen_end - chosen_start, 1.0)
    overlap_ratio = overlap / np.maximum(np.minimum(span, chosen_span), 1.0)
    width_ref = np.maximum(np.minimum(width, chosen_width), 1e-9)
    mid_close = np.abs(mid - chosen_mid) <= width_ref * float(cfg.channel_duplicate_mid_width_mult)
    width_close = _relative_array_diff(width, chosen_width) <= float(cfg.channel_duplicate_width_tolerance_pct)
    finite = np.isfinite(start) & np.isfinite(end) & np.isfinite(mid) & np.isfinite(width)
    return valid[:, np.newaxis] & finite & (overlap_ratio >= float(cfg.channel_duplicate_overlap_pct)) & mid_close & width_close


def _empty_channel_pack() -> dict[str, list[Series]]:
    return {metric: [] for metric in ("score", *_channel_metric_names())}


def _empty_channel_columns(index: pd.Index, prefix: str, slots: int, *, label: str) -> dict[str, Series]:
    out: dict[str, Series] = {}
    nan = pd.Series(np.nan, index=index, dtype="float64")
    for rank in range(slots):
        out[f"{prefix}_{label}_score_rank{rank}"] = nan
        for metric in _channel_metric_names():
            out[f"{prefix}_{label}_{metric}_rank{rank}"] = nan
    return out


def _channel_metric_names() -> tuple[str, ...]:
    return (
        "upper",
        "lower",
        "mid",
        "width",
        "width_pct",
        "data_distance_pct",
        "upper_distance_pct",
        "lower_distance_pct",
        "position",
        "slope_upper",
        "slope_lower",
        "slope_diff_pct",
        "width_change_pct",
        "shape",
        "overlap_start_index",
        "overlap_end_index",
        "overlap_bars",
        "resistance_slot",
        "support_slot",
        "resistance_line_id",
        "support_line_id",
        "active",
        "near_data",
        "upper_near_data",
        "lower_near_data",
        "upper_recent_touch_count",
        "lower_recent_touch_count",
        "inside",
        "stale_break",
        "breakout_up",
        "breakdown_down",
        "near_upper",
        "near_lower",
    )


def _series_matrix(values: Sequence[Series]) -> np.ndarray:
    if not values:
        return np.empty((0, 0), dtype="float64")
    return pd.concat(list(values), axis=1).astype("float64").to_numpy()


def _select_metric(values: np.ndarray, chosen: np.ndarray, valid: np.ndarray) -> np.ndarray:
    rows = np.arange(values.shape[0])
    selected = values[rows, chosen]
    return np.where(valid, selected, np.nan)


def _relative_array_diff(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominator = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1e-9)
    return np.abs(left - right) / denominator


def _distance_from_price_to_range(price: Series, range_low: Series, range_high: Series) -> Series:
    above = (price - range_high).clip(lower=0.0)
    below = (range_low - price).clip(lower=0.0)
    return (above + below).replace([np.inf, -np.inf], np.nan)


def _rolling_count(mask: Series, window: int) -> Series:
    clean = pd.Series(mask, index=mask.index).fillna(False).astype("int8")
    cumulative = clean.cumsum()
    return (cumulative - cumulative.shift(int(window), fill_value=0)).astype("float64")


def _rolling_all_true(condition: Series, window: int) -> Series:
    lookback = max(int(window), 1)
    values = condition.fillna(False).astype("float64")
    return values.rolling(lookback, min_periods=lookback).sum().ge(float(lookback))


def _column_or_body(frame: DataFrame, column: str, default: Series) -> Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return default


def _atr(frame: DataFrame, period: int) -> Series:
    high = _num(frame, "high")
    low = _num(frame, "low")
    close = _num(frame, "close")
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(int(period), min_periods=max(2, int(period) // 2)).mean()


def _num(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _clip01(value: Series) -> Series:
    return value.clip(lower=0.0, upper=1.0)


def _resolve_config(config: TrendlineChannelConfig | None, overrides: dict[str, object]) -> TrendlineChannelConfig:
    cfg = config or TrendlineChannelConfig()
    clean = {key: value for key, value in overrides.items() if value is not None}
    return replace(cfg, **clean) if clean else cfg


def _validate_dataframe(frame: DataFrame, cfg: TrendlineChannelConfig) -> None:
    required = {"open", "high", "low", "close", *_required_line_columns(cfg)}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataframe missing required trendline channel inputs: {missing}")


def _validate_config(cfg: TrendlineChannelConfig) -> None:
    if not cfg.source_prefix or not cfg.output_prefix or not cfg.output_label:
        raise ValueError("source_prefix, output_prefix, and output_label must be set")
    if cfg.source_line_count < 1:
        raise ValueError("source_line_count must be positive")
    if cfg.channel_output_count < 1:
        raise ValueError("channel_output_count must be positive")
    if cfg.channel_min_overlap_bars < 1:
        raise ValueError("channel_min_overlap_bars must be positive")
    if cfg.channel_slope_tolerance_pct <= 0.0:
        raise ValueError("channel_slope_tolerance_pct must be positive")
    if cfg.channel_parallel_width_change_pct <= 0.0 or cfg.channel_min_convergence_pct < 0.0:
        raise ValueError("channel width relation settings are invalid")
    if cfg.channel_min_width_pct <= 0.0 or cfg.channel_max_width_pct <= cfg.channel_min_width_pct:
        raise ValueError("channel width settings are invalid")
    if cfg.channel_preferred_width_pct <= 0.0:
        raise ValueError("channel_preferred_width_pct must be positive")
    if cfg.channel_data_proximity_pct < 0.0:
        raise ValueError("channel_data_proximity_pct must be non-negative")
    if cfg.channel_data_lookback_bars < 1:
        raise ValueError("channel_data_lookback_bars must be positive")
    if cfg.channel_min_recent_rail_touches < 0:
        raise ValueError("channel_min_recent_rail_touches must be non-negative")
    if cfg.channel_touch_tolerance_pct <= 0.0 or cfg.channel_touch_atr_mult < 0.0:
        raise ValueError("channel touch tolerances are invalid")
    if not 0.0 < cfg.channel_duplicate_overlap_pct <= 1.0:
        raise ValueError("channel_duplicate_overlap_pct must be between 0 and 1")
    if cfg.channel_duplicate_mid_width_mult <= 0.0:
        raise ValueError("channel_duplicate_mid_width_mult must be positive")
    if cfg.channel_duplicate_width_tolerance_pct <= 0.0:
        raise ValueError("channel_duplicate_width_tolerance_pct must be positive")
    if cfg.channel_breakout_grace_bars < 0:
        raise ValueError("channel_breakout_grace_bars must be non-negative")


__all__ = [
    "TrendlineChannelConfig",
    "add_trendline_channels",
]

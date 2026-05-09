from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


def _continuation_proof_columns(
    index: pd.Index,
    base: str,
    mask: Series,
    arrays: dict[str, np.ndarray],
    side: str,
) -> dict[str, Series]:
    """Expose the pole plus upper/lower consolidation boundaries for review.

    line1 is the impulse/pole. line2 is the upper consolidation boundary.
    line3 is the lower consolidation boundary. Values are only emitted on rows
    where the matching flag or pennant setup is active.
    """

    return _proof_line_columns(
        index,
        base,
        mask,
        {
            line_number: tuple(arrays[f"proof_line{line_number}_{field}_{side}"] for field in ("x1", "y1", "x2", "y2"))
            for line_number in (1, 2, 3)
        },
    )


def _geometry_boundary_proof_columns(
    index: pd.Index,
    base: str,
    mask: Series,
    arrays: dict[str, np.ndarray],
) -> dict[str, Series]:
    """Expose upper/lower fitted boundaries for two-line geometric patterns."""

    return _proof_line_columns(
        index,
        base,
        mask,
        {
            1: (
                arrays.get("geometry_upper_start_index", arrays["geometry_start_index"]),
                arrays.get("geometry_upper_anchor_start", arrays["geometry_upper_start"]),
                arrays["geometry_end_index"],
                arrays["geometry_upper"],
            ),
            2: (
                arrays.get("geometry_lower_start_index", arrays["geometry_start_index"]),
                arrays.get("geometry_lower_anchor_start", arrays["geometry_lower_start"]),
                arrays["geometry_end_index"],
                arrays["geometry_lower"],
            ),
        },
    )


def _proof_line_columns(
    index: pd.Index,
    base: str,
    mask: Series,
    lines: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, Series]:
    # TODO_DELETE_CHECK: proof-line columns are diagnostic evidence for visual
    # review and no-lookahead auditing. Before release/hyperopt scale-out, check
    # whether these columns should be gated or trimmed to reduce dataframe bloat.
    clean_mask = pd.Series(mask, index=index).fillna(False).astype("bool")
    columns: dict[str, Series] = {}
    for line_number, values in lines.items():
        for field, source in zip(("x1", "y1", "x2", "y2"), values):
            columns[f"{base}_line{line_number}_{field}"] = pd.Series(source, index=index, dtype="float64").where(clean_mask)
    return columns


def _dedupe_interval_level_events(
    mask: np.ndarray,
    first_index: np.ndarray,
    last_index: np.ndarray,
    primary_level: np.ndarray,
    secondary_level: np.ndarray | None,
    cooldown_bars: int,
    overlap_pct: float,
    level_tolerance_pct: float,
) -> np.ndarray:
    """Suppress repeated attention for the same no-lookahead structure.

    The detector keeps the first confirmed event. Later events are dropped when
    their anchor interval overlaps a prior kept event and their defining level
    or levels describe the same price area. This fixes repeated pattern spam at
    the source: one structure should not emit many setup rows just because the
    rolling window still recognizes it on later candles.
    """

    clean = np.asarray(mask, dtype=bool)
    selected = np.zeros(len(clean), dtype=bool)
    selected_rows: list[int] = []
    cooldown = max(int(cooldown_bars), 1)
    min_overlap = float(overlap_pct)
    tolerance = float(level_tolerance_pct)
    secondary = secondary_level if secondary_level is not None else None

    for row in np.flatnonzero(clean):
        required = [first_index[row], last_index[row], primary_level[row]]
        if secondary is not None:
            required.append(secondary[row])
        if not np.isfinite(required).all():
            continue
        duplicate = False
        for prior in reversed(selected_rows):
            if row - prior <= cooldown:
                duplicate = True
                break
            if _same_interval_level_structure(
                first_index[row],
                last_index[row],
                primary_level[row],
                secondary[row] if secondary is not None else np.nan,
                first_index[prior],
                last_index[prior],
                primary_level[prior],
                secondary[prior] if secondary is not None else np.nan,
                min_overlap,
                tolerance,
                secondary is not None,
            ):
                duplicate = True
                break
        if not duplicate:
            selected[row] = True
            selected_rows.append(int(row))
    return selected


def _same_interval_level_structure(
    first_x: float,
    last_x: float,
    primary: float,
    secondary: float,
    prior_first_x: float,
    prior_last_x: float,
    prior_primary: float,
    prior_secondary: float,
    min_overlap: float,
    level_tolerance_pct: float,
    require_secondary: bool,
) -> bool:
    span = max(float(last_x) - float(first_x), 1.0)
    prior_span = max(float(prior_last_x) - float(prior_first_x), 1.0)
    overlap = min(float(last_x), float(prior_last_x)) - max(float(first_x), float(prior_first_x))
    overlap_ratio = overlap / max(min(span, prior_span), 1.0)
    if overlap_ratio < float(min_overlap):
        return False
    primary_ref = max(abs(float(primary)), abs(float(prior_primary)), 1e-9)
    primary_close = abs(float(primary) - float(prior_primary)) / primary_ref <= float(level_tolerance_pct)
    if not primary_close:
        return False
    if not require_secondary:
        return True
    secondary_ref = max(abs(float(secondary)), abs(float(prior_secondary)), 1e-9)
    return abs(float(secondary) - float(prior_secondary)) / secondary_ref <= float(level_tolerance_pct)


def _lifecycle_state_from_events(mask: np.ndarray | Series, mature_bars: int, stale_bars: int) -> np.ndarray:
    """Carry sparse pattern events forward as mature/stale state.

    Values are intentionally simple:
    - ``0``: no active lifecycle.
    - ``2``: mature/current pattern window after the event row.
    - ``-1``: stale window after the mature window.

    The setup event remains the primary signal. This state only tells the
    strategy or diagnostic plot whether a recent pattern is still fresh enough
    to pay attention to, without re-emitting the same setup repeatedly.
    """

    clean = np.asarray(mask, dtype=bool)
    out = np.zeros(len(clean), dtype="int8")
    mature = max(int(mature_bars), 1)
    stale = max(int(stale_bars), 0)
    active_row = -1
    for row, is_event in enumerate(clean):
        if is_event:
            active_row = int(row)
        if active_row < 0:
            continue
        age = row - active_row
        if age <= mature:
            out[row] = 2
        elif age <= mature + stale:
            out[row] = -1
        else:
            active_row = -1
    return out


def _peak_confirmation_state(
    close: np.ndarray | Series,
    setup_mask: np.ndarray | Series,
    level: np.ndarray | Series,
    quality: np.ndarray | Series,
    mature_bars: int,
    stale_bars: int,
    *,
    top: bool,
) -> dict[str, np.ndarray]:
    """Carry peak structures from developing to confirmed.

    Values are deliberately peak-specific:
    - ``state == 1``: a valid peak structure is developing after the final touch.
    - ``state == 2``: price has confirmed through the reaction level.
    - ``state == 0``: no current structure.
    """

    closes = np.asarray(close, dtype="float64")
    setup = np.asarray(setup_mask, dtype=bool)
    levels = np.asarray(level, dtype="float64")
    qualities = np.asarray(quality, dtype="float64")
    rows = len(closes)
    developing = np.zeros(rows, dtype=bool)
    confirmed = np.zeros(rows, dtype=bool)
    state = np.zeros(rows, dtype="int8")
    carried_quality = np.zeros(rows, dtype="float64")

    mature = max(int(mature_bars), 1)
    stale = max(int(stale_bars), 0)
    max_age = mature + stale
    setup_row = -1
    confirmed_row = -1
    confirm_level = np.nan
    setup_quality = 0.0

    for row in range(rows):
        if setup[row] and np.isfinite(levels[row]):
            setup_row = int(row)
            confirmed_row = -1
            confirm_level = float(levels[row])
            setup_quality = float(qualities[row]) if np.isfinite(qualities[row]) else 0.0

        if setup_row < 0 or not np.isfinite(confirm_level):
            continue

        age = row - setup_row
        if confirmed_row < 0 and age > max_age:
            setup_row = -1
            confirmed_row = -1
            confirm_level = np.nan
            setup_quality = 0.0
            continue

        carried_quality[row] = setup_quality

        close_now = float(closes[row]) if np.isfinite(closes[row]) else np.nan
        if confirmed_row < 0 and np.isfinite(close_now):
            if top:
                is_confirmed = close_now <= confirm_level
            else:
                is_confirmed = close_now >= confirm_level
            if is_confirmed:
                confirmed_row = int(row)

        if confirmed_row >= 0:
            confirmed_age = row - confirmed_row
            if confirmed_age <= mature:
                confirmed[row] = True
                state[row] = 2
            elif confirmed_age <= max_age:
                state[row] = 0
            else:
                carried_quality[row] = 0.0
                setup_row = -1
                confirmed_row = -1
                confirm_level = np.nan
                setup_quality = 0.0
        elif age <= mature:
            developing[row] = True
            state[row] = 1
        else:
            state[row] = 0

    return {
        "developing": developing,
        "confirmed": confirmed,
        "state": state,
        "quality": carried_quality,
    }


def _carry_values_while_state(
    setup_mask: np.ndarray | Series,
    state: np.ndarray | Series,
    values: dict[str, np.ndarray | Series],
) -> dict[str, np.ndarray]:
    """Carry setup-row evidence only while a peak structure is current."""

    setup = np.asarray(setup_mask, dtype=bool)
    current_state = np.asarray(state, dtype="int8")
    sources = {name: np.asarray(value, dtype="float64") for name, value in values.items()}
    out = {name: np.full(len(current_state), np.nan, dtype="float64") for name in sources}
    last: dict[str, float] = {name: np.nan for name in sources}

    for row in range(len(current_state)):
        if setup[row]:
            for name, source in sources.items():
                last[name] = float(source[row]) if np.isfinite(source[row]) else np.nan
        if int(current_state[row]) <= 0:
            continue
        for name, value in last.items():
            out[name][row] = value
    return out


def _pattern_geometry_arrays(
    frame: DataFrame,
    cfg: PatternStructureConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    close = _num(frame["close"]).replace(0.0, np.nan).to_numpy(dtype="float64")
    open_ = _num(frame["open"]).to_numpy(dtype="float64")
    body_high = np.fmax(open_, close)
    body_low = np.fmin(open_, close)
    pp = cfg.pivot_prefix
    high_pivot = _num(frame[f"{pp}_pivot_high"]).to_numpy(dtype="float64")
    low_pivot = _num(frame[f"{pp}_pivot_low"]).to_numpy(dtype="float64")
    fallback_index = np.arange(len(frame), dtype="float64")
    high_index = _num(frame.get(f"{pp}_pivot_high_index", pd.Series(fallback_index, index=frame.index))).to_numpy(dtype="float64")
    low_index = _num(frame.get(f"{pp}_pivot_low_index", pd.Series(fallback_index, index=frame.index))).to_numpy(dtype="float64")
    micro_high, micro_high_index, micro_low, micro_low_index = _confirmed_micro_pivots(
        body_high,
        body_low,
        int(cfg.pattern_pivot_strength),
    )
    high_pivot = _combine_sparse_pivots(high_pivot, high_index, micro_high, micro_high_index)
    low_pivot = _combine_sparse_pivots(low_pivot, low_index, micro_low, micro_low_index)
    high_index = _combine_sparse_indexes(high_pivot, high_index, micro_high_index)
    low_index = _combine_sparse_indexes(low_pivot, low_index, micro_low_index)
    return close, body_high, body_low, high_pivot, high_index, low_pivot, low_index


def _recent_confirmed_pattern_pivots(
    row: int,
    start: int,
    high_pivot: np.ndarray,
    high_index: np.ndarray,
    low_pivot: np.ndarray,
    low_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    high_prices = high_pivot[start : row + 1]
    high_x = high_index[start : row + 1]
    low_prices = low_pivot[start : row + 1]
    low_x = low_index[start : row + 1]
    high_mask = np.isfinite(high_prices) & np.isfinite(high_x) & (high_x >= float(start)) & (high_x <= float(row))
    low_mask = np.isfinite(low_prices) & np.isfinite(low_x) & (low_x >= float(start)) & (low_x <= float(row))
    return high_x[high_mask], high_prices[high_mask], low_x[low_mask], low_prices[low_mask]


def _line_fit_with_error(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return 0.0, np.inf, np.nan
    xv = x[valid].astype("float64")
    yv = y[valid].astype("float64")
    x_mean = float(np.mean(xv))
    y_mean = float(np.mean(yv))
    denominator = float(np.sum((xv - x_mean) ** 2))
    if denominator <= 0.0:
        return 0.0, np.inf, np.nan
    slope = float(np.sum((xv - x_mean) * (yv - y_mean)) / denominator)
    intercept = y_mean - slope * x_mean
    fitted = slope * xv + intercept
    fit_error = float(np.mean(np.abs(yv - fitted)) / max(np.mean(np.abs(yv)), 1e-9))
    return slope, fit_error, intercept


def _line_segment_from_points(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return np.nan, np.nan, np.nan, np.nan
    xv = x[valid].astype("float64")
    yv = y[valid].astype("float64")
    order = np.argsort(xv)
    xv = xv[order]
    yv = yv[order]
    x1 = float(xv[0])
    x2 = float(xv[-1])
    return x1, float(yv[0]), x2, float(yv[-1])


def _line_value_at(x: float, x1: float, y1: float, x2: float, y2: float) -> float:
    if not np.isfinite([x, x1, y1, x2, y2]).all() or x2 == x1:
        return np.nan
    slope = (y2 - y1) / (x2 - x1)
    return float(y1 + slope * (x - x1))


def _prior_pattern_move(close: np.ndarray, first_anchor: float, span: float) -> tuple[int, float]:
    anchor = int(max(min(first_anchor, len(close) - 1), 0))
    lookback = int(max(span, 3.0))
    prior = max(0, anchor - lookback)
    if not np.isfinite(close[anchor]) or not np.isfinite(close[prior]) or close[prior] == 0.0:
        return 0, 0.0
    move_pct = abs(float(close[anchor]) - float(close[prior])) / max(abs(float(close[prior])), 1e-9)
    return (1 if close[anchor] >= close[prior] else -1), float(move_pct)


def _rolling_body_pct(frame: DataFrame, window: int) -> np.ndarray:
    """Return a timeframe-normalized typical body size for dynamic pattern gates."""

    close = _num(frame["close"]).replace(0.0, np.nan).abs()
    body_pct = (_num(frame["close"]) - _num(frame["open"])).abs() / close
    body_pct = body_pct.replace([np.inf, -np.inf], np.nan)
    return _rolling_scale_pct(body_pct, window)


def _rolling_atr_pct(frame: DataFrame, window: int) -> np.ndarray:
    """Return a no-lookahead rolling true-range percent scale.

    Use this for price/height comparisons that must adapt across timeframes.
    ATR/range catches wick-heavy candles where body size alone understates the
    market's normal movement.
    """

    high = _num(frame["high"])
    low = _num(frame["low"])
    close = _num(frame["close"]).replace(0.0, np.nan)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_pct = (true_range / close.abs()).replace([np.inf, -np.inf], np.nan)
    return _rolling_scale_pct(atr_pct, window)


def _rolling_pivot_prominence_pct(frame: DataFrame, pivot_prefix: str, window: int) -> np.ndarray:
    """Return rolling confirmed-pivot prominence as a timeframe-aware scale.

    The source columns are sparse and no-lookahead: a pivot prominence only
    appears when that pivot is confirmed. The expanding fallback carries only
    historical confirmed prominence, never future pivots.
    """

    high_key = f"{pivot_prefix}_pivot_high_prominence_pct"
    low_key = f"{pivot_prefix}_pivot_low_prominence_pct"
    sources = []
    if high_key in frame:
        sources.append(_num(frame[high_key]))
    if low_key in frame:
        sources.append(_num(frame[low_key]))
    if not sources:
        return np.zeros(len(frame), dtype="float64")
    prominence_pct = pd.concat(sources, axis=1).mean(axis=1, skipna=True)
    prominence_pct = prominence_pct.where(prominence_pct > 0.0)
    return _rolling_scale_pct(prominence_pct, window, fallback=0.0)


def _rolling_scale_pct(series: Series, window: int, fallback: float | None = None) -> np.ndarray:
    clean = _num(series).replace([np.inf, -np.inf], np.nan)
    rolling_window = max(int(window), 3)
    min_periods = max(3, min(rolling_window, rolling_window // 3))
    rolling = clean.rolling(rolling_window, min_periods=min_periods).median()
    expanding = clean.expanding(min_periods=1).median()
    median = clean.median()
    fallback_value = float(fallback) if fallback is not None else (float(median) if np.isfinite(median) else 0.0)
    return rolling.combine_first(expanding).fillna(fallback_value).to_numpy(dtype="float64")


def _prior_opposite_pivot_context(
    pivot: np.ndarray,
    pivot_index: np.ndarray,
    first_x: float,
    first_price: float,
    window: int,
    *,
    top: bool,
) -> dict[str, float]:
    """Find the last opposite pivot that defines the move into P1."""

    valid = np.isfinite(pivot) & np.isfinite(pivot_index)
    candidates = np.flatnonzero(valid & (pivot_index < float(first_x)) & (pivot_index >= float(first_x) - float(window)))
    if candidates.size == 0 or not np.isfinite(first_price) or first_price == 0.0:
        return {"move_pct": 0.0, "price": np.nan, "index": np.nan}
    last = int(candidates[int(np.nanargmax(pivot_index[candidates]))])
    prior_price = float(pivot[last])
    if not np.isfinite(prior_price) or prior_price == 0.0:
        return {"move_pct": 0.0, "price": np.nan, "index": np.nan}
    if top:
        move_pct = max((float(first_price) - prior_price) / max(abs(float(first_price)), 1e-9), 0.0)
    else:
        move_pct = max((prior_price - float(first_price)) / max(abs(float(first_price)), 1e-9), 0.0)
    return {"move_pct": float(move_pct), "price": prior_price, "index": float(pivot_index[last])}


def _prior_impulse_score(
    close: np.ndarray,
    base_x: float,
    base_price: float,
    first_x: float,
    first_price: float,
    *,
    top: bool,
    min_bars: int,
) -> float:
    """Score whether the move into the first peak/trough was directional.

    Equal-level peaks inside a broad range can still have a local prior pivot
    move. This helper requires that move to behave like an impulse rather than
    a noisy sideways path.
    """

    if not np.isfinite([base_x, base_price, first_x, first_price]).all():
        return 0.0
    start = int(max(np.floor(float(base_x)), 0))
    stop = int(min(np.ceil(float(first_x)), len(close) - 1))
    span = stop - start
    if span < max(int(min_bars), 1):
        return 0.0
    base = float(base_price)
    first = float(first_price)
    if top:
        net = first - base
    else:
        net = base - first
    if net <= 0.0:
        return 0.0

    sample = np.asarray(close[start : stop + 1], dtype="float64")
    sample = sample[np.isfinite(sample)]
    if sample.size < 2:
        path = abs(first - base)
    else:
        path = abs(float(sample[0]) - base)
        path += float(np.nansum(np.abs(np.diff(sample))))
        path += abs(first - float(sample[-1]))
    return _clip_value(abs(net) / max(path, abs(net), 1e-9))


def _threshold_scale_pct(scale_pct: np.ndarray, row: int, minimum: float = 0.0) -> float:
    floor = max(float(minimum), 0.0)
    if row < len(scale_pct) and np.isfinite(scale_pct[row]):
        return max(float(scale_pct[row]), floor)
    finite = scale_pct[np.isfinite(scale_pct)]
    if finite.size:
        return max(float(np.nanmedian(finite)), floor)
    return floor


def _threshold_body_pct(body_pct: np.ndarray, row: int) -> float:
    return _threshold_scale_pct(body_pct, row, minimum=0.0005)


def _dynamic_height_tolerance_pct(
    static_ceiling_pct: float,
    body_ref_pct: float,
    atr_ref_pct: float,
    prominence_ref_pct: float,
    body_mult: float,
    atr_mult: float,
    prominence_mult: float,
) -> float:
    """Scale level/height tolerance by current market movement, not a hard pct.

    ``static_ceiling_pct`` is deliberately only a maximum sanity cap. The actual
    decision tolerance comes from rolling body, ATR/range, and confirmed pivot
    prominence so the same indicator can run on 1h, 4h, 1d, and 3d candles
    without one timeframe inheriting another timeframe's fixed percent logic.
    """

    cap = max(float(static_ceiling_pct), 1e-9)
    dynamic = max(
        float(body_ref_pct) * float(body_mult),
        float(atr_ref_pct) * float(atr_mult),
        float(prominence_ref_pct) * float(prominence_mult),
        1e-9,
    )
    return float(min(cap, dynamic))


def _dynamic_level_tolerance_pct(static_cap_pct: float, body_ref_pct: float, body_mult: float, floor_pct: float = 0.006) -> float:
    """Compatibility wrapper for older code paths.

    New indicators should call ``_dynamic_height_tolerance_pct`` and include at
    least body plus ATR/range scale. The static percentage must remain a cap,
    never the main cross-timeframe decision rule.
    """

    return _dynamic_height_tolerance_pct(
        static_cap_pct,
        body_ref_pct,
        0.0,
        0.0,
        body_mult,
        0.0,
        0.0,
    )


def _reaction_level_between(values: np.ndarray, first_x: float, second_x: float, *, find_low: bool) -> tuple[float, float]:
    start = int(max(np.floor(first_x), 0))
    stop = int(min(np.ceil(second_x), len(values) - 1))
    if stop <= start:
        return np.nan, np.nan
    sample = np.asarray(values[start : stop + 1], dtype="float64")
    if not np.isfinite(sample).any():
        return np.nan, np.nan
    local = int(np.nanargmin(sample) if find_low else np.nanargmax(sample))
    return float(start + local), float(sample[local])


def _top_base_is_held(body_low: np.ndarray, first_x: float, last_x: float, base_price: float, buffer_pct: float) -> bool:
    if not np.isfinite(base_price) or base_price <= 0.0:
        return False
    start = int(max(np.floor(first_x), 0))
    stop = int(min(np.ceil(last_x), len(body_low) - 1))
    if stop <= start:
        return False
    lows = np.asarray(body_low[start : stop + 1], dtype="float64")
    if not np.isfinite(lows).any():
        return False
    return float(np.nanmin(lows)) >= float(base_price) * (1.0 - float(buffer_pct))


def _bottom_base_is_held(body_high: np.ndarray, first_x: float, last_x: float, base_price: float, buffer_pct: float) -> bool:
    if not np.isfinite(base_price) or base_price <= 0.0:
        return False
    start = int(max(np.floor(first_x), 0))
    stop = int(min(np.ceil(last_x), len(body_high) - 1))
    if stop <= start:
        return False
    highs = np.asarray(body_high[start : stop + 1], dtype="float64")
    if not np.isfinite(highs).any():
        return False
    return float(np.nanmax(highs)) <= float(base_price) * (1.0 + float(buffer_pct))


def _top_p1_dominance_score(
    high_pivot: np.ndarray,
    high_index: np.ndarray,
    row: int,
    base_x: float,
    first_x: float,
    first_price: float,
    tolerance_pct: float,
) -> float:
    if not np.isfinite(base_x):
        return 0.0
    confirmed = np.arange(len(high_pivot)) <= int(row)
    mask = confirmed & np.isfinite(high_pivot) & np.isfinite(high_index) & (high_index >= float(base_x)) & (high_index < float(first_x))
    if not mask.any():
        return 1.0
    prior_max = float(np.nanmax(high_pivot[mask]))
    overshoot = max(prior_max - float(first_price), 0.0)
    if overshoot <= 0.0:
        return 1.0
    return _clip_value(1.0 - overshoot / max(abs(float(first_price)) * float(tolerance_pct), 1e-9))


def _bottom_p1_dominance_score(
    low_pivot: np.ndarray,
    low_index: np.ndarray,
    row: int,
    base_x: float,
    first_x: float,
    first_price: float,
    tolerance_pct: float,
) -> float:
    if not np.isfinite(base_x):
        return 0.0
    confirmed = np.arange(len(low_pivot)) <= int(row)
    mask = confirmed & np.isfinite(low_pivot) & np.isfinite(low_index) & (low_index >= float(base_x)) & (low_index < float(first_x))
    if not mask.any():
        return 1.0
    prior_min = float(np.nanmin(low_pivot[mask]))
    undershoot = max(float(first_price) - prior_min, 0.0)
    if undershoot <= 0.0:
        return 1.0
    return _clip_value(1.0 - undershoot / max(abs(float(first_price)) * float(tolerance_pct), 1e-9))


def _peak_retest_quality(
    similarity: float,
    depth_pct: float,
    min_depth: float,
    prior_move_pct: float,
    min_prior_move: float,
    turn_pct: float,
    dominance_score: float,
    between_cleanliness_score: float,
    span: float,
    min_bars: int,
    max_bars: int,
) -> float:
    span_mid = (float(min_bars) + float(max_bars)) / 2.0
    span_score = _clip_value(1.0 - abs(float(span) - span_mid) / max(span_mid, 1.0))
    depth_score = _clip_value(float(depth_pct) / max(float(min_depth) * 2.0, 1e-9))
    prior_score = _clip_value(float(prior_move_pct) / max(float(min_prior_move) * 2.0, 1e-9))
    turn_score = _clip_value(float(turn_pct) / max(float(min_depth), 1e-9))
    level_match_score = 0.50 + 0.50 * _clip_value(float(similarity))
    return _clip_value(
        0.24 * level_match_score
        + 0.22 * prior_score
        + 0.16 * depth_score
        + 0.12 * turn_score
        + 0.10 * span_score
        + 0.10 * float(dominance_score)
        + 0.06 * float(between_cleanliness_score)
    )


def _confirmed_micro_pivots(
    body_high: np.ndarray,
    body_low: np.ndarray,
    strength: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = len(body_high)
    pivot_high = np.full(rows, np.nan, dtype="float64")
    pivot_low = np.full(rows, np.nan, dtype="float64")
    high_index = np.full(rows, np.nan, dtype="float64")
    low_index = np.full(rows, np.nan, dtype="float64")
    wing = max(int(strength), 1)
    window = wing * 2 + 1
    if rows < window:
        return pivot_high, high_index, pivot_low, low_index

    for confirm_row in range(window - 1, rows):
        anchor = confirm_row - wing
        start = anchor - wing
        stop = anchor + wing + 1
        high_window = body_high[start:stop]
        low_window = body_low[start:stop]
        high_value = body_high[anchor]
        low_value = body_low[anchor]
        if np.isfinite(high_value) and np.isfinite(high_window).all() and high_value >= float(np.nanmax(high_window)):
            pivot_high[confirm_row] = float(high_value)
            high_index[confirm_row] = float(anchor)
        if np.isfinite(low_value) and np.isfinite(low_window).all() and low_value <= float(np.nanmin(low_window)):
            pivot_low[confirm_row] = float(low_value)
            low_index[confirm_row] = float(anchor)
    return pivot_high, high_index, pivot_low, low_index


def _combine_sparse_pivots(
    foundation_price: np.ndarray,
    foundation_index: np.ndarray,
    micro_price: np.ndarray,
    micro_index: np.ndarray,
) -> np.ndarray:
    out = micro_price.copy()
    foundation_valid = np.isfinite(foundation_price) & np.isfinite(foundation_index)
    out[foundation_valid] = foundation_price[foundation_valid]
    return out


def _combine_sparse_indexes(
    combined_price: np.ndarray,
    foundation_index: np.ndarray,
    micro_index: np.ndarray,
) -> np.ndarray:
    out = micro_index.copy()
    foundation_valid = np.isfinite(foundation_index)
    out[foundation_valid] = foundation_index[foundation_valid]
    out[~np.isfinite(combined_price)] = np.nan
    return out


def _slope_from_points(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return 0.0
    xv = x[valid].astype("float64")
    yv = y[valid].astype("float64")
    x_mean = float(np.mean(xv))
    y_mean = float(np.mean(yv))
    denominator = float(np.sum((xv - x_mean) ** 2))
    if denominator <= 0.0:
        return 0.0
    return float(np.sum((xv - x_mean) * (yv - y_mean)) / denominator)


def _clip_value(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(max(value, 0.0), 1.0))


def _nanmax_pair(left: float, right: float) -> float:
    if np.isfinite(left) and np.isfinite(right):
        return float(max(left, right))
    if np.isfinite(left):
        return float(left)
    if np.isfinite(right):
        return float(right)
    return np.nan


def _num(series: Series) -> Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")

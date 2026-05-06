from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pandas import Series


PivotMethod = Literal["body", "atr_zigzag"]


def build_clean_pivot_source(
    *,
    body_high: Series,
    body_low: Series,
    atr: Series,
    bar_index: Series,
    strength: int,
    method: PivotMethod = "body",
    zigzag_atr_mult: float = 1.75,
    touch_atr_mult: float = 0.30,
    touch_pct: float = 0.0015,
    min_prominence_atr: float = 0.0,
    min_prominence_pct: float = 0.0,
    min_pivot_spacing_bars: int = 1,
    min_pivot_distance_atr: float = 0.0,
    min_pivot_distance_pct: float = 0.0,
) -> dict[str, Series]:
    """Return no-lookahead cleaned body pivots used by structure indicators.

    Pivots are emitted only after confirmation. Same-side pivot runs are
    collapsed and emitted when the opposite-side pivot confirms, so downstream
    indicators see the cleaned pivot only when it is knowable.
    """

    if method == "atr_zigzag":
        raw = _confirmed_atr_zigzag_pivots(
            body_high=body_high,
            body_low=body_low,
            atr=atr,
            bar_index=bar_index,
            atr_mult=float(zigzag_atr_mult),
            touch_atr_mult=float(touch_atr_mult),
            touch_pct=float(touch_pct),
        )
    else:
        raw = _confirmed_body_pivots(
            body_high=body_high,
            body_low=body_low,
            atr=atr,
            bar_index=bar_index,
            strength=int(strength),
        )

    filtered = _filter_raw_pivots(
        raw,
        atr=atr,
        min_prominence_atr=float(min_prominence_atr),
        min_prominence_pct=float(min_prominence_pct),
        min_pivot_spacing_bars=int(min_pivot_spacing_bars),
        min_pivot_distance_atr=float(min_pivot_distance_atr),
        min_pivot_distance_pct=float(min_pivot_distance_pct),
    )
    return _clean_alternating_pivots(filtered, body_high.index)


def _confirmed_body_pivots(
    *,
    body_high: Series,
    body_low: Series,
    atr: Series,
    bar_index: Series,
    strength: int,
) -> dict[str, Series]:
    window = int(strength) * 2 + 1
    high_window_max = body_high.rolling(window, min_periods=window).max()
    high_window_min = body_high.rolling(window, min_periods=window).min()
    low_window_max = body_low.rolling(window, min_periods=window).max()
    low_window_min = body_low.rolling(window, min_periods=window).min()

    high_candidate = body_high.shift(strength)
    low_candidate = body_low.shift(strength)
    high_atr = atr.shift(strength).replace(0.0, np.nan)
    low_atr = atr.shift(strength).replace(0.0, np.nan)
    high_prominence = ((high_candidate - high_window_min) / high_atr).replace([np.inf, -np.inf], np.nan)
    low_prominence = ((low_window_max - low_candidate) / low_atr).replace([np.inf, -np.inf], np.nan)
    high_prominence_pct = _safe_div(high_candidate - high_window_min, high_candidate)
    low_prominence_pct = _safe_div(low_window_max - low_candidate, low_candidate)
    high_is_pivot = high_candidate.notna() & high_candidate.ge(high_window_max)
    low_is_pivot = low_candidate.notna() & low_candidate.le(low_window_min)

    return {
        "pivot_high": high_candidate.where(high_is_pivot),
        "pivot_low": low_candidate.where(low_is_pivot),
        "pivot_high_index": (bar_index - float(strength)).where(high_is_pivot),
        "pivot_low_index": (bar_index - float(strength)).where(low_is_pivot),
        "pivot_high_available_index": bar_index.where(high_is_pivot),
        "pivot_low_available_index": bar_index.where(low_is_pivot),
        "pivot_high_prominence": high_prominence.where(high_is_pivot),
        "pivot_low_prominence": low_prominence.where(low_is_pivot),
        "pivot_high_prominence_pct": high_prominence_pct.where(high_is_pivot),
        "pivot_low_prominence_pct": low_prominence_pct.where(low_is_pivot),
        "pivot_high_score": pd.Series(np.where(high_is_pivot, 0.35, np.nan), index=body_high.index),
        "pivot_low_score": pd.Series(np.where(low_is_pivot, 0.35, np.nan), index=body_high.index),
    }


def _confirmed_atr_zigzag_pivots(
    *,
    body_high: Series,
    body_low: Series,
    atr: Series,
    bar_index: Series,
    atr_mult: float,
    touch_atr_mult: float,
    touch_pct: float,
) -> dict[str, Series]:
    rows = len(body_high)
    out = {
        "pivot_high": np.full(rows, np.nan, dtype="float64"),
        "pivot_low": np.full(rows, np.nan, dtype="float64"),
        "pivot_high_index": np.full(rows, np.nan, dtype="float64"),
        "pivot_low_index": np.full(rows, np.nan, dtype="float64"),
        "pivot_high_available_index": np.full(rows, np.nan, dtype="float64"),
        "pivot_low_available_index": np.full(rows, np.nan, dtype="float64"),
        "pivot_high_prominence": np.full(rows, np.nan, dtype="float64"),
        "pivot_low_prominence": np.full(rows, np.nan, dtype="float64"),
        "pivot_high_prominence_pct": np.full(rows, np.nan, dtype="float64"),
        "pivot_low_prominence_pct": np.full(rows, np.nan, dtype="float64"),
        "pivot_high_score": np.full(rows, np.nan, dtype="float64"),
        "pivot_low_score": np.full(rows, np.nan, dtype="float64"),
    }
    if rows < 3:
        return {name: pd.Series(values, index=body_high.index, dtype="float64") for name, values in out.items()}

    high_values = body_high.to_numpy(dtype="float64")
    low_values = body_low.to_numpy(dtype="float64")
    atr_values = atr.replace(0.0, np.nan).to_numpy(dtype="float64")
    bar_values = bar_index.to_numpy(dtype="float64")

    direction = 0
    extreme_price = float((high_values[0] + low_values[0]) / 2.0)
    extreme_row = 0
    extreme_side = ""
    touches = 1

    for row in range(1, rows):
        if not np.isfinite(high_values[row]) or not np.isfinite(low_values[row]):
            continue
        threshold = float(atr_values[row] * atr_mult) if np.isfinite(atr_values[row]) else 0.0
        if threshold <= 0.0:
            threshold = max(abs(float(high_values[row])) * float(touch_pct), 1e-9)

        if direction >= 0:
            if high_values[row] >= extreme_price:
                if abs(high_values[row] - extreme_price) <= max(threshold * float(touch_atr_mult), abs(high_values[row]) * float(touch_pct)):
                    touches += 1
                else:
                    touches = 1
                extreme_price = float(high_values[row])
                extreme_row = row
                extreme_side = "high"
            elif extreme_price - low_values[row] >= threshold and extreme_side == "high":
                _emit_zigzag_high(out, extreme_row, row, extreme_price, bar_values, atr_values, touches)
                direction = -1
                extreme_price = float(low_values[row])
                extreme_row = row
                extreme_side = "low"
                touches = 1

        if direction <= 0:
            if low_values[row] <= extreme_price:
                if abs(low_values[row] - extreme_price) <= max(threshold * float(touch_atr_mult), abs(low_values[row]) * float(touch_pct)):
                    touches += 1
                else:
                    touches = 1
                extreme_price = float(low_values[row])
                extreme_row = row
                extreme_side = "low"
            elif high_values[row] - extreme_price >= threshold and extreme_side == "low":
                _emit_zigzag_low(out, extreme_row, row, extreme_price, bar_values, atr_values, touches)
                direction = 1
                extreme_price = float(high_values[row])
                extreme_row = row
                extreme_side = "high"
                touches = 1

    return {name: pd.Series(values, index=body_high.index, dtype="float64") for name, values in out.items()}


def _emit_zigzag_high(
    out: dict[str, np.ndarray],
    anchor_row: int,
    confirm_row: int,
    price: float,
    bar_values: np.ndarray,
    atr_values: np.ndarray,
    touch_count: int,
) -> None:
    row = int(confirm_row)
    atr_value = atr_values[anchor_row] if np.isfinite(atr_values[anchor_row]) and atr_values[anchor_row] > 0.0 else np.nan
    prominence = abs(float(price) - float(np.nanmin([price, price - atr_value if np.isfinite(atr_value) else price]))) / max(atr_value, 1e-9)
    out["pivot_high"][row] = float(price)
    out["pivot_high_index"][row] = float(bar_values[anchor_row])
    out["pivot_high_available_index"][row] = float(bar_values[row])
    out["pivot_high_prominence"][row] = float(prominence)
    out["pivot_high_prominence_pct"][row] = 0.0
    out["pivot_high_score"][row] = _pivot_quality_score(float(prominence), int(touch_count))


def _emit_zigzag_low(
    out: dict[str, np.ndarray],
    anchor_row: int,
    confirm_row: int,
    price: float,
    bar_values: np.ndarray,
    atr_values: np.ndarray,
    touch_count: int,
) -> None:
    row = int(confirm_row)
    atr_value = atr_values[anchor_row] if np.isfinite(atr_values[anchor_row]) and atr_values[anchor_row] > 0.0 else np.nan
    prominence = abs(float(price) - float(np.nanmax([price, price + atr_value if np.isfinite(atr_value) else price]))) / max(atr_value, 1e-9)
    out["pivot_low"][row] = float(price)
    out["pivot_low_index"][row] = float(bar_values[anchor_row])
    out["pivot_low_available_index"][row] = float(bar_values[row])
    out["pivot_low_prominence"][row] = float(prominence)
    out["pivot_low_prominence_pct"][row] = 0.0
    out["pivot_low_score"][row] = _pivot_quality_score(float(prominence), int(touch_count))


def _filter_raw_pivots(
    raw: dict[str, Series],
    *,
    atr: Series,
    min_prominence_atr: float,
    min_prominence_pct: float,
    min_pivot_spacing_bars: int,
    min_pivot_distance_atr: float,
    min_pivot_distance_pct: float,
) -> dict[str, Series]:
    out = {key: value.copy() for key, value in raw.items()}
    for side in ("high", "low"):
        price = out[f"pivot_{side}"]
        pivot_index = out[f"pivot_{side}_index"]
        prominence = out[f"pivot_{side}_prominence"]
        prominence_pct = out[f"pivot_{side}_prominence_pct"]
        confirmed = (
            price.notna()
            & pivot_index.notna()
            & prominence.fillna(0.0).ge(float(min_prominence_atr))
            & prominence_pct.fillna(0.0).ge(float(min_prominence_pct))
        )
        confirmed = _filter_pivot_noise_values(
            confirmed,
            price,
            pivot_index,
            atr.reindex(price.index),
            int(min_pivot_spacing_bars),
            float(min_pivot_distance_atr),
            float(min_pivot_distance_pct),
        )
        for suffix in (
            "",
            "_index",
            "_available_index",
            "_prominence",
            "_prominence_pct",
            "_score",
        ):
            key = f"pivot_{side}{suffix}"
            out[key] = out[key].where(confirmed)
    return out


def _filter_pivot_noise_values(
    confirmed: Series,
    candidate_price: Series,
    candidate_index: Series,
    candidate_atr: Series,
    min_pivot_spacing_bars: int,
    min_pivot_distance_atr: float,
    min_pivot_distance_pct: float,
) -> Series:
    raw_price = candidate_price.where(confirmed)
    raw_index = candidate_index.where(confirmed)
    prev_price = raw_price.dropna().shift(1).reindex(candidate_price.index).ffill()
    prev_index = raw_index.dropna().shift(1).reindex(candidate_price.index).ffill()
    spacing_ok = (raw_index - prev_index).ge(float(min_pivot_spacing_bars)) | prev_index.isna()
    distance = (raw_price - prev_price).abs()
    atr_ok = distance.ge(candidate_atr * float(min_pivot_distance_atr)) | prev_price.isna()
    pct_ok = (distance / raw_price.abs()).ge(float(min_pivot_distance_pct)) | prev_price.isna()
    return (confirmed & spacing_ok.fillna(False) & atr_ok.fillna(False) & pct_ok.fillna(False)).fillna(False)


def _clean_alternating_pivots(raw: dict[str, Series], index: pd.Index) -> dict[str, Series]:
    rows = len(index)
    out = {
        "pivot_high": np.full(rows, np.nan, dtype="float64"),
        "pivot_low": np.full(rows, np.nan, dtype="float64"),
        "pivot_high_index": np.full(rows, np.nan, dtype="float64"),
        "pivot_low_index": np.full(rows, np.nan, dtype="float64"),
        "pivot_high_available_index": np.full(rows, np.nan, dtype="float64"),
        "pivot_low_available_index": np.full(rows, np.nan, dtype="float64"),
        "pivot_high_prominence": np.full(rows, np.nan, dtype="float64"),
        "pivot_low_prominence": np.full(rows, np.nan, dtype="float64"),
        "pivot_high_prominence_pct": np.full(rows, np.nan, dtype="float64"),
        "pivot_low_prominence_pct": np.full(rows, np.nan, dtype="float64"),
        "pivot_high_score": np.full(rows, np.nan, dtype="float64"),
        "pivot_low_score": np.full(rows, np.nan, dtype="float64"),
    }
    events: list[dict[str, float | int | str]] = []
    for row in np.flatnonzero(raw["pivot_high"].notna().to_numpy()):
        events.append(
            {
                "row": int(row),
                "side": "high",
                "price": float(raw["pivot_high"].iloc[row]),
                "index": float(raw["pivot_high_index"].iloc[row]),
                "available_index": float(raw["pivot_high_available_index"].iloc[row]),
                "prominence": float(raw["pivot_high_prominence"].iloc[row]),
                "prominence_pct": float(raw["pivot_high_prominence_pct"].iloc[row]),
                "score": float(raw["pivot_high_score"].iloc[row]),
            }
        )
    for row in np.flatnonzero(raw["pivot_low"].notna().to_numpy()):
        events.append(
            {
                "row": int(row),
                "side": "low",
                "price": float(raw["pivot_low"].iloc[row]),
                "index": float(raw["pivot_low_index"].iloc[row]),
                "available_index": float(raw["pivot_low_available_index"].iloc[row]),
                "prominence": float(raw["pivot_low_prominence"].iloc[row]),
                "prominence_pct": float(raw["pivot_low_prominence_pct"].iloc[row]),
                "score": float(raw["pivot_low_score"].iloc[row]),
            }
        )
    events.sort(key=lambda item: (int(item["row"]), 0 if item["side"] == "low" else 1))

    pending: dict[str, float | int | str] | None = None
    for event in events:
        if pending is None:
            pending = {**event, "count": 1}
            continue
        if event["side"] == pending["side"]:
            pending["count"] = int(pending["count"]) + 1
            if (event["side"] == "high" and float(event["price"]) >= float(pending["price"])) or (
                event["side"] == "low" and float(event["price"]) <= float(pending["price"])
            ):
                pending.update({key: event[key] for key in ("row", "side", "price", "index", "available_index", "prominence", "prominence_pct", "score")})
            continue
        _emit_clean_pivot(out, pending, int(event["row"]))
        pending = {**event, "count": 1}

    out["pivot_high_confirmed"] = np.isfinite(out["pivot_high"])
    out["pivot_low_confirmed"] = np.isfinite(out["pivot_low"])
    return {
        name: pd.Series(values, index=index, dtype="bool" if name.endswith("confirmed") else "float64")
        for name, values in out.items()
    }


def _emit_clean_pivot(out: dict[str, np.ndarray], pending: dict[str, float | int | str], available_row: int) -> None:
    side = str(pending["side"])
    price = float(pending["price"])
    anchor_index = float(pending["index"])
    prominence = float(pending["prominence"])
    prominence_pct = float(pending["prominence_pct"])
    score = max(float(pending["score"]), _pivot_quality_score(prominence, int(pending["count"])))
    if side == "high":
        out["pivot_high"][available_row] = price
        out["pivot_high_index"][available_row] = anchor_index
        out["pivot_high_available_index"][available_row] = float(available_row)
        out["pivot_high_prominence"][available_row] = prominence
        out["pivot_high_prominence_pct"][available_row] = prominence_pct
        out["pivot_high_score"][available_row] = score
    else:
        out["pivot_low"][available_row] = price
        out["pivot_low_index"][available_row] = anchor_index
        out["pivot_low_available_index"][available_row] = float(available_row)
        out["pivot_low_prominence"][available_row] = prominence
        out["pivot_low_prominence_pct"][available_row] = prominence_pct
        out["pivot_low_score"][available_row] = score


def _pivot_quality_score(prominence: float, touch_count: int) -> float:
    prominence_score = min(max(float(prominence), 0.0) / 4.0, 1.0)
    touch_score = min(max(int(touch_count), 1) / 4.0, 1.0)
    return float(min(1.0, 0.35 + 0.35 * prominence_score + 0.30 * touch_score))


def _safe_div(numerator: Series, denominator: Series) -> Series:
    denominator = denominator.where(denominator.abs() > 0.0, np.nan)
    return numerator / denominator


__all__ = ["PivotMethod", "build_clean_pivot_source"]

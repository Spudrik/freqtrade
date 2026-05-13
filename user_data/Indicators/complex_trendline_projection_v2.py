from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

try:
    from .pivot_foundation import build_clean_pivot_source
except Exception:  # pragma: no cover - optional fallback for standalone notebooks
    from pivot_foundation import build_clean_pivot_source  # type: ignore[no-redef]

LineSide = Literal["resistance", "support"]

_PROFILE_FIELDS = (
    "max_anchor_bars",
    "max_projection_bars",
    "max_active_line_distance_atr_mult",
)
_TIMEFRAME_PROFILES: dict[str, dict[str, object]] = {
    "1h": {
        "max_anchor_bars": 50,
        "max_projection_bars": 50,
        "max_active_line_distance_atr_mult": 6.0,
    },
    "4h": {
        "max_anchor_bars": 50,
        "max_projection_bars": 50,
        "max_active_line_distance_atr_mult": 6.0,
    },
    "8h": {
        "max_anchor_bars": 80,
        "max_projection_bars": 80,
        "max_active_line_distance_atr_mult": 7.0,
    },
    "1d": {
        "max_anchor_bars": 80,
        "max_projection_bars": 90,
        "max_active_line_distance_atr_mult": 8.0,
    },
    "3d": {
        "max_anchor_bars": 90,
        "max_projection_bars": 110,
        "max_active_line_distance_atr_mult": 9.0,
    },
}
_TIMEFRAME_ALIASES = {
    "1h": "1h",
    "60m": "1h",
    "4h": "4h",
    "240m": "4h",
    "8h": "8h",
    "480m": "8h",
    "1d": "1d",
    "1day": "1d",
    "3d": "3d",
    "3day": "3d",
}


@dataclass(frozen=True)
class TrendlineProjectionV2Config:
    """First-pass cleaned-pivot trendline generator.

    This version keeps the strategy-facing path fixed: generate cleaned pivot
    pairs, remove invalid p1->p2 spans, remove extreme angles/spans, join
    endpoint continuations, delete contained duplicate lines, absorb nearby
    weaker lines into stronger lines, then export the strongest support and
    resistance slots.

    Tunable first-pass levers:
    - ``pivot_strength``: confirmed body pivot strength. A pivot is emitted only
      after this many candles have confirmed it.
    - ``max_slope_pct_per_bar``: removes extreme angle lines, normalized by
      price so the same rule can work across coins/timeframes.
    - ``min_anchor_bars``: removes pivot pairs that are too close together.
    - ``max_anchor_bars``: removes seed lines whose p1->p2 span is too long to
      be a clean initial trendline definition.
    - ``max_projection_bars``: caps each line after its second anchor.
    - ``max_active_line_distance_*``: suppresses projected lines once price has
      moved too far away for the line to be useful on the active timeframe.
    - ``duplicate_*``: same-side lines with similar slope and a latest pivot
      close to the stronger line are treated as one structure. The merge gate
      deliberately uses the most recent pivot, not the line origin, so old
      detail is not collapsed just because two lines started near each other.
      Absorbed pivots can refit the surviving line, but only while the refit
      remains anchored to real pivots and all absorbed pivots stay close to the
      resulting straight line.
    - ``anchor_break_*``: nuanced p1->p2 body-break handling. Minor body breaks
      near either anchor, or short/shallow interior breaks, can be allowed so
      valid human-looking lines are not deleted by pivot granularity.

    Strategy-facing outputs:
    - ``*_resistance_line_rankN`` / ``*_support_line_rankN``: compact ranked
      local trendline slots. Channels and higher-level patterns are intentionally
      split into separate consumers of these columns.
    """

    output_prefix: str = "tlv2"
    timeframe: str = "4h"
    pivot_strength: int = 2
    raw_line_output_count: int = 3

    min_anchor_bars: int = 10
    max_anchor_bars: int = 50
    min_pivot_prominence_atr: float = 0.35
    max_slope_pct_per_bar: float = 0.004
    max_projection_bars: int = 50
    max_active_line_distance_pct: float = 0.08
    max_active_line_distance_atr_mult: float = 6.0

    absorb_width_scale: float = 0.30
    duplicate_line_proximity_pct: float = 0.0060
    duplicate_angle_tolerance_pct: float = 0.20
    duplicate_min_overlap_bars: int = 8
    anchor_break_edge_bars: int = 3
    anchor_break_max_run: int = 2
    anchor_break_max_pct: float = 0.0035
    anchor_break_max_atr_mult: float = 0.40
    projection_break_candles: int = 5
    projection_break_tolerance_pct: float = 0.0015
    projection_break_atr_mult: float = 0.25

    join_angle_tolerance_pct: float = 0.15
    join_midpoint_tolerance_pct: float = 0.012
    max_joined_pivots: int = 6
    shared_pivot_merge_min: int = 2
    nearby_merge_proximity_pct: float = 0.0025


def add_trendline_projection_v2(
    dataframe: DataFrame,
    config: TrendlineProjectionV2Config | None = None,
    **overrides: object,
) -> DataFrame:
    """Append first-pass v2 trendline evidence columns."""

    cfg = _resolve_config(config, overrides)
    _validate_config(cfg)
    _validate_dataframe(dataframe)

    frame = dataframe.copy()
    base = _base_inputs(frame, cfg)
    p = cfg.output_prefix

    new_cols: dict[str, Series] = {
        f"{p}_atr": base["atr"],
        f"{p}_body_high": base["body_high"],
        f"{p}_body_low": base["body_low"],
        f"{p}_pivot_high": base["pivot_high"],
        f"{p}_pivot_low": base["pivot_low"],
        f"{p}_pivot_high_index": base["pivot_high_index"],
        f"{p}_pivot_low_index": base["pivot_low_index"],
        f"{p}_pivot_high_available_index": base["pivot_high_available_index"],
        f"{p}_pivot_low_available_index": base["pivot_low_available_index"],
        f"{p}_pivot_high_prominence": base["pivot_high_prominence"],
        f"{p}_pivot_low_prominence": base["pivot_low_prominence"],
        f"{p}_pivot_high_score": base["pivot_high_score"],
        f"{p}_pivot_low_score": base["pivot_low_score"],
    }
    sequence_candidates = _build_sequence_candidate_table(base, cfg)
    raw_resistance_cols = _rank_sequence_candidate_columns(
        sequence_candidates,
        base,
        "resistance",
        "resistance",
        cfg,
        slots=int(cfg.raw_line_output_count),
    )
    raw_support_cols = _rank_sequence_candidate_columns(
        sequence_candidates,
        base,
        "support",
        "support",
        cfg,
        slots=int(cfg.raw_line_output_count),
    )
    new_cols.update(raw_resistance_cols)
    new_cols.update(raw_support_cols)
    existing = [col for col in frame.columns if str(col).startswith(f"{p}_")]
    clean = frame.drop(columns=existing).copy() if existing else frame.copy()
    return pd.concat([clean, pd.DataFrame(new_cols, index=frame.index)], axis=1)


def _base_inputs(frame: DataFrame, cfg: TrendlineProjectionV2Config) -> dict[str, Series]:
    open_ = _num(frame, "open")
    close = _num(frame, "close")
    high = _num(frame, "high")
    low = _num(frame, "low")
    body_high = pd.concat([open_, close], axis=1).max(axis=1)
    body_low = pd.concat([open_, close], axis=1).min(axis=1)
    atr = _atr(frame, 14)
    bar_index = pd.Series(np.arange(len(frame), dtype="float64"), index=frame.index)
    pivots = build_clean_pivot_source(
        body_high=body_high,
        body_low=body_low,
        atr=atr,
        bar_index=bar_index,
        strength=int(cfg.pivot_strength),
    )
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "body_high": body_high,
        "body_low": body_low,
        "atr": atr,
        "bar_index": bar_index,
        **pivots,
    }


def _build_sequence_candidate_table(base: dict[str, Series], cfg: TrendlineProjectionV2Config) -> DataFrame:
    """Run the fixed line sequence requested for strategy-facing output."""

    raw_candidates = pd.concat(
        [
            _candidate_lines(base, "resistance", cfg),
            _candidate_lines(base, "support", cfg),
        ],
        ignore_index=True,
    )
    if raw_candidates.empty:
        return raw_candidates

    joined = _join_endpoint_continuations(raw_candidates, cfg)
    subset_clean = _absorb_shared_pivot_smaller_lines(joined, cfg)
    merged = _absorb_nearby_weaker_lines(subset_clean, float(cfg.nearby_merge_proximity_pct), cfg)
    return merged.reset_index(drop=True)


def _rank_sequence_candidate_columns(
    candidates: DataFrame,
    base: dict[str, Series],
    side: LineSide,
    output_side: str,
    cfg: TrendlineProjectionV2Config,
    *,
    slots: int,
) -> dict[str, Series]:
    index = base["close"].index
    rows = len(index)
    if rows == 0 or candidates.empty:
        return _empty_ranked_line_columns(index, cfg.output_prefix, output_side, int(slots))

    side_candidates = candidates[candidates["side"].eq(side)].copy()
    if side_candidates.empty:
        return _empty_ranked_line_columns(index, cfg.output_prefix, output_side, int(slots))

    side_candidates = side_candidates.sort_values(
        ["absorbed_pivot_count", "pivot_count", "score", "span"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    top = _empty_top_line_arrays(rows, int(slots))
    x_values = np.arange(rows, dtype="float64")
    close_values = base["close"].to_numpy(dtype="float64")
    atr_values = base["atr"].to_numpy(dtype="float64")
    for candidate_index, candidate in side_candidates.iterrows():
        live_start = int(np.ceil(float(candidate.get("live_start", candidate["x_new"]))))
        end = int(np.floor(float(candidate["projection_end"])))
        start = max(live_start, 0)
        end = min(end, rows - 1)
        if end < start:
            continue
        xs = x_values[start : end + 1]
        line = float(candidate["intercept"]) + float(candidate["slope"]) * xs
        score = np.full(rows, -np.inf, dtype="float64")
        close_segment = close_values[start : end + 1]
        atr_segment = atr_values[start : end + 1]
        active_distance = np.maximum(
            np.abs(close_segment) * float(cfg.max_active_line_distance_pct),
            atr_segment * float(cfg.max_active_line_distance_atr_mult),
        )
        valid_line = np.isfinite(line) & np.isfinite(active_distance) & (np.abs(line - close_segment) <= active_distance)
        if not np.any(valid_line):
            continue
        segment = slice(start, end + 1)
        score[segment] = float(candidate["score"])
        score[segment] = np.where(valid_line, score[segment], -np.inf)

        line_values = np.full(rows, np.nan, dtype="float64")
        line_values[segment] = np.where(valid_line, line, np.nan)
        line_id = float(candidate_index + 1)
        metrics = {
            "line": line_values,
            "slope": _constant_metric(rows, float(candidate["slope"]), start, end),
            "slope_pct": _constant_metric(rows, float(candidate["slope_pct"]), start, end),
            "anchor_old_index": _constant_metric(rows, float(candidate["x_old"]), start, end),
            "anchor_new_index": _constant_metric(rows, float(candidate["x_new"]), start, end),
            "projection_end_index": _constant_metric(rows, float(candidate["projection_end"]), start, end),
            "line_id": _constant_metric(rows, line_id, start, end),
            "pivot_count": _constant_metric(rows, float(candidate.get("pivot_count", 2.0)), start, end),
            "absorbed_pivot_count": _constant_metric(
                rows,
                float(candidate.get("absorbed_pivot_count", candidate.get("pivot_count", 2.0))),
                start,
                end,
            ),
            "line_width": _constant_metric(rows, float(candidate.get("half_width", 0.0)), start, end),
            "last_confirm_index": _constant_metric(
                rows,
                float(candidate.get("x_end", candidate["x_new"])),
                start,
                end,
            ),
        }
        _insert_top_line_candidate(top, score, metrics)

    return _top_line_arrays_to_columns(top, index, cfg.output_prefix, output_side)


def _constant_metric(rows: int, value: float, start: int, end: int) -> np.ndarray:
    out = np.full(rows, np.nan, dtype="float64")
    if np.isfinite(value) and end >= start:
        out[start : end + 1] = value
    return out


def _empty_top_line_arrays(rows: int, slots: int) -> dict[str, np.ndarray]:
    top = {
        "score": np.full((rows, slots), -np.inf, dtype="float64"),
        "line": np.full((rows, slots), np.nan, dtype="float64"),
        "slope": np.full((rows, slots), np.nan, dtype="float64"),
        "slope_pct": np.full((rows, slots), np.nan, dtype="float64"),
        "anchor_old_index": np.full((rows, slots), np.nan, dtype="float64"),
        "anchor_new_index": np.full((rows, slots), np.nan, dtype="float64"),
        "projection_end_index": np.full((rows, slots), np.nan, dtype="float64"),
        "line_id": np.full((rows, slots), np.nan, dtype="float64"),
        "pivot_count": np.full((rows, slots), np.nan, dtype="float64"),
        "absorbed_pivot_count": np.full((rows, slots), np.nan, dtype="float64"),
        "line_width": np.full((rows, slots), np.nan, dtype="float64"),
        "last_confirm_index": np.full((rows, slots), np.nan, dtype="float64"),
    }
    return top


def _insert_top_line_candidate(top: dict[str, np.ndarray], score: np.ndarray, metrics: dict[str, np.ndarray]) -> None:
    inserted = np.zeros(score.shape[0], dtype=bool)
    slots = top["score"].shape[1]
    for rank in range(slots):
        mask = (~inserted) & (score > top["score"][:, rank])
        if not np.any(mask):
            continue
        if rank < slots - 1:
            for key, values in top.items():
                values[mask, rank + 1 :] = values[mask, rank:-1]
        top["score"][mask, rank] = score[mask]
        for key, candidate_values in metrics.items():
            top[key][mask, rank] = candidate_values[mask]
        inserted[mask] = True
        if np.all(inserted):
            break


def _top_line_arrays_to_columns(
    top: dict[str, np.ndarray],
    index: pd.Index,
    prefix: str,
    side: str,
) -> dict[str, Series]:
    slots = top["score"].shape[1]
    out: dict[str, Series] = {}
    for rank in range(slots):
        valid = np.isfinite(top["score"][:, rank]) & (top["score"][:, rank] > -np.inf)
        line = np.where(valid, top["line"][:, rank], np.nan)
        line_series = pd.Series(line, index=index, dtype="float64")
        out[f"{prefix}_{side}_line_rank{rank}"] = line_series
        out[f"{prefix}_{side}_score_rank{rank}"] = pd.Series(
            np.where(valid, top["score"][:, rank], np.nan),
            index=index,
            dtype="float64",
        )
        for metric in (
            "slope",
            "slope_pct",
            "anchor_old_index",
            "anchor_new_index",
            "projection_end_index",
            "line_id",
            "pivot_count",
            "absorbed_pivot_count",
            "line_width",
            "last_confirm_index",
        ):
            out[f"{prefix}_{side}_{metric}_rank{rank}"] = pd.Series(
                np.where(valid, top[metric][:, rank], np.nan),
                index=index,
                dtype="float64",
            )
    return out


def _empty_ranked_line_columns(index: pd.Index, prefix: str, side: str, slots: int) -> dict[str, Series]:
    out: dict[str, Series] = {}
    for rank in range(slots):
        out.update(_empty_line_columns(index, prefix, side, rank))
    return out


def _empty_line_columns(index: pd.Index, prefix: str, side: str, rank: int) -> dict[str, Series]:
    nan = pd.Series(np.nan, index=index, dtype="float64")
    return {
        f"{prefix}_{side}_line_rank{rank}": nan,
        f"{prefix}_{side}_score_rank{rank}": nan,
        f"{prefix}_{side}_slope_rank{rank}": nan,
        f"{prefix}_{side}_slope_pct_rank{rank}": nan,
        f"{prefix}_{side}_anchor_old_index_rank{rank}": nan,
        f"{prefix}_{side}_anchor_new_index_rank{rank}": nan,
        f"{prefix}_{side}_projection_end_index_rank{rank}": nan,
        f"{prefix}_{side}_line_id_rank{rank}": nan,
        f"{prefix}_{side}_pivot_count_rank{rank}": nan,
        f"{prefix}_{side}_absorbed_pivot_count_rank{rank}": nan,
        f"{prefix}_{side}_line_width_rank{rank}": nan,
        f"{prefix}_{side}_last_confirm_index_rank{rank}": nan,
    }


def _candidate_lines(
    base: dict[str, Series],
    side: LineSide,
    cfg: TrendlineProjectionV2Config,
    *,
    min_anchor_bars: int | None = None,
    max_anchor_bars: int | None = None,
    max_projection_bars: int | None = None,
    min_pivot_prominence_atr: float | None = None,
    line_kind: str = "raw",
) -> DataFrame:
    if side == "resistance":
        pivot_price = base["pivot_high"]
        pivot_index = base["pivot_high_index"]
        pivot_available = base["pivot_high_available_index"]
        pivot_prominence = base["pivot_high_prominence"]
        pivot_score = base["pivot_high_score"]
    else:
        pivot_price = base["pivot_low"]
        pivot_index = base["pivot_low_index"]
        pivot_available = base["pivot_low_available_index"]
        pivot_prominence = base["pivot_low_prominence"]
        pivot_score = base["pivot_low_score"]

    prominence_floor = float(cfg.min_pivot_prominence_atr if min_pivot_prominence_atr is None else min_pivot_prominence_atr)
    pivot_frame = pd.DataFrame(
        {
            "price": pivot_price,
            "anchor_index": pivot_index,
            "available_index": pivot_available,
            "prominence": pivot_prominence,
            "score": pivot_score,
        }
    )
    pivot_frame = pivot_frame[
        pivot_frame["price"].notna()
        & pivot_frame["anchor_index"].notna()
        & pivot_frame["available_index"].notna()
        & pivot_frame["prominence"].fillna(0.0).ge(prominence_floor)
    ].dropna(subset=["price", "anchor_index", "available_index", "prominence"])
    prices = pivot_frame["price"].to_numpy(dtype="float64")
    indexes = pivot_frame["anchor_index"].to_numpy(dtype="float64")
    available_indexes = pivot_frame["available_index"].to_numpy(dtype="float64")
    prominences = pivot_frame["prominence"].to_numpy(dtype="float64")
    pivot_scores = pivot_frame["score"].fillna(0.0).to_numpy(dtype="float64")
    rows: list[dict[str, float | str]] = []
    anchor_min = float(cfg.min_anchor_bars if min_anchor_bars is None else min_anchor_bars)
    anchor_cap = float(cfg.max_anchor_bars if max_anchor_bars is None else max_anchor_bars)
    projection_cap = float(cfg.max_projection_bars if max_projection_bars is None else max_projection_bars)
    for old_pos in range(len(prices) - 1):
        for new_pos in range(old_pos + 1, len(prices)):
            x_old = indexes[old_pos]
            x_new = indexes[new_pos]
            span = x_new - x_old
            if span < anchor_min or span > anchor_cap:
                continue
            p_old = prices[old_pos]
            p_new = prices[new_pos]
            slope = (p_new - p_old) / span
            price_ref = max(abs(p_new), 1e-9)
            slope_pct = abs(slope) / price_ref
            if not np.isfinite(slope_pct) or slope_pct > float(cfg.max_slope_pct_per_bar):
                continue
            prominence = float((prominences[old_pos] + prominences[new_pos]) / 2.0)
            intercept = p_new - slope * x_new
            if not _anchor_span_is_clear(
                base,
                side,
                int(round(x_old)),
                int(round(x_new)),
                float(slope),
                float(intercept),
                cfg,
            ):
                continue
            raw_projection_end = min(float(len(base["close"]) - 1), x_new + projection_cap)
            projection_end = float(
                _projection_end_after_sustained_break(
                    base=base,
                    side=side,
                    slope=float(slope),
                    intercept=float(intercept),
                    start_index=int(round(x_new)),
                    max_end=int(round(raw_projection_end)),
                    cfg=cfg,
                )
            )
            live_start = max(
                float(available_indexes[new_pos]),
                x_new + float(cfg.pivot_strength),
            )
            if projection_end < live_start:
                continue
            span_score = min(span / max(anchor_min * 10.0, 1.0), 1.0)
            slope_score = max(0.0, 1.0 - slope_pct / max(float(cfg.max_slope_pct_per_bar), 1e-9))
            prom_score = min(prominence / max(prominence_floor * 5.0, 1.0), 1.0)
            pivot_quality = float((pivot_scores[old_pos] + pivot_scores[new_pos]) / 2.0)
            score = 0.38 * span_score + 0.24 * slope_score + 0.26 * prom_score + 0.12 * pivot_quality
            rows.append(
                {
                    "side": side,
                    "x_old": float(x_old),
                    "x_new": float(x_new),
                    "x_end": float(x_new),
                    "y_old": float(p_old),
                    "y_new": float(p_new),
                    "y_end": float(p_new),
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "projection_end": float(projection_end),
                    "live_start": float(live_start),
                    "span": float(span),
                    "slope_pct": float(slope_pct),
                    "prominence": prominence,
                    "pivot_quality": pivot_quality,
                    "score": float(score),
                    "pivot_count": 2.0,
                    "pivot_path": f"{int(round(x_old))}|{int(round(x_new))}",
                    "price_path": f"{float(p_old)}|{float(p_new)}",
                    "source_count": 1.0,
                    "absorbed_pivot_count": 2.0,
                    "absorbed_line_count": 0.0,
                    "half_width": 0.0,
                    "line_kind": line_kind,
                }
            )
    return pd.DataFrame(rows)


def _anchor_span_is_clear(
    base: dict[str, Series],
    side: LineSide,
    x_old: int,
    x_new: int,
    slope: float,
    intercept: float,
    cfg: TrendlineProjectionV2Config,
) -> bool:
    """Reject only meaningful body breaks between p1 and p2.

    Pivots are discrete approximations. A candle very close to either anchor,
    or a short/shallow interior breach, can be the same human-visible turn that
    the pivot picker represented with a neighbouring candle. Those exceptions
    prevent valid seed lines from being deleted before confirmation logic can
    assess them.
    """

    start = max(x_old + 1, 0)
    end = min(x_new, len(base["close"]))
    if end <= start:
        return True

    xs = np.arange(start, end, dtype="float64")
    line = intercept + slope * xs
    if side == "resistance":
        body = base["body_high"].iloc[start:end].to_numpy(dtype="float64")
        breach = body > line
        beyond = body - line
    else:
        body = base["body_low"].iloc[start:end].to_numpy(dtype="float64")
        breach = body < line
        beyond = line - body

    if not bool(np.any(breach)):
        return True

    return _anchor_breaches_are_allowed(
        breach=breach,
        beyond=beyond,
        line=line,
        atr=base["atr"].iloc[start:end].to_numpy(dtype="float64"),
        x_values=xs,
        x_old=x_old,
        x_new=x_new,
        edge_bars=int(cfg.anchor_break_edge_bars),
        max_run=int(cfg.anchor_break_max_run),
        max_pct=float(cfg.anchor_break_max_pct),
        max_atr_mult=float(cfg.anchor_break_max_atr_mult),
    )


def _anchor_breaches_are_allowed(
    *,
    breach: np.ndarray,
    beyond: np.ndarray,
    line: np.ndarray,
    atr: np.ndarray,
    x_values: np.ndarray,
    x_old: int,
    x_new: int,
    edge_bars: int,
    max_run: int,
    max_pct: float,
    max_atr_mult: float,
) -> bool:
    breach_indexes = np.flatnonzero(breach)
    if len(breach_indexes) == 0:
        return True
    breached_x = x_values[breach_indexes]
    near_anchor = (breached_x <= float(x_old + edge_bars)) | (breached_x >= float(x_new - edge_bars))
    interior = breach_indexes[~near_anchor]
    if len(interior) == 0:
        return True

    interior_breach = np.zeros_like(breach, dtype=bool)
    interior_breach[interior] = True
    max_interior_run = _max_true_run(interior_breach)
    pct_beyond = beyond[interior] / np.maximum(np.abs(line[interior]), 1e-9)
    atr_beyond = beyond[interior] / np.maximum(atr[interior], 1e-9)
    shallow_enough = bool(np.nanmax(pct_beyond) <= float(max_pct)) or bool(np.nanmax(atr_beyond) <= float(max_atr_mult))
    return max_interior_run <= int(max_run) and shallow_enough


def _max_true_run(values: np.ndarray) -> int:
    best = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _projection_end_after_sustained_break(
    *,
    base: dict[str, Series],
    side: str,
    slope: float,
    intercept: float,
    start_index: int,
    max_end: int,
    cfg: TrendlineProjectionV2Config,
) -> int:
    start = max(int(start_index) + 1, 0)
    end = min(int(max_end), len(base["close"]) - 1)
    if end <= start:
        return int(max_end)
    xs = np.arange(start, end + 1, dtype="float64")
    line = intercept + slope * xs
    tolerance = np.maximum(
        np.abs(line) * float(cfg.projection_break_tolerance_pct),
        base["atr"].iloc[start : end + 1].fillna(0.0).to_numpy(dtype="float64") * float(cfg.projection_break_atr_mult),
    )
    if side == "resistance":
        body = base["body_high"].iloc[start : end + 1].to_numpy(dtype="float64")
        broken = body > line + tolerance
    else:
        body = base["body_low"].iloc[start : end + 1].to_numpy(dtype="float64")
        broken = body < line - tolerance
    run = 0
    for offset, value in enumerate(broken):
        run = run + 1 if bool(value) else 0
        if run >= int(cfg.projection_break_candles):
            return start + offset
    return int(max_end)


def _candidate_max_distance_to_line(candidate: pd.Series, stronger: pd.Series) -> float:
    pivots = _pivot_tuple(candidate["pivot_path"])
    prices = _price_tuple(candidate.to_dict())
    if len(pivots) != len(prices):
        return 0.0
    slope = float(stronger["slope"])
    intercept = float(stronger["intercept"])
    distances = [abs(float(price) - (intercept + slope * float(pivot))) for pivot, price in zip(pivots, prices, strict=False)]
    return float(max(distances)) if distances else 0.0


def _join_endpoint_continuations(candidates: DataFrame, cfg: TrendlineProjectionV2Config) -> DataFrame:
    """Join A->B and B->C into A->B->C, then remove exact component lines."""

    if candidates.empty:
        return candidates.copy()

    output_rows: list[dict[str, float | str]] = candidates.to_dict("records")
    for side, side_candidates in candidates.groupby("side", sort=False):
        side_frame = side_candidates.copy()
        side_frame["norm_slope"] = _candidate_norm_slope(side_frame)
        by_old_anchor = {float(anchor): group for anchor, group in side_frame.groupby("x_old")}
        frontier = side_frame.to_dict("records")
        for pivot_count in range(3, int(cfg.max_joined_pivots) + 1):
            next_frontier: list[dict[str, float | str]] = []
            seen_paths: set[str] = set()
            for path in frontier:
                path_pivots = _pivot_tuple(path["pivot_path"])
                path_prices = _price_tuple(path)
                continuations = by_old_anchor.get(float(path_pivots[-1]))
                if continuations is None:
                    continue
                path_norm = _row_norm_slope(pd.Series(path))
                for segment in continuations.itertuples(index=False):
                    next_pivot = int(round(float(segment.x_new)))
                    if next_pivot <= path_pivots[-1] or next_pivot in path_pivots:
                        continue
                    if not _angle_within_tolerance(path_norm, float(segment.norm_slope), float(cfg.join_angle_tolerance_pct)):
                        continue
                    new_pivots = (*path_pivots, next_pivot)
                    new_prices = (*path_prices, float(segment.y_new))
                    candidate = _candidate_from_path(
                        side=str(side),
                        pivots=new_pivots,
                        prices=new_prices,
                        projection_end=float(segment.projection_end),
                        live_start=float(segment.live_start),
                        prominence=float((float(path["prominence"]) + float(segment.prominence)) / 2.0),
                        score=min(float(path["score"]) * 0.65 + float(segment.score) * 0.35 + 0.06, 1.0),
                        cfg=cfg,
                    )
                    if candidate is None:
                        continue
                    key = str(candidate["pivot_path"])
                    if key in seen_paths:
                        continue
                    seen_paths.add(key)
                    candidate["line_kind"] = "joined"
                    next_frontier.append(candidate)
            if not next_frontier:
                break
            output_rows.extend(next_frontier)
            frontier = next_frontier

    joined = pd.DataFrame(output_rows)
    joined = joined.sort_values(["pivot_count", "score", "span"], ascending=False)
    joined = joined.drop_duplicates(["side", "pivot_path"], keep="first").reset_index(drop=True)
    return _absorb_shared_pivot_smaller_lines(joined, cfg)


def _candidate_from_path(
    *,
    side: str,
    pivots: tuple[int, ...],
    prices: tuple[float, ...],
    projection_end: float,
    live_start: float,
    prominence: float,
    score: float,
    cfg: TrendlineProjectionV2Config,
) -> dict[str, float | str] | None:
    if len(pivots) < 2 or len(prices) < 2:
        return None
    x_old = float(pivots[0])
    x_new = float(pivots[1])
    x_end = float(pivots[-1])
    y_old = float(prices[0])
    y_new = float(prices[1])
    y_end = float(prices[-1])
    span = x_end - x_old
    if span <= 0.0:
        return None
    slope = (y_end - y_old) / span
    intercept = y_end - slope * x_end
    line_at_pivots = np.array([intercept + slope * float(pivot) for pivot in pivots], dtype="float64")
    prices_array = np.array(prices, dtype="float64")
    midpoint_error = np.max(np.abs(prices_array - line_at_pivots) / np.maximum(np.abs(prices_array), 1e-9))
    if midpoint_error > float(cfg.join_midpoint_tolerance_pct):
        return None
    price_ref = max((abs(y_old) + abs(y_end)) / 2.0, 1e-9)
    slope_pct = abs(slope) / price_ref
    if not np.isfinite(slope_pct) or slope_pct > float(cfg.max_slope_pct_per_bar):
        return None
    if projection_end < live_start:
        return None
    return {
        "side": side,
        "x_old": x_old,
        "x_new": x_new,
        "x_end": x_end,
        "x_join": float(pivots[-2]),
        "y_old": y_old,
        "y_join": float(prices[-2]),
        "y_new": y_new,
        "y_end": y_end,
        "slope": float(slope),
        "intercept": float(intercept),
        "projection_end": projection_end,
        "live_start": float(live_start),
        "span": float(span),
        "slope_pct": float(slope_pct),
        "prominence": prominence,
        "score": score,
        "pivot_count": float(len(pivots)),
        "pivot_path": "|".join(str(pivot) for pivot in pivots),
        "price_path": "|".join(str(float(price)) for price in prices),
        "source_count": float(len(pivots) - 1),
        "absorbed_pivot_count": float(len(pivots)),
        "absorbed_line_count": 0.0,
        "half_width": 0.0,
    }


def _absorb_shared_pivot_smaller_lines(candidates: DataFrame, cfg: TrendlineProjectionV2Config) -> DataFrame:
    """Replace exact component lines after the larger line becomes knowable."""

    if candidates.empty:
        return candidates.copy()

    kept_rows: list[pd.Series] = []
    sorted_candidates = candidates.sort_values(["pivot_count", "span", "score"], ascending=False)
    for _, side_candidates in sorted_candidates.groupby("side", sort=False):
        side_kept: list[pd.Series] = []
        pre_replacement_rows: list[pd.Series] = []
        for _, candidate in side_candidates.iterrows():
            candidate_pivots = _pivot_set(candidate["pivot_path"])
            absorbed = False
            earliest_replacement_live_start = np.inf
            for kept in side_kept:
                kept_pivots = _pivot_set(kept["pivot_path"])
                if len(candidate_pivots) < int(cfg.shared_pivot_merge_min):
                    continue
                if candidate_pivots.issubset(kept_pivots):
                    kept["absorbed_line_count"] = float(kept.get("absorbed_line_count", 0.0)) + 1.0
                    replacement_live_start = float(kept.get("live_start", kept["x_new"]))
                    earliest_replacement_live_start = min(earliest_replacement_live_start, replacement_live_start)
                    absorbed = True
            if not absorbed:
                side_kept.append(candidate)
                continue
            pre_replacement = _truncate_candidate_before(candidate, earliest_replacement_live_start - 1.0)
            if pre_replacement is not None:
                pre_replacement["line_kind"] = "pre_join_component"
                pre_replacement["replacement_live_start"] = earliest_replacement_live_start
                pre_replacement_rows.append(pre_replacement)
        kept_rows.extend(side_kept)
        kept_rows.extend(pre_replacement_rows)

    if not kept_rows:
        return candidates.head(0).copy()
    merged = pd.DataFrame(kept_rows).sort_values(["side", "score", "span"], ascending=[True, False, False]).reset_index(drop=True)
    pivot_counts = merged["pivot_path"].map(lambda value: float(len(_pivot_set(value))))
    existing_absorbed = pd.to_numeric(merged.get("absorbed_pivot_count", pivot_counts), errors="coerce")
    merged["absorbed_pivot_count"] = np.maximum(existing_absorbed.fillna(pivot_counts), pivot_counts)
    if "absorbed_line_count" not in merged:
        merged["absorbed_line_count"] = 0.0
    merged["absorbed_line_count"] = pd.to_numeric(merged["absorbed_line_count"], errors="coerce").fillna(0.0)
    return merged


def _truncate_candidate_before(candidate: pd.Series, cutoff_index: float) -> pd.Series | None:
    """Keep an absorbed/deleted line only until the stronger replacement is live."""

    row = candidate.copy()
    live_start = float(row.get("live_start", row["x_new"]))
    projection_end = min(float(row["projection_end"]), float(cutoff_index))
    if projection_end < live_start:
        return None
    row["projection_end"] = float(projection_end)
    return row


def _absorb_nearby_weaker_lines(candidates: DataFrame, proximity_pct: float, cfg: TrendlineProjectionV2Config) -> DataFrame:
    """Remove weaker duplicate lines and transfer unique pivot confirmations."""

    if candidates.empty:
        return candidates.copy()

    out_rows: list[pd.Series] = []
    sorted_candidates = candidates.sort_values(["pivot_count", "score", "span"], ascending=False).copy()
    sorted_candidates["norm_slope"] = _candidate_norm_slope(sorted_candidates)
    for _, side_candidates in sorted_candidates.groupby("side", sort=False):
        kept: list[dict[str, object]] = []
        for _, candidate in side_candidates.iterrows():
            candidate_pivots = _pivot_set(candidate["pivot_path"])
            candidate_points = _pivot_price_map(candidate)
            absorbed = False
            for kept_item in kept:
                kept_row = kept_item["row"]
                if not isinstance(kept_row, pd.Series):
                    continue
                if _candidate_is_weaker_or_equal(candidate, kept_row) and _candidate_is_absorbable_duplicate(
                    candidate,
                    kept_row,
                    proximity_pct,
                    cfg,
                ):
                    kept_item["absorbed_pivots"].update(candidate_pivots)
                    absorbed_points = kept_item["absorbed_points"]
                    if isinstance(absorbed_points, dict):
                        absorbed_points.update(candidate_points)
                    kept_item["absorbed_line_count"] = (
                        float(kept_item["absorbed_line_count"])
                        + 1.0
                        + max(float(candidate.get("absorbed_line_count", 0.0)), 0.0)
                    )
                    kept_item["line_width"] = max(
                        float(kept_item["line_width"]),
                        _candidate_max_distance_to_line(candidate, kept_row),
                    )
                    absorbed = True
                    break
            if not absorbed:
                kept.append(
                    {
                        "row": candidate.copy(),
                        "absorbed_pivots": set(candidate_pivots),
                        "absorbed_points": candidate_points,
                        "absorbed_line_count": max(float(candidate.get("absorbed_line_count", 0.0)), 0.0),
                        "line_width": 0.0,
                    }
                )
        for kept_item in kept:
            row = kept_item["row"]
            if isinstance(row, pd.Series):
                row = row.copy()
                row["absorbed_pivot_count"] = max(
                    float(row.get("absorbed_pivot_count", 0.0)),
                    float(len(kept_item["absorbed_pivots"])),
                )
                row["absorbed_line_count"] = float(kept_item["absorbed_line_count"])
                row["nearby_merge_proximity_pct"] = float(proximity_pct)
                row["half_width"] = float(kept_item["line_width"]) * float(cfg.absorb_width_scale)
                if float(row["absorbed_line_count"]) > 0.0:
                    row["line_kind"] = "nearby_absorbed"
                row["score"] = min(
                    1.0,
                    float(row.get("score", 0.0))
                    + 0.025 * float(row["absorbed_line_count"])
                    + 0.015 * max(float(row["absorbed_pivot_count"]) - float(row.get("pivot_count", 2.0)), 0.0),
                )
                if float(row.get("projection_end", row["x_new"])) >= float(row.get("live_start", row["x_new"])):
                    out_rows.append(row)

    if not out_rows:
        return candidates.head(0).copy()
    merged = pd.DataFrame(out_rows).sort_values(
        ["side", "absorbed_pivot_count", "score", "span"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    cleaned = _absorb_shared_pivot_smaller_lines(merged, cfg)
    return _preserve_absorbed_line_counts(cleaned, candidates, merged)


def _preserve_absorbed_line_counts(result: DataFrame, *sources: DataFrame) -> DataFrame:
    """Keep merge evidence counts from being reset by later cleanup passes."""

    if result.empty or "pivot_path" not in result or "side" not in result:
        return result
    count_sources = []
    for source in sources:
        if source.empty or "absorbed_line_count" not in source or "pivot_path" not in source or "side" not in source:
            continue
        count_sources.append(
            source.assign(absorbed_line_count=pd.to_numeric(source["absorbed_line_count"], errors="coerce").fillna(0.0))
            .groupby(["side", "pivot_path"], dropna=False)["absorbed_line_count"]
            .max()
        )
    if not count_sources:
        return result
    source_counts = pd.concat(count_sources, axis=1).max(axis=1)
    out = result.copy()
    current = pd.to_numeric(out.get("absorbed_line_count", 0.0), errors="coerce").fillna(0.0)
    keys = pd.MultiIndex.from_frame(out[["side", "pivot_path"]])
    preserved = pd.Series(keys.map(source_counts).to_numpy(dtype="float64"), index=out.index).fillna(0.0)
    out["absorbed_line_count"] = np.maximum(current.to_numpy(dtype="float64"), preserved.to_numpy(dtype="float64"))
    return out


def _candidate_is_weaker_or_equal(candidate: pd.Series, stronger: pd.Series) -> bool:
    candidate_key = (
        float(candidate.get("pivot_count", 0.0)),
        float(candidate.get("score", 0.0)),
        float(candidate.get("span", 0.0)),
    )
    stronger_key = (
        float(stronger.get("pivot_count", 0.0)),
        float(stronger.get("score", 0.0)),
        float(stronger.get("span", 0.0)),
    )
    return candidate_key <= stronger_key


def _candidate_is_absorbable_duplicate(
    candidate: pd.Series,
    stronger: pd.Series,
    proximity_pct: float,
    cfg: TrendlineProjectionV2Config,
) -> bool:
    """Return true when two same-side lines express the same structure."""

    if str(candidate.get("side")) != str(stronger.get("side")):
        return False
    left_norm = _row_norm_slope(candidate)
    right_norm = _row_norm_slope(stronger)
    if not _angle_within_tolerance(left_norm, right_norm, float(cfg.duplicate_angle_tolerance_pct)):
        return False
    if not _candidate_last_pivot_near_line(candidate, stronger, max(float(proximity_pct), float(cfg.duplicate_line_proximity_pct))):
        return False
    if _candidate_pivots_near_line(candidate, stronger, max(float(proximity_pct), float(cfg.duplicate_line_proximity_pct))):
        return True
    return _candidate_lines_overlap_same_structure(
        candidate,
        stronger,
        max(float(proximity_pct), float(cfg.duplicate_line_proximity_pct)),
        int(cfg.duplicate_min_overlap_bars),
    )


def _candidate_last_pivot_near_line(candidate: pd.Series, stronger: pd.Series, proximity_pct: float) -> bool:
    pivots = _pivot_tuple(candidate["pivot_path"])
    prices = _price_tuple(candidate.to_dict())
    if not pivots or len(pivots) != len(prices):
        return False
    last_pivot = float(pivots[-1])
    last_price = float(prices[-1])
    stronger_price = float(stronger["intercept"]) + float(stronger["slope"]) * last_pivot
    denominator = max(abs(last_price), 1e-9)
    return abs(last_price - stronger_price) / denominator <= float(proximity_pct)


def _candidate_lines_overlap_same_structure(
    candidate: pd.Series,
    stronger: pd.Series,
    proximity_pct: float,
    min_overlap_bars: int,
) -> bool:
    start = max(
        float(candidate.get("live_start", candidate.get("x_new", 0.0))),
        float(stronger.get("live_start", stronger.get("x_new", 0.0))),
    )
    end = min(float(candidate.get("projection_end", 0.0)), float(stronger.get("projection_end", 0.0)))
    if end - start < float(min_overlap_bars):
        return False
    sample_x = np.array([start, (start + end) / 2.0, end], dtype="float64")
    candidate_values = float(candidate["intercept"]) + float(candidate["slope"]) * sample_x
    stronger_values = float(stronger["intercept"]) + float(stronger["slope"]) * sample_x
    denominator = np.maximum(np.maximum(np.abs(candidate_values), np.abs(stronger_values)), 1e-9)
    distance_pct = np.abs(candidate_values - stronger_values) / denominator
    return bool(np.all(np.isfinite(distance_pct)) and np.nanmax(distance_pct) <= float(proximity_pct))


def _candidate_pivots_near_line(candidate: pd.Series, stronger: pd.Series, proximity_pct: float) -> bool:
    pivots = _pivot_tuple(candidate["pivot_path"])
    prices = _price_tuple(candidate.to_dict())
    if len(pivots) != len(prices):
        return False
    slope = float(stronger["slope"])
    intercept = float(stronger["intercept"])
    for pivot, price in zip(pivots, prices, strict=False):
        stronger_price = intercept + slope * float(pivot)
        denominator = max(abs(float(price)), 1e-9)
        if abs(float(price) - stronger_price) / denominator > proximity_pct:
            return False
    return True


def _candidate_norm_slope(candidates: DataFrame) -> Series:
    end_price = candidates["y_end"] if "y_end" in candidates else candidates["y_new"]
    price_ref = ((candidates["y_old"].abs() + end_price.abs()) / 2.0).replace(0.0, np.nan)
    return (candidates["slope"] / price_ref).replace([np.inf, -np.inf], np.nan)


def _row_norm_slope(row: pd.Series) -> float:
    norm_slope = row.get("norm_slope", np.nan)
    if np.isfinite(norm_slope):
        return float(norm_slope)
    end_price = row.get("y_end", row["y_new"])
    price_ref = max((abs(float(row["y_old"])) + abs(float(end_price))) / 2.0, 1e-9)
    return float(row["slope"]) / price_ref


def _pivot_tuple(path: object) -> tuple[int, ...]:
    return tuple(int(part) for part in str(path).split("|") if part != "")


def _price_tuple(row: dict[str, object]) -> tuple[float, ...]:
    price_path = row.get("price_path")
    if price_path is not None and str(price_path) != "nan":
        return tuple(float(part) for part in str(price_path).split("|") if part != "")
    return (float(row["y_old"]), float(row["y_new"]))


def _pivot_price_map(row: pd.Series) -> dict[int, float]:
    pivots = _pivot_tuple(row["pivot_path"])
    prices = _price_tuple(row.to_dict())
    if len(pivots) != len(prices):
        return {}
    return {int(pivot): float(price) for pivot, price in zip(pivots, prices, strict=False)}


def _angle_within_tolerance(left_norm_slope: float, right_norm_slope: float, tolerance_pct: float) -> bool:
    if not np.isfinite(left_norm_slope) or not np.isfinite(right_norm_slope):
        return False
    denominator = max(abs(left_norm_slope), abs(right_norm_slope), 1e-9)
    return abs(left_norm_slope - right_norm_slope) / denominator <= tolerance_pct


def _pivot_set(path: object) -> set[int]:
    return {int(part) for part in str(path).split("|") if part != ""}


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


def _resolve_config(config: TrendlineProjectionV2Config | None, overrides: dict[str, object]) -> TrendlineProjectionV2Config:
    base = config or TrendlineProjectionV2Config()
    valid = {field.name for field in fields(TrendlineProjectionV2Config)}
    clean = {key: value for key, value in overrides.items() if value is not None}
    unknown = sorted(set(clean).difference(valid))
    if unknown:
        raise TypeError(f"Unknown trendline v2 config override(s): {', '.join(unknown)}")

    requested = replace(base, **{name: clean[name] for name in clean if name in valid})
    timeframe = _normalize_timeframe(str(requested.timeframe))
    requested = replace(requested, timeframe=timeframe)
    profile = _TIMEFRAME_PROFILES[timeframe]
    if config is None:
        profiled = replace(requested, **profile)
    else:
        default = TrendlineProjectionV2Config()
        profile_updates = {
            name: profile[name]
            for name in _PROFILE_FIELDS
            if name not in clean and getattr(config, name) == getattr(default, name)
        }
        profiled = replace(requested, **profile_updates)
    if clean:
        profiled = replace(profiled, **{name: clean[name] for name in clean if name in valid})
        profiled = replace(profiled, timeframe=_normalize_timeframe(str(profiled.timeframe)))
    return profiled


def _normalize_timeframe(value: str) -> str:
    key = str(value).strip().lower()
    if key not in _TIMEFRAME_ALIASES:
        allowed = ", ".join(sorted(_TIMEFRAME_PROFILES))
        raise ValueError(f"timeframe must be one of: {allowed}")
    return _TIMEFRAME_ALIASES[key]


def _validate_dataframe(frame: DataFrame) -> None:
    required = {"open", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataframe missing required columns: {missing}")


def _validate_config(cfg: TrendlineProjectionV2Config) -> None:
    _normalize_timeframe(str(cfg.timeframe))
    if cfg.pivot_strength < 1:
        raise ValueError("pivot_strength must be positive")
    if cfg.raw_line_output_count < 1:
        raise ValueError("raw_line_output_count must be positive")
    if cfg.min_anchor_bars < 1:
        raise ValueError("min_anchor_bars must be positive")
    if cfg.max_anchor_bars < cfg.min_anchor_bars:
        raise ValueError("max_anchor_bars must be greater than or equal to min_anchor_bars")
    if cfg.min_pivot_prominence_atr < 0.0:
        raise ValueError("min_pivot_prominence_atr must be non-negative")
    if cfg.max_slope_pct_per_bar <= 0.0:
        raise ValueError("max_slope_pct_per_bar must be positive")
    if cfg.max_projection_bars < 1:
        raise ValueError("max_projection_bars must be positive")
    if cfg.max_active_line_distance_pct <= 0.0 or cfg.max_active_line_distance_atr_mult <= 0.0:
        raise ValueError("active line distance settings must be positive")
    if cfg.absorb_width_scale < 0.0:
        raise ValueError("absorb_width_scale must be non-negative")
    if cfg.duplicate_line_proximity_pct <= 0.0 or cfg.duplicate_angle_tolerance_pct <= 0.0:
        raise ValueError("duplicate tolerances must be positive")
    if cfg.duplicate_min_overlap_bars < 1:
        raise ValueError("duplicate_min_overlap_bars must be positive")
    if cfg.anchor_break_edge_bars < 0 or cfg.anchor_break_max_run < 0:
        raise ValueError("anchor break bars/run settings must be non-negative")
    if cfg.anchor_break_max_pct < 0.0 or cfg.anchor_break_max_atr_mult < 0.0:
        raise ValueError("anchor break tolerance settings must be non-negative")
    if cfg.projection_break_candles < 1:
        raise ValueError("projection_break_candles must be positive")
    if cfg.projection_break_tolerance_pct < 0.0 or cfg.projection_break_atr_mult < 0.0:
        raise ValueError("projection break tolerance settings must be non-negative")
    if cfg.join_angle_tolerance_pct <= 0.0:
        raise ValueError("join_angle_tolerance_pct must be positive")
    if cfg.join_midpoint_tolerance_pct <= 0.0:
        raise ValueError("join_midpoint_tolerance_pct must be positive")
    if cfg.max_joined_pivots < 2:
        raise ValueError("max_joined_pivots must be at least 2")
    if cfg.shared_pivot_merge_min < 1:
        raise ValueError("shared_pivot_merge_min must be positive")
    if cfg.nearby_merge_proximity_pct <= 0.0:
        raise ValueError("nearby_merge_proximity_pct must be positive")

__all__ = [
    "TrendlineProjectionV2Config",
    "add_trendline_projection_v2",
]

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from pattern_common import (
    _bottom_base_is_held,
    _bottom_p1_dominance_score,
    _carry_values_while_state,
    _clip_value,
    _dedupe_interval_level_events,
    _dynamic_height_tolerance_pct,
    _num,
    _pattern_geometry_arrays,
    _peak_confirmation_state,
    _peak_retest_quality,
    _prior_impulse_score,
    _prior_opposite_pivot_context,
    _proof_line_columns,
    _rolling_atr_pct,
    _rolling_body_pct,
    _rolling_pivot_prominence_pct,
    _threshold_body_pct,
    _threshold_scale_pct,
    _top_base_is_held,
    _top_p1_dominance_score,
    _with_foundation_pivots,
)


@dataclass(frozen=True)
class PatternPeakConfig:
    output_prefix: str = "pat"
    timeframe: str = "4h"
    pivot_prefix: str = "pf"
    atr_period: int = 14
    pivot_strength: int = 2
    pivot_min_prominence_atr: float = 0.35
    pivot_min_prominence_pct: float = 0.0
    pivot_min_spacing_bars: int = 1
    pivot_min_distance_atr: float = 0.0
    pivot_min_distance_pct: float = 0.0
    pattern_pivot_strength: int = 1
    entry_cooldown_bars: int = 8
    pattern_lifecycle_mature_bars: int = 12
    pattern_lifecycle_stale_bars: int = 12
    double_duplicate_overlap_pct: float = 0.70
    double_duplicate_neckline_tolerance_pct: float = 0.012
    peak_dynamic_body_window: int = 30
    peak_dynamic_scale_window: int = 30
    peak_premove_body_mult: float = 6.0
    peak_level_tolerance_body_mult: float = 2.25
    peak_level_tolerance_atr_mult: float = 0.35
    peak_level_tolerance_prominence_mult: float = 0.20
    peak_reaction_body_mult: float = 3.0
    peak_base_return_buffer_body_mult: float = 0.8
    triple_pattern_window: int = 110
    min_triple_pattern_bars: int = 12
    max_triple_pattern_bars: int = 90
    min_triple_spacing_bars: int = 4
    triple_max_candidate_pivots: int = 8
    triple_peak_tolerance_pct: float = 0.040
    min_triple_neckline_depth_pct: float = 0.018
    min_triple_prior_move_pct: float = 0.040
    min_triple_first_pivot_move_pct: float = 0.030
    triple_reaction_max_bars: int = 24
    min_triple_reaction_score: float = 0.12
    min_triple_quality: float = 0.72
    include_pattern_diagnostics: bool = False


def add_pattern_peaks(
    dataframe: DataFrame,
    timeframe: str = "4h",
    config: PatternPeakConfig | None = None,
    **overrides: object,
) -> DataFrame:
    cfg = _resolve_config(config, timeframe, overrides)
    _validate_ohlcv(dataframe)
    frame = _with_foundation_pivots(
        dataframe,
        pivot_prefix=cfg.pivot_prefix,
        atr_period=int(cfg.atr_period),
        pivot_strength=int(cfg.pivot_strength),
        pivot_min_prominence_atr=float(cfg.pivot_min_prominence_atr),
        pivot_min_prominence_pct=float(cfg.pivot_min_prominence_pct),
        pivot_min_spacing_bars=int(cfg.pivot_min_spacing_bars),
        pivot_min_distance_atr=float(cfg.pivot_min_distance_atr),
        pivot_min_distance_pct=float(cfg.pivot_min_distance_pct),
    )
    columns = _triple_reversal_columns(frame, cfg)
    existing = [
        column
        for column in dataframe.columns
        if str(column).startswith((f"{cfg.output_prefix}_triple_top_", f"{cfg.output_prefix}_triple_bottom_"))
    ]
    clean = dataframe.drop(columns=existing).copy() if existing else dataframe.copy()
    return pd.concat([clean, pd.DataFrame(columns, index=frame.index)], axis=1)


def _resolve_config(
    config: PatternPeakConfig | None,
    timeframe: str,
    overrides: dict[str, object],
) -> PatternPeakConfig:
    cfg = config or PatternPeakConfig(timeframe=str(timeframe))
    valid = {field.name for field in fields(PatternPeakConfig)}
    clean = {key: value for key, value in overrides.items() if value is not None}
    unknown = sorted(set(clean).difference(valid))
    if unknown:
        raise TypeError(f"Unknown peak config override(s): {', '.join(unknown)}")
    if "timeframe" not in clean:
        clean["timeframe"] = str(timeframe)
    return replace(cfg, **clean) if clean else cfg


def _validate_ohlcv(dataframe: DataFrame) -> None:
    missing = [column for column in ("open", "high", "low", "close") if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Dataframe missing required columns: {', '.join(missing)}")
    if dataframe.empty:
        raise ValueError("Dataframe must not be empty")


def _triple_reversal_columns(frame: DataFrame, cfg: PatternPeakConfig) -> dict[str, Series]:
    """Detect triple top / triple bottom attention at the third confirmed pivot.

    A triple top is three resistance touches after an upward move. A triple
    bottom is three support touches after a downward move. The indicator emits
    attention at the third confirmed touch; the strategy decides whether to
    fade the level, wait for a neckline break, or ignore it.
    """

    p = cfg.output_prefix
    arrays = _triple_reversal_arrays(frame, cfg)
    close = _num(frame["close"]).to_numpy(dtype="float64")
    top_raw = pd.Series(arrays["triple_top_setup_short"], index=frame.index, dtype="bool").fillna(False)
    bottom_raw = pd.Series(arrays["triple_bottom_setup_long"], index=frame.index, dtype="bool").fillna(False)
    top = pd.Series(
        _dedupe_interval_level_events(
            arrays["triple_top_setup_short"],
            arrays["triple_top_first_index"],
            arrays["triple_top_third_index"],
            arrays["triple_top_level"],
            arrays["triple_top_neckline"],
            int(cfg.entry_cooldown_bars),
            float(cfg.double_duplicate_overlap_pct),
            float(cfg.double_duplicate_neckline_tolerance_pct),
        ),
        index=frame.index,
        dtype="bool",
    ).fillna(False)
    bottom = pd.Series(
        _dedupe_interval_level_events(
            arrays["triple_bottom_setup_long"],
            arrays["triple_bottom_first_index"],
            arrays["triple_bottom_third_index"],
            arrays["triple_bottom_level"],
            arrays["triple_bottom_neckline"],
            int(cfg.entry_cooldown_bars),
            float(cfg.double_duplicate_overlap_pct),
            float(cfg.double_duplicate_neckline_tolerance_pct),
        ),
        index=frame.index,
        dtype="bool",
    ).fillna(False)
    top_confirmation = _peak_confirmation_state(
        close,
        top,
        arrays["triple_top_neckline"],
        arrays["triple_top_quality"],
        int(cfg.pattern_lifecycle_mature_bars),
        int(cfg.pattern_lifecycle_stale_bars),
        top=True,
    )
    bottom_confirmation = _peak_confirmation_state(
        close,
        bottom,
        arrays["triple_bottom_neckline"],
        arrays["triple_bottom_quality"],
        int(cfg.pattern_lifecycle_mature_bars),
        int(cfg.pattern_lifecycle_stale_bars),
        top=False,
    )
    top_developing = pd.Series(top_confirmation["developing"], index=frame.index, dtype="bool")
    bottom_developing = pd.Series(bottom_confirmation["developing"], index=frame.index, dtype="bool")
    top_confirmed = pd.Series(top_confirmation["confirmed"], index=frame.index, dtype="bool")
    bottom_confirmed = pd.Series(bottom_confirmation["confirmed"], index=frame.index, dtype="bool")
    top_current_quality = pd.Series(top_confirmation["quality"], index=frame.index, dtype="float64")
    bottom_current_quality = pd.Series(bottom_confirmation["quality"], index=frame.index, dtype="float64")
    top_state = pd.Series(top_confirmation["state"], index=frame.index, dtype="int8")
    bottom_state = pd.Series(bottom_confirmation["state"], index=frame.index, dtype="int8")
    top_carried = _carry_values_while_state(
        top,
        top_state,
        {
            "p1_index": arrays["triple_top_first_index"],
            "p2_index": arrays["triple_top_second_index"],
            "p3_index": arrays["triple_top_third_index"],
            "level": arrays["triple_top_level"],
            "neckline": arrays["triple_top_neckline"],
        },
    )
    bottom_carried = _carry_values_while_state(
        bottom,
        bottom_state,
        {
            "p1_index": arrays["triple_bottom_first_index"],
            "p2_index": arrays["triple_bottom_second_index"],
            "p3_index": arrays["triple_bottom_third_index"],
            "level": arrays["triple_bottom_level"],
            "neckline": arrays["triple_bottom_neckline"],
        },
    )
    top_present = top_developing | top_confirmed
    bottom_present = bottom_developing | bottom_confirmed
    top_confirmation_level = pd.Series(top_carried["neckline"], index=frame.index, dtype="float64")
    bottom_confirmation_level = pd.Series(bottom_carried["neckline"], index=frame.index, dtype="float64")
    top_level = pd.Series(top_carried["level"], index=frame.index, dtype="float64")
    bottom_level = pd.Series(bottom_carried["level"], index=frame.index, dtype="float64")
    columns = {
        f"{p}_triple_top_pattern_present": top_present.fillna(False),
        f"{p}_triple_bottom_pattern_present": bottom_present.fillna(False),
        f"{p}_triple_top_pattern_confirmed": top_confirmed.fillna(False),
        f"{p}_triple_bottom_pattern_confirmed": bottom_confirmed.fillna(False),
        f"{p}_triple_top_indicator_score": top_current_quality.where(top_present, 0.0),
        f"{p}_triple_bottom_indicator_score": bottom_current_quality.where(bottom_present, 0.0),
        f"{p}_triple_top_confirmation_level": top_confirmation_level,
        f"{p}_triple_bottom_confirmation_level": bottom_confirmation_level,
        f"{p}_triple_top_target_level": top_confirmation_level - (top_level - top_confirmation_level).clip(lower=0.0),
        f"{p}_triple_bottom_target_level": bottom_confirmation_level + (bottom_confirmation_level - bottom_level).clip(lower=0.0),
        f"{p}_triple_top_p1_index": pd.Series(top_carried["p1_index"], index=frame.index, dtype="float64"),
        f"{p}_triple_top_p2_index": pd.Series(top_carried["p2_index"], index=frame.index, dtype="float64"),
        f"{p}_triple_top_p3_index": pd.Series(top_carried["p3_index"], index=frame.index, dtype="float64"),
        f"{p}_triple_bottom_p1_index": pd.Series(bottom_carried["p1_index"], index=frame.index, dtype="float64"),
        f"{p}_triple_bottom_p2_index": pd.Series(bottom_carried["p2_index"], index=frame.index, dtype="float64"),
        f"{p}_triple_bottom_p3_index": pd.Series(bottom_carried["p3_index"], index=frame.index, dtype="float64"),
    }
    if bool(getattr(cfg, "include_pattern_diagnostics", False)):
        columns.update(
            {
                f"{p}_triple_top_structure_short": top_raw,
                f"{p}_triple_bottom_structure_long": bottom_raw,
                f"{p}_triple_top_structure_quality": pd.Series(
                    arrays["triple_top_quality"], index=frame.index, dtype="float64"
                ).where(top_raw, 0.0),
                f"{p}_triple_bottom_structure_quality": pd.Series(
                    arrays["triple_bottom_quality"], index=frame.index, dtype="float64"
                ).where(bottom_raw, 0.0),
                f"{p}_triple_top_reaction_score": pd.Series(
                    arrays["triple_top_reaction_score"], index=frame.index, dtype="float64"
                ).where(top_raw, 0.0),
                f"{p}_triple_bottom_reaction_score": pd.Series(
                    arrays["triple_bottom_reaction_score"], index=frame.index, dtype="float64"
                ).where(bottom_raw, 0.0),
                **_proof_line_columns(
                    frame.index,
                    f"{p}_triple_top_short",
                    top,
                    {
                        1: (
                            arrays["triple_top_first_index"],
                            arrays["triple_top_first_price"],
                            arrays["triple_top_third_index"],
                            arrays["triple_top_third_price"],
                        ),
                        2: (
                            arrays["triple_top_first_index"],
                            arrays["triple_top_neckline"],
                            arrays["triple_top_third_index"],
                            arrays["triple_top_neckline"],
                        ),
                    },
                ),
                **_proof_line_columns(
                    frame.index,
                    f"{p}_triple_bottom_long",
                    bottom,
                    {
                        1: (
                            arrays["triple_bottom_first_index"],
                            arrays["triple_bottom_first_price"],
                            arrays["triple_bottom_third_index"],
                            arrays["triple_bottom_third_price"],
                        ),
                        2: (
                            arrays["triple_bottom_first_index"],
                            arrays["triple_bottom_neckline"],
                            arrays["triple_bottom_third_index"],
                            arrays["triple_bottom_neckline"],
                        ),
                    },
                ),
            }
        )
    return columns


def _triple_reversal_arrays(frame: DataFrame, cfg: PatternPeakConfig) -> dict[str, np.ndarray]:
    close, body_high, body_low, high_pivot, high_index, low_pivot, low_index = _pattern_geometry_arrays(frame, cfg)
    rows = len(frame)
    out = _empty_triple_output(rows)
    window = int(cfg.triple_pattern_window)
    min_bars = int(cfg.min_triple_pattern_bars)
    max_bars = int(cfg.max_triple_pattern_bars)
    min_spacing = int(cfg.min_triple_spacing_bars)
    max_candidates = int(cfg.triple_max_candidate_pivots)
    scale_window = int(getattr(cfg, "peak_dynamic_scale_window", cfg.peak_dynamic_body_window))
    body_pct = _rolling_body_pct(frame, scale_window)
    atr_pct = _rolling_atr_pct(frame, scale_window)
    prominence_pct = _rolling_pivot_prominence_pct(frame, str(cfg.pivot_prefix), scale_window)

    valid_close = np.isfinite(close) & (close != 0.0)
    candidate_rows = np.flatnonzero(
        valid_close
        & (
            (np.isfinite(high_pivot) & np.isfinite(high_index))
            | (np.isfinite(low_pivot) & np.isfinite(low_index))
        )
    )
    for row in candidate_rows:
        if np.isfinite(high_pivot[row]) and np.isfinite(high_index[row]):
            top = _score_triple_top(
                row,
                close,
                body_low,
                high_pivot,
                high_index,
                low_pivot,
                low_index,
                window,
                min_bars,
                max_bars,
                min_spacing,
                max_candidates,
                float(cfg.triple_peak_tolerance_pct),
                float(cfg.min_triple_neckline_depth_pct),
                float(cfg.min_triple_prior_move_pct),
                float(cfg.min_triple_first_pivot_move_pct),
                int(cfg.triple_reaction_max_bars),
                float(cfg.min_triple_reaction_score),
                body_pct,
                atr_pct,
                prominence_pct,
                float(cfg.peak_premove_body_mult),
                float(cfg.peak_level_tolerance_body_mult),
                float(cfg.peak_level_tolerance_atr_mult),
                float(cfg.peak_level_tolerance_prominence_mult),
                float(cfg.peak_reaction_body_mult),
                float(cfg.peak_base_return_buffer_body_mult),
                float(getattr(cfg, "peak_prior_impulse_min_efficiency", 0.0)),
                int(getattr(cfg, "peak_prior_impulse_min_bars", 1)),
            )
            if top["quality"] >= float(cfg.min_triple_quality):
                _store_triple(out, row, "triple_top", top)
        if np.isfinite(low_pivot[row]) and np.isfinite(low_index[row]):
            bottom = _score_triple_bottom(
                row,
                close,
                body_high,
                high_pivot,
                high_index,
                low_pivot,
                low_index,
                window,
                min_bars,
                max_bars,
                min_spacing,
                max_candidates,
                float(cfg.triple_peak_tolerance_pct),
                float(cfg.min_triple_neckline_depth_pct),
                float(cfg.min_triple_prior_move_pct),
                float(cfg.min_triple_first_pivot_move_pct),
                int(cfg.triple_reaction_max_bars),
                float(cfg.min_triple_reaction_score),
                body_pct,
                atr_pct,
                prominence_pct,
                float(cfg.peak_premove_body_mult),
                float(cfg.peak_level_tolerance_body_mult),
                float(cfg.peak_level_tolerance_atr_mult),
                float(cfg.peak_level_tolerance_prominence_mult),
                float(cfg.peak_reaction_body_mult),
                float(cfg.peak_base_return_buffer_body_mult),
                float(getattr(cfg, "peak_prior_impulse_min_efficiency", 0.0)),
                int(getattr(cfg, "peak_prior_impulse_min_bars", 1)),
            )
            if bottom["quality"] >= float(cfg.min_triple_quality):
                _store_triple(out, row, "triple_bottom", bottom)
    return out


def _empty_triple_output(rows: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in ("triple_top", "triple_bottom"):
        out[f"{name}_quality"] = np.zeros(rows, dtype="float64")
        out[f"{name}_setup_short" if name == "triple_top" else f"{name}_setup_long"] = np.zeros(rows, dtype=bool)
        for field in ("first_index", "second_index", "third_index", "first_price", "second_price", "third_price"):
            out[f"{name}_{field}"] = np.full(rows, np.nan, dtype="float64")
        out[f"{name}_neckline"] = np.full(rows, np.nan, dtype="float64")
        out[f"{name}_neckline_index"] = np.full(rows, np.nan, dtype="float64")
        out[f"{name}_level"] = np.full(rows, np.nan, dtype="float64")
        out[f"{name}_reaction_score"] = np.zeros(rows, dtype="float64")
    return out


def _score_triple_top(
    row: int,
    close: np.ndarray,
    body_low: np.ndarray,
    high_pivot: np.ndarray,
    high_index: np.ndarray,
    low_pivot: np.ndarray,
    low_index: np.ndarray,
    window: int,
    min_bars: int,
    max_bars: int,
    min_spacing: int,
    max_candidates: int,
    tolerance_pct: float,
    min_depth: float,
    min_prior_move: float,
    min_first_pivot_move: float,
    reaction_max_bars: int,
    min_reaction_score: float,
    body_pct: np.ndarray,
    atr_pct: np.ndarray,
    prominence_pct: np.ndarray,
    premove_body_mult: float,
    level_body_mult: float,
    level_atr_mult: float,
    level_prominence_mult: float,
    reaction_body_mult: float,
    base_buffer_body_mult: float,
    prior_impulse_min_efficiency: float,
    prior_impulse_min_bars: int,
) -> dict[str, float]:
    third_x = float(high_index[row])
    third_y = float(high_pivot[row])
    valid = np.isfinite(high_pivot[:row]) & np.isfinite(high_index[:row])
    prior_rows = np.flatnonzero(valid & (high_index[:row] >= third_x - window) & (high_index[:row] < third_x - min_spacing))
    if prior_rows.size > max_candidates:
        prior_rows = prior_rows[-max_candidates:]
    best = _empty_triple_candidate(third_x, third_y)
    body_ref = _threshold_body_pct(body_pct, row)
    atr_ref = _threshold_scale_pct(atr_pct, row, minimum=0.0005)
    prominence_ref = _threshold_scale_pct(prominence_pct, row)
    level_tolerance_pct = _dynamic_height_tolerance_pct(
        float(tolerance_pct),
        body_ref,
        atr_ref,
        prominence_ref,
        float(level_body_mult),
        float(level_atr_mult),
        float(level_prominence_mult),
    )
    move_threshold = max(float(min_prior_move), float(min_first_pivot_move), body_ref * float(premove_body_mult))
    reaction_threshold = max(float(min_depth) * 0.55, body_ref * float(reaction_body_mult))
    base_buffer_pct = body_ref * float(base_buffer_body_mult)
    for second_row in prior_rows:
        second_x = float(high_index[second_row])
        if third_x - second_x < min_spacing:
            continue
        first_candidates = prior_rows[high_index[prior_rows] < second_x - min_spacing]
        for first_row in first_candidates:
            first_x = float(high_index[first_row])
            span = third_x - first_x
            if span < min_bars or span > max_bars:
                continue
            prices = np.asarray([float(high_pivot[first_row]), float(high_pivot[second_row]), third_y], dtype="float64")
            level = float(np.nanmedian(prices))
            similarity = _same_level_score(prices, level, level_tolerance_pct)
            if similarity <= 0.0:
                continue
            overshoot_score = _triple_top_overshoot_score(high_pivot, high_index, row, first_x, third_x, level, level_tolerance_pct)
            if overshoot_score <= 0.0:
                continue
            prior = _prior_opposite_pivot_context(
                low_pivot,
                low_index,
                first_x,
                float(high_pivot[first_row]),
                window,
                top=True,
            )
            first_move_pct = float(prior["move_pct"])
            if first_move_pct < move_threshold:
                continue
            impulse_score = _prior_impulse_score(
                close,
                prior["index"],
                prior["price"],
                first_x,
                float(high_pivot[first_row]),
                top=True,
                min_bars=prior_impulse_min_bars,
            )
            if impulse_score < float(prior_impulse_min_efficiency):
                continue
            dominance_score = _top_p1_dominance_score(
                high_pivot,
                high_index,
                row,
                prior["index"],
                first_x,
                float(high_pivot[first_row]),
                level_tolerance_pct,
            )
            dominance_score = min(dominance_score, impulse_score)
            if dominance_score <= 0.0:
                continue
            if not _top_base_is_held(body_low, first_x, third_x, prior["price"], base_buffer_pct):
                continue
            neckline = _triple_top_body_reaction(
                body_low,
                first_x,
                second_x,
                third_x,
                level,
                reaction_threshold,
                reaction_max_bars,
            )
            if not np.isfinite([neckline["price"], neckline["index"]]).all():
                continue
            depth_pct = (level - neckline["price"]) / max(abs(float(close[row])), 1e-9)
            if depth_pct < reaction_threshold:
                continue
            reaction_pct = max((third_y - float(close[row])) / max(abs(float(close[row])), 1e-9), 0.0)
            reaction_score = _clip_value(max(reaction_pct, depth_pct * 0.35) / max(reaction_threshold, 1e-9))
            if reaction_score < min_reaction_score:
                continue
            quality = _peak_retest_quality(
                similarity,
                depth_pct,
                reaction_threshold,
                first_move_pct,
                move_threshold,
                max(reaction_pct, depth_pct * 0.35),
                dominance_score,
                overshoot_score,
                span,
                min_bars,
                max_bars,
            )
            if quality > best["quality"]:
                best = {
                    "quality": quality,
                    "first_index": first_x,
                    "second_index": second_x,
                    "third_index": third_x,
                    "first_price": float(prices[0]),
                    "second_price": float(prices[1]),
                    "third_price": third_y,
                    "neckline": float(neckline["price"]),
                    "neckline_index": float(neckline["index"]),
                    "level": level,
                    "reaction_score": reaction_score,
                }
    return best


def _score_triple_bottom(
    row: int,
    close: np.ndarray,
    body_high: np.ndarray,
    high_pivot: np.ndarray,
    high_index: np.ndarray,
    low_pivot: np.ndarray,
    low_index: np.ndarray,
    window: int,
    min_bars: int,
    max_bars: int,
    min_spacing: int,
    max_candidates: int,
    tolerance_pct: float,
    min_depth: float,
    min_prior_move: float,
    min_first_pivot_move: float,
    reaction_max_bars: int,
    min_reaction_score: float,
    body_pct: np.ndarray,
    atr_pct: np.ndarray,
    prominence_pct: np.ndarray,
    premove_body_mult: float,
    level_body_mult: float,
    level_atr_mult: float,
    level_prominence_mult: float,
    reaction_body_mult: float,
    base_buffer_body_mult: float,
    prior_impulse_min_efficiency: float,
    prior_impulse_min_bars: int,
) -> dict[str, float]:
    third_x = float(low_index[row])
    third_y = float(low_pivot[row])
    valid = np.isfinite(low_pivot[:row]) & np.isfinite(low_index[:row])
    prior_rows = np.flatnonzero(valid & (low_index[:row] >= third_x - window) & (low_index[:row] < third_x - min_spacing))
    if prior_rows.size > max_candidates:
        prior_rows = prior_rows[-max_candidates:]
    best = _empty_triple_candidate(third_x, third_y)
    body_ref = _threshold_body_pct(body_pct, row)
    atr_ref = _threshold_scale_pct(atr_pct, row, minimum=0.0005)
    prominence_ref = _threshold_scale_pct(prominence_pct, row)
    level_tolerance_pct = _dynamic_height_tolerance_pct(
        float(tolerance_pct),
        body_ref,
        atr_ref,
        prominence_ref,
        float(level_body_mult),
        float(level_atr_mult),
        float(level_prominence_mult),
    )
    move_threshold = max(float(min_prior_move), float(min_first_pivot_move), body_ref * float(premove_body_mult))
    reaction_threshold = max(float(min_depth) * 0.55, body_ref * float(reaction_body_mult))
    base_buffer_pct = body_ref * float(base_buffer_body_mult)
    for second_row in prior_rows:
        second_x = float(low_index[second_row])
        if third_x - second_x < min_spacing:
            continue
        first_candidates = prior_rows[low_index[prior_rows] < second_x - min_spacing]
        for first_row in first_candidates:
            first_x = float(low_index[first_row])
            span = third_x - first_x
            if span < min_bars or span > max_bars:
                continue
            prices = np.asarray([float(low_pivot[first_row]), float(low_pivot[second_row]), third_y], dtype="float64")
            level = float(np.nanmedian(prices))
            similarity = _same_level_score(prices, level, level_tolerance_pct)
            if similarity <= 0.0:
                continue
            undershoot_score = _triple_bottom_undershoot_score(low_pivot, low_index, row, first_x, third_x, level, level_tolerance_pct)
            if undershoot_score <= 0.0:
                continue
            prior = _prior_opposite_pivot_context(
                high_pivot,
                high_index,
                first_x,
                float(low_pivot[first_row]),
                window,
                top=False,
            )
            first_move_pct = float(prior["move_pct"])
            if first_move_pct < move_threshold:
                continue
            impulse_score = _prior_impulse_score(
                close,
                prior["index"],
                prior["price"],
                first_x,
                float(low_pivot[first_row]),
                top=False,
                min_bars=prior_impulse_min_bars,
            )
            if impulse_score < float(prior_impulse_min_efficiency):
                continue
            dominance_score = _bottom_p1_dominance_score(
                low_pivot,
                low_index,
                row,
                prior["index"],
                first_x,
                float(low_pivot[first_row]),
                level_tolerance_pct,
            )
            dominance_score = min(dominance_score, impulse_score)
            if dominance_score <= 0.0:
                continue
            if not _bottom_base_is_held(body_high, first_x, third_x, prior["price"], base_buffer_pct):
                continue
            neckline = _triple_bottom_body_reaction(
                body_high,
                first_x,
                second_x,
                third_x,
                level,
                reaction_threshold,
                reaction_max_bars,
            )
            if not np.isfinite([neckline["price"], neckline["index"]]).all():
                continue
            depth_pct = (neckline["price"] - level) / max(abs(float(close[row])), 1e-9)
            if depth_pct < reaction_threshold:
                continue
            reaction_pct = max((float(close[row]) - third_y) / max(abs(float(close[row])), 1e-9), 0.0)
            reaction_score = _clip_value(max(reaction_pct, depth_pct * 0.35) / max(reaction_threshold, 1e-9))
            if reaction_score < min_reaction_score:
                continue
            quality = _peak_retest_quality(
                similarity,
                depth_pct,
                reaction_threshold,
                first_move_pct,
                move_threshold,
                max(reaction_pct, depth_pct * 0.35),
                dominance_score,
                undershoot_score,
                span,
                min_bars,
                max_bars,
            )
            if quality > best["quality"]:
                best = {
                    "quality": quality,
                    "first_index": first_x,
                    "second_index": second_x,
                    "third_index": third_x,
                    "first_price": float(prices[0]),
                    "second_price": float(prices[1]),
                    "third_price": third_y,
                    "neckline": float(neckline["price"]),
                    "neckline_index": float(neckline["index"]),
                    "level": level,
                    "reaction_score": reaction_score,
                }
    return best


def _empty_triple_candidate(third_x: float, third_y: float) -> dict[str, float]:
    return {
        "quality": 0.0,
        "first_index": np.nan,
        "second_index": np.nan,
        "third_index": third_x,
        "first_price": np.nan,
        "second_price": np.nan,
        "third_price": third_y,
        "neckline": np.nan,
        "neckline_index": np.nan,
        "level": np.nan,
        "reaction_score": 0.0,
    }


def _same_level_score(prices: np.ndarray, level: float, tolerance_pct: float) -> float:
    reference = max(abs(float(level)), 1e-9)
    max_deviation = float(np.nanmax(np.abs(prices - float(level)))) / reference
    return _clip_value(1.0 - max_deviation / max(float(tolerance_pct), 1e-9))


def _triple_top_overshoot_score(
    high_pivot: np.ndarray,
    high_index: np.ndarray,
    row: int,
    first_x: float,
    third_x: float,
    level: float,
    tolerance_pct: float,
) -> float:
    confirmed = np.arange(len(high_pivot)) <= row
    between = confirmed & np.isfinite(high_pivot) & np.isfinite(high_index) & (high_index >= first_x) & (high_index <= third_x)
    if not between.any():
        return 1.0
    overshoot = max(float(np.nanmax(high_pivot[between])) - float(level), 0.0)
    return _clip_value(1.0 - overshoot / max(abs(float(level)) * float(tolerance_pct), 1e-9))


def _triple_bottom_undershoot_score(
    low_pivot: np.ndarray,
    low_index: np.ndarray,
    row: int,
    first_x: float,
    third_x: float,
    level: float,
    tolerance_pct: float,
) -> float:
    confirmed = np.arange(len(low_pivot)) <= row
    between = confirmed & np.isfinite(low_pivot) & np.isfinite(low_index) & (low_index >= first_x) & (low_index <= third_x)
    if not between.any():
        return 1.0
    undershoot = max(float(level) - float(np.nanmin(low_pivot[between])), 0.0)
    return _clip_value(1.0 - undershoot / max(abs(float(level)) * float(tolerance_pct), 1e-9))


def _triple_top_body_reaction(
    body_low: np.ndarray,
    first_x: float,
    second_x: float,
    third_x: float,
    level: float,
    min_depth: float,
    reaction_max_bars: int,
) -> dict[str, float]:
    left = _body_low_reaction(body_low, first_x, second_x, level, reaction_max_bars)
    right = _body_low_reaction(body_low, second_x, third_x, level, reaction_max_bars)
    if left["depth_pct"] < float(min_depth) or right["depth_pct"] < float(min_depth):
        return {"price": np.nan, "index": np.nan}
    prices = np.asarray([left["price"], right["price"]], dtype="float64")
    indexes = np.asarray([left["index"], right["index"]], dtype="float64")
    return {"price": float(np.nanmedian(prices)), "index": float(np.nanmean(indexes))}


def _triple_bottom_body_reaction(
    body_high: np.ndarray,
    first_x: float,
    second_x: float,
    third_x: float,
    level: float,
    min_depth: float,
    reaction_max_bars: int,
) -> dict[str, float]:
    left = _body_high_reaction(body_high, first_x, second_x, level, reaction_max_bars)
    right = _body_high_reaction(body_high, second_x, third_x, level, reaction_max_bars)
    if left["depth_pct"] < float(min_depth) or right["depth_pct"] < float(min_depth):
        return {"price": np.nan, "index": np.nan}
    prices = np.asarray([left["price"], right["price"]], dtype="float64")
    indexes = np.asarray([left["index"], right["index"]], dtype="float64")
    return {"price": float(np.nanmedian(prices)), "index": float(np.nanmean(indexes))}


def _body_low_reaction(
    body_low: np.ndarray,
    start_x: float,
    end_x: float,
    level: float,
    reaction_max_bars: int,
) -> dict[str, float]:
    start = int(max(np.floor(start_x), 0))
    stop = int(min(np.ceil(min(float(end_x), float(start_x) + float(reaction_max_bars))), len(body_low) - 1))
    if stop <= start or not np.isfinite(level) or level == 0.0:
        return {"price": np.nan, "index": np.nan, "depth_pct": 0.0}
    lows = np.asarray(body_low[start : stop + 1], dtype="float64")
    if not np.isfinite(lows).any():
        return {"price": np.nan, "index": np.nan, "depth_pct": 0.0}
    local = int(np.nanargmin(lows))
    price = float(lows[local])
    return {"price": price, "index": float(start + local), "depth_pct": max((float(level) - price) / max(abs(float(level)), 1e-9), 0.0)}


def _body_high_reaction(
    body_high: np.ndarray,
    start_x: float,
    end_x: float,
    level: float,
    reaction_max_bars: int,
) -> dict[str, float]:
    start = int(max(np.floor(start_x), 0))
    stop = int(min(np.ceil(min(float(end_x), float(start_x) + float(reaction_max_bars))), len(body_high) - 1))
    if stop <= start or not np.isfinite(level) or level == 0.0:
        return {"price": np.nan, "index": np.nan, "depth_pct": 0.0}
    highs = np.asarray(body_high[start : stop + 1], dtype="float64")
    if not np.isfinite(highs).any():
        return {"price": np.nan, "index": np.nan, "depth_pct": 0.0}
    local = int(np.nanargmax(highs))
    price = float(highs[local])
    return {"price": price, "index": float(start + local), "depth_pct": max((price - float(level)) / max(abs(float(level)), 1e-9), 0.0)}


def _store_triple(out: dict[str, np.ndarray], row: int, name: str, candidate: dict[str, float]) -> None:
    out[f"{name}_quality"][row] = float(candidate["quality"])
    setup_key = f"{name}_setup_short" if name == "triple_top" else f"{name}_setup_long"
    out[setup_key][row] = True
    for field in ("first_index", "second_index", "third_index", "first_price", "second_price", "third_price"):
        out[f"{name}_{field}"][row] = float(candidate[field])
    out[f"{name}_neckline"][row] = float(candidate["neckline"])
    out[f"{name}_neckline_index"][row] = float(candidate["neckline_index"])
    out[f"{name}_level"][row] = float(candidate["level"])
    out[f"{name}_reaction_score"][row] = float(candidate["reaction_score"])


__all__ = ["PatternPeakConfig", "add_pattern_peaks", "_triple_reversal_columns"]

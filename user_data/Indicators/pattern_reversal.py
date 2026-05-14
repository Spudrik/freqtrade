from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from pattern_common import (
    _clip_value,
    _carry_values_while_state,
    _line_value_at,
    _pattern_geometry_arrays,
    _bottom_base_is_held,
    _bottom_p1_dominance_score,
    _peak_confirmation_state,
    _peak_retest_quality,
    _num,
    _prior_pattern_move,
    _prior_impulse_score,
    _prior_opposite_pivot_context,
    _proof_line_columns,
    _reaction_level_between,
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
class PatternReversalConfig:
    output_prefix: str = "pat"
    timeframe: str = "4h"
    pivot_prefix: str = "pf"
    atr_period: int = 14
    pivot_strength: int = 2
    pivot_min_prominence_atr: float = 0.35
    pivot_min_spacing_bars: int = 1
    pivot_min_distance_atr: float = 0.0
    pattern_pivot_strength: int = 1
    entry_cooldown_bars: int = 8
    pattern_lifecycle_mature_bars: int = 12
    pattern_lifecycle_stale_bars: int = 12
    duplicate_overlap_ratio: float = 0.70
    duplicate_neckline_tolerance_body_mult: float = 2.0
    duplicate_neckline_tolerance_atr_mult: float = 0.50
    peak_dynamic_body_window: int = 30
    peak_dynamic_scale_window: int = 30
    peak_premove_body_mult: float = 6.0
    peak_premove_atr_mult: float = 2.0
    peak_level_tolerance_body_mult: float = 2.25
    peak_level_tolerance_atr_mult: float = 0.35
    peak_level_tolerance_prominence_mult: float = 0.20
    peak_depth_body_mult: float = 4.0
    peak_depth_atr_mult: float = 1.0
    peak_reaction_body_mult: float = 3.0
    peak_base_return_buffer_body_mult: float = 0.8
    peak_prior_impulse_min_bars: int = 3
    peak_prior_impulse_min_efficiency: float = 0.40
    double_pattern_window: int = 80
    min_double_pattern_bars: int = 8
    max_double_pattern_bars: int = 64
    double_reaction_max_bars: int = 24
    min_double_quality: float = 0.72
    min_double_reaction_score: float = 0.12
    min_double_between_cleanliness_score: float = 0.38
    head_shoulders_window: int = 100
    min_head_shoulders_bars: int = 12
    max_head_shoulders_bars: int = 90
    head_shoulders_premove_body_mult: float = 6.0
    head_shoulders_premove_atr_mult: float = 2.0
    head_shoulders_p1_prior_window: int = 48
    head_shoulders_p1_prior_atr_mult: float = 3.0
    head_shoulders_prominence_body_mult: float = 1.5
    head_shoulders_prominence_atr_mult: float = 0.35
    head_shoulders_neckline_depth_body_mult: float = 4.0
    head_shoulders_neckline_depth_atr_mult: float = 1.0
    head_shoulders_base_return_buffer_body_mult: float = 0.8
    min_head_shoulders_p1_dominance_score: float = 0.0
    min_head_shoulders_peak_cleanliness_score: float = 0.20
    min_head_shoulders_quality: float = 0.72
    min_head_shoulders_shoulder_score: float = 0.45
    min_head_shoulders_time_balance_score: float = 0.45
    min_head_shoulders_neckline_score: float = 0.35
    min_head_shoulders_neckline_position_score: float = 0.25
    min_head_shoulders_head_position_score: float = 0.20
    min_head_shoulders_neckline_clearance_score: float = 0.22
    min_head_shoulders_neckline_body_respect_ratio: float = 0.70
    head_shoulders_max_neckline_slope_atr_per_bar: float = 0.15
    min_head_shoulders_right_reaction_score: float = 0.10
    head_shoulders_setup_monitor_bars: int = 16
    head_shoulders_neckline_proximity_body_mult: float = 3.0
    head_shoulders_neckline_proximity_atr_mult: float = 0.50
    head_shoulders_max_candidate_pivots: int = 14
    include_pattern_diagnostics: bool = False


def add_pattern_reversal(
    dataframe: DataFrame,
    timeframe: str = "4h",
    config: PatternReversalConfig | None = None,
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
        pivot_min_prominence_pct=0.0,
        pivot_min_spacing_bars=int(cfg.pivot_min_spacing_bars),
        pivot_min_distance_atr=float(cfg.pivot_min_distance_atr),
        pivot_min_distance_pct=0.0,
    )
    columns = {**_double_reversal_columns(frame, cfg), **_head_shoulders_columns(frame, cfg)}
    existing = [
        column
        for column in dataframe.columns
        if str(column).startswith(
            (
                f"{cfg.output_prefix}_double_top_",
                f"{cfg.output_prefix}_double_bottom_",
                f"{cfg.output_prefix}_head_shoulders_",
                f"{cfg.output_prefix}_inverse_head_shoulders_",
            )
        )
    ]
    clean = dataframe.drop(columns=existing).copy() if existing else dataframe.copy()
    return pd.concat([clean, pd.DataFrame(columns, index=frame.index)], axis=1)


def _resolve_config(
    config: PatternReversalConfig | None,
    timeframe: str,
    overrides: dict[str, object],
) -> PatternReversalConfig:
    cfg = config or PatternReversalConfig(timeframe=str(timeframe))
    valid = {field.name for field in fields(PatternReversalConfig)}
    clean = {key: value for key, value in overrides.items() if value is not None}
    unknown = sorted(set(clean).difference(valid))
    if unknown:
        raise TypeError(f"Unknown reversal config override(s): {', '.join(unknown)}")
    if "timeframe" not in clean:
        clean["timeframe"] = str(timeframe)
    return replace(cfg, **clean) if clean else cfg


def _validate_ohlcv(dataframe: DataFrame) -> None:
    missing = [column for column in ("open", "high", "low", "close") if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Dataframe missing required columns: {', '.join(missing)}")
    if dataframe.empty:
        raise ValueError("Dataframe must not be empty")


def _dynamic_pattern_scale_pct(
    body_ref_pct: float,
    atr_ref_pct: float,
    prominence_ref_pct: float = 0.0,
    *,
    body_mult: float = 0.0,
    atr_mult: float = 0.0,
    prominence_mult: float = 0.0,
    minimum: float = 1e-9,
) -> float:
    """Return a local scale without a fixed price-percent floor or cap.

    Reversal patterns compare prices as normalized distances internally, but the
    thresholds must come from the candle regime being scored. This keeps a quiet
    1h consolidation, a volatile 4h reversal, and a 3d swing from inheriting the
    same absolute percentage gate.
    """

    return float(
        max(
            float(body_ref_pct) * float(body_mult),
            float(atr_ref_pct) * float(atr_mult),
            float(prominence_ref_pct) * float(prominence_mult),
            float(minimum),
        )
    )


def _double_reversal_columns(frame: DataFrame, cfg: PatternReversalConfig) -> dict[str, Series]:
    """Detect double top / double bottom attention at the second confirmed pivot."""

    p = cfg.output_prefix
    arrays = _double_reversal_arrays(frame, cfg)
    raw_top = pd.Series(arrays["double_top_structure_short"], index=frame.index, dtype="bool").fillna(False)
    raw_bottom = pd.Series(arrays["double_bottom_structure_long"], index=frame.index, dtype="bool").fillna(False)
    close = _num(frame["close"]).to_numpy(dtype="float64")
    top = pd.Series(
        _dedupe_double_events(
            arrays["double_top_structure_short"],
            arrays["double_top_first_index"],
            arrays["double_top_second_index"],
            arrays["double_top_neckline"],
            arrays["double_top_duplicate_tolerance_pct"],
            int(cfg.entry_cooldown_bars),
            float(cfg.duplicate_overlap_ratio),
        ),
        index=frame.index,
        dtype="bool",
    )
    bottom = pd.Series(
        _dedupe_double_events(
            arrays["double_bottom_structure_long"],
            arrays["double_bottom_first_index"],
            arrays["double_bottom_second_index"],
            arrays["double_bottom_neckline"],
            arrays["double_bottom_duplicate_tolerance_pct"],
            int(cfg.entry_cooldown_bars),
            float(cfg.duplicate_overlap_ratio),
        ),
        index=frame.index,
        dtype="bool",
    )
    top_confirmation = _peak_confirmation_state(
        close,
        top,
        arrays["double_top_neckline"],
        arrays["double_top_quality"],
        int(cfg.pattern_lifecycle_mature_bars),
        int(cfg.pattern_lifecycle_stale_bars),
        top=True,
    )
    bottom_confirmation = _peak_confirmation_state(
        close,
        bottom,
        arrays["double_bottom_neckline"],
        arrays["double_bottom_quality"],
        int(cfg.pattern_lifecycle_mature_bars),
        int(cfg.pattern_lifecycle_stale_bars),
        top=False,
    )
    top_neckline_score = pd.Series(arrays["double_top_neckline_score"], index=frame.index, dtype="float64")
    bottom_neckline_score = pd.Series(arrays["double_bottom_neckline_score"], index=frame.index, dtype="float64")
    top_reaction_score = pd.Series(arrays["double_top_reaction_score"], index=frame.index, dtype="float64")
    bottom_reaction_score = pd.Series(arrays["double_bottom_reaction_score"], index=frame.index, dtype="float64")
    top_cleanliness_score = pd.Series(arrays["double_top_between_cleanliness_score"], index=frame.index, dtype="float64")
    bottom_cleanliness_score = pd.Series(arrays["double_bottom_between_cleanliness_score"], index=frame.index, dtype="float64")
    top_quality = pd.Series(arrays["double_top_quality"], index=frame.index, dtype="float64")
    bottom_quality = pd.Series(arrays["double_bottom_quality"], index=frame.index, dtype="float64")
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
            "p1_index": arrays["double_top_first_index"],
            "p2_index": arrays["double_top_second_index"],
            "p1_price": arrays["double_top_first_price"],
            "p2_price": arrays["double_top_second_price"],
            "neckline": arrays["double_top_neckline"],
        },
    )
    bottom_carried = _carry_values_while_state(
        bottom,
        bottom_state,
        {
            "p1_index": arrays["double_bottom_first_index"],
            "p2_index": arrays["double_bottom_second_index"],
            "p1_price": arrays["double_bottom_first_price"],
            "p2_price": arrays["double_bottom_second_price"],
            "neckline": arrays["double_bottom_neckline"],
        },
    )
    top_present = top_developing | top_confirmed
    bottom_present = bottom_developing | bottom_confirmed
    top_confirmation_level = pd.Series(top_carried["neckline"], index=frame.index, dtype="float64")
    bottom_confirmation_level = pd.Series(bottom_carried["neckline"], index=frame.index, dtype="float64")
    top_level = pd.concat(
        [
            pd.Series(top_carried["p1_price"], index=frame.index, dtype="float64"),
            pd.Series(top_carried["p2_price"], index=frame.index, dtype="float64"),
        ],
        axis=1,
    ).mean(axis=1)
    bottom_level = pd.concat(
        [
            pd.Series(bottom_carried["p1_price"], index=frame.index, dtype="float64"),
            pd.Series(bottom_carried["p2_price"], index=frame.index, dtype="float64"),
        ],
        axis=1,
    ).mean(axis=1)
    columns = {
        f"{p}_double_top_pattern_present": top_present.fillna(False),
        f"{p}_double_bottom_pattern_present": bottom_present.fillna(False),
        f"{p}_double_top_pattern_confirmed": top_confirmed.fillna(False),
        f"{p}_double_bottom_pattern_confirmed": bottom_confirmed.fillna(False),
        f"{p}_double_top_indicator_score": top_current_quality.where(top_present, 0.0),
        f"{p}_double_bottom_indicator_score": bottom_current_quality.where(bottom_present, 0.0),
        f"{p}_double_top_confirmation_level": top_confirmation_level,
        f"{p}_double_bottom_confirmation_level": bottom_confirmation_level,
        f"{p}_double_top_target_level": top_confirmation_level - (top_level - top_confirmation_level).clip(lower=0.0),
        f"{p}_double_bottom_target_level": bottom_confirmation_level + (bottom_confirmation_level - bottom_level).clip(lower=0.0),
        f"{p}_double_top_p1_index": pd.Series(top_carried["p1_index"], index=frame.index, dtype="float64"),
        f"{p}_double_top_p2_index": pd.Series(top_carried["p2_index"], index=frame.index, dtype="float64"),
        f"{p}_double_bottom_p1_index": pd.Series(bottom_carried["p1_index"], index=frame.index, dtype="float64"),
        f"{p}_double_bottom_p2_index": pd.Series(bottom_carried["p2_index"], index=frame.index, dtype="float64"),
    }
    if bool(getattr(cfg, "include_pattern_diagnostics", False)):
        columns.update(
            {
                f"{p}_double_top_structure_short": raw_top,
                f"{p}_double_bottom_structure_long": raw_bottom,
                f"{p}_double_top_structure_quality": top_quality.where(raw_top, 0.0),
                f"{p}_double_bottom_structure_quality": bottom_quality.where(raw_bottom, 0.0),
                f"{p}_double_top_neckline_score": top_neckline_score.where(raw_top, 0.0),
                f"{p}_double_top_reaction_score": top_reaction_score.where(raw_top, 0.0),
                f"{p}_double_top_between_cleanliness_score": top_cleanliness_score.where(raw_top, 0.0),
                f"{p}_double_bottom_neckline_score": bottom_neckline_score.where(raw_bottom, 0.0),
                f"{p}_double_bottom_reaction_score": bottom_reaction_score.where(raw_bottom, 0.0),
                f"{p}_double_bottom_between_cleanliness_score": bottom_cleanliness_score.where(raw_bottom, 0.0),
                **_proof_line_columns(
                    frame.index,
                    f"{p}_double_top_short",
                    top,
                    {
                        1: (
                            arrays["double_top_first_index"],
                            arrays["double_top_first_price"],
                            arrays["double_top_second_index"],
                            arrays["double_top_second_price"],
                        ),
                        2: (
                            arrays["double_top_first_index"],
                            arrays["double_top_neckline"],
                            arrays["double_top_second_index"],
                            arrays["double_top_neckline"],
                        ),
                    },
                ),
                **_proof_line_columns(
                    frame.index,
                    f"{p}_double_bottom_long",
                    bottom,
                    {
                        1: (
                            arrays["double_bottom_first_index"],
                            arrays["double_bottom_first_price"],
                            arrays["double_bottom_second_index"],
                            arrays["double_bottom_second_price"],
                        ),
                        2: (
                            arrays["double_bottom_first_index"],
                            arrays["double_bottom_neckline"],
                            arrays["double_bottom_second_index"],
                            arrays["double_bottom_neckline"],
                        ),
                    },
                ),
            }
        )
    return columns


def _double_reversal_arrays(frame: DataFrame, cfg: PatternReversalConfig) -> dict[str, np.ndarray]:
    close, body_high, body_low, high_pivot, high_index, low_pivot, low_index = _pattern_geometry_arrays(frame, cfg)
    rows = len(frame)
    out = {
        "double_top_quality": np.zeros(rows, dtype="float64"),
        "double_bottom_quality": np.zeros(rows, dtype="float64"),
        "double_top_structure_short": np.zeros(rows, dtype=bool),
        "double_bottom_structure_long": np.zeros(rows, dtype=bool),
        "double_top_setup_short": np.zeros(rows, dtype=bool),
        "double_bottom_setup_long": np.zeros(rows, dtype=bool),
        "double_top_first_index": np.full(rows, np.nan, dtype="float64"),
        "double_top_second_index": np.full(rows, np.nan, dtype="float64"),
        "double_top_first_price": np.full(rows, np.nan, dtype="float64"),
        "double_top_second_price": np.full(rows, np.nan, dtype="float64"),
        "double_top_neckline_index": np.full(rows, np.nan, dtype="float64"),
        "double_top_neckline": np.full(rows, np.nan, dtype="float64"),
        "double_top_duplicate_tolerance_pct": np.zeros(rows, dtype="float64"),
        "double_top_neckline_score": np.zeros(rows, dtype="float64"),
        "double_top_reaction_score": np.zeros(rows, dtype="float64"),
        "double_top_between_cleanliness_score": np.zeros(rows, dtype="float64"),
        "double_bottom_first_index": np.full(rows, np.nan, dtype="float64"),
        "double_bottom_second_index": np.full(rows, np.nan, dtype="float64"),
        "double_bottom_first_price": np.full(rows, np.nan, dtype="float64"),
        "double_bottom_second_price": np.full(rows, np.nan, dtype="float64"),
        "double_bottom_neckline_index": np.full(rows, np.nan, dtype="float64"),
        "double_bottom_neckline": np.full(rows, np.nan, dtype="float64"),
        "double_bottom_duplicate_tolerance_pct": np.zeros(rows, dtype="float64"),
        "double_bottom_neckline_score": np.zeros(rows, dtype="float64"),
        "double_bottom_reaction_score": np.zeros(rows, dtype="float64"),
        "double_bottom_between_cleanliness_score": np.zeros(rows, dtype="float64"),
    }
    window = int(cfg.double_pattern_window)
    min_bars = int(cfg.min_double_pattern_bars)
    max_bars = int(cfg.max_double_pattern_bars)
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
            top = _score_double_top(
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
                int(cfg.double_reaction_max_bars),
                float(cfg.min_double_reaction_score),
                float(cfg.min_double_between_cleanliness_score),
                body_pct,
                atr_pct,
                prominence_pct,
                float(cfg.peak_premove_body_mult),
                float(cfg.peak_premove_atr_mult),
                float(cfg.peak_level_tolerance_body_mult),
                float(cfg.peak_level_tolerance_atr_mult),
                float(cfg.peak_level_tolerance_prominence_mult),
                float(cfg.peak_depth_body_mult),
                float(cfg.peak_depth_atr_mult),
                float(cfg.peak_reaction_body_mult),
                float(cfg.peak_base_return_buffer_body_mult),
                float(cfg.duplicate_neckline_tolerance_body_mult),
                float(cfg.duplicate_neckline_tolerance_atr_mult),
                float(getattr(cfg, "peak_prior_impulse_min_efficiency", 0.0)),
                int(getattr(cfg, "peak_prior_impulse_min_bars", 1)),
            )
            if top["quality"] >= float(cfg.min_double_quality):
                out["double_top_quality"][row] = float(top["quality"])
                out["double_top_structure_short"][row] = True
                out["double_top_setup_short"][row] = True
                out["double_top_first_index"][row] = float(top["first_index"])
                out["double_top_second_index"][row] = float(top["second_index"])
                out["double_top_first_price"][row] = float(top["first_price"])
                out["double_top_second_price"][row] = float(top["second_price"])
                out["double_top_neckline_index"][row] = float(top["neckline_index"])
                out["double_top_neckline"][row] = float(top["neckline"])
                out["double_top_duplicate_tolerance_pct"][row] = float(top["duplicate_tolerance_pct"])
                out["double_top_neckline_score"][row] = float(top["neckline_score"])
                out["double_top_reaction_score"][row] = float(top["reaction_score"])
                out["double_top_between_cleanliness_score"][row] = float(top["between_cleanliness_score"])
        if np.isfinite(low_pivot[row]) and np.isfinite(low_index[row]):
            bottom = _score_double_bottom(
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
                int(cfg.double_reaction_max_bars),
                float(cfg.min_double_reaction_score),
                float(cfg.min_double_between_cleanliness_score),
                body_pct,
                atr_pct,
                prominence_pct,
                float(cfg.peak_premove_body_mult),
                float(cfg.peak_premove_atr_mult),
                float(cfg.peak_level_tolerance_body_mult),
                float(cfg.peak_level_tolerance_atr_mult),
                float(cfg.peak_level_tolerance_prominence_mult),
                float(cfg.peak_depth_body_mult),
                float(cfg.peak_depth_atr_mult),
                float(cfg.peak_reaction_body_mult),
                float(cfg.peak_base_return_buffer_body_mult),
                float(cfg.duplicate_neckline_tolerance_body_mult),
                float(cfg.duplicate_neckline_tolerance_atr_mult),
                float(getattr(cfg, "peak_prior_impulse_min_efficiency", 0.0)),
                int(getattr(cfg, "peak_prior_impulse_min_bars", 1)),
            )
            if bottom["quality"] >= float(cfg.min_double_quality):
                out["double_bottom_quality"][row] = float(bottom["quality"])
                out["double_bottom_structure_long"][row] = True
                out["double_bottom_setup_long"][row] = True
                out["double_bottom_first_index"][row] = float(bottom["first_index"])
                out["double_bottom_second_index"][row] = float(bottom["second_index"])
                out["double_bottom_first_price"][row] = float(bottom["first_price"])
                out["double_bottom_second_price"][row] = float(bottom["second_price"])
                out["double_bottom_neckline_index"][row] = float(bottom["neckline_index"])
                out["double_bottom_neckline"][row] = float(bottom["neckline"])
                out["double_bottom_duplicate_tolerance_pct"][row] = float(bottom["duplicate_tolerance_pct"])
                out["double_bottom_neckline_score"][row] = float(bottom["neckline_score"])
                out["double_bottom_reaction_score"][row] = float(bottom["reaction_score"])
                out["double_bottom_between_cleanliness_score"][row] = float(bottom["between_cleanliness_score"])
    return out


def _score_double_top(
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
    reaction_max_bars: int,
    min_reaction_score: float,
    min_between_cleanliness_score: float,
    body_pct: np.ndarray,
    atr_pct: np.ndarray,
    prominence_pct: np.ndarray,
    premove_body_mult: float,
    premove_atr_mult: float,
    level_body_mult: float,
    level_atr_mult: float,
    level_prominence_mult: float,
    depth_body_mult: float,
    depth_atr_mult: float,
    reaction_body_mult: float,
    base_buffer_body_mult: float,
    duplicate_tolerance_body_mult: float,
    duplicate_tolerance_atr_mult: float,
    prior_impulse_min_efficiency: float,
    prior_impulse_min_bars: int,
) -> dict[str, float]:
    second_x = float(high_index[row])
    second_y = float(high_pivot[row])
    high_mask = np.isfinite(high_pivot[:row]) & np.isfinite(high_index[:row])
    candidates = np.flatnonzero(high_mask & (high_index[:row] < second_x - min_bars) & (high_index[:row] >= second_x - window))
    best = {
        "quality": 0.0,
        "first_index": np.nan,
        "second_index": second_x,
        "first_price": np.nan,
        "second_price": second_y,
        "neckline_index": np.nan,
        "neckline": np.nan,
        "duplicate_tolerance_pct": 0.0,
        "neckline_score": 0.0,
        "reaction_score": 0.0,
        "between_cleanliness_score": 0.0,
    }
    body_ref = _threshold_body_pct(body_pct, row)
    atr_ref = _threshold_scale_pct(atr_pct, row, minimum=0.0005)
    prominence_ref = _threshold_scale_pct(prominence_pct, row)
    level_tolerance_pct = _dynamic_pattern_scale_pct(
        body_ref,
        atr_ref,
        prominence_ref,
        body_mult=level_body_mult,
        atr_mult=level_atr_mult,
        prominence_mult=level_prominence_mult,
    )
    move_threshold = _dynamic_pattern_scale_pct(
        body_ref,
        atr_ref,
        body_mult=premove_body_mult,
        atr_mult=premove_atr_mult,
    )
    depth_threshold = _dynamic_pattern_scale_pct(
        body_ref,
        atr_ref,
        body_mult=depth_body_mult,
        atr_mult=depth_atr_mult,
    )
    reaction_threshold = max(depth_threshold * 0.55, body_ref * float(reaction_body_mult))
    base_buffer_pct = body_ref * float(base_buffer_body_mult)
    duplicate_tolerance_pct = _dynamic_pattern_scale_pct(
        body_ref,
        atr_ref,
        body_mult=duplicate_tolerance_body_mult,
        atr_mult=duplicate_tolerance_atr_mult,
    )
    for candidate in candidates:
        first_x = float(high_index[candidate])
        span = second_x - first_x
        if span < min_bars or span > max_bars:
            continue
        first_y = float(high_pivot[candidate])
        peak_ref = max((first_y + second_y) / 2.0, 1e-9)
        similarity = 1.0 - abs(second_y - first_y) / max(peak_ref * level_tolerance_pct, 1e-9)
        if similarity <= 0.0:
            continue
        between_cleanliness_score = _double_top_between_cleanliness_score(
            high_pivot,
            high_index,
            row,
            first_x,
            second_x,
            max(first_y, second_y),
            peak_ref,
            level_tolerance_pct,
        )
        if between_cleanliness_score < float(min_between_cleanliness_score):
            continue
        prior = _prior_opposite_pivot_context(
            low_pivot,
            low_index,
            first_x,
            first_y,
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
            first_y,
            top=True,
            min_bars=prior_impulse_min_bars,
        )
        if impulse_score < float(prior_impulse_min_efficiency):
            continue
        dominance_score = _top_p1_dominance_score(high_pivot, high_index, row, prior["index"], first_x, first_y, level_tolerance_pct)
        dominance_score = min(dominance_score, impulse_score)
        if dominance_score <= 0.0:
            continue
        if not _top_base_is_held(body_low, first_x, second_x, prior["price"], base_buffer_pct):
            continue
        first_reaction_pct = _top_body_reaction_pct(body_low, first_x, second_x, peak_ref, reaction_max_bars)
        if first_reaction_pct < reaction_threshold:
            continue
        neckline_index, neckline = _reaction_level_between(body_low, first_x, second_x, find_low=True)
        if not np.isfinite([neckline_index, neckline]).all():
            continue
        depth_pct = (peak_ref - neckline) / max(abs(close[row]), 1e-9)
        turn_pct = max((second_y - float(close[row])) / max(abs(float(close[row])), 1e-9), 0.0)
        reaction_score = _clip_value(first_reaction_pct / max(reaction_threshold, 1e-9))
        if reaction_score < float(min_reaction_score):
            continue
        quality = _peak_retest_quality(
            similarity,
            depth_pct,
            depth_threshold,
            first_move_pct,
            move_threshold,
            max(turn_pct, first_reaction_pct * 0.35),
            dominance_score,
            between_cleanliness_score,
            span,
            min_bars,
            max_bars,
        )
        if quality > best["quality"]:
            best = {
                "quality": quality,
                "first_index": first_x,
                "second_index": second_x,
                "first_price": first_y,
                "second_price": second_y,
                "neckline_index": neckline_index,
                "neckline": neckline,
                "duplicate_tolerance_pct": duplicate_tolerance_pct,
                "neckline_score": dominance_score,
                "reaction_score": reaction_score,
                "between_cleanliness_score": between_cleanliness_score,
            }
    return best


def _score_double_bottom(
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
    reaction_max_bars: int,
    min_reaction_score: float,
    min_between_cleanliness_score: float,
    body_pct: np.ndarray,
    atr_pct: np.ndarray,
    prominence_pct: np.ndarray,
    premove_body_mult: float,
    premove_atr_mult: float,
    level_body_mult: float,
    level_atr_mult: float,
    level_prominence_mult: float,
    depth_body_mult: float,
    depth_atr_mult: float,
    reaction_body_mult: float,
    base_buffer_body_mult: float,
    duplicate_tolerance_body_mult: float,
    duplicate_tolerance_atr_mult: float,
    prior_impulse_min_efficiency: float,
    prior_impulse_min_bars: int,
) -> dict[str, float]:
    second_x = float(low_index[row])
    second_y = float(low_pivot[row])
    low_mask = np.isfinite(low_pivot[:row]) & np.isfinite(low_index[:row])
    candidates = np.flatnonzero(low_mask & (low_index[:row] < second_x - min_bars) & (low_index[:row] >= second_x - window))
    best = {
        "quality": 0.0,
        "first_index": np.nan,
        "second_index": second_x,
        "first_price": np.nan,
        "second_price": second_y,
        "neckline_index": np.nan,
        "neckline": np.nan,
        "duplicate_tolerance_pct": 0.0,
        "neckline_score": 0.0,
        "reaction_score": 0.0,
        "between_cleanliness_score": 0.0,
    }
    body_ref = _threshold_body_pct(body_pct, row)
    atr_ref = _threshold_scale_pct(atr_pct, row, minimum=0.0005)
    prominence_ref = _threshold_scale_pct(prominence_pct, row)
    level_tolerance_pct = _dynamic_pattern_scale_pct(
        body_ref,
        atr_ref,
        prominence_ref,
        body_mult=level_body_mult,
        atr_mult=level_atr_mult,
        prominence_mult=level_prominence_mult,
    )
    move_threshold = _dynamic_pattern_scale_pct(
        body_ref,
        atr_ref,
        body_mult=premove_body_mult,
        atr_mult=premove_atr_mult,
    )
    depth_threshold = _dynamic_pattern_scale_pct(
        body_ref,
        atr_ref,
        body_mult=depth_body_mult,
        atr_mult=depth_atr_mult,
    )
    reaction_threshold = max(depth_threshold * 0.55, body_ref * float(reaction_body_mult))
    base_buffer_pct = body_ref * float(base_buffer_body_mult)
    duplicate_tolerance_pct = _dynamic_pattern_scale_pct(
        body_ref,
        atr_ref,
        body_mult=duplicate_tolerance_body_mult,
        atr_mult=duplicate_tolerance_atr_mult,
    )
    for candidate in candidates:
        first_x = float(low_index[candidate])
        span = second_x - first_x
        if span < min_bars or span > max_bars:
            continue
        first_y = float(low_pivot[candidate])
        trough_ref = max((first_y + second_y) / 2.0, 1e-9)
        similarity = 1.0 - abs(second_y - first_y) / max(trough_ref * level_tolerance_pct, 1e-9)
        if similarity <= 0.0:
            continue
        between_cleanliness_score = _double_bottom_between_cleanliness_score(
            low_pivot,
            low_index,
            row,
            first_x,
            second_x,
            min(first_y, second_y),
            trough_ref,
            level_tolerance_pct,
        )
        if between_cleanliness_score < float(min_between_cleanliness_score):
            continue
        prior = _prior_opposite_pivot_context(
            high_pivot,
            high_index,
            first_x,
            first_y,
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
            first_y,
            top=False,
            min_bars=prior_impulse_min_bars,
        )
        if impulse_score < float(prior_impulse_min_efficiency):
            continue
        dominance_score = _bottom_p1_dominance_score(low_pivot, low_index, row, prior["index"], first_x, first_y, level_tolerance_pct)
        dominance_score = min(dominance_score, impulse_score)
        if dominance_score <= 0.0:
            continue
        if not _bottom_base_is_held(body_high, first_x, second_x, prior["price"], base_buffer_pct):
            continue
        first_reaction_pct = _bottom_body_reaction_pct(body_high, first_x, second_x, trough_ref, reaction_max_bars)
        if first_reaction_pct < reaction_threshold:
            continue
        neckline_index, neckline = _reaction_level_between(body_high, first_x, second_x, find_low=False)
        if not np.isfinite([neckline_index, neckline]).all():
            continue
        depth_pct = (neckline - trough_ref) / max(abs(close[row]), 1e-9)
        turn_pct = max((float(close[row]) - second_y) / max(abs(float(close[row])), 1e-9), 0.0)
        reaction_score = _clip_value(first_reaction_pct / max(reaction_threshold, 1e-9))
        if reaction_score < float(min_reaction_score):
            continue
        quality = _peak_retest_quality(
            similarity,
            depth_pct,
            depth_threshold,
            first_move_pct,
            move_threshold,
            max(turn_pct, first_reaction_pct * 0.35),
            dominance_score,
            between_cleanliness_score,
            span,
            min_bars,
            max_bars,
        )
        if quality > best["quality"]:
            best = {
                "quality": quality,
                "first_index": first_x,
                "second_index": second_x,
                "first_price": first_y,
                "second_price": second_y,
                "neckline_index": neckline_index,
                "neckline": neckline,
                "duplicate_tolerance_pct": duplicate_tolerance_pct,
                "neckline_score": dominance_score,
                "reaction_score": reaction_score,
                "between_cleanliness_score": between_cleanliness_score,
            }
    return best


def _top_body_reaction_pct(
    body_low: np.ndarray,
    first_x: float,
    second_x: float,
    reference_price: float,
    reaction_max_bars: int,
) -> float:
    start = int(max(np.floor(first_x), 0))
    stop = int(min(np.ceil(min(float(second_x), float(first_x) + float(reaction_max_bars))), len(body_low) - 1))
    if stop <= start or not np.isfinite(reference_price) or reference_price == 0.0:
        return 0.0
    lows = np.asarray(body_low[start : stop + 1], dtype="float64")
    if not np.isfinite(lows).any():
        return 0.0
    return max((float(reference_price) - float(np.nanmin(lows))) / max(abs(float(reference_price)), 1e-9), 0.0)


def _bottom_body_reaction_pct(
    body_high: np.ndarray,
    first_x: float,
    second_x: float,
    reference_price: float,
    reaction_max_bars: int,
) -> float:
    start = int(max(np.floor(first_x), 0))
    stop = int(min(np.ceil(min(float(second_x), float(first_x) + float(reaction_max_bars))), len(body_high) - 1))
    if stop <= start or not np.isfinite(reference_price) or reference_price == 0.0:
        return 0.0
    highs = np.asarray(body_high[start : stop + 1], dtype="float64")
    if not np.isfinite(highs).any():
        return 0.0
    return max((float(np.nanmax(highs)) - float(reference_price)) / max(abs(float(reference_price)), 1e-9), 0.0)


def _double_top_between_cleanliness_score(
    high_pivot: np.ndarray,
    high_index: np.ndarray,
    row: int,
    first_x: float,
    second_x: float,
    allowed_peak: float,
    peak_ref: float,
    tolerance_pct: float,
) -> float:
    confirmed_rows = np.arange(len(high_pivot)) <= int(row)
    between = confirmed_rows & np.isfinite(high_pivot) & np.isfinite(high_index) & (high_index > first_x) & (high_index < second_x)
    if not between.any():
        return 1.0
    max_between = float(np.nanmax(high_pivot[between]))
    overshoot = max(max_between - float(allowed_peak), 0.0)
    tolerance = max(abs(float(peak_ref)) * float(tolerance_pct) * 0.50, 1e-9)
    return _clip_value(1.0 - overshoot / tolerance)


def _double_bottom_between_cleanliness_score(
    low_pivot: np.ndarray,
    low_index: np.ndarray,
    row: int,
    first_x: float,
    second_x: float,
    allowed_trough: float,
    trough_ref: float,
    tolerance_pct: float,
) -> float:
    confirmed_rows = np.arange(len(low_pivot)) <= int(row)
    between = confirmed_rows & np.isfinite(low_pivot) & np.isfinite(low_index) & (low_index > first_x) & (low_index < second_x)
    if not between.any():
        return 1.0
    min_between = float(np.nanmin(low_pivot[between]))
    undershoot = max(float(allowed_trough) - min_between, 0.0)
    tolerance = max(abs(float(trough_ref)) * float(tolerance_pct) * 0.50, 1e-9)
    return _clip_value(1.0 - undershoot / tolerance)


def _dedupe_double_events(
    mask: np.ndarray,
    first_index: np.ndarray,
    second_index: np.ndarray,
    neckline: np.ndarray,
    neckline_tolerance_pct: np.ndarray,
    cooldown_bars: int,
    overlap_pct: float,
) -> np.ndarray:
    """Suppress repeated double-pattern attention without looking ahead.

    The first confirmed setup is kept. Later setups are dropped when their
    anchor span overlaps a prior kept setup and their neckline describes the
    same level. This prevents one broad top/bottom from emitting several
    strategy-facing attention rows while preserving raw structure diagnostics.
    """

    clean = np.asarray(mask, dtype=bool)
    selected = np.zeros(len(clean), dtype=bool)
    selected_rows: list[int] = []
    cooldown = max(int(cooldown_bars), 1)
    min_overlap = float(overlap_pct)
    tolerance = np.asarray(neckline_tolerance_pct, dtype="float64")
    for row in np.flatnonzero(clean):
        if not np.isfinite([first_index[row], second_index[row], neckline[row]]).all():
            continue
        duplicate = False
        for prior in reversed(selected_rows):
            if row - prior <= cooldown:
                duplicate = True
                break
            if _same_double_structure(
                first_index[row],
                second_index[row],
                neckline[row],
                first_index[prior],
                second_index[prior],
                neckline[prior],
                min_overlap,
                max(float(tolerance[row]), float(tolerance[prior]), 1e-9),
            ):
                duplicate = True
                break
        if not duplicate:
            selected[row] = True
            selected_rows.append(int(row))
    return selected


def _same_double_structure(
    first_x: float,
    second_x: float,
    neckline: float,
    prior_first_x: float,
    prior_second_x: float,
    prior_neckline: float,
    min_overlap: float,
    neckline_tolerance_pct: float,
) -> bool:
    span = max(float(second_x) - float(first_x), 1.0)
    prior_span = max(float(prior_second_x) - float(prior_first_x), 1.0)
    overlap = min(float(second_x), float(prior_second_x)) - max(float(first_x), float(prior_first_x))
    overlap_ratio = overlap / max(min(span, prior_span), 1.0)
    neckline_ref = max(abs(float(neckline)), abs(float(prior_neckline)), 1e-9)
    neckline_close = abs(float(neckline) - float(prior_neckline)) / neckline_ref <= float(neckline_tolerance_pct)
    return overlap_ratio >= float(min_overlap) and neckline_close


def _head_shoulders_columns(frame: DataFrame, cfg: PatternReversalConfig) -> dict[str, Series]:
    """Detect head-and-shoulders / inverse head-and-shoulders at the right shoulder."""

    p = cfg.output_prefix
    arrays = _head_shoulders_arrays(frame, cfg)
    close = _num(frame["close"]).to_numpy(dtype="float64")
    hs = pd.Series(
        _dedupe_head_shoulders_events(
            arrays["head_shoulders_setup_short"],
            arrays["head_shoulders_left_index"],
            arrays["head_shoulders_head_index"],
            arrays["head_shoulders_right_index"],
            arrays["head_shoulders_neckline_left"],
            arrays["head_shoulders_neckline_right"],
            arrays["head_shoulders_duplicate_tolerance_pct"],
            int(cfg.entry_cooldown_bars),
            float(cfg.duplicate_overlap_ratio),
        ),
        index=frame.index,
        dtype="bool",
    )
    inverse = pd.Series(
        _dedupe_head_shoulders_events(
            arrays["inverse_head_shoulders_setup_long"],
            arrays["inverse_head_shoulders_left_index"],
            arrays["inverse_head_shoulders_head_index"],
            arrays["inverse_head_shoulders_right_index"],
            arrays["inverse_head_shoulders_neckline_left"],
            arrays["inverse_head_shoulders_neckline_right"],
            arrays["inverse_head_shoulders_duplicate_tolerance_pct"],
            int(cfg.entry_cooldown_bars),
            float(cfg.duplicate_overlap_ratio),
        ),
        index=frame.index,
        dtype="bool",
    )
    hs = _dedupe_setup_series(
        hs,
        int(cfg.entry_cooldown_bars),
    )
    inverse = _dedupe_setup_series(
        inverse,
        int(cfg.entry_cooldown_bars),
    )
    hs_state = _head_shoulders_state(
        close,
        hs,
        arrays["head_shoulders_left_index"],
        arrays["head_shoulders_head_index"],
        arrays["head_shoulders_right_index"],
        arrays["head_shoulders_head_price"],
        arrays["head_shoulders_neckline_left"],
        arrays["head_shoulders_neckline_right"],
        arrays["head_shoulders_quality"],
        int(cfg.pattern_lifecycle_mature_bars),
        int(cfg.pattern_lifecycle_stale_bars),
        inverse=False,
    )
    inverse_state = _head_shoulders_state(
        close,
        inverse,
        arrays["inverse_head_shoulders_left_index"],
        arrays["inverse_head_shoulders_head_index"],
        arrays["inverse_head_shoulders_right_index"],
        arrays["inverse_head_shoulders_head_price"],
        arrays["inverse_head_shoulders_neckline_left"],
        arrays["inverse_head_shoulders_neckline_right"],
        arrays["inverse_head_shoulders_quality"],
        int(cfg.pattern_lifecycle_mature_bars),
        int(cfg.pattern_lifecycle_stale_bars),
        inverse=True,
    )
    hs_state_series = pd.Series(hs_state["state"], index=frame.index, dtype="int8")
    inverse_state_series = pd.Series(inverse_state["state"], index=frame.index, dtype="int8")
    hs_quality = pd.Series(hs_state["quality"], index=frame.index, dtype="float64")
    inverse_quality = pd.Series(inverse_state["quality"], index=frame.index, dtype="float64")
    hs_present = hs_state_series.gt(0)
    inverse_present = inverse_state_series.gt(0)
    hs_confirmed = pd.Series(hs_state["confirmed"], index=frame.index, dtype="bool")
    inverse_confirmed = pd.Series(inverse_state["confirmed"], index=frame.index, dtype="bool")
    hs_confirmation_level = pd.Series(hs_state["neckline"], index=frame.index, dtype="float64")
    inverse_confirmation_level = pd.Series(inverse_state["neckline"], index=frame.index, dtype="float64")
    hs_head_price = pd.Series(hs_state["head_price"], index=frame.index, dtype="float64")
    inverse_head_price = pd.Series(inverse_state["head_price"], index=frame.index, dtype="float64")
    columns = {
        f"{p}_head_shoulders_pattern_present": hs_present.fillna(False),
        f"{p}_inverse_head_shoulders_pattern_present": inverse_present.fillna(False),
        f"{p}_head_shoulders_pattern_confirmed": hs_confirmed.fillna(False),
        f"{p}_inverse_head_shoulders_pattern_confirmed": inverse_confirmed.fillna(False),
        f"{p}_head_shoulders_indicator_score": hs_quality.where(hs_present, 0.0),
        f"{p}_inverse_head_shoulders_indicator_score": inverse_quality.where(inverse_present, 0.0),
        f"{p}_head_shoulders_confirmation_level": hs_confirmation_level,
        f"{p}_inverse_head_shoulders_confirmation_level": inverse_confirmation_level,
        f"{p}_head_shoulders_target_level": hs_confirmation_level + (hs_confirmation_level - hs_head_price),
        f"{p}_inverse_head_shoulders_target_level": inverse_confirmation_level + (inverse_confirmation_level - inverse_head_price),
        f"{p}_head_shoulders_p1_index": pd.Series(hs_state["p1_index"], index=frame.index, dtype="float64"),
        f"{p}_head_shoulders_p2_index": pd.Series(hs_state["p2_index"], index=frame.index, dtype="float64"),
        f"{p}_head_shoulders_p3_index": pd.Series(hs_state["p3_index"], index=frame.index, dtype="float64"),
        f"{p}_inverse_head_shoulders_p1_index": pd.Series(inverse_state["p1_index"], index=frame.index, dtype="float64"),
        f"{p}_inverse_head_shoulders_p2_index": pd.Series(inverse_state["p2_index"], index=frame.index, dtype="float64"),
        f"{p}_inverse_head_shoulders_p3_index": pd.Series(inverse_state["p3_index"], index=frame.index, dtype="float64"),
    }
    if bool(getattr(cfg, "include_pattern_diagnostics", False)):
        columns.update(
            {
                **_proof_line_columns(
                    frame.index,
                    f"{p}_head_shoulders_short",
                    hs,
                    {
                        1: (
                            arrays["head_shoulders_left_index"],
                            arrays["head_shoulders_left_price"],
                            arrays["head_shoulders_head_index"],
                            arrays["head_shoulders_head_price"],
                        ),
                        2: (
                            arrays["head_shoulders_head_index"],
                            arrays["head_shoulders_head_price"],
                            arrays["head_shoulders_right_index"],
                            arrays["head_shoulders_right_price"],
                        ),
                        3: (
                            arrays["head_shoulders_left_index"],
                            arrays["head_shoulders_neckline_left"],
                            arrays["head_shoulders_right_index"],
                            arrays["head_shoulders_neckline_right"],
                        ),
                    },
                ),
                **_proof_line_columns(
                    frame.index,
                    f"{p}_inverse_head_shoulders_long",
                    inverse,
                    {
                        1: (
                            arrays["inverse_head_shoulders_left_index"],
                            arrays["inverse_head_shoulders_left_price"],
                            arrays["inverse_head_shoulders_head_index"],
                            arrays["inverse_head_shoulders_head_price"],
                        ),
                        2: (
                            arrays["inverse_head_shoulders_head_index"],
                            arrays["inverse_head_shoulders_head_price"],
                            arrays["inverse_head_shoulders_right_index"],
                            arrays["inverse_head_shoulders_right_price"],
                        ),
                        3: (
                            arrays["inverse_head_shoulders_left_index"],
                            arrays["inverse_head_shoulders_neckline_left"],
                            arrays["inverse_head_shoulders_right_index"],
                            arrays["inverse_head_shoulders_neckline_right"],
                        ),
                    },
                ),
            }
        )
    return columns


def _head_shoulders_state(
    close: np.ndarray,
    setup_mask: np.ndarray | Series,
    left_index: np.ndarray,
    head_index: np.ndarray,
    right_index: np.ndarray,
    head_price: np.ndarray,
    neckline_left: np.ndarray,
    neckline_right: np.ndarray,
    quality: np.ndarray,
    mature_bars: int,
    stale_bars: int,
    *,
    inverse: bool,
) -> dict[str, np.ndarray]:
    rows = len(close)
    setup = np.asarray(setup_mask, dtype=bool)
    state = np.zeros(rows, dtype="int8")
    carried_quality = np.zeros(rows, dtype="float64")
    carried = {
        "p1_index": np.full(rows, np.nan, dtype="float64"),
        "p2_index": np.full(rows, np.nan, dtype="float64"),
        "p3_index": np.full(rows, np.nan, dtype="float64"),
        "head_price": np.full(rows, np.nan, dtype="float64"),
        "neckline": np.full(rows, np.nan, dtype="float64"),
    }
    mature = max(int(mature_bars), 1)
    stale = max(int(stale_bars), 0)
    max_age = mature + stale
    setup_row = -1
    confirmed_row = -1
    left_x = np.nan
    head_x = np.nan
    right_x = np.nan
    neck_left = np.nan
    neck_right = np.nan
    active_head_price = np.nan
    setup_quality = 0.0

    for row in range(rows):
        if setup[row] and np.isfinite(
            [left_index[row], head_index[row], right_index[row], head_price[row], neckline_left[row], neckline_right[row]]
        ).all():
            setup_row = int(row)
            confirmed_row = -1
            left_x = float(left_index[row])
            head_x = float(head_index[row])
            right_x = float(right_index[row])
            active_head_price = float(head_price[row])
            neck_left = float(neckline_left[row])
            neck_right = float(neckline_right[row])
            setup_quality = float(quality[row]) if np.isfinite(quality[row]) else 0.0

        if setup_row < 0:
            continue
        age = row - setup_row
        if confirmed_row < 0 and age > max_age:
            setup_row = -1
            confirmed_row = -1
            setup_quality = 0.0
            continue

        neckline_now = _line_value_at(float(row), left_x, neck_left, right_x, neck_right)
        if not np.isfinite(neckline_now):
            continue
        carried["p1_index"][row] = left_x
        carried["p2_index"][row] = head_x
        carried["p3_index"][row] = right_x
        carried["head_price"][row] = active_head_price
        carried["neckline"][row] = neckline_now
        carried_quality[row] = setup_quality

        close_now = float(close[row]) if np.isfinite(close[row]) else np.nan
        if confirmed_row < 0 and np.isfinite(close_now):
            confirmed = close_now >= neckline_now if inverse else close_now <= neckline_now
            if confirmed:
                confirmed_row = int(row)

        if confirmed_row >= 0:
            confirmed_age = row - confirmed_row
            if confirmed_age <= mature:
                state[row] = 2
            elif confirmed_age <= max_age:
                state[row] = 0
            else:
                setup_row = -1
                confirmed_row = -1
                setup_quality = 0.0
                carried_quality[row] = 0.0
                for values in carried.values():
                    values[row] = np.nan
        elif age <= mature:
            state[row] = 1
        else:
            state[row] = 0

    inactive = state <= 0
    carried_quality[inactive] = 0.0
    for values in carried.values():
        values[inactive] = np.nan

    return {
        **carried,
        "state": state,
        "confirmed": state == 2,
        "quality": carried_quality,
    }


def _head_shoulders_arrays(frame: DataFrame, cfg: PatternReversalConfig) -> dict[str, np.ndarray]:
    close, body_high, body_low, high_pivot, high_index, low_pivot, low_index = _pattern_geometry_arrays(frame, cfg)
    rows = len(frame)
    scale_window = int(getattr(cfg, "peak_dynamic_scale_window", cfg.peak_dynamic_body_window))
    body_pct = _rolling_body_pct(frame, scale_window)
    atr_pct = _rolling_atr_pct(frame, scale_window)
    prominence_pct = _rolling_pivot_prominence_pct(frame, cfg.pivot_prefix, scale_window)
    names = (
        "head_shoulders",
        "inverse_head_shoulders",
    )
    out: dict[str, np.ndarray] = {}
    for name in names:
        out[f"{name}_quality"] = np.zeros(rows, dtype="float64")
        out[f"{name}_left_index"] = np.full(rows, np.nan, dtype="float64")
        out[f"{name}_head_index"] = np.full(rows, np.nan, dtype="float64")
        out[f"{name}_right_index"] = np.full(rows, np.nan, dtype="float64")
        out[f"{name}_left_price"] = np.full(rows, np.nan, dtype="float64")
        out[f"{name}_head_price"] = np.full(rows, np.nan, dtype="float64")
        out[f"{name}_right_price"] = np.full(rows, np.nan, dtype="float64")
        out[f"{name}_neckline_left"] = np.full(rows, np.nan, dtype="float64")
        out[f"{name}_neckline_right"] = np.full(rows, np.nan, dtype="float64")
        out[f"{name}_duplicate_tolerance_pct"] = np.zeros(rows, dtype="float64")
        out[f"{name}_shoulder_score"] = np.zeros(rows, dtype="float64")
        out[f"{name}_head_score"] = np.zeros(rows, dtype="float64")
        out[f"{name}_time_balance_score"] = np.zeros(rows, dtype="float64")
        out[f"{name}_head_position_score"] = np.zeros(rows, dtype="float64")
        out[f"{name}_neckline_score"] = np.zeros(rows, dtype="float64")
        out[f"{name}_neckline_position_score"] = np.zeros(rows, dtype="float64")
        out[f"{name}_neckline_clearance_score"] = np.zeros(rows, dtype="float64")
        out[f"{name}_neckline_body_respect_ratio"] = np.zeros(rows, dtype="float64")
        out[f"{name}_reaction_score"] = np.zeros(rows, dtype="float64")
    out["head_shoulders_structure_short"] = np.zeros(rows, dtype=bool)
    out["inverse_head_shoulders_structure_long"] = np.zeros(rows, dtype=bool)
    out["head_shoulders_setup_short"] = np.zeros(rows, dtype=bool)
    out["inverse_head_shoulders_setup_long"] = np.zeros(rows, dtype=bool)

    for row in range(rows):
        if not np.isfinite(close[row]) or close[row] == 0.0:
            continue
        if np.isfinite(high_pivot[row]) and np.isfinite(high_index[row]):
            hs = _score_head_shoulders(
                row,
                close,
                body_high,
                body_low,
                high_pivot,
                high_index,
                low_pivot,
                low_index,
                body_pct,
                atr_pct,
                prominence_pct,
                cfg,
                inverse=False,
            )
            if hs["quality"] >= float(cfg.min_head_shoulders_quality):
                _store_head_shoulders(out, row, "head_shoulders", hs)
                out["head_shoulders_structure_short"][row] = True
        if np.isfinite(low_pivot[row]) and np.isfinite(low_index[row]):
            inv = _score_head_shoulders(
                row,
                close,
                body_high,
                body_low,
                high_pivot,
                high_index,
                low_pivot,
                low_index,
                body_pct,
                atr_pct,
                prominence_pct,
                cfg,
                inverse=True,
            )
            if inv["quality"] >= float(cfg.min_head_shoulders_quality):
                _store_head_shoulders(out, row, "inverse_head_shoulders", inv)
                out["inverse_head_shoulders_structure_long"][row] = True
    _activate_head_shoulders_setups(
        out,
        close,
        body_high,
        body_low,
        body_pct,
        atr_pct,
        cfg,
        name="head_shoulders",
        inverse=False,
    )
    _activate_head_shoulders_setups(
        out,
        close,
        body_high,
        body_low,
        body_pct,
        atr_pct,
        cfg,
        name="inverse_head_shoulders",
        inverse=True,
    )
    return out


def _score_head_shoulders(
    row: int,
    close: np.ndarray,
    body_high: np.ndarray,
    body_low: np.ndarray,
    high_pivot: np.ndarray,
    high_index: np.ndarray,
    low_pivot: np.ndarray,
    low_index: np.ndarray,
    body_pct: np.ndarray,
    atr_pct: np.ndarray,
    prominence_pct: np.ndarray,
    cfg: PatternReversalConfig,
    *,
    inverse: bool,
) -> dict[str, float]:
    pivot = low_pivot if inverse else high_pivot
    pivot_index = low_index if inverse else high_index
    neckline_pivot = high_pivot if inverse else low_pivot
    neckline_index = high_index if inverse else low_index
    right_x = float(pivot_index[row])
    right_y = float(pivot[row])
    pivot_mask = np.isfinite(pivot[:row]) & np.isfinite(pivot_index[:row])
    recent = np.flatnonzero(pivot_mask & (pivot_index[:row] < right_x) & (pivot_index[:row] >= right_x - int(cfg.head_shoulders_window)))
    best = _empty_head_shoulders_candidate(right_x, right_y)
    if len(recent) < 2:
        return best

    min_bars = int(cfg.min_head_shoulders_bars)
    max_bars = int(cfg.max_head_shoulders_bars)
    body_ref = _threshold_body_pct(body_pct, row)
    atr_ref = _threshold_scale_pct(atr_pct, row, minimum=0.0005)
    prominence_ref = _threshold_scale_pct(prominence_pct, row)
    shoulder_tolerance_pct = _dynamic_pattern_scale_pct(
        body_ref,
        atr_ref,
        prominence_ref,
        body_mult=cfg.peak_level_tolerance_body_mult,
        atr_mult=cfg.peak_level_tolerance_atr_mult,
        prominence_mult=cfg.peak_level_tolerance_prominence_mult,
    )
    prominence_threshold = _dynamic_pattern_scale_pct(
        body_ref,
        atr_ref,
        prominence_ref,
        body_mult=cfg.head_shoulders_prominence_body_mult,
        atr_mult=cfg.head_shoulders_prominence_atr_mult,
        prominence_mult=cfg.peak_level_tolerance_prominence_mult,
    )
    depth_threshold = _dynamic_pattern_scale_pct(
        body_ref,
        atr_ref,
        body_mult=cfg.head_shoulders_neckline_depth_body_mult,
        atr_mult=cfg.head_shoulders_neckline_depth_atr_mult,
    )
    prior_move_threshold = _dynamic_pattern_scale_pct(
        body_ref,
        atr_ref,
        body_mult=cfg.head_shoulders_premove_body_mult,
        atr_mult=cfg.head_shoulders_premove_atr_mult,
    )
    base_buffer_pct = body_ref * float(cfg.head_shoulders_base_return_buffer_body_mult)
    duplicate_tolerance_pct = _dynamic_pattern_scale_pct(
        body_ref,
        atr_ref,
        body_mult=cfg.duplicate_neckline_tolerance_body_mult,
        atr_mult=cfg.duplicate_neckline_tolerance_atr_mult,
    )
    neckline_proximity_pct = _dynamic_pattern_scale_pct(
        body_ref,
        atr_ref,
        body_mult=cfg.head_shoulders_neckline_proximity_body_mult,
        atr_mult=cfg.head_shoulders_neckline_proximity_atr_mult,
    )
    max_candidates = max(int(getattr(cfg, "head_shoulders_max_candidate_pivots", 18)), 3)
    candidate_start = max(0, len(recent) - max_candidates)
    for left_pos in range(candidate_start, len(recent) - 1):
        left = recent[left_pos]
        for head_pos in range(left_pos + 1, len(recent)):
            head = recent[head_pos]
            left_x = float(pivot_index[left])
            head_x = float(pivot_index[head])
            span = right_x - left_x
            if span < min_bars or span > max_bars or head_x <= left_x or head_x >= right_x:
                continue
            left_time = head_x - left_x
            right_time = right_x - head_x
            time_balance_score = min(left_time, right_time) / max(left_time, right_time, 1e-9)
            if time_balance_score < float(cfg.min_head_shoulders_time_balance_score):
                continue
            head_position = (head_x - left_x) / max(span, 1e-9)
            head_position_score = _clip_value(1.0 - abs(head_position - 0.5) / 0.5)
            if head_position_score < float(cfg.min_head_shoulders_head_position_score):
                continue
            left_y = float(pivot[left])
            head_y = float(pivot[head])
            shoulder_ref = max((abs(left_y) + abs(right_y)) / 2.0, 1e-9)
            shoulder_similarity = 1.0 - abs(right_y - left_y) / max(shoulder_ref * shoulder_tolerance_pct, 1e-9)
            if shoulder_similarity < float(cfg.min_head_shoulders_shoulder_score):
                continue
            if inverse:
                pattern_prominence_pct = (min(left_y, right_y) - head_y) / max(abs(close[row]), 1e-9)
                prior_direction_required = -1
            else:
                pattern_prominence_pct = (head_y - max(left_y, right_y)) / max(abs(close[row]), 1e-9)
                prior_direction_required = 1
            if pattern_prominence_pct < prominence_threshold:
                continue
            if not _head_shoulders_p1_prior_atr_move(
                body_high,
                body_low,
                left_x,
                left_y,
                atr_ref,
                int(cfg.head_shoulders_p1_prior_window),
                float(cfg.head_shoulders_p1_prior_atr_mult),
                inverse=inverse,
            ):
                continue
            peak_cleanliness_score = _head_shoulders_peak_cleanliness_score(
                pivot,
                pivot_index,
                row,
                left_x,
                head_x,
                right_x,
                left_y,
                head_y,
                right_y,
                shoulder_tolerance_pct,
                inverse=inverse,
            )
            if peak_cleanliness_score < float(cfg.min_head_shoulders_peak_cleanliness_score):
                continue
            confirmed_rows = np.arange(len(neckline_pivot)) <= row
            left_neck_mask = (
                confirmed_rows
                & np.isfinite(neckline_pivot)
                & np.isfinite(neckline_index)
                & (neckline_index > left_x)
                & (neckline_index < head_x)
            )
            right_neck_mask = (
                confirmed_rows
                & np.isfinite(neckline_pivot)
                & np.isfinite(neckline_index)
                & (neckline_index > head_x)
                & (neckline_index < right_x)
            )
            if not left_neck_mask.any() or not right_neck_mask.any():
                continue
            left_neck_rows = np.flatnonzero(left_neck_mask)
            right_neck_rows = np.flatnonzero(right_neck_mask)
            left_neck_row = int(left_neck_rows[-1])
            right_neck_row = int(right_neck_rows[-1])
            neckline_left = float(neckline_pivot[left_neck_row])
            neckline_right = float(neckline_pivot[right_neck_row])
            if inverse:
                depth_pct = (min(neckline_left, neckline_right) - head_y) / max(abs(close[row]), 1e-9)
            else:
                depth_pct = (head_y - max(neckline_left, neckline_right)) / max(abs(close[row]), 1e-9)
            if depth_pct < depth_threshold:
                continue
            left_neck_x = float(neckline_index[left_neck_row])
            right_neck_x = float(neckline_index[right_neck_row])
            neckline_position_score = _head_shoulders_neckline_position_score(
                left_x,
                head_x,
                right_x,
                left_neck_x,
                right_neck_x,
            )
            if neckline_position_score < float(cfg.min_head_shoulders_neckline_position_score):
                continue
            neckline_score = _head_shoulders_neckline_score(
                neckline_left,
                neckline_right,
                left_x,
                right_x,
                float(close[row]),
                atr_ref,
                float(cfg.head_shoulders_max_neckline_slope_atr_per_bar),
            )
            if neckline_score < float(cfg.min_head_shoulders_neckline_score):
                continue
            neckline_clearance_score = _head_shoulders_neckline_clearance_score(
                left_y,
                head_y,
                right_y,
                neckline_left,
                neckline_right,
                left_x,
                head_x,
                right_x,
                float(close[row]),
                depth_threshold,
                inverse=inverse,
            )
            if neckline_clearance_score < float(cfg.min_head_shoulders_neckline_clearance_score):
                continue
            neckline_body_respect_ratio = _head_shoulders_neckline_body_respect_ratio(
                body_high,
                body_low,
                left_x,
                right_x,
                neckline_left,
                neckline_right,
                float(close[row]),
                neckline_proximity_pct,
                inverse=inverse,
            )
            if neckline_body_respect_ratio < float(cfg.min_head_shoulders_neckline_body_respect_ratio):
                continue
            prior_direction, close_prior_move_pct = _prior_pattern_move(close, left_x, span)
            if prior_direction != prior_direction_required:
                continue
            prior = _prior_opposite_pivot_context(
                neckline_pivot,
                neckline_index,
                left_x,
                left_y,
                int(cfg.head_shoulders_window),
                top=not inverse,
            )
            prior_move_pct = max(float(prior["move_pct"]), float(close_prior_move_pct))
            if prior_move_pct < prior_move_threshold:
                continue
            impulse_score = _prior_impulse_score(
                close,
                prior["index"],
                prior["price"],
                left_x,
                left_y,
                top=not inverse,
                min_bars=int(cfg.peak_prior_impulse_min_bars),
            )
            if impulse_score < float(cfg.peak_prior_impulse_min_efficiency):
                continue
            if inverse:
                p1_dominance_score = _bottom_p1_dominance_score(
                    low_pivot,
                    low_index,
                    row,
                    prior["index"],
                    left_x,
                    left_y,
                    shoulder_tolerance_pct,
                )
            else:
                p1_dominance_score = _top_p1_dominance_score(
                    high_pivot,
                    high_index,
                    row,
                    prior["index"],
                    left_x,
                    left_y,
                    shoulder_tolerance_pct,
                )
            p1_dominance_score = min(p1_dominance_score, impulse_score)
            if p1_dominance_score < float(cfg.min_head_shoulders_p1_dominance_score):
                continue
            if inverse:
                base_held = _bottom_base_is_held(body_high, left_x, right_x, prior["price"], base_buffer_pct)
            else:
                base_held = _top_base_is_held(body_low, left_x, right_x, prior["price"], base_buffer_pct)
            if not base_held:
                continue
            quality = _head_shoulders_quality(
                shoulder_similarity,
                pattern_prominence_pct,
                prominence_threshold,
                depth_pct,
                depth_threshold,
                prior_move_pct,
                prior_move_threshold,
                impulse_score,
                peak_cleanliness_score,
                neckline_score,
                time_balance_score,
                head_position_score,
                neckline_position_score,
                neckline_clearance_score,
                neckline_body_respect_ratio,
                span,
                min_bars,
                max_bars,
            )
            if quality > best["quality"]:
                best = {
                    "quality": quality,
                    "left_index": left_x,
                    "head_index": head_x,
                    "right_index": right_x,
                    "left_price": left_y,
                    "head_price": head_y,
                    "right_price": right_y,
                    "neckline_left": neckline_left,
                    "neckline_right": neckline_right,
                    "duplicate_tolerance_pct": duplicate_tolerance_pct,
                    "shoulder_score": shoulder_similarity,
                    "head_score": _clip_value(pattern_prominence_pct / max(prominence_threshold * 2.0, 1e-9)),
                    "time_balance_score": time_balance_score,
                    "head_position_score": head_position_score,
                    "neckline_score": neckline_score,
                    "neckline_position_score": neckline_position_score,
                    "neckline_clearance_score": neckline_clearance_score,
                    "neckline_body_respect_ratio": neckline_body_respect_ratio,
                    "reaction_score": 0.0,
                }
    return best


def _dedupe_setup_series(mask: Series, cooldown_bars: int) -> Series:
    clean = mask.fillna(False).astype("bool")
    cooldown = max(int(cooldown_bars), 1)
    if cooldown <= 1:
        return clean
    recent = clean.shift(1, fill_value=False).rolling(cooldown, min_periods=1).max().astype("bool")
    return clean & ~recent


def _dedupe_head_shoulders_events(
    mask: np.ndarray,
    left_index: np.ndarray,
    head_index: np.ndarray,
    right_index: np.ndarray,
    neckline_left: np.ndarray,
    neckline_right: np.ndarray,
    neckline_tolerance_pct: np.ndarray,
    cooldown_bars: int,
    overlap_pct: float,
) -> np.ndarray:
    clean = np.asarray(mask, dtype=bool)
    selected = np.zeros(len(clean), dtype=bool)
    selected_rows: list[int] = []
    cooldown = max(int(cooldown_bars), 1)
    tolerance = np.asarray(neckline_tolerance_pct, dtype="float64")
    for row in np.flatnonzero(clean):
        if not np.isfinite([left_index[row], head_index[row], right_index[row], neckline_left[row], neckline_right[row]]).all():
            continue
        duplicate = False
        for prior in reversed(selected_rows):
            if row - prior <= cooldown:
                duplicate = True
                break
            if _same_head_shoulders_structure(
                left_index[row],
                head_index[row],
                right_index[row],
                neckline_left[row],
                neckline_right[row],
                left_index[prior],
                head_index[prior],
                right_index[prior],
                neckline_left[prior],
                neckline_right[prior],
                overlap_pct,
                max(float(tolerance[row]), float(tolerance[prior]), 1e-9),
            ):
                duplicate = True
                break
        if not duplicate:
            selected[row] = True
            selected_rows.append(int(row))
    return selected


def _same_head_shoulders_structure(
    left_a: float,
    head_a: float,
    right_a: float,
    neckline_left_a: float,
    neckline_right_a: float,
    left_b: float,
    head_b: float,
    right_b: float,
    neckline_left_b: float,
    neckline_right_b: float,
    overlap_pct: float,
    neckline_tolerance_pct: float,
) -> bool:
    span_a = max(float(right_a) - float(left_a), 1e-9)
    span_b = max(float(right_b) - float(left_b), 1e-9)
    overlap = max(0.0, min(float(right_a), float(right_b)) - max(float(left_a), float(left_b)))
    if overlap / max(min(span_a, span_b), 1e-9) < float(overlap_pct):
        return False
    head_close = abs(float(head_a) - float(head_b)) <= max(span_a, span_b) * 0.20
    neckline_a = (float(neckline_left_a) + float(neckline_right_a)) / 2.0
    neckline_b = (float(neckline_left_b) + float(neckline_right_b)) / 2.0
    neckline_ref = max(abs(neckline_a), abs(neckline_b), 1e-9)
    neckline_close = abs(neckline_a - neckline_b) / neckline_ref <= float(neckline_tolerance_pct)
    return bool(head_close and neckline_close)


def _empty_head_shoulders_candidate(right_x: float, right_y: float) -> dict[str, float]:
    return {
        "quality": 0.0,
        "left_index": np.nan,
        "head_index": np.nan,
        "right_index": right_x,
        "left_price": np.nan,
        "head_price": np.nan,
        "right_price": right_y,
        "neckline_left": np.nan,
        "neckline_right": np.nan,
        "duplicate_tolerance_pct": 0.0,
        "shoulder_score": 0.0,
        "head_score": 0.0,
        "time_balance_score": 0.0,
        "head_position_score": 0.0,
        "neckline_score": 0.0,
        "neckline_position_score": 0.0,
        "neckline_clearance_score": 0.0,
        "neckline_body_respect_ratio": 0.0,
        "reaction_score": 0.0,
    }


def _store_head_shoulders(out: dict[str, np.ndarray], row: int, name: str, candidate: dict[str, float]) -> None:
    for field in (
        "quality",
        "left_index",
        "head_index",
        "right_index",
        "left_price",
        "head_price",
        "right_price",
        "neckline_left",
        "neckline_right",
        "duplicate_tolerance_pct",
        "shoulder_score",
        "head_score",
        "time_balance_score",
        "head_position_score",
        "neckline_score",
        "neckline_position_score",
        "neckline_clearance_score",
        "neckline_body_respect_ratio",
        "reaction_score",
    ):
        out[f"{name}_{field}"][row] = float(candidate[field])


def _copy_head_shoulders_candidate(out: dict[str, np.ndarray], source_row: int, target_row: int, name: str) -> None:
    for field in (
        "quality",
        "left_index",
        "head_index",
        "right_index",
        "left_price",
        "head_price",
        "right_price",
        "neckline_left",
        "neckline_right",
        "duplicate_tolerance_pct",
        "shoulder_score",
        "head_score",
        "time_balance_score",
        "head_position_score",
        "neckline_score",
        "neckline_position_score",
        "neckline_clearance_score",
        "neckline_body_respect_ratio",
        "reaction_score",
    ):
        out[f"{name}_{field}"][target_row] = float(out[f"{name}_{field}"][source_row])


def _activate_head_shoulders_setups(
    out: dict[str, np.ndarray],
    close: np.ndarray,
    body_high: np.ndarray,
    body_low: np.ndarray,
    body_pct: np.ndarray,
    atr_pct: np.ndarray,
    cfg: PatternReversalConfig,
    *,
    name: str,
    inverse: bool,
) -> None:
    """Emit H&S setup when the right shoulder is confirmed.

    Pattern detection should say "pay attention, this may be H&S", not wait
    for a neckline break. The confirmed right-shoulder row is the earliest
    no-lookahead point where the full shape exists. The old neckline-only setup
    path was too late and missed usable forming patterns; it remains as a
    fallback if the right-shoulder confirmation row has not yet reacted enough.
    """

    rows = len(close)
    structure_key = f"{name}_structure_long" if inverse else f"{name}_structure_short"
    setup_key = f"{name}_setup_long" if inverse else f"{name}_setup_short"
    monitor_bars = max(int(cfg.head_shoulders_setup_monitor_bars), 1)
    structure_rows = np.flatnonzero(out[structure_key])
    for structure_row in structure_rows:
        left_x = float(out[f"{name}_left_index"][structure_row])
        right_x = float(out[f"{name}_right_index"][structure_row])
        neckline_left = float(out[f"{name}_neckline_left"][structure_row])
        neckline_right = float(out[f"{name}_neckline_right"][structure_row])
        right_price = float(out[f"{name}_right_price"][structure_row])
        if not np.isfinite([left_x, right_x, neckline_left, neckline_right, right_price]).all():
            continue
        start = int(structure_row)
        stop = min(rows - 1, start + monitor_bars)
        if np.isfinite(close[start]) and close[start] != 0.0:
            proximity_pct = _dynamic_pattern_scale_pct(
                _threshold_body_pct(body_pct, start),
                _threshold_scale_pct(atr_pct, start, minimum=0.0005),
                body_mult=cfg.head_shoulders_neckline_proximity_body_mult,
                atr_mult=cfg.head_shoulders_neckline_proximity_atr_mult,
            )
            tolerance = abs(float(close[start])) * proximity_pct
            if inverse:
                reaction_score = _clip_value((float(close[start]) - right_price) / max(tolerance, 1e-9))
                if reaction_score >= float(cfg.min_head_shoulders_right_reaction_score):
                    out[setup_key][start] = True
                    out[f"{name}_reaction_score"][start] = reaction_score
                    continue
            else:
                reaction_score = _clip_value((right_price - float(close[start])) / max(tolerance, 1e-9))
                if reaction_score >= float(cfg.min_head_shoulders_right_reaction_score):
                    out[setup_key][start] = True
                    out[f"{name}_reaction_score"][start] = reaction_score
                    continue
        for row in range(start, stop + 1):
            neckline_now = _line_value_at(float(row), left_x, neckline_left, right_x, neckline_right)
            if not np.isfinite(neckline_now) or not np.isfinite(close[row]) or close[row] == 0.0:
                continue
            proximity_pct = _dynamic_pattern_scale_pct(
                _threshold_body_pct(body_pct, row),
                _threshold_scale_pct(atr_pct, row, minimum=0.0005),
                body_mult=cfg.head_shoulders_neckline_proximity_body_mult,
                atr_mult=cfg.head_shoulders_neckline_proximity_atr_mult,
            )
            tolerance = abs(float(close[row])) * proximity_pct
            if inverse:
                approaching_neckline = float(body_high[row]) >= neckline_now - tolerance
                shoulder_bounce = float(close[row]) >= right_price
                reaction_score = _clip_value((float(close[row]) - right_price) / max(tolerance, 1e-9))
                if approaching_neckline and shoulder_bounce and reaction_score >= float(cfg.min_head_shoulders_right_reaction_score):
                    out[setup_key][row] = True
                    _copy_head_shoulders_candidate(out, int(structure_row), row, name)
                    out[f"{name}_reaction_score"][row] = reaction_score
                    break
            else:
                approaching_neckline = float(body_low[row]) <= neckline_now + tolerance
                shoulder_reject = float(close[row]) <= right_price
                reaction_score = _clip_value((right_price - float(close[row])) / max(tolerance, 1e-9))
                if approaching_neckline and shoulder_reject and reaction_score >= float(cfg.min_head_shoulders_right_reaction_score):
                    out[setup_key][row] = True
                    _copy_head_shoulders_candidate(out, int(structure_row), row, name)
                    out[f"{name}_reaction_score"][row] = reaction_score
                    break


def _head_shoulders_quality(
    shoulder_similarity: float,
    prominence_pct: float,
    min_prominence: float,
    depth_pct: float,
    min_depth: float,
    prior_move_pct: float,
    min_prior_move: float,
    impulse_score: float,
    peak_cleanliness_score: float,
    neckline_score: float,
    time_balance_score: float,
    head_position_score: float,
    neckline_position_score: float,
    neckline_clearance_score: float,
    neckline_body_respect_ratio: float,
    span: float,
    min_bars: int,
    max_bars: int,
) -> float:
    span_mid = (float(min_bars) + float(max_bars)) / 2.0
    span_score = _clip_value(1.0 - abs(float(span) - span_mid) / max(span_mid, 1.0))
    prominence_score = _clip_value(prominence_pct / max(min_prominence * 2.0, 1e-9))
    depth_score = _clip_value(depth_pct / max(min_depth * 2.5, 1e-9))
    prior_score = _clip_value(prior_move_pct / max(min_prior_move * 2.0, 1e-9))
    return _clip_value(
        0.21 * shoulder_similarity
        + 0.21 * prominence_score
        + 0.16 * depth_score
        + 0.11 * prior_score
        + 0.07 * float(impulse_score)
        + 0.06 * float(peak_cleanliness_score)
        + 0.06 * span_score
        + 0.04 * time_balance_score
        + 0.02 * head_position_score
        + 0.07 * neckline_score
        + 0.03 * neckline_position_score
        + 0.03 * neckline_clearance_score
        + 0.03 * neckline_body_respect_ratio
    )


def _head_shoulders_p1_prior_atr_move(
    body_high: np.ndarray,
    body_low: np.ndarray,
    p1_x: float,
    p1_price: float,
    atr_ref_pct: float,
    lookback_bars: int,
    atr_mult: float,
    *,
    inverse: bool,
) -> bool:
    """Require a real directional move into the left shoulder before H&S scoring."""

    if not np.isfinite([p1_x, p1_price, atr_ref_pct]).all() or p1_price == 0.0:
        return False
    lookback = max(int(lookback_bars), 1)
    anchor = int(max(np.floor(float(p1_x)), 0))
    start = max(0, anchor - lookback)
    stop = min(anchor, len(body_high) - 1)
    if stop <= start:
        return False
    threshold = abs(float(p1_price)) * max(float(atr_ref_pct), 0.0) * max(float(atr_mult), 0.0)
    if threshold <= 0.0:
        return True
    if inverse:
        prior_high = np.asarray(body_high[start:stop], dtype="float64")
        if not np.isfinite(prior_high).any():
            return False
        return float(np.nanmax(prior_high)) - float(p1_price) >= threshold
    prior_low = np.asarray(body_low[start:stop], dtype="float64")
    if not np.isfinite(prior_low).any():
        return False
    return float(p1_price) - float(np.nanmin(prior_low)) >= threshold


def _head_shoulders_peak_cleanliness_score(
    pivot: np.ndarray,
    pivot_index: np.ndarray,
    row: int,
    left_x: float,
    head_x: float,
    right_x: float,
    left_y: float,
    head_y: float,
    right_y: float,
    tolerance_pct: float,
    *,
    inverse: bool,
) -> float:
    confirmed_rows = np.arange(len(pivot)) <= int(row)
    between = (
        confirmed_rows
        & np.isfinite(pivot)
        & np.isfinite(pivot_index)
        & (pivot_index >= float(left_x))
        & (pivot_index <= float(right_x))
    )
    if not between.any():
        return 0.0

    anchor_tolerance = 0.50
    anchor_mask = (
        np.abs(pivot_index - float(left_x)) <= anchor_tolerance
    ) | (
        np.abs(pivot_index - float(head_x)) <= anchor_tolerance
    ) | (
        np.abs(pivot_index - float(right_x)) <= anchor_tolerance
    )
    extra = between & ~anchor_mask
    reference = max(abs(float(head_y)), abs(float(left_y)), abs(float(right_y)), 1e-9)
    tolerance = reference * max(float(tolerance_pct), 1e-9)

    if inverse:
        head_breach = max(float(head_y) - float(np.nanmin(pivot[between])), 0.0)
        shoulder_level = min(float(left_y), float(right_y))
        extra_major = extra & (pivot < shoulder_level - tolerance)
    else:
        head_breach = max(float(np.nanmax(pivot[between])) - float(head_y), 0.0)
        shoulder_level = max(float(left_y), float(right_y))
        extra_major = extra & (pivot > shoulder_level + tolerance)

    dominance_score = _clip_value(1.0 - head_breach / max(tolerance, 1e-9))
    extra_count = int(np.count_nonzero(extra_major))
    noise_score = _clip_value(1.0 - extra_count / 2.0)
    return min(dominance_score, noise_score)


def _head_shoulders_neckline_position_score(
    left_x: float,
    head_x: float,
    right_x: float,
    left_neck_x: float,
    right_neck_x: float,
) -> float:
    left_span = max(float(head_x) - float(left_x), 1e-9)
    right_span = max(float(right_x) - float(head_x), 1e-9)
    left_pos = (float(left_neck_x) - float(left_x)) / left_span
    right_pos = (float(right_neck_x) - float(head_x)) / right_span
    return min(_interval_edge_clearance_score(left_pos), _interval_edge_clearance_score(right_pos))


def _interval_edge_clearance_score(position: float) -> float:
    if position <= 0.0 or position >= 1.0:
        return 0.0
    edge_clearance = min(float(position), 1.0 - float(position))
    return _clip_value(edge_clearance / 0.25)


def _head_shoulders_neckline_body_respect_ratio(
    body_high: np.ndarray,
    body_low: np.ndarray,
    left_x: float,
    right_x: float,
    neckline_left: float,
    neckline_right: float,
    reference_price: float,
    tolerance_pct: float,
    *,
    inverse: bool,
) -> float:
    start = max(int(np.floor(left_x)), 0)
    stop = min(int(np.ceil(right_x)), len(body_high) - 1)
    if stop <= start:
        return 0.0
    rows = np.arange(start, stop + 1)
    neckline = np.array(
        [_line_value_at(float(row), float(left_x), float(neckline_left), float(right_x), float(neckline_right)) for row in rows],
        dtype="float64",
    )
    valid = np.isfinite(neckline)
    if not valid.any():
        return 0.0
    tolerance = max(abs(float(reference_price)) * float(tolerance_pct), 1e-9)
    if inverse:
        respected = np.asarray(body_high[rows], dtype="float64") <= neckline + tolerance
    else:
        respected = np.asarray(body_low[rows], dtype="float64") >= neckline - tolerance
    return float(np.mean(respected[valid]))


def _head_shoulders_neckline_clearance_score(
    left_y: float,
    head_y: float,
    right_y: float,
    neckline_left: float,
    neckline_right: float,
    left_x: float,
    head_x: float,
    right_x: float,
    reference_price: float,
    min_depth: float,
    *,
    inverse: bool,
) -> float:
    neckline_at_head = _line_value_at(float(head_x), float(left_x), float(neckline_left), float(right_x), float(neckline_right))
    if not np.isfinite(neckline_at_head):
        return 0.0
    reference = max(abs(float(reference_price)), 1e-9)
    if inverse:
        clearances = (
            float(neckline_left) - float(left_y),
            float(neckline_at_head) - float(head_y),
            float(neckline_right) - float(right_y),
        )
    else:
        clearances = (
            float(left_y) - float(neckline_left),
            float(head_y) - float(neckline_at_head),
            float(right_y) - float(neckline_right),
        )
    min_clearance_pct = min(clearances) / reference
    return _clip_value(min_clearance_pct / max(float(min_depth) * 0.75, 1e-9))


def _head_shoulders_neckline_score(
    neckline_left: float,
    neckline_right: float,
    left_x: float,
    right_x: float,
    reference_price: float,
    atr_ref_pct: float,
    max_slope_atr_per_bar: float,
) -> float:
    span = max(float(right_x) - float(left_x), 1.0)
    slope_pct = abs(float(neckline_right) - float(neckline_left)) / max(abs(float(reference_price)), 1e-9) / span
    max_slope_pct_per_bar = max(float(atr_ref_pct) * float(max_slope_atr_per_bar), 1e-9)
    return _clip_value(1.0 - slope_pct / max_slope_pct_per_bar)

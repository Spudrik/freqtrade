from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from pattern_common import (
    _clip_value,
    _combine_sparse_indexes,
    _combine_sparse_pivots,
    _confirmed_micro_pivots,
    _continuation_proof_columns,
    _line_segment_from_points,
    _line_value_at,
    _nanmax_pair,
    _num,
    _slope_from_points,
    _with_foundation_pivots,
)


@dataclass(frozen=True)
class PatternContinuationConfig:
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
    entry_cooldown_bars: int = 8
    impulse_window: int = 36
    min_impulse_bars: int = 3
    min_impulse_pct: float = 0.055
    min_impulse_efficiency: float = 0.34
    min_impulse_volume_ratio: float = 1.05
    impulse_extreme_tolerance_pct: float = 0.004
    impulse_dominance_mult: float = 1.08
    min_impulse_dominance_score: float = 0.55
    min_impulse_break_score: float = 0.15
    max_consolidation_drift_pct_per_bar: float = 0.00075
    max_impulse_extreme_age_bars: int = 32
    pattern_pivot_strength: int = 1
    min_pattern_bars: int = 4
    min_setup_retrace_pct: float = 0.04
    max_setup_retrace_pct: float = 0.65
    max_setup_range_pct: float = 0.30
    max_setup_to_impulse_range_mult: float = 1.05
    max_pattern_breakout_pct: float = 0.018
    max_pattern_boundary_excursion_pct: float = 0.006
    pattern_side_dominance_mult: float = 1.03
    min_pattern_side_pivots: int = 2
    min_boundary_slope_pct_per_bar: float = 0.00018
    max_flag_counter_slope_pct_per_bar: float = 0.0008
    parallel_slope_tolerance_pct_per_bar: float = 0.0008
    min_continuation_pole_score: float = 0.50
    min_continuation_retrace_score: float = 0.18
    min_continuation_containment_score: float = 0.62
    min_continuation_terminal_score: float = 0.12
    min_continuation_boundary_touch_score: float = 0.58
    min_continuation_boundary_span_score: float = 0.35
    min_flag_quality: float = 0.72
    min_pennant_quality: float = 0.72
    include_pattern_diagnostics: bool = False


def add_pattern_continuation(
    dataframe: DataFrame,
    timeframe: str = "4h",
    config: PatternContinuationConfig | None = None,
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
    columns = _flag_pennant_columns(frame, {}, {}, cfg)
    existing = [
        column
        for column in dataframe.columns
        if str(column).startswith(
            (
                f"{cfg.output_prefix}_impulse_",
                f"{cfg.output_prefix}_pattern_",
                f"{cfg.output_prefix}_continuation_",
                f"{cfg.output_prefix}_setup_",
                f"{cfg.output_prefix}_flag_",
                f"{cfg.output_prefix}_pennant_",
            )
        )
    ]
    clean = dataframe.drop(columns=existing).copy() if existing else dataframe.copy()
    return pd.concat([clean, pd.DataFrame(columns, index=frame.index)], axis=1)


def _resolve_config(
    config: PatternContinuationConfig | None,
    timeframe: str,
    overrides: dict[str, object],
) -> PatternContinuationConfig:
    cfg = config or PatternContinuationConfig(timeframe=str(timeframe))
    valid = {field.name for field in fields(PatternContinuationConfig)}
    clean = {key: value for key, value in overrides.items() if value is not None}
    unknown = sorted(set(clean).difference(valid))
    if unknown:
        raise TypeError(f"Unknown continuation config override(s): {', '.join(unknown)}")
    if "timeframe" not in clean:
        clean["timeframe"] = str(timeframe)
    return replace(cfg, **clean) if clean else cfg


def _validate_ohlcv(dataframe: DataFrame) -> None:
    missing = [column for column in ("open", "high", "low", "close", "volume") if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Dataframe missing required columns: {', '.join(missing)}")
    if dataframe.empty:
        raise ValueError("Dataframe must not be empty")


def _flag_pennant_columns(
    frame: DataFrame,
    sequence: dict[str, Series],
    channel: dict[str, Series],
    cfg: PatternContinuationConfig,
) -> dict[str, Series]:
    p = cfg.output_prefix
    _ = sequence, channel
    anchored = _anchored_flag_pennant_arrays(frame, cfg)
    impulse_up_pct = pd.Series(anchored["impulse_up_pct"], index=frame.index, dtype="float64")
    impulse_down_pct = pd.Series(anchored["impulse_down_pct"], index=frame.index, dtype="float64")
    setup_range_pct = pd.Series(anchored["setup_range_pct"], index=frame.index, dtype="float64")
    setup_to_impulse = pd.Series(anchored["setup_to_impulse"], index=frame.index, dtype="float64")
    long_retrace = pd.Series(anchored["setup_retrace_long"], index=frame.index, dtype="float64")
    short_retrace = pd.Series(anchored["setup_retrace_short"], index=frame.index, dtype="float64")
    flag_quality_long = pd.Series(anchored["flag_quality_long"], index=frame.index, dtype="float64")
    flag_quality_short = pd.Series(anchored["flag_quality_short"], index=frame.index, dtype="float64")
    pennant_quality_long = pd.Series(anchored["pennant_quality_long"], index=frame.index, dtype="float64")
    pennant_quality_short = pd.Series(anchored["pennant_quality_short"], index=frame.index, dtype="float64")
    flag_setup_long = _dedupe_continuation_events(
        pd.Series(anchored["flag_setup_long"], index=frame.index, dtype="bool"),
        int(cfg.entry_cooldown_bars),
    )
    flag_setup_short = _dedupe_continuation_events(
        pd.Series(anchored["flag_setup_short"], index=frame.index, dtype="bool"),
        int(cfg.entry_cooldown_bars),
    )
    pennant_setup_long = _dedupe_continuation_events(
        pd.Series(anchored["pennant_setup_long"], index=frame.index, dtype="bool"),
        int(cfg.entry_cooldown_bars),
    )
    pennant_setup_short = _dedupe_continuation_events(
        pd.Series(anchored["pennant_setup_short"], index=frame.index, dtype="bool"),
        int(cfg.entry_cooldown_bars),
    )
    flag_setup_long, pennant_setup_long = _resolve_continuation_name_conflicts(
        flag_setup_long,
        pennant_setup_long,
        flag_quality_long,
        pennant_quality_long,
    )
    flag_setup_short, pennant_setup_short = _resolve_continuation_name_conflicts(
        flag_setup_short,
        pennant_setup_short,
        flag_quality_short,
        pennant_quality_short,
    )

    columns = {
        f"{p}_impulse_up_pct": impulse_up_pct,
        f"{p}_impulse_down_pct": impulse_down_pct,
        f"{p}_impulse_age_long": pd.Series(anchored["impulse_age_long"], index=frame.index, dtype="float64"),
        f"{p}_impulse_age_short": pd.Series(anchored["impulse_age_short"], index=frame.index, dtype="float64"),
        f"{p}_impulse_start_index_long": pd.Series(anchored["impulse_start_index_long"], index=frame.index, dtype="float64"),
        f"{p}_impulse_end_index_long": pd.Series(anchored["impulse_end_index_long"], index=frame.index, dtype="float64"),
        f"{p}_impulse_start_index_short": pd.Series(anchored["impulse_start_index_short"], index=frame.index, dtype="float64"),
        f"{p}_impulse_end_index_short": pd.Series(anchored["impulse_end_index_short"], index=frame.index, dtype="float64"),
        f"{p}_impulse_efficiency_long": pd.Series(anchored["impulse_efficiency_long"], index=frame.index, dtype="float64"),
        f"{p}_impulse_efficiency_short": pd.Series(anchored["impulse_efficiency_short"], index=frame.index, dtype="float64"),
        f"{p}_impulse_dominance_long": pd.Series(anchored["impulse_dominance_long"], index=frame.index, dtype="float64"),
        f"{p}_impulse_dominance_short": pd.Series(anchored["impulse_dominance_short"], index=frame.index, dtype="float64"),
        f"{p}_impulse_break_score_long": pd.Series(anchored["impulse_break_score_long"], index=frame.index, dtype="float64"),
        f"{p}_impulse_break_score_short": pd.Series(anchored["impulse_break_score_short"], index=frame.index, dtype="float64"),
        f"{p}_pattern_volume_score_long": pd.Series(anchored["pattern_volume_score_long"], index=frame.index, dtype="float64"),
        f"{p}_pattern_volume_score_short": pd.Series(anchored["pattern_volume_score_short"], index=frame.index, dtype="float64"),
        f"{p}_continuation_pole_score_long": pd.Series(anchored["continuation_pole_score_long"], index=frame.index, dtype="float64"),
        f"{p}_continuation_pole_score_short": pd.Series(anchored["continuation_pole_score_short"], index=frame.index, dtype="float64"),
        f"{p}_continuation_retrace_score_long": pd.Series(anchored["continuation_retrace_score_long"], index=frame.index, dtype="float64"),
        f"{p}_continuation_retrace_score_short": pd.Series(anchored["continuation_retrace_score_short"], index=frame.index, dtype="float64"),
        f"{p}_continuation_containment_score_long": pd.Series(anchored["continuation_containment_score_long"], index=frame.index, dtype="float64"),
        f"{p}_continuation_containment_score_short": pd.Series(anchored["continuation_containment_score_short"], index=frame.index, dtype="float64"),
        f"{p}_continuation_terminal_score_long": pd.Series(anchored["continuation_terminal_score_long"], index=frame.index, dtype="float64"),
        f"{p}_continuation_terminal_score_short": pd.Series(anchored["continuation_terminal_score_short"], index=frame.index, dtype="float64"),
        f"{p}_continuation_boundary_touch_score_long": pd.Series(anchored["continuation_boundary_touch_score_long"], index=frame.index, dtype="float64"),
        f"{p}_continuation_boundary_touch_score_short": pd.Series(anchored["continuation_boundary_touch_score_short"], index=frame.index, dtype="float64"),
        f"{p}_continuation_boundary_span_score_long": pd.Series(anchored["continuation_boundary_span_score_long"], index=frame.index, dtype="float64"),
        f"{p}_continuation_boundary_span_score_short": pd.Series(anchored["continuation_boundary_span_score_short"], index=frame.index, dtype="float64"),
        f"{p}_flag_shape_score_long": pd.Series(anchored["flag_shape_score_long"], index=frame.index, dtype="float64"),
        f"{p}_flag_shape_score_short": pd.Series(anchored["flag_shape_score_short"], index=frame.index, dtype="float64"),
        f"{p}_pennant_shape_score_long": pd.Series(anchored["pennant_shape_score_long"], index=frame.index, dtype="float64"),
        f"{p}_pennant_shape_score_short": pd.Series(anchored["pennant_shape_score_short"], index=frame.index, dtype="float64"),
        f"{p}_setup_range_pct": setup_range_pct,
        f"{p}_setup_to_impulse_range": setup_to_impulse,
        f"{p}_setup_retrace_long": long_retrace,
        f"{p}_setup_retrace_short": short_retrace,
        f"{p}_pattern_contraction_long": pd.Series(anchored["pattern_contraction_long"], index=frame.index, dtype="float64"),
        f"{p}_pattern_contraction_short": pd.Series(anchored["pattern_contraction_short"], index=frame.index, dtype="float64"),
        f"{p}_pattern_high_count_long": pd.Series(anchored["pattern_high_count_long"], index=frame.index, dtype="float64"),
        f"{p}_pattern_low_count_long": pd.Series(anchored["pattern_low_count_long"], index=frame.index, dtype="float64"),
        f"{p}_pattern_high_count_short": pd.Series(anchored["pattern_high_count_short"], index=frame.index, dtype="float64"),
        f"{p}_pattern_low_count_short": pd.Series(anchored["pattern_low_count_short"], index=frame.index, dtype="float64"),
        f"{p}_pattern_high_slope_pct_long": pd.Series(anchored["pattern_high_slope_pct_long"], index=frame.index, dtype="float64"),
        f"{p}_pattern_low_slope_pct_long": pd.Series(anchored["pattern_low_slope_pct_long"], index=frame.index, dtype="float64"),
        f"{p}_pattern_high_slope_pct_short": pd.Series(anchored["pattern_high_slope_pct_short"], index=frame.index, dtype="float64"),
        f"{p}_pattern_low_slope_pct_short": pd.Series(anchored["pattern_low_slope_pct_short"], index=frame.index, dtype="float64"),
        **_continuation_strategy_columns(
            frame.index,
            p,
            "flag",
            flag_setup_long,
            flag_setup_short,
            flag_quality_long,
            flag_quality_short,
            anchored,
        ),
        **_continuation_strategy_columns(
            frame.index,
            p,
            "pennant",
            pennant_setup_long,
            pennant_setup_short,
            pennant_quality_long,
            pennant_quality_short,
            anchored,
        ),
    }
    if bool(getattr(cfg, "include_pattern_diagnostics", False)):
        columns.update(
            {
                **_continuation_proof_columns(frame.index, f"{p}_flag_long", flag_setup_long, anchored, "long"),
                **_continuation_proof_columns(frame.index, f"{p}_flag_short", flag_setup_short, anchored, "short"),
                **_continuation_proof_columns(frame.index, f"{p}_pennant_long", pennant_setup_long, anchored, "long"),
                **_continuation_proof_columns(frame.index, f"{p}_pennant_short", pennant_setup_short, anchored, "short"),
            }
        )
    return columns


def _continuation_strategy_columns(
    index: pd.Index,
    prefix: str,
    name: str,
    long_mask: Series,
    short_mask: Series,
    long_quality: Series,
    short_quality: Series,
    anchored: dict[str, np.ndarray],
) -> dict[str, Series]:
    """Return the compact strategy-facing continuation contract.

    Long and short internals are still scored separately because the geometry
    is asymmetric after an impulse. The strategy-facing output compresses that
    into one pattern flag, one direction, one score, and the active upper/lower
    consolidation rails. ``direction`` is shape direction, not a trade command:
    +1 means bullish-continuation structure, -1 means bearish-continuation
    structure, and 0 means no active pattern on that row.
    """

    long_active = pd.Series(long_mask, index=index).fillna(False).astype("bool")
    short_active = pd.Series(short_mask, index=index).fillna(False).astype("bool")
    present = long_active | short_active
    direction = pd.Series(np.select([long_active, short_active], [1, -1], default=0), index=index, dtype="int8")
    score = pd.Series(
        np.where(long_active, long_quality, np.where(short_active, short_quality, 0.0)),
        index=index,
        dtype="float64",
    )
    upper_long = pd.Series(anchored["proof_line2_y2_long"], index=index, dtype="float64").where(long_active)
    upper_short = pd.Series(anchored["proof_line2_y2_short"], index=index, dtype="float64").where(short_active)
    lower_long = pd.Series(anchored["proof_line3_y2_long"], index=index, dtype="float64").where(long_active)
    lower_short = pd.Series(anchored["proof_line3_y2_short"], index=index, dtype="float64").where(short_active)
    return {
        f"{prefix}_{name}_pattern_present": present,
        f"{prefix}_{name}_direction": direction,
        f"{prefix}_{name}_indicator_score": score.where(present, 0.0),
        f"{prefix}_{name}_upper": upper_long.combine_first(upper_short),
        f"{prefix}_{name}_lower": lower_long.combine_first(lower_short),
    }


def _anchored_flag_pennant_arrays(frame: DataFrame, cfg: PatternContinuationConfig) -> dict[str, np.ndarray]:
    """Score continuation setups after a specific impulse extreme.

    The variable impulse-end anchor makes pure rolling-vector formulas a poor
    fit here. This uses bounded NumPy windows instead of row-wise dataframe
    mutation: each row evaluates recent impulse candidates and then looks only
    at confirmed pivots whose anchors occur after that impulse high/low.
    """

    rows = len(frame)
    close = _num(frame["close"]).replace(0.0, np.nan).to_numpy(dtype="float64")
    open_ = _num(frame["open"]).to_numpy(dtype="float64")
    volume = _num(frame["volume"]).replace(0.0, np.nan).to_numpy(dtype="float64")
    body_high = np.fmax(open_, close)
    body_low = np.fmin(open_, close)
    pp = cfg.pivot_prefix
    high_pivot = _num(frame[f"{pp}_pivot_high"]).to_numpy(dtype="float64")
    low_pivot = _num(frame[f"{pp}_pivot_low"]).to_numpy(dtype="float64")
    fallback_index = np.arange(rows, dtype="float64")
    high_index = _num(frame.get(f"{pp}_pivot_high_index", pd.Series(fallback_index, index=frame.index))).to_numpy(dtype="float64")
    low_index = _num(frame.get(f"{pp}_pivot_low_index", pd.Series(fallback_index, index=frame.index))).to_numpy(dtype="float64")
    # Flags/pennants need local consolidation swings. The cleaned pivot
    # foundation remains the shared source, but it can be too sparse inside
    # short flags.
    # This internal source is still no-lookahead: each micro pivot is emitted
    # only after ``pattern_pivot_strength`` future candles confirm it.
    pattern_high_pivot, pattern_high_index, pattern_low_pivot, pattern_low_index = _confirmed_micro_pivots(
        body_high,
        body_low,
        int(cfg.pattern_pivot_strength),
    )
    high_pivot = _combine_sparse_pivots(high_pivot, high_index, pattern_high_pivot)
    low_pivot = _combine_sparse_pivots(low_pivot, low_index, pattern_low_pivot)
    high_index = _combine_sparse_indexes(high_pivot, high_index, pattern_high_index)
    low_index = _combine_sparse_indexes(low_pivot, low_index, pattern_low_index)

    out = _empty_pattern_arrays(rows)
    for row in range(rows):
        long_metrics = _best_anchored_pattern_for_side(
            row,
            1,
            body_high,
            body_low,
            close,
            volume,
            high_pivot,
            low_pivot,
            high_index,
            low_index,
            cfg,
        )
        short_metrics = _best_anchored_pattern_for_side(
            row,
            -1,
            body_high,
            body_low,
            close,
            volume,
            high_pivot,
            low_pivot,
            high_index,
            low_index,
            cfg,
        )
        _write_side_metrics(out, row, long_metrics, "long")
        _write_side_metrics(out, row, short_metrics, "short")

        long_best = max(long_metrics["flag_quality"], long_metrics["pennant_quality"])
        short_best = max(short_metrics["flag_quality"], short_metrics["pennant_quality"])
        dominance = float(cfg.pattern_side_dominance_mult)
        if long_best > 0.0 and short_best > 0.0:
            if long_best >= short_best * dominance:
                out["flag_setup_short"][row] = False
                out["pennant_setup_short"][row] = False
                out["flag_quality_short"][row] = 0.0
                out["pennant_quality_short"][row] = 0.0
            elif short_best >= long_best * dominance:
                out["flag_setup_long"][row] = False
                out["pennant_setup_long"][row] = False
                out["flag_quality_long"][row] = 0.0
                out["pennant_quality_long"][row] = 0.0
            else:
                out["flag_setup_long"][row] = False
                out["pennant_setup_long"][row] = False
                out["flag_setup_short"][row] = False
                out["pennant_setup_short"][row] = False
                out["flag_quality_long"][row] = 0.0
                out["pennant_quality_long"][row] = 0.0
                out["flag_quality_short"][row] = 0.0
                out["pennant_quality_short"][row] = 0.0

    return out


def _dedupe_continuation_events(mask: Series, cooldown_bars: int) -> Series:
    clean = mask.fillna(False).astype("bool")
    cooldown = max(int(cooldown_bars), 1)
    if cooldown <= 1:
        return clean
    recent = clean.shift(1, fill_value=False).rolling(cooldown, min_periods=1).max().astype("bool")
    return clean & ~recent


def _resolve_continuation_name_conflicts(
    flag: Series,
    pennant: Series,
    flag_quality: Series,
    pennant_quality: Series,
) -> tuple[Series, Series]:
    both = flag.fillna(False) & pennant.fillna(False)
    keep_flag = ~(both & pennant_quality.gt(flag_quality))
    keep_pennant = ~(both & flag_quality.ge(pennant_quality))
    return (flag & keep_flag).fillna(False), (pennant & keep_pennant).fillna(False)


def _empty_pattern_arrays(rows: int) -> dict[str, np.ndarray]:
    floats = {
        "impulse_up_pct",
        "impulse_down_pct",
        "impulse_age_long",
        "impulse_age_short",
        "impulse_start_index_long",
        "impulse_end_index_long",
        "impulse_start_index_short",
        "impulse_end_index_short",
        "impulse_efficiency_long",
        "impulse_efficiency_short",
        "impulse_dominance_long",
        "impulse_dominance_short",
        "impulse_break_score_long",
        "impulse_break_score_short",
        "pattern_volume_score_long",
        "pattern_volume_score_short",
        "setup_range_pct",
        "setup_to_impulse",
        "setup_retrace_long",
        "setup_retrace_short",
        "pattern_contraction_long",
        "pattern_contraction_short",
        "pattern_high_count_long",
        "pattern_low_count_long",
        "pattern_high_count_short",
        "pattern_low_count_short",
        "pattern_high_slope_pct_long",
        "pattern_low_slope_pct_long",
        "pattern_high_slope_pct_short",
        "pattern_low_slope_pct_short",
        "continuation_pole_score_long",
        "continuation_pole_score_short",
        "continuation_retrace_score_long",
        "continuation_retrace_score_short",
        "continuation_containment_score_long",
        "continuation_containment_score_short",
        "continuation_terminal_score_long",
        "continuation_terminal_score_short",
        "continuation_boundary_touch_score_long",
        "continuation_boundary_touch_score_short",
        "continuation_boundary_span_score_long",
        "continuation_boundary_span_score_short",
        "flag_shape_score_long",
        "flag_shape_score_short",
        "pennant_shape_score_long",
        "pennant_shape_score_short",
        "flag_quality_long",
        "flag_quality_short",
        "pennant_quality_long",
        "pennant_quality_short",
    }
    for side in ("long", "short"):
        for line_number in (1, 2, 3):
            for field in ("x1", "y1", "x2", "y2"):
                floats.add(f"proof_line{line_number}_{field}_{side}")
    out = {name: np.full(rows, np.nan, dtype="float64") for name in floats}
    out.update(
        {
            "flag_setup_long": np.zeros(rows, dtype=bool),
            "flag_setup_short": np.zeros(rows, dtype=bool),
            "pennant_setup_long": np.zeros(rows, dtype=bool),
            "pennant_setup_short": np.zeros(rows, dtype=bool),
        }
    )
    return out


def _empty_side_metrics() -> dict[str, float | bool]:
    metrics: dict[str, float | bool] = {
        "impulse_pct": 0.0,
        "age": np.nan,
        "impulse_start": np.nan,
        "impulse_end": np.nan,
        "impulse_efficiency": 0.0,
        "impulse_dominance": 0.0,
        "impulse_break_score": 0.0,
        "volume_score": 0.0,
        "setup_range_pct": np.nan,
        "setup_to_impulse": np.nan,
        "retrace": np.nan,
        "pattern_contraction": 0.0,
        "high_count": 0.0,
        "low_count": 0.0,
        "high_slope_pct": 0.0,
        "low_slope_pct": 0.0,
        "pole_score": 0.0,
        "retrace_score": 0.0,
        "containment_score": 0.0,
        "terminal_score": 0.0,
        "boundary_touch_score": 0.0,
        "boundary_span_score": 0.0,
        "flag_shape_score": 0.0,
        "pennant_shape_score": 0.0,
        "flag_quality": 0.0,
        "pennant_quality": 0.0,
        "flag_setup": False,
        "pennant_setup": False,
    }
    for line_number in (1, 2, 3):
        for field in ("x1", "y1", "x2", "y2"):
            metrics[f"proof_line{line_number}_{field}"] = np.nan
    return metrics


def _best_anchored_pattern_for_side(
    row: int,
    direction: int,
    body_high: np.ndarray,
    body_low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    high_pivot: np.ndarray,
    low_pivot: np.ndarray,
    high_index: np.ndarray,
    low_index: np.ndarray,
    cfg: PatternContinuationConfig,
) -> dict[str, float | bool]:
    best = _empty_side_metrics()
    latest_end = row - int(cfg.min_pattern_bars)
    if latest_end <= 1 or not np.isfinite(close[row]) or close[row] == 0.0:
        return best

    search_start = max(0, row - int(cfg.impulse_window) + 1)
    max_age = int(cfg.max_impulse_extreme_age_bars)
    earliest_end = max(search_start + 1, row - max_age)
    for impulse_end in range(earliest_end, latest_end + 1):
        if direction > 0:
            start_slice = body_low[search_start : impulse_end + 1]
            if not np.isfinite(start_slice).any() or not np.isfinite(body_high[impulse_end]):
                continue
            impulse_start = search_start + int(np.nanargmin(start_slice))
            start_price = float(body_low[impulse_start])
            end_price = float(body_high[impulse_end])
            if impulse_start >= impulse_end or start_price <= 0.0 or end_price <= start_price:
                continue
            leg_extreme = float(np.nanmax(body_high[impulse_start : impulse_end + 1]))
            if end_price < leg_extreme * (1.0 - float(cfg.impulse_extreme_tolerance_pct)):
                continue
        else:
            start_slice = body_high[search_start : impulse_end + 1]
            if not np.isfinite(start_slice).any() or not np.isfinite(body_low[impulse_end]):
                continue
            impulse_start = search_start + int(np.nanargmax(start_slice))
            start_price = float(body_high[impulse_start])
            end_price = float(body_low[impulse_end])
            if impulse_start >= impulse_end or end_price <= 0.0 or start_price <= end_price:
                continue
            leg_extreme = float(np.nanmin(body_low[impulse_start : impulse_end + 1]))
            if end_price > leg_extreme * (1.0 + float(cfg.impulse_extreme_tolerance_pct)):
                continue

        if impulse_end - impulse_start < int(cfg.min_impulse_bars):
            continue
        impulse_range = abs(end_price - start_price)
        impulse_pct = impulse_range / max(abs(start_price), 1e-9)
        if impulse_pct < float(cfg.min_impulse_pct):
            continue
        impulse_efficiency = _impulse_efficiency(close, impulse_start, impulse_end, impulse_range)
        if impulse_efficiency < float(cfg.min_impulse_efficiency):
            continue
        opposite_impulse_pct = _opposite_impulse_pct(search_start, impulse_end, direction, body_high, body_low)
        dominance_score = _impulse_dominance_score(impulse_pct, opposite_impulse_pct, float(cfg.impulse_dominance_mult))
        if dominance_score < float(cfg.min_impulse_dominance_score):
            continue
        break_score = _impulse_break_score(search_start, impulse_start, direction, end_price, body_high, body_low, impulse_range)
        if break_score < float(cfg.min_impulse_break_score):
            continue

        metrics = _score_anchored_pattern(
            row,
            direction,
            impulse_start,
            impulse_end,
            start_price,
            end_price,
            impulse_range,
            impulse_pct,
            impulse_efficiency,
            dominance_score,
            break_score,
            body_high,
            body_low,
            close,
            volume,
            high_pivot,
            low_pivot,
            high_index,
            low_index,
            cfg,
        )
        if max(metrics["flag_quality"], metrics["pennant_quality"]) > max(best["flag_quality"], best["pennant_quality"]):
            best = metrics

    return best


def _score_anchored_pattern(
    row: int,
    direction: int,
    impulse_start: int,
    impulse_end: int,
    start_price: float,
    end_price: float,
    impulse_range: float,
    impulse_pct: float,
    impulse_efficiency: float,
    impulse_dominance: float,
    impulse_break_score: float,
    body_high: np.ndarray,
    body_low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    high_pivot: np.ndarray,
    low_pivot: np.ndarray,
    high_index: np.ndarray,
    low_index: np.ndarray,
    cfg: PatternContinuationConfig,
) -> dict[str, float | bool]:
    metrics = _empty_side_metrics()
    age = row - impulse_end
    if age < int(cfg.min_pattern_bars) or age > int(cfg.max_impulse_extreme_age_bars):
        return metrics

    seg_start = impulse_end + 1
    seg_high = body_high[seg_start : row + 1]
    seg_low = body_low[seg_start : row + 1]
    if len(seg_high) == 0 or not np.isfinite(seg_high).any() or not np.isfinite(seg_low).any():
        return metrics

    setup_high = float(np.nanmax(seg_high))
    setup_low = float(np.nanmin(seg_low))
    setup_range = max(setup_high - setup_low, 0.0)
    setup_range_pct = setup_range / max(abs(float(close[row])), 1e-9)
    setup_to_impulse = setup_range / max(impulse_range, 1e-9)
    if setup_range_pct > float(cfg.max_setup_range_pct) or setup_to_impulse > float(cfg.max_setup_to_impulse_range_mult):
        return metrics
    pattern_contraction = _pattern_contraction_score(seg_high, seg_low)
    volume_score = _volume_pattern_score(volume, int(impulse_start), int(impulse_end), int(row), cfg)

    if direction > 0:
        if setup_high > end_price * (1.0 + float(cfg.max_pattern_breakout_pct)):
            return metrics
        retrace = (end_price - setup_low) / max(impulse_range, 1e-9)
    else:
        if setup_low < end_price * (1.0 - float(cfg.max_pattern_breakout_pct)):
            return metrics
        retrace = (setup_high - end_price) / max(impulse_range, 1e-9)
    if retrace < float(cfg.min_setup_retrace_pct) or retrace > float(cfg.max_setup_retrace_pct):
        return metrics

    high_x, high_y, low_x, low_y = _pattern_pivot_points(row, impulse_end, high_pivot, low_pivot, high_index, low_index)
    high_count = float(len(high_x))
    low_count = float(len(low_x))
    if high_count < float(cfg.min_pattern_side_pivots) or low_count < float(cfg.min_pattern_side_pivots):
        return metrics

    high_slope = _slope_from_points(high_x, high_y)
    low_slope = _slope_from_points(low_x, low_y)
    upper_x1, upper_y1, upper_x2, upper_y2 = _line_segment_from_points(high_x, high_y)
    lower_x1, lower_y1, lower_x2, lower_y2 = _line_segment_from_points(low_x, low_y)
    upper_now = _line_value_at(float(row), upper_x1, upper_y1, upper_x2, upper_y2)
    lower_now = _line_value_at(float(row), lower_x1, lower_y1, lower_x2, lower_y2)
    if np.isfinite(upper_now) and np.isfinite(lower_now) and upper_now > lower_now:
        boundary_tolerance = abs(float(close[row])) * float(cfg.max_pattern_boundary_excursion_pct)
        if direction > 0 and float(body_low[row]) < lower_now - boundary_tolerance:
            return metrics
        if direction < 0 and float(body_high[row]) > upper_now + boundary_tolerance:
            return metrics
    containment_score = _continuation_containment_score(
        int(impulse_end) + 1,
        int(row),
        body_high,
        body_low,
        close,
        upper_x1,
        upper_y1,
        upper_x2,
        upper_y2,
        lower_x1,
        lower_y1,
        lower_x2,
        lower_y2,
        float(cfg.max_pattern_boundary_excursion_pct),
    )
    terminal_score = _continuation_terminal_score(
        int(row),
        int(direction),
        body_high,
        body_low,
        close,
        upper_now,
        lower_now,
    )
    high_slope_pct = high_slope / max(abs(float(close[row])), 1e-9)
    low_slope_pct = low_slope / max(abs(float(close[row])), 1e-9)
    slope_diff = abs(high_slope_pct - low_slope_pct)
    parallel_score = _clip_value(1.0 - slope_diff / max(float(cfg.parallel_slope_tolerance_pct_per_bar), 1e-9))
    min_slope = float(cfg.min_boundary_slope_pct_per_bar)
    convergence_score = _clip_value((low_slope_pct - high_slope_pct) / max(min_slope * 6.0, 1e-9))
    max_counter = float(cfg.max_flag_counter_slope_pct_per_bar)
    drift_tolerance = float(cfg.max_consolidation_drift_pct_per_bar)
    mean_boundary_slope = (high_slope_pct + low_slope_pct) / 2.0
    if direction > 0:
        counter_drift_score = _clip_value((drift_tolerance - mean_boundary_slope) / max(drift_tolerance * 2.0, 1e-9))
        flag_shape_score = (
            parallel_score
            * counter_drift_score
            * _clip_value((max_counter - max(high_slope_pct, low_slope_pct)) / max(max_counter * 2.0, 1e-9))
        )
        pennant_shape_score = convergence_score if high_slope_pct <= -min_slope and low_slope_pct >= min_slope else 0.0
    else:
        counter_drift_score = _clip_value((mean_boundary_slope + drift_tolerance) / max(drift_tolerance * 2.0, 1e-9))
        flag_shape_score = (
            parallel_score
            * counter_drift_score
            * _clip_value((min(high_slope_pct, low_slope_pct) + max_counter) / max(max_counter * 2.0, 1e-9))
        )
        pennant_shape_score = convergence_score if high_slope_pct <= -min_slope and low_slope_pct >= min_slope else 0.0

    impulse_score = _clip_value(impulse_pct / max(float(cfg.min_impulse_pct) * 2.0, 1e-9))
    pole_score = _clip_value(0.45 * impulse_score + 0.25 * float(impulse_efficiency) + 0.18 * float(impulse_dominance) + 0.12 * float(impulse_break_score))
    compact_score = _clip_value(1.0 - setup_to_impulse / max(float(cfg.max_setup_to_impulse_range_mult), 1e-9))
    boundary_touch_score = _clip_value(min(high_count, low_count) / max(float(cfg.min_pattern_side_pivots) + 1.0, 1.0))
    boundary_span_score = _continuation_boundary_span_score(high_x, low_x, float(impulse_end), float(row))
    retrace_mid = (float(cfg.min_setup_retrace_pct) + float(cfg.max_setup_retrace_pct)) / 2.0
    retrace_half = max((float(cfg.max_setup_retrace_pct) - float(cfg.min_setup_retrace_pct)) / 2.0, 1e-9)
    retrace_score = _clip_value(1.0 - abs(retrace - retrace_mid) / retrace_half)
    age_score = _clip_value(1.0 - age / max(float(cfg.max_impulse_extreme_age_bars) * 1.25, 1.0))
    flag_quality = _clip_value(
        0.13 * impulse_score
        + 0.08 * float(impulse_efficiency)
        + 0.09 * float(impulse_dominance)
        + 0.10 * float(impulse_break_score)
        + 0.12 * compact_score
        + 0.18 * flag_shape_score
        + 0.05 * boundary_touch_score
        + 0.03 * boundary_span_score
        + 0.04 * pattern_contraction
        + 0.05 * volume_score
        + 0.08 * retrace_score
        + 0.04 * containment_score
        + 0.01 * age_score
    )
    pennant_quality = _clip_value(
        0.13 * impulse_score
        + 0.08 * float(impulse_efficiency)
        + 0.09 * float(impulse_dominance)
        + 0.10 * float(impulse_break_score)
        + 0.12 * compact_score
        + 0.18 * pennant_shape_score
        + 0.05 * boundary_touch_score
        + 0.03 * boundary_span_score
        + 0.04 * pattern_contraction
        + 0.05 * volume_score
        + 0.08 * retrace_score
        + 0.04 * containment_score
        + 0.01 * age_score
    )
    component_gate = (
        pole_score >= float(cfg.min_continuation_pole_score)
        and retrace_score >= float(cfg.min_continuation_retrace_score)
        and containment_score >= float(cfg.min_continuation_containment_score)
        and terminal_score >= float(cfg.min_continuation_terminal_score)
        and boundary_touch_score >= float(cfg.min_continuation_boundary_touch_score)
        and boundary_span_score >= float(cfg.min_continuation_boundary_span_score)
    )

    metrics.update(
        {
            "impulse_pct": float(impulse_pct),
            "age": float(age),
            "impulse_start": float(impulse_start),
            "impulse_end": float(impulse_end),
            "impulse_efficiency": float(impulse_efficiency),
            "impulse_dominance": float(impulse_dominance),
            "impulse_break_score": float(impulse_break_score),
            "volume_score": float(volume_score),
            "setup_range_pct": float(setup_range_pct),
            "setup_to_impulse": float(setup_to_impulse),
            "retrace": float(retrace),
            "pattern_contraction": float(pattern_contraction),
            "high_count": high_count,
            "low_count": low_count,
            "high_slope_pct": float(high_slope_pct),
            "low_slope_pct": float(low_slope_pct),
            "pole_score": float(pole_score),
            "retrace_score": float(retrace_score),
            "containment_score": float(containment_score),
            "terminal_score": float(terminal_score),
            "boundary_touch_score": float(boundary_touch_score),
            "boundary_span_score": float(boundary_span_score),
            "flag_shape_score": float(flag_shape_score),
            "pennant_shape_score": float(pennant_shape_score),
            "flag_quality": flag_quality,
            "pennant_quality": pennant_quality,
            "flag_setup": bool(component_gate and flag_shape_score > 0.0 and flag_quality >= float(cfg.min_flag_quality)),
            "pennant_setup": bool(component_gate and pennant_shape_score > 0.0 and pennant_quality >= float(cfg.min_pennant_quality)),
            "proof_line1_x1": float(impulse_start),
            "proof_line1_y1": float(start_price),
            "proof_line1_x2": float(impulse_end),
            "proof_line1_y2": float(end_price),
            "proof_line2_x1": float(upper_x1),
            "proof_line2_y1": float(upper_y1),
            "proof_line2_x2": float(upper_x2),
            "proof_line2_y2": float(upper_y2),
            "proof_line3_x1": float(lower_x1),
            "proof_line3_y1": float(lower_y1),
            "proof_line3_x2": float(lower_x2),
            "proof_line3_y2": float(lower_y2),
        }
    )
    return metrics


def _continuation_boundary_span_score(high_x: np.ndarray, low_x: np.ndarray, impulse_end: float, row: float) -> float:
    """Require both consolidation rails to span a meaningful part of the setup.

    This is a false-positive guard for flags/pennants where two pivots exist on
    each side but are clustered in one corner, making the proof lines look
    cleaner than the actual consolidation.
    """

    setup_span = max(float(row) - float(impulse_end), 1.0)
    high_span = float(np.nanmax(high_x) - np.nanmin(high_x)) if len(high_x) else 0.0
    low_span = float(np.nanmax(low_x) - np.nanmin(low_x)) if len(low_x) else 0.0
    return _clip_value(min(high_span, low_span) / setup_span)


def _continuation_containment_score(
    start: int,
    end: int,
    body_high: np.ndarray,
    body_low: np.ndarray,
    close: np.ndarray,
    upper_x1: float,
    upper_y1: float,
    upper_x2: float,
    upper_y2: float,
    lower_x1: float,
    lower_y1: float,
    lower_x2: float,
    lower_y2: float,
    tolerance_pct: float,
) -> float:
    if not np.isfinite([upper_x1, upper_y1, upper_x2, upper_y2, lower_x1, lower_y1, lower_x2, lower_y2]).all():
        return 0.0
    if upper_x2 == upper_x1 or lower_x2 == lower_x1:
        return 0.0
    start_i = max(int(start), 0)
    end_i = min(int(end), len(close) - 1)
    if end_i < start_i:
        return 0.0
    rows = np.arange(start_i, end_i + 1, dtype="float64")
    upper = np.array([_line_value_at(float(x), upper_x1, upper_y1, upper_x2, upper_y2) for x in rows], dtype="float64")
    lower = np.array([_line_value_at(float(x), lower_x1, lower_y1, lower_x2, lower_y2) for x in rows], dtype="float64")
    highs = body_high[start_i : end_i + 1]
    lows = body_low[start_i : end_i + 1]
    closes = close[start_i : end_i + 1]
    tolerance = np.abs(closes) * float(tolerance_pct)
    valid = np.isfinite(upper) & np.isfinite(lower) & np.isfinite(highs) & np.isfinite(lows) & (upper > lower)
    if not valid.any():
        return 0.0
    contained = (highs <= upper + tolerance) & (lows >= lower - tolerance)
    return _clip_value(float(np.sum(contained & valid)) / max(float(np.sum(valid)), 1.0))


def _continuation_terminal_score(
    row: int,
    direction: int,
    body_high: np.ndarray,
    body_low: np.ndarray,
    close: np.ndarray,
    upper_now: float,
    lower_now: float,
) -> float:
    if not np.isfinite([upper_now, lower_now]).all() or upper_now <= lower_now:
        return 0.0
    if row < 0 or row >= len(close) or not np.isfinite(close[row]) or close[row] == 0.0:
        return 0.0
    width = max(float(upper_now) - float(lower_now), abs(float(close[row])) * 0.001, 1e-9)
    if direction > 0:
        if not np.isfinite(body_high[row]):
            return 0.0
        distance = max(float(upper_now) - float(body_high[row]), 0.0)
    else:
        if not np.isfinite(body_low[row]):
            return 0.0
        distance = max(float(body_low[row]) - float(lower_now), 0.0)
    return _clip_value(1.0 - distance / max(width * 0.85, 1e-9))


def _impulse_efficiency(close: np.ndarray, impulse_start: int, impulse_end: int, impulse_range: float) -> float:
    leg = close[int(impulse_start) : int(impulse_end) + 1]
    valid = leg[np.isfinite(leg)]
    if len(valid) < 2:
        return 0.0
    travelled = float(np.sum(np.abs(np.diff(valid))))
    if travelled <= 0.0:
        return 0.0
    return _clip_value(float(impulse_range) / travelled)


def _opposite_impulse_pct(
    search_start: int,
    impulse_end: int,
    direction: int,
    body_high: np.ndarray,
    body_low: np.ndarray,
) -> float:
    start = max(int(search_start), 0)
    stop = max(int(impulse_end) + 1, start + 1)
    highs = body_high[start:stop]
    lows = body_low[start:stop]
    if len(highs) < 2 or not np.isfinite(highs).any() or not np.isfinite(lows).any():
        return 0.0
    best = 0.0
    if direction > 0:
        peak = -np.inf
        for high_value, low_value in zip(highs, lows):
            if np.isfinite(high_value):
                peak = max(peak, float(high_value))
            if np.isfinite(low_value) and np.isfinite(peak) and peak > 0.0:
                best = max(best, (peak - float(low_value)) / max(abs(peak), 1e-9))
    else:
        trough = np.inf
        for high_value, low_value in zip(highs, lows):
            if np.isfinite(low_value):
                trough = min(trough, float(low_value))
            if np.isfinite(high_value) and np.isfinite(trough) and trough > 0.0:
                best = max(best, (float(high_value) - trough) / max(abs(trough), 1e-9))
    return float(max(best, 0.0))


def _impulse_dominance_score(impulse_pct: float, opposite_impulse_pct: float, dominance_mult: float) -> float:
    if not np.isfinite(impulse_pct) or impulse_pct <= 0.0:
        return 0.0
    required = max(float(opposite_impulse_pct) * max(float(dominance_mult), 1.0), 1e-9)
    if required <= 1e-8:
        return 1.0
    return _clip_value(float(impulse_pct) / required)


def _impulse_break_score(
    search_start: int,
    impulse_start: int,
    direction: int,
    end_price: float,
    body_high: np.ndarray,
    body_low: np.ndarray,
    impulse_range: float,
) -> float:
    pre_high = body_high[int(search_start) : int(impulse_start) + 1]
    pre_low = body_low[int(search_start) : int(impulse_start) + 1]
    if len(pre_high) < 3 or not np.isfinite(pre_high).any() or not np.isfinite(pre_low).any():
        return 0.50
    if direction > 0:
        prior_high = float(np.nanmax(pre_high))
        break_size = float(end_price) - prior_high
    else:
        prior_low = float(np.nanmin(pre_low))
        break_size = prior_low - float(end_price)
    if break_size <= 0.0:
        return 0.0
    return _clip_value(break_size / max(float(impulse_range) * 0.25, 1e-9))


def _volume_pattern_score(
    volume: np.ndarray,
    impulse_start: int,
    impulse_end: int,
    row: int,
    cfg: PatternContinuationConfig,
) -> float:
    pre_start = max(0, int(impulse_start) - max(int(cfg.impulse_window) // 2, 3))
    pole = volume[int(impulse_start) : int(impulse_end) + 1]
    pre = volume[pre_start : int(impulse_start)]
    setup = volume[int(impulse_end) + 1 : int(row) + 1]
    if not np.isfinite(pole).any() or not np.isfinite(setup).any():
        return 0.50
    pole_mean = float(np.nanmean(pole))
    setup_mean = float(np.nanmean(setup))
    pre_mean = float(np.nanmean(pre)) if np.isfinite(pre).any() else setup_mean
    if pole_mean <= 0.0:
        return 0.50
    pole_score = _clip_value((pole_mean / max(pre_mean, 1e-9)) / max(float(cfg.min_impulse_volume_ratio), 1e-9))
    dry_score = _clip_value(1.20 - (setup_mean / max(pole_mean, 1e-9)))
    return _clip_value(0.55 * pole_score + 0.45 * dry_score)


def _pattern_contraction_score(seg_high: np.ndarray, seg_low: np.ndarray) -> float:
    rows = min(len(seg_high), len(seg_low))
    if rows < 6:
        return 0.50
    midpoint = max(2, rows // 2)
    first_high = seg_high[:midpoint]
    first_low = seg_low[:midpoint]
    second_high = seg_high[midpoint:]
    second_low = seg_low[midpoint:]
    if not np.isfinite(first_high).any() or not np.isfinite(first_low).any():
        return 0.50
    if not np.isfinite(second_high).any() or not np.isfinite(second_low).any():
        return 0.50
    first_range = float(np.nanmax(first_high) - np.nanmin(first_low))
    second_range = float(np.nanmax(second_high) - np.nanmin(second_low))
    if first_range <= 0.0:
        return 0.50
    return _clip_value(1.15 - (second_range / max(first_range, 1e-9)))


def _pattern_pivot_points(
    row: int,
    impulse_end: int,
    high_pivot: np.ndarray,
    low_pivot: np.ndarray,
    high_index: np.ndarray,
    low_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start = int(impulse_end) + 1
    stop = int(row) + 1
    high_prices = high_pivot[start:stop]
    high_x = high_index[start:stop]
    low_prices = low_pivot[start:stop]
    low_x = low_index[start:stop]
    high_mask = np.isfinite(high_prices) & np.isfinite(high_x) & (high_x > float(impulse_end)) & (high_x <= float(row))
    low_mask = np.isfinite(low_prices) & np.isfinite(low_x) & (low_x > float(impulse_end)) & (low_x <= float(row))
    return high_x[high_mask], high_prices[high_mask], low_x[low_mask], low_prices[low_mask]


def _write_side_metrics(out: dict[str, np.ndarray], row: int, metrics: dict[str, float | bool], label: str) -> None:
    impulse_key = "impulse_up_pct" if label == "long" else "impulse_down_pct"
    retrace_key = "setup_retrace_long" if label == "long" else "setup_retrace_short"
    out[impulse_key][row] = float(metrics["impulse_pct"])
    out[f"impulse_age_{label}"][row] = float(metrics["age"]) if np.isfinite(float(metrics["age"])) else np.nan
    out[f"impulse_start_index_{label}"][row] = (
        float(metrics["impulse_start"]) if np.isfinite(float(metrics["impulse_start"])) else np.nan
    )
    out[f"impulse_end_index_{label}"][row] = (
        float(metrics["impulse_end"]) if np.isfinite(float(metrics["impulse_end"])) else np.nan
    )
    out[f"impulse_efficiency_{label}"][row] = float(metrics["impulse_efficiency"])
    out[f"impulse_dominance_{label}"][row] = float(metrics["impulse_dominance"])
    out[f"impulse_break_score_{label}"][row] = float(metrics["impulse_break_score"])
    out[f"pattern_volume_score_{label}"][row] = float(metrics["volume_score"])
    out["setup_range_pct"][row] = _nanmax_pair(out["setup_range_pct"][row], float(metrics["setup_range_pct"]))
    out["setup_to_impulse"][row] = _nanmax_pair(out["setup_to_impulse"][row], float(metrics["setup_to_impulse"]))
    out[retrace_key][row] = float(metrics["retrace"])
    out[f"pattern_contraction_{label}"][row] = float(metrics["pattern_contraction"])
    out[f"pattern_high_count_{label}"][row] = float(metrics["high_count"])
    out[f"pattern_low_count_{label}"][row] = float(metrics["low_count"])
    out[f"pattern_high_slope_pct_{label}"][row] = float(metrics["high_slope_pct"])
    out[f"pattern_low_slope_pct_{label}"][row] = float(metrics["low_slope_pct"])
    out[f"continuation_pole_score_{label}"][row] = float(metrics["pole_score"])
    out[f"continuation_retrace_score_{label}"][row] = float(metrics["retrace_score"])
    out[f"continuation_containment_score_{label}"][row] = float(metrics["containment_score"])
    out[f"continuation_terminal_score_{label}"][row] = float(metrics["terminal_score"])
    out[f"continuation_boundary_touch_score_{label}"][row] = float(metrics["boundary_touch_score"])
    out[f"continuation_boundary_span_score_{label}"][row] = float(metrics["boundary_span_score"])
    out[f"flag_shape_score_{label}"][row] = float(metrics["flag_shape_score"])
    out[f"pennant_shape_score_{label}"][row] = float(metrics["pennant_shape_score"])
    out[f"flag_quality_{label}"][row] = float(metrics["flag_quality"])
    out[f"pennant_quality_{label}"][row] = float(metrics["pennant_quality"])
    out[f"flag_setup_{label}"][row] = bool(metrics["flag_setup"])
    out[f"pennant_setup_{label}"][row] = bool(metrics["pennant_setup"])
    for line_number in (1, 2, 3):
        for field in ("x1", "y1", "x2", "y2"):
            value = float(metrics[f"proof_line{line_number}_{field}"])
            out[f"proof_line{line_number}_{field}_{label}"][row] = value if np.isfinite(value) else np.nan

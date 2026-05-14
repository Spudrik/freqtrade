from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

try:
    from .pivot_foundation import build_clean_pivot_source
except Exception:  # pragma: no cover - standalone review scripts import directly
    from pivot_foundation import build_clean_pivot_source  # type: ignore[no-redef]

from pattern_common import (
    _carry_values_while_state,
    _clip_value,
    _dedupe_interval_level_events,
    _dynamic_height_tolerance_pct,
    _line_fit_with_error,
    _line_value_at,
    _num,
    _prior_impulse_score,
    _prior_opposite_pivot_context,
    _proof_line_columns,
    _rolling_atr_pct,
    _rolling_body_pct,
    _rolling_pivot_prominence_pct,
    _threshold_body_pct,
    _threshold_scale_pct,
)


@dataclass(frozen=True)
class PatternWolfeWaveConfig:
    """Minimal Wolfe Wave / three-drive exhaustion detector.

    This implementation intentionally stays narrow. It only aims to be
    practically useful for Wolfe-style exhaustion waves:
    - confirmed five-pivot exhaustion structure
    - prior move into point 1
    - dynamic same-level / line-touch tolerance
    - developing then confirmed lifecycle

    Bearish candidates use ``high-low-high-low-high`` pivots and confirm when
    price closes through the ``2-4`` support line. Bullish candidates mirror
    that with ``low-high-low-high-low`` pivots and confirm through the ``2-4``
    resistance line.
    """

    output_prefix: str = "pww"
    pivot_prefix: str = "pf"
    atr_period: int = 14
    pivot_strength: int = 2
    pivot_min_prominence_atr: float = 0.35
    pivot_min_prominence_pct: float = 0.0
    pivot_min_spacing_bars: int = 1
    pivot_min_distance_atr: float = 0.0
    pivot_min_distance_pct: float = 0.0
    scale_window: int = 30
    min_pattern_bars: int = 18
    max_pattern_bars: int = 110
    min_leg_spacing_bars: int = 3
    prior_window: int = 64
    min_prior_move_pct: float = 0.05
    min_prior_impulse_bars: int = 3
    min_prior_impulse_score: float = 0.40
    min_boundary_slope_pct_per_bar: float = 0.00015
    max_boundary_fit_error_pct: float = 0.020
    max_width_expansion_ratio: float = 1.15
    min_internal_reaction_pct: float = 0.015
    min_internal_reaction_body_mult: float = 1.25
    min_internal_reaction_atr_mult: float = 0.30
    min_internal_reaction_prominence_mult: float = 0.0
    p5_line_tolerance_pct: float = 0.040
    p5_line_tolerance_body_mult: float = 2.25
    p5_line_tolerance_atr_mult: float = 0.35
    p5_line_tolerance_prominence_mult: float = 0.20
    confirmation_break_pct: float = 0.012
    confirmation_break_body_mult: float = 1.00
    confirmation_break_atr_mult: float = 0.25
    min_wave_quality: float = 0.72
    entry_cooldown_bars: int = 12
    duplicate_overlap_pct: float = 0.70
    duplicate_level_tolerance_pct: float = 0.012
    lifecycle_mature_bars: int = 12
    lifecycle_stale_bars: int = 12
    include_pattern_diagnostics: bool = False


def add_pattern_wolfe_waves(
    dataframe: DataFrame,
    config: PatternWolfeWaveConfig | None = None,
    **overrides: object,
) -> DataFrame:
    """Append minimal Wolfe Wave / three-drive exhaustion columns."""

    cfg = _resolve_config(config, overrides)
    _validate_config(cfg)
    _validate_dataframe(dataframe)

    base = _with_foundation_pivots(dataframe, cfg)
    arrays = _wolfe_wave_arrays(base, cfg)
    index = base.index
    p = cfg.output_prefix
    close = _num(base["close"]).to_numpy(dtype="float64")
    body_pct = _rolling_body_pct(base, int(cfg.scale_window))
    atr_pct = _rolling_atr_pct(base, int(cfg.scale_window))
    prominence_pct = _rolling_pivot_prominence_pct(base, cfg.pivot_prefix, int(cfg.scale_window))

    bearish_setup = pd.Series(
        _dedupe_interval_level_events(
            arrays["bearish_setup_raw"],
            arrays["bearish_first_index"],
            arrays["bearish_fifth_index"],
            arrays["bearish_confirmation_level_event"],
            arrays["bearish_target_level_event"],
            int(cfg.entry_cooldown_bars),
            float(cfg.duplicate_overlap_pct),
            float(cfg.duplicate_level_tolerance_pct),
        ),
        index=index,
        dtype="bool",
    ).fillna(False)
    bullish_setup = pd.Series(
        _dedupe_interval_level_events(
            arrays["bullish_setup_raw"],
            arrays["bullish_first_index"],
            arrays["bullish_fifth_index"],
            arrays["bullish_confirmation_level_event"],
            arrays["bullish_target_level_event"],
            int(cfg.entry_cooldown_bars),
            float(cfg.duplicate_overlap_pct),
            float(cfg.duplicate_level_tolerance_pct),
        ),
        index=index,
        dtype="bool",
    ).fillna(False)

    bearish_state = _project_wave_state(
        close,
        bearish_setup.to_numpy(dtype=bool),
        arrays,
        body_pct,
        atr_pct,
        prominence_pct,
        cfg,
        side="bearish",
    )
    bullish_state = _project_wave_state(
        close,
        bullish_setup.to_numpy(dtype=bool),
        arrays,
        body_pct,
        atr_pct,
        prominence_pct,
        cfg,
        side="bullish",
    )

    bearish_present = pd.Series(bearish_state["state"], index=index, dtype="int8").gt(0)
    bullish_present = pd.Series(bullish_state["state"], index=index, dtype="int8").gt(0)
    columns: dict[str, Series] = {
        f"{p}_bearish_pattern_present": bearish_present.fillna(False),
        f"{p}_bullish_pattern_present": bullish_present.fillna(False),
        f"{p}_bearish_pattern_confirmed": pd.Series(bearish_state["confirmed"], index=index, dtype="bool").fillna(False),
        f"{p}_bullish_pattern_confirmed": pd.Series(bullish_state["confirmed"], index=index, dtype="bool").fillna(False),
        f"{p}_bearish_indicator_score": pd.Series(bearish_state["quality"], index=index, dtype="float64"),
        f"{p}_bullish_indicator_score": pd.Series(bullish_state["quality"], index=index, dtype="float64"),
        f"{p}_bearish_confirmation_level": pd.Series(bearish_state["confirmation_level"], index=index, dtype="float64"),
        f"{p}_bullish_confirmation_level": pd.Series(bullish_state["confirmation_level"], index=index, dtype="float64"),
        f"{p}_bearish_target_level": pd.Series(bearish_state["target_level"], index=index, dtype="float64"),
        f"{p}_bullish_target_level": pd.Series(bullish_state["target_level"], index=index, dtype="float64"),
        **_wolfe_pivot_index_columns(
            index,
            bearish_setup.to_numpy(dtype=bool),
            bullish_setup.to_numpy(dtype=bool),
            bearish_state["state"],
            bullish_state["state"],
            arrays,
            p,
        ),
    }
    if bool(cfg.include_pattern_diagnostics):
        columns.update(
            _wolfe_diagnostic_columns(
                base.index,
                bearish_setup.to_numpy(dtype=bool),
                bullish_setup.to_numpy(dtype=bool),
                bearish_state["state"],
                bullish_state["state"],
                arrays,
                p,
            )
        )

    existing = [col for col in dataframe.columns if str(col).startswith(f"{p}_")]
    clean = dataframe.drop(columns=existing).copy() if existing else dataframe.copy()
    return pd.concat([clean, pd.DataFrame(columns, index=index)], axis=1)


def _wolfe_pivot_index_columns(
    index: pd.Index,
    bearish_setup: np.ndarray,
    bullish_setup: np.ndarray,
    bearish_state: np.ndarray,
    bullish_state: np.ndarray,
    arrays: dict[str, np.ndarray],
    prefix: str,
) -> dict[str, Series]:
    columns: dict[str, Series] = {}
    for side in ("bearish", "bullish"):
        setup = bearish_setup if side == "bearish" else bullish_setup
        state = bearish_state if side == "bearish" else bullish_state
        carried = _carry_values_while_state(
            setup,
            state,
            {f"p{number}_index": arrays[f"{side}_p{number}_index"] for number in range(1, 6)},
        )
        for name, values in carried.items():
            columns[f"{prefix}_{side}_{name}"] = pd.Series(values, index=index, dtype="float64")
    return columns


def _resolve_config(
    config: PatternWolfeWaveConfig | None,
    overrides: dict[str, object],
) -> PatternWolfeWaveConfig:
    cfg = config or PatternWolfeWaveConfig()
    valid = {field.name for field in fields(PatternWolfeWaveConfig)}
    clean = {key: value for key, value in overrides.items() if value is not None}
    unknown = sorted(set(clean).difference(valid))
    if unknown:
        raise TypeError(f"Unknown Wolfe Wave config override(s): {', '.join(unknown)}")
    return replace(cfg, **clean) if clean else cfg


def _validate_config(cfg: PatternWolfeWaveConfig) -> None:
    if not cfg.output_prefix or not cfg.pivot_prefix:
        raise ValueError("output_prefix and pivot_prefix must be set")
    if int(cfg.atr_period) < 2:
        raise ValueError("atr_period must be at least 2")
    if int(cfg.pivot_strength) < 1:
        raise ValueError("pivot_strength must be at least 1")
    if float(cfg.pivot_min_prominence_atr) < 0.0:
        raise ValueError("pivot_min_prominence_atr must be non-negative")
    if float(cfg.pivot_min_prominence_pct) < 0.0:
        raise ValueError("pivot_min_prominence_pct must be non-negative")
    if int(cfg.pivot_min_spacing_bars) < 1:
        raise ValueError("pivot_min_spacing_bars must be at least 1")
    if float(cfg.pivot_min_distance_atr) < 0.0:
        raise ValueError("pivot_min_distance_atr must be non-negative")
    if float(cfg.pivot_min_distance_pct) < 0.0:
        raise ValueError("pivot_min_distance_pct must be non-negative")
    if int(cfg.scale_window) < 3:
        raise ValueError("scale_window must be at least 3")
    if int(cfg.min_pattern_bars) < 5:
        raise ValueError("min_pattern_bars must be at least 5")
    if int(cfg.max_pattern_bars) <= int(cfg.min_pattern_bars):
        raise ValueError("max_pattern_bars must be greater than min_pattern_bars")
    if int(cfg.min_leg_spacing_bars) < 1:
        raise ValueError("min_leg_spacing_bars must be at least 1")
    if int(cfg.prior_window) < 2:
        raise ValueError("prior_window must be at least 2")
    if float(cfg.min_prior_move_pct) < 0.0:
        raise ValueError("min_prior_move_pct must be non-negative")
    if int(cfg.min_prior_impulse_bars) < 1:
        raise ValueError("min_prior_impulse_bars must be at least 1")
    if not 0.0 <= float(cfg.min_prior_impulse_score) <= 1.0:
        raise ValueError("min_prior_impulse_score must be between 0 and 1")
    if float(cfg.min_boundary_slope_pct_per_bar) <= 0.0:
        raise ValueError("min_boundary_slope_pct_per_bar must be positive")
    if float(cfg.max_boundary_fit_error_pct) <= 0.0:
        raise ValueError("max_boundary_fit_error_pct must be positive")
    if float(cfg.max_width_expansion_ratio) < 1.0:
        raise ValueError("max_width_expansion_ratio must be at least 1")
    if float(cfg.min_internal_reaction_pct) < 0.0:
        raise ValueError("min_internal_reaction_pct must be non-negative")
    if float(cfg.min_internal_reaction_body_mult) < 0.0:
        raise ValueError("min_internal_reaction_body_mult must be non-negative")
    if float(cfg.min_internal_reaction_atr_mult) < 0.0:
        raise ValueError("min_internal_reaction_atr_mult must be non-negative")
    if float(cfg.min_internal_reaction_prominence_mult) < 0.0:
        raise ValueError("min_internal_reaction_prominence_mult must be non-negative")
    if float(cfg.p5_line_tolerance_pct) <= 0.0:
        raise ValueError("p5_line_tolerance_pct must be positive")
    if float(cfg.p5_line_tolerance_body_mult) < 0.0:
        raise ValueError("p5_line_tolerance_body_mult must be non-negative")
    if float(cfg.p5_line_tolerance_atr_mult) < 0.0:
        raise ValueError("p5_line_tolerance_atr_mult must be non-negative")
    if float(cfg.p5_line_tolerance_prominence_mult) < 0.0:
        raise ValueError("p5_line_tolerance_prominence_mult must be non-negative")
    if float(cfg.confirmation_break_pct) <= 0.0:
        raise ValueError("confirmation_break_pct must be positive")
    if float(cfg.confirmation_break_body_mult) < 0.0:
        raise ValueError("confirmation_break_body_mult must be non-negative")
    if float(cfg.confirmation_break_atr_mult) < 0.0:
        raise ValueError("confirmation_break_atr_mult must be non-negative")
    if not 0.0 <= float(cfg.min_wave_quality) <= 1.0:
        raise ValueError("min_wave_quality must be between 0 and 1")
    if int(cfg.entry_cooldown_bars) < 1:
        raise ValueError("entry_cooldown_bars must be at least 1")
    if not 0.0 <= float(cfg.duplicate_overlap_pct) <= 1.0:
        raise ValueError("duplicate_overlap_pct must be between 0 and 1")
    if float(cfg.duplicate_level_tolerance_pct) < 0.0:
        raise ValueError("duplicate_level_tolerance_pct must be non-negative")
    if int(cfg.lifecycle_mature_bars) < 1:
        raise ValueError("lifecycle_mature_bars must be at least 1")
    if int(cfg.lifecycle_stale_bars) < 0:
        raise ValueError("lifecycle_stale_bars must be non-negative")


def _validate_dataframe(dataframe: DataFrame) -> None:
    required = {"open", "high", "low", "close"}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise KeyError(f"Pattern Wolfe Waves requires columns: {', '.join(missing)}")


def _with_foundation_pivots(dataframe: DataFrame, cfg: PatternWolfeWaveConfig) -> DataFrame:
    pp = cfg.pivot_prefix
    required = {
        f"{pp}_pivot_high",
        f"{pp}_pivot_low",
        f"{pp}_pivot_high_index",
        f"{pp}_pivot_low_index",
        f"{pp}_pivot_high_prominence_pct",
        f"{pp}_pivot_low_prominence_pct",
    }
    if required.issubset(set(dataframe.columns)):
        return dataframe.copy()

    open_ = _num(dataframe["open"])
    close = _num(dataframe["close"]).replace(0.0, np.nan)
    body_high = pd.concat([open_, close], axis=1).max(axis=1)
    body_low = pd.concat([open_, close], axis=1).min(axis=1)
    atr = _atr(dataframe, int(cfg.atr_period))
    bar_index = pd.Series(np.arange(len(dataframe), dtype="float64"), index=dataframe.index)
    pivots = build_clean_pivot_source(
        body_high=body_high,
        body_low=body_low,
        atr=atr,
        bar_index=bar_index,
        strength=int(cfg.pivot_strength),
        min_prominence_atr=float(cfg.pivot_min_prominence_atr),
        min_prominence_pct=float(cfg.pivot_min_prominence_pct),
        min_pivot_spacing_bars=int(cfg.pivot_min_spacing_bars),
        min_pivot_distance_atr=float(cfg.pivot_min_distance_atr),
        min_pivot_distance_pct=float(cfg.pivot_min_distance_pct),
    )
    out = dataframe.copy()
    existing = [col for col in out.columns if str(col).startswith(f"{pp}_")]
    if existing:
        out = out.drop(columns=existing)
    for name, series in pivots.items():
        out[f"{pp}_{name}"] = series
    return out


def _atr(dataframe: DataFrame, period: int) -> Series:
    high = _num(dataframe["high"])
    low = _num(dataframe["low"])
    close = _num(dataframe["close"])
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(int(period), min_periods=1).mean()


def _wolfe_wave_arrays(frame: DataFrame, cfg: PatternWolfeWaveConfig) -> dict[str, np.ndarray]:
    rows = len(frame)
    out = _empty_output(rows)
    close = _num(frame["close"]).to_numpy(dtype="float64")
    pp = cfg.pivot_prefix
    high_pivot = _num(frame[f"{pp}_pivot_high"]).to_numpy(dtype="float64")
    low_pivot = _num(frame[f"{pp}_pivot_low"]).to_numpy(dtype="float64")
    high_index = _num(frame[f"{pp}_pivot_high_index"]).to_numpy(dtype="float64")
    low_index = _num(frame[f"{pp}_pivot_low_index"]).to_numpy(dtype="float64")
    body_pct = _rolling_body_pct(frame, int(cfg.scale_window))
    atr_pct = _rolling_atr_pct(frame, int(cfg.scale_window))
    prominence_pct = _rolling_pivot_prominence_pct(frame, pp, int(cfg.scale_window))
    events = _pivot_events(high_pivot, high_index, low_pivot, low_index)

    for offset in range(4, len(events)):
        sequence = events[offset - 4 : offset + 1]
        candidate = _evaluate_sequence(
            sequence,
            close,
            high_pivot,
            high_index,
            low_pivot,
            low_index,
            body_pct,
            atr_pct,
            prominence_pct,
            cfg,
        )
        if candidate is None:
            continue
        row = int(candidate["setup_row"])
        side = str(candidate["side"])
        prefix = side
        if float(candidate["quality"]) <= float(out[f"{prefix}_quality_event"][row]):
            continue
        out[f"{prefix}_setup_raw"][row] = True
        out[f"{prefix}_quality_event"][row] = float(candidate["quality"])
        out[f"{prefix}_first_index"][row] = float(candidate["p1_index"])
        out[f"{prefix}_fifth_index"][row] = float(candidate["p5_index"])
        out[f"{prefix}_confirmation_level_event"][row] = float(candidate["confirmation_level_event"])
        out[f"{prefix}_target_level_event"][row] = float(candidate["target_level_event"])
        out[f"{prefix}_confirm_x1"][row] = float(candidate["confirm_x1"])
        out[f"{prefix}_confirm_y1"][row] = float(candidate["confirm_y1"])
        out[f"{prefix}_confirm_x2"][row] = float(candidate["confirm_x2"])
        out[f"{prefix}_confirm_y2"][row] = float(candidate["confirm_y2"])
        out[f"{prefix}_target_x1"][row] = float(candidate["target_x1"])
        out[f"{prefix}_target_y1"][row] = float(candidate["target_y1"])
        out[f"{prefix}_target_x2"][row] = float(candidate["target_x2"])
        out[f"{prefix}_target_y2"][row] = float(candidate["target_y2"])
        out[f"{prefix}_p1_index"][row] = float(candidate["p1_index"])
        out[f"{prefix}_p2_index"][row] = float(candidate["p2_index"])
        out[f"{prefix}_p3_index"][row] = float(candidate["p3_index"])
        out[f"{prefix}_p4_index"][row] = float(candidate["p4_index"])
        out[f"{prefix}_p5_index"][row] = float(candidate["p5_index"])
        out[f"{prefix}_p1_price"][row] = float(candidate["p1_price"])
        out[f"{prefix}_p2_price"][row] = float(candidate["p2_price"])
        out[f"{prefix}_p3_price"][row] = float(candidate["p3_price"])
        out[f"{prefix}_p4_price"][row] = float(candidate["p4_price"])
        out[f"{prefix}_p5_price"][row] = float(candidate["p5_price"])
        out[f"{prefix}_line1_x1"][row] = float(candidate["line1_x1"])
        out[f"{prefix}_line1_y1"][row] = float(candidate["line1_y1"])
        out[f"{prefix}_line1_x2"][row] = float(candidate["line1_x2"])
        out[f"{prefix}_line1_y2"][row] = float(candidate["line1_y2"])
        out[f"{prefix}_line2_x1"][row] = float(candidate["line2_x1"])
        out[f"{prefix}_line2_y1"][row] = float(candidate["line2_y1"])
        out[f"{prefix}_line2_x2"][row] = float(candidate["line2_x2"])
        out[f"{prefix}_line2_y2"][row] = float(candidate["line2_y2"])
        out[f"{prefix}_line3_x1"][row] = float(candidate["line3_x1"])
        out[f"{prefix}_line3_y1"][row] = float(candidate["line3_y1"])
        out[f"{prefix}_line3_x2"][row] = float(candidate["line3_x2"])
        out[f"{prefix}_line3_y2"][row] = float(candidate["line3_y2"])
    return out


def _empty_output(rows: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for side in ("bearish", "bullish"):
        out[f"{side}_setup_raw"] = np.zeros(rows, dtype=bool)
        out[f"{side}_quality_event"] = np.zeros(rows, dtype="float64")
        out[f"{side}_first_index"] = np.full(rows, np.nan, dtype="float64")
        out[f"{side}_fifth_index"] = np.full(rows, np.nan, dtype="float64")
        out[f"{side}_confirmation_level_event"] = np.full(rows, np.nan, dtype="float64")
        out[f"{side}_target_level_event"] = np.full(rows, np.nan, dtype="float64")
        for name in (
            "confirm_x1",
            "confirm_y1",
            "confirm_x2",
            "confirm_y2",
            "target_x1",
            "target_y1",
            "target_x2",
            "target_y2",
            "p1_index",
            "p2_index",
            "p3_index",
            "p4_index",
            "p5_index",
            "p1_price",
            "p2_price",
            "p3_price",
            "p4_price",
            "p5_price",
            "line1_x1",
            "line1_y1",
            "line1_x2",
            "line1_y2",
            "line2_x1",
            "line2_y1",
            "line2_x2",
            "line2_y2",
            "line3_x1",
            "line3_y1",
            "line3_x2",
            "line3_y2",
        ):
            out[f"{side}_{name}"] = np.full(rows, np.nan, dtype="float64")
    return out


def _pivot_events(
    high_pivot: np.ndarray,
    high_index: np.ndarray,
    low_pivot: np.ndarray,
    low_index: np.ndarray,
) -> list[dict[str, float | int | str]]:
    events: list[dict[str, float | int | str]] = []
    for row in np.flatnonzero(np.isfinite(high_pivot) & np.isfinite(high_index)):
        events.append(
            {
                "row": int(row),
                "side": "high",
                "index": float(high_index[row]),
                "price": float(high_pivot[row]),
            }
        )
    for row in np.flatnonzero(np.isfinite(low_pivot) & np.isfinite(low_index)):
        events.append(
            {
                "row": int(row),
                "side": "low",
                "index": float(low_index[row]),
                "price": float(low_pivot[row]),
            }
        )
    events.sort(key=lambda item: (int(item["row"]), float(item["index"])))
    return events


def _evaluate_sequence(
    sequence: list[dict[str, float | int | str]],
    close: np.ndarray,
    high_pivot: np.ndarray,
    high_index: np.ndarray,
    low_pivot: np.ndarray,
    low_index: np.ndarray,
    body_pct: np.ndarray,
    atr_pct: np.ndarray,
    prominence_pct: np.ndarray,
    cfg: PatternWolfeWaveConfig,
) -> dict[str, float | int | str] | None:
    sides = tuple(str(item["side"]) for item in sequence)
    if sides == ("high", "low", "high", "low", "high"):
        return _evaluate_side(
            sequence,
            close,
            low_pivot,
            low_index,
            body_pct,
            atr_pct,
            prominence_pct,
            cfg,
            side="bearish",
        )
    if sides == ("low", "high", "low", "high", "low"):
        return _evaluate_side(
            sequence,
            close,
            high_pivot,
            high_index,
            body_pct,
            atr_pct,
            prominence_pct,
            cfg,
            side="bullish",
        )
    return None


def _evaluate_side(
    sequence: list[dict[str, float | int | str]],
    close: np.ndarray,
    opposite_pivot: np.ndarray,
    opposite_index: np.ndarray,
    body_pct: np.ndarray,
    atr_pct: np.ndarray,
    prominence_pct: np.ndarray,
    cfg: PatternWolfeWaveConfig,
    *,
    side: str,
) -> dict[str, float | int | str] | None:
    p1, p2, p3, p4, p5 = sequence
    setup_row = int(p5["row"])
    p1_x = float(p1["index"])
    p2_x = float(p2["index"])
    p3_x = float(p3["index"])
    p4_x = float(p4["index"])
    p5_x = float(p5["index"])
    p1_price = float(p1["price"])
    p2_price = float(p2["price"])
    p3_price = float(p3["price"])
    p4_price = float(p4["price"])
    p5_price = float(p5["price"])
    if not np.isfinite(
        [p1_x, p2_x, p3_x, p4_x, p5_x, p1_price, p2_price, p3_price, p4_price, p5_price]
    ).all():
        return None
    if not (p1_x < p2_x < p3_x < p4_x < p5_x):
        return None
    leg_spans = np.diff(np.array([p1_x, p2_x, p3_x, p4_x, p5_x], dtype="float64"))
    if np.any(leg_spans < float(cfg.min_leg_spacing_bars)):
        return None
    span = p5_x - p1_x
    if span < float(cfg.min_pattern_bars) or span > float(cfg.max_pattern_bars):
        return None
    if setup_row >= len(close) or not np.isfinite(close[setup_row]) or close[setup_row] == 0.0:
        return None

    body_ref = _threshold_body_pct(body_pct, setup_row)
    atr_ref = _threshold_scale_pct(atr_pct, setup_row)
    prominence_ref = _threshold_scale_pct(prominence_pct, setup_row)
    level_tol_pct = _dynamic_height_tolerance_pct(
        float(cfg.p5_line_tolerance_pct),
        body_ref,
        atr_ref,
        prominence_ref,
        float(cfg.p5_line_tolerance_body_mult),
        float(cfg.p5_line_tolerance_atr_mult),
        float(cfg.p5_line_tolerance_prominence_mult),
    )
    reference = max(abs(float(close[setup_row])), 1e-9)
    level_tol = reference * level_tol_pct
    min_internal_reaction_pct = _dynamic_height_tolerance_pct(
        float(cfg.min_internal_reaction_pct),
        body_ref,
        atr_ref,
        prominence_ref,
        float(cfg.min_internal_reaction_body_mult),
        float(cfg.min_internal_reaction_atr_mult),
        float(cfg.min_internal_reaction_prominence_mult),
    )
    min_internal_reaction = reference * min_internal_reaction_pct

    top = side == "bearish"
    prior = _prior_opposite_pivot_context(
        opposite_pivot,
        opposite_index,
        p1_x,
        p1_price,
        int(cfg.prior_window),
        top=top,
    )
    if float(prior["move_pct"]) < float(cfg.min_prior_move_pct):
        return None
    impulse_score = _prior_impulse_score(
        close,
        float(prior["index"]),
        float(prior["price"]),
        p1_x,
        p1_price,
        top=top,
        min_bars=int(cfg.min_prior_impulse_bars),
    )
    if impulse_score < float(cfg.min_prior_impulse_score):
        return None

    guide_p5 = _line_value_at(p5_x, p1_x, p1_price, p3_x, p3_price)
    confirm_at_p5 = _line_value_at(p5_x, p2_x, p2_price, p4_x, p4_price)
    confirm_at_row = _line_value_at(float(setup_row), p2_x, p2_price, p4_x, p4_price)
    target_at_row = _line_value_at(float(setup_row), p1_x, p1_price, p4_x, p4_price)
    if not np.isfinite([guide_p5, confirm_at_p5, confirm_at_row, target_at_row]).all():
        return None

    upper_slope, upper_fit_error, _ = _line_fit_with_error(
        np.array([p1_x, p3_x, p5_x], dtype="float64"),
        np.array([p1_price, p3_price, p5_price], dtype="float64"),
    )
    lower_slope = (p4_price - p2_price) / max(p4_x - p2_x, 1e-9)
    slope_scale = reference
    upper_slope_pct = upper_slope / slope_scale
    lower_slope_pct = lower_slope / slope_scale
    if top:
        if (
            (p1_price - p2_price) < min_internal_reaction
            or (p3_price - p4_price) < min_internal_reaction
        ):
            return None
        if upper_slope_pct < float(cfg.min_boundary_slope_pct_per_bar):
            return None
        if lower_slope_pct < float(cfg.min_boundary_slope_pct_per_bar):
            return None
        if p3_price < p1_price + level_tol:
            return None
        if p4_price < p2_price + level_tol * 0.5:
            return None
        if p5_price < p3_price - level_tol:
            return None
        if p5_price < guide_p5 - level_tol or abs(p5_price - guide_p5) > level_tol:
            return None
        width_mid = p3_price - _line_value_at(p3_x, p2_x, p2_price, p4_x, p4_price)
        width_end = p5_price - confirm_at_p5
    else:
        if (
            (p2_price - p1_price) < min_internal_reaction
            or (p4_price - p3_price) < min_internal_reaction
        ):
            return None
        if upper_slope_pct > -float(cfg.min_boundary_slope_pct_per_bar):
            return None
        if lower_slope_pct > -float(cfg.min_boundary_slope_pct_per_bar):
            return None
        if p3_price > p1_price - level_tol:
            return None
        if p4_price > p2_price - level_tol * 0.5:
            return None
        if p5_price > p3_price + level_tol:
            return None
        if p5_price > guide_p5 + level_tol or abs(p5_price - guide_p5) > level_tol:
            return None
        width_mid = _line_value_at(p3_x, p2_x, p2_price, p4_x, p4_price) - p3_price
        width_end = confirm_at_p5 - p5_price
    if not np.isfinite([width_mid, width_end]).all() or width_mid <= 0.0 or width_end <= 0.0:
        return None
    if width_end / max(width_mid, 1e-9) > float(cfg.max_width_expansion_ratio):
        return None
    if upper_fit_error > float(cfg.max_boundary_fit_error_pct):
        return None

    slope_gap_ratio = abs(upper_slope_pct - lower_slope_pct) / max(abs(upper_slope_pct), abs(lower_slope_pct), 1e-9)
    prior_move_score = _clip_value(float(prior["move_pct"]) / max(float(cfg.min_prior_move_pct) * 2.0, 1e-9))
    fit_score = _clip_value(1.0 - upper_fit_error / max(float(cfg.max_boundary_fit_error_pct), 1e-9))
    terminal_touch_score = _clip_value(1.0 - abs(p5_price - guide_p5) / max(level_tol, 1e-9))
    width_score = _clip_value(1.0 - max((width_end / max(width_mid, 1e-9)) - 1.0, 0.0) / max(float(cfg.max_width_expansion_ratio) - 1.0, 1e-9))
    parallel_score = _clip_value(1.0 - slope_gap_ratio)
    quality = _clip_value(
        0.24 * prior_move_score
        + 0.22 * impulse_score
        + 0.18 * fit_score
        + 0.16 * terminal_touch_score
        + 0.12 * width_score
        + 0.08 * parallel_score
    )
    if quality < float(cfg.min_wave_quality):
        return None

    line1_y2 = _line_value_at(float(setup_row), p1_x, p1_price, p3_x, p3_price)
    line2_y2 = confirm_at_row
    line3_y2 = target_at_row
    if not np.isfinite([line1_y2, line2_y2, line3_y2]).all():
        return None

    return {
        "side": side,
        "setup_row": int(setup_row),
        "quality": float(quality),
        "p1_index": p1_x,
        "p2_index": p2_x,
        "p3_index": p3_x,
        "p4_index": p4_x,
        "p5_index": p5_x,
        "p1_price": p1_price,
        "p2_price": p2_price,
        "p3_price": p3_price,
        "p4_price": p4_price,
        "p5_price": p5_price,
        "confirmation_level_event": float(confirm_at_row),
        "target_level_event": float(target_at_row),
        "confirm_x1": p2_x,
        "confirm_y1": p2_price,
        "confirm_x2": p4_x,
        "confirm_y2": p4_price,
        "target_x1": p1_x,
        "target_y1": p1_price,
        "target_x2": p4_x,
        "target_y2": p4_price,
        "line1_x1": p1_x,
        "line1_y1": p1_price,
        "line1_x2": float(setup_row),
        "line1_y2": float(line1_y2),
        "line2_x1": p2_x,
        "line2_y1": p2_price,
        "line2_x2": float(setup_row),
        "line2_y2": float(line2_y2),
        "line3_x1": p1_x,
        "line3_y1": p1_price,
        "line3_x2": float(setup_row),
        "line3_y2": float(line3_y2),
    }


def _project_wave_state(
    close: np.ndarray,
    setup: np.ndarray,
    arrays: dict[str, np.ndarray],
    body_pct: np.ndarray,
    atr_pct: np.ndarray,
    prominence_pct: np.ndarray,
    cfg: PatternWolfeWaveConfig,
    *,
    side: str,
) -> dict[str, np.ndarray]:
    rows = len(close)
    state = np.zeros(rows, dtype="int8")
    confirmed = np.zeros(rows, dtype=bool)
    quality = np.zeros(rows, dtype="float64")
    confirmation_level = np.full(rows, np.nan, dtype="float64")
    target_level = np.full(rows, np.nan, dtype="float64")
    mature = max(int(cfg.lifecycle_mature_bars), 1)
    stale = max(int(cfg.lifecycle_stale_bars), 0)
    max_age = mature + stale
    active_row = -1
    confirmed_row = -1
    active_quality = 0.0
    confirm_x1 = np.nan
    confirm_y1 = np.nan
    confirm_x2 = np.nan
    confirm_y2 = np.nan
    target_x1 = np.nan
    target_y1 = np.nan
    target_x2 = np.nan
    target_y2 = np.nan

    for row in range(rows):
        if setup[row]:
            active_row = int(row)
            confirmed_row = -1
            active_quality = float(arrays[f"{side}_quality_event"][row])
            confirm_x1 = float(arrays[f"{side}_confirm_x1"][row])
            confirm_y1 = float(arrays[f"{side}_confirm_y1"][row])
            confirm_x2 = float(arrays[f"{side}_confirm_x2"][row])
            confirm_y2 = float(arrays[f"{side}_confirm_y2"][row])
            target_x1 = float(arrays[f"{side}_target_x1"][row])
            target_y1 = float(arrays[f"{side}_target_y1"][row])
            target_x2 = float(arrays[f"{side}_target_x2"][row])
            target_y2 = float(arrays[f"{side}_target_y2"][row])

        if active_row < 0:
            continue

        age = row - active_row
        if confirmed_row < 0 and age > max_age:
            active_row = -1
            active_quality = 0.0
            continue

        current_confirm = _line_value_at(float(row), confirm_x1, confirm_y1, confirm_x2, confirm_y2)
        current_target = _line_value_at(float(row), target_x1, target_y1, target_x2, target_y2)
        if not np.isfinite(current_confirm):
            active_row = -1
            active_quality = 0.0
            continue

        confirmation_level[row] = current_confirm
        target_level[row] = current_target
        quality[row] = active_quality

        break_tol_pct = _dynamic_height_tolerance_pct(
            float(cfg.confirmation_break_pct),
            _threshold_body_pct(body_pct, row),
            _threshold_scale_pct(atr_pct, row),
            _threshold_scale_pct(prominence_pct, row),
            float(cfg.confirmation_break_body_mult),
            float(cfg.confirmation_break_atr_mult),
            0.0,
        )
        break_tol = max(abs(float(close[row])), 1e-9) * break_tol_pct if np.isfinite(close[row]) else np.nan
        if confirmed_row < 0 and np.isfinite([close[row], break_tol]).all():
            if side == "bearish":
                is_confirmed = float(close[row]) <= current_confirm - break_tol
            else:
                is_confirmed = float(close[row]) >= current_confirm + break_tol
            if is_confirmed:
                confirmed_row = int(row)

        if confirmed_row >= 0:
            confirmed_age = row - confirmed_row
            if confirmed_age <= mature:
                confirmed[row] = True
                state[row] = 2
            elif confirmed_age > max_age:
                quality[row] = 0.0
                active_row = -1
                confirmed_row = -1
                active_quality = 0.0
        elif age <= mature:
            state[row] = 1

    return {
        "state": state,
        "confirmed": confirmed,
        "quality": quality * (state > 0),
        "confirmation_level": confirmation_level,
        "target_level": target_level,
    }


def _wolfe_diagnostic_columns(
    index: pd.Index,
    bearish_setup: np.ndarray,
    bullish_setup: np.ndarray,
    bearish_state: np.ndarray,
    bullish_state: np.ndarray,
    arrays: dict[str, np.ndarray],
    prefix: str,
) -> dict[str, Series]:
    columns: dict[str, Series] = {}
    for side in ("bearish", "bullish"):
        setup = bearish_setup if side == "bearish" else bullish_setup
        state = bearish_state if side == "bearish" else bullish_state
        carried = _carry_values_while_state(
            setup,
            state,
            {
                f"{side}_p1_index": arrays[f"{side}_p1_index"],
                f"{side}_p2_index": arrays[f"{side}_p2_index"],
                f"{side}_p3_index": arrays[f"{side}_p3_index"],
                f"{side}_p4_index": arrays[f"{side}_p4_index"],
                f"{side}_p5_index": arrays[f"{side}_p5_index"],
                f"{side}_p1_price": arrays[f"{side}_p1_price"],
                f"{side}_p2_price": arrays[f"{side}_p2_price"],
                f"{side}_p3_price": arrays[f"{side}_p3_price"],
                f"{side}_p4_price": arrays[f"{side}_p4_price"],
                f"{side}_p5_price": arrays[f"{side}_p5_price"],
            },
        )
        for name, values in carried.items():
            out_name = name.replace(f"{side}_", f"{prefix}_{side}_")
            columns[out_name] = pd.Series(values, index=index, dtype="float64")
        columns.update(
            _proof_line_columns(
                index,
                f"{prefix}_{side}",
                pd.Series(setup, index=index, dtype="bool"),
                {
                    1: (
                        arrays[f"{side}_line1_x1"],
                        arrays[f"{side}_line1_y1"],
                        arrays[f"{side}_line1_x2"],
                        arrays[f"{side}_line1_y2"],
                    ),
                    2: (
                        arrays[f"{side}_line2_x1"],
                        arrays[f"{side}_line2_y1"],
                        arrays[f"{side}_line2_x2"],
                        arrays[f"{side}_line2_y2"],
                    ),
                    3: (
                        arrays[f"{side}_line3_x1"],
                        arrays[f"{side}_line3_y1"],
                        arrays[f"{side}_line3_x2"],
                        arrays[f"{side}_line3_y2"],
                    ),
                },
            )
        )
    return columns


__all__ = [
    "PatternWolfeWaveConfig",
    "add_pattern_wolfe_waves",
]

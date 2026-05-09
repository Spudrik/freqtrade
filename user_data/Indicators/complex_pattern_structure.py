from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

try:
    from .pivot_foundation import build_clean_pivot_source
except Exception:  # pragma: no cover - standalone review scripts import this module directly
    from pivot_foundation import build_clean_pivot_source  # type: ignore[no-redef]

from pattern_common import _num
from pattern_channel import _channel_pattern_columns
from pattern_continuation import _flag_pennant_columns
from pattern_geometry import _triangle_wedge_columns
from pattern_multi_peak import _triple_reversal_columns
from pattern_range import _rectangle_range_columns
from pattern_reversal import _double_reversal_columns, _head_shoulders_columns


@dataclass(frozen=True)
class PatternStructureConfig:
    """Foundation-facing pattern context scaffold.

    Pattern detection consumes pivots directly from ``pivot_foundation``.
    It must not depend on TLV2 or any unrelated indicator namespace just to
    access shared foundation data. Ranked trendlines/channels may still be
    consumed separately when a pattern family explicitly needs that evidence.

    This public module owns config, validation, shared context, and final
    dataframe assembly. Pattern-family internals live in separate modules so
    flags, reversals, and geometric review logic can be refined independently.
    """

    output_prefix: str = "pat"
    pivot_prefix: str = "pf"
    pivot_strength: int = 2
    pivot_min_prominence_atr: float = 0.0
    pivot_min_prominence_pct: float = 0.0
    pivot_min_spacing_bars: int = 1
    pivot_min_distance_atr: float = 0.0
    pivot_min_distance_pct: float = 0.0
    # TODO_DELETE_CHECK: trendline_prefix is a compatibility no-op after TLV2
    # line-pair pattern detection was retired. Pivots come from
    # pivot_foundation directly, not through trendline/channel namespaces.
    trendline_prefix: str = "tlv2"
    # Channel prefix is still an optional channel-evidence input when explicit
    # channel review/scaffolding is enabled.
    channel_prefix: str = "tlv2"
    channel_label: str = "local_channel"
    channel_rank: int = 0
    sequence_window: int = 28
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
    # Defaults are intentionally conservative: this indicator should emit
    # "pattern attention" only when the structure is visually defensible, not
    # every time a loose continuation could be forced onto noisy candles.
    min_flag_quality: float = 0.72
    min_pennant_quality: float = 0.72
    # Triangle/wedge is still WIP and remains excluded from composite outputs by
    # default. These levers constrain the named geometry detector so it only
    # emits attention when a recent pivot envelope is visually defensible.
    triangle_window: int = 150
    min_triangle_bars: int = 18
    min_triangle_width_pct: float = 0.006
    max_triangle_width_pct: float = 0.22
    geometry_boundary_touch_tolerance_pct: float = 0.012
    min_triangle_prior_move_pct: float = 0.025
    min_geometry_containment_ratio: float = 0.68
    min_geometry_contraction_score: float = 0.16
    min_geometry_candidate_score: float = 5.0
    max_geometry_fit_error_atr: float = 2.0
    max_geometry_body_excursion_pct: float = 0.018
    # Boundary-respect is stricter than broad containment. Containment asks
    # whether price generally sits inside the pattern; boundary-respect asks
    # whether the selected rails actually behave like support/resistance rather
    # than cutting through bodies. This is especially important for the broad
    # experimental compression label.
    geometry_boundary_tolerance_pct: float = 0.005
    geometry_boundary_snap_max_pct: float = 0.006
    min_geometry_boundary_respect_ratio: float = 0.80
    max_compression_boundary_intrusion_pct: float = 0.007
    # Aggressive geometry rails can turn spike/crash candles into fake
    # triangles or wedges. Slopes are normalized by current price per candle.
    max_geometry_boundary_slope_pct_per_bar: float = 0.0024
    aggressive_geometry_slope_pct_per_bar: float = 0.0014
    aggressive_geometry_min_side_touches: int = 3
    min_compression_side_touches: int = 3
    min_compression_anchor_balance_score: float = 0.45
    min_compression_touch_balance_score: float = 0.42
    min_compression_contraction_score: float = 0.28
    min_geometry_side_switches: int = 2
    min_geometry_recent_touch_score: float = 0.24
    min_geometry_anchor_balance_score: float = 0.18
    min_geometry_touch_balance_score: float = 0.20
    geometry_max_last_anchor_age_window_mult: float = 0.38
    geometry_candidate_early_start_count: int = 2
    geometry_candidate_middle_start_count: int = 6
    geometry_candidate_recent_start_count: int = 4
    geometry_candidate_min_span_mult: float = 0.45
    geometry_candidate_min_span_bars: int = 2
    geometry_line_fit_tolerance_mult: float = 2.0
    geometry_envelope_eval_step: int = 2
    geometry_envelope_min_span_mult: float = 2.5
    geometry_envelope_start_count: int = 40
    geometry_envelope_pair_min_span_mult: float = 0.38
    geometry_envelope_touch_atr_mult: float = 0.95
    geometry_envelope_max_width_atr: float = 11.5
    geometry_envelope_max_side_pivots: int = 11
    geometry_envelope_max_pair_options: int = 7
    geometry_envelope_min_touch_span_ratio: float = 0.25
    geometry_envelope_merge_slope_tolerance_pct: float = 0.018
    geometry_envelope_proximity_atr_mult: float = 1.10
    geometry_envelope_proximity_grace_bars: int = 14
    geometry_envelope_proximity_ramp_bars: int = 3
    geometry_envelope_proximity_penalty_weight: float = 1.0
    geometry_pivot_dominance_bars: int = 24
    geometry_pivot_dominance_atr_mult: float = 0.16
    geometry_pivot_merge_bars: int = 10
    geometry_pivot_merge_atr_mult: float = 0.45
    geometry_line_touch_weight: float = 0.38
    geometry_line_fit_weight: float = 0.30
    geometry_line_span_weight: float = 0.18
    geometry_line_recency_weight: float = 0.14
    geometry_slot_duplicate_price_tolerance_pct: float = 0.010
    geometry_slot_duplicate_start_tolerance: float = 0.20
    geometry_body_invalidation_atr: float = 0.25
    # Experimental lower-timeframe normalization. Defaults are neutral. When
    # enabled, short-candle data can expand the triangle/wedge lookback so a
    # 1h chart can inspect a similar calendar structure to a 4h chart without
    # hardcoding timeframe names.
    geometry_window_time_scale_power: float = 0.0
    geometry_window_reference_seconds: float = 14400.0
    geometry_window_max_mult: float = 4.0
    geometry_timeframe_seconds: float = 0.0
    geometry_management_max_age_bars: int = 36
    geometry_output_slots: int = 1
    include_pattern_diagnostics: bool = False
    flat_boundary_slope_pct_per_bar: float = 0.00035
    include_compression_patterns: bool = False
    include_triangle_wedge_patterns: bool = False
    double_pattern_window: int = 80
    min_double_pattern_bars: int = 8
    max_double_pattern_bars: int = 64
    double_peak_tolerance_pct: float = 0.040
    min_double_neckline_depth_pct: float = 0.025
    min_double_prior_move_pct: float = 0.040
    min_double_first_pivot_move_pct: float = 0.030
    double_reaction_max_bars: int = 24
    # Hard rule for height/level comparisons: the decision tolerance must scale
    # with timeframe. Static percent tolerances below are ceilings, not the
    # primary same-level rule.
    peak_dynamic_body_window: int = 30
    peak_dynamic_scale_window: int = 30
    peak_premove_body_mult: float = 6.0
    peak_level_tolerance_body_mult: float = 2.25
    peak_level_tolerance_atr_mult: float = 0.35
    peak_level_tolerance_prominence_mult: float = 0.20
    peak_reaction_body_mult: float = 3.0
    peak_base_return_buffer_body_mult: float = 0.8
    peak_prior_impulse_min_bars: int = 3
    peak_prior_impulse_min_efficiency: float = 0.40
    # Double peaks are intentionally simple: require a meaningful move into P1,
    # a reaction away from P1, and a same-level P2 inside a bounded window. The
    # threshold is calibrated for "pay attention" pattern evidence, not final
    # trade certainty.
    min_double_quality: float = 0.72
    min_double_reaction_score: float = 0.12
    min_double_between_cleanliness_score: float = 0.38
    double_duplicate_overlap_pct: float = 0.70
    double_duplicate_neckline_tolerance_pct: float = 0.012
    include_double_reversal_patterns: bool = False
    head_shoulders_window: int = 100
    min_head_shoulders_bars: int = 12
    max_head_shoulders_bars: int = 90
    shoulder_tolerance_pct: float = 0.060
    min_head_prominence_pct: float = 0.025
    min_head_shoulders_neckline_depth_pct: float = 0.018
    min_head_shoulders_prior_move_pct: float = 0.055
    min_head_shoulders_quality: float = 0.78
    min_head_shoulders_shoulder_score: float = 0.45
    min_head_shoulders_time_balance_score: float = 0.45
    min_head_shoulders_neckline_score: float = 0.35
    min_head_shoulders_neckline_position_score: float = 0.25
    min_head_shoulders_head_position_score: float = 0.20
    min_head_shoulders_neckline_clearance_score: float = 0.22
    min_head_shoulders_neckline_body_respect_ratio: float = 0.70
    head_shoulders_max_neckline_slope_pct_per_bar: float = 0.0030
    min_head_shoulders_right_reaction_score: float = 0.10
    head_shoulders_setup_monitor_bars: int = 16
    head_shoulders_neckline_proximity_pct: float = 0.018
    # H&S can span many minor pivots. Limit the search for performance, but do
    # not only inspect the last few pivots or broad 1d/4h structures disappear.
    head_shoulders_max_candidate_pivots: int = 14
    include_head_shoulders_patterns: bool = False
    rectangle_window: int = 80
    min_rectangle_bars: int = 16
    rectangle_min_width_pct: float = 0.008
    rectangle_max_width_pct: float = 0.18
    rectangle_boundary_tolerance_pct: float = 0.006
    rectangle_management_buffer_pct: float = 0.002
    rectangle_min_side_touches: int = 3
    min_rectangle_containment_ratio: float = 0.80
    min_rectangle_quality: float = 0.88
    include_rectangle_patterns: bool = False
    triple_pattern_window: int = 110
    min_triple_pattern_bars: int = 18
    max_triple_pattern_bars: int = 96
    min_triple_spacing_bars: int = 5
    triple_max_candidate_pivots: int = 16
    triple_peak_tolerance_pct: float = 0.040
    min_triple_neckline_depth_pct: float = 0.025
    min_triple_prior_move_pct: float = 0.040
    min_triple_first_pivot_move_pct: float = 0.030
    triple_reaction_max_bars: int = 36
    min_triple_reaction_score: float = 0.12
    # Triple peaks use the same reversal premise and scoring threshold as
    # double peaks; the only structural difference is the third same-level
    # touch.
    min_triple_quality: float = 0.72
    include_triple_reversal_patterns: bool = False
    channel_pattern_window: int = 120
    broadening_pattern_window: int = 180
    min_channel_pattern_bars: int = 18
    channel_pattern_min_side_pivots: int = 3
    channel_pattern_min_width_pct: float = 0.010
    channel_pattern_max_width_pct: float = 0.24
    min_channel_pattern_slope_pct_per_bar: float = 0.00020
    channel_parallel_slope_tolerance_pct_per_bar: float = 0.00045
    max_channel_pattern_fit_error_pct: float = 0.16
    min_channel_pattern_containment_ratio: float = 0.58
    channel_pattern_boundary_tolerance_pct: float = 0.006
    channel_pattern_management_buffer_pct: float = 0.002
    min_channel_pattern_quality: float = 0.70
    include_channel_patterns: bool = False
    min_broadening_expansion_pct: float = 0.08
    min_broadening_slope_gap_pct_per_bar: float = 0.00015
    min_broadening_prior_move_pct: float = 0.0
    min_broadening_quality: float = 0.70
    include_broadening_patterns: bool = False
    pattern_lifecycle_mature_bars: int = 12
    pattern_lifecycle_stale_bars: int = 12
    # TODO_DELETE_CHECK: keep this default False. The old triangle/wedge path is
    # retained only for visual comparison while developing the replacement.
    include_unvalidated_geometric_patterns: bool = False
    # TODO_DELETE_CHECK: generic channel+sequence context was too broad and made
    # pat_setup_* behave like loose market structure rather than named patterns.
    # Keep disabled unless explicitly reviewing that scaffold; remove if named
    # patterns no longer need this broad compatibility path.
    include_channel_sequence_context: bool = False
    min_channel_score: float = 0.45
    min_sequence_score: float = 0.40
    entry_cooldown_bars: int = 8


def add_pattern_structure(
    dataframe: DataFrame,
    config: PatternStructureConfig | None = None,
    **overrides: object,
) -> DataFrame:
    """Append pattern context columns derived from the new indicator foundations."""

    cfg = _resolve_config(config, overrides)
    _validate_config(cfg)

    source = dataframe.copy()
    _validate_ohlcv_dataframe(source)
    frame = _with_foundation_pivots(source, cfg)
    _validate_dataframe(frame, cfg)
    sequence = _pivot_sequence_columns(frame, cfg)
    channel = _channel_columns(frame, cfg)
    prototypes = {
        **_flag_pennant_columns(frame, sequence, channel, cfg),
        **_triangle_wedge_columns(frame, sequence, cfg),
        **_rectangle_range_columns(frame, cfg),
        **_channel_pattern_columns(frame, cfg),
        **_double_reversal_columns(frame, cfg),
        **_triple_reversal_columns(frame, cfg),
        **_head_shoulders_columns(frame, cfg),
    }
    context = _context_columns(sequence, channel, prototypes, frame.index, cfg)

    p = cfg.output_prefix
    existing = [col for col in source.columns if str(col).startswith(f"{p}_")]
    clean = source.drop(columns=existing).copy() if existing else source.copy()
    features = pd.DataFrame({**sequence, **channel, **prototypes, **context}, index=frame.index)
    return pd.concat([clean, features], axis=1)


def _pivot_sequence_columns(frame: DataFrame, cfg: PatternStructureConfig) -> dict[str, Series]:
    p = cfg.output_prefix
    pp = cfg.pivot_prefix
    window = int(cfg.sequence_window)
    high_pivot = _num(frame[f"{pp}_pivot_high"])
    low_pivot = _num(frame[f"{pp}_pivot_low"])

    last_high = high_pivot.ffill()
    last_low = low_pivot.ffill()
    prev_high = high_pivot.dropna().shift(1).reindex(frame.index).ffill()
    prev_low = low_pivot.dropna().shift(1).reindex(frame.index).ffill()

    high_event = high_pivot.notna()
    low_event = low_pivot.notna()
    higher_high = high_event & high_pivot.gt(prev_high)
    lower_high = high_event & high_pivot.lt(prev_high)
    higher_low = low_event & low_pivot.gt(prev_low)
    lower_low = low_event & low_pivot.lt(prev_low)

    hh_count = _rolling_count(higher_high, window)
    lh_count = _rolling_count(lower_high, window)
    hl_count = _rolling_count(higher_low, window)
    ll_count = _rolling_count(lower_low, window)
    pivot_event_count = _rolling_count(high_event | low_event, window)
    high_count = _rolling_count(high_event, window)
    low_count = _rolling_count(low_event, window)
    bar_index = _bar_index(frame)
    high_index = _num(frame.get(f"{pp}_pivot_high_index", bar_index.where(high_event)))
    low_index = _num(frame.get(f"{pp}_pivot_low_index", bar_index.where(low_event)))
    close = _num(frame["close"]).replace(0.0, np.nan)
    high_slope = _rolling_pivot_slope(high_pivot, high_index, high_event, window)
    low_slope = _rolling_pivot_slope(low_pivot, low_index, low_event, window)
    high_slope_pct = (high_slope / close.abs()).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    low_slope_pct = (low_slope / close.abs()).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    up_score = _clip01((hh_count + hl_count) / max(float(window) / 4.0, 1.0))
    down_score = _clip01((lh_count + ll_count) / max(float(window) / 4.0, 1.0))
    converging_score = _clip01((lh_count + hl_count) / max(float(window) / 5.0, 1.0))
    expanding_score = _clip01((hh_count + ll_count) / max(float(window) / 5.0, 1.0))
    structure_score = pd.concat([up_score, down_score, converging_score, expanding_score], axis=1).max(axis=1)
    sequence_bias = pd.Series(
        np.select(
            [up_score.gt(down_score + 0.10), down_score.gt(up_score + 0.10)],
            [1, -1],
            default=0,
        ),
        index=frame.index,
        dtype="int8",
    )

    return {
        f"{p}_pivot_last_high": last_high,
        f"{p}_pivot_last_low": last_low,
        f"{p}_pivot_higher_high": higher_high.fillna(False),
        f"{p}_pivot_lower_high": lower_high.fillna(False),
        f"{p}_pivot_higher_low": higher_low.fillna(False),
        f"{p}_pivot_lower_low": lower_low.fillna(False),
        f"{p}_sequence_higher_high_count": hh_count,
        f"{p}_sequence_lower_high_count": lh_count,
        f"{p}_sequence_higher_low_count": hl_count,
        f"{p}_sequence_lower_low_count": ll_count,
        f"{p}_sequence_up_score": up_score,
        f"{p}_sequence_down_score": down_score,
        f"{p}_sequence_converging_score": converging_score,
        f"{p}_sequence_expanding_score": expanding_score,
        f"{p}_sequence_structure_score": structure_score,
        f"{p}_sequence_bias": sequence_bias,
        f"{p}_sequence_event_count": pivot_event_count,
        f"{p}_sequence_high_count": high_count,
        f"{p}_sequence_low_count": low_count,
        f"{p}_sequence_high_slope_pct": high_slope_pct,
        f"{p}_sequence_low_slope_pct": low_slope_pct,
    }


def _channel_columns(frame: DataFrame, cfg: PatternStructureConfig) -> dict[str, Series]:
    p = cfg.output_prefix
    cp = cfg.channel_prefix
    label = cfg.channel_label
    rank = int(cfg.channel_rank)
    index = frame.index
    required = _channel_required_columns(cfg)
    if any(column not in frame.columns for column in required):
        false = pd.Series(False, index=index)
        zero = pd.Series(0.0, index=index, dtype="float64")
        nan = pd.Series(np.nan, index=index, dtype="float64")
        return {
            f"{p}_channel_active": false,
            f"{p}_channel_near_data": false,
            f"{p}_channel_upper_recent_touch_count": zero,
            f"{p}_channel_lower_recent_touch_count": zero,
            f"{p}_channel_score": zero,
            f"{p}_channel_upper": nan,
            f"{p}_channel_lower": nan,
            f"{p}_channel_mid": nan,
            f"{p}_channel_position": nan,
            f"{p}_channel_width_pct": nan,
            f"{p}_channel_shape": nan,
            f"{p}_channel_parallel_setup": false,
            f"{p}_channel_converging_setup": false,
            f"{p}_channel_diverging_setup": false,
            f"{p}_channel_near_upper": false,
            f"{p}_channel_near_lower": false,
            f"{p}_channel_inside": false,
            f"{p}_channel_breakout_long": false,
            f"{p}_channel_breakdown_short": false,
        }

    def channel(metric: str) -> Series:
        return _num(frame[f"{cp}_{label}_{metric}_rank{rank}"])

    score = channel("score")
    active = _bool(frame[f"{cp}_{label}_active_rank{rank}"])
    breakout_up = _bool(frame[f"{cp}_{label}_breakout_up_rank{rank}"])
    breakdown_down = _bool(frame[f"{cp}_{label}_breakdown_down_rank{rank}"])
    near_upper = _bool(frame[f"{cp}_{label}_near_upper_rank{rank}"])
    near_lower = _bool(frame[f"{cp}_{label}_near_lower_rank{rank}"])
    inside = _bool(frame[f"{cp}_{label}_inside_rank{rank}"])
    near_data_col = f"{cp}_{label}_near_data_rank{rank}"
    upper_touch_col = f"{cp}_{label}_upper_recent_touch_count_rank{rank}"
    lower_touch_col = f"{cp}_{label}_lower_recent_touch_count_rank{rank}"
    near_data = _bool(frame[near_data_col]) if near_data_col in frame.columns else active
    upper_touch_count = _num(frame[upper_touch_col]) if upper_touch_col in frame.columns else pd.Series(np.nan, index=index)
    lower_touch_count = _num(frame[lower_touch_col]) if lower_touch_col in frame.columns else pd.Series(np.nan, index=index)
    valid = active & score.ge(float(cfg.min_channel_score))
    shape = channel("shape")

    return {
        f"{p}_channel_active": valid.fillna(False),
        f"{p}_channel_near_data": (valid & near_data).fillna(False),
        f"{p}_channel_upper_recent_touch_count": upper_touch_count.where(valid, 0.0),
        f"{p}_channel_lower_recent_touch_count": lower_touch_count.where(valid, 0.0),
        f"{p}_channel_score": score.where(valid, 0.0),
        f"{p}_channel_upper": channel("upper").where(valid),
        f"{p}_channel_lower": channel("lower").where(valid),
        f"{p}_channel_mid": channel("mid").where(valid),
        f"{p}_channel_position": channel("position").where(valid),
        f"{p}_channel_width_pct": channel("width_pct").where(valid),
        f"{p}_channel_shape": shape.where(valid),
        f"{p}_channel_parallel_setup": (valid & shape.eq(0.0)).fillna(False),
        f"{p}_channel_converging_setup": (valid & shape.eq(1.0)).fillna(False),
        f"{p}_channel_diverging_setup": (valid & shape.eq(-1.0)).fillna(False),
        f"{p}_channel_near_upper": (valid & near_upper).fillna(False),
        f"{p}_channel_near_lower": (valid & near_lower).fillna(False),
        f"{p}_channel_inside": (valid & inside).fillna(False),
        f"{p}_channel_breakout_long": (valid & breakout_up).fillna(False),
        f"{p}_channel_breakdown_short": (valid & breakdown_down).fillna(False),
    }


def _context_columns(
    sequence: dict[str, Series],
    channel: dict[str, Series],
    prototypes: dict[str, Series],
    index: pd.Index,
    cfg: PatternStructureConfig,
) -> dict[str, Series]:
    p = cfg.output_prefix
    sequence_up = sequence[f"{p}_sequence_up_score"]
    sequence_down = sequence[f"{p}_sequence_down_score"]
    sequence_converging = sequence[f"{p}_sequence_converging_score"]
    sequence_bias = sequence[f"{p}_sequence_bias"]
    channel_active = channel[f"{p}_channel_active"]
    channel_parallel = channel[f"{p}_channel_parallel_setup"]
    channel_converging = channel[f"{p}_channel_converging_setup"]
    near_lower = channel[f"{p}_channel_near_lower"]
    near_upper = channel[f"{p}_channel_near_upper"]
    flag_setup_long = prototypes[f"{p}_flag_setup_long"]
    flag_setup_short = prototypes[f"{p}_flag_setup_short"]
    pennant_setup_long = prototypes[f"{p}_pennant_setup_long"]
    pennant_setup_short = prototypes[f"{p}_pennant_setup_short"]
    rectangle_setup = prototypes[f"{p}_rectangle_setup"]
    rectangle_go_long = prototypes[f"{p}_rectangle_go_long"]
    rectangle_go_short = prototypes[f"{p}_rectangle_go_short"]
    ascending_channel = prototypes[f"{p}_ascending_channel_setup"]
    descending_channel = prototypes[f"{p}_descending_channel_setup"]
    ascending_channel_go_long = prototypes[f"{p}_ascending_channel_go_long"]
    ascending_channel_go_short = prototypes[f"{p}_ascending_channel_go_short"]
    descending_channel_go_long = prototypes[f"{p}_descending_channel_go_long"]
    descending_channel_go_short = prototypes[f"{p}_descending_channel_go_short"]
    broadening_top = prototypes[f"{p}_broadening_top_setup_short"]
    broadening_bottom = prototypes[f"{p}_broadening_bottom_setup_long"]
    broadening_top_go_short = prototypes[f"{p}_broadening_top_go_short"]
    broadening_bottom_go_long = prototypes[f"{p}_broadening_bottom_go_long"]
    double_setup_long = prototypes[f"{p}_double_bottom_setup_long"]
    double_setup_short = prototypes[f"{p}_double_top_setup_short"]
    double_state_long = _num(prototypes[f"{p}_double_bottom_state"])
    double_state_short = _num(prototypes[f"{p}_double_top_state"])
    double_current_long = double_state_long.gt(0.0)
    double_current_short = double_state_short.gt(0.0)
    double_confirmed_long = double_state_long.eq(2.0)
    double_confirmed_short = double_state_short.eq(2.0)
    double_clean_long = prototypes[f"{p}_double_bottom_clean_long"]
    double_clean_short = prototypes[f"{p}_double_top_clean_short"]
    triple_setup_long = prototypes[f"{p}_triple_bottom_setup_long"]
    triple_setup_short = prototypes[f"{p}_triple_top_setup_short"]
    triple_state_long = _num(prototypes[f"{p}_triple_bottom_state"])
    triple_state_short = _num(prototypes[f"{p}_triple_top_state"])
    triple_current_long = triple_state_long.gt(0.0)
    triple_current_short = triple_state_short.gt(0.0)
    triple_confirmed_long = triple_state_long.eq(2.0)
    triple_confirmed_short = triple_state_short.eq(2.0)
    head_shoulders_setup_long = prototypes[f"{p}_inverse_head_shoulders_setup_long"]
    head_shoulders_setup_short = prototypes[f"{p}_head_shoulders_setup_short"]
    geometric_setup_long = pd.Series(False, index=index)
    geometric_setup_short = pd.Series(False, index=index)
    if bool(cfg.include_double_reversal_patterns):
        reversal_setup_long = double_current_long
        reversal_setup_short = double_current_short
        reversal_entry_long = double_confirmed_long
        reversal_entry_short = double_confirmed_short
    else:
        reversal_setup_long = pd.Series(False, index=index)
        reversal_setup_short = pd.Series(False, index=index)
        reversal_entry_long = pd.Series(False, index=index)
        reversal_entry_short = pd.Series(False, index=index)
    if bool(cfg.include_rectangle_patterns):
        rectangle_setup_long = rectangle_go_long
        rectangle_setup_short = rectangle_go_short
    else:
        rectangle_setup_long = pd.Series(False, index=index)
        rectangle_setup_short = pd.Series(False, index=index)
    if bool(cfg.include_channel_patterns):
        channel_pattern_long = ascending_channel_go_long | descending_channel_go_long
        channel_pattern_short = ascending_channel_go_short | descending_channel_go_short
    else:
        channel_pattern_long = pd.Series(False, index=index)
        channel_pattern_short = pd.Series(False, index=index)
    if bool(cfg.include_broadening_patterns):
        broadening_pattern_long = broadening_bottom
        broadening_pattern_short = broadening_top
    else:
        broadening_pattern_long = pd.Series(False, index=index)
        broadening_pattern_short = pd.Series(False, index=index)
    if bool(cfg.include_triple_reversal_patterns):
        triple_reversal_long = triple_current_long
        triple_reversal_short = triple_current_short
        triple_reversal_entry_long = triple_confirmed_long
        triple_reversal_entry_short = triple_confirmed_short
    else:
        triple_reversal_long = pd.Series(False, index=index)
        triple_reversal_short = pd.Series(False, index=index)
        triple_reversal_entry_long = pd.Series(False, index=index)
        triple_reversal_entry_short = pd.Series(False, index=index)
    if bool(cfg.include_head_shoulders_patterns):
        head_shoulders_long = head_shoulders_setup_long
        head_shoulders_short = head_shoulders_setup_short
    else:
        head_shoulders_long = pd.Series(False, index=index)
        head_shoulders_short = pd.Series(False, index=index)
    # TODO_DELETE_CHECK: this broad scaffold is redundant once pattern-specific
    # setup columns are available. It failed as a primary setup source because
    # sequence+channel evidence can describe ordinary structure, not a named
    # chart pattern.
    if bool(cfg.include_channel_sequence_context):
        channel_sequence_long = (
            sequence_up.ge(float(cfg.min_sequence_score) * 0.65)
            & channel_active
            & (channel_parallel | channel_converging | near_lower)
        )
        channel_sequence_short = (
            sequence_down.ge(float(cfg.min_sequence_score) * 0.65)
            & channel_active
            & (channel_parallel | channel_converging | near_upper)
        )
    else:
        channel_sequence_long = pd.Series(False, index=index)
        channel_sequence_short = pd.Series(False, index=index)

    long_setup = (
        flag_setup_long
        | pennant_setup_long
        | geometric_setup_long
        | rectangle_setup_long
        | channel_pattern_long
        | broadening_pattern_long
        | reversal_setup_long
        | triple_reversal_long
        | head_shoulders_long
        | channel_sequence_long
    )
    short_setup = (
        flag_setup_short
        | pennant_setup_short
        | geometric_setup_short
        | rectangle_setup_short
        | channel_pattern_short
        | broadening_pattern_short
        | reversal_setup_short
        | triple_reversal_short
        | head_shoulders_short
        | channel_sequence_short
    )
    contraction_setup = channel_active & channel_converging & sequence_converging.ge(float(cfg.min_sequence_score))
    long_entry_source = (
        flag_setup_long
        | pennant_setup_long
        | geometric_setup_long
        | rectangle_setup_long
        | channel_pattern_long
        | broadening_pattern_long
        | reversal_entry_long
        | triple_reversal_entry_long
        | head_shoulders_long
        | channel_sequence_long
    )
    short_entry_source = (
        flag_setup_short
        | pennant_setup_short
        | geometric_setup_short
        | rectangle_setup_short
        | channel_pattern_short
        | broadening_pattern_short
        | reversal_entry_short
        | triple_reversal_entry_short
        | head_shoulders_short
        | channel_sequence_short
    )

    market_context = pd.Series(
        np.select(
            [
                flag_setup_long | pennant_setup_long,
                flag_setup_short | pennant_setup_short,
                geometric_setup_long,
                geometric_setup_short,
                rectangle_setup_long,
                rectangle_setup_short,
                channel_pattern_long,
                channel_pattern_short,
                broadening_pattern_long,
                broadening_pattern_short,
                reversal_setup_long,
                reversal_setup_short,
                triple_reversal_long,
                triple_reversal_short,
                head_shoulders_long,
                head_shoulders_short,
                long_setup | sequence_bias.gt(0),
                short_setup | sequence_bias.lt(0),
            ],
            [2, -2, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
            default=0,
        ),
        index=index,
        dtype="int8",
    )
    entry_long = _dedupe_events(
        long_entry_source
        | (near_lower & long_entry_source),
        cfg.entry_cooldown_bars,
    )
    entry_short = _dedupe_events(
        short_entry_source
        | (near_upper & short_entry_source),
        cfg.entry_cooldown_bars,
    )
    long_setup_event = (
        _dedupe_events(flag_setup_long, cfg.entry_cooldown_bars)
        | _dedupe_events(pennant_setup_long, cfg.entry_cooldown_bars)
        | _dedupe_events(rectangle_setup_long, cfg.entry_cooldown_bars)
        | _dedupe_events(channel_pattern_long, cfg.entry_cooldown_bars)
        | _dedupe_events(broadening_pattern_long, cfg.entry_cooldown_bars)
        | _dedupe_events(reversal_setup_long & double_setup_long, cfg.entry_cooldown_bars)
        | _dedupe_events(triple_reversal_long, cfg.entry_cooldown_bars)
        | _dedupe_events(head_shoulders_long, cfg.entry_cooldown_bars)
        | _dedupe_events(near_lower & long_setup, cfg.entry_cooldown_bars)
    )
    short_setup_event = (
        _dedupe_events(flag_setup_short, cfg.entry_cooldown_bars)
        | _dedupe_events(pennant_setup_short, cfg.entry_cooldown_bars)
        | _dedupe_events(rectangle_setup_short, cfg.entry_cooldown_bars)
        | _dedupe_events(channel_pattern_short, cfg.entry_cooldown_bars)
        | _dedupe_events(broadening_pattern_short, cfg.entry_cooldown_bars)
        | _dedupe_events(reversal_setup_short & double_setup_short, cfg.entry_cooldown_bars)
        | _dedupe_events(triple_reversal_short, cfg.entry_cooldown_bars)
        | _dedupe_events(head_shoulders_short, cfg.entry_cooldown_bars)
        | _dedupe_events(near_upper & short_setup, cfg.entry_cooldown_bars)
    )
    return {
        f"{p}_foundation_ready": sequence[f"{p}_sequence_event_count"].ge(2.0).fillna(False),
        f"{p}_setup_long": long_setup.fillna(False),
        f"{p}_setup_short": short_setup.fillna(False),
        f"{p}_setup_contraction": contraction_setup.fillna(False),
        f"{p}_entry_channel_lower_reaction_long": _dedupe_events(near_lower & long_entry_source, cfg.entry_cooldown_bars),
        f"{p}_entry_channel_upper_reaction_short": _dedupe_events(near_upper & short_entry_source, cfg.entry_cooldown_bars),
        f"{p}_entry_flag_setup_long": _dedupe_events(flag_setup_long, cfg.entry_cooldown_bars),
        f"{p}_entry_flag_setup_short": _dedupe_events(flag_setup_short, cfg.entry_cooldown_bars),
        f"{p}_entry_pennant_setup_long": _dedupe_events(pennant_setup_long, cfg.entry_cooldown_bars),
        f"{p}_entry_pennant_setup_short": _dedupe_events(pennant_setup_short, cfg.entry_cooldown_bars),
        f"{p}_entry_rectangle_go_long": _dedupe_events(rectangle_setup_long, cfg.entry_cooldown_bars),
        f"{p}_entry_rectangle_go_short": _dedupe_events(rectangle_setup_short, cfg.entry_cooldown_bars),
        f"{p}_entry_ascending_channel_go_long": _dedupe_events(
            channel_pattern_long & ascending_channel_go_long, cfg.entry_cooldown_bars
        ),
        f"{p}_entry_ascending_channel_go_short": _dedupe_events(
            channel_pattern_short & ascending_channel_go_short, cfg.entry_cooldown_bars
        ),
        f"{p}_entry_descending_channel_go_long": _dedupe_events(
            channel_pattern_long & descending_channel_go_long, cfg.entry_cooldown_bars
        ),
        f"{p}_entry_descending_channel_go_short": _dedupe_events(
            channel_pattern_short & descending_channel_go_short, cfg.entry_cooldown_bars
        ),
        f"{p}_entry_broadening_bottom_go_long": _dedupe_events(
            broadening_pattern_long & broadening_bottom_go_long, cfg.entry_cooldown_bars
        ),
        f"{p}_entry_broadening_top_go_short": _dedupe_events(
            broadening_pattern_short & broadening_top_go_short, cfg.entry_cooldown_bars
        ),
        f"{p}_entry_double_bottom_setup_long": _dedupe_events(reversal_setup_long & double_setup_long, cfg.entry_cooldown_bars),
        f"{p}_entry_double_top_setup_short": _dedupe_events(reversal_setup_short & double_setup_short, cfg.entry_cooldown_bars),
        f"{p}_entry_double_bottom_confirmed_long": _dedupe_events(reversal_entry_long, cfg.entry_cooldown_bars),
        f"{p}_entry_double_top_confirmed_short": _dedupe_events(reversal_entry_short, cfg.entry_cooldown_bars),
        f"{p}_entry_double_bottom_clean_long": _dedupe_events(reversal_setup_long & double_clean_long, cfg.entry_cooldown_bars),
        f"{p}_entry_double_top_clean_short": _dedupe_events(reversal_setup_short & double_clean_short, cfg.entry_cooldown_bars),
        f"{p}_entry_triple_bottom_setup_long": _dedupe_events(triple_reversal_long, cfg.entry_cooldown_bars),
        f"{p}_entry_triple_top_setup_short": _dedupe_events(triple_reversal_short, cfg.entry_cooldown_bars),
        f"{p}_entry_triple_bottom_confirmed_long": _dedupe_events(triple_reversal_entry_long, cfg.entry_cooldown_bars),
        f"{p}_entry_triple_top_confirmed_short": _dedupe_events(triple_reversal_entry_short, cfg.entry_cooldown_bars),
        f"{p}_entry_inverse_head_shoulders_setup_long": _dedupe_events(head_shoulders_long, cfg.entry_cooldown_bars),
        f"{p}_entry_head_shoulders_setup_short": _dedupe_events(head_shoulders_short, cfg.entry_cooldown_bars),
        f"{p}_attention_flag_long": _dedupe_events(flag_setup_long, cfg.entry_cooldown_bars),
        f"{p}_attention_flag_short": _dedupe_events(flag_setup_short, cfg.entry_cooldown_bars),
        f"{p}_attention_pennant_long": _dedupe_events(pennant_setup_long, cfg.entry_cooldown_bars),
        f"{p}_attention_pennant_short": _dedupe_events(pennant_setup_short, cfg.entry_cooldown_bars),
        f"{p}_attention_rectangle_long": _dedupe_events(rectangle_setup_long, cfg.entry_cooldown_bars),
        f"{p}_attention_rectangle_short": _dedupe_events(rectangle_setup_short, cfg.entry_cooldown_bars),
        f"{p}_attention_ascending_channel_long": _dedupe_events(
            channel_pattern_long & ascending_channel_go_long, cfg.entry_cooldown_bars
        ),
        f"{p}_attention_ascending_channel_short": _dedupe_events(
            channel_pattern_short & ascending_channel_go_short, cfg.entry_cooldown_bars
        ),
        f"{p}_attention_descending_channel_long": _dedupe_events(
            channel_pattern_long & descending_channel_go_long, cfg.entry_cooldown_bars
        ),
        f"{p}_attention_descending_channel_short": _dedupe_events(
            channel_pattern_short & descending_channel_go_short, cfg.entry_cooldown_bars
        ),
        f"{p}_attention_broadening_bottom_long": _dedupe_events(
            broadening_pattern_long, cfg.entry_cooldown_bars
        ),
        f"{p}_attention_broadening_top_short": _dedupe_events(
            broadening_pattern_short, cfg.entry_cooldown_bars
        ),
        f"{p}_attention_double_bottom_long": _dedupe_events(reversal_setup_long & double_setup_long, cfg.entry_cooldown_bars),
        f"{p}_attention_double_top_short": _dedupe_events(reversal_setup_short & double_setup_short, cfg.entry_cooldown_bars),
        f"{p}_attention_double_bottom_confirmed_long": _dedupe_events(reversal_entry_long, cfg.entry_cooldown_bars),
        f"{p}_attention_double_top_confirmed_short": _dedupe_events(reversal_entry_short, cfg.entry_cooldown_bars),
        f"{p}_attention_double_bottom_clean_long": _dedupe_events(reversal_setup_long & double_clean_long, cfg.entry_cooldown_bars),
        f"{p}_attention_double_top_clean_short": _dedupe_events(reversal_setup_short & double_clean_short, cfg.entry_cooldown_bars),
        f"{p}_attention_triple_bottom_long": _dedupe_events(triple_reversal_long, cfg.entry_cooldown_bars),
        f"{p}_attention_triple_top_short": _dedupe_events(triple_reversal_short, cfg.entry_cooldown_bars),
        f"{p}_attention_triple_bottom_confirmed_long": _dedupe_events(triple_reversal_entry_long, cfg.entry_cooldown_bars),
        f"{p}_attention_triple_top_confirmed_short": _dedupe_events(triple_reversal_entry_short, cfg.entry_cooldown_bars),
        f"{p}_attention_inverse_head_shoulders_long": _dedupe_events(head_shoulders_long, cfg.entry_cooldown_bars),
        f"{p}_attention_head_shoulders_short": _dedupe_events(head_shoulders_short, cfg.entry_cooldown_bars),
        f"{p}_long_setup": long_setup_event.fillna(False),
        f"{p}_short_setup": short_setup_event.fillna(False),
        f"{p}_attention_long": entry_long.fillna(False),
        f"{p}_attention_short": entry_short.fillna(False),
        f"{p}_suggested_entry_long": entry_long.fillna(False),
        f"{p}_suggested_entry_short": entry_short.fillna(False),
        f"{p}_market_context": market_context,
    }















































































def _resolve_config(config: PatternStructureConfig | None, overrides: dict[str, object]) -> PatternStructureConfig:
    cfg = config or PatternStructureConfig()
    valid = {field.name for field in fields(PatternStructureConfig)}
    unknown = sorted(key for key, value in overrides.items() if value is not None and key not in valid)
    if unknown:
        raise TypeError(f"Unknown PatternStructureConfig override(s): {', '.join(unknown)}")
    clean = {key: value for key, value in overrides.items() if value is not None}
    return replace(cfg, **clean)


def _validate_config(cfg: PatternStructureConfig) -> None:
    if not cfg.output_prefix or not cfg.pivot_prefix or not cfg.channel_prefix:
        raise ValueError("output_prefix, pivot_prefix, and channel_prefix must be set")
    if not cfg.channel_label:
        raise ValueError("channel_label must be set")
    if int(cfg.channel_rank) < 0:
        raise ValueError("channel_rank must be non-negative")
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
    if int(cfg.sequence_window) < 4:
        raise ValueError("sequence_window must be at least 4")
    if int(cfg.impulse_window) < 4:
        raise ValueError("impulse_window must be at least 4")
    if int(cfg.min_impulse_bars) < 1:
        raise ValueError("min_impulse_bars must be at least 1")
    if float(cfg.min_impulse_pct) < 0.0:
        raise ValueError("min_impulse_pct must be non-negative")
    if not 0.0 <= float(cfg.min_impulse_efficiency) <= 1.0:
        raise ValueError("min_impulse_efficiency must be between 0 and 1")
    if float(cfg.min_impulse_volume_ratio) <= 0.0:
        raise ValueError("min_impulse_volume_ratio must be positive")
    if float(cfg.impulse_extreme_tolerance_pct) < 0.0:
        raise ValueError("impulse_extreme_tolerance_pct must be non-negative")
    if float(cfg.impulse_dominance_mult) < 1.0:
        raise ValueError("impulse_dominance_mult must be at least 1")
    if not 0.0 <= float(cfg.min_impulse_dominance_score) <= 1.0:
        raise ValueError("min_impulse_dominance_score must be between 0 and 1")
    if not 0.0 <= float(cfg.min_impulse_break_score) <= 1.0:
        raise ValueError("min_impulse_break_score must be between 0 and 1")
    if float(cfg.max_consolidation_drift_pct_per_bar) < 0.0:
        raise ValueError("max_consolidation_drift_pct_per_bar must be non-negative")
    if int(cfg.max_impulse_extreme_age_bars) < 1:
        raise ValueError("max_impulse_extreme_age_bars must be at least 1")
    if int(cfg.pattern_pivot_strength) < 1:
        raise ValueError("pattern_pivot_strength must be at least 1")
    if int(cfg.min_pattern_bars) < 1:
        raise ValueError("min_pattern_bars must be at least 1")
    if not 0.0 <= float(cfg.min_setup_retrace_pct) < float(cfg.max_setup_retrace_pct):
        raise ValueError("min_setup_retrace_pct must be lower than max_setup_retrace_pct")
    if float(cfg.max_setup_retrace_pct) > 1.5:
        raise ValueError("max_setup_retrace_pct should be expressed as an impulse fraction")
    if float(cfg.max_setup_range_pct) <= 0.0:
        raise ValueError("max_setup_range_pct must be positive")
    if float(cfg.max_setup_to_impulse_range_mult) <= 0.0:
        raise ValueError("max_setup_to_impulse_range_mult must be positive")
    if float(cfg.max_pattern_breakout_pct) < 0.0:
        raise ValueError("max_pattern_breakout_pct must be non-negative")
    if float(cfg.max_pattern_boundary_excursion_pct) < 0.0:
        raise ValueError("max_pattern_boundary_excursion_pct must be non-negative")
    if float(cfg.pattern_side_dominance_mult) < 1.0:
        raise ValueError("pattern_side_dominance_mult must be at least 1")
    if int(cfg.min_pattern_side_pivots) < 1:
        raise ValueError("min_pattern_side_pivots must be at least 1")
    if float(cfg.min_boundary_slope_pct_per_bar) < 0.0:
        raise ValueError("min_boundary_slope_pct_per_bar must be non-negative")
    if float(cfg.max_flag_counter_slope_pct_per_bar) < 0.0:
        raise ValueError("max_flag_counter_slope_pct_per_bar must be non-negative")
    if float(cfg.parallel_slope_tolerance_pct_per_bar) <= 0.0:
        raise ValueError("parallel_slope_tolerance_pct_per_bar must be positive")
    if not 0.0 <= float(cfg.min_continuation_pole_score) <= 1.0:
        raise ValueError("min_continuation_pole_score must be between 0 and 1")
    if not 0.0 <= float(cfg.min_continuation_retrace_score) <= 1.0:
        raise ValueError("min_continuation_retrace_score must be between 0 and 1")
    if not 0.0 <= float(cfg.min_continuation_containment_score) <= 1.0:
        raise ValueError("min_continuation_containment_score must be between 0 and 1")
    if not 0.0 <= float(cfg.min_continuation_terminal_score) <= 1.0:
        raise ValueError("min_continuation_terminal_score must be between 0 and 1")
    if not 0.0 <= float(cfg.min_continuation_boundary_touch_score) <= 1.0:
        raise ValueError("min_continuation_boundary_touch_score must be between 0 and 1")
    if not 0.0 <= float(cfg.min_continuation_boundary_span_score) <= 1.0:
        raise ValueError("min_continuation_boundary_span_score must be between 0 and 1")
    if not 0.0 <= float(cfg.min_flag_quality) <= 1.0:
        raise ValueError("min_flag_quality must be between 0 and 1")
    if not 0.0 <= float(cfg.min_pennant_quality) <= 1.0:
        raise ValueError("min_pennant_quality must be between 0 and 1")
    if int(cfg.triangle_window) < 8:
        raise ValueError("triangle_window must be at least 8")
    if int(cfg.min_triangle_bars) < 4:
        raise ValueError("min_triangle_bars must be at least 4")
    if int(cfg.min_triangle_bars) >= int(cfg.triangle_window):
        raise ValueError("min_triangle_bars must be lower than triangle_window")
    if not 0.0 < float(cfg.min_triangle_width_pct) < float(cfg.max_triangle_width_pct):
        raise ValueError("min_triangle_width_pct must be lower than max_triangle_width_pct")
    if float(cfg.geometry_boundary_touch_tolerance_pct) < 0.0:
        raise ValueError("geometry_boundary_touch_tolerance_pct must be non-negative")
    if float(cfg.min_triangle_prior_move_pct) < 0.0:
        raise ValueError("min_triangle_prior_move_pct must be non-negative")
    if not 0.0 <= float(cfg.min_geometry_containment_ratio) <= 1.0:
        raise ValueError("min_geometry_containment_ratio must be between 0 and 1")
    if not 0.0 <= float(cfg.min_geometry_contraction_score) <= 1.0:
        raise ValueError("min_geometry_contraction_score must be between 0 and 1")
    if float(cfg.min_geometry_candidate_score) < 0.0:
        raise ValueError("min_geometry_candidate_score must be non-negative")
    if float(cfg.max_geometry_fit_error_atr) <= 0.0:
        raise ValueError("max_geometry_fit_error_atr must be positive")
    if float(cfg.max_geometry_body_excursion_pct) < 0.0:
        raise ValueError("max_geometry_body_excursion_pct must be non-negative")
    if float(cfg.geometry_boundary_tolerance_pct) < 0.0:
        raise ValueError("geometry_boundary_tolerance_pct must be non-negative")
    if float(cfg.geometry_boundary_snap_max_pct) < 0.0:
        raise ValueError("geometry_boundary_snap_max_pct must be non-negative")
    if not 0.0 <= float(cfg.min_geometry_boundary_respect_ratio) <= 1.0:
        raise ValueError("min_geometry_boundary_respect_ratio must be between 0 and 1")
    if float(cfg.max_compression_boundary_intrusion_pct) < 0.0:
        raise ValueError("max_compression_boundary_intrusion_pct must be non-negative")
    if float(cfg.max_geometry_boundary_slope_pct_per_bar) <= 0.0:
        raise ValueError("max_geometry_boundary_slope_pct_per_bar must be positive")
    if float(cfg.aggressive_geometry_slope_pct_per_bar) < 0.0:
        raise ValueError("aggressive_geometry_slope_pct_per_bar must be non-negative")
    if int(cfg.aggressive_geometry_min_side_touches) < 2:
        raise ValueError("aggressive_geometry_min_side_touches must be at least 2")
    if int(cfg.min_compression_side_touches) < 2:
        raise ValueError("min_compression_side_touches must be at least 2")
    if not 0.0 <= float(cfg.min_compression_anchor_balance_score) <= 1.0:
        raise ValueError("min_compression_anchor_balance_score must be between 0 and 1")
    if not 0.0 <= float(cfg.min_compression_touch_balance_score) <= 1.0:
        raise ValueError("min_compression_touch_balance_score must be between 0 and 1")
    if not 0.0 <= float(cfg.min_compression_contraction_score) <= 1.0:
        raise ValueError("min_compression_contraction_score must be between 0 and 1")
    if int(cfg.min_geometry_side_switches) < 0:
        raise ValueError("min_geometry_side_switches must be non-negative")
    if not 0.0 <= float(cfg.min_geometry_recent_touch_score) <= 1.0:
        raise ValueError("min_geometry_recent_touch_score must be between 0 and 1")
    if not 0.0 <= float(cfg.min_geometry_anchor_balance_score) <= 1.0:
        raise ValueError("min_geometry_anchor_balance_score must be between 0 and 1")
    if not 0.0 <= float(cfg.min_geometry_touch_balance_score) <= 1.0:
        raise ValueError("min_geometry_touch_balance_score must be between 0 and 1")
    if float(cfg.geometry_max_last_anchor_age_window_mult) < 0.0:
        raise ValueError("geometry_max_last_anchor_age_window_mult must be non-negative")
    if int(cfg.geometry_candidate_early_start_count) < 0:
        raise ValueError("geometry_candidate_early_start_count must be non-negative")
    if int(cfg.geometry_candidate_middle_start_count) < 0:
        raise ValueError("geometry_candidate_middle_start_count must be non-negative")
    if int(cfg.geometry_candidate_recent_start_count) < 0:
        raise ValueError("geometry_candidate_recent_start_count must be non-negative")
    if (
        int(cfg.geometry_candidate_early_start_count)
        + int(cfg.geometry_candidate_middle_start_count)
        + int(cfg.geometry_candidate_recent_start_count)
        < 2
    ):
        raise ValueError("geometry candidate start counts must total at least 2")
    if float(cfg.geometry_candidate_min_span_mult) <= 0.0:
        raise ValueError("geometry_candidate_min_span_mult must be positive")
    if int(cfg.geometry_candidate_min_span_bars) < 1:
        raise ValueError("geometry_candidate_min_span_bars must be at least 1")
    if float(cfg.geometry_line_fit_tolerance_mult) <= 0.0:
        raise ValueError("geometry_line_fit_tolerance_mult must be positive")
    if int(cfg.geometry_envelope_eval_step) < 1:
        raise ValueError("geometry_envelope_eval_step must be at least 1")
    if float(cfg.geometry_envelope_min_span_mult) <= 0.0:
        raise ValueError("geometry_envelope_min_span_mult must be positive")
    if int(cfg.geometry_envelope_start_count) < 1:
        raise ValueError("geometry_envelope_start_count must be at least 1")
    if float(cfg.geometry_envelope_pair_min_span_mult) <= 0.0:
        raise ValueError("geometry_envelope_pair_min_span_mult must be positive")
    if float(cfg.geometry_envelope_touch_atr_mult) <= 0.0:
        raise ValueError("geometry_envelope_touch_atr_mult must be positive")
    if float(cfg.geometry_envelope_max_width_atr) <= 0.0:
        raise ValueError("geometry_envelope_max_width_atr must be positive")
    if int(cfg.geometry_envelope_max_side_pivots) < 2:
        raise ValueError("geometry_envelope_max_side_pivots must be at least 2")
    if int(cfg.geometry_envelope_max_pair_options) < 1:
        raise ValueError("geometry_envelope_max_pair_options must be at least 1")
    if not 0.0 <= float(cfg.geometry_envelope_min_touch_span_ratio) <= 1.0:
        raise ValueError("geometry_envelope_min_touch_span_ratio must be between 0 and 1")
    if float(cfg.geometry_envelope_merge_slope_tolerance_pct) < 0.0:
        raise ValueError("geometry_envelope_merge_slope_tolerance_pct must be non-negative")
    if float(cfg.geometry_envelope_proximity_atr_mult) <= 0.0:
        raise ValueError("geometry_envelope_proximity_atr_mult must be positive")
    if int(cfg.geometry_envelope_proximity_grace_bars) < 0:
        raise ValueError("geometry_envelope_proximity_grace_bars must be non-negative")
    if int(cfg.geometry_envelope_proximity_ramp_bars) < 1:
        raise ValueError("geometry_envelope_proximity_ramp_bars must be at least 1")
    if float(cfg.geometry_envelope_proximity_penalty_weight) < 0.0:
        raise ValueError("geometry_envelope_proximity_penalty_weight must be non-negative")
    if int(cfg.geometry_pivot_dominance_bars) < 0:
        raise ValueError("geometry_pivot_dominance_bars must be non-negative")
    if float(cfg.geometry_pivot_dominance_atr_mult) < 0.0:
        raise ValueError("geometry_pivot_dominance_atr_mult must be non-negative")
    if int(cfg.geometry_pivot_merge_bars) < 0:
        raise ValueError("geometry_pivot_merge_bars must be non-negative")
    if float(cfg.geometry_pivot_merge_atr_mult) < 0.0:
        raise ValueError("geometry_pivot_merge_atr_mult must be non-negative")
    line_weights = [
        float(cfg.geometry_line_touch_weight),
        float(cfg.geometry_line_fit_weight),
        float(cfg.geometry_line_span_weight),
        float(cfg.geometry_line_recency_weight),
    ]
    if any(weight < 0.0 for weight in line_weights):
        raise ValueError("geometry line weights must be non-negative")
    if sum(line_weights) <= 0.0:
        raise ValueError("at least one geometry line weight must be positive")
    if float(cfg.geometry_slot_duplicate_price_tolerance_pct) < 0.0:
        raise ValueError("geometry_slot_duplicate_price_tolerance_pct must be non-negative")
    if not 0.0 <= float(cfg.geometry_slot_duplicate_start_tolerance) <= 1.0:
        raise ValueError("geometry_slot_duplicate_start_tolerance must be between 0 and 1")
    if float(cfg.geometry_body_invalidation_atr) <= 0.0:
        raise ValueError("geometry_body_invalidation_atr must be positive")
    if float(cfg.geometry_window_time_scale_power) < 0.0:
        raise ValueError("geometry_window_time_scale_power must be non-negative")
    if float(cfg.geometry_window_reference_seconds) <= 0.0:
        raise ValueError("geometry_window_reference_seconds must be positive")
    if float(cfg.geometry_window_max_mult) < 1.0:
        raise ValueError("geometry_window_max_mult must be at least 1")
    if float(cfg.geometry_timeframe_seconds) < 0.0:
        raise ValueError("geometry_timeframe_seconds must be non-negative")
    if int(cfg.geometry_management_max_age_bars) < 1:
        raise ValueError("geometry_management_max_age_bars must be at least 1")
    if not 1 <= int(cfg.geometry_output_slots) <= 4:
        raise ValueError("geometry_output_slots must be between 1 and 4")
    if float(cfg.flat_boundary_slope_pct_per_bar) <= 0.0:
        raise ValueError("flat_boundary_slope_pct_per_bar must be positive")
    if int(cfg.double_pattern_window) < 8:
        raise ValueError("double_pattern_window must be at least 8")
    if int(cfg.min_double_pattern_bars) < 2:
        raise ValueError("min_double_pattern_bars must be at least 2")
    if int(cfg.max_double_pattern_bars) <= int(cfg.min_double_pattern_bars):
        raise ValueError("max_double_pattern_bars must be greater than min_double_pattern_bars")
    if int(cfg.double_pattern_window) < int(cfg.max_double_pattern_bars):
        raise ValueError("double_pattern_window must be at least max_double_pattern_bars")
    if float(cfg.double_peak_tolerance_pct) <= 0.0:
        raise ValueError("double_peak_tolerance_pct must be positive")
    if float(cfg.min_double_neckline_depth_pct) < 0.0:
        raise ValueError("min_double_neckline_depth_pct must be non-negative")
    if float(cfg.min_double_prior_move_pct) < 0.0:
        raise ValueError("min_double_prior_move_pct must be non-negative")
    if float(cfg.min_double_first_pivot_move_pct) < 0.0:
        raise ValueError("min_double_first_pivot_move_pct must be non-negative")
    if int(cfg.double_reaction_max_bars) < 1:
        raise ValueError("double_reaction_max_bars must be at least 1")
    if int(cfg.peak_dynamic_body_window) < 3:
        raise ValueError("peak_dynamic_body_window must be at least 3")
    if int(cfg.peak_dynamic_scale_window) < 3:
        raise ValueError("peak_dynamic_scale_window must be at least 3")
    if float(cfg.peak_premove_body_mult) <= 0.0:
        raise ValueError("peak_premove_body_mult must be positive")
    if float(cfg.peak_level_tolerance_body_mult) <= 0.0:
        raise ValueError("peak_level_tolerance_body_mult must be positive")
    if float(cfg.peak_level_tolerance_atr_mult) < 0.0:
        raise ValueError("peak_level_tolerance_atr_mult must be non-negative")
    if float(cfg.peak_level_tolerance_prominence_mult) < 0.0:
        raise ValueError("peak_level_tolerance_prominence_mult must be non-negative")
    if float(cfg.peak_reaction_body_mult) <= 0.0:
        raise ValueError("peak_reaction_body_mult must be positive")
    if float(cfg.peak_base_return_buffer_body_mult) < 0.0:
        raise ValueError("peak_base_return_buffer_body_mult must be non-negative")
    if int(cfg.peak_prior_impulse_min_bars) < 1:
        raise ValueError("peak_prior_impulse_min_bars must be at least 1")
    if not 0.0 <= float(cfg.peak_prior_impulse_min_efficiency) <= 1.0:
        raise ValueError("peak_prior_impulse_min_efficiency must be between 0 and 1")
    if not 0.0 <= float(cfg.min_double_quality) <= 1.0:
        raise ValueError("min_double_quality must be between 0 and 1")
    if not 0.0 <= float(cfg.min_double_reaction_score) <= 1.0:
        raise ValueError("min_double_reaction_score must be between 0 and 1")
    if not 0.0 <= float(cfg.min_double_between_cleanliness_score) <= 1.0:
        raise ValueError("min_double_between_cleanliness_score must be between 0 and 1")
    if not 0.0 <= float(cfg.double_duplicate_overlap_pct) <= 1.0:
        raise ValueError("double_duplicate_overlap_pct must be between 0 and 1")
    if float(cfg.double_duplicate_neckline_tolerance_pct) < 0.0:
        raise ValueError("double_duplicate_neckline_tolerance_pct must be non-negative")
    if int(cfg.head_shoulders_window) < 12:
        raise ValueError("head_shoulders_window must be at least 12")
    if int(cfg.min_head_shoulders_bars) < 4:
        raise ValueError("min_head_shoulders_bars must be at least 4")
    if int(cfg.max_head_shoulders_bars) <= int(cfg.min_head_shoulders_bars):
        raise ValueError("max_head_shoulders_bars must be greater than min_head_shoulders_bars")
    if int(cfg.head_shoulders_window) < int(cfg.max_head_shoulders_bars):
        raise ValueError("head_shoulders_window must be at least max_head_shoulders_bars")
    if float(cfg.shoulder_tolerance_pct) <= 0.0:
        raise ValueError("shoulder_tolerance_pct must be positive")
    if float(cfg.min_head_prominence_pct) < 0.0:
        raise ValueError("min_head_prominence_pct must be non-negative")
    if float(cfg.min_head_shoulders_neckline_depth_pct) < 0.0:
        raise ValueError("min_head_shoulders_neckline_depth_pct must be non-negative")
    if float(cfg.min_head_shoulders_prior_move_pct) < 0.0:
        raise ValueError("min_head_shoulders_prior_move_pct must be non-negative")
    if not 0.0 <= float(cfg.min_head_shoulders_quality) <= 1.0:
        raise ValueError("min_head_shoulders_quality must be between 0 and 1")
    if not 0.0 <= float(cfg.min_head_shoulders_shoulder_score) <= 1.0:
        raise ValueError("min_head_shoulders_shoulder_score must be between 0 and 1")
    if not 0.0 <= float(cfg.min_head_shoulders_time_balance_score) <= 1.0:
        raise ValueError("min_head_shoulders_time_balance_score must be between 0 and 1")
    if not 0.0 <= float(cfg.min_head_shoulders_neckline_score) <= 1.0:
        raise ValueError("min_head_shoulders_neckline_score must be between 0 and 1")
    if not 0.0 <= float(cfg.min_head_shoulders_neckline_position_score) <= 1.0:
        raise ValueError("min_head_shoulders_neckline_position_score must be between 0 and 1")
    if not 0.0 <= float(cfg.min_head_shoulders_head_position_score) <= 1.0:
        raise ValueError("min_head_shoulders_head_position_score must be between 0 and 1")
    if not 0.0 <= float(cfg.min_head_shoulders_neckline_clearance_score) <= 1.0:
        raise ValueError("min_head_shoulders_neckline_clearance_score must be between 0 and 1")
    if not 0.0 <= float(cfg.min_head_shoulders_neckline_body_respect_ratio) <= 1.0:
        raise ValueError("min_head_shoulders_neckline_body_respect_ratio must be between 0 and 1")
    if float(cfg.head_shoulders_max_neckline_slope_pct_per_bar) <= 0.0:
        raise ValueError("head_shoulders_max_neckline_slope_pct_per_bar must be positive")
    if not 0.0 <= float(cfg.min_head_shoulders_right_reaction_score) <= 1.0:
        raise ValueError("min_head_shoulders_right_reaction_score must be between 0 and 1")
    if int(cfg.head_shoulders_setup_monitor_bars) < 1:
        raise ValueError("head_shoulders_setup_monitor_bars must be at least 1")
    if float(cfg.head_shoulders_neckline_proximity_pct) < 0.0:
        raise ValueError("head_shoulders_neckline_proximity_pct must be non-negative")
    if int(cfg.rectangle_window) < 8:
        raise ValueError("rectangle_window must be at least 8")
    if int(cfg.min_rectangle_bars) < 4:
        raise ValueError("min_rectangle_bars must be at least 4")
    if int(cfg.min_rectangle_bars) >= int(cfg.rectangle_window):
        raise ValueError("min_rectangle_bars must be lower than rectangle_window")
    if not 0.0 < float(cfg.rectangle_min_width_pct) < float(cfg.rectangle_max_width_pct):
        raise ValueError("rectangle_min_width_pct must be lower than rectangle_max_width_pct")
    if float(cfg.rectangle_boundary_tolerance_pct) <= 0.0:
        raise ValueError("rectangle_boundary_tolerance_pct must be positive")
    if float(cfg.rectangle_management_buffer_pct) < 0.0:
        raise ValueError("rectangle_management_buffer_pct must be non-negative")
    if int(cfg.rectangle_min_side_touches) < 2:
        raise ValueError("rectangle_min_side_touches must be at least 2")
    if not 0.0 <= float(cfg.min_rectangle_containment_ratio) <= 1.0:
        raise ValueError("min_rectangle_containment_ratio must be between 0 and 1")
    if not 0.0 <= float(cfg.min_rectangle_quality) <= 1.0:
        raise ValueError("min_rectangle_quality must be between 0 and 1")
    if int(cfg.triple_pattern_window) < 12:
        raise ValueError("triple_pattern_window must be at least 12")
    if int(cfg.min_triple_pattern_bars) < 6:
        raise ValueError("min_triple_pattern_bars must be at least 6")
    if int(cfg.max_triple_pattern_bars) <= int(cfg.min_triple_pattern_bars):
        raise ValueError("max_triple_pattern_bars must be greater than min_triple_pattern_bars")
    if int(cfg.triple_pattern_window) < int(cfg.max_triple_pattern_bars):
        raise ValueError("triple_pattern_window must be at least max_triple_pattern_bars")
    if int(cfg.min_triple_spacing_bars) < 1:
        raise ValueError("min_triple_spacing_bars must be at least 1")
    if int(cfg.triple_max_candidate_pivots) < 3:
        raise ValueError("triple_max_candidate_pivots must be at least 3")
    if float(cfg.triple_peak_tolerance_pct) <= 0.0:
        raise ValueError("triple_peak_tolerance_pct must be positive")
    if float(cfg.min_triple_neckline_depth_pct) < 0.0:
        raise ValueError("min_triple_neckline_depth_pct must be non-negative")
    if float(cfg.min_triple_prior_move_pct) < 0.0:
        raise ValueError("min_triple_prior_move_pct must be non-negative")
    if float(cfg.min_triple_first_pivot_move_pct) < 0.0:
        raise ValueError("min_triple_first_pivot_move_pct must be non-negative")
    if int(cfg.triple_reaction_max_bars) < 1:
        raise ValueError("triple_reaction_max_bars must be at least 1")
    if not 0.0 <= float(cfg.min_triple_reaction_score) <= 1.0:
        raise ValueError("min_triple_reaction_score must be between 0 and 1")
    if not 0.0 <= float(cfg.min_triple_quality) <= 1.0:
        raise ValueError("min_triple_quality must be between 0 and 1")
    if int(cfg.channel_pattern_window) < 12:
        raise ValueError("channel_pattern_window must be at least 12")
    if int(cfg.broadening_pattern_window) < 12:
        raise ValueError("broadening_pattern_window must be at least 12")
    if int(cfg.min_channel_pattern_bars) < 6:
        raise ValueError("min_channel_pattern_bars must be at least 6")
    if int(cfg.min_channel_pattern_bars) >= int(cfg.channel_pattern_window):
        raise ValueError("min_channel_pattern_bars must be lower than channel_pattern_window")
    if int(cfg.min_channel_pattern_bars) >= int(cfg.broadening_pattern_window):
        raise ValueError("min_channel_pattern_bars must be lower than broadening_pattern_window")
    if int(cfg.channel_pattern_min_side_pivots) < 2:
        raise ValueError("channel_pattern_min_side_pivots must be at least 2")
    if not 0.0 < float(cfg.channel_pattern_min_width_pct) < float(cfg.channel_pattern_max_width_pct):
        raise ValueError("channel_pattern_min_width_pct must be lower than channel_pattern_max_width_pct")
    if float(cfg.min_channel_pattern_slope_pct_per_bar) <= 0.0:
        raise ValueError("min_channel_pattern_slope_pct_per_bar must be positive")
    if float(cfg.channel_parallel_slope_tolerance_pct_per_bar) <= 0.0:
        raise ValueError("channel_parallel_slope_tolerance_pct_per_bar must be positive")
    if float(cfg.max_channel_pattern_fit_error_pct) <= 0.0:
        raise ValueError("max_channel_pattern_fit_error_pct must be positive")
    if not 0.0 <= float(cfg.min_channel_pattern_containment_ratio) <= 1.0:
        raise ValueError("min_channel_pattern_containment_ratio must be between 0 and 1")
    if float(cfg.channel_pattern_boundary_tolerance_pct) <= 0.0:
        raise ValueError("channel_pattern_boundary_tolerance_pct must be positive")
    if float(cfg.channel_pattern_management_buffer_pct) < 0.0:
        raise ValueError("channel_pattern_management_buffer_pct must be non-negative")
    if not 0.0 <= float(cfg.min_channel_pattern_quality) <= 1.0:
        raise ValueError("min_channel_pattern_quality must be between 0 and 1")
    if float(cfg.min_broadening_expansion_pct) <= 0.0:
        raise ValueError("min_broadening_expansion_pct must be positive")
    if float(cfg.min_broadening_slope_gap_pct_per_bar) <= 0.0:
        raise ValueError("min_broadening_slope_gap_pct_per_bar must be positive")
    if float(cfg.min_broadening_prior_move_pct) < 0.0:
        raise ValueError("min_broadening_prior_move_pct must be non-negative")
    if not 0.0 <= float(cfg.min_broadening_quality) <= 1.0:
        raise ValueError("min_broadening_quality must be between 0 and 1")
    if int(cfg.pattern_lifecycle_mature_bars) < 1:
        raise ValueError("pattern_lifecycle_mature_bars must be at least 1")
    if int(cfg.pattern_lifecycle_stale_bars) < 0:
        raise ValueError("pattern_lifecycle_stale_bars must be non-negative")
    if not 0.0 <= float(cfg.min_channel_score) <= 1.0:
        raise ValueError("min_channel_score must be between 0 and 1")
    if not 0.0 <= float(cfg.min_sequence_score) <= 1.0:
        raise ValueError("min_sequence_score must be between 0 and 1")
    if int(cfg.entry_cooldown_bars) < 1:
        raise ValueError("entry_cooldown_bars must be at least 1")


def _with_foundation_pivots(frame: DataFrame, cfg: PatternStructureConfig) -> DataFrame:
    open_ = _num(frame["open"])
    close = _num(frame["close"])
    body_high = pd.concat([open_, close], axis=1).max(axis=1)
    body_low = pd.concat([open_, close], axis=1).min(axis=1)
    pivots = build_clean_pivot_source(
        body_high=body_high,
        body_low=body_low,
        atr=_atr(frame, 14),
        bar_index=_bar_index(frame),
        strength=int(cfg.pivot_strength),
        min_prominence_atr=float(cfg.pivot_min_prominence_atr),
        min_prominence_pct=float(cfg.pivot_min_prominence_pct),
        min_pivot_spacing_bars=int(cfg.pivot_min_spacing_bars),
        min_pivot_distance_atr=float(cfg.pivot_min_distance_atr),
        min_pivot_distance_pct=float(cfg.pivot_min_distance_pct),
    )
    out = frame.copy()
    pp = cfg.pivot_prefix
    for name, series in pivots.items():
        out[f"{pp}_{name}"] = series
    return out


def _validate_ohlcv_dataframe(frame: DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Pattern Structure requires OHLCV columns. Missing: {', '.join(missing)}")


def _validate_dataframe(frame: DataFrame, cfg: PatternStructureConfig) -> None:
    pp = cfg.pivot_prefix
    required = set()
    required.update(
        {
            f"{pp}_pivot_high",
            f"{pp}_pivot_low",
            f"{pp}_pivot_high_index",
            f"{pp}_pivot_low_index",
            f"{pp}_pivot_high_prominence_pct",
            f"{pp}_pivot_low_prominence_pct",
        }
    )
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(
            "Pattern Structure requires the new pivot foundation columns. "
            f"Missing: {', '.join(missing[:12])}"
        )


def _channel_required_columns(cfg: PatternStructureConfig) -> list[str]:
    cp = cfg.channel_prefix
    label = cfg.channel_label
    rank = int(cfg.channel_rank)
    return [
        f"{cp}_{label}_score_rank{rank}",
        f"{cp}_{label}_active_rank{rank}",
        f"{cp}_{label}_upper_rank{rank}",
        f"{cp}_{label}_lower_rank{rank}",
        f"{cp}_{label}_mid_rank{rank}",
        f"{cp}_{label}_position_rank{rank}",
        f"{cp}_{label}_width_pct_rank{rank}",
        f"{cp}_{label}_shape_rank{rank}",
        f"{cp}_{label}_inside_rank{rank}",
        f"{cp}_{label}_near_upper_rank{rank}",
        f"{cp}_{label}_near_lower_rank{rank}",
        f"{cp}_{label}_breakout_up_rank{rank}",
        f"{cp}_{label}_breakdown_down_rank{rank}",
    ]




def _bar_index(frame: DataFrame) -> Series:
    return pd.Series(np.arange(len(frame), dtype="float64"), index=frame.index)


def _atr(frame: DataFrame, period: int = 14) -> Series:
    high = _num(frame["high"])
    low = _num(frame["low"])
    close = _num(frame["close"])
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


def _rolling_pivot_slope(price: Series, pivot_index: Series, event: Series, window: int) -> Series:
    clean_event = pd.Series(event, index=price.index).fillna(False)
    x = _num(pivot_index).where(clean_event)
    y = _num(price).where(clean_event)
    n = clean_event.astype("float64").rolling(window, min_periods=1).sum()
    sum_x = x.fillna(0.0).rolling(window, min_periods=1).sum()
    sum_y = y.fillna(0.0).rolling(window, min_periods=1).sum()
    sum_xy = (x * y).fillna(0.0).rolling(window, min_periods=1).sum()
    sum_x2 = (x * x).fillna(0.0).rolling(window, min_periods=1).sum()
    denominator = (n * sum_x2 - sum_x * sum_x).replace(0.0, np.nan)
    slope = ((n * sum_xy - sum_x * sum_y) / denominator).replace([np.inf, -np.inf], np.nan)
    return slope.where(n.ge(2.0), 0.0).fillna(0.0)


def _rolling_extreme_position(series: Series, window: int, mode: str) -> Series:
    clean = _num(series)
    min_periods = max(4, int(window) // 3)

    def locate(values: np.ndarray) -> float:
        valid = np.isfinite(values)
        if valid.sum() < min_periods:
            return np.nan
        compact = values[valid]
        source_positions = np.flatnonzero(valid)
        local_pos = int(np.nanargmin(compact) if mode == "min" else np.nanargmax(compact))
        return float(source_positions[local_pos])

    return clean.rolling(int(window), min_periods=min_periods).apply(locate, raw=True)


def _bool(series: Series) -> Series:
    data = pd.Series(series, index=series.index)
    if pd.api.types.is_bool_dtype(data) or str(data.dtype) == "boolean":
        return data.astype("boolean").fillna(False).astype("bool")
    numeric = pd.to_numeric(data, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0.0).ne(0.0)
    return data.notna() & data.ne(False)


def _clip01(series: Series) -> Series:
    return _num(series).clip(0.0, 1.0).fillna(0.0)


def _rolling_count(mask: Series, window: int) -> Series:
    clean = pd.Series(mask, index=mask.index).fillna(False).astype("int8")
    cumulative = clean.cumsum()
    return (cumulative - cumulative.shift(int(window), fill_value=0)).astype("float64")


def _dedupe_events(mask: Series, cooldown_bars: int) -> Series:
    clean = pd.Series(mask, index=mask.index).fillna(False).astype("bool")
    prior_recent = _rolling_count(clean.shift(1, fill_value=False), max(int(cooldown_bars), 1))
    return (clean & prior_recent.eq(0.0)).fillna(False)


__all__ = [
    "PatternStructureConfig",
    "add_pattern_structure",
]

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

LineSide = Literal["resistance", "support"]
LineMode = Literal["raw", "confirmed", "absorbed"]


@dataclass(frozen=True)
class TrendlineProjectionV2Config:
    """First-pass body-pivot trendline generator.

    This version deliberately does less than the older prototype. It creates a
    broad raw candidate set first, applies only basic sanity trims, and extracts
    optional fuzzy zones / convergence hotspots before any heavy merge/refit
    logic can delete useful evidence.

    Tunable first-pass levers:
    - ``pivot_strength``: confirmed body pivot strength. A pivot is emitted only
      after this many candles have confirmed it.
    - ``candidate_pivot_count``: rolling strategy-facing candidate depth. The
      diagnostic helper can still inspect all pivots in the supplied window.
    - ``max_slope_pct_per_bar``: removes extreme angle lines, normalized by
      price so the same rule can work across coins/timeframes.
    - ``min_anchor_bars``: removes pivot pairs that are too close together.
    - ``max_anchor_bars``: removes seed lines whose p1->p2 span is too long to
      be a clean initial trendline definition.
    - ``max_projection_bars``: caps each line after its second anchor.
    - ``structural_max_anchor_bars`` / ``structural_max_projection_bars``:
      optional broader caps for slower, shallow market structure lines. These
      are kept separate from local raw lines because they solve a different
      problem: floors/ceilings that need more time between first two anchors.
    - ``channel_*``: optional channel detector settings. Channels pair a real
      resistance line with a real support line and score both parallel ranges
      and converging wedge-like ranges. Channel source lines can use weaker
      local pivots than standalone trendlines because ranges often form from
      smaller repeated touches.
    - ``density_proximity_pct`` / ``density_min_lines``: define same-side fuzzy
      support/resistance bands.
    - ``hotspot_proximity_pct`` / ``hotspot_min_lines``: define all-line
      convergence hotspots.
    - ``confirmation_pivot_strength``: weaker local pivots used only to confirm
      a seed line. Seed anchors remain the stronger ``pivot_strength`` pivots.
    - ``confirmation_tolerance_pct`` / ``confirmation_atr_mult``: how close a
      later real pivot must be to the original p1->p2 seed line to confirm it.
    - ``absorb_touch_tolerance_pct`` / ``absorb_angle_tolerance_pct``: optional
      evidence-gathering mode. Nearby similar lines can add score/width, but
      they cannot rotate the seed geometry away from real pivots.
    - ``anchor_break_*``: nuanced p1->p2 body-break handling. Minor body breaks
      near either anchor, or short/shallow interior breaks, can be allowed so
      valid human-looking lines are not deleted by pivot granularity.
    - ``max_raw_lines_plotted``: plotting safety only; it is not an indicator
      filter.

    Strategy-facing outputs:
    - ``*_resistance_line_rankN`` / ``*_support_line_rankN``: raw p1->p2 seed
      candidates.
    - ``*_confirmed_resistance_line_rankN`` /
      ``*_confirmed_support_line_rankN``: seed geometry that has at least one
      later real-pivot confirmation.
    - ``*_absorbed_resistance_line_rankN`` /
      ``*_absorbed_support_line_rankN``: confirmed seed geometry with nearby
      same-side evidence absorbed into score/width. Geometry remains anchored
      to the original seed line.
    - ``*_structural_resistance_line_rankN`` /
      ``*_structural_support_line_rankN``: longer-span p1->p2 candidates for
      broad floors/ceilings. These are disabled by default because they are
      slower and should be reviewed separately from local line behaviour.
    - ``*_local_channel_upper_rankN`` / ``*_local_channel_lower_rankN``:
      optional paired support/resistance structures from weaker local pivots.
      These are intended as active channel evidence. ``*_channel_*`` remains as
      a compatibility alias for this local-channel output.
    - ``*_structural_channel_upper_rankN`` /
      ``*_structural_channel_lower_rankN``: optional paired longer-span
      support/resistance structures. These are broader market context, not
      precise local entry channels.
    - ``*_fuzzy_resistance_price_rankN`` / ``*_fuzzy_support_price_rankN``:
      same-side dense line zones. These are extra evidence, not core
      trendlines.
    - ``*_hotspot_price_rankN``: places where many support/resistance candidates
      converge around the same candle/price area.
    """

    output_prefix: str = "tlv2"
    pivot_strength: int = 5
    confirmation_pivot_strength: int = 3
    candidate_pivot_count: int = 36
    raw_line_output_count: int = 14
    confirmed_line_output_count: int = 14
    absorbed_line_output_count: int = 14
    structural_line_output_count: int = 10
    channel_output_count: int = 5
    emit_confirmed_lines: bool = False
    emit_absorbed_lines: bool = False
    emit_structural_lines: bool = False
    emit_channel_lines: bool = False
    emit_structural_channel_lines: bool = False
    fuzzy_zone_count: int = 3
    hotspot_count: int = 3

    min_anchor_bars: int = 10
    max_anchor_bars: int = 50
    structural_max_anchor_bars: int = 180
    min_pivot_prominence_atr: float = 0.35
    min_confirmation_pivot_prominence_atr: float = 0.18
    max_slope_pct_per_bar: float = 0.012
    max_projection_bars: int = 50
    structural_max_projection_bars: int = 320

    channel_source_line_count: int = 10
    channel_candidate_pool_size: int = 320
    channel_min_pivot_prominence_atr: float = 0.12
    channel_min_anchor_bars: int = 6
    channel_max_anchor_bars: int = 120
    channel_max_projection_bars: int = 180
    channel_min_overlap_bars: int = 10
    channel_slope_tolerance_pct: float = 0.45
    channel_parallel_width_change_pct: float = 0.35
    channel_min_convergence_pct: float = 0.12
    channel_min_width_pct: float = 0.004
    channel_max_width_pct: float = 0.24
    channel_min_containment_ratio: float = 0.48
    channel_touch_tolerance_pct: float = 0.0050
    channel_touch_atr_mult: float = 0.55
    channel_min_pivot_touches: int = 4
    channel_min_rail_touches: int = 1
    channel_min_recent_rail_touches: int = 1
    channel_recent_touch_fraction: float = 0.45
    channel_min_position_coverage: float = 0.14
    channel_duplicate_overlap_pct: float = 0.60
    channel_duplicate_mid_width_mult: float = 0.45
    channel_duplicate_slope_tolerance_pct: float = 0.25
    channel_breakout_grace_bars: int = 3

    confirmation_tolerance_pct: float = 0.0040
    confirmation_atr_mult: float = 0.35
    confirmation_max_bars: int = 80
    confirmation_min_pivots: int = 3
    absorb_touch_tolerance_pct: float = 0.0045
    absorb_angle_tolerance_pct: float = 0.18
    absorb_width_scale: float = 0.30
    anchor_break_edge_bars: int = 3
    anchor_break_max_run: int = 2
    anchor_break_max_pct: float = 0.0035
    anchor_break_max_atr_mult: float = 0.40
    projection_break_candles: int = 5
    projection_break_tolerance_pct: float = 0.0015
    projection_break_atr_mult: float = 0.25

    density_proximity_pct: float = 0.0075
    density_min_lines: int = 8
    hotspot_proximity_pct: float = 0.0050
    hotspot_min_lines: int = 10
    join_angle_tolerance_pct: float = 0.15
    join_midpoint_tolerance_pct: float = 0.012
    max_joined_pivots: int = 6
    shared_pivot_merge_min: int = 2
    nearby_merge_proximity_pcts: tuple[float, ...] = (0.01, 0.02, 0.03)
    max_raw_lines_plotted: int = 180

    plot_break_on_line_change_pct: float = 0.035


@dataclass(frozen=True)
class TrendlineProjectionV2Diagnostic:
    """Diagnostic-only payload for visual review scripts."""

    frame: DataFrame
    raw_candidates: DataFrame
    structural_candidates: DataFrame
    channel_source_candidates: DataFrame
    raw_channel_candidates: DataFrame
    structural_channel_candidates: DataFrame
    channel_candidates: DataFrame
    confirmed_candidates: DataFrame
    absorbed_candidates: DataFrame
    joined_candidates: DataFrame
    shared_pivot_merged_candidates: DataFrame
    nearby_merged_candidates: dict[float, DataFrame]
    fuzzy_zones: DataFrame
    hotspots: DataFrame
    counts: dict[str, int]


def add_trendline_projection_v2(
    dataframe: DataFrame,
    config: TrendlineProjectionV2Config | None = None,
    **overrides: object,
) -> DataFrame:
    """Append first-pass v2 trendline evidence columns.

    The output intentionally keeps raw trimmed lines, fuzzy zones, and hotspots
    separate. Strategies can later decide whether any of these concepts are
    useful, but this module does not create entries, exits, stake, or risk
    decisions.
    """

    cfg = _resolve_config(config, overrides)
    _validate_config(cfg)
    _validate_dataframe(dataframe)

    frame = dataframe.copy()
    base = _base_inputs(frame, cfg)
    resistance = _rolling_candidate_pack(base, "resistance", cfg, mode="raw")
    support = _rolling_candidate_pack(base, "support", cfg, mode="raw")
    structural_required = bool(cfg.emit_structural_lines or cfg.emit_structural_channel_lines)
    structural_resistance = (
        _rolling_candidate_pack(
            base,
            "resistance",
            cfg,
            mode="raw",
            max_anchor_bars=int(cfg.structural_max_anchor_bars),
            max_projection_bars=int(cfg.structural_max_projection_bars),
        )
        if structural_required
        else _empty_pack()
    )
    structural_support = (
        _rolling_candidate_pack(
            base,
            "support",
            cfg,
            mode="raw",
            max_anchor_bars=int(cfg.structural_max_anchor_bars),
            max_projection_bars=int(cfg.structural_max_projection_bars),
        )
        if structural_required
        else _empty_pack()
    )
    confirmed_resistance = _rolling_candidate_pack(base, "resistance", cfg, mode="confirmed") if cfg.emit_confirmed_lines else _empty_pack()
    confirmed_support = _rolling_candidate_pack(base, "support", cfg, mode="confirmed") if cfg.emit_confirmed_lines else _empty_pack()
    absorbed_resistance = _rolling_candidate_pack(base, "resistance", cfg, mode="absorbed") if cfg.emit_absorbed_lines else _empty_pack()
    absorbed_support = _rolling_candidate_pack(base, "support", cfg, mode="absorbed") if cfg.emit_absorbed_lines else _empty_pack()
    channel_source_resistance = (
        _rolling_candidate_pack(
            base,
            "resistance",
            cfg,
            mode="raw",
            min_anchor_bars=int(cfg.channel_min_anchor_bars),
            max_anchor_bars=int(cfg.channel_max_anchor_bars),
            max_projection_bars=int(cfg.channel_max_projection_bars),
            pivot_family="confirmation",
            min_pivot_prominence_atr=float(cfg.channel_min_pivot_prominence_atr),
        )
        if cfg.emit_channel_lines
        else _empty_pack()
    )
    channel_source_support = (
        _rolling_candidate_pack(
            base,
            "support",
            cfg,
            mode="raw",
            min_anchor_bars=int(cfg.channel_min_anchor_bars),
            max_anchor_bars=int(cfg.channel_max_anchor_bars),
            max_projection_bars=int(cfg.channel_max_projection_bars),
            pivot_family="confirmation",
            min_pivot_prominence_atr=float(cfg.channel_min_pivot_prominence_atr),
        )
        if cfg.emit_channel_lines
        else _empty_pack()
    )
    p = cfg.output_prefix

    new_cols: dict[str, Series] = {
        f"{p}_atr": base["atr"],
        f"{p}_body_high": base["body_high"],
        f"{p}_body_low": base["body_low"],
        f"{p}_pivot_high": base["pivot_high"],
        f"{p}_pivot_low": base["pivot_low"],
        f"{p}_pivot_high_prominence": base["pivot_high_prominence"],
        f"{p}_pivot_low_prominence": base["pivot_low_prominence"],
        f"{p}_confirm_pivot_high": base["confirm_pivot_high"],
        f"{p}_confirm_pivot_low": base["confirm_pivot_low"],
        f"{p}_confirm_pivot_high_prominence": base["confirm_pivot_high_prominence"],
        f"{p}_confirm_pivot_low_prominence": base["confirm_pivot_low_prominence"],
    }
    raw_resistance_cols = _rank_raw_candidate_columns(resistance, "resistance", cfg, slots=int(cfg.raw_line_output_count), index=frame.index)
    raw_support_cols = _rank_raw_candidate_columns(support, "support", cfg, slots=int(cfg.raw_line_output_count), index=frame.index)
    structural_resistance_cols = _rank_raw_candidate_columns(
        structural_resistance,
        "structural_resistance",
        cfg,
        slots=int(cfg.structural_line_output_count),
        index=frame.index,
    )
    structural_support_cols = _rank_raw_candidate_columns(
        structural_support,
        "structural_support",
        cfg,
        slots=int(cfg.structural_line_output_count),
        index=frame.index,
    )
    new_cols.update(raw_resistance_cols)
    new_cols.update(raw_support_cols)
    new_cols.update(structural_resistance_cols)
    new_cols.update(structural_support_cols)
    if cfg.emit_channel_lines:
        channel_source_resistance_cols = _rank_raw_candidate_columns(
            channel_source_resistance,
            "channel_source_resistance",
            cfg,
            slots=int(cfg.channel_source_line_count),
            index=frame.index,
        )
        channel_source_support_cols = _rank_raw_candidate_columns(
            channel_source_support,
            "channel_source_support",
            cfg,
            slots=int(cfg.channel_source_line_count),
            index=frame.index,
        )
        local_channel_cols = _channel_columns_from_ranked(
            {**channel_source_resistance_cols, **channel_source_support_cols},
            base,
            cfg,
            source_label="channel_source",
            output_label="local_channel",
        )
        new_cols.update(local_channel_cols)
        new_cols.update(_alias_channel_columns(local_channel_cols, p, source_label="local_channel", alias_label="channel"))
    else:
        new_cols.update(_empty_channel_columns(frame.index, p, int(cfg.channel_output_count), label="local_channel"))
        new_cols.update(_empty_channel_columns(frame.index, p, int(cfg.channel_output_count), label="channel"))
    if cfg.emit_structural_channel_lines:
        structural_channel_cols = _channel_columns_from_ranked(
            {**structural_resistance_cols, **structural_support_cols},
            base,
            cfg,
            source_label="structural",
            output_label="structural_channel",
            source_count=int(cfg.structural_line_output_count),
        )
        new_cols.update(structural_channel_cols)
    else:
        new_cols.update(_empty_channel_columns(frame.index, p, int(cfg.channel_output_count), label="structural_channel"))
    new_cols.update(
        _rank_raw_candidate_columns(
            confirmed_resistance,
            "confirmed_resistance",
            cfg,
            slots=int(cfg.confirmed_line_output_count),
            index=frame.index,
        )
    )
    new_cols.update(
        _rank_raw_candidate_columns(
            confirmed_support,
            "confirmed_support",
            cfg,
            slots=int(cfg.confirmed_line_output_count),
            index=frame.index,
        )
    )
    new_cols.update(
        _rank_raw_candidate_columns(
            absorbed_resistance,
            "absorbed_resistance",
            cfg,
            slots=int(cfg.absorbed_line_output_count),
            index=frame.index,
        )
    )
    new_cols.update(
        _rank_raw_candidate_columns(
            absorbed_support,
            "absorbed_support",
            cfg,
            slots=int(cfg.absorbed_line_output_count),
            index=frame.index,
        )
    )
    new_cols.update(
        _cluster_columns(
            resistance,
            label="fuzzy_resistance",
            close=base["close"],
            proximity_pct=float(cfg.density_proximity_pct),
            min_lines=int(cfg.density_min_lines),
            slots=int(cfg.fuzzy_zone_count),
            prefix=p,
        )
    )
    new_cols.update(
        _cluster_columns(
            support,
            label="fuzzy_support",
            close=base["close"],
            proximity_pct=float(cfg.density_proximity_pct),
            min_lines=int(cfg.density_min_lines),
            slots=int(cfg.fuzzy_zone_count),
            prefix=p,
        )
    )
    new_cols.update(
        _hotspot_columns(
            resistance,
            support,
            close=base["close"],
            proximity_pct=float(cfg.hotspot_proximity_pct),
            min_lines=int(cfg.hotspot_min_lines),
            slots=int(cfg.hotspot_count),
            prefix=p,
        )
    )

    existing = [col for col in frame.columns if str(col).startswith(f"{p}_")]
    clean = frame.drop(columns=existing).copy() if existing else frame.copy()
    return pd.concat([clean, pd.DataFrame(new_cols, index=frame.index)], axis=1)


def build_trendline_projection_v2_diagnostic(
    dataframe: DataFrame,
    config: TrendlineProjectionV2Config | None = None,
    **overrides: object,
) -> TrendlineProjectionV2Diagnostic:
    """Build a static first-pass diagnostic view over the supplied dataframe.

    This helper is for visual review only. It draws all eligible pivot-pair
    candidates inside the provided window, then derives fuzzy zones and hotspots
    from that broad trimmed set. Strategy code should use
    ``add_trendline_projection_v2`` instead.
    """

    cfg = _resolve_config(config, overrides)
    _validate_config(cfg)
    _validate_dataframe(dataframe)

    frame = dataframe.copy()
    base = _base_inputs(frame, cfg)
    candidates = pd.concat(
        [
            _diagnostic_candidates(base, "resistance", cfg),
            _diagnostic_candidates(base, "support", cfg),
        ],
        ignore_index=True,
    )
    structural_candidates = pd.concat(
        [
            _diagnostic_candidates(
                base,
                "resistance",
                cfg,
                max_anchor_bars=int(cfg.structural_max_anchor_bars),
                max_projection_bars=int(cfg.structural_max_projection_bars),
                line_kind="structural_raw",
            ),
            _diagnostic_candidates(
                base,
                "support",
                cfg,
                max_anchor_bars=int(cfg.structural_max_anchor_bars),
                max_projection_bars=int(cfg.structural_max_projection_bars),
                line_kind="structural_raw",
            ),
        ],
        ignore_index=True,
    )
    channel_source_candidates = pd.concat(
        [
            _diagnostic_candidates(
                base,
                "resistance",
                cfg,
                min_anchor_bars=int(cfg.channel_min_anchor_bars),
                max_anchor_bars=int(cfg.channel_max_anchor_bars),
                max_projection_bars=int(cfg.channel_max_projection_bars),
                pivot_family="confirmation",
                min_pivot_prominence_atr=float(cfg.channel_min_pivot_prominence_atr),
                line_kind="channel_source",
            ),
            _diagnostic_candidates(
                base,
                "support",
                cfg,
                min_anchor_bars=int(cfg.channel_min_anchor_bars),
                max_anchor_bars=int(cfg.channel_max_anchor_bars),
                max_projection_bars=int(cfg.channel_max_projection_bars),
                pivot_family="confirmation",
                min_pivot_prominence_atr=float(cfg.channel_min_pivot_prominence_atr),
                line_kind="channel_source",
            ),
        ],
        ignore_index=True,
    )
    raw_channel_candidates = _diagnostic_channel_candidates(candidates, base, cfg, line_kind="raw_channel")
    structural_channel_candidates = _diagnostic_channel_candidates(
        structural_candidates,
        base,
        cfg,
        line_kind="structural_channel",
    )
    channel_candidates = _diagnostic_channel_candidates(channel_source_candidates, base, cfg, line_kind="channel")
    confirmed_candidates = _confirm_candidate_table(candidates, base, cfg)
    absorbed_candidates = _absorb_confirmed_keep_seed_table(confirmed_candidates, base, cfg)
    joined_candidates = confirmed_candidates
    shared_pivot_merged_candidates = absorbed_candidates
    nearby_merged_candidates = {float(cfg.absorb_touch_tolerance_pct): absorbed_candidates}
    fuzzy_res = _diagnostic_cluster_table(
        frame=frame,
        candidates=candidates[candidates["side"].eq("resistance")],
        label="fuzzy_resistance",
        proximity_pct=float(cfg.density_proximity_pct),
        min_lines=int(cfg.density_min_lines),
        slots=int(cfg.fuzzy_zone_count),
    )
    fuzzy_sup = _diagnostic_cluster_table(
        frame=frame,
        candidates=candidates[candidates["side"].eq("support")],
        label="fuzzy_support",
        proximity_pct=float(cfg.density_proximity_pct),
        min_lines=int(cfg.density_min_lines),
        slots=int(cfg.fuzzy_zone_count),
    )
    fuzzy = pd.concat([fuzzy_res, fuzzy_sup], axis=1)
    hotspots = _diagnostic_hotspot_table(
        frame=frame,
        candidates=candidates,
        proximity_pct=float(cfg.hotspot_proximity_pct),
        min_lines=int(cfg.hotspot_min_lines),
        slots=int(cfg.hotspot_count),
    )
    counts = {
        "pivot_high": int(base["pivot_high"].notna().sum()),
        "pivot_low": int(base["pivot_low"].notna().sum()),
        "anchor_clear_raw_resistance": int(candidates["side"].eq("resistance").sum()),
        "anchor_clear_raw_support": int(candidates["side"].eq("support").sum()),
        "raw_resistance": int(candidates["side"].eq("resistance").sum()),
        "raw_support": int(candidates["side"].eq("support").sum()),
        "structural_resistance": int(structural_candidates["side"].eq("resistance").sum()) if not structural_candidates.empty else 0,
        "structural_support": int(structural_candidates["side"].eq("support").sum()) if not structural_candidates.empty else 0,
        "channel_source_resistance": int(channel_source_candidates["side"].eq("resistance").sum()) if not channel_source_candidates.empty else 0,
        "channel_source_support": int(channel_source_candidates["side"].eq("support").sum()) if not channel_source_candidates.empty else 0,
        "raw_channels": int(len(raw_channel_candidates)),
        "structural_channels": int(len(structural_channel_candidates)),
        "channels": int(len(channel_candidates)),
        "confirmed_resistance": int(confirmed_candidates["side"].eq("resistance").sum()) if not confirmed_candidates.empty else 0,
        "confirmed_support": int(confirmed_candidates["side"].eq("support").sum()) if not confirmed_candidates.empty else 0,
        "absorbed_resistance": int(absorbed_candidates["side"].eq("resistance").sum()) if not absorbed_candidates.empty else 0,
        "absorbed_support": int(absorbed_candidates["side"].eq("support").sum()) if not absorbed_candidates.empty else 0,
        "fuzzy_resistance_points": int(fuzzy_res.filter(like="_price_rank0").notna().sum().sum()),
        "fuzzy_support_points": int(fuzzy_sup.filter(like="_price_rank0").notna().sum().sum()),
        "hotspot_points": int(hotspots.filter(like="_price_rank0").notna().sum().sum()),
    }
    for proximity, nearby_candidates in nearby_merged_candidates.items():
        label = f"nearby_merged_{int(round(proximity * 100))}pct"
        counts[f"{label}_resistance"] = (
            int(nearby_candidates["side"].eq("resistance").sum()) if not nearby_candidates.empty else 0
        )
        counts[f"{label}_support"] = int(nearby_candidates["side"].eq("support").sum()) if not nearby_candidates.empty else 0
    base_columns = {key: value for key, value in base.items() if isinstance(value, Series) and key not in frame.columns}
    enriched = pd.concat([frame, pd.DataFrame(base_columns, index=frame.index)], axis=1)
    return TrendlineProjectionV2Diagnostic(
        frame=enriched,
        raw_candidates=candidates,
        structural_candidates=structural_candidates,
        channel_source_candidates=channel_source_candidates,
        raw_channel_candidates=raw_channel_candidates,
        structural_channel_candidates=structural_channel_candidates,
        channel_candidates=channel_candidates,
        confirmed_candidates=confirmed_candidates,
        absorbed_candidates=absorbed_candidates,
        joined_candidates=joined_candidates,
        shared_pivot_merged_candidates=shared_pivot_merged_candidates,
        nearby_merged_candidates=nearby_merged_candidates,
        fuzzy_zones=fuzzy,
        hotspots=hotspots,
        counts=counts,
    )


def _base_inputs(frame: DataFrame, cfg: TrendlineProjectionV2Config) -> dict[str, Series]:
    open_ = _num(frame, "open")
    close = _num(frame, "close")
    high = _num(frame, "high")
    low = _num(frame, "low")
    body_high = pd.concat([open_, close], axis=1).max(axis=1)
    body_low = pd.concat([open_, close], axis=1).min(axis=1)
    atr = _atr(frame, 14)
    bar_index = pd.Series(np.arange(len(frame), dtype="float64"), index=frame.index)
    pivots = _confirmed_body_pivots(
        body_high=body_high,
        body_low=body_low,
        atr=atr,
        bar_index=bar_index,
        strength=int(cfg.pivot_strength),
    )
    confirmation_pivots = _prefixed_pivot_keys(
        _confirmed_body_pivots(
            body_high=body_high,
            body_low=body_low,
            atr=atr,
            bar_index=bar_index,
            strength=int(cfg.confirmation_pivot_strength),
        ),
        "confirm_",
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
        **confirmation_pivots,
    }


def _prefixed_pivot_keys(values: dict[str, Series], prefix: str) -> dict[str, Series]:
    return {f"{prefix}{key}": value for key, value in values.items()}


def _confirmed_body_pivots(
    *,
    body_high: Series,
    body_low: Series,
    atr: Series,
    bar_index: Series,
    strength: int,
) -> dict[str, Series]:
    window = strength * 2 + 1
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
    high_is_pivot = high_candidate.notna() & high_candidate.ge(high_window_max)
    low_is_pivot = low_candidate.notna() & low_candidate.le(low_window_min)

    return {
        "pivot_high": high_candidate.where(high_is_pivot),
        "pivot_low": low_candidate.where(low_is_pivot),
        "pivot_high_index": (bar_index - float(strength)).where(high_is_pivot),
        "pivot_low_index": (bar_index - float(strength)).where(low_is_pivot),
        "pivot_high_prominence": high_prominence.where(high_is_pivot),
        "pivot_low_prominence": low_prominence.where(low_is_pivot),
    }


def _rolling_candidate_pack(
    base: dict[str, Series],
    side: LineSide,
    cfg: TrendlineProjectionV2Config,
    *,
    mode: LineMode,
    min_anchor_bars: int | None = None,
    max_anchor_bars: int | None = None,
    max_projection_bars: int | None = None,
    pivot_family: Literal["primary", "confirmation"] = "primary",
    min_pivot_prominence_atr: float | None = None,
) -> dict[str, list[Series]]:
    if side == "resistance":
        event_price = base["confirm_pivot_high"] if pivot_family == "confirmation" else base["pivot_high"]
        event_index = base["confirm_pivot_high_index"] if pivot_family == "confirmation" else base["pivot_high_index"]
        event_prominence = (
            base["confirm_pivot_high_prominence"] if pivot_family == "confirmation" else base["pivot_high_prominence"]
        )
        confirm_price = base["confirm_pivot_high"]
        confirm_index = base["confirm_pivot_high_index"]
        confirm_prominence = base["confirm_pivot_high_prominence"]
    else:
        event_price = base["confirm_pivot_low"] if pivot_family == "confirmation" else base["pivot_low"]
        event_index = base["confirm_pivot_low_index"] if pivot_family == "confirmation" else base["pivot_low_index"]
        event_prominence = base["confirm_pivot_low_prominence"] if pivot_family == "confirmation" else base["pivot_low_prominence"]
        confirm_price = base["confirm_pivot_low"]
        confirm_index = base["confirm_pivot_low_index"]
        confirm_prominence = base["confirm_pivot_low_prominence"]
    prominence_floor = float(cfg.min_pivot_prominence_atr if min_pivot_prominence_atr is None else min_pivot_prominence_atr)

    recent = _recent_events(
        event_price.where(event_prominence.fillna(0.0).ge(prominence_floor)),
        event_index,
        event_prominence,
        base["close"].index,
        int(cfg.candidate_pivot_count),
    )
    confirmation_recent = _recent_events(
        confirm_price.where(confirm_prominence.fillna(0.0).ge(float(cfg.min_confirmation_pivot_prominence_atr))),
        confirm_index,
        confirm_prominence,
        base["close"].index,
        int(cfg.candidate_pivot_count),
    )
    pack = _empty_pack()
    for newer in range(int(cfg.candidate_pivot_count) - 1):
        for older in range(newer + 1, int(cfg.candidate_pivot_count)):
            candidate = _rolling_candidate_from_pair(
                base,
                recent,
                confirmation_recent,
                newer,
                older,
                cfg,
                mode=mode,
                min_anchor_bars=min_anchor_bars,
                max_anchor_bars=max_anchor_bars,
                max_projection_bars=max_projection_bars,
                min_pivot_prominence_atr=prominence_floor,
            )
            for key, value in candidate.items():
                pack[key].append(value)
    return pack


def _recent_events(
    event_price: Series,
    event_index: Series,
    event_prominence: Series,
    index: pd.Index,
    count: int,
) -> dict[str, list[Series]]:
    prices = event_price.dropna()
    indexes = event_index.where(event_price.notna()).dropna()
    prominences = event_prominence.where(event_price.notna()).dropna()
    return {
        "price": [prices.shift(offset).reindex(index).ffill() for offset in range(count)],
        "index": [indexes.shift(offset).reindex(index).ffill() for offset in range(count)],
        "prominence": [prominences.shift(offset).reindex(index).ffill() for offset in range(count)],
    }


def _rolling_candidate_from_pair(
    base: dict[str, Series],
    recent: dict[str, list[Series]],
    confirmation_recent: dict[str, list[Series]],
    newer: int,
    older: int,
    cfg: TrendlineProjectionV2Config,
    *,
    mode: LineMode,
    min_anchor_bars: int | None = None,
    max_anchor_bars: int | None = None,
    max_projection_bars: int | None = None,
    min_pivot_prominence_atr: float | None = None,
) -> dict[str, Series]:
    p_new = recent["price"][newer]
    p_old = recent["price"][older]
    x_new = recent["index"][newer]
    x_old = recent["index"][older]
    prom_new = recent["prominence"][newer]
    prom_old = recent["prominence"][older]
    close = base["close"]
    bar_index = base["bar_index"]
    anchor_min = float(cfg.min_anchor_bars if min_anchor_bars is None else min_anchor_bars)
    anchor_cap = float(cfg.max_anchor_bars if max_anchor_bars is None else max_anchor_bars)
    projection_cap = float(cfg.max_projection_bars if max_projection_bars is None else max_projection_bars)
    prominence_floor = float(cfg.min_pivot_prominence_atr if min_pivot_prominence_atr is None else min_pivot_prominence_atr)

    span = x_new - x_old
    prominence = pd.concat([prom_new, prom_old], axis=1).mean(axis=1)
    valid_anchor = (
        p_new.notna()
        & p_old.notna()
        & x_new.notna()
        & x_old.notna()
        & span.ge(anchor_min)
        & span.le(anchor_cap)
    )
    slope = _safe_div(p_new - p_old, span).where(valid_anchor)
    intercept = p_new - slope * x_new
    line = (intercept + slope * bar_index).clip(lower=0.0)
    slope_pct = (slope.abs() / close.abs().replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    projection_end = x_new + projection_cap
    pivot_count = pd.Series(2.0, index=close.index, dtype="float64")
    absorbed_pivot_count = pivot_count.copy()
    line_width = pd.Series(np.nan, index=close.index, dtype="float64")
    last_confirm_index = x_new.copy()
    if mode in {"confirmed", "absorbed"}:
        confirmation = _rolling_line_touch_metrics(
            base,
            confirmation_recent,
            x_new=x_new,
            slope=slope,
            intercept=intercept,
            price_tolerance_pct=float(cfg.confirmation_tolerance_pct),
            atr_tolerance_mult=float(cfg.confirmation_atr_mult),
            max_confirm_bars=int(cfg.confirmation_max_bars),
        )
        pivot_count = pivot_count + confirmation["touch_count"]
        absorbed_pivot_count = pivot_count.copy()
        last_confirm_index = confirmation["last_touch_index"].where(
            confirmation["touch_count"].gt(0.0),
            x_new,
        )
        line_width = confirmation["line_width"] * float(cfg.absorb_width_scale)
        projection_end = last_confirm_index + projection_cap
        if mode == "absorbed":
            absorbed = _rolling_line_touch_metrics(
                base,
                confirmation_recent,
                x_new=x_new,
                slope=slope,
                intercept=intercept,
                price_tolerance_pct=float(cfg.absorb_touch_tolerance_pct),
                atr_tolerance_mult=float(cfg.confirmation_atr_mult),
                max_confirm_bars=int(cfg.confirmation_max_bars),
            )
            absorbed_pivot_count = 2.0 + absorbed["touch_count"]
            line_width = absorbed["line_width"] * float(cfg.absorb_width_scale)
    in_segment = bar_index.ge(x_old) & bar_index.le(projection_end)
    valid = (
        valid_anchor
        & prominence.fillna(0.0).ge(prominence_floor)
        & slope_pct.le(float(cfg.max_slope_pct_per_bar))
        & in_segment
    ).fillna(False)
    if mode in {"confirmed", "absorbed"}:
        valid &= pivot_count.ge(float(cfg.confirmation_min_pivots))

    span_score = _clip01(span / max(anchor_min * 8.0, 1.0))
    slope_score = _clip01(1.0 - slope_pct / max(float(cfg.max_slope_pct_per_bar), 1e-9))
    prom_score = _clip01(prominence / max(prominence_floor * 5.0, 1.0))
    age_score = _clip01(1.0 - (bar_index - x_new) / max(projection_cap, 1.0))
    confirmation_score = _clip01((pivot_count - 2.0) / 4.0)
    absorption_score = _clip01((absorbed_pivot_count - pivot_count) / 4.0)
    score = _clip01(
        0.30 * span_score
        + 0.20 * slope_score
        + 0.22 * prom_score
        + 0.10 * age_score
        + 0.14 * confirmation_score
        + 0.04 * absorption_score
    ).where(valid)
    line_id = (x_old * 100_000.0 + x_new).where(valid)
    return {
        "line": line.where(valid),
        "score": score,
        "slope": slope.where(valid),
        "slope_pct": slope_pct.where(valid),
        "anchor_old_index": x_old.where(valid),
        "anchor_new_index": x_new.where(valid),
        "projection_end_index": projection_end.where(valid),
        "line_id": line_id,
        "pivot_count": pivot_count.where(valid),
        "absorbed_pivot_count": absorbed_pivot_count.where(valid),
        "line_width": line_width.where(valid),
        "last_confirm_index": last_confirm_index.where(valid),
    }


def _rolling_line_touch_metrics(
    base: dict[str, Series],
    confirmation_recent: dict[str, list[Series]],
    *,
    x_new: Series,
    slope: Series,
    intercept: Series,
    price_tolerance_pct: float,
    atr_tolerance_mult: float,
    max_confirm_bars: int,
) -> dict[str, Series]:
    index = base["close"].index
    touch_count = pd.Series(0.0, index=index, dtype="float64")
    last_touch_index = x_new.copy()
    line_width = pd.Series(0.0, index=index, dtype="float64")
    atr_tolerance = base["atr"].fillna(0.0) * float(atr_tolerance_mult)
    for pivot_price, pivot_index in zip(confirmation_recent["price"], confirmation_recent["index"], strict=False):
        line_at_pivot = intercept + slope * pivot_index
        error = (pivot_price - line_at_pivot).abs()
        tolerance = pd.concat(
            [
                pivot_price.abs() * float(price_tolerance_pct),
                atr_tolerance,
            ],
            axis=1,
        ).max(axis=1)
        touches = (
            pivot_price.notna()
            & pivot_index.notna()
            & pivot_index.gt(x_new)
            & pivot_index.le(x_new + float(max_confirm_bars))
            & error.le(tolerance)
        ).fillna(False)
        touch_count = touch_count + touches.astype("float64")
        last_touch_index = last_touch_index.where(~touches | pivot_index.le(last_touch_index), pivot_index)
        line_width = pd.concat([line_width, error.where(touches).fillna(0.0)], axis=1).max(axis=1)
    return {
        "touch_count": touch_count,
        "last_touch_index": last_touch_index,
        "line_width": line_width.replace(0.0, np.nan),
    }


def _rank_raw_candidate_columns(
    pack: dict[str, list[Series]],
    side: str,
    cfg: TrendlineProjectionV2Config,
    *,
    slots: int,
    index: pd.Index | None = None,
) -> dict[str, Series]:
    p = cfg.output_prefix
    index = index if index is not None else (pack["line"][0].index if pack["line"] else pd.RangeIndex(0))
    out: dict[str, Series] = {}
    if not pack["line"]:
        for rank in range(slots):
            out.update(_empty_line_columns(index, p, side, rank))
        return out

    score_values = _series_matrix(pack["score"])
    score_values = np.where(np.isfinite(score_values), score_values, -np.inf)
    metric_values = {key: _series_matrix(pack[key]) for key in pack if key != "score"}
    remaining = score_values.copy()
    rows = np.arange(len(index))
    for rank in range(slots):
        chosen = np.argmax(remaining, axis=1)
        chosen_score = remaining[rows, chosen]
        valid = np.isfinite(chosen_score) & (chosen_score > -np.inf)
        line = _select_metric(metric_values["line"], chosen, valid)
        line_id = _select_metric(metric_values["line_id"], chosen, valid)
        out.update(
            {
                f"{p}_{side}_line_rank{rank}": pd.Series(line, index=index, dtype="float64"),
                f"{p}_{side}_plot_rank{rank}": _plot_safe_line(
                    pd.Series(line, index=index, dtype="float64"),
                    pd.Series(line_id, index=index, dtype="float64"),
                    float(cfg.plot_break_on_line_change_pct),
                ),
                f"{p}_{side}_score_rank{rank}": pd.Series(np.where(valid, chosen_score, np.nan), index=index, dtype="float64"),
            }
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
            out[f"{p}_{side}_{metric}_rank{rank}"] = pd.Series(
                _select_metric(metric_values[metric], chosen, valid),
                index=index,
                dtype="float64",
            )
        remaining[rows, chosen] = -np.inf
    return out


def _empty_line_columns(index: pd.Index, prefix: str, side: str, rank: int) -> dict[str, Series]:
    nan = pd.Series(np.nan, index=index, dtype="float64")
    return {
        f"{prefix}_{side}_line_rank{rank}": nan,
        f"{prefix}_{side}_plot_rank{rank}": nan,
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


def _channel_columns_from_ranked(
    line_columns: dict[str, Series],
    base: dict[str, Series],
    cfg: TrendlineProjectionV2Config,
    *,
    source_label: str,
    output_label: str,
    source_count: int | None = None,
) -> dict[str, Series]:
    pack = _empty_channel_pack()
    count = int(cfg.channel_source_line_count if source_count is None else source_count)
    for resistance_rank in range(count):
        for support_rank in range(count):
            candidate = _rolling_channel_from_ranked_pair(
                line_columns,
                base,
                cfg,
                source_label=source_label,
                resistance_rank=resistance_rank,
                support_rank=support_rank,
            )
            for key, value in candidate.items():
                pack[key].append(value)
    return _rank_channel_columns(pack, cfg, index=base["close"].index, label=output_label)


def _alias_channel_columns(
    columns: dict[str, Series],
    prefix: str,
    *,
    source_label: str,
    alias_label: str,
) -> dict[str, Series]:
    source_token = f"{prefix}_{source_label}_"
    alias_token = f"{prefix}_{alias_label}_"
    return {name.replace(source_token, alias_token, 1): value for name, value in columns.items()}


def _rolling_channel_from_ranked_pair(
    line_columns: dict[str, Series],
    base: dict[str, Series],
    cfg: TrendlineProjectionV2Config,
    *,
    source_label: str,
    resistance_rank: int,
    support_rank: int,
) -> dict[str, Series]:
    index = base["close"].index
    p = cfg.output_prefix

    def get(name: str) -> Series:
        return line_columns.get(name, pd.Series(np.nan, index=index, dtype="float64"))

    res = f"{p}_{source_label}_resistance"
    sup = f"{p}_{source_label}_support"
    upper = get(f"{res}_line_rank{resistance_rank}")
    lower = get(f"{sup}_line_rank{support_rank}")
    res_slope = get(f"{res}_slope_rank{resistance_rank}")
    sup_slope = get(f"{sup}_slope_rank{support_rank}")
    res_score = get(f"{res}_score_rank{resistance_rank}")
    sup_score = get(f"{sup}_score_rank{support_rank}")
    res_old = get(f"{res}_anchor_old_index_rank{resistance_rank}")
    sup_old = get(f"{sup}_anchor_old_index_rank{support_rank}")
    res_new = get(f"{res}_anchor_new_index_rank{resistance_rank}")
    sup_new = get(f"{sup}_anchor_new_index_rank{support_rank}")
    res_end = get(f"{res}_projection_end_index_rank{resistance_rank}")
    sup_end = get(f"{sup}_projection_end_index_rank{support_rank}")
    res_pivots = get(f"{res}_pivot_count_rank{resistance_rank}")
    sup_pivots = get(f"{sup}_pivot_count_rank{support_rank}")

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
    width_change_pct = ((span_end_width - span_start_width) / span_start_width.abs().replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    parallel_score = _clip01(1.0 - width_change_pct.abs() / max(float(cfg.channel_parallel_width_change_pct), 1e-9))
    convergence_score = _clip01((-width_change_pct - float(cfg.channel_min_convergence_pct)) / max(float(cfg.channel_parallel_width_change_pct), 1e-9))
    relation_score = pd.concat([parallel_score, convergence_score], axis=1).max(axis=1)
    channel_shape = pd.Series(0.0, index=index, dtype="float64").where(
        convergence_score.le(parallel_score),
        1.0,
    )
    channel_shape = channel_shape.where(width_change_pct.le(float(cfg.channel_parallel_width_change_pct)), -1.0)

    line_score = pd.concat([res_score, sup_score], axis=1).mean(axis=1)
    overlap_score = _clip01(overlap_bars / max(float(cfg.channel_min_overlap_bars) * 3.0, 1.0))
    slope_score = _clip01(1.0 - slope_diff_pct / max(float(cfg.channel_slope_tolerance_pct), 1e-9))
    width_score = _clip01(1.0 - (width_pct - 0.08).abs() / 0.08)
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
    return {
        "upper": upper.where(valid),
        "lower": lower.where(valid),
        "mid": ((upper + lower) / 2.0).where(valid),
        "width": width.where(valid),
        "width_pct": width_pct.where(valid),
        "position": position.where(valid),
        "score": score.where(valid),
        "slope_upper": res_slope.where(valid),
        "slope_lower": sup_slope.where(valid),
        "slope_diff_pct": slope_diff_pct.where(valid),
        "width_change_pct": width_change_pct.where(valid),
        "shape": channel_shape.where(valid),
        "overlap_start_index": overlap_start.where(valid),
        "overlap_end_index": overlap_end.where(valid),
        "overlap_bars": overlap_bars.where(valid),
        "source_resistance_slot": pd.Series(float(resistance_rank), index=index).where(valid),
        "source_support_slot": pd.Series(float(support_rank), index=index).where(valid),
        "breakout_up": close.gt(upper + tolerance).astype("float64").where(valid),
        "breakdown_down": close.lt(lower - tolerance).astype("float64").where(valid),
        "near_upper": (upper - close).abs().le(tolerance).astype("float64").where(valid),
        "near_lower": (close - lower).abs().le(tolerance).astype("float64").where(valid),
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
    cfg: TrendlineProjectionV2Config,
    *,
    index: pd.Index,
    label: str,
) -> dict[str, Series]:
    p = cfg.output_prefix
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
        for metric in (
            "upper",
            "lower",
            "mid",
            "width",
            "width_pct",
            "position",
            "slope_upper",
            "slope_lower",
            "slope_diff_pct",
            "width_change_pct",
            "shape",
            "overlap_start_index",
            "overlap_end_index",
            "overlap_bars",
            "source_resistance_slot",
            "source_support_slot",
            "breakout_up",
            "breakdown_down",
            "near_upper",
            "near_lower",
        ):
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
    cfg: TrendlineProjectionV2Config,
) -> np.ndarray:
    """Row-wise duplicate suppression for strategy-facing ranked channels."""

    if "mid" not in metric_values or metric_values["mid"].size == 0:
        return np.zeros((len(valid), 0), dtype=bool)

    rows = np.arange(len(valid))
    start = metric_values["overlap_start_index"]
    end = metric_values["overlap_end_index"]
    mid = metric_values["mid"]
    width = metric_values["width"]
    upper_slope = metric_values["slope_upper"]
    lower_slope = metric_values["slope_lower"]

    chosen_start = start[rows, chosen][:, np.newaxis]
    chosen_end = end[rows, chosen][:, np.newaxis]
    chosen_mid = mid[rows, chosen][:, np.newaxis]
    chosen_width = width[rows, chosen][:, np.newaxis]
    chosen_upper_slope = upper_slope[rows, chosen][:, np.newaxis]
    chosen_lower_slope = lower_slope[rows, chosen][:, np.newaxis]

    overlap = np.minimum(end, chosen_end) - np.maximum(start, chosen_start)
    span = np.maximum(end - start, 1.0)
    chosen_span = np.maximum(chosen_end - chosen_start, 1.0)
    overlap_ratio = overlap / np.maximum(np.minimum(span, chosen_span), 1.0)
    width_ref = np.maximum(np.minimum(width, chosen_width), 1e-9)
    mid_close = np.abs(mid - chosen_mid) <= width_ref * float(cfg.channel_duplicate_mid_width_mult)
    upper_slope_close = _relative_array_diff(upper_slope, chosen_upper_slope) <= float(cfg.channel_duplicate_slope_tolerance_pct)
    lower_slope_close = _relative_array_diff(lower_slope, chosen_lower_slope) <= float(cfg.channel_duplicate_slope_tolerance_pct)
    finite = np.isfinite(start) & np.isfinite(end) & np.isfinite(mid) & np.isfinite(width)
    return (
        valid[:, np.newaxis]
        & finite
        & (overlap_ratio >= float(cfg.channel_duplicate_overlap_pct))
        & mid_close
        & upper_slope_close
        & lower_slope_close
    )


def _relative_array_diff(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominator = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1e-9)
    return np.abs(left - right) / denominator


def _empty_channel_pack() -> dict[str, list[Series]]:
    return {
        "upper": [],
        "lower": [],
        "mid": [],
        "width": [],
        "width_pct": [],
        "position": [],
        "score": [],
        "slope_upper": [],
        "slope_lower": [],
        "slope_diff_pct": [],
        "width_change_pct": [],
        "shape": [],
        "overlap_start_index": [],
        "overlap_end_index": [],
        "overlap_bars": [],
        "source_resistance_slot": [],
        "source_support_slot": [],
        "breakout_up": [],
        "breakdown_down": [],
        "near_upper": [],
        "near_lower": [],
    }


def _empty_channel_columns(index: pd.Index, prefix: str, slots: int, *, label: str) -> dict[str, Series]:
    out: dict[str, Series] = {}
    nan = pd.Series(np.nan, index=index, dtype="float64")
    for rank in range(slots):
        out[f"{prefix}_{label}_score_rank{rank}"] = nan
        for metric in (
            "upper",
            "lower",
            "mid",
            "width",
            "width_pct",
            "position",
            "slope_upper",
            "slope_lower",
            "slope_diff_pct",
            "width_change_pct",
            "shape",
            "overlap_start_index",
            "overlap_end_index",
            "overlap_bars",
            "source_resistance_slot",
            "source_support_slot",
            "breakout_up",
            "breakdown_down",
            "near_upper",
            "near_lower",
        ):
            out[f"{prefix}_{label}_{metric}_rank{rank}"] = nan
    return out


def _cluster_columns(
    pack: dict[str, list[Series]],
    *,
    label: str,
    close: Series,
    proximity_pct: float,
    min_lines: int,
    slots: int,
    prefix: str,
) -> dict[str, Series]:
    if not pack["line"]:
        return _empty_cluster_columns(close.index, prefix, label, slots, include_bias=False)

    line_values = _series_matrix(pack["line"])
    score_values = _series_matrix(pack["score"])
    clusters = _cluster_matrix_rows(
        line_values=line_values,
        score_values=score_values,
        close=close.to_numpy(dtype="float64"),
        proximity_pct=proximity_pct,
        min_lines=min_lines,
        slots=slots,
    )
    return _cluster_output_columns(clusters, close.index, prefix, label, include_bias=False)


def _hotspot_columns(
    resistance: dict[str, list[Series]],
    support: dict[str, list[Series]],
    *,
    close: Series,
    proximity_pct: float,
    min_lines: int,
    slots: int,
    prefix: str,
) -> dict[str, Series]:
    if not resistance["line"] and not support["line"]:
        return _empty_cluster_columns(close.index, prefix, "hotspot", slots, include_bias=True)

    res_lines = _series_matrix(resistance["line"]) if resistance["line"] else np.empty((len(close), 0))
    sup_lines = _series_matrix(support["line"]) if support["line"] else np.empty((len(close), 0))
    res_scores = _series_matrix(resistance["score"]) if resistance["score"] else np.empty((len(close), 0))
    sup_scores = _series_matrix(support["score"]) if support["score"] else np.empty((len(close), 0))
    line_values = np.concatenate([res_lines, sup_lines], axis=1)
    score_values = np.concatenate([res_scores, sup_scores], axis=1)
    labels = np.concatenate([np.ones(res_lines.shape[1]), -np.ones(sup_lines.shape[1])])
    clusters = _cluster_matrix_rows(
        line_values=line_values,
        score_values=score_values,
        close=close.to_numpy(dtype="float64"),
        proximity_pct=proximity_pct,
        min_lines=min_lines,
        slots=slots,
        labels=labels,
    )
    return _cluster_output_columns(clusters, close.index, prefix, "hotspot", include_bias=True)


def _cluster_matrix_rows(
    *,
    line_values: np.ndarray,
    score_values: np.ndarray,
    close: np.ndarray,
    proximity_pct: float,
    min_lines: int,
    slots: int,
    labels: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    rows = line_values.shape[0]
    out = {
        "price": np.full((rows, slots), np.nan, dtype="float64"),
        "strength": np.full((rows, slots), np.nan, dtype="float64"),
        "line_count": np.full((rows, slots), np.nan, dtype="float64"),
        "width": np.full((rows, slots), np.nan, dtype="float64"),
        "side_bias": np.full((rows, slots), np.nan, dtype="float64"),
    }
    # This bounded NumPy row loop avoids materializing a huge rows x lines x lines
    # proximity cube while still keeping all dataframe calculations columnar.
    for row in range(rows):
        values = line_values[row]
        finite = np.isfinite(values)
        if finite.sum() < min_lines:
            continue
        width = abs(close[row]) * proximity_pct if np.isfinite(close[row]) else np.nan
        if not np.isfinite(width) or width <= 0.0:
            continue
        row_labels = labels[finite] if labels is not None else None
        clusters = _line_value_clusters(
            values=values[finite],
            scores=score_values[row][finite],
            width=width,
            min_lines=min_lines,
            labels=row_labels,
            slots=slots,
        )
        for slot, cluster in enumerate(clusters):
            out["price"][row, slot] = cluster["price"]
            out["strength"][row, slot] = cluster["strength"]
            out["line_count"][row, slot] = cluster["line_count"]
            out["width"][row, slot] = cluster["width"]
            out["side_bias"][row, slot] = cluster["side_bias"]
    return out


def _line_value_clusters(
    *,
    values: np.ndarray,
    scores: np.ndarray,
    width: float,
    min_lines: int,
    labels: np.ndarray | None,
    slots: int,
) -> list[dict[str, float]]:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_scores = np.nan_to_num(scores[order], nan=0.0, posinf=0.0, neginf=0.0)
    sorted_labels = labels[order] if labels is not None else None
    found: list[dict[str, float]] = []
    for start in range(len(sorted_values)):
        end = np.searchsorted(sorted_values, sorted_values[start] + width, side="right")
        count = end - start
        if count < min_lines:
            continue
        cluster_values = sorted_values[start:end]
        cluster_scores = sorted_scores[start:end]
        weights = np.maximum(cluster_scores, 0.01)
        price = float(np.average(cluster_values, weights=weights))
        cluster_width = float(max(cluster_values[-1] - cluster_values[0], width * 0.25))
        mean_score = float(np.mean(cluster_scores))
        strength = float(count * (0.5 + mean_score))
        side_bias = 0.0
        if sorted_labels is not None:
            side_bias = float(np.mean(sorted_labels[start:end]))
        found.append(
            {
                "price": price,
                "strength": strength,
                "line_count": float(count),
                "width": cluster_width,
                "side_bias": side_bias,
            }
        )
    found.sort(key=lambda item: (item["strength"], item["line_count"]), reverse=True)
    selected: list[dict[str, float]] = []
    for cluster in found:
        if all(abs(cluster["price"] - kept["price"]) > width for kept in selected):
            selected.append(cluster)
        if len(selected) >= slots:
            break
    return selected


def _cluster_output_columns(
    clusters: dict[str, np.ndarray],
    index: pd.Index,
    prefix: str,
    label: str,
    *,
    include_bias: bool,
) -> dict[str, Series]:
    out: dict[str, Series] = {}
    slots = clusters["price"].shape[1]
    for rank in range(slots):
        price = pd.Series(clusters["price"][:, rank], index=index, dtype="float64")
        width = pd.Series(clusters["width"][:, rank], index=index, dtype="float64")
        out[f"{prefix}_{label}_price_rank{rank}"] = price
        out[f"{prefix}_{label}_upper_rank{rank}"] = price + width / 2.0
        out[f"{prefix}_{label}_lower_rank{rank}"] = price - width / 2.0
        out[f"{prefix}_{label}_strength_rank{rank}"] = pd.Series(clusters["strength"][:, rank], index=index, dtype="float64")
        out[f"{prefix}_{label}_line_count_rank{rank}"] = pd.Series(clusters["line_count"][:, rank], index=index, dtype="float64")
        out[f"{prefix}_{label}_width_rank{rank}"] = width
        if include_bias:
            out[f"{prefix}_{label}_side_bias_rank{rank}"] = pd.Series(clusters["side_bias"][:, rank], index=index, dtype="float64")
    return out


def _empty_cluster_columns(index: pd.Index, prefix: str, label: str, slots: int, *, include_bias: bool) -> dict[str, Series]:
    clusters = {
        "price": np.full((len(index), slots), np.nan),
        "strength": np.full((len(index), slots), np.nan),
        "line_count": np.full((len(index), slots), np.nan),
        "width": np.full((len(index), slots), np.nan),
        "side_bias": np.full((len(index), slots), np.nan),
    }
    return _cluster_output_columns(clusters, index, prefix, label, include_bias=include_bias)


def _diagnostic_candidates(
    base: dict[str, Series],
    side: LineSide,
    cfg: TrendlineProjectionV2Config,
    *,
    min_anchor_bars: int | None = None,
    max_anchor_bars: int | None = None,
    max_projection_bars: int | None = None,
    pivot_family: Literal["primary", "confirmation"] = "primary",
    min_pivot_prominence_atr: float | None = None,
    line_kind: str = "raw",
) -> DataFrame:
    if side == "resistance":
        pivot_price = base["confirm_pivot_high"] if pivot_family == "confirmation" else base["pivot_high"]
        pivot_index = base["confirm_pivot_high_index"] if pivot_family == "confirmation" else base["pivot_high_index"]
        pivot_prominence = (
            base["confirm_pivot_high_prominence"] if pivot_family == "confirmation" else base["pivot_high_prominence"]
        )
    else:
        pivot_price = base["confirm_pivot_low"] if pivot_family == "confirmation" else base["pivot_low"]
        pivot_index = base["confirm_pivot_low_index"] if pivot_family == "confirmation" else base["pivot_low_index"]
        pivot_prominence = base["confirm_pivot_low_prominence"] if pivot_family == "confirmation" else base["pivot_low_prominence"]

    prominence_floor = float(cfg.min_pivot_prominence_atr if min_pivot_prominence_atr is None else min_pivot_prominence_atr)
    allowed = pivot_price.notna() & pivot_prominence.fillna(0.0).ge(prominence_floor)
    prices = pivot_price.where(allowed).dropna().to_numpy(dtype="float64")
    indexes = pivot_index.where(allowed).dropna().to_numpy(dtype="float64")
    prominences = pivot_prominence.where(allowed).dropna().to_numpy(dtype="float64")
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
            projection_end = min(float(len(base["close"]) - 1), x_new + projection_cap)
            if projection_cap > float(cfg.max_projection_bars):
                projection_end = float(
                    _projection_end_after_sustained_break(
                        base=base,
                        side=side,
                        slope=float(slope),
                        intercept=float(intercept),
                        start_index=int(round(x_new)),
                        max_end=int(round(projection_end)),
                        cfg=cfg,
                    )
                )
            span_score = min(span / max(anchor_min * 10.0, 1.0), 1.0)
            slope_score = max(0.0, 1.0 - slope_pct / max(float(cfg.max_slope_pct_per_bar), 1e-9))
            prom_score = min(prominence / max(prominence_floor * 5.0, 1.0), 1.0)
            score = 0.42 * span_score + 0.28 * slope_score + 0.30 * prom_score
            rows.append(
                {
                    "side": side,
                    "x_old": float(x_old),
                    "x_new": float(x_new),
                    "y_old": float(p_old),
                    "y_new": float(p_new),
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "projection_end": float(projection_end),
                    "span": float(span),
                    "slope_pct": float(slope_pct),
                    "prominence": prominence,
                    "score": float(score),
                    "pivot_count": 2.0,
                    "pivot_path": f"{int(round(x_old))}|{int(round(x_new))}",
                    "price_path": f"{float(p_old)}|{float(p_new)}",
                    "source_count": 1.0,
                    "line_kind": line_kind,
                }
            )
    return pd.DataFrame(rows)


def _diagnostic_channel_candidates(
    candidates: DataFrame,
    base: dict[str, Series],
    cfg: TrendlineProjectionV2Config,
    *,
    line_kind: str,
) -> DataFrame:
    columns = [
        "line_kind",
        "res_pivot_path",
        "sup_pivot_path",
        "start",
        "end",
        "overlap_bars",
        "upper_slope",
        "lower_slope",
        "upper_intercept",
        "lower_intercept",
        "width_start",
        "width_end",
        "width_mid",
        "width_pct_mid",
        "width_change_pct",
        "shape",
        "slope_diff_pct",
        "containment_ratio",
        "upper_touch_count",
        "lower_touch_count",
        "upper_recent_touch_count",
        "lower_recent_touch_count",
        "touch_balance_score",
        "recent_touch_score",
        "position_coverage",
        "empty_space_score",
        "pivot_touch_count",
        "score",
    ]
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    pool_size = int(cfg.channel_candidate_pool_size)
    resistance = candidates[candidates["side"].eq("resistance")].sort_values(["score", "span"], ascending=False).head(pool_size)
    support = candidates[candidates["side"].eq("support")].sort_values(["score", "span"], ascending=False).head(pool_size)
    rows: list[dict[str, float | str]] = []
    for _, upper in resistance.iterrows():
        for _, lower in support.iterrows():
            start = int(round(max(float(upper["x_new"]), float(lower["x_new"]))))
            end = int(round(min(float(upper["projection_end"]), float(lower["projection_end"]))))
            overlap = end - start
            if overlap < int(cfg.channel_min_overlap_bars):
                continue
            metrics = _diagnostic_channel_metrics(upper, lower, base, start, end, cfg)
            if metrics is None:
                continue
            if metrics["width_pct_mid"] < float(cfg.channel_min_width_pct) or metrics["width_pct_mid"] > float(cfg.channel_max_width_pct):
                continue
            if metrics["containment_ratio"] < float(cfg.channel_min_containment_ratio):
                continue
            if metrics["pivot_touch_count"] < int(cfg.channel_min_pivot_touches):
                continue
            if metrics["upper_touch_count"] < int(cfg.channel_min_rail_touches):
                continue
            if metrics["lower_touch_count"] < int(cfg.channel_min_rail_touches):
                continue
            if metrics["upper_recent_touch_count"] < int(cfg.channel_min_recent_rail_touches):
                continue
            if metrics["lower_recent_touch_count"] < int(cfg.channel_min_recent_rail_touches):
                continue
            if metrics["position_coverage"] < float(cfg.channel_min_position_coverage):
                continue
            relation_score = max(metrics["parallel_score"], metrics["convergence_score"])
            if relation_score <= 0.0:
                continue
            overlap_score = min(overlap / max(float(cfg.channel_min_overlap_bars) * 3.0, 1.0), 1.0)
            touch_score = min(metrics["pivot_touch_count"] / 10.0, 1.0)
            line_score = (float(upper["score"]) + float(lower["score"])) / 2.0
            tight_score = max(
                0.0,
                1.0
                - (metrics["width_pct_mid"] - float(cfg.channel_min_width_pct))
                / max(float(cfg.channel_max_width_pct) - float(cfg.channel_min_width_pct), 1e-9),
            )
            score = (
                0.16 * metrics["containment_ratio"]
                + 0.15 * touch_score
                + 0.15 * metrics["touch_balance_score"]
                + 0.12 * metrics["recent_touch_score"]
                + 0.14 * metrics["empty_space_score"]
                + 0.12 * relation_score
                + 0.07 * overlap_score
                + 0.05 * line_score
                + 0.04 * tight_score
            )
            rows.append(
                {
                    "line_kind": line_kind,
                    "res_pivot_path": str(upper["pivot_path"]),
                    "sup_pivot_path": str(lower["pivot_path"]),
                    "start": float(start),
                    "end": float(end),
                    "overlap_bars": float(overlap),
                    "upper_slope": float(upper["slope"]),
                    "lower_slope": float(lower["slope"]),
                    "upper_intercept": float(upper["intercept"]),
                    "lower_intercept": float(lower["intercept"]),
                    "width_start": metrics["width_start"],
                    "width_end": metrics["width_end"],
                    "width_mid": metrics["width_mid"],
                    "width_pct_mid": metrics["width_pct_mid"],
                    "width_change_pct": metrics["width_change_pct"],
                    "shape": metrics["shape"],
                    "slope_diff_pct": metrics["slope_diff_pct"],
                    "containment_ratio": metrics["containment_ratio"],
                    "upper_touch_count": metrics["upper_touch_count"],
                    "lower_touch_count": metrics["lower_touch_count"],
                    "upper_recent_touch_count": metrics["upper_recent_touch_count"],
                    "lower_recent_touch_count": metrics["lower_recent_touch_count"],
                    "touch_balance_score": metrics["touch_balance_score"],
                    "recent_touch_score": metrics["recent_touch_score"],
                    "position_coverage": metrics["position_coverage"],
                    "empty_space_score": metrics["empty_space_score"],
                    "pivot_touch_count": metrics["pivot_touch_count"],
                    "score": float(score),
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    ranked = pd.DataFrame(rows, columns=columns).sort_values("score", ascending=False).reset_index(drop=True)
    return _suppress_duplicate_channel_rows(ranked, cfg)


def _diagnostic_channel_metrics(
    upper: pd.Series,
    lower: pd.Series,
    base: dict[str, Series],
    start: int,
    end: int,
    cfg: TrendlineProjectionV2Config,
) -> dict[str, float] | None:
    if end <= start or start < 0 or end >= len(base["close"]):
        return None
    xs = np.arange(start, end + 1, dtype="float64")
    upper_line = float(upper["intercept"]) + float(upper["slope"]) * xs
    lower_line = float(lower["intercept"]) + float(lower["slope"]) * xs
    width = upper_line - lower_line
    if np.any(~np.isfinite(width)) or np.any(width <= 0.0):
        return None
    mid_idx = int(round((start + end) / 2.0))
    close_ref = max(abs(float(base["close"].iloc[mid_idx])), 1e-9)
    width_start = float(width[0])
    width_end = float(width[-1])
    width_mid = float(width[len(width) // 2])
    width_pct_mid = width_mid / close_ref
    width_change_pct = (width_end - width_start) / max(abs(width_start), 1e-9)
    slope_denom = max(abs(float(upper["slope"])), abs(float(lower["slope"])), 1e-9)
    slope_diff_pct = abs(float(upper["slope"]) - float(lower["slope"])) / slope_denom
    parallel_score = min(1.0, max(0.0, 1.0 - abs(width_change_pct) / max(float(cfg.channel_parallel_width_change_pct), 1e-9)))
    convergence_score = min(
        1.0,
        max(0.0, (-width_change_pct - float(cfg.channel_min_convergence_pct)) / max(float(cfg.channel_parallel_width_change_pct), 1e-9)),
    )
    if convergence_score > parallel_score:
        shape = 1.0
    elif width_change_pct > float(cfg.channel_parallel_width_change_pct):
        shape = -1.0
    else:
        shape = 0.0

    body_high = base["body_high"].iloc[start : end + 1].to_numpy(dtype="float64")
    body_low = base["body_low"].iloc[start : end + 1].to_numpy(dtype="float64")
    atr = base["atr"].iloc[start : end + 1].fillna(0.0).to_numpy(dtype="float64")
    tolerance = np.maximum(np.abs((upper_line + lower_line) / 2.0) * float(cfg.channel_touch_tolerance_pct), atr * float(cfg.channel_touch_atr_mult))
    contained = (body_high <= upper_line + tolerance) & (body_low >= lower_line - tolerance)
    sample_end = len(contained)
    trimmed = 0
    while sample_end > 1 and trimmed < int(cfg.channel_breakout_grace_bars) and not bool(contained[sample_end - 1]):
        sample_end -= 1
        trimmed += 1
    containment_sample = contained[:sample_end]
    containment_ratio = float(np.mean(containment_sample)) if len(containment_sample) else 0.0
    close_values = base["close"].iloc[start : end + 1].to_numpy(dtype="float64")
    position = (close_values - lower_line) / np.maximum(width, 1e-9)
    position_sample = position[:sample_end]
    position_sample = position_sample[np.isfinite(position_sample)]
    if len(position_sample):
        clipped_position = np.clip(position_sample, 0.0, 1.0)
        position_coverage = float(np.quantile(clipped_position, 0.90) - np.quantile(clipped_position, 0.10))
    else:
        position_coverage = 0.0

    high_pivot = base["confirm_pivot_high"].combine_first(base["pivot_high"]).iloc[start : end + 1].to_numpy(dtype="float64")
    low_pivot = base["confirm_pivot_low"].combine_first(base["pivot_low"]).iloc[start : end + 1].to_numpy(dtype="float64")
    high_valid = np.isfinite(high_pivot)
    low_valid = np.isfinite(low_pivot)
    active_indexes = np.arange(start, end + 1, dtype="int64")
    upper_overlap_touches = {
        int(active_indexes[pos])
        for pos in np.flatnonzero(high_valid & (np.abs(high_pivot - upper_line) <= tolerance))
    }
    lower_overlap_touches = {
        int(active_indexes[pos])
        for pos in np.flatnonzero(low_valid & (np.abs(low_pivot - lower_line) <= tolerance))
    }
    recent_bars = max(1, int(round((end - start + 1) * float(cfg.channel_recent_touch_fraction))))
    recent_start = max(start, end - recent_bars + 1)
    upper_recent_touches = {pivot for pivot in upper_overlap_touches if pivot >= recent_start}
    lower_recent_touches = {pivot for pivot in lower_overlap_touches if pivot >= recent_start}
    upper_touches = len(upper_overlap_touches)
    lower_touches = len(lower_overlap_touches)
    touch_max = max(float(max(upper_touches, lower_touches)), 1.0)
    touch_balance_score = min(float(min(upper_touches, lower_touches)) / touch_max, 1.0)
    recent_touch_score = min(
        min(float(len(upper_recent_touches)), float(len(lower_recent_touches)))
        / max(float(cfg.channel_min_recent_rail_touches) * 2.0, 1.0),
        1.0,
    )
    coverage_score = min(position_coverage / 0.55, 1.0)
    empty_space_score = 0.45 * coverage_score + 0.35 * touch_balance_score + 0.20 * recent_touch_score
    return {
        "width_start": width_start,
        "width_end": width_end,
        "width_mid": width_mid,
        "width_pct_mid": float(width_pct_mid),
        "width_change_pct": float(width_change_pct),
        "shape": shape,
        "slope_diff_pct": float(slope_diff_pct),
        "parallel_score": float(parallel_score),
        "convergence_score": float(convergence_score),
        "containment_ratio": containment_ratio,
        "upper_touch_count": float(upper_touches),
        "lower_touch_count": float(lower_touches),
        "upper_recent_touch_count": float(len(upper_recent_touches)),
        "lower_recent_touch_count": float(len(lower_recent_touches)),
        "touch_balance_score": float(touch_balance_score),
        "recent_touch_score": float(recent_touch_score),
        "position_coverage": float(position_coverage),
        "empty_space_score": float(empty_space_score),
        "pivot_touch_count": float(upper_touches + lower_touches),
    }


def _suppress_duplicate_channel_rows(channels: DataFrame, cfg: TrendlineProjectionV2Config) -> DataFrame:
    """Keep the clearest channel when candidates describe the same geometry."""

    if channels.empty:
        return channels.copy()

    kept: list[pd.Series] = []
    for _, candidate in channels.sort_values(["score", "width_pct_mid"], ascending=[False, True]).iterrows():
        if any(_channels_are_duplicates(candidate, kept_channel, cfg) for kept_channel in kept):
            continue
        kept.append(candidate)

    if not kept:
        return channels.head(0).copy()
    return pd.DataFrame(kept, columns=channels.columns).reset_index(drop=True)


def _channels_are_duplicates(left: pd.Series, right: pd.Series, cfg: TrendlineProjectionV2Config) -> bool:
    overlap_start = max(float(left["start"]), float(right["start"]))
    overlap_end = min(float(left["end"]), float(right["end"]))
    overlap = overlap_end - overlap_start
    if overlap <= 0.0:
        return False
    left_span = max(float(left["end"]) - float(left["start"]), 1.0)
    right_span = max(float(right["end"]) - float(right["start"]), 1.0)
    overlap_ratio = overlap / max(min(left_span, right_span), 1.0)
    if overlap_ratio < float(cfg.channel_duplicate_overlap_pct):
        return False

    check_x = (overlap_start + overlap_end) / 2.0
    left_upper = float(left["upper_intercept"]) + float(left["upper_slope"]) * check_x
    left_lower = float(left["lower_intercept"]) + float(left["lower_slope"]) * check_x
    right_upper = float(right["upper_intercept"]) + float(right["upper_slope"]) * check_x
    right_lower = float(right["lower_intercept"]) + float(right["lower_slope"]) * check_x
    left_width = max(left_upper - left_lower, 1e-9)
    right_width = max(right_upper - right_lower, 1e-9)
    if left_width <= 0.0 or right_width <= 0.0:
        return False

    mid_distance = abs(((left_upper + left_lower) / 2.0) - ((right_upper + right_lower) / 2.0))
    width_ref = max(min(left_width, right_width), 1e-9)
    if mid_distance > width_ref * float(cfg.channel_duplicate_mid_width_mult):
        return False

    upper_slope_diff = _relative_float_diff(float(left["upper_slope"]), float(right["upper_slope"]))
    lower_slope_diff = _relative_float_diff(float(left["lower_slope"]), float(right["lower_slope"]))
    return max(upper_slope_diff, lower_slope_diff) <= float(cfg.channel_duplicate_slope_tolerance_pct)


def _relative_float_diff(left: float, right: float) -> float:
    denominator = max(abs(left), abs(right), 1e-9)
    return abs(left - right) / denominator


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


def _confirm_candidate_table(candidates: DataFrame, base: dict[str, Series], cfg: TrendlineProjectionV2Config) -> DataFrame:
    if candidates.empty:
        return candidates.copy()
    rows: list[dict[str, float | str]] = []
    for _, row in candidates.iterrows():
        confirmed = _confirm_candidate_row(row, base, cfg)
        if confirmed is not None:
            rows.append(confirmed)
    if not rows:
        return candidates.head(0).copy()
    return pd.DataFrame(rows).sort_values(["pivot_count", "score", "span"], ascending=False).reset_index(drop=True)


def _confirm_candidate_row(row: pd.Series, base: dict[str, Series], cfg: TrendlineProjectionV2Config) -> dict[str, float | str] | None:
    side = str(row["side"])
    pivots = _diagnostic_confirmation_pivots(base, side, cfg)
    if pivots.empty:
        return None
    seed_pivots = _pivot_tuple(row["pivot_path"])
    seed_prices = _price_tuple(row.to_dict())
    if len(seed_pivots) < 2 or len(seed_pivots) != len(seed_prices):
        return None
    slope = float(row["slope"])
    intercept = float(row["intercept"])
    x_new = int(round(float(row["x_new"])))
    path: dict[int, float] = {int(pivot): float(price) for pivot, price in zip(seed_pivots[:2], seed_prices[:2], strict=False)}
    errors: list[float] = []
    for pivot in pivots.itertuples(index=False):
        pivot_x = int(pivot.x)
        if pivot_x <= x_new or pivot_x > x_new + int(cfg.confirmation_max_bars) or pivot_x in path:
            continue
        expected = intercept + slope * float(pivot_x)
        error = abs(float(pivot.price) - expected)
        tolerance = max(
            abs(float(pivot.price)) * float(cfg.confirmation_tolerance_pct),
            float(pivot.atr) * float(cfg.confirmation_atr_mult),
        )
        if error <= tolerance:
            path[pivot_x] = float(pivot.price)
            errors.append(error)
    if len(path) < int(cfg.confirmation_min_pivots):
        return None
    ordered = sorted(path.items())
    last_pivot = ordered[-1][0]
    projection_end = _projection_end_after_sustained_break(
        base=base,
        side=side,
        slope=slope,
        intercept=intercept,
        start_index=last_pivot,
        max_end=min(len(base["close"]) - 1, last_pivot + int(cfg.max_projection_bars)),
        cfg=cfg,
    )
    candidate = row.to_dict()
    candidate.update(
        {
            "x_old": float(ordered[0][0]),
            "x_new": float(last_pivot),
            "seed_x_new": float(x_new),
            "y_old": float(intercept + slope * float(ordered[0][0])),
            "y_new": float(intercept + slope * float(last_pivot)),
            "projection_end": float(projection_end),
            "pivot_count": float(len(ordered)),
            "pivot_path": "|".join(str(pivot) for pivot, _ in ordered),
            "price_path": "|".join(str(float(price)) for _, price in ordered),
            "avg_touch_error_pct": float(np.mean([err / max(abs(price), 1.0) for err, (_, price) in zip(errors, ordered[2:], strict=False)]))
            if errors
            else 0.0,
            "max_touch_error_pct": float(np.max([err / max(abs(price), 1.0) for err, (_, price) in zip(errors, ordered[2:], strict=False)]))
            if errors
            else 0.0,
            "line_kind": "seed_no_redraw",
        }
    )
    candidate["score"] = min(
        1.0,
        float(row.get("score", 0.0)) * 0.45 + 0.14 * float(len(ordered)) - float(candidate["avg_touch_error_pct"]) * 10.0,
    )
    return candidate


def _diagnostic_confirmation_pivots(base: dict[str, Series], side: str, cfg: TrendlineProjectionV2Config) -> DataFrame:
    if side == "resistance":
        pivot_price = base["confirm_pivot_high"]
        pivot_index = base["confirm_pivot_high_index"]
        pivot_prominence = base["confirm_pivot_high_prominence"]
    else:
        pivot_price = base["confirm_pivot_low"]
        pivot_index = base["confirm_pivot_low_index"]
        pivot_prominence = base["confirm_pivot_low_prominence"]
    allowed = pivot_price.notna() & pivot_prominence.fillna(0.0).ge(float(cfg.min_confirmation_pivot_prominence_atr))
    rows = []
    for event_idx in np.flatnonzero(allowed.to_numpy()):
        pivot_x = int(round(float(pivot_index.iloc[event_idx])))
        if 0 <= pivot_x < len(base["close"]):
            rows.append(
                {
                    "x": pivot_x,
                    "price": float(pivot_price.iloc[event_idx]),
                    "atr": float(base["atr"].iloc[pivot_x]) if pd.notna(base["atr"].iloc[pivot_x]) else 0.0,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["x", "price", "atr"])
    return pd.DataFrame(rows).drop_duplicates(["x", "price"]).sort_values("x").reset_index(drop=True)


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


def _absorb_confirmed_keep_seed_table(candidates: DataFrame, base: dict[str, Series], cfg: TrendlineProjectionV2Config) -> DataFrame:
    if candidates.empty:
        return candidates.copy()
    out_rows: list[pd.Series] = []
    sorted_candidates = candidates.sort_values(["pivot_count", "score", "span"], ascending=False)
    for _, side_candidates in sorted_candidates.groupby("side", sort=False):
        kept: list[dict[str, object]] = []
        for _, candidate in side_candidates.iterrows():
            absorbed = False
            for kept_item in kept:
                kept_row = kept_item["row"]
                if not isinstance(kept_row, pd.Series):
                    continue
                if _candidate_is_weaker_or_equal(candidate, kept_row) and _candidate_is_absorbable_keep_seed(
                    candidate,
                    kept_row,
                    cfg,
                ):
                    kept_item["absorbed_pivots"].update(_pivot_set(candidate["pivot_path"]))
                    kept_item["absorbed_line_count"] = int(kept_item["absorbed_line_count"]) + 1
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
                        "absorbed_pivots": set(_pivot_set(candidate["pivot_path"])),
                        "absorbed_line_count": 0,
                        "line_width": 0.0,
                    }
                )
        for kept_item in kept:
            row = kept_item["row"]
            if isinstance(row, pd.Series):
                row = row.copy()
                row["absorbed_pivot_count"] = float(len(kept_item["absorbed_pivots"]))
                row["absorbed_line_count"] = float(kept_item["absorbed_line_count"])
                row["half_width"] = float(kept_item["line_width"]) * float(cfg.absorb_width_scale)
                row["line_kind"] = "absorb_keep_seed_geometry"
                row["score"] = min(
                    1.0,
                    float(row.get("score", 0.0))
                    + 0.02 * float(row["absorbed_pivot_count"])
                    + 0.025 * float(row["absorbed_line_count"]),
                )
                out_rows.append(row)
    if not out_rows:
        return candidates.head(0).copy()
    return pd.DataFrame(out_rows).sort_values(
        ["side", "absorbed_pivot_count", "score", "span"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def _candidate_is_absorbable_keep_seed(candidate: pd.Series, stronger: pd.Series, cfg: TrendlineProjectionV2Config) -> bool:
    left_norm = _row_norm_slope(candidate)
    right_norm = _row_norm_slope(stronger)
    if not _angle_within_tolerance(left_norm, right_norm, float(cfg.absorb_angle_tolerance_pct)):
        return False
    return _candidate_pivots_near_line(candidate, stronger, float(cfg.absorb_touch_tolerance_pct))


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
    """Build raw plus A->B->C continuation chains when angles stay similar."""

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
                continuations = by_old_anchor.get(float(path["x_new"]))
                if continuations is None:
                    continue
                path_pivots = _pivot_tuple(path["pivot_path"])
                path_prices = _price_tuple(path)
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
    return joined.drop_duplicates(["side", "pivot_path"], keep="first").reset_index(drop=True)


def _candidate_from_path(
    *,
    side: str,
    pivots: tuple[int, ...],
    prices: tuple[float, ...],
    projection_end: float,
    prominence: float,
    score: float,
    cfg: TrendlineProjectionV2Config,
) -> dict[str, float | str] | None:
    x_old = float(pivots[0])
    x_new = float(pivots[-1])
    y_old = float(prices[0])
    y_new = float(prices[-1])
    span = x_new - x_old
    if span <= 0.0:
        return None
    slope = (y_new - y_old) / span
    intercept = y_new - slope * x_new
    line_at_pivots = np.array([intercept + slope * float(pivot) for pivot in pivots], dtype="float64")
    prices_array = np.array(prices, dtype="float64")
    midpoint_error = np.max(np.abs(prices_array - line_at_pivots) / np.maximum(np.abs(prices_array), 1e-9))
    if midpoint_error > float(cfg.join_midpoint_tolerance_pct):
        return None
    price_ref = max((abs(y_old) + abs(y_new)) / 2.0, 1e-9)
    slope_pct = abs(slope) / price_ref
    if not np.isfinite(slope_pct) or slope_pct > float(cfg.max_slope_pct_per_bar):
        return None
    return {
        "side": side,
        "x_old": x_old,
        "x_new": x_new,
        "x_join": float(pivots[-2]),
        "y_old": y_old,
        "y_join": float(prices[-2]),
        "y_new": y_new,
        "slope": float(slope),
        "intercept": float(intercept),
        "projection_end": projection_end,
        "span": float(span),
        "slope_pct": float(slope_pct),
        "prominence": prominence,
        "score": score,
        "pivot_count": float(len(pivots)),
        "pivot_path": "|".join(str(pivot) for pivot in pivots),
        "price_path": "|".join(str(float(price)) for price in prices),
        "source_count": float(len(pivots) - 1),
    }


def _absorb_shared_pivot_smaller_lines(candidates: DataFrame, cfg: TrendlineProjectionV2Config) -> DataFrame:
    """Remove smaller lines when all their pivots are contained in a larger line."""

    if candidates.empty:
        return candidates.copy()

    kept_rows: list[pd.Series] = []
    sorted_candidates = candidates.sort_values(["pivot_count", "span", "score"], ascending=False)
    for _, side_candidates in sorted_candidates.groupby("side", sort=False):
        side_kept: list[pd.Series] = []
        for _, candidate in side_candidates.iterrows():
            candidate_pivots = _pivot_set(candidate["pivot_path"])
            absorbed = False
            for kept in side_kept:
                kept_pivots = _pivot_set(kept["pivot_path"])
                if len(candidate_pivots) < int(cfg.shared_pivot_merge_min):
                    continue
                if candidate_pivots.issubset(kept_pivots):
                    absorbed = True
                    break
            if not absorbed:
                side_kept.append(candidate)
        kept_rows.extend(side_kept)

    if not kept_rows:
        return candidates.head(0).copy()
    merged = pd.DataFrame(kept_rows).sort_values(["side", "score", "span"], ascending=[True, False, False]).reset_index(drop=True)
    merged["absorbed_pivot_count"] = merged["pivot_path"].map(lambda value: float(len(_pivot_set(value))))
    merged["absorbed_line_count"] = 0.0
    return merged


def _absorb_nearby_weaker_lines(candidates: DataFrame, proximity_pct: float) -> DataFrame:
    """Remove weaker lines near stronger lines and transfer unique pivot confirmations."""

    if candidates.empty:
        return candidates.copy()

    out_rows: list[pd.Series] = []
    sorted_candidates = candidates.sort_values(["pivot_count", "score", "span"], ascending=False)
    for _, side_candidates in sorted_candidates.groupby("side", sort=False):
        kept: list[dict[str, object]] = []
        for _, candidate in side_candidates.iterrows():
            candidate_pivots = _pivot_set(candidate["pivot_path"])
            absorbed = False
            for kept_item in kept:
                kept_row = kept_item["row"]
                if not isinstance(kept_row, pd.Series):
                    continue
                if _candidate_is_weaker_or_equal(candidate, kept_row) and _candidate_pivots_near_line(
                    candidate,
                    kept_row,
                    proximity_pct,
                ):
                    kept_item["absorbed_pivots"].update(candidate_pivots)
                    kept_item["absorbed_line_count"] = int(kept_item["absorbed_line_count"]) + 1
                    absorbed = True
                    break
            if not absorbed:
                kept.append(
                    {
                        "row": candidate.copy(),
                        "absorbed_pivots": set(candidate_pivots),
                        "absorbed_line_count": 0,
                    }
                )
        for kept_item in kept:
            row = kept_item["row"]
            if isinstance(row, pd.Series):
                row = row.copy()
                row["absorbed_pivot_count"] = float(len(kept_item["absorbed_pivots"]))
                row["absorbed_line_count"] = float(kept_item["absorbed_line_count"])
                row["nearby_merge_proximity_pct"] = float(proximity_pct)
                out_rows.append(row)

    if not out_rows:
        return candidates.head(0).copy()
    return pd.DataFrame(out_rows).sort_values(
        ["side", "absorbed_pivot_count", "score", "span"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


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
    price_ref = ((candidates["y_old"].abs() + candidates["y_new"].abs()) / 2.0).replace(0.0, np.nan)
    return (candidates["slope"] / price_ref).replace([np.inf, -np.inf], np.nan)


def _row_norm_slope(row: pd.Series) -> float:
    price_ref = max((abs(float(row["y_old"])) + abs(float(row["y_new"]))) / 2.0, 1e-9)
    return float(row["slope"]) / price_ref


def _pivot_tuple(path: object) -> tuple[int, ...]:
    return tuple(int(part) for part in str(path).split("|") if part != "")


def _price_tuple(row: dict[str, object]) -> tuple[float, ...]:
    price_path = row.get("price_path")
    if price_path is not None and str(price_path) != "nan":
        return tuple(float(part) for part in str(price_path).split("|") if part != "")
    return (float(row["y_old"]), float(row["y_new"]))


def _angle_within_tolerance(left_norm_slope: float, right_norm_slope: float, tolerance_pct: float) -> bool:
    if not np.isfinite(left_norm_slope) or not np.isfinite(right_norm_slope):
        return False
    denominator = max(abs(left_norm_slope), abs(right_norm_slope), 1e-9)
    return abs(left_norm_slope - right_norm_slope) / denominator <= tolerance_pct


def _pivot_set(path: object) -> set[int]:
    return {int(part) for part in str(path).split("|") if part != ""}


def _diagnostic_cluster_table(
    *,
    frame: DataFrame,
    candidates: DataFrame,
    label: str,
    proximity_pct: float,
    min_lines: int,
    slots: int,
) -> DataFrame:
    if candidates.empty:
        return pd.DataFrame(index=frame.index)
    line_values, score_values, _ = _candidate_matrix_for_frame(frame, candidates)
    clusters = _cluster_matrix_rows(
        line_values=line_values,
        score_values=score_values,
        close=_num(frame, "close").to_numpy(dtype="float64"),
        proximity_pct=proximity_pct,
        min_lines=min_lines,
        slots=slots,
    )
    return pd.DataFrame(_cluster_output_columns(clusters, frame.index, "tlv2", label, include_bias=False), index=frame.index)


def _diagnostic_hotspot_table(
    *,
    frame: DataFrame,
    candidates: DataFrame,
    proximity_pct: float,
    min_lines: int,
    slots: int,
) -> DataFrame:
    if candidates.empty:
        return pd.DataFrame(index=frame.index)
    line_values, score_values, labels = _candidate_matrix_for_frame(frame, candidates)
    clusters = _cluster_matrix_rows(
        line_values=line_values,
        score_values=score_values,
        close=_num(frame, "close").to_numpy(dtype="float64"),
        proximity_pct=proximity_pct,
        min_lines=min_lines,
        slots=slots,
        labels=labels,
    )
    return pd.DataFrame(_cluster_output_columns(clusters, frame.index, "tlv2", "hotspot", include_bias=True), index=frame.index)


def _candidate_matrix_for_frame(frame: DataFrame, candidates: DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = len(frame)
    line_values = np.full((rows, len(candidates)), np.nan, dtype="float64")
    score_values = np.full((rows, len(candidates)), np.nan, dtype="float64")
    labels = np.where(candidates["side"].eq("resistance").to_numpy(), 1.0, -1.0)
    x_values = np.arange(rows, dtype="float64")
    for col, candidate in enumerate(candidates.itertuples(index=False)):
        start = max(int(candidate.x_old), 0)
        end = min(int(candidate.projection_end), rows - 1)
        if end < start:
            continue
        xs = x_values[start : end + 1]
        line_values[start : end + 1, col] = float(candidate.intercept) + float(candidate.slope) * xs
        score_values[start : end + 1, col] = float(candidate.score)
    return line_values, score_values, labels


def _empty_pack() -> dict[str, list[Series]]:
    return {
        "line": [],
        "score": [],
        "slope": [],
        "slope_pct": [],
        "anchor_old_index": [],
        "anchor_new_index": [],
        "projection_end_index": [],
        "line_id": [],
        "pivot_count": [],
        "absorbed_pivot_count": [],
        "line_width": [],
        "last_confirm_index": [],
    }


def _series_matrix(values: Sequence[Series]) -> np.ndarray:
    if not values:
        return np.empty((0, 0), dtype="float64")
    return pd.concat(list(values), axis=1).astype("float64").to_numpy()


def _select_metric(values: np.ndarray, chosen: np.ndarray, valid: np.ndarray) -> np.ndarray:
    rows = np.arange(values.shape[0])
    selected = values[rows, chosen]
    return np.where(valid, selected, np.nan)


def _plot_safe_line(line: Series, line_id: Series, break_pct: float) -> Series:
    line_float = line.astype("float64")
    id_change = line_id.ne(line_id.shift(1))
    jump = (line_float.pct_change(fill_method=None).abs() > break_pct).fillna(False)
    return line_float.where(~id_change & ~jump)


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


def _safe_div(numerator: Series, denominator: Series) -> Series:
    return numerator / denominator.replace(0.0, np.nan)


def _clip01(value: Series) -> Series:
    return value.clip(lower=0.0, upper=1.0)


def _resolve_config(config: TrendlineProjectionV2Config | None, overrides: dict[str, object]) -> TrendlineProjectionV2Config:
    cfg = config or TrendlineProjectionV2Config()
    clean = {key: value for key, value in overrides.items() if value is not None}
    return replace(cfg, **clean) if clean else cfg


def _validate_dataframe(frame: DataFrame) -> None:
    required = {"open", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataframe missing required columns: {missing}")


def _validate_config(cfg: TrendlineProjectionV2Config) -> None:
    if cfg.pivot_strength < 1:
        raise ValueError("pivot_strength must be positive")
    if cfg.confirmation_pivot_strength < 1:
        raise ValueError("confirmation_pivot_strength must be positive")
    if cfg.candidate_pivot_count < 3:
        raise ValueError("candidate_pivot_count must be at least 3")
    if cfg.raw_line_output_count < 1:
        raise ValueError("raw_line_output_count must be positive")
    if cfg.confirmed_line_output_count < 1 or cfg.absorbed_line_output_count < 1 or cfg.structural_line_output_count < 1:
        raise ValueError("confirmed/absorbed/structural line output counts must be positive")
    if cfg.channel_output_count < 1:
        raise ValueError("channel_output_count must be positive")
    if cfg.fuzzy_zone_count < 1 or cfg.hotspot_count < 1:
        raise ValueError("cluster output counts must be positive")
    if cfg.min_anchor_bars < 1:
        raise ValueError("min_anchor_bars must be positive")
    if cfg.max_anchor_bars < cfg.min_anchor_bars:
        raise ValueError("max_anchor_bars must be greater than or equal to min_anchor_bars")
    if cfg.structural_max_anchor_bars < cfg.max_anchor_bars:
        raise ValueError("structural_max_anchor_bars must be greater than or equal to max_anchor_bars")
    if cfg.min_pivot_prominence_atr < 0.0:
        raise ValueError("min_pivot_prominence_atr must be non-negative")
    if cfg.min_confirmation_pivot_prominence_atr < 0.0:
        raise ValueError("min_confirmation_pivot_prominence_atr must be non-negative")
    if cfg.max_slope_pct_per_bar <= 0.0:
        raise ValueError("max_slope_pct_per_bar must be positive")
    if cfg.max_projection_bars < 1:
        raise ValueError("max_projection_bars must be positive")
    if cfg.structural_max_projection_bars < cfg.max_projection_bars:
        raise ValueError("structural_max_projection_bars must be greater than or equal to max_projection_bars")
    if cfg.channel_source_line_count < 2:
        raise ValueError("channel_source_line_count must be at least 2")
    if cfg.channel_candidate_pool_size < cfg.channel_source_line_count:
        raise ValueError("channel_candidate_pool_size must be greater than or equal to channel_source_line_count")
    if cfg.channel_min_pivot_prominence_atr < 0.0:
        raise ValueError("channel_min_pivot_prominence_atr must be non-negative")
    if cfg.channel_min_anchor_bars < 1:
        raise ValueError("channel_min_anchor_bars must be positive")
    if cfg.channel_max_anchor_bars < cfg.min_anchor_bars:
        raise ValueError("channel_max_anchor_bars must be greater than or equal to min_anchor_bars")
    if cfg.channel_max_anchor_bars < cfg.channel_min_anchor_bars:
        raise ValueError("channel_max_anchor_bars must be greater than or equal to channel_min_anchor_bars")
    if cfg.channel_max_projection_bars < 1:
        raise ValueError("channel_max_projection_bars must be positive")
    if cfg.channel_min_overlap_bars < 1:
        raise ValueError("channel_min_overlap_bars must be positive")
    if cfg.channel_slope_tolerance_pct <= 0.0:
        raise ValueError("channel_slope_tolerance_pct must be positive")
    if cfg.channel_parallel_width_change_pct <= 0.0 or cfg.channel_min_convergence_pct < 0.0:
        raise ValueError("channel width relation settings are invalid")
    if cfg.channel_min_width_pct <= 0.0 or cfg.channel_max_width_pct <= cfg.channel_min_width_pct:
        raise ValueError("channel width settings are invalid")
    if not 0.0 <= cfg.channel_min_containment_ratio <= 1.0:
        raise ValueError("channel_min_containment_ratio must be between 0 and 1")
    if cfg.channel_touch_tolerance_pct <= 0.0 or cfg.channel_touch_atr_mult < 0.0:
        raise ValueError("channel touch tolerances are invalid")
    if cfg.channel_min_pivot_touches < 2:
        raise ValueError("channel_min_pivot_touches must be at least 2")
    if cfg.channel_min_rail_touches < 1:
        raise ValueError("channel_min_rail_touches must be positive")
    if cfg.channel_min_recent_rail_touches < 0:
        raise ValueError("channel_min_recent_rail_touches must be non-negative")
    if not 0.0 < cfg.channel_recent_touch_fraction <= 1.0:
        raise ValueError("channel_recent_touch_fraction must be between 0 and 1")
    if not 0.0 <= cfg.channel_min_position_coverage <= 1.0:
        raise ValueError("channel_min_position_coverage must be between 0 and 1")
    if not 0.0 < cfg.channel_duplicate_overlap_pct <= 1.0:
        raise ValueError("channel_duplicate_overlap_pct must be between 0 and 1")
    if cfg.channel_duplicate_mid_width_mult <= 0.0:
        raise ValueError("channel_duplicate_mid_width_mult must be positive")
    if cfg.channel_duplicate_slope_tolerance_pct <= 0.0:
        raise ValueError("channel_duplicate_slope_tolerance_pct must be positive")
    if cfg.channel_breakout_grace_bars < 0:
        raise ValueError("channel_breakout_grace_bars must be non-negative")
    if cfg.confirmation_tolerance_pct <= 0.0 or cfg.confirmation_atr_mult < 0.0:
        raise ValueError("confirmation tolerances must be positive")
    if cfg.confirmation_max_bars < 1 or cfg.confirmation_min_pivots < 2:
        raise ValueError("confirmation bars/pivots settings are invalid")
    if cfg.absorb_touch_tolerance_pct <= 0.0 or cfg.absorb_angle_tolerance_pct <= 0.0:
        raise ValueError("absorb tolerances must be positive")
    if cfg.absorb_width_scale < 0.0:
        raise ValueError("absorb_width_scale must be non-negative")
    if cfg.anchor_break_edge_bars < 0 or cfg.anchor_break_max_run < 0:
        raise ValueError("anchor break bars/run settings must be non-negative")
    if cfg.anchor_break_max_pct < 0.0 or cfg.anchor_break_max_atr_mult < 0.0:
        raise ValueError("anchor break tolerance settings must be non-negative")
    if cfg.projection_break_candles < 1:
        raise ValueError("projection_break_candles must be positive")
    if cfg.projection_break_tolerance_pct < 0.0 or cfg.projection_break_atr_mult < 0.0:
        raise ValueError("projection break tolerance settings must be non-negative")
    if cfg.density_proximity_pct <= 0.0 or cfg.hotspot_proximity_pct <= 0.0:
        raise ValueError("cluster proximity settings must be positive")
    if cfg.density_min_lines < 2 or cfg.hotspot_min_lines < 2:
        raise ValueError("cluster min line settings must be at least 2")
    if cfg.join_angle_tolerance_pct <= 0.0:
        raise ValueError("join_angle_tolerance_pct must be positive")
    if cfg.join_midpoint_tolerance_pct <= 0.0:
        raise ValueError("join_midpoint_tolerance_pct must be positive")
    if cfg.max_joined_pivots < 2:
        raise ValueError("max_joined_pivots must be at least 2")
    if cfg.shared_pivot_merge_min < 1:
        raise ValueError("shared_pivot_merge_min must be positive")
    if not cfg.nearby_merge_proximity_pcts:
        raise ValueError("nearby_merge_proximity_pcts must not be empty")
    if any(proximity <= 0.0 for proximity in cfg.nearby_merge_proximity_pcts):
        raise ValueError("nearby_merge_proximity_pcts values must be positive")
    if cfg.max_raw_lines_plotted < 1:
        raise ValueError("max_raw_lines_plotted must be positive")


__all__ = [
    "TrendlineProjectionV2Config",
    "TrendlineProjectionV2Diagnostic",
    "add_trendline_projection_v2",
    "build_trendline_projection_v2_diagnostic",
]

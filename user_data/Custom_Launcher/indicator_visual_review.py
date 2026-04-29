#!/usr/bin/env python3
"""Generate visual indicator review plots against OHLCV candles.

This is deliberately separate from ``indicator_validation_runner.py``. The
validation runner checks score contracts and forward sorting. This script checks
whether indicator geometry and levels make visual sense on candles.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

THIS_DIR = Path(__file__).resolve().parent
USER_DATA_DIR = THIS_DIR.parent
PROJECT_ROOT = USER_DATA_DIR.parent
RUNTIME_DIR = USER_DATA_DIR / "Indicator_External_Validator"
OHLCV_COLUMNS = ("date", "open", "high", "low", "close", "volume")
IndicatorName = Literal["pa", "tl", "stl", "vol", "vp", "pat", "vc", "rs", "regime"]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from user_data.Indicators.complex_pattern_structure import add_pattern_structure
from user_data.Indicators.complex_pivot_structure import PivotStructureConfig, add_pivot_structure
from user_data.Indicators.complex_relative_strength import add_relative_strength
from user_data.Indicators.complex_structural_trendlines import StructuralTrendlineConfig, add_structural_trendlines
from user_data.Indicators.complex_trendline_projection import TrendlineProjectionConfig, add_trendline_projection
from user_data.Indicators.complex_volatility_cycles import add_volatility_cycles
from user_data.Indicators.complex_volume_indicators import add_complex_volume_indicators
from user_data.Indicators.complex_volume_profile import VolumeProfileConfig, add_volume_profile
from user_data.Indicators.market_regime import (
    REGIME_CODE_BEAR,
    REGIME_CODE_BULL,
    REGIME_CODE_CHOP,
    REGIME_CODE_CRASH,
    add_market_regime,
)


@dataclass(frozen=True)
class PlotContext:
    pair_key: str
    pair_label: str
    timeframe: str
    source_path: Path
    output_dir: Path
    tail: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot indicator geometry/levels for visual review.")
    parser.add_argument("--indicator", choices=("pa", "tl", "stl", "vol", "vp", "pat", "vc", "rs", "regime"), required=True, help="Indicator to plot.")
    parser.add_argument("--datadir", default=str(USER_DATA_DIR / "data"), help="Freqtrade data directory.")
    parser.add_argument("--pair", default="BTC/USDT:USDT", help="Pair to plot, e.g. BTC/USDT:USDT.")
    parser.add_argument("--benchmark", default="BTC/USDT:USDT", help="Benchmark pair for relative strength plots.")
    parser.add_argument("--timeframe", default="1h", help="Timeframe to plot.")
    parser.add_argument("--tail", type=int, default=360, help="Candles to show in the final plot.")
    parser.add_argument("--calc-tail", type=int, default=2600, help="Candles to load before plotting.")
    parser.add_argument("--output-dir", default=str(RUNTIME_DIR / "visual_reviews"), help="Output directory.")
    parser.add_argument("--profile-window", type=int, default=96, help="Volume profile rolling window.")
    parser.add_argument("--profile-bins", type=int, default=48, help="Volume profile price bins.")
    parser.add_argument("--profile-chunk-size", type=int, default=512, help="Volume profile chunk size.")
    parser.add_argument("--vp-view", choices=("all", "market-context", "node-actions"), default="all", help="Volume profile plot set.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = find_candle_file(Path(args.datadir), args.pair, args.timeframe)
    if path is None:
        raise SystemExit(f"No candle file found for pair={args.pair} timeframe={args.timeframe} in {args.datadir}")
    frame = load_candle_frame(path).tail(max(int(args.calc_tail), int(args.tail))).copy()
    pair_key = normalize_pair_key(args.pair)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / str(args.indicator) / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    context = PlotContext(
        pair_key=pair_key,
        pair_label=args.pair,
        timeframe=args.timeframe,
        source_path=path,
        output_dir=output_dir,
        tail=int(args.tail),
    )
    if args.indicator == "stl":
        outputs = plot_structural_trendlines(frame, context)
    elif args.indicator == "pa":
        outputs = plot_pivot_structure(frame, context)
    elif args.indicator == "tl":
        outputs = plot_trendline_projection(frame, context)
    elif args.indicator == "vol":
        outputs = plot_complex_volume(frame, context)
    elif args.indicator == "vp":
        outputs = plot_volume_profile(
            frame,
            context,
            window=int(args.profile_window),
            bins=int(args.profile_bins),
            chunk_size=int(args.profile_chunk_size),
            vp_view=str(args.vp_view),
        )
    elif args.indicator == "pat":
        outputs = plot_pattern_structure(frame, context)
    elif args.indicator == "vc":
        outputs = plot_volatility_cycles(frame, context)
    elif args.indicator == "rs":
        benchmark_path = find_candle_file(Path(args.datadir), args.benchmark, args.timeframe)
        if benchmark_path is None:
            raise SystemExit(f"No benchmark candle file found for {args.benchmark} {args.timeframe}")
        benchmark = load_candle_frame(benchmark_path).tail(max(int(args.calc_tail), int(args.tail))).copy()
        outputs = plot_relative_strength(frame, benchmark, context, args.benchmark)
    elif args.indicator == "regime":
        outputs = plot_market_regime(frame, context)
    else:
        raise SystemExit(f"Unsupported indicator: {args.indicator}")

    for output in outputs:
        print(output)
    return 0


def plot_structural_trendlines(frame: DataFrame, context: PlotContext) -> list[Path]:
    data = add_pivot_structure(
        frame,
        PivotStructureConfig(strength=5, strengths=(3, 5, 8, 13, 21), min_prominence_atr=0.35, prefix="pa"),
    )
    data = add_structural_trendlines(data, StructuralTrendlineConfig(pivot_prefix="pa", output_prefix="stl", ranked_line_count=3))
    view = data.tail(context.tail).copy()
    fig, ax = plt.subplots(figsize=(15, 8), dpi=140)
    draw_candles(ax, view)
    resistance_touches = []
    support_touches = []
    for rank in range(3):
        resistance_touches.extend(draw_structural_line(ax, data, view, "resistance", rank, "#d62828"))
        support_touches.extend(draw_structural_line(ax, data, view, "support", rank, "#078f4f"))
    draw_event_markers(ax, view, "stl_suggested_entry_long", "low", "#84cc16", "^", "suggest long")
    draw_event_markers(ax, view, "stl_suggested_entry_short", "high", "#e11d48", "v", "suggest short")
    ax.set_title(
        f"{context.pair_label} {context.timeframe} structural trendlines | "
        f"res touches {len(resistance_touches)}, sup touches {len(support_touches)}"
    )
    finish_axes(ax, view)
    output = context.output_dir / f"{context.pair_key}_{context.timeframe}_stl_touchspan.png"
    fig.savefig(output)
    plt.close(fig)
    return [output]


def plot_pivot_structure(frame: DataFrame, context: PlotContext) -> list[Path]:
    data = add_pivot_structure(frame, PivotStructureConfig(strength=5, strengths=(3, 5, 8, 13), prefix="pa"))
    view = data.tail(context.tail).copy()
    fig, ax = plt.subplots(figsize=(15, 8), dpi=140)
    draw_candles(ax, view)
    draw_line_series(ax, view, "pa_resistance_line", "#d97706", "resistance", 1.4)
    draw_line_series(ax, view, "pa_support_line", "#2563eb", "support", 1.4)
    for strength, alpha in ((5, 0.90), (13, 0.60)):
        draw_price_markers(ax, view, f"pa_pivot_high_{strength}", "#dc2626", "v", f"pivot high {strength}", alpha=alpha)
        draw_price_markers(ax, view, f"pa_pivot_low_{strength}", "#16a34a", "^", f"pivot low {strength}", alpha=alpha)
    draw_event_markers(ax, view, "pa_ms_bullish_bos", "high", "#15803d", "^", "bull BOS")
    draw_event_markers(ax, view, "pa_ms_bearish_bos", "low", "#b91c1c", "v", "bear BOS")
    draw_event_markers(ax, view, "pa_ms_bullish_choch", "high", "#22c55e", "^", "bull CHoCH")
    draw_event_markers(ax, view, "pa_ms_bearish_choch", "low", "#f43f5e", "v", "bear CHoCH")
    draw_event_markers(ax, view, "pa_suggested_entry_long", "low", "#84cc16", "^", "suggest long")
    draw_event_markers(ax, view, "pa_suggested_entry_short", "high", "#e11d48", "v", "suggest short")
    ax.set_title(f"{context.pair_label} {context.timeframe} pivot structure")
    finish_axes(ax, view)
    output = context.output_dir / f"{context.pair_key}_{context.timeframe}_pa_structure.png"
    fig.savefig(output)
    plt.close(fig)
    return [output, plot_score_panel(data, context, "pa", "pivot structure scores", ("pa_ms_up_sequence_score", "pa_ms_down_sequence_score", "pa_channel_compression"))]


def plot_trendline_projection(frame: DataFrame, context: PlotContext) -> list[Path]:
    data = add_pivot_structure(frame, PivotStructureConfig(strength=5, strengths=(3, 5, 8, 13), prefix="pa"))
    data = add_trendline_projection(
        data,
        TrendlineProjectionConfig(
            strength=5,
            strengths=(3, 5, 8, 13),
            pivot_prefix="pa",
            output_prefix="tl",
            missing_pivot_mode="raise",
        ),
    )
    view = data.tail(context.tail).copy()
    fig, ax = plt.subplots(figsize=(15, 8), dpi=140)
    draw_candles(ax, view)
    for rank, alpha in ((0, 0.95), (1, 0.65), (2, 0.45)):
        draw_line_series(ax, view, f"tl_resistance_plot_rank{rank}", "#ef4444", f"res rank{rank}", 1.9 - rank * 0.35, alpha=alpha)
        draw_line_series(ax, view, f"tl_support_plot_rank{rank}", "#2563eb", f"sup rank{rank}", 1.9 - rank * 0.35, alpha=alpha)
    draw_event_markers(ax, view, "tl_resistance_reject", "high", "#dc2626", "v", "res reject")
    draw_event_markers(ax, view, "tl_support_reclaim", "low", "#16a34a", "^", "sup reclaim")
    draw_event_markers(ax, view, "tl_resistance_breakout_1", "high", "#15803d", "^", "res breakout")
    draw_event_markers(ax, view, "tl_support_breakdown_1", "low", "#b91c1c", "v", "sup breakdown")
    draw_event_markers(ax, view, "tl_suggested_entry_long", "low", "#84cc16", "^", "suggest long")
    draw_event_markers(ax, view, "tl_suggested_entry_short", "high", "#e11d48", "v", "suggest short")
    ax.set_title(f"{context.pair_label} {context.timeframe} local trendline projection")
    finish_axes(ax, view)
    output = context.output_dir / f"{context.pair_key}_{context.timeframe}_tl_projection.png"
    fig.savefig(output)
    plt.close(fig)
    return [output, plot_score_panel(data, context, "tl", "local trendline scores", ("tl_support_quality", "tl_resistance_quality", "tl_channel_compression"))]


def plot_volume_profile(
    frame: DataFrame,
    context: PlotContext,
    *,
    window: int,
    bins: int,
    chunk_size: int,
    vp_view: str = "all",
) -> list[Path]:
    data = add_volume_profile(
        frame,
        VolumeProfileConfig(window=window, bins=bins, chunk_size=chunk_size),
    )
    if vp_view == "market-context":
        return [plot_volume_profile_market_context(data, context, window=window, bins=bins)]
    if vp_view == "node-actions":
        return [plot_volume_profile_node_actions(data, context, window=window, bins=bins)]
    return [
        plot_volume_profile_levels(data, context, window=window, bins=bins),
        plot_volume_profile_nodes(data, context, window=window, bins=bins),
        plot_volume_profile_events(data, context, window=window, bins=bins),
        plot_volume_profile_scores(data, context),
    ]


def plot_volume_profile_market_context(data: DataFrame, context: PlotContext, *, window: int, bins: int) -> Path:
    view = data.tail(context.tail).copy()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 8),
        dpi=140,
        sharex=True,
        gridspec_kw={"height_ratios": [4.0, 1.15]},
    )
    ax = axes[0]
    context_ax = axes[1]
    shade_vp_market_context_regions(ax, view)
    draw_candles(ax, view)
    draw_vp_market_context(context_ax, view)
    ax.set_title(
        f"{context.pair_label} {context.timeframe} VP market context | "
        f"window {window}, bins {bins}"
    )
    finish_panel_axes(fig, axes, view)
    output = context.output_dir / f"{context.pair_key}_{context.timeframe}_vp_market_context.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_volume_profile_node_actions(data: DataFrame, context: PlotContext, *, window: int, bins: int) -> Path:
    view = data.tail(context.tail).copy()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 9),
        dpi=140,
        sharex=True,
        gridspec_kw={"height_ratios": [4.2, 1.15]},
    )
    ax = axes[0]
    lane_ax = axes[1]
    shade_regions(ax, bool_array(view, "vp_node_hold_long"), "#dcfce7", "hold long")
    shade_regions(ax, bool_array(view, "vp_node_hold_short"), "#fee2e2", "hold short")
    draw_candles(ax, view)
    draw_conditional_line_series(ax, view, "vp_hvn_above", "vp_actionable_hvn_above", "#b45309", "actionable HVN above", 1.0, alpha=0.50)
    draw_conditional_line_series(ax, view, "vp_hvn_below", "vp_actionable_hvn_below", "#b45309", "actionable HVN below", 1.0, alpha=0.50)
    draw_conditional_line_series(ax, view, "vp_lvn_above", "vp_actionable_lvn_above", "#7c3aed", "actionable LVN above", 1.0, alpha=0.50)
    draw_conditional_line_series(ax, view, "vp_lvn_below", "vp_actionable_lvn_below", "#7c3aed", "actionable LVN below", 1.0, alpha=0.50)
    draw_event_markers(ax, view, "vp_node_entry_long", "low", "#00d5ff", "*", "node entry long", size=95, edgecolor="#003b49")
    draw_event_markers(ax, view, "vp_node_entry_short", "high", "#ff00d4", "X", "node entry short", size=75, edgecolor="#4a003f")
    draw_event_markers(ax, view, "vp_node_exit_long", "high", "#f97316", "v", "node exit long", size=55, edgecolor="#7c2d12")
    draw_event_markers(ax, view, "vp_node_exit_short", "low", "#2563eb", "^", "node exit short", size=55, edgecolor="#1e3a8a")
    draw_vp_node_action_lanes(lane_ax, view)
    ax.set_title(
        f"{context.pair_label} {context.timeframe} VP HVN/LVN node actions | "
        f"window {window}, bins {bins}"
    )
    finish_panel_axes(fig, axes, view)
    output = context.output_dir / f"{context.pair_key}_{context.timeframe}_vp_node_actions.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_volume_profile_levels(data: DataFrame, context: PlotContext, *, window: int, bins: int) -> Path:
    view = data.tail(context.tail).copy()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 9),
        dpi=140,
        sharex=True,
        gridspec_kw={"height_ratios": [4.2, 1.0]},
    )
    ax = axes[0]
    flag_ax = axes[1]
    draw_candles(ax, view)
    draw_line_series(ax, view, "vp_poc", "#111111", "POC", 1.8)
    draw_line_series(ax, view, "vp_vah", "#d97706", "VAH", 1.25)
    draw_line_series(ax, view, "vp_val", "#2563eb", "VAL", 1.25)
    draw_vp_market_context(flag_ax, view)
    ax.set_title(
        f"{context.pair_label} {context.timeframe} volume profile levels | "
        f"window {window}, bins {bins}"
    )
    finish_panel_axes(fig, axes, view)
    output = context.output_dir / f"{context.pair_key}_{context.timeframe}_vp_levels.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_volume_profile_nodes(data: DataFrame, context: PlotContext, *, window: int, bins: int) -> Path:
    view = data.tail(context.tail).copy()
    fig, ax = plt.subplots(figsize=(15, 8), dpi=140)
    draw_candles(ax, view)
    draw_line_series(ax, view, "vp_hvn_above", "#b45309", "HVN above", 0.9, alpha=0.45)
    draw_line_series(ax, view, "vp_hvn_below", "#b45309", "HVN below", 0.9, alpha=0.45)
    draw_line_series(ax, view, "vp_lvn_above", "#7c3aed", "LVN above", 0.9, alpha=0.45)
    draw_line_series(ax, view, "vp_lvn_below", "#7c3aed", "LVN below", 0.9, alpha=0.45)
    ax.set_title(
        f"{context.pair_label} {context.timeframe} volume profile nodes | "
        f"window {window}, bins {bins}"
    )
    finish_axes(ax, view)
    output = context.output_dir / f"{context.pair_key}_{context.timeframe}_vp_nodes.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_volume_profile_events(data: DataFrame, context: PlotContext, *, window: int, bins: int) -> Path:
    view = data.tail(context.tail).copy()
    fig, ax = plt.subplots(figsize=(15, 8), dpi=140)
    draw_candles(ax, view)
    draw_event_markers(ax, view, "vp_vah_breakout_with_pressure", "high", "#15803d", "^", "VAH breakout")
    draw_event_markers(ax, view, "vp_val_breakdown_with_pressure", "low", "#b91c1c", "v", "VAL breakdown")
    draw_event_markers(ax, view, "vp_lower_rejection_with_pressure", "low", "#16a34a", "^", "VAL rejection")
    draw_event_markers(ax, view, "vp_upper_rejection_with_pressure", "high", "#dc2626", "v", "VAH rejection")
    draw_event_markers(ax, view, "vp_lvn_accept_long", "high", "#65a30d", "^", "LVN accept long")
    draw_event_markers(ax, view, "vp_lvn_accept_short", "low", "#be123c", "v", "LVN accept short")
    draw_event_markers(ax, view, "vp_lvn_fast_traverse_long", "high", "#22c55e", "^", "LVN traverse long")
    draw_event_markers(ax, view, "vp_lvn_fast_traverse_short", "low", "#f43f5e", "v", "LVN traverse short")
    draw_event_markers(ax, view, "vp_entry_trigger_long", "low", "#00d5ff", "*", "entry trigger long", size=90, edgecolor="#003b49")
    draw_event_markers(ax, view, "vp_entry_trigger_short", "high", "#ff00d4", "X", "entry trigger short", size=70, edgecolor="#4a003f")
    ax.set_title(
        f"{context.pair_label} {context.timeframe} volume profile events | "
        f"window {window}, bins {bins}"
    )
    finish_axes(ax, view)
    output = context.output_dir / f"{context.pair_key}_{context.timeframe}_vp_events.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_volume_profile_scores(data: DataFrame, context: PlotContext) -> Path:
    view = data.tail(context.tail).copy()
    fig, axes = plt.subplots(4, 1, figsize=(15, 11), dpi=140, sharex=True)
    xs = np.arange(len(view))
    axes[0].plot(xs, pd.to_numeric(view["close"], errors="coerce"), color="#111111", linewidth=1.1, label="close")
    axes[0].set_title(f"{context.pair_label} {context.timeframe} volume profile scores")
    for column, color, label in (
        ("vp_score_long", "#15803d", "long"),
        ("vp_score_short", "#b91c1c", "short"),
        ("vp_score_abs", "#1d4ed8", "abs"),
    ):
        if column in view.columns:
            axes[1].plot(xs, pd.to_numeric(view[column], errors="coerce"), color=color, linewidth=1.0, label=label)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("entry score")
    for column, color, label in (
        ("vp_context_score_bull", "#00a5ff", "bull context"),
        ("vp_context_score_bear", "#ff00b8", "bear context"),
        ("vp_context_score_balance", "#f59e0b", "balance context"),
    ):
        if column in view.columns:
            axes[2].plot(xs, pd.to_numeric(view[column], errors="coerce"), color=color, linewidth=1.0, label=label)
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_ylabel("context")
    for column, color, label in (
        ("vp_delta_ratio", "#15803d", "delta ratio"),
        ("vp_poc_delta_ratio", "#7c3aed", "poc delta"),
        ("vp_value_direction_pct", "#f97316", "value direction"),
    ):
        if column in view.columns:
            axes[3].plot(xs, pd.to_numeric(view[column], errors="coerce"), color=color, linewidth=0.9, label=label)
    axes[3].axhline(0.0, color="#111111", linewidth=0.7, alpha=0.4)
    axes[3].set_ylabel("pressure")
    finish_panel_axes(fig, axes, view)
    output = context.output_dir / f"{context.pair_key}_{context.timeframe}_vp_scores.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_pattern_structure(frame: DataFrame, context: PlotContext) -> list[Path]:
    data = add_pivot_structure(frame, PivotStructureConfig(strength=5, strengths=(3, 5, 8, 13), prefix="pa"))
    data = add_trendline_projection(data, TrendlineProjectionConfig(pivot_prefix="pa", output_prefix="tl", missing_pivot_mode="raise"))
    data = add_pattern_structure(data, pivot_prefix="pa", trendline_prefix="tl", output_prefix="pat")
    view = data.tail(context.tail).copy()
    fig, ax = plt.subplots(figsize=(15, 8), dpi=140)
    draw_candles(ax, view)
    draw_event_markers(ax, view, "pat_flag_long", "low", "#16a34a", "^", "flag long")
    draw_event_markers(ax, view, "pat_flag_short", "high", "#dc2626", "v", "flag short")
    draw_event_markers(ax, view, "pat_breakout_long", "high", "#15803d", "^", "breakout long")
    draw_event_markers(ax, view, "pat_breakout_short", "low", "#b91c1c", "v", "breakout short")
    draw_event_markers(ax, view, "pat_suggested_entry_long", "low", "#84cc16", "^", "suggest long")
    draw_event_markers(ax, view, "pat_suggested_entry_short", "high", "#e11d48", "v", "suggest short")
    ax.set_title(f"{context.pair_label} {context.timeframe} pattern structure")
    finish_axes(ax, view)
    output = context.output_dir / f"{context.pair_key}_{context.timeframe}_pat_events.png"
    fig.savefig(output)
    plt.close(fig)
    score_output = plot_score_panel(
        data,
        context,
        "pat",
        "pattern structure scores",
        ("pat_impulse_up_score", "pat_impulse_down_score", "pat_range_contraction_score", "pat_retrace_long", "pat_retrace_short"),
    )
    return [output, score_output]


def plot_volatility_cycles(frame: DataFrame, context: PlotContext) -> list[Path]:
    data = add_pivot_structure(frame, PivotStructureConfig(strength=5, strengths=(3, 5, 8, 13), prefix="pa"))
    data = add_trendline_projection(data, TrendlineProjectionConfig(pivot_prefix="pa", output_prefix="tl", missing_pivot_mode="raise"))
    data = add_volatility_cycles(data, trendline_prefix="tl", output_prefix="vc")
    view = data.tail(context.tail).copy()
    fig, ax = plt.subplots(figsize=(15, 8), dpi=140)
    draw_candles(ax, view)
    draw_event_markers(ax, view, "vc_expansion_long", "high", "#15803d", "^", "expansion long")
    draw_event_markers(ax, view, "vc_expansion_short", "low", "#b91c1c", "v", "expansion short")
    draw_event_markers(ax, view, "vc_exhaustion_up", "high", "#f97316", "v", "exhaustion up")
    draw_event_markers(ax, view, "vc_exhaustion_down", "low", "#0ea5e9", "^", "exhaustion down")
    draw_event_markers(ax, view, "vc_suggested_entry_long", "low", "#84cc16", "^", "suggest long")
    draw_event_markers(ax, view, "vc_suggested_entry_short", "high", "#e11d48", "v", "suggest short")
    ax.set_title(f"{context.pair_label} {context.timeframe} volatility cycles")
    finish_axes(ax, view)
    output = context.output_dir / f"{context.pair_key}_{context.timeframe}_vc_events.png"
    fig.savefig(output)
    plt.close(fig)
    score_output = plot_score_panel(
        data,
        context,
        "vc",
        "volatility cycle scores",
        ("vc_compression_score", "vc_expansion_score", "vc_exhaustion_score", "vc_atr_ratio", "vc_rvol"),
    )
    return [output, score_output]


def plot_complex_volume(frame: DataFrame, context: PlotContext) -> list[Path]:
    data = add_complex_volume_indicators(frame)
    view = data.tail(context.tail).copy()
    fig, ax = plt.subplots(figsize=(15, 8), dpi=140)
    draw_candles(ax, view)
    draw_line_series(ax, view, "vol_rvwap", "#64748b", "rolling VWAP", 1.0)
    draw_line_series(ax, view, "vol_avwap", "#111827", "anchored VWAP", 1.5)
    draw_line_series(ax, view, "vol_avwap_upper", "#f97316", "AVWAP upper", 0.9, alpha=0.55)
    draw_line_series(ax, view, "vol_avwap_lower", "#0ea5e9", "AVWAP lower", 0.9, alpha=0.55)
    draw_event_markers(ax, view, "vol_liq_stoprun_long", "low", "#16a34a", "^", "stoprun long")
    draw_event_markers(ax, view, "vol_liq_stoprun_short", "high", "#dc2626", "v", "stoprun short")
    draw_event_markers(ax, view, "vol_avwap_reclaim_long", "high", "#15803d", "^", "AVWAP reclaim")
    draw_event_markers(ax, view, "vol_avwap_reject_short", "low", "#b91c1c", "v", "AVWAP reject")
    draw_event_markers(ax, view, "vol_vol_breakout_confirm_long", "high", "#22c55e", "^", "volume breakout")
    draw_event_markers(ax, view, "vol_vol_breakout_confirm_short", "low", "#f43f5e", "v", "volume breakdown")
    draw_event_markers(ax, view, "vol_suggested_entry_long", "low", "#84cc16", "^", "suggest long")
    draw_event_markers(ax, view, "vol_suggested_entry_short", "high", "#e11d48", "v", "suggest short")
    ax.set_title(f"{context.pair_label} {context.timeframe} complex volume")
    finish_axes(ax, view)
    output = context.output_dir / f"{context.pair_key}_{context.timeframe}_vol_events.png"
    fig.savefig(output)
    plt.close(fig)
    score_output = plot_score_panel(
        data,
        context,
        "vol",
        "complex volume scores",
        ("vol_delta_zscore", "vol_rvol", "vol_close_location", "vol_cvd_trend"),
    )
    return [output, score_output]


def plot_relative_strength(frame: DataFrame, benchmark: DataFrame, context: PlotContext, benchmark_label: str) -> list[Path]:
    data = add_relative_strength(frame, benchmark, prefix="rs")
    view = data.tail(context.tail).copy()
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), dpi=140, sharex=True)
    axes[0].plot(np.arange(len(view)), pd.to_numeric(view["close"], errors="coerce"), color="#111111", linewidth=1.1, label=context.pair_label)
    axes[0].plot(np.arange(len(view)), normalized_overlay(view["rs_benchmark_close"], view["close"]), color="#64748b", linewidth=1.0, label=f"{benchmark_label} normalized")
    axes[0].set_title(f"{context.pair_label} {context.timeframe} relative strength vs {benchmark_label}")
    axes[0].legend(loc="best")
    axes[1].plot(np.arange(len(view)), normalize_01(view["rs_line"]), color="#2563eb", linewidth=1.0, label="RS line normalized")
    axes[1].plot(np.arange(len(view)), pd.to_numeric(view["rs_percentile"], errors="coerce"), color="#f97316", linewidth=0.9, label="RS percentile")
    axes[1].legend(loc="best")
    for column, color, label in (
        ("rs_score_long", "#15803d", "long"),
        ("rs_score_short", "#b91c1c", "short"),
        ("rs_score_abs", "#1d4ed8", "abs"),
    ):
        axes[2].plot(np.arange(len(view)), pd.to_numeric(view[column], errors="coerce"), color=color, linewidth=1.0, label=label)
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].legend(loc="best")
    finish_panel_axes(fig, axes, view)
    output = context.output_dir / f"{context.pair_key}_{context.timeframe}_rs_scores.png"
    fig.savefig(output)
    plt.close(fig)
    return [output]


def plot_market_regime(frame: DataFrame, context: PlotContext) -> list[Path]:
    data = add_market_regime(frame, {"regime_confirm_bars": 3})
    view = data.tail(context.tail).copy()
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), dpi=140, sharex=True)
    axes[0].plot(np.arange(len(view)), pd.to_numeric(view["close"], errors="coerce"), color="#111111", linewidth=1.1)
    axes[0].set_title(f"{context.pair_label} {context.timeframe} market regime")
    for code, color, label in (
        (REGIME_CODE_BULL, "#dcfce7", "bull"),
        (REGIME_CODE_BEAR, "#fee2e2", "bear"),
        (REGIME_CODE_CRASH, "#fecaca", "crash"),
        (REGIME_CODE_CHOP, "#f1f5f9", "chop"),
    ):
        mask = pd.to_numeric(view["regime_code"], errors="coerce").eq(float(code)).to_numpy()
        shade_regions(axes[0], mask, color, label)
    for column, color, label in (
        ("regime_bull_score", "#15803d", "bull"),
        ("regime_bear_score", "#b91c1c", "bear"),
        ("regime_chop_score", "#64748b", "chop"),
        ("regime_crash_score", "#7f1d1d", "crash"),
    ):
        axes[1].plot(np.arange(len(view)), pd.to_numeric(view[column], errors="coerce"), color=color, linewidth=1.0, label=label)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(loc="best")
    for column, color, label in (
        ("regime_adx", "#2563eb", "ADX"),
        ("regime_atr_ratio", "#f97316", "ATR ratio"),
        ("regime_volume_ratio", "#16a34a", "volume ratio"),
    ):
        axes[2].plot(np.arange(len(view)), pd.to_numeric(view[column], errors="coerce"), color=color, linewidth=0.9, label=label)
    axes[2].legend(loc="best")
    finish_panel_axes(fig, axes, view)
    output = context.output_dir / f"{context.pair_key}_{context.timeframe}_regime_state.png"
    fig.savefig(output)
    plt.close(fig)
    return [output]


def plot_score_panel(data: DataFrame, context: PlotContext, prefix: str, title: str, extra_columns: tuple[str, ...]) -> Path:
    view = data.tail(context.tail).copy()
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), dpi=140, sharex=True)
    axes[0].plot(np.arange(len(view)), pd.to_numeric(view["close"], errors="coerce"), color="#111111", linewidth=1.1)
    axes[0].set_title(f"{context.pair_label} {context.timeframe} {title}")
    for suffix, color, label in (
        ("score_long", "#15803d", "long"),
        ("score_short", "#b91c1c", "short"),
        ("score_abs", "#1d4ed8", "abs"),
    ):
        column = f"{prefix}_{suffix}"
        if column in view.columns:
            axes[1].plot(np.arange(len(view)), pd.to_numeric(view[column], errors="coerce"), color=color, linewidth=1.0, label=label)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(loc="best")
    for column in extra_columns:
        if column not in view.columns:
            continue
        axes[2].plot(np.arange(len(view)), pd.to_numeric(view[column], errors="coerce"), linewidth=0.9, label=column.removeprefix(f"{prefix}_"))
    axes[2].legend(loc="best")
    finish_panel_axes(fig, axes, view)
    output = context.output_dir / f"{context.pair_key}_{context.timeframe}_{prefix}_scores.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def draw_candles(ax: plt.Axes, view: DataFrame) -> None:
    width = 0.55
    xs = np.arange(len(view))
    for x, (_, row) in zip(xs, view.iterrows(), strict=False):
        open_ = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        color = "#159467" if close >= open_ else "#b83232"
        ax.vlines(x, low, high, color=color, linewidth=0.55, alpha=0.55)
        ax.add_patch(
            Rectangle(
                (x - width / 2.0, min(open_, close)),
                width,
                max(abs(close - open_), 1e-12),
                facecolor=color,
                edgecolor=color,
                alpha=0.32,
                linewidth=0.2,
            )
        )


def draw_structural_line(ax: plt.Axes, data: DataFrame, view: DataFrame, side: str, rank: int, color: str) -> list[tuple[float, float]]:
    row = data.iloc[-1]
    prefix = f"stl_{side}"
    required = (
        f"{prefix}_slope_rank{rank}",
        f"{prefix}_intercept_rank{rank}",
        f"{prefix}_touch_start_index_rank{rank}",
        f"{prefix}_touch_end_index_rank{rank}",
    )
    if any(column not in row.index or pd.isna(row[column]) for column in required):
        return []
    slope = float(row[f"{prefix}_slope_rank{rank}"])
    intercept = float(row[f"{prefix}_intercept_rank{rank}"])
    start = float(row[f"{prefix}_touch_start_index_rank{rank}"])
    end = float(row[f"{prefix}_touch_end_index_rank{rank}"])
    view_start = float(view["pa_bar_index"].iloc[0])
    view_end = float(view["pa_bar_index"].iloc[-1])
    x1 = max(start, view_start)
    x2 = min(end, view_end)
    if x2 > x1:
        ax.plot(
            [x_for_bar(data, view, x1), x_for_bar(data, view, x2)],
            [line_y(intercept, slope, x1), line_y(intercept, slope, x2)],
            color=color,
            linewidth=3.0 - rank * 0.45,
            alpha=0.95 - rank * 0.15,
            label=f"{side} rank{rank}",
        )
    return draw_pivot_touch_markers(ax, data, view, side, rank, color, slope, intercept, start, end)


def draw_pivot_touch_markers(
    ax: plt.Axes,
    data: DataFrame,
    view: DataFrame,
    side: str,
    rank: int,
    color: str,
    slope: float,
    intercept: float,
    start: float,
    end: float,
) -> list[tuple[float, float]]:
    strength = 21
    pivot_col = f"pa_pivot_{'high' if side == 'resistance' else 'low'}_{strength}"
    if pivot_col not in data.columns:
        return []
    touches: list[tuple[float, float]] = []
    pivots = data[data[pivot_col].notna()].copy()
    pivots["pivot_bar"] = pd.to_numeric(pivots["pa_bar_index"], errors="coerce") - float(strength)
    pivots = pivots[(pivots["pivot_bar"] >= start - 1.0) & (pivots["pivot_bar"] <= end + 1.0)]
    for _, pivot in pivots.iterrows():
        pivot_bar = float(pivot["pivot_bar"])
        pivot_idx = nearest_bar_index(data, pivot_bar)
        if pivot_idx not in view.index:
            continue
        price = float(pivot[pivot_col])
        line_price = line_y(intercept, slope, pivot_bar)
        tolerance = max(float(pivot["pa_atr"]) * 0.85, abs(price) * 0.004)
        if abs(price - line_price) <= tolerance:
            xpos = view.index.get_loc(pivot_idx)
            ax.scatter(
                [xpos],
                [price],
                color=color,
                s=max(20, 40 - rank * 7),
                marker="v" if side == "resistance" else "^",
                zorder=5,
            )
            touches.append((pivot_bar, price))
    return touches


def draw_line_series(ax: plt.Axes, view: DataFrame, column: str, color: str, label: str, linewidth: float, *, alpha: float = 0.82) -> None:
    if column not in view.columns:
        return
    values = pd.to_numeric(view[column], errors="coerce")
    if values.notna().sum() < 2:
        return
    ax.plot(np.arange(len(view)), values, color=color, linewidth=linewidth, alpha=alpha, label=label)


def draw_conditional_line_series(
    ax: plt.Axes,
    view: DataFrame,
    value_column: str,
    condition_column: str,
    color: str,
    label: str,
    linewidth: float,
    *,
    alpha: float = 0.82,
) -> None:
    if value_column not in view.columns:
        return
    values = pd.to_numeric(view[value_column], errors="coerce").where(bool_array(view, condition_column))
    if values.notna().sum() < 2:
        return
    ax.plot(np.arange(len(view)), values, color=color, linewidth=linewidth, alpha=alpha, label=label)


def draw_event_markers(
    ax: plt.Axes,
    view: DataFrame,
    column: str,
    price_column: str,
    color: str,
    marker: str,
    label: str,
    *,
    size: float = 35,
    edgecolor: str | None = None,
) -> None:
    if column not in view.columns:
        return
    mask = view[column].fillna(False).astype("bool")
    if not mask.any():
        return
    xs = np.flatnonzero(mask.to_numpy())
    prices = pd.to_numeric(view.loc[mask, price_column], errors="coerce")
    ax.scatter(xs, prices, color=color, marker=marker, s=size, label=label, zorder=6, edgecolors=edgecolor)


def draw_vp_market_context(ax: plt.Axes, view: DataFrame) -> None:
    xs = np.arange(len(view))
    plotted = False
    if "vp_market_context" not in view.columns:
        return
    context = pd.to_numeric(view["vp_market_context"], errors="coerce").fillna(0).astype("int8")
    for value, color, label in vp_market_context_styles():
        active = context.eq(value).to_numpy()
        if not active.any():
            continue
        lane = float(value)
        y_values = np.where(active, lane, np.nan)
        ax.fill_between(xs, lane - 0.32, lane + 0.32, where=active, step="post", color=color, alpha=0.20)
        ax.step(xs, y_values, where="post", color=color, linewidth=2.0, label=label)
        plotted = True
    ax.axhline(0.0, color="#111111", linewidth=0.8, alpha=0.35)
    ax.set_ylim(-2.65, 2.65)
    ax.set_yticks([-2.0, -1.0, 0.0, 1.0, 2.0])
    ax.set_yticklabels(["bear", "bearish chop", "undefined", "bullish chop", "bull"])
    ax.set_ylabel("VP Market context")
    if plotted:
        ax.legend(loc="upper left", ncol=4)


def draw_vp_node_action_lanes(ax: plt.Axes, view: DataFrame) -> None:
    xs = np.arange(len(view))
    plotted = False
    for column, color, label, lane in (
        ("vp_node_entry_long", "#00d5ff", "entry long", 2.0),
        ("vp_node_hold_long", "#16a34a", "hold long", 1.0),
        ("vp_node_exit_long", "#f97316", "exit long", 0.35),
        ("vp_node_exit_short", "#2563eb", "exit short", -0.35),
        ("vp_node_hold_short", "#dc2626", "hold short", -1.0),
        ("vp_node_entry_short", "#ff00d4", "entry short", -2.0),
    ):
        active = bool_array(view, column)
        if not active.any():
            continue
        y_values = np.where(active, lane, np.nan)
        ax.fill_between(xs, lane - 0.18, lane + 0.18, where=active, step="post", color=color, alpha=0.20)
        ax.step(xs, y_values, where="post", color=color, linewidth=1.8, label=label)
        plotted = True
    ax.axhline(0.0, color="#111111", linewidth=0.8, alpha=0.35)
    ax.set_ylim(-2.45, 2.45)
    ax.set_yticks([-2.0, -1.0, -0.35, 0.35, 1.0, 2.0])
    ax.set_yticklabels(["entry short", "hold short", "exit short", "exit long", "hold long", "entry long"])
    ax.set_ylabel("VP node action")
    if plotted:
        ax.legend(loc="upper left", ncol=3)


def bool_array(view: DataFrame, column: str) -> np.ndarray:
    if column not in view.columns:
        return np.zeros(len(view), dtype="bool")
    return view[column].fillna(False).astype("bool").to_numpy()


def shade_vp_market_context_regions(ax: plt.Axes, view: DataFrame) -> None:
    if "vp_market_context" not in view.columns:
        return
    context = pd.to_numeric(view["vp_market_context"], errors="coerce").fillna(0).astype("int8")
    for value, color, label in vp_market_context_styles():
        mask = context.eq(value).to_numpy()
        shade_regions(ax, mask, color, label)


def vp_market_context_styles() -> tuple[tuple[int, str, str], ...]:
    return (
        (2, "#00a5ff", "bull +2"),
        (1, "#22c55e", "bullish chop +1"),
        (-1, "#f59e0b", "bearish chop -1"),
        (-2, "#ff00b8", "bear -2"),
    )


def draw_price_markers(
    ax: plt.Axes,
    view: DataFrame,
    column: str,
    color: str,
    marker: str,
    label: str,
    *,
    alpha: float = 0.85,
    size: float = 28,
) -> None:
    if column not in view.columns:
        return
    prices = pd.to_numeric(view[column], errors="coerce")
    mask = prices.notna()
    if not mask.any():
        return
    xs = np.flatnonzero(mask.to_numpy())
    ax.scatter(xs, prices.loc[mask], color=color, marker=marker, s=size, label=label, alpha=alpha, zorder=5)


def normalized_overlay(series: Series, target: Series) -> Series:
    source = pd.to_numeric(series, errors="coerce")
    base = pd.to_numeric(target, errors="coerce")
    first_source = source.dropna().iloc[0] if source.notna().any() else np.nan
    first_base = base.dropna().iloc[0] if base.notna().any() else np.nan
    if not np.isfinite(first_source) or not np.isfinite(first_base) or first_source == 0.0:
        return source
    return source / first_source * first_base


def normalize_01(series: Series) -> Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    low = values.min()
    high = values.max()
    if not np.isfinite(low) or not np.isfinite(high) or abs(float(high - low)) <= 1e-12:
        return pd.Series(0.0, index=values.index, dtype="float64")
    return ((values - low) / (high - low)).clip(0.0, 1.0)


def shade_regions(ax: plt.Axes, mask: np.ndarray, color: str, label: str) -> None:
    if not mask.any():
        return
    start: int | None = None
    label_used = False
    for idx, active in enumerate(mask.tolist() + [False]):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            ax.axvspan(start, idx - 1, facecolor=color, alpha=0.28, linewidth=0.0, label=label if not label_used else None)
            label_used = True
            start = None


def finish_axes(ax: plt.Axes, view: DataFrame) -> None:
    step = max(len(view) // 8, 1)
    xs = np.arange(len(view))
    ax.set_xticks(xs[::step])
    ax.set_xticklabels([value.strftime("%m-%d %H:%M") for value in view.index[::step]], rotation=35, ha="right")
    ax.grid(True, alpha=0.18)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best")
    ax.figure.tight_layout()


def finish_panel_axes(fig: plt.Figure, axes: np.ndarray, view: DataFrame) -> None:
    step = max(len(view) // 8, 1)
    xs = np.arange(len(view))
    for ax in axes:
        ax.grid(True, alpha=0.18)
        handles, labels = ax.get_legend_handles_labels()
        if handles and ax.get_legend() is None:
            ax.legend(handles, labels, loc="best")
    axes[-1].set_xticks(xs[::step])
    axes[-1].set_xticklabels([value.strftime("%m-%d %H:%M") for value in view.index[::step]], rotation=35, ha="right")
    fig.tight_layout()


def x_for_bar(data: DataFrame, view: DataFrame, bar: float) -> int:
    idx = nearest_bar_index(data, bar)
    if idx not in view.index:
        return int(np.clip(round(float(bar) - float(view["pa_bar_index"].iloc[0])), 0, len(view) - 1))
    return int(view.index.get_loc(idx))


def nearest_bar_index(data: DataFrame, bar: float) -> pd.Timestamp:
    bar_index = pd.to_numeric(data["pa_bar_index"], errors="coerce")
    return (bar_index - float(bar)).abs().idxmin()


def line_y(intercept: float, slope: float, x_value: float) -> float:
    return float(intercept) + float(slope) * float(x_value)


def find_candle_file(datadir: Path, pair: str, timeframe: str) -> Path | None:
    pair_key = normalize_pair_key(pair)
    candidate_keys = {pair_key}
    if pair_key.endswith("_USDT_USDT"):
        candidate_keys.add(pair_key.removesuffix("_USDT"))
    elif pair_key.endswith("_USDT"):
        candidate_keys.add(f"{pair_key}_USDT")
    candidates: list[Path] = []
    for path in datadir.rglob(f"*-{timeframe}*"):
        if not path.is_file():
            continue
        name = path.name
        if "funding_rate" in name or "-mark" in name or "_mark" in name:
            continue
        parsed_key = normalize_pair_key(name.split(f"-{timeframe}", 1)[0])
        if parsed_key in candidate_keys and path.suffix.lower() in {".feather", ".parquet", ".json", ".gz"}:
            candidates.append(path)
    return sorted(candidates, key=lambda item: ("binance" not in str(item).lower(), str(item)))[0] if candidates else None


def load_candle_frame(path: Path) -> DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".feather"):
        frame = pd.read_feather(path)
    elif suffixes.endswith(".parquet"):
        frame = pd.read_parquet(path)
    elif suffixes.endswith(".json") or suffixes.endswith(".json.gz"):
        frame = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported data file: {path}")
    missing = [column for column in OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns {missing}")
    frame = frame.loc[:, list(OHLCV_COLUMNS)].copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame["volume"] = frame["volume"].fillna(0.0)
    return frame.set_index("date", drop=False)


def normalize_pair_key(value: str) -> str:
    text = str(value or "").upper().strip()
    text = text.replace("/", "_").replace(":", "_").replace("-", "_")
    text = re.sub(r"[^A-Z0-9_]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


if __name__ == "__main__":
    raise SystemExit(main())

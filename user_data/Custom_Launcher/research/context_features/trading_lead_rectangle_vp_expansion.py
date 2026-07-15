from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from trader_rule_backtests import (
    clamp01,
    export_combined_signals,
    flag,
    fmt,
    fmt_pct,
    line_break_score,
    line_cross_score,
    mean_scores,
    near_distance_score,
    num,
    pressure_ratio_score,
    resolve_exit,
    rolling_level_cross_score,
    safe_name,
    summarize_trades,
    volume_ratio_score,
    vp_guard_score,
)


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT = USER_DATA_DIR / "research_news_data" / "context_features" / "confluence_cache" / "trader_confluence_1h_latest.parquet"
DEFAULT_OUTPUT_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"


@dataclass(frozen=True)
class ExitProfile:
    profile_id: str
    hold_hours: int
    stop_loss: float
    take_profit: float


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    concept_id: str
    direction: str
    trader_question: str
    visible_market_state: str
    thresholds: tuple[float, ...]
    score_builder: Callable[[DataFrame], Series]
    gate_builder: Callable[[DataFrame], Series]
    exit_profiles: tuple[ExitProfile, ...]


COMMON_EXITS = (
    ExitProfile("fast_follow_through", 10, 0.020, 0.050),
    ExitProfile("balanced_follow_through", 24, 0.025, 0.055),
    ExitProfile("wide_acceptance", 48, 0.035, 0.075),
    ExitProfile("quick_pop", 12, 0.018, 0.035),
)


SHORT_EXITS = (
    ExitProfile("fast_breakdown", 10, 0.020, 0.050),
    ExitProfile("balanced_breakdown", 24, 0.025, 0.055),
    ExitProfile("wide_breakdown", 36, 0.035, 0.065),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand rectangle/compression plus VP trader-rule variants.")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="rectangle_vp_expansion_20260605")
    parser.add_argument("--round-trip-fee", type=float, default=0.0010)
    parser.add_argument("--slippage", type=float, default=0.0002)
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    variants = build_variants()
    plan = {
        "mode": "setup-only unless --execute is supplied",
        "snapshot": str(args.snapshot),
        "purpose": "Expand the strongest rectangle breakout plus VP context lead into nearby trader-readable variants.",
        "entry_timing": "Signal at hour close, enter next candle open.",
        "variant_count": len(variants),
        "variants": [variant.variant_id for variant in variants],
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.snapshot.exists():
        raise FileNotFoundError(args.snapshot)

    frame = pd.read_parquet(args.snapshot)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume"):
        if column not in frame:
            raise ValueError(f"Missing required column: {column}")

    summary, trades, selected = run_suite(
        frame,
        variants,
        round_trip_fee=float(args.round_trip_fee),
        slippage=float(args.slippage),
        min_trades=int(args.min_trades),
    )
    combined_trades = simulate_selected(frame, variants, selected, round_trip_fee=args.round_trip_fee, slippage=args.slippage)
    combined_summary = summarize_trades(
        combined_trades,
        "rectangle_vp_expansion_selected_block",
        "rectangle_vp_expansion",
        "both",
        "Can expanded rectangle/compression plus VP variants produce a broader, still-clean breakout rule block?",
        "Selected variants are non-overlapped by priority, then entered next candle open.",
        frame,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = safe_name(args.tag)
    summary_path = args.output_dir / f"trading_lead_{tag}_summary.csv"
    trades_path = args.output_dir / f"trading_lead_{tag}_trades.csv"
    selected_path = args.output_dir / f"trading_lead_{tag}_selected.csv"
    combined_path = args.output_dir / f"trading_lead_{tag}_combined_trades.csv"
    signals_path = args.output_dir / f"trading_lead_signals_{tag}_selected.parquet"
    markdown_path = args.output_dir / f"trading_lead_{tag}.md"
    meta_path = args.output_dir / f"trading_lead_{tag}_meta.json"

    summary.to_csv(summary_path, index=False)
    trades.to_csv(trades_path, index=False)
    selected.to_csv(selected_path, index=False)
    combined_trades.to_csv(combined_path, index=False)
    export_combined_signals(combined_trades, signals_path)
    markdown_path.write_text(markdown_report(summary, selected, combined_summary), encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                **plan,
                "snapshot_rows": int(len(frame)),
                "snapshot_start": str(frame["date"].min()),
                "snapshot_end": str(frame["date"].max()),
                "summary": str(summary_path),
                "trades": str(trades_path),
                "selected": str(selected_path),
                "combined_trades": str(combined_path),
                "signals": str(signals_path),
                "markdown": str(markdown_path),
                "combined_summary": combined_summary,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "trades": str(trades_path),
                "selected": str(selected_path),
                "combined_trades": str(combined_path),
                "signals": str(signals_path),
                "markdown": str(markdown_path),
                "variants_tested": int(len(summary)),
                "selected_variants": int(len(selected)),
                "combined_return": combined_summary.get("total_return"),
                "combined_trades_count": combined_summary.get("trades"),
            },
            indent=2,
        )
    )
    return 0


def build_variants() -> tuple[VariantSpec, ...]:
    thresholds = (0.35, 0.45, 0.55, 0.65, 0.75)
    strict_thresholds = (0.45, 0.55, 0.65, 0.75)
    return (
        VariantSpec(
            "rect_4h_upper_break_vp_long",
            "rectangle_upper_break_vp",
            "long",
            "When a 4h rectangle breaks upward and VP supports the move, does price follow through?",
            "4h rectangle exists, close breaks its upper boundary, VP is not fighting the long, and volume/structure helps.",
            thresholds,
            lambda f: mean_scores(f, [rect_break_score(f, "4h", "long"), "st_4h_pg2_rectangle_indicator_score", vp_stack_score(f, "long"), volume_ratio_score(f, 24), pressure_ratio_score(f, 24, "long")]),
            lambda f: flag(f, "st_4h_pg2_rectangle_pattern_present") & rect_break_score(f, "4h", "long").gt(0.0) & vp_guard_score(f, "long").ge(0.25),
            COMMON_EXITS,
        ),
        VariantSpec(
            "rect_1d_upper_break_vp_long",
            "rectangle_upper_break_vp",
            "long",
            "When a daily rectangle breaks upward and VP supports the move, does price follow through?",
            "1d rectangle exists, close breaks its upper boundary, and 1h/4h VP or volume confirms.",
            thresholds,
            lambda f: mean_scores(f, [rect_break_score(f, "1d", "long"), "st_1d_pg2_rectangle_indicator_score", vp_stack_score(f, "long"), volume_ratio_score(f, 24), pressure_ratio_score(f, 24, "long")]),
            lambda f: flag(f, "st_1d_pg2_rectangle_pattern_present") & rect_break_score(f, "1d", "long").gt(0.0) & vp_guard_score(f, "long").ge(0.25),
            COMMON_EXITS,
        ),
        VariantSpec(
            "rect_multi_tf_upper_break_long",
            "rectangle_multi_tf_break",
            "long",
            "If either 4h or 1d rectangle resistance breaks and the broader structure agrees, is the upside cleaner?",
            "A higher-timeframe rectangle breaks upward with bullish structure and no strong VP objection.",
            strict_thresholds,
            lambda f: mean_scores(f, [rect_any_break_score(f, "long"), rectangle_presence_score(f), "conf_structure_breakout_trigger_score", "conf_structure_bullish_state_score", vp_stack_score(f, "long"), "conf_volume_bullish_confirmation"]),
            lambda f: rect_any_break_score(f, "long").gt(0.0) & rectangle_presence_score(f).ge(0.5) & num(f, "conf_structure_breakout_trigger_score").ge(0.20),
            COMMON_EXITS,
        ),
        VariantSpec(
            "rect_squeeze_upper_release_long",
            "rectangle_squeeze_release",
            "long",
            "When a rectangle is squeezed and breaks upward, does the release travel further?",
            "Rectangle/compression squeeze is active, price breaks the upper boundary, and volume expands.",
            thresholds,
            lambda f: mean_scores(f, [rect_any_break_score(f, "long"), squeeze_score(f), "conf_price_compression_24h", volume_ratio_score(f, 24), "conf_volume_bullish_impulse_short", vp_stack_score(f, "long")]),
            lambda f: rect_any_break_score(f, "long").gt(0.0) & squeeze_score(f).ge(0.5) & volume_ratio_score(f, 24).ge(0.35),
            COMMON_EXITS,
        ),
        VariantSpec(
            "rect_break_above_vah_long",
            "rectangle_break_value_area_acceptance",
            "long",
            "If rectangle resistance breaks while price accepts above VP value area, does continuation improve?",
            "Rectangle break upward plus price above/through VAH or VP long score says value is accepting higher.",
            thresholds,
            lambda f: mean_scores(f, [rect_any_break_score(f, "long"), vah_accept_score(f), "st_1h_vp_score_long", "st_4h_vp_score_long", "conf_volume_bullish_confirmation"]),
            lambda f: rect_any_break_score(f, "long").gt(0.0) & vah_accept_score(f).ge(0.35),
            COMMON_EXITS,
        ),
        VariantSpec(
            "rect_break_lvn_thin_air_long",
            "rectangle_break_lvn_travel",
            "long",
            "If rectangle resistance breaks into thin VP liquidity above, does price travel faster?",
            "Upper rectangle break with LVN/thinness above, meaning less visible VP resistance immediately overhead.",
            thresholds,
            lambda f: mean_scores(f, [rect_any_break_score(f, "long"), "st_1h_vp_lvn_above_thinness", "st_4h_vp_lvn_above_thinness", "conf_lvn_fast_travel_up_score", "conf_volume_bullish_confirmation"]),
            lambda f: rect_any_break_score(f, "long").gt(0.0) & lvn_thinness_score(f, "long").ge(0.25),
            COMMON_EXITS,
        ),
        VariantSpec(
            "rect_break_tlv2_confluence_long",
            "rectangle_break_tlv2_confluence",
            "long",
            "If rectangle resistance breaks while TLV2 resistance also gives way, is that a stronger long?",
            "Rectangle upper break and nearby TLV2 resistance break point to the same breakout area.",
            thresholds,
            lambda f: mean_scores(f, [rect_any_break_score(f, "long"), tlv2_break_score(f, "long"), "st_1h_tlv2_resistance_score_rank0", vp_stack_score(f, "long"), volume_ratio_score(f, 24)]),
            lambda f: rect_any_break_score(f, "long").gt(0.0) & tlv2_break_score(f, "long").gt(0.0),
            COMMON_EXITS,
        ),
        VariantSpec(
            "rect_break_orderbook_accept_long",
            "rectangle_break_orderbook_acceptance",
            "long",
            "If rectangle resistance breaks and orderbook acceptance agrees, does that improve longs?",
            "Rectangle upper break plus orderbook acceptance, pressure agreement, or reduced breakout failure risk.",
            thresholds,
            lambda f: mean_scores(f, [rect_any_break_score(f, "long"), orderbook_accept_score(f, "long"), vp_stack_score(f, "long"), "conf_volume_bullish_confirmation"]),
            lambda f: rect_any_break_score(f, "long").gt(0.0) & orderbook_accept_score(f, "long").ge(0.25),
            COMMON_EXITS,
        ),
        VariantSpec(
            "compression_range_break_vp_long",
            "compression_range_break_vp",
            "long",
            "When compression breaks the recent range high and VP supports long, does price continue?",
            "Compression is present, recent range high breaks, volume rises, and VP is not blocking the move.",
            thresholds,
            lambda f: mean_scores(f, ["conf_price_compression_24h", rolling_level_cross_score(f, 24, "high", "long"), vp_stack_score(f, "long"), volume_ratio_score(f, 24), pressure_ratio_score(f, 24, "long")]),
            lambda f: num(f, "conf_price_compression_24h").ge(0.40) & rolling_level_cross_score(f, 24, "high", "long").gt(0.0) & vp_guard_score(f, "long").ge(0.25),
            COMMON_EXITS,
        ),
        VariantSpec(
            "rect_4h_lower_break_vp_short",
            "rectangle_lower_break_vp",
            "short",
            "When a 4h rectangle breaks downward and VP supports the short, does price follow through?",
            "4h rectangle exists, close breaks its lower boundary, VP is not fighting the short, and volume/structure helps.",
            thresholds,
            lambda f: mean_scores(f, [rect_break_score(f, "4h", "short"), "st_4h_pg2_rectangle_indicator_score", vp_stack_score(f, "short"), volume_ratio_score(f, 24), pressure_ratio_score(f, 24, "short")]),
            lambda f: flag(f, "st_4h_pg2_rectangle_pattern_present") & rect_break_score(f, "4h", "short").gt(0.0) & vp_guard_score(f, "short").ge(0.25),
            SHORT_EXITS,
        ),
        VariantSpec(
            "rect_squeeze_lower_release_short",
            "rectangle_squeeze_release",
            "short",
            "When a rectangle is squeezed and breaks downward, does the release travel lower?",
            "Rectangle/compression squeeze is active, price breaks the lower boundary, and bearish volume expands.",
            thresholds,
            lambda f: mean_scores(f, [rect_any_break_score(f, "short"), squeeze_score(f), "conf_price_compression_24h", volume_ratio_score(f, 24), "conf_volume_bearish_impulse_short", vp_stack_score(f, "short")]),
            lambda f: rect_any_break_score(f, "short").gt(0.0) & squeeze_score(f).ge(0.5) & volume_ratio_score(f, 24).ge(0.35),
            SHORT_EXITS,
        ),
        VariantSpec(
            "rect_break_below_val_short",
            "rectangle_break_value_area_acceptance",
            "short",
            "If rectangle support breaks while price accepts below VP value area, does continuation improve?",
            "Rectangle break downward plus price below/through VAL or VP short score says value is accepting lower.",
            thresholds,
            lambda f: mean_scores(f, [rect_any_break_score(f, "short"), val_accept_score(f), "st_1h_vp_score_short", "st_4h_vp_score_short", "conf_volume_bearish_confirmation"]),
            lambda f: rect_any_break_score(f, "short").gt(0.0) & val_accept_score(f).ge(0.35),
            SHORT_EXITS,
        ),
    )


def run_suite(
    frame: DataFrame,
    variants: tuple[VariantSpec, ...],
    *,
    round_trip_fee: float,
    slippage: float,
    min_trades: int,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    trade_frames: list[DataFrame] = []
    for variant in variants:
        score = clamp01(variant.score_builder(frame)).fillna(0.0)
        gate = variant.gate_builder(frame).fillna(False).astype(bool)
        for exit_profile in variant.exit_profiles:
            for threshold in variant.thresholds:
                signal = gate & score.ge(threshold)
                trades = simulate_variant(
                    frame,
                    variant,
                    exit_profile,
                    signal,
                    score,
                    threshold,
                    round_trip_fee,
                    slippage,
                )
                if not trades.empty:
                    trade_frames.append(trades)
                summary_rows.append(
                    summarize_trades(
                        trades,
                        f"{variant.variant_id}__{exit_profile.profile_id}__thr_{threshold:.2f}",
                        variant.concept_id,
                        variant.direction,
                        variant.trader_question,
                        variant.visible_market_state,
                        frame,
                        threshold=threshold,
                    )
                    | {
                        "base_variant_id": variant.variant_id,
                        "exit_profile": exit_profile.profile_id,
                        "hold_hours": exit_profile.hold_hours,
                        "stop_loss": exit_profile.stop_loss,
                        "take_profit": exit_profile.take_profit,
                    }
                )
    summary = pd.DataFrame(summary_rows)
    all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    if summary.empty:
        return summary, all_trades, summary
    summary["usable_sample_size"] = summary["trades"].ge(min_trades)
    summary["positive_quality"] = (
        summary["usable_sample_size"]
        & summary["total_return"].gt(0.0)
        & summary["max_drawdown"].gt(-0.25)
        & (summary["profit_factor"].ge(1.20) | summary["win_rate"].ge(0.52))
    )
    summary["rank_score"] = (
        summary["total_return"].fillna(0.0)
        + summary["profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 5.0) / 10.0
        + summary["win_rate"].fillna(0.0) / 4.0
        + summary["trades"].fillna(0.0).clip(0.0, 40.0) / 200.0
        + summary["max_drawdown"].fillna(-1.0)
    )
    selected = (
        summary[summary["positive_quality"]]
        .sort_values(["rank_score", "total_return", "profit_factor"], ascending=[False, False, False])
        .groupby("base_variant_id", as_index=False, group_keys=False)
        .head(1)
        .groupby("concept_id", as_index=False, group_keys=False)
        .head(2)
        .head(12)
        .reset_index(drop=True)
    )
    return summary.sort_values(["positive_quality", "rank_score"], ascending=[False, False]), all_trades, selected


def simulate_variant(
    frame: DataFrame,
    variant: VariantSpec,
    exit_profile: ExitProfile,
    signal: Series,
    score: Series,
    threshold: float,
    round_trip_fee: float,
    slippage: float,
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    open_ = num(frame, "open")
    high = num(frame, "high")
    low = num(frame, "low")
    close = num(frame, "close")
    dates = frame["date"]
    i = 0
    last_entry_index = len(frame) - exit_profile.hold_hours - 2
    while i <= last_entry_index:
        if not bool(signal.iloc[i]):
            i += 1
            continue
        entry_idx = i + 1
        exit_idx = i + exit_profile.hold_hours
        entry_price = float(open_.iloc[entry_idx])
        if not np.isfinite(entry_price) or entry_price <= 0.0:
            i += 1
            continue
        exit_price, exit_reason = resolve_exit(
            variant.direction,
            entry_price,
            high.iloc[entry_idx : exit_idx + 1],
            low.iloc[entry_idx : exit_idx + 1],
            close.iloc[exit_idx],
            exit_profile.stop_loss,
            exit_profile.take_profit,
        )
        gross_return = exit_price / entry_price - 1.0 if variant.direction == "long" else entry_price / exit_price - 1.0
        rows.append(
            {
                "variant_id": f"{variant.variant_id}__{exit_profile.profile_id}__thr_{threshold:.2f}",
                "base_variant_id": variant.variant_id,
                "concept_id": variant.concept_id,
                "direction": variant.direction,
                "exit_profile": exit_profile.profile_id,
                "hold_hours": exit_profile.hold_hours,
                "stop_loss": exit_profile.stop_loss,
                "take_profit": exit_profile.take_profit,
                "threshold": threshold,
                "signal_date": dates.iloc[i],
                "entry_date": dates.iloc[entry_idx],
                "exit_date": dates.iloc[exit_idx],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "net_return": gross_return - round_trip_fee - slippage,
                "exit_reason": exit_reason,
                "signal_score": float(score.iloc[i]) if np.isfinite(score.iloc[i]) else np.nan,
            }
        )
        i = exit_idx + 1
    return pd.DataFrame(rows)


def simulate_selected(
    frame: DataFrame,
    variants: tuple[VariantSpec, ...],
    selected: DataFrame,
    *,
    round_trip_fee: float,
    slippage: float,
) -> DataFrame:
    if selected.empty:
        return DataFrame()
    by_id = {variant.variant_id: variant for variant in variants}
    candidates: list[dict[str, Any]] = []
    for priority, row in selected.reset_index(drop=True).iterrows():
        variant = by_id.get(str(row["base_variant_id"]))
        if variant is None:
            continue
        exit_profile = ExitProfile(
            str(row["exit_profile"]),
            int(row["hold_hours"]),
            float(row["stop_loss"]),
            float(row["take_profit"]),
        )
        score = clamp01(variant.score_builder(frame)).fillna(0.0)
        signal = variant.gate_builder(frame).fillna(False).astype(bool) & score.ge(float(row["threshold"]))
        for idx in np.flatnonzero(signal.to_numpy()):
            candidates.append({"idx": int(idx), "priority": int(priority), "variant": variant, "exit": exit_profile, "threshold": float(row["threshold"]), "score": float(score.iloc[idx])})
    candidates = sorted(candidates, key=lambda item: (item["idx"], item["priority"]))
    rows: list[dict[str, Any]] = []
    open_ = num(frame, "open")
    high = num(frame, "high")
    low = num(frame, "low")
    close = num(frame, "close")
    dates = frame["date"]
    flat_after = -1
    for candidate in candidates:
        i = int(candidate["idx"])
        variant = candidate["variant"]
        exit_profile = candidate["exit"]
        if i <= flat_after or i > len(frame) - exit_profile.hold_hours - 2:
            continue
        entry_idx = i + 1
        exit_idx = i + exit_profile.hold_hours
        entry_price = float(open_.iloc[entry_idx])
        if not np.isfinite(entry_price) or entry_price <= 0.0:
            continue
        exit_price, exit_reason = resolve_exit(
            variant.direction,
            entry_price,
            high.iloc[entry_idx : exit_idx + 1],
            low.iloc[entry_idx : exit_idx + 1],
            close.iloc[exit_idx],
            exit_profile.stop_loss,
            exit_profile.take_profit,
        )
        gross_return = exit_price / entry_price - 1.0 if variant.direction == "long" else entry_price / exit_price - 1.0
        rows.append(
            {
                "variant_id": "rectangle_vp_expansion_selected_block",
                "rule_id": variant.variant_id,
                "concept_id": variant.concept_id,
                "direction": variant.direction,
                "threshold": candidate["threshold"],
                "signal_date": dates.iloc[i],
                "entry_date": dates.iloc[entry_idx],
                "exit_date": dates.iloc[exit_idx],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "net_return": gross_return - round_trip_fee - slippage,
                "exit_reason": exit_reason,
                "signal_score": candidate["score"],
            }
        )
        flat_after = exit_idx
    return pd.DataFrame(rows)


def rect_break_score(frame: DataFrame, timeframe: str, direction: str) -> Series:
    line = f"st_{timeframe}_pg2_rectangle_upper" if direction == "long" else f"st_{timeframe}_pg2_rectangle_lower"
    return line_break_score(frame, line, direction)


def rect_any_break_score(frame: DataFrame, direction: str) -> Series:
    return pd.concat(
        [rect_break_score(frame, "4h", direction), rect_break_score(frame, "1d", direction), rect_break_score(frame, "1h", direction)],
        axis=1,
    ).max(axis=1, skipna=True)


def rectangle_presence_score(frame: DataFrame) -> Series:
    return pd.concat(
        [
            flag(frame, "st_4h_pg2_rectangle_pattern_present").astype(float),
            flag(frame, "st_1d_pg2_rectangle_pattern_present").astype(float),
            flag(frame, "st_1h_pg2_rectangle_pattern_present").astype(float),
        ],
        axis=1,
    ).max(axis=1, skipna=True)


def squeeze_score(frame: DataFrame) -> Series:
    return pd.concat(
        [
            flag(frame, "st_4h_pg2_rectangle_squeeze_active").astype(float),
            flag(frame, "st_1d_pg2_rectangle_squeeze_active").astype(float),
            flag(frame, "st_1h_pg2_rectangle_squeeze_active").astype(float),
            flag(frame, "st_4h_pg2_compression_squeeze_active").astype(float),
            flag(frame, "st_1d_pg2_compression_squeeze_active").astype(float),
            num(frame, "conf_price_compression_24h").clip(0.0, 1.0),
        ],
        axis=1,
    ).max(axis=1, skipna=True)


def vp_stack_score(frame: DataFrame, direction: str) -> Series:
    side = "long" if direction == "long" else "short"
    return pd.concat(
        [
            num(frame, f"st_1h_vp_score_{side}").clip(0.0, 1.0),
            num(frame, f"st_4h_vp_score_{side}").clip(0.0, 1.0),
            num(frame, f"st_1d_vp_score_{side}").clip(0.0, 1.0),
            flag(frame, f"st_1h_vp_node_entry_{side}").astype(float),
            flag(frame, f"st_4h_vp_node_entry_{side}").astype(float),
            flag(frame, f"st_1h_vp_node_hold_{side}").astype(float),
        ],
        axis=1,
    ).max(axis=1, skipna=True)


def vah_accept_score(frame: DataFrame) -> Series:
    return pd.concat(
        [
            line_cross_score(frame, "st_1h_vp_vah", "long", 0.0),
            line_cross_score(frame, "st_4h_vp_vah", "long", 0.0),
            flag(frame, "st_1h_vp_above_value_area").astype(float),
            flag(frame, "st_4h_vp_above_value_area").astype(float),
            num(frame, "st_1h_vp_value_area_position").clip(0.0, 1.0),
        ],
        axis=1,
    ).max(axis=1, skipna=True)


def val_accept_score(frame: DataFrame) -> Series:
    value_area_pos_short = (1.0 - num(frame, "st_1h_vp_value_area_position")).clip(0.0, 1.0)
    return pd.concat(
        [
            line_cross_score(frame, "st_1h_vp_val", "short", 0.0),
            line_cross_score(frame, "st_4h_vp_val", "short", 0.0),
            flag(frame, "st_1h_vp_below_value_area").astype(float),
            flag(frame, "st_4h_vp_below_value_area").astype(float),
            value_area_pos_short,
        ],
        axis=1,
    ).max(axis=1, skipna=True)


def lvn_thinness_score(frame: DataFrame, direction: str) -> Series:
    if direction == "long":
        columns = ["st_1h_vp_lvn_above_thinness", "st_4h_vp_lvn_above_thinness", "conf_lvn_up_thinness"]
    else:
        columns = ["st_1h_vp_lvn_below_thinness", "st_4h_vp_lvn_below_thinness", "conf_lvn_down_thinness"]
    return pd.concat([num(frame, column).clip(0.0, 1.0) for column in columns], axis=1).max(axis=1, skipna=True)


def tlv2_break_score(frame: DataFrame, direction: str) -> Series:
    if direction == "long":
        return pd.concat(
            [
                line_cross_score(frame, "st_1h_tlv2_resistance_line_rank0", "long", 0.003),
                line_cross_score(frame, "st_4h_tlv2_resistance_line_rank0", "long", 0.003),
            ],
            axis=1,
        ).max(axis=1, skipna=True)
    return pd.concat(
        [
            line_cross_score(frame, "st_1h_tlv2_support_line_rank0", "short", 0.003),
            line_cross_score(frame, "st_4h_tlv2_support_line_rank0", "short", 0.003),
        ],
        axis=1,
    ).max(axis=1, skipna=True)


def orderbook_accept_score(frame: DataFrame, direction: str) -> Series:
    if direction == "long":
        columns = [
            "ob_spot_obts_breakout_acceptance_score",
            "ob_linear_obts_breakout_acceptance_score",
            "ob_inverse_obts_breakout_acceptance_score",
            "ob_spot_obts_pressure_price_agreement",
            "ob_linear_obts_pressure_price_agreement",
        ]
    else:
        columns = [
            "ob_spot_obts_breakdown_acceptance_score",
            "ob_linear_obts_breakdown_acceptance_score",
            "ob_inverse_obts_breakdown_acceptance_score",
            "ob_spot_obts_pressure_price_agreement",
            "ob_linear_obts_pressure_price_agreement",
        ]
    return pd.concat([num(frame, column).clip(0.0, 1.0) for column in columns], axis=1).max(axis=1, skipna=True)


def markdown_report(summary: DataFrame, selected: DataFrame, combined_summary: dict[str, Any]) -> str:
    lines = [
        "# Rectangle + VP Expansion Report",
        "",
        "This expands the strongest surviving lead family: rectangle/compression breakouts where VP context may improve entry quality.",
        "",
        "## Combined Selected Block",
        "",
        f"1. Trades: {combined_summary.get('trades', 0)}",
        f"2. Win rate: {fmt_pct(combined_summary.get('win_rate'))}",
        f"3. Total return: {fmt_pct(combined_summary.get('total_return'))}",
        f"4. Max drawdown: {fmt_pct(combined_summary.get('max_drawdown'))}",
        f"5. Profit factor: {fmt(combined_summary.get('profit_factor'))}",
        "",
        "## Selected Variants",
        "",
    ]
    if selected.empty:
        lines.append("No expanded variant passed the quality filter.")
    else:
        for idx, row in selected.reset_index(drop=True).iterrows():
            lines.append(f"{idx + 1}. `{row['variant_id']}`")
            lines.append(f"   - Trader question: {row['trader_question']}")
            lines.append(f"   - Market state: {row['visible_market_state']}")
            lines.append(f"   - Result: {int(row['trades'])} trades, {fmt_pct(row['win_rate'])} wins, {fmt_pct(row['total_return'])} return, {fmt_pct(row['max_drawdown'])} max drawdown, PF {fmt(row['profit_factor'])}.")
    lines.extend(["", "## Top 25 Variants", ""])
    top = summary.sort_values(["positive_quality", "rank_score"], ascending=[False, False]).head(25)
    for idx, row in top.reset_index(drop=True).iterrows():
        lines.append(f"{idx + 1}. `{row['variant_id']}`")
        lines.append(f"   - What a trader saw: {row['visible_market_state']}")
        lines.append(f"   - Result: {int(row['trades'])} trades, {fmt_pct(row['win_rate'])} wins, {fmt_pct(row['total_return'])} return, {fmt_pct(row['max_drawdown'])} max drawdown, PF {fmt(row['profit_factor'])}.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

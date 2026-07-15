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
    mean_scores,
    num,
    pressure_ratio_score,
    resolve_exit,
    rolling_level_cross_score,
    safe_name,
    summarize_trades,
    volume_ratio_score,
    vp_guard_score,
)
from trading_lead_rectangle_vp_expansion import (
    ExitProfile,
    lvn_thinness_score,
    orderbook_accept_score,
    rect_any_break_score,
    squeeze_score,
    vah_accept_score,
    val_accept_score,
    vp_stack_score,
)


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT = USER_DATA_DIR / "research_news_data" / "context_features" / "confluence_cache" / "trader_confluence_1h_latest.parquet"
DEFAULT_OUTPUT_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"


@dataclass(frozen=True)
class RefinedSpec:
    rule_id: str
    concept_id: str
    direction: str
    trader_question: str
    visible_market_state: str
    thresholds: tuple[float, ...]
    score_builder: Callable[[DataFrame], Series]
    gate_builder: Callable[[DataFrame], Series]
    exit_profiles: tuple[ExitProfile, ...]


LONG_EXITS = (
    ExitProfile("fast_10h_5pct", 10, 0.020, 0.050),
    ExitProfile("medium_24h_55pct", 24, 0.025, 0.055),
    ExitProfile("wide_48h_75pct", 48, 0.035, 0.075),
)

SHORT_EXITS = (
    ExitProfile("fast_8h_35pct", 8, 0.018, 0.035),
    ExitProfile("fast_10h_5pct", 10, 0.020, 0.050),
    ExitProfile("medium_18h_5pct", 18, 0.025, 0.050),
    ExitProfile("medium_24h_55pct", 24, 0.025, 0.055),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Refine rectangle/VP leads that were promising but noisy.")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="rectangle_vp_refinement_20260605")
    parser.add_argument("--round-trip-fee", type=float, default=0.0010)
    parser.add_argument("--slippage", type=float, default=0.0002)
    parser.add_argument("--min-trades", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    specs = build_specs()
    if not args.execute:
        print(json.dumps({"mode": "setup-only", "specs": [spec.rule_id for spec in specs], "count": len(specs)}, indent=2))
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
        specs,
        round_trip_fee=args.round_trip_fee,
        slippage=args.slippage,
        min_trades=args.min_trades,
    )
    combined = simulate_selected(frame, specs, selected, round_trip_fee=args.round_trip_fee, slippage=args.slippage)
    combined_summary = summarize_trades(
        combined,
        "rectangle_vp_refinement_selected_block",
        "rectangle_vp_refinement",
        "both",
        "Can focused filters repair noisy compression and weak VAL-break exits?",
        "Only selected refined variants are entered; signals are non-overlapped by rank.",
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
    combined.to_csv(combined_path, index=False)
    export_combined_signals(combined, signals_path)
    markdown_path.write_text(markdown_report(summary, selected, combined_summary), encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "snapshot": str(args.snapshot),
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
                "selected": str(selected_path),
                "signals": str(signals_path),
                "variants_tested": int(len(summary)),
                "selected_variants": int(len(selected)),
                "combined_trades": int(combined_summary.get("trades", 0)),
                "combined_return": combined_summary.get("total_return"),
                "combined_profit_factor": combined_summary.get("profit_factor"),
            },
            indent=2,
        )
    )
    return 0


def build_specs() -> tuple[RefinedSpec, ...]:
    thresholds = (0.35, 0.45, 0.55, 0.65, 0.75)
    return (
        RefinedSpec(
            "compression_break_vp_long_strong_compression",
            "compression_long_rework",
            "long",
            "Does compression breakout work better only when compression is genuinely strong?",
            "Price breaks a recent high after strong compression, with VP not fighting the long.",
            thresholds,
            lambda f: mean_scores(f, ["conf_price_compression_24h", rolling_level_cross_score(f, 24, "high", "long"), vp_stack_score(f, "long"), volume_ratio_score(f, 24), pressure_ratio_score(f, 24, "long")]),
            lambda f: num(f, "conf_price_compression_24h").ge(0.75) & rolling_level_cross_score(f, 24, "high", "long").gt(0.0) & vp_guard_score(f, "long").ge(0.25),
            LONG_EXITS,
        ),
        RefinedSpec(
            "compression_break_vp_long_volume_persistent",
            "compression_long_rework",
            "long",
            "Does compression breakout need persistent 24h volume, not just a one-candle pop?",
            "Compression break with 24h volume above normal and positive pressure.",
            thresholds,
            lambda f: mean_scores(f, ["conf_price_compression_24h", num(f, "px_volume_z_24h").clip(0.0, 4.0) / 4.0, pressure_ratio_score(f, 24, "long"), vp_stack_score(f, "long")]),
            lambda f: num(f, "conf_price_compression_24h").ge(0.55) & rolling_level_cross_score(f, 24, "high", "long").gt(0.0) & num(f, "px_volume_z_24h").ge(1.40) & pressure_ratio_score(f, 24, "long").ge(0.35),
            LONG_EXITS,
        ),
        RefinedSpec(
            "compression_break_vp_long_value_acceptance",
            "compression_long_rework",
            "long",
            "Does compression breakout need VP value acceptance above VAH?",
            "Compression and range break plus price accepted above value area.",
            thresholds,
            lambda f: mean_scores(f, ["conf_price_compression_24h", vah_accept_score(f), "st_1h_vp_score_long", "st_4h_vp_score_long", "conf_volume_bullish_confirmation"]),
            lambda f: num(f, "conf_price_compression_24h").ge(0.45) & rolling_level_cross_score(f, 24, "high", "long").gt(0.0) & vah_accept_score(f).ge(0.55),
            LONG_EXITS,
        ),
        RefinedSpec(
            "compression_break_vp_long_not_overextended",
            "compression_long_rework",
            "long",
            "Does compression breakout fail less when price is not already stretched at the range top?",
            "Compression break occurs, but range position is not extremely overextended.",
            thresholds,
            lambda f: mean_scores(f, ["conf_price_compression_24h", rolling_level_cross_score(f, 24, "high", "long"), vp_stack_score(f, "long"), volume_ratio_score(f, 24)]),
            lambda f: num(f, "conf_price_compression_24h").ge(0.45) & rolling_level_cross_score(f, 24, "high", "long").gt(0.0) & num(f, "px_range_position_24h").between(0.55, 0.93),
            LONG_EXITS,
        ),
        RefinedSpec(
            "compression_break_vp_long_orderbook_accept",
            "compression_long_rework",
            "long",
            "Does orderbook acceptance filter the noisy compression breakout long?",
            "Compression break plus orderbook acceptance or pressure agreement.",
            thresholds,
            lambda f: mean_scores(f, ["conf_price_compression_24h", rolling_level_cross_score(f, 24, "high", "long"), orderbook_accept_score(f, "long"), vp_stack_score(f, "long")]),
            lambda f: num(f, "conf_price_compression_24h").ge(0.45) & rolling_level_cross_score(f, 24, "high", "long").gt(0.0) & orderbook_accept_score(f, "long").ge(0.35),
            LONG_EXITS,
        ),
        RefinedSpec(
            "rect_val_short_no_bull_volume",
            "val_break_short_rework",
            "short",
            "Does VAL breakdown work better when bullish volume is not fighting the short?",
            "Rectangle support breaks below value, but bullish volume/VP opposition is low.",
            thresholds,
            lambda f: mean_scores(f, [rect_any_break_score(f, "short"), val_accept_score(f), vp_short_minus_long_score(f), quiet_bull_volume_score(f), pressure_ratio_score(f, 24, "short")]),
            lambda f: rect_any_break_score(f, "short").gt(0.0) & val_accept_score(f).ge(0.35) & quiet_bull_volume_score(f).ge(0.55),
            SHORT_EXITS,
        ),
        RefinedSpec(
            "rect_val_short_vp_opposition_low",
            "val_break_short_rework",
            "short",
            "Does VAL breakdown need long-side VP opposition to be weak?",
            "Price breaks below rectangle/value and VP long scores are weak compared with short scores.",
            thresholds,
            lambda f: mean_scores(f, [rect_any_break_score(f, "short"), val_accept_score(f), vp_short_minus_long_score(f), "st_1h_vp_score_short", "st_4h_vp_score_short"]),
            lambda f: rect_any_break_score(f, "short").gt(0.0) & val_accept_score(f).ge(0.35) & vp_short_minus_long_score(f).ge(0.45),
            SHORT_EXITS,
        ),
        RefinedSpec(
            "rect_val_short_moderate_volume_break",
            "val_break_short_rework",
            "short",
            "Does VAL breakdown work better as a controlled break, not a crowded high-volume panic candle?",
            "Break below value happens with bearish pressure, but recent volume z-score is not extremely hot.",
            thresholds,
            lambda f: mean_scores(f, [rect_any_break_score(f, "short"), val_accept_score(f), quiet_bull_volume_score(f), pressure_ratio_score(f, 24, "short"), range_mid_short_score(f)]),
            lambda f: rect_any_break_score(f, "short").gt(0.0) & val_accept_score(f).ge(0.35) & num(f, "px_volume_z_6h").le(0.85) & pressure_ratio_score(f, 24, "short").ge(0.25),
            SHORT_EXITS,
        ),
        RefinedSpec(
            "rect_val_short_range_mid_break",
            "val_break_short_rework",
            "short",
            "Does VAL breakdown work better before price has already reached the range floor?",
            "Break below value while price still has room to move toward the lower range.",
            thresholds,
            lambda f: mean_scores(f, [rect_any_break_score(f, "short"), val_accept_score(f), range_mid_short_score(f), vp_short_minus_long_score(f)]),
            lambda f: rect_any_break_score(f, "short").gt(0.0) & val_accept_score(f).ge(0.35) & num(f, "px_range_position_24h").between(0.35, 0.70),
            SHORT_EXITS,
        ),
        RefinedSpec(
            "rect_squeeze_long_quality_keep",
            "quality_subset_keep",
            "long",
            "Does the cleaner rectangle squeeze-release long remain useful as a standalone lead?",
            "Rectangle/compression squeeze releases upward with strong score.",
            thresholds,
            lambda f: mean_scores(f, [rect_any_break_score(f, "long"), squeeze_score(f), "conf_price_compression_24h", "conf_volume_bullish_impulse_short", vp_stack_score(f, "long")]),
            lambda f: rect_any_break_score(f, "long").gt(0.0) & squeeze_score(f).ge(0.5) & volume_ratio_score(f, 24).ge(0.35),
            LONG_EXITS,
        ),
        RefinedSpec(
            "rect_orderbook_accept_long_quality_keep",
            "quality_subset_keep",
            "long",
            "Does rectangle break with orderbook acceptance remain one of the clean leads?",
            "Rectangle resistance breaks and orderbook acceptance agrees.",
            thresholds,
            lambda f: mean_scores(f, [rect_any_break_score(f, "long"), orderbook_accept_score(f, "long"), vp_stack_score(f, "long"), "conf_volume_bullish_confirmation"]),
            lambda f: rect_any_break_score(f, "long").gt(0.0) & orderbook_accept_score(f, "long").ge(0.25),
            LONG_EXITS,
        ),
    )


def quiet_bull_volume_score(frame: DataFrame) -> Series:
    bullish = num(frame, "conf_volume_bullish_confirmation").fillna(0.0).clip(0.0, 1.0)
    impulse = num(frame, "conf_volume_bullish_impulse_short").fillna(0.0).clip(0.0, 1.0)
    vol_hot = (num(frame, "px_volume_z_6h").fillna(0.0).clip(0.0, 3.0) / 3.0).clip(0.0, 1.0)
    return (1.0 - pd.concat([bullish, impulse, vol_hot], axis=1).mean(axis=1)).clip(0.0, 1.0)


def vp_short_minus_long_score(frame: DataFrame) -> Series:
    short_score = pd.concat(
        [
            num(frame, "st_1h_vp_score_short"),
            num(frame, "st_4h_vp_score_short"),
            flag(frame, "st_1h_vp_node_entry_short").astype(float),
            flag(frame, "st_1h_vp_node_hold_short").astype(float),
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    long_score = pd.concat(
        [
            num(frame, "st_1h_vp_score_long"),
            num(frame, "st_4h_vp_score_long"),
            flag(frame, "st_1h_vp_node_entry_long").astype(float),
            flag(frame, "st_1h_vp_node_hold_long").astype(float),
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    return ((short_score - long_score) + 1.0).div(2.0).clip(0.0, 1.0)


def range_mid_short_score(frame: DataFrame) -> Series:
    pos = num(frame, "px_range_position_24h")
    return (1.0 - (pos - 0.45).abs() / 0.45).clip(0.0, 1.0)


def run_suite(
    frame: DataFrame,
    specs: tuple[RefinedSpec, ...],
    *,
    round_trip_fee: float,
    slippage: float,
    min_trades: int,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    rows: list[dict[str, Any]] = []
    trade_frames: list[DataFrame] = []
    for spec in specs:
        score = clamp01(spec.score_builder(frame)).fillna(0.0)
        gate = spec.gate_builder(frame).fillna(False).astype(bool)
        for exit_profile in spec.exit_profiles:
            for threshold in spec.thresholds:
                signal = gate & score.ge(threshold)
                trades = simulate_spec(frame, spec, exit_profile, signal, score, threshold, round_trip_fee, slippage)
                if not trades.empty:
                    trade_frames.append(trades)
                rows.append(
                    summarize_trades(
                        trades,
                        f"{spec.rule_id}__{exit_profile.profile_id}__thr_{threshold:.2f}",
                        spec.concept_id,
                        spec.direction,
                        spec.trader_question,
                        spec.visible_market_state,
                        frame,
                        threshold=threshold,
                    )
                    | {
                        "rule_id": spec.rule_id,
                        "exit_profile": exit_profile.profile_id,
                        "hold_hours": exit_profile.hold_hours,
                        "stop_loss": exit_profile.stop_loss,
                        "take_profit": exit_profile.take_profit,
                    }
                )
    summary = pd.DataFrame(rows)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    summary["usable_sample_size"] = summary["trades"].ge(min_trades)
    summary["positive_quality"] = (
        summary["usable_sample_size"]
        & summary["total_return"].gt(0.0)
        & summary["max_drawdown"].gt(-0.20)
        & (summary["profit_factor"].ge(1.35) | summary["win_rate"].ge(0.55))
    )
    summary["rank_score"] = (
        summary["total_return"].fillna(0.0)
        + summary["profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 5.0) / 10.0
        + summary["win_rate"].fillna(0.0) / 4.0
        + summary["trades"].fillna(0.0).clip(0.0, 50.0) / 250.0
        + summary["max_drawdown"].fillna(-1.0)
    )
    selected = (
        summary[summary["positive_quality"]]
        .sort_values(["rank_score", "total_return", "profit_factor"], ascending=[False, False, False])
        .groupby("rule_id", as_index=False, group_keys=False)
        .head(1)
        .groupby("concept_id", as_index=False, group_keys=False)
        .head(3)
        .head(12)
        .reset_index(drop=True)
    )
    return summary.sort_values(["positive_quality", "rank_score"], ascending=[False, False]), trades, selected


def simulate_spec(
    frame: DataFrame,
    spec: RefinedSpec,
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
            spec.direction,
            entry_price,
            high.iloc[entry_idx : exit_idx + 1],
            low.iloc[entry_idx : exit_idx + 1],
            close.iloc[exit_idx],
            exit_profile.stop_loss,
            exit_profile.take_profit,
        )
        gross_return = exit_price / entry_price - 1.0 if spec.direction == "long" else entry_price / exit_price - 1.0
        rows.append(
            {
                "variant_id": f"{spec.rule_id}__{exit_profile.profile_id}__thr_{threshold:.2f}",
                "rule_id": spec.rule_id,
                "concept_id": spec.concept_id,
                "direction": spec.direction,
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
                "exit_profile": exit_profile.profile_id,
                "hold_hours": exit_profile.hold_hours,
                "stop_loss": exit_profile.stop_loss,
                "take_profit": exit_profile.take_profit,
            }
        )
        i = exit_idx + 1
    return pd.DataFrame(rows)


def simulate_selected(
    frame: DataFrame,
    specs: tuple[RefinedSpec, ...],
    selected: DataFrame,
    *,
    round_trip_fee: float,
    slippage: float,
) -> DataFrame:
    if selected.empty:
        return DataFrame()
    by_id = {spec.rule_id: spec for spec in specs}
    candidates: list[dict[str, Any]] = []
    for priority, row in selected.reset_index(drop=True).iterrows():
        spec = by_id.get(str(row["rule_id"]))
        if spec is None:
            continue
        exit_profile = ExitProfile(str(row["exit_profile"]), int(row["hold_hours"]), float(row["stop_loss"]), float(row["take_profit"]))
        score = clamp01(spec.score_builder(frame)).fillna(0.0)
        signal = spec.gate_builder(frame).fillna(False).astype(bool) & score.ge(float(row["threshold"]))
        for idx in np.flatnonzero(signal.to_numpy()):
            candidates.append({"idx": int(idx), "priority": int(priority), "spec": spec, "exit": exit_profile, "threshold": float(row["threshold"]), "score": float(score.iloc[idx])})
    rows: list[dict[str, Any]] = []
    open_ = num(frame, "open")
    high = num(frame, "high")
    low = num(frame, "low")
    close = num(frame, "close")
    dates = frame["date"]
    flat_after = -1
    for item in sorted(candidates, key=lambda x: (x["idx"], x["priority"])):
        i = item["idx"]
        spec = item["spec"]
        exit_profile = item["exit"]
        if i <= flat_after or i > len(frame) - exit_profile.hold_hours - 2:
            continue
        entry_idx = i + 1
        exit_idx = i + exit_profile.hold_hours
        entry_price = float(open_.iloc[entry_idx])
        if not np.isfinite(entry_price) or entry_price <= 0.0:
            continue
        exit_price, exit_reason = resolve_exit(
            spec.direction,
            entry_price,
            high.iloc[entry_idx : exit_idx + 1],
            low.iloc[entry_idx : exit_idx + 1],
            close.iloc[exit_idx],
            exit_profile.stop_loss,
            exit_profile.take_profit,
        )
        gross_return = exit_price / entry_price - 1.0 if spec.direction == "long" else entry_price / exit_price - 1.0
        rows.append(
            {
                "variant_id": "rectangle_vp_refinement_selected_block",
                "rule_id": spec.rule_id,
                "concept_id": spec.concept_id,
                "direction": spec.direction,
                "threshold": item["threshold"],
                "signal_date": dates.iloc[i],
                "entry_date": dates.iloc[entry_idx],
                "exit_date": dates.iloc[exit_idx],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "net_return": gross_return - round_trip_fee - slippage,
                "exit_reason": exit_reason,
                "signal_score": item["score"],
            }
        )
        flat_after = exit_idx
    return pd.DataFrame(rows)


def markdown_report(summary: DataFrame, selected: DataFrame, combined_summary: dict[str, Any]) -> str:
    lines = [
        "# Rectangle + VP Refinement Report",
        "",
        "This tries to repair the noisy broad compression long and improve the VAL-break short by adding trader-readable filters.",
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
        lines.append("No refined variant passed the quality filter.")
    else:
        for idx, row in selected.reset_index(drop=True).iterrows():
            lines.append(f"{idx + 1}. `{row['variant_id']}`")
            lines.append(f"   - Trader question: {row['trader_question']}")
            lines.append(f"   - Market state: {row['visible_market_state']}")
            lines.append(f"   - Result: {int(row['trades'])} trades, {fmt_pct(row['win_rate'])} wins, {fmt_pct(row['total_return'])} return, {fmt_pct(row['max_drawdown'])} max drawdown, PF {fmt(row['profit_factor'])}.")
    lines.extend(["", "## Top 30 Variants", ""])
    for idx, row in summary.sort_values(["positive_quality", "rank_score"], ascending=[False, False]).head(30).reset_index(drop=True).iterrows():
        lines.append(f"{idx + 1}. `{row['variant_id']}`")
        lines.append(f"   - What a trader saw: {row['visible_market_state']}")
        lines.append(f"   - Result: {int(row['trades'])} trades, {fmt_pct(row['win_rate'])} wins, {fmt_pct(row['total_return'])} return, {fmt_pct(row['max_drawdown'])} max drawdown, PF {fmt(row['profit_factor'])}.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

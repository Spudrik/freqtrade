from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame, Series

from trader_rule_backtests import (
    RuleSpec,
    clamp01,
    export_combined_signals,
    flag,
    fmt,
    fmt_pct,
    line_break_score,
    line_cross_score,
    line_retest_score,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine broad multi-timeframe custom-indicator BTC entry leads.")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y%m%d_mtf_indicator_mining"))
    parser.add_argument("--round-trip-fee", type=float, default=0.0010)
    parser.add_argument("--slippage", type=float, default=0.0002)
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--max-selected", type=int, default=60)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    rules = build_rules()
    plan = {
        "mode": "setup-only unless --execute is supplied",
        "snapshot": str(args.snapshot),
        "purpose": "Create many trader-readable MTF indicator entry leads for later confluence, exit, and risk work.",
        "entry_timing": "Signal at hour close, enter next candle open.",
        "rule_count": len(rules),
        "rule_ids": [rule.rule_id for rule in rules],
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.snapshot.exists():
        raise FileNotFoundError(args.snapshot)

    frame = pd.read_parquet(args.snapshot)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for column in ("open", "high", "low", "close"):
        if column not in frame:
            raise ValueError(f"Missing required column: {column}")

    summary, trades = run_suite(frame, rules, args.round_trip_fee, args.slippage)
    selected = select_candidates(summary, min_trades=args.min_trades, max_selected=args.max_selected)
    combined_trades = simulate_selected(frame, rules, selected, args.round_trip_fee, args.slippage)
    combined_summary = summarize_trades(
        combined_trades,
        "mtf_indicator_mining_selected_block",
        "mtf_indicator_mining",
        "both",
        "Can mined MTF indicator leads form a useful broad entry pool before confluence refinement?",
        "Selected MTF VP/TLV2/market-structure/pattern candidates are entered one at a time by priority.",
        frame,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = safe_name(str(args.tag))
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
                "rule_variants": int(len(summary)),
                "selected_candidates": int(len(selected)),
                "combined_return": combined_summary.get("total_return"),
                "combined_trades_count": combined_summary.get("trades"),
            },
            indent=2,
            default=str,
        )
    )
    return 0


def build_rules() -> tuple[RuleSpec, ...]:
    rules: list[RuleSpec] = []
    thresholds = (0.35, 0.45, 0.55, 0.65, 0.75)
    for tf in ("1h", "4h", "1d"):
        rules.extend(vp_rules(tf, thresholds))
        rules.extend(tlv2_rules(tf, thresholds))
        rules.extend(market_structure_rules(tf, thresholds))
        for pattern in ("triangle", "wedge", "compression", "rectangle"):
            rules.extend(pattern_rules(tf, pattern, thresholds))
    return tuple(rules)


def vp_rules(tf: str, thresholds: tuple[float, ...]) -> list[RuleSpec]:
    return [
        RuleSpec(
            f"mtf_{tf}_vp_poc_reclaim_long",
            "mtf_vp_poc_reclaim_long",
            "long",
            f"When price reclaims the {tf} VP point of control with bullish pressure, does the long continue?",
            f"Price is reclaiming {tf} POC and bullish VP/volume/structure agrees.",
            10,
            0.018,
            0.040,
            thresholds,
            lambda f, tf=tf: mean_scores(f, [line_cross_score(f, f"st_{tf}_vp_poc", "long"), line_break_score(f, f"st_{tf}_vp_poc", "long"), f"st_{tf}_vp_score_long", "conf_volume_bullish_confirmation", "conf_structure_bullish_state_score"]),
            lambda f, tf=tf: line_cross_score(f, f"st_{tf}_vp_poc", "long").gt(0.0) | (line_break_score(f, f"st_{tf}_vp_poc", "long").gt(0.0) & num(f, f"st_{tf}_vp_score_long").ge(0.35)),
        ),
        RuleSpec(
            f"mtf_{tf}_vp_poc_reject_short",
            "mtf_vp_poc_reject_short",
            "short",
            f"When price rejects or loses the {tf} VP point of control with bearish pressure, does downside continue?",
            f"Price is below/rejecting {tf} POC and bearish VP/volume/structure agrees.",
            10,
            0.018,
            0.040,
            thresholds,
            lambda f, tf=tf: mean_scores(f, [line_cross_score(f, f"st_{tf}_vp_poc", "short"), line_break_score(f, f"st_{tf}_vp_poc", "short"), f"st_{tf}_vp_score_short", "conf_volume_bearish_confirmation", "conf_structure_bearish_state_score"]),
            lambda f, tf=tf: line_cross_score(f, f"st_{tf}_vp_poc", "short").gt(0.0) | (line_break_score(f, f"st_{tf}_vp_poc", "short").gt(0.0) & num(f, f"st_{tf}_vp_score_short").ge(0.35)),
        ),
        RuleSpec(
            f"mtf_{tf}_vp_vah_accept_long",
            "mtf_vp_vah_accept_long",
            "long",
            f"When price accepts above {tf} VAH with bullish pressure, does it continue toward higher value?",
            f"Price is above {tf} VAH and VP/volume indicate value accepting higher.",
            12,
            0.020,
            0.050,
            thresholds,
            lambda f, tf=tf: mean_scores(f, [line_break_score(f, f"st_{tf}_vp_vah", "long"), f"st_{tf}_vp_score_long", f"st_{tf}_vp_lvn_above_thinness", "conf_volume_bullish_confirmation", "conf_structure_breakout_trigger_score"]),
            lambda f, tf=tf: line_break_score(f, f"st_{tf}_vp_vah", "long").gt(0.0) & num(f, f"st_{tf}_vp_score_long").ge(0.25),
        ),
        RuleSpec(
            f"mtf_{tf}_vp_val_accept_short",
            "mtf_vp_val_accept_short",
            "short",
            f"When price accepts below {tf} VAL with bearish pressure, does it continue toward lower value?",
            f"Price is below {tf} VAL and VP/volume indicate value accepting lower.",
            12,
            0.020,
            0.050,
            thresholds,
            lambda f, tf=tf: mean_scores(f, [line_break_score(f, f"st_{tf}_vp_val", "short"), f"st_{tf}_vp_score_short", f"st_{tf}_vp_lvn_below_thinness", "conf_volume_bearish_confirmation", "conf_structure_breakdown_trigger_score"]),
            lambda f, tf=tf: line_break_score(f, f"st_{tf}_vp_val", "short").gt(0.0) & num(f, f"st_{tf}_vp_score_short").ge(0.25),
        ),
        RuleSpec(
            f"mtf_{tf}_vp_val_reclaim_long",
            "mtf_vp_val_reclaim_long",
            "long",
            f"When price reclaims {tf} VAL after weakness, does it bounce?",
            f"Price reclaims {tf} VAL and bearish pressure is failing.",
            8,
            0.016,
            0.035,
            thresholds,
            lambda f, tf=tf: mean_scores(f, [line_cross_score(f, f"st_{tf}_vp_val", "long"), f"st_{tf}_vp_score_long", "conf_volume_bullish_confirmation", "conf_ob_breakdown_failure", "conf_structure_bullish_state_score"]),
            lambda f, tf=tf: line_cross_score(f, f"st_{tf}_vp_val", "long").gt(0.0) & num(f, f"st_{tf}_vp_score_long").ge(0.20),
        ),
        RuleSpec(
            f"mtf_{tf}_vp_vah_reject_short",
            "mtf_vp_vah_reject_short",
            "short",
            f"When price rejects {tf} VAH after strength, does it fade?",
            f"Price rejects/falls below {tf} VAH and bullish pressure is failing.",
            8,
            0.016,
            0.035,
            thresholds,
            lambda f, tf=tf: mean_scores(f, [line_cross_score(f, f"st_{tf}_vp_vah", "short"), f"st_{tf}_vp_score_short", "conf_volume_bearish_confirmation", "conf_ob_breakout_failure", "conf_structure_bearish_state_score"]),
            lambda f, tf=tf: line_cross_score(f, f"st_{tf}_vp_vah", "short").gt(0.0) & num(f, f"st_{tf}_vp_score_short").ge(0.20),
        ),
        RuleSpec(
            f"mtf_{tf}_vp_lvn_traverse_long",
            "mtf_vp_lvn_traverse_long",
            "long",
            f"When price breaks upward into thin {tf} VP liquidity, does it travel quickly?",
            f"LVN/thinness is above price and bullish breakout pressure appears.",
            8,
            0.018,
            0.045,
            thresholds,
            lambda f, tf=tf: mean_scores(f, [f"st_{tf}_vp_lvn_above_thinness", "conf_lvn_up_thinness", "conf_structure_breakout_trigger_score", "conf_volume_bullish_confirmation"]),
            lambda f, tf=tf: num(f, f"st_{tf}_vp_lvn_above_thinness").ge(0.20) & (flag(f, "px_close_breakout_24h") | num(f, "conf_structure_breakout_trigger_score").ge(0.25)),
        ),
        RuleSpec(
            f"mtf_{tf}_vp_lvn_traverse_short",
            "mtf_vp_lvn_traverse_short",
            "short",
            f"When price breaks downward into thin {tf} VP liquidity, does it travel quickly?",
            f"LVN/thinness is below price and bearish breakdown pressure appears.",
            8,
            0.018,
            0.045,
            thresholds,
            lambda f, tf=tf: mean_scores(f, [f"st_{tf}_vp_lvn_below_thinness", "conf_lvn_down_thinness", "conf_structure_breakdown_trigger_score", "conf_volume_bearish_confirmation"]),
            lambda f, tf=tf: num(f, f"st_{tf}_vp_lvn_below_thinness").ge(0.20) & (flag(f, "px_close_breakdown_24h") | num(f, "conf_structure_breakdown_trigger_score").ge(0.25)),
        ),
    ]


def tlv2_rules(tf: str, thresholds: tuple[float, ...]) -> list[RuleSpec]:
    return [
        RuleSpec(
            f"mtf_{tf}_tlv2_res_break_long",
            "mtf_tlv2_resistance_break_long",
            "long",
            f"When price breaks {tf} TLV2 resistance with volume, does the breakout continue?",
            f"Price crosses a scored {tf} TLV2 resistance line with bullish pressure.",
            12,
            0.020,
            0.050,
            thresholds,
            lambda f, tf=tf: mean_scores(f, [line_cross_score(f, f"st_{tf}_tlv2_resistance_line_rank0", "long", 0.002), f"st_{tf}_tlv2_resistance_score_rank0", "conf_volume_bullish_confirmation", "conf_structure_bullish_state_score", vp_guard_score(f, "long")]),
            lambda f, tf=tf: line_cross_score(f, f"st_{tf}_tlv2_resistance_line_rank0", "long", 0.002).gt(0.0) & num(f, f"st_{tf}_tlv2_resistance_score_rank0").ge(0.30),
        ),
        RuleSpec(
            f"mtf_{tf}_tlv2_sup_break_short",
            "mtf_tlv2_support_break_short",
            "short",
            f"When price breaks {tf} TLV2 support with volume, does downside continue?",
            f"Price crosses below a scored {tf} TLV2 support line with bearish pressure.",
            12,
            0.020,
            0.050,
            thresholds,
            lambda f, tf=tf: mean_scores(f, [line_cross_score(f, f"st_{tf}_tlv2_support_line_rank0", "short", 0.002), f"st_{tf}_tlv2_support_score_rank0", "conf_volume_bearish_confirmation", "conf_structure_bearish_state_score", vp_guard_score(f, "short")]),
            lambda f, tf=tf: line_cross_score(f, f"st_{tf}_tlv2_support_line_rank0", "short", 0.002).gt(0.0) & num(f, f"st_{tf}_tlv2_support_score_rank0").ge(0.30),
        ),
        RuleSpec(
            f"mtf_{tf}_tlv2_res_retest_long",
            "mtf_tlv2_resistance_retest_long",
            "long",
            f"After breaking {tf} TLV2 resistance, does a retest hold produce a better long?",
            f"Price retests old {tf} resistance as support and VP/pressure agrees.",
            24,
            0.020,
            0.050,
            thresholds,
            lambda f, tf=tf: mean_scores(f, [line_retest_score(f, f"st_{tf}_tlv2_resistance_line_rank0", "long", 0.006), f"st_{tf}_tlv2_resistance_score_rank0", vp_guard_score(f, "long"), "conf_volume_bullish_confirmation"]),
            lambda f, tf=tf: line_retest_score(f, f"st_{tf}_tlv2_resistance_line_rank0", "long", 0.006).gt(0.0) & num(f, f"st_{tf}_tlv2_resistance_score_rank0").ge(0.30),
        ),
        RuleSpec(
            f"mtf_{tf}_tlv2_sup_retest_short",
            "mtf_tlv2_support_retest_short",
            "short",
            f"After breaking {tf} TLV2 support, does a retest reject produce a better short?",
            f"Price retests old {tf} support as resistance and VP/pressure agrees.",
            24,
            0.020,
            0.050,
            thresholds,
            lambda f, tf=tf: mean_scores(f, [line_retest_score(f, f"st_{tf}_tlv2_support_line_rank0", "short", 0.006), f"st_{tf}_tlv2_support_score_rank0", vp_guard_score(f, "short"), "conf_volume_bearish_confirmation"]),
            lambda f, tf=tf: line_retest_score(f, f"st_{tf}_tlv2_support_line_rank0", "short", 0.006).gt(0.0) & num(f, f"st_{tf}_tlv2_support_score_rank0").ge(0.30),
        ),
    ]


def market_structure_rules(tf: str, thresholds: tuple[float, ...]) -> list[RuleSpec]:
    return [
        RuleSpec(
            f"mtf_{tf}_bos_bull_long",
            "mtf_bos_bull_continuation_long",
            "long",
            f"When {tf} BOS turns bullish with VP/volume agreement, does long continuation work?",
            f"{tf} market structure breaks bullish and value/volume agrees.",
            12,
            0.020,
            0.050,
            thresholds,
            lambda f, tf=tf: mean_scores(f, [f"st_{tf}_ms_bos_to_bull", f"st_{tf}_vp_score_long", "conf_volume_bullish_confirmation", "conf_structure_bullish_state_score"]),
            lambda f, tf=tf: flag(f, f"st_{tf}_ms_bos_to_bull") | (num(f, "conf_structure_bullish_state_score").ge(0.45) & num(f, f"st_{tf}_vp_score_long").ge(0.25)),
        ),
        RuleSpec(
            f"mtf_{tf}_bos_bear_short",
            "mtf_bos_bear_continuation_short",
            "short",
            f"When {tf} BOS turns bearish with VP/volume agreement, does short continuation work?",
            f"{tf} market structure breaks bearish and value/volume agrees.",
            12,
            0.020,
            0.050,
            thresholds,
            lambda f, tf=tf: mean_scores(f, [f"st_{tf}_ms_bos_to_bear", f"st_{tf}_vp_score_short", "conf_volume_bearish_confirmation", "conf_structure_bearish_state_score"]),
            lambda f, tf=tf: flag(f, f"st_{tf}_ms_bos_to_bear") | (num(f, "conf_structure_bearish_state_score").ge(0.45) & num(f, f"st_{tf}_vp_score_short").ge(0.25)),
        ),
        RuleSpec(
            f"mtf_{tf}_choch_bull_reversal_long",
            "mtf_choch_bull_reversal_long",
            "long",
            f"When {tf} CHoCH turns bullish at support/value, does reversal continuation work?",
            f"{tf} structure changes bullish near support/value and bearish pressure fades.",
            10,
            0.018,
            0.040,
            thresholds,
            lambda f, tf=tf: mean_scores(f, [f"st_{tf}_ms_choch_to_bull", f"st_{tf}_vp_score_long", near_distance_score(f, f"st_{tf}_tlv2_support_distance_atr_rank0"), "conf_volume_bullish_confirmation"]),
            lambda f, tf=tf: flag(f, f"st_{tf}_ms_choch_to_bull") & (num(f, f"st_{tf}_vp_score_long").ge(0.20) | near_distance_score(f, f"st_{tf}_tlv2_support_distance_atr_rank0").ge(0.30)),
        ),
        RuleSpec(
            f"mtf_{tf}_choch_bear_reversal_short",
            "mtf_choch_bear_reversal_short",
            "short",
            f"When {tf} CHoCH turns bearish at resistance/value, does reversal continuation work?",
            f"{tf} structure changes bearish near resistance/value and bullish pressure fades.",
            10,
            0.018,
            0.040,
            thresholds,
            lambda f, tf=tf: mean_scores(f, [f"st_{tf}_ms_choch_to_bear", f"st_{tf}_vp_score_short", near_distance_score(f, f"st_{tf}_tlv2_resistance_distance_atr_rank0"), "conf_volume_bearish_confirmation"]),
            lambda f, tf=tf: flag(f, f"st_{tf}_ms_choch_to_bear") & (num(f, f"st_{tf}_vp_score_short").ge(0.20) | near_distance_score(f, f"st_{tf}_tlv2_resistance_distance_atr_rank0").ge(0.30)),
        ),
    ]


def pattern_rules(tf: str, pattern: str, thresholds: tuple[float, ...]) -> list[RuleSpec]:
    prefix = f"st_{tf}_pg2_{pattern}"
    return [
        RuleSpec(
            f"mtf_{tf}_{pattern}_upper_break_long",
            f"mtf_{pattern}_upper_break_long",
            "long",
            f"When a {tf} {pattern} breaks upward with volume, does it follow through?",
            f"{tf} {pattern} is present, price breaks the upper boundary, and bullish volume/VP agrees.",
            12,
            0.020,
            0.050,
            thresholds,
            lambda f, prefix=prefix: mean_scores(f, [f"{prefix}_indicator_score", line_break_score(f, f"{prefix}_upper", "long"), "conf_volume_bullish_confirmation", vp_guard_score(f, "long"), "conf_structure_breakout_trigger_score"]),
            lambda f, prefix=prefix: flag(f, f"{prefix}_pattern_present") & line_break_score(f, f"{prefix}_upper", "long").gt(0.0),
        ),
        RuleSpec(
            f"mtf_{tf}_{pattern}_lower_break_short",
            f"mtf_{pattern}_lower_break_short",
            "short",
            f"When a {tf} {pattern} breaks downward with volume, does it follow through?",
            f"{tf} {pattern} is present, price breaks the lower boundary, and bearish volume/VP agrees.",
            12,
            0.020,
            0.050,
            thresholds,
            lambda f, prefix=prefix: mean_scores(f, [f"{prefix}_indicator_score", line_break_score(f, f"{prefix}_lower", "short"), "conf_volume_bearish_confirmation", vp_guard_score(f, "short"), "conf_structure_breakdown_trigger_score"]),
            lambda f, prefix=prefix: flag(f, f"{prefix}_pattern_present") & line_break_score(f, f"{prefix}_lower", "short").gt(0.0),
        ),
        RuleSpec(
            f"mtf_{tf}_{pattern}_lower_reclaim_long",
            f"mtf_{pattern}_lower_reclaim_long",
            "long",
            f"When price reclaims the lower side of a {tf} {pattern}, does it bounce?",
            f"Price retests/reclaims the lower {tf} {pattern} boundary with bullish pressure.",
            10,
            0.018,
            0.040,
            thresholds,
            lambda f, prefix=prefix: mean_scores(f, [f"{prefix}_indicator_score", line_retest_score(f, f"{prefix}_lower", "long", 0.006), "conf_volume_bullish_confirmation", vp_guard_score(f, "long")]),
            lambda f, prefix=prefix: flag(f, f"{prefix}_pattern_present") & line_retest_score(f, f"{prefix}_lower", "long", 0.006).gt(0.0),
        ),
        RuleSpec(
            f"mtf_{tf}_{pattern}_upper_reject_short",
            f"mtf_{pattern}_upper_reject_short",
            "short",
            f"When price rejects the upper side of a {tf} {pattern}, does it fade?",
            f"Price retests/rejects the upper {tf} {pattern} boundary with bearish pressure.",
            10,
            0.018,
            0.040,
            thresholds,
            lambda f, prefix=prefix: mean_scores(f, [f"{prefix}_indicator_score", line_retest_score(f, f"{prefix}_upper", "short", 0.006), "conf_volume_bearish_confirmation", vp_guard_score(f, "short")]),
            lambda f, prefix=prefix: flag(f, f"{prefix}_pattern_present") & line_retest_score(f, f"{prefix}_upper", "short", 0.006).gt(0.0),
        ),
    ]


def run_suite(frame: DataFrame, rules: tuple[RuleSpec, ...], fee: float, slippage: float) -> tuple[DataFrame, DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    trade_frames: list[DataFrame] = []
    for rule in rules:
        score = clamp01(rule.score_builder(frame))
        gate = rule.gate_builder(frame).fillna(False).astype(bool)
        for threshold in rule.thresholds:
            signal = gate & score.ge(threshold)
            trades = simulate_rule(frame, rule, signal, score, threshold, fee, slippage)
            trade_frames.append(trades)
            summary_rows.append(
                summarize_trades(
                    trades,
                    f"{rule.rule_id}__thr_{threshold:.2f}",
                    rule.concept_id,
                    rule.direction,
                    rule.trader_question,
                    rule.visible_market_state,
                    frame,
                    threshold=threshold,
                )
            )
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["usable_sample_size"] = summary["trades"].ge(5)
        summary["positive_quality"] = (
            summary["usable_sample_size"]
            & summary["total_return"].gt(0.0)
            & summary["profit_factor"].gt(1.05)
            & summary["max_drawdown"].gt(-0.15)
        )
        summary["candidate_score"] = (
            summary["total_return"].clip(lower=-0.25, upper=0.50) * 100.0
            + summary["profit_factor"].replace(float("inf"), 5.0).clip(0.0, 5.0) * 12.0
            + summary["win_rate"].fillna(0.0) * 30.0
            + summary["trades"].clip(0, 40) * 0.35
            + summary["max_drawdown"].clip(lower=-0.25, upper=0.0) * 50.0
        )
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    return summary, trades


def simulate_rule(frame: DataFrame, rule: RuleSpec, signal: Series, score: Series, threshold: float, fee: float, slippage: float) -> DataFrame:
    rows: list[dict[str, Any]] = []
    flat_after = -1
    open_ = num(frame, "open")
    high = num(frame, "high")
    low = num(frame, "low")
    close = num(frame, "close")
    dates = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    signal_idx = list(signal[signal.fillna(False)].index)
    for idx in signal_idx:
        if idx <= flat_after:
            continue
        entry_idx = idx + 1
        exit_idx = min(entry_idx + rule.hold_hours, len(frame) - 1)
        if entry_idx >= len(frame) or exit_idx <= entry_idx:
            continue
        entry_price = float(open_.iloc[entry_idx])
        fallback_close = float(close.iloc[exit_idx])
        if not pd.notna(entry_price) or entry_price <= 0.0 or not pd.notna(fallback_close):
            continue
        exit_price, exit_reason = resolve_exit(
            rule.direction,
            entry_price,
            high.iloc[entry_idx : exit_idx + 1],
            low.iloc[entry_idx : exit_idx + 1],
            fallback_close,
            rule.stop_loss,
            rule.take_profit,
        )
        gross = exit_price / entry_price - 1.0 if rule.direction == "long" else entry_price / exit_price - 1.0
        net = gross - fee - slippage
        rows.append(
            {
                "rule_id": rule.rule_id,
                "concept_id": rule.concept_id,
                "direction": rule.direction,
                "threshold": threshold,
                "signal_date": dates.iloc[idx],
                "entry_date": dates.iloc[entry_idx],
                "exit_date": dates.iloc[exit_idx],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross,
                "net_return": net,
                "exit_reason": exit_reason,
                "signal_score": float(score.iloc[idx]) if pd.notna(score.iloc[idx]) else None,
            }
        )
        flat_after = exit_idx
    return pd.DataFrame(rows)


def select_candidates(summary: DataFrame, *, min_trades: int, max_selected: int) -> DataFrame:
    if summary.empty:
        return summary
    candidates = summary[summary["trades"].ge(min_trades)].copy()
    candidates = candidates[candidates["total_return"].gt(0.0) & candidates["profit_factor"].gt(1.05)]
    if candidates.empty:
        return candidates
    candidates = candidates.sort_values(["candidate_score", "total_return", "profit_factor"], ascending=[False, False, False])
    return candidates.groupby("concept_id", as_index=False, group_keys=False).head(2).head(max_selected).reset_index(drop=True)


def simulate_selected(frame: DataFrame, rules: tuple[RuleSpec, ...], selected: DataFrame, fee: float, slippage: float) -> DataFrame:
    if selected.empty:
        return pd.DataFrame()
    rule_map = {rule.rule_id: rule for rule in rules}
    all_signals: list[pd.DataFrame] = []
    for _, row in selected.iterrows():
        rule_id = str(row["variant_id"]).split("__thr_")[0]
        rule = rule_map.get(rule_id)
        if rule is None:
            continue
        threshold = float(row["threshold"])
        score = clamp01(rule.score_builder(frame))
        gate = rule.gate_builder(frame).fillna(False).astype(bool)
        signal = gate & score.ge(threshold)
        idx = signal[signal.fillna(False)].index
        if len(idx) == 0:
            continue
        all_signals.append(
            pd.DataFrame(
                {
                    "idx": idx,
                    "priority": float(row.get("candidate_score", 0.0)),
                    "rule_id": rule.rule_id,
                    "threshold": threshold,
                    "signal_score": score.loc[idx].to_numpy(),
                }
            )
        )
    if not all_signals:
        return pd.DataFrame()
    signals = pd.concat(all_signals, ignore_index=True).sort_values(["idx", "priority"], ascending=[True, False])
    signals = signals.drop_duplicates("idx", keep="first")
    rows: list[DataFrame] = []
    flat_after = -1
    for _, signal_row in signals.iterrows():
        idx = int(signal_row["idx"])
        if idx <= flat_after:
            continue
        rule = rule_map[str(signal_row["rule_id"])]
        signal = pd.Series(False, index=frame.index)
        signal.iloc[idx] = True
        score = pd.Series(0.0, index=frame.index)
        score.iloc[idx] = float(signal_row["signal_score"])
        trades = simulate_rule(frame, rule, signal, score, float(signal_row["threshold"]), fee, slippage)
        if not trades.empty:
            rows.append(trades)
            exit_date = pd.to_datetime(trades.iloc[-1]["exit_date"], utc=True)
            matches = frame.index[pd.to_datetime(frame["date"], utc=True).eq(exit_date)]
            flat_after = int(matches.max()) if len(matches) else idx + rule.hold_hours
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def markdown_report(summary: DataFrame, selected: DataFrame, combined_summary: dict[str, Any]) -> str:
    lines = [
        "# MTF Indicator Lead Mining",
        "",
        "This is a broad but structured lead-mining pass over custom VP, TLV2, market-structure, and pattern-geometry columns.",
        "",
        "## Combined Selected Block",
        "",
        f"1. Trades: {combined_summary.get('trades', 0)}",
        f"2. Win rate: {fmt_pct(combined_summary.get('win_rate'))}",
        f"3. Total return: {fmt_pct(combined_summary.get('total_return'))}",
        f"4. Max drawdown: {fmt_pct(combined_summary.get('max_drawdown'))}",
        f"5. Profit factor: {fmt(combined_summary.get('profit_factor'))}",
        "",
        "## Selected Candidates",
        "",
    ]
    if selected.empty:
        lines.append("No candidates passed the loose standalone filter.")
    else:
        for _, row in selected.head(60).iterrows():
            lines.append(f"1. `{row['variant_id']}`")
            lines.append(f"   - Trader question: {row['trader_question']}")
            lines.append(f"   - What a trader saw: {row['visible_market_state']}")
            lines.append(f"   - Result: {int(row['trades'])} trades, {fmt_pct(row['win_rate'])} wins, {fmt_pct(row['total_return'])} return, {fmt_pct(row['max_drawdown'])} drawdown, PF {fmt(row['profit_factor'])}.")
    lines.extend(["", "## Top Raw Variants", ""])
    top = summary.sort_values(["candidate_score", "total_return"], ascending=[False, False]).head(30)
    for _, row in top.iterrows():
        lines.append(f"1. `{row['variant_id']}` - {int(row['trades'])} trades, {fmt_pct(row['total_return'])} return, PF {fmt(row['profit_factor'])}.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

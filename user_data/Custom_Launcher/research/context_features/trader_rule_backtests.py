from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT = USER_DATA_DIR / "research_news_data" / "context_features" / "confluence_cache" / "trader_confluence_1h_latest.parquet"
DEFAULT_OUTPUT_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    concept_id: str
    direction: str
    trader_question: str
    visible_market_state: str
    hold_hours: int
    stop_loss: float
    take_profit: float
    thresholds: tuple[float, ...]
    score_builder: Callable[[DataFrame], Series]
    gate_builder: Callable[[DataFrame], Series]


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest trader-readable concept leads as standalone BTC 1h trading rules.")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="trader_rule_backtests")
    parser.add_argument("--round-trip-fee", type=float, default=0.0010)
    parser.add_argument("--slippage", type=float, default=0.0002)
    parser.add_argument("--min-trades-per-year", type=float, default=8.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    plan = {
        "mode": "setup-only unless --execute is supplied",
        "snapshot": str(args.snapshot),
        "entry_timing": "Signal at hour close, enter next candle open.",
        "costs": {"round_trip_fee": args.round_trip_fee, "slippage": args.slippage},
        "purpose": "Test each promising concept as a standalone long/short entry rule, then build a combined multi-scenario block from usable rules.",
        "rules": [rule.rule_id for rule in build_rules()],
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.snapshot.exists():
        raise FileNotFoundError(args.snapshot)

    frame = pd.read_parquet(args.snapshot)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    require_columns(frame, ("open", "high", "low", "close"))
    rules = build_rules()
    summary, trades, selected = run_rule_suite(
        frame,
        rules,
        round_trip_fee=float(args.round_trip_fee),
        slippage=float(args.slippage),
        min_trades_per_year=float(args.min_trades_per_year),
    )
    combined_trades = simulate_combined(
        frame,
        selected,
        round_trip_fee=float(args.round_trip_fee),
        slippage=float(args.slippage),
    )
    combined_summary = summarize_trades(
        combined_trades,
        "combined_multi_scenario_rule_block",
        "combined",
        "both",
        "Use the strongest non-overlapping standalone rule variants as a multi-scenario BTC rule block.",
        "Different long/short structure, orderbook, momentum, compression, and reclaim states can all trigger entries.",
        frame,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = safe_name(args.tag)
    summary_path = args.output_dir / f"trader_rule_backtests_{safe_tag}_summary.csv"
    trades_path = args.output_dir / f"trader_rule_backtests_{safe_tag}_trades.csv"
    selected_path = args.output_dir / f"trader_rule_backtests_{safe_tag}_selected_rules.csv"
    combined_trades_path = args.output_dir / f"trader_rule_backtests_{safe_tag}_combined_trades.csv"
    signals_path = args.output_dir / f"trader_rule_signals_{safe_tag}.parquet"
    markdown_path = args.output_dir / f"trader_rule_backtests_{safe_tag}.md"
    meta_path = args.output_dir / f"trader_rule_backtests_{safe_tag}_meta.json"
    summary.to_csv(summary_path, index=False)
    trades.to_csv(trades_path, index=False)
    selected.to_csv(selected_path, index=False)
    combined_trades.to_csv(combined_trades_path, index=False)
    export_combined_signals(combined_trades, signals_path)
    markdown_path.write_text(markdown_report(summary, selected, combined_summary, frame), encoding="utf-8")
    meta = {
        **plan,
        "snapshot_rows": int(len(frame)),
        "snapshot_start": str(frame["date"].min()),
        "snapshot_end": str(frame["date"].max()),
        "rule_variants": int(len(summary)),
        "trade_rows": int(len(trades)),
        "selected_rules": int(len(selected)),
        "combined_summary": combined_summary,
        "outputs": {
            "summary": str(summary_path),
            "trades": str(trades_path),
            "selected_rules": str(selected_path),
            "combined_trades": str(combined_trades_path),
            "signals": str(signals_path),
            "markdown": str(markdown_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "trades": str(trades_path), "selected_rules": str(selected_path), "combined_trades": str(combined_trades_path), "signals": str(signals_path), "markdown": str(markdown_path), "meta": str(meta_path), "rule_variants": int(len(summary)), "selected_rules": int(len(selected)), "combined_return": combined_summary.get("total_return")}, indent=2))
    return 0


def build_rules() -> tuple[RuleSpec, ...]:
    common_thresholds = (0.45, 0.55, 0.65, 0.75)
    loose_thresholds = (0.35, 0.45, 0.55, 0.65)
    return (
        RuleSpec(
            "range_high_breakout_long",
            "range_high_breakout_continuation",
            "long",
            "When price breaks the recent range high with bullish pressure, does it keep going up?",
            "Price is near/breaking the 24h/72h range high and volume or structure confirms the break.",
            6,
            0.020,
            0.040,
            common_thresholds,
            lambda f: mean_scores(f, ["conf_range_position_high_stack", "conf_structure_breakout_trigger_score", "conf_volume_bullish_confirmation"]),
            lambda f: flag(f, "px_close_breakout_24h") | num(f, "conf_structure_breakout_trigger_score").ge(0.34),
        ),
        RuleSpec(
            "range_low_breakdown_short",
            "range_low_breakdown_continuation",
            "short",
            "When price breaks the recent range low with bearish pressure, does it keep going down?",
            "Price is near/breaking the 24h/72h range low and bearish volume or structure confirms the break.",
            6,
            0.020,
            0.040,
            common_thresholds,
            lambda f: mean_scores(f, ["conf_range_position_low_stack", "conf_structure_breakdown_trigger_score", "conf_volume_bearish_confirmation"]),
            lambda f: flag(f, "px_close_breakdown_24h") | num(f, "conf_structure_breakdown_trigger_score").ge(0.34),
        ),
        RuleSpec(
            "compression_breakout_long",
            "compression_breakout_release",
            "long",
            "After compression, does an upside break release into continuation?",
            "Price was compressed, then range/high break and bullish volume appear.",
            8,
            0.018,
            0.045,
            common_thresholds,
            lambda f: mean_scores(f, ["conf_price_compression_24h", "conf_range_position_high_stack", "conf_volume_bullish_impulse_short", "conf_structure_breakout_trigger_score"]),
            lambda f: num(f, "conf_price_compression_24h").ge(0.45) & (flag(f, "px_close_breakout_24h") | num(f, "conf_structure_breakout_trigger_score").ge(0.34)),
        ),
        RuleSpec(
            "compression_breakdown_short",
            "compression_breakdown_release",
            "short",
            "After compression, does a downside break release lower?",
            "Price was compressed, then range/support break and bearish volume appear.",
            8,
            0.018,
            0.045,
            common_thresholds,
            lambda f: mean_scores(f, ["conf_price_compression_24h", "conf_range_position_low_stack", "conf_volume_bearish_impulse_short", "conf_structure_breakdown_trigger_score"]),
            lambda f: num(f, "conf_price_compression_24h").ge(0.45) & (flag(f, "px_close_breakdown_24h") | num(f, "conf_structure_breakdown_trigger_score").ge(0.34)),
        ),
        RuleSpec(
            "structure_resistance_rejection_short",
            "structure_resistance_rejection",
            "short",
            "When price is near resistance and fails acceptance, does rejection pay?",
            "Price is near a range/VP/TLV2 resistance stack and rejection/fakeout pressure is visible.",
            6,
            0.018,
            0.035,
            common_thresholds,
            lambda f: mean_scores(f, ["conf_structure_resistance_stack_score", "conf_range_position_high_stack", "conf_ob_breakout_failure", "conf_ob_ask_absorption", "conf_volume_bearish_confirmation"]),
            lambda f: num(f, "conf_structure_resistance_stack_score").ge(0.30) & num(f, "conf_range_position_high_stack").ge(0.50),
        ),
        RuleSpec(
            "structure_breakout_acceptance_long",
            "structure_breakout_acceptance",
            "long",
            "When resistance breaks with structure/VP acceptance, does the move continue?",
            "Breakout trigger appears with bullish state and value/range acceptance.",
            6,
            0.018,
            0.040,
            common_thresholds,
            lambda f: mean_scores(f, ["conf_structure_breakout_trigger_score", "conf_structure_bullish_state_score", "conf_range_position_high_stack", "conf_lvn_up_thinness", "conf_volume_bullish_confirmation"]),
            lambda f: num(f, "conf_structure_breakout_trigger_score").ge(0.34),
        ),
        RuleSpec(
            "structure_breakdown_crash_short",
            "structure_breakdown_crash_detection",
            "short",
            "When support breaks with structure/VP pressure, does downside continue?",
            "Breakdown trigger appears with bearish state and support/range weakness.",
            6,
            0.020,
            0.045,
            common_thresholds,
            lambda f: mean_scores(f, ["conf_structure_breakdown_trigger_score", "conf_structure_bearish_state_score", "conf_range_position_low_stack", "conf_lvn_down_thinness", "conf_volume_bearish_confirmation"]),
            lambda f: num(f, "conf_structure_breakdown_trigger_score").ge(0.34),
        ),
        RuleSpec(
            "orderbook_breakdown_short",
            "orderbook_breakdown_risk",
            "short",
            "When orderbook support weakens during downside pressure, does price keep breaking down?",
            "Support is weak/removed, downside vacuum or bearish pressure is visible, and price is not strong.",
            6,
            0.018,
            0.040,
            common_thresholds,
            lambda f: mean_scores(f, ["conf_ob_support_removed_strength", "conf_ob_downside_vacuum_after_support_removed", "conf_ob_bearish_pressure_agreement", "conf_ob_support_persistence_weak", "conf_volume_bearish_confirmation"]),
            lambda f: num(f, "conf_ob_venue_count_present").ge(1) & num(f, "conf_ob_bearish_pressure_agreement").ge(0.15),
        ),
        RuleSpec(
            "orderbook_resistance_evaporation_long",
            "orderbook_wall_distance_shift",
            "long",
            "When ask resistance retreats and the book leans bullish, does upside travel improve?",
            "Orderbook resistance weakens, walls are stretched/removed, and bullish pressure appears.",
            6,
            0.018,
            0.040,
            common_thresholds,
            lambda f: mean_scores(f, ["conf_ob_resistance_removed_strength", "conf_ob_upside_vacuum_after_resistance_removed", "conf_ob_bullish_pressure_agreement", "conf_ob_resistance_persistence_weak", "conf_volume_bullish_confirmation"]),
            lambda f: num(f, "conf_ob_venue_count_present").ge(1) & num(f, "conf_ob_bullish_pressure_agreement").ge(0.15),
        ),
        RuleSpec(
            "orderbook_failed_breakout_short",
            "orderbook_failed_breakout_rejection",
            "short",
            "When price is high but orderbook rejects acceptance, does short-side rejection pay?",
            "Price is near a high/resistance area and orderbook shows breakout failure, ask absorption, or pressure divergence.",
            6,
            0.018,
            0.035,
            common_thresholds,
            lambda f: mean_scores(f, ["conf_range_position_high_stack", "conf_ob_breakout_failure", "conf_ob_ask_absorption", "conf_ob_pressure_divergence", "conf_volume_bearish_confirmation"]),
            lambda f: num(f, "conf_ob_venue_count_present").ge(1) & num(f, "conf_range_position_high_stack").ge(0.50),
        ),
        RuleSpec(
            "support_reclaim_after_break_long",
            "support_reclaim_after_downside_break",
            "long",
            "After a downside break starts, can support reclaim mark a long bounce/resumption?",
            "An early downside break is visible, then support rebuild, breakdown failure, or bearish pressure fade appears.",
            6,
            0.018,
            0.035,
            loose_thresholds,
            lambda f: mean_scores(f, ["conf_epsilon_support_reclaim_after_break_score", "conf_epsilon_downside_first_break_exhaustion_score", "conf_ob_breakdown_failure", "conf_ob_support_bounce", "conf_volume_bullish_confirmation"]),
            lambda f: num(f, "conf_epsilon_support_reclaim_after_break_setup_component_score").ge(0.25),
        ),
        RuleSpec(
            "orderbook_panic_after_break_short",
            "orderbook_breakdown_risk",
            "short",
            "After a downside break starts, does panic-book behaviour mark continuation?",
            "An early downside break is visible and orderbook support is removed with bearish pressure/vacuum.",
            4,
            0.016,
            0.035,
            loose_thresholds,
            lambda f: mean_scores(f, ["conf_epsilon_orderbook_panic_after_break_score", "conf_epsilon_downside_first_break_continuation_score", "conf_ob_downside_vacuum", "conf_ob_bearish_pressure_agreement"]),
            lambda f: num(f, "conf_epsilon_orderbook_panic_after_break_setup_component_score").ge(0.25),
        ),
        RuleSpec(
            "bullish_volume_pressure_long",
            "bullish_volume_pressure_breakout",
            "long",
            "When bullish volume pressure is unusually high, does upside follow-through improve?",
            "Short and 24h bullish volume pressure are both elevated.",
            4,
            0.014,
            0.030,
            common_thresholds,
            lambda f: mean_scores(f, ["conf_volume_bullish_confirmation", "conf_volume_bullish_impulse_short", "px_volume_pressure_24h", "px_volume_pressure_6h"]),
            lambda f: num(f, "px_volume_pressure_6h").gt(0.0),
        ),
        RuleSpec(
            "bearish_volume_pressure_short",
            "bearish_volume_pressure_breakdown",
            "short",
            "When bearish volume pressure is unusually high, does downside follow-through improve?",
            "Short and 24h bearish volume pressure are both elevated.",
            4,
            0.014,
            0.030,
            common_thresholds,
            lambda f: mean_scores(f, ["conf_volume_bearish_confirmation", "conf_volume_bearish_impulse_short", -num(f, "px_volume_pressure_24h"), -num(f, "px_volume_pressure_6h")]),
            lambda f: num(f, "px_volume_pressure_6h").lt(0.0),
        ),
        RuleSpec(
            "price_momentum_breakout_long",
            "price_momentum_breakout_continuation",
            "long",
            "When recent momentum is strong and price is high in range, does it continue?",
            "Recent 3h/6h/24h returns and range position are strong.",
            4,
            0.016,
            0.035,
            common_thresholds,
            lambda f: mean_scores(f, [rank01(num(f, "px_return_3h")), rank01(num(f, "px_return_6h")), rank01(num(f, "px_return_24h")), "conf_range_position_high_stack"]),
            lambda f: num(f, "px_return_3h").gt(0.0) & num(f, "px_return_6h").gt(0.0),
        ),
        RuleSpec(
            "price_momentum_breakdown_short",
            "price_momentum_breakdown_continuation",
            "short",
            "When recent momentum is weak and price is low in range, does downside continue?",
            "Recent 3h/6h/24h returns are negative and price is near low range.",
            4,
            0.016,
            0.035,
            common_thresholds,
            lambda f: mean_scores(f, [rank01(-num(f, "px_return_3h")), rank01(-num(f, "px_return_6h")), rank01(-num(f, "px_return_24h")), "conf_range_position_low_stack"]),
            lambda f: num(f, "px_return_3h").lt(0.0) & num(f, "px_return_6h").lt(0.0),
        ),
        RuleSpec(
            "bull_trend_regime_long",
            "bull_trend_regime_continuation_filter",
            "long",
            "When the broader market regime is bullish, do bullish breaks work better?",
            "Bull-trend regime is high and bullish volume/structure confirms.",
            8,
            0.020,
            0.050,
            common_thresholds,
            lambda f: mean_scores(f, ["conf_price_bull_trend_regime", "conf_structure_bullish_state_score", "conf_volume_bullish_confirmation", "conf_range_position_high_stack"]),
            lambda f: num(f, "conf_price_bull_trend_regime").ge(0.35),
        ),
        RuleSpec(
            "volatility_expansion_drawdown_short",
            "volatility_expansion_drawdown_risk",
            "short",
            "When volatility expands with bearish pressure, does drawdown risk rise?",
            "Compression is gone, hourly range expands, and bearish pressure is visible.",
            6,
            0.018,
            0.040,
            common_thresholds,
            lambda f: mean_scores(f, ["conf_price_range_expansion_24h", "px_range_1h", "conf_volume_bearish_confirmation", "conf_structure_bearish_state_score"]),
            lambda f: num(f, "conf_price_range_expansion_24h").ge(0.50),
        ),
        RuleSpec(
            "sieve_tlv2_resistance_breakout_long",
            "borrowed_sieve_tlv2_resistance_breakout",
            "long",
            "When price breaks a TLV2 resistance line with bullish structure/volume, does the break keep going?",
            "Price is above a scored TLV2 resistance line and bullish structure, VP, or volume confirms the break.",
            8,
            0.018,
            0.045,
            loose_thresholds,
            lambda f: mean_scores(f, ["conf_gamma_bull_tlv2_resistance_break_score", "conf_beta_structure_resistance_breakout_continuation_score", rank01(num(f, "st_1h_tlv2_resistance_score_rank0")), line_break_score(f, "st_1h_tlv2_resistance_line_rank0", "long"), "conf_volume_bullish_confirmation"]),
            lambda f: line_break_score(f, "st_1h_tlv2_resistance_line_rank0", "long").gt(0.0) | num(f, "conf_gamma_bull_tlv2_resistance_break_trigger_component_score").ge(0.25),
        ),
        RuleSpec(
            "sieve_tlv2_support_breakdown_short",
            "borrowed_sieve_tlv2_support_breakdown",
            "short",
            "When price breaks a TLV2 support line with bearish structure/volume, does downside continue?",
            "Price is below a scored TLV2 support line and bearish structure, VP, or volume confirms the break.",
            8,
            0.018,
            0.045,
            loose_thresholds,
            lambda f: mean_scores(f, ["conf_beta_structure_support_breakdown_continuation_score", rank01(num(f, "st_1h_tlv2_support_score_rank0")), line_break_score(f, "st_1h_tlv2_support_line_rank0", "short"), "conf_volume_bearish_confirmation", "conf_structure_breakdown_trigger_score"]),
            lambda f: line_break_score(f, "st_1h_tlv2_support_line_rank0", "short").gt(0.0) | num(f, "conf_beta_structure_support_breakdown_continuation_trigger_component_score").ge(0.25),
        ),
        RuleSpec(
            "sieve_tlv2_vp_support_reclaim_long",
            "borrowed_sieve_tlv2_vp_support_reclaim",
            "long",
            "When price is near support and VP/structure says buyers reclaimed it, does a long bounce work?",
            "Price is near TLV2/support or VP node support and bullish reclaim, BOS/CHoCH, or volume appears.",
            8,
            0.018,
            0.040,
            loose_thresholds,
            lambda f: mean_scores(f, ["conf_gamma_bull_reclaim_after_support_defense_score", "conf_multi_tf_support_bounce_score", "st_1h_vp_node_entry_long", "st_1h_vp_score_long", near_distance_score(f, "st_1h_tlv2_support_distance_atr_rank0"), "conf_volume_bullish_confirmation", "st_1h_ms_bos_to_bull", "st_4h_ms_bos_to_bull"]),
            lambda f: flag(f, "st_1h_vp_node_entry_long") | (near_distance_score(f, "st_1h_tlv2_support_distance_atr_rank0").ge(0.45) & num(f, "conf_volume_bullish_confirmation").ge(0.20)),
        ),
        RuleSpec(
            "sieve_tlv2_vp_resistance_reject_short",
            "borrowed_sieve_tlv2_vp_resistance_reject",
            "short",
            "When price is near resistance and VP/structure says sellers defended it, does a short rejection work?",
            "Price is near TLV2/resistance or VP node resistance and bearish rejection, CHoCH, or volume appears.",
            8,
            0.018,
            0.040,
            loose_thresholds,
            lambda f: mean_scores(f, ["conf_multi_tf_resistance_rejection_score", "conf_beta_structure_resistance_rejection_failure_score", "st_1h_vp_node_entry_short", "st_1h_vp_score_short", near_distance_score(f, "st_1h_tlv2_resistance_distance_atr_rank0"), "conf_volume_bearish_confirmation", "st_1h_ms_choch_to_bear", "st_4h_ms_choch_to_bear"]),
            lambda f: flag(f, "st_1h_vp_node_entry_short") | (near_distance_score(f, "st_1h_tlv2_resistance_distance_atr_rank0").ge(0.45) & num(f, "conf_volume_bearish_confirmation").ge(0.20)),
        ),
        RuleSpec(
            "sieve_vp_poc_reclaim_long",
            "borrowed_sieve_vp_poc_reclaim",
            "long",
            "When price reclaims the VP point of control with bullish pressure, does that become a usable long?",
            "Price is above current/prior POC and VP long score, bullish volume, or bullish structure supports it.",
            6,
            0.016,
            0.035,
            loose_thresholds,
            lambda f: mean_scores(f, [line_break_score(f, "st_1h_vp_poc", "long"), line_break_score(f, "st_1h_vp_prior_poc", "long"), "st_1h_vp_score_long", "st_4h_vp_score_long", "conf_volume_bullish_confirmation", "conf_structure_bullish_state_score"]),
            lambda f: (line_break_score(f, "st_1h_vp_poc", "long").gt(0.0) | line_break_score(f, "st_1h_vp_prior_poc", "long").gt(0.0)) & num(f, "px_return_3h").gt(0.0),
        ),
        RuleSpec(
            "sieve_vp_poc_reject_short",
            "borrowed_sieve_vp_poc_reject",
            "short",
            "When price loses/rejects the VP point of control with bearish pressure, does downside continue?",
            "Price is below current/prior POC and VP short score, bearish volume, or bearish structure supports it.",
            6,
            0.016,
            0.035,
            loose_thresholds,
            lambda f: mean_scores(f, [line_break_score(f, "st_1h_vp_poc", "short"), line_break_score(f, "st_1h_vp_prior_poc", "short"), "st_1h_vp_score_short", "st_4h_vp_score_short", "conf_volume_bearish_confirmation", "conf_structure_bearish_state_score"]),
            lambda f: (line_break_score(f, "st_1h_vp_poc", "short").gt(0.0) | line_break_score(f, "st_1h_vp_prior_poc", "short").gt(0.0)) & num(f, "px_return_3h").lt(0.0),
        ),
        RuleSpec(
            "sieve_vp_lvn_fast_travel_long",
            "borrowed_sieve_vp_lvn_fast_travel_up",
            "long",
            "When there is thin VP liquidity above and price breaks upward, does it travel quickly?",
            "LVN/thin value area sits above price and upside breakout plus bullish volume appears.",
            6,
            0.016,
            0.040,
            loose_thresholds,
            lambda f: mean_scores(f, ["conf_lvn_fast_travel_up_score", "conf_beta_structure_lvn_up_continuation_score", "st_1h_vp_lvn_above_thinness", "conf_lvn_up_thinness", "conf_structure_breakout_trigger_score", "conf_volume_bullish_confirmation"]),
            lambda f: num(f, "conf_lvn_up_thinness").ge(0.20) & (flag(f, "px_close_breakout_24h") | num(f, "conf_structure_breakout_trigger_score").ge(0.25)),
        ),
        RuleSpec(
            "sieve_vp_lvn_fast_travel_short",
            "borrowed_sieve_vp_lvn_fast_travel_down",
            "short",
            "When there is thin VP liquidity below and price breaks downward, does it travel quickly?",
            "LVN/thin value area sits below price and downside breakdown plus bearish volume appears.",
            6,
            0.016,
            0.040,
            loose_thresholds,
            lambda f: mean_scores(f, ["conf_lvn_fast_travel_down_score", "conf_beta_structure_lvn_down_continuation_score", "st_1h_vp_lvn_below_thinness", "conf_lvn_down_thinness", "conf_structure_breakdown_trigger_score", "conf_volume_bearish_confirmation"]),
            lambda f: num(f, "conf_lvn_down_thinness").ge(0.20) & (flag(f, "px_close_breakdown_24h") | num(f, "conf_structure_breakdown_trigger_score").ge(0.25)),
        ),
        RuleSpec(
            "sieve_mtf_bos_support_ride_long",
            "borrowed_sieve_mtf_bos_support_ride",
            "long",
            "When 4h/1d structure is bullish and support is being ridden, does the long side have edge?",
            "Higher-timeframe BOS/structure is bullish while support/VP and volume are not fighting the move.",
            12,
            0.022,
            0.055,
            loose_thresholds,
            lambda f: mean_scores(f, ["st_4h_ms_bos_to_bull", "st_1d_ms_bos_to_bull", "st_4h_vp_score_long", "conf_structure_bullish_state_score", "conf_gamma_bull_pullback_higher_low_resume_score", "conf_volume_bullish_confirmation"]),
            lambda f: (flag(f, "st_4h_ms_bos_to_bull") | flag(f, "st_1d_ms_bos_to_bull") | num(f, "conf_structure_bullish_state_score").ge(0.45)) & num(f, "st_4h_vp_score_long").ge(0.30),
        ),
        RuleSpec(
            "sieve_mtf_bos_prior_breakdown_short",
            "borrowed_sieve_mtf_bos_prior_breakdown",
            "short",
            "When 4h/1d structure turns bearish and 1h breaks down, does short continuation work?",
            "Higher-timeframe BOS/CHoCH is bearish and 1h price/volume confirms a lower break.",
            8,
            0.020,
            0.045,
            loose_thresholds,
            lambda f: mean_scores(f, ["st_4h_ms_bos_to_bear", "st_1d_ms_bos_to_bear", "st_4h_vp_score_short", "conf_structure_bearish_state_score", "conf_structure_breakdown_trigger_score", "conf_volume_bearish_confirmation"]),
            lambda f: (flag(f, "st_4h_ms_bos_to_bear") | flag(f, "st_1d_ms_bos_to_bear") | num(f, "conf_structure_bearish_state_score").ge(0.45)) & num(f, "conf_structure_breakdown_trigger_score").ge(0.25),
        ),
        RuleSpec(
            "sieve_pattern_rectangle_breakout_long",
            "borrowed_sieve_rectangle_breakout",
            "long",
            "When a rectangle/compression area breaks upward, does the breakout have follow-through?",
            "4h/1d rectangle or compression is present and price closes above the upper boundary with volume.",
            10,
            0.020,
            0.050,
            loose_thresholds,
            lambda f: mean_scores(f, ["st_4h_pg2_rectangle_indicator_score", "st_1d_pg2_rectangle_indicator_score", line_break_score(f, "st_4h_pg2_rectangle_upper", "long"), line_break_score(f, "st_1d_pg2_rectangle_upper", "long"), "conf_volume_bullish_confirmation", "conf_structure_breakout_trigger_score"]),
            lambda f: (flag(f, "st_4h_pg2_rectangle_pattern_present") | flag(f, "st_1d_pg2_rectangle_pattern_present")) & (line_break_score(f, "st_4h_pg2_rectangle_upper", "long").gt(0.0) | line_break_score(f, "st_1d_pg2_rectangle_upper", "long").gt(0.0)),
        ),
        RuleSpec(
            "sieve_pattern_rectangle_breakdown_short",
            "borrowed_sieve_rectangle_breakdown",
            "short",
            "When a rectangle/compression area breaks downward, does the breakdown have follow-through?",
            "4h/1d rectangle or compression is present and price closes below the lower boundary with volume.",
            10,
            0.020,
            0.050,
            loose_thresholds,
            lambda f: mean_scores(f, ["st_4h_pg2_rectangle_indicator_score", "st_1d_pg2_rectangle_indicator_score", line_break_score(f, "st_4h_pg2_rectangle_lower", "short"), line_break_score(f, "st_1d_pg2_rectangle_lower", "short"), "conf_volume_bearish_confirmation", "conf_structure_breakdown_trigger_score"]),
            lambda f: (flag(f, "st_4h_pg2_rectangle_pattern_present") | flag(f, "st_1d_pg2_rectangle_pattern_present")) & (line_break_score(f, "st_4h_pg2_rectangle_lower", "short").gt(0.0) | line_break_score(f, "st_1d_pg2_rectangle_lower", "short").gt(0.0)),
        ),
        RuleSpec(
            "sieve_exact_tlv2_res_cross_long",
            "borrowed_sieve_exact_tlv2_resistance_cross",
            "long",
            "If we copy the Sieve TLV2 resistance-cross idea more closely, does the long breakout still work?",
            "Price crosses above a scored TLV2 resistance line, volume expands, pressure is bullish, and VP context is not fighting the long.",
            48,
            0.020,
            0.020,
            (0.30, 0.40, 0.50, 0.60),
            lambda f: mean_scores(f, [line_cross_score(f, "st_1h_tlv2_resistance_line_rank0", "long", 0.003), "st_1h_tlv2_resistance_score_rank0", volume_ratio_score(f, 24), pressure_ratio_score(f, 24, "long"), vp_guard_score(f, "long")]),
            lambda f: line_cross_score(f, "st_1h_tlv2_resistance_line_rank0", "long", 0.003).gt(0.0) & num(f, "st_1h_tlv2_resistance_score_rank0").ge(0.50) & volume_ratio_score(f, 24).ge(0.50) & pressure_ratio_score(f, 24, "long").ge(0.50) & vp_guard_score(f, "long").ge(0.35),
        ),
        RuleSpec(
            "sieve_exact_tlv2_sup_cross_short",
            "borrowed_sieve_exact_tlv2_support_cross",
            "short",
            "If we copy the Sieve TLV2 support-cross idea more closely, does the short breakdown still work?",
            "Price crosses below a scored TLV2 support line, volume expands, pressure is bearish, and VP context is not fighting the short.",
            48,
            0.020,
            0.020,
            (0.30, 0.40, 0.50, 0.60),
            lambda f: mean_scores(f, [line_cross_score(f, "st_1h_tlv2_support_line_rank0", "short", 0.003), "st_1h_tlv2_support_score_rank0", volume_ratio_score(f, 24), pressure_ratio_score(f, 24, "short"), vp_guard_score(f, "short")]),
            lambda f: line_cross_score(f, "st_1h_tlv2_support_line_rank0", "short", 0.003).gt(0.0) & num(f, "st_1h_tlv2_support_score_rank0").ge(0.50) & volume_ratio_score(f, 24).ge(0.50) & pressure_ratio_score(f, 24, "short").ge(0.50) & vp_guard_score(f, "short").ge(0.35),
        ),
        RuleSpec(
            "sieve_exact_tlv2_vp_retest_long",
            "borrowed_sieve_exact_tlv2_res_retest_vp_node",
            "long",
            "After a resistance break, does a retest near the TLV2 line with VP node support produce better long entries?",
            "Price remains near a strong TLV2 resistance-turned-support level while VP node/score and market pressure support long continuation.",
            48,
            0.020,
            0.020,
            (0.30, 0.40, 0.50, 0.60),
            lambda f: mean_scores(f, [line_retest_score(f, "st_1h_tlv2_resistance_line_rank0", "long", 0.006), "st_1h_tlv2_resistance_score_rank0", "st_1h_vp_node_hold_long", "st_1h_vp_score_long", pressure_ratio_score(f, 24, "long")]),
            lambda f: line_retest_score(f, "st_1h_tlv2_resistance_line_rank0", "long", 0.006).gt(0.0) & num(f, "st_1h_tlv2_resistance_score_rank0").ge(0.50) & (flag(f, "st_1h_vp_node_hold_long") | num(f, "st_1h_vp_score_long").ge(0.25)) & pressure_ratio_score(f, 24, "long").ge(0.45),
        ),
        RuleSpec(
            "sieve_exact_tlv2_vp_retest_short",
            "borrowed_sieve_exact_tlv2_sup_retest_vp_node",
            "short",
            "After a support break, does a retest near the TLV2 line with VP node resistance produce better short entries?",
            "Price remains near a strong TLV2 support-turned-resistance level while VP node/score and market pressure support short continuation.",
            48,
            0.020,
            0.020,
            (0.30, 0.40, 0.50, 0.60),
            lambda f: mean_scores(f, [line_retest_score(f, "st_1h_tlv2_support_line_rank0", "short", 0.006), "st_1h_tlv2_support_score_rank0", "st_1h_vp_node_hold_short", "st_1h_vp_score_short", pressure_ratio_score(f, 24, "short")]),
            lambda f: line_retest_score(f, "st_1h_tlv2_support_line_rank0", "short", 0.006).gt(0.0) & num(f, "st_1h_tlv2_support_score_rank0").ge(0.50) & (flag(f, "st_1h_vp_node_hold_short") | num(f, "st_1h_vp_score_short").ge(0.25)) & pressure_ratio_score(f, 24, "short").ge(0.45),
        ),
        RuleSpec(
            "sieve_prior_day_high_break_vp_long",
            "borrowed_sieve_prior_day_high_break_vp_node",
            "long",
            "When price breaks the prior-day high with VP/pressure support, does upside continuation work?",
            "Price crosses above the previous 24h high while VP long context, bullish pressure, and volume expansion are present.",
            48,
            0.020,
            0.020,
            (0.30, 0.40, 0.50, 0.60),
            lambda f: mean_scores(f, [rolling_level_cross_score(f, 24, "high", "long"), "st_1h_vp_score_long", "st_4h_vp_score_long", volume_ratio_score(f, 24), pressure_ratio_score(f, 24, "long")]),
            lambda f: rolling_level_cross_score(f, 24, "high", "long").gt(0.0) & vp_guard_score(f, "long").ge(0.35) & pressure_ratio_score(f, 24, "long").ge(0.45),
        ),
        RuleSpec(
            "sieve_prior_day_low_break_vp_short",
            "borrowed_sieve_prior_day_low_break_vp_node",
            "short",
            "When price breaks the prior-day low with VP/pressure support, does downside continuation work?",
            "Price crosses below the previous 24h low while VP short context, bearish pressure, and volume expansion are present.",
            48,
            0.020,
            0.020,
            (0.30, 0.40, 0.50, 0.60),
            lambda f: mean_scores(f, [rolling_level_cross_score(f, 24, "low", "short"), "st_1h_vp_score_short", "st_4h_vp_score_short", volume_ratio_score(f, 24), pressure_ratio_score(f, 24, "short")]),
            lambda f: rolling_level_cross_score(f, 24, "low", "short").gt(0.0) & vp_guard_score(f, "short").ge(0.35) & pressure_ratio_score(f, 24, "short").ge(0.45),
        ),
    )


def run_rule_suite(
    frame: DataFrame,
    rules: tuple[RuleSpec, ...],
    *,
    round_trip_fee: float,
    slippage: float,
    min_trades_per_year: float,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    trade_frames: list[DataFrame] = []
    for rule in rules:
        score = clamp01(rule.score_builder(frame))
        gate = rule.gate_builder(frame).fillna(False).astype(bool)
        for threshold in rule.thresholds:
            signal = gate & score.ge(threshold)
            trades = simulate_rule(frame, rule, signal, score, threshold, round_trip_fee, slippage)
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
    all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    summary["usable_trade_volume"] = summary["trades_per_year"].ge(min_trades_per_year)
    summary["usable_active_trade_volume"] = summary["trades_per_active_year"].ge(min_trades_per_year)
    summary["usable_sample_size"] = summary["trades"].ge(10)
    summary["beats_buy_hold"] = summary["total_return"].gt(summary["buy_hold_return"])
    summary["positive_quality"] = (
        summary["usable_sample_size"]
        & (summary["usable_trade_volume"] | summary["usable_active_trade_volume"])
        & summary["total_return"].gt(0.0)
        & summary["max_drawdown"].gt(-0.35)
        & (summary["win_rate"].ge(0.48) | summary["profit_factor"].ge(1.20))
    )
    selected = select_rules(summary)
    return summary.sort_values(["positive_quality", "total_return", "profit_factor"], ascending=[False, False, False]), all_trades, selected


def select_rules(summary: DataFrame) -> DataFrame:
    candidates = summary[summary["positive_quality"].fillna(False)].copy()
    if candidates.empty:
        candidates = summary[summary["trades"].ge(10) & summary["total_return"].gt(0.0)].copy()
    if candidates.empty:
        return DataFrame()
    candidates["rank_score"] = (
        candidates["total_return"].fillna(0.0)
        + candidates["profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 4.0) / 10.0
        + candidates["win_rate"].fillna(0.0) / 5.0
        + candidates["trades_per_year"].fillna(0.0).clip(0.0, 40.0) / 100.0
        + candidates["max_drawdown"].fillna(-1.0)
    )
    candidates = candidates.sort_values("rank_score", ascending=False)
    return candidates.groupby("concept_id", as_index=False, group_keys=False).head(1).head(10).reset_index(drop=True)


def simulate_rule(
    frame: DataFrame,
    rule: RuleSpec,
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
    last_entry_index = len(frame) - rule.hold_hours - 2
    while i <= last_entry_index:
        if not bool(signal.iloc[i]):
            i += 1
            continue
        entry_idx = i + 1
        exit_idx = i + rule.hold_hours
        entry_price = float(open_.iloc[entry_idx])
        if not np.isfinite(entry_price) or entry_price <= 0.0:
            i += 1
            continue
        exit_price, exit_reason = resolve_exit(rule.direction, entry_price, high.iloc[entry_idx : exit_idx + 1], low.iloc[entry_idx : exit_idx + 1], close.iloc[exit_idx], rule.stop_loss, rule.take_profit)
        if not np.isfinite(exit_price) or exit_price <= 0.0:
            i += 1
            continue
        gross_return = exit_price / entry_price - 1.0 if rule.direction == "long" else entry_price / exit_price - 1.0
        net_return = gross_return - round_trip_fee - slippage
        rows.append(
            {
                "variant_id": f"{rule.rule_id}__thr_{threshold:.2f}",
                "rule_id": rule.rule_id,
                "concept_id": rule.concept_id,
                "direction": rule.direction,
                "threshold": threshold,
                "signal_date": dates.iloc[i],
                "entry_date": dates.iloc[entry_idx],
                "exit_date": dates.iloc[exit_idx],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "net_return": net_return,
                "exit_reason": exit_reason,
                "signal_score": float(score.iloc[i]) if np.isfinite(score.iloc[i]) else np.nan,
            }
        )
        i = exit_idx + 1
    return pd.DataFrame(rows)


def simulate_combined(
    frame: DataFrame,
    selected: DataFrame,
    *,
    round_trip_fee: float,
    slippage: float,
) -> DataFrame:
    if selected.empty:
        return DataFrame()
    rules_by_id = {rule.rule_id: rule for rule in build_rules()}
    candidates: list[dict[str, Any]] = []
    for priority, row in selected.reset_index(drop=True).iterrows():
        rule_id = str(row["variant_id"]).split("__thr_")[0]
        threshold = float(row["threshold"])
        rule = rules_by_id.get(rule_id)
        if rule is None:
            continue
        score = clamp01(rule.score_builder(frame))
        signal = rule.gate_builder(frame).fillna(False).astype(bool) & score.ge(threshold)
        for idx in np.flatnonzero(signal.to_numpy()):
            candidates.append({"idx": int(idx), "priority": int(priority), "rule": rule, "threshold": threshold, "score": float(score.iloc[idx]) if np.isfinite(score.iloc[idx]) else np.nan})
    if not candidates:
        return DataFrame()
    candidates = sorted(candidates, key=lambda x: (x["idx"], x["priority"]))
    rows: list[dict[str, Any]] = []
    open_ = num(frame, "open")
    high = num(frame, "high")
    low = num(frame, "low")
    close = num(frame, "close")
    dates = frame["date"]
    flat_after = -1
    for candidate in candidates:
        i = candidate["idx"]
        rule = candidate["rule"]
        if i <= flat_after or i > len(frame) - rule.hold_hours - 2:
            continue
        entry_idx = i + 1
        exit_idx = i + rule.hold_hours
        entry_price = float(open_.iloc[entry_idx])
        if not np.isfinite(entry_price) or entry_price <= 0.0:
            continue
        exit_price, exit_reason = resolve_exit(rule.direction, entry_price, high.iloc[entry_idx : exit_idx + 1], low.iloc[entry_idx : exit_idx + 1], close.iloc[exit_idx], rule.stop_loss, rule.take_profit)
        gross_return = exit_price / entry_price - 1.0 if rule.direction == "long" else entry_price / exit_price - 1.0
        rows.append(
            {
                "variant_id": "combined_multi_scenario_rule_block",
                "rule_id": rule.rule_id,
                "concept_id": rule.concept_id,
                "direction": rule.direction,
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


def export_combined_signals(combined_trades: DataFrame, output_path: Path) -> None:
    columns = [
        "date",
        "enter_long",
        "enter_short",
        "enter_tag",
        "rule_id",
        "concept_id",
        "threshold",
        "signal_score",
    ]
    if combined_trades.empty:
        pd.DataFrame(columns=columns).to_parquet(output_path, index=False)
        return
    signals = combined_trades[
        [
            "signal_date",
            "direction",
            "rule_id",
            "concept_id",
            "threshold",
            "signal_score",
        ]
    ].copy()
    signals["date"] = pd.to_datetime(signals["signal_date"], utc=True, errors="coerce")
    signals["enter_long"] = signals["direction"].eq("long").astype("int8")
    signals["enter_short"] = signals["direction"].eq("short").astype("int8")
    signals["enter_tag"] = signals["rule_id"].astype(str)
    signals = signals[columns].dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="first")
    signals.to_parquet(output_path, index=False)


def resolve_exit(direction: str, entry_price: float, highs: Series, lows: Series, fallback_close: float, stop_loss: float, take_profit: float) -> tuple[float, str]:
    if direction == "long":
        stop_price = entry_price * (1.0 - stop_loss)
        take_price = entry_price * (1.0 + take_profit)
        for high, low in zip(highs, lows):
            if np.isfinite(low) and low <= stop_price:
                return stop_price, "stop_loss"
            if np.isfinite(high) and high >= take_price:
                return take_price, "take_profit"
    else:
        stop_price = entry_price * (1.0 + stop_loss)
        take_price = entry_price * (1.0 - take_profit)
        for high, low in zip(highs, lows):
            if np.isfinite(high) and high >= stop_price:
                return stop_price, "stop_loss"
            if np.isfinite(low) and low <= take_price:
                return take_price, "take_profit"
    return float(fallback_close), "time_exit"


def summarize_trades(
    trades: DataFrame,
    variant_id: str,
    concept_id: str,
    direction: str,
    trader_question: str,
    visible_market_state: str,
    frame: DataFrame,
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    start = pd.to_datetime(frame["date"].min(), utc=True)
    end = pd.to_datetime(frame["date"].max(), utc=True)
    years = max((end - start).total_seconds() / (365.25 * 24 * 3600), 1e-9)
    buy_hold_return = float(num(frame, "close").iloc[-1] / num(frame, "close").iloc[0] - 1.0)
    if trades.empty:
        return {
            "variant_id": variant_id,
            "concept_id": concept_id,
            "direction": direction,
            "threshold": threshold,
            "trader_question": trader_question,
            "visible_market_state": visible_market_state,
            "trades": 0,
            "trades_per_year": 0.0,
            "active_trade_span_years": 0.0,
            "trades_per_active_year": 0.0,
            "win_rate": np.nan,
            "avg_trade_return": np.nan,
            "median_trade_return": np.nan,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": np.nan,
            "buy_hold_return": buy_hold_return,
        }
    returns = pd.to_numeric(trades["net_return"], errors="coerce").dropna()
    entry_dates = pd.to_datetime(trades["entry_date"], utc=True, errors="coerce").dropna()
    if len(entry_dates) >= 2:
        active_years = max((entry_dates.max() - entry_dates.min()).total_seconds() / (365.25 * 24 * 3600), 1e-9)
    else:
        active_years = years
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    wins = returns[returns.gt(0.0)]
    losses = returns[returns.lt(0.0)]
    profit_factor = float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else np.inf
    return {
        "variant_id": variant_id,
        "concept_id": concept_id,
        "direction": direction,
        "threshold": threshold,
        "trader_question": trader_question,
        "visible_market_state": visible_market_state,
        "trades": int(len(returns)),
        "trades_per_year": float(len(returns) / years),
        "active_trade_span_years": float(active_years),
        "trades_per_active_year": float(len(returns) / active_years),
        "win_rate": float(returns.gt(0.0).mean()),
        "avg_trade_return": float(returns.mean()),
        "median_trade_return": float(returns.median()),
        "total_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "profit_factor": profit_factor,
        "buy_hold_return": buy_hold_return,
    }


def markdown_report(summary: DataFrame, selected: DataFrame, combined_summary: dict[str, Any], frame: DataFrame) -> str:
    lines = [
        "# Trader Rule Backtest Report",
        "",
        "This is a research-only conversion of concept leads into simple BTC 1h trading rules. It is not a production strategy.",
        "",
        "## Test Rules",
        "",
        "1. Signals use only decision-hour columns.",
        "2. Entry is next candle open.",
        "3. Exit uses fixed hold, take-profit, or stop-loss. If stop and take-profit occur in the same candle, stop-loss wins.",
        "4. Costs include configured fees and slippage.",
        "",
        "## Combined Block",
        "",
        f"1. Trades: {combined_summary.get('trades', 0)}",
        f"2. Trades/year: {fmt(combined_summary.get('trades_per_year'))}",
        f"3. Win rate: {fmt_pct(combined_summary.get('win_rate'))}",
        f"4. Total return: {fmt_pct(combined_summary.get('total_return'))}",
        f"5. Buy-and-hold return over same full snapshot: {fmt_pct(combined_summary.get('buy_hold_return'))}",
        f"6. Max drawdown: {fmt_pct(combined_summary.get('max_drawdown'))}",
        f"7. Profit factor: {fmt(combined_summary.get('profit_factor'))}",
        "",
        "## Selected Rule Variants",
        "",
    ]
    if selected.empty:
        lines.append("No rule variants passed the initial positive-quality filter.")
    else:
        for _, row in selected.iterrows():
            lines.append(f"1. `{row['variant_id']}`")
            lines.append(f"   - Trader question: {row['trader_question']}")
            lines.append(f"   - Trades/year: {fmt(row['trades_per_year'])}; win rate: {fmt_pct(row['win_rate'])}; total return: {fmt_pct(row['total_return'])}; max drawdown: {fmt_pct(row['max_drawdown'])}; profit factor: {fmt(row['profit_factor'])}")
    lines.extend(["", "## Top Standalone Variants", ""])
    top = summary.sort_values(["positive_quality", "total_return", "profit_factor"], ascending=[False, False, False]).head(25)
    for _, row in top.iterrows():
        lines.append(f"1. `{row['variant_id']}`")
        lines.append(f"   - What a trader saw: {row['visible_market_state']}")
        lines.append(f"   - Result: {int(row['trades'])} trades, {fmt(row['trades_per_year'])}/year, {fmt_pct(row['win_rate'])} wins, {fmt_pct(row['total_return'])} return, {fmt_pct(row['max_drawdown'])} max drawdown, profit factor {fmt(row['profit_factor'])}.")
    return "\n".join(lines) + "\n"


def mean_scores(frame: DataFrame, columns_or_series: list[str | Series]) -> Series:
    parts: list[Series] = []
    for item in columns_or_series:
        if isinstance(item, str):
            parts.append(num(frame, item))
        else:
            parts.append(pd.to_numeric(item, errors="coerce"))
    if not parts:
        return pd.Series(np.nan, index=frame.index)
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def rank01(series: Series) -> Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.rolling(720, min_periods=168).rank(pct=True)


def clamp01(series: Series | float) -> Series:
    if isinstance(series, Series):
        return pd.to_numeric(series, errors="coerce").clip(0.0, 1.0)
    return pd.Series(series).clip(0.0, 1.0)


def num(frame: DataFrame, column: str) -> Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def flag(frame: DataFrame, column: str) -> Series:
    return num(frame, column).fillna(0.0).gt(0.0)


def line_break_score(frame: DataFrame, line_column: str, direction: str) -> Series:
    close = num(frame, "close")
    line = num(frame, line_column)
    valid = close.gt(0.0) & line.gt(0.0)
    if direction == "long":
        score = close.gt(line)
    else:
        score = close.lt(line)
    return (score & valid).astype(float)


def line_cross_score(frame: DataFrame, line_column: str, direction: str, buffer_pct: float = 0.0) -> Series:
    close = num(frame, "close")
    line = num(frame, line_column)
    valid = close.gt(0.0) & line.gt(0.0)
    if direction == "long":
        level = line.mul(1.0 + buffer_pct)
        crossed = close.gt(level) & close.shift(1).le(level.shift(1))
    else:
        level = line.mul(1.0 - buffer_pct)
        crossed = close.lt(level) & close.shift(1).ge(level.shift(1))
    return (crossed & valid).astype(float)


def line_retest_score(frame: DataFrame, line_column: str, direction: str, buffer_pct: float = 0.006) -> Series:
    close = num(frame, "close")
    high = num(frame, "high")
    low = num(frame, "low")
    line = num(frame, line_column)
    valid = close.gt(0.0) & line.gt(0.0)
    upper = line.mul(1.0 + buffer_pct)
    lower = line.mul(1.0 - buffer_pct)
    if direction == "long":
        retest = low.le(upper) & close.ge(lower) & close.ge(close.shift(1))
    else:
        retest = high.ge(lower) & close.le(upper) & close.le(close.shift(1))
    return (retest & valid).astype(float)


def rolling_level_cross_score(frame: DataFrame, window: int, level_side: str, direction: str) -> Series:
    close = num(frame, "close")
    source = num(frame, "high" if level_side == "high" else "low")
    if direction == "long":
        level = source.shift(1).rolling(window, min_periods=max(4, window // 4)).max()
        crossed = close.gt(level) & close.shift(1).le(level.shift(1))
    else:
        level = source.shift(1).rolling(window, min_periods=max(4, window // 4)).min()
        crossed = close.lt(level) & close.shift(1).ge(level.shift(1))
    return crossed.fillna(False).astype(float)


def near_distance_score(frame: DataFrame, distance_column: str, max_atr: float = 2.0) -> Series:
    distance = num(frame, distance_column)
    return (1.0 - (distance / max_atr)).clip(0.0, 1.0)


def volume_ratio_score(frame: DataFrame, window: int) -> Series:
    volume = num(frame, "volume").clip(lower=0.0)
    baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
    ratio = volume / baseline
    return ((ratio - 0.8) / 0.8).clip(0.0, 1.0)


def pressure_ratio_score(frame: DataFrame, window: int, direction: str) -> Series:
    close = num(frame, "close")
    open_ = num(frame, "open")
    high = num(frame, "high")
    low = num(frame, "low")
    volume = num(frame, "volume").clip(lower=0.0).fillna(0.0)
    candle_range = (high - low).replace(0.0, np.nan)
    body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
    close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
    pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
    directional_volume = (pressure * volume).fillna(0.0)
    baseline = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
    ratio = directional_volume.rolling(window, min_periods=max(2, window // 3)).sum() / baseline
    directional = ratio if direction == "long" else -ratio
    return ((directional - 0.03) / 0.22).clip(0.0, 1.0)


def vp_guard_score(frame: DataFrame, direction: str) -> Series:
    side = "long" if direction == "long" else "short"
    other = "short" if side == "long" else "long"
    score = num(frame, f"st_1h_vp_score_{side}")
    other_score = num(frame, f"st_1h_vp_score_{other}")
    node = flag(frame, f"st_1h_vp_node_entry_{side}") | flag(frame, f"st_1h_vp_node_hold_{side}")
    directional_score = (score - other_score).clip(0.0, 1.0)
    return pd.concat([score.clip(0.0, 1.0), directional_score, node.astype(float)], axis=1).max(axis=1, skipna=True)


def require_columns(frame: DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip()) or "latest"


def fmt(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(numeric):
        return "inf" if numeric > 0 else "n/a"
    return f"{numeric:.3f}"


def fmt_pct(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric * 100.0:.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())

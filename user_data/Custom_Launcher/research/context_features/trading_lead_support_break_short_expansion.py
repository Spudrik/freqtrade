from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from trader_rule_backtests import (
    line_cross_score,
    mean_scores,
    near_distance_score,
    num,
    pressure_ratio_score,
    rolling_level_cross_score,
    safe_name,
    volume_ratio_score,
)
from trading_lead_rectangle_vp_expansion import ExitProfile, lvn_thinness_score, orderbook_accept_score, val_accept_score
from trading_lead_rectangle_vp_refinement import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SNAPSHOT,
    RefinedSpec,
    markdown_report,
    quiet_bull_volume_score,
    range_mid_short_score,
    run_suite,
    simulate_selected,
    vp_short_minus_long_score,
)
from trader_rule_backtests import export_combined_signals, summarize_trades


SHORT_EXITS = (
    ExitProfile("fast_8h_35pct", 8, 0.018, 0.035),
    ExitProfile("fast_10h_5pct", 10, 0.020, 0.050),
    ExitProfile("medium_18h_5pct", 18, 0.025, 0.050),
    ExitProfile("medium_24h_55pct", 24, 0.025, 0.055),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand sparse VAL/VP-opposition short into adjacent support-break short leads.")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="support_break_short_expansion_20260605")
    parser.add_argument("--round-trip-fee", type=float, default=0.0010)
    parser.add_argument("--slippage", type=float, default=0.0002)
    parser.add_argument("--min-trades", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    specs = build_specs()
    if not args.execute:
        print(json.dumps({"mode": "setup-only", "count": len(specs), "specs": [spec.rule_id for spec in specs]}, indent=2))
        return 0
    if not args.snapshot.exists():
        raise FileNotFoundError(args.snapshot)
    frame = pd.read_parquet(args.snapshot)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

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
        "support_break_short_expansion_selected_block",
        "support_break_short_expansion",
        "short",
        "Can the VAL/VP-opposition short lead expand into other support-break shorts?",
        "Selected short variants are non-overlapped by rank.",
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
            "prior_low_break_vp_opposition_low_short",
            "support_break_vp_opposition_short",
            "short",
            "Does a prior-low break work when VP is not defending the long side?",
            "Price breaks the recent low while VP long support/opposition is weak.",
            thresholds,
            lambda f: mean_scores(f, [rolling_level_cross_score(f, 24, "low", "short"), vp_short_minus_long_score(f), quiet_bull_volume_score(f), pressure_ratio_score(f, 24, "short")]),
            lambda f: rolling_level_cross_score(f, 24, "low", "short").gt(0.0) & vp_short_minus_long_score(f).ge(0.45),
            SHORT_EXITS,
        ),
        RefinedSpec(
            "prior_low_break_quiet_bull_volume_short",
            "support_break_quiet_bull_volume_short",
            "short",
            "Does a prior-low break improve when bullish volume is quiet?",
            "Recent low breaks, and bullish volume pressure is not pushing back.",
            thresholds,
            lambda f: mean_scores(f, [rolling_level_cross_score(f, 24, "low", "short"), quiet_bull_volume_score(f), pressure_ratio_score(f, 24, "short"), range_mid_short_score(f)]),
            lambda f: rolling_level_cross_score(f, 24, "low", "short").gt(0.0) & quiet_bull_volume_score(f).ge(0.55),
            SHORT_EXITS,
        ),
        RefinedSpec(
            "tlv2_support_break_vp_opposition_low_short",
            "tlv2_support_break_vp_opposition_short",
            "short",
            "Does breaking TLV2 support work when VP is not defending the long side?",
            "TLV2 support gives way and VP long support/opposition is weak.",
            thresholds,
            lambda f: mean_scores(f, [line_cross_score(f, "st_1h_tlv2_support_line_rank0", "short", 0.003), "st_1h_tlv2_support_score_rank0", vp_short_minus_long_score(f), pressure_ratio_score(f, 24, "short")]),
            lambda f: line_cross_score(f, "st_1h_tlv2_support_line_rank0", "short", 0.003).gt(0.0) & num(f, "st_1h_tlv2_support_score_rank0").ge(0.45) & vp_short_minus_long_score(f).ge(0.45),
            SHORT_EXITS,
        ),
        RefinedSpec(
            "structure_breakdown_vp_opposition_low_short",
            "structure_breakdown_vp_opposition_short",
            "short",
            "Does bearish structure breakdown work when VP long support is weak?",
            "Bearish breakdown/structure state appears and VP is not supporting longs.",
            thresholds,
            lambda f: mean_scores(f, ["conf_structure_breakdown_trigger_score", "conf_structure_bearish_state_score", vp_short_minus_long_score(f), quiet_bull_volume_score(f), pressure_ratio_score(f, 24, "short")]),
            lambda f: num(f, "conf_structure_breakdown_trigger_score").ge(0.34) & vp_short_minus_long_score(f).ge(0.45),
            SHORT_EXITS,
        ),
        RefinedSpec(
            "val_break_lvn_down_thin_air_short",
            "val_break_lvn_down_short",
            "short",
            "Does a VAL/support break work better when VP shows thin space below?",
            "Price breaks below value and there is thin VP/liquidity below, so it may travel faster.",
            thresholds,
            lambda f: mean_scores(f, [val_accept_score(f), lvn_thinness_score(f, "short"), vp_short_minus_long_score(f), pressure_ratio_score(f, 24, "short")]),
            lambda f: val_accept_score(f).ge(0.35) & lvn_thinness_score(f, "short").ge(0.30) & vp_short_minus_long_score(f).ge(0.45),
            SHORT_EXITS,
        ),
        RefinedSpec(
            "support_break_orderbook_accept_short",
            "support_break_orderbook_accept_short",
            "short",
            "Does support breakdown improve when orderbook acceptance agrees?",
            "Support or recent low breaks and orderbook breakdown acceptance agrees with downside.",
            thresholds,
            lambda f: mean_scores(f, [rolling_level_cross_score(f, 24, "low", "short"), orderbook_accept_score(f, "short"), vp_short_minus_long_score(f), pressure_ratio_score(f, 24, "short")]),
            lambda f: rolling_level_cross_score(f, 24, "low", "short").gt(0.0) & orderbook_accept_score(f, "short").ge(0.25),
            SHORT_EXITS,
        ),
        RefinedSpec(
            "near_tlv2_support_breakdown_pressure_short",
            "tlv2_near_support_pressure_short",
            "short",
            "Does bearish pressure near TLV2 support warn before/at breakdown?",
            "Price is near TLV2 support, bearish pressure is active, and VP long support is weak.",
            thresholds,
            lambda f: mean_scores(f, [near_distance_score(f, "st_1h_tlv2_support_distance_atr_rank0"), "st_1h_tlv2_support_score_rank0", vp_short_minus_long_score(f), pressure_ratio_score(f, 24, "short")]),
            lambda f: near_distance_score(f, "st_1h_tlv2_support_distance_atr_rank0").ge(0.45) & pressure_ratio_score(f, 24, "short").ge(0.35) & vp_short_minus_long_score(f).ge(0.45),
            SHORT_EXITS,
        ),
        RefinedSpec(
            "range_mid_support_break_short",
            "range_mid_support_break_short",
            "short",
            "Does a support break work better while there is still room to fall?",
            "Support breaks from mid-range, not after price is already at the floor.",
            thresholds,
            lambda f: mean_scores(f, [rolling_level_cross_score(f, 24, "low", "short"), range_mid_short_score(f), vp_short_minus_long_score(f), quiet_bull_volume_score(f)]),
            lambda f: rolling_level_cross_score(f, 24, "low", "short").gt(0.0) & num(f, "px_range_position_24h").between(0.35, 0.72),
            SHORT_EXITS,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())

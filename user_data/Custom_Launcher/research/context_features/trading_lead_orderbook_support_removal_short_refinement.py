from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from trader_rule_backtests import (
    clamp01,
    export_combined_signals,
    mean_scores,
    num,
    pressure_ratio_score,
    rolling_level_cross_score,
    safe_name,
    summarize_trades,
)
from trading_lead_rectangle_vp_expansion import ExitProfile, val_accept_score
from trading_lead_rectangle_vp_refinement import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SNAPSHOT,
    RefinedSpec,
    markdown_report,
    quiet_bull_volume_score,
    run_suite,
    simulate_selected,
    vp_short_minus_long_score,
)


SHORT_EXITS = (
    ExitProfile("fast_6h_25pct", 6, 0.014, 0.025),
    ExitProfile("fast_8h_35pct", 8, 0.018, 0.035),
    ExitProfile("fast_10h_5pct", 10, 0.020, 0.050),
    ExitProfile("medium_18h_5pct", 18, 0.025, 0.050),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test sharper orderbook support-removal refinements for sparse short leads.")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="orderbook_support_removal_short_refinement_20260605")
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
        "orderbook_support_removal_short_refinement_selected_block",
        "orderbook_support_removal_short_refinement",
        "short",
        "Can orderbook support-removal make sparse support/value shorts cleaner?",
        "Only refined support-removal short variants are selected; signals are non-overlapped by rank.",
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
    strict = (0.55, 0.65, 0.75)
    return (
        RefinedSpec(
            "val_break_support_removed_vp_short",
            "val_break_orderbook_support_removed_short",
            "short",
            "Does VAL breakdown improve when orderbook support is removed?",
            "Price accepts below value, VP long opposition is weak, and orderbook support has been removed/cleared.",
            thresholds,
            lambda f: mean_scores(f, [val_accept_score(f), vp_short_minus_long_score(f), ob_support_removed_score(f), pressure_ratio_score(f, 24, "short")]),
            lambda f: val_accept_score(f).ge(0.35) & vp_short_minus_long_score(f).ge(0.45) & ob_support_removed_score(f).ge(0.25),
            SHORT_EXITS,
        ),
        RefinedSpec(
            "val_break_downside_vacuum_after_support_removed_short",
            "val_break_orderbook_vacuum_short",
            "short",
            "Does VAL breakdown improve when removed support leaves thin space below?",
            "Price breaks below value and orderbook shows downside vacuum after support removal.",
            thresholds,
            lambda f: mean_scores(f, [val_accept_score(f), vp_short_minus_long_score(f), num(f, "conf_ob_downside_vacuum_after_support_removed"), pressure_ratio_score(f, 24, "short")]),
            lambda f: val_accept_score(f).ge(0.35) & vp_short_minus_long_score(f).ge(0.45) & num(f, "conf_ob_downside_vacuum_after_support_removed").ge(0.20),
            SHORT_EXITS,
        ),
        RefinedSpec(
            "val_break_no_bounce_failure_short",
            "val_break_no_orderbook_bounce_short",
            "short",
            "Does VAL breakdown improve when orderbook is not showing bounce/failure?",
            "Price breaks below value, VP long support is weak, and orderbook bounce/failure warnings are low.",
            thresholds,
            lambda f: mean_scores(f, [val_accept_score(f), vp_short_minus_long_score(f), ob_no_bounce_score(f), quiet_bull_volume_score(f)]),
            lambda f: val_accept_score(f).ge(0.35) & vp_short_minus_long_score(f).ge(0.45) & ob_no_bounce_score(f).ge(0.60),
            SHORT_EXITS,
        ),
        RefinedSpec(
            "support_break_support_cleared_short",
            "support_break_orderbook_support_cleared_short",
            "short",
            "Does recent support break work only when orderbook says support cleared?",
            "Recent low/support breaks and orderbook support-cleared score agrees.",
            thresholds,
            lambda f: mean_scores(f, [rolling_level_cross_score(f, 24, "low", "short"), num(f, "conf_ob_support_cleared"), vp_short_minus_long_score(f), pressure_ratio_score(f, 24, "short")]),
            lambda f: rolling_level_cross_score(f, 24, "low", "short").gt(0.0) & num(f, "conf_ob_support_cleared").ge(0.25),
            SHORT_EXITS,
        ),
        RefinedSpec(
            "support_break_removed_no_bounce_short",
            "support_break_removed_no_bounce_short",
            "short",
            "Does support break improve when support is removed and bounce warnings are low?",
            "Recent low breaks, support is removed, and bounce/failure signals are quiet.",
            strict,
            lambda f: mean_scores(f, [rolling_level_cross_score(f, 24, "low", "short"), ob_support_removed_score(f), ob_no_bounce_score(f), pressure_ratio_score(f, 24, "short")]),
            lambda f: rolling_level_cross_score(f, 24, "low", "short").gt(0.0) & ob_support_removed_score(f).ge(0.25) & ob_no_bounce_score(f).ge(0.60),
            SHORT_EXITS,
        ),
        RefinedSpec(
            "support_break_bearish_book_pressure_short",
            "support_break_bearish_book_pressure_short",
            "short",
            "Does support break improve when orderbook pressure agrees with downside?",
            "Recent low breaks with bearish orderbook pressure agreement and weak VP long support.",
            thresholds,
            lambda f: mean_scores(f, [rolling_level_cross_score(f, 24, "low", "short"), ob_bearish_pressure_score(f), vp_short_minus_long_score(f), quiet_bull_volume_score(f)]),
            lambda f: rolling_level_cross_score(f, 24, "low", "short").gt(0.0) & ob_bearish_pressure_score(f).ge(0.45) & vp_short_minus_long_score(f).ge(0.45),
            SHORT_EXITS,
        ),
        RefinedSpec(
            "support_break_fast_target_only_short",
            "support_break_fast_invalidation_short",
            "short",
            "Does support break only work as a quick target-or-out short?",
            "Recent low breaks with weak VP long support; exit profile is deliberately fast.",
            strict,
            lambda f: mean_scores(f, [rolling_level_cross_score(f, 24, "low", "short"), vp_short_minus_long_score(f), quiet_bull_volume_score(f), pressure_ratio_score(f, 24, "short")]),
            lambda f: rolling_level_cross_score(f, 24, "low", "short").gt(0.0) & vp_short_minus_long_score(f).ge(0.55),
            (ExitProfile("fast_6h_25pct", 6, 0.014, 0.025), ExitProfile("fast_8h_35pct", 8, 0.018, 0.035)),
        ),
    )


def ob_support_removed_score(frame):
    return clamp01(
        mean_scores(
            frame,
            [
                "conf_ob_support_removed_strength",
                "conf_ob_support_cleared",
                "conf_ob_downside_vacuum_after_support_removed",
                "conf_ob_support_persistence_weak",
            ],
        )
    )


def ob_no_bounce_score(frame):
    bounce = mean_scores(frame, ["conf_ob_support_bounce", "conf_ob_breakdown_failure", "conf_ob_bid_absorption"])
    return (1.0 - bounce.fillna(0.0).clip(0.0, 1.0)).clip(0.0, 1.0)


def ob_bearish_pressure_score(frame):
    return clamp01(mean_scores(frame, ["conf_ob_bearish_pressure_agreement", "conf_ob_support_cleared", "conf_ob_downside_vacuum"]))


if __name__ == "__main__":
    raise SystemExit(main())

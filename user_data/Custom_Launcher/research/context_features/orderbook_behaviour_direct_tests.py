from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from orderbook_focused_tests import (
    CANDIDATE_COLUMNS,
    RESULT_COLUMNS,
    FocusedTest,
    TestWindow,
    candidate_summary,
    evaluate_feature,
    load_analysis_frame,
    matching_columns,
    slice_window,
)


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_FEATURES = USER_DATA_DIR / "orderbook_data" / "live" / "exports" / "orderbook_features_1h_behaviour_latest.parquet"
DEFAULT_OHLCV = USER_DATA_DIR / "data" / "binance" / "BTC_USDT-1h.feather"
DEFAULT_OUTPUT_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"

WINDOWS = (
    TestWindow("quiet_control", "2025-07-25", "2025-08-11", "control"),
    TestWindow("volatile_control", "2025-10-15", "2025-10-31", "control"),
    TestWindow("event_stress", "2026-01-29", "2026-02-16", "stress"),
)

TESTS = (
    FocusedTest(
        test_id="event_impulse",
        objective="Check whether pressure impulse features identify short event-style directional moves.",
        hypothesis="Large bullish or bearish pressure impulse buckets should lead short-horizon returns/path outcomes versus controls.",
        feature_includes=("beh_bullish_impulse_score", "beh_bearish_impulse_score", "beh_pressure_impulse_imbalance"),
        feature_excludes=(),
        direction="two_sided",
        primary_targets=("future_return_1h", "future_return_3h", "future_return_6h", "future_max_upside_6h", "future_max_drawdown_6h"),
        pass_rule="Effect must beat random buckets in at least two windows and weaken under shuffled labels.",
    ),
    FocusedTest(
        test_id="wall_evaporation",
        objective="Check whether disappearing bid/ask walls precede breakouts or drawdowns.",
        hypothesis="Ask-wall evaporation should align with upside path potential; bid-wall evaporation should align with downside path risk.",
        feature_includes=("beh_ask_wall_evaporation_1h", "beh_bid_wall_evaporation_1h", "beh_wall_evaporation_imbalance"),
        feature_excludes=(),
        direction="two_sided",
        primary_targets=("future_return_3h", "future_return_6h", "future_max_upside_24h", "future_max_drawdown_24h"),
        pass_rule="Directional effect must be coherent outside a single crash window and weaker after shuffled labels.",
    ),
    FocusedTest(
        test_id="persistent_zones",
        objective="Check whether persistent wall zones behave like support/resistance areas.",
        hypothesis="Persistent ask or bid zones should alter 24h/72h path outcomes more than random hours.",
        feature_includes=("beh_persistent_ask_zone_24h", "beh_persistent_bid_zone_24h", "beh_zone_compression_score"),
        feature_excludes=(),
        direction="two_sided",
        primary_targets=("future_return_24h", "future_max_upside_24h", "future_max_drawdown_24h", "future_max_upside_72h", "future_max_drawdown_72h"),
        pass_rule="Same-sign path effect in at least two windows, with random-control beat in at least two windows.",
    ),
    FocusedTest(
        test_id="book_fragility",
        objective="Check whether fragile-book states precede outsized movement or adverse path risk.",
        hypothesis="Liquidity-vacuum and spread-shock buckets should precede larger future path movement than ordinary hours.",
        feature_includes=("beh_fragile_book_score", "beh_spread_shock_24h", "beh_liquidity_vacuum_up", "beh_liquidity_vacuum_down"),
        feature_excludes=(),
        direction="top_decile",
        primary_targets=("future_return_3h", "future_return_6h", "future_max_upside_24h", "future_max_drawdown_24h"),
        pass_rule="Top-decile fragility must beat random controls and not be reproduced by shuffled labels.",
    ),
    FocusedTest(
        test_id="rebuild_acceptance",
        objective="Check whether wall rebuild after a move signals acceptance or rejection of the new area.",
        hypothesis="Support rebuild after upside movement should differ from resistance rebuild after downside movement on 6h/24h outcomes.",
        feature_includes=(
            "beh_post_breakout_support_rebuild",
            "beh_post_breakdown_resistance_rebuild",
            "beh_resistance_removed_score",
            "beh_support_removed_score",
        ),
        feature_excludes=(),
        direction="two_sided",
        primary_targets=("future_return_6h", "future_return_24h", "future_max_upside_24h", "future_max_drawdown_24h"),
        pass_rule="Effect must appear in at least two windows and be weaker after shuffled labels.",
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup or run direct tests for trader-behaviour orderbook features.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--ohlcv", type=Path, default=DEFAULT_OHLCV)
    parser.add_argument("--pair", default="BTC/USDT")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--snapshot-confirmed", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--random-controls", type=int, default=200)
    args = parser.parse_args()

    plan = test_plan_payload(args.features, args.ohlcv, args.output_dir, args.pair)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.snapshot_confirmed:
        parser.error("--execute requires --snapshot-confirmed after pausing collectors and exporting fresh parquet snapshots.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, candidates = run_tests(
        features_path=args.features,
        ohlcv_path=args.ohlcv,
        pair=args.pair,
        bootstrap_iterations=max(100, int(args.bootstrap_iterations)),
        random_controls=max(50, int(args.random_controls)),
    )
    report_path = args.output_dir / "orderbook_behaviour_direct_tests_report.csv"
    candidate_path = args.output_dir / "orderbook_behaviour_direct_tests_candidates.csv"
    summary_path = args.output_dir / "orderbook_behaviour_direct_tests_summary.json"
    results.to_csv(report_path, index=False)
    candidates.to_csv(candidate_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "plan": plan,
                "report_path": str(report_path),
                "candidate_path": str(candidate_path),
                "rows": int(len(results)),
                "candidate_rows": int(len(candidates)),
                "candidate_passes": int(candidates["candidate_pass"].sum()) if not candidates.empty else 0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report_path": str(report_path), "candidate_path": str(candidate_path), "summary_path": str(summary_path)}, indent=2))
    return 0


def test_plan_payload(features_path: Path, ohlcv_path: Path, output_dir: Path, pair: str) -> dict[str, Any]:
    return {
        "mode": "setup-only unless --execute and --snapshot-confirmed are supplied",
        "features_path": str(features_path),
        "ohlcv_path": str(ohlcv_path),
        "pair": pair,
        "output_dir": str(output_dir),
        "objective": "Test behaviour features as a trader would read them: wall evaporation, pressure impulse, fragile books, persistent zones, and rebuild/acceptance.",
        "pre_run_requirements": [
            "pause relevant orderbook collectors before making the parquet snapshot",
            "run from parquet/feather snapshots only; do not query live SQLite during research tests",
            "compare quiet control, volatile control, and event stress windows before accepting a signal",
            "record executed results in ai_guidance_docs/context_research_detailed_findings.md",
        ],
        "windows": [window.__dict__ for window in WINDOWS],
        "tests": [test.__dict__ for test in TESTS],
        "controls": [
            "same-size random-hour buckets",
            "shuffled future labels",
            "quiet control versus volatile control versus stress window comparison",
        ],
        "candidate_rule": (
            "candidate_pass requires at least three tested windows, same sign in at least two, random-control beat "
            "in at least two, and shuffled labels weaker in at least two."
        ),
    }


def run_tests(
    *,
    features_path: Path,
    ohlcv_path: Path,
    pair: str,
    bootstrap_iterations: int,
    random_controls: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = load_analysis_frame(features_path, ohlcv_path, pair)
    rows: list[dict[str, Any]] = []
    for window in WINDOWS:
        window_frame = slice_window(frame, window)
        for test in TESTS:
            for column in matching_columns(window_frame, test):
                rows.extend(evaluate_feature(window_frame, window, test, column, bootstrap_iterations, random_controls))

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if not results.empty:
        results = results.sort_values(["test_id", "window", "effect_score"], ascending=[True, True, False])
    candidates = candidate_summary(results)
    if candidates.empty:
        candidates = pd.DataFrame(columns=CANDIDATE_COLUMNS)
    return results, candidates


if __name__ == "__main__":
    raise SystemExit(main())

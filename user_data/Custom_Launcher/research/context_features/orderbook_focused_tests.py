from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_FEATURES = USER_DATA_DIR / "orderbook_data" / "live" / "exports" / "orderbook_features_1h_20260523_232420.parquet"
DEFAULT_OHLCV = USER_DATA_DIR / "data" / "binance" / "BTC_USDT-1h.feather"
DEFAULT_OUTPUT_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"

RESULT_COLUMNS = (
    "window",
    "window_role",
    "test_id",
    "objective",
    "hypothesis",
    "feature",
    "tail",
    "target",
    "n",
    "rest_n",
    "effect_delta",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "random_abs_mean",
    "random_abs_p95",
    "shuffled_delta",
    "effect_score",
    "pre_registered_pass_rule",
)
CANDIDATE_COLUMNS = (
    "test_id",
    "feature",
    "tail",
    "target",
    "windows_tested",
    "same_sign_windows",
    "beats_random_windows",
    "bootstrap_excludes_zero_windows",
    "shuffled_weaker_windows",
    "mean_abs_effect",
    "candidate_pass",
)


@dataclass(frozen=True)
class TestWindow:
    name: str
    start: str
    end: str
    role: str


@dataclass(frozen=True)
class FocusedTest:
    test_id: str
    objective: str
    hypothesis: str
    feature_includes: tuple[str, ...]
    feature_excludes: tuple[str, ...]
    direction: str
    primary_targets: tuple[str, ...]
    pass_rule: str


WINDOWS = (
    TestWindow("quiet_control", "2025-07-25", "2025-08-11", "control"),
    TestWindow("volatile_control", "2025-10-15", "2025-10-31", "control"),
    TestWindow("event_stress", "2026-01-29", "2026-02-16", "stress"),
)

COMMON_EXCLUDES = ("_z_", "_delta_", "_std", "_max_abs")

TESTS = (
    FocusedTest(
        test_id="wall_proximity",
        objective="Check whether close walls have stable directional or path-risk effects.",
        hypothesis="Near ask/bid wall distance should have a repeatable effect versus random-hour controls.",
        feature_includes=(
            "nearest_ask_wall_distance_bps_min",
            "nearest_bid_wall_distance_bps_min",
            "wall_candidate_distance_min",
        ),
        feature_excludes=COMMON_EXCLUDES,
        direction="two_sided",
        primary_targets=("future_return_24h", "future_max_upside_72h", "future_max_drawdown_72h"),
        pass_rule=(
            "Same sign in at least two of three windows, larger than random controls in at least two windows, "
            "and weaker after shuffled labels."
        ),
    ),
    FocusedTest(
        test_id="wall_notional_score",
        objective="Check whether unusually large wall notional/score predicts rejection, absorption, or drawdown risk.",
        hypothesis="Top-decile wall size/score buckets should show stronger future path effect than the rest of the sample.",
        feature_includes=(
            "nearest_ask_wall_notional_p95",
            "nearest_bid_wall_notional_p95",
            "nearest_ask_wall_score_p95",
            "nearest_bid_wall_score_p95",
        ),
        feature_excludes=COMMON_EXCLUDES,
        direction="top_decile",
        primary_targets=("future_return_24h", "future_max_upside_72h", "future_max_drawdown_72h"),
        pass_rule=(
            "Top-decile effect beats random top-decile controls in at least two windows and is not reproduced "
            "by shuffled labels."
        ),
    ),
    FocusedTest(
        test_id="liquidity_vacuum",
        objective="Check whether thin near-price liquidity predicts downside path risk.",
        hypothesis="Low bid/ask liquidity or high thinness should increase future max drawdown or breakout path movement.",
        feature_includes=(
            "bid_liquidity_5bps_min",
            "ask_liquidity_5bps_min",
            "depth_thinness_score",
            "total_depth_top20_mean",
        ),
        feature_excludes=COMMON_EXCLUDES,
        direction="fragility",
        primary_targets=("future_return_6h", "future_return_24h", "future_max_drawdown_72h"),
        pass_rule=(
            "Fragility bucket has a coherent path-risk effect in at least two windows, beats random controls, "
            "and has bootstrap support in at least one window."
        ),
    ),
    FocusedTest(
        test_id="pressure_flip",
        objective="Check whether rapidly changing book pressure predicts short-term movement.",
        hypothesis="High pressure flip count and imbalance volatility should predict larger 1h/6h moves.",
        feature_includes=("pressure_flip_count", "imbalance_top20_std", "pressure_delta_std"),
        feature_excludes=("_z_", "_delta_1h", "_delta_6h", "_max_abs"),
        direction="top_decile",
        primary_targets=("future_return_1h", "future_return_6h", "future_max_drawdown_72h"),
        pass_rule=(
            "Top-decile flip/volatility bucket beats random and shuffled controls in at least two windows."
        ),
    ),
    FocusedTest(
        test_id="spread_fragility",
        objective="Check whether spread and gap states identify fragile conditions.",
        hypothesis="Wide spread or large max gap should precede larger adverse paths.",
        feature_includes=("spread_bps_mean", "spread_bps_p95", "spread_bps_last", "max_gap_seconds"),
        feature_excludes=COMMON_EXCLUDES,
        direction="top_decile",
        primary_targets=("future_return_6h", "future_return_24h", "future_max_drawdown_72h"),
        pass_rule=(
            "Effect appears outside one crash-only window, beats random controls in at least two windows, "
            "and weakens under shuffled labels."
        ),
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Focused parquet-only orderbook tests with explicit objective/pass rules.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--ohlcv", type=Path, default=DEFAULT_OHLCV)
    parser.add_argument("--pair", default="BTC/USDT", help="Canonical feature pair to test against the OHLCV file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true", help="Actually run tests and write reports. Omit for setup/plan only.")
    parser.add_argument(
        "--snapshot-confirmed",
        action="store_true",
        help="Required with --execute to confirm collectors were paused and parquet snapshots are current.",
    )
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
    report_path = args.output_dir / "orderbook_focused_tests_report.csv"
    candidate_path = args.output_dir / "orderbook_focused_tests_candidates.csv"
    results.to_csv(report_path, index=False)
    candidates.to_csv(candidate_path, index=False)
    summary_path = args.output_dir / "orderbook_focused_tests_summary.json"
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
    print(
        json.dumps(
            {
                "report_path": str(report_path),
                "candidate_path": str(candidate_path),
                "summary_path": str(summary_path),
                "rows": int(len(results)),
                "candidate_rows": int(len(candidates)),
            },
            indent=2,
        )
    )
    return 0


def test_plan_payload(features_path: Path, ohlcv_path: Path, output_dir: Path, pair: str) -> dict[str, Any]:
    return {
        "mode": "setup-only unless --execute and --snapshot-confirmed are supplied",
        "features_path": str(features_path),
        "ohlcv_path": str(ohlcv_path),
        "pair": pair,
        "output_dir": str(output_dir),
        "pre_run_requirements": [
            "pause relevant orderbook collectors before making the parquet snapshot",
            "export a fresh 1h parquet snapshot from compacted orderbook features",
            "run from parquet/feather snapshots only; do not query live SQLite during research tests",
            "record snapshot paths in ai_guidance_docs/context_research_detailed_findings.md after execution",
        ],
        "windows": [window.__dict__ for window in WINDOWS],
        "tests": [test.__dict__ for test in TESTS],
        "controls": [
            "same-size random-hour buckets",
            "shuffled future labels",
            "quiet control versus volatile control versus stress window comparison",
            "opposite-tail sanity checks where applicable",
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
            columns = matching_columns(window_frame, test)
            for column in columns:
                rows.extend(evaluate_feature(window_frame, window, test, column, bootstrap_iterations, random_controls))

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if not results.empty:
        results = results.sort_values(["test_id", "window", "effect_score"], ascending=[True, True, False])
    return results, candidate_summary(results)


def load_analysis_frame(features_path: Path, ohlcv_path: Path, pair: str) -> pd.DataFrame:
    features = pd.read_parquet(features_path)
    if "canonical_pair" in features.columns:
        features = features[features["canonical_pair"].astype(str).str.upper() == str(pair).upper()].copy()
    prices = pd.read_feather(ohlcv_path, columns=["date", "open", "high", "low", "close", "volume"])
    features["date"] = pd.to_datetime(features["date"], utc=True, errors="coerce")
    prices["date"] = pd.to_datetime(prices["date"], utc=True, errors="coerce")
    frame = features.merge(prices, on="date", how="left").sort_values("date").reset_index(drop=True)
    for horizon in (1, 3, 6, 24, 72):
        frame[f"future_return_{horizon}h"] = frame["close"].shift(-horizon) / frame["close"] - 1.0
        future = frame["close"].shift(-1).iloc[::-1]
        frame[f"future_max_upside_{horizon}h"] = future.rolling(horizon, min_periods=horizon).max().iloc[::-1] / frame["close"] - 1.0
        frame[f"future_max_drawdown_{horizon}h"] = future.rolling(horizon, min_periods=horizon).min().iloc[::-1] / frame["close"] - 1.0
    return frame


def slice_window(frame: pd.DataFrame, window: TestWindow) -> pd.DataFrame:
    start = pd.Timestamp(window.start, tz="UTC")
    end = pd.Timestamp(window.end, tz="UTC")
    return frame[(frame["date"] >= start) & (frame["date"] < end)].copy()


def matching_columns(frame: pd.DataFrame, test: FocusedTest) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if not column.startswith("ob1h_"):
            continue
        if not any(include in column for include in test.feature_includes):
            continue
        if any(exclude in column for exclude in test.feature_excludes):
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.notna().sum() >= 48 and series.nunique(dropna=True) >= 5:
            columns.append(column)
    return columns


def evaluate_feature(
    frame: pd.DataFrame,
    window: TestWindow,
    test: FocusedTest,
    column: str,
    bootstrap_iterations: int,
    random_controls: int,
) -> list[dict[str, Any]]:
    series = pd.to_numeric(frame[column], errors="coerce")
    tails = tail_masks(series, test.direction)
    rows: list[dict[str, Any]] = []
    for tail_name, mask in tails.items():
        valid = mask & series.notna()
        if int(valid.sum()) < 12:
            continue
        rest = (~mask) & series.notna()
        for target in test.primary_targets:
            observed = effect_delta(frame[target], valid, rest)
            ci_low, ci_high = bootstrap_delta(frame[target], valid, rest, bootstrap_iterations)
            random_mean, random_p95 = random_bucket_baseline(frame[target], int(valid.sum()), random_controls)
            shuffled = shuffled_label_delta(frame[target], valid, rest)
            rows.append(
                {
                    "window": window.name,
                    "window_role": window.role,
                    "test_id": test.test_id,
                    "objective": test.objective,
                    "hypothesis": test.hypothesis,
                    "feature": column,
                    "tail": tail_name,
                    "target": target,
                    "n": int(valid.sum()),
                    "rest_n": int(rest.sum()),
                    "effect_delta": observed,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "random_abs_mean": random_mean,
                    "random_abs_p95": random_p95,
                    "shuffled_delta": shuffled,
                    "effect_score": abs(observed) - random_p95,
                    "pre_registered_pass_rule": test.pass_rule,
                }
            )
    return rows


def candidate_summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)

    rows: list[dict[str, Any]] = []
    for (test_id, feature, tail, target), group in results.groupby(["test_id", "feature", "tail", "target"], dropna=False):
        effect_values = pd.to_numeric(group["effect_delta"], errors="coerce")
        finite = group[effect_values.notna() & np.isfinite(effect_values)]
        if finite.empty:
            continue
        finite_effects = pd.to_numeric(finite["effect_delta"], errors="coerce")
        finite_random_p95 = pd.to_numeric(finite["random_abs_p95"], errors="coerce")
        finite_shuffled = pd.to_numeric(finite["shuffled_delta"], errors="coerce")
        finite_ci_low = pd.to_numeric(finite["bootstrap_ci_low"], errors="coerce")
        finite_ci_high = pd.to_numeric(finite["bootstrap_ci_high"], errors="coerce")
        signs = np.sign(finite_effects.to_numpy())
        majority_sign = 1.0 if float(signs.sum()) >= 0 else -1.0
        same_sign_windows = int((signs == majority_sign).sum())
        beats_random_windows = int((finite_effects.abs() > finite_random_p95).sum())
        bootstrap_excludes_zero_windows = int(((finite_ci_low > 0.0) | (finite_ci_high < 0.0)).sum())
        shuffled_weaker_windows = int((finite_effects.abs() > finite_shuffled.abs()).sum())
        windows_tested = int(finite["window"].nunique())
        candidate_pass = (
            windows_tested >= 3
            and same_sign_windows >= 2
            and beats_random_windows >= 2
            and shuffled_weaker_windows >= 2
        )
        rows.append(
            {
                "test_id": test_id,
                "feature": feature,
                "tail": tail,
                "target": target,
                "windows_tested": windows_tested,
                "same_sign_windows": same_sign_windows,
                "beats_random_windows": beats_random_windows,
                "bootstrap_excludes_zero_windows": bootstrap_excludes_zero_windows,
                "shuffled_weaker_windows": shuffled_weaker_windows,
                "mean_abs_effect": float(finite_effects.abs().mean()),
                "candidate_pass": bool(candidate_pass),
            }
        )

    candidates = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    if not candidates.empty:
        candidates = candidates.sort_values(["candidate_pass", "mean_abs_effect"], ascending=[False, False])
    return candidates


def tail_masks(series: pd.Series, direction: str) -> dict[str, pd.Series]:
    if direction == "fragility":
        return {
            "low10": series <= series.quantile(0.10),
            "high10": series >= series.quantile(0.90),
        }
    if direction == "top_decile":
        return {"high10": series >= series.quantile(0.90)}
    return {
        "low10": series <= series.quantile(0.10),
        "high10": series >= series.quantile(0.90),
    }


def effect_delta(target: pd.Series, mask: pd.Series, rest: pd.Series) -> float:
    selected = pd.to_numeric(target[mask], errors="coerce").dropna()
    baseline = pd.to_numeric(target[rest], errors="coerce").dropna()
    if selected.empty or baseline.empty:
        return float("nan")
    return float(selected.mean() - baseline.mean())


def bootstrap_delta(target: pd.Series, mask: pd.Series, rest: pd.Series, iterations: int) -> tuple[float, float]:
    selected = pd.to_numeric(target[mask], errors="coerce").dropna().to_numpy()
    baseline = pd.to_numeric(target[rest], errors="coerce").dropna().to_numpy()
    if len(selected) < 3 or len(baseline) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(1337)
    values = []
    for _ in range(iterations):
        sample_a = rng.choice(selected, size=len(selected), replace=True)
        sample_b = rng.choice(baseline, size=len(baseline), replace=True)
        values.append(float(np.mean(sample_a) - np.mean(sample_b)))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def random_bucket_baseline(target: pd.Series, n: int, iterations: int) -> tuple[float, float]:
    values = pd.to_numeric(target, errors="coerce").dropna().to_numpy()
    if len(values) <= n or n < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(2026)
    effects = []
    indexes = np.arange(len(values))
    for _ in range(iterations):
        selected_idx = rng.choice(indexes, size=n, replace=False)
        selected_mask = np.zeros(len(values), dtype=bool)
        selected_mask[selected_idx] = True
        effects.append(abs(float(values[selected_mask].mean() - values[~selected_mask].mean())))
    return float(np.mean(effects)), float(np.quantile(effects, 0.95))


def shuffled_label_delta(target: pd.Series, mask: pd.Series, rest: pd.Series) -> float:
    values = pd.to_numeric(target, errors="coerce")
    rng = np.random.default_rng(9001)
    shuffled = values.copy()
    valid_values = shuffled.dropna().to_numpy()
    rng.shuffle(valid_values)
    shuffled.loc[shuffled.notna()] = valid_values
    return effect_delta(shuffled, mask, rest)


if __name__ == "__main__":
    raise SystemExit(main())

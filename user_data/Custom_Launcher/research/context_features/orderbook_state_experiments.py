from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from orderbook_focused_tests import (
    CANDIDATE_COLUMNS,
    RESULT_COLUMNS,
    bootstrap_delta,
    candidate_summary,
    effect_delta,
    random_bucket_baseline,
    shuffled_label_delta,
)


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_FEATURES = USER_DATA_DIR / "orderbook_data" / "live" / "exports" / "orderbook_features_1h_behaviour_latest.parquet"
DEFAULT_OHLCV = USER_DATA_DIR / "data" / "binance" / "BTC_USDT-1h.feather"
DEFAULT_OUTPUT_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"


@dataclass(frozen=True)
class Experiment:
    test_id: str
    objective: str
    hypothesis: str
    mask_builder: Callable[[pd.DataFrame], pd.Series]
    primary_targets: tuple[str, ...]
    pass_rule: str


def main() -> int:
    parser = argparse.ArgumentParser(description="State-based orderbook experiments using parquet snapshots only.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--ohlcv", type=Path, default=DEFAULT_OHLCV)
    parser.add_argument("--pair", default="BTC/USDT")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--snapshot-confirmed", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--random-controls", type=int, default=200)
    parser.add_argument("--min-pass-windows", type=int, default=6)
    parser.add_argument("--tag", default="v2", help="Output filename tag so new experiment batches do not overwrite older reports.")
    args = parser.parse_args()

    frame = load_frame(args.features, args.ohlcv, args.pair)
    windows = monthly_windows(frame)
    plan = {
        "mode": "setup-only unless --execute and --snapshot-confirmed are supplied",
        "features_path": str(args.features),
        "ohlcv_path": str(args.ohlcv),
        "pair": args.pair,
        "windows": windows,
        "experiments": [
            {
                "test_id": experiment.test_id,
                "objective": experiment.objective,
                "hypothesis": experiment.hypothesis,
                "primary_targets": experiment.primary_targets,
                "pass_rule": experiment.pass_rule,
            }
            for experiment in EXPERIMENTS
        ],
        "candidate_rule": (
            f"candidate_pass requires at least {args.min_pass_windows} monthly windows, same sign in at least 70% of tested windows, "
            "random-control beat in at least 50%, shuffled labels weaker in at least 60%, and bootstrap support in at least 3 windows."
        ),
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.snapshot_confirmed:
        parser.error("--execute requires --snapshot-confirmed after confirming the parquet/feather snapshots are current.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = run_experiments(frame, windows, args.bootstrap_iterations, args.random_controls)
    candidates = state_candidate_summary(results, min_pass_windows=args.min_pass_windows)
    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag.strip()) or "latest"
    report_path = args.output_dir / f"orderbook_state_experiments_{safe_tag}_report.csv"
    candidate_path = args.output_dir / f"orderbook_state_experiments_{safe_tag}_candidates.csv"
    summary_path = args.output_dir / f"orderbook_state_experiments_{safe_tag}_summary.json"
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


def load_frame(features_path: Path, ohlcv_path: Path, pair: str) -> pd.DataFrame:
    features = pd.read_parquet(features_path)
    if "canonical_pair" in features.columns:
        features = features[features["canonical_pair"].astype(str).str.upper() == str(pair).upper()].copy()
    prices = pd.read_feather(ohlcv_path, columns=["date", "open", "high", "low", "close", "volume"])
    features["date"] = pd.to_datetime(features["date"], utc=True, errors="coerce")
    features = features.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    features["feature_present"] = 1.0
    prices["date"] = pd.to_datetime(prices["date"], utc=True, errors="coerce")
    prices["date"] = prices["date"] + pd.Timedelta(hours=1)
    if features.empty:
        frame = prices.copy()
        frame["feature_present"] = 0.0
    else:
        start = features["date"].min()
        end = features["date"].max()
        base = prices[(prices["date"] >= start) & (prices["date"] <= end)].copy()
        frame = base.merge(features, on="date", how="left", sort=False)
        frame["feature_present"] = pd.to_numeric(frame["feature_present"], errors="coerce").fillna(0.0)
    frame = frame.sort_values("date").reset_index(drop=True)
    frame = append_price_context(frame)
    frame = append_orderbook_derived_context(frame)
    for horizon in (1, 3, 6, 24, 72):
        frame[f"future_return_{horizon}h"] = frame["close"].shift(-horizon) / frame["close"] - 1.0
        future = frame["close"].shift(-1).iloc[::-1]
        frame[f"future_max_upside_{horizon}h"] = future.rolling(horizon, min_periods=horizon).max().iloc[::-1] / frame["close"] - 1.0
        frame[f"future_max_drawdown_{horizon}h"] = future.rolling(horizon, min_periods=horizon).min().iloc[::-1] / frame["close"] - 1.0
    frame["breakout_up_24h"] = (frame["future_max_upside_24h"] >= 0.04).astype(float).where(frame["future_max_upside_24h"].notna())
    frame["breakout_up_72h"] = (frame["future_max_upside_72h"] >= 0.05).astype(float).where(frame["future_max_upside_72h"].notna())
    frame["shock_down_24h"] = (frame["future_max_drawdown_24h"] <= -0.04).astype(float).where(frame["future_max_drawdown_24h"].notna())
    frame["shock_down_72h"] = (frame["future_max_drawdown_72h"] <= -0.05).astype(float).where(frame["future_max_drawdown_72h"].notna())
    frame = append_trader_event_labels(frame)
    return frame


def append_price_context(frame: pd.DataFrame) -> pd.DataFrame:
    candle_range = (frame["high"] - frame["low"]).replace(0, pd.NA)
    body_pressure = ((frame["close"] - frame["open"]) / candle_range).clip(-1.0, 1.0).fillna(0.0)
    signed_volume = body_pressure * frame["volume"]
    for window in (24, 72):
        prior_high = frame["high"].shift(1).rolling(window, min_periods=window // 2).max()
        prior_low = frame["low"].shift(1).rolling(window, min_periods=window // 2).min()
        prior_range = (prior_high - prior_low).replace(0, pd.NA)
        frame[f"close_range_pos_{window}h"] = ((frame["close"] - prior_low) / prior_range).clip(0.0, 1.0)
        frame[f"close_breakout_{window}h"] = (frame["close"] > prior_high).astype(float)
        frame[f"close_breakdown_{window}h"] = (frame["close"] < prior_low).astype(float)
        volume_sum = frame["volume"].rolling(window, min_periods=window // 2).sum()
        frame[f"volume_pressure_{window}h"] = signed_volume.rolling(window, min_periods=window // 2).sum() / volume_sum.replace(0, pd.NA)
    return frame


def append_orderbook_derived_context(frame: pd.DataFrame) -> pd.DataFrame:
    frame["price_return_1h"] = frame["close"].pct_change(1, fill_method=None)
    frame["price_return_3h"] = frame["close"].pct_change(3, fill_method=None)
    frame["price_return_6h"] = frame["close"].pct_change(6, fill_method=None)
    frame["price_return_24h"] = frame["close"].pct_change(24, fill_method=None)
    rolling_range_6h = (frame["high"].rolling(6, min_periods=3).max() - frame["low"].rolling(6, min_periods=3).min()) / frame["close"]
    rolling_range_24h = (frame["high"].rolling(24, min_periods=12).max() - frame["low"].rolling(24, min_periods=12).min()) / frame["close"]
    frame["price_stall_6h"] = rolling_range_6h <= rolling_range_6h.rolling(168, min_periods=48).quantile(0.35)
    frame["price_stall_24h"] = rolling_range_24h <= rolling_range_24h.rolling(168, min_periods=48).quantile(0.35)
    frame["near_72h_high"] = col(frame, "close_range_pos_72h") >= 0.80
    frame["near_72h_low"] = col(frame, "close_range_pos_72h") <= 0.20
    frame["near_24h_high"] = col(frame, "close_range_pos_24h") >= 0.75
    frame["near_24h_low"] = col(frame, "close_range_pos_24h") <= 0.25
    return frame


def append_trader_event_labels(frame: pd.DataFrame) -> pd.DataFrame:
    prior_high_24h = frame["high"].shift(1).rolling(24, min_periods=12).max()
    prior_low_24h = frame["low"].shift(1).rolling(24, min_periods=12).min()
    future_high_6h = _future_extreme(frame["high"], 6, "max")
    future_low_6h = _future_extreme(frame["low"], 6, "min")
    future_high_24h = _future_extreme(frame["high"], 24, "max")
    future_low_24h = _future_extreme(frame["low"], 24, "min")
    future_close_6h = frame["close"].shift(-6)
    future_close_24h = frame["close"].shift(-24)
    breakout_attempt_6h = future_high_6h >= prior_high_24h * (1.0 + 0.001)
    breakdown_attempt_6h = future_low_6h <= prior_low_24h * (1.0 - 0.001)
    breakout_attempt_24h = future_high_24h >= prior_high_24h * (1.0 + 0.001)
    breakdown_attempt_24h = future_low_24h <= prior_low_24h * (1.0 - 0.001)
    frame["breakout_attempt_next_6h"] = breakout_attempt_6h.astype(float).where(prior_high_24h.notna())
    frame["breakout_success_next_6h"] = (breakout_attempt_6h & (future_close_6h > prior_high_24h * (1.0 + 0.001))).astype(float).where(prior_high_24h.notna())
    frame["breakout_failure_next_6h"] = (breakout_attempt_6h & (future_close_6h < prior_high_24h * (1.0 - 0.001))).astype(float).where(prior_high_24h.notna())
    frame["breakdown_attempt_next_6h"] = breakdown_attempt_6h.astype(float).where(prior_low_24h.notna())
    frame["breakdown_success_next_6h"] = (breakdown_attempt_6h & (future_close_6h < prior_low_24h * (1.0 - 0.001))).astype(float).where(prior_low_24h.notna())
    frame["breakdown_failure_next_6h"] = (breakdown_attempt_6h & (future_close_6h > prior_low_24h * (1.0 + 0.001))).astype(float).where(prior_low_24h.notna())
    frame["failed_breakout_next_24h"] = (breakout_attempt_24h & (future_close_24h < prior_high_24h)).astype(float).where(prior_high_24h.notna())
    frame["failed_breakdown_next_24h"] = (breakdown_attempt_24h & (future_close_24h > prior_low_24h)).astype(float).where(prior_low_24h.notna())
    frame["large_drawdown_next_6h"] = (frame["future_max_drawdown_6h"] <= -0.03).astype(float).where(frame["future_max_drawdown_6h"].notna())
    frame["large_drawdown_next_24h"] = (frame["future_max_drawdown_24h"] <= -0.04).astype(float).where(frame["future_max_drawdown_24h"].notna())
    frame["fakeout_next_24h"] = (
        (breakout_attempt_24h & (future_close_24h < prior_high_24h))
        | (breakdown_attempt_24h & (future_close_24h > prior_low_24h))
    ).astype(float).where(prior_high_24h.notna() & prior_low_24h.notna())
    frame["time_to_plus_2pct"] = _time_to_threshold(frame, up_pct=0.02, down_pct=None, horizon=24)
    frame["time_to_minus_2pct"] = _time_to_threshold(frame, up_pct=None, down_pct=-0.02, horizon=24)
    up_first = _hit_before(frame, up_pct=0.03, down_pct=-0.02, horizon=24, direction="up")
    down_first = _hit_before(frame, up_pct=0.03, down_pct=-0.02, horizon=24, direction="down")
    frame["hit_plus_3pct_before_minus_2pct"] = up_first
    frame["hit_minus_3pct_before_plus_2pct"] = down_first
    frame["max_up_before_down_6h"] = frame["future_max_upside_6h"] - frame["future_max_drawdown_6h"].abs()
    frame["max_down_before_up_6h"] = frame["future_max_drawdown_6h"].abs() - frame["future_max_upside_6h"]
    return frame


def _future_extreme(series: pd.Series, horizon: int, method: str) -> pd.Series:
    future = series.shift(-1).iloc[::-1]
    rolling = future.rolling(horizon, min_periods=horizon)
    result = rolling.max() if method == "max" else rolling.min()
    return result.iloc[::-1]


def _time_to_threshold(frame: pd.DataFrame, *, up_pct: float | None, down_pct: float | None, horizon: int) -> pd.Series:
    close = frame["close"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    out = np.full(len(frame), np.nan)
    for i in range(len(frame)):
        if not np.isfinite(close[i]) or close[i] == 0.0:
            continue
        for step in range(1, horizon + 1):
            j = i + step
            if j >= len(frame):
                break
            if up_pct is not None and high[j] >= close[i] * (1.0 + up_pct):
                out[i] = float(step)
                break
            if down_pct is not None and low[j] <= close[i] * (1.0 + down_pct):
                out[i] = float(step)
                break
    return pd.Series(out, index=frame.index)


def _hit_before(frame: pd.DataFrame, *, up_pct: float, down_pct: float, horizon: int, direction: str) -> pd.Series:
    close = frame["close"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    out = np.full(len(frame), np.nan)
    for i in range(len(frame)):
        if not np.isfinite(close[i]) or close[i] == 0.0:
            continue
        result = np.nan
        for step in range(1, horizon + 1):
            j = i + step
            if j >= len(frame):
                break
            hit_up = high[j] >= close[i] * (1.0 + up_pct)
            hit_down = low[j] <= close[i] * (1.0 + down_pct)
            if hit_up and hit_down:
                result = 0.5
                break
            if hit_up:
                result = 1.0 if direction == "up" else 0.0
                break
            if hit_down:
                result = 1.0 if direction == "down" else 0.0
                break
        out[i] = result
    return pd.Series(out, index=frame.index)


def monthly_windows(frame: pd.DataFrame) -> list[dict[str, str]]:
    dated = frame.dropna(subset=["date"]).copy()
    dated["month"] = dated["date"].dt.to_period("M").astype(str)
    windows = []
    for month, group in dated.groupby("month", sort=True):
        coverage_rows = int(col(group, "feature_present").fillna(1.0).gt(0.0).sum())
        if coverage_rows < 300:
            continue
        start = group["date"].min().strftime("%Y-%m-%d")
        end = (group["date"].max() + pd.Timedelta(hours=1)).strftime("%Y-%m-%d")
        windows.append({"name": month, "start": start, "end": end, "role": "monthly"})
    return windows


def run_experiments(
    frame: pd.DataFrame,
    windows: list[dict[str, str]],
    bootstrap_iterations: int,
    random_controls: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for window in windows:
        window_frame = slice_window(frame, window)
        for experiment in EXPERIMENTS:
            mask = experiment.mask_builder(window_frame).fillna(False)
            rows.extend(evaluate_state(window_frame, window, experiment, mask, bootstrap_iterations, random_controls))
    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if not results.empty:
        results = results.sort_values(["test_id", "window", "effect_score"], ascending=[True, True, False])
    return results


def slice_window(frame: pd.DataFrame, window: dict[str, str]) -> pd.DataFrame:
    start = pd.Timestamp(window["start"], tz="UTC")
    end = pd.Timestamp(window["end"], tz="UTC")
    return frame[(frame["date"] >= start) & (frame["date"] < end)].copy()


def evaluate_state(
    frame: pd.DataFrame,
    window: dict[str, str],
    experiment: Experiment,
    mask: pd.Series,
    bootstrap_iterations: int,
    random_controls: int,
) -> list[dict[str, object]]:
    valid = mask.reindex(frame.index, fill_value=False).astype(bool)
    base = col(frame, "feature_present").fillna(1.0) > 0.0
    valid = valid & base
    if int(valid.sum()) < 12:
        return []
    rest = base & ~valid
    rows: list[dict[str, object]] = []
    for target in experiment.primary_targets:
        observed = effect_delta(frame[target], valid, rest)
        ci_low, ci_high = bootstrap_delta(frame[target], valid, rest, bootstrap_iterations)
        random_mean, random_p95 = random_bucket_baseline(frame[target], int(valid.sum()), random_controls)
        shuffled = shuffled_label_delta(frame[target], valid, rest)
        rows.append(
            {
                "window": window["name"],
                "window_role": window["role"],
                "test_id": experiment.test_id,
                "objective": experiment.objective,
                "hypothesis": experiment.hypothesis,
                "feature": experiment.test_id,
                "tail": "state",
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
                "pre_registered_pass_rule": experiment.pass_rule,
            }
        )
    return rows


def state_candidate_summary(results: pd.DataFrame, *, min_pass_windows: int = 6) -> pd.DataFrame:
    base = candidate_summary(results)
    if base.empty:
        return base
    rows = []
    for _, row in base.iterrows():
        windows = int(row["windows_tested"])
        same_sign = int(row["same_sign_windows"])
        beats_random = int(row["beats_random_windows"])
        bootstrap = int(row["bootstrap_excludes_zero_windows"])
        shuffled = int(row["shuffled_weaker_windows"])
        row = row.to_dict()
        same_sign_required = min(windows, max(3, int(np.ceil(windows * 0.70))))
        beats_random_required = min(windows, max(3, int(np.ceil(windows * 0.50))))
        shuffled_required = min(windows, max(3, int(np.ceil(windows * 0.60))))
        bootstrap_required = min(windows, 3)
        row["candidate_pass"] = bool(
            windows >= min_pass_windows
            and same_sign >= same_sign_required
            and beats_random >= beats_random_required
            and shuffled >= shuffled_required
            and bootstrap >= bootstrap_required
        )
        rows.append(row)
    candidates = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    return candidates.sort_values(["candidate_pass", "mean_abs_effect"], ascending=[False, False])


def col(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(frame.get(name, pd.Series(index=frame.index, dtype=float)), errors="coerce")


def high(frame: pd.DataFrame, name: str, quantile: float = 0.80) -> pd.Series:
    series = col(frame, name)
    return series >= series.quantile(quantile)


def low(frame: pd.DataFrame, name: str, quantile: float = 0.20) -> pd.Series:
    series = col(frame, name)
    return series <= series.quantile(quantile)


def pos(frame: pd.DataFrame, name: str) -> pd.Series:
    return col(frame, name).fillna(0.0) > 0


ZONE = "ob1h_bybit_spot_beh_zone_compression_score"
ASK_ZONE = "ob1h_bybit_spot_beh_persistent_ask_zone_24h"
BID_ZONE = "ob1h_bybit_spot_beh_persistent_bid_zone_24h"
SPREAD = "ob1h_bybit_spot_beh_spread_shock_24h"
RES_REMOVED = "ob1h_bybit_spot_beh_resistance_removed_score"
SUP_REMOVED = "ob1h_bybit_spot_beh_support_removed_score"
BULL_IMPULSE = "ob1h_bybit_spot_beh_bullish_impulse_score"
BEAR_IMPULSE = "ob1h_bybit_spot_beh_bearish_impulse_score"
VAC_UP = "ob1h_bybit_spot_beh_liquidity_vacuum_up"
VAC_DOWN = "ob1h_bybit_spot_beh_liquidity_vacuum_down"
ASK_EVAP = "ob1h_bybit_spot_beh_ask_wall_evaporation_1h"
BID_EVAP = "ob1h_bybit_spot_beh_bid_wall_evaporation_1h"
ASK_DIST = "ob1h_bybit_spot_nearest_ask_wall_distance_bps_min"
BID_DIST = "ob1h_bybit_spot_nearest_bid_wall_distance_bps_min"
ASK_DIST_SLOPE = "ob1h_bybit_spot_nearest_ask_wall_distance_bps_slope"
BID_DIST_SLOPE = "ob1h_bybit_spot_nearest_bid_wall_distance_bps_slope"
ASK_NOTIONAL_STD = "ob1h_bybit_spot_nearest_ask_wall_notional_std"
BID_NOTIONAL_STD = "ob1h_bybit_spot_nearest_bid_wall_notional_std"
ASK_SCORE_STD = "ob1h_bybit_spot_nearest_ask_wall_score_std"
BID_SCORE_STD = "ob1h_bybit_spot_nearest_bid_wall_score_std"
ASK_SCORE_SLOPE = "ob1h_bybit_spot_nearest_ask_wall_score_slope"
BID_SCORE_SLOPE = "ob1h_bybit_spot_nearest_bid_wall_score_slope"
PRESSURE_DELTA = "ob1h_bybit_spot_pressure_delta"
PRESSURE_FLIPS = "ob1h_bybit_spot_pressure_flip_count"
ALPHA = "ob1h_bybit_spot_alpha"


EXPERIMENTS = (
    Experiment(
        "compressed_near_resistance",
        "Test compression when price is near the upper 72h range.",
        "Compressed books near resistance should change 24h/72h upside and drawdown paths.",
        lambda f: high(f, ZONE) & (col(f, "close_range_pos_72h") >= 0.80),
        ("future_return_24h", "future_max_upside_72h", "future_max_drawdown_72h", "breakout_up_72h"),
        "Must survive monthly windows and beat random/shuffled controls.",
    ),
    Experiment(
        "compressed_near_support",
        "Test compression when price is near the lower 72h range.",
        "Compressed books near support should change bounce or breakdown path risk.",
        lambda f: high(f, ZONE) & (col(f, "close_range_pos_72h") <= 0.20),
        ("future_return_24h", "future_max_upside_72h", "future_max_drawdown_72h", "shock_down_72h"),
        "Must survive monthly windows and beat random/shuffled controls.",
    ),
    Experiment(
        "resistance_removed_breakout_acceptance",
        "Test removed resistance while price is already high in its range.",
        "Resistance removal near the high should precede upside acceptance more than random hours.",
        lambda f: high(f, RES_REMOVED) & ((col(f, "close_range_pos_24h") >= 0.75) | pos(f, "close_breakout_24h")),
        ("future_return_6h", "future_return_24h", "future_max_upside_24h", "future_max_upside_72h", "breakout_up_72h"),
        "Must survive monthly windows and beat random/shuffled controls.",
    ),
    Experiment(
        "support_removed_breakdown_risk",
        "Test removed support while price is already low in its range.",
        "Support removal near the low should precede drawdown or breakdown risk more than random hours.",
        lambda f: high(f, SUP_REMOVED) & ((col(f, "close_range_pos_24h") <= 0.25) | pos(f, "close_breakdown_24h")),
        ("future_return_6h", "future_return_24h", "future_max_drawdown_24h", "future_max_drawdown_72h", "shock_down_72h"),
        "Must survive monthly windows and beat random/shuffled controls.",
    ),
    Experiment(
        "thin_book_after_resistance_removed",
        "Test resistance removal plus upside liquidity vacuum.",
        "If resistance is removed and the book is thin above, upside path should expand.",
        lambda f: high(f, RES_REMOVED) & high(f, VAC_UP),
        ("future_return_6h", "future_max_upside_24h", "future_max_upside_72h", "breakout_up_72h"),
        "Must survive monthly windows and beat random/shuffled controls.",
    ),
    Experiment(
        "thin_book_after_support_removed",
        "Test support removal plus downside liquidity vacuum.",
        "If support is removed and the book is thin below, drawdown path should expand.",
        lambda f: high(f, SUP_REMOVED) & high(f, VAC_DOWN),
        ("future_return_6h", "future_max_drawdown_24h", "future_max_drawdown_72h", "shock_down_72h"),
        "Must survive monthly windows and beat random/shuffled controls.",
    ),
    Experiment(
        "compression_with_spread_shock",
        "Test compressed zones with spread shock.",
        "Compression plus spread shock should identify unstable transition hours.",
        lambda f: high(f, ZONE) & high(f, SPREAD),
        ("future_return_24h", "future_max_upside_72h", "future_max_drawdown_72h"),
        "Must survive monthly windows and beat random/shuffled controls.",
    ),
    Experiment(
        "bullish_pressure_breakout_setup",
        "Test bullish pressure while price presses the top of its range.",
        "Bullish pressure plus high price location should precede upside continuation.",
        lambda f: high(f, BULL_IMPULSE) & (col(f, "volume_pressure_24h") > 0) & (col(f, "close_range_pos_24h") >= 0.70),
        ("future_return_3h", "future_return_6h", "future_max_upside_24h", "breakout_up_24h"),
        "Must survive monthly windows and beat random/shuffled controls.",
    ),
    Experiment(
        "bearish_pressure_breakdown_setup",
        "Test bearish pressure while price presses the bottom of its range.",
        "Bearish pressure plus low price location should precede downside continuation.",
        lambda f: high(f, BEAR_IMPULSE) & (col(f, "volume_pressure_24h") < 0) & (col(f, "close_range_pos_24h") <= 0.30),
        ("future_return_3h", "future_return_6h", "future_max_drawdown_24h", "shock_down_24h"),
        "Must survive monthly windows and beat random/shuffled controls.",
    ),
    Experiment(
        "ask_wall_evaporation_with_breakout",
        "Test ask wall evaporation during breakout pressure.",
        "Ask wall evaporation while price is high should precede upside path expansion.",
        lambda f: high(f, ASK_EVAP) & high(f, BULL_IMPULSE) & (col(f, "close_range_pos_24h") >= 0.70),
        ("future_return_3h", "future_return_6h", "future_max_upside_24h", "breakout_up_24h"),
        "Must survive monthly windows and beat random/shuffled controls.",
    ),
    Experiment(
        "bid_wall_evaporation_with_breakdown",
        "Test bid wall evaporation during breakdown pressure.",
        "Bid wall evaporation while price is low should precede downside path expansion.",
        lambda f: high(f, BID_EVAP) & high(f, BEAR_IMPULSE) & (col(f, "close_range_pos_24h") <= 0.30),
        ("future_return_3h", "future_return_6h", "future_max_drawdown_24h", "shock_down_24h"),
        "Must survive monthly windows and beat random/shuffled controls.",
    ),
    Experiment(
        "persistent_ask_rejection_zone",
        "Test persistent ask walls close above price.",
        "Persistent ask zones close to price should act like resistance or slow upside.",
        lambda f: high(f, ASK_ZONE) & low(f, ASK_DIST, 0.35) & (col(f, "close_range_pos_72h") >= 0.60),
        ("future_return_24h", "future_max_upside_72h", "future_max_drawdown_72h"),
        "Must survive monthly windows and beat random/shuffled controls.",
    ),
    Experiment(
        "persistent_bid_support_zone",
        "Test persistent bid walls close below price.",
        "Persistent bid zones close to price should act like support or slow downside.",
        lambda f: high(f, BID_ZONE) & low(f, BID_DIST, 0.35) & (col(f, "close_range_pos_72h") <= 0.40),
        ("future_return_24h", "future_max_upside_72h", "future_max_drawdown_72h"),
        "Must survive monthly windows and beat random/shuffled controls.",
    ),
    Experiment(
        "combined_upside_acceptance",
        "Combine compression, resistance removal, bullish pressure, and volume pressure.",
        "A confluence of removed resistance, compression, bullish pressure, and positive volume pressure should be stronger than each component.",
        lambda f: high(f, ZONE, 0.65) & high(f, RES_REMOVED, 0.65) & high(f, BULL_IMPULSE, 0.65) & (col(f, "volume_pressure_24h") > 0),
        ("future_return_6h", "future_return_24h", "future_max_upside_24h", "future_max_upside_72h", "breakout_up_72h"),
        "Combination must beat random/shuffled controls across monthly windows.",
    ),
    Experiment(
        "combined_downside_risk",
        "Combine compression, support removal, bearish pressure, and negative volume pressure.",
        "A confluence of removed support, compression, bearish pressure, and negative volume pressure should be stronger than each component.",
        lambda f: high(f, ZONE, 0.65) & high(f, SUP_REMOVED, 0.65) & high(f, BEAR_IMPULSE, 0.65) & (col(f, "volume_pressure_24h") < 0),
        ("future_return_6h", "future_return_24h", "future_max_drawdown_24h", "future_max_drawdown_72h", "shock_down_72h"),
        "Combination must beat random/shuffled controls across monthly windows.",
    ),
    Experiment(
        "ask_wall_chasing_price_rejection",
        "Approximate ask-wall migration toward price near resistance.",
        "If ask-wall distance is shrinking while price is near the high, the wall may be chasing price and capping upside.",
        lambda f: low(f, ASK_DIST_SLOPE, 0.25) & (col(f, "near_72h_high") > 0),
        ("future_return_6h", "future_return_24h", "future_max_upside_24h", "future_max_upside_72h", "breakout_up_72h"),
        "Migration proxy must survive monthly controls.",
    ),
    Experiment(
        "bid_wall_chasing_price_support",
        "Approximate bid-wall migration toward price near support.",
        "If bid-wall distance is shrinking while price is near the low, the wall may be chasing price and supporting downside risk reversal.",
        lambda f: low(f, BID_DIST_SLOPE, 0.25) & (col(f, "near_72h_low") > 0),
        ("future_return_6h", "future_return_24h", "future_max_drawdown_24h", "future_max_drawdown_72h", "shock_down_72h"),
        "Migration proxy must survive monthly controls.",
    ),
    Experiment(
        "ask_absorption_rejection",
        "Test ask-wall absorption/rejection proxy.",
        "Strong/persistent ask wall plus bullish pressure and stalled price near resistance should reduce upside follow-through.",
        lambda f: high(f, ASK_ZONE, 0.65) & high(f, BULL_IMPULSE, 0.65) & (col(f, "near_72h_high") > 0) & (col(f, "price_stall_6h") > 0),
        ("future_return_6h", "future_return_24h", "future_max_upside_24h", "future_max_upside_72h", "breakout_up_72h"),
        "Absorption proxy must survive monthly controls.",
    ),
    Experiment(
        "bid_absorption_breakdown_risk",
        "Test bid-wall absorption/breakdown proxy.",
        "Strong/persistent bid wall plus bearish pressure and stalled price near support should increase downside path risk.",
        lambda f: high(f, BID_ZONE, 0.65) & high(f, BEAR_IMPULSE, 0.65) & (col(f, "near_72h_low") > 0) & (col(f, "price_stall_6h") > 0),
        ("future_return_6h", "future_return_24h", "future_max_drawdown_24h", "future_max_drawdown_72h", "shock_down_72h"),
        "Absorption proxy must survive monthly controls.",
    ),
    Experiment(
        "bullish_pressure_exhaustion",
        "Test bullish pressure exhaustion.",
        "Bullish pressure after a recent rise, stalled price, and resistance compression should warn that upside is exhausted.",
        lambda f: high(f, BULL_IMPULSE, 0.65) & (col(f, "price_return_6h") > 0) & (col(f, "price_stall_6h") > 0) & high(f, ZONE, 0.65),
        ("future_return_3h", "future_return_6h", "future_return_24h", "future_max_upside_24h", "future_max_drawdown_24h"),
        "Exhaustion proxy must survive monthly controls.",
    ),
    Experiment(
        "bearish_pressure_exhaustion",
        "Test bearish pressure exhaustion.",
        "Bearish pressure after a recent fall, stalled price, and support compression may warn that downside is exhausted.",
        lambda f: high(f, BEAR_IMPULSE, 0.65) & (col(f, "price_return_6h") < 0) & (col(f, "price_stall_6h") > 0) & high(f, ZONE, 0.65),
        ("future_return_3h", "future_return_6h", "future_return_24h", "future_max_upside_24h", "future_max_drawdown_24h"),
        "Exhaustion proxy must survive monthly controls.",
    ),
    Experiment(
        "ask_wall_flicker_spoof_risk",
        "Approximate ask-wall flicker/spoof risk.",
        "High ask-wall variability plus ask evaporation near resistance should behave differently from persistent resistance.",
        lambda f: high(f, ASK_SCORE_STD, 0.70) & high(f, ASK_NOTIONAL_STD, 0.70) & high(f, ASK_EVAP, 0.65) & (col(f, "near_72h_high") > 0),
        ("future_return_3h", "future_return_6h", "future_max_upside_24h", "breakout_up_24h"),
        "Flicker proxy must survive monthly controls.",
    ),
    Experiment(
        "bid_wall_flicker_spoof_risk",
        "Approximate bid-wall flicker/spoof risk.",
        "High bid-wall variability plus bid evaporation near support should behave differently from persistent support.",
        lambda f: high(f, BID_SCORE_STD, 0.70) & high(f, BID_NOTIONAL_STD, 0.70) & high(f, BID_EVAP, 0.65) & (col(f, "near_72h_low") > 0),
        ("future_return_3h", "future_return_6h", "future_max_drawdown_24h", "shock_down_24h"),
        "Flicker proxy must survive monthly controls.",
    ),
    Experiment(
        "post_breakout_support_rebuild_acceptance",
        "Test post-breakout support rebuild.",
        "If price breaks out and new bid support rebuilds, upside continuation should improve.",
        lambda f: high(f, "ob1h_bybit_spot_beh_post_breakout_support_rebuild", 0.65) & (pos(f, "close_breakout_24h") | (col(f, "near_72h_high") > 0)),
        ("future_return_6h", "future_return_24h", "future_max_upside_24h", "future_max_upside_72h", "breakout_up_72h"),
        "Rebuild proxy must survive monthly controls.",
    ),
    Experiment(
        "post_breakdown_resistance_rebuild_acceptance",
        "Test post-breakdown resistance rebuild.",
        "If price breaks down and new ask resistance rebuilds, downside continuation should improve.",
        lambda f: high(f, "ob1h_bybit_spot_beh_post_breakdown_resistance_rebuild", 0.65) & (pos(f, "close_breakdown_24h") | (col(f, "near_72h_low") > 0)),
        ("future_return_6h", "future_return_24h", "future_max_drawdown_24h", "future_max_drawdown_72h", "shock_down_72h"),
        "Rebuild proxy must survive monthly controls.",
    ),
    Experiment(
        "unstable_pressure_flip_high",
        "Test unstable high-pressure flip state.",
        "Frequent pressure flips with spread shock should identify unstable book states with larger paths.",
        lambda f: high(f, PRESSURE_FLIPS, 0.75) & high(f, SPREAD, 0.65),
        ("future_return_3h", "future_return_6h", "future_max_upside_24h", "future_max_drawdown_24h"),
        "Instability proxy must survive monthly controls.",
    ),
    Experiment(
        "alpha_breakout_acceptance",
        "Objective List Alpha: test accepted breakout state.",
        "Price accepted above a prior resistance zone with support rebuild should precede upside path expansion.",
        lambda f: high(f, f"{ALPHA}_breakout_acceptance_score", 0.70),
        ("future_return_3h", "future_return_6h", "future_return_24h", "future_max_upside_24h", "breakout_up_24h", "breakout_up_72h"),
        "Alpha acceptance state must beat random/shuffled controls across monthly windows.",
    ),
    Experiment(
        "alpha_breakout_failure",
        "Objective List Alpha: test failed breakout/rejection state.",
        "Price rejection at resistance should reduce future upside and increase drawdown risk.",
        lambda f: high(f, f"{ALPHA}_breakout_failure_score", 0.70),
        ("future_return_3h", "future_return_6h", "future_return_24h", "future_max_upside_24h", "future_max_drawdown_24h", "breakout_up_72h"),
        "Alpha failure state must beat random/shuffled controls across monthly windows.",
    ),
    Experiment(
        "alpha_breakdown_acceptance",
        "Objective List Alpha: test accepted breakdown state.",
        "Price accepted below support with resistance rebuild should precede downside path expansion.",
        lambda f: high(f, f"{ALPHA}_breakdown_acceptance_score", 0.70),
        ("future_return_3h", "future_return_6h", "future_return_24h", "future_max_drawdown_24h", "shock_down_24h", "shock_down_72h"),
        "Alpha breakdown acceptance must beat random/shuffled controls across monthly windows.",
    ),
    Experiment(
        "alpha_breakdown_failure",
        "Objective List Alpha: test failed breakdown/reclaim state.",
        "A support reclaim after a breakdown attempt should improve forward return or reduce drawdown.",
        lambda f: high(f, f"{ALPHA}_breakdown_failure_score", 0.70),
        ("future_return_3h", "future_return_6h", "future_return_24h", "future_max_upside_24h", "future_max_drawdown_24h"),
        "Alpha breakdown failure state must beat random/shuffled controls across monthly windows.",
    ),
    Experiment(
        "alpha_ask_absorption",
        "Objective List Alpha: test ask absorption.",
        "Repeated pressure into a stable ask zone without acceptance should stall or reject upside.",
        lambda f: high(f, f"{ALPHA}_ask_absorption_score", 0.70) | high(f, f"{ALPHA}_ask_absorption_attempt_count_6h", 0.80),
        ("future_return_3h", "future_return_6h", "future_max_upside_24h", "future_max_drawdown_24h", "breakout_up_24h"),
        "Absorption must survive monthly controls before being treated as a signal.",
    ),
    Experiment(
        "alpha_bid_absorption",
        "Objective List Alpha: test bid absorption.",
        "Repeated pressure into a stable bid zone without downside acceptance should stall downside or bounce.",
        lambda f: high(f, f"{ALPHA}_bid_absorption_score", 0.70) | high(f, f"{ALPHA}_bid_absorption_attempt_count_6h", 0.80),
        ("future_return_3h", "future_return_6h", "future_max_upside_24h", "future_max_drawdown_24h", "shock_down_24h"),
        "Absorption must survive monthly controls before being treated as a signal.",
    ),
    Experiment(
        "alpha_upside_vacuum_after_resistance_removed",
        "Objective List Alpha: test upside liquidity vacuum after resistance removal.",
        "Removed resistance plus thin ask-side liquidity should expand the upside path.",
        lambda f: high(f, f"{ALPHA}_upside_vacuum_after_resistance_removed", 0.70),
        ("future_return_3h", "future_return_6h", "future_max_upside_24h", "future_max_upside_72h", "breakout_up_24h"),
        "Directional vacuum must beat random/shuffled controls.",
    ),
    Experiment(
        "alpha_downside_vacuum_after_support_removed",
        "Objective List Alpha: test downside liquidity vacuum after support removal.",
        "Removed support plus thin bid-side liquidity should expand the downside path.",
        lambda f: high(f, f"{ALPHA}_downside_vacuum_after_support_removed", 0.70),
        ("future_return_3h", "future_return_6h", "future_max_drawdown_24h", "future_max_drawdown_72h", "shock_down_24h"),
        "Directional vacuum must beat random/shuffled controls.",
    ),
    Experiment(
        "alpha_pressure_agreement_continuation",
        "Objective List Alpha: test pressure agreement continuation.",
        "Book pressure agreeing with price direction should improve continuation odds.",
        lambda f: high(f, f"{ALPHA}_pressure_price_agreement", 0.70) & (col(f, f"{ALPHA}_pressure_duration_hours") >= 2),
        ("future_return_3h", "future_return_6h", "future_max_upside_24h", "future_max_drawdown_24h"),
        "Pressure agreement must produce stable continuation effects.",
    ),
    Experiment(
        "alpha_pressure_divergence_exhaustion",
        "Objective List Alpha: test pressure divergence/exhaustion.",
        "Strong pressure without price progress, or against price, should warn of reversal or failed continuation.",
        lambda f: high(f, f"{ALPHA}_pressure_divergence", 0.70),
        ("future_return_3h", "future_return_6h", "future_return_24h", "future_max_upside_24h", "future_max_drawdown_24h"),
        "Divergence must survive monthly controls before being treated as reversal evidence.",
    ),
    Experiment(
        "trader_state_breakout_acceptance",
        "Trader-state raw L2: test accepted breakout with support rebuild.",
        "A real breakout should show price accepted above resistance, bid support rebuilt, and upside liquidity remained open.",
        lambda f: high(f, "obts_breakout_acceptance_score", 0.70),
        ("breakout_success_next_6h", "hit_plus_3pct_before_minus_2pct", "future_return_6h", "future_max_upside_24h"),
        "Raw L2 acceptance must beat random/shuffled controls on event/path labels.",
    ),
    Experiment(
        "trader_state_breakout_failure",
        "Trader-state raw L2: test failed breakout/rejection.",
        "A failed breakout should show resistance touch/rejection, ask absorption, or pressure divergence before weaker upside or fakeout.",
        lambda f: high(f, "obts_breakout_failure_score", 0.70) | high(f, "obts_ask_absorption_score", 0.75),
        ("breakout_failure_next_6h", "failed_breakout_next_24h", "fakeout_next_24h", "future_max_drawdown_24h"),
        "Failure state must be stable against random/shuffled controls.",
    ),
    Experiment(
        "trader_state_pre_breakout_failure_risk",
        "Trader-state raw L2: test pre-rejection breakout failure risk.",
        "Price pressing stable resistance with absorption/divergence but without acceptance or rejection yet should precede failure/fakeout more than random hours.",
        lambda f: high(f, "obts_pre_breakout_failure_risk_score", 0.70),
        ("breakout_failure_next_6h", "failed_breakout_next_24h", "fakeout_next_24h", "future_max_drawdown_24h"),
        "Pre-failure state must beat random/shuffled controls before being treated as forward risk.",
    ),
    Experiment(
        "trader_state_resistance_rejection_resolved",
        "Trader-state raw L2: test resolved resistance rejection.",
        "If the prior inverted winner was really a resolved rejection state, it should reduce new future fakeout/failure labels or mark post-event chop.",
        lambda f: high(f, "obts_resistance_rejection_resolved_score", 0.70),
        ("breakout_failure_next_6h", "failed_breakout_next_24h", "fakeout_next_24h", "future_return_6h", "future_max_upside_24h"),
        "Resolved-state effect must be consistent and weaker under shuffled labels.",
    ),
    Experiment(
        "trader_state_resistance_cleared",
        "Trader-state raw L2: test resistance cleared and support rebuilt.",
        "Accepted price above prior resistance with support rebuild or removed resistance should precede upside continuation.",
        lambda f: high(f, "obts_resistance_cleared_score", 0.70),
        ("breakout_success_next_6h", "hit_plus_3pct_before_minus_2pct", "future_return_6h", "future_max_upside_24h"),
        "Cleared-resistance state must improve upside/event labels versus controls.",
    ),
    Experiment(
        "trader_state_post_failure_chop",
        "Trader-state raw L2: test post-failure chop after rejection.",
        "After rejection resolves into compression and price stalls, forward path should be lower-energy or less directional.",
        lambda f: high(f, "obts_post_failure_chop_score", 0.70),
        ("future_return_6h", "future_max_upside_24h", "future_max_drawdown_24h", "fakeout_next_24h"),
        "Post-failure chop must show stable path compression or fakeout-risk behaviour.",
    ),
    Experiment(
        "trader_state_ask_absorption_before_reaction",
        "Trader-state raw L2: test ask absorption before price reaction.",
        "Buy pressure into stable resistance while price stalls, before acceptance/rejection, should warn of absorption and weaker upside.",
        lambda f: high(f, "obts_ask_absorption_before_reaction_score", 0.70),
        ("breakout_failure_next_6h", "future_return_6h", "future_max_upside_24h", "future_max_drawdown_24h"),
        "Absorption-before-reaction must beat random/shuffled controls.",
    ),
    Experiment(
        "trader_state_breakdown_acceptance",
        "Trader-state raw L2: test accepted breakdown with resistance rebuild.",
        "A real breakdown should show price accepted below support, ask resistance rebuilt, and downside liquidity remained open.",
        lambda f: high(f, "obts_breakdown_acceptance_score", 0.70),
        ("breakdown_success_next_6h", "hit_minus_3pct_before_plus_2pct", "future_return_6h", "future_max_drawdown_24h"),
        "Raw L2 breakdown acceptance must beat random/shuffled controls.",
    ),
    Experiment(
        "trader_state_breakdown_failure",
        "Trader-state raw L2: test failed breakdown/reclaim.",
        "A failed breakdown should show support touch/bounce, bid absorption, or pressure divergence before weaker downside or reclaim.",
        lambda f: high(f, "obts_breakdown_failure_score", 0.70) | high(f, "obts_bid_absorption_score", 0.75),
        ("breakdown_failure_next_6h", "failed_breakdown_next_24h", "fakeout_next_24h", "future_max_upside_24h"),
        "Failure/reclaim state must be stable against random/shuffled controls.",
    ),
    Experiment(
        "trader_state_pre_breakdown_failure_risk",
        "Trader-state raw L2: test pre-bounce breakdown failure risk.",
        "Price pressing stable support with sell pressure/absorption but without breakdown acceptance or bounce yet should precede reclaim/failure more than random hours.",
        lambda f: high(f, "obts_pre_breakdown_failure_risk_score", 0.70),
        ("breakdown_failure_next_6h", "failed_breakdown_next_24h", "fakeout_next_24h", "future_max_upside_24h"),
        "Pre-breakdown-failure state must beat random/shuffled controls before being treated as forward reclaim risk.",
    ),
    Experiment(
        "trader_state_support_bounce_resolved",
        "Trader-state raw L2: test resolved support bounce.",
        "If support failure has already resolved into a bounce, it should reduce new future breakdown failure labels or mark post-event chop.",
        lambda f: high(f, "obts_support_bounce_resolved_score", 0.70),
        ("breakdown_failure_next_6h", "failed_breakdown_next_24h", "fakeout_next_24h", "future_return_6h", "future_max_drawdown_24h"),
        "Resolved support-bounce effect must be consistent and weaker under shuffled labels.",
    ),
    Experiment(
        "trader_state_support_cleared",
        "Trader-state raw L2: test support cleared and resistance rebuilt.",
        "Accepted price below prior support with resistance rebuild or removed support should precede downside continuation.",
        lambda f: high(f, "obts_support_cleared_score", 0.70),
        ("breakdown_success_next_6h", "hit_minus_3pct_before_plus_2pct", "future_return_6h", "future_max_drawdown_24h"),
        "Cleared-support state must improve downside/event labels versus controls.",
    ),
    Experiment(
        "trader_state_bid_absorption_before_reaction",
        "Trader-state raw L2: test bid absorption before price reaction.",
        "Sell pressure into stable support while price stalls, before breakdown acceptance/bounce, should warn of absorption and weaker downside.",
        lambda f: high(f, "obts_bid_absorption_before_reaction_score", 0.70),
        ("breakdown_failure_next_6h", "future_return_6h", "future_max_upside_24h", "future_max_drawdown_24h"),
        "Bid absorption-before-reaction must beat random/shuffled controls.",
    ),
    Experiment(
        "trader_state_resistance_compression_rejection",
        "Trader-state raw L2: persistent resistance during compression.",
        "Persistent nearby resistance plus compressed zones should cap upside or increase breakout failure odds.",
        lambda f: high(f, "obts_zone_compression_score", 0.65) & high(f, "obts_resistance_zone_stability_score", 0.65) & (col(f, "obts_range_position_24h") >= 0.60),
        ("breakout_failure_next_6h", "failed_breakout_next_24h", "future_max_upside_24h", "future_return_24h"),
        "Compression/resistance must survive path-label controls.",
    ),
    Experiment(
        "trader_state_support_removed_vacuum",
        "Trader-state raw L2: support removed into downside vacuum.",
        "Support removal plus thin bid-side liquidity should increase drawdown or breakdown odds.",
        lambda f: high(f, "obts_downside_vacuum_after_support_removed", 0.70),
        ("breakdown_success_next_6h", "large_drawdown_next_6h", "large_drawdown_next_24h", "future_max_drawdown_24h"),
        "Downside vacuum must beat random/shuffled controls.",
    ),
    Experiment(
        "trader_state_resistance_removed_vacuum",
        "Trader-state raw L2: resistance removed into upside vacuum.",
        "Resistance removal plus thin ask-side liquidity should improve upside path odds.",
        lambda f: high(f, "obts_upside_vacuum_after_resistance_removed", 0.70),
        ("breakout_success_next_6h", "hit_plus_3pct_before_minus_2pct", "future_max_upside_24h", "future_return_24h"),
        "Upside vacuum must beat random/shuffled controls.",
    ),
    Experiment(
        "trader_state_pressure_divergence",
        "Trader-state raw L2: pressure divergence/exhaustion.",
        "Strong pressure against price movement should identify exhaustion, reversal, or failed continuation.",
        lambda f: high(f, "obts_pressure_divergence", 0.70) | high(f, "obts_absorption_exhaustion_score", 0.70),
        ("future_return_3h", "future_return_6h", "future_max_upside_24h", "future_max_drawdown_24h", "fakeout_next_24h"),
        "Divergence/exhaustion must survive random/shuffled controls.",
    ),
)


if __name__ == "__main__":
    raise SystemExit(main())

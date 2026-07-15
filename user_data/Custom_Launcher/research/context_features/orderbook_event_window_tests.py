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
    bootstrap_delta,
    candidate_summary,
    effect_delta,
    random_bucket_baseline,
    shuffled_label_delta,
)
from orderbook_state_experiments import load_frame


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_FEATURES = USER_DATA_DIR / "orderbook_data" / "historical_bybit" / "features" / "orderbook_trader_state_1h_latest.parquet"
DEFAULT_OHLCV = USER_DATA_DIR / "data" / "binance" / "BTC_USDT-1h.feather"
DEFAULT_OUTPUT_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"


@dataclass(frozen=True)
class EventWindow:
    window_id: str
    objective: str
    base_builder: Callable[[pd.DataFrame], pd.Series]
    candidates: tuple[tuple[str, str, float], ...]
    targets: tuple[str, ...]
    pass_rule: str


RESULT_COLUMNS = (
    "window",
    "event_window_id",
    "objective",
    "candidate_id",
    "feature",
    "tail",
    "target",
    "base_n",
    "selected_n",
    "rest_n",
    "base_event_rate",
    "selected_event_rate",
    "rest_event_rate",
    "effect_delta",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "random_abs_mean",
    "random_abs_p95",
    "shuffled_delta",
    "effect_score",
    "pass_rule",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Event-window orderbook tests around support/resistance touches.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--ohlcv", type=Path, default=DEFAULT_OHLCV)
    parser.add_argument("--pair", default="BTC/USDT")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--snapshot-confirmed", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--random-controls", type=int, default=200)
    parser.add_argument("--min-pass-windows", type=int, default=4)
    parser.add_argument("--tag", default="event_windows_v3")
    args = parser.parse_args()

    frame = load_frame(args.features, args.ohlcv, args.pair)
    windows = monthly_windows(frame)
    plan = {
        "mode": "setup-only unless --execute and --snapshot-confirmed are supplied",
        "features_path": str(args.features),
        "ohlcv_path": str(args.ohlcv),
        "pair": args.pair,
        "windows": windows,
        "event_windows": [
            {
                "window_id": event.window_id,
                "objective": event.objective,
                "candidates": [candidate[0] for candidate in event.candidates],
                "targets": event.targets,
                "pass_rule": event.pass_rule,
            }
            for event in EVENT_WINDOWS
        ],
        "candidate_rule": (
            f"candidate_pass requires at least {args.min_pass_windows} windows, same sign in at least 70%, "
            "random-control beat in at least 50%, shuffled labels weaker in at least 60%, and bootstrap support in at least 3 windows."
        ),
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.snapshot_confirmed:
        parser.error("--execute requires --snapshot-confirmed after confirming parquet/feather snapshots are current.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = run_event_windows(frame, windows, args.bootstrap_iterations, args.random_controls)
    candidates = event_candidate_summary(results, min_pass_windows=args.min_pass_windows)
    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag.strip()) or "latest"
    report_path = args.output_dir / f"orderbook_event_window_tests_{safe_tag}_report.csv"
    candidate_path = args.output_dir / f"orderbook_event_window_tests_{safe_tag}_candidates.csv"
    summary_path = args.output_dir / f"orderbook_event_window_tests_{safe_tag}_summary.json"
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


def monthly_windows(frame: pd.DataFrame) -> list[dict[str, str]]:
    dated = frame.dropna(subset=["date"]).copy()
    dated["month"] = dated["date"].dt.strftime("%Y-%m")
    windows = []
    for month, group in dated.groupby("month", sort=True):
        coverage_rows = int(num(group, "feature_present").fillna(1.0).gt(0.0).sum())
        if coverage_rows < 300:
            continue
        windows.append(
            {
                "name": month,
                "start": group["date"].min().strftime("%Y-%m-%d"),
                "end": (group["date"].max() + pd.Timedelta(hours=1)).strftime("%Y-%m-%d"),
            }
        )
    return windows


def run_event_windows(
    frame: pd.DataFrame,
    windows: list[dict[str, str]],
    bootstrap_iterations: int,
    random_controls: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for window in windows:
        start = pd.Timestamp(window["start"], tz="UTC")
        end = pd.Timestamp(window["end"], tz="UTC")
        chunk = frame[(frame["date"] >= start) & (frame["date"] < end)].copy()
        feature_present = num(chunk, "feature_present").fillna(1.0) > 0.0
        for event in EVENT_WINDOWS:
            base = event.base_builder(chunk).fillna(False).astype(bool) & feature_present
            if int(base.sum()) < 24:
                continue
            event_chunk = with_context_labels(chunk, base)
            for candidate_id, feature, quantile in event.candidates:
                values = num(event_chunk, feature)
                threshold = values[base].quantile(quantile)
                if pd.isna(threshold):
                    continue
                selected = base & (values >= threshold)
                rest = base & ~selected
                if int(selected.sum()) < 8 or int(rest.sum()) < 8:
                    continue
                for target in event.targets:
                    if target not in event_chunk:
                        continue
                    rows.append(
                        evaluate_candidate(
                            event_chunk,
                            window["name"],
                            event,
                            candidate_id,
                            feature,
                            selected,
                            rest,
                            target,
                            bootstrap_iterations,
                            random_controls,
                        )
                    )
    if not rows:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.DataFrame(rows, columns=RESULT_COLUMNS).sort_values(["event_window_id", "candidate_id", "target", "window"])


def with_context_labels(frame: pd.DataFrame, base: pd.Series) -> pd.DataFrame:
    out = frame.copy()
    add_context_label(out, base, "future_return_6h", "context_top_return_6h", 0.70, "high")
    add_context_label(out, base, "future_return_6h", "context_worst_return_6h", 0.30, "low")
    add_context_label(out, base, "future_max_upside_24h", "context_top_upside_24h", 0.70, "high")
    add_context_label(out, base, "future_max_drawdown_24h", "context_worst_drawdown_24h", 0.30, "low")
    return out


def add_context_label(
    frame: pd.DataFrame,
    base: pd.Series,
    source_column: str,
    target_column: str,
    quantile: float,
    side: str,
) -> None:
    source = num(frame, source_column)
    threshold = source[base & source.notna()].quantile(quantile)
    if pd.isna(threshold):
        frame[target_column] = np.nan
    elif side == "low":
        frame[target_column] = (source <= threshold).astype(float).where(source.notna())
    else:
        frame[target_column] = (source >= threshold).astype(float).where(source.notna())


def evaluate_candidate(
    frame: pd.DataFrame,
    window: str,
    event: EventWindow,
    candidate_id: str,
    feature: str,
    selected: pd.Series,
    rest: pd.Series,
    target: str,
    bootstrap_iterations: int,
    random_controls: int,
) -> dict[str, object]:
    source = frame[target]
    observed = effect_delta(source, selected, rest)
    ci_low, ci_high = bootstrap_delta(source, selected, rest, bootstrap_iterations)
    random_mean, random_p95 = random_bucket_baseline(source, int(selected.sum()), random_controls)
    shuffled = shuffled_label_delta(source, selected, rest)
    return {
        "window": window,
        "event_window_id": event.window_id,
        "objective": event.objective,
        "candidate_id": candidate_id,
        "feature": feature,
        "tail": "high",
        "target": target,
        "base_n": int((selected | rest).sum()),
        "selected_n": int(selected.sum()),
        "rest_n": int(rest.sum()),
        "base_event_rate": float(source[selected | rest].mean()),
        "selected_event_rate": float(source[selected].mean()),
        "rest_event_rate": float(source[rest].mean()),
        "effect_delta": observed,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "random_abs_mean": random_mean,
        "random_abs_p95": random_p95,
        "shuffled_delta": shuffled,
        "effect_score": abs(observed) - random_p95,
        "pass_rule": event.pass_rule,
    }


def event_candidate_summary(results: pd.DataFrame, *, min_pass_windows: int) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=[*CANDIDATE_COLUMNS, "event_window_id"])
    compatible = results.rename(columns={"candidate_id": "test_id"})
    summary = candidate_summary(compatible)
    if summary.empty:
        return summary
    event_lookup = results.groupby(["candidate_id", "target"])["event_window_id"].agg(lambda s: ",".join(sorted(set(s)))).to_dict()
    rows = []
    for _, item in summary.iterrows():
        row = item.to_dict()
        windows = int(row["windows_tested"])
        same_sign = int(row["same_sign_windows"])
        beats_random = int(row["beats_random_windows"])
        bootstrap = int(row["bootstrap_excludes_zero_windows"])
        shuffled = int(row["shuffled_weaker_windows"])
        row["event_window_id"] = event_lookup.get((row["test_id"], row["target"]), "")
        row["candidate_pass"] = bool(
            windows >= min_pass_windows
            and same_sign >= min(windows, max(3, int(np.ceil(windows * 0.70))))
            and beats_random >= min(windows, max(3, int(np.ceil(windows * 0.50))))
            and shuffled >= min(windows, max(3, int(np.ceil(windows * 0.60))))
            and bootstrap >= min(windows, 3)
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["candidate_pass", "mean_abs_effect"], ascending=[False, False])


def num(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(index=frame.index, dtype=float)), errors="coerce")


def high(frame: pd.DataFrame, column: str, quantile: float = 0.70) -> pd.Series:
    series = num(frame, column)
    return series >= series.quantile(quantile)


ZONE_DISTANCE_BPS = 20.0
ZONE_STABILITY_MIN = 0.25
ZONE_STRENGTH_MIN = 0.45
RESISTANCE_RANGE_POSITION_MIN = 0.60
SUPPORT_RANGE_POSITION_MAX = 0.40


def near_resistance(frame: pd.DataFrame) -> pd.Series:
    return (
        (num(frame, "obts_resistance_zone_distance_bps") <= ZONE_DISTANCE_BPS)
        & (num(frame, "obts_resistance_zone_stability_score") >= ZONE_STABILITY_MIN)
        & (num(frame, "obts_resistance_zone_strength") >= ZONE_STRENGTH_MIN)
        & (num(frame, "obts_range_position_24h") >= RESISTANCE_RANGE_POSITION_MIN)
    )


def near_support(frame: pd.DataFrame) -> pd.Series:
    return (
        (num(frame, "obts_support_zone_distance_bps") <= ZONE_DISTANCE_BPS)
        & (num(frame, "obts_support_zone_stability_score") >= ZONE_STABILITY_MIN)
        & (num(frame, "obts_support_zone_strength") >= ZONE_STRENGTH_MIN)
        & (num(frame, "obts_range_position_24h") <= SUPPORT_RANGE_POSITION_MAX)
    )


EVENT_WINDOWS = (
    EventWindow(
        "resistance_touch_failure",
        "Within resistance-touch hours, test which states separate failure/rejection from successful breakout.",
        lambda frame: near_resistance(frame),
        (
            ("pre_failure_risk", "obts_pre_breakout_failure_risk_score", 0.70),
            ("resolved_rejection", "obts_resistance_rejection_resolved_score", 0.70),
            ("ask_absorption_before_reaction", "obts_ask_absorption_before_reaction_score", 0.70),
            ("zone_compression", "obts_zone_compression_score", 0.70),
            ("pressure_divergence", "obts_pressure_divergence", 0.70),
        ),
        ("breakout_failure_next_6h", "failed_breakout_next_24h", "fakeout_next_24h", "context_worst_return_6h", "context_worst_drawdown_24h"),
        "Candidate should increase failure/fakeout or drawdown labels inside resistance-touch windows and survive controls.",
    ),
    EventWindow(
        "resistance_touch_confirmation",
        "Within resistance-touch hours, test breakout confirmation states.",
        lambda frame: near_resistance(frame),
        (
            ("resistance_cleared", "obts_resistance_cleared_score", 0.70),
            ("breakout_acceptance", "obts_breakout_acceptance_score", 0.70),
            ("upside_vacuum", "obts_upside_vacuum_after_resistance_removed", 0.70),
            ("pressure_agreement", "obts_pressure_price_agreement", 0.70),
        ),
        ("breakout_success_next_6h", "hit_plus_3pct_before_minus_2pct", "context_top_return_6h", "context_top_upside_24h"),
        "Candidate should improve success/upside labels inside resistance-touch windows and survive controls.",
    ),
    EventWindow(
        "support_touch_failure",
        "Within support-touch hours, test failed-breakdown/bounce states.",
        lambda frame: near_support(frame),
        (
            ("pre_breakdown_failure_risk", "obts_pre_breakdown_failure_risk_score", 0.70),
            ("support_bounce_resolved", "obts_support_bounce_resolved_score", 0.70),
            ("bid_absorption_before_reaction", "obts_bid_absorption_before_reaction_score", 0.70),
            ("zone_compression", "obts_zone_compression_score", 0.70),
            ("pressure_divergence", "obts_pressure_divergence", 0.70),
        ),
        ("breakdown_failure_next_6h", "failed_breakdown_next_24h", "fakeout_next_24h", "context_top_return_6h", "context_top_upside_24h"),
        "Candidate should increase breakdown-failure/reclaim labels inside support-touch windows and survive controls.",
    ),
    EventWindow(
        "support_touch_crash_detection",
        "Within support-touch hours, test breakdown/crash confirmation states.",
        lambda frame: near_support(frame),
        (
            ("support_cleared", "obts_support_cleared_score", 0.70),
            ("breakdown_acceptance", "obts_breakdown_acceptance_score", 0.70),
            ("downside_vacuum", "obts_downside_vacuum_after_support_removed", 0.70),
            ("support_removed", "obts_support_zone_removed_strength_1h", 0.70),
        ),
        ("breakdown_success_next_6h", "large_drawdown_next_6h", "large_drawdown_next_24h", "context_worst_return_6h", "context_worst_drawdown_24h"),
        "Candidate should improve breakdown/crash labels inside support-touch windows and survive controls.",
    ),
)


if __name__ == "__main__":
    raise SystemExit(main())

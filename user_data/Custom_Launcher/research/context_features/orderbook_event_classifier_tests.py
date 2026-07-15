from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline

from orderbook_event_window_tests import (
    DEFAULT_FEATURES,
    DEFAULT_OHLCV,
    DEFAULT_OUTPUT_DIR,
    EVENT_WINDOWS,
    monthly_windows,
    num,
    with_context_labels,
)
from orderbook_state_experiments import load_frame


PRICE_FEATURES = (
    "px_return_1h",
    "px_return_3h",
    "px_return_6h",
    "px_range_1h",
    "px_range_position_24h",
    "px_volume_z_24h",
    "px_volatility_24h",
)

ORDERBOOK_FEATURES = (
    "obts_zone_compression_score",
    "obts_resistance_zone_distance_bps",
    "obts_support_zone_distance_bps",
    "obts_resistance_zone_strength",
    "obts_support_zone_strength",
    "obts_resistance_zone_stability_score",
    "obts_support_zone_stability_score",
    "obts_resistance_zone_removed_strength_1h",
    "obts_support_zone_removed_strength_1h",
    "obts_resistance_zone_added_strength_1h",
    "obts_support_zone_added_strength_1h",
    "obts_pressure_direction",
    "obts_pressure_duration_hours",
    "obts_pressure_price_agreement",
    "obts_pressure_divergence",
    "obts_upside_liquidity_vacuum_near",
    "obts_downside_liquidity_vacuum_near",
    "obts_breakout_failure_score",
    "obts_breakdown_failure_score",
    "obts_ask_absorption_score",
    "obts_bid_absorption_score",
    "obts_pre_breakout_failure_risk_score",
    "obts_resistance_rejection_resolved_score",
    "obts_resistance_cleared_score",
    "obts_post_failure_chop_score",
    "obts_ask_absorption_before_reaction_score",
    "obts_pre_breakdown_failure_risk_score",
    "obts_support_bounce_resolved_score",
    "obts_support_cleared_score",
    "obts_post_breakdown_chop_score",
    "obts_bid_absorption_before_reaction_score",
    "obts_range_position_24h",
    "obts_range_position_72h",
    "obts_price_trend_state_24h",
    "obts_volatility_expansion_state",
)


RESULT_COLUMNS = (
    "event_window_id",
    "target",
    "holdout_window",
    "feature_set",
    "train_rows",
    "test_rows",
    "test_event_rate",
    "roc_auc",
    "average_precision",
    "balanced_accuracy",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Leave-one-month event classifier tests for orderbook trader-state features.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--ohlcv", type=Path, default=DEFAULT_OHLCV)
    parser.add_argument("--pair", default="BTC/USDT")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--snapshot-confirmed", action="store_true")
    parser.add_argument("--tag", default="event_classifier_v1")
    args = parser.parse_args()

    frame = add_price_features(load_frame(args.features, args.ohlcv, args.pair))
    windows = monthly_windows(frame)
    plan = {
        "mode": "setup-only unless --execute and --snapshot-confirmed are supplied",
        "objective": "Test whether compact orderbook behaviour states beat price-only controls inside support/resistance event windows.",
        "features_path": str(args.features),
        "ohlcv_path": str(args.ohlcv),
        "windows": windows,
        "feature_sets": {
            "price_only": list(PRICE_FEATURES),
            "price_plus_orderbook": [*PRICE_FEATURES, *ORDERBOOK_FEATURES],
        },
        "targets": {event.window_id: list(event.targets) for event in EVENT_WINDOWS},
        "pass_rule": (
            "Strict pass requires positive price_plus_orderbook ROC-AUC delta in at least 3 holdout windows, "
            "mean delta >= 0.03, and mean orderbook ROC-AUC >= 0.55. Watchlist requires the same delta rule with mean AUC >= 0.53."
        ),
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.snapshot_confirmed:
        parser.error("--execute requires --snapshot-confirmed after confirming parquet/feather snapshots are current.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = run_tests(frame, windows)
    summary = summarize_results(results)
    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag.strip()) or "latest"
    report_path = args.output_dir / f"orderbook_event_classifier_tests_{safe_tag}_report.csv"
    summary_path = args.output_dir / f"orderbook_event_classifier_tests_{safe_tag}_summary.csv"
    meta_path = args.output_dir / f"orderbook_event_classifier_tests_{safe_tag}_meta.json"
    results.to_csv(report_path, index=False)
    summary.to_csv(summary_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "plan": plan,
                "report_path": str(report_path),
                "summary_path": str(summary_path),
                "rows": int(len(results)),
                "summary_rows": int(len(summary)),
                "strict_passes": int(summary["strict_pass"].sum()) if not summary.empty else 0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report_path": str(report_path), "summary_path": str(summary_path), "meta_path": str(meta_path)}, indent=2))
    return 0


def add_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    close = num(out, "close")
    high = num(out, "high")
    low = num(out, "low")
    volume = num(out, "volume")
    out["px_return_1h"] = close.pct_change(1)
    out["px_return_3h"] = close.pct_change(3)
    out["px_return_6h"] = close.pct_change(6)
    out["px_range_1h"] = (high - low) / close.replace(0.0, np.nan)
    low_24h = low.rolling(24, min_periods=12).min()
    high_24h = high.rolling(24, min_periods=12).max()
    out["px_range_position_24h"] = (close - low_24h) / (high_24h - low_24h).replace(0.0, np.nan)
    out["px_volume_z_24h"] = (volume - volume.rolling(24, min_periods=12).mean()) / volume.rolling(24, min_periods=12).std().replace(0.0, np.nan)
    out["px_volatility_24h"] = out["px_return_1h"].rolling(24, min_periods=12).std()
    return out


def run_tests(frame: pd.DataFrame, windows: list[dict[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    feature_present = num(frame, "feature_present").fillna(1.0) > 0.0
    for event in EVENT_WINDOWS:
        event_frame = frame.copy()
        event_frame["_month"] = event_frame["date"].dt.strftime("%Y-%m")
        base = event.base_builder(event_frame).fillna(False).astype(bool) & feature_present
        event_frame = add_month_context_labels(event_frame, windows, event, base)
        event_frame = event_frame[base].copy()
        for target in event.targets:
            if target not in event_frame:
                continue
            for holdout in windows:
                train = event_frame[event_frame["_month"] != holdout["name"]]
                test = event_frame[event_frame["_month"] == holdout["name"]]
                for feature_set, columns in {
                    "price_only": PRICE_FEATURES,
                    "price_plus_orderbook": [*PRICE_FEATURES, *ORDERBOOK_FEATURES],
                }.items():
                    metric = fit_score(train, test, target, [column for column in columns if column in event_frame])
                    if metric:
                        rows.append({"event_window_id": event.window_id, "target": target, "holdout_window": holdout["name"], "feature_set": feature_set, **metric})
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def add_month_context_labels(frame: pd.DataFrame, windows: list[dict[str, str]], event, base: pd.Series) -> pd.DataFrame:
    out = frame.copy()
    for window in windows:
        start = pd.Timestamp(window["start"], tz="UTC")
        end = pd.Timestamp(window["end"], tz="UTC")
        mask = (out["date"] >= start) & (out["date"] < end)
        labelled = with_context_labels(out.loc[mask].copy(), base.loc[mask])
        for column in labelled.columns:
            if column.startswith("context_"):
                out.loc[mask, column] = labelled[column]
    return out


def fit_score(train: pd.DataFrame, test: pd.DataFrame, target: str, columns: list[str]) -> dict[str, object] | None:
    train_y = pd.to_numeric(train[target], errors="coerce")
    test_y = pd.to_numeric(test[target], errors="coerce")
    train_mask = train_y.notna()
    test_mask = test_y.notna()
    if int(train_mask.sum()) < 80 or int(test_mask.sum()) < 24:
        return None
    y_train = train_y[train_mask].astype(int)
    y_test = test_y[test_mask].astype(int)
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return None
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=120,
                    learning_rate=0.04,
                    max_leaf_nodes=7,
                    min_samples_leaf=12,
                    l2_regularization=0.2,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(train.loc[train_mask, columns], y_train)
    probabilities = model.predict_proba(test.loc[test_mask, columns])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "test_event_rate": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
    }


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    pivot = results.pivot_table(
        index=["event_window_id", "target", "holdout_window"],
        columns="feature_set",
        values="roc_auc",
        aggfunc="first",
    ).reset_index()
    if not {"price_only", "price_plus_orderbook"}.issubset(pivot.columns):
        return pd.DataFrame()
    pivot["auc_delta"] = pivot["price_plus_orderbook"] - pivot["price_only"]
    summary = (
        pivot.groupby(["event_window_id", "target"])
        .agg(
            windows=("auc_delta", "count"),
            positive_delta_windows=("auc_delta", lambda s: int((s > 0.0).sum())),
            mean_price_auc=("price_only", "mean"),
            mean_orderbook_auc=("price_plus_orderbook", "mean"),
            mean_auc_delta=("auc_delta", "mean"),
        )
        .reset_index()
    )
    summary["watchlist_pass"] = (
        (summary["windows"] >= 3)
        & (summary["positive_delta_windows"] >= 3)
        & (summary["mean_auc_delta"] >= 0.03)
        & (summary["mean_orderbook_auc"] >= 0.53)
    )
    summary["strict_pass"] = summary["watchlist_pass"] & (summary["mean_orderbook_auc"] >= 0.55)
    return summary.sort_values(["strict_pass", "watchlist_pass", "mean_auc_delta"], ascending=[False, False, False])


if __name__ == "__main__":
    raise SystemExit(main())

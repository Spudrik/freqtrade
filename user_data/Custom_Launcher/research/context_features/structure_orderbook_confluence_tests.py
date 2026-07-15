from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from user_data.Custom_Launcher.research.context_features.orderbook_event_classifier_tests import ORDERBOOK_FEATURES


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_STRUCTURAL = USER_DATA_DIR / "research_news_data" / "context_features" / "structural_cache" / "btc_structural_features_1h_latest.parquet"
DEFAULT_ORDERBOOK = USER_DATA_DIR / "orderbook_data" / "historical_bybit" / "features" / "orderbook_trader_state_1h_latest.parquet"
DEFAULT_OUTPUT_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"
MIN_ORDERBOOK_COVERAGE_RATIO = 0.50

PRICE_FEATURES = (
    "st_price_return_1h",
    "st_price_return_3h",
    "st_price_return_6h",
    "st_volume_pressure_6h",
    "st_volume_pressure_24h",
    "st_volume_z_24h",
    "st_range_position_24h",
    "st_range_position_72h",
)

LEVEL_FEATURES = tuple(
    column
    for tf in ("1h", "4h", "1d")
    for column in (
        f"st_{tf}_tlv2_resistance_score_rank0",
        f"st_{tf}_tlv2_resistance_distance_atr_rank0",
        f"st_{tf}_tlv2_support_score_rank0",
        f"st_{tf}_tlv2_support_distance_atr_rank0",
        f"st_{tf}_ms_structure_event",
        f"st_{tf}_ms_state",
        f"st_{tf}_ms_bos_to_bull",
        f"st_{tf}_ms_bos_to_bear",
        f"st_{tf}_ms_choch_to_bull",
        f"st_{tf}_ms_choch_to_bear",
    )
)

VP_FEATURES = tuple(
    column
    for tf in ("1h", "4h", "1d")
    for column in (
        f"st_{tf}_vp_score_long",
        f"st_{tf}_vp_score_short",
        f"st_{tf}_vp_score_abs",
        f"st_{tf}_vp_state",
        f"st_{tf}_vp_market_context",
        f"st_{tf}_vp_value_area_position",
        f"st_{tf}_vp_delta_ratio",
        f"st_{tf}_vp_poc_delta_ratio",
        f"st_{tf}_vp_vah_breakout_with_pressure",
        f"st_{tf}_vp_val_breakdown_with_pressure",
        f"st_{tf}_vp_upper_rejection_with_pressure",
        f"st_{tf}_vp_lower_rejection_with_pressure",
        f"st_{tf}_vp_hvn_below_reclaim",
        f"st_{tf}_vp_hvn_above_reject",
    )
)

SETUP_FEATURES = (
    "st_breakout_structure_setup",
    "st_breakdown_structure_setup",
    "st_failed_breakout_structure_risk",
    "st_failed_breakdown_structure_risk",
    "st_near_tlv2_resistance_1h",
    "st_near_tlv2_support_1h",
)

TARGETS_BY_EVENT = {
    "breakout_confirmation": ("breakout_success_next_6h",),
    "breakout_failure": ("breakout_failure_next_6h", "failed_breakout_next_24h", "fakeout_next_24h"),
    "crash_detection": ("breakdown_success_next_6h", "large_drawdown_next_6h", "large_drawdown_next_24h"),
    "support_fakeout": ("breakdown_failure_next_6h", "failed_breakdown_next_24h", "fakeout_next_24h"),
    "support_sweep_reclaim": ("breakdown_failure_next_6h", "failed_breakdown_next_24h", "fakeout_next_24h"),
    "val_reclaim": ("breakdown_failure_next_6h", "failed_breakdown_next_24h", "fakeout_next_24h"),
    "tlv2_support_reclaim": ("breakdown_failure_next_6h", "failed_breakdown_next_24h", "fakeout_next_24h"),
    "vp_lower_rejection": ("breakdown_failure_next_6h", "failed_breakdown_next_24h", "fakeout_next_24h"),
    "range_low_reclaim": ("breakdown_failure_next_6h", "failed_breakdown_next_24h", "fakeout_next_24h"),
    "tlv2_resistance_reject": ("breakout_failure_next_6h", "failed_breakout_next_24h", "fakeout_next_24h"),
    "vah_rejection": ("breakout_failure_next_6h", "failed_breakout_next_24h", "fakeout_next_24h"),
    "breakout_acceptance": ("breakout_success_next_6h",),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Structure + orderbook confluence tests using cached custom indicators.")
    parser.add_argument("--structural", type=Path, default=DEFAULT_STRUCTURAL)
    parser.add_argument("--orderbook", type=Path, default=DEFAULT_ORDERBOOK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--snapshot-confirmed", action="store_true")
    parser.add_argument("--tag", default="structure_orderbook_v1")
    args = parser.parse_args()

    plan = {
        "mode": "setup-only unless --execute and --snapshot-confirmed are supplied",
        "objective": "Test trader-style structure-first setups, then measure whether VP and orderbook add useful confirmation.",
        "structural_cache": str(args.structural),
        "orderbook_cache": str(args.orderbook),
        "feature_sets": ["price", "structure", "structure_vp", "structure_orderbook", "structure_vp_orderbook"],
        "events": TARGETS_BY_EVENT,
        "pass_rule": (
            "Strict pass requires structure_vp_orderbook mean AUC >= 0.55, positive delta over structure_vp in >=3 windows, "
            "and mean delta >= 0.03. FreqAI-ready allows mean AUC >= 0.55, positive delta in >=2 windows, and mean delta >= 0.02."
        ),
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.snapshot_confirmed:
        parser.error("--execute requires --snapshot-confirmed after confirming parquet snapshots are current.")

    frame = load_frame(args.structural, args.orderbook)
    results = run_tests(frame)
    summary = summarize_results(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag) or "latest"
    report_path = args.output_dir / f"structure_orderbook_confluence_{safe_tag}_report.csv"
    summary_path = args.output_dir / f"structure_orderbook_confluence_{safe_tag}_summary.csv"
    meta_path = args.output_dir / f"structure_orderbook_confluence_{safe_tag}_meta.json"
    results.to_csv(report_path, index=False)
    summary.to_csv(summary_path, index=False)
    meta_path.write_text(
        json.dumps({**plan, "rows": int(len(results)), "summary_rows": int(len(summary)), "strict_passes": int(summary["strict_pass"].sum()) if not summary.empty else 0}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(report_path), "summary": str(summary_path), "meta": str(meta_path)}, indent=2))
    return 0


def load_frame(structural_path: Path, orderbook_path: Path) -> DataFrame:
    structure = pd.read_parquet(structural_path)
    structure["date"] = pd.to_datetime(structure["date"], utc=True, errors="coerce")
    structure = structure.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    orderbook = pd.read_parquet(orderbook_path)
    orderbook["date"] = pd.to_datetime(orderbook["date"], utc=True, errors="coerce")
    orderbook = orderbook.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    diagnostic_columns = [
        column
        for column in ("obts_feature_present", "obts_coverage_ratio", "obts_sample_rows_1m")
        if column in orderbook.columns
    ]
    keep = ["date", *diagnostic_columns, *[column for column in ORDERBOOK_FEATURES if column in orderbook.columns]]
    frame = structure.merge(orderbook[keep], on="date", how="left", sort=False)
    if "obts_feature_present" in frame:
        base_present = pd.to_numeric(frame["obts_feature_present"], errors="coerce").fillna(0.0).gt(0.0)
    else:
        feature_columns = [column for column in ORDERBOOK_FEATURES if column in frame.columns]
        base_present = frame[feature_columns].notna().any(axis=1) if feature_columns else pd.Series(False, index=frame.index)
    if "obts_coverage_ratio" in frame:
        coverage_ok = pd.to_numeric(frame["obts_coverage_ratio"], errors="coerce").fillna(0.0).ge(MIN_ORDERBOOK_COVERAGE_RATIO)
    else:
        coverage_ok = pd.Series(True, index=frame.index)
    frame["orderbook_present"] = (base_present & coverage_ok).astype(float)
    frame = append_labels(frame)
    return frame


def append_labels(frame: DataFrame) -> DataFrame:
    out = frame.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    for horizon in (1, 3, 6, 24, 72):
        out[f"future_return_{horizon}h"] = close.shift(-horizon) / close - 1.0
        future_high = high.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).max().iloc[::-1]
        future_low = low.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).min().iloc[::-1]
        out[f"future_max_upside_{horizon}h"] = future_high / close - 1.0
        out[f"future_max_drawdown_{horizon}h"] = future_low / close - 1.0
    prior_high_24h = high.shift(1).rolling(24, min_periods=12).max()
    prior_low_24h = low.shift(1).rolling(24, min_periods=12).min()
    future_high_6h = high.shift(-1).iloc[::-1].rolling(6, min_periods=6).max().iloc[::-1]
    future_low_6h = low.shift(-1).iloc[::-1].rolling(6, min_periods=6).min().iloc[::-1]
    future_close_6h = close.shift(-6)
    breakout_attempt = future_high_6h.gt(prior_high_24h * 1.001)
    breakdown_attempt = future_low_6h.lt(prior_low_24h * 0.999)
    current_resistance_setup = prior_high_24h.notna() & (high.ge(prior_high_24h * 0.995) | close.ge(prior_high_24h * 0.990))
    current_support_setup = prior_low_24h.notna() & (low.le(prior_low_24h * 1.005) | close.le(prior_low_24h * 1.010))
    out["breakout_success_next_6h"] = (breakout_attempt & future_close_6h.gt(prior_high_24h * 1.001)).astype(float).where(prior_high_24h.notna())
    out["breakout_failure_next_6h"] = (breakout_attempt & future_close_6h.lt(prior_high_24h * 0.999)).astype(float).where(prior_high_24h.notna())
    out["breakdown_success_next_6h"] = (breakdown_attempt & future_close_6h.lt(prior_low_24h * 0.999)).astype(float).where(prior_low_24h.notna())
    out["breakdown_failure_next_6h"] = (breakdown_attempt & future_close_6h.gt(prior_low_24h * 1.001)).astype(float).where(prior_low_24h.notna())
    out["failed_breakout_next_24h"] = (out["future_max_upside_24h"].gt(0.01) & out["future_return_24h"].lt(0.0)).astype(float).where(out["future_return_24h"].notna())
    out["failed_breakdown_next_24h"] = (out["future_max_drawdown_24h"].lt(-0.01) & out["future_return_24h"].gt(0.0)).astype(float).where(out["future_return_24h"].notna())
    out["large_drawdown_next_6h"] = out["future_max_drawdown_6h"].le(-0.03).astype(float).where(out["future_max_drawdown_6h"].notna())
    out["large_drawdown_next_24h"] = out["future_max_drawdown_24h"].le(-0.04).astype(float).where(out["future_max_drawdown_24h"].notna())
    out["fakeout_next_24h"] = (out["failed_breakout_next_24h"].eq(1.0) | out["failed_breakdown_next_24h"].eq(1.0)).astype(float).where(out["future_return_24h"].notna())
    out["current_resistance_setup"] = current_resistance_setup.astype(float).where(prior_high_24h.notna())
    out["current_support_setup"] = current_support_setup.astype(float).where(prior_low_24h.notna())
    out["breakout_success_from_current_setup_6h"] = (
        current_resistance_setup & breakout_attempt & future_close_6h.gt(prior_high_24h * 1.001)
    ).astype(float).where(current_resistance_setup)
    out["breakout_failure_from_current_setup_6h"] = (
        current_resistance_setup & breakout_attempt & future_close_6h.lt(prior_high_24h * 0.999)
    ).astype(float).where(current_resistance_setup)
    out["breakdown_success_from_current_setup_6h"] = (
        current_support_setup & breakdown_attempt & future_close_6h.lt(prior_low_24h * 0.999)
    ).astype(float).where(current_support_setup)
    out["breakdown_failure_from_current_setup_6h"] = (
        current_support_setup & breakdown_attempt & future_close_6h.gt(prior_low_24h * 1.001)
    ).astype(float).where(current_support_setup)
    out["fakeout_from_current_setup_24h"] = (
        (current_resistance_setup & out["failed_breakout_next_24h"].eq(1.0))
        | (current_support_setup & out["failed_breakdown_next_24h"].eq(1.0))
    ).astype(float).where(current_resistance_setup | current_support_setup)
    return out


def run_tests(frame: DataFrame) -> DataFrame:
    rows: list[dict[str, object]] = []
    frame = frame[pd.to_numeric(frame.get("orderbook_present"), errors="coerce").fillna(0.0).gt(0.0)].copy()
    frame["_month"] = frame["date"].dt.strftime("%Y-%m")
    months = [month for month, count in frame["_month"].value_counts().sort_index().items() if count >= 300]
    for event_id, targets in TARGETS_BY_EVENT.items():
        event_frame = frame[event_mask(frame, event_id)].copy()
        if len(event_frame) < 100:
            continue
        for target in targets:
            if target not in event_frame:
                continue
            for holdout in months:
                train = event_frame[event_frame["_month"] != holdout]
                test = event_frame[event_frame["_month"] == holdout]
                for feature_set, columns in feature_sets().items():
                    metric = fit_score(train, test, target, existing_columns(event_frame, columns))
                    if metric:
                        rows.append({"event_id": event_id, "target": target, "holdout_window": holdout, "feature_set": feature_set, **metric})
    return pd.DataFrame(rows)


def event_mask(frame: DataFrame, event_id: str) -> pd.Series:
    if event_id in {"all", "all_rows", "none"}:
        return pd.Series(True, index=frame.index)
    vol_bull = num(frame, "st_volume_pressure_6h").gt(0.0)
    vol_bear = num(frame, "st_volume_pressure_6h").lt(0.0)
    range_low = num(frame, "st_range_position_24h").le(0.35)
    range_high = num(frame, "st_range_position_24h").ge(0.65)
    support_score = num(frame, "st_1h_tlv2_support_score_rank0").ge(0.50)
    resistance_score = num(frame, "st_1h_tlv2_resistance_score_rank0").ge(0.50)
    near_support = num(frame, "st_near_tlv2_support_1h").gt(0.0)
    near_resistance = num(frame, "st_near_tlv2_resistance_1h").gt(0.0)
    lower_rejection = bool_col(frame, "st_1h_vp_lower_rejection_with_pressure")
    upper_rejection = bool_col(frame, "st_1h_vp_upper_rejection_with_pressure")
    hvn_reclaim = bool_col(frame, "st_1h_vp_hvn_below_reclaim")
    hvn_reject = bool_col(frame, "st_1h_vp_hvn_above_reject")
    below_value = bool_col(frame, "st_1h_vp_below_value_area")
    above_value = bool_col(frame, "st_1h_vp_above_value_area")
    node_hold_long = bool_col(frame, "st_1h_vp_node_hold_long")
    node_hold_short = bool_col(frame, "st_1h_vp_node_hold_short")
    vp_long = num(frame, "st_1h_vp_score_long").ge(0.25)
    vp_short = num(frame, "st_1h_vp_score_short").ge(0.25)
    value_pos = num(frame, "st_1h_vp_value_area_position")
    val_breakdown = bool_col(frame, "st_1h_vp_val_breakdown_with_pressure")
    vah_breakout = bool_col(frame, "st_1h_vp_vah_breakout_with_pressure")
    ms_bull = bool_col(frame, "st_1h_ms_bos_to_bull") | bool_col(frame, "st_1h_ms_choch_to_bull")
    ms_bear = bool_col(frame, "st_1h_ms_bos_to_bear") | bool_col(frame, "st_1h_ms_choch_to_bear")
    if event_id == "breakout_confirmation":
        return pd.to_numeric(frame.get("st_breakout_structure_setup"), errors="coerce").fillna(0.0).ge(2.0)
    if event_id == "breakout_failure":
        return pd.to_numeric(frame.get("st_failed_breakout_structure_risk"), errors="coerce").fillna(0.0).ge(2.0)
    if event_id == "crash_detection":
        return pd.to_numeric(frame.get("st_breakdown_structure_setup"), errors="coerce").fillna(0.0).ge(2.0)
    if event_id == "support_fakeout":
        return pd.to_numeric(frame.get("st_failed_breakdown_structure_risk"), errors="coerce").fillna(0.0).ge(2.0)
    if event_id == "support_sweep_reclaim":
        return range_low & (below_value | node_hold_long | lower_rejection | hvn_reclaim | ms_bull) & (vol_bull | vp_long)
    if event_id == "val_reclaim":
        return (below_value | value_pos.le(0.15)) & vp_long & (vol_bull | hvn_reclaim | lower_rejection)
    if event_id == "tlv2_support_reclaim":
        return near_support & support_score & (vol_bull | vp_long)
    if event_id == "vp_lower_rejection":
        return (below_value | value_pos.le(0.15) | node_hold_long | lower_rejection | hvn_reclaim) & vp_long
    if event_id == "range_low_reclaim":
        return range_low & (vol_bull | vp_long) & (ms_bull | below_value | node_hold_long | lower_rejection | hvn_reclaim)
    if event_id == "tlv2_resistance_reject":
        return near_resistance & resistance_score & (vol_bear | vp_short)
    if event_id == "vah_rejection":
        return range_high & (above_value | value_pos.ge(0.85) | node_hold_short | upper_rejection | hvn_reject) & vp_short
    if event_id == "breakout_acceptance":
        return range_high & (above_value | vah_breakout | ms_bull | num(frame, "st_1h_vp_score_long").ge(0.35)) & vol_bull
    raise ValueError(f"Unknown event: {event_id}")


def feature_sets() -> dict[str, tuple[str, ...]]:
    structure = (*PRICE_FEATURES, *LEVEL_FEATURES, *SETUP_FEATURES)
    structure_vp = (*structure, *VP_FEATURES)
    orderbook = tuple(column for column in ORDERBOOK_FEATURES)
    return {
        "price": PRICE_FEATURES,
        "structure": structure,
        "structure_vp": structure_vp,
        "structure_orderbook": (*structure, *orderbook),
        "structure_vp_orderbook": (*structure_vp, *orderbook),
    }


def existing_columns(frame: DataFrame, columns: tuple[str, ...]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def fit_score(train: DataFrame, test: DataFrame, target: str, columns: list[str]) -> dict[str, object] | None:
    train_y = pd.to_numeric(train[target], errors="coerce")
    test_y = pd.to_numeric(test[target], errors="coerce")
    train_mask = train_y.notna()
    test_mask = test_y.notna()
    if int(train_mask.sum()) < 80 or int(test_mask.sum()) < 18:
        return None
    y_train = train_y[train_mask].astype(int)
    y_test = test_y[test_mask].astype(int)
    if y_train.nunique() < 2 or y_test.nunique() < 2 or not columns:
        return None
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=4,
                    min_samples_leaf=18,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(train.loc[train_mask, columns], y_train)
    probabilities = model.predict_proba(test.loc[test_mask, columns])[:, 1]
    predictions = probabilities >= 0.5
    return {
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "test_event_rate": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
    }


def summarize_results(results: DataFrame) -> DataFrame:
    if results.empty:
        return pd.DataFrame()
    pivot = results.pivot_table(index=["event_id", "target", "holdout_window"], columns="feature_set", values="roc_auc", aggfunc="first").reset_index()
    rows = []
    for (event_id, target), group in pivot.groupby(["event_id", "target"]):
        item = {"event_id": event_id, "target": target, "windows": int(len(group))}
        for column in ("price", "structure", "structure_vp", "structure_orderbook", "structure_vp_orderbook"):
            item[f"mean_auc_{column}"] = float(group[column].mean()) if column in group else np.nan
        if {"structure_vp", "structure_vp_orderbook"}.issubset(group.columns):
            delta = group["structure_vp_orderbook"] - group["structure_vp"]
            item["mean_ob_delta_vs_structure_vp"] = float(delta.mean())
            item["positive_ob_delta_windows"] = int(delta.gt(0.0).sum())
        else:
            item["mean_ob_delta_vs_structure_vp"] = np.nan
            item["positive_ob_delta_windows"] = 0
        item["strict_pass"] = bool(
            item["windows"] >= 3
            and item["positive_ob_delta_windows"] >= 3
            and item["mean_ob_delta_vs_structure_vp"] >= 0.03
            and item["mean_auc_structure_vp_orderbook"] >= 0.55
        )
        item["freqai_ready"] = bool(
            item["windows"] >= 3
            and item["positive_ob_delta_windows"] >= 2
            and item["mean_ob_delta_vs_structure_vp"] >= 0.02
            and item["mean_auc_structure_vp_orderbook"] >= 0.55
        )
        rows.append(item)
    return pd.DataFrame(rows).sort_values(["strict_pass", "freqai_ready", "mean_ob_delta_vs_structure_vp"], ascending=[False, False, False])


def num(frame: DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")


def bool_col(frame: DataFrame, column: str) -> pd.Series:
    return num(frame, column).fillna(0.0).gt(0.0)


if __name__ == "__main__":
    raise SystemExit(main())

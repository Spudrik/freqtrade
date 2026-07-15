from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.trader_confluence_hypotheses import (  # noqa: E402
    TraderHypothesis,
    hypothesis_by_id,
    implemented_hypotheses,
)
from user_data.Custom_Launcher.research.context_features.trader_confluence_feature_taxonomy import (  # noqa: E402
    coarse_source_family,
    source_detail,
)
from user_data.Custom_Launcher.research.context_features.source_coverage_audit import (  # noqa: E402
    build_source_audit,
)


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT = USER_DATA_DIR / "research_news_data" / "context_features" / "confluence_cache" / "trader_confluence_1h_latest.parquet"
DEFAULT_OUTPUT_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"
SOURCE_DETAIL_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "bearish_event_breakdown_confluence": (
        "price_ohlcv",
        "structure_volume_profile",
        "structure_tlv2_support_resistance",
        "structure_bos_choch_market_structure",
        "orderbook_spot",
        "orderbook_bybit_linear",
        "orderbook_bybit_inverse",
        "context_topic_severity",
        "context_gdelt_events",
        "context_global_market_macro",
    ),
    "bullish_event_breakout_confluence": (
        "price_ohlcv",
        "structure_volume_profile",
        "structure_tlv2_support_resistance",
        "structure_bos_choch_market_structure",
        "orderbook_spot",
        "orderbook_bybit_linear",
        "orderbook_bybit_inverse",
        "context_topic_severity",
        "context_gdelt_events",
        "context_global_market_macro",
    ),
    "quiet_technical_breakout_confluence": (
        "price_ohlcv",
        "structure_volume_profile",
        "structure_tlv2_support_resistance",
        "structure_bos_choch_market_structure",
        "orderbook_spot",
        "orderbook_bybit_linear",
        "orderbook_bybit_inverse",
        "context_article_source_activity",
    ),
    "quiet_technical_breakdown_confluence": (
        "price_ohlcv",
        "structure_volume_profile",
        "structure_tlv2_support_resistance",
        "structure_bos_choch_market_structure",
        "orderbook_spot",
        "orderbook_bybit_linear",
        "orderbook_bybit_inverse",
        "context_article_source_activity",
    ),
    "context_pressure_plus_structure_break": (
        "price_ohlcv",
        "structure_volume_profile",
        "structure_tlv2_support_resistance",
        "structure_bos_choch_market_structure",
        "context_topic_severity",
        "context_gdelt_events",
        "context_global_market_macro",
    ),
    "macro_context_liquidity_stress_breakdown": (
        "price_ohlcv",
        "structure_volume_profile",
        "structure_tlv2_support_resistance",
        "structure_bos_choch_market_structure",
        "orderbook_spot",
        "orderbook_bybit_linear",
        "orderbook_bybit_inverse",
        "context_topic_severity",
        "context_gdelt_events",
        "context_global_market_macro",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run hypothesis-led direct tests on the trader-confluence 1h snapshot.")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="trader_confluence_v1")
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--model-ablations", action="store_true")
    parser.add_argument("--max-model-features", type=int, default=220)
    parser.add_argument("--max-model-ablation-candidates", type=int, default=8)
    parser.add_argument("--disable-source-detail-gate", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    hypotheses = implemented_hypotheses()
    plan = {
        "mode": "setup-only unless --execute is supplied",
        "snapshot": str(args.snapshot),
        "hypotheses": [h.to_dict() for h in hypotheses],
        "controls": [
            "same_regime_without_trigger",
            "random_eligible_rows",
            "opposite_direction_setup where available",
            "shuffled_labels for binary score AUC",
        ],
        "model_ablations_enabled": bool(args.model_ablations),
        "source_detail_gate_enabled": not bool(args.disable_source_detail_gate),
        "max_model_ablation_candidates": int(args.max_model_ablation_candidates),
        "pass_interpretation": "AUC around 0.55 can be useful if stable, event-scoped, and better than controls/ablations.",
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.snapshot.exists():
        raise FileNotFoundError(args.snapshot)

    frame = pd.read_parquet(args.snapshot)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    missing_columns = missing_hypothesis_columns(frame, hypotheses)
    if missing_columns:
        raise ValueError(f"Implemented hypothesis columns are missing: {missing_columns}")
    source_masks: dict[str, dict[str, Series]] | None = None
    source_audit = DataFrame()
    if not args.disable_source_detail_gate:
        source_audit, source_masks = build_source_audit(frame, 0.01)
    reports, summaries, model_rows = run_direct_tests(
        frame,
        hypotheses,
        min_rows=int(args.min_rows),
        run_model_ablations=bool(args.model_ablations),
        max_model_features=int(args.max_model_features),
        max_model_ablation_candidates=int(args.max_model_ablation_candidates),
        source_masks=source_masks,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag.strip()) or "latest"
    report_path = args.output_dir / f"trader_confluence_direct_tests_{safe_tag}_report.csv"
    summary_path = args.output_dir / f"trader_confluence_direct_tests_{safe_tag}_summary.csv"
    model_path = args.output_dir / f"trader_confluence_direct_tests_{safe_tag}_model_ablations.csv"
    markdown_path = args.output_dir / f"trader_confluence_direct_tests_{safe_tag}_report.md"
    meta_path = args.output_dir / f"trader_confluence_direct_tests_{safe_tag}_meta.json"
    source_audit_path = args.output_dir / f"trader_confluence_direct_tests_{safe_tag}_source_audit.csv"
    reports.to_csv(report_path, index=False)
    summaries.to_csv(summary_path, index=False)
    model_rows.to_csv(model_path, index=False)
    if not source_audit.empty:
        source_audit.to_csv(source_audit_path, index=False)
    markdown_path.write_text(markdown_report(hypotheses, summaries, reports, model_rows, frame.columns), encoding="utf-8")
    meta = {
        "plan": plan,
        "snapshot_rows": int(len(frame)),
        "snapshot_columns": int(len(frame.columns)),
        "report_rows": int(len(reports)),
        "summary_rows": int(len(summaries)),
        "model_ablation_rows": int(len(model_rows)),
        "outputs": {
            "report": str(report_path),
            "summary": str(summary_path),
            "model_ablations": str(model_path),
            "markdown": str(markdown_path),
            "source_audit": str(source_audit_path) if not source_audit.empty else None,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "summary": str(summary_path), "model_ablations": str(model_path), "markdown": str(markdown_path), "meta": str(meta_path), "rows": int(len(reports)), "summary_rows": int(len(summaries))}, indent=2))
    return 0


def run_direct_tests(
    frame: DataFrame,
    hypotheses: tuple[TraderHypothesis, ...],
    *,
    min_rows: int,
    run_model_ablations: bool,
    max_model_features: int,
    max_model_ablation_candidates: int,
    source_masks: dict[str, dict[str, Series]] | None = None,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    detail_rows: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        missing = [column for column in (hypothesis.setup_column, hypothesis.trigger_column, hypothesis.score_column) if column not in frame]
        if missing:
            detail_rows.append(skip_row(hypothesis, "missing_hypothesis_columns", {"missing_columns": ",".join(missing)}))
            continue
        eligible = eligible_mask(frame, hypothesis, source_masks)
        setup = eligible & flag(frame, hypothesis.setup_column)
        trigger = setup & flag(frame, hypothesis.trigger_column)
        same_regime_without_trigger = setup & ~trigger
        random_control = deterministic_random_mask(eligible & ~trigger, int(trigger.sum()), seed=stable_seed(hypothesis.hypothesis_id))
        opposite = opposite_direction_mask(frame, hypothesis)
        missing_ingredient_controls = missing_ingredient_control_masks(frame, hypothesis, setup, trigger)
        for target in hypothesis.target_columns:
            if target not in frame:
                detail_rows.append(skip_row(hypothesis, "missing_target", {"target": target}))
                continue
            rows = [
                group_metrics(frame, hypothesis, target, trigger, "trigger"),
                group_metrics(frame, hypothesis, target, same_regime_without_trigger, "same_regime_without_trigger"),
                group_metrics(frame, hypothesis, target, random_control, "random_eligible_control"),
            ]
            if opposite is not None:
                rows.append(group_metrics(frame, hypothesis, target, eligible & opposite, "opposite_direction_control"))
            for group_name, control_mask in missing_ingredient_controls.items():
                rows.append(group_metrics(frame, hypothesis, target, control_mask, f"same_setup_missing_{group_name}_control"))
            detail_rows.extend(rows)
            detail_rows.append(score_metrics(frame, hypothesis, target, setup, min_rows))
            detail_rows.append(price_structure_baseline_metrics(frame, hypothesis, target, setup, min_rows))
    details = pd.DataFrame(detail_rows)
    summary = summarize_details(details)
    model_rows: list[dict[str, Any]] = []
    if run_model_ablations:
        hypothesis_map = {hypothesis.hypothesis_id: hypothesis for hypothesis in hypotheses}
        for candidate in model_ablation_candidates(summary, max_model_ablation_candidates):
            hypothesis = hypothesis_map.get(str(candidate.get("hypothesis_id", "")))
            target = str(candidate.get("target", ""))
            if hypothesis is None or target not in frame or not is_binary_target(frame[target]):
                continue
            print(json.dumps({"phase": "model_ablation", "hypothesis_id": hypothesis.hypothesis_id, "target": target}), flush=True)
            setup = eligible_mask(frame, hypothesis, source_masks) & flag(frame, hypothesis.setup_column)
            model_rows.extend(model_ablation_tests(frame, hypothesis, target, setup, max_model_features))
    models = pd.DataFrame(model_rows)
    return details, summary, models


def eligible_mask(frame: DataFrame, hypothesis: TraderHypothesis, source_masks: dict[str, dict[str, Series]] | None = None) -> Series:
    mask = pd.Series(True, index=frame.index)
    required_details = required_source_details(hypothesis)
    if source_masks is not None:
        for detail in required_details:
            detail_mask = source_masks.get(detail, {}).get("usable")
            if detail_mask is None:
                return pd.Series(False, index=frame.index)
            mask &= detail_mask.reindex(frame.index).fillna(False)
        return mask.fillna(False)
    if "structure" in hypothesis.source_families and "structure_present" in frame:
        mask &= flag(frame, "structure_present")
    if "context" in hypothesis.source_families and "context_present" in frame:
        mask &= flag(frame, "context_present")
    if "orderbook" in hypothesis.source_families:
        if "conf_ob_venue_count_present" in frame:
            mask &= num(frame, "conf_ob_venue_count_present").fillna(0.0).gt(0.0)
        else:
            mask &= flag(frame, "ob_spot_present") | flag(frame, "ob_linear_present") | flag(frame, "ob_inverse_present")
    return mask.fillna(False)


def required_source_details(hypothesis: TraderHypothesis) -> tuple[str, ...]:
    explicit = SOURCE_DETAIL_REQUIREMENTS.get(hypothesis.hypothesis_id)
    if explicit:
        return explicit
    details = ["price_ohlcv"]
    if "structure" in hypothesis.source_families:
        details.extend(
            [
                "structure_volume_profile",
                "structure_tlv2_support_resistance",
                "structure_bos_choch_market_structure",
                "structure_pattern_geometry",
            ]
        )
    if "orderbook" in hypothesis.source_families:
        details.extend(["orderbook_spot", "orderbook_bybit_linear", "orderbook_bybit_inverse"])
    if "context" in hypothesis.source_families:
        details.extend(["context_topic_severity", "context_gdelt_events"])
    return tuple(dict.fromkeys(details))


def group_metrics(frame: DataFrame, hypothesis: TraderHypothesis, target: str, mask: Series, group_id: str) -> dict[str, Any]:
    target_values = pd.to_numeric(frame.loc[mask, target], errors="coerce").dropna()
    row: dict[str, Any] = base_row(hypothesis, target, group_id)
    row["rows"] = int(len(target_values))
    if target_values.empty:
        return row
    row["target_mean"] = float(target_values.mean())
    row["target_median"] = float(target_values.median())
    row["target_min"] = float(target_values.min())
    row["target_max"] = float(target_values.max())
    if is_binary_series(target_values):
        row["event_rate"] = float(target_values.mean())
        row["positive_rows"] = int(target_values.eq(1.0).sum())
    return row


def score_metrics(frame: DataFrame, hypothesis: TraderHypothesis, target: str, setup_mask: Series, min_rows: int) -> dict[str, Any]:
    row = base_row(hypothesis, target, "score_within_setup")
    score = pd.to_numeric(frame[hypothesis.score_column], errors="coerce")
    target_values = pd.to_numeric(frame[target], errors="coerce")
    mask = setup_mask & score.notna() & target_values.notna()
    row["rows"] = int(mask.sum())
    if int(mask.sum()) < max(30, min_rows // 2):
        row["skip_reason"] = "too_few_setup_rows"
        return row
    y = target_values[mask]
    x = score[mask]
    row["target_mean"] = float(y.mean())
    row["score_mean"] = float(x.mean())
    row["score_top20_target_mean"] = float(y[x >= x.quantile(0.80)].mean()) if x.nunique() > 1 else np.nan
    row["score_bottom80_target_mean"] = float(y[x < x.quantile(0.80)].mean()) if x.nunique() > 1 else np.nan
    row["score_top20_lift"] = row["score_top20_target_mean"] - row["score_bottom80_target_mean"] if np.isfinite(row["score_top20_target_mean"]) and np.isfinite(row["score_bottom80_target_mean"]) else np.nan
    if is_binary_series(y):
        if y.nunique() >= 2 and x.nunique() >= 2:
            row["roc_auc"] = float(roc_auc_score(y.astype(int), x))
            row["average_precision"] = float(average_precision_score(y.astype(int), x))
            shuffled = y.sample(frac=1.0, random_state=stable_seed(hypothesis.hypothesis_id + target)).reset_index(drop=True)
            row["shuffled_roc_auc"] = float(roc_auc_score(shuffled.astype(int), x.reset_index(drop=True))) if shuffled.nunique() >= 2 else np.nan
        row["event_rate"] = float(y.mean())
        row.update(monthly_binary_score(frame.loc[mask, ["date"]].copy(), y, x))
    else:
        corr = spearmanr(x, y, nan_policy="omit")
        row["spearman"] = float(corr.correlation) if corr.correlation is not None and np.isfinite(corr.correlation) else np.nan
        row["spearman_pvalue"] = float(corr.pvalue) if corr.pvalue is not None and np.isfinite(corr.pvalue) else np.nan
    oriented_lift, orientation_note = orient_lift(hypothesis, target, row.get("score_top20_lift", np.nan))
    row["oriented_top20_lift"] = oriented_lift
    row["orientation_note"] = orientation_note
    return row


def price_structure_baseline_metrics(frame: DataFrame, hypothesis: TraderHypothesis, target: str, setup_mask: Series, min_rows: int) -> dict[str, Any]:
    row = base_row(hypothesis, target, "price_structure_baseline_score")
    score = price_structure_baseline_score(frame, hypothesis, target)
    target_values = pd.to_numeric(frame[target], errors="coerce")
    mask = setup_mask & score.notna() & target_values.notna()
    row["rows"] = int(mask.sum())
    if int(mask.sum()) < max(30, min_rows // 2):
        row["skip_reason"] = "too_few_setup_rows"
        return row
    y = target_values[mask]
    x = score[mask]
    row["target_mean"] = float(y.mean())
    row["score_mean"] = float(x.mean())
    row["score_top20_target_mean"] = float(y[x >= x.quantile(0.80)].mean()) if x.nunique() > 1 else np.nan
    row["score_bottom80_target_mean"] = float(y[x < x.quantile(0.80)].mean()) if x.nunique() > 1 else np.nan
    row["score_top20_lift"] = row["score_top20_target_mean"] - row["score_bottom80_target_mean"] if np.isfinite(row["score_top20_target_mean"]) and np.isfinite(row["score_bottom80_target_mean"]) else np.nan
    if is_binary_series(y):
        if y.nunique() >= 2 and x.nunique() >= 2:
            row["roc_auc"] = float(roc_auc_score(y.astype(int), x))
            row["average_precision"] = float(average_precision_score(y.astype(int), x))
        row["event_rate"] = float(y.mean())
        row.update(monthly_binary_score(frame.loc[mask, ["date"]].copy(), y, x))
    else:
        corr = spearmanr(x, y, nan_policy="omit")
        row["spearman"] = float(corr.correlation) if corr.correlation is not None and np.isfinite(corr.correlation) else np.nan
        row["spearman_pvalue"] = float(corr.pvalue) if corr.pvalue is not None and np.isfinite(corr.pvalue) else np.nan
    oriented_lift, orientation_note = orient_lift(hypothesis, target, row.get("score_top20_lift", np.nan))
    row["oriented_top20_lift"] = oriented_lift
    row["orientation_note"] = orientation_note
    return row


def price_structure_baseline_score(frame: DataFrame, hypothesis: TraderHypothesis, target: str) -> Series:
    target_lower = target.lower()
    bullish = mean_columns(
        frame,
        [
            "conf_structure_breakout_trigger_score",
            "conf_volume_bullish_confirmation",
            "px_close_breakout_24h",
            "px_range_position_72h",
        ],
    )
    bearish = pd.concat(
        [
            num(frame, "conf_structure_breakdown_trigger_score"),
            num(frame, "conf_volume_bearish_confirmation"),
            num(frame, "px_close_breakdown_24h"),
            1.0 - num(frame, "px_range_position_72h").clip(lower=0.0, upper=1.0),
        ],
        axis=1,
    ).mean(axis=1, skipna=True)
    resistance_reject = mean_columns(
        frame,
        [
            "conf_structure_resistance_stack_score",
            "conf_near_range_high",
            "conf_volume_bearish_confirmation",
        ],
    )
    support_bounce = mean_columns(
        frame,
        [
            "conf_structure_support_stack_score",
            "conf_near_range_low",
            "conf_volume_bullish_confirmation",
        ],
    )
    if "breakout_failure" in target_lower or "failed_breakout" in target_lower:
        return resistance_reject
    if "breakdown_failure" in target_lower or "failed_breakdown" in target_lower:
        return support_bounce
    if "breakdown" in target_lower or "drawdown" in target_lower or "minus" in target_lower:
        return bearish
    if "breakout" in target_lower or "upside" in target_lower or "plus" in target_lower:
        return bullish
    if hypothesis.direction == "bearish":
        return bearish
    if hypothesis.direction == "bullish":
        return bullish
    return pd.concat([bullish, bearish], axis=1).max(axis=1)


def mean_columns(frame: DataFrame, columns: list[str]) -> Series:
    values = [num(frame, column) for column in columns if column in frame]
    if not values:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.concat(values, axis=1).mean(axis=1, skipna=True)


def model_ablation_tests(frame: DataFrame, hypothesis: TraderHypothesis, target: str, setup_mask: Series, max_features: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y_all = pd.to_numeric(frame[target], errors="coerce")
    usable = setup_mask & y_all.notna()
    if int(usable.sum()) < 240 or y_all[usable].nunique() < 2:
        return rows
    work = frame.loc[usable].copy()
    work["_month"] = work["date"].dt.strftime("%Y-%m")
    months = [month for month, count in work["_month"].value_counts().items() if count >= 80]
    if len(months) < 2:
        return rows
    for feature_set, columns in feature_sets(work, max_features).items():
        for holdout in sorted(months):
            train = work[work["_month"] < holdout]
            test = work[work["_month"] == holdout]
            if train.empty:
                continue
            metric = fit_binary_model(train, test, target, columns)
            if metric:
                rows.append({"hypothesis_id": hypothesis.hypothesis_id, "target": target, "feature_set": feature_set, "holdout_window": holdout, "validation_type": "walk_forward_month", **metric})
    return rows


def model_ablation_candidates(summary: DataFrame, limit: int) -> list[dict[str, Any]]:
    if summary.empty or limit <= 0:
        return []
    work = summary.copy()
    if "roc_auc" in work:
        work = work[pd.to_numeric(work["roc_auc"], errors="coerce").notna()]
    if work.empty:
        return []
    if "watchlist_positive" in work and work["watchlist_positive"].fillna(False).astype(bool).any():
        work = work[work["watchlist_positive"].fillna(False).astype(bool)]
    for column in ("roc_auc", "oriented_top20_lift", "rows_trigger"):
        if column not in work:
            work[column] = np.nan
    work = work.sort_values(["roc_auc", "oriented_top20_lift", "rows_trigger"], ascending=[False, False, False])
    return work.head(int(limit)).to_dict("records")


def feature_sets(frame: DataFrame, max_features: int) -> dict[str, list[str]]:
    all_features = [column for column in frame.columns if is_model_feature(column, frame[column])]
    price_structure = [
        c
        for c in all_features
        if c.startswith("px_")
        or c.startswith("st_")
        or c.startswith("conf_structure_")
        or c.startswith("conf_volume_")
        or c.startswith("conf_near_")
        or c.startswith("conf_lvn_")
    ]
    sets = {
        "price": [c for c in all_features if c.startswith("px_")],
        "structure": [c for c in all_features if c.startswith("st_") or c.startswith("conf_structure_") or c.startswith("conf_volume_") or c.startswith("conf_near_") or c.startswith("conf_lvn_")],
        "orderbook": [c for c in all_features if c.startswith("ob_") or c.startswith("conf_ob_")],
        "context": [c for c in all_features if c.startswith("ctx_") or c.startswith("context_") or c.startswith("conf_context_")],
        "structure_volume_profile": [c for c in all_features if source_detail(c) == "structure_volume_profile" or c.startswith("conf_lvn_")],
        "structure_tlv2_support_resistance": [c for c in all_features if source_detail(c) == "structure_tlv2_support_resistance"],
        "structure_bos_choch_market_structure": [c for c in all_features if source_detail(c) == "structure_bos_choch_market_structure"],
        "structure_pattern_geometry": [c for c in all_features if source_detail(c) == "structure_pattern_geometry"],
        "context_article_source_activity": [c for c in all_features if source_detail(c) == "context_article_source_activity"],
        "context_topic_severity": [c for c in all_features if source_detail(c) == "context_topic_severity"],
        "context_gdelt_events": [c for c in all_features if source_detail(c) == "context_gdelt_events"],
        "context_gkg_documents": [c for c in all_features if source_detail(c) == "context_gkg_documents"],
        "context_google_trends": [c for c in all_features if source_detail(c) == "context_google_trends"],
        "context_btc_etf_flows": [c for c in all_features if source_detail(c) == "context_btc_etf_flows"],
        "context_global_market_macro": [c for c in all_features if source_detail(c) == "context_global_market_macro"],
        "confluence_only": [c for c in all_features if c.startswith("conf_")],
        "full": all_features,
        "minus_orderbook": [c for c in all_features if not feature_depends_on_family(c, "orderbook")],
        "minus_context": [c for c in all_features if not feature_depends_on_family(c, "context")],
        "minus_structure": [c for c in all_features if not feature_depends_on_family(c, "structure")],
        "minus_structure_volume_profile": [c for c in all_features if source_detail(c) != "structure_volume_profile" and not c.startswith("conf_lvn_")],
        "minus_structure_tlv2_support_resistance": [c for c in all_features if source_detail(c) != "structure_tlv2_support_resistance"],
        "minus_structure_bos_choch_market_structure": [c for c in all_features if source_detail(c) != "structure_bos_choch_market_structure"],
        "minus_structure_pattern_geometry": [c for c in all_features if source_detail(c) != "structure_pattern_geometry"],
        "minus_context_article_source_activity": [c for c in all_features if source_detail(c) != "context_article_source_activity"],
        "minus_context_topic_severity": [c for c in all_features if source_detail(c) != "context_topic_severity"],
        "minus_context_gdelt_events": [c for c in all_features if source_detail(c) != "context_gdelt_events"],
        "minus_context_gkg_documents": [c for c in all_features if source_detail(c) != "context_gkg_documents"],
        "minus_context_google_trends": [c for c in all_features if source_detail(c) != "context_google_trends"],
        "minus_context_btc_etf_flows": [c for c in all_features if source_detail(c) != "context_btc_etf_flows"],
        "minus_context_global_market_macro": [c for c in all_features if source_detail(c) != "context_global_market_macro"],
        "minus_volume_pressure": [c for c in all_features if not feature_depends_on_volume(c, frame.columns)],
        "minus_multi_timeframe_confirmation": [c for c in all_features if not (c.startswith("st_4h_") or c.startswith("st_1d_") or c.startswith("conf_structure_resistance_stack") or c.startswith("conf_structure_support_stack"))],
        "minus_ob_spot": [c for c in all_features if not c.startswith("ob_spot_") and not orderbook_composite_feature(c)],
        "minus_ob_linear": [c for c in all_features if not c.startswith("ob_linear_") and not orderbook_composite_feature(c)],
        "minus_ob_inverse": [c for c in all_features if not c.startswith("ob_inverse_") and not orderbook_composite_feature(c)],
        "only_ob_spot": [*price_structure, *[c for c in all_features if c.startswith("ob_spot_")]],
        "only_ob_linear": [*price_structure, *[c for c in all_features if c.startswith("ob_linear_")]],
        "only_ob_inverse": [*price_structure, *[c for c in all_features if c.startswith("ob_inverse_")]],
        "all_venues_no_composites": [*price_structure, *[c for c in all_features if c.startswith("ob_")]],
    }
    trimmed = {name: trim_features(frame, columns, max_features) for name, columns in sets.items() if columns}
    return {name: columns for name, columns in trimmed.items() if columns}


def fit_binary_model(train: DataFrame, test: DataFrame, target: str, columns: list[str]) -> dict[str, Any] | None:
    if not columns:
        return None
    train_y = pd.to_numeric(train[target], errors="coerce")
    test_y = pd.to_numeric(test[target], errors="coerce")
    train_mask = train_y.notna()
    test_mask = test_y.notna()
    if int(train_mask.sum()) < 120 or int(test_mask.sum()) < 40:
        return None
    y_train = train_y[train_mask].astype(int)
    y_test = test_y[test_mask].astype(int)
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return None
    usable_columns = observed_model_columns(train.loc[train_mask], columns)
    if not usable_columns:
        return None
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(max_iter=120, learning_rate=0.04, max_leaf_nodes=7, min_samples_leaf=18, l2_regularization=0.2, random_state=42)),
        ]
    )
    model.fit(train.loc[train_mask, usable_columns], y_train)
    probabilities = model.predict_proba(test.loc[test_mask, usable_columns])[:, 1]
    return {
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "test_event_rate": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "feature_count": int(len(usable_columns)),
    }


def observed_model_columns(frame: DataFrame, columns: list[str]) -> list[str]:
    usable: list[str] = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not values.notna().any():
            continue
        variance = float(values.var(skipna=True)) if values.notna().sum() > 1 else 0.0
        if not np.isfinite(variance) or variance == 0.0:
            continue
        usable.append(column)
    return usable


def summarize_details(details: DataFrame) -> DataFrame:
    if details.empty:
        return DataFrame()
    score_rows = details[details["group_id"].eq("score_within_setup")].copy()
    if score_rows.empty:
        return DataFrame()
    group_pivot = details[details["group_id"].isin(["trigger", "same_regime_without_trigger", "random_eligible_control"])].pivot_table(
        index=["hypothesis_id", "target"],
        columns="group_id",
        values=["rows", "target_mean", "event_rate"],
        aggfunc="first",
    )
    if not group_pivot.empty:
        group_pivot.columns = [f"{metric}_{group}" for metric, group in group_pivot.columns]
        group_pivot = group_pivot.reset_index()
    baseline_rows = details[details["group_id"].eq("price_structure_baseline_score")].copy()
    baseline_pivot = DataFrame()
    if not baseline_rows.empty:
        baseline_columns = [
            column
            for column in (
                "rows",
                "roc_auc",
                "average_precision",
                "score_top20_lift",
                "oriented_top20_lift",
                "spearman",
                "spearman_pvalue",
                "monthly_windows",
                "monthly_auc_mean",
                "monthly_auc_positive_windows",
            )
            if column in baseline_rows.columns
        ]
        baseline_pivot = baseline_rows[["hypothesis_id", "target", *baseline_columns]].copy()
        baseline_pivot = baseline_pivot.rename(columns={column: f"baseline_{column}" for column in baseline_columns})
    summary_cols = [
        "hypothesis_id",
        "target",
        "direction",
        "rows",
        "event_rate",
        "roc_auc",
        "average_precision",
        "shuffled_roc_auc",
        "score_top20_lift",
        "oriented_top20_lift",
        "orientation_note",
        "spearman",
        "spearman_pvalue",
        "monthly_windows",
        "monthly_auc_mean",
        "monthly_auc_positive_windows",
    ]
    available = [column for column in summary_cols if column in score_rows.columns]
    out = score_rows[available].copy()
    if not group_pivot.empty:
        out = out.merge(group_pivot, on=["hypothesis_id", "target"], how="left")
    if not baseline_pivot.empty:
        out = out.merge(baseline_pivot, on=["hypothesis_id", "target"], how="left")
    out["watchlist_positive"] = out.apply(watchlist_rule, axis=1)
    return out.sort_values([column for column in ("watchlist_positive", "roc_auc", "score_top20_lift") if column in out], ascending=False)


def monthly_binary_score(date_frame: DataFrame, y: Series, score: Series) -> dict[str, Any]:
    tmp = DataFrame({"date": pd.to_datetime(date_frame["date"], utc=True), "target": y.to_numpy(), "score": score.to_numpy()})
    tmp["_month"] = tmp["date"].dt.strftime("%Y-%m")
    aucs: list[float] = []
    for _, group in tmp.groupby("_month"):
        if len(group) < 40 or group["target"].nunique() < 2 or group["score"].nunique() < 2:
            continue
        aucs.append(float(roc_auc_score(group["target"].astype(int), group["score"])))
    return {
        "monthly_windows": int(len(aucs)),
        "monthly_auc_mean": float(np.mean(aucs)) if aucs else np.nan,
        "monthly_auc_positive_windows": int(sum(auc >= 0.55 for auc in aucs)),
    }


def trim_features(frame: DataFrame, columns: list[str], max_features: int) -> list[str]:
    scored: list[tuple[float, str]] = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        non_null = float(values.notna().mean())
        if non_null < 0.05:
            continue
        variance = float(values.var(skipna=True)) if values.notna().any() else 0.0
        if not np.isfinite(variance) or variance == 0.0:
            continue
        scored.append((non_null * np.log1p(variance), column))
    return [column for _, column in sorted(scored, reverse=True)[:max_features]]


def is_model_feature(column: str, series: Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False
    lower = column.lower()
    if lower in {"open", "high", "low", "close", "volume"}:
        return False
    blocked_prefixes = ("future_", "hit_", "time_to_")
    blocked_suffixes = ("_next_6h", "_next_24h")
    if lower in {"schema_version"} or lower.startswith(blocked_prefixes) or lower.endswith(blocked_suffixes):
        return False
    if "generated_at" in lower or "source_future_violation" in lower:
        return False
    if lower.endswith("_present") or lower.endswith("_row_present") or lower.endswith("_low_coverage"):
        return False
    diagnostic_tokens = (
        "coverage",
        "source_age",
        "missing",
        "stale",
        "available_at",
        "source_min_ts",
        "source_max_ts",
        "min_source",
        "max_source",
        "quality",
        "debug",
    )
    if any(token in lower for token in diagnostic_tokens):
        return False
    absolute_level_tokens = (
        "_line_",
        "_zone_price",
        "_wall_price",
        "mid_price",
        "_last_swing_high",
        "_last_swing_low",
        "_break_level",
        "_invalidation_level",
        "prior_poc",
        "prior_vah",
        "prior_val",
        "_poc",
        "_vah",
        "_val",
    )
    if any(token in lower for token in absolute_level_tokens):
        return False
    if lower.startswith("ob_") and "_price" in lower and "price_agreement" not in lower:
        return False
    return True


def orient_lift(hypothesis: TraderHypothesis, target: str, lift: Any) -> tuple[float, str]:
    try:
        value = float(lift)
    except Exception:
        return np.nan, "unavailable"
    if not np.isfinite(value):
        return np.nan, "unavailable"
    lower = target.lower()
    if lower.startswith("time_to_"):
        return -value, "lower_time_is_better"
    if "max_drawdown" in lower or "minus" in lower:
        return -value if hypothesis.direction in {"bearish", "bidirectional"} else value, "more_negative_or_faster_down_is_bearish"
    if "future_return" in lower:
        if hypothesis.direction == "bearish":
            return -value, "negative_return_is_bearish"
        if hypothesis.direction == "bullish":
            return value, "positive_return_is_bullish"
    return value, "higher_target_is_better"


def watchlist_rule(row: pd.Series) -> bool:
    rows = float(row.get("rows")) if pd.notna(row.get("rows")) else 0.0
    trigger_rows = float(row.get("rows_trigger")) if pd.notna(row.get("rows_trigger")) else 0.0
    control_rows = float(row.get("rows_same_regime_without_trigger")) if pd.notna(row.get("rows_same_regime_without_trigger")) else 0.0
    random_rows = float(row.get("rows_random_eligible_control")) if pd.notna(row.get("rows_random_eligible_control")) else 0.0
    if rows < 100 or trigger_rows < 50 or control_rows < 100 or random_rows < 50:
        return False
    target = str(row.get("target", "")).lower()
    auc = float(row.get("roc_auc")) if pd.notna(row.get("roc_auc")) else np.nan
    shuffled = float(row.get("shuffled_roc_auc")) if pd.notna(row.get("shuffled_roc_auc")) else np.nan
    windows = float(row.get("monthly_windows")) if pd.notna(row.get("monthly_windows")) else 0.0
    positive_windows = float(row.get("monthly_auc_positive_windows")) if pd.notna(row.get("monthly_auc_positive_windows")) else 0.0
    if np.isfinite(auc):
        monthly_ok = windows == 0.0 or positive_windows >= max(2.0, windows * 0.50)
        shuffle_ok = not np.isfinite(shuffled) or (auc - shuffled) >= 0.02
        trigger_rate = float(row.get("event_rate_trigger")) if pd.notna(row.get("event_rate_trigger")) else np.nan
        control_rate = float(row.get("event_rate_same_regime_without_trigger")) if pd.notna(row.get("event_rate_same_regime_without_trigger")) else np.nan
        random_rate = float(row.get("event_rate_random_eligible_control")) if pd.notna(row.get("event_rate_random_eligible_control")) else np.nan
        baseline_auc = float(row.get("baseline_roc_auc")) if pd.notna(row.get("baseline_roc_auc")) else np.nan
        same_regime_lift_ok = not (np.isfinite(trigger_rate) and np.isfinite(control_rate)) or trigger_rate >= control_rate + 0.03
        random_lift_ok = not (np.isfinite(trigger_rate) and np.isfinite(random_rate)) or trigger_rate >= random_rate + 0.03
        baseline_ok = not np.isfinite(baseline_auc) or auc >= baseline_auc + 0.01
        return bool(auc >= 0.55 and monthly_ok and shuffle_ok and same_regime_lift_ok and random_lift_ok and baseline_ok)

    oriented = float(row.get("oriented_top20_lift")) if pd.notna(row.get("oriented_top20_lift")) else np.nan
    spearman = float(row.get("spearman")) if pd.notna(row.get("spearman")) else np.nan
    pvalue = float(row.get("spearman_pvalue")) if pd.notna(row.get("spearman_pvalue")) else np.nan
    if not np.isfinite(oriented):
        return False
    if target.startswith("time_to_"):
        material_lift = oriented >= 1.0
    elif "return" in target or "drawdown" in target or "upside" in target:
        material_lift = oriented >= 0.002
    else:
        material_lift = oriented > 0.0
    correlation_ok = not np.isfinite(spearman) or abs(spearman) >= 0.05
    significance_ok = not np.isfinite(pvalue) or pvalue <= 0.10
    return bool(material_lift and correlation_ok and significance_ok)


def missing_hypothesis_columns(frame: DataFrame, hypotheses: tuple[TraderHypothesis, ...]) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for hypothesis in hypotheses:
        columns = [hypothesis.setup_column, hypothesis.trigger_column, hypothesis.score_column]
        absent = [column for column in columns if column not in frame]
        if absent:
            missing[hypothesis.hypothesis_id] = absent
    return missing


def feature_depends_on_family(column: str, family: str) -> bool:
    lower = column.lower()
    if source_detail(lower) == family:
        return True
    if coarse_source_family(lower) == family:
        return True
    if family == "orderbook":
        if lower.startswith("ob_") or lower.startswith("conf_ob_"):
            return True
    if family == "context":
        if lower.startswith("ctx_") or lower.startswith("context_") or lower.startswith("conf_context_"):
            return True
    if family == "structure":
        if lower.startswith("st_") or lower.startswith("conf_structure_") or lower.startswith("conf_near_") or lower.startswith("conf_lvn_"):
            return True
    if lower.startswith("conf_"):
        for hypothesis_id, hypothesis in hypothesis_by_id().items():
            prefix = f"conf_{hypothesis_id}_"
            if lower.startswith(prefix) and family in hypothesis.source_families:
                return True
    return False


def feature_depends_on_volume(column: str, all_columns: pd.Index | list[str] | tuple[str, ...]) -> bool:
    lower = column.lower()
    if "volume" in lower:
        return True
    if lower.startswith("conf_"):
        for hypothesis_id in hypothesis_by_id():
            prefix = f"conf_{hypothesis_id}_"
            component_prefix = f"{prefix}cmp_"
            if not lower.startswith(prefix):
                continue
            return any(str(candidate).lower().startswith(component_prefix) and "volume" in str(candidate).lower() for candidate in all_columns)
    return False


def orderbook_composite_feature(column: str) -> bool:
    lower = column.lower()
    if lower.startswith("conf_ob_"):
        return True
    if lower.startswith("conf_"):
        for hypothesis_id, hypothesis in hypothesis_by_id().items():
            if lower.startswith(f"conf_{hypothesis_id}_") and "orderbook" in hypothesis.source_families:
                return True
    return False


def markdown_report(
    hypotheses: tuple[TraderHypothesis, ...],
    summaries: DataFrame,
    details: DataFrame,
    model_rows: DataFrame,
    frame_columns: pd.Index | list[str] | tuple[str, ...],
) -> str:
    lines = [
        "# Trader Confluence Direct Test Report",
        "",
        "This report is generated from frozen parquet snapshots. It is meant to explain what behaviour was tested before any FreqAI promotion.",
        "",
        "Source-detail clean-window gating is enabled unless the run metadata says otherwise. Hypotheses are eligible only where required source-detail blocks have usable rows.",
        "",
    ]
    summary_by_key = {
        (row["hypothesis_id"], row["target"]): row
        for _, row in summaries.iterrows()
    } if not summaries.empty else {}
    for hypothesis in hypotheses:
        component_columns = hypothesis_component_columns(hypothesis.hypothesis_id, frame_columns)
        component_details = source_detail_summary(component_columns)
        lines.extend(
            [
                f"## {hypothesis.hypothesis_id}",
                "",
                f"- Theory: {hypothesis.theory}",
                f"- Direction: `{hypothesis.direction}`",
                f"- Source families: `{', '.join(hypothesis.source_families)}`",
                f"- Source detail groups: `{component_details}`",
                f"- Required source-detail blocks: `{', '.join(required_source_details(hypothesis))}`",
                f"- Setup column: `{hypothesis.setup_column}`",
                f"- Trigger column: `{hypothesis.trigger_column}`",
                f"- Score column: `{hypothesis.score_column}`",
                f"- Setup components: {format_component_columns(component_columns.get('setup', []))}",
                f"- Trigger components: {format_component_columns(component_columns.get('trigger', []))}",
                f"- Score components: {format_component_columns(component_columns.get('score', []))}",
                "",
                "| Target | Rows | Trigger Rows | Trigger Mean/Rate | Same-Regime Mean/Rate | Random Mean/Rate | AUC | Baseline AUC | Shuffled AUC | AP | Top Lift | Baseline Lift | Monthly +/N | Spearman | Watchlist |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for target in hypothesis.target_columns:
            row = summary_by_key.get((hypothesis.hypothesis_id, target))
            if row is None:
                lines.append(f"| `{target}` | 0 | 0 |  |  |  |  |  |  |  |  |  |  |  | no data |")
                continue
            monthly = ""
            if pd.notna(row.get("monthly_auc_positive_windows")) and pd.notna(row.get("monthly_windows")):
                monthly = f"{int(row.get('monthly_auc_positive_windows'))}/{int(row.get('monthly_windows'))}"
            lines.append(
                f"| `{target}` | {fmt_int(row.get('rows'))} | {fmt_int(row.get('rows_trigger'))} | "
                f"{fmt_float(first_present(row, ['event_rate_trigger', 'target_mean_trigger']))} | "
                f"{fmt_float(first_present(row, ['event_rate_same_regime_without_trigger', 'target_mean_same_regime_without_trigger']))} | "
                f"{fmt_float(first_present(row, ['event_rate_random_eligible_control', 'target_mean_random_eligible_control']))} | "
                f"{fmt_float(row.get('roc_auc'))} | {fmt_float(row.get('baseline_roc_auc'))} | {fmt_float(row.get('shuffled_roc_auc'))} | "
                f"{fmt_float(row.get('average_precision'))} | {fmt_float(row.get('oriented_top20_lift'))} | {fmt_float(row.get('baseline_oriented_top20_lift'))} | "
                f"{monthly} | {fmt_float(row.get('spearman'))} | `{bool(row.get('watchlist_positive'))}` |"
            )
        lines.append("")
    if not model_rows.empty:
        aggregate = (
            model_rows.groupby(["hypothesis_id", "target", "feature_set"])
            .agg(windows=("roc_auc", "count"), mean_auc=("roc_auc", "mean"), positive_windows=("roc_auc", lambda s: int((s >= 0.55).sum())))
            .reset_index()
            .sort_values("mean_auc", ascending=False)
            .head(40)
        )
        lines.extend(["## Model Ablation Snapshot", "", aggregate.to_markdown(index=False), ""])
    return "\n".join(lines) + "\n"


def hypothesis_component_columns(hypothesis_id: str, columns: pd.Index | list[str] | tuple[str, ...]) -> dict[str, list[str]]:
    prefix = f"conf_{hypothesis_id}_cmp_"
    result = {"setup": [], "trigger": [], "score": []}
    for column in columns:
        name = str(column)
        if not name.startswith(prefix):
            continue
        for role in result:
            role_prefix = f"{prefix}{role}_"
            if name.startswith(role_prefix):
                result[role].append(name)
                break
    return {role: sorted(values) for role, values in result.items()}


def format_component_columns(columns: list[str]) -> str:
    if not columns:
        return "`none`"
    return ", ".join(f"`{column}`" for column in columns)


def source_detail_summary(component_columns: dict[str, list[str]]) -> str:
    details: set[str] = set()
    for columns in component_columns.values():
        for column in columns:
            details.update(source_details_for_component(column))
    return ", ".join(sorted(details)) if details else "none"


def source_details_for_component(column: str) -> set[str]:
    lower = column.lower()
    details = {source_detail(lower)}
    if "context" in lower or "macro" in lower:
        details.update({"context_composite", "context_topic_severity", "context_gdelt_events", "context_global_market_macro"})
    if "quiet" in lower:
        details.update({"context_article_source_activity", "context_composite"})
    if "resistance_stack" in lower or "support_stack" in lower or "near_key_level" in lower:
        details.update({"structure_volume_profile", "structure_tlv2_support_resistance", "structure_pattern_geometry", "structure_composite"})
    if "breakout" in lower or "breakdown" in lower:
        details.update({"price_ohlcv", "structure_bos_choch_market_structure", "structure_volume_profile", "structure_composite"})
    if "volume" in lower:
        details.add("price_ohlcv")
    if "lvn" in lower:
        details.add("structure_volume_profile")
    if any(token in lower for token in ("book", "wall", "vacuum", "liquidity", "resistance_removed", "support_removed", "support_rebuild", "pressure")):
        details.update({"orderbook_composite", "orderbook_spot", "orderbook_bybit_linear", "orderbook_bybit_inverse"})
    return {detail for detail in details if detail != "hypothesis_confluence"}


def missing_ingredient_control_masks(frame: DataFrame, hypothesis: TraderHypothesis, setup: Series, trigger: Series) -> dict[str, Series]:
    grouped_columns = component_columns_by_ingredient(frame, hypothesis.hypothesis_id, "trigger")
    controls: dict[str, Series] = {}
    if len(grouped_columns) < 2:
        return controls
    group_active = {
        group_name: component_group_active(frame, columns)
        for group_name, columns in grouped_columns.items()
        if columns
    }
    for group_name, active in group_active.items():
        other_groups = [mask for other_name, mask in group_active.items() if other_name != group_name]
        if not other_groups:
            continue
        other_active = pd.concat(other_groups, axis=1).all(axis=1)
        controls[group_name] = (setup & ~trigger & ~active & other_active).fillna(False)
    return controls


def component_columns_by_ingredient(frame: DataFrame, hypothesis_id: str, role: str) -> dict[str, list[str]]:
    prefix = f"conf_{hypothesis_id}_cmp_{role}_"
    groups = {"structure": [], "volume": [], "orderbook": []}
    for column in frame.columns:
        name = str(column)
        if not name.startswith(prefix):
            continue
        group = ingredient_group_for_component(name)
        if group:
            groups[group].append(name)
    return {group: sorted(columns) for group, columns in groups.items() if columns}


def ingredient_group_for_component(column: str) -> str | None:
    lower = column.lower()
    component_name = lower
    for marker in ("_cmp_setup_", "_cmp_trigger_", "_cmp_score_"):
        if marker in lower:
            component_name = lower.split(marker, 1)[1]
            break
    if "volume" in component_name or "_vol_" in component_name:
        return "volume"
    orderbook_tokens = (
        "orderbook",
        "_ob_",
        "book",
        "wall",
        "vacuum",
        "liquidity",
        "resistance_removed",
        "support_removed",
        "support_rebuild",
        "bid_",
        "ask_",
        "venue",
    )
    if any(token in component_name for token in orderbook_tokens):
        return "orderbook"
    structure_tokens = (
        "structure",
        "resistance_stack",
        "support_stack",
        "near_key_level",
        "near_range",
        "breakout",
        "breakdown",
        "range_",
        "lvn",
        "poc",
        "vah",
        "val",
        "compression",
        "trendline",
        "tlv",
        "vp",
        "bos",
        "choch",
        "acceptance",
        "reclaim",
        "rejection",
    )
    if any(token in component_name for token in structure_tokens):
        return "structure"
    return None


def component_group_active(frame: DataFrame, columns: list[str]) -> Series:
    if not columns:
        return pd.Series(False, index=frame.index)
    values = frame[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return values.gt(0.0).any(axis=1).fillna(False)


def first_present(row: pd.Series, columns: list[str]) -> Any:
    for column in columns:
        value = row.get(column)
        if pd.notna(value):
            return value
    return np.nan


def fmt_float(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return ""
    return f"{number:.3f}" if np.isfinite(number) else ""


def fmt_int(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return "0"
    return str(int(number)) if np.isfinite(number) else "0"


def opposite_direction_mask(frame: DataFrame, hypothesis: TraderHypothesis) -> Series | None:
    opposite = {
        "bearish_event_breakdown_confluence": "conf_bullish_event_breakout_confluence_score",
        "bullish_event_breakout_confluence": "conf_bearish_event_breakdown_confluence_score",
        "quiet_technical_breakout_confluence": "conf_quiet_technical_breakdown_confluence_score",
        "quiet_technical_breakdown_confluence": "conf_quiet_technical_breakout_confluence_score",
        "multi_tf_resistance_rejection": "conf_multi_tf_support_bounce_score",
        "multi_tf_support_bounce": "conf_multi_tf_resistance_rejection_score",
        "orderbook_resistance_evaporation_breakout": "conf_orderbook_support_removal_breakdown_score",
        "orderbook_support_removal_breakdown": "conf_orderbook_resistance_evaporation_breakout_score",
    }.get(hypothesis.hypothesis_id)
    if not opposite or opposite not in frame:
        return None
    return pd.to_numeric(frame[opposite], errors="coerce").fillna(0.0).ge(0.20)


def deterministic_random_mask(mask: Series, count: int, *, seed: int) -> Series:
    out = pd.Series(False, index=mask.index)
    indices = list(mask[mask].index)
    if count <= 0 or not indices:
        return out
    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=min(count, len(indices)), replace=False)
    out.loc[chosen] = True
    return out


def skip_row(hypothesis: TraderHypothesis, reason: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row = base_row(hypothesis, "", "skipped")
    row["skip_reason"] = reason
    if extra:
        row.update(extra)
    return row


def base_row(hypothesis: TraderHypothesis, target: str, group_id: str) -> dict[str, Any]:
    return {
        "hypothesis_id": hypothesis.hypothesis_id,
        "direction": hypothesis.direction,
        "source_families": ",".join(hypothesis.source_families),
        "target": target,
        "group_id": group_id,
        "rows": 0,
    }


def num(frame: DataFrame, column: str) -> Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def flag(frame: DataFrame, column: str) -> Series:
    return num(frame, column).fillna(0.0).gt(0.0)


def is_binary_target(series: Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna().unique()
    return len(values) > 0 and set(np.round(values, 8)).issubset({0.0, 1.0})


def is_binary_series(series: Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna().unique()
    return len(values) > 0 and set(np.round(values, 8)).issubset({0.0, 1.0})


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


if __name__ == "__main__":
    raise SystemExit(main())

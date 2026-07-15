from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.source_coverage_audit import build_source_audit  # noqa: E402
from user_data.Custom_Launcher.research.context_features.trader_confluence_direct_tests import (  # noqa: E402
    eligible_mask,
    is_binary_target,
    monthly_binary_score,
    price_structure_baseline_score,
    stable_seed,
)
from user_data.Custom_Launcher.research.context_features.trader_confluence_hypotheses import (  # noqa: E402
    TraderHypothesis,
    implemented_hypotheses,
)


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT = USER_DATA_DIR / "research_news_data" / "context_features" / "confluence_cache" / "trader_confluence_1h_latest.parquet"
DEFAULT_OUTPUT_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep trader-confluence component gates before promoting hypotheses to FreqAI.")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="trader_confluence_threshold_sweep")
    parser.add_argument("--component-thresholds", default="0.20,0.30,0.35,0.45")
    parser.add_argument("--score-thresholds", default="0.20,0.30,0.35,0.45,0.55")
    parser.add_argument("--min-setup-rows", type=int, default=100)
    parser.add_argument("--min-trigger-rows", type=int, default=50)
    parser.add_argument("--top", type=int, default=80)
    parser.add_argument("--disable-source-detail-gate", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    plan = {
        "mode": "setup-only unless --execute is supplied",
        "snapshot": str(args.snapshot),
        "component_thresholds": parse_float_list(args.component_thresholds),
        "score_thresholds": parse_float_list(args.score_thresholds),
        "min_setup_rows": int(args.min_setup_rows),
        "min_trigger_rows": int(args.min_trigger_rows),
        "source_detail_gate_enabled": not bool(args.disable_source_detail_gate),
        "purpose": "Find multi-component gate settings with enough rows, directional control lift, baseline lift, and stable AUC before FreqAI.",
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.snapshot.exists():
        raise FileNotFoundError(args.snapshot)

    frame = pd.read_parquet(args.snapshot)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    source_masks = None
    if not args.disable_source_detail_gate:
        _, source_masks = build_source_audit(frame, 0.01)
    rows = run_sweep(
        frame,
        implemented_hypotheses(),
        component_thresholds=plan["component_thresholds"],
        score_thresholds=plan["score_thresholds"],
        min_setup_rows=int(args.min_setup_rows),
        min_trigger_rows=int(args.min_trigger_rows),
        source_masks=source_masks,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag.strip()) or "latest"
    csv_path = args.output_dir / f"trader_confluence_threshold_sweep_{safe_tag}.csv"
    markdown_path = args.output_dir / f"trader_confluence_threshold_sweep_{safe_tag}.md"
    collapsed_csv_path = args.output_dir / f"trader_confluence_threshold_sweep_{safe_tag}_collapsed_pass_summary.csv"
    collapsed_markdown_path = args.output_dir / f"trader_confluence_threshold_sweep_{safe_tag}_collapsed_pass_summary.md"
    meta_path = args.output_dir / f"trader_confluence_threshold_sweep_{safe_tag}_meta.json"
    collapsed = collapsed_pass_summary(rows)
    rows.to_csv(csv_path, index=False)
    collapsed.to_csv(collapsed_csv_path, index=False)
    markdown_path.write_text(markdown_report(rows, int(args.top)), encoding="utf-8")
    collapsed_markdown_path.write_text(collapsed_markdown_report(collapsed), encoding="utf-8")
    meta = {
        **plan,
        "rows": int(len(rows)),
        "collapsed_pass_summary_rows": int(len(collapsed)),
        "outputs": {
            "csv": str(csv_path),
            "markdown": str(markdown_path),
            "collapsed_pass_summary_csv": str(collapsed_csv_path),
            "collapsed_pass_summary_markdown": str(collapsed_markdown_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "markdown": str(markdown_path), "collapsed_pass_summary_csv": str(collapsed_csv_path), "collapsed_pass_summary_markdown": str(collapsed_markdown_path), "meta": str(meta_path), "rows": int(len(rows)), "collapsed_pass_summary_rows": int(len(collapsed))}, indent=2))
    return 0


def run_sweep(
    frame: DataFrame,
    hypotheses: tuple[TraderHypothesis, ...],
    *,
    component_thresholds: list[float],
    score_thresholds: list[float],
    min_setup_rows: int,
    min_trigger_rows: int,
    source_masks: dict[str, dict[str, Series]] | None,
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        setup_components = component_frame(frame, hypothesis.hypothesis_id, "setup")
        trigger_components = component_frame(frame, hypothesis.hypothesis_id, "trigger")
        if setup_components.empty or trigger_components.empty:
            continue
        eligible = eligible_mask(frame, hypothesis, source_masks)
        score = pd.to_numeric(frame[hypothesis.score_column], errors="coerce")
        for target in hypothesis.target_columns:
            if target not in frame or not is_binary_target(frame[target]):
                continue
            target_values = pd.to_numeric(frame[target], errors="coerce")
            baseline = price_structure_baseline_score(frame, hypothesis, target)
            for component_threshold in component_thresholds:
                setup_count_frame = setup_components.ge(component_threshold)
                trigger_count_frame = trigger_components.ge(component_threshold)
                max_setup_count = int(setup_count_frame.shape[1])
                max_trigger_count = int(trigger_count_frame.shape[1])
                setup_score = setup_components.mean(axis=1, skipna=True).fillna(0.0)
                trigger_score = trigger_components.mean(axis=1, skipna=True).fillna(0.0)
                for min_setup_count in range(1, max_setup_count + 1):
                    setup_count = setup_count_frame.sum(axis=1)
                    for setup_threshold in score_thresholds:
                        setup = eligible & setup_score.ge(setup_threshold) & setup_count.ge(min_setup_count)
                        setup_valid = setup & target_values.notna() & score.notna()
                        setup_rows = int(setup_valid.sum())
                        if setup_rows < min_setup_rows:
                            continue
                        for min_trigger_count in range(1, max_trigger_count + 1):
                            trigger_count = trigger_count_frame.sum(axis=1)
                            for trigger_threshold in score_thresholds:
                                trigger = setup & trigger_score.ge(trigger_threshold) & trigger_count.ge(min_trigger_count)
                                row = evaluate_candidate(
                                    frame,
                                    hypothesis,
                                    target,
                                    target_values,
                                    score,
                                    baseline,
                                    setup_valid,
                                    trigger,
                                    component_threshold,
                                    setup_threshold,
                                    trigger_threshold,
                                    min_setup_count,
                                    min_trigger_count,
                                    min_trigger_rows,
                                )
                                if row:
                                    rows.append(row)
    if not rows:
        return DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(["pass_candidate", "auc_minus_baseline", "roc_auc", "trigger_lift_vs_same"], ascending=[False, False, False, False])


def evaluate_candidate(
    frame: DataFrame,
    hypothesis: TraderHypothesis,
    target: str,
    target_values: Series,
    score: Series,
    baseline: Series,
    setup_valid: Series,
    trigger: Series,
    component_threshold: float,
    setup_threshold: float,
    trigger_threshold: float,
    min_setup_count: int,
    min_trigger_count: int,
    min_trigger_rows: int,
) -> dict[str, Any] | None:
    y = target_values[setup_valid]
    x = score[setup_valid]
    if y.nunique() < 2 or x.nunique() < 2:
        return None
    trigger_mask = trigger & target_values.notna()
    same_mask = setup_valid & ~trigger
    trigger_rows = int(trigger_mask.sum())
    same_rows = int(same_mask.sum())
    if trigger_rows <= 0 or same_rows <= 0:
        return None
    random_mask = deterministic_random_mask(setup_valid & ~trigger, trigger_rows, seed=stable_seed(hypothesis.hypothesis_id + target + str(component_threshold)))
    trigger_rate = float(target_values[trigger_mask].mean())
    same_rate = float(target_values[same_mask].mean())
    random_rate = float(target_values[random_mask & target_values.notna()].mean()) if int(random_mask.sum()) else np.nan
    roc_auc = float(roc_auc_score(y.astype(int), x))
    baseline_mask = setup_valid & baseline.notna()
    baseline_auc = np.nan
    if int(baseline_mask.sum()) >= 50:
        by = target_values[baseline_mask]
        bx = baseline[baseline_mask]
        if by.nunique() >= 2 and bx.nunique() >= 2:
            baseline_auc = float(roc_auc_score(by.astype(int), bx))
    monthly = monthly_binary_score(frame.loc[setup_valid, ["date"]].copy(), y, x)
    auc_minus_baseline = roc_auc - baseline_auc if np.isfinite(baseline_auc) else np.nan
    pass_candidate = (
        trigger_rows >= min_trigger_rows
        and trigger_rate >= same_rate + 0.03
        and (not np.isfinite(random_rate) or trigger_rate >= random_rate + 0.03)
        and roc_auc >= 0.55
        and (not np.isfinite(baseline_auc) or roc_auc >= baseline_auc + 0.01)
        and monthly.get("monthly_auc_positive_windows", 0) >= max(2, monthly.get("monthly_windows", 0) * 0.5)
    )
    return {
        "hypothesis_id": hypothesis.hypothesis_id,
        "target": target,
        "component_threshold": float(component_threshold),
        "setup_threshold": float(setup_threshold),
        "trigger_threshold": float(trigger_threshold),
        "min_setup_count": int(min_setup_count),
        "min_trigger_count": int(min_trigger_count),
        "setup_rows": int(setup_valid.sum()),
        "trigger_rows": trigger_rows,
        "same_regime_rows": same_rows,
        "trigger_event_rate": trigger_rate,
        "same_regime_event_rate": same_rate,
        "random_event_rate": random_rate,
        "trigger_lift_vs_same": trigger_rate - same_rate,
        "trigger_lift_vs_random": trigger_rate - random_rate if np.isfinite(random_rate) else np.nan,
        "roc_auc": roc_auc,
        "baseline_roc_auc": baseline_auc,
        "auc_minus_baseline": auc_minus_baseline,
        "average_precision": float(average_precision_score(y.astype(int), x)),
        **monthly,
        "pass_candidate": bool(pass_candidate),
    }


def component_frame(frame: DataFrame, hypothesis_id: str, role: str) -> DataFrame:
    prefix = f"conf_{hypothesis_id}_cmp_{role}_"
    columns = [column for column in frame.columns if column.startswith(prefix)]
    if not columns:
        return DataFrame(index=frame.index)
    return frame[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def deterministic_random_mask(mask: Series, count: int, *, seed: int) -> Series:
    out = pd.Series(False, index=mask.index)
    indices = list(mask[mask].index)
    if count <= 0 or not indices:
        return out
    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=min(count, len(indices)), replace=False)
    out.loc[chosen] = True
    return out


def collapsed_pass_summary(rows: DataFrame) -> DataFrame:
    if rows.empty or "pass_candidate" not in rows:
        return DataFrame()
    passed = rows[rows["pass_candidate"].fillna(False).astype(bool)].copy()
    if passed.empty:
        return DataFrame(
            columns=[
                "hypothesis_id",
                "target",
                "pass_variants",
                "best_component_threshold",
                "best_setup_threshold",
                "best_trigger_threshold",
                "best_min_setup_count",
                "best_min_trigger_count",
                "best_setup_rows",
                "best_trigger_rows",
                "best_trigger_event_rate",
                "best_same_regime_event_rate",
                "best_random_event_rate",
                "best_trigger_lift_vs_same",
                "best_trigger_lift_vs_random",
                "best_roc_auc",
                "best_baseline_roc_auc",
                "best_auc_minus_baseline",
                "best_average_precision",
                "best_monthly_auc_positive_windows",
                "best_monthly_windows",
                "median_trigger_event_rate",
                "median_same_regime_event_rate",
                "median_roc_auc",
            ]
        )
    sort_columns = ["auc_minus_baseline", "roc_auc", "trigger_lift_vs_same", "trigger_rows"]
    best_rows = (
        passed.sort_values(sort_columns, ascending=[False, False, False, False], na_position="last")
        .drop_duplicates(["hypothesis_id", "target"], keep="first")
        .set_index(["hypothesis_id", "target"])
    )
    aggregate = (
        passed.groupby(["hypothesis_id", "target"])
        .agg(
            pass_variants=("pass_candidate", "size"),
            median_trigger_event_rate=("trigger_event_rate", "median"),
            median_same_regime_event_rate=("same_regime_event_rate", "median"),
            median_roc_auc=("roc_auc", "median"),
        )
        .reset_index()
    )
    best_columns = [
        "component_threshold",
        "setup_threshold",
        "trigger_threshold",
        "min_setup_count",
        "min_trigger_count",
        "setup_rows",
        "trigger_rows",
        "trigger_event_rate",
        "same_regime_event_rate",
        "random_event_rate",
        "trigger_lift_vs_same",
        "trigger_lift_vs_random",
        "roc_auc",
        "baseline_roc_auc",
        "auc_minus_baseline",
        "average_precision",
        "monthly_auc_positive_windows",
        "monthly_windows",
    ]
    best = best_rows[best_columns].rename(columns={column: f"best_{column}" for column in best_columns}).reset_index()
    out = aggregate.merge(best, on=["hypothesis_id", "target"], how="left")
    ordered_columns = [
        "hypothesis_id",
        "target",
        "pass_variants",
        "best_component_threshold",
        "best_setup_threshold",
        "best_trigger_threshold",
        "best_min_setup_count",
        "best_min_trigger_count",
        "best_setup_rows",
        "best_trigger_rows",
        "best_trigger_event_rate",
        "best_same_regime_event_rate",
        "best_random_event_rate",
        "best_trigger_lift_vs_same",
        "best_trigger_lift_vs_random",
        "best_roc_auc",
        "best_baseline_roc_auc",
        "best_auc_minus_baseline",
        "best_average_precision",
        "best_monthly_auc_positive_windows",
        "best_monthly_windows",
        "median_trigger_event_rate",
        "median_same_regime_event_rate",
        "median_roc_auc",
    ]
    return out[ordered_columns].sort_values(
        ["best_auc_minus_baseline", "best_roc_auc", "best_trigger_lift_vs_same"],
        ascending=[False, False, False],
        na_position="last",
    )


def markdown_report(rows: DataFrame, top: int) -> str:
    lines = ["# Trader Confluence Threshold Sweep", ""]
    if rows.empty:
        return "\n".join([*lines, "No sweep rows were produced.", ""])
    passed = rows[rows["pass_candidate"].fillna(False).astype(bool)]
    lines.append(f"- Sweep rows: `{len(rows)}`")
    lines.append(f"- Passing candidates: `{len(passed)}`")
    lines.append("")
    preview_columns = [
        "hypothesis_id",
        "target",
        "component_threshold",
        "setup_threshold",
        "trigger_threshold",
        "min_setup_count",
        "min_trigger_count",
        "setup_rows",
        "trigger_rows",
        "trigger_event_rate",
        "same_regime_event_rate",
        "roc_auc",
        "baseline_roc_auc",
        "auc_minus_baseline",
        "monthly_auc_positive_windows",
        "monthly_windows",
    ]
    for title, frame in (("Passing Candidates", passed), ("Best Non-Passing Candidates", rows[~rows["pass_candidate"].fillna(False).astype(bool)])):
        lines.extend([f"## {title}", ""])
        if frame.empty:
            lines.extend(["None.", ""])
            continue
        lines.append(frame[preview_columns].head(top).to_markdown(index=False))
        lines.append("")
    return "\n".join(lines) + "\n"


def collapsed_markdown_report(rows: DataFrame) -> str:
    lines = ["# Trader Confluence Threshold Sweep Collapsed Pass Summary", ""]
    if rows.empty:
        return "\n".join([*lines, "No passing hypothesis/target groups were produced.", ""])
    lines.append(f"- Hypothesis/target groups with passing variants: `{len(rows)}`")
    lines.append("- Each row collapses all passing threshold variants for one hypothesis and target.")
    lines.append("")
    preview_columns = [
        "hypothesis_id",
        "target",
        "pass_variants",
        "best_component_threshold",
        "best_setup_threshold",
        "best_trigger_threshold",
        "best_min_setup_count",
        "best_min_trigger_count",
        "best_trigger_rows",
        "best_trigger_lift_vs_same",
        "best_roc_auc",
        "best_baseline_roc_auc",
        "best_auc_minus_baseline",
        "best_monthly_auc_positive_windows",
        "best_monthly_windows",
    ]
    lines.append(rows[preview_columns].to_markdown(index=False))
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())

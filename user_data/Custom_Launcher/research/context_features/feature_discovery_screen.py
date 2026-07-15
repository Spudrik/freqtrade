from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_FEATURE_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "feature_discovery"
DEFAULT_OUTPUT_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"
DEFAULT_TARGETS = (
    "breakout_success_next_6h",
    "breakdown_success_next_6h",
    "large_drawdown_next_6h",
    "large_drawdown_next_24h",
    "hit_plus_3pct_before_minus_2pct",
    "hit_minus_3pct_before_plus_2pct",
    "fakeout_next_24h",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Screen controlled feature-discovery candidates with controls and a tree-model shortlist.")
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--dictionary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--min-rows", type=int, default=1000)
    parser.add_argument("--min-positive", type=int, default=30)
    parser.add_argument("--top-univariate-per-target", type=int, default=120)
    parser.add_argument("--top-report-rows", type=int, default=80)
    parser.add_argument("--skip-tree", action="store_true", help="Write univariate/control screening only; useful when broad tree shortlists are too slow.")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    candidates_path = args.candidates or latest_file(DEFAULT_FEATURE_DIR, "feature_discovery_candidates_*.parquet")
    dictionary_path = args.dictionary or infer_dictionary_path(candidates_path)
    plan = {
        "mode": "setup-only unless --execute is supplied",
        "candidates": str(candidates_path) if candidates_path else None,
        "dictionary": str(dictionary_path) if dictionary_path else None,
        "targets": parse_csv(args.targets),
        "min_rows": int(args.min_rows),
        "min_positive": int(args.min_positive),
        "skip_tree": bool(args.skip_tree),
        "purpose": "Use feature discovery as a screening layer before writing hand-authored trader hypotheses.",
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if candidates_path is None or not candidates_path.exists():
        raise FileNotFoundError(candidates_path)

    frame = pd.read_parquet(candidates_path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    dictionary = pd.read_csv(dictionary_path) if dictionary_path and dictionary_path.exists() else DataFrame()
    features = [column for column in frame.columns if column.startswith("fd__")]
    targets = [target for target in parse_csv(args.targets) if target in frame]
    univariate = screen_univariate(
        frame,
        features,
        targets,
        dictionary,
        min_rows=int(args.min_rows),
        min_positive=int(args.min_positive),
    )
    tree_rows = DataFrame()
    if not args.skip_tree:
        tree_rows = screen_tree_shortlists(
            frame,
            univariate,
            targets,
            top_per_target=int(args.top_univariate_per_target),
            min_positive=int(args.min_positive),
        )
    combined = combine_results(univariate, tree_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = safe_name(args.tag)
    univariate_path = args.output_dir / f"feature_discovery_screen_{safe_tag}_univariate.csv"
    tree_path = args.output_dir / f"feature_discovery_screen_{safe_tag}_tree.csv"
    combined_path = args.output_dir / f"feature_discovery_screen_{safe_tag}_combined.csv"
    markdown_path = args.output_dir / f"feature_discovery_screen_{safe_tag}.md"
    meta_path = args.output_dir / f"feature_discovery_screen_{safe_tag}_meta.json"
    univariate.to_csv(univariate_path, index=False)
    tree_rows.to_csv(tree_path, index=False)
    combined.to_csv(combined_path, index=False)
    markdown_path.write_text(markdown_report(combined, int(args.top_report_rows)), encoding="utf-8")
    meta = {
        **plan,
        "candidate_rows": int(len(frame)),
        "candidate_features": int(len(features)),
        "targets_found": targets,
        "univariate_rows": int(len(univariate)),
        "tree_rows": int(len(tree_rows)),
        "stable_shortlist_rows": int(combined["stable_shortlist"].fillna(False).astype(bool).sum()) if not combined.empty else 0,
        "outputs": {
            "univariate": str(univariate_path),
            "tree": str(tree_path),
            "combined": str(combined_path),
            "markdown": str(markdown_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"combined": str(combined_path), "markdown": str(markdown_path), "meta": str(meta_path), "stable_shortlist_rows": meta["stable_shortlist_rows"]}, indent=2))
    return 0


def screen_univariate(
    frame: DataFrame,
    features: list[str],
    targets: list[str],
    dictionary: DataFrame,
    *,
    min_rows: int,
    min_positive: int,
) -> DataFrame:
    dictionary_map = dictionary.set_index("feature").to_dict("index") if not dictionary.empty and "feature" in dictionary else {}
    rows: list[dict[str, Any]] = []
    months = frame["date"].dt.to_period("M").astype(str)
    for target in targets:
        y_all = pd.to_numeric(frame[target], errors="coerce")
        if not is_binary(y_all):
            continue
        for feature in features:
            x_all = pd.to_numeric(frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
            valid = y_all.notna() & x_all.notna()
            row_count = int(valid.sum())
            positives = int(y_all[valid].sum())
            negatives = int(row_count - positives)
            if row_count < min_rows or positives < min_positive or negatives < min_positive:
                continue
            y = y_all[valid].astype(int)
            x = x_all[valid]
            if x.nunique(dropna=True) < 3:
                continue
            auc = float(roc_auc_score(y, x))
            oriented_auc = auc if auc >= 0.5 else 1.0 - auc
            orientation = "higher_value_more_likely" if auc >= 0.5 else "lower_value_more_likely"
            feature_mask = pd.Series(False, index=frame.index)
            feature_mask.loc[valid[valid].index] = top_quantile_mask(x, orientation, 0.10).values
            top_rate = float(y_all[feature_mask & y_all.notna()].mean()) if int((feature_mask & y_all.notna()).sum()) else np.nan
            base_rate = float(y.mean())
            random_rate = random_control_rate(y_all, valid, int(feature_mask.sum()), feature + target)
            monthly = monthly_stability(months[valid], y, x, orientation)
            shuffled_auc = shuffled_oriented_auc(y, x, feature + target)
            info = dictionary_map.get(feature, {})
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "base_column": info.get("base_column", ""),
                    "transform": info.get("transform", ""),
                    "source_family": info.get("source_family", ""),
                    "source_detail": info.get("source_detail", ""),
                    "plain_english": info.get("plain_english", ""),
                    "rows": row_count,
                    "positive_rows": positives,
                    "base_event_rate": base_rate,
                    "top_decile_event_rate": top_rate,
                    "random_event_rate": random_rate,
                    "lift_vs_base": top_rate - base_rate if np.isfinite(top_rate) else np.nan,
                    "lift_vs_random": top_rate - random_rate if np.isfinite(top_rate) and np.isfinite(random_rate) else np.nan,
                    "roc_auc": auc,
                    "oriented_auc": oriented_auc,
                    "shuffled_oriented_auc": shuffled_auc,
                    "orientation": orientation,
                    **monthly,
                }
            )
    if not rows:
        return DataFrame()
    out = pd.DataFrame(rows)
    out["univariate_pass"] = (
        out["oriented_auc"].ge(0.55)
        & out["lift_vs_base"].ge(0.03)
        & out["lift_vs_random"].ge(0.02)
        & out["oriented_auc"].ge(out["shuffled_oriented_auc"].fillna(0.5) + 0.02)
        & out["monthly_positive_ratio"].ge(0.55)
        & out["monthly_windows"].ge(4)
    )
    return out.sort_values(["univariate_pass", "oriented_auc", "lift_vs_base"], ascending=[False, False, False])


def screen_tree_shortlists(
    frame: DataFrame,
    univariate: DataFrame,
    targets: list[str],
    *,
    top_per_target: int,
    min_positive: int,
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    if univariate.empty:
        return DataFrame()
    for target in targets:
        target_uni = univariate[univariate["target"].eq(target)].head(top_per_target)
        features = list(dict.fromkeys(target_uni["feature"].dropna().astype(str)))
        if not features:
            continue
        y_all = pd.to_numeric(frame[target], errors="coerce")
        valid = y_all.notna()
        dates = frame.loc[valid, "date"]
        y = y_all[valid].astype(int)
        if int(y.sum()) < min_positive or int((1 - y).sum()) < min_positive:
            continue
        split_date = dates.quantile(0.70)
        train_mask = dates.lt(split_date)
        test_mask = dates.ge(split_date)
        if int(y[train_mask].sum()) < min_positive or int(y[test_mask].sum()) < 10:
            continue
        x = frame.loc[valid, features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        usable_features = [column for column in x.columns if x.loc[train_mask, column].notna().any() and x.loc[test_mask, column].notna().any()]
        if not usable_features:
            continue
        x = x[usable_features]
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=160,
                        max_depth=5,
                        min_samples_leaf=25,
                        random_state=stable_seed(target),
                        n_jobs=-1,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        model.fit(x.loc[train_mask], y.loc[train_mask])
        proba = model.predict_proba(x.loc[test_mask])[:, 1]
        if len(np.unique(y.loc[test_mask])) < 2:
            continue
        test_auc = float(roc_auc_score(y.loc[test_mask], proba))
        test_ap = float(average_precision_score(y.loc[test_mask], proba))
        importances = model.named_steps["model"].feature_importances_
        for feature, importance in sorted(zip(usable_features, importances), key=lambda item: item[1], reverse=True):
            uni = target_uni[target_uni["feature"].eq(feature)].head(1)
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "tree_importance": float(importance),
                    "tree_test_auc": test_auc,
                    "tree_test_average_precision": test_ap,
                    "tree_train_rows": int(train_mask.sum()),
                    "tree_test_rows": int(test_mask.sum()),
                    "tree_test_positive_rows": int(y.loc[test_mask].sum()),
                    "source_family": uni["source_family"].iloc[0] if not uni.empty else "",
                    "source_detail": uni["source_detail"].iloc[0] if not uni.empty else "",
                }
            )
    return pd.DataFrame(rows).sort_values(["tree_test_auc", "tree_importance"], ascending=[False, False]) if rows else DataFrame()


def combine_results(univariate: DataFrame, tree_rows: DataFrame) -> DataFrame:
    if univariate.empty:
        return DataFrame()
    out = univariate.copy()
    if not tree_rows.empty:
        tree_summary = (
            tree_rows.groupby(["target", "feature"], as_index=False)
            .agg(
                tree_importance=("tree_importance", "max"),
                tree_test_auc=("tree_test_auc", "max"),
                tree_test_average_precision=("tree_test_average_precision", "max"),
            )
        )
        out = out.merge(tree_summary, on=["target", "feature"], how="left")
    else:
        out["tree_importance"] = np.nan
        out["tree_test_auc"] = np.nan
        out["tree_test_average_precision"] = np.nan
    if tree_rows.empty:
        out["stable_shortlist"] = out["univariate_pass"].fillna(False)
    else:
        out["stable_shortlist"] = (
            out["univariate_pass"].fillna(False)
            & (out["tree_test_auc"].fillna(0.5).ge(0.53) | out["tree_importance"].fillna(0.0).ge(0.01))
        )
    return out.sort_values(["stable_shortlist", "univariate_pass", "tree_importance", "oriented_auc", "lift_vs_base"], ascending=[False, False, False, False, False])


def markdown_report(rows: DataFrame, top: int) -> str:
    lines = ["# Feature Discovery Screen", ""]
    if rows.empty:
        return "\n".join([*lines, "No rows were produced.", ""])
    stable = rows[rows["stable_shortlist"].fillna(False).astype(bool)].copy()
    lines.extend(
        [
            f"- Rows scored: `{len(rows)}`",
            f"- Stable shortlist rows: `{len(stable)}`",
            "",
            "## Plain English Read",
            "",
        ]
    )
    if stable.empty:
        lines.append("No feature passed the full stability screen. Use the ranked CSV as exploratory only.")
    else:
        grouped = stable.groupby(["target", "source_detail"]).size().sort_values(ascending=False).head(12)
        lines.append("Shortlist clusters:")
        for (target, detail), count in grouped.items():
            lines.append(f"- `{target}`: `{count}` stable rows from `{detail}`.")
    lines.extend(["", "## Top Stable Features", ""])
    display = stable.head(top) if not stable.empty else rows.head(top)
    header = "| Target | Source | Feature Meaning | Direction | Lift | AUC | Tree AUC | Read |"
    lines.extend([header, "|---|---|---|---|---:|---:|---:|---|"])
    for _, row in display.iterrows():
        meaning = str(row.get("plain_english", "") or row.get("base_column", "")).replace("|", "/")
        read = plain_read(row)
        lines.append(
            f"| `{row.get('target')}` | `{row.get('source_detail')}` | {meaning[:130]} | "
            f"`{row.get('orientation')}` | {fmt(row.get('lift_vs_base'))} | {fmt(row.get('oriented_auc'))} | "
            f"{fmt(row.get('tree_test_auc'))} | {read} |"
        )
    return "\n".join(lines) + "\n"


def plain_read(row: pd.Series) -> str:
    if bool(row.get("stable_shortlist", False)):
        return "Worth turning into a hypothesis."
    if bool(row.get("univariate_pass", False)):
        return "Interesting, but tree model did not strongly confirm it."
    return "Exploratory only."


def is_binary(series: Series) -> bool:
    values = set(pd.to_numeric(series.dropna(), errors="coerce").dropna().unique())
    return bool(values) and values.issubset({0, 1, 0.0, 1.0})


def top_quantile_mask(series: Series, orientation: str, quantile: float) -> Series:
    if orientation == "higher_value_more_likely":
        threshold = series.quantile(1.0 - quantile)
        return series.ge(threshold)
    threshold = series.quantile(quantile)
    return series.le(threshold)


def random_control_rate(y_all: Series, valid: Series, count: int, seed_text: str) -> float:
    eligible = list(valid[valid].index)
    if count <= 0 or not eligible:
        return np.nan
    rng = np.random.default_rng(stable_seed(seed_text))
    chosen = rng.choice(eligible, size=min(count, len(eligible)), replace=False)
    return float(y_all.loc[chosen].mean())


def monthly_stability(months: Series, y: Series, x: Series, orientation: str) -> dict[str, Any]:
    windows = 0
    positive = 0
    aucs: list[float] = []
    for month in sorted(months.dropna().unique()):
        mask = months.eq(month)
        if int(mask.sum()) < 80:
            continue
        my = y[mask]
        mx = x[mask]
        if my.nunique() < 2 or mx.nunique() < 2:
            continue
        auc = float(roc_auc_score(my, mx))
        oriented = auc if orientation == "higher_value_more_likely" else 1.0 - auc
        windows += 1
        positive += int(oriented > 0.5)
        aucs.append(oriented)
    return {
        "monthly_windows": windows,
        "monthly_positive_windows": positive,
        "monthly_positive_ratio": positive / windows if windows else np.nan,
        "monthly_oriented_auc_mean": float(np.mean(aucs)) if aucs else np.nan,
    }


def shuffled_oriented_auc(y: Series, x: Series, seed_text: str) -> float:
    shuffled = y.sample(frac=1.0, random_state=stable_seed("shuffle-" + seed_text)).reset_index(drop=True)
    values = x.reset_index(drop=True)
    if shuffled.nunique() < 2 or values.nunique() < 2:
        return np.nan
    auc = float(roc_auc_score(shuffled, values))
    return max(auc, 1.0 - auc)


def latest_file(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True) if root.exists() else []
    return matches[0] if matches else None


def infer_dictionary_path(candidates: Path | None) -> Path | None:
    if candidates is None:
        return None
    name = candidates.name.replace(".parquet", "_dictionary.csv")
    return candidates.with_name(name)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def safe_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    return clean[:120] or "latest"


def stable_seed(text: str) -> int:
    digest = hashlib.sha256(str(text).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())

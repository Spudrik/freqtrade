from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.trader_confluence_feature_taxonomy import (  # noqa: E402
    coarse_source_family,
    source_detail,
)


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT = USER_DATA_DIR / "research_news_data" / "context_features" / "confluence_cache" / "trader_confluence_1h_latest.parquet"
DEFAULT_OUTPUT_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "feature_discovery"
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
    parser = argparse.ArgumentParser(description="Build controlled feature-discovery candidates from the frozen trader-confluence snapshot.")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--max-base-columns", type=int, default=520)
    parser.add_argument("--max-base-per-source-detail", type=int, default=45)
    parser.add_argument("--min-non-null", type=int, default=500)
    parser.add_argument("--min-unique", type=int, default=3)
    parser.add_argument("--include-context", action="store_true", help="Include context columns. Default keeps context enabled only when present rows are non-sparse.")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    plan = {
        "mode": "setup-only unless --execute is supplied",
        "snapshot": str(args.snapshot),
        "output_dir": str(args.output_dir),
        "max_base_columns": int(args.max_base_columns),
        "max_base_per_source_detail": int(args.max_base_per_source_detail),
        "min_non_null": int(args.min_non_null),
        "min_unique": int(args.min_unique),
        "purpose": "Generate broad but controlled feature variants for discovery before hand-authored hypotheses.",
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.snapshot.exists():
        raise FileNotFoundError(args.snapshot)

    frame = pd.read_parquet(args.snapshot)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    selected = select_base_columns(
        frame,
        max_total=int(args.max_base_columns),
        max_per_detail=int(args.max_base_per_source_detail),
        min_non_null=int(args.min_non_null),
        min_unique=int(args.min_unique),
    )
    candidates, dictionary = build_candidates(frame, selected)
    for target in DEFAULT_TARGETS:
        if target in frame:
            candidates[target] = pd.to_numeric(frame[target], errors="coerce")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = safe_name(args.tag)
    parquet_path = args.output_dir / f"feature_discovery_candidates_{safe_tag}.parquet"
    dictionary_path = args.output_dir / f"feature_discovery_candidates_{safe_tag}_dictionary.csv"
    meta_path = args.output_dir / f"feature_discovery_candidates_{safe_tag}_meta.json"
    candidates.to_parquet(parquet_path, index=False)
    dictionary.to_csv(dictionary_path, index=False)
    meta = {
        **plan,
        "snapshot_rows": int(len(frame)),
        "selected_base_columns": int(len(selected)),
        "candidate_rows": int(len(candidates)),
        "candidate_columns": int(len(candidates.columns)),
        "feature_columns": int(sum(column.startswith("fd__") for column in candidates.columns)),
        "outputs": {"parquet": str(parquet_path), "dictionary": str(dictionary_path)},
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"parquet": str(parquet_path), "dictionary": str(dictionary_path), "meta": str(meta_path), "rows": int(len(candidates)), "columns": int(len(candidates.columns))}, indent=2))
    return 0


def select_base_columns(
    frame: DataFrame,
    *,
    max_total: int,
    max_per_detail: int,
    min_non_null: int,
    min_unique: int,
) -> list[str]:
    numeric = frame.select_dtypes(include=["number", "bool"]).columns
    rows: list[dict[str, Any]] = []
    for column in numeric:
        if not usable_base_column(column):
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        non_null = int(series.notna().sum())
        if non_null < min_non_null:
            continue
        unique = int(series.nunique(dropna=True))
        if unique < min_unique:
            continue
        variance = float(series.replace([np.inf, -np.inf], np.nan).var(skipna=True))
        if not np.isfinite(variance) or variance <= 0.0:
            continue
        detail = source_detail(column)
        family = coarse_source_family(column)
        importance_hint = base_importance_hint(column, detail)
        rows.append(
            {
                "column": column,
                "source_family": family,
                "source_detail": detail,
                "non_null": non_null,
                "unique": unique,
                "variance": variance,
                "importance_hint": importance_hint,
            }
        )
    if not rows:
        return []
    table = pd.DataFrame(rows)
    table = table.sort_values(["source_detail", "importance_hint", "non_null", "unique"], ascending=[True, False, False, False])
    capped = table.groupby("source_detail", group_keys=False).head(max_per_detail)
    capped = capped.sort_values(["importance_hint", "non_null", "unique"], ascending=[False, False, False]).head(max_total)
    return list(capped["column"])


def usable_base_column(column: str) -> bool:
    lower = column.lower()
    detail = source_detail(lower)
    family = coarse_source_family(lower)
    if family in {"target_label", "metadata", "quality_flag", "time"}:
        return False
    if detail in {"target_label", "context_availability_metadata", "structure_availability", "quality_flag", "metadata"}:
        return False
    if lower in {"open", "high", "low", "close"}:
        return False
    if lower.startswith("future_") or lower.startswith(("hit_", "time_to_")) or lower.endswith(("_next_6h", "_next_24h")):
        return False
    if lower.startswith("conf_") and (
        "_cmp_" in lower
        or lower.endswith(("_setup", "_trigger", "_score", "_component_score", "_component_count"))
    ):
        return False
    if "source_future_violation" in lower or "available_at" in lower:
        return False
    return True


def base_importance_hint(column: str, detail: str) -> int:
    lower = column.lower()
    score = 0
    for token in (
        "breakout",
        "breakdown",
        "bos",
        "choch",
        "pressure",
        "volume",
        "z_",
        "distance",
        "removed",
        "vacuum",
        "absorption",
        "severity",
        "intensity",
        "confluence",
        "persistence",
        "value_area",
        "lvn",
        "hvn",
        "tlv2",
    ):
        if token in lower:
            score += 1
    if detail.startswith(("structure_", "orderbook_", "context_topic", "context_global")):
        score += 1
    return score


def build_candidates(frame: DataFrame, base_columns: list[str]) -> tuple[DataFrame, DataFrame]:
    out = DataFrame({"date": frame["date"]})
    dictionary_rows: list[dict[str, Any]] = []
    new_columns: dict[str, Series] = {}
    for column in base_columns:
        series = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        transforms = transform_series(series)
        for transform_name, transformed in transforms.items():
            feature_name = f"fd__{safe_name(column)}__{transform_name}"
            new_columns[feature_name] = transformed.astype("float32")
            dictionary_rows.append(
                {
                    "feature": feature_name,
                    "base_column": column,
                    "transform": transform_name,
                    "source_family": coarse_source_family(column),
                    "source_detail": source_detail(column),
                    "plain_english": describe_transform(column, transform_name),
                }
            )
    if new_columns:
        out = pd.concat([out, pd.DataFrame(new_columns)], axis=1)
    dictionary = pd.DataFrame(dictionary_rows)
    return out, dictionary


def transform_series(series: Series) -> dict[str, Series]:
    transforms: dict[str, Series] = {"level": series}
    unique = series.nunique(dropna=True)
    if unique <= 2:
        transforms["activity_24h"] = series.fillna(0.0).rolling(24, min_periods=6).sum()
        transforms["activity_72h"] = series.fillna(0.0).rolling(72, min_periods=12).sum()
        return transforms
    transforms["delta_3h"] = series.diff(3)
    transforms["delta_24h"] = series.diff(24)
    transforms["z_168h"] = zscore(series, 168, min_periods=48)
    transforms["pct_rank_720h"] = series.rolling(720, min_periods=120).rank(pct=True)
    transforms["above_mean_24h"] = series.gt(series.rolling(24, min_periods=8).mean()).astype(float).where(series.notna())
    return transforms


def zscore(series: Series, window: int, *, min_periods: int) -> Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std().replace(0.0, np.nan)
    return (series - mean) / std


def describe_transform(column: str, transform: str) -> str:
    labels = {
        "level": "current value",
        "delta_3h": "change over the last 3 hours",
        "delta_24h": "change over the last 24 hours",
        "z_168h": "how abnormal the value is versus roughly the last week",
        "pct_rank_720h": "where the value sits versus roughly the last month",
        "above_mean_24h": "whether the value is above its recent 24h mean",
        "activity_24h": "how often the flag appeared in the last 24 hours",
        "activity_72h": "how often the flag appeared in the last 72 hours",
    }
    return f"{labels.get(transform, transform)} for {column}"


def safe_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    return clean[:150] or "value"


if __name__ == "__main__":
    raise SystemExit(main())

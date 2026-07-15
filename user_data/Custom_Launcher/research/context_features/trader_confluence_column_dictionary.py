from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.trader_confluence_feature_taxonomy import (
    coarse_source_family,
    source_detail,
)


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT = USER_DATA_DIR / "research_news_data" / "context_features" / "confluence_cache" / "trader_confluence_1h_latest.parquet"
DEFAULT_OUTPUT_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "confluence_cache"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a readable column dictionary for the trader-confluence 1h snapshot.")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    plan = {
        "mode": "setup-only unless --execute is supplied",
        "snapshot": str(args.snapshot),
        "outputs": {
            "csv": str(args.output_dir / "trader_confluence_column_dictionary.csv"),
            "markdown": str(args.output_dir / "trader_confluence_column_dictionary.md"),
            "summary": str(args.output_dir / "trader_confluence_column_dictionary_summary.json"),
        },
        "purpose": "Make every model/report column explainable by source family, role, coverage, and rough semantics.",
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.snapshot.exists():
        raise FileNotFoundError(args.snapshot)

    frame = pd.read_parquet(args.snapshot)
    dictionary = build_dictionary(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "trader_confluence_column_dictionary.csv"
    md_path = args.output_dir / "trader_confluence_column_dictionary.md"
    summary_path = args.output_dir / "trader_confluence_column_dictionary_summary.json"
    dictionary.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown(dictionary, frame), encoding="utf-8")
    summary = {
        "snapshot": str(args.snapshot),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "families": dictionary["family"].value_counts().sort_index().to_dict(),
        "roles": dictionary["role"].value_counts().sort_index().to_dict(),
        "low_coverage_columns_under_1pct": int(dictionary["non_null_ratio"].lt(0.01).sum()),
        "all_null_columns": dictionary.loc[dictionary["non_null_ratio"].eq(0.0), "column"].tolist(),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "markdown": str(md_path), "summary": str(summary_path), "columns": int(len(dictionary))}, indent=2))
    return 0


def build_dictionary(frame: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    total = max(1, len(frame))
    for column in frame.columns:
        series = frame[column]
        numeric = pd.api.types.is_numeric_dtype(series)
        non_null = int(series.notna().sum())
        item: dict[str, Any] = {
            "column": column,
            "source_family": coarse_source_family(column),
            "source_detail": source_detail(column),
            "family": source_detail(column),
            "role": role_for_column(column),
            "dtype": str(series.dtype),
            "is_numeric": bool(numeric),
            "non_null_rows": non_null,
            "non_null_ratio": float(non_null / total),
            "non_zero_ratio": np.nan,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "description": description_for_column(column),
        }
        if numeric:
            values = pd.to_numeric(series, errors="coerce")
            item["non_zero_ratio"] = float(values.fillna(0.0).ne(0.0).sum() / total)
            item["min"] = float(values.min()) if values.notna().any() else np.nan
            item["max"] = float(values.max()) if values.notna().any() else np.nan
            item["mean"] = float(values.mean()) if values.notna().any() else np.nan
        rows.append(item)
    return pd.DataFrame(rows).sort_values(["source_family", "source_detail", "role", "column"]).reset_index(drop=True)


def role_for_column(column: str) -> str:
    lower = column.lower()
    if lower in {"date"} or lower.endswith("_generated_at") or "source_" in lower:
        return "timestamp_metadata"
    if "present" in lower or "coverage" in lower or "violation" in lower or "missing" in lower:
        return "quality_flag"
    if lower.startswith("px_"):
        if "volume" in lower:
            return "volume"
        if "range" in lower or "compression" in lower:
            return "range_compression"
        return "price_action"
    if "return" in lower or "drawdown" in lower or "upside" in lower or "breakout_success" in lower or "breakdown_success" in lower or "fakeout" in lower or "time_to_" in lower or "hit_" in lower:
        return "target_or_path_label"
    if "vp_" in lower or "value_area" in lower or "poc" in lower or "vah" in lower or "val" in lower or "hvn" in lower or "lvn" in lower:
        return "volume_profile"
    if "tlv2" in lower or "support" in lower or "resistance" in lower:
        return "support_resistance"
    if "bos" in lower or "choch" in lower or "higher_" in lower or "lower_" in lower or "structure" in lower:
        return "market_structure"
    if "pressure" in lower or "imbalance" in lower or "microprice" in lower:
        return "pressure_flow"
    if "wall" in lower or "zone" in lower or "liquidity" in lower or "vacuum" in lower or "absorption" in lower:
        return "orderbook_liquidity"
    if "article" in lower or "topic" in lower or "severity" in lower or "confluence" in lower or "persistence" in lower or "gdelt" in lower or "gkg" in lower:
        return "context_event"
    if "google_trends" in lower or "fred" in lower or "fear_greed" in lower or "equity" in lower or "gold" in lower or "dxy" in lower:
        return "macro_global"
    if "volume" in lower:
        return "volume"
    if "range" in lower or "compression" in lower:
        return "range_compression"
    return "general_numeric"


def description_for_column(column: str) -> str:
    if column.startswith("conf_") and column.endswith("_score"):
        return "Trader-readable confluence score built from setup, trigger, and source-family confirmation components."
    if column.startswith("conf_") and column.endswith("_setup"):
        return "Current precondition mask for a named trader hypothesis."
    if column.startswith("conf_") and column.endswith("_trigger"):
        return "Current trigger mask for a named trader hypothesis."
    if column.endswith("_present"):
        return "Source availability flag after timestamp and coverage checks."
    if column.endswith("_source_future_violation"):
        return "Lookahead validation flag; should remain zero before modelling."
    if column.startswith("future_") or column.endswith("_next_6h") or column.endswith("_next_24h"):
        return "Future label for research scoring only; never use as a model feature."
    if column.startswith("ctx_"):
        return "Timestamp-aligned context/news/macro feature from the external context builder."
    if column.startswith("ob_"):
        return "Timestamp-aligned Bybit orderbook trader-state feature for the named market."
    if column.startswith("st_"):
        return "Prebuilt custom structural indicator feature from VP, TLV2, BOS/CHoCH, or pattern tools."
    if column.startswith("px_"):
        return "Derived price/volume baseline feature built from past and current OHLCV only."
    return ""


def to_markdown(dictionary: DataFrame, frame: DataFrame) -> str:
    lines = [
        "# Trader Confluence Column Dictionary",
        "",
        "This file is generated from the latest confluence snapshot. It is a readability aid for agents and reports; detailed numeric results stay in CSV/JSON reports.",
        "",
        f"- Snapshot rows: `{len(frame)}`",
        f"- Snapshot columns: `{len(frame.columns)}`",
        "",
        "## Family Counts",
        "",
    ]
    lines.append("### Coarse Source Families")
    lines.append("")
    for family, count in dictionary["source_family"].value_counts().sort_index().items():
        lines.append(f"- `{family}`: `{int(count)}`")
    lines.extend(["", "### Source Detail Groups", ""])
    for detail, count in dictionary["source_detail"].value_counts().sort_index().items():
        lines.append(f"- `{detail}`: `{int(count)}`")
    lines.extend(["", "## Columns", ""])
    preview_columns = ["column", "source_family", "source_detail", "role", "non_null_ratio", "non_zero_ratio", "description"]
    for detail, group in dictionary.groupby("source_detail", sort=True):
        lines.extend([f"### {detail}", ""])
        lines.append(group[preview_columns].to_markdown(index=False))
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from trading_lead_confluence_checks import build_filters, confluence_combos  # noqa: E402


USER_DATA_DIR = Path(__file__).resolve().parents[3]
REPORTS_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"
DEFAULT_FEATURES = USER_DATA_DIR / "research_news_data" / "context_features" / "confluence_cache" / "trader_confluence_1h_latest.parquet"


DEFAULT_FILTERS = (
    "compression_plus_range_break",
    "regime_plus_range_break",
    "range_break_agrees",
    "vp_plus_range_break",
    "structure_trigger_agrees",
    "strong_volume_plus_structure",
    "orderbook_plus_structure",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export reusable signal parquets after applying named confluence filters.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--filters", nargs="*", default=list(DEFAULT_FILTERS))
    parser.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    args = parser.parse_args()

    if not args.features.exists():
        raise FileNotFoundError(args.features)
    if not args.signals.exists():
        raise FileNotFoundError(args.signals)

    features = load_features(args.features)
    signals = load_signals(args.signals)
    merged = signals.merge(features, on="date", how="left", suffixes=("", "_feature"))
    masks = build_named_masks(merged)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = safe_name(args.tag)
    rows: list[dict[str, Any]] = []
    for filter_id in args.filters:
        if filter_id not in masks:
            rows.append({"filter_id": filter_id, "status": "missing_filter", "signal_rows": 0, "path": ""})
            continue
        filtered = signals[masks[filter_id].reindex(signals.index).fillna(False).astype(bool)].copy()
        path = args.output_dir / f"trading_lead_signals_{tag}_{safe_name(filter_id)}.parquet"
        write_signals(filtered, path)
        rows.append({"filter_id": filter_id, "status": "exported", "signal_rows": int(len(filtered)), "path": str(path)})

    summary = pd.DataFrame(rows)
    summary_path = args.output_dir / f"trader_confluence_filtered_signal_exports_{tag}.csv"
    md_path = args.output_dir / f"trader_confluence_filtered_signal_exports_{tag}.md"
    meta_path = args.output_dir / f"trader_confluence_filtered_signal_exports_{tag}_meta.json"
    summary.to_csv(summary_path, index=False)
    md_path.write_text(markdown_report(summary, args), encoding="utf-8")
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "features": str(args.features),
        "signals": str(args.signals),
        "input_signals": int(len(signals)),
        "outputs": {"summary": str(summary_path), "markdown": str(md_path)},
        "exports": rows,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


def load_features(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    return frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")


def load_signals(path: Path) -> pd.DataFrame:
    signals = pd.read_parquet(path)
    signals["date"] = pd.to_datetime(signals["date"], utc=True, errors="coerce")
    signals = signals.dropna(subset=["date"]).sort_values(["date", "rule_id"]).drop_duplicates(["date", "rule_id"], keep="first").reset_index(drop=True)
    signals["direction"] = signals["direction"].astype(str)
    return signals


def build_named_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {}
    for spec in build_filters(frame):
        masks[spec.filter_id] = spec.mask_builder(frame).fillna(False).astype(bool)
    for combo_id, parts, _question in confluence_combos():
        if all(part in masks for part in parts):
            mask = pd.Series(True, index=frame.index)
            for part in parts:
                mask &= masks[part]
            masks[combo_id] = mask.fillna(False).astype(bool)
    return masks


def write_signals(signals: pd.DataFrame, path: Path) -> None:
    columns = ["date", "enter_long", "enter_short", "enter_tag", "rule_id", "concept_id", "threshold", "signal_score", "direction", "signal_source"]
    out = signals.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = 0 if column.startswith("enter_") else ""
    out = out[columns].sort_values(["date", "rule_id"]).drop_duplicates(["date", "rule_id"], keep="first").reset_index(drop=True)
    out.to_parquet(path, index=False)


def markdown_report(summary: pd.DataFrame, args: argparse.Namespace) -> str:
    lines = [
        "# Confluence Filtered Signal Exports",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Purpose: turn promising confluence filters into reusable signal parquets for exit research and later strategy validation.",
        "",
        f"Input signals: `{args.signals}`",
        "",
    ]
    lines.append(summary.to_markdown(index=False))
    return "\n".join(lines) + "\n"


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)).strip("_")[:140] or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())

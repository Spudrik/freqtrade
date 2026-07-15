from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.source_coverage_audit import build_source_audit  # noqa: E402
from user_data.Custom_Launcher.research.context_features.trader_confluence_direct_tests import (  # noqa: E402
    required_source_details,
)
from user_data.Custom_Launcher.research.context_features.trader_confluence_hypotheses import hypothesis_by_id  # noqa: E402


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT = USER_DATA_DIR / "research_news_data" / "context_features" / "confluence_cache" / "trader_confluence_1h_latest.parquet"
DEFAULT_SUMMARY = USER_DATA_DIR / "research_news_data" / "context_features" / "reports" / "trader_confluence_direct_tests_source_detail_gated_summary.csv"
DEFAULT_OUTPUT_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a source-detail preflight artifact before promoting a trader hypothesis into FreqAI.")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--direct-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--hypothesis-id", default="macro_context_liquidity_stress_breakdown")
    parser.add_argument("--target", default="large_drawdown_next_6h")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--min-setup-rows", type=int, default=2000)
    parser.add_argument("--min-trigger-rows", type=int, default=75)
    parser.add_argument("--min-trigger-positives", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag.strip()) or "latest"
    plan = {
        "mode": "setup-only unless --execute is supplied",
        "snapshot": str(args.snapshot),
        "direct_summary": str(args.direct_summary),
        "hypothesis_id": args.hypothesis_id,
        "target": args.target,
        "outputs": {
            "json": str(args.output_dir / f"trader_confluence_freqai_preflight_{safe_tag}.json"),
            "markdown": str(args.output_dir / f"trader_confluence_freqai_preflight_{safe_tag}.md"),
        },
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.snapshot.exists():
        raise FileNotFoundError(args.snapshot)
    if not args.direct_summary.exists():
        raise FileNotFoundError(args.direct_summary)

    frame = pd.read_parquet(args.snapshot)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    summary = pd.read_csv(args.direct_summary)
    result = build_preflight(
        frame,
        summary,
        hypothesis_id=args.hypothesis_id,
        target=args.target,
        min_setup_rows=int(args.min_setup_rows),
        min_trigger_rows=int(args.min_trigger_rows),
        min_trigger_positives=int(args.min_trigger_positives),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"trader_confluence_freqai_preflight_{safe_tag}.json"
    md_path = args.output_dir / f"trader_confluence_freqai_preflight_{safe_tag}.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown(result), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "verdict": result["verdict"]}, indent=2))
    return 0 if result["verdict"] in {"pilot_ready", "conditional_pilot"} else 2


def build_preflight(
    frame: DataFrame,
    summary: DataFrame,
    *,
    hypothesis_id: str,
    target: str,
    min_setup_rows: int,
    min_trigger_rows: int,
    min_trigger_positives: int,
) -> dict[str, Any]:
    hypotheses = hypothesis_by_id()
    if hypothesis_id not in hypotheses:
        return {"verdict": "blocked", "errors": [f"unknown hypothesis_id {hypothesis_id}"]}
    hypothesis = hypotheses[hypothesis_id]
    required = required_source_details(hypothesis)
    _, masks = build_source_audit(frame, 0.01)
    usable = pd.Series(True, index=frame.index)
    missing_blocks: list[str] = []
    block_rows: dict[str, int] = {}
    for detail in required:
        detail_mask = masks.get(detail, {}).get("usable")
        if detail_mask is None:
            missing_blocks.append(detail)
            usable &= False
            block_rows[detail] = 0
            continue
        detail_mask = detail_mask.reindex(frame.index).fillna(False)
        block_rows[detail] = int(detail_mask.sum())
        usable &= detail_mask
    setup = usable & numeric(frame, hypothesis.setup_column).fillna(0.0).gt(0.0)
    trigger = setup & numeric(frame, hypothesis.trigger_column).fillna(0.0).gt(0.0)
    target_values = numeric(frame, target)
    trigger_target = target_values[trigger & target_values.notna()]
    setup_target = target_values[setup & target_values.notna()]
    row = direct_summary_row(summary, hypothesis_id, target)
    checks = {
        "required_blocks_present": not missing_blocks,
        "setup_rows_min": int(setup.sum()) >= min_setup_rows,
        "trigger_rows_min": int(trigger.sum()) >= min_trigger_rows,
        "trigger_positives_min": int(trigger_target.eq(1.0).sum()) >= min_trigger_positives if is_binary(trigger_target) else True,
        "direct_summary_exists": bool(row),
        "auc_min": float_or_nan(row.get("roc_auc")) >= 0.55 if row else False,
        "beats_shuffled": (float_or_nan(row.get("roc_auc")) - float_or_nan(row.get("shuffled_roc_auc"))) >= 0.02 if row else False,
        "beats_baseline": (float_or_nan(row.get("roc_auc")) - float_or_nan(row.get("baseline_roc_auc"))) >= 0.01 if row else False,
        "watchlist_positive": bool(row.get("watchlist_positive")) if row else False,
    }
    errors = [name for name, passed in checks.items() if not passed]
    if errors:
        verdict = "blocked"
    elif int(trigger_target.eq(1.0).sum()) < 30:
        verdict = "conditional_pilot"
    else:
        verdict = "pilot_ready"
    return {
        "schema_version": 1,
        "hypothesis_id": hypothesis_id,
        "target": target,
        "verdict": verdict,
        "failed_checks": errors,
        "required_source_detail_blocks": list(required),
        "source_detail_usable_rows": block_rows,
        "usable_intersection_rows": int(usable.sum()),
        "usable_start": iso(frame.loc[usable, "date"].min()) if usable.any() else "",
        "usable_end": iso(frame.loc[usable, "date"].max()) if usable.any() else "",
        "setup_rows": int(setup.sum()),
        "trigger_rows": int(trigger.sum()),
        "setup_target_rows": int(len(setup_target)),
        "trigger_target_rows": int(len(trigger_target)),
        "trigger_positive_rows": int(trigger_target.eq(1.0).sum()) if is_binary(trigger_target) else None,
        "trigger_event_rate": float(trigger_target.mean()) if len(trigger_target) and is_binary(trigger_target) else None,
        "direct_summary": row,
        "checks": checks,
        "notes": [
            "This is a preflight artifact, not a model result.",
            "FreqAI queueing should remain blocked unless verdict is conditional_pilot or pilot_ready.",
            "conditional_pilot means useful enough to test, but not enough to claim promotion.",
        ],
    }


def direct_summary_row(summary: DataFrame, hypothesis_id: str, target: str) -> dict[str, Any]:
    if summary.empty:
        return {}
    rows = summary[(summary["hypothesis_id"].astype(str) == hypothesis_id) & (summary["target"].astype(str) == target)]
    if rows.empty:
        return {}
    return {key: json_value(value) for key, value in rows.iloc[0].to_dict().items()}


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Trader Confluence FreqAI Preflight",
        "",
        f"- Hypothesis: `{result.get('hypothesis_id')}`",
        f"- Target: `{result.get('target')}`",
        f"- Verdict: `{result.get('verdict')}`",
        f"- Failed checks: `{', '.join(result.get('failed_checks') or []) or 'none'}`",
        "",
        "## Source Blocks",
        "",
    ]
    for block in result.get("required_source_detail_blocks", []):
        rows = result.get("source_detail_usable_rows", {}).get(block, 0)
        lines.append(f"- `{block}`: `{rows}` usable rows")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            f"- Usable intersection rows: `{result.get('usable_intersection_rows')}`",
            f"- Usable range: `{result.get('usable_start')}` to `{result.get('usable_end')}`",
            f"- Setup rows: `{result.get('setup_rows')}`",
            f"- Trigger rows: `{result.get('trigger_rows')}`",
            f"- Trigger positive rows: `{result.get('trigger_positive_rows')}`",
            f"- Trigger event rate: `{result.get('trigger_event_rate')}`",
            "",
            "## Direct Summary",
            "",
        ]
    )
    row = result.get("direct_summary") or {}
    for key in ("roc_auc", "baseline_roc_auc", "shuffled_roc_auc", "average_precision", "rows_trigger", "event_rate_trigger", "watchlist_positive"):
        lines.append(f"- `{key}`: `{row.get(key, '')}`")
    return "\n".join(lines) + "\n"


def numeric(frame: DataFrame, column: str) -> Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def is_binary(series: Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna().unique()
    return len(values) > 0 and set(np.round(values, 8)).issubset({0.0, 1.0})


def float_or_nan(value: Any) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float("nan")
    return parsed


def json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def iso(value: Any) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())

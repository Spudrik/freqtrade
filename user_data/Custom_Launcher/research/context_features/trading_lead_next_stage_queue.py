from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


USER_DATA_DIR = Path(__file__).resolve().parents[3]
REPORTS_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"
DEFAULT_REGISTRY = REPORTS_DIR / "trading_lead_registry_latest.csv"
DEFAULT_CONFLUENCE = REPORTS_DIR / "trading_lead_confluence_checks_20260605_strict.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the next trader-lead research queue.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--confluence", type=Path, default=DEFAULT_CONFLUENCE)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    parser.add_argument("--max-tasks", type=int, default=60)
    args = parser.parse_args()

    registry = pd.read_csv(args.registry)
    confluence = pd.read_csv(args.confluence) if args.confluence.exists() else pd.DataFrame()
    queue = build_queue(registry, confluence, max_tasks=args.max_tasks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = safe_name(args.tag)
    csv_path = args.output_dir / f"trading_lead_next_stage_queue_{tag}.csv"
    md_path = args.output_dir / f"trading_lead_next_stage_queue_{tag}.md"
    meta_path = args.output_dir / f"trading_lead_next_stage_queue_{tag}_meta.json"
    queue.to_csv(csv_path, index=False)
    md_path.write_text(markdown_queue(queue), encoding="utf-8")
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_count": int(len(queue)),
        "task_type_counts": queue["task_type"].value_counts().to_dict(),
        "outputs": {"csv": str(csv_path), "markdown": str(md_path)},
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


def build_queue(registry: pd.DataFrame, confluence: pd.DataFrame, max_tasks: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    priority = 1
    if not confluence.empty:
        watch = confluence[confluence["verdict"].isin(["worth_more_testing", "watchlist"])].copy()
        watch = watch.sort_values(["quality_delta", "filtered_trades"], ascending=[False, False])
        for _, row in watch.head(12).iterrows():
            rows.append(
                {
                    "queue_priority": priority,
                    "task_type": "tailored_confluence_retest",
                    "lead_or_group": row["group_id"],
                    "trader_question": "Retest the filter that improved this exact entry family, then check whether trade count remains usable.",
                    "specific_test": row["filter_id"],
                    "pass_condition": "Filtered version keeps at least 10 trades, improves profit factor or average return, and survives month/window review.",
                    "why_now": f"Watchlist improvement: trades={row['filtered_trades']}, win={row['win_rate']}, PF={row['profit_factor']}, delta={row['quality_delta']}.",
                    "data_scope": "Selected Sieve-derived trades plus confluence cache.",
                    "next_if_pass": "Move to exit-tailoring test for this entry family.",
                    "next_if_fail": "Park this filter and try a different confluence family.",
                }
            )
            priority += 1

    strategy_ready = registry[registry["lead_status"].eq("candidate_for_strategy_research")].copy()
    strategy_ready = strategy_ready.sort_values("quality_score", ascending=False)
    for _, row in strategy_ready.head(16).iterrows():
        rows.append(
            {
                "queue_priority": priority,
                "task_type": "exit_research",
                "lead_or_group": row["lead_id"],
                "trader_question": row["trader_question"],
                "specific_test": row["exit_research_needed"],
                "pass_condition": "Exit variant improves return/drawdown or profit factor without destroying the original edge.",
                "why_now": row["evidence_summary"],
                "data_scope": "Existing signal/trade rows; no live DB reads.",
                "next_if_pass": "Add risk-sizing overlay tests.",
                "next_if_fail": "Keep original exit and test a different exit family.",
            }
        )
        priority += 1

    for _, row in registry.head(36).iterrows():
        rows.append(
            {
                "queue_priority": priority,
                "task_type": "per_lead_confluence",
                "lead_or_group": row["lead_id"],
                "trader_question": row["trader_question"],
                "specific_test": row["confluence_candidates"],
                "pass_condition": "Extra evidence improves quality versus the lead alone and keeps enough rows/trades to be meaningful.",
                "why_now": row["evidence_summary"],
                "data_scope": row["source_detail_blocks"],
                "next_if_pass": "Promote to entry-rule or exit-research stage.",
                "next_if_fail": "Rework one condition at a time or park.",
            }
        )
        priority += 1
        if len(rows) >= max_tasks:
            break
    queue = pd.DataFrame(rows).head(max_tasks)
    return queue


def markdown_queue(queue: pd.DataFrame) -> str:
    lines = [
        "# Trading Lead Next-Stage Queue",
        "",
        "This queue turns the lead registry into concrete next tests. It prioritizes tailored confluence first, then exits for entry families with real trade evidence.",
        "",
    ]
    cols = ["queue_priority", "task_type", "lead_or_group", "specific_test", "pass_condition", "next_if_pass"]
    lines.append(queue[cols].to_markdown(index=False))
    return "\n".join(lines) + "\n"


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)).strip("_")[:120] or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())

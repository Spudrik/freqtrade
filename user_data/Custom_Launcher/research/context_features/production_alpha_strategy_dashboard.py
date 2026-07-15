from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
REPORTS_DIR = ROOT / "user_data" / "research_news_data" / "context_features" / "reports"


LANE_RULES = [
    ("position_management", ("SignalStack", "signal_stack", "stack")),
    ("orderbook_risk", ("Orderbook", "orderbook", "ObExit", "ob_exit", "CrashExit", "crash_exit")),
    ("sparse_high_quality", ("Rare", "rare", "Complete", "complete", "Quality", "quality")),
    ("lower_drawdown", ("Strict", "strict", "NoOvertrade", "no_overtrade", "NoPoc", "no_poc")),
    ("balanced", ("Balanced", "balanced", "NoTiny", "no_tiny")),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create compact production-alpha strategy candidate, decision, and next-test files."
    )
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = safe_name(args.tag)
    paths = sorted(args.reports_dir.glob("production_alpha_*_promotion_gate.csv"))
    candidates = build_candidate_table(paths)
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["candidate_score", "return_pct", "profit_factor"],
            ascending=[False, False, False],
        ).head(args.max_candidates)
    queue = build_next_tests(candidates)

    outputs = {
        "candidate_table_csv": str(args.output_dir / f"production_alpha_strategy_candidates_{tag}.csv"),
        "candidate_table_md": str(args.output_dir / f"production_alpha_strategy_candidates_{tag}.md"),
        "next_tests_csv": str(args.output_dir / f"production_alpha_strategy_next_tests_{tag}.csv"),
        "next_tests_md": str(args.output_dir / f"production_alpha_strategy_next_tests_{tag}.md"),
        "decision_snapshot": str(args.output_dir / "production_alpha_strategy_decision_snapshot.md"),
        "meta": str(args.output_dir / f"production_alpha_strategy_dashboard_{tag}_meta.json"),
    }
    meta = {
        "mode": "setup-only unless --execute is supplied",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reports_scanned": len(paths),
        "candidate_count": int(len(candidates)),
        "next_test_count": int(len(queue)),
        "outputs": outputs,
    }
    if not args.execute:
        print(json.dumps(meta, indent=2))
        return 0

    candidates.to_csv(outputs["candidate_table_csv"], index=False)
    Path(outputs["candidate_table_md"]).write_text(candidate_markdown(candidates), encoding="utf-8")
    queue.to_csv(outputs["next_tests_csv"], index=False)
    Path(outputs["next_tests_md"]).write_text(next_tests_markdown(queue), encoding="utf-8")
    Path(outputs["decision_snapshot"]).write_text(decision_snapshot(candidates, queue), encoding="utf-8")
    Path(outputs["meta"]).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


def build_candidate_table(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in paths:
        promotion_md = path.with_suffix(".md")
        if is_rejected_report(promotion_md):
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if "role" not in frame.columns:
            continue
        audited = frame[frame["role"].eq("audited")].copy()
        if audited.empty:
            continue
        row = audited.iloc[0].to_dict()
        baseline_rows = frame[frame["role"].eq("baseline")]
        baseline = baseline_rows.iloc[0].to_dict() if not baseline_rows.empty else {}
        strategy = str(row.get("strategy") or "")
        return_pct = as_float(row.get("return_pct"))
        drawdown = as_float(row.get("max_drawdown_pct"))
        profit_factor = as_float(row.get("profit_factor"))
        win_rate = as_float(row.get("win_rate_pct"))
        trades = int(as_float(row.get("trades")))
        lane = classify_lane(strategy, path.name, return_pct, drawdown, profit_factor, trades)
        baseline_return = as_float(baseline.get("return_pct")) if baseline else None
        baseline_drawdown = as_float(baseline.get("max_drawdown_pct")) if baseline else None
        baseline_pf = as_float(baseline.get("profit_factor")) if baseline else None
        rows.append(
            {
                "strategy": strategy,
                "lane": lane,
                "use_case": use_case(lane),
                "trades": trades,
                "return_pct": round(return_pct, 3),
                "max_drawdown_pct": round(drawdown, 3),
                "profit_factor": round(profit_factor, 3),
                "win_rate_pct": round(win_rate, 3),
                "return_vs_market_ratio": round(as_float(row.get("return_vs_market_ratio")), 3),
                "baseline_return_delta": round(return_pct - baseline_return, 3) if baseline_return is not None else None,
                "baseline_drawdown_delta": round(drawdown - baseline_drawdown, 3) if baseline_drawdown is not None else None,
                "baseline_pf_delta": round(profit_factor - baseline_pf, 3) if baseline_pf is not None else None,
                "candidate_score": round(candidate_score(return_pct, drawdown, profit_factor, win_rate, trades), 3),
                "verdict": candidate_verdict(return_pct, drawdown, profit_factor, win_rate, trades),
                "next_action": next_action(lane, return_pct, drawdown, profit_factor, trades),
                "source_zip": row.get("source_zip"),
                "promotion_csv": str(path),
                "promotion_md": str(promotion_md),
            }
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame = frame.sort_values("promotion_csv").drop_duplicates("strategy", keep="last")
    frame = apply_relative_lanes(frame)
    return frame


def build_next_tests(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "priority",
                "lane",
                "test_type",
                "strategy",
                "trader_question",
                "pass_condition",
                "read_first",
            ]
        )
    rows: list[dict[str, object]] = []
    priority = 1

    def add(candidate: pd.Series, test_type: str, question: str, pass_condition: str) -> None:
        nonlocal priority
        rows.append(
            {
                "priority": priority,
                "lane": candidate["lane"],
                "test_type": test_type,
                "strategy": candidate["strategy"],
                "trader_question": question,
                "pass_condition": pass_condition,
                "read_first": candidate["promotion_md"],
            }
        )
        priority += 1

    lane_order = [
        "position_management",
        "balanced",
        "high_profit_factor",
        "lower_drawdown",
        "sparse_high_quality",
        "orderbook_risk",
        "max_return",
    ]
    for lane in lane_order:
        lane_frame = candidates[candidates["lane"].eq(lane)].sort_values(
            ["candidate_score", "profit_factor"], ascending=[False, False]
        )
        if lane_frame.empty:
            continue
        candidate = lane_frame.iloc[0]
        if lane == "position_management":
            add(
                candidate,
                "smaller_add_sizes",
                "Do smaller same-direction adds keep the stacked-trade quality while reducing reserved-capital drag?",
                "Improve drawdown or profit factor while preserving most return and at least 25 add trades.",
            )
            add(
                candidate,
                "profitable_only_add",
                "Does adding only after the open trade is already profitable improve risk-adjusted performance?",
                "Improve profit factor or drawdown without cutting stacked-trade count below useful levels.",
            )
            add(
                candidate,
                "different_family_add",
                "Is a later same-direction signal stronger when it comes from a different entry family/source?",
                "Stacked subset remains high quality and whole strategy improves versus the same branch.",
            )
        elif lane == "balanced":
            add(
                candidate,
                "failure_family_audit",
                "Which family is still causing the worst drawdown in the balanced branch?",
                "Find one concrete family-specific change to test next, not a broad exit.",
            )
        elif lane == "high_profit_factor":
            add(
                candidate,
                "quality_preservation",
                "Can this high-profit-factor branch keep its clean trade profile while adding only closely related trades?",
                "Keep profit factor above 3 while increasing trade count or preserving low drawdown.",
            )
        elif lane == "lower_drawdown":
            add(
                candidate,
                "drawdown_cluster_replay",
                "Can we explain the worst drawdown cluster in trader terms before changing the strategy?",
                "Produce a targeted exit/filter hypothesis tied to the actual losing cluster.",
            )
        elif lane == "sparse_high_quality":
            add(
                candidate,
                "standalone_or_sidecar",
                "Should this sparse high-quality branch run as a sidecar rather than be merged into a broad block?",
                "Keep high profit factor and enough yearly distribution to justify sidecar testing.",
            )
        elif lane == "orderbook_risk":
            add(
                candidate,
                "invalidation_specificity",
                "Can orderbook risk exits be narrowed to exact invalidation states instead of broad warnings?",
                "Reduce losses more than winners and improve drawdown without large return damage.",
            )
        elif lane == "max_return":
            add(
                candidate,
                "worst_family_filter",
                "Can the highest-return branch remove or filter one explainable weak family without losing its edge?",
                "Improve drawdown or profit factor while keeping return competitive.",
            )
        add(
            candidate,
            "long_short_ratio_gate",
            "Is the branch reasonably long/short balanced, or should it be treated as a specialist?",
            "Long/short trade count is not more imbalanced than 80/20, or the branch is explicitly documented as specialist.",
        )
    return pd.DataFrame(rows)


def classify_lane(strategy: str, filename: str, return_pct: float, drawdown: float, profit_factor: float, trades: int) -> str:
    text = f"{strategy} {filename}"
    for lane, terms in LANE_RULES[:3]:
        if any(term in text for term in terms):
            return lane
    if profit_factor >= 3.0 and trades >= 50:
        return "high_profit_factor"
    if drawdown <= 5.0 and return_pct >= 1000:
        return "lower_drawdown"
    for lane, terms in LANE_RULES[3:]:
        if any(term in text for term in terms):
            return lane
    return "max_return"


def apply_relative_lanes(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "return_pct" not in frame.columns:
        return frame
    out = frame.copy()
    non_special = ~out["lane"].isin(["position_management", "orderbook_risk", "sparse_high_quality"])
    if non_special.any():
        max_return = float(out.loc[non_special, "return_pct"].max())
        max_mask = non_special & out["return_pct"].ge(max_return * 0.98)
        out.loc[max_mask, "lane"] = "max_return"
        balanced_mask = non_special & ~max_mask & out["return_pct"].ge(2500) & out["max_drawdown_pct"].le(8)
        out.loc[balanced_mask, "lane"] = "balanced"
        quality_mask = non_special & ~max_mask & ~balanced_mask & out["profit_factor"].ge(3.0)
        out.loc[quality_mask, "lane"] = "high_profit_factor"
    out["use_case"] = out["lane"].map(use_case)
    return out


def use_case(lane: str) -> str:
    return {
        "position_management": "Add/hold/reduce logic and risk-quality testing.",
        "orderbook_risk": "Crash/invalidation warning and defensive exits.",
        "sparse_high_quality": "Sidecar candidate with fewer but cleaner trades.",
        "high_profit_factor": "Cleaner quality branch where profit factor and win rate matter more than max return.",
        "lower_drawdown": "Defensive or balanced branch where lower downside matters.",
        "balanced": "Main candidate when return, risk, and trade count are all important.",
        "max_return": "Aggressive return benchmark; not automatically the final answer.",
    }.get(lane, "Unclassified research branch.")


def candidate_verdict(return_pct: float, drawdown: float, profit_factor: float, win_rate: float, trades: int) -> str:
    if trades < 20:
        return "sparse_watchlist"
    if drawdown > 20:
        return "risk_rework"
    if profit_factor >= 3.0 and win_rate >= 58 and trades >= 50:
        return "quality_candidate"
    if return_pct >= 3000 and drawdown <= 7 and profit_factor >= 2.0:
        return "strong_candidate"
    if profit_factor >= 2.0 and drawdown <= 10:
        return "watchlist"
    return "rework_or_park"


def next_action(lane: str, return_pct: float, drawdown: float, profit_factor: float, trades: int) -> str:
    if drawdown > 20:
        return "Audit drawdown cluster before any merge."
    if lane == "position_management":
        return "Test smaller/profitable-only/different-family add variants."
    if lane == "orderbook_risk":
        return "Narrow orderbook invalidation to exact failure states."
    if lane == "sparse_high_quality":
        return "Check standalone sidecar viability and yearly distribution."
    if trades < 50:
        return "Increase sample or keep as sidecar only."
    if profit_factor >= 3.0:
        return "Preserve as high-quality branch; test robustness."
    if return_pct >= 3000:
        return "Audit worst families and keep as main candidate lane."
    return "Rework one condition at a time."


def candidate_score(return_pct: float, drawdown: float, profit_factor: float, win_rate: float, trades: int) -> float:
    trade_score = min(trades, 350) / 350.0
    return_score = min(max(return_pct, 0.0), 5000.0) / 5000.0
    dd_score = max(0.0, 1.0 - min(drawdown, 25.0) / 25.0)
    pf_score = min(profit_factor, 4.0) / 4.0
    win_score = min(max(win_rate, 0.0), 75.0) / 75.0
    return 100.0 * (0.25 * return_score + 0.25 * dd_score + 0.25 * pf_score + 0.15 * win_score + 0.10 * trade_score)


def candidate_markdown(frame: pd.DataFrame) -> str:
    lines = [
        "# Production Alpha Strategy Candidates",
        "",
        "Compact table for agent decision-making. Read this before detailed promotion reports.",
        "",
    ]
    if frame.empty:
        lines.append("No candidates found.")
        return "\n".join(lines) + "\n"
    cols = [
        "lane",
        "strategy",
        "trades",
        "return_pct",
        "max_drawdown_pct",
        "profit_factor",
        "win_rate_pct",
        "verdict",
        "next_action",
    ]
    lines.append(frame[cols].to_markdown(index=False))
    return "\n".join(lines) + "\n"


def next_tests_markdown(frame: pd.DataFrame) -> str:
    lines = [
        "# Production Alpha Next Tests",
        "",
        "Short queue of next tests chosen from the compact candidate table.",
        "",
    ]
    if frame.empty:
        lines.append("No next tests generated.")
        return "\n".join(lines) + "\n"
    cols = ["priority", "lane", "test_type", "strategy", "trader_question", "pass_condition"]
    lines.append(frame[cols].to_markdown(index=False))
    return "\n".join(lines) + "\n"


def decision_snapshot(candidates: pd.DataFrame, queue: pd.DataFrame) -> str:
    lines = [
        "# Production Alpha Strategy Decision Snapshot",
        "",
        "Purpose: minimize token use. Start here, then open only the linked detailed report for the lane being changed.",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Acceptance Criteria",
        "",
        "1. Each entry family should have its own associated exit/invalidation logic.",
        "2. Broad generic exits are only acceptable for true crash/downturn escalation or multi-indicator/data-source reversal confluence.",
        "3. Normal candidate strategies should keep long/short trade count within `80/20` one way unless explicitly treated as specialist lanes.",
        "4. Promoted production-alpha candidates must beat same-window buy-and-hold/market-change and relevant baselines.",
        "",
        "## Current Candidate Lanes",
        "",
    ]
    if candidates.empty:
        lines.append("No candidate promotion reports found.")
        return "\n".join(lines) + "\n"
    lane_best = candidates.sort_values(["candidate_score", "profit_factor"], ascending=[False, False]).groupby("lane").head(1)
    for _, row in lane_best.iterrows():
        lines.extend(
            [
                f"### {row['lane']}",
                "",
                f"1. Strategy: `{row['strategy']}`",
                f"2. Use case: {row['use_case']}",
                f"3. Result: `{row['trades']}` trades, `{row['return_pct']:.2f}%` return, `{row['max_drawdown_pct']:.2f}%` drawdown, `{row['profit_factor']:.2f}` PF, `{row['win_rate_pct']:.1f}%` win rate.",
                f"4. Verdict: `{row['verdict']}`",
                f"5. Next action: {row['next_action']}",
                f"6. Detailed report: `{row['promotion_md']}`",
                "",
            ]
        )
    lines.extend(["## Next Tests", ""])
    if queue.empty:
        lines.append("No next tests generated.")
    else:
        for _, row in queue.head(8).iterrows():
            lines.extend(
                [
                    f"### {int(row['priority'])}. {row['test_type']}",
                    "",
                    f"1. Lane: `{row['lane']}`",
                    f"2. Strategy: `{row['strategy']}`",
                    f"3. Trader question: {row['trader_question']}",
                    f"4. Pass condition: {row['pass_condition']}",
                    f"5. Read first: `{row['read_first']}`",
                    "",
                ]
            )
    return "\n".join(lines) + "\n"


def as_float(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def is_rejected_report(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False
    verdict_section = text.split("## verdict", 1)[-1] if "## verdict" in text else text
    return (
        "reject" in verdict_section
        or "rejected" in verdict_section
        or "do not promote" in verdict_section
        or "not promote" in verdict_section
    )


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)).strip("_")[:120] or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())

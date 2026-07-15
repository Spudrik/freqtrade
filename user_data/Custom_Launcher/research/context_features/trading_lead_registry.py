from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


USER_DATA_DIR = Path(__file__).resolve().parents[3]
REPORTS_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"

LIFECYCLE_LEDGER = REPORTS_DIR / "trader_concept_lifecycle_ledger.csv"
INDEPENDENT_LEADS = REPORTS_DIR / "trader_independent_promising_leads.csv"
NEXT_CYCLE_PLAN = REPORTS_DIR / "trader_concept_next_cycle_plan.csv"
RULE_SUMMARY = REPORTS_DIR / "trader_rule_backtests_sieve_exact_min10_20260605_summary.csv"
RULE_SELECTED = REPORTS_DIR / "trader_rule_backtests_sieve_exact_min10_20260605_selected_rules.csv"
VALIDATED_ENTRY_LEADS = REPORTS_DIR / "trading_lead_validated_entry_leads.csv"
MTF_INDICATOR_MINING_SELECTED = REPORTS_DIR / "trading_lead_20260606_mtf_indicator_mining_selected.csv"
SIEVE_RUNTIME_CANDIDATES = REPORTS_DIR / "trader_sieve_runtime_candidate_leads.csv"

STATUS_SCORE = {
    "candidate_for_strategy_research": 100,
    "parked_working": 95,
    "freqai_promising": 90,
    "sweep_promising": 82,
    "direct_promising": 75,
    "needs_rework": 45,
    "deferred_for_data": 25,
    "rejected_for_now": 0,
}

BRANCH_PRIORITY = {
    "structure_volume": 1.00,
    "downside_risk": 0.95,
    "orderbook_state": 0.90,
    "regime_confluence": 0.80,
    "sieve_borrowed": 0.78,
    "news_context": 0.40,
    "macro_global": 0.35,
}


@dataclass(frozen=True)
class LeadRecord:
    lead_id: str
    source_file: str
    source_record_id: str
    branch: str
    lead_status: str
    lead_stage: str
    trader_question: str
    visible_market_state: str
    source_detail_blocks: str
    feature_columns: str
    direction: str
    target_or_outcome: str
    evidence_summary: str
    rows: Any
    trigger_rows_or_trades: Any
    quality_score: float
    confluence_candidates: str
    exit_research_needed: str
    risk_sizing_candidates: str
    next_action: str
    evidence_paths: str


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a trader-readable registry of BTC research leads.")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--max-leads", type=int, default=100)
    parser.add_argument("--min-leads", type=int, default=50)
    parser.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    args = parser.parse_args()

    reports_dir = args.reports_dir
    leads = build_registry(reports_dir)
    registry = rank_and_trim(leads, max_leads=args.max_leads)
    if len(registry) < args.min_leads:
        raise RuntimeError(f"Only {len(registry)} leads found; expected at least {args.min_leads}.")

    confluence_plan = build_confluence_plan(registry)
    reports_dir.mkdir(parents=True, exist_ok=True)
    registry_csv = reports_dir / f"trading_lead_registry_{safe_name(args.tag)}.csv"
    registry_md = reports_dir / f"trading_lead_registry_{safe_name(args.tag)}.md"
    registry_latest = reports_dir / "trading_lead_registry_latest.csv"
    plan_csv = reports_dir / f"trading_lead_confluence_plan_{safe_name(args.tag)}.csv"
    plan_md = reports_dir / f"trading_lead_confluence_plan_{safe_name(args.tag)}.md"
    meta_path = reports_dir / f"trading_lead_registry_{safe_name(args.tag)}_meta.json"

    registry.to_csv(registry_csv, index=False)
    registry.to_csv(registry_latest, index=False)
    confluence_plan.to_csv(plan_csv, index=False)
    registry_md.write_text(markdown_registry(registry, confluence_plan), encoding="utf-8")
    plan_md.write_text(markdown_confluence_plan(confluence_plan), encoding="utf-8")
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lead_count": int(len(registry)),
        "confluence_check_count": int(len(confluence_plan)),
        "branch_counts": registry["branch"].value_counts().to_dict(),
        "stage_counts": registry["lead_stage"].value_counts().to_dict(),
        "outputs": {
            "registry_csv": str(registry_csv),
            "registry_latest": str(registry_latest),
            "registry_md": str(registry_md),
            "confluence_plan_csv": str(plan_csv),
            "confluence_plan_md": str(plan_md),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


def build_registry(reports_dir: Path) -> list[LeadRecord]:
    records: list[LeadRecord] = []
    records.extend(load_validated_entry_records(reports_dir / VALIDATED_ENTRY_LEADS.name))
    records.extend(load_selected_rule_records(reports_dir / RULE_SELECTED.name))
    records.extend(load_rule_summary_records(reports_dir / RULE_SUMMARY.name))
    records.extend(load_mtf_indicator_mining_records(reports_dir / MTF_INDICATOR_MINING_SELECTED.name))
    records.extend(load_sieve_runtime_candidate_records(reports_dir / SIEVE_RUNTIME_CANDIDATES.name))
    records.extend(load_independent_records(reports_dir / INDEPENDENT_LEADS.name))
    records.extend(load_next_cycle_records(reports_dir / NEXT_CYCLE_PLAN.name))
    records.extend(load_lifecycle_records(reports_dir / LIFECYCLE_LEDGER.name))
    return records


def load_sieve_runtime_candidate_records(path: Path) -> list[LeadRecord]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    records: list[LeadRecord] = []
    for _, row in frame.iterrows():
        strategy = text(row.get("strategy"))
        direction = text(row.get("direction", infer_direction(strategy)))
        trades = intish(row.get("trade_count"))
        winrate = floatish(row.get("winrate"))
        total_return = floatish(row.get("profit_total"))
        profit_factor = floatish(row.get("profit_factor"))
        drawdown = floatish(row.get("max_drawdown_pct"))
        quality = 120.0 + min(max(floatish(row.get("candidate_score")), 0.0), 180.0) * 0.40
        evidence = (
            f"{trades} entry-sieve trades; {winrate:.1%} win rate; "
            f"{total_return:.1%} return; {drawdown:.1%} max drawdown; "
            f"{profit_factor:.2f} profit factor."
        )
        records.append(
            LeadRecord(
                lead_id=f"lead__sieve_runtime__{safe_name(strategy)}",
                source_file=path.name,
                source_record_id=strategy,
                branch=text(row.get("branch", "sieve_borrowed")),
                lead_status="direct_promising",
                lead_stage="sieve_runtime_candidate",
                trader_question=text(row.get("trader_question")),
                visible_market_state=text(row.get("visible_market_state")),
                source_detail_blocks=text(row.get("source_detail_blocks", source_blocks_for("sieve_borrowed"))),
                feature_columns=text(row.get("feature_columns", strategy)),
                direction=direction,
                target_or_outcome=text(row.get("target_or_outcome", "Entry-sieve validation return, win rate, drawdown, and profit factor.")),
                evidence_summary=evidence,
                rows="",
                trigger_rows_or_trades=trades,
                quality_score=quality,
                confluence_candidates=confluence_candidates_for("sieve_borrowed", direction),
                exit_research_needed=exit_research_for(strategy, direction),
                risk_sizing_candidates=risk_sizing_for("sieve_borrowed"),
                next_action=text(row.get("next_action", "Retest as a direct trader rule, then apply confluence filters and tailored exits.")),
                evidence_paths=text(row.get("evidence_paths", str(path))),
            )
        )
    return records


def load_validated_entry_records(path: Path) -> list[LeadRecord]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    records: list[LeadRecord] = []
    for _, row in frame.iterrows():
        branch = text(row.get("branch", "structure_volume"))
        direction = text(row.get("direction", "both"))
        records.append(
            LeadRecord(
                lead_id=text(row.get("lead_id")),
                source_file=path.name,
                source_record_id=text(row.get("source_record_id", row.get("lead_id"))),
                branch=branch,
                lead_status=text(row.get("lead_status", "candidate_for_strategy_research")),
                lead_stage=text(row.get("lead_stage", "freqtrade_validated_entry_rule")),
                trader_question=text(row.get("trader_question")),
                visible_market_state=text(row.get("visible_market_state")),
                source_detail_blocks=text(row.get("source_detail_blocks", source_blocks_for(branch))),
                feature_columns=text(row.get("feature_columns")),
                direction=direction,
                target_or_outcome=text(row.get("target_or_outcome", "Freqtrade validation return, win rate, drawdown, and profit factor.")),
                evidence_summary=text(row.get("evidence_summary")),
                rows=text(row.get("rows")),
                trigger_rows_or_trades=intish(row.get("trigger_rows_or_trades")),
                quality_score=floatish(row.get("quality_score", 180.0)),
                confluence_candidates=text(row.get("confluence_candidates", confluence_candidates_for(branch, direction))),
                exit_research_needed=text(row.get("exit_research_needed", exit_research_for(text(row.get("lead_id")), direction))),
                risk_sizing_candidates=text(row.get("risk_sizing_candidates", risk_sizing_for(branch))),
                next_action=text(row.get("next_action", "Test tailored confluence, then exits and risk overlays.")),
                evidence_paths=text(row.get("evidence_paths")),
            )
        )
    return records


def load_selected_rule_records(path: Path) -> list[LeadRecord]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return [rule_record(row, path.name, "tested_entry_rule", selected=True) for _, row in frame.iterrows()]


def load_rule_summary_records(path: Path) -> list[LeadRecord]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    frame = frame[frame.get("usable_sample_size", False).fillna(False).astype(bool)]
    frame = frame.sort_values(["positive_quality", "profit_factor", "total_return"], ascending=[False, False, False])
    return [rule_record(row, path.name, "tested_entry_rule_variant", selected=False) for _, row in frame.head(30).iterrows()]


def rule_record(row: pd.Series, source_file: str, stage: str, selected: bool) -> LeadRecord:
    concept_id = text(row.get("concept_id"))
    variant_id = text(row.get("variant_id", concept_id))
    direction = text(row.get("direction", infer_direction(concept_id)))
    quality = 150.0 if selected else 105.0
    if boolish(row.get("positive_quality")):
        quality += 20.0
    quality += min(floatish(row.get("profit_factor")) * 6.0, 18.0)
    quality += min(max(floatish(row.get("trades")) - 10.0, 0.0) / 4.0, 12.0)
    evidence = (
        f"trades {intish(row.get('trades'))}; win_rate {floatish(row.get('win_rate')):.3f}; "
        f"total_return {floatish(row.get('total_return')):.3f}; profit_factor {floatish(row.get('profit_factor')):.3f}; "
        f"max_drawdown {floatish(row.get('max_drawdown')):.3f}"
    )
    return LeadRecord(
        lead_id=f"lead__sieve__{safe_name(variant_id)}",
        source_file=source_file,
        source_record_id=variant_id,
        branch="sieve_borrowed",
        lead_status="candidate_for_strategy_research" if selected else "direct_promising",
        lead_stage=stage,
        trader_question=text(row.get("trader_question")),
        visible_market_state=text(row.get("visible_market_state")),
        source_detail_blocks="Sieve-derived custom indicators; OHLCV; VP/TLV2/pattern/volume-pressure signals where the source rule uses them.",
        feature_columns=variant_id,
        direction=direction,
        target_or_outcome="Standalone entry-rule backtest return, win rate, drawdown, and profit factor.",
        evidence_summary=evidence,
        rows="",
        trigger_rows_or_trades=intish(row.get("trades")),
        quality_score=quality,
        confluence_candidates=confluence_candidates_for("sieve_borrowed", direction),
        exit_research_needed=exit_research_for(concept_id, direction),
        risk_sizing_candidates=risk_sizing_for("sieve_borrowed"),
        next_action="Test confluence filters first, then tune rule-specific exits only for variants that keep trade count.",
        evidence_paths=str(REPORTS_DIR / source_file),
    )


def load_mtf_indicator_mining_records(path: Path) -> list[LeadRecord]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    records: list[LeadRecord] = []
    for _, row in frame.iterrows():
        variant_id = text(row.get("variant_id"))
        concept_id = text(row.get("concept_id", variant_id))
        direction = text(row.get("direction", infer_direction(variant_id)))
        trades = intish(row.get("trades"))
        profit_factor = floatish(row.get("profit_factor"))
        total_return = floatish(row.get("total_return"))
        win_rate = floatish(row.get("win_rate"))
        drawdown = floatish(row.get("max_drawdown"))
        candidate_score = floatish(row.get("candidate_score"))
        quality = 140.0 + min(max(candidate_score, 0.0), 120.0) * 0.50
        evidence = (
            f"{trades} fast-harness trades; {win_rate:.1%} win rate; "
            f"{total_return:.1%} return; {drawdown:.1%} max drawdown; "
            f"{profit_factor:.2f} profit factor."
        )
        records.append(
            LeadRecord(
                lead_id=f"lead__mtf_indicator__{safe_name(variant_id)}",
                source_file=path.name,
                source_record_id=variant_id,
                branch="structure_volume",
                lead_status="direct_promising",
                lead_stage="direct_rule_mining_candidate",
                trader_question=text(row.get("trader_question")),
                visible_market_state=text(row.get("visible_market_state")),
                source_detail_blocks="OHLCV; custom indicators: VP, TLV2, BOS/CHoCH, and pattern geometry across 1h/4h/1d.",
                feature_columns=variant_id,
                direction=direction,
                target_or_outcome="Fast direct simulation using next-candle entry and fixed stop/target/time exit; not yet Freqtrade validated.",
                evidence_summary=evidence,
                rows="",
                trigger_rows_or_trades=trades,
                quality_score=quality,
                confluence_candidates=confluence_candidates_for("structure_volume", direction),
                exit_research_needed=exit_research_for(concept_id, direction),
                risk_sizing_candidates=risk_sizing_for("structure_volume"),
                next_action="Promote strongest variants to Freqtrade validation, then test per-lead confluence and exits. Do not merge all mined leads blindly.",
                evidence_paths=str(path),
            )
        )
    return records


def load_independent_records(path: Path) -> list[LeadRecord]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    records: list[LeadRecord] = []
    for _, row in frame.iterrows():
        family = text(row.get("lead_family"))
        branch = text(row.get("branch", branch_from_text(family)))
        status = text(row.get("best_status", "direct_promising"))
        quality = 85.0 + STATUS_SCORE.get(status, 0) + max(0, 20 - intish(row.get("lead_priority")))
        records.append(
            LeadRecord(
                lead_id=f"lead__family__{safe_name(family)}",
                source_file=path.name,
                source_record_id=family,
                branch=branch,
                lead_status=status,
                lead_stage="lead_family",
                trader_question=text(row.get("best_trader_question")),
                visible_market_state=human_state_from_family(family, branch),
                source_detail_blocks=source_blocks_for(branch),
                feature_columns=family,
                direction=infer_direction(family),
                target_or_outcome=text(row.get("best_target_label")),
                evidence_summary=join_nonempty(row.get("best_result"), row.get("best_baseline_result"), row.get("best_control_result")),
                rows="",
                trigger_rows_or_trades=intish(row.get("record_count")),
                quality_score=quality,
                confluence_candidates=confluence_candidates_for(branch, infer_direction(family)),
                exit_research_needed=exit_research_for(family, infer_direction(family)),
                risk_sizing_candidates=risk_sizing_for(branch),
                next_action=text(row.get("next_action")),
                evidence_paths=text(row.get("evidence_paths")),
            )
        )
    return records


def load_next_cycle_records(path: Path) -> list[LeadRecord]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    frame = frame.sort_values("cycle_priority", ascending=True)
    return [concept_record(row, path.name, "next_cycle_candidate", 35.0) for _, row in frame.iterrows()]


def load_lifecycle_records(path: Path) -> list[LeadRecord]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    frame = frame[frame["status"].isin(["freqai_promising", "sweep_promising", "direct_promising", "needs_rework"])]
    frame = frame.copy()
    frame["rank_score"] = frame.apply(concept_rank_score, axis=1)
    frame = frame.sort_values("rank_score", ascending=False)
    return [concept_record(row, path.name, "concept_evidence", 0.0) for _, row in frame.head(180).iterrows()]


def concept_record(row: pd.Series, source_file: str, stage: str, bonus: float) -> LeadRecord:
    concept_id = text(row.get("concept_id"))
    branch = text(row.get("branch", branch_from_text(concept_id)))
    status = text(row.get("status", "direct_promising"))
    direction = infer_direction(" ".join([concept_id, text(row.get("trader_question")), text(row.get("target_label"))]))
    quality = concept_rank_score(row) + bonus
    return LeadRecord(
        lead_id=f"lead__concept__{safe_name(concept_id)}",
        source_file=source_file,
        source_record_id=concept_id,
        branch=branch,
        lead_status=status,
        lead_stage=stage,
        trader_question=text(row.get("trader_question")),
        visible_market_state=text(row.get("visible_market_state")),
        source_detail_blocks=text(row.get("source_detail_blocks", source_blocks_for(branch))),
        feature_columns=text(row.get("feature_columns")),
        direction=direction,
        target_or_outcome=text(row.get("target_label")),
        evidence_summary=join_nonempty(row.get("direct_result"), row.get("freqai_result"), row.get("baseline_result"), row.get("control_result"), row.get("stability_result")),
        rows=intish(row.get("rows")),
        trigger_rows_or_trades=intish(row.get("trigger_rows")),
        quality_score=quality,
        confluence_candidates=confluence_candidates_for(branch, direction),
        exit_research_needed=exit_research_for(concept_id, direction),
        risk_sizing_candidates=risk_sizing_for(branch),
        next_action=text(row.get("next_action")),
        evidence_paths=text(row.get("evidence_paths")),
    )


def concept_rank_score(row: pd.Series) -> float:
    status = text(row.get("status"))
    branch = text(row.get("branch"))
    base = STATUS_SCORE.get(status, 0) * BRANCH_PRIORITY.get(branch, 0.55)
    result_text = " ".join(text(row.get(c)) for c in ["direct_result", "freqai_result", "baseline_result", "control_result", "stability_result"])
    auc = max_number_after(result_text, "AUC")
    lift = max_number_after(result_text, "lift")
    trigger_rows = intish(row.get("trigger_rows"))
    rows = intish(row.get("rows"))
    if auc:
        base += min(max((auc - 0.50) * 120.0, 0.0), 35.0)
    if lift:
        base += min(max(lift * 40.0, 0.0), 20.0)
    if trigger_rows >= 10:
        base += min(trigger_rows / 20.0, 15.0)
    elif rows >= 100:
        base += 5.0
    if "monthly positive" in result_text.lower():
        base += 8.0
    return float(base)


def rank_and_trim(records: list[LeadRecord], max_leads: int) -> pd.DataFrame:
    frame = pd.DataFrame([r.__dict__ for r in records])
    if frame.empty:
        return frame
    frame = frame.sort_values(["quality_score", "lead_id"], ascending=[False, True])
    frame = frame.drop_duplicates(subset=["lead_id"], keep="first")
    frame = diversify(frame, max_leads=max_leads)
    frame.insert(0, "lead_rank", range(1, len(frame) + 1))
    frame["candidate_for_confluence"] = True
    frame["candidate_for_exit_research"] = frame["lead_status"].isin(["candidate_for_strategy_research", "freqai_promising", "sweep_promising", "direct_promising"])
    frame["candidate_for_risk_sizing"] = frame["candidate_for_exit_research"]
    return frame


def diversify(frame: pd.DataFrame, max_leads: int) -> pd.DataFrame:
    quotas = {
        "structure_volume": 28,
        "sieve_borrowed": 18,
        "orderbook_state": 18,
        "downside_risk": 18,
        "regime_confluence": 8,
        "news_context": 4,
        "macro_global": 4,
    }
    selected: list[pd.DataFrame] = []
    used: set[str] = set()
    for branch, quota in quotas.items():
        block = frame[frame["branch"].eq(branch)].head(quota)
        selected.append(block)
        used.update(block["lead_id"].tolist())
    chosen = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    if len(chosen) < max_leads:
        remainder = frame[~frame["lead_id"].isin(used)].head(max_leads - len(chosen))
        chosen = pd.concat([chosen, remainder], ignore_index=True)
    return chosen.sort_values(["quality_score", "lead_id"], ascending=[False, True]).head(max_leads).reset_index(drop=True)


def build_confluence_plan(registry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, lead in registry.iterrows():
        checks = confluence_check_specs(text(lead["branch"]), text(lead["direction"]))
        for priority, check in enumerate(checks, start=1):
            rows.append(
                {
                    "lead_id": lead["lead_id"],
                    "lead_rank": lead["lead_rank"],
                    "branch": lead["branch"],
                    "base_trader_question": lead["trader_question"],
                    "confluence_priority": priority,
                    "confluence_check": check["name"],
                    "plain_english_question": check["question"],
                    "added_data_family": check["family"],
                    "expected_improvement": check["expected"],
                    "pass_condition": check["pass"],
                    "data_readiness": check["readiness"],
                    "do_not_do": check["avoid"],
                }
            )
    return pd.DataFrame(rows)


def confluence_check_specs(branch: str, direction: str) -> list[dict[str, str]]:
    is_short = direction == "short" or "downside" in branch
    orderbook = {
        "name": "orderbook_confirms_or_warns",
        "question": "Does the book agree with the trade idea, or does it warn that liquidity is blocking the move?",
        "family": "orderbook",
        "expected": "Fewer bad entries, better drawdown control, and clearer high-risk rows.",
        "pass": "Entry quality improves without cutting trade count below a usable level.",
        "readiness": "Use orderbook-present rows only; do not fill missing book data as zero.",
        "avoid": "Do not treat no orderbook data as neutral confirmation.",
    }
    structure = {
        "name": "structure_level_alignment",
        "question": "Does the entry happen at a level a trader would care about, such as VP, TLV2, range, or prior high/low?",
        "family": "custom_structure",
        "expected": "Remove random mid-range entries and focus on levels where reaction/follow-through is plausible.",
        "pass": "Win rate, profit factor, or drawdown improves versus the same lead without the structure filter.",
        "readiness": "Use prebuilt indicator/cache columns only.",
        "avoid": "Do not compute slow custom indicators inside FreqAI or strategy code.",
    }
    volume = {
        "name": "volume_pressure_confirms",
        "question": "Does volume/pressure support the move, or is price moving without participation?",
        "family": "OHLCV_volume_pressure",
        "expected": "Better breakout/breakdown follow-through and fewer weak attempts.",
        "pass": "The filtered rows keep enough trades and improve follow-through or return quality.",
        "readiness": "Ready in confluence cache and Sieve-derived rule outputs.",
        "avoid": "Do not use volume as a standalone reason if level/structure is absent.",
    }
    regime = {
        "name": "regime_allows_trade",
        "question": "Is the broader market state suitable for this type of entry?",
        "family": "price_regime",
        "expected": "Breakouts in expansion regimes and rejections in range regimes should separate better.",
        "pass": "Month-by-month stability improves or drawdown falls.",
        "readiness": "Use existing price/regime/confluence cache.",
        "avoid": "Do not overfit a single month or one market phase.",
    }
    context = {
        "name": "news_context_risk_overlay",
        "question": "Is there external context that makes the trade more dangerous or more likely to accelerate?",
        "family": "news_context",
        "expected": "Better risk avoidance and leverage sizing once source-specific news features are ready.",
        "pass": "Only test after source-specific context coverage is proven for the same window.",
        "readiness": "Park until news/context extraction is source-specific and timestamp-safe.",
        "avoid": "Do not use broad article counts or unknown-source availability as a trading signal.",
    }
    post_start = {
        "name": "post_start_momentum_check",
        "question": "After a downside or breakout move starts, does the next few hours show continuation or fading?",
        "family": "post_start_path_state",
        "expected": "Useful for crash continuation, scaling, or early exit decisions rather than pre-entry prediction only.",
        "pass": "Improves hold/exit decisions for already-triggered entries.",
        "readiness": "Build from 1h OHLCV plus orderbook-present rows where available.",
        "avoid": "Do not leak future candles into the entry decision; use this only after the move has started.",
    }
    if branch == "orderbook_state":
        return [structure, volume, regime, post_start, context]
    if branch == "downside_risk" or is_short:
        return [orderbook, structure, volume, post_start, regime, context]
    if branch == "news_context":
        return [structure, orderbook, volume, regime]
    if branch == "regime_confluence":
        return [structure, volume, orderbook, context]
    return [structure, volume, orderbook, regime, context, post_start]


def confluence_candidates_for(branch: str, direction: str) -> str:
    return "; ".join(spec["name"] for spec in confluence_check_specs(branch, direction)[:4])


def exit_research_for(concept_id: str, direction: str) -> str:
    lower = concept_id.lower()
    if "breakout" in lower or "breakdown" in lower or "compression" in lower:
        return "Test follow-through exit, failed-acceptance exit, time stop, and next-level/next-wall target."
    if "reject" in lower or "fakeout" in lower:
        return "Test quick rejection target, reclaim invalidation exit, and short time stop."
    if "drawdown" in lower or direction == "short":
        return "Test crash-continuation hold, support-reclaim exit, pressure-fade exit, and volatility stop."
    return "Test fixed target, structure invalidation exit, pressure-fade exit, and time stop."


def risk_sizing_for(branch: str) -> str:
    if branch == "orderbook_state":
        return "Size down on missing/stale book; size up only when liquidity/pressure confirms and spread is normal."
    if branch == "news_context":
        return "Use source-specific severity, persistence, and event recency as risk overlays after context readiness."
    if branch == "downside_risk":
        return "Use drawdown risk, volatility expansion, support reclaim, and orderbook vacuum as leverage reducers/increasers."
    return "Use volatility, distance to invalidation, orderbook agreement, and broader regime as risk overlays."


def source_blocks_for(branch: str) -> str:
    mapping = {
        "structure_volume": "OHLCV; custom structure: VP/TLV2/BOS-CHoCH/pattern/volume-pressure caches.",
        "orderbook_state": "Bybit/orderbook trader-state features; orderbook-present scopes only.",
        "downside_risk": "OHLCV; custom structure; downside/crash labels; optional orderbook-present confirmation.",
        "regime_confluence": "OHLCV regime/range/compression features; custom structure filters.",
        "news_context": "News/context source-specific features only when timestamp-safe and coverage-ready.",
        "macro_global": "Macro/global values only when timestamp-safe and same-window coverage is proven.",
        "sieve_borrowed": "Sieve-derived strategy concepts using custom indicators and OHLCV-derived conditions.",
    }
    return mapping.get(branch, "Unknown source family; inspect evidence before testing.")


def human_state_from_family(family: str, branch: str) -> str:
    words = family.replace("_", " ")
    if branch == "orderbook_state":
        return f"Orderbook state suggests {words}; needs chart-level structure confirmation."
    if branch == "downside_risk":
        return f"Market is in a downside-risk situation around {words}."
    if branch == "structure_volume":
        return f"Price/structure/volume state suggests {words}."
    return f"Research lead family: {words}."


def branch_from_text(value: str) -> str:
    lower = value.lower()
    if "orderbook" in lower or "wall" in lower or "liquidity" in lower:
        return "orderbook_state"
    if "drawdown" in lower or "crash" in lower or "breakdown" in lower or "support" in lower:
        return "downside_risk"
    if "news" in lower or "context" in lower or "article" in lower:
        return "news_context"
    if "macro" in lower or "global" in lower or "regime" in lower:
        return "regime_confluence"
    return "structure_volume"


def infer_direction(value: str) -> str:
    lower = value.lower()
    short_words = ["short", "breakdown", "drawdown", "crash", "bear", "reject", "resistance", "support_break"]
    long_words = ["long", "breakout", "upside", "bull", "reclaim", "bounce"]
    short_score = sum(word in lower for word in short_words)
    long_score = sum(word in lower for word in long_words)
    if short_score > long_score:
        return "short"
    if long_score > short_score:
        return "long"
    return "both"


def markdown_registry(registry: pd.DataFrame, confluence_plan: pd.DataFrame) -> str:
    generated = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Trading Lead Registry",
        "",
        f"Generated: {generated}",
        "",
        "Purpose: maintain 50-100 trader-readable BTC leads that can move through confluence checks, tailored exits, and risk/leverage overlays.",
        "",
        "## Counts",
        "",
        f"- Leads: {len(registry)}",
        f"- Planned confluence checks: {len(confluence_plan)}",
        "",
        "### By Branch",
        "",
    ]
    for branch, count in registry["branch"].value_counts().items():
        lines.append(f"- {branch}: {count}")
    lines.extend(["", "## Top Leads", ""])
    display_cols = ["lead_rank", "lead_id", "branch", "direction", "lead_status", "trader_question", "evidence_summary", "next_action"]
    lines.append(registry[display_cols].head(40).to_markdown(index=False))
    lines.extend(["", "## Next Step", "", "Use the confluence plan to test whether extra source families improve each lead without destroying trade count. Successful entries then move to tailored exit research."])
    return "\n".join(lines) + "\n"


def markdown_confluence_plan(plan: pd.DataFrame) -> str:
    lines = [
        "# Trading Lead Confluence Plan",
        "",
        "Each row asks whether an extra source family improves an existing lead. These are not final trading rules yet.",
        "",
    ]
    cols = ["lead_rank", "lead_id", "confluence_priority", "confluence_check", "plain_english_question", "expected_improvement", "data_readiness"]
    lines.append(plan[cols].head(120).to_markdown(index=False))
    return "\n".join(lines) + "\n"


def safe_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean[:180] or "unknown"


def max_number_after(text_value: str, token: str) -> float:
    matches = re.findall(rf"{re.escape(token)}\s+(-?\d+(?:\.\d+)?)", text_value, flags=re.IGNORECASE)
    nums = [float(m) for m in matches]
    return max(nums) if nums else 0.0


def text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    return str(value)


def join_nonempty(*values: Any) -> str:
    parts = [text(v).strip() for v in values if text(v).strip()]
    return "; ".join(parts)


def floatish(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def intish(value: Any) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from trader_confluence_hypotheses import hypothesis_by_id
except ImportError:  # pragma: no cover - direct module execution fallback
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from trader_confluence_hypotheses import hypothesis_by_id


USER_DATA_DIR = Path(__file__).resolve().parents[3]
REPORTS_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"
QUEUE_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "freqai_queue"
FEATURE_DISCOVERY_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "feature_discovery"
FREQAI_RUNS_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "freqai_runs"

LEDGER_PATH = REPORTS_DIR / "trader_concept_lifecycle_ledger.csv"
SUMMARY_PATH = REPORTS_DIR / "trader_concept_lifecycle_summary.md"
INVENTORY_PATH = REPORTS_DIR / "trader_concept_evidence_inventory.csv"
NEXT_CYCLE_CSV_PATH = REPORTS_DIR / "trader_concept_next_cycle_plan.csv"
NEXT_CYCLE_MD_PATH = REPORTS_DIR / "trader_concept_next_cycle_plan.md"
INDEPENDENT_LEADS_CSV_PATH = REPORTS_DIR / "trader_independent_promising_leads.csv"
INDEPENDENT_LEADS_MD_PATH = REPORTS_DIR / "trader_independent_promising_leads.md"
CONCEPT_CARDS_CSV_PATH = REPORTS_DIR / "trader_concept_cards.csv"
CONCEPT_CARDS_MD_PATH = REPORTS_DIR / "trader_concept_cards.md"


STATUS_RANK = {
    "rejected_for_now": 0,
    "deferred_for_data": 1,
    "needs_rework": 2,
    "idea": 3,
    "direct_promising": 4,
    "sweep_promising": 5,
    "freqai_promising": 6,
    "parked_working": 7,
    "candidate_for_strategy_research": 8,
}


LEAD_CARD_OVERRIDES: dict[str, dict[str, str]] = {
    "orderbook_breakdown_risk": {
        "concept_title": "Orderbook breakdown risk",
        "trader_question": "When the book shows weak support or downside pressure, does price keep breaking down?",
        "visible_market_state": "Price is in or near a breakdown/crash-detection setup and orderbook depth/pressure suggests support is weak.",
        "source_detail_blocks": "Orderbook trader-state features; price baseline; crash-detection event scope.",
        "target_label": "Did downside continuation or breakdown success happen after that hour?",
        "development_note": "Working lead. Split into support removal, downside vacuum, and pressure-flip variants next.",
    },
    "orderbook_failed_breakout_rejection": {
        "concept_title": "Orderbook failed-breakout rejection",
        "trader_question": "When price is near resistance or VAH, can the orderbook help identify breakouts that fail?",
        "visible_market_state": "Price is near a resistance/rejection setup and orderbook features do not support clean upside acceptance.",
        "source_detail_blocks": "Orderbook trader-state features; VAH/rejection scope; price control.",
        "target_label": "Did the breakout attempt fail over the next 6h?",
        "development_note": "Working lead. Build one clean rejection/fakeout rule and compare against structure-only rejection.",
    },
    "support_reclaim_after_downside_break": {
        "concept_title": "Support reclaim after downside break",
        "trader_question": "After a downside break starts, can we identify rows where the danger fades because support is reclaimed?",
        "visible_market_state": "A support break has already happened, then features show support reclaim or bearish pressure fading.",
        "source_detail_blocks": "Epsilon downside-after-break features; price/structure; orderbook-present scopes where available.",
        "target_label": "Was broken support reclaimed within the next 6h?",
        "development_note": "Working lead. Useful for crash continuation versus slowing/crash-fade separation.",
    },
    "orderbook_fakeout_detection": {
        "concept_title": "Orderbook fakeout detection",
        "trader_question": "Can the book identify breakout or breakdown attempts that are likely to fake out?",
        "visible_market_state": "A breakout/breakdown attempt is active but orderbook/structure features suggest poor acceptance.",
        "source_detail_blocks": "Orderbook and structure/VP fakeout scopes.",
        "target_label": "Did the move fake out over the next 24h?",
        "development_note": "Working lead but needs a clearer direct rule because fakeout labels can be broad.",
    },
    "structure_breakout_acceptance": {
        "concept_title": "Structure breakout acceptance",
        "trader_question": "When price breaks resistance, do structure/VP conditions identify when the move is accepted?",
        "visible_market_state": "Price is in a breakout-acceptance setup near resistance/value and structure features support continuation.",
        "source_detail_blocks": "Structure/VP; price/volume; event scope with orderbook-present scoring where available.",
        "target_label": "Did price continue after the breakout attempt over the next 6h?",
        "development_note": "Working lead. Compare against compression and range-high breakout concepts.",
    },
    "structure_fakeout_detection": {
        "concept_title": "Structure fakeout detection",
        "trader_question": "Can structure/VP identify failed breakout or breakdown attempts before they become obvious?",
        "visible_market_state": "Price attempts a level break but structure/value-area behaviour suggests poor acceptance.",
        "source_detail_blocks": "Structure/VP fakeout scopes; price baseline.",
        "target_label": "Did a fakeout occur over the next 24h?",
        "development_note": "Working lead. Needs source-detail explanation of which VP/TLV2 pieces matter.",
    },
    "structure_resistance_rejection": {
        "concept_title": "Structure resistance rejection",
        "trader_question": "When price is near resistance, can structure/VP identify rejection rather than continuation?",
        "visible_market_state": "Price is near resistance/VAH/TLV2 resistance and structure features indicate failure to accept above.",
        "source_detail_blocks": "Structure/VP; VAH and resistance-rejection scopes.",
        "target_label": "Did the breakout/retest fail over the next 6h?",
        "development_note": "Working lead. Strong candidate for a risk filter around long entries.",
    },
    "structure_breakdown_crash_detection": {
        "concept_title": "Structure breakdown/crash detection",
        "trader_question": "Can structure/VP identify breakdowns that keep moving lower?",
        "visible_market_state": "Price is in a support/breakdown setup and structure features indicate downside continuation risk.",
        "source_detail_blocks": "Structure/VP; crash-detection scope; price control.",
        "target_label": "Did breakdown/crash continuation happen after that hour?",
        "development_note": "Working lead. Needs Q1 structural NaN readiness issue resolved before broad use.",
    },
    "range_high_breakout_continuation": {
        "concept_title": "Range-high breakout continuation",
        "trader_question": "When price breaks the recent range high with confirming conditions, does it keep moving up?",
        "visible_market_state": "Price is near/breaking recent range high with bullish volume or weak resistance.",
        "source_detail_blocks": "Price range position; structure/volume; optional orderbook resistance weakness.",
        "target_label": "Did breakout success happen over the next 6h?",
        "development_note": "Sweep-promising. Already has FreqAI siblings; compare scopes and preserve best threshold settings.",
    },
    "compression_breakout_release": {
        "concept_title": "Compression breakout release",
        "trader_question": "After compression, does a range-high break with bullish pressure release upward?",
        "visible_market_state": "Market was compressed, then price breaks upward with bullish volume/structure pressure.",
        "source_detail_blocks": "Compression/range state; structure/volume; price baseline.",
        "target_label": "Did upside breakout continuation happen over the next 6h?",
        "development_note": "Sweep-promising. Keep as a core bullish concept.",
    },
    "structure_support_breakdown_risk": {
        "concept_title": "Structure support breakdown risk",
        "trader_question": "When stacked support breaks with sell pressure, does drawdown risk increase?",
        "visible_market_state": "Support stack breaks and sell volume/structure pressure confirms the move.",
        "source_detail_blocks": "Structure support; VP/TLV2; volume pressure; price baseline.",
        "target_label": "Did a large drawdown or breakdown continuation happen after that hour?",
        "development_note": "Sweep-promising. Core downside risk concept.",
    },
    "compression_breakdown_release": {
        "concept_title": "Compression breakdown release",
        "trader_question": "After compression, does a range-low break with bearish pressure release downward?",
        "visible_market_state": "Market was compressed, then price breaks lower with bearish volume/structure pressure.",
        "source_detail_blocks": "Compression/range state; structure/volume; price baseline.",
        "target_label": "Did downside breakdown continuation happen over the next 6h?",
        "development_note": "Sweep-promising. Core bearish counterpart to compression breakout.",
    },
    "delta_expansion_drawdown_risk": {
        "concept_title": "Expansion drawdown risk",
        "trader_question": "When expansion follows weak structure, does drawdown risk rise rather than chop?",
        "visible_market_state": "Market is no longer compressed and bearish pressure/structure suggests expansion lower.",
        "source_detail_blocks": "Price expansion state; structure/volume; drawdown target.",
        "target_label": "Did a larger 24h drawdown happen?",
        "development_note": "Sweep-promising but needs cleaner trader wording and threshold review.",
    },
    "range_low_breakdown_continuation": {
        "concept_title": "Range-low breakdown continuation",
        "trader_question": "When price breaks the recent range low with confirming conditions, does it keep moving down?",
        "visible_market_state": "Price is near/breaking recent range low with bearish volume or weak support.",
        "source_detail_blocks": "Price range position; structure/volume; optional orderbook support weakness.",
        "target_label": "Did breakdown success happen over the next 6h?",
        "development_note": "Sweep-promising. Strong mirror of range-high breakout.",
    },
    "bearish_volume_pressure_breakdown": {
        "concept_title": "Bearish volume pressure breakdown",
        "trader_question": "When bearish volume pressure is high or abnormal, does downside follow-through become more likely?",
        "visible_market_state": "Bearish volume confirmation or 24h pressure is elevated versus recent history.",
        "source_detail_blocks": "Price/volume pressure features; structure cached price-volume state.",
        "target_label": "Did breakdown success happen over the next 6h?",
        "development_note": "Direct-promising feature clue. Convert to a named thresholded hypothesis.",
    },
    "bullish_volume_pressure_breakout": {
        "concept_title": "Bullish volume pressure breakout",
        "trader_question": "When bullish volume pressure is high or abnormal, does upside follow-through become more likely?",
        "visible_market_state": "Bullish volume confirmation or 6h/24h pressure is elevated versus recent history.",
        "source_detail_blocks": "Price/volume pressure features; structure cached price-volume state.",
        "target_label": "Did breakout success happen over the next 6h?",
        "development_note": "Direct-promising feature clue. Convert to a named thresholded hypothesis.",
    },
    "tlv2_resistance_distance_breakout_or_reject": {
        "concept_title": "TLV2 resistance distance behaviour",
        "trader_question": "Does movement relative to TLV2 resistance help identify breakout or rejection risk?",
        "visible_market_state": "Distance to TLV2 resistance changes materially over recent hours.",
        "source_detail_blocks": "TLV2 resistance distance/slope features across timeframes.",
        "target_label": "Did breakout/rejection/downside follow-through happen after that hour?",
        "development_note": "Direct-promising but currently too mixed. Split into breakout and rejection versions.",
    },
    "price_momentum_breakout_continuation": {
        "concept_title": "Price momentum breakout continuation",
        "trader_question": "When recent price momentum is already strong, does an upside break keep travelling?",
        "visible_market_state": "Recent 6h/24h return or range-position momentum is elevated before the decision hour.",
        "source_detail_blocks": "OHLCV price momentum; range-position state; price/structure controls.",
        "target_label": "Did upside breakout or follow-through happen over the next 6h?",
        "development_note": "Feature-discovery lead. Rework into a clean momentum-continuation direct test with anti-chase controls.",
    },
    "price_momentum_breakdown_continuation": {
        "concept_title": "Price momentum breakdown continuation",
        "trader_question": "When recent price momentum is already negative, does a downside break keep travelling?",
        "visible_market_state": "Recent 6h/24h return or range-position momentum is weak before the decision hour.",
        "source_detail_blocks": "OHLCV price momentum; range-position state; price/structure controls.",
        "target_label": "Did downside breakdown or follow-through happen over the next 6h?",
        "development_note": "Feature-discovery lead. Rework into a clean continuation-vs-exhaustion test.",
    },
    "volatility_expansion_drawdown_risk": {
        "concept_title": "Volatility expansion drawdown risk",
        "trader_question": "When volatility/range expands after compression, does downside risk increase?",
        "visible_market_state": "Hourly range expands or compression falls before the decision hour.",
        "source_detail_blocks": "OHLCV range expansion; confluence compression score; structure/volume controls.",
        "target_label": "Did a large drawdown happen over the next 6h?",
        "development_note": "Feature-discovery lead. Compare with the broader expansion drawdown concept and avoid duplicate counting.",
    },
    "market_structure_invalidation_drawdown": {
        "concept_title": "Market-structure invalidation drawdown",
        "trader_question": "When market-structure invalidation levels weaken, does drawdown risk rise?",
        "visible_market_state": "BOS/CHoCH invalidation level or related structure state shifts before the decision hour.",
        "source_detail_blocks": "BOS/CHoCH market-structure cached features; OHLCV controls.",
        "target_label": "Did a large drawdown happen over the next 6h?",
        "development_note": "Feature-discovery lead. Needs a named direct test around invalidation/reclaim behaviour.",
    },
    "tlv2_resistance_line_drawdown": {
        "concept_title": "TLV2 resistance-line drawdown risk",
        "trader_question": "When TLV2 resistance behaviour changes sharply, does downside risk rise?",
        "visible_market_state": "TLV2 resistance line or resistance distance changes over recent hours.",
        "source_detail_blocks": "TLV2 support/resistance line features; OHLCV controls.",
        "target_label": "Did a large drawdown or breakdown happen after that hour?",
        "development_note": "Feature-discovery lead. Split resistance rejection, resistance break, and downside invalidation variants.",
    },
    "bull_trend_regime_continuation_filter": {
        "concept_title": "Bull-trend regime continuation filter",
        "trader_question": "When the market is already in a bullish regime, does trend continuation become easier to identify?",
        "visible_market_state": "A cached bull-trend regime score is high or improving before the decision hour.",
        "source_detail_blocks": "Price regime state; structure/volume confirmation; OHLCV controls.",
        "target_label": "Did upside continuation happen, or did weak bull-regime readings warn of downside?",
        "development_note": "Feature-discovery lead. Split into bullish continuation and failed-regime downside variants.",
    },
    "orderbook_wall_distance_shift": {
        "concept_title": "Orderbook wall-distance shift",
        "trader_question": "When nearby Bybit walls move away from or toward price, does that help identify continuation risk?",
        "visible_market_state": "Nearest bid/ask wall distance changes over the last 3h/24h.",
        "source_detail_blocks": "Bybit linear orderbook trader-state wall-distance features; OHLCV controls.",
        "target_label": "Did breakout or breakdown continuation happen over the next 6h?",
        "development_note": "Feature-discovery lead. Rework into separate resistance-retreat and support-retreat direct tests.",
    },
}


@dataclass
class ConceptRecord:
    concept_id: str
    branch: str
    status: str
    trader_question: str
    visible_market_state: str
    source_detail_blocks: str
    feature_columns: str
    target_label: str
    test_window: str
    rows: int | None
    trigger_rows: int | None
    direct_result: str
    baseline_result: str
    control_result: str
    freqai_result: str
    stability_result: str
    verdict: str
    next_action: str
    evidence_paths: str
    last_updated: str


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _latest(pattern: str) -> Path | None:
    paths = sorted(REPORTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[0] if paths else None


def _all_reports(pattern: str) -> list[Path]:
    return sorted(REPORTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)


def _branch_for(concept_id: str, source_families: tuple[str, ...] = ()) -> str:
    lowered = concept_id.lower()
    families = set(source_families)
    if "context" in families or "news" in lowered or "context" in lowered:
        return "news_context"
    if "macro" in lowered:
        return "macro_global"
    if "orderbook" in lowered or "wall" in lowered or "liquidity" in lowered or "ob_" in lowered:
        return "orderbook_state"
    if "downside" in lowered or "drawdown" in lowered or "breakdown" in lowered or "support" in lowered:
        return "downside_risk"
    if "regime" in lowered or "confluence" in lowered or len(families) > 2:
        return "regime_confluence"
    return "structure_volume"


def _source_detail_blocks(source_families: tuple[str, ...], concept_id: str) -> str:
    details: list[str] = []
    lowered = concept_id.lower()
    if "price" in source_families:
        details.append("OHLCV/price behaviour")
    if "structure" in source_families:
        details.append("custom structure: VP/TLV2/BOS-CHoCH/pattern/volume-pressure caches")
    if "orderbook" in source_families:
        details.append("orderbook trader-state features: spot/Bybit where coverage exists")
    if "context" in source_families:
        details.append("context/news source blocks; only valid where coverage is proven")
    if lowered.startswith("feature_discovery"):
        details.append("feature-discovery candidate column")
    if not details:
        details.append("derived from available report metadata")
    return "; ".join(details)


def _hypothesis_text(concept_id: str, fallback: str = "") -> tuple[str, tuple[str, ...], str]:
    hypothesis = hypothesis_by_id().get(concept_id)
    if hypothesis is None:
        return fallback or concept_id.replace("_", " "), (), ""
    return hypothesis.theory, hypothesis.source_families, hypothesis.direction


def _target_text(target: str) -> str:
    translations = {
        "breakout_success_next_6h": "Did price keep going up after the breakout attempt within the next 6h?",
        "breakdown_success_next_6h": "Did price keep moving down after the breakdown attempt within the next 6h?",
        "large_drawdown_next_6h": "Did price suffer a large drawdown over the next 6h?",
        "large_drawdown_next_24h": "Did price suffer a large drawdown over the next 24h?",
        "support_reclaim_next_6h": "After a support break, was broken support reclaimed within 6h?",
        "downside_exhaustion_next_6h": "After a downside break, did the move stall or bounce within 6h?",
        "future_max_upside_6h": "How much upside path appeared over the next 6h?",
        "future_max_upside_24h": "How much upside path appeared over the next 24h?",
    }
    return translations.get(target, target.replace("_", " "))


def _defer_if_unready(status: str, source_families: tuple[str, ...], source_family_text: str = "") -> str:
    families = {family.lower() for family in source_families}
    text = source_family_text.lower()
    if "context" in families or "news" in families or "context" in text or "news" in text or "gkg" in text or "gdelt" in text:
        return "deferred_for_data"
    return status


def _fmt_float(value: Any, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def _int_or_none(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _direct_status(row: pd.Series) -> str:
    auc = float(row.get("roc_auc", 0) or 0)
    shuffled = float(row.get("shuffled_roc_auc", 0.5) or 0.5)
    baseline = row.get("baseline_roc_auc")
    baseline_delta = auc - float(baseline) if pd.notna(baseline) else 0.0
    trigger_rows = int(row.get("rows_trigger", 0) or 0)
    monthly_windows = float(row.get("monthly_windows", 0) or 0)
    monthly_positive = float(row.get("monthly_auc_positive_windows", 0) or 0)
    monthly_ratio = monthly_positive / monthly_windows if monthly_windows else 0.0
    lift = float(row.get("oriented_top20_lift", 0) or 0)
    if trigger_rows >= 30 and auc >= 0.68 and shuffled < 0.56 and baseline_delta >= -0.02 and monthly_ratio >= 0.75:
        return "direct_promising"
    if trigger_rows >= 20 and auc >= 0.60 and lift > 0 and shuffled < 0.58:
        return "needs_rework"
    return "rejected_for_now"


def _sweep_status(row: pd.Series) -> str:
    variants = int(row.get("pass_variants", 0) or 0)
    rows = int(row.get("best_trigger_rows", 0) or 0)
    auc_delta = float(row.get("best_auc_minus_baseline", 0) or 0)
    lift_same = float(row.get("best_trigger_lift_vs_same", 0) or 0)
    if variants >= 10 and rows >= 30 and auc_delta >= 0.02 and lift_same > 0:
        return "sweep_promising"
    if variants > 0 and rows >= 20 and lift_same > 0:
        return "needs_rework"
    return "rejected_for_now"


def _feature_status(row: pd.Series) -> str:
    if bool(row.get("stable_shortlist", False)) and float(row.get("oriented_auc", 0) or 0) >= 0.64:
        return "direct_promising"
    if bool(row.get("univariate_pass", False)) and float(row.get("lift_vs_base", 0) or 0) > 0:
        return "needs_rework"
    return "rejected_for_now"


def _freqai_status(row: pd.Series) -> str:
    auc_raw = row.get("roc_auc")
    auc = float(auc_raw) if pd.notna(auc_raw) else float("nan")
    delta = row.get("roc_auc_delta_vs_control")
    delta_value = float(delta) if pd.notna(delta) else 0.0
    top_delta = row.get("top_quintile_actual_rate_delta_vs_control")
    top_delta_value = float(top_delta) if pd.notna(top_delta) else 0.0
    corr_delta = row.get("prediction_actual_corr_delta_vs_control")
    corr_delta_value = float(corr_delta) if pd.notna(corr_delta) else 0.0
    if pd.isna(auc):
        if corr_delta_value >= 0.05 and abs(top_delta_value) >= 0.003:
            return "freqai_promising"
        if corr_delta_value > 0.01:
            return "needs_rework"
        return "rejected_for_now"
    if auc >= 0.57 and (delta_value >= 0.03 or top_delta_value >= 0.03):
        return "freqai_promising"
    if auc >= 0.53 and (delta_value > 0 or top_delta_value > 0):
        return "needs_rework"
    return "rejected_for_now"


def _record_key(record: ConceptRecord) -> tuple[str, str]:
    return record.concept_id, record.target_label


def _merge_records(records: list[ConceptRecord]) -> list[ConceptRecord]:
    merged: dict[tuple[str, str], ConceptRecord] = {}
    for record in records:
        key = _record_key(record)
        existing = merged.get(key)
        if existing is None:
            merged[key] = record
            continue
        if STATUS_RANK.get(record.status, 0) > STATUS_RANK.get(existing.status, 0):
            old_paths = set(filter(None, existing.evidence_paths.split("; ")))
            new_paths = set(filter(None, record.evidence_paths.split("; ")))
            record.evidence_paths = "; ".join(sorted(old_paths | new_paths))
            merged[key] = record
        else:
            paths = set(filter(None, existing.evidence_paths.split("; ")))
            paths.update(filter(None, record.evidence_paths.split("; ")))
            existing.evidence_paths = "; ".join(sorted(paths))
    return sorted(merged.values(), key=lambda r: (-STATUS_RANK.get(r.status, 0), r.branch, r.concept_id, r.target_label))


def import_direct_tests(path: Path | None) -> list[ConceptRecord]:
    if path is None:
        return []
    df = _read_csv(path)
    records: list[ConceptRecord] = []
    for _, row in df.iterrows():
        concept_id = str(row["hypothesis_id"])
        target = str(row["target"])
        theory, source_families, _direction = _hypothesis_text(concept_id)
        status = _defer_if_unready(_direct_status(row), source_families)
        trigger_rate = _fmt_float(row.get("target_mean_trigger"))
        same_rate = _fmt_float(row.get("target_mean_same_regime_without_trigger"))
        random_rate = _fmt_float(row.get("target_mean_random_eligible_control"))
        auc = _fmt_float(row.get("roc_auc"))
        baseline_auc = _fmt_float(row.get("baseline_roc_auc"))
        records.append(
            ConceptRecord(
                concept_id=concept_id,
                branch=_branch_for(concept_id, source_families),
                status=status,
                trader_question=theory,
                visible_market_state=f"Rows where {concept_id.replace('_', ' ')} setup/trigger score was active.",
                source_detail_blocks=_source_detail_blocks(source_families, concept_id),
                feature_columns=f"conf_{concept_id}_setup; conf_{concept_id}_trigger; conf_{concept_id}_score",
                target_label=target,
                test_window=path.stem,
                rows=_int_or_none(row.get("rows")),
                trigger_rows=_int_or_none(row.get("rows_trigger")),
                direct_result=f"AUC {auc}; trigger rate {trigger_rate}; same-regime rate {same_rate}; random rate {random_rate}",
                baseline_result=f"price/structure baseline AUC {baseline_auc}",
                control_result=f"shuffled AUC {_fmt_float(row.get('shuffled_roc_auc'))}; same-regime {same_rate}; random {random_rate}",
                freqai_result="",
                stability_result=f"monthly positive {row.get('monthly_auc_positive_windows', '')}/{row.get('monthly_windows', '')}",
                verdict=_verdict_for_status(status),
                next_action=_next_action_for_status(status),
                evidence_paths=str(path),
                last_updated=_now(),
            )
        )
    return records


def import_threshold_sweep(path: Path | None) -> list[ConceptRecord]:
    if path is None:
        return []
    df = _read_csv(path)
    records: list[ConceptRecord] = []
    for _, row in df.iterrows():
        concept_id = str(row["hypothesis_id"])
        target = str(row["target"])
        theory, source_families, _direction = _hypothesis_text(concept_id)
        status = _defer_if_unready(_sweep_status(row), source_families)
        records.append(
            ConceptRecord(
                concept_id=concept_id,
                branch=_branch_for(concept_id, source_families),
                status=status,
                trader_question=theory,
                visible_market_state=f"Threshold sweep for {concept_id.replace('_', ' ')}.",
                source_detail_blocks=_source_detail_blocks(source_families, concept_id),
                feature_columns=f"threshold variants over conf_{concept_id}_score/setup/trigger",
                target_label=target,
                test_window=path.stem,
                rows=_int_or_none(row.get("best_setup_rows")),
                trigger_rows=_int_or_none(row.get("best_trigger_rows")),
                direct_result=(
                    f"best AUC {_fmt_float(row.get('best_roc_auc'))}; trigger rate "
                    f"{_fmt_float(row.get('best_trigger_event_rate'))}; same-regime rate "
                    f"{_fmt_float(row.get('best_same_regime_event_rate'))}; pass variants {row.get('pass_variants', '')}"
                ),
                baseline_result=f"baseline AUC {_fmt_float(row.get('best_baseline_roc_auc'))}; delta {_fmt_float(row.get('best_auc_minus_baseline'))}",
                control_result=f"random rate {_fmt_float(row.get('best_random_event_rate'))}; lift vs same {_fmt_float(row.get('best_trigger_lift_vs_same'))}",
                freqai_result="",
                stability_result=f"monthly positive {row.get('best_monthly_auc_positive_windows', '')}/{row.get('best_monthly_windows', '')}",
                verdict=_verdict_for_status(status),
                next_action=_next_action_for_status(status),
                evidence_paths=str(path),
                last_updated=_now(),
            )
        )
    return records


def import_feature_discovery(path: Path | None, limit: int) -> list[ConceptRecord]:
    if path is None:
        return []
    df = _read_csv(path)
    if df.empty:
        return []
    df = df.sort_values(["stable_shortlist", "oriented_auc", "lift_vs_base"], ascending=[False, False, False]).head(limit)
    records: list[ConceptRecord] = []
    for _, row in df.iterrows():
        base = str(row["base_column"])
        target = str(row["target"])
        transform = str(row["transform"])
        concept_id = f"feature_discovery__{base}__{transform}"
        source_families = (str(row.get("source_family", "")),)
        status = _defer_if_unready(_feature_status(row), source_families, str(row.get("source_detail", "")))
        records.append(
            ConceptRecord(
                concept_id=concept_id,
                branch=_branch_for(concept_id, source_families),
                status=status,
                trader_question=f"Does {row.get('plain_english', base)} help rank {_target_text(target).lower()}",
                visible_market_state=f"One candidate feature transform was visible: {base} using {transform}.",
                source_detail_blocks=f"{row.get('source_family', '')}: {row.get('source_detail', '')}",
                feature_columns=str(row["feature"]),
                target_label=target,
                test_window="feature-discovery screen",
                rows=_int_or_none(row.get("rows")),
                trigger_rows=_int_or_none(row.get("positive_rows")),
                direct_result=(
                    f"oriented AUC {_fmt_float(row.get('oriented_auc'))}; top-decile event rate "
                    f"{_fmt_float(row.get('top_decile_event_rate'))}; base event rate {_fmt_float(row.get('base_event_rate'))}"
                ),
                baseline_result=f"lift vs base {_fmt_float(row.get('lift_vs_base'))}",
                control_result=f"random event rate {_fmt_float(row.get('random_event_rate'))}; shuffled AUC {_fmt_float(row.get('shuffled_oriented_auc'))}",
                freqai_result="",
                stability_result=f"monthly positive ratio {_fmt_float(row.get('monthly_positive_ratio'))}",
                verdict=_verdict_for_status(status),
                next_action="Convert this stable feature clue into a named trader concept, then direct-test the concept.",
                evidence_paths=str(path),
                last_updated=_now(),
            )
        )
    return records


def import_feature_discovery_paths(paths: list[Path], limit: int) -> list[ConceptRecord]:
    records: list[ConceptRecord] = []
    if not paths:
        return records
    per_file_limit = max(limit, 1)
    for path in paths:
        records.extend(import_feature_discovery(path, per_file_limit))
    return records


def import_freqai_ledger(path: Path | None, limit: int) -> list[ConceptRecord]:
    if path is None:
        return []
    df = _read_csv(path)
    if df.empty:
        return []
    df = df[df["status"].astype(str).str.lower().isin(["scored", "control_comparison"])].copy()
    if df.empty:
        return []
    for column in (
        "roc_auc_delta_vs_control",
        "top_quintile_actual_rate_delta_vs_control",
        "prediction_actual_corr_delta_vs_control",
        "roc_auc",
    ):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["_import_score"] = (
        df["roc_auc_delta_vs_control"].fillna(0).clip(lower=0)
        + df["top_quintile_actual_rate_delta_vs_control"].fillna(0).abs()
        + df["prediction_actual_corr_delta_vs_control"].fillna(0).clip(lower=0)
        + (df["roc_auc"].fillna(0) - 0.5).clip(lower=0)
    )
    df = df.sort_values(["_import_score", "roc_auc", "roc_auc_delta_vs_control"], ascending=False).head(limit)
    records: list[ConceptRecord] = []
    for _, row in df.iterrows():
        family = str(row.get("family", "freqai"))
        profile_id = str(row.get("profile_id", "unknown_profile"))
        target = str(row.get("objective", row.get("actual_column", "")))
        scope = str(row.get("scope", ""))
        concept_id = f"freqai_validation__{profile_id}__{scope}".replace("-", "_")
        status = _defer_if_unready(_freqai_status(row), (family,), profile_id)
        records.append(
            ConceptRecord(
                concept_id=concept_id,
                branch=_branch_for(concept_id, (family,)),
                status=status,
                trader_question=str(row.get("objective", "")).strip(),
                visible_market_state=f"FreqAI profile {profile_id} scored rows in scope {scope}.",
                source_detail_blocks=family,
                feature_columns=f"profile_id={profile_id}; freqai_identifier={row.get('freqai_identifier', '')}",
                target_label=str(row.get("objective", "")),
                test_window=f"{row.get('window', '')} {row.get('timerange', '')}".strip(),
                rows=_int_or_none(row.get("rows")),
                trigger_rows=None,
                direct_result="",
                baseline_result=f"control profile {row.get('control_profile_id', '')}; AUC delta {_fmt_float(row.get('roc_auc_delta_vs_control'))}",
                control_result=f"top bucket delta {_fmt_float(row.get('top_quintile_actual_rate_delta_vs_control'))}",
                freqai_result=f"AUC {_fmt_float(row.get('roc_auc'))}; AP {_fmt_float(row.get('average_precision'))}; corr {_fmt_float(row.get('prediction_actual_corr'))}",
                stability_result="See FreqAI queue/report artifacts for window split.",
                verdict=_verdict_for_status(status),
                next_action=_next_action_for_status(status),
                evidence_paths=str(path),
                last_updated=_now(),
            )
        )
    return records


def _verdict_for_status(status: str) -> str:
    if status in {"direct_promising", "sweep_promising", "freqai_promising"}:
        return "Worth more testing; evidence beats at least one useful control and should be preserved."
    if status == "needs_rework":
        return "Not ready, but there is enough signal or logic to rework instead of discard."
    if status == "deferred_for_data":
        return "Cannot be tested honestly with current source coverage."
    if status == "rejected_for_now":
        return "Reject for now; failed current controls or sample-size requirements."
    return "Track as an idea until tested."


def _next_action_for_status(status: str) -> str:
    if status == "direct_promising":
        return "Run threshold sweep or promote to focused FreqAI validation if not already done."
    if status == "sweep_promising":
        return "Queue focused FreqAI validation and preserve the best threshold settings."
    if status == "freqai_promising":
        return "Park as working and compare against sibling concepts before strategy research."
    if status == "needs_rework":
        return "Diagnose failure reason, change one condition, and retest with controls."
    if status == "deferred_for_data":
        return "Wait for coverage or rebuild source formatting before testing."
    if status == "rejected_for_now":
        return "Do not spend more cycles unless a new trader interpretation appears."
    return "Define columns, target, controls, and first direct test."


def build_inventory(imported_paths: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidates: list[Path] = []
    for directory, patterns in (
        (REPORTS_DIR, ("*.csv", "*.md", "*.json")),
        (QUEUE_DIR, ("queue_*\\freqai_experiment_queue.json",)),
        (FEATURE_DISCOVERY_DIR, ("*",)),
        (FREQAI_RUNS_DIR, ("*",)),
    ):
        if not directory.exists():
            continue
        for pattern in patterns:
            candidates.extend(path for path in directory.glob(pattern) if path.is_file())
    for path in sorted(set(candidates), key=lambda p: str(p).lower()):
        path_text = str(path)
        status = "imported" if path_text in imported_paths else "still_needed"
        if path.name in {LEDGER_PATH.name, SUMMARY_PATH.name, INVENTORY_PATH.name}:
            status = "primary_tracking_artifact"
        rows.append(
            {
                "artifact_path": path_text,
                "artifact_name": path.name,
                "artifact_status": status,
                "size_bytes": path.stat().st_size,
                "last_modified": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            }
        )
    return pd.DataFrame(rows)


def write_summary(ledger: pd.DataFrame, inventory: pd.DataFrame, path: Path) -> None:
    status_counts = ledger["status"].value_counts().to_dict() if not ledger.empty else {}
    branch_counts = ledger["branch"].value_counts().to_dict() if not ledger.empty else {}
    promising = ledger[ledger["status"].isin(["direct_promising", "sweep_promising", "freqai_promising"])]
    lines = [
        "# Trader Concept Lifecycle Summary",
        "",
        f"Generated: {_now()}",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in status_counts.items():
        lines.append(f"1. `{status}`: {count}")
    lines.extend(["", "## Branch Counts", ""])
    for branch, count in branch_counts.items():
        lines.append(f"1. `{branch}`: {count}")
    lines.extend(["", "## Current Promising Concepts", ""])
    if promising.empty:
        lines.append("No concepts currently meet the promising thresholds.")
    else:
        for _, row in promising.head(30).iterrows():
            lines.append(f"1. `{row['concept_id']}` / `{row['target_label']}`")
            lines.append(f"   - Status: `{row['status']}`")
            lines.append(f"   - Trader question: {row['trader_question']}")
            lines.append(f"   - Result: {row['direct_result'] or row['freqai_result']}")
            lines.append(f"   - Next: {row['next_action']}")
    lines.extend(
        [
            "",
            "## Evidence Inventory",
            "",
            f"1. Total artifacts inventoried: {len(inventory)}",
            f"2. Imported artifacts: {int((inventory['artifact_status'] == 'imported').sum()) if not inventory.empty else 0}",
            f"3. Still needing review/import: {int((inventory['artifact_status'] == 'still_needed').sum()) if not inventory.empty else 0}",
            "",
            "Old reports and queues were not deleted. They are evidence paths until the user approves cleanup.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _cycle_action(status: str, concept_id: str) -> str:
    if status == "sweep_promising":
        return "freqai_validate"
    if status == "direct_promising" and concept_id.startswith("feature_discovery__"):
        return "convert_feature_clue_to_named_concept"
    if status == "direct_promising":
        return "threshold_sweep_or_freqai_validate"
    if status == "needs_rework":
        return "rework_one_condition_and_retest"
    return "hold"


def build_next_cycle_plan(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame()
    quota = {
        "structure_volume": 5,
        "downside_risk": 5,
        "orderbook_state": 5,
        "regime_confluence": 4,
    }
    status_order = {"sweep_promising": 0, "direct_promising": 1, "needs_rework": 2}
    candidates = ledger[ledger["status"].isin(status_order)].copy()
    candidates["status_rank"] = candidates["status"].map(status_order)
    rows: list[pd.DataFrame] = []
    for branch, limit in quota.items():
        branch_rows = candidates[candidates["branch"].eq(branch)].copy()
        if branch_rows.empty:
            continue
        branch_rows = branch_rows.sort_values(["status_rank", "concept_id", "target_label"]).head(limit)
        rows.append(branch_rows)
    plan = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if plan.empty:
        return plan
    plan = plan.drop(columns=["status_rank"], errors="ignore")
    plan.insert(0, "cycle_priority", range(1, len(plan) + 1))
    plan["cycle_action"] = [
        _cycle_action(str(row.status), str(row.concept_id)) for row in plan.itertuples(index=False)
    ]
    plan["pass_fail_condition"] = [
        "Promote only if it beats random/shuffled/same-regime controls and keeps trader-readable logic."
        for _ in range(len(plan))
    ]
    return plan


def write_next_cycle_markdown(plan: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Trader Concept Next Cycle Plan",
        "",
        f"Generated: {_now()}",
        "",
        "This plan keeps branch quotas active so research does not over-focus on one early winner.",
        "",
    ]
    if plan.empty:
        lines.append("No eligible concepts were found for the next cycle.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    for _, row in plan.iterrows():
        lines.append(f"## {int(row['cycle_priority'])}. {row['concept_id']}")
        lines.append("")
        lines.append(f"1. Branch: `{row['branch']}`")
        lines.append(f"2. Current status: `{row['status']}`")
        lines.append(f"3. Action: `{row['cycle_action']}`")
        lines.append(f"4. Trader question: {row['trader_question']}")
        lines.append(f"5. Target: {row['target_label']}")
        lines.append(f"6. Current evidence: {row['direct_result'] or row['freqai_result']}")
        lines.append(f"7. Next step: {row['next_action']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _lead_family(row: pd.Series) -> str:
    concept_id = str(row["concept_id"]).lower()
    target = str(row["target_label"]).lower()
    question = str(row["trader_question"]).lower()
    text = f"{concept_id} {target} {question}"
    if "support_reclaim" in text:
        return "support_reclaim_after_downside_break"
    if "downside_exhaustion" in text:
        return "downside_break_exhaustion"
    if "fakeout" in text and "orderbook" in text:
        return "orderbook_fakeout_detection"
    if "fakeout" in text:
        return "structure_fakeout_detection"
    if "orderbook" in text and ("breakout_failure" in text or "vah_rejection" in text or "failed breakout" in text):
        return "orderbook_failed_breakout_rejection"
    if "orderbook" in text and ("breakdown" in text or "drawdown" in text or "crash" in text):
        return "orderbook_breakdown_risk"
    if "structure_vp_tree_breakdown_success" in text or "breakdown/crash continuation" in text:
        return "structure_breakdown_crash_detection"
    if "structure_vp_tree_breakout_failure" in text or "failed breakouts/rejections near resistance" in text:
        return "structure_resistance_rejection"
    if "structure_vp_tree_breakout_success" in text or "successful breakout/acceptance" in text:
        return "structure_breakout_acceptance"
    if "compression_breakout" in text or "compression breakout" in text:
        return "compression_breakout_release"
    if "compression_breakdown" in text or "compression breakdown" in text:
        return "compression_breakdown_release"
    if "range_high" in text or "range-high" in text:
        return "range_high_breakout_continuation"
    if "range_low" in text or "range-low" in text:
        return "range_low_breakdown_continuation"
    if "range_position" in text and "breakout" in text:
        return "range_high_breakout_continuation"
    if "range_position" in text and "breakdown" in text:
        return "range_low_breakdown_continuation"
    if "bull_trend_regime" in text:
        return "bull_trend_regime_continuation_filter"
    if "nearest_wall_distance" in text or "wall_distance" in text:
        return "orderbook_wall_distance_shift"
    if ("return_24h" in text or "return_6h" in text) and "breakout" in text:
        return "price_momentum_breakout_continuation"
    if ("return_24h" in text or "return_6h" in text) and "breakdown" in text:
        return "price_momentum_breakdown_continuation"
    if ("price_compression" in text or "px_range_1h" in text or "compression_24h" in text) and "drawdown" in text:
        return "volatility_expansion_drawdown_risk"
    if "support_breakdown" in text or "support breakdown" in text:
        return "structure_support_breakdown_risk"
    if "volume_bearish" in text or "volume_pressure_24h" in text and "breakdown" in text:
        return "bearish_volume_pressure_breakdown"
    if "volume_bullish" in text or "volume_pressure_6h" in text and "breakout" in text:
        return "bullish_volume_pressure_breakout"
    if "tlv2_support" in text:
        return "tlv2_support_distance_breakdown"
    if "tlv2_resistance_line" in text and "drawdown" in text:
        return "tlv2_resistance_line_drawdown"
    if "tlv2_resistance" in text:
        return "tlv2_resistance_distance_breakout_or_reject"
    if "ms_invalidation_level" in text and "drawdown" in text:
        return "market_structure_invalidation_drawdown"
    if "future_return" in text:
        return "structure_future_return_regime"
    if "future_drawdown" in text:
        return "structure_future_drawdown_regime"
    return str(row["concept_id"])


def build_independent_leads(ledger: pd.DataFrame) -> pd.DataFrame:
    promising_statuses = {"freqai_promising", "sweep_promising", "direct_promising"}
    frame = ledger[ledger["status"].isin(promising_statuses)].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["lead_family"] = frame.apply(_lead_family, axis=1)
    frame["status_rank"] = frame["status"].map(STATUS_RANK)
    grouped: list[dict[str, Any]] = []
    for lead_family, group in frame.groupby("lead_family", sort=True):
        group = group.sort_values(["status_rank", "rows"], ascending=[False, False])
        best = group.iloc[0]
        grouped.append(
            {
                "lead_family": lead_family,
                "best_status": best["status"],
                "branch": best["branch"],
                "record_count": len(group),
                "best_trader_question": best["trader_question"],
                "best_target_label": best["target_label"],
                "best_result": best["freqai_result"] or best["direct_result"],
                "best_baseline_result": best["baseline_result"],
                "best_control_result": best["control_result"],
                "evidence_paths": "; ".join(sorted(set("; ".join(group["evidence_paths"].fillna("")).split("; ")))),
                "next_action": _independent_next_action(best["status"], lead_family),
                "last_updated": _now(),
            }
        )
    result = pd.DataFrame(grouped)
    result["rank"] = result["best_status"].map(STATUS_RANK)
    result = result.sort_values(["rank", "record_count", "lead_family"], ascending=[False, False, True])
    result = result.drop(columns=["rank"])
    result.insert(0, "lead_priority", range(1, len(result) + 1))
    return result


def _independent_next_action(status: str, lead_family: str) -> str:
    if status == "freqai_promising":
        return "Park as a working lead, compare sibling scopes, and design one clean direct rule/report for the family."
    if status == "sweep_promising":
        return "Run focused FreqAI validation or compare with already scored sibling profiles."
    if lead_family.startswith("feature_discovery"):
        return "Convert the feature clue into a named trader concept."
    return "Retest with direct controls or threshold sweep before promoting."


def write_independent_leads_markdown(leads: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Independent Promising Trading Leads",
        "",
        f"Generated: {_now()}",
        "",
        "These are grouped lead families, not duplicate FreqAI scopes. The aim is to build toward 20 independent promising concepts.",
        "",
    ]
    if leads.empty:
        lines.append("No independent promising leads found yet.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    for _, row in leads.iterrows():
        lines.append(f"## {int(row['lead_priority'])}. {row['lead_family']}")
        lines.append("")
        lines.append(f"1. Best status: `{row['best_status']}`")
        lines.append(f"2. Branch: `{row['branch']}`")
        lines.append(f"3. Evidence records: {row['record_count']}")
        lines.append(f"4. Trader question: {row['best_trader_question']}")
        lines.append(f"5. Target: {row['best_target_label']}")
        lines.append(f"6. Best result: {row['best_result']}")
        lines.append(f"7. Next action: {row['next_action']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _confidence_tier(row: pd.Series) -> str:
    status = str(row["best_status"])
    records = int(row["record_count"])
    if status == "freqai_promising" and records >= 3:
        return "A - working lead"
    if status == "freqai_promising":
        return "B - promising validation"
    if status == "sweep_promising":
        return "B - sweep-promising"
    if status == "direct_promising" and records >= 3:
        return "C - strong feature clue"
    return "D - convert or rework"


def _card_next_stage(row: pd.Series) -> str:
    tier = _confidence_tier(row)
    lead_family = str(row["lead_family"])
    if tier.startswith("A"):
        return "Park as working, then design one clean direct-test report that explains the lead family in trader language."
    if tier.startswith("B") and str(row["best_status"]) == "sweep_promising":
        return "Run or compare focused FreqAI validation, then promote only if it improves over price/structure controls."
    if tier.startswith("B"):
        return "Retest sibling scopes and convert the best scope into a direct-test concept card."
    if "feature_discovery" in lead_family:
        return "Convert from raw feature clue into a named hypothesis with setup, trigger, target, and controls."
    return "Rework into a cleaner hypothesis or split into direction-specific versions."


def _compact_evidence_paths(paths_text: str, limit: int = 4) -> str:
    paths = [path for path in str(paths_text).split("; ") if path and path != "nan"]
    priority = (
        [path for path in paths if "freqai_research_results_ledger.csv" in path]
        + [path for path in paths if "threshold_sweep" in path]
        + [path for path in paths if "direct_tests" in path]
        + [path for path in paths if "feature_discovery" in path]
    )
    compact: list[str] = []
    for path in priority:
        if path not in compact:
            compact.append(path)
    return "; ".join(compact[:limit])


def build_concept_cards(leads: pd.DataFrame) -> pd.DataFrame:
    if leads.empty:
        return pd.DataFrame()
    cards: list[dict[str, Any]] = []
    for _, row in leads.iterrows():
        lead_family = str(row["lead_family"])
        override = LEAD_CARD_OVERRIDES.get(lead_family, {})
        cards.append(
            {
                "concept_id": lead_family,
                "concept_title": override.get("concept_title", lead_family.replace("_", " ").title()),
                "branch": row["branch"],
                "confidence_tier": _confidence_tier(row),
                "lifecycle_status": row["best_status"],
                "trader_question": override.get("trader_question", str(row["best_trader_question"])),
                "visible_market_state": override.get("visible_market_state", "See source-detail blocks and evidence rows."),
                "source_detail_blocks": override.get("source_detail_blocks", str(row["branch"])),
                "feature_columns_or_groups": "See lifecycle ledger rows for exact columns and feature profile ids.",
                "target_label": override.get("target_label", str(row["best_target_label"])),
                "best_result": row["best_result"],
                "baseline_result": row["best_baseline_result"],
                "control_result": row["best_control_result"],
                "record_count": row["record_count"],
                "evidence_paths": _compact_evidence_paths(str(row["evidence_paths"])),
                "development_note": override.get("development_note", str(row["next_action"])),
                "next_stage": _card_next_stage(row),
                "last_updated": _now(),
            }
        )
    cards_df = pd.DataFrame(cards)
    tier_order = {
        "A - working lead": 0,
        "B - promising validation": 1,
        "B - sweep-promising": 2,
        "C - strong feature clue": 3,
        "D - convert or rework": 4,
    }
    cards_df["_tier_order"] = cards_df["confidence_tier"].map(tier_order).fillna(9)
    cards_df = cards_df.sort_values(["_tier_order", "record_count", "concept_id"], ascending=[True, False, True])
    cards_df = cards_df.drop(columns=["_tier_order"])
    cards_df.insert(0, "concept_priority", range(1, len(cards_df) + 1))
    return cards_df


def write_concept_cards_markdown(cards: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Trader Concept Cards",
        "",
        f"Generated: {_now()}",
        "",
        "These are cleaned concept cards built from the lifecycle ledger. They are research leads, not trading rules.",
        "",
    ]
    if cards.empty:
        lines.append("No concept cards available.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    for _, row in cards.iterrows():
        lines.append(f"## {int(row['concept_priority'])}. {row['concept_title']}")
        lines.append("")
        lines.append(f"1. Concept id: `{row['concept_id']}`")
        lines.append(f"2. Branch: `{row['branch']}`")
        lines.append(f"3. Confidence: `{row['confidence_tier']}`")
        lines.append(f"4. Trader question: {row['trader_question']}")
        lines.append(f"5. What a trader saw: {row['visible_market_state']}")
        lines.append(f"6. Sources: {row['source_detail_blocks']}")
        lines.append(f"7. Asked afterwards: {row['target_label']}")
        lines.append(f"8. Best result: {row['best_result']}")
        control_text = f"{row['baseline_result']} {row['control_result']}".strip()
        lines.append(f"9. Controls/baseline: {control_text}")
        lines.append(f"10. Development note: {row['development_note']}")
        lines.append(f"11. Next stage: {row['next_stage']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_ledger(feature_limit: int, freqai_limit: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    direct_paths = _all_reports("trader_confluence_direct_tests_*_summary.csv")
    sweep_paths = _all_reports("trader_confluence_threshold_sweep_*_collapsed_pass_summary.csv")
    feature_paths = _all_reports("feature_discovery_screen_*_combined.csv")
    freqai_path = REPORTS_DIR / "freqai_research_results_ledger.csv"

    records: list[ConceptRecord] = []
    for direct_path in direct_paths:
        records.extend(import_direct_tests(direct_path))
    for sweep_path in sweep_paths:
        records.extend(import_threshold_sweep(sweep_path))
    records.extend(import_feature_discovery_paths(feature_paths, feature_limit))
    records.extend(import_freqai_ledger(freqai_path if freqai_path.exists() else None, freqai_limit))
    merged = _merge_records(records)
    ledger = pd.DataFrame([asdict(record) for record in merged])
    imported_paths = {
        str(path)
        for path in (*direct_paths, *sweep_paths, *feature_paths, freqai_path)
        if path is not None and path.exists()
    }
    inventory = build_inventory(imported_paths)
    return ledger, inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the trader concept lifecycle ledger from existing research evidence.")
    parser.add_argument("--feature-limit", type=int, default=60, help="Maximum feature-discovery rows to import.")
    parser.add_argument("--freqai-limit", type=int, default=200, help="Maximum FreqAI ledger rows to import.")
    parser.add_argument("--print-summary-json", action="store_true", help="Print a small JSON summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ledger, inventory = build_ledger(feature_limit=args.feature_limit, freqai_limit=args.freqai_limit)
    ledger.to_csv(LEDGER_PATH, index=False)
    inventory.to_csv(INVENTORY_PATH, index=False)
    write_summary(ledger, inventory, SUMMARY_PATH)
    next_cycle = build_next_cycle_plan(ledger)
    next_cycle.to_csv(NEXT_CYCLE_CSV_PATH, index=False)
    write_next_cycle_markdown(next_cycle, NEXT_CYCLE_MD_PATH)
    independent_leads = build_independent_leads(ledger)
    independent_leads.to_csv(INDEPENDENT_LEADS_CSV_PATH, index=False)
    write_independent_leads_markdown(independent_leads, INDEPENDENT_LEADS_MD_PATH)
    concept_cards = build_concept_cards(independent_leads)
    concept_cards.to_csv(CONCEPT_CARDS_CSV_PATH, index=False)
    write_concept_cards_markdown(concept_cards, CONCEPT_CARDS_MD_PATH)
    if args.print_summary_json:
        payload = {
            "ledger_path": str(LEDGER_PATH),
            "summary_path": str(SUMMARY_PATH),
            "inventory_path": str(INVENTORY_PATH),
            "next_cycle_csv_path": str(NEXT_CYCLE_CSV_PATH),
            "next_cycle_md_path": str(NEXT_CYCLE_MD_PATH),
            "independent_leads_csv_path": str(INDEPENDENT_LEADS_CSV_PATH),
            "independent_leads_md_path": str(INDEPENDENT_LEADS_MD_PATH),
            "concept_cards_csv_path": str(CONCEPT_CARDS_CSV_PATH),
            "concept_cards_md_path": str(CONCEPT_CARDS_MD_PATH),
            "rows": int(len(ledger)),
            "status_counts": ledger["status"].value_counts().to_dict() if not ledger.empty else {},
            "branch_counts": ledger["branch"].value_counts().to_dict() if not ledger.empty else {},
            "inventory_rows": int(len(inventory)),
            "next_cycle_rows": int(len(next_cycle)),
            "independent_leads": int(len(independent_leads)),
            "concept_cards": int(len(concept_cards)),
        }
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

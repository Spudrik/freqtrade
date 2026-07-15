from __future__ import annotations

from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
import argparse
import csv
import hashlib
import io
import json
import re
import sqlite3
import zipfile

from .builder import default_paths
from . import gkg_document_normalize as gkg


DEFAULT_RAW_DIR = Path("D:/FreqTradeStuffLargeData/research_news_data/gdelt/raw/gkg")
DEFAULT_OUTPUT_DIR = Path("C:/FreqTradeStuff/user_data/research_news_data/context_features/news_gkg_hourly")
CSV_FIELD_SIZE_LIMIT = 128 * 1024 * 1024

TOPIC_RULES: dict[str, dict[str, re.Pattern[str]]] = {
    "war_geopolitics": {
        "entity": re.compile(r"\b(russia|ukraine|iran|israel|taiwan|china|nato|hormuz|gaza|military|army|missile|sanction)\w*\b", re.I),
        "action": re.compile(r"\b(war|invasion|attack|airstrike|missile|conflict|sanction|embargo|escalat|ceasefire|troop|drone)\w*\b", re.I),
    },
    "banking_credit": {
        "entity": re.compile(r"\b(bank|lender|credit|deposit|debt|bond|loan|liquidit|default|insolv)\w*\b", re.I),
        "action": re.compile(r"\b(default|insolv|liquidit|bailout|rescue|downgrade|contagion|credit crunch|deposit run|bank run|facility)\w*\b", re.I),
    },
    "oil_energy": {
        "entity": re.compile(r"\b(oil|crude|brent|wti|opec|pipeline|tanker|refiner|gas|lng|energy|inventory|barrel)\w*\b", re.I),
        "action": re.compile(r"\b(cut|outage|disrupt|embargo|sanction|inventory|supply|demand|restore|resume|shutdown|blockade|production)\w*\b", re.I),
    },
    "regulation": {
        "entity": re.compile(r"\b(sec|cftc|regulat|regulator|court|lawsuit|congress|parliament|treasury|doj|fca|esma)\w*\b", re.I),
        "action": re.compile(r"\b(enforcement|charge|fine|ban|approve|reject|rule|compliance|settlement|investigation|license)\w*\b", re.I),
    },
    "macro_policy": {
        "entity": re.compile(r"\b(fed|fomc|ecb|central bank|treasury|inflation|cpi|ppi|jobs|payroll|unemployment|gdp|recession|yield|rate)\w*\b", re.I),
        "action": re.compile(r"\b(hike|cut|tighten|ease|stimulus|liquidity|minutes|speech|forecast|surprise|slowdown|growth)\w*\b", re.I),
    },
    "security_cyber": {
        "entity": re.compile(r"\b(exchange|wallet|protocol|network|system|cyber|ransomware|malware|hacker)\w*\b", re.I),
        "action": re.compile(r"\b(hack|exploit|breach|stolen|phishing|vulnerability|attack|compromise|theft)\w*\b", re.I),
    },
    "crypto_market": {
        "entity": re.compile(r"\b(bitcoin|btc|ethereum|crypto|cryptocurrency|stablecoin|binance|coinbase|tether|defi|blockchain)\b", re.I),
        "action": re.compile(r"\b(etf|liquidat|exchange|regulat|hack|adoption|mining|wallet|token|stablecoin|blockchain)\w*\b", re.I),
    },
}

DIRECTION_RULES: dict[str, re.Pattern[str]] = {
    "stress_increase": re.compile(r"\b(crisis|default|insolv|collapse|contagion|run|emergency|downgrade|selloff|panic|plunge)\w*\b", re.I),
    "stress_relief": re.compile(r"\b(rescue|bailout|support|facility|stabiliz|relief|recover|reopen|calm|backstop)\w*\b", re.I),
    "supply_disruption": re.compile(r"\b(outage|disrupt|shutdown|blockade|embargo|sanction|cut supply|halt|strike|attack)\w*\b", re.I),
    "supply_restoration": re.compile(r"\b(resume|restore|restart|reopen|increase production|return to service)\w*\b", re.I),
    "policy_tightening": re.compile(r"\b(hike|tighten|restrict|raise rates|hawkish|quantitative tightening)\w*\b", re.I),
    "policy_easing": re.compile(r"\b(cut rates|ease|stimulus|dovish|liquidity injection|quantitative easing)\w*\b", re.I),
    "regulatory_pressure": re.compile(r"\b(enforcement|charge|fine|ban|lawsuit|probe|investigation|reject)\w*\b", re.I),
    "regulatory_clarity": re.compile(r"\b(approve|approval|license|framework|guidance|settlement|rulemaking)\w*\b", re.I),
    "conflict_escalation": re.compile(r"\b(escalat|attack|strike|missile|invasion|troop|sanction|retaliat)\w*\b", re.I),
    "conflict_deescalation": re.compile(r"\b(ceasefire|peace|talks|truce|withdraw|deescalat|deal)\w*\b", re.I),
}

SEVERITY_TERMS = re.compile(
    r"\b(crisis|default|insolv|collapse|war|attack|strike|missile|death|dead|kill|emergency|"
    r"sanction|embargo|breach|hack|exploit|contagion|downgrade|outage|shutdown)\w*\b",
    re.I,
)

HIGH_QUALITY_SOURCES = re.compile(
    r"\b(reuters|bloomberg|wsj|ft\.com|financialtimes|cnbc|marketwatch|apnews|"
    r"federalreserve|ecb|imf|worldbank|treasury|sec\.gov|cftc|opec|eia|energy\.gov|coindesk)\b",
    re.I,
)

SOURCE_GROUP_RULES: dict[str, re.Pattern[str]] = {
    "wire_major": re.compile(r"\b(reuters|apnews|bloomberg|afp|associatedpress)\b", re.I),
    "market_financial": re.compile(r"\b(bloomberg|wsj|ft\.com|financialtimes|cnbc|marketwatch|seekingalpha|zerohedge|yahoo|investing|barrons)\b", re.I),
    "official_policy": re.compile(r"\b(federalreserve|ecb|imf|worldbank|treasury|sec\.gov|cftc|whitehouse|opec|eia|energy\.gov|bis)\b", re.I),
    "crypto_native": re.compile(r"\b(coindesk|cointelegraph|decrypt|theblock|bitcoinmagazine|cryptoslate|beincrypto)\b", re.I),
    "regional_local": re.compile(r"\b(tribune|herald|gazette|post|times|daily|news|observer|journal|star|sun)\b", re.I),
}

BEHAVIOR_RULES: dict[str, dict[str, re.Pattern[str]]] = {
    "panic_fear": {
        "signal": re.compile(r"\b(panic|fear|crisis|collapse|plunge|selloff|turmoil|chaos|emergency|worst|meltdown)\w*\b", re.I),
    },
    "hedge_safe_haven": {
        "signal": re.compile(r"\b(hedge|safe haven|gold|treasury|dollar|risk off|flight to safety|defensive|protection)\w*\b", re.I),
    },
    "risk_on_speculation": {
        "signal": re.compile(r"\b(rally|risk on|surge|boom|bullish|optimis|euphoria|speculat|fomo|record high|melt up)\w*\b", re.I),
    },
    "liquidity_squeeze": {
        "signal": re.compile(r"\b(liquidity|funding stress|margin call|credit crunch|repo|facility|cash crunch|withdrawal|redemption)\w*\b", re.I),
    },
    "contagion_risk": {
        "signal": re.compile(r"\b(contagion|spillover|systemic|counterparty|domino|spread to|ripple effect)\w*\b", re.I),
    },
    "supply_shock": {
        "signal": re.compile(r"\b(outage|shortage|supply shock|disruption|embargo|blockade|halt|shutdown|pipeline|tanker)\w*\b", re.I),
    },
    "demand_shock": {
        "signal": re.compile(r"\b(demand slump|weak demand|slowdown|contraction|recession|consumer weakness|inventory build)\w*\b", re.I),
    },
    "policy_shock": {
        "signal": re.compile(r"\b(surprise hike|surprise cut|emergency meeting|unexpected decision|hawkish shock|dovish shock|policy shock)\w*\b", re.I),
    },
    "regulatory_crackdown": {
        "signal": re.compile(r"\b(crackdown|enforcement|charges|lawsuit|ban|fine|probe|investigation|cease and desist)\w*\b", re.I),
    },
    "regulatory_clarity": {
        "signal": re.compile(r"\b(approval|approved|license|framework|guidance|settlement|rulebook|legal clarity)\w*\b", re.I),
    },
    "institutional_adoption": {
        "signal": re.compile(r"\b(institutional|etf|custody|asset manager|blackrock|fidelity|fund inflow|allocation|treasury reserve)\w*\b", re.I),
    },
    "retail_froth": {
        "signal": re.compile(r"\b(retail trader|meme|viral|reddit|tiktok|fomo|mania|pump|celebrity|influencer)\w*\b", re.I),
    },
    "conflict_escalation_behaviour": {
        "signal": re.compile(r"\b(escalat|attack|missile|airstrike|invasion|retaliat|troop build|sanction threat)\w*\b", re.I),
    },
    "conflict_deescalation_behaviour": {
        "signal": re.compile(r"\b(ceasefire|truce|peace talks|withdrawal|deal reached|deescalat|hostage release)\w*\b", re.I),
    },
}


@dataclass
class HourAgg:
    files: set[str] = field(default_factory=set)
    doc_count: int = 0
    unique_urls: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    high_quality_sources: set[str] = field(default_factory=set)
    tone_sum: float = 0.0
    tone_count: int = 0
    negative_sum: float = 0.0
    positive_sum: float = 0.0
    word_count_sum: float = 0.0
    topic_counts: Counter[str] = field(default_factory=Counter)
    topic_sources: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    topic_high_quality: set[str] = field(default_factory=set)
    topic_severity_sum: Counter[str] = field(default_factory=Counter)
    topic_severity_max: Counter[str] = field(default_factory=Counter)
    direction_scores: Counter[str] = field(default_factory=Counter)
    topic_direction_scores: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    source_group_doc_counts: Counter[str] = field(default_factory=Counter)
    source_group_sources: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    behavior_counts: Counter[str] = field(default_factory=Counter)
    behavior_sources: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    behavior_high_quality: set[str] = field(default_factory=set)
    behavior_severity_sum: Counter[str] = field(default_factory=Counter)
    behavior_severity_max: Counter[str] = field(default_factory=Counter)
    topic_behavior_counts: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    topic_behavior_severity_sum: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream GKG raw ZIPs into compact trader-readable 1h metadata features.")
    parser.add_argument("--app-dir", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--gdelt-db", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--tag", default="smoke")
    parser.add_argument("--audit-samples-per-topic", type=int, default=25)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--progress-every-files", type=int, default=250)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-backend", choices=("thread", "process"), default="thread")
    parser.add_argument("--skip-csv", action="store_true", help="Write parquet, audit CSV, and summary only.")
    args = parser.parse_args()

    start = parse_utc(args.start)
    end = parse_utc(args.end)
    if end <= start:
        raise SystemExit("--end must be after --start")
    if args.limit_files is not None and args.limit_files < 0:
        raise SystemExit("--limit-files must be non-negative")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    paths = default_paths(args.app_dir)
    gdelt_db = args.gdelt_db or paths.gdelt_db
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raise_csv_field_size_limit()

    zip_paths = filter_zip_paths(args.raw_dir, start, end)
    if args.limit_files is not None:
        zip_paths = zip_paths[: args.limit_files]

    hourly: dict[datetime, HourAgg] = {hour: HourAgg() for hour in iter_hours(start, end)}
    audit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_read = 0
    rows_skipped = 0
    corrupt_files: list[str] = []

    completed_files = 0
    if args.workers == 1:
        for zip_path in zip_paths:
            result = process_zip(zip_path, args.audit_samples_per_topic)
            completed_files += 1
            rows_read, rows_skipped = merge_result(hourly, audit, result, rows_read, rows_skipped, corrupt_files, args.audit_samples_per_topic)
            emit_progress(completed_files, len(zip_paths), rows_read, corrupt_files, args.progress_every_files)
    else:
        executor_cls = ProcessPoolExecutor if args.worker_backend == "process" else ThreadPoolExecutor
        with executor_cls(max_workers=args.workers) as pool:
            futures = {pool.submit(process_zip, zip_path, args.audit_samples_per_topic): zip_path for zip_path in zip_paths}
            for future in as_completed(futures):
                result = future.result()
                completed_files += 1
                rows_read, rows_skipped = merge_result(hourly, audit, result, rows_read, rows_skipped, corrupt_files, args.audit_samples_per_topic)
                emit_progress(completed_files, len(zip_paths), rows_read, corrupt_files, args.progress_every_files)

    frame = build_feature_frame(hourly, start, end)
    frame = join_existing_gdelt_hourly(frame, gdelt_db, start, end)

    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.tag)
    parquet_path = args.output_dir / f"gkg_hourly_metadata_{safe_tag}.parquet"
    csv_path = args.output_dir / f"gkg_hourly_metadata_{safe_tag}.csv"
    audit_path = args.output_dir / f"gkg_hourly_metadata_audit_{safe_tag}.csv"
    summary_path = args.output_dir / f"gkg_hourly_metadata_summary_{safe_tag}.json"

    frame.to_parquet(parquet_path, index=False)
    if not args.skip_csv:
        frame.to_csv(csv_path, index=False)
    audit_rows = [row for topic_rows in audit.values() for row in topic_rows]
    import pandas as pd

    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)

    summary = {
        "feature_schema_version": "gkg_hourly_metadata_smart_v4",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "tag": safe_tag,
        "raw_dir": str(args.raw_dir),
        "workers": int(args.workers),
        "worker_backend": args.worker_backend,
        "csv_written": not args.skip_csv,
        "files_planned": len(zip_paths),
        "expected_15m_files": expected_15m_files(start, end),
        "file_coverage_ratio": round(len(zip_paths) / max(1, expected_15m_files(start, end)), 6),
        "rows_read": rows_read,
        "rows_skipped": rows_skipped,
        "corrupt_files": corrupt_files[:50],
        "feature_rows": int(len(frame)),
        "feature_columns": int(len(frame.columns)),
        "nonzero_topic_hours": {
            topic: int((frame[f"news_{topic}_doc_count_1h"] > 0).sum())
            for topic in TOPIC_RULES
            if f"news_{topic}_doc_count_1h" in frame
        },
        "outputs": {
            "parquet": str(parquet_path),
            "csv": str(csv_path) if not args.skip_csv else None,
            "audit_csv": str(audit_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def process_zip(zip_path: Path, audit_limit: int) -> dict[str, Any]:
    file_hour = hour_floor(parse_stamp(zip_path.name))
    agg = HourAgg()
    agg.files.add(zip_path.name)
    audit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_read = 0
    rows_skipped = 0
    corrupt: str | None = None
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for member in [name for name in archive.namelist() if not name.endswith("/")]:
                with archive.open(member) as handle:
                    for raw_line in handle:
                        values = raw_line.decode("utf-8", errors="replace").rstrip("\r\n").split("\t")
                        parsed = parse_compact_gkg(values, zip_path.name)
                        if parsed is None:
                            rows_skipped += 1
                            continue
                        rows_read += 1
                        add_record(agg, parsed)
                        add_audit_samples(audit, parsed, file_hour, audit_limit)
    except (zipfile.BadZipFile, OSError) as exc:
        corrupt = f"{zip_path.name}: {exc}"
    return {
        "hour": file_hour,
        "agg": agg,
        "audit": dict(audit),
        "rows_read": rows_read,
        "rows_skipped": rows_skipped,
        "corrupt": corrupt,
    }


def merge_result(
    hourly: dict[datetime, HourAgg],
    audit: dict[str, list[dict[str, Any]]],
    result: dict[str, Any],
    rows_read: int,
    rows_skipped: int,
    corrupt_files: list[str],
    audit_limit: int,
) -> tuple[int, int]:
    hour = result["hour"]
    if hour not in hourly:
        return rows_read, rows_skipped
    target = hourly[hour]
    source: HourAgg = result["agg"]
    merge_hour(target, source)
    rows_read += int(result["rows_read"])
    rows_skipped += int(result["rows_skipped"])
    if result.get("corrupt"):
        corrupt_files.append(str(result["corrupt"]))
    for topic, rows in dict(result["audit"]).items():
        remaining = max(0, audit_limit - len(audit[topic]))
        if remaining:
            audit[topic].extend(rows[:remaining])
    return rows_read, rows_skipped


def merge_hour(target: HourAgg, source: HourAgg) -> None:
    target.files.update(source.files)
    target.doc_count += source.doc_count
    target.unique_urls.update(source.unique_urls)
    target.sources.update(source.sources)
    target.high_quality_sources.update(source.high_quality_sources)
    target.tone_sum += source.tone_sum
    target.tone_count += source.tone_count
    target.negative_sum += source.negative_sum
    target.positive_sum += source.positive_sum
    target.word_count_sum += source.word_count_sum
    target.topic_counts.update(source.topic_counts)
    target.topic_high_quality.update(source.topic_high_quality)
    target.topic_severity_sum.update(source.topic_severity_sum)
    for topic, value in source.topic_severity_max.items():
        target.topic_severity_max[topic] = max(target.topic_severity_max[topic], value)
    target.direction_scores.update(source.direction_scores)
    target.source_group_doc_counts.update(source.source_group_doc_counts)
    target.behavior_counts.update(source.behavior_counts)
    target.behavior_high_quality.update(source.behavior_high_quality)
    target.behavior_severity_sum.update(source.behavior_severity_sum)
    for behavior, value in source.behavior_severity_max.items():
        target.behavior_severity_max[behavior] = max(target.behavior_severity_max[behavior], value)
    for topic, sources in source.topic_sources.items():
        target.topic_sources[topic].update(sources)
    for topic, counter in source.topic_direction_scores.items():
        target.topic_direction_scores[topic].update(counter)
    for group, sources in source.source_group_sources.items():
        target.source_group_sources[group].update(sources)
    for behavior, sources in source.behavior_sources.items():
        target.behavior_sources[behavior].update(sources)
    for topic, counter in source.topic_behavior_counts.items():
        target.topic_behavior_counts[topic].update(counter)
    for topic, counter in source.topic_behavior_severity_sum.items():
        target.topic_behavior_severity_sum[topic].update(counter)


def emit_progress(
    completed_files: int,
    total_files: int,
    rows_read: int,
    corrupt_files: list[str],
    progress_every_files: int,
) -> None:
    if not progress_every_files or completed_files % progress_every_files:
        return
    print(
        json.dumps(
            {
                "progress_files": completed_files,
                "files_planned": total_files,
                "rows_read": rows_read,
                "corrupt_files": len(corrupt_files),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def parse_compact_gkg(values: list[str], zip_name: str) -> dict[str, Any] | None:
    if len(values) <= gkg.COL_DOCUMENT_IDENTIFIER:
        return None
    record_id = cell(values, gkg.COL_GKG_RECORD_ID)
    url = cell(values, gkg.COL_DOCUMENT_IDENTIFIER)
    if not record_id or not url:
        return None
    source = cell(values, gkg.COL_SOURCE_NAME)
    tone = gkg.parse_v2_tone(cell(values, gkg.COL_V2_TONE))
    text = compact_text(
        url=url,
        source=source,
        themes=f"{cell(values, gkg.COL_THEMES)} {cell(values, gkg.COL_V2_THEMES)}",
        persons=f"{cell(values, gkg.COL_PERSONS)} {cell(values, gkg.COL_V2_PERSONS)}",
        orgs=f"{cell(values, gkg.COL_ORGS)} {cell(values, gkg.COL_V2_ORGS)}",
        locations=f"{cell(values, gkg.COL_LOCATIONS)} {cell(values, gkg.COL_V2_LOCATIONS)}",
        all_names=cell(values, gkg.COL_ALL_NAMES),
    )
    topics, evidence = detect_topics(text)
    directions = detect_directions(text)
    behaviors = detect_behaviors(text)
    severity = severity_score(text, tone)
    source_quality = 1.5 if HIGH_QUALITY_SOURCES.search(source or url) else 1.0
    source_groups = detect_source_groups(source, url)
    return {
        "record_id": record_id,
        "zip_name": zip_name,
        "url": url,
        "url_hash": hashlib.sha256(url.encode("utf-8", errors="ignore")).hexdigest(),
        "source": source or "unknown",
        "topics": topics,
        "directions": directions,
        "behaviors": behaviors,
        "evidence": evidence,
        "severity": severity,
        "source_quality": source_quality,
        "source_groups": source_groups,
        "tone": tone.get("tone"),
        "positive_score": tone.get("positive_score"),
        "negative_score": tone.get("negative_score"),
        "word_count": tone.get("word_count"),
    }


def add_record(hour: HourAgg, row: dict[str, Any]) -> None:
    hour.doc_count += 1
    hour.unique_urls.add(str(row["url_hash"]))
    source = str(row["source"])
    hour.sources.add(source)
    if row["source_quality"] > 1.0:
        hour.high_quality_sources.add(source)
    for group in row["source_groups"]:
        hour.source_group_doc_counts[group] += 1
        hour.source_group_sources[group].add(source)
    if row.get("tone") is not None:
        hour.tone_sum += float(row["tone"])
        hour.tone_count += 1
    if row.get("positive_score") is not None:
        hour.positive_sum += float(row["positive_score"])
    if row.get("negative_score") is not None:
        hour.negative_sum += float(row["negative_score"])
    if row.get("word_count") is not None:
        hour.word_count_sum += float(row["word_count"])
    for direction in row["directions"]:
        hour.direction_scores[direction] += row["severity"]
    for behavior in row["behaviors"]:
        hour.behavior_counts[behavior] += 1
        hour.behavior_sources[behavior].add(source)
        hour.behavior_severity_sum[behavior] += row["severity"]
        hour.behavior_severity_max[behavior] = max(hour.behavior_severity_max[behavior], row["severity"])
        if row["source_quality"] > 1.0:
            hour.behavior_high_quality.add(behavior)
    for topic in row["topics"]:
        hour.topic_counts[topic] += 1
        hour.topic_sources[topic].add(source)
        hour.topic_severity_sum[topic] += row["severity"]
        hour.topic_severity_max[topic] = max(hour.topic_severity_max[topic], row["severity"])
        if row["source_quality"] > 1.0:
            hour.topic_high_quality.add(topic)
        for direction in row["directions"]:
            hour.topic_direction_scores[topic][direction] += row["severity"]
        for behavior in row["behaviors"]:
            hour.topic_behavior_counts[topic][behavior] += 1
            hour.topic_behavior_severity_sum[topic][behavior] += row["severity"]


def build_feature_frame(hourly: dict[datetime, HourAgg], start: datetime, end: datetime) -> pd.DataFrame:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    topic_windows: dict[str, deque[tuple[datetime, int, float, set[str]]]] = {topic: deque() for topic in TOPIC_RULES}
    behavior_windows: dict[str, deque[tuple[datetime, int, float, set[str]]]] = {behavior: deque() for behavior in BEHAVIOR_RULES}
    last_seen: dict[str, datetime] = {}
    behavior_last_seen: dict[str, datetime] = {}
    previous_severity: dict[str, deque[tuple[datetime, float]]] = {topic: deque() for topic in TOPIC_RULES}
    previous_behavior_severity: dict[str, deque[tuple[datetime, float]]] = {behavior: deque() for behavior in BEHAVIOR_RULES}

    for hour in iter_hours(start, end):
        agg = hourly.get(hour, HourAgg())
        row: dict[str, Any] = {
            "date": hour.isoformat(),
            "gkg_files_present_1h": len(agg.files),
            "gkg_expected_files_1h": 4,
            "gkg_files_present_ratio_1h": len(agg.files) / 4.0,
            "gkg_document_count_1h": agg.doc_count,
            "gkg_unique_url_count_1h": len(agg.unique_urls),
            "gkg_duplicate_ratio_1h": 0.0 if agg.doc_count == 0 else 1.0 - (len(agg.unique_urls) / agg.doc_count),
            "gkg_unique_source_count_1h": len(agg.sources),
            "gkg_high_quality_source_count_1h": len(agg.high_quality_sources),
            "gkg_avg_tone_1h": agg.tone_sum / agg.tone_count if agg.tone_count else 0.0,
            "gkg_positive_tone_sum_1h": agg.positive_sum,
            "gkg_negative_tone_sum_1h": agg.negative_sum,
            "gkg_avg_word_count_1h": agg.word_count_sum / agg.doc_count if agg.doc_count else 0.0,
        }
        for direction in DIRECTION_RULES:
            row[f"news_{direction}_score_1h"] = float(agg.direction_scores[direction])
        for group in SOURCE_GROUP_RULES:
            docs = int(agg.source_group_doc_counts[group])
            row[f"news_source_group_{group}_doc_count_1h"] = docs
            row[f"news_source_group_{group}_source_count_1h"] = len(agg.source_group_sources.get(group, set()))
            row[f"news_source_group_{group}_share_1h"] = 0.0 if agg.doc_count == 0 else docs / agg.doc_count

        for behavior in BEHAVIOR_RULES:
            count = int(agg.behavior_counts[behavior])
            severity_sum = float(agg.behavior_severity_sum[behavior])
            sources = set(agg.behavior_sources.get(behavior, set()))
            queue = behavior_windows[behavior]
            queue.append((hour, count, severity_sum, sources))
            while queue and queue[0][0] < hour - timedelta(hours=71):
                queue.popleft()
            severity_queue = previous_behavior_severity[behavior]
            severity_queue.append((hour, severity_sum))
            while severity_queue and severity_queue[0][0] < hour - timedelta(hours=24):
                severity_queue.popleft()
            age = None if behavior not in behavior_last_seen else int((hour - behavior_last_seen[behavior]).total_seconds() // 3600)
            if count > 0:
                behavior_last_seen[behavior] = hour
            count_6h, severity_6h, sources_6h, persistence_6h = rolling_parts(queue, hour, 6)
            count_24h, severity_24h, sources_24h, persistence_24h = rolling_parts(queue, hour, 24)
            count_72h, severity_72h, sources_72h, persistence_72h = rolling_parts(queue, hour, 72)
            prior_max_6h = max((item[1] for item in severity_queue if item[0] < hour and item[0] >= hour - timedelta(hours=6)), default=0.0)
            prior_max_24h = max((item[1] for item in severity_queue if item[0] < hour), default=0.0)
            row[f"news_behavior_{behavior}_doc_count_1h"] = count
            row[f"news_behavior_{behavior}_source_count_1h"] = len(sources)
            row[f"news_behavior_{behavior}_high_quality_flag_1h"] = 1 if behavior in agg.behavior_high_quality else 0
            row[f"news_behavior_{behavior}_severity_sum_1h"] = severity_sum
            row[f"news_behavior_{behavior}_severity_max_1h"] = float(agg.behavior_severity_max[behavior])
            row[f"news_behavior_{behavior}_intensity_1h"] = severity_sum * max(1, len(sources))
            row[f"news_behavior_{behavior}_doc_count_6h"] = count_6h
            row[f"news_behavior_{behavior}_doc_count_24h"] = count_24h
            row[f"news_behavior_{behavior}_doc_count_72h"] = count_72h
            row[f"news_behavior_{behavior}_intensity_6h"] = severity_6h * max(1, len(sources_6h))
            row[f"news_behavior_{behavior}_intensity_24h"] = severity_24h * max(1, len(sources_24h))
            row[f"news_behavior_{behavior}_intensity_72h"] = severity_72h * max(1, len(sources_72h))
            row[f"news_behavior_{behavior}_source_confluence_24h"] = len(sources_24h)
            row[f"news_behavior_{behavior}_persistence_hours_24h"] = persistence_24h
            row[f"news_behavior_{behavior}_persistence_hours_72h"] = persistence_72h
            row[f"news_behavior_{behavior}_severity_escalation_6h"] = max(0.0, severity_sum - prior_max_6h)
            row[f"news_behavior_{behavior}_severity_escalation_24h"] = max(0.0, severity_sum - prior_max_24h)
            row[f"news_behavior_{behavior}_first_mention_flag_1h"] = 1 if count > 0 and (age is None or age > 168) else 0
            row[f"news_behavior_{behavior}_age_hours"] = -1 if age is None else age

        for topic in TOPIC_RULES:
            count = int(agg.topic_counts[topic])
            severity_sum = float(agg.topic_severity_sum[topic])
            severity_max = float(agg.topic_severity_max[topic])
            sources = set(agg.topic_sources.get(topic, set()))
            queue = topic_windows[topic]
            queue.append((hour, count, severity_sum, sources))
            while queue and queue[0][0] < hour - timedelta(hours=71):
                queue.popleft()

            severity_queue = previous_severity[topic]
            severity_queue.append((hour, severity_sum))
            while severity_queue and severity_queue[0][0] < hour - timedelta(hours=24):
                severity_queue.popleft()
            prior_max_6h = max((item[1] for item in severity_queue if item[0] < hour and item[0] >= hour - timedelta(hours=6)), default=0.0)
            prior_max_24h = max((item[1] for item in severity_queue if item[0] < hour), default=0.0)
            count_6h, severity_6h, sources_6h, persistence_6h = rolling_parts(queue, hour, 6)
            count_24h, severity_24h, sources_24h, persistence_24h = rolling_parts(queue, hour, 24)
            count_72h, severity_72h, sources_72h, persistence_72h = rolling_parts(queue, hour, 72)

            age = None if topic not in last_seen else int((hour - last_seen[topic]).total_seconds() // 3600)
            first_flag = 1 if count > 0 and (age is None or age > 168) else 0
            if count > 0:
                last_seen[topic] = hour

            row[f"news_{topic}_doc_count_1h"] = count
            row[f"news_{topic}_source_count_1h"] = len(sources)
            row[f"news_{topic}_high_quality_source_flag_1h"] = 1 if topic in agg.topic_high_quality else 0
            row[f"news_{topic}_severity_sum_1h"] = severity_sum
            row[f"news_{topic}_severity_max_1h"] = severity_max
            row[f"news_{topic}_intensity_1h"] = severity_sum * max(1, len(sources))
            row[f"news_{topic}_doc_count_6h"] = count_6h
            row[f"news_{topic}_doc_count_24h"] = count_24h
            row[f"news_{topic}_doc_count_72h"] = count_72h
            row[f"news_{topic}_intensity_6h"] = severity_6h * max(1, len(sources_6h))
            row[f"news_{topic}_intensity_24h"] = severity_24h * max(1, len(sources_24h))
            row[f"news_{topic}_intensity_72h"] = severity_72h * max(1, len(sources_72h))
            row[f"news_{topic}_source_confluence_6h"] = len(sources_6h)
            row[f"news_{topic}_source_confluence_24h"] = len(sources_24h)
            row[f"news_{topic}_source_confluence_72h"] = len(sources_72h)
            row[f"news_{topic}_persistence_hours_6h"] = persistence_6h
            row[f"news_{topic}_persistence_hours_24h"] = persistence_24h
            row[f"news_{topic}_persistence_hours_72h"] = persistence_72h
            row[f"news_{topic}_first_mention_flag_1h"] = first_flag
            row[f"news_{topic}_first_mention_age_hours"] = -1 if age is None else age
            row[f"news_{topic}_severity_escalation_6h"] = max(0.0, severity_sum - prior_max_6h)
            row[f"news_{topic}_severity_escalation_24h"] = max(0.0, severity_sum - prior_max_24h)
            row[f"news_{topic}_follow_through_flag_24h"] = 1 if persistence_24h >= 3 and len(sources_24h) >= 3 else 0
            row[f"news_{topic}_high_quality_confirmed_1h"] = 1 if topic in agg.topic_high_quality and len(sources) >= 2 else 0
            for direction in DIRECTION_RULES:
                row[f"news_{topic}_{direction}_score_1h"] = float(agg.topic_direction_scores[topic][direction])
            for behavior in BEHAVIOR_RULES:
                row[f"news_{topic}_{behavior}_doc_count_1h"] = int(agg.topic_behavior_counts[topic][behavior])
                row[f"news_{topic}_{behavior}_severity_sum_1h"] = float(agg.topic_behavior_severity_sum[topic][behavior])
        rows.append(row)
    return pd.DataFrame(rows)


def rolling_parts(
    queue: deque[tuple[datetime, int, float, set[str]]],
    hour: datetime,
    window_hours: int,
) -> tuple[int, float, set[str], int]:
    floor = hour - timedelta(hours=window_hours - 1)
    rows = [item for item in queue if item[0] >= floor]
    count = sum(item[1] for item in rows)
    severity = sum(item[2] for item in rows)
    sources = set().union(*(item[3] for item in rows)) if rows else set()
    persistence = sum(1 for item in rows if item[1] > 0)
    return count, severity, sources, persistence


def join_existing_gdelt_hourly(frame: pd.DataFrame, db_path: Path, start: datetime, end: datetime) -> pd.DataFrame:
    import pandas as pd

    if not db_path.exists():
        return frame
    try:
        with sqlite3.connect(str(db_path)) as conn:
            gdelt = pd.read_sql_query(
                """
                SELECT date, event_count, conflict_event_count, protest_event_count,
                       coercion_event_count, sanctions_trade_url_count, oil_energy_url_count,
                       banking_credit_url_count, macro_url_count, crypto_url_count,
                       avg_tone_weighted, goldstein_weighted
                FROM gdelt_hourly_features
                WHERE date >= ? AND date < ?
                """,
                conn,
                params=(start.isoformat(), end.isoformat()),
            )
    except Exception:
        return frame
    if gdelt.empty:
        return frame
    gdelt = gdelt.rename(
        columns={
            "event_count": "gdelt_event_count_1h",
            "conflict_event_count": "gdelt_conflict_event_count_1h",
            "protest_event_count": "gdelt_protest_event_count_1h",
            "coercion_event_count": "gdelt_coercion_event_count_1h",
            "sanctions_trade_url_count": "gdelt_sanctions_trade_url_count_1h",
            "oil_energy_url_count": "gdelt_oil_energy_url_count_1h",
            "banking_credit_url_count": "gdelt_banking_credit_url_count_1h",
            "macro_url_count": "gdelt_macro_url_count_1h",
            "crypto_url_count": "gdelt_crypto_url_count_1h",
            "avg_tone_weighted": "gdelt_avg_tone_weighted_1h",
            "goldstein_weighted": "gdelt_goldstein_weighted_1h",
        }
    )
    merged = frame.merge(gdelt, on="date", how="left")
    for column in merged.columns:
        if column.startswith("gdelt_"):
            merged[column] = merged[column].fillna(0.0)
    return merged


def add_audit_samples(
    audit: dict[str, list[dict[str, Any]]],
    row: dict[str, Any],
    hour: datetime,
    limit: int,
) -> None:
    for topic in row["topics"]:
        rows = audit[topic]
        if len(rows) >= limit:
            continue
        rows.append(
            {
                "date": hour.isoformat(),
                "topic": topic,
                "source": row["source"],
                "severity": row["severity"],
                "directions": ";".join(row["directions"]),
                "evidence": ";".join(row["evidence"].get(topic, [])),
                "url": row["url"],
                "zip_name": row["zip_name"],
            }
        )


def detect_topics(text: str) -> tuple[list[str], dict[str, list[str]]]:
    topics: list[str] = []
    evidence: dict[str, list[str]] = {}
    for topic, rules in TOPIC_RULES.items():
        entity = rules["entity"].search(text)
        action = rules["action"].search(text)
        if entity and action:
            topics.append(topic)
            evidence[topic] = [entity.group(0), action.group(0)]
    return topics, evidence


def detect_directions(text: str) -> list[str]:
    return [name for name, pattern in DIRECTION_RULES.items() if pattern.search(text)]


def detect_behaviors(text: str) -> list[str]:
    return [name for name, rules in BEHAVIOR_RULES.items() if rules["signal"].search(text)]


def detect_source_groups(source: str, url: str) -> list[str]:
    text = f"{source} {url}"
    groups = [name for name, pattern in SOURCE_GROUP_RULES.items() if pattern.search(text)]
    return groups or ["other"]


def severity_score(text: str, tone: dict[str, float | None]) -> float:
    score = 1.0
    score += min(5, len(SEVERITY_TERMS.findall(text)))
    negative = tone.get("negative_score")
    if negative is not None:
        score += min(3.0, max(0.0, float(negative) / 5.0))
    return float(score)


def compact_text(
    *,
    url: str,
    source: str,
    themes: str,
    persons: str,
    orgs: str,
    locations: str,
    all_names: str,
) -> str:
    pieces = [
        url,
        source,
        themes,
        persons,
        orgs,
        locations,
        all_names,
    ]
    return re.sub(r"[_;,#|/\\-]+", " ", " ".join(pieces))


def filter_zip_paths(raw_dir: Path, start: datetime, end: datetime) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(raw_dir.glob("*.gkg.csv.zip")):
        try:
            stamp = parse_stamp(path.name)
        except ValueError:
            continue
        if start <= stamp < end:
            paths.append(path)
    return paths


def expected_15m_files(start: datetime, end: datetime) -> int:
    return int((end - start).total_seconds() // (15 * 60))


def iter_hours(start: datetime, end: datetime) -> Iterable[datetime]:
    current = hour_floor(start)
    while current < end:
        yield current
        current += timedelta(hours=1)


def hour_floor(value: datetime) -> datetime:
    return value.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def parse_stamp(name: str) -> datetime:
    match = re.search(r"(\d{14})", name)
    if not match:
        raise ValueError(f"missing GDELT stamp in {name}")
    return datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def parse_utc(value: str) -> datetime:
    clean = value.strip().replace("Z", "+00:00")
    if len(clean) == 10:
        clean = f"{clean}T00:00:00+00:00"
    parsed = datetime.fromisoformat(clean)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(second=0, microsecond=0)


def cell(values: list[str], index: int) -> str:
    return values[index].strip() if index < len(values) else ""


def raise_csv_field_size_limit() -> None:
    limit = CSV_FIELD_SIZE_LIMIT
    while limit > csv.field_size_limit():
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = int(limit / 10)


if __name__ == "__main__":
    raise SystemExit(main())

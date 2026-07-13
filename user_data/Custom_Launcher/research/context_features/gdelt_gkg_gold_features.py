from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
import argparse
import csv
import json
import sqlite3

from .builder import default_paths
from . import gdelt_gkg_normalized_schema as schema


DEFAULT_OUTPUT_DIR = Path("C:/FreqTradeStuff/user_data/research_news_data/context_features/reports")
GOLD_TOPIC_TABLE = "gdelt_gkg_topic_features_1h"
WAR_EVENT_TOPICS = {"threat", "force_posture", "coercion", "assault", "fight", "mass_violence"}


@dataclass
class GoldBuildSummary:
    db_path: str
    output_dir: str
    start: str
    end: str
    dry_run: bool
    hours_planned: int
    rows_written: int = 0
    csv_path: str | None = None
    json_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build first-pass GDELT/GKG Gold hourly topic features from Silver records."
    )
    parser.add_argument("--app-dir", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = default_paths(args.app_dir)
    db_path = args.db or paths.gdelt_db
    start = parse_utc(args.start)
    end = parse_utc(args.end)
    if end <= start:
        raise SystemExit("--end must be after --start")

    tag = safe_tag(args.tag) if args.tag else f"{stamp_tag(start)}_{stamp_tag(end)}"
    summary = build_gold_features(
        db_path=db_path,
        output_dir=args.output_dir,
        start=start,
        end=end,
        tag=tag,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


def build_gold_features(
    *,
    db_path: Path,
    output_dir: Path,
    start: datetime,
    end: datetime,
    tag: str,
    dry_run: bool = False,
) -> GoldBuildSummary:
    hours = list(iter_hours(start, end))
    summary = GoldBuildSummary(
        db_path=str(db_path),
        output_dir=str(output_dir),
        start=start.isoformat(),
        end=end.isoformat(),
        dry_run=dry_run,
        hours_planned=len(hours),
    )
    if not db_path.exists():
        return summary

    with schema.connect_db(db_path) as conn:
        ensure_gold_schema(conn)
        rows = [feature_row_for_hour(conn, hour) for hour in hours]
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"gdelt_gkg_gold_topic_features_{tag}.csv"
        json_path = output_dir / f"gdelt_gkg_gold_topic_features_{tag}.json"
        write_rows(csv_path, json_path, rows)
        summary.csv_path = str(csv_path)
        summary.json_path = str(json_path)
        if not dry_run:
            conn.execute("BEGIN IMMEDIATE")
            try:
                upsert_gold_rows(conn, rows)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            summary.rows_written = len(rows)
    return summary


def ensure_gold_schema(conn: sqlite3.Connection) -> None:
    schema.init_schema(conn)
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS {GOLD_TOPIC_TABLE} (
            feature_hour TEXT PRIMARY KEY,
            generated_at TEXT NOT NULL,
            event_count INTEGER NOT NULL DEFAULT 0,
            document_count INTEGER NOT NULL DEFAULT 0,
            source_count INTEGER NOT NULL DEFAULT 0,
            topic_source_diversity_24h INTEGER NOT NULL DEFAULT 0,
            war_geopolitics_intensity_24h REAL NOT NULL DEFAULT 0,
            banking_credit_confluence_24h INTEGER NOT NULL DEFAULT 0,
            oil_energy_first_mention INTEGER NOT NULL DEFAULT 0,
            regulation_persistence_72h REAL NOT NULL DEFAULT 0,
            macro_release_severity_max_24h REAL NOT NULL DEFAULT 0,
            conflict_event_count_24h INTEGER NOT NULL DEFAULT 0,
            silver_event_rows_24h INTEGER NOT NULL DEFAULT 0,
            silver_document_rows_24h INTEGER NOT NULL DEFAULT 0,
            missing_available_at_rows INTEGER NOT NULL DEFAULT 0,
            coverage_json TEXT NOT NULL DEFAULT '{{}}',
            metadata_json TEXT NOT NULL DEFAULT '{{}}'
        );

        CREATE INDEX IF NOT EXISTS idx_gdelt_gkg_topic_features_generated_at
            ON {GOLD_TOPIC_TABLE}(generated_at);
        """
    )


def feature_row_for_hour(conn: sqlite3.Connection, hour: datetime) -> dict[str, Any]:
    hour_start = hour.isoformat()
    hour_end = (hour + timedelta(hours=1)).isoformat()
    day_start = (hour - timedelta(hours=24)).isoformat()
    three_day_start = (hour - timedelta(hours=72)).isoformat()

    event_hour = _one(
        conn,
        """
        SELECT COUNT(*) AS rows,
               SUM(CASE WHEN topic IN ({war_topics}) OR impact_channel LIKE '%conflict%' THEN 1 ELSE 0 END) AS conflict_rows
        FROM gdelt_events_silver
        WHERE available_at >= ? AND available_at < ?
        """.format(war_topics=",".join("?" for _ in WAR_EVENT_TOPICS)),
        [*sorted(WAR_EVENT_TOPICS), hour_start, hour_end],
    )
    doc_hour = _one(
        conn,
        """
        SELECT COUNT(*) AS rows,
               COUNT(DISTINCT COALESCE(source_common_name, url_hash, document_id)) AS sources
        FROM gkg_documents_silver
        WHERE available_at >= ? AND available_at < ?
        """,
        [hour_start, hour_end],
    )
    rolling = _one(
        conn,
        """
        SELECT
            (SELECT COUNT(*) FROM gdelt_events_silver WHERE available_at >= ? AND available_at < ?) AS event_rows_24h,
            (SELECT COUNT(*) FROM gkg_documents_silver WHERE available_at >= ? AND available_at < ?) AS doc_rows_24h,
            (SELECT COUNT(DISTINCT source_common_name) FROM gkg_documents_silver WHERE available_at >= ? AND available_at < ? AND source_common_name IS NOT NULL) AS source_diversity_24h,
            (SELECT COUNT(DISTINCT source_common_name) FROM gkg_documents_silver WHERE available_at >= ? AND available_at < ? AND topic = 'banking_credit' AND source_common_name IS NOT NULL) AS banking_sources_24h,
            (SELECT MAX(COALESCE(severity_proxy, 0)) FROM gdelt_events_silver WHERE available_at >= ? AND available_at < ? AND topic IN ('public_statement', 'demand', 'threat', 'coercion')) AS macro_event_severity_24h,
            (SELECT COUNT(*) FROM gdelt_events_silver WHERE available_at >= ? AND available_at < ? AND (topic IN ('threat','force_posture','coercion','assault','fight','mass_violence') OR impact_channel LIKE '%conflict%')) AS conflict_events_24h
        """,
        [
            day_start,
            hour_end,
            day_start,
            hour_end,
            day_start,
            hour_end,
            day_start,
            hour_end,
            day_start,
            hour_end,
            day_start,
            hour_end,
        ],
    )
    geopolitics_docs = _scalar(
        conn,
        "SELECT COUNT(*) FROM gkg_documents_silver WHERE available_at >= ? AND available_at < ? AND topic = 'geopolitics'",
        [day_start, hour_end],
    )
    oil_now = _scalar(
        conn,
        "SELECT COUNT(*) FROM gkg_documents_silver WHERE available_at >= ? AND available_at < ? AND topic = 'oil_energy'",
        [hour_start, hour_end],
    )
    oil_prev = _scalar(
        conn,
        "SELECT COUNT(*) FROM gkg_documents_silver WHERE available_at >= ? AND available_at < ? AND topic = 'oil_energy'",
        [three_day_start, hour_start],
    )
    regulation_hours = _scalar(
        conn,
        """
        SELECT COUNT(DISTINCT substr(available_at, 1, 13))
        FROM gkg_documents_silver
        WHERE available_at >= ? AND available_at < ? AND topic = 'regulation'
        """,
        [three_day_start, hour_end],
    )
    missing_available = missing_available_rows(conn)
    coverage = {
        "window": "available_at aligned",
        "silver_event_rows_24h": int(rolling["event_rows_24h"] or 0),
        "silver_document_rows_24h": int(rolling["doc_rows_24h"] or 0),
    }
    metadata = {
        "feature_version": "gdelt_gkg_gold_features_v1",
        "topic_quality": "weak_silver_labels_pending_classifier_enrichment",
        "counts_are_supporting_context": True,
    }
    return {
        "feature_hour": hour_start,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "event_count": int(event_hour["rows"] or 0),
        "document_count": int(doc_hour["rows"] or 0),
        "source_count": int(doc_hour["sources"] or 0),
        "topic_source_diversity_24h": int(rolling["source_diversity_24h"] or 0),
        "war_geopolitics_intensity_24h": float((rolling["conflict_events_24h"] or 0) + geopolitics_docs),
        "banking_credit_confluence_24h": int(rolling["banking_sources_24h"] or 0),
        "oil_energy_first_mention": 1 if oil_now > 0 and oil_prev == 0 else 0,
        "regulation_persistence_72h": float(regulation_hours) / 72.0,
        "macro_release_severity_max_24h": float(rolling["macro_event_severity_24h"] or 0.0),
        "conflict_event_count_24h": int(rolling["conflict_events_24h"] or 0),
        "silver_event_rows_24h": int(rolling["event_rows_24h"] or 0),
        "silver_document_rows_24h": int(rolling["doc_rows_24h"] or 0),
        "missing_available_at_rows": missing_available,
        "coverage_json": json.dumps(coverage, sort_keys=True, separators=(",", ":")),
        "metadata_json": json.dumps(metadata, sort_keys=True, separators=(",", ":")),
    }


def upsert_gold_rows(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    placeholders = ", ".join("?" for _ in columns)
    updates = ", ".join(f"{column}=excluded.{column}" for column in columns if column != "feature_hour")
    conn.executemany(
        f"""
        INSERT INTO {GOLD_TOPIC_TABLE}({', '.join(columns)})
        VALUES ({placeholders})
        ON CONFLICT(feature_hour) DO UPDATE SET {updates}
        """,
        [tuple(row[column] for column in columns) for row in rows],
    )


def missing_available_rows(conn: sqlite3.Connection) -> int:
    total = 0
    for table in ("gdelt_events_silver", "gkg_documents_silver"):
        total += _scalar(
            conn,
            f"SELECT COUNT(*) FROM {table} WHERE available_at IS NULL OR TRIM(CAST(available_at AS TEXT)) = ''",
            [],
        )
    return int(total)


def write_rows(csv_path: Path, json_path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")


def _one(conn: sqlite3.Connection, sql: str, params: list[Any]) -> sqlite3.Row:
    row = conn.execute(sql, params).fetchone()
    if row is None:
        raise RuntimeError("expected one SQLite row")
    return row


def _scalar(conn: sqlite3.Connection, sql: str, params: list[Any]) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0] or 0) if row is not None else 0


def iter_hours(start: datetime, end: datetime) -> Iterable[datetime]:
    current = start.replace(minute=0, second=0, microsecond=0)
    while current < end:
        yield current
        current += timedelta(hours=1)


def parse_utc(value: str) -> datetime:
    clean = value.strip().replace("Z", "+00:00")
    if len(clean) == 10:
        clean = f"{clean}T00:00:00+00:00"
    parsed = datetime.fromisoformat(clean)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(second=0, microsecond=0)


def stamp_tag(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")


def safe_tag(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("._-") or "gold"


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.builder import default_paths  # noqa: E402


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"
DEFAULT_GDELT_RAW_DIR = Path("D:/FreqTradeStuffLargeData/research_news_data/gdelt/raw/export")
DEFAULT_GKG_RAW_DIR = Path("D:/FreqTradeStuffLargeData/research_news_data/gdelt/raw/gkg")


@dataclass(frozen=True)
class SourceSpec:
    source: str
    table: str
    interval_minutes: int
    metric_column: str
    raw_dir_arg: str
    raw_suffix: str


SOURCE_SPECS = (
    SourceSpec(
        source="gdelt_events",
        table="gdelt_hourly_features",
        interval_minutes=60,
        metric_column="event_count",
        raw_dir_arg="gdelt_raw_dir",
        raw_suffix=".export.CSV.zip",
    ),
    SourceSpec(
        source="gkg_documents",
        table="gdelt_gkg_file_features",
        interval_minutes=15,
        metric_column="document_count",
        raw_dir_arg="gkg_raw_dir",
        raw_suffix=".gkg.csv.zip",
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write read-only GDELT/GKG historical quality and coverage reports."
    )
    parser.add_argument("--app-dir", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=None, help="Path to gdelt_context.sqlite.")
    parser.add_argument("--feature-db", type=Path, default=None, help="Optional context_features.sqlite for silver completeness.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gdelt-raw-dir", type=Path, default=DEFAULT_GDELT_RAW_DIR)
    parser.add_argument("--gkg-raw-dir", type=Path, default=DEFAULT_GKG_RAW_DIR)
    parser.add_argument("--start", required=True, help="UTC start date/time, inclusive.")
    parser.add_argument("--end", required=True, help="UTC end date/time, exclusive.")
    parser.add_argument("--tag", default="", help="Filename tag. Defaults to current UTC timestamp.")
    parser.add_argument("--limit-files", type=int, default=None, help="Limit raw ZIP validation files per source.")
    args = parser.parse_args()

    if args.limit_files is not None and args.limit_files < 0:
        raise SystemExit("--limit-files must be zero or positive")

    paths = default_paths(args.app_dir)
    db_path = args.db or paths.gdelt_db
    feature_db = args.feature_db or paths.feature_db
    start = parse_utc(args.start, floor_minutes=15)
    end = parse_utc(args.end, floor_minutes=15)
    if end <= start:
        raise SystemExit("--end must be after --start")

    tag = safe_tag(args.tag) if args.tag else datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    manifest = write_quality_reports(
        db_path=db_path,
        feature_db=feature_db,
        output_dir=args.output_dir,
        start=start,
        end=end,
        tag=tag,
        gdelt_raw_dir=args.gdelt_raw_dir,
        gkg_raw_dir=args.gkg_raw_dir,
        limit_files=args.limit_files,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def write_quality_reports(
    *,
    db_path: Path,
    feature_db: Path,
    output_dir: Path,
    start: datetime,
    end: datetime,
    tag: str,
    gdelt_raw_dir: Path | None = None,
    gkg_raw_dir: Path | None = None,
    limit_files: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dirs = {
        "gdelt_raw_dir": gdelt_raw_dir,
        "gkg_raw_dir": gkg_raw_dir,
    }

    coverage_rows: list[dict[str, Any]]
    error_rows: list[dict[str, Any]]
    hole_rows: list[dict[str, Any]]
    if db_path.exists():
        with sqlite3.connect(str(db_path), timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            coverage_rows = build_sqlite_coverage_summary(conn, start, end)
            error_rows = build_error_breakdown(conn, start, end)
            hole_rows = build_source_hole_report(conn, start, end)
    else:
        coverage_rows = [
            _missing_db_row(spec, db_path, start, end)
            for spec in SOURCE_SPECS
        ]
        error_rows = []
        hole_rows = [
            {
                "source": spec.source,
                "status": "database_missing",
                "start": start.isoformat(),
                "end": (end - timedelta(minutes=spec.interval_minutes)).isoformat(),
                "expected_files": len(list(iter_expected_times(start, end, spec.interval_minutes))),
                "note": f"Database not found: {db_path}",
            }
            for spec in SOURCE_SPECS
        ]

    raw_rows = build_raw_zip_validation_summary(start, end, raw_dirs, limit_files)
    normalized_rows = build_normalized_completeness_summary(feature_db, db_path, start, end)

    reports = {
        "sqlite_coverage": _write_csv_json(output_dir, f"gdelt_gkg_sqlite_coverage_{tag}", coverage_rows),
        "error_breakdown": _write_csv_json(output_dir, f"gdelt_gkg_error_breakdown_{tag}", error_rows),
        "source_holes": _write_csv_json(output_dir, f"gdelt_gkg_source_holes_{tag}", hole_rows),
        "raw_zip_validation": _write_csv_json(output_dir, f"gdelt_gkg_raw_zip_validation_{tag}", raw_rows),
        "normalized_completeness": _write_csv_json(output_dir, f"gdelt_gkg_normalized_completeness_{tag}", normalized_rows),
    }
    manifest = {
        "db_path": str(db_path),
        "feature_db": str(feature_db),
        "output_dir": str(output_dir),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "tag": tag,
        "limit_files": limit_files,
        "reports": reports,
    }
    manifest_path = output_dir / f"gdelt_gkg_quality_manifest_{tag}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def build_sqlite_coverage_summary(conn: sqlite3.Connection, start: datetime, end: datetime) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        expected = len(list(iter_expected_times(start, end, spec.interval_minutes)))
        if not table_exists(conn, spec.table):
            rows.append(
                {
                    "source": spec.source,
                    "table": spec.table,
                    "status": "table_missing",
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "expected_files": expected,
                    "sqlite_rows": 0,
                    "successful_rows": 0,
                    "failed_rows": 0,
                    "missing_rows": expected,
                    "present_ratio": 0.0,
                    "successful_ratio": 0.0,
                    "metric_sum": 0.0,
                    "first_row_date": "",
                    "last_row_date": "",
                    "first_success_date": "",
                    "last_success_date": "",
                }
            )
            continue
        metric = spec.metric_column if column_exists(conn, spec.table, spec.metric_column) else "0"
        row = conn.execute(
            f"""
            SELECT COUNT(*) AS sqlite_rows,
                   SUM(CASE WHEN parse_error IS NULL THEN 1 ELSE 0 END) AS successful_rows,
                   SUM(CASE WHEN parse_error IS NOT NULL THEN 1 ELSE 0 END) AS failed_rows,
                   SUM(CASE WHEN parse_error IS NULL THEN COALESCE({metric}, 0) ELSE 0 END) AS metric_sum,
                   MIN(date) AS first_row_date,
                   MAX(date) AS last_row_date,
                   MIN(CASE WHEN parse_error IS NULL THEN date END) AS first_success_date,
                   MAX(CASE WHEN parse_error IS NULL THEN date END) AS last_success_date
            FROM {spec.table}
            WHERE date >= ? AND date < ?
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchone()
        sqlite_rows = int(row["sqlite_rows"] or 0)
        successful_rows = int(row["successful_rows"] or 0)
        failed_rows = int(row["failed_rows"] or 0)
        rows.append(
            {
                "source": spec.source,
                "table": spec.table,
                "status": "ok",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "expected_files": expected,
                "sqlite_rows": sqlite_rows,
                "successful_rows": successful_rows,
                "failed_rows": failed_rows,
                "missing_rows": max(0, expected - sqlite_rows),
                "present_ratio": ratio(sqlite_rows, expected),
                "successful_ratio": ratio(successful_rows, expected),
                "metric_sum": float(row["metric_sum"] or 0.0),
                "first_row_date": row["first_row_date"] or "",
                "last_row_date": row["last_row_date"] or "",
                "first_success_date": row["first_success_date"] or "",
                "last_success_date": row["last_success_date"] or "",
            }
        )
    return rows


def build_error_breakdown(conn: sqlite3.Connection, start: datetime, end: datetime) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        if not table_exists(conn, spec.table):
            continue
        records = conn.execute(
            f"""
            SELECT date, parse_error
            FROM {spec.table}
            WHERE date >= ? AND date < ?
              AND parse_error IS NOT NULL
            ORDER BY date
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for record in records:
            message = str(record["parse_error"] or "")
            category = classify_parse_error(message)
            key = (category, message[:160])
            item = grouped.setdefault(
                key,
                {
                    "source": spec.source,
                    "table": spec.table,
                    "error_category": category,
                    "parse_error_sample": message[:160],
                    "rows": 0,
                    "first_date": str(record["date"]),
                    "last_date": str(record["date"]),
                },
            )
            item["rows"] += 1
            item["last_date"] = str(record["date"])
        rows.extend(sorted(grouped.values(), key=lambda item: (item["source"], item["error_category"], item["first_date"])))
    return rows


def build_source_hole_report(conn: sqlite3.Connection, start: datetime, end: datetime) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        expected_times = list(iter_expected_times(start, end, spec.interval_minutes))
        if not table_exists(conn, spec.table):
            rows.append(_hole_row(spec, "table_missing", expected_times, "SQLite table is absent."))
            continue
        stored = {
            str(row["date"]): "sqlite_parse_error" if row["parse_error"] else "ok"
            for row in conn.execute(
                f"SELECT date, parse_error FROM {spec.table} WHERE date >= ? AND date < ?",
                (start.isoformat(), end.isoformat()),
            ).fetchall()
        }
        hole_times: list[tuple[datetime, str]] = []
        for stamp in expected_times:
            status = stored.get(stamp.isoformat(), "missing_sqlite_row")
            if status != "ok":
                hole_times.append((stamp, status))
        rows.extend(_segment_holes(spec, hole_times))
    return rows


def build_raw_zip_validation_summary(
    start: datetime,
    end: datetime,
    raw_dirs: dict[str, Path | None],
    limit_files: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        raw_dir = raw_dirs.get(spec.raw_dir_arg)
        stamps = list(iter_expected_times(start, end, spec.interval_minutes))
        if limit_files is not None:
            stamps = stamps[:limit_files]
        grouped: dict[str, dict[str, Any]] = {}
        if raw_dir is None:
            rows.append(_raw_group_row(spec, "", "raw_dir_not_configured", len(stamps), ""))
            continue
        for stamp in stamps:
            path = raw_dir / f"{stamp.strftime('%Y%m%d%H%M%S')}{spec.raw_suffix}"
            status, detail, file_size = validate_zip_file(path)
            item = grouped.setdefault(
                status,
                _raw_group_row(spec, str(raw_dir), status, 0, detail),
            )
            item["files"] += 1
            item["bytes_total"] += file_size
            if detail and not item["detail_sample"]:
                item["detail_sample"] = detail
        rows.extend(sorted(grouped.values(), key=lambda item: (item["source"], item["status"])))
    return rows


def build_normalized_completeness_summary(feature_db: Path, source_db: Path, start: datetime, end: datetime) -> list[dict[str, Any]]:
    rows = build_context_feature_completeness_summary(feature_db, start, end)
    rows.extend(build_historical_silver_completeness_summary(source_db, start, end))
    return rows


def build_context_feature_completeness_summary(feature_db: Path, start: datetime, end: datetime) -> list[dict[str, Any]]:
    if not feature_db.exists():
        return [
            {
                "database": str(feature_db),
                "table": "context_features_1h",
                "source": "silver_context_features",
                "status": "database_missing",
                "rows": 0,
                "columns_checked": 0,
                "non_null_cells": 0,
                "nonzero_cells": 0,
                "row_non_null_ratio": 0.0,
                "cell_non_null_ratio": 0.0,
                "first_date": "",
                "last_date": "",
                "future_available_at_rows": 0,
            }
        ]
    with sqlite3.connect(str(feature_db), timeout=10.0) as conn:
        conn.row_factory = sqlite3.Row
        if not table_exists(conn, "context_features_1h"):
            return [
                {
                    "database": str(feature_db),
                    "table": "context_features_1h",
                    "source": "silver_context_features",
                    "status": "table_missing",
                    "rows": 0,
                    "columns_checked": 0,
                    "non_null_cells": 0,
                    "nonzero_cells": 0,
                    "row_non_null_ratio": 0.0,
                    "cell_non_null_ratio": 0.0,
                    "first_date": "",
                    "last_date": "",
                    "future_available_at_rows": 0,
                }
            ]
        columns = table_columns(conn, "context_features_1h")
        rows: list[dict[str, Any]] = []
        for source, prefixes in {
            "gdelt_events": ("gdelt_",),
            "gkg_documents": ("gkg_",),
        }.items():
            feature_columns = [column for column in columns if column.startswith(prefixes)]
            rows.append(_feature_completeness_row(conn, feature_db, source, feature_columns, start, end))
        if table_exists(conn, "normalised_context_events"):
            rows.extend(_normalised_event_rows(conn, feature_db, start, end))
        return rows


def build_historical_silver_completeness_summary(source_db: Path, start: datetime, end: datetime) -> list[dict[str, Any]]:
    specs = (
        {
            "source": "gdelt_events_silver",
            "table": "gdelt_events_silver",
            "available_column": "available_at",
            "event_time_candidates": ("event_time", "event_date"),
            "source_candidates": ("source_url_hash", "source_hostname"),
            "raw_file_candidates": ("raw_file_id", "file_stamp"),
            "topic_candidates": ("topic", "cameo_topic"),
            "url_hash_candidates": ("source_url_hash",),
        },
        {
            "source": "gkg_documents_silver",
            "table": "gkg_documents_silver",
            "available_column": "available_at",
            "event_time_candidates": ("document_time", "gkg_date", "published_at"),
            "source_candidates": ("url_hash", "source_common_name", "source_name"),
            "raw_file_candidates": ("raw_file_id",),
            "topic_candidates": ("topic", "weak_topics_json"),
            "url_hash_candidates": ("url_hash",),
        },
    )
    if not source_db.exists():
        return [
            {
                "database": str(source_db),
                "table": str(spec["table"]),
                "source": str(spec["source"]),
                "status": "database_missing",
                "rows": 0,
                "first_date": "",
                "last_date": "",
                "missing_available_at_rows": 0,
                "distinct_sources": 0,
                "distinct_raw_files": 0,
                "missing_topic_rows": 0,
                "missing_url_hash_rows": 0,
                "available_before_event_time_rows": 0,
            }
            for spec in specs
        ]
    with sqlite3.connect(str(source_db), timeout=10.0) as conn:
        conn.row_factory = sqlite3.Row
        rows: list[dict[str, Any]] = []
        for spec in specs:
            table = str(spec["table"])
            if not table_exists(conn, table):
                rows.append(
                    {
                        "database": str(source_db),
                        "table": table,
                        "source": str(spec["source"]),
                        "status": "table_missing",
                        "rows": 0,
                        "first_date": "",
                        "last_date": "",
                        "missing_available_at_rows": 0,
                        "distinct_sources": 0,
                        "distinct_raw_files": 0,
                        "missing_topic_rows": 0,
                        "missing_url_hash_rows": 0,
                        "available_before_event_time_rows": 0,
                    }
                )
                continue
            columns = table_columns(conn, table)
            available_column = str(spec["available_column"])
            if available_column not in columns:
                rows.append(
                    {
                        "database": str(source_db),
                        "table": table,
                        "source": str(spec["source"]),
                        "status": "missing_available_at_column",
                        "rows": 0,
                        "first_date": "",
                        "last_date": "",
                        "missing_available_at_rows": 0,
                        "distinct_sources": 0,
                        "distinct_raw_files": 0,
                        "missing_topic_rows": 0,
                        "missing_url_hash_rows": 0,
                        "available_before_event_time_rows": 0,
                    }
                )
                continue
            rows.append(_historical_silver_table_row(conn, source_db, spec, columns, start, end))
        return rows


def _feature_completeness_row(
    conn: sqlite3.Connection,
    feature_db: Path,
    source: str,
    feature_columns: list[str],
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    base = {
        "database": str(feature_db),
        "table": "context_features_1h",
        "source": source,
        "status": "ok" if feature_columns else "no_matching_columns",
        "columns_checked": len(feature_columns),
    }
    row = conn.execute(
        """
        SELECT COUNT(*) AS rows,
               MIN(date) AS first_date,
               MAX(date) AS last_date,
               SUM(CASE WHEN max_source_available_at > date THEN 1 ELSE 0 END) AS future_available_at_rows
        FROM context_features_1h
        WHERE date >= ? AND date < ?
        """,
        (start.isoformat(), end.isoformat()),
    ).fetchone()
    total_rows = int(row["rows"] or 0)
    if not feature_columns or total_rows == 0:
        return {
            **base,
            "rows": total_rows,
            "non_null_cells": 0,
            "nonzero_cells": 0,
            "row_non_null_ratio": 0.0,
            "cell_non_null_ratio": 0.0,
            "first_date": row["first_date"] or "",
            "last_date": row["last_date"] or "",
            "future_available_at_rows": int(row["future_available_at_rows"] or 0),
        }
    non_null_expr = " + ".join(f"CASE WHEN {column} IS NOT NULL THEN 1 ELSE 0 END" for column in feature_columns)
    nonzero_expr = " + ".join(f"CASE WHEN COALESCE({column}, 0) != 0 THEN 1 ELSE 0 END" for column in feature_columns)
    stats = conn.execute(
        f"""
        SELECT SUM({non_null_expr}) AS non_null_cells,
               SUM({nonzero_expr}) AS nonzero_cells,
               SUM(CASE WHEN ({non_null_expr}) > 0 THEN 1 ELSE 0 END) AS rows_with_non_null
        FROM context_features_1h
        WHERE date >= ? AND date < ?
        """,
        (start.isoformat(), end.isoformat()),
    ).fetchone()
    non_null_cells = int(stats["non_null_cells"] or 0)
    return {
        **base,
        "rows": total_rows,
        "non_null_cells": non_null_cells,
        "nonzero_cells": int(stats["nonzero_cells"] or 0),
        "row_non_null_ratio": ratio(int(stats["rows_with_non_null"] or 0), total_rows),
        "cell_non_null_ratio": ratio(non_null_cells, total_rows * len(feature_columns)),
        "first_date": row["first_date"] or "",
        "last_date": row["last_date"] or "",
        "future_available_at_rows": int(row["future_available_at_rows"] or 0),
    }


def _normalised_event_rows(conn: sqlite3.Connection, feature_db: Path, start: datetime, end: datetime) -> list[dict[str, Any]]:
    if not {"source_family", "available_at"}.issubset(set(table_columns(conn, "normalised_context_events"))):
        return []
    records = conn.execute(
        """
        SELECT source_family,
               COUNT(*) AS rows,
               MIN(available_at) AS first_date,
               MAX(available_at) AS last_date
        FROM normalised_context_events
        WHERE available_at >= ? AND available_at < ?
        GROUP BY source_family
        ORDER BY source_family
        """,
        (start.isoformat(), end.isoformat()),
    ).fetchall()
    return [
        {
            "database": str(feature_db),
            "table": "normalised_context_events",
            "source": f"normalised_{row['source_family']}",
            "status": "ok",
            "rows": int(row["rows"] or 0),
            "columns_checked": 0,
            "non_null_cells": 0,
            "nonzero_cells": 0,
            "row_non_null_ratio": 0.0,
            "cell_non_null_ratio": 0.0,
            "first_date": row["first_date"] or "",
            "last_date": row["last_date"] or "",
            "future_available_at_rows": 0,
        }
        for row in records
    ]


def _historical_silver_table_row(
    conn: sqlite3.Connection,
    source_db: Path,
    spec: dict[str, Any],
    columns: list[str],
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    table = str(spec["table"])
    available_column = str(spec["available_column"])
    source_column = _first_existing(columns, spec["source_candidates"])
    raw_file_column = _first_existing(columns, spec["raw_file_candidates"])
    topic_column = _first_existing(columns, spec["topic_candidates"])
    url_hash_column = _first_existing(columns, spec["url_hash_candidates"])
    event_time_column = _first_existing(columns, spec["event_time_candidates"])

    distinct_sources_expr = f"COUNT(DISTINCT {source_column})" if source_column else "0"
    distinct_raw_files_expr = f"COUNT(DISTINCT {raw_file_column})" if raw_file_column else "0"
    missing_topic_expr = (
        f"SUM(CASE WHEN {topic_column} IS NULL OR TRIM(CAST({topic_column} AS TEXT)) = '' THEN 1 ELSE 0 END)"
        if topic_column
        else "0"
    )
    missing_url_hash_expr = (
        f"SUM(CASE WHEN {url_hash_column} IS NULL OR TRIM(CAST({url_hash_column} AS TEXT)) = '' THEN 1 ELSE 0 END)"
        if url_hash_column
        else "0"
    )
    available_before_event_expr = (
        f"SUM(CASE WHEN {available_column} < {event_time_column} THEN 1 ELSE 0 END)"
        if event_time_column
        else "0"
    )
    stats = conn.execute(
        f"""
        SELECT COUNT(*) AS rows,
               MIN({available_column}) AS first_date,
               MAX({available_column}) AS last_date,
               {distinct_sources_expr} AS distinct_sources,
               {distinct_raw_files_expr} AS distinct_raw_files,
               {missing_topic_expr} AS missing_topic_rows,
               {missing_url_hash_expr} AS missing_url_hash_rows,
               {available_before_event_expr} AS available_before_event_time_rows
        FROM {table}
        WHERE {available_column} >= ? AND {available_column} < ?
        """,
        (start.isoformat(), end.isoformat()),
    ).fetchone()
    missing_available = conn.execute(
        f"""
        SELECT COUNT(*) AS rows
        FROM {table}
        WHERE {available_column} IS NULL OR TRIM(CAST({available_column} AS TEXT)) = ''
        """
    ).fetchone()
    return {
        "database": str(source_db),
        "table": table,
        "source": str(spec["source"]),
        "status": "ok",
        "rows": int(stats["rows"] or 0),
        "first_date": stats["first_date"] or "",
        "last_date": stats["last_date"] or "",
        "missing_available_at_rows": int(missing_available["rows"] or 0),
        "distinct_sources": int(stats["distinct_sources"] or 0),
        "distinct_raw_files": int(stats["distinct_raw_files"] or 0),
        "missing_topic_rows": int(stats["missing_topic_rows"] or 0),
        "missing_url_hash_rows": int(stats["missing_url_hash_rows"] or 0),
        "available_before_event_time_rows": int(stats["available_before_event_time_rows"] or 0),
        "time_column": available_column,
        "event_time_column": event_time_column or "",
        "source_column": source_column or "",
        "raw_file_column": raw_file_column or "",
        "topic_column": topic_column or "",
        "url_hash_column": url_hash_column or "",
    }


def classify_parse_error(message: str) -> str:
    lower = message.lower()
    if re.match(r"^http\s+404\b", lower):
        return "http_404_terminal_missing"
    if re.match(r"^http\s+\d+", lower):
        return "http_error"
    if any(token in lower for token in ("timed out", "timeout", "connection", "refused", "reset", "dns", "ssl")):
        return "network_error"
    if any(token in lower for token in ("not a zip", "bad zip", "empty zip", "file is not a zip", "zipfile")):
        return "zip_error"
    if any(token in lower for token in ("field larger than", "csv", "decode", "delimiter", "parse")):
        return "parse_error"
    if not message.strip():
        return "empty_error_message"
    return "other_error"


def validate_zip_file(path: Path) -> tuple[str, str, int]:
    if not path.exists():
        return "missing_file", str(path), 0
    file_size = path.stat().st_size
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if not names:
                return "empty_zip", str(path), file_size
            bad_member = archive.testzip()
            if bad_member:
                return "corrupt_member", bad_member, file_size
    except zipfile.BadZipFile as exc:
        return "bad_zip", str(exc), file_size
    except OSError as exc:
        return "read_error", str(exc), file_size
    return "ok", "", file_size


def parse_utc(value: str, *, floor_minutes: int) -> datetime:
    clean = value.strip().replace("Z", "+00:00")
    if len(clean) == 10:
        clean = f"{clean}T00:00:00+00:00"
    parsed = datetime.fromisoformat(clean)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc).replace(second=0, microsecond=0)
    minute = (parsed.minute // floor_minutes) * floor_minutes
    return parsed.replace(minute=minute)


def iter_expected_times(start: datetime, end: datetime, interval_minutes: int) -> Iterable[datetime]:
    current = start
    step = timedelta(minutes=interval_minutes)
    while current < end:
        yield current
        current += step


def safe_tag(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._-") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone() is not None


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return column in table_columns(conn, table)


def _first_existing(columns: list[str], candidates: Iterable[str]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _segment_holes(spec: SourceSpec, holes: list[tuple[datetime, str]]) -> list[dict[str, Any]]:
    if not holes:
        return []
    rows: list[dict[str, Any]] = []
    start_stamp, current_status = holes[0]
    previous_stamp = start_stamp
    count = 1
    step = timedelta(minutes=spec.interval_minutes)
    for stamp, status in holes[1:]:
        if status == current_status and stamp == previous_stamp + step:
            previous_stamp = stamp
            count += 1
            continue
        rows.append(
            {
                "source": spec.source,
                "status": current_status,
                "start": start_stamp.isoformat(),
                "end": previous_stamp.isoformat(),
                "expected_files": count,
                "note": "",
            }
        )
        start_stamp = previous_stamp = stamp
        current_status = status
        count = 1
    rows.append(
        {
            "source": spec.source,
            "status": current_status,
            "start": start_stamp.isoformat(),
            "end": previous_stamp.isoformat(),
            "expected_files": count,
            "note": "",
        }
    )
    return rows


def _hole_row(spec: SourceSpec, status: str, expected_times: list[datetime], note: str) -> dict[str, Any]:
    return {
        "source": spec.source,
        "status": status,
        "start": expected_times[0].isoformat() if expected_times else "",
        "end": expected_times[-1].isoformat() if expected_times else "",
        "expected_files": len(expected_times),
        "note": note,
    }


def _raw_group_row(spec: SourceSpec, raw_dir: str, status: str, files: int, detail: str) -> dict[str, Any]:
    return {
        "source": spec.source,
        "raw_dir": raw_dir,
        "status": status,
        "files": files,
        "bytes_total": 0,
        "detail_sample": detail,
    }


def _missing_db_row(spec: SourceSpec, db_path: Path, start: datetime, end: datetime) -> dict[str, Any]:
    expected = len(list(iter_expected_times(start, end, spec.interval_minutes)))
    return {
        "source": spec.source,
        "table": spec.table,
        "status": "database_missing",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "expected_files": expected,
        "sqlite_rows": 0,
        "successful_rows": 0,
        "failed_rows": 0,
        "missing_rows": expected,
        "present_ratio": 0.0,
        "successful_ratio": 0.0,
        "metric_sum": 0.0,
        "first_row_date": "",
        "last_row_date": "",
        "first_success_date": "",
        "last_success_date": "",
        "note": f"Database not found: {db_path}",
    }


def _write_csv_json(output_dir: Path, stem: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path), "rows": len(rows)}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    if not fieldnames:
        fieldnames = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())

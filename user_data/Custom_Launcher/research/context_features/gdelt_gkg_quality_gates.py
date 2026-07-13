from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (  # noqa: E402
    gdelt_gkg_quality_reports as reports,
)
from user_data.Custom_Launcher.research.context_features.builder import default_paths  # noqa: E402


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"
GATE_FIELDNAMES = [
    "gate_id",
    "category",
    "source",
    "database",
    "table",
    "status",
    "reason",
    "observed",
    "expected",
    "ratio",
    "start",
    "end",
]

RAW_KIND_BY_SOURCE = {
    "gdelt_events": "export",
    "gkg_documents": "gkg",
}

SQLITE_PARAM_BATCH_SIZE = 500

SILVER_SPECS = (
    {
        "source": "gdelt_events_silver",
        "table": "gdelt_events_silver",
        "time_column": "event_time",
        "available_column": "available_at",
        "required_columns": (
            "event_id",
            "raw_file_id",
            "event_time",
            "available_at",
            "source_url_hash",
            "topic",
        ),
    },
    {
        "source": "gkg_documents_silver",
        "table": "gkg_documents_silver",
        "time_column": "document_time",
        "available_column": "available_at",
        "required_columns": (
            "document_id",
            "raw_file_id",
            "document_time",
            "available_at",
            "url_hash",
            "topic",
        ),
    },
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate read-only GDELT/GKG historical quality gates."
    )
    parser.add_argument("--app-dir", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=None, help="Path to gdelt_context.sqlite.")
    parser.add_argument(
        "--feature-db",
        type=Path,
        default=None,
        help="Optional context_features.sqlite.",
    )
    parser.add_argument("--start", required=True, help="UTC start date/time, inclusive.")
    parser.add_argument("--end", required=True, help="UTC end date/time, exclusive.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--tag",
        default="",
        help="Filename tag. Defaults to a deterministic window tag.",
    )
    args = parser.parse_args()

    paths = default_paths(args.app_dir)
    db_path = args.db or paths.gdelt_db
    feature_db = args.feature_db or paths.feature_db
    start = reports.parse_utc(args.start, floor_minutes=15)
    end = reports.parse_utc(args.end, floor_minutes=15)
    if end <= start:
        raise SystemExit("--end must be after --start")

    tag = reports.safe_tag(args.tag) if args.tag else deterministic_tag(start, end)
    manifest = write_quality_gates(
        db_path=db_path,
        feature_db=feature_db,
        output_dir=args.output_dir,
        start=start,
        end=end,
        tag=tag,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def write_quality_gates(
    *,
    db_path: Path,
    feature_db: Path | None,
    output_dir: Path,
    start: datetime,
    end: datetime,
    tag: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = evaluate_quality_gates(db_path=db_path, feature_db=feature_db, start=start, end=end)
    csv_path = output_dir / f"gdelt_gkg_quality_gates_{tag}.csv"
    json_path = output_dir / f"gdelt_gkg_quality_gates_{tag}.json"
    manifest_path = output_dir / f"gdelt_gkg_quality_gates_manifest_{tag}.json"

    write_gate_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "db_path": str(db_path),
        "feature_db": str(feature_db) if feature_db is not None else "",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "tag": tag,
        "gate_rows": {
            "csv": str(csv_path),
            "json": str(json_path),
            "rows": len(rows),
        },
        "manifest": str(manifest_path),
        "status_counts": {
            status: sum(1 for row in rows if row["status"] == status)
            for status in ("pass", "warn", "fail")
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def evaluate_quality_gates(
    *,
    db_path: Path,
    feature_db: Path | None,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not db_path.exists():
        rows.extend(_missing_source_database_rows(db_path, start, end))
    else:
        with connect_readonly(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows.extend(evaluate_aggregate_coverage(conn, db_path, start, end))
            rows.extend(evaluate_source_holes(conn, db_path, start, end))
            rows.extend(evaluate_raw_file_metadata(conn, db_path, start, end))
            rows.extend(evaluate_silver_tables(conn, db_path, start, end))

    if feature_db is not None:
        rows.extend(evaluate_feature_db(feature_db, start, end))
    return rows


def evaluate_aggregate_coverage(
    conn: sqlite3.Connection,
    db_path: Path,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    totals = {"expected": 0, "successful": 0, "failed": 0, "missing": 0}
    for spec in reports.SOURCE_SPECS:
        expected = len(list(reports.iter_expected_times(start, end, spec.interval_minutes)))
        totals["expected"] += expected
        if not reports.table_exists(conn, spec.table):
            totals["missing"] += expected
            rows.append(
                gate_row(
                    gate_id=f"coverage.{spec.source}",
                    category="aggregate_coverage",
                    source=spec.source,
                    database=db_path,
                    table=spec.table,
                    status="fail",
                    reason=f"{spec.table} is absent; expected {expected} intervals.",
                    observed=0,
                    expected=expected,
                    start=start,
                    end=end,
                )
            )
            continue
        parse_error_expr = (
            "parse_error IS NULL"
            if reports.column_exists(conn, spec.table, "parse_error")
            else "1 = 1"
        )
        stats = conn.execute(
            f"""
            SELECT COUNT(*) AS rows,
                   SUM(CASE WHEN {parse_error_expr} THEN 1 ELSE 0 END) AS successful_rows,
                   SUM(CASE WHEN NOT ({parse_error_expr}) THEN 1 ELSE 0 END) AS failed_rows
            FROM {spec.table}
            WHERE date >= ? AND date < ?
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchone()
        sqlite_rows = int(stats["rows"] or 0)
        successful_rows = int(stats["successful_rows"] or 0)
        failed_rows = int(stats["failed_rows"] or 0)
        missing_rows = max(0, expected - sqlite_rows)
        totals["successful"] += successful_rows
        totals["failed"] += failed_rows
        totals["missing"] += missing_rows
        status = status_from_counts(successful_rows, expected, missing_rows + failed_rows)
        rows.append(
            gate_row(
                gate_id=f"coverage.{spec.source}",
                category="aggregate_coverage",
                source=spec.source,
                database=db_path,
                table=spec.table,
                status=status,
                reason=(
                    f"successful={successful_rows}/{expected}; "
                    f"missing={missing_rows}; failed={failed_rows}."
                ),
                observed=successful_rows,
                expected=expected,
                start=start,
                end=end,
            )
        )

    total_status = status_from_counts(
        totals["successful"],
        totals["expected"],
        totals["missing"] + totals["failed"],
    )
    rows.insert(
        0,
        gate_row(
            gate_id="coverage.aggregate",
            category="aggregate_coverage",
            source="all",
            database=db_path,
            table="",
            status=total_status,
            reason=(
                f"successful={totals['successful']}/{totals['expected']}; "
                f"missing={totals['missing']}; failed={totals['failed']}."
            ),
            observed=totals["successful"],
            expected=totals["expected"],
            start=start,
            end=end,
        ),
    )
    return rows


def evaluate_source_holes(
    conn: sqlite3.Connection,
    db_path: Path,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in reports.SOURCE_SPECS:
        expected_times = list(reports.iter_expected_times(start, end, spec.interval_minutes))
        if not reports.table_exists(conn, spec.table):
            rows.append(
                gate_row(
                    gate_id=f"holes.{spec.source}",
                    category="source_holes",
                    source=spec.source,
                    database=db_path,
                    table=spec.table,
                    status="fail",
                    reason=f"{spec.table} is absent; holes cannot be resolved.",
                    observed=0,
                    expected=len(expected_times),
                    start=start,
                    end=end,
                )
            )
            continue

        parse_error_column = reports.column_exists(conn, spec.table, "parse_error")
        select_error = ", parse_error" if parse_error_column else ""
        stored = {}
        for row in conn.execute(
            f"SELECT date{select_error} FROM {spec.table} WHERE date >= ? AND date < ?",
            (start.isoformat(), end.isoformat()),
        ).fetchall():
            status = "sqlite_parse_error" if parse_error_column and row["parse_error"] else "ok"
            stored[str(row["date"])] = status
        holes = [
            (stamp, stored.get(stamp.isoformat(), "missing_sqlite_row"))
            for stamp in expected_times
            if stored.get(stamp.isoformat(), "missing_sqlite_row") != "ok"
        ]
        status = "pass" if not holes else "fail" if len(holes) == len(expected_times) else "warn"
        reason = "No source holes in the selected window."
        if holes:
            statuses = ",".join(sorted({hole_status for _, hole_status in holes}))
            reason = (
                f"{len(holes)} hole interval(s); "
                f"first={holes[0][0].isoformat()}; statuses={statuses}."
            )
        rows.append(
            gate_row(
                gate_id=f"holes.{spec.source}",
                category="source_holes",
                source=spec.source,
                database=db_path,
                table=spec.table,
                status=status,
                reason=reason,
                observed=len(expected_times) - len(holes),
                expected=len(expected_times),
                start=start,
                end=end,
            )
        )
    return rows


def evaluate_raw_file_metadata(
    conn: sqlite3.Connection,
    db_path: Path,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    if not reports.table_exists(conn, "gdelt_raw_files"):
        return [
            gate_row(
                gate_id="raw_metadata.gdelt_raw_files",
                category="raw_file_metadata",
                source="all",
                database=db_path,
                table="gdelt_raw_files",
                status="warn",
                reason="gdelt_raw_files is absent; raw metadata gates are not applicable.",
                observed=0,
                expected=0,
                start=start,
                end=end,
            )
        ]

    rows: list[dict[str, Any]] = []
    columns = set(reports.table_columns(conn, "gdelt_raw_files"))
    for spec in reports.SOURCE_SPECS:
        file_kind = RAW_KIND_BY_SOURCE[spec.source]
        stamps = [
            stamp.strftime("%Y%m%d%H%M%S")
            for stamp in reports.iter_expected_times(start, end, spec.interval_minutes)
        ]
        expected = len(stamps)
        if not stamps:
            present = parsed = missing_identity = missing_location = 0
        else:
            raw_id_expr = missing_expr("raw_file_id", columns)
            stamp_expr = missing_expr("gdelt_stamp", columns)
            kind_expr = missing_expr("file_kind", columns)
            status_expr = missing_expr("status", columns)
            location_expr = (
                "("
                + " AND ".join(
                    f"({missing_expr(column, columns)})"
                    for column in ("local_path", "file_name", "source_url")
                )
                + ")"
            )
            parsed_expr = "status = 'parsed'" if "status" in columns else "0"
            present = parsed = missing_identity = missing_location = 0
            for offset in range(0, len(stamps), SQLITE_PARAM_BATCH_SIZE):
                stamp_batch = stamps[offset : offset + SQLITE_PARAM_BATCH_SIZE]
                placeholders = ",".join("?" for _ in stamp_batch)
                params = [file_kind, *stamp_batch]
                stats = conn.execute(
                    f"""
                    SELECT COUNT(DISTINCT gdelt_stamp) AS present_stamps,
                           SUM(CASE WHEN {parsed_expr} THEN 1 ELSE 0 END) AS parsed_rows,
                           SUM(
                               CASE
                                   WHEN {raw_id_expr} OR {stamp_expr} OR {kind_expr} OR {status_expr}
                                   THEN 1 ELSE 0
                               END
                           ) AS missing_identity_rows,
                           SUM(CASE WHEN {location_expr} THEN 1 ELSE 0 END) AS missing_location_rows
                    FROM gdelt_raw_files
                    WHERE file_kind = ? AND gdelt_stamp IN ({placeholders})
                    """,
                    params,
                ).fetchone()
                present += int(stats["present_stamps"] or 0)
                parsed += int(stats["parsed_rows"] or 0)
                missing_identity += int(stats["missing_identity_rows"] or 0)
                missing_location += int(stats["missing_location_rows"] or 0)
        missing_stamps = max(0, expected - present)
        metadata_issues = missing_identity + missing_location + max(0, present - parsed)
        status = "pass"
        if present == 0 and expected:
            status = "fail"
        elif missing_stamps or metadata_issues:
            status = "warn"
        rows.append(
            gate_row(
                gate_id=f"raw_metadata.{spec.source}",
                category="raw_file_metadata",
                source=spec.source,
                database=db_path,
                table="gdelt_raw_files",
                status=status,
                reason=(
                    f"present_stamps={present}/{expected}; missing_stamps={missing_stamps}; "
                    f"parsed_rows={parsed}; missing_identity_rows={missing_identity}; "
                    f"missing_location_rows={missing_location}."
                ),
                observed=present,
                expected=expected,
                start=start,
                end=end,
            )
        )
    return rows


def evaluate_silver_tables(
    conn: sqlite3.Connection,
    db_path: Path,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SILVER_SPECS:
        table = str(spec["table"])
        source = str(spec["source"])
        if not reports.table_exists(conn, table):
            rows.append(
                gate_row(
                    gate_id=f"silver_completeness.{source}",
                    category="silver_completeness",
                    source=source,
                    database=db_path,
                    table=table,
                    status="fail",
                    reason=f"{table} is absent.",
                    observed=0,
                    expected=1,
                    start=start,
                    end=end,
                )
            )
            rows.append(
                gate_row(
                    gate_id=f"timestamp_safety.{source}",
                    category="timestamp_safety",
                    source=source,
                    database=db_path,
                    table=table,
                    status="fail",
                    reason=f"{table} is absent; timestamp safety cannot be evaluated.",
                    observed=0,
                    expected=1,
                    start=start,
                    end=end,
                )
            )
            continue

        columns = set(reports.table_columns(conn, table))
        required_columns = tuple(str(column) for column in spec["required_columns"])
        missing_columns = [column for column in required_columns if column not in columns]
        time_column = str(spec["time_column"])
        available_column = str(spec["available_column"])
        if time_column not in columns or available_column not in columns:
            rows.append(
                gate_row(
                    gate_id=f"silver_completeness.{source}",
                    category="silver_completeness",
                    source=source,
                    database=db_path,
                    table=table,
                    status="fail",
                    reason=f"Missing required column(s): {', '.join(missing_columns)}.",
                    observed=0,
                    expected=len(required_columns),
                    start=start,
                    end=end,
                )
            )
            rows.append(
                gate_row(
                    gate_id=f"timestamp_safety.{source}",
                    category="timestamp_safety",
                    source=source,
                    database=db_path,
                    table=table,
                    status="fail",
                    reason=(
                        f"Missing {available_column} or {time_column}; "
                        "timestamp safety cannot be evaluated."
                    ),
                    observed=0,
                    expected=1,
                    start=start,
                    end=end,
                )
            )
            continue

        stats = silver_stats(
            conn,
            table,
            columns,
            required_columns,
            time_column,
            available_column,
            start,
            end,
        )
        completeness_status = "pass"
        if stats["rows"] == 0:
            completeness_status = "fail"
        elif missing_columns or stats["missing_required_rows"]:
            completeness_status = "warn"
        rows.append(
            gate_row(
                gate_id=f"silver_completeness.{source}",
                category="silver_completeness",
                source=source,
                database=db_path,
                table=table,
                status=completeness_status,
                reason=(
                    f"rows={stats['rows']}; missing_columns={','.join(missing_columns) or 'none'}; "
                    f"rows_with_missing_required={stats['missing_required_rows']}."
                ),
                observed=stats["rows"] - stats["missing_required_rows"],
                expected=stats["rows"],
                start=start,
                end=end,
            )
        )

        timestamp_status = (
            "pass"
            if not stats["missing_available_rows"] and not stats["available_before_time_rows"]
            else "fail"
        )
        safe_timestamp_rows = (
            stats["rows"] - stats["missing_available_rows"] - stats["available_before_time_rows"]
        )
        rows.append(
            gate_row(
                gate_id=f"timestamp_safety.{source}",
                category="timestamp_safety",
                source=source,
                database=db_path,
                table=table,
                status=timestamp_status,
                reason=(
                    f"missing_available_at_rows={stats['missing_available_rows']}; "
                    f"available_before_{time_column}_rows={stats['available_before_time_rows']}."
                ),
                observed=safe_timestamp_rows,
                expected=stats["rows"],
                start=start,
                end=end,
            )
        )
    return rows


def evaluate_feature_db(feature_db: Path, start: datetime, end: datetime) -> list[dict[str, Any]]:
    if not feature_db.exists():
        return [
            gate_row(
                gate_id="feature_db.context_features_1h",
                category="feature_db",
                source="context_features",
                database=feature_db,
                table="context_features_1h",
                status="warn",
                reason="Feature database is absent; feature gates are not applicable.",
                observed=0,
                expected=0,
                start=start,
                end=end,
            )
        ]

    with connect_readonly(feature_db) as conn:
        conn.row_factory = sqlite3.Row
        if not reports.table_exists(conn, "context_features_1h"):
            return [
                gate_row(
                    gate_id="feature_db.context_features_1h",
                    category="feature_db",
                    source="context_features",
                    database=feature_db,
                    table="context_features_1h",
                    status="warn",
                    reason="context_features_1h is absent; feature gates are not applicable.",
                    observed=0,
                    expected=0,
                    start=start,
                    end=end,
                )
            ]
        columns = set(reports.table_columns(conn, "context_features_1h"))
        expected_hours = len(list(reports.iter_expected_times(start, end, 60)))
        date_column = "date" if "date" in columns else ""
        if not date_column:
            return [
                gate_row(
                    gate_id="feature_db.context_features_1h",
                    category="feature_db",
                    source="context_features",
                    database=feature_db,
                    table="context_features_1h",
                    status="fail",
                    reason="context_features_1h is missing date.",
                    observed=0,
                    expected=expected_hours,
                    start=start,
                    end=end,
                )
            ]
        max_source_column = (
            "max_source_available_at" if "max_source_available_at" in columns else ""
        )
        lookahead_expr = (
            f"SUM(CASE WHEN {max_source_column} > date THEN 1 ELSE 0 END)"
            if max_source_column
            else "0"
        )
        missing_source_expr = (
            f"SUM(CASE WHEN {missing_expr(max_source_column, columns)} THEN 1 ELSE 0 END)"
            if max_source_column
            else "0"
        )
        stats = conn.execute(
            f"""
            SELECT COUNT(*) AS rows,
                   {lookahead_expr} AS lookahead_rows,
                   {missing_source_expr} AS missing_source_timestamp_rows
            FROM context_features_1h
            WHERE date >= ? AND date < ?
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchone()
    feature_rows = int(stats["rows"] or 0)
    lookahead_rows = int(stats["lookahead_rows"] or 0)
    missing_source_rows = int(stats["missing_source_timestamp_rows"] or 0)
    status = "pass"
    if feature_rows == 0:
        status = "fail"
    elif feature_rows < expected_hours or lookahead_rows or missing_source_rows:
        status = "warn" if not lookahead_rows else "fail"
    return [
        gate_row(
            gate_id="feature_db.context_features_1h",
            category="feature_db",
            source="context_features",
            database=feature_db,
            table="context_features_1h",
            status=status,
            reason=(
                f"rows={feature_rows}/{expected_hours}; lookahead_rows={lookahead_rows}; "
                f"missing_max_source_available_at_rows={missing_source_rows}."
            ),
            observed=feature_rows,
            expected=expected_hours,
            start=start,
            end=end,
        )
    ]


def silver_stats(
    conn: sqlite3.Connection,
    table: str,
    columns: set[str],
    required_columns: tuple[str, ...],
    time_column: str,
    available_column: str,
    start: datetime,
    end: datetime,
) -> dict[str, int]:
    checked_columns = [column for column in required_columns if column in columns]
    missing_required_expr = " OR ".join(missing_expr(column, columns) for column in checked_columns)
    if not missing_required_expr:
        missing_required_expr = "0"
    stats = conn.execute(
        f"""
        SELECT COUNT(*) AS rows,
               SUM(CASE WHEN {missing_required_expr} THEN 1 ELSE 0 END) AS missing_required_rows,
               SUM(
                   CASE WHEN {missing_expr(available_column, columns)}
                   THEN 1 ELSE 0 END
               ) AS missing_available_rows,
               SUM(
                   CASE
                       WHEN NOT ({missing_expr(available_column, columns)})
                        AND NOT ({missing_expr(time_column, columns)})
                        AND {available_column} < {time_column}
                       THEN 1 ELSE 0
                   END
               ) AS available_before_time_rows
        FROM {table}
        WHERE {time_column} >= ? AND {time_column} < ?
        """,
        (start.isoformat(), end.isoformat()),
    ).fetchone()
    return {
        "rows": int(stats["rows"] or 0),
        "missing_required_rows": int(stats["missing_required_rows"] or 0),
        "missing_available_rows": int(stats["missing_available_rows"] or 0),
        "available_before_time_rows": int(stats["available_before_time_rows"] or 0),
    }


def connect_readonly(path: Path) -> sqlite3.Connection:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, timeout=10.0, uri=True)
    conn.execute("PRAGMA query_only=ON;")
    return conn


def missing_expr(column: str, columns: set[str]) -> str:
    if column not in columns:
        return "1"
    return f"{column} IS NULL OR TRIM(CAST({column} AS TEXT)) = ''"


def status_from_counts(successful: int, expected: int, issues: int) -> str:
    if expected == 0:
        return "pass"
    if successful == expected and issues == 0:
        return "pass"
    if successful == 0:
        return "fail"
    return "warn"


def gate_row(
    *,
    gate_id: str,
    category: str,
    source: str,
    database: Path | str,
    table: str,
    status: str,
    reason: str,
    observed: int,
    expected: int,
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "category": category,
        "source": source,
        "database": str(database),
        "table": table,
        "status": status,
        "reason": reason,
        "observed": int(observed),
        "expected": int(expected),
        "ratio": reports.ratio(observed, expected),
        "start": start.isoformat(),
        "end": end.isoformat(),
    }


def write_gate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [*GATE_FIELDNAMES]
    extras = sorted({key for row in rows for key in row} - set(fieldnames))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *extras])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def deterministic_tag(start: datetime, end: datetime) -> str:
    start_text = start.astimezone(UTC).strftime("%Y%m%d%H%M%S")
    end_text = end.astimezone(UTC).strftime("%Y%m%d%H%M%S")
    return f"{start_text}_{end_text}"


def _missing_source_database_rows(
    db_path: Path,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    rows = [
        gate_row(
            gate_id="coverage.aggregate",
            category="aggregate_coverage",
            source="all",
            database=db_path,
            table="",
            status="fail",
            reason=f"Database not found: {db_path}",
            observed=0,
            expected=1,
            start=start,
            end=end,
        )
    ]
    for spec in reports.SOURCE_SPECS:
        expected = len(list(reports.iter_expected_times(start, end, spec.interval_minutes)))
        rows.append(
            gate_row(
                gate_id=f"coverage.{spec.source}",
                category="aggregate_coverage",
                source=spec.source,
                database=db_path,
                table=spec.table,
                status="fail",
                reason=f"Database not found: {db_path}",
                observed=0,
                expected=expected,
                start=start,
                end=end,
            )
        )
        rows.append(
            gate_row(
                gate_id=f"holes.{spec.source}",
                category="source_holes",
                source=spec.source,
                database=db_path,
                table=spec.table,
                status="fail",
                reason=f"Database not found: {db_path}",
                observed=0,
                expected=expected,
                start=start,
                end=end,
            )
        )
    for spec in SILVER_SPECS:
        rows.append(
            gate_row(
                gate_id=f"silver_completeness.{spec['source']}",
                category="silver_completeness",
                source=str(spec["source"]),
                database=db_path,
                table=str(spec["table"]),
                status="fail",
                reason=f"Database not found: {db_path}",
                observed=0,
                expected=1,
                start=start,
                end=end,
            )
        )
    return rows


__all__ = [
    "evaluate_quality_gates",
    "write_quality_gates",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())

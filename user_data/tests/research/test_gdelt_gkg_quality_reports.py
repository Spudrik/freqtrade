from __future__ import annotations

import json
import sqlite3
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from user_data.Custom_Launcher.research.context_features import gdelt_gkg_quality_reports as reports


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def _create_gdelt_context_db(path: Path) -> None:
    with sqlite3.connect(str(path)) as conn:
        conn.executescript(
            """
            CREATE TABLE gdelt_hourly_features (
                date TEXT PRIMARY KEY,
                event_count REAL NOT NULL DEFAULT 0,
                parse_error TEXT
            );
            CREATE TABLE gdelt_gkg_file_features (
                date TEXT PRIMARY KEY,
                document_count REAL NOT NULL DEFAULT 0,
                parse_error TEXT
            );
            CREATE TABLE gdelt_events_silver (
                event_id TEXT PRIMARY KEY,
                raw_file_id TEXT,
                event_time TEXT,
                available_at TEXT NOT NULL,
                source_url_hash TEXT,
                topic TEXT
            );
            CREATE TABLE gkg_documents_silver (
                document_id TEXT PRIMARY KEY,
                raw_file_id TEXT,
                document_time TEXT,
                available_at TEXT NOT NULL,
                url_hash TEXT,
                topic TEXT
            );
            """
        )
        conn.executemany(
            "INSERT INTO gdelt_hourly_features(date, event_count, parse_error) VALUES (?, ?, ?)",
            [
                ("2022-07-28T00:00:00+00:00", 12.0, None),
                ("2022-07-28T01:00:00+00:00", 0.0, "connection refused"),
            ],
        )
        conn.executemany(
            "INSERT INTO gdelt_gkg_file_features(date, document_count, parse_error) VALUES (?, ?, ?)",
            [
                ("2022-07-28T00:00:00+00:00", 5.0, None),
                ("2022-07-28T00:15:00+00:00", 0.0, "HTTP 404"),
                ("2022-07-28T00:30:00+00:00", 0.0, "File is not a zip file"),
            ],
        )
        conn.execute(
            """
            INSERT INTO gdelt_events_silver(
                event_id, raw_file_id, event_time, available_at, source_url_hash, topic
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "event-1",
                "raw-1",
                "2022-07-28T00:00:00+00:00",
                "2022-07-28T00:00:00+00:00",
                "hash-1",
                "macro",
            ),
        )
        conn.execute(
            """
            INSERT INTO gkg_documents_silver(
                document_id, raw_file_id, document_time, available_at, url_hash, topic
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "doc-1",
                "raw-2",
                "2022-07-28T00:00:00+00:00",
                "2022-07-28T00:15:00+00:00",
                "hash-2",
                "crypto",
            ),
        )


def test_coverage_error_classification_and_holes(tmp_path):
    db_path = tmp_path / "gdelt_context.sqlite"
    _create_gdelt_context_db(db_path)
    start = _dt("2022-07-28T00:00:00+00:00")
    end = _dt("2022-07-28T03:00:00+00:00")

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        coverage = reports.build_sqlite_coverage_summary(conn, start, end)
        errors = reports.build_error_breakdown(conn, start, end)
        holes = reports.build_source_hole_report(conn, start, end)

    gdelt_row = next(row for row in coverage if row["source"] == "gdelt_events")
    assert gdelt_row["expected_files"] == 3
    assert gdelt_row["successful_rows"] == 1
    assert gdelt_row["failed_rows"] == 1
    assert gdelt_row["missing_rows"] == 1
    assert gdelt_row["metric_sum"] == 12.0

    categories = {(row["source"], row["error_category"]) for row in errors}
    assert ("gdelt_events", "network_error") in categories
    assert ("gkg_documents", "http_404_terminal_missing") in categories
    assert ("gkg_documents", "zip_error") in categories

    hole_statuses = {(row["source"], row["status"], row["start"]) for row in holes}
    assert ("gdelt_events", "sqlite_parse_error", "2022-07-28T01:00:00+00:00") in hole_statuses
    assert ("gdelt_events", "missing_sqlite_row", "2022-07-28T02:00:00+00:00") in hole_statuses
    assert ("gkg_documents", "missing_sqlite_row", "2022-07-28T00:45:00+00:00") in hole_statuses


def test_write_quality_reports_outputs_csv_json_and_silver_summary(tmp_path):
    db_path = tmp_path / "gdelt_context.sqlite"
    feature_db = tmp_path / "context_features.sqlite"
    output_dir = tmp_path / "reports"
    gdelt_raw_dir = tmp_path / "raw" / "export"
    gkg_raw_dir = tmp_path / "raw" / "gkg"
    gdelt_raw_dir.mkdir(parents=True)
    gkg_raw_dir.mkdir(parents=True)
    _create_gdelt_context_db(db_path)
    _write_zip(gdelt_raw_dir / "20220728000000.export.CSV.zip", "20220728000000.export.CSV", "a\tb\n")
    (gkg_raw_dir / "20220728000000.gkg.csv.zip").write_text("not zip", encoding="utf-8")
    _create_feature_db(feature_db)

    manifest = reports.write_quality_reports(
        db_path=db_path,
        feature_db=feature_db,
        output_dir=output_dir,
        start=_dt("2022-07-28T00:00:00+00:00"),
        end=_dt("2022-07-28T01:00:00+00:00"),
        tag="unit",
        gdelt_raw_dir=gdelt_raw_dir,
        gkg_raw_dir=gkg_raw_dir,
        limit_files=1,
    )

    for report in manifest["reports"].values():
        assert Path(report["csv"]).exists()
        assert Path(report["json"]).exists()
    assert Path(manifest["manifest"]).exists()

    raw_rows = json.loads(Path(manifest["reports"]["raw_zip_validation"]["json"]).read_text(encoding="utf-8"))
    assert any(row["source"] == "gdelt_events" and row["status"] == "ok" for row in raw_rows)
    assert any(row["source"] == "gkg_documents" and row["status"] == "bad_zip" for row in raw_rows)

    normalized_rows = json.loads(Path(manifest["reports"]["normalized_completeness"]["json"]).read_text(encoding="utf-8"))
    gdelt_silver = next(row for row in normalized_rows if row["source"] == "gdelt_events")
    gkg_silver = next(row for row in normalized_rows if row["source"] == "gkg_documents")
    assert gdelt_silver["rows"] == 1
    assert gdelt_silver["columns_checked"] == 1
    assert gdelt_silver["cell_non_null_ratio"] == 1.0
    assert gkg_silver["future_available_at_rows"] == 0
    gdelt_events_silver = next(row for row in normalized_rows if row["source"] == "gdelt_events_silver")
    gkg_documents_silver = next(row for row in normalized_rows if row["source"] == "gkg_documents_silver")
    assert gdelt_events_silver["rows"] == 1
    assert gdelt_events_silver["distinct_raw_files"] == 1
    assert gkg_documents_silver["missing_topic_rows"] == 0


def _write_zip(path: Path, name: str, payload: str) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(name, payload)


def _create_feature_db(path: Path) -> None:
    with sqlite3.connect(str(path)) as conn:
        conn.executescript(
            """
            CREATE TABLE context_features_1h (
                date TEXT PRIMARY KEY,
                gdelt_event_count_1h REAL,
                gkg_doc_count_1h REAL,
                max_source_available_at TEXT
            );
            CREATE TABLE normalised_context_events (
                event_id TEXT PRIMARY KEY,
                source_family TEXT,
                available_at TEXT
            );
            """
        )
        conn.execute(
            """
            INSERT INTO context_features_1h(
                date, gdelt_event_count_1h, gkg_doc_count_1h, max_source_available_at
            )
            VALUES (?, ?, ?, ?)
            """,
            ("2022-07-28T00:00:00+00:00", 12.0, 5.0, "2022-07-28T00:00:00+00:00"),
        )
        conn.execute(
            "INSERT INTO normalised_context_events(event_id, source_family, available_at) VALUES (?, ?, ?)",
            ("news:1", "news", "2022-07-28T00:00:00+00:00"),
        )

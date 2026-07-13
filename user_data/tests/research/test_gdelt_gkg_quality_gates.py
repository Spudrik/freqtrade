from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from user_data.Custom_Launcher.research.context_features import gdelt_gkg_quality_gates as gates


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def test_quality_gates_flag_coverage_holes_raw_metadata_and_timestamp_safety(tmp_path):
    db_path = tmp_path / "gdelt_context.sqlite"
    feature_db = tmp_path / "context_features.sqlite"
    _create_source_db(db_path)
    _create_feature_db(feature_db)

    rows = gates.evaluate_quality_gates(
        db_path=db_path,
        feature_db=feature_db,
        start=_dt("2022-07-28T00:00:00+00:00"),
        end=_dt("2022-07-28T01:00:00+00:00"),
    )
    by_gate = {row["gate_id"]: row for row in rows}

    assert by_gate["coverage.aggregate"]["status"] == "warn"
    assert by_gate["coverage.gdelt_events"]["status"] == "pass"
    assert by_gate["coverage.gkg_documents"]["status"] == "warn"
    assert by_gate["holes.gkg_documents"]["status"] == "warn"
    assert "missing_sqlite_row" in by_gate["holes.gkg_documents"]["reason"]

    assert by_gate["raw_metadata.gdelt_events"]["status"] == "pass"
    assert by_gate["raw_metadata.gkg_documents"]["status"] == "warn"
    assert "present_stamps=2/4" in by_gate["raw_metadata.gkg_documents"]["reason"]

    assert by_gate["silver_completeness.gdelt_events_silver"]["status"] == "warn"
    assert (
        "rows_with_missing_required=1"
        in by_gate["silver_completeness.gdelt_events_silver"]["reason"]
    )
    assert by_gate["timestamp_safety.gdelt_events_silver"]["status"] == "fail"
    assert (
        "available_before_event_time_rows=1"
        in by_gate["timestamp_safety.gdelt_events_silver"]["reason"]
    )
    assert by_gate["timestamp_safety.gkg_documents_silver"]["status"] == "pass"
    assert by_gate["feature_db.context_features_1h"]["status"] == "fail"
    assert "lookahead_rows=1" in by_gate["feature_db.context_features_1h"]["reason"]


def test_write_quality_gates_outputs_csv_json_manifest_and_is_deterministic(tmp_path):
    db_path = tmp_path / "gdelt_context.sqlite"
    output_dir = tmp_path / "reports"
    _create_source_db(db_path)

    manifest = gates.write_quality_gates(
        db_path=db_path,
        feature_db=None,
        output_dir=output_dir,
        start=_dt("2022-07-28T00:00:00+00:00"),
        end=_dt("2022-07-28T01:00:00+00:00"),
        tag="unit",
    )

    csv_path = Path(manifest["gate_rows"]["csv"])
    json_path = Path(manifest["gate_rows"]["json"])
    manifest_path = Path(manifest["manifest"])
    assert csv_path.exists()
    assert json_path.exists()
    assert manifest_path.exists()
    assert manifest["status_counts"]["fail"] >= 1
    assert manifest == json.loads(manifest_path.read_text(encoding="utf-8"))

    rows = json.loads(json_path.read_text(encoding="utf-8"))
    assert rows[0]["gate_id"] == "coverage.aggregate"
    assert (
        csv_path.read_text(encoding="utf-8").splitlines()[0].startswith("gate_id,category,source")
    )
    assert (
        gates.deterministic_tag(
            _dt("2022-07-28T00:00:00+00:00"),
            _dt("2022-07-28T01:00:00+00:00"),
        )
        == "20220728000000_20220728010000"
    )


def _create_source_db(path: Path) -> None:
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
            CREATE TABLE gdelt_raw_files (
                raw_file_id TEXT PRIMARY KEY,
                file_kind TEXT,
                gdelt_stamp TEXT,
                source_url TEXT,
                local_path TEXT,
                file_name TEXT,
                status TEXT
            );
            CREATE TABLE gdelt_events_silver (
                event_id TEXT PRIMARY KEY,
                raw_file_id TEXT,
                event_time TEXT,
                available_at TEXT,
                source_url_hash TEXT,
                topic TEXT
            );
            CREATE TABLE gkg_documents_silver (
                document_id TEXT PRIMARY KEY,
                raw_file_id TEXT,
                document_time TEXT,
                available_at TEXT,
                url_hash TEXT,
                topic TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO gdelt_hourly_features(date, event_count, parse_error) VALUES (?, ?, ?)",
            ("2022-07-28T00:00:00+00:00", 7.0, None),
        )
        conn.executemany(
            """
            INSERT INTO gdelt_gkg_file_features(date, document_count, parse_error)
            VALUES (?, ?, ?)
            """,
            [
                ("2022-07-28T00:00:00+00:00", 3.0, None),
                ("2022-07-28T00:15:00+00:00", 5.0, None),
            ],
        )
        conn.executemany(
            """
            INSERT INTO gdelt_raw_files(
                raw_file_id, file_kind, gdelt_stamp, source_url, local_path, file_name, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "raw-export-1",
                    "export",
                    "20220728000000",
                    None,
                    "D:/raw/20220728000000.export.CSV.zip",
                    "20220728000000.export.CSV.zip",
                    "parsed",
                ),
                (
                    "raw-gkg-1",
                    "gkg",
                    "20220728000000",
                    None,
                    "D:/raw/20220728000000.gkg.csv.zip",
                    "20220728000000.gkg.csv.zip",
                    "parsed",
                ),
                ("raw-gkg-2", "gkg", "20220728001500", None, None, None, "pending"),
            ],
        )
        conn.executemany(
            """
            INSERT INTO gdelt_events_silver(
                event_id, raw_file_id, event_time, available_at, source_url_hash, topic
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "event-ok",
                    "raw-export-1",
                    "2022-07-28T00:10:00+00:00",
                    "2022-07-28T00:15:00+00:00",
                    "url-hash-1",
                    "macro",
                ),
                (
                    "event-bad-time",
                    "raw-export-1",
                    "2022-07-28T00:30:00+00:00",
                    "2022-07-28T00:20:00+00:00",
                    "",
                    "macro",
                ),
            ],
        )
        conn.execute(
            """
            INSERT INTO gkg_documents_silver(
                document_id, raw_file_id, document_time, available_at, url_hash, topic
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "doc-ok",
                "raw-gkg-1",
                "2022-07-28T00:05:00+00:00",
                "2022-07-28T00:15:00+00:00",
                "url-hash-2",
                "crypto",
            ),
        )


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
            """
        )
        conn.execute(
            """
            INSERT INTO context_features_1h(
                date, gdelt_event_count_1h, gkg_doc_count_1h, max_source_available_at
            )
            VALUES (?, ?, ?, ?)
            """,
            ("2022-07-28T00:00:00+00:00", 7.0, 8.0, "2022-07-28T00:15:00+00:00"),
        )

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from user_data.Custom_Launcher.research.context_features import gdelt_gkg_gold_features as gold
from user_data.Custom_Launcher.research.context_features import gdelt_gkg_normalized_schema as schema


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def test_gold_features_roll_up_silver_rows_with_quality_metadata(tmp_path):
    db_path = tmp_path / "gdelt_context.sqlite"
    output_dir = tmp_path / "reports"
    schema.init_db(db_path)
    with schema.connect_db(db_path) as conn:
        _insert_fixture_rows(conn)

    summary = gold.build_gold_features(
        db_path=db_path,
        output_dir=output_dir,
        start=_dt("2022-07-28T00:00:00+00:00"),
        end=_dt("2022-07-28T02:00:00+00:00"),
        tag="unit",
    )

    assert summary.rows_written == 2
    assert summary.csv_path is not None
    assert summary.json_path is not None

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT * FROM {gold.GOLD_TOPIC_TABLE} ORDER BY feature_hour"
        ).fetchall()

    assert len(rows) == 2
    first = rows[0]
    assert first["event_count"] == 1
    assert first["document_count"] == 3
    assert first["source_count"] == 3
    assert first["banking_credit_confluence_24h"] == 1
    assert first["oil_energy_first_mention"] == 1
    assert first["regulation_persistence_72h"] > 0
    assert first["macro_release_severity_max_24h"] == 0.8
    assert first["conflict_event_count_24h"] == 1
    metadata = json.loads(first["metadata_json"])
    assert metadata["topic_quality"] == "weak_silver_labels_pending_classifier_enrichment"


def test_gold_features_dry_run_writes_artifacts_but_not_table_rows(tmp_path):
    db_path = tmp_path / "gdelt_context.sqlite"
    output_dir = tmp_path / "reports"
    schema.init_db(db_path)

    summary = gold.build_gold_features(
        db_path=db_path,
        output_dir=output_dir,
        start=_dt("2022-07-28T00:00:00+00:00"),
        end=_dt("2022-07-28T01:00:00+00:00"),
        tag="dry",
        dry_run=True,
    )

    assert summary.rows_written == 0
    assert summary.csv_path is not None
    assert summary.json_path is not None
    with sqlite3.connect(db_path) as conn:
        count = conn.execute(f"SELECT COUNT(*) FROM {gold.GOLD_TOPIC_TABLE}").fetchone()[0]
    assert count == 0


def _insert_fixture_rows(conn):
    conn.execute(
        """
        INSERT INTO gdelt_raw_files(raw_file_id, file_kind, gdelt_stamp, status)
        VALUES ('raw-event', 'export', '20220728000000', 'parsed'),
               ('raw-gkg', 'gkg', '20220728000000', 'parsed')
        """
    )
    conn.execute(
        """
        INSERT INTO gdelt_events_silver(
            event_id, raw_file_id, event_time, available_at, topic, impact_channel,
            severity_proxy, source_url_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "event-1",
            "raw-event",
            "2022-07-28T00:05:00+00:00",
            "2022-07-28T00:10:00+00:00",
            "threat",
            "material_conflict",
            0.8,
            "source-event",
        ),
    )
    conn.executemany(
        """
        INSERT INTO gkg_documents_silver(
            document_id, raw_file_id, document_time, available_at, source_common_name,
            document_url, topic, url_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "doc-bank",
                "raw-gkg",
                "2022-07-28T00:00:00+00:00",
                "2022-07-28T00:15:00+00:00",
                "bank-source",
                "https://example.com/bank",
                "banking_credit",
                "hash-bank",
            ),
            (
                "doc-oil",
                "raw-gkg",
                "2022-07-28T00:00:00+00:00",
                "2022-07-28T00:20:00+00:00",
                "oil-source",
                "https://example.com/oil",
                "oil_energy",
                "hash-oil",
            ),
            (
                "doc-reg",
                "raw-gkg",
                "2022-07-28T00:00:00+00:00",
                "2022-07-28T00:25:00+00:00",
                "reg-source",
                "https://example.com/reg",
                "regulation",
                "hash-reg",
            ),
        ],
    )

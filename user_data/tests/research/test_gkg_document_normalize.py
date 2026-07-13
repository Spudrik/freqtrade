from datetime import datetime, timezone
import json
import sqlite3
import zipfile

from user_data.Custom_Launcher.research.context_features import gkg_document_normalize as normalizer
from user_data.Custom_Launcher.research.context_features import gdelt_gkg_normalized_schema as schema


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def _gkg_row(
    *,
    source_collection_identifier: str = "1",
    document_identifier: str = "https://example.com/bitcoin-fed-story",
    date: str = "20220101093000",
    extras: str = "PAGE_TIMESTAMP=20220101094530;LANGUAGE=en",
) -> list[str]:
    row = [""] * 27
    row[normalizer.COL_GKG_RECORD_ID] = "20220101093000-1"
    row[normalizer.COL_DATE] = date
    row[normalizer.COL_SOURCE_COLLECTION_IDENTIFIER] = source_collection_identifier
    row[normalizer.COL_SOURCE_NAME] = "example.com"
    row[normalizer.COL_DOCUMENT_IDENTIFIER] = document_identifier
    row[normalizer.COL_COUNTS] = "ARREST#2#police#1#United States#US#US#39#-97#US#12"
    row[normalizer.COL_V2_COUNTS] = "AFFECT#3#people#1#London, United Kingdom#UK#UKH9#51.5#-0.1#-2601889#20"
    row[normalizer.COL_THEMES] = "ECON_STOCKMARKET;TAX_FNCACT_BANKER"
    row[normalizer.COL_V2_THEMES] = "CRYPTO,123;WB_694_TRADE,456"
    row[normalizer.COL_LOCATIONS] = "1#United States#US#US#39#-97#US"
    row[normalizer.COL_V2_LOCATIONS] = "1#London, United Kingdom#UK#UKH9#51.5#-0.1#-2601889#222"
    row[normalizer.COL_PERSONS] = "Jerome Powell"
    row[normalizer.COL_V2_PERSONS] = "Satoshi Nakamoto,22"
    row[normalizer.COL_ORGS] = "Federal Reserve"
    row[normalizer.COL_V2_ORGS] = "Coinbase,72"
    row[normalizer.COL_V2_TONE] = "1.5,2.5,1.0,3.5,4.5,5.5,678"
    row[normalizer.COL_DATES] = "1#20220101#77"
    row[normalizer.COL_GCAM] = "wc:100,c1.1:2.25"
    row[normalizer.COL_ALL_NAMES] = "Bitcoin,50;Coinbase,72"
    row[normalizer.COL_AMOUNTS] = "1000,dollars,88"
    row[normalizer.COL_EXTRAS] = extras
    return row


def test_synthetic_gkg_row_normalizes_document_fields():
    record = normalizer.parse_gkg_row(
        _gkg_row(),
        available_at=_dt("2022-01-01T10:00:00+00:00"),
        raw_ref={"zip_name": "20220101100000.gkg.csv.zip", "row_number": 1},
    )

    assert record is not None
    assert record["gkg_record_id"] == "20220101093000-1"
    assert record["source_collection_identifier"] == 1
    assert record["source_name"] == "example.com"
    assert record["document_identifier"] == "https://example.com/bitcoin-fed-story"
    assert record["document_url"] == "https://example.com/bitcoin-fed-story"
    assert record["available_at"] == "2022-01-01T10:00:00+00:00"
    assert record["published_at"] == "2022-01-01T09:45:30+00:00"

    themes = json.loads(record["themes_json"])
    locations = json.loads(record["locations_json"])
    weak_topics = json.loads(record["weak_topics_json"])
    weak_entities = json.loads(record["weak_entities_json"])
    raw_ref = json.loads(record["raw_ref_json"])

    assert {"theme": "CRYPTO", "offset": 123, "version": "v2"} in themes
    assert locations[-1]["full_name"] == "London, United Kingdom"
    assert locations[-1]["offset"] == 222
    assert "crypto" in weak_topics
    assert {"type": "organization", "name": "Coinbase"} in weak_entities
    assert raw_ref["zip_name"] == "20220101100000.gkg.csv.zip"


def test_document_url_only_for_web_source_collection():
    record = normalizer.parse_gkg_row(
        _gkg_row(source_collection_identifier="2", document_identifier="broadcast-document-id"),
        available_at=_dt("2022-01-01T10:00:00+00:00"),
        raw_ref={},
    )

    assert record is not None
    assert record["document_identifier"] == "broadcast-document-id"
    assert record["document_url"] is None


def test_v2_tone_parses_named_fields():
    tone = normalizer.parse_v2_tone("1.5,2.5,1.0,3.5,4.5,5.5,678")

    assert tone == {
        "tone": 1.5,
        "positive_score": 2.5,
        "negative_score": 1.0,
        "polarity": 3.5,
        "activity_reference_density": 4.5,
        "self_group_reference_density": 5.5,
        "word_count": 678.0,
    }


def test_available_at_comes_from_zip_batch_stamp_and_published_at_falls_back_to_date(tmp_path, monkeypatch, capsys):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    db_path = tmp_path / "gdelt_context.sqlite"
    zip_path = raw_dir / "20220101100000.gkg.csv.zip"
    row = "\t".join(_gkg_row(extras="LANGUAGE=en"))
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("20220101100000.gkg.csv", row)

    monkeypatch.setattr(
        "sys.argv",
        [
            "gkg_document_normalize",
            "--raw-dir",
            str(raw_dir),
            "--db",
            str(db_path),
            "--limit-files",
            "1",
            "--limit-rows",
            "1",
        ],
    )

    assert normalizer.main() == 0
    captured = capsys.readouterr()
    assert '"rows_written": 1' in captured.out

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT available_at, published_at FROM gkg_documents_silver").fetchone()

    assert row["available_at"] == "2022-01-01T10:00:00+00:00"
    assert row["published_at"] == "2022-01-01T09:30:00+00:00"


def test_write_rows_uses_shared_normalized_schema(tmp_path):
    db_path = tmp_path / "gdelt_context.sqlite"
    schema.init_db(db_path)
    record = normalizer.parse_gkg_row(
        _gkg_row(),
        available_at=_dt("2022-01-01T10:00:00+00:00"),
        raw_ref={
            "zip_path": str(tmp_path / "20220101100000.gkg.csv.zip"),
            "zip_name": "20220101100000.gkg.csv.zip",
            "batch_stamp": "20220101100000",
            "row_number": 1,
        },
    )

    with normalizer.connect_db(db_path) as conn:
        inserted = normalizer.write_rows(conn, [record])
        saved = conn.execute(
            """
            SELECT document_id, raw_file_id, document_time, available_at, source_common_name,
                   document_url, canonical_url, url_hash, topic, raw_ref_json, parser_version
            FROM gkg_documents_silver
            """
        ).fetchone()
        raw_file = conn.execute(
            "SELECT file_kind, gdelt_stamp, status FROM gdelt_raw_files WHERE raw_file_id = ?",
            (saved["raw_file_id"],),
        ).fetchone()

    assert inserted == 1
    assert saved["document_id"] == "20220101093000-1"
    assert saved["document_time"] == "2022-01-01T09:45:30+00:00"
    assert saved["available_at"] == "2022-01-01T10:00:00+00:00"
    assert saved["source_common_name"] == "example.com"
    assert saved["document_url"] == "https://example.com/bitcoin-fed-story"
    assert saved["canonical_url"] == "https://example.com/bitcoin-fed-story"
    assert len(saved["url_hash"]) == 64
    assert saved["topic"] == "crypto"
    assert json.loads(saved["raw_ref_json"])["zip_name"] == "20220101100000.gkg.csv.zip"
    assert saved["parser_version"] == "gkg_document_normalize_v1"
    assert dict(raw_file) == {"file_kind": "gkg", "gdelt_stamp": "20220101100000", "status": "parsed"}


def test_dry_run_does_not_create_database(tmp_path, monkeypatch, capsys):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    db_path = tmp_path / "gdelt_context.sqlite"
    zip_path = raw_dir / "20220101100000.gkg.csv.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("20220101100000.gkg.csv", "\t".join(_gkg_row()))

    monkeypatch.setattr(
        "sys.argv",
        [
            "gkg_document_normalize",
            "--raw-dir",
            str(raw_dir),
            "--db",
            str(db_path),
            "--dry-run",
            "--limit-files",
            "1",
            "--limit-rows",
            "1",
        ],
    )

    assert normalizer.main() == 0
    captured = capsys.readouterr()

    assert '"files_planned": 1' in captured.out
    assert '"rows_written": 0' in captured.out
    assert not db_path.exists()


def test_cli_date_window_filters_batch_stamp_before_parsing_and_uses_exclusive_end(tmp_path, monkeypatch, capsys):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    db_path = tmp_path / "gdelt_context.sqlite"
    (raw_dir / "20220101095959.gkg.csv.zip").write_text("outside before window", encoding="utf-8")
    (raw_dir / "20220101110000.gkg.csv.zip").write_text("outside exclusive end", encoding="utf-8")
    zip_path = raw_dir / "20220101100000.gkg.csv.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("20220101100000.gkg.csv", "\t".join(_gkg_row()))

    monkeypatch.setattr(
        "sys.argv",
        [
            "gkg_document_normalize",
            "--raw-dir",
            str(raw_dir),
            "--db",
            str(db_path),
            "--dry-run",
            "--start",
            "20220101100000",
            "--end",
            "20220101110000",
        ],
    )

    assert normalizer.main() == 0
    summary = json.loads(capsys.readouterr().out)

    assert summary["files_planned"] == 1
    assert summary["files_read"] == 1
    assert summary["rows_read"] == 1
    assert summary["rows_written"] == 0
    assert not db_path.exists()

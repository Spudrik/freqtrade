from datetime import datetime, timezone
import json
import sqlite3
import zipfile

from user_data.Custom_Launcher.research.context_features import gdelt_event_normalize
from user_data.Custom_Launcher.research.context_features import gdelt_gkg_normalized_schema


def _stamp(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def _synthetic_export_row() -> list[str]:
    values = [""] * gdelt_event_normalize.GDELT_EXPORT_COLUMNS
    values[0] = "123456789"
    values[1] = "20240501"
    values[5] = "USAGOV"
    values[6] = "UNITED STATES"
    values[7] = "USA"
    values[12] = "GOV"
    values[15] = "IRNGOV"
    values[16] = "IRAN"
    values[17] = "IRN"
    values[22] = "GOV"
    values[25] = "1"
    values[26] = "190"
    values[27] = "190"
    values[28] = "19"
    values[29] = "4"
    values[30] = "-10.0"
    values[31] = "12"
    values[32] = "4"
    values[33] = "3"
    values[34] = "-2.5"
    values[53] = "IR"
    values[56] = "35.7000"
    values[57] = "51.4167"
    values[59] = "20240501121500"
    values[60] = "https://Example.com/news/story?id=1"
    return values


def test_parse_synthetic_61_column_export_row():
    row = gdelt_event_normalize.parse_gdelt_export_row(
        _synthetic_export_row(),
        file_stamp="20240501123000",
        raw_file="20240501123000.export.CSV.zip",
        file_member="20240501123000.export.CSV",
        row_number=7,
    )

    assert row["global_event_id"] == "123456789"
    assert row["event_date"] == "2024-05-01T00:00:00+00:00"
    assert row["available_at"] == "2024-05-01T12:15:00+00:00"
    assert row["date_added"] == "2024-05-01T12:15:00+00:00"
    assert row["file_stamp"] == "2024-05-01T12:30:00+00:00"
    assert row["source_hostname"] == "example.com"
    assert len(row["source_url_hash"]) == 64
    assert row["actor1_code"] == "USAGOV"
    assert row["actor1_country"] == "USA"
    assert row["actor2_code"] == "IRNGOV"
    assert row["actor2_country"] == "IRN"
    assert row["quad_class"] == 4
    assert row["goldstein_scale"] == -10.0
    assert row["num_mentions"] == 12.0
    assert row["action_geo_country"] == "IR"
    assert row["action_geo_lat"] == 35.7
    assert row["cameo_topic"] == "fight"
    assert row["cameo_impact"] == "material_conflict"
    assert row["cameo_direction"] == "risk_off"
    assert row["cameo_severity"] == 1.0

    raw_ref = json.loads(row["raw_ref_json"])
    assert raw_ref["raw_file"] == "20240501123000.export.CSV.zip"
    assert raw_ref["file_member"] == "20240501123000.export.CSV"
    assert raw_ref["row_number"] == 7
    assert raw_ref["column_count"] == 61


def test_available_at_prefers_dateadded_over_file_stamp():
    row = gdelt_event_normalize.parse_gdelt_export_row(
        _synthetic_export_row(),
        file_stamp="20240501123000",
    )

    assert row["available_at"] == "2024-05-01T12:15:00+00:00"


def test_available_at_falls_back_to_file_stamp_when_dateadded_missing():
    values = _synthetic_export_row()
    values[gdelt_event_normalize.COL_DATE_ADDED] = ""

    row = gdelt_event_normalize.parse_gdelt_export_row(
        values,
        file_stamp="20240501123000",
    )

    assert row["available_at"] == "2024-05-01T12:30:00+00:00"
    assert row["date_added"] is None


def test_insert_gdelt_events_silver_uses_local_fallback_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(gdelt_event_normalize, "gdelt_gkg_normalized_schema", None)
    row = gdelt_event_normalize.parse_gdelt_export_row(
        _synthetic_export_row(),
        file_stamp="20240501123000",
    )
    db_path = tmp_path / "gdelt_context.sqlite"

    with gdelt_event_normalize.connect_db(db_path) as conn:
        gdelt_event_normalize.ensure_gdelt_events_silver(conn)
        inserted = gdelt_event_normalize.insert_gdelt_events_silver(conn, [row])
        saved = conn.execute(
            "SELECT global_event_id, available_at, source_hostname, cameo_topic FROM gdelt_events_silver"
        ).fetchone()

    assert inserted == 1
    assert dict(saved) == {
        "global_event_id": "123456789",
        "available_at": "2024-05-01T12:15:00+00:00",
        "source_hostname": "example.com",
        "cameo_topic": "fight",
    }


def test_insert_gdelt_events_silver_uses_shared_normalized_schema(tmp_path):
    row = gdelt_event_normalize.parse_gdelt_export_row(
        _synthetic_export_row(),
        file_stamp="20240501123000",
        raw_file="20240501123000.export.CSV.zip",
    )
    db_path = tmp_path / "gdelt_context.sqlite"

    gdelt_gkg_normalized_schema.init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        gdelt_event_normalize.insert_gdelt_events_silver(conn, [row])
        saved = conn.execute(
            """
            SELECT event_id, raw_file_id, event_time, available_at, topic, impact_channel,
                   direction, severity_proxy, raw_ref_json, parser_version
            FROM gdelt_events_silver
            """
        ).fetchone()
        raw_file = conn.execute(
            "SELECT file_kind, gdelt_stamp, status FROM gdelt_raw_files WHERE raw_file_id = ?",
            (saved["raw_file_id"],),
        ).fetchone()

    assert saved["event_id"] == "event:123456789"
    assert saved["event_time"] == "2024-05-01T12:15:00+00:00"
    assert saved["available_at"] == "2024-05-01T12:15:00+00:00"
    assert saved["topic"] == "fight"
    assert saved["impact_channel"] == "material_conflict"
    assert saved["direction"] == "risk_off"
    assert saved["severity_proxy"] == 1.0
    assert json.loads(saved["raw_ref_json"])["raw_file"] == "20240501123000.export.CSV.zip"
    assert saved["parser_version"] == "gdelt_event_normalize_v1"
    assert dict(raw_file) == {"file_kind": "export", "gdelt_stamp": "20240501123000", "status": "parsed"}


def test_safe_available_at_accepts_iso_file_stamp_fallback():
    assert gdelt_event_normalize.safe_available_at(None, "2024-05-01T12:30:00+00:00") == _stamp(
        "2024-05-01T12:30:00+00:00"
    )


def test_cli_date_window_filters_file_stamp_before_parsing_and_uses_exclusive_end(tmp_path, monkeypatch, capsys):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    db_path = tmp_path / "gdelt_context.sqlite"
    (raw_dir / "20220101095959.export.CSV.zip").write_text("outside before window", encoding="utf-8")
    (raw_dir / "20220101110000.export.CSV.zip").write_text("outside exclusive end", encoding="utf-8")
    zip_path = raw_dir / "20220101100000.export.CSV.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("20220101100000.export.CSV", "\t".join(_synthetic_export_row()))

    monkeypatch.setattr(
        "sys.argv",
        [
            "gdelt_event_normalize",
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

    assert gdelt_event_normalize.main() == 0
    summary = json.loads(capsys.readouterr().out)

    assert summary["files_planned"] == 1
    assert summary["files_read"] == 1
    assert summary["rows_parsed"] == 1
    assert summary["failures"] == []
    assert not db_path.exists()

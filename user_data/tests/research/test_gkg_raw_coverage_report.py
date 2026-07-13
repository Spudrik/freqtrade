from __future__ import annotations

from datetime import datetime, timezone
import csv
import io
import json
import sqlite3
import zipfile

from user_data.Custom_Launcher.research.context_features import gkg_raw_coverage_report as report


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def _write_zip(path, payload="a\tb\n"):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("member.gkg.csv", payload)


def _zip_bytes(payload="a\tb\n") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("member.gkg.csv", payload)
    return buffer.getvalue()


def _create_status_db(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE gkg_raw_download_status (
                stamp TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                file_url TEXT NOT NULL,
                local_path TEXT NOT NULL,
                status TEXT NOT NULL,
                http_status INTEGER,
                bytes_downloaded INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                attempted_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO gkg_raw_download_status(
                stamp, date, file_url, local_path, status, http_status, bytes_downloaded,
                error, attempted_at, updated_at
            )
            VALUES (?, ?, 'url', 'path', ?, ?, 0, NULL, 'now', 'now')
            """,
            [
                ("20220728001500", "2022-07-28T00:15:00+00:00", "http_404", 404),
                ("20220728003000", "2022-07-28T00:30:00+00:00", "download_error", None),
                ("20220728004500", "2022-07-28T00:45:00+00:00", "exists", None),
            ],
        )


def test_report_counts_raw_coverage_and_writes_json_and_csv(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "reports"
    raw_dir.mkdir()
    db_path = tmp_path / "missing.sqlite"
    first = raw_dir / "20220728000000.gkg.csv.zip"
    second = raw_dir / "20220728003000.gkg.csv.zip"
    _write_zip(first, "abc")
    _write_zip(second, "12345")
    (raw_dir / "ignore.zip").write_bytes(b"ignored")

    payload = report.build_coverage_report(
        db_path=db_path,
        raw_dir=raw_dir,
        start=_dt("2022-07-28T00:00:00+00:00"),
        end=_dt("2022-07-28T01:00:00+00:00"),
        output_dir=output_dir,
        tag="unit",
    )

    assert payload["expected_15m_files"] == 4
    assert payload["present_raw_zip_files"] == 2
    assert payload["percent_complete"] == 50.0
    assert payload["missing_raw_files"] == 2
    assert payload["missing_intervals"] == [
        {"start_stamp": "20220728001500", "end_stamp_exclusive": "20220728003000", "missing_files": 1},
        {"start_stamp": "20220728004500", "end_stamp_exclusive": "20220728010000", "missing_files": 1},
    ]
    assert payload["first_present_stamp"] == "20220728000000"
    assert payload["latest_present_stamp"] == "20220728003000"
    assert payload["estimated_remaining_files"] == 2
    assert not db_path.exists()

    json_path = output_dir / "gkg_raw_coverage_unit.json"
    csv_path = output_dir / "gkg_raw_coverage_unit.csv"
    assert json.loads(json_path.read_text(encoding="utf-8"))["missing_raw_files"] == 2
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["row_type"] for row in rows] == ["summary", "missing_interval", "missing_interval"]
    assert rows[1]["missing_interval_start"] == "20220728001500"


def test_report_reads_download_status_table_read_only(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "reports"
    db_path = tmp_path / "gdelt_context.sqlite"
    raw_dir.mkdir()
    _write_zip(raw_dir / "20220728000000.gkg.csv.zip")
    _create_status_db(db_path)

    payload = report.build_coverage_report(
        db_path=db_path,
        raw_dir=raw_dir,
        start=_dt("2022-07-28T00:00:00+00:00"),
        end=_dt("2022-07-28T01:00:00+00:00"),
        output_dir=output_dir,
        tag="status",
    )

    assert payload["db_status_available"] is True
    assert payload["terminal_404_count"] == 1
    assert payload["failed_retry_status_count"] == 1
    assert payload["status_counts"] == {"download_error": 1, "exists": 1, "http_404": 1}
    assert payload["estimated_remaining_files"] == 2
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM gkg_raw_download_status").fetchone()[0] == 3


def test_validate_sample_checks_deterministic_present_zip_sample(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "reports"
    raw_dir.mkdir()
    _write_zip(raw_dir / "20220728000000.gkg.csv.zip")
    (raw_dir / "20220728001500.gkg.csv.zip").write_bytes(b"not a zip")
    _write_zip(raw_dir / "20220728003000.gkg.csv.zip")
    _write_zip(raw_dir / "20220728004500.gkg.csv.zip")

    payload = report.build_coverage_report(
        db_path=tmp_path / "missing.sqlite",
        raw_dir=raw_dir,
        start=_dt("2022-07-28T00:00:00+00:00"),
        end=_dt("2022-07-28T01:00:00+00:00"),
        output_dir=output_dir,
        tag="validate",
        validate_sample=10,
    )

    assert payload["validation_sample_checked"] == 4
    assert payload["validation_corrupt_count"] == 1
    assert payload["validation_corrupt_files"][0]["path"].endswith("20220728001500.gkg.csv.zip")


def test_main_writes_report_paths_to_stdout(tmp_path, monkeypatch, capsys):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "reports"
    raw_dir.mkdir()
    (raw_dir / "20220728000000.gkg.csv.zip").write_bytes(_zip_bytes())

    monkeypatch.setattr(
        "sys.argv",
        [
            "gkg_raw_coverage_report",
            "--db",
            str(tmp_path / "missing.sqlite"),
            "--raw-dir",
            str(raw_dir),
            "--start",
            "2022-07-28T00:00:00+00:00",
            "--end",
            "2022-07-28T00:30:00+00:00",
            "--output-dir",
            str(output_dir),
            "--tag",
            "cli",
            "--validate-sample",
            "1",
        ],
    )

    assert report.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["expected_15m_files"] == 2
    assert payload["validation_sample_checked"] == 1
    assert payload["json_path"].endswith("gkg_raw_coverage_cli.json")
    assert (output_dir / "gkg_raw_coverage_cli.json").exists()
    assert (output_dir / "gkg_raw_coverage_cli.csv").exists()

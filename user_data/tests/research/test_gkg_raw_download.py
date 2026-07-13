from __future__ import annotations

from datetime import datetime, timezone
import io
import json
import urllib.error
import zipfile

from user_data.Custom_Launcher.research.context_features import gkg_raw_download


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def _zip_bytes(payload: str = "a\tb\n") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("member.gkg.csv", payload)
    return buffer.getvalue()


def test_dry_run_counts_missing_raw_files_without_creating_db(tmp_path, monkeypatch, capsys):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    db_path = tmp_path / "gdelt_context.sqlite"
    (raw_dir / "20220728000000.gkg.csv.zip").write_bytes(_zip_bytes())

    monkeypatch.setattr(
        "sys.argv",
        [
            "gkg_raw_download",
            "--start",
            "2022-07-28T00:00:00+00:00",
            "--end",
            "2022-07-28T01:00:00+00:00",
            "--raw-dir",
            str(raw_dir),
            "--db",
            str(db_path),
            "--dry-run",
        ],
    )

    assert gkg_raw_download.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["files_planned"] == 3
    assert payload["dry_run"] is True
    assert not db_path.exists()


def test_next_missing_raw_window_skips_existing_and_terminal_404(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    db_path = tmp_path / "gdelt_context.sqlite"
    (raw_dir / "20220728000000.gkg.csv.zip").write_bytes(_zip_bytes())
    gkg_raw_download.init_db(db_path)
    with gkg_raw_download.connect_db(db_path) as conn:
        gkg_raw_download._write_status_rows(
            conn,
            [
                gkg_raw_download._status_row(
                    _dt("2022-07-28T00:15:00+00:00"),
                    "url",
                    raw_dir / "20220728001500.gkg.csv.zip",
                    "http_404",
                    http_status=404,
                    error="HTTP 404",
                    attempted_at="2022-07-28T00:16:00+00:00",
                )
            ],
        )

    assert gkg_raw_download._next_missing_raw_window(
        db_path=db_path,
        raw_dir=raw_dir,
        start=_dt("2022-07-28T00:00:00+00:00"),
        end=_dt("2022-07-29T00:00:00+00:00"),
        chunk_days=1,
        interval_minutes=15,
        validate_existing=False,
    ) == (_dt("2022-07-28T00:30:00+00:00"), _dt("2022-07-29T00:00:00+00:00"))


def test_next_missing_raw_window_can_defer_recent_retry_failures(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    db_path = tmp_path / "gdelt_context.sqlite"
    gkg_raw_download.init_db(db_path)
    with gkg_raw_download.connect_db(db_path) as conn:
        gkg_raw_download._write_status_rows(
            conn,
            [
                gkg_raw_download._status_row(
                    _dt("2022-07-28T00:00:00+00:00"),
                    "url",
                    raw_dir / "20220728000000.gkg.csv.zip",
                    "download_error",
                    error="temporary failure",
                    attempted_at=datetime.now(timezone.utc).isoformat(),
                )
            ],
        )

    assert gkg_raw_download._next_missing_raw_window(
        db_path=db_path,
        raw_dir=raw_dir,
        start=_dt("2022-07-28T00:00:00+00:00"),
        end=_dt("2022-07-28T01:00:00+00:00"),
        chunk_days=1,
        interval_minutes=15,
        validate_existing=False,
        defer_retry_failures_hours=72,
    ) == (_dt("2022-07-28T00:15:00+00:00"), _dt("2022-07-28T01:00:00+00:00"))


def test_download_one_writes_valid_zip_atomically(tmp_path, monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return _zip_bytes("ok\n")

    monkeypatch.setattr(gkg_raw_download.urllib.request, "urlopen", lambda request, timeout: FakeResponse())

    row = gkg_raw_download.download_one(_dt("2022-07-28T00:00:00+00:00"), tmp_path)

    assert row["status"] == "downloaded"
    assert row["bytes_downloaded"] > 0
    assert (tmp_path / "20220728000000.gkg.csv.zip").exists()
    assert not (tmp_path / "20220728000000.gkg.csv.zip.tmp").exists()


def test_download_one_records_404_without_temp_file(tmp_path, monkeypatch):
    def raise_404(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(gkg_raw_download.urllib.request, "urlopen", raise_404)

    row = gkg_raw_download.download_one(_dt("2022-07-28T00:00:00+00:00"), tmp_path)

    assert row["status"] == "http_404"
    assert row["http_status"] == 404
    assert not (tmp_path / "20220728000000.gkg.csv.zip").exists()
    assert not (tmp_path / "20220728000000.gkg.csv.zip.tmp").exists()


def test_main_flushes_completed_status_before_later_worker_failure(tmp_path, monkeypatch, capsys):
    raw_dir = tmp_path / "raw"
    db_path = tmp_path / "gdelt_context.sqlite"

    def fake_download_one(stamp_time, raw_dir_arg, *, validate_existing=False):
        stamp = stamp_time.strftime("%Y%m%d%H%M%S")
        if stamp == "20220728000000":
            return gkg_raw_download._status_row(
                stamp_time,
                gkg_raw_download.GKG_URL_TEMPLATE.format(stamp=stamp),
                raw_dir_arg / f"{stamp}.gkg.csv.zip",
                "downloaded",
                bytes_downloaded=123,
                attempted_at="2022-07-28T00:00:01+00:00",
            )
        raise RuntimeError("late failure")

    monkeypatch.setattr(gkg_raw_download, "download_one", fake_download_one)
    monkeypatch.setattr(
        "sys.argv",
        [
            "gkg_raw_download",
            "--start",
            "2022-07-28T00:00:00+00:00",
            "--end",
            "2022-07-28T00:30:00+00:00",
            "--raw-dir",
            str(raw_dir),
            "--db",
            str(db_path),
            "--max-workers",
            "1",
            "--flush-every",
            "1",
        ],
    )

    assert gkg_raw_download.main() == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "retry_needed"
    with gkg_raw_download.connect_db(db_path) as conn:
        rows = conn.execute(
            "SELECT stamp, status, bytes_downloaded, error FROM gkg_raw_download_status ORDER BY stamp"
        ).fetchall()
        raw_rows = conn.execute(
            "SELECT file_kind, gdelt_stamp, status FROM gdelt_raw_files ORDER BY gdelt_stamp"
        ).fetchall()

    assert [(row["stamp"], row["status"]) for row in rows] == [
        ("20220728000000", "downloaded"),
        ("20220728001500", "download_error"),
    ]
    assert rows[0]["bytes_downloaded"] == 123
    assert "late failure" in rows[1]["error"]

    assert [(row["file_kind"], row["gdelt_stamp"], row["status"]) for row in raw_rows] == [
        ("gkg", "20220728000000", "downloaded"),
        ("gkg", "20220728001500", "download_error_retryable"),
    ]


def test_main_skips_existing_files_and_terminal_404s(tmp_path, monkeypatch, capsys):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    db_path = tmp_path / "gdelt_context.sqlite"
    (raw_dir / "20220728000000.gkg.csv.zip").write_bytes(_zip_bytes())
    gkg_raw_download.init_db(db_path)
    with gkg_raw_download.connect_db(db_path) as conn:
        gkg_raw_download._write_status_rows(
            conn,
            [
                gkg_raw_download._status_row(
                    _dt("2022-07-28T00:15:00+00:00"),
                    gkg_raw_download.GKG_URL_TEMPLATE.format(stamp="20220728001500"),
                    raw_dir / "20220728001500.gkg.csv.zip",
                    "http_404",
                    http_status=404,
                    error="HTTP 404",
                    attempted_at="2022-07-28T00:15:01+00:00",
                )
            ],
        )

    opened_urls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return _zip_bytes("ok\n")

    def fake_urlopen(request, timeout):
        opened_urls.append(request.full_url)
        return FakeResponse()

    monkeypatch.setattr(gkg_raw_download.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        "sys.argv",
        [
            "gkg_raw_download",
            "--start",
            "2022-07-28T00:00:00+00:00",
            "--end",
            "2022-07-28T00:45:00+00:00",
            "--raw-dir",
            str(raw_dir),
            "--db",
            str(db_path),
            "--max-workers",
            "1",
        ],
    )

    assert gkg_raw_download.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["files_planned"] == 1
    assert opened_urls == [gkg_raw_download.GKG_URL_TEMPLATE.format(stamp="20220728003000")]
    with gkg_raw_download.connect_db(db_path) as conn:
        rows = conn.execute("SELECT stamp, status FROM gkg_raw_download_status ORDER BY stamp").fetchall()

    assert [(row["stamp"], row["status"]) for row in rows] == [
        ("20220728001500", "http_404"),
        ("20220728003000", "downloaded"),
    ]


def test_download_one_with_validate_existing_moves_corrupt_file_and_redownloads(tmp_path, monkeypatch):
    raw_path = tmp_path / "20220728000000.gkg.csv.zip"
    raw_path.write_bytes(b"not a zip")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return _zip_bytes("fresh\n")

    monkeypatch.setattr(gkg_raw_download.urllib.request, "urlopen", lambda request, timeout: FakeResponse())

    row = gkg_raw_download.download_one(
        _dt("2022-07-28T00:00:00+00:00"),
        tmp_path,
        validate_existing=True,
    )

    assert row["status"] == "downloaded"
    assert raw_path.exists()
    assert (tmp_path / "20220728000000.gkg.csv.zip.corrupt").exists()
    with zipfile.ZipFile(raw_path) as archive:
        assert archive.testzip() is None

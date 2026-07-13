from datetime import datetime, timezone

import pytest

from user_data.Custom_Launcher.research.context_features import gkg_backfill


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def test_successful_file_times_excludes_failed_rows(tmp_path):
    db_path = tmp_path / "gdelt_context.sqlite"
    gkg_backfill.init_db(db_path)
    ok_row = gkg_backfill._base_row(_dt("2022-07-28T00:45:00+00:00"), "ok")
    failed_row = gkg_backfill._error_row(
        _dt("2022-07-28T01:00:00+00:00"),
        "failed",
        "connection refused",
    )

    with gkg_backfill.connect_db(db_path) as conn:
        gkg_backfill._write_rows(conn, [ok_row, failed_row])

    start = _dt("2022-07-28T00:00:00+00:00")
    end = _dt("2022-07-28T02:00:00+00:00")

    assert gkg_backfill._successful_file_times(db_path, start, end) == {
        "2022-07-28T00:45:00+00:00",
    }
    assert gkg_backfill._failed_file_times(db_path, start, end) == {
        "2022-07-28T01:00:00+00:00",
    }


def test_next_missing_window_retries_failed_rows(tmp_path):
    db_path = tmp_path / "gdelt_context.sqlite"
    gkg_backfill.init_db(db_path)
    ok_row = gkg_backfill._base_row(_dt("2022-07-28T00:45:00+00:00"), "ok")
    failed_row = gkg_backfill._error_row(
        _dt("2022-07-28T01:00:00+00:00"),
        "failed",
        "connection refused",
    )

    with gkg_backfill.connect_db(db_path) as conn:
        gkg_backfill._write_rows(conn, [ok_row, failed_row])

    start = _dt("2022-07-28T00:45:00+00:00")
    end = _dt("2022-07-30T00:45:00+00:00")

    assert gkg_backfill._next_missing_window(db_path, start, end, 14, 15) == (
        _dt("2022-07-28T01:00:00+00:00"),
        end,
    )


def test_next_missing_window_skips_terminal_404_rows(tmp_path):
    db_path = tmp_path / "gdelt_context.sqlite"
    gkg_backfill.init_db(db_path)
    rows = [
        gkg_backfill._base_row(_dt("2022-07-28T00:00:00+00:00"), "ok"),
        gkg_backfill._error_row(_dt("2022-07-28T00:15:00+00:00"), "missing", "HTTP 404"),
        gkg_backfill._base_row(_dt("2022-07-28T00:30:00+00:00"), "ok"),
        gkg_backfill._error_row(_dt("2022-07-28T00:45:00+00:00"), "failed", "connection refused"),
    ]
    with gkg_backfill.connect_db(db_path) as conn:
        gkg_backfill._write_rows(conn, rows)

    start = _dt("2022-07-28T00:00:00+00:00")
    end = _dt("2022-07-28T01:00:00+00:00")

    assert gkg_backfill._terminal_failure_file_times(db_path, start, end) == {
        "2022-07-28T00:15:00+00:00",
    }
    assert gkg_backfill._next_missing_window(db_path, start, end, 14, 15) == (
        _dt("2022-07-28T00:45:00+00:00"),
        end,
    )


def test_next_missing_window_advances_after_terminal_404_is_only_gap(tmp_path):
    db_path = tmp_path / "gdelt_context.sqlite"
    gkg_backfill.init_db(db_path)
    rows = [
        gkg_backfill._base_row(_dt("2022-07-28T00:00:00+00:00"), "ok"),
        gkg_backfill._error_row(_dt("2022-07-28T00:15:00+00:00"), "missing", "HTTP 404"),
        gkg_backfill._base_row(_dt("2022-07-28T00:30:00+00:00"), "ok"),
        gkg_backfill._base_row(_dt("2022-07-28T00:45:00+00:00"), "ok"),
    ]
    with gkg_backfill.connect_db(db_path) as conn:
        gkg_backfill._write_rows(conn, rows)

    start = _dt("2022-07-28T00:00:00+00:00")
    end = _dt("2022-07-28T01:00:00+00:00")

    assert gkg_backfill._next_missing_window(db_path, start, end, 14, 15) is None


def test_raise_csv_field_size_limit_handles_large_gkg_records():
    gkg_backfill._raise_csv_field_size_limit()

    assert gkg_backfill.csv.field_size_limit() >= gkg_backfill.CSV_FIELD_SIZE_LIMIT


def test_failed_retry_filter_runs_before_limit(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "gdelt_context.sqlite"
    gkg_backfill.init_db(db_path)
    rows = [
        gkg_backfill._base_row(_dt("2022-07-28T00:00:00+00:00"), "ok"),
        gkg_backfill._base_row(_dt("2022-07-28T00:15:00+00:00"), "ok"),
        gkg_backfill._error_row(_dt("2022-07-28T00:30:00+00:00"), "failed", "connection refused"),
        gkg_backfill._error_row(_dt("2022-07-28T00:45:00+00:00"), "failed", "connection refused"),
    ]
    with gkg_backfill.connect_db(db_path) as conn:
        gkg_backfill._write_rows(conn, rows)

    monkeypatch.setattr(
        "sys.argv",
        [
            "gkg_backfill",
            "--start",
            "2022-07-28T00:00:00+00:00",
            "--end",
            "2022-07-28T01:00:00+00:00",
            "--db",
            str(db_path),
            "--retry-failed-only",
            "--limit-files",
            "1",
            "--dry-run",
        ],
    )

    assert gkg_backfill.main() == 0
    captured = capsys.readouterr()

    assert '"files_planned": 1' in captured.out
    assert '"start": "2022-07-28T00:00:00+00:00"' in captured.out


def test_skip_existing_dry_run_counts_only_missing_after_success_filter(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "gdelt_context.sqlite"
    gkg_backfill.init_db(db_path)
    rows = [
        gkg_backfill._base_row(_dt("2022-07-28T00:00:00+00:00"), "ok"),
        gkg_backfill._base_row(_dt("2022-07-28T00:15:00+00:00"), "ok"),
    ]
    with gkg_backfill.connect_db(db_path) as conn:
        gkg_backfill._write_rows(conn, rows)

    monkeypatch.setattr(
        "sys.argv",
        [
            "gkg_backfill",
            "--start",
            "2022-07-28T00:00:00+00:00",
            "--end",
            "2022-07-28T01:00:00+00:00",
            "--db",
            str(db_path),
            "--skip-existing",
            "--dry-run",
        ],
    )

    assert gkg_backfill.main() == 0
    captured = capsys.readouterr()

    assert '"files_planned": 2' in captured.out


def test_retry_failed_only_rejects_skip_existing(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "gkg_backfill",
            "--start",
            "2022-07-28",
            "--end",
            "2022-07-29",
            "--retry-failed-only",
            "--skip-existing",
        ],
    )

    with pytest.raises(SystemExit, match="mutually exclusive"):
        gkg_backfill.main()

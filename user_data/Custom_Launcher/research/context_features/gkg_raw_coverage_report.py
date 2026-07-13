from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
import argparse
import csv
import json
import re
import sqlite3
import zipfile

from .builder import default_paths
from .gkg_raw_download import DEFAULT_RAW_DIR


GKG_NAME_RE = re.compile(r"^(?P<stamp>\d{14})\.gkg\.csv\.zip$", re.IGNORECASE)
INTERVAL_MINUTES = 15
TERMINAL_404_STATUSES = {"http_404"}
SUCCESS_STATUSES = {"downloaded", "exists"}
RETRY_STATUS_MARKERS = ("error", "fail", "retry", "timeout")


def main() -> int:
    parser = argparse.ArgumentParser(description="Report read-only GDELT GKG raw ZIP coverage and download progress.")
    parser.add_argument("--app-dir", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=None, help="Optional gdelt_context.sqlite with gkg_raw_download_status.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--start", required=True, help="Inclusive UTC start date/time.")
    parser.add_argument("--end", required=True, help="Exclusive UTC end date/time.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--validate-sample", type=int, default=0)
    args = parser.parse_args()

    if args.validate_sample < 0:
        raise SystemExit("--validate-sample must be non-negative")

    paths = default_paths(args.app_dir)
    db_path = args.db or paths.gdelt_db
    start = _parse_utc(args.start)
    end = _parse_utc(args.end)
    if end <= start:
        raise SystemExit("--end must be after --start")

    report = build_coverage_report(
        db_path=db_path,
        raw_dir=args.raw_dir,
        start=start,
        end=end,
        output_dir=args.output_dir,
        tag=args.tag,
        validate_sample=args.validate_sample,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build_coverage_report(
    *,
    db_path: Path,
    raw_dir: Path,
    start: datetime,
    end: datetime,
    output_dir: Path,
    tag: str | None = None,
    validate_sample: int = 0,
) -> dict[str, Any]:
    expected_stamps = [stamp.strftime("%Y%m%d%H%M%S") for stamp in _iter_times(start, end)]
    raw_files = _scan_raw_files(raw_dir)
    present_rows = [
        {
            "stamp": stamp,
            "path": str(raw_files[stamp]["path"]),
            "size_bytes": raw_files[stamp]["size_bytes"],
            "write_time_utc": raw_files[stamp]["write_time_utc"],
        }
        for stamp in expected_stamps
        if stamp in raw_files and raw_files[stamp]["size_bytes"] > 0
    ]
    present_stamps = {row["stamp"] for row in present_rows}
    missing_stamps = [stamp for stamp in expected_stamps if stamp not in present_stamps]
    status_summary = _read_download_status_summary(db_path, expected_stamps)
    validation_summary = _validate_sample(present_rows, validate_sample)

    expected_count = len(expected_stamps)
    present_count = len(present_rows)
    average_size = round(sum(row["size_bytes"] for row in present_rows) / present_count, 2) if present_count else 0.0
    remaining_files = max(expected_count - present_count - status_summary["terminal_404_count"], 0)
    tag_text = _clean_tag(tag) or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"gkg_raw_coverage_{tag_text}.json"
    csv_path = output_dir / f"gkg_raw_coverage_{tag_text}.csv"

    report: dict[str, Any] = {
        "report_type": "gkg_raw_coverage",
        "report_tag": tag_text,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path),
        "db_status_available": status_summary["db_status_available"],
        "raw_dir": str(raw_dir),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "interval_minutes": INTERVAL_MINUTES,
        "expected_15m_files": expected_count,
        "present_raw_zip_files": present_count,
        "percent_complete": round((present_count / expected_count) * 100, 4) if expected_count else 0.0,
        "missing_raw_files": len(missing_stamps),
        "missing_intervals": _segment_missing_stamps(missing_stamps),
        "terminal_404_count": status_summary["terminal_404_count"],
        "failed_retry_status_count": status_summary["failed_retry_status_count"],
        "status_counts": status_summary["status_counts"],
        "first_present_stamp": min(present_stamps) if present_stamps else None,
        "latest_present_stamp": max(present_stamps) if present_stamps else None,
        "newest_write_time_utc": max((row["write_time_utc"] for row in present_rows), default=None),
        "average_present_size_bytes": average_size,
        "estimated_remaining_files": remaining_files,
        "estimated_remaining_bytes": int(round(average_size * remaining_files)),
        "validation_sample_requested": validate_sample,
        "validation_sample_checked": validation_summary["validation_sample_checked"],
        "validation_corrupt_count": validation_summary["validation_corrupt_count"],
        "validation_corrupt_files": validation_summary["validation_corrupt_files"],
        "json_path": str(json_path),
        "csv_path": str(csv_path),
    }

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv_report(csv_path, report)
    return report


def _scan_raw_files(raw_dir: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    if not raw_dir.exists():
        return files
    for path in raw_dir.iterdir():
        if not path.is_file():
            continue
        match = GKG_NAME_RE.fullmatch(path.name)
        if not match:
            continue
        stat = path.stat()
        files[match.group("stamp")] = {
            "path": path,
            "size_bytes": stat.st_size,
            "write_time_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        }
    return files


def _read_download_status_summary(db_path: Path, expected_stamps: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "db_status_available": False,
        "terminal_404_count": 0,
        "failed_retry_status_count": 0,
        "status_counts": {},
    }
    if not expected_stamps or not db_path.exists():
        return summary

    with _connect_read_only(db_path) as conn:
        if not _table_exists(conn, "gkg_raw_download_status"):
            return summary
        rows = conn.execute(
            """
            SELECT stamp, status, http_status
            FROM gkg_raw_download_status
            WHERE stamp >= ? AND stamp < ?
            """,
            (expected_stamps[0], _next_stamp_text(expected_stamps[-1])),
        ).fetchall()

    expected = set(expected_stamps)
    status_counts: dict[str, int] = {}
    terminal_404_count = 0
    failed_retry_status_count = 0
    for row in rows:
        stamp = str(row["stamp"])
        if stamp not in expected:
            continue
        status = str(row["status"] or "")
        status_counts[status] = status_counts.get(status, 0) + 1
        if status in TERMINAL_404_STATUSES or row["http_status"] == 404:
            terminal_404_count += 1
        elif status not in SUCCESS_STATUSES and any(marker in status.lower() for marker in RETRY_STATUS_MARKERS):
            failed_retry_status_count += 1

    summary.update(
        {
            "db_status_available": True,
            "terminal_404_count": terminal_404_count,
            "failed_retry_status_count": failed_retry_status_count,
            "status_counts": dict(sorted(status_counts.items())),
        }
    )
    return summary


def _connect_read_only(db_path: Path) -> sqlite3.Connection:
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _segment_missing_stamps(stamps: list[str]) -> list[dict[str, Any]]:
    if not stamps:
        return []
    segments: list[dict[str, Any]] = []
    segment_start = stamps[0]
    previous = stamps[0]
    count = 1
    for stamp in stamps[1:]:
        if stamp == _next_stamp_text(previous):
            previous = stamp
            count += 1
            continue
        segments.append(_missing_segment(segment_start, previous, count))
        segment_start = previous = stamp
        count = 1
    segments.append(_missing_segment(segment_start, previous, count))
    return segments


def _missing_segment(start_stamp: str, last_stamp: str, count: int) -> dict[str, Any]:
    segment_end = _next_stamp_text(last_stamp)
    return {
        "start_stamp": start_stamp,
        "end_stamp_exclusive": segment_end,
        "missing_files": count,
    }


def _validate_sample(present_rows: list[dict[str, Any]], validate_sample: int) -> dict[str, Any]:
    paths = [Path(row["path"]) for row in _deterministic_sample(present_rows, validate_sample)]
    corrupt_files: list[dict[str, str]] = []
    for path in paths:
        error = _zip_validation_error(path)
        if error is not None:
            corrupt_files.append({"path": str(path), "error": error})
    return {
        "validation_sample_checked": len(paths),
        "validation_corrupt_count": len(corrupt_files),
        "validation_corrupt_files": corrupt_files,
    }


def _deterministic_sample(rows: list[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    if sample_size <= 0 or not rows:
        return []
    ordered = sorted(rows, key=lambda row: row["stamp"])
    if sample_size >= len(ordered):
        return ordered
    if sample_size == 1:
        return [ordered[len(ordered) // 2]]
    indices = {
        round(index * (len(ordered) - 1) / (sample_size - 1))
        for index in range(sample_size)
    }
    return [ordered[index] for index in sorted(indices)]


def _zip_validation_error(path: Path) -> str | None:
    try:
        with zipfile.ZipFile(path) as archive:
            members = [info for info in archive.infolist() if not info.is_dir()]
            if not members:
                return "empty zip"
            bad_member = archive.testzip()
            if bad_member:
                return f"corrupt zip member: {bad_member}"
    except zipfile.BadZipFile as exc:
        return f"bad zip: {exc}"
    except OSError as exc:
        return f"read error: {exc}"
    return None


def _write_csv_report(csv_path: Path, report: dict[str, Any]) -> None:
    fieldnames = [
        "row_type",
        "report_tag",
        "start",
        "end",
        "raw_dir",
        "db_path",
        "expected_15m_files",
        "present_raw_zip_files",
        "percent_complete",
        "missing_raw_files",
        "missing_interval_start",
        "missing_interval_end_exclusive",
        "missing_interval_files",
        "terminal_404_count",
        "failed_retry_status_count",
        "first_present_stamp",
        "latest_present_stamp",
        "newest_write_time_utc",
        "average_present_size_bytes",
        "estimated_remaining_files",
        "estimated_remaining_bytes",
        "validation_sample_requested",
        "validation_sample_checked",
        "validation_corrupt_count",
    ]
    summary_row = _csv_base_row(report)
    rows = [dict(summary_row, row_type="summary")]
    for interval in report["missing_intervals"]:
        rows.append(
            dict(
                summary_row,
                row_type="missing_interval",
                missing_interval_start=interval["start_stamp"],
                missing_interval_end_exclusive=interval["end_stamp_exclusive"],
                missing_interval_files=interval["missing_files"],
            )
        )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _csv_base_row(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_tag": report["report_tag"],
        "start": report["start"],
        "end": report["end"],
        "raw_dir": report["raw_dir"],
        "db_path": report["db_path"],
        "expected_15m_files": report["expected_15m_files"],
        "present_raw_zip_files": report["present_raw_zip_files"],
        "percent_complete": report["percent_complete"],
        "missing_raw_files": report["missing_raw_files"],
        "missing_interval_start": "",
        "missing_interval_end_exclusive": "",
        "missing_interval_files": "",
        "terminal_404_count": report["terminal_404_count"],
        "failed_retry_status_count": report["failed_retry_status_count"],
        "first_present_stamp": report["first_present_stamp"] or "",
        "latest_present_stamp": report["latest_present_stamp"] or "",
        "newest_write_time_utc": report["newest_write_time_utc"] or "",
        "average_present_size_bytes": report["average_present_size_bytes"],
        "estimated_remaining_files": report["estimated_remaining_files"],
        "estimated_remaining_bytes": report["estimated_remaining_bytes"],
        "validation_sample_requested": report["validation_sample_requested"],
        "validation_sample_checked": report["validation_sample_checked"],
        "validation_corrupt_count": report["validation_corrupt_count"],
    }


def _iter_times(start: datetime, end: datetime) -> Iterable[datetime]:
    current = start
    step = timedelta(minutes=INTERVAL_MINUTES)
    while current < end:
        yield current
        current += step


def _parse_utc(value: str) -> datetime:
    clean = value.strip().replace("Z", "+00:00")
    if len(clean) == 10:
        clean = f"{clean}T00:00:00+00:00"
    parsed = datetime.fromisoformat(clean)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc).replace(second=0, microsecond=0)
    minute = (parsed.minute // INTERVAL_MINUTES) * INTERVAL_MINUTES
    return parsed.replace(minute=minute)


def _next_stamp_text(stamp: str) -> str:
    parsed = datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return (parsed + timedelta(minutes=INTERVAL_MINUTES)).strftime("%Y%m%d%H%M%S")


def _clean_tag(tag: str | None) -> str | None:
    if tag is None:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag.strip())
    return cleaned or None


if __name__ == "__main__":
    raise SystemExit(main())

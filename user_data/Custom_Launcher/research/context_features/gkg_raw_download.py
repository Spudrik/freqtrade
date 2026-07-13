from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
import argparse
import io
import json
import sqlite3
import urllib.error
import urllib.request
import zipfile

from .builder import default_paths
from . import gdelt_gkg_normalized_schema as normalized_schema


GKG_URL_TEMPLATE = "http://data.gdeltproject.org/gdeltv2/{stamp}.gkg.csv.zip"
DEFAULT_RAW_DIR = Path("D:/FreqTradeStuffLargeData/research_news_data/gdelt/raw/gkg")
SCHEMA_VERSION = "1"
TERMINAL_STATUSES = {"http_404"}


@dataclass
class RawDownloadSummary:
    db_path: str
    raw_dir: str
    start: str
    end: str
    dry_run: bool
    files_planned: int
    files_downloaded: int = 0
    files_existing: int = 0
    files_terminal: int = 0
    retry_failures_deferred: int = 0
    files_failed: int = 0
    bytes_downloaded: int = 0
    first_file_time: str | None = None
    last_file_time: str | None = None
    failures: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failures"] = self.failures or []
        return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Download historical GDELT GKG raw ZIPs only; no CSV parsing.")
    parser.add_argument("--app-dir", type=Path, default=None)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--interval-minutes", type=int, default=15)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--next-missing-chunk-days", type=int, default=0)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-existing", action="store_true")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument(
        "--defer-retry-failures-hours",
        type=int,
        default=0,
        help="Temporarily skip retryable failure statuses newer than this many hours so the forward sweep can keep moving.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=1,
        help="Persist completed download status rows after this many finished futures; default writes each result immediately.",
    )
    args = parser.parse_args()

    if args.interval_minutes <= 0:
        raise SystemExit("--interval-minutes must be positive")
    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be positive")
    if args.limit_files is not None and args.limit_files < 0:
        raise SystemExit("--limit-files must be non-negative")
    if args.flush_every <= 0:
        raise SystemExit("--flush-every must be positive")
    if args.defer_retry_failures_hours < 0:
        raise SystemExit("--defer-retry-failures-hours must be non-negative")

    paths = default_paths(args.app_dir)
    db_path = args.db or paths.gdelt_db
    raw_dir = args.raw_dir or DEFAULT_RAW_DIR
    start = _parse_utc(args.start, args.interval_minutes)
    end = _parse_utc(args.end, args.interval_minutes)
    if end <= start:
        raise SystemExit("--end must be after --start")

    if args.next_missing_chunk_days > 0:
        if not args.dry_run:
            init_db(db_path)
        window = _next_missing_raw_window(
            db_path=db_path,
            raw_dir=raw_dir,
            start=start,
            end=end,
            chunk_days=args.next_missing_chunk_days,
            interval_minutes=args.interval_minutes,
            validate_existing=args.validate_existing,
            defer_retry_failures_hours=args.defer_retry_failures_hours,
        )
        if window is None:
            summary = RawDownloadSummary(str(db_path), str(raw_dir), start.isoformat(), end.isoformat(), bool(args.dry_run), 0)
            print(json.dumps({**summary.to_dict(), "status": "complete"}, indent=2, sort_keys=True))
            return 0
        start, end = window

    stamps = [
        stamp
        for stamp in _iter_times(start, end, args.interval_minutes)
        if not _raw_file_complete(raw_dir, stamp, args.validate_existing)
    ]
    if db_path.exists():
        terminal = _terminal_status_times(db_path, start, end)
        deferred_retry = _deferred_retry_status_times(db_path, start, end, args.defer_retry_failures_hours)
        skip_statuses = terminal | deferred_retry
        stamps = [stamp for stamp in stamps if stamp.strftime("%Y%m%d%H%M%S") not in skip_statuses]
    if args.limit_files is not None:
        stamps = stamps[: args.limit_files]

    summary = RawDownloadSummary(
        db_path=str(db_path),
        raw_dir=str(raw_dir),
        start=start.isoformat(),
        end=end.isoformat(),
        dry_run=bool(args.dry_run),
        files_planned=len(stamps),
        retry_failures_deferred=len(deferred_retry) if db_path.exists() else 0,
    )
    if args.dry_run:
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        return 0

    init_db(db_path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    pending_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    with connect_db(db_path) as conn:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {pool.submit(download_one, stamp, raw_dir, validate_existing=args.validate_existing): stamp for stamp in stamps}
            completed = 0
            for future in as_completed(futures):
                stamp = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = _worker_exception_row(stamp, raw_dir, exc)
                results.append(row)
                pending_rows.append(row)
                if len(pending_rows) >= args.flush_every:
                    _flush_status_rows(conn, pending_rows)
                    pending_rows.clear()

                if row["status"] in TERMINAL_STATUSES:
                    summary.files_terminal += 1
                elif row["status"] == "downloaded":
                    summary.files_downloaded += 1
                    summary.bytes_downloaded += int(row.get("bytes_downloaded") or 0)
                elif row["status"] == "exists":
                    summary.files_existing += 1
                else:
                    summary.files_failed += 1
                    failures.append(f"{row['date']}: {row.get('error') or row['status']}")
                completed += 1
                if args.progress_every and completed % args.progress_every == 0:
                    print(
                        json.dumps(
                            {
                                "progress": completed,
                                "files_planned": len(stamps),
                                "files_downloaded": summary.files_downloaded,
                                "files_failed": summary.files_failed,
                                "files_terminal": summary.files_terminal,
                                "statuses_flushed": completed - len(pending_rows),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            _flush_status_rows(conn, pending_rows)

    good_times = [row["date"] for row in results if row["status"] in {"downloaded", "exists"}]
    summary.first_file_time = min(good_times, default=None)
    summary.last_file_time = max(good_times, default=None)
    summary.failures = failures[:20]
    exit_code = 0 if summary.files_failed == 0 else 2
    status = "complete" if exit_code == 0 else "retry_needed"
    retry_message = (
        "retryable failures were recorded; rerun the same range to skip downloaded files and terminal 404s"
        if exit_code
        else "no retry needed"
    )
    print(json.dumps({**summary.to_dict(), "exit_code": exit_code, "retry_message": retry_message, "status": status}, indent=2, sort_keys=True))
    return exit_code


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db(db_path: Path) -> None:
    with connect_db(db_path) as conn:
        normalized_schema.init_schema(conn)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS gkg_raw_download_status (
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
            );

            CREATE INDEX IF NOT EXISTS idx_gkg_raw_download_status_status
                ON gkg_raw_download_status(status);
            CREATE INDEX IF NOT EXISTS idx_gkg_raw_download_status_date
                ON gkg_raw_download_status(date);
            """
        )
        conn.execute("INSERT OR REPLACE INTO schema_meta(key, value) VALUES(?, ?)", ("gkg_raw_download_version", SCHEMA_VERSION))


def download_one(stamp_time: datetime, raw_dir: Path, *, validate_existing: bool = False) -> dict[str, Any]:
    stamp = stamp_time.strftime("%Y%m%d%H%M%S")
    url = GKG_URL_TEMPLATE.format(stamp=stamp)
    raw_path = raw_dir / f"{stamp}.gkg.csv.zip"
    attempted_at = datetime.now(timezone.utc).isoformat()
    if _raw_file_complete(raw_dir, stamp_time, validate_existing=validate_existing):
        return _status_row(stamp_time, url, raw_path, "exists", attempted_at=attempted_at)
    if raw_path.exists() and raw_path.stat().st_size > 0 and validate_existing:
        corrupt_path = raw_path.with_name(f"{raw_path.name}.corrupt")
        try:
            raw_path.replace(corrupt_path)
        except OSError:
            return _status_row(
                stamp_time,
                url,
                raw_path,
                "download_error",
                error="existing raw file failed ZIP validation and could not be moved aside",
                attempted_at=attempted_at,
            )

    temp_path = raw_path.with_name(f"{raw_path.name}.tmp")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "FreqTradeStuff-raw-gkg-download/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        _validate_zip_bytes(data)
        temp_path.write_bytes(data)
        temp_path.replace(raw_path)
        return _status_row(
            stamp_time,
            url,
            raw_path,
            "downloaded",
            bytes_downloaded=len(data),
            attempted_at=attempted_at,
        )
    except urllib.error.HTTPError as exc:
        _remove_temp(temp_path)
        status = "http_404" if exc.code == 404 else "http_error"
        return _status_row(stamp_time, url, raw_path, status, http_status=exc.code, error=f"HTTP {exc.code}", attempted_at=attempted_at)
    except Exception as exc:
        _remove_temp(temp_path)
        return _status_row(stamp_time, url, raw_path, "download_error", error=str(exc), attempted_at=attempted_at)


def _write_status_rows(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO gkg_raw_download_status(
            stamp, date, file_url, local_path, status, http_status, bytes_downloaded,
            error, attempted_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(stamp) DO UPDATE SET
            date=excluded.date,
            file_url=excluded.file_url,
            local_path=excluded.local_path,
            status=excluded.status,
            http_status=excluded.http_status,
            bytes_downloaded=excluded.bytes_downloaded,
            error=excluded.error,
            attempted_at=excluded.attempted_at,
            updated_at=excluded.updated_at
        """,
        [
            (
                row["stamp"],
                row["date"],
                row["file_url"],
                row["local_path"],
                row["status"],
                row.get("http_status"),
                row.get("bytes_downloaded") or 0,
                row.get("error"),
                row["attempted_at"],
                datetime.now(timezone.utc).isoformat(),
            )
            for row in rows
        ],
    )
    for row in rows:
        _upsert_raw_metadata(conn, row)


def _flush_status_rows(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        _write_status_rows(conn, rows)
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _worker_exception_row(stamp_time: datetime, raw_dir: Path, exc: Exception) -> dict[str, Any]:
    stamp = stamp_time.strftime("%Y%m%d%H%M%S")
    return _status_row(
        stamp_time,
        GKG_URL_TEMPLATE.format(stamp=stamp),
        raw_dir / f"{stamp}.gkg.csv.zip",
        "download_error",
        error=f"worker exception: {exc}",
        attempted_at=datetime.now(timezone.utc).isoformat(),
    )


def _upsert_raw_metadata(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    status_map = {
        "downloaded": "downloaded",
        "exists": "downloaded",
        "http_404": "terminal_404",
        "http_error": "download_error_retryable",
        "download_error": "download_error_retryable",
    }
    status = status_map.get(str(row["status"]), str(row["status"]))
    metadata = {
        "source": "gkg_raw_download",
        "download_status": row["status"],
        "http_status": row.get("http_status"),
        "source_url": row.get("file_url"),
    }
    compressed_bytes = row.get("bytes_downloaded")
    if not compressed_bytes:
        path = Path(str(row["local_path"]))
        compressed_bytes = path.stat().st_size if path.exists() else None
    normalized_schema.upsert_raw_file_metadata(
        conn,
        file_kind="gkg",
        gdelt_stamp=str(row["stamp"]),
        source_url=None,
        local_path=str(row["local_path"]),
        compressed_bytes=None if compressed_bytes is None else int(compressed_bytes),
        fetched_at=str(row["attempted_at"]),
        status=status,
        parse_error=row.get("error"),
        metadata=metadata,
    )


def _next_missing_raw_window(
    *,
    db_path: Path,
    raw_dir: Path,
    start: datetime,
    end: datetime,
    chunk_days: int,
    interval_minutes: int,
    validate_existing: bool,
    defer_retry_failures_hours: int = 0,
) -> tuple[datetime, datetime] | None:
    terminal = _terminal_status_times(db_path, start, end) if db_path.exists() else set()
    deferred_retry = _deferred_retry_status_times(db_path, start, end, defer_retry_failures_hours) if db_path.exists() else set()
    skip_statuses = terminal | deferred_retry
    missing_start: datetime | None = None
    for stamp in _iter_times(start, end, interval_minutes):
        stamp_text = stamp.strftime("%Y%m%d%H%M%S")
        if stamp_text in skip_statuses or _raw_file_complete(raw_dir, stamp, validate_existing):
            continue
        missing_start = stamp
        break
    if missing_start is None:
        return None
    chunk_end = min(missing_start + timedelta(days=max(1, chunk_days)), end)
    return missing_start, chunk_end


def _deferred_retry_status_times(db_path: Path, start: datetime, end: datetime, defer_hours: int) -> set[str]:
    if defer_hours <= 0 or not db_path.exists():
        return set()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=defer_hours)
    with connect_db(db_path) as conn:
        if not _table_exists(conn, "gkg_raw_download_status"):
            return set()
        rows = conn.execute(
            """
            SELECT stamp
            FROM gkg_raw_download_status
            WHERE date >= ? AND date < ?
              AND status IN ('http_error', 'download_error')
              AND updated_at >= ?
            """,
            (start.isoformat(), end.isoformat(), cutoff.isoformat()),
        ).fetchall()
    return {str(row["stamp"]) for row in rows}


def _terminal_status_times(db_path: Path, start: datetime, end: datetime) -> set[str]:
    if not db_path.exists():
        return set()
    with connect_db(db_path) as conn:
        if not _table_exists(conn, "gkg_raw_download_status"):
            return set()
        rows = conn.execute(
            """
            SELECT stamp
            FROM gkg_raw_download_status
            WHERE date >= ? AND date < ?
              AND status IN ('http_404')
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    return {str(row["stamp"]) for row in rows}


def _status_row(
    stamp_time: datetime,
    url: str,
    raw_path: Path,
    status: str,
    *,
    http_status: int | None = None,
    bytes_downloaded: int = 0,
    error: str | None = None,
    attempted_at: str,
) -> dict[str, Any]:
    return {
        "stamp": stamp_time.strftime("%Y%m%d%H%M%S"),
        "date": stamp_time.isoformat(),
        "file_url": url,
        "local_path": str(raw_path),
        "status": status,
        "http_status": http_status,
        "bytes_downloaded": bytes_downloaded,
        "error": error,
        "attempted_at": attempted_at,
    }


def _raw_file_complete(raw_dir: Path, stamp_time: datetime, validate_existing: bool) -> bool:
    path = raw_dir / f"{stamp_time.strftime('%Y%m%d%H%M%S')}.gkg.csv.zip"
    if not path.exists() or path.stat().st_size <= 0:
        return False
    if not validate_existing:
        return True
    try:
        with zipfile.ZipFile(path) as archive:
            return bool(archive.namelist()) and archive.testzip() is None
    except zipfile.BadZipFile:
        return False


def _validate_zip_bytes(data: bytes) -> None:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if not archive.namelist():
            raise ValueError("empty zip")
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"corrupt zip member: {bad_member}")


def _remove_temp(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _iter_times(start: datetime, end: datetime, interval_minutes: int) -> Iterable[datetime]:
    current = start
    step = timedelta(minutes=interval_minutes)
    while current < end:
        yield current
        current += step


def _parse_utc(value: str, interval_minutes: int) -> datetime:
    clean = value.strip().replace("Z", "+00:00")
    if len(clean) == 10:
        clean = f"{clean}T00:00:00+00:00"
    parsed = datetime.fromisoformat(clean)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc).replace(second=0, microsecond=0)
    minute = (parsed.minute // interval_minutes) * interval_minutes
    return parsed.replace(minute=minute)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import argparse
import csv
import io
import json
import re
import sqlite3
import urllib.error
import urllib.request
import zipfile

from .builder import default_paths


GDELT_URL_TEMPLATE = "http://data.gdeltproject.org/gdeltv2/{stamp}.export.CSV.zip"
DEFAULT_RAW_DIR = Path("D:/FreqTradeStuffLargeData/research_news_data/gdelt/raw/export")
SCHEMA_VERSION = "1"

COL_EVENT_CODE = 26
COL_EVENT_BASE_CODE = 27
COL_EVENT_ROOT_CODE = 28
COL_QUAD_CLASS = 29
COL_GOLDSTEIN = 30
COL_NUM_MENTIONS = 31
COL_NUM_SOURCES = 32
COL_NUM_ARTICLES = 33
COL_AVG_TONE = 34
COL_SOURCE_URL = 60

TOPIC_PATTERNS = {
    "sanctions_trade_url_count": re.compile(r"\b(sanction|tariff|trade-war|trade_war|export-control|embargo)\b", re.I),
    "oil_energy_url_count": re.compile(r"\b(oil|crude|brent|wti|opec|natural-gas|lng|energy)\b", re.I),
    "banking_credit_url_count": re.compile(r"\b(bank|banking|credit|liquidity|default|insolv|debt|downgrade)\w*\b", re.I),
    "macro_url_count": re.compile(r"\b(fed|fomc|central-bank|inflation|cpi|ppi|jobs|gdp|recession|treasury|yield|rates?)\b", re.I),
    "crypto_url_count": re.compile(r"\b(bitcoin|btc|ethereum|eth|crypto|defi|stablecoin)\b", re.I),
}


@dataclass
class BackfillSummary:
    db_path: str
    raw_dir: str
    start: str
    end: str
    dry_run: bool
    files_planned: int
    files_written: int = 0
    files_failed: int = 0
    event_rows_read: int = 0
    first_feature_hour: str | None = None
    last_feature_hour: str | None = None
    failures: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failures"] = self.failures or []
        return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill hourly GDELT event aggregates for context feature research.")
    parser.add_argument("--app-dir", type=Path, default=None)
    parser.add_argument("--start", required=True, help="UTC start date/time, inclusive. Example: 2026-03-01")
    parser.add_argument("--end", required=True, help="UTC end date/time, exclusive. Example: 2026-05-01")
    parser.add_argument("--step-hours", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--discard-raw", action="store_true", help="Parse downloaded files without saving zip files to disk.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip hours already present in gdelt_hourly_features.")
    parser.add_argument(
        "--fill-failed-quarter-hours",
        action="store_true",
        help="Retry failed hourly rows using available :00/:15/:30/:45 GDELT event files.",
    )
    parser.add_argument("--progress-every", type=int, default=0)
    args = parser.parse_args()

    if args.step_hours <= 0:
        raise SystemExit("--step-hours must be positive")
    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be positive")

    paths = default_paths(args.app_dir)
    db_path = args.db or paths.gdelt_db
    raw_dir = args.raw_dir or DEFAULT_RAW_DIR
    start = _parse_utc(args.start)
    end = _parse_utc(args.end)
    if end <= start:
        raise SystemExit("--end must be after --start")
    if args.fill_failed_quarter_hours:
        return _run_fill_failed_quarter_hours(args, db_path, raw_dir, start, end)

    hours = list(_iter_hours(start, end, args.step_hours))
    if args.limit_files is not None:
        hours = hours[: args.limit_files]
    if args.skip_existing and not args.dry_run:
        init_db(db_path)
        existing = _existing_success_hours(db_path, start, end)
        hours = [hour for hour in hours if hour.isoformat() not in existing]

    summary = BackfillSummary(
        db_path=str(db_path),
        raw_dir=str(raw_dir),
        start=start.isoformat(),
        end=end.isoformat(),
        dry_run=bool(args.dry_run),
        files_planned=len(hours),
    )
    if args.dry_run:
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        return 0

    init_db(db_path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(
                _download_and_parse_hour,
                hour,
                raw_dir,
                use_cache=not args.no_cache,
                write_cache=not args.discard_raw,
            ): hour
            for hour in hours
        }
        completed = 0
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            if row.get("parse_error"):
                failures.append(f"{row['date']}: {row['parse_error']}")
            completed += 1
            if args.progress_every and completed % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "progress": completed,
                            "files_planned": len(hours),
                            "files_failed": len(failures),
                            "event_rows_read": int(sum(item["event_count"] for item in rows if not item.get("parse_error"))),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    with connect_db(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            _write_rows(conn, rows)
            _write_run(conn, summary, rows, failures)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    good_rows = [row for row in rows if not row.get("parse_error")]
    summary.files_written = len(good_rows)
    summary.files_failed = len(failures)
    summary.event_rows_read = int(sum(row["event_count"] for row in good_rows))
    summary.first_feature_hour = min((row["date"] for row in good_rows), default=None)
    summary.last_feature_hour = max((row["date"] for row in good_rows), default=None)
    summary.failures = failures[:20]
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0 if not failures else 2


def _run_fill_failed_quarter_hours(
    args: argparse.Namespace,
    db_path: Path,
    raw_dir: Path,
    start: datetime,
    end: datetime,
) -> int:
    failed_hours = _failed_event_hours(db_path, start, end)
    candidates: list[datetime] = []
    for hour in failed_hours:
        for minute_offset in (0, 15, 30, 45):
            candidate = hour + timedelta(minutes=minute_offset)
            if start <= candidate < end:
                candidates.append(candidate)
    if args.skip_existing:
        existing = _existing_success_timestamps(db_path, start, end)
        candidates = [candidate for candidate in candidates if candidate.isoformat() not in existing]
    if args.limit_files is not None:
        candidates = candidates[: args.limit_files]

    summary = BackfillSummary(
        db_path=str(db_path),
        raw_dir=str(raw_dir),
        start=start.isoformat(),
        end=end.isoformat(),
        dry_run=bool(args.dry_run),
        files_planned=len(candidates),
    )
    if args.dry_run:
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        return 0
    if not candidates:
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        return 0

    init_db(db_path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(
                _download_and_parse_hour,
                candidate,
                raw_dir,
                use_cache=not args.no_cache,
                write_cache=not args.discard_raw,
            ): candidate
            for candidate in candidates
        }
        completed = 0
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            if row.get("parse_error"):
                failures.append(f"{row['date']}: {row['parse_error']}")
            completed += 1
            if args.progress_every and completed % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "progress": completed,
                            "files_planned": len(candidates),
                            "files_failed": len(failures),
                            "event_rows_read": int(sum(item["event_count"] for item in rows if not item.get("parse_error"))),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    with connect_db(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            _write_rows(conn, rows)
            _write_run(conn, summary, rows, failures)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    good_rows = [row for row in rows if not row.get("parse_error")]
    summary.files_written = len(good_rows)
    summary.files_failed = len(failures)
    summary.event_rows_read = int(sum(row["event_count"] for row in good_rows))
    summary.first_feature_hour = min((row["date"] for row in good_rows), default=None)
    summary.last_feature_hour = max((row["date"] for row in good_rows), default=None)
    summary.failures = failures[:20]
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db(db_path: Path) -> None:
    with connect_db(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS gdelt_hourly_features (
                date TEXT PRIMARY KEY,
                file_url TEXT NOT NULL,
                downloaded_at TEXT NOT NULL,
                event_count REAL NOT NULL DEFAULT 0,
                num_mentions_sum REAL NOT NULL DEFAULT 0,
                num_sources_sum REAL NOT NULL DEFAULT 0,
                num_articles_sum REAL NOT NULL DEFAULT 0,
                avg_tone_weighted REAL,
                goldstein_weighted REAL,
                conflict_event_count REAL NOT NULL DEFAULT 0,
                protest_event_count REAL NOT NULL DEFAULT 0,
                coercion_event_count REAL NOT NULL DEFAULT 0,
                sanctions_trade_url_count REAL NOT NULL DEFAULT 0,
                oil_energy_url_count REAL NOT NULL DEFAULT 0,
                banking_credit_url_count REAL NOT NULL DEFAULT 0,
                macro_url_count REAL NOT NULL DEFAULT 0,
                crypto_url_count REAL NOT NULL DEFAULT 0,
                parse_error TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_gdelt_hourly_parse_error ON gdelt_hourly_features(parse_error);

            CREATE TABLE IF NOT EXISTS gdelt_backfill_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                start TEXT NOT NULL,
                end TEXT NOT NULL,
                files_planned INTEGER NOT NULL,
                files_written INTEGER NOT NULL,
                files_failed INTEGER NOT NULL,
                event_rows_read INTEGER NOT NULL,
                notes_json TEXT NOT NULL
            );
            """
        )
        conn.execute("INSERT OR REPLACE INTO schema_meta(key, value) VALUES(?, ?)", ("version", SCHEMA_VERSION))


def _download_and_parse_hour(hour: datetime, raw_dir: Path, *, use_cache: bool, write_cache: bool) -> dict[str, Any]:
    stamp = hour.strftime("%Y%m%d%H%M%S")
    url = GDELT_URL_TEMPLATE.format(stamp=stamp)
    raw_path = raw_dir / f"{stamp}.export.CSV.zip"
    try:
        if raw_path.exists() and use_cache:
            data = raw_path.read_bytes()
        else:
            request = urllib.request.Request(url, headers={"User-Agent": "FreqTradeStuff-context-research/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read()
            if write_cache:
                raw_path.write_bytes(data)
        return _parse_zip(hour, url, data)
    except urllib.error.HTTPError as exc:
        return _error_row(hour, url, f"HTTP {exc.code}")
    except Exception as exc:
        return _error_row(hour, url, str(exc))


def _parse_zip(hour: datetime, url: str, data: bytes) -> dict[str, Any]:
    row = _base_row(hour, url)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if not names:
            raise ValueError("empty zip")
        with archive.open(names[0], "r") as handle:
            reader = csv.reader(io.TextIOWrapper(handle, encoding="utf-8", errors="replace"), delimiter="\t")
            article_weight_sum = 0.0
            tone_weight_sum = 0.0
            goldstein_weight_sum = 0.0
            for values in reader:
                if len(values) <= COL_SOURCE_URL:
                    continue
                num_mentions = _to_float(values[COL_NUM_MENTIONS])
                num_sources = _to_float(values[COL_NUM_SOURCES])
                num_articles = _to_float(values[COL_NUM_ARTICLES])
                avg_tone = _to_float(values[COL_AVG_TONE])
                goldstein = _to_float(values[COL_GOLDSTEIN])
                weight = max(num_articles, 1.0)
                row["event_count"] += 1.0
                row["num_mentions_sum"] += num_mentions
                row["num_sources_sum"] += num_sources
                row["num_articles_sum"] += num_articles
                tone_weight_sum += avg_tone * weight
                goldstein_weight_sum += goldstein * weight
                article_weight_sum += weight
                _add_event_family_counts(row, values)
            if article_weight_sum:
                row["avg_tone_weighted"] = tone_weight_sum / article_weight_sum
                row["goldstein_weighted"] = goldstein_weight_sum / article_weight_sum
    return row


def _add_event_family_counts(row: dict[str, Any], values: list[str]) -> None:
    event_code = values[COL_EVENT_CODE].strip()
    event_base = values[COL_EVENT_BASE_CODE].strip()
    event_root = values[COL_EVENT_ROOT_CODE].strip()
    quad_class = values[COL_QUAD_CLASS].strip()
    url = values[COL_SOURCE_URL]
    if quad_class == "4" or event_root in {"18", "19", "20"}:
        row["conflict_event_count"] += 1.0
    if event_root == "14":
        row["protest_event_count"] += 1.0
    if event_root == "17":
        row["coercion_event_count"] += 1.0
    if event_root == "16" or event_base.startswith("163") or event_code.startswith("163"):
        row["sanctions_trade_url_count"] += 1.0
    for column, pattern in TOPIC_PATTERNS.items():
        if pattern.search(url):
            row[column] += 1.0


def _base_row(hour: datetime, url: str) -> dict[str, Any]:
    return {
        "date": hour.isoformat(),
        "file_url": url,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "event_count": 0.0,
        "num_mentions_sum": 0.0,
        "num_sources_sum": 0.0,
        "num_articles_sum": 0.0,
        "avg_tone_weighted": None,
        "goldstein_weighted": None,
        "conflict_event_count": 0.0,
        "protest_event_count": 0.0,
        "coercion_event_count": 0.0,
        "sanctions_trade_url_count": 0.0,
        "oil_energy_url_count": 0.0,
        "banking_credit_url_count": 0.0,
        "macro_url_count": 0.0,
        "crypto_url_count": 0.0,
        "parse_error": None,
    }


def _error_row(hour: datetime, url: str, message: str) -> dict[str, Any]:
    row = _base_row(hour, url)
    row["parse_error"] = message[:500]
    return row


def _write_rows(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(_base_row(datetime(2000, 1, 1, tzinfo=timezone.utc), "").keys())
    placeholders = ", ".join("?" for _ in columns)
    updates = ", ".join(f"{column}=excluded.{column}" for column in columns if column != "date")
    conn.executemany(
        f"""
        INSERT INTO gdelt_hourly_features({', '.join(columns)})
        VALUES ({placeholders})
        ON CONFLICT(date) DO UPDATE SET {updates}
        """,
        [tuple(row[column] for column in columns) for row in rows],
    )


def _write_run(
    conn: sqlite3.Connection,
    summary: BackfillSummary,
    rows: list[dict[str, Any]],
    failures: list[str],
) -> None:
    good_rows = [row for row in rows if not row.get("parse_error")]
    notes = {"failures": failures[:50]}
    conn.execute(
        """
        INSERT INTO gdelt_backfill_runs(
            started_at, start, end, files_planned, files_written, files_failed, event_rows_read, notes_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            summary.start,
            summary.end,
            summary.files_planned,
            len(good_rows),
            len(failures),
            int(sum(row["event_count"] for row in good_rows)),
            json.dumps(notes, separators=(",", ":"), sort_keys=True),
        ),
    )


def _existing_success_hours(db_path: Path, start: datetime, end: datetime) -> set[str]:
    with connect_db(db_path) as conn:
        rows = conn.execute(
            """
            SELECT date
            FROM gdelt_hourly_features
            WHERE date >= ?
              AND date < ?
              AND parse_error IS NULL
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    return {str(row["date"]) for row in rows}


def _failed_event_hours(db_path: Path, start: datetime, end: datetime) -> list[datetime]:
    if not db_path.exists():
        return []
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT date
            FROM gdelt_hourly_features
            WHERE date >= ?
              AND date < ?
              AND parse_error IS NOT NULL
            ORDER BY date
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    failed: list[datetime] = []
    for row in rows:
        parsed = _parse_stored_timestamp(str(row["date"]))
        if parsed is not None and parsed.minute == 0 and parsed.second == 0:
            failed.append(parsed)
    return failed


def _existing_success_timestamps(db_path: Path, start: datetime, end: datetime) -> set[str]:
    if not db_path.exists():
        return set()
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT date
            FROM gdelt_hourly_features
            WHERE date >= ?
              AND date < ?
              AND parse_error IS NULL
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    return {str(row["date"]) for row in rows}


def _parse_utc(value: str) -> datetime:
    clean = value.strip().replace("Z", "+00:00")
    if len(clean) == 10:
        clean = f"{clean}T00:00:00+00:00"
    parsed = datetime.fromisoformat(clean)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def _parse_stored_timestamp(value: str) -> datetime | None:
    try:
        clean = value.strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(clean)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0)


def _iter_hours(start: datetime, end: datetime, step_hours: int):
    current = start
    step = timedelta(hours=step_hours)
    while current < end:
        yield current
        current += step


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())

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


GKG_URL_TEMPLATE = "http://data.gdeltproject.org/gdeltv2/{stamp}.gkg.csv.zip"
DEFAULT_RAW_DIR = Path("D:/FreqTradeStuffLargeData/research_news_data/gdelt/raw/gkg")
SCHEMA_VERSION = "1"
CSV_FIELD_SIZE_LIMIT = 128 * 1024 * 1024
TERMINAL_PARSE_ERROR_PREFIXES = ("HTTP 404",)

COL_SOURCE = 3
COL_URL = 4
COL_THEMES = 7
COL_V2_THEMES = 8
COL_PERSONS = 11
COL_V2_PERSONS = 12
COL_ORGS = 13
COL_V2_ORGS = 14
COL_V2_TONE = 15
COL_ALL_NAMES = 23

TOPIC_PATTERNS = {
    "crypto": re.compile(r"\b(bitcoin|btc|ethereum|eth|crypto|cryptocurrency|blockchain|defi|stablecoin|coinbase|binance)\b", re.I),
    "bitcoin": re.compile(r"\b(bitcoin|btc)\b", re.I),
    "ethereum": re.compile(r"\b(ethereum|eth)\b", re.I),
    "stablecoin_liquidity": re.compile(r"\b(stablecoin|tether|usdt|usdc|liquidity|money_supply|repo|balance_sheet)\b", re.I),
    "regulation": re.compile(r"\b(regulat|sec|cftc|mica|lawsuit|legal|compliance|enforcement|legislation|law)\w*\b", re.I),
    "etf_institutional": re.compile(r"\b(etf|exchange_traded|exchange-traded|blackrock|fidelity|grayscale|institutional|fund_flow)\w*\b", re.I),
    "security_exploit": re.compile(r"\b(hack|exploit|breach|security|stolen|phishing|vulnerability|ransomware)\w*\b", re.I),
    "macro_economic": re.compile(r"\b(econ|macro|gdp|recession|inflation|cpi|ppi|jobs|unemployment|treasury|yield|federal_reserve|central_bank|monetary_policy|epu_)\w*\b", re.I),
    "central_bank": re.compile(r"\b(central_bank|federal_reserve|fed|fomc|ecb|boe|boj|monetary_policy)\w*\b", re.I),
    "rates": re.compile(r"\b(rate_hike|rate_cut|interest_rate|yield|treasury|gilts?|bonds?)\w*\b", re.I),
    "inflation": re.compile(r"\b(inflation|cpi|ppi|pce|consumer_price|producer_price)\w*\b", re.I),
    "jobs_labor": re.compile(r"\b(jobs?|payroll|employment|unemployment|labor|labour|wage|jobless)\w*\b", re.I),
    "recession_growth": re.compile(r"\b(recession|slowdown|growth|contraction|expansion|soft_landing|hard_landing)\w*\b", re.I),
    "banking_credit": re.compile(r"\b(bank|banking|credit|liquidity_crisis|bank_run|deposit|lending|loan|default|insolvenc|contagion|debt)\w*\b", re.I),
    "oil_energy": re.compile(r"\b(oil|crude|brent|wti|opec|natural_gas|lng|energy|hormuz)\b", re.I),
    "sanctions_trade": re.compile(r"\b(sanction|tariff|trade_war|export_control|embargo|blacklist)\w*\b", re.I),
    "war_geopolitics": re.compile(r"\b(war|conflict|military|missile|attack|ukraine|russia|israel|iran|gaza|taiwan|geopolit|terror|armedconflict)\w*\b", re.I),
}


@dataclass
class GkgBackfillSummary:
    db_path: str
    raw_dir: str
    start: str
    end: str
    dry_run: bool
    files_planned: int
    files_written: int = 0
    files_failed: int = 0
    documents_read: int = 0
    first_file_time: str | None = None
    last_file_time: str | None = None
    failures: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failures"] = self.failures or []
        return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill GDELT GKG document/theme aggregates for context feature research.")
    parser.add_argument("--app-dir", type=Path, default=None)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--interval-minutes", type=int, default=15)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--next-missing-chunk-days", type=int, default=0)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--discard-raw", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--retry-failed-only",
        action="store_true",
        help="Retry only rows already present with parse_error set inside the requested window.",
    )
    parser.add_argument("--progress-every", type=int, default=0)
    args = parser.parse_args()

    _raise_csv_field_size_limit()
    if args.interval_minutes <= 0:
        raise SystemExit("--interval-minutes must be positive")
    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be positive")
    if args.retry_failed_only and args.skip_existing:
        raise SystemExit("--retry-failed-only and --skip-existing are mutually exclusive")

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
        window = (
            _next_missing_window(db_path, start, end, args.next_missing_chunk_days, args.interval_minutes)
            if db_path.exists()
            else (start, min(start + timedelta(days=max(1, args.next_missing_chunk_days)), end))
        )
        if window is None:
            summary = GkgBackfillSummary(
                str(db_path),
                str(raw_dir),
                start.isoformat(),
                end.isoformat(),
                bool(args.dry_run),
                0,
            )
            print(json.dumps({**summary.to_dict(), "status": "complete"}, indent=2, sort_keys=True))
            return 0
        start, end = window
    stamps = list(_iter_times(start, end, args.interval_minutes))
    if args.retry_failed_only:
        if not args.dry_run:
            init_db(db_path)
        failed = _failed_file_times(db_path, start, end) if db_path.exists() else set()
        stamps = [stamp for stamp in stamps if stamp.isoformat() in failed]
    if args.skip_existing:
        if not args.dry_run:
            init_db(db_path)
        existing = _successful_file_times(db_path, start, end) if db_path.exists() else set()
        stamps = [stamp for stamp in stamps if stamp.isoformat() not in existing]
    if args.limit_files is not None:
        stamps = stamps[: args.limit_files]

    summary = GkgBackfillSummary(
        db_path=str(db_path),
        raw_dir=str(raw_dir),
        start=start.isoformat(),
        end=end.isoformat(),
        dry_run=bool(args.dry_run),
        files_planned=len(stamps),
    )
    if args.dry_run:
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        return 0

    init_db(db_path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(
                _download_and_parse_file,
                stamp,
                raw_dir,
                use_cache=not args.no_cache,
                write_cache=not args.discard_raw,
            ): stamp
            for stamp in stamps
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
                            "files_planned": len(stamps),
                            "files_failed": len(failures),
                            "documents_read": int(sum(item["document_count"] for item in rows if not item.get("parse_error"))),
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
    summary.documents_read = int(sum(row["document_count"] for row in good_rows))
    summary.first_file_time = min((row["date"] for row in good_rows), default=None)
    summary.last_file_time = max((row["date"] for row in good_rows), default=None)
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


def _raise_csv_field_size_limit() -> None:
    limit = CSV_FIELD_SIZE_LIMIT
    while limit > csv.field_size_limit():
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = int(limit / 10)


def init_db(db_path: Path) -> None:
    topic_defs = ",\n                ".join(f"{topic}_doc_count REAL NOT NULL DEFAULT 0" for topic in TOPIC_PATTERNS)
    with connect_db(db_path) as conn:
        conn.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS gdelt_gkg_file_features (
                date TEXT PRIMARY KEY,
                file_url TEXT NOT NULL,
                downloaded_at TEXT NOT NULL,
                document_count REAL NOT NULL DEFAULT 0,
                source_count REAL NOT NULL DEFAULT 0,
                word_count_sum REAL NOT NULL DEFAULT 0,
                tone_weight_sum REAL NOT NULL DEFAULT 0,
                tone_sum REAL NOT NULL DEFAULT 0,
                positive_tone_sum REAL NOT NULL DEFAULT 0,
                negative_tone_sum REAL NOT NULL DEFAULT 0,
                polarity_sum REAL NOT NULL DEFAULT 0,
                activity_sum REAL NOT NULL DEFAULT 0,
                theme_count_sum REAL NOT NULL DEFAULT 0,
                unique_theme_count REAL NOT NULL DEFAULT 0,
                top_theme_doc_count REAL NOT NULL DEFAULT 0,
                {topic_defs},
                parse_error TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_gkg_file_parse_error ON gdelt_gkg_file_features(parse_error);

            CREATE TABLE IF NOT EXISTS gdelt_gkg_backfill_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                start TEXT NOT NULL,
                end TEXT NOT NULL,
                files_planned INTEGER NOT NULL,
                files_written INTEGER NOT NULL,
                files_failed INTEGER NOT NULL,
                documents_read INTEGER NOT NULL,
                notes_json TEXT NOT NULL
            );
            """
        )
        conn.execute("INSERT OR REPLACE INTO schema_meta(key, value) VALUES(?, ?)", ("gkg_version", SCHEMA_VERSION))


def _download_and_parse_file(stamp_time: datetime, raw_dir: Path, *, use_cache: bool, write_cache: bool) -> dict[str, Any]:
    stamp = stamp_time.strftime("%Y%m%d%H%M%S")
    url = GKG_URL_TEMPLATE.format(stamp=stamp)
    raw_path = raw_dir / f"{stamp}.gkg.csv.zip"
    try:
        if raw_path.exists() and use_cache:
            data = raw_path.read_bytes()
        else:
            request = urllib.request.Request(url, headers={"User-Agent": "FreqTradeStuff-context-research/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            if write_cache:
                raw_path.write_bytes(data)
        return _parse_zip(stamp_time, url, data)
    except urllib.error.HTTPError as exc:
        return _error_row(stamp_time, url, f"HTTP {exc.code}")
    except Exception as exc:
        return _error_row(stamp_time, url, str(exc))


def _parse_zip(stamp_time: datetime, url: str, data: bytes) -> dict[str, Any]:
    _raise_csv_field_size_limit()
    row = _base_row(stamp_time, url)
    sources: set[str] = set()
    theme_counts: dict[str, int] = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if not names:
            raise ValueError("empty zip")
        with archive.open(names[0], "r") as handle:
            reader = csv.reader(io.TextIOWrapper(handle, encoding="utf-8", errors="replace"), delimiter="\t")
            for values in reader:
                if len(values) <= COL_V2_TONE:
                    continue
                row["document_count"] += 1.0
                source = values[COL_SOURCE].strip()
                if source:
                    sources.add(source)
                themes = _theme_tokens(values[COL_THEMES] if len(values) > COL_THEMES else "", values[COL_V2_THEMES] if len(values) > COL_V2_THEMES else "")
                row["theme_count_sum"] += float(len(themes))
                for theme in set(themes):
                    theme_counts[theme] = theme_counts.get(theme, 0) + 1
                combined = _combined_text(values, themes)
                for topic, pattern in TOPIC_PATTERNS.items():
                    if pattern.search(combined):
                        row[f"{topic}_doc_count"] += 1.0
                _add_tone(row, values[COL_V2_TONE])
    row["source_count"] = float(len(sources))
    row["unique_theme_count"] = float(len(theme_counts))
    row["top_theme_doc_count"] = float(max(theme_counts.values(), default=0))
    return row


def _base_row(stamp_time: datetime, url: str) -> dict[str, Any]:
    row = {
        "date": stamp_time.isoformat(),
        "file_url": url,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "document_count": 0.0,
        "source_count": 0.0,
        "word_count_sum": 0.0,
        "tone_weight_sum": 0.0,
        "tone_sum": 0.0,
        "positive_tone_sum": 0.0,
        "negative_tone_sum": 0.0,
        "polarity_sum": 0.0,
        "activity_sum": 0.0,
        "theme_count_sum": 0.0,
        "unique_theme_count": 0.0,
        "top_theme_doc_count": 0.0,
        "parse_error": None,
    }
    for topic in TOPIC_PATTERNS:
        row[f"{topic}_doc_count"] = 0.0
    return row


def _error_row(stamp_time: datetime, url: str, message: str) -> dict[str, Any]:
    row = _base_row(stamp_time, url)
    row["parse_error"] = message[:500]
    return row


def _theme_tokens(themes: str, v2_themes: str) -> list[str]:
    tokens: list[str] = []
    for raw in f"{themes};{v2_themes}".split(";"):
        token = raw.split(",", 1)[0].strip().upper()
        if token:
            tokens.append(token)
    return tokens


def _combined_text(values: list[str], themes: list[str]) -> str:
    parts = [
        " ".join(themes),
        values[COL_SOURCE] if len(values) > COL_SOURCE else "",
        values[COL_URL] if len(values) > COL_URL else "",
        values[COL_PERSONS] if len(values) > COL_PERSONS else "",
        values[COL_V2_PERSONS] if len(values) > COL_V2_PERSONS else "",
        values[COL_ORGS] if len(values) > COL_ORGS else "",
        values[COL_V2_ORGS] if len(values) > COL_V2_ORGS else "",
        values[COL_ALL_NAMES] if len(values) > COL_ALL_NAMES else "",
    ]
    return " ".join(parts).replace("-", "_").replace("/", "_")


def _add_tone(row: dict[str, Any], tone_value: str) -> None:
    values = [_to_float(part) for part in tone_value.split(",")]
    if len(values) < 5:
        return
    weight = values[6] if len(values) > 6 and values[6] > 0 else 1.0
    row["tone_weight_sum"] += weight
    row["tone_sum"] += values[0] * weight
    row["positive_tone_sum"] += values[1] * weight
    row["negative_tone_sum"] += values[2] * weight
    row["polarity_sum"] += values[3] * weight
    row["activity_sum"] += values[4] * weight
    row["word_count_sum"] += weight


def _write_rows(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(_base_row(datetime(2000, 1, 1, tzinfo=timezone.utc), "").keys())
    placeholders = ", ".join("?" for _ in columns)
    updates = ", ".join(f"{column}=excluded.{column}" for column in columns if column != "date")
    conn.executemany(
        f"""
        INSERT INTO gdelt_gkg_file_features({', '.join(columns)})
        VALUES ({placeholders})
        ON CONFLICT(date) DO UPDATE SET {updates}
        """,
        [tuple(row[column] for column in columns) for row in rows],
    )


def _write_run(conn: sqlite3.Connection, summary: GkgBackfillSummary, rows: list[dict[str, Any]], failures: list[str]) -> None:
    good_rows = [row for row in rows if not row.get("parse_error")]
    conn.execute(
        """
        INSERT INTO gdelt_gkg_backfill_runs(
            started_at, start, end, files_planned, files_written, files_failed, documents_read, notes_json
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
            int(sum(row["document_count"] for row in good_rows)),
            json.dumps({"failures": failures[:50]}, separators=(",", ":"), sort_keys=True),
        ),
    )


def _successful_file_times(db_path: Path, start: datetime, end: datetime) -> set[str]:
    with connect_db(db_path) as conn:
        rows = conn.execute(
            """
            SELECT date
            FROM gdelt_gkg_file_features
            WHERE date >= ? AND date < ?
              AND parse_error IS NULL
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    return {str(row["date"]) for row in rows}


def _terminal_failure_file_times(db_path: Path, start: datetime, end: datetime) -> set[str]:
    with connect_db(db_path) as conn:
        rows = conn.execute(
            """
            SELECT date, parse_error
            FROM gdelt_gkg_file_features
            WHERE date >= ? AND date < ?
              AND parse_error IS NOT NULL
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    return {
        str(row["date"])
        for row in rows
        if _is_terminal_parse_error(str(row["parse_error"] or ""))
    }


def _failed_file_times(db_path: Path, start: datetime, end: datetime) -> set[str]:
    with connect_db(db_path) as conn:
        rows = conn.execute(
            """
            SELECT date
            FROM gdelt_gkg_file_features
            WHERE date >= ? AND date < ?
              AND parse_error IS NOT NULL
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    return {str(row["date"]) for row in rows}


def _complete_or_terminal_file_times(db_path: Path, start: datetime, end: datetime) -> set[str]:
    completed = _successful_file_times(db_path, start, end)
    completed.update(_terminal_failure_file_times(db_path, start, end))
    return completed


def _is_terminal_parse_error(message: str) -> bool:
    return any(message.startswith(prefix) for prefix in TERMINAL_PARSE_ERROR_PREFIXES)


def _next_missing_window(db_path: Path, start: datetime, end: datetime, chunk_days: int, interval_minutes: int) -> tuple[datetime, datetime] | None:
    existing = _complete_or_terminal_file_times(db_path, start, end)
    for stamp in _iter_times(start, end, interval_minutes):
        if stamp.isoformat() not in existing:
            return stamp, min(stamp + timedelta(days=max(1, chunk_days)), end)
    return None


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


def _iter_times(start: datetime, end: datetime, interval_minutes: int):
    current = start
    step = timedelta(minutes=interval_minutes)
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

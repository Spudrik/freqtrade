from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import argparse
import csv
import hashlib
import io
import json
import re
import sqlite3
import zipfile

from .builder import default_paths


DEFAULT_RAW_DIR = Path("D:/FreqTradeStuffLargeData/research_news_data/gdelt/raw/gkg")
CSV_FIELD_SIZE_LIMIT = 128 * 1024 * 1024

COL_GKG_RECORD_ID = 0
COL_DATE = 1
COL_SOURCE_COLLECTION_IDENTIFIER = 2
COL_SOURCE_NAME = 3
COL_DOCUMENT_IDENTIFIER = 4
COL_COUNTS = 5
COL_V2_COUNTS = 6
COL_THEMES = 7
COL_V2_THEMES = 8
COL_LOCATIONS = 9
COL_V2_LOCATIONS = 10
COL_PERSONS = 11
COL_V2_PERSONS = 12
COL_ORGS = 13
COL_V2_ORGS = 14
COL_V2_TONE = 15
COL_DATES = 16
COL_GCAM = 17
COL_ALL_NAMES = 23
COL_AMOUNTS = 24
COL_EXTRAS = 26

SCHEMA_COLUMNS = [
    "gkg_record_id",
    "gkg_date",
    "source_collection_identifier",
    "source_name",
    "document_identifier",
    "document_url",
    "available_at",
    "published_at",
    "tone",
    "positive_score",
    "negative_score",
    "polarity",
    "activity_reference_density",
    "self_group_reference_density",
    "word_count",
    "v2_tone_json",
    "themes_json",
    "persons_json",
    "orgs_json",
    "locations_json",
    "all_names_json",
    "counts_json",
    "amounts_json",
    "dates_json",
    "gcam_json",
    "extras_json",
    "weak_topics_json",
    "weak_entities_json",
    "raw_ref_json",
    "inserted_at",
    "updated_at",
]

NORMALIZED_SCHEMA_COLUMNS = [
    "document_id",
    "raw_file_id",
    "gkg_record_id",
    "document_time",
    "available_at",
    "published_at",
    "source_common_name",
    "source_collection_identifier",
    "document_url",
    "canonical_url",
    "url_hash",
    "themes_json",
    "v2_themes_json",
    "persons_json",
    "orgs_json",
    "locations_json",
    "all_names_json",
    "amounts_json",
    "counts_json",
    "dates_json",
    "gcam_json",
    "extras_json",
    "tone",
    "positive_tone",
    "negative_tone",
    "polarity",
    "activity_density",
    "self_group_reference_density",
    "word_count",
    "topic",
    "entities_json",
    "evidence_json",
    "raw_ref_json",
    "parser_version",
    "normalized_schema_version",
    "payload_json",
    "updated_at",
]

TOPIC_PATTERNS = {
    "crypto": re.compile(r"\b(bitcoin|btc|ethereum|eth|crypto|cryptocurrency|blockchain|defi|stablecoin)\b", re.I),
    "regulation": re.compile(r"\b(regulat|sec|cftc|mica|lawsuit|enforcement|compliance)\w*\b", re.I),
    "macro": re.compile(r"\b(inflation|cpi|fomc|federal_reserve|central_bank|rates?|treasury|recession|jobs?)\w*\b", re.I),
    "banking_credit": re.compile(r"\b(bank|banking|credit|liquidity|default|debt|insolvenc)\w*\b", re.I),
    "geopolitics": re.compile(r"\b(war|conflict|sanction|tariff|russia|ukraine|israel|iran|taiwan)\w*\b", re.I),
    "security_exploit": re.compile(r"\b(hack|exploit|breach|stolen|phishing|vulnerability|ransomware)\w*\b", re.I),
}

EXTRA_TIMESTAMP_KEYS = re.compile(r"(PUB|PUBLISH|PUBLICATION|DATE|TIME|TIMESTAMP)", re.I)


@dataclass
class NormalizeSummary:
    db_path: str
    raw_dir: str
    dry_run: bool
    files_planned: int
    files_read: int = 0
    rows_read: int = 0
    rows_written: int = 0
    rows_skipped: int = 0
    first_available_at: str | None = None
    last_available_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize raw GDELT GKG 2.0 ZIP rows into document-level silver records.")
    parser.add_argument("--app-dir", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--start", default=None, help="Inclusive UTC batch-stamp lower bound.")
    parser.add_argument("--end", default=None, help="Exclusive UTC batch-stamp upper bound.")
    args = parser.parse_args()

    if args.limit_files is not None and args.limit_files < 0:
        raise SystemExit("--limit-files must be non-negative")
    if args.limit_rows is not None and args.limit_rows < 0:
        raise SystemExit("--limit-rows must be non-negative")
    start_at = _parse_window_bound(args.start, "--start")
    end_at = _parse_window_bound(args.end, "--end")
    _validate_window(start_at, end_at)

    _raise_csv_field_size_limit()
    paths = default_paths(args.app_dir)
    db_path = args.db or paths.gdelt_db
    raw_dir = args.raw_dir or DEFAULT_RAW_DIR
    zip_paths = sorted(raw_dir.glob("*.gkg.csv.zip"))
    zip_paths = _filter_paths_for_window(zip_paths, start_at, end_at)
    if args.limit_files is not None:
        zip_paths = zip_paths[: args.limit_files]

    summary = NormalizeSummary(
        db_path=str(db_path),
        raw_dir=str(raw_dir),
        dry_run=bool(args.dry_run),
        files_planned=len(zip_paths),
    )
    rows = list(_iter_records(zip_paths, args.limit_rows, summary))
    summary.rows_read = len(rows) + summary.rows_skipped
    summary.first_available_at = min((str(row["available_at"]) for row in rows), default=None)
    summary.last_available_at = max((str(row["available_at"]) for row in rows), default=None)

    if not args.dry_run:
        init_db(db_path)
        with connect_db(db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                summary.rows_written = write_rows(conn, rows)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

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
        _try_normalized_schema(conn, db_path)
        if not _table_exists(conn, "gkg_documents_silver"):
            _create_fallback_schema(conn)


def write_rows(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    table_columns = _table_columns(conn, "gkg_documents_silver")
    if "document_id" in table_columns and "raw_file_id" in table_columns:
        columns = [column for column in NORMALIZED_SCHEMA_COLUMNS if column in table_columns]
        payload_rows = [_to_normalized_schema_row(row, now) for row in rows]
        _upsert_raw_files(conn, payload_rows, now)
        conflict_column = "document_id"
        immutable_columns = {"document_id"}
    else:
        columns = [column for column in SCHEMA_COLUMNS if column in table_columns]
        payload_rows = [_to_fallback_schema_row(row, now) for row in rows]
        conflict_column = "gkg_record_id"
        immutable_columns = {"gkg_record_id", "inserted_at"}

    placeholders = ", ".join("?" for _ in columns)
    updates = ", ".join(f"{column}=excluded.{column}" for column in columns if column not in immutable_columns)
    conn.executemany(
        f"""
        INSERT INTO gkg_documents_silver({', '.join(columns)})
        VALUES ({placeholders})
        ON CONFLICT({conflict_column}) DO UPDATE SET {updates}
        """,
        [tuple(row.get(column) for column in columns) for row in payload_rows],
    )
    return len(rows)


def parse_gkg_row(values: list[str], *, available_at: datetime, raw_ref: dict[str, Any]) -> dict[str, Any] | None:
    if len(values) <= COL_DOCUMENT_IDENTIFIER:
        return None
    gkg_record_id = _cell(values, COL_GKG_RECORD_ID)
    document_identifier = _cell(values, COL_DOCUMENT_IDENTIFIER)
    if not gkg_record_id or not document_identifier:
        return None

    source_collection_identifier = _to_int(_cell(values, COL_SOURCE_COLLECTION_IDENTIFIER))
    tone = parse_v2_tone(_cell(values, COL_V2_TONE))
    themes = _parse_name_offsets(_cell(values, COL_THEMES), _cell(values, COL_V2_THEMES), "theme")
    persons = _parse_name_offsets(_cell(values, COL_PERSONS), _cell(values, COL_V2_PERSONS), "name")
    orgs = _parse_name_offsets(_cell(values, COL_ORGS), _cell(values, COL_V2_ORGS), "name")
    locations = _parse_locations(_cell(values, COL_LOCATIONS), _cell(values, COL_V2_LOCATIONS))
    all_names = _parse_name_offsets("", _cell(values, COL_ALL_NAMES), "name")
    counts = _parse_counts(_cell(values, COL_COUNTS), _cell(values, COL_V2_COUNTS))
    amounts = _parse_amounts(_cell(values, COL_AMOUNTS))
    dates = _parse_dates(_cell(values, COL_DATES))
    gcam = _parse_gcam(_cell(values, COL_GCAM))
    extras = _parse_extras(_cell(values, COL_EXTRAS))
    gkg_date = _cell(values, COL_DATE)
    published_at = _published_at(extras, gkg_date)

    topic_text = " ".join(
        [
            document_identifier,
            _cell(values, COL_SOURCE_NAME),
            " ".join(str(item.get("theme", "")) for item in themes),
            " ".join(str(item.get("name", "")) for item in persons + orgs + all_names),
            " ".join(str(item.get("full_name", "")) for item in locations),
        ]
    ).replace("-", "_").replace("/", "_")

    return {
        "gkg_record_id": gkg_record_id,
        "gkg_date": gkg_date,
        "source_collection_identifier": source_collection_identifier,
        "source_name": _cell(values, COL_SOURCE_NAME) or None,
        "document_identifier": document_identifier,
        "document_url": document_identifier if source_collection_identifier == 1 else None,
        "available_at": available_at.isoformat(),
        "published_at": published_at,
        "tone": tone.get("tone"),
        "positive_score": tone.get("positive_score"),
        "negative_score": tone.get("negative_score"),
        "polarity": tone.get("polarity"),
        "activity_reference_density": tone.get("activity_reference_density"),
        "self_group_reference_density": tone.get("self_group_reference_density"),
        "word_count": tone.get("word_count"),
        "v2_tone_json": _json(tone),
        "themes_json": _json(themes),
        "persons_json": _json(persons),
        "orgs_json": _json(orgs),
        "locations_json": _json(locations),
        "all_names_json": _json(all_names),
        "counts_json": _json(counts),
        "amounts_json": _json(amounts),
        "dates_json": _json(dates),
        "gcam_json": _json(gcam),
        "extras_json": _json(extras),
        "weak_topics_json": _json(_weak_topics(topic_text)),
        "weak_entities_json": _json(_weak_entities(persons, orgs, locations, all_names)),
        "raw_ref_json": _json(raw_ref),
    }


def parse_v2_tone(value: str) -> dict[str, float | None]:
    parts = [_to_float_or_none(part) for part in value.split(",")]
    keys = [
        "tone",
        "positive_score",
        "negative_score",
        "polarity",
        "activity_reference_density",
        "self_group_reference_density",
        "word_count",
    ]
    return {key: parts[index] if index < len(parts) else None for index, key in enumerate(keys)}


def _iter_records(zip_paths: list[Path], limit_rows: int | None, summary: NormalizeSummary) -> Iterable[dict[str, Any]]:
    remaining = limit_rows
    for zip_path in zip_paths:
        if remaining == 0:
            break
        summary.files_read += 1
        batch_stamp = _batch_stamp_from_path(zip_path)
        available_at = _parse_gkg_timestamp(batch_stamp)
        with zipfile.ZipFile(zip_path) as archive:
            for member in [name for name in archive.namelist() if not name.endswith("/")]:
                with archive.open(member, "r") as handle:
                    reader = csv.reader(io.TextIOWrapper(handle, encoding="utf-8", errors="replace"), delimiter="\t")
                    for row_number, values in enumerate(reader, start=1):
                        if remaining == 0:
                            break
                        record = parse_gkg_row(
                            values,
                            available_at=available_at,
                            raw_ref={
                                "zip_path": str(zip_path),
                                "zip_name": zip_path.name,
                                "zip_member": member,
                                "batch_stamp": batch_stamp,
                                "row_number": row_number,
                            },
                        )
                        if record is None:
                            summary.rows_skipped += 1
                            continue
                        yield record
                        if remaining is not None:
                            remaining -= 1


def _try_normalized_schema(conn: sqlite3.Connection, db_path: Path) -> None:
    try:
        from . import gdelt_gkg_normalized_schema
    except Exception:
        return

    for name in ("init_schema", "ensure_gkg_documents_silver", "ensure_schema", "init_db", "create_tables"):
        func = getattr(gdelt_gkg_normalized_schema, name, None)
        if not callable(func):
            continue
        try:
            func(conn)
        except TypeError:
            func(db_path)
        return


def _create_fallback_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS gkg_documents_silver (
            gkg_record_id TEXT PRIMARY KEY,
            gkg_date TEXT NOT NULL,
            source_collection_identifier INTEGER,
            source_name TEXT,
            document_identifier TEXT NOT NULL,
            document_url TEXT,
            available_at TEXT NOT NULL,
            published_at TEXT,
            tone REAL,
            positive_score REAL,
            negative_score REAL,
            polarity REAL,
            activity_reference_density REAL,
            self_group_reference_density REAL,
            word_count REAL,
            v2_tone_json TEXT NOT NULL,
            themes_json TEXT NOT NULL,
            persons_json TEXT NOT NULL,
            orgs_json TEXT NOT NULL,
            locations_json TEXT NOT NULL,
            all_names_json TEXT NOT NULL,
            counts_json TEXT NOT NULL,
            amounts_json TEXT NOT NULL,
            dates_json TEXT NOT NULL,
            gcam_json TEXT NOT NULL,
            extras_json TEXT NOT NULL,
            weak_topics_json TEXT NOT NULL,
            weak_entities_json TEXT NOT NULL,
            raw_ref_json TEXT NOT NULL,
            inserted_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_gkg_documents_available_at ON gkg_documents_silver(available_at);
        CREATE INDEX IF NOT EXISTS idx_gkg_documents_published_at ON gkg_documents_silver(published_at);
        CREATE INDEX IF NOT EXISTS idx_gkg_documents_source_name ON gkg_documents_silver(source_name);
        """
    )


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    columns: set[str] = set()
    for row in conn.execute(f"PRAGMA table_info({table})").fetchall():
        try:
            columns.add(str(row["name"]))
        except (TypeError, KeyError):
            columns.add(str(row[1]))
    return columns


def _to_fallback_schema_row(row: dict[str, Any], now: str) -> dict[str, Any]:
    payload = dict(row)
    payload.setdefault("inserted_at", now)
    payload["updated_at"] = now
    return payload


def _to_normalized_schema_row(row: dict[str, Any], now: str) -> dict[str, Any]:
    weak_topics = _loads_json(row.get("weak_topics_json"), [])
    raw_ref = _loads_json(row.get("raw_ref_json"), {})
    themes = _loads_json(row.get("themes_json"), [])
    canonical_url = _canonicalize_url(row.get("document_url"))
    payload = {
        "document_identifier": row.get("document_identifier"),
        "source_name": row.get("source_name"),
        "gkg_date": row.get("gkg_date"),
        "source_collection_identifier": row.get("source_collection_identifier"),
        "document_url": row.get("document_url"),
        "v2_tone": _loads_json(row.get("v2_tone_json"), {}),
        "weak_topics": weak_topics,
    }
    return {
        "document_id": row.get("gkg_record_id"),
        "raw_file_id": _raw_file_id(raw_ref),
        "gkg_record_id": row.get("gkg_record_id"),
        "document_time": row.get("published_at") or row.get("available_at"),
        "available_at": row.get("available_at"),
        "published_at": row.get("published_at"),
        "source_common_name": row.get("source_name"),
        "source_collection_identifier": None
        if row.get("source_collection_identifier") is None
        else str(row.get("source_collection_identifier")),
        "document_url": row.get("document_url"),
        "canonical_url": canonical_url,
        "url_hash": _url_hash(canonical_url),
        "themes_json": row.get("themes_json"),
        "v2_themes_json": _json([item for item in themes if item.get("version") == "v2"]),
        "persons_json": row.get("persons_json"),
        "orgs_json": row.get("orgs_json"),
        "locations_json": row.get("locations_json"),
        "all_names_json": row.get("all_names_json"),
        "amounts_json": row.get("amounts_json"),
        "counts_json": row.get("counts_json"),
        "dates_json": row.get("dates_json"),
        "gcam_json": row.get("gcam_json"),
        "extras_json": row.get("extras_json"),
        "tone": row.get("tone"),
        "positive_tone": row.get("positive_score"),
        "negative_tone": row.get("negative_score"),
        "polarity": row.get("polarity"),
        "activity_density": row.get("activity_reference_density"),
        "self_group_reference_density": row.get("self_group_reference_density"),
        "word_count": row.get("word_count"),
        "topic": weak_topics[0] if weak_topics else None,
        "entities_json": row.get("weak_entities_json"),
        "evidence_json": _json({"weak_topics": weak_topics}),
        "raw_ref_json": row.get("raw_ref_json"),
        "parser_version": "gkg_document_normalize_v1",
        "normalized_schema_version": "1",
        "payload_json": _json(payload),
        "updated_at": now,
    }


def _upsert_raw_files(conn: sqlite3.Connection, rows: list[dict[str, Any]], now: str) -> None:
    if not rows or not _table_exists(conn, "gdelt_raw_files"):
        return
    seen: set[str] = set()
    raw_rows: list[tuple[Any, ...]] = []
    for row in rows:
        raw_file_id = str(row.get("raw_file_id") or "")
        if not raw_file_id or raw_file_id in seen:
            continue
        seen.add(raw_file_id)
        raw_ref = _loads_json(row.get("raw_ref_json"), {})
        raw_rows.append(
            (
                raw_file_id,
                "gkg",
                str(raw_ref.get("batch_stamp") or "unknown"),
                raw_ref.get("zip_path"),
                raw_ref.get("zip_name"),
                "parsed",
                now,
                _json({"parser": "gkg_document_normalize_v1"}),
            )
        )
    conn.executemany(
        """
        INSERT INTO gdelt_raw_files(
            raw_file_id, file_kind, gdelt_stamp, local_path, file_name, status, parsed_at, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(raw_file_id) DO UPDATE SET
            status=excluded.status,
            parsed_at=excluded.parsed_at,
            metadata_json=excluded.metadata_json
        """,
        raw_rows,
    )


def _raw_file_id(raw_ref: dict[str, Any]) -> str:
    batch_stamp = str(raw_ref.get("batch_stamp") or "unknown")
    zip_name = str(raw_ref.get("zip_name") or raw_ref.get("zip_path") or "")
    digest = hashlib.sha256(zip_name.encode("utf-8")).hexdigest()[:16]
    return f"gkg_{batch_stamp}_{digest}"


def _canonicalize_url(value: Any) -> str | None:
    try:
        from . import gdelt_gkg_normalized_schema
    except Exception:
        gdelt_gkg_normalized_schema = None
    if gdelt_gkg_normalized_schema is not None:
        return gdelt_gkg_normalized_schema.canonicalize_url(None if value is None else str(value))
    text = "" if value is None else str(value).strip()
    return text or None


def _url_hash(value: str | None) -> str | None:
    if not value:
        return None
    try:
        from . import gdelt_gkg_normalized_schema
    except Exception:
        gdelt_gkg_normalized_schema = None
    if gdelt_gkg_normalized_schema is not None:
        return gdelt_gkg_normalized_schema.url_hash(value)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _loads_json(value: Any, default: Any) -> Any:
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _parse_name_offsets(legacy_value: str, v2_value: str, key: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for token in _tokens(legacy_value):
        items.append({key: token, "offset": None, "version": "legacy"})
    for token in _tokens(v2_value):
        name, offset = _split_last_int(token, ",")
        items.append({key: name, "offset": offset, "version": "v2"})
    return [item for item in items if item[key]]


def _parse_locations(legacy_value: str, v2_value: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    fields = ["location_type", "full_name", "country_code", "adm1_code", "latitude", "longitude", "feature_id", "offset"]
    for version, value in (("legacy", legacy_value), ("v2", v2_value)):
        for token in _tokens(value):
            parts = token.split("#")
            row = {field: parts[index] if index < len(parts) and parts[index] else None for index, field in enumerate(fields)}
            row["version"] = version
            row["location_type"] = _to_int(row["location_type"])
            row["latitude"] = _to_float_or_none(row["latitude"])
            row["longitude"] = _to_float_or_none(row["longitude"])
            row["offset"] = _to_int(row["offset"])
            items.append(row)
    return items


def _parse_counts(legacy_value: str, v2_value: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for version, value in (("legacy", legacy_value), ("v2", v2_value)):
        for token in _tokens(value):
            parts = token.split("#")
            items.append({"version": version, "fields": parts})
    return items


def _parse_amounts(value: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for token in _tokens(value):
        parts = token.split(",")
        items.append(
            {
                "amount": _to_float_or_none(parts[0]) if parts else None,
                "object": parts[1] if len(parts) > 1 else None,
                "offset": _to_int(parts[2]) if len(parts) > 2 else None,
                "raw": token,
            }
        )
    return items


def _parse_dates(value: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for token in _tokens(value):
        parts = token.split("#")
        items.append(
            {
                "date_type": parts[0] if parts else None,
                "date": parts[1] if len(parts) > 1 else None,
                "offset": _to_int(parts[2]) if len(parts) > 2 else None,
                "raw": token,
            }
        )
    return items


def _parse_gcam(value: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for token in [part.strip() for part in value.split(",") if part.strip()]:
        if ":" in token:
            dimension, score = token.split(":", 1)
        elif "=" in token:
            dimension, score = token.split("=", 1)
        else:
            dimension, score = token, ""
        items.append({"dimension": dimension, "score": _to_float_or_none(score), "raw": token})
    return items


def _parse_extras(value: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for token in _tokens(value):
        if "=" in token:
            key, raw_value = token.split("=", 1)
        elif ":" in token:
            key, raw_value = token.split(":", 1)
        else:
            key, raw_value = "", token
        items.append({"key": key, "value": raw_value, "raw": token})
    return items


def _published_at(extras: list[dict[str, Any]], gkg_date: str) -> str | None:
    for item in extras:
        key = str(item.get("key") or "")
        raw_value = str(item.get("value") or item.get("raw") or "")
        if key and not EXTRA_TIMESTAMP_KEYS.search(key):
            continue
        parsed = _parse_timestamp_candidate(raw_value)
        if parsed is not None:
            return parsed.isoformat()
    parsed_date = _parse_timestamp_candidate(gkg_date)
    return parsed_date.isoformat() if parsed_date is not None else None


def _parse_timestamp_candidate(value: str) -> datetime | None:
    clean = value.strip().replace("Z", "+00:00")
    compact_match = re.search(r"\b(\d{14})\b", clean)
    if compact_match:
        return _parse_gkg_timestamp(compact_match.group(1))
    date_match = re.search(r"\b(\d{8})\b", clean)
    if date_match:
        return _parse_gkg_timestamp(f"{date_match.group(1)}000000")
    try:
        parsed = datetime.fromisoformat(clean)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_gkg_timestamp(value: str) -> datetime:
    parsed = datetime.strptime(value[:14], "%Y%m%d%H%M%S")
    return parsed.replace(tzinfo=timezone.utc)


def _batch_stamp_from_path(path: Path) -> str:
    match = re.search(r"(\d{14})", path.name)
    if not match:
        raise ValueError(f"Could not infer GKG batch stamp from {path}")
    return match.group(1)


def _filter_paths_for_window(
    paths: list[Path],
    start_at: datetime | None,
    end_at: datetime | None,
) -> list[Path]:
    if start_at is None and end_at is None:
        return paths
    return [
        path
        for path in paths
        if _stamp_in_window(_parse_gkg_timestamp(_batch_stamp_from_path(path)), start_at, end_at)
    ]


def _stamp_in_window(stamp: datetime, start_at: datetime | None, end_at: datetime | None) -> bool:
    return (start_at is None or stamp >= start_at) and (end_at is None or stamp < end_at)


def _parse_window_bound(value: str | None, option_name: str) -> datetime | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        raise SystemExit(f"{option_name} must be a UTC timestamp in ISO-8601 or YYYYMMDDHHMMSS format")
    try:
        if re.fullmatch(r"\d{14}", text):
            return _parse_gkg_timestamp(text)
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit(f"{option_name} must be a UTC timestamp in ISO-8601 or YYYYMMDDHHMMSS format") from exc
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _validate_window(start_at: datetime | None, end_at: datetime | None) -> None:
    if start_at is not None and end_at is not None and end_at <= start_at:
        raise SystemExit("--end must be after --start")


def _weak_topics(text: str) -> list[str]:
    return [topic for topic, pattern in TOPIC_PATTERNS.items() if pattern.search(text)]


def _weak_entities(
    persons: list[dict[str, Any]],
    orgs: list[dict[str, Any]],
    locations: list[dict[str, Any]],
    all_names: list[dict[str, Any]],
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entity_type, rows, key in (
        ("person", persons, "name"),
        ("organization", orgs, "name"),
        ("location", locations, "full_name"),
        ("name", all_names, "name"),
    ):
        for row in rows:
            value = str(row.get(key) or "").strip()
            if not value:
                continue
            marker = (entity_type, value.lower())
            if marker in seen:
                continue
            seen.add(marker)
            candidates.append({"type": entity_type, "name": value})
    return candidates[:50]


def _tokens(value: str) -> list[str]:
    return [token.strip() for token in value.split(";") if token.strip()]


def _split_last_int(value: str, separator: str) -> tuple[str, int | None]:
    if separator not in value:
        return value, None
    name, maybe_int = value.rsplit(separator, 1)
    parsed = _to_int(maybe_int)
    if parsed is None:
        return value, None
    return name, parsed


def _cell(values: list[str], index: int) -> str:
    return values[index].strip() if index < len(values) else ""


def _raise_csv_field_size_limit() -> None:
    limit = CSV_FIELD_SIZE_LIMIT
    while limit > csv.field_size_limit():
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = int(limit / 10)


def _to_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _to_float_or_none(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())

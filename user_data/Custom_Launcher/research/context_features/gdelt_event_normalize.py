from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse
import argparse
import csv
import hashlib
import io
import json
import re
import sqlite3
import zipfile

from .builder import default_paths

try:
    from . import gdelt_gkg_normalized_schema
except Exception:  # pragma: no cover - exercised when shared schema is absent or mid-edit.
    gdelt_gkg_normalized_schema = None


DEFAULT_RAW_DIR = Path("D:/FreqTradeStuffLargeData/research_news_data/gdelt/raw/export")
GDELT_EXPORT_COLUMNS = 61
SCHEMA_VERSION = "gdelt_events_silver_v1"

COL_GLOBAL_EVENT_ID = 0
COL_SQL_DATE = 1
COL_ACTOR1_CODE = 5
COL_ACTOR1_NAME = 6
COL_ACTOR1_COUNTRY = 7
COL_ACTOR1_TYPE1 = 12
COL_ACTOR2_CODE = 15
COL_ACTOR2_NAME = 16
COL_ACTOR2_COUNTRY = 17
COL_ACTOR2_TYPE1 = 22
COL_IS_ROOT_EVENT = 25
COL_EVENT_CODE = 26
COL_EVENT_BASE_CODE = 27
COL_EVENT_ROOT_CODE = 28
COL_QUAD_CLASS = 29
COL_GOLDSTEIN = 30
COL_NUM_MENTIONS = 31
COL_NUM_SOURCES = 32
COL_NUM_ARTICLES = 33
COL_AVG_TONE = 34
COL_ACTION_GEO_COUNTRY = 53
COL_ACTION_GEO_LAT = 56
COL_ACTION_GEO_LONG = 57
COL_DATE_ADDED = 59
COL_SOURCE_URL = 60

SILVER_COLUMNS = [
    "global_event_id",
    "event_date",
    "available_at",
    "date_added",
    "file_stamp",
    "source_hostname",
    "source_url",
    "source_url_hash",
    "actor1_name",
    "actor1_country",
    "actor1_type",
    "actor2_name",
    "actor2_country",
    "actor2_type",
    "is_root_event",
    "event_code",
    "event_base_code",
    "event_root_code",
    "quad_class",
    "goldstein_scale",
    "num_mentions",
    "num_sources",
    "num_articles",
    "avg_tone",
    "action_geo_country",
    "action_geo_lat",
    "action_geo_long",
    "cameo_topic",
    "cameo_impact",
    "cameo_direction",
    "cameo_severity",
    "raw_ref_json",
    "ingested_at",
]

TOPIC_BY_ROOT = {
    "01": "public_statement",
    "02": "appeal",
    "03": "intent_to_cooperate",
    "04": "consultation",
    "05": "diplomatic_cooperation",
    "06": "material_cooperation",
    "07": "aid",
    "08": "yield_concession",
    "09": "investigation",
    "10": "demand",
    "11": "disapproval",
    "12": "rejection",
    "13": "threat",
    "14": "protest",
    "15": "force_posture",
    "16": "reduce_relations",
    "17": "coercion",
    "18": "assault",
    "19": "fight",
    "20": "mass_violence",
}

IMPACT_BY_QUAD_CLASS = {
    "1": "verbal_cooperation",
    "2": "material_cooperation",
    "3": "verbal_conflict",
    "4": "material_conflict",
}


@dataclass
class NormalizeSummary:
    db_path: str
    raw_dir: str
    dry_run: bool
    files_planned: int
    files_read: int = 0
    rows_read: int = 0
    rows_parsed: int = 0
    rows_inserted: int = 0
    rows_skipped: int = 0
    failures: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failures"] = self.failures or []
        return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize raw GDELT event export ZIPs into gdelt_events_silver.")
    parser.add_argument("--app-dir", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--file-glob", default="*.export.CSV.zip")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--start", default=None, help="Inclusive UTC file-stamp lower bound.")
    parser.add_argument("--end", default=None, help="Exclusive UTC file-stamp upper bound.")
    args = parser.parse_args()

    if args.limit_files is not None and args.limit_files < 0:
        raise SystemExit("--limit-files must be non-negative")
    if args.limit_rows is not None and args.limit_rows < 0:
        raise SystemExit("--limit-rows must be non-negative")
    start_at = _parse_window_bound(args.start, "--start")
    end_at = _parse_window_bound(args.end, "--end")
    _validate_window(start_at, end_at)

    paths = default_paths(args.app_dir)
    db_path = args.db or paths.gdelt_db
    raw_dir = args.raw_dir or DEFAULT_RAW_DIR
    files = sorted(raw_dir.glob(args.file_glob)) if raw_dir.exists() else []
    files = _filter_paths_for_window(files, start_at, end_at)
    if args.limit_files is not None:
        files = files[: args.limit_files]

    summary = NormalizeSummary(
        db_path=str(db_path),
        raw_dir=str(raw_dir),
        dry_run=bool(args.dry_run),
        files_planned=len(files),
    )
    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    remaining_rows = args.limit_rows

    for zip_path in files:
        if remaining_rows == 0:
            break
        try:
            parsed_rows, rows_read, rows_skipped = parse_gdelt_export_zip(
                zip_path,
                limit_rows=remaining_rows,
            )
        except Exception as exc:
            failures.append(f"{zip_path}: {exc}")
            continue
        summary.files_read += 1
        summary.rows_read += rows_read
        summary.rows_skipped += rows_skipped
        summary.rows_parsed += len(parsed_rows)
        rows.extend(parsed_rows)
        if remaining_rows is not None:
            remaining_rows = max(0, remaining_rows - len(parsed_rows))

    if not args.dry_run and rows:
        with connect_db(db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                ensure_gdelt_events_silver(conn)
                summary.rows_inserted = insert_gdelt_events_silver(conn, rows)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    summary.failures = failures[:20]
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0 if not failures else 2


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def ensure_gdelt_events_silver(conn: sqlite3.Connection) -> None:
    if gdelt_gkg_normalized_schema is not None:
        for attr in ("init_schema", "ensure_gdelt_events_silver", "ensure_normalized_schema"):
            schema_func = getattr(gdelt_gkg_normalized_schema, attr, None)
            if schema_func is None:
                continue
            schema_func(conn)
            return
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS gdelt_events_silver (
            global_event_id TEXT PRIMARY KEY,
            event_date TEXT,
            available_at TEXT NOT NULL,
            date_added TEXT,
            file_stamp TEXT,
            source_hostname TEXT,
            source_url TEXT,
            source_url_hash TEXT,
            actor1_name TEXT,
            actor1_country TEXT,
            actor1_type TEXT,
            actor2_name TEXT,
            actor2_country TEXT,
            actor2_type TEXT,
            is_root_event INTEGER,
            event_code TEXT,
            event_base_code TEXT,
            event_root_code TEXT,
            quad_class INTEGER,
            goldstein_scale REAL,
            num_mentions REAL,
            num_sources REAL,
            num_articles REAL,
            avg_tone REAL,
            action_geo_country TEXT,
            action_geo_lat REAL,
            action_geo_long REAL,
            cameo_topic TEXT,
            cameo_impact TEXT,
            cameo_direction TEXT,
            cameo_severity REAL,
            raw_ref_json TEXT NOT NULL,
            ingested_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_gdelt_events_silver_available_at
            ON gdelt_events_silver(available_at);
        CREATE INDEX IF NOT EXISTS idx_gdelt_events_silver_source_hash
            ON gdelt_events_silver(source_url_hash);
        CREATE INDEX IF NOT EXISTS idx_gdelt_events_silver_topic_available
            ON gdelt_events_silver(cameo_topic, available_at);
        """
    )
    conn.execute("INSERT OR REPLACE INTO schema_meta(key, value) VALUES(?, ?)", ("gdelt_events_silver", SCHEMA_VERSION))


def insert_gdelt_events_silver(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> int:
    table_columns = _table_columns(conn, "gdelt_events_silver")
    if "event_id" in table_columns and "payload_json" in table_columns:
        return _insert_shared_schema_rows(conn, rows)
    payload = [tuple(row.get(column) for column in SILVER_COLUMNS) for row in rows]
    if not payload:
        return 0
    placeholders = ", ".join("?" for _ in SILVER_COLUMNS)
    columns_sql = ", ".join(SILVER_COLUMNS)
    conn.executemany(
        f"""
        INSERT OR REPLACE INTO gdelt_events_silver ({columns_sql})
        VALUES ({placeholders})
        """,
        payload,
    )
    return len(payload)


def _insert_shared_schema_rows(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> int:
    events: list[tuple[Any, ...]] = []
    raw_files: dict[str, tuple[Any, ...]] = {}
    ingested_at = datetime.now(timezone.utc).isoformat()
    for row in rows:
        raw_ref = json.loads(row.get("raw_ref_json") or "{}")
        raw_file_id = _raw_file_id(row, raw_ref)
        raw_files[raw_file_id] = (
            raw_file_id,
            "export",
            _compact_stamp(row.get("file_stamp") or row.get("available_at")),
            row.get("source_url"),
            row.get("source_url"),
            raw_ref.get("raw_file"),
            Path(raw_ref["raw_file"]).name if raw_ref.get("raw_file") else None,
            None,
            None,
            None,
            None,
            None,
            ingested_at,
            "parsed",
            None,
            json.dumps({"parser": "gdelt_event_normalize"}, sort_keys=True),
        )
        payload = {
            "available_at": row.get("available_at"),
            "date_added": row.get("date_added"),
            "file_stamp": row.get("file_stamp"),
            "source_hostname": row.get("source_hostname"),
            "actor1_name": row.get("actor1_name"),
            "actor1_country": row.get("actor1_country"),
            "actor1_type": row.get("actor1_type"),
            "actor2_name": row.get("actor2_name"),
            "actor2_country": row.get("actor2_country"),
            "actor2_type": row.get("actor2_type"),
            "is_root_event": row.get("is_root_event"),
            "action_geo_country": row.get("action_geo_country"),
            "action_geo_lat": row.get("action_geo_lat"),
            "action_geo_long": row.get("action_geo_long"),
            "cameo_topic": row.get("cameo_topic"),
            "cameo_impact": row.get("cameo_impact"),
            "cameo_direction": row.get("cameo_direction"),
            "cameo_severity": row.get("cameo_severity"),
            "raw_ref": raw_ref,
        }
        events.append(
            (
                _event_id(row),
                raw_file_id,
                row.get("global_event_id"),
                row.get("available_at"),
                row.get("available_at"),
                row.get("event_date"),
                row.get("actor1_code"),
                row.get("actor1_name"),
                row.get("actor1_country"),
                row.get("actor2_code"),
                row.get("actor2_name"),
                row.get("actor2_country"),
                row.get("event_code"),
                row.get("event_base_code"),
                row.get("event_root_code"),
                row.get("quad_class"),
                row.get("goldstein_scale"),
                _none_or_int(row.get("num_mentions")),
                _none_or_int(row.get("num_sources")),
                _none_or_int(row.get("num_articles")),
                row.get("avg_tone"),
                row.get("source_url"),
                row.get("source_url"),
                row.get("source_url_hash"),
                row.get("action_geo_country"),
                row.get("action_geo_lat"),
                row.get("action_geo_long"),
                row.get("cameo_topic"),
                row.get("cameo_impact"),
                row.get("cameo_direction"),
                row.get("cameo_severity"),
                row.get("raw_ref_json"),
                "gdelt_event_normalize_v1",
                "cameo_root_quad_v1",
                "unset",
                "unset",
                "unset",
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                ingested_at,
                ingested_at,
            )
        )
    if not events:
        return 0
    conn.executemany(
        """
        INSERT OR REPLACE INTO gdelt_raw_files (
            raw_file_id, file_kind, gdelt_stamp, source_url, canonical_source_url, local_path, file_name,
            file_sha256, compressed_bytes, uncompressed_bytes, row_count, fetched_at, parsed_at, status,
            parse_error, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        list(raw_files.values()),
    )
    conn.executemany(
        """
        INSERT OR REPLACE INTO gdelt_events_silver (
            event_id, raw_file_id, global_event_id, event_time, available_at, event_date,
            actor1_code, actor1_name, actor1_country, actor2_code, actor2_name, actor2_country,
            event_code, event_base_code, event_root_code, quad_class, goldstein_scale, num_mentions,
            num_sources, num_articles, avg_tone, source_url, canonical_source_url, source_url_hash,
            action_geo_country, action_geo_lat, action_geo_lon, topic, impact_channel, direction,
            severity_proxy, raw_ref_json, parser_version, taxonomy_version, entity_model_version,
            classifier_version, source_config_hash, payload_json, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        events,
    )
    return len(events)


def parse_gdelt_export_zip(zip_path: Path, *, limit_rows: int | None = None) -> tuple[list[dict[str, Any]], int, int]:
    file_stamp = file_stamp_from_path(zip_path)
    rows: list[dict[str, Any]] = []
    rows_read = 0
    rows_skipped = 0
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if not members:
            raise ValueError("empty zip")
        for member in members:
            with archive.open(member, "r") as handle:
                reader = csv.reader(io.TextIOWrapper(handle, encoding="utf-8", errors="replace"), delimiter="\t")
                for values in reader:
                    if limit_rows is not None and len(rows) >= limit_rows:
                        return rows, rows_read, rows_skipped
                    rows_read += 1
                    try:
                        rows.append(
                            parse_gdelt_export_row(
                                values,
                                file_stamp=file_stamp,
                                raw_file=str(zip_path),
                                file_member=member,
                                row_number=rows_read,
                            )
                        )
                    except ValueError:
                        rows_skipped += 1
    return rows, rows_read, rows_skipped


def parse_gdelt_export_tsv_line(
    line: str,
    *,
    file_stamp: datetime | str | None = None,
    raw_file: str | None = None,
    file_member: str | None = None,
    row_number: int | None = None,
) -> dict[str, Any]:
    values = next(csv.reader([line], delimiter="\t"))
    return parse_gdelt_export_row(
        values,
        file_stamp=file_stamp,
        raw_file=raw_file,
        file_member=file_member,
        row_number=row_number,
    )


def parse_gdelt_export_row(
    values: list[str],
    *,
    file_stamp: datetime | str | None = None,
    raw_file: str | None = None,
    file_member: str | None = None,
    row_number: int | None = None,
) -> dict[str, Any]:
    if len(values) < GDELT_EXPORT_COLUMNS:
        raise ValueError(f"expected at least {GDELT_EXPORT_COLUMNS} GDELT export columns, got {len(values)}")

    date_added = _parse_gdelt_timestamp(_value(values, COL_DATE_ADDED))
    fallback_stamp = _coerce_stamp(file_stamp)
    available_at = safe_available_at(date_added, fallback_stamp)
    if available_at is None:
        raise ValueError("missing safe available_at; DATEADDED and file stamp are both unavailable")

    event_root = _value(values, COL_EVENT_ROOT_CODE)
    quad_class = _value(values, COL_QUAD_CLASS)
    goldstein = _to_float(_value(values, COL_GOLDSTEIN))
    source_url = _value(values, COL_SOURCE_URL)
    source_hostname = hostname_from_url(source_url)
    event_date = _parse_sql_date(_value(values, COL_SQL_DATE))
    cameo_fields = derive_cameo_fields(event_root, quad_class, goldstein)
    ingested_at = datetime.now(timezone.utc).isoformat()
    raw_ref = {
        "raw_file": raw_file,
        "file_member": file_member,
        "row_number": row_number,
        "column_count": len(values),
        "global_event_id": _value(values, COL_GLOBAL_EVENT_ID),
        "sql_date": _value(values, COL_SQL_DATE),
        "date_added": _value(values, COL_DATE_ADDED),
        "source_url": source_url,
    }

    return {
        "global_event_id": _value(values, COL_GLOBAL_EVENT_ID),
        "event_date": event_date.isoformat() if event_date is not None else None,
        "available_at": available_at.isoformat(),
        "date_added": date_added.isoformat() if date_added is not None else None,
        "file_stamp": fallback_stamp.isoformat() if fallback_stamp is not None else None,
        "source_hostname": source_hostname,
        "source_url": source_url,
        "source_url_hash": url_hash(source_url),
        "actor1_code": _value(values, COL_ACTOR1_CODE) or None,
        "actor1_name": _value(values, COL_ACTOR1_NAME) or None,
        "actor1_country": _value(values, COL_ACTOR1_COUNTRY) or None,
        "actor1_type": _value(values, COL_ACTOR1_TYPE1) or None,
        "actor2_code": _value(values, COL_ACTOR2_CODE) or None,
        "actor2_name": _value(values, COL_ACTOR2_NAME) or None,
        "actor2_country": _value(values, COL_ACTOR2_COUNTRY) or None,
        "actor2_type": _value(values, COL_ACTOR2_TYPE1) or None,
        "is_root_event": _to_int(_value(values, COL_IS_ROOT_EVENT)),
        "event_code": _value(values, COL_EVENT_CODE) or None,
        "event_base_code": _value(values, COL_EVENT_BASE_CODE) or None,
        "event_root_code": event_root or None,
        "quad_class": _to_int(quad_class),
        "goldstein_scale": goldstein,
        "num_mentions": _to_float(_value(values, COL_NUM_MENTIONS)),
        "num_sources": _to_float(_value(values, COL_NUM_SOURCES)),
        "num_articles": _to_float(_value(values, COL_NUM_ARTICLES)),
        "avg_tone": _to_float(_value(values, COL_AVG_TONE)),
        "action_geo_country": _value(values, COL_ACTION_GEO_COUNTRY) or None,
        "action_geo_lat": _to_float(_value(values, COL_ACTION_GEO_LAT)),
        "action_geo_long": _to_float(_value(values, COL_ACTION_GEO_LONG)),
        "cameo_topic": cameo_fields["cameo_topic"],
        "cameo_impact": cameo_fields["cameo_impact"],
        "cameo_direction": cameo_fields["cameo_direction"],
        "cameo_severity": cameo_fields["cameo_severity"],
        "raw_ref_json": json.dumps(raw_ref, sort_keys=True, separators=(",", ":")),
        "ingested_at": ingested_at,
    }


def safe_available_at(date_added: datetime | str | None, file_stamp: datetime | str | None = None) -> datetime | None:
    parsed_date_added = _coerce_stamp(date_added)
    if parsed_date_added is not None:
        return parsed_date_added
    return _coerce_stamp(file_stamp)


def derive_cameo_fields(event_root: str, quad_class: str, goldstein: float | None) -> dict[str, Any]:
    root = event_root.zfill(2) if event_root else ""
    quad = str(quad_class).strip()
    topic = TOPIC_BY_ROOT.get(root, "unknown")
    impact = IMPACT_BY_QUAD_CLASS.get(quad, "unknown")

    if quad in {"3", "4"} or (goldstein is not None and goldstein < -1.0):
        direction = "risk_off"
    elif quad in {"1", "2"} or (goldstein is not None and goldstein > 1.0):
        direction = "risk_on"
    else:
        direction = "neutral"

    base = abs(goldstein or 0.0) / 10.0
    if quad == "4":
        base = max(base, 0.75)
    elif quad == "3":
        base = max(base, 0.45)
    elif quad == "2":
        base = max(base, 0.25)
    severity = max(0.0, min(1.0, base))

    return {
        "cameo_topic": topic,
        "cameo_impact": impact,
        "cameo_direction": direction,
        "cameo_severity": severity,
    }


def file_stamp_from_path(path: Path) -> datetime | None:
    match = re.search(r"(\d{14})", path.name)
    if not match:
        return None
    return _parse_gdelt_timestamp(match.group(1))


def _filter_paths_for_window(
    paths: list[Path],
    start_at: datetime | None,
    end_at: datetime | None,
) -> list[Path]:
    if start_at is None and end_at is None:
        return paths
    filtered: list[Path] = []
    for path in paths:
        stamp = file_stamp_from_path(path)
        if stamp is None:
            raise SystemExit(f"Could not infer GDELT file stamp from {path}")
        if _stamp_in_window(stamp, start_at, end_at):
            filtered.append(path)
    return filtered


def _stamp_in_window(stamp: datetime, start_at: datetime | None, end_at: datetime | None) -> bool:
    return (start_at is None or stamp >= start_at) and (end_at is None or stamp < end_at)


def _parse_window_bound(value: str | None, option_name: str) -> datetime | None:
    if value is None:
        return None
    parsed = _coerce_stamp(value)
    if parsed is None:
        raise SystemExit(f"{option_name} must be a UTC timestamp in ISO-8601 or YYYYMMDDHHMMSS format")
    return parsed


def _validate_window(start_at: datetime | None, end_at: datetime | None) -> None:
    if start_at is not None and end_at is not None and end_at <= start_at:
        raise SystemExit("--end must be after --start")


def hostname_from_url(url: str) -> str | None:
    if not url:
        return None
    parsed = urlparse(url if "://" in url else f"https://{url}")
    hostname = parsed.hostname
    return hostname.lower() if hostname else None


def url_hash(url: str) -> str | None:
    if not url:
        return None
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}


def _raw_file_id(row: dict[str, Any], raw_ref: dict[str, Any]) -> str:
    basis = raw_ref.get("raw_file") or row.get("file_stamp") or row.get("available_at") or "unknown"
    return hashlib.sha256(str(basis).encode("utf-8")).hexdigest()


def _event_id(row: dict[str, Any]) -> str:
    if row.get("global_event_id"):
        return f"event:{row['global_event_id']}"
    basis = "|".join(str(row.get(key) or "") for key in ("available_at", "source_url_hash", "event_code", "raw_ref_json"))
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _compact_stamp(value: str | None) -> str:
    parsed = _coerce_stamp(value)
    if parsed is None:
        return "unknown"
    return parsed.strftime("%Y%m%d%H%M%S")


def _parse_sql_date(value: str) -> datetime | None:
    text = value.strip()
    if not re.fullmatch(r"\d{8}", text):
        return None
    return datetime.strptime(text, "%Y%m%d").replace(tzinfo=timezone.utc)


def _parse_gdelt_timestamp(value: str) -> datetime | None:
    text = value.strip()
    if not re.fullmatch(r"\d{14}", text):
        return None
    return datetime.strptime(text, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def _coerce_stamp(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = value.strip()
    if not text:
        return None
    gdelt_stamp = _parse_gdelt_timestamp(text)
    if gdelt_stamp is not None:
        return gdelt_stamp
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _value(values: list[str], index: int) -> str:
    return values[index].strip() if index < len(values) else ""


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _none_or_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())

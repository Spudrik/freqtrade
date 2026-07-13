from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json
import sqlite3
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit


SCHEMA_VERSION = "1"
GOLD_TABLE_NAME = "gdelt_gkg_hourly_features_v1"

_TRACKING_QUERY_PREFIXES = ("utm_",)
_TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
}


def connect_db(db_path: Path | str) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: Path | str) -> None:
    with connect_db(db_path) as conn:
        init_schema(conn)


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS gdelt_raw_files (
            raw_file_id TEXT PRIMARY KEY,
            file_kind TEXT NOT NULL,
            gdelt_stamp TEXT NOT NULL,
            source_url TEXT,
            canonical_source_url TEXT,
            local_path TEXT,
            file_name TEXT,
            file_sha256 TEXT,
            compressed_bytes INTEGER,
            uncompressed_bytes INTEGER,
            row_count INTEGER,
            fetched_at TEXT,
            parsed_at TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            parse_error TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{{}}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(file_kind, gdelt_stamp, canonical_source_url, local_path)
        );

        CREATE INDEX IF NOT EXISTS idx_gdelt_raw_files_kind_stamp
            ON gdelt_raw_files(file_kind, gdelt_stamp);
        CREATE INDEX IF NOT EXISTS idx_gdelt_raw_files_status
            ON gdelt_raw_files(status);
        CREATE INDEX IF NOT EXISTS idx_gdelt_raw_files_sha256
            ON gdelt_raw_files(file_sha256);

        CREATE TABLE IF NOT EXISTS gdelt_events_silver (
            event_id TEXT PRIMARY KEY,
            raw_file_id TEXT NOT NULL,
            global_event_id TEXT,
            event_time TEXT NOT NULL,
            available_at TEXT NOT NULL,
            event_date TEXT,
            actor1_code TEXT,
            actor1_name TEXT,
            actor1_country TEXT,
            actor2_code TEXT,
            actor2_name TEXT,
            actor2_country TEXT,
            event_code TEXT,
            event_base_code TEXT,
            event_root_code TEXT,
            quad_class INTEGER,
            goldstein_scale REAL,
            num_mentions INTEGER,
            num_sources INTEGER,
            num_articles INTEGER,
            avg_tone REAL,
            source_url TEXT,
            canonical_source_url TEXT,
            source_url_hash TEXT,
            action_geo_country TEXT,
            action_geo_lat REAL,
            action_geo_lon REAL,
            topic TEXT,
            subtopic TEXT,
            impact_channel TEXT,
            direction TEXT,
            severity_proxy REAL,
            confidence REAL,
            entities_json TEXT NOT NULL DEFAULT '[]',
            assets_json TEXT NOT NULL DEFAULT '[]',
            evidence_json TEXT NOT NULL DEFAULT '{{}}',
            raw_ref_json TEXT NOT NULL DEFAULT '{{}}',
            parser_version TEXT NOT NULL DEFAULT 'unset',
            taxonomy_version TEXT NOT NULL DEFAULT 'unset',
            entity_model_version TEXT NOT NULL DEFAULT 'unset',
            classifier_version TEXT NOT NULL DEFAULT 'unset',
            source_config_hash TEXT NOT NULL DEFAULT 'unset',
            normalized_schema_version TEXT NOT NULL DEFAULT '{SCHEMA_VERSION}',
            content_hash TEXT,
            duplicate_group TEXT,
            story_id TEXT,
            payload_json TEXT NOT NULL DEFAULT '{{}}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(raw_file_id) REFERENCES gdelt_raw_files(raw_file_id)
        );

        CREATE INDEX IF NOT EXISTS idx_gdelt_events_silver_raw_file
            ON gdelt_events_silver(raw_file_id);
        CREATE INDEX IF NOT EXISTS idx_gdelt_events_silver_event_time
            ON gdelt_events_silver(event_time);
        CREATE INDEX IF NOT EXISTS idx_gdelt_events_silver_available_at
            ON gdelt_events_silver(available_at);
        CREATE INDEX IF NOT EXISTS idx_gdelt_events_silver_url_hash
            ON gdelt_events_silver(source_url_hash);
        CREATE INDEX IF NOT EXISTS idx_gdelt_events_silver_codes
            ON gdelt_events_silver(event_root_code, event_base_code, event_code);
        CREATE INDEX IF NOT EXISTS idx_gdelt_events_silver_topic_available
            ON gdelt_events_silver(topic, available_at);
        CREATE INDEX IF NOT EXISTS idx_gdelt_events_silver_story
            ON gdelt_events_silver(story_id);

        CREATE TABLE IF NOT EXISTS gkg_documents_silver (
            document_id TEXT PRIMARY KEY,
            raw_file_id TEXT NOT NULL,
            gkg_record_id TEXT,
            document_time TEXT NOT NULL,
            available_at TEXT NOT NULL,
            published_at TEXT,
            source_common_name TEXT,
            source_group TEXT,
            source_quality REAL,
            source_collection_identifier TEXT,
            document_url TEXT,
            canonical_url TEXT,
            url_hash TEXT,
            title TEXT,
            summary TEXT,
            body_excerpt TEXT,
            themes_json TEXT NOT NULL DEFAULT '[]',
            v2_themes_json TEXT NOT NULL DEFAULT '[]',
            persons_json TEXT NOT NULL DEFAULT '[]',
            orgs_json TEXT NOT NULL DEFAULT '[]',
            locations_json TEXT NOT NULL DEFAULT '[]',
            all_names_json TEXT NOT NULL DEFAULT '[]',
            amounts_json TEXT NOT NULL DEFAULT '[]',
            counts_json TEXT NOT NULL DEFAULT '[]',
            dates_json TEXT NOT NULL DEFAULT '[]',
            gcam_json TEXT NOT NULL DEFAULT '{{}}',
            extras_json TEXT NOT NULL DEFAULT '{{}}',
            tone REAL,
            positive_tone REAL,
            negative_tone REAL,
            polarity REAL,
            activity_density REAL,
            self_group_reference_density REAL,
            word_count REAL,
            topic TEXT,
            subtopic TEXT,
            impact_channel TEXT,
            direction TEXT,
            severity_proxy REAL,
            confidence REAL,
            entities_json TEXT NOT NULL DEFAULT '[]',
            assets_json TEXT NOT NULL DEFAULT '[]',
            evidence_json TEXT NOT NULL DEFAULT '{{}}',
            raw_ref_json TEXT NOT NULL DEFAULT '{{}}',
            parser_version TEXT NOT NULL DEFAULT 'unset',
            taxonomy_version TEXT NOT NULL DEFAULT 'unset',
            entity_model_version TEXT NOT NULL DEFAULT 'unset',
            classifier_version TEXT NOT NULL DEFAULT 'unset',
            source_config_hash TEXT NOT NULL DEFAULT 'unset',
            normalized_schema_version TEXT NOT NULL DEFAULT '{SCHEMA_VERSION}',
            content_hash TEXT,
            duplicate_group TEXT,
            story_id TEXT,
            payload_json TEXT NOT NULL DEFAULT '{{}}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(raw_file_id) REFERENCES gdelt_raw_files(raw_file_id)
        );

        CREATE INDEX IF NOT EXISTS idx_gkg_documents_silver_raw_file
            ON gkg_documents_silver(raw_file_id);
        CREATE INDEX IF NOT EXISTS idx_gkg_documents_silver_document_time
            ON gkg_documents_silver(document_time);
        CREATE INDEX IF NOT EXISTS idx_gkg_documents_silver_available_at
            ON gkg_documents_silver(available_at);
        CREATE INDEX IF NOT EXISTS idx_gkg_documents_silver_url_hash
            ON gkg_documents_silver(url_hash);
        CREATE INDEX IF NOT EXISTS idx_gkg_documents_silver_source
            ON gkg_documents_silver(source_common_name);
        CREATE INDEX IF NOT EXISTS idx_gkg_documents_silver_topic_available
            ON gkg_documents_silver(topic, available_at);
        CREATE INDEX IF NOT EXISTS idx_gkg_documents_silver_story
            ON gkg_documents_silver(story_id);

        CREATE TABLE IF NOT EXISTS story_clusters (
            story_id TEXT PRIMARY KEY,
            cluster_key TEXT NOT NULL,
            canonical_url_hash TEXT,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            representative_url TEXT,
            representative_title TEXT,
            topic_tags_json TEXT NOT NULL DEFAULT '[]',
            document_count INTEGER NOT NULL DEFAULT 0,
            event_count INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{{}}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_story_clusters_cluster_key
            ON story_clusters(cluster_key);
        CREATE INDEX IF NOT EXISTS idx_story_clusters_seen
            ON story_clusters(first_seen_at, last_seen_at);
        CREATE INDEX IF NOT EXISTS idx_story_clusters_url_hash
            ON story_clusters(canonical_url_hash);

        CREATE TABLE IF NOT EXISTS story_members (
            story_id TEXT NOT NULL,
            member_type TEXT NOT NULL,
            member_id TEXT NOT NULL,
            raw_file_id TEXT,
            observed_at TEXT NOT NULL,
            relation_score REAL,
            metadata_json TEXT NOT NULL DEFAULT '{{}}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(story_id, member_type, member_id),
            FOREIGN KEY(story_id) REFERENCES story_clusters(story_id),
            FOREIGN KEY(raw_file_id) REFERENCES gdelt_raw_files(raw_file_id)
        );

        CREATE INDEX IF NOT EXISTS idx_story_members_member
            ON story_members(member_type, member_id);
        CREATE INDEX IF NOT EXISTS idx_story_members_raw_file
            ON story_members(raw_file_id);
        CREATE INDEX IF NOT EXISTS idx_story_members_observed_at
            ON story_members(observed_at);

        CREATE TABLE IF NOT EXISTS {GOLD_TABLE_NAME} (
            feature_hour TEXT NOT NULL,
            feature_namespace TEXT NOT NULL DEFAULT 'gdelt_gkg',
            source_file_count INTEGER NOT NULL DEFAULT 0,
            document_count INTEGER NOT NULL DEFAULT 0,
            event_count INTEGER NOT NULL DEFAULT 0,
            story_count INTEGER NOT NULL DEFAULT 0,
            source_count INTEGER NOT NULL DEFAULT 0,
            avg_tone REAL,
            avg_goldstein REAL,
            conflict_event_count INTEGER NOT NULL DEFAULT 0,
            macro_doc_count INTEGER NOT NULL DEFAULT 0,
            crypto_doc_count INTEGER NOT NULL DEFAULT 0,
            geopolitics_doc_count INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{{}}',
            schema_version TEXT NOT NULL DEFAULT '{SCHEMA_VERSION}',
            generated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(feature_hour, feature_namespace)
        );

        CREATE INDEX IF NOT EXISTS idx_gdelt_gkg_hourly_features_v1_generated_at
            ON {GOLD_TABLE_NAME}(generated_at);
        CREATE INDEX IF NOT EXISTS idx_gdelt_gkg_hourly_features_v1_schema_version
            ON {GOLD_TABLE_NAME}(schema_version);
        """
    )
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES(?, ?)",
        ("gdelt_gkg_normalized_schema_version", SCHEMA_VERSION),
    )


def stable_hash(*parts: object, length: int | None = None) -> str:
    payload = "\x1f".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest if length is None else digest[:length]


def stable_id(prefix: str, *parts: object, length: int = 32) -> str:
    clean_prefix = prefix.strip().lower().replace(" ", "_")
    if not clean_prefix:
        raise ValueError("prefix is required")
    return f"{clean_prefix}_{stable_hash(*parts, length=length)}"


def canonicalize_url(url: str | None) -> str | None:
    if url is None:
        return None
    raw = url.strip()
    if not raw:
        return None

    parsed = urlsplit(raw)
    if not parsed.netloc and "." in parsed.path.split("/", 1)[0]:
        parsed = urlsplit(f"//{raw}")
    scheme = (parsed.scheme or "http").lower()
    hostname = (parsed.hostname or "").lower()
    if hostname.startswith("www."):
        hostname = hostname[4:]

    netloc = hostname
    if parsed.port and not ((scheme == "http" and parsed.port == 80) or (scheme == "https" and parsed.port == 443)):
        netloc = f"{netloc}:{parsed.port}"

    path = quote(unquote(parsed.path or "/"), safe="/:@")
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not _is_tracking_query_key(key)
    ]
    query = urlencode(sorted(query_items), doseq=True)
    return urlunsplit((scheme, netloc, path, query, ""))


def url_hash(url: str | None) -> str | None:
    canonical = canonicalize_url(url)
    if canonical is None:
        return None
    return stable_hash("url", canonical)


def raw_file_id(file_kind: str, gdelt_stamp: str, source_url: str | None = None, local_path: str | Path | None = None) -> str:
    canonical_url = canonicalize_url(source_url)
    local_path_text = str(local_path) if local_path is not None else None
    return stable_id("raw", file_kind.strip().lower(), gdelt_stamp.strip(), canonical_url, local_path_text)


def upsert_raw_file_metadata(
    conn: sqlite3.Connection,
    *,
    file_kind: str,
    gdelt_stamp: str,
    source_url: str | None = None,
    local_path: str | Path | None = None,
    file_sha256: str | None = None,
    compressed_bytes: int | None = None,
    uncompressed_bytes: int | None = None,
    row_count: int | None = None,
    fetched_at: str | None = None,
    parsed_at: str | None = None,
    status: str = "pending",
    parse_error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    clean_kind = file_kind.strip().lower()
    clean_stamp = gdelt_stamp.strip()
    if not clean_kind:
        raise ValueError("file_kind is required")
    if not clean_stamp:
        raise ValueError("gdelt_stamp is required")

    local_path_text = str(local_path) if local_path is not None else None
    file_name = Path(local_path_text).name if local_path_text else None
    canonical_url = canonicalize_url(source_url)
    metadata_json = json.dumps(metadata or {}, sort_keys=True, separators=(",", ":"))
    raw_id = raw_file_id(clean_kind, clean_stamp, source_url, local_path_text)

    conn.execute(
        """
        INSERT INTO gdelt_raw_files (
            raw_file_id, file_kind, gdelt_stamp, source_url, canonical_source_url,
            local_path, file_name, file_sha256, compressed_bytes, uncompressed_bytes,
            row_count, fetched_at, parsed_at, status, parse_error, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(raw_file_id) DO UPDATE SET
            source_url = excluded.source_url,
            canonical_source_url = excluded.canonical_source_url,
            local_path = excluded.local_path,
            file_name = excluded.file_name,
            file_sha256 = excluded.file_sha256,
            compressed_bytes = excluded.compressed_bytes,
            uncompressed_bytes = excluded.uncompressed_bytes,
            row_count = excluded.row_count,
            fetched_at = excluded.fetched_at,
            parsed_at = excluded.parsed_at,
            status = excluded.status,
            parse_error = excluded.parse_error,
            metadata_json = excluded.metadata_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            raw_id,
            clean_kind,
            clean_stamp,
            source_url,
            canonical_url,
            local_path_text,
            file_name,
            file_sha256,
            compressed_bytes,
            uncompressed_bytes,
            row_count,
            fetched_at,
            parsed_at,
            status,
            parse_error,
            metadata_json,
        ),
    )
    return raw_id


def list_raw_file_metadata(
    conn: sqlite3.Connection,
    *,
    file_kind: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    where: list[str] = []
    params: list[str] = []
    if file_kind is not None:
        where.append("file_kind = ?")
        params.append(file_kind.strip().lower())
    if status is not None:
        where.append("status = ?")
        params.append(status)

    sql = "SELECT * FROM gdelt_raw_files"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY gdelt_stamp, file_kind, raw_file_id"
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def _is_tracking_query_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in _TRACKING_QUERY_KEYS or lowered.startswith(_TRACKING_QUERY_PREFIXES)


__all__ = [
    "GOLD_TABLE_NAME",
    "SCHEMA_VERSION",
    "canonicalize_url",
    "connect_db",
    "init_db",
    "init_schema",
    "list_raw_file_metadata",
    "raw_file_id",
    "stable_hash",
    "stable_id",
    "upsert_raw_file_metadata",
    "url_hash",
]

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "2"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: Path) -> None:
    conn = connect_db(db_path)
    try:
        conn.executescript(
            f"""
            BEGIN IMMEDIATE;

            CREATE TABLE IF NOT EXISTS articles (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                source_group TEXT,
                source_type TEXT NOT NULL,
                region TEXT,
                topic TEXT,
                market_relevance TEXT,
                source_url TEXT,
                canonical_url TEXT,
                title TEXT NOT NULL,
                summary TEXT,
                published_at TEXT,
                collected_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                language TEXT,
                detected_assets_json TEXT,
                detected_entities_json TEXT,
                event_type TEXT,
                impact_direction TEXT,
                impact_scope TEXT,
                severity INTEGER,
                confidence INTEGER,
                raw_json TEXT NOT NULL,
                raw_file_path TEXT
            );

            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                source_group TEXT,
                source_type TEXT,
                enabled INTEGER,
                url_or_query TEXT,
                region TEXT,
                topic TEXT,
                market_relevance TEXT,
                last_success_at TEXT,
                last_failure_at TEXT,
                last_error TEXT,
                last_http_status INTEGER,
                items_last_fetch INTEGER DEFAULT 0,
                inserted_last_fetch INTEGER DEFAULT 0,
                duplicates_last_fetch INTEGER DEFAULT 0,
                avg_items_per_day REAL
            );

            CREATE TABLE IF NOT EXISTS source_fetch_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                fetched_count INTEGER DEFAULT 0,
                inserted_count INTEGER DEFAULT 0,
                duplicate_count INTEGER DEFAULT 0,
                error_message TEXT
            );

            CREATE TABLE IF NOT EXISTS collector_runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                stopped_at TEXT,
                status TEXT NOT NULL,
                pid INTEGER,
                config_path TEXT,
                db_path TEXT,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS article_categories (
                article_id TEXT,
                category TEXT,
                confidence REAL,
                PRIMARY KEY(article_id, category)
            );

            CREATE TABLE IF NOT EXISTS article_assets (
                article_id TEXT,
                asset TEXT,
                confidence REAL,
                PRIMARY KEY(article_id, asset)
            );

            CREATE TABLE IF NOT EXISTS article_forward_returns (
                article_id TEXT,
                asset TEXT,
                horizon TEXT,
                return_pct REAL,
                max_favorable_pct REAL,
                max_adverse_pct REAL,
                calculated_at TEXT,
                PRIMARY KEY(article_id, asset, horizon)
            );

            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace TEXT NOT NULL,
                value TEXT NOT NULL,
                UNIQUE(namespace, value)
            );

            CREATE TABLE IF NOT EXISTS source_tags (
                source_id TEXT NOT NULL,
                tag_id INTEGER NOT NULL,
                source TEXT NOT NULL DEFAULT 'config',
                confidence REAL NOT NULL DEFAULT 1.0,
                PRIMARY KEY(source_id, tag_id)
            );

            CREATE TABLE IF NOT EXISTS article_tags (
                article_id TEXT NOT NULL,
                tag_id INTEGER NOT NULL,
                matched_by TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                PRIMARY KEY(article_id, tag_id, matched_by)
            );

            CREATE TABLE IF NOT EXISTS article_scores (
                article_id TEXT PRIMARY KEY,
                market_relevance_score REAL NOT NULL DEFAULT 0,
                crypto_relevance_score REAL NOT NULL DEFAULT 0,
                tradfi_relevance_score REAL NOT NULL DEFAULT 0,
                macro_relevance_score REAL NOT NULL DEFAULT 0,
                urgency_score REAL NOT NULL DEFAULT 0,
                duplicate_penalty REAL NOT NULL DEFAULT 0,
                final_priority_score REAL NOT NULL DEFAULT 0,
                scoring_version TEXT NOT NULL,
                scoring_reasons_json TEXT NOT NULL DEFAULT '[]',
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS source_stats_daily (
                source_id TEXT NOT NULL,
                stat_date TEXT NOT NULL,
                articles_seen INTEGER NOT NULL DEFAULT 0,
                relevant_articles INTEGER NOT NULL DEFAULT 0,
                high_priority_articles INTEGER NOT NULL DEFAULT 0,
                duplicate_articles INTEGER NOT NULL DEFAULT 0,
                avg_priority_score REAL NOT NULL DEFAULT 0,
                usefulness_score REAL NOT NULL DEFAULT 0,
                PRIMARY KEY(source_id, stat_date)
            );

            CREATE INDEX IF NOT EXISTS idx_tags_namespace_value ON tags(namespace, value);
            CREATE INDEX IF NOT EXISTS idx_article_tags_article_id ON article_tags(article_id);
            CREATE INDEX IF NOT EXISTS idx_article_tags_tag_id ON article_tags(tag_id);
            CREATE INDEX IF NOT EXISTS idx_source_tags_source_id ON source_tags(source_id);
            CREATE INDEX IF NOT EXISTS idx_article_scores_priority ON article_scores(final_priority_score);
            CREATE INDEX IF NOT EXISTS idx_source_stats_daily_source_date ON source_stats_daily(source_id, stat_date);

            INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', '{SCHEMA_VERSION}');

            COMMIT;
            """
        )
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(url)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, ""))


def article_id_from(source_id: str, guid: str | None, canonical_url: str | None, title: str, published_at: str | None) -> str:
    guid_text = (guid or "").strip()
    if guid_text:
        return _stable_hash(f"{source_id}|guid|{guid_text}")
    url_text = _normalize_url(canonical_url or "")
    if url_text:
        return _stable_hash(f"url|{url_text}")
    return _stable_hash(f"{source_id}|{title}|{published_at or ''}")


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _clean_tag_part(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split()).replace(" ", "_")


def iter_source_config_tags(source: dict[str, Any]) -> list[tuple[str, str, float, str]]:
    tags: list[tuple[str, str, float, str]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(namespace: Any, value: Any, confidence: float = 1.0, matched_by: str = "config") -> None:
        namespace_text = _clean_tag_part(namespace)
        value_text = _clean_tag_part(value)
        if not namespace_text or not value_text:
            return
        key = (namespace_text, value_text, matched_by)
        if key in seen:
            return
        seen.add(key)
        tags.append((namespace_text, value_text, float(confidence), matched_by))

    for raw_tag in source.get("tags") or []:
        if not isinstance(raw_tag, str) or ":" not in raw_tag:
            continue
        namespace, value = raw_tag.split(":", 1)
        add(namespace, value, 1.0, "config")

    inherited_fields = {
        "source_group": source.get("source_group"),
        "source_type": source.get("source_type") or source.get("type"),
        "region": source.get("region"),
        "topic": source.get("topic"),
        "market_relevance": source.get("market_relevance"),
    }
    for namespace, value in inherited_fields.items():
        add(namespace, value, 1.0, "config_inherited")

    return tags


def _iter_source_config_tags(source: dict[str, Any]) -> list[tuple[str, str, float, str]]:
    return iter_source_config_tags(source)


def get_or_create_tag(conn: sqlite3.Connection, namespace: str, value: str) -> int:
    namespace_text = _clean_tag_part(namespace)
    value_text = _clean_tag_part(value)
    if not namespace_text or not value_text:
        raise ValueError("Tag namespace and value are required.")
    conn.execute(
        "INSERT OR IGNORE INTO tags(namespace, value) VALUES (?, ?)",
        (namespace_text, value_text),
    )
    row = conn.execute(
        "SELECT id FROM tags WHERE namespace = ? AND value = ?",
        (namespace_text, value_text),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Could not create tag {namespace_text}:{value_text}")
    return int(row["id"] if isinstance(row, sqlite3.Row) else row[0])


def apply_source_config_tags(conn: sqlite3.Connection, source: dict[str, Any]) -> None:
    source_id = str(source.get("id") or source.get("source_id") or "").strip()
    if not source_id:
        return
    for namespace, value, confidence, matched_by in iter_source_config_tags(source):
        tag_id = get_or_create_tag(conn, namespace, value)
        conn.execute(
            """
            INSERT INTO source_tags(source_id, tag_id, source, confidence)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(source_id, tag_id) DO UPDATE SET
                source=CASE WHEN source_tags.source = 'config' THEN source_tags.source ELSE excluded.source END,
                confidence=MAX(source_tags.confidence, excluded.confidence)
            """,
            (source_id, tag_id, matched_by or "config", float(confidence)),
        )


def apply_article_tags(conn: sqlite3.Connection, article_id: str, tags: list[tuple[Any, Any, Any, Any]]) -> None:
    article_id_text = str(article_id or "").strip()
    if not article_id_text:
        return
    for namespace, value, confidence, matched_by in tags:
        namespace_text = _clean_tag_part(namespace)
        value_text = _clean_tag_part(value)
        matched_by_text = _clean_tag_part(matched_by) or "rule"
        if not namespace_text or not value_text:
            continue
        tag_id = get_or_create_tag(conn, namespace_text, value_text)
        conn.execute(
            """
            INSERT INTO article_tags(article_id, tag_id, matched_by, confidence)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(article_id, tag_id, matched_by) DO UPDATE SET
                confidence=MAX(article_tags.confidence, excluded.confidence)
            """,
            (article_id_text, tag_id, matched_by_text, float(confidence or 0.0)),
        )


def upsert_article_score(conn: sqlite3.Connection, article_id: str, score_payload: dict[str, Any]) -> None:
    article_id_text = str(article_id or "").strip()
    if not article_id_text:
        return
    reasons = score_payload.get("scoring_reasons_json", score_payload.get("reasons", []))
    if not isinstance(reasons, str):
        reasons = _json_text(reasons if isinstance(reasons, list) else [])
    conn.execute(
        """
        INSERT INTO article_scores(
            article_id, market_relevance_score, crypto_relevance_score, tradfi_relevance_score,
            macro_relevance_score, urgency_score, duplicate_penalty, final_priority_score,
            scoring_version, scoring_reasons_json, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(article_id) DO UPDATE SET
            market_relevance_score=excluded.market_relevance_score,
            crypto_relevance_score=excluded.crypto_relevance_score,
            tradfi_relevance_score=excluded.tradfi_relevance_score,
            macro_relevance_score=excluded.macro_relevance_score,
            urgency_score=excluded.urgency_score,
            duplicate_penalty=excluded.duplicate_penalty,
            final_priority_score=excluded.final_priority_score,
            scoring_version=excluded.scoring_version,
            scoring_reasons_json=excluded.scoring_reasons_json,
            updated_at=excluded.updated_at
        """,
        (
            article_id_text,
            float(score_payload.get("market_relevance_score") or 0),
            float(score_payload.get("crypto_relevance_score") or 0),
            float(score_payload.get("tradfi_relevance_score") or 0),
            float(score_payload.get("macro_relevance_score") or 0),
            float(score_payload.get("urgency_score") or 0),
            float(score_payload.get("duplicate_penalty") or 0),
            float(score_payload.get("final_priority_score") or 0),
            str(score_payload.get("scoring_version") or "unknown"),
            str(reasons),
            str(score_payload.get("updated_at") or _utc_now()),
        ),
    )


def update_source_daily_stats(conn: sqlite3.Connection, source_id: str, stat_date: str) -> None:
    source_id_text = str(source_id or "").strip()
    stat_date_text = str(stat_date or _utc_now()[:10])[:10]
    if not source_id_text:
        return

    article_row = conn.execute(
        """
        SELECT
            COUNT(a.id) AS article_count,
            COALESCE(SUM(CASE WHEN COALESCE(s.final_priority_score, 0) >= 35 THEN 1 ELSE 0 END), 0) AS relevant_count,
            COALESCE(SUM(CASE WHEN COALESCE(s.final_priority_score, 0) >= 70 THEN 1 ELSE 0 END), 0) AS high_count,
            COALESCE(AVG(s.final_priority_score), 0) AS avg_score
        FROM articles a
        LEFT JOIN article_scores s ON s.article_id = a.id
        WHERE a.source_id = ? AND substr(COALESCE(a.collected_at, a.updated_at, ''), 1, 10) = ?
        """,
        (source_id_text, stat_date_text),
    ).fetchone()
    duplicate_row = conn.execute(
        """
        SELECT COALESCE(SUM(duplicate_count), 0)
        FROM source_fetch_log
        WHERE source_id = ? AND substr(COALESCE(finished_at, started_at, ''), 1, 10) = ?
        """,
        (source_id_text, stat_date_text),
    ).fetchone()

    article_count = int(article_row["article_count"] if isinstance(article_row, sqlite3.Row) else article_row[0])
    relevant_count = int(article_row["relevant_count"] if isinstance(article_row, sqlite3.Row) else article_row[1])
    high_count = int(article_row["high_count"] if isinstance(article_row, sqlite3.Row) else article_row[2])
    avg_score = float(article_row["avg_score"] if isinstance(article_row, sqlite3.Row) else article_row[3])
    duplicate_count = int(duplicate_row[0] if duplicate_row is not None else 0)
    articles_seen = article_count + duplicate_count

    if articles_seen:
        relevant_rate = relevant_count / articles_seen
        high_rate = high_count / articles_seen
        duplicate_rate = duplicate_count / articles_seen
        usefulness_score = max(0.0, min(100.0, (avg_score * 0.6) + (relevant_rate * 25.0) + (high_rate * 25.0) - (duplicate_rate * 20.0)))
    else:
        usefulness_score = 0.0

    conn.execute(
        """
        INSERT INTO source_stats_daily(
            source_id, stat_date, articles_seen, relevant_articles, high_priority_articles,
            duplicate_articles, avg_priority_score, usefulness_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_id, stat_date) DO UPDATE SET
            articles_seen=excluded.articles_seen,
            relevant_articles=excluded.relevant_articles,
            high_priority_articles=excluded.high_priority_articles,
            duplicate_articles=excluded.duplicate_articles,
            avg_priority_score=excluded.avg_priority_score,
            usefulness_score=excluded.usefulness_score
        """,
        (
            source_id_text,
            stat_date_text,
            articles_seen,
            relevant_count,
            high_count,
            duplicate_count,
            avg_score,
            usefulness_score,
        ),
    )


def upsert_article(conn: sqlite3.Connection, article: dict[str, Any]) -> bool:
    columns = [
        "id",
        "source_id",
        "source_group",
        "source_type",
        "region",
        "topic",
        "market_relevance",
        "source_url",
        "canonical_url",
        "title",
        "summary",
        "published_at",
        "collected_at",
        "updated_at",
        "language",
        "detected_assets_json",
        "detected_entities_json",
        "event_type",
        "impact_direction",
        "impact_scope",
        "severity",
        "confidence",
        "raw_json",
        "raw_file_path",
    ]
    values = {key: article.get(key) for key in columns}
    values["updated_at"] = values.get("updated_at") or values.get("collected_at") or _utc_now()
    values["raw_json"] = values.get("raw_json") or _json_text({})
    values["title"] = values.get("title") or ""
    values["source_id"] = values.get("source_id") or ""
    values["source_type"] = values.get("source_type") or ""
    values["collected_at"] = values.get("collected_at") or _utc_now()
    values["id"] = values.get("id") or article_id_from(
        str(values["source_id"]),
        article.get("guid"),
        values.get("canonical_url"),
        str(values["title"]),
        values.get("published_at"),
    )

    insert_sql = f"INSERT OR IGNORE INTO articles ({', '.join(columns)}) VALUES ({', '.join(f':{c}' for c in columns)})"
    update_sql = """
        UPDATE articles SET
            source_group = COALESCE(source_group, :source_group),
            source_type = COALESCE(NULLIF(source_type, ''), :source_type),
            region = COALESCE(region, :region),
            topic = COALESCE(topic, :topic),
            market_relevance = COALESCE(market_relevance, :market_relevance),
            source_url = COALESCE(source_url, :source_url),
            canonical_url = COALESCE(canonical_url, :canonical_url),
            title = CASE WHEN COALESCE(title, '') = '' THEN :title ELSE title END,
            summary = CASE WHEN COALESCE(summary, '') = '' THEN :summary ELSE summary END,
            published_at = COALESCE(published_at, :published_at),
            updated_at = :updated_at,
            language = COALESCE(language, :language),
            detected_assets_json = COALESCE(NULLIF(detected_assets_json, ''), :detected_assets_json),
            detected_entities_json = COALESCE(NULLIF(detected_entities_json, ''), :detected_entities_json),
            event_type = COALESCE(event_type, :event_type),
            impact_direction = COALESCE(impact_direction, :impact_direction),
            impact_scope = COALESCE(impact_scope, :impact_scope),
            severity = COALESCE(severity, :severity),
            confidence = COALESCE(confidence, :confidence),
            raw_json = CASE WHEN LENGTH(COALESCE(raw_json, '')) < LENGTH(:raw_json) THEN :raw_json ELSE raw_json END,
            raw_file_path = COALESCE(raw_file_path, :raw_file_path)
        WHERE id = :id
    """
    cur = conn.execute(insert_sql, values)
    inserted = cur.rowcount == 1
    if not inserted:
        conn.execute(update_sql, values)

    assets = article.get("detected_assets")
    if isinstance(assets, list):
        for asset in assets:
            conn.execute(
                "INSERT OR IGNORE INTO article_assets(article_id, asset, confidence) VALUES (?, ?, ?)",
                (values["id"], str(asset), 1.0),
            )
    categories = article.get("detected_categories")
    if isinstance(categories, list):
        for category in categories:
            conn.execute(
                "INSERT OR IGNORE INTO article_categories(article_id, category, confidence) VALUES (?, ?, ?)",
                (values["id"], str(category), 1.0),
            )

    conn.commit()
    return inserted


def record_fetch(conn: sqlite3.Connection, fetch_record: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO source_fetch_log(
            source_id, source_type, started_at, finished_at, status,
            fetched_count, inserted_count, duplicate_count, error_message
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(fetch_record.get("source_id") or ""),
            str(fetch_record.get("source_type") or ""),
            str(fetch_record.get("started_at") or _utc_now()),
            fetch_record.get("finished_at"),
            str(fetch_record.get("status") or "unknown"),
            int(fetch_record.get("fetched_count") or 0),
            int(fetch_record.get("inserted_count") or 0),
            int(fetch_record.get("duplicate_count") or 0),
            fetch_record.get("error_message"),
        ),
    )
    conn.commit()


def upsert_source_health(conn: sqlite3.Connection, source: dict[str, Any], health: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO sources(
            source_id, source_group, source_type, enabled, url_or_query, region, topic,
            market_relevance, last_success_at, last_failure_at, last_error, last_http_status,
            items_last_fetch, inserted_last_fetch, duplicates_last_fetch, avg_items_per_day
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_id) DO UPDATE SET
            source_group=excluded.source_group,
            source_type=excluded.source_type,
            enabled=excluded.enabled,
            url_or_query=excluded.url_or_query,
            region=excluded.region,
            topic=excluded.topic,
            market_relevance=excluded.market_relevance,
            last_success_at=excluded.last_success_at,
            last_failure_at=excluded.last_failure_at,
            last_error=excluded.last_error,
            last_http_status=excluded.last_http_status,
            items_last_fetch=excluded.items_last_fetch,
            inserted_last_fetch=excluded.inserted_last_fetch,
            duplicates_last_fetch=excluded.duplicates_last_fetch,
            avg_items_per_day=excluded.avg_items_per_day
        """,
        (
            str(source.get("id") or ""),
            source.get("source_group"),
            str(source.get("type") or ""),
            1 if source.get("enabled") else 0,
            source.get("url") or source.get("query") or source.get("url_or_query"),
            source.get("region"),
            source.get("topic"),
            source.get("market_relevance"),
            health.get("last_success_at"),
            health.get("last_failure_at"),
            health.get("last_error"),
            health.get("last_http_status"),
            int(health.get("items_last_fetch") or 0),
            int(health.get("inserted_last_fetch") or 0),
            int(health.get("duplicates_last_fetch") or 0),
            health.get("avg_items_per_day"),
        ),
    )
    conn.commit()


def update_status(status_path: Path, status: dict[str, Any]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_status(status_path: Path) -> dict[str, Any]:
    if not status_path.exists():
        return {}
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _csv_export(db_path: Path, csv_path: Path, query: str) -> int:
    if not db_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("", encoding="utf-8")
        return 0
    conn = connect_db(db_path)
    try:
        rows = conn.execute(query).fetchall()
    finally:
        conn.close()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer: csv.DictWriter[str] | None = None
        for row in rows:
            data = dict(row)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(data.keys()))
                writer.writeheader()
            writer.writerow(data)
            count += 1
    return count


def export_articles_csv(db_path: Path, csv_path: Path) -> int:
    return _csv_export(db_path, csv_path, "SELECT * FROM articles ORDER BY collected_at DESC, title ASC")


def export_source_health_csv(db_path: Path, csv_path: Path) -> int:
    return _csv_export(db_path, csv_path, "SELECT * FROM sources ORDER BY source_group, source_id")

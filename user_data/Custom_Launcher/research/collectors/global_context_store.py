from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1"


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db(db_path: Path) -> None:
    conn = connect_db(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
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

            CREATE TABLE IF NOT EXISTS context_sources (
                source_id TEXT PRIMARY KEY,
                source_group TEXT NOT NULL,
                source_type TEXT NOT NULL,
                enabled INTEGER NOT NULL,
                market_relevance TEXT,
                url TEXT,
                last_success_at TEXT,
                last_failure_at TEXT,
                last_error TEXT,
                last_score REAL,
                last_signal TEXT,
                last_value REAL,
                last_unit TEXT,
                last_notes TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS global_context_ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                source_ts TEXT,
                source_id TEXT NOT NULL,
                source_group TEXT NOT NULL,
                source_type TEXT NOT NULL,
                metric_key TEXT NOT NULL,
                score REAL,
                signal TEXT,
                value REAL,
                unit TEXT,
                notes TEXT,
                raw_json TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_global_context_source_ts ON global_context_ticks(source_id, ts);
            CREATE INDEX IF NOT EXISTS idx_global_context_group_ts ON global_context_ticks(source_group, ts);
            """
        )
        conn.execute("INSERT OR REPLACE INTO schema_meta(key, value) VALUES(?, ?)", ("version", SCHEMA_VERSION))
        conn.commit()
    finally:
        conn.close()


def upsert_collector_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    started_at: str,
    status: str,
    pid: int,
    config_path: str,
    db_path: str,
    notes: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO collector_runs(run_id, started_at, stopped_at, status, pid, config_path, db_path, notes)
        VALUES (?, ?, NULL, ?, ?, ?, ?, ?)
        """,
        (run_id, started_at, status, pid, config_path, db_path, notes),
    )


def update_collector_run(conn: sqlite3.Connection, run_id: str, status: str, notes: str | None = None) -> None:
    conn.execute(
        "UPDATE collector_runs SET status = ?, stopped_at = ?, notes = ? WHERE run_id = ?",
        (status, utc_now(), notes, run_id),
    )


def upsert_source_status(conn: sqlite3.Connection, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload.setdefault("updated_at", utc_now())
    columns = [
        "source_id",
        "source_group",
        "source_type",
        "enabled",
        "market_relevance",
        "url",
        "last_success_at",
        "last_failure_at",
        "last_error",
        "last_score",
        "last_signal",
        "last_value",
        "last_unit",
        "last_notes",
        "updated_at",
    ]
    values = [payload.get(column) for column in columns]
    placeholders = ", ".join("?" for _ in columns)
    updates = """
        source_group=excluded.source_group,
        source_type=excluded.source_type,
        enabled=excluded.enabled,
        market_relevance=excluded.market_relevance,
        url=excluded.url,
        last_success_at=COALESCE(excluded.last_success_at, context_sources.last_success_at),
        last_failure_at=COALESCE(excluded.last_failure_at, context_sources.last_failure_at),
        last_error=excluded.last_error,
        last_score=COALESCE(excluded.last_score, context_sources.last_score),
        last_signal=COALESCE(excluded.last_signal, context_sources.last_signal),
        last_value=COALESCE(excluded.last_value, context_sources.last_value),
        last_unit=COALESCE(excluded.last_unit, context_sources.last_unit),
        last_notes=COALESCE(excluded.last_notes, context_sources.last_notes),
        updated_at=excluded.updated_at
    """
    conn.execute(
        f"INSERT INTO context_sources({', '.join(columns)}) VALUES({placeholders}) "
        f"ON CONFLICT(source_id) DO UPDATE SET {updates}",
        values,
    )


def insert_context_tick(conn: sqlite3.Connection, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload.setdefault("ts", utc_now())
    payload.setdefault("created_at", utc_now())
    if payload.get("raw_json") is not None and not isinstance(payload["raw_json"], str):
        payload["raw_json"] = json.dumps(payload["raw_json"], separators=(",", ":"), sort_keys=True)
    columns = list(payload.keys())
    conn.execute(
        f"INSERT INTO global_context_ticks({', '.join(columns)}) VALUES({', '.join('?' for _ in columns)})",
        [payload[column] for column in columns],
    )


def fetch_latest_context_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT t.*
        FROM global_context_ticks t
        JOIN (
            SELECT source_id, MAX(id) AS latest_id
            FROM global_context_ticks
            GROUP BY source_id
        ) latest ON latest.latest_id = t.id
        ORDER BY t.source_group, t.source_id
        """
    ).fetchall()
    return [dict(row) for row in rows]


def fetch_source_health(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT source_id, source_group, source_type, enabled, market_relevance, last_success_at,
               last_failure_at, last_error, last_score, last_signal, last_value, last_unit,
               last_notes, updated_at
        FROM context_sources
        ORDER BY source_group, source_id
        """
    ).fetchall()
    return [dict(row) for row in rows]


def fetch_storage_summary(conn: sqlite3.Connection) -> dict[str, int]:
    source_rows = int(conn.execute("SELECT COUNT(*) FROM context_sources").fetchone()[0])
    tick_rows = int(conn.execute("SELECT COUNT(*) FROM global_context_ticks").fetchone()[0])
    return {"source_rows": source_rows, "tick_rows": tick_rows}


def export_latest_csv(db_path: Path, csv_path: Path) -> int:
    conn = connect_db(db_path)
    try:
        rows = fetch_latest_context_rows(conn)
    finally:
        conn.close()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ts",
        "source_ts",
        "source_id",
        "source_group",
        "source_type",
        "metric_key",
        "score",
        "signal",
        "value",
        "unit",
        "notes",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return len(rows)

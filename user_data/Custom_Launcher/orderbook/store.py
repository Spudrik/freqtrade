from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
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

            CREATE TABLE IF NOT EXISTS stream_status (
                stream_id TEXT PRIMARY KEY,
                exchange TEXT NOT NULL,
                market_type TEXT NOT NULL,
                pair TEXT NOT NULL,
                symbol TEXT NOT NULL,
                depth_levels INTEGER NOT NULL,
                stream_mode TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT,
                last_message_at TEXT,
                last_metric_at TEXT,
                message_count INTEGER DEFAULT 0,
                metric_count INTEGER DEFAULT 0,
                snapshot_count INTEGER DEFAULT 0,
                reconnect_count INTEGER DEFAULT 0,
                error_count INTEGER DEFAULT 0,
                last_error TEXT,
                best_bid REAL,
                best_ask REAL,
                mid_price REAL,
                spread_bps REAL,
                imbalance_top20 REAL,
                bid_pressure_ratio_60s REAL,
                ask_pressure_ratio_60s REAL,
                nearest_bid_wall_distance_bps REAL,
                nearest_ask_wall_distance_bps REAL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS orderbook_metric_ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                exchange TEXT NOT NULL,
                market_type TEXT NOT NULL,
                pair TEXT NOT NULL,
                symbol TEXT NOT NULL,
                book_valid INTEGER NOT NULL,
                depth_levels INTEGER NOT NULL,
                message_count_interval INTEGER DEFAULT 0,
                best_bid REAL,
                best_ask REAL,
                mid_price REAL,
                spread_bps REAL,
                microprice REAL,
                microprice_offset_bps REAL,
                bid_notional_top1 REAL,
                ask_notional_top1 REAL,
                imbalance_top1 REAL,
                bid_notional_top5 REAL,
                ask_notional_top5 REAL,
                imbalance_top5 REAL,
                bid_notional_top10 REAL,
                ask_notional_top10 REAL,
                imbalance_top10 REAL,
                bid_notional_top20 REAL,
                ask_notional_top20 REAL,
                imbalance_top20 REAL,
                bid_liquidity_5bps REAL,
                ask_liquidity_5bps REAL,
                imbalance_5bps REAL,
                bid_liquidity_10bps REAL,
                ask_liquidity_10bps REAL,
                imbalance_10bps REAL,
                bid_liquidity_25bps REAL,
                ask_liquidity_25bps REAL,
                imbalance_25bps REAL,
                bid_liquidity_50bps REAL,
                ask_liquidity_50bps REAL,
                imbalance_50bps REAL,
                nearest_bid_wall_price REAL,
                nearest_bid_wall_distance_bps REAL,
                nearest_bid_wall_notional REAL,
                nearest_bid_wall_score REAL,
                nearest_ask_wall_price REAL,
                nearest_ask_wall_distance_bps REAL,
                nearest_ask_wall_notional REAL,
                nearest_ask_wall_score REAL,
                strongest_bid_wall_price_50bps REAL,
                strongest_bid_wall_score_50bps REAL,
                strongest_ask_wall_price_50bps REAL,
                strongest_ask_wall_score_50bps REAL,
                strong_bid_pressure INTEGER,
                strong_ask_pressure INTEGER,
                extreme_bid_pressure INTEGER,
                extreme_ask_pressure INTEGER,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_orderbook_metric_ticks_pair_ts ON orderbook_metric_ticks(pair, ts);
            CREATE INDEX IF NOT EXISTS idx_orderbook_metric_ticks_symbol_ts ON orderbook_metric_ticks(symbol, ts);
            CREATE INDEX IF NOT EXISTS idx_orderbook_metric_ticks_exchange_ts ON orderbook_metric_ticks(exchange, ts);

            CREATE TABLE IF NOT EXISTS orderbook_metric_bars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_start TEXT NOT NULL,
                ts_end TEXT NOT NULL,
                timeframe_seconds INTEGER NOT NULL,
                exchange TEXT NOT NULL,
                market_type TEXT NOT NULL,
                pair TEXT NOT NULL,
                symbol TEXT NOT NULL,
                valid_samples INTEGER NOT NULL,
                expected_samples INTEGER NOT NULL,
                spread_bps_mean REAL,
                spread_bps_max REAL,
                microprice_offset_bps_mean REAL,
                imbalance_top20_mean REAL,
                imbalance_top20_min REAL,
                imbalance_top20_max REAL,
                imbalance_10bps_mean REAL,
                imbalance_25bps_mean REAL,
                bid_pressure_seconds REAL,
                ask_pressure_seconds REAL,
                bid_pressure_ratio REAL,
                ask_pressure_ratio REAL,
                max_bid_pressure_streak_seconds REAL,
                max_ask_pressure_streak_seconds REAL,
                nearest_bid_wall_min_distance_bps REAL,
                nearest_ask_wall_min_distance_bps REAL,
                strongest_bid_wall_score REAL,
                strongest_ask_wall_score REAL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_orderbook_metric_bars_pair_tf_ts ON orderbook_metric_bars(pair, timeframe_seconds, ts_start);

            CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                exchange TEXT NOT NULL,
                market_type TEXT NOT NULL,
                pair TEXT NOT NULL,
                symbol TEXT NOT NULL,
                depth_levels INTEGER NOT NULL,
                best_bid REAL,
                best_ask REAL,
                mid_price REAL,
                bids_json TEXT NOT NULL,
                asks_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_orderbook_snapshots_pair_ts ON orderbook_snapshots(pair, ts);

            CREATE TABLE IF NOT EXISTS capacity_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                level TEXT NOT NULL,
                data_dir TEXT NOT NULL,
                db_path TEXT NOT NULL,
                data_dir_mb REAL NOT NULL,
                db_mb REAL NOT NULL,
                warning_threshold_mb REAL NOT NULL,
                critical_threshold_mb REAL NOT NULL,
                message TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS storage_stats (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.execute("INSERT OR REPLACE INTO schema_meta(key, value) VALUES(?, ?)", ("version", "1"))
        conn.commit()
    finally:
        conn.close()


def upsert_collector_run(conn: sqlite3.Connection, run_id: str, started_at: str, status: str, pid: int, config_path: str, db_path: str, notes: str | None = None) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO collector_runs(run_id, started_at, stopped_at, status, pid, config_path, db_path, notes)
        VALUES (?, ?, NULL, ?, ?, ?, ?, ?)
        """,
        (run_id, started_at, status, pid, config_path, db_path, notes),
    )


def update_stream_status(conn: sqlite3.Connection, payload: dict[str, Any]) -> None:
    columns = [
        "stream_id",
        "exchange",
        "market_type",
        "pair",
        "symbol",
        "depth_levels",
        "stream_mode",
        "status",
        "started_at",
        "last_message_at",
        "last_metric_at",
        "message_count",
        "metric_count",
        "snapshot_count",
        "reconnect_count",
        "error_count",
        "last_error",
        "best_bid",
        "best_ask",
        "mid_price",
        "spread_bps",
        "imbalance_top20",
        "bid_pressure_ratio_60s",
        "ask_pressure_ratio_60s",
        "nearest_bid_wall_distance_bps",
        "nearest_ask_wall_distance_bps",
        "updated_at",
    ]
    now = utc_now()
    values = [payload.get(column) for column in columns]
    if values[-1] is None:
        values[-1] = now
    placeholders = ", ".join("?" for _ in columns)
    updates = ", ".join(f"{col}=excluded.{col}" for col in columns[1:])
    conn.execute(
        f"INSERT INTO stream_status({', '.join(columns)}) VALUES({placeholders}) "
        f"ON CONFLICT(stream_id) DO UPDATE SET {updates}",
        values,
    )


def insert_metric_tick(conn: sqlite3.Connection, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload.setdefault("created_at", utc_now())
    columns = list(payload.keys())
    values = [payload[key] for key in columns]
    conn.execute(
        f"INSERT INTO orderbook_metric_ticks({', '.join(columns)}) VALUES({', '.join('?' for _ in columns)})",
        values,
    )


def insert_metric_bar(conn: sqlite3.Connection, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload.setdefault("created_at", utc_now())
    columns = list(payload.keys())
    conn.execute(
        f"INSERT INTO orderbook_metric_bars({', '.join(columns)}) VALUES({', '.join('?' for _ in columns)})",
        [payload[key] for key in columns],
    )


def insert_snapshot(conn: sqlite3.Connection, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload.setdefault("created_at", utc_now())
    payload["bids_json"] = json.dumps(payload.get("bids_json") if isinstance(payload.get("bids_json"), list) else payload.get("bids") or [], separators=(",", ":"))
    payload["asks_json"] = json.dumps(payload.get("asks_json") if isinstance(payload.get("asks_json"), list) else payload.get("asks") or [], separators=(",", ":"))
    payload.pop("bids", None)
    payload.pop("asks", None)
    columns = list(payload.keys())
    conn.execute(
        f"INSERT INTO orderbook_snapshots({', '.join(columns)}) VALUES({', '.join('?' for _ in columns)})",
        [payload[key] for key in columns],
    )


def insert_capacity_alert(
    conn: sqlite3.Connection,
    *,
    level: str,
    data_dir: str,
    db_path: str,
    data_dir_mb: float,
    db_mb: float,
    warning_threshold_mb: float,
    critical_threshold_mb: float,
    message: str,
) -> None:
    conn.execute(
        """
        INSERT INTO capacity_alerts(ts, level, data_dir, db_path, data_dir_mb, db_mb, warning_threshold_mb, critical_threshold_mb, message)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (utc_now(), level, data_dir, db_path, data_dir_mb, db_mb, warning_threshold_mb, critical_threshold_mb, message),
    )


def fetch_latest_metrics(conn: sqlite3.Connection, limit: int = 200) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT *
        FROM orderbook_metric_ticks
        ORDER BY ts DESC, id DESC
        LIMIT ?
        """,
        (max(1, int(limit)),),
    ).fetchall()
    return [dict(row) for row in rows]


def fetch_stream_status(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM stream_status ORDER BY pair").fetchall()
    return [dict(row) for row in rows]


def fetch_storage_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    metric_rows = int(conn.execute("SELECT COUNT(*) FROM orderbook_metric_ticks").fetchone()[0])
    snapshot_rows = int(conn.execute("SELECT COUNT(*) FROM orderbook_snapshots").fetchone()[0])
    bar_rows = int(conn.execute("SELECT COUNT(*) FROM orderbook_metric_bars").fetchone()[0])
    stream_rows = int(conn.execute("SELECT COUNT(*) FROM stream_status").fetchone()[0])
    return {
        "metric_rows": metric_rows,
        "snapshot_rows": snapshot_rows,
        "bar_rows": bar_rows,
        "stream_rows": stream_rows,
    }

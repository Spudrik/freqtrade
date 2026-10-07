from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import argparse
import hashlib
import json
import math
import sqlite3
import sys

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


SCHEMA_VERSION = "2"
DEFAULT_USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = DEFAULT_USER_DATA_DIR / "collector_data" / "orderbook" / "orderbook_events.sqlite"
DEFAULT_EXPORT_DIR = DEFAULT_USER_DATA_DIR / "orderbook_data" / "live" / "exports"
DEFAULT_OHLCV_PATH = DEFAULT_USER_DATA_DIR / "data" / "binance" / "BTC_USDT-1h.feather"
DEFAULT_PAIRS = ("BTC/USDT",)
DEFAULT_MARKET_KEYS = ("binance_spot", "binance_usdm_futures", "bybit_spot", "bybit_linear")
DEFAULT_EXPECTED_SAMPLES_PER_HOUR = 3600
DEFAULT_MIN_COVERAGE_RATIO = 0.50

BASE_COLUMNS = {
    "date",
    "canonical_pair",
    "generated_at",
    "source_min_ts",
    "source_max_ts",
    "raw_tick_rows",
    "market_context_rows",
    "feature_schema_version",
    "feature_config_hash",
}

RAW_NUMERIC_COLUMNS = (
    "spread_bps",
    "microprice_offset_bps",
    "bid_notional_top1",
    "ask_notional_top1",
    "imbalance_top1",
    "bid_notional_top5",
    "ask_notional_top5",
    "imbalance_top5",
    "bid_notional_top10",
    "ask_notional_top10",
    "imbalance_top10",
    "bid_notional_top20",
    "ask_notional_top20",
    "imbalance_top20",
    "bid_liquidity_5bps",
    "ask_liquidity_5bps",
    "imbalance_5bps",
    "bid_liquidity_10bps",
    "ask_liquidity_10bps",
    "imbalance_10bps",
    "bid_liquidity_25bps",
    "ask_liquidity_25bps",
    "imbalance_25bps",
    "bid_liquidity_50bps",
    "ask_liquidity_50bps",
    "imbalance_50bps",
    "nearest_bid_wall_distance_bps",
    "nearest_bid_wall_price",
    "nearest_bid_wall_notional",
    "nearest_bid_wall_score",
    "nearest_ask_wall_distance_bps",
    "nearest_ask_wall_price",
    "nearest_ask_wall_notional",
    "nearest_ask_wall_score",
    "strongest_bid_wall_price_50bps",
    "strongest_bid_wall_score_50bps",
    "strongest_ask_wall_price_50bps",
    "strongest_ask_wall_score_50bps",
)

MARKET_CONTEXT_COLUMNS = (
    "funding_rate",
    "open_interest",
    "long_ratio",
    "short_ratio",
    "long_short_ratio",
    "taker_buy_volume",
    "taker_sell_volume",
    "taker_buy_sell_ratio",
)

ROLLING_SOURCE_SUFFIXES = (
    "pressure_delta",
    "imbalance_top20_mean",
    "imbalance_top20_std",
    "spread_bps_mean",
    "spread_bps_last",
    "spread_bps_p95",
    "microprice_offset_bps_mean",
    "total_depth_top20_mean",
    "nearest_ask_wall_distance_bps_min",
    "nearest_bid_wall_distance_bps_min",
    "nearest_ask_wall_notional_p95",
    "nearest_bid_wall_notional_p95",
    "nearest_ask_wall_score_p95",
    "nearest_bid_wall_score_p95",
    "bid_liquidity_5bps_min",
    "ask_liquidity_5bps_min",
    "depth_thinness_score",
    "pressure_flip_count",
    "wall_support_resistance_delta",
)

CROSS_ROLLING_COLUMNS = (
    "ob1h_x_pressure_delta_mean",
    "ob1h_x_imbalance_top20_mean",
    "ob1h_x_spread_bps_mean",
    "ob1h_x_total_depth_top20_mean",
    "ob1h_x_confluence_long_score",
    "ob1h_x_confluence_short_score",
)


@dataclass
class CompactionSummary:
    mode: str
    dry_run: bool
    db_path: str
    pairs: list[str]
    build_start: str | None = None
    build_end: str | None = None
    source_rows_found: dict[str, int] = field(default_factory=dict)
    source_rows_loaded: dict[str, int] = field(default_factory=dict)
    feature_rows_written: int = 0
    feature_column_count: int = 0
    export_path: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "dry_run": self.dry_run,
            "db_path": self.db_path,
            "pairs": self.pairs,
            "build_start": self.build_start,
            "build_end": self.build_end,
            "source_rows_found": self.source_rows_found,
            "source_rows_loaded": self.source_rows_loaded,
            "feature_rows_written": self.feature_rows_written,
            "feature_column_count": self.feature_column_count,
            "export_path": self.export_path,
            "warnings": self.warnings,
        }


@dataclass
class ValidationSummary:
    db_path: str
    pairs: list[str]
    rows_checked: int = 0
    violations: dict[str, int] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "db_path": self.db_path,
            "pairs": self.pairs,
            "rows_checked": self.rows_checked,
            "violations": self.violations,
            "coverage": self.coverage,
            "warnings": self.warnings,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def feature_config_hash() -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "raw_numeric_columns": RAW_NUMERIC_COLUMNS,
        "market_context_columns": MARKET_CONTEXT_COLUMNS,
        "default_market_keys": DEFAULT_MARKET_KEYS,
        "expected_samples_per_hour": DEFAULT_EXPECTED_SAMPLES_PER_HOUR,
        "objective_alpha": "zone_state_price_interaction_v1",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def compact_orderbook_features(
    db_path: Path = DEFAULT_DB_PATH,
    *,
    pairs: tuple[str, ...] = DEFAULT_PAIRS,
    mode: str = "update",
    overlap_days: int = 3,
    export_parquet: bool = False,
    export_csv: bool = False,
    export_dir: Path = DEFAULT_EXPORT_DIR,
    min_coverage_ratio: float = DEFAULT_MIN_COVERAGE_RATIO,
    ohlcv_path: Path | None = DEFAULT_OHLCV_PATH,
) -> CompactionSummary:
    if mode not in {"dry-run", "update", "full-rebuild"}:
        raise ValueError("mode must be one of: dry-run, update, full-rebuild")
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    dry_run = mode == "dry-run"
    selected_pairs = tuple(_normalise_pair(pair) for pair in pairs)
    summary = CompactionSummary(mode=mode, dry_run=dry_run, db_path=str(db_path), pairs=list(selected_pairs))

    with _connect(db_path) as conn:
        if not dry_run:
            _init_feature_tables(conn)
        if selected_pairs == ("all",):
            selected_pairs = tuple(_available_pairs(conn))
            summary.pairs = list(selected_pairs)
        if not selected_pairs:
            summary.warnings.append("No orderbook pairs were selected.")
            return summary

        all_features: list[DataFrame] = []
        source_rows_found: dict[str, int] = {}
        source_rows_loaded: dict[str, int] = {}
        build_starts: list[pd.Timestamp] = []
        build_ends: list[pd.Timestamp] = []
        for pair in selected_pairs:
            pair_values = _pair_values(conn, pair)
            if not pair_values:
                summary.warnings.append(f"No stored orderbook pair values found for {pair}.")
                continue
            stats = _raw_tick_stats(conn, pair_values)
            source_rows_found[pair] = int(stats.get("rows") or 0)
            if not stats.get("max_ts") or not stats.get("min_ts"):
                summary.warnings.append(f"No raw orderbook ticks found for {pair}.")
                continue
            build_start, build_end = _build_window(conn, pair, stats, mode=mode, overlap_days=overlap_days)
            if build_end < build_start:
                summary.warnings.append(f"No closed hourly windows to compact for {pair}.")
                continue
            build_starts.append(build_start)
            build_ends.append(build_end)
            if dry_run:
                source_rows_loaded[pair] = _planned_tick_count(
                    conn,
                    pair_values,
                    build_start - pd.Timedelta(hours=1),
                    build_end,
                )
                continue

            raw_start = build_start - pd.Timedelta(hours=1)
            raw_end = build_end
            ticks = _load_ticks(conn, pair_values, raw_start, raw_end)
            context = _load_market_context(conn, pair, raw_start, raw_end)
            source_rows_loaded[pair] = int(len(ticks))
            if ticks.empty:
                summary.warnings.append(f"No raw tick rows loaded for {pair} in selected compaction window.")
                continue
            features = _build_pair_features(
                ticks,
                context,
                pair,
                build_start,
                build_end,
                min_coverage_ratio=min_coverage_ratio,
                ohlcv_path=ohlcv_path,
            )
            all_features.append(features)

        summary.source_rows_found = source_rows_found
        summary.source_rows_loaded = source_rows_loaded
        if build_starts:
            summary.build_start = _iso(min(build_starts))
        if build_ends:
            summary.build_end = _iso(max(build_ends))
        if dry_run:
            return summary
        if not all_features:
            return summary

        output = pd.concat(all_features, ignore_index=True)
        output = output.replace([np.inf, -np.inf], np.nan)
        feature_columns = [column for column in output.columns if column not in BASE_COLUMNS]
        _ensure_feature_columns(conn, feature_columns)
        _write_feature_rows(conn, output)
        _write_states(conn, output)
        conn.commit()
        summary.feature_rows_written = int(len(output))
        summary.feature_column_count = int(len(feature_columns))

    if export_parquet or export_csv:
        path, _ = export_orderbook_features(db_path, export_dir, parquet=export_parquet, csv=export_csv)
        summary.export_path = str(path)
    return summary


def export_orderbook_features(
    db_path: Path = DEFAULT_DB_PATH,
    export_dir: Path = DEFAULT_EXPORT_DIR,
    *,
    parquet: bool = True,
    csv: bool = False,
) -> tuple[Path, int]:
    db_path = Path(db_path)
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        frame = pd.read_sql_query("SELECT * FROM orderbook_features_1h ORDER BY canonical_pair, date", conn)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = export_dir / f"orderbook_features_1h_{stamp}.parquet"
    if parquet:
        frame.to_parquet(output, index=False)
    if csv:
        csv_path = export_dir / f"orderbook_features_1h_{stamp}.csv"
        frame.to_csv(csv_path, index=False)
        if not parquet:
            output = csv_path
    return output, int(len(frame))


def validate_orderbook_features(
    db_path: Path = DEFAULT_DB_PATH,
    *,
    pairs: tuple[str, ...] = DEFAULT_PAIRS,
) -> ValidationSummary:
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    selected_pairs = tuple(_normalise_pair(pair) for pair in pairs)
    summary = ValidationSummary(db_path=str(db_path), pairs=list(selected_pairs))
    with _connect(db_path) as conn:
        if not _table_exists(conn, "orderbook_features_1h"):
            summary.warnings.append("orderbook_features_1h does not exist.")
            return summary
        if selected_pairs == ("all",):
            selected_pairs = tuple(_available_feature_pairs(conn))
            summary.pairs = list(selected_pairs)
        violations = {
            "lookahead_source_after_feature_date": 0,
            "source_min_after_source_max": 0,
            "duplicate_feature_rows": 0,
            "missing_source_timestamps_with_ticks": 0,
            "raw_tick_window_mismatch": 0,
            "missing_hourly_feature_rows": 0,
        }
        coverage: dict[str, Any] = {}
        total_rows = 0
        for pair in selected_pairs:
            frame = pd.read_sql_query(
                "SELECT * FROM orderbook_features_1h WHERE canonical_pair = ? ORDER BY date",
                conn,
                params=(pair,),
            )
            if frame.empty:
                summary.warnings.append(f"No compacted feature rows found for {pair}.")
                continue
            total_rows += int(len(frame))
            dates = pd.to_datetime(frame["date"], utc=True, errors="coerce")
            source_min = pd.to_datetime(frame["source_min_ts"], utc=True, errors="coerce")
            source_max = pd.to_datetime(frame["source_max_ts"], utc=True, errors="coerce")
            raw_rows = pd.to_numeric(frame["raw_tick_rows"], errors="coerce").fillna(0)
            violations["lookahead_source_after_feature_date"] += int(((source_max.notna()) & (dates.notna()) & (source_max > dates)).sum())
            violations["source_min_after_source_max"] += int(((source_min.notna()) & (source_max.notna()) & (source_min > source_max)).sum())
            violations["missing_source_timestamps_with_ticks"] += int(((raw_rows > 0) & (source_min.isna() | source_max.isna())).sum())
            dupes = frame.duplicated(subset=["date", "canonical_pair"]).sum()
            violations["duplicate_feature_rows"] += int(dupes)
            full_hours = pd.date_range(dates.min(), dates.max(), freq="1h") if dates.notna().any() else pd.DatetimeIndex([])
            missing_hours = full_hours.difference(pd.DatetimeIndex(dates.dropna()))
            violations["missing_hourly_feature_rows"] += int(len(missing_hours))
            pair_values = _pair_values(conn, pair)
            mismatch_count = _raw_tick_window_mismatch_count(conn, pair_values, frame)
            violations["raw_tick_window_mismatch"] += int(mismatch_count)
            coverage[pair] = {
                "rows": int(len(frame)),
                "first_feature_hour": str(dates.min()) if dates.notna().any() else None,
                "last_feature_hour": str(dates.max()) if dates.notna().any() else None,
                "missing_hourly_rows": int(len(missing_hours)),
                "raw_tick_rows": int(raw_rows.sum()),
                "coverage_mean": _column_mean(frame, "ob1h_x_coverage_mean"),
                "coverage_min": _column_min(frame, "ob1h_x_coverage_mean"),
                "coverage_max": _column_max(frame, "ob1h_x_coverage_mean"),
                "market_ready_count_mean": _column_mean(frame, "ob1h_x_market_ready_count"),
                "hours_all_markets_ready": int((pd.to_numeric(frame.get("ob1h_x_market_ready_count"), errors="coerce") == len(DEFAULT_MARKET_KEYS)).sum())
                if "ob1h_x_market_ready_count" in frame
                else 0,
                "hours_no_markets_ready": int((pd.to_numeric(frame.get("ob1h_x_market_ready_count"), errors="coerce") == 0).sum())
                if "ob1h_x_market_ready_count" in frame
                else 0,
                "raw_tick_window_mismatches": int(mismatch_count),
            }
        summary.rows_checked = total_rows
        summary.violations = violations
        summary.coverage = coverage
        for name, count in violations.items():
            if count:
                summary.warnings.append(f"{name}: {count}")
    return summary


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def _init_feature_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS orderbook_features_1h (
            date TEXT NOT NULL,
            canonical_pair TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            source_min_ts TEXT,
            source_max_ts TEXT,
            raw_tick_rows INTEGER NOT NULL,
            market_context_rows INTEGER NOT NULL,
            feature_schema_version TEXT NOT NULL,
            feature_config_hash TEXT NOT NULL,
            PRIMARY KEY (date, canonical_pair)
        );
        CREATE TABLE IF NOT EXISTS orderbook_feature_build_state (
            canonical_pair TEXT PRIMARY KEY,
            last_successful_build_time TEXT NOT NULL,
            last_raw_tick_seen TEXT,
            last_feature_hour_built TEXT,
            schema_version TEXT NOT NULL,
            feature_config_hash TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_orderbook_features_pair_date
            ON orderbook_features_1h(canonical_pair, date);
        """
    )


def _ensure_feature_columns(conn: sqlite3.Connection, feature_columns: list[str]) -> None:
    existing = {str(row[1]) for row in conn.execute("PRAGMA table_info(orderbook_features_1h)").fetchall()}
    for column in feature_columns:
        if column not in existing:
            conn.execute(f"ALTER TABLE orderbook_features_1h ADD COLUMN {column} REAL")


def _write_feature_rows(conn: sqlite3.Connection, frame: DataFrame) -> None:
    generated_at = utc_now()
    frame = frame.copy()
    frame["generated_at"] = generated_at
    columns = list(frame.columns)
    placeholders = ", ".join("?" for _ in columns)
    assignments = ", ".join(f"{column}=excluded.{column}" for column in columns if column not in {"date", "canonical_pair"})
    sql = f"""
        INSERT INTO orderbook_features_1h ({', '.join(columns)})
        VALUES ({placeholders})
        ON CONFLICT(date, canonical_pair) DO UPDATE SET {assignments}
    """
    rows = [tuple(_sqlite_value(value) for value in row) for row in frame[columns].itertuples(index=False, name=None)]
    conn.executemany(sql, rows)


def _write_states(conn: sqlite3.Connection, frame: DataFrame) -> None:
    now = utc_now()
    config_hash = feature_config_hash()
    for pair, group in frame.groupby("canonical_pair"):
        conn.execute(
            """
            INSERT OR REPLACE INTO orderbook_feature_build_state (
                canonical_pair, last_successful_build_time, last_raw_tick_seen,
                last_feature_hour_built, schema_version, feature_config_hash, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pair,
                now,
                str(group["source_max_ts"].dropna().max()) if group["source_max_ts"].notna().any() else None,
                str(group["date"].max()),
                SCHEMA_VERSION,
                config_hash,
                now,
            ),
        )


def _available_pairs(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT DISTINCT canonical_pair FROM stream_status ORDER BY canonical_pair").fetchall()
    return [str(row[0]) for row in rows if row[0]]


def _available_feature_pairs(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT DISTINCT canonical_pair FROM orderbook_features_1h ORDER BY canonical_pair").fetchall()
    return [str(row[0]) for row in rows if row[0]]


def _pair_values(conn: sqlite3.Connection, canonical_pair: str) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT pair FROM stream_status WHERE canonical_pair = ? ORDER BY pair",
        (canonical_pair,),
    ).fetchall()
    values = [str(row[0]) for row in rows if row[0]]
    if canonical_pair not in values:
        values.append(canonical_pair)
    margin_value = f"{canonical_pair}:USDT"
    if margin_value not in values:
        values.append(margin_value)
    return values


def _raw_tick_stats(conn: sqlite3.Connection, pair_values: list[str]) -> dict[str, Any]:
    placeholders = ", ".join("?" for _ in pair_values)
    row = conn.execute(
        f"""
        SELECT COUNT(*) AS rows, MIN(ts) AS min_ts, MAX(ts) AS max_ts
        FROM orderbook_metric_ticks
        WHERE pair IN ({placeholders})
        """,
        pair_values,
    ).fetchone()
    return {"rows": int(row["rows"] or 0), "min_ts": row["min_ts"], "max_ts": row["max_ts"]}


def _planned_tick_count(conn: sqlite3.Connection, pair_values: list[str], start: pd.Timestamp, end: pd.Timestamp) -> int:
    placeholders = ", ".join("?" for _ in pair_values)
    row = conn.execute(
        f"""
        SELECT COUNT(*) AS rows
        FROM orderbook_metric_ticks
        WHERE pair IN ({placeholders})
          AND ts >= ?
          AND ts < ?
        """,
        [*pair_values, _iso(start), _iso(end)],
    ).fetchone()
    return int(row["rows"] or 0)


def _raw_tick_window_mismatch_count(conn: sqlite3.Connection, pair_values: list[str], frame: DataFrame) -> int:
    mismatches = 0
    for row in frame[["date", "raw_tick_rows"]].itertuples(index=False):
        date = _parse_ts(str(row.date))
        start = date - pd.Timedelta(hours=1)
        end = date
        expected = _planned_tick_count(conn, pair_values, start, end)
        stored = int(row.raw_tick_rows or 0)
        if expected != stored:
            mismatches += 1
    return mismatches


def _build_window(
    conn: sqlite3.Connection,
    canonical_pair: str,
    stats: dict[str, Any],
    *,
    mode: str,
    overlap_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    min_ts = _parse_ts(str(stats["min_ts"]))
    max_ts = _parse_ts(str(stats["max_ts"]))
    first_hour = min_ts.floor("h") + pd.Timedelta(hours=1)
    last_closed_hour = max_ts.floor("h")
    state = None
    if _table_exists(conn, "orderbook_feature_build_state"):
        state = conn.execute(
            "SELECT * FROM orderbook_feature_build_state WHERE canonical_pair = ?",
            (canonical_pair,),
        ).fetchone()
    if mode == "full-rebuild" or state is None or not state["last_feature_hour_built"]:
        return first_hour, last_closed_hour
    previous = _parse_ts(str(state["last_feature_hour_built"]))
    start = max(first_hour, (previous - pd.Timedelta(days=max(1, int(overlap_days)))).floor("h"))
    return start, last_closed_hour


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _load_ticks(conn: sqlite3.Connection, pair_values: list[str], start: pd.Timestamp, end: pd.Timestamp) -> DataFrame:
    placeholders = ", ".join("?" for _ in pair_values)
    columns = [
        "ts",
        "market_key",
        "canonical_pair",
        "pair",
        "book_valid",
        "best_bid",
        "best_ask",
        "mid_price",
        "message_count_interval",
        *RAW_NUMERIC_COLUMNS,
        "bid_wall_candidates_json",
        "ask_wall_candidates_json",
        "bid_liquidity_zones_json",
        "ask_liquidity_zones_json",
        "strong_bid_pressure",
        "strong_ask_pressure",
        "extreme_bid_pressure",
        "extreme_ask_pressure",
    ]
    query = f"""
        SELECT {', '.join(columns)}
        FROM orderbook_metric_ticks
        WHERE pair IN ({placeholders})
          AND ts >= ?
          AND ts < ?
        ORDER BY ts, market_key
    """
    return pd.read_sql_query(query, conn, params=[*pair_values, _iso(start), _iso(end)])


def _load_market_context(conn: sqlite3.Connection, canonical_pair: str, start: pd.Timestamp, end: pd.Timestamp) -> DataFrame:
    columns = ["ts", "market_key", "canonical_pair", *MARKET_CONTEXT_COLUMNS]
    query = f"""
        SELECT {', '.join(columns)}
        FROM market_context_ticks
        WHERE canonical_pair = ?
          AND ts >= ?
          AND ts < ?
        ORDER BY ts, market_key
    """
    return pd.read_sql_query(query, conn, params=[canonical_pair, _iso(start), _iso(end)])


def _build_pair_features(
    ticks: DataFrame,
    context: DataFrame,
    canonical_pair: str,
    build_start: pd.Timestamp,
    build_end: pd.Timestamp,
    *,
    min_coverage_ratio: float,
    ohlcv_path: Path | None,
) -> DataFrame:
    prepared = ticks.copy()
    prepared["ts"] = pd.to_datetime(prepared["ts"], utc=True, errors="coerce")
    prepared = prepared.dropna(subset=["ts"])
    prepared["date"] = prepared["ts"].dt.floor("h") + pd.Timedelta(hours=1)
    prepared = prepared[(prepared["date"] >= build_start) & (prepared["date"] <= build_end)].copy()
    for column in [*RAW_NUMERIC_COLUMNS, "book_valid", "message_count_interval", "strong_bid_pressure", "strong_ask_pressure", "extreme_bid_pressure", "extreme_ask_pressure"]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    hours = pd.date_range(build_start, build_end, freq="1h")
    rows: list[dict[str, Any]] = []
    market_features_by_hour: dict[pd.Timestamp, dict[str, dict[str, float]]] = {hour: {} for hour in hours}
    grouped = prepared.groupby(["date", "market_key"], sort=True)
    for (date, market_key), group in grouped:
        market = _safe_column_part(str(market_key))
        features = _market_tick_features(group, market, min_coverage_ratio=min_coverage_ratio)
        market_features_by_hour[pd.Timestamp(date)][str(market_key)] = features

    context_features_by_hour = _context_features(context, build_start, build_end)
    config_hash = feature_config_hash()
    for hour in hours:
        hour_ticks = prepared[prepared["date"].eq(hour)]
        actual_ts = pd.to_datetime(hour_ticks["ts"], utc=True, errors="coerce").dropna() if not hour_ticks.empty else Series(dtype="datetime64[ns, UTC]")
        row: dict[str, Any] = {
            "date": _iso(hour),
            "canonical_pair": canonical_pair,
            "generated_at": utc_now(),
            "source_min_ts": _iso(pd.Timestamp(actual_ts.min())) if not actual_ts.empty else None,
            "source_max_ts": _iso(pd.Timestamp(actual_ts.max())) if not actual_ts.empty else None,
            "raw_tick_rows": int(len(hour_ticks)),
            "market_context_rows": int(context_features_by_hour.get(hour, {}).get("_rows", 0)),
            "feature_schema_version": SCHEMA_VERSION,
            "feature_config_hash": config_hash,
        }
        for market_key in DEFAULT_MARKET_KEYS:
            market = _safe_column_part(market_key)
            features = market_features_by_hour[hour].get(market_key)
            if features:
                row.update(features)
            else:
                row.update(_empty_market_features(market))
        for key, value in context_features_by_hour.get(hour, {}).items():
            if key != "_rows":
                row[key] = value
        row.update(_cross_market_features(row))
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame = _append_rolling_features(frame)
    frame = _append_orderbook_behaviour_features(frame)
    frame = append_objective_alpha_features(frame, ohlcv_path=ohlcv_path, canonical_pair=canonical_pair)
    return frame


def _market_tick_features(group: DataFrame, market: str, *, min_coverage_ratio: float) -> dict[str, float]:
    prefix = f"ob1h_{market}"
    row_count = int(len(group))
    valid = pd.to_numeric(group["book_valid"], errors="coerce").eq(1)
    # Older crossed books could carry book_valid=1 with absent metrics.
    for column in ("best_bid", "best_ask", "mid_price", "spread_bps"):
        values = _numeric(group, column)
        valid &= np.isfinite(values) & values.gt(0.0)
    valid &= _numeric(group, "best_ask").gt(_numeric(group, "best_bid"))
    for column in RAW_NUMERIC_COLUMNS:
        values = _numeric(group, column)
        valid &= values.isna() | np.isfinite(values)
    for depth in (1, 5, 10, 20):
        for side in ("bid", "ask"):
            values = _numeric(group, f"{side}_notional_top{depth}")
            valid &= np.isfinite(values) & values.ge(0.0)
    group = group.loc[valid]
    valid_count = len(group)
    ts = pd.to_datetime(group["ts"], utc=True, errors="coerce").dropna().sort_values()
    coverage = min(1.0, valid_count / DEFAULT_EXPECTED_SAMPLES_PER_HOUR)
    valid_ratio = valid_count / row_count if row_count else 0.0
    gaps = ts.diff().dt.total_seconds().dropna()
    out: dict[str, float] = {
        f"{prefix}_tick_rows": float(row_count),
        f"{prefix}_valid_tick_rows": float(valid_count),
        f"{prefix}_coverage_ratio": float(coverage),
        f"{prefix}_valid_book_ratio": float(valid_ratio),
        f"{prefix}_ready": float(coverage >= min_coverage_ratio and valid_ratio > 0.0),
        f"{prefix}_max_gap_seconds": float(gaps.max()) if not gaps.empty else math.nan,
        f"{prefix}_message_count_sum": _sum(group["message_count_interval"]),
        f"{prefix}_message_count_mean": _mean(group["message_count_interval"]),
        f"{prefix}_message_count_max": _max(group["message_count_interval"]),
    }
    for column in RAW_NUMERIC_COLUMNS:
        series = pd.to_numeric(group[column], errors="coerce")
        out.update(_stat_packet(prefix, column, series))

    out[f"{prefix}_total_depth_top1_mean"] = _mean(group["bid_notional_top1"]) + _mean(group["ask_notional_top1"])
    out[f"{prefix}_total_depth_top5_mean"] = _mean(group["bid_notional_top5"]) + _mean(group["ask_notional_top5"])
    out[f"{prefix}_total_depth_top10_mean"] = _mean(group["bid_notional_top10"]) + _mean(group["ask_notional_top10"])
    out[f"{prefix}_total_depth_top20_mean"] = _mean(group["bid_notional_top20"]) + _mean(group["ask_notional_top20"])
    out[f"{prefix}_depth_top20_bid_share"] = _ratio(_mean(group["bid_notional_top20"]), _mean(group["bid_notional_top20"]) + _mean(group["ask_notional_top20"]))
    out[f"{prefix}_depth_thinness_score"] = _ratio(1.0, math.log1p(max(0.0, out[f"{prefix}_total_depth_top20_mean"])))

    bid_pressure = pd.to_numeric(group["strong_bid_pressure"], errors="coerce").fillna(0.0)
    ask_pressure = pd.to_numeric(group["strong_ask_pressure"], errors="coerce").fillna(0.0)
    extreme_bid = pd.to_numeric(group["extreme_bid_pressure"], errors="coerce").fillna(0.0)
    extreme_ask = pd.to_numeric(group["extreme_ask_pressure"], errors="coerce").fillna(0.0)
    pressure_sign = np.sign(bid_pressure.to_numpy() - ask_pressure.to_numpy())
    out[f"{prefix}_strong_bid_pressure_ratio"] = float(bid_pressure.mean()) if valid_count else math.nan
    out[f"{prefix}_strong_ask_pressure_ratio"] = float(ask_pressure.mean()) if valid_count else math.nan
    out[f"{prefix}_extreme_bid_pressure_ratio"] = float(extreme_bid.mean()) if valid_count else math.nan
    out[f"{prefix}_extreme_ask_pressure_ratio"] = float(extreme_ask.mean()) if valid_count else math.nan
    out[f"{prefix}_pressure_delta"] = out[f"{prefix}_strong_bid_pressure_ratio"] - out[f"{prefix}_strong_ask_pressure_ratio"]
    out[f"{prefix}_extreme_pressure_delta"] = out[f"{prefix}_extreme_bid_pressure_ratio"] - out[f"{prefix}_extreme_ask_pressure_ratio"]
    out[f"{prefix}_pressure_flip_count"] = float(_sign_flip_count(pressure_sign))
    out[f"{prefix}_pressure_agreement_ratio"] = float(abs(np.nansum(pressure_sign)) / len(pressure_sign)) if len(pressure_sign) else math.nan

    out[f"{prefix}_wall_support_resistance_delta"] = _max(group["strongest_bid_wall_score_50bps"]) - _max(group["strongest_ask_wall_score_50bps"])
    out[f"{prefix}_near_bid_wall_ratio_10bps"] = _condition_ratio(group["nearest_bid_wall_distance_bps"], lambda series: series <= 10.0)
    out[f"{prefix}_near_ask_wall_ratio_10bps"] = _condition_ratio(group["nearest_ask_wall_distance_bps"], lambda series: series <= 10.0)
    out[f"{prefix}_near_bid_wall_ratio_25bps"] = _condition_ratio(group["nearest_bid_wall_distance_bps"], lambda series: series <= 25.0)
    out[f"{prefix}_near_ask_wall_ratio_25bps"] = _condition_ratio(group["nearest_ask_wall_distance_bps"], lambda series: series <= 25.0)
    out.update(_json_wall_features(group, "bid", prefix))
    out.update(_json_wall_features(group, "ask", prefix))
    out.update(_json_liquidity_zone_features(group, "bid", prefix))
    out.update(_json_liquidity_zone_features(group, "ask", prefix))
    return out


def _empty_market_features(market: str) -> dict[str, float]:
    prefix = f"ob1h_{market}"
    return {
        f"{prefix}_tick_rows": 0.0,
        f"{prefix}_valid_tick_rows": 0.0,
        f"{prefix}_coverage_ratio": 0.0,
        f"{prefix}_valid_book_ratio": 0.0,
        f"{prefix}_ready": 0.0,
        f"{prefix}_message_count_sum": 0.0,
    }


def _stat_packet(prefix: str, name: str, series: Series) -> dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {
            f"{prefix}_{name}_mean": math.nan,
            f"{prefix}_{name}_p50": math.nan,
            f"{prefix}_{name}_p95": math.nan,
            f"{prefix}_{name}_std": math.nan,
            f"{prefix}_{name}_min": math.nan,
            f"{prefix}_{name}_max": math.nan,
            f"{prefix}_{name}_last": math.nan,
            f"{prefix}_{name}_slope": math.nan,
        }
    return {
        f"{prefix}_{name}_mean": float(numeric.mean()),
        f"{prefix}_{name}_p50": float(numeric.quantile(0.50)),
        f"{prefix}_{name}_p95": float(numeric.quantile(0.95)),
        f"{prefix}_{name}_std": float(numeric.std(ddof=0)),
        f"{prefix}_{name}_min": float(numeric.min()),
        f"{prefix}_{name}_max": float(numeric.max()),
        f"{prefix}_{name}_last": float(numeric.iloc[-1]),
        f"{prefix}_{name}_slope": _slope(numeric),
    }


def _context_features(context: DataFrame, build_start: pd.Timestamp, build_end: pd.Timestamp) -> dict[pd.Timestamp, dict[str, float]]:
    if context.empty:
        return {}
    prepared = context.copy()
    prepared["ts"] = pd.to_datetime(prepared["ts"], utc=True, errors="coerce")
    prepared = prepared.dropna(subset=["ts"])
    prepared["date"] = prepared["ts"].dt.floor("h") + pd.Timedelta(hours=1)
    prepared = prepared[(prepared["date"] >= build_start) & (prepared["date"] <= build_end)].copy()
    for column in MARKET_CONTEXT_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    output: dict[pd.Timestamp, dict[str, float]] = {}
    for (date, market_key), group in prepared.groupby(["date", "market_key"], sort=True):
        market = _safe_column_part(str(market_key))
        prefix = f"ob1h_{market}_ctx"
        row = output.setdefault(pd.Timestamp(date), {"_rows": 0.0})
        row["_rows"] += float(len(group))
        for column in MARKET_CONTEXT_COLUMNS:
            series = pd.to_numeric(group[column], errors="coerce").dropna()
            row[f"{prefix}_{column}_mean"] = float(series.mean()) if not series.empty else math.nan
            row[f"{prefix}_{column}_last"] = float(series.iloc[-1]) if not series.empty else math.nan
        buy = pd.to_numeric(group["taker_buy_volume"], errors="coerce").sum()
        sell = pd.to_numeric(group["taker_sell_volume"], errors="coerce").sum()
        row[f"{prefix}_taker_pressure"] = _ratio(buy - sell, buy + sell)
        oi = pd.to_numeric(group["open_interest"], errors="coerce").dropna()
        row[f"{prefix}_open_interest_delta"] = float(oi.iloc[-1] - oi.iloc[0]) if len(oi) >= 2 else math.nan
    return output


def _cross_market_features(row: dict[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    market_parts = [_safe_column_part(value) for value in DEFAULT_MARKET_KEYS]
    ready_values = [_float(row.get(f"ob1h_{market}_ready")) for market in market_parts]
    ready_count = sum(1 for value in ready_values if value >= 1.0)
    output["ob1h_x_market_ready_count"] = float(ready_count)
    output["ob1h_x_market_ready_ratio"] = ready_count / len(market_parts)
    coverage = [_float(row.get(f"ob1h_{market}_coverage_ratio")) for market in market_parts]
    output["ob1h_x_coverage_mean"] = float(np.nanmean(coverage)) if coverage else math.nan
    output["ob1h_x_coverage_min"] = float(np.nanmin(coverage)) if coverage else math.nan

    metric_map = {
        "pressure_delta": "pressure_delta",
        "extreme_pressure_delta": "extreme_pressure_delta",
        "imbalance_top20_mean": "imbalance_top20_mean",
        "imbalance_10bps_mean": "imbalance_10bps_mean",
        "spread_bps_mean": "spread_bps_mean",
        "microprice_offset_bps_mean": "microprice_offset_bps_mean",
        "total_depth_top20_mean": "total_depth_top20_mean",
        "wall_support_resistance_delta": "wall_support_resistance_delta",
    }
    for output_name, suffix in metric_map.items():
        values = [_float(row.get(f"ob1h_{market}_{suffix}")) for market in market_parts]
        valid = [value for value in values if not math.isnan(value)]
        output[f"ob1h_x_{output_name}"] = float(np.nanmean(valid)) if valid else math.nan
        output[f"ob1h_x_{output_name}_std"] = float(np.nanstd(valid)) if valid else math.nan
        output[f"ob1h_x_{output_name}_max_abs"] = float(np.nanmax(np.abs(valid))) if valid else math.nan
        output[f"ob1h_x_{output_name}_agreement"] = _sign_agreement(valid)

    pairs = (
        ("binance_usdm_futures", "binance_spot", "binance_futures_minus_spot"),
        ("bybit_linear", "bybit_spot", "bybit_linear_minus_spot"),
        ("bybit_linear", "binance_usdm_futures", "bybit_linear_minus_binance_futures"),
        ("bybit_spot", "binance_spot", "bybit_spot_minus_binance_spot"),
    )
    for left, right, label in pairs:
        left_part = _safe_column_part(left)
        right_part = _safe_column_part(right)
        for suffix in ("pressure_delta", "imbalance_top20_mean", "spread_bps_mean", "total_depth_top20_mean", "wall_support_resistance_delta"):
            output[f"ob1h_x_{label}_{suffix}"] = _float(row.get(f"ob1h_{left_part}_{suffix}")) - _float(row.get(f"ob1h_{right_part}_{suffix}"))

    long_scores = []
    short_scores = []
    for market in market_parts:
        pressure = _float(row.get(f"ob1h_{market}_pressure_delta"))
        imbalance = _float(row.get(f"ob1h_{market}_imbalance_top20_mean"))
        wall = _float(row.get(f"ob1h_{market}_wall_support_resistance_delta"))
        ready = _float(row.get(f"ob1h_{market}_ready"))
        if ready <= 0:
            continue
        long_scores.append(_clip01((pressure + 1.0) / 2.0) * 0.40 + _clip01((imbalance + 1.0) / 2.0) * 0.35 + _clip01((wall + 10.0) / 20.0) * 0.25)
        short_scores.append(_clip01((-pressure + 1.0) / 2.0) * 0.40 + _clip01((-imbalance + 1.0) / 2.0) * 0.35 + _clip01((-wall + 10.0) / 20.0) * 0.25)
    output["ob1h_x_confluence_long_score"] = float(np.nanmean(long_scores)) if long_scores else math.nan
    output["ob1h_x_confluence_short_score"] = float(np.nanmean(short_scores)) if short_scores else math.nan
    output["ob1h_x_confluence_delta"] = output["ob1h_x_confluence_long_score"] - output["ob1h_x_confluence_short_score"]
    return output


def _append_rolling_features(frame: DataFrame) -> DataFrame:
    output = frame.sort_values(["canonical_pair", "date"]).reset_index(drop=True).copy()
    selected = list(CROSS_ROLLING_COLUMNS)
    for market_key in DEFAULT_MARKET_KEYS:
        market = _safe_column_part(market_key)
        selected.extend(f"ob1h_{market}_{suffix}" for suffix in ROLLING_SOURCE_SUFFIXES)
    rolling_frames: list[DataFrame] = []
    for column in selected:
        if column not in output:
            continue
        values = pd.to_numeric(output[column], errors="coerce")
        grouped = values.groupby(output["canonical_pair"])
        rolling_frames.append(
            DataFrame(
                {
                    f"{column}_delta_1h": grouped.diff(1),
                    f"{column}_delta_6h": grouped.diff(6),
                    f"{column}_z_24h": grouped.transform(lambda series: _rolling_zscore(series, 24, 12)),
                    f"{column}_z_7d": grouped.transform(lambda series: _rolling_zscore(series, 168, 48)),
                },
                index=output.index,
            )
        )
    if not rolling_frames:
        return output
    return pd.concat([output, *rolling_frames], axis=1).copy()


def _rolling_zscore(series: Series, window: int, min_periods: int) -> Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / std.replace(0, np.nan)


def _append_orderbook_behaviour_features(frame: DataFrame) -> DataFrame:
    output = frame.sort_values(["canonical_pair", "date"]).reset_index(drop=True).copy()
    behaviour_frames: list[DataFrame] = []
    for market_key in DEFAULT_MARKET_KEYS:
        market = _safe_column_part(market_key)
        prefix = f"ob1h_{market}"
        if f"{prefix}_ready" not in output:
            continue
        behaviour = _market_behaviour_features(output, prefix)
        if not behaviour.empty:
            behaviour_frames.append(behaviour)
    if not behaviour_frames:
        return output
    return pd.concat([output, *behaviour_frames], axis=1).copy()


def _market_behaviour_features(frame: DataFrame, prefix: str) -> DataFrame:
    grouped_key = frame["canonical_pair"]
    tick_rows = _numeric(frame, f"{prefix}_tick_rows").fillna(0.0)
    valid_ratio = _numeric(frame, f"{prefix}_valid_book_ratio").fillna(0.0)
    active = ((tick_rows > 0.0) & (valid_ratio > 0.0)).astype(float)
    pressure = _numeric(frame, f"{prefix}_pressure_delta")
    ask_distance = _numeric(frame, f"{prefix}_nearest_ask_wall_distance_bps_min")
    bid_distance = _numeric(frame, f"{prefix}_nearest_bid_wall_distance_bps_min")
    ask_notional = _numeric(frame, f"{prefix}_nearest_ask_wall_notional_p95")
    bid_notional = _numeric(frame, f"{prefix}_nearest_bid_wall_notional_p95")
    ask_score = _numeric(frame, f"{prefix}_nearest_ask_wall_score_p95")
    bid_score = _numeric(frame, f"{prefix}_nearest_bid_wall_score_p95")
    ask_liquidity = _numeric(frame, f"{prefix}_ask_liquidity_5bps_min")
    bid_liquidity = _numeric(frame, f"{prefix}_bid_liquidity_5bps_min")
    spread_z = _numeric(frame, f"{prefix}_spread_bps_p95_z_24h")
    pressure_z = _numeric(frame, f"{prefix}_pressure_delta_z_24h")
    ask_near = _numeric(frame, f"{prefix}_near_ask_wall_ratio_25bps")
    bid_near = _numeric(frame, f"{prefix}_near_bid_wall_ratio_25bps")

    ask_notional_drop = _positive_pct_drop(ask_notional, grouped_key)
    bid_notional_drop = _positive_pct_drop(bid_notional, grouped_key)
    ask_score_drop = _positive_pct_drop(ask_score, grouped_key)
    bid_score_drop = _positive_pct_drop(bid_score, grouped_key)
    ask_distance_rise = _positive_diff_score(ask_distance, grouped_key, 25.0)
    bid_distance_rise = _positive_diff_score(bid_distance, grouped_key, 25.0)
    ask_wall_closer = _negative_diff_score(ask_distance, grouped_key, 25.0)
    bid_wall_closer = _negative_diff_score(bid_distance, grouped_key, 25.0)
    bullish_pressure_shift = _clip01_series(_positive_group_diff(pressure, grouped_key) / 1.0).combine(
        _clip01_series(pressure_z / 3.0),
        max,
    )
    bearish_pressure_shift = _clip01_series(_negative_group_diff(pressure, grouped_key) / 1.0).combine(
        _clip01_series(-pressure_z / 3.0),
        max,
    )

    ask_wall_evaporation = _row_mean(ask_notional_drop, ask_score_drop, ask_distance_rise) * active
    bid_wall_evaporation = _row_mean(bid_notional_drop, bid_score_drop, bid_distance_rise) * active
    spread_shock = _clip01_series(spread_z / 3.0) * active
    liquidity_vacuum_up = _clip01_series(-_numeric(frame, f"{prefix}_ask_liquidity_5bps_min_z_24h") / 3.0) * active
    liquidity_vacuum_down = _clip01_series(-_numeric(frame, f"{prefix}_bid_liquidity_5bps_min_z_24h") / 3.0) * active

    persistent_ask_6h = _rolling_mean(ask_near, grouped_key, 6, 3) * active
    persistent_bid_6h = _rolling_mean(bid_near, grouped_key, 6, 3) * active
    persistent_ask_24h = _rolling_mean(ask_near, grouped_key, 24, 12) * active
    persistent_bid_24h = _rolling_mean(bid_near, grouped_key, 24, 12) * active
    average_zone_distance = (ask_distance + bid_distance) / 2.0
    zone_compression = _clip01_series(1.0 - average_zone_distance / 50.0) * _row_mean(persistent_ask_6h, persistent_bid_6h) * active

    resistance_removed = _row_mean(ask_wall_evaporation, ask_distance_rise, bullish_pressure_shift) * active
    support_removed = _row_mean(bid_wall_evaporation, bid_distance_rise, bearish_pressure_shift) * active
    bullish_impulse = _row_mean(resistance_removed, bullish_pressure_shift, liquidity_vacuum_up, bid_wall_closer) * active
    bearish_impulse = _row_mean(support_removed, bearish_pressure_shift, liquidity_vacuum_down, ask_wall_closer) * active
    false_breakout_risk = _row_mean(persistent_ask_24h, ask_wall_closer, bearish_pressure_shift) * active
    false_breakdown_risk = _row_mean(persistent_bid_24h, bid_wall_closer, bullish_pressure_shift) * active
    support_rebuild = _row_mean(bid_wall_closer, persistent_bid_6h, bullish_pressure_shift) * active
    resistance_rebuild = _row_mean(ask_wall_closer, persistent_ask_6h, bearish_pressure_shift) * active
    fragile_book = _row_mean(spread_shock, liquidity_vacuum_up, liquidity_vacuum_down, _clip01_series(_numeric(frame, f"{prefix}_depth_thinness_score_z_24h") / 3.0)) * active

    return DataFrame(
        {
            f"{prefix}_beh_ask_wall_evaporation_1h": ask_wall_evaporation,
            f"{prefix}_beh_bid_wall_evaporation_1h": bid_wall_evaporation,
            f"{prefix}_beh_ask_wall_closer_1h": ask_wall_closer * active,
            f"{prefix}_beh_bid_wall_closer_1h": bid_wall_closer * active,
            f"{prefix}_beh_bullish_pressure_shift_1h": bullish_pressure_shift * active,
            f"{prefix}_beh_bearish_pressure_shift_1h": bearish_pressure_shift * active,
            f"{prefix}_beh_spread_shock_24h": spread_shock,
            f"{prefix}_beh_liquidity_vacuum_up": liquidity_vacuum_up,
            f"{prefix}_beh_liquidity_vacuum_down": liquidity_vacuum_down,
            f"{prefix}_beh_persistent_ask_zone_6h": persistent_ask_6h,
            f"{prefix}_beh_persistent_bid_zone_6h": persistent_bid_6h,
            f"{prefix}_beh_persistent_ask_zone_24h": persistent_ask_24h,
            f"{prefix}_beh_persistent_bid_zone_24h": persistent_bid_24h,
            f"{prefix}_beh_zone_compression_score": zone_compression,
            f"{prefix}_beh_resistance_removed_score": resistance_removed,
            f"{prefix}_beh_support_removed_score": support_removed,
            f"{prefix}_beh_bullish_impulse_score": bullish_impulse,
            f"{prefix}_beh_bearish_impulse_score": bearish_impulse,
            f"{prefix}_beh_false_breakout_risk": false_breakout_risk,
            f"{prefix}_beh_false_breakdown_risk": false_breakdown_risk,
            f"{prefix}_beh_post_breakout_support_rebuild": support_rebuild,
            f"{prefix}_beh_post_breakdown_resistance_rebuild": resistance_rebuild,
            f"{prefix}_beh_fragile_book_score": fragile_book,
            f"{prefix}_beh_wall_evaporation_imbalance": ask_wall_evaporation - bid_wall_evaporation,
            f"{prefix}_beh_pressure_impulse_imbalance": bullish_pressure_shift - bearish_pressure_shift,
        },
        index=frame.index,
    )


def append_objective_alpha_features(
    frame: DataFrame,
    *,
    ohlcv_path: Path | None = DEFAULT_OHLCV_PATH,
    canonical_pair: str | None = None,
) -> DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"], utc=True, errors="coerce")
    alpha_columns = [column for column in output.columns if column.startswith("ob1h_alpha_") or "_alpha_" in column]
    if alpha_columns:
        output = output.drop(columns=alpha_columns)
    price_context = _load_alpha_price_context(ohlcv_path)
    if price_context.empty:
        return output
    start = output["date"].min()
    end = output["date"].max()
    price_context = price_context[(price_context["date"] >= start) & (price_context["date"] <= end)].copy()
    if price_context.empty:
        return output

    merged = output[["date"]].merge(price_context, on="date", how="left", sort=False)
    output = pd.concat([output.reset_index(drop=True), merged.drop(columns=["date"]).reset_index(drop=True)], axis=1)
    if canonical_pair and "canonical_pair" not in output:
        output["canonical_pair"] = canonical_pair

    alpha_frames: list[DataFrame] = []
    for market_key in DEFAULT_MARKET_KEYS:
        market = _safe_column_part(market_key)
        prefix = f"ob1h_{market}"
        if f"{prefix}_ready" not in output:
            continue
        alpha = _market_objective_alpha_features(output, prefix)
        if not alpha.empty:
            alpha_frames.append(alpha)
    temp_columns = [column for column in output.columns if column.startswith("__alpha_")]
    output = output.drop(columns=temp_columns)
    if not alpha_frames:
        return output
    return pd.concat([output, *alpha_frames], axis=1).copy()


def _load_alpha_price_context(ohlcv_path: Path | None) -> DataFrame:
    if ohlcv_path is None:
        return DataFrame()
    path = Path(ohlcv_path)
    if not path.exists():
        return DataFrame()
    if path.suffix.lower() == ".parquet":
        prices = pd.read_parquet(path)
    else:
        prices = pd.read_feather(path)
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(prices.columns):
        return DataFrame()
    prices = prices.loc[:, ["date", "open", "high", "low", "close", "volume"]].copy()
    prices["date"] = pd.to_datetime(prices["date"], utc=True, errors="coerce")
    prices = prices.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume"):
        prices[column] = pd.to_numeric(prices[column], errors="coerce")

    close = prices["close"]
    high = prices["high"]
    low = prices["low"]
    candle_range = (high - low).replace(0.0, np.nan)
    true_range_pct = candle_range / close.replace(0.0, np.nan)
    body_pressure = ((close - prices["open"]) / candle_range).clip(-1.0, 1.0).fillna(0.0)
    signed_volume = body_pressure * prices["volume"]

    features = DataFrame(
        {
            "date": prices["date"] + pd.Timedelta(hours=1),
            "__alpha_open": prices["open"],
            "__alpha_high": high,
            "__alpha_low": low,
            "__alpha_close": close,
            "__alpha_volume": prices["volume"],
            "__alpha_return_1h": close.pct_change(1, fill_method=None),
            "__alpha_return_3h": close.pct_change(3, fill_method=None),
            "__alpha_return_6h": close.pct_change(6, fill_method=None),
            "__alpha_return_24h": close.pct_change(24, fill_method=None),
        }
    )
    for window in (24, 72):
        rolling_high = high.rolling(window, min_periods=window // 2).max()
        rolling_low = low.rolling(window, min_periods=window // 2).min()
        rolling_range = (rolling_high - rolling_low).replace(0.0, np.nan)
        range_position = ((close - rolling_low) / rolling_range).clip(0.0, 1.0)
        trend_return = close / close.shift(window) - 1.0
        features[f"ob1h_alpha_range_position_{window}h"] = range_position
        features[f"ob1h_alpha_price_trend_state_{window}h"] = np.select(
            [trend_return >= 0.01, trend_return <= -0.01],
            [1.0, -1.0],
            default=0.0,
        )

    higher_high = (high > high.shift(1)).astype(float)
    higher_low = (low > low.shift(1)).astype(float)
    lower_high = (high < high.shift(1)).astype(float)
    lower_low = (low < low.shift(1)).astype(float)
    features["ob1h_alpha_higher_high_sequence"] = (higher_high + higher_low).rolling(6, min_periods=3).sum()
    features["ob1h_alpha_lower_low_sequence"] = (lower_high + lower_low).rolling(6, min_periods=3).sum()
    short_vol = true_range_pct.rolling(6, min_periods=3).mean()
    medium_vol = true_range_pct.rolling(24, min_periods=12).mean()
    features["ob1h_alpha_volatility_expansion_state"] = (short_vol / medium_vol.replace(0.0, np.nan) - 1.0).clip(-3.0, 3.0)
    features["ob1h_alpha_volume_pressure_6h"] = (
        signed_volume.rolling(6, min_periods=3).sum() / prices["volume"].rolling(6, min_periods=3).sum().replace(0.0, np.nan)
    )
    rolling_range_3h = (high.rolling(3, min_periods=2).max() - low.rolling(3, min_periods=2).min()) / close.replace(0.0, np.nan)
    rolling_range_24h = (high.rolling(24, min_periods=12).max() - low.rolling(24, min_periods=12).min()) / close.replace(0.0, np.nan)
    features["ob1h_alpha_price_stall_3h"] = (
        rolling_range_3h <= rolling_range_24h.rolling(168, min_periods=48).quantile(0.35)
    ).astype(float)
    return features


def _market_objective_alpha_features(frame: DataFrame, prefix: str) -> DataFrame:
    grouped_key = frame["canonical_pair"] if "canonical_pair" in frame else Series("BTC/USDT", index=frame.index)
    close = _numeric(frame, "__alpha_close")
    high = _numeric(frame, "__alpha_high")
    low = _numeric(frame, "__alpha_low")
    price_return_1h = _numeric(frame, "__alpha_return_1h").fillna(0.0)
    price_return_3h = _numeric(frame, "__alpha_return_3h").fillna(0.0)
    tick_rows = _numeric(frame, f"{prefix}_tick_rows").fillna(0.0)
    valid_ratio = _numeric(frame, f"{prefix}_valid_book_ratio").fillna(0.0)
    active = ((tick_rows > 0.0) & (valid_ratio > 0.0) & close.notna()).astype(float)

    pressure = _numeric(frame, f"{prefix}_pressure_delta").fillna(0.0).clip(-1.0, 1.0)
    pressure_diff = pressure.groupby(grouped_key).diff().fillna(0.0)
    pressure_direction = Series(np.select([pressure >= 0.10, pressure <= -0.10], [1.0, -1.0], default=0.0), index=frame.index)
    price_direction = Series(np.sign(price_return_1h.fillna(0.0)), index=frame.index)
    pressure_price_agreement = ((pressure_direction != 0.0) & (price_direction != 0.0) & (pressure_direction == price_direction)).astype(float)
    pressure_divergence = (
        pressure.abs()
        * (((pressure_direction != 0.0) & (price_direction != 0.0) & (pressure_direction != price_direction)).astype(float)
           + _numeric(frame, "ob1h_alpha_price_stall_3h").fillna(0.0))
    ).clip(0.0, 1.0)

    ask_distance = _numeric(frame, f"{prefix}_nearest_ask_wall_distance_bps_min")
    bid_distance = _numeric(frame, f"{prefix}_nearest_bid_wall_distance_bps_min")
    ask_price = _numeric(frame, f"{prefix}_nearest_ask_wall_price_last")
    bid_price = _numeric(frame, f"{prefix}_nearest_bid_wall_price_last")
    ask_price = ask_price.where(ask_price.notna(), close * (1.0 + ask_distance / 10000.0))
    bid_price = bid_price.where(bid_price.notna(), close * (1.0 - bid_distance / 10000.0))
    ask_score = _numeric(frame, f"{prefix}_nearest_ask_wall_score_p95")
    bid_score = _numeric(frame, f"{prefix}_nearest_bid_wall_score_p95")
    ask_notional = _numeric(frame, f"{prefix}_nearest_ask_wall_notional_p95")
    bid_notional = _numeric(frame, f"{prefix}_nearest_bid_wall_notional_p95")

    resistance = _zone_state(ask_price, ask_distance, ask_score, ask_notional, close, grouped_key, active)
    support = _zone_state(bid_price, bid_distance, bid_score, bid_notional, close, grouped_key, active)
    resistance_distance = resistance["distance_bps"]
    support_distance = support["distance_bps"]
    zone_width = ((ask_price - bid_price) / close.replace(0.0, np.nan) * 10000.0).where(
        ask_price.notna() & bid_price.notna(),
        ask_distance + bid_distance,
    )

    prior_resistance = _group_shift(ask_price, grouped_key, 1)
    prior_support = _group_shift(bid_price, grouped_key, 1)
    approach_resistance = ((resistance_distance <= 50.0) | (((ask_price - high) / close.replace(0.0, np.nan) * 10000.0) <= 50.0)).astype(float) * active
    approach_support = ((support_distance <= 50.0) | (((low - bid_price) / close.replace(0.0, np.nan) * 10000.0) <= 50.0)).astype(float) * active
    touch_resistance = (high >= ask_price * (1.0 - 5.0 / 10000.0)).astype(float) * active
    touch_support = (low <= bid_price * (1.0 + 5.0 / 10000.0)).astype(float) * active
    recent_touch_resistance = (_rolling_sum(touch_resistance, grouped_key, 3, 1) > 0.0).astype(float)
    recent_touch_support = (_rolling_sum(touch_support, grouped_key, 3, 1) > 0.0).astype(float)
    close_above_prior_resistance = (close > prior_resistance * (1.0 + 5.0 / 10000.0)).astype(float).where(prior_resistance.notna(), 0.0)
    close_below_prior_support = (close < prior_support * (1.0 - 5.0 / 10000.0)).astype(float).where(prior_support.notna(), 0.0)
    accepted_above = (_rolling_sum(close_above_prior_resistance, grouped_key, 3, 1) >= 2.0).astype(float) * active
    accepted_below = (_rolling_sum(close_below_prior_support, grouped_key, 3, 1) >= 2.0).astype(float) * active
    rejected_resistance = recent_touch_resistance * (close < prior_resistance * (1.0 - 5.0 / 10000.0)).astype(float).fillna(0.0) * active
    bounced_support = recent_touch_support * (close > prior_support * (1.0 + 5.0 / 10000.0)).astype(float).fillna(0.0) * active

    pressure_pos = pressure.clip(lower=0.0)
    pressure_neg = (-pressure).clip(lower=0.0)
    ask_absorption = touch_resistance * pressure_pos * resistance["stability_score"] * (1.0 - accepted_above)
    bid_absorption = touch_support * pressure_neg * support["stability_score"] * (1.0 - accepted_below)
    ask_attempt_count_6h = _rolling_sum((touch_resistance * (pressure_pos > 0.10).astype(float)), grouped_key, 6, 1) * active
    bid_attempt_count_6h = _rolling_sum((touch_support * (pressure_neg > 0.10).astype(float)), grouped_key, 6, 1) * active
    absorption_exhaustion = _row_mean(ask_absorption, bid_absorption, pressure_divergence) * active

    liquidity_vacuum_up = _numeric(frame, f"{prefix}_beh_liquidity_vacuum_up").fillna(0.0).clip(0.0, 1.0) * active
    liquidity_vacuum_down = _numeric(frame, f"{prefix}_beh_liquidity_vacuum_down").fillna(0.0).clip(0.0, 1.0) * active
    upside_after_removed = liquidity_vacuum_up * resistance["removed_strength_1h"]
    downside_after_removed = liquidity_vacuum_down * support["removed_strength_1h"]
    vacuum_direction_agrees = (
        liquidity_vacuum_up * _clip01_series(price_return_1h / 0.01)
        + liquidity_vacuum_down * _clip01_series(-price_return_1h / 0.01)
    ).clip(0.0, 1.0) * active

    support_rebuild = _row_mean(support["added_strength_1h"], support["stability_score"])
    resistance_rebuild = _row_mean(resistance["added_strength_1h"], resistance["stability_score"])
    breakout_acceptance = accepted_above * _row_mean(support_rebuild, pressure_price_agreement, liquidity_vacuum_up) * active
    breakout_failure = rejected_resistance * _row_mean(resistance["stability_score"], ask_absorption, pressure_divergence) * active
    breakdown_acceptance = accepted_below * _row_mean(resistance_rebuild, pressure_price_agreement, liquidity_vacuum_down) * active
    breakdown_failure = bounced_support * _row_mean(support["stability_score"], bid_absorption, pressure_divergence) * active
    hold_above_hours = _consecutive_boolean_duration(close_above_prior_resistance > 0.0, grouped_key)
    hold_below_hours = _consecutive_boolean_duration(close_below_prior_support > 0.0, grouped_key)
    acceptance_hold_hours = pd.concat([hold_above_hours, hold_below_hours], axis=1).max(axis=1) * active

    pressure_duration = _consecutive_signed_duration(pressure_direction, grouped_key) * active
    pressure_flip_strength = (pressure_diff.abs() * (pressure_direction != _group_shift(pressure_direction, grouped_key, 1)).astype(float)).clip(0.0, 2.0) / 2.0
    pressure_acceleration = pressure_diff.clip(-1.0, 1.0) * active

    p = f"{prefix}_alpha"
    return DataFrame(
        {
            f"{p}_objective_alpha_ready": active,
            f"{p}_nearest_resistance_zone_distance_bps": resistance_distance,
            f"{p}_nearest_support_zone_distance_bps": support_distance,
            f"{p}_resistance_zone_age_hours": resistance["age_hours"],
            f"{p}_support_zone_age_hours": support["age_hours"],
            f"{p}_resistance_zone_strength": resistance["strength"],
            f"{p}_support_zone_strength": support["strength"],
            f"{p}_zone_width_bps": zone_width,
            f"{p}_resistance_zone_added_strength_1h": resistance["added_strength_1h"],
            f"{p}_resistance_zone_removed_strength_1h": resistance["removed_strength_1h"],
            f"{p}_support_zone_added_strength_1h": support["added_strength_1h"],
            f"{p}_support_zone_removed_strength_1h": support["removed_strength_1h"],
            f"{p}_resistance_zone_migration_bps_3h": resistance["migration_bps_3h"],
            f"{p}_support_zone_migration_bps_3h": support["migration_bps_3h"],
            f"{p}_resistance_zone_stability_score": resistance["stability_score"],
            f"{p}_support_zone_stability_score": support["stability_score"],
            f"{p}_price_approached_resistance_1h": approach_resistance,
            f"{p}_price_touched_resistance_1h": touch_resistance,
            f"{p}_price_rejected_resistance_3h": rejected_resistance,
            f"{p}_price_accepted_above_resistance_3h": accepted_above,
            f"{p}_price_approached_support_1h": approach_support,
            f"{p}_price_touched_support_1h": touch_support,
            f"{p}_price_bounced_support_3h": bounced_support,
            f"{p}_price_accepted_below_support_3h": accepted_below,
            f"{p}_ask_absorption_score": ask_absorption,
            f"{p}_bid_absorption_score": bid_absorption,
            f"{p}_ask_absorption_attempt_count_6h": ask_attempt_count_6h,
            f"{p}_bid_absorption_attempt_count_6h": bid_attempt_count_6h,
            f"{p}_absorption_exhaustion_score": absorption_exhaustion,
            f"{p}_breakout_acceptance_score": breakout_acceptance,
            f"{p}_breakout_failure_score": breakout_failure,
            f"{p}_breakdown_acceptance_score": breakdown_acceptance,
            f"{p}_breakdown_failure_score": breakdown_failure,
            f"{p}_acceptance_hold_hours": acceptance_hold_hours,
            f"{p}_upside_liquidity_vacuum_near": liquidity_vacuum_up,
            f"{p}_downside_liquidity_vacuum_near": liquidity_vacuum_down,
            f"{p}_upside_vacuum_after_resistance_removed": upside_after_removed,
            f"{p}_downside_vacuum_after_support_removed": downside_after_removed,
            f"{p}_vacuum_direction_agrees_with_price": vacuum_direction_agrees,
            f"{p}_pressure_direction": pressure_direction * active,
            f"{p}_pressure_duration_hours": pressure_duration,
            f"{p}_pressure_flip_strength": pressure_flip_strength * active,
            f"{p}_pressure_acceleration": pressure_acceleration,
            f"{p}_pressure_price_agreement": pressure_price_agreement * active,
            f"{p}_pressure_divergence": pressure_divergence * active,
        },
        index=frame.index,
    )


def _zone_state(
    zone_price: Series,
    distance_bps: Series,
    score: Series,
    notional: Series,
    close: Series,
    grouped_key: Series,
    active: Series,
) -> dict[str, Series]:
    present = (zone_price.notna() & distance_bps.notna() & (score.fillna(0.0) > 0.0)).astype(float)
    age = _zone_age_hours(zone_price, grouped_key)
    score_component = _clip01_series(np.log1p(score.fillna(0.0)) / math.log1p(8.0))
    notional_rank = notional.groupby(grouped_key).transform(lambda series: series.rank(pct=True)).fillna(0.0)
    near_component = _clip01_series(1.0 - distance_bps / 100.0)
    persistence_6h = _rolling_mean(present, grouped_key, 6, 1)
    strength = _row_mean(score_component, notional_rank, _clip01_series(age / 24.0), near_component, persistence_6h) * active * present
    price_change_bps = (zone_price.groupby(grouped_key).diff().abs() / close.replace(0.0, np.nan) * 10000.0)
    stability = strength * _clip01_series(1.0 - _rolling_mean(price_change_bps, grouped_key, 6, 1) / 25.0)
    return {
        "distance_bps": distance_bps * active,
        "age_hours": age * active,
        "strength": strength,
        "added_strength_1h": _positive_group_diff(strength, grouped_key) * active,
        "removed_strength_1h": _negative_group_diff(strength, grouped_key) * active,
        "migration_bps_3h": (zone_price.groupby(grouped_key).diff(3) / close.replace(0.0, np.nan) * 10000.0).fillna(0.0) * active,
        "stability_score": stability,
    }


def _zone_age_hours(zone_price: Series, grouped_key: Series, tolerance_bps: float = 15.0) -> Series:
    output = Series(0.0, index=zone_price.index, dtype="float64")
    for _, index in zone_price.groupby(grouped_key).groups.items():
        current_price = math.nan
        age = 0.0
        for row_index in index:
            price = _float(zone_price.loc[row_index])
            if math.isnan(price):
                current_price = math.nan
                age = 0.0
                output.loc[row_index] = 0.0
                continue
            if math.isnan(current_price):
                current_price = price
                age = 1.0
            else:
                distance = abs((price - current_price) / current_price * 10000.0) if current_price else math.inf
                if distance <= tolerance_bps:
                    age += 1.0
                    current_price = (0.80 * current_price) + (0.20 * price)
                else:
                    current_price = price
                    age = 1.0
            output.loc[row_index] = age
    return output


def _numeric(frame: DataFrame, column: str) -> Series:
    if column not in frame:
        return Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _positive_group_diff(series: Series, grouped_key: Series) -> Series:
    return series.groupby(grouped_key).diff().clip(lower=0.0)


def _negative_group_diff(series: Series, grouped_key: Series) -> Series:
    return (-series.groupby(grouped_key).diff()).clip(lower=0.0)


def _positive_diff_score(series: Series, grouped_key: Series, scale: float) -> Series:
    return _clip01_series(_positive_group_diff(series, grouped_key) / scale)


def _negative_diff_score(series: Series, grouped_key: Series, scale: float) -> Series:
    return _clip01_series(_negative_group_diff(series, grouped_key) / scale)


def _positive_pct_drop(series: Series, grouped_key: Series) -> Series:
    prior = series.groupby(grouped_key).shift(1)
    return _clip01_series((prior - series) / prior.abs().replace(0.0, np.nan))


def _rolling_mean(series: Series, grouped_key: Series, window: int, min_periods: int) -> Series:
    return series.groupby(grouped_key).transform(lambda value: value.rolling(window, min_periods=min_periods).mean())


def _rolling_sum(series: Series, grouped_key: Series, window: int, min_periods: int) -> Series:
    return series.groupby(grouped_key).transform(lambda value: value.rolling(window, min_periods=min_periods).sum())


def _group_shift(series: Series, grouped_key: Series, periods: int) -> Series:
    return series.groupby(grouped_key).shift(periods)


def _consecutive_boolean_duration(mask: Series, grouped_key: Series) -> Series:
    output = Series(0.0, index=mask.index, dtype="float64")
    prepared = mask.fillna(False).astype(bool)
    for _, index in prepared.groupby(grouped_key).groups.items():
        count = 0.0
        for row_index in index:
            if bool(prepared.loc[row_index]):
                count += 1.0
            else:
                count = 0.0
            output.loc[row_index] = count
    return output


def _consecutive_signed_duration(sign: Series, grouped_key: Series) -> Series:
    output = Series(0.0, index=sign.index, dtype="float64")
    prepared = pd.to_numeric(sign, errors="coerce").fillna(0.0)
    for _, index in prepared.groupby(grouped_key).groups.items():
        count = 0.0
        previous = 0.0
        for row_index in index:
            value = float(prepared.loc[row_index])
            if value != 0.0 and value == previous:
                count += 1.0
            elif value != 0.0:
                count = 1.0
            else:
                count = 0.0
            previous = value
            output.loc[row_index] = count
    return output


def _clip01_series(series: Series) -> Series:
    return pd.to_numeric(series, errors="coerce").clip(lower=0.0, upper=1.0).fillna(0.0)


def _row_mean(*series: Series) -> Series:
    return pd.concat(series, axis=1).mean(axis=1, skipna=True).fillna(0.0)


def _json_wall_features(group: DataFrame, side: str, prefix: str) -> dict[str, float]:
    column = f"{side}_wall_candidates_json"
    counts = []
    scores = []
    distances = []
    notionals = []
    for value in group[column].dropna():
        records = _json_records(value)
        counts.append(len(records))
        for record in records:
            scores.append(_float(record.get("score")))
            distances.append(_float(record.get("distance_bps")))
            notionals.append(_float(record.get("notional")))
    return {
        f"{prefix}_{side}_wall_candidate_count_mean": float(np.mean(counts)) if counts else 0.0,
        f"{prefix}_{side}_wall_candidate_count_max": float(np.max(counts)) if counts else 0.0,
        f"{prefix}_{side}_wall_candidate_score_max": float(np.nanmax(scores)) if scores else math.nan,
        f"{prefix}_{side}_wall_candidate_score_mean": float(np.nanmean(scores)) if scores else math.nan,
        f"{prefix}_{side}_wall_candidate_distance_min": float(np.nanmin(distances)) if distances else math.nan,
        f"{prefix}_{side}_wall_candidate_notional_max": float(np.nanmax(notionals)) if notionals else math.nan,
    }


def _json_liquidity_zone_features(group: DataFrame, side: str, prefix: str) -> dict[str, float]:
    column = f"{side}_liquidity_zones_json"
    by_kind: dict[str, dict[str, list[float]]] = {
        "high": {"scores": [], "distances": [], "notionals": [], "counts": []},
        "low": {"scores": [], "distances": [], "notionals": [], "counts": []},
    }
    for value in group[column].dropna():
        records = _json_records(value)
        counts = {"high": 0, "low": 0}
        for record in records:
            kind = str(record.get("kind") or "").lower()
            if kind not in by_kind:
                continue
            counts[kind] += 1
            by_kind[kind]["scores"].append(_float(record.get("score")))
            by_kind[kind]["distances"].append(_float(record.get("distance_bps")))
            by_kind[kind]["notionals"].append(_float(record.get("notional")))
        for kind in counts:
            by_kind[kind]["counts"].append(float(counts[kind]))
    output: dict[str, float] = {}
    for kind, values in by_kind.items():
        output[f"{prefix}_{side}_liquidity_zone_{kind}_count_mean"] = float(np.mean(values["counts"])) if values["counts"] else 0.0
        output[f"{prefix}_{side}_liquidity_zone_{kind}_score_max"] = float(np.nanmax(values["scores"])) if values["scores"] else math.nan
        output[f"{prefix}_{side}_liquidity_zone_{kind}_distance_min"] = float(np.nanmin(values["distances"])) if values["distances"] else math.nan
        output[f"{prefix}_{side}_liquidity_zone_{kind}_notional_max"] = float(np.nanmax(values["notionals"])) if values["notionals"] else math.nan
    return output


def _json_records(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    try:
        loaded = json.loads(str(value))
    except Exception:
        return []
    if not isinstance(loaded, list):
        return []
    return [item for item in loaded if isinstance(item, dict)]


def _sign_flip_count(values: np.ndarray) -> int:
    nonzero = [int(value) for value in values if value != 0 and not np.isnan(value)]
    if len(nonzero) < 2:
        return 0
    return sum(1 for left, right in zip(nonzero, nonzero[1:]) if left != right)


def _condition_ratio(series: Series, predicate: Any) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return math.nan
    return float(predicate(numeric).mean())


def _slope(series: Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 2:
        return math.nan
    x = np.linspace(0.0, 1.0, len(numeric))
    try:
        return float(np.polyfit(x, numeric.to_numpy(dtype=float), 1)[0])
    except Exception:
        return math.nan


def _sign_agreement(values: list[float]) -> float:
    valid = np.array([value for value in values if not math.isnan(value)])
    if len(valid) == 0:
        return math.nan
    signs = np.sign(valid)
    signs = signs[signs != 0]
    if len(signs) == 0:
        return 0.0
    return float(abs(signs.sum()) / len(signs))


def _mean(series: Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else math.nan


def _sum(series: Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.sum()) if not numeric.empty else 0.0


def _column_mean(frame: DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else None


def _column_min(frame: DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(numeric.min()) if not numeric.empty else None


def _column_max(frame: DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(numeric.max()) if not numeric.empty else None


def _max(series: Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.max()) if not numeric.empty else math.nan


def _ratio(numerator: float, denominator: float) -> float:
    if denominator is None or denominator == 0 or math.isnan(float(denominator)):
        return math.nan
    return float(numerator) / float(denominator)


def _clip01(value: float) -> float:
    if value is None or math.isnan(float(value)):
        return math.nan
    return float(max(0.0, min(1.0, value)))


def _float(value: Any) -> float:
    try:
        if value is None:
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def _sqlite_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return _iso(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _parse_ts(value: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Could not parse timestamp: {value}")
    return pd.Timestamp(ts)


def _iso(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat()


def _normalise_pair(pair: str) -> str:
    text = str(pair or "").strip().upper()
    if text == "ALL":
        return "all"
    return text.split(":", 1)[0]


def _safe_column_part(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compact raw orderbook ticks into 1h numeric feature rows.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--pairs", default=",".join(DEFAULT_PAIRS), help="Comma-separated canonical pairs, or 'all'.")
    parser.add_argument("--mode", choices=("dry-run", "update", "full-rebuild", "validate"), default="dry-run")
    parser.add_argument("--overlap-days", type=int, default=3)
    parser.add_argument("--export-parquet", action="store_true")
    parser.add_argument("--export-csv", action="store_true")
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--min-coverage-ratio", type=float, default=DEFAULT_MIN_COVERAGE_RATIO)
    parser.add_argument("--ohlcv-path", type=Path, default=DEFAULT_OHLCV_PATH)
    args = parser.parse_args()

    pairs = tuple(part.strip() for part in args.pairs.replace(";", ",").split(",") if part.strip())
    if args.mode == "validate":
        summary = validate_orderbook_features(args.db, pairs=pairs or DEFAULT_PAIRS)
        json.dump(summary.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    summary = compact_orderbook_features(
        args.db,
        pairs=pairs or DEFAULT_PAIRS,
        mode=args.mode,
        overlap_days=args.overlap_days,
        export_parquet=args.export_parquet,
        export_csv=args.export_csv,
        export_dir=args.export_dir,
        min_coverage_ratio=args.min_coverage_ratio,
        ohlcv_path=args.ohlcv_path,
    )
    json.dump(summary.to_dict(), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_DIR = ROOT / "user_data" / "Custom_Launcher"
if str(LAUNCHER_DIR) not in sys.path:
    sys.path.insert(0, str(LAUNCHER_DIR))

from launcher_v2.services.orderbook_service import OrderBookService
from orderbook.market_context import _bybit_period
from orderbook.markets import (
    DEFAULT_MARKET_PROFILE_KEYS,
    MARKET_PROFILES,
    normalize_market_profile_keys,
    normalize_pairs_for_profiles,
)
from orderbook.store import connect_db, init_db, insert_metric_tick, insert_market_context, update_stream_status
from orderbook.streams import apply_book_update, build_stream_url, build_subscribe_message, parse_stream_message


def test_market_profiles_normalize_same_pair_across_all_markets() -> None:
    records = normalize_pairs_for_profiles(
        ["BTC/USDT:USDT"],
        ["binance_spot", "binance_usdm_futures", "bybit_spot", "bybit_linear"],
        max_symbols=12,
        depth_levels=20,
        stream_update_ms=500,
    )

    assert [record["market_key"] for record in records] == [
        "binance_spot",
        "binance_usdm_futures",
        "bybit_spot",
        "bybit_linear",
    ]
    assert {record["symbol"] for record in records} == {"BTCUSDT"}
    assert len({record["stream_id"] for record in records}) == 4
    assert {record["canonical_pair"] for record in records} == {"BTC/USDT"}
    assert normalize_market_profile_keys([]) == DEFAULT_MARKET_PROFILE_KEYS
    assert normalize_market_profile_keys(["not_a_market"]) == []
    assert normalize_pairs_for_profiles(["BTC/USDT"], [], max_symbols=1) == []


def test_stream_url_and_subscription_generation_for_profiles() -> None:
    binance_records = normalize_pairs_for_profiles(
        ["BTC/USDT"],
        ["binance_spot"],
        max_symbols=1,
        depth_levels=20,
        stream_update_ms=100,
    )
    binance_url = build_stream_url(MARKET_PROFILES["binance_spot"], binance_records)
    assert binance_url == "wss://stream.binance.com:9443/stream?streams=btcusdt@depth20@100ms"

    bybit_records = normalize_pairs_for_profiles(
        ["BTC/USDT"],
        ["bybit_linear"],
        max_symbols=1,
        depth_levels=20,
        stream_update_ms=500,
    )
    assert bybit_records[0]["stream_depth"] == 50
    assert bybit_records[0]["stream_update_ms"] == 20
    subscribe = json.loads(build_subscribe_message(MARKET_PROFILES["bybit_linear"], bybit_records) or "{}")
    assert subscribe == {"op": "subscribe", "args": ["orderbook.50.BTCUSDT"]}


def test_stream_parsers_and_bybit_delta_application() -> None:
    symbol, bids, asks, update_type = parse_stream_message(
        MARKET_PROFILES["binance_usdm_futures"],
        json.dumps(
            {
                "stream": "btcusdt@depth20@500ms",
                "data": {"s": "BTCUSDT", "b": [["100.0", "1.0"]], "a": [["101.0", "2.0"]]},
            }
        ),
    )
    assert (symbol, bids, asks, update_type) == ("BTCUSDT", [(100.0, 1.0)], [(101.0, 2.0)], "snapshot")

    state = {"bids": [], "asks": []}
    symbol, bids, asks, update_type = parse_stream_message(
        MARKET_PROFILES["bybit_linear"],
        json.dumps(
            {
                "topic": "orderbook.50.BTCUSDT",
                "type": "snapshot",
                "data": {"s": "BTCUSDT", "b": [["100.0", "1.0"], ["99.0", "2.0"]], "a": [["101.0", "1.0"]]},
            }
        ),
    )
    assert symbol == "BTCUSDT"
    apply_book_update(state, "bybit_linear", bids, asks, update_type, 50)

    _symbol, bids, asks, update_type = parse_stream_message(
        MARKET_PROFILES["bybit_linear"],
        json.dumps(
            {
                "topic": "orderbook.50.BTCUSDT",
                "type": "delta",
                "data": {"s": "BTCUSDT", "b": [["100.0", "0"], ["98.0", "3.0"]], "a": [["101.0", "5.0"]]},
            }
        ),
    )
    apply_book_update(state, "bybit_linear", bids, asks, update_type, 50)
    assert state["bids"] == [(99.0, 2.0), (98.0, 3.0)]
    assert state["asks"] == [(101.0, 5.0)]


def test_sqlite_keeps_duplicate_symbols_separate_by_stream_id(tmp_path: Path) -> None:
    db_path = tmp_path / "orderbook.sqlite"
    init_db(db_path)
    records = normalize_pairs_for_profiles(
        ["BTC/USDT"],
        ["binance_spot", "binance_usdm_futures"],
        max_symbols=1,
    )
    with connect_db(db_path) as conn:
        for record in records:
            update_stream_status(conn, _status_payload(record))
        insert_market_context(
            conn,
            {
                "ts": "2026-04-29T00:00:00+00:00",
                "market_key": "binance_usdm_futures",
                "venue": "binance",
                "market_type": "futures",
                "margin_type": "usdm",
                "quote_asset": "USDT",
                "canonical_pair": "BTC/USDT",
                "symbol": "BTCUSDT",
                "funding_rate": 0.0001,
            },
        )
        conn.commit()
        rows = conn.execute("SELECT stream_id, market_key, symbol FROM stream_status ORDER BY market_key").fetchall()
        context_count = conn.execute("SELECT COUNT(*) FROM market_context_ticks").fetchone()[0]

    assert [tuple(row) for row in rows] == [
        ("binance_spot:BTCUSDT", "binance_spot", "BTCUSDT"),
        ("binance_usdm_futures:BTCUSDT", "binance_usdm_futures", "BTCUSDT"),
    ]
    assert context_count == 1


def test_service_generates_multi_profile_config_and_single_pair_arg() -> None:
    service = OrderBookService(LAUNCHER_DIR, sys.executable)
    state = {
        "market_profiles": ["binance_spot", "binance_usdm_futures", "bybit_spot", "bybit_linear"],
        "depth_levels": "20",
        "stream_update_ms": "500",
        "metric_interval_seconds": "1",
        "context_poll_seconds": "300",
        "snapshot_interval_seconds": "60",
        "max_symbols": "12",
        "capacity_warning_mb": "500",
        "capacity_critical_mb": "2000",
    }

    runtime = service.build_runtime_config(state)
    valid, preview = service.normalized_pairs(["BTC/USDT"], state)
    command = service.build_collector_command(state, ["BTC/USDT"])

    assert runtime["market_profiles"] == ["binance_spot", "binance_usdm_futures", "bybit_spot", "bybit_linear"]
    assert len(valid) == 4
    assert len(preview) == 4
    assert command[-2:] == ["--pairs", "BTC/USDT"]


def test_bybit_context_period_mapping_keeps_shared_binance_style_default() -> None:
    assert _bybit_period("5m") == "5min"
    assert _bybit_period("15m") == "15min"
    assert _bybit_period("1h") == "1h"


def test_comparison_rows_calculate_spot_perp_basis(tmp_path: Path) -> None:
    service = OrderBookService(LAUNCHER_DIR, sys.executable)
    state = {"data_dir": str(tmp_path)}
    db_path = service.paths(state)["db"]
    init_db(db_path)
    with connect_db(db_path) as conn:
        insert_metric_tick(conn, _metric_payload("binance_spot", "spot", 100.0, 101.0, 0.20, 4000.0))
        insert_metric_tick(conn, _metric_payload("binance_usdm_futures", "futures", 101.0, 102.0, 0.35, 8000.0))
        conn.commit()

    rows = service.comparison_rows(state)

    assert len(rows) == 1
    assert rows[0][0:3] == ("BTC/USDT", "binance_spot", "binance_usdm_futures")
    assert float(rows[0][5]) > 0
    assert float(rows[0][7]) == 2.0
    assert rows[0][10] == "futures_bid_leads"


def test_init_db_migrates_old_schema_before_new_indexes(tmp_path: Path) -> None:
    db_path = tmp_path / "old.sqlite"
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(
            """
            CREATE TABLE orderbook_metric_ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                exchange TEXT NOT NULL,
                market_type TEXT NOT NULL,
                pair TEXT NOT NULL,
                symbol TEXT NOT NULL,
                book_valid INTEGER NOT NULL,
                depth_levels INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE orderbook_metric_bars (
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
                created_at TEXT NOT NULL
            );
            CREATE TABLE orderbook_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                exchange TEXT NOT NULL,
                market_type TEXT NOT NULL,
                pair TEXT NOT NULL,
                symbol TEXT NOT NULL,
                depth_levels INTEGER NOT NULL,
                bids_json TEXT NOT NULL,
                asks_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE stream_status (
                stream_id TEXT PRIMARY KEY,
                exchange TEXT NOT NULL,
                market_type TEXT NOT NULL,
                pair TEXT NOT NULL,
                symbol TEXT NOT NULL,
                depth_levels INTEGER NOT NULL,
                stream_mode TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )

    init_db(db_path)

    with sqlite3.connect(str(db_path)) as conn:
        tick_columns = {row[1] for row in conn.execute("PRAGMA table_info(orderbook_metric_ticks)").fetchall()}
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(orderbook_metric_ticks)").fetchall()}
    assert {"stream_id", "market_key", "canonical_pair"}.issubset(tick_columns)
    assert "idx_orderbook_metric_ticks_stream_ts" in indexes


def _status_payload(record: dict[str, object]) -> dict[str, object]:
    return {
        "stream_id": record["stream_id"],
        "market_key": record["market_key"],
        "venue": record["venue"],
        "exchange": record["market_key"],
        "market_type": record["market_type"],
        "margin_type": record["margin_type"],
        "quote_asset": record["quote_asset"],
        "canonical_pair": record["canonical_pair"],
        "pair": record["pair"],
        "symbol": record["symbol"],
        "depth_levels": record["stream_depth"],
        "stream_mode": "partial_depth",
        "status": "running",
        "updated_at": "2026-04-29T00:00:00+00:00",
    }


def _metric_payload(market_key: str, market_type: str, best_bid: float, best_ask: float, imbalance: float, depth: float) -> dict[str, object]:
    mid_price = (best_bid + best_ask) / 2.0
    return {
        "ts": "2026-04-29T00:00:00+00:00",
        "stream_id": f"{market_key}:BTCUSDT",
        "market_key": market_key,
        "venue": market_key.split("_", 1)[0],
        "exchange": market_key,
        "market_type": market_type,
        "margin_type": "spot" if market_type == "spot" else "usdm",
        "quote_asset": "USDT",
        "canonical_pair": "BTC/USDT",
        "pair": "BTC/USDT",
        "symbol": "BTCUSDT",
        "book_valid": 1,
        "depth_levels": 20,
        "message_count_interval": 1,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid_price": mid_price,
        "spread_bps": (best_ask - best_bid) / mid_price * 10000.0,
        "bid_notional_top20": depth / 2.0,
        "ask_notional_top20": depth / 2.0,
        "imbalance_top20": imbalance,
        "strong_bid_pressure": 1 if imbalance >= 0.35 else 0,
        "nearest_bid_wall_distance_bps": 10.0,
    }

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from orderbook.market_context import fetch_market_context
from orderbook.markets import (
    MARKET_PROFILES,
    normalize_market_profile_keys,
    normalize_pairs_for_profiles,
)
from orderbook.metrics import (
    aggregate_metric_ticks,
    calculate_orderbook_metrics,
    estimate_storage_usage,
)
from orderbook.streams import (
    apply_book_update,
    build_stream_url,
    build_subscribe_message,
    parse_stream_message,
    stream_records_by_profile,
)
from orderbook.store import (
    connect_db,
    init_db,
    insert_capacity_alert,
    insert_market_context,
    insert_metric_bar,
    insert_metric_tick,
    insert_snapshot,
    update_stream_status,
    upsert_collector_run,
)

try:
    import websocket  # type: ignore
except Exception as exc:  # pragma: no cover
    websocket = None
    WEBSOCKET_IMPORT_ERROR = exc
else:
    WEBSOCKET_IMPORT_ERROR = None


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = THIS_DIR / "config" / "sources.json"
DEFAULT_DATA_DIR = THIS_DIR.parents[1] / "orderbook_data" / "live"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "orderbook_events.sqlite"
DEFAULT_STATUS_PATH = DEFAULT_DATA_DIR / "collector_status.json"
DEFAULT_PID_PATH = DEFAULT_DATA_DIR / "collector.pid"
DEFAULT_STOP_PATH = DEFAULT_DATA_DIR / "collector.stop"
DEFAULT_LOG_PATH = DEFAULT_DATA_DIR / "logs" / "orderbook_collector.log"

VALID_DEPTH_LEVELS = {depth for profile in MARKET_PROFILES.values() for depth in profile.supported_depths}
VALID_STREAM_UPDATE_MS = {interval for profile in MARKET_PROFILES.values() for interval in profile.supported_update_ms}


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return loaded if isinstance(loaded, type(default)) else default


def configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )


def ensure_runtime_dirs(data_dir: Path) -> None:
    for folder in [data_dir, data_dir / "logs", data_dir / "exports", data_dir / "analysis", data_dir / "raw"]:
        folder.mkdir(parents=True, exist_ok=True)


def estimate_directory_size_mb(path: Path) -> float:
    if not path.exists() or not path.is_dir():
        return 0.0
    total = 0
    for root, _, files in os.walk(path):
        root_path = Path(root)
        for name in files:
            candidate = root_path / name
            try:
                total += candidate.stat().st_size
            except OSError:
                continue
    return round(total / (1024.0 * 1024.0), 2)


def read_config(path: Path) -> dict[str, Any]:
    payload = load_json(path, {})
    if not isinstance(payload, dict):
        payload = {}
    defaults = load_json(DEFAULT_CONFIG_PATH, {})
    if isinstance(defaults, dict):
        for key, value in defaults.items():
            payload.setdefault(key, value)
    payload.setdefault("market_profiles", ["binance_spot", "binance_usdm_futures", "bybit_spot", "bybit_linear"])
    payload.setdefault("exchange", "multi_market")
    payload.setdefault("market_type", "multi")
    payload.setdefault("stream_mode", "partial_depth")
    payload.setdefault("depth_levels", 20)
    payload.setdefault("stream_update_ms", 500)
    payload.setdefault("metric_interval_seconds", 1)
    payload.setdefault("context_poll_seconds", 300)
    payload.setdefault("context_period", "5m")
    payload.setdefault("bar_intervals_seconds", [60, 300, 3600])
    payload.setdefault("snapshot_interval_seconds", 60)
    payload.setdefault("store_snapshots", True)
    payload.setdefault("store_raw_events", False)
    payload.setdefault("max_symbols", 12)
    payload.setdefault("capacity_warning_mb", 500)
    payload.setdefault("capacity_critical_mb", 2000)
    payload.setdefault("wall_score_threshold", 4.0)
    payload.setdefault("pressure_threshold", 0.35)
    payload.setdefault("extreme_pressure_threshold", 0.60)
    payload.setdefault("liquidity_bps_bands", [5, 10, 25, 50])
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone multi-market order book collector")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--status-file", type=Path, default=DEFAULT_STATUS_PATH)
    parser.add_argument("--pid-file", type=Path, default=DEFAULT_PID_PATH)
    parser.add_argument("--stop-file", type=Path, default=DEFAULT_STOP_PATH)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--pairs", type=str, default="")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def parse_pair_tokens(raw_pairs: str) -> list[str]:
    tokens = [part.strip() for part in str(raw_pairs or "").replace("\n", ",").split(",") if part.strip()]
    return tokens


def main() -> int:
    if websocket is None:
        print("Missing dependency websocket-client. Install with: pip install websocket-client")
        if WEBSOCKET_IMPORT_ERROR is not None:
            print(f"Import error: {WEBSOCKET_IMPORT_ERROR}")
        return 2

    args = parse_args()
    ensure_runtime_dirs(args.data_dir)
    configure_logging(args.log_file)
    init_db(args.db)
    config = read_config(args.config)

    depth_levels = int(config.get("depth_levels", 20))
    stream_update_ms = int(config.get("stream_update_ms", 500))
    if depth_levels not in VALID_DEPTH_LEVELS:
        raise ValueError(f"Unsupported depth_levels={depth_levels}. Supported: {sorted(VALID_DEPTH_LEVELS)}")
    if stream_update_ms not in VALID_STREAM_UPDATE_MS:
        raise ValueError(f"Unsupported stream_update_ms={stream_update_ms}. Supported: {sorted(VALID_STREAM_UPDATE_MS)}")

    profile_keys = normalize_market_profile_keys(config.get("market_profiles"))
    if not profile_keys:
        status_payload = {"status": "error", "last_error": "No supported market profiles configured."}
        save_json(args.status_file, status_payload)
        return 1
    pair_records = normalize_pairs_for_profiles(
        parse_pair_tokens(args.pairs),
        profile_keys,
        max_symbols=int(config.get("max_symbols", 12)),
        depth_levels=depth_levels,
        stream_update_ms=stream_update_ms,
    )
    if not pair_records:
        status_payload = {"status": "error", "last_error": "No usable whitelist pairs found. Add pairs on the Pairs tab first."}
        save_json(args.status_file, status_payload)
        return 1

    stream_state: dict[str, dict[str, Any]] = {}
    for record in pair_records:
        stream_id = record["stream_id"]
        retained_depth = min(depth_levels, int(record["stream_depth"]))
        stream_state[stream_id] = {
            "pair": record["pair"],
            "canonical_pair": record["canonical_pair"],
            "symbol": record["symbol"],
            "stream_id": stream_id,
            "market_key": record["market_key"],
            "venue": record["venue"],
            "market_type": record["market_type"],
            "margin_type": record["margin_type"],
            "quote_asset": record["quote_asset"],
            "stream_depth": record["stream_depth"],
            "stream_update_ms": record["stream_update_ms"],
            "retained_depth": retained_depth,
            "bids": [],
            "asks": [],
            "last_message_at": None,
            "last_metric_at": None,
            "message_count": 0,
            "message_count_interval": 0,
            "metric_count": 0,
            "snapshot_count": 0,
            "reconnect_count": 0,
            "error_count": 0,
            "last_error": None,
            "status": "running",
            "started_at": utc_now(),
            "recent_ticks": [],
        }

    run_id = hashlib.sha256(f"{os.getpid()}|{utc_now()}|orderbook".encode("utf-8")).hexdigest()
    args.pid_file.parent.mkdir(parents=True, exist_ok=True)
    args.pid_file.write_text(str(os.getpid()), encoding="utf-8")
    try:
        args.stop_file.unlink(missing_ok=True)
    except Exception:
        pass

    conn = connect_db(args.db)
    try:
        upsert_collector_run(conn, run_id, utc_now(), "running", os.getpid(), str(args.config), str(args.db))
        conn.commit()
    finally:
        conn.close()

    metric_interval = max(1, int(config.get("metric_interval_seconds", 1)))
    snapshot_interval = max(1, int(config.get("snapshot_interval_seconds", 60)))
    context_poll_seconds = max(30, int(config.get("context_poll_seconds", 300)))
    context_period = str(config.get("context_period") or "5m")
    bar_intervals = [int(v) for v in config.get("bar_intervals_seconds", [60, 300, 3600]) if int(v) > 0]
    store_snapshots = bool(config.get("store_snapshots", True))
    symbols = [item["symbol"] for item in pair_records]
    pairs = sorted({item["pair"] for item in pair_records})
    canonical_pairs = sorted({item["canonical_pair"] for item in pair_records})
    market_profile_labels = [MARKET_PROFILES[key].label for key in profile_keys]
    max_retained_depth = max(int(state["retained_depth"]) for state in stream_state.values())

    status_payload: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "pid": os.getpid(),
        "started_at": utc_now(),
        "heartbeat_at": utc_now(),
        "last_message_at": None,
        "last_metric_at": None,
        "exchange": "multi_market",
        "market_type": "multi",
        "market_profiles": profile_keys,
        "market_profile_labels": market_profile_labels,
        "stream_mode": "partial_depth",
        "depth_levels": depth_levels,
        "stream_update_ms": stream_update_ms,
        "metric_interval_seconds": metric_interval,
        "snapshot_interval_seconds": snapshot_interval,
        "context_poll_seconds": context_poll_seconds,
        "context_period": context_period,
        "pair_count": len(pairs),
        "pairs": pairs,
        "canonical_pairs": canonical_pairs,
        "symbols": symbols,
        "active_streams": len(pair_records),
        "message_count_total": 0,
        "metric_count_total": 0,
        "snapshot_count_total": 0,
        "context_count_total": 0,
        "reconnect_count_total": 0,
        "last_error": None,
        "db_path": str(args.db),
        "config_path": str(args.config),
        "data_dir": str(args.data_dir),
        "data_dir_mb": 0.0,
        "db_mb": 0.0,
        "estimated_mb_per_day": 0.0,
        "estimated_metric_rows_per_day": 0.0,
        "estimated_snapshot_rows_per_day": 0.0,
        "capacity_level": "ok",
    }
    save_json(args.status_file, status_payload)
    conn = connect_db(args.db)
    try:
        for stream_id in sorted(stream_state):
            state = stream_state[stream_id]
            update_stream_status(
                conn,
                {
                    "stream_id": stream_id,
                    "market_key": state["market_key"],
                    "venue": state["venue"],
                    "exchange": state["market_key"],
                    "market_type": state["market_type"],
                    "margin_type": state["margin_type"],
                    "quote_asset": state["quote_asset"],
                    "canonical_pair": state["canonical_pair"],
                    "pair": state["pair"],
                    "symbol": state["symbol"],
                    "depth_levels": state["retained_depth"],
                    "stream_mode": "partial_depth",
                    "status": "running",
                    "started_at": state["started_at"],
                    "last_message_at": None,
                    "last_metric_at": None,
                    "message_count": 0,
                    "metric_count": 0,
                    "snapshot_count": 0,
                    "reconnect_count": 0,
                    "error_count": 0,
                    "last_error": None,
                    "best_bid": None,
                    "best_ask": None,
                    "mid_price": None,
                    "spread_bps": None,
                    "imbalance_top20": None,
                    "bid_pressure_ratio_60s": None,
                    "ask_pressure_ratio_60s": None,
                    "nearest_bid_wall_distance_bps": None,
                    "nearest_ask_wall_distance_bps": None,
                    "updated_at": utc_now(),
                },
            )
        conn.commit()
    finally:
        conn.close()

    stop_event = threading.Event()
    books_lock = threading.Lock()
    ws_holders: dict[str, Any] = {}
    grouped_records = stream_records_by_profile(pair_records)

    def ws_runner(profile_key: str, records: list[dict[str, Any]]) -> None:
        profile = MARKET_PROFILES[profile_key]
        reconnect_delay = 5
        stream_by_symbol = {str(record["symbol"]).upper(): str(record["stream_id"]) for record in records}
        stream_url = build_stream_url(profile, records)

        def on_message(_ws: Any, message: str) -> None:
            nonlocal reconnect_delay
            reconnect_delay = 5
            try:
                symbol, bids, asks, update_type = parse_stream_message(profile, message)
                if not symbol:
                    return
                stream_id = stream_by_symbol.get(symbol.upper())
                if not stream_id:
                    return
                now = utc_now()
                with books_lock:
                    state = stream_state[stream_id]
                    apply_book_update(state, profile_key, bids, asks, update_type, int(state["retained_depth"]))
                    state["last_message_at"] = now
                    state["message_count"] += 1
                    state["message_count_interval"] += 1
                    state["status"] = "running"
                    status_payload["last_message_at"] = now
                    status_payload["message_count_total"] = int(status_payload.get("message_count_total", 0)) + 1
            except Exception as exc:
                logging.exception("Could not parse %s websocket message", profile_key)
                status_payload["last_error"] = str(exc)

        def on_error(_ws: Any, error: Any) -> None:
            text = str(error)
            logging.error("%s websocket error: %s", profile_key, text)
            status_payload["last_error"] = text
            with books_lock:
                for record in records:
                    state = stream_state[str(record["stream_id"])]
                    state["error_count"] += 1
                    state["last_error"] = text
                    state["status"] = "error"

        def on_close(_ws: Any, _status_code: Any, _msg: Any) -> None:
            logging.warning("%s websocket closed", profile_key)

        def on_open(ws: Any) -> None:
            logging.info("%s websocket connected", profile_key)
            subscribe = build_subscribe_message(profile, records)
            if subscribe:
                ws.send(subscribe)

        while not stop_event.is_set():
            try:
                ws_app = websocket.WebSocketApp(stream_url, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
                ws_holders[profile_key] = ws_app
                ws_app.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:
                logging.exception("%s websocket runner failure", profile_key)
                status_payload["last_error"] = str(exc)
            if stop_event.is_set():
                break
            with books_lock:
                for record in records:
                    state = stream_state[str(record["stream_id"])]
                    state["reconnect_count"] += 1
            status_payload["reconnect_count_total"] = int(status_payload.get("reconnect_count_total", 0)) + 1
            time.sleep(reconnect_delay)
            reconnect_delay = min(30, reconnect_delay * 2)

    ws_threads = [
        threading.Thread(target=ws_runner, args=(profile_key, records), daemon=True)
        for profile_key, records in grouped_records.items()
    ]
    for ws_thread in ws_threads:
        ws_thread.start()

    last_metric_ts = 0.0
    last_snapshot_ts = 0.0
    last_context_ts = 0.0
    bar_last_run = {interval: 0.0 for interval in bar_intervals}
    last_capacity_ts = 0.0
    terminal_status = "stopped"
    once_deadline = time.time() + 30.0 if args.once else None
    try:
        while True:
            if args.stop_file.exists():
                terminal_status = "stopped"
                break
            now_ts = time.time()
            now_iso = utc_now()

            if now_ts - last_metric_ts >= metric_interval:
                conn = connect_db(args.db)
                try:
                    inserted_any_metric = False
                    with books_lock:
                        for stream_id, state in stream_state.items():
                            if not state["bids"] or not state["asks"]:
                                continue
                            metric_config = dict(config)
                            metric_config["depth_levels"] = state["retained_depth"]
                            metrics = calculate_orderbook_metrics(
                                state["pair"],
                                state["symbol"],
                                state["bids"],
                                state["asks"],
                                metric_config,
                                state["message_count_interval"],
                            )
                            metrics["ts"] = now_iso
                            metrics["stream_id"] = stream_id
                            metrics["market_key"] = state["market_key"]
                            metrics["venue"] = state["venue"]
                            metrics["exchange"] = state["market_key"]
                            metrics["market_type"] = state["market_type"]
                            metrics["margin_type"] = state["margin_type"]
                            metrics["quote_asset"] = state["quote_asset"]
                            metrics["canonical_pair"] = state["canonical_pair"]
                            insert_metric_tick(conn, metrics)
                            inserted_any_metric = True
                            state["last_metric_at"] = now_iso
                            state["metric_count"] += 1
                            status_payload["metric_count_total"] = int(status_payload.get("metric_count_total", 0)) + 1
                            state["message_count_interval"] = 0
                            reduced = {
                                "ts": now_iso,
                                "book_valid": metrics.get("book_valid"),
                                "spread_bps": metrics.get("spread_bps"),
                                "microprice_offset_bps": metrics.get("microprice_offset_bps"),
                                "imbalance_top20": metrics.get("imbalance_top20"),
                                "imbalance_10bps": metrics.get("imbalance_10bps"),
                                "imbalance_25bps": metrics.get("imbalance_25bps"),
                                "strong_bid_pressure": metrics.get("strong_bid_pressure"),
                                "strong_ask_pressure": metrics.get("strong_ask_pressure"),
                                "nearest_bid_wall_distance_bps": metrics.get("nearest_bid_wall_distance_bps"),
                                "nearest_ask_wall_distance_bps": metrics.get("nearest_ask_wall_distance_bps"),
                                "strongest_bid_wall_score_50bps": metrics.get("strongest_bid_wall_score_50bps"),
                                "strongest_ask_wall_score_50bps": metrics.get("strongest_ask_wall_score_50bps"),
                            }
                            state["recent_ticks"].append(reduced)
                            max_keep = max(bar_intervals) if bar_intervals else 3600
                            cutoff = now_ts - max_keep - 5
                            state["recent_ticks"] = [item for item in state["recent_ticks"] if _iso_to_ts(item["ts"]) >= cutoff]
                            ticks_60 = [
                                item
                                for item in state["recent_ticks"]
                                if _iso_to_ts(str(item.get("ts") or "")) >= (now_ts - 60.0) and int(item.get("book_valid", 0)) == 1
                            ]
                            bid_ratio_60s = (
                                sum(1 for item in ticks_60 if int(item.get("strong_bid_pressure", 0)) == 1) / len(ticks_60)
                                if ticks_60
                                else None
                            )
                            ask_ratio_60s = (
                                sum(1 for item in ticks_60 if int(item.get("strong_ask_pressure", 0)) == 1) / len(ticks_60)
                                if ticks_60
                                else None
                            )
                            update_stream_status(
                                conn,
                                {
                                    "stream_id": stream_id,
                                    "market_key": state["market_key"],
                                    "venue": state["venue"],
                                    "exchange": state["market_key"],
                                    "market_type": state["market_type"],
                                    "margin_type": state["margin_type"],
                                    "quote_asset": state["quote_asset"],
                                    "canonical_pair": state["canonical_pair"],
                                    "pair": state["pair"],
                                    "symbol": state["symbol"],
                                    "depth_levels": state["retained_depth"],
                                    "stream_mode": "partial_depth",
                                    "status": state["status"],
                                    "started_at": state["started_at"],
                                    "last_message_at": state["last_message_at"],
                                    "last_metric_at": state["last_metric_at"],
                                    "message_count": state["message_count"],
                                    "metric_count": state["metric_count"],
                                    "snapshot_count": state["snapshot_count"],
                                    "reconnect_count": state["reconnect_count"],
                                    "error_count": state["error_count"],
                                    "last_error": state["last_error"],
                                    "best_bid": metrics.get("best_bid"),
                                    "best_ask": metrics.get("best_ask"),
                                    "mid_price": metrics.get("mid_price"),
                                    "spread_bps": metrics.get("spread_bps"),
                                    "imbalance_top20": metrics.get("imbalance_top20"),
                                    "bid_pressure_ratio_60s": bid_ratio_60s,
                                    "ask_pressure_ratio_60s": ask_ratio_60s,
                                    "nearest_bid_wall_distance_bps": metrics.get("nearest_bid_wall_distance_bps"),
                                    "nearest_ask_wall_distance_bps": metrics.get("nearest_ask_wall_distance_bps"),
                                    "updated_at": now_iso,
                                },
                            )
                        conn.commit()
                    status_payload["last_metric_at"] = now_iso
                    status_payload["heartbeat_at"] = now_iso
                    save_json(args.status_file, status_payload)
                finally:
                    conn.close()
                last_metric_ts = now_ts
                if args.once and inserted_any_metric:
                    terminal_status = "stopped"
                    break
                if args.once and once_deadline is not None and now_ts >= once_deadline:
                    if int(status_payload.get("message_count_total", 0)) <= 0:
                        raise RuntimeError("No websocket messages received within once timeout.")
                    if int(status_payload.get("metric_count_total", 0)) <= 0:
                        raise RuntimeError("No valid orderbook data received within once timeout.")

            if store_snapshots and now_ts - last_snapshot_ts >= snapshot_interval:
                conn = connect_db(args.db)
                try:
                    with books_lock:
                        for stream_id, state in stream_state.items():
                            if not state["bids"] or not state["asks"]:
                                continue
                            best_bid = state["bids"][0][0]
                            best_ask = state["asks"][0][0]
                            insert_snapshot(
                                conn,
                                {
                                    "ts": now_iso,
                                    "stream_id": stream_id,
                                    "market_key": state["market_key"],
                                    "venue": state["venue"],
                                    "exchange": state["market_key"],
                                    "market_type": state["market_type"],
                                    "margin_type": state["margin_type"],
                                    "quote_asset": state["quote_asset"],
                                    "canonical_pair": state["canonical_pair"],
                                    "pair": state["pair"],
                                    "symbol": state["symbol"],
                                    "depth_levels": state["retained_depth"],
                                    "best_bid": best_bid,
                                    "best_ask": best_ask,
                                    "mid_price": (best_bid + best_ask) / 2.0,
                                    "bids": state["bids"][: int(state["retained_depth"])],
                                    "asks": state["asks"][: int(state["retained_depth"])],
                                },
                            )
                            state["snapshot_count"] += 1
                            status_payload["snapshot_count_total"] = int(status_payload.get("snapshot_count_total", 0)) + 1
                        conn.commit()
                finally:
                    conn.close()
                last_snapshot_ts = now_ts

            for interval in bar_intervals:
                if now_ts - bar_last_run[interval] < interval:
                    continue
                conn = connect_db(args.db)
                try:
                    with books_lock:
                        for stream_id, state in stream_state.items():
                            cutoff = now_ts - interval - 1
                            ticks = [tick for tick in state["recent_ticks"] if _iso_to_ts(tick["ts"]) >= cutoff]
                            if not ticks:
                                continue
                            agg = aggregate_metric_ticks(ticks, interval, expected_samples=interval)
                            insert_metric_bar(
                                conn,
                                {
                                    "ts_start": _ts_to_iso(now_ts - interval),
                                    "ts_end": now_iso,
                                    "timeframe_seconds": interval,
                                    "stream_id": stream_id,
                                    "market_key": state["market_key"],
                                    "venue": state["venue"],
                                    "exchange": state["market_key"],
                                    "market_type": state["market_type"],
                                    "margin_type": state["margin_type"],
                                    "quote_asset": state["quote_asset"],
                                    "canonical_pair": state["canonical_pair"],
                                    "pair": state["pair"],
                                    "symbol": state["symbol"],
                                    **agg,
                                },
                            )
                        conn.commit()
                finally:
                    conn.close()
                bar_last_run[interval] = now_ts

            if now_ts - last_context_ts >= context_poll_seconds:
                context_rows: list[dict[str, Any]] = []
                for record in pair_records:
                    try:
                        row = fetch_market_context(record, period=context_period)
                    except Exception as exc:
                        logging.warning("Market context fetch failed for %s: %s", record.get("stream_id"), exc)
                        status_payload["last_error"] = str(exc)
                        continue
                    if row:
                        row["ts"] = now_iso
                        context_rows.append(row)
                if context_rows:
                    conn = connect_db(args.db)
                    try:
                        for row in context_rows:
                            insert_market_context(conn, row)
                        conn.commit()
                    finally:
                        conn.close()
                    status_payload["context_count_total"] = int(status_payload.get("context_count_total", 0)) + len(context_rows)
                    save_json(args.status_file, status_payload)
                last_context_ts = now_ts

            if now_ts - last_capacity_ts >= 30:
                data_dir_mb = estimate_directory_size_mb(args.data_dir)
                db_mb = round(args.db.stat().st_size / (1024.0 * 1024.0), 2) if args.db.exists() else 0.0
                estimate = estimate_storage_usage(
                    len(pair_records),
                    metric_interval,
                    snapshot_interval,
                    max_retained_depth,
                    store_snapshots=store_snapshots,
                )
                warning_mb = float(config.get("capacity_warning_mb", 500))
                critical_mb = float(config.get("capacity_critical_mb", 2000))
                capacity_level = "ok"
                if data_dir_mb >= critical_mb:
                    capacity_level = "critical"
                elif data_dir_mb >= warning_mb:
                    capacity_level = "warning"
                status_payload["data_dir_mb"] = data_dir_mb
                status_payload["db_mb"] = db_mb
                status_payload["estimated_mb_per_day"] = round(estimate["estimated_total_mb_per_day"], 2)
                status_payload["estimated_metric_rows_per_day"] = round(estimate["metric_rows_per_day"], 2)
                status_payload["estimated_snapshot_rows_per_day"] = round(estimate["snapshot_rows_per_day"], 2)
                status_payload["capacity_level"] = capacity_level
                if capacity_level in {"warning", "critical"}:
                    conn = connect_db(args.db)
                    try:
                        insert_capacity_alert(
                            conn,
                            level=capacity_level,
                            data_dir=str(args.data_dir),
                            db_path=str(args.db),
                            data_dir_mb=data_dir_mb,
                            db_mb=db_mb,
                            warning_threshold_mb=warning_mb,
                            critical_threshold_mb=critical_mb,
                            message=f"Capacity level {capacity_level}",
                        )
                        conn.commit()
                    finally:
                        conn.close()
                save_json(args.status_file, status_payload)
                last_capacity_ts = now_ts

            time.sleep(0.2)
    except Exception as exc:
        logging.exception("Orderbook collector crashed")
        terminal_status = "error"
        status_payload["status"] = "error"
        status_payload["last_error"] = str(exc)
        save_json(args.status_file, status_payload)
        return 1
    finally:
        stop_event.set()
        for ws_app in list(ws_holders.values()):
            if ws_app is not None:
                try:
                    ws_app.close()
                except Exception:
                    pass
        status_payload["status"] = terminal_status
        status_payload["pid"] = None
        status_payload["heartbeat_at"] = utc_now()
        save_json(args.status_file, status_payload)
        conn = connect_db(args.db)
        try:
            conn.execute(
                "UPDATE collector_runs SET status = ?, stopped_at = ?, notes = ? WHERE run_id = ?",
                (terminal_status, utc_now(), status_payload.get("last_error"), run_id),
            )
            conn.commit()
        finally:
            conn.close()
        try:
            args.pid_file.unlink(missing_ok=True)
        except Exception:
            pass
        if args.stop_file.exists():
            try:
                args.stop_file.unlink()
            except Exception:
                pass
    return 0


def _iso_to_ts(value: str) -> float:
    from datetime import datetime

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _ts_to_iso(value: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(value, timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())

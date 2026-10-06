from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import requests
import orjson

APP_DIR = Path(__file__).resolve().parents[2]
USER_DATA_DIR = APP_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from .metrics import aggregate_metric_ticks, calculate_orderbook_metrics  # noqa: E402
from .store import init_db, insert_metric_bar  # noqa: E402


DEFAULT_DB_PATH = USER_DATA_DIR / "collector_data" / "orderbook" / "orderbook_events.sqlite"
DEFAULT_RAW_DIR = Path("D:/FreqTradeStuffLargeData/orderbook_data/historical_bybit/spot_raw")
DEFAULT_BASE_URL = "https://quote-saver.bycsi.com/orderbook/spot"
DEFAULT_CONFIG = {
    "depth_levels": 20,
    "wall_score_threshold": 4.0,
    "pressure_threshold": 0.35,
    "extreme_pressure_threshold": 0.6,
    "liquidity_bps_bands": [5, 10, 25, 50],
}
INSERT_COLUMNS = (
    "ts",
    "stream_id",
    "market_key",
    "venue",
    "exchange",
    "market_type",
    "margin_type",
    "quote_asset",
    "canonical_pair",
    "pair",
    "symbol",
    "book_valid",
    "depth_levels",
    "message_count_interval",
    "best_bid",
    "best_ask",
    "mid_price",
    "spread_bps",
    "microprice",
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
    "nearest_bid_wall_price",
    "nearest_bid_wall_distance_bps",
    "nearest_bid_wall_notional",
    "nearest_bid_wall_score",
    "nearest_ask_wall_price",
    "nearest_ask_wall_distance_bps",
    "nearest_ask_wall_notional",
    "nearest_ask_wall_score",
    "strongest_bid_wall_price_50bps",
    "strongest_bid_wall_score_50bps",
    "strongest_ask_wall_price_50bps",
    "strongest_ask_wall_score_50bps",
    "bid_wall_candidates_json",
    "ask_wall_candidates_json",
    "bid_liquidity_zones_json",
    "ask_liquidity_zones_json",
    "strong_bid_pressure",
    "strong_ask_pressure",
    "extreme_bid_pressure",
    "extreme_ask_pressure",
    "created_at",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Import free historical Bybit OB200 data into ticks or live-compatible 1m bars.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--pair", default="BTC/USDT")
    parser.add_argument("--start-date", required=True, help="Inclusive UTC date, YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="Inclusive UTC date, YYYY-MM-DD.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--sample-seconds", type=int, default=1)
    parser.add_argument("--replace", action="store_true", help="Replace existing rows from this historical stream/date.")
    parser.add_argument("--summary-only", action="store_true", help="Write missing live-compatible 1m bars without storing metric ticks.")
    parser.add_argument("--gap-start", default="", help="Inclusive ISO-8601 UTC boundary for --summary-only.")
    parser.add_argument("--gap-end", default="", help="Exclusive ISO-8601 UTC boundary for --summary-only.")
    parser.add_argument("--stream-id", default="", help="Target stream ID. Summary mode defaults to bybit_spot:<symbol>.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.sample_seconds < 1:
        raise ValueError("--sample-seconds must be at least 1")
    if args.summary_only and 60 % args.sample_seconds != 0:
        raise ValueError("--sample-seconds must divide evenly into 60 with --summary-only")

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if end < start:
        raise ValueError("--end-date must be on or after --start-date")

    if args.summary_only:
        if args.replace:
            raise ValueError("--replace is not supported with --summary-only; summary imports are insert-missing only")
        if not args.gap_start or not args.gap_end:
            raise ValueError("--summary-only requires --gap-start and --gap-end")
        summary = import_summary_range(
            db_path=args.db_path,
            raw_dir=args.raw_dir,
            base_url=args.base_url.rstrip("/"),
            symbol=args.symbol.upper(),
            pair=args.pair,
            start=start,
            end=end,
            gap_start=_parse_utc(args.gap_start),
            gap_end=_parse_utc(args.gap_end),
            sample_seconds=args.sample_seconds,
            stream_id=args.stream_id or f"bybit_spot:{args.symbol.upper()}",
            dry_run=args.dry_run,
        )
    else:
        summary = import_range(
            db_path=args.db_path,
            raw_dir=args.raw_dir,
            base_url=args.base_url.rstrip("/"),
            symbol=args.symbol.upper(),
            pair=args.pair,
            start=start,
            end=end,
            sample_seconds=args.sample_seconds,
            replace=args.replace,
            dry_run=args.dry_run,
        )
    print(json.dumps(summary, indent=2))
    return 0


def import_range(
    *,
    db_path: Path,
    raw_dir: Path,
    base_url: str,
    symbol: str,
    pair: str,
    start: date,
    end: date,
    sample_seconds: int,
    replace: bool,
    dry_run: bool,
) -> dict[str, Any]:
    init_db(db_path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    days = list(_date_range(start, end))
    stream_id = _stream_id(symbol)
    summary: dict[str, Any] = {
        "db_path": str(db_path),
        "symbol": symbol,
        "pair": pair,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "sample_seconds": sample_seconds,
        "dry_run": dry_run,
        "days": len(days),
        "downloaded": 0,
        "already_present": 0,
        "rows_inserted": 0,
        "missing_days": [],
        "failed_days": [],
    }
    with sqlite3.connect(str(db_path), timeout=60.0) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        for day in days:
            url = _daily_url(base_url, symbol, day)
            zip_path = raw_dir / Path(url).name
            if not zip_path.exists():
                if dry_run:
                    summary["downloaded"] += 1
                    continue
                ok = _download(url, zip_path)
                if not ok:
                    summary["missing_days"].append(day.isoformat())
                    continue
                summary["downloaded"] += 1
            else:
                summary["already_present"] += 1
            if dry_run:
                continue
            try:
                rows = _import_day(
                    conn,
                    zip_path,
                    stream_id=stream_id,
                    symbol=symbol,
                    pair=pair,
                    day=day,
                    sample_seconds=sample_seconds,
                    replace=replace,
                )
            except Exception as exc:
                summary["failed_days"].append({"date": day.isoformat(), "error": str(exc)})
                continue
            summary["rows_inserted"] += rows
    return summary


def import_summary_range(
    *,
    db_path: Path,
    raw_dir: Path,
    base_url: str,
    symbol: str,
    pair: str,
    start: date,
    end: date,
    gap_start: datetime,
    gap_end: datetime,
    sample_seconds: int,
    stream_id: str,
    dry_run: bool,
) -> dict[str, Any]:
    if gap_end <= gap_start:
        raise ValueError("--gap-end must be after --gap-start")
    if gap_start.date() < start or gap_end.date() > end + timedelta(days=1):
        raise ValueError("Gap boundaries must fall within the requested archive date range")

    raw_dir.mkdir(parents=True, exist_ok=True)
    archives: list[tuple[date, Path]] = []
    missing_days: list[str] = []
    downloaded = 0
    for day in _date_range(start, end):
        url = _daily_url(base_url, symbol, day)
        zip_path = raw_dir / Path(url).name
        if not zip_path.exists():
            if dry_run or not _download(url, zip_path):
                missing_days.append(day.isoformat())
                continue
            downloaded += 1
        archives.append((day, zip_path))

    if missing_days:
        raise FileNotFoundError(f"Required archive days are unavailable: {', '.join(missing_days)}")

    bars = _build_summary_bars(
        archives,
        stream_id=stream_id,
        symbol=symbol,
        pair=pair,
        gap_start=gap_start,
        gap_end=gap_end,
        sample_seconds=sample_seconds,
    )
    inserted = 0
    skipped_existing = 0
    if not dry_run:
        init_db(db_path)
        with sqlite3.connect(str(db_path), timeout=60.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            existing = {
                row[0]
                for row in conn.execute(
                    """
                    SELECT ts_start FROM orderbook_metric_bars
                    WHERE stream_id = ? AND timeframe_seconds = 60 AND ts_start >= ? AND ts_start < ?
                    """,
                    (stream_id, gap_start.isoformat(), gap_end.isoformat()),
                )
            }
            for bar in bars:
                if bar["ts_start"] in existing:
                    skipped_existing += 1
                    continue
                insert_metric_bar(conn, bar)
                inserted += 1
            conn.commit()

    valid_samples = [int(bar["valid_samples"]) for bar in bars]
    return {
        "db_path": str(db_path),
        "raw_dir": str(raw_dir),
        "symbol": symbol,
        "pair": pair,
        "stream_id": stream_id,
        "gap_start": gap_start.isoformat(),
        "gap_end": gap_end.isoformat(),
        "sample_seconds": sample_seconds,
        "dry_run": dry_run,
        "archive_days": len(archives),
        "downloaded": downloaded,
        "bars_built": len(bars),
        "bars_inserted": inserted,
        "bars_skipped_existing": skipped_existing,
        "first_bar_start": bars[0]["ts_start"] if bars else None,
        "last_bar_end": bars[-1]["ts_end"] if bars else None,
        "minimum_valid_samples": min(valid_samples) if valid_samples else 0,
        "maximum_valid_samples": max(valid_samples) if valid_samples else 0,
    }


def _build_summary_bars(
    archives: list[tuple[date, Path]],
    *,
    stream_id: str,
    symbol: str,
    pair: str,
    gap_start: datetime,
    gap_end: datetime,
    sample_seconds: int,
) -> list[dict[str, Any]]:
    timeframe = 60
    expected_samples = timeframe // sample_seconds
    next_sample = gap_start
    bar_start = gap_start
    bar_end = bar_start + timedelta(seconds=timeframe)
    ticks: list[dict[str, Any]] = []
    bars: list[dict[str, Any]] = []

    def emit_sample(bids: dict[float, float], asks: dict[float, float], message_count: int) -> None:
        nonlocal next_sample, bar_start, bar_end, ticks
        if bids and asks:
            ticks.append(
                _metric_payload(
                    ts_ms=int(next_sample.timestamp() * 1000),
                    stream_id=stream_id,
                    pair=pair,
                    symbol=symbol,
                    bids=bids,
                    asks=asks,
                    message_count=message_count,
                )
            )
        next_sample += timedelta(seconds=sample_seconds)
        if next_sample >= bar_end:
            if ticks:
                bars.append(_summary_bar(ticks, stream_id, symbol, pair, bar_start, bar_end, expected_samples))
            ticks = []
            bar_start = bar_end
            bar_end = bar_start + timedelta(seconds=timeframe)

    for day, zip_path in archives:
        day_start = datetime.combine(day, datetime.min.time(), timezone.utc)
        day_end = day_start + timedelta(days=1)
        if day_end <= gap_start or day_start >= gap_end:
            continue
        bids: dict[float, float] = {}
        asks: dict[float, float] = {}
        message_count = 0
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
            if not names:
                raise ValueError(f"Archive is empty: {zip_path}")
            with archive.open(names[0]) as handle:
                for raw_line in handle:
                    if not raw_line:
                        continue
                    message = orjson.loads(raw_line)
                    ts_ms = int(message.get("ts") or message.get("cts") or 0)
                    if ts_ms <= 0:
                        continue
                    message_ts = datetime.fromtimestamp(ts_ms / 1000, timezone.utc)
                    while next_sample <= message_ts and next_sample < min(day_end, gap_end):
                        emit_sample(bids, asks, message_count)
                        message_count = 0
                    if message_ts >= gap_end:
                        break
                    data = message.get("data") or {}
                    if message.get("type") == "snapshot":
                        bids = _levels_to_book(data.get("b"))
                        asks = _levels_to_book(data.get("a"))
                    else:
                        _apply_delta(bids, data.get("b"))
                        _apply_delta(asks, data.get("a"))
                    message_count += 1
        while next_sample < min(day_end, gap_end):
            emit_sample(bids, asks, message_count)
            message_count = 0

    return bars


def _summary_bar(
    ticks: list[dict[str, Any]],
    stream_id: str,
    symbol: str,
    pair: str,
    start: datetime,
    end: datetime,
    expected_samples: int,
) -> dict[str, Any]:
    return {
        "ts_start": start.isoformat(),
        "ts_end": end.isoformat(),
        "timeframe_seconds": 60,
        "stream_id": stream_id,
        "market_key": "bybit_spot",
        "venue": "bybit",
        "exchange": "bybit_spot",
        "market_type": "spot",
        "margin_type": "spot",
        "quote_asset": "USDT",
        "canonical_pair": pair,
        "pair": pair,
        "symbol": symbol,
        **aggregate_metric_ticks(ticks, 60, expected_samples=expected_samples),
    }


def _download(url: str, output: Path) -> bool:
    temp = output.with_suffix(output.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=120) as response:
        if response.status_code == 404:
            return False
        response.raise_for_status()
        with temp.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temp.replace(output)
    return True


def _import_day(
    conn: sqlite3.Connection,
    zip_path: Path,
    *,
    stream_id: str,
    symbol: str,
    pair: str,
    day: date,
    sample_seconds: int,
    replace: bool,
) -> int:
    start_ts = datetime.combine(day, datetime.min.time(), timezone.utc)
    end_ts = start_ts + timedelta(days=1)
    if replace:
        conn.execute(
            "DELETE FROM orderbook_metric_ticks WHERE stream_id = ? AND ts >= ? AND ts < ?",
            (stream_id, start_ts.isoformat(), end_ts.isoformat()),
        )
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    next_sample_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)
    pending: list[tuple[Any, ...]] = []
    inserted = 0
    message_count = 0
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if not names:
            return 0
        with archive.open(names[0]) as handle:
            for raw_line in handle:
                if not raw_line:
                    continue
                message = orjson.loads(raw_line)
                ts_ms = int(message.get("ts") or message.get("cts") or 0)
                if ts_ms <= 0:
                    continue
                data = message.get("data") or {}
                if message.get("type") == "snapshot":
                    bids = _levels_to_book(data.get("b"))
                    asks = _levels_to_book(data.get("a"))
                else:
                    _apply_delta(bids, data.get("b"))
                    _apply_delta(asks, data.get("a"))
                message_count += 1
                while next_sample_ms <= ts_ms and next_sample_ms < end_ms:
                    if bids and asks:
                        pending.append(
                            _metric_row(
                                ts_ms=next_sample_ms,
                                stream_id=stream_id,
                                pair=pair,
                                symbol=symbol,
                                bids=bids,
                                asks=asks,
                                message_count=message_count,
                            )
                        )
                        message_count = 0
                        if len(pending) >= 1000:
                            _insert_rows(conn, pending)
                            inserted += len(pending)
                            pending.clear()
                    next_sample_ms += sample_seconds * 1000
    if pending:
        _insert_rows(conn, pending)
        inserted += len(pending)
    conn.commit()
    return inserted


def _metric_row(
    *,
    ts_ms: int,
    stream_id: str,
    pair: str,
    symbol: str,
    bids: dict[float, float],
    asks: dict[float, float],
    message_count: int,
) -> tuple[Any, ...]:
    row = _metric_payload(
        ts_ms=ts_ms,
        stream_id=stream_id,
        pair=pair,
        symbol=symbol,
        bids=bids,
        asks=asks,
        message_count=message_count,
    )
    for key in ("bid_wall_candidates_json", "ask_wall_candidates_json", "bid_liquidity_zones_json", "ask_liquidity_zones_json"):
        row[key] = json.dumps(row.get(key) or [], separators=(",", ":"))
    return tuple(row.get(column) for column in INSERT_COLUMNS)


def _metric_payload(
    *,
    ts_ms: int,
    stream_id: str,
    pair: str,
    symbol: str,
    bids: dict[float, float],
    asks: dict[float, float],
    message_count: int,
) -> dict[str, Any]:
    bid_levels = sorted(bids.items(), key=lambda item: item[0], reverse=True)
    ask_levels = sorted(asks.items(), key=lambda item: item[0])
    metrics = calculate_orderbook_metrics(pair, symbol, bid_levels, ask_levels, DEFAULT_CONFIG, message_count)
    return {
        **metrics,
        "ts": datetime.fromtimestamp(ts_ms / 1000, timezone.utc).isoformat(),
        "stream_id": stream_id,
        "market_key": "bybit_spot",
        "venue": "bybit",
        "exchange": "bybit",
        "market_type": "spot",
        "margin_type": "spot",
        "quote_asset": "USDT",
        "canonical_pair": pair,
        "pair": pair,
        "symbol": symbol,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _insert_rows(conn: sqlite3.Connection, rows: list[tuple[Any, ...]]) -> None:
    placeholders = ",".join("?" for _ in INSERT_COLUMNS)
    columns = ",".join(INSERT_COLUMNS)
    conn.executemany(f"INSERT INTO orderbook_metric_ticks ({columns}) VALUES ({placeholders})", rows)


def _levels_to_book(levels: Any) -> dict[float, float]:
    book: dict[float, float] = {}
    for price, qty in levels or []:
        p = float(price)
        q = float(qty)
        if q > 0:
            book[p] = q
    return book


def _apply_delta(book: dict[float, float], levels: Any) -> None:
    for price, qty in levels or []:
        p = float(price)
        q = float(qty)
        if q <= 0:
            book.pop(p, None)
        else:
            book[p] = q


def _daily_url(base_url: str, symbol: str, day: date) -> str:
    return f"{base_url}/{symbol}/{day.isoformat()}_{symbol}_ob200.data.zip"


def _stream_id(symbol: str) -> str:
    return f"hist_bybit_spot_{symbol}_ob200"


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp must include a UTC offset: {value}")
    return parsed.astimezone(timezone.utc)


def _date_range(start: date, end: date) -> Iterable[date]:
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


if __name__ == "__main__":
    raise SystemExit(main())

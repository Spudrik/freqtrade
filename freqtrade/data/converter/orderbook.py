"""
Utilities for converting raw order book archives into aggregated feature dataframes.
"""

import json
import logging
import zipfile
from bisect import bisect_left
from pathlib import Path

import ccxt
import pandas as pd

from freqtrade.misc import pair_to_filename


logger = logging.getLogger(__name__)

DEFAULT_ORDERBOOK_LEVELS = (5, 25, 100)
ORDERBOOK_FEATURE_FORMATS = ("feather", "parquet")


def orderbook_feature_filename(
    datadir: Path,
    pair: str,
    timeframe: str,
    *,
    category: str = "linear",
    depth: int = 500,
    data_format: str = "feather",
) -> Path:
    if data_format not in ORDERBOOK_FEATURE_FORMATS:
        raise ValueError(
            f"Unsupported order book feature format '{data_format}'. "
            f"Supported formats: {', '.join(ORDERBOOK_FEATURE_FORMATS)}."
        )
    pair_s = pair_to_filename(pair)
    return (
        datadir
        / "orderbook_features"
        / category
        / f"{pair_s}-{timeframe}-ob{depth}.{data_format}"
    )


def store_orderbook_features(
    datadir: Path,
    pair: str,
    timeframe: str,
    dataframe: pd.DataFrame,
    *,
    category: str = "linear",
    depth: int = 500,
    data_format: str = "feather",
) -> Path:
    filename = orderbook_feature_filename(
        datadir, pair, timeframe, category=category, depth=depth, data_format=data_format
    )
    filename.parent.mkdir(parents=True, exist_ok=True)
    if data_format == "feather":
        dataframe.reset_index(drop=True).to_feather(
            filename, compression="lz4", compression_level=9
        )
    elif data_format == "parquet":
        dataframe.reset_index(drop=True).to_parquet(filename)
    else:
        raise ValueError(
            f"Unsupported order book feature format '{data_format}'. "
            f"Supported formats: {', '.join(ORDERBOOK_FEATURE_FORMATS)}."
        )
    return filename


def load_orderbook_features(
    datadir: Path,
    pair: str,
    timeframe: str,
    *,
    category: str = "linear",
    depth: int = 500,
    data_format: str = "feather",
) -> pd.DataFrame:
    filename = orderbook_feature_filename(
        datadir, pair, timeframe, category=category, depth=depth, data_format=data_format
    )
    if not filename.exists():
        return pd.DataFrame()
    if data_format == "feather":
        return pd.read_feather(filename)
    if data_format == "parquet":
        return pd.read_parquet(filename)
    raise ValueError(
        f"Unsupported order book feature format '{data_format}'. "
        f"Supported formats: {', '.join(ORDERBOOK_FEATURE_FORMATS)}."
    )


def _reset_book_side(levels: list[list[str]]) -> tuple[list[float], dict[float, float]]:
    prices = []
    sizes: dict[float, float] = {}
    for price_raw, size_raw in levels:
        price = float(price_raw)
        size = float(size_raw)
        if size <= 0:
            continue
        prices.append(price)
        sizes[price] = size
    prices.sort()
    return prices, sizes


def _apply_delta(
    price_levels: list[float], sizes: dict[float, float], updates: list[list[str]]
) -> None:
    for price_raw, size_raw in updates:
        price = float(price_raw)
        size = float(size_raw)
        if price in sizes:
            if size <= 0:
                del sizes[price]
                idx = bisect_left(price_levels, price)
                if idx < len(price_levels) and price_levels[idx] == price:
                    del price_levels[idx]
            else:
                sizes[price] = size
        elif size > 0:
            sizes[price] = size
            price_levels.insert(bisect_left(price_levels, price), price)


def _top_levels(
    price_levels: list[float], sizes: dict[float, float], count: int, reverse: bool
) -> list[tuple[float, float]]:
    if reverse:
        selected = price_levels[-count:][::-1]
    else:
        selected = price_levels[:count]
    return [(price, sizes[price]) for price in selected]


def _calculate_book_metrics(
    bid_levels: list[float],
    bid_sizes: dict[float, float],
    ask_levels: list[float],
    ask_sizes: dict[float, float],
    levels: tuple[int, ...],
) -> dict[str, float]:
    if not bid_levels or not ask_levels:
        return {}

    best_bid = bid_levels[-1]
    best_ask = ask_levels[0]
    best_bid_size = bid_sizes[best_bid]
    best_ask_size = ask_sizes[best_ask]
    mid_price = (best_bid + best_ask) / 2
    spread = best_ask - best_bid
    total_top = best_bid_size + best_ask_size
    microprice = (
        ((best_ask * best_bid_size) + (best_bid * best_ask_size)) / total_top
        if total_top > 0
        else mid_price
    )

    metrics: dict[str, float] = {
        "mid_price": mid_price,
        "spread": spread,
        "spread_bps": (spread / mid_price) * 10000 if mid_price > 0 else 0.0,
        "microprice": microprice,
        "best_bid_size": best_bid_size,
        "best_ask_size": best_ask_size,
    }

    for level in levels:
        bid_depth = sum(size for _, size in _top_levels(bid_levels, bid_sizes, level, True))
        ask_depth = sum(size for _, size in _top_levels(ask_levels, ask_sizes, level, False))
        total_depth = bid_depth + ask_depth
        metrics[f"bid_depth_{level}"] = bid_depth
        metrics[f"ask_depth_{level}"] = ask_depth
        metrics[f"imbalance_{level}"] = (
            (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0.0
        )
    return metrics


def _update_bucket(bucket: dict | None, bucket_start_ms: int, metrics: dict[str, float]) -> dict:
    if bucket is None:
        return {
            "bucket_start_ms": bucket_start_ms,
            "updates": 1,
            "mid_open": metrics["mid_price"],
            "mid_high": metrics["mid_price"],
            "mid_low": metrics["mid_price"],
            "mid_close": metrics["mid_price"],
            "spread_bps_sum": metrics["spread_bps"],
            "spread_bps_max": metrics["spread_bps"],
            "microprice_sum": metrics["microprice"],
            "best_bid_size_sum": metrics["best_bid_size"],
            "best_ask_size_sum": metrics["best_ask_size"],
            **{
                f"bid_depth_{level}_sum": metrics[f"bid_depth_{level}"]
                for level in DEFAULT_ORDERBOOK_LEVELS
            },
            **{
                f"ask_depth_{level}_sum": metrics[f"ask_depth_{level}"]
                for level in DEFAULT_ORDERBOOK_LEVELS
            },
            **{
                f"imbalance_{level}_sum": metrics[f"imbalance_{level}"]
                for level in DEFAULT_ORDERBOOK_LEVELS
            },
        }

    bucket["updates"] += 1
    bucket["mid_high"] = max(bucket["mid_high"], metrics["mid_price"])
    bucket["mid_low"] = min(bucket["mid_low"], metrics["mid_price"])
    bucket["mid_close"] = metrics["mid_price"]
    bucket["spread_bps_sum"] += metrics["spread_bps"]
    bucket["spread_bps_max"] = max(bucket["spread_bps_max"], metrics["spread_bps"])
    bucket["microprice_sum"] += metrics["microprice"]
    bucket["best_bid_size_sum"] += metrics["best_bid_size"]
    bucket["best_ask_size_sum"] += metrics["best_ask_size"]
    for level in DEFAULT_ORDERBOOK_LEVELS:
        bucket[f"bid_depth_{level}_sum"] += metrics[f"bid_depth_{level}"]
        bucket[f"ask_depth_{level}_sum"] += metrics[f"ask_depth_{level}"]
        bucket[f"imbalance_{level}_sum"] += metrics[f"imbalance_{level}"]
    return bucket


def _finalize_bucket(bucket: dict) -> dict:
    updates = bucket["updates"]
    row = {
        "date": pd.to_datetime(bucket["bucket_start_ms"], unit="ms", utc=True),
        "mid_open": bucket["mid_open"],
        "mid_high": bucket["mid_high"],
        "mid_low": bucket["mid_low"],
        "mid_close": bucket["mid_close"],
        "spread_bps_mean": bucket["spread_bps_sum"] / updates,
        "spread_bps_max": bucket["spread_bps_max"],
        "microprice_mean": bucket["microprice_sum"] / updates,
        "best_bid_size_mean": bucket["best_bid_size_sum"] / updates,
        "best_ask_size_mean": bucket["best_ask_size_sum"] / updates,
        "updates": updates,
    }
    for level in DEFAULT_ORDERBOOK_LEVELS:
        row[f"bid_depth_{level}_mean"] = bucket[f"bid_depth_{level}_sum"] / updates
        row[f"ask_depth_{level}_mean"] = bucket[f"ask_depth_{level}_sum"] / updates
        row[f"imbalance_{level}_mean"] = bucket[f"imbalance_{level}_sum"] / updates
    return row


def convert_bybit_orderbook_archive_to_features(
    files: list[Path],
    timeframe: str,
    *,
    max_rows: int | None = None,
) -> pd.DataFrame:
    timeframe_ms = ccxt.Exchange.parse_timeframe(timeframe) * 1000
    rows: list[dict] = []
    bid_levels: list[float] = []
    ask_levels: list[float] = []
    bid_sizes: dict[float, float] = {}
    ask_sizes: dict[float, float] = {}
    current_bucket: dict | None = None
    processed_rows = 0

    for path in sorted(files):
        logger.info("Converting Bybit order book archive to features: %s", path)
        with zipfile.ZipFile(path) as zipf:
            name = zipf.namelist()[0]
            with zipf.open(name) as fp:
                for raw_line in fp:
                    if max_rows is not None and processed_rows >= max_rows:
                        break
                    event = json.loads(raw_line)
                    if event["type"] == "snapshot":
                        bid_levels, bid_sizes = _reset_book_side(event["data"]["b"])
                        ask_levels, ask_sizes = _reset_book_side(event["data"]["a"])
                    else:
                        _apply_delta(bid_levels, bid_sizes, event["data"]["b"])
                        _apply_delta(ask_levels, ask_sizes, event["data"]["a"])

                    metrics = _calculate_book_metrics(
                        bid_levels, bid_sizes, ask_levels, ask_sizes, DEFAULT_ORDERBOOK_LEVELS
                    )
                    if not metrics:
                        processed_rows += 1
                        continue

                    event_ts = int(event.get("cts") or event.get("ts"))
                    bucket_start_ms = event_ts - (event_ts % timeframe_ms)
                    if (
                        current_bucket is not None
                        and current_bucket["bucket_start_ms"] != bucket_start_ms
                    ):
                        rows.append(_finalize_bucket(current_bucket))
                        current_bucket = None
                    current_bucket = _update_bucket(current_bucket, bucket_start_ms, metrics)
                    processed_rows += 1

                if max_rows is not None and processed_rows >= max_rows:
                    break

    if current_bucket is not None:
        rows.append(_finalize_bucket(current_bucket))

    return pd.DataFrame(rows)

from __future__ import annotations

import statistics
from typing import Any


def normalize_freqtrade_pair_to_binance_symbol(pair: str) -> str | None:
    raw = str(pair or "").strip().upper()
    if not raw or "/" not in raw:
        return None
    base, quote_part = raw.split("/", 1)
    base = base.strip()
    quote = quote_part.split(":", 1)[0].strip()
    if not base or not quote:
        return None
    if not base.isalnum() or not quote.isalnum():
        return None
    return f"{base}{quote}"


def normalize_whitelist_pairs(raw_pairs: list[str], max_symbols: int) -> list[dict[str, str]]:
    seen: set[str] = set()
    normalized: list[dict[str, str]] = []
    for pair in raw_pairs:
        pair_text = str(pair or "").strip()
        symbol = normalize_freqtrade_pair_to_binance_symbol(pair_text)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        normalized.append({"pair": pair_text, "symbol": symbol})
        if len(normalized) >= max(1, int(max_symbols)):
            break
    return normalized


def parse_book_side(levels: Any, *, reverse: bool = False) -> list[tuple[float, float]]:
    parsed: list[tuple[float, float]] = []
    if not isinstance(levels, list):
        return parsed
    for level in levels:
        if not isinstance(level, (list, tuple)) or len(level) < 2:
            continue
        try:
            price = float(level[0])
            qty = float(level[1])
        except Exception:
            continue
        if price <= 0 or qty < 0:
            continue
        parsed.append((price, qty))
    parsed.sort(key=lambda item: item[0], reverse=reverse)
    return parsed


def _sum_notional(levels: list[tuple[float, float]], top_n: int) -> float:
    return sum(price * qty for price, qty in levels[:top_n])


def _imbalance(bid_value: float, ask_value: float) -> float | None:
    denom = bid_value + ask_value
    if denom <= 0:
        return None
    return (bid_value - ask_value) / denom


def _liquidity_in_band(levels: list[tuple[float, float]], lower: float, upper: float) -> float:
    total = 0.0
    for price, qty in levels:
        if lower <= price <= upper:
            total += price * qty
    return total


def _median_notional(levels: list[tuple[float, float]]) -> float | None:
    notionals = [price * qty for price, qty in levels if qty > 0]
    if not notionals:
        return None
    return float(statistics.median(notionals))


def _wall_info(side_levels: list[tuple[float, float]], mid_price: float, threshold: float, is_bid: bool) -> tuple[dict[str, float | None], dict[str, float | None]]:
    median_notional = _median_notional(side_levels)
    nearest = {"price": None, "distance_bps": None, "notional": None, "score": None}
    strongest_50 = {"price": None, "score": None}
    best_nearest_distance: float | None = None
    best_strongest_score: float | None = None
    if not median_notional or median_notional <= 0:
        return nearest, strongest_50

    for price, qty in side_levels:
        notional = price * qty
        score = notional / median_notional
        distance_bps = abs((price - mid_price) / mid_price * 10000.0)
        if score >= threshold:
            if best_nearest_distance is None or distance_bps < best_nearest_distance:
                best_nearest_distance = distance_bps
                nearest = {
                    "price": price,
                    "distance_bps": distance_bps,
                    "notional": notional,
                    "score": score,
                }
        if distance_bps <= 50:
            if best_strongest_score is None or score > best_strongest_score:
                best_strongest_score = score
                strongest_50 = {"price": price, "score": score}
    _ = is_bid
    return nearest, strongest_50


def calculate_orderbook_metrics(pair: str, symbol: str, bids: list[tuple[float, float]], asks: list[tuple[float, float]], config: dict[str, Any], message_count_interval: int) -> dict[str, Any]:
    depth_levels = int(config.get("depth_levels", 20))
    book_valid = bool(bids and asks)
    output: dict[str, Any] = {
        "pair": pair,
        "symbol": symbol,
        "book_valid": int(book_valid),
        "depth_levels": depth_levels,
        "message_count_interval": int(message_count_interval),
    }
    if not book_valid:
        return output

    best_bid, top_bid_qty = bids[0]
    best_ask, top_ask_qty = asks[0]
    if best_bid <= 0 or best_ask <= 0 or best_ask < best_bid:
        return output

    mid_price = (best_bid + best_ask) / 2.0
    spread_bps = (best_ask - best_bid) / mid_price * 10000.0 if mid_price else None
    microprice = None
    microprice_offset_bps = None
    denom = top_bid_qty + top_ask_qty
    if denom > 0 and mid_price > 0:
        microprice = (best_ask * top_bid_qty + best_bid * top_ask_qty) / denom
        microprice_offset_bps = (microprice - mid_price) / mid_price * 10000.0

    top1_bid = _sum_notional(bids, 1)
    top1_ask = _sum_notional(asks, 1)
    top5_bid = _sum_notional(bids, 5)
    top5_ask = _sum_notional(asks, 5)
    top10_bid = _sum_notional(bids, 10)
    top10_ask = _sum_notional(asks, 10)
    top20_bid = _sum_notional(bids, 20)
    top20_ask = _sum_notional(asks, 20)

    liquidity_bands = [int(value) for value in config.get("liquidity_bps_bands", [5, 10, 25, 50])]
    band_values: dict[int, tuple[float, float, float | None]] = {}
    for band in liquidity_bands:
        lower = mid_price * (1.0 - (band / 10000.0))
        upper = mid_price * (1.0 + (band / 10000.0))
        bid_liq = _liquidity_in_band(bids, lower, mid_price)
        ask_liq = _liquidity_in_band(asks, mid_price, upper)
        band_values[band] = (bid_liq, ask_liq, _imbalance(bid_liq, ask_liq))

    wall_threshold = float(config.get("wall_score_threshold", 4.0))
    nearest_bid, strongest_bid = _wall_info(bids, mid_price, wall_threshold, True)
    nearest_ask, strongest_ask = _wall_info(asks, mid_price, wall_threshold, False)

    imbalance_top20 = _imbalance(top20_bid, top20_ask)
    pressure_threshold = float(config.get("pressure_threshold", 0.35))
    extreme_threshold = float(config.get("extreme_pressure_threshold", 0.60))

    output.update(
        {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid_price,
            "spread_bps": spread_bps,
            "microprice": microprice,
            "microprice_offset_bps": microprice_offset_bps,
            "bid_notional_top1": top1_bid,
            "ask_notional_top1": top1_ask,
            "imbalance_top1": _imbalance(top1_bid, top1_ask),
            "bid_notional_top5": top5_bid,
            "ask_notional_top5": top5_ask,
            "imbalance_top5": _imbalance(top5_bid, top5_ask),
            "bid_notional_top10": top10_bid,
            "ask_notional_top10": top10_ask,
            "imbalance_top10": _imbalance(top10_bid, top10_ask),
            "bid_notional_top20": top20_bid,
            "ask_notional_top20": top20_ask,
            "imbalance_top20": imbalance_top20,
            "bid_liquidity_5bps": band_values.get(5, (None, None, None))[0],
            "ask_liquidity_5bps": band_values.get(5, (None, None, None))[1],
            "imbalance_5bps": band_values.get(5, (None, None, None))[2],
            "bid_liquidity_10bps": band_values.get(10, (None, None, None))[0],
            "ask_liquidity_10bps": band_values.get(10, (None, None, None))[1],
            "imbalance_10bps": band_values.get(10, (None, None, None))[2],
            "bid_liquidity_25bps": band_values.get(25, (None, None, None))[0],
            "ask_liquidity_25bps": band_values.get(25, (None, None, None))[1],
            "imbalance_25bps": band_values.get(25, (None, None, None))[2],
            "bid_liquidity_50bps": band_values.get(50, (None, None, None))[0],
            "ask_liquidity_50bps": band_values.get(50, (None, None, None))[1],
            "imbalance_50bps": band_values.get(50, (None, None, None))[2],
            "nearest_bid_wall_price": nearest_bid["price"],
            "nearest_bid_wall_distance_bps": nearest_bid["distance_bps"],
            "nearest_bid_wall_notional": nearest_bid["notional"],
            "nearest_bid_wall_score": nearest_bid["score"],
            "nearest_ask_wall_price": nearest_ask["price"],
            "nearest_ask_wall_distance_bps": nearest_ask["distance_bps"],
            "nearest_ask_wall_notional": nearest_ask["notional"],
            "nearest_ask_wall_score": nearest_ask["score"],
            "strongest_bid_wall_price_50bps": strongest_bid["price"],
            "strongest_bid_wall_score_50bps": strongest_bid["score"],
            "strongest_ask_wall_price_50bps": strongest_ask["price"],
            "strongest_ask_wall_score_50bps": strongest_ask["score"],
            "strong_bid_pressure": int(bool(imbalance_top20 is not None and imbalance_top20 >= pressure_threshold)),
            "strong_ask_pressure": int(bool(imbalance_top20 is not None and imbalance_top20 <= -pressure_threshold)),
            "extreme_bid_pressure": int(bool(imbalance_top20 is not None and imbalance_top20 >= extreme_threshold)),
            "extreme_ask_pressure": int(bool(imbalance_top20 is not None and imbalance_top20 <= -extreme_threshold)),
        }
    )
    return output


def aggregate_metric_ticks(ticks: list[dict[str, Any]], timeframe_seconds: int, expected_samples: int) -> dict[str, Any]:
    valid = [tick for tick in ticks if int(tick.get("book_valid", 0)) == 1]
    spreads = [float(t["spread_bps"]) for t in valid if t.get("spread_bps") is not None]
    micro_offsets = [float(t["microprice_offset_bps"]) for t in valid if t.get("microprice_offset_bps") is not None]
    imb20 = [float(t["imbalance_top20"]) for t in valid if t.get("imbalance_top20") is not None]
    imb10 = [float(t["imbalance_10bps"]) for t in valid if t.get("imbalance_10bps") is not None]
    imb25 = [float(t["imbalance_25bps"]) for t in valid if t.get("imbalance_25bps") is not None]
    bid_pressure_seconds = sum(1.0 for t in valid if int(t.get("strong_bid_pressure", 0)) == 1)
    ask_pressure_seconds = sum(1.0 for t in valid if int(t.get("strong_ask_pressure", 0)) == 1)
    return {
        "timeframe_seconds": timeframe_seconds,
        "valid_samples": len(valid),
        "expected_samples": expected_samples,
        "spread_bps_mean": (sum(spreads) / len(spreads)) if spreads else None,
        "spread_bps_max": max(spreads) if spreads else None,
        "microprice_offset_bps_mean": (sum(micro_offsets) / len(micro_offsets)) if micro_offsets else None,
        "imbalance_top20_mean": (sum(imb20) / len(imb20)) if imb20 else None,
        "imbalance_top20_min": min(imb20) if imb20 else None,
        "imbalance_top20_max": max(imb20) if imb20 else None,
        "imbalance_10bps_mean": (sum(imb10) / len(imb10)) if imb10 else None,
        "imbalance_25bps_mean": (sum(imb25) / len(imb25)) if imb25 else None,
        "bid_pressure_seconds": bid_pressure_seconds,
        "ask_pressure_seconds": ask_pressure_seconds,
        "bid_pressure_ratio": (bid_pressure_seconds / max(1.0, float(expected_samples))),
        "ask_pressure_ratio": (ask_pressure_seconds / max(1.0, float(expected_samples))),
        "max_bid_pressure_streak_seconds": None,
        "max_ask_pressure_streak_seconds": None,
        "nearest_bid_wall_min_distance_bps": min((float(t["nearest_bid_wall_distance_bps"]) for t in valid if t.get("nearest_bid_wall_distance_bps") is not None), default=None),
        "nearest_ask_wall_min_distance_bps": min((float(t["nearest_ask_wall_distance_bps"]) for t in valid if t.get("nearest_ask_wall_distance_bps") is not None), default=None),
        "strongest_bid_wall_score": max((float(t["strongest_bid_wall_score_50bps"]) for t in valid if t.get("strongest_bid_wall_score_50bps") is not None), default=None),
        "strongest_ask_wall_score": max((float(t["strongest_ask_wall_score_50bps"]) for t in valid if t.get("strongest_ask_wall_score_50bps") is not None), default=None),
    }


def estimate_storage_usage(pair_count: int, metric_interval_seconds: int, snapshot_interval_seconds: int, depth_levels: int, store_snapshots: bool) -> dict[str, float]:
    pairs = max(0, int(pair_count))
    metric_interval = max(1, int(metric_interval_seconds))
    snapshot_interval = max(1, int(snapshot_interval_seconds))
    metric_rows_per_day = pairs * 86400.0 / metric_interval
    snapshot_rows_per_day = pairs * 86400.0 / snapshot_interval if store_snapshots else 0.0
    estimated_metric_mb_per_day = metric_rows_per_day * 900.0 / 1048576.0
    estimated_snapshot_mb_per_day = snapshot_rows_per_day * max(2000.0, float(depth_levels) * 2.0 * 80.0) / 1048576.0
    return {
        "metric_rows_per_day": metric_rows_per_day,
        "snapshot_rows_per_day": snapshot_rows_per_day,
        "estimated_metric_mb_per_day": estimated_metric_mb_per_day,
        "estimated_snapshot_mb_per_day": estimated_snapshot_mb_per_day,
        "estimated_total_mb_per_day": estimated_metric_mb_per_day + estimated_snapshot_mb_per_day,
    }


if __name__ == "__main__":
    bids = parse_book_side([["100.0", "1.2"], ["99.9", "2.0"], ["99.8", "3.0"]], reverse=True)
    asks = parse_book_side([["100.1", "1.1"], ["100.2", "1.5"], ["100.3", "2.2"]], reverse=False)
    sample_config = {"depth_levels": 20, "liquidity_bps_bands": [5, 10, 25, 50], "wall_score_threshold": 4.0}
    metrics = calculate_orderbook_metrics("BTC/USDT:USDT", "BTCUSDT", bids, asks, sample_config, 25)
    print("pair:", normalize_freqtrade_pair_to_binance_symbol("BTC/USDT:USDT"))
    print("spread_bps:", metrics.get("spread_bps"))

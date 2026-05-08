from __future__ import annotations

import json
import math
import statistics
from datetime import datetime
from typing import Any

from .markets import pair_to_symbol, MARKET_PROFILES, normalize_whitelist_pairs


def normalize_freqtrade_pair_to_binance_symbol(pair: str) -> str | None:
    return pair_to_symbol(pair, MARKET_PROFILES["binance_usdm_futures"])


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


def _distance_bps(price: float, reference: float) -> float | None:
    if reference <= 0:
        return None
    return abs((price - reference) / reference * 10000.0)


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


def _wall_candidates(
    side_levels: list[tuple[float, float]],
    mid_price: float,
    *,
    side: str,
    min_score: float,
    max_distance_bps: float,
    max_count: int,
) -> list[dict[str, float | str]]:
    median_notional = _median_notional(side_levels)
    if not median_notional or median_notional <= 0:
        return []
    candidates: list[dict[str, float | str]] = []
    for price, qty in side_levels:
        notional = price * qty
        score = notional / median_notional
        distance = _distance_bps(price, mid_price)
        if distance is None or distance > max_distance_bps or score < min_score:
            continue
        candidates.append(
            {
                "side": side,
                "price": float(price),
                "distance_bps": float(distance),
                "notional": float(notional),
                "score": float(score),
            }
        )
    candidates.sort(key=lambda item: (float(item["distance_bps"]), -float(item["score"])))
    return candidates[: max(0, int(max_count))]


def _liquidity_zone_candidates(
    side_levels: list[tuple[float, float]],
    mid_price: float,
    *,
    side: str,
    zone_width_bps: float,
    max_distance_bps: float,
    high_count: int,
    low_count: int,
) -> list[dict[str, float | str]]:
    if mid_price <= 0 or zone_width_bps <= 0 or max_distance_bps <= 0:
        return []
    bucket_count = max(1, int(math.ceil(max_distance_bps / zone_width_bps)))
    buckets: list[dict[str, float | int]] = []
    for bucket_index in range(bucket_count):
        inner_bps = bucket_index * zone_width_bps
        outer_bps = min(max_distance_bps, (bucket_index + 1) * zone_width_bps)
        if side == "bid":
            lower_price = mid_price * (1.0 - outer_bps / 10000.0)
            upper_price = mid_price * (1.0 - inner_bps / 10000.0)
        else:
            lower_price = mid_price * (1.0 + inner_bps / 10000.0)
            upper_price = mid_price * (1.0 + outer_bps / 10000.0)
        notional = _liquidity_in_band(side_levels, lower_price, upper_price)
        buckets.append(
            {
                "bucket_index": bucket_index,
                "inner_bps": float(inner_bps),
                "outer_bps": float(outer_bps),
                "lower_price": float(lower_price),
                "upper_price": float(upper_price),
                "mid_price": float((lower_price + upper_price) / 2.0),
                "distance_bps": float((inner_bps + outer_bps) / 2.0),
                "notional": float(notional),
            }
        )
    positive_notionals = [float(bucket["notional"]) for bucket in buckets if float(bucket["notional"]) > 0]
    baseline = float(statistics.median(positive_notionals)) if positive_notionals else 1.0
    for bucket in buckets:
        bucket["score"] = float(bucket["notional"]) / baseline if baseline > 0 else 0.0

    high_buckets = sorted(buckets, key=lambda item: (-float(item["notional"]), int(item["bucket_index"])))[: max(0, int(high_count))]
    high_indexes = {int(bucket["bucket_index"]) for bucket in high_buckets}
    low_pool = [bucket for bucket in buckets if int(bucket["bucket_index"]) not in high_indexes]
    low_buckets = sorted(low_pool, key=lambda item: (float(item["notional"]), int(item["bucket_index"])))[: max(0, int(low_count))]

    zones: list[dict[str, float | str]] = []
    for kind, selected in (("high", high_buckets), ("low", low_buckets)):
        for rank, bucket in enumerate(selected, start=1):
            zones.append(
                {
                    "side": side,
                    "kind": kind,
                    "rank": float(rank),
                    "bucket_index": float(bucket["bucket_index"]),
                    "inner_bps": float(bucket["inner_bps"]),
                    "outer_bps": float(bucket["outer_bps"]),
                    "lower_price": float(bucket["lower_price"]),
                    "upper_price": float(bucket["upper_price"]),
                    "mid_price": float(bucket["mid_price"]),
                    "distance_bps": float(bucket["distance_bps"]),
                    "notional": float(bucket["notional"]),
                    "score": float(bucket["score"]),
                }
            )
    return zones


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
    wall_candidate_min_score = float(config.get("wall_candidate_min_score", max(2.0, wall_threshold * 0.5)))
    wall_candidate_max_distance_bps = float(config.get("wall_candidate_max_distance_bps", 100.0))
    wall_candidate_count = int(config.get("wall_candidate_count", 3))
    nearest_bid, strongest_bid = _wall_info(bids, mid_price, wall_threshold, True)
    nearest_ask, strongest_ask = _wall_info(asks, mid_price, wall_threshold, False)
    bid_wall_candidates = _wall_candidates(
        bids,
        mid_price,
        side="bid",
        min_score=wall_candidate_min_score,
        max_distance_bps=wall_candidate_max_distance_bps,
        max_count=wall_candidate_count,
    )
    ask_wall_candidates = _wall_candidates(
        asks,
        mid_price,
        side="ask",
        min_score=wall_candidate_min_score,
        max_distance_bps=wall_candidate_max_distance_bps,
        max_count=wall_candidate_count,
    )

    zone_width_bps = float(config.get("liquidity_zone_width_bps", 10.0))
    zone_max_bps = float(config.get("liquidity_zone_max_bps", 80.0))
    high_zone_count = int(config.get("liquidity_high_zone_count", 2))
    low_zone_count = int(config.get("liquidity_low_zone_count", 2))
    bid_liquidity_zones = _liquidity_zone_candidates(
        bids,
        mid_price,
        side="bid",
        zone_width_bps=zone_width_bps,
        max_distance_bps=zone_max_bps,
        high_count=high_zone_count,
        low_count=low_zone_count,
    )
    ask_liquidity_zones = _liquidity_zone_candidates(
        asks,
        mid_price,
        side="ask",
        zone_width_bps=zone_width_bps,
        max_distance_bps=zone_max_bps,
        high_count=high_zone_count,
        low_count=low_zone_count,
    )

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
            "bid_wall_candidates_json": bid_wall_candidates,
            "ask_wall_candidates_json": ask_wall_candidates,
            "bid_liquidity_zones_json": bid_liquidity_zones,
            "ask_liquidity_zones_json": ask_liquidity_zones,
            "strong_bid_pressure": int(bool(imbalance_top20 is not None and imbalance_top20 >= pressure_threshold)),
            "strong_ask_pressure": int(bool(imbalance_top20 is not None and imbalance_top20 <= -pressure_threshold)),
            "extreme_bid_pressure": int(bool(imbalance_top20 is not None and imbalance_top20 >= extreme_threshold)),
            "extreme_ask_pressure": int(bool(imbalance_top20 is not None and imbalance_top20 <= -extreme_threshold)),
        }
    )
    return output


def _json_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return value if isinstance(value, list) else []


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _log_score(value: float, cap: float = 10.0) -> float:
    return min(1.0, math.log1p(max(0.0, value)) / math.log1p(cap))


def _aggregate_wall_blocks(ticks: list[dict[str, Any]], side: str, valid_count: int, *, max_blocks: int = 3, tolerance_bps: float = 3.0) -> list[dict[str, float | str]]:
    observations: list[dict[str, Any]] = []
    key = f"{side}_wall_candidates_json"
    for tick in ticks:
        tick_ts = _parse_ts(tick.get("ts"))
        for item in _json_list(tick.get(key)):
            try:
                price = float(item["price"])
                score = float(item.get("score", 0.0))
                notional = float(item.get("notional", 0.0))
                distance_bps = float(item.get("distance_bps", 0.0))
            except (KeyError, TypeError, ValueError):
                continue
            observations.append(
                {
                    "ts": tick_ts,
                    "price": price,
                    "score": score,
                    "notional": notional,
                    "distance_bps": distance_bps,
                }
            )
    groups: list[dict[str, Any]] = []
    for observation in observations:
        target: dict[str, Any] | None = None
        for group in groups:
            distance = _distance_bps(float(observation["price"]), float(group["price"]))
            if distance is not None and distance <= tolerance_bps:
                target = group
                break
        if target is None:
            target = {"price": float(observation["price"]), "items": []}
            groups.append(target)
        target["items"].append(observation)
        prices = [float(item["price"]) for item in target["items"]]
        weights = [max(float(item["score"]), 0.01) for item in target["items"]]
        weight_sum = sum(weights)
        target["price"] = sum(price * weight for price, weight in zip(prices, weights)) / weight_sum if weight_sum > 0 else sum(prices) / len(prices)

    blocks: list[dict[str, float | str]] = []
    denominator = max(1, int(valid_count))
    for group in groups:
        items = group["items"]
        scores = [float(item["score"]) for item in items]
        notionals = [float(item["notional"]) for item in items]
        distances = [float(item["distance_bps"]) for item in items]
        timestamps = [item["ts"] for item in items if item.get("ts") is not None]
        first_seen = min(timestamps).isoformat() if timestamps else ""
        last_seen = max(timestamps).isoformat() if timestamps else ""
        age_seconds = (max(timestamps) - min(timestamps)).total_seconds() if len(timestamps) >= 2 else 0.0
        persistence = len(items) / denominator
        score_mean = sum(scores) / len(scores)
        score_max = max(scores)
        min_distance = min(distances)
        resilience = min(
            1.0,
            (0.35 * persistence)
            + (0.25 * _log_score(score_mean))
            + (0.25 * _log_score(score_max))
            + (0.15 * (1.0 - min(min_distance, 100.0) / 100.0)),
        )
        blocks.append(
            {
                "side": side,
                "price": float(group["price"]),
                "distance_bps": float(min_distance),
                "last_distance_bps": float(distances[-1]),
                "notional_mean": float(sum(notionals) / len(notionals)),
                "notional_max": float(max(notionals)),
                "score_mean": float(score_mean),
                "score_max": float(score_max),
                "seen_count": float(len(items)),
                "persistence": float(persistence),
                "age_seconds": float(age_seconds),
                "resilience_score": float(resilience),
                "first_seen": first_seen,
                "last_seen": last_seen,
            }
        )
    blocks.sort(key=lambda item: (-float(item["resilience_score"]), float(item["distance_bps"])))
    for slot, block in enumerate(blocks[: max(0, int(max_blocks))], start=1):
        block["slot"] = float(slot)
    return blocks[: max(0, int(max_blocks))]


def _aggregate_liquidity_zone_blocks(ticks: list[dict[str, Any]], side: str, valid_count: int, *, per_kind_count: int = 2) -> list[dict[str, float | str]]:
    key = f"{side}_liquidity_zones_json"
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for tick in ticks:
        for item in _json_list(tick.get(key)):
            try:
                kind = str(item["kind"])
                bucket_index = int(float(item["bucket_index"]))
            except (KeyError, TypeError, ValueError):
                continue
            groups.setdefault((kind, bucket_index), []).append(item)

    denominator = max(1, int(valid_count))
    zones: list[dict[str, float | str]] = []
    for (kind, bucket_index), items in groups.items():
        def mean_field(name: str) -> float:
            values = []
            for item in items:
                try:
                    values.append(float(item[name]))
                except (KeyError, TypeError, ValueError):
                    continue
            return float(sum(values) / len(values)) if values else 0.0

        score_mean = mean_field("score")
        persistence = len(items) / denominator
        rank_score = (score_mean * persistence) if kind == "high" else ((1.0 / (1.0 + score_mean)) * persistence)
        zones.append(
            {
                "side": side,
                "kind": kind,
                "bucket_index": float(bucket_index),
                "lower_price": mean_field("lower_price"),
                "upper_price": mean_field("upper_price"),
                "mid_price": mean_field("mid_price"),
                "distance_bps": mean_field("distance_bps"),
                "notional_mean": mean_field("notional"),
                "score_mean": float(score_mean),
                "persistence": float(persistence),
                "rank_score": float(rank_score),
            }
        )

    selected: list[dict[str, float | str]] = []
    for kind in ("high", "low"):
        kind_zones = [zone for zone in zones if zone["kind"] == kind]
        kind_zones.sort(key=lambda item: (-float(item["rank_score"]), float(item["distance_bps"])))
        for slot, zone in enumerate(kind_zones[: max(0, int(per_kind_count))], start=1):
            zone = dict(zone)
            zone["slot"] = float(slot)
            selected.append(zone)
    return selected


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
        "bid_wall_blocks_json": _aggregate_wall_blocks(valid, "bid", len(valid)),
        "ask_wall_blocks_json": _aggregate_wall_blocks(valid, "ask", len(valid)),
        "bid_liquidity_zones_json": _aggregate_liquidity_zone_blocks(valid, "bid", len(valid)),
        "ask_liquidity_zones_json": _aggregate_liquidity_zone_blocks(valid, "ask", len(valid)),
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

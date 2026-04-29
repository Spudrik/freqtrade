from __future__ import annotations

import json
from bisect import bisect_left
from typing import Any

from .markets import MARKET_PROFILES, MarketProfile
from .metrics import parse_book_side


def build_stream_url(profile: MarketProfile, records: list[dict[str, Any]]) -> str:
    if profile.ws_protocol == "binance_combined":
        streams = []
        for record in records:
            symbol = str(record["symbol"]).lower()
            depth = int(record.get("stream_depth") or profile.default_depth)
            update_ms = int(record.get("stream_update_ms") or profile.default_update_ms)
            suffix = f"@{update_ms}ms" if update_ms != 1000 else ""
            streams.append(f"{symbol}@depth{depth}{suffix}")
        return profile.ws_endpoint + "/".join(streams)
    return profile.ws_endpoint


def build_subscribe_message(profile: MarketProfile, records: list[dict[str, Any]]) -> str | None:
    if profile.ws_protocol != "bybit_public":
        return None
    args = [f"orderbook.{int(record.get('stream_depth') or profile.default_depth)}.{record['symbol']}" for record in records]
    return json.dumps({"op": "subscribe", "args": args})


def stream_records_by_profile(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["market_key"]), []).append(record)
    return grouped


def parse_stream_message(profile: MarketProfile, message: str) -> tuple[str | None, list[tuple[float, float]], list[tuple[float, float]], str]:
    payload = json.loads(message)
    if not isinstance(payload, dict):
        return None, [], [], ""
    if profile.ws_protocol == "binance_combined":
        stream = str(payload.get("stream") or "")
        data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        if not isinstance(data, dict):
            return None, [], [], ""
        symbol = str(data.get("s") or stream.split("@", 1)[0]).upper()
        bids_raw = data.get("b") if data.get("b") is not None else data.get("bids")
        asks_raw = data.get("a") if data.get("a") is not None else data.get("asks")
        return symbol, parse_book_side(bids_raw or [], reverse=True), parse_book_side(asks_raw or [], reverse=False), "snapshot"
    if profile.ws_protocol == "bybit_public":
        topic = str(payload.get("topic") or "")
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        if not isinstance(data, dict):
            return None, [], [], ""
        symbol = str(data.get("s") or topic.rsplit(".", 1)[-1]).upper()
        bids = parse_book_side(data.get("b") or [], reverse=True)
        asks = parse_book_side(data.get("a") or [], reverse=False)
        return symbol, bids, asks, str(payload.get("type") or "delta")
    return None, [], [], ""


def apply_book_update(
    state: dict[str, Any],
    profile_key: str,
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    update_type: str,
    depth_levels: int,
) -> None:
    profile = MARKET_PROFILES[profile_key]
    if profile.ws_protocol == "bybit_public" and update_type == "delta":
        state["bids"] = _apply_delta(state.get("bids") or [], bids, reverse=True)[:depth_levels]
        state["asks"] = _apply_delta(state.get("asks") or [], asks, reverse=False)[:depth_levels]
        return
    if bids:
        state["bids"] = bids[:depth_levels]
    if asks:
        state["asks"] = asks[:depth_levels]


def _apply_delta(
    current: list[tuple[float, float]], updates: list[tuple[float, float]], *, reverse: bool
) -> list[tuple[float, float]]:
    sizes = {price: qty for price, qty in current}
    prices = sorted(sizes)
    for price, qty in updates:
        if price in sizes:
            if qty <= 0:
                del sizes[price]
                idx = bisect_left(prices, price)
                if idx < len(prices) and prices[idx] == price:
                    del prices[idx]
            else:
                sizes[price] = qty
        elif qty > 0:
            sizes[price] = qty
            prices.insert(bisect_left(prices, price), price)
    ordered = prices[::-1] if reverse else prices
    return [(price, sizes[price]) for price in ordered]

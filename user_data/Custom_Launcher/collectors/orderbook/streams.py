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
        if not any(key in data for key in ("b", "bids")) or not any(key in data for key in ("a", "asks")):
            return None, [], [], ""
        symbol = str(data.get("s") or stream.split("@", 1)[0]).upper()
        bids_raw = data.get("b") if data.get("b") is not None else data.get("bids")
        asks_raw = data.get("a") if data.get("a") is not None else data.get("asks")
        return (
            symbol,
            parse_book_side(bids_raw, reverse=True, strict=True),
            parse_book_side(asks_raw, reverse=False, strict=True),
            "snapshot",
        )
    if profile.ws_protocol == "bybit_public":
        topic = str(payload.get("topic") or "")
        data = payload.get("data")
        update_type = str(payload.get("type") or "")
        if not topic.startswith("orderbook.") or update_type not in {"snapshot", "delta"} or not isinstance(data, dict):
            return None, [], [], ""
        update_id = data.get("u")
        is_reset_snapshot = (
            update_type == "delta"
            and isinstance(update_id, int)
            and not isinstance(update_id, bool)
            and update_id == 1
        )
        if update_type == "snapshot" or is_reset_snapshot:
            if "b" not in data or "a" not in data:
                raise ValueError("Bybit snapshot must include both bid and ask sides")
            update_type = "snapshot"
        elif not any(key in data for key in ("b", "a")):
            return None, [], [], ""
        symbol = str(data.get("s") or topic.rsplit(".", 1)[-1]).upper()
        bids = parse_book_side(data["b"], reverse=True, strict=True) if "b" in data else []
        asks = parse_book_side(data["a"], reverse=False, strict=True) if "a" in data else []
        return symbol, bids, asks, update_type
    return None, [], [], ""


def apply_book_update(
    state: dict[str, Any],
    profile_key: str,
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    update_type: str,
    depth_levels: int,
) -> bool:
    profile = MARKET_PROFILES[profile_key]
    if profile.ws_protocol == "bybit_public":
        if update_type == "snapshot":
            state["bids"] = bids[:depth_levels]
            state["asks"] = asks[:depth_levels]
            state["awaiting_snapshot"] = False
            return True
        if update_type != "delta" or state.get("awaiting_snapshot", True):
            return False
        state["bids"] = _apply_delta(state.get("bids") or [], bids, reverse=True)[:depth_levels]
        state["asks"] = _apply_delta(state.get("asks") or [], asks, reverse=False)[:depth_levels]
        return True
    if update_type != "snapshot":
        return False
    state["bids"] = bids[:depth_levels]
    state["asks"] = asks[:depth_levels]
    return True


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

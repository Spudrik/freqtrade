from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .markets import MARKET_PROFILES


def fetch_public_json(url: str, *, timeout_seconds: int = 10, user_agent: str = "FreQ-OrderBookCollector/1.0") -> Any:
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_market_context(record: dict[str, Any], *, period: str = "5m", timeout_seconds: int = 10) -> dict[str, Any] | None:
    profile = MARKET_PROFILES.get(str(record.get("market_key") or ""))
    if profile is None or not profile.context_supported:
        return None
    if profile.market_key == "binance_usdm_futures":
        return fetch_binance_usdm_context(record, period=period, timeout_seconds=timeout_seconds)
    if profile.market_key == "bybit_linear":
        return fetch_bybit_linear_context(record, period=period, timeout_seconds=timeout_seconds)
    return None


def fetch_binance_usdm_context(record: dict[str, Any], *, period: str, timeout_seconds: int) -> dict[str, Any]:
    symbol = str(record["symbol"])
    base = {
        "market_key": record["market_key"],
        "venue": record["venue"],
        "market_type": record["market_type"],
        "margin_type": record["margin_type"],
        "quote_asset": record["quote_asset"],
        "canonical_pair": record["canonical_pair"],
        "symbol": symbol,
    }
    raw: dict[str, Any] = {}
    funding = fetch_public_json(
        "https://fapi.binance.com/fapi/v1/fundingRate?" + urlencode({"symbol": symbol, "limit": 1}),
        timeout_seconds=timeout_seconds,
    )
    open_interest = fetch_public_json(
        "https://fapi.binance.com/fapi/v1/openInterest?" + urlencode({"symbol": symbol}),
        timeout_seconds=timeout_seconds,
    )
    ratio = fetch_public_json(
        "https://fapi.binance.com/futures/data/globalLongShortAccountRatio?"
        + urlencode({"symbol": symbol, "period": period, "limit": 1}),
        timeout_seconds=timeout_seconds,
    )
    taker = fetch_public_json(
        "https://fapi.binance.com/futures/data/takerlongshortRatio?"
        + urlencode({"symbol": symbol, "period": period, "limit": 1}),
        timeout_seconds=timeout_seconds,
    )
    raw.update({"funding": funding, "open_interest": open_interest, "long_short": ratio, "taker": taker})
    funding_row = funding[-1] if isinstance(funding, list) and funding else {}
    ratio_row = ratio[-1] if isinstance(ratio, list) and ratio else {}
    taker_row = taker[-1] if isinstance(taker, list) and taker else {}
    return {
        **base,
        "source_ts": _ms_to_iso(funding_row.get("fundingTime") or open_interest.get("time") or ratio_row.get("timestamp") or taker_row.get("timestamp")),
        "funding_rate": _float_or_none(funding_row.get("fundingRate")),
        "open_interest": _float_or_none(open_interest.get("openInterest")),
        "long_ratio": _float_or_none(ratio_row.get("longAccount")),
        "short_ratio": _float_or_none(ratio_row.get("shortAccount")),
        "long_short_ratio": _float_or_none(ratio_row.get("longShortRatio") or taker_row.get("buySellRatio")),
        "taker_buy_volume": _float_or_none(taker_row.get("buyVol")),
        "taker_sell_volume": _float_or_none(taker_row.get("sellVol")),
        "taker_buy_sell_ratio": _float_or_none(taker_row.get("buySellRatio")),
        "raw_json": json.dumps(raw, separators=(",", ":"), sort_keys=True),
    }


def fetch_bybit_linear_context(record: dict[str, Any], *, period: str, timeout_seconds: int) -> dict[str, Any]:
    symbol = str(record["symbol"])
    category = "linear"
    bybit_period = _bybit_period(period)
    base = {
        "market_key": record["market_key"],
        "venue": record["venue"],
        "market_type": record["market_type"],
        "margin_type": record["margin_type"],
        "quote_asset": record["quote_asset"],
        "canonical_pair": record["canonical_pair"],
        "symbol": symbol,
    }
    funding = fetch_public_json(
        "https://api.bybit.com/v5/market/funding/history?" + urlencode({"category": category, "symbol": symbol, "limit": 1}),
        timeout_seconds=timeout_seconds,
    )
    open_interest = fetch_public_json(
        "https://api.bybit.com/v5/market/open-interest?"
        + urlencode({"category": category, "symbol": symbol, "intervalTime": bybit_period, "limit": 1}),
        timeout_seconds=timeout_seconds,
    )
    ratio = fetch_public_json(
        "https://api.bybit.com/v5/market/account-ratio?"
        + urlencode({"category": category, "symbol": symbol, "period": bybit_period, "limit": 1}),
        timeout_seconds=timeout_seconds,
    )
    trades = fetch_public_json(
        "https://api.bybit.com/v5/market/recent-trade?"
        + urlencode({"category": category, "symbol": symbol, "limit": 200}),
        timeout_seconds=timeout_seconds,
    )
    funding_row = _bybit_first(funding)
    oi_row = _bybit_first(open_interest)
    ratio_row = _bybit_first(ratio)
    taker_buy, taker_sell = _bybit_taker_volumes(trades)
    long_ratio = _float_or_none(ratio_row.get("buyRatio"))
    short_ratio = _float_or_none(ratio_row.get("sellRatio"))
    return {
        **base,
        "source_ts": _ms_to_iso(funding_row.get("fundingRateTimestamp") or oi_row.get("timestamp") or ratio_row.get("timestamp")),
        "funding_rate": _float_or_none(funding_row.get("fundingRate")),
        "open_interest": _float_or_none(oi_row.get("openInterest")),
        "long_ratio": long_ratio,
        "short_ratio": short_ratio,
        "long_short_ratio": (long_ratio / short_ratio) if long_ratio is not None and short_ratio else None,
        "taker_buy_volume": taker_buy,
        "taker_sell_volume": taker_sell,
        "taker_buy_sell_ratio": (taker_buy / taker_sell) if taker_buy is not None and taker_sell else None,
        "raw_json": json.dumps(
            {"funding": funding, "open_interest": open_interest, "long_short": ratio, "recent_trades": trades},
            separators=(",", ":"),
            sort_keys=True,
        ),
    }


def _bybit_first(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    result = payload.get("result")
    if not isinstance(result, dict):
        return {}
    rows = result.get("list")
    if isinstance(rows, list) and rows:
        return rows[0] if isinstance(rows[0], dict) else {}
    return {}


def _bybit_taker_volumes(payload: Any) -> tuple[float | None, float | None]:
    if not isinstance(payload, dict):
        return None, None
    rows = payload.get("result", {}).get("list") if isinstance(payload.get("result"), dict) else []
    if not isinstance(rows, list):
        return None, None
    buy = 0.0
    sell = 0.0
    for row in rows:
        if not isinstance(row, dict):
            continue
        size = _float_or_none(row.get("size"))
        if size is None:
            continue
        if str(row.get("side") or "").lower() == "buy":
            buy += size
        elif str(row.get("side") or "").lower() == "sell":
            sell += size
    return buy, sell


def _bybit_period(period: str) -> str:
    value = str(period or "").strip()
    return {"5m": "5min", "15m": "15min", "30m": "30min"}.get(value, value or "5min")


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _ms_to_iso(value: Any) -> str | None:
    from datetime import datetime, timezone

    try:
        return datetime.fromtimestamp(float(value) / 1000.0, timezone.utc).isoformat()
    except Exception:
        return None

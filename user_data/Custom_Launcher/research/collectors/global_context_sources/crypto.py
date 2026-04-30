from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

from .common import (
    clamp,
    compact_usd,
    fetch_public_json,
    float_or_none,
    fmt_pct,
    mean,
    nested_float,
    result,
    score_label,
    score_signal,
    unix_to_iso,
)


def fetch_coingecko_markets(source: dict[str, Any], config: dict[str, Any]) -> Any:
    url = str(source.get("url") or "").strip()
    params = {
        "vs_currency": str(source.get("vs_currency") or "usd"),
        "ids": ",".join(str(value) for value in source.get("coin_ids") or ["bitcoin", "ethereum"]),
        "sparkline": "false",
        "price_change_percentage": "1h,24h,7d",
    }
    separator = "&" if "?" in url else "?"
    return fetch_public_json(
        url + separator + urlencode(params),
        timeout_seconds=int(config.get("request_timeout_seconds", 20)),
        user_agent=str(config.get("user_agent") or "FreQ-GlobalContextCollector/1.0"),
    )


def normalize_fear_greed(source: dict[str, Any], payload: Any, *, store_raw: bool = True) -> dict[str, Any]:
    rows = payload.get("data") if isinstance(payload, dict) else []
    row = rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], dict) else {}
    value = float_or_none(row.get("value"))
    classification = str(row.get("value_classification") or "")
    score = clamp(value if value is not None else 50.0)
    notes = classification or score_label(score)
    return result(
        source,
        metric_key="fear_greed_index",
        score=score,
        signal=score_signal(score),
        value=value,
        unit="index",
        notes=notes,
        source_ts=unix_to_iso(row.get("timestamp")),
        raw=payload if store_raw else None,
    )


def normalize_coingecko_global(source: dict[str, Any], payload: Any, *, store_raw: bool = True) -> dict[str, Any]:
    data = payload.get("data") if isinstance(payload, dict) else {}
    if not isinstance(data, dict):
        data = {}
    market_cap = nested_float(data, "total_market_cap", "usd")
    volume = nested_float(data, "total_volume", "usd")
    btc_dom = nested_float(data, "market_cap_percentage", "btc")
    change_24h = float_or_none(data.get("market_cap_change_percentage_24h_usd"))
    score = clamp(50.0 + (change_24h or 0.0) * 5.0)
    notes = f"cap ${compact_usd(market_cap)}, volume ${compact_usd(volume)}, BTC dom {fmt_pct(btc_dom)}, 24h cap {fmt_pct(change_24h)}"
    return result(
        source,
        metric_key="global_market_cap_change_24h",
        score=score,
        signal=score_signal(score),
        value=change_24h,
        unit="percent",
        notes=notes,
        raw=payload if store_raw else None,
    )


def normalize_coingecko_markets(source: dict[str, Any], payload: Any, *, store_raw: bool = True) -> dict[str, Any]:
    rows = payload if isinstance(payload, list) else []
    changes_24h: list[float] = []
    changes_7d: list[float] = []
    note_parts: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or row.get("id") or "").upper()
        price = float_or_none(row.get("current_price"))
        change_24h = float_or_none(row.get("price_change_percentage_24h"))
        change_7d = float_or_none(row.get("price_change_percentage_7d_in_currency"))
        if change_24h is not None:
            changes_24h.append(change_24h)
        if change_7d is not None:
            changes_7d.append(change_7d)
        if symbol:
            note_parts.append(f"{symbol} ${compact_usd(price)} 24h {fmt_pct(change_24h)} 7d {fmt_pct(change_7d)}")
    avg_24h = mean(changes_24h)
    avg_7d = mean(changes_7d)
    score = clamp(50.0 + (avg_24h or 0.0) * 4.0 + (avg_7d or 0.0) * 1.5)
    return result(
        source,
        metric_key="btc_eth_avg_change_24h",
        score=score,
        signal=score_signal(score),
        value=avg_24h,
        unit="percent",
        notes="; ".join(note_parts) or "No coin rows returned.",
        raw=payload if store_raw else None,
    )

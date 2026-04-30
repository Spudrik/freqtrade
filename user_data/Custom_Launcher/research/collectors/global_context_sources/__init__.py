from __future__ import annotations

from typing import Any

from .crypto import fetch_coingecko_markets, normalize_coingecko_global, normalize_coingecko_markets, normalize_fear_greed
from .defi import normalize_defillama_chains, normalize_defillama_stablecoins
from .equities import fetch_stooq_quotes, normalize_stooq_quotes
from .fred import fetch_fred_series_basket, normalize_fred_series_basket


def fetch_source_payload(source: dict[str, Any], config: dict[str, Any]) -> Any:
    source_type = str(source.get("type") or "").strip()
    if source_type == "coingecko_markets":
        return fetch_coingecko_markets(source, config)
    if source_type == "stooq_quotes":
        return fetch_stooq_quotes(source, config)
    if source_type == "fred_series_basket":
        return fetch_fred_series_basket(source, config)
    return None


def normalize_source_payload(source: dict[str, Any], payload: Any, *, store_raw: bool = True) -> dict[str, Any]:
    source_type = str(source.get("type") or "").strip()
    if source_type == "alternative_fng":
        return normalize_fear_greed(source, payload, store_raw=store_raw)
    if source_type == "coingecko_global":
        return normalize_coingecko_global(source, payload, store_raw=store_raw)
    if source_type == "coingecko_markets":
        return normalize_coingecko_markets(source, payload, store_raw=store_raw)
    if source_type == "defillama_stablecoins":
        return normalize_defillama_stablecoins(source, payload, store_raw=store_raw)
    if source_type == "defillama_chains":
        return normalize_defillama_chains(source, payload, store_raw=store_raw)
    if source_type == "stooq_quotes":
        return normalize_stooq_quotes(source, payload, store_raw=store_raw)
    if source_type == "fred_series_basket":
        return normalize_fred_series_basket(source, payload, store_raw=store_raw)
    raise ValueError(f"Unsupported global context source type: {source_type}")

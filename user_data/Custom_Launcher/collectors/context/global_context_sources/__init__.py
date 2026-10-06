from __future__ import annotations

from typing import Any

from .crypto import (
    fetch_coingecko_markets,
    fetch_deribit_options,
    normalize_coingecko_global,
    normalize_coingecko_markets,
    normalize_coingecko_relative_strength,
    normalize_deribit_options,
    normalize_fear_greed,
)
from .defi import (
    normalize_defillama_chains,
    normalize_defillama_dex_volume,
    normalize_defillama_stablecoin_chains,
    normalize_defillama_stablecoins,
)
from .equities import fetch_stooq_quotes, normalize_stooq_quotes
from .etf_flows import fetch_farside_btc_etf_flows, normalize_farside_btc_etf_flows
from .flows import (
    fetch_blockscout_wallet_page,
    fetch_bybit_btc_address_page,
    fetch_bybit_liquidations,
    fetch_hyperliquid_clearinghouse,
    normalize_bybit_btc_address_page,
    normalize_bybit_eth_erc20,
    normalize_bybit_eth_native,
    normalize_bybit_liquidations,
    normalize_hyperliquid_clearinghouse,
)
from .fred import fetch_fred_series_basket, normalize_fred_series_basket
from .trends import fetch_google_trends_interest, normalize_google_trends_interest


def fetch_source_payload(source: dict[str, Any], config: dict[str, Any]) -> Any:
    source_type = str(source.get("type") or "").strip()
    if source_type == "coingecko_markets":
        return fetch_coingecko_markets(source, config)
    if source_type == "coingecko_relative_strength":
        return fetch_coingecko_markets(source, config)
    if source_type == "deribit_options":
        return fetch_deribit_options(source, config)
    if source_type == "stooq_quotes":
        return fetch_stooq_quotes(source, config)
    if source_type == "fred_series_basket":
        return fetch_fred_series_basket(source, config)
    if source_type == "google_trends_interest":
        return fetch_google_trends_interest(source, config)
    if source_type == "farside_btc_etf_flows":
        return fetch_farside_btc_etf_flows(source, config)
    if source_type == "hyperliquid_clearinghouse":
        return fetch_hyperliquid_clearinghouse(source, config)
    if source_type in {"bybit_eth_native_wallet", "bybit_eth_erc20_wallet"}:
        return fetch_blockscout_wallet_page(source, config)
    if source_type == "bybit_btc_wallet":
        return fetch_bybit_btc_address_page(source, config)
    if source_type == "bybit_liquidation_observer":
        return fetch_bybit_liquidations(source, config)
    return None


def normalize_source_payload(
    source: dict[str, Any], payload: Any, *, store_raw: bool = True
) -> dict[str, Any] | list[dict[str, Any]]:
    source_type = str(source.get("type") or "").strip()
    direct_normalizers = {
        "coingecko_relative_strength": normalize_coingecko_relative_strength,
        "deribit_options": normalize_deribit_options,
        "defillama_stablecoin_chains": normalize_defillama_stablecoin_chains,
        "defillama_dex_volume": normalize_defillama_dex_volume,
        "hyperliquid_clearinghouse": normalize_hyperliquid_clearinghouse,
        "bybit_eth_native_wallet": normalize_bybit_eth_native,
        "bybit_eth_erc20_wallet": normalize_bybit_eth_erc20,
        "bybit_btc_wallet": normalize_bybit_btc_address_page,
        "bybit_liquidation_observer": normalize_bybit_liquidations,
    }
    direct_normalizer = direct_normalizers.get(source_type)
    if direct_normalizer is not None:
        return direct_normalizer(source, payload, store_raw=store_raw)
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
    if source_type == "google_trends_interest":
        return normalize_google_trends_interest(source, payload, store_raw=store_raw)
    if source_type == "farside_btc_etf_flows":
        return normalize_farside_btc_etf_flows(source, payload, store_raw=store_raw)
    raise ValueError(f"Unsupported global context source type: {source_type}")

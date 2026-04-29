from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_MARKET_PROFILE_KEYS = [
    "binance_spot",
    "binance_usdm_futures",
    "bybit_spot",
    "bybit_linear",
]


@dataclass(frozen=True)
class MarketProfile:
    market_key: str
    label: str
    venue: str
    market_type: str
    margin_type: str
    quote_asset: str
    ws_protocol: str
    ws_endpoint: str
    supported_depths: tuple[int, ...]
    supported_update_ms: tuple[int, ...]
    default_depth: int
    default_update_ms: int
    context_supported: bool = False
    bybit_category: str = ""


MARKET_PROFILES: dict[str, MarketProfile] = {
    "binance_spot": MarketProfile(
        market_key="binance_spot",
        label="Binance Spot",
        venue="binance",
        market_type="spot",
        margin_type="spot",
        quote_asset="USDT",
        ws_protocol="binance_combined",
        ws_endpoint="wss://stream.binance.com:9443/stream?streams=",
        supported_depths=(5, 10, 20),
        supported_update_ms=(100, 1000),
        default_depth=20,
        default_update_ms=100,
    ),
    "binance_usdm_futures": MarketProfile(
        market_key="binance_usdm_futures",
        label="Binance USD-M Futures",
        venue="binance",
        market_type="futures",
        margin_type="usdm",
        quote_asset="USDT",
        ws_protocol="binance_combined",
        ws_endpoint="wss://fstream.binance.com/stream?streams=",
        supported_depths=(5, 10, 20),
        supported_update_ms=(100, 250, 500),
        default_depth=20,
        default_update_ms=500,
        context_supported=True,
    ),
    "bybit_spot": MarketProfile(
        market_key="bybit_spot",
        label="Bybit Spot",
        venue="bybit",
        market_type="spot",
        margin_type="spot",
        quote_asset="USDT",
        ws_protocol="bybit_public",
        ws_endpoint="wss://stream.bybit.com/v5/public/spot",
        supported_depths=(1, 50, 200, 1000),
        supported_update_ms=(20, 100, 200),
        default_depth=50,
        default_update_ms=20,
        bybit_category="spot",
    ),
    "bybit_linear": MarketProfile(
        market_key="bybit_linear",
        label="Bybit Linear",
        venue="bybit",
        market_type="futures",
        margin_type="linear",
        quote_asset="USDT",
        ws_protocol="bybit_public",
        ws_endpoint="wss://stream.bybit.com/v5/public/linear",
        supported_depths=(1, 50, 200, 1000),
        supported_update_ms=(20, 100, 200),
        default_depth=50,
        default_update_ms=20,
        context_supported=True,
        bybit_category="linear",
    ),
}


def market_profile_options() -> list[tuple[str, str]]:
    return [(key, MARKET_PROFILES[key].label) for key in DEFAULT_MARKET_PROFILE_KEYS]


def normalize_market_profile_keys(raw: Any) -> list[str]:
    if isinstance(raw, str):
        tokens = [part.strip() for part in raw.replace(",", " ").split()]
    elif isinstance(raw, (list, tuple, set)):
        tokens = [str(part).strip() for part in raw]
    else:
        tokens = []
    selected: list[str] = []
    for token in tokens:
        if token in MARKET_PROFILES and token not in selected:
            selected.append(token)
    if not tokens:
        return list(DEFAULT_MARKET_PROFILE_KEYS)
    return selected


def canonical_pair(pair: str) -> str | None:
    raw = str(pair or "").strip().upper()
    if not raw or "/" not in raw:
        return None
    base, quote_part = raw.split("/", 1)
    quote = quote_part.split(":", 1)[0]
    base = base.strip()
    quote = quote.strip()
    if not base or not quote:
        return None
    if not base.isalnum() or not quote.isalnum():
        return None
    return f"{base}/{quote}"


def pair_to_symbol(pair: str, profile: MarketProfile) -> str | None:
    canonical = canonical_pair(pair)
    if not canonical:
        return None
    base, quote = canonical.split("/", 1)
    if quote != profile.quote_asset:
        return None
    return f"{base}{quote}"


def resolve_profile_depth(profile: MarketProfile, requested_depth: int) -> int:
    requested = int(requested_depth)
    if requested in profile.supported_depths:
        return requested
    larger = [depth for depth in profile.supported_depths if depth >= requested]
    return min(larger) if larger else profile.default_depth


def resolve_profile_update_ms(profile: MarketProfile, requested_update_ms: int) -> int:
    requested = int(requested_update_ms)
    if requested in profile.supported_update_ms:
        return requested
    return profile.default_update_ms


def normalize_whitelist_pairs(raw_pairs: list[str], max_symbols: int) -> list[dict[str, str]]:
    seen: set[str] = set()
    normalized: list[dict[str, str]] = []
    for pair in raw_pairs:
        pair_text = str(pair or "").strip()
        canonical = canonical_pair(pair_text)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        normalized.append({"pair": pair_text, "canonical_pair": canonical, "symbol": canonical.replace("/", "")})
        if len(normalized) >= max(1, int(max_symbols)):
            break
    return normalized


def normalize_pairs_for_profiles(
    raw_pairs: list[str],
    profile_keys: list[str],
    *,
    max_symbols: int,
    depth_levels: int = 20,
    stream_update_ms: int = 500,
) -> list[dict[str, Any]]:
    pairs = normalize_whitelist_pairs(raw_pairs, max_symbols=max_symbols)
    selected_profiles = [key for key in profile_keys if key in MARKET_PROFILES]
    records: list[dict[str, Any]] = []
    for pair_record in pairs:
        for profile_key in selected_profiles:
            profile = MARKET_PROFILES[profile_key]
            symbol = pair_to_symbol(pair_record["canonical_pair"], profile)
            if not symbol:
                continue
            records.append(
                {
                    "pair": pair_record["pair"],
                    "canonical_pair": pair_record["canonical_pair"],
                    "symbol": symbol,
                    "stream_id": f"{profile.market_key}:{symbol}",
                    "market_key": profile.market_key,
                    "venue": profile.venue,
                    "market_type": profile.market_type,
                    "margin_type": profile.margin_type,
                    "quote_asset": profile.quote_asset,
                    "stream_depth": resolve_profile_depth(profile, depth_levels),
                    "stream_update_ms": resolve_profile_update_ms(profile, stream_update_ms),
                }
            )
    return records

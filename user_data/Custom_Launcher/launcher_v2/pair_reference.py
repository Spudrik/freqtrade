from __future__ import annotations

from typing import Any, Iterable


PAIR_REFERENCE_GROUPS: list[dict[str, Any]] = [
    {"name": "Top Market Cap", "note": "Large-cap crypto reference list.", "pairs": ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "TRX", "AVAX", "LINK"]},
    {"name": "Top Volume", "note": "Static proxy for liquid pairs.", "pairs": ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX", "WIF", "PEPE", "SUI", "NEAR", "LTC", "BCH"]},
    {"name": "Top Volatility", "note": "Higher volatility candidates.", "pairs": ["WIF", "PEPE", "BONK", "BOME", "DOGE", "SHIB", "ORDI", "TIA", "SEI", "SUI", "INJ", "JUP", "PENDLE", "ENA", "STRK"]},
    {"name": "L1 Networks", "note": "Base-layer networks.", "pairs": ["BTC", "ETH", "SOL", "BNB", "ADA", "AVAX", "TRX", "DOT", "ATOM", "NEAR", "ICP", "APT", "SUI", "SEI", "ALGO"]},
    {"name": "DeFi", "note": "DEX, lending, yield, and governance names.", "pairs": ["UNI", "AAVE", "MKR", "LDO", "CRV", "COMP", "SNX", "SUSHI", "YFI", "1INCH", "PENDLE", "ENA", "DYDX", "GMX", "CAKE"]},
    {"name": "AI/Data", "note": "AI, data, compute, and indexing narratives.", "pairs": ["TAO", "RENDER", "RNDR", "FET", "AGIX", "OCEAN", "ARKM", "GRT", "WLD", "NMR", "PHB", "AI"]},
    {"name": "Payments", "note": "Payments, settlement, and fast-transfer networks.", "pairs": ["XRP", "XLM", "LTC", "BCH", "TRX", "DASH", "CELO", "HBAR", "IOTA", "ALGO"]},
]

SPEED_PAIR_SYMBOLS = ["BTC", "ETH", "SOL", "BNB", "ADA"]


def format_pair_symbols(symbols: Iterable[str], *, futures: bool = True) -> list[str]:
    suffix = ":USDT" if futures else ""
    return [f"{str(symbol).upper()}/USDT{suffix}" for symbol in symbols]


def pair_group_label(group: dict[str, Any]) -> str:
    return f"{len(group.get('pairs') or [])}x {group.get('name') or ''}".strip()


def pair_reference_choice_labels() -> list[str]:
    return [pair_group_label(group) for group in PAIR_REFERENCE_GROUPS]


def pair_reference_group_by_label(label: str) -> dict[str, Any] | None:
    clean = str(label or "").strip()
    aliases = {"15x High Beta": "15x Top Volatility", "High Beta": "Top Volatility"}
    clean = aliases.get(clean, clean)
    for group in PAIR_REFERENCE_GROUPS:
        if clean in {pair_group_label(group), str(group.get("name") or "")}:
            return group
    return None


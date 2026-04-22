from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from news_research_store import iter_source_config_tags

SCORING_VERSION = "rules_v1"

ASSET_RULES: dict[str, list[str]] = {
    "BTC": [r"\bBTC\b", r"\bBITCOIN\b", r"\bXBT\b"],
    "ETH": [r"\bETH\b", r"\bETHEREUM\b"],
    "SOL": [r"\bSOL\b", r"\bSOLANA\b"],
    "XRP": [r"\bXRP\b", r"\bRIPPLE\b"],
    "ADA": [r"\bADA\b", r"\bCARDANO\b"],
    "DOGE": [r"\bDOGE\b", r"\bDOGECOIN\b"],
    "BNB": [r"\bBNB\b", r"\bBINANCE COIN\b"],
    "AVAX": [r"\bAVAX\b", r"\bAVALANCHE\b"],
    "LINK": [r"\bLINK\b", r"\bCHAINLINK\b"],
    "DOT": [r"\bDOT\b", r"\bPOLKADOT\b"],
    "UNI": [r"\bUNI\b", r"\bUNISWAP\b"],
    "AAVE": [r"\bAAVE\b"],
    "LTC": [r"\bLTC\b", r"\bLITECOIN\b"],
    "BCH": [r"\bBCH\b", r"\bBITCOIN CASH\b"],
    "XMR": [r"\bXMR\b", r"\bMONERO\b"],
    "USDT": [r"\bUSDT\b", r"\bTETHER\b"],
    "USDC": [r"\bUSDC\b", r"\bUSD COIN\b"],
    "DAI": [r"\bDAI\b"],
}

STABLECOINS = {"USDT", "USDC", "DAI"}

EVENT_RULES: list[tuple[str, list[str]]] = [
    ("exchange_listing", [r"\bwill list\b", r"\blisted\b", r"\blisting\b", r"\bavailable for trading\b", r"\bnew listing\b"]),
    ("exchange_delisting", [r"\bdelist", r"\bdelisting\b", r"\bdelisted\b", r"\bremove trading pairs\b"]),
    ("stablecoin_depeg", [r"\bdepeg", r"\blost its peg\b", r"\bbreaks? (?:the )?peg\b"]),
    ("stablecoin", [r"\bstablecoin\b", r"\busdt\b", r"\busdc\b"]),
    ("hack_exploit", [r"\bhack", r"\bexploit", r"\bstolen\b", r"\bbreach\b", r"\bsecurity incident\b"]),
    ("regulation", [r"\bregulat", r"\bsec\b", r"\bcftc\b", r"\bfca\b", r"\bpolicy\b", r"\brulemaking\b"]),
    ("enforcement", [r"\blawsuit\b", r"\bcharges\b", r"\bsettlement\b", r"\benforcement\b", r"\bfine\b"]),
    ("rates", [r"\bfed\b", r"federal reserve", r"\binterest rate", r"\brate cut\b", r"\brate hike\b", r"\byields?\b"]),
    ("inflation", [r"\binflation\b", r"\bcpi\b", r"\bppi\b", r"\bconsumer prices\b"]),
    ("war_geopolitics", [r"\bwar\b", r"\bgeopolit", r"\bsanctions?\b", r"\btariffs?\b", r"\btaiwan\b"]),
    ("energy_shock", [r"\boil prices?\b", r"\bopec\b", r"\benergy crisis\b", r"\bnatural gas\b", r"\bstrait of hormuz\b"]),
    ("banking_stress", [r"\bbank run\b", r"\bbanking stress\b", r"\bliquidity crisis\b", r"\bcredit suisse\b", r"\binsolvenc"]),
    ("protocol_upgrade", [r"\bupgrade\b", r"\bhard fork\b", r"\bmainnet\b", r"\btestnet\b"]),
    ("staking", [r"\bstaking\b", r"\bstake\b", r"\bvalidator\b"]),
    ("funding", [r"\bfunding\b", r"\braise\b", r"\bseries [abc]\b"]),
    ("partnership", [r"\bpartnership\b", r"\bintegrat", r"\bcollaborat"]),
    ("outage", [r"\boutage\b", r"\bincident\b", r"\bdegraded\b", r"\bmaintenance\b"]),
    ("airdrop", [r"\bairdrop\b"]),
    ("token_unlock", [r"\btoken unlock\b", r"\bvesting\b"]),
    ("etf", [r"\betf\b"]),
    ("liquidation", [r"\bliquidation\b", r"\bliquidated\b"]),
]

SOURCE_PRIORITY_BOOSTS = {"critical": 40, "high": 25}
SOURCE_TYPE_BOOSTS = {
    "central_bank": 35,
    "regulator": 35,
    "crypto_exchange": 35,
    "crypto_media": 20,
    "aggregator": 10,
}
EVENT_BOOSTS = {
    "exchange_listing": 35,
    "exchange_delisting": 40,
    "stablecoin_depeg": 45,
    "hack_exploit": 45,
    "regulation": 35,
    "enforcement": 35,
    "rates": 30,
    "inflation": 30,
    "war_geopolitics": 25,
    "energy_shock": 25,
    "banking_stress": 35,
}
NOISE_RULES: list[tuple[str, int, list[str]]] = [
    ("price_prediction_only", -25, [r"\bprice prediction\b", r"\bwill .* hit \$", r"\bnext 100x\b"]),
    ("promo_or_education", -20, [r"\bsponsored\b", r"\bpromo(?:tion)?\b", r"\blearn\b", r"\bwhat is\b", r"\bguide\b"]),
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_tag_part(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split()).replace(" ", "_")


def _article_text(article: dict[str, Any]) -> str:
    return " ".join(
        str(article.get(key) or "")
        for key in ("title", "summary", "event_type", "impact_scope", "detected_entities_json")
    ).lower()


def _add_tag(
    tags: list[tuple[str, str, float, str]],
    seen: set[tuple[str, str, str]],
    namespace: Any,
    value: Any,
    confidence: float,
    matched_by: str,
) -> None:
    namespace_text = _clean_tag_part(namespace)
    value_text = _clean_tag_part(value)
    matched_by_text = _clean_tag_part(matched_by) or "rule"
    if not namespace_text or not value_text:
        return
    key = (namespace_text, value_text, matched_by_text)
    if key in seen:
        return
    seen.add(key)
    tags.append((namespace_text, value_text, float(confidence), matched_by_text))


def detect_assets(text: str) -> list[str]:
    found: list[str] = []
    for asset, patterns in ASSET_RULES.items():
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
            found.append(asset)
    return found


def detect_event_tags(text: str) -> list[str]:
    found: list[str] = []
    for event_tag, patterns in EVENT_RULES:
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
            found.append(event_tag)
    return found


def _semantic_source_tags(source: dict[str, Any]) -> list[tuple[str, str]]:
    source_group = _clean_tag_part(source.get("source_group"))
    topic = _clean_tag_part(source.get("topic"))
    market_relevance = _clean_tag_part(source.get("market_relevance"))
    tags: list[tuple[str, str]] = []

    if market_relevance in SOURCE_PRIORITY_BOOSTS:
        tags.append(("source_priority", market_relevance))
    if source_group == "central_banks":
        tags.append(("source_type", "central_bank"))
        tags.append(("asset_class", "tradfi"))
    if source_group == "regulators":
        tags.append(("source_type", "regulator"))
        tags.append(("market_topic", "regulation"))
    if source_group == "exchange_announcements":
        tags.append(("source_type", "crypto_exchange"))
        tags.append(("asset_class", "crypto"))
    if source_group in {"crypto", "crypto_media"}:
        tags.append(("source_type", "crypto_media"))
        tags.append(("asset_class", "crypto"))
    if source_group in {"macro_economics", "official_data_releases"}:
        tags.append(("asset_class", "tradfi"))
        tags.append(("market_topic", "macro"))
    if source_group in {"protocol_foundations", "defi_protocols", "infrastructure"}:
        tags.append(("asset_class", "crypto"))
    if topic in {"cpi", "ppi"}:
        tags.append(("market_topic", "inflation"))
    if any(token in topic for token in ("fed", "boe", "ecb", "boj", "rates", "liquidity")):
        tags.append(("market_topic", "rates"))
    if "litigation" in topic:
        tags.append(("market_topic", "enforcement"))
    if "energy" in topic:
        tags.append(("market_topic", "energy_shock"))
    if "china" in topic or "geopolitics" in topic:
        tags.append(("market_topic", "war_geopolitics"))
    return tags


def tags_from_source(source: dict[str, Any]) -> list[tuple[str, str, float, str]]:
    tags: list[tuple[str, str, float, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for namespace, value, confidence, matched_by in iter_source_config_tags(source):
        _add_tag(tags, seen, namespace, value, confidence, matched_by)
    for namespace, value in _semantic_source_tags(source):
        _add_tag(tags, seen, namespace, value, 0.9, "source_metadata_rule")
    return tags


def tags_from_article(article: dict[str, Any], source: dict[str, Any]) -> list[tuple[str, str, float, str]]:
    tags: list[tuple[str, str, float, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for namespace, value, confidence, matched_by in tags_from_source(source):
        _add_tag(tags, seen, namespace, value, confidence, matched_by)

    text = _article_text(article)
    assets = article.get("detected_assets")
    if not isinstance(assets, list):
        assets = detect_assets(text)
    for asset in assets:
        _add_tag(tags, seen, "asset", asset, 1.0, "asset_rule")
    for event_tag in detect_event_tags(text):
        _add_tag(tags, seen, "market_topic", event_tag, 1.0, "event_rule")
    for noise_tag, _penalty, patterns in NOISE_RULES:
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
            _add_tag(tags, seen, "noise", noise_tag, 0.8, "noise_rule")
    return tags


def _tag_values(tags: list[tuple[str, str, float, str]], namespace: str) -> set[str]:
    return {value for ns, value, _confidence, _matched_by in tags if ns == namespace}


def score_article(article: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    tags = tags_from_article(article, source)
    source_priorities = _tag_values(tags, "source_priority")
    source_types = _tag_values(tags, "source_type")
    event_tags = _tag_values(tags, "market_topic")
    assets = _tag_values(tags, "asset")
    noises = _tag_values(tags, "noise")
    asset_classes = _tag_values(tags, "asset_class")

    market_score = 0.0
    crypto_score = 0.0
    tradfi_score = 0.0
    macro_score = 0.0
    urgency_score = 0.0
    duplicate_penalty = 15.0 if article.get("_is_duplicate") else 0.0
    reasons: list[dict[str, Any]] = []

    def add(component: str, reason: str, points: float) -> None:
        nonlocal market_score, crypto_score, tradfi_score, macro_score, urgency_score
        if component == "crypto":
            crypto_score += points
        elif component == "tradfi":
            tradfi_score += points
        elif component == "macro":
            macro_score += points
        elif component == "urgency":
            urgency_score += points
        else:
            market_score += points
        reasons.append({"component": component, "reason": reason, "points": points})

    for priority in sorted(source_priorities):
        points = SOURCE_PRIORITY_BOOSTS.get(priority, 0)
        if points:
            add("market", f"source_priority:{priority}", points)
    for source_type in sorted(source_types):
        points = SOURCE_TYPE_BOOSTS.get(source_type, 0)
        if points:
            component = "crypto" if source_type in {"crypto_exchange", "crypto_media", "aggregator"} else "tradfi"
            add(component, f"source_type:{source_type}", points)
    for event_tag in sorted(event_tags):
        points = EVENT_BOOSTS.get(event_tag, 0)
        if points:
            component = "macro" if event_tag in {"rates", "inflation", "war_geopolitics", "energy_shock", "banking_stress"} else "market"
            add(component, f"market_topic:{event_tag}", points)
            urgency_score += min(20, points / 2)
    for asset in sorted(assets):
        if asset == "BTC":
            points = 20
        elif asset == "ETH":
            points = 15
        elif asset in STABLECOINS:
            points = 20
        else:
            points = 8
        add("crypto", f"asset:{asset}", points)
    if "crypto" in asset_classes and not assets:
        add("crypto", "asset_class:crypto", 10)
    if "tradfi" in asset_classes:
        add("tradfi", "asset_class:tradfi", 10)

    for noise_tag in sorted(noises):
        penalty = next((points for name, points, _patterns in NOISE_RULES if name == noise_tag), 0)
        if penalty:
            add("market", f"noise:{noise_tag}", penalty)
    if duplicate_penalty:
        reasons.append({"component": "duplicate", "reason": "duplicate_observation", "points": -duplicate_penalty})

    raw_final = market_score + crypto_score + tradfi_score + macro_score + urgency_score - duplicate_penalty
    final_score = max(0.0, min(100.0, raw_final))
    return {
        "market_relevance_score": max(0.0, market_score),
        "crypto_relevance_score": max(0.0, crypto_score),
        "tradfi_relevance_score": max(0.0, tradfi_score),
        "macro_relevance_score": max(0.0, macro_score),
        "urgency_score": max(0.0, urgency_score),
        "duplicate_penalty": duplicate_penalty,
        "final_priority_score": final_score,
        "scoring_version": SCORING_VERSION,
        "scoring_reasons_json": json.dumps(reasons, ensure_ascii=False, sort_keys=True),
        "updated_at": _utc_now(),
    }

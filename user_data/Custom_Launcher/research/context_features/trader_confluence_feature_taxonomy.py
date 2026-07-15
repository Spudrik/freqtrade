from __future__ import annotations


CONTEXT_TOPIC_PREFIXES = (
    "war_geopolitics",
    "sanctions_trade",
    "oil_energy",
    "banking_credit",
    "china_property_credit",
    "inflation_rates",
    "official_macro_release",
    "security_exploit",
    "stablecoin_liquidity",
    "dollar_risk_off",
    "ai_bubble_risk",
    "pandemic_health_shock",
    "recession_growth",
)


def coarse_source_family(column: str) -> str:
    lower = column.lower()
    if lower == "date":
        return "time"
    if lower in {"open", "high", "low", "close", "volume"} or lower.startswith("px_"):
        return "price"
    if lower.startswith("st_") or lower == "structure_present":
        return "structure"
    if lower.startswith(("ob_spot_", "ob_linear_", "ob_inverse_")):
        return "orderbook"
    if lower.startswith(("ctx_", "context_")):
        return "context"
    if lower.startswith("conf_"):
        return "confluence"
    if lower.startswith("future_") or lower.startswith(("hit_", "time_to_")) or lower.endswith(("_next_6h", "_next_24h")):
        return "target_label"
    if lower.endswith(("_present", "_missing", "_coverage")):
        return "quality_flag"
    return "metadata"


def source_detail(column: str) -> str:
    lower = column.lower()
    if lower == "date":
        return "time"
    if lower in {"open", "high", "low", "close", "volume"} or lower.startswith("px_"):
        return "price_ohlcv"
    if lower == "structure_present":
        return "structure_availability"
    if lower.startswith("st_"):
        return structure_detail(lower)
    if lower.startswith("ob_spot_"):
        return "orderbook_spot"
    if lower.startswith("ob_linear_"):
        return "orderbook_bybit_linear"
    if lower.startswith("ob_inverse_"):
        return "orderbook_bybit_inverse"
    if lower.startswith(("ctx_", "context_")):
        return context_detail(lower)
    if lower.startswith("conf_context_"):
        return "context_composite"
    if lower.startswith("conf_ob_"):
        return "orderbook_composite"
    if lower.startswith(("conf_volume_", "conf_near_", "conf_lvn_", "conf_structure_")):
        return "structure_composite"
    if lower.startswith("conf_"):
        return "hypothesis_confluence"
    if lower.startswith("future_") or lower.startswith(("hit_", "time_to_")) or lower.endswith(("_next_6h", "_next_24h")):
        return "target_label"
    if lower.endswith(("_present", "_missing", "_coverage")):
        return "quality_flag"
    return "metadata"


def structure_detail(lower_column: str) -> str:
    if "_vp_" in lower_column:
        return "structure_volume_profile"
    if "_tlv2_" in lower_column or "near_tlv2" in lower_column:
        return "structure_tlv2_support_resistance"
    if "_ms_" in lower_column:
        return "structure_bos_choch_market_structure"
    if "_pg2_" in lower_column:
        return "structure_pattern_geometry"
    if lower_column.startswith(("st_price_", "st_volume_", "st_range_")):
        return "structure_cached_price_volume_state"
    if lower_column.startswith("st_failed_") or lower_column.startswith(("st_breakout_", "st_breakdown_")):
        return "structure_composite_setups"
    return "structure_other"


def context_detail(lower_column: str) -> str:
    if (
        lower_column in {"context_present", "context_row_present"}
        or "available_at" in lower_column
        or lower_column.endswith("_source_future_violation")
        or lower_column.endswith("_source_age_hours")
        or lower_column.endswith("_source_min_ts")
        or lower_column.endswith("_source_max_ts")
    ):
        return "context_availability_metadata"
    if lower_column.startswith("ctx_gdelt_"):
        return "context_gdelt_events"
    if lower_column.startswith("ctx_gkg_"):
        return "context_gkg_documents"
    if lower_column.startswith("ctx_google_trends_"):
        return "context_google_trends"
    if lower_column.startswith("ctx_btc_spot_etf_"):
        return "context_btc_etf_flows"
    if lower_column.startswith(("ctx_fred_", "ctx_global_", "ctx_us_equity_", "ctx_safe_haven_", "ctx_btc_dominance", "ctx_fear_greed", "ctx_btc_eth_", "ctx_stablecoin_", "ctx_defi_")):
        return "context_global_market_macro"
    stripped = lower_column.removeprefix("ctx_")
    if "_source_count_" in stripped and not stripped.startswith(("unique_source_", "source_group_")):
        return "context_topic_severity"
    if stripped.startswith(CONTEXT_TOPIC_PREFIXES) or "intensity" in stripped or "severity" in stripped or "persistence" in stripped:
        return "context_topic_severity"
    if stripped.startswith(("article_", "news_", "web_", "unique_source_", "source_group_", "topic_", "top_topic_", "max_sources_", "btc_mentions", "eth_mentions", "macro_mentions")):
        return "context_article_source_activity"
    return "context_other"

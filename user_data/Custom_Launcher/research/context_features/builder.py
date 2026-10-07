from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import re
import sqlite3

import numpy as np
import pandas as pd


SCHEMA_VERSION = 10
RETURN_REPORT_HORIZONS = (1, 6, 24, 72, 168, 336, 720, 2160)
PRIMARY_REPORT_HORIZON = 168

BASE_FEATURE_COLUMNS = [
    "article_count_1h",
    "article_count_6h",
    "article_count_24h",
    "news_article_count_6h",
    "news_article_count_24h",
    "web_article_count_6h",
    "web_article_count_24h",
    "unique_source_count_6h",
    "unique_source_count_24h",
    "source_group_count_6h",
    "source_group_count_24h",
    "article_count_z_24h",
    "article_count_z_7d",
    "btc_mentions_6h",
    "eth_mentions_6h",
    "macro_mentions_6h",
    "regulation_mentions_24h",
    "etf_mentions_24h",
    "hack_security_mentions_24h",
    "liquidation_mentions_24h",
    "rates_inflation_mentions_24h",
    "max_sources_same_topic_6h",
    "max_sources_same_topic_24h",
    "topic_count_24h",
    "top_topic_share_6h",
    "news_volume_acceleration_6h",
    "fear_greed_value",
    "fear_greed_delta_24h",
    "btc_dominance_pct",
    "global_market_cap_change_24h",
    "btc_eth_avg_change_24h",
    "stablecoin_supply_change_1d",
    "stablecoin_supply_change_7d",
    "defi_tvl_weighted_change_7d",
    "global_equity_risk_basket_change",
    "us_equity_risk_basket_change",
    "safe_haven_proxy_change",
    "fred_rates_risk_daily_change",
    "fred_us_equity_daily_change",
    "google_trends_bitcoin",
    "google_trends_btc",
    "google_trends_crypto",
    "google_trends_bitcoin_etf",
    "google_trends_ethereum",
    "google_trends_crypto_attention_composite",
    "google_trends_recession",
    "google_trends_inflation",
    "google_trends_interest_rates",
    "google_trends_bank_crisis",
    "google_trends_war",
    "google_trends_macro_attention_composite",
    "google_trends_stock_market_crash",
    "google_trends_vix",
    "google_trends_gold_price",
    "google_trends_oil_price",
    "google_trends_bond_yields",
    "google_trends_market_stress_attention_composite",
    "google_trends_china_property_crisis",
    "google_trends_ai_bubble",
    "google_trends_systemic_risk_attention_composite",
    "btc_spot_etf_net_flow_musd",
    "btc_spot_etf_positive_fund_count",
    "btc_spot_etf_negative_fund_count",
    "btc_spot_etf_ibit_flow_musd",
    "btc_spot_etf_fbtc_flow_musd",
    "btc_spot_etf_gbtc_flow_musd",
]

EVENT_FAMILY_PATTERNS = {
    "war_geopolitics": re.compile(r"\b(war|conflict|geopolit|ukraine|russia|israel|iran|gaza|taiwan|missile|military)\w*|market_topic:war_geopolitics|market_topic:geopolitics", re.I),
    "sanctions_trade": re.compile(r"\b(sanction|tariff|trade war|export control|embargo|blacklist)\w*|market_topic:war_geopolitics", re.I),
    "oil_energy": re.compile(r"\b(oil|crude|brent|wti|opec|natural gas|lng|energy crisis|hormuz)\b|market_topic:energy_shock", re.I),
    "banking_credit": re.compile(r"\b(bank|banking|credit|liquidity crisis|bank run|deposit|lending|loan|default|insolvenc|contagion|downgrade|debt distress|restructuring)\w*|market_topic:banking_stress", re.I),
    "rates": re.compile(r"\b(fed|fomc|federal reserve|ecb|boe|boj|central bank|rate cut|rate hike|interest rate|yield|treasury|gilts?)\w*|market_topic:rates|asset_class:rates", re.I),
    "inflation": re.compile(r"\b(inflation|cpi|ppi|consumer prices|producer prices|pce|price pressure)\w*|market_topic:inflation", re.I),
    "central_bank": re.compile(r"\b(central bank|fed|federal reserve|ecb|boe|boj|bis|monetary policy|speech|minutes)\w*|source_type:central_bank|source_group:central_banks", re.I),
    "official_data_release": re.compile(r"\b(payroll|jobs report|cpi|ppi|pce|gdp|retail sales|ism|pmi|unemployment|claims)\w*|source_group:official_data_releases", re.I),
    "recession_growth": re.compile(r"\b(recession|slowdown|growth|gdp|contraction|expansion|soft landing|hard landing)\w*", re.I),
    "jobs_labor": re.compile(r"\b(jobs|payroll|employment|unemployment|labor market|labour market|wage|jobless claims)\w*", re.I),
    "liquidity_stablecoin": re.compile(r"\b(liquidity|stablecoin|usdt|usdc|tether|circle|money supply|m2|repo|balance sheet)\w*|market_topic:liquidity|market_topic:stablecoin", re.I),
    "regulation_legal": re.compile(r"\b(regulat|sec|cftc|mica|lawsuit|court|legal|compliance|enforcement|settlement)\w*|market_topic:regulation", re.I),
    "etf_institutional": re.compile(r"\b(etf|exchange-traded|blackrock|fidelity|grayscale|institutional|fund flows?)\w*", re.I),
    "security_exploit": re.compile(r"\b(hack|exploit|breach|security|stolen|drain|phishing|vulnerability|ransomware)\w*", re.I),
    "exchange_listing_delisting": re.compile(r"\b(listing|delisting|launchpool|perpetual|futures listing)\w*|market_topic:exchange_listing|market_topic:exchange_delisting", re.I),
    "crypto_native": re.compile(r"\b(bitcoin|btc|ethereum|eth|crypto|defi|protocol|stablecoin|mining|staking|layer 2|l2)\b|asset_class:crypto", re.I),
}

EVENT_FAMILY_COLUMNS = [
    column
    for family in EVENT_FAMILY_PATTERNS
    for column in (
        f"{family}_count_1h",
        f"{family}_count_6h",
        f"{family}_count_24h",
        f"{family}_source_count_24h",
        f"{family}_z_7d",
        f"{family}_acceleration_6h",
    )
]

GDELT_FEATURE_COLUMNS = [
    "gdelt_event_count_1h",
    "gdelt_event_count_6h",
    "gdelt_event_count_24h",
    "gdelt_num_articles_24h",
    "gdelt_num_mentions_24h",
    "gdelt_num_sources_24h",
    "gdelt_avg_tone_24h",
    "gdelt_goldstein_24h",
    "gdelt_conflict_event_count_24h",
    "gdelt_protest_event_count_24h",
    "gdelt_coercion_event_count_24h",
    "gdelt_sanctions_trade_url_count_24h",
    "gdelt_oil_energy_url_count_24h",
    "gdelt_banking_credit_url_count_24h",
    "gdelt_macro_url_count_24h",
    "gdelt_crypto_url_count_24h",
]

GKG_TOPIC_KEYS = [
    "crypto",
    "bitcoin",
    "ethereum",
    "stablecoin_liquidity",
    "regulation",
    "etf_institutional",
    "security_exploit",
    "macro_economic",
    "central_bank",
    "rates",
    "inflation",
    "jobs_labor",
    "recession_growth",
    "banking_credit",
    "oil_energy",
    "sanctions_trade",
    "war_geopolitics",
]

GKG_FEATURE_COLUMNS = [
    "gkg_doc_count_1h",
    "gkg_doc_count_6h",
    "gkg_doc_count_24h",
    "gkg_doc_count_z_7d",
    "gkg_file_count_1h",
    "gkg_file_completeness_1h",
    "gkg_source_count_24h",
    "gkg_word_count_24h",
    "gkg_avg_tone_24h",
    "gkg_positive_tone_24h",
    "gkg_negative_tone_24h",
    "gkg_polarity_tone_24h",
    "gkg_activity_tone_24h",
    "gkg_theme_count_24h",
    "gkg_unique_theme_count_24h",
    "gkg_top_theme_share_24h",
    *[
        column
        for topic in GKG_TOPIC_KEYS
        for column in (
            f"gkg_{topic}_doc_count_24h",
            f"gkg_{topic}_z_7d",
        )
    ],
]

DEFAULT_WEIGHTED_TOPIC_CONFIG = {
    "war_geopolitical": {
        "base_weight": 1.6,
        "pattern": r"\b(war|invasion|missile|airstrike|military|geopolit|ukraine|russia|israel|iran|gaza|taiwan|hormuz|nato|ceasefire|peace talks?|truce)\w*",
        "gdelt_proxy": ["conflict_event_count", 1.6],
    },
    "peace_talks_deescalation": {
        "base_weight": 1.25,
        "pattern": r"\b(peace talks?|ceasefire|truce|de-escalat|deescalat|negotiat|armistice|hostage deal|diplomatic talks?)\w*",
    },
    "banking_liquidity": {
        "base_weight": 1.7,
        "pattern": r"\b(bank run|bank failure|deposit|liquidity|credit stress|funding|repo|default|insolvenc|contagion|systemic|downgrade|debt distress|credit crunch)\w*",
        "gdelt_proxy": ["banking_credit_url_count", 1.7],
    },
    "china_property_credit": {
        "base_weight": 1.65,
        "pattern": r"\b(china property|chinese property|property developer|real estate crisis|evergrande|country garden|vanke|shadow banking|local government financing|lgfv|housing slump|mortgage boycott)\w*",
    },
    "inflation_rates": {
        "base_weight": 1.25,
        "pattern": r"\b(inflation|cpi|ppi|pce|rate hike|rate cut|interest rate|yield|treasury|fomc|fed|ecb|boe|boj|monetary policy)\w*",
        "gdelt_proxy": ["macro_url_count", 0.55],
    },
    "official_macro_release": {
        "base_weight": 1.35,
        "pattern": r"\b(payroll|jobs report|unemployment|jobless claims|gdp|retail sales|ism|pmi|cpi|ppi|pce|minutes|statement)\w*|source_group:official_data_releases",
        "gdelt_proxy": ["macro_url_count", 0.4],
    },
    "oil_energy_shock": {
        "base_weight": 1.35,
        "pattern": r"\b(oil|crude|brent|wti|opec|natural gas|lng|energy crisis|hormuz|supply shock|oil embargo|pipeline attack)\w*",
        "gdelt_proxy": ["oil_energy_url_count", 1.35],
    },
    "regulation_enforcement": {
        "base_weight": 1.3,
        "pattern": r"\b(sec|cftc|regulat|lawsuit|court|enforcement|settlement|ban|fine|criminal|compliance|mica|sanction)\w*",
        "gdelt_proxy": ["sanctions_trade_url_count", 1.0],
    },
    "etf_institutional_flow": {
        "base_weight": 1.1,
        "pattern": r"\b(etf|exchange-traded|blackrock|fidelity|grayscale|institutional|fund flow|inflow|outflow|allocation)\w*",
    },
    "security_exploit": {
        "base_weight": 1.45,
        "pattern": r"\b(hack|exploit|breach|security incident|stolen|drain|phishing|vulnerability|ransomware|bridge attack)\w*",
    },
    "stablecoin_liquidity": {
        "base_weight": 1.45,
        "pattern": r"\b(stablecoin|usdt|usdc|tether|circle|depeg|redemption|reserve|liquidity|money supply|m2)\w*",
        "gdelt_proxy": ["crypto_url_count", 0.25],
    },
    "liquidation_leverage": {
        "base_weight": 1.2,
        "pattern": r"\b(liquidat|leverage|open interest|funding rate|margin call|short squeeze|long squeeze|forced selling)\w*",
        "gdelt_proxy": ["crypto_url_count", 0.15],
    },
    "crypto_market_structure": {
        "base_weight": 1.0,
        "pattern": r"\b(bitcoin|btc|ethereum|eth|crypto|defi|mining|staking|halving|exchange listing|delisting|protocol|layer 2|l2)\w*",
        "gdelt_proxy": ["crypto_url_count", 0.8],
    },
    "dollar_risk_off": {
        "base_weight": 1.25,
        "pattern": r"\b(dollar|dxy|risk-off|safe haven|yen|gold|treasury|equity selloff|volatility|vix)\w*",
        "gdelt_proxy": ["macro_url_count", 0.35],
    },
    "pandemic_health_shock": {
        "base_weight": 1.8,
        "pattern": r"\b(covid|coronavirus|pandemic|lockdown|variant|public health emergency|who emergency|travel ban|quarantine)\w*",
    },
    "ai_bubble_risk": {
        "base_weight": 1.25,
        "pattern": r"\b(ai bubble|artificial intelligence bubble|chip bubble|nvidia bubble|datacenter capex|data center capex|ai crash|ai valuation|ai spending warning)\w*",
    },
}


def _load_weighted_topic_config() -> dict[str, dict[str, Any]]:
    config_path = Path(__file__).with_name("weighted_topic_taxonomy.json")
    config = DEFAULT_WEIGHTED_TOPIC_CONFIG
    if config_path.exists():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("topics"), dict):
                config = loaded["topics"]
        except Exception:
            config = DEFAULT_WEIGHTED_TOPIC_CONFIG
    compiled: dict[str, dict[str, Any]] = {}
    for topic, values in config.items():
        if not isinstance(values, dict) or not values.get("pattern"):
            continue
        compiled[str(topic)] = {
            **values,
            "base_weight": float(values.get("base_weight", 1.0)),
            "pattern": re.compile(str(values["pattern"]), re.I),
        }
    return compiled


WEIGHTED_TOPIC_PATTERNS = _load_weighted_topic_config()

WEIGHTED_TOPIC_FEATURE_COLUMNS = [
    column
    for topic in WEIGHTED_TOPIC_PATTERNS
    for column in (
        f"{topic}_intensity_1h",
        f"{topic}_intensity_6h",
        f"{topic}_intensity_24h",
        f"{topic}_confluence_6h",
        f"{topic}_confluence_24h",
        f"{topic}_persistence_hours_24h",
        f"{topic}_severity_max_24h",
        f"{topic}_official_confirmed_intensity_24h",
        f"{topic}_novelty_z_30d",
        f"{topic}_intensity_acceleration_6h",
    )
]

WEIGHTED_STACK_FEATURE_COLUMNS = [
    "weighted_context_total_intensity_6h",
    "weighted_context_total_intensity_24h",
    "systemic_risk_stack_intensity_24h",
    "macro_policy_stack_intensity_24h",
    "crypto_policy_stack_intensity_24h",
    "crypto_stress_stack_intensity_24h",
    "cross_topic_confluence_max_24h",
    "cross_topic_persistence_count_24h",
    "high_severity_topic_count_24h",
    "official_confirmed_context_intensity_24h",
]

FEATURE_COLUMNS = [
    *BASE_FEATURE_COLUMNS,
    *EVENT_FAMILY_COLUMNS,
    *WEIGHTED_TOPIC_FEATURE_COLUMNS,
    *WEIGHTED_STACK_FEATURE_COLUMNS,
    *GDELT_FEATURE_COLUMNS,
    *GKG_FEATURE_COLUMNS,
]

DEBUG_NUMERIC_COLUMNS = [
    "news_rows_24h",
    "web_rows_24h",
    "global_metrics_available",
    "missing_available_at_rows",
    "feature_history_hours",
]

GLOBAL_FEATURES = {
    "fear_greed_index": "fear_greed_value",
    "btc_dominance_pct": "btc_dominance_pct",
    "global_market_cap_change_24h": "global_market_cap_change_24h",
    "btc_eth_avg_change_24h": "btc_eth_avg_change_24h",
    "stablecoin_supply_change_1d": "stablecoin_supply_change_1d",
    "stablecoin_supply_change_7d": "stablecoin_supply_change_7d",
    "defi_tvl_weighted_change_7d": "defi_tvl_weighted_change_7d",
    "global_equity_risk_basket_change": "global_equity_risk_basket_change",
    "us_equity_risk_basket_change": "us_equity_risk_basket_change",
    "safe_haven_proxy_change": "safe_haven_proxy_change",
    "fred_rates_risk_daily_change": "fred_rates_risk_daily_change",
    "fred_us_equity_daily_change": "fred_us_equity_daily_change",
    "google_trends_bitcoin": "google_trends_bitcoin",
    "google_trends_btc": "google_trends_btc",
    "google_trends_crypto": "google_trends_crypto",
    "google_trends_bitcoin_etf": "google_trends_bitcoin_etf",
    "google_trends_ethereum": "google_trends_ethereum",
    "google_trends_crypto_attention_composite": "google_trends_crypto_attention_composite",
    "google_trends_recession": "google_trends_recession",
    "google_trends_inflation": "google_trends_inflation",
    "google_trends_interest_rates": "google_trends_interest_rates",
    "google_trends_bank_crisis": "google_trends_bank_crisis",
    "google_trends_war": "google_trends_war",
    "google_trends_macro_attention_composite": "google_trends_macro_attention_composite",
    "google_trends_stock_market_crash": "google_trends_stock_market_crash",
    "google_trends_vix": "google_trends_vix",
    "google_trends_gold_price": "google_trends_gold_price",
    "google_trends_oil_price": "google_trends_oil_price",
    "google_trends_bond_yields": "google_trends_bond_yields",
    "google_trends_market_stress_attention_composite": "google_trends_market_stress_attention_composite",
    "google_trends_china_property_crisis": "google_trends_china_property_crisis",
    "google_trends_ai_bubble": "google_trends_ai_bubble",
    "google_trends_systemic_risk_attention_composite": "google_trends_systemic_risk_attention_composite",
    "btc_spot_etf_net_flow_musd": "btc_spot_etf_net_flow_musd",
    "btc_spot_etf_positive_fund_count": "btc_spot_etf_positive_fund_count",
    "btc_spot_etf_negative_fund_count": "btc_spot_etf_negative_fund_count",
    "btc_spot_etf_ibit_flow_musd": "btc_spot_etf_ibit_flow_musd",
    "btc_spot_etf_fbtc_flow_musd": "btc_spot_etf_fbtc_flow_musd",
    "btc_spot_etf_gbtc_flow_musd": "btc_spot_etf_gbtc_flow_musd",
}

TAG_PATTERNS = {
    "btc": re.compile(r"\b(bitcoin|btc)\b", re.I),
    "eth": re.compile(r"\b(ethereum|eth)\b", re.I),
    "macro": re.compile(r"\b(macro|fed|federal reserve|fomc|inflation|cpi|ppi|jobs|employment|treasury|central bank|ecb|boe)\b", re.I),
    "regulation": re.compile(r"\b(regulat|sec|cftc|mica|lawsuit|court|legal|compliance)\w*", re.I),
    "etf": re.compile(r"\b(etf|exchange-traded)\b", re.I),
    "hack_security": re.compile(r"\b(hack|exploit|breach|security|stolen|drain|phishing)\w*", re.I),
    "liquidation": re.compile(r"\b(liquidat|open interest|leverage)\w*", re.I),
    "rates_inflation": re.compile(r"\b(rate|rates|inflation|cpi|ppi|fed|fomc|yield|treasury)\b", re.I),
    **EVENT_FAMILY_PATTERNS,
}


@dataclass(frozen=True)
class ContextFeaturePaths:
    app_dir: Path
    news_db: Path
    web_db: Path
    global_db: Path
    gdelt_db: Path
    feature_db: Path
    export_dir: Path
    report_dir: Path
    btc_ohlcv_path: Path


@dataclass
class BuildSummary:
    mode: str
    dry_run: bool
    feature_db: str
    rows_written: int = 0
    affected_hours: int = 0
    feature_column_count: int = len(FEATURE_COLUMNS)
    numeric_export_column_count: int = len(FEATURE_COLUMNS) + len(DEBUG_NUMERIC_COLUMNS)
    source_rows_found: dict[str, int] = field(default_factory=dict)
    source_rows_loaded: dict[str, int] = field(default_factory=dict)
    missing_available_at_rows: dict[str, int] = field(default_factory=dict)
    build_start: str | None = None
    build_end: str | None = None
    last_feature_hour_built: str | None = None
    last_raw_event_seen: str | None = None
    output_table: str = "context_features_1h"
    export_path: str | None = None
    report_path: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_paths(app_dir: Path | None = None) -> ContextFeaturePaths:
    resolved_app_dir = Path(app_dir) if app_dir else Path(__file__).resolve().parents[2]
    user_data_dir = resolved_app_dir.parent
    data_root = user_data_dir / "research_news_data"
    collector_root = user_data_dir / "collector_data"
    feature_root = data_root / "context_features"
    return ContextFeaturePaths(
        app_dir=resolved_app_dir,
        news_db=collector_root / "news" / "news_events.sqlite",
        web_db=collector_root / "web" / "web_events.sqlite",
        global_db=collector_root / "global_context" / "global_context.sqlite",
        gdelt_db=data_root / "gdelt" / "gdelt_context.sqlite",
        feature_db=feature_root / "context_features.sqlite",
        export_dir=feature_root / "exports",
        report_dir=feature_root / "reports",
        btc_ohlcv_path=user_data_dir / "data" / "binance" / "BTC_USDT-1h.feather",
    )


def feature_config_hash() -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "debug_numeric_columns": DEBUG_NUMERIC_COLUMNS,
        "global_features": GLOBAL_FEATURES,
        "event_families": sorted(EVENT_FAMILY_PATTERNS),
        "weighted_topics": _weighted_topic_hash_payload(),
        "gdelt_features": GDELT_FEATURE_COLUMNS,
        "gkg_features": GKG_FEATURE_COLUMNS,
        "tag_patterns": sorted(TAG_PATTERNS),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _weighted_topic_hash_payload() -> dict[str, dict[str, Any]]:
    payload = {}
    for topic, config in WEIGHTED_TOPIC_PATTERNS.items():
        payload[topic] = {
            "base_weight": config.get("base_weight"),
            "pattern": getattr(config.get("pattern"), "pattern", ""),
            "gdelt_proxy": config.get("gdelt_proxy"),
        }
    return payload


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_feature_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    feature_defs = ",\n                ".join(f"{column} REAL" for column in FEATURE_COLUMNS)
    debug_defs = ",\n                ".join(f"{column} REAL" for column in DEBUG_NUMERIC_COLUMNS)
    with sqlite3.connect(str(db_path), timeout=10.0) as conn:
        conn.executescript(
            f"""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS normalised_context_events (
                event_id TEXT PRIMARY KEY,
                source_name TEXT,
                source_family TEXT NOT NULL,
                source_group TEXT,
                asset_scope TEXT,
                title TEXT,
                summary TEXT,
                url TEXT,
                published_at TEXT,
                fetched_at TEXT,
                available_at TEXT NOT NULL,
                topic_key TEXT,
                duplicate_key TEXT,
                weighted_topic_key TEXT,
                weighted_topic_intensity REAL,
                severity_weight REAL,
                confidence_weight REAL,
                source_quality_weight REAL,
                asset_relevance_weight REAL,
                high_severity_flag REAL,
                topic_intensities_json TEXT,
                metric_key TEXT,
                metric_value REAL,
                tags_json TEXT,
                raw_db TEXT,
                raw_table TEXT,
                raw_id TEXT,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_context_events_available_at
                ON normalised_context_events(available_at);
            CREATE INDEX IF NOT EXISTS idx_context_events_family_available_at
                ON normalised_context_events(source_family, available_at);

            CREATE TABLE IF NOT EXISTS context_features_1h (
                date TEXT PRIMARY KEY,
                {feature_defs},
                {debug_defs},
                generated_at TEXT NOT NULL,
                min_source_available_at TEXT,
                max_source_available_at TEXT,
                schema_version INTEGER NOT NULL,
                feature_config_hash TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS feature_build_state (
                id INTEGER PRIMARY KEY CHECK(id = 1),
                last_successful_build_time TEXT,
                last_raw_event_seen TEXT,
                last_feature_hour_built TEXT,
                schema_version INTEGER NOT NULL,
                feature_config_hash TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        _ensure_context_feature_columns(conn)
        _ensure_normalised_event_columns(conn)


def _ensure_context_feature_columns(conn: sqlite3.Connection) -> None:
    existing = {str(row[1]) for row in conn.execute("PRAGMA table_info(context_features_1h)").fetchall()}
    expected = {column: "REAL" for column in [*FEATURE_COLUMNS, *DEBUG_NUMERIC_COLUMNS]}
    expected.update(
        {
            "generated_at": "TEXT",
            "min_source_available_at": "TEXT",
            "max_source_available_at": "TEXT",
            "schema_version": "INTEGER",
            "feature_config_hash": "TEXT",
        }
    )
    for column, column_type in expected.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE context_features_1h ADD COLUMN {column} {column_type}")


def _ensure_normalised_event_columns(conn: sqlite3.Connection) -> None:
    existing = {str(row[1]) for row in conn.execute("PRAGMA table_info(normalised_context_events)").fetchall()}
    expected = {
        "duplicate_key": "TEXT",
        "weighted_topic_key": "TEXT",
        "weighted_topic_intensity": "REAL",
        "severity_weight": "REAL",
        "confidence_weight": "REAL",
        "source_quality_weight": "REAL",
        "asset_relevance_weight": "REAL",
        "high_severity_flag": "REAL",
        "topic_intensities_json": "TEXT",
    }
    for column, column_type in expected.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE normalised_context_events ADD COLUMN {column} {column_type}")


def read_feature_status(feature_db: Path) -> dict[str, Any]:
    if not feature_db.exists():
        return {
            "feature_db": str(feature_db),
            "feature_rows": 0,
            "feature_column_count": len(FEATURE_COLUMNS),
            "numeric_export_column_count": len(FEATURE_COLUMNS) + len(DEBUG_NUMERIC_COLUMNS),
        }
    init_feature_db(feature_db)
    with sqlite3.connect(str(feature_db), timeout=10.0) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT COUNT(*) AS rows_total, MIN(date) AS min_date, MAX(date) AS max_date
            FROM context_features_1h
            """
        ).fetchone()
        state = conn.execute("SELECT * FROM feature_build_state WHERE id = 1").fetchone()
    return {
        "feature_db": str(feature_db),
        "feature_rows": int(row["rows_total"] or 0),
        "first_feature_hour": row["min_date"],
        "last_feature_hour": row["max_date"],
        "last_successful_build_time": state["last_successful_build_time"] if state else None,
        "last_raw_event_seen": state["last_raw_event_seen"] if state else None,
        "feature_column_count": len(FEATURE_COLUMNS),
        "numeric_export_column_count": len(FEATURE_COLUMNS) + len(DEBUG_NUMERIC_COLUMNS),
    }


def build_context_features(
    paths: ContextFeaturePaths,
    *,
    mode: str = "update",
    overlap_days: int = 7,
    export_parquet: bool = False,
    export_csv: bool = False,
    make_report: bool = False,
) -> BuildSummary:
    if mode not in {"dry-run", "update", "full-rebuild"}:
        raise ValueError("mode must be one of: dry-run, update, full-rebuild")
    dry_run = mode == "dry-run"
    warnings: list[str] = []
    previous_state = _read_state(paths.feature_db, initialize=not dry_run) if paths.feature_db.exists() else {}
    if not dry_run:
        init_feature_db(paths.feature_db)

    source_stats = {
        "news": _article_source_stats(paths.news_db),
        "web": _article_source_stats(paths.web_db),
        "global": _global_source_stats(paths.global_db),
        "gdelt": _gdelt_source_stats(paths.gdelt_db),
        "gkg": _gkg_source_stats(paths.gdelt_db),
    }
    source_rows_found = {name: stats["rows"] for name, stats in source_stats.items()}
    missing_available = {name: stats["missing_available_at"] for name, stats in source_stats.items()}
    for name, stats in source_stats.items():
        if stats["missing_available_at"]:
            warnings.append(f"{name} has {stats['missing_available_at']} row(s) without a safe availability timestamp; they are skipped.")
        if stats.get("warning"):
            warnings.append(str(stats["warning"]))

    max_seen = max((stats["max_available_at"] for stats in source_stats.values() if stats["max_available_at"]), default=None)
    min_seen = min((stats["min_available_at"] for stats in source_stats.values() if stats["min_available_at"]), default=None)
    summary = BuildSummary(
        mode=mode,
        dry_run=dry_run,
        feature_db=str(paths.feature_db),
        source_rows_found=source_rows_found,
        missing_available_at_rows=missing_available,
        last_raw_event_seen=max_seen,
        last_feature_hour_built=previous_state.get("last_feature_hour_built"),
        warnings=warnings,
    )
    if not min_seen or not max_seen:
        summary.warnings.append("No source rows with safe availability timestamps were found.")
        return summary

    overlap_days = max(1, int(overlap_days))
    max_window = pd.Timedelta(days=30)
    max_seen_ts = _parse_ts(max_seen)
    min_seen_ts = _parse_ts(min_seen)
    config_changed = bool(previous_state) and str(previous_state.get("feature_config_hash") or "") != feature_config_hash()
    if config_changed:
        summary.warnings.append("Feature configuration changed; the next update will rebuild the full available derived feature range.")
    if mode == "full-rebuild" or not previous_state.get("last_feature_hour_built") or config_changed:
        build_start = min_seen_ts.floor("h")
    else:
        previous_hour = _parse_ts(str(previous_state["last_feature_hour_built"]))
        build_start = max(min_seen_ts.floor("h"), (previous_hour - pd.Timedelta(days=overlap_days)).floor("h"))
    build_end = max_seen_ts.ceil("h")
    if build_end < build_start:
        summary.warnings.append("No affected feature hours after applying the build window.")
        return summary
    summary.build_start = _iso(build_start)
    summary.build_end = _iso(build_end)
    summary.affected_hours = int(((build_end - build_start) / pd.Timedelta(hours=1)) + 1)
    if dry_run:
        summary.source_rows_loaded = dict(source_rows_found)
        return summary

    history_start = (build_start - max_window).floor("h")
    article_frames = [
        _load_articles(paths.news_db, "news", history_start, build_end),
        _load_articles(paths.web_db, "web", history_start, build_end),
    ]
    articles = pd.concat([frame for frame in article_frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in article_frames) else _empty_articles()
    globals_df = _load_global_context(paths.global_db, history_start, build_end)
    gdelt_df = _load_gdelt_context(paths.gdelt_db, history_start, build_end)
    gkg_df = _load_gkg_context(paths.gdelt_db, history_start, build_end)
    source_rows_loaded = {
        "news": int((articles["source_family"] == "news").sum()) if not articles.empty else 0,
        "web": int((articles["source_family"] == "web").sum()) if not articles.empty else 0,
        "global": len(globals_df),
        "gdelt": len(gdelt_df),
        "gkg": len(gkg_df),
    }
    summary.source_rows_loaded = source_rows_loaded

    feature_frame = _build_feature_frame(articles, globals_df, gdelt_df, gkg_df, build_start, build_end, int(sum(missing_available.values())))

    generated_at = utc_now()
    _write_normalised_events(paths.feature_db, articles, globals_df, generated_at)
    _write_feature_rows(paths.feature_db, feature_frame, mode=mode, generated_at=generated_at)
    last_hour = str(feature_frame["date"].iloc[-1]) if not feature_frame.empty else None
    _write_state(paths.feature_db, max_seen, last_hour, generated_at)
    summary.rows_written = int(len(feature_frame))
    summary.last_feature_hour_built = last_hour

    if export_parquet or export_csv:
        export_path, _ = export_features(paths.feature_db, paths.export_dir, parquet=export_parquet, csv=export_csv)
        summary.export_path = str(export_path)
    if make_report:
        report_path = create_btc_return_report(paths.feature_db, paths.btc_ohlcv_path, paths.report_dir)
        summary.report_path = str(report_path)
    return summary


def export_features(feature_db: Path, export_dir: Path, *, parquet: bool = True, csv: bool = False) -> tuple[Path, int]:
    if not feature_db.exists():
        raise FileNotFoundError(feature_db)
    export_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "date",
        *FEATURE_COLUMNS,
        *DEBUG_NUMERIC_COLUMNS,
        "generated_at",
        "min_source_available_at",
        "max_source_available_at",
        "schema_version",
        "feature_config_hash",
    ]
    with sqlite3.connect(str(feature_db), timeout=10.0) as conn:
        frame = pd.read_sql_query(
            f"SELECT {', '.join(columns)} FROM context_features_1h ORDER BY date",
            conn,
        )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = export_dir / f"context_features_1h_{stamp}.parquet"
    if parquet:
        frame.to_parquet(output, index=False)
        frame.to_parquet(export_dir / "context_features_1h_latest.parquet", index=False)
    if csv:
        csv_path = export_dir / f"context_features_1h_{stamp}.csv"
        frame.to_csv(csv_path, index=False)
        if not parquet:
            output = csv_path
    return output, len(frame)


def validate_no_lookahead(feature_db: Path) -> dict[str, Any]:
    if not feature_db.exists():
        raise FileNotFoundError(feature_db)
    with sqlite3.connect(str(feature_db), timeout=10.0) as conn:
        frame = pd.read_sql_query(
            "SELECT date, max_source_available_at FROM context_features_1h ORDER BY date",
            conn,
        )
    if frame.empty:
        return {"rows_checked": 0, "violations": 0}
    dates = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    max_source = pd.to_datetime(frame["max_source_available_at"], utc=True, errors="coerce")
    violations = frame[(max_source.notna()) & (dates.notna()) & (max_source > dates)]
    return {
        "rows_checked": int(len(frame)),
        "violations": int(len(violations)),
        "first_violation_date": str(violations["date"].iloc[0]) if not violations.empty else None,
    }


def create_btc_return_report(feature_db: Path, btc_ohlcv_path: Path, report_dir: Path) -> Path:
    if not feature_db.exists():
        raise FileNotFoundError(feature_db)
    if not btc_ohlcv_path.exists():
        raise FileNotFoundError(btc_ohlcv_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(feature_db), timeout=10.0) as conn:
        features = pd.read_sql_query(
            f"SELECT date, {', '.join(FEATURE_COLUMNS)} FROM context_features_1h ORDER BY date",
            conn,
        )
    prices = pd.read_feather(btc_ohlcv_path)[["date", "close"]].copy()
    prices["date"] = pd.to_datetime(prices["date"], utc=True).dt.floor("h")
    for horizon in RETURN_REPORT_HORIZONS:
        prices[f"future_return_{horizon}h"] = prices["close"].shift(-horizon) / prices["close"] - 1.0
        prices[f"future_up_{horizon}h"] = _binary_label(
            prices[f"future_return_{horizon}h"],
            prices[f"future_return_{horizon}h"] > 0,
        )
    features["date"] = pd.to_datetime(features["date"], utc=True).dt.floor("h")
    merged = features.merge(prices, on="date", how="inner").dropna(subset=["future_return_6h"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"context_btc_return_report_{stamp}.csv"
    prediction_path = report_dir / f"context_btc_predictions_{stamp}.csv"
    family_report_path = report_dir / f"context_feature_family_report_{stamp}.csv"
    rows: list[dict[str, Any]] = []
    for feature in FEATURE_COLUMNS:
        series = pd.to_numeric(merged[feature], errors="coerce")
        for horizon in RETURN_REPORT_HORIZONS:
            label = pd.to_numeric(merged[f"future_return_{horizon}h"], errors="coerce")
            valid = pd.DataFrame({"feature": series, "label": label}).dropna()
            if len(valid) >= 5 and valid["feature"].nunique() > 1 and valid["label"].nunique() > 1:
                corr = valid["feature"].corr(valid["label"])
            else:
                corr = np.nan
            rows.append({"feature": feature, "label": f"future_return_{horizon}h", "correlation": corr})
    correlation_frame = pd.DataFrame(rows)
    correlation_frame.to_csv(report_path, index=False)
    _write_family_report(merged, correlation_frame, family_report_path)
    _write_prediction_export(merged, prediction_path)
    return report_path


def _write_family_report(frame: pd.DataFrame, correlations: pd.DataFrame, output: Path) -> None:
    feature_groups: dict[str, list[str]] = {}
    for feature in FEATURE_COLUMNS:
        feature_groups.setdefault(_feature_group(feature), []).append(feature)
    full_metrics = _ridge_metrics(frame, [feature for feature in FEATURE_COLUMNS if feature in frame])
    rows: list[dict[str, Any]] = []
    for group, features in sorted(feature_groups.items()):
        available = [feature for feature in features if feature in frame]
        primary_label = f"future_return_{PRIMARY_REPORT_HORIZON}h"
        primary_corr = correlations[(correlations["feature"].isin(available)) & (correlations["label"] == primary_label)].copy()
        primary_corr["abs_correlation"] = primary_corr["correlation"].abs()
        best = primary_corr.sort_values("abs_correlation", ascending=False).head(1)
        group_metrics = _ridge_metrics(frame, available)
        minus_metrics = _ridge_metrics(frame, [feature for feature in FEATURE_COLUMNS if feature in frame and feature not in available])
        rows.append(
            {
                "family": group,
                "feature_count": len(available),
                "active_feature_count": sum(_is_active_feature(frame[feature]) for feature in available),
                f"mean_abs_corr_{PRIMARY_REPORT_HORIZON}h": primary_corr["abs_correlation"].mean() if not primary_corr.empty else np.nan,
                f"max_abs_corr_{PRIMARY_REPORT_HORIZON}h": primary_corr["abs_correlation"].max() if not primary_corr.empty else np.nan,
                f"best_feature_{PRIMARY_REPORT_HORIZON}h": best["feature"].iloc[0] if not best.empty else "",
                f"best_feature_corr_{PRIMARY_REPORT_HORIZON}h": best["correlation"].iloc[0] if not best.empty else np.nan,
                f"group_only_return_corr_{PRIMARY_REPORT_HORIZON}h": group_metrics.get(f"return_corr_{PRIMARY_REPORT_HORIZON}h"),
                f"group_only_direction_accuracy_{PRIMARY_REPORT_HORIZON}h": group_metrics.get(f"direction_accuracy_{PRIMARY_REPORT_HORIZON}h"),
                f"full_return_corr_{PRIMARY_REPORT_HORIZON}h": full_metrics.get(f"return_corr_{PRIMARY_REPORT_HORIZON}h"),
                f"full_direction_accuracy_{PRIMARY_REPORT_HORIZON}h": full_metrics.get(f"direction_accuracy_{PRIMARY_REPORT_HORIZON}h"),
                f"minus_family_return_corr_{PRIMARY_REPORT_HORIZON}h": minus_metrics.get(f"return_corr_{PRIMARY_REPORT_HORIZON}h"),
                f"minus_family_direction_accuracy_{PRIMARY_REPORT_HORIZON}h": minus_metrics.get(f"direction_accuracy_{PRIMARY_REPORT_HORIZON}h"),
                "direction_accuracy_drop_when_removed": _metric_drop(full_metrics, minus_metrics, f"direction_accuracy_{PRIMARY_REPORT_HORIZON}h"),
                "return_corr_drop_when_removed": _metric_drop(full_metrics, minus_metrics, f"return_corr_{PRIMARY_REPORT_HORIZON}h"),
            }
        )
    pd.DataFrame(rows).to_csv(output, index=False)


def _feature_group(feature: str) -> str:
    if feature.startswith("gkg_"):
        return "gdelt_gkg"
    if feature.startswith("gdelt_"):
        return "gdelt_events"
    for family in EVENT_FAMILY_PATTERNS:
        if feature.startswith(f"{family}_"):
            return family
    if feature in set(GLOBAL_FEATURES.values()) | {"fear_greed_delta_24h"}:
        return "global_market_context"
    if feature.endswith("_mentions_6h") or feature.endswith("_mentions_24h"):
        return "legacy_text_tags"
    if "topic" in feature or "confluence" in feature or "same_topic" in feature:
        return "confluence"
    if "source" in feature or "article_count" in feature or "news_volume" in feature or feature.endswith("_article_count_6h") or feature.endswith("_article_count_24h"):
        return "source_activity"
    return "price_volume_context"


def _is_active_feature(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return bool(len(values) >= 5 and values.nunique() > 1)


def _training_ready_features(frame: pd.DataFrame, features: list[str]) -> list[str]:
    return [
        feature
        for feature in features
        if feature in frame and pd.to_numeric(frame[feature], errors="coerce").notna().any()
    ]


def _ridge_metrics(frame: pd.DataFrame, features: list[str]) -> dict[str, float]:
    usable_features = [feature for feature in features if feature in frame and _is_active_feature(frame[feature])]
    target = f"future_return_{PRIMARY_REPORT_HORIZON}h"
    usable = frame.dropna(subset=[target]).copy()
    if len(usable) < 30 or not usable_features:
        return {}
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return {}
    split = max(20, int(len(usable) * 0.7))
    train = usable.iloc[:split]
    test = usable.iloc[split:]
    if test.empty:
        return {}
    usable_features = _training_ready_features(train, usable_features)
    if not usable_features:
        return {}
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    model.fit(train[usable_features], train[target])
    predicted = pd.Series(model.predict(test[usable_features]), index=test.index)
    actual = pd.to_numeric(test[target], errors="coerce")
    valid = pd.DataFrame({"predicted": predicted, "actual": actual}).dropna()
    if len(valid) < 5:
        return {}
    direction_accuracy = ((valid["predicted"] > 0).astype(float) == (valid["actual"] > 0).astype(float)).mean()
    return_corr = valid["predicted"].corr(valid["actual"]) if valid["predicted"].nunique() > 1 and valid["actual"].nunique() > 1 else np.nan
    return {
        f"direction_accuracy_{PRIMARY_REPORT_HORIZON}h": float(direction_accuracy),
        f"return_corr_{PRIMARY_REPORT_HORIZON}h": float(return_corr) if pd.notna(return_corr) else np.nan,
    }


def _metric_drop(full_metrics: dict[str, float], minus_metrics: dict[str, float], key: str) -> float:
    full = full_metrics.get(key)
    minus = minus_metrics.get(key)
    if full is None or minus is None or pd.isna(full) or pd.isna(minus):
        return np.nan
    return float(full - minus)


def _binary_label(source: pd.Series, condition: pd.Series) -> pd.Series:
    return condition.astype(float).where(source.notna())


def _write_prediction_export(frame: pd.DataFrame, output: Path) -> None:
    target = f"future_return_{PRIMARY_REPORT_HORIZON}h"
    up_target = f"future_up_{PRIMARY_REPORT_HORIZON}h"
    fallback_columns = ["date", target, up_target]
    usable = frame.dropna(subset=[target]).copy()
    feature_cols = [col for col in FEATURE_COLUMNS if col in usable.columns and _is_active_feature(usable[col])]
    if len(usable) < 30:
        usable[[column for column in fallback_columns if column in usable]].to_csv(output, index=False)
        return
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        usable[[column for column in fallback_columns if column in usable]].to_csv(output, index=False)
        return
    split = max(20, int(len(usable) * 0.7))
    train = usable.iloc[:split]
    test = usable.iloc[split:]
    if test.empty:
        usable[[column for column in fallback_columns if column in usable]].to_csv(output, index=False)
        return
    feature_cols = _training_ready_features(train, feature_cols)
    if not feature_cols:
        usable[[column for column in fallback_columns if column in usable]].to_csv(output, index=False)
        return
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    model.fit(train[feature_cols], train[target])
    predicted = model.predict(test[feature_cols])
    result = test[fallback_columns].copy()
    result[f"predicted_return_{PRIMARY_REPORT_HORIZON}h"] = predicted
    result[f"predicted_up_{PRIMARY_REPORT_HORIZON}h"] = (result[f"predicted_return_{PRIMARY_REPORT_HORIZON}h"] > 0).astype(float)
    result[f"direction_correct_{PRIMARY_REPORT_HORIZON}h"] = (
        result[f"predicted_up_{PRIMARY_REPORT_HORIZON}h"] == result[up_target]
    ).astype(float)
    result.to_csv(output, index=False)


def _build_feature_frame(
    articles: pd.DataFrame,
    globals_df: pd.DataFrame,
    gdelt_df: pd.DataFrame,
    gkg_df: pd.DataFrame,
    build_start: pd.Timestamp,
    build_end: pd.Timestamp,
    missing_available_total: int,
) -> pd.DataFrame:
    hours = pd.date_range(build_start, build_end, freq="h", tz="UTC")
    output = pd.DataFrame({"date": [_iso(hour) for hour in hours]})
    defaults = pd.DataFrame(
        0.0,
        index=output.index,
        columns=[*FEATURE_COLUMNS, *DEBUG_NUMERIC_COLUMNS],
    )
    output = pd.concat([output, defaults], axis=1)
    source_min: list[str | None] = []
    source_max: list[str | None] = []
    article_count_1h: list[float] = []
    family_count_1h: dict[str, list[float]] = {family: [] for family in EVENT_FAMILY_PATTERNS}
    weighted_topic_1h: dict[str, list[float]] = {topic: [] for topic in WEIGHTED_TOPIC_PATTERNS}

    articles = articles.sort_values("available_at").reset_index(drop=True)
    for index, hour in enumerate(hours):
        w1 = _window(articles, hour, 1)
        w6 = _window(articles, hour, 6)
        w24 = _window(articles, hour, 24)
        prev6 = _window_between(articles, hour - pd.Timedelta(hours=12), hour - pd.Timedelta(hours=6))
        output.at[index, "article_count_1h"] = float(len(w1))
        output.at[index, "article_count_6h"] = float(len(w6))
        output.at[index, "article_count_24h"] = float(len(w24))
        output.at[index, "news_article_count_6h"] = float((w6["source_family"] == "news").sum()) if not w6.empty else 0.0
        output.at[index, "news_article_count_24h"] = float((w24["source_family"] == "news").sum()) if not w24.empty else 0.0
        output.at[index, "web_article_count_6h"] = float((w6["source_family"] == "web").sum()) if not w6.empty else 0.0
        output.at[index, "web_article_count_24h"] = float((w24["source_family"] == "web").sum()) if not w24.empty else 0.0
        output.at[index, "unique_source_count_6h"] = float(w6["source_name"].nunique()) if not w6.empty else 0.0
        output.at[index, "unique_source_count_24h"] = float(w24["source_name"].nunique()) if not w24.empty else 0.0
        output.at[index, "source_group_count_6h"] = float(w6["source_group"].nunique()) if not w6.empty else 0.0
        output.at[index, "source_group_count_24h"] = float(w24["source_group"].nunique()) if not w24.empty else 0.0
        output.at[index, "btc_mentions_6h"] = float(w6["has_btc"].sum()) if not w6.empty else 0.0
        output.at[index, "eth_mentions_6h"] = float(w6["has_eth"].sum()) if not w6.empty else 0.0
        output.at[index, "macro_mentions_6h"] = float(w6["has_macro"].sum()) if not w6.empty else 0.0
        output.at[index, "regulation_mentions_24h"] = float(w24["has_regulation"].sum()) if not w24.empty else 0.0
        output.at[index, "etf_mentions_24h"] = float(w24["has_etf"].sum()) if not w24.empty else 0.0
        output.at[index, "hack_security_mentions_24h"] = float(w24["has_hack_security"].sum()) if not w24.empty else 0.0
        output.at[index, "liquidation_mentions_24h"] = float(w24["has_liquidation"].sum()) if not w24.empty else 0.0
        output.at[index, "rates_inflation_mentions_24h"] = float(w24["has_rates_inflation"].sum()) if not w24.empty else 0.0
        output.at[index, "max_sources_same_topic_6h"] = _max_sources_same_topic(w6)
        output.at[index, "max_sources_same_topic_24h"] = _max_sources_same_topic(w24)
        output.at[index, "topic_count_24h"] = float(w24["topic_key"].nunique()) if not w24.empty else 0.0
        output.at[index, "top_topic_share_6h"] = _top_topic_share(w6)
        output.at[index, "news_volume_acceleration_6h"] = float(len(w6) - len(prev6))
        for family in EVENT_FAMILY_PATTERNS:
            flag = f"has_{family}"
            count_1h = _flag_sum(w1, flag)
            count_6h = _flag_sum(w6, flag)
            count_24h = _flag_sum(w24, flag)
            prev_6h = _flag_sum(prev6, flag)
            output.at[index, f"{family}_count_1h"] = count_1h
            output.at[index, f"{family}_count_6h"] = count_6h
            output.at[index, f"{family}_count_24h"] = count_24h
            output.at[index, f"{family}_source_count_24h"] = _flag_source_count(w24, flag)
            output.at[index, f"{family}_acceleration_6h"] = count_6h - prev_6h
            family_count_1h[family].append(count_1h)
        for topic in WEIGHTED_TOPIC_PATTERNS:
            intensity_column = f"{topic}_event_intensity"
            count_1h = _sum_numeric(w1, intensity_column)
            count_6h = _sum_numeric(w6, intensity_column)
            count_24h = _sum_numeric(w24, intensity_column)
            prev_6h = _sum_numeric(prev6, intensity_column)
            output.at[index, f"{topic}_intensity_1h"] = count_1h
            output.at[index, f"{topic}_intensity_6h"] = count_6h
            output.at[index, f"{topic}_intensity_24h"] = count_24h
            output.at[index, f"{topic}_confluence_6h"] = _weighted_topic_confluence(w6, topic)
            output.at[index, f"{topic}_confluence_24h"] = _weighted_topic_confluence(w24, topic)
            output.at[index, f"{topic}_persistence_hours_24h"] = _weighted_topic_persistence(w24, topic)
            output.at[index, f"{topic}_severity_max_24h"] = _weighted_topic_severity_max(w24, topic)
            output.at[index, f"{topic}_official_confirmed_intensity_24h"] = _weighted_topic_official_intensity(w24, topic)
            output.at[index, f"{topic}_intensity_acceleration_6h"] = count_6h - prev_6h
            weighted_topic_1h[topic].append(count_1h)
        output.at[index, "news_rows_24h"] = float((w24["source_family"] == "news").sum()) if not w24.empty else 0.0
        output.at[index, "web_rows_24h"] = float((w24["source_family"] == "web").sum()) if not w24.empty else 0.0
        output.at[index, "missing_available_at_rows"] = float(missing_available_total)
        output.at[index, "feature_history_hours"] = float(max(0.0, (hour - hours[0]) / pd.Timedelta(hours=1)))
        article_count_1h.append(float(len(w1)))
        contributing = list(w24["available_at"]) if not w24.empty else []
        source_min.append(_iso(min(contributing)) if contributing else None)
        source_max.append(_iso(max(contributing)) if contributing else None)

    counts = pd.Series(article_count_1h)
    output["article_count_z_24h"] = _zscore(counts, 24, 6)
    output["article_count_z_7d"] = _zscore(counts, 168, 24)
    for family, values in family_count_1h.items():
        output[f"{family}_z_7d"] = _zscore(pd.Series(values), 168, 24)
    for topic, values in weighted_topic_1h.items():
        output[f"{topic}_novelty_z_30d"] = _zscore(pd.Series(values), 720, 72)
    _add_global_features(output, globals_df, hours, source_min, source_max)
    _add_gdelt_features(output, gdelt_df, hours, source_min, source_max)
    _add_gkg_features(output, gkg_df, hours, source_min, source_max)
    _add_weighted_context_stacks(output)
    output = output.copy()
    output["min_source_available_at"] = source_min
    output["max_source_available_at"] = source_max
    return output


def _add_global_features(output: pd.DataFrame, globals_df: pd.DataFrame, hours: pd.DatetimeIndex, source_min: list[str | None], source_max: list[str | None]) -> None:
    if globals_df.empty:
        return
    hours_frame = pd.DataFrame({"date_ts": hours})
    available_counts = np.zeros(len(output), dtype=float)
    latest_available: list[pd.Timestamp | None] = [None] * len(output)
    for metric_key, column in GLOBAL_FEATURES.items():
        metric = globals_df[globals_df["metric_key"] == metric_key][["available_at", "metric_value"]].dropna(subset=["available_at"]).sort_values("available_at")
        if metric.empty:
            output[column] = np.nan
            continue
        merged = pd.merge_asof(hours_frame, metric, left_on="date_ts", right_on="available_at", direction="backward")
        output[column] = pd.to_numeric(merged["metric_value"], errors="coerce")
        present = merged["metric_value"].notna()
        available_counts += present.astype(float).to_numpy()
        for idx, value in enumerate(merged["available_at"]):
            if pd.notna(value):
                ts = pd.Timestamp(value)
                latest_available[idx] = ts if latest_available[idx] is None else max(latest_available[idx], ts)
    output["global_metrics_available"] = available_counts
    if "fear_greed_value" in output:
        output["fear_greed_delta_24h"] = output["fear_greed_value"] - output["fear_greed_value"].shift(24)
    for idx, ts in enumerate(latest_available):
        if ts is None:
            continue
        current_max = _parse_optional(source_max[idx])
        source_max[idx] = _iso(max(current_max, ts) if current_max is not None else ts)
        current_min = _parse_optional(source_min[idx])
        source_min[idx] = _iso(min(current_min, ts) if current_min is not None else ts)


def _add_gdelt_features(output: pd.DataFrame, gdelt_df: pd.DataFrame, hours: pd.DatetimeIndex, source_min: list[str | None], source_max: list[str | None]) -> None:
    if gdelt_df.empty:
        return
    hourly = gdelt_df.copy()
    hourly["event_time"] = pd.to_datetime(hourly["date"], utc=True, errors="coerce")
    hourly["available_hour"] = hourly["event_time"].dt.ceil("h")
    hourly = hourly.dropna(subset=["event_time", "available_hour"])
    if hourly.empty:
        return
    numeric_columns = [
        "event_count",
        "num_articles_sum",
        "num_mentions_sum",
        "num_sources_sum",
        "avg_tone_weighted",
        "goldstein_weighted",
        "conflict_event_count",
        "protest_event_count",
        "coercion_event_count",
        "sanctions_trade_url_count",
        "oil_energy_url_count",
        "banking_credit_url_count",
        "macro_url_count",
        "crypto_url_count",
    ]
    for column in numeric_columns:
        if column not in hourly:
            hourly[column] = 0.0
        hourly[column] = pd.to_numeric(hourly[column], errors="coerce").fillna(0.0)
    grouped = hourly.groupby("available_hour", as_index=True)[numeric_columns].sum().reindex(hours, fill_value=0.0)
    output["gdelt_event_count_1h"] = grouped["event_count"].to_numpy()
    output["gdelt_event_count_6h"] = grouped["event_count"].rolling(6, min_periods=1).sum().to_numpy()
    output["gdelt_event_count_24h"] = grouped["event_count"].rolling(24, min_periods=1).sum().to_numpy()
    output["gdelt_num_articles_24h"] = grouped["num_articles_sum"].rolling(24, min_periods=1).sum().to_numpy()
    output["gdelt_num_mentions_24h"] = grouped["num_mentions_sum"].rolling(24, min_periods=1).sum().to_numpy()
    output["gdelt_num_sources_24h"] = grouped["num_sources_sum"].rolling(24, min_periods=1).sum().to_numpy()
    output["gdelt_conflict_event_count_24h"] = grouped["conflict_event_count"].rolling(24, min_periods=1).sum().to_numpy()
    output["gdelt_protest_event_count_24h"] = grouped["protest_event_count"].rolling(24, min_periods=1).sum().to_numpy()
    output["gdelt_coercion_event_count_24h"] = grouped["coercion_event_count"].rolling(24, min_periods=1).sum().to_numpy()
    output["gdelt_sanctions_trade_url_count_24h"] = grouped["sanctions_trade_url_count"].rolling(24, min_periods=1).sum().to_numpy()
    output["gdelt_oil_energy_url_count_24h"] = grouped["oil_energy_url_count"].rolling(24, min_periods=1).sum().to_numpy()
    output["gdelt_banking_credit_url_count_24h"] = grouped["banking_credit_url_count"].rolling(24, min_periods=1).sum().to_numpy()
    output["gdelt_macro_url_count_24h"] = grouped["macro_url_count"].rolling(24, min_periods=1).sum().to_numpy()
    output["gdelt_crypto_url_count_24h"] = grouped["crypto_url_count"].rolling(24, min_periods=1).sum().to_numpy()
    output["gdelt_avg_tone_24h"] = grouped["avg_tone_weighted"].rolling(24, min_periods=1).mean().to_numpy()
    output["gdelt_goldstein_24h"] = grouped["goldstein_weighted"].rolling(24, min_periods=1).mean().to_numpy()
    _add_gdelt_weighted_topics(output, grouped)
    active_hours = set(grouped[grouped["event_count"] > 0].index)
    source_bounds = hourly.groupby("available_hour")["event_time"].agg(["min", "max"])
    for idx, hour in enumerate(hours):
        if hour not in active_hours:
            continue
        bounds = source_bounds.loc[hour]
        current_max = _parse_optional(source_max[idx])
        source_max[idx] = _iso(max(current_max, bounds["max"]) if current_max is not None else bounds["max"])
        current_min = _parse_optional(source_min[idx])
        source_min[idx] = _iso(min(current_min, bounds["min"]) if current_min is not None else bounds["min"])


def _add_gdelt_weighted_topics(output: pd.DataFrame, grouped: pd.DataFrame) -> None:
    source_count_24h = grouped["num_sources_sum"].rolling(24, min_periods=1).sum()
    source_diversity = np.log1p(source_count_24h).replace(0, np.nan).fillna(1.0)
    for topic, config in WEIGHTED_TOPIC_PATTERNS.items():
        proxy = config.get("gdelt_proxy")
        if not proxy or len(proxy) != 2:
            continue
        source_column = str(proxy[0])
        weight = float(proxy[1])
        if source_column not in grouped:
            continue
        intensity_1h = np.log1p(grouped[source_column].clip(lower=0.0)) * weight
        intensity_6h = intensity_1h.rolling(6, min_periods=1).sum()
        intensity_24h = intensity_1h.rolling(24, min_periods=1).sum()
        output[f"{topic}_intensity_1h"] += intensity_1h.to_numpy()
        output[f"{topic}_intensity_6h"] += intensity_6h.to_numpy()
        output[f"{topic}_intensity_24h"] += intensity_24h.to_numpy()
        output[f"{topic}_confluence_6h"] += (intensity_6h * source_diversity).to_numpy()
        output[f"{topic}_confluence_24h"] += (intensity_24h * source_diversity).to_numpy()
        output[f"{topic}_persistence_hours_24h"] += (intensity_1h > 0).astype(float).rolling(24, min_periods=1).sum().to_numpy()
        output[f"{topic}_severity_max_24h"] = np.maximum(
            output[f"{topic}_severity_max_24h"].to_numpy(),
            np.where(intensity_24h.to_numpy() > 0, min(1.5, 1.0 + float(weight) / 4.0), 0.0),
        )
        output[f"{topic}_intensity_acceleration_6h"] += (
            intensity_6h - intensity_1h.shift(6).rolling(6, min_periods=1).sum().fillna(0.0)
        ).to_numpy()
        output[f"{topic}_novelty_z_30d"] = _zscore(pd.Series(output[f"{topic}_intensity_1h"]), 720, 72).to_numpy()


def _add_gkg_features(output: pd.DataFrame, gkg_df: pd.DataFrame, hours: pd.DatetimeIndex, source_min: list[str | None], source_max: list[str | None]) -> None:
    if gkg_df.empty:
        return
    gkg = gkg_df.copy()
    gkg["available_hour"] = pd.to_datetime(gkg["date"], utc=True, errors="coerce").dt.ceil("h")
    gkg = gkg.dropna(subset=["available_hour"])
    if gkg.empty:
        return
    numeric_columns = [
        "document_count",
        "source_count",
        "word_count_sum",
        "tone_weight_sum",
        "tone_sum",
        "positive_tone_sum",
        "negative_tone_sum",
        "polarity_sum",
        "activity_sum",
        "theme_count_sum",
        "unique_theme_count",
        "top_theme_doc_count",
        *[f"{topic}_doc_count" for topic in GKG_TOPIC_KEYS],
    ]
    for column in numeric_columns:
        if column not in gkg:
            gkg[column] = 0.0
        gkg[column] = pd.to_numeric(gkg[column], errors="coerce").fillna(0.0)
    gkg["_file_count"] = 1.0
    grouped = gkg.groupby("available_hour", as_index=True)[[*numeric_columns, "_file_count"]].sum().reindex(hours, fill_value=0.0)
    doc_1h = grouped["document_count"]
    doc_24h = doc_1h.rolling(24, min_periods=1).sum()
    tone_weight_24h = grouped["tone_weight_sum"].rolling(24, min_periods=1).sum().replace(0, np.nan)
    output["gkg_doc_count_1h"] = doc_1h.to_numpy()
    output["gkg_doc_count_6h"] = doc_1h.rolling(6, min_periods=1).sum().to_numpy()
    output["gkg_doc_count_24h"] = doc_24h.to_numpy()
    output["gkg_doc_count_z_7d"] = _zscore(doc_24h, 168, 24).to_numpy()
    output["gkg_file_count_1h"] = grouped["_file_count"].to_numpy()
    output["gkg_file_completeness_1h"] = np.minimum(grouped["_file_count"] / 4.0, 1.0).to_numpy()
    output["gkg_source_count_24h"] = grouped["source_count"].rolling(24, min_periods=1).sum().to_numpy()
    output["gkg_word_count_24h"] = grouped["word_count_sum"].rolling(24, min_periods=1).sum().to_numpy()
    output["gkg_avg_tone_24h"] = (grouped["tone_sum"].rolling(24, min_periods=1).sum() / tone_weight_24h).fillna(0.0).to_numpy()
    output["gkg_positive_tone_24h"] = (grouped["positive_tone_sum"].rolling(24, min_periods=1).sum() / tone_weight_24h).fillna(0.0).to_numpy()
    output["gkg_negative_tone_24h"] = (grouped["negative_tone_sum"].rolling(24, min_periods=1).sum() / tone_weight_24h).fillna(0.0).to_numpy()
    output["gkg_polarity_tone_24h"] = (grouped["polarity_sum"].rolling(24, min_periods=1).sum() / tone_weight_24h).fillna(0.0).to_numpy()
    output["gkg_activity_tone_24h"] = (grouped["activity_sum"].rolling(24, min_periods=1).sum() / tone_weight_24h).fillna(0.0).to_numpy()
    output["gkg_theme_count_24h"] = grouped["theme_count_sum"].rolling(24, min_periods=1).sum().to_numpy()
    output["gkg_unique_theme_count_24h"] = grouped["unique_theme_count"].rolling(24, min_periods=1).sum().to_numpy()
    output["gkg_top_theme_share_24h"] = (grouped["top_theme_doc_count"].rolling(24, min_periods=1).sum() / doc_24h.replace(0, np.nan)).fillna(0.0).to_numpy()
    for topic in GKG_TOPIC_KEYS:
        column = f"{topic}_doc_count"
        series_24h = grouped[column].rolling(24, min_periods=1).sum()
        output[f"gkg_{topic}_doc_count_24h"] = series_24h.to_numpy()
        output[f"gkg_{topic}_z_7d"] = _zscore(series_24h, 168, 24).to_numpy()
    active_hours = set(grouped[grouped["_file_count"] > 0].index)
    for idx, hour in enumerate(hours):
        if hour not in active_hours:
            continue
        current_max = _parse_optional(source_max[idx])
        source_max[idx] = _iso(max(current_max, hour) if current_max is not None else hour)
        current_min = _parse_optional(source_min[idx])
        source_min[idx] = _iso(min(current_min, hour) if current_min is not None else hour)


def _zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std()
    return ((series - mean) / std.replace(0, np.nan)).fillna(0.0)


def _window(frame: pd.DataFrame, hour: pd.Timestamp, hours: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    start = hour - pd.Timedelta(hours=hours)
    mask = (frame["available_at"] > start) & (frame["available_at"] <= hour)
    return frame.loc[mask]


def _window_between(frame: pd.DataFrame, earlier: pd.Timestamp, later: pd.Timestamp) -> pd.DataFrame:
    if frame.empty:
        return frame
    if later < earlier:
        raise ValueError("later must be greater than or equal to earlier")
    mask = (frame["available_at"] > earlier) & (frame["available_at"] <= later)
    return frame.loc[mask]


def _max_sources_same_topic(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    grouped = frame.groupby("topic_key")["source_name"].nunique()
    return float(grouped.max()) if not grouped.empty else 0.0


def _top_topic_share(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    counts = frame["topic_key"].value_counts()
    return float(counts.iloc[0] / len(frame)) if not counts.empty else 0.0


def _flag_sum(frame: pd.DataFrame, flag: str) -> float:
    if frame.empty or flag not in frame:
        return 0.0
    return float(pd.to_numeric(frame[flag], errors="coerce").fillna(0.0).sum())


def _flag_source_count(frame: pd.DataFrame, flag: str) -> float:
    if frame.empty or flag not in frame:
        return 0.0
    flagged = frame[pd.to_numeric(frame[flag], errors="coerce").fillna(0.0) > 0]
    return float(flagged["source_name"].nunique()) if not flagged.empty else 0.0


def _sum_numeric(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def _weighted_topic_frame(frame: pd.DataFrame, topic: str) -> pd.DataFrame:
    column = f"{topic}_event_intensity"
    if frame.empty or column not in frame:
        return frame.iloc[0:0]
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame.loc[values > 0].copy()


def _weighted_topic_confluence(frame: pd.DataFrame, topic: str) -> float:
    topic_frame = _weighted_topic_frame(frame, topic)
    if topic_frame.empty:
        return 0.0
    intensity = _sum_numeric(topic_frame, f"{topic}_event_intensity")
    source_count = max(1, topic_frame["source_name"].nunique())
    group_count = max(1, topic_frame["source_group"].nunique())
    duplicate_count = max(1, topic_frame["duplicate_key"].nunique()) if "duplicate_key" in topic_frame else len(topic_frame)
    diversity = np.log1p(source_count) * np.log1p(group_count)
    duplicate_penalty = min(1.0, duplicate_count / max(1, len(topic_frame)))
    return float(intensity * diversity * duplicate_penalty)


def _weighted_topic_persistence(frame: pd.DataFrame, topic: str) -> float:
    topic_frame = _weighted_topic_frame(frame, topic)
    if topic_frame.empty:
        return 0.0
    return float(topic_frame["available_at"].dt.floor("h").nunique())


def _weighted_topic_severity_max(frame: pd.DataFrame, topic: str) -> float:
    topic_frame = _weighted_topic_frame(frame, topic)
    if topic_frame.empty or "severity_weight" not in topic_frame:
        return 0.0
    return float(pd.to_numeric(topic_frame["severity_weight"], errors="coerce").fillna(0.0).max())


def _weighted_topic_official_intensity(frame: pd.DataFrame, topic: str) -> float:
    topic_frame = _weighted_topic_frame(frame, topic)
    if topic_frame.empty:
        return 0.0
    source_quality = _numeric_series(topic_frame, "source_quality_weight")
    confidence = _numeric_series(topic_frame, "confidence_weight")
    official = topic_frame[
        (source_quality >= 1.25)
        | (confidence >= 1.15)
    ]
    return _sum_numeric(official, f"{topic}_event_intensity")


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _add_weighted_context_stacks(output: pd.DataFrame) -> None:
    zero = pd.Series(0.0, index=output.index)
    topic_24h = {topic: output.get(f"{topic}_intensity_24h", zero) for topic in WEIGHTED_TOPIC_PATTERNS}
    topic_6h = {topic: output.get(f"{topic}_intensity_6h", zero) for topic in WEIGHTED_TOPIC_PATTERNS}
    stack = lambda keys: sum((topic_24h.get(key, zero) for key in keys), zero.copy())
    output["weighted_context_total_intensity_6h"] = sum(topic_6h.values())
    output["weighted_context_total_intensity_24h"] = sum(topic_24h.values())
    output["systemic_risk_stack_intensity_24h"] = stack(
        [
            "war_geopolitical",
            "banking_liquidity",
            "stablecoin_liquidity",
            "oil_energy_shock",
            "china_property_credit",
            "pandemic_health_shock",
        ]
    )
    output["macro_policy_stack_intensity_24h"] = stack(
        [
            "inflation_rates",
            "official_macro_release",
            "dollar_risk_off",
            "ai_bubble_risk",
        ]
    )
    output["crypto_policy_stack_intensity_24h"] = stack(
        [
            "regulation_enforcement",
            "etf_institutional_flow",
            "stablecoin_liquidity",
        ]
    )
    output["crypto_stress_stack_intensity_24h"] = stack(
        [
            "security_exploit",
            "stablecoin_liquidity",
            "liquidation_leverage",
            "crypto_market_structure",
        ]
    )
    confluence_columns = [f"{topic}_confluence_24h" for topic in WEIGHTED_TOPIC_PATTERNS]
    persistence_columns = [f"{topic}_persistence_hours_24h" for topic in WEIGHTED_TOPIC_PATTERNS]
    severity_columns = [f"{topic}_severity_max_24h" for topic in WEIGHTED_TOPIC_PATTERNS]
    official_columns = [f"{topic}_official_confirmed_intensity_24h" for topic in WEIGHTED_TOPIC_PATTERNS]
    output["cross_topic_confluence_max_24h"] = output[confluence_columns].max(axis=1)
    output["cross_topic_persistence_count_24h"] = (output[persistence_columns] >= 3.0).sum(axis=1).astype(float)
    output["high_severity_topic_count_24h"] = (output[severity_columns] >= 1.5).sum(axis=1).astype(float)
    output["official_confirmed_context_intensity_24h"] = output[official_columns].sum(axis=1)


def _article_source_stats(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {"rows": 0, "missing_available_at": 0, "min_available_at": None, "max_available_at": None, "warning": f"Missing database: {db_path}"}
    with sqlite3.connect(str(db_path), timeout=10.0) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS rows_total,
                   SUM(CASE WHEN collected_at IS NULL OR TRIM(collected_at) = '' THEN 1 ELSE 0 END) AS missing_available,
                   MIN(collected_at) AS min_available,
                   MAX(collected_at) AS max_available
            FROM articles
            """
        ).fetchone()
    return {
        "rows": int(row[0] or 0),
        "missing_available_at": int(row[1] or 0),
        "min_available_at": row[2],
        "max_available_at": row[3],
    }


def _global_source_stats(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {"rows": 0, "missing_available_at": 0, "min_available_at": None, "max_available_at": None, "warning": f"Missing database: {db_path}"}
    with sqlite3.connect(str(db_path), timeout=10.0) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS rows_total,
                   SUM(CASE WHEN ts IS NULL OR TRIM(ts) = '' THEN 1 ELSE 0 END) AS missing_available,
                   MIN(ts) AS min_available,
                   MAX(ts) AS max_available
            FROM global_context_ticks
            """
        ).fetchone()
    return {
        "rows": int(row[0] or 0),
        "missing_available_at": int(row[1] or 0),
        "min_available_at": row[2],
        "max_available_at": row[3],
    }


def _gdelt_source_stats(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {"rows": 0, "missing_available_at": 0, "min_available_at": None, "max_available_at": None}
    try:
        with sqlite3.connect(str(db_path), timeout=10.0) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS rows_total,
                       SUM(CASE WHEN date IS NULL OR TRIM(date) = '' THEN 1 ELSE 0 END) AS missing_available,
                       MIN(date) AS min_available,
                       MAX(date) AS max_available
                FROM gdelt_hourly_features
                WHERE parse_error IS NULL
                """
            ).fetchone()
    except sqlite3.Error as exc:
        return {"rows": 0, "missing_available_at": 0, "min_available_at": None, "max_available_at": None, "warning": f"GDELT database unreadable: {exc}"}
    return {
        "rows": int(row[0] or 0),
        "missing_available_at": int(row[1] or 0),
        "min_available_at": row[2],
        "max_available_at": row[3],
    }


def _gkg_source_stats(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {"rows": 0, "missing_available_at": 0, "min_available_at": None, "max_available_at": None}
    try:
        with sqlite3.connect(str(db_path), timeout=10.0) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS rows_total,
                       SUM(CASE WHEN date IS NULL OR TRIM(date) = '' THEN 1 ELSE 0 END) AS missing_available,
                       MIN(date) AS min_available,
                       MAX(date) AS max_available
                FROM gdelt_gkg_file_features
                WHERE parse_error IS NULL
                """
            ).fetchone()
    except sqlite3.Error as exc:
        return {"rows": 0, "missing_available_at": 0, "min_available_at": None, "max_available_at": None, "warning": f"GDELT GKG database unreadable: {exc}"}
    return {
        "rows": int(row[0] or 0),
        "missing_available_at": int(row[1] or 0),
        "min_available_at": row[2],
        "max_available_at": row[3],
    }


def _load_articles(db_path: Path, source_family: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not db_path.exists():
        return _empty_articles()
    with sqlite3.connect(str(db_path), timeout=10.0) as conn:
        query = """
            SELECT
                a.id AS raw_id,
                a.source_id AS source_name,
                a.source_group,
                a.source_type,
                a.market_relevance,
                a.source_url,
                a.canonical_url,
                a.title,
                a.summary,
                a.published_at,
                a.collected_at AS available_at,
                a.updated_at,
                a.event_type,
                GROUP_CONCAT(DISTINCT t.namespace || ':' || t.value) AS tags_text,
                GROUP_CONCAT(DISTINCT c.category) AS categories_text,
                GROUP_CONCAT(DISTINCT aa.asset) AS assets_text
            FROM articles a
            LEFT JOIN article_tags at ON at.article_id = a.id
            LEFT JOIN tags t ON t.id = at.tag_id
            LEFT JOIN article_categories c ON c.article_id = a.id
            LEFT JOIN article_assets aa ON aa.article_id = a.id
            WHERE a.collected_at IS NOT NULL
              AND TRIM(a.collected_at) != ''
              AND a.collected_at >= ?
              AND a.collected_at <= ?
            GROUP BY a.id
            ORDER BY a.collected_at
        """
        frame = pd.read_sql_query(query, conn, params=(_iso(start), _iso(end)))
    if frame.empty:
        return _empty_articles()
    frame["source_family"] = source_family
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True, errors="coerce", format="mixed")
    frame = frame.dropna(subset=["available_at"]).copy()
    frame["event_id"] = source_family + ":" + frame["raw_id"].astype(str)
    frame["url"] = frame["canonical_url"].fillna("").where(frame["canonical_url"].fillna("") != "", frame["source_url"].fillna(""))
    frame["text_blob"] = (
        frame[["title", "summary", "event_type", "tags_text", "categories_text", "assets_text", "source_group"]]
        .fillna("")
        .agg(" ".join, axis=1)
    )
    frame["duplicate_key"] = frame["title"].fillna("").map(_duplicate_key)
    frame["severity_weight"] = frame["text_blob"].map(_severity_weight)
    frame["confidence_weight"] = frame["text_blob"].map(_confidence_weight)
    frame["source_quality_weight"] = frame.apply(_source_quality_weight, axis=1)
    frame["asset_relevance_weight"] = frame.apply(_asset_relevance_weight, axis=1)
    for key, pattern in TAG_PATTERNS.items():
        frame[f"has_{key}"] = frame["text_blob"].map(lambda value, regex=pattern: 1.0 if regex.search(str(value)) else 0.0)
    for topic, config in WEIGHTED_TOPIC_PATTERNS.items():
        pattern = config["pattern"]
        base_weight = float(config["base_weight"])
        matched = frame["text_blob"].map(lambda value, regex=pattern: 1.0 if regex.search(str(value)) else 0.0)
        frame[f"{topic}_event_intensity"] = (
            matched
            * base_weight
            * frame["severity_weight"]
            * frame["confidence_weight"]
            * frame["source_quality_weight"]
            * frame["asset_relevance_weight"]
        )
    topic_columns = [f"{topic}_event_intensity" for topic in WEIGHTED_TOPIC_PATTERNS]
    if topic_columns:
        topic_values = frame[topic_columns].fillna(0.0)
        frame["weighted_topic_intensity"] = topic_values.max(axis=1)
        topic_keys = list(WEIGHTED_TOPIC_PATTERNS)
        max_index = topic_values.to_numpy().argmax(axis=1)
        frame["weighted_topic_key"] = [
            topic_keys[int(idx)] if float(frame["weighted_topic_intensity"].iloc[row_idx]) > 0 else "none"
            for row_idx, idx in enumerate(max_index)
        ]
        frame["topic_intensities_json"] = frame.apply(_topic_intensities_json, axis=1)
    else:
        frame["weighted_topic_intensity"] = 0.0
        frame["weighted_topic_key"] = "none"
        frame["topic_intensities_json"] = "{}"
    frame["high_severity_flag"] = (
        (pd.to_numeric(frame["severity_weight"], errors="coerce").fillna(0.0) >= 1.5)
        & (pd.to_numeric(frame["weighted_topic_intensity"], errors="coerce").fillna(0.0) > 0)
    ).astype(float)
    frame["topic_key"] = frame.apply(_article_topic_key, axis=1)
    return frame


def _load_global_context(db_path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame(columns=["event_id", "source_name", "source_family", "source_group", "available_at", "metric_key", "metric_value"])
    with sqlite3.connect(str(db_path), timeout=10.0) as conn:
        frame = pd.read_sql_query(
            """
            SELECT id AS raw_id, source_id AS source_name, source_group, source_type, metric_key,
                   value, score, ts AS available_at, source_ts, created_at, notes
            FROM global_context_ticks
            WHERE ts IS NOT NULL
              AND TRIM(ts) != ''
              AND ts >= ?
              AND ts <= ?
            ORDER BY ts
            """,
            conn,
            params=(_iso(start), _iso(end)),
        )
    if frame.empty:
        return frame
    frame["source_family"] = "global"
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True, errors="coerce", format="mixed")
    frame = frame.dropna(subset=["available_at"]).copy()
    frame["event_id"] = "global:" + frame["raw_id"].astype(str)
    frame["metric_value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame


def _load_gdelt_context(db_path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    columns = [
        "date",
        "event_count",
        "num_articles_sum",
        "num_mentions_sum",
        "num_sources_sum",
        "avg_tone_weighted",
        "goldstein_weighted",
        "conflict_event_count",
        "protest_event_count",
        "coercion_event_count",
        "sanctions_trade_url_count",
        "oil_energy_url_count",
        "banking_credit_url_count",
        "macro_url_count",
        "crypto_url_count",
    ]
    if not db_path.exists():
        return pd.DataFrame(columns=columns)
    try:
        with sqlite3.connect(str(db_path), timeout=10.0) as conn:
            frame = pd.read_sql_query(
                f"""
                SELECT {', '.join(columns)}
                FROM gdelt_hourly_features
                WHERE date >= ? AND date <= ?
                  AND parse_error IS NULL
                ORDER BY date
                """,
                conn,
                params=(_iso(start), _iso(end)),
            )
    except sqlite3.Error:
        return pd.DataFrame(columns=columns)
    return frame


def _load_gkg_context(db_path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    columns = [
        "date",
        "document_count",
        "source_count",
        "word_count_sum",
        "tone_weight_sum",
        "tone_sum",
        "positive_tone_sum",
        "negative_tone_sum",
        "polarity_sum",
        "activity_sum",
        "theme_count_sum",
        "unique_theme_count",
        "top_theme_doc_count",
        *[f"{topic}_doc_count" for topic in GKG_TOPIC_KEYS],
    ]
    if not db_path.exists():
        return pd.DataFrame(columns=columns)
    try:
        with sqlite3.connect(str(db_path), timeout=10.0) as conn:
            frame = pd.read_sql_query(
                f"""
                SELECT {', '.join(columns)}
                FROM gdelt_gkg_file_features
                WHERE date >= ? AND date <= ?
                  AND parse_error IS NULL
                ORDER BY date
                """,
                conn,
                params=(_iso(start), _iso(end)),
            )
    except sqlite3.Error:
        return pd.DataFrame(columns=columns)
    return frame


def _article_topic_key(row: pd.Series) -> str:
    weighted = str(row.get("weighted_topic_key") or "")
    if weighted and weighted != "none":
        return weighted
    for topic in WEIGHTED_TOPIC_PATTERNS:
        if float(row.get(f"{topic}_event_intensity") or 0.0) > 0:
            return topic
    for column in ("event_type", "categories_text", "tags_text", "source_group"):
        value = str(row.get(column) or "").strip()
        if value:
            return value.split(",")[0].lower()[:80]
    return "unknown"


def _topic_intensities_json(row: pd.Series) -> str:
    payload = {
        topic: round(float(row.get(f"{topic}_event_intensity") or 0.0), 6)
        for topic in WEIGHTED_TOPIC_PATTERNS
        if float(row.get(f"{topic}_event_intensity") or 0.0) > 0
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _duplicate_key(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return "unknown"
    return hashlib.sha1(cleaned[:160].encode("utf-8", errors="ignore")).hexdigest()[:16]


def _severity_weight(text: str) -> float:
    lowered = str(text).lower()
    if re.search(r"\b(crisis|emergency|invasion|default|insolven|bank run|collapse|systemic|depeg|exploit|hack|breach|war|missile|sanction|crackdown|criminal|covid|coronavirus|pandemic|lockdown|iran|ukraine|evergrande|country garden|hormuz)\w*", lowered):
        return 1.75
    if re.search(r"\b(warning|stress|downgrade|lawsuit|enforcement|settlement|fine|outflow|liquidat|volatility|shock|contagion|recession|peace talks?|ceasefire|truce|ai bubble|property developer|housing slump)\w*", lowered):
        return 1.35
    if re.search(r"\b(report|minutes|statement|data|release|update|proposal|consultation)\w*", lowered):
        return 1.1
    return 1.0


def _confidence_weight(text: str) -> float:
    lowered = str(text).lower()
    if "collector_confidence:official_source" in lowered or re.search(r"\b(official|announced|published|confirmed|statement|minutes|filing|court|data release)\w*", lowered):
        return 1.25
    if re.search(r"\b(reported|according to|sources said|expected|forecast)\w*", lowered):
        return 1.0
    if re.search(r"\b(rumou?r|unconfirmed|speculat|may|could)\w*", lowered):
        return 0.75
    return 1.0


def _source_quality_weight(row: pd.Series) -> float:
    text = " ".join(
        str(row.get(column) or "")
        for column in ("source_group", "source_type", "tags_text", "source_family")
    ).lower()
    if "official_data_releases" in text or "central_banks" in text or "official_source" in text:
        return 1.4
    if "source_priority:critical" in text:
        return 1.3
    if "news" in text:
        return 1.0
    if "web" in text:
        return 0.85
    return 0.95


def _asset_relevance_weight(row: pd.Series) -> float:
    text = " ".join(
        str(row.get(column) or "")
        for column in ("assets_text", "tags_text", "categories_text", "title", "summary", "market_relevance")
    ).lower()
    if re.search(r"\b(bitcoin|btc)\b", text):
        return 1.45
    if re.search(r"\b(ethereum|eth|crypto|defi|stablecoin|asset_class:crypto)\b", text):
        return 1.25
    if "market_relevance:critical" in text:
        return 1.2
    if re.search(r"\b(fed|fomc|central bank|inflation|rates?|treasury|oil|banking|risk|liquidity|asset_class:tradfi)\b", text):
        return 1.0
    return 0.55


def _write_normalised_events(feature_db: Path, articles: pd.DataFrame, globals_df: pd.DataFrame, updated_at: str) -> None:
    rows: list[tuple[Any, ...]] = []
    for row in articles.itertuples(index=False):
        tags = [str(getattr(row, name, "") or "") for name in ("tags_text", "categories_text", "assets_text")]
        rows.append(
            (
                row.event_id,
                row.source_name,
                row.source_family,
                row.source_group,
                _asset_scope(str(getattr(row, "assets_text", "") or "") + " " + str(getattr(row, "text_blob", "") or "")),
                getattr(row, "title", None),
                getattr(row, "summary", None),
                getattr(row, "url", None),
                getattr(row, "published_at", None),
                _iso(row.available_at),
                _iso(row.available_at),
                getattr(row, "topic_key", None),
                getattr(row, "duplicate_key", None),
                getattr(row, "weighted_topic_key", None),
                getattr(row, "weighted_topic_intensity", None),
                getattr(row, "severity_weight", None),
                getattr(row, "confidence_weight", None),
                getattr(row, "source_quality_weight", None),
                getattr(row, "asset_relevance_weight", None),
                getattr(row, "high_severity_flag", None),
                getattr(row, "topic_intensities_json", "{}"),
                None,
                None,
                json.dumps([tag for tag in tags if tag]),
                "news_web",
                "articles",
                str(row.raw_id),
                updated_at,
            )
        )
    for row in globals_df.itertuples(index=False):
        rows.append(
            (
                row.event_id,
                row.source_name,
                "global",
                row.source_group,
                "macro",
                None,
                getattr(row, "notes", None),
                None,
                getattr(row, "source_ts", None),
                _iso(row.available_at),
                _iso(row.available_at),
                row.metric_key,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "{}",
                row.metric_key,
                getattr(row, "metric_value", None),
                "[]",
                "global_context",
                "global_context_ticks",
                str(row.raw_id),
                updated_at,
            )
        )
    if not rows:
        return
    with sqlite3.connect(str(feature_db), timeout=10.0) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO normalised_context_events (
                event_id, source_name, source_family, source_group, asset_scope, title, summary, url,
                published_at, fetched_at, available_at, topic_key, duplicate_key,
                weighted_topic_key, weighted_topic_intensity, severity_weight, confidence_weight,
                source_quality_weight, asset_relevance_weight, high_severity_flag, topic_intensities_json,
                metric_key, metric_value,
                tags_json, raw_db, raw_table, raw_id, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def _write_feature_rows(feature_db: Path, frame: pd.DataFrame, *, mode: str, generated_at: str) -> None:
    if frame.empty:
        return
    config_hash = feature_config_hash()
    columns = ["date", *FEATURE_COLUMNS, *DEBUG_NUMERIC_COLUMNS, "generated_at", "min_source_available_at", "max_source_available_at", "schema_version", "feature_config_hash"]
    values = []
    for row in frame.itertuples(index=False):
        payload = {column: getattr(row, column) for column in frame.columns}
        payload["generated_at"] = generated_at
        payload["schema_version"] = SCHEMA_VERSION
        payload["feature_config_hash"] = config_hash
        values.append(tuple(payload.get(column) for column in columns))
    placeholders = ", ".join("?" for _ in columns)
    with sqlite3.connect(str(feature_db), timeout=10.0) as conn:
        if mode == "full-rebuild":
            conn.execute("DELETE FROM context_features_1h")
        conn.executemany(
            f"""
            INSERT OR REPLACE INTO context_features_1h ({', '.join(columns)})
            VALUES ({placeholders})
            """,
            values,
        )


def _read_state(feature_db: Path, *, initialize: bool = True) -> dict[str, Any]:
    if initialize:
        init_feature_db(feature_db)
    try:
        with sqlite3.connect(str(feature_db), timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM feature_build_state WHERE id = 1").fetchone()
    except sqlite3.Error:
        return {}
    return dict(row) if row else {}


def _write_state(feature_db: Path, last_raw_event_seen: str | None, last_feature_hour: str | None, updated_at: str) -> None:
    with sqlite3.connect(str(feature_db), timeout=10.0) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO feature_build_state (
                id, last_successful_build_time, last_raw_event_seen, last_feature_hour_built,
                schema_version, feature_config_hash, updated_at
            )
            VALUES (1, ?, ?, ?, ?, ?, ?)
            """,
            (updated_at, last_raw_event_seen, last_feature_hour, SCHEMA_VERSION, feature_config_hash(), updated_at),
        )


def _empty_articles() -> pd.DataFrame:
    columns = [
        "event_id",
        "raw_id",
        "source_name",
        "source_family",
        "source_group",
        "title",
        "summary",
        "url",
        "source_type",
        "market_relevance",
        "event_type",
        "tags_text",
        "categories_text",
        "assets_text",
        "published_at",
        "available_at",
        "topic_key",
        "text_blob",
        "duplicate_key",
        "severity_weight",
        "confidence_weight",
        "source_quality_weight",
        "asset_relevance_weight",
        "weighted_topic_key",
        "weighted_topic_intensity",
        "high_severity_flag",
        "topic_intensities_json",
        *[f"has_{key}" for key in TAG_PATTERNS],
        *[f"{topic}_event_intensity" for topic in WEIGHTED_TOPIC_PATTERNS],
    ]
    return pd.DataFrame(columns=columns)


def _asset_scope(text: str) -> str:
    lowered = text.lower()
    if "btc" in lowered or "bitcoin" in lowered:
        return "BTC"
    if "eth" in lowered or "ethereum" in lowered:
        return "ETH"
    if "crypto" in lowered:
        return "broad_crypto"
    if TAG_PATTERNS["macro"].search(lowered):
        return "macro"
    return "unknown"


def _parse_ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(pd.to_datetime(value, utc=True))


def _parse_optional(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def _iso(value: Any) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()

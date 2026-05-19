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


SCHEMA_VERSION = 2

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

FEATURE_COLUMNS = [*BASE_FEATURE_COLUMNS, *EVENT_FAMILY_COLUMNS]

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
    data_root = (resolved_app_dir / "../research_news_data").resolve()
    feature_root = data_root / "context_features"
    user_data_dir = resolved_app_dir.parent
    return ContextFeaturePaths(
        app_dir=resolved_app_dir,
        news_db=data_root / "news" / "news_events.sqlite",
        web_db=data_root / "web" / "web_events.sqlite",
        global_db=data_root / "global_context" / "global_context.sqlite",
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
        "tag_patterns": sorted(TAG_PATTERNS),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


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
    max_window = pd.Timedelta(days=7)
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

    history_start = (build_start - max_window).floor("h")
    article_frames = [
        _load_articles(paths.news_db, "news", history_start, build_end),
        _load_articles(paths.web_db, "web", history_start, build_end),
    ]
    articles = pd.concat([frame for frame in article_frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in article_frames) else _empty_articles()
    globals_df = _load_global_context(paths.global_db, history_start, build_end)
    source_rows_loaded = {
        "news": int((articles["source_family"] == "news").sum()) if not articles.empty else 0,
        "web": int((articles["source_family"] == "web").sum()) if not articles.empty else 0,
        "global": len(globals_df),
    }
    summary.source_rows_loaded = source_rows_loaded
    summary.build_start = _iso(build_start)
    summary.build_end = _iso(build_end)
    summary.affected_hours = int(((build_end - build_start) / pd.Timedelta(hours=1)) + 1)

    feature_frame = _build_feature_frame(articles, globals_df, build_start, build_end, int(sum(missing_available.values())))
    if dry_run:
        return summary

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
    columns = ["date", *FEATURE_COLUMNS, *DEBUG_NUMERIC_COLUMNS]
    with sqlite3.connect(str(feature_db), timeout=10.0) as conn:
        frame = pd.read_sql_query(
            f"SELECT {', '.join(columns)} FROM context_features_1h ORDER BY date",
            conn,
        )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = export_dir / f"context_features_1h_{stamp}.parquet"
    if parquet:
        frame.to_parquet(output, index=False)
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
    for horizon in (1, 6, 24):
        prices[f"future_return_{horizon}h"] = prices["close"].shift(-horizon) / prices["close"] - 1.0
        prices[f"future_up_{horizon}h"] = (prices[f"future_return_{horizon}h"] > 0).astype(float)
    features["date"] = pd.to_datetime(features["date"], utc=True).dt.floor("h")
    merged = features.merge(prices, on="date", how="inner").dropna(subset=["future_return_6h"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"context_btc_return_report_{stamp}.csv"
    prediction_path = report_dir / f"context_btc_predictions_{stamp}.csv"
    family_report_path = report_dir / f"context_feature_family_report_{stamp}.csv"
    rows: list[dict[str, Any]] = []
    for feature in FEATURE_COLUMNS:
        series = pd.to_numeric(merged[feature], errors="coerce")
        for horizon in (1, 6, 24):
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
        corr_6h = correlations[(correlations["feature"].isin(available)) & (correlations["label"] == "future_return_6h")].copy()
        corr_6h["abs_correlation"] = corr_6h["correlation"].abs()
        best = corr_6h.sort_values("abs_correlation", ascending=False).head(1)
        group_metrics = _ridge_metrics(frame, available)
        minus_metrics = _ridge_metrics(frame, [feature for feature in FEATURE_COLUMNS if feature in frame and feature not in available])
        rows.append(
            {
                "family": group,
                "feature_count": len(available),
                "active_feature_count": sum(_is_active_feature(frame[feature]) for feature in available),
                "mean_abs_corr_6h": corr_6h["abs_correlation"].mean() if not corr_6h.empty else np.nan,
                "max_abs_corr_6h": corr_6h["abs_correlation"].max() if not corr_6h.empty else np.nan,
                "best_feature_6h": best["feature"].iloc[0] if not best.empty else "",
                "best_feature_corr_6h": best["correlation"].iloc[0] if not best.empty else np.nan,
                "group_only_return_corr_6h": group_metrics.get("return_corr_6h"),
                "group_only_direction_accuracy_6h": group_metrics.get("direction_accuracy_6h"),
                "full_return_corr_6h": full_metrics.get("return_corr_6h"),
                "full_direction_accuracy_6h": full_metrics.get("direction_accuracy_6h"),
                "minus_family_return_corr_6h": minus_metrics.get("return_corr_6h"),
                "minus_family_direction_accuracy_6h": minus_metrics.get("direction_accuracy_6h"),
                "direction_accuracy_drop_when_removed": _metric_drop(full_metrics, minus_metrics, "direction_accuracy_6h"),
                "return_corr_drop_when_removed": _metric_drop(full_metrics, minus_metrics, "return_corr_6h"),
            }
        )
    pd.DataFrame(rows).to_csv(output, index=False)


def _feature_group(feature: str) -> str:
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


def _ridge_metrics(frame: pd.DataFrame, features: list[str]) -> dict[str, float]:
    usable_features = [feature for feature in features if feature in frame and _is_active_feature(frame[feature])]
    usable = frame.dropna(subset=["future_return_6h"]).copy()
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
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    model.fit(train[usable_features], train["future_return_6h"])
    predicted = pd.Series(model.predict(test[usable_features]), index=test.index)
    actual = pd.to_numeric(test["future_return_6h"], errors="coerce")
    valid = pd.DataFrame({"predicted": predicted, "actual": actual}).dropna()
    if len(valid) < 5:
        return {}
    direction_accuracy = ((valid["predicted"] > 0).astype(float) == (valid["actual"] > 0).astype(float)).mean()
    return_corr = valid["predicted"].corr(valid["actual"]) if valid["predicted"].nunique() > 1 and valid["actual"].nunique() > 1 else np.nan
    return {
        "direction_accuracy_6h": float(direction_accuracy),
        "return_corr_6h": float(return_corr) if pd.notna(return_corr) else np.nan,
    }


def _metric_drop(full_metrics: dict[str, float], minus_metrics: dict[str, float], key: str) -> float:
    full = full_metrics.get(key)
    minus = minus_metrics.get(key)
    if full is None or minus is None or pd.isna(full) or pd.isna(minus):
        return np.nan
    return float(full - minus)


def _write_prediction_export(frame: pd.DataFrame, output: Path) -> None:
    usable = frame.dropna(subset=["future_return_6h"]).copy()
    feature_cols = [col for col in FEATURE_COLUMNS if col in usable.columns]
    if len(usable) < 30:
        usable[["date", "future_return_6h", "future_up_6h"]].to_csv(output, index=False)
        return
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        usable[["date", "future_return_6h", "future_up_6h"]].to_csv(output, index=False)
        return
    split = max(20, int(len(usable) * 0.7))
    train = usable.iloc[:split]
    test = usable.iloc[split:]
    if test.empty:
        usable[["date", "future_return_6h", "future_up_6h"]].to_csv(output, index=False)
        return
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    model.fit(train[feature_cols], train["future_return_6h"])
    predicted = model.predict(test[feature_cols])
    result = test[["date", "future_return_6h", "future_up_6h"]].copy()
    result["predicted_return_6h"] = predicted
    result["predicted_up_6h"] = (result["predicted_return_6h"] > 0).astype(float)
    result["direction_correct_6h"] = (result["predicted_up_6h"] == result["future_up_6h"]).astype(float)
    result.to_csv(output, index=False)


def _build_feature_frame(
    articles: pd.DataFrame,
    globals_df: pd.DataFrame,
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

    articles = articles.sort_values("available_at").reset_index(drop=True)
    for index, hour in enumerate(hours):
        w1 = _window(articles, hour, 1)
        w6 = _window(articles, hour, 6)
        w24 = _window(articles, hour, 24)
        prev6 = _window_between(articles, hour - pd.Timedelta(hours=6), hour - pd.Timedelta(hours=12))
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
    _add_global_features(output, globals_df, hours, source_min, source_max)
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


def _window_between(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if frame.empty:
        return frame
    mask = (frame["available_at"] > end) & (frame["available_at"] <= start)
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
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["available_at"]).copy()
    frame["event_id"] = source_family + ":" + frame["raw_id"].astype(str)
    frame["url"] = frame["canonical_url"].fillna("").where(frame["canonical_url"].fillna("") != "", frame["source_url"].fillna(""))
    frame["text_blob"] = (
        frame[["title", "summary", "event_type", "tags_text", "categories_text", "assets_text", "source_group"]]
        .fillna("")
        .agg(" ".join, axis=1)
    )
    for key, pattern in TAG_PATTERNS.items():
        frame[f"has_{key}"] = frame["text_blob"].map(lambda value, regex=pattern: 1.0 if regex.search(str(value)) else 0.0)
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
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["available_at"]).copy()
    frame["event_id"] = "global:" + frame["raw_id"].astype(str)
    frame["metric_value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame


def _article_topic_key(row: pd.Series) -> str:
    for column in ("event_type", "categories_text", "tags_text", "source_group"):
        value = str(row.get(column) or "").strip()
        if value:
            return value.split(",")[0].lower()[:80]
    return "unknown"


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
                published_at, fetched_at, available_at, topic_key, metric_key, metric_value,
                tags_json, raw_db, raw_table, raw_id, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        "published_at",
        "available_at",
        "topic_key",
        "text_blob",
        *[f"has_{key}" for key in TAG_PATTERNS],
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

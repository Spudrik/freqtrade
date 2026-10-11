"""Small news-only input and publication CLI for scheduled Luna briefs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import re
import sys
from urllib.parse import urlparse

from user_data.Custom_Launcher.collector_runtime import atomic_write_text
from user_data.Custom_Launcher.research.crypto_derivatives_snapshot import collect_crypto_derivatives_snapshot
from user_data.Custom_Launcher.research.paper_trial_snapshot import higher_snapshot, source_snapshot
from user_data.strategies.integrated_paper_context import CONTEXT_FILE, publish_luna_context


ALLOWED_PUBLISHERS = ("apnews.com", "reuters.com", "theguardian.com", "aljazeera.com", "coindesk.com")
GLOBAL_METRICS = ("source_id", "source_group", "source_type", "url", "enabled", "last_success_at",
                  "last_failure_at", "last_observation_at", "last_collected_at")
DERIVATIVE_NUMBERS = ("coverage", "usable_metric_count", "advertised_metric_count", "oi_change_5m",
                      "oi_change_1h", "oi_age_minutes", "taker_imbalance_5m", "taker_imbalance_1h",
                      "taker_age_minutes", "funding_rate", "mark_index_premium", "premium_age_minutes",
                      "global_long_short_ratio", "global_long_short_ratio_change_1h", "positioning_age_minutes")
BREADTH_FIELDS = ("status", "sample", "requested", "coverage", "share_advancing", "median_pct",
                  "dispersion_mad_pct", "btc_pct", "median_minus_btc_pct", "observation_at_utc", "age_minutes")
REPORTER_PACKET_FILE = CONTEXT_FILE.with_name("luna_reporter_packet.json")
HYPERLIQUID_SAMPLE_IDS = tuple(f"pilot_hyperliquid_sample_{index:02d}" for index in range(1, 9))
CRYPTO_SECTION_PARTS = ("market", "book", "options_positioning", "flows_attention")
CRYPTO_OPTIONS_POSITIONING_IDS = frozenset({
    "pilot_deribit_options_btc", "pilot_deribit_options_eth", *HYPERLIQUID_SAMPLE_IDS,
})
REPORTER_SECTION_FIELDS = {
    "news": frozenset({"observed_at_utc", "news_web_freshness", "leads", "coverage_status", "source_failure"}),
    "global": frozenset({
        "status", "heartbeat_at", "last_fetch_at", "last_message_at", "source_error",
        "failure", "failure_detail",
        "sources_enabled", "sources_total", "source_limit", "sources_truncated",
        "projection_source_count", "source_count", "sources", "unmapped_sources", "unmapped_source_count",
    }),
    "crypto": frozenset({
        "global_projection", "orderbook", "book_pressure", "crypto_derivatives", "higher_timeframes",
    }),
}
SOURCE_ROUTES = {
    "fred_rates_risk_daily": ("rates_fx", "fred_series_basket", "global"),
    "fred_us_equity_daily": ("equity_indices", "fred_series_basket", "global"),
    "google_trends_macro_attention": ("attention", "google_trends_interest", "global"),
    "google_trends_market_stress_attention": ("attention", "google_trends_interest", "global"),
    "google_trends_systemic_risk_attention": ("attention", "google_trends_interest", "global"),
    "stooq_global_equity_risk": ("equity_indices", "stooq_quotes", "global"),
    "stooq_safe_haven_risk": ("rates_fx", "stooq_quotes", "global"),
    "stooq_us_equity_risk": ("equity_indices", "stooq_quotes", "global"),
    "alternative_fear_greed": ("sentiment", "alternative_fng", "crypto"),
    "coingecko_btc_eth": ("market", "coingecko_markets", "crypto"),
    "coingecko_global": ("market", "coingecko_global", "crypto"),
    "defillama_chains": ("defi", "defillama_chains", "crypto"),
    "defillama_stablecoins": ("defi", "defillama_stablecoins", "crypto"),
    "farside_btc_spot_etf_flows": ("institutional_flows", "farside_btc_etf_flows", "crypto"),
    "google_trends_crypto_attention": ("attention", "google_trends_interest", "crypto"),
    "pilot_bybit_btc_porrow_wallet": ("crypto_exchange_wallet_transfers", "bybit_btc_wallet", "crypto"),
    "pilot_bybit_eth_erc20_wallet": ("crypto_exchange_wallet_transfers", "bybit_eth_erc20_wallet", "crypto"),
    "pilot_bybit_eth_native_wallet": ("crypto_exchange_wallet_transfers", "bybit_eth_native_wallet", "crypto"),
    "pilot_bybit_liquidations_60s": ("crypto_derivatives_liquidations", "bybit_liquidation_observer", "crypto"),
    "pilot_coingecko_btc_eth_relative_strength": ("crypto_relative_strength", "coingecko_relative_strength", "crypto"),
    "pilot_defillama_dex_volume": ("crypto_dex_activity", "defillama_dex_volume", "crypto"),
    "pilot_defillama_stablecoin_chains": ("crypto_stablecoins", "defillama_stablecoin_chains", "crypto"),
    "pilot_deribit_options_btc": ("crypto_options", "deribit_options", "crypto"),
    "pilot_deribit_options_eth": ("crypto_options", "deribit_options", "crypto"),
}
for _sample_id in HYPERLIQUID_SAMPLE_IDS:
    SOURCE_ROUTES[_sample_id] = ("crypto_perpetual_position_sample", "hyperliquid_clearinghouse", "crypto")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _number(value):
    return value if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) else None


def _allowed_url(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = urlparse(value)
        host = (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not host:
        return None
    return value if any(host == domain or host.endswith("." + domain) for domain in ALLOWED_PUBLISHERS) else None


def _freshness(raw, web=False):
    raw = raw if isinstance(raw, dict) else {}
    fields = ("status", "heartbeat_at", "last_fetch_at", "last_message_at")
    result = {key: raw.get(key, "unknown") for key in fields}
    if web:
        result["headline_window_seconds"] = raw.get("headline_window_seconds", "unknown")
    return result


def _lead(raw):
    if not isinstance(raw, dict) or not isinstance(raw.get("title"), str):
        return None
    # Keep a lead only when its own title/age clocks belong to an allowed primary URL.
    url = _allowed_url(raw.get("url"))
    if not url:
        return None
    extras = []
    for item in raw.get("additional_sources", []) if isinstance(raw.get("additional_sources"), list) else []:
        extra_url = _allowed_url(item.get("url")) if isinstance(item, dict) else None
        if extra_url:
            extras.append({"url": extra_url, "published_at": item.get("published_at"),
                           "collected_at": item.get("collected_at"),
                           "published_age_status": item.get("published_age_status", "unknown"),
                           "collected_age_status": item.get("collected_age_status", "unknown")})
    return {"title": raw["title"], "collector": raw.get("collector", "unknown"), "url": url,
            "published_at": raw.get("published_at"), "collected_at": raw.get("collected_at"),
            "published_age_status": raw.get("published_age_status", "unknown"),
            "collected_age_status": raw.get("collected_age_status", "unknown"),
            "additional_sources": extras}


def _headlines_from_snapshot(raw, now, source_failure=None):
    result = {"observed_at_utc": now.astimezone(timezone.utc).isoformat(),
              "news_web_freshness": {"news": _freshness(None), "web": _freshness(None, True)}, "leads": []}
    if source_failure is not None:
        result.update(coverage_status="unknown", source_failure=source_failure["failure"])
        return result
    if not isinstance(raw, dict):
        result.update(coverage_status="unknown", source_failure="invalid source_snapshot result")
        return result
    result["news_web_freshness"] = {"news": _freshness(raw.get("news")), "web": _freshness(raw.get("web"), True)}
    headlines = raw.get("recent_unique_headlines")
    if isinstance(headlines, list):
        result["leads"] = [lead for item in headlines if (lead := _lead(item)) is not None][:20]
    # Collector status/freshness is not a verdict that the reporter has usable coverage.
    result["coverage_status"] = "reported_by_collectors"
    return result


def headlines_packet(now=None):
    now = now or _now()
    try:
        raw = source_snapshot(now)
    except Exception as exc:
        return _headlines_from_snapshot(None, now, {"failure": type(exc).__name__})
    return _headlines_from_snapshot(raw, now)


def _error(raw):
    if isinstance(raw, dict):
        return {str(key): str(value)[:120] for key, value in list(raw.items())[:3] if value}
    return str(raw)[:160] if raw else None


def _coverage(raw):
    if not isinstance(raw, dict):
        return {"status": "unknown"}
    return {key: (_number(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value)
            for key, value in raw.items()
            if isinstance(value, (bool, int, float)) or key == "status" or key.endswith("_at") or key.endswith("_utc")}


def _source_clocks(raw):
    raw = raw if isinstance(raw, dict) else {}
    keys = ("status", "heartbeat_at", "last_fetch_at", "last_message_at", "sources_enabled", "sources_total")
    result = {key: raw.get(key, "unknown") for key in keys if key in raw or key == "status"}
    if raw.get("last_error"):
        result["source_error"] = str(raw["last_error"])[:160]
    return result


def _source_route(source):
    source_id = source.get("source_id")
    expected = SOURCE_ROUTES.get(source_id)
    actual = (source.get("source_group"), source.get("source_type"))
    if expected is None:
        return "unknown", "unmapped_source_id_group_type", None
    if actual != expected[:2]:
        return "unknown", "known_source_id_group_or_type_mismatch", expected[:2]
    return expected[2], "mapped", None


def _selected_global_metrics(source, metrics, now):
    source_id = source.get("source_id")
    source_type = source.get("source_type")
    if source_id == "farside_btc_spot_etf_flows":
        priorities = {
            "btc_spot_etf_net_flow_musd", "btc_spot_etf_positive_fund_count",
            "btc_spot_etf_negative_fund_count", "btc_spot_etf_fbtc_flow_musd",
            "btc_spot_etf_gbtc_flow_musd", "btc_spot_etf_ibit_flow_musd",
        }
        selected = [metric for metric in metrics if metric.get("metric_key") in priorities]
        return selected, "complete explicit BTC ETF net/fund-count and major-fund set"
    if source_type == "deribit_options":
        expiry_by_metric = {}
        for metric in metrics:
            key = metric.get("metric_key")
            if isinstance(key, str) and key.startswith("deribit_option_"):
                match = re.search(r"(20\d{2}-\d{2}-\d{2})$", key)
                if match:
                    try:
                        expiry_by_metric[key] = datetime.strptime(match.group(1), "%Y-%m-%d").date()
                    except ValueError:
                        continue
        expiries = sorted(set(expiry_by_metric.values()))
        if expiries:
            current_date = now.astimezone(timezone.utc).date()
            upcoming = [expiry for expiry in expiries if expiry >= current_date]
            chosen = upcoming[:3]
            if not chosen:
                return [], "no upcoming UTC expiries; expired-only metrics omitted"
            chosen_set = set(chosen)
            return ([metric for metric in metrics
                     if expiry_by_metric.get(metric.get("metric_key")) in chosen_set],
                    "all available option metrics for the nearest three UTC expiries")
        return [], "no parseable expiry keys in compact options projection"
    # Wallet rows are already capped by the source snapshot. Keep every delivered
    # metric and preserve its source/collection clocks; never label them broad flow.
    return metrics, "all metrics delivered by the compact source projection"


def _global(raw, now=None):
    if not isinstance(raw, dict):
        return {"status": "unknown"}
    now = now or _now()
    result = _source_clocks(raw)
    result.update({key: raw[key] for key in ("source_count", "source_limit", "sources_truncated") if key in raw})
    result["sources"] = []
    sources = raw.get("sources")
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict):
                continue
            item = {key: source[key] for key in GLOBAL_METRICS if key in source}
            metrics = source.get("latest_metrics")
            metrics = metrics if isinstance(metrics, list) else []
            metrics = [metric for metric in metrics if isinstance(metric, dict)]
            selected, selection = _selected_global_metrics(source, metrics, now)
            selected_keys = {metric.get("metric_key") for metric in selected}
            total = source.get("metric_count")
            total = total if isinstance(total, int) and not isinstance(total, bool) and total >= 0 else len(metrics)
            route, route_status, expected = _source_route(source)
            item.update({"mapping_role": route, "mapping_status": route_status,
                         "metrics_total": total, "metrics_available": len(metrics),
                         "metrics_returned": len(selected),
                         "metrics_truncated": bool(source.get("metrics_truncated"))
                                             or total > len(metrics) or len(selected) < len(metrics),
                         "omitted_metric_keys": [metric.get("metric_key") for metric in metrics
                                                  if metric.get("metric_key") not in selected_keys],
                         "upstream_omitted_metric_count": max(0, total - len(metrics)),
                         "upstream_omitted_metric_keys_status":
                             "not_exposed_by_compact_source_projection" if total > len(metrics) else "none",
                         "metric_selection": selection})
            if expected is not None:
                item["mapping_expected_group_type"] = list(expected)
            item["latest_metrics"] = [{"metric_key": metric.get("metric_key"),
                "value": _number(metric.get("value")), "unit": metric.get("unit"),
                "source_ts": metric.get("source_ts"), "collected_ts": metric.get("ts")}
                for metric in selected]
            if source.get("source_id") in {
                "pilot_bybit_btc_porrow_wallet", "pilot_bybit_eth_erc20_wallet",
                "pilot_bybit_eth_native_wallet",
            }:
                item["scope_note"] = (
                    "Address-specific exchange-wallet monitor, not broad whale-flow evidence; "
                    "assess each source timestamp for freshness."
                )
            result["sources"].append(item)
    return result


def _breadth(raw):
    if not isinstance(raw, dict):
        return {"status": "unknown"}
    result = {key: raw.get(key, "unknown") for key in BREADTH_FIELDS if key in raw or key in ("status", "observation_at_utc")}
    for key in BREADTH_FIELDS:
        if key in result and key not in ("status", "observation_at_utc"):
            result[key] = _number(result[key])
    if raw.get("errors"):
        result["source_errors"] = _error(raw["errors"])
    return result


def _derivatives(raw):
    if not isinstance(raw, dict):
        return {"status": "unknown"}
    result = {key: raw.get(key, "unknown") for key in ("status", "fetched_at_utc", "sources")}
    binance = raw.get("binance")
    if isinstance(binance, dict):
        symbols = {}
        for symbol, row in (binance.get("symbols") or {}).items():
            if not isinstance(row, dict):
                continue
            item = {"status": row.get("status", "unknown"), "fresh_inputs": row.get("fresh_inputs", {})}
            item.update({key: _number(row.get(key)) for key in DERIVATIVE_NUMBERS if key in row})
            for key in ("oi_observation_at_utc", "taker_observation_at_utc", "premium_observation_at_utc", "positioning_observation_at_utc"):
                if key in row:
                    item[key] = row[key]
            if row.get("unavailable_or_partial_metrics"):
                item["unavailable_or_partial_metrics"] = row["unavailable_or_partial_metrics"]
            if row.get("errors"):
                item["source_errors"] = _error(row["errors"])
            symbols[symbol] = item
        breadth = binance.get("breadth") if isinstance(binance.get("breadth"), dict) else {}
        result["binance"] = {"symbols": symbols,
            "breadth": {"status": breadth.get("status", "unknown"),
                        "eligible_contracts_found": breadth.get("eligible_contracts_found", 0),
                        "one_hour": _breadth(breadth.get("one_hour")),
                        "twenty_four_hour": _breadth(breadth.get("twenty_four_hour"))}}
        for key in ("premium_error", "exchange_info_error"):
            if binance.get(key):
                result["binance"].setdefault("source_errors", {})[key] = str(binance[key])[:120]
        if breadth.get("errors"):
            result["binance"]["breadth"]["source_errors"] = _error(breadth["errors"])
    bybit = raw.get("bybit_crosscheck")
    if isinstance(bybit, dict):
        rows = {}
        for symbol, row in (bybit.get("symbols") or {}).items():
            if isinstance(row, dict):
                rows[symbol] = {"status": row.get("status", "unknown"),
                                "funding_rate": _number(row.get("funding_rate")),
                                "mark_index_premium": _number(row.get("mark_index_premium"))}
        result["bybit_crosscheck"] = {"status": bybit.get("status", "unknown"),
            "observation_at_utc": bybit.get("observation_at_utc"), "age_minutes": _number(bybit.get("age_minutes")),
            "symbols": rows}
        if bybit.get("error"):
            result["bybit_crosscheck"]["source_error"] = str(bybit["error"])[:120]
    dvol = raw.get("deribit_dvol")
    if isinstance(dvol, dict):
        result["deribit_dvol"] = {currency: {"status": row.get("status", "unknown"),
            "close": _number(row.get("close")), "observation_at_utc": row.get("observation_at_utc"),
            "age_minutes": _number(row.get("age_minutes")), **({"source_error": str(row["error"])[:120]} if row.get("error") else {})}
            for currency, row in dvol.items() if isinstance(row, dict)}
    return result


def _higher(raw):
    if not isinstance(raw, list):
        return []
    fields = ("symbol", "timeframe", "source", "last_close_utc", "close", "last_candle_change_pct",
              "recent_change_pct", "recent_completed_hours", "recent_high", "recent_low")
    result = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        item = {key: (row[key] if key in ("symbol", "timeframe", "source", "last_close_utc") else _number(row[key]))
                for key in fields if key in row}
        facts = row.get("completed_candle_facts")
        if isinstance(facts, dict):
            item["completed_candle_facts"] = {
                key: (_number(facts[key]) if key in ("age_minutes", "completed_candle_count") else facts[key])
                for key in ("state", "age_minutes", "last_close_utc", "completed_candle_count") if key in facts}
        result.append(item)
    return result


def _market_from_snapshot(raw, now, source_failure=None):
    result = {"observed_at_utc": now.astimezone(timezone.utc).isoformat(), "global": {"status": "unknown"},
              "orderbook": {"status": "unknown"}, "book_pressure": {},
              "crypto_derivatives": {"status": "unknown"}, "higher_timeframes": {"status": "unknown", "windows": []}}
    if source_failure is None and isinstance(raw, dict):
        result["global"] = _global(raw.get("global"), now)
        result["orderbook"] = _source_clocks(raw.get("orderbook"))
        pressure = raw.get("book_pressure")
        if isinstance(pressure, dict):
            result["book_pressure"] = {pair: {"imbalance": _number(item.get("imbalance")),
                "coverage": _coverage(item.get("coverage"))} for pair, item in pressure.items() if isinstance(item, dict)}
    else:
        if source_failure is None:
            source_failure = {"failure": "ValueError", "detail": "invalid source_snapshot result"}
        failure = {"status": "unknown", "failure": source_failure["failure"],
                   "failure_detail": source_failure.get("detail", "")[:160]}
        result["global"] = failure
        result["orderbook"] = failure
        result["book_pressure"] = {}
    try:
        result["crypto_derivatives"] = _derivatives(collect_crypto_derivatives_snapshot(now))
    except Exception as exc:
        result["crypto_derivatives"] = {"status": "unknown", "failure": type(exc).__name__, "failure_detail": str(exc)[:160]}
    try:
        windows = _higher(higher_snapshot(now))
        expected = {"BTCUSDT:4h", "BTCUSDT:1d", "ETHUSDT:4h", "ETHUSDT:1d"}
        found = {f"{row.get('symbol')}:{row.get('timeframe')}" for row in windows}
        observed = expected <= found and all(
            row.get("completed_candle_facts", {}).get("state") == "observed"
            for row in windows if f"{row.get('symbol')}:{row.get('timeframe')}" in expected)
        result["higher_timeframes"] = {"status": "observed" if observed else "degraded" if windows else "unknown",
                                       "windows": windows}
        if expected - found:
            result["higher_timeframes"]["missing_windows"] = sorted(expected - found)
    except Exception as exc:
        result["higher_timeframes"] = {"status": "unknown", "failure": type(exc).__name__,
                                       "failure_detail": str(exc)[:160], "windows": []}
    return result


def market_packet(now=None):
    now = now or _now()
    try:
        raw = source_snapshot(now)
    except Exception as exc:
        return _market_from_snapshot(None, now, {"failure": type(exc).__name__, "detail": str(exc)})
    return _market_from_snapshot(raw, now)


def _role_sections(market):
    projection = market.get("global") if isinstance(market.get("global"), dict) else {"status": "unknown"}
    global_sources, crypto_sources, unmapped_sources = [], [], []
    for source in projection.get("sources", []) if isinstance(projection.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        role = source.get("mapping_role")
        if role == "global":
            global_sources.append(source)
        elif role == "crypto":
            if source.get("source_id") in HYPERLIQUID_SAMPLE_IDS:
                source["scope_note"] = "Delivered public positioning sample only; not market-wide positioning."
            crypto_sources.append(source)
        else:
            unmapped_sources.append(source)
    projection_meta = {key: value for key, value in projection.items() if key != "sources"}
    if "source_count" in projection_meta:
        projection_meta["projection_source_count"] = projection_meta.pop("source_count")
    global_slice = {**projection_meta, "sources": global_sources, "source_count": len(global_sources),
                    "unmapped_sources": unmapped_sources, "unmapped_source_count": len(unmapped_sources)}
    crypto_projection = {**projection_meta, "sources": crypto_sources, "source_count": len(crypto_sources)}
    crypto_slice = {
        "global_projection": crypto_projection,
        "orderbook": market.get("orderbook", {"status": "unknown"}),
        "book_pressure": market.get("book_pressure", {}),
        "crypto_derivatives": market.get("crypto_derivatives", {"status": "unknown"}),
        "higher_timeframes": market.get("higher_timeframes", {"status": "unknown", "windows": []}),
    }
    return {"global": global_slice, "crypto": crypto_slice}


def prepare_packet(now=None):
    """Read each existing projection once and freeze only the three role slices."""
    now = now or _now()
    source_failure = None
    try:
        raw = source_snapshot(now)
    except Exception as exc:
        raw = None
        source_failure = {"failure": type(exc).__name__, "detail": str(exc)}
    news = _headlines_from_snapshot(raw, now, source_failure)
    market = _market_from_snapshot(raw, now, source_failure)
    roles = _role_sections(market)
    return {"observed_at_utc": now.astimezone(timezone.utc).isoformat(),
            "sections": {"news": news, **roles}}


def _parse_clock(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("observed_at_utc must be a nonempty timezone-aware ISO clock")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("observed_at_utc must be a parseable timezone-aware ISO clock") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("observed_at_utc must include a timezone")
    return parsed.astimezone(timezone.utc)


def _source_metric_count(source, field):
    value = source.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"prepared source {field} is malformed")
    return value


def _crypto_part_payload(section, unmapped_sources, part):
    projection = section["global_projection"]
    sources = projection.get("sources")
    if not isinstance(sources, list) or not isinstance(unmapped_sources, list):
        raise ValueError("prepared crypto source inventory is malformed")
    if any(not isinstance(source, dict) or source.get("mapping_role") != "crypto" for source in sources):
        raise ValueError("prepared crypto section contains a foreign-role source")
    if any(not isinstance(source, dict) or source.get("mapping_role") != "unknown"
           for source in unmapped_sources):
        raise ValueError("prepared crypto inventory contains a mapped foreign-role source")

    if part == "market" or part == "book":
        selected_sources = []
    elif part == "options_positioning":
        selected_sources = [source for source in sources
                            if source.get("source_id") in CRYPTO_OPTIONS_POSITIONING_IDS]
    else:
        selected_sources = [source for source in sources
                            if source.get("source_id") not in CRYPTO_OPTIONS_POSITIONING_IDS]

    def inventory_row(source):
        return {
            "source_id": source.get("source_id"),
            "status": source.get("status", "unknown"),
            "status_note": "not exposed in compact source projection" if "status" not in source else "supplied",
            "mapping_status": source.get("mapping_status", "unknown"),
            "metrics_total": _source_metric_count(source, "metrics_total"),
            "metrics_available": _source_metric_count(source, "metrics_available"),
            "metrics_returned": _source_metric_count(source, "metrics_returned"),
        }

    inventory = [inventory_row(source) for source in sources]
    inventory.extend(inventory_row(source) for source in unmapped_sources)
    selected_metric_count = sum(_source_metric_count(source, "metrics_returned") for source in selected_sources)
    unmapped_included = len(unmapped_sources) if part == "market" else 0
    unmapped_metrics_included = (
        sum(_source_metric_count(source, "metrics_returned") for source in unmapped_sources)
        if part == "market" else 0
    )
    result = {
        "part": part,
        "source_counts": {"crypto_total": len(sources), "crypto_selected": len(selected_sources),
                          "unmapped_total": len(unmapped_sources), "unmapped_included": unmapped_included},
        "metric_counts": {
            "crypto_total": sum(_source_metric_count(source, "metrics_total") for source in sources),
            "crypto_available": sum(_source_metric_count(source, "metrics_available") for source in sources),
            "crypto_returned": sum(_source_metric_count(source, "metrics_returned") for source in sources),
            "crypto_selected_returned": selected_metric_count,
            "unmapped_total": sum(_source_metric_count(source, "metrics_total") for source in unmapped_sources),
            "unmapped_included": unmapped_metrics_included,
        },
    }
    if part == "market":
        result.update({"source_inventory": inventory, "unmapped_sources": unmapped_sources,
                       "crypto_derivatives": section["crypto_derivatives"],
                       "higher_timeframes": section["higher_timeframes"]})
    elif part == "book":
        result.update({"orderbook": section["orderbook"], "book_pressure": section["book_pressure"]})
    else:
        projection_fields = {key: value for key, value in projection.items() if key != "sources"}
        result["global_projection"] = {**projection_fields, "sources": selected_sources,
                                       "source_count": len(selected_sources)}
    return result


def _section_payload(packet, role, expected_observed_at, part=None):
    if not isinstance(packet, dict) or not isinstance(packet.get("sections"), dict):
        raise ValueError("prepared reporter packet is malformed")
    if role not in REPORTER_SECTION_FIELDS:
        raise ValueError(f"unsupported reporter role: {role}")
    observed_at = packet.get("observed_at_utc")
    if _parse_clock(observed_at) != _parse_clock(expected_observed_at):
        raise ValueError("prepared packet observed_at_utc does not match the expected clock")
    section = packet["sections"].get(role)
    if not isinstance(section, dict):
        raise ValueError(f"prepared packet is missing its {role} section")
    unexpected = set(section) - REPORTER_SECTION_FIELDS[role]
    if unexpected:
        raise ValueError(f"prepared {role} section has unexpected fields: {', '.join(sorted(unexpected))}")
    if role == "news" and not isinstance(section.get("leads"), list):
        raise ValueError("prepared news section is malformed")
    if role == "news" and _parse_clock(section.get("observed_at_utc")) != _parse_clock(observed_at):
        raise ValueError("prepared news section observed_at_utc does not match the packet clock")
    if role == "global" and not isinstance(section.get("sources"), list):
        raise ValueError("prepared global section is malformed")
    if role == "crypto" and not isinstance(section.get("crypto_derivatives"), dict):
        raise ValueError("prepared crypto section is malformed")
    if part is not None:
        if role != "crypto":
            raise ValueError("--part is supported only for the crypto role")
        if part not in CRYPTO_SECTION_PARTS:
            raise ValueError(f"unsupported crypto section part: {part}")
        global_section = packet["sections"].get("global")
        if not isinstance(global_section, dict) or not isinstance(global_section.get("unmapped_sources"), list):
            raise ValueError("prepared global unmapped-source inventory is malformed")
        section = _crypto_part_payload(section, global_section["unmapped_sources"], part)
    return {"observed_at_utc": observed_at, role: section}


def _load_packet(path=REPORTER_PACKET_FILE):
    try:
        with path.open("r", encoding="utf-8") as handle:
            packet = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"prepared reporter packet is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"prepared reporter packet is malformed JSON: {path}") from exc
    return packet


def _prepare_summary(packet, path=REPORTER_PACKET_FILE):
    sections = packet["sections"]
    news = sections["news"]
    global_data = sections["global"]
    crypto = sections["crypto"]
    crypto_projection = crypto["global_projection"]
    return {
        "observed_at_utc": packet["observed_at_utc"], "packet_path": str(path.resolve()),
        "sections": {
            "news": {"leads": len(news.get("leads", [])), "coverage": news.get("coverage_status", "unknown")},
            "global": {"sources": len(global_data.get("sources", [])),
                       "unmapped_sources": len(global_data.get("unmapped_sources", [])),
                       "coverage": global_data.get("status", "unknown")},
            "crypto": {"sources": len(crypto_projection.get("sources", [])),
                       "position_sample_sources": sum(
                           source.get("source_id") in HYPERLIQUID_SAMPLE_IDS
                           for source in crypto_projection.get("sources", []) if isinstance(source, dict)),
                       "orderbook": crypto.get("orderbook", {}).get("status", "unknown"),
                       "derivatives": crypto.get("crypto_derivatives", {}).get("status", "unknown"),
                       "higher_timeframes": crypto.get("higher_timeframes", {}).get("status", "unknown")},
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("headlines", help="emit filtered collector leads and news/web freshness")
    commands.add_parser("market", help="emit compact numeric/source coverage")
    commands.add_parser("prepare", help="fetch once and freeze the three scoped reporter slices")
    section = commands.add_parser("section", help="emit one frozen reporter slice at an expected clock")
    section.add_argument("role", choices=("news", "global", "crypto"))
    section.add_argument("--observed-at-utc", required=True)
    section.add_argument("--part", choices=CRYPTO_SECTION_PARTS,
                         help="emit one bounded crypto subsection; omitted for the full role slice")
    commands.add_parser("publish", help="validate and publish a JSON Luna context from stdin")
    args = parser.parse_args(argv)
    if args.command == "headlines":
        print(json.dumps(headlines_packet(), ensure_ascii=True, allow_nan=False, separators=(",", ":")))
    elif args.command == "market":
        print(json.dumps(market_packet(), ensure_ascii=True, allow_nan=False, separators=(",", ":")))
    elif args.command == "prepare":
        packet = prepare_packet()
        atomic_write_text(REPORTER_PACKET_FILE,
                          json.dumps(packet, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n")
        print(json.dumps(_prepare_summary(packet), ensure_ascii=True, allow_nan=False, separators=(",", ":")))
    elif args.command == "section":
        packet = _load_packet()
        output = _section_payload(packet, args.role, args.observed_at_utc, args.part)
        print(json.dumps(output, ensure_ascii=True, allow_nan=False, separators=(",", ":")))
    else:
        row = json.load(sys.stdin)
        if not isinstance(row, dict):
            raise ValueError("Luna context input must be a JSON object")
        publish_luna_context(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

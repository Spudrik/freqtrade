"""Small news-only input and publication CLI for scheduled Luna briefs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import sys
from urllib.parse import urlparse

from user_data.Custom_Launcher.research.crypto_derivatives_snapshot import collect_crypto_derivatives_snapshot
from user_data.Custom_Launcher.research.paper_trial_snapshot import higher_snapshot, source_snapshot
from user_data.strategies.integrated_paper_context import publish_luna_context


ALLOWED_PUBLISHERS = ("apnews.com", "reuters.com", "theguardian.com", "aljazeera.com", "coindesk.com")
GLOBAL_METRICS = ("source_id", "source_group", "source_type", "url", "enabled", "last_success_at",
                  "last_failure_at", "last_observation_at", "last_collected_at")
DERIVATIVE_NUMBERS = ("coverage", "usable_metric_count", "advertised_metric_count", "oi_change_5m",
                      "oi_change_1h", "oi_age_minutes", "taker_imbalance_5m", "taker_imbalance_1h",
                      "taker_age_minutes", "funding_rate", "mark_index_premium", "premium_age_minutes",
                      "global_long_short_ratio", "global_long_short_ratio_change_1h", "positioning_age_minutes")
BREADTH_FIELDS = ("status", "sample", "requested", "coverage", "share_advancing", "median_pct",
                  "dispersion_mad_pct", "btc_pct", "median_minus_btc_pct", "observation_at_utc", "age_minutes")


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


def headlines_packet(now=None):
    now = now or _now()
    result = {"observed_at_utc": now.astimezone(timezone.utc).isoformat(),
              "news_web_freshness": {"news": _freshness(None), "web": _freshness(None, True)}, "leads": []}
    try:
        raw = source_snapshot(now)
    except Exception as exc:
        result.update(coverage_status="unknown", source_failure=type(exc).__name__)
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


def _global(raw):
    if not isinstance(raw, dict):
        return {"status": "unknown"}
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
            item.update({"metrics_total": source.get("metric_count", len(metrics)),
                         "metrics_returned": min(4, len(metrics)),
                         "metrics_truncated": bool(source.get("metrics_truncated")) or len(metrics) > 4})
            item["latest_metrics"] = []
            if metrics:
                for metric in metrics[:4]:
                    if not isinstance(metric, dict):
                        continue
                    item["latest_metrics"].append({"metric_key": metric.get("metric_key"),
                        "value": _number(metric.get("value")), "unit": metric.get("unit"),
                        "source_ts": metric.get("source_ts"), "collected_ts": metric.get("ts")})
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


def market_packet(now=None):
    now = now or _now()
    result = {"observed_at_utc": now.astimezone(timezone.utc).isoformat(), "global": {"status": "unknown"},
              "orderbook": {"status": "unknown"}, "book_pressure": {},
              "crypto_derivatives": {"status": "unknown"}, "higher_timeframes": {"status": "unknown", "windows": []}}
    try:
        raw = source_snapshot(now)
        if not isinstance(raw, dict):
            raise ValueError("invalid source_snapshot result")
        result["global"] = _global(raw.get("global"))
        result["orderbook"] = _source_clocks(raw.get("orderbook"))
        pressure = raw.get("book_pressure")
        if isinstance(pressure, dict):
            result["book_pressure"] = {pair: {"imbalance": _number(item.get("imbalance")),
                "coverage": _coverage(item.get("coverage"))} for pair, item in pressure.items() if isinstance(item, dict)}
    except Exception as exc:
        failure = {"status": "unknown", "failure": type(exc).__name__, "failure_detail": str(exc)[:160]}
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("headlines", help="emit filtered collector leads and news/web freshness")
    commands.add_parser("market", help="emit compact numeric/source coverage")
    commands.add_parser("publish", help="validate and publish a JSON Luna context from stdin")
    args = parser.parse_args(argv)
    if args.command == "headlines":
        print(json.dumps(headlines_packet(), ensure_ascii=True, allow_nan=False, separators=(",", ":")))
    elif args.command == "market":
        print(json.dumps(market_packet(), ensure_ascii=True, allow_nan=False, separators=(",", ":")))
    else:
        row = json.load(sys.stdin)
        if not isinstance(row, dict):
            raise ValueError("Luna context input must be a JSON object")
        publish_luna_context(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

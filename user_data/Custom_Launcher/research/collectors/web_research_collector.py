from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import os
import re
import sqlite3
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from .news_research_store import (
    apply_article_tags,
    apply_source_config_tags,
    article_id_from,
    connect_db,
    init_db,
    load_status,
    record_fetch,
    update_status,
    update_source_daily_stats,
    upsert_article,
    upsert_article_score,
    upsert_source_health,
)
from .news_research_scoring import detect_assets, score_article, tags_from_article
from .research_collector_common import (
    configure_logging,
    detect_event_type,
    estimate_restart_gap_hours,
    evaluate_startup_stop_file,
    load_json,
    normalize_text,
    parse_rss_items,
    save_json,
    save_raw_snapshot,
    sleep_with_stop,
    slugify,
    stop_requested,
    utc_now,
    utc_stamp,
)


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = THIS_DIR.parent / "config" / "web_research_sources.json"
DEFAULT_DATA_DIR = THIS_DIR.parents[2] / "research_news_data" / "web"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "web_events.sqlite"
DEFAULT_STATUS_PATH = DEFAULT_DATA_DIR / "collector_status.json"
DEFAULT_PID_PATH = DEFAULT_DATA_DIR / "collector.pid"
DEFAULT_STOP_PATH = DEFAULT_DATA_DIR / "collector.stop"
DEFAULT_LOG_PATH = DEFAULT_DATA_DIR / "logs" / "web_collector.log"

STARTER_CONFIG: dict[str, Any] = {
    "version": 1,
    "poll_interval_seconds": 21600,
    "request_timeout_seconds": 20,
    "max_items_per_source": 60,
    "store_raw_payloads": True,
    "user_agent": "FreQ-WebResearchCollector/1.0",
    "sources": [
        {
            "id": "binance_support_announcements",
            "type": "html_links",
            "enabled": True,
            "source_group": "exchange_announcements",
            "region": "global",
            "topic": "binance_announcements",
            "market_relevance": "high",
            "url": "https://www.binance.com/en/support/announcement",
            "include_url_substrings": ["/support/announcement/"],
            "exclude_url_substrings": ["/support/announcement?", "#"],
            "include_text_substrings": ["list", "delist", "launchpool", "futures", "margin", "airdrop", "earn", "support"],
            "exclude_text_substrings": ["announcement center", "view more"],
            "min_text_length": 18,
        },
        {
            "id": "bybit_announcements",
            "type": "html_links",
            "enabled": True,
            "source_group": "exchange_announcements",
            "region": "global",
            "topic": "bybit_announcements",
            "market_relevance": "high",
            "url": "https://announcements.bybit.com/en/",
            "include_url_substrings": ["/en/article/"],
            "exclude_url_substrings": ["#"],
            "include_text_substrings": ["list", "delist", "launch", "upgrade", "support", "futures", "margin", "staking"],
            "exclude_text_substrings": ["announcement"],
            "min_text_length": 18,
        },
        {
            "id": "kraken_asset_listings",
            "type": "html_links",
            "enabled": True,
            "source_group": "exchange_announcements",
            "region": "global",
            "topic": "kraken_asset_listings",
            "market_relevance": "high",
            "url": "https://blog.kraken.com/product/asset-listings",
            "include_url_substrings": ["/product/asset-listings/"],
            "exclude_url_substrings": ["#"],
            "include_text_substrings": ["available for trading", "listing", "trade"],
            "exclude_text_substrings": ["asset listings"],
            "min_text_length": 18,
        },
        {
            "id": "coinbase_blog",
            "type": "html_links",
            "enabled": True,
            "source_group": "exchange_announcements",
            "region": "us",
            "topic": "coinbase_blog",
            "market_relevance": "medium",
            "url": "https://www.coinbase.com/blog",
            "include_url_substrings": ["/blog/"],
            "exclude_url_substrings": ["#", "?"],
            "include_text_substrings": ["listing", "perpetual", "futures", "staking", "institutional", "usdc", "base", "launch"],
            "exclude_text_substrings": ["blog"],
            "min_text_length": 18,
        },
        {
            "id": "ethereum_foundation_blog",
            "type": "html_links",
            "enabled": True,
            "source_group": "protocol_foundations",
            "region": "global",
            "topic": "ethereum_foundation",
            "market_relevance": "medium",
            "url": "https://blog.ethereum.org/",
            "include_url_substrings": ["/20"],
            "exclude_url_substrings": ["#"],
            "include_text_substrings": ["ethereum", "staking", "defi", "protocol", "security", "upgrade"],
            "exclude_text_substrings": ["subscribe"],
            "min_text_length": 18,
        },
        {
            "id": "uniswap_blog",
            "type": "html_links",
            "enabled": True,
            "source_group": "defi_protocols",
            "region": "global",
            "topic": "uniswap",
            "market_relevance": "medium",
            "url": "https://blog.uniswap.org/",
            "include_url_substrings": ["/"],
            "exclude_url_substrings": ["#", "/tag/", "/author/", "/cdn-cgi/"],
            "include_text_substrings": ["uniswap", "governance", "v4", "api", "liquidity", "defi", "launch"],
            "exclude_text_substrings": ["newsletter", "subscribe"],
            "min_text_length": 18,
        },
        {
            "id": "chainlink_blog",
            "type": "html_links",
            "enabled": True,
            "source_group": "infrastructure",
            "region": "global",
            "topic": "chainlink",
            "market_relevance": "medium",
            "url": "https://blog.chain.link/",
            "include_url_substrings": ["/"],
            "exclude_url_substrings": ["#", "/tag/", "/author/", "/category/"],
            "include_text_substrings": ["chainlink", "ccip", "oracle", "stablecoin", "runtime", "defi"],
            "exclude_text_substrings": ["subscribe", "newsletter"],
            "min_text_length": 18,
        },
        {
            "id": "coindesk_markets_rss",
            "type": "rss",
            "enabled": True,
            "source_group": "crypto_media",
            "region": "global",
            "topic": "markets",
            "market_relevance": "high",
            "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        },
        {
            "id": "cointelegraph_rss",
            "type": "rss",
            "enabled": True,
            "source_group": "crypto_media",
            "region": "global",
            "topic": "news",
            "market_relevance": "high",
            "url": "https://cointelegraph.com/rss.xml",
        },
    ],
}

def ensure_runtime_dirs(data_dir: Path) -> dict[str, Path]:
    raw_root = data_dir / "raw"
    dirs = {
        "data": data_dir,
        "html": raw_root / "html",
        "rss": raw_root / "rss",
        "api": raw_root / "api",
        "manual": raw_root / "manual",
        "logs": data_dir / "logs",
        "exports": data_dir / "exports",
        "analysis": data_dir / "analysis",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def ensure_config(config_path: Path) -> None:
    if config_path.exists():
        return
    if DEFAULT_CONFIG_PATH.exists():
        config_path.write_text(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        return
    save_json(config_path, STARTER_CONFIG)


def open_url(url: str, timeout_seconds: int, user_agent: str) -> bytes:
    request = urllib_request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.8",
        },
        method="GET",
    )
    with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
        return response.read()


def raw_snapshot_path(raw_dir: Path, source: dict[str, Any], suffix: str) -> Path:
    return raw_dir / slugify(source.get("id") or "source") / f"{utc_stamp()}{suffix}"


def read_config(config_path: Path) -> dict[str, Any]:
    ensure_config(config_path)
    payload = load_json(config_path, {})
    if not isinstance(payload, dict):
        return {}
    payload.setdefault("version", 1)
    payload.setdefault("poll_interval_seconds", 21600)
    payload.setdefault("request_timeout_seconds", 20)
    payload.setdefault("max_items_per_source", 60)
    payload.setdefault("store_raw_payloads", True)
    payload.setdefault("user_agent", "FreQ-WebResearchCollector/1.0")
    payload.setdefault("sources", [])
    return payload


class AnchorCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._current_href: str | None = None
        self._current_text: list[str] = []
        self.links: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = ""
        for key, value in attrs:
            if key.lower() == "href":
                href = value or ""
                break
        self._current_href = href
        self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._current_href is None:
            return
        text = re.sub(r"\s+", " ", "".join(self._current_text)).strip()
        href = self._current_href.strip()
        if href:
            self.links.append({"href": href, "text": html.unescape(text)})
        self._current_href = None
        self._current_text = []


def normalize_base(source: dict[str, Any], source_type: str, collected_at: str, raw_file_path: str, raw_payload: Any) -> dict[str, Any]:
    return {
        "source_id": str(source.get("id") or ""),
        "source_group": source.get("source_group"),
        "source_type": source_type,
        "region": source.get("region"),
        "topic": source.get("topic"),
        "market_relevance": source.get("market_relevance"),
        "source_url": source.get("url") or source.get("query"),
        "collected_at": collected_at,
        "updated_at": collected_at,
        "raw_json": json.dumps(raw_payload, ensure_ascii=False, sort_keys=True),
        "raw_file_path": raw_file_path,
        "impact_direction": "unknown",
        "severity": None,
        "confidence": None,
    }


def normalize_rss_item(source: dict[str, Any], item: dict[str, Any], collected_at: str, raw_file_path: str) -> dict[str, Any]:
    text = normalize_text(item.get("title"), item.get("summary"), " ".join(item.get("categories") or []))
    assets = detect_assets(text)
    article = normalize_base(source, "rss", collected_at, raw_file_path, item)
    title = str(item.get("title") or "Untitled")
    categories = [str(tag) for tag in item.get("categories") or [] if str(tag).strip()]
    article.update(
        {
            "canonical_url": str(item.get("link") or ""),
            "title": title,
            "summary": str(item.get("summary") or ""),
            "published_at": item.get("published_at"),
            "language": item.get("language"),
            "detected_assets_json": json.dumps(assets),
            "detected_entities_json": json.dumps(categories),
            "event_type": detect_event_type(text),
            "impact_scope": ", ".join(assets) if assets else (source.get("source_group") or source.get("topic")),
            "detected_assets": assets,
            "detected_categories": categories,
            "guid": item.get("guid"),
        }
    )
    article["id"] = article_id_from(article["source_id"], item.get("guid"), article["canonical_url"], title, article.get("published_at"))
    return article


def normalize_html_item(source: dict[str, Any], item: dict[str, Any], collected_at: str, raw_file_path: str) -> dict[str, Any]:
    title = str(item.get("title") or "Untitled")
    summary = str(item.get("summary") or "")
    text = normalize_text(title, summary, source.get("topic"), source.get("source_group"))
    assets = detect_assets(text)
    categories = [str(source.get("source_group") or ""), str(source.get("topic") or "")]
    article = normalize_base(source, "html_links", collected_at, raw_file_path, item)
    article.update(
        {
            "canonical_url": str(item.get("url") or ""),
            "title": title,
            "summary": summary,
            "published_at": item.get("published_at"),
            "language": item.get("language"),
            "detected_assets_json": json.dumps(assets),
            "detected_entities_json": json.dumps(categories),
            "event_type": detect_event_type(text),
            "impact_scope": ", ".join(assets) if assets else (source.get("source_group") or source.get("topic")),
            "detected_assets": assets,
            "detected_categories": [value for value in categories if value],
            "guid": item.get("guid") or item.get("url"),
        }
    )
    article["id"] = article_id_from(article["source_id"], article.get("guid"), article["canonical_url"], title, article.get("published_at"))
    return article


def _matches_substrings(value: str, substrings: list[str]) -> bool:
    if not substrings:
        return True
    lowered = value.lower()
    return any(str(token).lower() in lowered for token in substrings if str(token).strip())


def _matches_no_substrings(value: str, substrings: list[str]) -> bool:
    lowered = value.lower()
    return not any(str(token).lower() in lowered for token in substrings if str(token).strip())


def extract_html_links(page_url: str, html_text: str, source: dict[str, Any], max_items: int) -> list[dict[str, Any]]:
    parser = AnchorCollector()
    parser.feed(html_text)

    include_url = [str(value) for value in source.get("include_url_substrings") or []]
    exclude_url = [str(value) for value in source.get("exclude_url_substrings") or []]
    include_text = [str(value) for value in source.get("include_text_substrings") or []]
    exclude_text = [str(value) for value in source.get("exclude_text_substrings") or []]
    min_text_length = int(source.get("min_text_length") or 12)

    items: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for link in parser.links:
        href = str(link.get("href") or "").strip()
        title = re.sub(r"\s+", " ", str(link.get("text") or "")).strip()
        if not href or href.startswith(("javascript:", "mailto:", "#")):
            continue
        resolved = urllib_parse.urljoin(page_url, href)
        if resolved in seen_urls:
            continue
        if len(title) < min_text_length:
            continue
        if not _matches_substrings(resolved, include_url):
            continue
        if not _matches_no_substrings(resolved, exclude_url):
            continue
        if include_text and not _matches_substrings(title, include_text):
            continue
        if not _matches_no_substrings(title, exclude_text):
            continue
        seen_urls.add(resolved)
        items.append(
            {
                "url": resolved,
                "title": title,
                "guid": resolved,
                "summary": f"Scraped from {page_url}",
                "published_at": None,
                "language": None,
            }
        )
        if len(items) >= max_items:
            break
    return items


def fetch_rss_source(
    source: dict[str, Any], config: dict[str, Any], raw_dir: Path, max_items_override: int | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    timeout_seconds = int(config.get("request_timeout_seconds") or 20)
    user_agent = str(config.get("user_agent") or "FreQ-WebResearchCollector/1.0")
    configured_max_items = int(config.get("max_items_per_source") or 60)
    max_items = max(1, int(max_items_override)) if max_items_override is not None else configured_max_items
    raw_payload = open_url(str(source.get("url") or ""), timeout_seconds, user_agent)
    raw_path = raw_snapshot_path(raw_dir, source, ".xml")
    if config.get("store_raw_payloads", True):
        save_raw_snapshot(raw_path, raw_payload)
    items, feed_lang = parse_rss_items(raw_payload)
    collected_at = utc_now()
    articles = [normalize_rss_item(source, item, collected_at, str(raw_path if config.get("store_raw_payloads", True) else "")) for item in items[:max_items]]
    return articles, {"http_status": 200, "raw_file_path": str(raw_path if config.get("store_raw_payloads", True) else ""), "language": feed_lang}


def fetch_html_links_source(source: dict[str, Any], config: dict[str, Any], raw_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    timeout_seconds = int(config.get("request_timeout_seconds") or 20)
    user_agent = str(config.get("user_agent") or "FreQ-WebResearchCollector/1.0")
    max_items = int(config.get("max_items_per_source") or 60)
    page_url = str(source.get("url") or "").strip()
    raw_payload = open_url(page_url, timeout_seconds, user_agent)
    raw_path = raw_snapshot_path(raw_dir, source, ".html")
    if config.get("store_raw_payloads", True):
        save_raw_snapshot(raw_path, raw_payload)
    html_text = raw_payload.decode("utf-8", "replace")
    items = extract_html_links(page_url, html_text, source, max_items)
    collected_at = utc_now()
    articles = [normalize_html_item(source, item, collected_at, str(raw_path if config.get("store_raw_payloads", True) else "")) for item in items]
    return articles, {"http_status": 200, "raw_file_path": str(raw_path if config.get("store_raw_payloads", True) else "")}


def fetch_source(
    source: dict[str, Any], config: dict[str, Any], raw_dirs: dict[str, Path], max_items_override: int | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_type = str(source.get("type") or "").strip().lower()
    if source_type == "rss":
        return fetch_rss_source(source, config, raw_dirs["rss"], max_items_override=max_items_override)
    if source_type == "html_links":
        return fetch_html_links_source(source, config, raw_dirs["html"])
    raise ValueError(f"Unsupported source type: {source_type}")


def update_status_payload(status_path: Path, payload: dict[str, Any]) -> None:
    update_status(status_path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone Web Lab collector")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--status-file", type=Path, default=DEFAULT_STATUS_PATH)
    parser.add_argument("--pid-file", type=Path, default=DEFAULT_PID_PATH)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--stop-file", type=Path, default=DEFAULT_STOP_PATH)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=21600)
    parser.add_argument("--max-items-per-source", type=int, default=60)
    args = parser.parse_args()

    raw_dirs = ensure_runtime_dirs(args.data_dir)
    configure_logging(args.log_file)
    ensure_config(args.config)

    previous_status = load_status(args.status_file)
    restart_gap_hours, restart_gap_source = estimate_restart_gap_hours(previous_status)
    restart_gap_warning: str | None = None
    if restart_gap_hours is not None and restart_gap_hours > 2.0:
        restart_gap_warning = (
            "Collector appears to have been offline for approximately "
            f"{restart_gap_hours:.2f} hours. Missed articles will only be recovered if the source still exposes "
            "them in the current feed/page/API response."
        )
        logging.warning("%s (reference timestamp: %s)", restart_gap_warning, restart_gap_source or "unknown")

    stop_startup = evaluate_startup_stop_file(args.stop_file, stale_after_seconds=300)
    if stop_startup.get("cleared_stale_stop_file"):
        logging.info("Cleared stale stop file before startup: %s", args.stop_file)
    if stop_startup.get("fresh_stop_requested"):
        startup_time = utc_now()
        status_payload = {
            "status": "stopped",
            "status_reason": "user_requested",
            "pid": None,
            "started_at": startup_time,
            "stopped_at": startup_time,
            "heartbeat_at": startup_time,
            "last_error": None,
            "db_path": str(args.db),
            "config_path": str(args.config),
            "data_dir": str(args.data_dir),
            "cleared_stale_stop_file": bool(stop_startup.get("cleared_stale_stop_file")),
            "cleared_stale_stop_file_at": stop_startup.get("cleared_stale_stop_file_at"),
            "restart_gap_hours": restart_gap_hours,
            "restart_gap_warning": restart_gap_warning,
            "restart_catchup_attempted": False,
            "restart_catchup_mode": "simple_current_feed_recheck",
            "restart_catchup_notes": "Startup skipped because a fresh stop file was present.",
        }
        update_status_payload(args.status_file, status_payload)
        logging.info("Fresh stop file detected at startup. Exiting without starting collector loop.")
        return 0

    try:
        init_db(args.db)
        args.pid_file.parent.mkdir(parents=True, exist_ok=True)
        args.pid_file.write_text(str(os.getpid()), encoding="utf-8")
    except Exception as exc:
        startup_time = utc_now()
        startup_payload = {
            "status": "error",
            "status_reason": "startup_error",
            "pid": None,
            "started_at": startup_time,
            "stopped_at": startup_time,
            "heartbeat_at": startup_time,
            "last_error": str(exc),
            "db_path": str(args.db),
            "config_path": str(args.config),
            "data_dir": str(args.data_dir),
            "cleared_stale_stop_file": bool(stop_startup.get("cleared_stale_stop_file")),
            "cleared_stale_stop_file_at": stop_startup.get("cleared_stale_stop_file_at"),
            "restart_gap_hours": restart_gap_hours,
            "restart_gap_warning": restart_gap_warning,
            "restart_catchup_attempted": False,
            "restart_catchup_mode": "simple_current_feed_recheck",
            "restart_catchup_notes": "Startup failed before entering the collection loop.",
        }
        try:
            update_status_payload(args.status_file, startup_payload)
        except Exception:
            pass
        logging.exception("Collector startup failed")
        return 1

    run_id = hashlib.sha256(f"{os.getpid()}|{utc_now()}|web".encode("utf-8")).hexdigest()
    started_at = utc_now()
    conn = connect_db(args.db)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO collector_runs(run_id, started_at, stopped_at, status, pid, config_path, db_path, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, started_at, None, "running", os.getpid(), str(args.config), str(args.db), None),
        )
        conn.commit()
    finally:
        conn.close()

    status_payload: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "status_reason": "normal_exit",
        "pid": os.getpid(),
        "started_at": started_at,
        "heartbeat_at": started_at,
        "last_cycle_finished_at": None,
        "last_fetch_at": None,
        "total_articles": 0,
        "new_articles_last_cycle": 0,
        "last_error": None,
        "db_path": str(args.db),
        "config_path": str(args.config),
        "data_dir": str(args.data_dir),
        "cleared_stale_stop_file": bool(stop_startup.get("cleared_stale_stop_file")),
        "cleared_stale_stop_file_at": stop_startup.get("cleared_stale_stop_file_at"),
        "restart_gap_hours": restart_gap_hours,
        "restart_gap_warning": restart_gap_warning,
        "restart_catchup_attempted": False,
        "restart_catchup_mode": "simple_current_feed_recheck",
        "restart_catchup_notes": None,
    }
    update_status_payload(args.status_file, status_payload)
    logging.info("Web collector started pid=%s config=%s db=%s", os.getpid(), args.config, args.db)

    terminal_status = "stopped"
    status_reason = "normal_exit"
    terminal_notes: str | None = None
    first_cycle = True
    try:
        while True:
            if stop_requested(args.stop_file):
                logging.info("Stop requested before cycle start.")
                status_reason = "user_requested"
                break
            config = read_config(args.config)
            cycle_interval = int(config.get("poll_interval_seconds") or args.interval_seconds or 21600)
            max_items = int(config.get("max_items_per_source") or args.max_items_per_source or 60)
            sources = [source for source in config.get("sources") or [] if isinstance(source, dict) and source.get("enabled", True)]
            cycle_catchup = bool(first_cycle and (restart_gap_hours or 0.0) > 2.0)
            if cycle_catchup:
                status_payload.update(
                    {
                        "restart_catchup_attempted": True,
                        "restart_catchup_mode": "simple_current_feed_recheck",
                        "restart_catchup_notes": "First cycle after downtime: RSS max items temporarily increased where safe.",
                    }
                )
                update_status_payload(args.status_file, status_payload)
            cycle_started_at = utc_now()
            cycle_inserted = 0
            cycle_error: str | None = None
            conn = connect_db(args.db)
            try:
                for source in sources:
                    source_id = str(source.get("id") or "source")
                    source_type = str(source.get("type") or "").lower()
                    try:
                        try:
                            apply_source_config_tags(conn, source)
                        except Exception:
                            logging.exception("Source tagging failed for %s", source_id)
                        source_max_items = max_items
                        max_items_override: int | None = None
                        if cycle_catchup and source_type == "rss":
                            source_max_items = min(100, max(max_items, max_items * 2))
                            max_items_override = source_max_items
                        articles, meta = fetch_source(source, config, raw_dirs, max_items_override=max_items_override)
                        articles = articles[:source_max_items]
                        fetched_count = len(articles)
                        inserted_count = 0
                        duplicate_count = 0
                        for article in articles:
                            try:
                                inserted = upsert_article(conn, article)
                                article["_is_duplicate"] = not inserted
                                if inserted:
                                    inserted_count += 1
                                else:
                                    duplicate_count += 1
                                try:
                                    article_id = str(article.get("id") or "")
                                    apply_article_tags(conn, article_id, tags_from_article(article, source))
                                    upsert_article_score(conn, article_id, score_article(article, source))
                                except Exception:
                                    logging.exception("Scoring/tagging failed for article from %s", source_id)
                            except sqlite3.OperationalError as exc:
                                logging.exception("DB error while upserting article for %s", source_id)
                                cycle_error = str(exc)
                        upsert_source_health(
                            conn,
                            source,
                            {
                                "last_success_at": utc_now(),
                                "last_failure_at": None,
                                "last_error": None,
                                "last_http_status": meta.get("http_status"),
                                "items_last_fetch": fetched_count,
                                "inserted_last_fetch": inserted_count,
                                "duplicates_last_fetch": duplicate_count,
                                "avg_items_per_day": None,
                            },
                        )
                        record_fetch(
                            conn,
                            {
                                "source_id": source_id,
                                "source_type": source_type,
                                "started_at": cycle_started_at,
                                "finished_at": utc_now(),
                                "status": "success",
                                "fetched_count": fetched_count,
                                "inserted_count": inserted_count,
                                "duplicate_count": duplicate_count,
                                "error_message": None,
                            },
                        )
                        try:
                            update_source_daily_stats(conn, source_id, cycle_started_at[:10])
                        except Exception:
                            logging.exception("Daily source stats update failed for %s", source_id)
                        conn.commit()
                        cycle_inserted += inserted_count
                        status_payload.update(
                            {
                                "heartbeat_at": utc_now(),
                                "last_fetch_at": utc_now(),
                                "last_error": None,
                                "new_articles_last_cycle": inserted_count,
                                "total_articles": int(conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]),
                            }
                        )
                        update_status_payload(args.status_file, status_payload)
                    except Exception as exc:
                        err_text = str(exc)
                        logging.exception("Source fetch failed for %s", source_id)
                        cycle_error = err_text
                        upsert_source_health(
                            conn,
                            source,
                            {
                                "last_success_at": None,
                                "last_failure_at": utc_now(),
                                "last_error": err_text,
                                "last_http_status": None,
                                "items_last_fetch": 0,
                                "inserted_last_fetch": 0,
                                "duplicates_last_fetch": 0,
                                "avg_items_per_day": None,
                            },
                        )
                        record_fetch(
                            conn,
                            {
                                "source_id": source_id,
                                "source_type": source_type,
                                "started_at": cycle_started_at,
                                "finished_at": utc_now(),
                                "status": "error",
                                "fetched_count": 0,
                                "inserted_count": 0,
                                "duplicate_count": 0,
                                "error_message": err_text,
                            },
                        )
                        try:
                            update_source_daily_stats(conn, source_id, cycle_started_at[:10])
                        except Exception:
                            logging.exception("Daily source stats update failed for %s", source_id)
                        conn.commit()
                        status_payload.update(
                            {
                                "heartbeat_at": utc_now(),
                                "last_error": err_text,
                                "new_articles_last_cycle": 0,
                                "total_articles": int(conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]),
                            }
                        )
                        update_status_payload(args.status_file, status_payload)

                total_articles = int(conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0])
                status_payload.update(
                    {
                        "heartbeat_at": utc_now(),
                        "last_cycle_finished_at": utc_now(),
                        "total_articles": total_articles,
                        "new_articles_last_cycle": cycle_inserted,
                        "last_error": cycle_error,
                    }
                )
                update_status_payload(args.status_file, status_payload)
                conn.execute(
                    "UPDATE collector_runs SET status = ?, stopped_at = ?, notes = ? WHERE run_id = ?",
                    ("running", None, cycle_error, run_id),
                )
                conn.commit()
            finally:
                conn.close()

            if args.once:
                status_reason = "once_complete"
                break
            if stop_requested(args.stop_file):
                status_reason = "user_requested"
                break
            if sleep_with_stop(args.stop_file, cycle_interval):
                status_reason = "user_requested"
                break
            first_cycle = False
    except KeyboardInterrupt:
        status_reason = "keyboard_interrupt"
    except SystemExit as exc:
        status_reason = "system_exit"
        terminal_notes = str(exc) if str(exc) else None
    except Exception as exc:
        logging.exception("Collector crashed")
        terminal_status = "error"
        status_reason = "runtime_error"
        terminal_notes = str(exc)
        status_payload.update({"status": "error", "status_reason": status_reason, "last_error": str(exc), "heartbeat_at": utc_now()})
        update_status_payload(args.status_file, status_payload)
        conn = connect_db(args.db)
        try:
            conn.execute(
                "UPDATE collector_runs SET status = ?, stopped_at = ?, notes = ? WHERE run_id = ?",
                ("error", utc_now(), str(exc), run_id),
            )
            conn.commit()
        finally:
            conn.close()
        return 1
    finally:
        try:
            args.pid_file.unlink(missing_ok=True)
        except Exception:
            pass
        if args.stop_file.exists():
            try:
                args.stop_file.unlink()
            except Exception:
                pass
        status_payload.update(
            {
                "status": terminal_status,
                "status_reason": status_reason,
                "stopped_at": utc_now(),
                "heartbeat_at": utc_now(),
                "notes": terminal_notes,
            }
        )
        update_status_payload(args.status_file, status_payload)
        conn = connect_db(args.db)
        try:
            conn.execute(
                "UPDATE collector_runs SET status = ?, stopped_at = ? WHERE run_id = ?",
                (terminal_status, utc_now(), run_id),
            )
            conn.commit()
        finally:
            conn.close()
        logging.info("Web collector exited with status=%s", terminal_status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

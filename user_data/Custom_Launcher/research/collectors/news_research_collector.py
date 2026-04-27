from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
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
DEFAULT_CONFIG_PATH = THIS_DIR.parent / "config" / "news_research_sources.json"
DEFAULT_DATA_DIR = THIS_DIR.parents[2] / "research_news_data" / "news"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "news_events.sqlite"
DEFAULT_STATUS_PATH = DEFAULT_DATA_DIR / "collector_status.json"
DEFAULT_PID_PATH = DEFAULT_DATA_DIR / "collector.pid"
DEFAULT_STOP_PATH = DEFAULT_DATA_DIR / "collector.stop"
DEFAULT_LOG_PATH = DEFAULT_DATA_DIR / "logs" / "news_collector.log"

def ensure_runtime_dirs(data_dir: Path) -> dict[str, Path]:
    raw_root = data_dir / "raw"
    dirs = {
        "data": data_dir,
        "rss": raw_root / "rss",
        "gdelt": raw_root / "gdelt",
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
    else:
        save_json(
            config_path,
            {
                "version": 1,
                "poll_interval_seconds": 900,
                "request_timeout_seconds": 20,
                "max_items_per_source": 100,
                "store_raw_payloads": True,
                "user_agent": "FreQ-NewsResearchCollector/1.0",
                "sources": [],
            },
        )


def open_url(url: str, timeout_seconds: int, user_agent: str) -> bytes:
    request = urllib_request.Request(url, headers={"User-Agent": user_agent}, method="GET")
    with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
        return response.read()


def raw_snapshot_path(raw_dir: Path, source: dict[str, Any], suffix: str) -> Path:
    source_type = slugify(source.get("type") or "api")
    source_id = slugify(source.get("id") or "source")
    return raw_dir / source_type / source_id / f"{utc_stamp()}{suffix}"


def read_config(config_path: Path) -> dict[str, Any]:
    ensure_config(config_path)
    payload = load_json(config_path, {})
    if not isinstance(payload, dict):
        return {}
    payload.setdefault("version", 1)
    payload.setdefault("poll_interval_seconds", 900)
    payload.setdefault("request_timeout_seconds", 20)
    payload.setdefault("max_items_per_source", 100)
    payload.setdefault("store_raw_payloads", True)
    payload.setdefault("user_agent", "FreQ-NewsResearchCollector/1.0")
    payload.setdefault("sources", [])
    return payload


def save_status_with_defaults(status_path: Path, payload: dict[str, Any]) -> None:
    update_status(status_path, payload)


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
    article.update(
        {
            "canonical_url": str(item.get("link") or ""),
            "title": title,
            "summary": str(item.get("summary") or ""),
            "published_at": item.get("published_at"),
            "language": item.get("language"),
            "detected_assets_json": json.dumps(assets),
            "detected_entities_json": json.dumps(item.get("categories") or []),
            "event_type": detect_event_type(text),
            "impact_scope": ", ".join(assets) if assets else (source.get("source_group") or source.get("topic")),
            "detected_assets": assets,
            "detected_categories": list(item.get("categories") or []),
            "guid": item.get("guid"),
        }
    )
    article["id"] = article_id_from(article["source_id"], item.get("guid"), article["canonical_url"], title, article.get("published_at"))
    return article


def normalize_gdelt_item(source: dict[str, Any], item: dict[str, Any], collected_at: str, raw_file_path: str) -> dict[str, Any]:
    title = str(item.get("title") or "Untitled")
    summary = str(item.get("seendate") or item.get("snippet") or item.get("excerpt") or "")
    text = normalize_text(title, summary, item.get("domain"), item.get("language"))
    assets = detect_assets(text)
    article = normalize_base(source, "gdelt_doc", collected_at, raw_file_path, item)
    article.update(
        {
            "canonical_url": str(item.get("url") or ""),
            "title": title,
            "summary": summary,
            "published_at": item.get("seendate") or item.get("published_at"),
            "language": item.get("language"),
            "detected_assets_json": json.dumps(assets),
            "detected_entities_json": json.dumps([item.get("domain"), item.get("sourceCountry")]),
            "event_type": detect_event_type(text),
            "impact_scope": ", ".join(assets) if assets else (source.get("source_group") or source.get("topic")),
            "detected_assets": assets,
            "detected_categories": [str(item.get("domain") or ""), str(item.get("sourceCountry") or "")],
            "guid": item.get("guid") or item.get("url"),
        }
    )
    article["id"] = article_id_from(article["source_id"], article.get("guid"), article["canonical_url"], title, article.get("published_at"))
    return article


def normalize_cryptopanic_item(source: dict[str, Any], item: dict[str, Any], collected_at: str, raw_file_path: str) -> dict[str, Any]:
    title = str(item.get("title") or "Untitled")
    summary = str(item.get("body") or item.get("metadata", {}).get("description") or "")
    text = normalize_text(title, summary)
    assets = detect_assets(text)
    article = normalize_base(source, "cryptopanic", collected_at, raw_file_path, item)
    article.update(
        {
            "canonical_url": str(item.get("url") or ""),
            "title": title,
            "summary": summary,
            "published_at": item.get("published_at") or item.get("created_at"),
            "language": item.get("language"),
            "detected_assets_json": json.dumps(assets),
            "detected_entities_json": json.dumps(item.get("currencies") or []),
            "event_type": detect_event_type(text),
            "impact_scope": ", ".join(assets) if assets else (source.get("source_group") or source.get("topic")),
            "detected_assets": assets,
            "detected_categories": [str(currency.get("code") or "") for currency in item.get("currencies") or [] if isinstance(currency, dict)],
            "guid": item.get("id") or item.get("uuid") or item.get("url"),
        }
    )
    article["id"] = article_id_from(article["source_id"], article.get("guid"), article["canonical_url"], title, article.get("published_at"))
    return article


def fetch_rss_source(
    source: dict[str, Any], config: dict[str, Any], raw_dir: Path, max_items_override: int | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    timeout_seconds = int(config.get("request_timeout_seconds") or 20)
    user_agent = str(config.get("user_agent") or "FreQ-NewsResearchCollector/1.0")
    configured_max_items = int(config.get("max_items_per_source") or 100)
    max_items = max(1, int(max_items_override)) if max_items_override is not None else configured_max_items
    raw_payload = open_url(str(source.get("url") or ""), timeout_seconds, user_agent)
    raw_path = raw_snapshot_path(raw_dir, source, ".xml")
    if config.get("store_raw_payloads", True):
        save_raw_snapshot(raw_path, raw_payload)
    items, feed_lang = parse_rss_items(raw_payload)
    collected_at = utc_now()
    articles = [normalize_rss_item(source, item, collected_at, str(raw_path if config.get("store_raw_payloads", True) else "")) for item in items[:max_items]]
    return articles, {"http_status": 200, "raw_file_path": str(raw_path if config.get("store_raw_payloads", True) else ""), "language": feed_lang}


def fetch_gdelt_doc_source(source: dict[str, Any], config: dict[str, Any], raw_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    timeout_seconds = int(config.get("request_timeout_seconds") or 20)
    user_agent = str(config.get("user_agent") or "FreQ-NewsResearchCollector/1.0")
    max_items = int(config.get("max_items_per_source") or 100)
    params = {
        "query": str(source.get("query") or ""),
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(max_items),
        "sort": "DateDesc",
    }
    url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urllib_parse.urlencode(params)
    raw_payload = open_url(url, timeout_seconds, user_agent)
    raw_path = raw_snapshot_path(raw_dir, source, ".json")
    if config.get("store_raw_payloads", True):
        save_raw_snapshot(raw_path, raw_payload)
    payload = json.loads(raw_payload.decode("utf-8", "replace"))
    items = payload.get("articles") or payload.get("response", {}).get("articles") or []
    collected_at = utc_now()
    articles = [normalize_gdelt_item(source, item, collected_at, str(raw_path if config.get("store_raw_payloads", True) else "")) for item in items[:max_items] if isinstance(item, dict)]
    http_status_raw = payload.get("status")
    try:
        http_status = int(http_status_raw) if http_status_raw is not None else 200
    except Exception:
        http_status = 200
    return articles, {"http_status": http_status, "raw_file_path": str(raw_path if config.get("store_raw_payloads", True) else "")}


def fetch_cryptopanic_source(source: dict[str, Any], config: dict[str, Any], raw_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    token = str(source.get("auth_token") or os.environ.get("CRYPTOPANIC_AUTH_TOKEN") or "").strip()
    if not token:
        logging.warning("Skipping CryptoPanic source %s because no auth token was provided.", source.get("id") or "unknown")
        return [], {"http_status": None, "warning": "missing auth token"}
    timeout_seconds = int(config.get("request_timeout_seconds") or 20)
    user_agent = str(config.get("user_agent") or "FreQ-NewsResearchCollector/1.0")
    max_items = int(config.get("max_items_per_source") or 100)
    params = {"auth_token": token, "public": "true", "kind": "news"}
    url = "https://cryptopanic.com/api/v1/posts/?" + urllib_parse.urlencode(params)
    raw_payload = open_url(url, timeout_seconds, user_agent)
    raw_path = raw_snapshot_path(raw_dir, source, ".json")
    if config.get("store_raw_payloads", True):
        save_raw_snapshot(raw_path, raw_payload)
    payload = json.loads(raw_payload.decode("utf-8", "replace"))
    items = payload.get("results") or payload.get("posts") or []
    collected_at = utc_now()
    articles = [normalize_cryptopanic_item(source, item, collected_at, str(raw_path if config.get("store_raw_payloads", True) else "")) for item in items[:max_items] if isinstance(item, dict)]
    return articles, {"http_status": 200, "raw_file_path": str(raw_path if config.get("store_raw_payloads", True) else "")}


def fetch_source(
    source: dict[str, Any], config: dict[str, Any], raw_dirs: dict[str, Path], max_items_override: int | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_type = str(source.get("type") or "").strip().lower()
    if source_type == "rss":
        return fetch_rss_source(source, config, raw_dirs["rss"], max_items_override=max_items_override)
    if source_type == "gdelt_doc":
        return fetch_gdelt_doc_source(source, config, raw_dirs["gdelt"])
    if source_type == "cryptopanic":
        return fetch_cryptopanic_source(source, config, raw_dirs["api"])
    raise ValueError(f"Unsupported source type: {source_type}")


def update_status_payload(status_path: Path, payload: dict[str, Any]) -> None:
    save_status_with_defaults(status_path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone News Lab collector")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--status-file", type=Path, default=DEFAULT_STATUS_PATH)
    parser.add_argument("--pid-file", type=Path, default=DEFAULT_PID_PATH)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--stop-file", type=Path, default=DEFAULT_STOP_PATH)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=900)
    parser.add_argument("--max-items-per-source", type=int, default=100)
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

    run_id = hashlib.sha256(f"{os.getpid()}|{utc_now()}".encode("utf-8")).hexdigest()
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
    logging.info("News collector started pid=%s config=%s db=%s", os.getpid(), args.config, args.db)

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
            cycle_interval = int(config.get("poll_interval_seconds") or args.interval_seconds or 900)
            max_items = int(config.get("max_items_per_source") or args.max_items_per_source or 100)
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
                                logging.exception("DB error while upserting article")
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
        logging.info("News collector exited with status=%s", terminal_status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

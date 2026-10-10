"""Bounded, read-only news packet scanner with an explicit post-review ack."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from urllib.parse import urlsplit, urlunsplit

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from user_data.Custom_Launcher.collector_runtime import atomic_write_text
from user_data.strategies.integrated_paper_context import _utc


ROOT = Path(__file__).resolve().parents[3]
STATE_PATH = ROOT / "user_data/research_news_data/context_features/integrated_paper_20260926/news_sentinel_state.json"
SOURCES = ("news", "web")
PAGE_LIMIT = 80
LOOKBACK = timedelta(minutes=60)
READ_LIMIT_EXTRA = 2
STATE_VERSION = 1


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _blank_state() -> dict:
    return {"schema_version": STATE_VERSION, "sources": {}, "recently_alerted_story_ids": [],
            "last_reported_coverage_fingerprint": None}


def _read_state(path: Path) -> tuple[dict, str]:
    if not path.is_file():
        return _blank_state(), "missing"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Sentinel state is unreadable; refusing to reset it: {path}: {exc}") from exc
    if (not isinstance(state, dict) or state.get("schema_version") != STATE_VERSION
            or not isinstance(state.get("sources"), dict)
            or not isinstance(state.get("recently_alerted_story_ids"), list)):
        raise RuntimeError(f"Sentinel state has an unsupported or malformed schema: {path}")
    return state, _digest(state)


def _clock(value: object, now: datetime) -> tuple[str, float | None, datetime | None]:
    if not isinstance(value, str) or not value.strip():
        return "missing", None, None
    try:
        parsed = _utc(value)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return "invalid", None, None
    age = (now.astimezone(timezone.utc) - parsed).total_seconds()
    if age < 0:
        return "future", None, parsed
    return "known", round(age / 60, 1), parsed


def _norm_title(value: str) -> str:
    return " ".join(value.casefold().split())


def _norm_url(value: str) -> str:
    try:
        parts = urlsplit(value.strip())
        if not parts.scheme or not parts.netloc:
            return ""
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, ""))
    except ValueError:
        return ""


def _file_identity(path: Path) -> tuple[str, str]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return f"{stat.st_dev}:{stat.st_ino}", _digest(str(resolved).casefold())


def _source_status_path(root: Path, source: str) -> Path:
    return root / "user_data" / "collector_data" / source / "collector_status.json"


def _load_object(path: Path, description: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _scan_source(root: Path, source: str, state_entry: dict | None, now: datetime, limit: int) -> dict:
    result = {"name": source, "issues": [], "rows": [], "update": None,
              "backlog": False, "rows_read": 0, "fetch_state": "unknown", "db_state": "unavailable"}
    issue = result["issues"].append
    status_path = _source_status_path(root, source)
    if not status_path.is_file():
        issue("status_missing")
        result["status_state"] = "missing"
        return result
    try:
        status = _load_object(status_path, "collector status")
    except ValueError:
        issue("status_invalid")
        result["status_state"] = "invalid"
        return result
    result["status_state"] = "available"
    if str(status.get("status", "")).casefold() != "running":
        issue("collector_not_running")
    if status.get("last_error"):
        issue("last_error_present")

    interval = None
    config_path_value = status.get("config_path")
    try:
        config_path = Path(config_path_value) if isinstance(config_path_value, str) else None
        if config_path is None or not config_path.is_absolute():
            raise ValueError("configured config_path is missing or not absolute")
        interval_value = _load_object(config_path, "collector config").get("poll_interval_seconds")
        if type(interval_value) is not int or interval_value <= 0:
            raise ValueError("poll_interval_seconds must be a positive integer")
        interval = interval_value
    except ValueError:
        issue("poll_cadence_unknown")
    result["poll_interval_seconds"] = interval
    fetch_state, fetch_age, fetch_at = _clock(status.get("last_fetch_at"), now)
    if interval is None:
        result["fetch_state"] = "unknown"
        issue("poll_cadence_unknown")
    elif fetch_state != "known":
        result["fetch_state"] = fetch_state
        issue(f"last_fetch_{fetch_state}")
    elif (now.astimezone(timezone.utc) - fetch_at).total_seconds() > 2 * interval + 60:
        result["fetch_state"] = "stale"
        issue("last_fetch_stale")
    else:
        result["fetch_state"] = "fresh"
    result["last_fetch_age_minutes"] = fetch_age

    db_value = status.get("db_path")
    try:
        db_path = Path(db_value) if isinstance(db_value, str) else None
        if db_path is None or not db_path.is_absolute():
            raise ValueError("configured db_path is missing or not absolute")
        identity, path_hash = _file_identity(db_path)
    except (OSError, ValueError):
        issue("database_missing_or_path_invalid")
        result["db_state"] = "missing"
        return result

    cursor = 0
    initial_complete = False
    cutoff = (now.astimezone(timezone.utc) - LOOKBACK).isoformat()
    if state_entry is not None:
        cursor = state_entry.get("checkpoint")
        if type(cursor) is not int or cursor < 0:
            issue("cursor_invalid")
            result["db_state"] = "cursor_mismatch"
            return result
        if state_entry.get("database_identity") != identity or state_entry.get("database_path_sha256") != path_hash:
            issue("database_reset")
            result["db_state"] = "cursor_mismatch"
            return result
        cutoff = state_entry.get("initial_cutoff_utc")
        initial_complete = state_entry.get("initial_complete") is True
        if not isinstance(cutoff, str):
            issue("cursor_invalid")
            result["db_state"] = "cursor_mismatch"
            return result

    try:
        uri = db_path.resolve(strict=True).as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=2)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only=ON")
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='articles'").fetchone() is None:
                raise sqlite3.DatabaseError("articles table is missing")
            max_rowid = int(conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM articles").fetchone()[0])
            if cursor and conn.execute("SELECT 1 FROM articles WHERE rowid=?", (cursor,)).fetchone() is None:
                issue("database_reset")
                result["db_state"] = "cursor_mismatch"
                return result
            fields = ("rowid AS _sentinel_rowid, substr(CAST(title AS TEXT),1,601) AS title, "
                      "length(CAST(title AS TEXT))>600 AS title_long, "
                      "substr(CAST(canonical_url AS TEXT),1,1201) AS canonical_url, "
                      "length(CAST(canonical_url AS TEXT))>1200 AS canonical_long, "
                      "substr(CAST(source_url AS TEXT),1,1201) AS source_url, "
                      "length(CAST(source_url AS TEXT))>1200 AS source_long, "
                      "substr(CAST(published_at AS TEXT),1,100) AS published_at, "
                      "substr(CAST(collected_at AS TEXT),1,100) AS collected_at, "
                      "substr(CAST(source_group AS TEXT),1,100) AS source_group")
            if initial_complete:
                sql, params = f"SELECT {fields} FROM articles WHERE rowid>? ORDER BY rowid LIMIT ?", (cursor, limit + READ_LIMIT_EXTRA)
            else:
                sql, params = (f"SELECT {fields} FROM articles WHERE rowid>? AND collected_at>=? ORDER BY rowid LIMIT ?",
                               (cursor, cutoff, limit + READ_LIMIT_EXTRA))
            rows = [dict(row) for row in conn.execute(sql, params).fetchall()]
    except sqlite3.Error:
        issue("database_unavailable_or_schema_invalid")
        result["db_state"] = "unavailable"
        return result

    result["db_state"] = "available"
    result["rows"] = [{**row, "_source": source} for row in rows]
    result["rows_read"] = len(rows)
    result["max_rowid"] = max_rowid
    result["start_checkpoint"] = cursor
    result["limit_reached"] = len(rows) == limit + READ_LIMIT_EXTRA
    result["update"] = {"checkpoint": cursor, "database_identity": identity,
                        "database_path_sha256": path_hash, "initial_cutoff_utc": cutoff,
                        "initial_complete": initial_complete}
    return result


def _evidence(row: dict, now: datetime) -> dict:
    pub_state, pub_age, _ = _clock(row.get("published_at"), now)
    col_state, col_age, _ = _clock(row.get("collected_at"), now)
    title = str(row.get("title") or "")
    url = str(row.get("canonical_url") or row.get("source_url") or "")
    return {"collector": row["_source"], "title": title[:600], "title_truncated": bool(row.get("title_long")),
            "url": url[:1200], "published_at": row.get("published_at"), "published_clock": pub_state,
            "published_age_minutes": pub_age, "collected_at": row.get("collected_at"),
            "collected_clock": col_state, "collected_age_minutes": col_age,
            "source_group": row.get("source_group")}


def scan_packet(root: Path = ROOT, state_path: Path = STATE_PATH, now: datetime | None = None,
                page_limit: int = PAGE_LIMIT) -> dict:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or type(page_limit) is not int or not 1 <= page_limit <= PAGE_LIMIT:
        raise ValueError("scan clock must be aware and page_limit must be 1..80")
    state, base_hash = _read_state(state_path)
    source_results = [_scan_source(root, name, state["sources"].get(name), now, page_limit) for name in SOURCES]
    candidates = [row for source in source_results for row in source["rows"]]
    candidates.sort(key=lambda row: (_clock(row.get("collected_at"), now)[2] or datetime.max.replace(tzinfo=timezone.utc),
                                    row["_source"], row["_sentinel_rowid"]))
    items, title_map, url_map, represented = [], {}, {}, set()
    omitted_unique = 0
    for row in candidates:
        evidence = _evidence(row, now)
        title_key = _norm_title(evidence["title"]) if evidence["title"] and not evidence["title_truncated"] else ""
        selected_url_is_long = row.get("canonical_long") if row.get("canonical_url") else row.get("source_long")
        url_key = _norm_url(evidence["url"]) if evidence["url"] and not selected_url_is_long else ""
        matches = {id(mapping[key]): mapping[key] for key, mapping in
                   ((title_key, title_map), (url_key, url_map)) if key and key in mapping}
        if matches:
            item = min(matches.values(), key=lambda found: items.index(found))
            for other in matches.values():
                if other is item:
                    continue
                item["evidence"].extend(other["evidence"])
                item["title_keys"].update(other["title_keys"])
                item["url_keys"].update(other["url_keys"])
                other["active"] = False
                for key in other["title_keys"]:
                    title_map[key] = item
                for key in other["url_keys"]:
                    url_map[key] = item
            item["evidence"].append(evidence)
            represented.add((row["_source"], row["_sentinel_rowid"]))
        elif len([item for item in items if item["active"]]) < page_limit:
            item = {"active": True, "evidence": [evidence], "title_keys": {title_key} if title_key else set(),
                    "url_keys": {url_key} if url_key else set()}
            items.append(item)
            if title_key:
                title_map[title_key] = item
            if url_key:
                url_map[url_key] = item
            represented.add((row["_source"], row["_sentinel_rowid"]))
        else:
            omitted_unique += 1

    updates, checkpoints, coverage, issue_codes = {}, {}, {}, {}
    for source in source_results:
        name = source["name"]
        update = source["update"]
        if update is not None:
            checkpoint = source["start_checkpoint"]
            for row in source["rows"]:
                if (name, row["_sentinel_rowid"]) not in represented:
                    break
                checkpoint = row["_sentinel_rowid"]
            if not source["rows"] and not update["initial_complete"]:
                # The first-scan window deliberately excludes older retained history.
                checkpoint = source["max_rowid"]
            backlog = source["limit_reached"] or checkpoint < (source["rows"][-1]["_sentinel_rowid"] if source["rows"] else checkpoint)
            update = {**update, "checkpoint": checkpoint, "initial_complete": update["initial_complete"] or not backlog,
                      "max_observed_rowid": source["max_rowid"]}
            updates[name] = update
            checkpoints[name] = checkpoint
            source["backlog"] = backlog
            if backlog:
                source["issues"].append("backlog")
        codes = sorted(set(source["issues"]))
        issue_codes[name] = codes
        coverage[name] = {"status_state": source.get("status_state", "unknown"),
                          "fetch_state": source["fetch_state"], "db_state": source["db_state"], "issue_codes": codes}
        coverage[name].update({"poll_interval_seconds": source.get("poll_interval_seconds"),
                               "last_fetch_age_minutes": source.get("last_fetch_age_minutes"),
                               "rows_read": source["rows_read"], "backlog": source["backlog"]})
    coverage_fingerprint = _digest(issue_codes)
    prev_coverage = state.get("last_reported_coverage_fingerprint")
    headlines = []
    alerted = set(state["recently_alerted_story_ids"])
    for item in items:
        if not item["active"]:
            continue
        primary, *additional = item["evidence"]
        seed = _norm_url(primary["url"]) or _norm_title(primary["title"])
        story_key = _digest(seed) if seed else ""
        headlines.append({**primary, "story_key": story_key, "previously_alerted": story_key in alerted,
                          "additional_sources": additional})
    ack_body = {"schema_version": STATE_VERSION, "base_state_sha256": base_hash, "checkpoints": checkpoints,
                "source_updates": updates, "coverage_fingerprint": coverage_fingerprint,
                "packet_fingerprint": _digest(sorted(represented)), "generated_at_utc": now.astimezone(timezone.utc).isoformat()}
    ack = {**ack_body, "scan_id": _digest(ack_body)}
    return {"schema_version": STATE_VERSION, "generated_at_utc": ack_body["generated_at_utc"],
            "initial_lookback_minutes": 60, "source_coverage": coverage,
            "coverage_fingerprint": coverage_fingerprint,
            "coverage_changed": prev_coverage != coverage_fingerprint,
            "coverage_attention": any(issue_codes.values()), "headlines": headlines,
            "recently_alerted_story_ids": state["recently_alerted_story_ids"][-64:],
            "headline_count": len(headlines), "omitted_unique_count": omitted_unique,
            "truncated": omitted_unique > 0 or any(source["backlog"] for source in source_results), "ack": ack}


def _record_story_keys(state: dict, keys: list[str]) -> None:
    known = list(state["recently_alerted_story_ids"])
    for key in keys:
        clean = " ".join(str(key).split())
        if not clean or len(clean) > 200:
            raise ValueError("story keys must be non-empty and at most 200 characters")
        if clean in known:
            known.remove(clean)
        known.append(clean)
    state["recently_alerted_story_ids"] = known[-64:]


def acknowledge(packet_ack: dict, root: Path, state_path: Path, story_keys: list[str] | None = None,
                delivery_confirmed: bool = False) -> dict:
    if not isinstance(packet_ack, dict) or packet_ack.get("schema_version") != STATE_VERSION:
        raise RuntimeError("ack packet schema is invalid")
    body = {key: value for key, value in packet_ack.items() if key != "scan_id"}
    if packet_ack.get("scan_id") != _digest(body):
        raise RuntimeError("ack packet scan_id does not match its returned checkpoints")
    checkpoints, updates = packet_ack.get("checkpoints"), packet_ack.get("source_updates")
    if not isinstance(checkpoints, dict) or not isinstance(updates, dict):
        raise RuntimeError("ack packet checkpoints/source_updates are malformed")
    if checkpoints != {name: value.get("checkpoint") for name, value in updates.items()}:
        raise RuntimeError("ack checkpoints differ from the exact returned source updates")
    story_keys = story_keys or []
    if story_keys and not delivery_confirmed:
        raise RuntimeError("story keys may be recorded only after confirmed alert delivery")
    state, current_hash = _read_state(state_path)
    if current_hash != packet_ack.get("base_state_sha256"):
        raise RuntimeError("sentinel cursor changed after this packet; refusing stale ack")
    for name, update in updates.items():
        if name not in SOURCES or type(update.get("checkpoint")) is not int:
            raise RuntimeError("ack source update is invalid")
        status = _load_object(_source_status_path(root, name), "collector status")
        db_path = Path(status.get("db_path", ""))
        identity, path_hash = _file_identity(db_path)
        if identity != update.get("database_identity") or path_hash != update.get("database_path_sha256"):
            raise RuntimeError(f"{name} database changed after scan; refusing ack")
        if update["checkpoint"] > update.get("max_observed_rowid", -1):
            raise RuntimeError(f"{name} ack is beyond the scanned rows")
        prior = state["sources"].get(name, {})
        if update["checkpoint"] < prior.get("checkpoint", 0):
            raise RuntimeError(f"{name} ack moves its checkpoint backwards")
        if update["checkpoint"]:
            uri = db_path.resolve(strict=True).as_uri() + "?mode=ro"
            with closing(sqlite3.connect(uri, uri=True, timeout=2)) as conn:
                if conn.execute("SELECT 1 FROM articles WHERE rowid=?", (update["checkpoint"],)).fetchone() is None:
                    raise RuntimeError(f"{name} checkpoint row disappeared; refusing ack")
    state["sources"].update({name: {key: value[key] for key in
                              ("checkpoint", "database_identity", "database_path_sha256", "initial_cutoff_utc", "initial_complete")}
                              for name, value in updates.items()})
    state["last_reported_coverage_fingerprint"] = packet_ack.get("coverage_fingerprint")
    _record_story_keys(state, story_keys)
    atomic_write_text(state_path, json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    return {"status": "acknowledged", "checkpoints": checkpoints, "story_keys_recorded": len(story_keys)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Read bounded collector headlines; ack only after Luna review.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("scan", help="read current unseen headlines and source coverage as JSON")
    ack_parser = sub.add_parser("ack", help="commit exact scan checkpoints after review")
    ack_parser.add_argument("--story-key", action="append", default=[])
    ack_parser.add_argument("--delivery-confirmed", action="store_true")
    args = parser.parse_args()
    if args.command == "scan":
        print(json.dumps(scan_packet(), ensure_ascii=False, separators=(",", ":")))
        return 0
    packet_ack = json.loads(sys.stdin.read())
    result = acknowledge(packet_ack, ROOT, STATE_PATH, args.story_key, args.delivery_confirmed)
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

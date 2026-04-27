from __future__ import annotations

import json
import logging
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .news_research_scoring import detect_event_tags


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip()).strip("_").lower()
    return cleaned or "source"


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return payload if isinstance(payload, type(default)) else default


def configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


def save_raw_snapshot(path: Path, payload: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(payload, encoding="utf-8")


def normalize_text(*parts: Any) -> str:
    return " ".join(str(part or "") for part in parts).lower()


def detect_event_type(text: str) -> str:
    event_tags = detect_event_tags(text)
    return event_tags[0] if event_tags else "unknown"


def parse_rss_items(content: bytes) -> tuple[list[dict[str, Any]], str | None]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid RSS/Atom XML: {exc}") from exc

    atom_ns = "{http://www.w3.org/2005/Atom}"
    items: list[dict[str, Any]] = []
    feed_lang = None
    channel = root.find("channel")
    if channel is not None:
        lang_node = channel.find("language")
        if lang_node is not None and lang_node.text:
            feed_lang = lang_node.text.strip()

    nodes = list(root.findall(f"{atom_ns}entry")) if root.tag.endswith("feed") else []
    if not nodes:
        nodes = list(channel.findall("item")) if channel is not None else list(root.findall(".//item"))

    for node in nodes:
        link = node.findtext("link") or ""
        if not link:
            link_el = node.find(f"{atom_ns}link")
            if link_el is not None:
                link = link_el.attrib.get("href", "")
        categories = [cat.text.strip() for cat in node.findall("category") if cat.text and cat.text.strip()]
        if not categories:
            categories = [cat.text.strip() for cat in node.findall(f"{atom_ns}category") if cat.text and cat.text.strip()]
        items.append(
            {
                "title": node.findtext("title") or "",
                "link": link,
                "guid": node.findtext("guid") or node.findtext(f"{atom_ns}id") or "",
                "published_at": node.findtext("pubDate")
                or node.findtext(f"{atom_ns}updated")
                or node.findtext(f"{atom_ns}published")
                or "",
                "summary": node.findtext("description")
                or node.findtext(f"{atom_ns}summary")
                or node.findtext(f"{atom_ns}content")
                or "",
                "categories": categories,
                "language": feed_lang,
            }
        )
    return items, feed_lang


def stop_requested(stop_file: Path) -> bool:
    return stop_file.exists()


def sleep_with_stop(stop_file: Path, seconds: int) -> bool:
    deadline = time.time() + max(1, seconds)
    while time.time() < deadline:
        if stop_requested(stop_file):
            return True
        time.sleep(min(5.0, max(0.5, deadline - time.time())))
    return stop_requested(stop_file)


def _parse_iso(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def evaluate_startup_stop_file(stop_file: Path, stale_after_seconds: int = 300) -> dict[str, Any]:
    result: dict[str, Any] = {
        "fresh_stop_requested": False,
        "cleared_stale_stop_file": False,
        "cleared_stale_stop_file_at": None,
    }
    if not stop_file.exists():
        return result
    try:
        age_seconds = max(0.0, time.time() - stop_file.stat().st_mtime)
        if age_seconds > max(0, stale_after_seconds):
            stop_file.unlink(missing_ok=True)
            result["cleared_stale_stop_file"] = True
            result["cleared_stale_stop_file_at"] = utc_now()
            return result
    except Exception:
        # If we cannot inspect/clear the file safely, treat it as a real stop request.
        pass
    result["fresh_stop_requested"] = True
    return result


def estimate_restart_gap_hours(previous_status: dict[str, Any]) -> tuple[float | None, str | None]:
    if not isinstance(previous_status, dict) or not previous_status:
        return None, None
    now_dt = datetime.now(timezone.utc)
    for key in ("heartbeat_at", "last_heartbeat_at", "stopped_at", "last_success_at", "started_at"):
        stamp = _parse_iso(previous_status.get(key))
        if stamp is None:
            continue
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        gap_hours = max(0.0, (now_dt - stamp).total_seconds() / 3600.0)
        return round(gap_hours, 2), key
    return None, None

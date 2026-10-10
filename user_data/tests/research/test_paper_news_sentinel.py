from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from user_data.Custom_Launcher.research.paper_news_sentinel import (
    acknowledge,
    scan_packet,
)


NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)


def _article(title: str, url: str, collected: str, published: str | None = "2026-10-09T11:45:00+00:00") -> tuple:
    return (title, url, url, published, collected, "market_plumbing")


def _source(root: Path, name: str, rows: list[tuple], *, last_fetch: str = "2026-10-09T11:50:00+00:00",
            poll_seconds: int = 900, db_exists: bool = True) -> Path:
    folder = root / "user_data" / "collector_data" / name
    folder.mkdir(parents=True, exist_ok=True)
    db = folder / f"{name}.sqlite"
    if db_exists:
        conn = sqlite3.connect(db)
        try:
            conn.execute("CREATE TABLE articles (id TEXT PRIMARY KEY, title TEXT, canonical_url TEXT, source_url TEXT, published_at TEXT, collected_at TEXT, source_group TEXT)")
            conn.executemany("INSERT INTO articles (title,canonical_url,source_url,published_at,collected_at,source_group) VALUES (?,?,?,?,?,?)", rows)
            conn.commit()
        finally:
            conn.close()
    config = root / f"{name}_config.json"
    config.write_text(json.dumps({"poll_interval_seconds": poll_seconds}), encoding="utf-8")
    status = {"status": "running", "config_path": str(config), "db_path": str(db), "last_fetch_at": last_fetch}
    (folder / "collector_status.json").write_text(json.dumps(status), encoding="utf-8")
    return db


def _append_article(db: Path, row: tuple) -> None:
    conn = sqlite3.connect(db)
    try:
        conn.execute("INSERT INTO articles (title,canonical_url,source_url,published_at,collected_at,source_group) VALUES (?,?,?,?,?,?)", row)
        conn.commit()
    finally:
        conn.close()


def _state(tmp_path: Path) -> Path:
    return tmp_path / "news_sentinel_state.json"


def test_first_scan_repeats_read_only_until_exact_ack_then_advances(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _source(root, "news", [_article("Bank event", "https://example.test/one", "2026-10-09T11:30:00+00:00")])
    state = _state(tmp_path)
    first = scan_packet(root, state, NOW)
    repeated = scan_packet(root, state, NOW)
    assert first["headline_count"] == repeated["headline_count"] == 1
    assert not state.exists()
    acknowledge(first["ack"], root, state)
    after_ack = scan_packet(root, state, NOW)
    assert after_ack["headlines"] == []


def test_empty_initial_lookback_ack_does_not_flood_retained_history(tmp_path: Path) -> None:
    root = tmp_path / "root"
    db = _source(root, "news", [_article("Old retained story", "https://example.test/old", "2026-10-09T10:30:00+00:00")])
    state = _state(tmp_path)

    bootstrap = scan_packet(root, state, NOW)
    assert bootstrap["headlines"] == []
    assert bootstrap["ack"]["checkpoints"] == {"news": 1}
    acknowledge(bootstrap["ack"], root, state)

    _append_article(db, _article("New story", "https://example.test/new", "2026-10-09T11:45:00+00:00"))
    next_poll = scan_packet(root, state, NOW)
    assert [item["title"] for item in next_poll["headlines"]] == ["New story"]
    assert next_poll["ack"]["checkpoints"] == {"news": 2}


def test_page_backlog_is_oldest_first_and_not_skipped_by_ack(tmp_path: Path) -> None:
    root = tmp_path / "root"
    rows = [_article(f"Event {i}", f"https://example.test/{i}", f"2026-10-09T11:{30+i:02}:00+00:00") for i in range(5)]
    _source(root, "news", rows)
    state = _state(tmp_path)
    pages = []
    for _ in range(3):
        packet = scan_packet(root, state, NOW, page_limit=2)
        pages.extend(item["title"] for item in packet["headlines"])
        acknowledge(packet["ack"], root, state)
    assert pages == ["Event 0", "Event 1", "Event 2", "Event 3", "Event 4"]


def test_publication_missing_invalid_and_future_clocks_are_explicit(tmp_path: Path) -> None:
    root = tmp_path / "root"
    rows = [
        _article("Missing date", "https://example.test/missing", "2026-10-09T11:30:00+00:00", None),
        _article("Bad date", "https://example.test/bad", "2026-10-09T11:31:00+00:00", "not-a-date"),
        _article("Future date", "https://example.test/future", "2026-10-09T11:32:00+00:00", "2026-10-09T13:00:00+00:00"),
    ]
    _source(root, "news", rows)
    packet = scan_packet(root, _state(tmp_path), NOW)
    clocks = {item["title"]: item["published_clock"] for item in packet["headlines"]}
    assert clocks == {"Missing date": "missing", "Bad date": "invalid", "Future date": "future"}


def test_stale_collector_missing_status_and_missing_db_are_visible_without_creation(tmp_path: Path) -> None:
    root = tmp_path / "root"
    news_db = _source(root, "news", [], last_fetch="2026-10-09T10:00:00+00:00")
    web_db = _source(root, "web", [], poll_seconds=21600, db_exists=False)
    (root / "user_data/collector_data/web/collector_status.json").unlink()
    packet = scan_packet(root, _state(tmp_path), NOW)
    assert packet["source_coverage"]["news"]["fetch_state"] == "stale"
    assert "status_missing" in packet["source_coverage"]["web"]["issue_codes"]
    assert news_db.is_file() and not web_db.exists()

    status = {"status": "running", "config_path": str(root / "web_config.json"),
              "db_path": str(web_db), "last_fetch_at": "2026-10-09T11:50:00+00:00"}
    (root / "user_data/collector_data/web/collector_status.json").write_text(json.dumps(status), encoding="utf-8")
    packet = scan_packet(root, _state(tmp_path), NOW)
    assert "database_missing_or_path_invalid" in packet["source_coverage"]["web"]["issue_codes"]
    assert not web_db.exists()


def test_title_url_dedup_preserves_second_collector_evidence(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _source(root, "news", [_article("  Exchange   halts withdrawals ", "HTTPS://Example.test/incident/", "2026-10-09T11:30:00+00:00")])
    _source(root, "web", [_article("exchange halts withdrawals", "https://example.test/incident", "2026-10-09T11:31:00+00:00")], poll_seconds=21600)
    packet = scan_packet(root, _state(tmp_path), NOW)
    assert packet["headline_count"] == 1
    assert packet["headlines"][0]["collector"] == "news"
    assert packet["headlines"][0]["additional_sources"][0]["collector"] == "web"
    assert packet["ack"]["checkpoints"] == {"news": 1, "web": 1}


def test_story_key_uses_canonical_url_across_title_changes(tmp_path: Path) -> None:
    root = tmp_path / "root"
    db = _source(root, "news", [_article("Initial title", "https://example.test/event", "2026-10-09T11:30:00+00:00")])
    state = _state(tmp_path)
    first = scan_packet(root, state, NOW)
    first_key = first["headlines"][0]["story_key"]
    assert first_key
    acknowledge(first["ack"], root, state, [first_key], delivery_confirmed=True)

    _append_article(db, _article("Changed title, same event", "HTTPS://EXAMPLE.TEST/event/", "2026-10-09T11:40:00+00:00"))
    second = scan_packet(root, state, NOW)
    assert len(second["headlines"]) == 1
    assert second["headlines"][0]["story_key"] == first_key
    assert second["headlines"][0]["previously_alerted"] is True


def test_readable_event_memory_is_visible_for_a_different_url(tmp_path: Path) -> None:
    root = tmp_path / "root"
    db = _source(root, "news", [_article("ExampleBank closure", "https://example.test/first", "2026-10-09T11:30:00+00:00")])
    state = _state(tmp_path)
    first = scan_packet(root, state, NOW)
    event_memory = "event:2026-10-09:ExampleBank regulator closure"
    acknowledge(first["ack"], root, state, [first["headlines"][0]["story_key"], event_memory], delivery_confirmed=True)

    _append_article(db, _article("Regulator closes ExampleBank", "https://example.test/second", "2026-10-09T11:40:00+00:00"))
    second = scan_packet(root, state, NOW)
    assert second["headlines"][0]["previously_alerted"] is False
    assert event_memory in second["recently_alerted_story_ids"]


def test_recently_alerted_story_memory_remains_capped_at_64(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _source(root, "news", [_article("Candidate", "https://example.test/event", "2026-10-09T11:30:00+00:00")])
    state = _state(tmp_path)
    packet = scan_packet(root, state, NOW)
    memories = [f"event:2026-10-09:ExampleBank update {index}" for index in range(66)]

    acknowledge(packet["ack"], root, state, memories, delivery_confirmed=True)
    saved = json.loads(state.read_text(encoding="utf-8"))["recently_alerted_story_ids"]
    next_poll = scan_packet(root, state, NOW)
    assert saved == memories[-64:]
    assert next_poll["recently_alerted_story_ids"] == memories[-64:]


def test_ack_requires_delivery_for_story_keys_and_rejects_stale_cursor_packet(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _source(root, "news", [_article("Candidate", "https://example.test/event", "2026-10-09T11:30:00+00:00")])
    state = _state(tmp_path)
    packet = scan_packet(root, state, NOW)
    with pytest.raises(RuntimeError, match="confirmed alert delivery"):
        acknowledge(packet["ack"], root, state, ["candidate-1"], False)
    acknowledge(packet["ack"], root, state, ["candidate-1"], True)
    with pytest.raises(RuntimeError, match="cursor changed"):
        acknowledge(packet["ack"], root, state)
    assert json.loads(state.read_text(encoding="utf-8"))["recently_alerted_story_ids"] == ["candidate-1"]


def test_database_replacement_rejects_existing_cursor(tmp_path: Path) -> None:
    root = tmp_path / "root"
    db = _source(root, "news", [
        _article("A", "https://example.test/a", "2026-10-09T11:30:00+00:00"),
        _article("B", "https://example.test/b", "2026-10-09T11:31:00+00:00"),
    ])
    state = _state(tmp_path)
    first = scan_packet(root, state, NOW)
    acknowledge(first["ack"], root, state)
    db.unlink()
    _source(root, "news", [_article("Replacement", "https://example.test/new", "2026-10-09T11:40:00+00:00")])
    packet = scan_packet(root, state, NOW)
    assert packet["headlines"] == []
    assert "database_reset" in packet["source_coverage"]["news"]["issue_codes"]

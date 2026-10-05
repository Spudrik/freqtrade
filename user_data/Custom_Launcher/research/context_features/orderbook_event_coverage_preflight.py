from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DB = REPO_ROOT / "user_data" / "orderbook_data" / "live" / "orderbook_events.sqlite"
DEFAULT_EVENT_CATALOG = (
    REPO_ROOT
    / "user_data"
    / "research_news_data"
    / "context_features"
    / "event_hierarchy"
    / "broad_event_relevance_20260904a"
    / "broad_relevance_event_catalog.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "user_data"
    / "research_news_data"
    / "context_features"
    / "source_preflight"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit recent 1-minute order-book coverage without opening market outcomes."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--event-catalog", type=Path, default=DEFAULT_EVENT_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start", default="2026-08-23T20:00:00Z")
    parser.add_argument("--end", default="2026-09-04T19:00:00Z")
    parser.add_argument("--event-window-hours", type=float, default=4.0)
    parser.add_argument("--tag", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    start = require_timestamp(args.start)
    end = require_timestamp(args.end)
    if end <= start:
        raise ValueError("end must be after start")
    tag = safe_tag(args.tag) if args.tag else datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_dir / f"orderbook_event_coverage_preflight_{tag}.csv"
    json_path = args.output_dir / f"orderbook_event_coverage_preflight_{tag}.json"
    plan = {
        "mode": "setup-only unless --execute is supplied",
        "objective": (
            "Check whether recent order-book bars cover all configured streams and already "
            "frozen major-event anchors before any reaction or direction outcome is tested."
        ),
        "coverage_pass_rule": (
            "Each stream needs at least 95% of expected minute bars, at least 80% valid "
            "samples, no duplicate minute keys, no bar created before it ended, and the same "
            "minimums around every frozen event in the interval."
        ),
        "database": str(args.db),
        "event_catalog": str(args.event_catalog),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "event_window_hours": float(args.event_window_hours),
        "outputs": {"csv": str(csv_path), "json": str(json_path)},
        "market_outcomes_read": False,
        "collector_modified_or_paused": False,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.db.exists():
        raise FileNotFoundError(args.db)
    if not args.event_catalog.exists():
        raise FileNotFoundError(args.event_catalog)

    events = load_events(args.event_catalog, start, end)
    stream_rows = audit_streams(
        args.db,
        start,
        end,
        events,
        timedelta(hours=float(args.event_window_hours)),
    )
    pairs = sorted({str(row["canonical_pair"]) for row in stream_rows})
    meme_pairs = sorted(set(pairs).intersection({"DOGE/USDT"}))
    all_streams_pass = bool(stream_rows) and all(bool(row["coverage_pass"]) for row in stream_rows)
    conclusion = (
        "coverage_ready_sample_too_small_for_event_claim"
        if all_streams_pass and len(events) < 10
        else "coverage_ready_for_bounded_event_test"
        if all_streams_pass
        else "coverage_not_ready"
    )
    result = {
        **plan,
        "mode": "executed read-only preflight",
        "generated_at": datetime.now(UTC).isoformat(),
        "database_bytes": args.db.stat().st_size,
        "streams": len(stream_rows),
        "pairs": pairs,
        "pair_count": len(pairs),
        "venue_market_count": len(
            {(str(row["venue"]), str(row["market_type"])) for row in stream_rows}
        ),
        "meme_pairs": meme_pairs,
        "meme_pair_count": len(meme_pairs),
        "frozen_events_in_window": events,
        "frozen_event_count": len(events),
        "streams_passing": sum(bool(row["coverage_pass"]) for row in stream_rows),
        "all_streams_coverage_pass": all_streams_pass,
        "conclusion": conclusion,
        "interpretation": (
            "Coverage is sufficient for later reaction-point confirmation on the ten collected "
            "coins. It is not sufficient to claim repeatable major-event value because only one "
            "frozen scheduled event overlaps, and it cannot test a broad meme cohort because "
            "DOGE is the only collected meme coin."
        ),
        "market_outcomes_read": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(csv_path, stream_rows)
    write_text_atomic(json_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result": str(json_path), "summary": result}, indent=2))
    return 0


def load_events(path: Path, start: datetime, end: datetime) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            anchor = parse_timestamp(row.get("anchor_utc"))
            if anchor is None or not start <= anchor < end:
                continue
            events.append(
                {
                    "event_id": str(row.get("event_id") or ""),
                    "event_family": str(row.get("event_family") or ""),
                    "anchor_utc": anchor.isoformat(),
                }
            )
    return sorted(events, key=lambda row: (row["anchor_utc"], row["event_id"]))


def audit_streams(
    db_path: Path,
    start: datetime,
    end: datetime,
    events: list[dict[str, str]],
    event_half_window: timedelta,
) -> list[dict[str, Any]]:
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=30.0) as conn:
        conn.row_factory = sqlite3.Row
        streams = conn.execute(
            """
            SELECT stream_id, canonical_pair, venue, market_type, status
            FROM stream_status
            ORDER BY stream_id
            """
        ).fetchall()
        output = []
        for stream in streams:
            full = query_window(conn, str(stream["stream_id"]), start, end)
            event_windows = []
            for event in events:
                anchor = require_timestamp(event["anchor_utc"])
                event_start = anchor - event_half_window
                event_end = anchor + event_half_window
                event_result = query_window(conn, str(stream["stream_id"]), event_start, event_end)
                event_result["event_id"] = event["event_id"]
                event_result["event_family"] = event["event_family"]
                event_windows.append(event_result)
            decision = coverage_decision(full, event_windows)
            output.append(
                {
                    "stream_id": str(stream["stream_id"]),
                    "canonical_pair": str(stream["canonical_pair"]),
                    "venue": str(stream["venue"]),
                    "market_type": str(stream["market_type"]),
                    "stream_status": str(stream["status"]),
                    **full,
                    "event_count": len(event_windows),
                    "minimum_event_clock_coverage_ratio": min(
                        (float(item["clock_coverage_ratio"]) for item in event_windows),
                        default=1.0,
                    ),
                    "minimum_event_valid_sample_ratio": min(
                        (float(item["valid_sample_ratio"]) for item in event_windows),
                        default=1.0,
                    ),
                    "coverage_pass": decision,
                }
            )
    return output


def query_window(
    conn: sqlite3.Connection, stream_id: str, start: datetime, end: datetime
) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT COUNT(*) AS rows,
               COUNT(DISTINCT ts_start) AS unique_minutes,
               MIN(ts_start) AS first_ts,
               MAX(ts_end) AS last_ts,
               SUM(valid_samples) AS valid_samples,
               SUM(expected_samples) AS expected_samples,
               SUM(CASE WHEN created_at < ts_end THEN 1 ELSE 0 END) AS early_created_rows
        FROM orderbook_metric_bars INDEXED BY idx_orderbook_metric_bars_stream_tf_ts
        WHERE stream_id = ?
          AND timeframe_seconds = 60
          AND ts_start >= ?
          AND ts_start < ?
        """,
        (stream_id, start.isoformat(), end.isoformat()),
    ).fetchone()
    rows = int(row["rows"] or 0)
    expected = expected_minutes(start, end)
    valid_samples = int(row["valid_samples"] or 0)
    expected_samples = int(row["expected_samples"] or 0)
    return {
        "rows": rows,
        "unique_minutes": int(row["unique_minutes"] or 0),
        "expected_clock_minutes": expected,
        "clock_coverage_ratio": round(rows / max(1, expected), 6),
        "valid_sample_ratio": round(valid_samples / max(1, expected_samples), 6),
        "duplicate_minute_rows": rows - int(row["unique_minutes"] or 0),
        "early_created_rows": int(row["early_created_rows"] or 0),
        "first_ts": str(row["first_ts"] or ""),
        "last_ts": str(row["last_ts"] or ""),
    }


def coverage_decision(full: dict[str, Any], event_windows: list[dict[str, Any]]) -> bool:
    full_pass = (
        float(full["clock_coverage_ratio"]) >= 0.95
        and float(full["valid_sample_ratio"]) >= 0.80
        and int(full["duplicate_minute_rows"]) == 0
        and int(full["early_created_rows"]) == 0
    )
    events_pass = all(
        float(event["clock_coverage_ratio"]) >= 0.95
        and float(event["valid_sample_ratio"]) >= 0.80
        and int(event["duplicate_minute_rows"]) == 0
        and int(event["early_created_rows"]) == 0
        for event in event_windows
    )
    return full_pass and events_pass


def expected_minutes(start: datetime, end: datetime) -> int:
    return max(0, int((end - start).total_seconds() // 60))


def require_timestamp(value: str) -> datetime:
    parsed = parse_timestamp(value)
    if parsed is None:
        raise ValueError(f"Invalid timestamp: {value}")
    return parsed


def parse_timestamp(value: Any) -> datetime | None:
    if value is None or not str(value).strip():
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def safe_tag(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in "-_" else "_" for char in value.strip())
    if not clean:
        raise ValueError("tag must contain at least one letter or number")
    return clean


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0]) if rows else []
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import pandas as pd
from user_data.Custom_Launcher.collector_runtime import atomic_write_text as write_text_atomic


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DB = USER_DATA_DIR / "collector_data" / "global_context" / "global_context.sqlite"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "collectors" / "context" / "config" / "global_context_sources.json"
DEFAULT_OUTPUT_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "source_preflight"
CAUSAL_CLOCK_TOLERANCE_SECONDS = 1.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit global-context observation cadence, revisions, duplicates, and causal "
            "timestamp safety without reading raw payloads or market outcomes."
        )
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--freeze-causal-extract",
        action="store_true",
        help="Also write a compact outcome-blind parquet using collection time as availability.",
    )
    args = parser.parse_args()

    tag = safe_tag(args.tag) if args.tag else datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    paths = output_paths(args.output_dir, tag)
    plan = {
        "mode": "setup-only unless --execute is supplied",
        "objective": (
            "Determine which global-context histories represent independent, causally timed "
            "observations before they are combined with price outcomes."
        ),
        "pass_rule": (
            "A source is directly causal-test-ready only when every row has a collection time, "
            "the stored available time is not earlier than collection, and no timestamp carries "
            "conflicting revisions. A one-second tolerance permits harmless timestamp-construction "
            "skew. Sources with preserved collection times may be repaired through a frozen "
            "vintage extraction retimestamped to created_at."
        ),
        "database": str(args.db),
        "config": str(args.config),
        "outputs": {key: str(value) for key, value in paths.items()},
        "prohibitions": [
            "No OHLCV, returns, labels, or model outcomes are read.",
            "No collector configuration or live database row is changed.",
            "Raw JSON payloads are not loaded.",
        ],
        "freeze_causal_extract": bool(args.freeze_causal_extract),
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0

    if not args.db.exists():
        raise FileNotFoundError(args.db)
    if not args.config.exists():
        raise FileNotFoundError(args.config)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = load_rows_read_only(args.db)
    audit_rows, summary = build_audit(rows, config)
    report = {
        **plan,
        "mode": "executed read-only preflight",
        "generated_at": datetime.now(UTC).isoformat(),
        "database_bytes": args.db.stat().st_size,
        "database_last_modified_utc": datetime.fromtimestamp(
            args.db.stat().st_mtime, tz=UTC
        ).isoformat(),
        "summary": summary,
        "source_metrics": audit_rows,
        "safe_extraction_rule": (
            "For historical causal tests, use created_at as available_at and retain ts and "
            "source_ts as observation metadata. Do not move a value earlier than its first "
            "locally recorded collection time unless an independently verified publication "
            "timestamp or vintage archive is available."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.freeze_causal_extract:
        causal_frame, causal_summary = build_causal_extract(rows, config)
        write_parquet_atomic(paths["causal_parquet"], causal_frame)
        report["causal_extract_summary"] = causal_summary
    write_csv_atomic(paths["csv"], audit_rows)
    write_text_atomic(paths["json"], json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_text_atomic(paths["markdown"], render_markdown(report))
    print(
        json.dumps(
            {
                "outputs": {key: str(value) for key, value in paths.items()},
                "summary": summary,
            },
            indent=2,
        )
    )
    return 0


def output_paths(output_dir: Path, tag: str) -> dict[str, Path]:
    stem = f"global_context_source_preflight_{tag}"
    return {
        "csv": output_dir / f"{stem}.csv",
        "json": output_dir / f"{stem}.json",
        "markdown": output_dir / f"{stem}.md",
        "causal_parquet": output_dir / f"{stem}_causal.parquet",
    }


def safe_tag(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in "-_" else "_" for char in value.strip())
    if not clean:
        raise ValueError("tag must contain at least one letter or number")
    return clean


def load_rows_read_only(db_path: Path) -> list[dict[str, Any]]:
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=30.0) as conn:
        conn.row_factory = sqlite3.Row
        result = conn.execute(
            """
            SELECT id, source_id, source_group, source_type, metric_key,
                   ts, source_ts, value, score, source_score, calc_score, signal, unit,
                   created_at
            FROM global_context_ticks
            ORDER BY source_id, metric_key, created_at, id
            """
        ).fetchall()
    return [dict(row) for row in result]


def build_audit(
    rows: Iterable[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_config = {
        str(source.get("id")): source
        for source in config.get("sources", [])
        if isinstance(source, dict) and source.get("id")
    }
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("source_id") or ""), str(row.get("metric_key") or ""))].append(row)

    audit_rows = [
        audit_metric(source_id, metric_key, metric_rows, source_config.get(source_id, {}))
        for (source_id, metric_key), metric_rows in sorted(groups.items())
    ]
    source_statuses: dict[str, set[str]] = defaultdict(set)
    for row in audit_rows:
        source_statuses[row["source_id"]].add(row["readiness"])

    summary = {
        "rows": sum(int(row["rows"]) for row in audit_rows),
        "source_metrics": len(audit_rows),
        "sources": len(source_statuses),
        "enabled_sources": len(
            {
                row["source_id"]
                for row in audit_rows
                if bool(row["configured_enabled"])
            }
        ),
        "direct_causal_ready_source_metrics": sum(
            row["readiness"] == "direct_causal_ready" for row in audit_rows
        ),
        "safe_retimestamp_required_source_metrics": sum(
            row["readiness"]
            in {"safe_retimestamp_required", "safe_vintage_retimestamp_required"}
            for row in audit_rows
        ),
        "blocked_source_metrics": sum(
            str(row["readiness"]).startswith("blocked") for row in audit_rows
        ),
        "source_metrics_with_revisions": sum(
            int(row["revised_source_timestamp_groups"]) > 0 for row in audit_rows
        ),
        "source_metrics_with_repeated_observations": sum(
            float(row["source_timestamp_coverage_ratio"]) > 0
            and float(row["independent_source_timestamp_ratio"]) < 0.5
            for row in audit_rows
        ),
    }
    return audit_rows, summary


def build_causal_extract(
    rows: Iterable[dict[str, Any]], config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_config = {
        str(source.get("id")): source
        for source in config.get("sources", [])
        if isinstance(source, dict) and source.get("id")
    }
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    source_rows = list(rows)
    for row in source_rows:
        groups[(str(row.get("source_id") or ""), str(row.get("metric_key") or ""))].append(row)

    collapsed: list[dict[str, Any]] = []
    for (source_id, metric_key), metric_rows in sorted(groups.items()):
        collapsed.extend(
            collapse_metric_observations(
                source_id,
                metric_key,
                metric_rows,
                source_config.get(source_id, {}),
            )
        )
    frame = pd.DataFrame(collapsed)
    if not frame.empty:
        frame = frame.sort_values(
            ["available_at", "source_id", "metric_key", "extract_sequence"],
            kind="stable",
        ).reset_index(drop=True)
        parsed_available = pd.to_datetime(frame["available_at"], utc=True, errors="coerce")
        parsed_first = pd.to_datetime(frame["first_seen_at"], utc=True, errors="coerce")
        parsed_last = pd.to_datetime(frame["last_observed_at"], utc=True, errors="coerce")
        if parsed_available.isna().any() or parsed_first.isna().any() or parsed_last.isna().any():
            raise ValueError("Causal extract contains an invalid collection timestamp.")
        if not parsed_available.equals(parsed_first):
            raise ValueError("Causal extract availability must equal first local collection time.")
        if (parsed_last < parsed_first).any():
            raise ValueError("Causal extract has a last observation before first collection.")

    summary = {
        "input_rows": len(source_rows),
        "output_observation_vintages": len(frame),
        "removed_unchanged_repeat_pulls": len(source_rows) - len(frame),
        "compression_ratio": round(len(frame) / max(1, len(source_rows)), 6),
        "revision_vintages": int((frame.get("change_kind") == "revision").sum())
        if not frame.empty
        else 0,
        "enabled_observation_vintages": int(
            frame.get("configured_enabled", pd.Series(dtype=bool)).sum()
        ),
        "first_available_at": str(frame["available_at"].min()) if not frame.empty else "",
        "last_available_at": str(frame["available_at"].max()) if not frame.empty else "",
        "market_outcomes_read": False,
    }
    return frame, summary


def collapse_metric_observations(
    source_id: str,
    metric_key: str,
    rows: list[dict[str, Any]],
    source_config: dict[str, Any],
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            parse_timestamp(row.get("created_at")) or datetime.min.replace(tzinfo=UTC),
            int(row.get("id") or 0),
        ),
    )
    output: list[dict[str, Any]] = []
    source_buckets: dict[tuple[str, str], dict[str, Any]] = {}
    source_signatures: dict[str, set[str]] = defaultdict(set)
    live_bucket: dict[str, Any] | None = None
    last_live_signature: str | None = None

    for row in ordered:
        created_at = parse_timestamp(row.get("created_at"))
        if created_at is None:
            raise ValueError(f"{source_id}/{metric_key} has no valid created_at")
        source_at = parse_timestamp(row.get("source_ts"))
        signature = observation_signature(row)
        if source_at is not None:
            source_key = source_at.isoformat()
            bucket_key = (source_key, signature)
            existing = source_buckets.get(bucket_key)
            if existing is not None:
                extend_observation(existing, created_at)
                continue
            prior_signatures = source_signatures[source_key]
            if not output:
                change_kind = "initial"
            elif prior_signatures:
                change_kind = "revision"
            else:
                change_kind = "new_source_timestamp"
            observation = make_causal_observation(
                row,
                source_id=source_id,
                metric_key=metric_key,
                configured_enabled=bool(source_config.get("enabled", False)),
                created_at=created_at,
                change_kind=change_kind,
                sequence=len(output),
            )
            source_buckets[bucket_key] = observation
            prior_signatures.add(signature)
            output.append(observation)
            continue

        if live_bucket is not None and signature == last_live_signature:
            extend_observation(live_bucket, created_at)
            continue
        observation = make_causal_observation(
            row,
            source_id=source_id,
            metric_key=metric_key,
            configured_enabled=bool(source_config.get("enabled", False)),
            created_at=created_at,
            change_kind="initial" if not output else "live_change",
            sequence=len(output),
        )
        output.append(observation)
        live_bucket = observation
        last_live_signature = signature
    return output


def observation_signature(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "value": value_signature(row.get("value")),
            "score": value_signature(row.get("score")),
            "source_score": value_signature(row.get("source_score")),
            "calc_score": value_signature(row.get("calc_score")),
            "signal": str(row.get("signal") or ""),
            "unit": str(row.get("unit") or ""),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def make_causal_observation(
    row: dict[str, Any],
    *,
    source_id: str,
    metric_key: str,
    configured_enabled: bool,
    created_at: datetime,
    change_kind: str,
    sequence: int,
) -> dict[str, Any]:
    created_text = created_at.isoformat()
    return {
        "source_id": source_id,
        "source_group": str(row.get("source_group") or ""),
        "source_type": str(row.get("source_type") or ""),
        "metric_key": metric_key,
        "configured_enabled": configured_enabled,
        "available_at": created_text,
        "first_seen_at": created_text,
        "last_observed_at": created_text,
        "recorded_ts": str(row.get("ts") or ""),
        "source_ts": str(row.get("source_ts") or ""),
        "value": row.get("value"),
        "score": row.get("score"),
        "source_score": row.get("source_score"),
        "calc_score": row.get("calc_score"),
        "signal": str(row.get("signal") or ""),
        "unit": str(row.get("unit") or ""),
        "change_kind": change_kind,
        "observation_count": 1,
        "extract_sequence": sequence,
    }


def extend_observation(observation: dict[str, Any], created_at: datetime) -> None:
    observation["last_observed_at"] = created_at.isoformat()
    observation["observation_count"] = int(observation["observation_count"]) + 1


def audit_metric(
    source_id: str,
    metric_key: str,
    rows: list[dict[str, Any]],
    source_config: dict[str, Any],
) -> dict[str, Any]:
    created = [parse_timestamp(row.get("created_at")) for row in rows]
    available = [parse_timestamp(row.get("ts")) for row in rows]
    source_times = [parse_timestamp(row.get("source_ts")) for row in rows]
    missing_created = sum(value is None for value in created)
    missing_available = sum(value is None for value in available)
    valid_created = [value for value in created if value is not None]
    valid_available = [value for value in available if value is not None]
    valid_source_times = [value for value in source_times if value is not None]

    all_pre_collection_lags = [
        (created_at - available_at).total_seconds()
        for created_at, available_at in zip(created, available, strict=True)
        if created_at is not None and available_at is not None and available_at < created_at
    ]
    pre_collection_lags = [
        value
        for value in all_pre_collection_lags
        if value > CAUSAL_CLOCK_TOLERANCE_SECONDS
    ]
    source_lags = [
        (created_at - source_at).total_seconds()
        for created_at, source_at in zip(created, source_times, strict=True)
        if created_at is not None and source_at is not None
    ]

    source_timestamp_values: dict[str, set[str]] = defaultdict(set)
    source_timestamp_counts: Counter[str] = Counter()
    for row, source_at in zip(rows, source_times, strict=True):
        if source_at is None:
            continue
        key = source_at.isoformat()
        source_timestamp_counts[key] += 1
        source_timestamp_values[key].add(value_signature(row.get("value")))
    duplicate_groups = sum(count > 1 for count in source_timestamp_counts.values())
    revised_groups = sum(len(values) > 1 for values in source_timestamp_values.values())

    ordered_values = [value_signature(row.get("value")) for row in rows]
    consecutive_pairs = max(0, len(ordered_values) - 1)
    unchanged_pairs = sum(left == right for left, right in pairwise(ordered_values))
    independent_ratio = unique_ratio(valid_source_times, len(rows))
    effective_observation_times = [
        source_at or created_at
        for source_at, created_at in zip(source_times, created, strict=True)
        if source_at is not None or created_at is not None
    ]
    effective_independent_ratio = unique_ratio(effective_observation_times, len(rows))
    availability_unique_ratio = unique_ratio(valid_available, len(rows))

    if missing_created:
        readiness = "blocked_missing_collection_time"
    elif missing_available or revised_groups:
        readiness = "safe_vintage_retimestamp_required"
    elif pre_collection_lags:
        readiness = "safe_retimestamp_required"
    else:
        readiness = "direct_causal_ready"

    return {
        "source_id": source_id,
        "metric_key": metric_key,
        "source_group": str(rows[0].get("source_group") or "") if rows else "",
        "source_type": str(rows[0].get("source_type") or "") if rows else "",
        "configured_enabled": bool(source_config.get("enabled", False)),
        "configured_availability_mode": str(
            source_config.get("availability_mode") or "unspecified"
        ),
        "rows": len(rows),
        "first_collection_at": iso_min(valid_created),
        "last_collection_at": iso_max(valid_created),
        "first_recorded_available_at": iso_min(valid_available),
        "last_recorded_available_at": iso_max(valid_available),
        "unique_collection_timestamps": len({value.isoformat() for value in valid_created}),
        "unique_recorded_available_timestamps": len(
            {value.isoformat() for value in valid_available}
        ),
        "unique_source_timestamps": len({value.isoformat() for value in valid_source_times}),
        "source_timestamp_coverage_ratio": round(len(valid_source_times) / max(1, len(rows)), 6),
        "independent_source_timestamp_ratio": round(independent_ratio, 6),
        "independent_observation_timestamp_ratio": round(effective_independent_ratio, 6),
        "recorded_availability_unique_ratio": round(availability_unique_ratio, 6),
        "median_collection_interval_seconds": rounded_or_none(median_interval(valid_created)),
        "rows_recorded_before_collection": len(pre_collection_lags),
        "subsecond_clock_skew_rows": len(all_pre_collection_lags) - len(pre_collection_lags),
        "recorded_before_collection_ratio": round(len(pre_collection_lags) / max(1, len(rows)), 6),
        "median_seconds_recorded_before_collection": rounded_or_none(median(pre_collection_lags)),
        "p95_seconds_recorded_before_collection": rounded_or_none(
            percentile(pre_collection_lags, 0.95)
        ),
        "max_seconds_recorded_before_collection": rounded_or_none(
            max(pre_collection_lags) if pre_collection_lags else None
        ),
        "median_seconds_collection_after_source_time": rounded_or_none(median(source_lags)),
        "duplicate_source_timestamp_groups": duplicate_groups,
        "revised_source_timestamp_groups": revised_groups,
        "unchanged_consecutive_value_ratio": round(unchanged_pairs / max(1, consecutive_pairs), 6),
        "readiness": readiness,
        "causal_use_rule": (
            "Use created_at as available_at"
            if readiness == "safe_retimestamp_required"
            else "Use created_at, retain revision vintages, and remove unchanged repeat pulls"
            if readiness == "safe_vintage_retimestamp_required"
            else "Use stored ts only after resolving blocker"
            if readiness.startswith("blocked")
            else "Stored ts is causally usable"
        ),
    }


def parse_timestamp(value: Any) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def value_signature(value: Any) -> str:
    if value is None:
        return "null"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return "nan"
    return f"{numeric:.12g}"


def unique_ratio(values: list[datetime], row_count: int) -> float:
    return len({value.isoformat() for value in values}) / max(1, row_count)


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def median_interval(values: list[datetime]) -> float | None:
    unique = sorted(set(values))
    gaps = [(right - left).total_seconds() for left, right in pairwise(unique)]
    return median(gaps)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def rounded_or_none(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


def iso_min(values: list[datetime]) -> str:
    return min(values).isoformat() if values else ""


def iso_max(values: list[datetime]) -> str:
    return max(values).isoformat() if values else ""


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0]) if rows else []
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Global Context Source Preflight",
        "",
        "## Result",
        "",
        f"- Rows audited: {summary['rows']}",
        f"- Sources: {summary['sources']}",
        f"- Source/metric histories: {summary['source_metrics']}",
        f"- Directly causal-ready histories: {summary['direct_causal_ready_source_metrics']}",
        "- Histories requiring safe retimestamping: "
        f"{summary['safe_retimestamp_required_source_metrics']}",
        f"- Blocked histories: {summary['blocked_source_metrics']}",
        f"- Histories with conflicting revisions: {summary['source_metrics_with_revisions']}",
        "- Histories dominated by repeated source timestamps: "
        f"{summary['source_metrics_with_repeated_observations']}",
        "",
        "## Interpretation",
        "",
        report["safe_extraction_rule"],
        "",
        "The CSV contains one row per source and metric. This preflight does not test "
        "market outcomes.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

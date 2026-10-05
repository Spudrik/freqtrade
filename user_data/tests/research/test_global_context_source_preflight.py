from __future__ import annotations

# ruff: noqa: S101
from user_data.Custom_Launcher.research.context_features.global_context_source_preflight import (
    audit_metric,
    build_audit,
    build_causal_extract,
    percentile,
)


def row(*, ts: str, source_ts: str, created_at: str, value: float) -> dict[str, object]:
    return {
        "source_id": "sample",
        "source_group": "attention",
        "source_type": "sample_type",
        "metric_key": "sample_metric",
        "ts": ts,
        "source_ts": source_ts,
        "created_at": created_at,
        "value": value,
        "score": value,
    }


def test_metric_is_direct_ready_when_availability_is_not_early() -> None:
    rows = [
        row(
            ts="2026-09-01T10:00:00+00:00",
            source_ts="2026-09-01T10:00:00+00:00",
            created_at="2026-09-01T10:00:00+00:00",
            value=10,
        ),
        row(
            ts="2026-09-01T11:00:00+00:00",
            source_ts="2026-09-01T11:00:00+00:00",
            created_at="2026-09-01T11:00:00+00:00",
            value=11,
        ),
    ]
    audit = audit_metric("sample", "sample_metric", rows, {"enabled": True})
    assert audit["readiness"] == "direct_causal_ready"
    assert audit["independent_source_timestamp_ratio"] == 1.0
    assert audit["median_collection_interval_seconds"] == 3600.0


def test_metric_requires_retimestamp_when_recorded_before_collection() -> None:
    rows = [
        row(
            ts="2026-09-01T10:00:00+00:00",
            source_ts="2026-09-01T10:00:00+00:00",
            created_at="2026-09-01T10:12:00+00:00",
            value=10,
        )
    ]
    audit = audit_metric("sample", "sample_metric", rows, {"enabled": True})
    assert audit["readiness"] == "safe_retimestamp_required"
    assert audit["rows_recorded_before_collection"] == 1
    assert audit["max_seconds_recorded_before_collection"] == 720.0
    assert audit["causal_use_rule"] == "Use created_at as available_at"


def test_subsecond_timestamp_construction_skew_is_tolerated() -> None:
    rows = [
        row(
            ts="2026-09-01T10:00:00+00:00",
            source_ts="2026-09-01T10:00:00+00:00",
            created_at="2026-09-01T10:00:00.100000+00:00",
            value=10,
        )
    ]
    audit = audit_metric("sample", "sample_metric", rows, {"enabled": True})
    assert audit["readiness"] == "direct_causal_ready"
    assert audit["rows_recorded_before_collection"] == 0
    assert audit["subsecond_clock_skew_rows"] == 1


def test_conflicting_value_for_same_source_time_is_blocked() -> None:
    rows = [
        row(
            ts="2026-09-01T10:00:00+00:00",
            source_ts="2026-09-01T09:00:00+00:00",
            created_at="2026-09-01T10:00:00+00:00",
            value=10,
        ),
        row(
            ts="2026-09-01T11:00:00+00:00",
            source_ts="2026-09-01T09:00:00+00:00",
            created_at="2026-09-01T11:00:00+00:00",
            value=12,
        ),
    ]
    audit = audit_metric("sample", "sample_metric", rows, {"enabled": True})
    assert audit["readiness"] == "safe_vintage_retimestamp_required"
    assert audit["duplicate_source_timestamp_groups"] == 1
    assert audit["revised_source_timestamp_groups"] == 1
    assert audit["independent_source_timestamp_ratio"] == 0.5
    assert audit["causal_use_rule"].startswith("Use created_at")


def test_build_audit_summarises_source_metrics() -> None:
    rows = [
        row(
            ts="2026-09-01T10:00:00+00:00",
            source_ts="2026-09-01T10:00:00+00:00",
            created_at="2026-09-01T10:00:00+00:00",
            value=10,
        )
    ]
    audit, summary = build_audit(rows, {"sources": [{"id": "sample", "enabled": True}]})
    assert len(audit) == 1
    assert summary["rows"] == 1
    assert summary["enabled_sources"] == 1
    assert summary["direct_causal_ready_source_metrics"] == 1


def test_causal_extract_retains_revision_vintages_and_collapses_repeat_pulls() -> None:
    rows = [
        row(
            ts="2026-09-01T09:00:00+00:00",
            source_ts="2026-09-01T09:00:00+00:00",
            created_at="2026-09-01T10:00:00+00:00",
            value=10,
        ),
        row(
            ts="2026-09-01T09:00:00+00:00",
            source_ts="2026-09-01T09:00:00+00:00",
            created_at="2026-09-01T10:30:00+00:00",
            value=10,
        ),
        row(
            ts="2026-09-01T09:00:00+00:00",
            source_ts="2026-09-01T09:00:00+00:00",
            created_at="2026-09-01T11:00:00+00:00",
            value=12,
        ),
    ]
    frame, summary = build_causal_extract(
        rows, {"sources": [{"id": "sample", "enabled": True}]}
    )
    assert len(frame) == 2
    assert frame["change_kind"].tolist() == ["initial", "revision"]
    assert frame["observation_count"].tolist() == [2, 1]
    assert frame["available_at"].tolist()[0] == "2026-09-01T10:00:00+00:00"
    assert summary["removed_unchanged_repeat_pulls"] == 1


def test_causal_extract_collapses_only_consecutive_equal_live_snapshots() -> None:
    values = [10, 10, 12, 10]
    rows = [
        row(
            ts=f"2026-09-01T{hour:02d}:00:00+00:00",
            source_ts="",
            created_at=f"2026-09-01T{hour:02d}:00:00+00:00",
            value=value,
        )
        for hour, value in zip(range(10, 14), values, strict=True)
    ]
    frame, _ = build_causal_extract(
        rows, {"sources": [{"id": "sample", "enabled": True}]}
    )
    assert frame["value"].tolist() == [10, 12, 10]
    assert frame["observation_count"].tolist() == [2, 1, 1]
    assert frame["change_kind"].tolist() == ["initial", "live_change", "live_change"]


def test_percentile_interpolates() -> None:
    assert percentile([0.0, 10.0], 0.95) == 9.5

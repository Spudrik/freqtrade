from __future__ import annotations

# ruff: noqa: S101
from datetime import UTC, datetime

from user_data.Custom_Launcher.research.context_features.orderbook_event_coverage_preflight import (
    coverage_decision,
    expected_minutes,
    parse_timestamp,
)


def test_expected_minutes_uses_half_open_window() -> None:
    start = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
    end = datetime(2026, 9, 1, 18, 0, tzinfo=UTC)
    assert expected_minutes(start, end) == 480


def test_coverage_decision_requires_full_and_event_quality() -> None:
    good = {
        "clock_coverage_ratio": 0.97,
        "valid_sample_ratio": 0.85,
        "duplicate_minute_rows": 0,
        "early_created_rows": 0,
    }
    assert coverage_decision(good, [good])
    weak_event = {**good, "valid_sample_ratio": 0.79}
    assert not coverage_decision(good, [weak_event])


def test_timestamp_parser_normalises_utc() -> None:
    parsed = parse_timestamp("2026-09-01T11:00:00+01:00")
    assert parsed == datetime(2026, 9, 1, 10, 0, tzinfo=UTC)

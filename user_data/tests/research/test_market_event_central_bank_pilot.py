# ruff: noqa: S101

from __future__ import annotations

import json

import pandas as pd
import pytest

from user_data.Custom_Launcher.research.context_features import (
    market_event_central_bank_pilot as pilot,
)


def test_pilot_has_three_rows_per_bank_and_required_fields() -> None:
    catalogue = pilot.build_catalogue(
        retrieved_at_utc="2026-09-04T12:00:00Z"
    )
    assert list(catalogue.columns) == list(pilot.OUTPUT_COLUMNS)
    assert catalogue["event_id"].is_unique
    assert catalogue.groupby("bank").size().to_dict() == {
        "ECB": 3,
        "BoE": 3,
        "BoJ": 3,
    }
    assert catalogue["expected_consensus"].isna().all()
    assert catalogue["source_retrieval_timestamp"].eq(
        "2026-09-04T12:00:00Z"
    ).all()


def test_local_release_conversion_handles_japan_standard_time() -> None:
    assert pilot.local_release_to_utc(
        "2020-03-16", "14:06", "Asia/Tokyo"
    ) == "2020-03-16T05:06:00Z"
    assert pilot.local_release_to_utc(
        "2024-01-23", "12:09", "Asia/Tokyo"
    ) == "2024-01-23T03:09:00Z"


def test_timing_quality_distinguishes_exact_and_date_only_records() -> None:
    catalogue = pilot.build_catalogue(retrieved_at_utc="2026-09-04T12:00:00Z")
    boj = catalogue.loc[catalogue["bank"].eq("BoJ")]
    date_only = catalogue.loc[catalogue["timing_quality"].eq("date_only")]
    assert set(boj["timing_quality"]) == {"actual_document_time"}
    assert (pd.to_datetime(date_only["available_at_utc"]).dt.hour == 0).all()
    assert len(date_only) == 6


def test_roles_cover_unchanged_changed_and_emergency_examples() -> None:
    catalogue = pilot.build_catalogue(retrieved_at_utc="2026-09-04T12:00:00Z")
    roles = set(catalogue["selection_role"])
    assert "unchanged_decision" in roles
    assert "changed_policy_or_instrument" in roles
    assert "timing_exception_or_emergency" in roles
    assert set(catalogue["scheduled"]) == {"scheduled", "unscheduled"}


def test_only_official_primary_domains_are_allowed() -> None:
    catalogue = pilot.build_catalogue(retrieved_at_utc="2026-09-04T12:00:00Z")
    for value in catalogue["official_source_urls"]:
        urls = json.loads(value)
        assert urls
        for url in urls:
            assert url.startswith("https://")
            assert any(domain in url for domain in pilot.OFFICIAL_DOMAINS.values())


def test_validation_rejects_duplicate_event_id() -> None:
    catalogue = pilot.build_catalogue(retrieved_at_utc="2026-09-04T12:00:00Z")
    catalogue.loc[catalogue.index[-1], "event_id"] = catalogue.loc[0, "event_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        pilot.validate_catalogue(catalogue)


def test_source_manifest_is_metadata_only_and_checks_status() -> None:
    class Response:
        status_code = 200
        content = b"official page fixture"

    calls: list[str] = []

    def requester(url: str, **_: object) -> Response:
        calls.append(url)
        return Response()

    urls = ["https://www.boj.or.jp/en/mopo/mpmdeci/state_2024/k240123a.htm"]
    manifest = pilot.fetch_source_manifest(
        urls,
        retrieved_at_utc="2026-09-04T12:00:00Z",
        requester=requester,
    )
    assert calls == urls
    assert manifest[0]["http_status"] == 200
    assert manifest[0]["retrieved_at_utc"] == "2026-09-04T12:00:00Z"
    assert "content" not in manifest[0]
    assert len(manifest[0]["content_sha256"]) == 64


def test_source_manifest_rejects_non_200() -> None:
    class Response:
        status_code = 503
        content = b"error"

    with pytest.raises(RuntimeError, match="503"):
        pilot.fetch_source_manifest(
            ["https://www.boj.or.jp/en/mopo/mpmdeci/state_2024/k240123a.htm"],
            retrieved_at_utc="2026-09-04T12:00:00Z",
            requester=lambda *_args, **_kwargs: Response(),
        )

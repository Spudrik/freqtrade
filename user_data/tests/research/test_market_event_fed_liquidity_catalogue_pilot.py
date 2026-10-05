# ruff: noqa: S101

from __future__ import annotations

import json

import pandas as pd
import pytest

from user_data.Custom_Launcher.research.context_features import (
    market_event_fed_liquidity_catalogue_pilot as pilot,
)


def _page(date_text: str, release_text: str, title: str) -> str:
    return (
        '<html><body>'
        f'<p class="article__time">{date_text}</p>'
        f'<h3 class="title">{title}</h3>'
        f'<p class="releaseTime">For release at {release_text}</p>'
        '</body></html>'
    )


def _pilot_pages() -> dict[str, str]:
    return {
        pilot.URL_2020_MEASURES: _page(
            "March 23, 2020",
            "8:00 a.m. EDT",
            "Federal Reserve announces extensive new measures to support the economy",
        ),
        pilot.URL_2023_BTFP: _page(
            "March 12, 2023",
            "6:15 p.m. EDT",
            "Federal Reserve Board announces additional funding for eligible institutions",
        ),
        pilot.URL_2024_BTFP_END: _page(
            "January 24, 2024",
            "7:00 p.m. EST",
            "Federal Reserve Board announces the BTFP will cease making new loans",
        ),
    }


def test_parser_converts_exact_fed_release_times_and_dst() -> None:
    parsed_2020 = pilot.FederalReservePageParser.parse(
        _pilot_pages()[pilot.URL_2020_MEASURES]
    )
    parsed_2023 = pilot.FederalReservePageParser.parse(
        _pilot_pages()[pilot.URL_2023_BTFP]
    )
    parsed_2024 = pilot.FederalReservePageParser.parse(
        _pilot_pages()[pilot.URL_2024_BTFP_END]
    )
    assert parsed_2020["available_at_utc"] == "2020-03-23T12:00:00Z"
    assert parsed_2023["available_at_utc"] == "2023-03-12T22:15:00Z"
    assert parsed_2024["available_at_utc"] == "2024-01-25T00:00:00Z"
    assert parsed_2020["official_release_local_text"].endswith("8:00 a.m. EDT")
    assert parsed_2024["official_release_local_text"].endswith("7:00 p.m. EST")


def test_parser_fails_clearly_when_exact_time_is_missing() -> None:
    page = (
        '<p class="article__time">March 23, 2020</p>'
        '<h3 class="title">Federal Reserve announcement</h3>'
    )
    with pytest.raises(
        pilot.OfficialReleaseTimeMissing,
        match=r"official release time absent.*cannot derive available_at UTC",
    ):
        pilot.FederalReservePageParser.parse(page)


def test_parser_rejects_wrong_seasonal_timezone_instead_of_guessing() -> None:
    page = _page(
        "March 23, 2020",
        "8:00 a.m. EST",
        "Federal Reserve announces extensive new measures",
    )
    with pytest.raises(ValueError, match="timezone abbreviation does not match"):
        pilot.FederalReservePageParser.parse(page)


def test_build_catalogue_has_exact_scope_and_null_expectations() -> None:
    catalogue = pilot.build_catalogue_from_pages(
        page_html=_pilot_pages(),
        retrieved_at_utc="2026-09-05T09:00:00Z",
    )
    assert list(catalogue.columns) == list(pilot.OUTPUT_COLUMNS)
    assert len(catalogue) == 3
    assert catalogue["event_id"].is_unique
    assert catalogue["expectation_consensus"].isna().all()
    assert catalogue["source_retrieval_timestamp"].eq(
        "2026-09-05T09:00:00Z"
    ).all()
    assert set(catalogue["timing_quality"]) == {"actual_document_time"}
    assert set(catalogue["scheduled_or_unscheduled"]) == {
        "scheduled",
        "unscheduled",
    }


def test_validation_rejects_duplicate_event_id() -> None:
    catalogue = pilot.build_catalogue_from_pages(
        page_html=_pilot_pages(),
        retrieved_at_utc="2026-09-05T09:00:00Z",
    )
    catalogue.loc[catalogue.index[-1], "event_id"] = catalogue.loc[0, "event_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        pilot.validate_catalogue(catalogue)


def test_validation_rejects_non_null_expectation_consensus() -> None:
    catalogue = pilot.build_catalogue_from_pages(
        page_html=_pilot_pages(),
        retrieved_at_utc="2026-09-05T09:00:00Z",
    )
    catalogue.loc[0, "expectation_consensus"] = "unknown"
    with pytest.raises(ValueError, match="Expectation/consensus"):
        pilot.validate_catalogue(catalogue)


def test_fetch_records_official_hash_without_storing_page_body() -> None:
    class Response:
        status_code = 200
        content = _pilot_pages()[pilot.URL_2020_MEASURES].encode("utf-8")

    calls: list[str] = []

    def requester(url: str, **_: object) -> Response:
        calls.append(url)
        return Response()

    record, manifest = pilot.fetch_and_parse_spec(
        pilot.PILOT_SPECS[0],
        retrieved_at_utc="2026-09-05T09:00:00Z",
        requester=requester,
    )
    assert calls == [pilot.URL_2020_MEASURES]
    assert record["expectation_consensus"] is None
    assert record["source_retrieval_timestamp"] == "2026-09-05T09:00:00Z"
    assert len(record["source_content_sha256"]) == 64
    assert manifest["http_status"] == 200
    assert manifest["content_sha256"] == record["source_content_sha256"]
    assert "content" not in manifest


def test_fetch_rejects_non_200_before_parsing() -> None:
    class Response:
        status_code = 503
        content = b"unavailable"

    with pytest.raises(RuntimeError, match="503"):
        pilot.fetch_and_parse_spec(
            pilot.PILOT_SPECS[0],
            retrieved_at_utc="2026-09-05T09:00:00Z",
            requester=lambda *_args, **_kwargs: Response(),
        )


def test_only_the_three_named_federal_reserve_urls_are_frozen() -> None:
    urls = {str(spec["source_url"]) for spec in pilot.PILOT_SPECS}
    assert urls == {
        pilot.URL_2020_MEASURES,
        pilot.URL_2023_BTFP,
        pilot.URL_2024_BTFP_END,
    }
    for url in urls:
        assert url.startswith("https://www.federalreserve.gov/")


def test_json_records_turn_nan_expectation_into_json_null() -> None:
    catalogue = pilot.build_catalogue_from_pages(
        page_html=_pilot_pages(),
        retrieved_at_utc="2026-09-05T09:00:00Z",
    )
    encoded = json.dumps(pilot._json_records(catalogue))
    assert '"expectation_consensus": null' in encoded
    assert pd.isna(catalogue.loc[0, "expectation_consensus"])

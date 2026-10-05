# ruff: noqa: S101

"""Outcome-blind tests for the official UK ONS CPI release catalogue."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import pandas as pd
import pytest

from user_data.Custom_Launcher.research.context_features import (
    market_event_uk_ons_cpi_catalogue as ons,
)


def html_for(spec: dict[str, str], timestamp: str, *, label: str = "Released") -> str:
    return (
        "<html><body>"
        f"<h1>{spec['expected_title']}</h1>"
        f"<p>{label}: {timestamp}</p>"
        "</body></html>"
    )


def complete_documents() -> dict[str, str]:
    documents: dict[str, str] = {}
    for index, spec in enumerate(ons.expected_release_specs()):
        reference_end = pd.Period(spec["reference_month"], freq="M").end_time
        release = reference_end + timedelta(days=15)
        clock = "9:30am" if index < 3 else "7:00am"
        stamp = f"{release.day} {release.strftime('%B %Y')} {clock}"
        label = "Release date" if index % 2 else "Released"
        documents[spec["source_url"]] = html_for(spec, stamp, label=label)
    return documents


def test_expected_release_specs_cover_fixed_publication_window() -> None:
    specs = ons.expected_release_specs()

    assert len(specs) == 72
    assert specs[0]["reference_month"] == "2019-12"
    assert specs[0]["source_url"].endswith("/ukconsumerpriceinflationdecember2019")
    assert specs[-1]["reference_month"] == "2025-11"
    assert len({spec["source_url"] for spec in specs}) == 72


def test_complete_catalogue_parses_both_labels_and_daylight_saving() -> None:
    catalogue = ons.build_catalogue(complete_documents())

    assert len(catalogue) == 72
    assert catalogue["anchor_utc"].nunique() == 72
    assert catalogue["official_release_clock_local"].value_counts().to_dict() == {
        "07:00": 69,
        "09:30": 3,
    }
    summer = catalogue.loc[catalogue["reference_month"].eq("2024-05")].iloc[0]
    winter = catalogue.loc[catalogue["reference_month"].eq("2024-11")].iloc[0]
    assert pd.Timestamp(summer["anchor_utc"]).hour == 6
    assert pd.Timestamp(winter["anchor_utc"]).hour == 7
    assert catalogue["market_expectation_status"].eq("unavailable_not_imputed").all()


def test_release_title_and_clock_must_be_unique() -> None:
    spec = ons.expected_release_specs()[4]
    timestamp = "17 June 2020 7:00am"

    with pytest.raises(ValueError, match="title"):
        ons.parse_release_page("<h1>Wrong release</h1><p>Released: " + timestamp + "</p>", spec)
    duplicated = html_for(spec, timestamp) + f"<p>Release date: {timestamp}</p>"
    with pytest.raises(ValueError, match="clock"):
        ons.parse_release_page(duplicated, spec)


def test_fetch_rejects_redirect_or_non_official_final_url() -> None:
    url = ons.expected_release_specs()[0]["source_url"]

    class Response:
        content = b"<html></html>"
        status_code = 200
        url = "https://example.com/release"

    def requester(*args: Any, **kwargs: Any) -> Response:
        assert kwargs["allow_redirects"] is False
        return Response()

    with pytest.raises(ValueError, match="exact approved"):
        ons.fetch_page(url, requester=requester)


def test_official_url_and_no_execute_are_strict(capsys: Any) -> None:
    url = ons.expected_release_specs()[0]["source_url"]
    assert ons.official_release_url(url)
    assert not ons.official_release_url(url.replace("https://", "http://"))
    assert not ons.official_release_url(url + "?view=1")
    assert not ons.official_release_url("https://www.ons.gov.uk.evil.example/releases/test")
    assert ons.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ready_not_executed"
    assert result["outcomes_will_be_read"] is False
    assert result["expected_release_pages"] == 72

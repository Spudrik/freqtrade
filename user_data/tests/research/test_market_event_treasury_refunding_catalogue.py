# ruff: noqa: S101

"""Outcome-blind tests for the Treasury refunding source catalogue."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from user_data.Custom_Launcher.research.context_features import (
    market_event_treasury_refunding_catalogue as treasury,
)


def archive_fixture() -> str:
    rows = []
    suffixes = {1: "st", 2: "nd", 3: "rd", 4: "th"}
    for year in range(treasury.START_YEAR, treasury.END_YEAR + 1):
        links = "".join(
            f'<th headers="{year}"><a href="/news/press-releases/'
            f'aa{year % 100:02d}{quarter:02d}">'
            f"{quarter}{suffixes[quarter]} Quarter</a></th>"
            for quarter in range(1, 5)
        )
        rows.append(f'<tr><th id="{year}">{year}</th></tr><tr>{links}</tr>')
    return f"<table><thead>{''.join(rows)}</thead></table>"


def test_archive_requires_four_unique_quarters_each_year() -> None:
    rows = treasury.parse_archive(archive_fixture())
    assert len(rows) == 24
    assert {row["archive_quarter"] for row in rows} == {1, 2, 3, 4}
    with pytest.raises(ValueError, match="quarters"):
        treasury.parse_archive(archive_fixture().replace("4th Quarter", "3rd Quarter", 1))


def test_statement_requires_one_timezone_aware_official_time() -> None:
    row = {
        "archive_year": 2024,
        "archive_quarter": 1,
        "source_url": "https://home.treasury.gov/news/press-releases/test",
    }
    parsed = treasury.parse_statement(
        (
            "<h2>Quarterly Refunding Statement of an Official</h2>"
            '<time class="datetime" datetime="2024-01-31T08:30:00-05:00">'
            "January 31, 2024</time>"
        ),
        row,
    )
    assert parsed["anchor_utc"] == "2024-01-31T13:30:00+00:00"
    assert parsed["market_consensus_status"] == "unavailable_not_imputed"
    with pytest.raises(ValueError, match="timezone-naive"):
        treasury.parse_statement(
            (
                "<h2>Quarterly Refunding Statement of an Official</h2>"
                '<time class="datetime" datetime="2024-01-31T08:30:00">'
                "January 31, 2024</time>"
            ),
            row,
        )
    with pytest.raises(ValueError, match="not Eastern time"):
        treasury.parse_statement(
            (
                "<h2>Quarterly Refunding Statement of an Official</h2>"
                '<time class="datetime" datetime="2024-01-31T08:30:00+01:00">'
                "January 31, 2024</time>"
            ),
            row,
        )


def test_catalogue_validation_requires_24_official_rows() -> None:
    records = []
    release_months = {1: 2, 2: 5, 3: 8, 4: 11}
    for year in range(2020, 2026):
        for quarter in range(1, 5):
            local = pd.Timestamp(
                year=year,
                month=release_months[quarter],
                day=1,
                hour=8,
                minute=30,
                tz="America/New_York",
            )
            records.append(
                {
                    "event_id": f"event-{year}-{quarter}",
                    "event_family": "treasury_quarterly_refunding_statement",
                    "archive_year": year,
                    "archive_quarter": quarter,
                    "decision_date_local": local.date().isoformat(),
                    "official_timestamp_text": local.isoformat(),
                    "title": "Quarterly Refunding Statement of an Official",
                    "source_url": (
                        "https://home.treasury.gov/news/press-releases/"
                        f"aa{year % 100:02d}{quarter:02d}"
                    ),
                    "archive_url": treasury.ARCHIVE_URL,
                    "timestamp_quality": "official_page_timezone_aware_datetime",
                    "market_consensus_status": "unavailable_not_imputed",
                    "whole_event_partition": treasury.PARTITIONS[year],
                    "selection_method": (
                        "complete_official_archive_one_statement_per_quarter"
                    ),
                    "anchor_utc": local.tz_convert("UTC"),
                }
            )
    frame = pd.DataFrame.from_records(records)
    frame["event_id"] = frame.apply(
        lambda row: (
            f"treasury_refunding_{row['decision_date_local']}_"
            f"{row['source_url'].rsplit('/', maxsplit=1)[-1]}"
        ),
        axis=1,
    )
    allowed = set(frame["source_url"])
    treasury.validate_catalogue(frame, allowed)
    with pytest.raises(ValueError, match="expected 24"):
        treasury.validate_catalogue(frame.iloc[:-1], allowed)


def test_official_url_is_strict_and_no_execute_is_offline(capsys) -> None:  # type: ignore[no-untyped-def]
    assert treasury.official_url("https://home.treasury.gov/news/press-releases/test")
    assert treasury.official_statement_url(
        "https://home.treasury.gov/news/press-releases/aa1234"
    )
    assert not treasury.official_statement_url("https://home.treasury.gov/not-a-release")
    assert not treasury.official_statement_url(
        "https://home.treasury.gov/news/press-releases/aa1234?query=yes"
    )
    assert not treasury.official_url("http://home.treasury.gov/news/test")
    assert not treasury.official_url("https://user@home.treasury.gov/news/test")
    assert not treasury.official_url("https://home.treasury.gov:8443/news/test")
    assert treasury.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ready_not_executed"
    assert result["outcomes_will_be_read"] is False

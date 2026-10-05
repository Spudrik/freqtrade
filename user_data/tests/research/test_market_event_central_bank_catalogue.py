# ruff: noqa: S101

"""Outcome-blind tests for the complete central-bank source catalogue."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from user_data.Custom_Launcher.research.context_features import (
    market_event_central_bank_catalogue as catalogue,
)


def test_timing_rule_documents_must_contain_exact_official_evidence() -> None:
    catalogue.verify_timing_sources(
        "Starting 21 July 2022 decisions move to 14:15 instead of 13:45.",
        "Published at 12.02pm instead of the regular time of 12pm.",
    )
    with pytest.raises(ValueError, match="ECB timing-rule"):
        catalogue.verify_timing_sources(
            "Decisions have a release time.",
            "Published at 12.02pm instead of the regular time of 12pm.",
        )


def test_ecb_archive_requires_eight_decisions_and_applies_time_change() -> None:
    dates = (
        "220203",
        "220310",
        "220414",
        "220609",
        "220721",
        "220908",
        "221027",
        "221215",
    )
    document = "".join(
        f'<dl><dt isodate="20{value[:2]}-{value[2:4]}-{value[4:6]}">date</dt><dd>'
        f'<a href="/press/pr/date/2022/html/ecb.mp{value}~hash.en.html">'
        "Monetary policy decisions</a></dd></dl>"
        for value in dates
    )
    records = catalogue.parse_ecb_archive(
        document, year=2022, archive_url="https://www.ecb.europa.eu/archive"
    )
    assert len(records) == 8
    by_date = {row["decision_date"]: row for row in records}
    assert by_date["2022-02-03"]["available_at_utc"] == "2022-02-03T12:45:00+00:00"
    assert by_date["2022-07-21"]["available_at_utc"] == "2022-07-21T12:15:00+00:00"
    assert all(row["expected_consensus"] is None for row in records)

    with pytest.raises(ValueError, match="expected 8"):
        catalogue.parse_ecb_archive(
            document.replace("Monetary policy decisions", "Other", 1),
            year=2022,
            archive_url="https://www.ecb.europa.eu/archive",
        )


def test_boe_archive_selects_only_eight_regular_pages_per_year() -> None:
    links = []
    for year in range(catalogue.START_YEAR, catalogue.END_YEAR + 1):
        for month in catalogue.BOE_MONTHS[:8]:
            links.append(
                f'<a href="https://www.bankofengland.co.uk/'
                f'monetary-policy-summary-and-minutes/{year}/{month}-{year}">x</a>'
            )
    links.append(
        '<a href="https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/'
        '2020/13march-2020">special</a>'
    )
    selected = catalogue.parse_boe_archive("".join(links))
    assert len(selected) == 48
    assert not any("13march" in value for value in selected)


def test_boe_page_uses_noon_and_documented_clock_exceptions() -> None:
    normal = catalogue.parse_boe_page(
        "<title>Bank Rate increased | Bank of England</title> Published on 03 February 2022",
        source_url=(
            "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/"
            "2022/february-2022"
        ),
    )
    assert normal["available_at_utc"] == "2022-02-03T12:00:00+00:00"
    assert normal["timing_quality"] == "official_standard_time_with_known_exceptions"

    early = catalogue.parse_boe_page(
        "<title>Bank Rate held | Bank of England</title> Published on 06 August 2020",
        source_url=(
            "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/"
            "2020/august-2020"
        ),
    )
    assert early["available_at_utc"] == "2020-08-06T06:00:00+00:00"
    assert early["timing_quality"] == "official_notice_exact_time"

    exception = catalogue.parse_boe_page(
        "<title>Bank Rate reduced | Bank of England</title> Published on 08 May 2025",
        source_url=(
            "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/"
            "2025/may-2025"
        ),
    )
    assert exception["available_at_utc"] == "2025-05-08T11:02:00+00:00"
    assert exception["timing_quality"] == "official_notice_exact_time"


def test_boe_notice_archive_and_text_are_required() -> None:
    links = "".join(
        f'<a href="{notice["url"]}">notice</a>'
        for notice in catalogue.BOE_TIMING_NOTICES.values()
    )
    documents = {
        str(notice["url"]): " ".join(str(value) for value in notice["proof"])
        for notice in catalogue.BOE_TIMING_NOTICES.values()
    }
    catalogue.verify_boe_known_timing_notices(links, documents)
    with pytest.raises(ValueError, match="missing timing notices"):
        catalogue.verify_boe_known_timing_notices(
            links.split("</a>", maxsplit=1)[-1], documents
        )


def test_boj_special_source_evidence_and_official_url_rules() -> None:
    catalogue.verify_boj_special_sources(
        "May 19, 2020 Calling of an Unscheduled Monetary Policy Meeting",
        (
            "Change in the Scheduled Dates of Monetary Policy Meetings "
            "Mar. 16 (Mon.) replaces Mar. 18 (Wed.), 19 (Thurs.)"
        ),
    )
    with pytest.raises(ValueError, match="unscheduled meeting"):
        catalogue.verify_boj_special_sources(
            "May 19, 2020 ordinary meeting",
            (
                "Change in the Scheduled Dates of Monetary Policy Meetings "
                "Mar. 16 (Mon.) replaces Mar. 18 (Wed.), 19 (Thurs.)"
            ),
        )
    with pytest.raises(ValueError, match="rescheduled March meeting"):
        catalogue.verify_boj_special_sources(
            "May 19, 2020 Calling of an Unscheduled Monetary Policy Meeting",
            "Change in the Scheduled Dates of Monetary Policy Meetings",
        )
    assert catalogue._official_domain("https://www.boj.or.jp/en/source")
    assert not catalogue._official_domain("http://www.boj.or.jp/en/source")
    assert not catalogue._official_domain("https://user@www.boj.or.jp/en/source")
    assert not catalogue._official_domain("https://www.boj.or.jp:8443/en/source")


def test_boj_archive_requires_49_unique_statement_ids() -> None:
    identifiers = [
        f"{year % 100:02d}{month:02d}01a"
        for year in range(2020, 2026)
        for month in range(1, 9)
    ] + ["200901b"]
    document = "".join(
        f'<a href="/en/mopo/mpmdeci/mpr_20{value[:2]}/k{value}.pdf">x</a>'
        for value in identifiers
    )
    parsed = catalogue.parse_boj_archive(document)
    assert len(parsed) == 49
    assert parsed[0][1].endswith(f"/k{parsed[0][0]}.htm")


def test_boj_page_uses_actual_page_time_and_labels_unscheduled_meeting() -> None:
    row = catalogue.parse_boj_page(
        (
            "Release dates and times: Statement on Monetary Policy -- "
            "Friday, May 22 at 10:01 Minutes of the meeting -- later"
        ),
        identifier="200522b",
        source_url="https://www.boj.or.jp/en/mopo/mpmdeci/state_2020/k200522b.htm",
    )
    assert row["available_at_utc"] == "2020-05-22T01:01:00+00:00"
    assert row["scheduled"] == "unscheduled"
    assert row["timing_quality"] == "actual_document_time"
    rescheduled = catalogue.parse_boj_page(
        (
            "Release dates and times: Enhancement of Monetary Easing -- "
            "Monday, March 16 at 14:06"
        ),
        identifier="200316b",
        source_url="https://www.boj.or.jp/en/mopo/mpmdeci/state_2020/k200316b.htm",
    )
    assert rescheduled["scheduled"] == "rescheduled_scheduled"

    missing_space = catalogue.parse_boj_page(
        (
            "Release dates and times: Change in the Guideline -- "
            "Wednesday, July 31at 12:56 Full text -- Thursday, August 1 at 14:00"
        ),
        identifier="240731a",
        source_url="https://www.boj.or.jp/en/mopo/mpmdeci/state_2024/k240731a.htm",
    )
    assert missing_space["available_at_utc"] == "2024-07-31T03:56:00+00:00"
    with pytest.raises(ValueError, match="no exact first release time"):
        catalogue.parse_boj_page(
            "Release date only",
            identifier="200522b",
            source_url="https://www.boj.or.jp/en/mopo/mpmdeci/state_2020/k200522b.htm",
        )


def test_complete_catalogue_validation_rejects_consensus_or_wrong_counts() -> None:
    records = []
    index = 0
    for bank, count in catalogue.EXPECTED_BANK_COUNTS.items():
        for number in range(count):
            index += 1
            records.append(
                {
                    "bank": bank,
                    "event_id": f"event-{index}",
                    "decision_date": "2024-01-01",
                    "available_at_utc": pd.Timestamp("2024-01-01", tz="UTC"),
                    "local_timezone": "UTC",
                    "timing_quality": "actual_document_time",
                    "scheduled": (
                        "unscheduled"
                        if bank == "BoJ" and number == 0
                        else "rescheduled_scheduled"
                        if bank == "BoJ" and number == 1
                        else "scheduled"
                    ),
                    "source_url": {
                        "ECB": "https://www.ecb.europa.eu/source",
                        "BoE": "https://www.bankofengland.co.uk/source",
                        "BoJ": "https://www.boj.or.jp/source",
                    }[bank],
                    "expected_consensus": None,
                    "whole_event_partition": "internal_validation_2024_2025",
                    "release_cluster_id": f"cluster-{index}",
                }
            )
    frame = pd.DataFrame.from_records(records)
    catalogue.validate_catalogue(frame)
    with_consensus = frame.copy()
    with_consensus.loc[0, "expected_consensus"] = 1.0
    with pytest.raises(ValueError, match="consensus"):
        catalogue.validate_catalogue(with_consensus)
    with pytest.raises(ValueError, match="Unexpected central-bank counts"):
        catalogue.validate_catalogue(frame.iloc[:-1])


def test_no_execute_does_not_fetch_sources(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        catalogue,
        "fetch_pages",
        lambda urls: (_ for _ in ()).throw(AssertionError("network was used")),
    )
    assert catalogue.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ready_not_executed"
    assert result["outcomes_will_be_read"] is False

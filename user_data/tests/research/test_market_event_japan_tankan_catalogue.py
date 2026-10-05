# ruff: noqa: S101

"""Outcome-blind tests for the official BoJ Tankan Summary catalogue."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import pandas as pd
import pytest

from user_data.Custom_Launcher.research.context_features import (
    market_event_japan_tankan_catalogue as tankan,
)


def complete_documents() -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for year, dates in tankan.EXPECTED_RELEASES.items():
        source = tankan.ARCHIVE_URLS[0 if year == 2020 else 1]
        for date_text in dates:
            stamp = pd.Timestamp(date_text)
            reference_month = {4: 3, 7: 6, 10: 9, 12: 12}[stamp.month]
            reference_name = {
                3: "March",
                6: "June",
                9: "September",
                12: "December",
            }[reference_month]
            label = stamp.strftime("%b. %d, %Y").replace(" 0", " ")
            archive = 2016 if year == 2020 else 2021
            href = (
                f"/en/statistics/tk/gaiyo/{archive}/"
                f"tka{str(year)[2:]}{reference_month:02d}.pdf"
            )
            grouped[source].append(
                "<tr>"
                f"<td>{label}</td><td>{reference_name} {year} Survey</td>"
                f"<td><a href='{href}'>PDF</a></td>"
                "</tr>"
            )
    documents = {
        source: "<html><body>Bank of Japan<table>" + "".join(rows) + "</table></body></html>"
        for source, rows in grouped.items()
    }
    documents[tankan.FAQ_URL] = (
        "<html><body>Bank of Japan. The Bank announces the Tankan release dates and "
        "times beforehand. The release times are always at 8:50 a.m. Japan Standard "
        "Time.</body></html>"
    )
    return documents


def test_complete_catalogue_has_fixed_dates_and_utc_clock() -> None:
    catalogue = tankan.build_catalogue(complete_documents())

    assert len(catalogue) == 24
    assert catalogue["anchor_utc"].nunique() == 24
    assert catalogue["official_release_clock_local"].eq("08:50").all()
    first = catalogue.iloc[0]
    assert first["release_date_local"] == "2020-04-01"
    assert pd.Timestamp(first["anchor_utc"]) == pd.Timestamp("2020-03-31T23:50:00Z")
    assert catalogue["market_expectation_status"].eq("unavailable_not_imputed").all()
    assert catalogue["whole_event_partition"].value_counts().to_dict() == {
        "development_2021_2023": 12,
        "internal_validation_2024_2025": 8,
        "context_only_2020": 4,
    }


def test_archive_parser_rejects_changed_date_or_unapproved_pdf() -> None:
    source = tankan.ARCHIVE_URLS[0]
    bad_date = (
        "<table><tr><td>Unknown</td><td>March 2020 Survey</td>"
        "<td><a href='/en/statistics/tk/gaiyo/2016/tka2003.pdf'>PDF</a></td></tr></table>"
    )
    with pytest.raises(ValueError, match="archive date"):
        tankan.parse_archive(bad_date, source)

    bad_pdf = (
        "<table><tr><td>Apr. 1, 2020</td><td>March 2020 Survey</td>"
        "<td><a href='https://example.com/tka2003.pdf'>PDF</a></td></tr></table>"
    )
    with pytest.raises(ValueError, match="approved Summary PDF"):
        tankan.parse_archive(bad_pdf, source)


def test_faq_must_retain_always_0850_rule() -> None:
    with pytest.raises(ValueError, match="08:50"):
        tankan.validate_faq("<html>Bank of Japan release schedule</html>")


def test_fetch_rejects_redirect_or_non_official_final_url() -> None:
    url = tankan.FAQ_URL

    class Response:
        content = b"<html>Bank of Japan</html>"
        status_code = 200
        url = "https://example.com/faq"
        encoding = "utf-8"

    def requester(*args: Any, **kwargs: Any) -> Response:
        assert kwargs["allow_redirects"] is False
        return Response()

    with pytest.raises(ValueError, match="exact approved"):
        tankan.fetch_page(url, requester=requester)


def test_urls_and_no_execute_are_strict(capsys: Any) -> None:
    assert tankan.official_boj_url(tankan.FAQ_URL, pdf=False)
    assert tankan.official_boj_url(
        "https://www.boj.or.jp/en/statistics/tk/gaiyo/2021/tka2512.pdf", pdf=True
    )
    assert not tankan.official_boj_url(tankan.FAQ_URL + "?view=1", pdf=False)
    assert not tankan.official_boj_url(
        "https://www.boj.or.jp.evil.example/en/statistics/tk/gaiyo/2021/tka2512.pdf",
        pdf=True,
    )
    assert tankan.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ready_not_executed"
    assert result["outcomes_will_be_read"] is False
    assert result["expected_release_rows"] == 24

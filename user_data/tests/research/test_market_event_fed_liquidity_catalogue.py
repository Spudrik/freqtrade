# ruff: noqa: S101

"""Network-free tests for the frozen Federal Reserve liquidity catalogue."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from user_data.Custom_Launcher.research.context_features import (
    market_event_fed_liquidity_catalogue as catalogue,
)


RETRIEVED_AT = "2026-09-05T12:00:00Z"


def _event_page(spec: dict[str, str]) -> str:
    """Create an official-looking page fixture with deterministic release clocks."""
    year = int(spec["year"])
    if year == 2020:
        date_text, release_text = "March 23, 2020", "8:00 a.m. EDT"
    elif year == 2021:
        date_text, release_text = "June 2, 2021", "8:00 a.m. EDT"
    elif "20230312" in spec["source_url"]:
        date_text, release_text = "March 12, 2023", "6:15 p.m. EDT"
    elif year == 2023:
        date_text, release_text = "March 19, 2023", "6:15 p.m. EDT"
    else:
        date_text, release_text = "January 24, 2024", "7:00 p.m. EST"
    return (
        "<html><body>"
        f'<p class="article__time">{date_text}</p>'
        f'<h3 class="title">Federal Reserve synthetic liquidity announcement '
        f'{spec["event_id"]}</h3>'
        f'<p class="releaseTime">For release at {release_text}</p>'
        "</body></html>"
    )


def _archive_page(year: int, pages: list[str]) -> str:
    links = "".join(
        f'<a href="{page}">Synthetic annual archive link {page}</a>'
        for page in pages
    )
    return f"<html><body><h1>{year} press releases</h1>{links}</body></html>"


@pytest.fixture(scope="module")
def synthetic_inputs() -> tuple[dict[int, str], dict[str, str]]:
    pages: dict[str, str] = {
        spec["source_url"]: _event_page(spec) for spec in catalogue.EVENT_SPECS
    }
    archives: dict[int, str] = {}
    for year, archive_url in catalogue.ARCHIVE_URLS.items():
        del archive_url  # The URL is used by build_catalogue, not by this fixture.
        pages_for_year = [
            spec["source_url"].rsplit("/", 1)[-1]
            for spec in catalogue.EVENT_SPECS
            if int(spec["year"]) == year
        ]
        archives[year] = _archive_page(year, pages_for_year)
    return archives, pages


@pytest.fixture(scope="module")
def synthetic_catalogue(
    synthetic_inputs: tuple[dict[int, str], dict[str, str]],
) -> pd.DataFrame:
    archives, pages = synthetic_inputs
    return catalogue.build_catalogue(
        archive_html=archives,
        page_html=pages,
        retrieved_at_utc=RETRIEVED_AT,
    )


def test_frozen_scope_has_40_unique_specs_and_expected_concentration() -> None:
    assert len(catalogue.EVENT_SPECS) == 40
    assert len({spec["event_id"] for spec in catalogue.EVENT_SPECS}) == 40
    assert Counter(spec["year"] for spec in catalogue.EVENT_SPECS) == Counter(
        {"2020": 32, "2021": 4, "2023": 3, "2024": 1}
    )
    assert Counter(
        spec["programme_episode_id"] for spec in catalogue.EVENT_SPECS
    ) == Counter(
        {
            "covid_2020_support": 32,
            "covid_2021_lifecycle": 4,
            "banking_stress_2023": 3,
            "btfp_lifecycle_2024": 1,
        }
    )


def test_spec_year_is_extracted_from_page_identifier() -> None:
    spec = catalogue._spec(
        "monetary20261231a.htm", episode="future_test_episode"
    )
    assert spec["year"] == "2026"
    assert spec["event_id"] == "fed_liquidity_monetary20261231a"
    with pytest.raises(ValueError, match="has no year"):
        catalogue._spec("monetary-no-year.htm", episode="invalid")


def test_reused_pilot_parser_recovers_exact_release_clock_and_timezone() -> None:
    page_2020 = _event_page(catalogue.EVENT_SPECS[0])
    spec_2024 = next(spec for spec in catalogue.EVENT_SPECS if spec["year"] == "2024")
    page_2024 = _event_page(spec_2024)

    parsed_2020 = catalogue.pilot.FederalReservePageParser.parse(page_2020)
    parsed_2024 = catalogue.pilot.FederalReservePageParser.parse(page_2024)

    assert parsed_2020["available_at_utc"] == "2020-03-23T12:00:00Z"
    assert parsed_2024["available_at_utc"] == "2024-01-25T00:00:00Z"
    assert parsed_2020["release_time_text"] == "8:00 a.m. EDT"
    assert parsed_2024["release_time_text"] == "7:00 p.m. EST"

    date_only = (
        '<p class="article__time">March 23, 2020</p>'
        '<h3 class="title">Federal Reserve announcement</h3>'
    )
    with pytest.raises(
        catalogue.pilot.OfficialReleaseTimeMissing,
        match="cannot derive available_at UTC",
    ):
        catalogue.pilot.FederalReservePageParser.parse(date_only)


def test_build_catalogue_keeps_year_episode_counts_and_clusters_duplicate_clocks(
    synthetic_catalogue: pd.DataFrame,
) -> None:
    assert list(synthetic_catalogue.columns) == list(catalogue.OUTPUT_COLUMNS)
    assert len(synthetic_catalogue) == 40
    assert synthetic_catalogue["event_id"].is_unique
    assert synthetic_catalogue["source_url"].is_unique

    years = pd.to_datetime(synthetic_catalogue["available_at_utc"], utc=True).dt.year
    assert Counter(years.astype(str)) == Counter(
        {"2020": 32, "2021": 4, "2023": 3, "2024": 1}
    )
    assert Counter(synthetic_catalogue["programme_episode_id"]) == Counter(
        {
            "covid_2020_support": 32,
            "covid_2021_lifecycle": 4,
            "banking_stress_2023": 3,
            "btfp_lifecycle_2024": 1,
        }
    )

    timestamp_groups = synthetic_catalogue.groupby("available_at_utc")
    assert (
        timestamp_groups["announcement_cluster_id"].nunique().eq(1).all()
    )
    assert synthetic_catalogue["announcement_cluster_id"].nunique() == 5
    assert (
        synthetic_catalogue["announcement_cluster_id"].value_counts().max() == 32
    )
    assert synthetic_catalogue["announcement_cluster_id"].str.startswith(
        "fed_release_"
    ).all()


def test_catalogue_has_no_historical_expectation_or_schedule_guess(
    synthetic_catalogue: pd.DataFrame,
) -> None:
    assert synthetic_catalogue["expectation_consensus"].isna().all()
    assert not synthetic_catalogue["announcement_schedule_known"].astype(bool).any()
    assert synthetic_catalogue["release_time_text"].notna().all()
    assert synthetic_catalogue["official_release_local_text"].str.contains(
        "For release at"
    ).all()


def test_build_catalogue_rejects_page_not_present_in_official_annual_archive(
    synthetic_inputs: tuple[dict[int, str], dict[str, str]],
) -> None:
    archives, pages = synthetic_inputs
    first_spec = catalogue.EVENT_SPECS[0]
    year = int(first_spec["year"])
    remaining_pages = [
        spec["source_url"].rsplit("/", 1)[-1]
        for spec in catalogue.EVENT_SPECS
        if spec["source_url"] != first_spec["source_url"]
        and int(spec["year"]) == year
    ]
    missing_link_archives = dict(archives)
    missing_link_archives[year] = _archive_page(year, remaining_pages)

    with pytest.raises(ValueError, match="not linked from 2020 archive"):
        catalogue.build_catalogue(
            archive_html=missing_link_archives,
            page_html=pages,
            retrieved_at_utc=RETRIEVED_AT,
        )


def test_validation_rejects_missing_wrong_count_duplicate_and_invalid_fields(
    synthetic_catalogue: pd.DataFrame,
) -> None:
    missing_column = synthetic_catalogue.drop(columns=["source_url"])
    with pytest.raises(ValueError, match="Catalogue missing columns"):
        catalogue.validate_catalogue(missing_column)

    wrong_count = synthetic_catalogue.iloc[:-1].copy()
    with pytest.raises(ValueError, match="Expected 40 records"):
        catalogue.validate_catalogue(wrong_count)

    duplicate_event = synthetic_catalogue.copy()
    duplicate_event.loc[1, "event_id"] = duplicate_event.loc[0, "event_id"]
    with pytest.raises(ValueError, match="Duplicate event_id"):
        catalogue.validate_catalogue(duplicate_event)

    duplicate_url = synthetic_catalogue.copy()
    duplicate_url.loc[1, "source_url"] = duplicate_url.loc[0, "source_url"]
    with pytest.raises(ValueError, match="Duplicate source_url"):
        catalogue.validate_catalogue(duplicate_url)

    missing_time = synthetic_catalogue.copy()
    missing_time.loc[0, "available_at_utc"] = None
    with pytest.raises(ValueError, match="exact official release time"):
        catalogue.validate_catalogue(missing_time)

    expectation = synthetic_catalogue.copy()
    expectation.loc[0, "expectation_consensus"] = "unknown"
    with pytest.raises(ValueError, match="Historical consensus"):
        catalogue.validate_catalogue(expectation)

    schedule = synthetic_catalogue.copy()
    schedule.loc[0, "announcement_schedule_known"] = True
    with pytest.raises(ValueError, match="preannounced"):
        catalogue.validate_catalogue(schedule)

    episode = synthetic_catalogue.copy()
    episode.loc[0, "programme_episode_id"] = "unfrozen_episode"
    with pytest.raises(ValueError, match="four frozen broad episodes"):
        catalogue.validate_catalogue(episode)


def test_no_execute_path_returns_ready_status_without_calling_execution(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_if_called(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("execute must not be called without --execute")

    monkeypatch.setattr(catalogue, "execute", fail_if_called)
    assert catalogue.main([]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {"status": "ready_not_executed", "event_pages": 40}


def test_execute_uses_requester_fake_and_writes_only_to_tmp_path(
    synthetic_inputs: tuple[dict[int, str], dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archives, pages = synthetic_inputs
    archive_by_url = {
        url: archives[year] for year, url in catalogue.ARCHIVE_URLS.items()
    }
    calls: list[tuple[str, dict[str, object]]] = []

    class Response:
        status_code = 200
        encoding = "utf-8"

        def __init__(self, content: str) -> None:
            self.content = content.encode("utf-8")

    def requester(url: str, **kwargs: object) -> Response:
        calls.append((url, kwargs))
        if url in archive_by_url:
            return Response(archive_by_url[url])
        if url in pages:
            return Response(pages[url])
        raise AssertionError(f"Unexpected request: {url}")

    output_root = tmp_path / "fed_liquidity_catalogue"
    monkeypatch.setattr(catalogue, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(catalogue, "CATALOGUE_PATH", output_root / "catalogue.csv")
    monkeypatch.setattr(catalogue, "MANIFEST_PATH", output_root / "manifest.json")
    monkeypatch.setattr(catalogue, "FREEZE_PATH", output_root / "freeze.json")
    monkeypatch.setattr(catalogue, "REPORT_PATH", output_root / "report.md")
    monkeypatch.setattr(catalogue, "RESULT_PATH", output_root / "result.json")

    result = catalogue.execute(overwrite=True, requester=requester)

    assert result["status"] == "completed_fed_liquidity_catalogue"
    assert result["event_pages"] == 40
    assert result["outcomes_read"] is False
    assert result["profit_used"] is False
    assert result["expectation_policy"].startswith("Unavailable")
    assert result["year_counts"] == {"2020": 32, "2021": 4, "2023": 3, "2024": 1}
    assert result["episode_counts"] == {
        "banking_stress_2023": 3,
        "btfp_lifecycle_2024": 1,
        "covid_2020_support": 32,
        "covid_2021_lifecycle": 4,
    }
    assert len(calls) == len(catalogue.ARCHIVE_URLS) + len(catalogue.EVENT_SPECS)
    assert all(
        kwargs["timeout"] == 45
        and "Federal Reserve" not in str(kwargs["headers"])
        for _url, kwargs in calls
    )

    output_paths = (
        catalogue.CATALOGUE_PATH,
        catalogue.MANIFEST_PATH,
        catalogue.FREEZE_PATH,
        catalogue.REPORT_PATH,
        catalogue.RESULT_PATH,
    )
    assert all(path.is_file() and path.is_relative_to(tmp_path) for path in output_paths)
    saved = pd.read_csv(catalogue.CATALOGUE_PATH)
    assert len(saved) == 40
    assert json.loads(catalogue.FREEZE_PATH.read_text(encoding="utf-8"))[
        "outcomes_read"
    ] is False
    first_url = catalogue.EVENT_SPECS[0]["source_url"]
    expected_hash = hashlib.sha256(pages[first_url].encode()).hexdigest()
    saved_hash = saved.loc[
        saved["source_url"].eq(first_url), "source_content_sha256"
    ].iloc[0]
    assert expected_hash == saved_hash

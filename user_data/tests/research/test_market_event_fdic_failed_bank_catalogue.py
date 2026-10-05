# ruff: noqa: S101

"""Outcome-blind tests for the FDIC failed-bank source catalogue."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from user_data.Custom_Launcher.research.context_features import (
    market_event_fdic_failed_bank_catalogue as fdic,
)


def source_row(identifier: int, name: str, date: str) -> dict[str, Any]:
    timestamp = pd.Timestamp(date)
    year = timestamp.year
    display_date = f"{timestamp.month}/{timestamp.day}/{timestamp.year}"
    return {
        "ID": str(identifier),
        "NAME": name,
        "CERT": identifier,
        "FIN": str(10000 + identifier),
        "CITYST": "CITY, ST",
        "FAILDATE": display_date,
        "FAILYR": str(year),
        "SAVR": "DIF",
        "RESTYPE": "FAILURE",
        "RESTYPE1": "PA",
        "QBFDEP": 100,
        "QBFASSET": 120,
        "COST": 10.5,
        "CHCLASS1": "NM",
    }


def complete_source() -> pd.DataFrame:
    rows = [
        source_row(1, "A", "2020-01-03"),
        source_row(2, "B", "2020-04-03"),
        source_row(3, "C", "2020-10-16"),
        source_row(4, "D", "2020-10-23"),
        source_row(5, "SILICON VALLEY BANK", "2023-03-10"),
        source_row(6, "SIGNATURE BANK", "2023-03-12"),
        source_row(7, "E", "2023-05-01"),
        source_row(8, "F", "2023-07-28"),
        source_row(9, "G", "2023-11-03"),
        source_row(10, "H", "2024-04-26"),
        source_row(11, "I", "2024-10-18"),
        source_row(12, "J", "2025-01-17"),
        source_row(13, "K", "2025-06-27"),
    ]
    return pd.DataFrame.from_records(rows)


def test_complete_catalogue_is_date_only_and_groups_close_failures() -> None:
    catalogue = fdic.build_catalogue(complete_source())

    assert len(catalogue) == 13
    assert catalogue["official_release_time_utc"].isna().all()
    assert not catalogue["intraday_eligible"].any()
    svb = catalogue.loc[catalogue["institution_name"].eq("SILICON VALLEY BANK")].iloc[0]
    signature = catalogue.loc[catalogue["institution_name"].eq("SIGNATURE BANK")].iloc[0]
    assert svb["episode_id"] == signature["episode_id"]
    assert svb["episode_members"] == 2
    assert catalogue["episode_id"].nunique() == 12


def test_failure_year_must_match_date() -> None:
    source = complete_source()
    source.loc[0, "FAILYR"] = "2021"

    with pytest.raises(ValueError, match="year disagrees"):
        fdic.build_catalogue(source)


def test_flatten_requires_frozen_total_and_complete_rows() -> None:
    payload = {
        "meta": {
            "total": 13,
            "parameters": {"filters": fdic.FILTER},
        },
        "data": [{"data": row} for row in complete_source().to_dict(orient="records")],
    }
    frame = fdic.flatten_source(payload)
    assert len(frame) == 13
    payload["meta"]["total"] = 12
    with pytest.raises(ValueError, match="count changed"):
        fdic.flatten_source(payload)


def test_fetch_rejects_non_official_final_url() -> None:
    class Response:
        content = b'{"meta":{},"data":[]}'
        status_code = 200
        url = "https://example.com/failures"

    def requester(*args: Any, **kwargs: Any) -> Response:
        assert kwargs["allow_redirects"] is False
        return Response()

    with pytest.raises(ValueError, match="approved official URL"):
        fdic.fetch_source(requester=requester)


def test_official_url_and_no_execute_are_strict(capsys: Any) -> None:
    assert fdic.official_api_url(fdic.API_URL)
    assert fdic.official_api_url(
        fdic.API_URL + "?filters=FAILDATE%3A%5B2020-01-01+TO+2025-12-31%5D"
    )
    assert not fdic.official_api_url("http://api.fdic.gov/banks/failures")
    assert not fdic.official_api_url("https://api.fdic.gov/banks/history")
    assert not fdic.official_api_url(fdic.API_URL + "?unexpected=true")
    assert fdic.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ready_not_executed"
    assert result["outcomes_will_be_read"] is False
    assert result["time_precision"] == "date_only"

# ruff: noqa: S101

from __future__ import annotations

import json

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_cpi_freeze as cpi,
)


def _fixture_sources() -> tuple[dict, dict]:
    releases = {
        "2021-01-01": "2021-02-10",
        "2021-02-01": "2021-03-10",
        "2021-03-01": "2021-04-13",
    }
    initial = {}
    vintages = {}
    dates = [
        "2020-01-01",
        "2020-02-01",
        "2020-03-01",
        "2020-12-01",
        "2021-01-01",
        "2021-02-01",
        "2021-03-01",
    ]
    values = {
        "2020-01-01": 100.0,
        "2020-02-01": 100.0,
        "2020-03-01": 100.0,
        "2020-12-01": 100.0,
        "2021-01-01": 101.0,
        "2021-02-01": 103.02,
        "2021-03-01": 104.0502,
    }
    for definition in cpi.SERIES.values():
        series_id = definition["id"]
        initial[series_id] = {
            "observations": [
                {
                    "date": observation,
                    "realtime_start": release,
                    "realtime_end": "9999-12-31",
                    "value": str(values[observation]),
                }
                for observation, release in releases.items()
            ]
        }
        vintage_rows = []
        for observation in dates:
            row = {"date": observation}
            for release in releases.values():
                row[f"{series_id}_{release.replace('-', '')}"] = str(values[observation])
            vintage_rows.append(row)
        vintages[series_id] = {"observations": vintage_rows}
    return initial, vintages


def test_release_timestamp_uses_historical_eastern_dst() -> None:
    assert cpi._release_anchor("2021-01-13") == "2021-01-13T13:30:00Z"
    assert cpi._release_anchor("2021-07-13") == "2021-07-13T12:30:00Z"


def test_catalog_uses_release_vintage_and_previous_release_change() -> None:
    initial, vintages = _fixture_sources()
    catalog, coverage = cpi.build_release_catalog(initial, vintages)
    assert len(catalog) == 3
    assert coverage.empty
    assert pd.isna(catalog.iloc[0]["temperature_headline_mom"])
    assert catalog.iloc[1]["headline_mom_released_pct"] == 2.0
    assert catalog.iloc[1]["temperature_headline_mom"] == 1
    assert catalog.iloc[2]["headline_mom_released_pct"] == 1.0
    assert catalog.iloc[2]["temperature_headline_mom"] == -1


def test_missing_initial_observation_is_excluded_not_imputed() -> None:
    initial, vintages = _fixture_sources()
    for definition in cpi.SERIES.values():
        series_id = definition["id"]
        initial[series_id]["observations"][1]["value"] = "."
    catalog, coverage = cpi.build_release_catalog(initial, vintages)
    assert len(catalog) == 2
    assert len(coverage) == 1
    assert coverage.iloc[0]["handling"] == "excluded_not_imputed"


def test_missing_comparison_vintage_excludes_release_without_guessing() -> None:
    initial, vintages = _fixture_sources()
    series_id = cpi.SERIES["headline_mom"]["id"]
    prior_row = next(
        row
        for row in vintages[series_id]["observations"]
        if row["date"] == "2021-01-01"
    )
    prior_row[f"{series_id}_20210310"] = "."
    catalog, coverage = cpi.build_release_catalog(initial, vintages)
    assert "cpi_2021_02" not in set(catalog["event_id"])
    missing = coverage.loc[coverage["observation_month"].eq("2021-02-01")].iloc[0]
    assert missing["status"] == "required_release_vintage_comparison_missing"
    assert missing["handling"] == "excluded_not_imputed"


def test_api_key_is_removed_from_request_contract_and_freeze() -> None:
    contract = cpi._safe_request_contract(
        {"series_id": "CPIAUCSL", "api_key": "sentinel-secret", "file_type": "json"}
    )
    assert "api_key" not in contract
    initial, vintages = _fixture_sources()
    catalog, coverage = cpi.build_release_catalog(initial, vintages)
    freeze = cpi.build_freeze_document(catalog, coverage, {"requests": contract})
    assert "sentinel-secret" not in json.dumps(freeze)
    assert freeze["outcomes_read"] is False

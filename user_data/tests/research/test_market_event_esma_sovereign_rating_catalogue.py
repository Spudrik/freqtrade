# ruff: noqa: S101

"""Outcome-blind tests for the ESMA sovereign-rating source audit."""

from __future__ import annotations

import json

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_event_esma_sovereign_rating_catalogue as esma,
)


def parent_rows() -> list[dict[str, object]]:
    records = []
    for number, (code, config) in enumerate(esma.JURISDICTIONS.items()):
        records.append(
            {
                "id": f"parent-{number}",
                "craCode": "CRA",
                "craName": "Official CRA",
                "issuerName": config["aliases"][0],
                "countryName": config["label"],
                "countryCode": code,
                "ratingTypeCode": "S",
                "sectorCode": "SV",
                "ratedObjectCode": "ISR",
                "timeHorizonType": "L",
                "timeHorizonDescr": "Long-term",
                "localForeignCurrencyCode": "LC",
                "localForeignCurrencyValue": "Local currency",
            }
        )
    records.append(
        {
            **records[1],
            "id": "excluded-bank",
            "issuerName": "Bank of England",
        }
    )
    return records


def action_rows(parents: pd.DataFrame) -> list[dict[str, object]]:
    records = []
    for number, row in enumerate(parents.itertuples(index=False)):
        if row.id == "excluded-bank":
            continue
        records.append(
            {
                "id": f"action-{number}",
                "actionId": f"source-action-{number}",
                "parent_id": row.id,
                "actionsActionType": "AF",
                "actionsActionTypeLabel": "Affirmation",
                "actionsRatingValueLabel": "AA",
                "actionsRacValidityDatetime": "2022-01-01T12:00:00Z",
                "actionsDefaultFlag": "No",
            }
        )
    return records


def test_parent_selection_keeps_fixed_national_aliases_only() -> None:
    selected = esma.selected_parent_frame(parent_rows())
    assert set(selected["countryCode"]) == set(esma.JURISDICTIONS)
    assert "Bank of England" not in set(selected["issuerName"])


def test_source_action_sign_uses_rating_action_not_price() -> None:
    assert esma.source_action_sign("DG", "Downgrade") == "negative_rating_action"
    assert esma.source_action_sign("UP", "Upgrade") == "positive_rating_action"
    assert (
        esma.source_action_sign("WR", "Removed under negative watch")
        == "positive_rating_action"
    )
    assert (
        esma.source_action_sign("AF", "Affirmation")
        == "unsigned_or_lifecycle_action"
    )
    assert (
        esma.source_action_sign("SP", "Removed under suspension")
        == "positive_rating_action"
    )
    assert (
        esma.source_action_sign("DF", "Removed from default status")
        == "positive_rating_action"
    )


def test_same_country_agency_and_clock_is_one_publication() -> None:
    parents = esma.selected_parent_frame(parent_rows())
    actions = action_rows(parents)
    actions.append(
        {
            **actions[0],
            "id": "duplicate-dimension",
            "actionId": "another-source-action",
            "actionsActionType": "DG",
            "actionsActionTypeLabel": "Downgrade",
            "actionsRatingValueLabel": "AA-",
        }
    )
    raw = esma.build_action_rows(parents, actions)
    catalogue = esma.build_catalogue(raw)
    us = catalogue.loc[catalogue["country_code"].eq("US")].iloc[0]
    assert us["raw_action_rows"] == 2
    assert us["source_action_sign"] == "negative_rating_action"
    assert len(catalogue) == len(esma.JURISDICTIONS)


def test_repeated_action_id_across_parent_dimensions_is_allowed() -> None:
    source_parents = parent_rows()
    second_us_parent = {
        **source_parents[0],
        "id": "parent-us-second-dimension",
        "localForeignCurrencyValue": "Foreign currency",
    }
    source_parents.append(second_us_parent)
    parents = esma.selected_parent_frame(source_parents)
    actions = action_rows(parents)
    us_first = actions[0]
    actions.append(
        {
            **us_first,
            "parent_id": "parent-us-second-dimension",
        }
    )

    raw = esma.build_action_rows(parents, actions)

    assert raw.loc[raw["id_action"].eq(us_first["id"]), "parent_id"].nunique() == 2


def test_event_ids_keep_fractional_time_and_raw_cra_identity() -> None:
    parents = esma.selected_parent_frame(parent_rows())
    actions = action_rows(parents)
    first = actions[0]
    actions.extend(
        [
            {
                **first,
                "id": "fractional-a",
                "actionId": "fractional-a",
                "actionsRacValidityDatetime": "2022-01-01T12:00:00.001Z",
            },
            {
                **first,
                "id": "fractional-b",
                "actionId": "fractional-b",
                "actionsRacValidityDatetime": "2022-01-01T12:00:00.999Z",
            },
        ]
    )
    raw = esma.build_action_rows(parents, actions)
    catalogue = esma.build_catalogue(raw)
    us = catalogue.loc[catalogue["country_code"].eq("US")]

    assert us["event_id"].is_unique
    assert us["market_window_group_id"].nunique() == 3


def test_official_url_is_strict_and_no_execute_is_offline(capsys) -> None:  # type: ignore[no-untyped-def]
    assert esma.official_solr_url(esma.SOLR_URL)
    assert not esma.official_solr_url(
        "http://registers.esma.europa.eu/solr/esma_registers_radar/select"
    )
    assert not esma.official_solr_url(
        "https://registers.esma.europa.eu/solr/another_core/select"
    )
    assert esma.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ready_not_executed"
    assert result["outcomes_will_be_read"] is False
    assert result["profit_will_be_used"] is False


def test_jurisdiction_query_uses_explicit_or_terms() -> None:
    query = esma.jurisdiction_query()

    assert "countryCode:(US OR GB OR JP OR CN" in query
    assert all(code in query for code in esma.JURISDICTIONS)

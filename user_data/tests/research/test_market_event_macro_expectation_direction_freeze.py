# ruff: noqa: S101

from __future__ import annotations

import importlib

import pandas as pd
from pandas import DataFrame


MODULE = (
    "user_data.Custom_Launcher.research.context_features."
    "market_event_macro_expectation_direction_freeze"
)
freeze = importlib.import_module(MODULE)


def event_row(**values: object) -> dict[str, object]:
    close = pd.Timestamp("2025-01-01T13:25:00Z")
    return {
        "expectation_episode_id": "episode",
        "source_block": freeze.PARTITIONS[0],
        "official_event_id": "official",
        "official_anchor_utc": pd.Timestamp("2025-01-01T13:30:00Z"),
        "maximum_expectation_lead_minutes": 5.0,
        "headline_cpi_close_time": close,
        "headline_cpi_lead_minutes": 5.0,
        "core_cpi_close_time": close,
        "core_cpi_lead_minutes": 5.0,
        "nonfarm_payrolls_close_time": close,
        "nonfarm_payrolls_lead_minutes": 5.0,
        "unemployment_rate_close_time": close,
        "unemployment_rate_lead_minutes": 5.0,
        **values,
    }


def test_cpi_agreement_adds_one_whole_event_vote() -> None:
    proxy = {"official": {"headline": -1, "core": -1, "agreement": -1}}
    rows = DataFrame([event_row(headline_cpi=1, core_cpi=1)])
    routes = freeze.build_cpi_routes(rows, proxy)
    assert {row["route_id"] for row in routes} == {
        "cpi_headline_surprise",
        "cpi_core_surprise",
        "cpi_headline_core_agreement",
    }
    assert all(row["predicted_base_direction"] == -1 for row in routes)


def test_cpi_disagreement_does_not_force_an_agreement_vote() -> None:
    proxy = {"official": {"headline": -1, "core": 1, "agreement": 0}}
    rows = DataFrame([event_row(headline_cpi=1, core_cpi=-1)])
    routes = freeze.build_cpi_routes(rows, proxy)
    assert {row["route_id"] for row in routes} == {
        "cpi_headline_surprise",
        "cpi_core_surprise",
    }


def test_cpi_misaligned_cutoffs_do_not_create_an_agreement_vote() -> None:
    proxy = {"official": {"headline": -1, "core": -1, "agreement": -1}}
    rows = DataFrame(
        [
            event_row(
                headline_cpi=1,
                core_cpi=1,
                core_cpi_close_time=pd.Timestamp("2025-01-01T13:29:00Z"),
            )
        ]
    )
    routes = freeze.build_cpi_routes(rows, proxy)
    assert "cpi_headline_core_agreement" not in {
        row["route_id"] for row in routes
    }


def test_jobs_conflict_abstains_instead_of_forcing_direction() -> None:
    rows = DataFrame([event_row(nonfarm_payrolls=1, unemployment_rate=1)])
    assert freeze.build_jobs_routes(rows, {}) == []


def test_jobs_agreement_creates_balance_and_strict_routes() -> None:
    rows = DataFrame([event_row(nonfarm_payrolls=1, unemployment_rate=-1)])
    routes = freeze.build_jobs_routes(rows, {"official": -1})
    assert {row["route_id"] for row in routes} == {
        "jobs_available_component_balance",
        "jobs_two_component_agreement",
    }
    assert all(row["predicted_base_direction"] == -1 for row in routes)


def test_jobs_one_available_component_is_balance_only() -> None:
    rows = DataFrame([event_row(nonfarm_payrolls=-1, unemployment_rate=float("nan"))])
    routes = freeze.build_jobs_routes(rows, {})
    assert [row["route_id"] for row in routes] == [
        "jobs_available_component_balance"
    ]
    assert routes[0]["predicted_base_direction"] == 1


def test_jobs_misaligned_cutoffs_use_only_the_earlier_component() -> None:
    rows = DataFrame(
        [
            event_row(
                nonfarm_payrolls=1,
                unemployment_rate=1,
                nonfarm_payrolls_close_time=pd.Timestamp("2025-01-01T13:25:00Z"),
                unemployment_rate_close_time=pd.Timestamp("2025-01-01T13:29:00Z"),
            )
        ]
    )
    routes = freeze.build_jobs_routes(rows, {})
    assert [row["route_id"] for row in routes] == [
        "jobs_available_component_balance"
    ]
    assert routes[0]["predicted_base_direction"] == -1
    assert "earlier payroll cutoff only" in routes[0]["source_detail"]


def test_route_counts_require_both_directions_in_both_periods() -> None:
    records = []
    for partition in freeze.PARTITIONS:
        for index in range(freeze.MINIMUM_EVENTS_PER_PARTITION):
            records.append(
                {
                    "route_id": "balanced",
                    "family": "test",
                    "source_block": partition,
                    "expectation_episode_id": f"{partition}_{index}",
                    "predicted_base_direction": 1 if index % 2 else -1,
                }
            )
            records.append(
                {
                    "route_id": "constant",
                    "family": "test",
                    "source_block": partition,
                    "expectation_episode_id": f"constant_{partition}_{index}",
                    "predicted_base_direction": 1,
                }
            )
    counts = freeze.route_counts(DataFrame.from_records(records)).set_index("route_id")
    assert bool(counts.loc["balanced", "historical_direction_test_eligible"])
    assert not bool(counts.loc["constant", "historical_direction_test_eligible"])

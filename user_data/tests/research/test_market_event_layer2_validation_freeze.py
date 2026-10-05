# ruff: noqa: S101

"""Tests for the Layer 2 breadth-balanced validation freeze."""

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_event_layer2_validation_freeze as frozen,
)


def test_validation_batch_has_all_five_jointly_reviewed_siblings() -> None:
    routes = frozen.definitions()

    assert {row["branch_id"] for row in routes} == {
        "sec_activity_full_family_randomization",
        "sec_clock_and_equity_alternative_controls",
        "cpi_transmission_full_family_randomization",
        "cpi_common_event_and_self_momentum",
        "boj_activity_full_family_randomization",
    }


def test_randomization_routes_use_fixed_familywide_iteration_count() -> None:
    routes = frozen.definitions()
    randomization = [row for row in routes if "randomization" in row["branch_id"]]

    assert len(randomization) == 3
    assert {row["permutations"] for row in randomization} == {2000}


def test_common_event_route_requires_increment_over_self_momentum() -> None:
    route = next(
        row
        for row in frozen.definitions()
        if row["branch_id"] == "cpi_common_event_and_self_momentum"
    )

    assert "same events" in route["pass_rule"]
    assert ">=5 points" in route["pass_rule"]

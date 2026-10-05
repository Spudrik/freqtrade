# ruff: noqa: S101

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_freeze as g22z,
)


def test_generation22_freezes_five_active_and_three_parked_routes() -> None:
    branches = g22z.branch_definitions()
    active = [item for item in branches if item["status_at_freeze"].startswith("frozen_")]
    parked = [item for item in branches if item["status_at_freeze"].startswith("parked_")]

    assert len(active) == 5
    assert len(parked) == 3
    assert len({item["route_family"] for item in active}) == 5


def test_freqai_route_uses_multiple_unsigned_horizons_not_twenty_four_hours() -> None:
    route = next(
        item
        for item in g22z.branch_definitions()
        if item["branch_id"] == "g22e_freqai_unsigned_reaction_interaction_regression"
    )

    assert route["required_tool"] == "FreqAI"
    assert len(route["continuous_targets"]) == 20
    assert all("h24" not in target for target in route["continuous_targets"])
    assert route["training_and_evaluation"]["signed_direction_used"] is False


def test_single_levels_remain_coequal_in_multitimeframe_route() -> None:
    route = next(
        item
        for item in g22z.branch_definitions()
        if item["branch_id"] == "g22b_multitimeframe_level_convergence_and_precedence"
    )

    assert route["singles_are_coequal"] is True
    assert "isolated_single_timeframe" in route["geometry_states"]

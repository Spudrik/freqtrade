# ruff: noqa: S101

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation11_freeze as freeze,
)


def test_portfolio_keeps_seven_distinct_route_families() -> None:
    assert len(freeze.ROUTES) == 7
    assert len({route.family for route in freeze.ROUTES}) == 7
    assert {route.route_id for route in freeze.ROUTES} == {
        "activity_displacement",
        "trend_timeframe_interaction",
        "crypto_market_alignment",
        "cluster_obstacle_geometry",
        "external_calm_and_stress",
        "market_group_portability",
        "timeframe_horizon_persistence",
    }


def test_horizon_grid_is_not_a_single_selected_horizon() -> None:
    assert freeze.HORIZONS == (1, 2, 4, 8)
    assert freeze.HORIZON_BANDS == {"short": (1, 2), "medium": (4, 8)}


def test_direction_portfolio_includes_simple_comparators_and_combinations() -> None:
    assert "development_majority_path" in freeze.DIRECTION_METHODS
    assert "always_through_approach_baseline" in freeze.DIRECTION_METHODS
    assert "local_and_crypto_market_consensus" in freeze.DIRECTION_METHODS
    assert len(freeze.DIRECTION_METHODS) >= 5

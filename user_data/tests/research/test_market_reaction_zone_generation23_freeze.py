# ruff: noqa: S101

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_freeze as g23z,
)


def test_generation23_freezes_five_distinct_active_routes() -> None:
    branches = g23z.branch_definitions()
    active = [item for item in branches if item["status_at_freeze"].startswith("frozen_")]
    parked = [item for item in branches if item["status_at_freeze"].startswith("parked_")]

    assert len(active) == 5
    assert len(parked) == 5
    assert len({item["route_family"] for item in active}) == 5


def test_cluster_attribution_cannot_be_called_confirmation() -> None:
    branch = g23z.branch_definitions()[0]

    assert branch["parent_primary_cell"]["horizon_hours"] == 8
    assert len(branch["parent_controls"]) == 5
    assert branch["attribution_comparisons"][0] == "matched_isolated_anchor_component"
    assert len(branch["attribution_comparisons"]) == 4
    assert "cannot be called confirmation" in branch["pass_rule"]


def test_freqai_reuses_exact_twelve_features_and_two_transforms() -> None:
    branch = next(
        item
        for item in g23z.branch_definitions()
        if item["branch_id"] == "g23c_freqai_calibrated_unsigned_reaction"
    )

    assert len(branch["features"]) == 12
    assert len(branch["target_transforms"]) == 2
    assert len(branch["controls"]) == 5
    assert branch["training_and_evaluation"]["signed_direction_used"] is False


def test_new_level_families_are_fixed_before_outcomes() -> None:
    branches = {item["branch_id"]: item for item in g23z.branch_definitions()}
    round_grid = branches["g23d_round_number_and_price_grid_zones"]
    distribution = branches["g23e_causal_price_distribution_boundaries"]

    assert round_grid["step_multipliers"] == [1.0, 0.5, 0.25]
    assert distribution["lookback_hours"] == [24, 72, 168, 720]
    assert distribution["quantiles"] == [0.10, 0.25, 0.50, 0.75, 0.90]

# ruff: noqa: S101

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation21_freeze as g21z,
)


def test_generation21_freezes_five_diverse_active_routes() -> None:
    branches = g21z.branch_definitions()
    active = [item for item in branches if item["status_at_freeze"].startswith("frozen_")]
    parked = [item for item in branches if item["status_at_freeze"].startswith("parked_")]

    assert len(active) == 5
    assert len(parked) == 2
    assert len({item["route_family"] for item in active}) == 5


def test_one_minute_branch_is_disjoint_and_bounded() -> None:
    branch = next(
        item
        for item in g21z.branch_definitions()
        if item["branch_id"] == "g21e_disjoint_one_minute_rejection_diagnostic"
    )

    assert branch["selection"]["exclude_generation20_episode_times"]
    assert branch["selection"]["maximum_episodes"] == 18
    assert branch["distance_thresholds_atr"] == [0.5, 1.0]
    assert branch["horizons_minutes"] == [15, 60, 240]

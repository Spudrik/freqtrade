# ruff: noqa: S101

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation20_freeze as g20z,
)


def test_generation20_freezes_complete_broad_batch() -> None:
    branches = g20z.branch_definitions()
    active = [
        item for item in branches if str(item["status_at_freeze"]).startswith("frozen_")
    ]
    parked = [
        item for item in branches if str(item["status_at_freeze"]).startswith("parked_")
    ]

    assert len(branches) == 7
    assert len(active) == 5
    assert len(parked) == 2
    assert len({item["route_family"] for item in active}) == 5


def test_generation20_direction_lane_is_bounded_before_signed_paths() -> None:
    lane = next(
        item
        for item in g20z.branch_definitions()
        if item["branch_id"] == "g20e_donchian_onset_one_minute_replay"
    )
    selection = lane["episode_selection"]

    assert selection["minimum_episodes"] == 6
    assert selection["maximum_episodes"] == 12
    assert selection["signed_path_opened_only_after_selection_freeze"]
    assert "No FreqAI" in lane["pass_rule"]

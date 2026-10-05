# ruff: noqa: S101

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation19_joint_review as g19j,
)


def test_generation20_queue_is_broad_and_batch_gated() -> None:
    queue = g19j.next_queue()
    active = [item for item in queue if str(item["status"]).startswith("queued_active")]
    parked = [item for item in queue if str(item["status"]).startswith("parked_")]

    assert len(queue) == 7
    assert len(active) == 5
    assert len(parked) == 2
    assert len({item["route_family"] for item in active}) == 5
    assert all(item["earliest_eligible_generation"] == 20 for item in queue)


def test_direction_lane_is_bounded_and_not_automatic_freqai() -> None:
    lane = next(
        item
        for item in g19j.next_queue()
        if item["branch_id"] == "g20e_donchian_onset_one_minute_replay"
    )

    assert lane["status"] == "queued_active_bounded_direction_lane"
    assert "6-12" in lane["smallest_useful_test"]
    assert "no automatic FreqAI" in lane["smallest_useful_test"]

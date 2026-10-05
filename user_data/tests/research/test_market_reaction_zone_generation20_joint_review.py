# ruff: noqa: S101

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation20_joint_review as g20j,
)


def test_generation21_queue_preserves_five_diverse_active_routes() -> None:
    queue = g20j.generation21_queue()
    active = [item for item in queue if item["status"].startswith("active_")]
    parked = [item for item in queue if item["status"].startswith("parked_")]

    assert len(active) == 5
    assert len(parked) == 2
    assert len({item["route_family"] for item in active}) == 5


def test_rejected_avwap_is_not_tuned_in_same_holdouts() -> None:
    avwap = next(
        item
        for item in g20j.generation21_queue()
        if item["branch_id"] == "g21f_change_point_avwap_refinement"
    )

    assert avwap["status"] == "parked_rejected_parent"
    assert "event fitting" in avwap["park_reason"]

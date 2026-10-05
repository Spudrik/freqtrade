# ruff: noqa: S101

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation21_joint_review as g21j,
)


def test_generation22_queue_preserves_five_diverse_active_routes() -> None:
    queue = g21j.generation22_queue()
    active = [item for item in queue if item["status"].startswith("active_")]
    parked = [item for item in queue if item["status"].startswith("parked_")]

    assert len(active) == 5
    assert len(parked) == 3
    assert len({item["route_family"] for item in active}) == 5


def test_generation22_includes_freqai_without_signed_direction() -> None:
    freqai = next(
        item
        for item in g21j.generation22_queue()
        if item["branch_id"] == "g22e_freqai_unsigned_reaction_interaction_regression"
    )

    assert freqai["status"] == "active_pending_freeze"
    assert "unsigned" in freqai["plain_question"]
    assert "no signed direction" in freqai["scope_guard"]


def test_failed_direction_and_same_holdout_vp_tuning_are_parked() -> None:
    queue = {item["branch_id"]: item for item in g21j.generation22_queue()}

    assert queue["g22f_immediate_one_minute_direction_refinement"]["status"].startswith(
        "parked_"
    )
    assert queue["g22g_volume_profile_role_or_geometry_tuning"]["status"].startswith(
        "parked_"
    )

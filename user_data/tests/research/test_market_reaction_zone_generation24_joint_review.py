# ruff: noqa: S101

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_joint_review as g24j,
)


def test_generation25_queue_preserves_five_materially_distinct_routes() -> None:
    queue = g24j.generation25_queue()
    active = [item for item in queue if item["status"].startswith("active_")]
    parked = [item for item in queue if item["status"].startswith("parked_")]

    assert len(active) == 5
    assert len(parked) == 5
    assert len({item["route_family"] for item in active}) == 5
    assert any("VWAP" in item["plain_question"] for item in active)
    assert any("Volume Profile" in item["plain_question"] for item in active)
    assert any("orderbook" in item["plain_question"] for item in active)


def test_generation25_does_not_reopen_generation24_nulls_as_active() -> None:
    queue = {item["branch_id"]: item for item in g24j.generation25_queue()}

    assert queue["g25f_more_generic_indicator_level_tuning"]["status"].startswith(
        "parked_"
    )
    assert queue["g25g_more_cross_asset_band_repair"]["status"].startswith(
        "parked_"
    )
    assert queue["g25h_level_inclusive_activity_model"]["status"].startswith(
        "parked_"
    )


def test_vwap_candidate_requires_fresh_data_and_all_controls() -> None:
    branch = next(
        item
        for item in g24j.generation25_queue()
        if item["branch_id"] == "g25a_daily_anchored_vwap_fresh_confirmation"
    )

    assert "genuinely later data" in branch["plain_question"]
    assert "both fresh chronological blocks" in branch["pass_rule"]
    assert "recent ordinary" in branch["pass_rule"]
    assert "price-shift" in branch["pass_rule"]


def test_convergence_repair_keeps_the_whole_family_outcome_blind() -> None:
    branch = next(
        item
        for item in g24j.generation25_queue()
        if item["branch_id"] == "g25b_convergence_control_representation_repair"
    )

    assert "before outcomes" in branch["scope_guard"]
    assert "entire frozen" in branch["scope_guard"]
    assert "Do not select the best" in branch["scope_guard"]

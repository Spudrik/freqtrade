# ruff: noqa: S101

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_joint_review as g24j,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation25_freeze as g25z,
)


def synthetic_queue() -> dict:
    return {"siblings": g24j.generation25_queue()}


def synthetic_coverage() -> dict:
    return {
        "branch_gates": {
            "g25a_daily_anchored_vwap_fresh_confirmation": {
                "advance": False,
                "status": "parked_insufficient_fresh_ohlcv",
            },
            "g25b_convergence_control_representation_repair": {"advance": True},
            "g25c_connected_volume_profile_zones": {"advance": True},
            "g25d_market_state_only_activity_portability": {"advance": True},
            "g25e_external_source_overlap_gate": {
                "advance": False,
                "status": "parked_no_new_two_block_timestamp_safe_source",
            },
        }
    }


def test_generation25_freezes_only_three_supported_outcome_routes() -> None:
    branches = g25z.branch_definitions(synthetic_queue(), synthetic_coverage())
    active = [item for item in branches if item["status_at_freeze"].startswith("frozen_")]
    gated_parked = [
        item
        for item in branches
        if item["status_at_freeze"].startswith("parked_")
        and item.get("coverage_gate") is not None
    ]

    assert len(active) == 3
    assert len(gated_parked) == 2
    assert len({item["route_family"] for item in active}) == 3


def test_fresh_vwap_and_external_sources_remain_parked() -> None:
    branches = {
        item["branch_id"]: item
        for item in g25z.branch_definitions(synthetic_queue(), synthetic_coverage())
    }

    assert branches[
        "g25a_daily_anchored_vwap_fresh_confirmation"
    ]["status_at_freeze"].startswith("parked_")
    assert branches[
        "g25e_external_source_overlap_gate"
    ]["status_at_freeze"].startswith("parked_")


def test_connected_volume_profile_stays_research_only() -> None:
    branches = {
        item["branch_id"]: item
        for item in g25z.branch_definitions(synthetic_queue(), synthetic_coverage())
    }
    connected = branches["g25c_connected_volume_profile_zones"]

    assert connected["status_at_freeze"] == "frozen_pending_support"
    assert "research-only variants" in connected["scope_guard"]
    assert "Do not edit canonical indicators" in connected["scope_guard"]

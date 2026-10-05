# ruff: noqa: S101

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_freeze as g24z,
)


def test_generation24_freezes_five_distinct_active_routes() -> None:
    branches = g24z.branch_definitions()
    active = [item for item in branches if item["status_at_freeze"].startswith("frozen_")]
    parked = [item for item in branches if item["status_at_freeze"].startswith("parked_")]

    assert len(active) == 5
    assert len(parked) == 5
    assert len({item["route_family"] for item in active}) == 5


def test_freqai_stability_keeps_full_activity_family_and_six_controls() -> None:
    branch = next(
        item
        for item in g24z.branch_definitions()
        if item["branch_id"] == "g24a_freqai_activity_lead_stability_and_attribution"
    )

    assert len(branch["features"]) == 12
    assert len(branch["targets"]) == 8
    assert len(branch["controls"]) == 6
    assert len(branch["candidate_models"][0]["seeds"]) == 3


def test_context_repair_forbids_winner_filtering() -> None:
    branch = next(
        item
        for item in g24z.branch_definitions()
        if item["branch_id"] == "g24b_cross_asset_context_no_level_coverage_repair"
    )

    assert branch["winner_filtering_allowed"] is False
    assert branch["no_level_minimum_spacing_hours"] == 2
    assert "same_band_no_level_time" in branch["controls"]


def test_new_level_and_indicator_routes_are_causal_and_bounded() -> None:
    branches = {item["branch_id"]: item for item in g24z.branch_definitions()}
    vwap = branches["g24d_causal_anchored_vwap_zones"]
    generic = branches["g24e_generic_multitimeframe_indicator_context"]

    assert vwap["anchor_timeframes"] == ["1d", "1w", "1M"]
    assert vwap["dispersion_multipliers"] == [0.0, 1.0, 2.0]
    assert generic["arbitrary_multi_indicator_conjunctions_allowed"] is False
    assert generic["timeframes"] == ["1h", "4h", "1d"]

# ruff: noqa: S101

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation15_freeze as g15,
)


def test_generation15_freezes_seven_materially_distinct_siblings() -> None:
    freeze = g15.build_freeze()
    route_ids = {row["route_id"] for row in freeze["sibling_routes"]}

    assert freeze["status"] == "frozen_before_generation15_outcomes"
    assert freeze["source_summary"]["generation15_outcome_values_read"] is False
    assert route_ids == {
        "reaction_path_decomposition",
        "level_family_attribution",
        "source_timeframe_attribution",
        "contact_activity_combinations",
        "contact_lifecycle_and_approach",
        "explicit_cluster_composition",
        "historical_btc_orderbook_conditioning",
    }
    assert freeze["sibling_count"] == 7


def test_generation15_keeps_direction_profit_and_descendants_closed() -> None:
    freeze = g15.build_freeze()

    assert freeze["research_boundary"] == {
        "profit_used": False,
        "future_signed_direction_tested": False,
        "joint_55_percent_direction_target_reached": False,
        "trading_promotion": False,
    }
    assert freeze["sequencing"]["no_generation16_descendant_before_joint_review"]
    assert freeze["sequencing"]["automatic_descendant_launch"] is False
    assert freeze["runtime"]["maximum_workers"] == 4


def test_generation15_preserves_full_horizon_and_market_scope() -> None:
    freeze = g15.build_freeze()

    assert freeze["horizons_hours"] == [1, 2, 4, 8]
    assert set(freeze["cohorts"]) == {"normal", "meme"}
    assert set(freeze["normal_groups_predeclared"]) == {
        "btc_separate",
        "smart_contract_platforms",
        "other_established_alts",
    }
    assert "frozen top-ten" in freeze["meme_scope"]

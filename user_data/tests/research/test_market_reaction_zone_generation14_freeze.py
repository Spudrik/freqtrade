# ruff: noqa: S101

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation14_freeze as g14,
)


def test_generation14_registry_has_complete_broad_siblings() -> None:
    profiles, comparisons = g14.build_registry()

    assert profiles
    assert comparisons
    assert {item["stage"] for item in comparisons} == set(g14.STAGES)
    assert {item["cohort"] for item in comparisons} == set(g14.COHORTS)
    assert len({item["comparison_id"] for item in comparisons}) == len(comparisons)
    assert all(item["candidate"] in profiles for item in comparisons)
    assert all(item["baseline"] in profiles for item in comparisons)

    mtf_routes = {
        item["route_id"] for item in comparisons if item["stage"] == g14.MTF_STAGE
    }
    assert mtf_routes == {
        "eight_hour_participation_pressure",
        "eight_hour_volatility_compression",
        "eight_hour_full_decomposition",
        "eight_hour_level_identity",
        "eight_hour_cluster_interaction",
        "eight_hour_wider_market_interaction",
        "eight_hour_oscillator_interaction",
    }
    long_routes = {
        item["route_id"] for item in comparisons if item["stage"] == g14.LONG_STAGE
    }
    assert long_routes == {"four_hour_level_local_plus_eight_hour_activity"}


def test_generation14_combinations_have_component_and_timing_controls() -> None:
    _, comparisons = g14.build_registry()
    by_route: dict[tuple[str, str], set[str]] = {}
    for item in comparisons:
        by_route.setdefault((item["cohort"], item["route_id"]), set()).add(
            item["control_type"]
        )

    for cohort in g14.COHORTS:
        cluster = by_route[(cohort, "eight_hour_cluster_interaction")]
        assert "eight_hour_activity_only" in cluster
        assert "additional_component_without_eight_hour" in cluster
        assert any("shuffled" in name for name in cluster)
        long_controls = by_route[
            (cohort, "four_hour_level_local_plus_eight_hour_activity")
        ]
        assert {"local_activity_only", "eight_hour_activity_only"}.issubset(
            long_controls
        )
        assert {
            "causal_72h_old_local_activity",
            "causal_72h_old_eight_hour_activity",
        }.issubset(long_controls)
        assert any("shuffled" in name for name in long_controls)


def test_generation14_labels_and_groups_are_predeclared() -> None:
    assert g14.REACTION_PRICE_THRESHOLDS_ATR == (0.35, 0.5, 0.75)
    assert g14.REACTION_VOLUME_THRESHOLDS == (1.1, 1.25, 1.5)
    assert g14.REACTION_HORIZONS == (1, 2, 4, 8)
    assert set(g14.NORMAL_GROUPS) == {
        "btc_separate",
        "smart_contract_platforms",
        "other_established_alts",
    }
    assert len(g14.NORMAL_GROUPS["smart_contract_platforms"]) == 5

# ruff: noqa: S101

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation12_freeze as g12,
)


def test_generation12_registry_has_complete_frozen_sibling_stages() -> None:
    profiles, comparisons = g12.build_registry()

    assert profiles
    assert comparisons
    assert {item["stage"] for item in comparisons} == set(g12.STAGES)
    assert {item["cohort"] for item in comparisons} == {"normal", "meme"}
    recent = [item for item in comparisons if item["stage"] == g12.STAGE_RECENT]
    assert recent
    assert {item["cohort"] for item in recent} == {"normal"}
    assert {item["route_id"] for item in recent} == {
        "level_identity_and_timeframe",
        "isolated_and_cluster_geometry",
        "local_activity_and_volatility",
        "local_trend_and_momentum",
        "wider_crypto_market_alignment",
    }
    assert all(item["candidate"] in profiles for item in comparisons)
    assert all(item["baseline"] in profiles for item in comparisons)


def test_every_route_declares_exactly_its_complete_control_ladder() -> None:
    _, comparisons = g12.build_registry()
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for item in comparisons:
        key = (item["stage"], item["cohort"], item["route_id"])
        grouped.setdefault(key, []).append(item)

    assert grouped
    for items in grouped.values():
        expected = items[0]["expected_controls_for_route"]
        assert len(items) == expected
        assert len({item["control_type"] for item in items}) == expected


def test_full_activity_decomposition_must_beat_both_subfamilies() -> None:
    _, comparisons = g12.build_registry()
    controls = {
        item["control_type"]
        for item in comparisons
        if item["stage"] == g12.STAGE_ATTRIBUTION
        and item["cohort"] == "normal"
        and item["route_id"] == "activity_full_decomposition"
    }

    assert controls == {
        "level_only",
        "subset_activity_participation",
        "subset_activity_volatility",
        "causal_stale",
        "within_period_nonself_shuffle",
    }


def test_three_family_combination_must_beat_pairwise_and_single_models() -> None:
    _, comparisons = g12.build_registry()
    controls = {
        item["control_type"]
        for item in comparisons
        if item["stage"] == g12.STAGE_ATTRIBUTION
        and item["cohort"] == "meme"
        and item["route_id"] == "activity_plus_trend_plus_market"
    }

    assert controls == {
        "pair_activity_trend",
        "pair_activity_market",
        "single_activity",
        "single_trend",
        "single_market",
        "level_only",
        "all_additions_causal_stale",
        "all_additions_within_period_nonself_shuffle",
    }


def test_profiles_contain_no_outcome_or_direction_columns() -> None:
    profiles, _ = g12.build_registry()

    forbidden = ("reaction_h", "volume_ratio_h", "future", "direction")
    assert all(
        not any(token in column for token in forbidden)
        for profile in profiles.values()
        for column in profile["feature_columns"]
    )

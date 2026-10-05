# ruff: noqa: S101

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation8 as g8,
)


def test_initial_registry_covers_every_cell_and_frozen_target() -> None:
    assert len(g8.QUESTIONS) == 19
    assert len(g8.PROFILES) == 404
    assert len(g8.COMPARISONS) == 385
    assert len(g8.profiles_for_cohort("normal")) == 212
    assert len(g8.profiles_for_cohort("meme")) == 192
    assert all(profile["seed"] == 42 for profile in g8.PROFILES.values())
    assert all(
        target.startswith("&-g6_")
        for profile in g8.PROFILES.values()
        for target in profile["targets"]
    )


def test_every_add_one_route_has_simpler_stale_and_shuffle_controls() -> None:
    grouped: dict[tuple[str, str], set[str]] = {}
    for comparison in g8.COMPARISONS:
        if comparison["route_type"] != "add_one_dimension":
            continue
        key = (comparison["question_id"], comparison["route_id"])
        grouped.setdefault(key, set()).add(comparison["control_type"])
    assert grouped
    assert all(
        controls
        == {
            "immediately_simpler",
            "causal_stale",
            "within_period_nonself_shuffle",
        }
        for controls in grouped.values()
    )


def test_complete_routes_include_base_stale_shuffle_and_every_leave_one_out() -> None:
    for question in g8.QUESTIONS:
        controls = {
            comparison["control_type"]
            for comparison in g8.COMPARISONS
            if comparison["question_id"] == question.question_id
            and comparison["route_id"] == "complete_attribution"
        }
        expected = {
            "all_dimensions_vs_base",
            "all_dimensions_causal_stale",
            "all_dimensions_nonself_shuffle",
            *(f"leave_one_out_{name}" for name, _ in question.dimensions),
        }
        assert controls == expected


def test_technical_smoke_keeps_one_baseline_and_complete_profile_per_cell() -> None:
    assert len(g8.smoke_profiles("normal")) == 20
    assert len(g8.smoke_profiles("meme")) == 18
    assert all(
        g8.PROFILES[profile]["role"] in {"baseline", "complete"}
        for profile in g8.smoke_profiles("normal")
    )


def test_orderbook_question_uses_only_smart_contract_pair_scope() -> None:
    profiles = [
        profile
        for profile in g8.PROFILES.values()
        if profile["branch_id"].startswith("g8g_")
    ]
    assert profiles
    assert all(profile["pair_scope"] == "smart_contract_platforms" for profile in profiles)


def test_confirmation_registry_uses_only_the_two_predeclared_seeds() -> None:
    profiles, comparisons = g8.build_registry(g8.CONFIRMATION_SEEDS)

    assert len(profiles) == 808
    assert len(comparisons) == 770
    assert {profile["seed"] for profile in profiles.values()} == {17, 73}
    assert {comparison["seed"] for comparison in comparisons} == {17, 73}


def test_active_comparisons_accepts_an_explicit_registry() -> None:
    profiles, comparisons = g8.build_registry((17,))
    question = g8.QUESTIONS[0]
    selected = {
        g8.profile_id(question, "baseline", 17),
        g8.profile_id(question, "add_identity", 17),
        g8.profile_id(question, "add_identity_stale", 17),
        g8.profile_id(question, "add_identity_shuffled", 17),
    }

    active = g8.active_comparisons(
        sorted(selected), registry_comparisons=comparisons
    )

    assert len(active) == 3
    assert {item["route_id"] for item in active} == {"identity"}
    assert all(item["candidate"] in profiles for item in active)

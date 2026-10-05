# ruff: noqa: S101

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation10_freeze as freeze,
)


def test_registry_keeps_both_mechanisms_separate() -> None:
    profiles, comparisons = freeze.build_registry()

    assert len(profiles) == 24
    assert len(comparisons) == 18
    assert {item["route_id"] for item in comparisons} == {
        "relative_volume",
        "participation_duration",
    }
    assert all(item["expected_controls_for_route"] == 3 for item in comparisons)


def test_candidate_never_combines_the_two_retained_mechanisms() -> None:
    profiles, _ = freeze.build_registry()
    candidate_profiles = [item for item in profiles.values() if item["role"] == "current"]

    for item in candidate_profiles:
        blocks = set(item["blocks"])
        assert not {
            "g8_participation_relative_volume",
            "g8_participation_duration",
        }.issubset(blocks)


def test_base_control_represents_candidate_unavailability() -> None:
    _, comparisons = freeze.build_registry()
    unavailable = [
        item
        for item in comparisons
        if item["control_type"] == "candidate_absent_or_unavailable"
    ]

    assert len(unavailable) == len(freeze.QUESTIONS) * len(freeze.SEEDS)
    assert all(item["baseline_role"] == "baseline" for item in unavailable)

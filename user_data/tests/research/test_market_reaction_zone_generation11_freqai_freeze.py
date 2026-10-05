# ruff: noqa: S101

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation11_freqai_freeze as freeze,
)


def test_registry_is_broad_and_has_complete_controls() -> None:
    profiles, comparisons = freeze.build_registry()

    assert len({item["question_id"] for item in comparisons}) == 7
    assert len({item["route_id"] for item in comparisons}) == 8
    assert {profile["cohort"] for profile in profiles.values()} == {"normal", "meme"}
    for (_cohort, route), rows in __import__("itertools").groupby(
        sorted(comparisons, key=lambda item: (item["cohort"], item["route_id"])),
        key=lambda item: (item["cohort"], item["route_id"]),
    ):
        grouped = list(rows)
        assert len(grouped) == grouped[0]["expected_controls_for_route"]


def test_targets_cover_reaction_and_volume_at_four_horizons() -> None:
    assert len(freeze.TARGETS) == 8
    for horizon in (1, 2, 4, 8):
        assert f"&-g11_reaction_h{horizon}" in freeze.TARGETS
        assert f"&-g11_volume_ratio_h{horizon}" in freeze.TARGETS


def test_combination_has_leave_one_family_out_controls() -> None:
    _, comparisons = freeze.build_registry()
    combination = [
        item
        for item in comparisons
        if item["cohort"] == "normal"
        and item["route_id"] == "low_dimensional_combination"
    ]

    assert len(combination) == 6
    assert sum(item["control_type"].startswith("leave_out_") for item in combination) == 3

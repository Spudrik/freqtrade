# ruff: noqa: S101

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation13_freeze as g13,
)


def test_generation13_registry_is_broad_and_balanced() -> None:
    profiles, comparisons = g13.build_registry()

    assert len(profiles) == 58
    assert len(comparisons) == 54
    for cohort in g13.COHORTS:
        mtf_profiles = [
            row
            for row in profiles.values()
            if row["cohort"] == cohort and row["stage"] == g13.MTF_STAGE
        ]
        long_profiles = [
            row
            for row in profiles.values()
            if row["cohort"] == cohort and row["stage"] == g13.LONG_STAGE
        ]
        assert len(mtf_profiles) == 19
        assert len(long_profiles) == 10


def test_every_freqai_route_has_three_distinct_controls() -> None:
    _, comparisons = g13.build_registry()
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in comparisons:
        grouped.setdefault((row["stage"], row["cohort"], row["route_id"]), []).append(row)

    assert len(grouped) == 18
    for rows in grouped.values():
        assert len(rows) == 3
        assert len({row["baseline"] for row in rows}) == 3
        assert all(row["expected_controls_for_route"] == 3 for row in rows)


def test_higher_timeframe_profiles_keep_one_hour_activity_base() -> None:
    profiles, _ = g13.build_registry()
    current = profiles[g13.profile_id(g13.MTF_STAGE, "normal", "4h_trend_momentum")]

    assert set(g13.mtf_base_columns()).issubset(current["feature_columns"])
    assert set(g13.mtf_columns("4h", "trend_momentum")).issubset(
        current["feature_columns"]
    )


def test_long_horizon_targets_are_direction_neutral() -> None:
    assert g13.LONG_HORIZONS == (12, 24, 48)
    assert all("reaction" in target or "volume_ratio" in target for target in g13.LONG_TARGETS)
    assert not any("direction" in target or "profit" in target for target in g13.LONG_TARGETS)

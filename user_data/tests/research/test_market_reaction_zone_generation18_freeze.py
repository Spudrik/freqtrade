# ruff: noqa: S101

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation18_freeze as g18,
)


def test_generation18_freezes_all_siblings_together() -> None:
    branches = g18.branch_definitions()

    assert len(branches) == 6
    assert sum(item["status_at_freeze"] == "frozen_pending" for item in branches) == 5
    assert sum(item["status_at_freeze"].startswith("parked") for item in branches) == 1
    assert branches[-1]["branch_id"] == "g18f_direction_after_reaction_gate"


def test_generation18_preserves_breadth_and_direction_boundary() -> None:
    branches = {item["branch_id"]: item for item in g18.branch_definitions()}

    assert branches["g18c_level_source_confirmation"]["volume_profile_specs"]
    assert branches["g18c_level_source_confirmation"]["donchian_specs"]
    assert branches["g18d_context_and_market_regimes"][
        "maximum_active_blocks_per_interaction"
    ] == 2
    assert branches["g18e_coin_group_and_timeframe_portability"][
        "source_timeframes"
    ] == ["1h", "4h", "8h"]
    assert "55%" in branches["g18f_direction_after_reaction_gate"]["park_reason"]


def test_confirmation_periods_are_two_non_overlapping_blocks_per_cohort() -> None:
    for periods in g18.CONFIRMATION_PERIODS.values():
        assert len(periods) == 2
        assert periods[0]["end_utc_exclusive"] <= periods[1]["start_utc"]

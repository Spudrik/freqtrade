# ruff: noqa: S101

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation19_freeze as g19,
)


def test_generation19_freezes_complete_breadth_layer() -> None:
    branches = g19.branch_definitions()

    assert len(branches) == 7
    assert sum(item["status_at_freeze"] == "frozen_pending" for item in branches) == 5
    assert sum(item["status_at_freeze"].startswith("parked") for item in branches) == 2


def test_generation19_keeps_small_interactions_and_multiple_horizons() -> None:
    branches = {item["branch_id"]: item for item in g19.branch_definitions()}

    assert branches["g19b_regime_level_incremental_combinations"][
        "maximum_active_blocks_per_interaction"
    ] == 2
    assert g19.HORIZONS_HOURS == (1, 2, 4, 8)
    assert branches["g19d_recross_timing_and_activity_onset"]["pre_windows_hours"] == [
        2,
        4,
    ]


def test_generation19_direction_and_missing_external_sources_stay_parked() -> None:
    branches = {item["branch_id"]: item for item in g19.branch_definitions()}

    assert branches["g19f_external_context_accumulation"][
        "status_at_freeze"
    ] == "parked_coverage"
    assert "55%" in branches["g19g_direction_after_reaction_gate"]["plain_question"]

# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation25_joint_review as g25j,
)


def test_closest_question_is_selected_without_changing_questions() -> None:
    scores = pd.DataFrame(
        {
            "scope_value": ["a", "a", "b", "b"],
            "metric": ["crossing_count"] * 4,
            "horizon_hours": [2] * 4,
            "market_scope": ["top_ten_memes"] * 4,
            "point_period_pass": [True, False, True, True],
            "strict_period_pass": [False, False, True, True],
            "equal_coin_difference": [0.2, 0.1, 0.05, 0.04],
        }
    )
    closest = g25j._closest_direct_question(scores)
    assert closest["scope_value"] == "b"
    assert closest["point_passes"] == 2


def test_terminal_review_does_not_launch_descendant_batch() -> None:
    result = g25j.review(g25j.DEFAULT_RUN_ID, overwrite=True)
    assert result["batch_execution_limit"]["generation25_batches_run"] == 1
    assert result["batch_execution_limit"]["descendant_batch_launched"] is False
    assert result["research_boundary"][
        "activity_success_not_reaction_and_direction_success"
    ] is True
    assert result["route_results"]["market_state_activity"][
        "retained_scope_target_rows"
    ] == 15

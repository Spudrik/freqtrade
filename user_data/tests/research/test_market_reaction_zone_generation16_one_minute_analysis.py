# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation16_one_minute_analysis as g16m,
)


def test_horizon_outcome_requires_both_excursion_and_volume() -> None:
    paths = []
    for horizon in g16m.DIRECTION_HORIZONS:
        paths.append(
            {
                "checkpoint_minutes": horizon,
                "first_up_threshold_minute": 2,
                "first_down_threshold_minute": None,
                "maximum_absolute_excursion_half_widths": 2.0,
                "volume_ratio_post_pre": 1.0 if horizon == 15 else 2.0,
                "close_displacement_half_widths": 1.0,
                "zone_overlap_fraction": 0.25,
                "close_crossings_of_reference": 1,
            }
        )

    outcomes = pd.DataFrame(g16m.horizon_outcomes(paths)).set_index(
        "horizon_minutes"
    )

    assert not bool(outcomes.loc[15, "reaction"])
    assert bool(outcomes.loc[60, "reaction"])
    assert outcomes.loc[60, "first_direction"] == "up"


def test_joint_score_counts_abstention_as_failure() -> None:
    calls = pd.DataFrame(
        {
            "episode_id": ["a", "b"],
            "cohort": ["normal", "normal"],
            "period": ["early", "late"],
            "pair": ["A", "B"],
            "method": ["test", "test"],
            "method_kind": ["causal_candidate", "causal_candidate"],
            "call_numeric": [1, 0],
            "call_direction": ["up", "abstain"],
            "issued": [True, False],
            "horizon_minutes": [60, 60],
            "reaction": [True, True],
            "first_direction": ["up", "up"],
            "first_direction_numeric": [1, 1],
            "direction_callable": [True, True],
            "direction_correct": [True, False],
            "joint_reaction_and_direction": [True, False],
        }
    )

    score = g16m.call_summary(calls).loc[lambda frame: frame["scope"].eq("all")].iloc[0]

    assert score["joint_success_rate_all_episodes"] == 0.5
    assert score["joint_success_rate_issued_calls"] == 1.0


def test_frozen_direction_methods_are_unique() -> None:
    assert len(g16m.CALL_METHODS) == len(set(g16m.CALL_METHODS))
    assert len(g16m.DIRECTION_HORIZONS) >= 4

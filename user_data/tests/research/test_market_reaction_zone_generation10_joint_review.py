# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation10_joint_review as review,
)


def test_control_period_summary_counts_point_and_strict_results(monkeypatch) -> None:
    monkeypatch.setattr(review, "EXPECTED_PERIOD_TESTS", 3)
    frame = pd.DataFrame(
        {
            "question_id": ["q"] * 3,
            "route_id": ["route"] * 3,
            "control_type": ["shuffle"] * 3,
            "period": ["late"] * 3,
            "point_positive": [True, "true", False],
            "strict_period_pass": [True, False, False],
            "provisional_period_pass": [True, True, False],
            "equal_coin_paired_mae_gain": [0.03, 0.01, -0.02],
        }
    )

    result = review.summarize_control_periods(frame).iloc[0]

    assert result["point_positive_tests"] == 2
    assert result["strict_tests"] == 1
    assert result["provisional_tests"] == 2
    assert result["minimum_equal_coin_paired_mae_gain"] == -0.02


def test_activity_summary_remains_descriptive(monkeypatch) -> None:
    monkeypatch.setattr(review, "EXPECTED_PERIOD_TESTS", 3)
    frame = pd.DataFrame(
        {
            "question_id": ["q"] * 3,
            "route_id": ["route"] * 3,
            "activity_regime": ["active"] * 3,
            "equal_coin_paired_mae_gain": [0.03, 0.02, -0.01],
            "descriptive_only": [True, True, True],
        }
    )

    result = review.summarize_activity_regimes(frame).iloc[0]

    assert result["point_positive_tests"] == 2
    assert result["point_negative_tests"] == 1
    assert bool(result["descriptive_only"])
    assert result["control_scope"].endswith("base_only")


def test_pair_observation_requires_every_control_and_test(monkeypatch) -> None:
    monkeypatch.setattr(review, "EXPECTED_PAIR_TESTS", 2)
    controls = [control_type for _, control_type in review.g10z.CONTROL_ROLES]
    frame = pd.DataFrame.from_records(
        [
            {
                "question_id": "q",
                "route_id": "route",
                "pair": "BNB/USDT:USDT",
                "control_type": control,
                "paired_mae_gain": gain,
            }
            for control in controls
            for gain in (0.01, 0.02)
        ]
    )

    controls_summary = review.summarize_pair_controls(frame)
    result = review.pair_observations(controls_summary).iloc[0]

    assert result["controls_all_six_point_positive"] == 3
    assert bool(result["all_three_controls_all_six_point_positive"])
    assert result["interpretation"].startswith("outcome_selected")

# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation9_joint_review as review,
)


def test_question_summary_preserves_smaller_group_provisional_evidence() -> None:
    decisions = pd.DataFrame(
        {
            "question_id": ["q", "q"],
            "group_id": ["smaller_group", "full_group"],
            "status": [review.MODEL_PROVISIONAL, review.MODEL_FAILED],
            "minimum_equal_coin_paired_mae_gain": [0.002, -0.001],
            "minimum_bootstrap_lower": [-0.001, -0.004],
        }
    )
    frozen = {
        "limited_three_source_family": {
            "questions": [
                {
                    "question_id": "q",
                    "cohort": "normal",
                    "plain_name": "plain",
                    "mechanism": "mechanism",
                    "target": "target",
                }
            ]
        }
    }

    result = review.model_question_summary(decisions, frozen).iloc[0]

    assert result["status"] == "retained_provisional_only"
    assert result["provisional_groups"] == "smaller_group"
    assert result["failed_groups"] == "full_group"


def test_one_minute_summary_keeps_normal_and_meme_separate() -> None:
    actual = pd.DataFrame(
        {
            "cohort": ["normal", "normal", "meme", "meme"],
            "pair": ["A", "B", "M1", "M2"],
            "reaction_60m": [True, True, True, False],
            "first_zone_resolution": ["breakout", "rejection", "rejection", "tie"],
        }
    )

    result = review.one_minute_cohort_summary(actual).set_index("cohort")

    assert result.loc["normal", "reaction_rate_60m"] == 1.0
    assert result.loc["meme", "reaction_rate_60m"] == 0.5
    assert result.loc["normal", "breakouts"] == 1
    assert result.loc["meme", "ties"] == 1


def test_direction_lead_requires_sample_eligibility_and_joint_floor() -> None:
    direction = pd.DataFrame(
        {
            "scope": ["all", "all", "normal"],
            "method": ["small_high", "eligible_high", "ignored_scope"],
            "joint_success_rate": [0.70, 0.60, 1.0],
            "conditional_direction_accuracy": [0.80, 0.70, 1.0],
            "issued_call_coverage": [1.0, 0.9, 1.0],
            "lead_eligible_sample_size": [False, True, True],
        }
    )

    result = review.all_scope_direction_summary(direction).set_index("method")

    assert not bool(result.loc["small_high", "eligible_as_direction_lead"])
    assert bool(result.loc["eligible_high", "eligible_as_direction_lead"])
    assert "ignored_scope" not in result.index

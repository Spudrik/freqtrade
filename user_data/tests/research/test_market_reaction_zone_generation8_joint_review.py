# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation8_joint_review as review,
)


def test_final_status_requires_all_three_seeds_to_remain_positive() -> None:
    strict = {
        seed: {"strict": True, "point_positive": True} for seed in (42, 17, 73)
    }
    provisional = {
        42: {"strict": True, "point_positive": True},
        17: {"strict": False, "point_positive": True},
        73: {"strict": True, "point_positive": True},
    }
    failed = {
        42: {"strict": True, "point_positive": True},
        17: {"strict": False, "point_positive": False},
        73: {"strict": True, "point_positive": True},
    }

    assert review.final_status(strict) == review.FINAL_STRICT
    assert review.final_status(provisional) == review.FINAL_PROVISIONAL
    assert review.final_status(failed) == review.FINAL_FAILED


def test_relationship_summary_preserves_smaller_group_evidence() -> None:
    frame = pd.DataFrame(
        {
            "cohort": ["normal", "normal"],
            "branch_id": ["branch", "branch"],
            "surface": ["main", "main"],
            "route_id": ["relative_volume", "relative_volume"],
            "route_type": ["add_one_dimension", "add_one_dimension"],
            "target": ["&-g6_future_volume_ratio_h1"] * 2,
            "group_id": ["smart_contract_platforms", "full_normal_cohort"],
            "final_status": [review.FINAL_STRICT, review.FINAL_FAILED],
            "minimum_relative_error_reduction": [0.02, -0.01],
            "median_relative_error_reduction": [0.04, 0.01],
        }
    )

    result = review.relationship_summary(frame).iloc[0]

    assert result["relationship_status"] == review.FINAL_STRICT
    assert result["strict_groups"] == "smart_contract_platforms"
    assert result["failed_groups"] == "full_normal_cohort"


def test_branch_completion_always_contains_all_eight_frozen_siblings() -> None:
    relationships = pd.DataFrame(
        columns=["branch_id", "relationship_status"]
    )

    result = review.branch_completion(relationships)

    assert len(result) == 8
    assert result["status"].eq("completed_without_seed42_candidate").all()

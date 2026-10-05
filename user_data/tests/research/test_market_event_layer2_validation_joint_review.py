# ruff: noqa: S101

from __future__ import annotations

from user_data.Custom_Launcher.research.context_features import (
    market_event_layer2_validation_joint_review as module,
)


def test_joint_review_covers_five_distinct_frozen_questions() -> None:
    activity = {
        "decisions": [
            {
                "branch_id": "sec_activity_full_family_randomization",
                "familywise_p_value": 0.004,
            },
            {
                "branch_id": "boj_activity_full_family_randomization",
                "familywise_p_value": 0.75,
            },
        ]
    }
    cpi = {
        "randomization_decision": {"familywise_p_value": 0.73},
        "common_event_decision": {
            "uplift_over_follower_own_on_same_calls": 0.04
        },
    }
    sec = {"decision": {"exact_first_public_clock_coverage": 0.0}}

    decisions = module.build_joint_decisions(activity, cpi, sec)

    assert len(decisions) == 5
    assert len({row["branch_id"] for row in decisions}) == 5
    assert decisions[0]["result"] == "retained_association"
    assert decisions[-1]["result"] == "not_retained"

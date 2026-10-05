# ruff: noqa: S101

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation18_joint_review as g18j,
)


def test_next_queue_is_broad_and_keeps_direction_parked() -> None:
    queue = g18j.next_branch_queue(portable_recross_count=4, broad_context_count=2)

    assert len(queue["siblings"]) == 7
    assert sum(item["status"] == "queued" for item in queue["siblings"]) == 5
    assert sum(item["status"].startswith("parked") for item in queue["siblings"]) == 2
    assert queue["no_descendant_before_all_siblings_terminal_and_jointly_reviewed"]


def test_portable_questions_requires_normal_and_meme() -> None:
    frame = pd.DataFrame(
        [
            {
                "status": "strict_holdout_confirmation",
                "scope": "vp",
                "metric": "recross",
                "market_scope": "all_normal",
            },
            {
                "status": "strict_holdout_confirmation",
                "scope": "vp",
                "metric": "recross",
                "market_scope": "top_ten_memes",
            },
            {
                "status": "strict_holdout_confirmation",
                "scope": "donchian",
                "metric": "reaction",
                "market_scope": "all_normal",
            },
        ]
    )

    result = g18j.portable_questions(frame, ("scope", "metric"))

    assert result[["scope", "metric"]].to_dict("records") == [
        {"scope": "vp", "metric": "recross"}
    ]


def test_valid_precedence_requires_candidate_control_pass() -> None:
    head = pd.DataFrame(
        [
            {
                "status": "strict_holdout_confirmation",
                "scope_value": "vp",
                "candidate_timeframe": "4h",
                "metric": "reaction",
                "horizon_hours": 2,
                "market_scope": "all_normal",
            }
        ]
    )
    control = pd.DataFrame(
        [
            {
                "status": "strict_holdout_confirmation",
                "scope_kind": "family_timeframe",
                "scope_value": "vp",
                "source_timeframe": "8h",
                "metric": "reaction",
                "horizon_hours": 2,
                "market_scope": "all_normal",
            }
        ]
    )

    result = g18j.valid_timeframe_precedence(
        {"timeframe_head": head, "timeframe_control": control}
    )

    assert result.empty

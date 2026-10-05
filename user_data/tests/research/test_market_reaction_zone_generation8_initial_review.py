# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation8 as g8,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation8_initial_review as review,
)


def test_candidate_rows_keeps_strict_and_provisional_but_not_failed() -> None:
    decisions = pd.DataFrame(
        {
            "status": [
                "strict_seed42_candidate_for_seed_confirmation",
                "provisional_seed42_candidate_for_seed_confirmation",
                "not_retained_at_initial_seed",
            ],
            "all_controls_point_positive": [True, "True", False],
        }
    )

    selected = review.candidate_rows({"cohort": "normal", "decisions": decisions})

    assert selected["initial_strength"].tolist() == ["strict", "provisional"]
    assert selected["cohort"].eq("normal").all()


def test_confirmation_selection_rebuilds_every_control_for_both_seeds() -> None:
    question = next(
        item for item in g8.QUESTIONS if item.cell.cohort == "normal"
    )
    candidates = pd.DataFrame(
        [
            {
                "cohort": "normal",
                "question_id": question.question_id,
                "branch_id": question.cell.branch_id,
                "surface": question.cell.surface,
                "route_id": "identity",
                "route_type": "add_one_dimension",
                "target": question.targets[0],
                "group_id": "full_normal_cohort",
                "group_members": "BTC/USDT:USDT",
                "plain_name": question.plain_name,
                "initial_strength": "provisional",
                "expected_controls": 3,
            }
        ]
    )
    profiles, comparisons = g8.build_registry(g8.CONFIRMATION_SEEDS)

    selected = review.confirmation_selection(
        candidates,
        cohort="normal",
        profiles=profiles,
        comparisons=comparisons,
    )

    assert selected["candidate_row_count"] == 1
    assert selected["unique_question_routes"] == 1
    assert selected["profile_count"] == 8
    assert selected["comparison_count"] == 6
    assert all("seed42" not in item for item in selected["profile_ids"])


def test_conceptual_overlap_marks_same_route_and_target_in_both_cohorts() -> None:
    candidates = pd.DataFrame(
        {
            "branch_id": ["branch", "branch", "other"],
            "surface": ["main", "main", "main"],
            "route_id": ["width", "width", "volume"],
            "route_type": ["add_one_dimension"] * 3,
            "target": ["range", "range", "volume"],
            "cohort": ["normal", "meme", "normal"],
            "initial_strength": ["strict", "provisional", "provisional"],
            "group_id": ["normal", "memes", "normal"],
        }
    )

    overlap = review.conceptual_overlap(candidates)

    shared = overlap.loc[overlap["route_id"].eq("width")].iloc[0]
    assert bool(shared["cross_cohort_seed42_overlap"])
    assert shared["cohorts"] == "meme,normal"

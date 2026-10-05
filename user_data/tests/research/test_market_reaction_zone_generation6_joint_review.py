# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation6_joint_review as review,
)


def test_required_comparison_classification_is_strict_about_all_four() -> None:
    strong = {suffix: review.STRONG_STATUS for suffix in review.INTERACTION_SUFFIXES}
    near = dict(strong)
    near[review.INTERACTION_SUFFIXES[-1]] = review.PROVISIONAL_STATUS
    failed = dict(strong)
    failed[review.INTERACTION_SUFFIXES[-1]] = review.FAILED_STATUS

    assert review.classify_required_comparisons(strong).startswith("control_resistant")
    assert review.classify_required_comparisons(near).startswith("near_control")
    assert review.classify_required_comparisons(failed).startswith("partial_or_redundant")


def test_comparison_statuses_requires_each_component_and_stale_control() -> None:
    source = "example_source"
    target = "example_target"
    frame = pd.DataFrame(
        {
            "comparison_id": [f"{source}__{suffix}" for suffix in review.INTERACTION_SUFFIXES],
            "target": [target] * len(review.INTERACTION_SUFFIXES),
            "status": [review.STRONG_STATUS] * len(review.INTERACTION_SUFFIXES),
        }
    )

    statuses = review.comparison_statuses(frame, source=source, target=target)

    assert set(statuses) == set(review.INTERACTION_SUFFIXES)
    assert set(statuses.values()) == {review.STRONG_STATUS}


def test_cross_cohort_review_does_not_call_one_cohort_broad() -> None:
    frame = pd.DataFrame(
        {
            "row_type": ["source_plus_level", "source_plus_level"],
            "source": ["source", "source"],
            "target": [review.g6f.TARGET_COLUMNS[0]] * 2,
            "cohort": ["normal", "meme"],
            "required_comparison_class": [
                "control_resistant_against_both_components_and_both_stale_controls",
                "partial_or_redundant_did_not_beat_every_component",
            ],
        }
    )

    result = review.cross_cohort_rows(frame)

    assert result.loc[0, "cross_cohort_classification"] == (
        "normal_cohort_only_at_current_standard"
    )


def test_frozen_generation7_is_broad_bounded_and_directionless(tmp_path) -> None:
    frozen = review.frozen_generation7(tmp_path / "review.json")
    branches = frozen["branches"]

    assert len(branches) == 10
    assert sum(bool(item["weak_component_exception"]) for item in branches) == 2
    assert frozen["research_boundary"]["profit_optimization"] is False
    assert frozen["research_boundary"]["direction_prediction"] is False
    assert all(item["profile_ladder"] == review.required_profiles() for item in branches)
    assert all(item["targets"] for item in branches)

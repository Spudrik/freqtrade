# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation7_joint_review as review,
)


def test_cross_cohort_classification_keeps_strict_and_provisional_distinct() -> None:
    assert (
        review.cross_cohort_classification(review.STRICT, review.PROVISIONAL)
        == "strict_in_one_and_provisional_in_the_other"
    )
    assert (
        review.cross_cohort_classification(review.PROVISIONAL, review.PROVISIONAL)
        == "provisional_in_both_cohorts"
    )
    assert (
        review.cross_cohort_classification(review.FAILED, review.FAILED)
        == "not_reproduced_in_either_cohort"
    )


def test_best_group_summary_counts_persisted_booleans_safely() -> None:
    scores = pd.DataFrame(
        {
            "branch_id": ["branch"] * 4,
            "target": ["target"] * 4,
            "group_id": ["a", "a", "b", "b"],
            "strict_period_pass": ["True", "False", "True", "True"],
            "provisional_period_pass": ["True", "True", "True", "True"],
            "comparison_id": ["x", "y", "x", "y"],
            "equal_coin_paired_mae_gain": [0.1, 0.1, 0.2, 0.2],
            "bootstrap_lower": [-0.1, -0.1, 0.1, 0.1],
        }
    )

    result = review.best_group_summary(
        {"group_scores": scores}, "branch", "target"
    )

    assert result["group_id"] == "b"
    assert result["strict_checks"] == 2
    assert result["provisional_checks"] == 2


def test_frozen_generation8_is_broad_but_excludes_failed_descendants(tmp_path) -> None:
    batch = review.frozen_generation8(tmp_path / "joint_review.json")
    ids = {branch["id"] for branch in batch["branches"]}

    assert batch["status"] == "frozen_before_generation8_attribution_outcomes"
    assert len(ids) == 8
    assert any("level_geometry" in branch_id for branch_id in ids)
    assert any("participation" in branch_id for branch_id in ids)
    assert any("btc_context" in branch_id for branch_id in ids)
    assert any("cross_timeframe" in branch_id for branch_id in ids)
    assert any("orderbook" in branch_id for branch_id in ids)
    assert any("contact" in branch_id for branch_id in ids)
    assert batch["portfolio_summary"]["cluster_descendant_included"] is False
    assert batch["portfolio_summary"]["news_descendant_included"] is False
    assert batch["research_boundary"]["profit_optimization"] is False
    assert batch["research_boundary"]["direction_prediction"] is False


def test_every_generation8_sibling_has_common_controls_and_plain_name(tmp_path) -> None:
    batch = review.frozen_generation8(tmp_path / "joint_review.json")

    for branch in batch["branches"]:
        assert branch["plain_name"].endswith("?")
        assert branch["targets"]
        assert branch["attribution_dimensions"]
        assert branch["required_controls"] == review.COMMON_CONTROLS
        assert "three predeclared deterministic model seeds" in " ".join(
            branch["required_controls"]
        )

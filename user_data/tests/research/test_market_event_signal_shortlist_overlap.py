# ruff: noqa: S101

from __future__ import annotations

import json

import pandas as pd
from pandas import DataFrame

from user_data.Custom_Launcher.research.context_features import (
    market_event_signal_shortlist_overlap as shortlist,
)


def _membership() -> DataFrame:
    rows = []
    calls = {
        "route_a": [True, True, False, False],
        "route_b": [True, False, True, False],
    }
    for route_id, values in calls.items():
        for index, issued in enumerate(values):
            rows.append(
                {
                    "date": pd.Timestamp("2024-01-01", tz="UTC")
                    + pd.Timedelta(hours=index),
                    "pair": "BTC/USDT:USDT",
                    "period": "walk_forward_validation_2024",
                    "sample_id": f"sample_{index}",
                    "parent_episode_ids_json": json.dumps([f"episode_{index}"]),
                    "actual_reaction": index < 2,
                    "route_id": route_id,
                    "signal_issued": issued,
                    "activity_score": float(index),
                    "activity_threshold": 0.5,
                }
            )
    return DataFrame.from_records(rows)


def test_shortlist_has_one_representative_per_family() -> None:
    families = [str(item["family_id"]) for item in shortlist.REPRESENTATIVES]

    assert len(families) == 5
    assert len(set(families)) == 5


def test_overlap_metrics_keep_intersection_and_unique_calls_separate() -> None:
    result = shortlist.overlap_metrics(_membership(), "route_a", "route_b")

    assert result["both_calls"] == 1
    assert result["only_a_calls"] == 1
    assert result["only_b_calls"] == 1
    assert result["jaccard"] == 1 / 3
    assert result["both_reaction_rate"] == 1.0


def _decision_frame(*, jaccard: float, uplift: float) -> DataFrame:
    rows = []
    for period in shortlist.VALIDATION_PERIODS:
        rows.append(
            {
                "market_scope": "all_five_equal_weight",
                "route_a": "route_a",
                "route_b": "route_b",
                "period": period,
                "jaccard": jaccard,
                "both_unique_episodes": 30,
                "only_a_unique_episodes": 15,
                "only_b_unique_episodes": 15,
                "route_a_reaction_rate": 0.60,
                "route_b_reaction_rate": 0.61,
                "route_a_matched_strength_reaction_rate": 0.60,
                "route_b_matched_strength_reaction_rate": 0.61,
                "both_reaction_rate": 0.61 + uplift,
            }
        )
    return DataFrame.from_records(rows)


def test_overlap_decision_prefers_complementary_before_distinct() -> None:
    result = shortlist.decide_overlaps(
        _decision_frame(jaccard=0.50, uplift=0.03)
    ).iloc[0]

    assert result["decision"] == "complementary_overlap"
    assert result["complementary_overlap"]


def test_overlap_decision_marks_near_duplicate() -> None:
    result = shortlist.decide_overlaps(
        _decision_frame(jaccard=0.85, uplift=0.00)
    ).iloc[0]

    assert result["decision"] == "near_duplicate"
    assert result["near_duplicate"]

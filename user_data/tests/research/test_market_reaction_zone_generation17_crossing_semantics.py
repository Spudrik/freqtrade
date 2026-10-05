# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_crossing_semantics as g17s,
)


def semantic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "actual__zone_half_width_atr": [0.2, 0.8],
            "actual__abs_excursion_atr_h1": [0.7, 0.9],
            "actual__range_ratio_h1": [0.1, 1.0],
            "actual__volume_ratio_h1": [2.0, 1.0],
            "actual__pressure_change_h1": [0.25, 0.75],
        }
    )


def test_semantic_values_separate_rejection_and_two_sided_traffic() -> None:
    frame = semantic_frame()
    assert g17s.semantic_values(frame, "actual", "repeated_recross", 1).tolist() == [
        1.0,
        0.0,
    ]
    assert g17s.semantic_values(frame, "actual", "one_sided_rejection", 1).tolist() == [
        1.0,
        0.0,
    ]
    assert g17s.semantic_values(frame, "actual", "two_sided_traversal", 1).tolist() == [
        0.0,
        1.0,
    ]


def test_joint_decision_requires_both_controls_and_both_normal_windows() -> None:
    rows = []
    for window in ("standard_validation", "recent_normal_chronology"):
        for comparison in g17s.CONTROLS.values():
            rows.append(
                {
                    "metric": "repeated_recross",
                    "horizon_hours": 2,
                    "market_scope": "all_normal",
                    "window": window,
                    "comparison": comparison,
                    "effect_sign": "positive",
                    "point_pass": True,
                    "strict_pass": True,
                    "minimum_effect": 0.1,
                    "maximum_effect": 0.2,
                }
            )
    decisions = g17s.joint_decisions(pd.DataFrame(rows))
    assert decisions["complete_control_chronology"].iloc[0]
    assert decisions["strict_repeated"].iloc[0]

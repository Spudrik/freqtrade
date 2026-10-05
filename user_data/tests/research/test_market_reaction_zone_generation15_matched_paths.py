# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation15_matched_paths as g15,
)


def test_censored_timing_is_oriented_so_positive_means_faster_actual() -> None:
    frame = pd.DataFrame(
        {
            "actual__time_to_abs_0_5atr": [1.0, float("nan"), 5.0],
            "control__time_to_abs_0_5atr": [3.0, float("nan"), 2.0],
        }
    )

    values = g15.metric_values(frame, "censored_time_to_0_5atr", 4)

    assert values["actual_value"].tolist() == [1.0, 5.0, 5.0]
    assert values["control_value"].tolist() == [3.0, 5.0, 2.0]
    assert values["difference"].tolist() == [2.0, 0.0, -3.0]


def test_hit_metric_treats_missing_threshold_time_as_no_hit() -> None:
    frame = pd.DataFrame(
        {
            "actual__time_to_abs_0_5atr": [1.0, float("nan"), 5.0],
            "control__time_to_abs_0_5atr": [3.0, 2.0, float("nan")],
        }
    )

    values = g15.metric_values(frame, "hit_0_5atr", 4)

    assert values["actual_value"].tolist() == [1.0, 0.0, 0.0]
    assert values["control_value"].tolist() == [1.0, 1.0, 0.0]
    assert values["difference"].tolist() == [0.0, -1.0, 0.0]


def test_cell_decision_requires_complete_consistent_control_ladder() -> None:
    scores = pd.DataFrame(
        [
            {
                "route_id": "reaction_path_decomposition",
                "scope_value": "all",
                "metric": "volume_ratio",
                "horizon_hours": 1,
                "cohort": "normal",
                "window": "standard_validation",
                "analysis_period": period,
                "control": control,
                "eligible_coins": 10,
                "equal_coin_paired_difference": 0.10,
                "bootstrap_lower_95": 0.02,
                "bootstrap_upper_95": 0.18,
            }
            for period in ("early", "late")
            for control in g15.CONTROLS
        ]
    )

    decision = g15.cell_decisions(scores).iloc[0]

    assert bool(decision["point_pass"])
    assert bool(decision["strict_pass"])
    assert decision["consistent_effect_sign"] == "positive"

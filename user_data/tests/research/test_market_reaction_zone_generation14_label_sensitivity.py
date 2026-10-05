# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation14_label_sensitivity as g14l,
)


def test_reaction_label_respects_zone_price_and_volume_thresholds() -> None:
    frame = pd.DataFrame(
        {
            "actual__zone_half_width_atr": [0.2, 0.8, 0.2, 0.2],
            "actual__abs_excursion_atr_h1": [0.6, 0.7, 0.6, 0.4],
            "actual__volume_ratio_h1": [1.3, 1.3, 1.2, 1.3],
        }
    )

    output = g14l.reaction_label(
        frame,
        prefix="actual",
        horizon=1,
        price_threshold_atr=0.5,
        volume_threshold=1.25,
    )

    assert output.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_robust_decision_requires_original_and_seven_definitions() -> None:
    rows = []
    for price in (0.35, 0.5, 0.75):
        for volume in (1.1, 1.25, 1.5):
            label = g14l.definition_id(price, volume)
            point = label not in {
                g14l.definition_id(0.75, 1.5),
                g14l.definition_id(0.75, 1.25),
            }
            rows.append(
                {
                    "definition_id": label,
                    "cohort": "normal",
                    "window": "standard_validation",
                    "horizon_hours": 1,
                    "complete_control_period_ladder": True,
                    "point_pass": point,
                    "strict_pass": point,
                    "minimum_equal_coin_difference": 0.01,
                }
            )

    result = g14l.robust_decisions(pd.DataFrame(rows)).iloc[0]

    assert bool(result["robust_point"])
    assert bool(result["robust_strict"])
    assert result["point_positive_definitions"] == 7

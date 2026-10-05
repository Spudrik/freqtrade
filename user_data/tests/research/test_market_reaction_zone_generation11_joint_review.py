# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation11_joint_review as review,
)


def test_candidate_family_keeps_generic_range_distinct() -> None:
    assert (
        review.candidate_family(
            "timeframe_horizon_persistence",
            "source_timeframe=1h|level_family=generic_prior_range",
        )
        == "generic_prior_range_across_timeframes"
    )


def test_candidate_portability_uses_pairwise_control_differences() -> None:
    candidate = pd.DataFrame(
        {
            "route": ["timeframe_horizon_persistence"],
            "cohort": ["normal"],
            "cell": ["source_timeframe=1h|level_family=generic_prior_range"],
            "metric": ["reaction_rate"],
            "horizon_band": ["short"],
            "behaviour_sign": [1],
            "candidate": [True],
        }
    )
    rows = []
    for pair in ("A", "B"):
        for period in ("validation_early", "validation_late"):
            for horizon in (1, 2):
                for control in ("actual", *review.g11d.CONTROLS):
                    rows.append(
                        {
                            "route": "timeframe_horizon_persistence",
                            "cohort": "normal",
                            "pair": pair,
                            "period": period,
                            "control": control,
                            "cell": "source_timeframe=1h|level_family=generic_prior_range",
                            "horizon_hours": horizon,
                            "metric": "reaction_rate",
                            "minimum_coins": 2,
                            "pair_median": 0.7 if control == "actual" else 0.5,
                            "events": 10,
                        }
                    )

    result = review.candidate_portability(candidate, pd.DataFrame.from_records(rows))

    assert result.iloc[0]["coins_with_average_effect_in_declared_direction"] == 2
    assert result.iloc[0]["portability_label"] == "all_coins_average_same_direction"
    assert abs(result.iloc[0]["mean_equal_coin_effect"] - 0.2) < 1e-12


def test_freqai_queue_is_broad_and_direction_neutral() -> None:
    queue = review.next_freqai_queue()

    assert len(queue) == 7
    assert queue["question_id"].nunique() == 7
    assert queue["cohorts"].eq("normal,frozen_top_ten_memes").all()
    assert queue["horizons_hours"].eq("1,2,4,8").all()

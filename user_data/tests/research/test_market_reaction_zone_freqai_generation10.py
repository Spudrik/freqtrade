# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation10 as g10,
)


def test_mechanism_requires_every_declared_group() -> None:
    frame = pd.DataFrame(
        {
            "question_id": ["q"] * 3,
            "route_id": ["r"] * 3,
            "plain_name": ["plain"] * 3,
            "group_id": list(g10.g10z.DECLARED_GROUPS),
            "status": ["strict_three_seed_lead"] * 3,
            "minimum_equal_coin_paired_mae_gain": [0.01, 0.02, 0.03],
            "minimum_bootstrap_lower": [0.001, 0.002, 0.003],
        }
    )

    result = g10.mechanism_summary(frame).iloc[0]

    assert result["status"] == "confirmed_strict_all_declared_groups"
    assert result["minimum_equal_coin_paired_mae_gain"] == 0.01


def test_one_failed_group_fails_complete_mechanism() -> None:
    frame = pd.DataFrame(
        {
            "question_id": ["q"] * 3,
            "route_id": ["r"] * 3,
            "plain_name": ["plain"] * 3,
            "group_id": list(g10.g10z.DECLARED_GROUPS),
            "status": [
                "strict_three_seed_lead",
                "provisional_three_seed_lead",
                "failed_three_seed_replication",
            ],
            "minimum_equal_coin_paired_mae_gain": [0.01, 0.002, -0.001],
            "minimum_bootstrap_lower": [0.001, -0.001, -0.003],
        }
    )

    result = g10.mechanism_summary(frame).iloc[0]

    assert result["status"] == "failed_untouched_confirmation"
    assert result["failed_groups"] == g10.g10z.DECLARED_GROUPS[-1]


def test_missing_declared_group_is_integrity_error() -> None:
    frame = pd.DataFrame(
        {
            "question_id": ["q"] * 2,
            "route_id": ["r"] * 2,
            "plain_name": ["plain"] * 2,
            "group_id": list(g10.g10z.DECLARED_GROUPS[:2]),
            "status": ["strict_three_seed_lead"] * 2,
            "minimum_equal_coin_paired_mae_gain": [0.01, 0.02],
            "minimum_bootstrap_lower": [0.001, 0.002],
        }
    )

    try:
        g10.mechanism_summary(frame)
    except ValueError as exc:
        assert "group coverage drifted" in str(exc)
    else:
        raise AssertionError("Missing group should fail closed")

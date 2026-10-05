# ruff: noqa: S101

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
from pandas import DataFrame


MODULE = (
    "user_data.Custom_Launcher.research.context_features."
    "market_historical_news_interaction_direct"
)
direct = importlib.import_module(MODULE)


def outcome_frame() -> DataFrame:
    dates = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    return DataFrame(
        {
            "date": dates,
            "future_return_24h": [0.03, -0.03, 0.0],
            "future_max_upside_24h": [0.06, 0.01, 0.02],
            "future_max_drawdown_24h": [-0.01, -0.07, -0.02],
            "up_2pct_next_24h": [True, False, True],
            "down_2pct_next_24h": [False, True, True],
            "large_upside_next_24h": [True, False, False],
            "large_drawdown_next_24h": [False, True, False],
        }
    ).set_index("date")


def test_outcome_label_requires_reaction_and_expected_side_dominance() -> None:
    dates = outcome_frame().index
    rows = DataFrame(
        {
            "theory_id": ["up", "down", "tie"],
            "window_id": ["window"] * 3,
            "episode_id": ["a", "b", "c"],
            "expected_direction": [1, -1, 1],
            "feature_candle_open_utc": dates,
        }
    )
    labelled = direct.label_outcomes(
        rows,
        {"window": outcome_frame()},
        {"window": 0.03},
        date_column="feature_candle_open_utc",
        sample_kind="full_interaction",
    )
    assert labelled.loc[0, "joint_success"]
    assert labelled.loc[1, "joint_success"]
    assert labelled.loc[2, "direction_tie_abstention"]
    assert pd.isna(labelled.loc[2, "joint_success"])


def test_pair_summary_compares_identical_matched_rows() -> None:
    rows = DataFrame(
        {
            "theory_id": ["idea"] * 2,
            "window_id": ["window"] * 2,
            "control_kind": ["market_only"] * 2,
            "match_distance": [0.1, 0.2],
            "event_joint_success": [True, True],
            "control_joint_success": [False, True],
            "joint_success_difference": [1.0, 0.0],
            "event_reaction_success": [True, True],
            "control_reaction_success": [True, False],
            "reaction_success_difference": [0.0, 1.0],
            "event_direction_success": [True, True],
            "control_direction_success": [False, True],
            "direction_success_difference": [1.0, 0.0],
        }
    )
    summary = direct.summarize_pairs(rows).iloc[0]
    assert summary["complete_pair_count"] == 2
    assert summary["event_joint_rate"] == 1.0
    assert summary["control_joint_rate"] == 0.5
    assert summary["joint_rate_lift"] == 0.5


def test_familywise_null_links_same_event_across_theories() -> None:
    rows = DataFrame.from_records(
        [
            {
                "theory_id": theory,
                "window_id": "window",
                "control_kind": "market_only",
                "event_feature_candle_open_utc": pd.Timestamp(
                    "2024-01-01T00:00:00Z"
                )
                + pd.Timedelta(hours=episode),
                "episode_id": f"event_{episode}",
                "event_reaction_success": episode < 4,
                "control_reaction_success": False,
                "event_direction_success": episode < 4,
                "control_direction_success": False,
                "event_joint_success": episode < 4,
                "control_joint_success": False,
            }
            for theory in ("one", "two")
            for episode in range(6)
        ]
    )
    null = direct.familywise_null(rows)
    assert len(null) == direct.frozen.PERMUTATION_ITERATIONS * 3
    assert set(null["outcome_measure"]) == {
        "reaction_success",
        "direction_success",
        "joint_success",
    }
    assert (
        null.groupby("outcome_measure")["iteration"].nunique()
        == direct.frozen.PERMUTATION_ITERATIONS
    ).all()


def test_studentized_statistic_accounts_for_sample_size_and_variance() -> None:
    assert direct.paired_studentized_statistic(np.array([1, 1, 0, 0])) > 0
    assert direct.paired_studentized_statistic(np.array([-1, -1, 0, 0])) < 0


def test_rate_ignores_abstentions() -> None:
    assert direct._rate(pd.Series([True, False, np.nan])) == 0.5

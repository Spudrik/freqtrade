# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame

from user_data.Custom_Launcher.research.context_features import (
    market_event_meme_transmission_2026 as batch,
)


def test_frozen_questions_keep_prediction_separate_from_same_hour_description() -> None:
    assert batch.HORIZONS == (1, 3)
    assert batch.BTC_PRE_ACTIVITY_THRESHOLD == 1.0
    assert batch.COHORT_MAJORITY == 0.60
    assert batch.MINIMUM_MAIN_EPISODES > batch.MINIMUM_JOINT_EPISODES


def test_rolling_sensitivity_excludes_current_hour() -> None:
    rows = batch.SENSITIVITY_MINIMUM_HOURS + 5
    x = np.linspace(-0.01, 0.01, rows)
    frame = DataFrame(
        {
            "btc_close_log_return": x,
            "coin_close_log_return": 2 * x,
        }
    )
    before = batch._rolling_sensitivity(frame).iloc[-1]
    frame.loc[rows - 1, "coin_close_log_return"] = 100.0
    after = batch._rolling_sensitivity(frame).iloc[-1]

    assert np.isclose(before["ordinary_beta_to_btc"], 2.0)
    assert np.isclose(after["ordinary_beta_to_btc"], 2.0)


def test_cohort_majority_requires_complete_frozen_cohort() -> None:
    frame = DataFrame(
        {
            "sample_id": ["s"] * 2,
            "sample_kind": ["actual_event"] * 2,
            "episode_id": ["e"] * 2,
            "model_anchor_utc": pd.to_datetime(["2026-01-01T00:00:00Z"] * 2),
            "model_period": [batch.PERIODS[0]] * 2,
            "event_families_json": ["[]"] * 2,
            "event_kinds_json": ["[]"] * 2,
            "recent__relative_volume": [1.2] * 2,
            "btc_pre_activity_signal": [True] * 2,
            "btc_confirmed_reaction": [True] * 2,
            "btc_event_log_return": [0.01] * 2,
            "btc_initial_sign": [1.0] * 2,
            "horizon_hours": [1] * 2,
            "pair": ["A", "B"],
            "event_volume_ratio": [2.0, 2.0],
            "event_range_ratio": [2.0, 2.0],
            "lag_log_return": [0.01, 0.02],
            "residual_toward_initial_btc_z": [1.0, 1.0],
        }
    )
    result = batch.cohort_rows(frame, "test", ["A", "B", "C"]).iloc[0]

    assert not result["complete_cohort"]
    assert pd.isna(result["volume_majority_reacted"])


def test_csv_boolean_values_do_not_turn_false_strings_true() -> None:
    result = batch._coerce_bool(pd.Series(["True", "False"]))

    assert result.tolist() == [True, False]


def _score_fixture(rate_first: float = 0.60, rate_second: float = 0.60) -> DataFrame:
    rows = []
    for period, rate in zip(batch.PERIODS, (rate_first, rate_second), strict=True):
        rows.append(
            {
                "route": "confirmed_btc_to_later_group_direction",
                "outcome": "later_direction_matches_initial_btc",
                "cohort": "memes",
                "horizon_hours": 1,
                "condition": "btc_reaction_observed",
                "sample_kind": "actual_event",
                "period": period,
                "support": 10,
                "successes": int(rate * 10),
                "success_rate": rate,
            }
        )
    rows.extend(
        [
            {
                "route": "confirmed_btc_to_later_group_direction",
                "outcome": "later_direction_matches_initial_btc",
                "cohort": "memes",
                "horizon_hours": 1,
                "condition": "btc_reaction_observed",
                "sample_kind": "actual_event",
                "period": "all_2026",
                "support": 20,
                "successes": 12,
                "success_rate": 0.60,
            },
            {
                "route": "confirmed_btc_to_later_group_direction",
                "outcome": "later_direction_matches_initial_btc",
                "cohort": "memes",
                "horizon_hours": 1,
                "condition": "btc_reaction_observed",
                "sample_kind": "matched_control",
                "period": "all_2026",
                "support": 30,
                "successes": 15,
                "success_rate": 0.50,
            },
        ]
    )
    return DataFrame.from_records(rows)


def test_direction_decision_requires_two_temporally_consistent_halves() -> None:
    retained = batch.decide(_score_fixture()).iloc[0]
    inconsistent = batch.decide(_score_fixture(rate_second=0.40)).iloc[0]

    assert retained["meets_55_floor"]
    assert retained["event_specific"]
    assert not inconsistent["meets_55_floor"]

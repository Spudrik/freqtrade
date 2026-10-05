# ruff: noqa: S101

from __future__ import annotations

import pandas as pd
from pandas import DataFrame

from user_data.Custom_Launcher.research.context_features import (
    market_event_signal_fresh_2026 as run,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_signal_fresh_2026_freeze as frozen,
)


def test_fresh_surface_has_one_profile_per_family_and_one_target() -> None:
    registry = frozen.build_registry()

    assert registry["profile_count"] == 5
    assert len({item["family_id"] for item in frozen.REPRESENTATIVES}) == 5
    assert set(registry["profiles"]) == set(frozen.PROFILE_DEFINITIONS)
    assert registry["targets_declared_without_values"] == [frozen.PRIMARY_TARGET]


def test_model_period_keeps_two_whole_2026_halves() -> None:
    assert (
        frozen.model_period("2026-04-30T23:00:00Z")
        == "untouched_confirmation_2026_jan_apr"
    )
    assert (
        frozen.model_period("2026-05-01T00:00:00Z")
        == "untouched_confirmation_2026_may_aug"
    )


def _scope_rows(*, enrichment: float = 0.10, episodes: int = 25) -> DataFrame:
    rows = []
    for period in frozen.VALIDATION_PERIODS:
        rows.append(
            {
                "family_id": "local_participation_and_pressure",
                "route_id": "recent_local_participation",
                "profile_id": "recent_market_only",
                "market_scope": "all_five_equal_weight",
                "period": period,
                "sample_kind": "actual_event",
                "eligible_coins": 5,
                "call_independent_samples": episodes,
                "reaction_rate_calls": 0.60,
                "reaction_enrichment_vs_non_calls": enrichment,
            }
        )
        rows.append(
            {
                "family_id": "local_participation_and_pressure",
                "route_id": "recent_local_participation",
                "profile_id": "recent_market_only",
                "market_scope": "all_five_equal_weight",
                "period": period,
                "sample_kind": "matched_control",
                "eligible_coins": 5,
                "call_independent_samples": 40,
                "reaction_rate_calls": 0.50,
                "reaction_enrichment_vs_non_calls": 0.0,
            }
        )
    return DataFrame.from_records(rows)


def test_fresh_decision_requires_support_rate_and_enrichment_in_both_halves() -> None:
    passed = run.decide(_scope_rows(), frozen.VALIDATION_PERIODS).iloc[0]
    weak = run.decide(
        _scope_rows(enrichment=0.0), frozen.VALIDATION_PERIODS
    ).iloc[0]
    sparse = run.decide(
        _scope_rows(episodes=10), frozen.VALIDATION_PERIODS
    ).iloc[0]

    assert passed["fresh_activity_pass"]
    assert not weak["fresh_activity_pass"]
    assert not sparse["support_pass"]


def test_prediction_timerange_provides_calibration_warmup() -> None:
    start = pd.Timestamp(run.FULL_SETTINGS["timerange"].split("-")[0], tz="UTC")

    assert start <= frozen.CURRENT_START_UTC - pd.Timedelta(days=90)
    assert run.FULL_SETTINGS["backtest_days"] == 120


def test_score_pairs_coerces_merged_object_signal_to_boolean() -> None:
    membership = DataFrame(
        {
            "family_id": ["major_event_information"] * 2,
            "route_id": ["event_with_recent_confirmation"] * 2,
            "profile_id": ["event_plus_recent_market"] * 2,
            "period": [frozen.VALIDATION_PERIODS[0]] * 2,
            "pair": ["BTC/USDT:USDT"] * 2,
            "sample_kind": ["actual_event"] * 2,
            "sample_id": ["a", "b"],
            "parent_episode_ids_json": ['["episode_a"]', '["episode_b"]'],
            "activity_prediction": [1.0, 0.0],
            "activity_threshold": [0.5, 0.5],
            "activity_reference": [0.0, 0.0],
            "activity_signal": pd.Series([True, False], dtype=object),
            "actual_reaction": [True, False],
        }
    )

    result = run.score_pairs(membership).iloc[0]

    assert result["calls"] == 1
    assert result["reaction_rate_calls"] == 1.0

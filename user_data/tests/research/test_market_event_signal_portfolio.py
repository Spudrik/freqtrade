# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from user_data.Custom_Launcher.research.context_features import (
    market_event_signal_portfolio as portfolio,
)


def test_routes_cover_exactly_five_families() -> None:
    represented = {str(route["family_id"]) for route in portfolio.ROUTES}

    assert represented == set(portfolio.FAMILY_IDS)
    assert len(portfolio.ROUTES) == 11


def test_causal_quantile_excludes_current_and_future_rows() -> None:
    values = Series([1.0, 2.0, 100.0, -500.0])
    result = portfolio.causal_rolling_quantile(
        values, quantile=0.5, lookback=3, minimum=2
    )

    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == 1.5
    assert result.iloc[3] == 2.0


def test_calibrated_prediction_abstains_until_prior_history_exists() -> None:
    frame = DataFrame(
        {
            "pair": ["BTC/USDT:USDT"] * 5,
            "date": pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
            "target": [0.0, 1.0, 2.0, 3.0, 4.0],
            "target_mean": [0.0] * 5,
            "target_std": [1.0] * 5,
        }
    )
    result = portfolio.calibrated_prediction(
        frame.rename(
            columns={
                "target": "&-meb_log_volume_ratio_h1",
                "target_mean": "&-meb_log_volume_ratio_h1_mean",
                "target_std": "&-meb_log_volume_ratio_h1_std",
            }
        ),
        "&-meb_log_volume_ratio_h1",
        direction_confidence=False,
    )

    assert not result["signal_issued"].any()


def _decision_rows(*, second_period: bool = True, enrichment: float = 0.10) -> DataFrame:
    periods = list(portfolio.VALIDATION_PERIODS)
    if not second_period:
        periods = periods[:1]
    rows = []
    for period in periods:
        rows.append(
            {
                "family_id": "major_event_information",
                "route_id": "event_clock",
                "route_kind": "component",
                "market_scope": "asset:BTC/USDT:USDT",
                "period": period,
                "horizon_hours": 1,
                "activity_metric": "log_volume_ratio",
                "activity_target": "&-meb_log_volume_ratio_h1",
                "direction_metric": "close_return_atr",
                "direction_target": "&-meb_close_return_atr_h1",
                "activity_unique_episodes": 30,
                "reaction_rate_calls": 0.70,
                "reaction_enrichment_vs_non_calls": enrichment,
                "direction_unique_episodes": 30,
                "direction_correct_rate": 0.60,
                "direction_majority_control_rate": 0.50,
                "direction_trend_control_rate": 0.52,
                "joint_unique_episodes": 30,
                "joint_success_rate": 0.56,
                "joint_majority_control_rate": 0.45,
                "joint_trend_control_rate": 0.48,
                "original_raw_joint_unique_episodes": 4,
            }
        )
    return DataFrame.from_records(rows)


def test_route_decision_requires_both_periods_and_enrichment() -> None:
    passed = portfolio.route_decisions(_decision_rows()).iloc[0]
    incomplete = portfolio.route_decisions(
        _decision_rows(second_period=False)
    ).iloc[0]
    no_enrichment = portfolio.route_decisions(
        _decision_rows(enrichment=0.0)
    ).iloc[0]

    assert passed["activity_pass"]
    assert passed["direction_pass"]
    assert passed["joint_pass"]
    assert not incomplete["activity_pass"]
    assert not no_enrichment["activity_pass"]


def test_family_summary_preserves_families_without_a_lead() -> None:
    decisions = portfolio.route_decisions(_decision_rows())
    summary = portfolio.family_summary(decisions)

    assert set(summary["family_id"]) == set(portfolio.FAMILY_IDS)
    empty_family = summary.loc[
        summary["family_id"].eq("calculated_reaction_areas")
    ].iloc[0]
    assert empty_family["status"] == "no_calibrated_prototype_lead"
    event_family = summary.loc[
        summary["family_id"].eq("major_event_information")
    ].iloc[0]
    assert event_family["unique_activity_lead_cells"] == 1
    assert event_family["unique_direction_lead_cells"] == 1

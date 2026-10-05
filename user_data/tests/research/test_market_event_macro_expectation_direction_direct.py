# ruff: noqa: S101

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
from pandas import DataFrame


MODULE = (
    "user_data.Custom_Launcher.research.context_features."
    "market_event_macro_expectation_direction_direct"
)
direct = importlib.import_module(MODULE)


def orientation_rows(base_hits: list[int]) -> DataFrame:
    records = []
    for index, actual in enumerate(base_hits):
        records.append(
            {
                "route_id": "jobs_available_component_balance",
                "orientation_policy": "learn_rate_or_growth_on_development_only",
                "source_block": direct.frozen.PARTITIONS[0],
                "expectation_episode_id": f"event_{index}",
                "predicted_base_direction": 1,
                "actual_direction": actual,
            }
        )
    return DataFrame.from_records(records)


def test_jobs_orientation_can_select_growth_channel_from_development_only() -> None:
    choices = direct.orientation_choices(orientation_rows([-1, -1, -1, 1]))
    choice = choices["jobs_available_component_balance"]
    assert choice["factor"] == -1
    assert choice["label"] == "growth_risk_appetite"


def test_fixed_cpi_orientation_is_never_learned_from_outcomes() -> None:
    rows = DataFrame(
        {
            "route_id": ["cpi_core_surprise"],
            "orientation_policy": ["fixed_inverse_rate_pressure"],
            "source_block": [direct.frozen.PARTITIONS[0]],
            "expectation_episode_id": ["event"],
            "predicted_base_direction": [-1],
            "actual_direction": [1],
        }
    )
    assert direct.orientation_choices(rows)["cpi_core_surprise"]["factor"] == 1


def permutation_rows() -> DataFrame:
    records = []
    for episode, actual in [("one", 1), ("two", -1)]:
        for route in ["route_a", "route_b"]:
            for pair in direct.frozen.ASSETS:
                records.append(
                    {
                        "route_id": route,
                        "family": "family",
                        "source_block": direct.frozen.PARTITIONS[0],
                        "expectation_episode_id": episode,
                        "pair": pair,
                        "horizon_minutes": 5,
                        "predicted_base_direction": 1,
                        "orientation_policy": "fixed_inverse_rate_pressure",
                        "previous_release_proxy_direction": 0,
                        "official_event_id": episode,
                        "anchor_utc": pd.Timestamp("2025-01-01T00:00:00Z"),
                        "response_return": actual * 0.01,
                        "actual_direction": actual,
                        "pre_event_return": actual * 0.005,
                        "pretrend_direction": actual,
                        "ordinary_control_count": 12,
                        "ordinary_up_rate": 0.5,
                        "abstention_reason": None,
                    }
                )
    return DataFrame.from_records(records)


def test_whole_event_permutation_keeps_routes_and_assets_together() -> None:
    rows = permutation_rows()
    shuffled = direct.permute_whole_event_outcomes(
        rows, np.random.default_rng(4)
    )
    for _, event in shuffled.groupby("expectation_episode_id"):
        assert event["actual_direction"].nunique() == 1
    assert sorted(shuffled["actual_direction"].unique()) == [-1, 1]


def test_observed_statistics_requires_same_route_for_shared_horizon() -> None:
    decisions = DataFrame(
        {
            "route_id": ["one", "one", "two"],
            "pair": [direct.frozen.ASSETS[0], direct.frozen.ASSETS[1], direct.frozen.ASSETS[0]],
            "horizon_minutes": [15, 15, 30],
            "raw_candidate": [True, True, True],
        }
    )
    statistics = direct.observed_statistics(decisions)
    assert statistics["candidate_cells"] == 3
    assert statistics["max_candidate_cells_one_route"] == 2
    assert statistics["max_shared_horizons_one_route"] == 1


def test_summary_compares_each_control_on_the_same_events() -> None:
    rows = DataFrame(
        {
            "expectation_episode_id": ["one", "two", "three"],
            "direction_hit": [True, True, False],
            "development_majority_hit": [False, False, True],
            "pretrend_hit": [False, True, False],
            "previous_proxy_hit": [True, np.nan, np.nan],
            "rotated_hit": [False, False, True],
            "ordinary_control_accuracy": [0.5, 0.5, 0.5],
            "response_return": [0.01, 0.02, -0.01],
        }
    )
    summary = direct.summarize_group(rows)
    assert summary["previous_proxy_paired_events"] == 1
    assert summary["main_on_previous_proxy_events_accuracy"] == 1.0
    assert summary["previous_proxy_accuracy"] == 1.0
    assert summary["previous_proxy_paired_lift"] == 0.0
    assert summary["events_additional_to_previous_proxy"] == 2

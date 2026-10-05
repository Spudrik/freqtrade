# ruff: noqa: S101

from __future__ import annotations

import numpy as np
from pandas import DataFrame

from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_cpi_validation as validation,
)


def test_pre_release_control_compares_prediction_with_prior_window() -> None:
    rows = DataFrame.from_records(
        [
            {
                "event_id": "a",
                "anchor_utc": "2022-01-01T13:30:00Z",
                "whole_event_partition": "development_2021_2023",
                "clock": "immediate_1m",
                "scope": "btc",
                "sign_id": "temperature_core_mom",
                "horizon": 5,
                "predicted_direction": -1.0,
                "pre_return": 0.01,
                "direction_hit": True,
            }
        ]
    )
    summary = validation.pre_release_controls(rows).iloc[0]
    assert summary["n_events"] == 1
    assert summary["post_release_accuracy"] == 1.0
    assert summary["pre_release_accuracy"] == 0.0
    assert summary["post_minus_pre"] == 1.0


def test_joint_permutation_preserves_complete_sign_rows() -> None:
    event_count = 40
    event_ids = [f"event_{index}" for index in range(event_count)]
    partitions = ["development_2021_2023"] * 20 + [
        "internal_validation_2024_2025"
    ] * 20
    base_sign = np.array(([1.0, -1.0] * 20), dtype=float)
    catalog = DataFrame(
        {
            "event_id": event_ids,
            "whole_event_partition": partitions,
            **{
                column: base_sign.copy()
                for column in validation.frozen.ALL_SIGN_COLUMNS
            },
        }
    )
    minute = DataFrame.from_records(
        [
            {
                "event_id": event_id,
                "scope": "btc",
                "horizon": 5,
                "whole_event_partition": partition,
                "fomc_within_24h": False,
                "response_return": -sign * 0.01,
            }
            for event_id, partition, sign in zip(
                event_ids, partitions, base_sign, strict=True
            )
        ]
    )
    result = validation.whole_event_permutation(
        catalog, minute, iterations=40, seed=123
    )
    assert set(result["family"]) == {
        "full_frozen_immediate_family",
        "selected_mom_family",
    }
    assert result["observed_best_minimum_period_accuracy"].eq(1.0).all()
    assert result["familywise_probability"].between(0, 1).all()


def test_validation_decision_requires_all_negative_controls() -> None:
    parent = DataFrame.from_records(
        [
            {
                "sample_variant": "all_releases",
                "clock": "immediate_1m",
                "sign_id": "temperature_core_mom",
                "scope": scope,
                "horizon": horizon,
                "verdict": "retained_cross_period_direction_lead",
            }
            for scope in validation.SELECTED_SCOPES
            for horizon in validation.SHORT_HORIZONS
        ]
    )
    pre = DataFrame.from_records(
        [
            {
                "sign_id": "temperature_core_mom",
                "scope": scope,
                "horizon": horizon,
                "whole_event_partition": partition,
                "post_release_accuracy": 0.70,
                "pre_release_accuracy": 0.50,
            }
            for scope in validation.SELECTED_SCOPES
            for horizon in validation.SHORT_HORIZONS
            for partition in validation.direct.EVALUATION_PARTITIONS
        ]
    )
    matched = DataFrame.from_records(
        [
            {
                "sign_id": "temperature_core_mom",
                "scope": scope,
                "horizon": horizon,
                "mean_event_minus_matched": 0.15,
            }
            for scope in validation.SELECTED_SCOPES
            for horizon in validation.SHORT_HORIZONS
        ]
    )
    permutation = DataFrame.from_records(
        [
            {
                "family": "full_frozen_immediate_family",
                "familywise_probability": 0.01,
            }
        ]
    )
    decision = validation.validation_decision(parent, pre, matched, permutation)
    assert decision["status"] == "retained_for_true_consensus_and_confirmation_test"

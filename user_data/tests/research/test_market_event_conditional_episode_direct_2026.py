# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_event_conditional_episode_direct_2026 as direct,
)


def test_event_classes_are_kept_separate() -> None:
    assert (
        direct.event_class('["scheduled_us_macro"]')
        == "scheduled_policy_or_macro"
    )
    assert direct.event_class('["daily_cross_market_state"]') == "daily_cross_market_state"
    assert (
        direct.event_class('["scheduled_us_macro", "daily_cross_market_state"]')
        == "mixed_event_classes"
    )


def test_event_scope_keeps_class_and_specific_family() -> None:
    labels = direct.event_scope_labels('["scheduled_us_macro"]', '["us_cpi"]')
    assert labels == [
        "all_catalogued_events",
        "class:scheduled_policy_or_macro",
        "family:us_cpi",
    ]


def test_decision_needs_repetition_not_one_large_cell() -> None:
    base = {
        "route_id": "route",
        "role": "modifier",
        "condition": "state",
        "outcome": "volume_reaction",
        "control_match": "same_state",
        "event_scope": "all_catalogued_events",
        "pair": "BTC/USDT:USDT",
        "horizon_hours": 1,
        "false_event_count": 20,
        "true_event_count": 20,
        "conditional_effect": 0.40,
    }
    cells = pd.DataFrame(
        [
            {**base, "model_period": "walk_forward_validation_2024"},
            {
                **base,
                "model_period": "walk_forward_validation_2025",
                "true_event_count": 2,
            },
        ]
    )
    decision = direct.decide(cells).iloc[0]
    assert decision["verdict"].startswith("coverage_parked")


def test_opposing_material_periods_are_unresolved_not_retained() -> None:
    base = {
        "route_id": "route",
        "role": "modifier",
        "condition": "state",
        "outcome": "volume_reaction",
        "control_match": "same_state",
        "event_scope": "all_catalogued_events",
        "pair": "BTC/USDT:USDT",
        "horizon_hours": 1,
        "false_event_count": 20,
        "true_event_count": 20,
    }
    cells = pd.DataFrame(
        [
            {
                **base,
                "model_period": "development_2021_2023",
                "conditional_effect": 0.20,
            },
            {
                **base,
                "model_period": "walk_forward_validation_2024",
                "conditional_effect": 0.25,
            },
            {
                **base,
                "model_period": "walk_forward_validation_2025",
                "conditional_effect": -0.30,
            },
        ]
    )
    decision = direct.decide(cells).iloc[0]
    assert decision["verdict"] == "unresolved_effect_changes_sign_between_periods"

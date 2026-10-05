# ruff: noqa: S101

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from pandas import DataFrame

from user_data.Custom_Launcher.research.context_features import (
    market_event_freqai_breadth_freeze as freeze,
)


def test_decision_hour_never_precedes_source_time() -> None:
    exact = pd.Timestamp("2024-01-01T08:00:00Z")
    intrahour = pd.Timestamp("2024-01-01T08:30:01Z")
    assert freeze.decision_hour(exact) == exact
    assert freeze.decision_hour(intrahour) == pd.Timestamp("2024-01-01T09:00:00Z")


def test_event_episodes_join_only_nearby_events() -> None:
    frame = DataFrame(
        {
            "event_id": ["a", "b", "c"],
            "model_anchor_utc": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T07:00:00Z",
                    "2024-01-02T00:00:00Z",
                ],
                utc=True,
            ),
        }
    )
    result = freeze.assign_event_episodes(frame)
    assert result.loc[0, "event_episode_id"] == result.loc[1, "event_episode_id"]
    assert result.loc[1, "event_episode_id"] != result.loc[2, "event_episode_id"]


def test_profile_registry_keeps_components_separable() -> None:
    registry = freeze.build_registry()
    assert registry["profile_count"] == 14
    assert set(registry["profiles"]) == set(freeze.PROFILE_DEFINITIONS)
    assert len(registry["comparisons"]) == 6
    for comparison in registry["comparisons"]:
        candidate = registry["profiles"][comparison["candidate"]]
        component_features = {
            feature
            for component in comparison["components"]
            for feature in registry["profiles"][component]["feature_columns"]
        }
        assert set(candidate["feature_columns"]) == component_features


def test_sample_catalog_distinguishes_actual_sign_from_control() -> None:
    events = DataFrame(
        {
            "event_id": ["event_a", "event_b"],
            "event_episode_id": ["episode_1", "episode_1"],
            "event_family": ["us_cpi", "fomc_policy_decision"],
            "event_kind": ["scheduled_us_macro", "scheduled_us_policy"],
            "model_anchor_utc": pd.to_datetime(
                ["2024-01-10T14:00:00Z", "2024-01-10T14:00:00Z"], utc=True
            ),
            "exact_clock": [True, True],
            "scheduled": [True, True],
            "source_sign_primary": [-1.0, np.nan],
            "source_sign_secondary": [-1.0, np.nan],
            "crypto_relation_sign": [1.0, np.nan],
        }
    )
    controls = DataFrame(
        {
            "parent_event_id": ["event_a"],
            "parent_episode_id": ["episode_1"],
            "event_family": ["us_cpi"],
            "event_kind": ["scheduled_us_macro"],
            "control_anchor_utc": pd.to_datetime(["2024-01-03T14:00:00Z"], utc=True),
            "parent_exact_clock": [True],
            "parent_scheduled": [True],
        }
    )
    result = freeze.build_sample_catalog(events, controls)
    actual = result.loc[result["sample_kind"].eq("actual_event")].iloc[0]
    control = result.loc[result["sample_kind"].eq("matched_control")].iloc[0]
    assert actual["current_event_count"] == 2
    assert set(json.loads(actual["event_families_json"])) == {
        "fomc_policy_decision",
        "us_cpi",
    }
    assert actual["source_sign_available_count"] == 1
    assert control["current_event_count"] == 0
    assert control["source_sign_available_count"] == 0

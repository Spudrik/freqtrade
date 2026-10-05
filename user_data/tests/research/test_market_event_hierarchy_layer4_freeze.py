# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_layer4_freeze as freeze,
)


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair_id": "local_1h",
                "pair_family": "event_plus_local_technical_state",
                "event_source": "gdelt_activity_spike",
                "scope": "btc",
                "horizon_hours": 1,
                "state": "single_level",
            },
            {
                "pair_id": "local_2h",
                "pair_family": "event_plus_local_technical_state",
                "event_source": "gdelt_activity_spike",
                "scope": "btc",
                "horizon_hours": 2,
                "state": "single_level",
            },
        ]
    )


def _support(event_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, object]] = []
    background: list[dict[str, object]] = []
    for index in range(event_count):
        partition = (
            "development_2021_2023"
            if index < event_count // 2
            else "internal_validation_2024_2025"
        )
        event_id = f"event_{index}"
        metrics.append(
            {
                "event_id": event_id,
                "event_source": "gdelt_activity_spike",
                "whole_event_partition": partition,
                "sample_type": "event",
                "sample_anchor_utc": pd.Timestamp("2022-01-01T00:00:00Z")
                + pd.Timedelta(days=index),
                "pair": "BTC/USDT:USDT",
                "horizon_hours": 4,
                "signed_return": 0.02,
                "local_level_state": "single_level",
                "nearest_level": "ema50",
                "nearest_level_family": "moving_average",
            }
        )
        background.append(
            {
                "event_id": event_id,
                "event_source": "gdelt_activity_spike",
                "sample_type": "event",
                "background": "negative_30d_background",
            }
        )
    return pd.DataFrame(metrics), pd.DataFrame(background)


def test_retained_local_horizons_are_consolidated_before_chain_testing() -> None:
    groups = freeze.retained_local_groups(_candidate_rows())
    assert len(groups) == 1
    assert groups.iloc[0]["evidence_horizons"] == "1,2"


def test_chain_preflight_activates_only_with_full_whole_event_support() -> None:
    metrics, background = _support(12)
    active, parked = freeze.chain_preflight(_candidate_rows(), metrics, background)
    assert len(active) == 1
    assert parked.empty
    assert active.iloc[0]["full_chain_event_count"] == 12
    assert active.iloc[0]["development_full_count"] == 6
    assert active.iloc[0]["validation_full_count"] == 6


def test_chain_preflight_parks_a_tiny_intersection_without_opening_target() -> None:
    metrics, background = _support(8)
    active, parked = freeze.chain_preflight(_candidate_rows(), metrics, background)
    assert active.empty
    assert len(parked) == 1
    assert parked.iloc[0]["status"] == "coverage_parked"

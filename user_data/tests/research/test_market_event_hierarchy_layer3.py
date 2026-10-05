# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_layer3_direct as direct,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_layer3_freeze as freeze,
)


def test_freeze_builds_complete_unique_pair_and_parked_matrices() -> None:
    local = pd.DataFrame(
        [
            {
                "event_source": "official_fomc",
                "cohort": "btc",
                "horizon_hours": 2,
                "local_level_state": "single_level",
                "meets_55_floor_and_beats_far": True,
            },
            {
                "event_source": "gdelt_activity_spike",
                "cohort": "memes",
                "horizon_hours": 4,
                "local_level_state": "cluster",
                "meets_55_floor_and_beats_far": True,
            },
            {
                "event_source": "official_fomc",
                "cohort": "eth",
                "horizon_hours": 8,
                "local_level_state": "cluster",
                "meets_55_floor_and_beats_far": False,
            },
        ]
    )
    active = freeze.active_pair_matrix(local)
    assert len(active) == 2 + 32 + 2 + 20
    assert not active["pair_id"].duplicated().any()
    assert set(active["pair_family"]) == {
        "background_plus_event_confirmation",
        "event_plus_first_broad_response",
        "event_plus_local_technical_state",
        "event_plus_prior_30d_range_position",
    }
    assert len(freeze.parked_pair_matrix()) == 6


def _background_fixture() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    anchor = pd.Timestamp("2021-01-01T00:00:00Z")
    for source in freeze.EVENT_SOURCES:
        for index in range(20):
            partition = (
                direct.PARTITIONS[0] if index < 10 else direct.PARTITIONS[1]
            )
            background = (
                "negative_30d_background"
                if index % 10 < 5
                else "positive_30d_background"
            )
            records.append(
                {
                    "event_id": f"{source}_{index}",
                    "event_source": source,
                    "whole_event_partition": partition,
                    "sample_type": "event",
                    "control_type": "event",
                    "control_rank": 0,
                    "sample_anchor_utc": anchor + pd.Timedelta(days=index),
                    "background": background,
                    "positive_initial_move": True,
                    "positive_move_faded_by_half": background
                    == "negative_30d_background",
                }
            )
            records.append(
                {
                    "event_id": f"{source}_{index}",
                    "event_source": source,
                    "whole_event_partition": partition,
                    "sample_type": "control",
                    "control_type": direct.PRIMARY_CONTROL,
                    "control_rank": 1,
                    "sample_anchor_utc": anchor
                    + pd.Timedelta(days=index)
                    - pd.Timedelta(days=7),
                    "background": "negative_30d_background",
                    "positive_initial_move": True,
                    "positive_move_faded_by_half": False,
                }
            )
    return pd.DataFrame.from_records(records)


def test_background_pair_requires_event_and_background_increment() -> None:
    summary = direct.summarize_background_pairs(_background_fixture())
    assert len(summary) == 2
    assert summary["meets_pair_rule"].all()
    assert (summary["pair_fade_rate"] == 1.0).all()
    assert (summary["background_only_fade_rate"] == 0.0).all()


def test_broad_confirmation_uses_non_overlapping_return_after_hour_one() -> None:
    pairs = [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "BNB/USDT:USDT",
        "SOL/USDT:USDT",
        "XRP/USDT:USDT",
        "ADA/USDT:USDT",
        "TRX/USDT:USDT",
        "AVAX/USDT:USDT",
        "LINK/USDT:USDT",
    ]
    records: list[dict[str, object]] = []
    for pair in pairs:
        for horizon, signed_return in ((1, 0.01), (2, 0.03), (4, 0.04), (8, 0.05), (24, 0.06)):
            records.append(
                {
                    "sample_id": "event_1",
                    "event_id": "event_1",
                    "event_source": "official_fomc",
                    "whole_event_partition": direct.PARTITIONS[0],
                    "sample_type": "event",
                    "control_type": "event",
                    "control_rank": 0,
                    "sample_anchor_utc": pd.Timestamp("2022-01-01T00:00:00Z"),
                    "pair": pair,
                    "horizon_hours": horizon,
                    "signed_return": signed_return,
                    "activity_score": 2.0,
                }
            )
    contract = {
        "cohorts": {
            "established_alts": pairs[2:],
            "top_ten_traded_memes": [],
        }
    }
    result = direct.build_broad_response_rows(pd.DataFrame(records), contract)
    btc = result.loc[
        result["scope"].eq("btc") & result["horizon_hours"].eq(2)
    ].iloc[0]
    assert bool(btc["broad_confirmation"])
    assert bool(btc["continued"])
    assert np.isclose(btc["median_post_first_hour_return"], 1.03 / 1.01 - 1)


def test_single_state_pair_measures_increment_beyond_both_components() -> None:
    records: list[dict[str, object]] = []
    anchor = pd.Timestamp("2021-01-01T00:00:00Z")
    for index in range(24):
        state = "target" if index % 2 == 0 else "far"
        activity = 2.0 if state == "target" else 1.0
        control_activity = 1.1 if state == "target" else 1.0
        partition = direct.PARTITIONS[0] if index < 12 else direct.PARTITIONS[1]
        common = {
            "event_id": f"event_{index}",
            "event_source": "official_fomc",
            "whole_event_partition": partition,
            "pair": "BTC/USDT:USDT",
            "horizon_hours": 4,
            "state": state,
        }
        records.extend(
            [
                {
                    **common,
                    "sample_id": f"event_{index}|event",
                    "sample_type": "event",
                    "control_type": "event",
                    "control_rank": 0,
                    "sample_anchor_utc": anchor + pd.Timedelta(days=index),
                    "activity_score": activity,
                },
                {
                    **common,
                    "sample_id": f"event_{index}|control",
                    "sample_type": "control",
                    "control_type": direct.PRIMARY_CONTROL,
                    "control_rank": 1,
                    "sample_anchor_utc": anchor
                    + pd.Timedelta(days=index)
                    - pd.Timedelta(days=7),
                    "activity_score": control_activity,
                },
            ]
        )
    stats = direct._single_state_stats(
        pd.DataFrame.from_records(records), "state", "target", "far"
    )
    assert stats["target_event_count"] == 12
    assert stats["far_event_count"] == 12
    assert np.isclose(stats["event_target_minus_far"], 1.0)
    assert np.isclose(stats["control_target_minus_far"], 0.1)
    assert np.isclose(stats["pair_increment_beyond_components"], 0.9)
    assert direct._state_pair_pass(stats, shuffle_q75=0.2)


def test_repetition_needs_adjacent_horizon_or_both_sources() -> None:
    summary = pd.DataFrame(
        [
            {
                "pair_id": "a",
                "event_source": "official_fomc",
                "scope": "btc",
                "horizon_hours": 2,
                "state": "single",
                "meets_pair_rule": True,
            },
            {
                "pair_id": "b",
                "event_source": "official_fomc",
                "scope": "btc",
                "horizon_hours": 4,
                "state": "single",
                "meets_pair_rule": True,
            },
            {
                "pair_id": "isolated",
                "event_source": "gdelt_activity_spike",
                "scope": "eth",
                "horizon_hours": 24,
                "state": "cluster",
                "meets_pair_rule": True,
            },
        ]
    )
    assert direct._repeated_cell_ids(summary) == {"a", "b"}

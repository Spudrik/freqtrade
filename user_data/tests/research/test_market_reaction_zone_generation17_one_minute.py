# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_one_minute_acquisition as g17q,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_one_minute_analysis as g17a,
)


def test_finalized_interval_scales_with_anchor_and_adds_feature_warmup() -> None:
    event = pd.Timestamp("2026-01-10 00:00:00", tz="UTC")
    sample = pd.DataFrame(
        {
            "episode_id": ["episode"],
            "cohort": ["normal"],
            "pair": ["BTC/USDT:USDT"],
            "event_time": [event],
            "anchor_source_timeframe": ["4h"],
            "pre_context_hours": [48],
            "post_context_hours": [48],
        }
    )
    interval = g17q.finalized_intervals(sample).iloc[0]
    assert pd.Timestamp(interval["visible_start_utc"]) == event - pd.Timedelta(hours=48)
    assert pd.Timestamp(interval["interval_start_utc"]) == event - pd.Timedelta(hours=72)
    assert pd.Timestamp(interval["visible_end_exclusive_utc"]) == event + pd.Timedelta(
        hours=49
    )


def test_direction_helpers_are_causal_and_can_abstain() -> None:
    assert g17a.approach_call("from_below") == -1
    assert g17a.approach_call("from_above") == 1
    assert g17a.approach_call("inside") == 0
    assert g17a.threshold_call(20, 30, 70, reverse=True) == 1
    assert g17a.threshold_call(50, 30, 70, reverse=True) == 0
    assert g17a.count_vote(4) == 1
    assert g17a.count_vote(3) == 0


def test_approach_rejection_sign_and_density_gate() -> None:
    common = {
        "cohort": "normal",
        "period": "test",
        "pair": "BTC/USDT:USDT",
        "density_regime": "high",
        "match_return_15m": 0.0,
        "match_return_60m": 0.0,
        "match_return_240m": 0.0,
        "match_pressure_20m": 0.0,
        "contact_candle_pressure": 0.0,
        "completed_5m_pressure": 0.0,
        "rolling_5m_pressure_acceleration": 0.0,
        "completed_5m_volume_ratio_prior20": 1.0,
        "btc_orderbook_usable_coverage": False,
        "tf_1m__di_spread14": 0.0,
        "tf_5m__di_spread14": 0.0,
        "tf_15m__di_spread14": 0.0,
        "tf_1h__di_spread14": 0.0,
        "mtf_rsi_above_50_count": 3,
        "mtf_macd_positive_count": 3,
        "tf_1m__rsi14": 50.0,
        "tf_5m__rsi14": 50.0,
        "tf_1m__bollinger_position20": 0.5,
        "tf_5m__bollinger_position20": 0.5,
        "tf_1m__range_position20": 0.5,
        "tf_5m__range_position20": 0.5,
    }
    actual = pd.DataFrame(
        [
            {**common, "episode_id": "below", "approach_state": "from_below"},
            {
                **common,
                "episode_id": "above",
                "approach_state": "from_above",
                "density_regime": "low",
            },
            {**common, "episode_id": "inside", "approach_state": "inside"},
        ]
    )

    calls = g17a.causal_calls(actual)
    direct = calls.loc[calls["method"].eq("approach_rejection")].set_index("episode_id")
    gated = calls.loc[calls["method"].eq("density_approach_rejection")].set_index(
        "episode_id"
    )
    assert direct["call_numeric"].to_dict() == {"below": -1, "above": 1, "inside": 0}
    assert gated["call_numeric"].to_dict() == {"below": -1, "above": 0, "inside": 0}
    assert direct["call_direction"].to_dict() == {
        "below": "down",
        "above": "up",
        "inside": "abstain",
    }


def test_decision_table_requires_both_cohorts_for_broad_lead() -> None:
    rows = []
    for scope, rate, episodes in (("all", 0.60, 40), ("normal", 0.60, 20), ("meme", 0.55, 20)):
        rows.append(
            {
                "scope": scope,
                "horizon_minutes": 60,
                "method": "method",
                "method_kind": "causal_candidate",
                "independent_episodes": episodes,
                "issued_call_coverage": 1.0,
                "joint_success_rate_all_episodes": rate,
                "joint_wilson_lower": 0.51,
            }
        )
    decision = g17a.decision_table(pd.DataFrame(rows)).iloc[0]
    assert decision["broad_point_lead"]
    assert decision["uncertainty_supported_above_chance"]

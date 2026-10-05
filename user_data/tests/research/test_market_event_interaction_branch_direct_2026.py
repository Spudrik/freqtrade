# ruff: noqa: S101

from __future__ import annotations

import json

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_event_interaction_branch_direct_2026 as direct,
)


def test_role_context_deduplicates_related_response_domains() -> None:
    samples = pd.DataFrame(
        {
            "model_anchor_utc": [pd.Timestamp("2025-01-02", tz="UTC")],
            "sample_kind": ["actual_event"],
            "event_families_json": [json.dumps(["cross_market_fear"])],
        }
    )
    events = pd.DataFrame(
        {
            "model_anchor_utc": [
                pd.Timestamp("2025-01-02", tz="UTC"),
                pd.Timestamp("2025-01-02", tz="UTC"),
            ],
            "event_family": ["cross_market_fear", "cross_technology_equities"],
            "event_episode_id": ["e1", "e1"],
            "crypto_relation_sign": [-1.0, -1.0],
        }
    )

    result = direct.build_role_context(samples, events).iloc[0]

    assert result["response_domain_count"] == 1
    assert not bool(result["multiple_response_domains"])


def test_role_context_does_not_count_gdelt_as_upstream_story() -> None:
    samples = pd.DataFrame(
        {
            "model_anchor_utc": [pd.Timestamp("2025-01-02", tz="UTC")],
            "sample_kind": ["actual_event"],
            "event_families_json": [json.dumps(["gdelt_unexpected_activity"])],
        }
    )
    events = pd.DataFrame(
        {
            "model_anchor_utc": [pd.Timestamp("2025-01-02", tz="UTC")],
            "event_family": ["gdelt_unexpected_activity"],
            "event_episode_id": ["e1"],
            "crypto_relation_sign": [None],
        }
    )

    result = direct.build_role_context(samples, events).iloc[0]

    assert result["upstream_domain_count"] == 0
    assert result["media_attention_proxy_count"] == 1
    assert bool(result["current_is_media_attention_proxy"])


def test_actual_episode_rows_selects_one_anchor() -> None:
    rows = pd.DataFrame(
        {
            "sample_kind": ["actual_event", "actual_event"],
            "parent_episode_ids_json": [json.dumps(["e1"]), json.dumps(["e1"])],
            "model_period": ["p1", "p1"],
            "pair": ["BTC", "BTC"],
            "horizon_hours": [1, 1],
            "date": [
                pd.Timestamp("2025-01-01 01:00", tz="UTC"),
                pd.Timestamp("2025-01-01 02:00", tz="UTC"),
            ],
            "sample_id": ["early", "late"],
        }
    )

    earliest = direct._actual_episode_rows(rows, keep="first")
    latest = direct._actual_episode_rows(rows, keep="last")

    assert earliest.iloc[0]["sample_id"] == "early"
    assert latest.iloc[0]["sample_id"] == "late"


def test_comparison_balances_event_family_composition() -> None:
    summary = pd.DataFrame(
        {
            "model_period": ["p1", "p1", "p1", "p1", "p1"],
            "pair": ["BTC"] * 5,
            "horizon_hours": [1] * 5,
            "family": ["a", "a", "b", "b", "unmatched"],
            "state": ["test", "baseline", "test", "baseline", "test"],
            "event_count": [100, 10, 10, 100, 100],
            "raw_event_rate": [1.0, 0.0, 0.0, 1.0, 1.0],
            "matched_control_rate": [0.5] * 5,
            "mean_event_minus_control": [0.5, 0.0, 0.0, 0.5, 0.5],
            "median_controls_per_event": [4.0] * 5,
        }
    )

    result = direct._comparison_cells(
        summary,
        branch_id="test",
        group_column="state",
        stratum_column="family",
        test_group="test",
        baseline_group="baseline",
        comparison="family_balanced",
    ).iloc[0]

    assert result["matched_strata"] == 2
    assert result["test_count"] == 110
    assert result["baseline_count"] == 110
    assert result["test_rate"] == 0.5
    assert result["baseline_rate"] == 0.5
    assert result["conditional_effect"] == 0.0

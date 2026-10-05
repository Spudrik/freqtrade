# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_joint_review as g22j,
)


def test_generation23_queue_preserves_five_diverse_active_routes() -> None:
    queue = g22j.generation23_queue()
    active = [item for item in queue if item["status"].startswith("active_")]
    parked = [item for item in queue if item["status"].startswith("parked_")]

    assert len(active) == 5
    assert len(parked) == 5
    assert len({item["route_family"] for item in active}) == 5


def test_generation23_repairs_are_bounded_and_not_confirmation() -> None:
    queue = {item["branch_id"]: item for item in g22j.generation23_queue()}

    mtf = queue["g23a_multitimeframe_cluster_incremental_attribution"]
    freqai = queue["g23c_freqai_calibrated_unsigned_reaction"]
    future = queue["g23f_multitimeframe_new_period_confirmation"]

    assert "same-holdout attribution" in mtf["scope_guard"]
    assert "same 12" in freqai["scope_guard"]
    assert "no profit or signed direction" in freqai["scope_guard"]
    assert future["status"] == "parked_awaiting_new_data"


def test_ladder_summary_does_not_promote_incomplete_positive_rows() -> None:
    frame = pd.DataFrame(
        {
            "scope_kind": ["complete", "incomplete"],
            "scope_value": ["weak", "large"],
            "metric": ["crossing_count", "crossing_count"],
            "horizon_hours": [8, 8],
            "market_scope": ["all_normal", "all_normal"],
            "status": ["not_retained", "not_retained"],
            "complete_control_period_ladder": [True, False],
            "minimum_equal_coin_difference": [0.01, 0.8],
            "minimum_bootstrap_lower": [-0.02, 0.7],
        }
    )

    summary = g22j.ladder_summary(frame)

    assert summary["complete_ladder_rows"] == 1
    assert summary["complete_positive_minimum_rows"] == 1
    assert summary["incomplete_positive_minimum_rows"] == 1
    assert summary["best_complete_ladder_row"]["scope_value"] == "weak"


def test_freqai_summary_keeps_descriptive_ranking_unretained() -> None:
    decisions = pd.DataFrame(
        {
            "cohort": ["normal"],
            "market_scope": ["BTC_only"],
            "target": ["future_volume_ratio_h1"],
            "complete_control_and_period_ladder": [True],
            "all_controls_both_periods_pass": [False],
            "minimum_candidate_top_above_training_median_fraction": [0.76],
            "minimum_candidate_top_minus_bottom": [0.85],
        }
    )
    controls = pd.DataFrame(
        {
            "control": ["constant_training_median"],
            "candidate_gate_pass": [True],
            "control_metrics_pass": [False],
            "period_control_pass": [False],
        }
    )

    summary = g22j.freqai_summary(decisions, controls)

    assert summary["all_controls_both_periods_pass_rows"] == 0
    assert summary["best_descriptive_ranking_row"]["retained"] is False

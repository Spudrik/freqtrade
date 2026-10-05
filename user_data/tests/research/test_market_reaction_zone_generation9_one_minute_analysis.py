from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation9_one_minute_analysis as g9m,
)


def test_approach_maps_to_through_direction_without_outcome_data() -> None:
    assert g9m.through_level_name("from_below") == "lvn_above"
    assert g9m.through_level_name("from_above") == "lvn_below"


def test_direction_portfolio_has_multiple_fixed_methods_and_comparators() -> None:
    assert len(g9m.CALL_METHODS) == 9
    assert "leave_one_out_cohort_majority" in g9m.CALL_METHODS
    assert "btc_orderbook_pressure_if_usable" in g9m.CALL_METHODS
    assert set(g9m.CONTROL_KINDS) == {
        "same_state_no_level",
        "causal_stale_level_contact",
        "deterministic_shifted_level_contact",
    }


def test_call_summary_never_promotes_twelve_episode_diagnostic() -> None:
    calls = pd.DataFrame.from_records(
        [
            {
                "episode_id": f"e{index}",
                "cohort": "normal" if index < 6 else "meme",
                "method": "pre_60m_trend",
                "issued": True,
                "reaction_60m": True,
                "actual_direction_callable": True,
                "direction_correct": index < 8,
                "joint_reaction_and_direction": index < 8,
            }
            for index in range(12)
        ]
    )
    summary = g9m.call_summary(calls)
    overall = summary.loc[summary["scope"].eq("all")].iloc[0]
    assert overall["joint_success_rate"] == 8 / 12
    assert bool(overall["observed_at_or_above_65pct"])
    assert not bool(overall["lead_eligible_sample_size"])
    assert overall["status"] == "diagnostic_only_insufficient_independent_episodes"


def test_control_comparison_remains_paired_by_episode() -> None:
    actual = pd.DataFrame.from_records(
        [
            {
                "episode_id": "e1",
                "cohort": "normal",
                **{metric: 2.0 for metric in g9m.PAIR_METRICS},
            }
        ]
    )
    control = pd.DataFrame.from_records(
        [
            {
                "episode_id": "e1",
                "cohort": "normal",
                "event_kind": "same_state_no_level",
                **{metric: 1.0 for metric in g9m.PAIR_METRICS},
            }
        ]
    )
    result = g9m.control_comparisons(actual, control)
    selected = result.loc[
        result["cohort"].eq("normal")
        & result["control_kind"].eq("same_state_no_level")
    ]
    assert selected["paired_episodes"].eq(1).all()
    assert selected["median_paired_difference"].eq(1.0).all()

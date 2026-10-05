# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation11_direct_screen as g11d,
)


def test_reaction_requires_price_and_volume() -> None:
    frame = pd.DataFrame(
        {
            "zone_half_width_atr": [0.2, 0.8, 0.2],
            "abs_excursion_atr_h1": [0.6, 0.7, 0.6],
            "volume_ratio_h1": [1.3, 1.3, 1.2],
            "range_ratio_h1": [1.0, 1.0, 1.0],
            "dwell_fraction_h1": [0.0, 0.0, 0.0],
            "crossings_h1": [0.0, 0.0, 0.0],
        }
    )

    result = g11d.add_horizon_outcomes(frame, 1)

    assert result["reaction_rate"].tolist() == [1.0, 0.0, 0.0]


def test_resolution_treats_ties_and_unresolved_as_failures() -> None:
    frame = pd.DataFrame(
        {
            "time_to_away_0_5atr": [1.0, 3.0, 2.0, float("nan")],
            "time_to_through_0_5atr": [2.0, 1.0, 2.0, float("nan")],
        }
    )

    result = g11d.path_resolution(frame, 4)

    assert result.tolist() == [-1, 1, 0, 0]


def test_direction_vote_is_relative_to_approach() -> None:
    vote = pd.Series([1, -1, 1, 0], dtype="int8")
    approach = pd.Series([1, 1, -1, 1], dtype="int8")

    result = g11d.direction_vote_to_path(vote, approach)

    assert result.tolist() == [1, -1, -1, 0]


def test_pair_summary_uses_mean_for_binary_reaction_rate() -> None:
    rows = []
    for reaction in (1.0, 0.0, 0.0):
        row = {
            "cohort": "normal",
            "pair": "A/USDT:USDT",
            "period": "validation_early",
            "control": "actual",
            **{metric: 1.0 for metric in g11d.OUTCOME_METRICS},
        }
        row["reaction_rate"] = reaction
        rows.append(row)
    frame = pd.DataFrame.from_records(rows)

    result = g11d.aggregate_pair_cells(
        frame,
        route="test",
        cells=pd.Series(["cell"] * len(frame)),
        horizon=1,
    )

    reaction = result.loc[result["metric"].eq("reaction_rate")].iloc[0]
    assert reaction["pair_median"] == 1.0 / 3.0


def test_neutral_candidate_needs_both_periods_horizons_and_all_controls() -> None:
    rows = []
    for period in ("validation_early", "validation_late"):
        for horizon in (1, 2):
            for control in g11d.CONTROLS:
                rows.append(
                    {
                        "route": "activity_displacement",
                        "cohort": "normal",
                        "cell": "activity=active",
                        "metric": "volume_ratio",
                        "minimum_coins": 5,
                        "period": period,
                        "horizon_hours": horizon,
                        "comparison_control": control,
                        "effect_sign": 1,
                        "supported": True,
                    }
                )
    effects = pd.DataFrame.from_records(rows)

    result = g11d.repeated_neutral_candidates(effects)
    short = result.loc[result["horizon_band"].eq("short")].iloc[0]

    assert bool(short["candidate"]) is True

    missing = effects.iloc[:-1]
    rejected = g11d.repeated_neutral_candidates(missing)
    rejected_short = rejected.loc[rejected["horizon_band"].eq("short")].iloc[0]
    assert bool(rejected_short["candidate"]) is False


def test_direction_candidate_uses_55pct_both_periods_and_comparator_margin() -> None:
    scores = pd.DataFrame(
        {
            "scope_id": ["normal_full_cohort", "normal_full_cohort"],
            "scope_type": ["full_cohort", "full_cohort"],
            "cohort": ["normal", "normal"],
            "method": ["local_five_indicator_vote", "local_five_indicator_vote"],
            "source_timeframe": ["4h", "4h"],
            "horizon_hours": [4, 4],
            "period": ["validation_early", "validation_late"],
            "issued_calls": [60, 60],
            "coverage": [0.5, 0.5],
            "coins": [8, 8],
            "joint_success_rate": [0.56, 0.57],
            "joint_success_lower_95": [0.51, 0.52],
            "comparator_margin": [0.03, 0.04],
            "comparator_margin_lower_95": [0.01, 0.02],
        }
    )

    result = g11d.direction_candidates(scores).iloc[0]

    assert bool(result["point_candidate_55pct"]) is True
    assert bool(result["strict_candidate"]) is True

    scores.loc[1, "joint_success_rate"] = 0.54
    rejected = g11d.direction_candidates(scores).iloc[0]
    assert bool(rejected["point_candidate_55pct"]) is False

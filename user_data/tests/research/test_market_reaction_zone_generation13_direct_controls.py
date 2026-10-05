# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation13_direct_controls as module,
)


def test_frozen_source_contracts_validate() -> None:
    frozen, manifest = module.validate_contracts()

    assert frozen["status"] == "frozen_before_generation13_outcomes"
    assert manifest["status"] == "completed_shared_causal_event_cache"


def test_reaction_label_requires_excursion_and_volume() -> None:
    frame = pd.DataFrame(
        {
            "zone_half_width_atr": [0.25, 0.75, 0.25, 0.25],
            "abs_excursion_atr_h1": [0.5, 0.7, 0.8, np.nan],
            "volume_ratio_h1": [1.25, 2.0, 1.0, 2.0],
        }
    )

    result = module.reaction_label(frame, 1)

    assert result.iloc[:3].tolist() == [1.0, 0.0, 0.0]
    assert np.isnan(result.iloc[3])


def test_audit_feature_derivation_centres_source_rsi() -> None:
    source = {column: [1.0] for column in module.AUDIT_SOURCE_COLUMNS}
    source["state__rsi14"] = [62.0]

    result = module.derive_audit_features(pd.DataFrame(source))

    assert result.loc[0, "state__rsi14_centered"] == 12.0


def test_add_analysis_windows_keeps_standard_and_recent_separate() -> None:
    frame = pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                ["2026-05-01T00:00:00Z", "2026-07-20T00:00:00Z"]
            ),
            "period": ["validation_early", "validation_late"],
        }
    )

    result = module.add_analysis_windows(frame, "normal")

    assert set(result["window"]) == {
        module.g13z.STANDARD,
        module.g13z.RECENT,
    }
    assert len(result) == 4


def test_balance_quality_reuses_strong_and_usable_lenses() -> None:
    assert (
        module.balance_quality({"max_absolute_smd": 0.2, "median_absolute_smd": 0.08})
        == "strong"
    )
    assert (
        module.balance_quality({"max_absolute_smd": 0.4, "median_absolute_smd": 0.15})
        == "usable"
    )
    assert (
        module.balance_quality({"max_absolute_smd": 0.6, "median_absolute_smd": 0.1})
        == "weak_or_sparse"
    )


def test_route_decision_requires_every_period_and_both_controls() -> None:
    rows = []
    for period in ("validation_early", "validation_late"):
        for control in module.CONTROLS:
            rows.append(
                {
                    "cohort": "normal",
                    "window": module.g13z.STANDARD,
                    "analysis_period": period,
                    "control": control,
                    "source_scope": "all_source_timeframes",
                    "horizon_hours": 1,
                    "point_cell_pass": True,
                    "strict_cell_pass": True,
                    "equal_coin_reaction_rate_difference": 0.02,
                    "bootstrap_lower_95": 0.01,
                    "eligible_coins": 8,
                }
            )

    complete = module.route_decisions(pd.DataFrame(rows))
    incomplete = module.route_decisions(pd.DataFrame(rows[:-1]))

    assert bool(complete.iloc[0]["strict_pass"])
    assert not bool(incomplete.iloc[0]["point_pass"])


def test_fast_hourly_purge_matches_established_greedy_rule() -> None:
    frame = pd.DataFrame(
        {
            "pair": ["A"] * 4,
            "window": ["w"] * 4,
            "analysis_period": ["p"] * 4,
            "control": ["c"] * 4,
            "match_distance": [0.1, 0.2, 0.3, 0.4],
            "pre_distance_atr_abs_difference": [0.1] * 4,
            "actual_event_time": pd.to_datetime(
                ["2026-01-01T00:00Z", "2026-01-01T01:00Z", "2026-01-01T04:00Z", "2026-01-01T08:00Z"]
            ),
            "control_event_time": pd.to_datetime(
                ["2026-01-03T00:00Z", "2026-01-03T01:00Z", "2026-01-03T04:00Z", "2026-01-03T08:00Z"]
            ),
            "level_name": ["l"] * 4,
            "approach_state": ["from_below"] * 4,
        }
    )
    groups = ["pair", "window", "analysis_period", "control"]

    expected = module.g1.purge_overlapping_event_pairs(
        frame,
        separation_hours=2,
        group_columns=groups,
        left_time_column="actual_event_time",
        right_time_column="control_event_time",
    )
    actual = module.purge_overlapping_hourly_pairs(
        frame, separation_hours=2, group_columns=groups
    )

    assert actual["actual_event_time"].tolist() == expected["actual_event_time"].tolist()

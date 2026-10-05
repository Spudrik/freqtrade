# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation26_state_location as g26,
)


def test_supported_families_requires_every_support_cell() -> None:
    rows = []
    for family in ("good", "bad"):
        for cohort in g26.COHORTS:
            periods = (
                ("g18_meme_holdout_early", "g18_meme_holdout_late")
                if cohort == "meme"
                else ("g18_normal_holdout_early", "g18_normal_holdout_late")
            )
            for period in periods:
                for control in ("actual", *g26.CONTROLS):
                    for state in g26.STATES:
                        for index in range(g26.MIN_BROAD_SCOPE_PAIRS):
                            rows.append(
                                {
                                    "cohort": cohort,
                                    "pair": f"PAIR{index}",
                                    "g18_period": period,
                                    "level_family": family,
                                    "control": control,
                                    "activity_state": state,
                                    "event_rows": (
                                        g26.MIN_PAIR_ROWS
                                        if family == "good" or index > 0
                                        else g26.MIN_PAIR_ROWS - 1
                                    ),
                                }
                            )
    original = g26.ALL_LEVEL_FAMILIES
    try:
        g26.ALL_LEVEL_FAMILIES = ("good", "bad")
        assert g26.supported_families(pd.DataFrame.from_records(rows)) == ["good"]
    finally:
        g26.ALL_LEVEL_FAMILIES = original


def test_interaction_contrast_is_high_minus_low_location_effect() -> None:
    base = {
        "cohort": "normal",
        "pair": "BTC/USDT:USDT",
        "g18_period": "g18_normal_holdout_early",
        "scope_kind": "family_any_level",
        "scope_value": "adaptive_volume_profile_nodes",
        "metric": "crossings",
        "horizon_hours": 2,
        "comparison": "near_miss",
        "eligible_pair": True,
    }
    frame = pd.DataFrame.from_records(
        [
            {**base, "activity_state": "high", "difference": 0.30},
            {**base, "activity_state": "low", "difference": 0.10},
        ]
    )
    result = g26.interaction_contrasts(frame)
    assert len(result) == 1
    assert result.iloc[0]["high_state_level_effect"] == 0.30
    assert result.iloc[0]["low_state_level_effect"] == 0.10
    assert abs(result.iloc[0]["difference"] - 0.20) < 1e-12
    assert bool(result.iloc[0]["eligible_pair"])


def test_combined_decision_needs_location_and_interaction() -> None:
    keys = {
        "scope_kind": "family_any_level",
        "scope_value": "adaptive_volume_profile_nodes",
        "metric": "crossings",
        "horizon_hours": 2,
        "market_scope": "all_normal",
    }
    location = pd.DataFrame.from_records(
        [
            {
                **keys,
                "activity_state": "high",
                "status": "strict_holdout_confirmation",
            }
        ]
    )
    interaction = pd.DataFrame.from_records(
        [{**keys, "status": "point_holdout_confirmation"}]
    )
    result = g26.combined_decisions(location, interaction)
    assert result.iloc[0]["status"] == "point_same_holdout_interaction_lead"
    assert bool(result.iloc[0]["point_lead"])
    assert not bool(result.iloc[0]["strict_lead"])

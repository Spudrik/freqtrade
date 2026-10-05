# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation16_direct_attribution as g16d,
)


def test_contact_matching_excludes_future_outcomes() -> None:
    assert not set(g16d.CONTACT_MATCH_FEATURES).intersection(
        g16d.RAW_OUTCOME_COLUMNS
    )
    assert set(g16d.RAW_CONTROLS) == {
        "actual",
        "matched_random_time",
        "near_miss",
        "stale_72h",
        "price_shift",
    }


def test_unsigned_reaction_requires_excursion_and_volume() -> None:
    frame = pd.DataFrame(
        {
            "actual__zone_half_width_atr": [0.25, 0.25],
            "actual__abs_excursion_atr_h1": [1.0, 1.0],
            "actual__volume_ratio_h1": [1.0, 2.0],
        }
    )

    values = g16d.metric_values(frame, "actual", "unsigned_reaction", 1)

    assert values.tolist() == [0.0, 1.0]


def test_context_combinations_retain_component_ablations() -> None:
    frame = pd.DataFrame(
        {
            "local_activity_high": [True],
            "wider_activity_high": [True],
            "eight_hour_activity_high": [True],
        }
    )

    definitions = g16d.context_definitions(frame)
    all_three = [item for item in definitions if item[0] == "local_plus_wider_plus_8h"]

    assert {item[1] for item in all_three} == {
        "vs_local_plus_wider",
        "vs_local_plus_8h",
        "vs_wider_plus_8h",
        "vs_neither",
    }

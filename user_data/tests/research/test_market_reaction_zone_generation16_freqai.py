# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation16 as g16f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation13_freeze as g13z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation16_freqai_cache as g16c,
)
from user_data.strategies.MarketReactionZoneFreqAIGeneration16Strategy import (
    TARGET_COLUMNS,
)


def available_feature_columns() -> list[str]:
    return [
        f"{block}__value"
        for block in (
            g16c.LEVEL,
            g16c.MINIMAL,
            g16c.GEOMETRY,
            g16c.WIDER,
            g16c.EIGHT_HOUR,
            g16c.STALE_MINIMAL,
            g16c.STALE_LEVEL,
            g16c.SHUFFLED_MINIMAL,
            g16c.SHUFFLED_LEVEL,
        )
    ]


def test_registry_has_frozen_profiles_and_component_controls() -> None:
    registry = g16c.build_registry(available_feature_columns())
    normal = {
        profile["role"]
        for profile in registry["profiles"].values()
        if profile["cohort"] == "normal"
    }
    assert normal == set(g16c.PROFILE_ROLES)
    combined = [
        item
        for item in registry["comparisons"]
        if item["cohort"] == "normal" and item["route_id"] == "level_plus_wider_interaction"
    ]
    assert len(combined) == 3
    assert all(item["expected_controls_for_route"] == 3 for item in combined)


def test_target_contract_is_direction_neutral_and_multi_horizon() -> None:
    assert TARGET_COLUMNS == g16c.TARGETS
    assert {int(column.rsplit("h", 1)[1]) for column in TARGET_COLUMNS} == {1, 2, 4}
    assert not any("direction" in column for column in TARGET_COLUMNS)


def test_target_frame_requires_excursion_and_volume(monkeypatch) -> None:
    date = pd.Timestamp("2026-01-01", tz="UTC")
    anchors = pd.DataFrame(
        {
            "date": [date],
            "period": ["validation_early"],
            "pair": ["BTC/USDT:USDT"],
            "cohort": ["normal"],
            "market_group": ["btc"],
            "smart_contract_platform": [False],
            "anchor_level_family": ["generic_prior_range"],
            "anchor_level_name": ["prior_high"],
            "anchor_source_timeframe": ["1h"],
            "anchor_representation": ["settled"],
        }
    )
    source = pd.DataFrame(
        {
            "event_time": [date],
            "level_family": ["generic_prior_range"],
            "level_name": ["prior_high"],
            "source_timeframe": ["1h"],
            "representation": ["settled"],
            "control": ["actual"],
            "zone_half_width_atr": [0.2],
            **{
                f"volume_ratio_h{horizon}": [1.5 if horizon != 2 else 1.0]
                for horizon in g16c.HORIZONS
            },
            **{f"range_ratio_h{horizon}": [1.0] for horizon in g16c.HORIZONS},
            **{f"crossings_h{horizon}": [2.0] for horizon in g16c.HORIZONS},
            **{f"abs_excursion_atr_h{horizon}": [0.75] for horizon in g16c.HORIZONS},
        }
    )
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: source.copy())

    targets = g16c.target_frame(anchors, {"source_path": "unused"})

    assert targets["&-g16_reaction_h1"].iloc[0] == 1.0
    assert targets["&-g16_reaction_h2"].iloc[0] == 0.0


def test_exact_contact_with_missing_measurement_is_retained_but_unready(
    monkeypatch,
) -> None:
    date = pd.Timestamp("2026-01-01", tz="UTC")
    anchors = pd.DataFrame(
        {
            "date": [date],
            "anchor_level_family": ["generic_prior_range"],
            "anchor_level_name": ["prior_high"],
            "anchor_source_timeframe": ["1h"],
            "anchor_representation": ["settled"],
        }
    )
    source = pd.DataFrame(
        {
            "event_time": [date],
            "level_family": ["generic_prior_range"],
            "level_name": ["prior_high"],
            "source_timeframe": ["1h"],
            "representation": ["settled"],
            "control": ["actual"],
            "contact_volume_ratio": [float("nan")],
            "contact_range_ratio": [1.0],
            "contact_pressure_change": [0.1],
        }
    )
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: source.copy())

    merged = g16c.merge_contact_state(anchors, {"source_path": "unused", "pair": "P"})

    assert len(merged) == 1
    assert pd.isna(merged["contact_volume_ratio"].iloc[0])


def test_recent_period_assignment_uses_frozen_chronology() -> None:
    periods = g16f.g12z.RECENT_PERIODS
    date = pd.Timestamp(periods[0]["start"])
    frame = pd.DataFrame({"date": [date], "period": ["exposed_recent_diagnostic"]})
    manifest = {
        "evaluation_window": g13z.RECENT,
        "recent_periods": list(periods),
    }

    assigned = g16f.assign_evaluation_periods(frame, manifest)

    assert assigned["period"].iloc[0] == periods[0]["period"]

# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation17 as g17f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_freeze as g17z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_freqai_cache as g17c,
)
from user_data.strategies.MarketReactionZoneFreqAIGeneration17Strategy import (
    TARGET_COLUMNS,
)


def available_columns() -> list[str]:
    blocks = (
        g17c.MINIMAL,
        g17c.GEOMETRY,
        g17c.STALE_MINIMAL,
        g17c.STALE_GEOMETRY,
        g17c.SHUFFLED_MINIMAL,
        g17c.SHUFFLED_GEOMETRY,
        g17c.LOCAL_ACTIVITY,
        g17c.LOCAL_TREND,
        g17c.WIDER,
        g17c.EIGHT_HOUR_ACTIVITY,
        g17c.EIGHT_HOUR_TREND,
        g17c.ORDERBOOK,
        g17c.STALE_ORDERBOOK,
        g17c.SHUFFLED_ORDERBOOK,
    )
    columns = [f"{block}__value" for block in blocks]
    columns.extend(f"{g17c.MINIMAL}__{suffix}" for suffix in g17c.COUNT_SUFFIXES)
    columns.extend(f"{g17c.MINIMAL}__{suffix}" for suffix in g17c.PROXIMITY_SUFFIXES)
    return columns


def test_six_sibling_branches_and_profiles_are_frozen_together() -> None:
    branches = g17z.branch_definitions()
    assert len(branches) == 6
    assert len({branch["branch_id"] for branch in branches}) == 6
    frozen_profiles = {
        profile for branch in branches for profile in branch.get("profiles", [])
    }
    assert frozen_profiles == set(g17c.PROFILE_SPECS)


def test_registry_has_component_and_external_control_ladders() -> None:
    registry = g17c.build_registry(available_columns())
    normal_profiles = {
        profile["role"]
        for profile in registry["profiles"].values()
        if profile["cohort"] == "normal"
    }
    assert normal_profiles == set(g17c.PROFILE_SPECS)
    complete = [
        item
        for item in registry["comparisons"]
        if item["cohort"] == "normal"
        and item["route_id"] == "complete_reaction_context"
    ]
    orderbook = [
        item
        for item in registry["comparisons"]
        if item["cohort"] == "normal"
        and item["route_id"] == "orderbook_current_increment"
    ]
    assert len(complete) == 4
    assert len(orderbook) == 3
    assert all(item["expected_controls_for_route"] == 4 for item in complete)
    assert all(item["expected_controls_for_route"] == 3 for item in orderbook)


def test_target_contract_is_multihorizon_and_direction_neutral() -> None:
    assert TARGET_COLUMNS == g17c.TARGETS
    assert {int(column.rsplit("h", 1)[1]) for column in TARGET_COLUMNS} == {1, 2, 4}
    assert not any("direction" in column for column in TARGET_COLUMNS)


def test_target_frame_distinguishes_recross_traversal_and_reaction(monkeypatch) -> None:
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
            **{f"volume_ratio_h{h}": [1.5] for h in g17c.HORIZONS},
            **{f"crossings_h{h}": [2.0] for h in g17c.HORIZONS},
            **{f"abs_excursion_atr_h{h}": [0.8] for h in g17c.HORIZONS},
            **{f"away_excursion_atr_h{h}": [0.7] for h in g17c.HORIZONS},
            **{f"through_excursion_atr_h{h}": [0.6] for h in g17c.HORIZONS},
        }
    )
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: source.copy())

    targets = g17c.target_frame(anchors, {"source_path": "unused"})

    assert targets["&-g17_repeated_recross_h1"].iloc[0] == 1.0
    assert targets["&-g17_two_sided_traversal_h1"].iloc[0] == 1.0
    assert targets["&-g17_reaction_h1"].iloc[0] == 1.0


def test_recent_period_assignment_uses_frozen_windows() -> None:
    period = g17f.g12z.RECENT_PERIODS[0]
    frame = pd.DataFrame(
        {"date": [pd.Timestamp(period["start"])], "period": ["old"]}
    )
    manifest = {
        "evaluation_window": g17f.g13z.RECENT,
        "recent_periods": list(g17f.g12z.RECENT_PERIODS),
    }
    assigned = g17f.assign_evaluation_periods(frame, manifest)
    assert assigned["period"].iloc[0] == period["period"]

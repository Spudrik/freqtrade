# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation19_freqai_cache as g19c,
)


def available_columns() -> list[str]:
    return [
        *(f"{g19c.LOCAL_TREND}__value_{number}" for number in range(2)),
        *(f"{g19c.LEVEL}__value_{number}" for number in range(2)),
        *(f"{g19c.STALE_LEVEL}__value_{number}" for number in range(2)),
        *(f"{g19c.LOCAL_ACTIVITY}__value_{number}" for number in range(2)),
        g19c.DISPERSION_COLUMN,
    ]


def test_registry_matches_all_frozen_profiles_and_control_ladders() -> None:
    registry = g19c.build_registry(available_columns())
    normal_roles = {
        profile["role"]
        for profile in registry["profiles"].values()
        if profile["cohort"] == "normal"
    }
    assert normal_roles == set(g19c.PROFILE_SPECS)
    for route in ("level_plus_dispersion", "level_plus_local_activity"):
        comparisons = [
            item
            for item in registry["comparisons"]
            if item["cohort"] == "normal" and item["route_id"] == route
        ]
        assert len(comparisons) == 4
        assert all(item["expected_controls_for_route"] == 4 for item in comparisons)


def test_target_contract_is_multihorizon_and_direction_neutral() -> None:
    assert {int(column.rsplit("h", 1)[1]) for column in g19c.TARGETS} == {1, 2, 4, 8}
    assert not any("direction" in column or "profit" in column for column in g19c.TARGETS)


def test_target_frame_materializes_volume_and_crossing_counts(monkeypatch) -> None:
    date = pd.Timestamp("2026-01-01", tz="UTC")
    anchors = pd.DataFrame(
        {
            "date": [date],
            "period": ["validation_early"],
            "pair": ["BTC/USDT:USDT"],
            "cohort": ["normal"],
            "market_group": ["btc"],
            "smart_contract_platform": [False],
            "anchor_level_family": ["adaptive_volume_profile"],
            "anchor_level_name": ["nearest_hvn"],
            "anchor_source_timeframe": ["1h"],
            "anchor_representation": ["settled"],
        }
    )
    source = pd.DataFrame(
        {
            "event_time": [date],
            "level_family": ["adaptive_volume_profile"],
            "level_name": ["nearest_hvn"],
            "source_timeframe": ["1h"],
            "representation": ["settled"],
            "control": ["actual"],
            **{f"volume_ratio_h{horizon}": [1.5] for horizon in g19c.HORIZONS},
            **{f"crossings_h{horizon}": [2.0] for horizon in g19c.HORIZONS},
        }
    )

    def fake_read(path: str, *, columns: list[str]) -> pd.DataFrame:
        _ = path
        return (anchors if "date" in columns else source)[columns].copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read)
    result = g19c.target_frame(
        {
            "pair": "BTC/USDT:USDT",
            "parent_evaluation_path": "anchors",
            "source_path": "source",
        }
    )
    assert result["&-g19_future_volume_ratio_h8"].iloc[0] == 1.5
    assert result["&-g19_crossing_count_h8"].iloc[0] == 2.0

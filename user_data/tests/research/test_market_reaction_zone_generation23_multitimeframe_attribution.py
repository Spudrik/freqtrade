# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_multitimeframe_attribution as g23m,
)


def synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    dates = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
    base = pd.DataFrame(
        {
            "date": dates,
            "pre_close": [99.0, 100.0, 101.0],
            "close": [100.0, 101.0, 102.0],
            "base_atr": [2.0, 2.0, 2.0],
        }
    )
    actual = pd.DataFrame(
        {
            "cohort": ["normal"],
            "pair": ["TEST/USDT:USDT"],
            "period": ["g18_normal_holdout_early"],
            "control": ["actual"],
            "event_time": [dates[1]],
            "base_index": [1],
            "level_name": ["1h__family_a__high"],
            "level_price": [100.5],
            "zone_half_width": [0.5],
            "zone_half_width_atr": [0.25],
            "base_atr": [2.0],
            "pre_distance_atr": [0.25],
            "approach_state": ["already_inside_or_unclear"],
            "source_open": [dates[0]],
            "anchor_source_family": ["family_a"],
            "anchor_role": ["high"],
            "anchor_source_timeframe": ["1h"],
            "geometry_state": [g23m.GEOMETRY],
        }
    )
    surfaces: list[dict[str, object]] = []
    for timeframe, hours, level in (("1h", 1, 100.5), ("4h", 4, 100.8), ("1d", 24, 100.2)):
        surfaces.append(
            {
                "level_name": f"{timeframe}__family_{timeframe}__high",
                "family": f"family_{timeframe}",
                "role": "high",
                "source_timeframe": timeframe,
                "timeframe_hours": hours,
                "level": np.array([level, level, level]),
                "source_open": pd.Series(dates),
            }
        )
    surfaces[0]["level_name"] = "1h__family_a__high"
    return base, actual, surfaces


def test_same_time_attribution_emits_three_fixed_comparisons() -> None:
    base, actual, surfaces = synthetic_inputs()

    controls = g23m.same_time_component_controls(actual, base, surfaces)

    assert set(controls["control"]) == {
        "highest_timeframe_component",
        "nearest_non_anchor_component",
        "same_time_current_close_pseudo_cluster",
    }
    assert controls["event_time"].nunique() == 1


def test_attribution_registry_matches_frozen_route() -> None:
    frozen = g23m.load_freeze()

    assert frozen["status"] == "frozen_before_generation23_outcomes"
    assert len(g23m.CONTROLS) == 9
    assert g23m.ATTRIBUTION_COMPARISONS[0].startswith("matched_isolated")


def test_isolated_anchor_control_selects_a_matching_candidate() -> None:
    cluster = pd.DataFrame(
        {
            "event_time": pd.to_datetime(["2026-07-20T05:00:00Z"]),
            "period": ["early"],
            "anchor_source_family": ["volume_profile"],
            "anchor_role": ["poc"],
            "anchor_source_timeframe": ["1h"],
        }
    )
    isolated = pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                ["2026-07-20T01:00:00Z", "2026-07-20T02:00:00Z"]
            ),
            "period": ["early", "early"],
            "anchor_source_family": ["volume_profile", "volume_profile"],
            "anchor_role": ["poc", "poc"],
            "anchor_source_timeframe": ["1h", "1h"],
        }
    )

    result = g23m.matched_isolated_anchor_controls(
        cluster, isolated, "BTC/USDT:USDT"
    )

    assert len(result) == 1
    assert result.iloc[0]["control"] == "matched_isolated_anchor_component"

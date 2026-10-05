# ruff: noqa: S101

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation19_recross_analysis as g19r,
)


def test_crossing_count_uses_sign_changes_and_preserves_missing() -> None:
    close = np.array([[9.0, 11.0, 9.0], [11.0, 12.0, 13.0], [np.nan, 11.0, 9.0]])
    levels = np.array([10.0, 10.0, 10.0])
    previous = np.array([11.0, 9.0, 11.0])

    result = g19r.crossing_count(close, levels, previous)

    assert result[0] == 3.0
    assert result[1] == 1.0
    assert np.isnan(result[2])


def test_first_crossing_hour_distinguishes_none_from_missing() -> None:
    close = np.array([[9.0, 11.0], [11.0, 12.0], [np.nan, 9.0]])
    levels = np.array([10.0, 10.0, 10.0])
    previous = np.array([9.0, 11.0, 11.0])

    result = g19r.first_crossing_hour(close, levels, previous)

    assert result[0] == 2.0
    assert result[1] == 0.0
    assert np.isnan(result[2])


def test_coordinate_scopes_keep_families_and_frozen_exact_levels() -> None:
    frame = pd.DataFrame(
        {
            "cohort": ["normal"] * 3,
            "pair": ["BTC/USDT:USDT"] * 3,
            "g18_period": ["p1"] * 3,
            "control": ["actual"] * 3,
            "event_time": pd.to_datetime(["2026-08-01T00:00:00Z"] * 3),
            "level_family": [
                "adaptive_volume_profile_nodes",
                "adaptive_volume_profile_nodes",
                "rolling_vwap_deviation_bands",
            ],
            "level_name": [
                "vp_lb72_bins48_nearest_hvn_q80",
                "vp_lb168_bins48_poc",
                "rolling_vwap_lb72_upper_2sd",
            ],
            "pre_distance_atr": [0.2, 0.3, 0.1],
        }
    )

    result = g19r.coordinate_scopes(frame)

    assert set(result["scope_kind"]) == {"family_any_level", "single_level"}
    assert "vp_lb168_bins48_poc" not in set(
        result.loc[result["scope_kind"].eq("single_level"), "scope_value"]
    )

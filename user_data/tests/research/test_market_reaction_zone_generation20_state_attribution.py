# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation20_state_attribution as g20s,
)


def test_crossing_count_preserves_missing_paths() -> None:
    close = np.array([[9.0, 11.0, 9.0], [11.0, 12.0, 13.0], [np.nan, 9.0, 11.0]])
    levels = np.array([10.0, 10.0, 10.0])
    previous = np.array([11.0, 9.0, 11.0])

    result = g20s.g19_crossing_count(close, levels, previous)

    assert result[0] == 3.0
    assert result[1] == 1.0
    assert np.isnan(result[2])


def test_empirical_percentile_uses_only_supplied_calibration() -> None:
    values = pd.Series([0.0, 2.0, 4.0, np.nan])
    calibration = np.array([1.0, 2.0, 3.0])

    result = g20s.empirical_percentile(values, calibration)

    assert result[:3].tolist() == [0.0, 2.0 / 3.0, 1.0]
    assert np.isnan(result[3])


def test_donchian_control_registry_has_frozen_no_contact_baseline() -> None:
    assert g20s.DONCHIAN_CONTROLS[0] == (
        "breakout_state_matched_no_current_boundary_contact"
    )
    assert len(g20s.DONCHIAN_CONTROLS) == 4


def test_activity_states_are_named_and_outcome_free() -> None:
    dates = pd.date_range("2025-01-01", periods=600, freq="h", tz="UTC")
    state = pd.DataFrame(
        {
            "date": dates,
            "relative_volume": np.linspace(0.5, 2.0, len(dates)),
            "volume_acceleration": np.linspace(-1.0, 1.0, len(dates)),
            "prior_range_atr": np.linspace(0.2, 2.0, len(dates)),
            "atr_fraction": np.linspace(0.001, 0.05, len(dates)),
        }
    )

    result, calibration = g20s.add_activity_states(state, "normal")

    assert set(result["activity_state"]) <= {
        "quiet",
        "moderate",
        "extreme",
        "unavailable",
    }
    assert calibration["lower_tertile"] < calibration["upper_tertile"]
    assert "future" not in " ".join(result.columns).lower()

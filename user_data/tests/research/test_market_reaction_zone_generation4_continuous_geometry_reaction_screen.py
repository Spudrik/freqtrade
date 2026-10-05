from __future__ import annotations

# ruff: noqa: S101
import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation4_continuous_geometry_reaction_screen as g4c_screen,
)


def test_g4c_monotonic_orientation_requires_ordered_middle_band() -> None:
    assert g4c_screen.monotonic_orientation(1.0, 2.0, 3.0) == "increasing"
    assert g4c_screen.monotonic_orientation(3.0, 2.0, 1.0) == "decreasing"
    assert g4c_screen.monotonic_orientation(1.0, 3.0, 2.0) == "not_monotonic"
    assert g4c_screen.monotonic_orientation(1.0, 1.0, 1.0) == "not_monotonic"


def test_g4c_reconstructed_outcomes_start_after_contact() -> None:
    close = np.full(30, 100.0)
    close[1:25:2] = 100.2
    close[2:25:2] = 99.8
    base = pd.DataFrame({"close": close})
    base.loc[26:29, "close"] = [101.0, 102.0, 103.0, 104.0]
    events = pd.DataFrame(
        {
            "base_index": [25],
            "level_price": [100.0],
            "zone_half_width": [0.5],
        }
    )

    realized, leave = g4c_screen.reconstructed_outcomes(events, base=base)

    assert np.isfinite(realized[0])
    assert leave[0] == 1.0


def test_g4c_reconstructed_leave_time_is_censored_after_four_hours() -> None:
    base = pd.DataFrame({"close": np.linspace(99.9, 100.1, 40)})
    events = pd.DataFrame(
        {
            "base_index": [25],
            "level_price": [100.0],
            "zone_half_width": [1.0],
        }
    )

    _, leave = g4c_screen.reconstructed_outcomes(events, base=base)

    assert leave[0] == 5.0

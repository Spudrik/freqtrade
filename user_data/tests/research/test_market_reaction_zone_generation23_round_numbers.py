# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_round_numbers as g23r,
)


def synthetic_base() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC"),
            "pre_close": [np.nan, 61_234.0, 61_680.0, 62_050.0],
        }
    )


def test_round_grid_uses_previous_close_decimal_scale() -> None:
    surfaces = g23r.grid_surfaces(synthetic_base(), "BTC/USDT:USDT", "actual")

    assert len(surfaces) == 3
    assert np.isnan(surfaces[0]["level"][0])
    assert surfaces[0]["level"][1] == 61_000.0
    assert surfaces[1]["level"][1] == 61_000.0
    assert surfaces[2]["level"][1] == 61_250.0


def test_round_grid_controls_change_location_without_future_inputs() -> None:
    base = synthetic_base()
    actual = g23r.grid_surfaces(base, "BTC/USDT:USDT", "actual")
    random_grid = g23r.grid_surfaces(
        base, "BTC/USDT:USDT", "random_mantissa_grid"
    )
    shifted = g23r.grid_surfaces(
        base, "BTC/USDT:USDT", "half_step_phase_shifted_grid"
    )

    assert not np.allclose(actual[0]["level"][1:], random_grid[0]["level"][1:])
    assert not np.allclose(actual[0]["level"][1:], shifted[0]["level"][1:])


def test_round_number_registry_matches_frozen_route() -> None:
    frozen = g23r.load_freeze()

    assert frozen["status"] == "frozen_before_generation23_outcomes"
    assert g23r.CONTROLS[-1] == "stale_72h"

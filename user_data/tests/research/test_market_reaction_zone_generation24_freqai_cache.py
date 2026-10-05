# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_freqai_cache as g24c,
)


def feature_frame(rows: int = 500) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=rows, freq="1h", tz="UTC"),
            **{
                column: np.arange(rows, dtype=float) + number
                for number, column in enumerate(g24c.FEATURE_COLUMNS)
            },
        }
    )
    frame[f"ready__{g24c.READY_BLOCK}"] = True
    return frame


def test_registry_has_every_model_seed_control_profile() -> None:
    registry = g24c.build_registry()

    assert len(registry["profiles"]) == 2 * 4 * 5
    assert len(registry["comparisons"]) == 2 * 4 * 6
    assert len(g24c.TARGETS) == 8


def test_permuted_level_block_uses_only_earlier_rows() -> None:
    pair = "TEST/USDT:USDT"
    frame = feature_frame()
    shifted = g24c.past_shift_level_block(frame, pair)
    lag = g24c.permuted_level_lag(pair)

    for column in g24c.LEVEL_FEATURES:
        np.testing.assert_allclose(
            shifted[column].iloc[lag:].to_numpy(),
            frame[column].iloc[:-lag].to_numpy(),
        )
        assert shifted[column].iloc[:lag].isna().all()


def test_freqai_route_matches_frozen_batch() -> None:
    frozen, branch = g24c.load_branch()

    assert frozen["status"] == "frozen_before_generation24_outcomes"
    assert branch["target_transform"] == "pair_training_empirical_percentile"
    assert set(g24c.CONTROLS) == set(branch["controls"])

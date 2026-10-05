# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_freqai_cache as g23c,
)
from user_data.strategies.MarketReactionZoneFreqAIGeneration23Strategy import (
    TARGET_COLUMNS,
)


def test_registry_freezes_both_transforms_and_all_five_controls() -> None:
    registry = g23c.build_registry()

    assert len(g23c.FEATURE_COLUMNS) == 12
    assert len(g23c.RAW_TARGETS) == 20
    assert len(g23c.TARGETS) == 40
    assert tuple(TARGET_COLUMNS) == g23c.TARGETS
    assert len(registry["profiles"]) == 8
    assert {
        item["control_type"]
        for item in registry["comparisons"]
        if item["cohort"] == "normal"
    } == {
        "constant_training_median",
        "level_geometry_only_model",
        "market_state_only_model",
        "within_pair_time_shuffled_training_labels",
        "generation22_uncalibrated_parent",
    }
    assert not any("profit" in target or "direction" in target for target in g23c.TARGETS)


def test_empirical_percentile_uses_midrank_for_ties_and_bounded_tails() -> None:
    values = pd.Series([-1.0, 1.0, 2.0, 4.0, np.nan])
    reference = pd.Series([1.0, 2.0, 2.0, 3.0])

    result = g23c.empirical_percentile(values, reference)

    assert result.iloc[0] == 0.0
    assert result.iloc[1] == 0.125
    assert result.iloc[2] == 0.5
    assert result.iloc[3] == 1.0
    assert np.isnan(result.iloc[4])


def test_target_calibration_reads_only_pre_evaluation_training_rows() -> None:
    start, cutoff = g23c.calibration_bounds("normal")
    dates = pd.DatetimeIndex(
        [
            start,
            start + pd.Timedelta(hours=1),
            cutoff - pd.Timedelta(hours=1),
            cutoff,
            cutoff + pd.Timedelta(hours=1),
        ]
    )
    frame = pd.DataFrame(
        {
            "date": dates,
            "period": ["training", "training", "training", "early", "early"],
            f"ready__{g23c.READY_BLOCK}": True,
        }
    )
    for target in g23c.RAW_TARGETS:
        frame[target] = [1.0, 2.0, 3.0, 1000.0, 2000.0]

    transformed, audit = g23c.transform_targets(
        frame, frame, cohort="normal", pair="BTC/USDT:USDT"
    )
    raw_target = g23c.RAW_TARGETS[0]
    residual = g23c.target_name("pair_training_median_residual", raw_target)
    percentile = g23c.target_name("pair_training_empirical_percentile", raw_target)

    assert audit[0]["calibration_rows"] == 3
    assert audit[0]["training_median"] == 2.0
    assert transformed.loc[3, residual] == 998.0
    assert transformed.loc[3, percentile] == 1.0

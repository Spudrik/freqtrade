from __future__ import annotations

# ruff: noqa: S101
import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation4_continuous_geometry_controls as controls,
)


def test_g4c_control_band_comparisons_preserve_adjacent_monotonic_steps() -> None:
    assert controls.band_comparisons("increasing") == (
        ("low_to_middle", "middle", "low"),
        ("middle_to_high", "high", "middle"),
    )
    assert controls.band_comparisons("decreasing") == (
        ("high_to_middle", "middle", "high"),
        ("middle_to_low", "low", "middle"),
    )


def test_g4c_matched_permutation_is_deterministic_for_repeated_positive_effect() -> None:
    frame = pd.DataFrame(
        {
            "pair": np.repeat([f"COIN{index}" for index in range(6)], 20),
            "oriented_delta": np.ones(120),
        }
    )

    first = controls.matched_permutation_test(frame, permutations=255, seed_key="fixed")
    second = controls.matched_permutation_test(frame, permutations=255, seed_key="fixed")

    assert first == second
    assert first[0] == 1.0
    assert first[1] <= controls.PERMUTATION_ALPHA


def test_g4c_attach_control_state_uses_event_base_index() -> None:
    base = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
            "pre_close": [99.0, 100.0, 101.0, 102.0],
            "base_atr": [1.0, 1.0, 2.0, 2.0],
        }
    )
    state = pd.DataFrame({"date": base["date"]})
    for offset, column in enumerate(controls.STATE_CONTROL_FEATURES):
        state[column] = np.arange(4, dtype=float) + offset
    events = pd.DataFrame(
        {
            "base_index": [1, 3],
            "event_time": base.loc[[1, 3], "date"].reset_index(drop=True),
            "level_price": [101.0, 106.0],
        }
    )

    result = controls.attach_control_state(events, base=base, state=state)

    assert result[controls.STATE_CONTROL_FEATURES[0]].tolist() == [1.0, 3.0]
    assert result["pre_distance_atr"].tolist() == [1.0, 2.0]


def test_g4c_no_level_control_requires_both_adjusted_adjacent_bands() -> None:
    rows: list[dict[str, object]] = []
    for band, adjusted in (("middle", 2.0), ("low", 1.0)):
        for index in range(60):
            row: dict[str, object] = {
                "feature_band": band,
                "pair": f"COIN{index % 6}",
                "actual_minus_no_level": adjusted,
            }
            for offset, column in enumerate(controls.STATE_CONTROL_FEATURES):
                value = float((index + offset) % 17)
                row[f"actual_state__{column}"] = value
                row[f"no_level_state__{column}"] = value
            rows.append(row)
    candidate = pd.Series(
        {
            "analysis_scope": "normal_alts",
            "feature": "example_feature",
            "outcome": "example_outcome",
            "early_orientation": "increasing",
        }
    )

    result = controls.no_level_control_cell(
        pd.DataFrame(rows),
        candidate=candidate,
        period="validation_early",
        comparison="low_to_middle",
        stronger_band="middle",
        weaker_band="low",
    )

    assert result["support_pass"] is True
    assert result["control_pass"] is True
    assert result["equal_coin_adjusted_contrast_median"] == 1.0

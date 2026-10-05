# ruff: noqa: S101

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation19_acceptance_zone as g19a,
)


def test_nearest_zone_selects_closest_valid_candidate() -> None:
    zones = g19a.g2d.DensityZoneSet(
        family="repeated_close_density",
        history_hours=168,
        bin_width_atr=0.25,
        levels=np.array([[90.0, 101.0, 110.0], [90.0, 100.0, 110.0]]),
        half_widths=np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
        valid=np.array([[True, True, True], [False, False, False]]),
        support_counts=np.array([[5.0, 10.0, 4.0], [1.0, 1.0, 1.0]]),
        source_counts=np.array([168.0, 168.0]),
    )

    level, width, support, rank = g19a.nearest_zone(zones, np.array([100.0, 100.0]))

    assert level[0] == 101.0
    assert width[0] == 2.0
    assert support[0] == 10.0
    assert rank[0] == 1.0
    assert np.isnan(level[1])


def test_reaction_columns_separate_fresh_arrival_from_occupancy() -> None:
    frame = pd.DataFrame(
        {
            "approach_state": ["from_above", "already_inside_or_unclear"],
            **{
                f"crossings_h{horizon}": [1.0, 2.0]
                for horizon in g19a.HORIZONS
            },
            **{
                f"dwell_fraction_h{horizon}": [0.2, 0.8]
                for horizon in g19a.HORIZONS
            },
        }
    )

    g19a.add_reaction_columns(frame)

    assert frame["arrival_class"].tolist() == [
        "fresh_arrival",
        "already_inside_or_unclear",
    ]
    assert frame["metric__any_recross_h8"].tolist() == [1.0, 1.0]
    assert frame["metric__repeated_recross_h8"].tolist() == [0.0, 1.0]


def test_research_manifest_uses_two_later_blocks() -> None:
    for cohort in ("normal", "meme"):
        periods = g19a.research_manifest(cohort)["data"]["chronological_periods"]
        assert len(periods) == 2
        assert periods[0]["end_utc_exclusive"] <= periods[1]["start_utc"]


def test_width_and_coverage_thresholds_are_frozen_before_results() -> None:
    assert g19a.MIN_COVERAGE_RATIO == 0.50
    assert g19a.MAX_MEAN_WIDTH_RATIO == 1.25

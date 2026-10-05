from __future__ import annotations

# ruff: noqa: S101
import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation4_continuous_geometry_preflight as g4c_preflight,
)


def test_g4c_interval_coverage_uses_union_not_double_counting() -> None:
    result = g4c_preflight.interval_coverage_fraction(
        [(1.0, 3.0), (2.0, 4.0), (6.0, 7.0)],
        lower=0.0,
        upper=8.0,
    )

    assert result == 0.5


def test_g4c_tied_development_edges_remain_unsupported() -> None:
    values = pd.Series([0.0, 0.0, 1.0])

    result = g4c_preflight.assign_frozen_band(
        values,
        low_upper=0.0,
        middle_upper=0.0,
        distinct=False,
    )

    assert result.isna().all()


def test_g4c_no_ahead_level_is_always_in_high_room_band() -> None:
    values = pd.Series([np.nan, -0.5, 0.5, 2.0])
    no_ahead = pd.Series([True, False, False, False])

    result = g4c_preflight.assign_frozen_band(
        values,
        low_upper=0.0,
        middle_upper=1.0,
        distinct=True,
        no_ahead=no_ahead,
        missing_policy="no_ahead_level_is_high",
    )

    assert result.tolist() == ["high", "low", "middle", "high"]


def test_g4c_group_counts_deduplicate_named_levels_in_one_mechanism() -> None:
    result = g4c_preflight.count_role_groups_within(
        {
            ("price_average", "activity_transit"): 0.2,
            ("rolling_extreme", "activity_transit"): 0.8,
            ("acceptance", "acceptance_stickiness"): 0.4,
        },
        role="activity_transit",
        threshold=1.0,
    )

    assert result == 2

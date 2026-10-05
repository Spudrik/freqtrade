# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation6 as g6f,
)


def test_independent_event_dates_prevents_overlapping_four_hour_targets() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01 00:00Z",
                    "2026-01-01 04:00Z",
                    "2026-01-01 05:00Z",
                    "2026-01-01 06:00Z",
                ]
            ),
            "period": ["a", "a", "a", "b"],
        }
    )

    result = g6f.independent_event_dates(frame)

    assert result["date"].tolist() == pd.to_datetime(
        ["2026-01-01 00:00Z", "2026-01-01 05:00Z", "2026-01-01 06:00Z"]
    ).tolist()


def test_stale_placebo_uses_only_same_period_sources_at_least_72h_old() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01 00:00Z",
                    "2026-01-04 00:00Z",
                    "2026-01-05 00:00Z",
                    "2026-01-10 00:00Z",
                ]
            ),
            "period": ["a", "a", "b", "b"],
            "source__value": [1.0, 2.0, 3.0, 4.0],
        }
    )

    result = g6f.attach_stale_placebo(frame, block="source")

    assert result.loc[1, "source_placebo__value"] == 1.0
    assert np.isnan(result.loc[2, "source_placebo__value"])
    assert result.loc[3, "source_placebo__value"] == 3.0
    assert (
        result.loc[result["source_date"].notna(), "source_date"]
        <= result.loc[result["source_date"].notna(), "date"] - pd.Timedelta(hours=72)
    ).all()


def test_every_source_ladder_has_seven_profiles_on_one_common_mask() -> None:
    for source in g6f.SOURCE_BLOCKS:
        profiles = {
            profile_id: definition
            for profile_id, definition in g6f.PROFILES.items()
            if profile_id.startswith(f"{source}__")
        }
        readiness = {
            tuple(definition["required_ready_blocks"])
            for definition in profiles.values()
        }

        assert len(profiles) == 7
        assert len(readiness) == 1
        assert {
            "state",
            "level",
            "level_placebo",
            source,
            f"{source}_placebo",
        } == set(next(iter(readiness)))


def test_regression_row_reports_plain_paired_error_gain() -> None:
    frame = pd.DataFrame(
        {
            "target__actual": np.arange(40, dtype=float),
            "target__candidate": np.arange(40, dtype=float) + 0.5,
            "target__baseline": np.arange(40, dtype=float) + 2.0,
        }
    )

    result = g6f.regression_row(
        frame,
        target="target",
        candidate_column="target__candidate",
        baseline_column="target__baseline",
    )

    assert np.isclose(result["candidate_mae"], 0.5)
    assert np.isclose(result["baseline_mae"], 2.0)
    assert np.isclose(result["paired_mae_gain"], 1.5)


def test_target_values_removes_pressure_direction() -> None:
    frame = pd.DataFrame(
        {
            "volume_ratio_h1": [2.0],
            "range_ratio_h1": [3.0],
            "abs_excursion_atr_h4": [4.0],
            "pressure_change_h1": [-5.0],
            "dwell_fraction_h4": [0.25],
        }
    )

    result = g6f.target_values(frame)

    assert result[g6f.TARGET_COLUMNS[3]].iloc[0] == 5.0
    assert set(result) == set(g6f.TARGET_COLUMNS)

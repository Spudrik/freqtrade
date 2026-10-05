# ruff: noqa: S101

from __future__ import annotations

import pandas as pd
from pandas import DataFrame, Series

from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_layer2_direct as direct,
)


def test_forward_windows_start_at_the_anchor() -> None:
    values = Series([1.0, 3.0, 2.0, 6.0, 4.0])
    assert direct.future_max(values, 3).tolist()[:3] == [3.0, 6.0, 6.0]
    assert direct.future_min(values, 3).tolist()[:3] == [1.0, 2.0, 2.0]
    assert direct.future_sum(values, 3).tolist()[:3] == [6.0, 11.0, 12.0]


def test_boundary_reaction_requires_inward_move_before_outward_break() -> None:
    inward = DataFrame(
        {
            "high": [10.0, 10.1, 9.9],
            "low": [9.8, 9.3, 9.0],
        }
    )
    success, reason = direct.boundary_reaction(inward, lower=5.0, upper=10.0, atr=1.0)
    assert success is True
    assert reason == "upper_inward_half_atr_first"

    outward = DataFrame(
        {
            "high": [10.6, 10.7, 10.8],
            "low": [9.9, 9.4, 9.0],
        }
    )
    success, reason = direct.boundary_reaction(outward, lower=5.0, upper=10.0, atr=1.0)
    assert success is False
    assert reason == "upper_outward_first_or_same_candle"


def test_local_context_marks_warmup_rows_as_insufficient() -> None:
    frame = DataFrame(
        {
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.0] * 10,
            "volume": [1.0] * 10,
        }
    )
    context = direct.local_context(frame, Series([1.0] * 10))
    assert set(context["local_level_state"]) == {"insufficient_level_history"}


def test_activity_summary_requires_cross_period_and_leave_year_support() -> None:
    records = []
    for year, partition in (
        (2021, "development_2021_2023"),
        (2022, "development_2021_2023"),
        (2024, "internal_validation_2024_2025"),
        (2025, "internal_validation_2024_2025"),
    ):
        for number in range(3):
            records.append(
                {
                    "event_source": "official_fomc",
                    "scope": "btc",
                    "horizon_hours": 4,
                    "control_type": direct.PRIMARY_CONTROL,
                    "paired_success": number < 2,
                        "paired_uplift": 1.0 if number < 2 else -0.5,
                        "activity_score": 2.0,
                        "control_activity_score": 1.0,
                        "abs_return_ratio": 2.0,
                        "control_abs_return_ratio": 1.0,
                        "range_ratio": 2.0,
                        "control_range_ratio": 1.0,
                        "volume_ratio": 2.0,
                        "control_volume_ratio": 1.0,
                    "unusual_activity": number == 0,
                    "whole_event_partition": partition,
                    "event_year": year,
                }
            )
    summary = direct.summarize_activity(DataFrame.from_records(records))
    assert len(summary) == 1
    assert bool(summary.iloc[0]["meets_55_floor"])


def test_paired_activity_keeps_the_real_control_type() -> None:
    common = {
        "event_id": "event_a",
        "event_source": "official_fomc",
        "scope": "btc",
        "horizon_hours": 4,
        "whole_event_partition": "development_2021_2023",
        "sample_anchor_utc": pd.Timestamp("2022-01-01T19:00:00Z"),
        "abs_return_ratio": 1.0,
        "range_ratio": 1.0,
        "volume_ratio": 1.0,
        "signed_return": 0.01,
        "unusual_activity": False,
        "asset_count": 1,
    }
    scopes = DataFrame.from_records(
        [
            {
                **common,
                "sample_id": "event",
                "sample_type": "event",
                "control_type": "event",
                "control_rank": 0,
                "activity_score": 2.0,
            },
            {
                **common,
                "sample_id": "control",
                "sample_type": "control",
                "control_type": direct.PRIMARY_CONTROL,
                "control_rank": 1,
                "activity_score": 1.0,
            },
        ]
    )
    paired = direct.paired_activity_rows(scopes)
    assert paired["control_type"].tolist() == [direct.PRIMARY_CONTROL]
    assert paired["paired_success"].tolist() == [True]


def test_background_contrast_requires_repetition_and_better_than_controls() -> None:
    rows = []
    for source_type, control_type, negative_fades, positive_fades in (
        ("event", "event", 8, 2),
        ("control", direct.PRIMARY_CONTROL, 6, 5),
    ):
        for partition in (
            "development_2021_2023",
            "internal_validation_2024_2025",
        ):
            for background, fades in (
                ("negative_30d_background", negative_fades),
                ("positive_30d_background", positive_fades),
            ):
                for number in range(10):
                    rows.append(
                        {
                            "event_source": "official_fomc",
                            "sample_type": source_type,
                            "control_type": control_type,
                            "whole_event_partition": partition,
                            "background": background,
                            "positive_initial_move": True,
                            "positive_move_faded_by_half": number < fades,
                        }
                    )
    contrast = direct.summarize_background_contrasts(DataFrame.from_records(rows))
    assert len(contrast) == 1
    assert bool(contrast.iloc[0]["meets_55_floor_and_control"])


def test_sample_table_keeps_event_and_control_rows_distinct() -> None:
    events = DataFrame(
        {
            "event_id": ["event_a"],
            "event_source": ["official_fomc"],
            "event_family": ["scheduled"],
            "anchor_utc": pd.to_datetime(["2024-01-01T19:00:00Z"]),
            "whole_event_partition": ["internal_validation_2024_2025"],
        }
    )
    controls = DataFrame(
        {
            "event_id": ["event_a"],
            "event_source": ["official_fomc"],
            "event_anchor_utc": pd.to_datetime(["2024-01-01T19:00:00Z"]),
            "control_type": ["matched_prior_state"],
            "control_rank": [1],
            "control_anchor_utc": pd.to_datetime(["2023-12-04T19:00:00Z"]),
        }
    )
    samples = direct.sample_table(events, controls)
    assert len(samples) == 2
    assert samples["sample_id"].is_unique
    assert set(samples["sample_type"]) == {"event", "control"}

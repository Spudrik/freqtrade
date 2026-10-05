from __future__ import annotations

# ruff: noqa: S101
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation4_fresh_arrival_support as g4b_support,
)


def support_rows(
    *, family: str, names: tuple[str, ...], periods: tuple[str, ...]
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check_family": family,
                "check_name": name,
                "period": period,
                "support_passed": True,
            }
            for name in names
            for period in periods
        ]
    )


def test_g4b_metadata_reader_never_requests_reaction_values() -> None:
    forbidden = ("outcome", "actual__", "control__", "delta__")

    assert not any(
        column == prefix or column.startswith(prefix)
        for column in g4b_support.METADATA_COLUMNS
        for prefix in forbidden
    )


def test_g4b_family_support_requires_every_validation_period() -> None:
    periods = ("validation_early", "validation_late")
    frame = support_rows(
        family="matched_event_state",
        names=g4b_support.MATCHED_STATE_CONTROLS,
        periods=periods,
    )
    frame.loc[
        frame["check_name"].eq("event_state__near_miss")
        & frame["period"].eq("validation_late"),
        "support_passed",
    ] = False

    result = g4b_support.family_passed(
        frame,
        family="matched_event_state",
        required=g4b_support.MATCHED_STATE_CONTROLS,
        validation_periods=periods,
    )

    assert result["event_state__near_miss"] is False
    assert result["event_state__already_inside"] is True
    assert result["event_state__repeat_contact"] is True


def test_g4b_any_control_group_cannot_stitch_different_controls_across_periods() -> None:
    periods = ("validation_early", "validation_late")
    frame = support_rows(
        family="location_control",
        names=("current_vp_hvn", "current_vp_lvn"),
        periods=periods,
    )
    frame.loc[
        frame["check_name"].eq("current_vp_hvn")
        & frame["period"].eq("validation_late"),
        "support_passed",
    ] = False
    frame.loc[
        frame["check_name"].eq("current_vp_lvn")
        & frame["period"].eq("validation_early"),
        "support_passed",
    ] = False

    result = g4b_support.control_group_passed(
        frame,
        group={
            "mode": "any_same_control",
            "controls": ("current_vp_hvn", "current_vp_lvn"),
        },
        validation_periods=periods,
    )

    assert result is False

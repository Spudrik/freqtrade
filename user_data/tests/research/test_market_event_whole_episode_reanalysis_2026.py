# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_event_whole_episode_reanalysis_2026 as episode,
)


def test_json_list_requires_a_list() -> None:
    assert episode._json_list('["event_1", "event_2"]') == ["event_1", "event_2"]


def test_combined_2026_keeps_original_periods() -> None:
    frame = pd.DataFrame(
        {
            "model_period": [
                "walk_forward_validation_2025",
                "untouched_confirmation_2026_jan_apr",
            ],
            "value": [1, 2],
        }
    )
    combined = episode._with_combined_2026(frame)
    assert combined["model_period"].tolist() == [
        "walk_forward_validation_2025",
        "untouched_confirmation_2026_jan_apr",
        "confirmation_2026",
    ]


def test_role_corrections_do_not_call_volume_the_driver() -> None:
    corrections = {item["finding"]: item for item in episode.ROLE_CORRECTIONS}
    assert (
        corrections["recent_volume_persistence"]["correct_role"]
        == "background_or_readiness"
    )
    assert (
        corrections["catalogued_event_clock"]["correct_role"]
        == "possible_driver_or_timing_anchor"
    )


def test_actual_episode_uses_only_earliest_anchor() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["later", "early", "control"],
            "sample_kind": ["actual_event", "actual_event", "matched_control"],
            "analysis_unit_id": ["episode_1", "episode_1", "control_1"],
            "model_period": ["period", "period", "period"],
            "horizon_hours": [1, 1, 1],
            "model_anchor_utc": pd.to_datetime(
                ["2026-01-01 02:00Z", "2026-01-01 01:00Z", "2026-01-02 01:00Z"]
            ),
        }
    )
    selected = episode._earliest_actual_anchor_per_episode(frame)
    assert set(selected["sample_id"]) == {"early", "control"}

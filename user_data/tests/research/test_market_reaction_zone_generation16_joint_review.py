# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation16_joint_review as g16j,
)


def evidence_rows(
    *,
    route: str,
    scope: str,
    comparisons: tuple[str, ...],
    windows: tuple[str, ...],
    point: bool = True,
    strict: bool = True,
    sign: str = "positive",
) -> list[dict[str, object]]:
    return [
        {
            "route_id": route,
            "scope_value": scope,
            "metric": "crossings",
            "horizon_hours": 1,
            "market_scope": "all_normal",
            "window": window,
            "comparison": comparison,
            "effect_sign": sign,
            "point_pass": point,
            "strict_pass": strict,
            "minimum_effect": 0.1,
            "maximum_effect": 0.2,
        }
        for comparison in comparisons
        for window in windows
    ]


def test_whole_question_requires_every_control_in_both_normal_windows() -> None:
    rows = evidence_rows(
        route="completed_contact_activity_control",
        scope="all_retained_levels",
        comparisons=("genuine_near_miss", "matched_ordinary_time"),
        windows=(g16j.STANDARD, g16j.RECENT),
    )
    complete = g16j.collapse_direct(pd.DataFrame(rows))
    assert complete["status"].iloc[0] == "strict_repeated_lead"

    missing = pd.DataFrame(rows[:-1])
    rejected = g16j.collapse_direct(missing)
    assert rejected["status"].iloc[0] == "not_retained_after_whole_question_gate"


def test_current_density_is_rejected_when_placebo_repeats_same_effect() -> None:
    windows = (g16j.STANDARD, g16j.RECENT)
    rows = evidence_rows(
        route="level_density_and_overlap",
        scope="current_density",
        comparisons=("dense_vs_isolated",),
        windows=windows,
    )
    rows.extend(
        evidence_rows(
            route="level_density_and_overlap",
            scope="stale_density_placebo",
            comparisons=("dense_vs_isolated",),
            windows=windows,
        )
    )
    rows.extend(
        evidence_rows(
            route="level_density_and_overlap",
            scope="shuffled_density_placebo",
            comparisons=("dense_vs_isolated",),
            windows=windows,
            point=False,
            strict=False,
        )
    )

    decision = g16j.collapse_direct(pd.DataFrame(rows))

    assert decision["status"].iloc[0] == "not_retained_after_whole_question_gate"
    assert decision["placebo_contamination"].iloc[0] == "stale_density_placebo"


def test_branch_queue_preserves_breadth_and_bounded_direction_lane() -> None:
    queue = g16j.branch_queue()
    assert len(queue) >= 5
    assert {item["branch_batch"] for item in queue}.issuperset(
        {
            "g17a_density_geometry_decomposition",
            "g17d_broad_level_sources",
            "g17e_external_context_regimes",
            "g17f_bounded_one_minute_expansion",
        }
    )
    direction = next(
        item for item in queue if item["branch_batch"] == "g17f_bounded_one_minute_expansion"
    )
    assert "55%" in direction["gate"]

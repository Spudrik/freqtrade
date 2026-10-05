# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_joint_review as g17j,
)


def test_next_queue_keeps_multiple_siblings_and_parks_failed_direction() -> None:
    freqai = pd.DataFrame(
        {
            "strict_all_three_cells": [True],
            "route_id": ["density_counts_increment"],
        }
    )
    semantics = pd.DataFrame({"strict_repeated": [True]})
    atlas = pd.DataFrame({"strict_repeated": [False]})
    minute = pd.DataFrame(
        {"broad_point_lead": [False], "cohort_specific_point_lead": [False]}
    )
    queue = g17j.next_branch_queue(freqai, semantics, atlas, minute)
    assert len(queue) == 6
    assert queue[0]["status"] == "queued"
    assert queue[2]["status"] == "parked_no_strict_parent"
    assert queue[-1]["status"] == "parked_current_rules_failed"


def test_friendly_target_expands_machine_label() -> None:
    assert g17j.friendly_target("&-g17_repeated_recross_h2") == (
        "repeated recross over 2 hours"
    )

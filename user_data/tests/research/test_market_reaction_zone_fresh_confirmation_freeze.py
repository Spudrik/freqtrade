# ruff: noqa: S101

from __future__ import annotations

import pandas as pd
from pandas import DataFrame, Series

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_fresh_confirmation_freeze as fresh,
)


def test_fresh_period_boundaries_are_half_open() -> None:
    dates = Series(
        pd.to_datetime(
            [
                "2026-08-19T23:00:00Z",
                "2026-08-20T00:00:00Z",
                "2026-09-04T23:00:00Z",
                "2026-09-05T00:00:00Z",
                "2026-09-19T23:00:00Z",
                "2026-09-20T00:00:00Z",
            ],
            utc=True,
        )
    )
    assert fresh.assign_fresh_period(dates).tolist() == [
        "outside_fresh_confirmation",
        "fresh_early",
        "fresh_early",
        "fresh_late",
        "fresh_late",
        "outside_fresh_confirmation",
    ]


def test_required_data_bounds_include_training_warmup_and_outcome_tail() -> None:
    normal_start, normal_end = fresh.required_data_bounds("normal")
    meme_start, meme_end = fresh.required_data_bounds("meme")
    assert normal_start == pd.Timestamp("2025-07-21T00:00:00Z")
    assert meme_start == pd.Timestamp("2026-02-11T00:00:00Z")
    assert normal_end == pd.Timestamp("2026-09-20T07:00:00Z")
    assert meme_end == normal_end


def test_parent_evidence_freezes_three_exact_questions() -> None:
    document = fresh.build_freeze_document()
    assert document["status"] == "frozen_before_fresh_confirmation_outcomes"
    assert document["all_three_questions_frozen_together"] is True
    assert len(document["siblings"]) == 3

    state, convergence, vwap = document["siblings"]
    assert state["id"] == "fresh_market_state_activity"
    assert len(state["retained_scope_target_questions"]) == 15
    assert len(state["profiles"]) == 16
    assert convergence["exact_question"] == {
        "scope_value": "all_60_frozen_definitions",
        "metric": "crossing_count",
        "horizon_hours": 2,
        "market_scope": "top_ten_memes",
    }
    assert vwap["exact_question"]["scope_value"].endswith("centre_0.0")
    assert vwap["exact_question"]["market_scope"] == "all_normal"
    assert document["future_reaction_outcomes_read"] is False


def test_coverage_summary_requires_every_cohort_pair_cell() -> None:
    rows = []
    for cohort in ("normal", "meme"):
        for number in range(10):
            rows.append(
                {
                    "cohort": cohort,
                    "pair": f"{cohort}-{number}",
                    "exists": True,
                    "available_end_utc": "2026-09-20T07:00:00Z",
                    "duplicate_timestamps": 0,
                    "hourly_gaps": 0,
                    "pair_coverage_pass": True,
                }
            )
    inventory = DataFrame.from_records(rows)
    assert fresh.coverage_summary(inventory)["all_pair_coverage_pass"] is True

    inventory.loc[0, "hourly_gaps"] = 1
    inventory.loc[0, "pair_coverage_pass"] = False
    summary = fresh.coverage_summary(inventory)
    assert summary["all_pair_coverage_pass"] is False
    assert summary["hourly_gaps"] == 1

# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation5_external_context_preflight as g5e,
)


def test_global_independence_removes_overlapping_market_paths() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    frame = pd.DataFrame(
        {
            "date": [start, start + pd.Timedelta(hours=2), start + pd.Timedelta(hours=5)],
            "period": ["validation_early"] * 3,
            "pair": ["A", "B", "C"],
            "selection_hash": ["a", "b", "c"],
        }
    )

    selected = g5e.select_global_independent(frame, hours=4)

    assert selected["pair"].tolist() == ["A", "C"]


def test_overlap_gate_keeps_missing_rows_unavailable() -> None:
    frame = pd.DataFrame(
        {
            "period": ["validation_early"] * 60,
            "pair": [f"COIN{i % 6}" for i in range(60)],
            "cohort": ["normal"] * 60,
        }
    )
    usable = pd.Series([True] * 49 + [False] * 11)

    rows = g5e.summarize_overlap(
        frame,
        usable=usable,
        surface_id="surface",
        context_block="block",
        validation_periods=("validation_early",),
        source_note="test",
    )

    assert rows[0]["timestamp_safe_ready_events"] == 49
    assert rows[0]["period_gate_passed"] is False
    assert rows[0]["reaction_outcomes_opened"] is False


def test_block_decision_requires_both_periods() -> None:
    overlap = pd.DataFrame(
        {
            "surface_id": ["surface", "surface"],
            "cohort": ["normal", "normal"],
            "context_block": ["gdelt", "gdelt"],
            "period": ["validation_early", "validation_late"],
            "timestamp_safe_ready_events": [55, 49],
            "ready_coins": [6, 6],
            "period_gate_passed": [True, False],
        }
    )

    decision = g5e.block_decisions(overlap).iloc[0]

    assert bool(decision["passed_both_periods"]) is False
    assert decision["classification"] == "park_without_outcomes_insufficient_common_support"


def test_block_decision_opens_only_fair_cell() -> None:
    overlap = pd.DataFrame(
        {
            "surface_id": ["surface", "surface"],
            "cohort": ["normal", "normal"],
            "context_block": ["global", "global"],
            "period": ["validation_early", "validation_late"],
            "timestamp_safe_ready_events": [80, 70],
            "ready_coins": [8, 7],
            "period_gate_passed": [True, True],
        }
    )

    decision = g5e.block_decisions(overlap).iloc[0]

    assert bool(decision["passed_both_periods"]) is True
    assert decision["classification"] == "open_low_dimensional_conditioning_outcomes"

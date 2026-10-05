# ruff: noqa: S101

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from user_data.Custom_Launcher.research.context_features import (
    market_event_regional_nominal_clock_direct as direct,
)


def test_control_map_rejects_collision_reached_by_shifted_clock() -> None:
    anchor = pd.Timestamp("2024-06-20 01:30:00Z")
    week_one = anchor - pd.Timedelta(weeks=1)
    catalogue = pd.DataFrame(
        {
            "event_id": ["event_one"],
            "event_family": ["china_nbs_price_release"],
            "anchor_utc": [anchor],
        }
    )
    external = pd.DataFrame(
        {
            "anchor_utc": [week_one + pd.Timedelta(hours=4, minutes=20)],
            "source_group": ["external"],
        }
    )

    controls = direct.build_control_map(catalogue, external)

    assert len(controls) == direct.CONTROL_COUNT
    assert controls.iloc[0]["weeks_before_event"] == 2
    assert not controls["control_anchor_utc"].eq(week_one).any()


def test_candidate_anchor_offsets_use_exact_candle_start_minutes() -> None:
    anchor = pd.Timestamp("2024-06-20 01:30:00Z")
    catalogue = pd.DataFrame(
        {
            "event_id": ["event_one"],
            "event_family": ["china_nbs_price_release"],
            "whole_event_partition": ["internal_validation_2024_2025"],
            "anchor_utc": [anchor],
            "other_frozen_event_within_4h": [False],
        }
    )
    controls = pd.DataFrame(
        {
            "event_id": ["event_one"] * direct.CONTROL_COUNT,
            "control_rank": list(range(1, direct.CONTROL_COUNT + 1)),
            "control_anchor_utc": [
                anchor - pd.Timedelta(weeks=week)
                for week in range(1, direct.CONTROL_COUNT + 1)
            ],
        }
    )

    rows = direct.candidate_anchor_rows(catalogue, controls)
    event_rows = rows.loc[rows["candidate_slot"].eq(0)].sort_values("offset_minutes")

    assert event_rows["offset_minutes"].tolist() == [-30, 0, 30]
    assert event_rows["analysis_anchor_utc"].tolist() == [
        anchor - pd.Timedelta(minutes=30),
        anchor,
        anchor + pd.Timedelta(minutes=30),
    ]


def test_every_candidate_is_scored_symmetrically_against_other_twelve() -> None:
    metrics = np.ones((direct.CANDIDATE_COUNT, len(direct.METRIC_COLUMNS)))
    metrics[0] = 2.0

    scores, ratios = direct.symmetric_candidate_scores(metrics)

    assert scores[0] == pytest.approx(2.0)
    assert ratios[0].tolist() == pytest.approx([2.0, 2.0, 2.0])
    assert scores[1] == pytest.approx(1.0)


def test_zero_end_to_end_movement_does_not_discard_range_and_volume() -> None:
    metrics = np.ones((direct.CANDIDATE_COUNT, len(direct.METRIC_COLUMNS)))
    metrics[0, 0] = 0.0

    scores, ratios = direct.symmetric_candidate_scores(metrics)

    assert scores[0] == pytest.approx(1.0)
    assert ratios[0].tolist() == pytest.approx([0.0, 1.0, 1.0])
    assert np.isfinite(scores).all()


def test_linked_choice_is_shared_across_all_rows_of_one_event() -> None:
    matrix = np.vstack(
        [
            np.arange(direct.CANDIDATE_COUNT),
            np.arange(direct.CANDIDATE_COUNT) + 100,
            np.arange(direct.CANDIDATE_COUNT) + 200,
        ]
    )
    codes = np.array([0, 0, 1])
    choices = np.array([2, 4])

    selected = direct.select_permuted_scores(matrix, codes, choices)

    assert selected.tolist() == [2, 102, 204]


def test_familywide_decision_uses_global_route_maximum_distribution() -> None:
    routes = pd.DataFrame(
        {
            "all_direct_gates_pass": [True, False],
            "observed_route_strength": [3.0, 4.0],
            "shifted_only_timing_mismatch": [False, False],
        }
    )
    null = pd.DataFrame({"global_max_route_strength": np.ones(99)})

    decisions = direct.add_familywide_decisions(routes, null)

    assert decisions.iloc[0]["familywide_probability"] == pytest.approx(0.01)
    assert bool(decisions.iloc[0]["familywide_gate_pass"])
    assert pd.isna(decisions.iloc[1]["familywide_probability"])


def test_artifact_hash_validation_detects_changed_file(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text("frozen", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {"path": str(path), "sha256": digest}

    assert direct.validate_artifact_record(record, label="test") == path
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        direct.validate_artifact_record(record, label="test")


def test_execute_refuses_partial_outputs_without_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_names = (
        "CONTROL_PATH",
        "EXTERNAL_PATH",
        "COVERAGE_PATH",
        "WINDOW_COVERAGE_PATH",
        "METRICS_PATH",
        "SCORES_PATH",
        "SUMMARY_PATH",
        "ROUTES_PATH",
        "NULL_PATH",
        "REPORT_PATH",
        "RESULT_PATH",
    )
    for name in output_names:
        monkeypatch.setattr(direct, name, tmp_path / f"{name.casefold()}.tmp")
    direct.CONTROL_PATH.write_text("partial", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Partial regional activity outputs"):
        direct.execute()

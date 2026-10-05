# ruff: noqa: S101

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame


MODULE = (
    "user_data.Custom_Launcher.research.context_features."
    "market_event_eu_council_sanctions_activity_direct"
)
eu = importlib.import_module(MODULE)


def synthetic_members() -> DataFrame:
    return DataFrame(
        {
            "event_id": ["member_a", "member_b"],
            "episode_id": ["episode_a", "episode_a"],
            "member_clock": pd.to_datetime(
                ["2023-07-01T10:00:00Z", "2023-07-01T12:30:00Z"], utc=True
            ),
            "analysis_partition": ["development_2023"] * 2,
        }
    )


def empty_external() -> DataFrame:
    return DataFrame(
        {
            "anchor_utc": pd.Series([], dtype="datetime64[ns, UTC]"),
            "source_group": pd.Series([], dtype="object"),
        }
    )


def minute_frame(*, periods: int = 1600, gap_at: int | None = None) -> DataFrame:
    dates = pd.date_range("2023-01-01T00:00:00Z", periods=periods, freq="1min")
    if gap_at is not None:
        dates = dates.delete(gap_at)
    rows = len(dates)
    return DataFrame(
        {
            "date": dates,
            "open": np.full(rows, 100.0),
            "high": np.full(rows, 101.0),
            "low": np.full(rows, 99.0),
            "close": np.full(rows, 100.0),
            "volume": np.arange(1, rows + 1, dtype=float),
        }
    )


def test_contract_hash_and_source_hash_rejection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(eu, "CONTRACT_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="contract changed"):
        eu.load_contract_and_source()
    corrupt = tmp_path / "source.json"
    corrupt.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        eu.validate_artifact_record(
            {"path": str(corrupt), "sha256": "0" * 64}, label="source"
        )


def test_frozen_source_counts_and_manual_exclusion() -> None:
    contract, selected, scored = eu.load_contract_and_source()
    assert contract["status"] == (
        "frozen_outcome_blind_eu_sanctions_activity_analysis_v2_after_source_only_preflight"
    )
    assert contract["supersession"]["v1_contract"]["sha256"] == (
        "c361f6f57b942d9de6b68979f21d13a20e23dc781804e8e836e9a3b4bab0d182"
    )
    assert contract["supersession"]["v1_source_only_preflight"][
        "episodes_with_incomplete_controls"
    ] == 28
    assert contract["pre_run_assertions"]["scored_whole_episodes"] == 67
    assert len(selected) == 98
    assert scored["episode_id"].nunique() == 67
    assert scored.groupby("analysis_partition")["episode_id"].nunique().to_dict() == {
        "development_2023": 22,
        "holdout_2025": 20,
        "internal_validation_2024": 25,
    }
    assert scored["selection_status"].eq("include").all()
    assert not scored["event_id"].isin(
        ["eu_sanctions_20230123_2a4f003e49e6", "eu_sanctions_20240629_18655a89ac99"]
    ).any()


def test_v2_flags_supersession_and_assertions_are_hard_gates() -> None:
    contract = json.loads(eu.CONTRACT_PATH.read_text(encoding="utf-8"))
    eu._validate_contract(contract)
    prohibited = copy.deepcopy(contract)
    prohibited["profit_used"] = True
    with pytest.raises(ValueError, match="prohibited outcome"):
        eu._validate_contract(prohibited)
    changed = copy.deepcopy(contract)
    changed["supersession"]["v1_source_only_preflight"][
        "episodes_with_12_controls"
    ] = 40
    with pytest.raises(ValueError, match="preflight lineage changed"):
        eu._validate_contract(changed)
    changed = copy.deepcopy(contract)
    changed["pre_run_assertions"]["asset_horizon_routes"] = 7
    with pytest.raises(ValueError, match="pre-run assertions changed"):
        eu._validate_contract(changed)


def test_whole_episode_controls_preserve_shift_order_year_weekday_and_spacing() -> None:
    members = synthetic_members()
    controls = eu.build_control_map(members, members)
    assert controls["shift_weeks"].tolist() == [-1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6]
    candidates = eu.candidate_member_rows(members, controls)
    original_spacing = pd.Timedelta(hours=2, minutes=30)
    for _, group in candidates.groupby("candidate_slot"):
        clocks = group.sort_values("member_order")["candidate_member_clock"]
        assert clocks.iloc[1] - clocks.iloc[0] == original_spacing
        assert clocks.dt.year.eq(2023).all()
        assert clocks.dt.weekday.eq(5).all()
    assert eu.candidate_structure_violations(candidates) == {
        "same_year_violations": 0,
        "same_weekday_violations": 0,
        "internal_member_spacing_violations": 0,
    }


def test_incomplete_controls_fail_early() -> None:
    members = synthetic_members()
    blocked = []
    for shift in eu._shift_order():
        blocked.extend(members["member_clock"] + pd.Timedelta(weeks=shift))
    selected = pd.concat(
        [
            members,
            DataFrame(
                {
                    "event_id": [f"block_{index}" for index in range(len(blocked))],
                    "episode_id": [f"block_episode_{index}" for index in range(len(blocked))],
                    "member_clock": blocked,
                    "analysis_partition": ["development_2023"] * len(blocked),
                }
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="only 0 source-clean controls"):
        eu.build_control_map(members, selected)


def test_external_anchors_are_diagnostic_and_never_reject_controls() -> None:
    members = synthetic_members()
    controls = eu.build_control_map(members, members)
    candidates = eu.candidate_member_rows(members, controls)
    external = DataFrame(
        {
            "anchor_utc": candidates["candidate_member_clock"],
            "source_group": ["external"] * len(candidates),
        }
    )
    assert len(eu.build_control_map(members, members)) == 12
    diagnosed = eu.add_external_context_diagnostics(candidates, members, external)
    context = diagnosed.drop_duplicates(["episode_id", "candidate_slot"])
    assert context["external_anchor_count_within_4h_of_any_member"].gt(0).all()
    assert not context["descriptive_collision_clean"].any()


def test_multi_member_metrics_use_candidate_median() -> None:
    rows = []
    for slot in range(eu.CANDIDATE_COUNT):
        for member_order, multiplier in enumerate((1.0, 3.0), start=1):
            value = float(slot + 1) * multiplier
            rows.append(
                {
                    "episode_id": "episode_a",
                    "analysis_partition": "development_2023",
                    "candidate_slot": slot,
                    "candidate_kind": "event" if slot == 0 else "control",
                    "shift_weeks": slot,
                    "pair": eu.ASSETS[0],
                    "horizon_minutes": 60,
                    "offset_minutes": 0,
                    "member_count": 2,
                    "member_order": member_order,
                    "window_status": "usable",
                    "absolute_end_return": value / 100,
                    "full_high_low_range": value / 50,
                    "total_volume": value * 100,
                    "prior_60m_total_volume": value,
                    "prior_240m_absolute_end_return": value / 200,
                    "prior_240m_full_high_low_range": value / 150,
                    "external_anchor_count_within_4h_of_any_member": 0,
                    "external_source_group_count_within_4h_of_any_member": 0,
                    "external_anchor_count_within_26h_of_any_member": 0,
                    "external_source_group_count_within_26h_of_any_member": 0,
                    "descriptive_collision_clean": True,
                }
            )
    aggregated = eu.aggregate_episode_metrics(DataFrame.from_records(rows))
    event = aggregated.loc[aggregated["candidate_slot"].eq(0)].iloc[0]
    assert event["absolute_end_return"] == pytest.approx(0.02)
    assert event["full_high_low_range"] == pytest.approx(0.04)
    assert event["total_volume"] == pytest.approx(200.0)
    assert np.isfinite(event["activity_score"])


def test_complete_gap_and_zero_return_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eu, "OFFSETS", (0,))
    monkeypatch.setattr(eu, "HORIZONS", (3,))
    frame = minute_frame(periods=400)
    anchor = frame.iloc[300]["date"]
    candidate = DataFrame(
        {
            "episode_id": ["episode_a"],
            "analysis_partition": ["development_2023"],
            "candidate_slot": [0],
            "candidate_kind": ["event"],
            "shift_weeks": [0],
            "member_order": [1],
            "member_count": [1],
            "source_event_id": ["member_a"],
            "source_member_clock": [anchor],
            "candidate_member_clock": [anchor],
        }
    )
    complete = eu.extract_member_metrics(candidate, {eu.ASSETS[0]: frame})
    assert complete.iloc[0]["window_status"] == "usable"
    assert complete.iloc[0]["absolute_end_return"] == 0.0
    assert complete.iloc[0]["full_high_low_range"] > 0
    assert complete.iloc[0]["total_volume"] > 0

    gapped = minute_frame(periods=400, gap_at=301)
    result = eu.extract_member_metrics(candidate, {eu.ASSETS[0]: gapped})
    assert result.iloc[0]["window_status"] == "timestamp_gap"


def test_trailing_diagnostics_never_read_anchor_or_future() -> None:
    frame = minute_frame(periods=500)
    baseline = eu._trailing_metrics(frame, 300)
    changed = frame.copy()
    changed.loc[300:, ["open", "high", "low", "close", "volume"]] = 1_000_000.0
    assert eu._trailing_metrics(changed, 300) == baseline


def test_symmetric_scores_keep_zero_return_and_candidate_linkage() -> None:
    values = np.column_stack(
        [
            np.arange(eu.CANDIDATE_COUNT, dtype=float),
            np.arange(1, eu.CANDIDATE_COUNT + 1, dtype=float),
            np.arange(10, 10 + eu.CANDIDATE_COUNT, dtype=float),
        ]
    )
    scores, ratios = eu.regional.symmetric_candidate_scores(values)
    assert np.isfinite(scores).all()
    assert ratios[0, 0] == 0.0
    matrix = np.tile(np.arange(eu.CANDIDATE_COUNT, dtype=float), (4, 1))
    codes = np.array([0, 0, 1, 1])
    selected = eu.regional.select_permuted_scores(matrix, codes, np.array([3, 8]))
    assert selected.tolist() == [3.0, 3.0, 8.0, 8.0]


def score_rows() -> DataFrame:
    rows = []
    for partition in eu.PARTITIONS:
        for episode in range(12):
            for offset in eu.OFFSETS:
                row = {
                    "episode_id": f"{partition}_{episode}",
                    "analysis_partition": partition,
                    "pair": eu.ASSETS[0],
                    "horizon_minutes": 60,
                    "offset_minutes": offset,
                }
                row.update({column: 1.3 + episode / 100 for column in eu.SCORE_COLUMNS})
                rows.append(row)
    return DataFrame.from_records(rows)


def test_only_natural_background_support_is_scored() -> None:
    cells = eu.summarize_cells(score_rows())
    assert set(cells["view"]) == {"natural_background_all_episodes"}
    assert set(cells["whole_episodes"]) == {12}


def passing_cells() -> DataFrame:
    rows = []
    for partition in eu.PARTITIONS:
        for view in eu.VIEWS:
            for offset in eu.OFFSETS:
                rows.append(
                    {
                        "pair": eu.ASSETS[0],
                        "horizon_minutes": 60,
                        "offset_minutes": offset,
                        "analysis_partition": partition,
                        "view": view,
                        "whole_episodes": 12,
                        "minimum_whole_episodes": 10,
                        "median_activity_score": 1.30,
                        "above_control_rate": 0.70,
                        "studentized_log_activity": 2.0 - abs(offset) / 1000,
                        "primary_cell_pass": offset == 0,
                        "adjacent_cell_pass": offset in (-60, 60),
                    }
                )
    return DataFrame.from_records(rows)


def test_fixed_primary_adjacent_pair_must_pass_every_partition() -> None:
    cells = passing_cells()
    route = eu.observed_route_decisions(cells).iloc[0]
    assert route["all_direct_gates_pass"]
    assert route["chosen_primary_offset_minutes"] == 0
    broken = cells.copy()
    mask = (
        broken["offset_minutes"].eq(60)
        & broken["analysis_partition"].eq("holdout_2025")
        & broken["view"].eq("natural_background_all_episodes")
    )
    broken.loc[mask, "adjacent_cell_pass"] = False
    broken.loc[
        broken["offset_minutes"].eq(-60), "adjacent_cell_pass"
    ] = False
    assert not eu.observed_route_decisions(broken).iloc[0]["all_direct_gates_pass"]


def test_familywide_best_pair_search_uses_all_six_routes() -> None:
    scores = pd.concat(
        [
            score_rows().assign(pair=pair, horizon_minutes=horizon)
            for pair in eu.ASSETS
            for horizon in eu.HORIZONS
        ],
        ignore_index=True,
    )
    specs = eu.build_route_specs(scores)
    assert len(specs) == 6
    selected = scores[eu.SCORE_COLUMNS[0]].to_numpy(dtype=float)
    strengths = [eu.route_strength_from_scores(selected, spec) for spec in specs]
    assert len(strengths) == 6
    assert np.isfinite(strengths).all()


def test_existing_result_hash_validation_and_no_prohibited_claims(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected_paths = {
        key: tmp_path / f"{key}.csv" for key in eu.EXPECTED_RESULT_ARTIFACT_PATHS
    }
    for path in expected_paths.values():
        path.write_text(path.name, encoding="utf-8")
    script = tmp_path / "script.py"
    contract = tmp_path / "contract.json"
    external = tmp_path / "external.json"
    market = tmp_path / "market.feather"
    for path in (script, contract, external, market):
        path.write_text(path.name, encoding="utf-8")
    monkeypatch.setattr(eu, "EXPECTED_RESULT_ARTIFACT_PATHS", expected_paths)
    monkeypatch.setattr(eu, "ANALYSIS_PATH", script)
    monkeypatch.setattr(eu, "CONTRACT_PATH", contract)
    result = {
        "status": "completed_eu_sanctions_unsigned_activity_analysis",
        "market_outcomes_opened_before_contract": False,
        "profit_used": False,
        "direction_tested": False,
        "causal_claim_permitted": False,
        "artifacts": {key: eu.artifact(path) for key, path in expected_paths.items()},
        "analysis_script": eu.artifact(script),
        "analysis_contract": eu.artifact(contract),
        "external_source_artifacts": [
            {"label": "external", **eu.artifact(external)}
        ],
        "market_source_artifacts": [{"pair": "BTC", **eu.artifact(market)}],
    }
    eu.validate_existing_result(result)
    next(iter(expected_paths.values())).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        eu.validate_existing_result(result)
    result["profit_used"] = True
    with pytest.raises(ValueError, match="prohibited claim"):
        eu.validate_existing_result(result)


def test_existing_result_rejects_missing_extra_and_true_precontract_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected_paths = {
        key: tmp_path / f"{key}.csv" for key in eu.EXPECTED_RESULT_ARTIFACT_PATHS
    }
    for path in expected_paths.values():
        path.write_text(path.name, encoding="utf-8")
    script = tmp_path / "script.py"
    contract = tmp_path / "contract.json"
    script.write_text("script", encoding="utf-8")
    contract.write_text("contract", encoding="utf-8")
    monkeypatch.setattr(eu, "EXPECTED_RESULT_ARTIFACT_PATHS", expected_paths)
    monkeypatch.setattr(eu, "ANALYSIS_PATH", script)
    monkeypatch.setattr(eu, "CONTRACT_PATH", contract)
    result = {
        "status": "completed_eu_sanctions_unsigned_activity_analysis",
        "market_outcomes_opened_before_contract": False,
        "profit_used": False,
        "direction_tested": False,
        "causal_claim_permitted": False,
        "artifacts": {key: eu.artifact(path) for key, path in expected_paths.items()},
        "analysis_script": eu.artifact(script),
        "analysis_contract": eu.artifact(contract),
        "external_source_artifacts": [],
        "market_source_artifacts": [],
    }
    missing = copy.deepcopy(result)
    missing["artifacts"].pop(next(iter(expected_paths)))
    with pytest.raises(ValueError, match="artifact key set changed"):
        eu.validate_existing_result(missing)
    extra = copy.deepcopy(result)
    extra["artifacts"]["unexpected"] = eu.artifact(script)
    with pytest.raises(ValueError, match="artifact key set changed"):
        eu.validate_existing_result(extra)
    opened = copy.deepcopy(result)
    opened["market_outcomes_opened_before_contract"] = True
    with pytest.raises(ValueError, match="outcome-blind contract flag"):
        eu.validate_existing_result(opened)


def test_execute_refuses_partial_outputs_before_loading_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    partial = tmp_path / "partial.csv"
    result_path = tmp_path / "result.json"
    partial.write_text("partial", encoding="utf-8")
    monkeypatch.setattr(eu, "RESULT_PATH", result_path)
    monkeypatch.setattr(eu, "PLANNED_OUTPUT_PATHS", (partial, result_path))

    def source_loader_must_not_run() -> None:
        raise AssertionError("source loader ran before partial-output refusal")

    monkeypatch.setattr(eu, "load_contract_and_source", source_loader_must_not_run)
    with pytest.raises(FileExistsError, match="refusing to mix artifacts"):
        eu.execute()


def test_outputs_and_report_keep_frozen_interpretation_boundaries() -> None:
    assert len(eu.PLANNED_OUTPUT_PATHS) == 8
    assert len(set(eu.PLANNED_OUTPUT_PATHS)) == 8
    metric_names = set(eu.METRIC_COLUMNS) | {
        "prior_60m_total_volume",
        "prior_240m_absolute_end_return",
        "prior_240m_full_high_low_range",
    }
    assert "profit" not in metric_names
    assert "direction" not in metric_names
    report = eu.render_report(
        DataFrame(
            {
                "retained_unsigned_activity_association": [False],
            }
        ),
        DataFrame(
            {
                "prior_60m_total_volume": [1.0],
                "window_status": ["usable"],
            }
        ),
        synthetic_members(),
        DataFrame({"candidate_slot": range(12)}),
        DataFrame(
            {
                "episode_id": ["episode_a", "episode_a"],
                "candidate_slot": [0, 1],
                "candidate_kind": ["event", "control"],
                "external_anchor_count_within_26h_of_any_member": [1, 0],
            }
        ),
    )
    assert "## Driver / event clock" in report
    assert "## Pre-event readiness diagnostics" in report
    assert "## Collisions / confluence" in report
    assert "## Later activity outcome" in report
    assert "did not select controls or replace the event" in report
    assert "not source-specific causation" in report

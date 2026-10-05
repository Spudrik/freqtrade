from __future__ import annotations

# ruff: noqa: S101 - pytest assertions are the test oracle.
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "Custom_Launcher/research/context_features"
    / "market_signal_portfolio_stage1_preflight.py"
)
SPEC = importlib.util.spec_from_file_location("stage1_preflight", SCRIPT)
assert SPEC and SPEC.loader
stage1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage1)


def _coverage_frame() -> pd.DataFrame:
    rows = []
    for prototype in stage1.PROTOTYPES:
        rows.append(
            stage1._row(
                prototype["prototype_id"],
                source_sibling="fixture",
                cohort="fixture",
                pair="fixture",
                horizon="1h",
                independent_episode_count=1,
                actual_count=1,
                eligible=True,
            )
        )
    return pd.DataFrame(rows, columns=stage1.COVERAGE_COLUMNS)


def test_registry_has_exact_five_families_and_nine_prototypes() -> None:
    assert len(stage1.PROTOTYPES) == 9
    assert len({item["family_id"] for item in stage1.PROTOTYPES}) == 5
    assert [item["prototype_id"] for item in stage1.PROTOTYPES].count(
        "live_news_web_activity_warning"
    ) == 1
    assert [item["prototype_id"] for item in stage1.PROTOTYPES].count(
        "local_multitimeframe_support_resistance_map"
    ) == 1


def test_forbidden_outcome_projection_is_rejected_before_file_access() -> None:
    with pytest.raises(ValueError, match="Forbidden outcome columns"):
        stage1.read_parquet_projected(Path("not-opened.parquet"), ["event_id", "response_return"])
    with pytest.raises(ValueError, match="Forbidden outcome columns"):
        stage1.read_parquet_projected(Path("not-opened.parquet"), ["event_time", "volume_ratio_h1"])
    with pytest.raises(ValueError, match="Forbidden outcome columns"):
        stage1.read_parquet_projected(
            Path("not-opened.parquet"), ["event_time", "invented_future_return"]
        )


def test_hash_drift_fails_loudly(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash drift"):
        stage1._verify_contract("fixture", {"path": str(source), "sha256": "0" * 64})


def test_g17_inventory_contract_uses_source_path_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "pair.parquet"
    source.write_bytes(b"fixture")
    seen: list[dict[str, str]] = []

    def capture(_contracts, _name, item):
        seen.append(item)
        raise RuntimeError("stop after inventory normalization")

    monkeypatch.setattr(stage1, "_verified_child", capture)
    with pytest.raises(RuntimeError, match="inventory normalization"):
        stage1.build_level_coverage(
            [
                {
                    "pair": "BTC/USDT:USDT",
                    "source_path": str(source),
                    "source_sha256": "a" * 64,
                }
            ],
            [],
        )
    assert seen == [{"path": str(source), "sha256": "a" * 64}]


def test_future_source_timestamp_is_rejected() -> None:
    frame = pd.DataFrame(
        {
            "available_at": ["2026-01-01T01:00:00Z"],
            "decision_at": ["2026-01-01T00:00:00Z"],
        }
    )
    with pytest.raises(ValueError, match="future timestamp"):
        stage1.assert_no_future_source(frame, "available_at", "decision_at", "fixture")


def test_mixed_iso_timestamp_precision_is_supported_without_guessing() -> None:
    frame = pd.DataFrame(
        {
            "available_at": [
                "2026-01-01 00:00:00+00:00",
                "2026-01-02 00:00:00.123456+00:00",
            ],
            "decision_at": [
                "2026-01-01 00:00:00+00:00",
                "2026-01-02 00:01:00+00:00",
            ],
        }
    )
    stage1.assert_no_future_source(frame, "available_at", "decision_at", "mixed precision fixture")


def test_conflicting_independent_episode_duplicates_are_rejected() -> None:
    frame = pd.DataFrame(
        {
            "episode": ["one", "one"],
            "decision": ["2026-01-01", "2026-01-02"],
        }
    )
    with pytest.raises(ValueError, match="conflicting duplicate"):
        stage1.deduplicate_independent(frame, ["episode"], ["decision"], "fixture")


def test_cpi_counts_whole_episodes_and_preserves_partitions() -> None:
    rows = []
    for episode, partition, direction in (("a", "development", 1), ("b", "validation", -1)):
        for pair in ("BTC/USDT:USDT", "ETH/USDT:USDT"):
            for horizon in (5, 15, 30, 60):
                rows.append(
                    {
                        "route_id": "cpi_headline_core_agreement",
                        "expectation_episode_id": episode,
                        "source_block": partition,
                        "anchor_utc": "2026-01-01T13:30:00Z",
                        "pair": pair,
                        "horizon_minutes": horizon,
                        "predicted_direction": direction,
                        "abstention_reason": "",
                        "ordinary_control_count": 12,
                    }
                )
    coverage = stage1.build_cpi_coverage(pd.DataFrame(rows))
    assert len(coverage) == 8
    assert {row["independent_episode_count"] for row in coverage} == {2}
    assert all(
        json.loads(row["partition_counts_json"]) == {"development": 1, "validation": 1}
        for row in coverage
    )
    assert all(row["positive_sign_support"] == 1 for row in coverage)
    assert all(row["negative_sign_support"] == 1 for row in coverage)


def test_source_siblings_are_not_pooled_under_live_news_web_prototype() -> None:
    events = []
    controls = []
    for family in ("live_news_activity_spike", "live_web_activity_spike", "btc_dominance_change"):
        for index, partition in enumerate(("development", "validation")):
            event_id = f"{family}-{index}"
            events.append(
                {
                    "family": family,
                    "available_at": f"2026-01-0{index + 1}T00:00:00Z",
                    "anchor_utc": f"2026-01-0{index + 1}T00:00:00Z",
                    "whole_event_partition": partition,
                    "source_direction": 1,
                    "predicted_direction": 1 if index == 0 else -1,
                    "is_event": True,
                    "event_id": event_id,
                    "coverage_ready": True,
                }
            )
            controls.append({"family": family, "event_id": event_id})
    verdicts = pd.DataFrame(
        [
            {
                "family": "live_news_activity_spike",
                "verdict": "repeatable_activity_only_lead",
                "activity_cells": "btc:1h; eth:4h",
            },
            {
                "family": "live_web_activity_spike",
                "verdict": "repeatable_activity_only_lead",
                "activity_cells": "top_ten_memes:1h; btc:24h",
            },
            {
                "family": "btc_dominance_change",
                "verdict": "repeatable_direction_only_lead",
                "activity_cells": "",
            },
        ]
    )
    coverage = stage1.build_independent_context_coverage(
        pd.DataFrame(events), pd.DataFrame(controls), verdicts
    )
    live = [row for row in coverage if row["prototype_id"] == "live_news_web_activity_warning"]
    assert {row["source_sibling"] for row in live} == {
        "live_news_activity_spike",
        "live_web_activity_spike",
    }
    assert all(row["horizon"] != "24h" for row in live)


def test_transmission_waits_for_confirmation_and_deduplicates_horizons() -> None:
    rows = []
    for cohort in ("established_alts", "memes"):
        for horizon in (1, 3):
            rows.append(
                {
                    "sample_kind": "actual_event",
                    "episode_id": "episode-a",
                    "model_anchor_utc": "2026-01-01T00:00:00Z",
                    "model_period": "validation",
                    "btc_confirmed_reaction": True,
                    "horizon_hours": horizon,
                    "cohort": cohort,
                }
            )
    coverage = stage1.build_transmission_coverage(pd.DataFrame(rows))
    assert len(coverage) == 2
    assert all(row["independent_episode_count"] == 1 for row in coverage)
    assert all("one_hour_after_parent_anchor" in row["decision_time_rule"] for row in coverage)


def test_recent_volume_rejects_future_contributing_data() -> None:
    calls = pd.DataFrame(
        [
            {
                "sample_kind": "actual_event",
                "analysis_unit_id": "a",
                "model_anchor_utc": "2026-01-01T00:00:00Z",
                "model_period": "validation",
                "pair": "BTC/USDT:USDT",
                "signal_id": "recent_volume_persistence",
                "observation_utc": "2026-01-01T00:00:00Z",
                "latest_contributing_data_utc": "2026-01-01T01:00:00Z",
            }
        ]
    )
    with pytest.raises(ValueError, match="future contributing"):
        stage1.build_recent_volume_coverage(calls)


def test_coverage_validation_rejects_family_pooling_or_duplicate_identity() -> None:
    coverage = _coverage_frame()
    stage1.validate_coverage(coverage)
    pooled = coverage.loc[~coverage["prototype_id"].eq("btc_dominance_relative_meme_rotation")]
    with pytest.raises(ValueError, match="exact frozen prototype registry"):
        stage1.validate_coverage(pooled)
    duplicate = pd.concat([coverage, coverage.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate coverage"):
        stage1.validate_coverage(duplicate)


def test_local_map_uses_frozen_sources_and_separates_decision_types() -> None:
    contracts: list[dict[str, object]] = []
    sources = stage1._load_level_map_contracts(contracts)
    rows = stage1.build_local_map_coverage(sources)
    assert rows
    assert {row["episode_type"] for row in rows} == {
        "source_event",
        "downstream_confirmation",
    }
    assert all(
        row["materialization_status"] == "coverage_ready_for_materialization" for row in rows
    )
    generic = [
        row for row in rows if row["source_family"] == "generic_ma_bollinger_negative_control"
    ]
    assert generic
    assert all(row["intended_level_role"] == "negative_control" for row in generic)
    assert all(row["above_price_resistance"] == "control_only_not_promoted" for row in generic)


def test_local_map_preserves_source_specific_prior_evidence_and_timeframes() -> None:
    contracts: list[dict[str, object]] = []
    sources = stage1._load_level_map_contracts(contracts)
    rows = stage1.build_local_map_coverage(sources)
    by_family: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_family.setdefault(str(row["source_family"]), []).append(row)

    expected = {
        "adaptive_volume_profile_nodes": (
            "retained_crossing_and_traffic_not_direction",
            "retained crossing and traffic evidence",
        ),
        "donchian_boundaries": (
            "retained_unsigned_reaction_not_direction",
            "retained unsigned-reaction evidence",
        ),
        "weekly_pivot_grid": ("not_broadly_retained", "were not broadly retained"),
        "rolling_vwap_deviation_bands": (
            "not_broadly_retained",
            "were not broadly retained",
        ),
        "generic_ma_bollinger_negative_control": (
            "negative_control_not_retained",
            "remain negative controls",
        ),
        "change_point_anchored_vwap": (
            "btc_specific_traffic_only",
            "retained BTC-specific traffic evidence only",
        ),
        "abnormal_candle_structural_coordinates": (
            "null_after_controls",
            "were null after the frozen controls",
        ),
        "multitimeframe_level_convergence": (
            "null_after_full_controls",
            "was null after the full frozen controls",
        ),
        "completed_period_landmarks": (
            "null_after_full_controls",
            "were null after the full frozen controls",
        ),
        "causal_trend_channel_boundaries": (
            "null_after_full_controls",
            "were null after the full frozen controls",
        ),
        "rolling_price_distribution_boundaries": (
            "near_miss_not_retained",
            "were near-misses and were not retained",
        ),
        "round_number_price_grid": (
            "near_miss_not_retained",
            "were near-misses and were not retained",
        ),
        "three_plus_timeframe_cluster_attribution": (
            "no_timeframe_precedence_or_cluster_superiority",
            "did not prove timeframe precedence or cluster superiority",
        ),
        "current_and_completed_session_anchored_vwap": (
            "coverage_frozen_incomplete_8h_crossing_candidate",
            "Current-session VWAP 8h crossing remains an incomplete, coverage-frozen candidate",
        ),
        "generic_multitimeframe_indicator_context": (
            "negative_null_control",
            "remains a negative/null control",
        ),
    }
    assert set(by_family) == set(expected)
    assert all(
        row["prior_evidence_class"] == expected[source_family][0]
        and expected[source_family][1] in str(row["plain_limitation"])
        for source_family, family_rows in by_family.items()
        for row in family_rows
    )
    assert all(row["prior_evidence_class"] != "not_imported_source_freeze_only" for row in rows)
    assert all(
        {str(row["episode_type"]) for row in family_rows}
        == {"source_event", "downstream_confirmation"}
        for family_rows in by_family.values()
    )
    attribution = by_family["three_plus_timeframe_cluster_attribution"]
    assert {str(row["source_timeframes"]) for row in attribution} == {"1h;4h;1d;1w"}


def test_stage1_output_contract_has_exactly_three_artifacts() -> None:
    assert stage1.OUTPUT_NAMES == {
        "market_signal_portfolio_stage1_coverage.csv",
        "market_signal_portfolio_stage1_freeze.json",
        "market_signal_portfolio_stage1_preflight_result.json",
    }


def test_partial_output_is_refused_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    freeze = output / stage1.FREEZE_PATH.name
    freeze.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(stage1, "OUTPUT_ROOT", output)
    monkeypatch.setattr(stage1, "FREEZE_PATH", freeze)
    monkeypatch.setattr(stage1, "COVERAGE_PATH", output / stage1.COVERAGE_PATH.name)
    monkeypatch.setattr(stage1, "RESULT_PATH", output / stage1.RESULT_PATH.name)
    with pytest.raises(FileExistsError, match="Partial Stage-1 output"):
        stage1.run_preflight()


def test_result_uses_exact_terminal_paths_and_rejects_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    freeze = output / stage1.FREEZE_PATH.name
    coverage_path = output / stage1.COVERAGE_PATH.name
    result_path = output / stage1.RESULT_PATH.name
    monkeypatch.setattr(stage1, "OUTPUT_ROOT", output)
    monkeypatch.setattr(stage1, "FREEZE_PATH", freeze)
    monkeypatch.setattr(stage1, "COVERAGE_PATH", coverage_path)
    monkeypatch.setattr(stage1, "RESULT_PATH", result_path)
    monkeypatch.setattr(
        stage1, "OUTPUT_NAMES", frozenset({freeze.name, coverage_path.name, result_path.name})
    )
    freeze.write_text("{}", encoding="utf-8")
    _coverage_frame().to_csv(coverage_path, index=False)
    result = stage1._result_payload(_coverage_frame())
    result["artifacts"]["freeze"]["path"] = str(tmp_path / "substitute.json")
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact path drift"):
        stage1.validate_existing_result()

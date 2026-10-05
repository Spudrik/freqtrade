"""Outcome-blind reconstruction preflight for the five-family signal portfolio.

This module projects only source, call, identity, timestamp, and pre-decision fields
from already-frozen evidence.  It never scores a later market outcome.
"""

# ruff: noqa: E501 - frozen trader-readable contracts are intentionally explicit.

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from pandas import DataFrame


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import market_reaction_zone_generation0 as g0  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_ROOT = (
    REPO_ROOT
    / "user_data/research_news_data/context_features/event_hierarchy"
    / "market_signal_portfolio_stage1_20260913a"
)
FREEZE_PATH = OUTPUT_ROOT / "market_signal_portfolio_stage1_freeze.json"
COVERAGE_PATH = OUTPUT_ROOT / "market_signal_portfolio_stage1_coverage.csv"
RESULT_PATH = OUTPUT_ROOT / "market_signal_portfolio_stage1_preflight_result.json"
OUTPUT_NAMES = frozenset({FREEZE_PATH.name, COVERAGE_PATH.name, RESULT_PATH.name})

SOURCE_RESULTS: dict[str, dict[str, str]] = {
    "macro_expectation_direction": {
        "path": str(
            REPO_ROOT
            / "user_data/research_news_data/context_features/event_hierarchy"
            / "macro_expectation_catalogue_20260910a/official_actual_freeze_20260911a"
            / "direction_batch_20260911a/direct_review_20260911a"
            / "macro_expectation_direction_result.json"
        ),
        "sha256": "833c52435923bca660b488575e040b4feba696a179b213b072fb7b451a495a92",
    },
    "layer2_individual_links": {
        "path": str(
            REPO_ROOT
            / "user_data/research_news_data/context_features/event_hierarchy"
            / "layer2_individual_links_20260903a/direct_review_20260904a"
            / "layer2_direct_result.json"
        ),
        "sha256": "6fb27587a6e1160855e561e90dc70c71afc5b114e400b566de289fef2bd393bb",
    },
    "independent_context_freeze": {
        "path": str(
            REPO_ROOT
            / "user_data/research_news_data/context_features/event_hierarchy"
            / "independent_context_sources_20260904a"
            / "independent_context_freeze_result.json"
        ),
        "sha256": "ca87532085e4b9a45644c08a3a111e780228068a75fdaf6845376ef11bfbd5ba",
    },
    "independent_context_direct": {
        "path": str(
            REPO_ROOT
            / "user_data/research_news_data/context_features/event_hierarchy"
            / "independent_context_sources_20260904a/direct_review_20260904a"
            / "independent_context_direct_result.json"
        ),
        "sha256": "958d7df7173d6a356e23556a5b46a840eb57ceb34aa8c1d6bb2b0a37ae5b0411",
    },
    "simple_signal_families": {
        "path": str(
            REPO_ROOT
            / "user_data/research_news_data/context_features/event_hierarchy"
            / "event_signal_fresh_2026_20260909a/event_simple_signal_families_20260909a"
            / "simple_signal_family_result.json"
        ),
        "sha256": "a65118b9eacfb2160b6cc29ad121dd9b0cecf2625901a1820952b5e9dc0ecf87",
    },
    "meme_transmission": {
        "path": str(
            REPO_ROOT
            / "user_data/research_news_data/context_features/event_hierarchy"
            / "event_signal_fresh_2026_20260909a/event_meme_transmission_20260909a"
            / "meme_transmission_result.json"
        ),
        "sha256": "9ab149aaa293aa597a165ffa95f16b341eaadf613c32deaef62ff53b5c1f3855",
    },
    "g18_level_confirmation": {
        "path": str(
            REPO_ROOT
            / "user_data/research_news_data/context_features/market_reaction_zones"
            / "generation11_review/generation11_branches"
            / "g12_chronology_attribution_and_combinations/g13_broad_siblings"
            / "g14_broad_combinations/g15_broad_direct/g16_broad_attribution"
            / "g17_broad_branch_layer/g18_broad_confirmation/direct_confirmation"
            / "g18_direct_confirmation_20260823a/g18_direct_confirmation_result.json"
        ),
        "sha256": "f9911f1b21053f6cc7cf9fd8020eb79cc16e305cb4d2441e3d3a069fae99d48a",
    },
}

LEVEL_MAP_ROOT = (
    REPO_ROOT
    / "user_data/research_news_data/context_features/market_reaction_zones"
    / "generation11_review/generation11_branches"
    / "g12_chronology_attribution_and_combinations/g13_broad_siblings"
    / "g14_broad_combinations/g15_broad_direct/g16_broad_attribution"
    / "g17_broad_branch_layer"
)
LEVEL_MAP_CONTRACTS: dict[str, dict[str, str]] = {
    "g17_registry": {
        "path": str(LEVEL_MAP_ROOT / "level_source_atlas/g17_level_source_registry_freeze.json"),
        "sha256": "724f3fff2b55fc000eee39a9cc533a7cd5cfe885b1e47345358e9181cdfea6ad",
    },
    "g17_inventory": {
        "path": str(
            LEVEL_MAP_ROOT
            / "level_source_atlas/g17_rational_level_source_atlas_20260822a"
            / "pair_inventory.csv"
        ),
        "sha256": "c96decb11c1b5b5417e62512ebe1d5b1046ea7e039e7859005752937cd4cad13",
    },
    "g20_change_point_avwap": {
        "path": str(
            LEVEL_MAP_ROOT
            / "g18_broad_confirmation/g19_broad_combinations/g20_broad_siblings"
            / "change_point_avwap/g20_change_point_support_freeze.json"
        ),
        "sha256": "e9d17c4160bd540ba87ab6281ad23764ebb3b111e66a5964b9a92c6774655bb5",
    },
    "g21_structural_candle": {
        "path": str(
            LEVEL_MAP_ROOT
            / "g18_broad_confirmation/g19_broad_combinations/g20_broad_siblings"
            / "g21_broad_siblings/structural_levels/g21_structural_level_support_freeze.json"
        ),
        "sha256": "94b356e37d6e7456e00f2dc2cb04e915f232bfd93c77110b5f0e0ad83189cfe1",
    },
    "g22_multitimeframe": {
        "path": str(
            LEVEL_MAP_ROOT
            / "g18_broad_confirmation/g19_broad_combinations/g20_broad_siblings"
            / "g21_broad_siblings/g22_broad_siblings/multitimeframe_convergence"
            / "g22_multitimeframe_support_freeze.json"
        ),
        "sha256": "16ea85ed0ad225f015e7942472eaae9bfd1633815433a9a36d68387338f1bea8",
    },
    "g22_period_landmarks": {
        "path": str(
            LEVEL_MAP_ROOT
            / "g18_broad_confirmation/g19_broad_combinations/g20_broad_siblings"
            / "g21_broad_siblings/g22_broad_siblings/period_landmarks"
            / "g22_period_landmark_support_freeze.json"
        ),
        "sha256": "f83995e87199b345944085a04ae2fc3b8b1f369979b7dfda9984d19d76fc8999",
    },
    "g22_trend_channels": {
        "path": str(
            LEVEL_MAP_ROOT
            / "g18_broad_confirmation/g19_broad_combinations/g20_broad_siblings"
            / "g21_broad_siblings/g22_broad_siblings/trend_channels"
            / "g22_trend_channel_support_freeze.json"
        ),
        "sha256": "f678c569f1f14f0e6d49f620c0f2703fa06377ef46b23aa2114909608471d52d",
    },
    "g23_price_distribution": {
        "path": str(
            LEVEL_MAP_ROOT
            / "g18_broad_confirmation/g19_broad_combinations/g20_broad_siblings"
            / "g21_broad_siblings/g22_broad_siblings/g23_broad_siblings"
            / "price_distribution/g23_price_distribution_support_freeze.json"
        ),
        "sha256": "bf1eddc1d5232fd4327845f491d3b52b2924e55c56484498017a2aab20b09bca",
    },
    "g23_round_numbers": {
        "path": str(
            LEVEL_MAP_ROOT
            / "g18_broad_confirmation/g19_broad_combinations/g20_broad_siblings"
            / "g21_broad_siblings/g22_broad_siblings/g23_broad_siblings"
            / "round_numbers/g23_round_number_support_freeze.json"
        ),
        "sha256": "8793ee073b0a5c87c2a3a8b1e0968eb64903d9fbac5ddf339962b2ad912b61a2",
    },
    "g23_multitimeframe_attribution": {
        "path": str(
            LEVEL_MAP_ROOT
            / "g18_broad_confirmation/g19_broad_combinations/g20_broad_siblings"
            / "g21_broad_siblings/g22_broad_siblings/g23_broad_siblings"
            / "multitimeframe_attribution/g23_multitimeframe_attribution_support_freeze.json"
        ),
        "sha256": "7a66478be6eba2d671cec93b79143b1dd8847dfd2cb474d1b5b5f0ac6f878a27",
    },
    "g24_anchored_vwap": {
        "path": str(
            LEVEL_MAP_ROOT
            / "g18_broad_confirmation/g19_broad_combinations/g20_broad_siblings"
            / "g21_broad_siblings/g22_broad_siblings/g23_broad_siblings"
            / "g24_broad_siblings/anchored_vwap/g24_anchored_vwap_support_freeze.json"
        ),
        "sha256": "1da7ebcfc4c0206dc9349c2aa27947437481f2a1857a4789746fcf1915318139",
    },
    "g24_generic_indicator_context": {
        "path": str(
            LEVEL_MAP_ROOT
            / "g18_broad_confirmation/g19_broad_combinations/g20_broad_siblings"
            / "g21_broad_siblings/g22_broad_siblings/g23_broad_siblings"
            / "g24_broad_siblings/generic_indicator_context"
            / "g24_generic_indicator_context_support_freeze.json"
        ),
        "sha256": "0ec9cf512330566beb5efa51d9fed3443e78876d9533310254a74cc9bb475ce2",
    },
}

LEVEL_MAP_PRIOR_EVIDENCE: dict[str, dict[str, str]] = {
    "adaptive_volume_profile_nodes": {
        "class": "retained_crossing_and_traffic_not_direction",
        "limitation": "Adaptive volume-profile nodes retained crossing and traffic evidence, but not signed direction evidence.",
    },
    "donchian_boundaries": {
        "class": "retained_unsigned_reaction_not_direction",
        "limitation": "Rolling high/low and Donchian boundaries retained unsigned-reaction evidence, but not signed direction evidence.",
    },
    "weekly_pivot_grid": {
        "class": "not_broadly_retained",
        "limitation": "Weekly pivots were not broadly retained by the frozen prior evidence.",
    },
    "rolling_vwap_deviation_bands": {
        "class": "not_broadly_retained",
        "limitation": "Rolling VWAP deviation bands were not broadly retained by the frozen prior evidence.",
    },
    "generic_ma_bollinger_negative_control": {
        "class": "negative_control_not_retained",
        "limitation": "Generic moving-average and Bollinger coordinates remain negative controls, not promoted level evidence.",
    },
    "change_point_anchored_vwap": {
        "class": "btc_specific_traffic_only",
        "limitation": "Change-point anchored VWAP retained BTC-specific traffic evidence only, not broad or directional evidence.",
    },
    "abnormal_candle_structural_coordinates": {
        "class": "null_after_controls",
        "limitation": "Structural abnormal-candle levels were null after the frozen controls.",
    },
    "multitimeframe_level_convergence": {
        "class": "null_after_full_controls",
        "limitation": "Multi-timeframe level convergence was null after the full frozen controls.",
    },
    "completed_period_landmarks": {
        "class": "null_after_full_controls",
        "limitation": "Completed-period landmarks were null after the full frozen controls.",
    },
    "causal_trend_channel_boundaries": {
        "class": "null_after_full_controls",
        "limitation": "Causal trend-channel boundaries were null after the full frozen controls.",
    },
    "rolling_price_distribution_boundaries": {
        "class": "near_miss_not_retained",
        "limitation": "Rolling price-distribution boundaries were near-misses and were not retained.",
    },
    "round_number_price_grid": {
        "class": "near_miss_not_retained",
        "limitation": "Round-number grids were near-misses and were not retained.",
    },
    "three_plus_timeframe_cluster_attribution": {
        "class": "no_timeframe_precedence_or_cluster_superiority",
        "limitation": "Three-plus-timeframe attribution did not prove timeframe precedence or cluster superiority.",
    },
    "current_and_completed_session_anchored_vwap": {
        "class": "coverage_frozen_incomplete_8h_crossing_candidate",
        "limitation": "Current-session VWAP 8h crossing remains an incomplete, coverage-frozen candidate rather than retained evidence.",
    },
    "generic_multitimeframe_indicator_context": {
        "class": "negative_null_control",
        "limitation": "Generic multi-timeframe indicator context remains a negative/null control, not promoted context evidence.",
    },
}

PROTOTYPES: tuple[dict[str, str], ...] = (
    {
        "family_id": "major_event_information",
        "prototype_id": "cpi_headline_core_agreement_direction",
        "episode_type": "source_event",
        "role": "driver_and_event_scoped_direction",
        "decision_time_rule": "official_release_time_after_timestamp_safe_actuals_and_expectations_are_available",
    },
    {
        "family_id": "major_event_information",
        "prototype_id": "fomc_event_activity_warning",
        "episode_type": "source_event",
        "role": "driver_and_unsigned_activity_warning",
        "decision_time_rule": "official_fomc_release_clock",
    },
    {
        "family_id": "major_event_information",
        "prototype_id": "live_news_web_activity_warning",
        "episode_type": "source_event",
        "role": "driver_and_unsigned_activity_warning",
        "decision_time_rule": "source_defined_first_available_time",
    },
    {
        "family_id": "background_and_market_leadership",
        "prototype_id": "negative_background_positive_event_fade",
        "episode_type": "downstream_confirmation",
        "role": "conditional_fade_modifier_and_confirmation",
        "decision_time_rule": "event_plus_4h_after_positive_btc_close_with_causally_prior_negative_30d_return",
    },
    {
        "family_id": "calculated_reaction_areas",
        "prototype_id": "calculated_area_contact_traffic",
        "episode_type": "level_contact",
        "role": "reaction_location_and_unsigned_traffic",
        "decision_time_rule": "contact_time_after_the_level_source_candle_is_available",
    },
    {
        "family_id": "calculated_reaction_areas",
        "prototype_id": "local_multitimeframe_support_resistance_map",
        "episode_type": "source_event_and_downstream_confirmation",
        "role": "timestamp_safe_local_modifier_and_context_map",
        "decision_time_rule": "map_only_levels_settled_and_available_at_the_specific_decision_time",
    },
    {
        "family_id": "local_participation_and_pressure",
        "prototype_id": "recent_volume_persistence",
        "episode_type": "source_event",
        "role": "unsigned_readiness_and_persistence",
        "decision_time_rule": "anchor_time_using_only_completed_recent_volume",
    },
    {
        "family_id": "cross_asset_transmission_and_amplification",
        "prototype_id": "confirmed_btc_to_group_activity",
        "episode_type": "downstream_confirmation",
        "role": "leader_confirmation_and_activity_transmission",
        "decision_time_rule": "one_hour_after_parent_anchor_when_the_initial_btc_response_is_observable",
    },
    {
        "family_id": "cross_asset_transmission_and_amplification",
        "prototype_id": "btc_dominance_relative_meme_rotation",
        "episode_type": "source_event",
        "role": "relative_direction_only",
        "decision_time_rule": "source_observation_anchor_after_dominance_value_is_available",
    },
)

COVERAGE_COLUMNS = [
    "family_id",
    "prototype_id",
    "episode_type",
    "role",
    "decision_time_rule",
    "source_sibling",
    "cohort",
    "pair",
    "horizon",
    "independent_episode_count",
    "partition_counts_json",
    "positive_sign_support",
    "negative_sign_support",
    "actual_count",
    "control_count",
    "missing_required_inputs",
    "source_timestamp_quality",
    "latest_contributing_data_rule",
    "eligible",
    "plain_limitation",
    "source_family",
    "coordinate_type",
    "source_timeframes",
    "lookback_or_session_basis",
    "availability_settlement_rule",
    "above_price_resistance",
    "below_price_support",
    "nearest_distance",
    "room_to_opposing_level",
    "approach_side",
    "age_and_prior_touches",
    "broken_reclaimed_rejected_state",
    "independent_cluster_membership",
    "intended_level_role",
    "prior_evidence_class",
    "materialization_status",
    "source_support_row_count",
]

FORBIDDEN_PARQUET_COLUMNS = frozenset(
    {
        "response_return",
        "actual_direction",
        "direction_hit",
        "pretrend_hit",
        "previous_proxy_hit",
        "ordinary_control_accuracy",
        "rotated_hit",
        "development_majority_hit",
        "return_4h",
        "return_24h",
        "positive_move_faded_by_half",
        "negative_initial_move",
        "negative_move_extended",
        "volume_majority_reacted",
        "range_majority_reacted",
        "later_direction_aligned",
        "amplified_beyond_ordinary",
        "elevated_volume_fraction",
        "elevated_range_fraction",
        "cohort_lag_log_return",
        "cohort_absolute_lag_return",
        "median_residual_toward_btc_z",
    }
)
FORBIDDEN_LEVEL_PREFIXES = (
    "abs_excursion_",
    "away_excursion_",
    "through_excursion_",
    "volume_ratio_",
    "crossings_",
)
ALLOWED_PARQUET_COLUMNS = frozenset(
    {
        "route_id",
        "expectation_episode_id",
        "source_block",
        "anchor_utc",
        "pair",
        "horizon_minutes",
        "predicted_direction",
        "abstention_reason",
        "ordinary_control_count",
        "event_id",
        "event_source",
        "whole_event_partition",
        "sample_type",
        "pre_return_30d",
        "background",
        "positive_initial_move",
        "sample_kind",
        "analysis_unit_id",
        "model_anchor_utc",
        "model_period",
        "signal_id",
        "observation_utc",
        "latest_contributing_data_utc",
        "episode_id",
        "btc_confirmed_reaction",
        "horizon_hours",
        "cohort",
        "level_family",
        "level_name",
        "control",
        "event_time",
        "period",
        "source_available_at",
    }
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _path_sha(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def _verify_contract(name: str, item: dict[str, Any]) -> Path:
    path = Path(item["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{name} missing: {path}")
    actual = g0.sha256_file(path)
    expected = str(item["sha256"])
    if actual != expected:
        raise ValueError(f"{name} hash drift: expected {expected}, got {actual}")
    return path


def read_parquet_projected(path: Path, columns: Iterable[str]) -> DataFrame:
    requested = list(columns)
    forbidden = sorted(
        column
        for column in requested
        if column not in ALLOWED_PARQUET_COLUMNS
        or column in FORBIDDEN_PARQUET_COLUMNS
        or column.startswith(FORBIDDEN_LEVEL_PREFIXES)
    )
    if forbidden:
        raise ValueError(f"Forbidden outcome columns requested: {forbidden}")
    schema = set(pq.read_schema(path).names)
    missing = sorted(set(requested) - schema)
    if missing:
        raise ValueError(f"Projected source schema drift at {path}: {missing}")
    return pd.read_parquet(path, columns=requested)


def _require_columns(frame: DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} schema drift: {missing}")


def _timestamps(frame: DataFrame, columns: Iterable[str], label: str) -> DataFrame:
    output = frame.copy()
    for column in columns:
        output[column] = pd.to_datetime(output[column], utc=True, errors="raise", format="mixed")
        if output[column].isna().any():
            raise ValueError(f"{label} has missing {column}")
    return output


def assert_no_future_source(
    frame: DataFrame, source_column: str, decision_column: str, label: str
) -> None:
    checked = _timestamps(frame, [source_column, decision_column], label)
    future = checked[source_column] > checked[decision_column]
    if future.any():
        raise ValueError(f"{label} has {int(future.sum())} future timestamp violations")


def deduplicate_independent(
    frame: DataFrame, key: list[str], stable: list[str], label: str
) -> DataFrame:
    _require_columns(frame, [*key, *stable], label)
    if frame[key].isna().any().any():
        raise ValueError(f"{label} has a missing independent key")
    conflicts = []
    duplicated = frame[frame.duplicated(key, keep=False)]
    for column in stable:
        if (
            not duplicated.empty
            and duplicated.groupby(key, dropna=False)[column].nunique(dropna=False).gt(1).any()
        ):
            conflicts.append(column)
    if conflicts:
        raise ValueError(f"{label} has conflicting duplicate keys: {conflicts}")
    return frame.drop_duplicates(key).copy()


def _partition_counts(frame: DataFrame, id_column: str, part_column: str) -> str:
    values = (
        frame.groupby(part_column, observed=True)[id_column]
        .nunique()
        .sort_index()
        .astype(int)
        .to_dict()
    )
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def _row(prototype_id: str, **values: Any) -> dict[str, Any]:
    prototype = next(p for p in PROTOTYPES if p["prototype_id"] == prototype_id)
    defaults: dict[str, Any] = {
        **prototype,
        "source_sibling": "",
        "cohort": "",
        "pair": "",
        "horizon": "",
        "independent_episode_count": 0,
        "partition_counts_json": "{}",
        "positive_sign_support": "",
        "negative_sign_support": "",
        "actual_count": 0,
        "control_count": 0,
        "missing_required_inputs": "",
        "source_timestamp_quality": "",
        "latest_contributing_data_rule": "",
        "eligible": False,
        "plain_limitation": "",
        "source_family": "",
        "coordinate_type": "",
        "source_timeframes": "",
        "lookback_or_session_basis": "",
        "availability_settlement_rule": "",
        "above_price_resistance": "",
        "below_price_support": "",
        "nearest_distance": "",
        "room_to_opposing_level": "",
        "approach_side": "",
        "age_and_prior_touches": "",
        "broken_reclaimed_rejected_state": "",
        "independent_cluster_membership": "",
        "intended_level_role": "",
        "prior_evidence_class": "",
        "materialization_status": "",
        "source_support_row_count": 0,
    }
    defaults.update(values)
    return {column: defaults[column] for column in COVERAGE_COLUMNS}


def _load_json_sources() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    results: dict[str, dict[str, Any]] = {}
    contracts: list[dict[str, Any]] = []
    for name, item in SOURCE_RESULTS.items():
        path = _verify_contract(name, item)
        results[name] = json.loads(path.read_text(encoding="utf-8"))
        contracts.append({"name": name, **_path_sha(path)})
    return results, contracts


def _verified_child(contracts: list[dict[str, Any]], name: str, item: dict[str, Any]) -> Path:
    path = _verify_contract(name, item)
    contracts.append({"name": name, **_path_sha(path)})
    return path


def build_cpi_coverage(rows: DataFrame) -> list[dict[str, Any]]:
    required = [
        "route_id",
        "expectation_episode_id",
        "source_block",
        "anchor_utc",
        "pair",
        "horizon_minutes",
        "predicted_direction",
        "abstention_reason",
        "ordinary_control_count",
    ]
    _require_columns(rows, required, "CPI call rows")
    rows = rows.loc[rows["route_id"].eq("cpi_headline_core_agreement")].copy()
    rows = rows.loc[rows["abstention_reason"].fillna("").eq("")]
    rows = _timestamps(rows, ["anchor_utc"], "CPI call rows")
    output: list[dict[str, Any]] = []
    for (pair, horizon), group in rows.groupby(
        ["pair", "horizon_minutes"], observed=True, sort=True
    ):
        episodes = deduplicate_independent(
            group,
            ["expectation_episode_id"],
            ["source_block", "anchor_utc", "predicted_direction"],
            "CPI episodes",
        )
        signs = pd.to_numeric(episodes["predicted_direction"], errors="raise")
        output.append(
            _row(
                "cpi_headline_core_agreement_direction",
                source_sibling="cpi_headline_and_core_agreement",
                cohort="BTC_ETH",
                pair=str(pair),
                horizon=f"{int(horizon)}m",
                independent_episode_count=len(episodes),
                partition_counts_json=_partition_counts(
                    episodes, "expectation_episode_id", "source_block"
                ),
                positive_sign_support=int((signs > 0).sum()),
                negative_sign_support=int((signs < 0).sum()),
                actual_count=len(episodes),
                control_count=int(
                    pd.to_numeric(episodes["ordinary_control_count"], errors="raise").sum()
                ),
                source_timestamp_quality="official_release_minute_with_timestamp_safe_expectation_cutoff",
                latest_contributing_data_rule="actuals_and_both_expectations_must_exist_by_the_official_release_decision_time",
                eligible=bool(len(episodes)),
                plain_limitation="CPI only; signed calls abstain unless headline and core surprise signs agree.",
            )
        )
    return output


def build_fomc_and_background_coverage(
    events: DataFrame, controls: DataFrame, background: DataFrame
) -> list[dict[str, Any]]:
    events = _timestamps(events, ["anchor_utc"], "Layer-2 events")
    fomc = events.loc[events["event_source"].eq("official_fomc")].copy()
    fomc = deduplicate_independent(
        fomc, ["event_id"], ["anchor_utc", "whole_event_partition"], "FOMC events"
    )
    fomc_controls = controls.loc[controls["event_source"].eq("official_fomc")]
    output = []
    for pair in (
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "BNB/USDT:USDT",
        "ADA/USDT:USDT",
        "TRX/USDT:USDT",
    ):
        output.append(
            _row(
                "fomc_event_activity_warning",
                source_sibling="official_fomc",
                cohort="BTC_ETH_established",
                pair=pair,
                horizon="1h;4h;8h",
                independent_episode_count=len(fomc),
                partition_counts_json=_partition_counts(fomc, "event_id", "whole_event_partition"),
                actual_count=len(fomc),
                control_count=len(fomc_controls),
                source_timestamp_quality="official_scheduled_release_clock",
                latest_contributing_data_rule="the_official_event clock is the decision anchor",
                eligible=bool(len(fomc)),
                plain_limitation="Unsigned activity warning; no FOMC direction is inferred without surprise information.",
            )
        )

    required = [
        "event_id",
        "event_source",
        "whole_event_partition",
        "sample_type",
        "pre_return_30d",
        "background",
        "positive_initial_move",
    ]
    _require_columns(background, required, "Layer-2 background rows")
    selected = background.loc[
        background["background"].eq("negative_30d_background")
        & background["positive_initial_move"].astype(bool)
    ].copy()
    for source in ("official_fomc", "gdelt_activity_spike"):
        source_rows = selected.loc[selected["event_source"].eq(source)]
        actual = source_rows.loc[source_rows["sample_type"].eq("event")]
        control = source_rows.loc[source_rows["sample_type"].eq("control")]
        actual = deduplicate_independent(
            actual,
            ["event_id"],
            ["whole_event_partition", "positive_initial_move"],
            f"{source} fade confirmations",
        )
        output.append(
            _row(
                "negative_background_positive_event_fade",
                source_sibling=source,
                cohort="BTC",
                pair="BTC/USDT:USDT",
                horizon="event+4h_to_event+24h",
                independent_episode_count=len(actual),
                partition_counts_json=_partition_counts(
                    actual, "event_id", "whole_event_partition"
                ),
                actual_count=len(actual),
                control_count=int(control["event_id"].nunique()),
                source_timestamp_quality="source_event_clock_plus_observed_four_hour_close",
                latest_contributing_data_rule="decision waits for event+4h; prior 30d return ends before the event",
                eligible=bool(len(actual)),
                plain_limitation="FOMC and GDELT remain separate siblings; the first four-hour move is confirmation, not a root-cause replacement.",
            )
        )
    return output


def _parse_cells(value: Any) -> list[tuple[str, int]]:
    if pd.isna(value):
        return []
    cells = []
    for token in str(value).split(";"):
        scope, horizon = token.strip().rsplit(":", 1)
        cells.append((scope, int(horizon.removesuffix("h"))))
    return cells


def build_independent_context_coverage(
    events: DataFrame, controls: DataFrame, verdicts: DataFrame
) -> list[dict[str, Any]]:
    required = [
        "family",
        "available_at",
        "anchor_utc",
        "whole_event_partition",
        "source_direction",
        "predicted_direction",
        "is_event",
        "event_id",
        "coverage_ready",
    ]
    _require_columns(events, required, "Independent-context events")
    events = events.loc[
        events["is_event"].astype(bool) & events["coverage_ready"].astype(bool)
    ].copy()
    assert_no_future_source(events, "available_at", "anchor_utc", "Independent context")
    output: list[dict[str, Any]] = []
    for family in ("live_news_activity_spike", "live_web_activity_spike"):
        verdict = verdicts.loc[verdicts["family"].eq(family)]
        if len(verdict) != 1 or verdict.iloc[0]["verdict"] != "repeatable_activity_only_lead":
            raise ValueError(f"Retained source verdict drift for {family}")
        family_events = deduplicate_independent(
            events.loc[events["family"].eq(family)],
            ["event_id"],
            ["available_at", "anchor_utc", "whole_event_partition"],
            family,
        )
        family_controls = controls.loc[controls["family"].eq(family)]
        for scope, horizon in _parse_cells(verdict.iloc[0]["activity_cells"]):
            if horizon not in {1, 4, 8}:
                continue
            output.append(
                _row(
                    "live_news_web_activity_warning",
                    source_sibling=family,
                    cohort=scope,
                    pair=scope,
                    horizon=f"{horizon}h",
                    independent_episode_count=len(family_events),
                    partition_counts_json=_partition_counts(
                        family_events, "event_id", "whole_event_partition"
                    ),
                    actual_count=len(family_events),
                    control_count=len(family_controls),
                    source_timestamp_quality="recorded_first_available_source_time_not_later_than_anchor",
                    latest_contributing_data_rule="only source observations recorded by the first-available anchor are eligible",
                    eligible=True,
                    plain_limitation="Only the previously retained live clean scope/horizon cell is registered; news and web remain separate source siblings.",
                )
            )

    family = "btc_dominance_change"
    verdict = verdicts.loc[verdicts["family"].eq(family)]
    if len(verdict) != 1 or verdict.iloc[0]["verdict"] != "repeatable_direction_only_lead":
        raise ValueError("BTC-dominance source verdict drift")
    dominance = deduplicate_independent(
        events.loc[events["family"].eq(family)],
        ["event_id"],
        ["available_at", "anchor_utc", "whole_event_partition", "predicted_direction"],
        family,
    )
    signs = pd.to_numeric(dominance["predicted_direction"], errors="raise")
    output.append(
        _row(
            "btc_dominance_relative_meme_rotation",
            source_sibling=family,
            cohort="btc_relative_to_top_ten_memes",
            pair="BTC_vs_top_ten_memes",
            horizon="8h",
            independent_episode_count=len(dominance),
            partition_counts_json=_partition_counts(dominance, "event_id", "whole_event_partition"),
            positive_sign_support=int((signs > 0).sum()),
            negative_sign_support=int((signs < 0).sum()),
            actual_count=len(dominance),
            control_count=len(controls.loc[controls["family"].eq(family)]),
            source_timestamp_quality="recorded_source_observation_available_before_or_at_anchor",
            latest_contributing_data_rule="dominance observation must be available by its source anchor",
            eligible=bool(len(dominance)),
            plain_limitation="Relative BTC-versus-meme direction only; never absolute market direction.",
        )
    )
    return output


def build_recent_volume_coverage(calls: DataFrame) -> list[dict[str, Any]]:
    required = [
        "sample_kind",
        "analysis_unit_id",
        "model_anchor_utc",
        "model_period",
        "pair",
        "signal_id",
        "observation_utc",
        "latest_contributing_data_utc",
    ]
    _require_columns(calls, required, "Simple signal calls")
    calls = calls.loc[calls["signal_id"].eq("recent_volume_persistence")].copy()
    calls = _timestamps(
        calls,
        ["model_anchor_utc", "observation_utc", "latest_contributing_data_utc"],
        "Recent-volume calls",
    )
    if (calls["observation_utc"] > calls["model_anchor_utc"]).any() or (
        calls["latest_contributing_data_utc"] > calls["model_anchor_utc"]
    ).any():
        raise ValueError("Recent-volume calls contain future contributing timestamps")
    output = []
    for pair, group in calls.groupby("pair", observed=True, sort=True):
        actual = group.loc[group["sample_kind"].eq("actual_event")]
        controls = group.loc[group["sample_kind"].eq("matched_control")]
        actual = deduplicate_independent(
            actual,
            ["analysis_unit_id"],
            ["model_anchor_utc", "model_period"],
            f"recent volume {pair}",
        )
        output.append(
            _row(
                "recent_volume_persistence",
                source_sibling="recent_volume_persistence",
                cohort="BTC_ETH_established",
                pair=str(pair),
                horizon="1h;4h;8h",
                independent_episode_count=len(actual),
                partition_counts_json=_partition_counts(actual, "analysis_unit_id", "model_period"),
                actual_count=len(actual),
                control_count=int(controls["analysis_unit_id"].nunique()),
                source_timestamp_quality="completed_feature_observation_at_or_before_anchor",
                latest_contributing_data_rule="latest contributing candle must not be later than the call anchor",
                eligible=bool(len(actual)),
                plain_limitation="Readiness/persistence only; it does not attribute market activity to any event.",
            )
        )
    return output


def build_transmission_coverage(groups: DataFrame) -> list[dict[str, Any]]:
    required = [
        "sample_kind",
        "episode_id",
        "model_anchor_utc",
        "model_period",
        "btc_confirmed_reaction",
        "horizon_hours",
        "cohort",
    ]
    _require_columns(groups, required, "Meme transmission groups")
    groups = _timestamps(groups, ["model_anchor_utc"], "Meme transmission groups")
    groups["decision_utc"] = groups["model_anchor_utc"] + pd.Timedelta(hours=1)
    output = []
    for cohort, cohort_rows in groups.groupby("cohort", observed=True, sort=True):
        actual = cohort_rows.loc[
            cohort_rows["sample_kind"].eq("actual_event")
            & cohort_rows["btc_confirmed_reaction"].astype(bool)
        ]
        controls = cohort_rows.loc[
            cohort_rows["sample_kind"].eq("matched_control")
            & cohort_rows["btc_confirmed_reaction"].astype(bool)
        ]
        actual = deduplicate_independent(
            actual,
            ["episode_id", "decision_utc"],
            ["model_period", "btc_confirmed_reaction"],
            f"BTC transmission {cohort}",
        )
        output.append(
            _row(
                "confirmed_btc_to_group_activity",
                source_sibling="confirmed_btc_initial_response",
                cohort=str(cohort),
                pair=str(cohort),
                horizon="1h;3h_after_confirmation",
                independent_episode_count=len(actual),
                partition_counts_json=_partition_counts(actual, "episode_id", "model_period"),
                actual_count=len(actual),
                control_count=int(controls["episode_id"].nunique()),
                source_timestamp_quality="one_hour_btc_confirmation_close_is_observable",
                latest_contributing_data_rule="decision time is parent anchor plus one completed BTC hour",
                eligible=bool(len(actual)),
                plain_limitation="Established and meme cohorts remain separate; this is downstream activity confirmation, not advance direction.",
            )
        )
    return output


def build_level_coverage(
    inventory: list[dict[str, Any]], contracts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    allowed = {"adaptive_volume_profile_nodes", "rolling_vwap_deviation_bands"}
    counters: Counter[tuple[str, str, str, str]] = Counter()
    actual_ids: dict[tuple[str, str], set[tuple[str, str, str]]] = {}
    for item in inventory:
        path = _verified_child(
            contracts,
            f"g17_pair_{item['pair']}",
            {"path": item["source_path"], "sha256": item["source_sha256"]},
        )
        frame = read_parquet_projected(
            path,
            [
                "pair",
                "level_family",
                "level_name",
                "control",
                "event_time",
                "period",
                "source_available_at",
                "cohort",
            ],
        )
        frame = frame.loc[frame["level_family"].isin(allowed)].copy()
        frame = _timestamps(frame, ["event_time"], str(path))
        frame["source_available_at"] = pd.to_datetime(
            frame["source_available_at"], utc=True, errors="raise", format="mixed"
        )
        actual_source = frame.loc[frame["control"].eq("actual"), "source_available_at"]
        if actual_source.isna().any():
            raise ValueError(f"G17 actual contacts lack source availability at {path}")
        source_known = frame["source_available_at"].notna()
        if (
            frame.loc[source_known, "source_available_at"] > frame.loc[source_known, "event_time"]
        ).any():
            raise ValueError(f"G17 level source is future-dated at {path}")
        frame["family_count"] = frame.groupby(["pair", "control", "event_time"], observed=True)[
            "level_family"
        ].transform("nunique")
        singles = frame.loc[frame["family_count"].eq(1)].copy()
        singles["level_identity"] = singles["level_name"].astype(str)
        singles = singles.drop_duplicates(["pair", "control", "event_time", "level_identity"])
        for (cohort, family, period, control), group in singles.groupby(
            ["cohort", "level_family", "period", "control"],
            observed=True,
            sort=False,
        ):
            counters[(str(cohort), f"{family}:single_level", str(period), str(control))] += len(
                group
            )
            if control == "actual":
                actual_ids.setdefault((str(cohort), f"{family}:single_level"), set()).update(
                    zip(
                        group["pair"].astype(str),
                        group["event_time"].astype(str),
                        group["level_identity"].astype(str),
                        strict=True,
                    )
                )
        clusters = frame.loc[frame["family_count"].ge(2)].copy()
        cluster_keys = ["pair", "control", "event_time", "period", "cohort"]
        clusters = clusters.groupby(cluster_keys, observed=True, as_index=False).agg(
            level_identity=(
                "level_family",
                lambda values: "cluster:" + "+".join(sorted(set(map(str, values)))),
            )
        )
        sibling = "adaptive_volume_profile_nodes+rolling_vwap_deviation_bands:independent_cluster"
        for (cohort, period, control), group in clusters.groupby(
            ["cohort", "period", "control"], observed=True, sort=False
        ):
            counters[(str(cohort), sibling, str(period), str(control))] += len(group)
            if control == "actual":
                actual_ids.setdefault((str(cohort), sibling), set()).update(
                    zip(
                        group["pair"].astype(str),
                        group["event_time"].astype(str),
                        group["level_identity"].astype(str),
                        strict=True,
                    )
                )

    output = []
    for cohort, sibling in sorted(actual_ids):
        partitions = {
            period: counters[(cohort, sibling, period, "actual")]
            for period in sorted({key[2] for key in counters if key[:2] == (cohort, sibling)})
            if counters[(cohort, sibling, period, "actual")]
        }
        controls = sum(
            count
            for (row_cohort, row_sibling, _period, control), count in counters.items()
            if (row_cohort, row_sibling) == (cohort, sibling) and control != "actual"
        )
        output.append(
            _row(
                "calculated_area_contact_traffic",
                source_sibling=sibling,
                cohort=cohort,
                pair="cohort_pairs",
                horizon="1h;2h;4h",
                independent_episode_count=len(actual_ids[(cohort, sibling)]),
                partition_counts_json=json.dumps(partitions, sort_keys=True, separators=(",", ":")),
                actual_count=sum(partitions.values()),
                control_count=controls,
                source_timestamp_quality="source_level_candle_available_at_or_before_contact",
                latest_contributing_data_rule="the calculated level is fixed from data available no later than contact time",
                eligible=True,
                plain_limitation="A pair+contact-time+named-level (or independent-family cluster) is one contact, not independent confirmation by a market event.",
            )
        )
    return output


def _load_level_map_contracts(
    contracts: list[dict[str, Any]],
) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    for name, item in LEVEL_MAP_CONTRACTS.items():
        path = _verified_child(contracts, f"level_map_{name}", item)
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            _require_columns(frame, ["cohort", "pair"], name)
            sources[name] = frame
            continue
        freeze = json.loads(path.read_text(encoding="utf-8"))
        if not str(freeze.get("status", "")).startswith("frozen_before_generation"):
            raise ValueError(f"{name} is not an outcome-blind source freeze")
        outcome_flags = {
            key: value
            for key, value in freeze.items()
            if key
            in {
                "future_outcome_columns_read",
                "future_outcome_values_read",
                "future_signed_direction_used",
                "parameter_selection_used_event_outcomes",
                "profit_used",
            }
        }
        if any(value is not False for value in outcome_flags.values()):
            raise ValueError(f"{name} outcome-blind flags drifted: {outcome_flags}")
        sources[name] = freeze
    g17 = sources["g17_registry"]
    families = Counter(str(item["family"]) for item in g17["source_specs"])
    expected = {
        "adaptive_volume_profile_nodes": 72,
        "donchian_boundaries": 8,
        "generic_ma_bollinger_negative_control": 8,
        "rolling_vwap_deviation_bands": 20,
        "weekly_pivot_grid": 5,
    }
    if dict(families) != expected or g17.get("source_spec_count") != 113:
        raise ValueError("G17 level-source registry drifted")
    return sources


def _level_map_specs(sources: dict[str, Any]) -> list[dict[str, str]]:
    g17 = sources["g17_registry"]
    g17_names: dict[str, list[str]] = {}
    for item in g17["source_specs"]:
        g17_names.setdefault(str(item["family"]), []).append(str(item["name"]))

    specs: list[dict[str, str]] = [
        {
            "contract": "g17_registry",
            "source_family": "adaptive_volume_profile_nodes",
            "coordinate_type": ";".join(sorted(g17_names["adaptive_volume_profile_nodes"])),
            "source_timeframes": "1h_native",
            "basis": "72h;168h;720h adaptive 24/48/96/Freedman-Diaconis bins",
            "settlement": "settled from completed hourly data before contact",
            "roles": "support;resistance;centre/reference;cluster_component",
            "cluster": "available",
        },
        {
            "contract": "g17_registry",
            "source_family": "donchian_boundaries",
            "coordinate_type": ";".join(sorted(g17_names["donchian_boundaries"])),
            "source_timeframes": "1h_native",
            "basis": "prior 24h;72h;168h;720h high/low boundaries",
            "settlement": "prior completed-window boundary shifted one hourly candle",
            "roles": "support;resistance;boundary;cluster_component",
            "cluster": "available",
        },
        {
            "contract": "g17_registry",
            "source_family": "weekly_pivot_grid",
            "coordinate_type": ";".join(sorted(g17_names["weekly_pivot_grid"])),
            "source_timeframes": "1w_structural_higher",
            "basis": "previous completed ISO-style weekly high/low/close pivot grid",
            "settlement": "previous completed week only",
            "roles": "support;resistance;centre/reference;cluster_component",
            "cluster": "available",
        },
        {
            "contract": "g17_registry",
            "source_family": "rolling_vwap_deviation_bands",
            "coordinate_type": ";".join(sorted(g17_names["rolling_vwap_deviation_bands"])),
            "source_timeframes": "1h_native",
            "basis": "24h;72h;168h;720h rolling VWAP with 1/2 deviation bands",
            "settlement": "prior completed hourly observations only",
            "roles": "support;resistance;centre/reference;cluster_component",
            "cluster": "available",
        },
        {
            "contract": "g17_registry",
            "source_family": "generic_ma_bollinger_negative_control",
            "coordinate_type": ";".join(sorted(g17_names["generic_ma_bollinger_negative_control"])),
            "source_timeframes": "1h_native",
            "basis": "SMA20/50/200;EMA20/50;Bollinger20 upper/mid/lower",
            "settlement": "prior completed hourly observations only",
            "roles": "negative_control",
            "cluster": "control_only_not_promoted",
        },
        {
            "contract": "g20_change_point_avwap",
            "source_family": "change_point_anchored_vwap",
            "coordinate_type": ";".join(sources["g20_change_point_avwap"]["levels"]),
            "source_timeframes": "1h_native",
            "basis": "causal completed-candle change point with centre and +/-1.5 sigma",
            "settlement": "completed change-point candle only",
            "roles": "support;resistance;centre/reference;cluster_component",
            "cluster": "derivable_not_materialized",
        },
        {
            "contract": "g21_structural_candle",
            "source_family": "abnormal_candle_structural_coordinates",
            "coordinate_type": ";".join(sources["g21_structural_candle"]["levels"]),
            "source_timeframes": "1h_native",
            "basis": "completed abnormal-candle open/close/body/high/low/typical price",
            "settlement": "structural candle must be complete before use",
            "roles": "support;resistance;boundary;centre/reference;cluster_component",
            "cluster": "derivable_not_materialized",
        },
        {
            "contract": "g22_multitimeframe",
            "source_family": "multitimeframe_level_convergence",
            "coordinate_type": ";".join(sources["g22_multitimeframe"]["families"]),
            "source_timeframes": ";".join(sources["g22_multitimeframe"]["source_timeframes"]),
            "basis": "native;adjacent_higher;structural_higher completed sources",
            "settlement": "each source timeframe must have completed before the 1h anchor",
            "roles": "support;resistance;boundary;centre/reference;cluster_component",
            "cluster": "available",
        },
        {
            "contract": "g22_period_landmarks",
            "source_family": "completed_period_landmarks",
            "coordinate_type": ";".join(sources["g22_period_landmarks"]["coordinates"]),
            "source_timeframes": "1d;1w;1M_structural_higher",
            "basis": ";".join(sources["g22_period_landmarks"]["periods"]),
            "settlement": "previous completed UTC day, ISO week, or UTC month only",
            "roles": "support;resistance;boundary;centre/reference;cluster_component",
            "cluster": "derivable_not_materialized",
        },
        {
            "contract": "g22_trend_channels",
            "source_family": "causal_trend_channel_boundaries",
            "coordinate_type": ";".join(sources["g22_trend_channels"]["coordinates"]),
            "source_timeframes": "1h_native_with_structural_lookbacks",
            "basis": ";".join(f"{value}h" for value in sources["g22_trend_channels"]["lookbacks"]),
            "settlement": "completed candles only",
            "roles": "support;resistance;boundary;centre/reference;cluster_component",
            "cluster": "derivable_not_materialized",
        },
        {
            "contract": "g23_price_distribution",
            "source_family": "rolling_price_distribution_boundaries",
            "coordinate_type": "q10;q25;q50;q75;q90",
            "source_timeframes": "1h_native_with_structural_lookbacks",
            "basis": ";".join(
                f"{value}h" for value in sources["g23_price_distribution"]["lookbacks"]
            ),
            "settlement": "completed hourly typical prices shifted one hour",
            "roles": "support;resistance;boundary;centre/reference;cluster_component",
            "cluster": "derivable_not_materialized",
        },
        {
            "contract": "g23_round_numbers",
            "source_family": "round_number_price_grid",
            "coordinate_type": "nearest grid at 1.0x;0.5x;0.25x base step",
            "source_timeframes": "1h_native_price_scale",
            "basis": "base step from previous close order of magnitude",
            "settlement": "previous close fixes the causal grid before the anchor",
            "roles": "support;resistance;boundary;cluster_component",
            "cluster": "derivable_not_materialized",
        },
        {
            "contract": "g23_multitimeframe_attribution",
            "source_family": "three_plus_timeframe_cluster_attribution",
            "coordinate_type": "highest_timeframe;nearest_non_anchor;anchor_component",
            "source_timeframes": "1h;4h;1d;1w",
            "basis": "G22 three-plus-timeframe clusters with component attribution",
            "settlement": "all contributing timeframe sources completed before anchor",
            "roles": "cluster_component",
            "cluster": "available",
        },
        {
            "contract": "g24_anchored_vwap",
            "source_family": "current_and_completed_session_anchored_vwap",
            "coordinate_type": "centre;upper_1sigma;lower_1sigma;upper_2sigma;lower_2sigma",
            "source_timeframes": "1d;1w;1M_structural_higher",
            "basis": "current session through previous candle;previous completed session",
            "settlement": "current session excludes the active candle;completed mode uses prior session",
            "roles": "support;resistance;boundary;centre/reference;cluster_component",
            "cluster": "derivable_not_materialized",
        },
        {
            "contract": "g24_generic_indicator_context",
            "source_family": "generic_multitimeframe_indicator_context",
            "coordinate_type": "RSI;Bollinger position/width;MACD histogram;EMA spread/slope;ATR fraction state bands",
            "source_timeframes": "1h;4h;1d",
            "basis": "preceding completed-candle indicator training tertiles",
            "settlement": "preceding completed candle only",
            "roles": "negative_control",
            "cluster": "not_applicable_context_control",
        },
    ]
    return specs


def build_local_map_coverage(sources: dict[str, Any]) -> list[dict[str, Any]]:
    specs = _level_map_specs(sources)
    source_families = {spec["source_family"] for spec in specs}
    if source_families != set(LEVEL_MAP_PRIOR_EVIDENCE):
        raise ValueError("Local-map prior-evidence registry drifted")
    g17_inventory = sources["g17_inventory"]
    output: list[dict[str, Any]] = []
    for spec in specs:
        source = sources[spec["contract"]]
        prior_evidence = LEVEL_MAP_PRIOR_EVIDENCE[spec["source_family"]]
        if spec["contract"] == "g17_registry":
            inventory = g17_inventory[["cohort", "pair"]].drop_duplicates()
            support_rows = 0
        else:
            inventory = pd.DataFrame(source["inventory"])[["cohort", "pair", "rows"]]
            support_rows = None
        for cohort, rows in inventory.groupby("cohort", observed=True, sort=True):
            pairs = ";".join(sorted(rows["pair"].astype(str).unique()))
            row_support = (
                support_rows
                if support_rows is not None
                else int(pd.to_numeric(rows["rows"], errors="raise").sum())
            )
            is_control = spec["roles"] == "negative_control"
            price_fields = (
                "not_applicable_context_control"
                if spec["source_family"] == "generic_multitimeframe_indicator_context"
                else "derivable_not_materialized"
            )
            explicit_sides = any(
                token in spec["coordinate_type"]
                for token in ("upper", "lower", "event_high", "event_low", "period_high")
            )
            side_state = "available" if explicit_sides else price_fields
            for episode_type in ("source_event", "downstream_confirmation"):
                decision_rule = (
                    "map BTC/ETH and the event target coin at the source-event decision time"
                    if episode_type == "source_event"
                    else "map each follower coin after the frozen BTC confirmation becomes observable"
                )
                output.append(
                    _row(
                        "local_multitimeframe_support_resistance_map",
                        episode_type=episode_type,
                        role="timestamp_safe_local_modifier_and_context_map",
                        decision_time_rule=decision_rule,
                        source_sibling=spec["contract"],
                        cohort=str(cohort),
                        pair=pairs,
                        horizon="decision_time_context_only",
                        independent_episode_count=0,
                        actual_count=0,
                        control_count=0,
                        source_timestamp_quality="outcome_blind_causal_support_freeze",
                        latest_contributing_data_rule=spec["settlement"],
                        eligible=True,
                        plain_limitation=(
                            f"{prior_evidence['limitation']} Coverage is ready only for a later "
                            "event/confirmation-time materializer; no new outcome scoring or "
                            "decision-anchor join was run."
                        ),
                        source_family=spec["source_family"],
                        coordinate_type=spec["coordinate_type"],
                        source_timeframes=spec["source_timeframes"],
                        lookback_or_session_basis=spec["basis"],
                        availability_settlement_rule=spec["settlement"],
                        above_price_resistance=(
                            "control_only_not_promoted" if is_control else side_state
                        ),
                        below_price_support=(
                            "control_only_not_promoted" if is_control else side_state
                        ),
                        nearest_distance=price_fields,
                        room_to_opposing_level=price_fields,
                        approach_side=price_fields,
                        age_and_prior_touches="derivable_not_materialized",
                        broken_reclaimed_rejected_state="derivable_not_materialized",
                        independent_cluster_membership=spec["cluster"],
                        intended_level_role=spec["roles"],
                        prior_evidence_class=prior_evidence["class"],
                        materialization_status="coverage_ready_for_materialization",
                        source_support_row_count=row_support,
                    )
                )
    return output


def _load_and_build() -> tuple[DataFrame, list[dict[str, Any]]]:
    results, contracts = _load_json_sources()
    level_map_sources = _load_level_map_contracts(contracts)

    macro_path = _verified_child(
        contracts,
        "macro_direction_calls",
        results["macro_expectation_direction"]["artifacts"]["direction_rows"],
    )
    macro = read_parquet_projected(
        macro_path,
        [
            "route_id",
            "expectation_episode_id",
            "source_block",
            "anchor_utc",
            "pair",
            "horizon_minutes",
            "predicted_direction",
            "abstention_reason",
            "ordinary_control_count",
        ],
    )

    layer2 = results["layer2_individual_links"]
    layer2_freeze_path = _verified_child(contracts, "layer2_freeze", layer2["freeze_contract"])
    layer2_root = layer2_freeze_path.parent
    event_path = layer2_root / "layer2_event_catalog.csv"
    control_path = layer2_root / "layer2_control_map.csv"
    for name, path in (("layer2_events", event_path), ("layer2_controls", control_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{name} missing: {path}")
        contracts.append({"name": name, **_path_sha(path)})
    layer2_events = pd.read_csv(event_path)
    layer2_controls = pd.read_csv(control_path)
    background_path = _verified_child(
        contracts, "layer2_background", layer2["detail_artifacts"]["background_details"]
    )
    background = read_parquet_projected(
        background_path,
        [
            "event_id",
            "event_source",
            "whole_event_partition",
            "sample_type",
            "pre_return_30d",
            "background",
            "positive_initial_move",
        ],
    )

    independent_freeze = results["independent_context_freeze"]
    independent_direct = results["independent_context_direct"]
    independent_events_path = _verified_child(
        contracts, "independent_events", independent_freeze["artifacts"]["events"]
    )
    independent_controls_path = _verified_child(
        contracts, "independent_controls", independent_freeze["artifacts"]["controls"]
    )
    verdict_path = _verified_child(
        contracts, "independent_verdicts", independent_direct["artifacts"]["verdicts"]
    )
    independent_events = pd.read_csv(independent_events_path)
    independent_controls = pd.read_csv(independent_controls_path)
    verdicts = pd.read_csv(verdict_path)

    simple = results["simple_signal_families"]
    _verified_child(contracts, "simple_signal_freeze", simple["freeze_contract"])
    calls_path = _verified_child(
        contracts, "simple_signal_calls", simple["artifacts"]["calls_on_d"]
    )
    calls = read_parquet_projected(
        calls_path,
        [
            "sample_kind",
            "analysis_unit_id",
            "model_anchor_utc",
            "model_period",
            "pair",
            "signal_id",
            "observation_utc",
            "latest_contributing_data_utc",
        ],
    )

    meme = results["meme_transmission"]
    _verified_child(contracts, "meme_transmission_freeze", meme["freeze_contract"])
    group_path = _verified_child(
        contracts, "meme_transmission_groups", meme["artifacts"]["group_details"]
    )
    groups = read_parquet_projected(
        group_path,
        [
            "sample_kind",
            "episode_id",
            "model_anchor_utc",
            "model_period",
            "btc_confirmed_reaction",
            "horizon_hours",
            "cohort",
        ],
    )

    rows = []
    rows.extend(build_cpi_coverage(macro))
    rows.extend(build_fomc_and_background_coverage(layer2_events, layer2_controls, background))
    rows.extend(
        build_independent_context_coverage(independent_events, independent_controls, verdicts)
    )
    rows.extend(build_recent_volume_coverage(calls))
    rows.extend(build_transmission_coverage(groups))
    rows.extend(build_level_coverage(results["g18_level_confirmation"]["inventory"], contracts))
    rows.extend(build_local_map_coverage(level_map_sources))
    coverage = pd.DataFrame(rows, columns=COVERAGE_COLUMNS).sort_values(
        ["family_id", "prototype_id", "source_sibling", "cohort", "pair", "horizon"],
        kind="stable",
    )
    return coverage.reset_index(drop=True), contracts


def validate_coverage(coverage: DataFrame) -> None:
    if list(coverage.columns) != COVERAGE_COLUMNS:
        raise ValueError("Coverage schema drift")
    registry_ids = {item["prototype_id"] for item in PROTOTYPES}
    if len(PROTOTYPES) != 9 or len({item["family_id"] for item in PROTOTYPES}) != 5:
        raise ValueError("Frozen registry must contain exactly 9 prototypes in 5 families")
    if set(coverage["prototype_id"]) != registry_ids:
        raise ValueError("Coverage does not contain the exact frozen prototype registry")
    identity = [
        "prototype_id",
        "episode_type",
        "source_sibling",
        "source_family",
        "cohort",
        "pair",
        "horizon",
    ]
    if coverage.duplicated(identity).any():
        raise ValueError("Duplicate coverage identity")
    if (pd.to_numeric(coverage["independent_episode_count"], errors="raise") < 0).any():
        raise ValueError("Negative coverage count")


def _freeze_payload(contracts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "status": "frozen_outcome_blind_stage1_portfolio_preflight",
        "purpose": "Reconstruct five reason-labelled offline prototype families at their honest decision times before any new outcome scoring.",
        "baseline": "Newest retained or conditional evidence already frozen in its authoritative source result.",
        "hypotheses": [
            {
                "family_id": p["family_id"],
                "prototype_id": p["prototype_id"],
                "role": p["role"],
                "decision_time_rule": p["decision_time_rule"],
            }
            for p in PROTOTYPES
        ],
        "family_count": 5,
        "prototype_count": 9,
        "episode_type_separation": [
            "source_event",
            "level_contact",
            "downstream_confirmation",
        ],
        "accepted_facts_not_retested": [
            "CPI headline/core agreement direction is event scoped and requires timestamp-safe actuals and expectations.",
            "FOMC and retained live news/web routes are unsigned activity warnings.",
            "Recent volume is readiness/persistence and is not causal attribution.",
            "Single calculated levels and independent clusters remain distinct reaction-location representations.",
            "The local support/resistance map is a timestamp-safe conditional context map, not a standalone direction claim.",
            "Generic G17 moving-average and Bollinger rows remain negative controls and are not promoted levels.",
            "BTC confirmation must be observed before downstream group transmission begins.",
            "BTC dominance is relative meme rotation, not absolute market direction.",
        ],
        "later_outcome_controls": {
            "major_events": "retain each source family's frozen matched-event controls",
            "background": "compare event-conditioned downstream confirmations with the frozen source sibling controls",
            "calculated_areas": "retain real contacts and the G17 source-defined contact controls without pooling cohorts",
            "recent_volume": "compare with the frozen no-call or matched-anchor baseline",
            "cross_asset": "compare the same parent episodes with frozen controls and simpler BTC-only confirmation",
        },
        "threshold_policy": "Inherit only exact thresholds from each frozen source result or freeze; this preflight invents none.",
        "decision_meanings": {
            "pass": "the prototype is reconstructable at its honest decision time",
            "revise": "a named source representation needs a separately frozen correction before outcomes",
            "park": "required timestamp-safe or pre-outcome input is unavailable",
        },
        "scope_guards": {
            "market_outcomes_read": False,
            "profit_used": False,
            "trades_tested": False,
            "orders_created": False,
            "freqai_used": False,
            "new_model_trained": False,
            "target_outcome_columns_permitted": False,
        },
        "source_artifact_contracts": sorted(contracts, key=lambda item: item["name"]),
        "analysis_script": _path_sha(Path(__file__)),
        "planned_outputs": sorted(OUTPUT_NAMES),
    }


def _result_payload(coverage: DataFrame) -> dict[str, Any]:
    eligible = coverage.groupby("prototype_id", observed=True)["eligible"].any().to_dict()
    eligible_ids = sorted(key for key, value in eligible.items() if bool(value))
    blocked_ids = sorted(key for key, value in eligible.items() if not bool(value))
    reasons: dict[str, list[str]] = {}
    for prototype_id in blocked_ids:
        values = coverage.loc[
            coverage["prototype_id"].eq(prototype_id), "missing_required_inputs"
        ].dropna()
        reasons[prototype_id] = sorted({str(value) for value in values if str(value)})
    return {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "status": "stage1_outcome_blind_coverage_preflight_complete",
        "family_count": 5,
        "prototype_count": 9,
        "coverage_row_count": len(coverage),
        "eligible_prototype_count": len(eligible_ids),
        "blocked_prototype_count": len(blocked_ids),
        "eligible_prototype_ids": eligible_ids,
        "blocked_prototype_ids": blocked_ids,
        "blocked_reasons": reasons,
        "scope_guards": {
            "market_outcomes_read": False,
            "profit_used": False,
            "trades_tested": False,
            "orders_created": False,
            "freqai_used": False,
        },
        "artifacts": {
            "freeze": _path_sha(FREEZE_PATH),
            "coverage": _path_sha(COVERAGE_PATH),
        },
        "expected_output_files": sorted(OUTPUT_NAMES),
    }


def validate_existing_result() -> dict[str, Any]:
    if not OUTPUT_ROOT.is_dir():
        raise FileNotFoundError(OUTPUT_ROOT)
    names = {path.name for path in OUTPUT_ROOT.iterdir() if path.is_file()}
    if names != OUTPUT_NAMES:
        raise ValueError(f"Output folder contents drift: {sorted(names)}")
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    if result.get("family_count") != 5 or result.get("prototype_count") != 9:
        raise ValueError("Existing result registry counts drift")
    if result.get("expected_output_files") != sorted(OUTPUT_NAMES):
        raise ValueError("Existing result output contract drift")
    expected_paths = {"freeze": FREEZE_PATH.resolve(), "coverage": COVERAGE_PATH.resolve()}
    if set(result.get("artifacts", {})) != set(expected_paths):
        raise ValueError("Existing result artifact keys drift")
    for key, expected_path in expected_paths.items():
        artifact = result["artifacts"][key]
        if Path(artifact["path"]).resolve() != expected_path:
            raise ValueError(f"Existing {key} artifact path drift")
        _verify_contract(f"existing_{key}", artifact)
    coverage = pd.read_csv(COVERAGE_PATH)
    validate_coverage(coverage)
    if len(coverage) != result.get("coverage_row_count"):
        raise ValueError("Existing result coverage count drift")
    return result


def run_preflight(*, overwrite: bool = False) -> dict[str, Any]:
    present = [path for path in (FREEZE_PATH, COVERAGE_PATH, RESULT_PATH) if path.exists()]
    if present and not overwrite:
        if len(present) != 3:
            raise FileExistsError(f"Partial Stage-1 output exists: {present}")
        return validate_existing_result()
    coverage, contracts = _load_and_build()
    validate_coverage(coverage)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(_freeze_payload(contracts), FREEZE_PATH)
    g0.atomic_write_csv(coverage, COVERAGE_PATH)
    g0.atomic_write_json(_result_payload(coverage), RESULT_PATH)
    return validate_existing_result()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run_preflight(overwrite=args.overwrite)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

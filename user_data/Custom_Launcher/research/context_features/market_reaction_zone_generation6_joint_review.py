from __future__ import annotations

# Bound numerical libraries before importing pandas helpers.
# The frozen portfolio contains long trader-readable prose literals by design.
# ruff: noqa: E402, E501
import os


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation6 as g6f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


OUTPUT_ROOT = (
    REPO_ROOT / "user_data" / "research_news_data" / "context_features" / "market_reaction_zones"
)
FREQAI_ROOT = OUTPUT_ROOT / "generation6_branches" / "g6_freqai_source_ladders"
DIRECT_ROOT = OUTPUT_ROOT / "generation6_branches" / "g6_direct_screen" / "g6_direct_full_20260821a"
NORMAL_RUN_ID = "g6_freqai_full_normal_20260821a"
MEME_RUN_ID = "g6_freqai_full_meme_20260821a"
REVIEW_ROOT = OUTPUT_ROOT / "generation6_review"
DEFAULT_REVIEW_ID = "g6_joint_review_20260821a"
FROZEN_G7_PATH = REVIEW_ROOT / "g7_frozen_pairwise_batch.json"

STRONG_STATUS = "control_resistant_incremental_freqai_lead"
PROVISIONAL_STATUS = "provisional_incremental_freqai_lead"
FAILED_STATUS = "not_reproduced_by_freqai"
INTERACTION_SUFFIXES = (
    "level_adds_to_source",
    "source_adds_to_level",
    "combined_beats_stale_source",
    "combined_beats_stale_level",
)
CORE_COMPARISONS = {
    "calculated_areas": (
        "calculated_areas__level_adds_to_market",
        "calculated_areas__current_beats_stale_level",
        "calculated_areas__level_only_beats_stale",
    ),
    "recent_market_state": (
        "ohlcv_state__market_adds_to_level",
        "ohlcv_state__current_beats_stale_state",
        "ohlcv_state__state_only_beats_stale",
    ),
}
TARGET_PLAIN_NAMES = {
    g6f.TARGET_COLUMNS[0]: "next-hour volume relative to its prior baseline",
    g6f.TARGET_COLUMNS[1]: "next-hour range relative to its prior baseline",
    g6f.TARGET_COLUMNS[2]: "largest absolute four-hour movement in prior ATR units",
    g6f.TARGET_COLUMNS[3]: "magnitude of the next-hour pressure change",
    g6f.TARGET_COLUMNS[4]: "fraction of the next four closes dwelling in the zone",
}


def true_mask(values: pd.Series) -> pd.Series:
    """Read persisted booleans without treating the string ``False`` as truthy."""
    return values.eq(True) | values.astype(str).str.casefold().eq("true")


def terminal_run_paths(run_id: str) -> dict[str, Path]:
    root = FREQAI_ROOT / run_id
    return {
        "root": root,
        "result": root / "g6_freqai_result.json",
        "decisions": root / "g6_freqai_comparison_decisions.csv",
        "pair_scores": root / "g6_freqai_pair_scores.csv",
        "groups": root / "g6_freqai_group_transfer.csv",
    }


def load_terminal_run(run_id: str, expected_cohort: str) -> dict[str, Any]:
    paths = terminal_run_paths(run_id)
    missing = [str(path) for key, path in paths.items() if key != "root" and not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete Generation 6 review source: {missing}")
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    integrity = result.get("integrity", {})
    problems: list[str] = []
    if result.get("status") != "completed_freqai_source_ladders":
        problems.append(f"status={result.get('status')!r}")
    if result.get("cohort") != expected_cohort:
        problems.append(f"cohort={result.get('cohort')!r}")
    if result.get("profiles_completed") != 84:
        problems.append(f"profiles={result.get('profiles_completed')!r}")
    if result.get("comparisons_completed") != 72:
        problems.append(f"comparisons={result.get('comparisons_completed')!r}")
    if integrity.get("unequal_prediction_key_comparisons") != 0:
        problems.append("unequal prediction keys")
    if integrity.get("duplicate_prediction_rows_removed") != 0:
        problems.append("duplicate prediction rows")
    if integrity.get("profit_used") is not False:
        problems.append("profit target was used")
    if integrity.get("direction_prediction") is not False:
        problems.append("direction target was used")
    if integrity.get("all_profile_commands_terminal") is not True:
        problems.append("non-terminal profile command")
    if problems:
        raise ValueError(f"Generation 6 run {run_id} failed joint-review integrity: {problems}")
    return {
        "run_id": run_id,
        "cohort": expected_cohort,
        "paths": paths,
        "result": result,
        "decisions": pd.read_csv(paths["decisions"]),
        "pair_scores": pd.read_csv(paths["pair_scores"]),
        "groups": pd.read_csv(paths["groups"]),
    }


def comparison_statuses(
    decisions: DataFrame,
    *,
    source: str,
    target: str,
    suffixes: Sequence[str] = INTERACTION_SUFFIXES,
) -> dict[str, str]:
    expected = {f"{source}__{suffix}": suffix for suffix in suffixes}
    selected = decisions.loc[
        decisions["comparison_id"].isin(expected) & decisions["target"].eq(target)
    ]
    if len(selected) != len(expected):
        raise ValueError(
            f"Expected {len(expected)} decisions for {source}/{target}; found {len(selected)}"
        )
    return {
        expected[str(row.comparison_id)]: str(row.status)
        for row in selected.itertuples(index=False)
    }


def classify_required_comparisons(statuses: dict[str, str]) -> str:
    values = tuple(statuses.values())
    strong = values.count(STRONG_STATUS)
    provisional = values.count(PROVISIONAL_STATUS)
    failed = values.count(FAILED_STATUS)
    if strong == len(values):
        return "control_resistant_against_both_components_and_both_stale_controls"
    if strong == len(values) - 1 and provisional == 1 and failed == 0:
        return "near_control_resistant_one_uncertainty_check_short"
    if strong >= 2:
        return "partial_or_redundant_did_not_beat_every_component"
    return "not_reproduced_beyond_both_components"


def period_effects(
    pair_scores: DataFrame,
    *,
    comparison_id: str,
    target: str,
) -> list[dict[str, Any]]:
    selected = pair_scores.loc[
        pair_scores["comparison_id"].eq(comparison_id) & pair_scores["target"].eq(target)
    ].copy()
    rows: list[dict[str, Any]] = []
    for period, group in selected.groupby("period", sort=False):
        relative = pd.to_numeric(group["relative_mae_gain"], errors="coerce").dropna()
        raw = pd.to_numeric(group["paired_mae_gain"], errors="coerce").dropna()
        rows.append(
            {
                "period": str(period),
                "coins": len(relative),
                "positive_coins": int(relative.gt(0.0).sum()),
                "mean_relative_error_reduction_pct": (
                    float(relative.mean() * 100.0) if len(relative) else None
                ),
                "mean_raw_error_reduction": float(raw.mean()) if len(raw) else None,
            }
        )
    return rows


def source_target_rows(run: dict[str, Any]) -> DataFrame:
    records: list[dict[str, Any]] = []
    decisions = run["decisions"]
    pair_scores = run["pair_scores"]
    for source, definition in g6f.SOURCE_BLOCKS.items():
        for target in g6f.TARGET_COLUMNS:
            statuses = comparison_statuses(
                decisions,
                source=source,
                target=target,
            )
            values = tuple(statuses.values())
            source_only = decisions.loc[
                decisions["comparison_id"].eq(f"{source}__source_only_beats_stale")
                & decisions["target"].eq(target),
                "status",
            ]
            source_market = decisions.loc[
                decisions["comparison_id"].eq(f"{source}__source_vs_market")
                & decisions["target"].eq(target),
                "status",
            ]
            records.append(
                {
                    "row_type": "source_plus_level",
                    "cohort": run["cohort"],
                    "lane": str(definition["lane"]),
                    "source": source,
                    "target": target,
                    "target_plain_name": TARGET_PLAIN_NAMES[target],
                    "required_comparison_class": classify_required_comparisons(statuses),
                    "strong_required_comparisons": values.count(STRONG_STATUS),
                    "provisional_required_comparisons": values.count(PROVISIONAL_STATUS),
                    "failed_required_comparisons": values.count(FAILED_STATUS),
                    "required_comparison_statuses": json.dumps(statuses, sort_keys=True),
                    "source_only_beats_stale": (
                        str(source_only.iloc[0]) if len(source_only) == 1 else "missing"
                    ),
                    "source_only_beats_market_state": (
                        str(source_market.iloc[0]) if len(source_market) == 1 else "missing"
                    ),
                    "source_adds_to_level_effects": json.dumps(
                        period_effects(
                            pair_scores,
                            comparison_id=f"{source}__source_adds_to_level",
                            target=target,
                        ),
                        sort_keys=True,
                    ),
                    "level_adds_to_source_effects": json.dumps(
                        period_effects(
                            pair_scores,
                            comparison_id=f"{source}__level_adds_to_source",
                            target=target,
                        ),
                        sort_keys=True,
                    ),
                }
            )
    for family, comparisons in CORE_COMPARISONS.items():
        for target in g6f.TARGET_COLUMNS:
            selected = decisions.loc[
                decisions["comparison_id"].isin(comparisons) & decisions["target"].eq(target)
            ]
            if len(selected) != len(comparisons):
                raise ValueError(f"Incomplete core review for {family}/{target}")
            statuses = {
                str(row.comparison_id): str(row.status) for row in selected.itertuples(index=False)
            }
            values = tuple(statuses.values())
            first_comparison = comparisons[0]
            records.append(
                {
                    "row_type": "core_component",
                    "cohort": run["cohort"],
                    "lane": family,
                    "source": family,
                    "target": target,
                    "target_plain_name": TARGET_PLAIN_NAMES[target],
                    "required_comparison_class": classify_required_comparisons(statuses),
                    "strong_required_comparisons": values.count(STRONG_STATUS),
                    "provisional_required_comparisons": values.count(PROVISIONAL_STATUS),
                    "failed_required_comparisons": values.count(FAILED_STATUS),
                    "required_comparison_statuses": json.dumps(statuses, sort_keys=True),
                    "source_only_beats_stale": "not_applicable_core_component",
                    "source_only_beats_market_state": "not_applicable_core_component",
                    "source_adds_to_level_effects": json.dumps(
                        period_effects(
                            pair_scores,
                            comparison_id=first_comparison,
                            target=target,
                        ),
                        sort_keys=True,
                    ),
                    "level_adds_to_source_effects": "[]",
                }
            )
    return DataFrame.from_records(records)


def cross_cohort_rows(source_rows: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    for (row_type, source, target), group in source_rows.groupby(
        ["row_type", "source", "target"], sort=True
    ):
        classes = dict(zip(group["cohort"], group["required_comparison_class"], strict=True))
        strict = {
            cohort
            for cohort, value in classes.items()
            if value == "control_resistant_against_both_components_and_both_stale_controls"
        }
        near = {
            cohort
            for cohort, value in classes.items()
            if value == "near_control_resistant_one_uncertainty_check_short"
        }
        if strict == {"normal", "meme"}:
            classification = "broad_repeatable_normal_and_meme"
        elif len(strict) == 1 and len(near) == 1:
            classification = "broad_lead_one_cohort_strict_one_near"
        elif strict == {"normal"}:
            classification = "normal_cohort_only_at_current_standard"
        elif strict == {"meme"}:
            classification = "meme_cohort_only_at_current_standard"
        else:
            classification = "not_jointly_retained"
        records.append(
            {
                "row_type": row_type,
                "source": source,
                "target": target,
                "target_plain_name": TARGET_PLAIN_NAMES[str(target)],
                "cross_cohort_classification": classification,
                "normal_class": classes.get("normal", "missing"),
                "meme_class": classes.get("meme", "missing"),
            }
        )
    return DataFrame.from_records(records)


def group_interaction_rows(run: dict[str, Any]) -> DataFrame:
    frame = run["groups"].copy()
    records: list[dict[str, Any]] = []
    for group_id in sorted(frame["group_id"].unique()):
        for source in g6f.SOURCE_BLOCKS:
            for target in g6f.TARGET_COLUMNS:
                comparison_ids = {f"{source}__{suffix}" for suffix in INTERACTION_SUFFIXES}
                selected = frame.loc[
                    frame["group_id"].eq(group_id)
                    & frame["comparison_id"].isin(comparison_ids)
                    & frame["target"].eq(target)
                ]
                if len(selected) != len(comparison_ids):
                    raise ValueError(
                        f"Incomplete group interaction rows for {group_id}/{source}/{target}"
                    )
                retained = int(true_mask(selected["retained"]).sum())
                records.append(
                    {
                        "cohort": run["cohort"],
                        "group_id": group_id,
                        "source": source,
                        "target": target,
                        "retained_required_comparisons": retained,
                        "classification": (
                            "provisional_group_pattern_all_four_point_checks"
                            if retained == 4
                            else "not_retained_across_all_group_point_checks"
                        ),
                        "uncertainty_boundary": (
                            "Group rows are secondary point-estimate checks; the full-cohort "
                            "bootstrap decision remains the stronger standard."
                        ),
                    }
                )
    return DataFrame.from_records(records)


def direct_overlap_summary(
    path: Path,
    *,
    keys: Sequence[str],
) -> dict[str, Any]:
    frame = pd.read_csv(path)
    candidates = frame.loc[true_mask(frame["candidate"])].copy()
    common: list[dict[str, Any]] = []
    for key, group in candidates.groupby(list(keys), dropna=False, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        cohorts = set(group["cohort"].astype(str))
        signs = set(pd.to_numeric(group["behaviour_sign"], errors="coerce").dropna())
        if cohorts == {"normal", "meme"} and len(signs) == 1:
            common.append(
                {
                    **dict(zip(keys, values, strict=True)),
                    "behaviour_sign": float(next(iter(signs))),
                }
            )
    return {
        "path": str(path),
        "sha256": g0.sha256_file(path),
        "candidate_rows": len(candidates),
        "cross_cohort_same_sign_cells": len(common),
        "cross_cohort_same_sign_examples": common[:50],
    }


def direct_review() -> dict[str, Any]:
    result_path = DIRECT_ROOT / "g6_direct_screen_result.json"
    decisions_path = DIRECT_ROOT / "g6_direct_screen_branch_decisions.csv"
    if not result_path.is_file() or not decisions_path.is_file():
        raise FileNotFoundError("The complete seven-sibling direct screen is missing.")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed_all_seven_sibling_direct_screens":
        raise ValueError("The direct Generation 6 portfolio is not terminal.")
    summary = result.get("summary", {})
    if summary.get("branches_completed") != 7:
        raise ValueError("The direct Generation 6 review must contain seven siblings.")
    if summary.get("profit_used") is not False or summary.get("direction_prediction") is not False:
        raise ValueError("The direct screen crossed the profit/direction boundary.")
    branches = pd.read_csv(decisions_path)
    return {
        "result_path": str(result_path),
        "result_sha256": g0.sha256_file(result_path),
        "branch_decisions_path": str(decisions_path),
        "branch_decisions_sha256": g0.sha256_file(decisions_path),
        "branches": branches.to_dict(orient="records"),
        "overlap": {
            "calculated_areas": direct_overlap_summary(
                DIRECT_ROOT / "g6a_calculated_area_candidates.csv",
                keys=("level_family", "source_timeframe", "metric"),
            ),
            "timeframe_relationships": direct_overlap_summary(
                DIRECT_ROOT / "g6b_timeframe_relationship_candidates.csv",
                keys=("higher_timeframe", "relationship", "metric"),
            ),
            "ohlcv_and_indicators": direct_overlap_summary(
                DIRECT_ROOT / "g6c_ohlcv_indicator_candidates.csv",
                keys=("feature", "level_family", "metric"),
            ),
            "cross_market": direct_overlap_summary(
                DIRECT_ROOT / "g6d_cross_market_candidates.csv",
                keys=("feature", "level_family", "metric"),
            ),
            "orderbook": direct_overlap_summary(
                DIRECT_ROOT / "g6e_orderbook_candidates.csv",
                keys=("feature", "level_family", "metric"),
            ),
            "news": direct_overlap_summary(
                DIRECT_ROOT / "g6f_news_context_candidates.csv",
                keys=("feature", "level_family", "metric"),
            ),
        },
    }


def required_profiles() -> list[str]:
    return [
        "level_only",
        "level_plus_component_a",
        "level_plus_component_b",
        "component_a_plus_component_b_without_level",
        "level_plus_both_components",
        "stale_level_plus_both_components",
        "level_plus_stale_component_a_plus_component_b",
        "level_plus_component_a_plus_stale_component_b",
    ]


def branch(
    branch_id: str,
    plain_name: str,
    *,
    component_a: str,
    component_b: str,
    scope: str,
    targets: Sequence[str],
    hypothesis: str,
    parent_evidence: Sequence[str],
    exception: str | None = None,
) -> dict[str, Any]:
    return {
        "id": branch_id,
        "plain_name": plain_name,
        "status": "frozen_for_outcome_blind_preflight",
        "component_a": component_a,
        "component_b": component_b,
        "scope": scope,
        "targets": list(targets),
        "hypothesis": hypothesis,
        "parent_evidence": list(parent_evidence),
        "weak_component_exception": exception,
        "profile_ladder": required_profiles(),
        "required_controls": [
            "identical eligible pair-date rows for every profile in the branch",
            "strict chronological development and two validation periods",
            "current ingredient versus a causal copy at least 72 hours old",
            "current level versus stale or false location",
            "complete model versus level-only and each one-component model",
            "complete model versus the two context components without level",
            "equal-coin summary, per-coin result, weekly-block uncertainty, and frozen groups",
            "independent contacts separated beyond the longest target horizon",
        ],
        "decision_rule": (
            "For a named target, the complete interaction must reduce error in both "
            "validation periods versus level-only, each level-plus-one-component model, "
            "the two components without a level, and every applicable stale control. "
            "The uncertainty lower bound must remain above zero, the frozen group coin "
            "rule must pass, and no one coin may carry the conclusion. A branch can retain "
            "one target and park another; targets are never averaged into a pass."
        ),
    }


def frozen_generation7(review_path: Path) -> dict[str, Any]:
    volume = g6f.TARGET_COLUMNS[0]
    range_target = g6f.TARGET_COLUMNS[1]
    excursion = g6f.TARGET_COLUMNS[2]
    dwell = g6f.TARGET_COLUMNS[4]
    branches = [
        branch(
            "g7a_local_participation_and_btc_activity",
            "Does local volume/pressure plus BTC activity improve reaction estimates at a level?",
            component_a="ohlcv_volume_pressure",
            component_b="cross_market_btc",
            scope="all supported levels; normal and meme cohorts remain separate",
            targets=(volume, range_target),
            hypothesis=(
                "Local participation describes the coin's immediate readiness to move, "
                "while BTC activity supplies broader crypto force and the level supplies location."
            ),
            parent_evidence=(
                "Both local volume/pressure and BTC state survived all four component/stale tests for next-hour volume in normal and meme cohorts.",
                "BTC state also retained range evidence in normal coins and was near the strict standard in memes.",
            ),
        ),
        branch(
            "g7b_local_participation_and_volatility",
            "Does local volume/pressure plus volatility or compression improve reaction estimates?",
            component_a="ohlcv_volume_pressure",
            component_b="ohlcv_volatility_range",
            scope="all supported levels; normal and meme cohorts remain separate",
            targets=(volume, range_target, excursion),
            hypothesis=(
                "Participation and available movement capacity are complementary: volume/pressure "
                "describes engagement while volatility/compression describes how much movement is feasible."
            ),
            parent_evidence=(
                "Volume/pressure was strict for volume in both cohorts and for range in normal coins.",
                "Volatility/range was strict for normal volume/range and near strict for meme volume.",
            ),
        ),
        branch(
            "g7c_local_participation_and_trend_momentum",
            "Does local volume/pressure plus trend strength or momentum improve reaction estimates?",
            component_a="ohlcv_volume_pressure",
            component_b="ohlcv_trend_momentum",
            scope="all supported levels with normal and meme results reported separately",
            targets=(volume, range_target),
            hypothesis=(
                "Participation may matter differently when recent movement has persistent strength; "
                "the test remains direction-neutral by using magnitudes rather than up/down signs."
            ),
            parent_evidence=(
                "Trend/momentum was strict for normal range and meme volume, while local volume/pressure retained volume in both cohorts.",
                "The differing target support makes this a conditional interaction question, not a universal trend claim.",
            ),
        ),
        branch(
            "g7d_cross_timeframe_agreement_and_local_participation",
            "Does genuine cross-timeframe agreement become useful when local participation is changing?",
            component_a="different_mechanism_cross_timeframe_agreement",
            component_b="ohlcv_volume_pressure",
            scope=(
                "4h, 8h, and 1d different-mechanism agreements; compare with isolated "
                "components and width/contact-frequency-matched controls"
            ),
            targets=(volume, range_target, excursion),
            hypothesis=(
                "Independent higher-timeframe coordinates may identify a meaningful location, "
                "but only local participation can distinguish an active encounter from an inert one."
            ),
            parent_evidence=(
                "Direct tests repeated positive volume/range/excursion differences for different-mechanism agreement at 4h, 8h, and 1d in both cohorts.",
                "The broad timeframe FreqAI block did not add beyond every component, so width and explicit dependency controls are mandatory rather than assuming precedence.",
            ),
        ),
        branch(
            "g7e_prior_range_level_and_local_participation",
            "Do prior session/week/month range references interact with local participation?",
            component_a="generic_prior_range_level_geometry",
            component_b="ohlcv_volume_pressure",
            scope="generic prior-range levels by source timeframe, normal and meme cohorts separate",
            targets=(volume, range_target, excursion),
            hypothesis=(
                "Widely visible prior extremes provide a causal location and local participation "
                "determines whether contact produces unusual activity."
            ),
            parent_evidence=(
                "One-hour generic prior-range levels showed positive volume, range, and four-hour excursion differences against random-time, shifted-price, and stale-level controls in both cohorts.",
                "The aggregate level and local participation blocks independently retained prediction value.",
            ),
        ),
        branch(
            "g7f_prior_volume_profile_area_and_compression",
            "Do prior four-hour Volume Profile areas identify repeatable dampening under compression?",
            component_a="volume_profile_explicit_prior_4h_geometry",
            component_b="ohlcv_volatility_range",
            scope="explicit prior 4h Volume Profile areas; normal and meme cohorts separate",
            targets=(range_target, volume, dwell),
            hypothesis=(
                "Some prior accepted-value areas may absorb movement rather than amplify it, "
                "especially when current volatility is compressed; lower reaction magnitude is valid evidence."
            ),
            parent_evidence=(
                "The direct screen found slightly lower next-hour range at prior 4h Volume Profile areas than random-time, shifted, and stale controls in both cohorts.",
                "Volatility/range state retained broad normal evidence and near-strict meme volume evidence.",
            ),
        ),
        branch(
            "g7g_independent_level_family_cluster",
            "Do two independent calculated level families form a reaction cluster beyond either level alone?",
            component_a="first_causal_level_family",
            component_b="second_dependency_independent_level_family",
            scope=(
                "different-mechanism two-family clusters with component identity, distance, "
                "zone width, and contact frequency frozen; singles and clusters remain co-equal"
            ),
            targets=(volume, range_target, excursion, dwell),
            hypothesis=(
                "Two genuinely independent calculations near the same price may concentrate attention "
                "or liquidity in a way that neither coordinate supplies alone."
            ),
            parent_evidence=(
                "Direct cross-timeframe and market-group screens repeatedly separated different-mechanism clusters from isolated levels for volume, range, and excursion.",
                "Generation 6 aggregate level geometry retained all non-pressure reaction targets in both cohorts.",
            ),
        ),
        branch(
            "g7h_meme_eth_activity_and_local_participation",
            "For meme coins, does ETH activity add a distinct condition beyond local participation?",
            component_a="ohlcv_volume_pressure",
            component_b="cross_market_eth",
            scope="frozen top-ten meme cohort only; DOGE counted once",
            targets=(volume, excursion, dwell),
            hypothesis=(
                "Meme participation may respond to broader alt-market activity not fully represented by BTC, "
                "while local participation and a level retain timing and location."
            ),
            parent_evidence=(
                "The meme group retained all four point-estimate checks for ETH state and four-hour dwell, with near-strict excursion evidence.",
                "This remains a meme-group lead because the full uncertainty standard did not pass all four comparisons.",
            ),
        ),
        branch(
            "g7i_orderbook_and_btc_activity_at_levels",
            "Does historical BTC order-book state matter only when BTC market activity is also elevated?",
            component_a="orderbook_btc",
            component_b="cross_market_btc",
            scope=(
                "BTC pair-local Bybit evidence and BTC-wide context for established altcoins reported "
                "separately; venue labels retained; no imputation to pair-local alt books"
            ),
            targets=(volume, range_target),
            hypothesis=(
                "Displayed liquidity may condition a reaction only when the broader BTC market is active; "
                "averaging across quiet and active BTC states can hide that interaction."
            ),
            parent_evidence=(
                "Order-book direct interactions repeated in several level families, but the broad FreqAI block did not add reliably beyond the level.",
                "BTC activity did retain independent incremental volume/range information, giving one rational conditional repair rather than an unrestricted order-book search.",
            ),
            exception=(
                "One weak-component exception permitted by Objective 02b: order-book state failed as a "
                "broad average and is tested only under the predeclared BTC-activity mechanism."
            ),
        ),
        branch(
            "g7j_gdelt_activity_and_local_participation",
            "Does aggregate information activity matter only when local participation is changing at a level?",
            component_a="news_gdelt",
            component_b="ohlcv_volume_pressure",
            scope=(
                "source-ready aggregate GDELT activity/tone/conflict fields only; missing remains unavailable; "
                "topic, macro, live-news, and web blocks remain parked"
            ),
            targets=(volume, range_target),
            hypothesis=(
                "Information activity may affect market reaction only when it coincides with observable local "
                "participation, so the average GDELT-at-level effect can be near zero."
            ),
            parent_evidence=(
                "Direct GDELT interactions repeated for activity/tone in several families, while the aggregate FreqAI block failed to beat both components consistently.",
                "The test is therefore a bounded interaction challenge, not evidence that GDELT already works.",
            ),
            exception=(
                "Second and final weak-component exception in this frozen portfolio: aggregate GDELT is "
                "paired only with the already-retained local participation mechanism."
            ),
        ),
    ]
    weak_exceptions = [item for item in branches if item["weak_component_exception"]]
    if len(weak_exceptions) > 2:
        raise ValueError("Generation 7 may not expand into unrestricted weak-component fishing.")
    return {
        "schema_version": 1,
        "generation": 7,
        "branch_layer": 7,
        "status": "frozen_before_generation7_reaction_outcomes",
        "frozen_at_utc": g0.utc_now(),
        "authorization": (
            "The user approved the complete breadth-first continuation through Generation 10 "
            "when each layer is justified by the complete preceding joint review."
        ),
        "parent_joint_review": str(review_path),
        "generation_rule": (
            "Run outcome-blind coverage and timestamp checks for all ten siblings, then complete, "
            "honestly park, or reject every sibling before any result launches a descendant."
        ),
        "research_boundary": {
            "profit_optimization": False,
            "direction_prediction": False,
            "entry_or_exit_construction": False,
            "pressure_direction": "Pressure may be an input magnitude; signed future direction remains closed.",
            "single_and_cluster_rule": "Single levels and genuine clusters are co-equal.",
            "headline_count_rule": "Multiple targets, controls, coins, or level families do not become independent edges by being counted separately.",
        },
        "portfolio_summary": {
            "siblings": len(branches),
            "weak_component_exceptions": len(weak_exceptions),
            "retained_route_families": [
                "calculated level geometry",
                "local OHLCV participation",
                "volatility and compression",
                "trend strength and momentum",
                "cross-market BTC or ETH activity",
                "cross-timeframe and independent-family clusters",
                "historical order-book liquidity",
                "timestamp-safe aggregate information activity",
            ],
        },
        "branches": branches,
        "common_stop_rule": (
            "Park a branch if common support fails, the complete model does not beat both one-component "
            "models, stale information reproduces it, signs conflict across validation periods, uncertainty "
            "includes no improvement, or one coin carries the result. Do not tune thresholds after failure."
        ),
        "next_generation_boundary": (
            "Only the complete Generation 7 joint review may freeze detailed attribution or rational "
            "representation refinements for Generation 8."
        ),
    }


def build_review(review_id: str) -> dict[str, Any]:
    direct = direct_review()
    runs = {
        "normal": load_terminal_run(NORMAL_RUN_ID, "normal"),
        "meme": load_terminal_run(MEME_RUN_ID, "meme"),
    }
    source_rows = pd.concat([source_target_rows(run) for run in runs.values()], ignore_index=True)
    cross_rows = cross_cohort_rows(source_rows)
    group_rows = pd.concat(
        [group_interaction_rows(run) for run in runs.values()], ignore_index=True
    )
    review_dir = REVIEW_ROOT / review_id
    review_dir.mkdir(parents=True, exist_ok=True)
    source_path = review_dir / "g6_source_target_review.csv"
    cross_path = review_dir / "g6_cross_cohort_review.csv"
    group_path = review_dir / "g6_group_interaction_review.csv"
    g0.atomic_write_csv(source_rows, source_path)
    g0.atomic_write_csv(cross_rows, cross_path)
    g0.atomic_write_csv(group_rows, group_path)

    broad = cross_rows.loc[
        cross_rows["cross_cohort_classification"].eq("broad_repeatable_normal_and_meme")
    ]
    broad_near = cross_rows.loc[
        cross_rows["cross_cohort_classification"].eq("broad_lead_one_cohort_strict_one_near")
    ]
    group_all_four = group_rows.loc[group_rows["retained_required_comparisons"].eq(4)]
    review_path = review_dir / "g6_joint_review.json"
    frozen = frozen_generation7(review_path)
    g0.atomic_write_json(frozen, FROZEN_G7_PATH)
    result = {
        "schema_version": 1,
        "review_id": review_id,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation6_joint_review_and_frozen_generation7",
        "source_runs": {
            cohort: {
                "run_id": run["run_id"],
                "result_path": str(run["paths"]["result"]),
                "result_sha256": g0.sha256_file(run["paths"]["result"]),
                "profiles": int(run["result"]["profiles_completed"]),
                "comparisons": int(run["result"]["comparisons_completed"]),
            }
            for cohort, run in runs.items()
        },
        "direct_portfolio": direct,
        "joint_counts": {
            "source_target_rows": len(source_rows),
            "broad_control_resistant_rows": len(broad),
            "broad_one_cohort_near_rows": len(broad_near),
            "provisional_group_all_four_rows": len(group_all_four),
        },
        "broad_control_resistant_relationships": broad.to_dict(orient="records"),
        "broad_one_cohort_near_relationships": broad_near.to_dict(orient="records"),
        "plain_findings": [
            (
                "Calculated level geometry repeatedly improved estimates of next-hour volume and range, "
                "four-hour absolute movement, and four-hour zone dwell in normal and meme cohorts. "
                "It did not reliably improve pressure-change magnitude."
            ),
            (
                "Recent OHLCV state added smaller information beyond the level, especially for volume "
                "and range. Local volume/pressure was the most consistent named state block."
            ),
            (
                "BTC activity added a small but repeated increment to level-based volume estimates in "
                "both cohorts and to range estimates most clearly in normal coins."
            ),
            (
                "The direct screens produced many repeated conditional differences, but most timeframe, "
                "news, cohort-relative, ETH, and order-book blocks did not beat both component-only "
                "FreqAI models. They remain redundant, conditional, group-specific, or parked rather "
                "than being counted as independent edges."
            ),
            (
                "Generic prior-range levels were the clearest cross-cohort direct location lead. "
                "Different-mechanism cross-timeframe clusters also repeated, but require explicit "
                "width, contact-frequency, and component-dependency controls."
            ),
            (
                "No result predicts up/down direction, profitability, entry, or exit. The retained "
                "measurements concern reaction magnitude, participation, range, and dwell only."
            ),
        ],
        "artifacts": {
            "source_target_review": g6f.artifact_record(source_path),
            "cross_cohort_review": g6f.artifact_record(cross_path),
            "group_interaction_review": g6f.artifact_record(group_path),
            "frozen_generation7": g6f.artifact_record(FROZEN_G7_PATH),
        },
        "generation7_siblings": len(frozen["branches"]),
        "interpretation_boundary": (
            "Generation 6 supports several small reaction-estimation components, not ten trading "
            "edges. Generation 7 tests whether selected components work better together than alone."
        ),
    }
    g0.atomic_write_json(result, review_path)
    return {**result, "result_path": str(review_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Jointly review all terminal Generation 6 direct/FreqAI siblings and freeze "
            "the balanced Generation 7 pairwise portfolio."
        )
    )
    parser.add_argument("--review-id", default=DEFAULT_REVIEW_ID)
    args = parser.parse_args(argv)
    if not args.review_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in args.review_id
    ):
        raise ValueError(f"Invalid review id: {args.review_id!r}")
    result = build_review(args.review_id)
    print(json.dumps(result, indent=2, default=g0.json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

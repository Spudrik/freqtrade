from __future__ import annotations

# The frozen portfolio contains long trader-readable prose by design.
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

import numpy as np
import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation6_preflight as g6,
)


OUTPUT_ROOT = (
    REPO_ROOT
    / "user_data"
    / "research_news_data"
    / "context_features"
    / "market_reaction_zones"
)
FREQAI_ROOT = (
    OUTPUT_ROOT / "generation7_branches" / "g7_freqai_pairwise_interactions"
)
NORMAL_RUN_ID = "g7_freqai_full_normal_20260821a"
MEME_RUN_ID = "g7_freqai_full_meme_20260821a"
FROZEN_G7 = OUTPUT_ROOT / "generation6_review" / "g7_frozen_pairwise_batch.json"
REVIEW_ROOT = OUTPUT_ROOT / "generation7_review"
DEFAULT_REVIEW_ID = "g7_joint_review_20260821a"
FROZEN_G8_PATH = REVIEW_ROOT / "g8_frozen_attribution_batch.json"

STRICT = "control_resistant_complete_interaction"
PROVISIONAL = "provisional_complete_interaction"
FAILED = "complete_interaction_not_reproduced"
TARGET_PLAIN_NAMES = {
    "&-g6_future_volume_ratio_h1": "next-hour volume relative to its prior baseline",
    "&-g6_future_range_ratio_h1": "next-hour range relative to its prior baseline",
    "&-g6_absolute_excursion_atr_h4": "largest absolute four-hour movement in prior ATR units",
    "&-g6_absolute_pressure_change_h1": "magnitude of the next-hour pressure change",
    "&-g6_dwell_fraction_h4": "fraction of the next four closes dwelling in the zone",
}


def true_mask(values: pd.Series) -> pd.Series:
    return values.eq(True) | values.astype(str).str.casefold().eq("true")


def run_paths(run_id: str) -> dict[str, Path]:
    root = FREQAI_ROOT / run_id
    return {
        "root": root,
        "result": root / "g7_freqai_result.json",
        "decisions": root / "g7_freqai_branch_target_decisions.csv",
        "group_decisions": root / "g7_freqai_group_decisions.csv",
        "group_scores": root / "g7_freqai_group_scores.csv",
        "pair_scores": root / "g7_freqai_pair_scores.csv",
        "eligibility": root / "g7_freqai_comparison_eligibility.csv",
    }


def load_terminal_run(  # noqa: C901 - terminal integrity gates stay explicit
    run_id: str, cohort: str
) -> dict[str, Any]:
    paths = run_paths(run_id)
    missing = [
        str(path)
        for key, path in paths.items()
        if key != "root" and not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Incomplete Generation 7 review source: {missing}")
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    expected_targets = 24 if cohort == "normal" else 25
    problems = []
    if result.get("status") != "completed_generation7_freqai_pairwise_interactions":
        problems.append(f"status={result.get('status')!r}")
    if result.get("cohort") != cohort:
        problems.append(f"cohort={result.get('cohort')!r}")
    if result.get("profiles_completed") != 72:
        problems.append(f"profiles={result.get('profiles_completed')!r}")
    if result.get("branches_completed") != 9:
        problems.append(f"branches={result.get('branches_completed')!r}")
    if result.get("branch_targets_completed") != expected_targets:
        problems.append(f"branch targets={result.get('branch_targets_completed')!r}")
    integrity = result.get("integrity", {})
    if integrity.get("unequal_prediction_key_comparisons") != 0:
        problems.append("unequal prediction keys")
    if integrity.get("duplicate_prediction_rows_removed") != 0:
        problems.append("duplicate prediction rows")
    if integrity.get("profit_used") is not False:
        problems.append("profit used")
    if integrity.get("direction_prediction") is not False:
        problems.append("direction used")
    if integrity.get("all_profile_commands_terminal") is not True:
        problems.append("non-terminal profiles")
    if integrity.get("all_complete_models_challenged_by_seven_controls") is not True:
        problems.append("incomplete seven-control ladder")
    if problems:
        raise ValueError(f"Generation 7 {cohort} integrity failed: {problems}")
    output = {"run_id": run_id, "cohort": cohort, "result": result, "paths": paths}
    for key in ("decisions", "group_decisions", "group_scores", "pair_scores", "eligibility"):
        output[key] = pd.read_csv(paths[key])
    if int(output["eligibility"]["comparison_id"].nunique()) != 63:
        raise ValueError(f"Generation 7 {cohort} does not contain 63 comparisons.")
    if not true_mask(output["eligibility"]["identical_prediction_keys"]).all():
        raise ValueError(f"Generation 7 {cohort} has unequal prediction rows.")
    return output


def status_for(run: dict[str, Any], branch_id: str, target: str) -> dict[str, Any]:
    selected = run["decisions"].loc[
        run["decisions"]["branch_id"].eq(branch_id)
        & run["decisions"]["target"].eq(target)
    ]
    if selected.empty:
        return {
            "status": "not_applicable",
            "strict_groups": "",
            "provisional_groups": "",
        }
    if len(selected) != 1:
        raise ValueError(f"Duplicate branch-target decision for {branch_id}/{target}")
    row = selected.iloc[0]
    return {
        "status": str(row["status"]),
        "strict_groups": "" if pd.isna(row["strict_groups"]) else str(row["strict_groups"]),
        "provisional_groups": (
            "" if pd.isna(row["provisional_groups"]) else str(row["provisional_groups"])
        ),
    }


def best_group_summary(run: dict[str, Any], branch_id: str, target: str) -> dict[str, Any]:
    selected = run["group_scores"].loc[
        run["group_scores"]["branch_id"].eq(branch_id)
        & run["group_scores"]["target"].eq(target)
    ].copy()
    if selected.empty:
        return {
            "group_id": "not_applicable",
            "strict_checks": 0,
            "provisional_checks": 0,
            "checks": 0,
            "minimum_point_gain": np.nan,
            "minimum_lower_bound": np.nan,
        }
    summaries = (
        selected.groupby("group_id", as_index=False, dropna=False)
        .agg(
            strict_checks=("strict_period_pass", lambda values: int(true_mask(values).sum())),
            provisional_checks=(
                "provisional_period_pass",
                lambda values: int(true_mask(values).sum()),
            ),
            checks=("comparison_id", "size"),
            minimum_point_gain=("equal_coin_paired_mae_gain", "min"),
            minimum_lower_bound=("bootstrap_lower", "min"),
        )
        .sort_values(
            ["strict_checks", "provisional_checks", "minimum_point_gain"],
            ascending=False,
        )
    )
    return summaries.iloc[0].to_dict()


def relative_gain_summary(
    run: dict[str, Any], branch_id: str, target: str, group_id: str
) -> dict[str, Any]:
    members: tuple[str, ...]
    if group_id == "frozen_top_ten_memes":
        members = g6.GROUPS[group_id]
    elif group_id == "full_normal_cohort":
        members = tuple(str(pair) for pair in run["pair_scores"]["pair"].unique())
    else:
        members = g6.GROUPS.get(group_id, ())
    selected = run["pair_scores"].loc[
        run["pair_scores"]["branch_id"].eq(branch_id)
        & run["pair_scores"]["target"].eq(target)
        & run["pair_scores"]["pair"].isin(members)
        & run["pair_scores"]["candidate_mae"].notna()
        & run["pair_scores"]["baseline_mae"].notna()
    ].copy()
    if selected.empty:
        return {"minimum_relative_gain": np.nan, "maximum_relative_gain": np.nan}
    grouped = selected.groupby(["comparison_id", "period"], observed=True).agg(
        candidate_mae=("candidate_mae", "mean"),
        baseline_mae=("baseline_mae", "mean"),
    )
    relative = (grouped["baseline_mae"] - grouped["candidate_mae"]) / grouped[
        "baseline_mae"
    ].replace(0.0, np.nan)
    return {
        "minimum_relative_gain": float(relative.min()),
        "maximum_relative_gain": float(relative.max()),
    }


def cross_cohort_classification(normal: str, meme: str) -> str:
    if normal == STRICT and meme == STRICT:
        return "strict_in_both_cohorts"
    if {normal, meme} == {STRICT, PROVISIONAL}:
        return "strict_in_one_and_provisional_in_the_other"
    if normal == PROVISIONAL and meme == PROVISIONAL:
        return "provisional_in_both_cohorts"
    if STRICT in {normal, meme}:
        return "strict_in_one_cohort_only"
    if PROVISIONAL in {normal, meme}:
        return "provisional_in_one_cohort_only"
    if "not_applicable" in {normal, meme}:
        return "single_cohort_not_reproduced_or_not_applicable"
    return "not_reproduced_in_either_cohort"


def joint_rows(normal: dict[str, Any], meme: dict[str, Any]) -> DataFrame:
    frozen = json.loads(FROZEN_G7.read_text(encoding="utf-8"))
    rows = []
    for branch in frozen["branches"]:
        branch_id = str(branch["id"])
        for target in branch["targets"]:
            normal_status = status_for(normal, branch_id, target)
            meme_status = status_for(meme, branch_id, target)
            normal_best = best_group_summary(normal, branch_id, target)
            meme_best = best_group_summary(meme, branch_id, target)
            normal_relative = relative_gain_summary(
                normal, branch_id, target, str(normal_best["group_id"])
            )
            meme_relative = relative_gain_summary(
                meme, branch_id, target, str(meme_best["group_id"])
            )
            rows.append(
                {
                    "branch_id": branch_id,
                    "plain_name": branch["plain_name"],
                    "target": target,
                    "target_plain_name": TARGET_PLAIN_NAMES[target],
                    "normal_status": normal_status["status"],
                    "normal_strict_groups": normal_status["strict_groups"],
                    "normal_provisional_groups": normal_status["provisional_groups"],
                    "normal_best_group": normal_best["group_id"],
                    "normal_strict_control_period_checks": normal_best["strict_checks"],
                    "normal_provisional_control_period_checks": normal_best[
                        "provisional_checks"
                    ],
                    "normal_total_control_period_checks": normal_best["checks"],
                    "normal_minimum_relative_error_reduction": normal_relative[
                        "minimum_relative_gain"
                    ],
                    "normal_maximum_relative_error_reduction": normal_relative[
                        "maximum_relative_gain"
                    ],
                    "meme_status": meme_status["status"],
                    "meme_strict_groups": meme_status["strict_groups"],
                    "meme_provisional_groups": meme_status["provisional_groups"],
                    "meme_best_group": meme_best["group_id"],
                    "meme_strict_control_period_checks": meme_best["strict_checks"],
                    "meme_provisional_control_period_checks": meme_best[
                        "provisional_checks"
                    ],
                    "meme_total_control_period_checks": meme_best["checks"],
                    "meme_minimum_relative_error_reduction": meme_relative[
                        "minimum_relative_gain"
                    ],
                    "meme_maximum_relative_error_reduction": meme_relative[
                        "maximum_relative_gain"
                    ],
                    "joint_classification": cross_cohort_classification(
                        normal_status["status"], meme_status["status"]
                    ),
                }
            )
    return DataFrame.from_records(rows)


def bottleneck_rows(normal: dict[str, Any], meme: dict[str, Any]) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for run in (normal, meme):
        for decision in run["decisions"].itertuples(index=False):
            if decision.status == FAILED:
                continue
            groups = []
            if isinstance(decision.strict_groups, str):
                groups.extend(item for item in decision.strict_groups.split(",") if item)
            if isinstance(decision.provisional_groups, str):
                groups.extend(item for item in decision.provisional_groups.split(",") if item)
            for group_id in dict.fromkeys(groups):
                selected = run["group_scores"].loc[
                    run["group_scores"]["branch_id"].eq(decision.branch_id)
                    & run["group_scores"]["target"].eq(decision.target)
                    & run["group_scores"]["group_id"].eq(group_id)
                ]
                for row in selected.itertuples(index=False):
                    rows.append(
                        {
                            "cohort": run["cohort"],
                            "branch_id": row.branch_id,
                            "plain_name": FROZEN_BRANCH_LOOKUP[row.branch_id]["plain_name"],
                            "target": row.target,
                            "group_id": group_id,
                            "baseline_role": row.baseline_role,
                            "period": row.period,
                            "equal_coin_paired_mae_gain": row.equal_coin_paired_mae_gain,
                            "bootstrap_lower": row.bootstrap_lower,
                            "bootstrap_upper": row.bootstrap_upper,
                            "positive_coins": row.positive_coins,
                            "largest_absolute_coin_share": row.largest_absolute_coin_share,
                            "strict_period_pass": row.strict_period_pass,
                            "provisional_period_pass": row.provisional_period_pass,
                        }
                    )
    return DataFrame.from_records(rows)


FROZEN_BRANCH_LOOKUP = {
    str(item["id"]): item
    for item in json.loads(FROZEN_G7.read_text(encoding="utf-8"))["branches"]
}


COMMON_CONTROLS = [
    "identical eligible pair-date rows for every profile in the sibling",
    "strict chronological development and two validation periods",
    "three predeclared deterministic model seeds for any retained attribution",
    "complete representation versus level-only and each immediately simpler attribution",
    "current ingredient versus causal stale copies and within-period nonself shuffles",
    "stable adjacent predictor ranges frozen without reading reaction outcomes",
    "equal-coin and predeclared group summaries with weekly-block uncertainty",
    "no one coin, week, feature, or remembered episode may carry the conclusion",
    "profit, future signed direction, entries, exits, and strategy promotion remain closed",
]


def g8_branch(
    branch_id: str,
    plain_name: str,
    hypothesis: str,
    parent_evidence: Sequence[str],
    cohorts: Sequence[str],
    targets: Sequence[str],
    attribution_dimensions: Sequence[str],
    method: str,
) -> dict[str, Any]:
    return {
        "id": branch_id,
        "plain_name": plain_name,
        "hypothesis": hypothesis,
        "parent_evidence": list(parent_evidence),
        "cohorts": list(cohorts),
        "targets": list(targets),
        "attribution_dimensions": list(attribution_dimensions),
        "method": method,
        "required_controls": COMMON_CONTROLS,
        "decision_rule": (
            "A detailed attribution survives only when its complete representation "
            "improves both validation periods against every immediately simpler, stale, "
            "and shuffled control; uncertainty must exclude no improvement in a "
            "predeclared coin group, and the ordering must remain stable across seeds."
        ),
        "status": "frozen_for_outcome_blind_generation8_preflight",
    }


def frozen_generation8(review_path: Path) -> dict[str, Any]:
    branches = [
        g8_branch(
            "g8a_level_geometry_attribution_for_participation_and_volatility",
            "Which causal level properties make participation plus volatility useful?",
            "The strict normal-coin interaction may depend on prominence, width, proximity, or timeframe composition rather than a generic level flag.",
            (
                "Normal next-hour volume passed all fourteen control-period checks for level plus participation plus volatility.",
                "The complete model reduced error by roughly 4.3-6.0% versus level-only and roughly 10% versus both context components without a current level.",
            ),
            ("normal", "meme"),
            ("&-g6_future_volume_ratio_h1", "&-g6_future_range_ratio_h1"),
            (
                "level family identity and dependency group",
                "prominence, strength, and support fraction",
                "zone width and thinness",
                "pre-contact distance and contact proximity",
                "native and higher-timeframe counts",
                "level age, persistence, and prior touch count",
            ),
            "Construct causal subblocks from the existing event provenance. Use leave-one-subblock-out and add-one-subblock ladders around the retained participation/volatility model; never tune a property cutoff to the reaction outcomes.",
        ),
        g8_branch(
            "g8b_local_participation_attribution_under_volatility",
            "Is the useful local-participation ingredient volume, pressure, persistence, or their combination?",
            "Volume engagement and pressure persistence may describe different forms of participation, so the broad block can hide which part is repeatable.",
            (
                "The complete normal volume interaction was strict against every component and stale control.",
                "The same broad interaction was provisional for range in both normal and meme cohorts.",
            ),
            ("normal", "meme"),
            ("&-g6_future_volume_ratio_h1", "&-g6_future_range_ratio_h1"),
            (
                "relative volume",
                "volume acceleration magnitude",
                "absolute pressure",
                "pressure persistence",
                "one-hour versus multi-hour duration",
                "stable adjacent intensity ranges",
            ),
            "Freeze predictor-only adjacent ranges from the development distribution, then compare volume-only, pressure-only, persistence-only, paired, stale, and shuffled representations on common rows.",
        ),
        g8_branch(
            "g8c_volatility_compression_attribution_under_participation",
            "Is the useful movement-capacity ingredient absolute volatility, compression, or their interaction?",
            "ATR/range describe available movement while Bollinger width/contraction describe compression; the strict broad block does not identify which mechanism matters.",
            (
                "Adding current volatility/compression to level plus participation reduced normal next-hour-volume error by about 0.76-0.82% in both validation periods.",
                "Replacing that block with its stale copy lost about 0.75-0.96%, while the meme range result remained positive across all fourteen controls but uncertain.",
            ),
            ("normal", "meme"),
            ("&-g6_future_volume_ratio_h1", "&-g6_future_range_ratio_h1"),
            (
                "ATR fraction",
                "prior range in ATR units",
                "Bollinger width",
                "range contraction ratio",
                "compression duration",
                "stable adjacent volatility and compression ranges",
            ),
            "Compare volatility-only, compression-only, and combined subblocks, including monotonic calibration across adjacent predictor bins and causal stale/shuffled controls.",
        ),
        g8_branch(
            "g8d_trend_momentum_attribution_with_participation",
            "Which direction-neutral trend-strength features add to local participation?",
            "Persistent trend strength and oscillator displacement may condition activity differently even when future up/down direction remains closed.",
            (
                "The broad trend/participation interaction was provisional for normal range and meme volume.",
                "All fourteen point-estimate controls were positive for meme next-hour volume, but only nine uncertainty checks were strict.",
            ),
            ("normal", "meme"),
            ("&-g6_future_volume_ratio_h1", "&-g6_future_range_ratio_h1"),
            (
                "EMA slope and moving-average separation magnitudes",
                "return slope and acceleration magnitudes",
                "ADX trend strength",
                "RSI displacement and change magnitudes",
                "MACD histogram and change magnitudes",
                "persistence duration and adjacent strength ranges",
            ),
            "Keep every feature direction-neutral. Compare persistent-trend, oscillator-displacement, acceleration, combined, stale, and shuffled subblocks without selecting indicators from validation outcomes.",
        ),
        g8_branch(
            "g8e_btc_context_freshness_horizon_and_intensity",
            "Which BTC activity horizon, freshness, and intensity conditions a level reaction?",
            "Immediate BTC movement, multi-hour activity, and BTC relative volume may carry different information; averaging them can dilute a small effect.",
            (
                "Local participation plus BTC activity was provisional for normal next-hour volume and range.",
                "The normal volume result passed thirteen of fourteen strict checks in the best full-cohort group; the matching meme result passed ten strict and thirteen point-positive checks.",
            ),
            ("normal", "meme"),
            ("&-g6_future_volume_ratio_h1", "&-g6_future_range_ratio_h1"),
            (
                "BTC absolute activity over 1h, 4h, and 24h",
                "BTC relative volume",
                "source age and persistence",
                "quiet, ordinary, and elevated adjacent intensity ranges",
                "BTC separated from established-alt and smart-contract groups",
            ),
            "Test all predeclared horizons as siblings rather than choosing the best horizon after outcomes. Each must beat local participation without BTC, stale BTC, and within-period shuffled BTC.",
        ),
        g8_branch(
            "g8f_cross_timeframe_incremental_value_and_room_geometry",
            "Which higher timeframe adds genuinely independent location information, and when is there room to react?",
            "Four-hour, eight-hour, and daily agreement may not be interchangeable; nearby opposing levels can also cap an otherwise active reaction.",
            (
                "Different-mechanism cross-timeframe agreement plus participation was provisional for normal next-hour volume across several predeclared groups.",
                "The full normal cohort passed thirteen of fourteen strict checks; the broad meme block did not pass the complete ladder.",
            ),
            ("normal", "meme"),
            (
                "&-g6_future_volume_ratio_h1",
                "&-g6_future_range_ratio_h1",
                "&-g6_absolute_excursion_atr_h4",
            ),
            (
                "4h, 8h, and 1d agreement tested separately and together",
                "same-mechanism versus different-mechanism agreement",
                "native-level contribution versus higher-timeframe contribution",
                "distance and width of the nearest opposing obstacle",
                "open room beyond the contacted zone",
                "agreement age, density, and contact ordering",
            ),
            "Use width/contact-frequency-matched controls and leave-one-timeframe-out ablations. Do not infer higher-timeframe precedence merely because its label is larger.",
        ),
        g8_branch(
            "g8g_orderbook_btc_attribution_for_smart_contract_platforms",
            "Which historical BTC order-book property adds to BTC activity for smart-contract-platform coins?",
            "Displayed-liquidity pressure may matter only under active BTC conditions, but the broad block is too coarse to identify whether pressure, coverage, or persistence supplies the lead.",
            (
                "The order-book plus BTC interaction was provisional for normal next-hour volume in the predeclared smart-contract-platform group.",
                "It passed twelve of fourteen strict checks and all fourteen point-positive checks in that group.",
            ),
            ("normal",),
            ("&-g6_future_volume_ratio_h1", "&-g6_future_range_ratio_h1"),
            (
                "BTC order-book pressure within 25 basis points",
                "absolute pressure and pressure tertile state",
                "source coverage and freshness",
                "pressure persistence and duration",
                "BTC activity horizon and intensity",
                "BTC pair-local evidence separated from BTC-wide altcoin context",
            ),
            "Retain venue/source labels and missingness. Compare pressure, coverage, regime, persistence, and combined blocks against BTC activity alone, stale order book, stale BTC, and shuffled controls; do not imply pair-local altcoin books.",
        ),
        g8_branch(
            "g8h_first_contact_occupancy_and_retest_state",
            "Do the retained interactions differ on first contact, continued occupancy, and later retests?",
            "A level can attract an initial reaction, absorb activity while occupied, or weaken after repeated contact; mixing these states can blur otherwise coherent small effects.",
            (
                "Generation 7 found one strict and several provisional interactions but aggregated independent contacts without detailed touch-state attribution.",
                "The objective explicitly keeps first contact, occupancy, and repeated contact as separate causal questions.",
            ),
            ("normal", "meme"),
            ("&-g6_future_volume_ratio_h1", "&-g6_future_range_ratio_h1"),
            (
                "first recorded causal contact",
                "continued zone occupancy",
                "first retest and later retest count",
                "time since prior contact",
                "level age and persistence",
                "prior reaction history available before the current contact",
            ),
            "Apply the frozen touch-state definitions to the retained participation/volatility, participation/trend, and participation/BTC surfaces. Compare states on common causal rows without choosing a favourable retest number from outcomes.",
        ),
    ]
    return {
        "schema_version": 1,
        "generation": 8,
        "branch_layer": 8,
        "status": "frozen_before_generation8_attribution_outcomes",
        "frozen_at_utc": g0.utc_now(),
        "authorization": (
            "Objective 02b Sections 15.5.3 and 15.5.6 authorize one complete, "
            "evidence-contingent Generation 8 attribution/refinement layer after the "
            "terminal Generation 7 joint review."
        ),
        "parent_joint_review": str(review_path),
        "generation_rule": (
            "Run outcome-blind coverage and representation checks for all eight siblings, "
            "then complete, park, or reject every sibling before any Generation 9 work."
        ),
        "portfolio_summary": {
            "siblings": len(branches),
            "parent_mechanisms": [
                "level geometry",
                "local volume and pressure",
                "volatility and compression",
                "trend strength and momentum",
                "BTC activity",
                "cross-timeframe agreement",
                "historical BTC order-book state",
                "contact and retest state",
            ],
            "cluster_descendant_included": False,
            "cluster_reason": (
                "The explicit independent-family cluster failed its complete seven-control "
                "ladder in both cohorts; Generation 8 will not tune cluster composition to "
                "those outcomes. Single-level and prior direct cluster records remain valid."
            ),
            "news_descendant_included": False,
            "news_reason": (
                "The bounded GDELT weak-component exception was not reproduced, so it is "
                "parked instead of receiving detailed threshold searches."
            ),
        },
        "common_stop_rule": (
            "Park an attribution when the complete representation fails an immediately "
            "simpler, stale, or shuffled control; signs conflict across validation periods "
            "or seeds; uncertainty includes no improvement; or one coin/episode carries it. "
            "Do not repair a failure by outcome-selected bins, thresholds, timeframes, or groups."
        ),
        "research_boundary": {
            "profit_optimization": False,
            "direction_prediction": False,
            "entry_or_exit_construction": False,
            "canonical_indicator_edit": False,
            "one_minute_direction_lane": "queued for Generation 9 only if Generation 8 evidence supports it",
        },
        "branches": branches,
    }


def build_review(review_id: str) -> dict[str, Any]:
    normal = load_terminal_run(NORMAL_RUN_ID, "normal")
    meme = load_terminal_run(MEME_RUN_ID, "meme")
    joint = joint_rows(normal, meme)
    bottlenecks = bottleneck_rows(normal, meme)
    review_dir = REVIEW_ROOT / review_id
    review_dir.mkdir(parents=True, exist_ok=True)
    joint_path = review_dir / "g7_joint_branch_target_review.csv"
    bottleneck_path = review_dir / "g7_retained_control_bottlenecks.csv"
    g0.atomic_write_csv(joint, joint_path)
    g0.atomic_write_csv(bottlenecks, bottleneck_path)
    result_path = review_dir / "g7_joint_review.json"
    frozen_g8 = frozen_generation8(result_path)
    g0.atomic_write_json(frozen_g8, FROZEN_G8_PATH)
    classifications = joint["joint_classification"].value_counts().to_dict()
    result = {
        "schema_version": 1,
        "review_id": review_id,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation7_joint_review_and_frozen_generation8",
        "normal_run": {
            "run_id": NORMAL_RUN_ID,
            "result_path": str(normal["paths"]["result"]),
            "result_sha256": g0.sha256_file(normal["paths"]["result"]),
        },
        "meme_run": {
            "run_id": MEME_RUN_ID,
            "result_path": str(meme["paths"]["result"]),
            "result_sha256": g0.sha256_file(meme["paths"]["result"]),
        },
        "branch_target_rows": len(joint),
        "joint_classifications": classifications,
        "strict_normal_branch_targets": int(joint["normal_status"].eq(STRICT).sum()),
        "strict_meme_branch_targets": int(joint["meme_status"].eq(STRICT).sum()),
        "provisional_both_cohort_branch_targets": int(
            joint["joint_classification"].eq("provisional_in_both_cohorts").sum()
        ),
        "primary_finding": (
            "Normal-coin next-hour volume is the only strict complete interaction: current "
            "level geometry plus local participation plus volatility/compression beat all "
            "seven simpler/stale controls in both validation periods. Its smallest incremental "
            "improvement was under 1%, consistent with a small technical edge rather than a "
            "deterministic rule. The matching broad range interaction was provisional in both "
            "normal and meme cohorts."
        ),
        "negative_findings": [
            "No meme branch-target passed all fourteen strict control-period checks.",
            "The explicit independent-family cluster did not add beyond both components and controls in either cohort; do not tune cluster pairings from these outcomes.",
            "Prior-range geometry, prior four-hour Volume Profile dampening, ETH context, and aggregate GDELT did not pass their complete interaction ladders.",
            "A failed interaction does not erase the earlier direct evidence that some standalone levels coincide with reactions; it means the tested added context did not improve prediction robustly.",
        ],
        "integrity": {
            "all_144_profiles_terminal": True,
            "all_126_comparisons_equal_row": True,
            "profit_used": False,
            "direction_prediction": False,
            "generation8_frozen_after_both_generation7_runs": True,
        },
        "artifacts": {
            "joint_branch_target_review": {
                "path": str(joint_path),
                "sha256": g0.sha256_file(joint_path),
            },
            "retained_control_bottlenecks": {
                "path": str(bottleneck_path),
                "sha256": g0.sha256_file(bottleneck_path),
            },
            "frozen_generation8_batch": {
                "path": str(FROZEN_G8_PATH),
                "sha256": g0.sha256_file(FROZEN_G8_PATH),
                "siblings": len(frozen_g8["branches"]),
            },
        },
        "next_action": (
            "Run outcome-blind coverage and causal representation preflight for every frozen "
            "Generation 8 sibling, then execute the complete attribution portfolio before "
            "opening any Generation 9 multi-source or one-minute descendant."
        ),
    }
    g0.atomic_write_json(result, result_path)
    return {**result, "result_path": str(result_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Jointly review terminal normal and meme Generation 7 interactions, then "
            "freeze the complete evidence-contingent Generation 8 attribution portfolio."
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

from __future__ import annotations

# The terminal review keeps long trader-readable interpretations by design.
# ruff: noqa: E402
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
    market_reaction_zone_freqai_generation8 as g8,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation8_initial_review as initial,
)


REVIEW_ROOT = g8.OUTPUT_ROOT / "generation8_review"
DEFAULT_REVIEW_ID = "g8_three_seed_joint_review_20260821a"
FREEZE_PATH = initial.DEFAULT_FREEZE_PATH
CONFIRMATION_RUNS = {
    "normal": "g8_freqai_confirm_normal_20260821a",
    "meme": "g8_freqai_confirm_meme_20260821a",
}
FINAL_STRICT = "confirmed_strict_across_three_seeds"
FINAL_PROVISIONAL = "confirmed_provisional_across_three_seeds"
FINAL_FAILED = "failed_seed_replication"
STATUS_RANK = {FINAL_FAILED: 0, FINAL_PROVISIONAL: 1, FINAL_STRICT: 2}
TARGET_NAMES = {
    "&-g6_future_volume_ratio_h1": "next-hour trading volume relative to its recent baseline",
    "&-g6_future_range_ratio_h1": "next-hour high-to-low range relative to its recent baseline",
    "&-g6_absolute_excursion_atr_h4": (
        "largest four-hour price movement regardless of direction, measured in recent "
        "ATR units"
    ),
}


def true_mask(values: pd.Series) -> pd.Series:
    return values.eq(True) | values.astype(str).str.casefold().eq("true")


def load_freeze(path: Path = FREEZE_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation8_confirmation_outcomes":
        raise ValueError("Generation 8 confirmation source is not frozen.")
    if frozen.get("initial_seed") != g8.INITIAL_SEED:
        raise ValueError("Generation 8 initial seed drifted.")
    if tuple(frozen.get("confirmation_seeds", ())) != g8.CONFIRMATION_SEEDS:
        raise ValueError("Generation 8 confirmation seeds drifted.")
    return frozen


def load_scored_run(
    run_id: str,
    *,
    cohort: str,
    expected_status: str,
) -> dict[str, Any]:
    root = g8.RECORD_ROOT / run_id
    paths = {
        "root": root,
        "result": root / "g8_freqai_result.json",
        "manifest": root / "g8_freqai_run_manifest.json",
        "decisions": root / "g8_freqai_route_decisions.csv",
        "group_scores": root / "g8_freqai_group_scores.csv",
        "pair_scores": root / "g8_freqai_pair_scores.csv",
        "eligibility": root / "g8_freqai_comparison_eligibility.csv",
    }
    missing = [str(path) for key, path in paths.items() if key != "root" and not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete Generation 8 scored run: {missing}")
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    integrity = result.get("integrity", {})
    problems: list[str] = []
    if result.get("status") != expected_status:
        problems.append(f"status={result.get('status')!r}")
    if result.get("cohort") != cohort:
        problems.append(f"cohort={result.get('cohort')!r}")
    if integrity.get("unequal_prediction_key_comparisons") != 0:
        problems.append("unequal prediction keys")
    if integrity.get("duplicate_prediction_rows_removed") != 0:
        problems.append("duplicate prediction rows")
    if integrity.get("profit_used") is not False:
        problems.append("profit used")
    if integrity.get("future_signed_direction") is not False:
        problems.append("future direction used")
    if integrity.get("all_profile_commands_terminal") is not True:
        problems.append("non-terminal profiles")
    if problems:
        raise ValueError(f"Generation 8 {cohort} run integrity failed: {problems}")
    output: dict[str, Any] = {
        "run_id": run_id,
        "cohort": cohort,
        "paths": paths,
        "result": result,
        "manifest": manifest,
    }
    for key in ("decisions", "group_scores", "pair_scores", "eligibility"):
        output[key] = pd.read_csv(paths[key])
    if not true_mask(output["eligibility"]["identical_prediction_keys"]).all():
        raise ValueError(f"Generation 8 {cohort} scored run has unequal rows.")
    return output


def load_all_runs(frozen: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    output: dict[str, dict[int, dict[str, Any]]] = {}
    for cohort in ("normal", "meme"):
        initial_source = frozen["source_runs"][cohort]
        initial_run = load_scored_run(
            str(initial_source["run_id"]),
            cohort=cohort,
            expected_status="completed_generation8_initial_seed_attribution",
        )
        if g0.sha256_file(initial_run["paths"]["result"]) != initial_source["result_sha256"]:
            raise ValueError(f"Generation 8 {cohort} initial result changed after freeze.")
        confirmation = load_scored_run(
            CONFIRMATION_RUNS[cohort],
            cohort=cohort,
            expected_status="completed_generation8_seed_confirmation",
        )
        selection = frozen["cohorts"][cohort]
        if confirmation["result"]["profiles_completed"] != selection["profile_count"]:
            raise ValueError(f"Generation 8 {cohort} confirmation profile count drifted.")
        if confirmation["result"]["comparisons_completed"] != selection["comparison_count"]:
            raise ValueError(f"Generation 8 {cohort} confirmation comparison count drifted.")
        output[cohort] = {
            g8.INITIAL_SEED: initial_run,
            **{seed: confirmation for seed in g8.CONFIRMATION_SEEDS},
        }
    return output


def exact_decision(
    run: dict[str, Any], candidate: dict[str, Any], seed: int
) -> pd.Series:
    decisions = run["decisions"]
    selected = decisions.loc[
        decisions["question_id"].eq(candidate["question_id"])
        & decisions["route_id"].eq(candidate["route_id"])
        & decisions["target"].eq(candidate["target"])
        & decisions["group_id"].eq(candidate["group_id"])
        & decisions["seed"].eq(seed)
    ]
    if len(selected) != 1:
        raise ValueError(
            "Expected one Generation 8 decision for "
            f"{candidate['question_id']}/{candidate['route_id']}/"
            f"{candidate['target']}/{candidate['group_id']}/seed{seed}; got {len(selected)}."
        )
    row = selected.iloc[0]
    if int(row["expected_controls"]) != int(row["controls_present"]):
        raise ValueError("Generation 8 confirmation decision lacks a frozen control.")
    if not bool(true_mask(pd.Series([row["both_validation_periods_present"]])).iloc[0]):
        raise ValueError("Generation 8 confirmation decision lacks a validation period.")
    return row


def score_rows(
    run: dict[str, Any], candidate: dict[str, Any], seed: int
) -> DataFrame:
    frame = run["group_scores"]
    return frame.loc[
        frame["question_id"].eq(candidate["question_id"])
        & frame["route_id"].eq(candidate["route_id"])
        & frame["target"].eq(candidate["target"])
        & frame["group_id"].eq(candidate["group_id"])
        & frame["seed"].eq(seed)
    ].copy()


def relative_gains(
    run: dict[str, Any], candidate: dict[str, Any], seed: int
) -> list[float]:
    members = str(candidate["group_members"]).split(",")
    frame = run["pair_scores"]
    selected = frame.loc[
        frame["question_id"].eq(candidate["question_id"])
        & frame["route_id"].eq(candidate["route_id"])
        & frame["target"].eq(candidate["target"])
        & frame["pair"].isin(members)
        & frame["seed"].eq(seed)
        & frame["candidate_mae"].notna()
        & frame["baseline_mae"].notna()
    ]
    grouped = selected.groupby(["comparison_id", "period"], observed=True).agg(
        candidate_mae=("candidate_mae", "mean"),
        baseline_mae=("baseline_mae", "mean"),
    )
    relative = (grouped["baseline_mae"] - grouped["candidate_mae"]) / grouped[
        "baseline_mae"
    ].replace(0.0, np.nan)
    return [float(value) for value in relative.dropna()]


def final_status(seed_details: dict[int, dict[str, bool]]) -> str:
    if all(detail["strict"] for detail in seed_details.values()):
        return FINAL_STRICT
    if all(detail["point_positive"] for detail in seed_details.values()):
        return FINAL_PROVISIONAL
    return FINAL_FAILED


def classify_candidate(
    candidate: dict[str, Any], runs: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    seed_details: dict[int, dict[str, bool]] = {}
    all_scores: list[DataFrame] = []
    all_relative: list[float] = []
    for seed in (g8.INITIAL_SEED, *g8.CONFIRMATION_SEEDS):
        run = runs[seed]
        decision = exact_decision(run, candidate, seed)
        strict = bool(
            true_mask(pd.Series([decision["all_controls_strict"]])).iloc[0]
        )
        point_positive = bool(
            true_mask(pd.Series([decision["all_controls_point_positive"]])).iloc[0]
        )
        seed_details[seed] = {"strict": strict, "point_positive": point_positive}
        all_scores.append(score_rows(run, candidate, seed))
        all_relative.extend(relative_gains(run, candidate, seed))
    scores = pd.concat(all_scores, ignore_index=True)
    expected_checks = int(candidate["expected_controls"]) * 2 * 3
    if len(scores) != expected_checks:
        raise ValueError(
            f"Generation 8 candidate has {len(scores)} checks, expected {expected_checks}."
        )
    status = final_status(seed_details)
    worst = scores.sort_values(
        ["equal_coin_paired_mae_gain", "bootstrap_lower"]
    ).iloc[0]
    strict_checks = int(true_mask(scores["strict_period_pass"]).sum())
    positive_checks = int(true_mask(scores["provisional_period_pass"]).sum())
    return {
        **candidate,
        "final_status": status,
        "seed42_strict": seed_details[42]["strict"],
        "seed17_strict": seed_details[17]["strict"],
        "seed73_strict": seed_details[73]["strict"],
        "seed42_point_positive": seed_details[42]["point_positive"],
        "seed17_point_positive": seed_details[17]["point_positive"],
        "seed73_point_positive": seed_details[73]["point_positive"],
        "strict_control_period_checks": strict_checks,
        "point_positive_control_period_checks": positive_checks,
        "total_control_period_checks": expected_checks,
        "minimum_equal_coin_error_gain": float(
            scores["equal_coin_paired_mae_gain"].min()
        ),
        "minimum_uncertainty_lower_bound": float(scores["bootstrap_lower"].min()),
        "maximum_uncertainty_upper_bound": float(scores["bootstrap_upper"].max()),
        "minimum_relative_error_reduction": (
            float(min(all_relative)) if all_relative else np.nan
        ),
        "median_relative_error_reduction": (
            float(np.median(all_relative)) if all_relative else np.nan
        ),
        "maximum_relative_error_reduction": (
            float(max(all_relative)) if all_relative else np.nan
        ),
        "worst_seed": int(worst["seed"]),
        "worst_period": str(worst["period"]),
        "worst_control": str(worst["control_type"]),
    }


def candidate_review(
    frozen: dict[str, Any], runs: dict[str, dict[int, dict[str, Any]]]
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort in ("normal", "meme"):
        for candidate in frozen["cohorts"][cohort]["candidate_rows"]:
            rows.append(classify_candidate(candidate, runs[cohort]))
    return DataFrame.from_records(rows)


def relationship_summary(review: DataFrame) -> DataFrame:
    keys = ["cohort", "branch_id", "surface", "route_id", "route_type", "target"]
    rows: list[dict[str, Any]] = []
    for key, frame in review.groupby(keys, observed=True, dropna=False):
        values = dict(zip(keys, key, strict=True))
        best_status = max(frame["final_status"], key=STATUS_RANK.get)
        rows.append(
            {
                **values,
                "target_plain_name": TARGET_NAMES.get(values["target"], values["target"]),
                "relationship_status": best_status,
                "strict_groups": ",".join(
                    sorted(
                        frame.loc[frame["final_status"].eq(FINAL_STRICT), "group_id"]
                    )
                ),
                "provisional_groups": ",".join(
                    sorted(
                        frame.loc[
                            frame["final_status"].eq(FINAL_PROVISIONAL), "group_id"
                        ]
                    )
                ),
                "failed_groups": ",".join(
                    sorted(
                        frame.loc[frame["final_status"].eq(FINAL_FAILED), "group_id"]
                    )
                ),
                "best_minimum_relative_error_reduction": float(
                    frame["minimum_relative_error_reduction"].max()
                ),
                "best_median_relative_error_reduction": float(
                    frame["median_relative_error_reduction"].max()
                ),
            }
        )
    return DataFrame.from_records(rows).sort_values(keys)


def branch_completion(relationships: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    by_id = {str(item["id"]): item for item in g8.FROZEN["branches"]}
    for branch_id, branch in by_id.items():
        selected = relationships.loc[relationships["branch_id"].eq(branch_id)]
        strict = int(selected["relationship_status"].eq(FINAL_STRICT).sum())
        provisional = int(
            selected["relationship_status"].eq(FINAL_PROVISIONAL).sum()
        )
        failed = int(selected["relationship_status"].eq(FINAL_FAILED).sum())
        if strict:
            status = "completed_with_strict_relationship_lead"
        elif provisional:
            status = "completed_with_provisional_relationship_lead"
        elif len(selected):
            status = "completed_without_seed_stable_relationship"
        else:
            status = "completed_without_seed42_candidate"
        rows.append(
            {
                "branch_id": branch_id,
                "plain_name": branch["plain_name"],
                "status": status,
                "strict_relationships": strict,
                "provisional_relationships": provisional,
                "failed_seed_replications": failed,
                "seed42_candidate_relationships": len(selected),
            }
        )
    return DataFrame.from_records(rows)


def cross_cohort_summary(relationships: DataFrame) -> DataFrame:
    keys = ["branch_id", "surface", "route_id", "route_type", "target"]
    rows: list[dict[str, Any]] = []
    for key, frame in relationships.groupby(keys, observed=True, dropna=False):
        values = dict(zip(keys, key, strict=True))
        status_by_cohort = dict(
            zip(frame["cohort"], frame["relationship_status"], strict=True)
        )
        normal = status_by_cohort.get("normal", "no_seed42_candidate")
        meme = status_by_cohort.get("meme", "no_seed42_candidate")
        both = normal in {FINAL_STRICT, FINAL_PROVISIONAL} and meme in {
            FINAL_STRICT,
            FINAL_PROVISIONAL,
        }
        rows.append(
            {
                **values,
                "target_plain_name": TARGET_NAMES.get(values["target"], values["target"]),
                "normal_status": normal,
                "meme_status": meme,
                "confirmed_in_both_cohorts": both,
            }
        )
    return DataFrame.from_records(rows).sort_values(keys)


def build_review(review_id: str) -> dict[str, Any]:
    frozen = load_freeze()
    runs = load_all_runs(frozen)
    candidates = candidate_review(frozen, runs)
    relationships = relationship_summary(candidates)
    branches = branch_completion(relationships)
    cross_cohort = cross_cohort_summary(relationships)
    review_dir = REVIEW_ROOT / review_id
    review_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "candidate_review": review_dir / "g8_three_seed_candidate_review.csv",
        "relationship_summary": review_dir / "g8_three_seed_relationship_summary.csv",
        "branch_completion": review_dir / "g8_branch_completion.csv",
        "cross_cohort": review_dir / "g8_cross_cohort_relationships.csv",
    }
    g0.atomic_write_csv(candidates, paths["candidate_review"])
    g0.atomic_write_csv(relationships, paths["relationship_summary"])
    g0.atomic_write_csv(branches, paths["branch_completion"])
    g0.atomic_write_csv(cross_cohort, paths["cross_cohort"])
    strict = relationships.loc[relationships["relationship_status"].eq(FINAL_STRICT)]
    provisional = relationships.loc[
        relationships["relationship_status"].eq(FINAL_PROVISIONAL)
    ]
    result_path = review_dir / "g8_three_seed_joint_review.json"
    result = {
        "schema_version": 1,
        "review_id": review_id,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation8_three_seed_joint_review",
        "all_eight_siblings_completed_before_descendants": len(branches) == 8,
        "candidate_group_rows": len(candidates),
        "candidate_group_row_statuses": candidates["final_status"].value_counts().to_dict(),
        "relationship_rows": len(relationships),
        "strict_relationships": len(strict),
        "provisional_relationships": len(provisional),
        "failed_relationships": int(
            relationships["relationship_status"].eq(FINAL_FAILED).sum()
        ),
        "cross_cohort_confirmed_relationships": int(
            true_mask(cross_cohort["confirmed_in_both_cohorts"]).sum()
        ),
        "branches_with_strict_leads": int(
            branches["status"].eq("completed_with_strict_relationship_lead").sum()
        ),
        "branches_with_provisional_only_leads": int(
            branches["status"].eq(
                "completed_with_provisional_relationship_lead"
            ).sum()
        ),
        "branches_without_seed_stable_leads": int(
            branches["status"].isin(
                {
                    "completed_without_seed_stable_relationship",
                    "completed_without_seed42_candidate",
                }
            ).sum()
        ),
        "strict_relationship_keys": [
            f"{row.cohort}|{row.branch_id}|{row.surface}|{row.route_id}|{row.target}|{row.strict_groups}"
            for row in strict.itertuples(index=False)
        ],
        "provisional_relationship_keys": [
            f"{row.cohort}|{row.branch_id}|{row.surface}|{row.route_id}|{row.target}|{row.provisional_groups}"
            for row in provisional.itertuples(index=False)
        ],
        "integrity": {
            "normal_initial_profiles": 212,
            "meme_initial_profiles": 192,
            "normal_confirmation_profiles": 138,
            "meme_confirmation_profiles": 62,
            "all_prediction_keys_equal": True,
            "duplicate_prediction_rows": 0,
            "profit_used": False,
            "future_signed_direction_used": False,
            "confirmation_registry_frozen_before_confirmation_outcomes": True,
        },
        "interpretation_boundary": (
            "A confirmed relationship means the named information repeatedly reduced "
            "prediction error for an unsigned market-reaction measure against all frozen "
            "controls. It is not proof of causation, direction, profit, or a trading rule."
        ),
        "artifacts": {
            key: {"path": str(path), "sha256": g0.sha256_file(path)}
            for key, path in paths.items()
        },
        "next_action": (
            "Explain the retained relationships in plain language, freeze a breadth-first "
            "Generation 9 batch from the complete portfolio, and only then run bounded "
            "one-minute direction replays or multi-source combinations."
        ),
    }
    g0.atomic_write_json(result, result_path)
    return {**result, "result_path": str(result_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Combine frozen Generation 8 seed-42 candidates with seed-17/73 confirmations "
            "and complete all eight siblings before descendants."
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

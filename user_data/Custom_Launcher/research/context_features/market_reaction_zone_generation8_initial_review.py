from __future__ import annotations

# Generation 8 review records intentionally retain long trader-readable text.
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


OUTPUT_ROOT = g8.OUTPUT_ROOT
FREQAI_ROOT = g8.RECORD_ROOT
REVIEW_ROOT = OUTPUT_ROOT / "generation8_review"
NORMAL_RUN_ID = "g8_freqai_full_normal_20260821a"
MEME_RUN_ID = "g8_freqai_full_meme_20260821a"
DEFAULT_REVIEW_ID = "g8_initial_joint_review_20260821a"
DEFAULT_FREEZE_PATH = REVIEW_ROOT / "g8_frozen_seed_confirmation_20260821a.json"
INITIAL_RETAINED = {
    "strict_seed42_candidate_for_seed_confirmation",
    "provisional_seed42_candidate_for_seed_confirmation",
}
EXPECTED = {
    "normal": {"profiles": 212, "comparisons": 202, "routes": 434},
    "meme": {"profiles": 192, "comparisons": 183, "routes": 101},
}


def true_mask(values: pd.Series) -> pd.Series:
    return values.eq(True) | values.astype(str).str.casefold().eq("true")


def run_paths(run_id: str) -> dict[str, Path]:
    root = FREQAI_ROOT / run_id
    return {
        "root": root,
        "result": root / "g8_freqai_result.json",
        "manifest": root / "g8_freqai_run_manifest.json",
        "decisions": root / "g8_freqai_route_decisions.csv",
        "group_scores": root / "g8_freqai_group_scores.csv",
        "pair_scores": root / "g8_freqai_pair_scores.csv",
        "eligibility": root / "g8_freqai_comparison_eligibility.csv",
        "prediction_audit": root / "g8_freqai_prediction_audit.csv",
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
        raise FileNotFoundError(f"Incomplete Generation 8 source: {missing}")
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    expected = EXPECTED[cohort]
    integrity = result.get("integrity", {})
    problems: list[str] = []
    if result.get("status") != "completed_generation8_initial_seed_attribution":
        problems.append(f"status={result.get('status')!r}")
    if result.get("cohort") != cohort:
        problems.append(f"cohort={result.get('cohort')!r}")
    if result.get("profiles_completed") != expected["profiles"]:
        problems.append(f"profiles={result.get('profiles_completed')!r}")
    if result.get("comparisons_completed") != expected["comparisons"]:
        problems.append(f"comparisons={result.get('comparisons_completed')!r}")
    if result.get("routes_scored") != expected["routes"]:
        problems.append(f"routes={result.get('routes_scored')!r}")
    if integrity.get("unequal_prediction_key_comparisons") != 0:
        problems.append("unequal prediction keys")
    if integrity.get("duplicate_prediction_rows_removed") != 0:
        problems.append("duplicate prediction rows")
    if integrity.get("profit_used") is not False:
        problems.append("profit used")
    if integrity.get("future_signed_direction") is not False:
        problems.append("future signed direction used")
    if integrity.get("all_profile_commands_terminal") is not True:
        problems.append("non-terminal profiles")
    if problems:
        raise ValueError(f"Generation 8 {cohort} integrity failed: {problems}")
    output: dict[str, Any] = {
        "run_id": run_id,
        "cohort": cohort,
        "result": result,
        "paths": paths,
    }
    for key in (
        "decisions",
        "group_scores",
        "pair_scores",
        "eligibility",
        "prediction_audit",
    ):
        output[key] = pd.read_csv(paths[key])
    if int(output["eligibility"]["comparison_id"].nunique()) != expected["comparisons"]:
        raise ValueError(f"Generation 8 {cohort} comparison count drifted.")
    if not true_mask(output["eligibility"]["identical_prediction_keys"]).all():
        raise ValueError(f"Generation 8 {cohort} has unequal prediction rows.")
    decisions = output["decisions"]
    if not decisions["seed"].eq(g8.INITIAL_SEED).all():
        raise ValueError(f"Generation 8 {cohort} initial review contains another seed.")
    if not true_mask(decisions["both_validation_periods_present"]).all():
        raise ValueError(f"Generation 8 {cohort} has incomplete validation periods.")
    if not decisions["expected_controls"].eq(decisions["controls_present"]).all():
        raise ValueError(f"Generation 8 {cohort} has an incomplete control ladder.")
    return output


def candidate_rows(run: dict[str, Any]) -> DataFrame:
    decisions = run["decisions"].copy()
    selected = decisions.loc[decisions["status"].isin(INITIAL_RETAINED)].copy()
    point_positive = true_mask(selected["all_controls_point_positive"])
    if not point_positive.all():
        raise ValueError("A frozen seed-42 candidate failed a point-positive control.")
    selected.insert(0, "cohort", run["cohort"])
    selected["initial_strength"] = selected["status"].map(
        {
            "strict_seed42_candidate_for_seed_confirmation": "strict",
            "provisional_seed42_candidate_for_seed_confirmation": "provisional",
        }
    )
    return selected


def confirmation_selection(
    candidates: DataFrame,
    *,
    cohort: str,
    profiles: dict[str, dict[str, Any]],
    comparisons: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    cohort_candidates = candidates.loc[candidates["cohort"].eq(cohort)].copy()
    route_keys = {
        (str(row.question_id), str(row.route_id))
        for row in cohort_candidates.itertuples(index=False)
    }
    chosen_comparisons = [
        item
        for item in comparisons
        if item["cohort"] == cohort
        and (str(item["question_id"]), str(item["route_id"])) in route_keys
    ]
    chosen_profiles = sorted(
        {
            str(item[field])
            for item in chosen_comparisons
            for field in ("candidate", "baseline")
        }
    )
    missing_routes = sorted(
        route_keys.difference(
            {
                (str(item["question_id"]), str(item["route_id"]))
                for item in chosen_comparisons
            }
        )
    )
    if missing_routes:
        raise ValueError(f"Confirmation registry lacks routes: {missing_routes}")
    if any(profile not in profiles for profile in chosen_profiles):
        raise ValueError("Confirmation selection references an unknown profile.")
    for question_id, route_id in sorted(route_keys):
        source = cohort_candidates.loc[
            cohort_candidates["question_id"].eq(question_id)
            & cohort_candidates["route_id"].eq(route_id)
        ]
        expected_controls = int(source["expected_controls"].iloc[0])
        for seed in g8.CONFIRMATION_SEEDS:
            present = {
                item["control_type"]
                for item in chosen_comparisons
                if item["question_id"] == question_id
                and item["route_id"] == route_id
                and int(item["seed"]) == seed
            }
            if len(present) != expected_controls:
                raise ValueError(
                    f"Confirmation controls drifted for {question_id}/{route_id}/seed{seed}."
                )
    candidate_fields = [
        "cohort",
        "question_id",
        "branch_id",
        "surface",
        "route_id",
        "route_type",
        "target",
        "group_id",
        "group_members",
        "plain_name",
        "initial_strength",
        "expected_controls",
    ]
    return {
        "cohort": cohort,
        "candidate_rows": cohort_candidates[candidate_fields].to_dict("records"),
        "candidate_row_count": len(cohort_candidates),
        "unique_question_routes": len(route_keys),
        "profile_ids": chosen_profiles,
        "profile_count": len(chosen_profiles),
        "comparison_ids": sorted(
            str(item["comparison_id"]) for item in chosen_comparisons
        ),
        "comparison_count": len(chosen_comparisons),
    }


def conceptual_overlap(candidates: DataFrame) -> DataFrame:
    keys = ["branch_id", "surface", "route_id", "route_type", "target"]
    grouped = (
        candidates.groupby(keys, as_index=False, observed=True)
        .agg(
            cohorts=("cohort", lambda values: ",".join(sorted(set(values)))),
            initial_strengths=(
                "initial_strength", lambda values: ",".join(sorted(set(values)))
            ),
            retained_group_rows=("group_id", "size"),
            group_ids=("group_id", lambda values: ",".join(sorted(set(values)))),
        )
        .sort_values(keys)
    )
    grouped["cross_cohort_seed42_overlap"] = grouped["cohorts"].eq("meme,normal")
    return grouped


def build_review(review_id: str, freeze_path: Path = DEFAULT_FREEZE_PATH) -> dict[str, Any]:
    normal = load_terminal_run(NORMAL_RUN_ID, "normal")
    meme = load_terminal_run(MEME_RUN_ID, "meme")
    candidates = pd.concat(
        [candidate_rows(normal), candidate_rows(meme)], ignore_index=True
    )
    registry_profiles, registry_comparisons = g8.build_registry(g8.CONFIRMATION_SEEDS)
    selections = {
        cohort: confirmation_selection(
            candidates,
            cohort=cohort,
            profiles=registry_profiles,
            comparisons=registry_comparisons,
        )
        for cohort in ("normal", "meme")
    }
    review_dir = REVIEW_ROOT / review_id
    review_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = review_dir / "g8_seed42_confirmation_candidates.csv"
    overlap_path = review_dir / "g8_seed42_conceptual_overlap.csv"
    g0.atomic_write_csv(candidates, candidates_path)
    overlap = conceptual_overlap(candidates)
    g0.atomic_write_csv(overlap, overlap_path)
    freeze = {
        "schema_version": 1,
        "generation": 8,
        "status": "frozen_before_generation8_confirmation_outcomes",
        "frozen_at_utc": g0.utc_now(),
        "source_runs": {
            cohort: {
                "run_id": run["run_id"],
                "result_path": str(run["paths"]["result"]),
                "result_sha256": g0.sha256_file(run["paths"]["result"]),
                "decisions_path": str(run["paths"]["decisions"]),
                "decisions_sha256": g0.sha256_file(run["paths"]["decisions"]),
            }
            for cohort, run in (("normal", normal), ("meme", meme))
        },
        "initial_seed": g8.INITIAL_SEED,
        "confirmation_seeds": list(g8.CONFIRMATION_SEEDS),
        "selection_rule": (
            "Freeze every seed-42 route, target, and predeclared coin group whose current "
            "features beat its immediately simpler, causal-stale, and deterministic "
            "nonself-shuffled controls on point estimates in both validation periods. "
            "Do not add failed routes after seeing confirmation outcomes."
        ),
        "final_decision_rule": (
            "Strict confirmation requires every frozen control-period comparison to remain "
            "point-positive with its weekly-block uncertainty above no improvement at seeds "
            "42, 17, and 73. Provisional confirmation requires every comparison to remain "
            "point-positive at all three seeds but permits uncertainty to include no "
            "improvement. Any non-positive comparison is a failed seed replication."
        ),
        "cohorts": selections,
        "research_boundary": {
            "profit_used": False,
            "future_signed_direction": False,
            "entry_exit_construction": False,
            "outcome_selected_groups_or_thresholds": False,
        },
    }
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(freeze, freeze_path)
    result_path = review_dir / "g8_initial_joint_review.json"
    result = {
        "schema_version": 1,
        "review_id": review_id,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation8_initial_joint_review_and_frozen_confirmation",
        "normal_profiles_reviewed": EXPECTED["normal"]["profiles"],
        "meme_profiles_reviewed": EXPECTED["meme"]["profiles"],
        "normal_candidate_group_rows": selections["normal"]["candidate_row_count"],
        "meme_candidate_group_rows": selections["meme"]["candidate_row_count"],
        "normal_unique_question_routes": selections["normal"][
            "unique_question_routes"
        ],
        "meme_unique_question_routes": selections["meme"]["unique_question_routes"],
        "cross_cohort_conceptual_overlaps": int(
            true_mask(overlap["cross_cohort_seed42_overlap"]).sum()
        ),
        "strict_seed42_group_rows": int(
            candidates["initial_strength"].eq("strict").sum()
        ),
        "provisional_seed42_group_rows": int(
            candidates["initial_strength"].eq("provisional").sum()
        ),
        "all_eight_siblings_initially_completed": True,
        "interpretation": (
            "These are candidates for deterministic-seed replication, not confirmed market "
            "relationships. Cross-cohort overlap is useful corroboration, while coherent "
            "smaller coin groups remain valid under the predeclared grouping rule."
        ),
        "artifacts": {
            "candidate_rows": {
                "path": str(candidates_path),
                "sha256": g0.sha256_file(candidates_path),
            },
            "conceptual_overlap": {
                "path": str(overlap_path),
                "sha256": g0.sha256_file(overlap_path),
            },
            "frozen_confirmation": {
                "path": str(freeze_path),
                "sha256": g0.sha256_file(freeze_path),
            },
        },
        "next_action": (
            "Run the exact frozen normal and meme confirmation registries for seeds 17 and "
            "73, then jointly classify every Generation 8 sibling before opening Generation 9."
        ),
    }
    g0.atomic_write_json(result, result_path)
    return {**result, "result_path": str(result_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Jointly validate Generation 8 seed-42 outputs and freeze exact seed-17/73 "
            "confirmation profiles before confirmation outcomes exist."
        )
    )
    parser.add_argument("--review-id", default=DEFAULT_REVIEW_ID)
    parser.add_argument("--freeze-path", type=Path, default=DEFAULT_FREEZE_PATH)
    args = parser.parse_args(argv)
    if not args.review_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in args.review_id
    ):
        raise ValueError(f"Invalid review id: {args.review_id!r}")
    result = build_review(args.review_id, args.freeze_path)
    print(json.dumps(result, indent=2, default=g0.json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

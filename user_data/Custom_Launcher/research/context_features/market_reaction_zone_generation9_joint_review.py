from __future__ import annotations

# Keep numerical libraries single-threaded. This review is deliberately lightweight.
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
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation9_freeze as g9z,
)


REVIEW_ROOT = g9z.OUTPUT_ROOT / "generation9_review"
DEFAULT_REVIEW_ID = "g9_joint_review_20260821a"
MODEL_RUN_IDS = {
    "normal": "g9_limited_multisource_normal_20260821a",
    "meme": "g9_limited_multisource_meme_20260821a",
}
ONE_MINUTE_RUN_ID = "g9_one_minute_diagnostic_20260821b"
MODEL_RECORD_ROOT = (
    g9z.OUTPUT_ROOT
    / "generation9_branches"
    / "g9_limited_multisource_freqai"
)
ONE_MINUTE_RECORD_ROOT = (
    g9z.OUTPUT_ROOT / "generation9_branches" / "g9_one_minute_replay"
)
MODEL_STRICT = "strict_three_seed_lead"
MODEL_PROVISIONAL = "provisional_three_seed_lead"
MODEL_FAILED = "failed_three_seed_replication"


def true_mask(values: pd.Series) -> pd.Series:
    return values.eq(True) | values.astype(str).str.casefold().eq("true")


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def verify_artifact(item: dict[str, Any], *, label: str) -> Path:
    path = Path(str(item["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if g0.sha256_file(path) != item["sha256"]:
        raise ValueError(f"{label} changed after it was recorded: {path}")
    return path


def load_freeze() -> dict[str, Any]:
    frozen = g9z.validate_existing_freeze(g9z.FREEZE_PATH)
    if frozen.get("generation") != 9:
        raise ValueError("The frozen sibling portfolio is not Generation 9.")
    if frozen["limited_three_source_family"]["question_count"] != len(
        g9z.MODEL_QUESTIONS
    ):
        raise ValueError("Generation 9 frozen question count drifted.")
    return frozen


def load_model_run(cohort: str) -> dict[str, Any]:
    run_id = MODEL_RUN_IDS[cohort]
    root = MODEL_RECORD_ROOT / run_id
    result_path = root / "g9_freqai_result.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    integrity = result.get("integrity", {})
    problems: list[str] = []
    if result.get("status") != "completed_generation9_limited_multisource":
        problems.append(f"status={result.get('status')!r}")
    if result.get("cohort") != cohort:
        problems.append(f"cohort={result.get('cohort')!r}")
    if integrity.get("unequal_prediction_key_comparisons") != 0:
        problems.append("unequal prediction keys")
    if integrity.get("duplicate_prediction_rows_removed") != 0:
        problems.append("duplicate prediction rows")
    if integrity.get("all_profile_commands_terminal") is not True:
        problems.append("non-terminal profiles")
    if integrity.get("profit_used") is not False:
        problems.append("profit used")
    if integrity.get("future_signed_direction") is not False:
        problems.append("future signed direction used")
    if integrity.get("profile_registry_frozen_before_model_outcomes") is not True:
        problems.append("registry was not frozen before outcomes")
    if problems:
        raise ValueError(f"Generation 9 {cohort} model run failed integrity: {problems}")
    decision_path = verify_artifact(
        result["artifacts"]["joint_seed_decisions"],
        label=f"Generation 9 {cohort} joint decisions",
    )
    decisions = pd.read_csv(decision_path)
    if len(decisions) != int(result["eligible_group_routes_scored"]):
        raise ValueError(f"Generation 9 {cohort} joint-decision count drifted.")
    return {
        "run_id": run_id,
        "result_path": result_path,
        "result": result,
        "decisions": decisions,
    }


def validate_model_portfolio(
    frozen: dict[str, Any], runs: dict[str, dict[str, Any]]
) -> None:
    family = frozen["limited_three_source_family"]
    if sum(run["result"]["profiles_completed"] for run in runs.values()) != int(
        family["profile_count"]
    ):
        raise ValueError("Generation 9 completed model profiles do not match the freeze.")
    if sum(run["result"]["comparisons_completed"] for run in runs.values()) != int(
        family["comparison_count"]
    ):
        raise ValueError("Generation 9 completed comparisons do not match the freeze.")
    if sum(run["result"]["questions_completed"] for run in runs.values()) != int(
        family["question_count"]
    ):
        raise ValueError("Generation 9 completed questions do not match the freeze.")


def combine_model_decisions(runs: dict[str, dict[str, Any]]) -> DataFrame:
    frames: list[DataFrame] = []
    for cohort, run in runs.items():
        frame = run["decisions"].copy()
        frame.insert(0, "cohort", cohort)
        frame.insert(1, "run_id", run["run_id"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["cohort", "question_id", "group_id"]
    )


def model_question_summary(
    decisions: DataFrame, frozen: dict[str, Any]
) -> DataFrame:
    declared = {
        item["question_id"]: item
        for item in frozen["limited_three_source_family"]["questions"]
    }
    rows: list[dict[str, Any]] = []
    for question_id, frame in decisions.groupby("question_id", observed=True):
        question = declared[str(question_id)]
        strict_groups = sorted(frame.loc[frame["status"].eq(MODEL_STRICT), "group_id"])
        provisional_groups = sorted(
            frame.loc[frame["status"].eq(MODEL_PROVISIONAL), "group_id"]
        )
        failed_groups = sorted(frame.loc[frame["status"].eq(MODEL_FAILED), "group_id"])
        if strict_groups:
            status = "retained_strict"
        elif provisional_groups:
            status = "retained_provisional_only"
        else:
            status = "not_retained"
        rows.append(
            {
                "question_id": question_id,
                "cohort": question["cohort"],
                "plain_name": question["plain_name"],
                "mechanism": question["mechanism"],
                "target": question["target"],
                "status": status,
                "strict_groups": ",".join(strict_groups),
                "provisional_groups": ",".join(provisional_groups),
                "failed_groups": ",".join(failed_groups),
                "minimum_equal_coin_paired_mae_gain": float(
                    frame["minimum_equal_coin_paired_mae_gain"].min()
                ),
                "best_minimum_equal_coin_paired_mae_gain": float(
                    frame["minimum_equal_coin_paired_mae_gain"].max()
                ),
                "best_minimum_bootstrap_lower": float(
                    frame["minimum_bootstrap_lower"].max()
                ),
            }
        )
    if set(rows_item["question_id"] for rows_item in rows) != set(declared):
        raise ValueError("Generation 9 joint decisions do not cover every frozen question.")
    return DataFrame.from_records(rows).sort_values(["cohort", "question_id"])


def load_one_minute_run() -> dict[str, Any]:
    root = ONE_MINUTE_RECORD_ROOT / ONE_MINUTE_RUN_ID
    record_path = root / "g9_one_minute_analysis_record.json"
    if not record_path.is_file():
        raise FileNotFoundError(record_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("status") != "completed_generation9_one_minute_diagnostic":
        raise ValueError("Generation 9 one-minute sibling is not terminal.")
    if record.get("run_id") != ONE_MINUTE_RUN_ID:
        raise ValueError("Generation 9 one-minute run id drifted.")
    paths = {
        key: verify_artifact(record["artifacts"][key], label=f"one-minute {key}")
        for key in (
            "actual_events",
            "analysis_coverage",
            "boundary_audit",
            "control_audit",
            "control_comparisons",
            "direction_scores",
            "summary",
        )
    }
    actual = pd.read_csv(paths["actual_events"])
    direction = pd.read_csv(paths["direction_scores"])
    control_audit = pd.read_csv(paths["control_audit"])
    boundary = pd.read_csv(paths["boundary_audit"])
    if len(actual) != int(record["summary"]["actual_episodes"]):
        raise ValueError("Generation 9 one-minute episode count drifted.")
    if int(actual["episode_id"].nunique()) != len(actual):
        raise ValueError("Generation 9 one-minute episodes are not independent rows.")
    return {
        "record_path": record_path,
        "record": record,
        "paths": paths,
        "actual": actual,
        "direction": direction,
        "control_audit": control_audit,
        "boundary": boundary,
    }


def one_minute_cohort_summary(actual: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort, frame in actual.groupby("cohort", observed=True):
        reacted = true_mask(frame["reaction_60m"])
        rows.append(
            {
                "cohort": cohort,
                "episodes": len(frame),
                "independent_pairs": int(frame["pair"].nunique()),
                "reacted_60m": int(reacted.sum()),
                "reaction_rate_60m": float(reacted.mean()),
                "breakouts": int(frame["first_zone_resolution"].eq("breakout").sum()),
                "rejections": int(frame["first_zone_resolution"].eq("rejection").sum()),
                "ties": int(frame["first_zone_resolution"].eq("tie").sum()),
            }
        )
    return DataFrame.from_records(rows).sort_values("cohort")


def all_scope_direction_summary(direction: DataFrame) -> DataFrame:
    selected = direction.loc[direction["scope"].eq("all")].copy()
    selected["meets_55pct_joint_floor"] = selected["joint_success_rate"].ge(0.55)
    selected["meets_65pct_joint_target"] = selected["joint_success_rate"].ge(0.65)
    selected["eligible_as_direction_lead"] = (
        selected["lead_eligible_sample_size"].pipe(true_mask)
        & selected["meets_55pct_joint_floor"]
    )
    return selected.sort_values(
        ["joint_success_rate", "conditional_direction_accuracy", "issued_call_coverage"],
        ascending=False,
    )


def build_follow_up_queue(
    model_questions: DataFrame,
    one_minute: dict[str, Any],
    g8_review: dict[str, Any],
) -> DataFrame:
    strict_keys = list(g8_review["strict_relationship_keys"])
    required_fragments = {
        "relative_volume": "|relative_volume|&-g6_future_volume_ratio_h1|",
        "participation_duration": "|participation_duration|&-g6_future_volume_ratio_h1|",
    }
    rows: list[dict[str, Any]] = []
    for mechanism, fragment in required_fragments.items():
        matches = [key for key in strict_keys if fragment in key]
        if not matches:
            raise ValueError(f"Missing strict Generation 8 source for {mechanism}.")
        rows.append(
            {
                "queue_id": f"g10_untouched_{mechanism}",
                "source_generation": 8,
                "source_status": "strict_across_three_seeds",
                "decision": "eligible_for_generation10_freeze",
                "plain_question": (
                    f"Does {mechanism.replace('_', ' ')} still improve unsigned next-hour "
                    "volume estimates in later untouched periods, relevant regimes, and "
                    "declared normal-coin groups?"
                ),
                "reason": (
                    "This single-source relationship was strict across all three seeds. "
                    "Generation 9 showed that combining it with other inputs did not add "
                    "stable value, so confirmation should keep it separate."
                ),
                "launch_now": False,
            }
        )
    for row in model_questions.loc[
        model_questions["status"].eq("retained_provisional_only")
    ].itertuples(index=False):
        rows.append(
            {
                "queue_id": f"park_{row.question_id}",
                "source_generation": 9,
                "source_status": "provisional_only",
                "decision": "park_not_eligible_for_generation10_confirmation",
                "plain_question": row.plain_name,
                "reason": (
                    "All three seeds had a positive point estimate in at least one group, "
                    "but uncertainty crossed zero and no group passed strictly across all "
                    "three seeds. Preserve the lead without treating it as confirmed."
                ),
                "launch_now": False,
            }
        )
    summary = one_minute["record"]["summary"]
    rows.append(
        {
            "queue_id": "park_g9_one_minute_direction_extension",
            "source_generation": 9,
            "source_status": "diagnostic_only",
            "decision": "park_until_reaction_gate_and_control_matching_improve",
            "plain_question": (
                "Can a larger independently selected one-minute sample distinguish whether "
                "price moves away from or through a level?"
            ),
            "reason": (
                f"Only {summary['actual_episodes']} episodes were tested, the conditioned "
                f"reaction rate was {summary['reaction_rate_60m']:.0%}, and only "
                f"{summary['control_balance_usable']['same_state_no_level']} matched "
                "no-level controls were state-balanced. No method met the joint 55% floor."
            ),
            "launch_now": False,
        }
    )
    return DataFrame.from_records(rows)


def load_generation8_review(frozen: dict[str, Any]) -> dict[str, Any]:
    item = frozen["source_contracts"]["generation8_terminal_joint_review"]
    path = verify_artifact(item, label="Generation 8 terminal joint review")
    review = json.loads(path.read_text(encoding="utf-8"))
    if review.get("status") != "completed_generation8_three_seed_joint_review":
        raise ValueError("Generation 8 source review is not terminal.")
    return review


def build_review(review_id: str) -> dict[str, Any]:
    frozen = load_freeze()
    model_runs = {cohort: load_model_run(cohort) for cohort in MODEL_RUN_IDS}
    validate_model_portfolio(frozen, model_runs)
    model_decisions = combine_model_decisions(model_runs)
    model_questions = model_question_summary(model_decisions, frozen)
    one_minute = load_one_minute_run()
    cohort_summary = one_minute_cohort_summary(one_minute["actual"])
    direction_summary = all_scope_direction_summary(one_minute["direction"])
    g8_review = load_generation8_review(frozen)
    follow_up = build_follow_up_queue(model_questions, one_minute, g8_review)

    review_dir = REVIEW_ROOT / review_id
    review_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "model_joint_decisions": review_dir / "g9_model_joint_decisions.csv",
        "model_question_summary": review_dir / "g9_model_question_summary.csv",
        "one_minute_cohort_summary": review_dir / "g9_one_minute_cohort_summary.csv",
        "one_minute_direction_summary": review_dir / "g9_one_minute_direction_summary.csv",
        "result_inspired_queue": review_dir / "g9_result_inspired_queue.csv",
    }
    g0.atomic_write_csv(model_decisions, paths["model_joint_decisions"])
    g0.atomic_write_csv(model_questions, paths["model_question_summary"])
    g0.atomic_write_csv(cohort_summary, paths["one_minute_cohort_summary"])
    g0.atomic_write_csv(direction_summary, paths["one_minute_direction_summary"])
    g0.atomic_write_csv(follow_up, paths["result_inspired_queue"])

    one_summary = one_minute["record"]["summary"]
    all_actual = one_minute["actual"]
    all_reacted = true_mask(all_actual["reaction_60m"])
    controls = one_minute["control_audit"]
    balanced_controls = {
        kind: int(
            true_mask(
                controls.loc[
                    controls["control_kind"].eq(kind), "balance_usable"
                ]
            ).sum()
        )
        for kind in sorted(controls["control_kind"].unique())
    }
    eligible = follow_up.loc[
        follow_up["decision"].eq("eligible_for_generation10_freeze")
    ]
    provisional = model_decisions.loc[
        model_decisions["status"].eq(MODEL_PROVISIONAL)
    ]
    result_path = review_dir / "g9_joint_review.json"
    result = {
        "schema_version": 1,
        "review_id": review_id,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation9_joint_review",
        "sequencing": {
            "both_sibling_families_frozen_before_outcomes": True,
            "both_sibling_families_terminal_before_review": True,
            "result_inspired_questions_queued_without_launch": True,
        },
        "limited_three_source_family": {
            "questions": len(model_questions),
            "profiles": sum(
                run["result"]["profiles_completed"] for run in model_runs.values()
            ),
            "comparisons": sum(
                run["result"]["comparisons_completed"] for run in model_runs.values()
            ),
            "eligible_group_routes": len(model_decisions),
            "strict_group_routes": int(
                model_decisions["status"].eq(MODEL_STRICT).sum()
            ),
            "provisional_group_routes": len(provisional),
            "failed_group_routes": int(
                model_decisions["status"].eq(MODEL_FAILED).sum()
            ),
            "strict_questions": int(
                model_questions["status"].eq("retained_strict").sum()
            ),
            "provisional_only_questions": int(
                model_questions["status"].eq("retained_provisional_only").sum()
            ),
            "plain_conclusion": (
                "No complete three-source model passed every frozen removal, stale-input, "
                "and shuffled-input control across all three seeds. Fresh-contact volume "
                "and trend-duration range questions stayed positive in limited normal-coin "
                "groups, but their uncertainty still included no improvement. Meme results "
                "did not reproduce either tested volume relationship."
            ),
        },
        "one_minute_family": {
            "episodes": len(all_actual),
            "independent_pairs": int(all_actual["pair"].nunique()),
            "reaction_count_60m": int(all_reacted.sum()),
            "reaction_rate_60m": float(all_reacted.mean()),
            "selection_was_conditioned_on_large_next_hour_volume": True,
            "breakouts": int(
                all_actual["first_zone_resolution"].eq("breakout").sum()
            ),
            "rejections": int(
                all_actual["first_zone_resolution"].eq("rejection").sum()
            ),
            "ties": int(all_actual["first_zone_resolution"].eq("tie").sum()),
            "balanced_controls": balanced_controls,
            "boundary_extensions_indicated": int(
                true_mask(one_minute["boundary"]["boundary_extension_indicated"]).sum()
            ),
            "direction_methods": len(direction_summary),
            "methods_meeting_joint_55pct_floor": int(
                direction_summary["meets_55pct_joint_floor"].sum()
            ),
            "methods_eligible_as_direction_leads": int(
                direction_summary["eligible_as_direction_lead"].sum()
            ),
            "best_observed_method_not_a_lead": one_summary[
                "best_observed_method_not_a_lead"
            ],
            "plain_conclusion": (
                "Six of twelve deliberately high-volume episodes also met the stricter "
                "minute-scale price-and-volume reaction rule. The paths split between "
                "breakouts and rejections, so a level did not itself provide direction. "
                "No direction method reached 55% joint reaction-and-direction success. "
                "The sample and well-balanced controls are too small for a lead."
            ),
        },
        "generation10_eligibility": {
            "eligible_mechanisms": len(eligible),
            "eligible_queue_ids": eligible["queue_id"].tolist(),
            "basis": (
                "Only the two strict Generation 8 single-source participation mechanisms "
                "remain eligible. Generation 9 provided evidence against merging them into "
                "a three-source model. Provisional Generation 9 and one-minute observations "
                "remain parked rather than promoted."
            ),
        },
        "completion_gate": {
            "three_independent_indications_from_two_route_families_at_55pct": False,
            "main_65pct_target": False,
            "objective_resolved": False,
        },
        "integrity": {
            "all_model_profiles_terminal": True,
            "all_prediction_keys_equal": True,
            "duplicate_prediction_rows": 0,
            "profit_used": False,
            "model_future_signed_direction_used": False,
            "one_minute_direction_only_inside_frozen_queue": True,
            "freqai_trained_on_twelve_episode_sample": False,
        },
        "interpretation_boundary": (
            "These are unsigned market-reaction and bounded path findings. They are not "
            "proof of causation, profit, entries, exits, or a trading strategy."
        ),
        "artifacts": {key: artifact(path) for key, path in paths.items()},
        "next_action": (
            "Freeze a narrow Generation 10 untouched-confirmation portfolio for the two "
            "strict single-source participation mechanisms. Keep them separate, include "
            "later unseen periods and realistic stale/outage controls, and do not extend "
            "the one-minute direction lane from this diagnostic result."
        ),
    }
    g0.atomic_write_json(result, result_path)
    return {**result, "result_path": str(result_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Complete the frozen Generation 9 joint review before any result-inspired "
            "Generation 10 work is allowed."
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

from __future__ import annotations

# Keep the terminal review lightweight and deterministic.
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
    market_reaction_zone_freqai_generation10 as g10f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation10_cache as g10c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation10_freeze as g10z,
)


REVIEW_ROOT = g10z.OUTPUT_ROOT / "generation10_review"
DEFAULT_REVIEW_ID = "g10_terminal_review_20260821a"
TERMINAL_RUN_ID = "g10_participation_confirmation_20260821b_target_schema_repair"
EXCLUDED_TECHNICAL_RUN_ID = "g10_participation_confirmation_20260821a"
EXPECTED_PERIOD_TESTS = len(g10z.SEEDS) * len(g10z.DECLARED_GROUPS)
EXPECTED_PAIR_TESTS = len(g10z.SEEDS) * len(g10z.CONFIRMATION_PERIODS)


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


def summarize_control_periods(group_scores: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["question_id", "route_id", "control_type", "period"]
    for key, frame in group_scores.groupby(keys, observed=True):
        question_id, route_id, control_type, period = key
        if len(frame) != EXPECTED_PERIOD_TESTS:
            raise ValueError(
                f"{question_id}/{control_type}/{period} has {len(frame)} tests; "
                f"expected {EXPECTED_PERIOD_TESTS}."
            )
        gains = pd.to_numeric(frame["equal_coin_paired_mae_gain"], errors="raise")
        rows.append(
            {
                "question_id": question_id,
                "route_id": route_id,
                "control_type": control_type,
                "period": period,
                "tests": len(frame),
                "point_positive_tests": int(true_mask(frame["point_positive"]).sum()),
                "strict_tests": int(true_mask(frame["strict_period_pass"]).sum()),
                "provisional_tests": int(
                    true_mask(frame["provisional_period_pass"]).sum()
                ),
                "mean_equal_coin_paired_mae_gain": float(gains.mean()),
                "minimum_equal_coin_paired_mae_gain": float(gains.min()),
                "maximum_equal_coin_paired_mae_gain": float(gains.max()),
            }
        )
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def summarize_activity_regimes(regime_scores: DataFrame) -> DataFrame:
    if not true_mask(regime_scores["descriptive_only"]).all():
        raise ValueError("Generation 10 activity-regime rows must remain descriptive.")
    rows: list[dict[str, Any]] = []
    keys = ["question_id", "route_id", "activity_regime"]
    for key, frame in regime_scores.groupby(keys, observed=True):
        question_id, route_id, regime = key
        if len(frame) != EXPECTED_PERIOD_TESTS:
            raise ValueError(
                f"{question_id}/{regime} has {len(frame)} descriptive tests; "
                f"expected {EXPECTED_PERIOD_TESTS}."
            )
        gains = pd.to_numeric(frame["equal_coin_paired_mae_gain"], errors="raise")
        rows.append(
            {
                "question_id": question_id,
                "route_id": route_id,
                "activity_regime": regime,
                "tests": len(frame),
                "point_positive_tests": int(gains.gt(0.0).sum()),
                "point_negative_tests": int(gains.lt(0.0).sum()),
                "mean_equal_coin_paired_mae_gain": float(gains.mean()),
                "minimum_equal_coin_paired_mae_gain": float(gains.min()),
                "maximum_equal_coin_paired_mae_gain": float(gains.max()),
                "descriptive_only": True,
                "control_scope": "current_candidate_vs_candidate_absent_base_only",
            }
        )
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def summarize_pair_controls(pair_scores: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["question_id", "route_id", "pair", "control_type"]
    for key, frame in pair_scores.groupby(keys, observed=True):
        question_id, route_id, pair, control_type = key
        if len(frame) != EXPECTED_PAIR_TESTS:
            raise ValueError(
                f"{question_id}/{pair}/{control_type} has {len(frame)} tests; "
                f"expected {EXPECTED_PAIR_TESTS}."
            )
        gains = pd.to_numeric(frame["paired_mae_gain"], errors="raise")
        positive = int(gains.gt(0.0).sum())
        rows.append(
            {
                "question_id": question_id,
                "route_id": route_id,
                "pair": pair,
                "control_type": control_type,
                "tests": len(frame),
                "point_positive_tests": positive,
                "all_point_positive": positive == len(frame),
                "mean_paired_mae_gain": float(gains.mean()),
                "minimum_paired_mae_gain": float(gains.min()),
                "maximum_paired_mae_gain": float(gains.max()),
                "descriptive_only": True,
            }
        )
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def pair_observations(pair_controls: DataFrame) -> DataFrame:
    expected_controls = {control_type for _, control_type in g10z.CONTROL_ROLES}
    rows: list[dict[str, Any]] = []
    keys = ["question_id", "route_id", "pair"]
    for key, frame in pair_controls.groupby(keys, observed=True):
        question_id, route_id, pair = key
        controls = set(frame["control_type"])
        if controls != expected_controls:
            raise ValueError(f"{question_id}/{pair} control coverage drifted: {controls}")
        all_controls_positive = bool(frame["all_point_positive"].all())
        rows.append(
            {
                "question_id": question_id,
                "route_id": route_id,
                "pair": pair,
                "controls_all_six_point_positive": int(
                    frame["all_point_positive"].sum()
                ),
                "all_three_controls_all_six_point_positive": all_controls_positive,
                "interpretation": (
                    "outcome_selected_single_pair_observation_not_confirmation"
                    if all_controls_positive
                    else "no_complete_pair_level_point_stability"
                ),
            }
        )
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def build_follow_up_queue(
    regime_summary: DataFrame, pair_summary: DataFrame
) -> DataFrame:
    active = regime_summary.loc[regime_summary["activity_regime"].eq("active")]
    if len(active) != len(g10z.QUESTIONS) or not active["point_positive_tests"].eq(
        active["tests"]
    ).all():
        raise ValueError("The expected active-regime observation is not present.")
    bnb = pair_summary.loc[
        pair_summary["pair"].eq("BNB/USDT:USDT")
        & pair_summary["all_three_controls_all_six_point_positive"]
    ]
    if len(bnb) != len(g10z.QUESTIONS):
        raise ValueError("The expected cross-mechanism BNB observation is not present.")
    return DataFrame.from_records(
        [
            {
                "queue_id": "future_activity_regime_replication",
                "source_status": "descriptive_result_inspired_observation",
                "decision": "requires_new_authorization_and_new_untouched_period",
                "plain_question": (
                    "Are current relative volume and participation-state duration useful "
                    "only when market activity is clearly quiet or active, rather than "
                    "during ordinary activity?"
                ),
                "reason": (
                    "Both mechanisms beat the plain model in every active-regime slice, "
                    "while ordinary-activity slices were mostly negative. These regime "
                    "rows were selected after the broad result and did not include the "
                    "full stale/shuffle ladder, so they are a question, not confirmation."
                ),
                "launch_now": False,
            },
            {
                "queue_id": "future_bnb_integrity_and_peer_replication",
                "source_status": "outcome_selected_single_pair_observation",
                "decision": "requires_new_authorization_and_new_untouched_period",
                "plain_question": (
                    "Why did BNB alone show point-positive improvement for both mechanisms "
                    "against all controls in both periods and all three seeds, and does a "
                    "predeclared economically similar peer group reproduce it?"
                ),
                "reason": (
                    "A single coin can reflect data shape, model behaviour, or chance. Audit "
                    "BNB first, then test a peer group selected for a rational reason before "
                    "opening outcomes; do not promote or tune to BNB alone."
                ),
                "launch_now": False,
            },
            {
                "queue_id": "future_reaction_selected_direction_handoff",
                "source_status": "programme_gap_not_generation10_evidence",
                "decision": "requires_separate_user_approved_objective",
                "plain_question": (
                    "Can stronger independently selected reaction episodes support a "
                    "directional study using only information known before contact?"
                ),
                "reason": (
                    "The programme has repeatable unsigned reaction evidence but no method "
                    "has reached 55% joint reaction-and-direction success. A broader "
                    "direction search is outside the completed Generation 10 authority."
                ),
                "launch_now": False,
            },
            {
                "queue_id": "park_broad_relative_volume_timing_claim",
                "source_status": "failed_untouched_confirmation",
                "decision": "park_broad_claim_do_not_tune",
                "plain_question": (
                    "Does current relative volume work as a broad timing-specific predictor "
                    "of next-hour reaction volume?"
                ),
                "reason": (
                    "It failed every declared group and usually did not beat its shuffled "
                    "timing control reliably enough across periods and seeds."
                ),
                "launch_now": False,
            },
            {
                "queue_id": "park_broad_participation_duration_timing_claim",
                "source_status": "failed_untouched_confirmation",
                "decision": "park_broad_claim_do_not_tune",
                "plain_question": (
                    "Does the duration of the current participation state work as a broad "
                    "timing-specific predictor of next-hour reaction volume?"
                ),
                "reason": (
                    "It failed every declared group and generally lost to the shuffled "
                    "version, especially during ordinary activity."
                ),
                "launch_now": False,
            },
        ]
    )


def validate_terminal_run(
    manifest: dict[str, Any],
    result: dict[str, Any],
    *,
    cache_path: Path,
) -> None:
    problems: list[str] = []
    if manifest.get("status") != "completed_generation10_participation_confirmation":
        problems.append(f"manifest status={manifest.get('status')!r}")
    if result.get("status") != "completed_generation10_participation_confirmation":
        problems.append(f"result status={result.get('status')!r}")
    if manifest.get("run_id") != TERMINAL_RUN_ID or result.get("run_id") != TERMINAL_RUN_ID:
        problems.append("terminal run id drift")
    integrity = result.get("integrity", {})
    if integrity.get("unequal_prediction_key_comparisons") != 0:
        problems.append("unequal prediction keys")
    if integrity.get("duplicate_prediction_rows_removed") != 0:
        problems.append("duplicate prediction rows")
    if integrity.get("all_profile_commands_terminal") is not True:
        problems.append("non-terminal profiles")
    if integrity.get("profit_used") is not False:
        problems.append("profit used")
    if integrity.get("future_signed_direction_used") is not False:
        problems.append("future signed direction used")
    if manifest["source_contracts"]["generation10_cache"]["sha256"] != g0.sha256_file(
        cache_path
    ):
        problems.append("run did not use the final repaired cache")
    if problems:
        raise ValueError(f"Generation 10 terminal run failed integrity: {problems}")


def load_terminal_evidence() -> dict[str, Any]:
    frozen = g10z.validate_existing_freeze(g10z.FREEZE_PATH)
    cache_path = g10c.RECORD_ROOT / "g10_cache_manifest.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache.get("status") != "completed_generation10_confirmation_cache":
        raise ValueError("Generation 10 cache is not terminal.")
    cache_freeze = cache["source_contracts"]["generation10_freeze"]
    if g0.sha256_file(g10z.FREEZE_PATH) != cache_freeze["sha256"]:
        raise ValueError("Generation 10 freeze changed after cache construction.")

    record_dir = g10f.RECORD_ROOT / TERMINAL_RUN_ID
    manifest_path = record_dir / "g10_freqai_run_manifest.json"
    result_path = record_dir / "g10_freqai_result.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    validate_terminal_run(manifest, result, cache_path=cache_path)

    paths = {
        key: verify_artifact(result["artifacts"][key], label=f"Generation 10 {key}")
        for key in (
            "pair_scores",
            "group_scores",
            "seed_decisions",
            "joint_group_decisions",
            "mechanism_summary",
            "activity_regime_scores",
        )
    }
    excluded_path = g10f.RECORD_ROOT / EXCLUDED_TECHNICAL_RUN_ID / "g10_freqai_run_manifest.json"
    excluded = json.loads(excluded_path.read_text(encoding="utf-8"))
    if excluded.get("status") != "failed":
        raise ValueError("Excluded Generation 10 technical run is not recorded as failed.")
    return {
        "frozen": frozen,
        "cache": cache,
        "cache_path": cache_path,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "result": result,
        "result_path": result_path,
        "paths": paths,
        "excluded": excluded,
        "excluded_path": excluded_path,
    }


def build_review(review_id: str) -> dict[str, Any]:
    evidence = load_terminal_evidence()
    mechanisms = pd.read_csv(evidence["paths"]["mechanism_summary"])
    group_scores = pd.read_csv(evidence["paths"]["group_scores"])
    regime_scores = pd.read_csv(evidence["paths"]["activity_regime_scores"])
    raw_pair_scores = pd.read_csv(evidence["paths"]["pair_scores"])
    control_periods = summarize_control_periods(group_scores)
    regimes = summarize_activity_regimes(regime_scores)
    pair_controls = summarize_pair_controls(raw_pair_scores)
    pairs = pair_observations(pair_controls)
    queue = build_follow_up_queue(regimes, pairs)

    if not mechanisms["status"].eq("failed_untouched_confirmation").all():
        raise ValueError("Generation 10 terminal mechanism status changed.")

    review_dir = REVIEW_ROOT / review_id
    review_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "control_period_summary": review_dir / "g10_control_period_summary.csv",
        "activity_regime_summary": review_dir / "g10_activity_regime_summary.csv",
        "pair_control_summary": review_dir / "g10_pair_control_summary.csv",
        "pair_observations": review_dir / "g10_pair_observations.csv",
        "result_inspired_queue": review_dir / "g10_result_inspired_queue.csv",
    }
    g0.atomic_write_csv(control_periods, paths["control_period_summary"])
    g0.atomic_write_csv(regimes, paths["activity_regime_summary"])
    g0.atomic_write_csv(pair_controls, paths["pair_control_summary"])
    g0.atomic_write_csv(pairs, paths["pair_observations"])
    g0.atomic_write_csv(queue, paths["result_inspired_queue"])

    active = regimes.loc[regimes["activity_regime"].eq("active")]
    typical = regimes.loc[regimes["activity_regime"].eq("typical")]
    bnb = pairs.loc[
        pairs["pair"].eq("BNB/USDT:USDT")
        & pairs["all_three_controls_all_six_point_positive"]
    ]
    result_path = review_dir / "g10_terminal_review.json"
    result = {
        "schema_version": 1,
        "review_id": review_id,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation10_terminal_review",
        "authorization_boundary": {
            "generation10_completed": True,
            "generation11_or_broader_direction_launched": False,
            "new_authorization_required_for_further_branch_layer": True,
        },
        "sequencing": {
            "questions_frozen_before_target_outcomes": True,
            "both_mechanisms_completed_before_terminal_review": True,
            "result_inspired_questions_queued_without_launch": True,
        },
        "technical_repair": {
            "excluded_run_id": EXCLUDED_TECHNICAL_RUN_ID,
            "excluded_run_status": evidence["excluded"]["status"],
            "failed_profiles": int(
                sum(
                    item.get("status") == "failed"
                    for item in evidence["excluded"]["commands"]
                )
            ),
            "reason": (
                "The first cache omitted standard target columns required by the reused "
                "research strategy schema. Four profiles failed before training. The cache "
                "was rebuilt to preserve that interface, a fresh run id was used, and the "
                "failed run contributed no evidence."
            ),
            "terminal_repaired_run_id": TERMINAL_RUN_ID,
        },
        "confirmation": {
            "period_label": evidence["result"]["period_label"],
            "globally_pristine_market_history": False,
            "profiles": int(evidence["result"]["profiles_completed"]),
            "comparisons": int(evidence["result"]["comparisons_completed"]),
            "questions": int(evidence["result"]["questions_completed"]),
            "declared_group_routes": int(evidence["result"]["declared_group_routes"]),
            "strict_group_routes": int(evidence["result"]["strict_group_routes"]),
            "provisional_group_routes": int(
                evidence["result"]["provisional_group_routes"]
            ),
            "failed_group_routes": int(evidence["result"]["failed_group_routes"]),
            "confirmed_mechanisms": 0,
            "failed_mechanisms": len(mechanisms),
            "failed_mechanism_ids": mechanisms["route_id"].tolist(),
            "plain_conclusion": (
                "Neither current relative volume nor participation-state duration remained "
                "a broadly reliable, timing-specific predictor of unsigned next-hour volume "
                "near calculated areas. Both failed every declared normal-coin group after "
                "the full base, stale, shuffled, period, seed, and uncertainty checks."
            ),
        },
        "result_inspired_observations": {
            "active_regime_point_positive": int(active["point_positive_tests"].sum()),
            "active_regime_tests": int(active["tests"].sum()),
            "typical_regime_point_positive": int(typical["point_positive_tests"].sum()),
            "typical_regime_tests": int(typical["tests"].sum()),
            "activity_regime_limit": (
                "These are post-result descriptive candidate-versus-base slices pooled "
                "across periods. They did not run the complete stale and shuffled ladder "
                "and cannot rescue the failed broad mechanisms."
            ),
            "bnb_mechanisms_all_controls_point_positive": len(bnb),
            "bnb_limit": (
                "BNB was found after opening outcomes and is one coin. Pair rows have no "
                "standalone uncertainty gate, so this is an audit and replication question, "
                "not a coin-specific edge."
            ),
        },
        "completion_gate": {
            "three_independent_indications_from_two_route_families_at_55pct": False,
            "main_65pct_target": False,
            "direction_confirmed": False,
            "objective_resolved": False,
        },
        "programme_conclusion": (
            "The broad reaction atlas still supports calculated areas as places where "
            "unsigned price/volume behaviour can change, but the final confirmation shows "
            "that the two strongest local-participation additions are conditional or "
            "period-sensitive rather than universal timing signals. The programme has not "
            "found the required independent, controlled directional indications. Objective "
            "02b therefore ends its authorized Generation 10 horizon unresolved."
        ),
        "integrity": {
            "all_terminal_profiles_completed": True,
            "all_prediction_keys_equal": True,
            "duplicate_prediction_rows": 0,
            "profit_used": False,
            "future_signed_direction_used": False,
            "technical_failed_run_excluded": True,
        },
        "interpretation_boundary": (
            "This is unsigned reaction-volume research. It does not prove causation, "
            "direction, profitability, an entry, an exit, or a trading strategy. The "
            "confirmation dates were target-and-mechanism-unopened but had been used for "
            "an unrelated absolute-excursion study, so they are not globally pristine."
        ),
        "source_contracts": {
            "generation10_freeze": artifact(g10z.FREEZE_PATH),
            "generation10_cache": artifact(evidence["cache_path"]),
            "terminal_run_manifest": artifact(evidence["manifest_path"]),
            "terminal_run_result": artifact(evidence["result_path"]),
            "excluded_technical_run_manifest": artifact(evidence["excluded_path"]),
        },
        "artifacts": {key: artifact(path) for key, path in paths.items()},
        "next_action": (
            "Do not launch another layer automatically. Ask the user whether to authorize "
            "a new untouched conditional-regime/BNB-audit batch, a separate bounded "
            "reaction-selected direction objective, or to close the research programme "
            "with the current unsigned reaction atlas."
        ),
    }
    g0.atomic_write_json(result, result_path)
    return {**result, "result_path": str(result_path.resolve())}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Complete the terminal Generation 10 joint review."
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

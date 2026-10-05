from __future__ import annotations

# Keep imports deterministic and single-threaded during registry construction.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation9_joint_review as g9r,
)


OUTPUT_ROOT = g9r.g9z.OUTPUT_ROOT
REVIEW_ROOT = OUTPUT_ROOT / "generation10_review"
FREEZE_PATH = REVIEW_ROOT / "g10_frozen_untouched_confirmation_20260821a.json"
G9_REVIEW_PATH = (
    OUTPUT_ROOT
    / "generation9_review"
    / g9r.DEFAULT_REVIEW_ID
    / "g9_joint_review.json"
)
G5_FRESH_ROOT = (
    OUTPUT_ROOT
    / "generation5_branches"
    / "g5c_new_period_normal_excursion_confirmation"
    / "g5c_fresh_preflight_20260821a"
)
G5_SOURCE_MANIFEST = G5_FRESH_ROOT / "g5c_source_manifest.json"
G5_SOURCE_BUILD_RECORD = G5_FRESH_ROOT / "g5c_source_build_record.json"
G5_PREFLIGHT_RECORD = G5_FRESH_ROOT / "g5c_event_preflight_record.json"
G5_RESULT_PATH = (
    OUTPUT_ROOT
    / "generation5_branches"
    / "g5c_new_period_normal_excursion_confirmation"
    / "g5c_full_normal_density_cluster_fresh_20260821a"
    / "g5c_result.json"
)

SEEDS = (42, 17, 73)
TARGET = "&-g6_future_volume_ratio_h1"
CONFIRMATION_PERIODS = ("g5c_confirmation_early", "g5c_confirmation_late")
TIMERANGE = "20260720-20260820"
TRAIN_DAYS = 110
BACKTEST_DAYS = 31
BASE_BLOCKS = (
    "g8_level_base",
    "g8_volatility_absolute",
    "g8_volatility_compression",
)
CONTROL_ROLES = (
    ("baseline", "candidate_absent_or_unavailable"),
    ("stale", "causal_72h_stale"),
    ("shuffled", "within_period_nonself_shuffle"),
)
DECLARED_GROUPS = (
    "established_altcoins",
    "full_normal_cohort",
    "smart_contract_platforms",
)


@dataclass(frozen=True)
class ConfirmationQuestion:
    question_id: str
    mechanism_id: str
    plain_name: str
    interpretation: str
    candidate_block: str


QUESTIONS = (
    ConfirmationQuestion(
        question_id="g10a_relative_volume_untouched_confirmation",
        mechanism_id="relative_volume",
        plain_name=(
            "Does current relative volume still improve next-hour reaction-volume "
            "estimates near calculated areas in the later confirmation month?"
        ),
        interpretation=(
            "Current completed-candle volume is compared with its causal recent median."
        ),
        candidate_block="g8_participation_relative_volume",
    ),
    ConfirmationQuestion(
        question_id="g10b_participation_duration_untouched_confirmation",
        mechanism_id="participation_duration",
        plain_name=(
            "Does the duration of the current participation state still improve "
            "next-hour reaction-volume estimates in the later confirmation month?"
        ),
        interpretation=(
            "Development-frozen low, middle, and high participation bands are paired "
            "with the number of consecutive hours spent in the current band."
        ),
        candidate_block="g8_participation_duration",
    ),
)


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def profile_id(question: ConfirmationQuestion, role: str, seed: int) -> str:
    return f"{question.question_id}__{role}__seed{seed}"


def build_registry() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    profiles: dict[str, dict[str, Any]] = {}
    comparisons: list[dict[str, Any]] = []
    for question in QUESTIONS:
        for seed in SEEDS:
            roles = {
                "baseline": BASE_BLOCKS,
                "current": (*BASE_BLOCKS, question.candidate_block),
                "stale": (*BASE_BLOCKS, f"{question.candidate_block}_stale"),
                "shuffled": (*BASE_BLOCKS, f"{question.candidate_block}_shuffled"),
            }
            for role, blocks in roles.items():
                identifier = profile_id(question, role, seed)
                profiles[identifier] = {
                    "profile_id": identifier,
                    "question_id": question.question_id,
                    "branch_id": question.question_id,
                    "cohort": "normal",
                    "surface": "later_chronological_confirmation",
                    "role": role,
                    "seed": seed,
                    "blocks": list(blocks),
                    "required_ready_blocks": [
                        f"g10_{question.mechanism_id}_common_support"
                    ],
                    "targets": [TARGET],
                    "plain_name": question.plain_name,
                    "mechanism": question.interpretation,
                    "pair_scope": "all",
                }
            candidate = profile_id(question, "current", seed)
            for role, control_type in CONTROL_ROLES:
                comparisons.append(
                    {
                        "comparison_id": (
                            f"{question.question_id}__current_vs_{role}__seed{seed}"
                        ),
                        "question_id": question.question_id,
                        "branch_id": question.question_id,
                        "cohort": "normal",
                        "surface": "later_chronological_confirmation",
                        "route_id": question.mechanism_id,
                        "route_type": "untouched_single_source_confirmation",
                        "control_type": control_type,
                        "candidate": candidate,
                        "baseline": profile_id(question, role, seed),
                        "baseline_role": role,
                        "seed": seed,
                        "targets": [TARGET],
                        "plain_name": question.plain_name,
                        "mechanism": question.interpretation,
                        "expected_controls_for_route": len(CONTROL_ROLES),
                        "result_group": "all_declared_groups",
                    }
                )
    return profiles, comparisons


def validate_sources() -> dict[str, Any]:
    g9 = json.loads(G9_REVIEW_PATH.read_text(encoding="utf-8"))
    if g9.get("status") != "completed_generation9_joint_review":
        raise ValueError("Generation 9 joint review is not terminal.")
    eligible = set(g9["generation10_eligibility"]["eligible_queue_ids"])
    if eligible != {
        "g10_untouched_relative_volume",
        "g10_untouched_participation_duration",
    }:
        raise ValueError("Generation 10 eligibility changed after the joint review.")
    source_manifest = json.loads(G5_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    periods = {
        item["id"]: (item["start_utc"], item["end_utc_exclusive"])
        for item in source_manifest["data"]["chronological_periods"]
    }
    expected = {
        "g5c_confirmation_early": (
            "2026-07-20T00:00:00Z",
            "2026-08-05T00:00:00Z",
        ),
        "g5c_confirmation_late": (
            "2026-08-05T00:00:00Z",
            "2026-08-20T00:00:00Z",
        ),
    }
    if {key: periods.get(key) for key in expected} != expected:
        raise ValueError("Generation 10 confirmation dates changed.")
    preflight = json.loads(G5_PREFLIGHT_RECORD.read_text(encoding="utf-8"))
    if preflight.get("target_values_compared_or_interpreted") is not False:
        raise ValueError("The fresh-source coverage preflight opened target outcomes.")
    prior = json.loads(G5_RESULT_PATH.read_text(encoding="utf-8"))
    prior_target = prior.get("target") or prior.get("target_column")
    if prior_target not in {
        "&-g5c_abs_excursion_atr_h1",
        "next_hour_absolute_excursion_in_prior_atr",
    }:
        raise ValueError(
            "The earlier fresh-period study was not limited to its declared absolute "
            f"excursion target: {prior_target!r}"
        )
    return {
        "generation9_joint_review": artifact(G9_REVIEW_PATH),
        "fresh_source_manifest": artifact(G5_SOURCE_MANIFEST),
        "fresh_source_build_record": artifact(G5_SOURCE_BUILD_RECORD),
        "fresh_source_coverage_preflight": artifact(G5_PREFLIGHT_RECORD),
        "earlier_unrelated_absolute_excursion_result": artifact(G5_RESULT_PATH),
    }


def validate_existing_freeze(path: Path = FREEZE_PATH) -> dict[str, Any]:
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation10_volume_target_outcomes":
        raise ValueError("Generation 10 freeze is not terminal.")
    for item in frozen["source_contracts"].values():
        source = Path(item["path"])
        if not source.is_file() or g0.sha256_file(source) != item["sha256"]:
            raise ValueError(f"Frozen Generation 10 source changed: {source}")
    return frozen


def freeze_generation10() -> dict[str, Any]:
    if FREEZE_PATH.is_file():
        return validate_existing_freeze()
    sources = validate_sources()
    profiles, comparisons = build_registry()
    frozen = {
        "schema_version": 1,
        "objective": "objective_02b_market_reaction_zone_discovery",
        "generation": 10,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation10_volume_target_outcomes",
        "questions": [asdict(question) for question in QUESTIONS],
        "question_count": len(QUESTIONS),
        "cohort": "normal",
        "pairs": list(
            json.loads(G5_SOURCE_MANIFEST.read_text(encoding="utf-8"))["data"][
                "pairs"
            ]
        ),
        "declared_groups": list(DECLARED_GROUPS),
        "confirmation_periods": list(CONFIRMATION_PERIODS),
        "timerange": TIMERANGE,
        "train_days": TRAIN_DAYS,
        "backtest_days": BACKTEST_DAYS,
        "seeds": list(SEEDS),
        "target": TARGET,
        "base_blocks": list(BASE_BLOCKS),
        "controls": [
            {"role": role, "type": control_type}
            for role, control_type in CONTROL_ROLES
        ],
        "profiles": profiles,
        "profile_count": len(profiles),
        "comparisons": comparisons,
        "comparison_count": len(comparisons),
        "strict_pass_rule": (
            "For every declared group, the current mechanism must beat the base model, "
            "its causal 72-hour-old version, and its deterministic within-period nonself "
            "shuffle in both chronological confirmation halves and all three seeds. Each "
            "period must have enough independent rows and coins, no one coin may dominate, "
            "and the weekly-block uncertainty lower bound must remain above no improvement."
        ),
        "regime_review": {
            "activity_states": ["quiet", "typical", "active"],
            "definition": (
                "Use the original Generation 8 development-frozen relative-volume "
                "boundaries. Regime rows are descriptive stability checks and cannot rescue "
                "a failed complete confirmation route."
            ),
        },
        "source_contracts": sources,
        "period_honesty": {
            "later_than_all_mechanism_development_and_internal_validation": True,
            "relative_volume_target_compared_before_this_freeze": False,
            "participation_duration_target_compared_before_this_freeze": False,
            "globally_pristine_market_history": False,
            "reason_not_globally_pristine": (
                "The same dates were previously used by a separate density-cluster study "
                "whose only scored target was unsigned one-hour absolute excursion."
            ),
            "allowed_label": "target_and_mechanism_unopened_chronological_confirmation",
            "forbidden_label": "never_before_seen_market_history",
        },
        "research_boundary": {
            "unsigned_next_hour_volume_only": True,
            "profit_used": False,
            "future_signed_direction_used": False,
            "entry_exit_construction": False,
            "strategy_promotion": False,
            "one_minute_direction_extension": False,
        },
        "sequencing": (
            "Both eligible single-source mechanisms are frozen together. Both must finish "
            "or be honestly parked before the terminal Generation 10 joint review."
        ),
    }
    REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(frozen, FREEZE_PATH)
    return frozen


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the narrow Generation 10 confirmation portfolio."
    )
    _ = parser.parse_args(argv)
    result = freeze_generation10()
    print(
        json.dumps(
            {
                "status": result["status"],
                "questions": result["question_count"],
                "profiles": result["profile_count"],
                "comparisons": result["comparison_count"],
                "period_label": result["period_honesty"]["allowed_label"],
                "freeze_path": str(FREEZE_PATH.resolve()),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

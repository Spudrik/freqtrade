"""Freeze the next market-reaction confirmations and audit OHLCV readiness.

This module reads only completed parent summaries and OHLCV timestamps.  It never
opens the future reaction values reserved for the two fresh confirmation blocks.
"""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_level_source_atlas as g17l,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_anchored_vwap as g24v,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_freqai_cache as g24cache,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_joint_review as g24j,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation25_convergence_representation as g25conv,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation25_joint_review as g25j,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation25_market_state_activity as g25state,
)


ANALYSIS_PATH = Path(__file__).resolve()
BATCH_ID = "market_reaction_fresh_confirmation_20260903a"
OUTPUT_ROOT = g25j.REVIEW_ROOT.parent / "fresh_confirmation"
FREEZE_PATH = OUTPUT_ROOT / "fresh_confirmation_freeze_20260903a.json"
COVERAGE_ROOT = OUTPUT_ROOT / "coverage"
DEFAULT_COVERAGE_ID = "fresh_confirmation_coverage_20260903a"

FRESH_PERIODS = (
    {
        "id": "fresh_early",
        "start_utc": "2026-08-20T00:00:00Z",
        "end_utc_exclusive": "2026-09-05T00:00:00Z",
    },
    {
        "id": "fresh_late",
        "start_utc": "2026-09-05T00:00:00Z",
        "end_utc_exclusive": "2026-09-20T00:00:00Z",
    },
)
MAXIMUM_OUTCOME_HOURS = 8
STATE_TRAINING_DAYS = {"normal": 365, "meme": 160}
FEATURE_WARMUP_DAYS = 30
STATE_ROLES = (
    g25state.CANDIDATE_ROLE,
    g25state.SHUFFLED_ROLE,
)

G25_REVIEW_PATH = g25j.REVIEW_ROOT / g25j.DEFAULT_RUN_ID / "g25_joint_review.json"
G24_REVIEW_PATH = g24j.REVIEW_ROOT / g24j.DEFAULT_REVIEW_ID / "g24_joint_review.json"
G25_STATE_RESULT = g25j.result_paths()["market_state_activity"]
G25_CONVERGENCE_RESULT = g25j.result_paths()["convergence"]
G24_PROFILE_REGISTRY = g24cache.REGISTRY_PATH


def artifact(path: Path) -> dict[str, Any]:
    return g25j.artifact(path)


def _load_json(path: Path, expected_status: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != expected_status:
        raise ValueError(f"Unexpected status for {path}: {value.get('status')}")
    return value


def _verified_csv(result: dict[str, Any], artifact_name: str) -> DataFrame:
    item = result["artifacts"][artifact_name]
    path = Path(item["path"])
    if g0.sha256_file(path) != item["sha256"]:
        raise ValueError(f"Parent result artifact changed: {path}")
    return pd.read_csv(path)


def _truthy(values: Series) -> Series:
    return values.astype(str).str.casefold().eq("true")


def selected_activity_questions() -> list[dict[str, str]]:
    result = _load_json(
        G25_STATE_RESULT,
        "completed_generation25_market_state_activity",
    )
    decisions = _verified_csv(result, "decisions")
    required = {"cohort", "market_scope", "target", "all_model_seed_cells_pass"}
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ValueError(f"Generation 25 state decisions lack columns: {missing}")
    retained = decisions.loc[_truthy(decisions["all_model_seed_cells_pass"])]
    questions = (
        retained[["cohort", "market_scope", "target"]]
        .drop_duplicates()
        .sort_values(["cohort", "market_scope", "target"], kind="stable")
    )
    output = [
        {
            "cohort": str(row.cohort),
            "market_scope": str(row.market_scope),
            "target": str(row.target),
        }
        for row in questions.itertuples(index=False)
    ]
    if len(output) != 15:
        raise ValueError(f"Expected 15 retained state questions, got {len(output)}.")
    return output


def selected_freqai_profiles() -> list[dict[str, Any]]:
    registry = _load_json(
        G24_PROFILE_REGISTRY,
        "frozen_before_generation24_freqai_target_materialization",
    )
    output: list[dict[str, Any]] = []
    for profile_id, profile in registry["profiles"].items():
        if str(profile["role"]) not in STATE_ROLES:
            continue
        output.append(
            {
                "profile_id": str(profile_id),
                "cohort": str(profile["cohort"]),
                "role": str(profile["role"]),
                "model_class": str(profile["model_class"]),
                "seed": int(profile["seed"]),
                "feature_columns": list(profile["feature_columns"]),
                "feature_cache": str(profile["feature_cache"]),
                "target_cache": str(profile["target_cache"]),
            }
        )
    output.sort(key=lambda item: (item["cohort"], item["model_class"], item["seed"], item["role"]))
    if len(output) != 16:
        raise ValueError(f"Expected 16 state/shuffled FreqAI profiles, got {len(output)}.")
    return output


def selected_convergence_question(g25_review: dict[str, Any]) -> dict[str, Any]:
    question = dict(
        g25_review["route_results"]["convergence_control_representation"][
            "closest_question"
        ]
    )
    expected = {
        "scope_value": "all_60_frozen_definitions",
        "metric": "crossing_count",
        "horizon_hours": 2,
        "market_scope": "top_ten_memes",
    }
    if any(question.get(key) != value for key, value in expected.items()):
        raise ValueError("Generation 25 convergence confirmation question drifted.")
    return expected


def selected_vwap_question(g24_review: dict[str, Any]) -> dict[str, Any]:
    audit = g24_review["route_results"]["causal_anchored_vwap"][
        "fresh_candidate_audit"
    ]
    question = dict(audit["question"])
    if question != g24j.VWAP_FRESH_CANDIDATE:
        raise ValueError("Generation 24 daily VWAP confirmation question drifted.")
    return question


def cohort_pairs() -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for cohort, path in g17l.COHORT_MANIFESTS.items():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        pairs = [str(pair) for pair in manifest["data"]["pairs"]]
        if len(pairs) != 10 or len(set(pairs)) != 10:
            raise ValueError(f"Expected ten unique {cohort} pairs, got {len(pairs)}.")
        output[str(cohort)] = pairs
    if set(output) != {"normal", "meme"}:
        raise ValueError(f"Unexpected cohort registry: {sorted(output)}")
    return output


def period_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    return (
        pd.Timestamp(FRESH_PERIODS[0]["start_utc"]),
        pd.Timestamp(FRESH_PERIODS[-1]["end_utc_exclusive"]),
    )


def required_data_bounds(cohort: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end_exclusive = period_bounds()
    first_required = start - pd.Timedelta(
        days=STATE_TRAINING_DAYS[cohort] + FEATURE_WARMUP_DAYS
    )
    last_required = end_exclusive + pd.Timedelta(hours=MAXIMUM_OUTCOME_HOURS - 1)
    return first_required, last_required


def assign_fresh_period(dates: Series) -> Series:
    values = pd.to_datetime(dates, utc=True, errors="raise")
    output = Series("outside_fresh_confirmation", index=values.index, dtype="object")
    for period in FRESH_PERIODS:
        mask = values.ge(pd.Timestamp(period["start_utc"])) & values.lt(
            pd.Timestamp(period["end_utc_exclusive"])
        )
        output.loc[mask] = str(period["id"])
    return output


def build_freeze_document() -> dict[str, Any]:
    g25_review = _load_json(
        G25_REVIEW_PATH,
        "completed_generation25_joint_review_one_batch_only",
    )
    g24_review = _load_json(G24_REVIEW_PATH, "completed_generation24_joint_review")
    queued = {
        str(item["question"]): str(item["status"])
        for item in g25_review["next_questions_not_launched"]
    }
    if len(queued) != 3:
        raise ValueError("Generation 25 did not leave exactly three confirmation questions.")
    pairs = cohort_pairs()
    convergence = selected_convergence_question(g25_review)
    vwap = selected_vwap_question(g24_review)
    profiles = selected_freqai_profiles()
    return {
        "schema_version": 1,
        "batch_id": BATCH_ID,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_fresh_confirmation_outcomes",
        "plain_question": (
            "Do the previously selected busy-period forecast, pooled meme convergence "
            "area, and daily current-session VWAP behaviour repeat on two untouched "
            "chronological blocks?"
        ),
        "all_three_questions_frozen_together": True,
        "fresh_periods": list(FRESH_PERIODS),
        "data_contract": {
            "timeframe": "1h",
            "maximum_outcome_horizon_hours": MAXIMUM_OUTCOME_HOURS,
            "training_days_by_cohort": STATE_TRAINING_DAYS,
            "feature_warmup_days": FEATURE_WARMUP_DAYS,
            "required_last_candle_rule": (
                "The final in-block 1h decision at 2026-09-19 23:00 UTC needs the "
                "2026-09-20 07:00 UTC candle for its complete eight-hour future path."
            ),
            "cohort_pairs": pairs,
            "distinct_pairs": len(set(pairs["normal"]) | set(pairs["meme"])),
            "duplicate_timestamps_allowed": 0,
            "hourly_gaps_allowed_in_required_window": 0,
        },
        "siblings": [
            {
                "id": "fresh_market_state_activity",
                "route_family": "recent_ohlcv_and_cross_market_state",
                "question": (
                    "Can the frozen OHLCV-only market-state model still rank unusually "
                    "busy future volume and range periods?"
                ),
                "retained_scope_target_questions": selected_activity_questions(),
                "profiles": profiles,
                "controls": list(g25state.CONTROLS),
                "pass_rule": (
                    "Each scope/target is judged independently. It must pass both fresh "
                    "blocks against the training-median constant, shuffled training "
                    "labels, simple recent activity, and the same simple activity delayed "
                    "24 hours in all four frozen model/seed cells."
                ),
            },
            {
                "id": "fresh_meme_convergence_crossing",
                "route_family": "level_cluster_and_convergence",
                "question": (
                    "Do all 60 frozen round-number and rolling-price-distribution "
                    "definitions, pooled without winner selection, mark extra two-hour "
                    "crossing traffic across the frozen top-ten meme cohort?"
                ),
                "exact_question": convergence,
                "controls": list(g25conv.CONTROLS),
                "minimum_rows_per_pair_control_period": 10,
                "minimum_eligible_coins": 5,
                "pass_rule": (
                    "The pooled real contacts must beat every component, equal-density, "
                    "ordinary-time, near-miss, old, displaced, and recent-analogue control "
                    "in both fresh blocks, with at least 70% positive eligible coins, a "
                    "positive 95% coin-bootstrap lower bound, and no one-coin dominance."
                ),
            },
            {
                "id": "fresh_daily_current_session_vwap",
                "route_family": "isolated_anchored_price_level",
                "question": (
                    "Does the daily current-session VWAP centre, calculated only through "
                    "the previous completed hour, mark extra eight-hour crossing traffic "
                    "across normal coins?"
                ),
                "exact_question": vwap,
                "zone_half_width_atr": g24v.ZONE_HALF_WIDTH_ATR,
                "controls": list(g24v.CONTROLS),
                "minimum_rows_per_pair_control_period": 10,
                "minimum_eligible_coins": 5,
                "pass_rule": (
                    "Real VWAP contacts must beat matched ordinary times, genuine near "
                    "misses, recent historical analogues, old VWAP values, and displaced "
                    "VWAP values in both fresh blocks under the same coin-support, "
                    "bootstrap, and concentration checks."
                ),
            },
        ],
        "execution_gate": {
            "future_outcomes_may_open_only_after_all_pair_coverage_passes": True,
            "event_and_control_support_must_then_be_frozen_before_outcomes": True,
            "partial_first_block_interpretation_allowed": False,
            "threshold_or_feature_changes_after_freeze_allowed": False,
        },
        "excluded": {
            "signed_direction": True,
            "profit_or_trading_action": True,
            "new_indicator_tuning": True,
            "external_context": (
                "Excluded from this confirmation because no source has the required "
                "two-block timestamp-safe overlap."
            ),
        },
        "queued_parent_questions": queued,
        "source_contracts": {
            "analysis_script": artifact(ANALYSIS_PATH),
            "generation25_joint_review": artifact(G25_REVIEW_PATH),
            "generation24_joint_review": artifact(G24_REVIEW_PATH),
            "generation25_state_result": artifact(G25_STATE_RESULT),
            "generation25_convergence_result": artifact(G25_CONVERGENCE_RESULT),
            "generation24_freqai_profile_registry": artifact(G24_PROFILE_REGISTRY),
            "normal_cohort_manifest": artifact(g17l.COHORT_MANIFESTS["normal"]),
            "meme_cohort_manifest": artifact(g17l.COHORT_MANIFESTS["meme"]),
        },
        "future_reaction_outcomes_read": False,
        "future_signed_direction_read": False,
        "profit_read": False,
    }


def freeze() -> dict[str, Any]:
    if FREEZE_PATH.is_file():
        existing = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        if existing.get("status") != "frozen_before_fresh_confirmation_outcomes":
            raise ValueError("Existing fresh-confirmation freeze has an invalid status.")
        for item in existing["source_contracts"].values():
            path = Path(item["path"])
            if g0.sha256_file(path) != item["sha256"]:
                raise ValueError(f"Fresh-confirmation source contract changed: {path}")
        return existing
    document = build_freeze_document()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(document, FREEZE_PATH)
    return document


def coverage_inventory(frozen: dict[str, Any]) -> DataFrame:
    rows: list[dict[str, Any]] = []
    pair_map = frozen["data_contract"]["cohort_pairs"]
    for cohort in ("normal", "meme"):
        required_start, required_last = required_data_bounds(cohort)
        for pair in pair_map[cohort]:
            path = g0.ohlcv_path(str(pair), "1h")
            row: dict[str, Any] = {
                "cohort": cohort,
                "pair": pair,
                "path": str(path.resolve()),
                "required_start_utc": required_start.isoformat(),
                "required_last_candle_utc": required_last.isoformat(),
                "exists": path.is_file(),
                "rows_in_required_window": 0,
                "available_start_utc": None,
                "available_end_utc": None,
                "duplicate_timestamps": None,
                "hourly_gaps": None,
                "start_coverage_pass": False,
                "outcome_tail_coverage_pass": False,
                "pair_coverage_pass": False,
            }
            if not path.is_file():
                rows.append(row)
                continue
            dates = pd.to_datetime(
                pd.read_feather(path, columns=["date"])["date"],
                utc=True,
                errors="raise",
            ).sort_values(kind="stable")
            relevant = dates.loc[dates.ge(required_start) & dates.le(required_last)]
            duplicates = int(relevant.duplicated().sum())
            gaps = int(
                relevant.drop_duplicates().diff().dt.total_seconds().div(3600).gt(1.5).sum()
            )
            available_start = dates.min() if len(dates) else pd.NaT
            available_end = dates.max() if len(dates) else pd.NaT
            start_pass = bool(pd.notna(available_start) and available_start <= required_start)
            end_pass = bool(pd.notna(available_end) and available_end >= required_last)
            row.update(
                {
                    "rows_in_required_window": len(relevant),
                    "available_start_utc": (
                        available_start.isoformat() if pd.notna(available_start) else None
                    ),
                    "available_end_utc": (
                        available_end.isoformat() if pd.notna(available_end) else None
                    ),
                    "duplicate_timestamps": duplicates,
                    "hourly_gaps": gaps,
                    "start_coverage_pass": start_pass,
                    "outcome_tail_coverage_pass": end_pass,
                    "pair_coverage_pass": bool(
                        start_pass and end_pass and duplicates == 0 and gaps == 0
                    ),
                }
            )
            rows.append(row)
    return DataFrame.from_records(rows).sort_values(["cohort", "pair"]).reset_index(drop=True)


def coverage_summary(inventory: DataFrame) -> dict[str, Any]:
    passed = inventory["pair_coverage_pass"].fillna(False).astype(bool)
    end_values = pd.to_datetime(inventory["available_end_utc"], utc=True, errors="coerce")
    common_end = end_values.min()
    return {
        "cohort_pair_cells": len(inventory),
        "distinct_pairs": int(inventory["pair"].nunique()),
        "missing_files": int((~inventory["exists"].astype(bool)).sum()),
        "duplicate_timestamps": int(
            pd.to_numeric(inventory["duplicate_timestamps"], errors="coerce").sum()
        ),
        "hourly_gaps": int(pd.to_numeric(inventory["hourly_gaps"], errors="coerce").sum()),
        "pair_cells_passing": int(passed.sum()),
        "all_pair_coverage_pass": bool(len(inventory) == 20 and passed.all()),
        "common_latest_1h_candle_utc": (
            common_end.isoformat() if pd.notna(common_end) else None
        ),
        "required_latest_1h_candle_utc": required_data_bounds("normal")[1].isoformat(),
    }


def run_coverage(run_id: str, *, overwrite: bool = False) -> dict[str, Any]:
    frozen = freeze()
    run_dir = COVERAGE_ROOT / run_id
    result_path = run_dir / "fresh_confirmation_coverage_result.json"
    inventory_path = run_dir / "fresh_confirmation_ohlcv_inventory.csv"
    if result_path.is_file() and not overwrite:
        return json.loads(result_path.read_text(encoding="utf-8"))
    inventory = coverage_inventory(frozen)
    summary = coverage_summary(inventory)
    result = {
        "schema_version": 1,
        "batch_id": BATCH_ID,
        "run_id": run_id,
        "created_at_utc": g0.utc_now(),
        "status": (
            "ready_to_freeze_fresh_event_support"
            if summary["all_pair_coverage_pass"]
            else "waiting_for_complete_fresh_ohlcv"
        ),
        "coverage": summary,
        "next_action": (
            "Build and freeze all three event/model support surfaces before opening outcomes."
            if summary["all_pair_coverage_pass"]
            else "Wait for or download the missing hourly candles; do not open partial outcomes."
        ),
        "event_and_control_support_still_required": True,
        "future_reaction_outcomes_read": False,
        "future_signed_direction_read": False,
        "profit_read": False,
        "artifacts": {
            "freeze": artifact(FREEZE_PATH),
            "ohlcv_inventory": None,
        },
        "result_path": str(result_path.resolve()),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(inventory, inventory_path)
    result["artifacts"]["ohlcv_inventory"] = artifact(inventory_path)
    g0.atomic_write_json(result, result_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_COVERAGE_ID)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run_coverage(args.run_id, overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

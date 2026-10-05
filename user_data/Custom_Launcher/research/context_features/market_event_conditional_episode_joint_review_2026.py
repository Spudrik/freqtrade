"""Collapse the conditional episode screen into mechanism-level branch decisions."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_conditional_episode_direct_2026 as direct,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_conditional_episode_freeze_2026 as frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
OUTPUT_ROOT = direct.OUTPUT_ROOT / "joint_review_20260909a"
CANDIDATE_PATH = OUTPUT_ROOT / "mechanism_candidate_patterns.csv"
RESULT_PATH = OUTPUT_ROOT / "conditional_episode_joint_review.json"


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_contract(contract: Mapping[str, Any]) -> None:
    path = Path(str(contract["path"]))
    if not path.is_file() or g0.sha256_file(path) != contract["sha256"]:
        raise ValueError(f"Source contract changed: {path}")


def load_sources() -> tuple[dict[str, Any], DataFrame, DataFrame]:
    result = _load_json(direct.RESULT_PATH)
    if result.get("status") != "completed_conditional_episode_direct_batch":
        raise ValueError("Conditional direct batch is not terminal.")
    for contract in result["source_contracts"].values():
        _verify_contract(contract)
    for contract in result["artifacts"].values():
        _verify_contract(contract)
    decisions = pd.read_csv(direct.DECISIONS_PATH)
    cells = pd.read_csv(direct.CELLS_PATH)
    return result, decisions, cells


def coherent_patterns(decisions: DataFrame) -> DataFrame:
    retained = decisions.loc[decisions["verdict"].str.startswith("retained")].copy()
    retained = retained.loc[
        retained["event_scope"].eq("all_catalogued_events")
        | retained["event_scope"].str.startswith("class:")
    ]
    retained["effect_sign"] = np.sign(
        retained["median_conditional_effect_supported_periods"]
    ).astype(int)
    keys = ["route_id", "condition", "outcome", "event_scope", "effect_sign"]
    broad = (
        retained.groupby([*keys, "horizon_hours"], observed=True, sort=False)
        .agg(
            pair_count=("pair", "nunique"),
            pairs=("pair", lambda values: ";".join(sorted(set(values)))),
        )
        .reset_index()
    )
    broad = broad.loc[broad["pair_count"].ge(3)].copy()
    broad["pattern_kind"] = "same_horizon_at_least_three_pairs"
    broad["horizons"] = broad["horizon_hours"].astype(str)

    multi = (
        retained.groupby([*keys, "pair"], observed=True, sort=False)
        .agg(
            horizon_count=("horizon_hours", "nunique"),
            horizons=(
                "horizon_hours",
                lambda values: ";".join(map(str, sorted(set(values)))),
            ),
        )
        .reset_index()
    )
    multi = multi.loc[multi["horizon_count"].ge(2)].copy()
    multi["pattern_kind"] = "same_pair_at_least_two_horizons"
    multi["horizon_hours"] = np.nan
    multi["pair_count"] = 1
    multi["pairs"] = multi["pair"]
    columns = [
        "pattern_kind",
        "route_id",
        "condition",
        "outcome",
        "event_scope",
        "effect_sign",
        "horizon_hours",
        "horizons",
        "pair_count",
        "pairs",
    ]
    return pd.concat([broad[columns], multi[columns]], ignore_index=True)


def _has_pattern(
    patterns: DataFrame,
    *,
    condition: str,
    event_scope: str,
    effect_sign: int,
    pattern_kind: str,
    horizon: int | None = None,
) -> bool:
    mask = (
        patterns["condition"].eq(condition)
        & patterns["event_scope"].eq(event_scope)
        & patterns["effect_sign"].eq(effect_sign)
        & patterns["pattern_kind"].eq(pattern_kind)
    )
    if horizon is not None:
        mask &= patterns["horizon_hours"].eq(horizon)
    return bool(mask.any())


def _current_support(
    cells: DataFrame, *, condition: str, event_scope: str, horizon: int
) -> dict[str, Any]:
    found = cells.loc[
        cells["condition"].eq(condition)
        & cells["event_scope"].eq(event_scope)
        & cells["horizon_hours"].eq(horizon)
        & cells["model_period"].eq("confirmation_2026")
    ]
    median_effect = float(found["conditional_effect"].median()) if len(found) else None
    return {
        "maximum_true_state_episode_count": int(found["true_event_count"].max())
        if len(found)
        else 0,
        "pairs_with_at_least_ten_true_state_episodes": int(
            found.loc[found["true_event_count"].ge(10), "pair"].nunique()
        ),
        "median_conditional_effect": median_effect,
    }


def mechanism_decisions(patterns: DataFrame, cells: DataFrame) -> list[dict[str, Any]]:
    daily_scope = "class:daily_cross_market_state"
    discrete_scope = "class:discrete_media_finance_or_corporate"
    multi_broad = _has_pattern(
        patterns,
        condition="multiple_known_families_24h",
        event_scope=daily_scope,
        effect_sign=1,
        pattern_kind="same_horizon_at_least_three_pairs",
        horizon=1,
    )
    multi_horizons = _has_pattern(
        patterns,
        condition="multiple_known_families_24h",
        event_scope=daily_scope,
        effect_sign=1,
        pattern_kind="same_pair_at_least_two_horizons",
    )
    aligned_broad = _has_pattern(
        patterns,
        condition="aligned_signed_accumulation",
        event_scope="all_catalogued_events",
        effect_sign=-1,
        pattern_kind="same_horizon_at_least_three_pairs",
        horizon=1,
    )
    outer_broad = _has_pattern(
        patterns,
        condition="outer_prior_range",
        event_scope=discrete_scope,
        effect_sign=1,
        pattern_kind="same_horizon_at_least_three_pairs",
        horizon=1,
    )
    return [
        {
            "mechanism_id": "cross_market_family_overlap",
            "status": (
                "queue_controlled_branch_not_fresh_confirmed"
                if multi_broad and multi_horizons
                else "do_not_branch"
            ),
            "plain_finding": (
                "When several frozen global-market families were active within a day, "
                "daily cross-market-state episodes had more extra future volume than "
                "single-family episodes across several crypto coins."
            ),
            "role": "cross_market_overlap_or_market_wide_pressure",
            "not_proven": (
                "This is not yet evidence that several independent news stories aligned. "
                "The families may be several measurements of one global-market move."
            ),
            "current_period": _current_support(
                cells,
                condition="multiple_known_families_24h",
                event_scope=daily_scope,
                horizon=1,
            ),
            "next_test": (
                "Deduplicate related dollar yield equity fear and oil measurements into "
                "economic drivers then compare one-driver versus genuinely distinct-driver "
                "overlap on whole episodes."
            ),
        },
        {
            "mechanism_id": "aligned_signed_accumulation_suppression",
            "status": (
                "queue_source_semantics_audit_before_market_model"
                if aligned_broad
                else "do_not_branch"
            ),
            "plain_finding": (
                "In older rows, two or more same-signed catalogue items were followed by "
                "less extra volume than the comparison state in several coins."
            ),
            "role": "possible_saturation_duplicate_or_mislabelled_accumulation",
            "not_proven": (
                "Do not conclude that aligned news suppresses markets. Current support "
                "almost disappears and the signed fields may duplicate one underlying move."
            ),
            "current_period": _current_support(
                cells,
                condition="aligned_signed_accumulation",
                event_scope="all_catalogued_events",
                horizon=1,
            ),
            "next_test": (
                "Audit which source families and signs created each count and whether they "
                "were genuinely distinct known events before reading more outcomes."
            ),
        },
        {
            "mechanism_id": "outer_range_discrete_event_activity",
            "status": (
                "queue_single_horizon_confirmation_branch"
                if outer_broad
                else "do_not_branch"
            ),
            "plain_finding": (
                "Older discrete media financial or corporate episodes near the outer fifth "
                "of the prior 30-day range had more one-hour activity in three coins."
            ),
            "role": "market_location_modifier",
            "not_proven": (
                "The later periods lack enough matching outer-range episodes and the effect "
                "did not yet repeat across longer horizons."
            ),
            "current_period": _current_support(
                cells,
                condition="outer_prior_range",
                event_scope=discrete_scope,
                horizon=1,
            ),
            "next_test": (
                "Use one frozen one-hour test on more whole discrete events and keep upper "
                "and lower range edges separate only if support permits."
            ),
        },
        {
            "mechanism_id": "continuous_event_level_modifier",
            "status": "no_new_broad_branch_from_this_batch",
            "plain_finding": (
                "No continuous level or cluster condition repeated broadly enough across "
                "coins and horizons in this event-only screen."
            ),
            "role": "local_modifier_not_confirmed_here",
            "not_proven": (
                "This does not weaken the separate general evidence that calculated areas "
                "are reaction or traffic locations."
            ),
            "next_test": (
                "Do not refine from these event cells; retain the existing reaction-zone "
                "programme and wait for better event-specific support."
            ),
        },
        {
            "mechanism_id": "established_and_meme_follow_through",
            "status": "coverage_parked_more_whole_episodes_required",
            "plain_finding": (
                "The existing confirmed-Bitcoin episodes did not supply both condition "
                "states and matched controls in enough periods for a fair modifier test."
            ),
            "role": "transmission_question_unanswered",
            "not_proven": "Coverage failure is neither a positive nor a negative result.",
            "next_test": (
                "Keep collecting complete events and only test memes after Bitcoin has a "
                "verified reaction; preserve established and meme cohorts separately."
            ),
        },
        {
            "mechanism_id": "true_expectation_surprise",
            "status": "coverage_parked_missing_timestamped_consensus",
            "plain_finding": (
                "The current data still cannot distinguish actual-versus-expected news from "
                "change versus the previous release."
            ),
            "role": "signed_driver_question_unanswered",
            "not_proven": "No expectation-based direction test was performed.",
            "next_test": (
                "Reconstruct a timestamped consensus source or capture it prospectively; "
                "do not substitute current signed proxies."
            ),
        },
    ]


def run(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        existing = _load_json(RESULT_PATH)
        if existing.get("status") != "completed_conditional_episode_joint_review":
            raise ValueError("Existing joint review is not terminal.")
        return existing
    direct_result, decisions, cells = load_sources()
    patterns = coherent_patterns(decisions)
    mechanisms = mechanism_decisions(patterns, cells)
    g0.atomic_write_csv(patterns, CANDIDATE_PATH)
    result = {
        "schema_version": 1,
        "status": "completed_conditional_episode_joint_review",
        "created_at_utc": g0.utc_now(),
        "run_id": frozen.RUN_ID,
        "raw_decision_cells": len(decisions),
        "raw_retained_cells": int(
            decisions["verdict"].str.startswith("retained").sum()
        ),
        "period_dependent_unresolved_cells": int(
            decisions["verdict"].str.startswith("unresolved").sum()
        ),
        "coherent_cross_pair_or_multi_horizon_patterns": len(patterns),
        "mechanism_decisions": mechanisms,
        "batch_decision": (
            "The complete sibling batch has been reviewed. Queue only the three named "
            "mechanism branches; do not treat raw cells as independent discoveries. "
            "Expectation and follower routes remain coverage-parked and event-level "
            "modification produced no new broad branch."
        ),
        "freqai_decision": (
            "Do not launch a broad FreqAI grid. First run the deduplication and narrowly "
            "controlled direct checks. Use FreqAI later only if a readable relationship "
            "survives and needs magnitude ranking or abstention calibration."
        ),
        "scope": {
            "profit_used": False,
            "trading_rule_tested": False,
            "fresh_confirmation_claimed": False,
            "descendant_branches_launched": False,
        },
        "source_contracts": {
            "analysis_script": artifact(ANALYSIS_PATH),
            "direct_result": artifact(direct.RESULT_PATH),
            "direct_decisions": artifact(direct.DECISIONS_PATH),
            "direct_cells": artifact(direct.CELLS_PATH),
        },
        "artifacts": {"candidate_patterns": artifact(CANDIDATE_PATH)},
        "parent_route_summary": direct_result["route_summary"],
    }
    g0.atomic_write_json(result, RESULT_PATH)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = run(overwrite=args.overwrite)
    print(json.dumps(result, indent=2, default=g0.json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Review every Generation 19 sibling together and queue Generation 20 breadth."""

from __future__ import annotations

# Repository-local imports follow the root-path bootstrap.
# ruff: noqa: E402
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
    market_reaction_zone_freqai_generation19 as g19f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation19_acceptance_zone as g19a,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation19_freeze as g19z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation19_recross_analysis as g19r,
)


DEFAULT_REVIEW_ID = "g19_joint_review_20260827a"
REVIEW_ROOT = g19z.OUTPUT_ROOT / "joint_review"
RECROSS_RESULT = (
    g19r.RECORD_ROOT / g19r.DEFAULT_RUN_ID / "g19_recross_analysis_result.json"
)
ACCEPTANCE_RESULT = (
    g19a.RECORD_ROOT / g19a.DEFAULT_RUN_ID / "g19_acceptance_zone_result.json"
)
FREQAI_RESULT = (
    g19f.RECORD_ROOT
    / g19f.DEFAULT_JOINT_REVIEW_ID
    / "g19_freqai_joint_review.json"
)


def artifact(path: Path) -> dict[str, Any]:
    return g19z.artifact(path)


def load_result(path: Path, status: str) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != status:
        raise ValueError(f"Non-terminal Generation 19 result: {path}")
    return result


def verify_artifacts(result: dict[str, Any]) -> None:
    for item in result.get("artifacts", {}).values():
        path = Path(item["path"])
        if not path.is_file() or g0.sha256_file(path) != item["sha256"]:
            raise ValueError(f"Generation 19 artifact changed or disappeared: {path}")


def read_artifact(result: dict[str, Any], name: str) -> DataFrame:
    item = result["artifacts"][name]
    path = Path(item["path"])
    if g0.sha256_file(path) != item["sha256"]:
        raise ValueError(f"Generation 19 artifact changed before review: {path}")
    return pd.read_csv(path)


def branch_statuses(
    frozen: dict[str, Any],
    recross: dict[str, Any],
    acceptance: dict[str, Any],
    freqai: dict[str, Any],
) -> list[dict[str, Any]]:
    completed = {
        *recross["branches_completed"],
        acceptance["branch_completed"],
        freqai["branch_completed"],
    }
    rows: list[dict[str, Any]] = []
    for branch in frozen["branches"]:
        branch_id = branch["branch_id"]
        if branch_id in completed:
            status = "completed_before_joint_review"
        elif str(branch["status_at_freeze"]).startswith("parked_"):
            status = str(branch["status_at_freeze"])
        else:
            status = "missing_terminal_result"
        rows.append(
            {
                "branch_id": branch_id,
                "status": status,
                "plain_question": branch["plain_question"],
            }
        )
    if len(rows) != 7 or any(row["status"] == "missing_terminal_result" for row in rows):
        raise ValueError("Not every Generation 19 sibling is completed or honestly parked.")
    return rows


def next_queue() -> list[dict[str, Any]]:
    return [
        {
            "branch_id": "g20a_volume_profile_traffic_state_attribution",
            "status": "queued_active",
            "route_family": "causal_traffic_state_attribution",
            "source_result": "g19d persistent adaptive-Volume-Profile traffic",
            "observation": (
                "Adaptive Volume Profile areas retained persistent recross traffic before "
                "and after contact across established coins, memes, and rational groups."
            ),
            "plain_hypothesis": (
                "The current Volume Profile area adds information about sustained traffic "
                "after matching the traffic already present before contact."
            ),
            "alternative_explanation": (
                "Volume Profile merely labels an area that was already busy and adds no "
                "information beyond recent crossings, volume, range, and volatility."
            ),
            "controls": [
                "matched_precontact_crossing_and_volume_state",
                "near_miss",
                "stale_72h",
                "price_shift",
            ],
            "smallest_useful_test": (
                "One direct matched-state comparison over the frozen normal and meme cohorts."
            ),
            "required_data": "existing causal 1h/4h/8h level and OHLCV cache",
            "cost": "medium",
            "earliest_eligible_generation": 20,
        },
        {
            "branch_id": "g20b_donchian_onset_causal_isolation",
            "status": "queued_active",
            "route_family": "boundary_onset_mechanism",
            "source_result": "g19d new Donchian recross onset",
            "observation": (
                "Donchian contacts retained new recross onset across the all-normal group "
                "after two- and four-hour quiet prewindows."
            ),
            "plain_hypothesis": (
                "A current Donchian boundary locates newly starting traffic beyond an "
                "ordinary breakout or volatility-expansion state."
            ),
            "alternative_explanation": (
                "The generic breakout candle and expanding range explain the onset; the "
                "calculated boundary itself adds nothing."
            ),
            "controls": [
                "breakout_state_matched_no_boundary",
                "range_and_volume_matched",
                "stale_boundary",
                "price_shift",
            ],
            "smallest_useful_test": (
                "One direct onset comparison using the predeclared 2h/4h prewindows and "
                "1h/2h/4h/8h postwindows."
            ),
            "required_data": "existing causal Donchian and OHLCV cache",
            "cost": "medium",
            "earliest_eligible_generation": 20,
        },
        {
            "branch_id": "g20c_level_local_activity_regime_stability",
            "status": "queued_active",
            "route_family": "context_regime_stability",
            "source_result": "g19b level-plus-local-activity two-hour volume lead",
            "observation": (
                "Level plus local activity was strict for two-hour volume in standard "
                "normal and meme cells but failed the later normal chronology."
            ),
            "plain_hypothesis": (
                "The combination helps only in a predeclared moderate-activity or "
                "non-saturated regime and becomes redundant when activity is already extreme."
            ),
            "alternative_explanation": "The two standard-cell result was temporal overfit.",
            "controls": [
                "level_only",
                "local_activity_only",
                "stale_level_plus_activity",
                "identical_rows_across_regimes",
            ],
            "smallest_useful_test": (
                "Direct frozen activity-state stratification first; one low-dimensional "
                "FreqAI ladder only if the direct relationship repeats."
            ),
            "required_data": "existing level, local activity, and chronology caches",
            "cost": "medium",
            "earliest_eligible_generation": 20,
        },
        {
            "branch_id": "g20d_change_point_anchored_vwap_level",
            "status": "queued_active",
            "route_family": "new_rational_level_source",
            "source_result": "Objective 02b rational new-indicator breadth route",
            "observation": (
                "Current families locate traffic, but exact coordinates remain narrow and "
                "the repeated-close acceptance representation failed."
            ),
            "plain_hypothesis": (
                "A causally confirmed activity change point followed by anchored VWAP and "
                "volume-weighted dispersion bands locates a distinct reaction area."
            ),
            "alternative_explanation": (
                "Any recent anchor or ordinary rolling VWAP performs equally well."
            ),
            "controls": [
                "rolling_vwap_width_matched",
                "random_recent_anchor",
                "stale_anchor",
                "price_shift",
                "near_miss",
            ],
            "smallest_useful_test": (
                "One compact causal direct screen with a single frozen change-point "
                "definition before considering variants."
            ),
            "required_data": "existing OHLCV; no indicator baseline edit",
            "cost": "medium",
            "earliest_eligible_generation": 20,
        },
        {
            "branch_id": "g20e_donchian_onset_one_minute_replay",
            "status": "queued_active_bounded_direction_lane",
            "route_family": "evidence_triggered_one_minute_microscope",
            "source_result": "g19d all-normal Donchian new-onset pattern",
            "observation": (
                "The direction-neutral Donchian onset pattern is distinct and repeated, "
                "meeting the gate for one bounded lower-timeframe diagnostic."
            ),
            "plain_hypothesis": (
                "Causal 1m approach pressure and immediate path state distinguish movement "
                "through from movement away during this specific onset pattern."
            ),
            "alternative_explanation": (
                "Simple approach trend or majority path performs equally well."
            ),
            "controls": [
                "majority_path",
                "simple_approach_trend",
                "no_level_matched_episode",
            ],
            "smallest_useful_test": (
                "Freeze 6-12 independent episodes without signed-path inspection, then run "
                "one direct aligned 1m pass; no automatic FreqAI descendant."
            ),
            "required_data": "targeted 1m OHLCV around frozen episodes",
            "cost": "bounded_medium",
            "earliest_eligible_generation": 20,
        },
        {
            "branch_id": "g20f_meme_lvn_fresh_data_confirmation",
            "status": "parked_waiting_for_independent_data",
            "route_family": "market_specific_replication",
            "source_result": "g19c meme-specific 72h/96-bin LVN crossing",
            "observation": (
                "The exact LVN retained only under the predeclared meme label."
            ),
            "plain_hypothesis": "The meme-specific LVN relationship repeats on fresh data.",
            "alternative_explanation": "The exact coordinate is a historical coincidence.",
            "controls": ["same four location controls"],
            "smallest_useful_test": "Repeat only after a genuinely new meme holdout exists.",
            "required_data": "new independent meme-coin chronology",
            "cost": "deferred",
            "earliest_eligible_generation": 20,
        },
        {
            "branch_id": "g20g_external_context_accumulation",
            "status": "parked_coverage",
            "route_family": "external_context",
            "source_result": "g19f parked source coverage",
            "observation": "No external source spans two complete required blocks.",
            "plain_hypothesis": (
                "Orderbook, news, or global context may condition otherwise similar reactions."
            ),
            "alternative_explanation": "Any apparent gain is short-window source selection.",
            "controls": ["OHLCV_only_identical_rows", "source_stale", "source_shuffled"],
            "smallest_useful_test": "Re-audit only after two complete timestamp-ready blocks.",
            "required_data": "future timestamp-safe source accumulation",
            "cost": "deferred",
            "earliest_eligible_generation": 20,
        },
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-id", default=DEFAULT_REVIEW_ID)
    args = parser.parse_args(argv)
    frozen = json.loads(g19z.FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation19_outcomes":
        raise ValueError("Generation 19 freeze is invalid.")
    recross = load_result(RECROSS_RESULT, "completed_generation19_recross_analysis")
    acceptance = load_result(
        ACCEPTANCE_RESULT, "completed_generation19_acceptance_zone"
    )
    freqai = load_result(FREQAI_RESULT, "completed_generation19_freqai_joint_review")
    for result in (recross, acceptance, freqai):
        verify_artifacts(result)
    statuses = branch_statuses(frozen, recross, acceptance, freqai)
    coordinate = read_artifact(recross, "coordinate_decisions")
    market = read_artifact(recross, "market_mechanism_decisions")
    timing = read_artifact(recross, "timing_decisions")
    acceptance_decisions = read_artifact(acceptance, "decisions")
    freqai_decisions = read_artifact(freqai, "joint_decisions")
    strict_coordinate = coordinate.loc[
        coordinate["status"].eq("strict_holdout_confirmation")
    ]
    strict_market = market.loc[market["status"].eq("strict_holdout_confirmation")]
    strict_timing = timing.loc[timing["status"].eq("strict_holdout_confirmation")]
    strict_acceptance = acceptance_decisions.loc[
        acceptance_decisions["status"].eq("strict_holdout_confirmation")
    ]
    strict_freqai = freqai_decisions.loc[
        freqai_decisions["strict_at_least_two_cells"].astype(bool)
    ]
    point_freqai = freqai_decisions.loc[
        freqai_decisions["point_at_least_two_cells"].astype(bool)
    ]
    queue = next_queue()
    review_dir = REVIEW_ROOT / args.review_id
    review_dir.mkdir(parents=True, exist_ok=True)
    statuses_path = review_dir / "g19_sibling_statuses.csv"
    queue_path = review_dir / "g20_sibling_branch_queue.json"
    result_path = review_dir / "g19_joint_review.json"
    g0.atomic_write_csv(DataFrame.from_records(statuses), statuses_path)
    g0.atomic_write_json(
        {
            "schema_version": 1,
            "generation": 20,
            "created_at_utc": g0.utc_now(),
            "status": "queued_after_complete_generation19_joint_review",
            "active_siblings": sum(
                str(item["status"]).startswith("queued_active") for item in queue
            ),
            "parked_siblings": sum(str(item["status"]).startswith("parked_") for item in queue),
            "siblings": queue,
        },
        queue_path,
    )
    output = {
        "schema_version": 1,
        "generation": 19,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation19_joint_review",
        "all_siblings_terminal_or_parked_before_review": True,
        "summary": {
            "strict_coordinate_rows": len(strict_coordinate),
            "strict_market_specific_rows": len(strict_market),
            "strict_timing_rows": len(strict_timing),
            "strict_acceptance_rows": len(strict_acceptance),
            "freqai_strict_at_least_two_cells": len(strict_freqai),
            "freqai_point_at_least_two_cells": len(point_freqai),
            "freqai_normal_standard_and_recent": int(
                freqai_decisions["normal_standard_and_recent_point"].astype(bool).sum()
            ),
            "joint_reaction_and_direction_at_or_above_55pct": False,
        },
        "interpretation": {
            "adaptive_volume_profile": (
                "retained as a persistent busy-area marker; contact-caused onset was not shown"
            ),
            "donchian_boundaries": (
                "retained as a new-recross-onset lead in established coins, pending generic "
                "breakout-state isolation"
            ),
            "level_plus_local_activity": (
                "retained as a regime-sensitive two-hour volume lead in standard normal and "
                "meme cells; rejected as broad chronology-stable because later normal failed"
            ),
            "repeated_close_acceptance": "rejected after all five controls",
            "direction": "still unproven; only one bounded Donchian-onset replay is queued",
        },
        "research_boundary": {
            "profit_used": False,
            "future_signed_direction_used_in_main_batch": False,
            "trading_promotion": False,
            "descendant_executed_during_review": False,
        },
        "source_results": {
            "generation19_freeze": artifact(g19z.FREEZE_PATH),
            "recross": artifact(RECROSS_RESULT),
            "acceptance": artifact(ACCEPTANCE_RESULT),
            "freqai": artifact(FREQAI_RESULT),
        },
        "artifacts": {
            "sibling_statuses": artifact(statuses_path),
            "generation20_queue": artifact(queue_path),
        },
    }
    g0.atomic_write_json(output, result_path)
    print(json.dumps({**output, "result_path": str(result_path.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

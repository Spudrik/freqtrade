"""Freeze Generation 20's complete breadth-first sibling batch before outcomes."""

from __future__ import annotations

# Repository-local imports follow the root-path bootstrap.
# ruff: noqa: E402
import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation19_joint_review as g19j,
)


DEFAULT_BATCH_ID = "g20_broad_siblings_20260827a"
OUTPUT_ROOT = g19j.REVIEW_ROOT.parent / "g20_broad_siblings"
FREEZE_PATH = OUTPUT_ROOT.parent / "g20_broad_siblings_freeze_20260827a.json"
PARENT_REVIEW_PATH = (
    g19j.REVIEW_ROOT / g19j.DEFAULT_REVIEW_ID / "g19_joint_review.json"
)
PARENT_QUEUE_PATH = (
    g19j.REVIEW_ROOT / g19j.DEFAULT_REVIEW_ID / "g20_sibling_branch_queue.json"
)
HORIZONS_HOURS = (1, 2, 4, 8)
LATER_PERIODS_REQUIRED = 2
MIN_NORMAL_GROUP_COINS = 5
ONE_MINUTE_EPISODE_MINIMUM = 6
ONE_MINUTE_EPISODE_MAXIMUM = 12


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def load_parent() -> tuple[dict[str, Any], dict[str, Any]]:
    review = json.loads(PARENT_REVIEW_PATH.read_text(encoding="utf-8"))
    queue = json.loads(PARENT_QUEUE_PATH.read_text(encoding="utf-8"))
    if review.get("status") != "completed_generation19_joint_review":
        raise ValueError("Generation 19 joint review is not terminal.")
    if not review.get("all_siblings_terminal_or_parked_before_review"):
        raise ValueError("Generation 19 sibling layer was not complete before review.")
    if queue.get("status") != "queued_after_complete_generation19_joint_review":
        raise ValueError("Generation 20 sibling queue is invalid.")
    if queue.get("active_siblings") != 5 or queue.get("parked_siblings") != 2:
        raise ValueError("Generation 20 queue does not preserve five active routes.")
    return review, queue


def branch_definitions() -> list[dict[str, Any]]:
    return [
        {
            "branch_id": "g20a_volume_profile_traffic_state_attribution",
            "status_at_freeze": "frozen_pending",
            "route_family": "causal_traffic_state_attribution",
            "plain_question": (
                "Does a current adaptive Volume Profile area add information about "
                "continued traffic after matching the traffic already underway?"
            ),
            "surface": "adaptive_volume_profile_nodes",
            "precontact_windows_hours": [2, 4],
            "precontact_matching_fields": [
                "crossing_count",
                "relative_volume",
                "range_over_atr",
                "absolute_return_over_atr",
            ],
            "controls": [
                "matched_random_time",
                "near_miss",
                "stale_72h",
                "price_shift",
            ],
            "targets": [
                f"{metric}_h{horizon}"
                for metric in ("any_recross", "repeated_recross", "crossing_count")
                for horizon in HORIZONS_HOURS
            ],
            "pass_rule": (
                "Retain only if the current Profile area beats all four controls after "
                "precontact-state matching in both later blocks under a broad or honestly "
                "predeclared market-group label. A persistent result is a traffic-state "
                "lead, not proof that contact caused the activity."
            ),
        },
        {
            "branch_id": "g20b_donchian_onset_causal_isolation",
            "status_at_freeze": "frozen_pending",
            "route_family": "boundary_onset_mechanism",
            "plain_question": (
                "Does the current Donchian boundary identify newly starting traffic beyond "
                "an otherwise similar breakout or volatility-expansion candle?"
            ),
            "surface": "donchian_boundaries",
            "precontact_windows_hours": [2, 4],
            "onset_definition": "zero prewindow crossings and at least one postwindow crossing",
            "state_matching_fields": [
                "absolute_return_over_atr",
                "range_over_atr",
                "relative_volume",
                "close_location",
                "pretrend_over_atr",
            ],
            "controls": [
                "breakout_state_matched_no_current_boundary_contact",
                "near_miss",
                "stale_72h",
                "price_shift",
            ],
            "postcontact_windows_hours": list(HORIZONS_HOURS),
            "pass_rule": (
                "Retain only if new onset exceeds every control in both later blocks for "
                "the all-normal group. BTC-only results remain asset-specific."
            ),
        },
        {
            "branch_id": "g20c_level_local_activity_regime_stability",
            "status_at_freeze": "frozen_pending_direct_only",
            "route_family": "context_regime_stability",
            "plain_question": (
                "Is the Generation 19 level-plus-local-activity volume lead confined to a "
                "causally named activity regime, or was it temporal overfit?"
            ),
            "activity_score_inputs": [
                "relative_volume",
                "volume_acceleration",
                "prior_range_atr",
                "atr_fraction",
            ],
            "activity_states": ["quiet", "moderate", "extreme"],
            "state_boundaries": "outcome-blind within-pair tertiles frozen before targets",
            "controls": [
                "matched_random_time",
                "near_miss",
                "stale_72h",
                "price_shift",
            ],
            "primary_target": "future_volume_ratio_h2",
            "secondary_targets": [
                f"future_volume_ratio_h{horizon}" for horizon in HORIZONS_HOURS
            ],
            "pass_rule": (
                "A named state must preserve a positive current-level increment against all "
                "controls in both later normal blocks. Generation 20 performs the direct "
                "test only; any FreqAI descendant waits for the complete joint review."
            ),
        },
        {
            "branch_id": "g20d_change_point_anchored_vwap_level",
            "status_at_freeze": "frozen_pending",
            "route_family": "new_rational_level_source",
            "plain_question": (
                "Do causal change-point-anchored VWAP and dispersion bands locate reaction "
                "areas beyond ordinary rolling VWAP and artificial coordinates?"
            ),
            "change_point_definition": {
                "completed_candle_only": True,
                "volume_zscore_minimum": 2.0,
                "range_over_atr_minimum": 1.5,
                "absolute_return_over_atr_minimum": 0.75,
                "maximum_anchor_age_hours": 168,
            },
            "levels": [
                "anchored_vwap",
                "anchored_vwap_plus_1p5_sigma",
                "anchored_vwap_minus_1p5_sigma",
            ],
            "zone_half_width_atr": 0.25,
            "controls": [
                "rolling_vwap_168h_width_matched",
                "random_recent_anchor",
                "near_miss",
                "stale_72h",
                "price_shift",
            ],
            "targets": [
                f"{metric}_h{horizon}"
                for metric in ("any_recross", "repeated_recross", "dwell_fraction", "volume_ratio")
                for horizon in HORIZONS_HOURS
            ],
            "pass_rule": (
                "Retain only a named coordinate that beats every control in both later "
                "blocks without widening or changing the anchor definition after outcomes."
            ),
        },
        {
            "branch_id": "g20e_donchian_onset_one_minute_replay",
            "status_at_freeze": "frozen_pending_bounded_direction_lane",
            "route_family": "evidence_triggered_one_minute_microscope",
            "plain_question": (
                "Within independently selected Donchian new-onset episodes, can causal 1m "
                "approach pressure distinguish the immediate path through from away?"
            ),
            "episode_selection": {
                "cohort": "normal",
                "surface": "donchian_boundaries",
                "prewindow_hours": 4,
                "prewindow_crossings": 0,
                "postwindow_hours": 4,
                "minimum_postwindow_crossings": 1,
                "independence_gap_hours": 24,
                "minimum_episodes": ONE_MINUTE_EPISODE_MINIMUM,
                "maximum_episodes": ONE_MINUTE_EPISODE_MAXIMUM,
                "signed_path_opened_only_after_selection_freeze": True,
            },
            "replay_window": {"hours_before": 24, "hours_after": 24},
            "fixed_calls": [
                "approach_trend_15m",
                "approach_trend_60m",
                "signed_volume_pressure_15m",
                "signed_volume_pressure_60m",
                "momentum_vote_15m",
                "combined_pressure_and_momentum_vote",
            ],
            "controls": [
                "majority_path",
                "simple_approach_trend",
                "matched_no_level_episode",
            ],
            "success_reporting": [
                "reaction_success",
                "conditional_direction_success",
                "joint_success",
                "coverage",
                "abstentions",
            ],
            "pass_rule": (
                "Report the complete fixed call set. A lead requires at least 55% joint "
                "success on issued calls, adequate coverage, and improvement over all "
                "controls. No FreqAI run or descendant occurs inside Generation 20."
            ),
        },
        {
            "branch_id": "g20f_meme_lvn_fresh_data_confirmation",
            "status_at_freeze": "parked_waiting_for_independent_data",
            "route_family": "market_specific_replication",
            "plain_question": (
                "Does the exact meme LVN relationship repeat on genuinely new data?"
            ),
            "park_reason": "No independent later meme block yet exists.",
        },
        {
            "branch_id": "g20g_external_context_accumulation",
            "status_at_freeze": "parked_coverage",
            "route_family": "external_context",
            "plain_question": (
                "Do orderbook, news, or global sources condition level reactions after two "
                "complete timestamp-ready blocks exist?"
            ),
            "park_reason": "No source currently spans two complete required later blocks.",
        },
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    args = parser.parse_args(argv)
    review, queue = load_parent()
    branches = branch_definitions()
    queued_ids = {item["branch_id"] for item in queue["siblings"]}
    frozen_ids = {item["branch_id"] for item in branches}
    if queued_ids != frozen_ids:
        raise ValueError("Generation 20 freeze does not match the reviewed queue.")
    active = [
        item for item in branches if str(item["status_at_freeze"]).startswith("frozen_")
    ]
    parked = [
        item for item in branches if str(item["status_at_freeze"]).startswith("parked_")
    ]
    if len(active) != 5 or len(parked) != 2:
        raise ValueError("Generation 20 must freeze five active and two parked siblings.")
    payload = {
        "schema_version": 1,
        "generation": 20,
        "batch_id": args.batch_id,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation20_outcomes",
        "all_siblings_frozen_together": True,
        "active_siblings": len(active),
        "parked_siblings": len(parked),
        "horizons_hours": list(HORIZONS_HOURS),
        "later_periods_required": LATER_PERIODS_REQUIRED,
        "minimum_normal_group_coins": MIN_NORMAL_GROUP_COINS,
        "branches": branches,
        "batch_rule": (
            "Implement and preflight every active sibling before opening any Generation 20 "
            "outcome. Finish all five before one joint review queues descendants."
        ),
        "research_boundary": {
            "profit_used": False,
            "main_batch_signed_direction_used": False,
            "one_minute_lane_is_only_directional_exception": True,
            "no_trading_promotion": True,
        },
        "source_contracts": {
            "generation19_joint_review": artifact(PARENT_REVIEW_PATH),
            "generation20_queue": artifact(PARENT_QUEUE_PATH),
        },
        "parent_summary": review["summary"],
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if FREEZE_PATH.is_file():
        existing = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        left = {key: value for key, value in existing.items() if key != "created_at_utc"}
        right = {key: value for key, value in payload.items() if key != "created_at_utc"}
        if left != right:
            raise ValueError("Generation 20 freeze changed after its first write.")
    else:
        g0.atomic_write_json(payload, FREEZE_PATH)
    print(json.dumps({**payload, "freeze_path": str(FREEZE_PATH.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

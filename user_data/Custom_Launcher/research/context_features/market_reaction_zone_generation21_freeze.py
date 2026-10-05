"""Freeze Generation 21's complete five-route breadth batch before outcomes."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
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
    market_reaction_zone_generation20_joint_review as g20j,
)


DEFAULT_BATCH_ID = "g21_broad_siblings_20260827a"
OUTPUT_ROOT = g20j.REVIEW_ROOT.parent / "g21_broad_siblings"
FREEZE_PATH = g20j.REVIEW_ROOT.parent / "g21_broad_siblings_freeze_20260827a.json"
PARENT_REVIEW = g20j.REVIEW_ROOT / g20j.DEFAULT_REVIEW_ID / "g20_joint_review.json"
PARENT_QUEUE = g20j.REVIEW_ROOT / g20j.DEFAULT_REVIEW_ID / "g21_sibling_branch_queue.json"
HORIZONS_HOURS = (1, 2, 4, 8)
LOCATION_CONTROLS = ("matched_random_time", "near_miss", "stale_72h", "price_shift")
ONE_MINUTE_THRESHOLDS_ATR = (0.5, 1.0)
ONE_MINUTE_HORIZONS = (15, 60, 240)


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def load_parent() -> tuple[dict[str, Any], dict[str, Any]]:
    review = json.loads(PARENT_REVIEW.read_text(encoding="utf-8"))
    queue = json.loads(PARENT_QUEUE.read_text(encoding="utf-8"))
    if review.get("status") != "completed_generation20_joint_review":
        raise ValueError("Generation 20 joint review is not terminal.")
    if not review.get("all_five_active_siblings_terminal_before_review"):
        raise ValueError("Generation 20 siblings were not complete before review.")
    if queue.get("status") != "queued_after_complete_generation20_joint_review":
        raise ValueError("Generation 21 queue is invalid.")
    if queue.get("active_siblings") != 5 or queue.get("parked_siblings") != 2:
        raise ValueError("Generation 21 queue lost its five-route breadth.")
    return review, queue


def branch_definitions() -> list[dict[str, Any]]:
    return [
        {
            "branch_id": "g21a_btc_volume_profile_role_specificity",
            "status_at_freeze": "frozen_pending",
            "route_family": "asset_specific_level_mechanism",
            "plain_question": (
                "Which causal Volume Profile role, if any, explains BTC's repeated "
                "four-hour crossing result, and is it honestly absent or different in "
                "predeclared non-BTC groups?"
            ),
            "role_groups": {
                "point_of_control": ["poc"],
                "value_area_boundary": ["nearest_value_boundary"],
                "high_volume_node": ["nearest_hvn_q80", "nearest_hvn_q90"],
                "low_volume_node": ["nearest_lvn_q10", "nearest_lvn_q20"],
            },
            "profile_lookbacks_hours": [72, 168, 720],
            "profile_bin_modes": [24, 48, 96, "fd"],
            "precontact_state_window_hours": 4,
            "state_matching_fields": [
                "crossing_count",
                "relative_volume",
                "range_over_atr",
                "absolute_return_over_atr",
            ],
            "primary_target": "repeated_recross_h4",
            "secondary_targets": [
                f"{metric}_h{horizon}"
                for metric in ("any_recross", "repeated_recross", "crossing_count")
                for horizon in (2, 4, 8)
            ],
            "controls": list(LOCATION_CONTROLS),
            "market_scopes": [
                "btc_separate",
                "smart_contract_platforms",
                "other_established_alts",
                "top_ten_memes",
            ],
            "pass_rule": (
                "A named role must beat all four controls in both later blocks. BTC "
                "remains explicitly asset-specific unless a predeclared non-BTC group "
                "independently passes."
            ),
        },
        {
            "branch_id": "g21b_multitimeframe_market_state_reaction_atlas",
            "status_at_freeze": "frozen_pending",
            "route_family": "level_free_market_state",
            "plain_question": (
                "Without calculated levels, do completed volume, range, volatility, "
                "pressure, and trend states identify unusually active future periods?"
            ),
            "source_timeframes": ["1h", "4h", "8h"],
            "state_blocks": {
                "volume_pressure": [
                    "relative_volume",
                    "volume_acceleration",
                    "absolute_pressure",
                    "pressure_persistence",
                ],
                "volatility_range": [
                    "atr_fraction",
                    "prior_range_atr",
                    "bollinger_width",
                    "range_contraction",
                ],
                "trend_momentum": [
                    "ema20_slope",
                    "ma_separation",
                    "return_slope",
                    "return_acceleration",
                    "adx14",
                    "rsi14_centered",
                    "macd_histogram",
                ],
            },
            "activity_states": ["quiet", "moderate", "extreme"],
            "state_thresholds": "pre-confirmation within-pair tertiles",
            "targets": [
                f"{metric}_forward_{steps}_source_candles"
                for metric in (
                    "volume_ratio",
                    "range_ratio",
                    "absolute_displacement_atr",
                )
                for steps in (1, 2, 4)
            ],
            "controls": [
                "matched_moderate_state",
                "causal_72h_stale_state",
                "within_pair_time_shuffle",
                "same_time_other_coin_state",
            ],
            "direction_prediction": False,
            "pass_rule": (
                "A named state block and source timeframe must beat all controls in both "
                "later blocks under a broad or predeclared coin-group label."
            ),
        },
        {
            "branch_id": "g21c_state_matched_single_and_cluster_levels",
            "status_at_freeze": "frozen_pending",
            "route_family": "single_vs_cluster_geometry",
            "plain_question": (
                "After matching traffic already underway, do isolated Volume Profile "
                "levels or clusters containing independent level families add reaction "
                "information?"
            ),
            "geometry_states": {
                "isolated_single": "independent_family_count == 1",
                "two_family_cluster": "independent_family_count == 2",
                "three_plus_family_cluster": "independent_family_count >= 3",
            },
            "anchor_family": "adaptive_volume_profile_nodes",
            "precontact_state_window_hours": 4,
            "state_matching_fields": [
                "crossing_count",
                "relative_volume",
                "range_over_atr",
                "absolute_return_over_atr",
            ],
            "targets": [
                f"{metric}_h{horizon}"
                for metric in ("any_recross", "repeated_recross", "crossing_count")
                for horizon in (2, 4, 8)
            ],
            "controls": list(LOCATION_CONTROLS),
            "pass_rule": (
                "Singles and clusters are separate co-equal questions. A geometry state "
                "must beat all controls in both later blocks without borrowing a result "
                "from another geometry."
            ),
        },
        {
            "branch_id": "g21d_abnormal_candle_structural_levels",
            "status_at_freeze": "frozen_pending",
            "route_family": "new_rational_level_source",
            "plain_question": (
                "Do fixed prices from a causal high-volume expansion candle locate later "
                "reaction areas better than artificial prices?"
            ),
            "event_definition": {
                "completed_candle_only": True,
                "volume_zscore_minimum": 2.0,
                "range_over_atr_minimum": 1.5,
                "absolute_return_over_atr_minimum": 0.75,
                "maximum_level_age_hours": 168,
            },
            "levels": [
                "event_open",
                "event_close",
                "event_body_midpoint",
                "event_high",
                "event_low",
                "event_typical_price",
            ],
            "zone_half_width_atr": 0.25,
            "controls": [
                "matched_random_time",
                "random_recent_high_activity_candle",
                "near_miss",
                "stale_72h",
                "price_shift",
            ],
            "targets": [
                f"{metric}_h{horizon}"
                for metric in (
                    "any_recross",
                    "repeated_recross",
                    "dwell_fraction",
                    "volume_ratio",
                )
                for horizon in HORIZONS_HOURS
            ],
            "pass_rule": (
                "A named structural price must beat every control in both later blocks. "
                "The fixed event definition cannot be tuned after outcomes."
            ),
        },
        {
            "branch_id": "g21e_disjoint_one_minute_rejection_diagnostic",
            "status_at_freeze": "frozen_pending_bounded_direction_lane",
            "route_family": "bounded_direction_microscope",
            "plain_question": (
                "On Donchian episodes disjoint from Generation 20, is immediate rejection "
                "repeatable at more meaningful distances, and can causal pressure or "
                "momentum beat a naive rejection rule?"
            ),
            "selection": {
                "exclude_generation20_episode_times": True,
                "future_signed_path_used": False,
                "prewindow_crossings": 0,
                "prewindow_hours": 4,
                "minimum_postwindow_crossings": 1,
                "postwindow_hours": 4,
                "independence_hours": 24,
                "minimum_episodes": 12,
                "maximum_episodes": 18,
            },
            "distance_thresholds_atr": list(ONE_MINUTE_THRESHOLDS_ATR),
            "horizons_minutes": list(ONE_MINUTE_HORIZONS),
            "candidate_calls": [
                "approach_trend_15m",
                "approach_trend_60m",
                "signed_volume_pressure_15m",
                "signed_volume_pressure_60m",
                "momentum_vote_15m",
                "combined_pressure_and_momentum_vote",
            ],
            "controls": [
                "naive_rejection",
                "majority_path",
                "matched_no_level_episode",
            ],
            "pass_rule": (
                "Report every fixed distance/horizon/call. A lead needs at least 55% "
                "joint success, at least 50% call coverage, and improvement over every "
                "control; it remains exploratory pending new-date confirmation."
            ),
        },
        {
            "branch_id": "g21f_change_point_avwap_refinement",
            "status_at_freeze": "parked_rejected_parent",
            "route_family": "rejected_level_source",
            "park_reason": (
                "No Generation 20 AVWAP coordinate passed the five-control ladder; "
                "same-date parameter tuning would be event fitting."
            ),
        },
        {
            "branch_id": "g21g_external_context_conditioning",
            "status_at_freeze": "parked_coverage",
            "route_family": "external_context",
            "park_reason": (
                "No timestamp-ready external source yet spans two complete evaluation blocks."
            ),
        },
    ]


def freeze(*, overwrite: bool = False) -> dict[str, Any]:
    review, queue = load_parent()
    if FREEZE_PATH.is_file() and not overwrite:
        frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        if frozen.get("status") != "frozen_before_generation21_outcomes":
            raise ValueError("Invalid existing Generation 21 freeze.")
        return frozen
    branches = branch_definitions()
    active = [item for item in branches if item["status_at_freeze"].startswith("frozen_")]
    parked = [item for item in branches if item["status_at_freeze"].startswith("parked_")]
    frozen = {
        "schema_version": 1,
        "generation": 21,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation21_outcomes",
        "batch_id": DEFAULT_BATCH_ID,
        "all_siblings_frozen_together": True,
        "active_siblings": len(active),
        "parked_siblings": len(parked),
        "batch_rule": (
            "Implement and preflight all five active siblings before opening any "
            "Generation 21 outcome; jointly review all five before descendants."
        ),
        "branches": branches,
        "research_boundary": {
            "profit_used": False,
            "main_batch_signed_direction_used": False,
            "one_minute_lane_is_only_directional_exception": True,
            "no_trading_promotion": True,
        },
        "parent_summary": review["breadth_assessment"],
        "source_contracts": {
            "generation20_joint_review": artifact(PARENT_REVIEW),
            "generation21_queue": artifact(PARENT_QUEUE),
        },
        "queue_active_siblings": queue["active_siblings"],
    }
    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(frozen, FREEZE_PATH)
    return frozen


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(freeze(overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

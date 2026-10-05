"""Freeze Generation 22's complete five-route breadth batch before outcomes."""

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
    market_reaction_zone_generation21_joint_review as g21j,
)


DEFAULT_BATCH_ID = "g22_broad_siblings_20260827a"
OUTPUT_ROOT = g21j.REVIEW_ROOT.parent / "g22_broad_siblings"
FREEZE_PATH = g21j.REVIEW_ROOT.parent / "g22_broad_siblings_freeze_20260827a.json"
PARENT_REVIEW = g21j.REVIEW_ROOT / g21j.DEFAULT_REVIEW_ID / "g21_joint_review.json"
PARENT_QUEUE = g21j.REVIEW_ROOT / g21j.DEFAULT_REVIEW_ID / "g22_sibling_branch_queue.json"
HORIZONS_HOURS = (1, 2, 4, 8)
LEVEL_CONTROLS = (
    "matched_random_time",
    "random_recent_analogue",
    "near_miss",
    "stale_definition",
    "price_shift",
)
MARKET_SCOPES = (
    "btc_separate",
    "smart_contract_platforms",
    "other_established_alts",
    "all_normal",
    "top_ten_memes",
)


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def load_parent() -> tuple[dict[str, Any], dict[str, Any]]:
    review = json.loads(PARENT_REVIEW.read_text(encoding="utf-8"))
    queue = json.loads(PARENT_QUEUE.read_text(encoding="utf-8"))
    if review.get("status") != "completed_generation21_joint_review":
        raise ValueError("Generation 21 joint review is not terminal.")
    if not review.get("all_five_active_siblings_terminal_before_review"):
        raise ValueError("Generation 21 siblings were not complete before review.")
    if queue.get("status") != "queued_after_complete_generation21_joint_review":
        raise ValueError("Generation 22 queue is invalid.")
    if queue.get("active_siblings") != 5 or queue.get("parked_siblings") != 3:
        raise ValueError("Generation 22 queue lost its five-route breadth.")
    return review, queue


def branch_definitions() -> list[dict[str, Any]]:
    return [
        {
            "branch_id": "g22a_completed_period_price_landmarks",
            "status_at_freeze": "frozen_pending",
            "route_family": "historical_price_landmarks",
            "plain_question": (
                "Do prices fixed by the last completed UTC day, ISO week, and calendar "
                "month locate later unsigned price/volume reaction areas?"
            ),
            "source_periods": ["previous_utc_day", "previous_iso_week", "previous_utc_month"],
            "coordinates": [
                "period_open",
                "period_high",
                "period_low",
                "period_close",
                "period_range_midpoint",
                "period_typical_price",
            ],
            "zone_half_width_atr": 0.25,
            "controls": list(LEVEL_CONTROLS),
            "targets": [
                f"{metric}_h{horizon}"
                for metric in (
                    "any_recross",
                    "repeated_recross",
                    "crossing_count",
                    "dwell_fraction",
                    "future_volume_ratio",
                )
                for horizon in HORIZONS_HOURS
            ],
            "pass_rule": (
                "A named period and coordinate must beat all five controls in both later "
                "blocks under a predeclared coin-group scope. Sparse monthly contacts are "
                "reported as incomplete, never promoted."
            ),
        },
        {
            "branch_id": "g22b_multitimeframe_level_convergence_and_precedence",
            "status_at_freeze": "frozen_pending",
            "route_family": "multitimeframe_level_convergence",
            "plain_question": (
                "Do isolated levels and independently calculated 1h, 4h, daily, and "
                "weekly clusters differ in reaction strength, and does the highest "
                "timeframe present add repeatable evidence?"
            ),
            "source_timeframes": ["1h", "4h", "1d", "1w"],
            "causal_level_families": [
                "donchian_prior_boundary",
                "rolling_volume_profile_poc",
                "completed_period_high_low",
                "causal_pivot",
            ],
            "cluster_radius_atr": 0.50,
            "geometry_states": [
                "isolated_single_timeframe",
                "two_timeframe_cluster",
                "three_plus_timeframe_cluster",
            ],
            "highest_timeframe_states": ["1h", "4h", "1d", "1w"],
            "singles_are_coequal": True,
            "zone_half_width_atr": 0.25,
            "controls": list(LEVEL_CONTROLS),
            "targets": [
                f"{metric}_h{horizon}"
                for metric in (
                    "any_recross",
                    "repeated_recross",
                    "crossing_count",
                    "dwell_fraction",
                    "future_volume_ratio",
                )
                for horizon in HORIZONS_HOURS
            ],
            "pass_rule": (
                "Each geometry and highest-timeframe state is a separate question and "
                "must beat all controls in both later blocks. No higher-timeframe "
                "precedence is assumed in advance."
            ),
        },
        {
            "branch_id": "g22c_causal_trend_channel_boundaries",
            "status_at_freeze": "frozen_pending",
            "route_family": "dynamic_trendline_levels",
            "plain_question": (
                "Do one-step-ahead prices from completed-candle log-price trend channels "
                "locate repeatable reaction zones?"
            ),
            "lookback_hours": [72, 168, 720],
            "fit": "ordinary least squares of log close through t-1, extrapolated once to t",
            "coordinates": ["trend_centre", "upper_1p5_residual_sd", "lower_1p5_residual_sd"],
            "zone_half_width_atr": 0.25,
            "controls": [
                "matched_random_time",
                "flat_rolling_log_mean",
                "near_miss",
                "stale_72h",
                "price_shift",
            ],
            "targets": [
                f"{metric}_h{horizon}"
                for metric in (
                    "any_recross",
                    "repeated_recross",
                    "crossing_count",
                    "dwell_fraction",
                    "future_volume_ratio",
                )
                for horizon in HORIZONS_HOURS
            ],
            "pass_rule": (
                "A fixed lookback and coordinate must beat all five controls in both later "
                "blocks. The regression and channel multiplier cannot be tuned on results."
            ),
        },
        {
            "branch_id": "g22d_cross_asset_context_conditioned_level_reactions",
            "status_at_freeze": "frozen_pending",
            "route_family": "cross_asset_market_context",
            "plain_question": (
                "Do quiet, synchronized, dispersed, or BTC-led completed market states "
                "change the unsigned reaction around a fixed broad basket of causal levels?"
            ),
            "context_windows_hours": [4, 24],
            "context_inputs": [
                "btc_absolute_return_over_atr",
                "equal_weight_median_absolute_return_over_atr",
                "equal_weight_relative_volume",
                "cross_coin_return_dispersion",
                "cross_coin_absolute_return_correlation",
            ],
            "context_states": [
                "quiet_coherent",
                "broad_synchronized_activity",
                "dispersed_idiosyncratic_activity",
                "btc_led_activity",
            ],
            "thresholds": "pre-confirmation normal-plus-meme causal tertiles",
            "level_basket": [
                "adaptive_volume_profile_nodes",
                "donchian_boundaries",
                "rolling_vwap_deviation_bands",
                "weekly_pivot_grid",
            ],
            "controls": list(LEVEL_CONTROLS),
            "required_same_state_no_level_control": True,
            "targets": [
                f"{metric}_h{horizon}"
                for metric in (
                    "any_recross",
                    "repeated_recross",
                    "crossing_count",
                    "dwell_fraction",
                    "future_volume_ratio",
                )
                for horizon in HORIZONS_HOURS
            ],
            "pass_rule": (
                "A context state must show that actual level contacts beat all artificial "
                "locations and same-state no-level times in both later blocks. Context "
                "alone is not evidence that a level mattered."
            ),
        },
        {
            "branch_id": "g22e_freqai_unsigned_reaction_interaction_regression",
            "status_at_freeze": "frozen_pending_freqai",
            "route_family": "low_dimensional_freqai_interactions",
            "plain_question": (
                "Can a small causal FreqAI regression rank multiple unsigned reaction "
                "behaviours from level geometry plus market state better than simpler models?"
            ),
            "required_tool": "FreqAI",
            "maximum_feature_columns_before_fixed_encodings": 12,
            "features": [
                "nearest_level_distance_atr",
                "independent_level_family_count_within_0p5atr",
                "source_role_fixed_encoding",
                "source_timeframe_fixed_encoding",
                "pre_crossing_count_4h",
                "relative_volume",
                "range_over_atr_4h",
                "absolute_return_over_atr_4h",
                "atr_fraction",
                "btc_absolute_activity_4h",
                "equal_weight_market_activity_4h",
                "cross_coin_dispersion_4h",
            ],
            "continuous_targets": [
                f"{metric}_h{horizon}"
                for metric in (
                    "maximum_absolute_excursion_atr",
                    "future_volume_ratio",
                    "future_range_ratio",
                    "crossing_count",
                    "dwell_fraction",
                )
                for horizon in HORIZONS_HOURS
            ],
            "controls": [
                "constant_training_median",
                "level_geometry_only_model",
                "market_state_only_model",
                "within_pair_time_shuffled_training_labels",
            ],
            "training_and_evaluation": {
                "walk_forward_only": True,
                "pre_confirmation_training_only_for_first_later_block": True,
                "later_blocks_scored_separately": True,
                "pair_and_time_purge_hours": 8,
                "profit_target_used": False,
                "signed_direction_used": False,
            },
            "reporting": [
                "median_absolute_error",
                "spearman_rank_correlation",
                "top_quartile_actual_reaction_mean",
                "bottom_quartile_actual_reaction_mean",
                "equal_coin_top_minus_bottom_difference",
            ],
            "pass_rule": (
                "For a named target and horizon, the full interaction model must beat all "
                "four controls in both later blocks, retain positive equal-coin top-minus-"
                "bottom separation, and place at least 55% of its top-quartile rows above "
                "the target's frozen training-median reaction threshold. This creates an "
                "exploratory reaction lead only, never strategy promotion."
            ),
        },
        {
            "branch_id": "g22f_immediate_one_minute_direction_refinement",
            "status_at_freeze": "parked_failed_direction_parent",
            "route_family": "bounded_direction_microscope",
            "park_reason": (
                "Generation 21's best call was 52.9% and lost to the 64.7% majority "
                "control. Return to reaction discovery before another direction batch."
            ),
        },
        {
            "branch_id": "g22g_volume_profile_role_or_geometry_tuning",
            "status_at_freeze": "parked_incomplete_or_rejected_parent",
            "route_family": "same_holdout_level_tuning",
            "park_reason": (
                "No complete role or geometry ladder survived; do not tune definitions "
                "from incomplete positive rows."
            ),
        },
        {
            "branch_id": "g22h_external_news_orderbook_web_conditioning",
            "status_at_freeze": "parked_coverage",
            "route_family": "external_context",
            "park_reason": (
                "Continue collection and re-audit later; current timestamp-ready sources "
                "still do not span two complete broad evaluation blocks."
            ),
        },
    ]


def freeze(*, overwrite: bool = False) -> dict[str, Any]:
    review, queue = load_parent()
    if FREEZE_PATH.is_file() and not overwrite:
        frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        if frozen.get("status") != "frozen_before_generation22_outcomes":
            raise ValueError("Invalid existing Generation 22 freeze.")
        return frozen
    branches = branch_definitions()
    active = [item for item in branches if item["status_at_freeze"].startswith("frozen_")]
    parked = [item for item in branches if item["status_at_freeze"].startswith("parked_")]
    frozen = {
        "schema_version": 1,
        "generation": 22,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation22_outcomes",
        "batch_id": DEFAULT_BATCH_ID,
        "all_siblings_frozen_together": True,
        "active_siblings": len(active),
        "parked_siblings": len(parked),
        "batch_rule": (
            "Implement and preflight all five active siblings before opening any "
            "Generation 22 outcome; jointly review all five before descendants."
        ),
        "branches": branches,
        "market_scopes": list(MARKET_SCOPES),
        "research_boundary": {
            "profit_used": False,
            "signed_direction_used": False,
            "freqai_is_unsigned_reaction_regression_only": True,
            "no_trading_promotion": True,
        },
        "parent_summary": review["breadth_assessment"],
        "source_contracts": {
            "generation21_joint_review": artifact(PARENT_REVIEW),
            "generation22_queue": artifact(PARENT_QUEUE),
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

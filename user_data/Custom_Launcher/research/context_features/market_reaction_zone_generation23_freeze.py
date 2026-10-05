"""Freeze Generation 23's balanced five-route batch before any outcomes."""

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
    market_reaction_zone_generation22_joint_review as g22j,
)


DEFAULT_BATCH_ID = "g23_broad_siblings_20260827a"
OUTPUT_ROOT = g22j.REVIEW_ROOT.parent / "g23_broad_siblings"
FREEZE_PATH = g22j.REVIEW_ROOT.parent / "g23_broad_siblings_freeze_20260827a.json"
PARENT_REVIEW = g22j.REVIEW_ROOT / g22j.DEFAULT_REVIEW_ID / "g22_joint_review.json"
PARENT_QUEUE = g22j.REVIEW_ROOT / g22j.DEFAULT_REVIEW_ID / "g23_sibling_branch_queue.json"
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
    if review.get("status") != "completed_generation22_joint_review":
        raise ValueError("Generation 22 joint review is not terminal.")
    if not review.get("all_five_active_siblings_terminal_before_review"):
        raise ValueError("Generation 22 siblings were not complete before review.")
    if queue.get("status") != "queued_after_complete_generation22_joint_review":
        raise ValueError("Generation 23 queue is invalid.")
    if queue.get("active_siblings") != 5 or queue.get("parked_siblings") != 5:
        raise ValueError("Generation 23 queue lost its five-route breadth.")
    return review, queue


def branch_definitions() -> list[dict[str, Any]]:
    return [
        {
            "branch_id": "g23a_multitimeframe_cluster_incremental_attribution",
            "status_at_freeze": "frozen_pending",
            "route_family": "multitimeframe_cluster_attribution",
            "plain_question": (
                "Does the fixed Generation 22 three-plus-timeframe cluster add unsigned "
                "reaction evidence beyond its component levels and fake same-density "
                "clusters?"
            ),
            "parent_primary_cell": {
                "geometry": "three_plus_timeframe_cluster",
                "metric": "crossing_count",
                "horizon_hours": 8,
                "market_scope": "all_normal",
            },
            "source_timeframes": ["1h", "4h", "1d", "1w"],
            "cluster_radius_atr": 0.50,
            "zone_half_width_atr": 0.25,
            "attribution_comparisons": [
                "matched_isolated_anchor_component",
                "highest_timeframe_component",
                "nearest_non_anchor_component",
                "same_time_current_close_pseudo_cluster",
            ],
            "parent_controls": list(LEVEL_CONTROLS),
            "secondary_targets": [
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
                "The primary cell must beat every component, pseudo-cluster, and parent "
                "control in both later blocks. Any positive result is same-holdout "
                "mechanism attribution only and cannot be called confirmation."
            ),
        },
        {
            "branch_id": "g23b_continuous_cross_asset_context_at_levels",
            "status_at_freeze": "frozen_pending",
            "route_family": "continuous_cross_asset_context",
            "plain_question": (
                "Do individual causal cross-asset context measurements, tested in broad "
                "bands, change unsigned reactions at a fixed level basket?"
            ),
            "context_windows_hours": [4, 24],
            "context_inputs": [
                "btc_absolute_return_over_atr",
                "equal_weight_median_absolute_return_over_atr",
                "equal_weight_relative_volume",
                "cross_coin_return_dispersion",
                "cross_coin_absolute_return_correlation",
            ],
            "band_definition": (
                "low, middle, and high training-calibration tertiles for one input at a "
                "time; no rare multi-input conjunctions"
            ),
            "controls": [*LEVEL_CONTROLS, "same_band_no_level_time"],
            "continuous_rank_association_is_diagnostic_only": True,
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
                "A named single-input band must beat all five artificial level controls "
                "and same-band no-level times in both later blocks. Context alone is not "
                "evidence that a level mattered."
            ),
        },
        {
            "branch_id": "g23c_freqai_calibrated_unsigned_reaction",
            "status_at_freeze": "frozen_pending_freqai",
            "route_family": "freqai_unsigned_reaction_calibration",
            "plain_question": (
                "Can the fixed 12-feature FreqAI design model pair-normalized unsigned "
                "reaction residuals or training-distribution ranks better than raw values?"
            ),
            "required_tool": "FreqAI",
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
            "raw_targets": [
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
            "target_transforms": [
                "pair_training_median_residual",
                "pair_training_empirical_percentile",
            ],
            "controls": [
                "constant_training_median",
                "level_geometry_only_model",
                "market_state_only_model",
                "within_pair_time_shuffled_training_labels",
                "generation22_uncalibrated_parent",
            ],
            "training_and_evaluation": {
                "walk_forward_only": True,
                "same_later_blocks_as_generation22": True,
                "pair_and_time_purge_hours": 8,
                "profit_target_used": False,
                "signed_direction_used": False,
            },
            "pass_rule": (
                "For one frozen transform, target, and horizon, the full model must beat "
                "all five controls in both later blocks, keep positive equal-coin rank "
                "separation, and place at least 55% of its highest prediction quartile "
                "above the frozen training median. This is exploratory only."
            ),
        },
        {
            "branch_id": "g23d_round_number_and_price_grid_zones",
            "status_at_freeze": "frozen_pending",
            "route_family": "price_scale_round_number_levels",
            "plain_question": (
                "Do causal round-number grids at three fixed decimal scales locate "
                "repeatable unsigned reactions across normal and meme coins?"
            ),
            "base_step": "10 ** (floor(log10(previous_close)) - 1)",
            "step_multipliers": [1.0, 0.5, 0.25],
            "level_selection": "nearest grid line to previous completed close",
            "zone_half_width_atr": 0.25,
            "controls": [
                "matched_random_time",
                "random_mantissa_grid",
                "half_step_phase_shifted_grid",
                "near_miss",
                "stale_72h",
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
                "A fixed grid scale must beat all five controls in both later blocks "
                "under a predeclared market scope. The decimal scale cannot be tuned on "
                "the evaluation outcomes."
            ),
        },
        {
            "branch_id": "g23e_causal_price_distribution_boundaries",
            "status_at_freeze": "frozen_pending",
            "route_family": "rolling_price_distribution_levels",
            "plain_question": (
                "Do fixed quantiles of completed hourly typical prices locate later "
                "unsigned reaction zones?"
            ),
            "lookback_hours": [24, 72, 168, 720],
            "quantiles": [0.10, 0.25, 0.50, 0.75, 0.90],
            "input_price": "completed_candle_typical_price",
            "causal_shift_hours": 1,
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
                "A fixed lookback and quantile must beat all five controls in both later "
                "blocks. Windows and quantiles cannot be selected or moved after outcomes."
            ),
        },
        {
            "branch_id": "g23f_multitimeframe_new_period_confirmation",
            "status_at_freeze": "parked_awaiting_new_data",
            "route_family": "untouched_future_confirmation",
            "park_reason": (
                "Preserve the exact Generation 22 cluster definition until two genuinely "
                "later untouched blocks are available."
            ),
        },
        {
            "branch_id": "g23g_period_landmark_refinement",
            "status_at_freeze": "parked_rejected_parent",
            "route_family": "historical_period_landmarks",
            "park_reason": "Weak complete effects with uncertainty crossing zero.",
        },
        {
            "branch_id": "g23h_trend_channel_refinement",
            "status_at_freeze": "parked_rejected_parent",
            "route_family": "trend_channel_levels",
            "park_reason": "Effectively flat complete ladders do not justify tuning.",
        },
        {
            "branch_id": "g23i_one_minute_direction_replay",
            "status_at_freeze": "parked_no_retained_reaction_parent",
            "route_family": "bounded_direction_microscope",
            "park_reason": "No retained direction-neutral parent reaction pattern.",
        },
        {
            "branch_id": "g23j_external_news_orderbook_web_context",
            "status_at_freeze": "parked_coverage",
            "route_family": "external_context",
            "park_reason": "Continue collection and re-audit timestamp coverage later.",
        },
    ]


def freeze(*, overwrite: bool = False) -> dict[str, Any]:
    review, queue = load_parent()
    if FREEZE_PATH.is_file() and not overwrite:
        frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        if frozen.get("status") != "frozen_before_generation23_outcomes":
            raise ValueError("Invalid existing Generation 23 freeze.")
        return frozen
    branches = branch_definitions()
    active = [item for item in branches if item["status_at_freeze"].startswith("frozen_")]
    parked = [item for item in branches if item["status_at_freeze"].startswith("parked_")]
    frozen = {
        "schema_version": 1,
        "generation": 23,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation23_outcomes",
        "batch_id": DEFAULT_BATCH_ID,
        "all_siblings_frozen_together": True,
        "active_siblings": len(active),
        "parked_siblings": len(parked),
        "batch_rule": (
            "Implement and preflight all five active siblings before opening any "
            "Generation 23 outcome; jointly review all five before descendants."
        ),
        "same_holdout_boundary": (
            "Attribution, representation repair, and calibration routes reusing the "
            "Generation 22 dates remain exploratory and cannot provide untouched confirmation."
        ),
        "branches": branches,
        "market_scopes": list(MARKET_SCOPES),
        "research_boundary": {
            "profit_used": False,
            "signed_direction_used": False,
            "freqai_is_unsigned_reaction_regression_only": True,
            "no_trading_promotion": True,
            "one_minute_direction_lane_open": False,
        },
        "parent_summary": review["breadth_assessment"],
        "source_contracts": {
            "generation22_joint_review": artifact(PARENT_REVIEW),
            "generation23_queue": artifact(PARENT_QUEUE),
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

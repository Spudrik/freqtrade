"""Freeze Generation 24's balanced five-route batch before any outcomes."""

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
    market_reaction_zone_generation23_joint_review as g23j,
)


DEFAULT_BATCH_ID = "g24_broad_siblings_20260828a"
OUTPUT_ROOT = g23j.REVIEW_ROOT.parent / "g24_broad_siblings"
FREEZE_PATH = g23j.REVIEW_ROOT.parent / "g24_broad_siblings_freeze_20260828a.json"
PARENT_REVIEW = g23j.REVIEW_ROOT / g23j.DEFAULT_REVIEW_ID / "g23_joint_review.json"
PARENT_QUEUE = g23j.REVIEW_ROOT / g23j.DEFAULT_REVIEW_ID / "g24_sibling_branch_queue.json"
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
    if review.get("status") != "completed_generation23_joint_review":
        raise ValueError("Generation 23 joint review is not terminal.")
    if not review.get("all_five_active_siblings_terminal_before_review"):
        raise ValueError("Generation 23 siblings were not complete before review.")
    if queue.get("status") != "queued_after_complete_generation23_joint_review":
        raise ValueError("Generation 24 queue is invalid.")
    if queue.get("active_siblings") != 5 or queue.get("parked_siblings") != 5:
        raise ValueError("Generation 24 queue lost its five-route breadth.")
    return review, queue


def reaction_targets() -> list[str]:
    return [
        f"{metric}_h{horizon}"
        for metric in (
            "any_recross",
            "repeated_recross",
            "crossing_count",
            "dwell_fraction",
            "future_volume_ratio",
        )
        for horizon in HORIZONS_HOURS
    ]


def branch_definitions() -> list[dict[str, Any]]:
    activity_targets = [
        f"{metric}_h{horizon}"
        for metric in ("future_volume_ratio", "future_range_ratio")
        for horizon in HORIZONS_HOURS
    ]
    return [
        {
            "branch_id": "g24a_freqai_activity_lead_stability_and_attribution",
            "status_at_freeze": "frozen_pending_freqai",
            "route_family": "freqai_unsigned_activity_robustness",
            "plain_question": (
                "Does pair-percentile FreqAI activity ranking survive fixed seeds and "
                "model families, and do level inputs add repeatable information beyond "
                "market state?"
            ),
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
            "targets": activity_targets,
            "target_transform": "pair_training_empirical_percentile",
            "candidate_models": [
                {
                    "model": "LightGBMRegressorMultiTarget",
                    "seeds": [2026082401, 2026082402, 2026082403],
                },
                {"model": "XGBoostRegressorMultiTarget", "seeds": [2026082401]},
            ],
            "controls": [
                "constant_training_median",
                "level_geometry_only_model",
                "market_state_only_model",
                "within_pair_time_shuffled_training_labels",
                "generation23_calibrated_parent",
                "permuted_level_feature_block",
            ],
            "pass_rule": (
                "A target and scope must retain positive equal-coin separation and at "
                "least 55% top-group-above-training-median in both later blocks, beat "
                "all six controls, and retain the effect across all three LightGBM seeds "
                "and the frozen XGBoost check. Same-holdout robustness is exploratory."
            ),
        },
        {
            "branch_id": "g24b_cross_asset_context_no_level_coverage_repair",
            "status_at_freeze": "frozen_pending",
            "route_family": "continuous_context_level_attribution",
            "plain_question": (
                "Do broad cross-asset states change reactions specifically at levels "
                "when adequately represented same-state no-level times are compared?"
            ),
            "context_windows_hours": [4, 24],
            "context_inputs": [
                "btc_absolute_return_over_atr",
                "equal_weight_median_absolute_return_over_atr",
                "equal_weight_relative_volume",
                "cross_coin_return_dispersion",
                "cross_coin_absolute_return_correlation",
            ],
            "bands": ["low", "middle", "high"],
            "no_level_minimum_spacing_hours": 2,
            "winner_filtering_allowed": False,
            "controls": [*LEVEL_CONTROLS, "same_band_no_level_time"],
            "targets": reaction_targets(),
            "pass_rule": (
                "One frozen input and band must beat all five artificial level controls "
                "and the adequately populated same-band no-level control in both periods."
            ),
        },
        {
            "branch_id": "g24c_round_distribution_convergence_attribution",
            "status_at_freeze": "frozen_pending",
            "route_family": "cross_family_level_convergence",
            "plain_question": (
                "Does convergence between causal round grids and rolling price quantiles "
                "add unsigned reaction evidence beyond either component?"
            ),
            "round_step_multipliers": [1.0, 0.5, 0.25],
            "distribution_lookback_hours": [24, 72, 168, 720],
            "distribution_quantiles": [0.10, 0.25, 0.50, 0.75, 0.90],
            "convergence_radius_atr": 0.25,
            "zone_half_width_atr": 0.25,
            "winner_filtering_allowed": False,
            "controls": [
                "isolated_round_component",
                "isolated_distribution_component",
                "equal_density_pseudo_convergence",
                *LEVEL_CONTROLS,
            ],
            "targets": reaction_targets(),
            "pass_rule": (
                "A predeclared component combination must beat both isolated components, "
                "the fake-convergence control, and all five location controls in both "
                "periods. This is same-holdout mechanism attribution only."
            ),
        },
        {
            "branch_id": "g24d_causal_anchored_vwap_zones",
            "status_at_freeze": "frozen_pending",
            "route_family": "anchored_volume_weighted_levels",
            "plain_question": (
                "Do causal anchored VWAP centres and volume-weighted dispersion bands "
                "locate repeatable unsigned reaction zones?"
            ),
            "anchor_timeframes": ["1d", "1w", "1M"],
            "anchor_modes": [
                "current_session_through_previous_completed_candle",
                "previous_completed_session",
            ],
            "dispersion_multipliers": [0.0, 1.0, 2.0],
            "minimum_completed_candles": {"1d": 6, "1w": 24, "1M": 72},
            "zone_half_width_atr": 0.25,
            "controls": list(LEVEL_CONTROLS),
            "targets": reaction_targets(),
            "pass_rule": (
                "A fixed anchor, mode, and band must beat all five controls in both later "
                "blocks under a predeclared market scope."
            ),
        },
        {
            "branch_id": "g24e_generic_multitimeframe_indicator_context",
            "status_at_freeze": "frozen_pending",
            "route_family": "generic_technical_context_at_levels",
            "plain_question": (
                "Do broad causal generic-indicator states alter unsigned reactions at a "
                "fixed level basket?"
            ),
            "timeframes": ["1h", "4h", "1d"],
            "indicator_inputs": [
                "rsi_14",
                "bollinger_position_20_2",
                "bollinger_width_20_2",
                "macd_histogram_12_26_9_over_atr",
                "ema_20_50_spread_over_atr",
                "ema_20_slope_over_atr",
                "atr_fraction_14",
            ],
            "bands": ["low", "middle", "high"],
            "cross_timeframe_agreement": ["all_low", "all_high"],
            "arbitrary_multi_indicator_conjunctions_allowed": False,
            "controls": [*LEVEL_CONTROLS, "same_state_no_level_time"],
            "targets": reaction_targets(),
            "pass_rule": (
                "A single indicator band or frozen same-indicator timeframe agreement "
                "must beat all five artificial level controls and same-state no-level "
                "times in both periods."
            ),
        },
        {
            "branch_id": "g24f_freqai_untouched_later_confirmation",
            "status_at_freeze": "parked_awaiting_new_data",
            "route_family": "untouched_future_confirmation",
            "park_reason": "Not enough genuinely later data exists for two untouched blocks.",
        },
        {
            "branch_id": "g24g_one_minute_direction_replay",
            "status_at_freeze": "parked_no_price_reaction_parent",
            "route_family": "bounded_direction_microscope",
            "park_reason": "The retained parent forecasts volume, not a price reaction zone.",
        },
        {
            "branch_id": "g24h_external_news_orderbook_web_context",
            "status_at_freeze": "parked_coverage_audit_continues",
            "route_family": "external_context",
            "park_reason": "Keep collecting until clean overlap supports a full batch.",
        },
        {
            "branch_id": "g24i_multitimeframe_cluster_tuning",
            "status_at_freeze": "parked_rejected_attribution",
            "route_family": "multitimeframe_cluster_attribution",
            "park_reason": "The complete Generation 23 attribution ladder failed.",
        },
        {
            "branch_id": "g24j_standalone_round_or_distribution_tuning",
            "status_at_freeze": "parked_weak_or_incomplete_parent",
            "route_family": "standalone_level_refinement",
            "park_reason": "Test rational convergence rather than tune incomplete winners.",
        },
    ]


def freeze(*, overwrite: bool = False) -> dict[str, Any]:
    review, queue = load_parent()
    if FREEZE_PATH.is_file() and not overwrite:
        frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        if frozen.get("status") != "frozen_before_generation24_outcomes":
            raise ValueError("Invalid existing Generation 24 freeze.")
        return frozen
    branches = branch_definitions()
    active = [item for item in branches if item["status_at_freeze"].startswith("frozen_")]
    parked = [item for item in branches if item["status_at_freeze"].startswith("parked_")]
    frozen = {
        "schema_version": 1,
        "generation": 24,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation24_outcomes",
        "batch_id": DEFAULT_BATCH_ID,
        "all_siblings_frozen_together": True,
        "active_siblings": len(active),
        "parked_siblings": len(parked),
        "batch_rule": (
            "Implement and preflight all five active siblings before opening any "
            "Generation 24 outcome; jointly review all five before descendants."
        ),
        "same_holdout_boundary": (
            "Robustness, coverage repair, and convergence routes reusing Generation 23 "
            "dates remain exploratory and cannot provide untouched confirmation."
        ),
        "branches": branches,
        "market_scopes": list(MARKET_SCOPES),
        "research_boundary": {
            "profit_used": False,
            "signed_direction_used": False,
            "no_trading_promotion": True,
            "one_minute_direction_lane_open": False,
        },
        "parent_summary": review["breadth_assessment"],
        "source_contracts": {
            "generation23_joint_review": artifact(PARENT_REVIEW),
            "generation24_queue": artifact(PARENT_QUEUE),
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

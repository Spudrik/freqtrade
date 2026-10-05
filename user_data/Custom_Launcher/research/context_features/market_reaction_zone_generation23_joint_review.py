"""Jointly review all Generation 23 siblings and queue a balanced Generation 24."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import Series
from pandas.api.types import is_bool_dtype


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation23 as g23f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_continuous_context as g23b,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_freeze as g23z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_multitimeframe_attribution as g23a,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_price_distribution as g23e,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_round_numbers as g23d,
)


ANALYSIS_PATH = Path(__file__).resolve()
DEFAULT_REVIEW_ID = "g23_joint_review_20260828a"
REVIEW_ROOT = g23z.OUTPUT_ROOT / "joint_review"

DIRECT_ROUTES = {
    "multitimeframe_incremental_attribution": {
        "result": g23a.RECORD_ROOT
        / g23a.DEFAULT_RUN_ID
        / "g23_multitimeframe_attribution_result.json",
        "controls": tuple(g23a.CONTROLS),
    },
    "continuous_cross_asset_context": {
        "result": g23b.RECORD_ROOT
        / g23b.DEFAULT_RUN_ID
        / "g23_continuous_context_result.json",
        "controls": tuple(g23b.CONTROLS),
    },
    "round_number_grids": {
        "result": g23d.RECORD_ROOT
        / g23d.DEFAULT_RUN_ID
        / "g23_round_numbers_result.json",
        "controls": tuple(g23d.CONTROLS),
    },
    "rolling_price_distributions": {
        "result": g23e.RECORD_ROOT
        / g23e.DEFAULT_RUN_ID
        / "g23_price_distribution_result.json",
        "controls": tuple(g23e.CONTROLS),
    },
}
FREQAI_NORMAL_RESULT = (
    g23f.RECORD_ROOT / f"{g23f.DEFAULT_RUN_STEM}_normal" / "g23_freqai_result.json"
)
FREQAI_MEME_RESULT = (
    g23f.RECORD_ROOT / f"{g23f.DEFAULT_RUN_STEM}_meme" / "g23_freqai_result.json"
)
FREQAI_JOINT_RESULT = (
    g23f.RECORD_ROOT
    / g23f.DEFAULT_JOINT_REVIEW_ID
    / "g23_freqai_joint_review.json"
)


def artifact(path: Path) -> dict[str, Any]:
    return g23z.artifact(path)


def as_bool(series: Series) -> Series:
    if is_bool_dtype(series.dtype):
        return series.fillna(False)
    return series.astype("string").str.strip().str.lower().eq("true")


def generation24_queue() -> list[dict[str, Any]]:
    """Keep five different evidence routes active; do not follow only the ML lead."""
    return [
        {
            "branch_id": "g24a_freqai_activity_lead_stability_and_attribution",
            "status": "active_pending_freeze",
            "route_family": "freqai_unsigned_activity_robustness",
            "parent_evidence": "g23c_empirical_percentile_future_volume_h2_all_normal",
            "plain_question": (
                "Does the calibrated FreqAI volume-activity lead survive different fixed "
                "seeds and simple model families, and do level features add repeatable "
                "information beyond market-state features?"
            ),
            "scope_guard": (
                "Freeze the whole unsigned volume/range percentile family across normal "
                "and meme cohorts, preserve all Generation 23 controls, add seed/model "
                "stability and feature-block permutation controls, and make no price-"
                "direction or trading claim."
            ),
        },
        {
            "branch_id": "g24b_cross_asset_context_no_level_coverage_repair",
            "status": "active_pending_freeze",
            "route_family": "continuous_context_level_attribution",
            "parent_evidence": "g23b_broad_bands_with_missing_same_band_no_level_cells",
            "plain_question": (
                "When broad cross-asset states occur, do level contacts react more than "
                "properly matched same-state times that are not near a level?"
            ),
            "scope_guard": (
                "Repair the no-level sampling for every frozen input and band, not only "
                "the best BTC rows. Keep the five artificial level controls, minimum "
                "row rules, both periods, and all market scopes unchanged."
            ),
        },
        {
            "branch_id": "g24c_round_distribution_convergence_attribution",
            "status": "active_pending_freeze",
            "route_family": "cross_family_level_convergence",
            "parent_evidence": "g23d_and_g23e_partial_round_and_distribution_patterns",
            "plain_question": (
                "Do causal round-number grids and rolling price-distribution boundaries "
                "produce more unsigned reaction when they converge than either level "
                "family does alone?"
            ),
            "scope_guard": (
                "Use the already frozen scales, lookbacks, quantiles, and widths. Compare "
                "convergence with both isolated components, equal-density fake clusters, "
                "stale definitions, shifts, near misses, and matched random times."
            ),
        },
        {
            "branch_id": "g24d_causal_anchored_vwap_zones",
            "status": "active_pending_freeze",
            "route_family": "anchored_volume_weighted_levels",
            "parent_evidence": "new_independent_level_family",
            "plain_question": (
                "Do completed-session daily, weekly, and monthly anchored VWAP centres "
                "and fixed volume-weighted dispersion bands locate repeatable unsigned "
                "reaction zones?"
            ),
            "scope_guard": (
                "All anchors and bands must use completed information only. Freeze widths "
                "and anchors before outcomes and compare with shifted, stale, near-miss, "
                "random-time, and random-price controls."
            ),
        },
        {
            "branch_id": "g24e_generic_multitimeframe_indicator_context",
            "status": "active_pending_freeze",
            "route_family": "generic_technical_context_at_levels",
            "parent_evidence": "approved_optional_generic_indicator_checks",
            "plain_question": (
                "Do broad causal RSI, Bollinger width/position, MACD magnitude, EMA slope, "
                "and ATR states across several timeframes change unsigned reactions at a "
                "fixed level basket?"
            ),
            "scope_guard": (
                "Test one indicator state at a time in training-derived broad bands, then "
                "a small predeclared cross-timeframe agreement set. Require same-state "
                "no-level times plus artificial level controls; do not mine arbitrary "
                "indicator conjunctions."
            ),
        },
        {
            "branch_id": "g24f_freqai_untouched_later_confirmation",
            "status": "parked_awaiting_new_data",
            "route_family": "untouched_future_confirmation",
            "park_reason": (
                "The Generation 23 activity lead reused the existing later blocks. Preserve "
                "its exact definition until enough genuinely later data exists for two "
                "untouched blocks."
            ),
        },
        {
            "branch_id": "g24g_one_minute_direction_replay",
            "status": "parked_no_price_reaction_parent",
            "route_family": "bounded_direction_microscope",
            "park_reason": (
                "The retained result predicts volume activity, not a proven price reaction "
                "zone. It does not yet justify spending the one-minute direction lane."
            ),
        },
        {
            "branch_id": "g24h_external_news_orderbook_web_context",
            "status": "parked_coverage_audit_continues",
            "route_family": "external_context",
            "park_reason": (
                "Keep collection and timestamp-coverage audits running. Promote a source "
                "only when it overlaps enough frozen events and clean windows."
            ),
        },
        {
            "branch_id": "g24i_multitimeframe_cluster_tuning",
            "status": "parked_rejected_attribution",
            "route_family": "multitimeframe_cluster_attribution",
            "park_reason": (
                "Generation 23 did not show incremental cluster evidence over all component "
                "and artificial controls, so radius or width tuning would chase noise."
            ),
        },
        {
            "branch_id": "g24j_standalone_round_or_distribution_tuning",
            "status": "parked_weak_or_incomplete_parent",
            "route_family": "standalone_level_refinement",
            "park_reason": (
                "Standalone parents did not complete their full cross-coin ladders. Test the "
                "rational cross-family convergence question instead of tuning winners."
            ),
        },
    ]


def load_result(path: Path, expected_status: str) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != expected_status:
        raise ValueError(f"Generation 23 result is not terminal: {path}")
    return result


def validate_results() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected = {
        "multitimeframe_incremental_attribution": (
            "completed_generation23_multitimeframe_attribution"
        ),
        "continuous_cross_asset_context": "completed_generation23_continuous_context",
        "round_number_grids": "completed_generation23_round_numbers",
        "rolling_price_distributions": "completed_generation23_price_distribution",
    }
    for name, item in DIRECT_ROUTES.items():
        load_result(Path(item["result"]), expected[name])
    normal = load_result(FREQAI_NORMAL_RESULT, "completed_generation23_freqai_cohort")
    meme = load_result(FREQAI_MEME_RESULT, "completed_generation23_freqai_cohort")
    joint = load_result(FREQAI_JOINT_RESULT, "completed_generation23_freqai_joint_review")
    if normal.get("cohort") != "normal" or meme.get("cohort") != "meme":
        raise ValueError("Generation 23 FreqAI cohort ordering changed.")
    return normal, meme, joint


def direct_summary(result: dict[str, Any], expected_controls: int) -> dict[str, Any]:
    decisions = pd.read_csv(result["artifacts"]["decisions"]["path"])
    scores = pd.read_csv(result["artifacts"]["scores"]["path"])
    complete = as_bool(decisions["complete_control_period_ladder"])
    keys = ["scope_kind", "scope_value", "metric", "horizon_hours", "market_scope"]
    counts = (
        scores.groupby(keys, observed=True)
        .agg(
            observed_period_control_cells=("point_period_pass", "size"),
            point_period_control_passes=("point_period_pass", "sum"),
            strict_period_control_passes=("strict_period_pass", "sum"),
            minimum_equal_coin_difference=("equal_coin_difference", "min"),
            minimum_positive_coin_fraction=("positive_coin_fraction", "min"),
            minimum_bootstrap_lower=("bootstrap_lower_95", "min"),
        )
        .reset_index()
    )
    ranked = counts.sort_values(
        [
            "point_period_control_passes",
            "strict_period_control_passes",
            "minimum_equal_coin_difference",
        ],
        ascending=False,
    )
    top = ranked.iloc[0].to_dict() if len(ranked) else None
    if top is not None:
        top = {
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in top.items()
        }
    return {
        "decision_rows": len(decisions),
        "complete_ladder_rows": int(complete.sum()),
        "strict_rows": int(decisions["status"].eq("strict_holdout_confirmation").sum()),
        "point_rows": int(decisions["status"].eq("point_holdout_confirmation").sum()),
        "expected_period_control_cells_per_complete_question": expected_controls * 2,
        "best_observed_near_miss": top,
    }


def freqai_summary(normal: dict[str, Any], meme: dict[str, Any]) -> dict[str, Any]:
    decisions = pd.concat(
        [
            pd.read_csv(normal["artifacts"]["decisions"]["path"]),
            pd.read_csv(meme["artifacts"]["decisions"]["path"]),
        ],
        ignore_index=True,
    )
    retained = decisions.loc[as_bool(decisions["all_controls_both_periods_pass"])]
    if len(retained) != 1:
        raise ValueError(f"Expected one Generation 23 FreqAI lead, got {len(retained)}.")
    lead = retained.iloc[0]
    result = normal if lead["cohort"] == "normal" else meme
    pair_scores = pd.read_csv(result["artifacts"]["pair_scores"]["path"])
    pair_cell = pair_scores.loc[
        pair_scores["target"].eq(lead["target"])
        & pair_scores["role"].eq("full_interaction")
    ].copy()
    consistency = []
    for period, cell in pair_cell.groupby("period", observed=True, sort=False):
        consistency.append(
            {
                "period": str(period),
                "coins": int(cell["pair"].nunique()),
                "positive_spearman_coins": int(
                    pd.to_numeric(cell["spearman_rank_correlation"], errors="coerce")
                    .gt(0)
                    .sum()
                ),
                "positive_top_minus_bottom_coins": int(
                    pd.to_numeric(cell["top_minus_bottom_difference"], errors="coerce")
                    .gt(0)
                    .sum()
                ),
                "top_group_above_training_median_at_least_55pct_coins": int(
                    pd.to_numeric(
                        cell["top_quartile_above_training_median_fraction"],
                        errors="coerce",
                    )
                    .ge(0.55)
                    .sum()
                ),
            }
        )
    comparisons = pd.read_csv(result["artifacts"]["control_comparisons"]["path"])
    lead_comparisons = comparisons.loc[
        comparisons["target"].eq(lead["target"])
        & comparisons["market_scope"].eq(lead["market_scope"])
    ].copy()
    model_rows = lead_comparisons.loc[
        lead_comparisons["control"].ne("constant_training_median")
    ]
    near_miss_counts = (
        comparisons.groupby(
            ["cohort", "market_scope", "target", "target_transform", "raw_target"],
            observed=True,
        )["period_control_pass"]
        .apply(lambda values: int(as_bool(values).sum()))
    )
    return {
        "decision_rows": len(decisions),
        "retained_rows": 1,
        "nine_of_ten_control_period_near_misses": int(near_miss_counts.eq(9).sum()),
        "lead": {
            "cohort": str(lead["cohort"]),
            "market_scope": str(lead["market_scope"]),
            "target": str(lead["target"]),
            "target_transform": str(lead["target_transform"]),
            "raw_target": str(lead["raw_target"]),
            "minimum_top_group_above_training_median_fraction": float(
                lead["minimum_candidate_top_above_training_median_fraction"]
            ),
            "minimum_top_minus_bottom": float(lead["minimum_candidate_top_minus_bottom"]),
            "pair_consistency": consistency,
            "smallest_model_mae_advantage": float(
                (
                    pd.to_numeric(model_rows["control_mae"], errors="coerce")
                    - pd.to_numeric(model_rows["candidate_mae"], errors="coerce")
                ).min()
            ),
            "smallest_model_spearman_advantage": float(
                (
                    pd.to_numeric(model_rows["candidate_spearman"], errors="coerce")
                    - pd.to_numeric(model_rows["control_spearman"], errors="coerce")
                ).min()
            ),
            "smallest_model_separation_advantage": float(
                (
                    pd.to_numeric(
                        model_rows["candidate_top_minus_bottom"], errors="coerce"
                    )
                    - pd.to_numeric(
                        model_rows["control_top_minus_bottom"], errors="coerce"
                    )
                ).min()
            ),
        },
    }


def review(run_id: str, *, overwrite: bool = False) -> int:
    normal, meme, _ = validate_results()
    run_dir = REVIEW_ROOT / run_id
    result_path = run_dir / "g23_joint_review.json"
    queue_path = run_dir / "g24_sibling_branch_queue.json"
    if result_path.is_file() and not overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    route_summaries: dict[str, Any] = {}
    for name, item in DIRECT_ROUTES.items():
        result = json.loads(Path(item["result"]).read_text(encoding="utf-8"))
        route_summaries[name] = direct_summary(result, len(item["controls"]))
    route_summaries["freqai_calibrated_unsigned_reaction"] = freqai_summary(normal, meme)
    queue = generation24_queue()
    active = [item for item in queue if item["status"].startswith("active_")]
    parked = [item for item in queue if item["status"].startswith("parked_")]
    run_dir.mkdir(parents=True, exist_ok=True)
    source_contracts = {
        "generation23_freeze": artifact(g23z.FREEZE_PATH),
        **{
            f"{name}_result": artifact(Path(item["result"]))
            for name, item in DIRECT_ROUTES.items()
        },
        "freqai_normal_result": artifact(FREQAI_NORMAL_RESULT),
        "freqai_meme_result": artifact(FREQAI_MEME_RESULT),
        "freqai_joint_result": artifact(FREQAI_JOINT_RESULT),
    }
    queue_record = {
        "schema_version": 1,
        "generation": 24,
        "created_at_utc": g0.utc_now(),
        "status": "queued_after_complete_generation23_joint_review",
        "active_siblings": len(active),
        "parked_siblings": len(parked),
        "batch_rule": (
            "Freeze and implement all five active siblings before opening any Generation "
            "24 outcome; jointly review every sibling before descendants."
        ),
        "siblings": queue,
        "source_contracts": source_contracts,
    }
    g0.atomic_write_json(queue_record, queue_path)
    result = {
        "schema_version": 1,
        "generation": 23,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation23_joint_review",
        "all_five_active_siblings_terminal_before_review": True,
        "headline": (
            "Pair-calibrated FreqAI produced one consistent exploratory forecast of "
            "two-hour volume activity across all ten normal coins. No direct level, "
            "cluster, or context route survived its complete ladder."
        ),
        "route_results": route_summaries,
        "plain_interpretation": {
            "freqai": (
                "The model can rank which hours are more likely to be followed by unusual "
                "two-hour volume. This was positive for every normal coin in both later "
                "periods, but its late advantage over market-state-only inputs was small. "
                "It predicts activity, not direction or a tradable reaction."
            ),
            "multitimeframe_attribution": (
                "The three-timeframe cluster did not add dependable reaction information "
                "beyond its components and fake clusters. Do not tune it."
            ),
            "continuous_context": (
                "Several BTC-only context rows beat all five artificial level controls, "
                "but the required same-context no-level comparison lacked enough rows. "
                "That is an incomplete attribution question, not a retained result."
            ),
            "round_numbers": (
                "The main round grid often preceded more volume than random, shifted, and "
                "near-miss grids, but it failed the stale-grid comparison across enough "
                "coins. The evidence may describe broad price location rather than a "
                "special current round-number effect."
            ),
            "price_distribution": (
                "Some short-window BTC quantiles showed repeated crossing differences, "
                "but their late stale-control cells were incomplete. They are suitable "
                "only for a predeclared convergence test, not standalone tuning."
            ),
        },
        "breadth_assessment": {
            "retained_generation23_routes": 1,
            "retained_direct_reaction_zone_routes": 0,
            "retained_result_is_unsigned_activity_forecast": True,
            "broad_cross_coin_price_reaction_found": False,
            "joint_reaction_and_direction_at_or_above_55pct": False,
            "one_minute_direction_lane_justified": False,
            "same_holdout_trading_promotion_justified": False,
        },
        "generation24_active_siblings": len(active),
        "generation24_parked_siblings": len(parked),
        "research_boundary": {
            "no_trading_promotion": True,
            "no_profit_target": True,
            "no_signed_direction": True,
            "incomplete_control_ladders_not_treated_as_leads": True,
            "activity_forecast_not_relabelled_as_price_reaction": True,
            "result_spawned_branches_queued_only_after_joint_review": True,
        },
        "artifacts": {"generation24_queue": artifact(queue_path)},
        "result_path": str(result_path.resolve()),
    }
    g0.atomic_write_json(result, result_path)
    print(json.dumps(result, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_REVIEW_ID)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    return review(args.run_id, overwrite=args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())

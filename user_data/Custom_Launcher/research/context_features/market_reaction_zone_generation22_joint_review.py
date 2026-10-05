"""Jointly review all frozen Generation 22 siblings and queue Generation 23."""

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
from pandas.api.types import is_bool_dtype


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation22 as g22f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_cross_asset_context as g22d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_freeze as g22z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_multitimeframe_convergence as g22b,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_period_landmarks as g22a,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_trend_channels as g22c,
)


DEFAULT_REVIEW_ID = "g22_joint_review_20260827a"
REVIEW_ROOT = g22z.OUTPUT_ROOT / "joint_review"

PERIOD_RESULT = (
    g22a.RECORD_ROOT / g22a.DEFAULT_RUN_ID / "g22_period_landmarks_result.json"
)
PERIOD_DECISIONS = (
    g22a.RECORD_ROOT / g22a.DEFAULT_RUN_ID / "g22_period_decisions.csv"
)
MTF_RESULT = (
    g22b.RECORD_ROOT / g22b.DEFAULT_RUN_ID / "g22_multitimeframe_result.json"
)
MTF_DECISIONS = (
    g22b.RECORD_ROOT / g22b.DEFAULT_RUN_ID / "g22_multitimeframe_decisions.csv"
)
MTF_PERIOD_SCORES = (
    g22b.RECORD_ROOT / g22b.DEFAULT_RUN_ID / "g22_multitimeframe_period_scores.csv"
)
TREND_RESULT = (
    g22c.RECORD_ROOT / g22c.DEFAULT_RUN_ID / "g22_trend_channels_result.json"
)
TREND_DECISIONS = (
    g22c.RECORD_ROOT / g22c.DEFAULT_RUN_ID / "g22_trend_decisions.csv"
)
CONTEXT_RESULT = (
    g22d.RECORD_ROOT / g22d.DEFAULT_RUN_ID / "g22_cross_asset_context_result.json"
)
CONTEXT_DECISIONS = (
    g22d.RECORD_ROOT / g22d.DEFAULT_RUN_ID / "g22_context_decisions.csv"
)
CONTEXT_PAIR_CONTRASTS = (
    g22d.RECORD_ROOT / g22d.DEFAULT_RUN_ID / "g22_context_pair_contrasts.csv"
)
FREQAI_NORMAL_DIR = g22f.RECORD_ROOT / f"{g22f.DEFAULT_RUN_STEM}_normal"
FREQAI_MEME_DIR = g22f.RECORD_ROOT / f"{g22f.DEFAULT_RUN_STEM}_meme"
FREQAI_JOINT_DIR = g22f.RECORD_ROOT / g22f.DEFAULT_JOINT_REVIEW_ID
FREQAI_NORMAL_RESULT = FREQAI_NORMAL_DIR / "g22_freqai_result.json"
FREQAI_MEME_RESULT = FREQAI_MEME_DIR / "g22_freqai_result.json"
FREQAI_JOINT_RESULT = FREQAI_JOINT_DIR / "g22_freqai_joint_review.json"
FREQAI_NORMAL_DECISIONS = FREQAI_NORMAL_DIR / "g22_freqai_decisions.csv"
FREQAI_MEME_DECISIONS = FREQAI_MEME_DIR / "g22_freqai_decisions.csv"
FREQAI_NORMAL_CONTROLS = FREQAI_NORMAL_DIR / "g22_control_comparisons.csv"
FREQAI_MEME_CONTROLS = FREQAI_MEME_DIR / "g22_control_comparisons.csv"


def artifact(path: Path) -> dict[str, Any]:
    return g22z.artifact(path)


def as_bool(series: Series) -> Series:
    """Read bool columns consistently from in-memory or CSV frames."""
    if is_bool_dtype(series.dtype):
        return series.fillna(False)
    return series.astype("string").str.strip().str.lower().eq("true")


def generation23_queue() -> list[dict[str, Any]]:
    """Return the balanced next batch spawned only after all G22 siblings finished."""
    return [
        {
            "branch_id": "g23a_multitimeframe_cluster_incremental_attribution",
            "status": "active_pending_freeze",
            "route_family": "multitimeframe_cluster_attribution",
            "parent_evidence": "g22b_three_plus_timeframe_cluster_near_miss",
            "plain_question": (
                "When three or more independently calculated timeframes place levels "
                "near the same price, does their combination add reaction evidence "
                "beyond each component level and beyond equally dense fake clusters?"
            ),
            "minimum_controls": 6,
            "scope_guard": (
                "Keep the Generation 22 levels, widths, cluster radius, outcomes, and "
                "holdouts fixed. Compare against contacted components, density-matched "
                "pseudo-clusters, random time, recent analogue, stale, and price-shift "
                "controls; this is same-holdout attribution, not new confirmation."
            ),
        },
        {
            "branch_id": "g23b_continuous_cross_asset_context_at_levels",
            "status": "active_pending_freeze",
            "route_family": "continuous_cross_asset_context",
            "parent_evidence": "g22d_sparse_categorical_context_representation_failure",
            "plain_question": (
                "Do broad, causal BTC activity, equal-weight market activity, and "
                "cross-coin dispersion values alter unsigned reactions at levels when "
                "they are tested continuously or in broad single-variable bands?"
            ),
            "minimum_controls": 5,
            "scope_guard": (
                "Do not combine rare categorical states. Freeze continuous variables "
                "and training-derived broad bands, then compare actual contacts with "
                "same-context no-level times and artificial level controls."
            ),
        },
        {
            "branch_id": "g23c_freqai_calibrated_unsigned_reaction",
            "status": "active_pending_freeze",
            "route_family": "freqai_unsigned_reaction_calibration",
            "parent_evidence": "g22e_ranking_without_incremental_control_survival",
            "plain_question": (
                "Can the same small causal FreqAI feature set predict a pair-normalized "
                "residual or training-window reaction rank more reliably than it "
                "predicted raw unsigned reaction values?"
            ),
            "minimum_controls": 5,
            "scope_guard": (
                "Freeze at most two rational target transforms, reuse the same 12 "
                "features and four Generation 22 model controls, add the uncalibrated "
                "parent as a control, and use no profit or signed direction."
            ),
        },
        {
            "branch_id": "g23d_round_number_and_price_grid_zones",
            "status": "active_pending_freeze",
            "route_family": "price_scale_round_number_levels",
            "parent_evidence": "new_independent_level_family",
            "plain_question": (
                "Do obvious price-scale-aware round numbers and their fixed subdivisions "
                "locate repeatable price or volume reaction zones across normal and meme "
                "coins?"
            ),
            "minimum_controls": 5,
            "scope_guard": (
                "Choose grids from the causal price scale only, not observed outcomes; "
                "compare with density-matched random mantissas, shifted grids, near "
                "misses, stale levels, and matched random times."
            ),
        },
        {
            "branch_id": "g23e_causal_price_distribution_boundaries",
            "status": "active_pending_freeze",
            "route_family": "rolling_price_distribution_levels",
            "parent_evidence": "new_independent_level_family",
            "plain_question": (
                "Do medians and fixed quantile boundaries from completed-candle price "
                "distributions over several horizons locate later reaction zones?"
            ),
            "minimum_controls": 5,
            "scope_guard": (
                "Use fixed 24, 72, 168, and 720 hour windows and predeclared quantiles "
                "from completed candles only; no quantile or window is selected from "
                "the evaluation outcomes."
            ),
        },
        {
            "branch_id": "g23f_multitimeframe_new_period_confirmation",
            "status": "parked_awaiting_new_data",
            "route_family": "untouched_future_confirmation",
            "plain_question": (
                "Should the Generation 22 three-timeframe cluster be called confirmed?"
            ),
            "park_reason": (
                "No. Its same-date control ladder failed. Preserve its exact definition "
                "for a genuinely later untouched confirmation after enough new hourly "
                "data exists for two evaluation blocks."
            ),
        },
        {
            "branch_id": "g23g_period_landmark_refinement",
            "status": "parked_rejected_parent",
            "route_family": "historical_period_landmarks",
            "plain_question": "Should completed day/week/month coordinates be tuned now?",
            "park_reason": (
                "Only weak complete-ladder differences appeared and their uncertainty "
                "crossed zero; tuning the same dates would chase noise."
            ),
        },
        {
            "branch_id": "g23h_trend_channel_refinement",
            "status": "parked_rejected_parent",
            "route_family": "trend_channel_levels",
            "plain_question": "Should trend-channel windows or widths be tuned now?",
            "park_reason": (
                "The complete ladders were effectively flat and did not justify a "
                "same-holdout parameter search."
            ),
        },
        {
            "branch_id": "g23i_one_minute_direction_replay",
            "status": "parked_no_retained_reaction_parent",
            "route_family": "bounded_direction_microscope",
            "plain_question": "Should one-minute direction work resume in this batch?",
            "park_reason": (
                "Not yet. Generation 22 retained no direction-neutral reaction pattern "
                "that justifies spending the bounded direction lane."
            ),
        },
        {
            "branch_id": "g23j_external_news_orderbook_web_context",
            "status": "parked_coverage",
            "route_family": "external_context",
            "plain_question": (
                "Should sparse news, web, orderbook, and global data enter this batch?"
            ),
            "park_reason": (
                "Keep collecting and re-audit timestamp coverage later; the current "
                "batch has five causal market-data routes with complete broad holdouts."
            ),
        },
    ]


def validate_results() -> tuple[dict[str, Any], ...]:
    paths = (
        PERIOD_RESULT,
        MTF_RESULT,
        TREND_RESULT,
        CONTEXT_RESULT,
        FREQAI_NORMAL_RESULT,
        FREQAI_MEME_RESULT,
        FREQAI_JOINT_RESULT,
    )
    results = tuple(json.loads(path.read_text(encoding="utf-8")) for path in paths)
    expected = (
        "completed_generation22_period_landmarks",
        "completed_generation22_multitimeframe_convergence",
        "completed_generation22_trend_channels",
        "completed_generation22_cross_asset_context",
        "completed_generation22_freqai_cohort",
        "completed_generation22_freqai_cohort",
        "completed_generation22_freqai_joint_review",
    )
    for result, status in zip(results, expected, strict=True):
        if result.get("status") != status:
            raise ValueError(f"Generation 22 sibling is not terminal: {result.get('status')}")
    if results[4].get("cohort") != "normal" or results[5].get("cohort") != "meme":
        raise ValueError("Generation 22 FreqAI cohort results are not normal then meme.")
    return results


def ladder_summary(frame: DataFrame) -> dict[str, Any]:
    complete = as_bool(frame["complete_control_period_ladder"])
    differences = pd.to_numeric(frame["minimum_equal_coin_difference"], errors="coerce")
    lower = pd.to_numeric(frame["minimum_bootstrap_lower"], errors="coerce")
    eligible = frame.loc[complete].copy()
    eligible["minimum_equal_coin_difference"] = differences.loc[complete]
    eligible["minimum_bootstrap_lower"] = lower.loc[complete]
    positive = eligible.loc[eligible["minimum_equal_coin_difference"].gt(0)]
    ranked_source = positive if len(positive) else eligible
    best = ranked_source.sort_values(
        ["minimum_bootstrap_lower", "minimum_equal_coin_difference"], ascending=False
    ).head(1)
    best_row: dict[str, Any] | None = None
    if len(best):
        row = best.iloc[0]
        best_row = {
            "scope_kind": str(row["scope_kind"]),
            "scope_value": str(row["scope_value"]),
            "metric": str(row["metric"]),
            "horizon_hours": int(row["horizon_hours"]),
            "market_scope": str(row["market_scope"]),
            "minimum_equal_coin_difference": float(
                row["minimum_equal_coin_difference"]
            ),
            "minimum_bootstrap_lower": float(row["minimum_bootstrap_lower"]),
        }
    return {
        "decision_rows": len(frame),
        "complete_ladder_rows": int(complete.sum()),
        "complete_positive_minimum_rows": int((complete & differences.gt(0)).sum()),
        "incomplete_ladder_rows": int((~complete).sum()),
        "incomplete_positive_minimum_rows": int(((~complete) & differences.gt(0)).sum()),
        "strict_rows": int(frame["status"].eq("strict_holdout_confirmation").sum()),
        "point_rows": int(frame["status"].eq("point_holdout_confirmation").sum()),
        "best_complete_ladder_row": best_row,
    }


def mtf_near_miss_summary(period_scores: DataFrame) -> dict[str, Any]:
    cell = period_scores.loc[
        period_scores["scope_kind"].eq("timeframe_geometry")
        & period_scores["scope_value"].eq("three_plus_timeframe_cluster")
        & period_scores["metric"].eq("crossing_count")
        & pd.to_numeric(period_scores["horizon_hours"], errors="coerce").eq(8)
        & period_scores["market_scope"].eq("all_normal")
    ].copy()
    if len(cell) != 10:
        raise ValueError(
            "Expected ten period/control rows for the fixed G22 MTF near-miss cell."
        )
    cell["strict_period_pass"] = as_bool(cell["strict_period_pass"])
    rows: list[dict[str, Any]] = []
    for _, row in cell.sort_values(["comparison", "period"]).iterrows():
        rows.append(
            {
                "period": str(row["period"]),
                "comparison": str(row["comparison"]),
                "eligible_coins": int(row["eligible_coins"]),
                "positive_coins": int(row["positive_coins"]),
                "equal_coin_difference": float(row["equal_coin_difference"]),
                "bootstrap_lower_95": float(row["bootstrap_lower_95"]),
                "strict_period_pass": bool(row["strict_period_pass"]),
            }
        )
    failed = sorted(
        {
            str(row["comparison"])
            for _, row in cell.loc[~cell["strict_period_pass"]].iterrows()
        }
    )
    return {
        "cell": {
            "scope": "three_plus_timeframe_cluster",
            "metric": "crossing_count",
            "horizon_hours": 8,
            "market_scope": "all_normal",
        },
        "period_control_rows": rows,
        "failed_strict_controls": failed,
        "complete_control_ladder_passed": False,
        "interpretation": (
            "Large differences versus several controls make this useful for bounded "
            "component attribution, but the random-recent-analogue and stale ladders "
            "did not both survive, so it is not a confirmed reaction level."
        ),
    }


def context_coverage_summary(pair_contrasts: DataFrame) -> dict[str, Any]:
    actual = pair_contrasts.loc[
        pair_contrasts["comparison"].eq("matched_random_time")
    ].copy()
    actual["event_rows_actual"] = pd.to_numeric(
        actual["event_rows_actual"], errors="coerce"
    )
    state = actual["scope_value"].astype(str).str.replace(
        r"^w(?:4|24)__", "", regex=True
    )
    actual = actual.assign(context_state=state)
    grouped = actual.groupby("context_state", sort=True)["event_rows_actual"]
    return {
        "actual_contact_rows_by_state": {
            str(name): {
                "minimum": int(values.min()),
                "median": float(values.median()),
                "maximum": int(values.max()),
            }
            for name, values in grouped
        },
        "complete_decision_ladders": 0,
        "interpretation": (
            "The rare quiet and synchronized conjunctions often left too few contacts "
            "for a complete cross-coin ladder. This is a representation/coverage failure, "
            "not evidence that continuous market context is useless."
        ),
    }


def control_count_summary(frame: DataFrame) -> dict[str, dict[str, int]]:
    summaries: dict[str, dict[str, int]] = {}
    for control, group in frame.groupby("control", sort=True):
        summaries[str(control)] = {
            "comparison_rows": len(group),
            "candidate_gate_pass_rows": int(as_bool(group["candidate_gate_pass"]).sum()),
            "control_metrics_pass_rows": int(
                as_bool(group["control_metrics_pass"]).sum()
            ),
            "period_control_pass_rows": int(
                as_bool(group["period_control_pass"]).sum()
            ),
        }
    return summaries


def freqai_summary(
    decisions: DataFrame, control_comparisons: DataFrame
) -> dict[str, Any]:
    complete = as_bool(decisions["complete_control_and_period_ladder"])
    passed = as_bool(decisions["all_controls_both_periods_pass"])
    top_fraction = pd.to_numeric(
        decisions["minimum_candidate_top_above_training_median_fraction"],
        errors="coerce",
    )
    separation = pd.to_numeric(
        decisions["minimum_candidate_top_minus_bottom"], errors="coerce"
    )
    ranked = decisions.assign(
        _top_fraction=top_fraction, _separation=separation
    ).sort_values(["_top_fraction", "_separation"], ascending=False)
    row = ranked.iloc[0]
    return {
        "decision_rows": len(decisions),
        "complete_control_and_period_ladders": int(complete.sum()),
        "all_controls_both_periods_pass_rows": int(passed.sum()),
        "best_descriptive_ranking_row": {
            "cohort": str(row["cohort"]),
            "market_scope": str(row["market_scope"]),
            "target": str(row["target"]),
            "minimum_top_quartile_above_training_median_fraction": float(
                row["_top_fraction"]
            ),
            "minimum_top_minus_bottom": float(row["_separation"]),
            "retained": False,
        },
        "control_comparison_counts": control_count_summary(control_comparisons),
    }


def review(run_id: str, *, overwrite: bool = False) -> int:
    validate_results()
    run_dir = REVIEW_ROOT / run_id
    result_path = run_dir / "g22_joint_review.json"
    queue_path = run_dir / "g23_sibling_branch_queue.json"
    if result_path.is_file() and not overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return 0

    period = pd.read_csv(PERIOD_DECISIONS)
    mtf = pd.read_csv(MTF_DECISIONS)
    mtf_scores = pd.read_csv(MTF_PERIOD_SCORES)
    trend = pd.read_csv(TREND_DECISIONS)
    context = pd.read_csv(CONTEXT_DECISIONS)
    context_pairs = pd.read_csv(CONTEXT_PAIR_CONTRASTS)
    freqai_normal = pd.read_csv(FREQAI_NORMAL_DECISIONS)
    freqai_meme = pd.read_csv(FREQAI_MEME_DECISIONS)
    freqai_normal_controls = pd.read_csv(FREQAI_NORMAL_CONTROLS)
    freqai_meme_controls = pd.read_csv(FREQAI_MEME_CONTROLS)

    if any(
        frame["status"].ne("not_retained").any()
        for frame in (period, mtf, trend, context)
    ):
        raise ValueError("Generation 22 direct routes unexpectedly retained a decision.")
    if any(
        frame["status"].ne("not_retained_across_both_later_blocks").any()
        for frame in (freqai_normal, freqai_meme)
    ):
        raise ValueError("Generation 22 FreqAI unexpectedly retained a decision.")

    queue = generation23_queue()
    active = [item for item in queue if item["status"].startswith("active_")]
    parked = [item for item in queue if item["status"].startswith("parked_")]
    run_dir.mkdir(parents=True, exist_ok=True)
    queue_record = {
        "schema_version": 1,
        "generation": 23,
        "created_at_utc": g0.utc_now(),
        "status": "queued_after_complete_generation22_joint_review",
        "active_siblings": len(active),
        "parked_siblings": len(parked),
        "batch_rule": (
            "Freeze and implement all five active siblings before opening any "
            "Generation 23 outcome; review every terminal sibling together before "
            "allowing any Generation 23 result to spawn a descendant."
        ),
        "same_holdout_boundary": (
            "Generation 23 repairs and attribution tests using the Generation 22 "
            "holdouts are exploratory. They cannot turn a Generation 22 near-miss "
            "into untouched confirmation."
        ),
        "siblings": queue,
        "source_contracts": {
            "generation22_freeze": artifact(g22z.FREEZE_PATH),
            "period_result": artifact(PERIOD_RESULT),
            "multitimeframe_result": artifact(MTF_RESULT),
            "trend_result": artifact(TREND_RESULT),
            "context_result": artifact(CONTEXT_RESULT),
            "freqai_normal_result": artifact(FREQAI_NORMAL_RESULT),
            "freqai_meme_result": artifact(FREQAI_MEME_RESULT),
            "freqai_joint_result": artifact(FREQAI_JOINT_RESULT),
        },
    }
    g0.atomic_write_json(queue_record, queue_path)

    result = {
        "schema_version": 1,
        "generation": 22,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation22_joint_review",
        "all_five_active_siblings_terminal_before_review": True,
        "headline": (
            "No Generation 22 route survived its complete control and later-date "
            "requirements. Multitimeframe clusters produced the strongest bounded "
            "near-miss; categorical context was too sparse; FreqAI ranked some future "
            "volume/range behaviour but added no dependable information over all controls."
        ),
        "route_results": {
            "completed_period_landmarks": ladder_summary(period),
            "multitimeframe_convergence": ladder_summary(mtf),
            "trend_channel_boundaries": ladder_summary(trend),
            "categorical_cross_asset_context": ladder_summary(context),
            "freqai_normal": freqai_summary(
                freqai_normal, freqai_normal_controls
            ),
            "freqai_meme": freqai_summary(freqai_meme, freqai_meme_controls),
        },
        "bounded_near_miss": mtf_near_miss_summary(mtf_scores),
        "context_representation_diagnosis": context_coverage_summary(context_pairs),
        "plain_interpretation": {
            "period_landmarks": (
                "Previous-day, week, and month coordinates did not beat the full "
                "artificial-location ladder consistently. Their best complete rows were "
                "small and their uncertainty still crossed zero."
            ),
            "multitimeframe_clusters": (
                "Prices near three or more independently calculated timeframe levels "
                "showed more later crossing activity than several controls in both date "
                "blocks. They still failed the complete stale/recent-analogue ladder, so "
                "the correct next question is what component created the difference—not "
                "whether to trade or tune the result."
            ),
            "trend_channels": (
                "Causal rolling channel centres and boundaries were effectively flat "
                "after complete controls and do not justify tuning their windows."
            ),
            "cross_asset_context": (
                "The narrow combined state labels created too few quiet and synchronized "
                "contacts for complete cross-coin comparisons. Broader continuous context "
                "tests are warranted as a representation repair."
            ),
            "freqai": (
                "The full model often put higher future volume or range in its highest "
                "prediction group, but raw-value errors were worse than simple baselines "
                "and it did not beat level-only, state-only, constant, and shifted-label "
                "controls across both later blocks. A small predeclared calibration repair "
                "is warranted; the parent result is not an ML edge."
            ),
        },
        "breadth_assessment": {
            "retained_generation22_routes": 0,
            "bounded_attribution_near_misses": 1,
            "representation_failures_worth_repairing": 1,
            "new_independent_level_families_queued": 2,
            "broad_cross_coin_reaction_found": False,
            "joint_reaction_and_direction_at_or_above_55pct": False,
            "one_minute_direction_lane_justified": False,
            "same_holdout_parameter_tuning_justified": False,
        },
        "generation23_active_siblings": len(active),
        "generation23_parked_siblings": len(parked),
        "research_boundary": {
            "no_trading_promotion": True,
            "no_profit_target": True,
            "no_signed_direction": True,
            "incomplete_control_ladders_not_treated_as_leads": True,
            "same_holdout_repairs_are_exploratory": True,
            "result_spawned_branches_queued_only_after_joint_review": True,
        },
        "artifacts": {"generation23_queue": artifact(queue_path)},
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

"""Jointly review all frozen Generation 21 siblings and queue Generation 22."""

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
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation21_freeze as g21z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation21_level_geometry as g21l,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation21_market_state as g21s,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation21_one_minute_replay as g21e,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation21_structural_levels as g21d,
)


DEFAULT_REVIEW_ID = "g21_joint_review_20260827a"
REVIEW_ROOT = g21z.OUTPUT_ROOT / "joint_review"
LEVEL_RESULT = g21l.RECORD_ROOT / g21l.DEFAULT_RUN_ID / "g21_level_geometry_result.json"
LEVEL_DECISIONS = g21l.RECORD_ROOT / g21l.DEFAULT_RUN_ID / "g21_level_decisions.csv"
STATE_RESULT = g21s.RECORD_ROOT / g21s.DEFAULT_RUN_ID / "g21_market_state_result.json"
STATE_DECISIONS = g21s.RECORD_ROOT / g21s.DEFAULT_RUN_ID / "g21_market_state_decisions.csv"
STRUCTURAL_RESULT = (
    g21d.RECORD_ROOT / g21d.DEFAULT_RUN_ID / "g21_structural_levels_result.json"
)
STRUCTURAL_DECISIONS = (
    g21d.RECORD_ROOT / g21d.DEFAULT_RUN_ID / "g21_structural_decisions.csv"
)
MINUTE_RESULT = g21e.RECORD_ROOT / g21e.DEFAULT_RUN_ID / "g21_one_minute_replay_result.json"
MINUTE_DECISIONS = g21e.RECORD_ROOT / g21e.DEFAULT_RUN_ID / "g21_one_minute_decisions.csv"
MINUTE_OUTCOMES = g21e.RECORD_ROOT / g21e.DEFAULT_RUN_ID / "g21_one_minute_episode_outcomes.csv"


def artifact(path: Path) -> dict[str, Any]:
    return g21z.artifact(path)


def generation22_queue() -> list[dict[str, Any]]:
    return [
        {
            "branch_id": "g22a_completed_period_price_landmarks",
            "status": "active_pending_freeze",
            "route_family": "historical_price_landmarks",
            "plain_question": (
                "Do prices fixed by the last completed day, week, and month—open, high, "
                "low, midpoint, and typical price—locate later price/volume reaction "
                "areas better than matched artificial prices?"
            ),
            "minimum_controls": 5,
            "scope_guard": (
                "Completed periods only; freeze every coordinate and width together, "
                "with no calendar-level chosen from outcomes."
            ),
        },
        {
            "branch_id": "g22b_multitimeframe_level_convergence_and_precedence",
            "status": "active_pending_freeze",
            "route_family": "multitimeframe_level_convergence",
            "plain_question": (
                "Do isolated levels and independently calculated 1h, 4h, daily, and "
                "weekly level clusters differ in reaction strength, and does a higher-"
                "timeframe level add evidence when it is close to a lower-timeframe one?"
            ),
            "minimum_controls": 5,
            "scope_guard": (
                "Singles remain co-equal with clusters; timeframe precedence is an "
                "empirical reaction question, not an assumed trading rule."
            ),
        },
        {
            "branch_id": "g22c_causal_trend_channel_boundaries",
            "status": "active_pending_freeze",
            "route_family": "dynamic_trendline_levels",
            "plain_question": (
                "Do causal rolling trend-channel centre and boundary prices over short, "
                "medium, and long windows locate repeatable reaction zones?"
            ),
            "minimum_controls": 5,
            "scope_guard": (
                "Use completed candles and fixed regression/residual definitions; compare "
                "against rolling mean, stale, shifted, near, and matched-time controls."
            ),
        },
        {
            "branch_id": "g22d_cross_asset_context_conditioned_level_reactions",
            "status": "active_pending_freeze",
            "route_family": "cross_asset_market_context",
            "plain_question": (
                "Are reactions at a fixed broad basket of levels more repeatable when BTC "
                "and equal-weight crypto breadth describe quiet, dispersed, or synchronized "
                "market activity?"
            ),
            "minimum_controls": 5,
            "scope_guard": (
                "Context must be causal and direction-blind; compare level contacts with "
                "same-state no-level times before claiming that the level adds information."
            ),
        },
        {
            "branch_id": "g22e_freqai_unsigned_reaction_interaction_regression",
            "status": "active_pending_freeze",
            "route_family": "low_dimensional_freqai_interactions",
            "plain_question": (
                "Can a small causal FreqAI regression using level distance/density, market "
                "state, and source role rank future unsigned price, volume, range, crossing, "
                "and dwell reactions better than level-only, state-only, constant, and "
                "time-shuffled controls?"
            ),
            "minimum_controls": 4,
            "scope_guard": (
                "Use continuous reaction outputs across several horizons, not profit or a "
                "single six-candle label; no signed direction and no feature fishing after "
                "holdouts open."
            ),
        },
        {
            "branch_id": "g22f_immediate_one_minute_direction_refinement",
            "status": "parked_failed_direction_parent",
            "route_family": "bounded_direction_microscope",
            "plain_question": "Should the six failed direction calls be tuned immediately?",
            "park_reason": (
                "No call reached 55% joint success or beat the leave-one-out majority "
                "control; return to reaction discovery before another direction batch."
            ),
        },
        {
            "branch_id": "g22g_volume_profile_role_or_geometry_tuning",
            "status": "parked_incomplete_or_rejected_parent",
            "route_family": "same_holdout_level_tuning",
            "plain_question": "Should Volume Profile roles or cluster definitions be tuned now?",
            "park_reason": (
                "No complete role or geometry ladder survived. Large positive rows with "
                "missing controls are insufficient evidence and must not drive same-date tuning."
            ),
        },
        {
            "branch_id": "g22h_external_news_orderbook_web_conditioning",
            "status": "parked_coverage",
            "route_family": "external_context",
            "plain_question": (
                "Do orderbook, news, web, or global-market observations condition reactions?"
            ),
            "park_reason": (
                "Keep collecting and re-audit later; no timestamp-ready external source yet "
                "spans two complete evaluation blocks for broad confirmation."
            ),
        },
    ]


def validate_results() -> tuple[dict[str, Any], ...]:
    results = tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (LEVEL_RESULT, STATE_RESULT, STRUCTURAL_RESULT, MINUTE_RESULT)
    )
    expected = (
        "completed_generation21_level_geometry",
        "completed_generation21_market_state",
        "completed_generation21_structural_levels",
        "completed_generation21_disjoint_one_minute_replay",
    )
    for result, status in zip(results, expected, strict=True):
        if result.get("status") != status:
            raise ValueError(f"Generation 21 sibling is not terminal: {result.get('status')}")
    return results


def ladder_summary(frame: DataFrame) -> dict[str, Any]:
    complete = frame["complete_control_period_ladder"].astype(bool)
    differences = pd.to_numeric(frame["minimum_equal_coin_difference"], errors="coerce")
    lower = pd.to_numeric(frame["minimum_bootstrap_lower"], errors="coerce")
    eligible = frame.loc[complete].copy()
    eligible["minimum_equal_coin_difference"] = differences.loc[complete]
    eligible["minimum_bootstrap_lower"] = lower.loc[complete]
    best = eligible.sort_values(
        ["minimum_bootstrap_lower", "minimum_equal_coin_difference"], ascending=False
    ).head(1)
    best_row = best.iloc[0].to_dict() if len(best) else None
    if best_row:
        for key in ("minimum_equal_coin_difference", "minimum_bootstrap_lower"):
            best_row[key] = float(best_row[key])
    return {
        "decision_rows": len(frame),
        "complete_ladder_rows": int(complete.sum()),
        "incomplete_ladder_rows": int((~complete).sum()),
        "incomplete_positive_minimum_rows": int(((~complete) & differences.gt(0)).sum()),
        "strict_rows": int(frame["status"].eq("strict_holdout_confirmation").sum()),
        "point_rows": int(frame["status"].eq("point_holdout_confirmation").sum()),
        "best_complete_ladder_row": best_row,
    }


def reaction_rate(
    outcomes: DataFrame, kind: str, threshold: float, horizon: int
) -> float:
    cell = outcomes.loc[
        outcomes["episode_kind"].eq(kind)
        & pd.to_numeric(outcomes["distance_threshold_atr"], errors="coerce").eq(threshold)
        & pd.to_numeric(outcomes["horizon_minutes"], errors="coerce").eq(horizon)
    ]
    return float(cell["reaction"].astype(bool).mean())


def review(run_id: str, *, overwrite: bool = False) -> int:
    level_result, state_result, structural_result, minute_result = validate_results()
    run_dir = REVIEW_ROOT / run_id
    result_path = run_dir / "g21_joint_review.json"
    queue_path = run_dir / "g22_sibling_branch_queue.json"
    if result_path.is_file() and not overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    level = pd.read_csv(LEVEL_DECISIONS)
    state = pd.read_csv(STATE_DECISIONS)
    structural = pd.read_csv(STRUCTURAL_DECISIONS)
    minute = pd.read_csv(MINUTE_DECISIONS)
    outcomes = pd.read_csv(MINUTE_OUTCOMES)
    if any(
        frame["status"].ne("not_retained").any()
        for frame in (level, state, structural, minute)
    ):
        raise ValueError("Generation 21 unexpectedly contains a retained decision.")
    queue = generation22_queue()
    active = [item for item in queue if item["status"].startswith("active_")]
    parked = [item for item in queue if item["status"].startswith("parked_")]
    run_dir.mkdir(parents=True, exist_ok=True)
    queue_record = {
        "schema_version": 1,
        "generation": 22,
        "created_at_utc": g0.utc_now(),
        "status": "queued_after_complete_generation21_joint_review",
        "active_siblings": len(active),
        "parked_siblings": len(parked),
        "batch_rule": (
            "Freeze and implement all five active siblings before opening any "
            "Generation 22 outcome; review them together before descendants."
        ),
        "siblings": queue,
        "source_contracts": {
            "generation21_freeze": artifact(g21z.FREEZE_PATH),
            "generation21_level_result": artifact(LEVEL_RESULT),
            "generation21_state_result": artifact(STATE_RESULT),
            "generation21_structural_result": artifact(STRUCTURAL_RESULT),
            "generation21_one_minute_result": artifact(MINUTE_RESULT),
        },
    }
    g0.atomic_write_json(queue_record, queue_path)
    level_ladder = ladder_summary(level)
    state_ladder = ladder_summary(state)
    structural_ladder = ladder_summary(structural)
    best_minute = minute.sort_values("actual_joint_success_rate", ascending=False).iloc[0]
    result = {
        "schema_version": 1,
        "generation": 21,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation21_joint_review",
        "all_five_active_siblings_terminal_before_review": True,
        "headline": (
            "No Generation 21 mechanism survived its complete control/date ladder. "
            "The earlier BTC Volume Profile lead did not resolve into a dependable role "
            "or cluster shape, and no one-minute direction call reached the 55% floor."
        ),
        "route_results": {
            "volume_profile_role_and_geometry": level_ladder,
            "level_free_multitimeframe_market_state": state_ladder,
            "abnormal_candle_structural_prices": structural_ladder,
            "disjoint_one_minute_direction": {
                "actual_episodes": int(minute_result["actual_episodes"]),
                "predeclared_method_distance_horizon_rows": len(minute),
                "retained_rows": int(minute["retained"].astype(bool).sum()),
                "best_method": str(best_minute["method"]),
                "best_distance_threshold_atr": float(
                    best_minute["distance_threshold_atr"]
                ),
                "best_horizon_minutes": int(best_minute["horizon_minutes"]),
                "best_joint_success_rate": float(
                    best_minute["actual_joint_success_rate"]
                ),
                "majority_control_at_best_cell": float(
                    best_minute["majority_path_joint_success_rate"]
                ),
            },
        },
        "plain_interpretation": {
            "volume_profile": (
                "BTC's earlier repeated-crossing result cannot yet be assigned to POC, "
                "value-area boundaries, high-volume nodes, low-volume nodes, isolated "
                "levels, or clusters. Positive rows missing a control or date block do "
                "not repair that gap."
            ),
            "market_state": (
                "Extreme volume/pressure, volatility/range, and trend/momentum bins on "
                "1h, 4h, and 8h did not reliably identify more future unsigned activity "
                "than all four state controls in both later blocks."
            ),
            "structural_prices": (
                "Open, close, midpoint, high, low, and typical price from a completed "
                "high-volume expansion candle did not consistently outperform all five "
                "artificial-location controls."
            ),
            "one_minute": (
                "At 0.5 ATR, 65% of the preselected reacting levels moved far enough "
                "within 15 minutes versus 12% of matched no-level episodes, but those "
                "levels were selected because an unsigned hourly reaction was already "
                "known. This describes timing inside known reactions; it does not prove "
                "the level predicted a reaction. Direction calls peaked at 52.9% and "
                "lost to the 64.7% leave-one-out majority control in that cell."
            ),
        },
        "selection_conditioned_one_minute_reaction_rates": {
            "actual_level_0p5atr_15m": reaction_rate(outcomes, "actual_level", 0.5, 15),
            "matched_no_level_0p5atr_15m": reaction_rate(
                outcomes, "matched_no_level", 0.5, 15
            ),
            "interpretation_limit": (
                "Descriptive only because actual episodes were selected using a known "
                "unsigned four-hour crossing response."
            ),
        },
        "breadth_assessment": {
            "strict_level_rows": int(level_result["strict_rows"]),
            "strict_state_rows": int(state_result["strict_rows"]),
            "strict_structural_rows": int(structural_result["strict_rows"]),
            "retained_direction_rows": int(minute_result["retained_methods"]),
            "broad_cross_coin_reaction_found": False,
            "joint_reaction_and_direction_at_or_above_55pct": False,
            "same_holdout_parameter_tuning_justified": False,
        },
        "generation22_active_siblings": len(active),
        "generation22_parked_siblings": len(parked),
        "research_boundary": {
            "no_trading_promotion": True,
            "no_profit_target": True,
            "incomplete_control_ladders_not_treated_as_leads": True,
            "result_spawned_branches_queued_only_after_joint_review": True,
        },
        "artifacts": {"generation22_queue": artifact(queue_path)},
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

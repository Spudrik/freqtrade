"""Jointly review all frozen Generation 20 siblings and queue Generation 21."""

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


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation20_change_point_avwap as g20v,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation20_freeze as g20z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation20_one_minute_replay as g20m,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation20_state_attribution as g20s,
)


DEFAULT_REVIEW_ID = "g20_joint_review_20260827a"
REVIEW_ROOT = g20z.OUTPUT_ROOT / "joint_review"
STATE_RESULT = (
    g20s.RECORD_ROOT / g20s.DEFAULT_RUN_ID / "g20_state_attribution_result.json"
)
STATE_DECISIONS = (
    g20s.RECORD_ROOT / g20s.DEFAULT_RUN_ID / "g20_state_decisions.csv"
)
AVWAP_RESULT = g20v.RECORD_ROOT / g20v.DEFAULT_RUN_ID / "g20_change_point_avwap_result.json"
AVWAP_DECISIONS = (
    g20v.RECORD_ROOT / g20v.DEFAULT_RUN_ID / "g20_change_point_decisions.csv"
)
ONE_MINUTE_RESULT = (
    g20m.RECORD_ROOT / g20m.DEFAULT_RUN_ID / "g20_one_minute_replay_result.json"
)
ONE_MINUTE_DECISIONS = (
    g20m.RECORD_ROOT / g20m.DEFAULT_RUN_ID / "g20_one_minute_decisions.csv"
)
ONE_MINUTE_OUTCOMES = (
    g20m.RECORD_ROOT / g20m.DEFAULT_RUN_ID / "g20_one_minute_episode_outcomes.csv"
)


def artifact(path: Path) -> dict[str, Any]:
    return g20z.artifact(path)


def generation21_queue() -> list[dict[str, Any]]:
    return [
        {
            "branch_id": "g21a_btc_volume_profile_role_specificity",
            "status": "active_pending_freeze",
            "route_family": "asset_specific_level_mechanism",
            "plain_question": (
                "Which causal Volume Profile roles, if any, explain BTC's repeated "
                "four-hour recross result, and do the same roles remain absent from "
                "predeclared groups of similar non-BTC coins?"
            ),
            "minimum_controls": 4,
            "scope_guard": (
                "Keep BTC explicitly separate; do not relabel it broad if established "
                "alt groups or memes fail."
            ),
        },
        {
            "branch_id": "g21b_multitimeframe_market_state_reaction_atlas",
            "status": "active_pending_freeze",
            "route_family": "level_free_market_state",
            "plain_question": (
                "Without calculated levels, do completed volume, range, volatility, "
                "pressure, and trend states identify unusually active periods across "
                "1h, 4h, and 8h clocks?"
            ),
            "minimum_controls": 4,
            "scope_guard": "Reaction first; no signed direction or profit target.",
        },
        {
            "branch_id": "g21c_state_matched_single_and_cluster_levels",
            "status": "active_pending_freeze",
            "route_family": "single_vs_cluster_geometry",
            "plain_question": (
                "After matching traffic already underway, do single current levels or "
                "clusters made from independent level families add repeatable reaction "
                "information across normal groups, memes, and multiple source timeframes?"
            ),
            "minimum_controls": 4,
            "scope_guard": "Singles and clusters are co-equal hypotheses.",
        },
        {
            "branch_id": "g21d_abnormal_candle_structural_levels",
            "status": "active_pending_freeze",
            "route_family": "new_rational_level_source",
            "plain_question": (
                "Do the body edges, wick extremes, or volume-weighted price of a causal "
                "high-volume expansion candle locate later reaction areas better than "
                "random, stale, near, and rolling-price controls?"
            ),
            "minimum_controls": 4,
            "scope_guard": (
                "Use one fixed rational event definition; do not tune it to known candles."
            ),
        },
        {
            "branch_id": "g21e_disjoint_one_minute_rejection_diagnostic",
            "status": "active_pending_freeze",
            "route_family": "bounded_direction_microscope",
            "plain_question": (
                "On direction-blind Donchian episodes disjoint from Generation 20, does "
                "immediate rejection versus breakthrough remain visible at fixed 0.5 and "
                "1.0 ATR distances over 15, 60, and 240 minutes, and can any causal call "
                "beat a naive rejection rule and matched no-level episodes?"
            ),
            "minimum_controls": 3,
            "scope_guard": (
                "Exploratory only; the 10-of-12 Generation 20 away paths were observed "
                "after selection and cannot be treated as a confirmed edge."
            ),
        },
        {
            "branch_id": "g21f_change_point_avwap_refinement",
            "status": "parked_rejected_parent",
            "route_family": "rejected_level_source",
            "plain_question": "Should the rejected change-point AVWAP definition be tuned?",
            "park_reason": (
                "No coordinate passed the complete five-control ladder; tuning the same "
                "holdouts would be event fitting without a new rational mechanism."
            ),
        },
        {
            "branch_id": "g21g_external_context_conditioning",
            "status": "parked_coverage",
            "route_family": "external_context",
            "plain_question": (
                "Do orderbook, news, web, or global-market observations condition reactions?"
            ),
            "park_reason": (
                "Wait until a timestamp-ready source spans two complete evaluation blocks."
            ),
        },
    ]


def validate_results() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    state = json.loads(STATE_RESULT.read_text(encoding="utf-8"))
    avwap = json.loads(AVWAP_RESULT.read_text(encoding="utf-8"))
    minute = json.loads(ONE_MINUTE_RESULT.read_text(encoding="utf-8"))
    expected = (
        (state, "completed_generation20_state_attribution"),
        (avwap, "completed_generation20_change_point_avwap"),
        (minute, "completed_generation20_one_minute_replay"),
    )
    for result, status in expected:
        if result.get("status") != status:
            raise ValueError(f"Generation 20 sibling is not terminal: {result.get('status')}")
    return state, avwap, minute


def review(run_id: str, *, overwrite: bool = False) -> int:
    state_result, avwap_result, minute_result = validate_results()
    run_dir = REVIEW_ROOT / run_id
    result_path = run_dir / "g20_joint_review.json"
    queue_path = run_dir / "g21_sibling_branch_queue.json"
    if result_path.is_file() and not overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    state = pd.read_csv(STATE_DECISIONS)
    minute = pd.read_csv(ONE_MINUTE_DECISIONS)
    minute_outcomes = pd.read_csv(ONE_MINUTE_OUTCOMES)
    strict = state.loc[state["status"].eq("strict_holdout_confirmation")]
    if len(strict) != 1:
        raise ValueError(f"Expected exactly one Generation 20 strict state row, got {len(strict)}.")
    lead = strict.iloc[0]
    actual_paths = minute_outcomes.loc[
        minute_outcomes["episode_kind"].eq("actual_level")
    ]
    queue = generation21_queue()
    active = [item for item in queue if item["status"].startswith("active_")]
    parked = [item for item in queue if item["status"].startswith("parked_")]
    run_dir.mkdir(parents=True, exist_ok=True)
    queue_record = {
        "schema_version": 1,
        "generation": 21,
        "created_at_utc": g0.utc_now(),
        "status": "queued_after_complete_generation20_joint_review",
        "active_siblings": len(active),
        "parked_siblings": len(parked),
        "batch_rule": (
            "Freeze and implement all five active siblings before opening any "
            "Generation 21 outcome; review them together before descendants."
        ),
        "siblings": queue,
        "source_contracts": {
            "generation20_freeze": artifact(g20z.FREEZE_PATH),
            "generation20_state_result": artifact(STATE_RESULT),
            "generation20_avwap_result": artifact(AVWAP_RESULT),
            "generation20_one_minute_result": artifact(ONE_MINUTE_RESULT),
        },
    }
    g0.atomic_write_json(queue_record, queue_path)
    result = {
        "schema_version": 1,
        "generation": 20,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation20_joint_review",
        "all_five_active_siblings_terminal_before_review": True,
        "headline": (
            "One BTC-specific adaptive Volume Profile traffic relationship survived; "
            "no broad reaction mechanism and no directional method did."
        ),
        "retained_result": {
            "branch_id": str(lead["branch_id"]),
            "market_scope": str(lead["market_scope"]),
            "plain_meaning": (
                "For BTC only, after matching the preceding four hours of crossing, "
                "volume, range, and movement state, contact with the nearest current "
                "adaptive Volume Profile area was followed by more repeated crossings "
                "over four hours than all four artificial-location controls in both "
                "later date blocks."
            ),
            "minimum_equal_coin_difference": float(
                lead["minimum_equal_coin_difference"]
            ),
            "interpretation_limit": (
                "This is a BTC traffic-location lead, not a broad coin rule, proof of "
                "causation, or a direction forecast."
            ),
        },
        "rejected_or_parked_results": {
            "donchian_onset_after_breakout_state_matching": "not_retained",
            "level_plus_local_activity_named_regimes": "not_retained",
            "change_point_anchored_vwap_coordinates": "not_retained",
            "one_minute_direction_calls": "not_retained",
        },
        "one_minute_plain_result": {
            "actual_episodes": len(actual_paths),
            "away_paths": int(
                actual_paths["path_relative_to_approach"].eq("away").sum()
            ),
            "through_paths": int(
                actual_paths["path_relative_to_approach"].eq("through").sum()
            ),
            "best_joint_success_rate": float(
                minute["actual_joint_success_rate"].max()
            ),
            "interpretation": (
                "The selected levels all moved at least one quarter ATR within an hour, "
                "but the six causal calls guessed the first side poorly. Ten of twelve "
                "paths initially moved away from the approached boundary; because that "
                "pattern was noticed after this sample was selected, it is only a lead "
                "for a disjoint diagnostic, not a result to trade."
            ),
        },
        "breadth_assessment": {
            "strict_state_rows": int(state_result["strict_rows"]),
            "strict_avwap_rows": int(avwap_result["strict_rows"]),
            "retained_direction_methods": int(minute_result["retained_methods"]),
            "broad_cross_coin_reaction_found": False,
            "joint_reaction_and_direction_at_or_above_55pct": False,
        },
        "generation21_active_siblings": len(active),
        "generation21_parked_siblings": len(parked),
        "research_boundary": {
            "no_trading_promotion": True,
            "no_profit_target": True,
            "result_spawned_branches_queued_only_after_joint_review": True,
        },
        "artifacts": {"generation21_queue": artifact(queue_path)},
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

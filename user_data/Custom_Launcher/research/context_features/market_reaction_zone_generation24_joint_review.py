"""Jointly review every Generation 24 sibling and queue a balanced Generation 25."""

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


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation24 as g24f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_joint_review as g23j,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_anchored_vwap as g24d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_context_coverage as g24b,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_freeze as g24z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_generic_indicator_context as g24e,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_round_distribution_convergence as g24c,
)


ANALYSIS_PATH = Path(__file__).resolve()
DEFAULT_REVIEW_ID = "g24_joint_review_20260828a"
REVIEW_ROOT = g24z.OUTPUT_ROOT / "joint_review"

DIRECT_ROUTES = {
    "cross_asset_context_at_levels": {
        "result": g24b.RECORD_ROOT
        / g24b.DEFAULT_RUN_ID
        / "g24_context_coverage_result.json",
        "status": "completed_generation24_context_coverage",
        "controls": tuple(g24b.CONTROLS),
    },
    "round_distribution_convergence": {
        "result": g24c.RECORD_ROOT
        / g24c.DEFAULT_RUN_ID
        / "g24_round_distribution_convergence_result.json",
        "status": "completed_generation24_round_distribution_convergence",
        "controls": tuple(g24c.CONTROLS),
    },
    "causal_anchored_vwap": {
        "result": g24d.RECORD_ROOT
        / g24d.DEFAULT_RUN_ID
        / "g24_anchored_vwap_result.json",
        "status": "completed_generation24_anchored_vwap",
        "controls": tuple(g24d.CONTROLS),
    },
    "generic_multitimeframe_indicator_context": {
        "result": g24e.RECORD_ROOT
        / g24e.DEFAULT_RUN_ID
        / "g24_generic_indicator_context_result.json",
        "status": "completed_generation24_generic_indicator_context",
        "controls": tuple(g24e.CONTROLS),
    },
}
FREQAI_NORMAL_RESULT = (
    g24f.RECORD_ROOT / f"{g24f.DEFAULT_RUN_STEM}_normal" / "g24_freqai_result.json"
)
FREQAI_MEME_RESULT = (
    g24f.RECORD_ROOT / f"{g24f.DEFAULT_RUN_STEM}_meme" / "g24_freqai_result.json"
)
FREQAI_JOINT_RESULT = (
    g24f.RECORD_ROOT
    / g24f.DEFAULT_JOINT_REVIEW_ID
    / "g24_freqai_joint_review.json"
)

VWAP_FRESH_CANDIDATE = {
    "scope_kind": "anchored_vwap_band",
    "scope_value": "1d__current_session_through_previous_completed_candle__centre_0.0",
    "metric": "crossing_count",
    "horizon_hours": 8,
    "market_scope": "all_normal",
}


def artifact(path: Path) -> dict[str, Any]:
    return g24z.artifact(path)


def native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def load_result(path: Path, expected_status: str) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != expected_status:
        raise ValueError(f"Generation 24 result is not terminal: {path}")
    return result


def validate_results() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for item in DIRECT_ROUTES.values():
        load_result(Path(item["result"]), str(item["status"]))
    normal = load_result(FREQAI_NORMAL_RESULT, "completed_generation24_freqai_cohort")
    meme = load_result(FREQAI_MEME_RESULT, "completed_generation24_freqai_cohort")
    joint = load_result(
        FREQAI_JOINT_RESULT, "completed_generation24_freqai_joint_review"
    )
    if normal.get("cohort") != "normal" or meme.get("cohort") != "meme":
        raise ValueError("Generation 24 FreqAI cohorts changed identity.")
    if not joint.get("both_cohorts_completed_before_review"):
        raise ValueError("Generation 24 FreqAI cohorts were not jointly reviewed.")
    return normal, meme, joint


def direct_control_profile(result: dict[str, Any]) -> list[dict[str, Any]]:
    scores = pd.read_csv(result["artifacts"]["scores"]["path"])
    rows: list[dict[str, Any]] = []
    for comparison, cell in scores.groupby("comparison", observed=True, sort=True):
        rows.append(
            {
                "comparison": str(comparison),
                "eligible_period_scope_cells": len(cell),
                "point_pass_fraction": float(
                    g23j.as_bool(cell["point_period_pass"]).mean()
                ),
                "strict_pass_fraction": float(
                    g23j.as_bool(cell["strict_period_pass"]).mean()
                ),
                "positive_equal_coin_difference_fraction": float(
                    pd.to_numeric(cell["equal_coin_difference"], errors="coerce")
                    .gt(0.0)
                    .mean()
                ),
                "median_equal_coin_difference": float(
                    pd.to_numeric(cell["equal_coin_difference"], errors="coerce").median()
                ),
            }
        )
    return rows


def direct_summary(result: dict[str, Any], controls: Sequence[str]) -> dict[str, Any]:
    summary = g23j.direct_summary(result, len(controls))
    summary["control_profile"] = direct_control_profile(result)
    return summary


def anchored_vwap_candidate_audit(result: dict[str, Any]) -> dict[str, Any]:
    scores = pd.read_csv(result["artifacts"]["scores"]["path"])
    selected = scores.copy()
    for key, value in VWAP_FRESH_CANDIDATE.items():
        selected = selected.loc[selected[key].eq(value)]
    if len(selected) != len(g24d.CONTROLS) * 2:
        raise ValueError("Generation 24 anchored-VWAP candidate ladder is incomplete.")
    columns = [
        "period",
        "comparison",
        "eligible_coins",
        "positive_coins",
        "positive_coin_fraction",
        "equal_coin_difference",
        "bootstrap_lower_95",
        "point_period_pass",
        "strict_period_pass",
    ]
    records = [
        {key: native(value) for key, value in row.items()}
        for row in selected[columns].sort_values(["period", "comparison"]).to_dict("records")
    ]
    return {
        "selected_only_after_complete_generation24_review": True,
        "question": dict(VWAP_FRESH_CANDIDATE),
        "point_period_control_passes": int(
            g23j.as_bool(selected["point_period_pass"]).sum()
        ),
        "strict_period_control_passes": int(
            g23j.as_bool(selected["strict_period_pass"]).sum()
        ),
        "required_period_control_cells": int(len(g24d.CONTROLS) * 2),
        "random_recent_eligible_coins": [
            int(value)
            for value in selected.loc[
                selected["comparison"].eq("random_recent_analogue"), "eligible_coins"
            ]
        ],
        "period_control_rows": records,
        "classification": "incomplete_control_coverage_with_positive_fresh_test_lead",
    }


def freqai_attribution_summary(
    normal: dict[str, Any], meme: dict[str, Any]
) -> dict[str, Any]:
    cohort_results = {"normal": normal, "meme": meme}
    profiles: dict[str, Any] = {}
    for cohort, result in cohort_results.items():
        comparisons = pd.read_csv(result["artifacts"]["control_comparisons"]["path"])
        decisions = pd.read_csv(result["artifacts"]["decisions"]["path"])
        model_decisions = pd.read_csv(
            result["artifacts"]["model_seed_decisions"]["path"]
        )
        controls: list[dict[str, Any]] = []
        for control, cell in comparisons.groupby("control", observed=True, sort=True):
            row: dict[str, Any] = {
                "control": str(control),
                "period_model_target_cells": len(cell),
                "candidate_gate_pass_fraction": float(
                    g23j.as_bool(cell["candidate_gate_pass"]).mean()
                ),
                "full_model_control_pass_fraction": float(
                    g23j.as_bool(cell["period_control_pass"]).mean()
                ),
                "mean_mae_advantage": float(
                    (
                        pd.to_numeric(cell["control_mae"], errors="coerce")
                        - pd.to_numeric(cell["candidate_mae"], errors="coerce")
                    ).mean()
                ),
            }
            if control != "constant_training_median":
                row.update(
                    {
                        "mean_spearman_advantage": float(
                            (
                                pd.to_numeric(
                                    cell["candidate_spearman"], errors="coerce"
                                )
                                - pd.to_numeric(
                                    cell["control_spearman"], errors="coerce"
                                )
                            ).mean()
                        ),
                        "mean_top_bottom_advantage": float(
                            (
                                pd.to_numeric(
                                    cell["candidate_top_minus_bottom"], errors="coerce"
                                )
                                - pd.to_numeric(
                                    cell["control_top_minus_bottom"], errors="coerce"
                                )
                            ).mean()
                        ),
                    }
                )
            controls.append(row)
        profiles[cohort] = {
            "target_scope_rows": len(decisions),
            "retained_target_scope_rows": int(
                g23j.as_bool(decisions["all_model_seed_cells_pass"]).sum()
            ),
            "model_seed_target_scope_rows": len(model_decisions),
            "complete_model_seed_ladders": int(
                g23j.as_bool(
                    model_decisions["complete_control_and_period_ladder"]
                ).sum()
            ),
            "passing_model_seed_ladders": int(
                g23j.as_bool(
                    model_decisions["all_controls_both_periods_pass"]
                ).sum()
            ),
            "controls": controls,
        }
    return {
        "cohorts": profiles,
        "plain_result": (
            "Recent market state repeatedly ranked future volume and range activity, "
            "but the full model rarely beat market-state-only or time-displaced level "
            "features. Level geometry did not add stable activity information."
        ),
        "classification": "provisional_market_activity_context_not_level_attribution",
    }


def generation25_queue() -> list[dict[str, Any]]:
    """Queue five distinct questions; do not recurse into a single Generation 24 row."""
    return [
        {
            "branch_id": "g25a_daily_anchored_vwap_fresh_confirmation",
            "status": "active_pending_freeze",
            "route_family": "anchored_vwap_untouched_confirmation",
            "parent_evidence": "g24_daily_current_session_vwap_centre_crossing_h8",
            "plain_question": (
                "On genuinely later data, does the current daily session's VWAP centre "
                "still locate unusually high eight-hour crossing traffic across normal coins?"
            ),
            "scope_guard": (
                "Keep the selected daily/current-session/centre/crossing/h8/all-normal "
                "question fixed. Repair recent-ordinary control density without changing "
                "the level, width, target, horizon, or market scope. Test new chronology, "
                "not the Generation 24 periods used to select it."
            ),
            "pass_rule": (
                "Retain only if both fresh chronological blocks have adequate normal-coin "
                "support and beat matched random, recent ordinary, near-miss, stale, and "
                "price-shift controls with positive equal-coin uncertainty."
            ),
        },
        {
            "branch_id": "g25b_convergence_control_representation_repair",
            "status": "active_pending_freeze",
            "route_family": "cluster_control_representation",
            "parent_evidence": "g24_no_complete_convergence_control_ladder",
            "plain_question": (
                "Can round-number and rolling-distribution convergence be tested fairly "
                "when every real convergence event receives equally supported controls?"
            ),
            "scope_guard": (
                "Repair control construction before outcomes, preserve the entire frozen "
                "round-step/lookback/quantile family, and use per-event or rational pooled "
                "controls that match density. Do not select the best Generation 24 combination."
            ),
            "pass_rule": (
                "First pass the representation gate. A market lead then requires convergence "
                "to beat both isolated components, fake convergence, and every artificial "
                "location control in both periods."
            ),
        },
        {
            "branch_id": "g25c_connected_volume_profile_zones",
            "status": "active_pending_freeze",
            "route_family": "volume_profile_zone_representation",
            "parent_evidence": "earlier_vp_activity_with_weak_exact_node_portability",
            "plain_question": (
                "Do connected Volume Profile high-volume areas and low-volume corridors "
                "represent market reaction locations better than treating one histogram "
                "bin centre as a level?"
            ),
            "scope_guard": (
                "Use research-only variants and fixed causal prominence/percentile rules. "
                "Compare connected zones with the current point-node representation and "
                "same-density, shifted, stale, near-miss, and random controls. Do not edit "
                "canonical indicators or tune to remembered events."
            ),
            "pass_rule": (
                "Retain a representation only if its rationale fixes measurable fragmentation, "
                "it adds reaction evidence beyond point nodes and density controls across "
                "periods, and its added complexity is justified."
            ),
        },
        {
            "branch_id": "g25d_market_state_only_activity_portability",
            "status": "active_pending_freeze",
            "route_family": "ohlcv_market_activity_context",
            "parent_evidence": "g24_full_model_tied_market_state_and_permuted_levels",
            "plain_question": (
                "Without any level claim, can recent OHLCV and broad-market state reliably "
                "rank which coming hours will have unusually high volume or range activity?"
            ),
            "scope_guard": (
                "Freeze the same unsigned volume/range percentile family across normal and "
                "meme cohorts, use multiple seeds and model families, and compare with "
                "constant, shuffled, stale-state, and simple recent-activity baselines."
            ),
            "pass_rule": (
                "Retain only as an activity-context lead if it repeats across models, periods, "
                "and its declared coin scope. It cannot be relabelled as a reaction zone, "
                "direction prediction, or trading signal."
            ),
        },
        {
            "branch_id": "g25e_external_source_overlap_gate",
            "status": "active_pending_freeze",
            "route_family": "external_source_timestamp_coverage",
            "parent_evidence": "requested_orderbook_news_global_breadth_with_parked_coverage",
            "plain_question": (
                "Do historical orderbook, GDELT/GKG, live media, or global-market sources now "
                "overlap enough causal level events to support one honest bounded experiment?"
            ),
            "scope_guard": (
                "Audit source-specific availability before reaction outcomes. Missing data "
                "remains missing, venues stay separate, and only proven timestamp-safe windows "
                "may advance. An unsupported source terminates as a coverage finding."
            ),
            "pass_rule": (
                "Advance only a source block with sufficient independent events in chronological "
                "training and evaluation windows and no future-timestamp or gap-carry failures."
            ),
        },
        {
            "branch_id": "g25f_more_generic_indicator_level_tuning",
            "status": "parked_generation24_null",
            "route_family": "generic_indicator_context_at_levels",
            "park_reason": (
                "One thousand complete Generation 24 questions produced no point or strict "
                "whole-ladder result; same-state no-level and recent-time controls often won."
            ),
        },
        {
            "branch_id": "g25g_more_cross_asset_band_repair",
            "status": "parked_repair_did_not_attribute_levels",
            "route_family": "cross_asset_context_level_attribution",
            "park_reason": (
                "Generation 24 repaired sampling broadly enough to create 480 complete "
                "questions, but none survived and no-level controls commonly explained more."
            ),
        },
        {
            "branch_id": "g25h_level_inclusive_activity_model",
            "status": "parked_redundant_level_block",
            "route_family": "freqai_level_activity_prediction",
            "park_reason": (
                "The full model did not reliably beat market-state-only or permuted-level "
                "models, so more level-inclusive seed tuning would chase model noise."
            ),
        },
        {
            "branch_id": "g25i_one_minute_direction_replay",
            "status": "parked_no_new_confirmed_parent_pattern",
            "route_family": "bounded_direction_microscope",
            "park_reason": (
                "Generation 24 retained no complete new reaction-zone result and prior frozen "
                "one-minute batches remained below the user's joint-success floor."
            ),
        },
        {
            "branch_id": "g25j_outcome_selected_coin_subgroups",
            "status": "parked_posthoc_grouping_risk",
            "route_family": "market_group_transfer",
            "park_reason": (
                "No new mechanism survived from which to define an outcome-independent subgroup. "
                "Do not form an easy-coin group from opened Generation 24 outcomes."
            ),
        },
    ]


def review(run_id: str, *, overwrite: bool = False) -> int:
    normal, meme, _ = validate_results()
    run_dir = REVIEW_ROOT / run_id
    result_path = run_dir / "g24_joint_review.json"
    queue_path = run_dir / "g25_sibling_branch_queue.json"
    if result_path.is_file() and not overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return 0

    route_summaries: dict[str, Any] = {}
    loaded_direct: dict[str, dict[str, Any]] = {}
    for name, item in DIRECT_ROUTES.items():
        result = load_result(Path(item["result"]), str(item["status"]))
        loaded_direct[name] = result
        route_summaries[name] = direct_summary(result, item["controls"])
    route_summaries["causal_anchored_vwap"]["fresh_candidate_audit"] = (
        anchored_vwap_candidate_audit(loaded_direct["causal_anchored_vwap"])
    )
    route_summaries["freqai_activity_stability_and_attribution"] = (
        freqai_attribution_summary(normal, meme)
    )

    queue = generation25_queue()
    active = [item for item in queue if item["status"].startswith("active_")]
    parked = [item for item in queue if item["status"].startswith("parked_")]
    if len(active) != 5 or len({item["route_family"] for item in active}) != 5:
        raise ValueError("Generation 25 queue lost five-route breadth.")

    source_contracts = {
        "generation24_freeze": artifact(g24z.FREEZE_PATH),
        **{
            f"{name}_result": artifact(Path(item["result"]))
            for name, item in DIRECT_ROUTES.items()
        },
        "freqai_normal_result": artifact(FREQAI_NORMAL_RESULT),
        "freqai_meme_result": artifact(FREQAI_MEME_RESULT),
        "freqai_joint_result": artifact(FREQAI_JOINT_RESULT),
        "analysis_script": artifact(ANALYSIS_PATH),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    queue_record = {
        "schema_version": 1,
        "generation": 25,
        "created_at_utc": g0.utc_now(),
        "status": "queued_after_complete_generation24_joint_review",
        "active_siblings": len(active),
        "parked_siblings": len(parked),
        "batch_rule": (
            "Freeze all five active siblings and their support gates before opening any "
            "Generation 25 outcome; jointly review every sibling before descendants."
        ),
        "siblings": queue,
        "source_contracts": source_contracts,
    }
    g0.atomic_write_json(queue_record, queue_path)

    result = {
        "schema_version": 1,
        "generation": 24,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation24_joint_review",
        "all_five_active_siblings_terminal_before_review": True,
        "headline": (
            "No Generation 24 question survived its complete robustness ladder. Recent "
            "market state still predicted unsigned activity, but level geometry was "
            "redundant. Daily current-session VWAP crossing traffic is the only direct "
            "fresh-confirmation candidate, and it remains incomplete rather than retained."
        ),
        "route_results": route_summaries,
        "plain_interpretation": {
            "freqai": (
                "The models often ranked busy coming hours, but replacing or time-displacing "
                "the level inputs changed little. The useful information was recent market "
                "activity, not dependable level geometry."
            ),
            "cross_asset_context": (
                "Broad BTC and cross-coin states did not make level contacts more reactive "
                "than the full artificial and same-state no-level ladder in both periods."
            ),
            "round_distribution_convergence": (
                "The exact convergence grid was too sparse for its artificial controls. "
                "This is a representation and coverage failure, not evidence that the "
                "underlying convergence mechanism is either useful or useless."
            ),
            "anchored_vwap": (
                "The current daily session VWAP centre showed extra eight-hour crossing "
                "traffic across normal coins against eight of ten period-control cells. "
                "The recent-ordinary control had enough rows for only one coin in each "
                "period, so a fresh test is justified but no lead is retained yet."
            ),
            "generic_indicators": (
                "RSI, Bollinger, MACD, EMA, and ATR states at levels produced no complete "
                "two-period result. Similar states away from levels often explained more, "
                "so further threshold tuning is parked."
            ),
        },
        "breadth_assessment": {
            "retained_generation24_routes": 0,
            "retained_direct_reaction_zone_routes": 0,
            "fresh_confirmation_candidates": 1,
            "representation_failures": 1,
            "provisional_nonlevel_activity_context_routes": 1,
            "broad_cross_coin_price_reaction_found": False,
            "joint_reaction_and_direction_at_or_above_55pct": False,
            "one_minute_direction_lane_justified": False,
            "same_holdout_trading_promotion_justified": False,
        },
        "generation25_active_siblings": len(active),
        "generation25_parked_siblings": len(parked),
        "research_boundary": {
            "no_trading_promotion": True,
            "no_profit_target": True,
            "no_signed_direction": True,
            "incomplete_control_ladders_not_treated_as_leads": True,
            "market_activity_not_relabelled_as_level_reaction": True,
            "generic_null_routes_not_tuned": True,
            "result_spawned_branches_queued_only_after_joint_review": True,
        },
        "artifacts": {"generation25_queue": artifact(queue_path)},
        "source_contracts": source_contracts,
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

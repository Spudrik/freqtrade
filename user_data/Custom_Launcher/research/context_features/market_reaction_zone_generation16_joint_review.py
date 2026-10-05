"""Collapse every frozen Generation 16 sibling before queuing descendants."""

from __future__ import annotations

# The repository root is added before project imports when run directly.
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
    market_reaction_zone_freqai_generation16 as g16f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation13_freeze as g13z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation16_direct_attribution as g16d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation16_freeze as g16z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation16_one_minute_analysis as g16m,
)


DEFAULT_REVIEW_ID = "g16_joint_review_20260822a"
REVIEW_ROOT = g16z.FREEZE_PATH.parent / "g16_broad_attribution" / "joint_review"
DIRECT_RESULT = g16d.RECORD_ROOT / g16d.DEFAULT_RUN_ID / "g16_direct_attribution_result.json"
FREQAI_JOINT_RESULT = (
    g16f.RECORD_ROOT / g16f.DEFAULT_JOINT_REVIEW_ID / "g16_freqai_joint_review.json"
)
ONE_MINUTE_RESULT = g16m.REPORT_ROOT / g16m.DEFAULT_RUN_ID / "g16_one_minute_summary.json"

STANDARD = g13z.STANDARD
RECENT = g13z.RECENT
NORMAL_WINDOWS = {STANDARD, RECENT}
MEME_WINDOWS = {STANDARD}

BASE_REQUIRED = {
    "completed_contact_activity_control": {
        "genuine_near_miss",
        "matched_ordinary_time",
    },
    "stale_shift_and_density_controls": {
        "causal_72h_old_level",
        "price_shifted_level",
        "matched_ordinary_time",
    },
    "pre_contact_to_post_contact_change": {
        "actual_transition_vs_zero",
        "genuine_near_miss",
        "matched_ordinary_time",
    },
}
CONTEXT_REQUIRED = {
    "local_activity": {"high_vs_low"},
    "wider_activity": {"high_vs_low"},
    "completed_8h_activity": {"high_vs_low"},
    "local_plus_wider": {"vs_local_only", "vs_wider_only", "vs_neither"},
    "local_plus_8h": {"vs_local_only", "vs_8h_only", "vs_neither"},
    "wider_plus_8h": {"vs_wider_only", "vs_8h_only", "vs_neither"},
    "local_plus_wider_plus_8h": {
        "vs_local_plus_wider",
        "vs_local_plus_8h",
        "vs_wider_plus_8h",
        "vs_neither",
    },
}
DENSITY_PLACEBOS = ("stale_density_placebo", "shuffled_density_placebo")


def artifact(path: Path) -> dict[str, Any]:
    return g16f.artifact(path)


def bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().eq("true")


def expected_windows(market_scope: str) -> set[str]:
    return MEME_WINDOWS if market_scope == "top10_memes" else NORMAL_WINDOWS


def required_comparisons(route_id: str, scope_value: str, frame: DataFrame) -> set[str]:
    if route_id in BASE_REQUIRED:
        return set(BASE_REQUIRED[route_id])
    if route_id == "local_wider_and_completed_8h_activity_interaction":
        return set(CONTEXT_REQUIRED[scope_value])
    if route_id in {"head_to_head_family_and_timeframe", "level_density_and_overlap"}:
        return set(frame["comparison"].astype(str).unique())
    raise ValueError(f"Unknown Generation 16 direct route: {route_id}")


def complete_evidence(
    frame: DataFrame, comparisons: set[str], windows: set[str]
) -> tuple[bool, bool, bool, str]:
    selected = frame.loc[
        frame["comparison"].isin(comparisons) & frame["window"].isin(windows)
    ].copy()
    complete = set(selected["comparison"]) == comparisons and all(
        set(group["window"]) == windows
        for _, group in selected.groupby("comparison", observed=True)
    )
    signs = set(selected["effect_sign"].astype(str)).difference({"mixed"})
    same_sign = len(signs) == 1 and not selected["effect_sign"].eq("mixed").any()
    point = bool(complete and same_sign and bool_series(selected["point_pass"]).all())
    strict = bool(point and bool_series(selected["strict_pass"]).all())
    sign = next(iter(signs)) if same_sign else "mixed"
    return complete, point, strict, sign


def density_current_decision(
    all_rows: DataFrame,
    current: DataFrame,
    *,
    metric: str,
    horizon: int,
    market_scope: str,
    windows: set[str],
) -> tuple[bool, bool, bool, str, list[str]]:
    complete, point, strict, sign = complete_evidence(current, {"dense_vs_isolated"}, windows)
    contamination: list[str] = []
    if point:
        for scope in DENSITY_PLACEBOS:
            placebo = all_rows.loc[
                all_rows["scope_value"].eq(scope)
                & all_rows["metric"].eq(metric)
                & all_rows["horizon_hours"].eq(horizon)
                & all_rows["market_scope"].eq(market_scope)
            ]
            p_complete, p_point, _, p_sign = complete_evidence(
                placebo, {"dense_vs_isolated"}, windows
            )
            if p_complete and p_point and p_sign == sign:
                contamination.append(scope)
    if contamination:
        point = False
        strict = False
    return complete, point, strict, sign, contamination


def claim_class(market_scope: str) -> str:
    if market_scope == "top10_memes":
        return "top10_meme_cohort"
    if market_scope == "btc_separate":
        return "btc_asset_specific"
    if market_scope == "all_normal":
        return "broad_normal_cohort"
    return "similar_coin_subgroup"


def collapse_direct(frame: DataFrame) -> DataFrame:
    output: list[dict[str, Any]] = []
    keys = ["route_id", "scope_value", "metric", "horizon_hours", "market_scope"]
    excluded = set(DENSITY_PLACEBOS)
    candidates = frame.loc[~frame["scope_value"].isin(excluded)].copy()
    for key, cell in candidates.groupby(keys, observed=True, sort=False):
        route_id, scope_value, metric, horizon, market_scope = key
        windows = expected_windows(str(market_scope))
        contamination: list[str] = []
        if route_id == "level_density_and_overlap" and scope_value == "current_density":
            complete, point, strict, sign, contamination = density_current_decision(
                frame,
                cell,
                metric=str(metric),
                horizon=int(horizon),
                market_scope=str(market_scope),
                windows=windows,
            )
            comparisons = {"dense_vs_isolated", *DENSITY_PLACEBOS}
        else:
            comparisons = required_comparisons(str(route_id), str(scope_value), cell)
            complete, point, strict, sign = complete_evidence(cell, comparisons, windows)
        if strict:
            status = "strict_repeated_lead"
        elif point:
            status = "point_repeated_lead"
        else:
            status = "not_retained_after_whole_question_gate"
        selected = cell.loc[cell["window"].isin(windows)]
        output.append(
            {
                **dict(zip(keys, key, strict=True)),
                "claim_class": claim_class(str(market_scope)),
                "status": status,
                "expected_windows": ",".join(sorted(windows)),
                "expected_comparisons": ",".join(sorted(comparisons)),
                "whole_question_complete": complete,
                "all_required_comparisons_point": point,
                "all_required_comparisons_strict": strict,
                "effect_sign": sign,
                "placebo_contamination": ",".join(contamination),
                "minimum_effect": float(
                    pd.to_numeric(selected["minimum_effect"], errors="coerce").min()
                ),
                "maximum_effect": float(
                    pd.to_numeric(selected["maximum_effect"], errors="coerce").max()
                ),
            }
        )
    return DataFrame.from_records(output).sort_values(keys).reset_index(drop=True)


def freqai_density_diagnostics() -> list[dict[str, Any]]:
    cells = (
        ("normal_standard", f"{g16f.DEFAULT_RUN_STEM}_standard_normal"),
        ("normal_recent", f"{g16f.DEFAULT_RUN_STEM}_recent_normal"),
        ("top10_memes", f"{g16f.DEFAULT_RUN_STEM}_standard_meme"),
    )
    frames: list[DataFrame] = []
    for cell_name, run_id in cells:
        frame = pd.read_csv(g16f.RECORD_ROOT / run_id / "g16_pair_scores.csv")
        frame["cell"] = cell_name
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    selected = combined.loc[
        combined["route_id"].eq("level_density_increment")
        & combined["target"].isin([f"&-g16_crossings_h{horizon}" for horizon in (1, 2, 4)])
    ].copy()
    rows: list[dict[str, Any]] = []
    for (target, cell), frame in selected.groupby(["target", "cell"], observed=True, sort=False):
        rows.append(
            {
                "target": target,
                "cell": cell,
                "coin_period_rows": len(frame),
                "positive_coin_period_rows": int(frame["paired_mae_gain"].gt(0).sum()),
                "mean_relative_mae_gain": float(frame["relative_mae_gain"].mean()),
                "minimum_relative_mae_gain": float(frame["relative_mae_gain"].min()),
                "mean_spearman_change": float(frame["spearman_change"].mean()),
            }
        )
    return rows


def reaction_classifier_diagnostics() -> dict[str, Any]:
    run_ids = (
        f"{g16f.DEFAULT_RUN_STEM}_standard_normal",
        f"{g16f.DEFAULT_RUN_STEM}_recent_normal",
        f"{g16f.DEFAULT_RUN_STEM}_standard_meme",
    )
    frames: list[DataFrame] = []
    for run_id in run_ids:
        frame = pd.read_csv(g16f.RECORD_ROOT / run_id / "g16_reaction_diagnostics.csv")
        frames.append(
            frame.loc[
                frame["route_id"].eq("completed_8h_increment")
                & frame["target"].eq("&-g16_reaction_h4")
            ]
        )
    selected = pd.concat(frames, ignore_index=True)
    accuracy = pd.to_numeric(selected["candidate_accuracy_at_half"], errors="coerce")
    majority = pd.to_numeric(selected["candidate_majority_accuracy"], errors="coerce")
    auc = pd.to_numeric(selected["candidate_rank_auc"], errors="coerce")
    balanced = pd.to_numeric(selected["candidate_balanced_accuracy_at_half"], errors="coerce")
    return {
        "periods": len(selected),
        "minimum_accuracy": float(accuracy.min()),
        "maximum_accuracy": float(accuracy.max()),
        "minimum_balanced_accuracy": float(balanced.min()),
        "maximum_balanced_accuracy": float(balanced.max()),
        "minimum_rank_auc": float(auc.min()),
        "maximum_rank_auc": float(auc.max()),
        "minimum_accuracy_gain_over_majority": float((accuracy - majority).min()),
        "maximum_accuracy_gain_over_majority": float((accuracy - majority).max()),
        "reaction_only_not_direction": True,
        "increment_status": "point_all_three_cells_not_strict",
    }


def branch_queue() -> list[dict[str, Any]]:
    return [
        {
            "branch_batch": "g17a_density_geometry_decomposition",
            "priority": 1,
            "question": (
                "Which continuous density or cluster-geometry components explain the "
                "robust FreqAI crossing gain?"
            ),
            "breadth": (
                "Normal, similar-coin subgroups, top-10 memes; 1h/4h/8h source "
                "levels; 1h/2h/4h outcomes."
            ),
            "gate": (
                "Each component and combination must beat contact-only, stale, shuffled, "
                "and every immediately simpler component in older and recent windows."
            ),
        },
        {
            "branch_batch": "g17b_crossing_semantics",
            "priority": 2,
            "question": (
                "Do calculated areas predict a useful kind of repeat interaction rather "
                "than merely any recrossing?"
            ),
            "breadth": (
                "First recross, repeated recross, dwell, rejection, false break, and "
                "width-normalised opportunity across all retained families and cohorts."
            ),
            "gate": (
                "Must remain stronger than matched ordinary time and genuine near miss "
                "after equal contact opportunity."
            ),
        },
        {
            "branch_batch": "g17c_reaction_probability_calibration",
            "priority": 3,
            "question": (
                "Can completed 8h, local, wider-market, and density state produce a "
                "calibrated probability that a direction-neutral reaction occurs?"
            ),
            "breadth": (
                "Separate normal and meme models, rolling unseen windows, component "
                "ablations, 1h/2h/4h reaction definitions."
            ),
            "gate": (
                "Beat contact-only and majority baselines in balanced accuracy, rank AUC, "
                "calibration, and weekly uncertainty; combinations beat all components."
            ),
        },
        {
            "branch_batch": "g17d_broad_level_sources",
            "priority": 4,
            "question": (
                "Do other rational level calculations locate reaction areas as well as "
                "or better than the retained VP, range, and round-number families?"
            ),
            "breadth": (
                "Existing custom VP variants plus VWAP/anchored VWAP, pivots, Donchian, "
                "volatility bands, and other causal level sources selected before outcomes."
            ),
            "gate": (
                "No event-specific tuning; test singles and clusters equally; require "
                "normal chronology repetition or a coherent similar-coin/meme cohort."
            ),
        },
        {
            "branch_batch": "g17e_external_context_regimes",
            "priority": 5,
            "question": (
                "Are crypto-specific area reactions more predictable when news, global "
                "markets, and order-book pressure are quiet or coherently aligned?"
            ),
            "breadth": (
                "Only source-ready periods; quiet/middle/shock news, global breadth, BTC "
                "order-book pressure, and explicit missing-source controls."
            ),
            "gate": (
                "Context must add information beyond local/contact/density baselines and "
                "survive source-missing and shuffled-context placebos."
            ),
        },
        {
            "branch_batch": "g17f_bounded_one_minute_expansion",
            "priority": 6,
            "question": (
                "After a robust reaction candidate is identified, can 1-minute pressure, "
                "trend, momentum, order book, and market context predict its direction?"
            ),
            "breadth": (
                "The five boundary-extension episodes plus a larger causal sample across "
                "normal, similar-coin, and meme strata with windows scaled to source TF."
            ),
            "gate": (
                "Joint reaction-and-direction success counts abstentions as failures; 55% "
                "is the minimum lead and 65% the target on independent episodes."
            ),
        },
    ]


def validate_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    direct = json.loads(DIRECT_RESULT.read_text(encoding="utf-8"))
    freqai = json.loads(FREQAI_JOINT_RESULT.read_text(encoding="utf-8"))
    minute = json.loads(ONE_MINUTE_RESULT.read_text(encoding="utf-8"))
    if direct.get("status") != "completed_generation16_direct_attribution":
        raise ValueError("Generation 16 direct sibling is not terminal.")
    if freqai.get("status") != "completed_generation16_freqai_joint_review":
        raise ValueError("Generation 16 FreqAI sibling is not terminal.")
    if minute.get("status") != "completed_generation16_one_minute_diagnostic":
        raise ValueError("Generation 16 one-minute sibling is not terminal.")
    return direct, freqai, minute


def run_review(review_id: str) -> dict[str, Any]:
    direct, freqai, minute = validate_inputs()
    direct_rows = pd.read_csv(direct["artifacts"]["period_leads"]["path"])
    collapsed = collapse_direct(direct_rows)
    freqai_rows = pd.read_csv(freqai["artifacts"]["joint_decisions"]["path"])
    output_dir = REVIEW_ROOT / review_id
    output_dir.mkdir(parents=True, exist_ok=True)
    direct_path = output_dir / "g16_direct_whole_question_decisions.csv"
    queue_path = output_dir / "g16_branch_queue.csv"
    g0.atomic_write_csv(collapsed, direct_path)
    queue = branch_queue()
    g0.atomic_write_csv(DataFrame.from_records(queue), queue_path)

    broad_normal_strict = collapsed.loc[
        collapsed["claim_class"].eq("broad_normal_cohort")
        & collapsed["status"].eq("strict_repeated_lead")
    ]
    core_direct = broad_normal_strict.loc[
        broad_normal_strict["route_id"].eq("completed_contact_activity_control")
    ]
    strict_freqai = freqai_rows.loc[freqai_rows["strict_all_three_cells"]].copy()
    point_freqai = freqai_rows.loc[
        freqai_rows["point_all_three_cells"] & ~freqai_rows["strict_all_three_cells"]
    ].copy()
    output = {
        "schema_version": 1,
        "generation": 16,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation16_joint_review",
        "all_frozen_siblings_terminal_before_review": True,
        "main_result": (
            "Calculated-area contact contains repeatable direction-neutral location "
            "information, especially for future recrossings. Continuous level-density "
            "and cluster-geometry inputs add robust FreqAI information about crossing "
            "counts. No 55% joint reaction-and-direction method was found."
        ),
        "direct_summary": {
            "whole_questions": len(collapsed),
            "strict_repeated_leads": int(collapsed["status"].eq("strict_repeated_lead").sum()),
            "point_repeated_leads": int(collapsed["status"].eq("point_repeated_lead").sum()),
            "broad_normal_strict": int(
                collapsed["claim_class"]
                .eq("broad_normal_cohort")
                .where(collapsed["status"].eq("strict_repeated_lead"), False)
                .sum()
            ),
            "core_contact_metrics": core_direct[
                ["metric", "horizon_hours", "effect_sign", "status"]
            ].to_dict("records"),
            "no_portable_head_to_head_family_or_timeframe": not bool(
                collapsed["route_id"]
                .eq("head_to_head_family_and_timeframe")
                .where(collapsed["status"].ne("not_retained_after_whole_question_gate"), False)
                .any()
            ),
            "no_portable_pre_to_post_activation": not bool(
                collapsed["route_id"]
                .eq("pre_contact_to_post_contact_change")
                .where(collapsed["status"].ne("not_retained_after_whole_question_gate"), False)
                .any()
            ),
        },
        "freqai_summary": {
            **freqai["summary"],
            "strict_all_three_details": strict_freqai[
                ["route_id", "target", "minimum_equal_coin_paired_mae_gain"]
            ].to_dict("records"),
            "point_all_three_not_strict_details": point_freqai[
                ["route_id", "target", "minimum_equal_coin_paired_mae_gain"]
            ].to_dict("records"),
            "density_crossing_diagnostics": freqai_density_diagnostics(),
            "reaction_classifier_diagnostics": reaction_classifier_diagnostics(),
        },
        "one_minute_summary": {
            "episodes": minute["actual_episodes"],
            "independent_pairs": minute["independent_pairs"],
            "best_method": minute["best_observed_causal_method_not_a_lead"],
            "methods_at_or_above_55pct": minute["methods_at_or_above_55pct"],
            "boundary_extension_indicated": minute["boundary_extension_indicated"],
        },
        "interpretation_boundaries": {
            "profit_used": False,
            "main_stage_direction_used": False,
            "reaction_accuracy_is_not_direction_accuracy": True,
            "crossing_count_prediction_is_not_a_trade_signal": True,
            "55_percent_joint_reaction_and_direction_reached": False,
            "trading_promotion_allowed": False,
        },
        "branch_layer": 1,
        "branch_batches_queued": len(queue),
        "automatic_descendant_launch": False,
        "source_contracts": {
            "generation16_freeze": artifact(g16z.FREEZE_PATH),
            "direct_result": artifact(DIRECT_RESULT),
            "freqai_joint_result": artifact(FREQAI_JOINT_RESULT),
            "one_minute_result": artifact(ONE_MINUTE_RESULT),
        },
        "artifacts": {
            "direct_whole_question_decisions": artifact(direct_path),
            "branch_queue": artifact(queue_path),
        },
    }
    result_path = output_dir / "g16_joint_review.json"
    g0.atomic_write_json(output, result_path)
    return {**output, "result_path": str(result_path.resolve())}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-id", default=DEFAULT_REVIEW_ID)
    args = parser.parse_args(argv)
    result = run_review(args.review_id)
    print(json.dumps(result, indent=2, default=g0.json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

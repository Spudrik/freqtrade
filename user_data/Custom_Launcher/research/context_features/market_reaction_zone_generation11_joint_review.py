"""Jointly review every Generation 11 direct sibling before descendants."""

from __future__ import annotations

# The review intentionally preserves long plain-language frozen questions.
# ruff: noqa: E402, E501
import os


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation11_direct_screen as g11d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation11_freeze as g11z,
)


DEFAULT_REVIEW_ID = "g11_broad_joint_review_20260822a"
DIRECT_RESULT = (
    g11d.DIRECT_ROOT / g11d.DEFAULT_RUN_ID / "g11_direct_screen_result.json"
)
REVIEW_ROOT = g11z.OUTPUT_ROOT / "generation11_review"


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def load_direct_result() -> tuple[dict[str, Any], dict[str, DataFrame]]:
    if not DIRECT_RESULT.is_file():
        raise FileNotFoundError(DIRECT_RESULT)
    result = json.loads(DIRECT_RESULT.read_text(encoding="utf-8"))
    if result.get("status") != "completed_all_seven_generation11_direct_routes":
        raise ValueError("Generation 11 direct siblings are not all terminal.")
    if result["summary"].get("routes_completed") != len(g11z.ROUTES):
        raise ValueError("Generation 11 direct route count is incomplete.")
    if result["summary"].get("profit_used") is not False:
        raise ValueError("Generation 11 direct result crossed the profit boundary.")
    frames: dict[str, DataFrame] = {}
    for key in (
        "neutral_effects",
        "neutral_candidates",
        "direction_scores",
        "direction_candidates",
        "route_decisions",
        "pair_condition_medians",
    ):
        contract = result["artifacts"][key]
        path = Path(contract["path"])
        if not path.is_file() or g0.sha256_file(path) != contract["sha256"]:
            raise ValueError(f"Generation 11 direct artifact changed: {path}")
        frames[key] = (
            pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        )
    return result, frames


def candidate_family(route: str, cell: str) -> str:
    if route == "timeframe_horizon_persistence" and "generic_prior_range" in cell:
        return "generic_prior_range_across_timeframes"
    if route == "timeframe_horizon_persistence" and "tlv2_ranked" in cell:
        return "ranked_trendline_level"
    if route == "crypto_market_alignment":
        return "local_and_wider_market_disagreement"
    if route == "cluster_obstacle_geometry":
        return "isolated_or_cluster_geometry"
    if route == "trend_timeframe_interaction":
        return "local_trend_strength_and_alignment"
    if route == "market_group_portability":
        return "rational_coin_group_portability"
    if route == "activity_displacement":
        return "local_activity_state"
    if route == "external_calm_and_stress":
        return "external_context_state"
    return route


def candidate_portability(
    candidates: DataFrame,
    pair_summaries: DataFrame,
) -> DataFrame:
    selected = candidates.loc[candidates["candidate"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    for candidate in selected.itertuples(index=False):
        horizons = g11z.HORIZON_BANDS[str(candidate.horizon_band)]
        periods = g11z.VALIDATION_PERIODS[str(candidate.cohort)]
        source = pair_summaries.loc[
            pair_summaries["route"].eq(candidate.route)
            & pair_summaries["cohort"].eq(candidate.cohort)
            & pair_summaries["cell"].eq(candidate.cell)
            & pair_summaries["metric"].eq(candidate.metric)
            & pair_summaries["horizon_hours"].isin(horizons)
            & pair_summaries["period"].isin(periods)
        ]
        actual = source.loc[source["control"].eq("actual")]
        differences: list[DataFrame] = []
        join_keys = [
            "route",
            "cohort",
            "pair",
            "period",
            "cell",
            "horizon_hours",
            "metric",
            "minimum_coins",
        ]
        for control in g11d.CONTROLS:
            paired = actual.merge(
                source.loc[source["control"].eq(control)],
                on=join_keys,
                how="inner",
                suffixes=("_actual", "_control"),
                validate="one_to_one",
            )
            paired["difference"] = (
                paired["pair_median_actual"] - paired["pair_median_control"]
            )
            differences.append(paired[["pair", "difference"]])
        per_pair = (
            pd.concat(differences, ignore_index=True)
            .groupby("pair", observed=True)["difference"]
            .agg(["mean", "min", "max", "size"])
            .reset_index()
        )
        sign = int(candidate.behaviour_sign)
        aligned = np.sign(per_pair["mean"]).eq(sign)
        leave_one_out = [
            int(np.sign(per_pair.loc[per_pair["pair"].ne(pair), "mean"].mean()))
            == sign
            for pair in per_pair["pair"]
        ]
        rows.append(
            {
                "route": candidate.route,
                "family": candidate_family(candidate.route, candidate.cell),
                "cohort": candidate.cohort,
                "cell": candidate.cell,
                "metric": candidate.metric,
                "horizon_band": candidate.horizon_band,
                "behaviour_sign": sign,
                "coins": len(per_pair),
                "coins_with_average_effect_in_declared_direction": int(aligned.sum()),
                "aligned_coin_fraction": float(aligned.mean()),
                "all_leave_one_coin_out_effects_keep_sign": bool(all(leave_one_out)),
                "mean_equal_coin_effect": float(per_pair["mean"].mean()),
                "minimum_coin_average_effect": float(per_pair["mean"].min()),
                "maximum_coin_average_effect": float(per_pair["mean"].max()),
                "portability_label": (
                    "all_coins_average_same_direction"
                    if aligned.all()
                    else "most_coins_average_same_direction"
                    if aligned.mean() >= 0.7
                    else "mixed_coin_averages"
                ),
            }
        )
    return DataFrame.from_records(rows).sort_values(
        ["family", "cohort", "metric", "horizon_band", "cell"]
    )


def candidate_effect_rollup(
    portability: DataFrame,
    effects: DataFrame,
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in portability.itertuples(index=False):
        horizons = g11z.HORIZON_BANDS[str(candidate.horizon_band)]
        periods = g11z.VALIDATION_PERIODS[str(candidate.cohort)]
        checks = effects.loc[
            effects["route"].eq(candidate.route)
            & effects["cohort"].eq(candidate.cohort)
            & effects["cell"].eq(candidate.cell)
            & effects["metric"].eq(candidate.metric)
            & effects["horizon_hours"].isin(horizons)
            & effects["period"].isin(periods)
            & effects["comparison_control"].isin(g11d.CONTROLS)
        ]
        rows.append(
            {
                **candidate._asdict(),
                "checks": len(checks),
                "minimum_events_in_any_side": int(
                    min(checks["events_actual"].min(), checks["events_control"].min())
                ),
                "minimum_coins_in_check": int(checks["coins"].min()),
                "mean_actual_value": float(checks["equal_coin_actual"].mean()),
                "mean_control_value": float(checks["equal_coin_control"].mean()),
                "mean_effect_across_all_checks": float(
                    checks["mean_pair_difference"].mean()
                ),
                "minimum_absolute_effect_across_checks": float(
                    checks["mean_pair_difference"].abs().min()
                ),
                "all_four_controls_both_periods_and_adjacent_horizons": (
                    len(checks) == len(g11d.CONTROLS) * 2 * 2
                ),
            }
        )
    return DataFrame.from_records(rows)


def family_summary(audit: DataFrame) -> DataFrame:
    if audit.empty:
        return DataFrame()
    return (
        audit.groupby("family", observed=True)
        .agg(
            exact_candidate_rows=("cell", "size"),
            cohorts=("cohort", lambda value: ",".join(sorted(set(value)))),
            metrics=("metric", lambda value: ",".join(sorted(set(value)))),
            reaction_rate_rows=("metric", lambda value: int(value.eq("reaction_rate").sum())),
            all_coin_rows=(
                "portability_label",
                lambda value: int(value.eq("all_coins_average_same_direction").sum()),
            ),
            minimum_aligned_coin_fraction=("aligned_coin_fraction", "min"),
            minimum_events=("minimum_events_in_any_side", "min"),
        )
        .reset_index()
        .sort_values(["reaction_rate_rows", "exact_candidate_rows"], ascending=False)
    )


def direction_failure_summary(
    direction_candidates: DataFrame,
    scores: DataFrame,
) -> DataFrame:
    nonbaseline = direction_candidates.loc[
        ~direction_candidates["method"].isin(g11d.BASELINE_METHODS)
    ].copy()
    rows: list[dict[str, Any]] = []
    for cohort in g11z.COHORTS:
        full_scope = (
            "normal_full_cohort" if cohort == "normal" else "meme_frozen_top_ten"
        )
        cell = nonbaseline.loc[nonbaseline["scope_id"].eq(full_scope)]
        best = cell.sort_values(
            ["minimum_period_joint_success", "minimum_comparator_margin"],
            ascending=False,
        ).head(1)
        baseline = direction_candidates.loc[
            direction_candidates["scope_id"].eq(full_scope)
            & direction_candidates["method"].isin(g11d.BASELINE_METHODS)
        ].sort_values("minimum_period_joint_success", ascending=False).head(1)
        score_cell = scores.loc[
            scores["scope_id"].eq(full_scope)
            & ~scores["method"].isin(g11d.BASELINE_METHODS)
        ]
        top_reaction = score_cell.sort_values(
            "reaction_rate_on_calls", ascending=False
        ).head(1)
        rows.append(
            {
                "cohort": cohort,
                "scope_id": full_scope,
                "direction_methods_tested": nonbaseline.loc[
                    nonbaseline["scope_id"].eq(full_scope), "method"
                ].nunique(),
                "point_candidates_at_or_above_55pct": int(
                    cell["point_candidate_55pct"].astype(bool).sum()
                ),
                "strict_candidates": int(cell["strict_candidate"].astype(bool).sum()),
                "best_combination_method": (
                    str(best.iloc[0]["method"]) if len(best) else "none"
                ),
                "best_combination_source_timeframe": (
                    str(best.iloc[0]["source_timeframe"]) if len(best) else "none"
                ),
                "best_combination_horizon_hours": (
                    int(best.iloc[0]["horizon_hours"]) if len(best) else 0
                ),
                "best_combination_minimum_period_joint_rate": (
                    float(best.iloc[0]["minimum_period_joint_success"])
                    if len(best)
                    else np.nan
                ),
                "best_simple_baseline_minimum_period_joint_rate": (
                    float(baseline.iloc[0]["minimum_period_joint_success"])
                    if len(baseline)
                    else np.nan
                ),
                "highest_reaction_rate_seen_on_any_combination_slice": (
                    float(top_reaction.iloc[0]["reaction_rate_on_calls"])
                    if len(top_reaction)
                    else np.nan
                ),
                "plain_conclusion": (
                    "Reactions were common in selected slices, but the tested inputs did "
                    "not identify whether the path would go through or away from the area."
                ),
            }
        )
    return DataFrame.from_records(rows)


def next_freqai_queue() -> DataFrame:
    rows = (
        (
            1,
            "level_identity_and_timeframe",
            "Does knowing that the contacted area is a causal prior-range level, and which timeframe produced it, improve multi-horizon reaction and volume forecasts beyond basic contact geometry?",
            "minimal contact geometry; stale level identity; within-period shuffled level identity",
            "direct lead; strongest cross-coin and cross-cohort reaction family",
        ),
        (
            2,
            "isolated_and_cluster_geometry",
            "Do isolated levels, independent clusters, and opposing overlaps add nonlinear information after level identity and timeframe are already known?",
            "level/timeframe base; stale geometry; shuffled geometry",
            "direct inverse isolated-level relationships; clusters themselves did not yet survive",
        ),
        (
            3,
            "local_trend_and_momentum",
            "Do causal trend strength, slopes, RSI, and MACD improve reaction forecasts at otherwise comparable calculated areas?",
            "level/timeframe base; stale local trend; shuffled local trend",
            "small repeated direct trend-conditioned leads, but no useful simple direction rule",
        ),
        (
            4,
            "local_activity_and_volatility",
            "Can nonlinear volume, pressure, volatility, and compression combinations add information even though the coarse activity-state direct route had no full-control lead?",
            "level/timeframe base; stale local state; shuffled local state",
            "breadth-preserving falsification route and check against earlier failed broad participation claims",
        ),
        (
            5,
            "wider_crypto_market_alignment",
            "Does BTC, ETH, and cohort state improve reaction forecasts when local and wider-market pressure agree or conflict?",
            "level/timeframe base; stale market state; shuffled market state",
            "repeated cross-coin unsigned reaction leads under local-versus-market disagreement",
        ),
        (
            6,
            "external_context_guard",
            "Within genuinely covered timestamps, do news activity or BTC order-book state improve the same reaction forecast beyond OHLCV and level information?",
            "identical covered rows without external values; stale external context; shuffled external context",
            "direct route had no full-control lead, so this remains a bounded nonlinear check rather than a favoured branch",
        ),
        (
            7,
            "low_dimensional_combination",
            "Does combining level identity, geometry, local trend, and wider-market state outperform the level-only model and each leave-one-family-out version?",
            "level/timeframe base; all extras stale; all extras shuffled; leave-one-family-out ablations",
            "tests whether several small effects cooperate without allowing an unrestricted feature soup",
        ),
    )
    return DataFrame.from_records(
        [
            {
                "batch_order": order,
                "question_id": question,
                "plain_question": plain,
                "required_controls": controls,
                "reason_for_inclusion": reason,
                "cohorts": "normal,frozen_top_ten_memes",
                "source_timeframes": "1h,4h,8h,1d",
                "horizons_hours": "1,2,4,8",
                "status": "approved_for_one_complete_initial_freqai_batch",
            }
            for order, question, plain, controls, reason in rows
        ]
    )


def build_review(review_id: str) -> dict[str, Any]:
    direct, frames = load_direct_result()
    portability = candidate_portability(
        frames["neutral_candidates"], frames["pair_condition_medians"]
    )
    audit = candidate_effect_rollup(portability, frames["neutral_effects"])
    families = family_summary(audit)
    direction = direction_failure_summary(
        frames["direction_candidates"], frames["direction_scores"]
    )
    queue = next_freqai_queue()
    review_dir = REVIEW_ROOT / review_id
    review_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "candidate_audit": review_dir / "g11_exact_candidate_audit.csv",
        "family_summary": review_dir / "g11_candidate_family_summary.csv",
        "direction_failure": review_dir / "g11_direction_failure_summary.csv",
        "route_completion": review_dir / "g11_route_completion.csv",
        "next_freqai_queue": review_dir / "g11_next_freqai_queue.csv",
    }
    outputs = {
        "candidate_audit": audit,
        "family_summary": families,
        "direction_failure": direction,
        "route_completion": frames["route_decisions"],
        "next_freqai_queue": queue,
    }
    for key, path in paths.items():
        g0.atomic_write_csv(outputs[key], path)

    reaction = audit.loc[audit["metric"].eq("reaction_rate")]
    generic = reaction.loc[
        reaction["family"].eq("generic_prior_range_across_timeframes")
    ]
    result_path = review_dir / "g11_joint_review.json"
    result = {
        "schema_version": 1,
        "review_id": review_id,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation11_joint_direct_review_and_freqai_queue",
        "sequence": {
            "all_seven_direct_routes_completed_before_review": True,
            "result_inspired_questions_queued_after_joint_review": True,
            "no_descendant_outcomes_opened_during_review": True,
        },
        "integrity": {
            "binary_reaction_summarized_as_per_coin_mean_rate": True,
            "continuous_outcomes_summarized_as_per_coin_medians": True,
            "equal_coin_weighting": True,
            "per_coin_audit_preserved": True,
            "all_four_location_controls_required": True,
            "both_validation_periods_required": True,
            "adjacent_horizons_required": True,
            "profit_used": False,
        },
        "direct_summary": {
            "routes_completed": direct["summary"]["routes_completed"],
            "exact_direction_neutral_candidates": len(audit),
            "reaction_rate_candidates": len(reaction),
            "candidate_families": len(families),
            "all_coin_average_direction_candidate_rows": int(
                audit["portability_label"]
                .eq("all_coins_average_same_direction")
                .sum()
            ),
            "direction_methods": len(g11z.DIRECTION_METHODS),
            "direction_point_leads_at_or_above_55pct": int(
                frames["direction_candidates"]["point_candidate_55pct"]
                .astype(bool)
                .sum()
            ),
            "direction_strict_leads": int(
                frames["direction_candidates"]["strict_candidate"].astype(bool).sum()
            ),
        },
        "plain_findings": {
            "strongest_reaction_family": (
                "Causal prior-range contacts repeatedly showed more price-plus-volume "
                "reaction than ordinary-time, shifted-price, stale-level, and near-miss "
                "contacts across normal coins and memes."
            ),
            "generic_prior_range_exact_reaction_rows": len(generic),
            "generic_prior_range_actual_rate_range": (
                [
                    float(generic["mean_actual_value"].min()),
                    float(generic["mean_actual_value"].max()),
                ]
                if len(generic)
                else []
            ),
            "generic_prior_range_control_gain_range": (
                [
                    float(generic["mean_effect_across_all_checks"].min()),
                    float(generic["mean_effect_across_all_checks"].max()),
                ]
                if len(generic)
                else []
            ),
            "cluster_result": (
                "Only isolated-level states produced repeated geometry rows. No cluster "
                "composition itself survived the complete frozen control ladder."
            ),
            "activity_and_external_result": (
                "Coarse activity-state and external-context routes produced no repeated "
                "full-control direct candidates. They remain bounded nonlinear checks, "
                "not favoured explanations."
            ),
            "direction_result": (
                "None of the simple local, market, consensus, activity, or obstacle rules "
                "reached 55% joint reaction-and-correct-path success in both validation "
                "periods. Common reactions therefore do not yet imply predictable direction."
            ),
        },
        "evidence_boundary": (
            "The combinations were frozen before this direct screen, but the underlying "
            "historical periods were used by earlier research. Every retained relationship "
            "is exploratory and requires chronological confirmation on genuinely later data."
        ),
        "next_batch": {
            "type": "one_complete_breadth_first_freqai_attribution_batch",
            "questions": len(queue),
            "cohorts": ["normal", "frozen_top_ten_memes"],
            "source_timeframes": list(g11z.SOURCE_TIMEFRAMES),
            "horizons_hours": list(g11z.HORIZONS),
            "primary_targets": [
                "multi-horizon reaction probability",
                "multi-horizon relative volume",
            ],
            "direction_prediction": False,
            "descendants_wait_for_whole_batch_review": True,
        },
        "source_contracts": {
            "generation11_direct_result": artifact(DIRECT_RESULT),
            "generation11_freeze": artifact(g11z.FREEZE_PATH),
        },
        "artifacts": {key: artifact(path) for key, path in paths.items()},
        "next_action": (
            "Freeze the seven-question FreqAI registry and its controls before model "
            "outcomes, then complete both cohorts before selecting confirmations."
        ),
    }
    g0.atomic_write_json(result, result_path)
    return {**result, "result_path": str(result_path.resolve())}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jointly review Generation 11 direct routes.")
    parser.add_argument("--review-id", default=DEFAULT_REVIEW_ID)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_review(args.review_id)
    print(json.dumps(result["direct_summary"], indent=2, sort_keys=True), flush=True)
    print(result["result_path"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

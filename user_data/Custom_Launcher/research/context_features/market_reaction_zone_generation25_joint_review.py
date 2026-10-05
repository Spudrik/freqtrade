"""Jointly review the single frozen Generation 25 sibling batch."""

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
    market_reaction_zone_generation25_connected_volume_profile as g25v,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation25_convergence_representation as g25r,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation25_freeze as g25z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation25_market_state_activity as g25a,
)


ANALYSIS_PATH = Path(__file__).resolve()
DEFAULT_RUN_ID = "g25_joint_review_20260829a"
REVIEW_ROOT = g25z.OUTPUT_ROOT / "joint_review"


def artifact(path: Path) -> dict[str, Any]:
    return g25z.artifact(path)


def result_paths() -> dict[str, Path]:
    return {
        "convergence": (
            g25r.RECORD_ROOT
            / g25r.DEFAULT_RUN_ID
            / "g25_convergence_representation_result.json"
        ),
        "connected_volume_profile": (
            g25v.RECORD_ROOT
            / g25v.DEFAULT_RUN_ID
            / "g25_connected_volume_profile_result.json"
        ),
        "market_state_activity": (
            g25a.RECORD_ROOT
            / g25a.DEFAULT_RUN_ID
            / "g25_market_state_activity_result.json"
        ),
        "coverage": g25z.COVERAGE_RESULT,
    }


def load_terminal_results() -> dict[str, dict[str, Any]]:
    expected = {
        "convergence": "completed_generation25_convergence_representation",
        "connected_volume_profile": "completed_generation25_connected_volume_profile",
        "market_state_activity": "completed_generation25_market_state_activity",
        "coverage": "completed_generation25_outcome_blind_coverage_gate",
    }
    output: dict[str, dict[str, Any]] = {}
    for name, path in result_paths().items():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != expected[name]:
            raise ValueError(f"Generation 25 {name} is not terminal.")
        output[name] = result
    return output


def _read_artifact(result: dict[str, Any], name: str) -> DataFrame:
    item = result["artifacts"][name]
    path = Path(item["path"])
    if g0.sha256_file(path) != item["sha256"]:
        raise ValueError(f"Generation 25 result artifact changed: {path}")
    return pd.read_csv(path)


def _closest_direct_question(scores: DataFrame) -> dict[str, Any]:
    keys = ["scope_value", "metric", "horizon_hours", "market_scope"]
    grouped = (
        scores.groupby(keys, observed=True)
        .agg(
            comparison_period_rows=("point_period_pass", "size"),
            point_passes=("point_period_pass", "sum"),
            strict_passes=("strict_period_pass", "sum"),
            minimum_effect=("equal_coin_difference", "min"),
            mean_effect=("equal_coin_difference", "mean"),
        )
        .reset_index()
        .sort_values(
            ["point_passes", "strict_passes", "minimum_effect"],
            ascending=False,
            kind="stable",
        )
    )
    row = grouped.iloc[0]
    return {
        "scope_value": str(row["scope_value"]),
        "metric": str(row["metric"]),
        "horizon_hours": int(row["horizon_hours"]),
        "market_scope": str(row["market_scope"]),
        "comparison_period_rows": int(row["comparison_period_rows"]),
        "point_passes": int(row["point_passes"]),
        "strict_passes": int(row["strict_passes"]),
        "minimum_effect": float(row["minimum_effect"]),
        "mean_effect": float(row["mean_effect"]),
    }


def direct_summary(result: dict[str, Any]) -> dict[str, Any]:
    decisions = _read_artifact(result, "decisions")
    scores = _read_artifact(result, "scores")
    return {
        "decision_rows": len(decisions),
        "complete_control_period_ladders": int(
            decisions["complete_control_period_ladder"].astype(bool).sum()
        ),
        "strict_rows": int(decisions["status"].eq("strict_holdout_confirmation").sum()),
        "point_rows": int(decisions["status"].eq("point_holdout_confirmation").sum()),
        "closest_question": _closest_direct_question(scores),
        "control_point_period_pass_rates": {
            str(control): float(cell["point_period_pass"].mean())
            for control, cell in scores.groupby("comparison", observed=True)
        },
    }


def market_state_summary(result: dict[str, Any]) -> dict[str, Any]:
    decisions = _read_artifact(result, "decisions")
    scopes = _read_artifact(result, "scope_scores")
    comparisons = _read_artifact(result, "control_comparisons")
    retained = decisions.loc[decisions["all_model_seed_cells_pass"].astype(bool)].copy()
    retained_keys = retained[["cohort", "market_scope", "target"]]
    candidate = scopes.loc[scopes["role"].eq(g25a.CANDIDATE_ROLE)].merge(
        retained_keys,
        on=["cohort", "market_scope", "target"],
        how="inner",
        validate="many_to_one",
    )
    by_cohort = retained.groupby("cohort", observed=True).size().to_dict()
    target_ranges: dict[str, Any] = {}
    for cohort, cell in candidate.groupby("cohort", observed=True):
        target_ranges[str(cohort)] = {
            "minimum_spearman": float(cell["spearman_rank_correlation"].min()),
            "maximum_spearman": float(cell["spearman_rank_correlation"].max()),
            "minimum_top_minus_bottom": float(cell["top_minus_bottom_difference"].min()),
            "maximum_top_minus_bottom": float(cell["top_minus_bottom_difference"].max()),
            "minimum_top_quartile_above_median_fraction": float(
                cell["top_quartile_above_training_median_fraction"].min()
            ),
            "maximum_top_quartile_above_median_fraction": float(
                cell["top_quartile_above_training_median_fraction"].max()
            ),
        }
    return {
        "decision_rows": len(decisions),
        "retained_scope_target_rows": len(retained),
        "retained_by_cohort": {str(key): int(value) for key, value in by_cohort.items()},
        "retained_targets": retained.to_dict("records"),
        "candidate_metric_ranges_on_retained_rows": target_ranges,
        "control_period_pass_rates": {
            f"{cohort}__{control}": float(cell["period_control_pass"].mean())
            for (cohort, control), cell in comparisons.groupby(
                ["cohort", "control"], observed=True
            )
        },
    }


def review(run_id: str, *, overwrite: bool = False) -> dict[str, Any]:
    run_dir = REVIEW_ROOT / run_id
    result_path = run_dir / "g25_joint_review.json"
    if result_path.is_file() and not overwrite:
        return json.loads(result_path.read_text(encoding="utf-8"))
    terminal = load_terminal_results()
    convergence = direct_summary(terminal["convergence"])
    volume_profile = direct_summary(terminal["connected_volume_profile"])
    state = market_state_summary(terminal["market_state_activity"])
    coverage = terminal["coverage"]
    result = {
        "schema_version": 1,
        "generation": 25,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation25_joint_review_one_batch_only",
        "batch_execution_limit": {
            "generation25_batches_run": 1,
            "descendant_batch_launched": False,
            "all_frozen_siblings_reviewed_together": True,
        },
        "route_results": {
            "convergence_control_representation": convergence,
            "connected_volume_profile": volume_profile,
            "market_state_activity": state,
            "daily_vwap_fresh_confirmation": coverage["branch_gates"][
                "g25a_daily_anchored_vwap_fresh_confirmation"
            ],
            "external_source_overlap": coverage["branch_gates"][
                "g25e_external_source_overlap_gate"
            ],
        },
        "plain_conclusions": [
            (
                "No new price-level or Volume Profile zone representation passed its whole "
                "two-period control ladder. Neither can yet be called a repeatable reaction zone."
            ),
            (
                "The pooled round-number plus price-distribution family showed a coherent but "
                "incomplete meme-coin two-hour crossing lead: 13 of 16 period/control checks "
                "passed, so it is a lead for later fresh confirmation, not a retained result."
            ),
            (
                "Connected Volume Profile areas did not consistently beat one-bin nodes, stale "
                "zones, shifted zones, near misses, and random locations. Keep this definition "
                "parked."
            ),
            (
                "FreqAI market-state-only models retained 15 of 40 scope/target questions against "
                "constant, shuffled-label, simple-current-activity, and stale-activity baselines. "
                "All eight unsigned meme activity horizons survived all four model/seed cells; "
                "normal coins retained a smaller subset."
            ),
            (
                "The FreqAI result predicts when volume or price range is likely to be unusually "
                "active. It does not predict direction, prove a reaction to a level, or justify "
                "a trade."
            ),
        ],
        "next_questions_not_launched": [
            {
                "priority": 1,
                "question": (
                    "Does the fixed market-state-only activity model remain useful on genuinely "
                    "later normal and meme data?"
                ),
                "status": "queued_for_future_fresh_coverage_not_launched",
            },
            {
                "priority": 2,
                "question": (
                    "Does the fixed pooled convergence representation repeat its two-hour "
                    "meme crossing pattern on fresh chronology?"
                ),
                "status": "queued_as_incomplete_lead_not_launched",
            },
            {
                "priority": 3,
                "question": (
                    "Can the already-selected daily anchored VWAP question be tested after two "
                    "complete fresh blocks exist?"
                ),
                "status": "parked_until_coverage_gate_passes",
            },
        ],
        "parked_after_joint_review": [
            "connected_volume_profile_current_definition",
            "new_external_source_modelling_without_two_clean_blocks",
            "one_minute_direction_replay_without_a_confirmed_reaction_parent",
            "trading_or_profit_promotion",
        ],
        "research_boundary": {
            "profit_used": False,
            "signed_direction_used": False,
            "activity_success_not_reaction_and_direction_success": True,
            "same_holdout_activity_findings_require_fresh_confirmation": True,
            "canonical_indicators_edited": False,
        },
        "source_contracts": {
            name: artifact(path) for name, path in result_paths().items()
        }
        | {
            "generation25_freeze": artifact(g25z.FREEZE_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
        "result_path": str(result_path.resolve()),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(result, result_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(review(args.run_id, overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

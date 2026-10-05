"""Jointly review all seven frozen Generation 15 siblings after their terminal runs."""

from __future__ import annotations

# Bind native pools before pandas/numpy imports.
# ruff: noqa: E402
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
    market_reaction_zone_generation15_anchor_contexts as g15c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation15_freeze as g15z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation15_matched_paths as g15m,
)


DEFAULT_RUN_ID = "g15_joint_review_20260822a"
MATCH_DIR = g15m.RECORD_ROOT / g15m.DEFAULT_RUN_ID
CONTEXT_DIR = g15c.RECORD_ROOT / g15c.DEFAULT_RUN_ID
REVIEW_ROOT = g15m.RECORD_ROOT.parent / "joint_review"
MATCH_RESULT = MATCH_DIR / "g15_matched_paths_result.json"
CONTEXT_RESULT = CONTEXT_DIR / "g15_anchor_contexts_result.json"
GROUP_MEMBERS = {
    **{key: tuple(value) for key, value in g15z.NORMAL_GROUPS.items()},
    "frozen_top_ten_memes": (),
}
GROUP_MIN_COINS = {
    "btc_separate": 1,
    "smart_contract_platforms": 5,
    "other_established_alts": 4,
    "frozen_top_ten_memes": 5,
}


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def validate_artifacts(result: dict[str, Any]) -> None:
    for label, record in result["artifacts"].items():
        path = Path(record["path"])
        if not path.is_file() or g0.sha256_file(path) != record["sha256"]:
            raise ValueError(f"Generation 15 artifact changed: {label}: {path}")


def load_terminal_results() -> tuple[dict[str, Any], dict[str, Any]]:
    if not MATCH_RESULT.is_file() or not CONTEXT_RESULT.is_file():
        raise FileNotFoundError("Both Generation 15 sibling result records are required.")
    matched = json.loads(MATCH_RESULT.read_text(encoding="utf-8"))
    context = json.loads(CONTEXT_RESULT.read_text(encoding="utf-8"))
    if matched.get("status") != "completed_generation15_matched_paths":
        raise ValueError("The three matched-path siblings are not terminal.")
    if context.get("status") != "completed_generation15_anchor_contexts":
        raise ValueError("The four anchor-context siblings are not terminal.")
    expected = {
        "reaction_path_decomposition",
        "level_family_attribution",
        "source_timeframe_attribution",
        "contact_activity_combinations",
        "contact_lifecycle_and_approach",
        "explicit_cluster_composition",
        "historical_btc_orderbook_conditioning",
    }
    completed = set(matched["routes_completed"]) | set(context["routes_completed"])
    if completed != expected:
        raise ValueError(f"Generation 15 sibling set is incomplete: {expected - completed}")
    validate_artifacts(matched)
    validate_artifacts(context)
    return matched, context


def matched_score_dominance(scores: DataFrame, pair_summary: DataFrame) -> DataFrame:
    keys = [
        "route_id",
        "scope_value",
        "metric",
        "horizon_hours",
        "cohort",
        "window",
        "analysis_period",
        "control",
    ]
    rows: list[dict[str, Any]] = []
    for key, cell in pair_summary.groupby(keys, observed=True, sort=False):
        eligible = cell.loc[cell["independent_rows"].ge(g15m.MIN_PAIR_ROWS)]
        absolute = eligible["paired_mean_difference"].abs()
        total = float(absolute.sum())
        share = float(absolute.max() / total) if total > 0.0 else np.nan
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "maximum_absolute_coin_share": share,
                "not_dominated_by_one_coin": bool(
                    len(eligible)
                    and total > 0.0
                    and share <= g15c.MAX_COIN_ABSOLUTE_SHARE
                ),
            }
        )
    dominance = DataFrame.from_records(rows)
    return scores.merge(dominance, on=keys, how="left", validate="one_to_one")


def matched_cells_with_dominance(
    cells: DataFrame, scores: DataFrame
) -> DataFrame:
    keys = [
        "route_id",
        "scope_value",
        "metric",
        "horizon_hours",
        "cohort",
        "window",
    ]
    rows: list[dict[str, Any]] = []
    for row in cells.itertuples(index=False):
        key = tuple(getattr(row, column) for column in keys)
        support = scores.loc[
            scores["route_id"].eq(key[0])
            & scores["scope_value"].eq(key[1])
            & scores["metric"].eq(key[2])
            & scores["horizon_hours"].eq(key[3])
            & scores["cohort"].eq(key[4])
            & scores["window"].eq(key[5])
            & scores["eligible_coins"].ge(g15m.MIN_COINS)
        ]
        dominance = bool(
            len(support) == row.expected_period_control_cells
            and support["not_dominated_by_one_coin"].all()
        )
        record = row._asdict()
        record["not_dominated_by_one_coin"] = dominance
        record["strict_pass_with_dominance"] = bool(row.strict_pass and dominance)
        rows.append(record)
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def matched_joint_with_dominance(cells: DataFrame) -> DataFrame:
    keys = ["route_id", "scope_value", "metric", "horizon_hours"]
    expected = {
        ("normal", "standard_validation"),
        ("normal", "recent_normal_chronology"),
        ("meme", "standard_validation"),
    }
    rows: list[dict[str, Any]] = []
    for key, cell in cells.groupby(keys, observed=True, sort=False):
        selected = cell.loc[
            [
                (cohort, window) in expected
                for cohort, window in zip(cell["cohort"], cell["window"], strict=True)
            ]
        ]
        available = set(zip(selected["cohort"], selected["window"], strict=True))
        complete = expected.issubset(available)
        point_rows = selected.loc[selected["point_pass"]]
        signs = set(point_rows["consistent_effect_sign"])
        same_sign = len(signs) == 1 and len(point_rows) == len(expected)
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "complete_three_cell_ladder": complete,
                "same_sign_across_three_cells": same_sign,
                "consistent_effect_sign": next(iter(signs)) if same_sign else "mixed",
                "point_all_three_cells": bool(
                    complete and same_sign and selected["point_pass"].all()
                ),
                "strict_all_three_cells_with_dominance": bool(
                    complete
                    and same_sign
                    and selected["strict_pass_with_dominance"].all()
                ),
                "point_cells": int(selected["point_pass"].sum()),
                "strict_cells_with_dominance": int(
                    selected["strict_pass_with_dominance"].sum()
                ),
            }
        )
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def matched_lead_summary(joint: DataFrame, scores: DataFrame) -> DataFrame:
    keys = ["route_id", "scope_value", "metric", "horizon_hours"]
    rows: list[dict[str, Any]] = []
    for lead in joint.loc[joint["point_all_three_cells"]].itertuples(index=False):
        key = tuple(getattr(lead, column) for column in keys)
        support = scores.loc[
            scores["route_id"].eq(key[0])
            & scores["scope_value"].eq(key[1])
            & scores["metric"].eq(key[2])
            & scores["horizon_hours"].eq(key[3])
            & scores["eligible_coins"].ge(g15m.MIN_COINS)
        ]
        relative = support["equal_coin_paired_difference"].div(
            support["equal_coin_control_mean"].abs().replace(0.0, np.nan)
        )
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "effect_sign": lead.consistent_effect_sign,
                "strict_with_dominance": bool(
                    lead.strict_all_three_cells_with_dominance
                ),
                "support_rows": len(support),
                "minimum_eligible_coins": int(support["eligible_coins"].min()),
                "minimum_independent_rows": int(support["independent_rows"].min()),
                "minimum_equal_coin_difference": float(
                    support["equal_coin_paired_difference"].min()
                ),
                "maximum_equal_coin_difference": float(
                    support["equal_coin_paired_difference"].max()
                ),
                "minimum_relative_difference": float(relative.min()),
                "maximum_relative_difference": float(relative.max()),
                "maximum_absolute_coin_share": float(
                    support["maximum_absolute_coin_share"].max()
                ),
            }
        )
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def group_for_pair(cohort: str, pair: str) -> str:
    if cohort == "meme":
        return "frozen_top_ten_memes"
    for group, members in g15z.NORMAL_GROUPS.items():
        if pair in members:
            return group
    raise KeyError((cohort, pair))


def point_group_scores(pair_summary: DataFrame, comparison: str) -> DataFrame:
    source = pair_summary.copy()
    source["analysis_group"] = [
        group_for_pair(cohort, pair)
        for cohort, pair in zip(source["cohort"], source["pair"], strict=True)
    ]
    keys = [
        "route_id",
        "scope_value",
        comparison,
        "metric",
        "horizon_hours",
        "analysis_group",
        "window",
        "analysis_period",
    ]
    rows: list[dict[str, Any]] = []
    for key, cell in source.groupby(keys, observed=True, sort=False):
        eligible = cell.loc[cell["independent_rows"].ge(g15m.MIN_PAIR_ROWS)]
        required = GROUP_MIN_COINS[key[5]]
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "required_group_coins": required,
                "eligible_coins": len(eligible),
                "complete_group_support": len(eligible) >= required,
                "independent_rows": int(eligible["independent_rows"].sum()),
                "equal_coin_paired_difference": float(
                    eligible["paired_mean_difference"].mean()
                )
                if len(eligible)
                else np.nan,
            }
        )
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def group_periods(group: str, window: str, *, orderbook: bool = False) -> tuple[str, ...]:
    if group == "frozen_top_ten_memes":
        return ("meme_validation_early", "meme_validation_late")
    if window == "recent_normal_chronology":
        periods = ("recent_confirmation_early", "recent_confirmation_late")
        return periods[:1] if orderbook else periods
    return ("validation_early", "validation_late")


def matched_group_cells(scores: DataFrame) -> DataFrame:
    keys = [
        "route_id",
        "scope_value",
        "metric",
        "horizon_hours",
        "analysis_group",
        "window",
    ]
    rows: list[dict[str, Any]] = []
    for key, cell in scores.groupby(keys, observed=True, sort=False):
        periods = group_periods(key[4], key[5])
        expected = {
            (period, control) for period in periods for control in g15m.CONTROLS
        }
        supported = cell.loc[cell["complete_group_support"]]
        lookup = set(zip(supported["analysis_period"], supported["control"], strict=True))
        effects = supported.loc[
            [
                (period, control) in expected
                for period, control in zip(
                    supported["analysis_period"], supported["control"], strict=True
                )
            ],
            "equal_coin_paired_difference",
        ]
        positive = (
            expected.issubset(lookup)
            and len(effects) == len(expected)
            and effects.gt(0).all()
        )
        negative = (
            expected.issubset(lookup)
            and len(effects) == len(expected)
            and effects.lt(0).all()
        )
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "effect_sign": "positive" if positive else "negative" if negative else "mixed",
                "point_pass": bool(positive or negative),
                "supported_cells": len(expected.intersection(lookup)),
                "expected_cells": len(expected),
            }
        )
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def context_group_cells(scores: DataFrame) -> DataFrame:
    keys = [
        "route_id",
        "scope_value",
        "metric",
        "horizon_hours",
        "analysis_group",
        "window",
    ]
    rows: list[dict[str, Any]] = []
    for key, cell in scores.groupby(keys, observed=True, sort=False):
        orderbook = key[0] == "historical_btc_orderbook_conditioning"
        periods = group_periods(key[4], key[5], orderbook=orderbook)
        primary, placebos = g15c.comparison_contract(key[0], key[1])
        expected_primary = {
            (period, comparison) for period in periods for comparison in primary
        }
        expected_placebo = {
            (period, comparison) for period in periods for comparison in placebos
        }
        supported = cell.loc[cell["complete_group_support"]].copy()
        if orderbook:
            supported = supported.loc[supported["independent_rows"].ge(50)]
        lookup = {
            (row.analysis_period, row.comparison): row
            for row in supported.itertuples(index=False)
        }
        primary_rows = [lookup[item] for item in expected_primary if item in lookup]
        effects = np.asarray(
            [float(row.equal_coin_paired_difference) for row in primary_rows], dtype=float
        )
        complete_primary = expected_primary.issubset(set(lookup))
        positive = (
            complete_primary
            and len(effects) == len(expected_primary)
            and np.all(effects > 0)
        )
        negative = (
            complete_primary
            and len(effects) == len(expected_primary)
            and np.all(effects < 0)
        )
        placebo_beaten = expected_placebo.issubset(set(lookup))
        if not placebos:
            placebo_beaten = True
        else:
            for period in periods:
                current = float(lookup[(period, primary[0])].equal_coin_paired_difference) if (
                    period,
                    primary[0],
                ) in lookup else np.nan
                for comparison in placebos:
                    placebo = float(lookup[(period, comparison)].equal_coin_paired_difference) if (
                        period,
                        comparison,
                    ) in lookup else np.nan
                    placebo_beaten = bool(
                        placebo_beaten
                        and np.isfinite(current)
                        and np.isfinite(placebo)
                        and abs(current) > abs(placebo)
                    )
        attribution = not (
            (
                key[0] == "explicit_cluster_composition"
                and str(key[1]).endswith("__any_presence")
            )
            or str(key[1]).endswith("__all_contacts_mean_descriptive")
        )
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "effect_sign": "positive" if positive else "negative" if negative else "mixed",
                "placebo_magnitude_beaten_every_period": placebo_beaten,
                "attribution_eligible": attribution,
                "point_pass": bool((positive or negative) and placebo_beaten),
                "supported_primary_cells": len(primary_rows),
                "expected_primary_cells": len(expected_primary),
            }
        )
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def cross_window_group_leads(cells: DataFrame) -> DataFrame:
    keys = ["route_id", "scope_value", "metric", "horizon_hours", "analysis_group"]
    rows: list[dict[str, Any]] = []
    for key, cell in cells.groupby(keys, observed=True, sort=False):
        expected_windows = (
            {"standard_validation"}
            if key[4] == "frozen_top_ten_memes"
            else {"standard_validation", "recent_normal_chronology"}
        )
        selected = cell.loc[cell["window"].isin(expected_windows)]
        passed = selected.loc[selected["point_pass"]]
        signs = set(passed["effect_sign"])
        complete = expected_windows.issubset(set(selected["window"]))
        same_sign = len(signs) == 1 and len(passed) == len(expected_windows)
        attribution = (
            bool(selected["attribution_eligible"].all())
            if "attribution_eligible" in selected and len(selected)
            else True
        )
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "complete_required_windows": complete,
                "same_sign": same_sign,
                "effect_sign": next(iter(signs)) if same_sign else "mixed",
                "attribution_eligible": attribution,
                "point_all_required_windows": bool(
                    complete and same_sign and selected["point_pass"].all()
                ),
                "windows_passed": int(selected["point_pass"].sum()),
                "windows_required": len(expected_windows),
            }
        )
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    matched_result, context_result = load_terminal_results()
    run_dir = REVIEW_ROOT / args.run_id
    result_path = run_dir / "g15_joint_review.json"

    matched_pairs = pd.read_csv(MATCH_DIR / "pair_path_differences.csv")
    matched_scores = pd.read_csv(MATCH_DIR / "equal_coin_path_scores.csv")
    matched_cells = pd.read_csv(MATCH_DIR / "path_cell_decisions.csv")
    context_pairs = pd.read_csv(CONTEXT_DIR / "pair_context_differences.csv")
    context_cells = pd.read_csv(CONTEXT_DIR / "context_cell_decisions.csv")
    context_joint = pd.read_csv(CONTEXT_DIR / "context_joint_decisions.csv")

    matched_scores = matched_score_dominance(matched_scores, matched_pairs)
    matched_cells = matched_cells_with_dominance(matched_cells, matched_scores)
    matched_joint = matched_joint_with_dominance(matched_cells)
    matched_leads = matched_lead_summary(matched_joint, matched_scores)

    matched_group_scores = point_group_scores(matched_pairs, "control")
    matched_group_cell_rows = matched_group_cells(matched_group_scores)
    matched_group_leads = cross_window_group_leads(matched_group_cell_rows)
    context_group_scores = point_group_scores(context_pairs, "comparison")
    context_group_cell_rows = context_group_cells(context_group_scores)
    context_group_leads = cross_window_group_leads(context_group_cell_rows)

    matched_scores_path = run_dir / "matched_scores_with_dominance.csv"
    matched_cells_path = run_dir / "matched_cells_with_dominance.csv"
    matched_joint_path = run_dir / "matched_joint_with_dominance.csv"
    matched_leads_path = run_dir / "matched_level_leads.csv"
    matched_group_path = run_dir / "matched_group_leads.csv"
    context_group_path = run_dir / "context_group_leads.csv"
    context_cell_leads_path = run_dir / "context_cell_leads.csv"
    g0.atomic_write_csv(matched_scores, matched_scores_path)
    g0.atomic_write_csv(matched_cells, matched_cells_path)
    g0.atomic_write_csv(matched_joint, matched_joint_path)
    g0.atomic_write_csv(matched_leads, matched_leads_path)
    g0.atomic_write_csv(matched_group_leads, matched_group_path)
    g0.atomic_write_csv(context_group_leads, context_group_path)
    g0.atomic_write_csv(
        context_cells.loc[context_cells["point_pass"]].reset_index(drop=True),
        context_cell_leads_path,
    )

    general = matched_leads.loc[
        matched_leads["route_id"].eq("reaction_path_decomposition")
    ]
    context_portable = context_joint.loc[context_joint["point_all_three_cells"]]
    context_group_portable = context_group_leads.loc[
        context_group_leads["point_all_required_windows"]
        & context_group_leads["attribution_eligible"]
    ]
    matched_group_portable = matched_group_leads.loc[
        matched_group_leads["point_all_required_windows"]
    ]
    result = {
        "schema_version": 1,
        "generation": 15,
        "run_id": args.run_id,
        "status": "completed_generation15_joint_review",
        "all_frozen_siblings_completed_or_parked_before_review": True,
        "source_contracts": {
            "generation15_freeze": artifact(g15z.FREEZE_PATH),
            "matched_paths_result": artifact(MATCH_RESULT),
            "anchor_contexts_result": artifact(CONTEXT_RESULT),
        },
        "integrity": {
            "sibling_routes_reviewed": 7,
            "matched_pair_failures": matched_result["integrity"]["pair_failures"],
            "context_pair_failures": context_result["integrity"]["pair_failures"],
            "future_signed_direction_used": False,
            "profit_used": False,
            "strict_one_coin_dominance_applied": True,
            "maximum_absolute_one_coin_share": g15c.MAX_COIN_ABSOLUTE_SHARE,
        },
        "findings": {
            "general_contact_path_point_leads": len(general),
            "general_contact_path_strict_leads": int(general["strict_with_dominance"].sum()),
            "general_contact_path_metrics": sorted(general["metric"].unique()),
            "all_level_family_timeframe_point_leads": len(matched_leads),
            "all_level_family_timeframe_strict_leads": int(
                matched_leads["strict_with_dominance"].sum()
            ),
            "context_full_three_cell_point_leads": len(context_portable),
            "context_attribution_eligible_group_leads": len(context_group_portable),
            "matched_portable_group_leads": len(matched_group_portable),
            "context_best_broad_cell_count": int(context_joint["point_cells"].max()),
            "context_interpretation": (
                "No activity combination, lifecycle/arrival state, attribution-eligible "
                "cluster composition, or source-limited order-book condition survived the "
                "full declared portability ladder."
            ),
            "direction_or_joint_55_percent_target": "not_tested",
        },
        "research_decision": {
            "retain": (
                "Calculated-area contact is a repeatable locator of subsequent volume, "
                "range, and level-crossing activity; family and timeframe differences "
                "remain reaction-type hypotheses, not directional signals."
            ),
            "do_not_claim": [
                "direction prediction",
                "profitability",
                "a universal contextual combination",
                "causal cluster value from overlapping any-presence flags",
                "order-book confirmation outside its timestamp-safe coverage",
            ],
            "generation16_automatic_launch": False,
            "next_batch_must_be_frozen_from_this_joint_review": True,
        },
        "artifacts": {
            "matched_scores_with_dominance": artifact(matched_scores_path),
            "matched_cells_with_dominance": artifact(matched_cells_path),
            "matched_joint_with_dominance": artifact(matched_joint_path),
            "matched_level_leads": artifact(matched_leads_path),
            "matched_group_leads": artifact(matched_group_path),
            "context_group_leads": artifact(context_group_path),
            "context_cell_leads": artifact(context_cell_leads_path),
        },
    }
    g0.atomic_write_json(result, result_path)
    print(json.dumps(result["findings"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

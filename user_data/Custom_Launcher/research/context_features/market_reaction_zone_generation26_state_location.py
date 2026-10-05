"""Test whether a frozen activity forecast and exact level contacts complement each other."""

from __future__ import annotations

# Bound numerical pools before pandas/numpy imports.
# ruff: noqa: E402
import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
):
    os.environ[_name] = "1"

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
    market_reaction_zone_generation17_level_source_atlas as g17l,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation18_direct_confirmation as g18d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation25_freeze as g25z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation25_joint_review as g25j,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation25_market_state_activity as g25a,
)


ANALYSIS_PATH = Path(__file__).resolve()
GENERATION = 26
DEFAULT_RUN_ID = "g26_state_location_interaction_20260901a"
OUTPUT_ROOT = g25z.OUTPUT_ROOT.parent / "g26_broad_siblings"
RECORD_ROOT = OUTPUT_ROOT / "state_location_interaction"
FREEZE_PATH = OUTPUT_ROOT / "g26_state_location_interaction_freeze_20260901a.json"
SUPPORT_PATH = RECORD_ROOT / "g26_state_location_support.csv"
PARENT_REVIEW = g25j.REVIEW_ROOT / g25j.DEFAULT_RUN_ID / "g25_joint_review.json"
G17_RESULT = (
    g17l.RECORD_ROOT
    / g17l.DEFAULT_RUN_ID
    / "g17_level_source_atlas_result.json"
)
G17_INVENTORY = (
    g17l.RECORD_ROOT / g17l.DEFAULT_RUN_ID / "pair_inventory.csv"
)

COHORTS = ("normal", "meme")
ALL_LEVEL_FAMILIES = (
    "adaptive_volume_profile_nodes",
    "rolling_vwap_deviation_bands",
    "donchian_boundaries",
    "weekly_pivot_grid",
    "generic_ma_bollinger_negative_control",
)
EXPECTED_SUPPORTED_FAMILIES = (
    "adaptive_volume_profile_nodes",
    "rolling_vwap_deviation_bands",
)
CONTROLS = ("matched_random_time", "near_miss", "stale_72h", "price_shift")
STATES = ("high", "low")
STATE_TARGETS = (
    "&-g24_percentile_future_volume_ratio_h2",
    "&-g24_percentile_future_range_ratio_h4",
)
HORIZONS = (1, 2, 4)
METRICS = ("volume_ratio", "range_ratio", "crossings", "unsigned_reaction")
MIN_PAIR_ROWS = 10
MIN_BROAD_SCOPE_PAIRS = 5


def artifact(path: Path) -> dict[str, Any]:
    return g25z.artifact(path)


def _validate_artifact(contract: dict[str, Any]) -> None:
    path = Path(contract["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if g0.sha256_file(path) != contract["sha256"]:
        raise ValueError(f"Frozen source changed: {path}")


def _parent_validation() -> None:
    parent = json.loads(PARENT_REVIEW.read_text(encoding="utf-8"))
    if parent.get("status") != "completed_generation25_joint_review_one_batch_only":
        raise ValueError("Generation 25 joint review is not terminal.")
    activity = parent["route_results"]["market_state_activity"]
    if int(activity["retained_scope_target_rows"]) != 15:
        raise ValueError("Generation 25 activity parent changed.")
    if not parent["research_boundary"]["same_holdout_activity_findings_require_fresh_confirmation"]:
        raise ValueError("Generation 25 exploratory boundary was lost.")


def ensemble_activity_state(cohort: str) -> DataFrame:
    """Build one frozen activity score without reading future target values."""
    parent = g25a.load_parent_manifest(cohort)
    candidate, _, _ = g25a._prediction_lookup(parent)
    parts: list[DataFrame] = []
    for (model_class, seed), frame in candidate.items():
        selected = frame[["date", "pair", *STATE_TARGETS]].copy()
        selected["model_class"] = str(model_class)
        selected["seed"] = int(seed)
        selected["cell_score"] = selected[list(STATE_TARGETS)].mean(axis=1)
        parts.append(selected[["date", "pair", "model_class", "seed", "cell_score"]])
    stacked = pd.concat(parts, ignore_index=True, sort=False)
    cell_count = stacked.groupby(["date", "pair"], observed=True).size()
    if not cell_count.eq(len(candidate)).all():
        raise ValueError(f"Incomplete Generation 25 prediction cells for {cohort}.")
    output = (
        stacked.groupby(["date", "pair"], observed=True, as_index=False)["cell_score"]
        .mean()
        .rename(columns={"date": "event_time", "cell_score": "activity_score"})
    )
    output["event_time"] = pd.to_datetime(output["event_time"], utc=True, errors="raise")
    output["g18_period"] = g18d.assign_confirmation_period(output["event_time"], cohort)
    output = output.loc[output["g18_period"].ne("outside_g18_confirmation")].copy()
    quantiles = (
        output.groupby(["pair", "g18_period"], observed=True)["activity_score"]
        .quantile([0.25, 0.75])
        .unstack()
        .rename(columns={0.25: "activity_q25", 0.75: "activity_q75"})
    )
    output = output.join(quantiles, on=["pair", "g18_period"])
    output["activity_state"] = np.select(
        [
            output["activity_score"].ge(output["activity_q75"]),
            output["activity_score"].le(output["activity_q25"]),
        ],
        ["high", "low"],
        default="middle",
    )
    return output


def _event_identity(path: Path) -> DataFrame:
    columns = (
        "cohort",
        "pair",
        "control",
        "event_time",
        "level_family",
        "level_name",
        "pre_distance_atr",
    )
    frame = pd.read_parquet(path, columns=list(columns))
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="raise")
    cohort_values = frame["cohort"].dropna().astype(str).unique()
    if len(cohort_values) != 1:
        raise ValueError(f"Unexpected cohort values in {path}: {cohort_values}")
    frame["g18_period"] = g18d.assign_confirmation_period(
        frame["event_time"], str(cohort_values[0])
    )
    return frame.loc[
        frame["g18_period"].ne("outside_g18_confirmation")
        & frame["level_family"].isin(ALL_LEVEL_FAMILIES)
        & frame["control"].isin({"actual", *CONTROLS})
    ].copy()


def _nearest_family_event(frame: DataFrame) -> DataFrame:
    keys = [
        "cohort",
        "pair",
        "g18_period",
        "control",
        "event_time",
        "level_family",
    ]
    return g18d.nearest_anchor(frame, keys)


def outcome_blind_support() -> DataFrame:
    """Count candidate/control/state rows without loading any reaction outcome."""
    states = {cohort: ensemble_activity_state(cohort) for cohort in COHORTS}
    event_paths = sorted(g18d.ATLAS_EVENT_ROOT.glob("*.parquet"))
    if len(event_paths) != 20:
        raise ValueError(f"Expected 20 level-event files, got {len(event_paths)}.")
    rows: list[DataFrame] = []
    for path in event_paths:
        frame = _event_identity(path)
        if frame.empty:
            continue
        cohort = str(frame["cohort"].iloc[0])
        pair = str(frame["pair"].iloc[0])
        state = states[cohort].loc[states[cohort]["pair"].eq(pair)]
        merged = frame.merge(
            state[["event_time", "g18_period", "activity_state"]],
            on=["event_time", "g18_period"],
            how="left",
            validate="many_to_one",
        )
        selected = _nearest_family_event(merged)
        grouped = (
            selected.loc[selected["activity_state"].isin(STATES)]
            .groupby(
                [
                    "cohort",
                    "pair",
                    "g18_period",
                    "level_family",
                    "control",
                    "activity_state",
                ],
                observed=True,
                sort=False,
            )
            .size()
            .rename("event_rows")
            .reset_index()
        )
        rows.append(grouped)
    if not rows:
        raise ValueError("No outcome-blind support rows were built.")
    return pd.concat(rows, ignore_index=True, sort=False)


def supported_families(support: DataFrame) -> list[str]:
    """Require broad group support in every cohort/period/control/state cell."""
    selected: list[str] = []
    for family in ALL_LEVEL_FAMILIES:
        cell = support.loc[support["level_family"].eq(family)].copy()
        eligible = (
            cell.assign(eligible_pair=cell["event_rows"].ge(MIN_PAIR_ROWS))
            .groupby(
                ["cohort", "g18_period", "control", "activity_state"],
                observed=True,
            )["eligible_pair"]
            .sum()
        )
        expected_cells = len(COHORTS) * 2 * (len(CONTROLS) + 1) * len(STATES)
        if len(eligible) == expected_cells and eligible.ge(MIN_BROAD_SCOPE_PAIRS).all():
            selected.append(family)
    return selected


def freeze_support() -> dict[str, Any]:
    _parent_validation()
    if FREEZE_PATH.is_file():
        frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        if frozen.get("status") != "frozen_before_generation26_reaction_outcomes":
            raise ValueError("Invalid Generation 26 freeze.")
        for contract in frozen["source_contracts"].values():
            _validate_artifact(contract)
        if tuple(frozen["selected_level_families"]) != EXPECTED_SUPPORTED_FAMILIES:
            raise ValueError("Frozen Generation 26 level families changed.")
        return frozen
    support = outcome_blind_support()
    selected = supported_families(support)
    if tuple(selected) != EXPECTED_SUPPORTED_FAMILIES:
        raise ValueError(
            f"Outcome-blind support selected {selected}, expected "
            f"{list(EXPECTED_SUPPORTED_FAMILIES)}."
        )
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(support, SUPPORT_PATH)
    parent_manifests = {
        f"g24_parent_manifest_{cohort}": artifact(g25a.parent_manifest_path(cohort))
        for cohort in COHORTS
    }
    frozen = {
        "schema_version": 1,
        "generation": GENERATION,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation26_reaction_outcomes",
        "plain_question": (
            "When the frozen market-state model expects unusually active trading, do "
            "exact current calculated-level contacts add more future unsigned activity "
            "than fake, stale, shifted, or nearly contacted locations, and more than the "
            "same levels during low predicted activity?"
        ),
        "baseline": (
            "Generation 25 market-state-only predictions ranked future activity, while "
            "Generation 24's broad level-feature model did not reliably improve on them."
        ),
        "state_definition": {
            "targets": list(STATE_TARGETS),
            "model_cells": [list(cell) for cell in sorted(g25a.g24cache.MODEL_CELLS)],
            "aggregation": (
                "mean of the two target predictions inside each fixed model/seed cell, "
                "then equal mean across the four cells"
            ),
            "high": "top quarter within each pair and frozen period",
            "low": "bottom quarter within each pair and frozen period",
            "future_targets_not_read_during_state_construction": True,
        },
        "all_level_families_considered": list(ALL_LEVEL_FAMILIES),
        "selected_level_families": selected,
        "selection_rule": (
            "Before reaction outcomes open, retain only families with at least ten rows "
            "per pair and at least five eligible pairs in every cohort, period, control, "
            "and high/low predicted-activity cell."
        ),
        "controls": list(CONTROLS),
        "metrics": list(METRICS),
        "horizons_hours": list(HORIZONS),
        "pass_rule": (
            "A lead must show a positive real-level advantage during high predicted "
            "activity and a positive high-minus-low interaction across both periods and "
            "all four controls, with at least 70% of eligible coins positive, positive "
            "equal-coin uncertainty, adequate rows, and no dominant coin."
        ),
        "interpretation_boundary": {
            "same_holdout_exploratory": True,
            "fresh_confirmation": False,
            "future_signed_direction_used": False,
            "profit_used": False,
            "trading_signal_claimed": False,
        },
        "support_artifact": artifact(SUPPORT_PATH),
        "source_contracts": {
            "analysis_script": artifact(ANALYSIS_PATH),
            "generation25_joint_review": artifact(PARENT_REVIEW),
            "generation17_level_source_result": artifact(G17_RESULT),
            "generation17_event_inventory": artifact(G17_INVENTORY),
            **parent_manifests,
        },
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(frozen, FREEZE_PATH)
    return frozen


def _outcome_columns() -> list[str]:
    return [
        "cohort",
        "pair",
        "control",
        "event_time",
        "level_family",
        "level_name",
        "pre_distance_atr",
        "zone_half_width_atr",
        *(
            column
            for horizon in HORIZONS
            for column in (
                f"abs_excursion_atr_h{horizon}",
                f"away_excursion_atr_h{horizon}",
                f"through_excursion_atr_h{horizon}",
                f"volume_ratio_h{horizon}",
                f"range_ratio_h{horizon}",
                f"dwell_fraction_h{horizon}",
                f"crossings_h{horizon}",
            )
        ),
    ]


def load_scoped_outcomes(frozen: dict[str, Any]) -> DataFrame:
    states = {cohort: ensemble_activity_state(cohort) for cohort in COHORTS}
    event_paths = sorted(g18d.ATLAS_EVENT_ROOT.glob("*.parquet"))
    rows: list[DataFrame] = []
    for path in event_paths:
        frame = pd.read_parquet(path, columns=_outcome_columns())
        frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="raise")
        cohort = str(frame["cohort"].dropna().astype(str).unique()[0])
        frame["g18_period"] = g18d.assign_confirmation_period(frame["event_time"], cohort)
        frame = frame.loc[
            frame["g18_period"].ne("outside_g18_confirmation")
            & frame["level_family"].isin(frozen["selected_level_families"])
            & frame["control"].isin({"actual", *CONTROLS})
        ].copy()
        if frame.empty:
            continue
        pair = str(frame["pair"].iloc[0])
        state = states[cohort].loc[states[cohort]["pair"].eq(pair)]
        frame = frame.merge(
            state[["event_time", "g18_period", "activity_state"]],
            on=["event_time", "g18_period"],
            how="left",
            validate="many_to_one",
        )
        frame = _nearest_family_event(frame)
        g18d.add_reaction_metrics(frame)
        for horizon in HORIZONS:
            frame[f"metric__range_ratio_h{horizon}"] = pd.to_numeric(
                frame[f"range_ratio_h{horizon}"], errors="coerce"
            )
        frame["scope_kind"] = "family_any_level"
        frame["scope_value"] = frame["level_family"]
        rows.append(frame)
    if not rows:
        raise ValueError("No Generation 26 outcome rows were loaded.")
    return pd.concat(rows, ignore_index=True, sort=False)


def outcome_summary(frame: DataFrame) -> DataFrame:
    keys = [
        "cohort",
        "pair",
        "g18_period",
        "scope_kind",
        "scope_value",
        "control",
        "activity_state",
    ]
    rows: list[DataFrame] = []
    selected = frame.loc[frame["activity_state"].isin(STATES)].copy()
    for metric in METRICS:
        for horizon in HORIZONS:
            column = f"metric__{metric}_h{horizon}"
            grouped = (
                selected[keys + [column]]
                .dropna(subset=[column])
                .groupby(keys, observed=True, sort=False)[column]
                .agg(["mean", "size"])
                .reset_index()
                .rename(columns={"mean": "outcome_mean", "size": "event_rows"})
            )
            grouped["metric"] = metric
            grouped["horizon_hours"] = horizon
            rows.append(grouped)
    return pd.concat(rows, ignore_index=True, sort=False)


def location_contrasts(summary: DataFrame) -> DataFrame:
    keys = [
        "cohort",
        "pair",
        "g18_period",
        "scope_kind",
        "scope_value",
        "activity_state",
        "metric",
        "horizon_hours",
    ]
    actual = summary.loc[summary["control"].eq("actual")].drop(columns="control")
    rows: list[DataFrame] = []
    for control in CONTROLS:
        baseline = summary.loc[summary["control"].eq(control)].drop(columns="control")
        merged = actual.merge(
            baseline,
            on=keys,
            how="inner",
            suffixes=("_actual", "_control"),
            validate="one_to_one",
        )
        merged["comparison"] = control
        merged["difference"] = merged["outcome_mean_actual"] - merged["outcome_mean_control"]
        merged["eligible_pair"] = (
            merged["event_rows_actual"].ge(MIN_PAIR_ROWS)
            & merged["event_rows_control"].ge(MIN_PAIR_ROWS)
        )
        rows.append(merged)
    return pd.concat(rows, ignore_index=True, sort=False)


def interaction_contrasts(location: DataFrame) -> DataFrame:
    keys = [
        "cohort",
        "pair",
        "g18_period",
        "scope_kind",
        "scope_value",
        "metric",
        "horizon_hours",
        "comparison",
    ]
    high = location.loc[location["activity_state"].eq("high")].drop(
        columns="activity_state"
    )
    low = location.loc[location["activity_state"].eq("low")].drop(
        columns="activity_state"
    )
    merged = high.merge(
        low,
        on=keys,
        how="inner",
        suffixes=("_high", "_low"),
        validate="one_to_one",
    )
    merged["high_state_level_effect"] = merged["difference_high"]
    merged["low_state_level_effect"] = merged["difference_low"]
    merged["difference"] = merged["difference_high"] - merged["difference_low"]
    merged["eligible_pair"] = (
        merged["eligible_pair_high"].astype(bool)
        & merged["eligible_pair_low"].astype(bool)
    )
    return merged


def combined_decisions(
    location_decisions: DataFrame, interaction_decisions: DataFrame
) -> DataFrame:
    keys = ["scope_kind", "scope_value", "metric", "horizon_hours", "market_scope"]
    high = location_decisions.loc[
        location_decisions["activity_state"].eq("high")
    ].drop(columns="activity_state")
    merged = high.merge(
        interaction_decisions,
        on=keys,
        how="outer",
        suffixes=("_high_location", "_interaction"),
        validate="one_to_one",
    )
    high_status = merged["status_high_location"].fillna("not_retained")
    interaction_status = merged["status_interaction"].fillna("not_retained")
    high_strict = high_status.eq("strict_holdout_confirmation")
    interaction_strict = interaction_status.eq("strict_holdout_confirmation")
    high_point = high_status.isin(
        ["strict_holdout_confirmation", "point_holdout_confirmation"]
    )
    interaction_point = interaction_status.isin(
        ["strict_holdout_confirmation", "point_holdout_confirmation"]
    )
    merged["status"] = np.select(
        [high_strict & interaction_strict, high_point & interaction_point],
        ["strict_same_holdout_interaction_lead", "point_same_holdout_interaction_lead"],
        default="not_retained",
    )
    merged["strict_lead"] = merged["status"].eq("strict_same_holdout_interaction_lead")
    merged["point_lead"] = merged["status"].isin(
        ["strict_same_holdout_interaction_lead", "point_same_holdout_interaction_lead"]
    )
    return merged.sort_values(keys).reset_index(drop=True)


def _write_outputs(run_dir: Path, outputs: dict[str, DataFrame]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, frame in outputs.items():
        path = run_dir / f"g26_{name}.csv"
        g0.atomic_write_csv(frame, path)
        paths[name] = path
    return paths


def run(run_id: str, *, overwrite: bool = False) -> dict[str, Any]:
    frozen = freeze_support()
    run_dir = RECORD_ROOT / run_id
    result_path = run_dir / "g26_state_location_result.json"
    if result_path.is_file() and not overwrite:
        return json.loads(result_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    outcomes = load_scoped_outcomes(frozen)
    summary = outcome_summary(outcomes)
    location = location_contrasts(summary)
    interaction = interaction_contrasts(location)
    location_scores = g18d.period_scores(
        location,
        ("scope_kind", "scope_value", "metric", "horizon_hours", "activity_state"),
    )
    location_decisions = g18d.whole_decisions(
        location_scores,
        ("scope_kind", "scope_value", "metric", "horizon_hours", "activity_state"),
        CONTROLS,
    )
    interaction_scores = g18d.period_scores(
        interaction,
        ("scope_kind", "scope_value", "metric", "horizon_hours"),
    )
    interaction_decisions = g18d.whole_decisions(
        interaction_scores,
        ("scope_kind", "scope_value", "metric", "horizon_hours"),
        CONTROLS,
    )
    decisions = combined_decisions(location_decisions, interaction_decisions)
    paths = _write_outputs(
        run_dir,
        {
            "outcome_summary": summary,
            "location_pair_contrasts": location,
            "interaction_pair_contrasts": interaction,
            "location_period_scores": location_scores,
            "location_decisions": location_decisions,
            "interaction_period_scores": interaction_scores,
            "interaction_decisions": interaction_decisions,
            "combined_decisions": decisions,
        },
    )
    retained = decisions.loc[decisions["point_lead"]].copy()
    result = {
        "schema_version": 1,
        "generation": GENERATION,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation26_state_location_interaction",
        "pairs": int(outcomes["pair"].nunique()),
        "event_rows": len(outcomes),
        "selected_level_families": list(frozen["selected_level_families"]),
        "controls": list(CONTROLS),
        "metrics": list(METRICS),
        "horizons_hours": list(HORIZONS),
        "decision_rows": len(decisions),
        "point_leads": int(decisions["point_lead"].sum()),
        "strict_leads": int(decisions["strict_lead"].sum()),
        "retained_rows": retained[
            [
                "scope_value",
                "metric",
                "horizon_hours",
                "market_scope",
                "status",
            ]
        ].to_dict("records"),
        "interpretation_boundary": frozen["interpretation_boundary"],
        "plain_result": (
            "This batch asks whether the existing activity forecast changes the added "
            "value of exact current level contacts. It does not predict direction or "
            "provide fresh chronological confirmation."
        ),
        "source_contracts": {
            "generation26_freeze": artifact(FREEZE_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
        "artifacts": {name: artifact(path) for name, path in paths.items()},
    }
    g0.atomic_write_json(result, result_path)
    return {**result, "result_path": str(result_path.resolve())}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(run(args.run_id, overwrite=bool(args.overwrite)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

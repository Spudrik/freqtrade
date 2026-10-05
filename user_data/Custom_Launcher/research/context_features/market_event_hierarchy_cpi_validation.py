"""Validate the selected short CPI reaction with event-wise negative controls.

This is a result-spawned robustness branch, not an independent confirmation.
It freezes the branch after the complete CPI sibling batch and tests whether the
short reaction is stronger than pre-release windows, matched non-release weeks,
and whole-event sign reshuffling.
"""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_cpi_direct as direct,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_cpi_freeze as frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
OUTPUT_ROOT = direct.OUTPUT_ROOT / "validation_20260904a"
FREEZE_PATH = OUTPUT_ROOT / "cpi_validation_freeze.json"
DIRECT_RESULT_PATH = direct.OUTPUT_ROOT / "cpi_direct_result.json"
MINUTE_OUTCOMES_PATH = direct.DETAIL_ROOT / "minute_outcomes.parquet"
DIRECTION_ROWS_PATH = direct.DETAIL_ROOT / "direction_rows.parquet"
ROUTE_DECISIONS_PATH = direct.OUTPUT_ROOT / "route_decisions.csv"

SELECTED_SIGNS = ("temperature_core_mom", "temperature_headline_mom")
SELECTED_HORIZONS = (5, 15, 30, 60, 120, 240)
SHORT_HORIZONS = (5, 15, 30, 60)
SELECTED_SCOPES = ("btc", "eth")
PLACEBO_WEEKS = tuple(range(1, 13))
PERMUTATION_ITERATIONS = 2000
PERMUTATION_SEED = 20260904
ROBUSTNESS_MARGIN = 0.05


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def build_branch_freeze() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "objective": "objective_02b_market_reaction_zone_discovery",
        "status": "frozen_result_spawned_cpi_validation_before_control_outcomes",
        "created_at_utc": g0.utc_now(),
        "parent_batch": artifact(DIRECT_RESULT_PATH),
        "parent_outcomes_already_reviewed": True,
        "independent_confirmation": False,
        "purpose": (
            "Determine whether the single short CPI reaction family survives negative "
            "controls; do not create additional CPI parameter variants."
        ),
        "selected_family": {
            "signs": list(SELECTED_SIGNS),
            "assets": list(SELECTED_SCOPES),
            "horizons_minutes": list(SELECTED_HORIZONS),
            "plain_effect": (
                "Hotter month-on-month inflation aligned with down and cooler aligned with "
                "up mainly during the first hour."
            ),
        },
        "frozen_siblings": [
            {
                "id": "pre_release_negative_control",
                "rule": "Apply the CPI direction to the equally long window before release.",
            },
            {
                "id": "same_clock_non_release_weeks",
                "rule": (
                    "Apply each release's sign to the same clock 1-12 weeks earlier, excluding "
                    "windows within 24 hours of another frozen CPI or FOMC event."
                ),
            },
            {
                "id": "whole_event_joint_permutation",
                "rule": (
                    "Shuffle complete CPI sign rows within development and validation, preserve "
                    "relationships among CPI measures, and compare the best route from the full "
                    "immediate family across both the all-release and FOMC-excluded versions."
                ),
                "iterations": PERMUTATION_ITERATIONS,
                "seed": PERMUTATION_SEED,
            },
            {
                "id": "calendar_year_stability",
                "rule": "Report every calendar year; do not remove a weak year.",
            },
            {
                "id": "timing_and_fade",
                "rule": "Report 5-240 minutes to show where the relationship ends.",
            },
        ],
        "retain_for_consensus_test_rule": (
            "Full-family event-wise permutation probability at most 5%; at least three of "
            "four 5-60 minute BTC routes and two of four ETH routes for core month-on-month "
            "survive the parent rule; and the median short route is at least five percentage "
            "points stronger after release than both pre-release and matched-week controls."
        ),
        "boundary": (
            "Passing only promotes the question to a true expectation-versus-release test. "
            "It does not create a trading rule."
        ),
        "analysis_script": artifact(ANALYSIS_PATH),
    }


def freeze_branch(*, overwrite: bool = False) -> dict[str, Any]:
    if FREEZE_PATH.is_file() and not overwrite:
        document = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        if document.get("status") != (
            "frozen_result_spawned_cpi_validation_before_control_outcomes"
        ):
            raise ValueError("Existing CPI validation freeze is not valid.")
        return document
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    document = build_branch_freeze()
    g0.atomic_write_json(document, FREEZE_PATH)
    return document


def load_inputs() -> tuple[dict[str, Any], DataFrame, DataFrame, DataFrame, DataFrame]:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("status") != (
        "frozen_result_spawned_cpi_validation_before_control_outcomes"
    ):
        raise ValueError("CPI validation was not frozen.")
    direct_result = json.loads(DIRECT_RESULT_PATH.read_text(encoding="utf-8"))
    if direct_result.get("status") != "completed_cpi_direct_review":
        raise ValueError("Parent CPI direct batch is not terminal.")
    if freeze["parent_batch"]["sha256"] != g0.sha256_file(DIRECT_RESULT_PATH):
        raise ValueError("Parent CPI result changed after validation freeze.")
    _, _, catalog = direct.load_frozen_inputs()
    catalog = catalog.copy()
    catalog["fomc_within_24h"] = direct.fomc_collision_flags(catalog)
    minute = pd.read_parquet(MINUTE_OUTCOMES_PATH)
    direction = pd.read_parquet(DIRECTION_ROWS_PATH)
    decisions = pd.read_csv(ROUTE_DECISIONS_PATH)
    return freeze, catalog, minute, direction, decisions


def _hit(predicted: float, actual_return: float) -> float:
    if pd.isna(predicted) or predicted == 0 or pd.isna(actual_return) or actual_return == 0:
        return np.nan
    return float(np.sign(actual_return) == np.sign(predicted))


def pre_release_controls(direction: DataFrame) -> DataFrame:
    selected = direction.loc[
        direction["clock"].eq("immediate_1m")
        & direction["scope"].isin(SELECTED_SCOPES)
        & direction["sign_id"].isin(SELECTED_SIGNS)
        & direction["horizon"].isin(SELECTED_HORIZONS)
    ].copy()
    selected["pre_release_hit"] = [
        _hit(predicted, prior)
        for predicted, prior in zip(
            selected["predicted_direction"], selected["pre_return"], strict=True
        )
    ]
    selected["event_year"] = pd.to_datetime(selected["anchor_utc"], utc=True).dt.year
    records: list[dict[str, Any]] = []
    keys = ["sign_id", "scope", "horizon", "whole_event_partition"]
    for key, group in selected.groupby(keys, sort=False, dropna=False):
        usable = group.loc[
            group["direction_hit"].notna() & group["pre_release_hit"].notna()
        ]
        post = usable["direction_hit"].astype(bool)
        pre = usable["pre_release_hit"]
        records.append(
            {
                **dict(zip(keys, key, strict=True)),
                "n_events": int(usable["event_id"].nunique()),
                "post_release_accuracy": float(post.mean()) if len(post) else np.nan,
                "pre_release_accuracy": float(pre.mean()) if len(pre) else np.nan,
                "post_minus_pre": float(post.mean() - pre.mean())
                if len(post) and len(pre)
                else np.nan,
            }
        )
    return DataFrame.from_records(records)


def year_stability(direction: DataFrame) -> DataFrame:
    selected = direction.loc[
        direction["clock"].eq("immediate_1m")
        & direction["scope"].isin(SELECTED_SCOPES)
        & direction["sign_id"].isin(SELECTED_SIGNS)
        & direction["horizon"].isin(SELECTED_HORIZONS)
    ].copy()
    selected["event_year"] = pd.to_datetime(selected["anchor_utc"], utc=True).dt.year
    usable = selected.loc[selected["direction_hit"].notna()].copy()
    usable["signed_return"] = (
        usable["response_return"] * usable["predicted_direction"]
    )
    return (
        usable.groupby(["sign_id", "scope", "horizon", "event_year"], as_index=False)
        .agg(
            n_events=("event_id", "nunique"),
            accuracy=("direction_hit", "mean"),
            median_signed_return=("signed_return", "median"),
        )
        .sort_values(["sign_id", "scope", "horizon", "event_year"], kind="stable")
    )


def matched_week_rows(catalog: DataFrame) -> DataFrame:
    fomc = pd.read_csv(direct.FOMC_CATALOG_PATH, usecols=["anchor_utc"])
    blocked = list(catalog["anchor_utc"]) + list(
        pd.to_datetime(fomc["anchor_utc"], utc=True)
    )
    records: list[dict[str, Any]] = []
    pair_by_scope = {"btc": "BTC/USDT:USDT", "eth": "ETH/USDT:USDT"}
    for scope, pair in pair_by_scope.items():
        frame = g0.load_ohlcv(g0.ohlcv_path(pair, "1m"))
        frame = frame.sort_values("date", kind="stable").drop_duplicates("date")
        frame = frame.reset_index(drop=True)
        positions = direct._position_by_date(frame)
        for event in catalog.itertuples(index=False):
            anchor = pd.Timestamp(event.anchor_utc)
            for sign_id in SELECTED_SIGNS:
                temperature = float(getattr(event, sign_id))
                if not np.isfinite(temperature) or temperature == 0:
                    continue
                prediction = -temperature
                for horizon in SELECTED_HORIZONS:
                    for weeks in PLACEBO_WEEKS:
                        control_anchor = anchor - pd.Timedelta(weeks=weeks)
                        if (
                            direct._minimum_event_distance_hours(blocked, control_anchor)
                            <= direct.EVENT_EXCLUSION_HOURS
                        ):
                            continue
                        position = positions.get(control_anchor)
                        if position is None:
                            continue
                        response = direct._future_window(frame, position, horizon)
                        if response is None:
                            continue
                        records.append(
                            {
                                "event_id": event.event_id,
                                "whole_event_partition": event.whole_event_partition,
                                "sign_id": sign_id,
                                "scope": scope,
                                "horizon": horizon,
                                "weeks_before_release": weeks,
                                "control_anchor_utc": control_anchor,
                                "predicted_direction": prediction,
                                "control_return": response["response_return"],
                                "control_hit": _hit(
                                    prediction, response["response_return"]
                                ),
                            }
                        )
    return DataFrame.from_records(records)


def matched_week_summary(
    direction: DataFrame, controls: DataFrame
) -> DataFrame:
    event_rows = direction.loc[
        direction["clock"].eq("immediate_1m")
        & direction["scope"].isin(SELECTED_SCOPES)
        & direction["sign_id"].isin(SELECTED_SIGNS)
        & direction["horizon"].isin(SELECTED_HORIZONS)
        & direction["whole_event_partition"].isin(direct.EVALUATION_PARTITIONS)
        & direction["direction_hit"].notna()
    ].copy()
    control_by_event = (
        controls.loc[
            controls["whole_event_partition"].isin(direct.EVALUATION_PARTITIONS)
        ]
        .groupby(["event_id", "sign_id", "scope", "horizon"], as_index=False)
        .agg(
            matched_week_accuracy=("control_hit", "mean"),
            matched_control_count=("control_hit", "count"),
        )
    )
    paired = event_rows.merge(
        control_by_event,
        on=["event_id", "sign_id", "scope", "horizon"],
        how="inner",
        validate="one_to_one",
    )
    paired["event_minus_matched"] = (
        paired["direction_hit"].astype(float) - paired["matched_week_accuracy"]
    )
    return (
        paired.groupby(["sign_id", "scope", "horizon"], as_index=False)
        .agg(
            n_events=("event_id", "nunique"),
            event_accuracy=("direction_hit", "mean"),
            matched_week_accuracy=("matched_week_accuracy", "mean"),
            mean_event_minus_matched=("event_minus_matched", "mean"),
            median_controls_per_event=("matched_control_count", "median"),
        )
        .sort_values(["sign_id", "scope", "horizon"], kind="stable")
    )


def _maximum_cross_period_accuracy(
    signs: np.ndarray,
    *,
    catalog: DataFrame,
    minute: DataFrame,
    sign_columns: Sequence[str],
) -> float:
    event_index = {event_id: index for index, event_id in enumerate(catalog["event_id"])}
    maximum = np.nan
    for sign_column in sign_columns:
        sign_position = list(frozen.ALL_SIGN_COLUMNS).index(sign_column)
        for exclude_fomc in (False, True):
            variant = minute.loc[~minute["fomc_within_24h"]] if exclude_fomc else minute
            for (scope, horizon), group in variant.groupby(
                ["scope", "horizon"], sort=False
            ):
                if scope not in SELECTED_SCOPES or horizon not in SELECTED_HORIZONS:
                    continue
                event_positions = np.array(
                    [event_index[event_id] for event_id in group["event_id"]], dtype=int
                )
                predicted = -signs[event_positions, sign_position]
                actual = np.sign(group["response_return"].to_numpy(dtype=float))
                partition = group["whole_event_partition"].to_numpy()
                accuracies: list[float] = []
                valid_route = True
                for name in direct.EVALUATION_PARTITIONS:
                    mask = (
                        (partition == name)
                        & np.isfinite(predicted)
                        & (predicted != 0)
                        & (actual != 0)
                    )
                    if int(mask.sum()) < direct.MIN_PARTITION_EVENTS:
                        valid_route = False
                        break
                    accuracies.append(float((predicted[mask] == actual[mask]).mean()))
                if valid_route:
                    route_statistic = min(accuracies)
                    maximum = (
                        route_statistic
                        if not np.isfinite(maximum)
                        else max(maximum, route_statistic)
                    )
    return float(maximum)


def whole_event_permutation(
    catalog: DataFrame,
    minute: DataFrame,
    *,
    iterations: int = PERMUTATION_ITERATIONS,
    seed: int = PERMUTATION_SEED,
) -> DataFrame:
    signs = catalog[list(frozen.ALL_SIGN_COLUMNS)].to_numpy(dtype=float)
    partition = catalog["whole_event_partition"].to_numpy()
    rng = np.random.default_rng(seed)
    families = {
        "full_frozen_immediate_family": tuple(frozen.ALL_SIGN_COLUMNS),
        "selected_mom_family": SELECTED_SIGNS,
    }
    observed = {
        family: _maximum_cross_period_accuracy(
            signs, catalog=catalog, minute=minute, sign_columns=columns
        )
        for family, columns in families.items()
    }
    exceedances = {family: 0 for family in families}
    null_values = {family: [] for family in families}
    partition_positions = [
        np.flatnonzero(partition == name) for name in direct.EVALUATION_PARTITIONS
    ]
    for _ in range(iterations):
        permuted = signs.copy()
        for positions in partition_positions:
            permuted[positions] = signs[rng.permutation(positions)]
        for family, columns in families.items():
            statistic = _maximum_cross_period_accuracy(
                permuted, catalog=catalog, minute=minute, sign_columns=columns
            )
            null_values[family].append(statistic)
            exceedances[family] += int(statistic >= observed[family])
    records = []
    for family in families:
        values = np.asarray(null_values[family], dtype=float)
        records.append(
            {
                "family": family,
                "iterations": iterations,
                "seed": seed,
                "observed_best_minimum_period_accuracy": observed[family],
                "null_median": float(np.nanmedian(values)),
                "null_95th_percentile": float(np.nanquantile(values, 0.95)),
                "familywise_probability": (exceedances[family] + 1) / (iterations + 1),
            }
        )
    return DataFrame.from_records(records)


def validation_decision(
    parent_decisions: DataFrame,
    pre: DataFrame,
    matched: DataFrame,
    permutation: DataFrame,
) -> dict[str, Any]:
    parent = parent_decisions.loc[
        parent_decisions["sample_variant"].eq("all_releases")
        & parent_decisions["clock"].eq("immediate_1m")
        & parent_decisions["sign_id"].eq("temperature_core_mom")
        & parent_decisions["horizon"].isin(SHORT_HORIZONS)
    ]
    pass_counts = (
        parent.assign(
            passed=parent["verdict"].eq("retained_cross_period_direction_lead")
        )
        .groupby("scope")["passed"]
        .sum()
        .to_dict()
    )
    pre_eval = pre.loc[
        pre["sign_id"].eq("temperature_core_mom")
        & pre["horizon"].isin(SHORT_HORIZONS)
        & pre["whole_event_partition"].isin(direct.EVALUATION_PARTITIONS)
    ]
    pre_route = (
        pre_eval.groupby(["scope", "horizon"], as_index=False)
        .agg(
            post_release_accuracy=("post_release_accuracy", "mean"),
            pre_release_accuracy=("pre_release_accuracy", "mean"),
        )
    )
    pre_margin = float(
        (pre_route["post_release_accuracy"] - pre_route["pre_release_accuracy"]).median()
    )
    matched_margin = float(
        matched.loc[
            matched["sign_id"].eq("temperature_core_mom")
            & matched["horizon"].isin(SHORT_HORIZONS),
            "mean_event_minus_matched",
        ].median()
    )
    full_probability = float(
        permutation.loc[
            permutation["family"].eq("full_frozen_immediate_family"),
            "familywise_probability",
        ].iloc[0]
    )
    passed = (
        int(pass_counts.get("btc", 0)) >= 3
        and int(pass_counts.get("eth", 0)) >= 2
        and pre_margin >= ROBUSTNESS_MARGIN
        and matched_margin >= ROBUSTNESS_MARGIN
        and full_probability <= 0.05
    )
    return {
        "status": (
            "retained_for_true_consensus_and_confirmation_test"
            if passed
            else "parked_after_negative_control_review"
        ),
        "btc_short_parent_routes_passed": int(pass_counts.get("btc", 0)),
        "eth_short_parent_routes_passed": int(pass_counts.get("eth", 0)),
        "median_post_minus_pre_accuracy": pre_margin,
        "median_event_minus_matched_week_accuracy": matched_margin,
        "full_family_permutation_probability": full_probability,
        "plain_boundary": (
            "A retained result remains a lead about CPI release reactions. It still needs "
            "timestamped market expectations and genuinely new future releases."
        ),
    }


def render_report(decision: Mapping[str, Any], permutation: DataFrame) -> str:
    full = permutation.loc[
        permutation["family"].eq("full_frozen_immediate_family")
    ].iloc[0]
    return "\n".join(
        [
            "# CPI Short-Reaction Validation",
            "",
            f"- Decision: `{decision['status']}`",
            (
                "- Parent short routes retained: "
                f"BTC `{decision['btc_short_parent_routes_passed']}/4`, "
                f"ETH `{decision['eth_short_parent_routes_passed']}/4`"
            ),
            (
                "- Median advantage over equally long pre-release windows: "
                f"`{decision['median_post_minus_pre_accuracy']:.1%}`"
            ),
            (
                "- Median advantage over matched non-release weeks: "
                f"`{decision['median_event_minus_matched_week_accuracy']:.1%}`"
            ),
            (
                "- Chance of an equally strong best result after whole-release reshuffling: "
                f"`{full['familywise_probability']:.2%}`"
            ),
            "",
            "## Plain conclusion",
            "",
            (
                "The branch tests whether the short CPI relationship is tied to the release "
                "rather than ordinary movement at similar times. Passing keeps one research "
                "lead; it does not turn the repeated windows or two highly related coins into "
                "independent discoveries."
            ),
            "",
        ]
    )


def execute(*, overwrite: bool = False, iterations: int = PERMUTATION_ITERATIONS) -> dict[str, Any]:
    result_path = OUTPUT_ROOT / "cpi_validation_result.json"
    if result_path.is_file() and not overwrite:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed_cpi_validation":
            raise ValueError("Existing CPI validation result is not terminal.")
        return result
    _, catalog, minute, direction, parent_decisions = load_inputs()
    pre = pre_release_controls(direction)
    years = year_stability(direction)
    matched_rows = matched_week_rows(catalog)
    matched = matched_week_summary(direction, matched_rows)
    permutation = whole_event_permutation(catalog, minute, iterations=iterations)
    decision = validation_decision(parent_decisions, pre, matched, permutation)

    summary_paths = {
        "pre_release_controls": OUTPUT_ROOT / "pre_release_controls.csv",
        "matched_week_summary": OUTPUT_ROOT / "matched_week_summary.csv",
        "year_stability": OUTPUT_ROOT / "year_stability.csv",
        "whole_event_permutation": OUTPUT_ROOT / "whole_event_permutation.csv",
    }
    detail_path = OUTPUT_ROOT / "matched_week_rows.parquet"
    g0.atomic_write_csv(pre, summary_paths["pre_release_controls"])
    g0.atomic_write_csv(matched, summary_paths["matched_week_summary"])
    g0.atomic_write_csv(years, summary_paths["year_stability"])
    g0.atomic_write_csv(permutation, summary_paths["whole_event_permutation"])
    g0.atomic_write_parquet(matched_rows, detail_path)
    report_path = OUTPUT_ROOT / "cpi_validation_plain_review.md"
    report_path.write_text(
        render_report(decision, permutation), encoding="utf-8", newline="\n"
    )
    result = {
        "schema_version": 1,
        "status": "completed_cpi_validation",
        "created_at_utc": g0.utc_now(),
        "decision": decision,
        "independent_confirmation": False,
        "profit_used": False,
        "freeze_contract": artifact(FREEZE_PATH),
        "parent_result": artifact(DIRECT_RESULT_PATH),
        "summary_artifacts": {
            name: artifact(path) for name, path in summary_paths.items()
        },
        "detail_artifact": artifact(detail_path),
        "plain_review": artifact(report_path),
        "analysis_script": artifact(ANALYSIS_PATH),
    }
    g0.atomic_write_json(result, result_path)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--iterations", type=int, default=PERMUTATION_ITERATIONS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.freeze:
        print(json.dumps(freeze_branch(overwrite=args.overwrite), indent=2))
        return os.EX_OK
    if args.execute:
        print(
            json.dumps(
                execute(overwrite=args.overwrite, iterations=args.iterations), indent=2
            )
        )
        return os.EX_OK
    print(
        json.dumps(
            {
                "status": "ready_not_executed",
                "next": "run --freeze before --execute",
                "output_root": str(OUTPUT_ROOT),
            },
            indent=2,
        )
    )
    return os.EX_OK


if __name__ == "__main__":
    raise SystemExit(main())

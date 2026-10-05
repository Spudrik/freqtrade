"""Run the bounded robustness review for the retained one-hour meme direction cell."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import fisher_exact


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_meme_transmission_2026 as parent,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
PAIR_PATH = parent.DETAIL_ROOT / "meme_transmission_pair_rows.parquet"
GROUP_PATH = parent.DETAIL_ROOT / "meme_transmission_group_rows.parquet"
OUTPUT_PATH = parent.OUTPUT_ROOT / "meme_transmission_direction_validation.json"
COIN_PATH = parent.OUTPUT_ROOT / "meme_transmission_direction_coin_breadth.csv"
FAMILY_PATH = parent.OUTPUT_ROOT / "meme_transmission_direction_family_diagnostic.csv"
RESHUFFLES = 10_000
RANDOM_SEED = 20260909


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def _binomial_upper_tail(successes: int, attempts: int) -> float:
    if attempts <= 0:
        return math.nan
    numerator = sum(math.comb(attempts, value) for value in range(successes, attempts + 1))
    return numerator / (2**attempts)


def _rate(frame: DataFrame, column: str) -> float:
    return float(frame[column].mean()) if len(frame) else math.nan


def _validate_inputs() -> dict[str, Any]:
    result = json.loads(parent.RESULT_PATH.read_text(encoding="utf-8"))
    if result.get("status") != "completed_event_meme_transmission_review":
        raise ValueError("The parent meme transmission batch is not terminal.")
    for name, path in (("pair_details", PAIR_PATH), ("group_details", GROUP_PATH)):
        if result["artifacts"][name]["sha256"] != g0.sha256_file(path):
            raise ValueError(f"Parent outcome details changed: {name}")
    return result


def _direction_cells(groups: DataFrame) -> tuple[DataFrame, float]:
    actual = groups.loc[
        groups["sample_kind"].eq("actual_event")
        & groups["complete_cohort"]
        & groups["btc_confirmed_reaction"]
        & groups["cohort"].isin(["memes", "established_alts"])
        & groups["horizon_hours"].isin(parent.HORIZONS)
    ].drop_duplicates(["episode_id", "cohort", "horizon_hours"])
    actual = actual.copy()
    actual["cohort_direction_sign"] = np.sign(actual["cohort_lag_log_return"])
    pivot = actual.pivot(
        index="episode_id",
        columns=["cohort", "horizon_hours"],
        values="cohort_direction_sign",
    ).dropna()
    initial = (
        actual.drop_duplicates("episode_id").set_index("episode_id")["btc_initial_sign"]
    ).reindex(pivot.index)
    observed_rates = pivot.eq(initial, axis=0).mean()
    observed_max = float(observed_rates.max())
    rng = np.random.default_rng(RANDOM_SEED)
    initial_values = initial.to_numpy()
    outcome_values = pivot.to_numpy()
    exceed = 0
    for _ in range(RESHUFFLES):
        shuffled = rng.permutation(initial_values)
        maximum = np.equal(outcome_values, shuffled[:, None]).mean(axis=0).max()
        exceed += maximum >= observed_max
    chance = (exceed + 1) / (RESHUFFLES + 1)
    rows = [
        {
            "cohort": cohort,
            "horizon_hours": horizon,
            "support": len(pivot),
            "success_rate": float(rate),
        }
        for (cohort, horizon), rate in observed_rates.items()
    ]
    return DataFrame.from_records(rows), chance


def _coin_breadth(pair_rows: DataFrame, meme_pairs: list[str]) -> DataFrame:
    rows = pair_rows.loc[
        pair_rows["sample_kind"].eq("actual_event")
        & pair_rows["pair"].isin(meme_pairs)
        & pair_rows["horizon_hours"].eq(1)
        & pair_rows["btc_confirmed_reaction"]
    ].drop_duplicates(["episode_id", "pair"])
    records: list[dict[str, Any]] = []
    for pair, group in rows.groupby("pair", sort=False):
        first = group.loc[group["model_period"].eq(parent.PERIODS[0])]
        second = group.loc[group["model_period"].eq(parent.PERIODS[1])]
        positive = group.loc[group["btc_initial_sign"].gt(0)]
        negative = group.loc[group["btc_initial_sign"].lt(0)]
        records.append(
            {
                "pair": pair,
                "support": len(group),
                "success_rate": _rate(group, "direction_aligned_with_initial_btc"),
                "first_half_support": len(first),
                "first_half_rate": _rate(first, "direction_aligned_with_initial_btc"),
                "second_half_support": len(second),
                "second_half_rate": _rate(second, "direction_aligned_with_initial_btc"),
                "positive_btc_support": len(positive),
                "positive_btc_rate": _rate(positive, "direction_aligned_with_initial_btc"),
                "negative_btc_support": len(negative),
                "negative_btc_rate": _rate(negative, "direction_aligned_with_initial_btc"),
                "meets_55_overall": bool(
                    _rate(group, "direction_aligned_with_initial_btc") >= parent.LEAD_RATE
                ),
            }
        )
    return DataFrame.from_records(records)


def _family_diagnostic(groups: DataFrame) -> DataFrame:
    rows = groups.loc[
        groups["sample_kind"].eq("actual_event")
        & groups["cohort"].eq("memes")
        & groups["horizon_hours"].eq(1)
        & groups["complete_cohort"]
        & groups["btc_confirmed_reaction"]
    ].drop_duplicates("episode_id")
    rows = rows.copy()
    rows["event_family"] = rows["event_families_json"].map(json.loads)
    rows = rows.explode("event_family")
    return (
        rows.groupby("event_family", sort=False)
        .agg(
            support=("episode_id", "nunique"),
            success_rate=("later_direction_aligned", "mean"),
        )
        .reset_index()
        .sort_values(["support", "event_family"], ascending=[False, True])
    )


def execute() -> dict[str, Any]:
    if OUTPUT_PATH.is_file():
        result = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        if result.get("status") != "completed_bounded_direction_validation":
            raise ValueError("Existing direction validation is not terminal.")
        return result
    parent_result = _validate_inputs()
    pair_rows = pd.read_parquet(PAIR_PATH)
    groups = pd.read_parquet(GROUP_PATH)
    meme_pairs = json.loads(parent.FREEZE_PATH.read_text(encoding="utf-8"))["cohorts"][
        "top_ten_traded_memes"
    ]
    direction_cells, familywise_chance = _direction_cells(groups)
    coin_breadth = _coin_breadth(pair_rows, meme_pairs)
    families = _family_diagnostic(groups)

    actual = groups.loc[
        groups["sample_kind"].eq("actual_event")
        & groups["cohort"].eq("memes")
        & groups["horizon_hours"].eq(1)
        & groups["complete_cohort"]
        & groups["btc_confirmed_reaction"]
    ].drop_duplicates("episode_id")
    controls = groups.loc[
        groups["sample_kind"].eq("matched_control")
        & groups["cohort"].eq("memes")
        & groups["horizon_hours"].eq(1)
        & groups["complete_cohort"]
        & groups["btc_confirmed_reaction"]
    ].drop_duplicates("sample_id")
    actual_successes = int(actual["later_direction_aligned"].sum())
    control_successes = int(controls["later_direction_aligned"].sum())
    fisher_p = float(
        fisher_exact(
            [
                [actual_successes, len(actual) - actual_successes],
                [control_successes, len(controls) - control_successes],
            ],
            alternative="greater",
        ).pvalue
    )
    positive = actual.loc[actual["btc_initial_sign"].gt(0)]
    negative = actual.loc[actual["btc_initial_sign"].lt(0)]
    first = actual.loc[actual["model_period"].eq(parent.PERIODS[0])]
    second = actual.loc[actual["model_period"].eq(parent.PERIODS[1])]

    btc = pair_rows.loc[
        pair_rows["sample_kind"].eq("actual_event")
        & pair_rows["pair"].eq(parent.BTC_PAIR)
        & pair_rows["horizon_hours"].eq(1)
        & pair_rows["btc_confirmed_reaction"]
    ].drop_duplicates("episode_id")
    btc = btc.copy()
    btc["btc_continued_initial_direction"] = np.sign(btc["btc_lag_log_return"]).eq(
        btc["btc_initial_sign"]
    )
    concurrent = actual.merge(
        btc[["episode_id", "btc_lag_log_return", "btc_continued_initial_direction"]],
        on="episode_id",
        how="inner",
        validate="one_to_one",
    )
    concurrent["meme_matches_next_hour_btc"] = np.sign(
        concurrent["cohort_lag_log_return"]
    ).eq(np.sign(concurrent["btc_lag_log_return"]))
    btc_continued = concurrent.loc[concurrent["btc_continued_initial_direction"]]
    btc_reversed = concurrent.loc[~concurrent["btc_continued_initial_direction"]]

    breadth_count = int(coin_breadth["meets_55_overall"].sum())
    later_half_rate = _rate(second, "later_direction_aligned")
    sign_balance_pass = (
        len(positive) >= parent.MINIMUM_HALF_EPISODES
        and len(negative) >= parent.MINIMUM_HALF_EPISODES
        and _rate(positive, "later_direction_aligned") >= parent.LEAD_RATE
        and _rate(negative, "later_direction_aligned") >= parent.LEAD_RATE
    )
    robust = bool(
        familywise_chance < 0.05
        and fisher_p < 0.05
        and later_half_rate >= parent.LEAD_RATE
        and breadth_count >= 6
        and sign_balance_pass
    )
    parent.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(coin_breadth, COIN_PATH)
    g0.atomic_write_csv(families, FAMILY_PATH)
    result = {
        "schema_version": 1,
        "status": "completed_bounded_direction_validation",
        "created_at_utc": g0.utc_now(),
        "scope": (
            "Retrospective robustness checks only for the predeclared one-hour meme "
            "direction lead; no threshold or horizon tuning."
        ),
        "selected_result": {
            "successes": actual_successes,
            "episodes": len(actual),
            "success_rate": _rate(actual, "later_direction_aligned"),
            "one_sided_binomial_chance": _binomial_upper_tail(
                actual_successes, len(actual)
            ),
            "matched_control_successes": control_successes,
            "matched_control_samples": len(controls),
            "matched_control_rate": _rate(controls, "later_direction_aligned"),
            "event_greater_than_control_fisher_p": fisher_p,
            "first_half_support": len(first),
            "first_half_rate": _rate(first, "later_direction_aligned"),
            "second_half_support": len(second),
            "second_half_rate": later_half_rate,
            "positive_btc_support": len(positive),
            "positive_btc_rate": _rate(positive, "later_direction_aligned"),
            "negative_btc_support": len(negative),
            "negative_btc_rate": _rate(negative, "later_direction_aligned"),
        },
        "direction_family_chance_control": {
            "cells": direction_cells.to_dict(orient="records"),
            "whole_episode_sign_reshuffles": RESHUFFLES,
            "random_seed": RANDOM_SEED,
            "probability_any_of_four_cells_reaches_observed_best": familywise_chance,
        },
        "breadth": {
            "meme_coins_at_or_above_55_percent": breadth_count,
            "meme_coin_count": len(coin_breadth),
        },
        "leader_continuation": {
            "btc_next_hour_continuation_rate": _rate(
                btc, "btc_continued_initial_direction"
            ),
            "meme_matches_concurrent_next_hour_btc_rate": _rate(
                concurrent, "meme_matches_next_hour_btc"
            ),
            "meme_matches_initial_btc_when_btc_continues": _rate(
                btc_continued, "later_direction_aligned"
            ),
            "meme_matches_initial_btc_when_btc_reverses": _rate(
                btc_reversed, "later_direction_aligned"
            ),
        },
        "robust_confirmation": robust,
        "decision": (
            "confirmed_one_hour_direction_lead"
            if robust
            else "exploratory_one_hour_lead_not_robustly_confirmed"
        ),
        "plain_interpretation": (
            "The one-hour rate is worth preserving as a lead only if it is broad across "
            "coins and signs, repeats in the later half, and survives the four-cell chance "
            "control. Otherwise it must not be presented as a dependable prediction."
        ),
        "parent_result": {
            "path": str(parent.RESULT_PATH.resolve()),
            "sha256": g0.sha256_file(parent.RESULT_PATH),
            "run_id": parent_result["run_id"],
        },
        "source_contracts": {
            "analysis_script": _artifact(ANALYSIS_PATH),
            "pair_details": _artifact(PAIR_PATH),
            "group_details": _artifact(GROUP_PATH),
        },
        "artifacts": {
            "coin_breadth": _artifact(COIN_PATH),
            "family_diagnostic": _artifact(FAMILY_PATH),
        },
    }
    g0.atomic_write_json(result, OUTPUT_PATH)
    return result


def main() -> int:
    print(json.dumps(execute(), indent=2, default=g0.json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Reinterpret frozen event results as interacting whole-market episodes.

This is a read-only reanalysis of existing frozen artifacts.  It does not refit a
model, change the event catalogue, or claim that a downstream market response is
the cause of an event-period move.
"""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
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
    market_event_simple_signal_families as simple,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
RUN_ID = "event_whole_episode_reanalysis_20260909a"
SOURCE_ROOT = simple.SOURCE_ROOT
OUTPUT_ROOT = SOURCE_ROOT / RUN_ID
RESULT_PATH = OUTPUT_ROOT / "whole_episode_reanalysis_result.json"

MEME_ROOT = SOURCE_ROOT / "event_meme_transmission_20260909a"
MEME_RESULT_PATH = MEME_ROOT / "meme_transmission_result.json"
MEME_VALIDATION_PATH = MEME_ROOT / "meme_transmission_direction_validation.json"

ORIGINAL_PERIODS = tuple(simple.PERIODS)
COMBINED_2026_PERIOD = "confirmation_2026"
BTC_PAIR = "BTC/USDT:USDT"

ROLE_CORRECTIONS: tuple[dict[str, str], ...] = (
    {
        "finding": "recent_volume_persistence",
        "old_risk": "Could be misread as the root cause of later activity.",
        "correct_role": "background_or_readiness",
        "plain_meaning": (
            "Activity already visible before the measured window often continues. "
            "It does not identify what created that activity."
        ),
    },
    {
        "finding": "catalogued_event_clock",
        "old_risk": (
            "A small addition after controlling for prior volume could be misread as "
            "evidence that news did not drive the activity."
        ),
        "correct_role": "possible_driver_or_timing_anchor",
        "plain_meaning": (
            "The pooled clock asks only whether a broad event label adds a generic "
            "warning after the market state is known. Event-family evidence is needed "
            "to judge whether particular news caused a reaction."
        ),
    },
    {
        "finding": "meme_same_hour_breadth",
        "old_risk": "Could be misread as a forecast available before Bitcoin moved.",
        "correct_role": "confirmation_and_transmission",
        "plain_meaning": (
            "Widespread meme activity shows that a Bitcoin reaction is spreading, but "
            "it is observed during the reaction."
        ),
    },
    {
        "finding": "meme_one_hour_direction",
        "old_risk": (
            "Could be rejected completely because its average changed between two "
            "short 2026 periods."
        ),
        "correct_role": "unresolved_conditional_lead",
        "plain_meaning": (
            "The changing result may reflect other market conditions. Preserve it as "
            "a lead to explain, but do not call it a dependable rule."
        ),
    },
    {
        "finding": "btc_continuation_during_following_hour",
        "old_risk": "Could be used as though it were known at the event time.",
        "correct_role": "downstream_confirmation_or_outcome",
        "plain_meaning": (
            "Meme direction is much clearer when Bitcoin keeps moving, but Bitcoin's "
            "continuation is only known as that next hour unfolds."
        ),
    },
    {
        "finding": "generic_levels_background_and_alignment_nulls",
        "old_risk": "Could be read as rejecting those inputs for every purpose.",
        "correct_role": "role_specific_non_confirmation",
        "plain_meaning": (
            "They failed only as broad stand-alone volume warnings in this test. That "
            "does not reject their reaction, fade, or conditional modifier roles."
        ),
    },
)


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON list.")
    return [str(item) for item in parsed]


def _records(frame: DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _with_combined_2026(frame: DataFrame) -> DataFrame:
    current = frame.loc[
        frame["model_period"].astype(str).str.startswith("untouched_confirmation_2026")
    ].copy()
    current["model_period"] = COMBINED_2026_PERIOD
    return pd.concat([frame, current], ignore_index=True)


def _earliest_actual_anchor_per_episode(frame: DataFrame) -> DataFrame:
    actual = frame.loc[frame["sample_kind"].eq("actual_event")].copy()
    controls = frame.loc[frame["sample_kind"].eq("matched_control")].copy()
    actual = actual.sort_values(
        [
            "model_period",
            "analysis_unit_id",
            "horizon_hours",
            "model_anchor_utc",
            "sample_id",
        ]
    ).drop_duplicates(
        ["model_period", "analysis_unit_id", "horizon_hours"], keep="first"
    )
    return pd.concat([actual, controls], ignore_index=True)


def _verify_artifacts(result: Mapping[str, Any]) -> None:
    for contract in result.get("artifacts", {}).values():
        path = Path(contract["path"])
        if not path.is_file() or g0.sha256_file(path) != contract["sha256"]:
            raise ValueError(f"Frozen source artifact changed: {path}")


def load_event_volume_rows() -> DataFrame:
    frozen = _load_json(simple.FREEZE_PATH)
    simple._validate_contracts(frozen)
    source_result = _load_json(simple.RESULT_PATH)
    if source_result.get("status") != "completed_simple_signal_family_review":
        raise ValueError("Simple-family source result is not terminal.")
    _verify_artifacts(source_result)

    calls = pd.read_parquet(simple.CALLS_PATH)
    calls["signal_issued"] = simple._coerce_bool(calls["signal_issued"])
    outcomes = simple._outcome_rows(calls, frozen)
    outcomes = outcomes.loc[
        outcomes["signal_id"].eq("recent_volume_persistence")
        & outcomes["pair"].eq(BTC_PAIR)
        & outcomes["model_period"].isin(ORIGINAL_PERIODS)
    ].copy()
    metadata_columns = [
        "sample_id",
        "sample_kind",
        "analysis_unit_id",
        "model_anchor_utc",
        "model_period",
        "pair",
        "signal_id",
        "feature_value",
        "signal_issued",
        "parent_episode_ids_json",
        "event_families_json",
        "event_kinds_json",
    ]
    metadata = calls.loc[
        calls["signal_id"].eq("recent_volume_persistence")
        & calls["pair"].eq(BTC_PAIR),
        metadata_columns,
    ].copy()
    keys = [
        "sample_id",
        "sample_kind",
        "analysis_unit_id",
        "model_anchor_utc",
        "model_period",
        "pair",
        "signal_id",
        "signal_issued",
    ]
    enriched = outcomes.merge(
        metadata,
        on=keys,
        how="left",
        validate="many_to_one",
    )
    enriched["pre_event_volume_high"] = enriched["signal_issued"].astype(bool)
    enriched["volume_reaction"] = pd.to_numeric(
        enriched["volume_reaction"], errors="coerce"
    )
    return _with_combined_2026(_earliest_actual_anchor_per_episode(enriched))


def summarize_event_volume_groups(rows: DataFrame) -> DataFrame:
    keys = [
        "model_period",
        "horizon_hours",
        "sample_kind",
        "pre_event_volume_high",
    ]
    eligible = rows.loc[rows["volume_reaction"].notna()].copy()
    return (
        eligible.groupby(keys, observed=True, sort=False)
        .agg(
            episode_count=("analysis_unit_id", "nunique"),
            future_high_volume_rate=("volume_reaction", "mean"),
            median_future_volume_ratio=("volume_ratio", "median"),
            median_future_range_atr=("range_atr", "median"),
            median_future_absolute_move_atr=("absolute_close_move_atr", "median"),
        )
        .reset_index()
    )


def _event_control_pairs(rows: DataFrame) -> DataFrame:
    actual = rows.loc[rows["sample_kind"].eq("actual_event")].copy()
    actual = actual.rename(
        columns={
            "analysis_unit_id": "episode_id",
            "volume_reaction": "actual_volume_reaction",
            "volume_ratio": "actual_volume_ratio",
            "pre_event_volume_high": "actual_pre_event_volume_high",
        }
    )
    controls = rows.loc[rows["sample_kind"].eq("matched_control")].copy()
    controls["episode_id"] = controls["parent_episode_ids_json"].map(_json_list)
    controls = controls.explode("episode_id")
    controls = controls.rename(
        columns={
            "sample_id": "control_sample_id",
            "volume_reaction": "control_volume_reaction",
            "volume_ratio": "control_volume_ratio",
            "pre_event_volume_high": "control_pre_event_volume_high",
        }
    )
    left_columns = [
        "sample_id",
        "episode_id",
        "model_period",
        "horizon_hours",
        "event_families_json",
        "actual_pre_event_volume_high",
        "actual_volume_reaction",
        "actual_volume_ratio",
    ]
    right_columns = [
        "control_sample_id",
        "episode_id",
        "model_period",
        "horizon_hours",
        "control_pre_event_volume_high",
        "control_volume_reaction",
        "control_volume_ratio",
    ]
    return actual[left_columns].merge(
        controls[right_columns],
        on=["episode_id", "model_period", "horizon_hours"],
        how="left",
        validate="one_to_many",
    )


def summarize_paired_event_controls(rows: DataFrame) -> DataFrame:
    pairs = _event_control_pairs(rows)
    records: list[dict[str, Any]] = []
    for match_rule, subset in (
        ("all_matched_clocks", pairs),
        (
            "same_pre_event_volume_state",
            pairs.loc[
                pairs["actual_pre_event_volume_high"].eq(
                    pairs["control_pre_event_volume_high"]
                )
            ],
        ),
    ):
        subset = subset.loc[subset["control_volume_reaction"].notna()].copy()
        event_keys = [
            "sample_id",
            "episode_id",
            "model_period",
            "horizon_hours",
            "actual_pre_event_volume_high",
        ]
        per_event = (
            subset.groupby(event_keys, observed=True, sort=False)
            .agg(
                actual_volume_reaction=("actual_volume_reaction", "first"),
                actual_volume_ratio=("actual_volume_ratio", "first"),
                matched_control_count=("control_sample_id", "nunique"),
                mean_control_volume_reaction=("control_volume_reaction", "mean"),
                median_control_volume_ratio=("control_volume_ratio", "median"),
            )
            .reset_index()
        )
        if per_event.empty:
            continue
        per_event["actual_minus_control"] = (
            per_event["actual_volume_reaction"]
            - per_event["mean_control_volume_reaction"]
        )
        per_event["actual_beats_control_median"] = per_event[
            "actual_volume_ratio"
        ].gt(per_event["median_control_volume_ratio"])
        summary_keys = [
            "model_period",
            "horizon_hours",
            "actual_pre_event_volume_high",
        ]
        summary = (
            per_event.groupby(summary_keys, observed=True, sort=False)
            .agg(
                event_count=("episode_id", "nunique"),
                actual_future_high_volume_rate=("actual_volume_reaction", "mean"),
                mean_matched_control_high_volume_rate=(
                    "mean_control_volume_reaction",
                    "mean",
                ),
                mean_event_minus_control=("actual_minus_control", "mean"),
                event_beats_control_median_rate=(
                    "actual_beats_control_median",
                    "mean",
                ),
                median_controls_per_event=("matched_control_count", "median"),
            )
            .reset_index()
        )
        summary.insert(0, "match_rule", match_rule)
        records.extend(_records(summary))
    return DataFrame.from_records(records)


def summarize_event_families(rows: DataFrame) -> DataFrame:
    actual = rows.loc[
        rows["sample_kind"].eq("actual_event") & rows["volume_reaction"].notna()
    ].copy()
    actual["event_family"] = actual["event_families_json"].map(_json_list)
    actual = actual.explode("event_family")
    summary = (
        actual.groupby(
            [
                "model_period",
                "event_family",
                "horizon_hours",
                "pre_event_volume_high",
            ],
            observed=True,
            sort=False,
        )
        .agg(
            episode_count=("analysis_unit_id", "nunique"),
            future_high_volume_rate=("volume_reaction", "mean"),
            median_future_volume_ratio=("volume_ratio", "median"),
        )
        .reset_index()
    )
    return summary.loc[summary["episode_count"].ge(3)].reset_index(drop=True)


def _load_meme_context_rows(event_volume_rows: DataFrame) -> DataFrame:
    meme_result = _load_json(MEME_RESULT_PATH)
    if meme_result.get("status") != "completed_event_meme_transmission_review":
        raise ValueError("Meme transmission source result is not terminal.")
    _verify_artifacts(meme_result)
    validation = _load_json(MEME_VALIDATION_PATH)
    if validation.get("status") != "completed_bounded_direction_validation":
        raise ValueError("Meme direction validation is not terminal.")
    _verify_artifacts(validation)

    group_path = Path(meme_result["artifacts"]["group_details"]["path"])
    pair_path = Path(meme_result["artifacts"]["pair_details"]["path"])
    groups = pd.read_parquet(group_path)
    pair_rows = pd.read_parquet(pair_path)

    selected = groups.loc[
        groups["sample_kind"].eq("actual_event")
        & groups["cohort"].isin(["memes", "established_alts"])
        & groups["horizon_hours"].eq(1)
        & groups["complete_cohort"].fillna(False).astype(bool)
        & groups["btc_confirmed_reaction"].fillna(False).astype(bool)
    ].copy()
    btc_later = pair_rows.loc[
        pair_rows["sample_kind"].eq("actual_event")
        & pair_rows["pair"].eq(BTC_PAIR)
        & pair_rows["horizon_hours"].eq(1)
        & pair_rows["btc_confirmed_reaction"].fillna(False).astype(bool),
        ["sample_id", "direction_aligned_with_initial_btc"],
    ].drop_duplicates("sample_id")
    btc_later = btc_later.rename(
        columns={"direction_aligned_with_initial_btc": "btc_continued_initial_move"}
    )
    selected = selected.merge(
        btc_later,
        on="sample_id",
        how="left",
        validate="many_to_one",
    )

    context = event_volume_rows.loc[
        event_volume_rows["sample_kind"].eq("actual_event")
        & event_volume_rows["horizon_hours"].eq(1)
        & event_volume_rows["model_period"].isin(ORIGINAL_PERIODS),
        ["sample_id", "pre_event_volume_high"],
    ].drop_duplicates("sample_id")

    calls = pd.read_parquet(simple.CALLS_PATH)
    calls["signal_issued"] = simple._coerce_bool(calls["signal_issued"])
    context_signals = [
        "negative_slow_background",
        "single_calculated_area",
        "calculated_area_cluster",
        "broad_crypto_directional_alignment",
    ]
    call_context = calls.loc[
        calls["sample_kind"].eq("actual_event")
        & calls["pair"].eq(BTC_PAIR)
        & calls["signal_id"].isin(context_signals),
        ["sample_id", "signal_id", "signal_issued"],
    ].pivot(index="sample_id", columns="signal_id", values="signal_issued")
    call_context = call_context.reset_index()
    selected = selected.merge(context, on="sample_id", how="left", validate="many_to_one")
    return selected.merge(
        call_context,
        on="sample_id",
        how="left",
        validate="many_to_one",
    )


def summarize_meme_contexts(rows: DataFrame) -> DataFrame:
    contexts: tuple[tuple[str, str, bool], ...] = (
        ("model_period", "unresolved_time_or_market_state", True),
        ("event_family", "possible_driver_identity", True),
        ("btc_initial_sign", "initial_btc_response", True),
        ("btc_pre_activity_signal", "background_or_readiness", True),
        ("pre_event_volume_high", "background_or_readiness", True),
        ("negative_slow_background", "slow_background_modifier", True),
        ("single_calculated_area", "local_reaction_modifier", True),
        ("calculated_area_cluster", "local_reaction_modifier", True),
        (
            "broad_crypto_directional_alignment",
            "pre_event_cross_market_background",
            True,
        ),
        (
            "btc_continued_initial_move",
            "downstream_confirmation_or_outcome",
            False,
        ),
    )
    working = rows.copy()
    working["event_family"] = working["event_families_json"].map(_json_list)
    records: list[dict[str, Any]] = []
    for context_name, role, known_at_event_time in contexts:
        source = working.explode("event_family") if context_name == "event_family" else working
        for (cohort, value), group in source.groupby(
            ["cohort", context_name], observed=True, dropna=False, sort=False
        ):
            records.append(
                {
                    "context_name": context_name,
                    "context_role": role,
                    "known_at_event_time": known_at_event_time,
                    "cohort": str(cohort),
                    "context_value": str(value),
                    "episode_count": int(group["sample_id"].nunique()),
                    "direction_aligned_with_initial_btc_rate": float(
                        pd.to_numeric(
                            group["later_direction_aligned"], errors="coerce"
                        ).mean()
                    ),
                    "volume_majority_reacted_rate": float(
                        pd.to_numeric(
                            group["volume_majority_reacted"], errors="coerce"
                        ).mean()
                    ),
                    "range_majority_reacted_rate": float(
                        pd.to_numeric(
                            group["range_majority_reacted"], errors="coerce"
                        ).mean()
                    ),
                    "amplified_beyond_ordinary_rate": float(
                        pd.to_numeric(
                            group["amplified_beyond_ordinary"], errors="coerce"
                        ).mean()
                    ),
                    "evidence_limit": (
                        "descriptive_only_small_cell"
                        if group["sample_id"].nunique() < 10
                        else "descriptive_only_existing_episodes"
                    ),
                }
            )
    return DataFrame.from_records(records)


def run(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        existing = _load_json(RESULT_PATH)
        if existing.get("status") != "completed_whole_episode_reanalysis":
            raise ValueError("Existing whole-episode result is not terminal.")
        return existing

    event_rows = load_event_volume_rows()
    group_summary = summarize_event_volume_groups(event_rows)
    paired_summary = summarize_paired_event_controls(event_rows)
    family_summary = summarize_event_families(event_rows)
    meme_rows = _load_meme_context_rows(event_rows)
    meme_summary = summarize_meme_contexts(meme_rows)

    result = {
        "schema_version": 1,
        "status": "completed_whole_episode_reanalysis",
        "created_at_utc": g0.utc_now(),
        "run_id": RUN_ID,
        "scope": {
            "source_results_changed": False,
            "frozen_outcomes_reused": True,
            "actual_episode_anchor_rule": "earliest_anchor_only",
            "new_model_fitted": False,
            "trading_rule_tested": False,
            "causality_proven_by_this_reanalysis": False,
        },
        "plain_objective": (
            "Correct overly single-cause interpretations by assigning each existing "
            "finding a role in the whole event episode and measuring conditional "
            "differences without changing frozen results."
        ),
        "role_corrections": list(ROLE_CORRECTIONS),
        "event_and_prior_volume": {
            "question_answered": (
                "Does a pooled event clock add future-volume information once the "
                "market's immediately prior volume state is separated?"
            ),
            "question_not_answered": (
                "This comparison cannot determine whether news, anticipation of news, "
                "or another upstream condition caused the prior high volume."
            ),
            "group_rates": _records(group_summary),
            "paired_event_control_rates": _records(paired_summary),
            "event_family_descriptions_minimum_three_episodes": _records(family_summary),
        },
        "meme_and_established_coin_context": {
            "episode_count": int(meme_rows["sample_id"].nunique()),
            "scope": (
                "One-hour descriptive splits after a confirmed Bitcoin reaction; "
                "small cells explain what to test later but cannot establish a rule."
            ),
            "context_splits": _records(meme_summary),
        },
        "existing_result_audit": [
            {
                "result": "CPI and FOMC short reaction activity",
                "decision": "retain",
                "reason": (
                    "Matched clocks and repeated releases already test these as event "
                    "drivers; existing market activity can still amplify or suppress them."
                ),
            },
            {
                "result": "negative slow background after a positive event impulse",
                "decision": "retain_as_conditional_modifier",
                "reason": (
                    "It was tested after event confirmation and describes fade risk, not "
                    "a universal bearish forecast."
                ),
            },
            {
                "result": "single levels and level clusters",
                "decision": "retain_for_reaction_and_modifier_tests",
                "reason": (
                    "Their generic volume-warning failure does not test whether they "
                    "alter a known event reaction's size, path, or reversal."
                ),
            },
            {
                "result": "recent volume persistence",
                "decision": "retain_as_background_or_confirmation",
                "reason": (
                    "It predicts continued activity but does not identify the upstream "
                    "cause of that activity."
                ),
            },
            {
                "result": "pooled event plus prior high volume",
                "decision": "withdraw_causal_wording_retain_narrow_forecast_result",
                "reason": (
                    "It measures incremental generic forecast value only; event-family "
                    "driver evidence must be judged separately."
                ),
            },
            {
                "result": "meme one-hour direction after Bitcoin reacts",
                "decision": "retain_as_unresolved_conditional_lead",
                "reason": (
                    "The average is not robust, but the period shift and dependence on "
                    "Bitcoin continuation justify testing missing market conditions."
                ),
            },
        ],
        "next_broad_batch": {
            "purpose": (
                "Test several layers across complete episodes before following any one "
                "promising split."
            ),
            "routes": [
                "event family plus pre-event market and narrative background",
                "event family plus measured surprise where historical expectations exist",
                "initial Bitcoin and Ethereum reaction plus established-coin transmission",
                "single levels and clusters as reaction-size, path, and reversal modifiers",
                "meme transmission only after Bitcoin has a verified reaction",
            ],
            "branch_rule": (
                "Complete all routes, then review them together. Queue supported branches; "
                "do not branch immediately from the first attractive cell."
            ),
        },
        "source_contracts": {
            "analysis_script": artifact(ANALYSIS_PATH),
            "simple_result": artifact(simple.RESULT_PATH),
            "simple_freeze": artifact(simple.FREEZE_PATH),
            "simple_calls": artifact(simple.CALLS_PATH),
            "meme_result": artifact(MEME_RESULT_PATH),
            "meme_validation": artifact(MEME_VALIDATION_PATH),
        },
    }
    g0.atomic_write_json(result, RESULT_PATH)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = run(overwrite=args.overwrite)
    print(json.dumps(result, indent=2, default=g0.json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

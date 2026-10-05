"""Run the frozen breadth-first conditional whole-event direct tests."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import math
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
    market_event_conditional_episode_freeze_2026 as frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_simple_signal_families as simple,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
OUTPUT_ROOT = frozen.OUTPUT_ROOT / "direct_review_20260909a"
CELLS_PATH = OUTPUT_ROOT / "conditional_episode_cells.csv"
DECISIONS_PATH = OUTPUT_ROOT / "conditional_episode_decisions.csv"
RESULT_PATH = OUTPUT_ROOT / "conditional_episode_direct_result.json"

BTC_PAIR = "BTC/USDT:USDT"
ETH_PAIR = "ETH/USDT:USDT"
MARKET_PAIRS = tuple(simple.PAIRS)
HORIZONS = (1, 4, 8)
ORIGINAL_PERIODS = (
    "development_2021_2023",
    "walk_forward_validation_2024",
    "walk_forward_validation_2025",
    "untouched_confirmation_2026_jan_apr",
    "untouched_confirmation_2026_may_aug",
)
DECISION_PERIODS = (
    "development_2021_2023",
    "walk_forward_validation_2024",
    "walk_forward_validation_2025",
    "confirmation_2026",
)

FEATURE_COLUMNS = (
    "recent__relative_volume",
    "background__pair_return_720h",
    "background__realised_volatility_720h",
    "background__range_position_720h",
    "cross_market__positive_breadth_4h",
    "event_confluence__families_known_within_24h",
    "event_confluence__positive_signed_known_within_24h",
    "event_confluence__negative_signed_known_within_24h",
    "single_level__present",
    "single_level__nearest_distance_atr",
    "single_level__source_role_encoding",
    "single_level__source_timeframe_encoding",
    "single_level__pre_crossing_count_4h",
    "cluster__present",
    "cluster__independent_family_count",
    "cluster__nearest_distance_atr",
    "cluster__pre_crossing_count_4h",
)

CONDITION_SPECS: tuple[dict[str, Any], ...] = (
    {
        "route_id": "event_plus_market_readiness_and_background",
        "condition": "pre_event_volume_high",
        "outcomes": ("volume_reaction",),
        "control_match": "same_state",
        "role": "readiness_modifier",
    },
    {
        "route_id": "event_plus_market_readiness_and_background",
        "condition": "negative_slow_background",
        "outcomes": ("volume_reaction",),
        "control_match": "same_state",
        "role": "slow_background_modifier",
    },
    {
        "route_id": "event_plus_market_readiness_and_background",
        "condition": "high_slow_volatility",
        "outcomes": ("volume_reaction",),
        "control_match": "same_state",
        "role": "slow_background_modifier",
    },
    {
        "route_id": "event_plus_market_readiness_and_background",
        "condition": "outer_prior_range",
        "outcomes": ("volume_reaction",),
        "control_match": "same_state",
        "role": "market_location_modifier",
    },
    {
        "route_id": "event_plus_market_readiness_and_background",
        "condition": "broad_crypto_aligned",
        "outcomes": ("volume_reaction",),
        "control_match": "same_state",
        "role": "cross_market_background_modifier",
    },
    {
        "route_id": "distinct_event_and_narrative_accumulation",
        "condition": "multiple_known_families_24h",
        "outcomes": ("volume_reaction",),
        "control_match": "parent_clocks",
        "role": "distinct_event_accumulator",
    },
    {
        "route_id": "distinct_event_and_narrative_accumulation",
        "condition": "aligned_signed_accumulation",
        "outcomes": ("volume_reaction",),
        "control_match": "parent_clocks",
        "role": "signed_event_accumulator",
    },
    {
        "route_id": "distinct_event_and_narrative_accumulation",
        "condition": "conflicting_signed_accumulation",
        "outcomes": ("volume_reaction",),
        "control_match": "parent_clocks",
        "role": "signed_event_conflict",
    },
    {
        "route_id": "continuous_coin_local_level_and_cluster_modifier",
        "condition": "near_cluster",
        "outcomes": ("range_reaction", "volume_reaction"),
        "control_match": "same_state",
        "role": "local_cluster_distance_modifier",
        "eligibility": "cluster_present",
    },
    {
        "route_id": "continuous_coin_local_level_and_cluster_modifier",
        "condition": "richer_cluster",
        "outcomes": ("range_reaction",),
        "control_match": "same_state",
        "role": "local_cluster_composition_modifier",
        "eligibility": "cluster_present",
    },
    {
        "route_id": "continuous_coin_local_level_and_cluster_modifier",
        "condition": "isolated_single_level",
        "outcomes": ("range_reaction",),
        "control_match": "same_state",
        "role": "isolated_level_modifier",
    },
)


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON list.")
    return [str(item) for item in parsed]


def _verify_contract(contract: Mapping[str, Any]) -> None:
    path = Path(str(contract["path"]))
    if not path.is_file() or g0.sha256_file(path) != contract["sha256"]:
        raise ValueError(f"Frozen contract changed: {path}")


def load_freeze() -> dict[str, Any]:
    contract = _load_json(frozen.FREEZE_PATH)
    if (
        contract.get("status")
        != "frozen_conditional_episode_batch_before_new_outcome_analysis"
    ):
        raise ValueError("Conditional episode batch is not frozen.")
    for source in contract["source_contracts"].values():
        _verify_contract(source)
    return contract


def event_class(event_kinds_json: Any) -> str:
    kinds = set(_json_list(event_kinds_json))
    matched = [
        name
        for name, members in {
            "scheduled_policy_or_macro": {
                "scheduled_us_policy",
                "scheduled_us_macro",
                "scheduled_uk_macro",
                "scheduled_asia_macro",
            },
            "discrete_media_finance_or_corporate": {
                "unexpected_media_activity",
                "european_finance_news",
                "corporate_technology_news",
            },
            "daily_cross_market_state": {"daily_cross_market_state"},
        }.items()
        if kinds.intersection(members)
    ]
    if len(matched) == 1:
        return matched[0]
    if len(matched) > 1:
        return "mixed_event_classes"
    return "unclassified_event"


def event_scope_labels(event_kinds_json: Any, event_families_json: Any) -> list[str]:
    labels = ["all_catalogued_events", f"class:{event_class(event_kinds_json)}"]
    labels.extend(f"family:{family}" for family in _json_list(event_families_json))
    return list(dict.fromkeys(labels))


def _with_combined_2026(frame: DataFrame) -> DataFrame:
    current = frame.loc[
        frame["model_period"].astype(str).str.startswith("untouched_confirmation_2026")
    ].copy()
    current["model_period"] = "confirmation_2026"
    return pd.concat([frame, current], ignore_index=True)


def _cache_inventory() -> list[dict[str, Any]]:
    manifest = _load_json(frozen.CACHE_MANIFEST_PATH)
    inventory = [
        item for item in manifest["inventory"] if str(item["pair"]) in MARKET_PAIRS
    ]
    if {str(item["pair"]) for item in inventory} != set(MARKET_PAIRS):
        raise ValueError("The event cache does not contain all five market pairs.")
    for item in inventory:
        _verify_contract(
            {"path": item["feature_path"], "sha256": item["feature_sha256"]}
        )
        _verify_contract(
            {"path": item["evaluation_path"], "sha256": item["evaluation_sha256"]}
        )
    return inventory


def load_market_rows() -> tuple[DataFrame, dict[str, float]]:
    parts: list[DataFrame] = []
    volatility_thresholds: dict[str, float] = {}
    evaluation_columns = [
        "date",
        "period",
        "sample_id",
        "sample_kind",
        "parent_episode_ids_json",
        "event_families_json",
        "event_kinds_json",
    ]
    for horizon in HORIZONS:
        evaluation_columns.extend(
            [
                f"raw_volume_ratio_h{horizon}",
                f"raw_range_atr_h{horizon}",
                f"raw_close_return_atr_h{horizon}",
            ]
        )
    for item in _cache_inventory():
        pair = str(item["pair"])
        features = pd.read_parquet(
            item["feature_path"], columns=["date", *FEATURE_COLUMNS]
        )
        outcomes = pd.read_parquet(item["evaluation_path"], columns=evaluation_columns)
        features["date"] = pd.to_datetime(features["date"], utc=True)
        outcomes["date"] = pd.to_datetime(outcomes["date"], utc=True)
        joined = outcomes.merge(features, on="date", how="left", validate="many_to_one")
        joined = joined.rename(columns={"period": "model_period"})
        joined["pair"] = pair
        development_volatility = pd.to_numeric(
            joined.loc[
                joined["model_period"].eq("development_2021_2023"),
                "background__realised_volatility_720h",
            ],
            errors="coerce",
        ).dropna()
        threshold = float(development_volatility.quantile(0.75))
        volatility_thresholds[pair] = threshold
        joined["pre_event_volume_high"] = pd.to_numeric(
            joined["recent__relative_volume"], errors="coerce"
        ).ge(1.0)
        joined["negative_slow_background"] = pd.to_numeric(
            joined["background__pair_return_720h"], errors="coerce"
        ).lt(0.0)
        joined["high_slow_volatility"] = pd.to_numeric(
            joined["background__realised_volatility_720h"], errors="coerce"
        ).ge(threshold)
        range_position = pd.to_numeric(
            joined["background__range_position_720h"], errors="coerce"
        )
        joined["outer_prior_range"] = range_position.le(0.20) | range_position.ge(0.80)
        breadth = pd.to_numeric(
            joined["cross_market__positive_breadth_4h"], errors="coerce"
        )
        joined["broad_crypto_aligned"] = breadth.le(0.20) | breadth.ge(0.80)
        family_count = pd.to_numeric(
            joined["event_confluence__families_known_within_24h"], errors="coerce"
        )
        positive = pd.to_numeric(
            joined["event_confluence__positive_signed_known_within_24h"],
            errors="coerce",
        )
        negative = pd.to_numeric(
            joined["event_confluence__negative_signed_known_within_24h"],
            errors="coerce",
        )
        joined["multiple_known_families_24h"] = family_count.ge(2)
        joined["aligned_signed_accumulation"] = (
            (positive.ge(2) & negative.eq(0)) | (negative.ge(2) & positive.eq(0))
        )
        joined["conflicting_signed_accumulation"] = positive.ge(1) & negative.ge(1)
        joined["isolated_single_level"] = pd.to_numeric(
            joined["single_level__present"], errors="coerce"
        ).eq(1)
        joined["cluster_present"] = pd.to_numeric(
            joined["cluster__present"], errors="coerce"
        ).eq(1)
        joined["near_cluster"] = joined["cluster_present"] & pd.to_numeric(
            joined["cluster__nearest_distance_atr"], errors="coerce"
        ).le(0.10)
        joined["richer_cluster"] = joined["cluster_present"] & pd.to_numeric(
            joined["cluster__independent_family_count"], errors="coerce"
        ).ge(3)
        for horizon in HORIZONS:
            part = joined[
                [
                    "sample_id",
                    "sample_kind",
                    "date",
                    "model_period",
                    "pair",
                    "parent_episode_ids_json",
                    "event_families_json",
                    "event_kinds_json",
                    *[
                        str(spec["condition"])
                        for spec in CONDITION_SPECS
                        if str(spec["condition"]) in joined.columns
                    ],
                    "cluster_present",
                ]
            ].copy()
            part["horizon_hours"] = horizon
            volume = pd.to_numeric(
                joined[f"raw_volume_ratio_h{horizon}"], errors="coerce"
            )
            range_atr = pd.to_numeric(
                joined[f"raw_range_atr_h{horizon}"], errors="coerce"
            )
            close_move = pd.to_numeric(
                joined[f"raw_close_return_atr_h{horizon}"], errors="coerce"
            ).abs()
            part["volume_reaction"] = volume.ge(1.0).where(volume.notna())
            part["range_reaction"] = range_atr.ge(1.0).where(range_atr.notna())
            part["absolute_move_reaction"] = close_move.ge(1.0).where(
                close_move.notna()
            )
            parts.append(part)
    rows = pd.concat(parts, ignore_index=True)
    for outcome in ("volume_reaction", "range_reaction", "absolute_move_reaction"):
        rows[outcome] = pd.to_numeric(rows[outcome], errors="coerce")
    return _with_combined_2026(rows), volatility_thresholds


def _event_control_pairs(
    rows: DataFrame,
    *,
    condition: str,
    outcome: str,
    control_match: str,
    eligibility: str | None,
    anchor_selection: str,
) -> DataFrame:
    actual = rows.loc[rows["sample_kind"].eq("actual_event")].copy()
    controls = rows.loc[rows["sample_kind"].eq("matched_control")].copy()
    if eligibility:
        actual = actual.loc[actual[eligibility].astype(bool)]
        controls = controls.loc[controls[eligibility].astype(bool)]
    actual["episode_id"] = actual["parent_episode_ids_json"].map(_json_list)
    actual = actual.explode("episode_id")
    actual = actual.sort_values(
        ["model_period", "episode_id", "pair", "horizon_hours", "date", "sample_id"]
    ).drop_duplicates(
        ["model_period", "episode_id", "pair", "horizon_hours"],
        keep="last" if anchor_selection == "latest" else "first",
    )
    actual["event_scope"] = actual.apply(
        lambda row: event_scope_labels(
            row["event_kinds_json"], row["event_families_json"]
        ),
        axis=1,
    )
    controls["episode_id"] = controls["parent_episode_ids_json"].map(_json_list)
    controls = controls.explode("episode_id")
    actual = actual.rename(
        columns={
            condition: "actual_state",
            outcome: "actual_outcome",
        }
    )
    controls = controls.rename(
        columns={
            "sample_id": "control_sample_id",
            condition: "control_state",
            outcome: "control_outcome",
        }
    )
    merged = actual[
        [
            "sample_id",
            "episode_id",
            "event_scope",
            "model_period",
            "pair",
            "horizon_hours",
            "actual_state",
            "actual_outcome",
        ]
    ].merge(
        controls[
            [
                "control_sample_id",
                "episode_id",
                "model_period",
                "pair",
                "horizon_hours",
                "control_state",
                "control_outcome",
            ]
        ],
        on=["episode_id", "model_period", "pair", "horizon_hours"],
        how="left",
        validate="one_to_many",
    )
    merged = merged.explode("event_scope")
    merged = merged.loc[
        merged["actual_outcome"].notna() & merged["control_outcome"].notna()
    ].copy()
    if control_match == "same_state":
        merged = merged.loc[merged["actual_state"].eq(merged["control_state"])]
    return merged


def modifier_cells(rows: DataFrame, spec: Mapping[str, Any], outcome: str) -> DataFrame:
    condition = str(spec["condition"])
    paired = _event_control_pairs(
        rows,
        condition=condition,
        outcome=outcome,
        control_match=str(spec["control_match"]),
        eligibility=(str(spec["eligibility"]) if spec.get("eligibility") else None),
        anchor_selection=(
            "latest"
            if str(spec["route_id"])
            == "distinct_event_and_narrative_accumulation"
            else "earliest"
        ),
    )
    event_keys = [
        "sample_id",
        "episode_id",
        "event_scope",
        "model_period",
        "pair",
        "horizon_hours",
        "actual_state",
    ]
    per_event = (
        paired.groupby(event_keys, observed=True, sort=False)
        .agg(
            actual_outcome=("actual_outcome", "first"),
            control_outcome=("control_outcome", "mean"),
            matched_control_count=("control_sample_id", "nunique"),
        )
        .reset_index()
    )
    per_event["event_excess"] = (
        per_event["actual_outcome"] - per_event["control_outcome"]
    )
    keys = ["event_scope", "model_period", "pair", "horizon_hours"]
    records: list[dict[str, Any]] = []
    for values, group in per_event.groupby(keys, observed=True, sort=False):
        by_state: dict[bool, DataFrame] = {
            state: group.loc[group["actual_state"].eq(state)] for state in (False, True)
        }
        false, true = by_state[False], by_state[True]
        records.append(
            {
                "route_id": str(spec["route_id"]),
                "role": str(spec["role"]),
                "condition": condition,
                "outcome": outcome,
                "control_match": str(spec["control_match"]),
                **dict(zip(keys, values, strict=True)),
                "false_event_count": int(false["episode_id"].nunique()),
                "true_event_count": int(true["episode_id"].nunique()),
                "false_actual_rate": float(false["actual_outcome"].mean()),
                "true_actual_rate": float(true["actual_outcome"].mean()),
                "false_control_rate": float(false["control_outcome"].mean()),
                "true_control_rate": float(true["control_outcome"].mean()),
                "false_event_excess": float(false["event_excess"].mean()),
                "true_event_excess": float(true["event_excess"].mean()),
                "conditional_effect": float(
                    true["event_excess"].mean() - false["event_excess"].mean()
                ),
                "median_controls_per_false_event": float(
                    false["matched_control_count"].median()
                ),
                "median_controls_per_true_event": float(
                    true["matched_control_count"].median()
                ),
            }
        )
    return DataFrame.from_records(records)


def _meme_artifacts() -> tuple[DataFrame, DataFrame]:
    result = _load_json(frozen.MEME_RESULT_PATH)
    for contract in result["artifacts"].values():
        _verify_contract(contract)
    pairs = pd.read_parquet(result["artifacts"]["pair_details"]["path"])
    groups = pd.read_parquet(result["artifacts"]["group_details"]["path"])
    return pairs, groups


def load_transmission_rows(market_rows: DataFrame) -> DataFrame:
    pairs, groups = _meme_artifacts()
    base_pairs = pairs.loc[pairs["horizon_hours"].eq(1)].copy()
    base_pairs["early_agrees_with_btc"] = np.sign(
        pd.to_numeric(base_pairs["event_log_return"], errors="coerce")
    ).eq(pd.to_numeric(base_pairs["btc_initial_sign"], errors="coerce"))
    group_members = {
        "established_alts": ["ETH/USDT:USDT", "BNB/USDT:USDT", "ADA/USDT:USDT", "TRX/USDT:USDT"],
        "memes": sorted(
            set(base_pairs["pair"].dropna().astype(str))
            - set(MARKET_PAIRS)
        ),
    }
    initial_parts: list[DataFrame] = []
    for cohort, members in group_members.items():
        part = base_pairs.loc[base_pairs["pair"].isin(members)].copy()
        initial = (
            part.groupby("sample_id", observed=True, sort=False)
            .agg(
                early_agreement_fraction=("early_agrees_with_btc", "mean"),
                early_member_count=("pair", "nunique"),
            )
            .reset_index()
        )
        initial["cohort"] = cohort
        initial["early_cohort_majority_agrees"] = initial[
            "early_agreement_fraction"
        ].ge(0.60)
        initial_parts.append(initial)
    initial = pd.concat(initial_parts, ignore_index=True)

    eth = base_pairs.loc[base_pairs["pair"].eq(ETH_PAIR), [
        "sample_id",
        "early_agrees_with_btc",
    ]].drop_duplicates("sample_id")
    eth = eth.rename(columns={"early_agrees_with_btc": "eth_early_agrees"})

    btc_context = market_rows.loc[
        market_rows["pair"].eq(BTC_PAIR)
        & market_rows["horizon_hours"].eq(1)
        & market_rows["model_period"].isin(ORIGINAL_PERIODS),
        ["sample_id", "pre_event_volume_high", "negative_slow_background"],
    ].drop_duplicates("sample_id")
    output = groups.merge(
        initial,
        on=["sample_id", "cohort"],
        how="left",
        validate="many_to_one",
    )
    output = output.merge(eth, on="sample_id", how="left", validate="many_to_one")
    output = output.merge(
        btc_context,
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    output["later_direction_aligned"] = pd.to_numeric(
        output["later_direction_aligned"], errors="coerce"
    )
    return _with_combined_2026(output)


def transmission_cells(
    rows: DataFrame,
    *,
    route_id: str,
    cohort: str,
    condition: str,
) -> DataFrame:
    selected = rows.loc[
        rows["cohort"].eq(cohort)
        & rows["complete_cohort"].fillna(False).astype(bool)
        & rows["btc_confirmed_reaction"].fillna(False).astype(bool)
        & rows["later_direction_aligned"].notna()
    ].copy()
    actual = selected.loc[selected["sample_kind"].eq("actual_event")].copy()
    controls = selected.loc[selected["sample_kind"].eq("matched_control")].copy()
    actual["event_scope"] = actual.apply(
        lambda row: event_scope_labels(
            row["event_kinds_json"], row["event_families_json"]
        ),
        axis=1,
    )
    actual = actual.rename(
        columns={condition: "actual_state", "later_direction_aligned": "actual_outcome"}
    )
    controls = controls.rename(
        columns={
            "sample_id": "control_sample_id",
            condition: "control_state",
            "later_direction_aligned": "control_outcome",
        }
    )
    merged = actual[
        [
            "sample_id",
            "episode_id",
            "event_scope",
            "model_period",
            "horizon_hours",
            "actual_state",
            "actual_outcome",
        ]
    ].merge(
        controls[
            [
                "control_sample_id",
                "episode_id",
                "model_period",
                "horizon_hours",
                "control_state",
                "control_outcome",
            ]
        ],
        on=["episode_id", "model_period", "horizon_hours"],
        how="left",
        validate="one_to_many",
    )
    merged = merged.explode("event_scope")
    merged = merged.loc[
        merged["control_outcome"].notna()
        & merged["actual_state"].eq(merged["control_state"])
    ].copy()
    keys = [
        "sample_id",
        "episode_id",
        "event_scope",
        "model_period",
        "horizon_hours",
        "actual_state",
    ]
    per_event = (
        merged.groupby(keys, observed=True, sort=False)
        .agg(
            actual_outcome=("actual_outcome", "first"),
            control_outcome=("control_outcome", "mean"),
            matched_control_count=("control_sample_id", "nunique"),
        )
        .reset_index()
    )
    per_event["event_excess"] = (
        per_event["actual_outcome"] - per_event["control_outcome"]
    )
    records: list[dict[str, Any]] = []
    summary_keys = ["event_scope", "model_period", "horizon_hours"]
    for values, group in per_event.groupby(summary_keys, observed=True, sort=False):
        false = group.loc[~group["actual_state"].astype(bool)]
        true = group.loc[group["actual_state"].astype(bool)]
        records.append(
            {
                "route_id": route_id,
                "role": "leader_confirmation_and_transmission",
                "condition": condition,
                "outcome": "later_direction_aligned_with_initial_btc",
                "control_match": "same_state_confirmed_btc_parent_clocks",
                **dict(zip(summary_keys, values, strict=True)),
                "pair": cohort,
                "false_event_count": int(false["episode_id"].nunique()),
                "true_event_count": int(true["episode_id"].nunique()),
                "false_actual_rate": float(false["actual_outcome"].mean()),
                "true_actual_rate": float(true["actual_outcome"].mean()),
                "false_control_rate": float(false["control_outcome"].mean()),
                "true_control_rate": float(true["control_outcome"].mean()),
                "false_event_excess": float(false["event_excess"].mean()),
                "true_event_excess": float(true["event_excess"].mean()),
                "conditional_effect": float(
                    true["event_excess"].mean() - false["event_excess"].mean()
                ),
                "median_controls_per_false_event": float(
                    false["matched_control_count"].median()
                ),
                "median_controls_per_true_event": float(
                    true["matched_control_count"].median()
                ),
            }
        )
    return DataFrame.from_records(records)


def decide(cells: DataFrame) -> DataFrame:
    keys = [
        "route_id",
        "role",
        "condition",
        "outcome",
        "control_match",
        "event_scope",
        "pair",
        "horizon_hours",
    ]
    records: list[dict[str, Any]] = []
    for values, group in cells.groupby(keys, observed=True, sort=False):
        decision_rows = group.loc[group["model_period"].isin(DECISION_PERIODS)].copy()
        eligible = decision_rows.loc[
            decision_rows["true_event_count"].ge(frozen.MINIMUM_EVENTS_PER_STATE)
            & decision_rows["false_event_count"].ge(frozen.MINIMUM_EVENTS_PER_STATE)
            & decision_rows["conditional_effect"].notna()
        ]
        material = eligible.loc[
            eligible["conditional_effect"].abs().ge(
                frozen.MINIMUM_CONDITIONAL_EFFECT
            )
        ]
        positive_periods = int(material["conditional_effect"].gt(0).sum())
        negative_periods = int(material["conditional_effect"].lt(0).sum())
        compatible_material_periods = max(positive_periods, negative_periods)
        conflicting_material_periods = min(positive_periods, negative_periods)
        eligible_period_order = {
            period: index for index, period in enumerate(DECISION_PERIODS)
        }
        latest = (
            eligible.assign(
                _period_order=eligible["model_period"].map(eligible_period_order)
            )
            .sort_values("_period_order")
            .tail(1)
        )
        latest_period = str(latest.iloc[0]["model_period"]) if len(latest) else ""
        latest_effect = (
            float(latest.iloc[0]["conditional_effect"]) if len(latest) else np.nan
        )
        if len(eligible) < 2:
            verdict = "coverage_parked_fewer_than_two_supported_periods"
        elif conflicting_material_periods:
            verdict = "unresolved_effect_changes_sign_between_periods"
        elif compatible_material_periods < 2:
            verdict = "not_retained_conditional_effect_not_repeated"
        else:
            verdict = "retained_retrospective_modifier_lead_needs_future_events"
        records.append(
            {
                **dict(zip(keys, values, strict=True)),
                "supported_period_count": len(eligible),
                "material_positive_period_count": positive_periods,
                "material_negative_period_count": negative_periods,
                "compatible_material_period_count": compatible_material_periods,
                "conflicting_material_period_count": conflicting_material_periods,
                "median_conditional_effect_supported_periods": float(
                    eligible["conditional_effect"].median()
                ),
                "latest_supported_period": latest_period,
                "latest_conditional_effect": latest_effect,
                "maximum_true_state_event_count": int(
                    decision_rows["true_event_count"].max()
                ),
                "maximum_false_state_event_count": int(
                    decision_rows["false_event_count"].max()
                ),
                "verdict": verdict,
            }
        )
    return DataFrame.from_records(records)


def route_summary(decisions: DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sibling in frozen.SIBLINGS:
        route_id = str(sibling["route_id"])
        cells = decisions.loc[decisions["route_id"].eq(route_id)]
        if route_id == "true_expectation_surprise":
            status = "coverage_parked_missing_timestamped_expectation"
        elif cells.empty or cells["verdict"].str.startswith("coverage_parked").all():
            status = "coverage_parked_current_rows"
        elif cells["verdict"].str.startswith("retained").any():
            status = "retrospective_conditional_leads_found"
        elif cells["verdict"].str.startswith("unresolved").any():
            status = "unresolved_period_dependent_effects_found"
        else:
            status = "no_repeated_conditional_lead_current_definition"
        records.append(
            {
                "route_id": route_id,
                "status": status,
                "decision_cells": len(cells),
                "retained_cells": int(cells["verdict"].str.startswith("retained").sum()),
                "unresolved_cells": int(
                    cells["verdict"].str.startswith("unresolved").sum()
                ),
                "coverage_parked_cells": int(
                    cells["verdict"].str.startswith("coverage_parked").sum()
                ),
            }
        )
    return records


def run(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        existing = _load_json(RESULT_PATH)
        if existing.get("status") != "completed_conditional_episode_direct_batch":
            raise ValueError("Existing direct result is not terminal.")
        return existing
    contract = load_freeze()
    market_rows, volatility_thresholds = load_market_rows()
    cell_parts: list[DataFrame] = []
    for spec in CONDITION_SPECS:
        for outcome in spec["outcomes"]:
            cell_parts.append(modifier_cells(market_rows, spec, str(outcome)))

    transmission = load_transmission_rows(market_rows)
    for condition in (
        "eth_early_agrees",
        "early_cohort_majority_agrees",
        "pre_event_volume_high",
    ):
        cell_parts.append(
            transmission_cells(
                transmission,
                route_id="initial_btc_eth_to_established_transmission",
                cohort="established_alts",
                condition=condition,
            )
        )
    for condition in (
        "pre_event_volume_high",
        "negative_slow_background",
        "early_cohort_majority_agrees",
    ):
        cell_parts.append(
            transmission_cells(
                transmission,
                route_id="verified_btc_to_meme_transmission",
                cohort="memes",
                condition=condition,
            )
        )
    cells = pd.concat(cell_parts, ignore_index=True)
    decisions = decide(cells)
    summaries = route_summary(decisions)
    g0.atomic_write_csv(cells, CELLS_PATH)
    g0.atomic_write_csv(decisions, DECISIONS_PATH)
    result = {
        "schema_version": 1,
        "status": "completed_conditional_episode_direct_batch",
        "created_at_utc": g0.utc_now(),
        "run_id": frozen.RUN_ID,
        "scope": {
            "retrospective_development_only": True,
            "new_freqai_model_fitted": False,
            "profit_used": False,
            "trading_rule_tested": False,
            "branches_launched": False,
        },
        "plain_objective": contract["plain_objective"],
        "route_summary": summaries,
        "volatility_thresholds_from_2021_2023_feature_distribution": (
            volatility_thresholds
        ),
        "interpretation_limits": [
            "A retained cell is a retrospective conditional lead not confirmation.",
            "A coverage result says the present rows cannot answer the question.",
            "A null applies only to the tested role scope horizon and representation.",
            "Correlated BTC and ETH results count as one connected market response.",
            "No descendant branch may start until this complete batch is jointly reviewed.",
        ],
        "source_contracts": {
            "analysis_script": artifact(ANALYSIS_PATH),
            "batch_freeze": artifact(frozen.FREEZE_PATH),
        },
        "artifacts": {
            "conditional_cells": artifact(CELLS_PATH),
            "decisions": artifact(DECISIONS_PATH),
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

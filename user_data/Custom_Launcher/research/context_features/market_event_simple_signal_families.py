"""Freeze and compare readable activity prototypes for all five signal families."""

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
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
RUN_ID = "event_simple_signal_families_20260909a"
SOURCE_ROOT = (
    REPO_ROOT
    / "user_data"
    / "research_news_data"
    / "context_features"
    / "event_hierarchy"
    / "event_signal_fresh_2026_20260909a"
)
SOURCE_FREEZE_PATH = SOURCE_ROOT / "event_signal_fresh_freeze.json"
SOURCE_SAMPLES_PATH = SOURCE_ROOT / "event_signal_fresh_sample_catalog.csv"
SOURCE_CACHE_MANIFEST_PATH = (
    SOURCE_ROOT / "freqai_cache" / "event_signal_fresh_cache_manifest.json"
)
OUTPUT_ROOT = SOURCE_ROOT / RUN_ID
DETAIL_ROOT = (
    Path(r"D:\FreqTradeStuffLargeData")
    / "research_outputs"
    / "event_hierarchy"
    / RUN_ID
)
FREEZE_PATH = OUTPUT_ROOT / "simple_signal_family_freeze.json"
CALLS_PATH = DETAIL_ROOT / "simple_signal_family_calls.parquet"
PAIR_SCORES_PATH = OUTPUT_ROOT / "simple_signal_family_pair_scores.csv"
SCOPE_SCORES_PATH = OUTPUT_ROOT / "simple_signal_family_scope_scores.csv"
DECISIONS_PATH = OUTPUT_ROOT / "simple_signal_family_decisions.csv"
RESULT_PATH = OUTPUT_ROOT / "simple_signal_family_result.json"

PAIRS = (
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "BNB/USDT:USDT",
    "ADA/USDT:USDT",
    "TRX/USDT:USDT",
)
PERIODS = (
    "walk_forward_validation_2024",
    "walk_forward_validation_2025",
    "untouched_confirmation_2026_jan_apr",
    "untouched_confirmation_2026_may_aug",
)
DECISION_PERIODS = (
    "walk_forward_validation_2024",
    "walk_forward_validation_2025",
    "confirmation_2026",
)
HORIZONS = (1, 4, 8)
LEAD_RATE = 0.55
STRONG_RATE = 0.65
MINIMUM_UPLIFT = 0.02
MINIMUM_UNITS = 20

SIGNALS: tuple[dict[str, Any], ...] = (
    {
        "signal_id": "catalogued_event_clock",
        "family_id": "major_event_information",
        "feature": "event_identity__is_actual_event",
        "operator": "event_or_matched_reference",
        "threshold": None,
        "reason": "A catalogued event or cross-market shock marks a time when activity may change.",
        "limitation": "This supplies timing only; it does not supply signed direction.",
    },
    {
        "signal_id": "negative_slow_background",
        "family_id": "background_and_market_leadership",
        "feature": "background__pair_return_720h",
        "operator": "less_than",
        "threshold": 0.0,
        "reason": "The coin entered the event after a negative thirty-day market background.",
        "limitation": "A negative background is context, not an automatic short call.",
    },
    {
        "signal_id": "single_calculated_area",
        "family_id": "calculated_reaction_areas",
        "feature": "single_level__present",
        "operator": "boolean_true",
        "threshold": True,
        "reason": "Price is near one independently calculated reaction area.",
        "limitation": "Area presence predicts possible traffic, not bounce or breakout direction.",
    },
    {
        "signal_id": "calculated_area_cluster",
        "family_id": "calculated_reaction_areas",
        "feature": "cluster__present",
        "operator": "boolean_true",
        "threshold": True,
        "reason": "Several independent calculated reaction areas overlap nearby.",
        "limitation": "A cluster can attract, reject, or break; direction remains unknown.",
    },
    {
        "signal_id": "recent_volume_persistence",
        "family_id": "local_participation_and_pressure",
        "feature": "recent__relative_volume",
        "operator": "greater_than_or_equal",
        "threshold": 1.0,
        "reason": "The last completed hour's volume is at least its causal seven-day median.",
        "limitation": "This is a general busy-market warning and is not event-specific.",
    },
    {
        "signal_id": "broad_crypto_directional_alignment",
        "family_id": "cross_asset_transmission_and_amplification",
        "feature": "cross_market__positive_breadth_4h",
        "operator": "outside_open_interval",
        "threshold": [0.20, 0.80],
        "reason": "At least four of five established crypto markets moved in the same direction.",
        "limitation": "Alignment describes participation; it does not prove which market led.",
    },
)

SCOPE_MEMBERS: Mapping[str, tuple[str, ...]] = {
    "btc": ("BTC/USDT:USDT",),
    "established_alts": (
        "ETH/USDT:USDT",
        "BNB/USDT:USDT",
        "ADA/USDT:USDT",
        "TRX/USDT:USDT",
    ),
    "all_five": PAIRS,
}
SCOPE_REQUIRED_FRACTION = {"btc": 1.0, "established_alts": 0.75, "all_five": 0.80}


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_bool(series: Series) -> Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_present = series.notna() & numeric.notna()
    if numeric_present.any() and numeric.loc[numeric_present].isin([0.0, 1.0]).all():
        return numeric.eq(1.0).where(series.notna(), False)
    values = series.astype("string").str.strip().str.lower()
    present = values.notna()
    if (present & ~values.isin(["true", "false", "1", "0"])).any():
        raise ValueError("Feature contains an unexpected boolean representation.")
    return values.isin(["true", "1"]).where(present, False)


def issue_signal(feature: Series, signal: Mapping[str, Any]) -> Series:
    operator = str(signal["operator"])
    if operator == "event_or_matched_reference":
        return Series(True, index=feature.index)
    if operator == "less_than":
        return pd.to_numeric(feature, errors="coerce").lt(float(signal["threshold"]))
    if operator == "greater_than_or_equal":
        return pd.to_numeric(feature, errors="coerce").ge(float(signal["threshold"]))
    if operator == "boolean_true":
        return _coerce_bool(feature)
    if operator == "outside_open_interval":
        lower, upper = signal["threshold"]
        numeric = pd.to_numeric(feature, errors="coerce")
        return numeric.le(float(lower)) | numeric.ge(float(upper))
    raise ValueError(f"Unsupported signal operator: {operator}")


def _sample_units() -> DataFrame:
    samples = pd.read_csv(SOURCE_SAMPLES_PATH)
    samples["model_anchor_utc"] = pd.to_datetime(samples["model_anchor_utc"], utc=True)
    samples = samples.loc[samples["model_period"].isin(PERIODS)].copy()
    samples["episode_ids"] = samples["parent_episode_ids_json"].map(json.loads)
    actual = samples.loc[samples["sample_kind"].eq("actual_event")].explode(
        "episode_ids"
    )
    actual = actual.sort_values(["episode_ids", "model_anchor_utc"]).drop_duplicates(
        "episode_ids", keep="first"
    )
    actual["analysis_unit_id"] = actual["episode_ids"].astype(str)
    controls = samples.loc[samples["sample_kind"].eq("matched_control")].drop_duplicates(
        "sample_id", keep="first"
    )
    controls["analysis_unit_id"] = controls["sample_id"]
    return pd.concat([actual, controls], ignore_index=True)[
        [
            "sample_id",
            "sample_kind",
            "analysis_unit_id",
            "model_anchor_utc",
            "model_period",
            "parent_episode_ids_json",
            "event_families_json",
            "event_kinds_json",
        ]
    ]


def _cache_inventory() -> dict[str, dict[str, Any]]:
    manifest = _load_json(SOURCE_CACHE_MANIFEST_PATH)
    if manifest.get("status") != "completed_event_freqai_cache":
        raise ValueError("The source feature cache is not terminal.")
    inventory = {str(item["pair"]): dict(item) for item in manifest["inventory"]}
    if set(PAIRS) != set(inventory):
        raise ValueError("The source feature cache does not match the five frozen pairs.")
    for pair, item in inventory.items():
        for path_key, hash_key in (
            ("feature_path", "feature_sha256"),
            ("evaluation_path", "evaluation_sha256"),
        ):
            path = Path(item[path_key])
            if g0.sha256_file(path) != item[hash_key]:
                raise ValueError(f"Frozen cache changed for {pair}: {path_key}")
    return inventory


def _build_call_catalog(
    samples: DataFrame, inventory: Mapping[str, Mapping[str, Any]]
) -> DataFrame:
    feature_names = sorted({str(signal["feature"]) for signal in SIGNALS})
    parts: list[DataFrame] = []
    for pair in PAIRS:
        features = pd.read_parquet(
            Path(inventory[pair]["feature_path"]), columns=["date", *feature_names]
        )
        features["date"] = pd.to_datetime(features["date"], utc=True)
        joined = samples.merge(
            features,
            left_on="model_anchor_utc",
            right_on="date",
            how="left",
            validate="many_to_one",
        ).drop(columns="date")
        for signal in SIGNALS:
            feature_name = str(signal["feature"])
            issued = issue_signal(joined[feature_name], signal)
            part = joined[
                [
                    "sample_id",
                    "sample_kind",
                    "analysis_unit_id",
                    "model_anchor_utc",
                    "model_period",
                    "parent_episode_ids_json",
                    "event_families_json",
                    "event_kinds_json",
                ]
            ].copy()
            part["pair"] = pair
            part["signal_id"] = signal["signal_id"]
            part["family_id"] = signal["family_id"]
            part["feature_name"] = feature_name
            part["feature_value"] = pd.to_numeric(
                joined[feature_name], errors="coerce"
            )
            part["signal_issued"] = issued.where(joined[feature_name].notna(), False)
            part["observation_utc"] = part["model_anchor_utc"]
            part["latest_contributing_data_utc"] = (
                part["model_anchor_utc"]
                if signal["signal_id"] == "catalogued_event_clock"
                else part["model_anchor_utc"] - pd.Timedelta(hours=1)
            )
            part["meaning"] = "activity_only"
            part["reason"] = signal["reason"]
            part["limitation"] = signal["limitation"]
            part["call_state"] = np.where(
                part["signal_issued"],
                np.where(
                    part["sample_kind"].eq("matched_control")
                    & part["signal_id"].eq("catalogued_event_clock"),
                    "matched_reference",
                    "issued",
                ),
                "abstain_condition_absent",
            )
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def freeze() -> dict[str, Any]:
    if FREEZE_PATH.is_file():
        existing = _load_json(FREEZE_PATH)
        if existing.get("status") != "frozen_simple_signal_families_before_outcomes":
            raise ValueError("Existing simple-family freeze is not terminal.")
        if existing["source_contracts"]["analysis_script"]["sha256"] != g0.sha256_file(
            ANALYSIS_PATH
        ):
            raise ValueError("Analysis script changed after the simple-family freeze.")
        return existing
    parent = _load_json(SOURCE_FREEZE_PATH)
    if parent.get("status") != "frozen_event_signal_fresh_2026_before_outcomes":
        raise ValueError("Parent event sample definition is not frozen.")
    if parent["frozen_artifacts"]["samples"]["sha256"] != g0.sha256_file(
        SOURCE_SAMPLES_PATH
    ):
        raise ValueError("Parent event samples changed.")
    inventory = _cache_inventory()
    samples = _sample_units()
    calls = _build_call_catalog(samples, inventory)
    represented = {str(item["family_id"]) for item in SIGNALS}
    if len(represented) != 5:
        raise ValueError("The simple prototype batch must cover exactly five families.")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    DETAIL_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(calls, CALLS_PATH)
    result = {
        "schema_version": 1,
        "status": "frozen_simple_signal_families_before_outcomes",
        "created_at_utc": g0.utc_now(),
        "run_id": RUN_ID,
        "plain_objective": (
            "Compare one readable activity representation from each approved signal "
            "family, plus separate single-area and cluster variants, before any new "
            "family combination is tested."
        ),
        "evidence_state": (
            "Retrospective portable-rule screening on periods whose broad outcomes have "
            "already been inspected; this cannot provide fresh confirmation."
        ),
        "signals": [dict(item) for item in SIGNALS],
        "family_count": len(represented),
        "signal_count": len(SIGNALS),
        "pairs": list(PAIRS),
        "periods": list(PERIODS),
        "decision_periods": list(DECISION_PERIODS),
        "horizons": list(HORIZONS),
        "primary_outcome": (
            "future average hourly volume is above the causal seven-day median "
            "available at the observation time"
        ),
        "secondary_descriptions": [
            "median future volume ratio",
            "median future price range measured in prior ATR",
            "median absolute close movement measured in prior ATR",
        ],
        "decision_rule": {
            "minimum_units_each_decision_period": MINIMUM_UNITS,
            "minimum_volume_reaction_rate": LEAD_RATE,
            "main_target_rate": STRONG_RATE,
            "minimum_uplift_over_no_call_or_matched_clock": MINIMUM_UPLIFT,
            "required_member_fraction_by_scope": SCOPE_REQUIRED_FRACTION,
            "controls": (
                "Each non-event-clock signal faces its no-signal rows and the same "
                "signal at matched control times. Event clock faces matched clocks."
            ),
        },
        "sample_support": {
            "actual_units": int(
                samples.loc[samples["sample_kind"].eq("actual_event"), "analysis_unit_id"].nunique()
            ),
            "control_units": int(
                samples.loc[
                    samples["sample_kind"].eq("matched_control"), "analysis_unit_id"
                ].nunique()
            ),
        },
        "scope": {
            "direction_tested": False,
            "profit_used": False,
            "freqai_used": False,
            "meme_coins_tested": False,
            "trading_rule_tested": False,
        },
        "source_contracts": {
            "analysis_script": artifact(ANALYSIS_PATH),
            "parent_freeze": artifact(SOURCE_FREEZE_PATH),
            "parent_samples": artifact(SOURCE_SAMPLES_PATH),
            "cache_manifest": artifact(SOURCE_CACHE_MANIFEST_PATH),
            "call_catalog": artifact(CALLS_PATH),
            "feature_caches": {
                pair: {
                    "path": item["feature_path"],
                    "sha256": item["feature_sha256"],
                }
                for pair, item in inventory.items()
            },
            "evaluation_caches": {
                pair: {
                    "path": item["evaluation_path"],
                    "sha256": item["evaluation_sha256"],
                }
                for pair, item in inventory.items()
            },
        },
        "future_outcomes_read_by_freeze": False,
    }
    g0.atomic_write_json(result, FREEZE_PATH)
    return result


def _validate_contracts(frozen: Mapping[str, Any]) -> None:
    contracts = frozen["source_contracts"]
    for name in (
        "analysis_script",
        "parent_freeze",
        "parent_samples",
        "cache_manifest",
        "call_catalog",
    ):
        contract = contracts[name]
        if g0.sha256_file(Path(contract["path"])) != contract["sha256"]:
            raise ValueError(f"Frozen source changed: {name}")
    for kind in ("feature_caches", "evaluation_caches"):
        for pair, contract in contracts[kind].items():
            if g0.sha256_file(Path(contract["path"])) != contract["sha256"]:
                raise ValueError(f"Frozen {kind} source changed for {pair}.")


def _outcome_rows(calls: DataFrame, frozen: Mapping[str, Any]) -> DataFrame:
    parts: list[DataFrame] = []
    for pair in PAIRS:
        evaluation_path = Path(
            frozen["source_contracts"]["evaluation_caches"][pair]["path"]
        )
        outcome_columns = ["date", "sample_id"]
        for horizon in HORIZONS:
            outcome_columns.extend(
                [
                    f"raw_volume_ratio_h{horizon}",
                    f"raw_range_atr_h{horizon}",
                    f"raw_close_return_atr_h{horizon}",
                ]
            )
        outcomes = pd.read_parquet(evaluation_path, columns=outcome_columns)
        outcomes["date"] = pd.to_datetime(outcomes["date"], utc=True)
        source = calls.loc[calls["pair"].eq(pair)].merge(
            outcomes,
            left_on=["model_anchor_utc", "sample_id"],
            right_on=["date", "sample_id"],
            how="left",
            validate="many_to_one",
        ).drop(columns="date")
        for horizon in HORIZONS:
            part = source[
                [
                    "sample_id",
                    "sample_kind",
                    "analysis_unit_id",
                    "model_anchor_utc",
                    "model_period",
                    "pair",
                    "signal_id",
                    "family_id",
                    "signal_issued",
                ]
            ].copy()
            part["horizon_hours"] = horizon
            part["volume_ratio"] = source[f"raw_volume_ratio_h{horizon}"]
            part["range_atr"] = source[f"raw_range_atr_h{horizon}"]
            part["absolute_close_move_atr"] = source[
                f"raw_close_return_atr_h{horizon}"
            ].abs()
            part["volume_reaction"] = part["volume_ratio"].ge(1).where(
                part["volume_ratio"].notna()
            )
            parts.append(part)
    output = pd.concat(parts, ignore_index=True)
    current = output.loc[
        output["model_period"].str.startswith("untouched_confirmation_2026")
    ].copy()
    current["model_period"] = "confirmation_2026"
    return pd.concat([output, current], ignore_index=True)


def score_pairs(rows: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    keys = ["signal_id", "family_id", "pair", "sample_kind", "model_period", "horizon_hours"]
    for values, group in rows.groupby(keys, observed=True, sort=False):
        eligible = group.loc[group["volume_reaction"].notna()]
        calls = eligible.loc[eligible["signal_issued"]]
        noncalls = eligible.loc[~eligible["signal_issued"]]
        call_rate = calls["volume_reaction"].mean()
        noncall_rate = noncalls["volume_reaction"].mean()
        records.append(
            {
                **dict(zip(keys, values, strict=True)),
                "eligible_units": eligible["analysis_unit_id"].nunique(),
                "call_units": calls["analysis_unit_id"].nunique(),
                "noncall_units": noncalls["analysis_unit_id"].nunique(),
                "volume_reaction_rate_calls": call_rate,
                "volume_reaction_rate_noncalls": noncall_rate,
                "call_minus_noncall": call_rate - noncall_rate,
                "median_volume_ratio_calls": calls["volume_ratio"].median(),
                "median_range_atr_calls": calls["range_atr"].median(),
                "median_absolute_close_move_atr_calls": calls[
                    "absolute_close_move_atr"
                ].median(),
            }
        )
    return DataFrame.from_records(records)


def add_scopes(pair_scores: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    group_keys = ["signal_id", "family_id", "sample_kind", "model_period", "horizon_hours"]
    metric_columns = [
        "eligible_units",
        "call_units",
        "noncall_units",
        "volume_reaction_rate_calls",
        "volume_reaction_rate_noncalls",
        "call_minus_noncall",
        "median_volume_ratio_calls",
        "median_range_atr_calls",
        "median_absolute_close_move_atr_calls",
    ]
    for scope, members in SCOPE_MEMBERS.items():
        subset = pair_scores.loc[pair_scores["pair"].isin(members)]
        for values, group in subset.groupby(group_keys, observed=True, sort=False):
            if set(group["pair"]) != set(members):
                continue
            record = {**dict(zip(group_keys, values, strict=True)), "market_scope": scope}
            record["member_count"] = len(members)
            for column in metric_columns:
                numeric = pd.to_numeric(group[column], errors="coerce")
                record[column] = (
                    int(numeric.min()) if column.endswith("_units") else float(numeric.mean())
                )
            records.append(record)
    return DataFrame.from_records(records)


def _score_lookup(
    scores: DataFrame,
    *,
    signal_id: str,
    pair: str,
    sample_kind: str,
    period: str,
    horizon: int,
) -> Series | None:
    found = scores.loc[
        scores["signal_id"].eq(signal_id)
        & scores["pair"].eq(pair)
        & scores["sample_kind"].eq(sample_kind)
        & scores["model_period"].eq(period)
        & scores["horizon_hours"].eq(horizon)
    ]
    return None if found.empty else found.iloc[0]


def decide(pair_scores: DataFrame, scope_scores: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    for signal in SIGNALS:
        signal_id = str(signal["signal_id"])
        for scope, members in SCOPE_MEMBERS.items():
            for horizon in HORIZONS:
                period_records: list[dict[str, Any]] = []
                for period in DECISION_PERIODS:
                    member_passes = 0
                    actual_rates: list[float] = []
                    comparator_rates: list[float] = []
                    control_rates: list[float] = []
                    supports: list[int] = []
                    for pair in members:
                        actual = _score_lookup(
                            pair_scores,
                            signal_id=signal_id,
                            pair=pair,
                            sample_kind="actual_event",
                            period=period,
                            horizon=horizon,
                        )
                        control = _score_lookup(
                            pair_scores,
                            signal_id=signal_id,
                            pair=pair,
                            sample_kind="matched_control",
                            period=period,
                            horizon=horizon,
                        )
                        if actual is None or control is None:
                            continue
                        actual_rate = float(actual["volume_reaction_rate_calls"])
                        control_rate = float(control["volume_reaction_rate_calls"])
                        comparator = (
                            control_rate
                            if signal_id == "catalogued_event_clock"
                            else float(actual["volume_reaction_rate_noncalls"])
                        )
                        support = int(actual["call_units"])
                        passed = (
                            support >= MINIMUM_UNITS
                            and actual_rate >= LEAD_RATE
                            and pd.notna(comparator)
                            and actual_rate - comparator >= MINIMUM_UPLIFT
                        )
                        member_passes += int(passed)
                        actual_rates.append(actual_rate)
                        comparator_rates.append(comparator)
                        control_rates.append(control_rate)
                        supports.append(support)
                    member_fraction = member_passes / len(members)
                    period_records.append(
                        {
                            "period": period,
                            "minimum_support": min(supports) if supports else 0,
                            "mean_actual_rate": float(np.mean(actual_rates))
                            if actual_rates
                            else np.nan,
                            "mean_comparator_rate": float(np.mean(comparator_rates))
                            if comparator_rates
                            else np.nan,
                            "mean_control_rate": float(np.mean(control_rates))
                            if control_rates
                            else np.nan,
                            "member_pass_fraction": member_fraction,
                            "period_pass": member_fraction
                            >= SCOPE_REQUIRED_FRACTION[scope],
                        }
                    )
                complete = len(period_records) == len(DECISION_PERIODS)
                general_pass = bool(
                    complete and all(item["period_pass"] for item in period_records)
                )
                event_specific = bool(
                    general_pass
                    and all(
                        item["mean_actual_rate"] - item["mean_control_rate"]
                        >= MINIMUM_UPLIFT
                        for item in period_records
                    )
                )
                support_pass = bool(
                    complete
                    and all(item["minimum_support"] >= MINIMUM_UNITS for item in period_records)
                )
                if not support_pass:
                    verdict = "coverage_parked"
                elif not general_pass:
                    verdict = "not_retained_against_simple_comparator"
                elif event_specific:
                    verdict = "retained_event_specific_activity_prototype"
                else:
                    verdict = "retained_general_activity_prototype_not_event_specific"
                records.append(
                    {
                        "signal_id": signal_id,
                        "family_id": signal["family_id"],
                        "market_scope": scope,
                        "horizon_hours": horizon,
                        "complete_periods": complete,
                        "minimum_support": min(
                            (item["minimum_support"] for item in period_records), default=0
                        ),
                        "minimum_member_pass_fraction": min(
                            (item["member_pass_fraction"] for item in period_records),
                            default=0.0,
                        ),
                        "minimum_reaction_rate": min(
                            (item["mean_actual_rate"] for item in period_records),
                            default=np.nan,
                        ),
                        "minimum_uplift_over_simple_comparator": min(
                            (
                                item["mean_actual_rate"]
                                - item["mean_comparator_rate"]
                                for item in period_records
                            ),
                            default=np.nan,
                        ),
                        "minimum_event_minus_control": min(
                            (
                                item["mean_actual_rate"] - item["mean_control_rate"]
                                for item in period_records
                            ),
                            default=np.nan,
                        ),
                        "meets_55_floor": general_pass,
                        "meets_65_target": bool(
                            general_pass
                            and all(
                                item["mean_actual_rate"] >= STRONG_RATE
                                for item in period_records
                            )
                        ),
                        "event_specific": event_specific,
                        "verdict": verdict,
                    }
                )
    return DataFrame.from_records(records)


def run() -> dict[str, Any]:
    if RESULT_PATH.is_file():
        result = _load_json(RESULT_PATH)
        if result.get("status") != "completed_simple_signal_family_review":
            raise ValueError("Existing simple-family result is not terminal.")
        return result
    frozen = freeze()
    _validate_contracts(frozen)
    calls = pd.read_parquet(CALLS_PATH)
    calls["signal_issued"] = _coerce_bool(calls["signal_issued"])
    rows = _outcome_rows(calls, frozen)
    pair_scores = score_pairs(rows)
    scope_scores = add_scopes(pair_scores)
    decisions = decide(pair_scores, scope_scores)
    g0.atomic_write_csv(pair_scores, PAIR_SCORES_PATH)
    g0.atomic_write_csv(scope_scores, SCOPE_SCORES_PATH)
    g0.atomic_write_csv(decisions, DECISIONS_PATH)
    result = {
        "schema_version": 1,
        "status": "completed_simple_signal_family_review",
        "created_at_utc": g0.utc_now(),
        "run_id": RUN_ID,
        "family_count": len({item["family_id"] for item in SIGNALS}),
        "signal_count": len(SIGNALS),
        "decision_cells": len(decisions),
        "retained_cells": int(decisions["verdict"].str.startswith("retained").sum()),
        "strong_cells": int(decisions["meets_65_target"].sum()),
        "scope": frozen["scope"],
        "freeze_contract": artifact(FREEZE_PATH),
        "artifacts": {
            "calls_on_d": artifact(CALLS_PATH),
            "pair_scores": artifact(PAIR_SCORES_PATH),
            "scope_scores": artifact(SCOPE_SCORES_PATH),
            "decisions": artifact(DECISIONS_PATH),
        },
    }
    g0.atomic_write_json(result, RESULT_PATH)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run"))
    args = parser.parse_args(argv)
    result = freeze() if args.phase == "freeze" else run()
    print(json.dumps(result, indent=2, default=g0.json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

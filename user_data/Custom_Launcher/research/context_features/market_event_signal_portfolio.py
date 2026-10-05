"""Turn the frozen event breadth predictions into reason-labelled signal prototypes."""

from __future__ import annotations

# The repository root is inserted before local research imports.
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
    market_event_freqai_breadth as breadth,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_freqai_breadth_cache as breadth_cache,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_freqai_breadth_freeze as breadth_freeze,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
SOURCE_RUN_ID = "event_freqai_breadth_20260908a"
SOURCE_RUN_DIR = breadth.RECORD_ROOT / SOURCE_RUN_ID
SOURCE_MANIFEST_PATH = SOURCE_RUN_DIR / "event_freqai_run_manifest.json"
SOURCE_RESULT_PATH = SOURCE_RUN_DIR / "event_freqai_result.json"
RUN_ID = "event_signal_portfolio_20260908b"
SUPERSEDES_RUN_ID = "event_signal_portfolio_20260908a"
OUTPUT_ROOT = breadth_freeze.OUTPUT_ROOT.parent / RUN_ID
FREEZE_PATH = OUTPUT_ROOT / "event_signal_portfolio_freeze.json"
CALLS_PATH = OUTPUT_ROOT / "event_signal_calls.csv"
PAIR_SCORES_PATH = OUTPUT_ROOT / "event_signal_pair_period_scores.csv"
SCOPE_SCORES_PATH = OUTPUT_ROOT / "event_signal_scope_period_scores.csv"
DECISIONS_PATH = OUTPUT_ROOT / "event_signal_route_decisions.csv"
FAMILY_PATH = OUTPUT_ROOT / "event_signal_family_summary.csv"
RESULT_PATH = OUTPUT_ROOT / "event_signal_portfolio_result.json"

VALIDATION_PERIODS = (
    "walk_forward_validation_2024",
    "walk_forward_validation_2025",
)
HORIZONS = (1, 4, 8)
ACTIVITY_METRICS = ("log_range_atr", "log_volume_ratio")
DIRECTION_METRICS = ("close_return_atr", "excursion_balance_atr")
CALIBRATION_LOOKBACK_HOURS = 24 * 90
MIN_CALIBRATION_HOURS = 24 * 7
ACTIVITY_SCORE_QUANTILE = 0.60
DIRECTION_CONFIDENCE_QUANTILE = 0.50
MIN_EPISODES = 20
LEAD_RATE = 0.55
STRONG_RATE = 0.65
MIN_REACTION_ENRICHMENT = 0.02

FAMILY_IDS = (
    "major_event_information",
    "background_and_market_leadership",
    "calculated_reaction_areas",
    "local_participation_and_pressure",
    "cross_asset_transmission_and_amplification",
)

ROUTES: tuple[dict[str, Any], ...] = (
    {
        "route_id": "event_clock",
        "family_id": "major_event_information",
        "profile_id": "event_identity_only",
        "route_kind": "component",
        "activity_metrics": ACTIVITY_METRICS,
        "reason": "A known event type and clock suggest that the market may become busier.",
    },
    {
        "route_id": "event_with_recent_confirmation",
        "family_id": "major_event_information",
        "supporting_families": ["local_participation_and_pressure"],
        "profile_id": "event_plus_recent_market",
        "route_kind": "combination",
        "activity_metrics": ACTIVITY_METRICS,
        "reason": (
            "A known event and the immediately visible market behaviour agree that "
            "activity may rise."
        ),
    },
    {
        "route_id": "slow_background",
        "family_id": "background_and_market_leadership",
        "profile_id": "slow_background_only",
        "route_kind": "component",
        "activity_metrics": ACTIVITY_METRICS,
        "reason": "The preceding market background changes how plausible a larger reaction is.",
    },
    {
        "route_id": "event_with_background",
        "family_id": "background_and_market_leadership",
        "supporting_families": ["major_event_information"],
        "profile_id": "event_plus_background",
        "route_kind": "combination",
        "activity_metrics": ACTIVITY_METRICS,
        "reason": "A known event is interpreted in the context of the preceding market background.",
    },
    {
        "route_id": "single_calculated_level",
        "family_id": "calculated_reaction_areas",
        "profile_id": "single_level_only",
        "route_kind": "component",
        "activity_metrics": ACTIVITY_METRICS,
        "reason": "Price is near one independently calculated area where traffic may change.",
    },
    {
        "route_id": "calculated_level_cluster",
        "family_id": "calculated_reaction_areas",
        "profile_id": "cluster_only",
        "route_kind": "component",
        "activity_metrics": ACTIVITY_METRICS,
        "reason": "Price is near several independently calculated areas at once.",
    },
    {
        "route_id": "event_with_single_level",
        "family_id": "calculated_reaction_areas",
        "supporting_families": ["major_event_information"],
        "profile_id": "event_plus_single_level",
        "route_kind": "combination",
        "activity_metrics": ACTIVITY_METRICS,
        "reason": "A known event occurs while price is near one calculated area.",
    },
    {
        "route_id": "event_with_level_cluster",
        "family_id": "calculated_reaction_areas",
        "supporting_families": ["major_event_information"],
        "profile_id": "event_plus_cluster",
        "route_kind": "combination",
        "activity_metrics": ACTIVITY_METRICS,
        "reason": "A known event occurs while several calculated areas overlap nearby.",
    },
    {
        "route_id": "recent_local_participation",
        "family_id": "local_participation_and_pressure",
        "profile_id": "recent_market_only",
        "route_kind": "component",
        "activity_metrics": ACTIVITY_METRICS,
        "reason": (
            "Recent price, range, and volume behaviour suggest that local "
            "participation is changing."
        ),
    },
    {
        "route_id": "cross_market_participation",
        "family_id": "cross_asset_transmission_and_amplification",
        "profile_id": "cross_market_only",
        "route_kind": "component",
        "activity_metrics": ACTIVITY_METRICS,
        "reason": (
            "Several established crypto markets are already moving or becoming active "
            "together."
        ),
    },
    {
        "route_id": "event_with_cross_market_confirmation",
        "family_id": "cross_asset_transmission_and_amplification",
        "supporting_families": ["major_event_information"],
        "profile_id": "event_plus_cross_market",
        "route_kind": "combination",
        "activity_metrics": ACTIVITY_METRICS,
        "reason": (
            "A known event coincides with activity already visible across established "
            "crypto markets."
        ),
    },
)

PRESERVED_EXTERNAL_LEADS = (
    {
        "signal_id": "cpi_short_release_reaction",
        "family_id": "major_event_information",
        "status": "retained_specialist_lead",
        "availability": (
            "Activity clock is usable; true directional expectation surprise is "
            "unavailable without a causal forecast source."
        ),
    },
    {
        "signal_id": "sec_item_2_02_short_activity",
        "family_id": "major_event_information",
        "status": "retained_activity_lead",
        "availability": (
            "Usable as an activity warning; causal first-public story timing remains "
            "incomplete."
        ),
    },
    {
        "signal_id": "negative_background_positive_move_fade",
        "family_id": "background_and_market_leadership",
        "status": "retained_conditional_lead",
        "availability": (
            "Can issue only four hours after an observed positive event move; fresh-event "
            "confirmation is still queued."
        ),
    },
    {
        "signal_id": "btc_dominance_meme_relative_rotation",
        "family_id": "cross_asset_transmission_and_amplification",
        "status": "retained_relative_direction_lead",
        "availability": (
            "Predicts BTC relative to the meme group over eight hours, not absolute up "
            "or down direction."
        ),
    },
    {
        "signal_id": "calculated_area_traffic",
        "family_id": "calculated_reaction_areas",
        "status": "retained_reaction_lead",
        "availability": (
            "Useful for likely traffic or reaction, with no general bounce or breakout "
            "direction."
        ),
    },
    {
        "signal_id": "live_news_and_web_activity",
        "family_id": "major_event_information",
        "status": "retained_secondary_activity_lead",
        "availability": (
            "Recent history supports activity timing, but semantic direction is not "
            "ready."
        ),
    },
)


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _target(metric: str, horizon: int) -> str:
    return f"&-meb_{metric}_h{horizon}"


def build_freeze() -> dict[str, Any]:
    source_manifest = _load_json(SOURCE_MANIFEST_PATH)
    source_result = _load_json(SOURCE_RESULT_PATH)
    available_profiles = {str(item["profile_id"]) for item in source_manifest["commands"]}
    required_profiles = {str(route["profile_id"]) for route in ROUTES}
    missing = sorted(required_profiles.difference(available_profiles))
    if missing:
        raise ValueError(f"Source breadth run lacks portfolio profiles: {missing}")
    if source_result.get("status") != "completed_event_freqai_breadth":
        raise ValueError("Source breadth run is not complete.")
    represented = {str(route["family_id"]) for route in ROUTES}
    if represented != set(FAMILY_IDS):
        raise ValueError("The active routes do not cover exactly the five signal families.")
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "supersedes_run_id": SUPERSEDES_RUN_ID,
        "supersession_reason": (
            "The first prototype counted one activity result once per direction metric "
            "and one direction result once per activity metric in its headline totals. "
            "This revision reports unique logical activity and direction cells."
        ),
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_signal_outcome_rescoring",
        "purpose": (
            "Convert an already-completed breadth experiment into separate, reason-labelled "
            "offline signal prototypes without mapping them to trades."
        ),
        "families": list(FAMILY_IDS),
        "routes": [
            {
                **route,
                "activity_metrics": list(route["activity_metrics"]),
                "supporting_families": list(route.get("supporting_families", [])),
            }
            for route in ROUTES
        ],
        "preserved_external_leads": list(PRESERVED_EXTERNAL_LEADS),
        "validation_periods": list(VALIDATION_PERIODS),
        "horizons_hours": list(HORIZONS),
        "direction_metrics": list(DIRECTION_METRICS),
        "calibration": {
            "normalization": (
                "Normalize each prediction by the contemporaneous FreqAI training-label "
                "mean and standard deviation. Direction confidence uses absolute predicted "
                "distance from zero divided by that standard deviation."
            ),
            "lookback_hours": CALIBRATION_LOOKBACK_HOURS,
            "minimum_prior_hours": MIN_CALIBRATION_HOURS,
            "current_and_future_excluded": True,
            "activity_issue_quantile": ACTIVITY_SCORE_QUANTILE,
            "direction_confidence_issue_quantile": DIRECTION_CONFIDENCE_QUANTILE,
            "reason_for_thresholds": (
                "Moderate thresholds retain enough independent events for evaluation while "
                "still selecting stronger model scores; they were fixed without inspecting "
                "this gate's outcome results."
            ),
        },
        "decision_rule": {
            "minimum_independent_episodes_each_period": MIN_EPISODES,
            "lead_rate": LEAD_RATE,
            "strong_rate": STRONG_RATE,
            "minimum_reaction_enrichment_over_non_calls": MIN_REACTION_ENRICHMENT,
            "activity": (
                "At least 55% of issued calls react and exceed non-calls by at least two "
                "percentage points in both years."
            ),
            "direction": (
                "At least 55% correct and better than both training-majority and recent-trend "
                "controls in both years."
            ),
            "joint": (
                "At least 55% both react and have correct direction, while beating both joint "
                "controls, in both years."
            ),
            "strong": "The corresponding rate is at least 65% in both years.",
        },
        "interpretation_boundary": {
            "retrospective_prototype_not_fresh_confirmation": True,
            "profit_used": False,
            "entry_or_exit_rule_created": False,
            "live_or_dry_run_orders_allowed": False,
            "reason_label_and_abstention_required": True,
            "negative_and_parked_evidence_preserved": True,
        },
        "source_contracts": {
            "analysis_script": artifact(ANALYSIS_PATH),
            "source_run_manifest": artifact(SOURCE_MANIFEST_PATH),
            "source_result": artifact(SOURCE_RESULT_PATH),
            "breadth_freeze": artifact(breadth_freeze.FREEZE_PATH),
            "breadth_cache": artifact(breadth_cache.CACHE_MANIFEST_PATH),
            "profile_registry": artifact(breadth_freeze.REGISTRY_PATH),
        },
    }


def freeze(*, overwrite: bool = False) -> dict[str, Any]:
    if FREEZE_PATH.is_file() and not overwrite:
        return _load_json(FREEZE_PATH)
    output = build_freeze()
    g0.atomic_write_json(output, FREEZE_PATH)
    return output


def verify_freeze() -> dict[str, Any]:
    frozen = _load_json(FREEZE_PATH)
    if frozen.get("status") != "frozen_before_signal_outcome_rescoring":
        raise ValueError("Signal portfolio is not frozen.")
    for item in frozen["source_contracts"].values():
        path = Path(item["path"])
        if not path.is_file() or g0.sha256_file(path) != item["sha256"]:
            raise ValueError(f"Frozen signal source changed: {path}")
    return frozen


def causal_rolling_quantile(
    values: Series,
    *,
    quantile: float,
    lookback: int = CALIBRATION_LOOKBACK_HOURS,
    minimum: int = MIN_CALIBRATION_HOURS,
) -> Series:
    """Return a rolling threshold formed only from rows before the current row."""
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.shift(1).rolling(lookback, min_periods=minimum).quantile(quantile)


def calibrated_prediction(
    prediction: DataFrame,
    target: str,
    *,
    direction_confidence: bool,
) -> DataFrame:
    mean_column = f"{target}_mean"
    std_column = f"{target}_std"
    required = ["pair", "date", target, mean_column, std_column]
    missing = sorted(set(required).difference(prediction.columns))
    if missing:
        raise ValueError(f"Prediction data lacks calibration fields: {missing}")
    output = prediction[required].copy()
    output[target] = pd.to_numeric(output[target], errors="coerce")
    output[mean_column] = pd.to_numeric(output[mean_column], errors="coerce")
    deviation = pd.to_numeric(output[std_column], errors="coerce").where(lambda x: x > 0)
    if direction_confidence:
        output["calibrated_score"] = output[target].abs() / deviation
        quantile = DIRECTION_CONFIDENCE_QUANTILE
    else:
        output["calibrated_score"] = (output[target] - output[mean_column]) / deviation
        quantile = ACTIVITY_SCORE_QUANTILE
    output = output.sort_values(["pair", "date"], kind="stable")
    output["causal_threshold"] = output.groupby("pair", sort=False)[
        "calibrated_score"
    ].transform(lambda values: causal_rolling_quantile(values, quantile=quantile))
    output["signal_issued"] = output["calibrated_score"].ge(
        output["causal_threshold"]
    ) & output["causal_threshold"].notna()
    return output.rename(
        columns={
            target: "prediction",
            mean_column: "training_mean",
            std_column: "training_std",
        }
    )


def _lookup(frame: DataFrame, column: str) -> dict[tuple[str, str, str], Any]:
    return frame.set_index(["pair", "period", "target"])[column].to_dict()


def _episode_ids(values: Series) -> set[str]:
    identifiers: set[str] = set()
    for value in values.dropna():
        identifiers.update(str(item) for item in json.loads(str(value)))
    return identifiers


def _rate(values: Series) -> float:
    return float(values.mean()) if len(values) else np.nan


def load_profile_predictions(
    source_manifest: Mapping[str, Any],
) -> dict[str, DataFrame]:
    required = {str(route["profile_id"]) for route in ROUTES}
    command_lookup = {
        str(item["profile_id"]): item for item in source_manifest["commands"]
    }
    frames: dict[str, DataFrame] = {}
    for profile_id in sorted(required):
        command = command_lookup[profile_id]
        frame, audit = breadth.g0f.load_predictions(
            Path(command["model_dir"]), tuple(str(pair) for pair in command["pairs"])
        )
        if frame.empty or int(audit["duplicate_pair_date_rows"]) != 0:
            raise ValueError(f"Invalid source predictions for {profile_id}: {audit}")
        frames[profile_id] = frame
    return frames


def build_signal_rows(
    *,
    source_manifest: Mapping[str, Any],
    predictions: Mapping[str, DataFrame],
    actual: DataFrame,
) -> DataFrame:
    reference = pd.read_csv(source_manifest["storage"]["training_reference"])
    median_lookup = _lookup(reference, "training_median")
    upper_lookup = _lookup(reference, "training_upper_quartile")
    absolute_lookup = _lookup(reference, "training_median_absolute")
    majority_lookup = _lookup(reference, "training_majority_positive")
    event_rows = actual.loc[
        actual["sample_kind"].eq("actual_event")
        & actual["period"].isin(VALIDATION_PERIODS)
    ].copy()
    outputs: list[DataFrame] = []
    calibration_cache: dict[tuple[str, str, bool], DataFrame] = {}
    for route in ROUTES:
        profile_id = str(route["profile_id"])
        prediction = predictions[profile_id]
        for horizon in HORIZONS:
            for activity_metric in route["activity_metrics"]:
                activity_target = _target(str(activity_metric), horizon)
                activity_key = (profile_id, activity_target, False)
                if activity_key not in calibration_cache:
                    calibration_cache[activity_key] = calibrated_prediction(
                        prediction, activity_target, direction_confidence=False
                    )
                activity_frame = calibration_cache[activity_key].rename(
                    columns={
                        "prediction": "activity_prediction",
                        "training_mean": "activity_training_mean",
                        "training_std": "activity_training_std",
                        "calibrated_score": "activity_score",
                        "causal_threshold": "activity_threshold",
                        "signal_issued": "activity_signal",
                    }
                )
                for direction_metric in DIRECTION_METRICS:
                    direction_target = _target(direction_metric, horizon)
                    direction_key = (profile_id, direction_target, True)
                    if direction_key not in calibration_cache:
                        calibration_cache[direction_key] = calibrated_prediction(
                            prediction, direction_target, direction_confidence=True
                        )
                    direction_frame = calibration_cache[direction_key].rename(
                        columns={
                            "prediction": "direction_prediction",
                            "training_mean": "direction_training_mean",
                            "training_std": "direction_training_std",
                            "calibrated_score": "direction_confidence",
                            "causal_threshold": "direction_threshold",
                            "signal_issued": "direction_signal",
                        }
                    )
                    merged = event_rows.merge(
                        activity_frame,
                        on=["pair", "date"],
                        how="left",
                        validate="one_to_one",
                    ).merge(
                        direction_frame,
                        on=["pair", "date"],
                        how="left",
                        validate="one_to_one",
                    )
                    keys = list(zip(merged["pair"], merged["period"], strict=True))
                    merged["activity_reference"] = [
                        median_lookup.get((str(pair), str(period), activity_target), np.nan)
                        for pair, period in keys
                    ]
                    merged["activity_raw_upper_quartile"] = [
                        upper_lookup.get((str(pair), str(period), activity_target), np.nan)
                        for pair, period in keys
                    ]
                    merged["direction_raw_median_absolute"] = [
                        absolute_lookup.get((str(pair), str(period), direction_target), np.nan)
                        for pair, period in keys
                    ]
                    merged["training_majority_up"] = [
                        bool(majority_lookup.get((str(pair), str(period), direction_target), False))
                        for pair, period in keys
                    ]
                    merged["actual_reaction"] = pd.to_numeric(
                        merged[activity_target], errors="coerce"
                    ).gt(pd.to_numeric(merged["activity_reference"], errors="coerce"))
                    merged["actual_up"] = pd.to_numeric(
                        merged[direction_target], errors="coerce"
                    ).gt(0.0)
                    merged["predicted_up"] = pd.to_numeric(
                        merged["direction_prediction"], errors="coerce"
                    ).gt(0.0)
                    merged["direction_correct"] = merged["predicted_up"].eq(
                        merged["actual_up"]
                    )
                    merged["majority_correct"] = merged["training_majority_up"].eq(
                        merged["actual_up"]
                    )
                    merged["trend_correct"] = pd.to_numeric(
                        merged["baseline_pair_return_4h"], errors="coerce"
                    ).gt(0.0).eq(merged["actual_up"])
                    merged["joint_signal"] = merged["activity_signal"].fillna(False) & merged[
                        "direction_signal"
                    ].fillna(False)
                    merged["joint_success"] = merged["actual_reaction"] & merged[
                        "direction_correct"
                    ]
                    merged["majority_joint_success"] = merged["actual_reaction"] & merged[
                        "majority_correct"
                    ]
                    merged["trend_joint_success"] = merged["actual_reaction"] & merged[
                        "trend_correct"
                    ]
                    merged["original_raw_joint_signal"] = pd.to_numeric(
                        merged["activity_prediction"], errors="coerce"
                    ).gt(
                        pd.to_numeric(merged["activity_raw_upper_quartile"], errors="coerce")
                    ) & pd.to_numeric(merged["direction_prediction"], errors="coerce").abs().gt(
                        pd.to_numeric(merged["direction_raw_median_absolute"], errors="coerce")
                    )
                    merged["family_id"] = str(route["family_id"])
                    merged["supporting_families_json"] = json.dumps(
                        route.get("supporting_families", [])
                    )
                    merged["route_id"] = str(route["route_id"])
                    merged["route_kind"] = str(route["route_kind"])
                    merged["reason"] = str(route["reason"])
                    merged["horizon_hours"] = horizon
                    merged["activity_metric"] = str(activity_metric)
                    merged["activity_target"] = activity_target
                    merged["direction_metric"] = direction_metric
                    merged["direction_target"] = direction_target
                    outputs.append(merged)
    return pd.concat(outputs, ignore_index=True)


def score_pair_periods(rows: DataFrame) -> DataFrame:
    keys = [
        "family_id",
        "route_id",
        "route_kind",
        "period",
        "pair",
        "horizon_hours",
        "activity_metric",
        "activity_target",
        "direction_metric",
        "direction_target",
    ]
    output: list[dict[str, Any]] = []
    for key, cell in rows.groupby(keys, observed=True, sort=False):
        eligible = cell.dropna(
            subset=[
                "activity_prediction",
                "activity_threshold",
                "direction_prediction",
                "direction_threshold",
                "activity_reference",
            ]
        )
        activity_calls = eligible.loc[eligible["activity_signal"]]
        activity_non_calls = eligible.loc[~eligible["activity_signal"]]
        direction_calls = eligible.loc[eligible["direction_signal"]]
        joint_calls = eligible.loc[eligible["joint_signal"]]
        original_calls = eligible.loc[eligible["original_raw_joint_signal"]]
        output.append(
            {
                **dict(zip(keys, key, strict=True)),
                "market_scope": f"asset:{key[4]}",
                "eligible_coins": 1,
                "eligible_events": len(eligible),
                "eligible_unique_episodes": len(
                    _episode_ids(eligible["parent_episode_ids_json"])
                ),
                "activity_calls": len(activity_calls),
                "activity_unique_episodes": len(
                    _episode_ids(activity_calls["parent_episode_ids_json"])
                ),
                "activity_call_rate": len(activity_calls) / max(1, len(eligible)),
                "reaction_rate_all": _rate(eligible["actual_reaction"]),
                "reaction_rate_calls": _rate(activity_calls["actual_reaction"]),
                "reaction_rate_non_calls": _rate(activity_non_calls["actual_reaction"]),
                "reaction_enrichment_vs_non_calls": _rate(
                    activity_calls["actual_reaction"]
                )
                - _rate(activity_non_calls["actual_reaction"]),
                "direction_calls": len(direction_calls),
                "direction_unique_episodes": len(
                    _episode_ids(direction_calls["parent_episode_ids_json"])
                ),
                "direction_correct_rate": _rate(direction_calls["direction_correct"]),
                "direction_majority_control_rate": _rate(
                    direction_calls["majority_correct"]
                ),
                "direction_trend_control_rate": _rate(direction_calls["trend_correct"]),
                "joint_calls": len(joint_calls),
                "joint_unique_episodes": len(
                    _episode_ids(joint_calls["parent_episode_ids_json"])
                ),
                "joint_reaction_rate": _rate(joint_calls["actual_reaction"]),
                "joint_conditional_direction_rate": _rate(
                    joint_calls.loc[joint_calls["actual_reaction"], "direction_correct"]
                ),
                "joint_success_rate": _rate(joint_calls["joint_success"]),
                "joint_majority_control_rate": _rate(
                    joint_calls["majority_joint_success"]
                ),
                "joint_trend_control_rate": _rate(joint_calls["trend_joint_success"]),
                "original_raw_joint_calls": len(original_calls),
                "original_raw_joint_unique_episodes": len(
                    _episode_ids(original_calls["parent_episode_ids_json"])
                ),
                "original_raw_joint_success_rate": _rate(
                    original_calls["joint_success"]
                ),
            }
        )
    return DataFrame.from_records(output)


RATE_COLUMNS = (
    "activity_call_rate",
    "reaction_rate_all",
    "reaction_rate_calls",
    "reaction_rate_non_calls",
    "reaction_enrichment_vs_non_calls",
    "direction_correct_rate",
    "direction_majority_control_rate",
    "direction_trend_control_rate",
    "joint_reaction_rate",
    "joint_conditional_direction_rate",
    "joint_success_rate",
    "joint_majority_control_rate",
    "joint_trend_control_rate",
    "original_raw_joint_success_rate",
)


def add_market_scopes(pair_scores: DataFrame) -> DataFrame:
    groups = {
        "established_alts_equal_weight": {
            "ETH/USDT:USDT",
            "BNB/USDT:USDT",
            "ADA/USDT:USDT",
            "TRX/USDT:USDT",
        },
        "all_five_equal_weight": set(breadth_freeze.NORMAL_PAIRS),
    }
    key_columns = [
        "family_id",
        "route_id",
        "route_kind",
        "period",
        "horizon_hours",
        "activity_metric",
        "activity_target",
        "direction_metric",
        "direction_target",
    ]
    aggregate_rows: list[dict[str, Any]] = []
    for scope, members in groups.items():
        selected = pair_scores.loc[pair_scores["pair"].isin(members)]
        for key, cell in selected.groupby(key_columns, observed=True, sort=False):
            if set(cell["pair"]) != members:
                continue
            aggregate_rows.append(
                {
                    **dict(zip(key_columns, key, strict=True)),
                    "pair": scope.upper(),
                    "market_scope": scope,
                    "eligible_coins": len(members),
                    "eligible_events": int(cell["eligible_events"].sum()),
                    "eligible_unique_episodes": int(
                        cell["eligible_unique_episodes"].min()
                    ),
                    "activity_calls": int(cell["activity_calls"].sum()),
                    "activity_unique_episodes": int(
                        cell["activity_unique_episodes"].min()
                    ),
                    "direction_calls": int(cell["direction_calls"].sum()),
                    "direction_unique_episodes": int(
                        cell["direction_unique_episodes"].min()
                    ),
                    "joint_calls": int(cell["joint_calls"].sum()),
                    "joint_unique_episodes": int(cell["joint_unique_episodes"].min()),
                    "original_raw_joint_calls": int(
                        cell["original_raw_joint_calls"].sum()
                    ),
                    "original_raw_joint_unique_episodes": int(
                        cell["original_raw_joint_unique_episodes"].min()
                    ),
                    **{
                        column: float(pd.to_numeric(cell[column], errors="coerce").mean())
                        for column in RATE_COLUMNS
                    },
                }
            )
    return pd.concat(
        [pair_scores, DataFrame.from_records(aggregate_rows)], ignore_index=True
    )


def route_decisions(scope_scores: DataFrame) -> DataFrame:
    keys = [
        "family_id",
        "route_id",
        "route_kind",
        "market_scope",
        "horizon_hours",
        "activity_metric",
        "activity_target",
        "direction_metric",
        "direction_target",
    ]
    expected = set(VALIDATION_PERIODS)
    decisions: list[dict[str, Any]] = []
    for key, cell in scope_scores.groupby(keys, observed=True, sort=False):
        complete = set(cell["period"]) == expected
        activity_pass = bool(
            complete
            and cell["activity_unique_episodes"].ge(MIN_EPISODES).all()
            and cell["reaction_rate_calls"].ge(LEAD_RATE).all()
            and cell["reaction_enrichment_vs_non_calls"]
            .ge(MIN_REACTION_ENRICHMENT)
            .all()
        )
        activity_strong = bool(
            activity_pass and cell["reaction_rate_calls"].ge(STRONG_RATE).all()
        )
        direction_pass = bool(
            complete
            and cell["direction_unique_episodes"].ge(MIN_EPISODES).all()
            and cell["direction_correct_rate"].ge(LEAD_RATE).all()
            and cell["direction_correct_rate"]
            .gt(cell["direction_majority_control_rate"])
            .all()
            and cell["direction_correct_rate"]
            .gt(cell["direction_trend_control_rate"])
            .all()
        )
        direction_strong = bool(
            direction_pass and cell["direction_correct_rate"].ge(STRONG_RATE).all()
        )
        joint_pass = bool(
            complete
            and cell["joint_unique_episodes"].ge(MIN_EPISODES).all()
            and cell["joint_success_rate"].ge(LEAD_RATE).all()
            and cell["joint_success_rate"]
            .gt(cell["joint_majority_control_rate"])
            .all()
            and cell["joint_success_rate"]
            .gt(cell["joint_trend_control_rate"])
            .all()
        )
        joint_strong = bool(
            joint_pass and cell["joint_success_rate"].ge(STRONG_RATE).all()
        )
        decisions.append(
            {
                **dict(zip(keys, key, strict=True)),
                "complete_periods": complete,
                "activity_status": (
                    "strong_retrospective_activity_prototype"
                    if activity_strong
                    else "retrospective_activity_prototype_lead"
                    if activity_pass
                    else "activity_prototype_not_retained"
                ),
                "direction_status": (
                    "strong_retrospective_direction_prototype"
                    if direction_strong
                    else "retrospective_direction_prototype_lead"
                    if direction_pass
                    else "direction_prototype_not_retained"
                ),
                "joint_status": (
                    "strong_retrospective_joint_prototype"
                    if joint_strong
                    else "retrospective_joint_prototype_lead"
                    if joint_pass
                    else "joint_prototype_not_retained"
                ),
                "activity_pass": activity_pass,
                "direction_pass": direction_pass,
                "joint_pass": joint_pass,
                "minimum_activity_unique_episodes": int(
                    cell["activity_unique_episodes"].min()
                ),
                "minimum_reaction_rate_calls": float(
                    cell["reaction_rate_calls"].min()
                ),
                "minimum_reaction_enrichment_vs_non_calls": float(
                    cell["reaction_enrichment_vs_non_calls"].min()
                ),
                "minimum_direction_unique_episodes": int(
                    cell["direction_unique_episodes"].min()
                ),
                "minimum_direction_correct_rate": float(
                    cell["direction_correct_rate"].min()
                ),
                "minimum_direction_margin_over_majority": float(
                    (
                        cell["direction_correct_rate"]
                        - cell["direction_majority_control_rate"]
                    ).min()
                ),
                "minimum_direction_margin_over_trend": float(
                    (
                        cell["direction_correct_rate"]
                        - cell["direction_trend_control_rate"]
                    ).min()
                ),
                "minimum_joint_unique_episodes": int(
                    cell["joint_unique_episodes"].min()
                ),
                "minimum_joint_success_rate": float(
                    cell["joint_success_rate"].min()
                ),
                "minimum_joint_margin_over_majority": float(
                    (
                        cell["joint_success_rate"]
                        - cell["joint_majority_control_rate"]
                    ).min()
                ),
                "minimum_joint_margin_over_trend": float(
                    (
                        cell["joint_success_rate"]
                        - cell["joint_trend_control_rate"]
                    ).min()
                ),
                "minimum_original_raw_joint_unique_episodes": int(
                    cell["original_raw_joint_unique_episodes"].min()
                ),
            }
        )
    return DataFrame.from_records(decisions)


def family_summary(decisions: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for family_id in FAMILY_IDS:
        cell = decisions.loc[decisions["family_id"].eq(family_id)]
        activity_leads = cell.loc[cell["activity_pass"]].drop_duplicates(
            ["route_id", "market_scope", "horizon_hours", "activity_metric"]
        )
        direction_leads = cell.loc[cell["direction_pass"]].drop_duplicates(
            ["route_id", "market_scope", "horizon_hours", "direction_metric"]
        )
        joint_leads = cell.loc[cell["joint_pass"]]
        rows.append(
            {
                "family_id": family_id,
                "routes_tested": int(cell["route_id"].nunique()),
                "decision_cells": len(cell),
                "unique_activity_lead_cells": len(activity_leads),
                "unique_direction_lead_cells": len(direction_leads),
                "joint_lead_cells": len(joint_leads),
                "status": (
                    "retrospective_joint_candidate_present"
                    if len(joint_leads)
                    else "retrospective_direction_candidate_present"
                    if len(direction_leads)
                    else "retrospective_activity_candidate_present"
                    if len(activity_leads)
                    else "no_calibrated_prototype_lead"
                ),
            }
        )
    return DataFrame.from_records(rows)


def score() -> dict[str, Any]:
    frozen = verify_freeze()
    source_manifest = _load_json(SOURCE_MANIFEST_PATH)
    predictions = load_profile_predictions(source_manifest)
    actual = breadth.load_actual(source_manifest)
    signal_rows = build_signal_rows(
        source_manifest=source_manifest,
        predictions=predictions,
        actual=actual,
    )
    pair_scores = score_pair_periods(signal_rows)
    scope_scores = add_market_scopes(pair_scores)
    decisions = route_decisions(scope_scores)
    families = family_summary(decisions)
    activity_leads = decisions.loc[decisions["activity_pass"]].drop_duplicates(
        [
            "family_id",
            "route_id",
            "market_scope",
            "horizon_hours",
            "activity_metric",
        ]
    )
    direction_leads = decisions.loc[decisions["direction_pass"]].drop_duplicates(
        [
            "family_id",
            "route_id",
            "market_scope",
            "horizon_hours",
            "direction_metric",
        ]
    )
    joint_leads = decisions.loc[decisions["joint_pass"]]
    calls = signal_rows.loc[
        signal_rows["activity_signal"]
        | signal_rows["direction_signal"]
        | signal_rows["joint_signal"]
    ].copy()
    output_columns = [
        "date",
        "pair",
        "period",
        "sample_id",
        "parent_episode_ids_json",
        "family_id",
        "supporting_families_json",
        "route_id",
        "route_kind",
        "reason",
        "horizon_hours",
        "activity_metric",
        "direction_metric",
        "activity_score",
        "activity_threshold",
        "activity_signal",
        "direction_prediction",
        "direction_confidence",
        "direction_threshold",
        "direction_signal",
        "joint_signal",
        "actual_reaction",
        "direction_correct",
        "joint_success",
    ]
    g0.atomic_write_csv(calls[output_columns], CALLS_PATH)
    g0.atomic_write_csv(pair_scores, PAIR_SCORES_PATH)
    g0.atomic_write_csv(scope_scores, SCOPE_SCORES_PATH)
    g0.atomic_write_csv(decisions, DECISIONS_PATH)
    g0.atomic_write_csv(families, FAMILY_PATH)
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "supersedes_run_id": SUPERSEDES_RUN_ID,
        "created_at_utc": g0.utc_now(),
        "status": "completed_retrospective_signal_portfolio_prototype",
        "prototype_only_not_fresh_confirmation": True,
        "profit_used": False,
        "families_represented": len(families),
        "routes_tested": int(decisions["route_id"].nunique()),
        "route_decision_cells": len(decisions),
        "unique_activity_lead_cells": len(activity_leads),
        "unique_direction_lead_cells": len(direction_leads),
        "joint_lead_cells": len(joint_leads),
        "original_gate_maximum_minimum_episodes": int(
            decisions["minimum_original_raw_joint_unique_episodes"].max()
        ),
        "family_status": families.to_dict(orient="records"),
        "source_freeze": artifact(FREEZE_PATH),
        "artifacts": {
            "signal_calls": artifact(CALLS_PATH),
            "pair_period_scores": artifact(PAIR_SCORES_PATH),
            "scope_period_scores": artifact(SCOPE_SCORES_PATH),
            "route_decisions": artifact(DECISIONS_PATH),
            "family_summary": artifact(FAMILY_PATH),
        },
        "interpretation": (
            "These results test whether completed FreqAI outputs can be converted into "
            "causally calibrated, reason-labelled calls. They reuse previously inspected "
            "years and therefore cannot provide fresh confirmation or authorize trading."
        ),
        "freeze_status": frozen["status"],
    }
    g0.atomic_write_json(result, RESULT_PATH)
    return {**result, "result_path": str(RESULT_PATH.resolve())}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--overwrite-freeze", action="store_true")
    args = parser.parse_args(argv)
    if args.freeze_only:
        result = freeze(overwrite=args.overwrite_freeze)
    else:
        if not FREEZE_PATH.is_file():
            raise FileNotFoundError(
                f"Freeze the signal portfolio before scoring outcomes: {FREEZE_PATH}"
            )
        result = score()
    print(json.dumps(result, indent=2, default=g0.json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

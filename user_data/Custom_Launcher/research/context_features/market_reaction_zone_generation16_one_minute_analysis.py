"""Run the frozen Generation 16 one-minute reaction-and-direction diagnostic."""

from __future__ import annotations

# The frozen twelve-episode replay is intentionally single-worker.
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
import hashlib
import json
import math
import sys
from collections.abc import Iterable
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
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_one_minute_replay as g3m,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_one_minute_replay_analysis as g3a,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation9_one_minute_analysis as g9a,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation16_freeze as g16z,
)


DEFAULT_RUN_ID = "g16_one_minute_direct_20260822a"
REPORT_ROOT = g16z.FREEZE_PATH.parent / "g16_broad_attribution" / "one_minute_direct"
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData")
    / "research_outputs"
    / "market_reaction_zones"
    / "generation16"
    / "one_minute_direct"
)
COVERAGE_RECORD_NAME = "g16_one_minute_coverage_record.json"
DIRECTION_HORIZONS = (15, 60, 240, 720)
REACTION_EXCURSION_HALF_WIDTHS = 1.0
REACTION_VOLUME_RATIO_MINIMUM = 1.5
VOLUME_CONFIRMATION_RATIO = 1.25
JOINT_FLOOR = 0.55
JOINT_MAIN_TARGET = 0.65
MINIMUM_EPISODES_FOR_REPEATABLE_LEAD = 20
CALL_METHODS = (
    "pre_15m_trend",
    "pre_60m_trend",
    "pre_240m_trend",
    "pre_20m_pressure",
    "contact_candle_pressure",
    "completed_5m_pressure",
    "rolling_5m_pressure_acceleration",
    "volume_confirmed_completed_5m_pressure",
    "pre60_contact_pressure_agreement",
    "pre60_rolling_pressure_agreement",
    "multi_timeframe_ema_vote",
    "multi_timeframe_pressure_vote",
    "btc_orderbook_pressure_if_usable",
    "leave_one_out_cohort_majority_comparator",
)


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def validate_inputs() -> tuple[dict[str, Any], Path, Path]:
    frozen = json.loads(g16z.FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation16_outcomes":
        raise ValueError("Generation 16 is not frozen before outcomes.")
    sample_record = frozen["artifacts"]["one_minute_frozen_sample"]
    sample_path = Path(sample_record["path"])
    if (
        not sample_path.is_file()
        or g0.sha256_file(sample_path) != sample_record["sha256"]
    ):
        raise ValueError("The frozen Generation 16 one-minute sample changed.")
    coverage_path = (
        g16z.FREEZE_ROOT
        / g16z.DEFAULT_RUN_ID
        / COVERAGE_RECORD_NAME
    )
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if coverage.get("status") != "passed":
        raise ValueError("Generation 16 one-minute coverage has not passed.")
    return frozen, sample_path, coverage_path


def request_contract(coverage_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "objective": (
            "Measure reaction and the first up/down move after a completed contact minute "
            "without using future price direction to select episodes or make calls."
        ),
        "freeze": artifact(g16z.FREEZE_PATH),
        "coverage": artifact(coverage_path),
        "movement_origin": (
            "The completed contact-minute close; direction begins on the next minute."
        ),
        "horizons_minutes": list(DIRECTION_HORIZONS),
        "reaction_definition": {
            "minimum_absolute_excursion_zone_half_widths": (
                REACTION_EXCURSION_HALF_WIDTHS
            ),
            "minimum_post_to_pre_mean_volume_ratio": REACTION_VOLUME_RATIO_MINIMUM,
        },
        "call_methods": list(CALL_METHODS),
        "call_availability": "A method abstains when its causal input is absent or tied.",
        "volume_confirmation_ratio": VOLUME_CONFIRMATION_RATIO,
        "acceptable_joint_floor": JOINT_FLOOR,
        "main_joint_target": JOINT_MAIN_TARGET,
        "minimum_episodes_for_repeatable_lead": MINIMUM_EPISODES_FOR_REPEATABLE_LEAD,
        "selection_used_future_reaction": False,
        "selection_used_future_direction": False,
        "freqai_training": False,
        "profit_used": False,
        "promotion_allowed": False,
        "worker_count": 1,
    }


def episode_identity(episode: Series) -> dict[str, Any]:
    return {
        "episode_id": str(episode["episode_id"]),
        "sample_selection_order": int(episode["sample_selection_order"]),
        "sample_stratum": str(episode["sample_stratum"]),
        "cohort": str(episode["cohort"]),
        "period": str(episode["analysis_period"]),
        "pair": str(episode["pair"]),
        "parent_contact_hour": pd.Timestamp(episode["event_time"]),
        "source_timeframe": str(episode["source_timeframe"]),
        "level_family": str(episode["level_family"]),
        "level_name": str(episode["level_name"]),
        "level_column": str(episode["level_column"]),
        "representation": str(episode["representation"]),
        "approach_state": str(episode["approach_state"]),
        "approach_relative_direction_callable": str(episode["approach_state"])
        in {"from_below", "from_above"},
        "source_open": pd.Timestamp(episode["source_open"]),
        "source_available_at": pd.Timestamp(episode["source_available_at"]),
        "level_price": float(episode["level_price"]),
        "zone_half_width": float(episode["zone_half_width"]),
        "zone_half_width_atr": float(episode["zone_half_width_atr"]),
        "base_atr": float(episode["base_atr"]),
    }


def path_reference_name(approach_state: str) -> str:
    if approach_state == "from_above":
        return "lvn_below"
    return "lvn_above"


def path_rows(
    *, episode: Series, minute: DataFrame, contact_time: pd.Timestamp
) -> list[dict[str, Any]]:
    return g3a.checkpoint_paths(
        minute=minute,
        timestamp=contact_time,
        reference_price=float(episode["level_price"]),
        zone_half_width=float(episode["zone_half_width"]),
        level_name=path_reference_name(str(episode["approach_state"])),
        checkpoints=g3a.CHECKPOINT_MINUTES[
            str(episode["cluster_causal_anchor_timeframe"])
        ],
        episode_id=str(episode["episode_id"]),
        cohort=str(episode["cohort"]),
        event_kind="actual_calculated_area_contact",
    )


def contact_state(minute: DataFrame, contact_time: pd.Timestamp) -> dict[str, float]:
    contact = g3a.minute_row_at(minute, contact_time)
    recent_start = contact_time - pd.Timedelta(minutes=4)
    previous_start = recent_start - pd.Timedelta(minutes=20)
    recent = minute.loc[
        minute["date"].between(recent_start, contact_time, inclusive="both")
    ].copy()
    previous = minute.loc[
        (minute["date"] >= previous_start) & (minute["date"] < recent_start)
    ].copy()
    pre60 = minute.loc[
        (minute["date"] >= contact_time - pd.Timedelta(minutes=60))
        & (minute["date"] < contact_time)
    ].copy()
    if len(recent) != 5 or len(previous) != 20 or len(pre60) != 60:
        raise ValueError("Incomplete causal contact-state window.")
    contact_range = float(contact["high"] - contact["low"])
    contact_pressure = (
        float((contact["close"] - contact["open"]) / contact_range)
        if contact_range > 0.0
        else 0.0
    )
    recent_pressure = g3a.volume_weighted_pressure(recent)
    previous_pressure = g3a.volume_weighted_pressure(previous)
    previous_volume = float(previous["volume"].mean())
    pre60_range = (pre60["high"] - pre60["low"]).replace(0.0, np.nan)
    return {
        "contact_candle_pressure": contact_pressure,
        "contact_volume_ratio_pre60": (
            float(contact["volume"] / pre60["volume"].mean())
            if pre60["volume"].mean() > 0.0
            else math.nan
        ),
        "contact_range_ratio_pre60": (
            contact_range / float(pre60_range.median())
            if np.isfinite(pre60_range.median()) and pre60_range.median() > 0.0
            else math.nan
        ),
        "completed_5m_pressure": recent_pressure,
        "prior_20m_pressure": previous_pressure,
        "rolling_5m_pressure_acceleration": recent_pressure - previous_pressure,
        "completed_5m_volume_ratio_prior20": (
            float(recent["volume"].mean() / previous_volume)
            if previous_volume > 0.0
            else math.nan
        ),
    }


def horizon_outcomes(paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in DIRECTION_HORIZONS:
        path = next(item for item in paths if item["checkpoint_minutes"] == horizon)
        first_up = path["first_up_threshold_minute"]
        first_down = path["first_down_threshold_minute"]
        if first_up is None and first_down is None:
            direction = "unreached"
        elif first_up is None:
            direction = "down"
        elif first_down is None:
            direction = "up"
        elif first_up == first_down:
            direction = "tie"
        else:
            direction = "up" if first_up < first_down else "down"
        direction_numeric = {"up": 1, "down": -1, "tie": 0, "unreached": 0}[
            direction
        ]
        excursion = float(path["maximum_absolute_excursion_half_widths"])
        volume_ratio = float(path["volume_ratio_post_pre"])
        rows.append(
            {
                "horizon_minutes": horizon,
                "reaction": bool(
                    excursion >= REACTION_EXCURSION_HALF_WIDTHS
                    and volume_ratio >= REACTION_VOLUME_RATIO_MINIMUM
                ),
                "absolute_excursion_half_widths": excursion,
                "volume_ratio_post_pre": volume_ratio,
                "first_direction": direction,
                "first_direction_numeric": direction_numeric,
                "direction_callable": direction_numeric != 0,
                "first_up_threshold_minute": first_up,
                "first_down_threshold_minute": first_down,
                "close_displacement_half_widths": float(
                    path["close_displacement_half_widths"]
                ),
                "zone_overlap_fraction": float(path["zone_overlap_fraction"]),
                "reference_crossings": int(path["close_crossings_of_reference"]),
            }
        )
    return rows


def analyze_episode(
    episode: Series, *, surfaces: g3a.PairSurfaces, orderbook: DataFrame
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    contact_time = g3a.first_parent_zone_contact(episode, surfaces.minute)
    paths = path_rows(episode=episode, minute=surfaces.minute, contact_time=contact_time)
    state = {
        **g3a.match_snapshot(surfaces.match, contact_time),
        **g3a.technical_snapshot(surfaces.technical, contact_time),
        **contact_state(surfaces.minute, contact_time),
        **g3a.orderbook_snapshot(orderbook, contact_time),
    }
    identity = episode_identity(episode)
    actual = {
        **identity,
        "contact_time": contact_time,
        "decision_time": contact_time + pd.Timedelta(minutes=1),
        "reference_price": float(episode["level_price"]),
        **state,
    }
    outcomes = [
        {**identity, "contact_time": contact_time, **outcome}
        for outcome in horizon_outcomes(paths)
    ]
    boundary = g3a.boundary_audit(
        episode=episode,
        contact_time=contact_time,
        minute=surfaces.minute,
        window_hours=g3m.WINDOW_HOURS,
    )
    return actual, outcomes, paths, boundary


def agreement_call(left: int, right: int) -> int:
    return left if left != 0 and left == right else 0


def causal_calls(actual: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for _, event in actual.iterrows():
        trend15 = g9a.finite_sign(event["match_return_15m"])
        trend60 = g9a.finite_sign(event["match_return_60m"])
        trend240 = g9a.finite_sign(event["match_return_240m"])
        pressure20 = g9a.finite_sign(event["match_pressure_20m"])
        contact_pressure = g9a.finite_sign(event["contact_candle_pressure"])
        rolling_pressure = g9a.finite_sign(event["completed_5m_pressure"])
        acceleration = g9a.finite_sign(event["rolling_5m_pressure_acceleration"])
        volume_confirmed = (
            rolling_pressure
            if float(event["completed_5m_volume_ratio_prior20"])
            >= VOLUME_CONFIRMATION_RATIO
            else 0
        )
        orderbook_call = 0
        if bool(event.get("btc_orderbook_usable_coverage", False)):
            orderbook_call = g9a.finite_sign(
                event.get("btc_orderbook_pressure_25bps_last")
            )
        methods = {
            "pre_15m_trend": trend15,
            "pre_60m_trend": trend60,
            "pre_240m_trend": trend240,
            "pre_20m_pressure": pressure20,
            "contact_candle_pressure": contact_pressure,
            "completed_5m_pressure": rolling_pressure,
            "rolling_5m_pressure_acceleration": acceleration,
            "volume_confirmed_completed_5m_pressure": volume_confirmed,
            "pre60_contact_pressure_agreement": agreement_call(
                trend60, contact_pressure
            ),
            "pre60_rolling_pressure_agreement": agreement_call(
                trend60, rolling_pressure
            ),
            "multi_timeframe_ema_vote": g9a.multi_timeframe_vote(
                event, "ema12_minus_ema26_pct"
            ),
            "multi_timeframe_pressure_vote": g9a.multi_timeframe_vote(
                event, "pressure_20bar"
            ),
            "btc_orderbook_pressure_if_usable": orderbook_call,
        }
        for method, call in methods.items():
            rows.append(
                {
                    "episode_id": event["episode_id"],
                    "cohort": event["cohort"],
                    "period": event["period"],
                    "pair": event["pair"],
                    "method": method,
                    "method_kind": "causal_candidate",
                    "call_numeric": call,
                    "call_direction": (
                        "up" if call > 0 else "down" if call < 0 else "abstain"
                    ),
                    "issued": call != 0,
                }
            )
    return DataFrame.from_records(rows)


def evaluated_calls(calls: DataFrame, outcomes: DataFrame) -> DataFrame:
    repeated = calls.merge(
        outcomes[
            [
                "episode_id",
                "horizon_minutes",
                "reaction",
                "first_direction",
                "first_direction_numeric",
                "direction_callable",
            ]
        ],
        on="episode_id",
        how="inner",
        validate="many_to_many",
    )
    repeated["direction_correct"] = (
        repeated["issued"]
        & repeated["direction_callable"]
        & repeated["call_numeric"].eq(repeated["first_direction_numeric"])
    )
    repeated["joint_reaction_and_direction"] = (
        repeated["reaction"] & repeated["direction_correct"]
    )
    majority_rows: list[dict[str, Any]] = []
    for horizon, horizon_outcomes_frame in outcomes.groupby(
        "horizon_minutes", observed=True
    ):
        for _, outcome in horizon_outcomes_frame.iterrows():
            peers = horizon_outcomes_frame.loc[
                horizon_outcomes_frame["cohort"].eq(outcome["cohort"])
                & ~horizon_outcomes_frame["episode_id"].eq(outcome["episode_id"])
                & horizon_outcomes_frame["direction_callable"]
            ]
            call = (
                int(np.sign(peers["first_direction_numeric"].sum()))
                if len(peers)
                else 0
            )
            issued = call != 0
            correct = bool(
                issued
                and bool(outcome["direction_callable"])
                and call == int(outcome["first_direction_numeric"])
            )
            majority_rows.append(
                {
                    "episode_id": outcome["episode_id"],
                    "cohort": outcome["cohort"],
                    "period": outcome["period"],
                    "pair": outcome["pair"],
                    "method": "leave_one_out_cohort_majority_comparator",
                    "method_kind": "outcome_only_non_deployable_comparator",
                    "call_numeric": call,
                    "call_direction": (
                        "up" if call > 0 else "down" if call < 0 else "abstain"
                    ),
                    "issued": issued,
                    "horizon_minutes": horizon,
                    "reaction": bool(outcome["reaction"]),
                    "first_direction": outcome["first_direction"],
                    "first_direction_numeric": int(
                        outcome["first_direction_numeric"]
                    ),
                    "direction_callable": bool(outcome["direction_callable"]),
                    "direction_correct": correct,
                    "joint_reaction_and_direction": bool(
                        outcome["reaction"] and correct
                    ),
                }
            )
    return pd.concat(
        [repeated, DataFrame.from_records(majority_rows)], ignore_index=True
    )


def call_summary(calls: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = [
        ("all", calls),
        *[(str(key), frame) for key, frame in calls.groupby("cohort", observed=True)],
    ]
    for scope, scoped in scopes:
        for (horizon, method), frame in scoped.groupby(
            ["horizon_minutes", "method"], observed=True
        ):
            issued = frame.loc[frame["issued"]]
            callable_issued = issued.loc[issued["direction_callable"]]
            reacted_callable_issued = callable_issued.loc[
                callable_issued["reaction"]
            ]
            direction_correct = int(callable_issued["direction_correct"].sum())
            conditional_correct = int(
                reacted_callable_issued["direction_correct"].sum()
            )
            joint = int(frame["joint_reaction_and_direction"].sum())
            joint_rate = joint / len(frame) if len(frame) else math.nan
            lower, upper = g9a.wilson_interval(joint, len(frame))
            rows.append(
                {
                    "scope": scope,
                    "horizon_minutes": int(horizon),
                    "method": method,
                    "method_kind": str(frame.iloc[0]["method_kind"]),
                    "independent_episodes": int(frame["episode_id"].nunique()),
                    "issued_calls": len(issued),
                    "abstentions": len(frame) - len(issued),
                    "issued_call_coverage": len(issued) / len(frame),
                    "reaction_successes": int(frame["reaction"].sum()),
                    "reaction_rate": float(frame["reaction"].mean()),
                    "callable_issued": len(callable_issued),
                    "direction_correct_all_callable": direction_correct,
                    "direction_accuracy_all_callable": (
                        direction_correct / len(callable_issued)
                        if len(callable_issued)
                        else math.nan
                    ),
                    "reacted_callable_issued": len(reacted_callable_issued),
                    "conditional_direction_correct": conditional_correct,
                    "conditional_direction_accuracy_given_reaction": (
                        conditional_correct / len(reacted_callable_issued)
                        if len(reacted_callable_issued)
                        else math.nan
                    ),
                    "joint_successes": joint,
                    "joint_success_rate_all_episodes": joint_rate,
                    "joint_success_rate_issued_calls": (
                        joint / len(issued) if len(issued) else math.nan
                    ),
                    "joint_wilson_lower": lower,
                    "joint_wilson_upper": upper,
                    "observed_at_or_above_55pct": bool(joint_rate >= JOINT_FLOOR),
                    "observed_at_or_above_65pct": bool(
                        joint_rate >= JOINT_MAIN_TARGET
                    ),
                    "repeatable_lead_sample_size": bool(
                        frame["episode_id"].nunique()
                        >= MINIMUM_EPISODES_FOR_REPEATABLE_LEAD
                    ),
                    "status": "diagnostic_only_insufficient_independent_episodes",
                }
            )
    return DataFrame.from_records(rows)


def period_summary(outcomes: DataFrame) -> DataFrame:
    return (
        outcomes.groupby(["cohort", "period", "horizon_minutes"], observed=True)
        .agg(
            episodes=("episode_id", "nunique"),
            reaction_rate=("reaction", "mean"),
            direction_callable_rate=("direction_callable", "mean"),
            median_excursion_half_widths=(
                "absolute_excursion_half_widths",
                "median",
            ),
            median_volume_ratio=("volume_ratio_post_pre", "median"),
        )
        .reset_index()
    )


def compact_summary(
    actual: DataFrame,
    outcomes: DataFrame,
    scores: DataFrame,
    boundaries: DataFrame,
) -> dict[str, Any]:
    causal = scores.loc[
        scores["scope"].eq("all")
        & scores["method_kind"].eq("causal_candidate")
    ].sort_values(
        ["joint_success_rate_all_episodes", "issued_call_coverage"],
        ascending=False,
    )
    best = causal.iloc[0].to_dict() if len(causal) else {}
    reaction_rates = {
        str(horizon): float(frame["reaction"].mean())
        for horizon, frame in outcomes.groupby("horizon_minutes", observed=True)
    }
    return {
        "status": "completed_generation16_one_minute_diagnostic",
        "actual_episodes": len(actual),
        "independent_pairs": int(actual["pair"].nunique()),
        "normal_episodes": int(actual["cohort"].eq("normal").sum()),
        "meme_episodes": int(actual["cohort"].eq("meme").sum()),
        "horizons_minutes": list(DIRECTION_HORIZONS),
        "reaction_rates": reaction_rates,
        "direction_methods": len(CALL_METHODS),
        "best_observed_causal_method_not_a_lead": best,
        "methods_at_or_above_55pct": int(
            causal["observed_at_or_above_55pct"].sum()
        ),
        "methods_at_or_above_65pct": int(
            causal["observed_at_or_above_65pct"].sum()
        ),
        "boundary_extension_indicated": int(
            boundaries["boundary_extension_indicated"].sum()
        ),
        "interpretation": (
            "Every percentage is diagnostic. Twelve distinct-pair episodes are below "
            "the frozen repeatability requirement, so a point estimate above 55% or 65% "
            "queues a fresh confirmation batch and is not a trading claim."
        ),
        "selection_conditioning_warning": (
            "Episodes were selected only from causal calculated-area contacts, but from "
            "families and timeframes retained after Generation 15 activity evidence."
        ),
        "freqai_trained": False,
        "profit_used": False,
        "promotion_allowed": False,
    }


def run_analysis(run_id: str) -> int:
    _, sample_path, coverage_path = validate_inputs()
    run_dir = REPORT_ROOT / run_id
    artifact_dir = ARTIFACT_ROOT / run_id
    record_path = run_dir / "g16_one_minute_analysis_record.json"
    if record_path.is_file():
        raise FileExistsError(record_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    request = request_contract(coverage_path)
    record: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "started_at_utc": g0.utc_now(),
        "request_contract": request,
        "request_sha256": hashlib.sha256(
            json.dumps(request, sort_keys=True).encode()
        ).hexdigest(),
    }
    g0.atomic_write_json(record, record_path)
    try:
        sample = pd.read_csv(sample_path)
        for column in ("event_time", "source_open", "source_available_at"):
            sample[column] = pd.to_datetime(sample[column], utc=True, errors="raise")
        coverage = g3a.audit_analysis_coverage(
            sample=sample, window_hours=g3m.WINDOW_HOURS
        )
        if not bool(coverage["passed"].all()):
            raise ValueError("The effective Generation 16 analysis windows are incomplete.")
        orderbook = g3a.load_orderbook_surface()
        actual_rows: list[dict[str, Any]] = []
        outcome_rows: list[dict[str, Any]] = []
        all_paths: list[dict[str, Any]] = []
        boundary_rows: list[dict[str, Any]] = []
        for _, episode in sample.sort_values("sample_selection_order").iterrows():
            surfaces = g3a.load_pair_surfaces(
                str(episode["pair"]), cohort=str(episode["cohort"])
            )
            actual, outcomes, paths, boundary = analyze_episode(
                episode, surfaces=surfaces, orderbook=orderbook
            )
            actual_rows.append(actual)
            outcome_rows.extend(outcomes)
            all_paths.extend(paths)
            boundary_rows.append(boundary)
            del surfaces
        actual = DataFrame.from_records(actual_rows).sort_values(
            "sample_selection_order"
        )
        outcomes = DataFrame.from_records(outcome_rows).sort_values(
            ["sample_selection_order", "horizon_minutes"]
        )
        paths = DataFrame.from_records(all_paths).sort_values(
            ["episode_id", "checkpoint_minutes"]
        )
        boundaries = DataFrame.from_records(boundary_rows).sort_values(
            "sample_selection_order"
        )
        calls = evaluated_calls(causal_calls(actual), outcomes)
        scores = call_summary(calls)
        periods = period_summary(outcomes)
        summary = compact_summary(actual, outcomes, scores, boundaries)
        outputs = {
            "contact_state": artifact_dir / "g16_one_minute_contact_state.csv",
            "horizon_outcomes": artifact_dir / "g16_one_minute_horizon_outcomes.csv",
            "checkpoint_paths": artifact_dir / "g16_one_minute_checkpoint_paths.csv",
            "direction_calls": run_dir / "g16_one_minute_direction_calls.csv",
            "direction_scores": run_dir / "g16_one_minute_direction_scores.csv",
            "period_summary": run_dir / "g16_one_minute_period_summary.csv",
            "boundary_audit": run_dir / "g16_one_minute_boundary_audit.csv",
            "analysis_coverage": run_dir / "g16_one_minute_analysis_coverage.csv",
        }
        frames = {
            "contact_state": actual,
            "horizon_outcomes": outcomes,
            "checkpoint_paths": paths,
            "direction_calls": calls,
            "direction_scores": scores,
            "period_summary": periods,
            "boundary_audit": boundaries,
            "analysis_coverage": coverage,
        }
        for name, frame in frames.items():
            g0.atomic_write_csv(frame, outputs[name])
        summary_path = run_dir / "g16_one_minute_summary.json"
        g0.atomic_write_json(summary, summary_path)
        record.update(
            {
                "status": summary["status"],
                "completed_at_utc": g0.utc_now(),
                "summary": summary,
                "artifacts": {name: artifact(path) for name, path in outputs.items()}
                | {"summary": artifact(summary_path)},
            }
        )
        g0.atomic_write_json(record, record_path)
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "failed_at_utc": g0.utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        g0.atomic_write_json(record, record_path)
        raise
    print(json.dumps(summary, indent=2, default=g0.json_default))
    return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    return run_analysis(str(args.run_id))


if __name__ == "__main__":
    raise SystemExit(main())

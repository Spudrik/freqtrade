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
    market_reaction_zone_generation9_freeze as g9z,
)


REPORT_ROOT = g9z.OUTPUT_ROOT / "generation9_branches" / "g9_one_minute_replay"
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData")
    / "research_outputs"
    / "market_reaction_zones"
    / "generation9_branches"
    / "g9_one_minute_replay"
)
COVERAGE_RECORD_NAME = "g9_one_minute_coverage_record.json"
STALE_HOURS = 72
SHIFT_ATR_MULTIPLE = 2.0
CONTROL_MINUTES = 120
CONTROL_EVENT_SEPARATION_MINUTES = 240
MIN_DIRECTION_EPISODES_FOR_LEAD = 20
JOINT_FLOOR = 0.55
JOINT_MAIN_TARGET = 0.65
CALL_METHODS = (
    "pre_15m_trend",
    "pre_60m_trend",
    "pre_240m_trend",
    "pre_20m_pressure",
    "multi_timeframe_ema_vote",
    "multi_timeframe_pressure_vote",
    "trend_pressure_agreement",
    "btc_orderbook_pressure_if_usable",
    "leave_one_out_cohort_majority",
)
CONTROL_KINDS = (
    "same_state_no_level",
    "causal_stale_level_contact",
    "deterministic_shifted_level_contact",
)
PAIR_METRICS = (
    "excursion_strength_60_half_widths",
    "volume_ratio_post_pre_60m",
    "realized_volatility_ratio_60m",
    "zone_overlap_fraction_60m",
    "reference_crossings_60m",
    "through_excursion_60_half_widths",
    "away_excursion_60_half_widths",
)


def stable_sign(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return 1 if digest[0] % 2 else -1


def validate_inputs(freeze_run_id: str) -> tuple[dict[str, Any], Path, Path, Path]:
    frozen = g9z.validate_existing_freeze(g9z.FREEZE_PATH)
    if frozen["run_id"] != freeze_run_id:
        raise ValueError("The requested one-minute freeze run id does not match Generation 9.")
    freeze_dir = g9z.RECORD_ROOT / freeze_run_id
    coverage_path = freeze_dir / COVERAGE_RECORD_NAME
    if not coverage_path.is_file():
        raise FileNotFoundError(coverage_path)
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if coverage.get("status") != "passed":
        raise ValueError("Generation 9 one-minute coverage has not passed.")
    sample_path = Path(frozen["artifacts"]["diagnostic_sample_csv"]["path"])
    if g0.sha256_file(sample_path) != frozen["artifacts"]["diagnostic_sample_csv"][
        "sha256"
    ]:
        raise ValueError("The frozen Generation 9 diagnostic sample changed.")
    return frozen, g9z.FREEZE_PATH, coverage_path, sample_path


def request_contract(
    *, freeze_run_id: str, freeze_path: Path, coverage_path: Path
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": g0.sha256_file(Path(__file__).resolve()),
        "freeze_run_id": freeze_run_id,
        "freeze_path": str(freeze_path.resolve()),
        "freeze_sha256": g0.sha256_file(freeze_path),
        "coverage_path": str(coverage_path.resolve()),
        "coverage_sha256": g0.sha256_file(coverage_path),
        "objective": (
            "Resolve the first minute-scale away-versus-through behaviour of twelve "
            "direction-blind, independently selected high-volume fresh level contacts."
        ),
        "selection_boundary": (
            "Episodes were selected using level contact, a sustained upper relative-volume "
            "band, and next-hour volume magnitude only; no future price direction was read."
        ),
        "movement_origin": (
            "The first frozen-zone contact minute is located inside the parent hour. Price "
            "direction begins on the next minute from the contact-minute close."
        ),
        "reaction_definition": {
            "horizon_minutes": g3a.REACTION_HORIZON_MINUTES,
            "minimum_absolute_excursion_zone_half_widths": (
                g3a.REACTION_EXCURSION_HALF_WIDTHS
            ),
            "minimum_post_to_pre_volume_ratio": g3a.REACTION_VOLUME_RATIO_MINIMUM,
        },
        "controls": {
            "same_state_no_level": (
                "Earlier same-pair minute with the same trend sign, closest causal market "
                "state, and no contact with any causally available calculated level."
            ),
            "causal_stale_level_contact": (
                "Contact during the pre-event window with the same indicator column frozen "
                "from a value available at least 72 hours before that window."
            ),
            "deterministic_shifted_level_contact": (
                "Contact during the pre-event window with an artificial price shifted exactly "
                "two causal base ATRs up or down by a frozen episode hash."
            ),
        },
        "direction_call_methods": list(CALL_METHODS),
        "call_availability": (
            "A method abstains when its pre-contact source is unavailable or tied."
        ),
        "reported_targets": {
            "acceptable_joint_floor": JOINT_FLOOR,
            "main_joint_target": JOINT_MAIN_TARGET,
            "minimum_independent_episodes_for_a_lead": MIN_DIRECTION_EPISODES_FOR_LEAD,
        },
        "checkpoints_by_anchor_minutes": {
            timeframe: list(g3a.CHECKPOINT_MINUTES[timeframe])
            for timeframe in g9z.ONE_MINUTE_TIMEFRAMES
        },
        "technical_inputs": list(g3a.TECHNICAL_DEFINITIONS),
        "technical_availability": "Last fully completed bar before contact.",
        "worker_count": 1,
        "freqai_training": False,
        "profit_optimization": False,
        "promotion_allowed": False,
    }


def episode_identity(episode: Series) -> dict[str, Any]:
    return {
        "episode_id": str(episode["episode_id"]),
        "sample_selection_order": int(episode["sample_selection_order"]),
        "cohort": str(episode["cohort"]),
        "period": str(episode["period"]),
        "pair": str(episode["pair"]),
        "parent_contact_hour": pd.Timestamp(episode["event_time"]),
        "source_timeframe": str(episode["source_timeframe"]),
        "level_family": str(episode["level_family"]),
        "level_name": str(episode["level_name"]),
        "level_column": str(episode["level_column"]),
        "representation": str(episode["representation"]),
        "approach_state": str(episode["approach_state"]),
        "source_open": pd.Timestamp(episode["source_open"]),
        "source_available_at": pd.Timestamp(episode["source_available_at"]),
        "level_price": float(episode["level_price"]),
        "zone_half_width": float(episode["zone_half_width"]),
        "zone_half_width_atr": float(episode["zone_half_width_atr"]),
        "base_atr": float(episode["base_atr"]),
        "parent_relative_volume": float(
            episode["g8_participation_relative_volume__relative_volume"]
        ),
        "parent_high_volume_state_duration_hours": float(
            episode[
                "g8_participation_duration__relative_volume_band_duration_hours"
            ]
        ),
        "parent_next_hour_volume_ratio": float(episode[g9z.VOLUME_TARGET]),
    }


def attach_parent_volume_ratio(sample: DataFrame) -> DataFrame:
    outputs: list[DataFrame] = []
    for cohort, cohort_sample in sample.groupby("cohort", observed=True):
        manifest = g9z.g8.cache_manifest(str(cohort))
        inventory = {str(item["pair"]): item for item in manifest["inventory"]}
        target_parts: list[DataFrame] = []
        for pair in cohort_sample["pair"].astype(str).unique():
            item = inventory[pair]
            target = pd.read_parquet(
                item["event_path"], columns=["date", g9z.VOLUME_TARGET]
            )
            target["date"] = pd.to_datetime(target["date"], utc=True)
            target["pair"] = pair
            target_parts.append(target)
        targets = pd.concat(target_parts, ignore_index=True)
        selected = cohort_sample.merge(
            targets,
            left_on=["pair", "event_time"],
            right_on=["pair", "date"],
            how="left",
            validate="one_to_one",
        ).drop(columns="date")
        values = pd.to_numeric(selected[g9z.VOLUME_TARGET], errors="coerce")
        if values.isna().any() or not values.ge(g9z.VOLUME_REACTION_THRESHOLD).all():
            raise ValueError("Frozen episodes do not reproduce their volume trigger.")
        outputs.append(selected)
    return pd.concat(outputs, ignore_index=True).sort_values(
        "sample_selection_order"
    )


def through_level_name(approach_state: str) -> str:
    if approach_state == "from_below":
        return "lvn_above"
    if approach_state == "from_above":
        return "lvn_below"
    raise ValueError(f"Generation 9 requires a clear approach, got {approach_state!r}.")


def path_rows(
    *,
    episode: Series,
    minute: DataFrame,
    timestamp: pd.Timestamp,
    reference_price: float,
    event_kind: str,
    checkpoints: Iterable[int],
) -> list[dict[str, Any]]:
    return g3a.checkpoint_paths(
        minute=minute,
        timestamp=timestamp,
        reference_price=reference_price,
        zone_half_width=float(episode["zone_half_width"]),
        level_name=through_level_name(str(episode["approach_state"])),
        checkpoints=checkpoints,
        episode_id=str(episode["episode_id"]),
        cohort=str(episode["cohort"]),
        event_kind=event_kind,
    )


def control_row(
    *,
    episode: Series,
    timestamp: pd.Timestamp,
    reference_price: float,
    event_kind: str,
    paths: list[dict[str, Any]],
    match: dict[str, float],
    technical: dict[str, float],
    orderbook: dict[str, Any],
    distance_mean: float,
    distance_max: float,
    balance_usable: bool,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        **episode_identity(episode),
        "event_kind": event_kind,
        "contact_time": timestamp,
        "reference_price": reference_price,
        "control_state_distance_mean": distance_mean,
        "control_state_distance_max": distance_max,
        "control_balance_usable": balance_usable,
        **(source or {}),
        **match,
        **technical,
        **orderbook,
        **g3a.primary_event_outcomes(paths, match),
    }


def model_cache_path(episode: Series) -> Path:
    manifest_path = g8_manifest_path(str(episode["cohort"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_dir = Path(manifest["storage"]["cache_dir"])
    stem = g0.pair_file_stem(str(episode["pair"]))
    return cache_dir / f"{stem}-{episode['source_timeframe']}-core-generic.parquet"


def g8_manifest_path(cohort: str) -> Path:
    return g9z.g8.g8p.G6_SOURCE_MANIFESTS[cohort]


def stale_level(episode: Series, *, visible_start: pd.Timestamp) -> dict[str, Any]:
    path = model_cache_path(episode)
    column = str(episode["level_column"])
    if not path.is_file():
        return {"available": False, "reason": "causal_level_cache_missing"}
    try:
        frame = pd.read_parquet(path, columns=["source_open", "available_at", column])
    except Exception as exc:
        return {"available": False, "reason": f"level_column_unavailable:{type(exc).__name__}"}
    frame["source_open"] = pd.to_datetime(frame["source_open"], utc=True)
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    values = pd.to_numeric(frame[column], errors="coerce")
    cutoff = visible_start - pd.Timedelta(hours=STALE_HOURS)
    eligible = frame.loc[frame["available_at"].le(cutoff) & values.notna()].copy()
    if eligible.empty:
        return {"available": False, "reason": "no_value_available_before_stale_cutoff"}
    row = eligible.iloc[-1]
    return {
        "available": True,
        "reference_price": float(row[column]),
        "source_open": pd.Timestamp(row["source_open"]),
        "source_available_at": pd.Timestamp(row["available_at"]),
        "stale_cutoff": cutoff,
        "cache_path": str(path),
    }


def select_level_contact_control(
    *,
    episode: Series,
    surfaces: g3a.PairSurfaces,
    actual_contact_time: pd.Timestamp,
    actual_match: dict[str, float],
    reference_price: float,
    visible_start: pd.Timestamp,
    source_available_at: pd.Timestamp | None,
) -> dict[str, Any]:
    half_width = float(episode["zone_half_width"])
    latest = actual_contact_time - pd.Timedelta(minutes=CONTROL_EVENT_SEPARATION_MINUTES)
    minute = surfaces.minute
    contact = minute["high"].ge(reference_price - half_width) & minute["low"].le(
        reference_price + half_width
    )
    candidates = minute.loc[
        contact
        & minute["date"].ge(visible_start)
        & minute["date"].le(latest)
        & minute["date"].dt.minute.mod(5).eq(0),
        ["date"],
    ].merge(surfaces.match, on="date", how="inner", validate="one_to_one")
    if source_available_at is not None:
        minimum_causal_time = source_available_at + pd.Timedelta(hours=STALE_HOURS)
        candidates = candidates.loc[candidates["date"].ge(minimum_causal_time)]
    candidates = candidates.dropna(subset=list(g3a.MATCH_FEATURES))
    if candidates.empty:
        raise ValueError("no_contact_candidates_in_frozen_pre_window")
    actual_vector = np.asarray(
        [actual_match[feature] for feature in g3a.MATCH_FEATURES], dtype=float
    )
    if not np.isfinite(actual_vector).all():
        raise ValueError("actual_match_state_incomplete")
    trend_sign = int(np.sign(actual_match["match_return_60m"]))
    candidates = candidates.loc[
        np.sign(candidates["match_return_60m"].to_numpy(dtype=float)) == trend_sign
    ].copy()
    if candidates.empty:
        raise ValueError("no_same_trend_contact_candidates")
    scale_source = surfaces.match.loc[
        surfaces.match["date"].ge(visible_start)
        & surfaces.match["date"].le(latest)
    ].dropna(subset=list(g3a.MATCH_FEATURES))
    scales = g3a.robust_scales(
        scale_source[list(g3a.MATCH_FEATURES)].to_numpy(dtype=float)
    )
    values = candidates[list(g3a.MATCH_FEATURES)].to_numpy(dtype=float)
    distances = np.abs(values - actual_vector[None, :]) / scales[None, :]
    candidates["distance_mean"] = distances.mean(axis=1)
    candidates["distance_max"] = distances.max(axis=1)
    selected = candidates.sort_values(
        ["distance_mean", "distance_max", "date"]
    ).iloc[0]
    snapshot = {
        feature: float(selected[feature]) for feature in g3a.MATCH_FEATURES
    }
    distance_mean = float(selected["distance_mean"])
    distance_max = float(selected["distance_max"])
    return {
        "timestamp": pd.Timestamp(selected["date"]),
        "match_snapshot": snapshot,
        "distance_mean": distance_mean,
        "distance_max": distance_max,
        "balance_usable": bool(
            distance_mean <= g3a.CONTROL_MEAN_DISTANCE_LIMIT
            and distance_max <= g3a.CONTROL_MAX_DISTANCE_LIMIT
        ),
        "candidate_count": len(candidates),
    }


def optional_level_control(
    *,
    episode: Series,
    surfaces: g3a.PairSurfaces,
    orderbook: DataFrame,
    actual_contact_time: pd.Timestamp,
    actual_match: dict[str, float],
    reference_price: float,
    visible_start: pd.Timestamp,
    event_kind: str,
    source_available_at: pd.Timestamp | None,
    source: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    try:
        selected = select_level_contact_control(
            episode=episode,
            surfaces=surfaces,
            actual_contact_time=actual_contact_time,
            actual_match=actual_match,
            reference_price=reference_price,
            visible_start=visible_start,
            source_available_at=source_available_at,
        )
    except ValueError as exc:
        return None, [], {
            "episode_id": str(episode["episode_id"]),
            "control_kind": event_kind,
            "available": False,
            "reason": str(exc),
        }
    timestamp = selected["timestamp"]
    paths = path_rows(
        episode=episode,
        minute=surfaces.minute,
        timestamp=timestamp,
        reference_price=reference_price,
        event_kind=event_kind,
        checkpoints=(1, 3, 5, 10, 15, 30, 60, CONTROL_MINUTES),
    )
    row = control_row(
        episode=episode,
        timestamp=timestamp,
        reference_price=reference_price,
        event_kind=event_kind,
        paths=paths,
        match=selected["match_snapshot"],
        technical=g3a.technical_snapshot(surfaces.technical, timestamp),
        orderbook=g3a.orderbook_snapshot(orderbook, timestamp),
        distance_mean=selected["distance_mean"],
        distance_max=selected["distance_max"],
        balance_usable=selected["balance_usable"],
        source=source,
    )
    audit = {
        "episode_id": str(episode["episode_id"]),
        "control_kind": event_kind,
        "available": True,
        "reason": "",
        "candidate_count": selected["candidate_count"],
        "control_time": timestamp,
        "distance_mean": selected["distance_mean"],
        "distance_max": selected["distance_max"],
        "balance_usable": selected["balance_usable"],
    }
    return row, paths, audit


def analyze_episode(
    episode: Series,
    *,
    surfaces: g3a.PairSurfaces,
    orderbook: DataFrame,
) -> dict[str, Any]:
    contact_time = g3a.first_parent_zone_contact(episode, surfaces.minute)
    anchor = str(episode["cluster_causal_anchor_timeframe"])
    checkpoints = g3a.CHECKPOINT_MINUTES[anchor]
    actual_match = g3a.match_snapshot(surfaces.match, contact_time)
    actual_technical = g3a.technical_snapshot(surfaces.technical, contact_time)
    actual_orderbook = g3a.orderbook_snapshot(orderbook, contact_time)
    actual_paths = path_rows(
        episode=episode,
        minute=surfaces.minute,
        timestamp=contact_time,
        reference_price=float(episode["level_price"]),
        event_kind="actual_fresh_level_contact",
        checkpoints=checkpoints,
    )
    actual = {
        **episode_identity(episode),
        "event_kind": "actual_fresh_level_contact",
        "contact_time": contact_time,
        "reference_price": float(episode["level_price"]),
        "control_state_distance_mean": math.nan,
        "control_state_distance_max": math.nan,
        "control_balance_usable": False,
        **actual_match,
        **actual_technical,
        **actual_orderbook,
        **g3a.primary_event_outcomes(actual_paths, actual_match),
    }
    controls: list[dict[str, Any]] = []
    paths: list[dict[str, Any]] = list(actual_paths)
    audits: list[dict[str, Any]] = []
    try:
        no_level = g3a.select_no_level_control(
            episode=episode,
            actual_contact_time=contact_time,
            actual_match=actual_match,
            surfaces=surfaces,
            window_hours=g3m.WINDOW_HOURS,
        )
        no_level_time = pd.Timestamp(no_level["timestamp"])
        no_level_reference = float(
            g3a.minute_row_at(surfaces.minute, no_level_time)["open"]
        )
        no_level_paths = path_rows(
            episode=episode,
            minute=surfaces.minute,
            timestamp=no_level_time,
            reference_price=no_level_reference,
            event_kind="same_state_no_level",
            checkpoints=(1, 3, 5, 10, 15, 30, 60, CONTROL_MINUTES),
        )
        controls.append(
            control_row(
                episode=episode,
                timestamp=no_level_time,
                reference_price=no_level_reference,
                event_kind="same_state_no_level",
                paths=no_level_paths,
                match=no_level["match_snapshot"],
                technical=g3a.technical_snapshot(surfaces.technical, no_level_time),
                orderbook=g3a.orderbook_snapshot(orderbook, no_level_time),
                distance_mean=float(no_level["distance_mean"]),
                distance_max=float(no_level["distance_max"]),
                balance_usable=bool(no_level["balance_usable"]),
            )
        )
        paths.extend(no_level_paths)
        audits.append(
            {
                "episode_id": str(episode["episode_id"]),
                "control_kind": "same_state_no_level",
                "available": True,
                "reason": "",
                "control_time": no_level_time,
                "distance_mean": no_level["distance_mean"],
                "distance_max": no_level["distance_max"],
                "balance_usable": no_level["balance_usable"],
            }
        )
    except ValueError as exc:
        audits.append(
            {
                "episode_id": str(episode["episode_id"]),
                "control_kind": "same_state_no_level",
                "available": False,
                "reason": str(exc),
            }
        )
    pre_hours, _ = g3m.WINDOW_HOURS[anchor]
    visible_start = pd.Timestamp(episode["event_time"]) - pd.Timedelta(hours=pre_hours)
    stale = stale_level(episode, visible_start=visible_start)
    if stale["available"]:
        row, control_paths, audit = optional_level_control(
            episode=episode,
            surfaces=surfaces,
            orderbook=orderbook,
            actual_contact_time=contact_time,
            actual_match=actual_match,
            reference_price=float(stale["reference_price"]),
            visible_start=visible_start,
            event_kind="causal_stale_level_contact",
            source_available_at=pd.Timestamp(stale["source_available_at"]),
            source={
                "control_level_source_open": stale["source_open"],
                "control_level_source_available_at": stale["source_available_at"],
                "control_level_stale_cutoff": stale["stale_cutoff"],
            },
        )
        if row is not None:
            controls.append(row)
            paths.extend(control_paths)
        audits.append(audit)
    else:
        audits.append(
            {
                "episode_id": str(episode["episode_id"]),
                "control_kind": "causal_stale_level_contact",
                "available": False,
                "reason": stale["reason"],
            }
        )
    shift_sign = stable_sign("g9_shift", episode["episode_id"])
    shifted_price = float(episode["level_price"]) + (
        shift_sign * SHIFT_ATR_MULTIPLE * float(episode["base_atr"])
    )
    row, control_paths, audit = optional_level_control(
        episode=episode,
        surfaces=surfaces,
        orderbook=orderbook,
        actual_contact_time=contact_time,
        actual_match=actual_match,
        reference_price=shifted_price,
        visible_start=visible_start,
        event_kind="deterministic_shifted_level_contact",
        source_available_at=None,
        source={
            "control_level_shift_atr": shift_sign * SHIFT_ATR_MULTIPLE,
            "control_level_shift_sign": shift_sign,
        },
    )
    if row is not None:
        controls.append(row)
        paths.extend(control_paths)
    audits.append(audit)
    boundary = g3a.boundary_audit(
        episode=episode,
        contact_time=contact_time,
        minute=surfaces.minute,
        window_hours=g3m.WINDOW_HOURS,
    )
    return {
        "actual": actual,
        "controls": controls,
        "paths": paths,
        "control_audits": audits,
        "boundary": boundary,
    }


def finite_sign(value: Any) -> int:
    numeric = g3a.finite_or_nan(value)
    return int(np.sign(numeric)) if np.isfinite(numeric) else 0


def multi_timeframe_vote(row: Series, feature: str) -> int:
    votes = []
    for prefix in g3a.TIMEFRAME_PREFIX.values():
        value = g3a.finite_or_nan(row.get(f"{prefix}__{feature}"))
        if np.isfinite(value):
            votes.append(int(np.sign(value)))
    return int(np.sign(sum(votes))) if votes else 0


def fixed_direction_calls(actual: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for _, event in actual.iterrows():
        trend60 = finite_sign(event["match_return_60m"])
        pressure = finite_sign(event["match_pressure_20m"])
        orderbook_call = 0
        if bool(event.get("btc_orderbook_usable_coverage", False)):
            orderbook_call = finite_sign(event.get("btc_orderbook_pressure_25bps_last"))
        methods = {
            "pre_15m_trend": finite_sign(event["match_return_15m"]),
            "pre_60m_trend": trend60,
            "pre_240m_trend": finite_sign(event["match_return_240m"]),
            "pre_20m_pressure": pressure,
            "multi_timeframe_ema_vote": multi_timeframe_vote(
                event, "ema12_minus_ema26_pct"
            ),
            "multi_timeframe_pressure_vote": multi_timeframe_vote(
                event, "pressure_20bar"
            ),
            "trend_pressure_agreement": (
                trend60 if trend60 != 0 and trend60 == pressure else 0
            ),
            "btc_orderbook_pressure_if_usable": orderbook_call,
        }
        through_sign = 1 if event["approach_state"] == "from_below" else -1
        actual_direction = int(event["first_direction_numeric"])
        for method, call in methods.items():
            issued = call != 0
            direction_correct = bool(
                issued and bool(event["direction_callable"]) and call == actual_direction
            )
            rows.append(
                {
                    "episode_id": event["episode_id"],
                    "cohort": event["cohort"],
                    "period": event["period"],
                    "pair": event["pair"],
                    "method": method,
                    "call_numeric": call,
                    "call_direction": "up" if call > 0 else "down" if call < 0 else "abstain",
                    "call_path": (
                        "through" if call == through_sign else "away" if issued else "abstain"
                    ),
                    "issued": issued,
                    "reaction_60m": bool(event["reaction_60m"]),
                    "actual_direction_callable": bool(event["direction_callable"]),
                    "actual_first_direction": event["first_direction"],
                    "actual_first_path": event["first_path_relative_to_approach"],
                    "direction_correct": direction_correct,
                    "path_correct": bool(
                        issued
                        and event["first_path_relative_to_approach"]
                        == ("through" if call == through_sign else "away")
                    ),
                    "joint_reaction_and_direction": bool(
                        event["reaction_60m"] and direction_correct
                    ),
                }
            )
    calls = DataFrame.from_records(rows)
    majority_rows: list[dict[str, Any]] = []
    for _, event in actual.iterrows():
        peers = actual.loc[
            actual["cohort"].eq(event["cohort"])
            & ~actual["episode_id"].eq(event["episode_id"])
            & actual["direction_callable"]
        ]
        call = int(np.sign(peers["first_direction_numeric"].sum())) if len(peers) else 0
        through_sign = 1 if event["approach_state"] == "from_below" else -1
        issued = call != 0
        correct = bool(
            issued
            and bool(event["direction_callable"])
            and call == int(event["first_direction_numeric"])
        )
        majority_rows.append(
            {
                "episode_id": event["episode_id"],
                "cohort": event["cohort"],
                "period": event["period"],
                "pair": event["pair"],
                "method": "leave_one_out_cohort_majority",
                "call_numeric": call,
                "call_direction": "up" if call > 0 else "down" if call < 0 else "abstain",
                "call_path": (
                    "through" if call == through_sign else "away" if issued else "abstain"
                ),
                "issued": issued,
                "reaction_60m": bool(event["reaction_60m"]),
                "actual_direction_callable": bool(event["direction_callable"]),
                "actual_first_direction": event["first_direction"],
                "actual_first_path": event["first_path_relative_to_approach"],
                "direction_correct": correct,
                "path_correct": bool(
                    issued
                    and event["first_path_relative_to_approach"]
                    == ("through" if call == through_sign else "away")
                ),
                "joint_reaction_and_direction": bool(event["reaction_60m"] and correct),
            }
        )
    return pd.concat([calls, DataFrame.from_records(majority_rows)], ignore_index=True)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return centre - margin, centre + margin


def call_summary(calls: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = [("all", calls), *list(calls.groupby("cohort", observed=True))]
    for scope, scoped in scopes:
        for method, frame in scoped.groupby("method", observed=True):
            issued = frame.loc[frame["issued"]]
            reacted = issued.loc[
                issued["reaction_60m"] & issued["actual_direction_callable"]
            ]
            correct = int(reacted["direction_correct"].sum())
            joint = int(frame["joint_reaction_and_direction"].sum())
            direction_lower, direction_upper = wilson_interval(correct, len(reacted))
            joint_lower, joint_upper = wilson_interval(joint, len(frame))
            joint_rate = joint / len(frame) if len(frame) else math.nan
            rows.append(
                {
                    "scope": scope,
                    "method": method,
                    "independent_episodes": int(frame["episode_id"].nunique()),
                    "issued_calls": len(issued),
                    "abstentions": len(frame) - len(issued),
                    "issued_call_coverage": len(issued) / len(frame),
                    "reaction_rate": float(frame["reaction_60m"].mean()),
                    "reacted_callable_issued": len(reacted),
                    "direction_correct_given_reaction": correct,
                    "conditional_direction_accuracy": (
                        correct / len(reacted) if len(reacted) else math.nan
                    ),
                    "conditional_direction_wilson_lower": direction_lower,
                    "conditional_direction_wilson_upper": direction_upper,
                    "joint_successes": joint,
                    "joint_success_rate": joint_rate,
                    "joint_wilson_lower": joint_lower,
                    "joint_wilson_upper": joint_upper,
                    "observed_at_or_above_55pct": bool(joint_rate >= JOINT_FLOOR),
                    "observed_at_or_above_65pct": bool(joint_rate >= JOINT_MAIN_TARGET),
                    "lead_eligible_sample_size": bool(
                        frame["episode_id"].nunique()
                        >= MIN_DIRECTION_EPISODES_FOR_LEAD
                    ),
                    "status": "diagnostic_only_insufficient_independent_episodes",
                }
            )
    return DataFrame.from_records(rows)


def control_comparisons(actual: DataFrame, controls: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for control_kind in CONTROL_KINDS:
        control = controls.loc[controls["event_kind"].eq(control_kind)]
        for cohort in ("all", "normal", "meme"):
            left = actual if cohort == "all" else actual.loc[actual["cohort"].eq(cohort)]
            right = control if cohort == "all" else control.loc[control["cohort"].eq(cohort)]
            paired = left.merge(
                right,
                on="episode_id",
                how="inner",
                suffixes=("__actual", "__control"),
                validate="one_to_one",
            )
            for metric in PAIR_METRICS:
                actual_values = pd.to_numeric(paired[f"{metric}__actual"], errors="coerce")
                control_values = pd.to_numeric(paired[f"{metric}__control"], errors="coerce")
                finite = np.isfinite(actual_values) & np.isfinite(control_values)
                difference = actual_values.loc[finite] - control_values.loc[finite]
                rows.append(
                    {
                        "cohort": cohort,
                        "control_kind": control_kind,
                        "metric": metric,
                        "paired_episodes": len(difference),
                        "actual_median": float(actual_values.loc[finite].median())
                        if finite.any()
                        else math.nan,
                        "control_median": float(control_values.loc[finite].median())
                        if finite.any()
                        else math.nan,
                        "median_paired_difference": float(difference.median())
                        if len(difference)
                        else math.nan,
                        "actual_greater_fraction": float(difference.gt(0.0).mean())
                        if len(difference)
                        else math.nan,
                    }
                )
    return DataFrame.from_records(rows)


def compact_summary(
    *,
    actual: DataFrame,
    controls: DataFrame,
    audits: DataFrame,
    calls: DataFrame,
    call_scores: DataFrame,
    boundaries: DataFrame,
) -> dict[str, Any]:
    overall = call_scores.loc[call_scores["scope"].eq("all")].sort_values(
        ["joint_success_rate", "issued_call_coverage"], ascending=False
    )
    best = overall.iloc[0].to_dict() if len(overall) else {}
    return {
        "status": "completed_generation9_one_minute_diagnostic",
        "actual_episodes": len(actual),
        "independent_pairs": int(actual["pair"].nunique()),
        "normal_episodes": int(actual["cohort"].eq("normal").sum()),
        "meme_episodes": int(actual["cohort"].eq("meme").sum()),
        "reaction_rate_60m": float(actual["reaction_60m"].mean()),
        "first_direction_counts": actual["first_direction"].value_counts().to_dict(),
        "first_path_counts": actual["first_path_relative_to_approach"].value_counts().to_dict(),
        "first_zone_resolution_counts": actual["first_zone_resolution"].value_counts().to_dict(),
        "control_availability": {
            kind: int(
                audits.loc[
                    audits["control_kind"].eq(kind) & audits["available"]
                ].shape[0]
            )
            for kind in CONTROL_KINDS
        },
        "control_balance_usable": {
            kind: int(
                controls.loc[
                    controls["event_kind"].eq(kind)
                    & controls["control_balance_usable"]
                ].shape[0]
            )
            for kind in CONTROL_KINDS
        },
        "boundary_extension_indicated": int(
            boundaries["boundary_extension_indicated"].sum()
        ),
        "direction_methods": int(calls["method"].nunique()),
        "best_observed_method_not_a_lead": best,
        "joint_target_interpretation": (
            "Observed percentages are diagnostic only. Twelve frozen episodes are below "
            "the predeclared minimum for a direction lead, so neither 55% nor 65% can be "
            "claimed even if a point estimate crosses those numbers."
        ),
        "selection_conditioning_warning": (
            "The sample was intentionally conditioned on a large next-hour volume response. "
            "Its reaction rate is not an estimate for ordinary level contacts."
        ),
        "freqai_trained": False,
        "profit_used": False,
        "promotion_allowed": False,
    }


def run_analysis(*, run_id: str, freeze_run_id: str) -> int:
    _, freeze_path, coverage_path, sample_path = validate_inputs(freeze_run_id)
    run_dir = REPORT_ROOT / run_id
    artifact_dir = ARTIFACT_ROOT / run_id
    record_path = run_dir / "g9_one_minute_analysis_record.json"
    if record_path.is_file():
        raise FileExistsError(record_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    request = request_contract(
        freeze_run_id=freeze_run_id,
        freeze_path=freeze_path,
        coverage_path=coverage_path,
    )
    record: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "started_at_utc": g0.utc_now(),
        "request_contract": request,
        "request_sha256": hashlib.sha256(
            json.dumps(request, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    g0.atomic_write_json(record, record_path)
    try:
        sample = pd.read_csv(sample_path)
        for column in ("event_time", "source_open", "source_available_at"):
            sample[column] = pd.to_datetime(sample[column], utc=True)
        sample = attach_parent_volume_ratio(sample)
        coverage = g3a.audit_analysis_coverage(
            sample=sample, window_hours=g3m.WINDOW_HOURS
        )
        if not bool(coverage["passed"].all()):
            raise ValueError("The effective Generation 9 analysis windows are incomplete.")
        orderbook = g3a.load_orderbook_surface()
        actual_rows: list[dict[str, Any]] = []
        control_rows: list[dict[str, Any]] = []
        path_rows_output: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        boundary_rows: list[dict[str, Any]] = []
        for _, episode in sample.sort_values("sample_selection_order").iterrows():
            pair = str(episode["pair"])
            # The frozen sample deliberately uses twelve distinct pairs. Retaining every
            # full minute/technical surface therefore adds memory without reuse.
            surfaces = g3a.load_pair_surfaces(pair, cohort=str(episode["cohort"]))
            result = analyze_episode(
                episode, surfaces=surfaces, orderbook=orderbook
            )
            actual_rows.append(result["actual"])
            control_rows.extend(result["controls"])
            path_rows_output.extend(result["paths"])
            audit_rows.extend(result["control_audits"])
            boundary_rows.append(result["boundary"])
            del surfaces
        actual = DataFrame.from_records(actual_rows).sort_values(
            "sample_selection_order"
        )
        controls = DataFrame.from_records(control_rows).sort_values(
            ["sample_selection_order", "event_kind"]
        )
        paths = DataFrame.from_records(path_rows_output).sort_values(
            ["cohort", "episode_id", "event_kind", "checkpoint_minutes"]
        )
        audits = DataFrame.from_records(audit_rows).sort_values(
            ["control_kind", "episode_id"]
        )
        boundaries = DataFrame.from_records(boundary_rows).sort_values(
            "sample_selection_order"
        )
        calls = fixed_direction_calls(actual)
        call_scores = call_summary(calls)
        comparisons = control_comparisons(actual, controls)
        summary = compact_summary(
            actual=actual,
            controls=controls,
            audits=audits,
            calls=calls,
            call_scores=call_scores,
            boundaries=boundaries,
        )
        paths_by_name = {
            "actual_events": artifact_dir / "g9_one_minute_actual_events.csv",
            "control_events": artifact_dir / "g9_one_minute_control_events.csv",
            "checkpoint_paths": artifact_dir / "g9_one_minute_checkpoint_paths.csv",
            "control_audit": run_dir / "g9_one_minute_control_audit.csv",
            "boundary_audit": run_dir / "g9_one_minute_boundary_audit.csv",
            "analysis_coverage": run_dir / "g9_one_minute_analysis_coverage.csv",
            "direction_calls": run_dir / "g9_one_minute_direction_calls.csv",
            "direction_scores": run_dir / "g9_one_minute_direction_scores.csv",
            "control_comparisons": run_dir / "g9_one_minute_control_comparisons.csv",
        }
        frames = {
            "actual_events": actual,
            "control_events": controls,
            "checkpoint_paths": paths,
            "control_audit": audits,
            "boundary_audit": boundaries,
            "analysis_coverage": coverage,
            "direction_calls": calls,
            "direction_scores": call_scores,
            "control_comparisons": comparisons,
        }
        for name, frame in frames.items():
            g0.atomic_write_csv(frame, paths_by_name[name])
        summary_path = run_dir / "g9_one_minute_summary.json"
        g0.atomic_write_json(summary, summary_path)
        record.update(
            {
                "status": summary["status"],
                "completed_at_utc": g0.utc_now(),
                "summary": summary,
                "artifacts": {
                    name: g3m.artifact_record(path)
                    for name, path in paths_by_name.items()
                }
                | {"summary": g3m.artifact_record(summary_path)},
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Generation 9 one-minute away-versus-through diagnostic."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--freeze-run-id", default=g9z.DEFAULT_RUN_ID)
    args = parser.parse_args(argv)
    return run_analysis(run_id=args.run_id, freeze_run_id=args.freeze_run_id)


if __name__ == "__main__":
    raise SystemExit(main())

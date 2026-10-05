"""Run Generation 21's disjoint Donchian one-minute rejection diagnostic."""

from __future__ import annotations

# Bind numerical pools before pandas/numpy imports.
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
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation20_one_minute_replay as g20m,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation20_state_attribution as g20s,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation21_freeze as g21z,
)


ANALYSIS_PATH = Path(__file__).resolve()
DEFAULT_RUN_ID = "g21_disjoint_one_minute_20260827a"
RECORD_ROOT = g21z.OUTPUT_ROOT / "one_minute_replay"
SELECTION_RECORD = RECORD_ROOT / "g21_one_minute_selection_freeze.json"
SELECTION_CSV = RECORD_ROOT / "g21_one_minute_frozen_episodes.csv"
COVERAGE_CSV = RECORD_ROOT / "g21_one_minute_coverage.csv"
PRE_WINDOW_HOURS = 4
POST_WINDOW_HOURS = 4
INDEPENDENCE_HOURS = 24
MINIMUM_EPISODES = 12
MAXIMUM_EPISODES = 18
DISTANCE_THRESHOLDS_ATR = g21z.ONE_MINUTE_THRESHOLDS_ATR
HORIZON_MINUTES = g21z.ONE_MINUTE_HORIZONS
JOINT_FLOOR = 0.55
MINIMUM_CALL_COVERAGE = 0.50
FIXED_CALLS = g20m.FIXED_CALLS
CONTROLS = ("naive_rejection", "majority_path", "matched_no_level_episode")
B_STATE_COLUMNS = g20m.B_STATE_COLUMNS


def artifact(path: Path) -> dict[str, Any]:
    return g21z.artifact(path)


def load_freeze() -> dict[str, Any]:
    frozen = json.loads(g21z.FREEZE_PATH.read_text(encoding="utf-8"))
    branch = next(
        item
        for item in frozen["branches"]
        if item["branch_id"] == "g21e_disjoint_one_minute_rejection_diagnostic"
    )
    selection = branch["selection"]
    if frozen.get("status") != "frozen_before_generation21_outcomes":
        raise ValueError("Generation 21 batch is not frozen.")
    if tuple(branch["candidate_calls"]) != FIXED_CALLS:
        raise ValueError("Generation 21 one-minute call registry drifted.")
    if tuple(branch["controls"]) != CONTROLS:
        raise ValueError("Generation 21 one-minute control registry drifted.")
    if tuple(branch["distance_thresholds_atr"]) != DISTANCE_THRESHOLDS_ATR:
        raise ValueError("Generation 21 one-minute distance registry drifted.")
    if tuple(branch["horizons_minutes"]) != HORIZON_MINUTES:
        raise ValueError("Generation 21 one-minute horizon registry drifted.")
    if selection["minimum_episodes"] != MINIMUM_EPISODES or selection[
        "maximum_episodes"
    ] != MAXIMUM_EPISODES:
        raise ValueError("Generation 21 one-minute sample bounds drifted.")
    return frozen


def prior_g20_actual_episodes() -> DataFrame:
    g20m.freeze_selection(overwrite=False)
    frame = pd.read_csv(g20m.SELECTION_CSV)
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
    return frame.loc[frame["episode_kind"].eq("actual_level")].copy()


def exclude_prior_episode_neighbourhoods(
    candidates: DataFrame, prior: DataFrame
) -> DataFrame:
    keep = np.ones(len(candidates), dtype=bool)
    times = pd.to_datetime(candidates["event_time"], utc=True)
    for pair, old in prior.groupby("pair", observed=True, sort=False):
        old_times = pd.to_datetime(old["event_time"], utc=True)
        indexes = np.flatnonzero(candidates["pair"].eq(pair).to_numpy())
        for index in indexes:
            distance = (old_times - times.iloc[index]).abs()
            if distance.lt(pd.Timedelta(hours=INDEPENDENCE_HOURS)).any():
                keep[index] = False
    return candidates.loc[keep].copy()


def independent_selection(candidates: DataFrame, maximum: int) -> DataFrame:
    source = candidates.copy()
    source["selection_hash"] = [
        g0.stable_hash_int(
            "|".join(
                (
                    "g21-disjoint-one-minute-selection",
                    str(row.pair),
                    pd.Timestamp(row.event_time).isoformat(),
                    str(row.level_name),
                )
            )
        )
        for row in source.itertuples(index=False)
    ]
    source.sort_values(
        ["selection_hash", "pair", "period", "event_time"],
        inplace=True,
        kind="stable",
    )
    selected: list[int] = []
    selected_set: set[int] = set()
    used: dict[str, list[pd.Timestamp]] = {}
    strata = list(source.groupby(["period", "pair"], observed=True, sort=True))
    while len(selected) < maximum:
        progressed = False
        for _, stratum in strata:
            for index, row in stratum.iterrows():
                if index in selected_set:
                    continue
                event_time = pd.Timestamp(row["event_time"])
                pair = str(row["pair"])
                if any(
                    abs(event_time - prior) < pd.Timedelta(hours=INDEPENDENCE_HOURS)
                    for prior in used.get(pair, [])
                ):
                    continue
                selected.append(index)
                selected_set.add(index)
                used.setdefault(pair, []).append(event_time)
                progressed = True
                break
            if len(selected) >= maximum:
                break
        if not progressed:
            break
    result = source.loc[selected].copy().reset_index(drop=True)
    result["sample_selection_order"] = np.arange(1, len(result) + 1)
    return result


def freeze_selection(*, overwrite: bool = False) -> dict[str, Any]:
    load_freeze()
    if SELECTION_RECORD.is_file() and not overwrite:
        record = json.loads(SELECTION_RECORD.read_text(encoding="utf-8"))
        if record.get("status") != "frozen_before_generation21_signed_paths":
            raise ValueError("Invalid Generation 21 one-minute selection freeze.")
        return record
    events = g20m.load_state_support_with_reactions()
    candidates = events.loc[
        events["control"].eq("actual")
        & events["approach_state"].isin({"from_below", "from_above"})
        & pd.to_numeric(events["pre_crossings_h4"], errors="coerce").eq(0.0)
        & pd.to_numeric(events["metric__crossing_count_h4"], errors="coerce").ge(1.0)
    ].copy()
    candidates = candidates.sort_values(
        ["pair", "event_time", "pre_distance_atr", "level_name"], kind="stable"
    ).drop_duplicates(["pair", "event_time"], keep="first")
    prior = prior_g20_actual_episodes()
    candidates = exclude_prior_episode_neighbourhoods(candidates.reset_index(drop=True), prior)
    selected = independent_selection(candidates, MAXIMUM_EPISODES)
    if len(selected) < MINIMUM_EPISODES:
        raise ValueError(
            f"Only {len(selected)} disjoint independent Donchian episodes were available."
        )
    selected["episode_id"] = [
        f"g21e-{number:03d}" for number in range(1, len(selected) + 1)
    ]
    selected["episode_kind"] = "actual_level"
    pool = events.loc[
        events["control"].eq("breakout_state_matched_no_current_boundary_contact")
    ].copy()
    controls = g20m.matched_no_level_rows(selected, pool)
    if len(controls) < MINIMUM_EPISODES:
        raise ValueError(
            f"Only {len(controls)} no-level controls matched the frozen episodes."
        )
    matched_ids = set(controls["matched_to_episode_id"].astype(str))
    selected = selected.loc[selected["episode_id"].isin(matched_ids)].copy()
    controls = controls.loc[controls["matched_to_episode_id"].isin(matched_ids)].copy()
    controls["episode_id"] = "control-" + controls["matched_to_episode_id"].astype(str)
    controls["episode_kind"] = "matched_no_level"
    controls["sample_selection_order"] = controls["matched_to_episode_id"].map(
        selected.set_index("episode_id")["sample_selection_order"]
    )
    columns = [
        "episode_id",
        "episode_kind",
        "matched_to_episode_id",
        "sample_selection_order",
        "cohort",
        "pair",
        "period",
        "event_time",
        "level_family",
        "level_name",
        "level_price",
        "zone_half_width",
        "base_atr",
        "pre_distance_atr",
        "approach_state",
        *B_STATE_COLUMNS,
    ]
    for column in columns:
        if column not in selected.columns:
            selected[column] = np.nan
        if column not in controls.columns:
            controls[column] = np.nan
    sample = pd.concat(
        [selected[columns], controls[columns]], ignore_index=True, sort=False
    ).sort_values(["sample_selection_order", "episode_kind"])
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(sample, SELECTION_CSV)
    record = {
        "schema_version": 1,
        "generation": 21,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation21_signed_paths",
        "branch_id": "g21e_disjoint_one_minute_rejection_diagnostic",
        "actual_episodes": len(selected),
        "matched_no_level_episodes": len(controls),
        "generation20_exclusion_hours_each_side": INDEPENDENCE_HOURS,
        "selection_used_unsigned_reaction": True,
        "selection_used_future_signed_direction": False,
        "selection_used_profit": False,
        "prewindow_hours": PRE_WINDOW_HOURS,
        "postwindow_hours": POST_WINDOW_HOURS,
        "independence_hours": INDEPENDENCE_HOURS,
        "replay_hours_before": 24,
        "replay_hours_after": 24,
        "distance_thresholds_atr": list(DISTANCE_THRESHOLDS_ATR),
        "horizons_minutes": list(HORIZON_MINUTES),
        "fixed_calls": list(FIXED_CALLS),
        "controls": list(CONTROLS),
        "artifact": artifact(SELECTION_CSV),
        "source_contracts": {
            "generation21_freeze": artifact(g21z.FREEZE_PATH),
            "generation20_selection_freeze": artifact(g20m.SELECTION_RECORD),
            "state_support_freeze": artifact(g20s.SUPPORT_MANIFEST),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }
    g0.atomic_write_json(record, SELECTION_RECORD)
    return record


def coverage_audit(sample: DataFrame) -> tuple[DataFrame, bool]:
    rows: list[dict[str, Any]] = []
    minute_cache: dict[str, DataFrame] = {}
    for episode in sample.itertuples(index=False):
        pair = str(episode.pair)
        path = g0.ohlcv_path(pair, "1m")
        if pair not in minute_cache:
            if not path.is_file():
                rows.append(
                    {
                        "episode_id": episode.episode_id,
                        "pair": pair,
                        "source_path": str(path.resolve()),
                        "status": "missing_file",
                        "coverage_ratio": 0.0,
                    }
                )
                continue
            minute_cache[pair] = g0.load_ohlcv(path).sort_values("date").drop_duplicates(
                "date"
            )
            minute_cache[pair]["date"] = pd.to_datetime(
                minute_cache[pair]["date"], utc=True
            )
        minute = minute_cache[pair]
        event_time = pd.Timestamp(episode.event_time)
        start = event_time - pd.Timedelta(hours=24)
        end = event_time + pd.Timedelta(hours=25)
        window = minute.loc[(minute["date"] >= start) & (minute["date"] < end)]
        expected = int((end - start) / pd.Timedelta(minutes=1))
        observed = int(window["date"].nunique())
        coverage = observed / expected
        maximum_gap = (
            float(window["date"].sort_values().diff().dt.total_seconds().max() / 60.0)
            if len(window) > 1
            else np.nan
        )
        passed = bool(
            coverage >= 0.995
            and len(window)
            and window["date"].min() <= start
            and window["date"].max() >= end - pd.Timedelta(minutes=1)
            and np.isfinite(maximum_gap)
            and maximum_gap <= 5.0
        )
        rows.append(
            {
                "episode_id": episode.episode_id,
                "pair": pair,
                "source_path": str(path.resolve()),
                "interval_start_utc": start,
                "interval_end_exclusive_utc": end,
                "expected_rows": expected,
                "observed_rows": observed,
                "coverage_ratio": coverage,
                "maximum_gap_minutes": maximum_gap,
                "status": "passed" if passed else "incomplete",
            }
        )
    result = DataFrame.from_records(rows)
    passed = bool(len(result) == len(sample) and result["status"].eq("passed").all())
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(result, COVERAGE_CSV)
    return result, passed


def episode_calls(episode: Series, minute: DataFrame) -> tuple[pd.Timestamp, list[dict[str, Any]]]:
    contact_time = (
        pd.Timestamp(episode["event_time"])
        if str(episode["episode_kind"]) == "matched_no_level"
        else g20m.first_contact_time(episode, minute)
    )
    pre15 = g20m.completed_window(minute, contact_time, 15)
    pre60 = g20m.completed_window(minute, contact_time, 60)
    trend15 = g20m.trend_call(pre15)
    trend60 = g20m.trend_call(pre60)
    pressure15 = g20m.pressure_call(pre15)
    pressure60 = g20m.pressure_call(pre60)
    momentum15 = g20m.momentum_vote(pre60)
    combined = pressure15 if pressure15 != 0 and pressure15 == momentum15 else 0
    calls = {
        "approach_trend_15m": trend15,
        "approach_trend_60m": trend60,
        "signed_volume_pressure_15m": pressure15,
        "signed_volume_pressure_60m": pressure60,
        "momentum_vote_15m": momentum15,
        "combined_pressure_and_momentum_vote": combined,
    }
    if set(calls) != set(FIXED_CALLS):
        raise ValueError("Generation 21 one-minute call registry drifted.")
    return contact_time, [
        {
            "episode_id": str(episode["episode_id"]),
            "episode_kind": str(episode["episode_kind"]),
            "method": method,
            "method_kind": "causal_candidate",
            "call_numeric": call,
            "issued": call != 0,
        }
        for method, call in calls.items()
    ]


def episode_outcomes(
    episode: Series, minute: DataFrame, contact_time: pd.Timestamp
) -> list[dict[str, Any]]:
    decision_time = contact_time + pd.Timedelta(minutes=1)
    if str(episode["episode_kind"]) == "matched_no_level":
        anchor_row = minute.loc[minute["date"].eq(contact_time)]
        if anchor_row.empty:
            raise ValueError(f"No minute decision anchor for {episode['episode_id']}.")
        anchor_price = float(anchor_row["open"].iloc[0])
    else:
        anchor_price = float(episode["level_price"])
    atr = float(episode["base_atr"])
    if not np.isfinite(anchor_price) or not np.isfinite(atr) or atr <= 0.0:
        raise ValueError(f"Invalid frozen price/ATR for {episode['episode_id']}.")
    approach = str(episode["approach_state"])
    through_direction = 1 if approach == "from_below" else -1
    rows: list[dict[str, Any]] = []
    for threshold_atr in DISTANCE_THRESHOLDS_ATR:
        distance = threshold_atr * atr
        for horizon in HORIZON_MINUTES:
            future = minute.loc[
                (minute["date"] >= decision_time)
                & (minute["date"] < decision_time + pd.Timedelta(minutes=horizon))
            ]
            close = pd.to_numeric(future["close"], errors="coerce").to_numpy(dtype=float)
            above = np.flatnonzero(close >= anchor_price + distance)
            below = np.flatnonzero(close <= anchor_price - distance)
            first_above = int(above[0]) if len(above) else None
            first_below = int(below[0]) if len(below) else None
            if first_above is None and first_below is None:
                target = 0
            elif first_below is None:
                target = 1
            elif first_above is None:
                target = -1
            elif first_above == first_below:
                target = 0
            else:
                target = 1 if first_above < first_below else -1
            path_kind = (
                "through"
                if target != 0 and target == through_direction
                else "rejection"
                if target != 0
                else "unreached_or_tied"
            )
            rows.append(
                {
                    "episode_id": str(episode["episode_id"]),
                    "episode_kind": str(episode["episode_kind"]),
                    "matched_to_episode_id": episode.get("matched_to_episode_id"),
                    "pair": str(episode["pair"]),
                    "period": str(episode["period"]),
                    "contact_time": contact_time,
                    "decision_time": decision_time,
                    "distance_threshold_atr": threshold_atr,
                    "horizon_minutes": horizon,
                    "reaction": target != 0,
                    "first_direction_numeric": target,
                    "path_relative_to_approach": path_kind,
                    "first_above_minute": first_above + 1 if first_above is not None else np.nan,
                    "first_below_minute": first_below + 1 if first_below is not None else np.nan,
                }
            )
    return rows


def control_calls(outcomes: DataFrame, sample: DataFrame) -> DataFrame:
    actual = outcomes.loc[outcomes["episode_kind"].eq("actual_level")].copy()
    sample_by_id = sample.set_index("episode_id")
    rows: list[dict[str, Any]] = []
    for (threshold, horizon), cell in actual.groupby(
        ["distance_threshold_atr", "horizon_minutes"], observed=True, sort=False
    ):
        for event in cell.itertuples(index=False):
            others = cell.loc[
                cell["episode_id"].ne(event.episode_id)
                & cell["first_direction_numeric"].ne(0),
                "first_direction_numeric",
            ]
            majority = g20m.finite_sign(float(others.sum())) if len(others) else 0
            approach = str(sample_by_id.loc[event.episode_id, "approach_state"])
            rejection = -1 if approach == "from_below" else 1 if approach == "from_above" else 0
            for method, call in (("naive_rejection", rejection), ("majority_path", majority)):
                rows.append(
                    {
                        "episode_id": event.episode_id,
                        "episode_kind": "actual_level",
                        "distance_threshold_atr": threshold,
                        "horizon_minutes": horizon,
                        "method": method,
                        "method_kind": "control",
                        "call_numeric": call,
                        "issued": call != 0,
                    }
                )
    return DataFrame.from_records(rows)


def score_methods(
    candidate_calls: DataFrame, controls: DataFrame, outcomes: DataFrame
) -> DataFrame:
    outcome_columns = [
        "episode_id",
        "episode_kind",
        "distance_threshold_atr",
        "horizon_minutes",
        "reaction",
        "first_direction_numeric",
    ]
    candidate_scored = candidate_calls.merge(
        outcomes[outcome_columns],
        on=["episode_id", "episode_kind"],
        how="inner",
        validate="many_to_many",
    )
    control_scored = controls.merge(
        outcomes[outcome_columns],
        on=[
            "episode_id",
            "episode_kind",
            "distance_threshold_atr",
            "horizon_minutes",
        ],
        how="inner",
        # Each outcome has the two predeclared control calls: naive rejection
        # and leave-one-out majority path.
        validate="many_to_one",
    )
    scored = pd.concat([candidate_scored, control_scored], ignore_index=True, sort=False)
    scored["correct"] = (
        scored["issued"]
        & scored["reaction"]
        & scored["call_numeric"].eq(scored["first_direction_numeric"])
    )
    rows: list[dict[str, Any]] = []
    keys = [
        "method_kind",
        "method",
        "episode_kind",
        "distance_threshold_atr",
        "horizon_minutes",
    ]
    for key, cell in scored.groupby(keys, observed=True, sort=False):
        kind, method, episode_kind, threshold, horizon = key
        issued_reacted = cell["issued"] & cell["reaction"]
        rows.append(
            {
                "method_kind": kind,
                "method": method,
                "episode_kind": episode_kind,
                "distance_threshold_atr": threshold,
                "horizon_minutes": horizon,
                "episodes": len(cell),
                "reaction_success_rate": float(cell["reaction"].mean()),
                "conditional_direction_success_rate": float(
                    cell.loc[issued_reacted, "correct"].mean()
                )
                if issued_reacted.any()
                else np.nan,
                "joint_success_rate": float(cell["correct"].mean()),
                "issued_call_coverage": float(cell["issued"].mean()),
                "abstentions": int((~cell["issued"]).sum()),
                "correct_joint_episodes": int(cell["correct"].sum()),
            }
        )
    return DataFrame.from_records(rows)


def decisions(scores: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold in DISTANCE_THRESHOLDS_ATR:
        for horizon in HORIZON_MINUTES:
            cell = scores.loc[
                scores["distance_threshold_atr"].eq(threshold)
                & scores["horizon_minutes"].eq(horizon)
            ]
            actual = cell.loc[
                cell["method_kind"].eq("causal_candidate")
                & cell["episode_kind"].eq("actual_level")
            ].set_index("method")
            no_level = cell.loc[
                cell["method_kind"].eq("causal_candidate")
                & cell["episode_kind"].eq("matched_no_level")
            ].set_index("method")
            controls = cell.loc[
                cell["method_kind"].eq("control")
                & cell["episode_kind"].eq("actual_level")
            ].set_index("method")
            naive = float(controls.loc["naive_rejection", "joint_success_rate"])
            majority = float(controls.loc["majority_path", "joint_success_rate"])
            for method in FIXED_CALLS:
                candidate = actual.loc[method]
                matched = float(no_level.loc[method, "joint_success_rate"])
                rate = float(candidate["joint_success_rate"])
                coverage = float(candidate["issued_call_coverage"])
                retained = bool(
                    int(candidate["episodes"]) >= MINIMUM_EPISODES
                    and rate >= JOINT_FLOOR
                    and coverage >= MINIMUM_CALL_COVERAGE
                    and rate > naive
                    and rate > majority
                    and rate > matched
                )
                rows.append(
                    {
                        "method": method,
                        "distance_threshold_atr": threshold,
                        "horizon_minutes": horizon,
                        "status": (
                            "bounded_point_lead_pending_new_date_confirmation"
                            if retained
                            else "not_retained"
                        ),
                        "actual_joint_success_rate": rate,
                        "actual_call_coverage": coverage,
                        "naive_rejection_joint_success_rate": naive,
                        "majority_path_joint_success_rate": majority,
                        "matched_no_level_joint_success_rate": matched,
                        "at_or_above_55pct": rate >= JOINT_FLOOR,
                        "adequate_coverage": coverage >= MINIMUM_CALL_COVERAGE,
                        "beats_all_controls": rate > max(naive, majority, matched),
                        "retained": retained,
                    }
                )
    return DataFrame.from_records(rows)


def execute(run_id: str, *, overwrite: bool = False) -> int:
    record = freeze_selection(overwrite=False)
    if g0.sha256_file(ANALYSIS_PATH) != record["source_contracts"]["analysis_script"]["sha256"]:
        raise ValueError("Generation 21 one-minute analysis changed after selection freeze.")
    run_dir = RECORD_ROOT / run_id
    result_path = run_dir / "g21_one_minute_replay_result.json"
    if result_path.is_file() and not overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    if g0.sha256_file(SELECTION_CSV) != record["artifact"]["sha256"]:
        raise ValueError("Frozen Generation 21 one-minute sample changed.")
    sample = pd.read_csv(SELECTION_CSV)
    sample["event_time"] = pd.to_datetime(sample["event_time"], utc=True)
    coverage, coverage_passed = coverage_audit(sample)
    if not coverage_passed:
        result = {
            "schema_version": 1,
            "generation": 21,
            "created_at_utc": g0.utc_now(),
            "status": "parked_missing_one_minute_coverage",
            "failed_intervals": int(coverage["status"].ne("passed").sum()),
            "coverage": artifact(COVERAGE_CSV),
            "selection": artifact(SELECTION_RECORD),
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        g0.atomic_write_json(result, result_path)
        print(json.dumps(result, indent=2))
        return 2
    outcomes: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    minute_cache: dict[str, DataFrame] = {}
    for episode in sample.to_dict(orient="records"):
        pair = str(episode["pair"])
        if pair not in minute_cache:
            minute_cache[pair] = g0.load_ohlcv(g0.ohlcv_path(pair, "1m")).sort_values(
                "date"
            )
            minute_cache[pair]["date"] = pd.to_datetime(
                minute_cache[pair]["date"], utc=True
            )
        contact_time, episode_call_rows = episode_calls(Series(episode), minute_cache[pair])
        outcomes.extend(episode_outcomes(Series(episode), minute_cache[pair], contact_time))
        calls.extend(episode_call_rows)
    outcome_frame = DataFrame.from_records(outcomes)
    candidate_call_frame = DataFrame.from_records(calls)
    control_frame = control_calls(outcome_frame, sample)
    score_frame = score_methods(candidate_call_frame, control_frame, outcome_frame)
    decision_frame = decisions(score_frame)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "coverage": COVERAGE_CSV,
        "outcomes": run_dir / "g21_one_minute_episode_outcomes.csv",
        "calls": run_dir / "g21_one_minute_calls.csv",
        "control_calls": run_dir / "g21_one_minute_control_calls.csv",
        "scores": run_dir / "g21_one_minute_method_scores.csv",
        "decisions": run_dir / "g21_one_minute_decisions.csv",
    }
    for name, frame in (
        ("outcomes", outcome_frame),
        ("calls", candidate_call_frame),
        ("control_calls", control_frame),
        ("scores", score_frame),
        ("decisions", decision_frame),
    ):
        g0.atomic_write_csv(frame, paths[name])
    result = {
        "schema_version": 1,
        "generation": 21,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation21_disjoint_one_minute_replay",
        "branch_completed": "g21e_disjoint_one_minute_rejection_diagnostic",
        "actual_episodes": int(record["actual_episodes"]),
        "matched_no_level_episodes": int(record["matched_no_level_episodes"]),
        "predeclared_method_distance_horizon_rows": len(decision_frame),
        "retained_methods": int(decision_frame["retained"].sum()),
        "best_joint_success_rate": float(decision_frame["actual_joint_success_rate"].max()),
        "research_boundary": {
            "profit_used": False,
            "freqai_training": False,
            "signed_paths_opened_only_after_selection_freeze": True,
            "generation20_episode_neighbourhoods_excluded": True,
            "promotion_allowed": False,
        },
        "source_contracts": {
            "generation21_freeze": artifact(g21z.FREEZE_PATH),
            "selection_freeze": artifact(SELECTION_RECORD),
        },
        "artifacts": {name: artifact(path) for name, path in paths.items()},
        "result_path": str(result_path.resolve()),
    }
    g0.atomic_write_json(result, result_path)
    print(json.dumps(result, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--prepare-selection", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    record = freeze_selection(overwrite=args.overwrite if args.prepare_selection else False)
    if args.prepare_selection:
        print(json.dumps(record, indent=2))
        return 0
    if args.audit_only:
        sample = pd.read_csv(SELECTION_CSV)
        sample["event_time"] = pd.to_datetime(sample["event_time"], utc=True)
        _, passed = coverage_audit(sample)
        return 0 if passed else 2
    return execute(args.run_id, overwrite=args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())

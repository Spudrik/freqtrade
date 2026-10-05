"""Run Generation 20's bounded Donchian-onset one-minute direction replay."""

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
    market_reaction_zone_generation1_localization as g1,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation20_freeze as g20z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation20_state_attribution as g20s,
)


DEFAULT_RUN_ID = "g20_donchian_one_minute_20260827a"
ANALYSIS_PATH = Path(__file__).resolve()
RECORD_ROOT = g20z.OUTPUT_ROOT / "one_minute_replay"
SELECTION_RECORD = RECORD_ROOT / "g20_one_minute_selection_freeze.json"
SELECTION_CSV = RECORD_ROOT / "g20_one_minute_frozen_episodes.csv"
COVERAGE_CSV = RECORD_ROOT / "g20_one_minute_coverage.csv"
HORIZON_MINUTES = 60
PRE_WINDOW_HOURS = 4
POST_WINDOW_HOURS = 4
INDEPENDENCE_HOURS = 24
MINIMUM_EPISODES = g20z.ONE_MINUTE_EPISODE_MINIMUM
MAXIMUM_EPISODES = g20z.ONE_MINUTE_EPISODE_MAXIMUM
JOINT_FLOOR = 0.55
MINIMUM_CALL_COVERAGE = 0.50
FIXED_CALLS = (
    "approach_trend_15m",
    "approach_trend_60m",
    "signed_volume_pressure_15m",
    "signed_volume_pressure_60m",
    "momentum_vote_15m",
    "combined_pressure_and_momentum_vote",
)
CONTROLS = ("majority_path", "simple_approach_trend", "matched_no_level_episode")
B_STATE_COLUMNS = (
    "pre_abs_return_over_atr_h4",
    "pre_range_over_atr_h4",
    "relative_volume",
    "pre_close_location_h4",
    "pre_trend_over_atr_h4",
)


def artifact(path: Path) -> dict[str, Any]:
    return g20z.artifact(path)


def load_freeze() -> dict[str, Any]:
    frozen = json.loads(g20z.FREEZE_PATH.read_text(encoding="utf-8"))
    branch = next(
        item
        for item in frozen["branches"]
        if item["branch_id"] == "g20e_donchian_onset_one_minute_replay"
    )
    if frozen.get("status") != "frozen_before_generation20_outcomes":
        raise ValueError("Generation 20 batch is not frozen.")
    if tuple(branch["fixed_calls"]) != FIXED_CALLS:
        raise ValueError("Generation 20 one-minute call set drifted after freeze.")
    if tuple(branch["controls"]) != CONTROLS:
        raise ValueError("Generation 20 one-minute controls drifted after freeze.")
    return frozen


def load_state_support_with_reactions() -> DataFrame:
    manifest = g20s.freeze_support(overwrite=False)
    parts: list[DataFrame] = []
    for item in manifest["inventory"]:
        if item["cohort"] != "normal":
            continue
        path = Path(item["path"])
        if g0.sha256_file(path) != item["sha256"]:
            raise ValueError(f"Generation 20 state support changed: {path}")
        frame = pd.read_parquet(path)
        frame = frame.loc[
            frame["branch_id"].eq("g20b_donchian_onset_causal_isolation")
        ].copy()
        frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
        base, _ = g20s.base_and_state(str(item["pair"]), "normal")
        g20s.add_future_metrics(frame, base)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True, sort=False)


def independent_selection(candidates: DataFrame, maximum: int) -> DataFrame:
    source = candidates.copy()
    source["selection_hash"] = [
        g0.stable_hash_int(
            "|".join(
                (
                    "g20-one-minute-selection",
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
    used: dict[str, list[pd.Timestamp]] = {}
    strata = list(source.groupby(["period", "pair"], observed=True, sort=True))
    while len(selected) < maximum:
        progressed = False
        for _, stratum in strata:
            for index, row in stratum.iterrows():
                if index in selected:
                    continue
                event_time = pd.Timestamp(row["event_time"])
                pair = str(row["pair"])
                if any(
                    abs(event_time - prior) < pd.Timedelta(hours=INDEPENDENCE_HOURS)
                    for prior in used.get(pair, [])
                ):
                    continue
                selected.append(index)
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


def matched_no_level_rows(actual: DataFrame, pool: DataFrame) -> DataFrame:
    rows: list[DataFrame] = []
    for (pair, period), left in actual.groupby(
        ["pair", "period"], observed=True, sort=False
    ):
        right = pool.loc[pool["pair"].eq(pair) & pool["period"].eq(period)].copy()
        left = left.copy().reset_index(drop=True)
        right = right.reset_index(drop=True)
        left["pre_distance_atr"] = 0.0
        right["pre_distance_atr"] = 0.0
        pairs, _ = g1.nearest_state_pairs(
            left,
            right,
            state_columns=B_STATE_COLUMNS,
            pre_distance_atr_caliper=0.10,
            minimum_event_separation_hours=8,
        )
        if not pairs:
            continue
        selected_left = left.iloc[[item[0] for item in pairs]].reset_index(drop=True)
        selected_right = right.iloc[[item[1] for item in pairs]].reset_index(drop=True)
        selected_right["matched_to_episode_id"] = selected_left["episode_id"].to_numpy()
        selected_right["match_distance"] = [item[2] for item in pairs]
        rows.append(selected_right)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else DataFrame()


def freeze_selection(*, overwrite: bool = False) -> dict[str, Any]:
    load_freeze()
    if SELECTION_RECORD.is_file() and not overwrite:
        record = json.loads(SELECTION_RECORD.read_text(encoding="utf-8"))
        if record.get("status") != "frozen_before_signed_one_minute_paths":
            raise ValueError("Invalid Generation 20 one-minute selection freeze.")
        return record
    events = load_state_support_with_reactions()
    candidates = events.loc[
        events["control"].eq("actual")
        & events["approach_state"].isin({"from_below", "from_above"})
        & pd.to_numeric(events["pre_crossings_h4"], errors="coerce").eq(0.0)
        & pd.to_numeric(
            events["metric__crossing_count_h4"], errors="coerce"
        ).ge(1.0)
    ].copy()
    # The reaction filter is unsigned; neither the future path sign nor path order is read.
    candidates = candidates.sort_values(
        ["pair", "event_time", "pre_distance_atr", "level_name"], kind="stable"
    ).drop_duplicates(["pair", "event_time"], keep="first")
    selected = independent_selection(candidates, MAXIMUM_EPISODES)
    if len(selected) < MINIMUM_EPISODES:
        raise ValueError(
            f"Only {len(selected)} independent Donchian-onset episodes were available."
        )
    selected["episode_id"] = [
        f"g20e-{number:03d}" for number in range(1, len(selected) + 1)
    ]
    selected["episode_kind"] = "actual_level"
    pool = events.loc[
        events["control"].eq(
            "breakout_state_matched_no_current_boundary_contact"
        )
    ].copy()
    controls = matched_no_level_rows(selected, pool)
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
        "generation": 20,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_signed_one_minute_paths",
        "branch_id": "g20e_donchian_onset_one_minute_replay",
        "actual_episodes": len(selected),
        "matched_no_level_episodes": len(controls),
        "selection_used_unsigned_reaction": True,
        "selection_used_future_signed_direction": False,
        "selection_used_profit": False,
        "prewindow_hours": PRE_WINDOW_HOURS,
        "postwindow_hours": POST_WINDOW_HOURS,
        "independence_hours": INDEPENDENCE_HOURS,
        "replay_hours_before": 24,
        "replay_hours_after": 24,
        "fixed_calls": list(FIXED_CALLS),
        "controls": list(CONTROLS),
        "artifact": artifact(SELECTION_CSV),
        "source_contracts": {
            "generation20_freeze": artifact(g20z.FREEZE_PATH),
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
        if pair not in minute_cache:
            path = g0.ohlcv_path(pair, "1m")
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
                "source_path": str(g0.ohlcv_path(pair, "1m").resolve()),
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
    g0.atomic_write_csv(result, COVERAGE_CSV)
    return result, passed


def finite_sign(value: float, tolerance: float = 1e-12) -> int:
    if not np.isfinite(value) or abs(value) <= tolerance:
        return 0
    return 1 if value > 0.0 else -1


def completed_window(minute: DataFrame, end: pd.Timestamp, length: int) -> DataFrame:
    start = end - pd.Timedelta(minutes=length)
    frame = minute.loc[(minute["date"] >= start) & (minute["date"] < end)].copy()
    return frame if len(frame) == length else DataFrame()


def trend_call(frame: DataFrame) -> int:
    if len(frame) < 2:
        return 0
    first = float(frame["close"].iloc[0])
    last = float(frame["close"].iloc[-1])
    return finite_sign(last - first)


def pressure_call(frame: DataFrame) -> int:
    if frame.empty:
        return 0
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    candle_range = (high - low).replace(0.0, np.nan)
    pressure = (2.0 * close - high - low).div(candle_range)
    denominator = float(volume.sum())
    value = (
        float((pressure.fillna(0.0) * volume).sum() / denominator)
        if denominator > 0
        else np.nan
    )
    return finite_sign(value)


def momentum_vote(frame: DataFrame) -> int:
    if len(frame) < 15:
        return 0
    close = pd.to_numeric(frame["close"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    votes = [
        finite_sign(float(close.iloc[-1] - close.iloc[-15])),
        finite_sign(
            float(
                close.ewm(span=5, adjust=False).mean().iloc[-1]
                - close.ewm(span=15, adjust=False).mean().iloc[-1]
            )
        ),
        finite_sign(
            float(
                close.iloc[-1]
                - (close.iloc[-15:] * volume.iloc[-15:]).sum()
                / volume.iloc[-15:].sum()
            )
        )
        if volume.iloc[-15:].sum() > 0.0
        else 0,
    ]
    score = sum(votes)
    return 1 if score >= 2 else -1 if score <= -2 else 0


def first_contact_time(episode: Series, minute: DataFrame) -> pd.Timestamp:
    event_time = pd.Timestamp(episode["event_time"])
    hour = minute.loc[
        (minute["date"] >= event_time)
        & (minute["date"] < event_time + pd.Timedelta(hours=1))
    ]
    level = float(episode["level_price"])
    width = float(episode["zone_half_width"])
    contact = hour.loc[
        (pd.to_numeric(hour["high"], errors="coerce") >= level - width)
        & (pd.to_numeric(hour["low"], errors="coerce") <= level + width)
    ]
    if contact.empty:
        raise ValueError(f"No minute contact inside frozen event {episode['episode_id']}.")
    return pd.Timestamp(contact["date"].iloc[0])


def episode_calls_and_outcome(
    episode: Series, minute: DataFrame
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    contact_time = (
        pd.Timestamp(episode["event_time"])
        if str(episode["episode_kind"]) == "matched_no_level"
        else first_contact_time(episode, minute)
    )
    pre15 = completed_window(minute, contact_time, 15)
    pre60 = completed_window(minute, contact_time, 60)
    trend15 = trend_call(pre15)
    trend60 = trend_call(pre60)
    pressure15 = pressure_call(pre15)
    pressure60 = pressure_call(pre60)
    momentum15 = momentum_vote(pre60)
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
        raise ValueError("Generation 20 one-minute call registry drifted.")
    decision_time = contact_time + pd.Timedelta(minutes=1)
    future = minute.loc[
        (minute["date"] >= decision_time)
        & (minute["date"] < decision_time + pd.Timedelta(minutes=HORIZON_MINUTES))
    ].copy()
    level = float(episode["level_price"])
    width = float(episode["zone_half_width"])
    close = pd.to_numeric(future["close"], errors="coerce").to_numpy(dtype=float)
    above = np.flatnonzero(close >= level + width)
    below = np.flatnonzero(close <= level - width)
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
    approach_state = str(episode["approach_state"])
    through_direction = 1 if approach_state == "from_below" else -1
    path_kind = (
        "through"
        if target != 0 and target == through_direction
        else "away"
        if target != 0
        else "unreached_or_tied"
    )
    outcome = {
        "episode_id": str(episode["episode_id"]),
        "episode_kind": str(episode["episode_kind"]),
        "matched_to_episode_id": episode.get("matched_to_episode_id"),
        "pair": str(episode["pair"]),
        "period": str(episode["period"]),
        "contact_time": contact_time,
        "decision_time": decision_time,
        "horizon_minutes": HORIZON_MINUTES,
        "reaction": target != 0,
        "first_direction_numeric": target,
        "path_relative_to_approach": path_kind,
        "first_above_minute": first_above + 1 if first_above is not None else np.nan,
        "first_below_minute": first_below + 1 if first_below is not None else np.nan,
    }
    call_rows = [
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
    return outcome, call_rows


def control_calls(outcomes: DataFrame, sample: DataFrame) -> DataFrame:
    actual = outcomes.loc[outcomes["episode_kind"].eq("actual_level")].copy()
    sample_by_id = sample.set_index("episode_id")
    rows: list[dict[str, Any]] = []
    for event in actual.itertuples(index=False):
        others = actual.loc[
            actual["episode_id"].ne(event.episode_id)
            & actual["first_direction_numeric"].ne(0),
            "first_direction_numeric",
        ]
        majority = finite_sign(float(others.sum())) if len(others) else 0
        approach = str(sample_by_id.loc[event.episode_id, "approach_state"])
        simple = 1 if approach == "from_below" else -1 if approach == "from_above" else 0
        for method, call in (("majority_path", majority), ("simple_approach_trend", simple)):
            rows.append(
                {
                    "episode_id": event.episode_id,
                    "episode_kind": "actual_level",
                    "method": method,
                    "method_kind": "control",
                    "call_numeric": call,
                    "issued": call != 0,
                }
            )
    return DataFrame.from_records(rows)


def score_methods(calls: DataFrame, outcomes: DataFrame) -> DataFrame:
    scored = calls.merge(
        outcomes[
            ["episode_id", "episode_kind", "reaction", "first_direction_numeric"]
        ],
        on=["episode_id", "episode_kind"],
        how="inner",
        validate="many_to_one",
    )
    scored["correct"] = (
        scored["issued"]
        & scored["reaction"]
        & scored["call_numeric"].eq(scored["first_direction_numeric"])
    )
    rows: list[dict[str, Any]] = []
    for (kind, method, episode_kind), cell in scored.groupby(
        ["method_kind", "method", "episode_kind"], observed=True, sort=False
    ):
        issued_reacted = cell["issued"] & cell["reaction"]
        rows.append(
            {
                "method_kind": kind,
                "method": method,
                "episode_kind": episode_kind,
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
    actual = scores.loc[
        scores["method_kind"].eq("causal_candidate")
        & scores["episode_kind"].eq("actual_level")
    ].set_index("method")
    no_level = scores.loc[
        scores["method_kind"].eq("causal_candidate")
        & scores["episode_kind"].eq("matched_no_level")
    ].set_index("method")
    controls = scores.loc[
        scores["method_kind"].eq("control")
        & scores["episode_kind"].eq("actual_level")
    ].set_index("method")
    majority = float(controls.loc["majority_path", "joint_success_rate"])
    simple = float(controls.loc["simple_approach_trend", "joint_success_rate"])
    rows: list[dict[str, Any]] = []
    for method in FIXED_CALLS:
        candidate = actual.loc[method]
        matched = float(no_level.loc[method, "joint_success_rate"])
        rate = float(candidate["joint_success_rate"])
        coverage = float(candidate["issued_call_coverage"])
        retained = bool(
            int(candidate["episodes"]) >= MINIMUM_EPISODES
            and rate >= JOINT_FLOOR
            and coverage >= MINIMUM_CALL_COVERAGE
            and rate > majority
            and rate > simple
            and rate > matched
        )
        rows.append(
            {
                "method": method,
                "status": (
                    "bounded_point_lead_pending_independent_confirmation"
                    if retained
                    else "not_retained"
                ),
                "actual_joint_success_rate": rate,
                "actual_call_coverage": coverage,
                "majority_path_joint_success_rate": majority,
                "simple_approach_joint_success_rate": simple,
                "matched_no_level_joint_success_rate": matched,
                "at_or_above_55pct": rate >= JOINT_FLOOR,
                "adequate_coverage": coverage >= MINIMUM_CALL_COVERAGE,
                "beats_all_controls": rate > max(majority, simple, matched),
                "retained": retained,
            }
        )
    return DataFrame.from_records(rows)


def execute(run_id: str, *, overwrite: bool = False) -> int:
    record = freeze_selection(overwrite=False)
    if (
        g0.sha256_file(ANALYSIS_PATH)
        != record["source_contracts"]["analysis_script"]["sha256"]
    ):
        raise ValueError("Generation 20 one-minute analysis changed after selection freeze.")
    run_dir = RECORD_ROOT / run_id
    result_path = run_dir / "g20_one_minute_replay_result.json"
    if result_path.is_file() and not overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    if g0.sha256_file(SELECTION_CSV) != record["artifact"]["sha256"]:
        raise ValueError("Frozen Generation 20 one-minute sample changed.")
    sample = pd.read_csv(SELECTION_CSV)
    sample["event_time"] = pd.to_datetime(sample["event_time"], utc=True)
    coverage, coverage_passed = coverage_audit(sample)
    if not coverage_passed:
        result = {
            "schema_version": 1,
            "generation": 20,
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
        outcome, episode_calls = episode_calls_and_outcome(
            Series(episode), minute_cache[pair]
        )
        outcomes.append(outcome)
        calls.extend(episode_calls)
    outcome_frame = DataFrame.from_records(outcomes)
    call_frame = pd.concat(
        [DataFrame.from_records(calls), control_calls(outcome_frame, sample)],
        ignore_index=True,
        sort=False,
    )
    score_frame = score_methods(call_frame, outcome_frame)
    decision_frame = decisions(score_frame)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "coverage": COVERAGE_CSV,
        "outcomes": run_dir / "g20_one_minute_episode_outcomes.csv",
        "calls": run_dir / "g20_one_minute_calls.csv",
        "scores": run_dir / "g20_one_minute_method_scores.csv",
        "decisions": run_dir / "g20_one_minute_decisions.csv",
    }
    for name, frame in (
        ("outcomes", outcome_frame),
        ("calls", call_frame),
        ("scores", score_frame),
        ("decisions", decision_frame),
    ):
        g0.atomic_write_csv(frame, paths[name])
    result = {
        "schema_version": 1,
        "generation": 20,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation20_one_minute_replay",
        "branch_completed": "g20e_donchian_onset_one_minute_replay",
        "actual_episodes": int(record["actual_episodes"]),
        "matched_no_level_episodes": int(record["matched_no_level_episodes"]),
        "retained_methods": int(decision_frame["retained"].sum()),
        "best_joint_success_rate": float(
            decision_frame["actual_joint_success_rate"].max()
        ),
        "research_boundary": {
            "profit_used": False,
            "freqai_training": False,
            "signed_paths_opened_only_after_selection_freeze": True,
            "promotion_allowed": False,
        },
        "source_contracts": {
            "generation20_freeze": artifact(g20z.FREEZE_PATH),
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

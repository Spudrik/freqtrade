"""Run Generation 13's matched no-level and near-miss reaction controls.

The match is intentionally outcome-blind.  Actual contacts and controls are paired
only on their already-causal activity/volatility state, approach side, source
timeframe, and pre-contact geometry.  Reaction outcomes are scored after matching.
"""

from __future__ import annotations

# Bind native numerical pools before importing pandas and the research modules.
# ruff: noqa: E402
import argparse
import json
import os
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
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
    market_reaction_zone_generation11_direct_screen as g11d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation11_freeze as g11z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation12_freeze as g12z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation13_freeze as g13z,
)


DEFAULT_RUN_ID = "g13_direct_controls_20260822a"
RECORD_ROOT = (
    g13z.FREEZE_PATH.parent / "g13_broad_siblings" / "direct_level_controls"
)
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation13_branches"
    / "g13_broad_siblings"
    / "direct_level_controls"
)
MAX_WORKERS = 4
HORIZONS = tuple(g11z.HORIZONS)
CONTROLS = tuple(g13z.DIRECT_CONTROLS)
MATCH_FEATURES = tuple(g13z.DIRECT_MATCH_FEATURES)
AUDIT_FEATURES = tuple(g13z.DIRECT_BALANCE_AUDIT_FEATURES)
AUDIT_SOURCE_MAP = {
    column: ("state__rsi14" if column == "state__rsi14_centered" else column)
    for column in AUDIT_FEATURES
}
AUDIT_SOURCE_COLUMNS = tuple(dict.fromkeys(AUDIT_SOURCE_MAP.values()))
MIN_PAIR_ROWS = 20
MIN_COINS = 5
BALANCE_STRONG_MAX = 0.25
BALANCE_STRONG_MEDIAN = 0.10
BALANCE_USABLE_MAX = 0.50
BALANCE_USABLE_MEDIAN = 0.20

META_COLUMNS = (
    "cohort",
    "pair",
    "source_timeframe",
    "level_family",
    "level_name",
    "control",
    "event_time",
    "period",
    "approach_state",
    "pre_distance_atr",
    "contact_close_distance_atr",
    "zone_half_width_atr",
)
OUTCOME_COLUMNS = tuple(
    column
    for horizon in HORIZONS
    for column in (f"abs_excursion_atr_h{horizon}", f"volume_ratio_h{horizon}")
)
READ_COLUMNS = tuple(
    dict.fromkeys((*META_COLUMNS, *MATCH_FEATURES, *AUDIT_SOURCE_COLUMNS, *OUTCOME_COLUMNS))
)


@dataclass(frozen=True)
class PairTask:
    cohort: str
    pair: str
    source_path: str
    source_sha256: str
    artifact_dir: str
    overwrite: bool


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def validate_contracts() -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = json.loads(g13z.FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation13_outcomes":
        raise ValueError("Generation 13 was not frozen before its outcomes.")
    direct = frozen["siblings"]["matched_no_level_and_near_miss_controls"]
    if tuple(direct["match_features"]) != MATCH_FEATURES:
        raise ValueError("Generation 13 direct-match feature contract drifted.")
    if any(column in MATCH_FEATURES for column in OUTCOME_COLUMNS):
        raise ValueError("Reaction outcomes entered the direct matching feature list.")
    manifest = json.loads(g13z.G6_EVENT_MANIFEST.read_text(encoding="utf-8"))
    source = frozen["source_contracts"]["generation6_event_manifest"]
    if g0.sha256_file(g13z.G6_EVENT_MANIFEST) != source["sha256"]:
        raise ValueError("The frozen Generation 6 event manifest changed.")
    if manifest.get("status") != "completed_shared_causal_event_cache":
        raise ValueError("The causal event cache is incomplete.")
    return frozen, manifest


def balance_quality(diagnostics: dict[str, Any]) -> str:
    maximum = float(diagnostics["max_absolute_smd"])
    median = float(diagnostics["median_absolute_smd"])
    if maximum <= BALANCE_STRONG_MAX and median <= BALANCE_STRONG_MEDIAN:
        return "strong"
    if maximum <= BALANCE_USABLE_MAX and median <= BALANCE_USABLE_MEDIAN:
        return "usable"
    return "weak_or_sparse"


def reaction_label(frame: DataFrame, horizon: int) -> Series:
    threshold = np.maximum(
        0.5,
        pd.to_numeric(frame["zone_half_width_atr"], errors="coerce").to_numpy(dtype=float),
    )
    excursion = pd.to_numeric(
        frame[f"abs_excursion_atr_h{horizon}"], errors="coerce"
    ).to_numpy(dtype=float)
    volume = pd.to_numeric(
        frame[f"volume_ratio_h{horizon}"], errors="coerce"
    ).to_numpy(dtype=float)
    valid = np.isfinite(threshold) & np.isfinite(excursion) & np.isfinite(volume)
    values = np.full(len(frame), np.nan, dtype=float)
    values[valid] = (
        (excursion[valid] >= threshold[valid]) & (volume[valid] >= 1.25)
    ).astype(float)
    return Series(values, index=frame.index, dtype=float)


def derive_audit_features(frame: DataFrame) -> DataFrame:
    output = frame.copy()
    for feature, source in AUDIT_SOURCE_MAP.items():
        values = pd.to_numeric(output[source], errors="coerce")
        output[feature] = values - 50.0 if feature == "state__rsi14_centered" else values
    return output


def add_analysis_windows(frame: DataFrame, cohort: str) -> DataFrame:
    frames: list[DataFrame] = []
    standard_periods = set(g11z.VALIDATION_PERIODS[cohort])
    standard = frame.loc[frame["period"].isin(standard_periods)].copy()
    if not standard.empty:
        standard["window"] = g13z.STANDARD
        standard["analysis_period"] = standard["period"].astype(str)
        frames.append(standard)
    if cohort == "normal":
        times = pd.to_datetime(frame["event_time"], utc=True, errors="coerce")
        for item in g12z.RECENT_PERIODS:
            start = pd.Timestamp(item["start"])
            stop = pd.Timestamp(item["stop"])
            recent = frame.loc[times.ge(start) & times.lt(stop)].copy()
            if recent.empty:
                continue
            recent["window"] = g13z.RECENT
            recent["analysis_period"] = item["period"]
            frames.append(recent)
    if not frames:
        return DataFrame(columns=[*frame.columns, "window", "analysis_period"])
    return pd.concat(frames, ignore_index=True, sort=False)


def causal_deduplicate(frame: DataFrame) -> DataFrame:
    ordered = frame.assign(
        _distance=pd.to_numeric(frame["contact_close_distance_atr"], errors="coerce").abs()
    ).sort_values(
        ["window", "analysis_period", "source_timeframe", "control", "event_time", "_distance"],
        kind="stable",
        na_position="last",
    )
    return (
        ordered.drop_duplicates(
            ["window", "analysis_period", "source_timeframe", "control", "event_time"],
            keep="first",
        )
        .drop(columns="_distance")
        .reset_index(drop=True)
    )


def matched_rows(actual: DataFrame, control: DataFrame, control_name: str) -> DataFrame:
    rows: list[DataFrame] = []
    keys = ("window", "analysis_period", "source_timeframe", "approach_state")
    actual_groups = actual.groupby(list(keys), observed=True, sort=False)
    control_groups = control.groupby(list(keys), observed=True, sort=False)
    for key in sorted(set(actual_groups.groups).intersection(control_groups.groups)):
        left = actual_groups.get_group(key).reset_index(drop=True)
        right = control_groups.get_group(key).reset_index(drop=True)
        pairs, audit = g1.nearest_state_pairs(
            left,
            right,
            state_columns=MATCH_FEATURES,
            minimum_event_separation_hours=max(HORIZONS),
        )
        if not pairs:
            continue
        left_positions = [item[0] for item in pairs]
        right_positions = [item[1] for item in pairs]
        selected_left = left.iloc[left_positions].reset_index(drop=True)
        selected_right = right.iloc[right_positions].reset_index(drop=True)
        match_balance = g1.balance_diagnostics(selected_left, selected_right, MATCH_FEATURES)
        audit_balance = g1.balance_diagnostics(selected_left, selected_right, AUDIT_FEATURES)
        result = DataFrame(
            {
                "cohort": selected_left["cohort"].astype(str),
                "pair": selected_left["pair"].astype(str),
                "window": key[0],
                "analysis_period": key[1],
                "source_timeframe": key[2],
                "approach_state": key[3],
                "control": control_name,
                "event_time": pd.to_datetime(selected_left["event_time"], utc=True),
                "actual_event_time": pd.to_datetime(selected_left["event_time"], utc=True),
                "control_event_time": pd.to_datetime(selected_right["event_time"], utc=True),
                "level_name": selected_left["level_name"].astype(str),
                "match_distance": [item[2] for item in pairs],
                "pre_distance_atr_abs_difference": (
                    pd.to_numeric(selected_left["pre_distance_atr"], errors="coerce")
                    - pd.to_numeric(selected_right["pre_distance_atr"], errors="coerce")
                ).abs(),
                "eligible_actual": audit["eligible_actual"],
                "eligible_control": audit["eligible_control"],
                "state_matchable_actual": audit["state_matchable_actual"],
                "match_balance_quality": balance_quality(match_balance),
                "match_max_absolute_smd": match_balance["max_absolute_smd"],
                "match_median_absolute_smd": match_balance["median_absolute_smd"],
                "audit_balance_quality": balance_quality(audit_balance),
                "audit_max_absolute_smd": audit_balance["max_absolute_smd"],
                "audit_median_absolute_smd": audit_balance["median_absolute_smd"],
            }
        )
        for column in (*MATCH_FEATURES, *AUDIT_FEATURES):
            result[f"actual__{column}"] = pd.to_numeric(
                selected_left[column], errors="coerce"
            ).to_numpy(dtype=float)
            result[f"control__{column}"] = pd.to_numeric(
                selected_right[column], errors="coerce"
            ).to_numpy(dtype=float)
        for horizon in HORIZONS:
            result[f"actual_reaction_h{horizon}"] = reaction_label(
                selected_left, horizon
            ).to_numpy(dtype=float)
            result[f"control_reaction_h{horizon}"] = reaction_label(
                selected_right, horizon
            ).to_numpy(dtype=float)
        rows.append(result)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else DataFrame()


def build_pair(task: PairTask) -> dict[str, Any]:
    source_path = Path(task.source_path)
    if not source_path.is_file() or g0.sha256_file(source_path) != task.source_sha256:
        raise ValueError(f"Frozen event source changed: {source_path}")
    output_path = Path(task.artifact_dir) / f"{task.cohort}__{g0.pair_file_stem(task.pair)}.parquet"
    if output_path.is_file() and not task.overwrite:
        existing = pd.read_parquet(output_path, columns=["pair", "control"])
        return {
            "status": "existing",
            "cohort": task.cohort,
            "pair": task.pair,
            "matched_rows": len(existing),
            "controls": sorted(existing["control"].astype(str).unique()),
            "path": str(output_path),
            "sha256": g0.sha256_file(output_path),
        }
    frame = pd.read_parquet(
        source_path,
        columns=list(READ_COLUMNS),
        filters=[("control", "in", ["actual", *CONTROLS])],
    )
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="coerce")
    frame = derive_audit_features(frame.dropna(subset=["event_time"]))
    actual_clock = set(
        frame.loc[frame["control"].eq("actual"), "event_time"].astype("int64").tolist()
    )
    control_mask = frame["control"].isin(CONTROLS)
    level_free = ~frame["event_time"].astype("int64").isin(actual_clock)
    removed_level_contacts = int((control_mask & ~level_free).sum())
    frame = frame.loc[~control_mask | level_free].copy()
    frame = add_analysis_windows(frame, task.cohort)
    frame = causal_deduplicate(frame)
    actual = frame.loc[frame["control"].eq("actual")].copy()
    outputs: list[DataFrame] = []
    for control_name in CONTROLS:
        control = frame.loc[frame["control"].eq(control_name)].copy()
        matched = matched_rows(actual, control, control_name)
        if not matched.empty:
            outputs.append(matched)
    combined = pd.concat(outputs, ignore_index=True, sort=False) if outputs else DataFrame()
    if not combined.empty:
        overlap = combined["control_event_time"].astype("int64").isin(actual_clock)
        if overlap.any():
            raise AssertionError("A matched control timestamp coincides with a real level contact.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(combined, output_path)
    return {
        "status": "built",
        "cohort": task.cohort,
        "pair": task.pair,
        "matched_rows": len(combined),
        "removed_control_rows_at_real_contacts": removed_level_contacts,
        "controls": sorted(combined["control"].astype(str).unique()) if len(combined) else [],
        "path": str(output_path),
        "sha256": g0.sha256_file(output_path),
    }


def safe_build_pair(task: PairTask) -> dict[str, Any]:
    try:
        return build_pair(task)
    except Exception as exc:
        return {
            "status": "failed",
            "cohort": task.cohort,
            "pair": task.pair,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_tasks(tasks: Sequence[PairTask], workers: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(safe_build_pair, task): task for task in tasks}
        for index, future in enumerate(as_completed(pending), start=1):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "phase": "g13_direct_pair_matching",
                        "processed": index,
                        "total": len(tasks),
                        "pair": result["pair"],
                        "cohort": result["cohort"],
                        "status": result["status"],
                    }
                ),
                flush=True,
            )
    return sorted(results, key=lambda item: (item["cohort"], item["pair"]))


def paired_balance(frame: DataFrame, features: Sequence[str]) -> dict[str, Any]:
    actual = frame[[f"actual__{column}" for column in features]].rename(
        columns={f"actual__{column}": column for column in features}
    )
    control = frame[[f"control__{column}" for column in features]].rename(
        columns={f"control__{column}": column for column in features}
    )
    return g1.balance_diagnostics(actual, control, features)


def purge_overlapping_hourly_pairs(
    pairs: DataFrame,
    *,
    separation_hours: int,
    group_columns: Sequence[str],
) -> DataFrame:
    """Apply the established greedy overlap rule in bounded time for hourly rows."""
    if pairs.empty:
        return pairs.copy()
    hour_ns = int(pd.Timedelta(hours=1).value)
    selected_frames: list[DataFrame] = []
    for _, group in pairs.groupby(list(group_columns), observed=True, sort=False):
        candidates = group.sort_values(
            [
                "match_distance",
                "pre_distance_atr_abs_difference",
                "actual_event_time",
                "control_event_time",
                "level_name",
                "approach_state",
            ],
            kind="stable",
        ).drop_duplicates(["actual_event_time", "control_event_time"], keep="first")
        blocked_hours: set[int] = set()
        retained: list[Any] = []
        for index, row in candidates.iterrows():
            actual_ns = int(pd.Timestamp(row["actual_event_time"]).value)
            control_ns = int(pd.Timestamp(row["control_event_time"]).value)
            if actual_ns % hour_ns or control_ns % hour_ns:
                raise ValueError("Generation 13 direct controls require hourly event clocks.")
            actual_hour = actual_ns // hour_ns
            control_hour = control_ns // hour_ns
            if actual_hour in blocked_hours or control_hour in blocked_hours:
                continue
            retained.append(index)
            for offset in range(-separation_hours, separation_hours + 1):
                blocked_hours.add(actual_hour + offset)
                blocked_hours.add(control_hour + offset)
        if retained:
            selected = candidates.loc[retained].copy()
            selected["independent_selection_order"] = np.arange(
                1, len(selected) + 1, dtype=np.int32
            )
            selected["independence_hours"] = separation_hours
            selected_frames.append(selected)
    if not selected_frames:
        return DataFrame(
            columns=[*pairs.columns, "independent_selection_order", "independence_hours"]
        )
    return pd.concat(selected_frames, ignore_index=True, sort=False)


def independent_rows(frame: DataFrame, *, horizon: int, scope: str) -> DataFrame:
    selected = frame if scope == "all_source_timeframes" else frame.loc[
        frame["source_timeframe"].eq(scope)
    ]
    groups = ["pair", "window", "analysis_period", "control"]
    if scope != "all_source_timeframes":
        groups.append("source_timeframe")
    return purge_overlapping_hourly_pairs(
        selected,
        separation_hours=horizon,
        group_columns=groups,
    )


def pair_metrics(frame: DataFrame) -> tuple[DataFrame, DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    scored_frames: list[DataFrame] = []
    scopes = ("all_source_timeframes", *g11z.SOURCE_TIMEFRAMES)
    for horizon in HORIZONS:
        for scope in scopes:
            independent = independent_rows(frame, horizon=horizon, scope=scope)
            if independent.empty:
                continue
            actual_column = f"actual_reaction_h{horizon}"
            control_column = f"control_reaction_h{horizon}"
            independent = independent.dropna(subset=[actual_column, control_column]).copy()
            independent["actual_reaction"] = independent[actual_column].astype(float)
            independent["control_reaction"] = independent[control_column].astype(float)
            independent["reaction_delta"] = (
                independent["actual_reaction"] - independent["control_reaction"]
            )
            independent["horizon_hours"] = horizon
            independent["source_scope"] = scope
            scored_frames.append(independent)
            keys = ("cohort", "pair", "window", "analysis_period", "control")
            for key, cell in independent.groupby(list(keys), observed=True, sort=False):
                match_balance = paired_balance(cell, MATCH_FEATURES)
                audit_balance = paired_balance(cell, AUDIT_FEATURES)
                match_quality = balance_quality(match_balance)
                audit_quality = balance_quality(audit_balance)
                metric_rows.append(
                    {
                        "cohort": key[0],
                        "pair": key[1],
                        "window": key[2],
                        "analysis_period": key[3],
                        "control": key[4],
                        "source_scope": scope,
                        "horizon_hours": horizon,
                        "matched_independent_rows": len(cell),
                        "actual_reaction_rate": float(cell["actual_reaction"].mean()),
                        "control_reaction_rate": float(cell["control_reaction"].mean()),
                        "reaction_rate_difference": float(cell["reaction_delta"].mean()),
                        "match_balance_quality": match_quality,
                        "match_max_absolute_smd": match_balance["max_absolute_smd"],
                        "match_median_absolute_smd": match_balance["median_absolute_smd"],
                        "audit_balance_quality": audit_quality,
                        "audit_max_absolute_smd": audit_balance["max_absolute_smd"],
                        "audit_median_absolute_smd": audit_balance["median_absolute_smd"],
                        "adequately_balanced": (
                            match_quality in {"strong", "usable"}
                            and audit_quality in {"strong", "usable"}
                        ),
                    }
                )
    metrics = DataFrame.from_records(metric_rows)
    scored = (
        pd.concat(scored_frames, ignore_index=True, sort=False)
        if scored_frames
        else DataFrame()
    )
    return metrics, scored


def cohort_scores(pair_summary: DataFrame, scored_rows: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    keys = (
        "cohort",
        "window",
        "analysis_period",
        "control",
        "source_scope",
        "horizon_hours",
    )
    for key, cell in pair_summary.groupby(list(keys), observed=True, sort=False):
        eligible = cell.loc[
            cell["adequately_balanced"]
            & cell["matched_independent_rows"].ge(MIN_PAIR_ROWS)
        ].copy()
        pair_names = eligible["pair"].astype(str).tolist()
        matching_rows = scored_rows.loc[
            scored_rows["cohort"].eq(key[0])
            & scored_rows["window"].eq(key[1])
            & scored_rows["analysis_period"].eq(key[2])
            & scored_rows["control"].eq(key[3])
            & scored_rows["source_scope"].eq(key[4])
            & scored_rows["horizon_hours"].eq(key[5])
            & scored_rows["pair"].isin(pair_names)
        ].copy()
        bootstrap = g11d.block_bootstrap(
            matching_rows,
            value_columns=("reaction_delta",),
            seed_key="g13-direct|" + "|".join(map(str, key)),
        )["reaction_delta"]
        rows.append(
            {
                "cohort": key[0],
                "window": key[1],
                "analysis_period": key[2],
                "control": key[3],
                "source_scope": key[4],
                "horizon_hours": int(key[5]),
                "eligible_coins": len(eligible),
                "positive_coins": int(eligible["reaction_rate_difference"].gt(0.0).sum()),
                "matched_independent_rows": int(eligible["matched_independent_rows"].sum()),
                "equal_coin_actual_reaction_rate": float(
                    eligible["actual_reaction_rate"].mean()
                )
                if len(eligible)
                else np.nan,
                "equal_coin_control_reaction_rate": float(
                    eligible["control_reaction_rate"].mean()
                )
                if len(eligible)
                else np.nan,
                "equal_coin_reaction_rate_difference": bootstrap[0],
                "bootstrap_lower_95": bootstrap[1],
                "bootstrap_upper_95": bootstrap[2],
                "point_cell_pass": bool(len(eligible) >= MIN_COINS and bootstrap[0] > 0.0),
                "strict_cell_pass": bool(
                    len(eligible) >= MIN_COINS and bootstrap[1] > 0.0
                ),
            }
        )
    return DataFrame.from_records(rows)


def route_decisions(scores: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ("cohort", "window", "source_scope", "horizon_hours")
    for key, cell in scores.groupby(list(keys), observed=True, sort=False):
        expected_periods = (
            list(g11z.VALIDATION_PERIODS[key[0]])
            if key[1] == g13z.STANDARD
            else [item["period"] for item in g12z.RECENT_PERIODS]
        )
        expected = {(period, control) for period in expected_periods for control in CONTROLS}
        observed = set(zip(cell["analysis_period"], cell["control"], strict=False))
        complete = observed == expected
        rows.append(
            {
                "cohort": key[0],
                "window": key[1],
                "source_scope": key[2],
                "horizon_hours": int(key[3]),
                "expected_cells": len(expected),
                "observed_cells": len(observed),
                "complete_control_period_ladder": complete,
                "point_pass": bool(complete and cell["point_cell_pass"].all()),
                "strict_pass": bool(complete and cell["strict_cell_pass"].all()),
                "minimum_equal_coin_difference": float(
                    cell["equal_coin_reaction_rate_difference"].min()
                ),
                "minimum_bootstrap_lower_95": float(cell["bootstrap_lower_95"].min()),
                "minimum_eligible_coins": int(cell["eligible_coins"].min()),
            }
        )
    return DataFrame.from_records(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen Generation 13 direct controls.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1 or args.workers > MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    frozen, manifest = validate_contracts()
    record_dir = RECORD_ROOT / args.run_id
    artifact_dir = ARTIFACT_ROOT / args.run_id / "pair_matches"
    result_path = record_dir / "g13_direct_control_result.json"
    if result_path.is_file() and not args.overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    tasks = [
        PairTask(
            cohort=item["cohort"],
            pair=item["pair"],
            source_path=item["event_path"],
            source_sha256=item["event_sha256"],
            artifact_dir=str(artifact_dir),
            overwrite=args.overwrite,
        )
        for item in manifest["tasks"]
    ]
    inventory = run_tasks(tasks, workers=args.workers)
    failures = [item for item in inventory if item["status"] == "failed"]
    if failures:
        raise RuntimeError(f"Generation 13 direct-control matching failed: {failures}")
    matched = pd.concat(
        [pd.read_parquet(item["path"]) for item in inventory],
        ignore_index=True,
        sort=False,
    )
    pair_summary, scored_rows = pair_metrics(matched)
    scores = cohort_scores(pair_summary, scored_rows)
    decisions = route_decisions(scores)
    record_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = record_dir / "pair_match_inventory.csv"
    pair_summary_path = record_dir / "pair_reaction_differences.csv"
    scores_path = record_dir / "equal_coin_control_scores.csv"
    decisions_path = record_dir / "direct_control_decisions.csv"
    g0.atomic_write_csv(DataFrame.from_records(inventory), inventory_path)
    g0.atomic_write_csv(pair_summary, pair_summary_path)
    g0.atomic_write_csv(scores, scores_path)
    g0.atomic_write_csv(decisions, decisions_path)
    result = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "completed_generation13_direct_controls",
        "created_at_utc": g0.utc_now(),
        "plain_question": frozen["siblings"]["matched_no_level_and_near_miss_controls"][
            "plain_question"
        ],
        "matching": {
            "features": list(MATCH_FEATURES),
            "approach_state_exact": True,
            "source_timeframe_exact": True,
            "pre_distance_atr_caliper": g1.PRE_DISTANCE_ATR_CALIPER,
            "state_rms_caliper": g1.MATCH_RMS_CALIPER,
            "maximum_control_reuse": g1.MAX_CONTROL_REUSE,
            "control_timestamps_at_any_real_contact_removed": True,
            "future_outcomes_used_for_matching": False,
        },
        "balance": {
            "audit_features": list(AUDIT_FEATURES),
            "audit_source_map": AUDIT_SOURCE_MAP,
            "strong_max_absolute_smd": BALANCE_STRONG_MAX,
            "strong_median_absolute_smd": BALANCE_STRONG_MEDIAN,
            "usable_max_absolute_smd": BALANCE_USABLE_MAX,
            "usable_median_absolute_smd": BALANCE_USABLE_MEDIAN,
            "thresholds_reused_from_generation1_localization": True,
        },
        "reaction_definition": frozen["siblings"][g13z.LONG_STAGE]["reaction_definition"],
        "inventory": artifact(inventory_path),
        "pair_reaction_differences": artifact(pair_summary_path),
        "equal_coin_control_scores": artifact(scores_path),
        "direct_control_decisions": artifact(decisions_path),
        "summary": {
            "pair_tasks": len(inventory),
            "matched_rows_before_horizon_purges": len(matched),
            "removed_control_rows_at_real_contacts": int(
                sum(item.get("removed_control_rows_at_real_contacts", 0) for item in inventory)
            ),
            "decision_rows": len(decisions),
            "point_pass_rows": int(decisions["point_pass"].sum()),
            "strict_pass_rows": int(decisions["strict_pass"].sum()),
            "profit_used": False,
            "future_signed_direction_used": False,
            "joint_55_percent_direction_target_reached": False,
        },
        "source_contracts": {
            "generation13_freeze": artifact(g13z.FREEZE_PATH),
            "generation6_event_manifest": artifact(g13z.G6_EVENT_MANIFEST),
        },
    }
    g0.atomic_write_json(result, result_path)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

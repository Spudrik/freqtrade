"""Score Generation 15 path decomposition, level-family, and timeframe siblings."""

from __future__ import annotations

# Bind native pools before pandas/numpy imports.
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
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


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
    market_reaction_zone_generation13_direct_controls as g13d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation15_freeze as g15z,
)


DEFAULT_RUN_ID = "g15_matched_paths_20260822a"
RECORD_ROOT = g15z.FREEZE_PATH.parent / "g15_broad_direct" / "matched_paths"
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation15_branches"
    / "matched_paths"
)
MAX_WORKERS = 4
MIN_PAIR_ROWS = 20
MIN_COINS = 5
CONTROLS = tuple(g15z.CONTROLS)
HORIZONS = tuple(g15z.HORIZONS)
METRICS = tuple(g15z.PATH_METRICS)

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
    dict.fromkeys(
        [
            "time_to_abs_0_5atr",
            *(
                column
                for horizon in HORIZONS
                for column in (
                    f"abs_excursion_atr_h{horizon}",
                    f"range_ratio_h{horizon}",
                    f"volume_ratio_h{horizon}",
                    f"pressure_change_h{horizon}",
                    f"dwell_fraction_h{horizon}",
                    f"crossings_h{horizon}",
                )
            ),
        ]
    )
)
READ_COLUMNS = tuple(
    dict.fromkeys(
        (
            *META_COLUMNS,
            *g13d.MATCH_FEATURES,
            *g13d.AUDIT_SOURCE_COLUMNS,
            *OUTCOME_COLUMNS,
        )
    )
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


def load_contracts() -> tuple[dict[str, Any], dict[str, Any]]:
    if not g15z.FREEZE_PATH.is_file():
        raise FileNotFoundError(g15z.FREEZE_PATH)
    freeze = json.loads(g15z.FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen_before_generation15_outcomes":
        raise ValueError("Generation 15 plan is not frozen.")
    source_record = freeze["source_contracts"]["generation6_event_manifest"]
    source_path = Path(source_record["path"])
    if not source_path.is_file() or g0.sha256_file(source_path) != source_record["sha256"]:
        raise ValueError("Frozen Generation 6 source manifest changed.")
    manifest = json.loads(source_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed_shared_causal_event_cache":
        raise ValueError("Generation 6 event cache is not terminal.")
    return freeze, manifest


def causal_deduplicate_by_family(frame: DataFrame) -> DataFrame:
    ordered = frame.assign(
        _distance=pd.to_numeric(
            frame["contact_close_distance_atr"], errors="coerce"
        ).abs()
    ).sort_values(
        [
            "window",
            "analysis_period",
            "source_timeframe",
            "level_family",
            "control",
            "event_time",
            "_distance",
            "level_name",
        ],
        kind="stable",
        na_position="last",
    )
    return (
        ordered.drop_duplicates(
            [
                "window",
                "analysis_period",
                "source_timeframe",
                "level_family",
                "control",
                "event_time",
            ],
            keep="first",
        )
        .drop(columns="_distance")
        .reset_index(drop=True)
    )


def matched_rows(actual: DataFrame, control: DataFrame, control_name: str) -> DataFrame:
    rows: list[DataFrame] = []
    keys = (
        "window",
        "analysis_period",
        "source_timeframe",
        "level_family",
        "approach_state",
    )
    actual_groups = actual.groupby(list(keys), observed=True, sort=False)
    control_groups = control.groupby(list(keys), observed=True, sort=False)
    for key in sorted(set(actual_groups.groups).intersection(control_groups.groups)):
        left = actual_groups.get_group(key).reset_index(drop=True)
        right = control_groups.get_group(key).reset_index(drop=True)
        pairs, audit = g1.nearest_state_pairs(
            left,
            right,
            state_columns=g13d.MATCH_FEATURES,
            minimum_event_separation_hours=max(HORIZONS),
        )
        if not pairs:
            continue
        left_positions = [item[0] for item in pairs]
        right_positions = [item[1] for item in pairs]
        selected_left = left.iloc[left_positions].reset_index(drop=True)
        selected_right = right.iloc[right_positions].reset_index(drop=True)
        result = DataFrame(
            {
                "cohort": selected_left["cohort"].astype(str),
                "pair": selected_left["pair"].astype(str),
                "window": key[0],
                "analysis_period": key[1],
                "source_timeframe": key[2],
                "level_family": key[3],
                "approach_state": key[4],
                "control": control_name,
                "event_time": pd.to_datetime(selected_left["event_time"], utc=True),
                "actual_event_time": pd.to_datetime(
                    selected_left["event_time"], utc=True
                ),
                "control_event_time": pd.to_datetime(
                    selected_right["event_time"], utc=True
                ),
                "level_name": selected_left["level_name"].astype(str),
                "match_distance": [item[2] for item in pairs],
                "pre_distance_atr_abs_difference": (
                    pd.to_numeric(selected_left["pre_distance_atr"], errors="coerce")
                    - pd.to_numeric(selected_right["pre_distance_atr"], errors="coerce")
                ).abs(),
                "eligible_actual": audit["eligible_actual"],
                "eligible_control": audit["eligible_control"],
                "state_matchable_actual": audit["state_matchable_actual"],
            }
        )
        for column in OUTCOME_COLUMNS:
            result[f"actual__{column}"] = pd.to_numeric(
                selected_left[column], errors="coerce"
            ).to_numpy(dtype=float)
            result[f"control__{column}"] = pd.to_numeric(
                selected_right[column], errors="coerce"
            ).to_numpy(dtype=float)
        rows.append(result)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else DataFrame()


def build_pair(task: PairTask) -> dict[str, Any]:
    source_path = Path(task.source_path)
    if not source_path.is_file() or g0.sha256_file(source_path) != task.source_sha256:
        raise ValueError(f"Frozen event source changed: {source_path}")
    output_path = Path(task.artifact_dir) / (
        f"{task.cohort}__{g0.pair_file_stem(task.pair)}.parquet"
    )
    if output_path.is_file() and not task.overwrite:
        existing = pd.read_parquet(output_path, columns=["pair", "control"])
        return {
            "status": "existing",
            "cohort": task.cohort,
            "pair": task.pair,
            "matched_rows": len(existing),
            "path": str(output_path),
            "sha256": g0.sha256_file(output_path),
        }
    frame = pd.read_parquet(
        source_path,
        columns=list(READ_COLUMNS),
        filters=[("control", "in", ["actual", *CONTROLS])],
    )
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="coerce")
    frame = g13d.derive_audit_features(frame)
    frame = g13d.add_analysis_windows(frame, task.cohort)
    frame = causal_deduplicate_by_family(frame)
    actual_clock = set(
        frame.loc[frame["control"].eq("actual"), "event_time"].astype("int64").tolist()
    )
    control_mask = frame["control"].isin(CONTROLS)
    contaminated = control_mask & frame["event_time"].astype("int64").isin(actual_clock)
    frame = frame.loc[~contaminated].copy()
    actual = frame.loc[frame["control"].eq("actual")].copy()
    matches = [
        matched_rows(
            actual,
            frame.loc[frame["control"].eq(control_name)].copy(),
            control_name,
        )
        for control_name in CONTROLS
    ]
    nonempty_matches = [item for item in matches if not item.empty]
    combined = (
        pd.concat(nonempty_matches, ignore_index=True, sort=False)
        if nonempty_matches
        else DataFrame()
    )
    if combined.empty:
        raise ValueError(f"No Generation 15 matched rows for {task.pair}.")
    if combined["control_event_time"].astype("int64").isin(actual_clock).any():
        raise AssertionError("A Generation 15 control coincides with a real contact.")
    g0.atomic_write_parquet(combined, output_path)
    return {
        "status": "built",
        "cohort": task.cohort,
        "pair": task.pair,
        "matched_rows": len(combined),
        "removed_control_rows_at_real_contacts": int(contaminated.sum()),
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
    if workers < 1 or workers > MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    if workers == 1:
        return [safe_build_pair(task) for task in tasks]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(safe_build_pair, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "phase": "g15_matched_pair_paths",
                        "processed": index,
                        "total": len(tasks),
                        "cohort": result.get("cohort"),
                        "pair": result.get("pair"),
                        "status": result.get("status"),
                    }
                ),
                flush=True,
            )
    return sorted(results, key=lambda item: (item.get("cohort", ""), item.get("pair", "")))


def metric_values(frame: DataFrame, metric: str, horizon: int) -> DataFrame:
    if metric == "absolute_excursion_atr":
        column = f"abs_excursion_atr_h{horizon}"
    elif metric in {"range_ratio", "volume_ratio", "dwell_fraction", "crossings"}:
        column = f"{metric}_h{horizon}"
    elif metric == "absolute_pressure_change":
        column = f"pressure_change_h{horizon}"
    elif metric in {"hit_0_5atr", "censored_time_to_0_5atr"}:
        column = "time_to_abs_0_5atr"
    else:
        raise KeyError(metric)
    actual = pd.to_numeric(frame[f"actual__{column}"], errors="coerce")
    control = pd.to_numeric(frame[f"control__{column}"], errors="coerce")
    if metric == "absolute_pressure_change":
        actual = actual.abs()
        control = control.abs()
    elif metric == "hit_0_5atr":
        actual = actual.le(horizon).astype(float)
        control = control.le(horizon).astype(float)
    elif metric == "censored_time_to_0_5atr":
        actual = actual.fillna(horizon + 1).clip(upper=horizon + 1)
        control = control.fillna(horizon + 1).clip(upper=horizon + 1)
        return DataFrame(
            {
                "actual_value": actual,
                "control_value": control,
                "difference": control - actual,
            },
            index=frame.index,
        )
    return DataFrame(
        {
            "actual_value": actual,
            "control_value": control,
            "difference": actual - control,
        },
        index=frame.index,
    )


def route_scopes(frame: DataFrame) -> list[tuple[str, str, DataFrame]]:
    rows = [("reaction_path_decomposition", "all", frame)]
    rows.extend(
        ("level_family_attribution", family, cell)
        for family, cell in frame.groupby("level_family", observed=True, sort=False)
        if family in g15z.LEVEL_FAMILIES
    )
    rows.extend(
        ("source_timeframe_attribution", timeframe, cell)
        for timeframe, cell in frame.groupby("source_timeframe", observed=True, sort=False)
        if timeframe in g15z.SOURCE_TIMEFRAMES
    )
    return rows


def summarize_matches(frame: DataFrame) -> tuple[DataFrame, DataFrame]:
    pair_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    for route_id, scope_value, source in route_scopes(frame):
        for horizon in HORIZONS:
            independent = g13d.purge_overlapping_hourly_pairs(
                source,
                separation_hours=horizon,
                group_columns=["pair", "window", "analysis_period", "control"],
            )
            if independent.empty:
                continue
            for metric in METRICS:
                values = metric_values(independent, metric, horizon)
                scored = pd.concat(
                    [
                        independent[
                            [
                                "cohort",
                                "pair",
                                "window",
                                "analysis_period",
                                "control",
                                "actual_event_time",
                            ]
                        ].reset_index(drop=True),
                        values.reset_index(drop=True),
                    ],
                    axis=1,
                ).dropna(subset=["actual_value", "control_value", "difference"])
                if scored.empty:
                    continue
                for key, cell in scored.groupby(
                    ["cohort", "pair", "window", "analysis_period", "control"],
                    observed=True,
                    sort=False,
                ):
                    pair_rows.append(
                        {
                            "route_id": route_id,
                            "scope_value": scope_value,
                            "metric": metric,
                            "horizon_hours": horizon,
                            "cohort": key[0],
                            "pair": key[1],
                            "window": key[2],
                            "analysis_period": key[3],
                            "control": key[4],
                            "independent_rows": len(cell),
                            "actual_mean": float(cell["actual_value"].mean()),
                            "control_mean": float(cell["control_value"].mean()),
                            "paired_mean_difference": float(cell["difference"].mean()),
                        }
                    )
                scored["week"] = pd.to_datetime(
                    scored["actual_event_time"], utc=True
                ).dt.floor("7D")
                weekly = (
                    scored.groupby(
                        [
                            "cohort",
                            "pair",
                            "window",
                            "analysis_period",
                            "control",
                            "week",
                        ],
                        observed=True,
                        sort=False,
                    )["difference"]
                    .agg(["sum", "size"])
                    .reset_index()
                )
                for row in weekly.itertuples(index=False):
                    weekly_rows.append(
                        {
                            "route_id": route_id,
                            "scope_value": scope_value,
                            "metric": metric,
                            "horizon_hours": horizon,
                            "cohort": row.cohort,
                            "pair": row.pair,
                            "window": row.window,
                            "analysis_period": row.analysis_period,
                            "control": row.control,
                            "week": row.week,
                            "difference_sum": float(row.sum),
                            "rows": int(row.size),
                        }
                    )
    return DataFrame.from_records(pair_rows), DataFrame.from_records(weekly_rows)


def bootstrap_weekly_blocks(
    frame: DataFrame, *, seed_key: str, samples: int = 500
) -> tuple[float, float, float]:
    if frame.empty:
        return np.nan, np.nan, np.nan
    pair_point = (
        frame.groupby("pair", observed=True)[["difference_sum", "rows"]]
        .sum()
        .eval("difference_sum / rows")
    )
    point = float(pair_point.mean())
    by_pair = {
        str(pair): list(zip(group["difference_sum"], group["rows"], strict=True))
        for pair, group in frame.groupby("pair", observed=True, sort=False)
    }
    if sum(len(items) for items in by_pair.values()) < 2:
        return point, np.nan, np.nan
    seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for sample in range(samples):
        rates: list[float] = []
        for items in by_pair.values():
            positions = rng.integers(0, len(items), len(items))
            total = 0.0
            rows = 0
            for position in positions:
                value, count = items[int(position)]
                total += float(value)
                rows += int(count)
            rates.append(total / rows)
        draws[sample] = float(np.mean(rates))
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def equal_coin_scores(pair_summary: DataFrame, weekly_blocks: DataFrame) -> DataFrame:
    keys = [
        "route_id",
        "scope_value",
        "metric",
        "horizon_hours",
        "cohort",
        "window",
        "analysis_period",
        "control",
    ]
    rows: list[dict[str, Any]] = []
    for key, cell in pair_summary.groupby(keys, observed=True, sort=False):
        eligible = cell.loc[cell["independent_rows"].ge(MIN_PAIR_ROWS)].copy()
        pairs = set(eligible["pair"].astype(str))
        blocks = weekly_blocks.loc[
            weekly_blocks["route_id"].eq(key[0])
            & weekly_blocks["scope_value"].eq(key[1])
            & weekly_blocks["metric"].eq(key[2])
            & weekly_blocks["horizon_hours"].eq(key[3])
            & weekly_blocks["cohort"].eq(key[4])
            & weekly_blocks["window"].eq(key[5])
            & weekly_blocks["analysis_period"].eq(key[6])
            & weekly_blocks["control"].eq(key[7])
            & weekly_blocks["pair"].isin(pairs)
        ]
        point, lower, upper = bootstrap_weekly_blocks(
            blocks, seed_key="|".join(map(str, key))
        )
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "eligible_coins": len(eligible),
                "positive_coins": int(eligible["paired_mean_difference"].gt(0).sum()),
                "negative_coins": int(eligible["paired_mean_difference"].lt(0).sum()),
                "independent_rows": int(eligible["independent_rows"].sum()),
                "equal_coin_actual_mean": float(eligible["actual_mean"].mean())
                if len(eligible)
                else np.nan,
                "equal_coin_control_mean": float(eligible["control_mean"].mean())
                if len(eligible)
                else np.nan,
                "equal_coin_paired_difference": point,
                "bootstrap_lower_95": lower,
                "bootstrap_upper_95": upper,
            }
        )
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def cell_decisions(scores: DataFrame) -> DataFrame:
    keys = [
        "route_id",
        "scope_value",
        "metric",
        "horizon_hours",
        "cohort",
        "window",
    ]
    rows: list[dict[str, Any]] = []
    for key, cell in scores.groupby(keys, observed=True, sort=False):
        expected = 2 * len(CONTROLS)
        supported = cell.loc[cell["eligible_coins"].ge(MIN_COINS)].copy()
        differences = supported["equal_coin_paired_difference"]
        all_positive = len(supported) == expected and differences.gt(0).all()
        all_negative = len(supported) == expected and differences.lt(0).all()
        sign = "positive" if all_positive else "negative" if all_negative else "mixed"
        strict = bool(
            (all_positive and supported["bootstrap_lower_95"].gt(0).all())
            or (all_negative and supported["bootstrap_upper_95"].lt(0).all())
        )
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "expected_period_control_cells": expected,
                "supported_period_control_cells": len(supported),
                "consistent_effect_sign": sign,
                "point_pass": bool(all_positive or all_negative),
                "strict_pass": strict,
                "minimum_eligible_coins": int(supported["eligible_coins"].min())
                if len(supported)
                else 0,
                "minimum_effect": float(differences.min())
                if len(differences)
                else np.nan,
                "maximum_effect": float(differences.max())
                if len(differences)
                else np.nan,
                "minimum_bootstrap_lower_95": float(
                    supported["bootstrap_lower_95"].min()
                )
                if len(supported)
                else np.nan,
                "maximum_bootstrap_upper_95": float(
                    supported["bootstrap_upper_95"].max()
                )
                if len(supported)
                else np.nan,
            }
        )
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def joint_decisions(cells: DataFrame) -> DataFrame:
    keys = ["route_id", "scope_value", "metric", "horizon_hours"]
    expected_cells = {
        ("normal", "standard_validation"),
        ("normal", "recent_normal_chronology"),
        ("meme", "standard_validation"),
    }
    rows: list[dict[str, Any]] = []
    for key, cell in cells.groupby(keys, observed=True, sort=False):
        available = set(zip(cell["cohort"], cell["window"], strict=True))
        complete = expected_cells.issubset(available)
        supported = cell.loc[
            cell.apply(lambda row: (row["cohort"], row["window"]) in expected_cells, axis=1)
        ]
        point = bool(complete and supported["point_pass"].all())
        strict = bool(complete and supported["strict_pass"].all())
        signs = set(supported.loc[supported["point_pass"], "consistent_effect_sign"])
        same_sign = len(signs) == 1 and len(supported) == 3
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "complete_three_cell_ladder": complete,
                "same_sign_across_three_cells": same_sign,
                "point_all_three_cells": bool(point and same_sign),
                "strict_all_three_cells": bool(strict and same_sign),
                "point_cells": int(supported["point_pass"].sum()),
                "strict_cells": int(supported["strict_pass"].sum()),
            }
        )
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    freeze, manifest = load_contracts()
    record_dir = RECORD_ROOT / args.run_id
    artifact_dir = ARTIFACT_ROOT / args.run_id / "pair_matches"
    result_path = record_dir / "g15_matched_paths_result.json"
    run_record_path = record_dir / "g15_matched_paths_run.json"
    if result_path.is_file() and not args.overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    tasks = [
        PairTask(
            cohort=str(item["cohort"]),
            pair=str(item["pair"]),
            source_path=str(item["event_path"]),
            source_sha256=str(item["event_sha256"]),
            artifact_dir=str(artifact_dir),
            overwrite=args.overwrite,
        )
        for item in manifest["tasks"]
    ]
    run_record = {
        "schema_version": 1,
        "generation": 15,
        "run_id": args.run_id,
        "status": "running",
        "started_at_utc": g0.utc_now(),
        "orchestrator_pid": os.getpid(),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "workers": args.workers,
        "worker_threads_each": 1,
        "pair_tasks": len(tasks),
        "artifact_dir": str(artifact_dir.resolve()),
        "expected_result": str(result_path.resolve()),
    }
    g0.atomic_write_json(run_record, run_record_path)
    try:
        inventory = run_tasks(tasks, args.workers)
        failures = [item for item in inventory if item["status"] == "failed"]
        if failures:
            raise RuntimeError(json.dumps(failures, indent=2, sort_keys=True))
        frames = [pd.read_parquet(item["path"]) for item in inventory]
        matched = pd.concat(frames, ignore_index=True, sort=False)
        pair_summary, weekly_blocks = summarize_matches(matched)
        scores = equal_coin_scores(pair_summary, weekly_blocks)
        decisions = cell_decisions(scores)
        joint = joint_decisions(decisions)

        inventory_path = record_dir / "pair_match_inventory.csv"
        pair_path = record_dir / "pair_path_differences.csv"
        scores_path = record_dir / "equal_coin_path_scores.csv"
        decisions_path = record_dir / "path_cell_decisions.csv"
        joint_path = record_dir / "path_joint_decisions.csv"
        g0.atomic_write_csv(DataFrame.from_records(inventory), inventory_path)
        g0.atomic_write_csv(pair_summary, pair_path)
        g0.atomic_write_csv(scores, scores_path)
        g0.atomic_write_csv(decisions, decisions_path)
        g0.atomic_write_csv(joint, joint_path)
        result = {
            "schema_version": 1,
            "generation": 15,
            "run_id": args.run_id,
            "status": "completed_generation15_matched_paths",
            "routes_completed": [
                "reaction_path_decomposition",
                "level_family_attribution",
                "source_timeframe_attribution",
            ],
            "source_contracts": {
                "generation15_freeze": artifact(g15z.FREEZE_PATH),
                "generation6_event_manifest": freeze["source_contracts"][
                    "generation6_event_manifest"
                ],
            },
            "runtime": {"workers": args.workers, "worker_threads_each": 1},
            "integrity": {
                "pair_tasks": len(tasks),
                "pair_failures": 0,
                "future_outcomes_used_for_matching": False,
                "control_times_at_any_real_contact_removed": True,
                "future_signed_direction_used": False,
                "profit_used": False,
            },
            "summary": {
                "matched_rows_before_horizon_purges": len(matched),
                "pair_metric_rows": len(pair_summary),
                "score_rows": len(scores),
                "cell_decision_rows": len(decisions),
                "joint_decision_rows": len(joint),
                "point_all_three_rows": int(joint["point_all_three_cells"].sum()),
                "strict_all_three_rows": int(joint["strict_all_three_cells"].sum()),
                "joint_55_percent_direction_target_reached": False,
            },
            "artifacts": {
                "inventory": artifact(inventory_path),
                "pair_path_differences": artifact(pair_path),
                "equal_coin_path_scores": artifact(scores_path),
                "path_cell_decisions": artifact(decisions_path),
                "path_joint_decisions": artifact(joint_path),
            },
        }
        g0.atomic_write_json(result, result_path)
        run_record.update(
            {
                "status": "completed",
                "completed_at_utc": g0.utc_now(),
                "result_path": str(result_path.resolve()),
                "result_sha256": g0.sha256_file(result_path),
            }
        )
        g0.atomic_write_json(run_record, run_record_path)
        print(json.dumps(result["summary"], indent=2, sort_keys=True))
    except Exception as exc:
        run_record.update(
            {
                "status": "failed",
                "failed_at_utc": g0.utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        g0.atomic_write_json(run_record, run_record_path)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

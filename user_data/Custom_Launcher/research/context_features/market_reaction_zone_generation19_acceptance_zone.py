"""Test Generation 19's repaired repeated-close acceptance-zone representation."""

from __future__ import annotations

# Bound numerical pools before dataframe imports.
# ruff: noqa: E402
import argparse
import json
import os
import sys
import time
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
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

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
    market_reaction_zone_generation2_density as g2d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_level_source_atlas as g17l,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation18_direct_confirmation as g18d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation18_freeze as g18z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation19_freeze as g19z,
)


DEFAULT_RUN_ID = "g19_acceptance_zone_20260823a"
RECORD_ROOT = g19z.OUTPUT_ROOT / "acceptance_zone"
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation19_branches"
    / "g19_broad_combinations"
    / "acceptance_zone"
)
HISTORY_HOURS = 168
BIN_WIDTH_ATR = 0.25
HORIZONS = g19z.HORIZONS_HOURS
CONTROLS = (
    "matched_random_time",
    "near_miss",
    "stale_72h",
    "price_shift",
    "rolling_vwap_width_matched",
)
G0_CONTROLS = ("actual", *CONTROLS[:-1])
MAX_WORKERS = 4
MIN_COVERAGE_RATIO = 0.50
MAX_MEAN_WIDTH_RATIO = 1.25


@dataclass(frozen=True)
class PairTask:
    cohort: str
    pair: str
    run_id: str
    overwrite: bool


def artifact(path: Path) -> dict[str, Any]:
    return g19z.artifact(path)


def load_freeze() -> dict[str, Any]:
    frozen = json.loads(g19z.FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation19_outcomes":
        raise ValueError("Generation 19 branch layer is not frozen.")
    return frozen


def pair_tasks(run_id: str, *, overwrite: bool) -> list[PairTask]:
    tasks: list[PairTask] = []
    for cohort, manifest_path in g17l.COHORT_MANIFESTS.items():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        tasks.extend(
            PairTask(cohort=cohort, pair=str(pair), run_id=run_id, overwrite=overwrite)
            for pair in manifest["data"]["pairs"]
        )
    if len(tasks) != 20:
        raise ValueError(f"Expected 20 Generation 19 acceptance tasks, got {len(tasks)}.")
    return tasks


def research_manifest(cohort: str) -> dict[str, Any]:
    periods = [
        {
            "id": item["id"],
            "role": "holdout",
            "start_utc": item["start_utc"],
            "end_utc_exclusive": item["end_utc_exclusive"],
        }
        for item in g18z.CONFIRMATION_PERIODS[cohort]
    ]
    return {
        "data": {
            "base_timeframe": "1h",
            "common_analysis_start_utc": "2026-06-01T00:00:00Z",
            "chronological_periods": periods,
        }
    }


def prepare_base(pair: str, cohort: str) -> DataFrame:
    return g0.prepare_base_market_frame(pair, research_manifest(cohort))


def nearest_zone(zones: g2d.DensityZoneSet, pre_close: np.ndarray) -> tuple[np.ndarray, ...]:
    distance = np.abs(zones.levels - pre_close[:, None])
    distance[~zones.valid] = np.inf
    rank = np.argmin(distance, axis=1)
    rows = np.arange(len(pre_close), dtype=np.int64)
    any_valid = zones.valid.any(axis=1)
    level = np.where(any_valid, zones.levels[rows, rank], np.nan)
    width = np.where(any_valid, zones.half_widths[rows, rank], np.nan)
    support = np.where(any_valid, zones.support_counts[rows, rank], np.nan)
    return level, width, support, rank.astype(float)


def rolling_vwap(base: DataFrame, window: int = HISTORY_HOURS) -> np.ndarray:
    high = pd.to_numeric(base["high"], errors="coerce")
    low = pd.to_numeric(base["low"], errors="coerce")
    close = pd.to_numeric(base["close"], errors="coerce")
    volume = pd.to_numeric(base["volume"], errors="coerce").clip(lower=0.0)
    typical = (high + low + close) / 3.0
    minimum = int(np.ceil(window * g2d.CLOSE_HISTORY_MINIMUM_FRACTION))
    numerator = (typical * volume).rolling(window, min_periods=minimum).sum().shift(1)
    denominator = volume.rolling(window, min_periods=minimum).sum().shift(1)
    return numerator.div(denominator.replace(0.0, np.nan)).to_numpy(dtype=float)


def add_reaction_columns(events: DataFrame) -> None:
    for horizon in HORIZONS:
        crossings = pd.to_numeric(events[f"crossings_h{horizon}"], errors="coerce")
        events[f"metric__any_recross_h{horizon}"] = np.where(
            crossings.notna(), crossings.ge(1.0).astype(float), np.nan
        )
        events[f"metric__repeated_recross_h{horizon}"] = np.where(
            crossings.notna(), crossings.ge(2.0).astype(float), np.nan
        )
        events[f"metric__dwell_fraction_h{horizon}"] = pd.to_numeric(
            events[f"dwell_fraction_h{horizon}"], errors="coerce"
        )
    events["arrival_class"] = np.where(
        events["approach_state"].isin(["from_above", "from_below"]),
        "fresh_arrival",
        "already_inside_or_unclear",
    )


def build_pair(task: PairTask) -> dict[str, Any]:
    started = time.perf_counter()
    output_path = ARTIFACT_ROOT / task.run_id / "pair_events" / (
        f"{task.cohort}__{g0.pair_file_stem(task.pair)}.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file() and not task.overwrite:
        frame = pd.read_parquet(output_path)
        return {
            "cohort": task.cohort,
            "pair": task.pair,
            "status": "existing",
            "event_rows": len(frame),
            "output_path": str(output_path.resolve()),
            "output_sha256": g0.sha256_file(output_path),
            "seconds": round(time.perf_counter() - started, 3),
        }
    base = prepare_base(task.pair, task.cohort)
    swings = g2d.confirmed_swing_points(base)
    zones = g2d.causal_density_zones(
        base,
        family="repeated_close_density",
        history_hours=HISTORY_HOURS,
        bin_width_atr=BIN_WIDTH_ATR,
        swings=swings,
    )
    pre_close = pd.to_numeric(base["pre_close"], errors="coerce").to_numpy(dtype=float)
    level, width, support, rank = nearest_zone(zones, pre_close)
    base["g19_acceptance_level"] = level
    base["g19_acceptance_width"] = width
    base["g19_acceptance_support"] = support
    base["g19_acceptance_rank"] = rank
    dates = pd.to_datetime(base["date"], utc=True, errors="raise")
    base["available_at"] = dates
    base["source_open"] = dates - pd.Timedelta(hours=HISTORY_HOURS)
    base["source_atr_14"] = base["base_atr"]
    paths = g0.future_path_matrices(base, max_horizon=max(HORIZONS))
    spec = g0.LevelSpec(
        name="repeated_close_density_168h_nearest",
        family="repeated_close_acceptance",
        batch="g19e",
        column="g19_acceptance_level",
        native_width_column="g19_acceptance_width",
        score_column="g19_acceptance_support",
        attribute_columns=("g19_acceptance_rank",),
    )
    frames = g0.build_spec_events(
        merged=base,
        paths=paths,
        spec=spec,
        pair=task.pair,
        timeframe="1h",
        zone_methods=("native_width",),
        controls=G0_CONTROLS,
        horizons=HORIZONS,
    )
    vwap_level = rolling_vwap(base)
    vwap_valid = np.isfinite(vwap_level) & np.isfinite(width) & (width > 0.0)
    vwap_spec = g0.LevelSpec(
        name="rolling_vwap_168h_width_matched",
        family="rolling_vwap_width_matched",
        batch="g19e_control",
        column="g19_acceptance_level",
    )
    vwap_events = g0.extract_episode_events(
        merged=base,
        paths=paths,
        spec=vwap_spec,
        pair=task.pair,
        timeframe="1h",
        level=vwap_level,
        valid=vwap_valid,
        half_width=width,
        source_available=dates,
        source_open=dates - pd.Timedelta(hours=HISTORY_HOURS),
        control="rolling_vwap_width_matched",
        zone_method="acceptance_native_width",
        horizons=HORIZONS,
        event_kind="contact",
    )
    if not vwap_events.empty:
        frames.append(vwap_events)
    if not frames:
        raise ValueError(f"No Generation 19 acceptance events for {task.pair}.")
    events = pd.concat(frames, ignore_index=True, sort=False)
    wanted_periods = {item["id"] for item in g18z.CONFIRMATION_PERIODS[task.cohort]}
    events = events.loc[events["period"].isin(wanted_periods)].copy()
    if events.empty:
        raise ValueError(f"No later Generation 19 acceptance rows for {task.pair}.")
    events["cohort"] = task.cohort
    add_reaction_columns(events)
    keep = [
        "cohort",
        "pair",
        "period",
        "control",
        "event_time",
        "approach_state",
        "arrival_class",
        "zone_half_width_atr",
        "level_score",
        *(
            column
            for horizon in HORIZONS
            for column in (
                f"metric__any_recross_h{horizon}",
                f"metric__repeated_recross_h{horizon}",
                f"metric__dwell_fraction_h{horizon}",
            )
        ),
    ]
    stored = events[keep].copy()
    g0.atomic_write_parquet(stored, output_path)
    controls = set(stored["control"].astype(str))
    missing_controls = sorted({"actual", *CONTROLS}.difference(controls))
    if missing_controls:
        raise ValueError(f"{task.pair} lacks acceptance controls: {missing_controls}")
    return {
        "cohort": task.cohort,
        "pair": task.pair,
        "status": "completed",
        "event_rows": len(stored),
        "controls": sorted(controls),
        "periods": sorted(stored["period"].astype(str).unique()),
        "output_path": str(output_path.resolve()),
        "output_sha256": g0.sha256_file(output_path),
        "seconds": round(time.perf_counter() - started, 3),
    }


def safe_build_pair(task: PairTask) -> dict[str, Any]:
    try:
        return build_pair(task)
    except Exception as exc:
        return {
            "cohort": task.cohort,
            "pair": task.pair,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_tasks(tasks: Sequence[PairTask], workers: int) -> list[dict[str, Any]]:
    selected_workers = max(1, min(int(workers), MAX_WORKERS))
    if selected_workers == 1:
        return [safe_build_pair(task) for task in tasks]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=selected_workers) as executor:
        futures = {executor.submit(safe_build_pair, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "phase": "g19_acceptance_pair",
                        "completed": len(results),
                        "total": len(tasks),
                        "pair": result["pair"],
                        "status": result["status"],
                    }
                ),
                flush=True,
            )
    order = {(task.cohort, task.pair): number for number, task in enumerate(tasks)}
    return sorted(results, key=lambda row: order[(row["cohort"], row["pair"])])


def summarize(inventory: Sequence[dict[str, Any]]) -> tuple[DataFrame, DataFrame, DataFrame]:
    frames = [pd.read_parquet(item["output_path"]) for item in inventory]
    events = pd.concat(frames, ignore_index=True, sort=False)
    events["g18_period"] = events["period"]
    scopes: list[DataFrame] = []
    for arrival in ("all", "fresh_arrival", "already_inside_or_unclear"):
        selected = (
            events.copy()
            if arrival == "all"
            else events.loc[events["arrival_class"].eq(arrival)].copy()
        )
        selected["scope_kind"] = "acceptance_arrival_state"
        selected["scope_value"] = arrival
        scopes.append(selected)
    scoped = pd.concat(scopes, ignore_index=True, sort=False)
    keys = [
        "cohort",
        "pair",
        "g18_period",
        "scope_kind",
        "scope_value",
        "control",
    ]
    rows: list[DataFrame] = []
    for metric in ("any_recross", "repeated_recross", "dwell_fraction"):
        for horizon in HORIZONS:
            column = f"metric__{metric}_h{horizon}"
            grouped = (
                scoped[keys + [column]]
                .dropna(subset=[column])
                .groupby(keys, observed=True, sort=False)[column]
                .agg(["mean", "size"])
                .reset_index()
                .rename(columns={"mean": "outcome_mean", "size": "event_rows"})
            )
            grouped["metric"] = metric
            grouped["horizon_hours"] = horizon
            rows.append(grouped)
    summary = pd.concat(rows, ignore_index=True, sort=False)
    contrasts = g18d.paired_contrasts(summary, CONTROLS)
    width_keys = [
        "cohort",
        "pair",
        "g18_period",
        "scope_kind",
        "scope_value",
        "control",
    ]
    width_summary = (
        scoped[width_keys + ["zone_half_width_atr"]]
        .dropna(subset=["zone_half_width_atr"])
        .groupby(width_keys, observed=True, sort=False)["zone_half_width_atr"]
        .agg(["mean", "size"])
        .reset_index()
        .rename(columns={"mean": "mean_width_atr", "size": "width_rows"})
    )
    actual_width = width_summary.loc[width_summary["control"].eq("actual")].drop(
        columns="control"
    )
    balance_rows: list[DataFrame] = []
    balance_keys = width_keys[:-1]
    for control in CONTROLS:
        control_width = width_summary.loc[width_summary["control"].eq(control)].drop(
            columns="control"
        )
        balanced = actual_width.merge(
            control_width,
            on=balance_keys,
            how="inner",
            suffixes=("_actual", "_control"),
            validate="one_to_one",
        )
        balanced["comparison"] = control
        minimum_rows = balanced[["width_rows_actual", "width_rows_control"]].min(axis=1)
        maximum_rows = balanced[["width_rows_actual", "width_rows_control"]].max(axis=1)
        balanced["coverage_ratio"] = minimum_rows.div(maximum_rows.replace(0, np.nan))
        minimum_width = balanced[
            ["mean_width_atr_actual", "mean_width_atr_control"]
        ].min(axis=1)
        maximum_width = balanced[
            ["mean_width_atr_actual", "mean_width_atr_control"]
        ].max(axis=1)
        balanced["mean_width_ratio"] = maximum_width.div(
            minimum_width.replace(0.0, np.nan)
        )
        balanced["comparable_width_and_coverage"] = (
            balanced["coverage_ratio"].ge(MIN_COVERAGE_RATIO)
            & balanced["mean_width_ratio"].le(MAX_MEAN_WIDTH_RATIO)
        )
        balance_rows.append(balanced)
    width_balance = pd.concat(balance_rows, ignore_index=True, sort=False)
    contrasts = contrasts.merge(
        width_balance[
            balance_keys
            + [
                "comparison",
                "mean_width_atr_actual",
                "mean_width_atr_control",
                "coverage_ratio",
                "mean_width_ratio",
                "comparable_width_and_coverage",
            ]
        ],
        on=balance_keys + ["comparison"],
        how="left",
        validate="many_to_one",
    )
    contrasts["eligible_pair"] &= contrasts[
        "comparable_width_and_coverage"
    ].fillna(False)
    question_keys = ("scope_kind", "scope_value", "metric", "horizon_hours")
    scores = g18d.period_scores(contrasts, question_keys)
    decisions = g18d.whole_decisions(scores, question_keys, CONTROLS)
    return contrasts, scores, decisions


def output_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "contrasts": run_dir / "g19_acceptance_pair_contrasts.csv",
        "scores": run_dir / "g19_acceptance_period_scores.csv",
        "decisions": run_dir / "g19_acceptance_decisions.csv",
        "result": run_dir / "g19_acceptance_zone_result.json",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    load_freeze()
    run_dir = RECORD_ROOT / args.run_id
    paths = output_paths(run_dir)
    if paths["result"].is_file() and not args.overwrite:
        print(paths["result"].read_text(encoding="utf-8"))
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    inventory = run_tasks(pair_tasks(args.run_id, overwrite=args.overwrite), args.workers)
    failures = [item for item in inventory if item["status"] == "failed"]
    if failures:
        raise RuntimeError(f"Generation 19 acceptance pair failures: {failures}")
    contrasts, scores, decisions = summarize(inventory)
    for name, frame in (
        ("contrasts", contrasts),
        ("scores", scores),
        ("decisions", decisions),
    ):
        g0.atomic_write_csv(frame, paths[name])
    result = {
        "schema_version": 1,
        "generation": 19,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation19_acceptance_zone",
        "branch_completed": "g19e_rational_acceptance_zone_extension",
        "pair_tasks_completed": len(inventory),
        "strict_question_rows": int(
            decisions["status"].eq("strict_holdout_confirmation").sum()
        ),
        "research_boundary": {
            "profit_used": False,
            "future_signed_direction_used": False,
            "direction_branch_remained_parked": True,
        },
        "source_contracts": {"generation19_freeze": artifact(g19z.FREEZE_PATH)},
        "inventory": inventory,
        "artifacts": {
            name: artifact(paths[name]) for name in ("contrasts", "scores", "decisions")
        },
        "result_path": str(paths["result"].resolve()),
    }
    g0.atomic_write_json(result, paths["result"])
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

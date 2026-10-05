"""Run Generation 18's frozen coin-group and 1h/4h/8h portability sibling."""

from __future__ import annotations

# Bind numerical pools before pandas/numpy imports.
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
    market_reaction_zone_generation17_level_source_atlas as g17l,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation18_direct_confirmation as g18d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation18_freeze as g18z,
)


DEFAULT_RUN_ID = "g18_timeframe_portability_20260823a"
RECORD_ROOT = g18z.OUTPUT_ROOT / "timeframe_portability"
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation18_branches"
    / "g18_broad_confirmation"
    / "timeframe_portability"
)
SOURCE_REGISTRY_PATH = RECORD_ROOT / "g18_timeframe_source_registry_freeze.json"
COHORT_MANIFESTS = g17l.COHORT_MANIFESTS
SOURCE_TIMEFRAMES = g18z.SOURCE_TIMEFRAMES
HORIZONS = g18z.HORIZONS_HOURS
CONTROLS = ("actual", *g18z.CONTROLS)
CONTROL_COMPARISONS = g18z.CONTROLS
METRICS = ("crossings", "unsigned_reaction", "volume_ratio", "repeated_recross")
PROFILE_LOOKBACK_BARS = 72
PROFILE_BINS = 48
PROFILE_CADENCE = {"1h": 4, "4h": 1, "8h": 1}
MIN_PAIR_ROWS = g18d.MIN_PAIR_ROWS


@dataclass(frozen=True)
class PairTask:
    cohort: str
    pair: str
    manifest_path: str
    output_dir: str
    overwrite: bool


def artifact(path: Path) -> dict[str, Any]:
    return g18z.artifact(path)


def load_freeze() -> dict[str, Any]:
    frozen = g18d.load_freeze()
    branch = next(
        item
        for item in frozen["branches"]
        if item["branch_id"] == "g18e_coin_group_and_timeframe_portability"
    )
    if tuple(branch["source_timeframes"]) != SOURCE_TIMEFRAMES:
        raise ValueError("Generation 18 source-timeframe surface drifted after freeze.")
    return frozen


def level_specs() -> list[g0.LevelSpec]:
    definitions = (
        ("vp72_bins48_hvn_q80", "adaptive_volume_profile_nodes"),
        ("vp72_bins48_lvn_q10", "adaptive_volume_profile_nodes"),
        ("donchian24_upper", "donchian_boundaries"),
        ("donchian24_lower", "donchian_boundaries"),
        ("donchian72_upper", "donchian_boundaries"),
        ("donchian72_lower", "donchian_boundaries"),
        ("vwap72_upper_2sd", "rolling_vwap_deviation_bands"),
        ("vwap72_lower_2sd", "rolling_vwap_deviation_bands"),
    )
    return [
        g0.LevelSpec(
            name=name,
            family=family,
            batch="g18e_timeframe_portability",
            column=f"level__{name}",
        )
        for name, family in definitions
    ]


def freeze_source_registry() -> dict[str, Any]:
    load_freeze()
    registry = {
        "schema_version": 1,
        "generation": 18,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation18_timeframe_outcomes",
        "branch_id": "g18e_coin_group_and_timeframe_portability",
        "source_timeframes": list(SOURCE_TIMEFRAMES),
        "source_specs": [g0.spec_to_dict(spec) for spec in level_specs()],
        "profile_contract": {
            "lookback_completed_source_bars": PROFILE_LOOKBACK_BARS,
            "bins": PROFILE_BINS,
            "roles": ["nearest_hvn_q80", "nearest_lvn_q10"],
            "recalculation_cadence_source_bars": PROFILE_CADENCE,
            "price_sample": "source-candle typical price",
            "weight": "source-candle volume",
        },
        "donchian_lookback_completed_source_bars": [24, 72],
        "rolling_vwap_lookback_completed_source_bars": 72,
        "rolling_vwap_deviation_multiple": 2,
        "availability": "after each completed source candle",
        "controls": list(CONTROLS),
        "zone_methods": ["standard_base_atr"],
        "horizons_hours": list(HORIZONS),
        "metrics": list(METRICS),
        "single_and_cluster_surfaces_equal": True,
        "higher_timeframe_has_automatic_precedence": False,
        "outcome_selected_parameters": False,
        "future_signed_direction_used": False,
        "profit_used": False,
        "source_contracts": {"generation18_freeze": artifact(g18z.FREEZE_PATH)},
    }
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    if SOURCE_REGISTRY_PATH.is_file():
        existing = json.loads(SOURCE_REGISTRY_PATH.read_text(encoding="utf-8"))
        left = {key: value for key, value in existing.items() if key != "created_at_utc"}
        right = {key: value for key, value in registry.items() if key != "created_at_utc"}
        if left != right:
            raise ValueError("Generation 18 timeframe registry changed after its freeze.")
        return existing
    g0.atomic_write_json(registry, SOURCE_REGISTRY_PATH)
    return registry


def profile_columns(source: DataFrame, timeframe: str) -> DataFrame:
    prices = ((source["high"] + source["low"] + source["close"]) / 3.0).to_numpy(
        dtype=float
    )
    weights = pd.to_numeric(source["volume"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(source["low"], errors="coerce").to_numpy(dtype=float)
    highs = pd.to_numeric(source["high"], errors="coerce").to_numpy(dtype=float)
    references = pd.to_numeric(source["close"], errors="coerce").to_numpy(dtype=float)
    hvn = np.full(len(source), np.nan)
    lvn = np.full(len(source), np.nan)
    cadence = PROFILE_CADENCE[timeframe]
    for index in range(PROFILE_LOOKBACK_BARS - 1, len(source), cadence):
        start = index - PROFILE_LOOKBACK_BARS + 1
        snapshot = g17l.profile_snapshot(
            prices[start : index + 1],
            weights[start : index + 1],
            lows[start : index + 1],
            highs[start : index + 1],
            references[index],
            PROFILE_BINS,
        )
        hvn[index] = snapshot["nearest_hvn_q80"]
        lvn[index] = snapshot["nearest_lvn_q10"]
    return DataFrame(
        {
            "level__vp72_bins48_hvn_q80": Series(hvn).ffill(),
            "level__vp72_bins48_lvn_q10": Series(lvn).ffill(),
        }
    )


def source_level_surface(pair: str, timeframe: str) -> DataFrame:
    path = g0.ohlcv_path(pair, timeframe)
    source = g0.load_ohlcv(path)
    source = source.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    high = pd.to_numeric(source["high"], errors="coerce")
    low = pd.to_numeric(source["low"], errors="coerce")
    close = pd.to_numeric(source["close"], errors="coerce")
    volume = pd.to_numeric(source["volume"], errors="coerce")
    typical = (high + low + close) / 3.0
    output = DataFrame(
        {
            "source_open": source["date"],
            "available_at": source["date"] + g0.timeframe_delta(timeframe),
            "source_atr_14": g0.wilder_atr(source, 14),
        }
    )
    output = pd.concat([output, profile_columns(source, timeframe)], axis=1)
    for lookback in (24, 72):
        output[f"level__donchian{lookback}_upper"] = high.rolling(
            lookback, min_periods=lookback
        ).max()
        output[f"level__donchian{lookback}_lower"] = low.rolling(
            lookback, min_periods=lookback
        ).min()
    weighted = typical * volume
    sum_volume = volume.rolling(72, min_periods=72).sum()
    mean = weighted.rolling(72, min_periods=72).sum().div(sum_volume)
    second = (typical.pow(2) * volume).rolling(72, min_periods=72).sum().div(sum_volume)
    deviation = (second - mean.pow(2)).clip(lower=0.0).pow(0.5)
    output["level__vwap72_upper_2sd"] = mean + 2.0 * deviation
    output["level__vwap72_lower_2sd"] = mean - 2.0 * deviation
    output.replace([np.inf, -np.inf], np.nan, inplace=True)
    return output


def merged_surface(pair: str, timeframe: str, manifest: dict[str, Any]) -> DataFrame:
    base = g0.prepare_base_market_frame(pair, manifest)
    source = source_level_surface(pair, timeframe)
    merged = pd.merge_asof(
        base.sort_values("date"),
        source.sort_values("available_at"),
        left_on="date",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    available = merged["available_at"].notna()
    if (merged.loc[available, "available_at"] > merged.loc[available, "date"]).any():
        raise AssertionError("Timeframe merge admitted a future source candle.")
    return merged


def add_cluster_metadata(events: DataFrame) -> DataFrame:
    keys = ["cohort", "pair", "control", "event_time"]
    events = events.copy()
    events["level_variant_key"] = (
        events["source_timeframe"].astype(str) + "|" + events["level_name"].astype(str)
    )
    counts = (
        events.groupby(keys, observed=True, sort=False)
        .agg(
            independent_family_count=("level_family", "nunique"),
            source_timeframe_count=("source_timeframe", "nunique"),
            convergent_variant_count=("level_variant_key", "nunique"),
        )
        .reset_index()
    )
    family_counts = (
        events.groupby([*keys, "level_family"], observed=True, sort=False)[
            "source_timeframe"
        ]
        .nunique()
        .rename("family_timeframe_count")
        .reset_index()
    )
    return events.merge(counts, on=keys, validate="many_to_one").merge(
        family_counts, on=[*keys, "level_family"], validate="many_to_one"
    )


def scoped_events(events: DataFrame) -> DataFrame:
    keys = ["cohort", "pair", "g18_period", "control", "event_time"]
    frames: list[DataFrame] = []
    family_timeframe = g18d.nearest_anchor(
        events, [*keys, "level_family", "source_timeframe"]
    )
    family_timeframe["scope_kind"] = "family_timeframe"
    family_timeframe["scope_value"] = family_timeframe["level_family"]
    frames.append(family_timeframe)

    isolated = events.loc[events["family_timeframe_count"].eq(1)].copy()
    isolated = g18d.nearest_anchor(isolated, [*keys, "level_family"])
    isolated["scope_kind"] = "isolated_family_contact"
    isolated["scope_value"] = isolated["level_family"]
    frames.append(isolated)

    same_family_cluster = events.loc[events["family_timeframe_count"].ge(2)].copy()
    same_family_cluster = g18d.nearest_anchor(
        same_family_cluster, [*keys, "level_family"]
    )
    same_family_cluster["scope_kind"] = "same_family_cross_timeframe_cluster"
    same_family_cluster["scope_value"] = same_family_cluster["level_family"]
    same_family_cluster["source_timeframe"] = "multi"
    frames.append(same_family_cluster)

    cross_family = events.loc[
        events["independent_family_count"].ge(2)
        & events["source_timeframe_count"].ge(2)
    ].copy()
    cross_family = g18d.nearest_anchor(cross_family, keys)
    cross_family["scope_kind"] = "cross_family_cross_timeframe_cluster"
    cross_family["scope_value"] = "two_or_more_families_across_timeframes"
    cross_family["source_timeframe"] = "multi"
    frames.append(cross_family)
    return pd.concat(frames, ignore_index=True, sort=False)


def pair_summary(events: DataFrame) -> DataFrame:
    scoped = scoped_events(events)
    keys = [
        "cohort",
        "pair",
        "g18_period",
        "scope_kind",
        "scope_value",
        "source_timeframe",
        "control",
    ]
    rows: list[DataFrame] = []
    for metric in METRICS:
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
    return pd.concat(rows, ignore_index=True, sort=False)


def build_pair(task: PairTask) -> dict[str, Any]:
    output_dir = Path(task.output_dir)
    stem = f"{task.cohort}__{g0.pair_file_stem(task.pair)}"
    summary_path = output_dir / "pair_summaries" / f"{stem}.parquet"
    audit_path = output_dir / "pair_audits" / f"{stem}.json"
    if summary_path.is_file() and audit_path.is_file() and not task.overwrite:
        return {**json.loads(audit_path.read_text(encoding="utf-8")), "status": "existing"}
    manifest = json.loads(Path(task.manifest_path).read_text(encoding="utf-8"))
    pair_freeze = {
        "schema_version": 1,
        "generation": 18,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_pair_timeframe_surface_before_outcomes",
        "cohort": task.cohort,
        "pair": task.pair,
        "source_timeframes": list(SOURCE_TIMEFRAMES),
        "source_registry": artifact(SOURCE_REGISTRY_PATH),
        "ohlcv": {
            timeframe: artifact(g0.ohlcv_path(task.pair, timeframe))
            for timeframe in SOURCE_TIMEFRAMES
        },
        "future_outcomes_opened": False,
    }
    pair_freeze_path = output_dir / "pair_source_freezes" / f"{stem}.json"
    g0.atomic_write_json(pair_freeze, pair_freeze_path)

    event_frames: list[DataFrame] = []
    for timeframe in SOURCE_TIMEFRAMES:
        merged = merged_surface(task.pair, timeframe, manifest)
        paths = g0.future_path_matrices(merged, max(HORIZONS))
        for spec in level_specs():
            event_frames.extend(
                g0.build_spec_events(
                    merged=merged,
                    paths=paths,
                    spec=spec,
                    pair=task.pair,
                    timeframe=timeframe,
                    zone_methods=("standard_base_atr",),
                    controls=CONTROLS,
                    horizons=HORIZONS,
                )
            )
    if not event_frames:
        raise ValueError(f"No timeframe-portability events for {task.pair}.")
    events = pd.concat(event_frames, ignore_index=True, sort=False)
    events["cohort"] = task.cohort
    events["event_time"] = pd.to_datetime(events["event_time"], utc=True, errors="raise")
    events["g18_period"] = g18d.assign_confirmation_period(
        events["event_time"], task.cohort
    )
    events = events.loc[events["g18_period"].ne("outside_g18_confirmation")].copy()
    g18d.add_reaction_metrics(events)
    events = add_cluster_metadata(events)
    summary = pair_summary(events)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(summary, summary_path)
    audit = {
        "status": "built",
        "cohort": task.cohort,
        "pair": task.pair,
        "events": len(events),
        "summary_rows": len(summary),
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": g0.sha256_file(summary_path),
        "source_freeze_path": str(pair_freeze_path.resolve()),
        "source_freeze_sha256": g0.sha256_file(pair_freeze_path),
    }
    g0.atomic_write_json(audit, audit_path)
    return audit


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
    with ProcessPoolExecutor(max_workers=max(1, min(int(workers), 4))) as executor:
        futures = {executor.submit(safe_build_pair, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "phase": "g18_timeframe_pair",
                        "processed": index,
                        "total": len(tasks),
                        "pair": result.get("pair"),
                        "status": result.get("status"),
                    }
                ),
                flush=True,
            )
    return sorted(results, key=lambda item: (item.get("cohort", ""), item.get("pair", "")))


def control_contrasts(summary: DataFrame) -> DataFrame:
    keys = [
        "cohort",
        "pair",
        "g18_period",
        "scope_kind",
        "scope_value",
        "source_timeframe",
        "metric",
        "horizon_hours",
    ]
    actual = summary.loc[summary["control"].eq("actual")].drop(columns="control")
    rows: list[DataFrame] = []
    for control in CONTROL_COMPARISONS:
        baseline = summary.loc[summary["control"].eq(control)].drop(columns="control")
        merged = actual.merge(
            baseline,
            on=keys,
            how="inner",
            suffixes=("_actual", "_control"),
            validate="one_to_one",
        )
        merged["comparison"] = control
        merged["difference"] = merged["outcome_mean_actual"] - merged["outcome_mean_control"]
        merged["eligible_pair"] = (
            merged["event_rows_actual"].ge(MIN_PAIR_ROWS)
            & merged["event_rows_control"].ge(MIN_PAIR_ROWS)
        )
        rows.append(merged)
    return pd.concat(rows, ignore_index=True, sort=False)


def timeframe_head_to_head(summary: DataFrame) -> DataFrame:
    source = summary.loc[
        summary["control"].eq("actual") & summary["scope_kind"].eq("family_timeframe")
    ].copy()
    keys = [
        "cohort",
        "pair",
        "g18_period",
        "scope_value",
        "metric",
        "horizon_hours",
    ]
    baseline = source.loc[source["source_timeframe"].eq("1h")].drop(
        columns=["control", "scope_kind", "source_timeframe"]
    )
    rows: list[DataFrame] = []
    for timeframe in ("4h", "8h"):
        candidate = source.loc[source["source_timeframe"].eq(timeframe)].drop(
            columns=["control", "scope_kind", "source_timeframe"]
        )
        merged = candidate.merge(
            baseline,
            on=keys,
            suffixes=("_candidate", "_1h"),
            validate="one_to_one",
        )
        merged["candidate_timeframe"] = timeframe
        merged["comparison"] = "1h_actual_contact"
        merged["difference"] = (
            merged["outcome_mean_candidate"] - merged["outcome_mean_1h"]
        )
        merged["eligible_pair"] = (
            merged["event_rows_candidate"].ge(MIN_PAIR_ROWS)
            & merged["event_rows_1h"].ge(MIN_PAIR_ROWS)
        )
        rows.append(merged)
    return pd.concat(rows, ignore_index=True, sort=False)


def cluster_component_contrasts(summary: DataFrame) -> DataFrame:
    source = summary.loc[summary["control"].eq("actual")].copy()
    keys = [
        "cohort",
        "pair",
        "g18_period",
        "scope_value",
        "metric",
        "horizon_hours",
    ]
    cluster = source.loc[
        source["scope_kind"].eq("same_family_cross_timeframe_cluster")
    ].drop(columns=["control", "scope_kind", "source_timeframe"])
    isolated = source.loc[source["scope_kind"].eq("isolated_family_contact")].copy()
    isolated["weighted_outcome"] = (
        isolated["outcome_mean"] * isolated["event_rows"]
    )
    isolated = (
        isolated.groupby(keys, observed=True, sort=False)
        .agg(
            weighted_outcome=("weighted_outcome", "sum"),
            event_rows=("event_rows", "sum"),
        )
        .reset_index()
    )
    isolated["outcome_mean"] = isolated["weighted_outcome"].div(
        isolated["event_rows"].replace(0, np.nan)
    )
    isolated.drop(columns="weighted_outcome", inplace=True)
    merged = cluster.merge(
        isolated,
        on=keys,
        suffixes=("_cluster", "_isolated"),
        validate="one_to_one",
    )
    merged["comparison"] = "isolated_same_family_contact"
    merged["difference"] = merged["outcome_mean_cluster"] - merged["outcome_mean_isolated"]
    merged["eligible_pair"] = (
        merged["event_rows_cluster"].ge(MIN_PAIR_ROWS)
        & merged["event_rows_isolated"].ge(MIN_PAIR_ROWS)
    )
    return merged


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    freeze_source_registry()
    run_dir = RECORD_ROOT / args.run_id
    artifact_dir = ARTIFACT_ROOT / args.run_id
    result_path = run_dir / "g18_timeframe_portability_result.json"
    if result_path.is_file() and not args.overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[PairTask] = []
    for cohort, manifest_path in COHORT_MANIFESTS.items():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tasks.extend(
            PairTask(
                cohort=cohort,
                pair=str(pair),
                manifest_path=str(manifest_path.resolve()),
                output_dir=str(artifact_dir.resolve()),
                overwrite=bool(args.overwrite),
            )
            for pair in manifest["data"]["pairs"]
        )
    inventory = run_tasks(tasks, args.workers)
    failures = [item for item in inventory if item["status"] == "failed"]
    if failures:
        raise RuntimeError(f"Generation 18 timeframe failures: {failures}")
    summary = pd.concat(
        [pd.read_parquet(item["summary_path"]) for item in inventory],
        ignore_index=True,
        sort=False,
    )
    controls = control_contrasts(summary)
    control_scores = g18d.period_scores(
        controls,
        ("scope_kind", "scope_value", "source_timeframe", "metric", "horizon_hours"),
    )
    control_decisions = g18d.whole_decisions(
        control_scores,
        ("scope_kind", "scope_value", "source_timeframe", "metric", "horizon_hours"),
        CONTROL_COMPARISONS,
    )
    head_to_head = timeframe_head_to_head(summary)
    head_scores = g18d.period_scores(
        head_to_head,
        ("scope_value", "candidate_timeframe", "metric", "horizon_hours"),
    )
    head_decisions = g18d.whole_decisions(
        head_scores,
        ("scope_value", "candidate_timeframe", "metric", "horizon_hours"),
        ("1h_actual_contact",),
    )
    clusters = cluster_component_contrasts(summary)
    cluster_scores = g18d.period_scores(
        clusters,
        ("scope_value", "metric", "horizon_hours"),
    )
    cluster_decisions = g18d.whole_decisions(
        cluster_scores,
        ("scope_value", "metric", "horizon_hours"),
        ("isolated_same_family_contact",),
    )
    outputs = {
        "inventory": run_dir / "g18_timeframe_pair_inventory.csv",
        "summary": artifact_dir / "g18_timeframe_pair_summary.parquet",
        "control_contrasts": run_dir / "g18_timeframe_control_contrasts.csv",
        "control_scores": run_dir / "g18_timeframe_control_scores.csv",
        "control_decisions": run_dir / "g18_timeframe_control_decisions.csv",
        "head_to_head": run_dir / "g18_timeframe_head_to_head.csv",
        "head_scores": run_dir / "g18_timeframe_head_scores.csv",
        "head_decisions": run_dir / "g18_timeframe_head_decisions.csv",
        "cluster_contrasts": run_dir / "g18_timeframe_cluster_contrasts.csv",
        "cluster_scores": run_dir / "g18_timeframe_cluster_scores.csv",
        "cluster_decisions": run_dir / "g18_timeframe_cluster_decisions.csv",
    }
    g0.atomic_write_csv(DataFrame(inventory), outputs["inventory"])
    g0.atomic_write_parquet(summary, outputs["summary"])
    for name, frame in (
        ("control_contrasts", controls),
        ("control_scores", control_scores),
        ("control_decisions", control_decisions),
        ("head_to_head", head_to_head),
        ("head_scores", head_scores),
        ("head_decisions", head_decisions),
        ("cluster_contrasts", clusters),
        ("cluster_scores", cluster_scores),
        ("cluster_decisions", cluster_decisions),
    ):
        g0.atomic_write_csv(frame, outputs[name])
    result = {
        "schema_version": 1,
        "generation": 18,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation18_timeframe_portability",
        "branch_id": "g18e_coin_group_and_timeframe_portability",
        "pairs_completed": len(inventory),
        "source_timeframes": list(SOURCE_TIMEFRAMES),
        "level_specs_per_timeframe": len(level_specs()),
        "strict_control_rows": int(
            control_decisions["status"].eq("strict_holdout_confirmation").sum()
        ),
        "strict_higher_vs_1h_rows": int(
            head_decisions["status"].eq("strict_holdout_confirmation").sum()
        ),
        "strict_cluster_vs_isolated_rows": int(
            cluster_decisions["status"].eq("strict_holdout_confirmation").sum()
        ),
        "research_boundary": {
            "profit_used": False,
            "future_signed_direction_used": False,
            "higher_timeframe_precedence_assumed": False,
        },
        "source_contracts": {
            "generation18_freeze": artifact(g18z.FREEZE_PATH),
            "source_registry": artifact(SOURCE_REGISTRY_PATH),
        },
        "artifacts": {name: artifact(path) for name, path in outputs.items()},
        "result_path": str(result_path.resolve()),
    }
    g0.atomic_write_json(result, result_path)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

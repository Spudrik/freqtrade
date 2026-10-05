"""Build and test Generation 17's frozen rational level-source atlas."""

from __future__ import annotations

# Bind numerical pools before pandas/numpy imports.
# ruff: noqa: E402
import argparse
import hashlib
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
    "VECLIB_MAXIMUM_THREADS",
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
    market_reaction_zone_generation16_direct_attribution as g16d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_freeze as g17z,
)


DEFAULT_RUN_ID = "g17_rational_level_source_atlas_20260822a"
MEME_MANIFEST = (
    g0.OUTPUT_ROOT / "generation2_shared" / "g2_meme_reaction_manifest.json"
)
COHORT_MANIFESTS = {"normal": g0.DEFAULT_MANIFEST, "meme": MEME_MANIFEST}
RECORD_ROOT = g17z.OUTPUT_ROOT / "level_source_atlas"
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation17_branches"
    / "g17_broad_branch_layer"
    / "level_source_atlas"
)
SOURCE_REGISTRY_PATH = RECORD_ROOT / "g17_level_source_registry_freeze.json"
HORIZONS = (1, 2, 4)
CONTROLS = ("actual", "matched_random_time", "near_miss", "stale_72h", "price_shift")
CONTROL_COMPARISONS = tuple(control for control in CONTROLS if control != "actual")
ZONE_METHODS = ("standard_base_atr",)
METRICS = ("unsigned_reaction", "volume_ratio", "crossings", "two_sided_traversal")
PROFILE_LOOKBACKS = (72, 168, 720)
PROFILE_BIN_MODES: tuple[int | str, ...] = (24, 48, 96, "fd")
PROFILE_CADENCE = {72: 4, 168: 8, 720: 24}
PROFILE_ROLES = (
    "poc",
    "nearest_value_boundary",
    "nearest_hvn_q80",
    "nearest_hvn_q90",
    "nearest_lvn_q10",
    "nearest_lvn_q20",
)
MIN_PAIR_ROWS = 20
BOOTSTRAP_SAMPLES = 2000


@dataclass(frozen=True)
class PairTask:
    cohort: str
    pair: str
    manifest_path: str
    output_dir: str
    overwrite: bool


def artifact(path: Path) -> dict[str, Any]:
    return g17z.artifact(path)


def load_branch_freeze() -> dict[str, Any]:
    frozen = json.loads(g17z.FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation17_outcomes":
        raise ValueError("Generation 17 branch layer is not frozen.")
    branch = next(
        item for item in frozen["branches"] if item["branch_id"] == "g17d_broad_level_sources"
    )
    expected = {
        "adaptive_volume_profile_nodes",
        "rolling_vwap_deviation_bands",
        "donchian_boundaries",
        "weekly_pivot_grid",
        "generic_ma_bollinger_negative_control",
    }
    if set(branch["fixed_source_families"]) != expected:
        raise ValueError("Generation 17 rational level families drifted after freeze.")
    return frozen


def vp_name(lookback: int, bins: int | str, role: str) -> str:
    return f"vp_lb{lookback}_bins{bins}_{role}"


def source_specs() -> list[g0.LevelSpec]:
    specs: list[g0.LevelSpec] = []
    for lookback in PROFILE_LOOKBACKS:
        for bins in PROFILE_BIN_MODES:
            for role in PROFILE_ROLES:
                name = vp_name(lookback, bins, role)
                specs.append(
                    g0.LevelSpec(
                        name=name,
                        family="adaptive_volume_profile_nodes",
                        batch="g17d_rational_sources",
                        column=f"level__{name}",
                    )
                )
    for lookback in (24, 72, 168, 720):
        for role in ("vwap", "upper_1sd", "lower_1sd", "upper_2sd", "lower_2sd"):
            name = f"rolling_vwap_lb{lookback}_{role}"
            specs.append(
                g0.LevelSpec(
                    name=name,
                    family="rolling_vwap_deviation_bands",
                    batch="g17d_rational_sources",
                    column=f"level__{name}",
                )
            )
    for lookback in (24, 72, 168, 720):
        for role in ("upper", "lower"):
            name = f"donchian_lb{lookback}_{role}"
            specs.append(
                g0.LevelSpec(
                    name=name,
                    family="donchian_boundaries",
                    batch="g17d_rational_sources",
                    column=f"level__{name}",
                )
            )
    for role in ("pivot", "r1", "s1", "r2", "s2"):
        specs.append(
            g0.LevelSpec(
                name=f"weekly_{role}",
                family="weekly_pivot_grid",
                batch="g17d_rational_sources",
                column=f"level__weekly_{role}",
            )
        )
    for role in (
        "sma20",
        "sma50",
        "sma200",
        "ema20",
        "ema50",
        "bollinger_upper20",
        "bollinger_mid20",
        "bollinger_lower20",
    ):
        specs.append(
            g0.LevelSpec(
                name=f"negative_{role}",
                family="generic_ma_bollinger_negative_control",
                batch="g17d_rational_sources",
                column=f"level__negative_{role}",
            )
        )
    return specs


def freeze_source_registry() -> dict[str, Any]:
    frozen = load_branch_freeze()
    specs = source_specs()
    registry = {
        "schema_version": 1,
        "generation": 17,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation17_level_outcomes",
        "branch_id": "g17d_broad_level_sources",
        "source_specs": [g0.spec_to_dict(spec) for spec in specs],
        "source_spec_count": len(specs),
        "profile_contract": {
            "lookback_hours": list(PROFILE_LOOKBACKS),
            "bin_modes": list(PROFILE_BIN_MODES),
            "recalculation_cadence_hours": {
                str(lookback): cadence
                for lookback, cadence in PROFILE_CADENCE.items()
            },
            "price_sample": "hourly typical price",
            "weight": "hourly quote volume proxy from local OHLCV volume",
            "value_area_fraction": 0.70,
            "hvn_quantiles": [0.80, 0.90],
            "lvn_quantiles": [0.10, 0.20],
            "adaptive_bins": "Freedman-Diaconis clipped to 24-96",
        },
        "controls": list(CONTROLS),
        "zone_methods": list(ZONE_METHODS),
        "horizons_hours": list(HORIZONS),
        "metrics": list(METRICS),
        "parameter_selection_used_event_outcomes": False,
        "future_signed_direction_used": False,
        "profit_used": False,
        "source_contracts": {
            "generation17_freeze": artifact(g17z.FREEZE_PATH),
            "normal_manifest": artifact(COHORT_MANIFESTS["normal"]),
            "meme_manifest": artifact(COHORT_MANIFESTS["meme"]),
        },
        "pass_rule": next(
            item["pass_rule"]
            for item in frozen["branches"]
            if item["branch_id"] == "g17d_broad_level_sources"
        ),
    }
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    if SOURCE_REGISTRY_PATH.is_file():
        existing = json.loads(SOURCE_REGISTRY_PATH.read_text(encoding="utf-8"))
        left = {key: value for key, value in existing.items() if key != "created_at_utc"}
        right = {key: value for key, value in registry.items() if key != "created_at_utc"}
        if left != right:
            raise ValueError("Generation 17 source registry changed after its freeze.")
        return existing
    g0.atomic_write_json(registry, SOURCE_REGISTRY_PATH)
    return registry


def nearest(candidates: np.ndarray, reference: float) -> float:
    finite = candidates[np.isfinite(candidates)]
    if not len(finite):
        return np.nan
    return float(finite[np.argmin(np.abs(finite - reference))])


def fd_bin_count(prices: np.ndarray) -> int:
    if len(prices) < 4:
        return 24
    spread = float(np.nanmax(prices) - np.nanmin(prices))
    iqr = float(np.nanquantile(prices, 0.75) - np.nanquantile(prices, 0.25))
    width = 2.0 * iqr * len(prices) ** (-1.0 / 3.0)
    if not np.isfinite(width) or width <= 0.0 or spread <= 0.0:
        return 48
    return int(np.clip(np.ceil(spread / width), 24, 96))


def profile_snapshot(
    prices: np.ndarray,
    weights: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    reference: float,
    bins_mode: int | str,
) -> dict[str, float]:
    valid = (
        np.isfinite(prices)
        & np.isfinite(weights)
        & np.isfinite(lows)
        & np.isfinite(highs)
        & (weights >= 0.0)
    )
    prices = prices[valid]
    weights = weights[valid]
    lows = lows[valid]
    highs = highs[valid]
    if len(prices) < 24 or not np.isfinite(reference):
        return {role: np.nan for role in PROFILE_ROLES}
    low = float(np.nanmin(lows))
    high = float(np.nanmax(highs))
    if not high > low:
        return {role: np.nan for role in PROFILE_ROLES}
    bins = fd_bin_count(prices) if bins_mode == "fd" else int(bins_mode)
    edges = np.linspace(low, high, bins + 1)
    histogram, _ = np.histogram(prices, bins=edges, weights=weights)
    centers = (edges[:-1] + edges[1:]) / 2.0
    if not np.isfinite(histogram).all() or float(histogram.sum()) <= 0.0:
        return {role: np.nan for role in PROFILE_ROLES}
    poc = float(centers[int(np.argmax(histogram))])
    order = np.argsort(histogram)[::-1]
    cumulative = np.cumsum(histogram[order])
    count = int(np.searchsorted(cumulative, 0.70 * histogram.sum(), side="left")) + 1
    value_centers = centers[order[:count]]
    value_boundary = nearest(
        np.array([float(value_centers.min()), float(value_centers.max())]), reference
    )

    def quantile_node(quantile: float, *, high_volume: bool) -> float:
        threshold = float(np.quantile(histogram, quantile))
        mask = histogram >= threshold if high_volume else histogram <= threshold
        return nearest(centers[mask], reference)

    return {
        "poc": poc,
        "nearest_value_boundary": value_boundary,
        "nearest_hvn_q80": quantile_node(0.80, high_volume=True),
        "nearest_hvn_q90": quantile_node(0.90, high_volume=True),
        "nearest_lvn_q10": quantile_node(0.10, high_volume=False),
        "nearest_lvn_q20": quantile_node(0.20, high_volume=False),
    }


def add_volume_profiles(frame: DataFrame) -> DataFrame:
    prices = ((frame["high"] + frame["low"] + frame["close"]) / 3.0).to_numpy(
        dtype=float
    )
    weights = pd.to_numeric(frame["volume"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    highs = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    references = pd.to_numeric(frame["pre_close"], errors="coerce").to_numpy(dtype=float)
    profile_columns: dict[str, Series] = {}
    for lookback in PROFILE_LOOKBACKS:
        cadence = PROFILE_CADENCE[lookback]
        for bins in PROFILE_BIN_MODES:
            outputs = {role: np.full(len(frame), np.nan) for role in PROFILE_ROLES}
            for index in range(lookback, len(frame), cadence):
                result = profile_snapshot(
                    prices[index - lookback : index],
                    weights[index - lookback : index],
                    lows[index - lookback : index],
                    highs[index - lookback : index],
                    references[index],
                    bins,
                )
                for role, value in result.items():
                    outputs[role][index] = value
            for role, values in outputs.items():
                profile_columns[f"level__{vp_name(lookback, bins, role)}"] = Series(
                    values, index=frame.index
                ).ffill()
    return pd.concat([frame, DataFrame(profile_columns, index=frame.index)], axis=1)


def add_vwap_donchian_and_generic(frame: DataFrame) -> None:
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    typical = (high + low + close) / 3.0
    for lookback in (24, 72, 168, 720):
        sum_volume = volume.rolling(lookback, min_periods=lookback).sum()
        mean = (typical * volume).rolling(lookback, min_periods=lookback).sum() / sum_volume
        second = (
            (typical.pow(2) * volume).rolling(lookback, min_periods=lookback).sum()
            / sum_volume
        )
        deviation = (second - mean.pow(2)).clip(lower=0.0).pow(0.5)
        mean = mean.shift(1)
        deviation = deviation.shift(1)
        frame[f"level__rolling_vwap_lb{lookback}_vwap"] = mean
        for multiple in (1, 2):
            frame[f"level__rolling_vwap_lb{lookback}_upper_{multiple}sd"] = (
                mean + multiple * deviation
            )
            frame[f"level__rolling_vwap_lb{lookback}_lower_{multiple}sd"] = (
                mean - multiple * deviation
            )
        frame[f"level__donchian_lb{lookback}_upper"] = (
            high.rolling(lookback, min_periods=lookback).max().shift(1)
        )
        frame[f"level__donchian_lb{lookback}_lower"] = (
            low.rolling(lookback, min_periods=lookback).min().shift(1)
        )
    frame["level__negative_sma20"] = close.rolling(20, min_periods=20).mean().shift(1)
    frame["level__negative_sma50"] = close.rolling(50, min_periods=50).mean().shift(1)
    frame["level__negative_sma200"] = close.rolling(200, min_periods=200).mean().shift(1)
    frame["level__negative_ema20"] = close.ewm(span=20, adjust=False).mean().shift(1)
    frame["level__negative_ema50"] = close.ewm(span=50, adjust=False).mean().shift(1)
    middle = close.rolling(20, min_periods=20).mean().shift(1)
    deviation = close.rolling(20, min_periods=20).std(ddof=0).shift(1)
    frame["level__negative_bollinger_mid20"] = middle
    frame["level__negative_bollinger_upper20"] = middle + 2.0 * deviation
    frame["level__negative_bollinger_lower20"] = middle - 2.0 * deviation


def add_weekly_pivots(frame: DataFrame) -> None:
    indexed = frame.set_index("date")
    weekly = indexed.resample("W-MON", label="left", closed="left").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last")
    )
    prior = weekly.shift(1)
    pivot = (prior["high"] + prior["low"] + prior["close"]) / 3.0
    levels = DataFrame(index=weekly.index)
    levels["level__weekly_pivot"] = pivot
    levels["level__weekly_r1"] = 2.0 * pivot - prior["low"]
    levels["level__weekly_s1"] = 2.0 * pivot - prior["high"]
    levels["level__weekly_r2"] = pivot + prior["high"] - prior["low"]
    levels["level__weekly_s2"] = pivot - prior["high"] + prior["low"]
    lookup = pd.DataFrame({"date": frame["date"]}).sort_values("date")
    expanded = pd.merge_asof(
        lookup,
        levels.reset_index().sort_values("date"),
        on="date",
        direction="backward",
    )
    for column in levels:
        frame[column] = expanded[column].to_numpy()


def level_surface(pair: str, manifest: dict[str, Any]) -> DataFrame:
    frame = g0.prepare_base_market_frame(pair, manifest)
    add_vwap_donchian_and_generic(frame)
    add_weekly_pivots(frame)
    frame = add_volume_profiles(frame)
    frame["available_at"] = frame["date"]
    frame["source_open"] = frame["date"]
    frame["source_atr_14"] = frame["base_atr"]
    return frame


def add_cluster_metadata(events: DataFrame) -> DataFrame:
    keys = ["cohort", "pair", "control", "event_time"]
    groups = events.groupby(keys, observed=True, sort=False)
    counts = groups.agg(
        independent_family_count=("level_family", "nunique"),
        convergent_variant_count=("level_name", "nunique"),
    ).reset_index()
    patterns = (
        groups["level_family"]
        .agg(lambda values: "+".join(sorted(set(map(str, values)))))
        .rename("family_pattern")
        .reset_index()
    )
    return events.merge(counts, on=keys, validate="many_to_one").merge(
        patterns, on=keys, validate="many_to_one"
    )


def build_pair(task: PairTask) -> dict[str, Any]:
    output_dir = Path(task.output_dir)
    stem = f"{task.cohort}__{g0.pair_file_stem(task.pair)}"
    event_path = output_dir / "pair_events" / f"{stem}.parquet"
    audit_path = output_dir / "pair_audits" / f"{stem}.json"
    if event_path.is_file() and audit_path.is_file() and not task.overwrite:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        return {**audit, "status": "existing"}
    manifest_path = Path(task.manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame = level_surface(task.pair, manifest)
    specs = source_specs()
    missing = sorted(spec.column for spec in specs if spec.column not in frame.columns)
    if missing:
        raise ValueError(f"Missing Generation 17 level columns: {missing}")
    finite_counts = {
        spec.name: int(pd.to_numeric(frame[spec.column], errors="coerce").notna().sum())
        for spec in specs
    }
    source_freeze = {
        "schema_version": 1,
        "generation": 17,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_pair_level_surface_before_outcomes",
        "cohort": task.cohort,
        "pair": task.pair,
        "rows": len(frame),
        "finite_counts": finite_counts,
        "source_registry": artifact(SOURCE_REGISTRY_PATH),
        "ohlcv": artifact(g0.ohlcv_path(task.pair, manifest["data"]["base_timeframe"])),
        "future_outcomes_opened": False,
    }
    pair_freeze_path = output_dir / "pair_source_freezes" / f"{stem}.json"
    g0.atomic_write_json(source_freeze, pair_freeze_path)
    paths = g0.future_path_matrices(frame, max(HORIZONS))
    outputs: list[DataFrame] = []
    for spec in specs:
        outputs.extend(
            g0.build_spec_events(
                merged=frame,
                paths=paths,
                spec=spec,
                pair=task.pair,
                timeframe="1h",
                zone_methods=ZONE_METHODS,
                controls=CONTROLS,
                horizons=HORIZONS,
            )
        )
    if not outputs:
        raise ValueError(f"No rational level events for {task.pair}.")
    raw_events = pd.concat(outputs, ignore_index=True, sort=False)
    raw_events["cohort"] = task.cohort
    events = add_cluster_metadata(raw_events)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(events, event_path)
    audit = {
        "status": "built",
        "cohort": task.cohort,
        "pair": task.pair,
        "level_specs": len(specs),
        "events": len(events),
        "actual_events": int(events["control"].eq("actual").sum()),
        "event_path": str(event_path.resolve()),
        "event_sha256": g0.sha256_file(event_path),
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
                        "phase": "g17_level_source_pair",
                        "processed": index,
                        "total": len(tasks),
                        "pair": result.get("pair"),
                        "status": result.get("status"),
                    }
                ),
                flush=True,
            )
    return sorted(results, key=lambda item: (item.get("cohort", ""), item.get("pair", "")))


def metric_values(frame: DataFrame, metric: str, horizon: int) -> Series:
    if metric == "volume_ratio":
        return pd.to_numeric(frame[f"volume_ratio_h{horizon}"], errors="coerce")
    if metric == "crossings":
        return pd.to_numeric(frame[f"crossings_h{horizon}"], errors="coerce")
    threshold = pd.to_numeric(frame["zone_half_width_atr"], errors="coerce").clip(
        lower=0.5
    )
    if metric == "unsigned_reaction":
        excursion = pd.to_numeric(
            frame[f"abs_excursion_atr_h{horizon}"], errors="coerce"
        )
        volume = pd.to_numeric(frame[f"volume_ratio_h{horizon}"], errors="coerce")
        output = ((excursion >= threshold) & (volume >= 1.25)).astype(float)
        output.loc[excursion.isna() | volume.isna() | threshold.isna()] = np.nan
        return output
    if metric == "two_sided_traversal":
        away = pd.to_numeric(frame[f"away_excursion_atr_h{horizon}"], errors="coerce")
        through = pd.to_numeric(
            frame[f"through_excursion_atr_h{horizon}"], errors="coerce"
        )
        output = ((away >= threshold) & (through >= threshold)).astype(float)
        output.loc[away.isna() | through.isna() | threshold.isna()] = np.nan
        return output
    raise KeyError(metric)


def scoped_events(events: DataFrame) -> DataFrame:
    frames: list[DataFrame] = []
    singles = events.copy()
    singles["scope_kind"] = "single_level"
    singles["scope_value"] = singles["level_name"]
    frames.append(singles)
    family = events.sort_values("level_name").drop_duplicates(
        ["cohort", "pair", "control", "event_time", "level_family"]
    )
    family["scope_kind"] = "family_any_level"
    family["scope_value"] = family["level_family"]
    frames.append(family)
    time_rows = events.sort_values("level_name").drop_duplicates(
        ["cohort", "pair", "control", "event_time"]
    )
    single_family = time_rows.loc[time_rows["independent_family_count"].eq(1)].copy()
    single_family["scope_kind"] = "single_independent_family_contact"
    single_family["scope_value"] = "exactly_one_independent_family"
    frames.append(single_family)
    clusters = time_rows.loc[time_rows["independent_family_count"].ge(2)].copy()
    clusters["scope_kind"] = "cross_family_cluster"
    clusters["scope_value"] = "two_or_more_independent_families"
    frames.append(clusters)
    patterns = time_rows.loc[time_rows["independent_family_count"].ge(2)].copy()
    patterns["scope_kind"] = "repeated_cluster_pattern"
    patterns["scope_value"] = patterns["family_pattern"]
    frames.append(patterns)
    return pd.concat(frames, ignore_index=True, sort=False)


def pair_summary_columns() -> list[str]:
    outcome_columns = [
        column
        for horizon in HORIZONS
        for column in (
            f"abs_excursion_atr_h{horizon}",
            f"away_excursion_atr_h{horizon}",
            f"through_excursion_atr_h{horizon}",
            f"volume_ratio_h{horizon}",
            f"crossings_h{horizon}",
        )
    ]
    return [
        "cohort",
        "pair",
        "period",
        "control",
        "event_time",
        "level_family",
        "level_name",
        "zone_half_width_atr",
        "independent_family_count",
        "family_pattern",
        *outcome_columns,
    ]


def pair_summaries(events: DataFrame) -> DataFrame:
    events = events[pair_summary_columns()].copy()
    scoped = scoped_events(events)
    rows: list[DataFrame] = []
    keys = [
        "cohort",
        "pair",
        "period",
        "scope_kind",
        "scope_value",
        "control",
    ]
    for horizon in HORIZONS:
        for metric in METRICS:
            selected = scoped[keys].copy()
            selected["value"] = metric_values(scoped, metric, horizon)
            grouped = (
                selected.dropna(subset=["value"])
                .groupby(keys, observed=True, sort=False)["value"]
                .agg(["mean", "size"])
                .reset_index()
                .rename(columns={"mean": "outcome_mean", "size": "event_rows"})
            )
            grouped["metric"] = metric
            grouped["horizon_hours"] = horizon
            rows.append(grouped)
    return pd.concat(rows, ignore_index=True, sort=False)


def paired_contrasts(summary: DataFrame) -> DataFrame:
    keys = [
        "cohort",
        "pair",
        "period",
        "scope_kind",
        "scope_value",
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


def bootstrap_coins(values: np.ndarray, key: str) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True).mean(
        axis=1
    )
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def expand_market_scopes(frame: DataFrame) -> DataFrame:
    rows: list[DataFrame] = []
    for (cohort, pair), item in frame.groupby(["cohort", "pair"], observed=True, sort=False):
        for market_scope in g16d.group_for_pair(str(cohort), str(pair)):
            copy = item.copy()
            copy["market_scope"] = market_scope
            rows.append(copy)
    return pd.concat(rows, ignore_index=True, sort=False)


def period_scores(contrasts: DataFrame) -> DataFrame:
    expanded = expand_market_scopes(contrasts.loc[contrasts["eligible_pair"]].copy())
    keys = [
        "scope_kind",
        "scope_value",
        "metric",
        "horizon_hours",
        "market_scope",
        "period",
        "comparison",
    ]
    rows: list[dict[str, Any]] = []
    for key, cell in expanded.groupby(keys, observed=True, sort=False):
        values = pd.to_numeric(cell["difference"], errors="coerce").dropna()
        lower, upper = bootstrap_coins(values.to_numpy(), "|".join(map(str, key)))
        absolute = values.abs()
        dominance = (
            float(absolute.max() / absolute.sum())
            if len(absolute) and absolute.sum() > 0.0
            else np.nan
        )
        required = g16d.GROUP_MIN_COINS[str(key[4])]
        fraction = float(values.gt(0.0).mean()) if len(values) else np.nan
        support = len(values) >= required
        point = bool(support and values.mean() > 0.0 and fraction >= 0.70)
        strict = bool(
            point
            and np.isfinite(lower)
            and lower > 0.0
            and (
                str(key[4]) == "btc_separate"
                or (np.isfinite(dominance) and dominance <= g16d.MAX_COIN_ABSOLUTE_SHARE)
            )
        )
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "eligible_coins": len(values),
                "required_coins": required,
                "positive_coins": int(values.gt(0.0).sum()),
                "positive_coin_fraction": fraction,
                "equal_coin_difference": float(values.mean()) if len(values) else np.nan,
                "bootstrap_lower_95": lower,
                "bootstrap_upper_95": upper,
                "maximum_one_coin_absolute_share": dominance,
                "point_period_pass": point,
                "strict_period_pass": strict,
            }
        )
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def whole_decisions(scores: DataFrame) -> DataFrame:
    keys = ["scope_kind", "scope_value", "metric", "horizon_hours", "market_scope"]
    rows: list[dict[str, Any]] = []
    for key, cell in scores.groupby(keys, observed=True, sort=False):
        expected_periods = (
            {"meme_validation_early", "meme_validation_late"}
            if key[4] == "top_ten_memes"
            else {"validation_early", "validation_late"}
        )
        selected = cell.loc[cell["period"].isin(expected_periods)]
        complete = bool(
            len(selected) == len(expected_periods) * len(CONTROL_COMPARISONS)
            and set(selected["period"]) == expected_periods
            and set(selected["comparison"]) == set(CONTROL_COMPARISONS)
        )
        point = bool(complete and selected["point_period_pass"].astype(bool).all())
        strict = bool(complete and selected["strict_period_pass"].astype(bool).all())
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "status": (
                    "strict_repeated"
                    if strict
                    else "point_repeated"
                    if point
                    else "not_retained"
                ),
                "complete_control_period_ladder": complete,
                "point_repeated": point,
                "strict_repeated": strict,
                "minimum_equal_coin_difference": float(selected["equal_coin_difference"].min())
                if len(selected)
                else np.nan,
                "minimum_bootstrap_lower": float(selected["bootstrap_lower_95"].min())
                if len(selected)
                else np.nan,
            }
        )
    decisions = DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)
    portable_keys = ["scope_kind", "scope_value", "metric", "horizon_hours"]
    portable: dict[tuple[Any, ...], bool] = {}
    for key, cell in decisions.groupby(portable_keys, observed=True, sort=False):
        required = cell.loc[cell["market_scope"].isin(["all_normal", "top_ten_memes"])]
        portable[key] = bool(
            set(required["market_scope"]) == {"all_normal", "top_ten_memes"}
            and required["strict_repeated"].astype(bool).all()
        )
    decisions["portable_normal_and_meme_strict"] = [
        portable[tuple(row[column] for column in portable_keys)]
        for _, row in decisions.iterrows()
    ]
    return decisions


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    registry = freeze_source_registry()
    run_dir = RECORD_ROOT / args.run_id
    artifact_dir = ARTIFACT_ROOT / args.run_id
    result_path = run_dir / "g17_level_source_atlas_result.json"
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
        raise RuntimeError(f"Generation 17 level-source failures: {failures}")
    summary_parts: list[DataFrame] = []
    for item in inventory:
        summary_parts.append(
            pair_summaries(
                pd.read_parquet(
                    item["event_path"], columns=pair_summary_columns()
                )
            )
        )
    pair_summary = pd.concat(summary_parts, ignore_index=True, sort=False)
    contrasts = paired_contrasts(pair_summary)
    scores = period_scores(contrasts)
    decisions = whole_decisions(scores)
    paths = {
        "inventory": run_dir / "pair_inventory.csv",
        "pair_summary": artifact_dir / "pair_level_summary.parquet",
        "pair_contrasts": artifact_dir / "pair_level_contrasts.parquet",
        "period_scores": run_dir / "level_source_period_scores.csv",
        "whole_decisions": run_dir / "level_source_whole_decisions.csv",
    }
    g0.atomic_write_csv(DataFrame.from_records(inventory), paths["inventory"])
    g0.atomic_write_parquet(pair_summary, paths["pair_summary"])
    g0.atomic_write_parquet(contrasts, paths["pair_contrasts"])
    g0.atomic_write_csv(scores, paths["period_scores"])
    g0.atomic_write_csv(decisions, paths["whole_decisions"])
    result = {
        "schema_version": 1,
        "generation": 17,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation17_level_source_atlas",
        "branch_id": "g17d_broad_level_sources",
        "pairs_completed": len(inventory),
        "level_specs": registry["source_spec_count"],
        "events": int(sum(int(item["events"]) for item in inventory)),
        "strict_repeated": int(decisions["strict_repeated"].sum()),
        "point_repeated": int(decisions["point_repeated"].sum()),
        "portable_normal_and_meme_strict": int(
            decisions["portable_normal_and_meme_strict"].sum()
        ),
        "single_and_cluster_scopes_both_tested": True,
        "research_boundary": {
            "profit_used": False,
            "future_signed_direction_used": False,
            "parameter_selection_used_outcomes": False,
        },
        "source_contracts": {
            "generation17_freeze": artifact(g17z.FREEZE_PATH),
            "source_registry": artifact(SOURCE_REGISTRY_PATH),
        },
        "artifacts": {key: artifact(path) for key, path in paths.items()},
    }
    g0.atomic_write_json(result, result_path)
    print(json.dumps({**result, "result_path": str(result_path.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

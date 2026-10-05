from __future__ import annotations

# Bound numerical libraries before pandas/FreqAI helpers are imported.
# ruff: noqa: E402
import os


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.market_reaction_zone_freqai_generation0 import (  # noqa: E501
    common_prediction_keys,
    eligibility_digest,
    load_predictions,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_freqai_generation1 import (  # noqa: E501
    run_profile as run_freqai_profile,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_freqai_generation4 import (  # noqa: E501
    MIN_SCORABLE_ROWS,
    deterministic_shift,
    regression_metrics,
    runtime_snapshot,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_freqai_generation5 import (  # noqa: E501
    BLOCK_BOOTSTRAP_SAMPLES,
    BLOCK_DAYS,
    deterministic_block_bootstrap,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    atomic_write_json,
    atomic_write_parquet,
    normalize_dates,
    sha256_file,
    utc_now,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation5_new_period_preflight import (  # noqa: E501
    validate_frozen_branch,
)
from user_data.strategies.MarketReactionZoneFreqAIGeneration5NewPeriodStrategy import (
    MAX_TARGET_HORIZON_HOURS,
    TARGET_COLUMN,
)


USER_DATA_DIR = REPO_ROOT / "user_data"
DEFAULT_CONFIG = USER_DATA_DIR / "configs" / "config_market_reaction_zone_freqai.example.json"
DEFAULT_PYTHON = (
    REPO_ROOT / "runtime" / "venvs" / "freqtrade-backtest-08" / "Scripts" / "python.exe"
)
STRATEGY_PATH = USER_DATA_DIR / "strategies"
STRATEGY_FILE = STRATEGY_PATH / "MarketReactionZoneFreqAIGeneration5NewPeriodStrategy.py"
DATA_DIR = USER_DATA_DIR / "data" / "binance"
OUTPUT_ROOT = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "market_reaction_zones"
)
FROZEN_BATCH = OUTPUT_ROOT / "generation4_review" / "g5_frozen_branch_batch.json"
RECORD_ROOT = (
    OUTPUT_ROOT
    / "generation5_branches"
    / "g5c_new_period_normal_excursion_confirmation"
)
LARGE_ROOT = Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
ARTIFACT_ROOT = (
    LARGE_ROOT
    / "generation5_branches"
    / "g5c_new_period_normal_excursion_confirmation"
)
G4D_RUN_ID = "g4d_full_g3d_normal_cluster_20260820b_time_baseline_repair"
G4D_RECORD_ROOT = (
    OUTPUT_ROOT / "generation4_branches" / "g4d_freqai_reaction_ablation"
)
G4D_ARTIFACT_ROOT = (
    LARGE_ROOT / "generation4_branches" / "g4d_freqai_reaction_ablation"
)
FRESH_PREFLIGHT_RUN_ID = "g5c_fresh_preflight_20260821a"
FRESH_PREFLIGHT_DIR = RECORD_ROOT / FRESH_PREFLIGHT_RUN_ID
FRESH_PREFLIGHT_RECORD = FRESH_PREFLIGHT_DIR / "g5c_event_preflight_record.json"
FRESH_SURFACE_ROOT = (
    LARGE_ROOT
    / "generation5_branches"
    / "g5c_new_period_normal_excursion_confirmation"
    / FRESH_PREFLIGHT_RUN_ID
    / "fresh_event_surfaces"
)

SURFACE_ID = "normal_confirmed_swing_density_cluster"
G4D_SURFACE_ID = "g3d_normal_density_cluster"
OLD_TARGET_COLUMN = "&-g4d_one_hour_absolute_excursion_in_prior_atr"
CONFIRMATION_START = pd.Timestamp("2026-07-20T00:00:00Z")
CONFIRMATION_END = pd.Timestamp("2026-08-20T00:00:00Z")
TIMERANGE = "20260720-20260820"
TRAIN_DAYS = 730
BACKTEST_DAYS = 31
CONFIRMATION_PERIODS = ("g5c_confirmation_early", "g5c_confirmation_late")
BRIDGE_PERIOD = "g5c_exposed_training_bridge"

MAX_WORKERS = 4
MIN_RETAIN_COINS = 5
OUTPUT_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1

WIDE_COLUMNS = (
    "wide__btc_return_1h",
    "wide__btc_return_24h",
    "wide__btc_atr_pct",
    "wide__top10_breadth",
    "wide__top10_mean_abs_return",
    "wide__top10_return_dispersion",
)
CURRENT_COLUMNS = (
    "level__pre_distance_atr",
    "level__zone_support_fraction",
    "level__zone_half_width_atr",
    "geometry__other_density_zone_count",
    "mtf__reference_level_count",
    "mtf__overlap_reference_level_count",
    "level__arrival_first_arrival_after_outside_interval",
    "level__arrival_near_miss",
    "level__arrival_already_inside",
    "level__arrival_repeat_contact",
    "level__scope_single_density_zone",
    "level__scope_density_cluster",
    "interaction__fresh_arrival_x_support_fraction",
    "interaction__support_fraction_x_reference_count",
)


def control_column(prefix: str, column: str) -> str:
    return f"{prefix}__{column.replace('__', '_', 1)}"


SHUFFLE_COLUMNS = tuple(control_column("shuffle", column) for column in CURRENT_COLUMNS)
STALE_COLUMNS = tuple(control_column("stale", column) for column in CURRENT_COLUMNS)

PROFILES: dict[str, dict[str, str]] = {
    "state_only": {
        "strategy": "MarketReactionZoneG5CStateOnlyFreqAIResearchStrategy",
        "role": "Prior-candle OHLCV, generic indicators, BTC, and cohort state only.",
    },
    "current": {
        "strategy": "MarketReactionZoneG5CCurrentFreqAIResearchStrategy",
        "role": "State plus the current causal density level and local geometry.",
    },
    "shuffled": {
        "strategy": "MarketReactionZoneG5CShuffledFreqAIResearchStrategy",
        "role": "State plus non-self density attributes shuffled within pair-period.",
    },
    "stale": {
        "strategy": "MarketReactionZoneG5CStaleFreqAIResearchStrategy",
        "role": "State plus density attributes from the prior event in the same period.",
    },
}

COMPARISONS = (
    ("current_vs_state", "current", "state_only"),
    ("current_vs_shuffled", "current", "shuffled"),
    ("current_vs_stale", "current", "stale"),
)


def parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def pair_stem(pair: str) -> str:
    return pair.strip().upper().replace("/", "_").replace(":", "_")


def g4d_source_contract() -> dict[str, Any]:
    record_dir = G4D_RECORD_ROOT / G4D_RUN_ID
    artifact_dir = G4D_ARTIFACT_ROOT / G4D_RUN_ID
    manifest_path = record_dir / "manifest.json"
    result_path = record_dir / "g4d_result.json"
    if not manifest_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(manifest_path if not manifest_path.is_file() else result_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("surface_id") != G4D_SURFACE_ID:
        raise ValueError("The historical G4D density-cluster source is not complete and fixed.")
    if result.get("surface_id") != G4D_SURFACE_ID:
        raise ValueError("The historical G4D result surface changed.")
    return {
        "record_dir": record_dir,
        "artifact_dir": artifact_dir,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "result_path": result_path,
        "result_sha256": sha256_file(result_path),
        "pairs": tuple(str(pair) for pair in manifest["pairs"]),
    }


def fresh_preflight_contract() -> dict[str, Any]:
    if not FRESH_PREFLIGHT_RECORD.is_file():
        raise FileNotFoundError(FRESH_PREFLIGHT_RECORD)
    record = json.loads(FRESH_PREFLIGHT_RECORD.read_text(encoding="utf-8"))
    if record.get("status") not in {
        "coverage_gate_partially_passed",
        "coverage_gate_passed",
    }:
        raise ValueError("The G5C fresh event preflight did not pass a coverage gate.")
    if not record.get("surface_coverage_gate", {}).get(SURFACE_ID, False):
        raise ValueError("The fixed density-cluster surface did not pass fresh coverage.")
    if record.get("target_values_compared_or_interpreted") is not False:
        raise ValueError("The coverage gate was not outcome-blind.")
    coverage_path = Path(str(record["coverage_path"]))
    inventory_path = Path(str(record["inventory_path"]))
    for path in (coverage_path, inventory_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "record": record,
        "record_sha256": sha256_file(FRESH_PREFLIGHT_RECORD),
        "coverage_path": coverage_path,
        "coverage_sha256": sha256_file(coverage_path),
        "inventory_path": inventory_path,
        "inventory_sha256": sha256_file(inventory_path),
    }


def attach_controls(events: DataFrame, *, pair: str) -> tuple[DataFrame, list[dict[str, Any]]]:
    output = events.sort_values("date", kind="stable").reset_index(drop=True).copy()
    for column in (*SHUFFLE_COLUMNS, *STALE_COLUMNS):
        output[column] = np.nan
    audits: list[dict[str, Any]] = []
    for period, positions in output.groupby("period", sort=False).groups.items():
        indexes = np.asarray(list(positions), dtype=np.int64)
        if len(indexes) < 2:
            audits.append(
                {
                    "pair": pair,
                    "period": str(period),
                    "rows": len(indexes),
                    "status": "too_small_for_nonself_control",
                }
            )
            continue
        shift = deterministic_shift(len(indexes), f"g5c|{SURFACE_ID}|{pair}|{period}")
        shuffled_source = np.roll(indexes, shift)
        if int(np.sum(shuffled_source == indexes)):
            raise AssertionError("G5C shuffled control retained self assignments.")
        for source, shuffled, stale in zip(
            CURRENT_COLUMNS,
            SHUFFLE_COLUMNS,
            STALE_COLUMNS,
            strict=True,
        ):
            output.loc[indexes, shuffled] = output.loc[shuffled_source, source].to_numpy()
            output.loc[indexes[1:], stale] = output.loc[indexes[:-1], source].to_numpy()
        audits.append(
            {
                "pair": pair,
                "period": str(period),
                "rows": len(indexes),
                "shuffle": shift,
                "shuffle_self_assignments": 0,
                "stale_missing_first_row": 1,
                "stale_future_source_violations": 0,
                "status": "completed",
            }
        )
    return output, audits


def build_pair_cache(
    pair: str,
    *,
    source: dict[str, Any],
    feature_dir: Path,
    event_dir: Path,
) -> dict[str, Any]:
    stem = pair_stem(pair)
    old_feature_path = source["artifact_dir"] / "feature_cache" / f"{stem}.parquet"
    old_event_path = source["artifact_dir"] / "event_cache" / f"{stem}.parquet"
    fresh_path = FRESH_SURFACE_ROOT / SURFACE_ID / f"{stem}.parquet"
    for path in (old_feature_path, old_event_path, fresh_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    old_features = pd.read_parquet(
        old_feature_path,
        columns=["date", *WIDE_COLUMNS, *CURRENT_COLUMNS],
    )
    old_features["date"] = normalize_dates(old_features["date"])
    old_events = pd.read_parquet(old_event_path)
    old_events["date"] = normalize_dates(old_events["date"])
    old_events["source_available_at"] = normalize_dates(old_events["source_available_at"])
    historical = old_events.merge(
        old_features,
        on="date",
        how="left",
        validate="one_to_one",
    ).rename(columns={OLD_TARGET_COLUMN: TARGET_COLUMN})
    historical = historical.loc[historical["date"].lt(pd.Timestamp("2026-04-01T00:00:00Z"))]

    fresh = pd.read_parquet(fresh_path)
    fresh["date"] = normalize_dates(fresh["date"])
    fresh["source_available_at"] = normalize_dates(fresh["source_available_at"])
    fresh = fresh.loc[
        fresh["period"].isin((BRIDGE_PERIOD, *CONFIRMATION_PERIODS))
    ].copy()
    keep = [
        "date",
        "period",
        "event_state",
        "level_identity",
        "approach_state",
        "source_available_at",
        *WIDE_COLUMNS,
        *CURRENT_COLUMNS,
        TARGET_COLUMN,
    ]
    events = pd.concat([historical[keep], fresh[keep]], ignore_index=True)
    events = events.sort_values("date", kind="stable").reset_index(drop=True)
    if events["date"].duplicated().any():
        raise ValueError(f"G5C combined event dates are not unique for {pair}.")
    if events["source_available_at"].isna().any() or events["source_available_at"].gt(
        events["date"]
    ).any():
        raise ValueError(f"G5C source availability is not causal for {pair}.")

    events, placebo_audit = attach_controls(events, pair=pair)
    required = [
        *WIDE_COLUMNS,
        *CURRENT_COLUMNS,
        *SHUFFLE_COLUMNS,
        *STALE_COLUMNS,
        TARGET_COLUMN,
    ]
    numeric = events[required].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    complete = numeric.notna().all(axis=1)
    incomplete_rows = int((~complete).sum())
    events = events.loc[complete].reset_index(drop=True)
    exposed = events.loc[events["date"].lt(CONFIRMATION_START)]
    if exposed.empty or exposed["date"].max() >= (
        CONFIRMATION_START - pd.Timedelta(hours=MAX_TARGET_HORIZON_HOURS)
    ):
        raise ValueError(f"G5C {pair} violates the one-hour train-boundary purge.")

    feature_cache = events[
        ["date", *WIDE_COLUMNS, *CURRENT_COLUMNS, *SHUFFLE_COLUMNS, *STALE_COLUMNS]
    ].copy()
    event_cache = events[
        [
            "date",
            "period",
            "event_state",
            "level_identity",
            "approach_state",
            "source_available_at",
            TARGET_COLUMN,
        ]
    ].copy()
    feature_path = feature_dir / f"{stem}.parquet"
    event_path = event_dir / f"{stem}.parquet"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(feature_cache, feature_path)
    atomic_write_parquet(event_cache, event_path)
    return {
        "pair": pair,
        "status": "completed",
        "historical_event_rows": len(historical),
        "fresh_event_rows": len(fresh),
        "complete_identical_profile_rows": len(event_cache),
        "incomplete_control_rows_dropped": incomplete_rows,
        "period_rows": event_cache.groupby("period").size().to_dict(),
        "feature_path": str(feature_path),
        "feature_sha256": sha256_file(feature_path),
        "event_path": str(event_path),
        "event_sha256": sha256_file(event_path),
        "event_key_digest": eligibility_digest(event_cache.assign(pair=pair)[["pair", "date"]]),
        "old_feature_path": str(old_feature_path),
        "old_feature_sha256": sha256_file(old_feature_path),
        "old_event_path": str(old_event_path),
        "old_event_sha256": sha256_file(old_event_path),
        "fresh_event_path": str(fresh_path),
        "fresh_event_sha256": sha256_file(fresh_path),
        "placebo_audit": placebo_audit,
        "target_values_used_for_feature_selection": False,
        "direction_prediction": False,
        "profit_optimization": False,
    }


def build_caches(
    *,
    source: dict[str, Any],
    pairs: Sequence[str],
    feature_dir: Path,
    event_dir: Path,
    workers: int,
) -> list[dict[str, Any]]:
    def safe(pair: str) -> dict[str, Any]:
        try:
            return build_pair_cache(
                pair,
                source=source,
                feature_dir=feature_dir,
                event_dir=event_dir,
            )
        except Exception as exc:
            return {"pair": pair, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    if workers == 1:
        return [safe(pair) for pair in pairs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(safe, pair): pair for pair in pairs}
        results = [future.result() for future in as_completed(futures)]
    return sorted(results, key=lambda row: row["pair"])


def post_control_coverage(cache_results: Sequence[dict[str, Any]]) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for period in CONFIRMATION_PERIODS:
        pair_rows = {
            str(item["pair"]): int(item.get("period_rows", {}).get(period, 0))
            for item in cache_results
        }
        primary = {pair: count for pair, count in pair_rows.items() if not pair.startswith("BTC/")}
        btc = {pair: count for pair, count in pair_rows.items() if pair.startswith("BTC/")}
        primary_rows = int(sum(primary.values()))
        primary_coins = int(sum(value > 0 for value in primary.values()))
        rows.append(
            {
                "surface_id": SURFACE_ID,
                "period": period,
                "primary_rows": primary_rows,
                "primary_coins": primary_coins,
                "btc_rows": int(sum(btc.values())),
                "pair_rows": json.dumps(pair_rows, sort_keys=True),
                "minimum_rows": MIN_SCORABLE_ROWS,
                "minimum_coins": MIN_RETAIN_COINS,
                "status": (
                    "supported"
                    if primary_rows >= MIN_SCORABLE_ROWS and primary_coins >= MIN_RETAIN_COINS
                    else "insufficient_common_support_after_controls"
                ),
                "outcome_values_used_for_coverage_decision": False,
            }
        )
    return DataFrame(rows)


def profile_config(
    base: dict[str, Any],
    *,
    identifier: str,
    pairs: Sequence[str],
    feature_dir: Path,
    event_dir: Path,
    technical_smoke: bool,
) -> dict[str, Any]:
    config = json.loads(json.dumps(base))
    config["exchange"]["pair_whitelist"] = list(pairs)
    freqai = config["freqai"]
    freqai["enabled"] = True
    freqai["identifier"] = identifier
    freqai["train_period_days"] = TRAIN_DAYS
    freqai["backtest_period_days"] = BACKTEST_DAYS
    freqai["save_backtest_models"] = False
    freqai["multitarget_parallel_training"] = False
    freqai["feature_parameters"]["plot_feature_importances"] = 0
    freqai["feature_parameters"]["include_corr_pairlist"] = []
    freqai["feature_parameters"]["include_timeframes"] = ["1h"]
    freqai["feature_parameters"]["include_shifted_candles"] = 0
    freqai["feature_parameters"]["label_period_candles"] = MAX_TARGET_HORIZON_HOURS
    freqai["data_split_parameters"] = {"test_size": 0, "shuffle": False}
    training = freqai["model_training_parameters"]
    training["n_jobs"] = 1
    training["min_child_samples"] = 10
    if technical_smoke:
        training["n_estimators"] = min(20, int(training.get("n_estimators", 100)))
    config["market_reaction_zone_g5c"] = {
        "feature_cache_dir": str(feature_dir.resolve()),
        "event_cache_dir": str(event_dir.resolve()),
        "maximum_target_horizon_hours": MAX_TARGET_HORIZON_HOURS,
        "fixed_training_cutoff_utc": str(CONFIRMATION_START),
        "single_unretrained_confirmation_span": True,
        "missing_source_policy": "retain_nan_and_exclude_from_common_event_mask",
    }
    return config


def build_commands(
    *,
    run_id: str,
    record_dir: Path,
    artifact_dir: Path,
    profiles: Sequence[str],
    pairs: Sequence[str],
    base_config: Path,
    python_exe: Path,
    feature_dir: Path,
    event_dir: Path,
    technical_smoke: bool,
) -> list[dict[str, Any]]:
    base = json.loads(base_config.read_text(encoding="utf-8"))
    commands: list[dict[str, Any]] = []
    for profile_id in profiles:
        definition = PROFILES[profile_id]
        profile_dir = artifact_dir / "profiles" / profile_id
        userdir = profile_dir / "user_data"
        export_dir = profile_dir / "export"
        identifier = f"{run_id}_{profile_id}"
        config_path = record_dir / f"config_{profile_id}.json"
        atomic_write_json(
            profile_config(
                base,
                identifier=identifier,
                pairs=pairs,
                feature_dir=feature_dir,
                event_dir=event_dir,
                technical_smoke=technical_smoke,
            ),
            config_path,
        )
        command = [
            str(python_exe),
            "-m",
            "freqtrade",
            "backtesting",
            "--userdir",
            str(userdir),
            "--strategy-path",
            str(STRATEGY_PATH),
            "--datadir",
            str(DATA_DIR),
            "--config",
            str(config_path),
            "--strategy",
            definition["strategy"],
            "--freqaimodel",
            "LightGBMRegressorMultiTarget",
            "--timerange",
            TIMERANGE,
            "--export",
            "signals",
            "--export-directory",
            str(export_dir),
            "--cache",
            "none",
        ]
        commands.append(
            {
                "profile_id": profile_id,
                "strategy": definition["strategy"],
                "role": definition["role"],
                "identifier": identifier,
                "config_path": str(config_path),
                "artifact_dir": str(profile_dir),
                "user_data_dir": str(userdir),
                "model_dir": str(userdir / "models" / identifier),
                "export_dir": str(export_dir),
                "command": command,
                "status": "pending",
                "attempts": 0,
            }
        )
    return commands


def prepare_run(
    *,
    run_id: str,
    pairs: Sequence[str],
    profiles: Sequence[str],
    base_config: Path,
    python_exe: Path,
    profile_workers: int,
    cache_workers: int,
    technical_smoke: bool,
) -> tuple[dict[str, Any], Path, Path]:
    branch = validate_frozen_branch()
    source = g4d_source_contract()
    fresh = fresh_preflight_contract()
    if not set(pairs).issubset(source["pairs"]):
        raise ValueError("Selected G5C pairs are not all present in the historical source.")
    record_dir = RECORD_ROOT / run_id
    artifact_dir = ARTIFACT_ROOT / run_id
    manifest_path = record_dir / "manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8")), manifest_path, artifact_dir
    record_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = artifact_dir / "feature_cache"
    event_dir = artifact_dir / "event_cache"
    cache_results = build_caches(
        source=source,
        pairs=pairs,
        feature_dir=feature_dir,
        event_dir=event_dir,
        workers=cache_workers,
    )
    atomic_write_parquet(DataFrame(cache_results), record_dir / "g5c_cache_inventory.parquet")
    failures = [row for row in cache_results if row["status"] == "failed"]
    if failures:
        atomic_write_json(
            {"run_id": run_id, "status": "cache_failed", "failures": failures},
            manifest_path,
        )
        raise ValueError(f"{len(failures)} G5C pair cache(s) failed.")
    coverage = post_control_coverage(cache_results)
    coverage_path = record_dir / "g5c_post_control_coverage.parquet"
    atomic_write_parquet(coverage, coverage_path)
    if not coverage["status"].eq("supported").all():
        atomic_write_json(
            {
                "run_id": run_id,
                "status": "post_control_coverage_failed",
                "coverage_path": str(coverage_path),
            },
            manifest_path,
        )
        raise ValueError("G5C fresh support fell below the frozen gate after fair controls.")
    commands = build_commands(
        run_id=run_id,
        record_dir=record_dir,
        artifact_dir=artifact_dir,
        profiles=profiles,
        pairs=pairs,
        base_config=base_config,
        python_exe=python_exe,
        feature_dir=feature_dir,
        event_dir=event_dir,
        technical_smoke=technical_smoke,
    )
    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "prepared",
        "created_at_utc": utc_now(),
        "technical_smoke_not_evidence": bool(technical_smoke),
        "frozen_branch_id": branch["id"],
        "surface_id": SURFACE_ID,
        "cohort": "normal",
        "objective": branch["hypothesis"],
        "strongest_alternative": branch["strongest_alternative"],
        "retain_interpretation": branch["retain_interpretation"],
        "park_interpretation": branch["park_interpretation"],
        "iteration_cap": int(branch["iteration_cap"]),
        "timerange": TIMERANGE,
        "confirmation_start_utc": str(CONFIRMATION_START),
        "confirmation_end_utc_exclusive": str(CONFIRMATION_END),
        "confirmation_periods": list(CONFIRMATION_PERIODS),
        "train_period_days": TRAIN_DAYS,
        "backtest_period_days": BACKTEST_DAYS,
        "single_training_cutoff_for_both_confirmation_slices": True,
        "profile_workers": int(profile_workers),
        "model_threads_per_profile": 1,
        "model_class": "LightGBMRegressorMultiTarget",
        "pairs": list(pairs),
        "profiles": list(profiles),
        "target": TARGET_COLUMN,
        "target_definition": (
            "Largest absolute next-hour move from the event close, divided by ATR known "
            "before the event; magnitude only, with no direction or profit label."
        ),
        "comparisons": [list(item) for item in COMPARISONS],
        "leakage_controls": {
            "identical_target_rows_for_every_profile": True,
            "all_features_available_by_event_open": True,
            "one_hour_training_boundary_purge": True,
            "confirmation_early_not_used_to_retrain_confirmation_late": True,
            "shuffled_control_is_nonself_within_pair_period": True,
            "stale_control_uses_only_prior_event_in_same_pair_period": True,
            "fresh_coverage_gate_completed_before_outcomes_opened": True,
            "target_values_used_for_feature_selection": False,
        },
        "declared_review_thresholds_not_native_scores": {
            "minimum_rows_per_confirmation_slice": MIN_SCORABLE_ROWS,
            "minimum_positive_coins": MIN_RETAIN_COINS,
            "required_periods": list(CONFIRMATION_PERIODS),
            "paired_equal_coin_mae_gain_must_be_positive": True,
            "block_bootstrap_lower_95_must_exceed_zero": True,
            "rank_change_must_not_be_negative": True,
            "leave_one_coin_out_minimum_gain_must_be_positive": True,
            "current_profile_calibration_slope_must_be_positive": True,
            "bootstrap_samples": BLOCK_BOOTSTRAP_SAMPLES,
            "bootstrap_block_days": BLOCK_DAYS,
            "arbitrary_percentage_gate": None,
        },
        "source_contracts": {
            "frozen_batch": str(FROZEN_BATCH),
            "frozen_batch_sha256": sha256_file(FROZEN_BATCH),
            "analysis_script_sha256": sha256_file(Path(__file__)),
            "strategy_sha256": sha256_file(STRATEGY_FILE),
            "g4d_source_manifest": str(source["manifest_path"]),
            "g4d_source_manifest_sha256": source["manifest_sha256"],
            "g4d_source_result": str(source["result_path"]),
            "g4d_source_result_sha256": source["result_sha256"],
            "fresh_preflight_record": str(FRESH_PREFLIGHT_RECORD),
            "fresh_preflight_record_sha256": fresh["record_sha256"],
            "fresh_coverage": str(fresh["coverage_path"]),
            "fresh_coverage_sha256": fresh["coverage_sha256"],
            "fresh_inventory": str(fresh["inventory_path"]),
            "fresh_inventory_sha256": fresh["inventory_sha256"],
            "cache_inventory": cache_results,
            "post_control_coverage": str(coverage_path),
            "post_control_coverage_sha256": sha256_file(coverage_path),
        },
        "storage": {
            "compact_record_dir": str(record_dir),
            "bulky_artifact_dir": str(artifact_dir),
            "feature_cache_dir": str(feature_dir),
            "event_cache_dir": str(event_dir),
            "save_backtest_models": False,
        },
        "direction_prediction": False,
        "profit_optimization": False,
        "commands": commands,
    }
    atomic_write_json(manifest, manifest_path)
    return manifest, manifest_path, artifact_dir


def runtime_preflight(manifest: dict[str, Any], *, python_exe: Path) -> dict[str, Any]:
    problems: list[str] = []
    warnings: list[str] = []
    if not python_exe.is_file():
        problems.append(f"missing worker interpreter: {python_exe}")
    if not STRATEGY_FILE.is_file():
        problems.append(f"missing strategy: {STRATEGY_FILE}")
    if not 1 <= int(manifest["profile_workers"]) <= MAX_WORKERS:
        problems.append("profile worker count is outside 1..4")
    if manifest.get("timerange") != TIMERANGE or int(manifest.get("backtest_period_days", 0)) != 31:
        problems.append("the single unretrained confirmation span changed")
    for item in manifest["source_contracts"]["cache_inventory"]:
        for label in ("feature", "event"):
            path = Path(str(item[f"{label}_path"]))
            if not path.is_file():
                problems.append(f"missing cache: {path}")
            elif sha256_file(path) != item[f"{label}_sha256"]:
                problems.append(f"cache hash changed: {path}")
    dependency: dict[str, Any] = {"returncode": None, "stdout": "", "stderr": ""}
    if python_exe.is_file():
        result = subprocess.run(
            [
                str(python_exe),
                "-c",
                "import freqtrade, lightgbm, pyarrow; print('g5c-worker-dependencies-ok')",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        dependency = {
            "returncode": int(result.returncode),
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
        if result.returncode:
            problems.append("worker dependency import failed")
    free_gib = round(shutil.disk_usage(ARTIFACT_ROOT).free / (1024**3), 3)
    if free_gib < 5.0:
        warnings.append(f"Only {free_gib} GiB is free on the bulky artifact drive.")
    return {
        "created_at_utc": utc_now(),
        "passed": not problems,
        "problems": problems,
        "warnings": warnings,
        "runtime_snapshot": runtime_snapshot(),
        "worker_interpreter": str(python_exe),
        "dependency_check": dependency,
        "profiles": len(manifest["profiles"]),
        "pairs": len(manifest["pairs"]),
        "profile_workers": manifest["profile_workers"],
        "model_threads_per_profile": 1,
        "bulky_storage_free_gib": free_gib,
        "direction_prediction": False,
        "profit_optimization": False,
    }


def run_manifest(manifest: dict[str, Any], manifest_path: Path) -> int:
    manifest["status"] = "running"
    manifest["orchestrator_pid"] = os.getpid()
    manifest.setdefault("started_at_utc", utc_now())
    atomic_write_json(manifest, manifest_path)
    items_by_id = {item["profile_id"]: item for item in manifest["commands"]}
    pending = [item for item in manifest["commands"] if item.get("status") != "completed"]
    processed = len(manifest["commands"]) - len(pending)
    workers = int(manifest["profile_workers"])
    for start in range(0, len(pending), workers):
        batch = pending[start : start + workers]
        for item in batch:
            item["status"] = "running"
            item["started_at_utc"] = utc_now()
        atomic_write_json(manifest, manifest_path)
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futures = {pool.submit(run_freqai_profile, item): item for item in batch}
            for future in as_completed(futures):
                result = future.result()
                item = items_by_id[str(result["profile_id"])]
                item.update(result)
                item["status"] = "completed" if result["returncode"] == 0 else "failed"
                processed += 1
                atomic_write_json(manifest, manifest_path)
                print(
                    json.dumps(
                        {
                            "phase": "g5c_freqai_profiles",
                            "processed": processed,
                            "total": len(manifest["commands"]),
                            "profile_id": item["profile_id"],
                            "status": item["status"],
                            "stderr_log": item["stderr_log"],
                        }
                    ),
                    flush=True,
                )
        failures = [item for item in batch if item["status"] == "failed"]
        if failures:
            manifest["status"] = "failed"
            manifest["failed_profile_ids"] = [item["profile_id"] for item in failures]
            manifest["finished_at_utc"] = utc_now()
            atomic_write_json(manifest, manifest_path)
            return int(failures[0]["returncode"])
    return 0


def load_event_cache(event_dir: Path, pair: str) -> DataFrame:
    frame = pd.read_parquet(event_dir / f"{pair_stem(pair)}.parquet")
    frame["date"] = normalize_dates(frame["date"])
    return frame


def target_band_thresholds(
    events_by_pair: dict[str, DataFrame],
) -> tuple[dict[str, tuple[float, float]], DataFrame]:
    thresholds: dict[str, tuple[float, float]] = {}
    rows: list[dict[str, Any]] = []
    for pair, events in events_by_pair.items():
        values = pd.to_numeric(
            events.loc[events["date"].lt(CONFIRMATION_START), TARGET_COLUMN],
            errors="coerce",
        ).dropna()
        low = float(values.quantile(1.0 / 3.0)) if len(values) >= MIN_SCORABLE_ROWS else np.nan
        high = float(values.quantile(2.0 / 3.0)) if len(values) >= MIN_SCORABLE_ROWS else np.nan
        supported = bool(np.isfinite(low) and np.isfinite(high) and low < high)
        if supported:
            thresholds[pair] = (low, high)
        rows.append(
            {
                "pair": pair,
                "target": TARGET_COLUMN,
                "exposed_training_rows": len(values),
                "lower_to_middle_boundary": low,
                "middle_to_upper_boundary": high,
                "status": "supported" if supported else "unscorable",
                "confirmation_outcomes_used": False,
            }
        )
    return thresholds, DataFrame(rows)


def assign_bands(frame: DataFrame, *, thresholds: dict[str, tuple[float, float]]) -> DataFrame:
    output = frame.copy()
    output["actual_band"] = np.nan
    output["prediction_band"] = np.nan
    for pair, (low, high) in thresholds.items():
        selected = output["pair"].eq(pair)
        for source, destination in (("actual", "actual_band"), ("prediction", "prediction_band")):
            values = pd.to_numeric(output.loc[selected, source], errors="coerce")
            output.loc[selected, destination] = np.select(
                (values.le(low), values.le(high)),
                (0.0, 1.0),
                default=2.0,
            )
    return output


def model_cutoff_audit(model_dir: Path, pairs: Sequence[str]) -> dict[str, Any]:
    prediction_dir = model_dir / "backtesting_predictions"
    files = sorted(prediction_dir.glob("*_prediction.feather"))
    epochs: list[int] = []
    unidentified: list[str] = []
    for path in files:
        match = re.search(r"_(\d+)_prediction\.feather$", path.name)
        if match is None:
            unidentified.append(path.name)
        else:
            epochs.append(int(match.group(1)))
    unique = sorted(set(epochs))
    expected = int(CONFIRMATION_START.timestamp())
    passed = len(files) == len(pairs) and unique == [expected] and not unidentified
    return {
        "files": len(files),
        "expected_files": len(pairs),
        "unique_training_cutoff_epochs": unique,
        "expected_training_cutoff_epoch": expected,
        "unidentified_files": unidentified,
        "passed": passed,
    }


def prediction_event_surface(
    manifest: dict[str, Any],
    *,
    record_dir: Path,
) -> tuple[DataFrame, dict[str, Any], DataFrame]:
    pairs = tuple(str(pair) for pair in manifest["pairs"])
    predictions: dict[str, DataFrame] = {}
    file_audits: list[dict[str, Any]] = []
    for item in manifest["commands"]:
        if item.get("status") != "completed":
            continue
        profile_id = str(item["profile_id"])
        model_dir = Path(str(item["model_dir"]))
        cutoff_audit = model_cutoff_audit(model_dir, pairs)
        if not cutoff_audit["passed"]:
            raise ValueError(f"G5C {profile_id} did not use one fixed July-20 model span.")
        frame, audit = load_predictions(model_dir, pairs)
        audit["profile_id"] = profile_id
        audit["training_cutoff_audit"] = cutoff_audit
        file_audits.append(audit)
        if frame.empty or TARGET_COLUMN not in frame:
            raise ValueError(f"No complete G5C FreqAI predictions for {profile_id}.")
        predictions[profile_id] = frame
    if not predictions:
        raise ValueError("No completed G5C prediction profiles are available.")
    common_keys, eligibility = common_prediction_keys(predictions, pairs)
    if common_keys.empty:
        raise ValueError("G5C profiles have no common prediction keys.")
    atomic_write_parquet(eligibility, record_dir / "g5c_prediction_eligibility.parquet")
    event_dir = Path(manifest["storage"]["event_cache_dir"])
    events_by_pair = {pair: load_event_cache(event_dir, pair) for pair in pairs}
    thresholds, threshold_frame = target_band_thresholds(events_by_pair)
    rows: list[DataFrame] = []
    coverage: list[dict[str, Any]] = []
    for profile_id, frame in predictions.items():
        fair = frame.merge(common_keys, on=["pair", "date"], how="inner", validate="one_to_one")
        for pair in pairs:
            events = events_by_pair[pair]
            events = events.loc[events["period"].isin(CONFIRMATION_PERIODS)].copy()
            pair_predictions = fair.loc[fair["pair"].eq(pair)]
            merged = events.merge(
                pair_predictions[["date", TARGET_COLUMN]],
                on="date",
                how="inner",
                suffixes=("_actual", "_prediction"),
                validate="one_to_one",
            )
            coverage.append(
                {
                    "profile_id": profile_id,
                    "pair": pair,
                    "eligible_confirmation_events": len(events),
                    "common_prediction_events": len(merged),
                    "excluded_confirmation_events": len(events) - len(merged),
                }
            )
            block = merged[
                ["date", "period", f"{TARGET_COLUMN}_actual", f"{TARGET_COLUMN}_prediction"]
            ].rename(
                columns={
                    f"{TARGET_COLUMN}_actual": "actual",
                    f"{TARGET_COLUMN}_prediction": "prediction",
                }
            )
            block["profile_id"] = profile_id
            block["pair"] = pair
            block["target"] = TARGET_COLUMN
            rows.append(block)
    output = pd.concat(rows, ignore_index=True) if rows else DataFrame()
    output = assign_bands(output, thresholds=thresholds)
    if output.empty or output[["prediction", "actual"]].isna().any().any():
        raise ValueError("G5C common event predictions are empty or incomplete.")
    if output.duplicated(["profile_id", "pair", "date", "target"]).any():
        raise ValueError("G5C scoring surface contains duplicate prediction keys.")
    audit = {
        "created_at_utc": utc_now(),
        "profile_files": file_audits,
        "common_prediction_rows": len(common_keys),
        "common_prediction_key_digest": eligibility_digest(common_keys),
        "common_event_prediction_rows": len(output),
        "event_coverage": coverage,
        "single_training_cutoff_for_both_confirmation_slices": True,
        "exposed_training_outcomes_used_only_for_calibration_bands": True,
        "confirmation_outcomes_used_to_set_bands": False,
    }
    return output, audit, threshold_frame


def score_scopes(manifest: dict[str, Any], predictions: DataFrame) -> DataFrame:
    pairs = tuple(str(pair) for pair in manifest["pairs"])
    primary = tuple(pair for pair in pairs if not pair.startswith("BTC/"))
    scopes = {
        "primary_cohort": primary,
        "btc_only": tuple(pair for pair in pairs if pair.startswith("BTC/")),
        **{f"pair::{pair}": (pair,) for pair in pairs},
    }
    rows: list[dict[str, Any]] = []
    for scope, members in scopes.items():
        selected = predictions.loc[predictions["pair"].isin(members)]
        scope_type = (
            "pair"
            if scope.startswith("pair::")
            else "btc"
            if scope == "btc_only"
            else "cohort"
        )
        for (profile_id, period), frame in selected.groupby(
            ["profile_id", "period"], observed=True, sort=False
        ):
            rows.append(
                {
                    "scope": scope,
                    "scope_type": scope_type,
                    "member_pairs": json.dumps(list(members)),
                    "member_pair_count": len(members),
                    "profile_id": profile_id,
                    "period": period,
                    "target": TARGET_COLUMN,
                    **regression_metrics(frame),
                }
            )
    return DataFrame(rows)


def paired_profile_comparisons(
    manifest: dict[str, Any], predictions: DataFrame, scores: DataFrame
) -> DataFrame:
    pairs = tuple(str(pair) for pair in manifest["pairs"])
    scopes = {
        "primary_cohort": tuple(pair for pair in pairs if not pair.startswith("BTC/")),
        "btc_only": tuple(pair for pair in pairs if pair.startswith("BTC/")),
    }
    rows: list[dict[str, Any]] = []
    for scope, members in scopes.items():
        for comparison_id, profile, control in COMPARISONS:
            selected = predictions.loc[
                predictions["profile_id"].isin((profile, control))
                & predictions["pair"].isin(members)
            ].copy()
            pivot = selected.pivot(
                index=["pair", "date", "period", "actual"],
                columns="profile_id",
                values="prediction",
            ).reset_index()
            if profile not in pivot or control not in pivot:
                continue
            pivot["paired_error_gain"] = (
                (pivot[control] - pivot["actual"]).abs()
                - (pivot[profile] - pivot["actual"]).abs()
            )
            for period in CONFIRMATION_PERIODS:
                frame = pivot.loc[pivot["period"].eq(period)].copy()
                point, lower, upper = deterministic_block_bootstrap(
                    frame,
                    seed_key=f"{manifest['run_id']}|{scope}|{comparison_id}|{period}",
                )
                coin = frame.groupby("pair", observed=True)["paired_error_gain"].mean()
                loo = [float(coin.drop(index=pair).mean()) for pair in coin.index if len(coin) > 1]
                control_mae = (
                    float(
                        frame.assign(control_error=(frame[control] - frame["actual"]).abs())
                        .groupby("pair", observed=True)["control_error"]
                        .mean()
                        .mean()
                    )
                    if not frame.empty
                    else np.nan
                )
                profile_score = scores.loc[
                    scores["scope"].eq(scope)
                    & scores["profile_id"].eq(profile)
                    & scores["period"].eq(period)
                ]
                control_score = scores.loc[
                    scores["scope"].eq(scope)
                    & scores["profile_id"].eq(control)
                    & scores["period"].eq(period)
                ]
                supported = bool(
                    len(frame) >= MIN_SCORABLE_ROWS
                    and len(coin) >= MIN_RETAIN_COINS
                    and len(profile_score) == 1
                    and len(control_score) == 1
                    and profile_score.iloc[0]["status"] == "scored"
                    and control_score.iloc[0]["status"] == "scored"
                )
                rows.append(
                    {
                        "scope": scope,
                        "comparison_id": comparison_id,
                        "surface_id": SURFACE_ID,
                        "period": period,
                        "target": TARGET_COLUMN,
                        "profile_id": profile,
                        "control_profile_id": control,
                        "rows": len(frame),
                        "coins": len(coin),
                        "supported": supported,
                        "equal_coin_paired_mae_gain": point,
                        "paired_mae_gain_percent_of_control": (
                            100.0 * point / control_mae
                            if np.isfinite(point) and np.isfinite(control_mae) and control_mae > 0.0
                            else np.nan
                        ),
                        "block_bootstrap_lower_95": lower,
                        "block_bootstrap_upper_95": upper,
                        "positive_coin_count": int(coin.gt(0.0).sum()),
                        "scoreable_coin_count": len(coin),
                        "leave_one_coin_out_minimum_gain": min(loo) if loo else np.nan,
                        "leave_one_coin_out_positive": bool(loo and min(loo) > 0.0),
                        "spearman_change": (
                            float(
                                profile_score.iloc[0]["prediction_actual_spearman"]
                                - control_score.iloc[0]["prediction_actual_spearman"]
                            )
                            if supported
                            else np.nan
                        ),
                        "profile_calibration_slope": (
                            float(profile_score.iloc[0]["calibration_slope"])
                            if supported
                            else np.nan
                        ),
                        "profile_band_rows": (
                            int(profile_score.iloc[0]["band_rows"]) if supported else 0
                        ),
                        "direction_prediction": False,
                        "profit_optimization": False,
                    }
                )
    return DataFrame(rows)


def comparison_passed(row: pd.Series) -> bool:
    return bool(
        row["supported"]
        and row["equal_coin_paired_mae_gain"] > 0.0
        and np.isfinite(row["block_bootstrap_lower_95"])
        and row["block_bootstrap_lower_95"] > 0.0
        and row["positive_coin_count"] >= MIN_RETAIN_COINS
        and row["leave_one_coin_out_positive"]
        and row["spearman_change"] >= 0.0
        and np.isfinite(row["profile_calibration_slope"])
        and row["profile_calibration_slope"] > 0.0
        and row["profile_band_rows"] >= MIN_SCORABLE_ROWS
    )


def retention_decision(manifest: dict[str, Any], comparisons: DataFrame) -> DataFrame:
    expected = {item[0] for item in COMPARISONS}
    period_checks: list[dict[str, Any]] = []
    complete = set(manifest["profiles"]) == set(PROFILES)
    repeated = complete
    for period in CONFIRMATION_PERIODS:
        selected = comparisons.loc[
            comparisons["scope"].eq("primary_cohort")
            & comparisons["period"].eq(period)
            & comparisons["comparison_id"].isin(expected)
        ]
        present = set(selected["comparison_id"]) == expected
        passes = {
            str(row.comparison_id): comparison_passed(pd.Series(row._asdict()))
            for row in selected.itertuples(index=False)
        }
        passed = bool(present and all(passes.values()))
        complete &= present
        repeated &= passed
        period_checks.append(
            {
                "period": period,
                "required_comparisons": sorted(expected),
                "present_comparisons": sorted(selected["comparison_id"].tolist()),
                "comparison_passes": passes,
                "passed": passed,
            }
        )
    if manifest.get("technical_smoke_not_evidence"):
        status = "technical_smoke_not_evidence"
        reason = "The smoke run checks plumbing only."
    elif not complete:
        status = "parked_insufficient_common_support"
        reason = "The identical-row confirmation surface could not apply every frozen control."
    elif repeated:
        status = "retained_later_period_excursion_lead"
        reason = (
            "The current causal density geometry beat state, shuffled, and stale controls "
            "in both unseen slices with uncertainty above zero and broad coin support."
        )
    else:
        status = "parked_not_repeated_beyond_controls"
        reason = (
            "The small historical excursion lead did not repeat beyond every control with "
            "uncertainty excluding no improvement in both fresh slices."
        )
    return DataFrame(
        [
            {
                "surface_id": SURFACE_ID,
                "target": TARGET_COLUMN,
                "status": status,
                "reason": reason,
                "period_checks": json.dumps(period_checks, sort_keys=True),
                "direction_prediction": False,
                "profit_optimization": False,
            }
        ]
    )


def score_run(
    manifest: dict[str, Any], *, record_dir: Path, artifact_dir: Path
) -> dict[str, Any]:
    predictions, audit, thresholds = prediction_event_surface(manifest, record_dir=record_dir)
    scores = score_scopes(manifest, predictions)
    comparisons = paired_profile_comparisons(manifest, predictions, scores)
    decision = retention_decision(manifest, comparisons)
    prediction_path = artifact_dir / "g5c_common_event_predictions.parquet"
    atomic_write_parquet(predictions, prediction_path)
    atomic_write_parquet(scores, record_dir / "g5c_regression_scores.parquet")
    atomic_write_parquet(comparisons, record_dir / "g5c_paired_profile_comparisons.parquet")
    atomic_write_parquet(decision, record_dir / "g5c_surface_decision.parquet")
    atomic_write_parquet(thresholds, record_dir / "g5c_exposed_training_band_thresholds.parquet")
    atomic_write_json(audit, record_dir / "g5c_prediction_audit.json")
    result = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "surface_id": SURFACE_ID,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "profiles": int(predictions["profile_id"].nunique()),
        "pairs": int(predictions["pair"].nunique()),
        "target": TARGET_COLUMN,
        "prediction_rows": len(predictions),
        "comparison_rows": len(comparisons),
        "decision": decision.iloc[0]["status"],
        "prediction_path": str(prediction_path),
        "interpretation": (
            "This is a magnitude-only later-period confirmation. Passing would mean the "
            "density geometry helps estimate reaction size; it would not identify direction "
            "or imply a profitable trade."
        ),
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(result, record_dir / "g5c_result.json")
    return result


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    parser = argparse.ArgumentParser(
        description="G5C later-period FreqAI confirmation of next-hour absolute excursion."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--profiles", default="all")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--profile-workers", type=int, default=4)
    parser.add_argument("--cache-workers", type=int, default=4)
    parser.add_argument("--technical-smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    for label, value in (
        ("profile-workers", args.profile_workers),
        ("cache-workers", args.cache_workers),
    ):
        if not 1 <= value <= MAX_WORKERS:
            raise ValueError(f"{label} must be in 1..{MAX_WORKERS}; received {value}.")
    source = g4d_source_contract()
    pairs = tuple(source["pairs"]) if args.pairs == "all" else parse_csv(args.pairs)
    unknown_pairs = sorted(set(pairs).difference(source["pairs"]))
    if not pairs or unknown_pairs:
        raise ValueError(f"Invalid G5C pairs: {unknown_pairs}")
    profiles = tuple(PROFILES) if args.profiles == "all" else parse_csv(args.profiles)
    unknown_profiles = sorted(set(profiles).difference(PROFILES))
    if not profiles or unknown_profiles:
        raise ValueError(f"Invalid G5C profiles: {unknown_profiles}")
    if not args.base_config.is_file():
        raise FileNotFoundError(args.base_config)
    manifest, manifest_path, artifact_dir = prepare_run(
        run_id=args.run_id,
        pairs=pairs,
        profiles=profiles,
        base_config=args.base_config,
        python_exe=args.python,
        profile_workers=args.profile_workers,
        cache_workers=args.cache_workers,
        technical_smoke=args.technical_smoke,
    )
    record_dir = manifest_path.parent
    preflight = runtime_preflight(manifest, python_exe=args.python)
    atomic_write_json(preflight, record_dir / "g5c_launch_preflight.json")
    manifest["launch_preflight"] = preflight
    atomic_write_json(manifest, manifest_path)
    if not preflight["passed"]:
        manifest["status"] = "preflight_failed"
        atomic_write_json(manifest, manifest_path)
        return 2
    if args.prepare_only:
        return 0
    if args.retry_failed:
        for item in manifest["commands"]:
            if item.get("status") == "failed":
                item["status"] = "pending"
        manifest.pop("failed_profile_ids", None)
        manifest["status"] = "prepared"
        atomic_write_json(manifest, manifest_path)
    elif manifest.get("status") in {"failed", "preflight_failed"}:
        raise ValueError("Use --retry-failed after diagnosing a failed G5C profile.")
    if not args.score_only and manifest.get("status") != "completed":
        returncode = run_manifest(manifest, manifest_path)
        if returncode:
            return returncode
    incomplete = [
        item["profile_id"]
        for item in manifest["commands"]
        if item.get("status") != "completed"
    ]
    if incomplete:
        raise ValueError(f"Cannot score incomplete G5C profiles: {incomplete}")
    result = score_run(manifest, record_dir=record_dir, artifact_dir=artifact_dir)
    manifest["status"] = "completed"
    manifest["finished_at_utc"] = utc_now()
    manifest["result"] = result
    atomic_write_json(manifest, manifest_path)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

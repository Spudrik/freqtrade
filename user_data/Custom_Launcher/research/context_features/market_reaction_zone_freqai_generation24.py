"""Run Generation 24 multi-seed, multi-model FreqAI activity robustness tests."""

from __future__ import annotations

# Bound native pools before numerical imports.
# ruff: noqa: E402
import argparse
import json
import os
import shutil
import subprocess
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
    market_reaction_zone_freqai_generation0 as g0f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation22 as g22f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation23 as g23f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_freeze as g24z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_freqai_cache as g24c,
)


ANALYSIS_PATH = Path(__file__).resolve()
DEFAULT_CONFIG = g22f.DEFAULT_CONFIG
DEFAULT_PYTHON = g22f.DEFAULT_PYTHON
STRATEGY_PATH = g22f.STRATEGY_PATH
STRATEGY_FILE = STRATEGY_PATH / "MarketReactionZoneFreqAIGeneration24Strategy.py"
STRATEGY_CLASS = "MarketReactionZoneG24ConfigurableFreqAIResearchStrategy"
DATA_DIR = g22f.DATA_DIR
RECORD_ROOT = g24z.OUTPUT_ROOT / "freqai"
ARTIFACT_ROOT = Path(
    "D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones/"
    "generation24_branches/g24_broad_siblings/freqai_runs"
)
DEFAULT_RUN_STEM = "g24_activity_stability_freqai_20260828a"
DEFAULT_SMOKE_STEM = "g24_activity_stability_freqai_smoke_20260828a"
DEFAULT_JOINT_REVIEW_ID = "g24_activity_stability_freqai_joint_20260828a"
MAX_WORKERS = 4
MIN_TRAINING_ROWS = g23f.MIN_TRAINING_ROWS
MIN_PREDICTION_ROWS = g23f.MIN_PREDICTION_ROWS
MIN_PAIR_SCORE_ROWS = g23f.MIN_PAIR_SCORE_ROWS
TARGET_PURGE_HOURS = max(g24c.HORIZONS)
PAIR_METRICS = tuple(g23f.PAIR_METRICS)


def artifact(path: Path) -> dict[str, Any]:
    return g24z.artifact(path)


def cohort_settings(cohort: str, *, technical_smoke: bool) -> dict[str, Any]:
    return g23f.cohort_settings(cohort, technical_smoke=technical_smoke)


def load_sources(
    cohort: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    frozen, _ = g24c.load_branch()
    cache_path = g24c.cache_manifest_path(cohort)
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    registry = json.loads(g24c.REGISTRY_PATH.read_text(encoding="utf-8"))
    if cache.get("status") != "completed_generation24_freqai_cache":
        raise ValueError(f"Generation 24 {cohort} FreqAI cache is not terminal.")
    if registry.get("status") != "frozen_before_generation24_freqai_target_materialization":
        raise ValueError("Generation 24 FreqAI registry is invalid.")
    if g0.sha256_file(g24c.REGISTRY_PATH) != cache["profile_registry"]["sha256"]:
        raise ValueError("Generation 24 FreqAI registry changed after target construction.")
    return frozen, cache, registry


def active_registry(
    registry: dict[str, Any], cohort: str, *, technical_smoke: bool
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    profiles = {
        identifier: dict(profile)
        for identifier, profile in registry["profiles"].items()
        if profile["cohort"] == cohort
    }
    comparisons = [dict(item) for item in registry["comparisons"] if item["cohort"] == cohort]
    if technical_smoke:
        smoke_cells = {g24c.MODEL_CELLS[0], g24c.MODEL_CELLS[-1]}
        profiles = {
            key: value
            for key, value in profiles.items()
            if (str(value["model_class"]), int(value["seed"])) in smoke_cells
        }
        comparisons = [
            item
            for item in comparisons
            if (str(item["model_class"]), int(item["seed"])) in smoke_cells
        ]
    expected_cells = 2 if technical_smoke else len(g24c.MODEL_CELLS)
    if len(profiles) != expected_cells * len(g24c.PROFILE_ROLES):
        raise ValueError(f"Generation 24 {cohort} profile ladder is incomplete.")
    if len(comparisons) != expected_cells * len(g24c.CONTROLS):
        raise ValueError(f"Generation 24 {cohort} control ladder is incomplete.")
    return profiles, comparisons


def profile_config(
    base: dict[str, Any],
    *,
    identifier: str,
    pairs: Sequence[str],
    feature_dir: Path,
    event_dir: Path,
    profile: dict[str, Any],
    train_days: int,
    backtest_days: int,
    technical_smoke: bool,
) -> dict[str, Any]:
    config = g23f.profile_config(
        base,
        identifier=identifier,
        pairs=pairs,
        feature_dir=feature_dir,
        event_dir=event_dir,
        profile=profile,
        train_days=train_days,
        backtest_days=backtest_days,
        technical_smoke=technical_smoke,
    )
    config["market_reaction_zone_g24"] = config.pop("market_reaction_zone_g23")
    parameters = config["freqai"]["model_training_parameters"]
    parameters["n_jobs"] = 1
    if profile["model_class"] == "XGBoostRegressorMultiTarget":
        parameters.pop("min_child_samples", None)
        parameters.pop("num_leaves", None)
        parameters.update(
            {
                "min_child_weight": 1.0,
                "tree_method": "hist",
                "verbosity": 0,
            }
        )
    return config


def cache_dirs(cache: dict[str, Any]) -> dict[str, Path]:
    item = cache["inventory"][0]
    return {
        "actual_feature": Path(item["feature_path"]).parent,
        "permuted_feature": Path(item["permuted_feature_path"]).parent,
        "actual_event": Path(item["event_path"]).parent,
        "shuffled_event": Path(item["shuffled_event_path"]).parent,
        "evaluation": Path(item["evaluation_path"]).parent,
    }


def training_thresholds(
    *, cohort: str, cache: dict[str, Any], settings: dict[str, Any]
) -> DataFrame:
    event_dir = cache_dirs(cache)["actual_event"]
    rows: list[dict[str, Any]] = []
    for pair in cache["pairs"]:
        frame = pd.read_parquet(
            event_dir / f"{g0.pair_file_stem(str(pair))}.parquet",
            columns=["date", f"ready__{g24c.READY_BLOCK}", *g24c.TARGETS],
        )
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
        ready = frame[f"ready__{g24c.READY_BLOCK}"].fillna(False).astype(bool)
        for period in settings["period_bounds"]:
            start = pd.Timestamp(period["start_utc"])
            cutoff = start - pd.Timedelta(hours=TARGET_PURGE_HOURS)
            training_start = start - pd.Timedelta(days=int(settings["train_days"]))
            mask = ready & frame["date"].ge(training_start) & frame["date"].lt(cutoff)
            for target in g24c.TARGETS:
                values = pd.to_numeric(frame.loc[mask, target], errors="coerce").dropna()
                rows.append(
                    {
                        "cohort": cohort,
                        "pair": pair,
                        "period": period["id"],
                        "target": target,
                        "training_start_utc": training_start,
                        "training_cutoff_exclusive_utc": cutoff,
                        "training_rows": len(values),
                        "training_median": float(values.median()) if len(values) else np.nan,
                    }
                )
    return DataFrame.from_records(rows)


def parent_run(cohort: str) -> tuple[dict[str, Any], Path, Path]:
    root = g23f.RECORD_ROOT / f"{g23f.DEFAULT_RUN_STEM}_{cohort}"
    manifest_path = root / "g23_freqai_run_manifest.json"
    result_path = root / "g23_freqai_result.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed_generation23_freqai_cohort":
        raise ValueError(f"Generation 23 {cohort} FreqAI parent is not terminal.")
    full = [item for item in manifest["commands"] if item["role"] == "full_interaction"]
    if len(full) != 1 or full[0].get("status") != "completed":
        raise ValueError(f"Generation 23 {cohort} full parent is unavailable.")
    return manifest, manifest_path, result_path


def build_manifest(
    *,
    run_id: str,
    cohort: str,
    base_config: Path,
    python_exe: Path,
    profile_workers: int,
    technical_smoke: bool,
) -> tuple[dict[str, Any], Path]:
    frozen, cache, registry = load_sources(cohort)
    profiles, comparisons = active_registry(registry, cohort, technical_smoke=technical_smoke)
    parent, parent_manifest_path, parent_result_path = parent_run(cohort)
    settings = cohort_settings(cohort, technical_smoke=technical_smoke)
    record_dir = RECORD_ROOT / run_id
    artifact_dir = ARTIFACT_ROOT / run_id
    manifest_path = record_dir / "g24_freqai_run_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, path in (
            ("strategy", STRATEGY_FILE),
            ("freeze", g24z.FREEZE_PATH),
            ("cache_manifest", g24c.cache_manifest_path(cohort)),
            ("profile_registry", g24c.REGISTRY_PATH),
            ("analysis_script", ANALYSIS_PATH),
            ("generation23_parent_manifest", parent_manifest_path),
            ("generation23_parent_result", parent_result_path),
        ):
            if g0.sha256_file(path) != existing["source_contracts"][name]["sha256"]:
                raise ValueError(f"Generation 24 {name} changed: {manifest_path}")
        return existing, manifest_path
    record_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    base = json.loads(base_config.read_text(encoding="utf-8"))
    dirs = cache_dirs(cache)
    pairs = tuple(str(pair) for pair in cache["pairs"])
    if technical_smoke:
        pairs = pairs[:1]
    thresholds = training_thresholds(cohort=cohort, cache=cache, settings=settings)
    thresholds = thresholds.loc[thresholds["pair"].isin(pairs)].copy()
    threshold_path = record_dir / "g24_training_median_thresholds.csv"
    g0.atomic_write_csv(thresholds, threshold_path)
    commands: list[dict[str, Any]] = []
    for number, (profile_identifier, profile) in enumerate(profiles.items(), start=1):
        short_id = f"p{number:03d}_{g22f.g6f.stable_digest(profile_identifier, 6)}"
        profile_dir = artifact_dir / "profiles" / short_id
        userdir = profile_dir / "user_data"
        export_dir = profile_dir / "export"
        identifier = f"g24-{g22f.g6f.stable_digest(f'{run_id}|{profile_identifier}', 16)}"
        config_path = record_dir / f"config_{short_id}.json"
        feature_dir = dirs[f"{profile['feature_cache']}_feature"]
        event_dir = dirs[f"{profile['target_cache']}_event"]
        g0.atomic_write_json(
            profile_config(
                base,
                identifier=identifier,
                pairs=pairs,
                feature_dir=feature_dir,
                event_dir=event_dir,
                profile=profile,
                train_days=int(settings["train_days"]),
                backtest_days=int(settings["backtest_days"]),
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
            STRATEGY_CLASS,
            "--freqaimodel",
            str(profile["model_class"]),
            "--timerange",
            str(settings["timerange"]),
            "--export",
            "signals",
            "--export-directory",
            str(export_dir),
            "--cache",
            "none",
        ]
        commands.append(
            {
                **profile,
                "short_id": short_id,
                "pairs": list(pairs),
                "strategy": STRATEGY_CLASS,
                "identifier": identifier,
                "config_path": str(config_path.resolve()),
                "artifact_dir": str(profile_dir.resolve()),
                "user_data_dir": str(userdir.resolve()),
                "model_dir": str((userdir / "models" / identifier).resolve()),
                "export_dir": str(export_dir.resolve()),
                "feature_cache_dir": str(feature_dir.resolve()),
                "event_cache_dir": str(event_dir.resolve()),
                "command": command,
                "status": "pending",
                "attempts": 0,
            }
        )
    _, branch = g24c.load_branch()
    parent_full = next(item for item in parent["commands"] if item["role"] == "full_interaction")
    manifest = {
        "schema_version": 1,
        "generation": 24,
        "run_id": run_id,
        "created_at_utc": g0.utc_now(),
        "status": "prepared",
        "run_stage": "multi_seed_multi_model_activity_robustness",
        "cohort": cohort,
        "pairs": list(pairs),
        "timerange": settings["timerange"],
        "train_period_days": settings["train_days"],
        "backtest_period_days": settings["backtest_days"],
        "validation_periods": settings["validation_periods"],
        "period_bounds": settings["period_bounds"],
        "technical_smoke_not_evidence": technical_smoke,
        "profile_workers": max(1, min(profile_workers, MAX_WORKERS)),
        "model_threads_per_profile": 1,
        "profiles": list(profiles),
        "profile_count": len(profiles),
        "comparisons": comparisons,
        "comparison_count": len(comparisons),
        "targets": list(g24c.TARGETS),
        "target_metadata": list(g24c.TARGET_METADATA),
        "maximum_target_horizon_hours": TARGET_PURGE_HOURS,
        "decision_rule": {
            "plain_rule": branch["pass_rule"],
            "candidate_top_quartile_above_training_median_minimum": 0.55,
            "all_six_controls_required": True,
            "both_later_blocks_required": True,
            "all_four_model_seed_cells_required": not technical_smoke,
        },
        "research_boundary": dict(frozen["research_boundary"]),
        "source_contracts": {
            "base_config": artifact(base_config),
            "python_executable": str(python_exe.resolve()),
            "strategy": artifact(STRATEGY_FILE),
            "freeze": artifact(g24z.FREEZE_PATH),
            "cache_manifest": artifact(g24c.cache_manifest_path(cohort)),
            "profile_registry": artifact(g24c.REGISTRY_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
            "training_median_thresholds": artifact(threshold_path),
            "target_calibration": artifact(g24c.calibration_path(cohort)),
            "generation23_parent_manifest": artifact(parent_manifest_path),
            "generation23_parent_result": artifact(parent_result_path),
            "generation23_parent_full_profile": parent_full,
            "cache_inventory": cache["inventory"],
        },
        "storage": {
            "record_dir": str(record_dir.resolve()),
            "bulky_artifact_dir": str(artifact_dir.resolve()),
            "actual_feature_cache_dir": str(dirs["actual_feature"].resolve()),
            "permuted_feature_cache_dir": str(dirs["permuted_feature"].resolve()),
            "actual_event_cache_dir": str(dirs["actual_event"].resolve()),
            "shuffled_event_cache_dir": str(dirs["shuffled_event"].resolve()),
            "evaluation_cache_dir": str(dirs["evaluation"].resolve()),
            "training_median_thresholds": str(threshold_path.resolve()),
            "save_backtest_models": False,
        },
        "runtime_preparation_snapshot": g22f.g7f.runtime_snapshot(),
        "commands": commands,
    }
    g0.atomic_write_json(manifest, manifest_path)
    return manifest, manifest_path


def dependency_check(python_exe: Path) -> tuple[dict[str, Any], list[str]]:
    if not python_exe.is_file():
        return {}, [f"Missing worker interpreter: {python_exe}"]
    check = subprocess.run(
        [
            str(python_exe),
            "-c",
            "import freqtrade, lightgbm, sklearn, xgboost; print(freqtrade.__version__)",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    dependency = {
        "returncode": check.returncode,
        "stdout": check.stdout.strip(),
        "stderr": check.stderr.strip(),
    }
    problems = (
        ["The worker cannot import the frozen FreqAI dependencies."] if check.returncode else []
    )
    return dependency, problems


def preflight_run(manifest: dict[str, Any], *, python_exe: Path) -> dict[str, Any]:
    dependency, problems = dependency_check(python_exe)
    if not STRATEGY_FILE.is_file():
        problems.append(f"Missing Generation 24 strategy: {STRATEGY_FILE}")
    required_features = {
        column for command in manifest["commands"] for column in command["feature_columns"]
    }
    start_raw, end_raw = str(manifest["timerange"]).split("-", maxsplit=1)
    prediction_start = pd.Timestamp(start_raw, tz="UTC")
    prediction_end = pd.Timestamp(end_raw, tz="UTC")
    training_start = prediction_start - pd.Timedelta(days=int(manifest["train_period_days"]))
    readiness: list[dict[str, Any]] = []
    inventory = {
        str(item["pair"]): item for item in manifest["source_contracts"]["cache_inventory"]
    }
    for pair in manifest["pairs"]:
        item = inventory[pair]
        for key in (
            "feature",
            "permuted_feature",
            "support",
            "event",
            "evaluation",
            "shuffled_event",
        ):
            path = Path(item[f"{key}_path"])
            if not path.is_file() or g0.sha256_file(path) != item[f"{key}_sha256"]:
                problems.append(f"Cache hash mismatch for {pair} and {key}.")
        for feature_key in ("feature_path", "permuted_feature_path"):
            try:
                pd.read_parquet(item[feature_key], columns=sorted(required_features))
            except Exception as exc:
                problems.append(f"{pair} lacks exact G24 features in {feature_key}: {exc}")
        for label_source, path_key in (
            ("actual", "event_path"),
            ("shuffled", "shuffled_event_path"),
        ):
            frame = pd.read_parquet(
                item[path_key],
                columns=["date", f"ready__{g24c.READY_BLOCK}", *g24c.TARGETS],
            )
            frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
            eligible = frame[f"ready__{g24c.READY_BLOCK}"].fillna(False).astype(bool)
            eligible &= frame[list(g24c.TARGETS)].notna().all(axis=1)
            training_rows = int(
                (
                    eligible & frame["date"].ge(training_start) & frame["date"].lt(prediction_start)
                ).sum()
            )
            prediction_rows = int(
                (
                    eligible & frame["date"].ge(prediction_start) & frame["date"].lt(prediction_end)
                ).sum()
            )
            readiness.append(
                {
                    "pair": pair,
                    "label_source": label_source,
                    "training_rows": training_rows,
                    "prediction_rows": prediction_rows,
                }
            )
            if training_rows < MIN_TRAINING_ROWS:
                problems.append(f"{pair}/{label_source} has only {training_rows} training rows.")
            required_prediction_rows = (
                24 if manifest["technical_smoke_not_evidence"] else MIN_PREDICTION_ROWS
            )
            if prediction_rows < required_prediction_rows:
                problems.append(
                    f"{pair}/{label_source} has only {prediction_rows} prediction rows."
                )
    parent = manifest["source_contracts"]["generation23_parent_full_profile"]
    if parent.get("status") != "completed" or not Path(parent["model_dir"]).is_dir():
        problems.append("Generation 23 parent prediction model is unavailable.")
    free_gib = shutil.disk_usage(Path(manifest["storage"]["bulky_artifact_dir"])).free / (1024**3)
    if free_gib < 20.0:
        problems.append(f"Only {free_gib:.2f} GiB free on the artifact drive.")
    return {
        "created_at_utc": g0.utc_now(),
        "passed": not problems,
        "problems": problems,
        "dependency_check": dependency,
        "readiness_audit": readiness,
        "parent_control_available": not any("parent" in problem for problem in problems),
        "profile_workers": manifest["profile_workers"],
        "model_threads_per_profile": 1,
        "runtime_snapshot": g22f.g7f.runtime_snapshot(),
        "bulky_storage_free_gib": round(free_gib, 3),
    }


def load_actual(manifest: dict[str, Any]) -> DataFrame:
    frames: list[DataFrame] = []
    event_dir = Path(manifest["storage"]["actual_event_cache_dir"])
    for pair in manifest["pairs"]:
        frame = pd.read_parquet(
            event_dir / f"{g0.pair_file_stem(pair)}.parquet",
            columns=["date", "period", *g24c.TARGETS],
        )
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
        frame["pair"] = pair
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    if output.duplicated(["pair", "date"]).any():
        raise ValueError("Duplicate pair/date rows in Generation 24 target cache.")
    return output


def generation23_parent_predictions(
    manifest: dict[str, Any],
) -> tuple[DataFrame, dict[str, Any]]:
    parent = manifest["source_contracts"]["generation23_parent_full_profile"]
    frame, audit = g0f.load_predictions(Path(parent["model_dir"]), tuple(manifest["pairs"]))
    missing = sorted(set(g24c.PARENT_TARGETS).difference(frame.columns))
    if missing:
        raise ValueError(f"Generation 23 parent predictions lack targets: {missing}")
    output = frame[["pair", "date", *g24c.PARENT_TARGETS]].rename(
        columns=dict(zip(g24c.PARENT_TARGETS, g24c.TARGETS, strict=True))
    )
    return output, audit


def pair_scores(
    *,
    manifest: dict[str, Any],
    predictions: dict[str, DataFrame],
    parent_prediction: DataFrame,
    actual: DataFrame,
) -> tuple[DataFrame, DataFrame]:
    common_inputs = {**predictions, "generation23_calibrated_parent": parent_prediction}
    common, audit = g0f.common_prediction_keys(common_inputs, tuple(manifest["pairs"]))
    fair = common.merge(actual, on=["pair", "date"], how="left", validate="one_to_one")
    thresholds = pd.read_csv(manifest["storage"]["training_median_thresholds"])
    threshold_lookup = thresholds.set_index(["pair", "period", "target"])[
        "training_median"
    ].to_dict()
    metadata = {str(item["profile_id"]): item for item in manifest["commands"]}
    rows: list[dict[str, Any]] = []

    def append_prediction(
        prediction: DataFrame,
        *,
        profile_id: str,
        role: str,
        model_class: str,
        seed: int,
    ) -> None:
        renamed = prediction[["pair", "date", *g24c.TARGETS]].rename(
            columns={target: f"prediction__{target}" for target in g24c.TARGETS}
        )
        merged = fair.merge(renamed, on=["pair", "date"], how="left", validate="one_to_one")
        for (pair, period), cell in merged.groupby(["pair", "period"], observed=True, sort=False):
            if period not in manifest["validation_periods"]:
                continue
            for target in g24c.TARGETS:
                threshold = float(threshold_lookup.get((pair, period, target), np.nan))
                rows.append(
                    {
                        "cohort": manifest["cohort"],
                        "pair": pair,
                        "period": period,
                        "profile_id": profile_id,
                        "role": role,
                        "model_class": model_class,
                        "seed": seed,
                        "target": target,
                        "training_median": threshold,
                        **g23f.prediction_metrics(
                            cell[target], cell[f"prediction__{target}"], threshold
                        ),
                    }
                )

    for profile_id, prediction in predictions.items():
        item = metadata[profile_id]
        append_prediction(
            prediction,
            profile_id=profile_id,
            role=str(item["role"]),
            model_class=str(item["model_class"]),
            seed=int(item["seed"]),
        )
    for model_class, seed in sorted(
        {(str(item["model_class"]), int(item["seed"])) for item in manifest["commands"]}
    ):
        append_prediction(
            parent_prediction,
            profile_id="generation23_calibrated_parent",
            role="generation23_calibrated_parent",
            model_class=model_class,
            seed=seed,
        )
        for (pair, period), cell in fair.groupby(["pair", "period"], observed=True, sort=False):
            if period not in manifest["validation_periods"]:
                continue
            for target in g24c.TARGETS:
                threshold = float(threshold_lookup.get((pair, period, target), np.nan))
                rows.append(
                    {
                        "cohort": manifest["cohort"],
                        "pair": pair,
                        "period": period,
                        "profile_id": "constant_training_median",
                        "role": "constant_training_median",
                        "model_class": model_class,
                        "seed": seed,
                        "target": target,
                        "training_median": threshold,
                        **g23f.prediction_metrics(
                            cell[target], Series(threshold, index=cell.index), threshold
                        ),
                    }
                )
    return DataFrame.from_records(rows), audit


def scope_scores(pair_frame: DataFrame) -> DataFrame:
    expanded: list[DataFrame] = []
    for (cohort, pair), cell in pair_frame.groupby(["cohort", "pair"], observed=True, sort=False):
        for scope in g22f.g16d.group_for_pair(str(cohort), str(pair)):
            copy = cell.copy()
            copy["market_scope"] = scope
            expanded.append(copy)
    source = pd.concat(expanded, ignore_index=True)
    keys = [
        "cohort",
        "period",
        "model_class",
        "seed",
        "role",
        "target",
        "market_scope",
    ]
    rows: list[dict[str, Any]] = []
    for key, cell in source.groupby(keys, observed=True, sort=False):
        eligible = cell.loc[cell["rows"].ge(MIN_PAIR_SCORE_ROWS)].copy()
        scope = str(key[-1])
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "eligible_coins": int(eligible["pair"].nunique()),
                "required_coins": g22f.g16d.GROUP_MIN_COINS[scope],
                "coin_support_pass": bool(
                    eligible["pair"].nunique() >= g22f.g16d.GROUP_MIN_COINS[scope]
                ),
                **{
                    metric: float(pd.to_numeric(eligible[metric], errors="coerce").mean())
                    for metric in PAIR_METRICS
                },
            }
        )
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def control_comparisons(
    scope_frame: DataFrame,
    validation_periods: Sequence[str],
    *,
    technical_smoke: bool,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    rows: list[dict[str, Any]] = []
    keys = ["cohort", "market_scope", "period", "model_class", "seed", "target"]
    for key, cell in scope_frame.groupby(keys, observed=True, sort=False):
        indexed = cell.set_index("role")
        if "full_interaction" not in indexed.index:
            continue
        candidate = indexed.loc["full_interaction"]
        candidate_gate = bool(
            candidate["coin_support_pass"]
            and candidate["top_minus_bottom_difference"] > 0.0
            and candidate["top_quartile_above_training_median_fraction"] >= 0.55
        )
        role_for_control = {
            "constant_training_median": "constant_training_median",
            "level_geometry_only_model": "level_geometry_only",
            "market_state_only_model": "market_state_only",
            "within_pair_time_shuffled_training_labels": (
                "within_pair_time_shuffled_training_labels"
            ),
            "generation23_calibrated_parent": "generation23_calibrated_parent",
            "permuted_level_feature_block": "permuted_level_feature_block",
        }
        for control, baseline_role in role_for_control.items():
            present = baseline_role in indexed.index
            baseline = indexed.loc[baseline_role] if present else Series(dtype=object)
            if control == "constant_training_median":
                metrics_pass = bool(
                    present
                    and candidate["median_absolute_error"] < baseline["median_absolute_error"]
                )
            else:
                metrics_pass = bool(
                    present
                    and candidate["median_absolute_error"] < baseline["median_absolute_error"]
                    and candidate["spearman_rank_correlation"]
                    > baseline["spearman_rank_correlation"]
                    and candidate["top_minus_bottom_difference"]
                    > baseline["top_minus_bottom_difference"]
                )
            rows.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "control": control,
                    "control_present": present,
                    "candidate_gate_pass": candidate_gate,
                    "candidate_mae": candidate["median_absolute_error"],
                    "control_mae": baseline.get("median_absolute_error", np.nan),
                    "candidate_spearman": candidate["spearman_rank_correlation"],
                    "control_spearman": baseline.get("spearman_rank_correlation", np.nan),
                    "candidate_top_minus_bottom": candidate["top_minus_bottom_difference"],
                    "control_top_minus_bottom": baseline.get("top_minus_bottom_difference", np.nan),
                    "candidate_top_above_training_median_fraction": candidate[
                        "top_quartile_above_training_median_fraction"
                    ],
                    "period_control_pass": bool(candidate_gate and metrics_pass),
                }
            )
    comparisons = DataFrame.from_records(rows)
    cell_rows: list[dict[str, Any]] = []
    expected_periods = set(validation_periods)
    cell_keys = ["cohort", "market_scope", "model_class", "seed", "target"]
    for key, cell in comparisons.groupby(cell_keys, observed=True, sort=False):
        complete = (
            set(cell["control"]) == set(g24c.CONTROLS) and set(cell["period"]) == expected_periods
        )
        passed = bool(complete and cell["period_control_pass"].all())
        cell_rows.append(
            {
                **dict(zip(cell_keys, key, strict=True)),
                "complete_control_and_period_ladder": complete,
                "all_controls_both_periods_pass": passed,
                "minimum_candidate_top_above_training_median_fraction": float(
                    cell["candidate_top_above_training_median_fraction"].min()
                ),
                "minimum_candidate_top_minus_bottom": float(
                    cell["candidate_top_minus_bottom"].min()
                ),
            }
        )
    cell_decisions = DataFrame.from_records(cell_rows)
    final_rows: list[dict[str, Any]] = []
    expected_cells = 2 if technical_smoke else len(g24c.MODEL_CELLS)
    final_keys = ["cohort", "market_scope", "target"]
    for key, cell in cell_decisions.groupby(final_keys, observed=True, sort=False):
        cell_count = len(cell)
        retained = bool(
            cell_count == expected_cells and cell["all_controls_both_periods_pass"].all()
        )
        final_rows.append(
            {
                **dict(zip(final_keys, key, strict=True)),
                "status": (
                    "technical_smoke_only"
                    if technical_smoke
                    else "exploratory_activity_robustness_lead"
                    if retained
                    else "not_retained_across_all_model_seed_cells"
                ),
                "model_seed_cells_present": cell_count,
                "required_model_seed_cells": expected_cells,
                "all_model_seed_cells_pass": retained,
                "minimum_candidate_top_above_training_median_fraction": float(
                    cell["minimum_candidate_top_above_training_median_fraction"].min()
                ),
                "minimum_candidate_top_minus_bottom": float(
                    cell["minimum_candidate_top_minus_bottom"].min()
                ),
            }
        )
    return comparisons, cell_decisions, DataFrame.from_records(final_rows)


def score_run(manifest: dict[str, Any], *, record_dir: Path) -> dict[str, Any]:
    predictions: dict[str, DataFrame] = {}
    audits: list[dict[str, Any]] = []
    for item in manifest["commands"]:
        frame, audit = g0f.load_predictions(Path(item["model_dir"]), tuple(item["pairs"]))
        audit.update(
            {
                "profile_id": item["profile_id"],
                "role": item["role"],
                "model_class": item["model_class"],
                "seed": item["seed"],
            }
        )
        audits.append(audit)
        if frame.empty:
            raise ValueError(f"No Generation 24 predictions for {item['profile_id']}")
        missing = sorted(set(g24c.TARGETS).difference(frame.columns))
        if missing:
            raise ValueError(f"Profile {item['profile_id']} lacks targets: {missing}")
        predictions[str(item["profile_id"])] = frame
    parent, parent_audit = generation23_parent_predictions(manifest)
    parent_audit.update(
        {
            "profile_id": "generation23_calibrated_parent",
            "role": "generation23_calibrated_parent",
        }
    )
    audits.append(parent_audit)
    actual = load_actual(manifest)
    pair, eligibility = pair_scores(
        manifest=manifest,
        predictions=predictions,
        parent_prediction=parent,
        actual=actual,
    )
    scopes = scope_scores(pair)
    comparisons, cells, decisions = control_comparisons(
        scopes,
        manifest["validation_periods"],
        technical_smoke=manifest["technical_smoke_not_evidence"],
    )
    paths = {
        "prediction_audit": record_dir / "g24_prediction_audit.csv",
        "prediction_eligibility": record_dir / "g24_prediction_eligibility.csv",
        "pair_scores": record_dir / "g24_pair_scores.csv",
        "scope_scores": record_dir / "g24_scope_scores.csv",
        "control_comparisons": record_dir / "g24_control_comparisons.csv",
        "model_seed_decisions": record_dir / "g24_model_seed_decisions.csv",
        "decisions": record_dir / "g24_freqai_decisions.csv",
    }
    for frame, path in (
        (DataFrame.from_records(audits), paths["prediction_audit"]),
        (eligibility, paths["prediction_eligibility"]),
        (pair, paths["pair_scores"]),
        (scopes, paths["scope_scores"]),
        (comparisons, paths["control_comparisons"]),
        (cells, paths["model_seed_decisions"]),
        (decisions, paths["decisions"]),
    ):
        g0.atomic_write_csv(frame, path)
    retained = int(decisions["all_model_seed_cells_pass"].sum())
    result = {
        "schema_version": 1,
        "generation": 24,
        "run_id": manifest["run_id"],
        "created_at_utc": g0.utc_now(),
        "status": (
            "completed_generation24_freqai_technical_smoke_not_evidence"
            if manifest["technical_smoke_not_evidence"]
            else "completed_generation24_freqai_cohort"
        ),
        "technical_smoke_not_evidence": manifest["technical_smoke_not_evidence"],
        "cohort": manifest["cohort"],
        "profiles_completed": len(manifest["commands"]),
        "targets_scored": len(g24c.TARGETS),
        "exploratory_activity_robustness_leads": retained,
        "not_retained": len(decisions) - retained,
        "integrity": {
            "all_profile_commands_terminal": all(
                item.get("status") == "completed" for item in manifest["commands"]
            ),
            "generation23_parent_compared_on_common_keys": True,
            "profit_used": False,
            "future_signed_direction_used": False,
            "auc_used": False,
        },
        "artifacts": {key: artifact(path) for key, path in paths.items()},
    }
    result_path = record_dir / "g24_freqai_result.json"
    g0.atomic_write_json(result, result_path)
    return {**result, "result_path": str(result_path.resolve())}


def run_cell(
    *,
    run_id: str,
    cohort: str,
    base_config: Path,
    python_exe: Path,
    profile_workers: int,
    technical_smoke: bool,
    prepare_only: bool,
) -> dict[str, Any]:
    manifest, manifest_path = build_manifest(
        run_id=run_id,
        cohort=cohort,
        base_config=base_config,
        python_exe=python_exe,
        profile_workers=profile_workers,
        technical_smoke=technical_smoke,
    )
    audit = preflight_run(manifest, python_exe=python_exe)
    g0.atomic_write_json(audit, manifest_path.parent / "g24_freqai_preflight.json")
    if not audit["passed"]:
        manifest["status"] = "blocked_preflight"
        manifest["preflight"] = audit
        g0.atomic_write_json(manifest, manifest_path)
        return {"status": "blocked_preflight", "preflight": audit}
    if prepare_only:
        manifest["status"] = "prepared_and_preflight_passed"
        manifest["preflight"] = audit
        g0.atomic_write_json(manifest, manifest_path)
        return {
            "status": "prepared_and_preflight_passed",
            "cohort": cohort,
            "profiles": len(manifest["commands"]),
            "manifest": str(manifest_path.resolve()),
        }
    returncode = g22f.g7f.run_manifest(manifest, manifest_path)
    if returncode:
        return {"status": "profile_failure", "returncode": returncode}
    manifest["status"] = "profiles_completed"
    manifest["finished_at_utc"] = g0.utc_now()
    g0.atomic_write_json(manifest, manifest_path)
    result = score_run(manifest, record_dir=manifest_path.parent)
    manifest["status"] = result["status"]
    manifest["result"] = result
    g0.atomic_write_json(manifest, manifest_path)
    return result


def joint_review(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if {item.get("cohort") for item in results} != {"normal", "meme"}:
        raise ValueError("Generation 24 FreqAI joint review requires both cohorts.")
    frames = [pd.read_csv(item["artifacts"]["decisions"]["path"]) for item in results]
    decisions = pd.concat(frames, ignore_index=True)
    record_dir = RECORD_ROOT / DEFAULT_JOINT_REVIEW_ID
    record_dir.mkdir(parents=True, exist_ok=True)
    decision_path = record_dir / "g24_freqai_joint_decisions.csv"
    g0.atomic_write_csv(decisions, decision_path)
    retained = int(decisions["all_model_seed_cells_pass"].sum())
    output = {
        "schema_version": 1,
        "generation": 24,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation24_freqai_joint_review",
        "both_cohorts_completed_before_review": True,
        "exploratory_activity_robustness_leads": retained,
        "not_retained": len(decisions) - retained,
        "research_boundary": {
            "profit_used": False,
            "signed_direction_used": False,
            "trading_promotion": False,
        },
        "source_results": [artifact(Path(item["result_path"])) for item in results],
        "artifacts": {"joint_decisions": artifact(decision_path)},
    }
    result_path = record_dir / "g24_freqai_joint_review.json"
    g0.atomic_write_json(output, result_path)
    return {**output, "result_path": str(result_path.resolve())}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("normal", "meme", "all"), default="all")
    parser.add_argument("--run-stem", default=DEFAULT_RUN_STEM)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--profile-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--technical-smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    cohorts = ("normal", "meme") if args.cohort == "all" else (args.cohort,)
    results = []
    for cohort in cohorts:
        results.append(
            run_cell(
                run_id=f"{args.run_stem}_{cohort}",
                cohort=cohort,
                base_config=args.base_config,
                python_exe=args.python_exe,
                profile_workers=args.profile_workers,
                technical_smoke=args.technical_smoke,
                prepare_only=args.prepare_only,
            )
        )
    if (
        len(results) == 2
        and not args.prepare_only
        and not args.technical_smoke
        and all(item.get("status") == "completed_generation24_freqai_cohort" for item in results)
    ):
        results.append(joint_review(results))
    print(json.dumps(results, indent=2))
    return (
        0 if all(not str(item.get("status", "")).startswith("blocked") for item in results) else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

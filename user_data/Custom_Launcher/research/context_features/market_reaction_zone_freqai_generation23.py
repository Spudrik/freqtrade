"""Run Generation 23's frozen pair-calibrated unsigned-reaction FreqAI route."""

from __future__ import annotations

# Bound numerical and model pools before pandas/FreqAI imports.
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
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_freqai_cache as g22c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_freeze as g23z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_freqai_cache as g23c,
)


ANALYSIS_PATH = Path(__file__).resolve()
DEFAULT_CONFIG = g22f.DEFAULT_CONFIG
DEFAULT_PYTHON = g22f.DEFAULT_PYTHON
STRATEGY_PATH = g22f.STRATEGY_PATH
STRATEGY_FILE = STRATEGY_PATH / "MarketReactionZoneFreqAIGeneration23Strategy.py"
STRATEGY_CLASS = "MarketReactionZoneG23ConfigurableFreqAIResearchStrategy"
DATA_DIR = g22f.DATA_DIR
RECORD_ROOT = g23z.OUTPUT_ROOT / "freqai"
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation23_branches"
    / "g23_broad_siblings"
    / "freqai"
)
DEFAULT_RUN_STEM = "g23_calibrated_unsigned_reaction_freqai_20260828a"
DEFAULT_SMOKE_STEM = "g23_calibrated_unsigned_reaction_freqai_smoke_20260828a"
DEFAULT_JOINT_REVIEW_ID = "g23_calibrated_unsigned_reaction_freqai_joint_20260828a"
MAX_WORKERS = 4
MIN_TRAINING_ROWS = g22f.MIN_TRAINING_ROWS
MIN_PREDICTION_ROWS = g22f.MIN_PREDICTION_ROWS
MIN_PAIR_SCORE_ROWS = g22f.MIN_PAIR_SCORE_ROWS
TARGET_PURGE_HOURS = max(g23c.HORIZONS)
PAIR_METRICS = tuple(g22f.PAIR_METRICS)
MODEL_CONTROLS = (
    "level_geometry_only",
    "market_state_only",
    "within_pair_time_shuffled_training_labels",
    "generation22_uncalibrated_parent",
)
ALL_CONTROLS = ("constant_training_median", *MODEL_CONTROLS)


def artifact(path: Path) -> dict[str, Any]:
    return g23z.artifact(path)


def cohort_settings(cohort: str, *, technical_smoke: bool) -> dict[str, Any]:
    return g22f.cohort_settings(cohort, technical_smoke=technical_smoke)


def parent_paths(cohort: str) -> tuple[Path, Path]:
    root = g22f.RECORD_ROOT / f"{g22f.DEFAULT_RUN_STEM}_{cohort}"
    return root / "g22_freqai_run_manifest.json", root / "g22_freqai_result.json"


def load_parent_run(cohort: str) -> tuple[dict[str, Any], Path, Path]:
    manifest_path, result_path = parent_paths(cohort)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed_generation22_freqai_cohort":
        raise ValueError(f"Generation 22 {cohort} FreqAI parent is not terminal.")
    if manifest.get("status") != "completed_generation22_freqai_cohort":
        raise ValueError(f"Generation 22 {cohort} FreqAI manifest is not terminal.")
    full = [item for item in manifest["commands"] if item["role"] == "full_interaction"]
    if len(full) != 1 or full[0].get("status") != "completed":
        raise ValueError(f"Generation 22 {cohort} full model is unavailable.")
    return manifest, manifest_path, result_path


def load_sources(
    cohort: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    frozen, _ = g23c.load_branch()
    cache_path = g23c.cache_manifest_path(cohort)
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    registry = json.loads(g23c.REGISTRY_PATH.read_text(encoding="utf-8"))
    if cache.get("status") != "completed_generation23_freqai_cache":
        raise ValueError(f"Generation 23 {cohort} FreqAI cache is not terminal.")
    if registry.get("status") != "frozen_before_generation23_freqai_target_materialization":
        raise ValueError("Generation 23 FreqAI registry is invalid.")
    if g0.sha256_file(g23c.REGISTRY_PATH) != cache["profile_registry"]["sha256"]:
        raise ValueError("Generation 23 FreqAI registry changed after target construction.")
    return frozen, cache, registry


def active_registry(
    registry: dict[str, Any], cohort: str
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    profiles = {
        identifier: dict(profile)
        for identifier, profile in registry["profiles"].items()
        if profile["cohort"] == cohort
    }
    comparisons = [
        dict(item) for item in registry["comparisons"] if item["cohort"] == cohort
    ]
    if set(profile["role"] for profile in profiles.values()) != set(g23c.PROFILE_ROLES):
        raise ValueError(f"Generation 23 {cohort} profile ladder is incomplete.")
    if len(comparisons) != 5:
        raise ValueError(f"Generation 23 {cohort} control ladder is incomplete.")
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
    config = g22f.profile_config(
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
    config["market_reaction_zone_g23"] = config.pop("market_reaction_zone_g22")
    return config


def event_dirs(cache: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    item = cache["inventory"][0]
    return (
        Path(item["feature_path"]).parent,
        Path(item["event_path"]).parent,
        Path(item["shuffled_event_path"]).parent,
        Path(item["evaluation_path"]).parent,
    )


def training_thresholds(
    *, cohort: str, cache: dict[str, Any], settings: dict[str, Any]
) -> DataFrame:
    _, event_dir, _, _ = event_dirs(cache)
    rows: list[dict[str, Any]] = []
    for pair in cache["pairs"]:
        frame = pd.read_parquet(
            event_dir / f"{g0.pair_file_stem(str(pair))}.parquet",
            columns=["date", f"ready__{g23c.READY_BLOCK}", *g23c.TARGETS],
        )
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
        ready = frame[f"ready__{g23c.READY_BLOCK}"].fillna(False).astype(bool)
        for period in settings["period_bounds"]:
            start = pd.Timestamp(period["start_utc"])
            cutoff = start - pd.Timedelta(hours=TARGET_PURGE_HOURS)
            training_start = start - pd.Timedelta(days=int(settings["train_days"]))
            time_mask = ready & frame["date"].ge(training_start) & frame["date"].lt(cutoff)
            for target in g23c.TARGETS:
                values = pd.to_numeric(frame.loc[time_mask, target], errors="coerce").dropna()
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
    profiles, comparisons = active_registry(registry, cohort)
    parent, parent_manifest_path, parent_result_path = load_parent_run(cohort)
    settings = cohort_settings(cohort, technical_smoke=technical_smoke)
    record_dir = RECORD_ROOT / run_id
    artifact_dir = ARTIFACT_ROOT / run_id
    manifest_path = record_dir / "g23_freqai_run_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, path in (
            ("strategy", STRATEGY_FILE),
            ("freeze", g23z.FREEZE_PATH),
            ("cache_manifest", g23c.cache_manifest_path(cohort)),
            ("profile_registry", g23c.REGISTRY_PATH),
            ("analysis_script", ANALYSIS_PATH),
            ("generation22_parent_manifest", parent_manifest_path),
            ("generation22_parent_result", parent_result_path),
        ):
            if g0.sha256_file(path) != existing["source_contracts"][name]["sha256"]:
                raise ValueError(f"Generation 23 {name} changed: {manifest_path}")
        return existing, manifest_path
    record_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    base = json.loads(base_config.read_text(encoding="utf-8"))
    feature_dir, actual_event_dir, shuffled_event_dir, evaluation_dir = event_dirs(cache)
    pairs = tuple(str(pair) for pair in cache["pairs"])
    if technical_smoke:
        pairs = pairs[:1]
    thresholds = training_thresholds(cohort=cohort, cache=cache, settings=settings)
    thresholds = thresholds.loc[thresholds["pair"].isin(pairs)].copy()
    threshold_path = record_dir / "g23_training_median_thresholds.csv"
    g0.atomic_write_csv(thresholds, threshold_path)
    commands: list[dict[str, Any]] = []
    for number, (profile_identifier, profile) in enumerate(profiles.items(), start=1):
        short_id = f"p{number:03d}_{g22f.g6f.stable_digest(profile_identifier, 6)}"
        profile_dir = artifact_dir / "profiles" / short_id
        userdir = profile_dir / "user_data"
        export_dir = profile_dir / "export"
        identifier = f"g23-{g22f.g6f.stable_digest(f'{run_id}|{profile_identifier}', 16)}"
        config_path = record_dir / f"config_{short_id}.json"
        selected_event_dir = (
            shuffled_event_dir if profile["target_cache"] == "shuffled" else actual_event_dir
        )
        g0.atomic_write_json(
            profile_config(
                base,
                identifier=identifier,
                pairs=pairs,
                feature_dir=feature_dir,
                event_dir=selected_event_dir,
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
            "LightGBMRegressorMultiTarget",
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
                "event_cache_dir": str(selected_event_dir.resolve()),
                "command": command,
                "status": "pending",
                "attempts": 0,
            }
        )
    _, branch = g23c.load_branch()
    manifest = {
        "schema_version": 1,
        "generation": 23,
        "run_id": run_id,
        "created_at_utc": g0.utc_now(),
        "status": "prepared",
        "run_stage": "pair_calibrated_unsigned_reaction_regression",
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
        "model_class": "LightGBMRegressorMultiTarget",
        "profiles": list(profiles),
        "profile_count": len(profiles),
        "comparisons": comparisons,
        "comparison_count": len(comparisons),
        "targets": list(g23c.TARGETS),
        "target_metadata": list(g23c.TARGET_METADATA),
        "maximum_target_horizon_hours": TARGET_PURGE_HOURS,
        "decision_rule": {
            "plain_rule": branch["pass_rule"],
            "constant_control": "full model must have lower equal-coin MAE",
            "model_controls": (
                "full model must have lower MAE, higher Spearman rank correlation, "
                "and larger top-minus-bottom separation"
            ),
            "candidate_requires_positive_separation_each_block": True,
            "candidate_top_quartile_above_training_median_minimum": 0.55,
            "both_later_blocks_required": True,
        },
        "research_boundary": dict(frozen["research_boundary"]),
        "source_contracts": {
            "base_config": artifact(base_config),
            "python_executable": str(python_exe.resolve()),
            "strategy": artifact(STRATEGY_FILE),
            "freeze": artifact(g23z.FREEZE_PATH),
            "cache_manifest": artifact(g23c.cache_manifest_path(cohort)),
            "profile_registry": artifact(g23c.REGISTRY_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
            "training_median_thresholds": artifact(threshold_path),
            "target_calibration": artifact(g23c.calibration_path(cohort)),
            "generation22_parent_manifest": artifact(parent_manifest_path),
            "generation22_parent_result": artifact(parent_result_path),
            "generation22_raw_cache": artifact(g22c.cache_manifest_path(cohort)),
            "cache_inventory": cache["inventory"],
            "generation22_parent_full_profile": next(
                item for item in parent["commands"] if item["role"] == "full_interaction"
            ),
        },
        "storage": {
            "record_dir": str(record_dir.resolve()),
            "bulky_artifact_dir": str(artifact_dir.resolve()),
            "feature_cache_dir": str(feature_dir.resolve()),
            "event_cache_dir": str(actual_event_dir.resolve()),
            "shuffled_event_cache_dir": str(shuffled_event_dir.resolve()),
            "evaluation_cache_dir": str(evaluation_dir.resolve()),
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
            "import freqtrade, lightgbm, sklearn; print(freqtrade.__version__)",
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
        ["The worker cannot import Freqtrade, LightGBM, and sklearn."]
        if check.returncode
        else []
    )
    return dependency, problems


def preflight_run(manifest: dict[str, Any], *, python_exe: Path) -> dict[str, Any]:
    dependency, problems = dependency_check(python_exe)
    if not STRATEGY_FILE.is_file():
        problems.append(f"Missing Generation 23 strategy: {STRATEGY_FILE}")
    required_features = {
        column for command in manifest["commands"] for column in command["feature_columns"]
    }
    start_raw, end_raw = str(manifest["timerange"]).split("-", maxsplit=1)
    prediction_start = pd.Timestamp(start_raw, tz="UTC")
    prediction_end = pd.Timestamp(end_raw, tz="UTC")
    training_start = prediction_start - pd.Timedelta(
        days=int(manifest["train_period_days"])
    )
    readiness: list[dict[str, Any]] = []
    inventory_by_pair = {
        str(item["pair"]): item for item in manifest["source_contracts"]["cache_inventory"]
    }
    for pair in manifest["pairs"]:
        item = inventory_by_pair[pair]
        for key in ("feature", "support", "event", "evaluation", "shuffled_event"):
            path = Path(item[f"{key}_path"])
            if not path.is_file() or g0.sha256_file(path) != item[f"{key}_sha256"]:
                problems.append(f"Cache hash mismatch for {pair} and {key}.")
        try:
            pd.read_parquet(item["feature_path"], columns=sorted(required_features))
        except Exception as exc:
            problems.append(f"{pair} lacks exact Generation 23 features: {exc}")
        for label_source, path_key in (
            ("actual", "event_path"),
            ("shuffled", "shuffled_event_path"),
        ):
            frame = pd.read_parquet(
                item[path_key],
                columns=["date", f"ready__{g23c.READY_BLOCK}", *g23c.TARGETS],
            )
            frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
            eligible = frame[f"ready__{g23c.READY_BLOCK}"].fillna(False).astype(bool)
            eligible &= frame[list(g23c.TARGETS)].notna().all(axis=1)
            training_rows = int(
                (
                    eligible
                    & frame["date"].ge(training_start)
                    & frame["date"].lt(prediction_start)
                ).sum()
            )
            prediction_rows = int(
                (
                    eligible
                    & frame["date"].ge(prediction_start)
                    & frame["date"].lt(prediction_end)
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
    parent = manifest["source_contracts"]["generation22_parent_full_profile"]
    if parent.get("status") != "completed" or not Path(parent["model_dir"]).is_dir():
        problems.append("Generation 22 parent prediction model is unavailable.")
    free_gib = shutil.disk_usage(Path(manifest["storage"]["bulky_artifact_dir"])).free / (
        1024**3
    )
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
    event_dir = Path(manifest["storage"]["event_cache_dir"])
    for pair in manifest["pairs"]:
        frame = pd.read_parquet(
            event_dir / f"{g0.pair_file_stem(pair)}.parquet",
            columns=["date", "period", *g23c.TARGETS],
        )
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
        frame["pair"] = pair
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    if output.duplicated(["pair", "date"]).any():
        raise ValueError("Duplicate pair/date rows in Generation 23 target cache.")
    return output


def calibrated_parent_predictions(manifest: dict[str, Any]) -> tuple[DataFrame, dict[str, Any]]:
    parent_profile = manifest["source_contracts"]["generation22_parent_full_profile"]
    raw, audit = g0f.load_predictions(
        Path(parent_profile["model_dir"]), tuple(manifest["pairs"])
    )
    missing = sorted(set(g23c.RAW_TARGETS).difference(raw.columns))
    if missing:
        raise ValueError(f"Generation 22 parent predictions lack targets: {missing}")
    parent_cache = json.loads(
        Path(manifest["source_contracts"]["generation22_raw_cache"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    raw_by_pair = {str(item["pair"]): item for item in parent_cache["inventory"]}
    parts: list[DataFrame] = []
    for pair, cell in raw.groupby("pair", observed=True, sort=False):
        source = pd.read_parquet(raw_by_pair[str(pair)]["event_path"])
        dates = pd.to_datetime(source["date"], utc=True, errors="raise")
        ready = source[f"ready__{g23c.READY_BLOCK}"].fillna(False).astype(bool)
        start, cutoff = g23c.calibration_bounds(manifest["cohort"])
        mask = ready & dates.ge(start) & dates.lt(cutoff)
        output = cell[["pair", "date"]].copy()
        for raw_target in g23c.RAW_TARGETS:
            reference = pd.to_numeric(source.loc[mask, raw_target], errors="coerce").dropna()
            median = float(reference.median())
            values = pd.to_numeric(cell[raw_target], errors="coerce")
            output[
                g23c.target_name("pair_training_median_residual", raw_target)
            ] = values - median
            output[
                g23c.target_name("pair_training_empirical_percentile", raw_target)
            ] = g23c.empirical_percentile(values, reference)
        parts.append(output)
    return pd.concat(parts, ignore_index=True), audit


def prediction_metrics(actual: Series, prediction: Series, threshold: float) -> dict[str, Any]:
    return g22f.prediction_metrics(actual, prediction, threshold)


def pair_scores(
    *,
    manifest: dict[str, Any],
    predictions: dict[str, DataFrame],
    actual: DataFrame,
) -> tuple[DataFrame, DataFrame]:
    common, audit = g0f.common_prediction_keys(predictions, tuple(manifest["pairs"]))
    fair = common.merge(actual, on=["pair", "date"], how="left", validate="one_to_one")
    thresholds = pd.read_csv(manifest["storage"]["training_median_thresholds"])
    threshold_lookup = thresholds.set_index(["pair", "period", "target"])[
        "training_median"
    ].to_dict()
    role_lookup = {
        str(item["profile_id"]): str(item["role"]) for item in manifest["commands"]
    }
    role_lookup["generation22_uncalibrated_parent"] = "generation22_uncalibrated_parent"
    rows: list[dict[str, Any]] = []
    for profile_id, prediction in predictions.items():
        columns = ["pair", "date", *g23c.TARGETS]
        renamed = prediction[columns].rename(
            columns={target: f"prediction__{target}" for target in g23c.TARGETS}
        )
        merged = fair.merge(renamed, on=["pair", "date"], how="left", validate="one_to_one")
        for (pair, period), cell in merged.groupby(
            ["pair", "period"], observed=True, sort=False
        ):
            if period not in manifest["validation_periods"]:
                continue
            for target in g23c.TARGETS:
                threshold = float(threshold_lookup.get((pair, period, target), np.nan))
                rows.append(
                    {
                        "cohort": manifest["cohort"],
                        "pair": pair,
                        "period": period,
                        "profile_id": profile_id,
                        "role": role_lookup[str(profile_id)],
                        "target": target,
                        "training_median": threshold,
                        **prediction_metrics(
                            cell[target], cell[f"prediction__{target}"], threshold
                        ),
                    }
                )
    for (pair, period), cell in fair.groupby(["pair", "period"], observed=True, sort=False):
        if period not in manifest["validation_periods"]:
            continue
        for target in g23c.TARGETS:
            threshold = float(threshold_lookup.get((pair, period, target), np.nan))
            rows.append(
                {
                    "cohort": manifest["cohort"],
                    "pair": pair,
                    "period": period,
                    "profile_id": "constant_training_median",
                    "role": "constant_training_median",
                    "target": target,
                    "training_median": threshold,
                    **prediction_metrics(
                        cell[target], Series(threshold, index=cell.index), threshold
                    ),
                }
            )
    return DataFrame.from_records(rows), audit


def scope_scores(pair_frame: DataFrame) -> DataFrame:
    return g22f.scope_scores(pair_frame)


def target_metadata(target: str) -> dict[str, str]:
    item = next(entry for entry in g23c.TARGET_METADATA if entry["target"] == target)
    return {
        "target_transform": str(item["transform"]),
        "raw_target": str(item["raw_target"]),
    }


def control_comparisons(
    scope_frame: DataFrame, validation_periods: Sequence[str]
) -> tuple[DataFrame, DataFrame]:
    rows: list[dict[str, Any]] = []
    for (cohort, scope, period, target), cell in scope_frame.groupby(
        ["cohort", "market_scope", "period", "target"], observed=True, sort=False
    ):
        indexed = cell.set_index("role")
        if "full_interaction" not in indexed.index:
            continue
        candidate = indexed.loc["full_interaction"]
        candidate_gate = bool(
            candidate["coin_support_pass"]
            and candidate["top_minus_bottom_difference"] > 0.0
            and candidate["top_quartile_above_training_median_fraction"] >= 0.55
        )
        metadata = target_metadata(str(target))
        for control in ALL_CONTROLS:
            present = control in indexed.index
            baseline = indexed.loc[control] if present else Series(dtype=object)
            if control == "constant_training_median":
                metrics_pass = bool(
                    present
                    and candidate["median_absolute_error"]
                    < baseline["median_absolute_error"]
                )
            else:
                metrics_pass = bool(
                    present
                    and candidate["median_absolute_error"]
                    < baseline["median_absolute_error"]
                    and candidate["spearman_rank_correlation"]
                    > baseline["spearman_rank_correlation"]
                    and candidate["top_minus_bottom_difference"]
                    > baseline["top_minus_bottom_difference"]
                )
            rows.append(
                {
                    "cohort": cohort,
                    "market_scope": scope,
                    "period": period,
                    "target": target,
                    **metadata,
                    "control": control,
                    "control_present": present,
                    "candidate_gate_pass": candidate_gate,
                    "candidate_mae": candidate["median_absolute_error"],
                    "control_mae": baseline.get("median_absolute_error", np.nan),
                    "candidate_spearman": candidate["spearman_rank_correlation"],
                    "control_spearman": baseline.get(
                        "spearman_rank_correlation", np.nan
                    ),
                    "candidate_top_minus_bottom": candidate[
                        "top_minus_bottom_difference"
                    ],
                    "control_top_minus_bottom": baseline.get(
                        "top_minus_bottom_difference", np.nan
                    ),
                    "candidate_top_above_training_median_fraction": candidate[
                        "top_quartile_above_training_median_fraction"
                    ],
                    "control_metrics_pass": metrics_pass,
                    "period_control_pass": bool(candidate_gate and metrics_pass),
                }
            )
    comparisons = DataFrame.from_records(rows)
    decision_rows: list[dict[str, Any]] = []
    expected_periods = set(validation_periods)
    for (cohort, scope, target), cell in comparisons.groupby(
        ["cohort", "market_scope", "target"], observed=True, sort=False
    ):
        controls = set(cell["control"])
        periods = set(cell["period"])
        complete = controls == set(ALL_CONTROLS) and periods == expected_periods
        retained = bool(complete and cell["period_control_pass"].all())
        decision_rows.append(
            {
                "cohort": cohort,
                "market_scope": scope,
                "target": target,
                **target_metadata(str(target)),
                "status": (
                    "exploratory_unsigned_reaction_lead"
                    if retained
                    else "not_retained_across_both_later_blocks"
                ),
                "complete_control_and_period_ladder": complete,
                "all_controls_both_periods_pass": retained,
                "minimum_candidate_top_above_training_median_fraction": float(
                    cell["candidate_top_above_training_median_fraction"].min()
                ),
                "minimum_candidate_top_minus_bottom": float(
                    cell["candidate_top_minus_bottom"].min()
                ),
            }
        )
    return comparisons, DataFrame.from_records(decision_rows)


def score_run(manifest: dict[str, Any], *, record_dir: Path) -> dict[str, Any]:
    predictions: dict[str, DataFrame] = {}
    audits: list[dict[str, Any]] = []
    for item in manifest["commands"]:
        frame, audit = g0f.load_predictions(Path(item["model_dir"]), tuple(item["pairs"]))
        audit["profile_id"] = item["profile_id"]
        audit["role"] = item["role"]
        audits.append(audit)
        if frame.empty:
            raise ValueError(f"No Generation 23 predictions for {item['profile_id']}")
        missing = sorted(set(g23c.TARGETS).difference(frame.columns))
        if missing:
            raise ValueError(f"Profile {item['profile_id']} lacks targets: {missing}")
        predictions[str(item["profile_id"])] = frame
    parent, parent_audit = calibrated_parent_predictions(manifest)
    parent_audit["profile_id"] = "generation22_uncalibrated_parent"
    parent_audit["role"] = "generation22_uncalibrated_parent"
    audits.append(parent_audit)
    predictions["generation22_uncalibrated_parent"] = parent
    actual = load_actual(manifest)
    pair, eligibility = pair_scores(
        manifest=manifest, predictions=predictions, actual=actual
    )
    scopes = scope_scores(pair)
    comparisons, decisions = control_comparisons(scopes, manifest["validation_periods"])
    paths = {
        "prediction_audit": record_dir / "g23_prediction_audit.csv",
        "prediction_eligibility": record_dir / "g23_prediction_eligibility.csv",
        "pair_scores": record_dir / "g23_pair_scores.csv",
        "scope_scores": record_dir / "g23_scope_scores.csv",
        "control_comparisons": record_dir / "g23_control_comparisons.csv",
        "decisions": record_dir / "g23_freqai_decisions.csv",
    }
    g0.atomic_write_csv(DataFrame.from_records(audits), paths["prediction_audit"])
    g0.atomic_write_csv(eligibility, paths["prediction_eligibility"])
    g0.atomic_write_csv(pair, paths["pair_scores"])
    g0.atomic_write_csv(scopes, paths["scope_scores"])
    g0.atomic_write_csv(comparisons, paths["control_comparisons"])
    g0.atomic_write_csv(decisions, paths["decisions"])
    retained = int(decisions["all_controls_both_periods_pass"].sum())
    result = {
        "schema_version": 1,
        "generation": 23,
        "run_id": manifest["run_id"],
        "created_at_utc": g0.utc_now(),
        "status": (
            "completed_generation23_freqai_technical_smoke_not_evidence"
            if manifest["technical_smoke_not_evidence"]
            else "completed_generation23_freqai_cohort"
        ),
        "technical_smoke_not_evidence": manifest["technical_smoke_not_evidence"],
        "cohort": manifest["cohort"],
        "profiles_completed": len(manifest["commands"]),
        "targets_scored": len(g23c.TARGETS),
        "exploratory_unsigned_reaction_leads": retained,
        "not_retained": len(decisions) - retained,
        "integrity": {
            "real_freqai_model_class": manifest["model_class"],
            "all_profile_commands_terminal": all(
                item.get("status") == "completed" for item in manifest["commands"]
            ),
            "generation22_parent_predictions_rescaled_before_comparison": True,
            "profit_used": False,
            "future_signed_direction_used": False,
            "auc_used": False,
        },
        "artifacts": {key: artifact(path) for key, path in paths.items()},
    }
    result_path = record_dir / "g23_freqai_result.json"
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
    g0.atomic_write_json(audit, manifest_path.parent / "g23_freqai_preflight.json")
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
        raise ValueError("Generation 23 FreqAI joint review requires both cohorts.")
    frames = [pd.read_csv(item["artifacts"]["decisions"]["path"]) for item in results]
    decisions = pd.concat(frames, ignore_index=True)
    record_dir = RECORD_ROOT / DEFAULT_JOINT_REVIEW_ID
    record_dir.mkdir(parents=True, exist_ok=True)
    decision_path = record_dir / "g23_freqai_joint_decisions.csv"
    g0.atomic_write_csv(decisions, decision_path)
    retained = int(decisions["all_controls_both_periods_pass"].sum())
    output = {
        "schema_version": 1,
        "generation": 23,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation23_freqai_joint_review",
        "both_cohorts_completed_before_review": True,
        "exploratory_unsigned_reaction_leads": retained,
        "not_retained": len(decisions) - retained,
        "research_boundary": {
            "profit_used": False,
            "signed_direction_used": False,
            "trading_promotion": False,
        },
        "source_results": [artifact(Path(item["result_path"])) for item in results],
        "artifacts": {"joint_decisions": artifact(decision_path)},
    }
    result_path = record_dir / "g23_freqai_joint_review.json"
    g0.atomic_write_json(output, result_path)
    return {**output, "result_path": str(result_path.resolve())}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("all", "normal", "meme"), default="all")
    parser.add_argument("--run-stem")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--profile-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--technical-smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.base_config.is_file():
        raise FileNotFoundError(args.base_config)
    if not args.python_exe.is_file():
        raise FileNotFoundError(args.python_exe)
    cohorts = ("normal", "meme") if args.cohort == "all" else (args.cohort,)
    stem = args.run_stem or (
        DEFAULT_SMOKE_STEM if args.technical_smoke else DEFAULT_RUN_STEM
    )
    results: list[dict[str, Any]] = []
    for cohort in cohorts:
        result = run_cell(
            run_id=f"{stem}_{cohort}",
            cohort=cohort,
            base_config=args.base_config,
            python_exe=args.python_exe,
            profile_workers=max(1, min(args.profile_workers, MAX_WORKERS)),
            technical_smoke=args.technical_smoke,
            prepare_only=args.prepare_only,
        )
        results.append(result)
        if not str(result["status"]).startswith(
            ("completed_", "prepared_and_preflight_passed")
        ):
            print(json.dumps(results, indent=2, default=g0.json_default), flush=True)
            return 2
    if len(results) == 2 and not args.technical_smoke and not args.prepare_only:
        results.append(joint_review(results))
    print(json.dumps(results, indent=2, default=g0.json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

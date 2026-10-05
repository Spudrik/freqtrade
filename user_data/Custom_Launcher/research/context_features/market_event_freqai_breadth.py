"""Run the frozen event/market/level breadth comparison through actual FreqAI."""

from __future__ import annotations

# Bound model and dataframe pools before numerical imports.
# ruff: noqa: E402
import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
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
    market_event_freqai_breadth_cache as cache,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_freqai_breadth_freeze as frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation0 as g0f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation22 as g22f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
DEFAULT_CONFIG = g22f.DEFAULT_CONFIG
DEFAULT_PYTHON = g22f.DEFAULT_PYTHON
STRATEGY_PATH = g22f.STRATEGY_PATH
STRATEGY_FILE = STRATEGY_PATH / "MarketEventFreqAIBreadthStrategy.py"
STRATEGY_CLASS = "MarketEventFreqAIBreadthResearchStrategy"
DATA_DIR = g22f.DATA_DIR
RECORD_ROOT = frozen.OUTPUT_ROOT / "freqai"
ARTIFACT_ROOT = Path(
    "D:/FreqTradeStuffLargeData/research_outputs/event_hierarchy/"
    "event_freqai_breadth_20260908a/freqai"
)
DEFAULT_RUN_ID = "event_freqai_breadth_20260908a"
DEFAULT_SMOKE_ID = "event_freqai_breadth_smoke_20260908a"
MAX_WORKERS = 4
MIN_TRAINING_ROWS = 600
MIN_PREDICTION_ROWS = 100
MIN_PAIR_SCORE_ROWS = 20
TARGET_PURGE_HOURS = max(frozen.HORIZONS)
FULL_SETTINGS = {
    "timerange": "20240101-20260101",
    "train_days": 900,
    "backtest_days": 180,
    "validation_periods": [
        "walk_forward_validation_2024",
        "walk_forward_validation_2025",
    ],
}
SMOKE_SETTINGS = {
    "timerange": "20251001-20260101",
    "train_days": 900,
    "backtest_days": 92,
    "validation_periods": ["walk_forward_validation_2025"],
}
PERIOD_BOUNDS = {
    "walk_forward_validation_2024": (
        pd.Timestamp("2024-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-01T00:00:00Z"),
    ),
    "walk_forward_validation_2025": (
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-01T00:00:00Z"),
    ),
}
SAMPLE_SCOPES = ("all_catalog", "actual_events", "matched_controls")
AGGREGATE_METRICS = (
    "median_absolute_error",
    "spearman_rank_correlation",
    "top_quartile_actual_reaction_mean",
    "bottom_quartile_actual_reaction_mean",
    "top_minus_bottom_difference",
    "top_quartile_above_reference_fraction",
    "correct_side_rate",
    "training_majority_correct_rate",
    "correct_minus_training_majority",
    "actual_positive_rate",
    "predicted_positive_rate",
)


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def settings(*, technical_smoke: bool) -> dict[str, Any]:
    return dict(SMOKE_SETTINGS if technical_smoke else FULL_SETTINGS)


def load_sources() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    freeze = frozen.load_freeze()
    manifest = _load_json(cache.CACHE_MANIFEST_PATH)
    registry = _load_json(frozen.REGISTRY_PATH)
    if manifest.get("status") != "completed_event_freqai_cache":
        raise ValueError("Event breadth cache is not terminal.")
    if registry.get("status") != "frozen_before_event_freqai_outcome_materialization":
        raise ValueError("Event breadth profile registry is not frozen.")
    if g0.sha256_file(frozen.FREEZE_PATH) != manifest["source_contracts"][
        "breadth_freeze"
    ]["sha256"]:
        raise ValueError("Event breadth freeze changed after cache materialization.")
    if g0.sha256_file(frozen.REGISTRY_PATH) != manifest["source_contracts"][
        "profile_registry"
    ]["sha256"]:
        raise ValueError("Event breadth profile registry changed after caching.")
    return freeze, manifest, registry


def active_profiles(
    registry: Mapping[str, Any], *, technical_smoke: bool
) -> dict[str, dict[str, Any]]:
    profiles = {
        str(profile_id): dict(profile)
        for profile_id, profile in registry["profiles"].items()
    }
    expected = set(frozen.PROFILE_DEFINITIONS)
    if set(profiles) != expected or len(profiles) != 14:
        raise ValueError("The frozen fourteen-profile breadth surface is incomplete.")
    if technical_smoke:
        return {"event_plus_recent_market": profiles["event_plus_recent_market"]}
    return profiles


def cache_dirs(manifest: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    item = manifest["inventory"][0]
    return (
        Path(item["feature_path"]).parent,
        Path(item["event_path"]).parent,
        Path(item["evaluation_path"]).parent,
    )


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
    config["market_event_freqai_breadth"] = config.pop(
        "market_reaction_zone_g22"
    )
    config["freqai"]["model_training_parameters"]["n_jobs"] = 1
    return config


def build_training_reference(
    *,
    manifest: Mapping[str, Any],
    pairs: Sequence[str],
    train_days: int,
) -> DataFrame:
    _, event_dir, _ = cache_dirs(manifest)
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        frame = pd.read_parquet(
            event_dir / f"{g0.pair_file_stem(pair)}.parquet",
            columns=["date", f"ready__{frozen.READY_BLOCK}", *frozen.TARGETS],
        )
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
        ready = frame[f"ready__{frozen.READY_BLOCK}"].fillna(False).astype(bool)
        for period, (start, _end) in PERIOD_BOUNDS.items():
            cutoff = start - pd.Timedelta(hours=TARGET_PURGE_HOURS)
            training_start = start - pd.Timedelta(days=train_days)
            mask = ready & frame["date"].ge(training_start) & frame["date"].lt(cutoff)
            for target in frozen.TARGETS:
                values = pd.to_numeric(frame.loc[mask, target], errors="coerce").dropna()
                median = float(values.median()) if len(values) else np.nan
                reference = 0.0 if target_kind(target) == "direction" else median
                positive_rate = (
                    float(values.gt(reference).mean()) if len(values) else np.nan
                )
                rows.append(
                    {
                        "pair": pair,
                        "period": period,
                        "target": target,
                        "training_start_utc": training_start,
                        "training_cutoff_exclusive_utc": cutoff,
                        "training_rows": len(values),
                        "training_median": median,
                        "training_upper_quartile": (
                            float(values.quantile(0.75)) if len(values) else np.nan
                        ),
                        "training_median_absolute": (
                            float(values.abs().median()) if len(values) else np.nan
                        ),
                        "training_positive_rate": positive_rate,
                        "training_majority_positive": bool(positive_rate >= 0.5)
                        if np.isfinite(positive_rate)
                        else False,
                    }
                )
    return DataFrame.from_records(rows)


def build_manifest(
    *,
    run_id: str,
    base_config: Path,
    python_exe: Path,
    profile_workers: int,
    technical_smoke: bool,
) -> tuple[dict[str, Any], Path]:
    freeze, cache_manifest, registry = load_sources()
    selected_settings = settings(technical_smoke=technical_smoke)
    profiles = active_profiles(registry, technical_smoke=technical_smoke)
    record_dir = RECORD_ROOT / run_id
    artifact_dir = ARTIFACT_ROOT / run_id
    manifest_path = record_dir / "event_freqai_run_manifest.json"
    if manifest_path.is_file():
        existing = _load_json(manifest_path)
        for name, path in (
            ("base_config", base_config),
            ("strategy", STRATEGY_FILE),
            ("breadth_freeze", frozen.FREEZE_PATH),
            ("cache_manifest", cache.CACHE_MANIFEST_PATH),
            ("profile_registry", frozen.REGISTRY_PATH),
            ("analysis_script", ANALYSIS_PATH),
        ):
            if g0.sha256_file(path) != existing["source_contracts"][name]["sha256"]:
                raise ValueError(f"Event breadth {name} changed after run preparation.")
        return existing, manifest_path
    record_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    feature_dir, event_dir, evaluation_dir = cache_dirs(cache_manifest)
    pairs = tuple(str(pair) for pair in cache_manifest["pairs"])
    if technical_smoke:
        pairs = pairs[:1]
    training_reference = build_training_reference(
        manifest=cache_manifest,
        pairs=pairs,
        train_days=int(selected_settings["train_days"]),
    )
    reference_path = record_dir / "event_freqai_training_reference.csv"
    g0.atomic_write_csv(training_reference, reference_path)
    base = _load_json(base_config)
    commands: list[dict[str, Any]] = []
    for number, (profile_id, profile) in enumerate(profiles.items(), start=1):
        short_id = f"p{number:03d}_{g22f.g6f.stable_digest(profile_id, 6)}"
        profile_dir = artifact_dir / "profiles" / short_id
        userdir = profile_dir / "user_data"
        export_dir = profile_dir / "export"
        identifier = (
            "meb-"
            + g22f.g6f.stable_digest(f"{run_id}|{profile_id}", 16)
        )
        config_path = record_dir / f"config_{short_id}.json"
        g0.atomic_write_json(
            profile_config(
                base,
                identifier=identifier,
                pairs=pairs,
                feature_dir=feature_dir,
                event_dir=event_dir,
                profile=profile,
                train_days=int(selected_settings["train_days"]),
                backtest_days=int(selected_settings["backtest_days"]),
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
            str(selected_settings["timerange"]),
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
                "command": command,
                "status": "pending",
                "attempts": 0,
            }
        )
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": g0.utc_now(),
        "status": "prepared",
        "run_stage": "event_market_level_breadth_comparison",
        "technical_smoke_not_evidence": technical_smoke,
        "pairs": list(pairs),
        "timerange": selected_settings["timerange"],
        "train_period_days": selected_settings["train_days"],
        "backtest_period_days": selected_settings["backtest_days"],
        "validation_periods": selected_settings["validation_periods"],
        "profile_workers": max(1, min(int(profile_workers), MAX_WORKERS)),
        "model_threads_per_profile": 1,
        "model_class": "LightGBMRegressorMultiTarget",
        "profiles": list(profiles),
        "profile_count": len(profiles),
        "comparisons": list(registry["comparisons"]),
        "targets": list(frozen.TARGETS),
        "decision_rule": {
            "reference": (
                "For activity targets, correct means predicting the correct side of "
                "the causally fixed training median. For signed targets, correct "
                "means predicting the observed sign."
            ),
            "lead": (
                "At least 55% correct, positive rank relationship, better than the "
                "training-period majority, and at least 20 independent episodes in "
                "both 2024 and 2025."
            ),
            "strong": "The same rule with at least 65% correct in both periods.",
            "combination": (
                "At least two percentage points more correct than both components, "
                "with lower median error and higher rank relationship, in both periods."
            ),
            "joint_call": (
                "Issue only when predicted volume exceeds its training upper quartile "
                "and absolute predicted direction exceeds the training median absolute "
                "direction. Both above-normal realised volume and correct direction "
                "must occur on the same issued call."
            ),
            "technical_smoke_is_not_evidence": True,
        },
        "research_boundary": {
            "profit_used": False,
            "trading_rule_tested": False,
            "future_outcomes_used_only_after_feature_freeze": True,
        },
        "source_contracts": {
            "base_config": artifact(base_config),
            "python_executable": str(python_exe.resolve()),
            "strategy": artifact(STRATEGY_FILE),
            "breadth_freeze": artifact(frozen.FREEZE_PATH),
            "cache_manifest": artifact(cache.CACHE_MANIFEST_PATH),
            "profile_registry": artifact(frozen.REGISTRY_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
            "training_reference": artifact(reference_path),
            "cache_inventory": cache_manifest["inventory"],
        },
        "storage": {
            "record_dir": str(record_dir.resolve()),
            "bulky_artifact_dir": str(artifact_dir.resolve()),
            "feature_cache_dir": str(feature_dir.resolve()),
            "event_cache_dir": str(event_dir.resolve()),
            "evaluation_cache_dir": str(evaluation_dir.resolve()),
            "training_reference": str(reference_path.resolve()),
            "save_backtest_models": False,
        },
        "runtime_preparation_snapshot": g22f.g7f.runtime_snapshot(),
        "frozen_event_counts": {
            "events": freeze["event_count"],
            "episodes": freeze["event_episode_count"],
            "actual_samples": freeze["actual_sample_count"],
            "control_samples": freeze["control_sample_count"],
        },
        "commands": commands,
    }
    g0.atomic_write_json(manifest, manifest_path)
    return manifest, manifest_path


def _episode_ids(values: Series) -> set[str]:
    identifiers: set[str] = set()
    for value in values.dropna():
        parsed = json.loads(str(value))
        if isinstance(parsed, list):
            identifiers.update(str(item) for item in parsed)
    return identifiers


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
    result = {
        "returncode": check.returncode,
        "stdout": check.stdout.strip(),
        "stderr": check.stderr.strip(),
    }
    problems = (
        ["The worker cannot import Freqtrade, LightGBM, and sklearn."]
        if check.returncode
        else []
    )
    return result, problems


def preflight_run(manifest: Mapping[str, Any], *, python_exe: Path) -> dict[str, Any]:
    dependency, problems = dependency_check(python_exe)
    if not STRATEGY_FILE.is_file():
        problems.append(f"Missing event breadth strategy: {STRATEGY_FILE}")
    required_features = {
        column for command in manifest["commands"] for column in command["feature_columns"]
    }
    inventory = {
        str(item["pair"]): item
        for item in manifest["source_contracts"]["cache_inventory"]
    }
    start_raw, end_raw = str(manifest["timerange"]).split("-", maxsplit=1)
    prediction_start = pd.Timestamp(start_raw, tz="UTC")
    prediction_end = pd.Timestamp(end_raw, tz="UTC")
    training_start = prediction_start - pd.Timedelta(
        days=int(manifest["train_period_days"])
    )
    readiness: list[dict[str, Any]] = []
    for pair in manifest["pairs"]:
        item = inventory[str(pair)]
        for kind in ("feature", "event", "evaluation"):
            path = Path(item[f"{kind}_path"])
            if not path.is_file() or g0.sha256_file(path) != item[f"{kind}_sha256"]:
                problems.append(f"Cache hash mismatch for {pair}/{kind}.")
        try:
            pd.read_parquet(item["feature_path"], columns=sorted(required_features))
        except Exception as exc:
            problems.append(f"{pair} lacks an exact frozen feature: {exc}")
        frame = pd.read_parquet(
            item["event_path"],
            columns=["date", f"ready__{frozen.READY_BLOCK}", *frozen.TARGETS],
        )
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
        eligible = frame[f"ready__{frozen.READY_BLOCK}"].fillna(False).astype(bool)
        eligible &= frame[list(frozen.TARGETS)].notna().all(axis=1)
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
                "training_rows_before_first_prediction": training_rows,
                "prediction_rows": prediction_rows,
            }
        )
        minimum_training = 250 if manifest["technical_smoke_not_evidence"] else MIN_TRAINING_ROWS
        minimum_prediction = 20 if manifest["technical_smoke_not_evidence"] else MIN_PREDICTION_ROWS
        if training_rows < minimum_training:
            problems.append(f"{pair} has only {training_rows} training rows.")
        if prediction_rows < minimum_prediction:
            problems.append(f"{pair} has only {prediction_rows} prediction rows.")
    evaluation = pd.read_parquet(
        inventory[str(manifest["pairs"][0])]["evaluation_path"],
        columns=[
            "period",
            "sample_kind",
            "parent_episode_ids_json",
        ],
    )
    episode_support: list[dict[str, Any]] = []
    for period in manifest["validation_periods"]:
        cell = evaluation.loc[
            evaluation["period"].eq(period)
            & evaluation["sample_kind"].eq("actual_event")
        ]
        episodes = len(_episode_ids(cell["parent_episode_ids_json"]))
        episode_support.append(
            {"period": period, "actual_rows": len(cell), "unique_episodes": episodes}
        )
        if not manifest["technical_smoke_not_evidence"] and episodes < 20:
            problems.append(f"{period} has only {episodes} unique event episodes.")
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
        "event_episode_support": episode_support,
        "profile_workers": manifest["profile_workers"],
        "model_threads_per_profile": 1,
        "runtime_snapshot": g22f.g7f.runtime_snapshot(),
        "bulky_storage_free_gib": round(free_gib, 3),
    }


def load_actual(manifest: Mapping[str, Any]) -> DataFrame:
    frames: list[DataFrame] = []
    evaluation_dir = Path(manifest["storage"]["evaluation_cache_dir"])
    for pair in manifest["pairs"]:
        frame = pd.read_parquet(
            evaluation_dir / f"{g0.pair_file_stem(str(pair))}.parquet"
        )
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
        frame["pair"] = pair
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    if output.duplicated(["pair", "date"]).any():
        raise ValueError("Event breadth evaluation cache has duplicate pair/date keys.")
    return output


def target_kind(target: str) -> str:
    return "direction" if any(
        marker in target for marker in ("close_return", "excursion_balance")
    ) else "activity"


def sample_scope_mask(frame: DataFrame, scope: str) -> Series:
    if scope == "all_catalog":
        return Series(True, index=frame.index)
    if scope == "actual_events":
        return frame["sample_kind"].eq("actual_event")
    if scope == "matched_controls":
        return frame["sample_kind"].eq("matched_control")
    raise ValueError(f"Unknown sample scope: {scope}")


def prediction_metrics(
    actual: Series,
    prediction: Series,
    *,
    reference: float,
    majority_positive: bool,
) -> dict[str, Any]:
    clean = DataFrame({"actual": actual, "prediction": prediction}).apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 4:
        return {
            "rows": len(clean),
            **{metric: np.nan for metric in AGGREGATE_METRICS},
        }
    ranked = clean.sort_values("prediction", kind="stable")
    bucket = max(1, len(ranked) // 4)
    top = ranked.tail(bucket)
    bottom = ranked.head(bucket)
    actual_positive = clean["actual"].gt(reference)
    predicted_positive = clean["prediction"].gt(reference)
    correct = actual_positive.eq(predicted_positive)
    majority = Series(bool(majority_positive), index=clean.index)
    majority_correct = actual_positive.eq(majority)
    spearman = clean["actual"].corr(clean["prediction"], method="spearman")
    top_mean = float(top["actual"].mean())
    bottom_mean = float(bottom["actual"].mean())
    return {
        "rows": len(clean),
        "median_absolute_error": float(
            (clean["actual"] - clean["prediction"]).abs().median()
        ),
        "spearman_rank_correlation": (
            float(spearman) if pd.notna(spearman) else np.nan
        ),
        "top_quartile_actual_reaction_mean": top_mean,
        "bottom_quartile_actual_reaction_mean": bottom_mean,
        "top_minus_bottom_difference": top_mean - bottom_mean,
        "top_quartile_above_reference_fraction": float(
            top["actual"].gt(reference).mean()
        ),
        "correct_side_rate": float(correct.mean()),
        "training_majority_correct_rate": float(majority_correct.mean()),
        "correct_minus_training_majority": float(
            correct.mean() - majority_correct.mean()
        ),
        "actual_positive_rate": float(actual_positive.mean()),
        "predicted_positive_rate": float(predicted_positive.mean()),
    }


def _reference_lookup(reference: DataFrame, column: str) -> dict[tuple[str, str, str], Any]:
    return reference.set_index(["pair", "period", "target"])[column].to_dict()


def pair_scores(
    *,
    manifest: Mapping[str, Any],
    predictions: Mapping[str, DataFrame],
    actual: DataFrame,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    common, audit = g0f.common_prediction_keys(
        dict(predictions), tuple(str(pair) for pair in manifest["pairs"])
    )
    fair = common.merge(actual, on=["pair", "date"], how="left", validate="one_to_one")
    reference = pd.read_csv(manifest["storage"]["training_reference"])
    median_lookup = _reference_lookup(reference, "training_median")
    majority_lookup = _reference_lookup(reference, "training_majority_positive")
    command_lookup = {
        str(item["profile_id"]): str(item["role"]) for item in manifest["commands"]
    }
    rows: list[dict[str, Any]] = []
    for profile_id, prediction in predictions.items():
        renamed = prediction[["pair", "date", *frozen.TARGETS]].rename(
            columns={target: f"prediction__{target}" for target in frozen.TARGETS}
        )
        merged = fair.merge(renamed, on=["pair", "date"], how="left", validate="one_to_one")
        for (pair, period), period_frame in merged.groupby(
            ["pair", "period"], observed=True, sort=False
        ):
            if period not in manifest["validation_periods"]:
                continue
            for scope in SAMPLE_SCOPES:
                cell = period_frame.loc[sample_scope_mask(period_frame, scope)]
                episode_count = (
                    len(_episode_ids(cell["parent_episode_ids_json"]))
                    if scope == "actual_events"
                    else int(cell["sample_id"].nunique())
                )
                for target in frozen.TARGETS:
                    key = (str(pair), str(period), target)
                    reference_value = (
                        0.0
                        if target_kind(target) == "direction"
                        else float(median_lookup.get(key, np.nan))
                    )
                    majority_positive = bool(majority_lookup.get(key, False))
                    rows.append(
                        {
                            "pair": pair,
                            "period": period,
                            "sample_scope": scope,
                            "profile_id": profile_id,
                            "role": command_lookup[str(profile_id)],
                            "target": target,
                            "target_kind": target_kind(target),
                            "reference_value": reference_value,
                            "independent_samples": episode_count,
                            **prediction_metrics(
                                cell[target],
                                cell[f"prediction__{target}"],
                                reference=reference_value,
                                majority_positive=majority_positive,
                            ),
                        }
                    )
    return DataFrame.from_records(rows), audit, fair


def scope_scores(pair_frame: DataFrame) -> DataFrame:
    individual = pair_frame.copy()
    individual["market_scope"] = "asset:" + individual["pair"].astype(str)
    individual["eligible_coins"] = 1
    keys = ["period", "sample_scope", "profile_id", "role", "target", "target_kind"]
    pooled_rows: list[dict[str, Any]] = []
    for key, cell in pair_frame.groupby(keys, observed=True, sort=False):
        eligible = cell.loc[cell["rows"].ge(MIN_PAIR_SCORE_ROWS)]
        pooled_rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "pair": "ALL_FIVE_EQUAL_WEIGHT",
                "market_scope": "all_five_equal_weight",
                "eligible_coins": int(eligible["pair"].nunique()),
                "reference_value": float(
                    pd.to_numeric(eligible["reference_value"], errors="coerce").mean()
                ),
                "independent_samples": int(eligible["independent_samples"].min()),
                "rows": int(eligible["rows"].sum()),
                **{
                    metric: float(pd.to_numeric(eligible[metric], errors="coerce").mean())
                    for metric in AGGREGATE_METRICS
                },
            }
        )
    return pd.concat(
        [individual, DataFrame.from_records(pooled_rows)], ignore_index=True
    )


def direct_decisions(
    scope_frame: DataFrame,
    validation_periods: Sequence[str],
    *,
    technical_smoke: bool,
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    source = scope_frame.loc[scope_frame["sample_scope"].eq("actual_events")]
    keys = ["market_scope", "profile_id", "role", "target", "target_kind"]
    expected = set(validation_periods)
    for key, cell in source.groupby(keys, observed=True, sort=False):
        complete = set(cell["period"]) == expected
        support = bool(
            complete
            and cell["independent_samples"].ge(20).all()
            and cell["eligible_coins"].ge(1).all()
        )
        repeated = bool(
            support
            and cell["correct_side_rate"].ge(0.55).all()
            and cell["correct_minus_training_majority"].gt(0.0).all()
            and cell["spearman_rank_correlation"].gt(0.0).all()
        )
        strong = bool(repeated and cell["correct_side_rate"].ge(0.65).all())
        status = (
            "technical_smoke_only"
            if technical_smoke
            else "strong_exploratory_signal"
            if strong
            else "exploratory_signal"
            if repeated
            else "not_retained_in_both_periods"
        )
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "status": status,
                "complete_periods": complete,
                "support_pass": support,
                "repeated_55_percent_pass": repeated,
                "strong_65_percent_pass": strong,
                "minimum_correct_side_rate": float(cell["correct_side_rate"].min()),
                "minimum_margin_over_training_majority": float(
                    cell["correct_minus_training_majority"].min()
                ),
                "minimum_rank_relationship": float(
                    cell["spearman_rank_correlation"].min()
                ),
                "minimum_independent_samples": int(cell["independent_samples"].min()),
            }
        )
    return DataFrame.from_records(rows)


def combination_decisions(
    scope_frame: DataFrame,
    comparisons: Sequence[Mapping[str, Any]],
    validation_periods: Sequence[str],
) -> tuple[DataFrame, DataFrame]:
    period_rows: list[dict[str, Any]] = []
    source = scope_frame.loc[scope_frame["sample_scope"].eq("actual_events")]
    for comparison in comparisons:
        candidate = str(comparison["candidate"])
        components = tuple(str(item) for item in comparison["components"])
        selected = source.loc[source["role"].isin((candidate, *components))]
        keys = ["market_scope", "period", "target", "target_kind"]
        for key, cell in selected.groupby(keys, observed=True, sort=False):
            indexed = cell.set_index("role")
            present = candidate in indexed.index and all(
                component in indexed.index for component in components
            )
            if not present:
                continue
            candidate_row = indexed.loc[candidate]
            component_rows = indexed.loc[list(components)]
            minimum_accuracy_uplift = float(
                candidate_row["correct_side_rate"]
                - component_rows["correct_side_rate"].max()
            )
            period_pass = bool(
                candidate_row["correct_side_rate"] >= 0.55
                and candidate_row["correct_minus_training_majority"] > 0.0
                and minimum_accuracy_uplift >= 0.02
                and candidate_row["median_absolute_error"]
                < component_rows["median_absolute_error"].min()
                and candidate_row["spearman_rank_correlation"]
                > component_rows["spearman_rank_correlation"].max()
                and candidate_row["independent_samples"] >= 20
            )
            period_rows.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "candidate": candidate,
                    "components": json.dumps(components),
                    "candidate_correct_side_rate": candidate_row[
                        "correct_side_rate"
                    ],
                    "best_component_correct_side_rate": component_rows[
                        "correct_side_rate"
                    ].max(),
                    "minimum_accuracy_uplift": minimum_accuracy_uplift,
                    "candidate_median_absolute_error": candidate_row[
                        "median_absolute_error"
                    ],
                    "best_component_median_absolute_error": component_rows[
                        "median_absolute_error"
                    ].min(),
                    "candidate_rank_relationship": candidate_row[
                        "spearman_rank_correlation"
                    ],
                    "best_component_rank_relationship": component_rows[
                        "spearman_rank_correlation"
                    ].max(),
                    "period_pass": period_pass,
                }
            )
    periods = DataFrame.from_records(period_rows)
    if periods.empty:
        return periods, DataFrame()
    final_rows: list[dict[str, Any]] = []
    keys = ["market_scope", "candidate", "components", "target", "target_kind"]
    expected = set(validation_periods)
    for key, cell in periods.groupby(keys, observed=True, sort=False):
        complete = set(cell["period"]) == expected
        retained = bool(complete and cell["period_pass"].all())
        final_rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "status": (
                    "combination_adds_information"
                    if retained
                    else "combination_not_retained"
                ),
                "complete_periods": complete,
                "all_periods_pass": retained,
                "minimum_accuracy_uplift": float(
                    cell["minimum_accuracy_uplift"].min()
                ),
            }
        )
    return periods, DataFrame.from_records(final_rows)


def _target(metric: str, horizon: int) -> str:
    return f"&-meb_{metric}_h{horizon}"


def joint_pair_scores(
    *,
    manifest: Mapping[str, Any],
    predictions: Mapping[str, DataFrame],
    fair: DataFrame,
) -> DataFrame:
    reference = pd.read_csv(manifest["storage"]["training_reference"])
    median_lookup = _reference_lookup(reference, "training_median")
    upper_lookup = _reference_lookup(reference, "training_upper_quartile")
    absolute_lookup = _reference_lookup(reference, "training_median_absolute")
    majority_lookup = _reference_lookup(reference, "training_majority_positive")
    command_lookup = {
        str(item["profile_id"]): str(item["role"]) for item in manifest["commands"]
    }
    rows: list[dict[str, Any]] = []
    for profile_id, prediction in predictions.items():
        renamed = prediction[["pair", "date", *frozen.TARGETS]].rename(
            columns={target: f"prediction__{target}" for target in frozen.TARGETS}
        )
        merged = fair.merge(renamed, on=["pair", "date"], how="left", validate="one_to_one")
        merged = merged.loc[merged["sample_kind"].eq("actual_event")]
        for (pair, period), cell in merged.groupby(
            ["pair", "period"], observed=True, sort=False
        ):
            if period not in manifest["validation_periods"]:
                continue
            for horizon in frozen.HORIZONS:
                volume_target = _target("log_volume_ratio", horizon)
                volume_key = (str(pair), str(period), volume_target)
                for direction_metric in ("close_return_atr", "excursion_balance_atr"):
                    direction_target = _target(direction_metric, horizon)
                    direction_key = (str(pair), str(period), direction_target)
                    required = [
                        volume_target,
                        direction_target,
                        f"prediction__{volume_target}",
                        f"prediction__{direction_target}",
                        "baseline_pair_return_4h",
                    ]
                    clean = cell.dropna(subset=required).copy()
                    volume_reference = float(median_lookup.get(volume_key, np.nan))
                    issue_volume = float(upper_lookup.get(volume_key, np.nan))
                    issue_direction = float(
                        absolute_lookup.get(direction_key, np.nan)
                    )
                    issued = (
                        clean[f"prediction__{volume_target}"].gt(issue_volume)
                        & clean[f"prediction__{direction_target}"].abs().gt(
                            issue_direction
                        )
                    )
                    issued_frame = clean.loc[issued]
                    actual_reaction = issued_frame[volume_target].gt(volume_reference)
                    predicted_up = issued_frame[f"prediction__{direction_target}"].gt(0.0)
                    actual_up = issued_frame[direction_target].gt(0.0)
                    direction_correct = predicted_up.eq(actual_up)
                    majority_up = bool(majority_lookup.get(direction_key, False))
                    majority_correct = actual_up.eq(majority_up)
                    trend_correct = issued_frame["baseline_pair_return_4h"].gt(0.0).eq(
                        actual_up
                    )
                    reacted = issued_frame.loc[actual_reaction]
                    reacted_direction = direction_correct.loc[actual_reaction]
                    rows.append(
                        {
                            "pair": pair,
                            "period": period,
                            "profile_id": profile_id,
                            "role": command_lookup[str(profile_id)],
                            "horizon_hours": horizon,
                            "direction_metric": direction_metric,
                            "eligible_rows": len(clean),
                            "issued_calls": len(issued_frame),
                            "issued_coverage": len(issued_frame) / max(1, len(clean)),
                            "issued_unique_episodes": len(
                                _episode_ids(issued_frame["parent_episode_ids_json"])
                            ),
                            "reaction_success_rate": (
                                float(actual_reaction.mean())
                                if len(issued_frame)
                                else np.nan
                            ),
                            "conditional_direction_success": (
                                float(reacted_direction.mean())
                                if len(reacted)
                                else np.nan
                            ),
                            "joint_success_rate": (
                                float((actual_reaction & direction_correct).mean())
                                if len(issued_frame)
                                else np.nan
                            ),
                            "training_majority_joint_rate": (
                                float((actual_reaction & majority_correct).mean())
                                if len(issued_frame)
                                else np.nan
                            ),
                            "simple_trend_joint_rate": (
                                float((actual_reaction & trend_correct).mean())
                                if len(issued_frame)
                                else np.nan
                            ),
                        }
                    )
    return DataFrame.from_records(rows)


def joint_scope_scores(pair_frame: DataFrame) -> DataFrame:
    individual = pair_frame.copy()
    individual["market_scope"] = "asset:" + individual["pair"].astype(str)
    individual["eligible_coins"] = 1
    keys = ["period", "profile_id", "role", "horizon_hours", "direction_metric"]
    metrics = (
        "issued_coverage",
        "reaction_success_rate",
        "conditional_direction_success",
        "joint_success_rate",
        "training_majority_joint_rate",
        "simple_trend_joint_rate",
    )
    rows: list[dict[str, Any]] = []
    for key, cell in pair_frame.groupby(keys, observed=True, sort=False):
        eligible = cell.loc[cell["eligible_rows"].ge(MIN_PAIR_SCORE_ROWS)]
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "pair": "ALL_FIVE_EQUAL_WEIGHT",
                "market_scope": "all_five_equal_weight",
                "eligible_coins": int(eligible["pair"].nunique()),
                "eligible_rows": int(eligible["eligible_rows"].sum()),
                "issued_calls": int(eligible["issued_calls"].sum()),
                "issued_unique_episodes": int(
                    eligible["issued_unique_episodes"].min()
                ),
                **{
                    metric: float(pd.to_numeric(eligible[metric], errors="coerce").mean())
                    for metric in metrics
                },
            }
        )
    return pd.concat([individual, DataFrame.from_records(rows)], ignore_index=True)


def joint_decisions(
    scope_frame: DataFrame,
    validation_periods: Sequence[str],
    *,
    technical_smoke: bool,
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["market_scope", "profile_id", "role", "horizon_hours", "direction_metric"]
    expected = set(validation_periods)
    for key, cell in scope_frame.groupby(keys, observed=True, sort=False):
        complete = set(cell["period"]) == expected
        retained = bool(
            complete
            and cell["issued_unique_episodes"].ge(20).all()
            and cell["joint_success_rate"].ge(0.55).all()
            and cell["joint_success_rate"].gt(
                cell["training_majority_joint_rate"]
            ).all()
            and cell["joint_success_rate"].gt(cell["simple_trend_joint_rate"]).all()
        )
        strong = bool(retained and cell["joint_success_rate"].ge(0.65).all())
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "status": (
                    "technical_smoke_only"
                    if technical_smoke
                    else "strong_joint_signal"
                    if strong
                    else "joint_signal_lead"
                    if retained
                    else "joint_signal_not_retained"
                ),
                "complete_periods": complete,
                "joint_55_percent_pass": retained,
                "joint_65_percent_pass": strong,
                "minimum_issued_unique_episodes": int(
                    cell["issued_unique_episodes"].min()
                ),
                "minimum_issued_coverage": float(cell["issued_coverage"].min()),
                "minimum_reaction_success": float(
                    cell["reaction_success_rate"].min()
                ),
                "minimum_conditional_direction_success": float(
                    cell["conditional_direction_success"].min()
                ),
                "minimum_joint_success": float(cell["joint_success_rate"].min()),
                "minimum_margin_over_majority": float(
                    (cell["joint_success_rate"] - cell["training_majority_joint_rate"]).min()
                ),
                "minimum_margin_over_simple_trend": float(
                    (cell["joint_success_rate"] - cell["simple_trend_joint_rate"]).min()
                ),
            }
        )
    return DataFrame.from_records(rows)


def score_run(manifest: dict[str, Any], *, record_dir: Path) -> dict[str, Any]:
    predictions: dict[str, DataFrame] = {}
    audits: list[dict[str, Any]] = []
    for item in manifest["commands"]:
        frame, audit = g0f.load_predictions(
            Path(item["model_dir"]), tuple(str(pair) for pair in item["pairs"])
        )
        audit.update({"profile_id": item["profile_id"], "role": item["role"]})
        audits.append(audit)
        if frame.empty:
            raise ValueError(f"No FreqAI predictions for {item['profile_id']}.")
        missing = sorted(set(frozen.TARGETS).difference(frame.columns))
        if missing:
            raise ValueError(f"Profile {item['profile_id']} lacks targets: {missing}")
        predictions[str(item["profile_id"])] = frame
    actual = load_actual(manifest)
    pairs, eligibility, fair = pair_scores(
        manifest=manifest, predictions=predictions, actual=actual
    )
    scopes = scope_scores(pairs)
    decisions = direct_decisions(
        scopes,
        manifest["validation_periods"],
        technical_smoke=manifest["technical_smoke_not_evidence"],
    )
    combination_periods, combinations = combination_decisions(
        scopes, manifest["comparisons"], manifest["validation_periods"]
    )
    joint_pairs = joint_pair_scores(
        manifest=manifest, predictions=predictions, fair=fair
    )
    joint_scopes = joint_scope_scores(joint_pairs)
    joint = joint_decisions(
        joint_scopes,
        manifest["validation_periods"],
        technical_smoke=manifest["technical_smoke_not_evidence"],
    )
    paths = {
        "prediction_audit": record_dir / "event_freqai_prediction_audit.csv",
        "prediction_eligibility": record_dir / "event_freqai_prediction_eligibility.csv",
        "pair_scores": record_dir / "event_freqai_pair_scores.csv",
        "scope_scores": record_dir / "event_freqai_scope_scores.csv",
        "direct_decisions": record_dir / "event_freqai_direct_decisions.csv",
        "combination_periods": record_dir / "event_freqai_combination_periods.csv",
        "combination_decisions": record_dir / "event_freqai_combination_decisions.csv",
        "joint_pair_scores": record_dir / "event_freqai_joint_pair_scores.csv",
        "joint_scope_scores": record_dir / "event_freqai_joint_scope_scores.csv",
        "joint_decisions": record_dir / "event_freqai_joint_decisions.csv",
    }
    outputs = {
        "prediction_audit": DataFrame.from_records(audits),
        "prediction_eligibility": eligibility,
        "pair_scores": pairs,
        "scope_scores": scopes,
        "direct_decisions": decisions,
        "combination_periods": combination_periods,
        "combination_decisions": combinations,
        "joint_pair_scores": joint_pairs,
        "joint_scope_scores": joint_scopes,
        "joint_decisions": joint,
    }
    for name, path in paths.items():
        g0.atomic_write_csv(outputs[name], path)
    result = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "created_at_utc": g0.utc_now(),
        "status": (
            "completed_event_freqai_technical_smoke_not_evidence"
            if manifest["technical_smoke_not_evidence"]
            else "completed_event_freqai_breadth"
        ),
        "technical_smoke_not_evidence": manifest["technical_smoke_not_evidence"],
        "profiles_completed": len(manifest["commands"]),
        "targets_scored": len(frozen.TARGETS),
        "direct_signal_leads": int(
            decisions["status"].isin(
                ["exploratory_signal", "strong_exploratory_signal"]
            ).sum()
        ),
        "combination_leads": int(
            combinations.get("all_periods_pass", Series(dtype=bool)).fillna(False).sum()
        ),
        "joint_signal_leads": int(
            joint.get("joint_55_percent_pass", Series(dtype=bool)).fillna(False).sum()
        ),
        "integrity": {
            "real_freqai_model_class": manifest["model_class"],
            "all_profile_commands_terminal": all(
                item.get("status") == "completed" for item in manifest["commands"]
            ),
            "all_profiles_scored_on_common_prediction_keys": True,
            "whole_event_periods_preserved": True,
            "profit_used": False,
        },
        "artifacts": {name: artifact(path) for name, path in paths.items()},
    }
    result_path = record_dir / "event_freqai_result.json"
    g0.atomic_write_json(result, result_path)
    return {**result, "result_path": str(result_path.resolve())}


def run(
    *,
    run_id: str,
    base_config: Path,
    python_exe: Path,
    profile_workers: int,
    technical_smoke: bool,
    prepare_only: bool,
) -> dict[str, Any]:
    manifest, manifest_path = build_manifest(
        run_id=run_id,
        base_config=base_config,
        python_exe=python_exe,
        profile_workers=profile_workers,
        technical_smoke=technical_smoke,
    )
    audit = preflight_run(manifest, python_exe=python_exe)
    preflight_path = manifest_path.parent / "event_freqai_preflight.json"
    g0.atomic_write_json(audit, preflight_path)
    if not audit["passed"]:
        manifest["status"] = "blocked_preflight"
        manifest["preflight"] = artifact(preflight_path)
        g0.atomic_write_json(manifest, manifest_path)
        return {"status": "blocked_preflight", "preflight": audit}
    if prepare_only:
        manifest["status"] = "prepared_and_preflight_passed"
        manifest["preflight"] = artifact(preflight_path)
        g0.atomic_write_json(manifest, manifest_path)
        return {
            "status": "prepared_and_preflight_passed",
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
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
    run_id = args.run_id or (
        DEFAULT_SMOKE_ID if args.technical_smoke else DEFAULT_RUN_ID
    )
    result = run(
        run_id=run_id,
        base_config=args.base_config,
        python_exe=args.python_exe,
        profile_workers=max(1, min(args.profile_workers, MAX_WORKERS)),
        technical_smoke=args.technical_smoke,
        prepare_only=args.prepare_only,
    )
    print(json.dumps(result, indent=2, default=g0.json_default), flush=True)
    return 0 if str(result["status"]).startswith(("completed_", "prepared_")) else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Run Generation 22's frozen low-dimensional unsigned-reaction FreqAI route."""

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
    market_reaction_zone_freqai_generation6 as g6f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation7 as g7f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation13 as g13,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation16_direct_attribution as g16d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation18_freeze as g18z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_freeze as g22z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_freqai_cache as g22c,
)


ANALYSIS_PATH = Path(__file__).resolve()
DEFAULT_CONFIG = g13.DEFAULT_CONFIG
DEFAULT_PYTHON = g13.DEFAULT_PYTHON
STRATEGY_PATH = g13.STRATEGY_PATH
STRATEGY_FILE = STRATEGY_PATH / "MarketReactionZoneFreqAIGeneration22Strategy.py"
STRATEGY_CLASS = "MarketReactionZoneG22ConfigurableFreqAIResearchStrategy"
DATA_DIR = g13.DATA_DIR
RECORD_ROOT = g22z.OUTPUT_ROOT / "freqai"
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation22_branches"
    / "g22_broad_siblings"
    / "freqai"
)
DEFAULT_RUN_STEM = "g22_unsigned_reaction_freqai_20260827a"
DEFAULT_SMOKE_STEM = "g22_unsigned_reaction_freqai_smoke_20260827a"
DEFAULT_JOINT_REVIEW_ID = "g22_unsigned_reaction_freqai_joint_20260827a"
MAX_WORKERS = 4
MIN_TRAINING_ROWS = 720
MIN_PREDICTION_ROWS = 120
MIN_PAIR_SCORE_ROWS = 24
TARGET_PURGE_HOURS = max(g22c.HORIZONS)
PAIR_METRICS = (
    "median_absolute_error",
    "spearman_rank_correlation",
    "top_quartile_actual_reaction_mean",
    "bottom_quartile_actual_reaction_mean",
    "top_minus_bottom_difference",
    "top_quartile_above_training_median_fraction",
)


def artifact(path: Path) -> dict[str, Any]:
    return g22z.artifact(path)


def cohort_settings(cohort: str, *, technical_smoke: bool) -> dict[str, Any]:
    periods = list(g18z.CONFIRMATION_PERIODS[cohort])
    start = pd.Timestamp(periods[0]["start_utc"])
    stop = pd.Timestamp(periods[-1]["end_utc_exclusive"])
    train_days = 365 if cohort == "normal" else 160
    backtest_days = 15
    if technical_smoke:
        stop = start + pd.Timedelta(days=2)
        backtest_days = 2
    return {
        "timerange": f"{start:%Y%m%d}-{stop:%Y%m%d}",
        "train_days": train_days,
        "backtest_days": backtest_days,
        "validation_periods": [item["id"] for item in periods],
        "period_bounds": periods,
    }


def load_sources(
    cohort: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    frozen = g22c.load_freeze()
    cache_path = g22c.cache_manifest_path(cohort)
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    registry = json.loads(g22c.REGISTRY_PATH.read_text(encoding="utf-8"))
    if cache.get("status") != "completed_generation22_freqai_cache":
        raise ValueError(f"Generation 22 {cohort} FreqAI cache is not terminal.")
    if registry.get("status") != "frozen_before_generation22_freqai_target_materialization":
        raise ValueError("Generation 22 FreqAI registry is invalid.")
    if g0.sha256_file(g22c.REGISTRY_PATH) != cache["profile_registry"]["sha256"]:
        raise ValueError("Generation 22 FreqAI registry changed after target construction.")
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
    if set(profile["role"] for profile in profiles.values()) != set(g22c.PROFILE_ROLES):
        raise ValueError(f"Generation 22 {cohort} profile ladder is incomplete.")
    if len(comparisons) != 4:
        raise ValueError(f"Generation 22 {cohort} control ladder is incomplete.")
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
    config = g13.profile_config(
        base,
        identifier=identifier,
        pairs=pairs,
        feature_dir=feature_dir,
        event_dir=event_dir,
        profile=profile,
        train_days=train_days,
        backtest_days=backtest_days,
        technical_smoke=technical_smoke,
        maximum_target_horizon_hours=TARGET_PURGE_HOURS,
    )
    config["market_reaction_zone_g22"] = config.pop("market_reaction_zone_g13")
    config["freqai"]["model_training_parameters"]["n_jobs"] = 1
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
            columns=["date", f"ready__{g22c.READY_BLOCK}", *g22c.TARGETS],
        )
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
        ready = frame[f"ready__{g22c.READY_BLOCK}"].fillna(False).astype(bool)
        for period in settings["period_bounds"]:
            start = pd.Timestamp(period["start_utc"])
            cutoff = start - pd.Timedelta(hours=TARGET_PURGE_HOURS)
            training_start = start - pd.Timedelta(days=int(settings["train_days"]))
            time_mask = ready & frame["date"].ge(training_start) & frame["date"].lt(cutoff)
            for target in g22c.TARGETS:
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
    settings = cohort_settings(cohort, technical_smoke=technical_smoke)
    record_dir = RECORD_ROOT / run_id
    artifact_dir = ARTIFACT_ROOT / run_id
    manifest_path = record_dir / "g22_freqai_run_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, path in (
            ("strategy", STRATEGY_FILE),
            ("freeze", g22z.FREEZE_PATH),
            ("cache_manifest", g22c.cache_manifest_path(cohort)),
            ("profile_registry", g22c.REGISTRY_PATH),
            ("analysis_script", ANALYSIS_PATH),
        ):
            if g0.sha256_file(path) != existing["source_contracts"][name]["sha256"]:
                raise ValueError(f"Generation 22 {name} changed: {manifest_path}")
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
    threshold_path = record_dir / "g22_training_median_thresholds.csv"
    g0.atomic_write_csv(thresholds, threshold_path)
    commands: list[dict[str, Any]] = []
    for number, (profile_identifier, profile) in enumerate(profiles.items(), start=1):
        short_id = f"p{number:03d}_{g6f.stable_digest(profile_identifier, 6)}"
        profile_dir = artifact_dir / "profiles" / short_id
        userdir = profile_dir / "user_data"
        export_dir = profile_dir / "export"
        identifier = f"g22-{g6f.stable_digest(f'{run_id}|{profile_identifier}', 16)}"
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
    branch = next(
        item
        for item in frozen["branches"]
        if item["branch_id"] == "g22e_freqai_unsigned_reaction_interaction_regression"
    )
    manifest = {
        "schema_version": 1,
        "generation": 22,
        "run_id": run_id,
        "created_at_utc": g0.utc_now(),
        "status": "prepared",
        "run_stage": "unsigned_reaction_interaction_regression",
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
        "targets": list(g22c.TARGETS),
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
            "freeze": artifact(g22z.FREEZE_PATH),
            "cache_manifest": artifact(g22c.cache_manifest_path(cohort)),
            "profile_registry": artifact(g22c.REGISTRY_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
            "training_median_thresholds": artifact(threshold_path),
            "cache_inventory": cache["inventory"],
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
        "runtime_preparation_snapshot": g7f.runtime_snapshot(),
        "commands": commands,
    }
    g0.atomic_write_json(manifest, manifest_path)
    return manifest, manifest_path


def preflight_run(manifest: dict[str, Any], *, python_exe: Path) -> dict[str, Any]:
    problems: list[str] = []
    if not python_exe.is_file():
        problems.append(f"Missing worker interpreter: {python_exe}")
        dependency: dict[str, Any] = {}
    else:
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
        if check.returncode:
            problems.append("The worker cannot import Freqtrade, LightGBM, and sklearn.")
    if not STRATEGY_FILE.is_file():
        problems.append(f"Missing Generation 22 strategy: {STRATEGY_FILE}")
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
            problems.append(f"{pair} lacks exact Generation 22 features: {exc}")
        for label_source, path_key in (
            ("actual", "event_path"),
            ("shuffled", "shuffled_event_path"),
        ):
            frame = pd.read_parquet(
                item[path_key],
                columns=["date", f"ready__{g22c.READY_BLOCK}", *g22c.TARGETS],
            )
            frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
            eligible = frame[f"ready__{g22c.READY_BLOCK}"].fillna(False).astype(bool)
            eligible &= frame[list(g22c.TARGETS)].notna().all(axis=1)
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
                problems.append(
                    f"{pair}/{label_source} has only {training_rows} training rows."
                )
            required_prediction_rows = (
                24
                if manifest["technical_smoke_not_evidence"]
                else MIN_PREDICTION_ROWS
            )
            if prediction_rows < required_prediction_rows:
                problems.append(
                    f"{pair}/{label_source} has only {prediction_rows} prediction rows."
                )
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
        "profile_workers": manifest["profile_workers"],
        "model_threads_per_profile": 1,
        "runtime_snapshot": g7f.runtime_snapshot(),
        "bulky_storage_free_gib": round(free_gib, 3),
    }


def load_actual(manifest: dict[str, Any]) -> DataFrame:
    frames: list[DataFrame] = []
    event_dir = Path(manifest["storage"]["event_cache_dir"])
    for pair in manifest["pairs"]:
        frame = pd.read_parquet(
            event_dir / f"{g0.pair_file_stem(pair)}.parquet",
            columns=["date", "period", *g22c.TARGETS],
        )
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
        frame["pair"] = pair
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    if output.duplicated(["pair", "date"]).any():
        raise ValueError("Duplicate pair/date rows in Generation 22 target cache.")
    return output


def prediction_metrics(actual: Series, prediction: Series, threshold: float) -> dict[str, Any]:
    clean = DataFrame({"actual": actual, "prediction": prediction}).apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 4:
        return {"rows": len(clean), **{metric: np.nan for metric in PAIR_METRICS}}
    ranked = clean.sort_values("prediction", kind="stable")
    bucket = max(1, len(ranked) // 4)
    top = ranked.tail(bucket)
    bottom = ranked.head(bucket)
    spearman = clean["actual"].corr(clean["prediction"], method="spearman")
    top_mean = float(top["actual"].mean())
    bottom_mean = float(bottom["actual"].mean())
    return {
        "rows": len(clean),
        "median_absolute_error": float((clean["actual"] - clean["prediction"]).abs().median()),
        "spearman_rank_correlation": float(spearman) if pd.notna(spearman) else np.nan,
        "top_quartile_actual_reaction_mean": top_mean,
        "bottom_quartile_actual_reaction_mean": bottom_mean,
        "top_minus_bottom_difference": top_mean - bottom_mean,
        "top_quartile_above_training_median_fraction": (
            float(top["actual"].gt(threshold).mean()) if np.isfinite(threshold) else np.nan
        ),
    }


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
    rows: list[dict[str, Any]] = []
    for profile_id, prediction in predictions.items():
        role = next(
            item["role"] for item in manifest["commands"] if item["profile_id"] == profile_id
        )
        columns = ["pair", "date", *g22c.TARGETS]
        renamed = prediction[columns].rename(
            columns={target: f"prediction__{target}" for target in g22c.TARGETS}
        )
        merged = fair.merge(renamed, on=["pair", "date"], how="left", validate="one_to_one")
        for (pair, period), cell in merged.groupby(
            ["pair", "period"], observed=True, sort=False
        ):
            if period not in manifest["validation_periods"]:
                continue
            for target in g22c.TARGETS:
                threshold = float(threshold_lookup.get((pair, period, target), np.nan))
                rows.append(
                    {
                        "cohort": manifest["cohort"],
                        "pair": pair,
                        "period": period,
                        "profile_id": profile_id,
                        "role": role,
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
        for target in g22c.TARGETS:
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
    expanded: list[DataFrame] = []
    for (cohort, pair), cell in pair_frame.groupby(
        ["cohort", "pair"], observed=True, sort=False
    ):
        for scope in g16d.group_for_pair(str(cohort), str(pair)):
            copy = cell.copy()
            copy["market_scope"] = scope
            expanded.append(copy)
    source = pd.concat(expanded, ignore_index=True)
    keys = ["cohort", "period", "role", "target", "market_scope"]
    rows: list[dict[str, Any]] = []
    for key, cell in source.groupby(keys, observed=True, sort=False):
        eligible = cell.loc[cell["rows"].ge(MIN_PAIR_SCORE_ROWS)].copy()
        scope = str(key[-1])
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "eligible_coins": int(eligible["pair"].nunique()),
                "required_coins": g16d.GROUP_MIN_COINS[scope],
                "coin_support_pass": bool(
                    eligible["pair"].nunique() >= g16d.GROUP_MIN_COINS[scope]
                ),
                **{
                    metric: float(pd.to_numeric(eligible[metric], errors="coerce").mean())
                    for metric in PAIR_METRICS
                },
            }
        )
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


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
        for control in (
            "constant_training_median",
            "level_geometry_only",
            "market_state_only",
            "within_pair_time_shuffled_training_labels",
        ):
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
                    "control": control,
                    "control_present": present,
                    "candidate_gate_pass": candidate_gate,
                    "candidate_mae": candidate["median_absolute_error"],
                    "control_mae": baseline.get("median_absolute_error", np.nan),
                    "candidate_spearman": candidate["spearman_rank_correlation"],
                    "control_spearman": baseline.get("spearman_rank_correlation", np.nan),
                    "candidate_top_minus_bottom": candidate["top_minus_bottom_difference"],
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
        complete = controls == {
            "constant_training_median",
            "level_geometry_only",
            "market_state_only",
            "within_pair_time_shuffled_training_labels",
        } and periods == expected_periods
        retained = bool(complete and cell["period_control_pass"].all())
        decision_rows.append(
            {
                "cohort": cohort,
                "market_scope": scope,
                "target": target,
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
        frame, audit = g0f.load_predictions(
            Path(item["model_dir"]), tuple(item["pairs"])
        )
        audit["profile_id"] = item["profile_id"]
        audit["role"] = item["role"]
        audits.append(audit)
        if frame.empty:
            raise ValueError(f"No Generation 22 predictions for {item['profile_id']}")
        missing = sorted(set(g22c.TARGETS).difference(frame.columns))
        if missing:
            raise ValueError(f"Profile {item['profile_id']} lacks targets: {missing}")
        predictions[str(item["profile_id"])] = frame
    actual = load_actual(manifest)
    pair, eligibility = pair_scores(
        manifest=manifest, predictions=predictions, actual=actual
    )
    scopes = scope_scores(pair)
    comparisons, decisions = control_comparisons(scopes, manifest["validation_periods"])
    paths = {
        "prediction_audit": record_dir / "g22_prediction_audit.csv",
        "prediction_eligibility": record_dir / "g22_prediction_eligibility.csv",
        "pair_scores": record_dir / "g22_pair_scores.csv",
        "scope_scores": record_dir / "g22_scope_scores.csv",
        "control_comparisons": record_dir / "g22_control_comparisons.csv",
        "decisions": record_dir / "g22_freqai_decisions.csv",
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
        "generation": 22,
        "run_id": manifest["run_id"],
        "created_at_utc": g0.utc_now(),
        "status": (
            "completed_generation22_freqai_technical_smoke_not_evidence"
            if manifest["technical_smoke_not_evidence"]
            else "completed_generation22_freqai_cohort"
        ),
        "technical_smoke_not_evidence": manifest["technical_smoke_not_evidence"],
        "cohort": manifest["cohort"],
        "profiles_completed": len(predictions),
        "targets_scored": len(g22c.TARGETS),
        "exploratory_unsigned_reaction_leads": retained,
        "not_retained": len(decisions) - retained,
        "integrity": {
            "real_freqai_model_class": manifest["model_class"],
            "all_profile_commands_terminal": all(
                item.get("status") == "completed" for item in manifest["commands"]
            ),
            "profit_used": False,
            "future_signed_direction_used": False,
            "auc_used": False,
        },
        "artifacts": {key: artifact(path) for key, path in paths.items()},
    }
    result_path = record_dir / "g22_freqai_result.json"
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
    g0.atomic_write_json(audit, manifest_path.parent / "g22_freqai_preflight.json")
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
    returncode = g7f.run_manifest(manifest, manifest_path)
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
        raise ValueError("Generation 22 FreqAI joint review requires normal and meme cohorts.")
    frames: list[DataFrame] = []
    for item in results:
        frames.append(pd.read_csv(item["artifacts"]["decisions"]["path"]))
    decisions = pd.concat(frames, ignore_index=True)
    record_dir = RECORD_ROOT / DEFAULT_JOINT_REVIEW_ID
    record_dir.mkdir(parents=True, exist_ok=True)
    decision_path = record_dir / "g22_freqai_joint_decisions.csv"
    g0.atomic_write_csv(decisions, decision_path)
    retained = int(decisions["all_controls_both_periods_pass"].sum())
    output = {
        "schema_version": 1,
        "generation": 22,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation22_freqai_joint_review",
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
    result_path = record_dir / "g22_freqai_joint_review.json"
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

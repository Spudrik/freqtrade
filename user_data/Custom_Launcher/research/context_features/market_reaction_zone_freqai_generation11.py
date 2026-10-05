"""Run the frozen Generation 11 FreqAI reaction-attribution batch."""

from __future__ import annotations

# Bound native numerical pools before importing pandas and FreqAI helpers.
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
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation7 as g7f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation8 as g8,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation11_freqai_cache as g11c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation11_freqai_freeze as g11z,
)


USER_DATA_DIR = REPO_ROOT / "user_data"
DEFAULT_CONFIG = (
    USER_DATA_DIR / "configs" / "config_market_reaction_zone_freqai.example.json"
)
DEFAULT_PYTHON = (
    REPO_ROOT
    / "runtime"
    / "venvs"
    / "freqtrade-backtest-08"
    / "Scripts"
    / "python.exe"
)
STRATEGY_PATH = USER_DATA_DIR / "strategies"
STRATEGY_FILE = STRATEGY_PATH / "MarketReactionZoneFreqAIGeneration11Strategy.py"
STRATEGY_CLASS = "MarketReactionZoneG11ConfigurableFreqAIResearchStrategy"
DATA_DIR = USER_DATA_DIR / "data" / "binance"
RECORD_ROOT = (
    g11z.g11r.REVIEW_ROOT
    / "generation11_branches"
    / "g11_freqai_initial_attribution"
)
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData")
    / "research_outputs"
    / "market_reaction_zones"
    / "generation11_branches"
    / "g11_freqai_initial_attribution"
)
DEFAULT_RUN_STEM = "g11_initial_attribution_20260822a"
DEFAULT_SMOKE_STEM = "g11_technical_smoke_20260822a"
MAX_WORKERS = 4
MIN_TRAINING_ROWS = 30
MIN_PAIR_SCORABLE_ROWS = 15
MAX_TARGET_HORIZON_HOURS = max(g11z.g11z.HORIZONS)


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def load_sources(
    cohort: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    frozen = g11z.validate_existing_freeze()
    cache_path = g11c.RECORD_ROOT / f"{cohort}_manifest.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache.get("status") != "completed_generation11_freqai_cache":
        raise ValueError(f"Generation 11 {cohort} cache is not terminal.")
    if cache.get("cohort") != cohort:
        raise ValueError(f"Generation 11 cache cohort mismatch: {cache_path}")
    source = cache["source_contracts"]["generation11_freqai_freeze"]
    if g0.sha256_file(g11z.FREEZE_PATH) != source["sha256"]:
        raise ValueError("Generation 11 freeze changed after cache materialization.")
    return frozen, cache, cache_path


def active_registry(
    frozen: dict[str, Any], cache: dict[str, Any], cohort: str
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    supported = set(cache["supported_profiles"])
    cohort_comparisons = [
        dict(item)
        for item in frozen["comparisons"]
        if item["cohort"] == cohort
    ]
    comparisons = [
        item
        for item in cohort_comparisons
        if item["candidate"] in supported and item["baseline"] in supported
    ]
    referenced = {
        profile_id
        for item in comparisons
        for profile_id in (item["candidate"], item["baseline"])
    }
    profiles = {
        profile_id: dict(profile)
        for profile_id, profile in frozen["profiles"].items()
        if profile_id in referenced
    }
    active_routes = {item["route_id"] for item in comparisons}
    all_routes = {item["route_id"] for item in cohort_comparisons}
    audit = {
        "supported_profiles": sorted(supported),
        "referenced_supported_profiles": list(profiles),
        "unreferenced_supported_profiles": sorted(supported - referenced),
        "parked_profiles": list(cache["parked_profiles"]),
        "active_routes": sorted(active_routes),
        "parked_routes": sorted(all_routes - active_routes),
        "active_comparisons": len(comparisons),
    }
    if not profiles or not comparisons:
        raise ValueError(f"No supported Generation 11 comparisons for {cohort}.")
    return profiles, comparisons, audit


def smoke_registry(
    profiles: dict[str, dict[str, Any]], comparisons: Sequence[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    selected_comparisons = [
        dict(item)
        for item in comparisons
        if item["route_id"] == "level_identity_and_timeframe"
    ]
    referenced = {
        profile_id
        for item in selected_comparisons
        for profile_id in (item["candidate"], item["baseline"])
    }
    return (
        {key: value for key, value in profiles.items() if key in referenced},
        selected_comparisons,
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
    config = g8.g6f.profile_config(
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
    research = config.pop("market_reaction_zone_g6")
    research.update(
        {
            "target_columns": list(profile["targets"]),
            "profile_role": profile["role"],
            "model_seed": int(profile["seed"]),
            "maximum_target_horizon_hours": MAX_TARGET_HORIZON_HOURS,
            "train_prediction_embargo_hours": MAX_TARGET_HORIZON_HOURS,
        }
    )
    config["market_reaction_zone_g11"] = research
    config["freqai"]["feature_parameters"][
        "label_period_candles"
    ] = MAX_TARGET_HORIZON_HOURS
    config["freqai"]["data_split_parameters"]["random_state"] = int(
        profile["seed"]
    )
    config["freqai"]["model_training_parameters"]["random_state"] = int(
        profile["seed"]
    )
    return config


def build_manifest(
    *,
    run_id: str,
    cohort: str,
    base_config: Path,
    python_exe: Path,
    profile_workers: int,
    technical_smoke: bool,
) -> tuple[dict[str, Any], Path, Path]:
    frozen, cache, cache_path = load_sources(cohort)
    profiles, comparisons, registry_audit = active_registry(
        frozen, cache, cohort
    )
    if technical_smoke:
        profiles, comparisons = smoke_registry(profiles, comparisons)
    record_dir = RECORD_ROOT / run_id
    artifact_dir = ARTIFACT_ROOT / run_id
    manifest_path = record_dir / "g11_freqai_run_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        contracts = existing["source_contracts"]
        if g0.sha256_file(STRATEGY_FILE) != contracts["strategy"]["sha256"]:
            raise ValueError(
                f"Strategy changed after Generation 11 run preparation: {manifest_path}"
            )
        if g0.sha256_file(cache_path) != contracts["cache_manifest"]["sha256"]:
            raise ValueError(
                f"Cache changed after Generation 11 run preparation: {manifest_path}"
            )
        return existing, manifest_path, artifact_dir
    record_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    base = json.loads(base_config.read_text(encoding="utf-8"))
    settings = dict(frozen["cohort_settings"][cohort])
    timerange = str(settings["timerange"])
    if technical_smoke:
        if cohort == "normal":
            timerange = "20250401-20250701"
            settings.update({"train_days": 365, "backtest_days": 90})
        else:
            timerange = "20260401-20260516"
            settings.update({"train_days": 160, "backtest_days": 45})
    feature_dir = Path(cache["inventory"][0]["feature_path"]).parent
    event_dir = Path(cache["inventory"][0]["event_path"]).parent
    pairs = tuple(str(pair) for pair in cache["pairs"])
    commands: list[dict[str, Any]] = []
    for number, (profile_id, profile) in enumerate(profiles.items(), start=1):
        short_id = f"p{number:03d}_{g8.g6f.stable_digest(profile_id, 6)}"
        profile_dir = artifact_dir / "profiles" / short_id
        userdir = profile_dir / "user_data"
        export_dir = profile_dir / "export"
        identifier = f"g11-{g8.g6f.stable_digest(f'{run_id}|{profile_id}', 16)}"
        config_path = record_dir / f"config_{short_id}.json"
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
            "LightGBMRegressorMultiTarget",
            "--timerange",
            timerange,
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
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": g0.utc_now(),
        "status": "prepared",
        "generation": 11,
        "run_stage": "initial_broad_freqai_attribution",
        "technical_smoke_not_evidence": technical_smoke,
        "evidence_label": frozen["evidence_label"],
        "cohort": cohort,
        "pairs": list(pairs),
        "timerange": timerange,
        "train_period_days": int(settings["train_days"]),
        "backtest_period_days": int(settings["backtest_days"]),
        "validation_periods": list(settings["validation_periods"]),
        "profile_workers": max(1, min(profile_workers, MAX_WORKERS)),
        "model_threads_per_profile": 1,
        "model_class": "LightGBMRegressorMultiTarget",
        "profiles": list(profiles),
        "profile_count": len(profiles),
        "comparisons": comparisons,
        "comparison_count": len(comparisons),
        "questions": sorted({item["question_id"] for item in comparisons}),
        "routes": sorted({item["route_id"] for item in comparisons}),
        "targets": list(g11z.TARGETS),
        "seeds": sorted({int(item["seed"]) for item in profiles.values()}),
        "profile_registry_frozen_before_model_outcomes": True,
        "profile_registry": {
            "profiles": profiles,
            "comparisons": comparisons,
            "support_audit": registry_audit,
        },
        "decision_rule": frozen["decision_rule"],
        "research_boundary": dict(frozen["research_boundary"]),
        "source_contracts": {
            "base_config": artifact(base_config),
            "python_executable": str(python_exe.resolve()),
            "strategy": artifact(STRATEGY_FILE),
            "generation11_freqai_freeze": artifact(g11z.FREEZE_PATH),
            "cache_manifest": artifact(cache_path),
            "cache_inventory": cache["inventory"],
        },
        "storage": {
            "record_dir": str(record_dir.resolve()),
            "bulky_artifact_dir": str(artifact_dir.resolve()),
            "feature_cache_dir": str(feature_dir.resolve()),
            "event_cache_dir": str(event_dir.resolve()),
            "save_backtest_models": False,
        },
        "runtime_preparation_snapshot": g7f.runtime_snapshot(),
        "commands": commands,
    }
    g0.atomic_write_json(manifest, manifest_path)
    return manifest, manifest_path, artifact_dir


def preflight_run(
    manifest: dict[str, Any], *, python_exe: Path
) -> dict[str, Any]:
    problems: list[str] = []
    dependency: dict[str, Any] = {}
    if not python_exe.is_file():
        problems.append(f"Missing worker interpreter: {python_exe}")
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
            problems.append("The worker cannot import Freqtrade and LightGBM.")
    if not STRATEGY_FILE.is_file():
        problems.append(f"Missing Generation 11 strategy: {STRATEGY_FILE}")
    for item in manifest["source_contracts"]["cache_inventory"]:
        for key in ("feature", "support", "event", "evaluation"):
            path = Path(item[f"{key}_path"])
            if not path.is_file() or g0.sha256_file(path) != item[f"{key}_sha256"]:
                problems.append(f"Cache hash mismatch for {item['pair']} and {key}.")
    start_raw, end_raw = str(manifest["timerange"]).split("-", maxsplit=1)
    prediction_start = pd.Timestamp(start_raw, tz="UTC")
    prediction_end = pd.Timestamp(end_raw, tz="UTC")
    training_start = prediction_start - pd.Timedelta(
        days=int(manifest["train_period_days"])
    )
    event_dir = Path(manifest["storage"]["event_cache_dir"])
    readiness: list[dict[str, Any]] = []
    requirements = {
        (tuple(item["required_ready_blocks"]), tuple(item["targets"]))
        for item in manifest["commands"]
    }
    for ready_blocks, targets in sorted(requirements):
        ready_columns = [f"ready__{block}" for block in ready_blocks]
        for pair in manifest["pairs"]:
            frame = pd.read_parquet(
                event_dir / f"{g0.pair_file_stem(pair)}.parquet",
                columns=["date", "period", *targets, *ready_columns],
            )
            frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
            eligible = frame[ready_columns].fillna(False).astype(bool).all(axis=1)
            eligible &= frame[list(targets)].notna().all(axis=1)
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
            period_counts = {
                period: int((eligible & frame["period"].eq(period)).sum())
                for period in manifest["validation_periods"]
            }
            readiness.append(
                {
                    "pair": pair,
                    "ready_blocks": ",".join(ready_blocks),
                    "training_rows": training_rows,
                    "prediction_rows": prediction_rows,
                    "validation_period_rows": json.dumps(
                        period_counts, sort_keys=True
                    ),
                }
            )
            if training_rows < MIN_TRAINING_ROWS:
                problems.append(
                    f"{pair} has only {training_rows} training rows for {ready_blocks}."
                )
            if prediction_rows < MIN_PAIR_SCORABLE_ROWS:
                problems.append(
                    f"{pair} has only {prediction_rows} prediction rows for {ready_blocks}."
                )
    free_gib = shutil.disk_usage(
        Path(manifest["storage"]["bulky_artifact_dir"])
    ).free / (1024**3)
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
            columns=["date", "period", *manifest["targets"]],
        )
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        frame["pair"] = pair
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    if output.duplicated(["pair", "date"]).any():
        raise ValueError("Duplicate pair/date rows in Generation 11 target cache.")
    return output


def group_definitions(cohort: str, pairs: Sequence[str]) -> list[dict[str, Any]]:
    groups = g7f.group_definitions(cohort, pairs)
    primary_id = "full_normal_cohort" if cohort == "normal" else "frozen_top_ten_memes"
    output = []
    for group in groups:
        output.append(
            {
                **group,
                "formal_decision_group": group["group_id"] == primary_id,
                "claim_scope": (
                    "frozen_full_cohort"
                    if group["group_id"] == primary_id
                    else "descriptive_predeclared_subgroup"
                ),
            }
        )
    if sum(bool(item["formal_decision_group"]) for item in output) != 1:
        raise ValueError(f"Generation 11 primary group drift for {cohort}.")
    return output


def rank_auc(actual: Series, score: Series) -> float:
    positive = actual.eq(1.0)
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    if not n_positive or not n_negative:
        return np.nan
    ranks = score.rank(method="average")
    rank_sum = float(ranks.loc[positive].sum())
    return (
        rank_sum - n_positive * (n_positive + 1) / 2
    ) / (n_positive * n_negative)


def binary_prediction_metrics(actual: Series, prediction: Series) -> dict[str, Any]:
    numeric = DataFrame({"actual": actual, "prediction": prediction}).apply(
        pd.to_numeric, errors="coerce"
    )
    numeric = numeric.loc[np.isfinite(numeric).all(axis=1)]
    if numeric.empty:
        return {
            "rows": 0,
            "base_rate": np.nan,
            "clipped_mae": np.nan,
            "brier_error": np.nan,
            "rank_auc": np.nan,
            "accuracy_at_half": np.nan,
            "majority_accuracy": np.nan,
            "accuracy_gain_over_majority": np.nan,
            "balanced_accuracy_at_half": np.nan,
            "top_quintile_rate": np.nan,
            "bottom_quintile_rate": np.nan,
            "top_minus_bottom_rate": np.nan,
        }
    y = numeric["actual"].clip(0.0, 1.0)
    raw = numeric["prediction"]
    probability = raw.clip(0.0, 1.0)
    predicted = probability.ge(0.5)
    positive = y.eq(1.0)
    negative = y.eq(0.0)
    sensitivity = float(predicted.loc[positive].mean()) if positive.any() else np.nan
    specificity = float((~predicted.loc[negative]).mean()) if negative.any() else np.nan
    balanced = (
        float(np.nanmean([sensitivity, specificity]))
        if positive.any() and negative.any()
        else np.nan
    )
    rank_fraction = raw.rank(method="average", pct=True)
    bottom = y.loc[rank_fraction.le(0.2)]
    top = y.loc[rank_fraction.gt(0.8)]
    base_rate = float(y.mean())
    accuracy = float(predicted.eq(positive).mean())
    majority = max(base_rate, 1.0 - base_rate)
    top_rate = float(top.mean()) if len(top) else np.nan
    bottom_rate = float(bottom.mean()) if len(bottom) else np.nan
    return {
        "rows": len(y),
        "base_rate": base_rate,
        "clipped_mae": float((y - probability).abs().mean()),
        "brier_error": float(((y - probability) ** 2).mean()),
        "rank_auc": rank_auc(y, raw),
        "accuracy_at_half": accuracy,
        "majority_accuracy": majority,
        "accuracy_gain_over_majority": accuracy - majority,
        "balanced_accuracy_at_half": balanced,
        "top_quintile_rate": top_rate,
        "bottom_quintile_rate": bottom_rate,
        "top_minus_bottom_rate": top_rate - bottom_rate,
    }


def prefixed(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def score_comparisons(
    *,
    manifest: dict[str, Any],
    predictions: dict[str, DataFrame],
    actual: DataFrame,
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    pair_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    eligibility_rows: list[dict[str, Any]] = []
    reaction_rows: list[dict[str, Any]] = []
    groups = group_definitions(str(manifest["cohort"]), manifest["pairs"])
    primary = next(item for item in groups if item["formal_decision_group"])
    for definition in manifest["comparisons"]:
        targets = tuple(str(target) for target in definition["targets"])
        fair, audit = g7f.fair_comparison_frame(
            candidate=predictions[str(definition["candidate"])],
            baseline=predictions[str(definition["baseline"])],
            actual=actual,
            targets=targets,
        )
        eligibility_rows.append({**definition, **audit})
        for (pair, period), frame in fair.groupby(
            ["pair", "period"], sort=False, observed=True
        ):
            for target in targets:
                pair_rows.append(
                    {
                        **definition,
                        "pair": pair,
                        "period": period,
                        "target": target,
                        **g7f.regression_metrics(
                            frame,
                            target=target,
                            candidate_column=f"{target}__candidate",
                            baseline_column=f"{target}__baseline",
                        ),
                    }
                )
        for group in groups:
            for period in manifest["validation_periods"]:
                for target in targets:
                    row = g7f.group_comparison_row(
                        fair,
                        manifest=manifest,
                        definition=definition,
                        target=target,
                        period=period,
                        group=group,
                    )
                    row["formal_decision_group"] = group["formal_decision_group"]
                    row["claim_scope"] = group["claim_scope"]
                    group_rows.append(row)
        for period in manifest["validation_periods"]:
            selected = fair.loc[
                fair["pair"].isin(primary["members"])
                & fair["period"].eq(period)
            ]
            for target in targets:
                if "reaction_h" not in target:
                    continue
                candidate_metrics = binary_prediction_metrics(
                    selected[f"{target}__actual"],
                    selected[f"{target}__candidate"],
                )
                baseline_metrics = binary_prediction_metrics(
                    selected[f"{target}__actual"],
                    selected[f"{target}__baseline"],
                )
                reaction_rows.append(
                    {
                        **definition,
                        "target": target,
                        "period": period,
                        "group_id": primary["group_id"],
                        **prefixed("candidate", candidate_metrics),
                        **prefixed("baseline", baseline_metrics),
                        "candidate_minus_baseline_auc": (
                            candidate_metrics["rank_auc"]
                            - baseline_metrics["rank_auc"]
                        ),
                        "candidate_minus_baseline_balanced_accuracy": (
                            candidate_metrics["balanced_accuracy_at_half"]
                            - baseline_metrics["balanced_accuracy_at_half"]
                        ),
                    }
                )
    return (
        DataFrame.from_records(pair_rows),
        DataFrame.from_records(group_rows),
        DataFrame.from_records(eligibility_rows),
        DataFrame.from_records(reaction_rows),
    )


def route_decisions(manifest: dict[str, Any], scores: DataFrame) -> DataFrame:
    formal = scores.loc[scores["formal_decision_group"].astype(bool)].copy()
    keys = ["question_id", "route_id", "route_type", "target", "seed"]
    expected_periods = set(manifest["validation_periods"])
    rows: list[dict[str, Any]] = []
    for key, frame in formal.groupby(keys, dropna=False):
        values = dict(zip(keys, key, strict=True))
        expected_controls = int(frame["expected_controls_for_route"].iloc[0])
        controls_present = int(frame["comparison_id"].nunique())
        periods_complete = all(
            set(group["period"]) == expected_periods
            for _, group in frame.groupby("comparison_id", observed=True)
        )
        strict = (
            controls_present == expected_controls
            and periods_complete
            and frame["strict_period_pass"].fillna(False).astype(bool).all()
        )
        point = (
            controls_present == expected_controls
            and periods_complete
            and frame["provisional_period_pass"].fillna(False).astype(bool).all()
        )
        if strict:
            status = "strict_initial_lead_pending_confirmation"
        elif point:
            status = "point_initial_lead_pending_confirmation"
        else:
            status = "not_retained_in_initial_attribution"
        failed = frame.loc[
            ~frame["provisional_period_pass"].fillna(False).astype(bool),
            [
                "comparison_id",
                "control_type",
                "period",
                "rows",
                "positive_coins",
                "equal_coin_paired_mae_gain",
                "bootstrap_lower",
                "not_dominated_by_one_coin",
            ],
        ]
        rows.append(
            {
                **values,
                "plain_question": frame["plain_question"].iloc[0],
                "status": status,
                "expected_controls": expected_controls,
                "controls_present": controls_present,
                "both_validation_periods_present": periods_complete,
                "all_controls_strict": strict,
                "all_controls_point_positive": point,
                "minimum_equal_coin_paired_mae_gain": float(
                    frame["equal_coin_paired_mae_gain"].min()
                ),
                "minimum_bootstrap_lower": float(
                    frame["bootstrap_lower"].min()
                ),
                "failed_point_checks": json.dumps(
                    failed.to_dict("records"),
                    sort_keys=True,
                    default=g0.json_default,
                ),
            }
        )
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def score_run(manifest: dict[str, Any], *, record_dir: Path) -> dict[str, Any]:
    predictions: dict[str, DataFrame] = {}
    audits: list[dict[str, Any]] = []
    for item in manifest["commands"]:
        frame, audit = g8.load_predictions(
            Path(item["model_dir"]), tuple(str(pair) for pair in item["pairs"])
        )
        audit["profile_id"] = item["profile_id"]
        audits.append(audit)
        if frame.empty:
            raise ValueError(
                f"No predictions for Generation 11 profile {item['profile_id']}"
            )
        missing = sorted(set(item["targets"]).difference(frame.columns))
        if missing:
            raise ValueError(f"Profile {item['profile_id']} lacks targets: {missing}")
        predictions[str(item["profile_id"])] = frame
    actual = load_actual(manifest)
    pair_scores, group_scores, eligibility, reaction = score_comparisons(
        manifest=manifest,
        predictions=predictions,
        actual=actual,
    )
    decisions = route_decisions(manifest, group_scores)
    paths = {
        "prediction_audit": record_dir / "g11_prediction_audit.csv",
        "comparison_eligibility": record_dir / "g11_comparison_eligibility.csv",
        "pair_scores": record_dir / "g11_pair_scores.csv",
        "group_scores": record_dir / "g11_group_scores.csv",
        "reaction_diagnostics": record_dir / "g11_reaction_diagnostics.csv",
        "route_decisions": record_dir / "g11_route_decisions.csv",
    }
    g0.atomic_write_csv(DataFrame.from_records(audits), paths["prediction_audit"])
    g0.atomic_write_csv(eligibility, paths["comparison_eligibility"])
    g0.atomic_write_csv(pair_scores, paths["pair_scores"])
    g0.atomic_write_csv(group_scores, paths["group_scores"])
    g0.atomic_write_csv(reaction, paths["reaction_diagnostics"])
    g0.atomic_write_csv(decisions, paths["route_decisions"])
    strict = int(decisions["all_controls_strict"].astype(bool).sum())
    point = int(
        (
            decisions["all_controls_point_positive"].astype(bool)
            & ~decisions["all_controls_strict"].astype(bool)
        ).sum()
    )
    unequal = int((~eligibility["identical_prediction_keys"]).sum())
    result = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "created_at_utc": g0.utc_now(),
        "status": (
            "completed_generation11_technical_smoke_not_evidence"
            if manifest["technical_smoke_not_evidence"]
            else "completed_generation11_initial_freqai_attribution"
        ),
        "technical_smoke_not_evidence": manifest["technical_smoke_not_evidence"],
        "evidence_label": manifest["evidence_label"],
        "cohort": manifest["cohort"],
        "profiles_completed": len(predictions),
        "questions_completed": len(manifest["questions"]),
        "routes_completed": len(manifest["routes"]),
        "comparisons_completed": len(manifest["comparisons"]),
        "target_routes_scored": len(decisions),
        "strict_initial_leads": strict,
        "point_initial_leads": point,
        "not_retained": len(decisions) - strict - point,
        "retained_route_targets": [
            f"{row.route_id}|{row.target}"
            for row in decisions.itertuples(index=False)
            if row.all_controls_point_positive
        ],
        "integrity": {
            "unequal_prediction_key_comparisons": unequal,
            "duplicate_prediction_rows_removed": int(
                sum(item.get("duplicate_pair_date_rows", 0) for item in audits)
            ),
            "all_profile_commands_terminal": all(
                item.get("status") == "completed" for item in manifest["commands"]
            ),
            "profile_registry_frozen_before_model_outcomes": True,
            "profit_used": False,
            "future_signed_direction": False,
        },
        "artifacts": {key: artifact(path) for key, path in paths.items()},
        "interpretation_boundary": (
            "Exploratory direction-neutral attribution on reused chronological periods. "
            "Reaction classification diagnostics are reported against their base rates, "
            "but no 55% reaction-only number is a direction result, profit result, entry, "
            "exit, strategy, or promotion decision."
        ),
    }
    result_path = record_dir / "g11_freqai_result.json"
    g0.atomic_write_json(result, result_path)
    return {**result, "result_path": str(result_path.resolve())}


def run_cohort(
    *,
    run_id: str,
    cohort: str,
    base_config: Path,
    python_exe: Path,
    profile_workers: int,
    technical_smoke: bool,
    prepare_only: bool,
) -> dict[str, Any]:
    manifest, manifest_path, _ = build_manifest(
        run_id=run_id,
        cohort=cohort,
        base_config=base_config,
        python_exe=python_exe,
        profile_workers=profile_workers,
        technical_smoke=technical_smoke,
    )
    audit = preflight_run(manifest, python_exe=python_exe)
    audit_path = manifest_path.parent / "g11_freqai_preflight.json"
    g0.atomic_write_json(audit, audit_path)
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
            "manifest": str(manifest_path.resolve()),
            "profiles": len(manifest["commands"]),
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


def reaction_skill_summary(results: Sequence[dict[str, Any]]) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        path = Path(result["artifacts"]["reaction_diagnostics"]["path"])
        diagnostics = pd.read_csv(path)
        unique_candidate = diagnostics.drop_duplicates(
            ["route_id", "target", "period"]
        )
        for route_id, route in diagnostics.groupby("route_id", observed=True):
            candidate = unique_candidate.loc[
                unique_candidate["route_id"].eq(route_id)
            ]
            rows.append(
                {
                    "cohort": result["cohort"],
                    "route_id": route_id,
                    "reaction_target_period_cells": len(candidate),
                    "minimum_balanced_accuracy": float(
                        candidate["candidate_balanced_accuracy_at_half"].min()
                    ),
                    "maximum_balanced_accuracy": float(
                        candidate["candidate_balanced_accuracy_at_half"].max()
                    ),
                    "minimum_rank_auc": float(candidate["candidate_rank_auc"].min()),
                    "maximum_rank_auc": float(candidate["candidate_rank_auc"].max()),
                    "minimum_top_minus_bottom_reaction_rate": float(
                        candidate["candidate_top_minus_bottom_rate"].min()
                    ),
                    "minimum_auc_gain_over_all_controls": float(
                        route["candidate_minus_baseline_auc"].min()
                    ),
                    "minimum_balanced_accuracy_gain_over_all_controls": float(
                        route[
                            "candidate_minus_baseline_balanced_accuracy"
                        ].min()
                    ),
                    "minimum_accuracy_gain_over_majority": float(
                        candidate["candidate_accuracy_gain_over_majority"].min()
                    ),
                }
            )
    return DataFrame.from_records(rows).sort_values(["route_id", "cohort"])


def plain_route_summary(joint: DataFrame) -> DataFrame:
    statuses = (
        "strict_in_both_cohorts_pending_confirmation",
        "point_positive_in_both_cohorts_pending_confirmation",
        "normal_cohort_only_lead_pending_confirmation",
        "meme_cohort_only_lead_pending_confirmation",
        "not_retained_after_joint_initial_batch",
    )
    rows: list[dict[str, Any]] = []
    for route_id, route in joint.groupby("route_id", observed=True):
        row: dict[str, Any] = {
            "route_id": route_id,
            "targets_reviewed": len(route),
        }
        for status in statuses:
            selected = route.loc[route["status"].eq(status), "target"]
            row[f"count__{status}"] = len(selected)
            row[f"targets__{status}"] = ",".join(selected)
        rows.append(row)
    return DataFrame.from_records(rows).sort_values("route_id")


def joint_review(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if {item.get("cohort") for item in results} != {"normal", "meme"}:
        raise ValueError("Generation 11 joint review requires both complete cohorts.")
    if any(item.get("technical_smoke_not_evidence") for item in results):
        raise ValueError("A technical smoke cannot enter the Generation 11 joint review.")
    frames = []
    for result in results:
        path = Path(result["artifacts"]["route_decisions"]["path"])
        frame = pd.read_csv(path)
        frame["cohort"] = result["cohort"]
        frames.append(frame)
    decisions = pd.concat(frames, ignore_index=True)
    rows: list[dict[str, Any]] = []
    for (route_id, target), frame in decisions.groupby(
        ["route_id", "target"], observed=True
    ):
        by_cohort = frame.set_index("cohort")
        normal = by_cohort.loc["normal"]
        meme = by_cohort.loc["meme"]
        normal_point = bool(normal["all_controls_point_positive"])
        meme_point = bool(meme["all_controls_point_positive"])
        normal_strict = bool(normal["all_controls_strict"])
        meme_strict = bool(meme["all_controls_strict"])
        if normal_strict and meme_strict:
            status = "strict_in_both_cohorts_pending_confirmation"
        elif normal_point and meme_point:
            status = "point_positive_in_both_cohorts_pending_confirmation"
        elif normal_point:
            status = "normal_cohort_only_lead_pending_confirmation"
        elif meme_point:
            status = "meme_cohort_only_lead_pending_confirmation"
        else:
            status = "not_retained_after_joint_initial_batch"
        rows.append(
            {
                "route_id": route_id,
                "target": target,
                "status": status,
                "normal_status": normal["status"],
                "meme_status": meme["status"],
                "normal_minimum_mae_gain": normal[
                    "minimum_equal_coin_paired_mae_gain"
                ],
                "meme_minimum_mae_gain": meme[
                    "minimum_equal_coin_paired_mae_gain"
                ],
                "retained_for_possible_confirmation": normal_point or meme_point,
            }
        )
    joint = DataFrame.from_records(rows).sort_values(["route_id", "target"])
    record_dir = RECORD_ROOT / "g11_initial_joint_review_20260822a"
    record_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = record_dir / "g11_joint_route_decisions.csv"
    route_summary_path = record_dir / "g11_plain_route_summary.csv"
    reaction_summary_path = record_dir / "g11_reaction_skill_summary.csv"
    g0.atomic_write_csv(joint, decisions_path)
    g0.atomic_write_csv(plain_route_summary(joint), route_summary_path)
    g0.atomic_write_csv(reaction_skill_summary(results), reaction_summary_path)
    retained = joint.loc[joint["retained_for_possible_confirmation"].astype(bool)]
    result_paths = [
        Path(
            item.get("result_path")
            or RECORD_ROOT / str(item["run_id"]) / "g11_freqai_result.json"
        )
        for item in results
    ]
    output = {
        "schema_version": 1,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation11_normal_meme_joint_freqai_review",
        "cohorts_completed_before_joint_review": ["normal", "meme"],
        "route_targets_reviewed": len(joint),
        "retained_route_targets": len(retained),
        "strict_in_both_cohorts": int(
            joint["status"].eq(
                "strict_in_both_cohorts_pending_confirmation"
            ).sum()
        ),
        "point_positive_in_both_cohorts": int(
            joint["status"].isin(
                {
                    "strict_in_both_cohorts_pending_confirmation",
                    "point_positive_in_both_cohorts_pending_confirmation",
                }
            ).sum()
        ),
        "normal_only_leads": int(
            joint["status"].eq(
                "normal_cohort_only_lead_pending_confirmation"
            ).sum()
        ),
        "meme_only_leads": int(
            joint["status"].eq(
                "meme_cohort_only_lead_pending_confirmation"
            ).sum()
        ),
        "parked_news_route": True,
        "automatic_descendant_launch": False,
        "next_action": (
            "Review all retained and null routes together, then freeze any confirmation "
            "or interaction batch. Do not tune failed routes or launch a descendant from "
            "the first attractive row."
        ),
        "source_results": [artifact(path) for path in result_paths],
        "artifacts": {
            "joint_route_decisions": artifact(decisions_path),
            "plain_route_summary": artifact(route_summary_path),
            "reaction_skill_summary": artifact(reaction_summary_path),
        },
    }
    output_path = record_dir / "g11_joint_freqai_review.json"
    g0.atomic_write_json(output, output_path)
    return {**output, "result_path": str(output_path.resolve())}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen Generation 11 FreqAI attribution batch."
    )
    parser.add_argument("--cohort", choices=("normal", "meme", "all"), default="all")
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
    results = []
    for cohort in cohorts:
        result = run_cohort(
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
    if (
        cohorts == ("normal", "meme")
        and not args.technical_smoke
        and not args.prepare_only
    ):
        results.append(joint_review(results))
    print(json.dumps(results, indent=2, default=g0.json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

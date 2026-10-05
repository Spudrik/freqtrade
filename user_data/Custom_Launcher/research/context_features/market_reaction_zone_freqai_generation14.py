"""Run Generation 14's frozen broad FreqAI combination siblings."""

from __future__ import annotations

# Bound numerical pools before pandas and FreqAI imports.
# ruff: noqa: E402
import argparse
import json
import os
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
    market_reaction_zone_freqai_generation7 as g7f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation8 as g8,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation11 as g11,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation13 as g13,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation13_cache as g13c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation14_cache as g14c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation14_external_readiness as g14e,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation14_freeze as g14z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation14_label_sensitivity as g14l,
)


DEFAULT_CONFIG = g13.DEFAULT_CONFIG
DEFAULT_PYTHON = g13.DEFAULT_PYTHON
STRATEGY_PATH = g13.STRATEGY_PATH
STRATEGY_FILE = STRATEGY_PATH / "MarketReactionZoneFreqAIGeneration14Strategy.py"
STRATEGY_CLASS = "MarketReactionZoneG14ConfigurableFreqAIResearchStrategy"
DATA_DIR = g13.DATA_DIR
RECORD_ROOT = g14z.FREEZE_PATH.parent / "g14_broad_combinations" / "freqai"
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation14_branches"
    / "g14_broad_combinations"
    / "freqai"
)
DEFAULT_RUN_STEM = "g14_broad_freqai_20260822a"
DEFAULT_SMOKE_STEM = "g14_technical_smoke_20260822a"
DEFAULT_JOINT_REVIEW_ID = "g14_joint_review_20260822a"
MAX_WORKERS = 4


def artifact(path: Path) -> dict[str, Any]:
    return g11.artifact(path)


def load_sources(
    *, cohort: str, stage: str
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    frozen = g14z.freeze_generation14()
    if frozen.get("status") != "frozen_before_generation14_outcomes":
        raise ValueError("Generation 14 freeze is invalid.")
    if stage == g14z.MTF_STAGE:
        cache_path = g13c.RECORD_ROOT / f"{cohort}_manifest.json"
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        expected = frozen["source_contracts"][f"generation13_{cohort}_cache"]
        if cache.get("status") != "completed_generation13_cache":
            raise ValueError(f"Generation 13 {cohort} cache is incomplete.")
        if g0.sha256_file(cache_path) != expected["sha256"]:
            raise ValueError(f"Generation 13 {cohort} cache changed after G14 freeze.")
        return frozen, cache, cache_path
    if stage != g14z.LONG_STAGE:
        raise ValueError(stage)
    cache_path = g14c.RECORD_ROOT / f"{cohort}_manifest.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache.get("status") != "completed_generation14_combined_context_cache":
        raise ValueError(f"Generation 14 {cohort} combined cache is incomplete.")
    source = cache["source_contracts"]["generation14_freeze"]
    if g0.sha256_file(g14z.FREEZE_PATH) != source["sha256"]:
        raise ValueError("Generation 14 freeze changed after combined cache construction.")
    return frozen, cache, cache_path


def active_registry(
    frozen: dict[str, Any], *, stage: str, cohort: str
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    comparisons = []
    for item in frozen["comparisons"]:
        if item["stage"] != stage or item["cohort"] != cohort:
            continue
        adapted = dict(item)
        adapted.setdefault("route_type", "bounded_component_or_pairwise_interaction")
        comparisons.append(adapted)
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
    if not profiles or not comparisons:
        raise ValueError(f"No active Generation 14 profiles for {stage}/{cohort}.")
    return profiles, comparisons


def smoke_registry(
    *,
    stage: str,
    profiles: dict[str, dict[str, Any]],
    comparisons: Sequence[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    selected_route = (
        "eight_hour_participation_pressure"
        if stage == g14z.MTF_STAGE
        else "four_hour_level_local_plus_eight_hour_activity"
    )
    selected = [
        dict(item) for item in comparisons if item["route_id"] == selected_route
    ]
    referenced = {
        profile_id
        for item in selected
        for profile_id in (item["candidate"], item["baseline"])
    }
    return (
        {key: value for key, value in profiles.items() if key in referenced},
        selected,
    )


def stage_inventory(cache: dict[str, Any], *, stage: str) -> list[dict[str, Any]]:
    if stage == g14z.LONG_STAGE:
        return [
            {
                "pair": item["pair"],
                **{
                    f"{kind}_path": item[f"{kind}_path"]
                    for kind in ("feature", "event", "evaluation")
                },
                **{
                    f"{kind}_sha256": item[f"{kind}_sha256"]
                    for kind in ("feature", "event", "evaluation")
                },
                "support_path": item["readiness_path"],
                "support_sha256": item["readiness_sha256"],
            }
            for item in cache["inventory"]
        ]
    return [
        {
            "pair": item["pair"],
            **{
                f"{kind}_path": item[f"mtf_{kind}_path"]
                for kind in ("feature", "event", "evaluation")
            },
            **{
                f"{kind}_sha256": item[f"mtf_{kind}_sha256"]
                for kind in ("feature", "event", "evaluation")
            },
            "support_path": item["mtf_support_path"],
            "support_sha256": item["mtf_support_sha256"],
        }
        for item in cache["inventory"]
    ]


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
    maximum_target_horizon_hours: int,
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
        maximum_target_horizon_hours=maximum_target_horizon_hours,
    )
    config["market_reaction_zone_g14"] = config.pop("market_reaction_zone_g13")
    return config


def coverage_supported_pairs(
    *,
    candidate_pairs: Sequence[str],
    event_dir: Path,
    profiles: dict[str, dict[str, Any]],
    timerange: str,
    train_days: int,
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    """Select one outcome-blind common pair surface for every profile in a cell."""
    start_raw, end_raw = timerange.split("-", maxsplit=1)
    prediction_start = pd.Timestamp(start_raw, tz="UTC")
    prediction_end = pd.Timestamp(end_raw, tz="UTC")
    training_start = prediction_start - pd.Timedelta(days=train_days)
    requirements = {
        (tuple(profile["required_ready_blocks"]), tuple(profile["targets"]))
        for profile in profiles.values()
    }
    supported: list[str] = []
    audit: list[dict[str, Any]] = []
    for pair in candidate_pairs:
        path = event_dir / f"{g0.pair_file_stem(pair)}.parquet"
        needed = {
            "date",
            *(target for _, targets in requirements for target in targets),
            *(
                f"ready__{block}"
                for blocks, _ in requirements
                for block in blocks
            ),
        }
        frame = pd.read_parquet(path, columns=sorted(needed))
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        minimum_training = np.inf
        minimum_prediction = np.inf
        failed_requirements = 0
        for ready_blocks, targets in requirements:
            ready_columns = [f"ready__{block}" for block in ready_blocks]
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
            minimum_training = min(minimum_training, training_rows)
            minimum_prediction = min(minimum_prediction, prediction_rows)
            if (
                training_rows < g11.MIN_TRAINING_ROWS
                or prediction_rows < g11.MIN_PAIR_SCORABLE_ROWS
            ):
                failed_requirements += 1
        accepted = failed_requirements == 0
        if accepted:
            supported.append(str(pair))
        audit.append(
            {
                "pair": str(pair),
                "accepted": accepted,
                "minimum_training_rows": int(minimum_training),
                "minimum_prediction_rows": int(minimum_prediction),
                "failed_profile_requirements": failed_requirements,
                "future_outcome_values_read": False,
            }
        )
    return tuple(supported), audit


def build_manifest(
    *,
    run_id: str,
    stage: str,
    window: str,
    cohort: str,
    base_config: Path,
    python_exe: Path,
    profile_workers: int,
    technical_smoke: bool,
) -> tuple[dict[str, Any], Path, Path]:
    frozen, cache, cache_path = load_sources(cohort=cohort, stage=stage)
    profiles, comparisons = active_registry(frozen, stage=stage, cohort=cohort)
    if technical_smoke:
        profiles, comparisons = smoke_registry(
            stage=stage, profiles=profiles, comparisons=comparisons
        )
    settings = g13.stage_settings(
        cohort=cohort, window=window, technical_smoke=technical_smoke
    )
    inventory = stage_inventory(cache, stage=stage)
    record_dir = RECORD_ROOT / run_id
    artifact_dir = ARTIFACT_ROOT / run_id
    manifest_path = record_dir / "g14_freqai_run_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        contracts = existing["source_contracts"]
        for name, path in (
            ("strategy", STRATEGY_FILE),
            ("freeze", g14z.FREEZE_PATH),
            ("cache_manifest", cache_path),
        ):
            if g0.sha256_file(path) != contracts[name]["sha256"]:
                raise ValueError(f"Generation 14 {name} changed: {manifest_path}")
        if any("support_path" not in item for item in contracts["cache_inventory"]):
            existing["source_contracts"]["cache_inventory"] = inventory
            existing["manifest_schema_repair"] = (
                "Added the required existing support-cache path/hash fields before "
                "preflight; commands, profiles, targets, and outcomes were unchanged."
            )
            g0.atomic_write_json(existing, manifest_path)
        if any("route_type" not in item for item in existing["comparisons"]):
            for item in existing["comparisons"]:
                item["route_type"] = "bounded_component_or_pairwise_interaction"
            if "profile_registry" in existing:
                existing["profile_registry"]["comparisons"] = existing["comparisons"]
            existing["manifest_comparison_schema_repair"] = (
                "Added the reporting-only route_type required by the shared scorer; "
                "profiles, features, targets, controls, and trained models were unchanged."
            )
            g0.atomic_write_json(existing, manifest_path)
        return existing, manifest_path, artifact_dir
    record_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    base = json.loads(base_config.read_text(encoding="utf-8"))
    feature_dir = Path(inventory[0]["feature_path"]).parent
    event_dir = Path(inventory[0]["event_path"]).parent
    evaluation_dir = Path(inventory[0]["evaluation_path"]).parent
    candidate_pairs = tuple(str(pair) for pair in cache["pairs"])
    pairs, pair_coverage_audit = coverage_supported_pairs(
        candidate_pairs=candidate_pairs,
        event_dir=event_dir,
        profiles=profiles,
        timerange=str(settings["timerange"]),
        train_days=int(settings["train_days"]),
    )
    if len(pairs) < 5:
        manifest = {
            "schema_version": 1,
            "generation": 14,
            "run_id": run_id,
            "created_at_utc": g0.utc_now(),
            "status": "parked_coverage_before_model_launch",
            "run_stage": stage,
            "evaluation_window": window,
            "technical_smoke_not_evidence": technical_smoke,
            "cohort": cohort,
            "candidate_pairs_before_common_support_gate": list(candidate_pairs),
            "pairs": list(pairs),
            "coverage_excluded_pairs": sorted(
                set(candidate_pairs).difference(pairs)
            ),
            "coverage_pair_audit": pair_coverage_audit,
            "minimum_required_pairs": 5,
            "profiles": list(profiles),
            "profile_count": len(profiles),
            "comparisons": comparisons,
            "comparison_count": len(comparisons),
            "targets": list(next(iter(profiles.values()))["targets"]),
            "future_outcome_values_read_for_coverage": False,
            "research_boundary": dict(frozen["research_boundary"]),
            "source_contracts": {
                "base_config": artifact(base_config),
                "python_executable": str(python_exe.resolve()),
                "strategy": artifact(STRATEGY_FILE),
                "freeze": artifact(g14z.FREEZE_PATH),
                "cache_manifest": artifact(cache_path),
                "cache_inventory": inventory,
            },
            "storage": {
                "record_dir": str(record_dir.resolve()),
                "bulky_artifact_dir": str(artifact_dir.resolve()),
                "feature_cache_dir": str(feature_dir.resolve()),
                "event_cache_dir": str(event_dir.resolve()),
                "evaluation_cache_dir": str(evaluation_dir.resolve()),
                "save_backtest_models": False,
            },
            "commands": [],
        }
        g0.atomic_write_json(manifest, manifest_path)
        return manifest, manifest_path, artifact_dir
    maximum_horizon = 8 if stage == g14z.MTF_STAGE else 48
    commands: list[dict[str, Any]] = []
    for number, (profile_id, profile) in enumerate(profiles.items(), start=1):
        short_id = f"p{number:03d}_{g8.g6f.stable_digest(profile_id, 6)}"
        profile_dir = artifact_dir / "profiles" / short_id
        userdir = profile_dir / "user_data"
        export_dir = profile_dir / "export"
        identifier = f"g14-{g8.g6f.stable_digest(f'{run_id}|{profile_id}', 16)}"
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
                maximum_target_horizon_hours=maximum_horizon,
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
        "generation": 14,
        "run_id": run_id,
        "created_at_utc": g0.utc_now(),
        "status": "prepared",
        "run_stage": stage,
        "evaluation_window": window,
        "technical_smoke_not_evidence": technical_smoke,
        "cohort": cohort,
        "pairs": list(pairs),
        "candidate_pairs_before_common_support_gate": list(candidate_pairs),
        "coverage_excluded_pairs": sorted(set(candidate_pairs).difference(pairs)),
        "coverage_pair_audit": pair_coverage_audit,
        "timerange": str(settings["timerange"]),
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
        "targets": list(next(iter(profiles.values()))["targets"]),
        "seeds": [g14z.SEED],
        "maximum_target_horizon_hours": maximum_horizon,
        "profile_registry_frozen_before_model_outcomes": True,
        "profile_registry": {"profiles": profiles, "comparisons": comparisons},
        "decision_rule": frozen["decision_rule"],
        "research_boundary": dict(frozen["research_boundary"]),
        "recent_periods": list(g14z.g12z.RECENT_PERIODS)
        if window == g14z.RECENT
        else [],
        "long_evaluation_source_timeframes": ["4h"]
        if stage == g14z.LONG_STAGE
        else [],
        "source_contracts": {
            "base_config": artifact(base_config),
            "python_executable": str(python_exe.resolve()),
            "strategy": artifact(STRATEGY_FILE),
            "freeze": artifact(g14z.FREEZE_PATH),
            "cache_manifest": artifact(cache_path),
            "cache_inventory": inventory,
        },
        "storage": {
            "record_dir": str(record_dir.resolve()),
            "bulky_artifact_dir": str(artifact_dir.resolve()),
            "feature_cache_dir": str(feature_dir.resolve()),
            "event_cache_dir": str(event_dir.resolve()),
            "evaluation_cache_dir": str(evaluation_dir.resolve()),
            "save_backtest_models": False,
        },
        "runtime_preparation_snapshot": g7f.runtime_snapshot(),
        "commands": commands,
    }
    g0.atomic_write_json(manifest, manifest_path)
    return manifest, manifest_path, artifact_dir


def assign_evaluation_periods(actual: DataFrame, manifest: dict[str, Any]) -> DataFrame:
    if manifest["evaluation_window"] != g14z.RECENT:
        return actual
    output = actual.copy()
    output["period"] = "outside_generation14_recent_windows"
    for definition in manifest["recent_periods"]:
        start = pd.Timestamp(definition["start"])
        stop = pd.Timestamp(definition["stop"])
        selected = output["date"].ge(start) & output["date"].lt(stop)
        output.loc[selected, "period"] = definition["period"]
    return output


def load_actual(manifest: dict[str, Any]) -> DataFrame:
    frames: list[DataFrame] = []
    evaluation_dir = Path(manifest["storage"]["evaluation_cache_dir"])
    extra = (
        ["anchor_source_timeframe", "market_group", "smart_contract_platform"]
        if manifest["run_stage"] == g14z.LONG_STAGE
        else ["market_group", "smart_contract_platform"]
    )
    for pair in manifest["pairs"]:
        frame = pd.read_parquet(
            evaluation_dir / f"{g0.pair_file_stem(pair)}.parquet",
            columns=["date", "period", *extra, *manifest["targets"]],
        )
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        frame["pair"] = pair
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    if output.duplicated(["pair", "date"]).any():
        raise ValueError("Duplicate pair/date rows in Generation 14 evaluation cache.")
    output = assign_evaluation_periods(output, manifest)
    if manifest["run_stage"] == g14z.LONG_STAGE:
        output = output.loc[output["anchor_source_timeframe"].eq("4h")].copy()
    return output


def preflight_run(manifest: dict[str, Any], *, python_exe: Path) -> dict[str, Any]:
    audit = g11.preflight_run(manifest, python_exe=python_exe)
    problems = list(audit["problems"])
    if not STRATEGY_FILE.is_file():
        problems.append(f"Missing Generation 14 strategy: {STRATEGY_FILE}")
    required_columns = {
        column for item in manifest["commands"] for column in item["feature_columns"]
    }
    for item in manifest["source_contracts"]["cache_inventory"]:
        try:
            pd.read_parquet(item["feature_path"], columns=sorted(required_columns))
        except Exception as exc:
            problems.append(f"{item['pair']} lacks exact G14 feature columns: {exc}")
    actual = load_actual(manifest)
    support_rows: list[dict[str, Any]] = []
    for period in manifest["validation_periods"]:
        selected = actual.loc[actual["period"].eq(period)]
        counts = selected.groupby("pair", observed=True).size()
        row = {
            "period": period,
            "anchor_source_timeframe": "4h"
            if manifest["run_stage"] == g14z.LONG_STAGE
            else "all",
            "rows": len(selected),
            "coins": int((counts > 0).sum()),
            "minimum_pair_rows": int(counts.min()) if len(counts) else 0,
        }
        support_rows.append(row)
        minimum_rows = 20 if manifest["run_stage"] == g14z.LONG_STAGE else 50
        if row["rows"] < minimum_rows or row["coins"] < 5:
            problems.append(f"Insufficient Generation 14 evaluation support: {row}")
    return {
        **audit,
        "passed": not problems,
        "problems": problems,
        "exact_feature_columns": len(required_columns),
        "evaluation_support": support_rows,
    }


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
            raise ValueError(f"No Generation 14 predictions for {item['profile_id']}")
        predictions[str(item["profile_id"])] = frame
    actual = load_actual(manifest)
    pair, group, eligibility, reaction = g11.score_comparisons(
        manifest=manifest, predictions=predictions, actual=actual
    )
    decisions = g11.route_decisions(manifest, group)
    source_scope = "4h" if manifest["run_stage"] == g14z.LONG_STAGE else "all"
    for frame in (pair, group, eligibility, reaction, decisions):
        frame["anchor_source_timeframe"] = source_scope
    paths = {
        "prediction_audit": record_dir / "g14_prediction_audit.csv",
        "comparison_eligibility": record_dir / "g14_comparison_eligibility.csv",
        "pair_scores": record_dir / "g14_pair_scores.csv",
        "group_scores": record_dir / "g14_group_scores.csv",
        "reaction_diagnostics": record_dir / "g14_reaction_diagnostics.csv",
        "route_decisions": record_dir / "g14_route_decisions.csv",
    }
    g0.atomic_write_csv(DataFrame.from_records(audits), paths["prediction_audit"])
    g0.atomic_write_csv(eligibility, paths["comparison_eligibility"])
    g0.atomic_write_csv(pair, paths["pair_scores"])
    g0.atomic_write_csv(group, paths["group_scores"])
    g0.atomic_write_csv(reaction, paths["reaction_diagnostics"])
    g0.atomic_write_csv(decisions, paths["route_decisions"])
    strict = int(decisions["all_controls_strict"].astype(bool).sum())
    point = int(
        (
            decisions["all_controls_point_positive"].astype(bool)
            & ~decisions["all_controls_strict"].astype(bool)
        ).sum()
    )
    result = {
        "schema_version": 1,
        "generation": 14,
        "run_id": manifest["run_id"],
        "created_at_utc": g0.utc_now(),
        "status": (
            "completed_generation14_technical_smoke_not_evidence"
            if manifest["technical_smoke_not_evidence"]
            else "completed_generation14_freqai_cell"
        ),
        "technical_smoke_not_evidence": manifest["technical_smoke_not_evidence"],
        "run_stage": manifest["run_stage"],
        "evaluation_window": manifest["evaluation_window"],
        "cohort": manifest["cohort"],
        "profiles_completed": len(predictions),
        "routes_completed": len(manifest["routes"]),
        "comparisons_completed": len(manifest["comparisons"]),
        "target_routes_scored": len(decisions),
        "strict_initial_leads": strict,
        "point_initial_leads": point,
        "not_retained": len(decisions) - strict - point,
        "integrity": {
            "unequal_prediction_key_comparisons": int(
                (~eligibility["identical_prediction_keys"]).sum()
            ),
            "duplicate_prediction_rows_removed": int(
                sum(item.get("duplicate_pair_date_rows", 0) for item in audits)
            ),
            "all_profile_commands_terminal": all(
                item.get("status") == "completed" for item in manifest["commands"]
            ),
            "profit_used": False,
            "future_signed_direction": False,
            "joint_55_percent_direction_target_reached": False,
        },
        "artifacts": {key: artifact(path) for key, path in paths.items()},
    }
    result_path = record_dir / "g14_freqai_result.json"
    g0.atomic_write_json(result, result_path)
    return {**result, "result_path": str(result_path.resolve())}


def run_cell(
    *,
    run_id: str,
    stage: str,
    window: str,
    cohort: str,
    base_config: Path,
    python_exe: Path,
    profile_workers: int,
    technical_smoke: bool,
    prepare_only: bool,
) -> dict[str, Any]:
    manifest, manifest_path, _ = build_manifest(
        run_id=run_id,
        stage=stage,
        window=window,
        cohort=cohort,
        base_config=base_config,
        python_exe=python_exe,
        profile_workers=profile_workers,
        technical_smoke=technical_smoke,
    )
    if str(manifest["status"]).startswith("completed_generation14_") and manifest.get(
        "result"
    ):
        result_path = manifest_path.parent / "g14_freqai_result.json"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        return {
            **dict(manifest["result"]),
            "result_path": str(result_path.resolve()),
        }
    if manifest["status"] == "parked_coverage_before_model_launch":
        result = {
            "schema_version": 1,
            "generation": 14,
            "run_id": run_id,
            "created_at_utc": g0.utc_now(),
            "status": "completed_generation14_freqai_cell_parked_coverage",
            "run_stage": stage,
            "evaluation_window": window,
            "cohort": cohort,
            "candidate_pairs": len(
                manifest["candidate_pairs_before_common_support_gate"]
            ),
            "supported_pairs": len(manifest["pairs"]),
            "coverage_excluded_pairs": manifest["coverage_excluded_pairs"],
            "future_outcome_values_read_for_coverage": False,
            "manifest": str(manifest_path.resolve()),
        }
        result_path = manifest_path.parent / "g14_freqai_result.json"
        g0.atomic_write_json(result, result_path)
        manifest["status"] = result["status"]
        manifest["result"] = result
        g0.atomic_write_json(manifest, manifest_path)
        return {**result, "result_path": str(result_path.resolve())}
    audit = preflight_run(manifest, python_exe=python_exe)
    g0.atomic_write_json(audit, manifest_path.parent / "g14_freqai_preflight.json")
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
            "run_stage": stage,
            "evaluation_window": window,
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


def joint_stage_decisions(results: Sequence[dict[str, Any]], stage: str) -> DataFrame:
    frames: list[DataFrame] = []
    for result in results:
        if result["run_stage"] != stage:
            continue
        if "route_decisions" not in result.get("artifacts", {}):
            continue
        frame = pd.read_csv(result["artifacts"]["route_decisions"]["path"])
        frame["cohort"] = result["cohort"]
        frame["evaluation_window"] = result["evaluation_window"]
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    keys = ["route_id", "target", "anchor_source_timeframe"]
    rows: list[dict[str, Any]] = []
    expected = {
        ("normal", g14z.STANDARD),
        ("normal", g14z.RECENT),
        ("meme", g14z.STANDARD),
    }
    parked_cells = {
        (item["cohort"], item["evaluation_window"])
        for item in results
        if item["run_stage"] == stage
        and item["status"] == "completed_generation14_freqai_cell_parked_coverage"
    }
    supported_expected = expected.difference(parked_cells)
    for key, cell in combined.groupby(keys, observed=True, sort=False):
        observed = set(zip(cell["cohort"], cell["evaluation_window"], strict=False))
        complete = observed == expected
        complete_supported = observed == supported_expected
        point_all = bool(
            complete and cell["all_controls_point_positive"].astype(bool).all()
        )
        strict_all = bool(complete and cell["all_controls_strict"].astype(bool).all())
        point_supported = bool(
            complete_supported
            and cell["all_controls_point_positive"].astype(bool).all()
        )
        strict_supported = bool(
            complete_supported and cell["all_controls_strict"].astype(bool).all()
        )
        standard = cell.loc[cell["evaluation_window"].eq(g14z.STANDARD)]
        standard_point = bool(
            set(standard["cohort"]) == {"normal", "meme"}
            and standard["all_controls_point_positive"].astype(bool).all()
        )
        rows.append(
            {
                "route_id": key[0],
                "target": key[1],
                "anchor_source_timeframe": key[2],
                "complete_three_cell_ladder": complete,
                "coverage_parked_cells": ",".join(
                    f"{cohort}:{window}" for cohort, window in sorted(parked_cells)
                ),
                "complete_supported_cell_ladder": complete_supported,
                "point_all_three_cells": point_all,
                "strict_all_three_cells": strict_all,
                "point_all_supported_cells": point_supported,
                "strict_all_supported_cells": strict_supported,
                "standard_both_cohorts_point": standard_point,
                "minimum_mae_gain": float(
                    cell["minimum_equal_coin_paired_mae_gain"].min()
                ),
                "minimum_bootstrap_lower": float(cell["minimum_bootstrap_lower"].min()),
            }
        )
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def coin_stability(results: Sequence[dict[str, Any]]) -> tuple[DataFrame, DataFrame]:
    frames: list[DataFrame] = []
    for result in results:
        if "pair_scores" not in result.get("artifacts", {}):
            continue
        frame = pd.read_csv(result["artifacts"]["pair_scores"]["path"])
        frame["cohort"] = result["cohort"]
        frame["evaluation_window"] = result["evaluation_window"]
        frame["run_stage"] = result["run_stage"]
        frames.append(frame)
    pairs = pd.concat(frames, ignore_index=True)
    group_rows: list[dict[str, Any]] = []
    for name, members in g14z.NORMAL_GROUPS.items():
        selected = pairs.loc[pairs["cohort"].eq("normal") & pairs["pair"].isin(members)]
        keys = [
            "run_stage",
            "evaluation_window",
            "route_id",
            "target",
            "period",
            "control_type",
        ]
        for key, cell in selected.groupby(keys, observed=True, sort=False):
            group_rows.append(
                {
                    "coin_group": name,
                    **dict(zip(keys, key, strict=False)),
                    "coins": int(cell["pair"].nunique()),
                    "rows": int(cell["rows"].sum()),
                    "mean_paired_mae_gain": float(cell["paired_mae_gain"].mean()),
                    "positive_coins": int(
                        cell.groupby("pair", observed=True)["paired_mae_gain"]
                        .mean()
                        .gt(0)
                        .sum()
                    ),
                }
            )
    group_frame = DataFrame.from_records(group_rows)

    leave_rows: list[dict[str, Any]] = []
    keys = [
        "run_stage",
        "cohort",
        "evaluation_window",
        "route_id",
        "target",
        "period",
        "control_type",
    ]
    for key, cell in pairs.groupby(keys, observed=True, sort=False):
        coin_means = cell.groupby("pair", observed=True)["paired_mae_gain"].mean()
        if len(coin_means) < 5:
            continue
        for excluded in coin_means.index:
            retained = coin_means.drop(excluded)
            leave_rows.append(
                {
                    **dict(zip(keys, key, strict=False)),
                    "excluded_pair": excluded,
                    "retained_coins": len(retained),
                    "leave_one_out_mean_gain": float(retained.mean()),
                }
            )
    return group_frame, DataFrame.from_records(leave_rows)


def validate_terminal_siblings() -> tuple[dict[str, Any], dict[str, Any]]:
    label_path = g14l.RECORD_ROOT / g14l.DEFAULT_RUN_ID / "g14_label_sensitivity_result.json"
    external_path = (
        g14e.RECORD_ROOT / g14e.DEFAULT_RUN_ID / "g14_external_readiness_result.json"
    )
    label = json.loads(label_path.read_text(encoding="utf-8"))
    external = json.loads(external_path.read_text(encoding="utf-8"))
    if label.get("status") != "completed_generation14_label_sensitivity":
        raise ValueError("Generation 14 label-sensitivity sibling is not terminal.")
    if external.get("status") != "completed_generation14_external_readiness":
        raise ValueError("Generation 14 external-readiness sibling is not terminal.")
    return label, external


def joint_review(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    expected = {
        (stage, window, cohort)
        for stage in g14z.STAGES
        for window, cohort in (
            (g14z.STANDARD, "normal"),
            (g14z.RECENT, "normal"),
            (g14z.STANDARD, "meme"),
        )
    }
    observed = {
        (item["run_stage"], item["evaluation_window"], item["cohort"])
        for item in results
    }
    if observed != expected:
        raise ValueError("Generation 14 joint review requires all six FreqAI cells.")
    label, external = validate_terminal_siblings()
    record_dir = RECORD_ROOT / DEFAULT_JOINT_REVIEW_ID
    record_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    summaries: dict[str, Any] = {}
    for stage in g14z.STAGES:
        decisions = joint_stage_decisions(results, stage)
        short = "mtf" if stage == g14z.MTF_STAGE else "long"
        path = record_dir / f"g14_joint_{short}_decisions.csv"
        g0.atomic_write_csv(decisions, path)
        outputs[short] = path
        summaries[short] = {
            "rows": len(decisions),
            "strict_all_three": int(decisions["strict_all_three_cells"].sum()),
            "point_all_three": int(decisions["point_all_three_cells"].sum()),
            "standard_both_cohorts": int(
                decisions["standard_both_cohorts_point"].sum()
            ),
            "point_all_supported_cells": int(
                decisions["point_all_supported_cells"].sum()
            ),
            "strict_all_supported_cells": int(
                decisions["strict_all_supported_cells"].sum()
            ),
        }
    groups, leave_one_out = coin_stability(results)
    group_path = record_dir / "g14_predeclared_coin_group_scores.csv"
    leave_path = record_dir / "g14_leave_one_coin_out_scores.csv"
    g0.atomic_write_csv(groups, group_path)
    g0.atomic_write_csv(leave_one_out, leave_path)
    outputs["coin_groups"] = group_path
    outputs["leave_one_coin_out"] = leave_path
    output = {
        "schema_version": 1,
        "generation": 14,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation14_joint_review",
        "all_frozen_siblings_completed_or_parked_before_review": True,
        "freqai_summary": summaries,
        "label_sensitivity_summary": {
            "robust_point_rows": label["robust_point_rows"],
            "robust_strict_rows": label["robust_strict_rows"],
        },
        "external_readiness_decisions": external["decisions"],
        "automatic_descendant_launch": False,
        "research_boundary": {
            "direction_tested": False,
            "joint_55_percent_direction_target_reached": False,
            "profit_used": False,
            "trading_promotion": False,
        },
        "source_results": [artifact(Path(item["result_path"])) for item in results],
        "source_label_sensitivity": artifact(Path(label["result_path"]))
        if "result_path" in label
        else artifact(
            g14l.RECORD_ROOT
            / g14l.DEFAULT_RUN_ID
            / "g14_label_sensitivity_result.json"
        ),
        "source_external_readiness": artifact(
            g14e.RECORD_ROOT
            / g14e.DEFAULT_RUN_ID
            / "g14_external_readiness_result.json"
        ),
        "artifacts": {key: artifact(path) for key, path in outputs.items()},
    }
    result_path = record_dir / "g14_joint_review.json"
    g0.atomic_write_json(output, result_path)
    return {**output, "result_path": str(result_path.resolve())}


def requested_runs(
    stage: str, window: str, cohort: str
) -> list[tuple[str, str, str]]:
    runs: list[tuple[str, str, str]] = []
    for selected_stage in g14z.STAGES:
        if stage not in ("all", selected_stage):
            continue
        for selected_window, selected_cohort in (
            (g14z.STANDARD, "normal"),
            (g14z.RECENT, "normal"),
            (g14z.STANDARD, "meme"),
        ):
            if window not in ("all", selected_window):
                continue
            if cohort not in ("all", selected_cohort):
                continue
            runs.append((selected_stage, selected_window, selected_cohort))
    if not runs:
        raise ValueError("No Generation 14 runs match the selection.")
    return runs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run frozen Generation 14 FreqAI cells.")
    parser.add_argument("--stage", choices=("all", *g14z.STAGES), default="all")
    parser.add_argument(
        "--window", choices=("all", g14z.STANDARD, g14z.RECENT), default="all"
    )
    parser.add_argument("--cohort", choices=("all", *g14z.COHORTS), default="all")
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
    runs = requested_runs(args.stage, args.window, args.cohort)
    stem = args.run_stem or (
        DEFAULT_SMOKE_STEM if args.technical_smoke else DEFAULT_RUN_STEM
    )
    results: list[dict[str, Any]] = []
    for stage, window, cohort in runs:
        short_stage = "mtf" if stage == g14z.MTF_STAGE else "long"
        short_window = "standard" if window == g14z.STANDARD else "recent"
        result = run_cell(
            run_id=f"{stem}_{short_stage}_{short_window}_{cohort}",
            stage=stage,
            window=window,
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
    if len(runs) == 6 and not args.technical_smoke and not args.prepare_only:
        results.append(joint_review(results))
    print(json.dumps(results, indent=2, default=g0.json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

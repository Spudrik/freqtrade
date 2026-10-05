"""Run Generation 17's frozen direction-neutral FreqAI branch profiles."""

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
    market_reaction_zone_freqai_generation14 as g14,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation12_freeze as g12z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation13_freeze as g13z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_freeze as g17z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_freqai_cache as g17c,
)


DEFAULT_CONFIG = g13.DEFAULT_CONFIG
DEFAULT_PYTHON = g13.DEFAULT_PYTHON
STRATEGY_PATH = g13.STRATEGY_PATH
STRATEGY_FILE = STRATEGY_PATH / "MarketReactionZoneFreqAIGeneration17Strategy.py"
STRATEGY_CLASS = "MarketReactionZoneG17ConfigurableFreqAIResearchStrategy"
DATA_DIR = g13.DATA_DIR
RECORD_ROOT = g17z.OUTPUT_ROOT / "freqai"
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation17_branches"
    / "g17_broad_branch_layer"
    / "freqai"
)
DEFAULT_RUN_STEM = "g17_broad_freqai_20260822a"
DEFAULT_SMOKE_STEM = "g17_technical_smoke_20260822a"
DEFAULT_JOINT_REVIEW_ID = "g17_freqai_joint_review_20260822a"
MAX_WORKERS = 4


def artifact(path: Path) -> dict[str, Any]:
    return g11.artifact(path)


def load_sources(cohort: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    frozen = g17c.load_freeze()
    cache_path = g17c.RECORD_ROOT / f"{cohort}_manifest.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache.get("status") != "completed_generation17_freqai_cache":
        raise ValueError(f"Generation 17 {cohort} FreqAI cache is not terminal.")
    registry = json.loads(g17c.REGISTRY_PATH.read_text(encoding="utf-8"))
    if registry.get("status") != "frozen_before_generation17_freqai_target_materialization":
        raise ValueError("Generation 17 FreqAI registry is invalid.")
    if (
        g0.sha256_file(g17z.FREEZE_PATH)
        != cache["source_contracts"]["generation17_freeze"]["sha256"]
    ):
        raise ValueError("Generation 17 freeze changed after cache construction.")
    if g0.sha256_file(g17c.REGISTRY_PATH) != cache["profile_registry"]["sha256"]:
        raise ValueError("Generation 17 registry changed after cache construction.")
    return frozen, cache, registry


def active_registry(
    registry: dict[str, Any], cohort: str
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    comparisons = [dict(item) for item in registry["comparisons"] if item["cohort"] == cohort]
    referenced = {
        identifier
        for item in comparisons
        for identifier in (item["candidate"], item["baseline"])
    }
    profiles = {
        identifier: dict(profile)
        for identifier, profile in registry["profiles"].items()
        if identifier in referenced
    }
    if not profiles or not comparisons:
        raise ValueError(f"No Generation 17 FreqAI registry for {cohort}.")
    return profiles, comparisons


def smoke_registry(
    profiles: dict[str, dict[str, Any]], comparisons: Sequence[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    selected = [
        dict(item)
        for item in comparisons
        if item["route_id"] == "density_counts_increment"
    ]
    referenced = {
        identifier
        for item in selected
        for identifier in (item["candidate"], item["baseline"])
    }
    return ({key: value for key, value in profiles.items() if key in referenced}, selected)


def cache_inventory(cache: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "pair": item["pair"],
            **{
                f"{kind}_path": item[f"{kind}_path"]
                for kind in ("feature", "support", "event", "evaluation")
            },
            **{
                f"{kind}_sha256": item[f"{kind}_sha256"]
                for kind in ("feature", "support", "event", "evaluation")
            },
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
        maximum_target_horizon_hours=4,
    )
    config["market_reaction_zone_g17"] = config.pop("market_reaction_zone_g13")
    return config


def build_manifest(
    *,
    run_id: str,
    window: str,
    cohort: str,
    base_config: Path,
    python_exe: Path,
    profile_workers: int,
    technical_smoke: bool,
) -> tuple[dict[str, Any], Path]:
    frozen, cache, registry = load_sources(cohort)
    profiles, comparisons = active_registry(registry, cohort)
    if technical_smoke:
        profiles, comparisons = smoke_registry(profiles, comparisons)
    settings = g13.stage_settings(cohort=cohort, window=window, technical_smoke=technical_smoke)
    inventory = cache_inventory(cache)
    record_dir = RECORD_ROOT / run_id
    artifact_dir = ARTIFACT_ROOT / run_id
    manifest_path = record_dir / "g17_freqai_run_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        contracts = existing["source_contracts"]
        for name, path in (
            ("strategy", STRATEGY_FILE),
            ("freeze", g17z.FREEZE_PATH),
            ("cache_manifest", g17c.RECORD_ROOT / f"{cohort}_manifest.json"),
            ("profile_registry", g17c.REGISTRY_PATH),
        ):
            if g0.sha256_file(path) != contracts[name]["sha256"]:
                raise ValueError(f"Generation 17 {name} changed: {manifest_path}")
        return existing, manifest_path
    record_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    base = json.loads(base_config.read_text(encoding="utf-8"))
    feature_dir = Path(inventory[0]["feature_path"]).parent
    event_dir = Path(inventory[0]["event_path"]).parent
    evaluation_dir = Path(inventory[0]["evaluation_path"]).parent
    candidate_pairs = tuple(str(pair) for pair in cache["pairs"])
    pairs, coverage_audit = g14.coverage_supported_pairs(
        candidate_pairs=candidate_pairs,
        event_dir=event_dir,
        profiles=profiles,
        timerange=str(settings["timerange"]),
        train_days=int(settings["train_days"]),
    )
    if len(pairs) < 5:
        raise ValueError(f"Generation 17 {window}/{cohort} has only {len(pairs)} supported pairs.")
    commands: list[dict[str, Any]] = []
    for number, (profile_identifier, profile) in enumerate(profiles.items(), start=1):
        short_id = f"p{number:03d}_{g8.g6f.stable_digest(profile_identifier, 6)}"
        profile_dir = artifact_dir / "profiles" / short_id
        userdir = profile_dir / "user_data"
        export_dir = profile_dir / "export"
        identifier = f"g17-{g8.g6f.stable_digest(f'{run_id}|{profile_identifier}', 16)}"
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
                "command": command,
                "status": "pending",
                "attempts": 0,
            }
        )
    manifest = {
        "schema_version": 1,
        "generation": 17,
        "run_id": run_id,
        "created_at_utc": g0.utc_now(),
        "status": "prepared",
        "run_stage": "broad_density_reaction_and_external_attribution",
        "evaluation_window": window,
        "technical_smoke_not_evidence": technical_smoke,
        "cohort": cohort,
        "pairs": list(pairs),
        "candidate_pairs_before_common_support_gate": list(candidate_pairs),
        "coverage_excluded_pairs": sorted(set(candidate_pairs).difference(pairs)),
        "coverage_pair_audit": coverage_audit,
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
        "branches": sorted({item["branch_id"] for item in comparisons}),
        "targets": list(g17c.TARGETS),
        "seeds": [g17c.SEED],
        "maximum_target_horizon_hours": 4,
        "profile_registry_frozen_before_model_outcomes": True,
        "profile_registry": {"profiles": profiles, "comparisons": comparisons},
        "decision_rule": {
            "crossing": next(
                branch["pass_rule"]
                for branch in frozen["branches"]
                if branch["branch_id"] == "g17a_density_geometry_decomposition"
            ),
            "reaction": next(
                branch["pass_rule"]
                for branch in frozen["branches"]
                if branch["branch_id"] == "g17c_reaction_probability_calibration"
            ),
            "external": next(
                branch["pass_rule"]
                for branch in frozen["branches"]
                if branch["branch_id"] == "g17e_external_context_regimes"
            ),
            "complete_model_must_beat_all_immediate_components": True,
        },
        "research_boundary": dict(frozen["research_boundary"]),
        "recent_periods": list(g12z.RECENT_PERIODS) if window == g13z.RECENT else [],
        "source_contracts": {
            "base_config": artifact(base_config),
            "python_executable": str(python_exe.resolve()),
            "strategy": artifact(STRATEGY_FILE),
            "freeze": artifact(g17z.FREEZE_PATH),
            "cache_manifest": artifact(g17c.RECORD_ROOT / f"{cohort}_manifest.json"),
            "profile_registry": artifact(g17c.REGISTRY_PATH),
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
    return manifest, manifest_path


def assign_evaluation_periods(actual: DataFrame, manifest: dict[str, Any]) -> DataFrame:
    if manifest["evaluation_window"] != g13z.RECENT:
        return actual
    output = actual.copy()
    output["period"] = "outside_generation17_recent_windows"
    for definition in manifest["recent_periods"]:
        selected = output["date"].ge(pd.Timestamp(definition["start"])) & output[
            "date"
        ].lt(pd.Timestamp(definition["stop"]))
        output.loc[selected, "period"] = definition["period"]
    return output


def load_actual(manifest: dict[str, Any]) -> DataFrame:
    frames: list[DataFrame] = []
    evaluation_dir = Path(manifest["storage"]["evaluation_cache_dir"])
    for pair in manifest["pairs"]:
        frame = pd.read_parquet(
            evaluation_dir / f"{g0.pair_file_stem(pair)}.parquet",
            columns=[
                "date",
                "period",
                "market_group",
                "smart_contract_platform",
                "anchor_source_timeframe",
                *manifest["targets"],
            ],
        )
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
        frame["pair"] = pair
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    if output.duplicated(["pair", "date"]).any():
        raise ValueError("Duplicate pair/date rows in Generation 17 evaluation cache.")
    return assign_evaluation_periods(output, manifest)


def preflight_run(manifest: dict[str, Any], *, python_exe: Path) -> dict[str, Any]:
    audit = g11.preflight_run(manifest, python_exe=python_exe)
    problems = list(audit["problems"])
    if not STRATEGY_FILE.is_file():
        problems.append(f"Missing Generation 17 strategy: {STRATEGY_FILE}")
    required = {
        column for command in manifest["commands"] for column in command["feature_columns"]
    }
    for item in manifest["source_contracts"]["cache_inventory"]:
        try:
            pd.read_parquet(item["feature_path"], columns=sorted(required))
        except Exception as exc:
            problems.append(f"{item['pair']} lacks exact Generation 17 features: {exc}")
    actual = load_actual(manifest)
    evaluation_support: list[dict[str, Any]] = []
    for period in manifest["validation_periods"]:
        selected = actual.loc[actual["period"].eq(period)]
        counts = selected.groupby("pair", observed=True).size()
        row = {
            "period": period,
            "rows": len(selected),
            "coins": int((counts > 0).sum()),
            "minimum_pair_rows": int(counts.min()) if len(counts) else 0,
        }
        evaluation_support.append(row)
        if row["rows"] < 50 or row["coins"] < 5:
            problems.append(f"Insufficient Generation 17 evaluation support: {row}")
    return {
        **audit,
        "passed": not problems,
        "problems": problems,
        "exact_feature_columns": len(required),
        "evaluation_support": evaluation_support,
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
            raise ValueError(f"No Generation 17 predictions for {item['profile_id']}")
        predictions[item["profile_id"]] = frame
    actual = load_actual(manifest)
    pair, group, eligibility, reaction = g11.score_comparisons(
        manifest=manifest, predictions=predictions, actual=actual
    )
    decisions = g11.route_decisions(manifest, group)
    paths = {
        "prediction_audit": record_dir / "g17_prediction_audit.csv",
        "comparison_eligibility": record_dir / "g17_comparison_eligibility.csv",
        "pair_scores": record_dir / "g17_pair_scores.csv",
        "group_scores": record_dir / "g17_group_scores.csv",
        "reaction_diagnostics": record_dir / "g17_reaction_diagnostics.csv",
        "route_decisions": record_dir / "g17_route_decisions.csv",
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
        "generation": 17,
        "run_id": manifest["run_id"],
        "created_at_utc": g0.utc_now(),
        "status": (
            "completed_generation17_technical_smoke_not_evidence"
            if manifest["technical_smoke_not_evidence"]
            else "completed_generation17_freqai_cell"
        ),
        "technical_smoke_not_evidence": manifest["technical_smoke_not_evidence"],
        "run_stage": manifest["run_stage"],
        "evaluation_window": manifest["evaluation_window"],
        "cohort": manifest["cohort"],
        "profiles_completed": len(predictions),
        "branches_completed": len(manifest["branches"]),
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
        },
        "artifacts": {key: artifact(path) for key, path in paths.items()},
    }
    result_path = record_dir / "g17_freqai_result.json"
    g0.atomic_write_json(result, result_path)
    return {**result, "result_path": str(result_path.resolve())}


def run_cell(
    *,
    run_id: str,
    window: str,
    cohort: str,
    base_config: Path,
    python_exe: Path,
    profile_workers: int,
    technical_smoke: bool,
    prepare_only: bool,
) -> dict[str, Any]:
    manifest, manifest_path = build_manifest(
        run_id=run_id,
        window=window,
        cohort=cohort,
        base_config=base_config,
        python_exe=python_exe,
        profile_workers=profile_workers,
        technical_smoke=technical_smoke,
    )
    if str(manifest["status"]).startswith("completed_generation17_") and manifest.get(
        "result"
    ):
        result_path = manifest_path.parent / "g17_freqai_result.json"
        return {**manifest["result"], "result_path": str(result_path.resolve())}
    audit = preflight_run(manifest, python_exe=python_exe)
    g0.atomic_write_json(audit, manifest_path.parent / "g17_freqai_preflight.json")
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


def joint_review(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    expected = {
        (g13z.STANDARD, "normal"),
        (g13z.RECENT, "normal"),
        (g13z.STANDARD, "meme"),
    }
    observed = {(item["evaluation_window"], item["cohort"]) for item in results}
    if observed != expected:
        raise ValueError("Generation 17 FreqAI joint review requires all three cells.")
    decision_frames: list[DataFrame] = []
    reaction_frames: list[DataFrame] = []
    for result in results:
        decision = pd.read_csv(result["artifacts"]["route_decisions"]["path"])
        reaction = pd.read_csv(result["artifacts"]["reaction_diagnostics"]["path"])
        for frame in (decision, reaction):
            frame["cohort"] = result["cohort"]
            frame["evaluation_window"] = result["evaluation_window"]
        decision_frames.append(decision)
        reaction_frames.append(reaction)
    combined = pd.concat(decision_frames, ignore_index=True)
    reaction_diagnostics = pd.concat(reaction_frames, ignore_index=True)
    reaction_diagnostics["all_probability_metrics_better"] = (
        reaction_diagnostics["candidate_rank_auc"]
        > reaction_diagnostics["baseline_rank_auc"]
    ) & (
        reaction_diagnostics["candidate_balanced_accuracy_at_half"]
        > reaction_diagnostics["baseline_balanced_accuracy_at_half"]
    ) & (
        reaction_diagnostics["candidate_brier_error"]
        < reaction_diagnostics["baseline_brier_error"]
    )
    rows: list[dict[str, Any]] = []
    for (route_id, target), cell in combined.groupby(
        ["route_id", "target"], observed=True, sort=False
    ):
        seen = set(zip(cell["evaluation_window"], cell["cohort"], strict=False))
        complete = seen == expected
        point = bool(complete and cell["all_controls_point_positive"].astype(bool).all())
        strict = bool(complete and cell["all_controls_strict"].astype(bool).all())
        probability_rows = reaction_diagnostics.loc[
            reaction_diagnostics["route_id"].eq(route_id)
            & reaction_diagnostics["target"].eq(target)
        ]
        probability_complete = bool(
            "reaction_h" not in target
            or (
                not probability_rows.empty
                and set(
                    zip(
                        probability_rows["evaluation_window"],
                        probability_rows["cohort"],
                        strict=False,
                    )
                )
                == expected
            )
        )
        probability_metrics_better = bool(
            "reaction_h" not in target
            or (
                probability_complete
                and probability_rows["all_probability_metrics_better"].astype(bool).all()
            )
        )
        calibrated_strict = strict and probability_metrics_better
        normal = cell.loc[cell["cohort"].eq("normal")]
        normal_repeat = bool(
            set(normal["evaluation_window"]) == {g13z.STANDARD, g13z.RECENT}
            and normal["all_controls_point_positive"].astype(bool).all()
        )
        standard = cell.loc[cell["evaluation_window"].eq(g13z.STANDARD)]
        both_cohorts = bool(
            set(standard["cohort"]) == {"normal", "meme"}
            and standard["all_controls_point_positive"].astype(bool).all()
        )
        if calibrated_strict:
            status = "strict_all_three_cells"
        elif strict and not probability_metrics_better:
            status = "mae_strict_but_probability_calibration_failed"
        elif point:
            status = "point_all_three_cells"
        elif normal_repeat:
            status = "normal_standard_and_recent_only"
        elif both_cohorts:
            status = "standard_normal_and_meme_only"
        else:
            status = "not_retained_across_generation17_cells"
        rows.append(
            {
                "route_id": route_id,
                "target": target,
                "status": status,
                "complete_three_cell_ladder": complete,
                "strict_all_three_cells": calibrated_strict,
                "mae_strict_all_three_cells": strict,
                "probability_metrics_better_all_rows": probability_metrics_better,
                "point_all_three_cells": point,
                "normal_standard_and_recent_point": normal_repeat,
                "standard_both_cohorts_point": both_cohorts,
                "minimum_equal_coin_paired_mae_gain": float(
                    cell["minimum_equal_coin_paired_mae_gain"].min()
                ),
                "minimum_bootstrap_lower": float(cell["minimum_bootstrap_lower"].min()),
            }
        )
    decisions = DataFrame.from_records(rows).sort_values(["route_id", "target"])
    record_dir = RECORD_ROOT / DEFAULT_JOINT_REVIEW_ID
    record_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = record_dir / "g17_freqai_joint_decisions.csv"
    probability_path = record_dir / "g17_reaction_probability_diagnostics.csv"
    g0.atomic_write_csv(decisions, decisions_path)
    g0.atomic_write_csv(reaction_diagnostics, probability_path)
    output = {
        "schema_version": 1,
        "generation": 17,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation17_freqai_joint_review",
        "branches_completed": [
            "g17a_density_geometry_decomposition",
            "g17c_reaction_probability_calibration",
            "g17e_external_context_regimes_orderbook",
        ],
        "external_routes_parked_for_coverage": [
            "historical_news_context",
            "non_crypto_global_market_context",
        ],
        "all_three_cells_completed_before_review": True,
        "summary": {
            "questions_targets": len(decisions),
            "strict_all_three": int(decisions["strict_all_three_cells"].sum()),
            "mae_strict_all_three": int(decisions["mae_strict_all_three_cells"].sum()),
            "point_all_three": int(decisions["point_all_three_cells"].sum()),
            "normal_standard_and_recent": int(
                decisions["normal_standard_and_recent_point"].sum()
            ),
            "standard_both_cohorts": int(decisions["standard_both_cohorts_point"].sum()),
        },
        "research_boundary": {
            "direction_tested": False,
            "profit_used": False,
            "trading_promotion": False,
        },
        "source_results": [artifact(Path(item["result_path"])) for item in results],
        "artifacts": {
            "joint_decisions": artifact(decisions_path),
            "reaction_probability_diagnostics": artifact(probability_path),
        },
    }
    result_path = record_dir / "g17_freqai_joint_review.json"
    g0.atomic_write_json(output, result_path)
    return {**output, "result_path": str(result_path.resolve())}


def requested_runs(window: str, cohort: str) -> list[tuple[str, str]]:
    candidates = (
        (g13z.STANDARD, "normal"),
        (g13z.RECENT, "normal"),
        (g13z.STANDARD, "meme"),
    )
    runs = [
        (selected_window, selected_cohort)
        for selected_window, selected_cohort in candidates
        if window in ("all", selected_window) and cohort in ("all", selected_cohort)
    ]
    if not runs:
        raise ValueError("No Generation 17 FreqAI cell matches the selection.")
    return runs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", choices=("all", g13z.STANDARD, g13z.RECENT), default="all")
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
    runs = requested_runs(args.window, args.cohort)
    stem = args.run_stem or (DEFAULT_SMOKE_STEM if args.technical_smoke else DEFAULT_RUN_STEM)
    results: list[dict[str, Any]] = []
    for window, cohort in runs:
        short_window = "standard" if window == g13z.STANDARD else "recent"
        result = run_cell(
            run_id=f"{stem}_{short_window}_{cohort}",
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
    if len(runs) == 3 and not args.technical_smoke and not args.prepare_only:
        results.append(joint_review(results))
    print(json.dumps(results, indent=2, default=g0.json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the frozen Generation 12 chronology, subfamily, and combination siblings."""

from __future__ import annotations

# Bound numerical pools before pandas and FreqAI imports.
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
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation11_freqai_cache as g11c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation12_freeze as g12z,
)


DEFAULT_CONFIG = g11.DEFAULT_CONFIG
DEFAULT_PYTHON = g11.DEFAULT_PYTHON
STRATEGY_PATH = g11.STRATEGY_PATH
STRATEGY_FILE = STRATEGY_PATH / "MarketReactionZoneFreqAIGeneration12Strategy.py"
STRATEGY_CLASS = "MarketReactionZoneG12ConfigurableFreqAIResearchStrategy"
DATA_DIR = g11.DATA_DIR
RECORD_ROOT = (
    g11.RECORD_ROOT.parent / "g12_chronology_attribution_and_combinations"
)
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData")
    / "research_outputs"
    / "market_reaction_zones"
    / "generation12_branches"
    / "g12_chronology_attribution_and_combinations"
)
DEFAULT_RUN_STEM = "g12_balanced_followup_20260822a"
DEFAULT_SMOKE_STEM = "g12_technical_smoke_20260822a"
MAX_WORKERS = 4


def artifact(path: Path) -> dict[str, Any]:
    return g11.artifact(path)


def load_sources(cohort: str) -> tuple[dict[str, Any], dict[str, Any], Path]:
    frozen = g12z.validate_existing_freeze()
    cache_path = g11c.RECORD_ROOT / f"{cohort}_manifest.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache.get("status") != "completed_generation11_freqai_cache":
        raise ValueError(f"Generation 11 {cohort} cache is not terminal.")
    source = frozen["source_contracts"][f"generation11_{cohort}_cache"]
    if g0.sha256_file(cache_path) != source["sha256"]:
        raise ValueError(f"Generation 11 {cohort} cache changed after G12 freeze.")
    return frozen, cache, cache_path


def active_registry(
    frozen: dict[str, Any], *, stage: str, cohort: str
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    comparisons = [
        dict(item)
        for item in frozen["comparisons"]
        if item["stage"] == stage and item["cohort"] == cohort
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
    if not comparisons or not profiles:
        raise ValueError(f"No Generation 12 registry for {stage}/{cohort}.")
    return profiles, comparisons


def smoke_registry(
    *,
    stage: str,
    profiles: dict[str, dict[str, Any]],
    comparisons: Sequence[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    selected_routes = (
        {"local_activity_and_volatility"}
        if stage == g12z.STAGE_RECENT
        else {"activity_participation", "activity_plus_trend"}
    )
    selected = [
        dict(item) for item in comparisons if item["route_id"] in selected_routes
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


def stage_settings(
    frozen: dict[str, Any], *, stage: str, cohort: str, technical_smoke: bool
) -> dict[str, Any]:
    if stage == g12z.STAGE_RECENT:
        if cohort != "normal":
            raise ValueError("The Generation 12 recent stage is normal-cohort only.")
        settings = {
            **frozen["recent_settings"],
            "validation_periods": [
                item["period"] for item in frozen["recent_periods"]
            ],
        }
    else:
        settings = dict(frozen["attribution_settings"][cohort])
    if technical_smoke:
        if stage == g12z.STAGE_RECENT:
            settings.update(
                {
                    "timerange": "20260701-20260821",
                    "train_days": 365,
                    "backtest_days": 30,
                }
            )
        elif cohort == "normal":
            settings.update(
                {
                    "timerange": "20250401-20250701",
                    "train_days": 365,
                    "backtest_days": 90,
                }
            )
        else:
            settings.update(
                {
                    "timerange": "20260401-20260516",
                    "train_days": 160,
                    "backtest_days": 45,
                }
            )
    return settings


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
    config = g11.profile_config(
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
    research = config.pop("market_reaction_zone_g11")
    research["feature_columns"] = list(profile["feature_columns"])
    config["market_reaction_zone_g12"] = research
    return config


def build_manifest(
    *,
    run_id: str,
    stage: str,
    cohort: str,
    base_config: Path,
    python_exe: Path,
    profile_workers: int,
    technical_smoke: bool,
) -> tuple[dict[str, Any], Path, Path]:
    frozen, cache, cache_path = load_sources(cohort)
    profiles, comparisons = active_registry(
        frozen, stage=stage, cohort=cohort
    )
    if technical_smoke:
        profiles, comparisons = smoke_registry(
            stage=stage,
            profiles=profiles,
            comparisons=comparisons,
        )
    settings = stage_settings(
        frozen,
        stage=stage,
        cohort=cohort,
        technical_smoke=technical_smoke,
    )
    record_dir = RECORD_ROOT / run_id
    artifact_dir = ARTIFACT_ROOT / run_id
    manifest_path = record_dir / "g12_freqai_run_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        contracts = existing["source_contracts"]
        if g0.sha256_file(STRATEGY_FILE) != contracts["strategy"]["sha256"]:
            raise ValueError(f"G12 strategy changed after preparation: {manifest_path}")
        if g0.sha256_file(cache_path) != contracts["cache_manifest"]["sha256"]:
            raise ValueError(f"G12 cache changed after preparation: {manifest_path}")
        if g0.sha256_file(g12z.FREEZE_PATH) != contracts["freeze"]["sha256"]:
            raise ValueError(f"G12 freeze changed after preparation: {manifest_path}")
        return existing, manifest_path, artifact_dir
    record_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    base = json.loads(base_config.read_text(encoding="utf-8"))
    feature_dir = Path(cache["inventory"][0]["feature_path"]).parent
    event_dir = Path(cache["inventory"][0]["event_path"]).parent
    pairs = tuple(str(pair) for pair in cache["pairs"])
    commands: list[dict[str, Any]] = []
    for number, (profile_id, profile) in enumerate(profiles.items(), start=1):
        short_id = f"p{number:03d}_{g8.g6f.stable_digest(profile_id, 6)}"
        profile_dir = artifact_dir / "profiles" / short_id
        userdir = profile_dir / "user_data"
        export_dir = profile_dir / "export"
        identifier = f"g12-{g8.g6f.stable_digest(f'{run_id}|{profile_id}', 16)}"
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
        "generation": 12,
        "run_id": run_id,
        "created_at_utc": g0.utc_now(),
        "status": "prepared",
        "run_stage": stage,
        "technical_smoke_not_evidence": technical_smoke,
        "evidence_label": frozen["evidence_boundaries"][stage],
        "cohort": cohort,
        "pairs": list(pairs),
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
        "targets": list(g12z.TARGETS),
        "seeds": [g12z.SEED],
        "profile_registry_frozen_before_model_outcomes": True,
        "profile_registry": {
            "profiles": profiles,
            "comparisons": comparisons,
        },
        "decision_rule": frozen["decision_rule"],
        "research_boundary": dict(frozen["research_boundary"]),
        "recent_periods": (
            list(frozen["recent_periods"])
            if stage == g12z.STAGE_RECENT
            else []
        ),
        "source_contracts": {
            "base_config": artifact(base_config),
            "python_executable": str(python_exe.resolve()),
            "strategy": artifact(STRATEGY_FILE),
            "freeze": artifact(g12z.FREEZE_PATH),
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


def assign_evaluation_periods(actual: DataFrame, manifest: dict[str, Any]) -> DataFrame:
    if manifest["run_stage"] != g12z.STAGE_RECENT:
        return actual
    output = actual.copy()
    output["period"] = "outside_generation12_recent_windows"
    for definition in manifest["recent_periods"]:
        start = pd.Timestamp(definition["start"])
        stop = pd.Timestamp(definition["stop"])
        selected = output["date"].ge(start) & output["date"].lt(stop)
        output.loc[selected, "period"] = definition["period"]
    return output


def preflight_run(manifest: dict[str, Any], *, python_exe: Path) -> dict[str, Any]:
    audit = g11.preflight_run(manifest, python_exe=python_exe)
    problems = list(audit["problems"])
    if not STRATEGY_FILE.is_file():
        problems.append(f"Missing Generation 12 strategy: {STRATEGY_FILE}")
    required_columns = {
        column
        for item in manifest["commands"]
        for column in item["feature_columns"]
    }
    for item in manifest["source_contracts"]["cache_inventory"]:
        frame = pd.read_parquet(
            item["feature_path"], columns=sorted(required_columns)
        )
        missing = required_columns.difference(frame.columns)
        if missing:
            problems.append(
                f"{item['pair']} lacks exact G12 columns {sorted(missing)}."
            )
    period_support: list[dict[str, Any]] = []
    actual = assign_evaluation_periods(g11.load_actual(manifest), manifest)
    for period in manifest["validation_periods"]:
        selected = actual.loc[actual["period"].eq(period)]
        counts = selected.groupby("pair", observed=True).size()
        row = {
            "period": period,
            "rows": len(selected),
            "coins": int((counts > 0).sum()),
            "minimum_pair_rows": int(counts.min()) if len(counts) else 0,
        }
        period_support.append(row)
        if row["rows"] < 50 or row["coins"] < 5:
            problems.append(f"Insufficient Generation 12 period support: {row}")
    return {
        **audit,
        "passed": not problems,
        "problems": problems,
        "exact_feature_columns": len(required_columns),
        "evaluation_period_support": period_support,
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
            raise ValueError(f"No Generation 12 predictions for {item['profile_id']}")
        predictions[str(item["profile_id"])] = frame
    actual = assign_evaluation_periods(g11.load_actual(manifest), manifest)
    pair_scores, group_scores, eligibility, reaction = g11.score_comparisons(
        manifest=manifest,
        predictions=predictions,
        actual=actual,
    )
    decisions = g11.route_decisions(manifest, group_scores)
    paths = {
        "prediction_audit": record_dir / "g12_prediction_audit.csv",
        "comparison_eligibility": record_dir / "g12_comparison_eligibility.csv",
        "pair_scores": record_dir / "g12_pair_scores.csv",
        "group_scores": record_dir / "g12_group_scores.csv",
        "reaction_diagnostics": record_dir / "g12_reaction_diagnostics.csv",
        "route_decisions": record_dir / "g12_route_decisions.csv",
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
    result = {
        "schema_version": 1,
        "generation": 12,
        "run_id": manifest["run_id"],
        "created_at_utc": g0.utc_now(),
        "status": (
            "completed_generation12_technical_smoke_not_evidence"
            if manifest["technical_smoke_not_evidence"]
            else f"completed_generation12_{manifest['run_stage']}"
        ),
        "technical_smoke_not_evidence": manifest["technical_smoke_not_evidence"],
        "evidence_label": manifest["evidence_label"],
        "run_stage": manifest["run_stage"],
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
        },
        "artifacts": {key: artifact(path) for key, path in paths.items()},
        "interpretation_boundary": manifest["evidence_label"],
    }
    result_path = record_dir / "g12_freqai_result.json"
    g0.atomic_write_json(result, result_path)
    return {**result, "result_path": str(result_path.resolve())}


def run_stage(
    *,
    run_id: str,
    stage: str,
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
        cohort=cohort,
        base_config=base_config,
        python_exe=python_exe,
        profile_workers=profile_workers,
        technical_smoke=technical_smoke,
    )
    audit = preflight_run(manifest, python_exe=python_exe)
    g0.atomic_write_json(audit, manifest_path.parent / "g12_freqai_preflight.json")
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


def joint_attribution_decisions(results: Sequence[dict[str, Any]]) -> DataFrame:
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
            status = "strict_in_both_cohorts_pending_later_confirmation"
        elif normal_point and meme_point:
            status = "point_positive_in_both_cohorts_pending_later_confirmation"
        elif normal_point:
            status = "normal_only_exploratory_lead"
        elif meme_point:
            status = "meme_only_exploratory_lead"
        else:
            status = "not_retained_after_generation12_attribution"
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
                "retained": normal_point or meme_point,
            }
        )
    return DataFrame.from_records(rows).sort_values(["route_id", "target"])


def joint_review(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    recent = [
        item for item in results if item["run_stage"] == g12z.STAGE_RECENT
    ]
    attribution = [
        item for item in results if item["run_stage"] == g12z.STAGE_ATTRIBUTION
    ]
    if len(recent) != 1 or {item["cohort"] for item in attribution} != {
        "normal",
        "meme",
    }:
        raise ValueError("Generation 12 joint review requires all three sibling runs.")
    recent_path = Path(recent[0]["artifacts"]["route_decisions"]["path"])
    recent_decisions = pd.read_csv(recent_path)
    attribution_joint = joint_attribution_decisions(attribution)
    record_dir = RECORD_ROOT / "g12_joint_review_20260822a"
    record_dir.mkdir(parents=True, exist_ok=True)
    recent_output = record_dir / "g12_recent_route_decisions.csv"
    attribution_output = record_dir / "g12_joint_attribution_decisions.csv"
    g0.atomic_write_csv(recent_decisions, recent_output)
    g0.atomic_write_csv(attribution_joint, attribution_output)
    recent_point = recent_decisions["all_controls_point_positive"].astype(bool)
    recent_strict = recent_decisions["all_controls_strict"].astype(bool)
    retained = attribution_joint["retained"].astype(bool)
    output = {
        "schema_version": 1,
        "generation": 12,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation12_joint_review",
        "all_frozen_siblings_completed_before_review": True,
        "recent_target_routes_reviewed": len(recent_decisions),
        "recent_strict_leads": int(recent_strict.sum()),
        "recent_point_leads": int((recent_point & ~recent_strict).sum()),
        "recent_not_retained": int((~recent_point).sum()),
        "attribution_target_routes_reviewed": len(attribution_joint),
        "attribution_retained": int(retained.sum()),
        "attribution_strict_both": int(
            attribution_joint["status"].eq(
                "strict_in_both_cohorts_pending_later_confirmation"
            ).sum()
        ),
        "attribution_point_both": int(
            attribution_joint["status"].isin(
                {
                    "strict_in_both_cohorts_pending_later_confirmation",
                    "point_positive_in_both_cohorts_pending_later_confirmation",
                }
            ).sum()
        ),
        "automatic_descendant_launch": False,
        "parallel_queue_preserved": g12z.validate_existing_freeze()[
            "parallel_queue_not_executed_in_generation12"
        ],
        "research_boundary": {
            "direction_tested": False,
            "joint_55_percent_direction_target_reached": False,
            "profit_used": False,
            "trading_promotion": False,
        },
        "source_results": [
            artifact(
                Path(
                    item.get("result_path")
                    or RECORD_ROOT / item["run_id"] / "g12_freqai_result.json"
                )
            )
            for item in results
        ],
        "artifacts": {
            "recent_route_decisions": artifact(recent_output),
            "joint_attribution_decisions": artifact(attribution_output),
        },
    }
    result_path = record_dir / "g12_joint_review.json"
    g0.atomic_write_json(output, result_path)
    return {**output, "result_path": str(result_path.resolve())}


def requested_runs(stage: str, cohort: str) -> list[tuple[str, str]]:
    runs: list[tuple[str, str]] = []
    if stage in ("all", g12z.STAGE_RECENT):
        if cohort in ("all", "normal"):
            runs.append((g12z.STAGE_RECENT, "normal"))
    if stage in ("all", g12z.STAGE_ATTRIBUTION):
        for item in (("normal",), ("meme",)):
            selected = item[0]
            if cohort in ("all", selected):
                runs.append((g12z.STAGE_ATTRIBUTION, selected))
    if not runs:
        raise ValueError("No Generation 12 runs match the stage/cohort selection.")
    return runs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run frozen Generation 12 siblings.")
    parser.add_argument(
        "--stage", choices=("all", *g12z.STAGES), default="all"
    )
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
    runs = requested_runs(args.stage, args.cohort)
    stem = args.run_stem or (
        DEFAULT_SMOKE_STEM if args.technical_smoke else DEFAULT_RUN_STEM
    )
    results = []
    for stage, cohort in runs:
        short_stage = "recent" if stage == g12z.STAGE_RECENT else "attribution"
        result = run_stage(
            run_id=f"{stem}_{short_stage}_{cohort}",
            stage=stage,
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
        runs
        == [
            (g12z.STAGE_RECENT, "normal"),
            (g12z.STAGE_ATTRIBUTION, "normal"),
            (g12z.STAGE_ATTRIBUTION, "meme"),
        ]
        and not args.technical_smoke
        and not args.prepare_only
    ):
        results.append(joint_review(results))
    print(json.dumps(results, indent=2, default=g0.json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

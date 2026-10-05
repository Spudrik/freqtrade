"""Run the frozen Generation 13 higher-timeframe and long-horizon FreqAI siblings."""

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
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation11_freqai_freeze as g11f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation12_freeze as g12z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation13_cache as g13c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation13_direct_controls as g13d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation13_freeze as g13z,
)


DEFAULT_CONFIG = g11.DEFAULT_CONFIG
DEFAULT_PYTHON = g11.DEFAULT_PYTHON
STRATEGY_PATH = g11.STRATEGY_PATH
STRATEGY_FILE = STRATEGY_PATH / "MarketReactionZoneFreqAIGeneration13Strategy.py"
STRATEGY_CLASS = "MarketReactionZoneG13ConfigurableFreqAIResearchStrategy"
DATA_DIR = g11.DATA_DIR
RECORD_ROOT = g13z.FREEZE_PATH.parent / "g13_broad_siblings" / "freqai"
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation13_branches"
    / "g13_broad_siblings"
    / "freqai"
)
DIRECT_RESULT = (
    g13d.RECORD_ROOT / g13d.DEFAULT_RUN_ID / "g13_direct_control_result.json"
)
DEFAULT_RUN_STEM = "g13_broad_freqai_20260822a"
DEFAULT_SMOKE_STEM = "g13_technical_smoke_20260822a"
MAX_WORKERS = 4
LONG_SOURCE_TIMEFRAMES = ("4h", "8h", "1d")


def artifact(path: Path) -> dict[str, Any]:
    return g11.artifact(path)


def load_sources(cohort: str) -> tuple[dict[str, Any], dict[str, Any], Path]:
    frozen = json.loads(g13z.FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation13_outcomes":
        raise ValueError("Generation 13 freeze is not valid.")
    cache_path = g13c.RECORD_ROOT / f"{cohort}_manifest.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache.get("status") != "completed_generation13_cache":
        raise ValueError(f"Generation 13 {cohort} cache is not terminal.")
    if cache.get("cohort") != cohort:
        raise ValueError(f"Generation 13 cache cohort mismatch: {cache_path}")
    source = cache["source_contracts"]["generation13_freeze"]
    if g0.sha256_file(g13z.FREEZE_PATH) != source["sha256"]:
        raise ValueError("Generation 13 freeze changed after cache materialization.")
    return frozen, cache, cache_path


def validate_direct_sibling() -> dict[str, Any]:
    if not DIRECT_RESULT.is_file():
        raise FileNotFoundError(
            "Generation 13 direct-control sibling must finish before FreqAI runs."
        )
    result = json.loads(DIRECT_RESULT.read_text(encoding="utf-8"))
    if result.get("status") != "completed_generation13_direct_controls":
        raise ValueError("Generation 13 direct-control sibling is not terminal.")
    return result


def active_registry(
    frozen: dict[str, Any],
    cache: dict[str, Any],
    *,
    stage: str,
    cohort: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    supported = set(cache["supported_profiles"])
    comparisons = [
        dict(item)
        for item in frozen["comparisons"]
        if item["stage"] == stage
        and item["cohort"] == cohort
        and item["candidate"] in supported
        and item["baseline"] in supported
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
    if not profiles or not comparisons:
        raise ValueError(f"No active Generation 13 profiles for {stage}/{cohort}.")
    return profiles, comparisons


def smoke_registry(
    *,
    stage: str,
    profiles: dict[str, dict[str, Any]],
    comparisons: Sequence[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    selected_route = (
        "4h_activity_volatility"
        if stage == g13z.MTF_STAGE
        else "local_activity_volatility"
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


def stage_inventory(
    cache: dict[str, Any], *, stage: str
) -> list[dict[str, Any]]:
    if stage == g13z.LONG_STAGE:
        prefix = "long"
        return [
            {
                "pair": item["pair"],
                **{
                    f"{kind}_path": item[f"{prefix}_{kind}_path"]
                    for kind in ("feature", "support", "event", "evaluation")
                },
                **{
                    f"{kind}_sha256": item[f"{prefix}_{kind}_sha256"]
                    for kind in ("feature", "support", "event", "evaluation")
                },
            }
            for item in cache["inventory"]
        ]
    return [
        {
            "pair": item["pair"],
            **{
                f"{kind}_path": item[f"mtf_{kind}_path"]
                for kind in ("feature", "support", "event", "evaluation")
            },
            **{
                f"{kind}_sha256": item[f"mtf_{kind}_sha256"]
                for kind in ("feature", "support", "event", "evaluation")
            },
        }
        for item in cache["inventory"]
    ]


def stage_settings(
    *, cohort: str, window: str, technical_smoke: bool
) -> dict[str, Any]:
    base = g11f.validate_existing_freeze()["cohort_settings"][cohort]
    settings = dict(base)
    if window == g13z.RECENT:
        if cohort != "normal":
            raise ValueError("Recent Generation 13 chronology is normal-cohort only.")
        settings.update(
            {
                "timerange": "20260401-20260821",
                "train_days": 365,
                "backtest_days": 30,
                "validation_periods": [
                    item["period"] for item in g12z.RECENT_PERIODS
                ],
            }
        )
    if technical_smoke:
        if cohort == "normal":
            settings.update(
                {
                    "timerange": "20260701-20260821",
                    "train_days": 365,
                    "backtest_days": 30,
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
    maximum_target_horizon_hours: int,
) -> dict[str, Any]:
    adapted_profile = {
        **profile,
        "blocks": list(profile["required_ready_blocks"]),
    }
    config = g11.profile_config(
        base,
        identifier=identifier,
        pairs=pairs,
        feature_dir=feature_dir,
        event_dir=event_dir,
        profile=adapted_profile,
        train_days=train_days,
        backtest_days=backtest_days,
        technical_smoke=technical_smoke,
    )
    research = config.pop("market_reaction_zone_g11")
    research.update(
        {
            "feature_columns": list(profile["feature_columns"]),
            "maximum_target_horizon_hours": maximum_target_horizon_hours,
            "train_prediction_embargo_hours": maximum_target_horizon_hours,
        }
    )
    config["market_reaction_zone_g13"] = research
    config["freqai"]["feature_parameters"][
        "label_period_candles"
    ] = maximum_target_horizon_hours
    return config


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
    direct = validate_direct_sibling()
    frozen, cache, cache_path = load_sources(cohort)
    profiles, comparisons = active_registry(
        frozen, cache, stage=stage, cohort=cohort
    )
    if technical_smoke:
        profiles, comparisons = smoke_registry(
            stage=stage,
            profiles=profiles,
            comparisons=comparisons,
        )
    settings = stage_settings(
        cohort=cohort,
        window=window,
        technical_smoke=technical_smoke,
    )
    inventory = stage_inventory(cache, stage=stage)
    record_dir = RECORD_ROOT / run_id
    artifact_dir = ARTIFACT_ROOT / run_id
    manifest_path = record_dir / "g13_freqai_run_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        contracts = existing["source_contracts"]
        if g0.sha256_file(STRATEGY_FILE) != contracts["strategy"]["sha256"]:
            raise ValueError(f"G13 strategy changed after preparation: {manifest_path}")
        if g0.sha256_file(cache_path) != contracts["cache_manifest"]["sha256"]:
            raise ValueError(f"G13 cache changed after preparation: {manifest_path}")
        if g0.sha256_file(g13z.FREEZE_PATH) != contracts["freeze"]["sha256"]:
            raise ValueError(f"G13 freeze changed after preparation: {manifest_path}")
        return existing, manifest_path, artifact_dir
    record_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    base = json.loads(base_config.read_text(encoding="utf-8"))
    feature_dir = Path(inventory[0]["feature_path"]).parent
    event_dir = Path(inventory[0]["event_path"]).parent
    evaluation_dir = Path(inventory[0]["evaluation_path"]).parent
    pairs = tuple(str(pair) for pair in cache["pairs"])
    maximum_horizon = 8 if stage == g13z.MTF_STAGE else 48
    commands: list[dict[str, Any]] = []
    for number, (profile_id, profile) in enumerate(profiles.items(), start=1):
        short_id = f"p{number:03d}_{g8.g6f.stable_digest(profile_id, 6)}"
        profile_dir = artifact_dir / "profiles" / short_id
        userdir = profile_dir / "user_data"
        export_dir = profile_dir / "export"
        identifier = f"g13-{g8.g6f.stable_digest(f'{run_id}|{profile_id}', 16)}"
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
        "generation": 13,
        "run_id": run_id,
        "created_at_utc": g0.utc_now(),
        "status": "prepared",
        "run_stage": stage,
        "evaluation_window": window,
        "technical_smoke_not_evidence": technical_smoke,
        "evidence_label": (
            "Later normal chronology confirmation"
            if window == g13z.RECENT
            else "Frozen standard validation"
        ),
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
        "targets": list(next(iter(profiles.values()))["targets"]),
        "seeds": [g13z.SEED],
        "maximum_target_horizon_hours": maximum_horizon,
        "profile_registry_frozen_before_model_outcomes": True,
        "profile_registry": {"profiles": profiles, "comparisons": comparisons},
        "decision_rule": frozen["decision_rule"],
        "research_boundary": dict(frozen["research_boundary"]),
        "recent_periods": list(g12z.RECENT_PERIODS) if window == g13z.RECENT else [],
        "source_contracts": {
            "base_config": artifact(base_config),
            "python_executable": str(python_exe.resolve()),
            "strategy": artifact(STRATEGY_FILE),
            "freeze": artifact(g13z.FREEZE_PATH),
            "cache_manifest": artifact(cache_path),
            "cache_inventory": inventory,
            "direct_control_sibling": artifact(DIRECT_RESULT),
            "direct_control_status": direct["status"],
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
    if manifest["evaluation_window"] != g13z.RECENT:
        return actual
    output = actual.copy()
    output["period"] = "outside_generation13_recent_windows"
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
        ["anchor_source_timeframe"]
        if manifest["run_stage"] == g13z.LONG_STAGE
        else []
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
        raise ValueError("Duplicate pair/date rows in Generation 13 evaluation cache.")
    return assign_evaluation_periods(output, manifest)


def preflight_run(manifest: dict[str, Any], *, python_exe: Path) -> dict[str, Any]:
    audit = g11.preflight_run(manifest, python_exe=python_exe)
    problems = list(audit["problems"])
    if not STRATEGY_FILE.is_file():
        problems.append(f"Missing Generation 13 strategy: {STRATEGY_FILE}")
    required_columns = {
        column for item in manifest["commands"] for column in item["feature_columns"]
    }
    for item in manifest["source_contracts"]["cache_inventory"]:
        try:
            pd.read_parquet(item["feature_path"], columns=sorted(required_columns))
        except Exception as exc:  # exact cache failure needs pair/path context
            problems.append(f"{item['pair']} lacks exact G13 feature columns: {exc}")
    support_rows: list[dict[str, Any]] = []
    actual = load_actual(manifest)
    source_scopes: Sequence[str | None] = (
        LONG_SOURCE_TIMEFRAMES
        if manifest["run_stage"] == g13z.LONG_STAGE
        else (None,)
    )
    for period in manifest["validation_periods"]:
        for source_timeframe in source_scopes:
            selected = actual.loc[actual["period"].eq(period)]
            if source_timeframe is not None:
                selected = selected.loc[
                    selected["anchor_source_timeframe"].eq(source_timeframe)
                ]
            counts = selected.groupby("pair", observed=True).size()
            row = {
                "period": period,
                "anchor_source_timeframe": source_timeframe or "all",
                "rows": len(selected),
                "coins": int((counts > 0).sum()),
                "minimum_pair_rows": int(counts.min()) if len(counts) else 0,
            }
            support_rows.append(row)
            minimum_rows = 20 if source_timeframe is not None else 50
            if row["rows"] < minimum_rows or row["coins"] < 5:
                problems.append(f"Insufficient Generation 13 evaluation support: {row}")
    return {
        **audit,
        "passed": not problems,
        "problems": problems,
        "exact_feature_columns": len(required_columns),
        "evaluation_support": support_rows,
    }


def score_comparisons(
    *,
    manifest: dict[str, Any],
    predictions: dict[str, DataFrame],
    actual: DataFrame,
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
    scopes: Sequence[str | None] = (
        LONG_SOURCE_TIMEFRAMES
        if manifest["run_stage"] == g13z.LONG_STAGE
        else (None,)
    )
    outputs: list[tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]] = []
    for source_timeframe in scopes:
        selected = actual
        if source_timeframe is not None:
            selected = actual.loc[
                actual["anchor_source_timeframe"].eq(source_timeframe)
            ].copy()
        pair, group, eligibility, reaction = g11.score_comparisons(
            manifest=manifest,
            predictions=predictions,
            actual=selected,
        )
        decisions = g11.route_decisions(manifest, group)
        for frame in (pair, group, eligibility, reaction, decisions):
            frame["anchor_source_timeframe"] = source_timeframe or "all"
        outputs.append((pair, group, eligibility, reaction, decisions))
    return tuple(
        pd.concat([item[index] for item in outputs], ignore_index=True, sort=False)
        for index in range(5)
    )  # type: ignore[return-value]


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
            raise ValueError(f"No Generation 13 predictions for {item['profile_id']}")
        predictions[str(item["profile_id"])] = frame
    actual = load_actual(manifest)
    pair, group, eligibility, reaction, decisions = score_comparisons(
        manifest=manifest,
        predictions=predictions,
        actual=actual,
    )
    paths = {
        "prediction_audit": record_dir / "g13_prediction_audit.csv",
        "comparison_eligibility": record_dir / "g13_comparison_eligibility.csv",
        "pair_scores": record_dir / "g13_pair_scores.csv",
        "group_scores": record_dir / "g13_group_scores.csv",
        "reaction_diagnostics": record_dir / "g13_reaction_diagnostics.csv",
        "route_decisions": record_dir / "g13_route_decisions.csv",
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
        "generation": 13,
        "run_id": manifest["run_id"],
        "created_at_utc": g0.utc_now(),
        "status": (
            "completed_generation13_technical_smoke_not_evidence"
            if manifest["technical_smoke_not_evidence"]
            else "completed_generation13_freqai_cell"
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
    result_path = record_dir / "g13_freqai_result.json"
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
    audit = preflight_run(manifest, python_exe=python_exe)
    g0.atomic_write_json(audit, manifest_path.parent / "g13_freqai_preflight.json")
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
        frame = pd.read_csv(result["artifacts"]["route_decisions"]["path"])
        frame["cohort"] = result["cohort"]
        frame["evaluation_window"] = result["evaluation_window"]
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    keys = ["route_id", "target", "anchor_source_timeframe"]
    rows: list[dict[str, Any]] = []
    for key, cell in combined.groupby(keys, observed=True, sort=False):
        indexed = cell.set_index(["cohort", "evaluation_window"])
        expected = {
            ("normal", g13z.STANDARD),
            ("normal", g13z.RECENT),
            ("meme", g13z.STANDARD),
        }
        observed = set(indexed.index)
        complete = observed == expected
        point_all = bool(
            complete and indexed["all_controls_point_positive"].astype(bool).all()
        )
        strict_all = bool(
            complete and indexed["all_controls_strict"].astype(bool).all()
        )
        standard = cell.loc[cell["evaluation_window"].eq(g13z.STANDARD)]
        standard_point = bool(
            set(standard["cohort"]) == {"normal", "meme"}
            and standard["all_controls_point_positive"].astype(bool).all()
        )
        if strict_all:
            status = "strict_all_cohorts_and_later_normal"
        elif point_all:
            status = "point_all_cohorts_and_later_normal"
        elif standard_point:
            status = "standard_both_cohorts_only"
        else:
            status = "not_retained_across_generation13_windows"
        rows.append(
            {
                "route_id": key[0],
                "target": key[1],
                "anchor_source_timeframe": key[2],
                "status": status,
                "complete_three_cell_ladder": complete,
                "point_all_three_cells": point_all,
                "strict_all_three_cells": strict_all,
                "standard_both_cohorts_point": standard_point,
                "minimum_mae_gain": float(
                    cell["minimum_equal_coin_paired_mae_gain"].min()
                ),
                "minimum_bootstrap_lower": float(
                    cell["minimum_bootstrap_lower"].min()
                ),
            }
        )
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def joint_review(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    expected = {
        (stage, window, cohort)
        for stage in g13z.STAGES
        for window, cohort in (
            (g13z.STANDARD, "normal"),
            (g13z.RECENT, "normal"),
            (g13z.STANDARD, "meme"),
        )
    }
    observed = {
        (item["run_stage"], item["evaluation_window"], item["cohort"])
        for item in results
    }
    if observed != expected:
        raise ValueError("Generation 13 joint review requires all six FreqAI cells.")
    direct = validate_direct_sibling()
    record_dir = RECORD_ROOT / "g13_joint_review_20260822a"
    record_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    summaries: dict[str, Any] = {}
    for stage in g13z.STAGES:
        decisions = joint_stage_decisions(results, stage)
        short = "mtf" if stage == g13z.MTF_STAGE else "long"
        path = record_dir / f"g13_joint_{short}_decisions.csv"
        g0.atomic_write_csv(decisions, path)
        outputs[short] = path
        summaries[short] = {
            "rows": len(decisions),
            "strict_all_three": int(decisions["strict_all_three_cells"].sum()),
            "point_all_three": int(decisions["point_all_three_cells"].sum()),
            "standard_both_cohorts": int(
                decisions["standard_both_cohorts_point"].sum()
            ),
        }
    output = {
        "schema_version": 1,
        "generation": 13,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation13_joint_review",
        "all_frozen_siblings_completed_before_review": True,
        "direct_control_summary": direct["summary"],
        "freqai_summary": summaries,
        "automatic_descendant_launch": False,
        "research_boundary": {
            "direction_tested": False,
            "joint_55_percent_direction_target_reached": False,
            "profit_used": False,
            "trading_promotion": False,
        },
        "source_results": [
            artifact(Path(item["result_path"])) for item in results
        ],
        "source_direct_control": artifact(DIRECT_RESULT),
        "artifacts": {key: artifact(path) for key, path in outputs.items()},
    }
    result_path = record_dir / "g13_joint_review.json"
    g0.atomic_write_json(output, result_path)
    return {**output, "result_path": str(result_path.resolve())}


def requested_runs(
    stage: str, window: str, cohort: str
) -> list[tuple[str, str, str]]:
    runs: list[tuple[str, str, str]] = []
    for selected_stage in g13z.STAGES:
        if stage not in ("all", selected_stage):
            continue
        candidates = (
            (g13z.STANDARD, "normal"),
            (g13z.RECENT, "normal"),
            (g13z.STANDARD, "meme"),
        )
        for selected_window, selected_cohort in candidates:
            if window not in ("all", selected_window):
                continue
            if cohort not in ("all", selected_cohort):
                continue
            runs.append((selected_stage, selected_window, selected_cohort))
    if not runs:
        raise ValueError("No Generation 13 runs match the stage/window/cohort selection.")
    return runs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run frozen Generation 13 FreqAI cells.")
    parser.add_argument("--stage", choices=("all", *g13z.STAGES), default="all")
    parser.add_argument(
        "--window", choices=("all", g13z.STANDARD, g13z.RECENT), default="all"
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
    runs = requested_runs(args.stage, args.window, args.cohort)
    stem = args.run_stem or (
        DEFAULT_SMOKE_STEM if args.technical_smoke else DEFAULT_RUN_STEM
    )
    results: list[dict[str, Any]] = []
    for stage, window, cohort in runs:
        short_stage = "mtf" if stage == g13z.MTF_STAGE else "long"
        short_window = "standard" if window == g13z.STANDARD else "recent"
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

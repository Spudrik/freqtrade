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
    market_reaction_zone_freqai_generation9 as g9,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation10_cache as g10c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation10_freeze as g10z,
)


USER_DATA_DIR = REPO_ROOT / "user_data"
DEFAULT_CONFIG = USER_DATA_DIR / "configs" / "config_market_reaction_zone_freqai.example.json"
DEFAULT_PYTHON = (
    REPO_ROOT
    / "runtime"
    / "venvs"
    / "freqtrade-backtest-08"
    / "Scripts"
    / "python.exe"
)
RECORD_ROOT = (
    g10z.OUTPUT_ROOT
    / "generation10_branches"
    / "g10_untouched_participation_confirmation"
)
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData")
    / "research_outputs"
    / "market_reaction_zones"
    / "generation10_branches"
    / "g10_untouched_participation_confirmation"
)
DEFAULT_RUN_ID = "g10_participation_confirmation_20260821a"
MAX_WORKERS = 4
MIN_TRAINING_ROWS = 30


def load_sources() -> tuple[dict[str, Any], dict[str, Any], Path]:
    frozen = g10z.validate_existing_freeze(g10z.FREEZE_PATH)
    cache_path = g10c.RECORD_ROOT / "g10_cache_manifest.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache.get("status") != "completed_generation10_confirmation_cache":
        raise ValueError("Generation 10 confirmation cache is not terminal.")
    frozen_source = cache["source_contracts"]["generation10_freeze"]
    if g0.sha256_file(g10z.FREEZE_PATH) != frozen_source["sha256"]:
        raise ValueError("Generation 10 freeze changed after cache construction.")
    return frozen, cache, cache_path


def build_manifest(
    *,
    run_id: str,
    base_config: Path,
    python_exe: Path,
    profile_workers: int,
) -> tuple[dict[str, Any], Path, Path]:
    frozen, cache, cache_path = load_sources()
    record_dir = RECORD_ROOT / run_id
    artifact_dir = ARTIFACT_ROOT / run_id
    manifest_path = record_dir / "g10_freqai_run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = manifest["source_contracts"]["generation10_freeze"]
        if g0.sha256_file(g10z.FREEZE_PATH) != source["sha256"]:
            raise ValueError("Generation 10 freeze changed after run preparation.")
        return manifest, manifest_path, artifact_dir
    record_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    base = json.loads(base_config.read_text(encoding="utf-8"))
    profiles = {key: dict(value) for key, value in frozen["profiles"].items()}
    comparisons = [dict(value) for value in frozen["comparisons"]]
    feature_dir = Path(cache["storage"]["feature_dir"])
    event_dir = Path(cache["storage"]["event_dir"])
    pairs = tuple(str(pair) for pair in frozen["pairs"])
    commands: list[dict[str, Any]] = []
    for number, (profile_id, profile) in enumerate(profiles.items(), start=1):
        short_id = f"p{number:03d}_{g8.g6f.stable_digest(profile_id, 6)}"
        profile_dir = artifact_dir / "profiles" / short_id
        userdir = profile_dir / "user_data"
        export_dir = profile_dir / "export"
        identifier = f"g10-{g8.g6f.stable_digest(f'{run_id}|{profile_id}', 16)}"
        config_path = record_dir / f"config_{short_id}.json"
        g0.atomic_write_json(
            g8.profile_config(
                base,
                identifier=identifier,
                pairs=pairs,
                feature_dir=feature_dir,
                event_dir=event_dir,
                profile=profile,
                train_days=int(frozen["train_days"]),
                backtest_days=int(frozen["backtest_days"]),
                technical_smoke=False,
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
            str(g8.STRATEGY_PATH),
            "--datadir",
            str(g8.DATA_DIR),
            "--config",
            str(config_path),
            "--strategy",
            g8.STRATEGY_CLASS,
            "--freqaimodel",
            "LightGBMRegressorMultiTarget",
            "--timerange",
            str(frozen["timerange"]),
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
                "strategy": g8.STRATEGY_CLASS,
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
        "generation": 10,
        "run_stage": "target_and_mechanism_unopened_chronological_confirmation",
        "technical_smoke_not_evidence": False,
        "cohort": "normal",
        "pairs": list(pairs),
        "timerange": str(frozen["timerange"]),
        "train_period_days": int(frozen["train_days"]),
        "backtest_period_days": int(frozen["backtest_days"]),
        "validation_periods": list(frozen["confirmation_periods"]),
        "declared_groups": list(frozen["declared_groups"]),
        "profile_workers": max(1, min(int(profile_workers), MAX_WORKERS)),
        "model_threads_per_profile": 1,
        "model_class": "LightGBMRegressorMultiTarget",
        "profiles": list(profiles),
        "profile_count": len(profiles),
        "comparisons": comparisons,
        "comparison_count": len(comparisons),
        "questions": sorted({item["question_id"] for item in profiles.values()}),
        "seeds": list(frozen["seeds"]),
        "profile_registry_frozen_before_model_outcomes": True,
        "profile_registry": {"profiles": profiles, "comparisons": comparisons},
        "research_boundary": dict(frozen["research_boundary"]),
        "source_contracts": {
            "base_config": {
                "path": str(base_config.resolve()),
                "sha256": g0.sha256_file(base_config),
            },
            "python_executable": str(python_exe.resolve()),
            "strategy": {
                "path": str(g8.STRATEGY_FILE.resolve()),
                "sha256": g0.sha256_file(g8.STRATEGY_FILE),
            },
            "generation10_freeze": {
                "path": str(g10z.FREEZE_PATH.resolve()),
                "sha256": g0.sha256_file(g10z.FREEZE_PATH),
            },
            "generation10_cache": {
                "path": str(cache_path.resolve()),
                "sha256": g0.sha256_file(cache_path),
            },
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


def preflight_run(manifest: dict[str, Any], *, python_exe: Path) -> dict[str, Any]:
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
    if not g8.STRATEGY_FILE.is_file():
        problems.append(f"Missing research strategy: {g8.STRATEGY_FILE}")
    for item in manifest["source_contracts"]["cache_inventory"]:
        for key in ("feature", "event"):
            path = Path(item[f"{key}_path"])
            if not path.is_file() or g0.sha256_file(path) != item[f"{key}_sha256"]:
                problems.append(f"Cache hash mismatch for {item['pair']} and {key}")
    start_raw, end_raw = str(manifest["timerange"]).split("-", maxsplit=1)
    prediction_start = pd.Timestamp(start_raw, tz="UTC")
    prediction_end = pd.Timestamp(end_raw, tz="UTC")
    training_start = prediction_start - pd.Timedelta(
        days=int(manifest["train_period_days"])
    )
    event_dir = Path(manifest["storage"]["event_cache_dir"])
    readiness: list[dict[str, Any]] = []
    requirements = {
        tuple(item["required_ready_blocks"]) for item in manifest["commands"]
    }
    for blocks in sorted(requirements):
        for pair in manifest["pairs"]:
            ready_columns = [f"ready__{block}" for block in blocks]
            frame = pd.read_parquet(
                event_dir / f"{g0.pair_file_stem(pair)}.parquet",
                columns=["date", g10z.TARGET, *ready_columns],
            )
            frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
            eligible = frame[ready_columns].fillna(False).astype(bool).all(axis=1)
            eligible &= frame[g10z.TARGET].notna()
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
                    "ready_blocks": ",".join(blocks),
                    "training_rows": training_rows,
                    "prediction_rows": prediction_rows,
                }
            )
            if training_rows < MIN_TRAINING_ROWS:
                problems.append(
                    f"{pair} has only {training_rows} training rows for {blocks}."
                )
            if prediction_rows < g7f.MIN_PAIR_SCORABLE_ROWS:
                problems.append(
                    f"{pair} has only {prediction_rows} prediction rows for {blocks}."
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


def load_actual_with_regime(manifest: dict[str, Any]) -> DataFrame:
    frames: list[DataFrame] = []
    event_dir = Path(manifest["storage"]["event_cache_dir"])
    for pair in manifest["pairs"]:
        frame = pd.read_parquet(
            event_dir / f"{g0.pair_file_stem(pair)}.parquet",
            columns=["date", "period", "activity_regime", g10z.TARGET],
        )
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        frame["pair"] = pair
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    if output.duplicated(["pair", "date"]).any():
        raise ValueError("Duplicate Generation 10 actual pair/date rows.")
    return output


def mechanism_summary(joint: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    expected_groups = set(g10z.DECLARED_GROUPS)
    for question_id, frame in joint.groupby("question_id", observed=True):
        groups = set(frame["group_id"])
        if groups != expected_groups:
            raise ValueError(
                f"Generation 10 {question_id} group coverage drifted: {groups}"
            )
        strict = frame["status"].eq("strict_three_seed_lead")
        provisional_or_strict = frame["status"].isin(
            {"strict_three_seed_lead", "provisional_three_seed_lead"}
        )
        if strict.all():
            status = "confirmed_strict_all_declared_groups"
        elif provisional_or_strict.all():
            status = "confirmed_provisional_all_declared_groups"
        else:
            status = "failed_untouched_confirmation"
        rows.append(
            {
                "question_id": question_id,
                "route_id": frame["route_id"].iloc[0],
                "plain_name": frame["plain_name"].iloc[0],
                "status": status,
                "strict_groups": ",".join(sorted(frame.loc[strict, "group_id"])),
                "provisional_groups": ",".join(
                    sorted(
                        frame.loc[
                            frame["status"].eq("provisional_three_seed_lead"),
                            "group_id",
                        ]
                    )
                ),
                "failed_groups": ",".join(
                    sorted(
                        frame.loc[
                            frame["status"].eq("failed_three_seed_replication"),
                            "group_id",
                        ]
                    )
                ),
                "minimum_equal_coin_paired_mae_gain": float(
                    frame["minimum_equal_coin_paired_mae_gain"].min()
                ),
                "minimum_bootstrap_lower": float(
                    frame["minimum_bootstrap_lower"].min()
                ),
            }
        )
    return DataFrame.from_records(rows).sort_values("question_id")


def regime_scores(
    *,
    manifest: dict[str, Any],
    predictions: dict[str, DataFrame],
    actual: DataFrame,
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    groups = g7f.group_definitions("normal", manifest["pairs"])
    declared = {item["group_id"] for item in groups}.intersection(
        manifest["declared_groups"]
    )
    groups = [item for item in groups if item["group_id"] in declared]
    for definition in manifest["comparisons"]:
        if definition["control_type"] != "candidate_absent_or_unavailable":
            continue
        fair, _ = g7f.fair_comparison_frame(
            candidate=predictions[definition["candidate"]],
            baseline=predictions[definition["baseline"]],
            actual=actual,
            targets=(g10z.TARGET,),
        )
        fair = fair.merge(
            actual[["pair", "date", "activity_regime"]],
            on=["pair", "date"],
            how="left",
            validate="one_to_one",
        )
        actual_column = f"{g10z.TARGET}__actual"
        candidate_column = f"{g10z.TARGET}__candidate"
        baseline_column = f"{g10z.TARGET}__baseline"
        for group in groups:
            for regime in ("quiet", "typical", "active"):
                selected = fair.loc[
                    fair["pair"].isin(group["members"])
                    & fair["activity_regime"].eq(regime)
                ].copy()
                numeric = selected[
                    [actual_column, candidate_column, baseline_column]
                ].apply(pd.to_numeric, errors="coerce")
                selected = selected.loc[np.isfinite(numeric).all(axis=1)].copy()
                selected["paired_gain"] = (
                    selected[actual_column] - selected[baseline_column]
                ).abs() - (
                    selected[actual_column] - selected[candidate_column]
                ).abs()
                coin_rows = selected.groupby("pair", observed=True).size()
                scorable = coin_rows.loc[coin_rows.ge(5)].index
                selected = selected.loc[selected["pair"].isin(scorable)]
                coin_gain = selected.groupby("pair", observed=True)["paired_gain"].mean()
                rows.append(
                    {
                        **definition,
                        "group_id": group["group_id"],
                        "activity_regime": regime,
                        "rows": len(selected),
                        "scorable_coins": len(coin_gain),
                        "positive_coins": int(coin_gain.gt(0.0).sum()),
                        "equal_coin_paired_mae_gain": (
                            float(coin_gain.mean()) if len(coin_gain) else np.nan
                        ),
                        "descriptive_only": True,
                    }
                )
    return DataFrame.from_records(rows)


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
            raise ValueError(f"No predictions for Generation 10 profile {item['profile_id']}")
        predictions[str(item["profile_id"])] = frame
    actual = load_actual_with_regime(manifest)
    pair_scores, group_scores, eligibility = g8.score_comparisons(
        manifest=manifest,
        predictions=predictions,
        actual=actual,
    )
    declared_scores = group_scores.loc[
        group_scores["group_id"].isin(manifest["declared_groups"])
    ].copy()
    seed_decisions = g9.route_decisions(manifest, declared_scores)
    joint = g9.joint_seed_decisions(seed_decisions)
    mechanisms = mechanism_summary(joint)
    regimes = regime_scores(manifest=manifest, predictions=predictions, actual=actual)
    paths = {
        "prediction_audit": record_dir / "g10_prediction_audit.csv",
        "comparison_eligibility": record_dir / "g10_comparison_eligibility.csv",
        "pair_scores": record_dir / "g10_pair_scores.csv",
        "group_scores": record_dir / "g10_declared_group_scores.csv",
        "seed_decisions": record_dir / "g10_seed_decisions.csv",
        "joint_group_decisions": record_dir / "g10_joint_group_decisions.csv",
        "mechanism_summary": record_dir / "g10_mechanism_summary.csv",
        "activity_regime_scores": record_dir / "g10_activity_regime_scores.csv",
    }
    g0.atomic_write_csv(DataFrame.from_records(audits), paths["prediction_audit"])
    g0.atomic_write_csv(eligibility, paths["comparison_eligibility"])
    g0.atomic_write_csv(pair_scores, paths["pair_scores"])
    g0.atomic_write_csv(declared_scores, paths["group_scores"])
    g0.atomic_write_csv(seed_decisions, paths["seed_decisions"])
    g0.atomic_write_csv(joint, paths["joint_group_decisions"])
    g0.atomic_write_csv(mechanisms, paths["mechanism_summary"])
    g0.atomic_write_csv(regimes, paths["activity_regime_scores"])
    strict_mechanisms = int(
        mechanisms["status"].eq("confirmed_strict_all_declared_groups").sum()
    )
    provisional_mechanisms = int(
        mechanisms["status"].eq("confirmed_provisional_all_declared_groups").sum()
    )
    result = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation10_participation_confirmation",
        "period_label": manifest["run_stage"],
        "globally_pristine_market_history": False,
        "cohort": "normal",
        "seeds": manifest["seeds"],
        "profiles_completed": len(predictions),
        "questions_completed": len(manifest["questions"]),
        "comparisons_completed": len(manifest["comparisons"]),
        "declared_group_routes": len(joint),
        "strict_group_routes": int(joint["status"].eq("strict_three_seed_lead").sum()),
        "provisional_group_routes": int(
            joint["status"].eq("provisional_three_seed_lead").sum()
        ),
        "failed_group_routes": int(
            joint["status"].eq("failed_three_seed_replication").sum()
        ),
        "strict_mechanisms_all_declared_groups": strict_mechanisms,
        "provisional_mechanisms_all_declared_groups": provisional_mechanisms,
        "failed_mechanisms": len(mechanisms)
        - strict_mechanisms
        - provisional_mechanisms,
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
            "profile_registry_frozen_before_volume_target_outcomes": True,
            "profit_used": False,
            "future_signed_direction_used": False,
        },
        "pass_rule": g10z.validate_existing_freeze(g10z.FREEZE_PATH)[
            "strict_pass_rule"
        ],
        "artifacts": {
            key: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": g0.sha256_file(path),
            }
            for key, path in paths.items()
        },
        "interpretation_boundary": (
            "Unsigned next-hour reaction-volume confirmation only. The same dates were "
            "previously used for an unrelated absolute-excursion question, so this is "
            "target-and-mechanism-unopened chronological evidence, not globally pristine "
            "market history. It is not direction, profit, an entry, an exit, or a strategy."
        ),
    }
    result_path = record_dir / "g10_freqai_result.json"
    g0.atomic_write_json(result, result_path)
    return {**result, "result_path": str(result_path)}


def run_generation10(
    *,
    run_id: str,
    base_config: Path,
    python_exe: Path,
    profile_workers: int,
) -> dict[str, Any]:
    manifest, manifest_path, _ = build_manifest(
        run_id=run_id,
        base_config=base_config,
        python_exe=python_exe,
        profile_workers=profile_workers,
    )
    audit = preflight_run(manifest, python_exe=python_exe)
    audit_path = manifest_path.parent / "g10_freqai_preflight.json"
    g0.atomic_write_json(audit, audit_path)
    if not audit["passed"]:
        manifest["status"] = "blocked_preflight"
        manifest["preflight"] = audit
        g0.atomic_write_json(manifest, manifest_path)
        return {"status": "blocked_preflight", "preflight": audit}
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen Generation 10 participation confirmations."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--profile-workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args(argv)
    if not args.base_config.is_file():
        raise FileNotFoundError(args.base_config)
    if not args.python_exe.is_file():
        raise FileNotFoundError(args.python_exe)
    result = run_generation10(
        run_id=args.run_id,
        base_config=args.base_config,
        python_exe=args.python_exe,
        profile_workers=max(1, min(args.profile_workers, MAX_WORKERS)),
    )
    print(json.dumps(result, indent=2, default=g0.json_default), flush=True)
    return 0 if str(result["status"]).startswith("completed_") else 2


if __name__ == "__main__":
    raise SystemExit(main())

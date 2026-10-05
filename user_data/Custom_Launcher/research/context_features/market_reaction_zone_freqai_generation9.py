from __future__ import annotations

# Bound each model process to one numerical thread. The manifest runner owns parallelism.
# ruff: noqa: E402
import os


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation8_cache as g8c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation9_freeze as g9z,
)


OUTPUT_ROOT = g9z.OUTPUT_ROOT
RECORD_ROOT = OUTPUT_ROOT / "generation9_branches" / "g9_limited_multisource_freqai"
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData")
    / "research_outputs"
    / "market_reaction_zones"
    / "generation9_branches"
    / "g9_limited_multisource_freqai"
)
DEFAULT_CONFIG = g8.DEFAULT_CONFIG
DEFAULT_PYTHON = g8.DEFAULT_PYTHON
MAX_WORKERS = 4


def load_frozen(path: Path = g9z.FREEZE_PATH) -> dict[str, Any]:
    frozen = g9z.validate_existing_freeze(path)
    if tuple(frozen["limited_three_source_family"]["seeds"]) != g9z.SEEDS:
        raise ValueError("The frozen Generation 9 model seeds drifted.")
    if frozen["limited_three_source_family"]["controls_per_question_seed"] != 9:
        raise ValueError("The frozen Generation 9 control count drifted.")
    for source in frozen["source_contracts"]["generation8_cache_manifests"].values():
        source_path = Path(source["path"])
        if not source_path.is_file() or g0.sha256_file(source_path) != source["sha256"]:
            raise ValueError(f"Frozen Generation 8 source changed: {source_path}")
    review = frozen["source_contracts"]["generation8_terminal_joint_review"]
    review_path = Path(review["path"])
    if not review_path.is_file() or g0.sha256_file(review_path) != review["sha256"]:
        raise ValueError("The Generation 8 terminal review changed after Generation 9 freeze.")
    return frozen


def cohort_registry(
    frozen: dict[str, Any], cohort: str
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    family = frozen["limited_three_source_family"]
    profiles = {
        profile_id: dict(profile)
        for profile_id, profile in family["profiles"].items()
        if profile["cohort"] == cohort
    }
    comparisons = [
        dict(item) for item in family["comparisons"] if item["cohort"] == cohort
    ]
    active = set(profiles)
    if any(
        item["candidate"] not in active or item["baseline"] not in active
        for item in comparisons
    ):
        raise ValueError("A Generation 9 comparison crosses cohort registries.")
    return profiles, comparisons


def build_manifest(
    *,
    run_id: str,
    cohort: str,
    frozen_path: Path,
    base_config: Path,
    python_exe: Path,
    profile_workers: int,
) -> tuple[dict[str, Any], Path, Path]:
    frozen = load_frozen(frozen_path)
    profiles, comparisons = cohort_registry(frozen, cohort)
    record_dir = RECORD_ROOT / run_id
    artifact_dir = ARTIFACT_ROOT / run_id
    manifest_path = record_dir / "g9_freqai_run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selection = manifest["source_contracts"]["frozen_generation9_selection"]
        if g0.sha256_file(frozen_path) != selection["sha256"]:
            raise ValueError("The frozen Generation 9 selection changed after run preparation.")
        return manifest, manifest_path, artifact_dir
    record_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    shared = g8.cache_manifest(cohort)
    inventory = [dict(item) for item in shared["inventory"]]
    feature_dir = Path(inventory[0]["feature_path"]).parent
    event_dir = Path(inventory[0]["event_path"]).parent
    pairs = tuple(str(pair) for pair in shared["pairs"])
    settings = dict(g7f.cohort_settings(cohort))
    base = json.loads(base_config.read_text(encoding="utf-8"))
    commands: list[dict[str, Any]] = []
    for number, (profile_id, profile) in enumerate(profiles.items(), start=1):
        subset = g8.profile_pairs(profile, pairs)
        if not subset:
            raise ValueError(f"No pairs remain for Generation 9 profile {profile_id}.")
        short_id = f"p{number:03d}_{g8.g6f.stable_digest(profile_id, 6)}"
        profile_dir = artifact_dir / "profiles" / short_id
        userdir = profile_dir / "user_data"
        export_dir = profile_dir / "export"
        identifier = f"g9-{g8.g6f.stable_digest(f'{run_id}|{profile_id}', 16)}"
        config_path = record_dir / f"config_{short_id}.json"
        g0.atomic_write_json(
            g8.profile_config(
                base,
                identifier=identifier,
                pairs=subset,
                feature_dir=feature_dir,
                event_dir=event_dir,
                profile=profile,
                train_days=int(settings["train_days"]),
                backtest_days=int(settings["backtest_days"]),
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
                "pairs": list(subset),
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
        "status": "prepared",
        "created_at_utc": g0.utc_now(),
        "generation": 9,
        "run_stage": "limited_three_source_all_seed_attribution",
        "technical_smoke_not_evidence": False,
        "cohort": cohort,
        "pairs": list(pairs),
        "timerange": str(settings["timerange"]),
        "train_period_days": int(settings["train_days"]),
        "backtest_period_days": int(settings["backtest_days"]),
        "validation_periods": list(settings["validation_periods"]),
        "profile_workers": max(1, min(int(profile_workers), MAX_WORKERS)),
        "model_threads_per_profile": 1,
        "model_class": "LightGBMRegressorMultiTarget",
        "profiles": list(profiles),
        "profile_count": len(profiles),
        "comparisons": comparisons,
        "comparison_count": len(comparisons),
        "questions": sorted({item["question_id"] for item in profiles.values()}),
        "seeds": list(g9z.SEEDS),
        "profile_registry_frozen_before_model_outcomes": True,
        "profile_registry": {"profiles": profiles, "comparisons": comparisons},
        "research_boundary": {
            "profit_used": False,
            "future_signed_direction": False,
            "entry_exit_construction": False,
            "strategy_promotion": False,
        },
        "source_contracts": {
            "base_config": str(base_config.resolve()),
            "base_config_sha256": g0.sha256_file(base_config),
            "python_executable": str(python_exe.resolve()),
            "strategy": str(g8.STRATEGY_FILE.resolve()),
            "strategy_sha256": g0.sha256_file(g8.STRATEGY_FILE),
            "cache_manifest": str(
                (g8c.RECORD_ROOT / f"{cohort}_manifest.json").resolve()
            ),
            "cache_manifest_sha256": g0.sha256_file(
                g8c.RECORD_ROOT / f"{cohort}_manifest.json"
            ),
            "cache_inventory": inventory,
            "frozen_generation9_selection": {
                "path": str(frozen_path.resolve()),
                "sha256": g0.sha256_file(frozen_path),
            },
        },
        "storage": {
            "record_dir": str(record_dir),
            "bulky_artifact_dir": str(artifact_dir),
            "feature_cache_dir": str(feature_dir),
            "event_cache_dir": str(event_dir),
            "save_backtest_models": False,
        },
        "runtime_preparation_snapshot": g7f.runtime_snapshot(),
        "commands": commands,
    }
    g0.atomic_write_json(manifest, manifest_path)
    return manifest, manifest_path, artifact_dir


def route_decisions(manifest: dict[str, Any], scores: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    keys = [
        "question_id",
        "branch_id",
        "surface",
        "route_id",
        "route_type",
        "target",
        "group_id",
        "group_members",
        "seed",
    ]
    expected_periods = set(manifest["validation_periods"])
    for key, frame in scores.groupby(keys, dropna=False):
        values = dict(zip(keys, key, strict=True))
        expected_controls = int(frame["expected_controls_for_route"].iloc[0])
        controls = int(frame["comparison_id"].nunique())
        periods_complete = all(
            set(group["period"]) == expected_periods
            for _, group in frame.groupby("comparison_id", observed=True)
        )
        strict = bool(
            controls == expected_controls
            and periods_complete
            and frame["strict_period_pass"].fillna(False).astype(bool).all()
        )
        provisional = bool(
            controls == expected_controls
            and periods_complete
            and frame["provisional_period_pass"].fillna(False).astype(bool).all()
        )
        declared_group = str(frame["result_group"].iloc[0])
        group_eligible = bool(
            declared_group == "all_declared_groups"
            or declared_group == values["group_id"]
        )
        if not group_eligible:
            status = "outside_predeclared_result_group"
        elif strict:
            status = "strict_seed_pass"
        elif provisional:
            status = "provisional_seed_pass"
        else:
            status = "seed_failed_one_or_more_controls"
        rows.append(
            {
                **values,
                "plain_name": frame["plain_name"].iloc[0],
                "mechanism": frame["mechanism"].iloc[0],
                "declared_result_group": declared_group,
                "group_eligible": group_eligible,
                "status": status,
                "expected_controls": expected_controls,
                "controls_present": controls,
                "both_validation_periods_present": periods_complete,
                "all_controls_strict": strict,
                "all_controls_point_positive": provisional,
                "minimum_equal_coin_paired_mae_gain": float(
                    frame["equal_coin_paired_mae_gain"].min()
                ),
                "minimum_bootstrap_lower": float(frame["bootstrap_lower"].min()),
            }
        )
    return DataFrame.from_records(rows)


def joint_seed_decisions(seed_decisions: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    keys = [
        "question_id",
        "branch_id",
        "surface",
        "route_id",
        "route_type",
        "target",
        "group_id",
        "group_members",
        "declared_result_group",
    ]
    eligible = seed_decisions.loc[seed_decisions["group_eligible"]].copy()
    for key, frame in eligible.groupby(keys, dropna=False):
        values = dict(zip(keys, key, strict=True))
        seeds = tuple(sorted(int(seed) for seed in frame["seed"].unique()))
        all_seeds_present = seeds == tuple(sorted(g9z.SEEDS))
        strict = bool(all_seeds_present and frame["all_controls_strict"].all())
        provisional = bool(
            all_seeds_present and frame["all_controls_point_positive"].all()
        )
        if strict:
            status = "strict_three_seed_lead"
        elif provisional:
            status = "provisional_three_seed_lead"
        else:
            status = "failed_three_seed_replication"
        rows.append(
            {
                **values,
                "plain_name": frame["plain_name"].iloc[0],
                "mechanism": frame["mechanism"].iloc[0],
                "status": status,
                "seeds_present": ",".join(str(seed) for seed in seeds),
                "all_three_seeds_present": all_seeds_present,
                "strict_seed_count": int(frame["all_controls_strict"].sum()),
                "point_positive_seed_count": int(
                    frame["all_controls_point_positive"].sum()
                ),
                "minimum_equal_coin_paired_mae_gain": float(
                    frame["minimum_equal_coin_paired_mae_gain"].min()
                ),
                "minimum_bootstrap_lower": float(
                    frame["minimum_bootstrap_lower"].min()
                ),
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
            raise ValueError(f"No predictions for Generation 9 profile {item['profile_id']}")
        missing = sorted(set(item["targets"]).difference(frame.columns))
        if missing:
            raise ValueError(f"Profile {item['profile_id']} lacks targets: {missing}")
        predictions[str(item["profile_id"])] = frame
    actual = g8.load_event_targets(manifest)
    pair_scores, group_scores, eligibility = g8.score_comparisons(
        manifest=manifest, predictions=predictions, actual=actual
    )
    seed_routes = route_decisions(manifest, group_scores)
    joint_routes = joint_seed_decisions(seed_routes)
    paths = {
        "prediction_audit": record_dir / "g9_freqai_prediction_audit.csv",
        "comparison_eligibility": record_dir / "g9_freqai_comparison_eligibility.csv",
        "pair_scores": record_dir / "g9_freqai_pair_scores.csv",
        "group_scores": record_dir / "g9_freqai_group_scores.csv",
        "seed_route_decisions": record_dir / "g9_freqai_seed_route_decisions.csv",
        "joint_seed_decisions": record_dir / "g9_freqai_joint_seed_decisions.csv",
    }
    g0.atomic_write_csv(DataFrame.from_records(audits), paths["prediction_audit"])
    g0.atomic_write_csv(eligibility, paths["comparison_eligibility"])
    g0.atomic_write_csv(pair_scores, paths["pair_scores"])
    g0.atomic_write_csv(group_scores, paths["group_scores"])
    g0.atomic_write_csv(seed_routes, paths["seed_route_decisions"])
    g0.atomic_write_csv(joint_routes, paths["joint_seed_decisions"])
    unequal = int((~eligibility["identical_prediction_keys"]).sum())
    strict = int(joint_routes["status"].eq("strict_three_seed_lead").sum())
    provisional = int(
        joint_routes["status"].eq("provisional_three_seed_lead").sum()
    )
    result = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "status": "completed_generation9_limited_multisource",
        "created_at_utc": g0.utc_now(),
        "cohort": manifest["cohort"],
        "seeds": manifest["seeds"],
        "profiles_completed": len(predictions),
        "questions_completed": len(manifest["questions"]),
        "comparisons_completed": len(manifest["comparisons"]),
        "eligible_group_routes_scored": len(joint_routes),
        "strict_three_seed_routes": strict,
        "provisional_three_seed_routes": provisional,
        "failed_three_seed_routes": len(joint_routes) - strict - provisional,
        "integrity": {
            "unequal_prediction_key_comparisons": unequal,
            "duplicate_prediction_rows_removed": int(
                sum(item.get("duplicate_pair_date_rows", 0) for item in audits)
            ),
            "profit_used": False,
            "future_signed_direction": False,
            "all_profile_commands_terminal": all(
                item.get("status") == "completed" for item in manifest["commands"]
            ),
            "profile_registry_frozen_before_model_outcomes": True,
        },
        "pass_rule": (
            "The complete three-source model must beat all nine predeclared controls in "
            "both validation periods and all three seeds for the declared market group."
        ),
        "artifacts": {
            key: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": g0.sha256_file(path),
            }
            for key, path in paths.items()
        },
        "interpretation_boundary": (
            "Direction-neutral reaction-magnitude estimation only; not causation, profit, "
            "an entry, an exit, or a trading rule."
        ),
    }
    result_path = record_dir / "g9_freqai_result.json"
    g0.atomic_write_json(result, result_path)
    return {**result, "result_path": str(result_path)}


def run_generation9(
    *,
    run_id: str,
    cohort: str,
    frozen_path: Path,
    base_config: Path,
    python_exe: Path,
    profile_workers: int,
) -> dict[str, Any]:
    manifest, manifest_path, _ = build_manifest(
        run_id=run_id,
        cohort=cohort,
        frozen_path=frozen_path,
        base_config=base_config,
        python_exe=python_exe,
        profile_workers=profile_workers,
    )
    audit = g8.preflight_run(manifest, python_exe=python_exe)
    audit_path = manifest_path.parent / "g9_freqai_preflight.json"
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
        description=(
            "Run the frozen Generation 9 limited three-source FreqAI family against "
            "leave-one-out, stale-input, and shuffled-input controls."
        )
    )
    parser.add_argument("--cohort", choices=("normal", "meme"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--freeze-path", type=Path, default=g9z.FREEZE_PATH)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--profile-workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args(argv)
    if not args.base_config.is_file():
        raise FileNotFoundError(args.base_config)
    if not args.python_exe.is_file():
        raise FileNotFoundError(args.python_exe)
    result = run_generation9(
        run_id=args.run_id,
        cohort=args.cohort,
        frozen_path=args.freeze_path,
        base_config=args.base_config,
        python_exe=args.python_exe,
        profile_workers=max(1, min(args.profile_workers, MAX_WORKERS)),
    )
    print(json.dumps(result, indent=2, default=g0.json_default), flush=True)
    return 0 if str(result["status"]).startswith("completed_") else 2


if __name__ == "__main__":
    raise SystemExit(main())

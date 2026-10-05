"""Run the frozen fresh-period market-activity profiles through FreqAI."""

from __future__ import annotations

# Bound numerical pools before importing research modules.
# ruff: noqa: E402
import argparse
import json
import os
import shutil
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

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation7 as g7f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation24 as g24f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_fresh_confirmation_freeze as freshz,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_fresh_confirmation_support as freshs,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_fresh_confirmation_two_route as two,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
RECORD_ROOT = freshz.OUTPUT_ROOT / "freqai_state"
ARTIFACT_ROOT = Path(
    "D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones/"
    "fresh_confirmation_freqai_state"
)
DEFAULT_WORKER = g24f.DEFAULT_PYTHON
DEFAULT_BASE_CONFIG = g24f.DEFAULT_CONFIG
DEFAULT_RUN_ID = "fresh_state_freqai_20260923a"
SMOKE_RUN_ID = "fresh_state_freqai_smoke_20260923a"
READY_COLUMN = f"ready__{freshs.STATE_READY_BLOCK}"
MAX_HORIZON_HOURS = 8


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def verify_artifact(item: dict[str, Any]) -> None:
    path = Path(item["path"])
    if not path.is_file() or g0.sha256_file(path) != item["sha256"]:
        raise ValueError(f"Frozen input changed: {path}")


def sources() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    frozen = freshz.freeze()
    split = two.freeze()
    outcome = two.materialize()
    support_path = freshs.support_result_path(two.SUPPORT_ID)
    support = json.loads(support_path.read_text(encoding="utf-8"))
    state = outcome["routes"]["fresh_market_state_activity"]
    if support["routes"]["fresh_market_state_activity"]["support_gate_pass"] is not True:
        raise ValueError("Fresh activity support did not pass.")
    if state["outcome_gate_pass"] is not True or len(state["inventory"]) != 20:
        raise ValueError("Fresh activity outcome support is incomplete.")
    if (
        outcome["source_contracts"]["two_route_freeze"]["sha256"]
        != artifact(two.FREEZE_PATH)["sha256"]
    ):
        raise ValueError("Two-route outcome belongs to another frozen split.")
    for row in state["inventory"]:
        for key in ("actual_target_cache", "shuffled_target_cache", "evaluation_cache"):
            verify_artifact(row[key])
    for row in support["routes"]["fresh_market_state_activity"]["inventory"]:
        verify_artifact(row["feature"])
        verify_artifact(row["label_mapping"])
    return frozen, split, support, outcome


def profiles_for(
    frozen: dict[str, Any], outcome: dict[str, Any], *, smoke: bool
) -> list[dict[str, Any]]:
    sibling = next(
        item for item in frozen["siblings"] if item["id"] == "fresh_market_state_activity"
    )
    inventory = outcome["routes"]["fresh_market_state_activity"]["inventory"]
    targets_by_cohort = {
        cohort: tuple(next(row for row in inventory if row["cohort"] == cohort)["targets"])
        for cohort in ("normal", "meme")
    }
    profiles = []
    for item in sibling["profiles"]:
        profile = dict(item)
        profile["targets"] = list(targets_by_cohort[profile["cohort"]])
        profile["required_ready_blocks"] = [freshs.STATE_READY_BLOCK]
        profiles.append(profile)
    if len(profiles) != 16:
        raise ValueError("Expected the 16 frozen model/control profiles.")
    if smoke:
        first_cell = (profiles[0]["model_class"], profiles[0]["seed"])
        profiles = [
            item
            for item in profiles
            if item["cohort"] == profiles[0]["cohort"]
            and (item["model_class"], item["seed"]) == first_cell
        ]
        if {item["role"] for item in profiles} != set(freshz.STATE_ROLES):
            raise ValueError("Technical smoke must include actual and shuffled labels.")
    return profiles


def build_manifest(
    *, run_id: str, smoke: bool, worker: Path, base_config: Path, workers: int
) -> tuple[dict[str, Any], Path]:
    frozen, _split, support, outcome = sources()
    record_dir = RECORD_ROOT / run_id
    manifest_path = record_dir / "fresh_state_freqai_run_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for source in existing["source_contracts"].values():
            verify_artifact(source)
        return existing, manifest_path
    profiles = profiles_for(frozen, outcome, smoke=smoke)
    record_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = ARTIFACT_ROOT / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    base = json.loads(base_config.read_text(encoding="utf-8"))
    support_rows = {
        (row["cohort"], row["pair"]): row
        for row in support["routes"]["fresh_market_state_activity"]["inventory"]
    }
    outcome_rows = {
        (row["cohort"], row["pair"]): row
        for row in outcome["routes"]["fresh_market_state_activity"]["inventory"]
    }
    commands = []
    for number, profile in enumerate(profiles, start=1):
        cohort = profile["cohort"]
        pairs = list(frozen["data_contract"]["cohort_pairs"][cohort])
        if smoke:
            pairs = pairs[:1]
        first_key = cohort, pairs[0]
        feature_dir = Path(support_rows[first_key]["feature"]["path"]).parent
        event_key = (
            "actual_target_cache"
            if profile["target_cache"] == "actual"
            else "shuffled_target_cache"
        )
        event_dir = Path(outcome_rows[first_key][event_key]["path"]).parent
        short_id = f"p{number:03d}_{g7f.g6f.stable_digest(profile['profile_id'], 6)}"
        profile_dir = artifact_dir / "profiles" / short_id
        userdir = profile_dir / "user_data"
        export_dir = profile_dir / "export"
        digest_input = f"{run_id}|{profile['profile_id']}"
        identifier = f"fresh-{g7f.g6f.stable_digest(digest_input, 16)}"
        config_path = record_dir / f"config_{short_id}.json"
        config = g24f.profile_config(
            base,
            identifier=identifier,
            pairs=pairs,
            feature_dir=feature_dir,
            event_dir=event_dir,
            profile=profile,
            train_days=int(frozen["data_contract"]["training_days_by_cohort"][cohort]),
            backtest_days=2 if smoke else 15,
            technical_smoke=smoke,
        )
        g0.atomic_write_json(config, config_path)
        timerange = "20260820-20260822" if smoke else "20260820-20260920"
        command = [
            str(worker),
            "-m",
            "freqtrade",
            "backtesting",
            "--userdir",
            str(userdir),
            "--strategy-path",
            str(g24f.STRATEGY_PATH),
            "--datadir",
            str(g24f.DATA_DIR),
            "--config",
            str(config_path),
            "--strategy",
            g24f.STRATEGY_CLASS,
            "--freqaimodel",
            str(profile["model_class"]),
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
                "pairs": pairs,
                "identifier": identifier,
                "config_path": str(config_path.resolve()),
                "artifact_dir": str(profile_dir.resolve()),
                "user_data_dir": str(userdir.resolve()),
                "model_dir": str((userdir / "models" / identifier).resolve()),
                "export_dir": str(export_dir.resolve()),
                "feature_cache_dir": str(feature_dir.resolve()),
                "event_cache_dir": str(event_dir.resolve()),
                "command": command,
                "status": "pending",
                "attempts": 0,
            }
        )
    manifest = {
        "schema_version": 1,
        "batch_id": frozen["batch_id"],
        "run_id": run_id,
        "created_at_utc": g0.utc_now(),
        "status": "prepared",
        "technical_smoke_not_evidence": smoke,
        "profile_workers": max(1, min(workers, 2)),
        "model_threads_per_profile": 1,
        "profiles": profiles,
        "fresh_periods": frozen["fresh_periods"],
        "frozen_question": next(
            item for item in frozen["siblings"] if item["id"] == "fresh_market_state_activity"
        ),
        "source_contracts": {
            "original_freeze": artifact(freshz.FREEZE_PATH),
            "two_route_freeze": artifact(two.FREEZE_PATH),
            "two_route_outcome": artifact(two.RESULT_PATH),
            "joint_support": artifact(freshs.support_result_path(two.SUPPORT_ID)),
            "runner": artifact(ANALYSIS_PATH),
            "strategy": artifact(g24f.STRATEGY_FILE),
            "base_config": artifact(base_config),
        },
        "storage": {
            "record_dir": str(record_dir.resolve()),
            "bulky_artifact_dir": str(artifact_dir.resolve()),
        },
        "commands": commands,
    }
    g0.atomic_write_json(manifest, manifest_path)
    return manifest, manifest_path


def preflight(manifest: dict[str, Any], *, worker: Path) -> dict[str, Any]:
    dependency, problems = g24f.dependency_check(worker)
    if not worker.is_file():
        problems.append(f"Missing worker: {worker}")
    if manifest["profile_workers"] > 2:
        problems.append("More than two profile workers requested.")
    for source in manifest["source_contracts"].values():
        try:
            verify_artifact(source)
        except (FileNotFoundError, ValueError) as exc:
            problems.append(str(exc))
    for item in manifest["commands"]:
        for pair in item["pairs"]:
            stem = g0.pair_file_stem(pair)
            feature_path = Path(item["feature_cache_dir"]) / f"{stem}.parquet"
            event_path = Path(item["event_cache_dir"]) / f"{stem}.parquet"
            try:
                feature = pd.read_parquet(
                    feature_path, columns=["date", READY_COLUMN, *item["feature_columns"]]
                )
                event = pd.read_parquet(
                    event_path, columns=["date", READY_COLUMN, *item["targets"]]
                )
                if len(feature) != len(event) or not feature["date"].equals(event["date"]):
                    raise ValueError("Feature and target timestamps differ.")
                dates = pd.to_datetime(event["date"], utc=True, errors="raise")
                ready = event[READY_COLUMN].fillna(False).astype(bool)
                ready &= event[item["targets"]].notna().all(axis=1)
                start = pd.Timestamp("2026-08-20T00:00:00Z")
                train_start = start - pd.Timedelta(
                    days=int(
                        json.loads(Path(item["config_path"]).read_text(encoding="utf-8"))["freqai"][
                            "train_period_days"
                        ]
                    )
                )
                train_cutoff = start - pd.Timedelta(hours=MAX_HORIZON_HOURS)
                end = pd.Timestamp(
                    "2026-08-22T00:00:00Z"
                    if manifest["technical_smoke_not_evidence"]
                    else "2026-09-20T00:00:00Z"
                )
                training_rows = int((ready & dates.ge(train_start) & dates.lt(train_cutoff)).sum())
                prediction_rows = int((ready & dates.ge(start) & dates.lt(end)).sum())
                minimum_prediction = 24 if manifest["technical_smoke_not_evidence"] else 120
                if training_rows < 1000 or prediction_rows < minimum_prediction:
                    raise ValueError(
                        f"Insufficient rows: training={training_rows}, prediction={prediction_rows}"
                    )
            except (OSError, KeyError, ValueError) as exc:
                problems.append(f"{item['profile_id']} {pair}: {exc}")
    free_gib = shutil.disk_usage(Path(manifest["storage"]["bulky_artifact_dir"])).free / 1024**3
    if free_gib < 20:
        problems.append(f"Only {free_gib:.1f} GiB free on D.")
    return {
        "created_at_utc": g0.utc_now(),
        "passed": not problems,
        "problems": problems,
        "dependency_check": dependency,
        "free_gib": round(free_gib, 2),
        "profile_count": len(manifest["commands"]),
        "model_threads_per_profile": 1,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--phase", choices=("prepare", "run"), default="prepare")
    parser.add_argument("--worker", type=Path, default=DEFAULT_WORKER)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--profile-workers", type=int, default=2)
    args = parser.parse_args(argv)
    run_id = args.run_id or (SMOKE_RUN_ID if args.smoke else DEFAULT_RUN_ID)
    manifest, path = build_manifest(
        run_id=run_id,
        smoke=args.smoke,
        worker=args.worker,
        base_config=args.base_config,
        workers=args.profile_workers,
    )
    check = preflight(manifest, worker=args.worker)
    check_path = path.parent / "fresh_state_freqai_preflight.json"
    g0.atomic_write_json(check, check_path)
    print(json.dumps({"run_id": run_id, "preflight": check, "manifest": str(path)}, indent=2))
    if not check["passed"]:
        return 2
    if args.phase == "run":
        return g7f.run_manifest(manifest, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

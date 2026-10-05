"""Build Generation 14's outcome-blind 8h-to-long-anchor cache."""

from __future__ import annotations

# Bound numerical pools before pandas/pyarrow imports.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation13_cache as g13c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation14_freeze as g14z,
)


DEFAULT_RUN_ID = "g14_combined_context_20260822a"
RECORD_ROOT = g14z.OUTPUT_ROOT / "generation14_shared" / DEFAULT_RUN_ID
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation14_shared"
    / DEFAULT_RUN_ID
)
MAX_WORKERS = 4
EIGHT_HOUR_BLOCKS = tuple(
    g14z.ready(g14z.EIGHT_HOUR, variant)
    for variant in ("current", "stale", "shuffled")
)
EIGHT_HOUR_COLUMNS = tuple(
    column
    for variant in ("current", "stale", "shuffled")
    for column in g14z.eight_hour_columns(variant=variant)
)
READINESS_COLUMNS = tuple(f"ready__{block}" for block in EIGHT_HOUR_BLOCKS)


@dataclass(frozen=True)
class PairTask:
    cohort: str
    pair: str
    source: dict[str, Any]
    artifact_root: str
    overwrite: bool


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def normalized_dates(frame: DataFrame, *, label: str) -> DataFrame:
    if "date" not in frame:
        raise ValueError(f"{label} has no date column.")
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"], utc=True, errors="coerce")
    if output["date"].isna().any():
        raise ValueError(f"{label} contains invalid dates.")
    if output["date"].duplicated().any():
        raise ValueError(f"{label} contains duplicate dates.")
    return output.sort_values("date").reset_index(drop=True)


def merge_eight_hour_features(long_features: DataFrame, mtf_features: DataFrame) -> DataFrame:
    left = normalized_dates(long_features, label="long feature cache")
    right = normalized_dates(mtf_features, label="8h feature cache")
    missing = sorted(set(EIGHT_HOUR_COLUMNS).difference(right.columns))
    if missing:
        raise ValueError(f"8h feature cache is missing columns: {missing}")
    collisions = sorted(set(EIGHT_HOUR_COLUMNS).intersection(left.columns))
    if collisions:
        raise ValueError(f"Long feature cache already contains G14 columns: {collisions}")
    return left.merge(
        right[["date", *EIGHT_HOUR_COLUMNS]],
        on="date",
        how="left",
        validate="one_to_one",
    )


def merge_readiness(left_frame: DataFrame, mtf_events: DataFrame) -> DataFrame:
    left = normalized_dates(left_frame, label="long event/evaluation cache")
    right = normalized_dates(mtf_events, label="8h readiness cache")
    missing = sorted(set(READINESS_COLUMNS).difference(right.columns))
    if missing:
        raise ValueError(f"8h readiness cache is missing columns: {missing}")
    collisions = sorted(set(READINESS_COLUMNS).intersection(left.columns))
    if collisions:
        raise ValueError(f"Long cache already contains G14 readiness: {collisions}")
    output = left.merge(
        right[["date", *READINESS_COLUMNS]],
        on="date",
        how="left",
        validate="one_to_one",
    )
    for column in READINESS_COLUMNS:
        output[column] = output[column].fillna(False).astype(bool)
    return output


def validate_source(path: Path, expected_sha256: str) -> None:
    if not path.is_file() or g0.sha256_file(path) != expected_sha256:
        raise ValueError(f"Frozen Generation 13 cache source changed: {path}")


def pair_paths(task: PairTask) -> dict[str, Path]:
    stem = g0.pair_file_stem(task.pair)
    root = Path(task.artifact_root) / task.cohort
    return {
        "feature": root / "long_feature_cache" / f"{stem}.parquet",
        "readiness": root / "long_readiness_support" / f"{stem}.parquet",
        "event": root / "long_event_cache" / f"{stem}.parquet",
        "evaluation": root / "long_evaluation_cache" / f"{stem}.parquet",
    }


def build_outcome_blind_pair(task: PairTask) -> dict[str, Any]:
    source = task.source
    for prefix in ("long_feature", "mtf_feature", "long_event", "mtf_event"):
        validate_source(Path(source[f"{prefix}_path"]), source[f"{prefix}_sha256"])
    paths = pair_paths(task)
    if (
        paths["feature"].is_file()
        and paths["readiness"].is_file()
        and not task.overwrite
    ):
        return {
            "status": "existing_outcome_blind_support",
            "cohort": task.cohort,
            "pair": task.pair,
            "feature": artifact(paths["feature"]),
            "readiness": artifact(paths["readiness"]),
        }
    long_features = pd.read_parquet(source["long_feature_path"])
    mtf_features = pd.read_parquet(
        source["mtf_feature_path"], columns=["date", *EIGHT_HOUR_COLUMNS]
    )
    features = merge_eight_hour_features(long_features, mtf_features)
    long_clock = pd.read_parquet(source["long_event_path"], columns=["date"])
    mtf_readiness = pd.read_parquet(
        source["mtf_event_path"], columns=["date", *READINESS_COLUMNS]
    )
    readiness = merge_readiness(long_clock, mtf_readiness)
    g0.atomic_write_parquet(features, paths["feature"])
    g0.atomic_write_parquet(readiness, paths["readiness"])
    missing_feature_rows = int(features[list(EIGHT_HOUR_COLUMNS)].isna().all(axis=1).sum())
    return {
        "status": "built_outcome_blind_support",
        "cohort": task.cohort,
        "pair": task.pair,
        "feature_rows": len(features),
        "readiness_rows": len(readiness),
        "missing_eight_hour_feature_rows": missing_feature_rows,
        "feature": artifact(paths["feature"]),
        "readiness": artifact(paths["readiness"]),
    }


def materialize_targets(task: PairTask) -> dict[str, Any]:
    source = task.source
    for prefix in ("long_event", "long_evaluation", "mtf_event"):
        validate_source(Path(source[f"{prefix}_path"]), source[f"{prefix}_sha256"])
    paths = pair_paths(task)
    if not paths["readiness"].is_file():
        raise FileNotFoundError(paths["readiness"])
    if paths["event"].is_file() and paths["evaluation"].is_file() and not task.overwrite:
        return {
            "status": "existing_materialized_targets",
            "cohort": task.cohort,
            "pair": task.pair,
            "event": artifact(paths["event"]),
            "evaluation": artifact(paths["evaluation"]),
        }
    mtf_readiness = pd.read_parquet(
        source["mtf_event_path"], columns=["date", *READINESS_COLUMNS]
    )
    events = merge_readiness(pd.read_parquet(source["long_event_path"]), mtf_readiness)
    evaluation = merge_readiness(
        pd.read_parquet(source["long_evaluation_path"]), mtf_readiness
    )
    g0.atomic_write_parquet(events, paths["event"])
    g0.atomic_write_parquet(evaluation, paths["evaluation"])
    return {
        "status": "materialized_targets_after_support_freeze",
        "cohort": task.cohort,
        "pair": task.pair,
        "event_rows": len(events),
        "evaluation_rows": len(evaluation),
        "event": artifact(paths["event"]),
        "evaluation": artifact(paths["evaluation"]),
    }


def load_sources(cohort: str) -> tuple[dict[str, Any], Path]:
    frozen = json.loads(g14z.FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation14_outcomes":
        raise ValueError("Generation 14 was not frozen before cache construction.")
    source_path = g13c.RECORD_ROOT / f"{cohort}_manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("status") != "completed_generation13_cache":
        raise ValueError(f"Generation 13 cache is incomplete for {cohort}.")
    expected = frozen["source_contracts"][f"generation13_{cohort}_cache"]
    if g0.sha256_file(source_path) != expected["sha256"]:
        raise ValueError(f"Generation 13 {cohort} manifest changed after G14 freeze.")
    return source, source_path


def tasks_for(cohort: str, *, overwrite: bool) -> tuple[list[PairTask], Path]:
    source, source_path = load_sources(cohort)
    tasks = [
        PairTask(
            cohort=cohort,
            pair=str(item["pair"]),
            source=dict(item),
            artifact_root=str(ARTIFACT_ROOT),
            overwrite=overwrite,
        )
        for item in source["inventory"]
    ]
    return tasks, source_path


def run_tasks(
    tasks: Sequence[PairTask],
    function: Any,
    *,
    workers: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, MAX_WORKERS))) as pool:
        futures = {pool.submit(function, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                output.append(future.result())
            except Exception as exc:
                output.append(
                    {
                        "status": "failed",
                        "cohort": task.cohort,
                        "pair": task.pair,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return sorted(output, key=lambda item: (item["cohort"], item["pair"]))


def build_cohort(cohort: str, *, workers: int, overwrite: bool) -> dict[str, Any]:
    tasks, source_path = tasks_for(cohort, overwrite=overwrite)
    support = run_tasks(tasks, build_outcome_blind_pair, workers=workers)
    failures = [item for item in support if item["status"] == "failed"]
    if failures:
        raise RuntimeError(f"Generation 14 support cache failed: {failures}")

    support_freeze_path = RECORD_ROOT / f"{cohort}_outcome_blind_support_freeze.json"
    support_freeze = {
        "schema_version": 1,
        "generation": 14,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_outcome_blind_before_long_targets_opened",
        "cohort": cohort,
        "reaction_outcome_columns_read": False,
        "feature_columns": list(EIGHT_HOUR_COLUMNS),
        "readiness_columns": list(READINESS_COLUMNS),
        "source_contracts": {
            "generation14_freeze": artifact(g14z.FREEZE_PATH),
            "generation13_cache": artifact(source_path),
        },
        "support": support,
    }
    g0.atomic_write_json(support_freeze, support_freeze_path)

    materialized = run_tasks(tasks, materialize_targets, workers=workers)
    failures = [item for item in materialized if item["status"] == "failed"]
    if failures:
        raise RuntimeError(f"Generation 14 target materialization failed: {failures}")

    inventory: list[dict[str, Any]] = []
    support_by_pair = {item["pair"]: item for item in support}
    target_by_pair = {item["pair"]: item for item in materialized}
    for task in tasks:
        paths = pair_paths(task)
        inventory.append(
            {
                "pair": task.pair,
                "feature_path": str(paths["feature"].resolve()),
                "feature_sha256": g0.sha256_file(paths["feature"]),
                "event_path": str(paths["event"].resolve()),
                "event_sha256": g0.sha256_file(paths["event"]),
                "evaluation_path": str(paths["evaluation"].resolve()),
                "evaluation_sha256": g0.sha256_file(paths["evaluation"]),
                "readiness_path": str(paths["readiness"].resolve()),
                "readiness_sha256": g0.sha256_file(paths["readiness"]),
                "support_status": support_by_pair[task.pair]["status"],
                "target_status": target_by_pair[task.pair]["status"],
            }
        )
    manifest = {
        "schema_version": 1,
        "generation": 14,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation14_combined_context_cache",
        "cohort": cohort,
        "pairs": [task.pair for task in tasks],
        "pair_count": len(tasks),
        "eight_hour_feature_columns": list(EIGHT_HOUR_COLUMNS),
        "readiness_columns": list(READINESS_COLUMNS),
        "outcome_blind_support_frozen_before_targets": True,
        "inventory": inventory,
        "source_contracts": {
            "generation14_freeze": artifact(g14z.FREEZE_PATH),
            "generation13_cache": artifact(source_path),
            "outcome_blind_support_freeze": artifact(support_freeze_path),
        },
    }
    manifest_path = RECORD_ROOT / f"{cohort}_manifest.json"
    g0.atomic_write_json(manifest, manifest_path)
    return {**manifest, "manifest_path": str(manifest_path.resolve())}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Generation 14 shared cache.")
    parser.add_argument("--cohort", choices=("all", *g14z.COHORTS), default="all")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    cohorts = g14z.COHORTS if args.cohort == "all" else (args.cohort,)
    results = [
        build_cohort(
            cohort,
            workers=max(1, min(args.workers, MAX_WORKERS)),
            overwrite=args.overwrite,
        )
        for cohort in cohorts
    ]
    print(
        json.dumps(
            [
                {
                    "status": item["status"],
                    "cohort": item["cohort"],
                    "pairs": item["pair_count"],
                    "manifest": item["manifest_path"],
                }
                for item in results
            ],
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

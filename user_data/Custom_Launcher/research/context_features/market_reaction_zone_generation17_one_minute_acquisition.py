"""Prepare, audit, and acquire Generation 17's frozen one-minute replay data."""

from __future__ import annotations

# Bind native pools before pandas/numpy imports.
# ruff: noqa: E402
import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import threading
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
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_one_minute_replay as g3m,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation4_one_minute_download as g4d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation4_one_minute_repair as g4r,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation13_cache as g13c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_freeze as g17z,
)


MAXIMUM_DOWNLOAD_WORKERS = 1
FEATURE_WARMUP_HOURS = 24
FREEZE_ROOT = g17z.OUTPUT_ROOT / g17z.DEFAULT_BATCH_ID
FREEZE_RECORD_NAME = "g17_one_minute_replay_freeze_record.json"
COVERAGE_PREFIX = "g17"
ANALYSIS_SAMPLE_NAME = "g17_one_minute_analysis_sample.csv"
ACQUISITION_NAME = "g17_one_minute_finalized_acquisition_intervals.csv"


def artifact(path: Path) -> dict[str, Any]:
    return g17z.artifact(path)


def load_parent_freeze() -> tuple[dict[str, Any], Path]:
    frozen = json.loads(g17z.FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation17_outcomes":
        raise ValueError("Generation 17 parent freeze is invalid.")
    sample = Path(frozen["one_minute_sample"]["sample"]["path"])
    if g0.sha256_file(sample) != frozen["one_minute_sample"]["sample"]["sha256"]:
        raise ValueError("Generation 17 frozen one-minute sample changed.")
    return frozen, sample


def source_metadata() -> DataFrame:
    frames: list[DataFrame] = []
    columns = [
        "cohort",
        "pair",
        "event_time",
        "level_family",
        "level_name",
        "source_timeframe",
        "representation",
        "control",
        "level_column",
        "approach_state",
        "source_open",
        "source_available_at",
    ]
    for cohort in ("normal", "meme"):
        manifest = json.loads(
            (g13c.RECORD_ROOT / f"{cohort}_manifest.json").read_text(encoding="utf-8")
        )
        for item in manifest["inventory"]:
            frame = pd.read_parquet(
                item["source_path"],
                columns=columns,
                filters=[("control", "==", "actual")],
            )
            frames.append(frame.drop(columns="control"))
    source = pd.concat(frames, ignore_index=True)
    source["event_time"] = pd.to_datetime(source["event_time"], utc=True, errors="raise")
    keys = [
        "cohort",
        "pair",
        "event_time",
        "level_family",
        "level_name",
        "source_timeframe",
        "representation",
    ]
    return source.sort_values(keys).drop_duplicates(keys)


def prepare_analysis_sample(frozen_path: Path) -> DataFrame:
    sample = pd.read_csv(frozen_path)
    sample["event_time"] = pd.to_datetime(sample["event_time"], utc=True, errors="raise")
    metadata = source_metadata()
    keys = [
        "cohort",
        "pair",
        "event_time",
        "level_family",
        "level_name",
        "source_timeframe",
        "representation",
    ]
    sample = sample.merge(metadata, on=keys, how="left", validate="one_to_one")
    required = ["level_column", "approach_state", "source_open", "source_available_at"]
    if sample[required].isna().any().any():
        raise ValueError("Generation 17 sample lacks exact causal source metadata.")
    sample = sample.sort_values(["cohort", "pair", "density_regime"]).reset_index(
        drop=True
    )
    sample["sample_selection_order"] = sample.index + 1
    sample["sample_stratum"] = sample["selection_stratum"]
    sample["analysis_period"] = sample["period"]
    sample["cluster_causal_anchor_timeframe"] = sample["anchor_source_timeframe"]
    sample["selected_for_one_minute_replay"] = True
    return sample


def finalized_intervals(sample: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for episode in sample.itertuples(index=False):
        event_time = pd.Timestamp(episode.event_time)
        pre_hours = int(episode.pre_context_hours)
        post_hours = int(episode.post_context_hours)
        visible_start = event_time - pd.Timedelta(hours=pre_hours)
        visible_end = event_time + pd.Timedelta(hours=post_hours + 1)
        rows.append(
            {
                "episode_ids": str(episode.episode_id),
                "cohorts": str(episode.cohort),
                "pair": str(episode.pair),
                "parent_contact_hours_utc": event_time.isoformat(),
                "causal_anchor_timeframes": str(episode.anchor_source_timeframe),
                "maximum_visible_pre_hours": pre_hours,
                "maximum_visible_post_hours": post_hours,
                "feature_warmup_hours": FEATURE_WARMUP_HOURS,
                "visible_start_utc": visible_start,
                "visible_end_exclusive_utc": visible_end,
                "interval_start_utc": visible_start
                - pd.Timedelta(hours=FEATURE_WARMUP_HOURS),
                "interval_end_exclusive_utc": visible_end,
                "exchange": "binance",
                "market_type": "futures",
                "timeframe": "1m",
                "data_format": "feather",
            }
        )
    merged = g3m.merge_overlapping_acquisition_intervals(DataFrame.from_records(rows))
    return merged.sort_values(["pair", "interval_start_utc"]).reset_index(drop=True)


def prepare_freeze_record() -> dict[str, Any]:
    frozen, frozen_sample = load_parent_freeze()
    FREEZE_ROOT.mkdir(parents=True, exist_ok=True)
    record_path = FREEZE_ROOT / FREEZE_RECORD_NAME
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "frozen_before_one_minute_paths":
            raise ValueError(f"Invalid Generation 17 replay freeze: {record_path}")
        if "request_sha256" not in record:
            request = {
                "parent_freeze_sha256": g0.sha256_file(g17z.FREEZE_PATH),
                "frozen_sample_sha256": g0.sha256_file(frozen_sample),
                "feature_warmup_hours": FEATURE_WARMUP_HOURS,
            }
            record["request_sha256"] = hashlib.sha256(
                json.dumps(request, sort_keys=True).encode()
            ).hexdigest()
            g0.atomic_write_json(record, record_path)
        return record
    sample = prepare_analysis_sample(frozen_sample)
    intervals = finalized_intervals(sample)
    sample_path = FREEZE_ROOT / ANALYSIS_SAMPLE_NAME
    acquisition_path = FREEZE_ROOT / ACQUISITION_NAME
    g0.atomic_write_csv(sample, sample_path)
    g0.atomic_write_csv(intervals, acquisition_path)
    record = {
        "schema_version": 1,
        "generation": 17,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_one_minute_paths",
        "run_id": g17z.DEFAULT_BATCH_ID,
        "request_sha256": hashlib.sha256(
            json.dumps(
                {
                    "parent_freeze_sha256": g0.sha256_file(g17z.FREEZE_PATH),
                    "frozen_sample_sha256": g0.sha256_file(frozen_sample),
                    "feature_warmup_hours": FEATURE_WARMUP_HOURS,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "selection_used_future_reaction_or_direction": False,
        "episodes": len(sample),
        "cohort_pair_units": len(sample[["cohort", "pair"]].drop_duplicates()),
        "density_balance": sample["density_regime"].value_counts().to_dict(),
        "visible_window_rule": {
            "1h": {"pre_hours": 12, "post_hours": 12},
            "4h": {"pre_hours": 48, "post_hours": 48},
            "8h": {"pre_hours": 96, "post_hours": 96},
        },
        "feature_warmup_hours": FEATURE_WARMUP_HOURS,
        "artifacts": {
            "sample_csv": artifact(sample_path),
            "acquisition_intervals_csv": artifact(acquisition_path),
        },
        "source_contracts": {
            "generation17_parent_freeze": artifact(g17z.FREEZE_PATH),
            "frozen_parent_sample": artifact(frozen_sample),
        },
        "parent_branch": next(
            item
            for item in frozen["branches"]
            if item["branch_id"] == "g17f_bounded_one_minute_expansion"
        ),
    }
    g0.atomic_write_json(record, record_path)
    return record


def run_coverage_audit() -> bool:
    prepare_freeze_record()
    try:
        g3m.audit_one_minute_coverage(
            g17z.DEFAULT_BATCH_ID,
            report_root=g17z.OUTPUT_ROOT,
            record_filename=FREEZE_RECORD_NAME,
            coverage_prefix=COVERAGE_PREFIX,
            update_freeze_record=False,
            acquisition_note=(
                "Generation 17 audited only the density-balanced, source-timeframe-scaled "
                "intervals frozen before any one-minute path was opened."
            ),
        )
    except RuntimeError:
        return False
    return True


def acquire_missing(*, max_workers: int) -> int:
    if not 1 <= max_workers <= MAXIMUM_DOWNLOAD_WORKERS:
        raise ValueError(f"max_workers must be 1-{MAXIMUM_DOWNLOAD_WORKERS}.")
    prepare_freeze_record()
    if run_coverage_audit():
        print("All frozen Generation 17 one-minute intervals already pass coverage.")
        return 0
    acquisition_path = FREEZE_ROOT / ACQUISITION_NAME
    coverage_path = FREEZE_ROOT / f"{COVERAGE_PREFIX}_one_minute_coverage_audit.csv"
    acquisition = pd.read_csv(acquisition_path)
    coverage = pd.read_csv(coverage_path)
    failed = g4d.validated_failed_download_rows(
        g4d.failed_coverage_rows(coverage), acquisition
    )
    record_path = FREEZE_ROOT / "g17_one_minute_download_record.json"
    record = (
        json.loads(record_path.read_text(encoding="utf-8"))
        if record_path.is_file()
        else {"schema_version": 1, "run_id": g17z.DEFAULT_BATCH_ID, "sessions": []}
    )
    if record.get("status") == "running":
        raise RuntimeError(f"A Generation 17 download is already running: {record_path}")
    session: dict[str, Any] = {
        "session_id": g0.utc_now().replace(":", "").replace("-", ""),
        "status": "running",
        "controller_pid": int(os.getpid()),
        "controller_python": str(g4d.CONTROLLER_PYTHON.resolve()),
        "started_at_utc": g0.utc_now(),
        "completed_at_utc": None,
        "max_pair_workers": max_workers,
        "jobs": [],
    }
    session_root = (
        g4r.ISOLATED_ROOT
        / "generation17"
        / g17z.DEFAULT_BATCH_ID
        / session["session_id"]
    )
    g4r.ensure_isolated_path(session_root)
    session_root.mkdir(parents=True, exist_ok=False)
    for number, row in enumerate(failed.to_dict(orient="records"), start=1):
        isolated = session_root / f"job_{number:03d}"
        target = Path(str(row["source_file"])).resolve()
        g4r.validate_main_target(target)
        command = g4r.isolated_download_args(str(row["pair"]), str(row["timerange"]), isolated)
        session["jobs"].append(
            {
                "job_id": f"{number:03d}",
                "pair": str(row["pair"]),
                "timerange": str(row["timerange"]),
                "interval_start_utc": str(row["interval_start_utc"]),
                "interval_end_exclusive_utc": str(row["interval_end_exclusive_utc"]),
                "expected_rows": int(row["expected_rows"]),
                "target_file": str(target),
                "isolated_datadir": str(isolated.resolve()),
                "command": subprocess.list2cmdline(command),
                "log_path": str(
                    (FREEZE_ROOT / "download_logs" / f"{g4d.safe_pair_name(row['pair'])}.log")
                    .resolve()
                ),
                "status": "queued",
                "pid": None,
                "return_code": None,
            }
        )
    record["status"] = "running"
    record["sessions"].append(session)
    g0.atomic_write_json(record, record_path)
    lock = threading.Lock()

    def update(job: dict[str, Any], **values: Any) -> None:
        with lock:
            job.update(values)
            g0.atomic_write_json(record, record_path)

    def run_pair(jobs: list[dict[str, Any]]) -> None:
        for job in jobs:
            isolated = Path(job["isolated_datadir"])
            isolated.mkdir(parents=True, exist_ok=False)
            command = g4r.isolated_download_args(
                str(job["pair"]), str(job["timerange"]), isolated
            )
            log_path = Path(job["log_path"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[{g0.utc_now()}] START {subprocess.list2cmdline(command)}\n")
                log.flush()
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                update(job, status="running", pid=int(process.pid))
                return_code = int(process.wait())
                if return_code:
                    update(job, status="download_failed", return_code=return_code)
                    continue
                merge = g4r.merge_isolated_download(job)
                update(
                    job,
                    status=(
                        "completed"
                        if merge["merge_status"] == "merged"
                        else "source_incomplete"
                    ),
                    return_code=return_code,
                    merge=merge,
                )
                g4r.cleanup_isolated_job(isolated)

    by_pair: dict[str, list[dict[str, Any]]] = {}
    for job in session["jobs"]:
        by_pair.setdefault(str(job["pair"]), []).append(job)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_pair, jobs) for jobs in by_pair.values()]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    failed_jobs = [job for job in session["jobs"] if job["status"] != "completed"]
    session.update(
        {
            "status": "completed" if not failed_jobs else "completed_with_failures",
            "completed_at_utc": g0.utc_now(),
            "completed_commands": len(session["jobs"]) - len(failed_jobs),
            "failed_commands": len(failed_jobs),
        }
    )
    record["status"] = session["status"]
    g0.atomic_write_json(record, record_path)
    g4r.cleanup_empty_session_root(session_root)
    if failed_jobs:
        return 1
    return 0 if run_coverage_audit() else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    prepare_freeze_record()
    if args.prepare_only:
        return 0
    if args.audit_only:
        return 0 if run_coverage_audit() else 2
    return acquire_missing(max_workers=int(args.max_workers))


if __name__ == "__main__":
    raise SystemExit(main())

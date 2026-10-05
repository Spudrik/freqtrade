from __future__ import annotations

# Acquire only the missing edges of the selectively expanded G4A context windows.
# The expansion set was frozen before primary reaction outcomes were opened.
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
import concurrent.futures
import json
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, DatetimeIndex


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation4_one_minute_repair as repair,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    atomic_write_json,
    sha256_file,
    utc_now,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation4_one_minute_breadth import (  # noqa: E501
    REPORT_ROOT,
    validate_g4_branch,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation4_one_minute_confirmation import (  # noqa: E501
    validated_method_artifact,
    validated_method_record,
)


SCHEMA_VERSION = 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download and atomically merge only missing one-minute runs from the frozen "
            "G4A selective context expansion."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args(argv)
    validate_g4_branch()
    return download_expansion_gaps(args.run_id, max_workers=int(args.max_workers))


def download_expansion_gaps(run_id: str, *, max_workers: int) -> int:  # noqa: C901
    if not 1 <= max_workers <= repair.MAXIMUM_DOWNLOAD_WORKERS:
        raise ValueError(f"max_workers must be between 1 and {repair.MAXIMUM_DOWNLOAD_WORKERS}")
    run_dir = REPORT_ROOT / run_id
    method_path = run_dir / "g4a_method_freeze_record.json"
    method = validated_method_record(method_path)
    acquisition_path = validated_method_artifact(method, "selective_expansion_intervals_csv")
    acquisition = pd.read_csv(acquisition_path)
    jobs_source = missing_interval_runs(acquisition)
    if jobs_source.empty:
        print("All frozen selective expansion intervals are already complete.")
        return 0

    record_path = run_dir / "g4a_expansion_download_record.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") == "running":
            raise RuntimeError(f"An expansion download is already running: {record_path}")
    else:
        record = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "sessions": []}

    started = utc_now()
    session_id = re.sub(r"[^0-9A-Za-z]+", "", started) + "expansion"
    session_root = repair.ISOLATED_ROOT / run_id / session_id
    repair.ensure_isolated_path(session_root)
    session_root.mkdir(parents=True, exist_ok=False)
    jobs: list[dict[str, Any]] = []
    for index, row in enumerate(jobs_source.to_dict(orient="records"), start=1):
        job_id = f"{session_id}_{index:03d}"
        isolated_datadir = session_root / job_id
        pair = str(row["pair"])
        command = repair.isolated_download_args(pair, str(row["timerange"]), isolated_datadir)
        jobs.append(
            {
                "job_id": job_id,
                "pair": pair,
                "timerange": str(row["timerange"]),
                "interval_start_utc": str(row["interval_start_utc"]),
                "interval_end_exclusive_utc": str(row["interval_end_exclusive_utc"]),
                "expected_rows": int(row["expected_rows"]),
                "target_file": str(Path(str(row["target_file"])).resolve()),
                "isolated_datadir": str(isolated_datadir.resolve()),
                "command": subprocess.list2cmdline(command),
                "log_path": str(
                    (
                        run_dir / "expansion_download_logs" / f"{repair.safe_pair_name(pair)}.log"
                    ).resolve()
                ),
                "status": "queued",
                "pid": None,
                "return_code": None,
                "started_at_utc": None,
                "completed_at_utc": None,
            }
        )
    session: dict[str, Any] = {
        "session_id": session_id,
        "status": "running",
        "controller_pid": int(os.getpid()),
        "controller_python": str(repair.CONTROLLER_PYTHON.resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "method_record_sha256": sha256_file(method_path),
        "acquisition_sha256": sha256_file(acquisition_path),
        "started_at_utc": started,
        "completed_at_utc": None,
        "max_pair_workers": max_workers,
        "pair_count": int(jobs_source["pair"].nunique()),
        "command_count": len(jobs),
        "missing_rows_requested": int(jobs_source["expected_rows"].sum()),
        "isolated_root": str(session_root.resolve()),
        "jobs": jobs,
    }
    record.update({"status": "running", "latest_session_id": session_id})
    record["sessions"].append(session)
    atomic_write_json(record, record_path)
    lock = threading.Lock()

    def update_job(job: dict[str, Any], **updates: Any) -> None:
        with lock:
            job.update(updates)
            atomic_write_json(record, record_path)

    jobs_by_pair: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        jobs_by_pair.setdefault(str(job["pair"]), []).append(job)

    def run_pair_jobs(pair_jobs: list[dict[str, Any]]) -> None:
        for job in pair_jobs:
            log_path = Path(str(job["log_path"]))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            isolated_datadir = Path(str(job["isolated_datadir"]))
            isolated_datadir.mkdir(parents=True, exist_ok=False)
            command = repair.isolated_download_args(
                str(job["pair"]), str(job["timerange"]), isolated_datadir
            )
            with log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(f"\n[{utc_now()}] START {subprocess.list2cmdline(command)}\n")
                log_handle.flush()
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=REPO_ROOT,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    update_job(
                        job,
                        status="running",
                        pid=int(process.pid),
                        started_at_utc=utc_now(),
                    )
                    return_code = int(process.wait())
                    if return_code != 0:
                        update_job(
                            job,
                            status="download_failed",
                            return_code=return_code,
                            completed_at_utc=utc_now(),
                        )
                    else:
                        merge = repair.merge_isolated_download(job)
                        update_job(
                            job,
                            status=(
                                "completed"
                                if merge["merge_status"] == "merged"
                                else "source_incomplete"
                            ),
                            return_code=return_code,
                            completed_at_utc=utc_now(),
                            merge=merge,
                        )
                        repair.cleanup_isolated_job(isolated_datadir)
                except Exception as exc:
                    update_job(
                        job,
                        status="repair_failed",
                        completed_at_utc=utc_now(),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                log_handle.write(
                    f"[{utc_now()}] END status={job['status']} return_code={job['return_code']}\n"
                )
                log_handle.flush()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_pair_jobs, pair_jobs) for pair_jobs in jobs_by_pair.values()]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    failed = [job for job in jobs if job["status"] != "completed"]
    unavailable = [job for job in failed if job["status"] == "source_incomplete"]
    session.update(
        {
            "status": "completed" if not failed else "completed_with_attrition",
            "completed_at_utc": utc_now(),
            "completed_commands": len(jobs) - len(failed),
            "source_incomplete_commands": len(unavailable),
            "failed_commands": len(failed) - len(unavailable),
        }
    )
    record["status"] = session["status"]
    atomic_write_json(record, record_path)
    repair.cleanup_empty_session_root(session_root)
    print(
        json.dumps(
            {
                "status": session["status"],
                "commands": len(jobs),
                "missing_rows_requested": session["missing_rows_requested"],
                "source_incomplete": len(unavailable),
                "failed": len(failed) - len(unavailable),
            },
            indent=2,
        )
    )
    return 0 if not failed else 1


def missing_interval_runs(acquisition: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    cached_path: Path | None = None
    cached_dates = DatetimeIndex([])
    for interval in acquisition.sort_values(
        ["expected_local_file", "interval_start_utc"]
    ).itertuples():
        target = Path(str(interval.expected_local_file)).resolve()
        repair.validate_main_target(target)
        if target != cached_path:
            cached_path = target
            cached_dates = (
                DatetimeIndex(
                    pd.to_datetime(pd.read_feather(target, columns=["date"])["date"], utc=True)
                )
                if target.is_file()
                else DatetimeIndex([])
            )
        start = pd.Timestamp(interval.interval_start_utc)
        end = pd.Timestamp(interval.interval_end_exclusive_utc)
        expected = pd.date_range(
            start=start,
            end=end - pd.Timedelta(minutes=1),
            freq="1min",
        )
        present = cached_dates[(cached_dates >= start) & (cached_dates < end)]
        missing = expected.difference(present)
        for run_start, run_end in contiguous_minute_runs(missing):
            end_exclusive = run_end + pd.Timedelta(minutes=1)
            rows.append(
                {
                    "episode_ids": str(interval.episode_ids),
                    "pair": str(interval.pair),
                    "interval_start_utc": run_start,
                    "interval_end_exclusive_utc": end_exclusive,
                    "expected_rows": int((end_exclusive - run_start).total_seconds() // 60),
                    "timerange": f"{int(run_start.timestamp())}-{int(end_exclusive.timestamp())}",
                    "target_file": str(target),
                }
            )
    return DataFrame(rows)


def contiguous_minute_runs(missing: DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if missing.empty:
        return []
    ordered = missing.sort_values()
    differences = pd.Series(ordered).diff().iloc[1:]
    split = np.flatnonzero(differences.ne(pd.Timedelta(minutes=1)).to_numpy())
    starts = np.r_[0, split + 1]
    ends = np.r_[split, len(ordered) - 1]
    return [
        (pd.Timestamp(ordered[start]), pd.Timestamp(ordered[end]))
        for start, end in zip(starts, ends, strict=True)
    ]


if __name__ == "__main__":
    raise SystemExit(main())

"""Acquire only Generation 21's frozen missing one-minute replay intervals."""

from __future__ import annotations

# Bind numerical pools before pandas imports.
# ruff: noqa: E402
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
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
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation4_one_minute_download as g4d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation4_one_minute_repair as g4r,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation20_one_minute_acquisition as g20a,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation21_one_minute_replay as g21e,
)


ANALYSIS_PATH = Path(__file__).resolve()
DEFAULT_RUN_ID = "g21_one_minute_acquisition_20260827a"
MAXIMUM_WORKERS = 2
RECORD_ROOT = g21e.RECORD_ROOT / "acquisition"
ACQUISITION_CSV = RECORD_ROOT / "g21_one_minute_acquisition_intervals.csv"
ACQUISITION_FREEZE = RECORD_ROOT / "g21_one_minute_acquisition_freeze.json"
DOWNLOAD_RECORD = RECORD_ROOT / "g21_one_minute_download_record.json"


def artifact(path: Path) -> dict[str, Any]:
    return g21e.artifact(path)


def prepare_acquisition(*, overwrite: bool = False) -> dict[str, Any]:
    g21e.freeze_selection(overwrite=False)
    if ACQUISITION_FREEZE.is_file() and not overwrite:
        frozen = json.loads(ACQUISITION_FREEZE.read_text(encoding="utf-8"))
        if frozen.get("status") != "frozen_before_generation21_one_minute_downloads":
            raise ValueError("Invalid Generation 21 acquisition freeze.")
        return frozen
    sample = pd.read_csv(g21e.SELECTION_CSV)
    sample["event_time"] = pd.to_datetime(sample["event_time"], utc=True)
    coverage, _ = g21e.coverage_audit(sample)
    missing = coverage.loc[coverage["status"].ne("passed")].copy()
    intervals = g20a.merge_intervals(missing)
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(intervals, ACQUISITION_CSV)
    frozen = {
        "schema_version": 1,
        "generation": 21,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation21_one_minute_downloads",
        "run_id": DEFAULT_RUN_ID,
        "missing_episode_intervals": len(missing),
        "merged_download_intervals": len(intervals),
        "pairs": sorted(intervals["pair"].unique()) if len(intervals) else [],
        "one_pair_downloaded_sequentially": True,
        "maximum_parallel_pairs": MAXIMUM_WORKERS,
        "artifact": artifact(ACQUISITION_CSV),
        "source_contracts": {
            "selection_freeze": artifact(g21e.SELECTION_RECORD),
            "selection_sample": artifact(g21e.SELECTION_CSV),
            "acquisition_script": artifact(ANALYSIS_PATH),
        },
    }
    g0.atomic_write_json(frozen, ACQUISITION_FREEZE)
    return frozen


def acquire(*, max_workers: int) -> int:  # noqa: C901 - audited job lifecycle
    if not 1 <= max_workers <= MAXIMUM_WORKERS:
        raise ValueError(f"max_workers must be 1-{MAXIMUM_WORKERS}.")
    frozen = prepare_acquisition(overwrite=False)
    if g0.sha256_file(ANALYSIS_PATH) != frozen["source_contracts"]["acquisition_script"][
        "sha256"
    ]:
        raise ValueError("Generation 21 acquisition script changed after interval freeze.")
    intervals = pd.read_csv(ACQUISITION_CSV)
    if intervals.empty:
        print("All frozen Generation 21 one-minute intervals already pass coverage.")
        return 0
    if DOWNLOAD_RECORD.is_file():
        existing = json.loads(DOWNLOAD_RECORD.read_text(encoding="utf-8"))
        if existing.get("status") == "running":
            raise RuntimeError(f"A Generation 21 download is already running: {DOWNLOAD_RECORD}")
    session_id = g0.utc_now().replace(":", "").replace("-", "")
    session_root = g4r.ISOLATED_ROOT / "generation21" / DEFAULT_RUN_ID / session_id
    g4r.ensure_isolated_path(session_root)
    session_root.mkdir(parents=True, exist_ok=False)
    jobs: list[dict[str, Any]] = []
    for number, row in enumerate(intervals.to_dict(orient="records"), start=1):
        isolated = session_root / f"job_{number:03d}"
        target = Path(str(row["target_file"])).resolve()
        g4r.validate_main_target(target)
        command = g4r.isolated_download_args(
            str(row["pair"]), str(row["timerange"]), isolated
        )
        jobs.append(
            {
                "job_id": f"{number:03d}",
                "pair": str(row["pair"]),
                "episode_ids": str(row["episode_ids"]),
                "timerange": str(row["timerange"]),
                "interval_start_utc": str(row["interval_start_utc"]),
                "interval_end_exclusive_utc": str(row["interval_end_exclusive_utc"]),
                "expected_rows": int(row["expected_rows"]),
                "target_file": str(target),
                "isolated_datadir": str(isolated.resolve()),
                "command": subprocess.list2cmdline(command),
                "log_path": str(
                    (
                        RECORD_ROOT
                        / "download_logs"
                        / f"{number:03d}_{g4d.safe_pair_name(str(row['pair']))}.log"
                    ).resolve()
                ),
                "status": "queued",
                "pid": None,
                "return_code": None,
            }
        )
    record: dict[str, Any] = {
        "schema_version": 1,
        "generation": 21,
        "run_id": DEFAULT_RUN_ID,
        "status": "running",
        "session_id": session_id,
        "controller_pid": os.getpid(),
        "started_at_utc": g0.utc_now(),
        "completed_at_utc": None,
        "maximum_parallel_pairs": max_workers,
        "jobs": jobs,
        "source_contracts": {"acquisition_freeze": artifact(ACQUISITION_FREEZE)},
    }
    g0.atomic_write_json(record, DOWNLOAD_RECORD)
    lock = threading.Lock()

    def update(job: dict[str, Any], **values: Any) -> None:
        with lock:
            job.update(values)
            g0.atomic_write_json(record, DOWNLOAD_RECORD)

    def run_pair(pair_jobs: list[dict[str, Any]]) -> None:
        for job in pair_jobs:
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
                update(job, status="running", pid=process.pid)
                return_code = int(process.wait())
                if return_code:
                    update(job, status="download_failed", return_code=return_code)
                    continue
                merge = g4r.merge_isolated_download(job)
                update(
                    job,
                    status=(
                        "completed" if merge["merge_status"] == "merged" else "source_incomplete"
                    ),
                    return_code=return_code,
                    merge=merge,
                )
                g4r.cleanup_isolated_job(isolated)

    by_pair: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        by_pair.setdefault(str(job["pair"]), []).append(job)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_pair, pair_jobs) for pair_jobs in by_pair.values()]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    failures = [job for job in jobs if job["status"] != "completed"]
    record.update(
        {
            "status": "completed" if not failures else "completed_with_failures",
            "completed_at_utc": g0.utc_now(),
            "completed_jobs": len(jobs) - len(failures),
            "failed_jobs": len(failures),
        }
    )
    g0.atomic_write_json(record, DOWNLOAD_RECORD)
    g4r.cleanup_empty_session_root(session_root)
    if failures:
        return 1
    sample = pd.read_csv(g21e.SELECTION_CSV)
    sample["event_time"] = pd.to_datetime(sample["event_time"], utc=True)
    _, passed = g21e.coverage_audit(sample)
    return 0 if passed else 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=MAXIMUM_WORKERS)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    frozen = prepare_acquisition(overwrite=args.overwrite)
    if args.prepare_only:
        print(json.dumps(frozen, indent=2))
        return 0
    return acquire(max_workers=args.max_workers)


if __name__ == "__main__":
    raise SystemExit(main())

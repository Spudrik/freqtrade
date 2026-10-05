"""Audit and acquire only the one-minute intervals frozen for Generation 16."""

from __future__ import annotations

# Bind native pools before pandas/numpy imports.
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
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pandas as pd


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
    market_reaction_zone_generation16_freeze as g16z,
)


MAXIMUM_DOWNLOAD_WORKERS = 1
COVERAGE_PREFIX = "g16"


def run_coverage_audit(run_id: str) -> bool:
    try:
        g3m.audit_one_minute_coverage(
            run_id,
            report_root=g16z.FREEZE_ROOT,
            record_filename=g16z.ONE_MINUTE_FREEZE_RECORD_NAME,
            coverage_prefix=COVERAGE_PREFIX,
            update_freeze_record=False,
            acquisition_note=(
                "Generation 16 audited only the exact anchor-scaled intervals frozen "
                "before any one-minute path was opened."
            ),
        )
    except RuntimeError:
        return False
    return True


def frozen_paths(run_id: str) -> tuple[Path, Path, Path, Path]:
    frozen = json.loads(g16z.FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation16_outcomes":
        raise ValueError("Generation 16 batch is not frozen.")
    if frozen.get("run_id") != run_id:
        raise ValueError(
            f"Run id {run_id!r} differs from frozen run {frozen.get('run_id')!r}."
        )
    run_dir = g16z.FREEZE_ROOT / run_id
    freeze_record = frozen["artifacts"]["one_minute_freeze_record"]
    freeze_record_path = Path(freeze_record["path"])
    if (
        not freeze_record_path.is_file()
        or g0.sha256_file(freeze_record_path) != freeze_record["sha256"]
    ):
        raise ValueError("The Generation 16 one-minute freeze record changed.")
    acquisition = frozen["artifacts"]["one_minute_acquisition_intervals"]
    acquisition_path = Path(acquisition["path"])
    if (
        not acquisition_path.is_file()
        or g0.sha256_file(acquisition_path) != acquisition["sha256"]
    ):
        raise ValueError("The frozen Generation 16 acquisition intervals changed.")
    return (
        run_dir,
        acquisition_path,
        run_dir / f"{COVERAGE_PREFIX}_one_minute_coverage_audit.csv",
        run_dir / f"{COVERAGE_PREFIX}_one_minute_coverage_record.json",
    )


def acquire_missing(run_id: str, *, max_workers: int) -> int:  # noqa: C901
    if not 1 <= max_workers <= MAXIMUM_DOWNLOAD_WORKERS:
        raise ValueError(
            f"max_workers must be between 1 and {MAXIMUM_DOWNLOAD_WORKERS}."
        )
    run_dir, acquisition_path, coverage_path, _ = frozen_paths(run_id)
    if run_coverage_audit(run_id):
        print("All frozen Generation 16 one-minute intervals already pass coverage.")
        return 0
    acquisition = pd.read_csv(acquisition_path)
    coverage = pd.read_csv(coverage_path)
    failed = g4d.validated_failed_download_rows(
        g4d.failed_coverage_rows(coverage), acquisition
    )
    record_path = run_dir / "g16_one_minute_download_record.json"
    if record_path.is_file():
        record: dict[str, Any] = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") == "running":
            raise RuntimeError(f"A Generation 16 download is already running: {record_path}")
    else:
        record = {"schema_version": 1, "run_id": run_id, "sessions": []}
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
    session_root = g4r.ISOLATED_ROOT / "generation16" / run_id / session["session_id"]
    g4r.ensure_isolated_path(session_root)
    session_root.mkdir(parents=True, exist_ok=False)
    for number, row in enumerate(failed.to_dict(orient="records"), start=1):
        isolated_datadir = session_root / f"job_{number:03d}"
        target_file = Path(str(row["source_file"])).resolve()
        g4r.validate_main_target(target_file)
        command = g4r.isolated_download_args(
            str(row["pair"]), str(row["timerange"]), isolated_datadir
        )
        session["jobs"].append(
            {
                "job_id": f"{number:03d}",
                "pair": str(row["pair"]),
                "timerange": str(row["timerange"]),
                "interval_start_utc": str(row["interval_start_utc"]),
                "interval_end_exclusive_utc": str(row["interval_end_exclusive_utc"]),
                "expected_rows": int(row["expected_rows"]),
                "target_file": str(target_file),
                "isolated_datadir": str(isolated_datadir.resolve()),
                "command": subprocess.list2cmdline(command),
                "log_path": str(
                    (run_dir / "download_logs" / f"{g4d.safe_pair_name(row['pair'])}.log").resolve()
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

    def run_pair(pair_jobs: list[dict[str, Any]]) -> None:
        for job in pair_jobs:
            isolated_datadir = Path(job["isolated_datadir"])
            isolated_datadir.mkdir(parents=True, exist_ok=False)
            command = g4r.isolated_download_args(
                str(job["pair"]), str(job["timerange"]), isolated_datadir
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
                g4r.cleanup_isolated_job(isolated_datadir)

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
    if not run_coverage_audit(run_id):
        return 2
    print(
        json.dumps(
            {
                "status": "completed_and_coverage_passed",
                "downloaded_intervals": len(session["jobs"]),
                "download_record": str(record_path.resolve()),
            },
            indent=2,
        )
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=g16z.DEFAULT_RUN_ID)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.audit_only:
        frozen_paths(args.run_id)
        return 0 if run_coverage_audit(args.run_id) else 2
    return acquire_missing(args.run_id, max_workers=int(args.max_workers))


if __name__ == "__main__":
    raise SystemExit(main())

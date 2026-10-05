from __future__ import annotations

# This runner only acquires intervals already frozen by G4A. It cannot select events,
# inspect reaction outcomes, or broaden the requested market/time ranges.
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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    atomic_write_json,
    sha256_file,
    utc_now,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation4_one_minute_breadth import (  # noqa: E501
    REPORT_ROOT,
    validate_g4_branch,
    validated_artifact_path,
)


SCHEMA_VERSION = 1
MAXIMUM_DOWNLOAD_WORKERS = 4
CONTROLLER_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
APPROVED_DATA_DIRECTORY = REPO_ROOT / "user_data" / "data" / "binance"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download only missing intervals from an immutable G4A one-minute "
            "coverage audit, with writes serialized within each pair."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args(argv)
    validate_g4_branch()
    return download_missing_coverage(args.run_id, max_workers=int(args.max_workers))


def download_missing_coverage(run_id: str, *, max_workers: int) -> int:  # noqa: C901
    if not 1 <= max_workers <= MAXIMUM_DOWNLOAD_WORKERS:
        raise ValueError(
            f"max_workers must be between 1 and {MAXIMUM_DOWNLOAD_WORKERS}"
        )
    if not CONTROLLER_PYTHON.is_file():
        raise FileNotFoundError(f"Configured controller Python is missing: {CONTROLLER_PYTHON}")

    run_dir = REPORT_ROOT / run_id
    freeze_path = run_dir / "g4a_freeze_record.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    acquisition_path = validated_artifact_path(
        freeze.get("artifacts", {}), "acquisition_intervals_csv"
    )
    coverage_artifact = freeze.get("one_minute_coverage", {}).get("coverage_csv", {})
    coverage_path = Path(str(coverage_artifact.get("path", "")))
    if not coverage_path.is_file():
        raise FileNotFoundError(
            "Run the G4A --coverage-audit before requesting missing downloads"
        )
    if sha256_file(coverage_path) != str(coverage_artifact.get("sha256", "")):
        raise ValueError("The frozen one-minute coverage audit changed after it was written")

    acquisition = pd.read_csv(acquisition_path)
    coverage = pd.read_csv(coverage_path)
    failed = failed_coverage_rows(coverage)
    if failed.empty:
        print("All frozen G4A one-minute intervals already pass coverage.")
        return 0
    failed = validated_failed_download_rows(failed, acquisition)

    record_path = run_dir / "g4a_download_run_record.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") == "running":
            raise RuntimeError(f"A G4A download session is already running: {record_path}")
    else:
        record = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "sessions": [],
        }

    session_started = utc_now()
    session_id = re.sub(r"[^0-9A-Za-z]+", "", session_started)
    jobs: list[dict[str, Any]] = []
    for index, row in enumerate(failed.to_dict(orient="records"), start=1):
        pair = str(row["pair"])
        log_path = run_dir / "download_logs" / f"{safe_pair_name(pair)}.log"
        command = download_command_args(row)
        jobs.append(
            {
                "job_id": f"{session_id}_{index:03d}",
                "pair": pair,
                "timerange": str(row["timerange"]),
                "command": subprocess.list2cmdline(command),
                "log_path": str(log_path.resolve()),
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
        "controller_python": str(CONTROLLER_PYTHON.resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "started_at_utc": session_started,
        "completed_at_utc": None,
        "max_pair_workers": max_workers,
        "pair_count": int(failed["pair"].nunique()),
        "command_count": len(jobs),
        "coverage_audit_sha256": sha256_file(coverage_path),
        "jobs": jobs,
    }
    record["status"] = "running"
    record["latest_session_id"] = session_id
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
            command = download_command_args(
                {"pair": job["pair"], "timerange": job["timerange"]}
            )
            with log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(
                    f"\n[{utc_now()}] START {subprocess.list2cmdline(command)}\n"
                )
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
                    update_job(
                        job,
                        status="completed" if return_code == 0 else "failed",
                        return_code=return_code,
                        completed_at_utc=utc_now(),
                    )
                except Exception as exc:
                    update_job(
                        job,
                        status="failed_to_launch",
                        completed_at_utc=utc_now(),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                log_handle.write(
                    f"[{utc_now()}] END status={job['status']} "
                    f"return_code={job['return_code']}\n"
                )
                log_handle.flush()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_pair_jobs, pair_jobs)
            for pair_jobs in jobs_by_pair.values()
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    failed_jobs = [job for job in jobs if job["status"] != "completed"]
    session.update(
        {
            "status": "completed" if not failed_jobs else "completed_with_failures",
            "completed_at_utc": utc_now(),
            "completed_commands": len(jobs) - len(failed_jobs),
            "failed_commands": len(failed_jobs),
        }
    )
    record["status"] = session["status"]
    atomic_write_json(record, record_path)
    print(
        json.dumps(
            {
                "status": session["status"],
                "commands": len(jobs),
                "failed_commands": len(failed_jobs),
                "record": str(record_path.resolve()),
            },
            indent=2,
        )
    )
    return 0 if not failed_jobs else 1


def failed_coverage_rows(coverage: DataFrame) -> DataFrame:
    passed = coverage["passed"].astype(str).str.lower().eq("true")
    return coverage.loc[~passed].copy()


def validated_failed_download_rows(
    failed: DataFrame, acquisition: DataFrame
) -> DataFrame:
    expected = acquisition[["pair", "timerange", "download_command"]].copy()
    if expected["download_command"].duplicated().any():
        raise ValueError("Frozen acquisition commands are not unique")
    enriched = failed.merge(
        expected.rename(columns={"pair": "frozen_pair"}),
        on="download_command",
        how="left",
        validate="one_to_one",
    )
    if enriched["timerange"].isna().any():
        raise ValueError("Coverage audit contains a command outside the frozen acquisition set")
    if not enriched["pair"].astype(str).eq(enriched["frozen_pair"].astype(str)).all():
        raise ValueError("Coverage audit pair differs from its frozen acquisition command")
    return enriched.drop(columns="frozen_pair")


def download_command_args(row: Mapping[str, Any]) -> list[str]:
    pair = str(row["pair"])
    timerange = str(row["timerange"])
    if not re.fullmatch(r"\d{10}-\d{10}", timerange):
        raise ValueError(f"Invalid frozen Unix-second timerange: {timerange}")
    if not re.fullmatch(r"[0-9A-Z]+/USDT:USDT", pair):
        raise ValueError(f"Invalid frozen Binance futures pair: {pair}")
    return [
        str(CONTROLLER_PYTHON.resolve()),
        "-m",
        "freqtrade",
        "download-data",
        "--exchange",
        "binance",
        "--trading-mode",
        "futures",
        "--candle-types",
        "futures",
        "--data-format-ohlcv",
        "feather",
        "--datadir",
        str(APPROVED_DATA_DIRECTORY.resolve()),
        "-p",
        pair,
        "-t",
        "1m",
        "--timerange",
        timerange,
    ]


def safe_pair_name(pair: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", pair).strip("_")


if __name__ == "__main__":
    raise SystemExit(main())

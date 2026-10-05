from __future__ import annotations

# Freqtrade only appends or prepends an existing OHLCV file. This repair runner uses
# isolated exact-range downloads so internal gaps can be verified and merged without
# downloading years of unrelated one-minute history.
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
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
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
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation4_one_minute_download import (  # noqa: E501
    CONTROLLER_PYTHON,
    MAXIMUM_DOWNLOAD_WORKERS,
    failed_coverage_rows,
    safe_pair_name,
    validated_failed_download_rows,
)


SCHEMA_VERSION = 1
MAIN_DATA_ROOT = REPO_ROOT / "user_data" / "data" / "binance"
ISOLATED_ROOT = Path(
    r"D:\FreqTradeStuffLargeData\research_temp\g4a_one_minute_interval_repair"
)
OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Repair frozen G4A one-minute gaps through isolated exact-range downloads "
            "and verified atomic merges."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args(argv)
    validate_g4_branch()
    return repair_missing_coverage(args.run_id, max_workers=int(args.max_workers))


def repair_missing_coverage(run_id: str, *, max_workers: int) -> int:  # noqa: C901
    if not 1 <= max_workers <= MAXIMUM_DOWNLOAD_WORKERS:
        raise ValueError(
            f"max_workers must be between 1 and {MAXIMUM_DOWNLOAD_WORKERS}"
        )
    if not CONTROLLER_PYTHON.is_file():
        raise FileNotFoundError(f"Configured controller Python is missing: {CONTROLLER_PYTHON}")
    if not ISOLATED_ROOT.drive or not ISOLATED_ROOT.drive.upper().startswith("D:"):
        raise ValueError(f"Isolated repair root must be on D drive: {ISOLATED_ROOT}")

    run_dir = REPORT_ROOT / run_id
    freeze_path = run_dir / "g4a_freeze_record.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    acquisition_path = validated_artifact_path(
        freeze.get("artifacts", {}), "acquisition_intervals_csv"
    )
    coverage_artifact = freeze.get("one_minute_coverage", {}).get("coverage_csv", {})
    coverage_path = Path(str(coverage_artifact.get("path", "")))
    if not coverage_path.is_file():
        raise FileNotFoundError("Run the G4A coverage audit before isolated repair")
    if sha256_file(coverage_path) != str(coverage_artifact.get("sha256", "")):
        raise ValueError("The one-minute coverage audit changed after it was recorded")

    acquisition = pd.read_csv(acquisition_path)
    coverage = pd.read_csv(coverage_path)
    failed = failed_coverage_rows(coverage)
    if failed.empty:
        print("All frozen G4A one-minute intervals already pass coverage.")
        return 0
    failed = validated_failed_download_rows(failed, acquisition)

    record_path = run_dir / "g4a_isolated_repair_run_record.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") == "running":
            raise RuntimeError(f"An isolated repair session is already running: {record_path}")
    else:
        record = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "sessions": []}

    started = utc_now()
    session_id = re.sub(r"[^0-9A-Za-z]+", "", started)
    session_root = ISOLATED_ROOT / run_id / session_id
    ensure_isolated_path(session_root)
    session_root.mkdir(parents=True, exist_ok=False)

    jobs: list[dict[str, Any]] = []
    for index, row in enumerate(failed.to_dict(orient="records"), start=1):
        job_id = f"{session_id}_{index:03d}"
        pair = str(row["pair"])
        isolated_datadir = session_root / job_id
        target_file = Path(str(row["source_file"])).resolve()
        validate_main_target(target_file)
        command = isolated_download_args(pair, str(row["timerange"]), isolated_datadir)
        jobs.append(
            {
                "job_id": job_id,
                "pair": pair,
                "timerange": str(row["timerange"]),
                "interval_start_utc": str(row["interval_start_utc"]),
                "interval_end_exclusive_utc": str(row["interval_end_exclusive_utc"]),
                "expected_rows": int(row["expected_rows"]),
                "target_file": str(target_file),
                "isolated_datadir": str(isolated_datadir.resolve()),
                "command": subprocess.list2cmdline(command),
                "log_path": str(
                    (run_dir / "repair_logs" / f"{safe_pair_name(pair)}.log").resolve()
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
        "controller_python": str(CONTROLLER_PYTHON.resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "started_at_utc": started,
        "completed_at_utc": None,
        "max_pair_workers": max_workers,
        "pair_count": int(failed["pair"].nunique()),
        "command_count": len(jobs),
        "isolated_root": str(session_root.resolve()),
        "coverage_audit_sha256": sha256_file(coverage_path),
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
            command = isolated_download_args(
                str(job["pair"]), str(job["timerange"]), isolated_datadir
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
                    if return_code != 0:
                        update_job(
                            job,
                            status="download_failed",
                            return_code=return_code,
                            completed_at_utc=utc_now(),
                        )
                    else:
                        merge = merge_isolated_download(job)
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
                        cleanup_isolated_job(isolated_datadir)
                except Exception as exc:
                    update_job(
                        job,
                        status="repair_failed",
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
    unavailable_jobs = [job for job in jobs if job["status"] == "source_incomplete"]
    session.update(
        {
            "status": "completed" if not failed_jobs else "completed_with_attrition",
            "completed_at_utc": utc_now(),
            "completed_commands": len(jobs) - len(failed_jobs),
            "source_incomplete_commands": len(unavailable_jobs),
            "failed_commands": len(failed_jobs) - len(unavailable_jobs),
        }
    )
    record["status"] = session["status"]
    atomic_write_json(record, record_path)
    cleanup_empty_session_root(session_root)
    print(
        json.dumps(
            {
                "status": session["status"],
                "commands": len(jobs),
                "source_incomplete": len(unavailable_jobs),
                "failed": len(failed_jobs) - len(unavailable_jobs),
                "record": str(record_path.resolve()),
            },
            indent=2,
        )
    )
    return 0 if not failed_jobs else 1


def isolated_download_args(pair: str, timerange: str, datadir: Path) -> list[str]:
    if not re.fullmatch(r"\d{10}-\d{10}", timerange):
        raise ValueError(f"Invalid frozen Unix-second timerange: {timerange}")
    if not re.fullmatch(r"[0-9A-Z]+/USDT:USDT", pair):
        raise ValueError(f"Invalid frozen Binance futures pair: {pair}")
    ensure_isolated_path(datadir)
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
        str(datadir.resolve()),
        "-p",
        pair,
        "-t",
        "1m",
        "--timerange",
        timerange,
    ]


def merge_isolated_download(job: dict[str, Any]) -> dict[str, Any]:
    isolated_datadir = Path(str(job["isolated_datadir"]))
    target = Path(str(job["target_file"]))
    validate_main_target(target)
    source = isolated_datadir / "futures" / target.name
    if not source.is_file():
        return {
            "merge_status": "source_incomplete",
            "reason": "isolated_source_file_missing",
            "actual_rows": 0,
            "missing_rows": int(job["expected_rows"]),
        }
    incoming = normalized_ohlcv(pd.read_feather(source))
    start = pd.Timestamp(str(job["interval_start_utc"]))
    end = pd.Timestamp(str(job["interval_end_exclusive_utc"]))
    selected = incoming.loc[
        (incoming["date"] >= start) & (incoming["date"] < end)
    ].copy()
    quality = interval_quality(selected, start, end, int(job["expected_rows"]))
    source_details = {
        "isolated_source_sha256": sha256_file(source),
        "isolated_source_rows": len(incoming),
        **quality,
    }
    if not quality["complete"]:
        return {
            "merge_status": "source_incomplete",
            "reason": "binance_did_not_return_complete_frozen_interval",
            **source_details,
        }

    existing = (
        normalized_ohlcv(pd.read_feather(target))
        if target.is_file()
        else DataFrame(columns=OHLCV_COLUMNS)
    )
    before_rows = len(existing)
    before_sha256 = sha256_file(target) if target.is_file() else None
    overlap_conflicts = count_overlap_conflicts(existing, selected)
    if overlap_conflicts:
        raise ValueError(f"Incoming candles conflict with {overlap_conflicts} stored minutes")
    merged = (
        pd.concat([existing, selected], ignore_index=True)
        .sort_values("date")
        .drop_duplicates("date", keep="first")
        .reset_index(drop=True)
    )
    verify = interval_quality(
        merged.loc[(merged["date"] >= start) & (merged["date"] < end)],
        start,
        end,
        int(job["expected_rows"]),
    )
    if not verify["complete"]:
        raise ValueError("Merged frame does not contain the complete frozen interval")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_target = target.with_name(f".{target.name}.{job['job_id']}.tmp.feather")
    merged.loc[:, OHLCV_COLUMNS].to_feather(
        temporary_target, compression_level=9, compression="lz4"
    )
    stored = normalized_ohlcv(pd.read_feather(temporary_target))
    if len(stored) != len(merged):
        temporary_target.unlink(missing_ok=True)
        raise ValueError("Atomic merge candidate changed row count after Feather round-trip")
    if target.is_file() and sha256_file(target) != before_sha256:
        temporary_target.unlink(missing_ok=True)
        raise RuntimeError("Target one-minute file changed during isolated merge")
    temporary_target.replace(target)
    return {
        "merge_status": "merged",
        "reason": "complete_exact_interval_atomically_merged",
        **source_details,
        "overlap_conflicts": overlap_conflicts,
        "target_rows_before": before_rows,
        "target_rows_after": len(merged),
        "target_rows_added": len(merged) - before_rows,
        "target_sha256_before": before_sha256,
        "target_sha256_after": sha256_file(target),
    }


def normalized_ohlcv(frame: DataFrame) -> DataFrame:
    missing = sorted(set(OHLCV_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"OHLCV frame is missing columns: {missing}")
    result = frame.loc[:, OHLCV_COLUMNS].copy()
    result["date"] = pd.to_datetime(result["date"], utc=True, errors="coerce")
    for column in OHLCV_COLUMNS[1:]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.sort_values("date").reset_index(drop=True)


def interval_quality(
    frame: DataFrame, start: pd.Timestamp, end: pd.Timestamp, expected_rows: int
) -> dict[str, Any]:
    ordered = normalized_ohlcv(frame)
    differences = ordered["date"].diff().dropna()
    numeric = ordered[OHLCV_COLUMNS[1:]]
    invalid = (
        ordered["date"].isna()
        | ~np.isfinite(numeric).all(axis=1)
        | numeric["volume"].lt(0.0)
        | numeric["high"].lt(numeric["low"])
        | numeric["high"].lt(numeric[["open", "close"]].max(axis=1))
        | numeric["low"].gt(numeric[["open", "close"]].min(axis=1))
    )
    first = ordered["date"].min() if len(ordered) else pd.NaT
    last = ordered["date"].max() if len(ordered) else pd.NaT
    duplicate_minutes = int(ordered["date"].duplicated().sum())
    gaps = int(differences.ne(pd.Timedelta(minutes=1)).sum())
    complete = bool(
        len(ordered) == expected_rows
        and duplicate_minutes == 0
        and gaps == 0
        and int(invalid.sum()) == 0
        and first == start
        and last == end - pd.Timedelta(minutes=1)
    )
    return {
        "complete": complete,
        "actual_rows": len(ordered),
        "missing_rows": expected_rows - len(ordered),
        "duplicate_minutes": duplicate_minutes,
        "non_one_minute_gaps": gaps,
        "invalid_ohlcv_rows": int(invalid.sum()),
        "first_actual_utc": None if pd.isna(first) else first.isoformat(),
        "last_actual_utc": None if pd.isna(last) else last.isoformat(),
    }


def count_overlap_conflicts(existing: DataFrame, incoming: DataFrame) -> int:
    overlap = existing.merge(incoming, on="date", how="inner", suffixes=("_old", "_new"))
    if overlap.empty:
        return 0
    conflict = np.zeros(len(overlap), dtype=bool)
    for column in OHLCV_COLUMNS[1:]:
        conflict |= ~np.isclose(
            overlap[f"{column}_old"].to_numpy(dtype=float),
            overlap[f"{column}_new"].to_numpy(dtype=float),
            rtol=1e-12,
            atol=0.0,
            equal_nan=True,
        )
    return int(conflict.sum())


def validate_main_target(path: Path) -> None:
    root = MAIN_DATA_ROOT.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Target is outside approved Binance data root: {resolved}")


def ensure_isolated_path(path: Path) -> None:
    root = ISOLATED_ROOT.resolve()
    resolved = path.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"Unsafe isolated repair path: {resolved}")


def cleanup_isolated_job(path: Path) -> None:
    ensure_isolated_path(path)
    if path.is_dir():
        shutil.rmtree(path)


def cleanup_empty_session_root(path: Path) -> None:
    ensure_isolated_path(path)
    if path.is_dir() and not any(path.iterdir()):
        path.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())

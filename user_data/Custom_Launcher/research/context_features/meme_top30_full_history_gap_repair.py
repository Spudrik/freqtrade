"""Audit and repair exact internal gaps in the frozen top-30 meme OHLCV pool."""

from __future__ import annotations

# Bind numerical pools before pandas imports.
# ruff: noqa: E402
import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
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

import numpy as np
import pandas as pd
import psutil
from pandas import DataFrame


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


ANALYSIS_PATH = Path(__file__).resolve()
COHORT_ROOT = (
    REPO_ROOT
    / "user_data"
    / "research_news_data"
    / "context_features"
    / "market_reaction_zones"
    / "meme_cohort"
)
MANIFEST_PATH = COHORT_ROOT / "meme_top30_data_pool_selection_20260906.json"
AUDIT_BEFORE_PATH = COHORT_ROOT / "meme_top30_integrity_audit_before_20260907.csv"
INTERVALS_PATH = COHORT_ROOT / "meme_top30_gap_intervals_20260907.csv"
FREEZE_PATH = COHORT_ROOT / "meme_top30_gap_repair_freeze_20260907.json"
RUN_RECORD_PATH = COHORT_ROOT / "meme_top30_gap_repair_run_20260907.json"
AUDIT_AFTER_PATH = COHORT_ROOT / "meme_top30_integrity_audit_after_20260907.csv"
ISOLATED_ROOT = Path(r"D:\FreqTradeStuffLargeData\research_temp\meme_top30_full_history_gap_repair")
MAXIMUM_WORKERS = 4
OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
TIMEFRAME_DELTAS = {
    "1m": pd.Timedelta(minutes=1),
    "3m": pd.Timedelta(minutes=3),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "2h": pd.Timedelta(hours=2),
    "4h": pd.Timedelta(hours=4),
    "6h": pd.Timedelta(hours=6),
    "8h": pd.Timedelta(hours=8),
    "12h": pd.Timedelta(hours=12),
    "1d": pd.Timedelta(days=1),
    "3d": pd.Timedelta(days=3),
    "1w": pd.Timedelta(weeks=1),
}


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def load_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    members = manifest.get("members", [])
    timeframes = manifest.get("requested_ohlcv", {}).get("timeframes", [])
    if len(members) != 30 or len(timeframes) != 15:
        raise ValueError("The frozen meme pool is not 30 pairs by 15 timeframes")
    return manifest


def stored_timeframe(timeframe: str) -> str:
    return "1Mo" if timeframe == "1M" else timeframe


def data_path(pair: str, timeframe: str, datadir: Path | None = None) -> Path:
    name = g0.ohlcv_path(pair, stored_timeframe(timeframe)).name
    if datadir is None:
        return g0.ohlcv_path(pair, stored_timeframe(timeframe))
    return datadir / "futures" / name


def invalid_ohlcv_rows(frame: DataFrame) -> int:
    numeric = frame[OHLCV_COLUMNS[1:]]
    invalid = (
        frame["date"].isna()
        | ~np.isfinite(numeric).all(axis=1)
        | numeric["volume"].lt(0.0)
        | numeric[["open", "high", "low", "close"]].le(0.0).any(axis=1)
        | numeric["high"].lt(numeric["low"])
        | numeric["high"].lt(numeric[["open", "close"]].max(axis=1))
        | numeric["low"].gt(numeric[["open", "close"]].min(axis=1))
    )
    return int(invalid.sum())


def _next_month(value: pd.Timestamp) -> pd.Timestamp:
    naive = value.tz_convert("UTC").tz_localize(None)
    result = naive.to_period("M").to_timestamp(how="start") + pd.offsets.MonthBegin(1)
    return result.tz_localize("UTC")


def audit_frame(
    pair: str, timeframe: str, path: Path, frame: DataFrame
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    original_dates = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    out_of_order = int(original_dates.diff().dropna().lt(pd.Timedelta(0)).sum())
    ordered = g4r.normalized_ohlcv(frame)
    duplicates = int(ordered["date"].duplicated().sum())
    unique = ordered.drop_duplicates("date", keep="first").reset_index(drop=True)
    gaps: list[dict[str, Any]] = []
    if timeframe == "1M":
        ordinals = unique["date"].dt.year * 12 + unique["date"].dt.month
        for index in range(1, len(unique)):
            missing = int(ordinals.iloc[index] - ordinals.iloc[index - 1] - 1)
            if missing <= 0:
                continue
            start = _next_month(pd.Timestamp(unique.iloc[index - 1]["date"]))
            end = pd.Timestamp(unique.iloc[index]["date"])
            gaps.append(_gap_row(pair, timeframe, path, start, end, missing))
    else:
        delta = TIMEFRAME_DELTAS[timeframe]
        differences = unique["date"].diff()
        for index in differences.index[differences.gt(delta)]:
            units = float(differences.iloc[index] / delta)
            rounded = round(units)
            if not np.isclose(units, rounded, rtol=0.0, atol=1e-9):
                raise ValueError(
                    f"Non-aligned {timeframe} spacing in {path} at row {index}: {units}"
                )
            missing = int(rounded - 1)
            start = pd.Timestamp(unique.iloc[index - 1]["date"]) + delta
            end = pd.Timestamp(unique.iloc[index]["date"])
            gaps.append(_gap_row(pair, timeframe, path, start, end, missing))
    invalid = invalid_ohlcv_rows(unique)
    summary = {
        "pair": pair,
        "timeframe": timeframe,
        "path": str(path.resolve()),
        "sha256": g0.sha256_file(path),
        "rows": len(ordered),
        "first_candle_utc": unique["date"].min(),
        "last_candle_utc": unique["date"].max(),
        "duplicate_timestamps": duplicates,
        "out_of_order_timestamps": out_of_order,
        "gap_events": len(gaps),
        "missing_candle_intervals": sum(row["expected_rows"] for row in gaps),
        "invalid_ohlcv_rows": invalid,
        "status": (
            "passed"
            if duplicates == 0 and out_of_order == 0 and not gaps and invalid == 0
            else "failed_integrity"
        ),
    }
    return summary, gaps


def _gap_row(
    pair: str,
    timeframe: str,
    path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    expected_rows: int,
) -> dict[str, Any]:
    return {
        "pair": pair,
        "timeframe": timeframe,
        "interval_start_utc": start,
        "interval_end_exclusive_utc": end,
        "expected_rows": expected_rows,
        "timerange": f"{int(start.timestamp())}-{int(end.timestamp())}",
        "target_file": str(path.resolve()),
        "target_sha256_at_freeze": g0.sha256_file(path),
    }


def audit_pool(
    manifest: Mapping[str, Any], *, datadir: Path | None = None
) -> tuple[DataFrame, DataFrame]:
    summaries: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for member in manifest["members"]:
        pair = str(member["freqtrade_pair"])
        for timeframe in manifest["requested_ohlcv"]["timeframes"]:
            timeframe = str(timeframe)
            path = data_path(pair, timeframe, datadir)
            if not path.is_file():
                summaries.append(
                    {
                        "pair": pair,
                        "timeframe": timeframe,
                        "path": str(path.resolve()),
                        "sha256": "",
                        "rows": 0,
                        "first_candle_utc": pd.NaT,
                        "last_candle_utc": pd.NaT,
                        "duplicate_timestamps": 0,
                        "out_of_order_timestamps": 0,
                        "gap_events": 0,
                        "missing_candle_intervals": 0,
                        "invalid_ohlcv_rows": 0,
                        "status": "missing_file",
                    }
                )
                continue
            summary, file_gaps = audit_frame(pair, timeframe, path, pd.read_feather(path))
            summaries.append(summary)
            gaps.extend(file_gaps)
    return DataFrame.from_records(summaries), DataFrame.from_records(gaps)


def audit_summary(audit: DataFrame) -> dict[str, Any]:
    failed = audit["status"].ne("passed")
    return {
        "pairs": int(audit["pair"].nunique()),
        "timeframes_per_pair": int(audit["timeframe"].nunique()),
        "audited_files": len(audit),
        "readable_nonempty_files": int(audit["rows"].gt(0).sum()),
        "total_rows": int(audit["rows"].sum()),
        "total_bytes": int(
            sum(Path(path).stat().st_size for path in audit.loc[audit["rows"].gt(0), "path"])
        ),
        "failed_files": int(failed.sum()),
        "timestamp_gap_events": int(audit["gap_events"].sum()),
        "missing_candle_intervals": int(audit["missing_candle_intervals"].sum()),
        "duplicate_timestamps": int(audit["duplicate_timestamps"].sum()),
        "out_of_order_timestamps": int(audit["out_of_order_timestamps"].sum()),
        "invalid_ohlcv_rows": int(audit["invalid_ohlcv_rows"].sum()),
    }


def update_manifest_integrity(
    manifest: dict[str, Any],
    *,
    status: str,
    audit: DataFrame,
    audit_path: Path,
) -> None:
    summary = audit_summary(audit)
    manifest["download"]["status"] = status
    manifest["download"]["integrity_check"] = {
        "status": "passed" if summary["failed_files"] == 0 else "failed",
        "audited_utc": g0.utc_now(),
        **summary,
        "total_gib": round(summary["total_bytes"] / 1024**3, 3),
        "audit": artifact(audit_path),
        "checks": [
            "required columns and readable non-empty file",
            "timestamps sorted and unique",
            "every adjacent fixed candle or calendar month is present",
            "no null non-finite or invalid OHLCV values",
        ],
    }
    g0.atomic_write_json(manifest, MANIFEST_PATH)


def prepare(*, overwrite: bool = False) -> dict[str, Any]:
    if FREEZE_PATH.is_file() and not overwrite:
        frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        if frozen.get("status") != "frozen_before_top30_gap_repair":
            raise ValueError("Invalid top-30 gap-repair freeze")
        return frozen
    manifest = load_manifest()
    audit, gaps = audit_pool(manifest)
    if len(audit) != 450:
        raise ValueError(f"Expected 450 meme OHLCV files, audited {len(audit)}")
    if audit["status"].eq("missing_file").any():
        raise ValueError("A full-file acquisition is required; gap repair cannot infer it")
    COHORT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(audit, AUDIT_BEFORE_PATH)
    g0.atomic_write_csv(gaps, INTERVALS_PATH)
    frozen = {
        "schema_version": 1,
        "status": "frozen_before_top30_gap_repair",
        "created_at_utc": g0.utc_now(),
        "profit_used": False,
        "selection_changed": False,
        "members": [str(row["freqtrade_pair"]) for row in manifest["members"]],
        "timeframes": list(manifest["requested_ohlcv"]["timeframes"]),
        "selected_workers_maximum": MAXIMUM_WORKERS,
        "audit_before": audit_summary(audit),
        "repair_jobs": len(gaps),
        "artifacts": {
            "audit_before": artifact(AUDIT_BEFORE_PATH),
            "gap_intervals": artifact(INTERVALS_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }
    g0.atomic_write_json(frozen, FREEZE_PATH)
    update_manifest_integrity(
        manifest,
        status="integrity_audit_failed_exact_gap_repair_frozen",
        audit=audit,
        audit_path=AUDIT_BEFORE_PATH,
    )
    return frozen


def isolated_download_args(pair: str, timeframe: str, timerange: str, datadir: Path) -> list[str]:
    if not re.fullmatch(r"\d{10}-\d{10}", timerange):
        raise ValueError(f"Invalid frozen Unix-second timerange: {timerange}")
    if not re.fullmatch(r"[0-9A-Z]+/USDT:USDT", pair):
        raise ValueError(f"Invalid frozen Binance futures pair: {pair}")
    if timeframe not in {*TIMEFRAME_DELTAS, "1M"}:
        raise ValueError(f"Unsupported frozen timeframe: {timeframe}")
    ensure_isolated_path(datadir)
    return [
        str(g4d.CONTROLLER_PYTHON.resolve()),
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
        timeframe,
        "--timerange",
        timerange,
        "--no-parallel-download",
    ]


def ensure_isolated_path(path: Path) -> None:
    root = ISOLATED_ROOT.resolve()
    resolved = path.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"Unsafe meme repair path: {resolved}")


def cleanup_isolated_job(path: Path) -> None:
    ensure_isolated_path(path)
    if path.is_dir():
        shutil.rmtree(path)


def cleanup_empty_session(path: Path) -> None:
    ensure_isolated_path(path)
    if path.is_dir() and not any(path.iterdir()):
        path.rmdir()


def interval_quality(
    frame: DataFrame,
    *,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    expected_rows: int,
) -> dict[str, Any]:
    ordered = g4r.normalized_ohlcv(frame)
    summary, gaps = audit_frame("quality/USDT:USDT", timeframe, ANALYSIS_PATH, ordered)
    first = ordered["date"].min() if len(ordered) else pd.NaT
    last = ordered["date"].max() if len(ordered) else pd.NaT
    if timeframe == "1M":
        expected_last = end - pd.offsets.MonthBegin(1)
    else:
        expected_last = end - TIMEFRAME_DELTAS[timeframe]
    complete = bool(
        len(ordered) == expected_rows
        and not gaps
        and summary["duplicate_timestamps"] == 0
        and summary["invalid_ohlcv_rows"] == 0
        and first == start
        and last == expected_last
    )
    return {
        "complete": complete,
        "actual_rows": len(ordered),
        "missing_rows": expected_rows - len(ordered),
        "gap_events": len(gaps),
        "duplicate_timestamps": summary["duplicate_timestamps"],
        "invalid_ohlcv_rows": summary["invalid_ohlcv_rows"],
        "first_actual_utc": None if pd.isna(first) else first.isoformat(),
        "last_actual_utc": None if pd.isna(last) else last.isoformat(),
    }


def merge_isolated_download(job: dict[str, Any]) -> dict[str, Any]:
    isolated = Path(str(job["isolated_datadir"]))
    target = Path(str(job["target_file"]))
    g4r.validate_main_target(target)
    source = data_path(str(job["pair"]), str(job["timeframe"]), isolated)
    if not source.is_file():
        return {"merge_status": "source_incomplete", "reason": "missing_source_file"}
    incoming = g4r.normalized_ohlcv(pd.read_feather(source))
    start = pd.Timestamp(str(job["interval_start_utc"]))
    end = pd.Timestamp(str(job["interval_end_exclusive_utc"]))
    selected = incoming.loc[incoming["date"].ge(start) & incoming["date"].lt(end)].copy()
    quality = interval_quality(
        selected,
        timeframe=str(job["timeframe"]),
        start=start,
        end=end,
        expected_rows=int(job["expected_rows"]),
    )
    if not quality["complete"]:
        return {
            "merge_status": "source_incomplete",
            "reason": "exchange_did_not_return_complete_frozen_interval",
            **quality,
        }
    existing = g4r.normalized_ohlcv(pd.read_feather(target))
    before_rows = len(existing)
    before_sha256 = g0.sha256_file(target)
    conflicts = g4r.count_overlap_conflicts(existing, selected)
    if conflicts:
        raise ValueError(f"Incoming candles conflict with {conflicts} stored rows")
    merged = (
        pd.concat([existing, selected], ignore_index=True)
        .sort_values("date")
        .drop_duplicates("date", keep="first")
        .reset_index(drop=True)
    )
    selected_after = merged.loc[merged["date"].ge(start) & merged["date"].lt(end)]
    verify = interval_quality(
        selected_after,
        timeframe=str(job["timeframe"]),
        start=start,
        end=end,
        expected_rows=int(job["expected_rows"]),
    )
    if not verify["complete"]:
        raise ValueError("Merged file does not contain the complete frozen interval")
    temporary = target.with_name(f".{target.name}.{job['job_id']}.tmp.feather")
    merged.loc[:, OHLCV_COLUMNS].to_feather(temporary, compression_level=9, compression="lz4")
    if g0.sha256_file(target) != before_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Target changed during atomic gap merge")
    temporary.replace(target)
    return {
        "merge_status": "merged",
        "reason": "complete_exact_gap_atomically_merged",
        **quality,
        "overlap_conflicts": conflicts,
        "target_rows_before": before_rows,
        "target_rows_after": len(merged),
        "target_rows_added": len(merged) - before_rows,
        "target_sha256_before": before_sha256,
        "target_sha256_after": g0.sha256_file(target),
    }


def acquire(*, max_workers: int, recover_stale_run: bool = False) -> int:  # noqa: C901
    if not 1 <= max_workers <= MAXIMUM_WORKERS:
        raise ValueError(f"max_workers must be 1-{MAXIMUM_WORKERS}")
    if not g4d.CONTROLLER_PYTHON.is_file():
        raise FileNotFoundError(f"Missing controller Python: {g4d.CONTROLLER_PYTHON}")
    frozen = prepare(overwrite=False)
    if frozen["artifacts"]["analysis_script"]["sha256"] != g0.sha256_file(ANALYSIS_PATH):
        raise ValueError("Repair script changed after the gap list was frozen")
    if frozen["artifacts"]["gap_intervals"]["sha256"] != g0.sha256_file(INTERVALS_PATH):
        raise ValueError("Frozen top-30 gap interval list changed")
    intervals = pd.read_csv(INTERVALS_PATH)
    if intervals.empty:
        print("The top-30 meme OHLCV pool has no internal gaps.")
        return 0
    previous_runs: list[dict[str, Any]] = []
    if RUN_RECORD_PATH.is_file():
        prior = json.loads(RUN_RECORD_PATH.read_text(encoding="utf-8"))
        if prior.get("status") == "running":
            controller_pid = int(prior.get("controller_pid", 0))
            if not recover_stale_run or psutil.pid_exists(controller_pid):
                raise RuntimeError(f"A top-30 gap repair is already running: {RUN_RECORD_PATH}")
            prior.update(
                {
                    "status": "aborted_after_verified_controller_exit",
                    "completed_at_utc": g0.utc_now(),
                    "stop_reason": "cleanup_path_validation_bug_fixed_before_resume",
                }
            )
            for prior_job in prior.get("jobs", []):
                if prior_job.get("status") == "completed":
                    cleanup_isolated_job(Path(str(prior_job["isolated_datadir"])))
            prior_session_paths = {
                Path(str(job["isolated_datadir"])).parent for job in prior.get("jobs", [])
            }
            for prior_session in prior_session_paths:
                cleanup_empty_session(prior_session)
            g0.atomic_write_json(prior, RUN_RECORD_PATH)
        previous_runs = [*prior.get("previous_runs", []), prior]
    for target, rows in intervals.groupby("target_file", sort=False):
        expected_hashes = set(rows["target_sha256_at_freeze"].astype(str))
        if len(expected_hashes) != 1 or g0.sha256_file(Path(target)) not in expected_hashes:
            raise RuntimeError(f"Frozen gap target changed before repair: {target}")

    session_id = re.sub(r"[^0-9A-Za-z]+", "", g0.utc_now())
    session_root = ISOLATED_ROOT / session_id
    ensure_isolated_path(session_root)
    session_root.mkdir(parents=True, exist_ok=False)
    jobs: list[dict[str, Any]] = []
    for number, row in enumerate(intervals.to_dict(orient="records"), start=1):
        isolated = session_root / f"job_{number:03d}"
        command = isolated_download_args(
            str(row["pair"]),
            str(row["timeframe"]),
            str(row["timerange"]),
            isolated,
        )
        jobs.append(
            {
                **row,
                "job_id": f"{number:03d}",
                "isolated_datadir": str(isolated.resolve()),
                "command": subprocess.list2cmdline(command),
                "log_path": str(
                    (
                        COHORT_ROOT / "meme_top30_gap_repair_logs_20260907" / f"{number:03d}.log"
                    ).resolve()
                ),
                "status": "queued",
                "pid": None,
                "return_code": None,
            }
        )
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": g0.utc_now(),
        "completed_at_utc": None,
        "controller_pid": os.getpid(),
        "controller_python": str(g4d.CONTROLLER_PYTHON.resolve()),
        "maximum_parallel_target_files": max_workers,
        "prelaunch_capacity_check": {
            "logical_processors": os.cpu_count(),
            "observed_project_work": "collectors_only; no competing research batch",
        },
        "jobs": jobs,
        "previous_runs": previous_runs,
        "source_contracts": {"freeze": artifact(FREEZE_PATH)},
    }
    g0.atomic_write_json(record, RUN_RECORD_PATH)
    lock = threading.Lock()

    def update(job: dict[str, Any], **values: Any) -> None:
        with lock:
            job.update(values)
            g0.atomic_write_json(record, RUN_RECORD_PATH)

    def run_target(target_jobs: list[dict[str, Any]]) -> None:
        for job in target_jobs:
            isolated = Path(str(job["isolated_datadir"]))
            isolated.mkdir(parents=True, exist_ok=False)
            command = isolated_download_args(
                str(job["pair"]),
                str(job["timeframe"]),
                str(job["timerange"]),
                isolated,
            )
            log_path = Path(str(job["log_path"]))
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
            merge = merge_isolated_download(job)
            update(
                job,
                status=("completed" if merge["merge_status"] == "merged" else "source_incomplete"),
                return_code=return_code,
                merge=merge,
            )
            if merge["merge_status"] == "merged":
                cleanup_isolated_job(isolated)

    by_target: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        by_target.setdefault(str(job["target_file"]), []).append(job)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_target, values) for values in by_target.values()]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    failures = [job for job in jobs if job["status"] != "completed"]
    manifest = load_manifest()
    audit_after, _ = audit_pool(manifest)
    g0.atomic_write_csv(audit_after, AUDIT_AFTER_PATH)
    after = audit_summary(audit_after)
    final_status = (
        "completed" if not failures and after["failed_files"] == 0 else "completed_with_failures"
    )
    record.update(
        {
            "status": final_status,
            "completed_at_utc": g0.utc_now(),
            "completed_jobs": len(jobs) - len(failures),
            "failed_jobs": len(failures),
            "rows_added": int(
                sum(job.get("merge", {}).get("target_rows_added", 0) for job in jobs)
            ),
            "audit_after": after,
            "artifacts": {"audit_after": artifact(AUDIT_AFTER_PATH)},
        }
    )
    g0.atomic_write_json(record, RUN_RECORD_PATH)
    update_manifest_integrity(
        manifest,
        status=(
            "complete_and_integrity_verified_after_exact_gap_repair"
            if final_status == "completed"
            else "integrity_gap_repair_incomplete"
        ),
        audit=audit_after,
        audit_path=AUDIT_AFTER_PATH,
    )
    cleanup_empty_session(session_root)
    return 0 if final_status == "completed" else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--overwrite-freeze", action="store_true")
    parser.add_argument("--recover-stale-run", action="store_true")
    parser.add_argument("--max-workers", type=int, default=MAXIMUM_WORKERS)
    args = parser.parse_args(argv)
    frozen = prepare(overwrite=args.overwrite_freeze)
    if args.audit_only:
        print(json.dumps(frozen, indent=2))
        return 0
    return acquire(
        max_workers=int(args.max_workers),
        recover_stale_run=bool(args.recover_stale_run),
    )


if __name__ == "__main__":
    raise SystemExit(main())

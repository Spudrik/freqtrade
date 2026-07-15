from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import argparse
import json
import shutil
import sqlite3
import subprocess
import sys

from .builder import default_paths
from .gkg_hourly_metadata_extract import DEFAULT_OUTPUT_DIR, DEFAULT_RAW_DIR
from .gkg_raw_coverage_report import build_coverage_report


REPO_ROOT = Path(__file__).resolve().parents[4]
PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
STATE_DIR = DEFAULT_OUTPUT_DIR / "pre2023_orchestrator"
REPORT_DIR = STATE_DIR / "coverage_reports"
LOG_DIR = REPO_ROOT / "user_data" / "research_news_data" / "gdelt" / "logs"
STATE_PATH = STATE_DIR / "gkg_pre2023_orchestrator_state.json"
YEARS = (2020, 2021, 2022)
MAX_WORKERS = 4
DOWNLOAD_CHUNK_DAYS = 30
RETRY_DEFER_HOURS = 0
ACCEPT_RETRY_GAPS_AFTER_DAYS = 7
MIN_DELETE_COVERAGE_RATIO = 0.90
MAX_GAP_ADVANCES_PER_TICK = 50
MAX_INLINE_GAP_DOWNLOAD_SECONDS = 20


@dataclass(frozen=True)
class YearWindow:
    year: int
    start: datetime
    end: datetime

    @property
    def tag(self) -> str:
        return f"year{self.year}_smart_v4"

    @property
    def summary_path(self) -> Path:
        return DEFAULT_OUTPUT_DIR / f"gkg_hourly_metadata_summary_{self.tag}.json"

    @property
    def parquet_path(self) -> Path:
        return DEFAULT_OUTPUT_DIR / f"gkg_hourly_metadata_{self.tag}.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Orchestrate pre-2023 GKG raw download, compact hourly extraction, and guarded raw deletion."
    )
    parser.add_argument("--status", action="store_true", help="Print current state and do not start work.")
    parser.add_argument("--tick", action="store_true", help="Run one orchestration tick.")
    parser.add_argument("--allow-delete", action="store_true", help="Allow deletion of validated raw GKG ZIPs.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--state-path", type=Path, default=STATE_PATH)
    args = parser.parse_args()

    if not args.status and not args.tick:
        parser.error("Choose --status or --tick")
    if args.raw_dir.resolve() != DEFAULT_RAW_DIR.resolve():
        raise SystemExit(f"Refusing unexpected raw dir: {args.raw_dir}")
    if args.output_dir.resolve() != DEFAULT_OUTPUT_DIR.resolve():
        raise SystemExit(f"Refusing unexpected output dir: {args.output_dir}")

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    state = load_state(args.state_path)
    if args.status:
        payload = build_status(state, args.raw_dir)
    else:
        payload = tick(state, args.state_path, args.raw_dir, allow_delete=args.allow_delete)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def tick(state: dict[str, Any], state_path: Path, raw_dir: Path, *, allow_delete: bool) -> dict[str, Any]:
    require_python()
    state.setdefault("years", {})
    state["updated_at"] = utc_now()
    messages: list[str] = []

    for window in year_windows():
        year_state = state["years"].setdefault(str(window.year), {"phase": "pending"})
        refresh_running_job(year_state)

        if year_state.get("phase") in {"extracting", "downloading", "gap_extracting", "gap_downloading"}:
            save_state(state_path, state)
            return build_tick_payload(state, messages + [f"{window.year} {year_state['phase']} still running."], active_year=window.year)

        if year_state.get("phase") == "complete":
            continue

        if year_state.get("phase") == "error":
            if recover_terminal_gap_error(year_state):
                save_state(state_path, state)
                return build_tick_payload(state, messages + [f"{window.year} terminal gap recorded after recovery."], active_year=window.year)
            save_state(state_path, state)
            return build_tick_payload(
                state,
                messages + [f"{window.year} is in error state; inspect {year_state.get('job_log', 'the job log')}."] ,
                active_year=window.year,
            )

        if year_state.get("phase") == "extracted":
            if allow_delete:
                year_state["available_summary_path"] = year_state.get("summary_path")
                year_state["available_parquet_path"] = year_state.get("parquet_path")
                year_state.setdefault("pending_gap_intervals", missing_intervals_from_report(year_state.get("latest_coverage_report")))
                deletion = delete_window_raw(raw_dir, window.year, window.start, window.end, year_state, label="available")
                year_state.update(deletion)
                if year_state.get("pending_gap_intervals"):
                    year_state["phase"] = "raw_deleted"
                else:
                    year_state["phase"] = "complete"
                save_state(state_path, state)
                return build_tick_payload(state, messages + [f"{window.year} raw deletion completed."], active_year=window.year)
            save_state(state_path, state)
            return build_tick_payload(state, messages + [f"{window.year} extracted; deletion held because --allow-delete was not set."], active_year=window.year)

        if year_state.get("phase") == "gap_extracted":
            if allow_delete:
                active_gap = year_state.get("active_gap") or {}
                deletion = delete_window_raw(
                    raw_dir,
                    window.year,
                    parse_iso(active_gap["start"]),
                    parse_iso(active_gap["end"]),
                    year_state,
                    label=f"gap{active_gap.get('index', 0):03d}",
                )
                done_gap = dict(active_gap)
                done_gap.update(
                    {
                        "deleted_manifest": deletion["deleted_manifest"],
                        "deleted_raw_files": deletion["deleted_raw_files"],
                        "deleted_raw_bytes": deletion["deleted_raw_bytes"],
                    }
                )
                year_state.setdefault("completed_gap_intervals", []).append(done_gap)
                year_state.pop("active_gap", None)
                year_state["phase"] = "raw_deleted"
                save_state(state_path, state)
                return build_tick_payload(state, messages + [f"{window.year} gap raw deletion completed."], active_year=window.year)
            save_state(state_path, state)
            return build_tick_payload(state, messages + [f"{window.year} gap extracted; deletion held because --allow-delete was not set."], active_year=window.year)

        if year_state.get("phase") == "raw_deleted":
            advance_messages = advance_gap_queue(window, year_state)
            messages.extend(advance_messages)
            if year_state.get("phase") != "raw_deleted":
                save_state(state_path, state)
                return build_tick_payload(state, messages, active_year=window.year)
            if year_state.get("pending_gap_intervals"):
                save_state(state_path, state)
                return build_tick_payload(state, messages, active_year=window.year)
            year_state["phase"] = "complete"
            save_state(state_path, state)
            return build_tick_payload(state, messages + [f"{window.year} complete for current known gaps."], active_year=window.year)

        if validate_extraction(window):
            year_state.update(
                {
                    "phase": "extracted",
                    "extracted_at": utc_now(),
                    "summary_path": str(window.summary_path),
                    "parquet_path": str(window.parquet_path),
                }
            )
            save_state(state_path, state)
            return build_tick_payload(state, messages + [f"{window.year} extraction already valid."], active_year=window.year)

        coverage = coverage_for_year(raw_dir, window)
        year_state["latest_coverage_report"] = coverage["json_path"]
        year_state["latest_percent_complete"] = coverage["percent_complete"]
        year_state["present_raw_zip_files"] = coverage["present_raw_zip_files"]
        year_state["expected_15m_files"] = coverage["expected_15m_files"]
        year_state["missing_raw_files"] = coverage["missing_raw_files"]
        year_state["coverage_is_partial"] = coverage["percent_complete"] < 99.999
        retry_summary = retry_status_summary(window)
        if retry_summary["count"]:
            year_state["retry_error_count"] = retry_summary["count"]
            year_state["retry_error_oldest_updated_at"] = retry_summary["oldest_updated_at"]
            year_state["retry_error_newest_updated_at"] = retry_summary["newest_updated_at"]
        else:
            year_state.pop("retry_error_count", None)
            year_state.pop("retry_error_oldest_updated_at", None)
            year_state.pop("retry_error_newest_updated_at", None)

        present_files = int(coverage["present_raw_zip_files"])
        percent_complete = float(coverage["percent_complete"])
        estimated_remaining_files = int(coverage.get("estimated_remaining_files") or 0)
        terminal_404_count = int(coverage.get("terminal_404_count") or 0)
        unresolved_retry_files = max(int(coverage["missing_raw_files"]) - terminal_404_count, 0)
        if (
            present_files > 0
            and unresolved_retry_files > 0
            and retry_summary["count"] >= unresolved_retry_files
            and retry_summary["oldest_updated_at"]
            and retry_error_accept_deadline_passed(str(retry_summary["oldest_updated_at"]))
        ):
            year_state["accepted_partial_after_retry_until"] = utc_now()
            year_state["accepted_partial_reason"] = (
                f"{unresolved_retry_files} retryable missing files remained unresolved for at least "
                f"{ACCEPT_RETRY_GAPS_AFTER_DAYS} days."
            )
            job = start_extraction(window)
            year_state.update(job)
            year_state["phase"] = "extracting"
            save_state(state_path, state)
            return build_tick_payload(
                state,
                messages + [f"{window.year} retry gap aged out; partial extraction started."],
                active_year=window.year,
            )
        if present_files > 0 and (
            percent_complete >= 99.999
            or estimated_remaining_files == 0
            or not can_download_missing_now(coverage, raw_dir)
        ):
            job = start_extraction(window)
            year_state.update(job)
            year_state["phase"] = "extracting"
            save_state(state_path, state)
            return build_tick_payload(state, messages + [f"{window.year} extraction started."], active_year=window.year)

        job = start_download(window)
        year_state.update(job)
        year_state["phase"] = "downloading"
        save_state(state_path, state)
        return build_tick_payload(state, messages + [f"{window.year} missing raw download started."], active_year=window.year)

    save_state(state_path, state)
    return build_tick_payload(state, messages + ["Pre-2023 GKG orchestration complete."], active_year=None)


def build_status(state: dict[str, Any], raw_dir: Path) -> dict[str, Any]:
    free = disk_free(raw_dir)
    return {
        "state_path": str(STATE_PATH),
        "raw_dir": str(raw_dir),
        "raw_drive_free_gb": round(free / 1024**3, 2),
        "years": state.get("years", {}),
    }


def build_tick_payload(state: dict[str, Any], messages: list[str], active_year: int | None) -> dict[str, Any]:
    return {
        "active_year": active_year,
        "messages": messages,
        "state_path": str(STATE_PATH),
        "years": state.get("years", {}),
    }


def coverage_for_year(raw_dir: Path, window: YearWindow) -> dict[str, Any]:
    return coverage_for_window(raw_dir, window.start, window.end, tag=f"year{window.year}_latest")


def coverage_for_window(raw_dir: Path, start: datetime, end: datetime, *, tag: str) -> dict[str, Any]:
    paths = default_paths(None)
    return build_coverage_report(
        db_path=paths.gdelt_db,
        raw_dir=raw_dir,
        start=start,
        end=end,
        output_dir=REPORT_DIR,
        tag=tag,
        validate_sample=0,
    )


def can_download_missing_now(coverage: dict[str, Any], raw_dir: Path) -> bool:
    estimated = int(coverage.get("estimated_remaining_bytes") or 0)
    if estimated <= 0:
        return True
    return disk_free(raw_dir) > int(estimated * 1.25)


def retry_status_summary(window: YearWindow) -> dict[str, Any]:
    paths = default_paths(None)
    db_path = paths.gdelt_db
    if not db_path.exists():
        return {"count": 0, "oldest_updated_at": None, "newest_updated_at": None}
    with sqlite3.connect(str(db_path), timeout=30.0) as conn:
        conn.row_factory = sqlite3.Row
        if not _table_exists(conn, "gkg_raw_download_status"):
            return {"count": 0, "oldest_updated_at": None, "newest_updated_at": None}
        row = conn.execute(
            """
            SELECT COUNT(*) AS count, MIN(updated_at) AS oldest_updated_at, MAX(updated_at) AS newest_updated_at
            FROM gkg_raw_download_status
            WHERE date >= ? AND date < ?
              AND status IN ('http_error', 'download_error')
            """,
            (window.start.isoformat(), window.end.isoformat()),
        ).fetchone()
    return {
        "count": int(row["count"] or 0),
        "oldest_updated_at": row["oldest_updated_at"],
        "newest_updated_at": row["newest_updated_at"],
    }


def retry_error_accept_deadline_passed(oldest_updated_at: str) -> bool:
    oldest = parse_iso(oldest_updated_at)
    deadline = oldest + timedelta(days=ACCEPT_RETRY_GAPS_AFTER_DAYS)
    return datetime.now(timezone.utc) >= deadline


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def start_extraction(window: YearWindow) -> dict[str, Any]:
    log_path = LOG_DIR / f"gkg_hourly_metadata_extract_{window.year}_{stamp_for_file()}.log"
    command = [
        str(PYTHON),
        "-m",
        "user_data.Custom_Launcher.research.context_features.gkg_hourly_metadata_extract",
        "--start",
        window.start.date().isoformat(),
        "--end",
        window.end.date().isoformat(),
        "--tag",
        window.tag,
        "--workers",
        str(MAX_WORKERS),
        "--worker-backend",
        "process",
        "--progress-every-files",
        "1000",
        "--skip-csv",
    ]
    process = launch_background(command, log_path)
    return {
        "job_type": "extract",
        "job_pid": process.pid,
        "job_started_at": utc_now(),
        "job_log": str(log_path),
        "summary_path": str(window.summary_path),
        "parquet_path": str(window.parquet_path),
    }


def start_download(window: YearWindow) -> dict[str, Any]:
    log_path = LOG_DIR / f"gkg_raw_download_{window.year}_{stamp_for_file()}.log"
    command = [
        str(PYTHON),
        "-m",
        "user_data.Custom_Launcher.research.context_features.gkg_raw_download",
        "--start",
        window.start.date().isoformat(),
        "--end",
        window.end.date().isoformat(),
        "--next-missing-chunk-days",
        str(DOWNLOAD_CHUNK_DAYS),
        "--max-workers",
        str(MAX_WORKERS),
        "--defer-retry-failures-hours",
        str(RETRY_DEFER_HOURS),
        "--progress-every",
        "500",
    ]
    process = launch_background(command, log_path)
    return {
        "job_type": "download",
        "job_pid": process.pid,
        "job_started_at": utc_now(),
        "job_log": str(log_path),
    }


def start_gap_download(window: YearWindow, gap: dict[str, Any]) -> dict[str, Any]:
    process, log_path = start_gap_download_process(window, gap)
    return {
        "job_type": "gap_download",
        "job_pid": process.pid,
        "job_started_at": utc_now(),
        "job_log": str(log_path),
    }


def start_gap_download_process(window: YearWindow, gap: dict[str, Any]) -> tuple[subprocess.Popen[Any], Path]:
    start = parse_iso(gap["start"])
    end = parse_iso(gap["end"])
    log_path = LOG_DIR / f"gkg_raw_download_{window.year}_gap{int(gap['index']):03d}_{stamp_for_file()}.log"
    command = [
        str(PYTHON),
        "-m",
        "user_data.Custom_Launcher.research.context_features.gkg_raw_download",
        "--start",
        start.isoformat(),
        "--end",
        end.isoformat(),
        "--max-workers",
        str(MAX_WORKERS),
        "--defer-retry-failures-hours",
        str(RETRY_DEFER_HOURS),
        "--progress-every",
        "500",
    ]
    process = launch_background(command, log_path)
    return process, log_path


def start_gap_extraction(window: YearWindow, gap: dict[str, Any]) -> dict[str, Any]:
    start = parse_iso(gap["start"])
    end = parse_iso(gap["end"])
    tag = f"year{window.year}_gap{int(gap['index']):03d}_smart_v4"
    log_path = LOG_DIR / f"gkg_hourly_metadata_extract_{window.year}_gap{int(gap['index']):03d}_{stamp_for_file()}.log"
    command = [
        str(PYTHON),
        "-m",
        "user_data.Custom_Launcher.research.context_features.gkg_hourly_metadata_extract",
        "--start",
        start.isoformat(),
        "--end",
        end.isoformat(),
        "--tag",
        tag,
        "--workers",
        str(MAX_WORKERS),
        "--worker-backend",
        "process",
        "--progress-every-files",
        "500",
        "--skip-csv",
    ]
    process = launch_background(command, log_path)
    return {
        "job_type": "gap_extract",
        "job_pid": process.pid,
        "job_started_at": utc_now(),
        "job_log": str(log_path),
        "summary_path": str(DEFAULT_OUTPUT_DIR / f"gkg_hourly_metadata_summary_{tag}.json"),
        "parquet_path": str(DEFAULT_OUTPUT_DIR / f"gkg_hourly_metadata_{tag}.parquet"),
    }


def launch_background(command: list[str], log_path: Path) -> subprocess.Popen[Any]:
    log_handle = log_path.open("a", encoding="utf-8")
    log_handle.write(f"[{utc_now()}] Starting: {' '.join(command)}\n")
    log_handle.flush()
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=flags,
    )


def advance_gap_queue(window: YearWindow, year_state: dict[str, Any]) -> list[str]:
    messages: list[str] = []
    advances = 0
    while advances < MAX_GAP_ADVANCES_PER_TICK:
        pending_gaps = year_state.get("pending_gap_intervals") or []
        if not pending_gaps:
            year_state["pending_gap_intervals"] = []
            return messages

        gap = pending_gaps.pop(0)
        gap["index"] = int(year_state.get("gap_sequence", 0)) + 1
        year_state["gap_sequence"] = gap["index"]
        year_state["pending_gap_intervals"] = pending_gaps
        year_state["active_gap"] = gap

        gap_coverage = coverage_for_gap(gap)
        year_state["active_gap_latest_coverage_report"] = gap_coverage["json_path"]
        present = int(gap_coverage["present_raw_zip_files"])
        missing = int(gap_coverage["missing_raw_files"])
        terminal = int(gap_coverage["terminal_404_count"])

        if missing > 0 and present == 0 and terminal >= missing:
            record_terminal_gap(year_state, gap, gap_coverage)
            messages.append(f"{window.year} gap{int(gap['index']):03d} recorded as terminal http_404.")
            advances += 1
            continue

        if present > 0 and terminal >= missing:
            job = start_gap_extraction(window, gap)
            year_state.update(job)
            year_state["phase"] = "gap_extracting"
            messages.append(f"{window.year} gap{int(gap['index']):03d} extraction started.")
            return messages

        process, log_path = start_gap_download_process(window, gap)
        year_state.update(
            {
                "job_type": "gap_download",
                "job_pid": process.pid,
                "job_started_at": utc_now(),
                "job_log": str(log_path),
                "phase": "gap_downloading",
            }
        )
        messages.append(f"{window.year} gap{int(gap['index']):03d} targeted download started.")

        if int(gap.get("missing_files") or 0) > 8:
            return messages

        try:
            process.wait(timeout=MAX_INLINE_GAP_DOWNLOAD_SECONDS)
        except subprocess.TimeoutExpired:
            return messages

        refresh_running_job(year_state)
        if year_state.get("phase") == "raw_deleted":
            result = year_state.get("last_gap_download_result", "gap_download_finished")
            messages.append(f"{window.year} gap{int(gap['index']):03d} {result}.")
            advances += 1
            continue
        return messages

    remaining = len(year_state.get("pending_gap_intervals") or [])
    messages.append(f"{window.year} gap batch limit reached; {remaining} pending gaps remain.")
    return messages


def coverage_for_gap(gap: dict[str, Any]) -> dict[str, Any]:
    start = parse_iso(gap["start"])
    end = parse_iso(gap["end"])
    return coverage_for_window(
        DEFAULT_RAW_DIR,
        start,
        end,
        tag=f"year{start.year}_gap{int(gap.get('index', 0)):03d}_download_check",
    )


def record_terminal_gap(
    year_state: dict[str, Any],
    active_gap: dict[str, Any],
    gap_coverage: dict[str, Any],
    *,
    recovered_from_error: bool = False,
) -> None:
    terminal_gap = dict(active_gap)
    terminal_gap.update(
        {
            "terminal_reason": "http_404",
            "coverage_report": gap_coverage["json_path"],
            "recorded_at": utc_now(),
        }
    )
    if recovered_from_error:
        terminal_gap["recovered_from_error"] = True
    year_state.setdefault("terminal_gap_intervals", []).append(terminal_gap)
    if year_state.get("job_log"):
        year_state["last_job_log"] = year_state["job_log"]
    year_state.pop("active_gap", None)
    year_state.pop("job_pid", None)
    year_state.pop("job_type", None)
    year_state.pop("job_started_at", None)
    year_state.pop("job_log", None)
    year_state["phase"] = "raw_deleted"
    year_state["download_finished_at"] = utc_now()
    year_state["last_gap_download_result"] = "terminal_404_recorded"
    year_state.pop("error", None)


def refresh_running_job(year_state: dict[str, Any]) -> None:
    phase = year_state.get("phase")
    pid = year_state.get("job_pid")
    if phase not in {"extracting", "downloading", "gap_extracting", "gap_downloading"} or not pid:
        return
    if process_running(int(pid)):
        return
    job_type = year_state.get("job_type")
    if job_type == "extract":
        year = int(Path(str(year_state["summary_path"])).stem.split("year", 1)[1].split("_", 1)[0])
        window = next(item for item in year_windows() if item.year == year)
        if validate_extraction(window):
            year_state["phase"] = "extracted"
            year_state["extracted_at"] = utc_now()
            year_state.pop("error", None)
        else:
            year_state["phase"] = "error"
            year_state["error"] = "Extraction process exited but summary/parquet validation failed."
    elif job_type == "download":
        year_state["phase"] = "pending"
        year_state["download_finished_at"] = utc_now()
        year_state.pop("error", None)
    elif job_type == "gap_download":
        active_gap = year_state.get("active_gap") or {}
        gap_start = parse_iso(active_gap["start"])
        gap_end = parse_iso(active_gap["end"])
        gap_coverage = coverage_for_window(
            DEFAULT_RAW_DIR,
            gap_start,
            gap_end,
            tag=f"year{gap_start.year}_gap{int(active_gap.get('index', 0)):03d}_download_check",
        )
        year_state["active_gap_latest_coverage_report"] = gap_coverage["json_path"]
        if int(gap_coverage["missing_raw_files"]) > int(gap_coverage["terminal_404_count"]):
            pending = year_state.get("pending_gap_intervals") or []
            year_state["pending_gap_intervals"] = [active_gap, *pending]
            year_state.pop("active_gap", None)
            year_state["phase"] = "raw_deleted"
            year_state["download_finished_at"] = utc_now()
            year_state["last_gap_download_result"] = "no_files_downloaded_requeued"
            year_state.pop("job_pid", None)
            return
        if int(gap_coverage["present_raw_zip_files"]) == 0 and int(gap_coverage["terminal_404_count"]) >= int(gap_coverage["missing_raw_files"]):
            record_terminal_gap(year_state, active_gap, gap_coverage)
            return
        year = int(gap_start.year)
        window = next(item for item in year_windows() if item.year == year)
        job = start_gap_extraction(window, active_gap)
        year_state.update(job)
        year_state["phase"] = "gap_extracting"
        year_state["download_finished_at"] = utc_now()
        return
    elif job_type == "gap_extract":
        active_gap = year_state.get("active_gap") or {}
        if validate_extraction_paths(
            summary_path=Path(str(year_state["summary_path"])),
            parquet_path=Path(str(year_state["parquet_path"])),
            start=parse_iso(active_gap["start"]),
            end=parse_iso(active_gap["end"]),
            min_coverage_ratio=0.0,
        ):
            year_state["phase"] = "gap_extracted"
            year_state["gap_extracted_at"] = utc_now()
            year_state.pop("error", None)
        else:
            year_state["phase"] = "error"
            year_state["error"] = "Gap extraction process exited but summary/parquet validation failed."
    year_state.pop("job_pid", None)


def recover_terminal_gap_error(year_state: dict[str, Any]) -> bool:
    if year_state.get("job_type") != "gap_extract":
        return False
    active_gap = year_state.get("active_gap") or {}
    report_path = year_state.get("active_gap_latest_coverage_report")
    if not active_gap or not report_path:
        return False
    path = Path(str(report_path))
    if not path.exists():
        return False
    report = json.loads(path.read_text(encoding="utf-8"))
    if int(report.get("present_raw_zip_files") or 0) != 0:
        return False
    if int(report.get("missing_raw_files") or 0) <= 0:
        return False
    if int(report.get("terminal_404_count") or 0) < int(report.get("missing_raw_files") or 0):
        return False
    terminal_gap = dict(active_gap)
    terminal_gap.update(
        {
            "terminal_reason": "http_404",
            "coverage_report": str(path),
            "recorded_at": utc_now(),
            "recovered_from_error": True,
        }
    )
    year_state.setdefault("terminal_gap_intervals", []).append(terminal_gap)
    year_state.pop("active_gap", None)
    year_state.pop("job_pid", None)
    year_state["phase"] = "raw_deleted"
    year_state["last_gap_download_result"] = "terminal_404_recorded"
    year_state["recovered_at"] = utc_now()
    year_state.pop("error", None)
    return True


def validate_extraction(window: YearWindow) -> bool:
    return validate_extraction_paths(
        summary_path=window.summary_path,
        parquet_path=window.parquet_path,
        start=window.start,
        end=window.end,
        min_coverage_ratio=MIN_DELETE_COVERAGE_RATIO,
    )


def validate_extraction_paths(
    *,
    summary_path: Path,
    parquet_path: Path,
    start: datetime,
    end: datetime,
    min_coverage_ratio: float,
) -> bool:
    if not summary_path.exists() or not parquet_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if summary.get("feature_schema_version") != "gkg_hourly_metadata_smart_v4":
        return False
    if summary.get("start") != start.isoformat() or summary.get("end") != end.isoformat():
        return False
    if int(summary.get("feature_rows") or 0) != expected_feature_rows(start, end):
        return False
    if summary.get("corrupt_files"):
        return False
    coverage = float(summary.get("file_coverage_ratio") or 0.0)
    return coverage >= min_coverage_ratio


def delete_window_raw(
    raw_dir: Path,
    year: int,
    start: datetime,
    end: datetime,
    year_state: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    raw_root = raw_dir.resolve()
    if raw_root != DEFAULT_RAW_DIR.resolve():
        raise RuntimeError(f"Refusing to delete from unexpected raw dir: {raw_root}")
    if label == "available":
        window = next(item for item in year_windows() if item.year == year)
        if not validate_extraction(window):
            raise RuntimeError(f"Refusing to delete {year} raw because extraction validation failed.")
    else:
        if not validate_extraction_paths(
            summary_path=Path(str(year_state["summary_path"])),
            parquet_path=Path(str(year_state["parquet_path"])),
            start=start,
            end=end,
            min_coverage_ratio=0.0,
        ):
            raise RuntimeError(f"Refusing to delete {year} {label} raw because extraction validation failed.")

    files = window_raw_files(raw_root, start, end)
    manifest_path = STATE_DIR / f"deleted_gkg_raw_{year}_{label}_{stamp_for_file()}.json"
    total_bytes = sum(path.stat().st_size for path in files)
    manifest = {
        "year": year,
        "label": label,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "deleted_at": utc_now(),
        "raw_dir": str(raw_root),
        "summary_path": str(year_state.get("summary_path")),
        "parquet_path": str(year_state.get("parquet_path")),
        "coverage_was_partial": bool(year_state.get("coverage_is_partial")),
        "latest_coverage_report": year_state.get("latest_coverage_report"),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": [str(path) for path in files],
    }
    tmp_path = manifest_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(manifest_path)

    for path in files:
        resolved = path.resolve()
        if resolved.parent != raw_root or not resolved.name.startswith(str(year)) or not resolved.name.endswith(".gkg.csv.zip"):
            raise RuntimeError(f"Refusing unexpected deletion target: {resolved}")
        resolved.unlink()

    return {
        "deleted_at": utc_now(),
        "deleted_manifest": str(manifest_path),
        "deleted_raw_files": len(files),
        "deleted_raw_bytes": total_bytes,
    }


def year_raw_files(raw_dir: Path, year: int) -> list[Path]:
    return sorted(path for path in raw_dir.glob(f"{year}*.gkg.csv.zip") if path.is_file())


def window_raw_files(raw_dir: Path, start: datetime, end: datetime) -> list[Path]:
    files: list[Path] = []
    year = start.year
    for path in year_raw_files(raw_dir, year):
        stamp = datetime.strptime(path.name[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        if start <= stamp < end:
            files.append(path)
    return files


def process_running(pid: int) -> bool:
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        check=False,
    )
    return str(pid) in result.stdout


def disk_free(path: Path) -> int:
    target = path if path.exists() else path.parent
    return shutil.disk_usage(target).free


def year_windows() -> list[YearWindow]:
    return [
        YearWindow(
            year=year,
            start=datetime(year, 1, 1, tzinfo=timezone.utc),
            end=datetime(year + 1, 1, 1, tzinfo=timezone.utc),
        )
        for year in YEARS
    ]


def expected_feature_rows(start: datetime, end: datetime) -> int:
    current = start.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    rows = 0
    while current < end:
        rows += 1
        current += timedelta(hours=1)
    return rows


def missing_intervals_from_report(report_path: str | None) -> list[dict[str, Any]]:
    if not report_path:
        return []
    path = Path(report_path)
    if not path.exists():
        return []
    report = json.loads(path.read_text(encoding="utf-8"))
    intervals: list[dict[str, Any]] = []
    for item in report.get("missing_intervals", []):
        start = datetime.strptime(item["start_stamp"], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        end = datetime.strptime(item["end_stamp_exclusive"], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        if end <= start:
            end = start + timedelta(minutes=15)
        intervals.append(
            {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "missing_files": int(item.get("missing_files") or 0),
            }
        )
    return intervals


def parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(second=0, microsecond=0)


def require_python() -> None:
    if not PYTHON.exists():
        raise RuntimeError(f"Required controller Python is missing: {PYTHON}")


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"created_at": utc_now(), "years": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stamp_for_file() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    raise SystemExit(main())

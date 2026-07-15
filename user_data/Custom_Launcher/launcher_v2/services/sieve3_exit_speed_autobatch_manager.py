from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, time
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from zoneinfo import ZoneInfo

from .entry_sieve_service import EntrySieveService, EntrySieveSettings


APP_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = APP_DIR.parents[1]
MAIN_RUNTIME_DIR = APP_DIR / "launcher_v2" / "runtime" / "entry_sieve"
RUNTIME_DIR = APP_DIR / "launcher_v2" / "runtime" / "entry_sieve_speed"
SOURCE_BATCH_FILE = MAIN_RUNTIME_DIR / "queues" / "sieve3_exit_autobatches.json"
BATCH_FILE = RUNTIME_DIR / "queues" / "sieve3_exit_speed_batches.json"
STATE_FILE = RUNTIME_DIR / "queues" / "sieve3_exit_speed_autobatch_state.json"
CORE_OVERRIDE_FILE = RUNTIME_DIR / "queues" / "sieve3_exit_speed_core_override.json"
CORE_PAIRS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "ADA/USDT:USDT",
    "SOL/USDT:USDT",
    "BNB/USDT:USDT",
]
SPEED_VALIDATION_WINDOW = {
    "name": "speed_validation_2024_2026",
    "regime": "mixed",
    "segment_type": "speed_validation",
    "market_state": "mixed",
    "timerange": "20240401-20260401",
    "rationale": "Shortened Sieve3 exit validation window for fast first-pass comparison. Hyperopt training windows remain normal.",
}
LOW_POWER_CORES = 8
HIGH_POWER_CORES = 16
LOCAL_TZ = ZoneInfo("Europe/London")


def _temporary_core_override(now: datetime) -> int | None:
    payload = _load_json(CORE_OVERRIDE_FILE, {})
    if not isinstance(payload, dict):
        return None
    try:
        cores = int(str(payload.get("max_cores") or "").strip())
    except (TypeError, ValueError):
        return None
    until_text = str(payload.get("until") or "").strip()
    if not until_text:
        return None
    try:
        until = datetime.fromisoformat(until_text)
    except ValueError:
        return None
    if until.tzinfo is None:
        until = until.replace(tzinfo=LOCAL_TZ)
    if now.astimezone(LOCAL_TZ) >= until.astimezone(LOCAL_TZ):
        return None
    return max(1, cores)


def _scheduled_cores(now: datetime | None = None) -> int:
    now = now or datetime.now(LOCAL_TZ)
    override = _temporary_core_override(now)
    if override is not None:
        return override
    weekday = now.weekday()
    local_time = now.time()
    if weekday <= 3 and time(7, 0) <= local_time < time(19, 0):
        return LOW_POWER_CORES
    if weekday == 4 and time(7, 0) <= local_time < time(16, 0):
        return LOW_POWER_CORES
    return HIGH_POWER_CORES


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return default


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _summary_for_job(job_id: str) -> dict[str, Any]:
    if not job_id:
        return {}
    return _load_json(RUNTIME_DIR / "results" / f"{job_id}.summary.json", {})


def _status_for_job(job_id: str) -> dict[str, Any]:
    if not job_id:
        return {}
    return _load_json(RUNTIME_DIR / "status" / f"{job_id}.json", {})


def _is_finished(payload: dict[str, Any]) -> bool:
    return str(payload.get("status") or "").lower() == "finished"


def _is_failed(payload: dict[str, Any]) -> bool:
    return str(payload.get("status") or "").lower() in {"failed", "error"}


def _has_errors(payload: dict[str, Any]) -> bool:
    try:
        return int(payload.get("error_row_count") or 0) > 0
    except (TypeError, ValueError):
        return bool(payload.get("has_error_rows"))


def _source_batches() -> list[dict[str, Any]]:
    payload = _load_json(SOURCE_BATCH_FILE, {})
    batches = payload.get("batches") if isinstance(payload, dict) else []
    return [batch for batch in batches if isinstance(batch, dict) and batch.get("id")]


def _speed_batch_number(batch_id: str) -> int:
    try:
        return int(str(batch_id).rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def _speed_batch_sort_key(batch: dict[str, Any]) -> int:
    return _speed_batch_number(str(batch.get("id") or ""))


def _source_batch_priority(batch: dict[str, Any]) -> tuple[int, int]:
    source_id = str(batch.get("id") or "")
    if source_id.startswith("sieve3_exit_level_zone_reversal_autobatch_"):
        group = 0
    elif source_id.startswith("sieve3_exit_standard_autobatch_"):
        group = 1
    elif source_id.startswith("sieve3_exit_pattern_autobatch_"):
        group = 2
    else:
        group = 1
    return group, _speed_batch_number(source_id)


def _latest_completed_speed_source() -> tuple[int, str]:
    latest_number = 0
    latest_source = ""
    for summary_path in sorted((RUNTIME_DIR / "results").glob("*entry_sieve3_exit_speed_batch_*.summary.json")):
        summary = _load_json(summary_path, {})
        if not _is_finished(summary) or _has_errors(summary):
            continue
        speed_batch = str(summary.get("strategy_batch") or "")
        number = _speed_batch_number(speed_batch)
        if number <= latest_number:
            continue
        job_id = str(summary.get("job_id") or summary_path.name.replace(".summary.json", ""))
        job = _load_json(RUNTIME_DIR / "jobs" / f"{job_id}.json", {})
        latest_number = number
        latest_source = str(job.get("source_batch") or "")
    return latest_number, latest_source


def _completed_speed_files() -> set[str]:
    completed: set[str] = set()
    for summary_path in sorted((RUNTIME_DIR / "results").glob("*entry_sieve3_exit_speed_batch_*.summary.json")):
        summary = _load_json(summary_path, {})
        if not _is_finished(summary) or _has_errors(summary):
            continue
        job_id = str(summary.get("job_id") or summary_path.name.replace(".summary.json", ""))
        job = _load_json(RUNTIME_DIR / "jobs" / f"{job_id}.json", {})
        for row in job.get("strategies") or []:
            if not isinstance(row, dict):
                continue
            strategy_file = str(row.get("strategy_file") or "").strip()
            if strategy_file:
                completed.add(Path(strategy_file).name)
    return completed


def _next_speed_number() -> int:
    numbers: list[int] = []
    state = _load_json(STATE_FILE, {})
    if isinstance(state, dict):
        for batch_id in [state.get("current_batch"), *(state.get("completed_batches") or [])]:
            number = _speed_batch_number(str(batch_id or ""))
            if number:
                numbers.append(number)
    payload = _load_json(BATCH_FILE, {})
    if isinstance(payload, dict):
        for batch in payload.get("batches") or []:
            if not isinstance(batch, dict):
                continue
            number = _speed_batch_number(str(batch.get("id") or ""))
            if number:
                numbers.append(number)
    for summary_path in sorted((RUNTIME_DIR / "results").glob("*entry_sieve3_exit_speed_batch_*.summary.json")):
        summary = _load_json(summary_path, {})
        if not _is_finished(summary) or _has_errors(summary):
            continue
        number = _speed_batch_number(str(summary.get("strategy_batch") or ""))
        if number:
            numbers.append(number)
    return (max(numbers) if numbers else 0) + 1


def _next_source_batch() -> tuple[int, dict[str, Any]] | tuple[None, None]:
    source_batches = _source_batches()
    completed_files = _completed_speed_files()
    for source_batch in sorted(source_batches, key=_source_batch_priority):
        source_id = str(source_batch.get("id") or "")
        include = [str(name) for name in (source_batch.get("include") or []) if str(name).strip()]
        pending = [name for name in include if name not in completed_files]
        if source_id and pending:
            filtered = dict(source_batch)
            filtered["include"] = pending
            if len(pending) != len(include):
                filtered["label"] = f"{source_batch.get('label', source_id)} - pending {len(pending)}/{len(include)} files"
            return _next_speed_number(), filtered
    return None, None


def _write_speed_batch(speed_number: int, source_batch: dict[str, Any]) -> dict[str, Any]:
    source_id = str(source_batch.get("id") or "")
    speed_id = f"sieve3_exit_speed_batch_{speed_number:03d}"
    speed_batch = {
        "id": speed_id,
        "label": f"Sieve3 exit speed batch {speed_number:03d} from {source_id} ({len(source_batch.get('include') or [])} files) | 5 core pairs | exit MultiMetric | speed validation 20240401-20260401",
        "include": list(source_batch.get("include") or []),
        "exclude": list(source_batch.get("exclude") or []),
        "window_mode": source_batch.get("window_mode", "pattern"),
        "source_batch": source_id,
        "description": "Isolated Sieve3 exit speed-validation lane. Hyperopt training windows remain routed normally; validation/backtest window is shortened to 20240401-20260401.",
    }
    payload = _load_json(BATCH_FILE, {"schema_version": 1, "batches": []})
    if not isinstance(payload, dict):
        payload = {"schema_version": 1, "batches": []}
    batches = payload.setdefault("batches", [])
    replaced = False
    for index, batch in enumerate(list(batches)):
        if isinstance(batch, dict) and batch.get("id") == speed_id:
            batches[index] = speed_batch
            replaced = True
            break
    if not replaced:
        batches.append(speed_batch)
    batches[:] = sorted([batch for batch in batches if isinstance(batch, dict) and batch.get("id")], key=_speed_batch_sort_key)
    payload["default_batch"] = speed_id
    payload["updated_at"] = datetime.now().astimezone().isoformat()
    _save_json(BATCH_FILE, payload)
    return speed_batch


def _settings(speed_batch: dict[str, Any], source_batch: dict[str, Any], *, cores: int) -> EntrySieveSettings:
    source_id = str(source_batch.get("id") or "")
    pattern_batch = str(source_batch.get("window_mode") or "").lower() == "pattern" or source_id.startswith("sieve3_exit_pattern_autobatch_")
    data = {
        "preset_name": "LauncherV2-auto",
        "preset_file": "launcher_v2/runtime/entry_sieve/jobs/presets_sieve3_exit_multimetric_tmp_20260613_5pairs.json",
        "market_windows_file": "explorer/config/market_windows.json",
        "auto_window_mode": pattern_batch,
        "auto_window_count": "2" if pattern_batch else "",
        "auto_validation_window": "full_cycle_2020_2026",
        "training_windows": [] if pattern_batch else ["long_cycle_mixed_2020_2022", "long_cycle_mixed_2023_2025"],
        "validation_windows": ["full_cycle_2020_2026"],
        "epochs": "300",
        "auto_epochs": False,
        "auto_epochs_cap": "300",
        "hyperopt_jobs": str(cores),
        "random_state": "42,1337",
        "sampling_seed": "42,1337",
        "split_venv_pipeline": True,
        "max_cores_allowed": str(cores),
        "backtest_worker_count": str(cores),
        "strategy_batch": str(speed_batch.get("id") or ""),
        "strategy_batch_file": str(BATCH_FILE),
        "strategy_filter": "sieve3_exit_*.py",
        "speed_run_mode": False,
        "speed_pair_count": "5",
        "speed_pairs": CORE_PAIRS,
        "normal_pair_group": "5x Core Speed Pairs",
        "normal_run_pairs": CORE_PAIRS,
        "take_profit_pct": "3",
        "stoploss_pct": "3",
        "control_entry_exits": False,
        "target_sweep_enabled": False,
    }
    allowed = {field.name for field in fields(EntrySieveSettings)}
    return EntrySieveSettings(**{key: value for key, value in data.items() if key in allowed})


def _patch_job(job_path: Path, speed_batch: dict[str, Any], source_batch: dict[str, Any]) -> dict[str, Any]:
    job = _load_json(job_path, {})
    source_id = str(source_batch.get("id") or "")
    job["auto_validation_window"] = SPEED_VALIDATION_WINDOW
    job["validation_windows"] = [SPEED_VALIDATION_WINDOW]
    job["hyperopt_spaces"] = "sell"
    job["target_family"] = "exits"
    job["speed_lane"] = True
    job["source_batch"] = source_id
    job["speed_validation_timerange"] = "20240401-20260401"
    job["speed_run_mode"] = False
    job["speed_pairs"] = CORE_PAIRS
    job["speed_pair_count"] = "5"
    job["normal_pair_group"] = "5x Core Speed Pairs"
    job["normal_run_pairs"] = CORE_PAIRS
    job["control_entry_exits"] = False
    job["strategy_batch_label"] = str(speed_batch.get("label") or speed_batch.get("id"))
    for row in job.get("strategies") or []:
        if isinstance(row, dict):
            row["strategy_batch"] = str(speed_batch.get("id") or "")
            row["strategy_batch_label"] = str(speed_batch.get("label") or speed_batch.get("id"))
            row["source_batch"] = source_id
    _save_json(job_path, job)
    return job


def _launch_job(job_path: Path, python_exe: str) -> int:
    job = _load_json(job_path, {})
    job_id = str(job.get("job_id") or job_path.stem)
    log_dir = RUNTIME_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout = (log_dir / f"{job_id}.stdout.log").open("w", encoding="utf-8", errors="replace")
    stderr = (log_dir / f"{job_id}.stderr.log").open("w", encoding="utf-8", errors="replace")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if sys.platform == "win32" else 0
    process = subprocess.Popen(
        [
            python_exe,
            "-u",
            "-m",
            "launcher_v2.services.entry_sieve_runner",
            "--job-file",
            str(job_path),
            "--resume-existing-results",
        ],
        cwd=str(APP_DIR),
        stdout=stdout,
        stderr=stderr,
        creationflags=flags,
    )
    stdout.close()
    stderr.close()
    return int(process.pid)


def run_once(*, dry_run: bool = False) -> int:
    service = EntrySieveService(APP_DIR, str(REPO_ROOT / ".venv" / "Scripts" / "python.exe"))
    service.runtime_dir = RUNTIME_DIR
    active = service.live_runs()
    if active:
        print(json.dumps({"decision": "active", "active": active[:3]}, indent=2, default=str))
        return 0

    state = _load_json(STATE_FILE, {})
    if not isinstance(state, dict):
        state = {}
    current_job_id = str(state.get("current_job_id") or "")
    if current_job_id:
        status = _status_for_job(current_job_id) or _summary_for_job(current_job_id)
        if _is_failed(status) or _has_errors(status):
            print(json.dumps({"decision": "blocked", "reason": "previous_speed_batch_failed", "job_id": current_job_id, "status": status}, indent=2, default=str))
            return 2
        if status and not _is_finished(status):
            print(json.dumps({"decision": "blocked", "reason": "previous_speed_batch_not_finished_but_not_live", "job_id": current_job_id, "status": status}, indent=2, default=str))
            return 2
        if _is_finished(status):
            state["completed_batches"] = list(dict.fromkeys([*state.get("completed_batches", []), state.get("current_batch", "")]))
            state["current_job_id"] = ""
            state["current_batch"] = ""

    speed_number, source_batch = _next_source_batch()
    if speed_number is None or source_batch is None:
        state["status"] = "finished"
        state["updated_at"] = datetime.now().astimezone().isoformat()
        _save_json(STATE_FILE, state)
        print(json.dumps({"decision": "finished"}, indent=2))
        return 0

    speed_batch = _write_speed_batch(int(speed_number), source_batch)
    cores = _scheduled_cores()
    settings = _settings(speed_batch, source_batch, cores=cores)
    job_path = service.build_job(settings)
    job = _patch_job(job_path, speed_batch, source_batch)
    job_id = str(job.get("job_id") or job_path.stem)
    state.update(
        {
            "status": "running",
            "current_batch": str(speed_batch.get("id") or ""),
            "current_job_id": job_id,
            "current_source_batch": str(source_batch.get("id") or ""),
            "last_launch_cores": cores,
            "last_launch_at": datetime.now().astimezone().isoformat(),
            "updated_at": datetime.now().astimezone().isoformat(),
            "batch_file": str(BATCH_FILE),
            "source_batch_file": str(SOURCE_BATCH_FILE),
        }
    )
    _save_json(STATE_FILE, state)
    if dry_run:
        print(json.dumps({"decision": "dry_run_launch", "speed_batch": speed_batch.get("id"), "source_batch": source_batch.get("id"), "job_id": job_id, "job_path": str(job_path)}, indent=2))
        return 0
    pid = _launch_job(job_path, str(REPO_ROOT / ".venv" / "Scripts" / "python.exe"))
    print(json.dumps({"decision": "launched", "speed_batch": speed_batch.get("id"), "source_batch": source_batch.get("id"), "job_id": job_id, "pid": pid, "cores": cores, "job_path": str(job_path)}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Keep isolated Sieve3 exit speed batches running.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return run_once(dry_run=bool(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from .entry_sieve_service import EntrySieveService, EntrySieveSettings


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Entry Sieve strategy batches sequentially.")
    parser.add_argument("--queue-file", required=True)
    return parser.parse_args(argv)


def _settings_from_payload(payload: dict[str, Any]) -> EntrySieveSettings:
    allowed = {field.name for field in fields(EntrySieveSettings)}
    data = {key: value for key, value in dict(payload or {}).items() if key in allowed}
    return EntrySieveSettings(**data)


def _update_queue(queue_file: Path, queue: dict[str, Any], **fields_to_update: Any) -> None:
    queue.update(fields_to_update)
    queue["updated_at"] = datetime.now().astimezone().isoformat()
    save_json(queue_file, queue)


def _run_batch_command(command: list[str], *, cwd: Path, log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8", errors="replace") as log_handle:
        log_handle.write(f"Started: {datetime.now().astimezone().isoformat()}\n")
        log_handle.write("Command: " + " ".join(command) + "\n\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
        return int(process.wait() or 0)


def _active_run_is_running(service: EntrySieveService, current_job_id: str = "") -> bool:
    return bool(service.live_runs(exclude_job_id=current_job_id))


def _wait_for_active_run(service: EntrySieveService, queue_file: Path, queue: dict[str, Any], poll_seconds: int) -> None:
    while _active_run_is_running(service):
        runs = service.live_runs()
        active_ids = ", ".join(str(run.get("job_id") or "") for run in runs)
        _update_queue(
            queue_file,
            queue,
            status="waiting",
            phase="waiting_for_active_run",
            active_job_id=active_ids,
            message=f"Waiting for active Entry Sieve run {active_ids}",
        )
        time.sleep(poll_seconds)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    queue_file = Path(args.queue_file).resolve()
    queue = load_json(queue_file, {})
    if not isinstance(queue, dict):
        raise SystemExit(f"Entry Sieve queue file is not an object: {queue_file}")

    app_dir = Path(str(queue.get("app_dir") or "")).resolve()
    python_exe = str(queue.get("python_exe") or sys.executable)
    service = EntrySieveService(app_dir, python_exe)
    base_settings = _settings_from_payload(queue.get("settings") if isinstance(queue.get("settings"), dict) else {})
    batch_ids = [str(batch_id).strip() for batch_id in queue.get("batch_ids") or [] if str(batch_id).strip()]
    poll_seconds = max(10, int(queue.get("poll_seconds") or 60))
    completed = list(queue.get("completed_batches") or [])
    failed = list(queue.get("failed_batches") or [])

    _update_queue(queue_file, queue, status="running", phase="starting", started_at=datetime.now().astimezone().isoformat())
    if bool(queue.get("wait_for_active", True)):
        _wait_for_active_run(service, queue_file, queue, poll_seconds)

    for index, batch_id in enumerate(batch_ids, start=1):
        if batch_id in completed:
            continue
        if bool(queue.get("wait_for_active", True)):
            _wait_for_active_run(service, queue_file, queue, poll_seconds)
        settings = EntrySieveSettings(**{**base_settings.__dict__, "strategy_batch": batch_id})
        _update_queue(queue_file, queue, status="running", phase="building_job", current_batch=batch_id, batch_index=index, batch_total=len(batch_ids))
        job_path = service.build_job(settings)
        job = load_json(job_path, {})
        job_id = str(job.get("job_id") or job_path.stem)
        command = [python_exe, "-u", "-m", "launcher_v2.services.entry_sieve_runner", "--job-file", str(job_path)]
        log_file = service.runtime_dir / "logs" / f"{job_id}.log"
        _update_queue(
            queue_file,
            queue,
            status="running",
            phase="running_batch",
            current_batch=batch_id,
            current_job_id=job_id,
            command=command,
            log_file=str(log_file),
            message=f"Running Entry Sieve batch {index}/{len(batch_ids)}: {batch_id}",
        )
        returncode = _run_batch_command(command, cwd=app_dir, log_file=log_file)
        if returncode != 0:
            failed.append({"batch": batch_id, "job_id": job_id, "returncode": returncode})
            _update_queue(queue_file, queue, status="failed", phase="failed", failed_batches=failed, message=f"Batch failed: {batch_id}")
            return int(returncode or 1)
        completed.append(batch_id)
        _update_queue(queue_file, queue, status="running", phase="batch_complete", completed_batches=completed, message=f"Completed batch {batch_id}")

    _update_queue(queue_file, queue, status="finished", phase="finished", finished_at=datetime.now().astimezone().isoformat(), completed_batches=completed, failed_batches=failed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

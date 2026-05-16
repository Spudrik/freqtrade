from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any


class EntrySieveLockError(RuntimeError):
    pass


def _process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        synchronize = 0x00100000
        wait_timeout = 0x00000102
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(synchronize, False, pid)
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == wait_timeout
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def lock_file(runtime_dir: Path) -> Path:
    return runtime_dir / "entry_sieve.lock"


def live_status_runs(runtime_dir: Path, *, exclude_job_id: str = "") -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    status_dir = runtime_dir / "status"
    if not status_dir.exists():
        return runs
    for path in status_dir.glob("*.json"):
        data = _read_json(path)
        if str(data.get("status") or "").lower() != "running":
            continue
        job_id = str(data.get("job_id") or path.stem)
        if exclude_job_id and job_id == exclude_job_id:
            continue
        try:
            pid = int(data.get("pid") or 0)
        except (TypeError, ValueError):
            pid = 0
        if not _process_is_running(pid):
            continue
        row = dict(data)
        row["job_id"] = job_id
        row["status_file"] = str(path)
        runs.append(row)
    return runs


def live_lock(runtime_dir: Path, *, exclude_job_id: str = "") -> dict[str, Any]:
    path = lock_file(runtime_dir)
    data = _read_json(path)
    if not data:
        return {}
    job_id = str(data.get("job_id") or "")
    if exclude_job_id and job_id == exclude_job_id:
        return {}
    try:
        pid = int(data.get("pid") or 0)
    except (TypeError, ValueError):
        pid = 0
    if _process_is_running(pid):
        data["path"] = str(path)
        return data
    try:
        path.unlink()
    except OSError:
        pass
    return {}


def live_entry_sieve_runs(runtime_dir: Path, *, exclude_job_id: str = "") -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    locked = live_lock(runtime_dir, exclude_job_id=exclude_job_id)
    if locked:
        locked = dict(locked)
        locked.setdefault("source", "lock")
        runs.append(locked)
    seen = {str(run.get("job_id") or "") for run in runs}
    for run in live_status_runs(runtime_dir, exclude_job_id=exclude_job_id):
        job_id = str(run.get("job_id") or "")
        if job_id in seen:
            continue
        run = dict(run)
        run.setdefault("source", "status")
        runs.append(run)
        seen.add(job_id)
    return runs


class EntrySieveRunLock(AbstractContextManager["EntrySieveRunLock"]):
    def __init__(self, runtime_dir: Path, *, job_id: str, owner: str = "runner") -> None:
        self.runtime_dir = runtime_dir
        self.job_id = job_id
        self.owner = owner
        self.path = lock_file(runtime_dir)
        self.acquired = False

    def __enter__(self) -> "EntrySieveRunLock":
        blockers = live_entry_sieve_runs(self.runtime_dir, exclude_job_id=self.job_id)
        if blockers:
            blocker = blockers[0]
            raise EntrySieveLockError(
                f"Entry Sieve is already running: {blocker.get('job_id')} "
                f"(pid {blocker.get('pid')}, source {blocker.get('source')})"
            )
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "job_id": self.job_id,
            "pid": os.getpid(),
            "owner": self.owner,
            "started_at": datetime.now().astimezone().isoformat(),
            "heartbeat_at": datetime.now().astimezone().isoformat(),
        }
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(str(self.path), flags)
        except FileExistsError:
            locked = live_lock(self.runtime_dir)
            if locked:
                raise EntrySieveLockError(
                    f"Entry Sieve is already running: {locked.get('job_id')} "
                    f"(pid {locked.get('pid')}, source lock)"
                ) from None
            return self.__enter__()
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2) + "\n")
        self.acquired = True
        return self

    def heartbeat(self, **fields: Any) -> None:
        if not self.acquired:
            return
        payload = _read_json(self.path)
        payload.update(fields)
        payload.update(
            {
                "job_id": self.job_id,
                "pid": os.getpid(),
                "owner": self.owner,
                "heartbeat_at": datetime.now().astimezone().isoformat(),
            }
        )
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not self.acquired:
            return
        payload = _read_json(self.path)
        if str(payload.get("job_id") or "") == self.job_id and int(payload.get("pid") or 0) == os.getpid():
            try:
                self.path.unlink()
            except OSError:
                pass
        self.acquired = False

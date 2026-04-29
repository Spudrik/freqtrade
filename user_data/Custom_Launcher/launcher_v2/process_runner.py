from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
import queue
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Iterable


def utf8_subprocess_env(extra: dict[str, str] | None = None, *, cwd: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if extra:
        env.update(extra)
    if cwd:
        project_root = str(Path(cwd).resolve())
        existing = env.get("PYTHONPATH", "")
        paths = existing.split(os.pathsep) if existing else []
        if project_root not in paths:
            env["PYTHONPATH"] = project_root + (os.pathsep + existing if existing else "")
    return env


@dataclass
class ProcessResult:
    returncode: int | None
    command: list[str]


class ProcessRunner:
    """Small subprocess runner used by LauncherV2.

    It forwards output into the shared queue as (stream_name, text) tuples.
    """

    def __init__(self, output_queue: "queue.Queue[tuple[str, str]]", *, log_dir: str | Path | None = None) -> None:
        self.output_queue = output_queue
        self.log_dir = Path(log_dir).expanduser() if log_dir is not None else None
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.last_command: list[str] = []
        self.output_owner = ""
        self.last_log_file: Path | None = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def run(self, command: Iterable[str], *, cwd: str | None = None, env: dict[str, str] | None = None, owner: str = "") -> None:
        if self.is_running():
            raise RuntimeError("A process is already running.")
        command_list = [str(item) for item in command]
        self.last_command = command_list
        self.output_owner = str(owner or "")
        self.last_log_file = self._start_log_file(command_list, cwd)
        resolved_cwd: str | None = None
        if cwd:
            candidate = Path(str(cwd).strip()).expanduser()
            if not candidate.exists():
                raise RuntimeError(f"Cannot start a process, the working directory '{candidate}' does not exist")
            resolved_cwd = str(candidate.resolve())
        popen_kwargs: dict[str, object] = {}
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True
        self.process = subprocess.Popen(
            command_list,
            cwd=resolved_cwd,
            env=utf8_subprocess_env(env, cwd=resolved_cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **popen_kwargs,
        )
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _start_log_file(self, command: list[str], cwd: str | None) -> Path | None:
        if self.log_dir is None:
            return None
        self.log_dir.mkdir(parents=True, exist_ok=True)
        owner = self.output_owner or "process"
        safe_owner = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in owner)
        path = self.log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_owner}.log"
        header = [
            f"Started: {datetime.now().astimezone().isoformat()}",
            f"Owner: {owner}",
            f"CWD: {cwd or ''}",
            "Command: " + " ".join(command),
            "",
        ]
        path.write_text("\n".join(header), encoding="utf-8")
        return path

    def _write_log(self, text: str) -> None:
        if self.last_log_file is None:
            return
        try:
            with self.last_log_file.open("a", encoding="utf-8", errors="replace") as handle:
                handle.write(text)
        except Exception:
            return

    def _reader(self) -> None:
        process = self.process
        if process is None:
            return
        try:
            assert process.stdout is not None
            for line in process.stdout:
                self._write_log(line)
                self.output_queue.put(("stdout", line))
            code = process.wait()
            status = f"Process exited with code {code}\n"
            self._write_log(status)
            self.output_queue.put(("status", status))
        except Exception as exc:
            error = f"Process reader error: {exc}\n"
            self._write_log(error)
            self.output_queue.put(("stderr", error))
        finally:
            if self.process is process:
                self.process = None

    def stop(self, *, timeout_seconds: float = 8.0) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        self._write_log("Stop requested; terminating process tree.\n")
        self.output_queue.put(("status", "Stop requested; terminating process tree.\n"))
        self._terminate_process_tree(process, force=False)
        try:
            process.wait(timeout=timeout_seconds)
        except Exception:
            self._terminate_process_tree(process, force=True)
            try:
                process.wait(timeout=2.0)
            except Exception:
                pass
        if process.poll() is None:
            try:
                process.kill()
            except Exception:
                pass
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _terminate_process_tree(self, process: subprocess.Popen[str], *, force: bool) -> None:
        if process.poll() is not None:
            return
        if sys.platform == "win32":
            self._terminate_windows_tree(process.pid, force=force)
            return
        self._terminate_posix_group(process.pid, force=force)

    def _terminate_windows_tree(self, pid: int, *, force: bool) -> None:
        command = ["taskkill", "/PID", str(pid), "/T"]
        if force:
            command.append("/F")
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
            if completed.stdout:
                self._write_log(completed.stdout)
        except Exception as exc:
            self._write_log(f"Process tree termination failed: {exc}\n")

    def _terminate_posix_group(self, pid: int, *, force: bool) -> None:
        signum = signal.SIGKILL if force else signal.SIGTERM
        try:
            os.killpg(os.getpgid(pid), signum)
        except ProcessLookupError:
            return
        except Exception as exc:
            self._write_log(f"Process group termination failed: {exc}\n")
            try:
                os.kill(pid, signum)
            except Exception:
                pass

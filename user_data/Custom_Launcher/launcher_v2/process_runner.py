from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
import queue
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
            self.process = None

    def stop(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

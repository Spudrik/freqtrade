"""Small shared runtime primitives for detached collectors."""

from __future__ import annotations

import contextlib
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable, Iterator


class CollectorBusyError(RuntimeError):
    """Raised when another process owns one of the collector resources."""


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace a UTF-8 text file using a same-directory temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def pid_create_time() -> float:
    """Return this process's psutil creation time for PID reuse checks."""
    import psutil

    return float(psutil.Process(os.getpid()).create_time())


def _canonical(path: Path) -> Path:
    return Path(os.path.normcase(str(path.expanduser().resolve(strict=False))))


@contextlib.contextmanager
def lock_resources(paths: Iterable[Path]) -> Iterator[None]:
    """Exclusively lock canonical resource paths until the context exits."""
    lock_paths = sorted({_canonical(Path(path)) for path in paths}, key=lambda item: str(item).casefold())
    handles: list[tuple[Path, object, bool]] = []
    try:
        for resource in lock_paths:
            lock_path = Path(f"{resource}.lock")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                handle = lock_path.open("a+b")
            except PermissionError as exc:
                raise CollectorBusyError(f"Collector resource is already owned or its lock is inaccessible: {resource}") from exc
            handles.append((lock_path, handle, False))
            if os.name == "nt":
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise CollectorBusyError(f"Collector resource is already owned: {resource}") from exc
            else:
                import fcntl

                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    raise CollectorBusyError(f"Collector resource is already owned: {resource}") from exc
            handles[-1] = (lock_path, handle, True)
        yield
    finally:
        for _, handle, acquired in reversed(handles):
            try:
                if acquired and os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                elif acquired:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            finally:
                handle.close()


def verified_worker(status: dict, module: str, db: Path, status_file: Path):
    """Return the live psutil process only when status and exact CLI identity agree."""
    import psutil

    pid = status.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        process = psutil.Process(pid)
        cmdline = process.cmdline()
        create_time = process.create_time()
    except psutil.NoSuchProcess:
        return None
    except (psutil.AccessDenied, OSError) as exc:
        raise RuntimeError(f"Cannot verify collector process identity for pid {pid}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Cannot read collector process identity for pid {pid}: {exc}") from exc

    try:
        expected_created = status["pid_create_time"]
        if isinstance(expected_created, bool):
            raise ValueError("Creation time cannot be a boolean")
        expected_created = float(expected_created)
        create_time = float(create_time)
        if not all(math.isfinite(value) and value > 0 for value in (expected_created, create_time)):
            raise ValueError("Creation times must be finite and positive")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"Invalid or missing pid_create_time for collector pid {pid}") from exc
    if abs(expected_created - create_time) > 0.01:
        return None

    expected_db = str(_canonical(Path(db)))
    expected_status = str(_canonical(Path(status_file)))
    module_token = str(module)
    module_matches = any(
        token == "-m" and index + 1 < len(cmdline) and cmdline[index + 1] == module_token
        for index, token in enumerate(cmdline)
    )
    if not module_matches:
        return None

    def option_value(option: str) -> str | None:
        for index, token in enumerate(cmdline):
            if token == option and index + 1 < len(cmdline):
                return cmdline[index + 1]
            if token.startswith(option + "="):
                return token.split("=", 1)[1]
        return None

    actual_db = option_value("--db")
    actual_status = option_value("--status-file")
    if actual_db is None or actual_status is None:
        return None
    if str(_canonical(Path(actual_db))) != expected_db or str(_canonical(Path(actual_status))) != expected_status:
        return None
    return process

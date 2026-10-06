from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
import csv
import json
import os
import sqlite3
import subprocess
import sys
import time
import webbrowser
from urllib.parse import urlparse

from user_data.Custom_Launcher.collector_runtime import CollectorBusyError, lock_resources, verified_worker
from ..preset_manager import AUTO_PRESET_NAME, collector_desired, read_presets, set_collector_desired


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def sqlite_readonly_connection(db_path: Path, *, timeout: float = 5.0) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", timeout=timeout, uri=True)
    try:
        yield conn
    finally:
        conn.close()


def export_query_csv_readonly(db_path: Path, csv_path: Path, query: str) -> int:
    if not db_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("", encoding="utf-8")
        return 0
    with sqlite_readonly_connection(db_path, timeout=10.0) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query).fetchall()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer: csv.DictWriter[str] | None = None
        for row in rows:
            data = dict(row)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(data.keys()))
                writer.writeheader()
            writer.writerow(data)
            count += 1
    return count


def resolve_app_path(app_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else Path(app_dir) / path).resolve()


def preferred_data_tools_python(app_dir: Path, fallback: str | None = None) -> str:
    if fallback is not None:
        configured = str(fallback).strip()
        if not configured:
            raise ValueError("Configured Python executable path is empty.")
        return str(resolve_app_path(app_dir, configured))
    project_root = Path(app_dir).resolve().parent.parent
    return str((project_root / ".venv" / "Scripts" / "python.exe").resolve())


def validate_python_exe(python_exe: str) -> str:
    path = Path(python_exe)
    if not path.is_file():
        raise FileNotFoundError(f"Configured Python executable not found: {path}")
    return str(path.resolve())


def data_tools_python_for_state(app_dir: Path, state: dict[str, Any], default: str | None) -> str:
    configured = state.get("python_exe") if "python_exe" in state else default
    return validate_python_exe(preferred_data_tools_python(app_dir, configured))


def validate_config_file(config_path: Path) -> Path:
    if not config_path.is_file():
        raise FileNotFoundError(f"Collector config file not found: {config_path}")
    return config_path


class CollectorStatusError(RuntimeError):
    pass


def _read_collector_status(status_path: Path) -> dict[str, Any]:
    if not status_path.exists():
        return {}
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CollectorStatusError(f"Could not read collector status {status_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CollectorStatusError(f"Collector status must contain a JSON object: {status_path}")
    return payload


def _resource_probe(db_path: Path, status_path: Path) -> None:
    try:
        with lock_resources([db_path, status_path]):
            pass
    except CollectorBusyError as exc:
        raise RuntimeError(
            f"Collector ownership is unknown for {status_path}; its database or status lock is held."
        ) from exc


def _process_option(process: Any, option: str) -> str | None:
    try:
        cmdline = process.cmdline()
    except Exception as exc:
        raise RuntimeError(f"Cannot inspect verified collector pid {process.pid}: {exc}") from exc
    for index, token in enumerate(cmdline):
        if token == option and index + 1 < len(cmdline):
            return cmdline[index + 1]
        if token.startswith(option + "="):
            return token.split("=", 1)[1]
    return None


def _preset_collector_state(app_dir: Path, key: str, preset_path: Path | None) -> dict[str, Any] | None:
    default_path = Path(app_dir) / "launcher_v2" / "config" / "presets.json"
    presets = read_presets(Path(preset_path) if preset_path is not None else default_path)
    auto_preset = presets.get(AUTO_PRESET_NAME, {})
    if not isinstance(auto_preset, dict):
        raise ValueError(f"Preset {AUTO_PRESET_NAME!r} must contain a JSON object.")
    return _collector_state_from_preset(auto_preset, key) or None


def _collector_state_from_preset(preset: dict[str, Any], key: str) -> dict[str, Any]:
    if key == "orderbook":
        field_names = ("data_dir", "config_path")
    else:
        field_names = ("data_dir", "config_path", "db_path")
    return {
        field: preset[f"{key}_{field}"]
        for field in field_names
        if f"{key}_{field}" in preset
    }


@contextmanager
def protect_collector_resource_changes(
    app_dir: Path, previous: dict[str, Any], updated: dict[str, Any]
) -> Iterator[None]:
    """Keep prior resource authority until a validated preset replacement completes."""
    from .global_context_service import GlobalContextService
    from .orderbook_service import OrderBookService

    research = ResearchCollectorService(app_dir)
    routes = (
        ("news", lambda state: research.paths(NEWS_PROFILE, **state), "collectors.context.news_research_collector"),
        ("web", lambda state: research.paths(WEB_PROFILE, **state), "collectors.context.web_research_collector"),
        ("global_context", GlobalContextService(app_dir).paths, "collectors.context.global_context_collector"),
        ("orderbook", OrderBookService(app_dir).paths, "collectors.orderbook.collector"),
    )
    merged = {**previous, **updated}
    with ExitStack() as resources:
        for key, paths_for_state, module in routes:
            prior_state = _collector_state_from_preset(previous, key)
            next_state = _collector_state_from_preset(merged, key)
            if key in {"news", "web"}:
                prior_state = {field: str(prior_state.get(field) or "") for field in ("data_dir", "config_path", "db_path")}
                next_state = {field: str(next_state.get(field) or "") for field in ("data_dir", "config_path", "db_path")}
            old_paths = paths_for_state(prior_state)
            new_paths = paths_for_state(next_state)
            if all(old_paths[part] == new_paths[part] for part in ("db", "status")):
                continue
            try:
                # Non-blocking: a start/stop holding launch while updating intent
                # causes rejection, never an inverted-lock wait.
                resources.enter_context(lock_resources([Path(f"{old_paths['status']}.launch")]))
                status = _read_collector_status(old_paths["status"])
                if verified_worker(status, module, old_paths["db"], old_paths["status"]) is not None:
                    raise RuntimeError("The prior collector resources are actively owned.")
                resources.enter_context(lock_resources([old_paths["db"], old_paths["status"]]))
            except (CollectorBusyError, CollectorStatusError, RuntimeError) as exc:
                raise RuntimeError(
                    f"Cannot change {key} database/status resource paths: {exc} Stop the collector first and wait for it to release its resources."
                ) from exc
        yield


def _request_collector_stop(
    *,
    app_dir: Path,
    key: str,
    preset_path: Path | None,
    state: dict[str, Any],
    paths_for_state: Any,
    module: str,
) -> Path:
    current_paths = paths_for_state(state)
    try:
        with lock_resources([Path(f"{current_paths['status']}.launch")]):
            set_collector_desired(app_dir, key, False, preset_path)
            candidates = [current_paths]
            previous = _preset_collector_state(app_dir, key, preset_path)
            if previous is not None:
                previous_paths = paths_for_state(previous)
                if any(previous_paths[part] != current_paths[part] for part in ("db", "status")):
                    candidates.append(previous_paths)

            owners: dict[int, Path] = {}
            for paths in candidates:
                status = _read_collector_status(paths["status"])
                process = verified_worker(status, module, paths["db"], paths["status"])
                if process is not None:
                    stop_value = _process_option(process, "--stop-file")
                    if not stop_value:
                        raise RuntimeError(f"Verified collector pid {process.pid} has no --stop-file argument.")
                    owners[int(process.pid)] = Path(stop_value)
                else:
                    _resource_probe(paths["db"], paths["status"])

            if len(owners) > 1:
                raise RuntimeError(
                    f"Multiple verified {key} collector owners use different resource paths; refusing to guess which one to stop."
                )
            stop_path = next(iter(owners.values())) if owners else current_paths["stop"]
            stop_path.parent.mkdir(parents=True, exist_ok=True)
            stop_path.write_text(utc_now() + "\n", encoding="utf-8")
            return stop_path
    except CollectorBusyError as exc:
        raise RuntimeError(f"Cannot safely stop {key} while its startup transition is in progress.") from exc


def _start_collector_detached(
    *,
    app_dir: Path,
    key: str,
    module: str,
    db_path: Path,
    status_path: Path,
    stop_path: Path,
    log_path: Path,
    command: Any,
    automatic: bool,
    preset_path: Path | None,
    prepare: Any = None,
) -> int | None:
    if automatic:
        if not collector_desired(app_dir, key, preset_path):
            return None
    launch_lock = Path(f"{status_path}.launch")
    try:
        with lock_resources([launch_lock]):
            if automatic and not collector_desired(app_dir, key, preset_path):
                return None
            if not automatic:
                set_collector_desired(app_dir, key, True, preset_path)
            try:
                status = _read_collector_status(status_path)
            except CollectorStatusError:
                _resource_probe(db_path, status_path)
                status = {}
            process = verified_worker(status, module, db_path, status_path)
            if process is not None:
                if stop_path.exists() or str(status.get("status") or "").lower() in {"stopping", "stop_requested"}:
                    if automatic:
                        return None
                    raise RuntimeError(f"The {key} collector is stopping; wait for it to release its resources before starting it again.")
                return int(process.pid)
            _resource_probe(db_path, status_path)
            if automatic and stop_path.exists():
                return None
            if prepare is not None:
                prepare()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stop_path.unlink(missing_ok=True)
            launch_started_at = time.time()
            with open(log_path, "ab") as log_handle:
                kwargs: dict[str, Any] = {
                    "stdin": subprocess.DEVNULL,
                    "stdout": log_handle,
                    "stderr": subprocess.STDOUT,
                    "cwd": str(app_dir),
                    "env": utf8_subprocess_env(Path(app_dir)),
                    "close_fds": True,
                }
                if os.name == "nt":
                    kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
                else:
                    kwargs["start_new_session"] = True
                spawned = subprocess.Popen(command() if callable(command) else command, **kwargs)

            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                status = _read_collector_status(status_path)
                if spawned.poll() is not None:
                    startup_text = str(status.get("started_at") or status.get("heartbeat_at") or "")
                    try:
                        startup_time = datetime.fromisoformat(startup_text.replace("Z", "+00:00"))
                        if startup_time.tzinfo is None:
                            startup_time = startup_time.replace(tzinfo=timezone.utc)
                        fresh_startup_status = startup_time.timestamp() >= launch_started_at - 0.1
                    except (TypeError, ValueError):
                        fresh_startup_status = False
                    if fresh_startup_status and str(status.get("status_reason") or "").lower() == "startup_error":
                        raise RuntimeError(f"The {key} collector failed during startup: {status.get('last_error') or 'unknown startup error'}")
                    try:
                        created_at = float(status.get("pid_create_time"))
                    except (TypeError, ValueError):
                        created_at = 0.0
                    process = verified_worker(status, module, db_path, status_path)
                    if (
                        str(status.get("status") or "").lower() == "running"
                        and created_at >= launch_started_at - 0.1
                        and process is not None
                    ):
                        return int(process.pid)
                    raise RuntimeError(f"The {key} collector exited before publishing a fresh verified running status.")
                try:
                    created_at = float(status.get("pid_create_time"))
                except (TypeError, ValueError):
                    created_at = 0.0
                if (
                    str(status.get("status") or "").lower() == "running"
                    and created_at >= launch_started_at - 0.1
                ):
                    process = verified_worker(status, module, db_path, status_path)
                    if process is not None:
                        return int(process.pid)
                time.sleep(0.1)
            raise RuntimeError(f"Timed out waiting for the {key} collector to publish a verified running status.")
    except CollectorBusyError as exc:
        raise RuntimeError(f"Cannot safely start {key} collector because its launch lock is unavailable: {exc}") from exc


def utf8_subprocess_env(app_dir: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if app_dir is not None:
        project_root = str(Path(app_dir).resolve().parents[1])
        inherited_pythonpath = str(env.get("PYTHONPATH") or "").strip()
        env["PYTHONPATH"] = os.pathsep.join(
            [project_root, inherited_pythonpath] if inherited_pythonpath else [project_root]
        )
    return env


def parse_minutes_to_seconds(value: str, default_minutes: int) -> int:
    text = str(value or "").strip()
    minutes = default_minutes
    if text:
        try:
            minutes = int(round(float(text)))
        except Exception:
            minutes = default_minutes
    return max(1, minutes) * 60


def is_process_running(pid: int) -> bool:
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
        return True
    except OSError:
        return False
    except Exception:
        return False


def open_path(path: Path) -> None:
    path = path.resolve()
    if path.is_dir():
        webbrowser.open(path.as_uri())
        return
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        webbrowser.open(path.as_uri())


@dataclass(frozen=True)
class CollectorProfile:
    key: str
    label: str
    collector_file: str
    sources_file: str
    data_dir_name: str
    db_file: str
    log_file: str
    default_interval_minutes: int


NEWS_PROFILE = CollectorProfile(
    key="news",
    label="News Lab",
    collector_file="collectors/context/news_research_collector.py",
    sources_file="collectors/context/config/news_research_sources.json",
    data_dir_name="../collector_data/news",
    db_file="news_events.sqlite",
    log_file="news_collector.log",
    default_interval_minutes=15,
)


WEB_PROFILE = CollectorProfile(
    key="web",
    label="Web Lab",
    collector_file="collectors/context/web_research_collector.py",
    sources_file="collectors/context/config/web_research_sources.json",
    data_dir_name="../collector_data/web",
    db_file="web_events.sqlite",
    log_file="web_collector.log",
    default_interval_minutes=360,
)


class ResearchCollectorService:
    def __init__(self, app_dir: Path, python_exe: str | None = None) -> None:
        self.app_dir = Path(app_dir).resolve()
        self.python_exe = preferred_data_tools_python(self.app_dir, python_exe)

    def app_path(self, value: str | Path) -> Path:
        return resolve_app_path(self.app_dir, value)

    def _resolve_data_dir(self, profile: CollectorProfile, data_dir: str) -> Path:
        text = str(data_dir or "").strip()
        default_dir = self.app_path(profile.data_dir_name)
        if not text:
            return default_dir
        return self.app_path(text)

    def _resolve_config_path(self, profile: CollectorProfile, config_path: str) -> Path:
        text = str(config_path or "").strip()
        default_config = self.app_path(profile.sources_file)
        if not text:
            return default_config
        return self.app_path(text)

    def _resolve_db_path(self, profile: CollectorProfile, db_path: str, resolved_data_dir: Path) -> Path:
        text = str(db_path or "").strip()
        default_db = resolved_data_dir / profile.db_file
        if not text:
            return default_db
        return self.app_path(text)

    def paths(self, profile: CollectorProfile, data_dir: str, config_path: str, db_path: str) -> dict[str, Path]:
        resolved_data_dir = self._resolve_data_dir(profile, data_dir)
        resolved_config = self._resolve_config_path(profile, config_path)
        resolved_db = self._resolve_db_path(profile, db_path, resolved_data_dir)
        return {
            "collector": self.app_path(profile.collector_file),
            "config": resolved_config,
            "data_dir": resolved_data_dir,
            "db": resolved_db,
            "status": resolved_data_dir / "collector_status.json",
            "pid": resolved_data_dir / "collector.pid",
            "stop": resolved_data_dir / "collector.stop",
            "log": resolved_data_dir / "logs" / profile.log_file,
        }

    def build_command(self, profile: CollectorProfile, state: dict[str, Any]) -> list[str]:
        paths = self.paths(profile, str(state.get("data_dir") or ""), str(state.get("config_path") or ""), str(state.get("db_path") or ""))
        python_exe = data_tools_python_for_state(self.app_dir, state, self.python_exe)
        if not paths["collector"].exists():
            raise FileNotFoundError(paths["collector"])
        validate_config_file(paths["config"])
        command = [
            python_exe,
            "-u",
            "-m",
            f"collectors.context.{Path(profile.collector_file).stem}",
            "--config",
            str(paths["config"]),
            "--data-dir",
            str(paths["data_dir"]),
            "--db",
            str(paths["db"]),
            "--status-file",
            str(paths["status"]),
            "--pid-file",
            str(paths["pid"]),
            "--log-file",
            str(paths["log"]),
            "--stop-file",
            str(paths["stop"]),
            "--interval-seconds",
            str(parse_minutes_to_seconds(str(state.get("interval_minutes") or ""), profile.default_interval_minutes)),
        ]
        if state.get("once"):
            command.append("--once")
        return command

    def start_detached(
        self,
        profile: CollectorProfile,
        state: dict[str, Any],
        *,
        automatic: bool = False,
        preset_path: Path | None = None,
    ) -> int | None:
        paths = self.paths(profile, str(state.get("data_dir") or ""), str(state.get("config_path") or ""), str(state.get("db_path") or ""))
        return _start_collector_detached(
            app_dir=self.app_dir,
            key=profile.key,
            module=f"collectors.context.{Path(profile.collector_file).stem}",
            db_path=paths["db"],
            status_path=paths["status"],
            stop_path=paths["stop"],
            log_path=paths["log"],
            command=lambda: self.build_command(profile, state),
            automatic=automatic,
            preset_path=preset_path,
        )

    def request_stop(self, profile: CollectorProfile, state: dict[str, Any], *, preset_path: Path | None = None) -> Path:
        paths = self.paths(profile, str(state.get("data_dir") or ""), str(state.get("config_path") or ""), str(state.get("db_path") or ""))
        return _request_collector_stop(
            app_dir=self.app_dir,
            key=profile.key,
            preset_path=preset_path,
            state=state,
            paths_for_state=lambda candidate: self.paths(
                profile,
                str(candidate.get("data_dir") or ""),
                str(candidate.get("config_path") or ""),
                str(candidate.get("db_path") or ""),
            ),
            module=f"collectors.context.{Path(profile.collector_file).stem}",
        )

    def read_status(self, profile: CollectorProfile, state: dict[str, Any]) -> dict[str, Any]:
        paths = self.paths(profile, str(state.get("data_dir") or ""), str(state.get("config_path") or ""), str(state.get("db_path") or ""))
        status = _read_collector_status(paths["status"])
        pid_text = ""
        if paths["pid"].exists():
            try:
                pid_text = paths["pid"].read_text(encoding="utf-8").strip()
            except Exception:
                pid_text = ""
        if pid_text:
            status["pid_text"] = pid_text
        process = verified_worker(
            status,
            f"collectors.context.{Path(profile.collector_file).stem}",
            paths["db"],
            paths["status"],
        )
        status["verified_running"] = process is not None
        status["verified_pid"] = int(process.pid) if process is not None else None
        return status

    def source_health_rows(self, profile: CollectorProfile, state: dict[str, Any]) -> list[tuple[Any, ...]]:
        paths = self.paths(profile, str(state.get("data_dir") or ""), str(state.get("config_path") or ""), str(state.get("db_path") or ""))
        if not paths["db"].exists():
            return []
        with sqlite_readonly_connection(paths["db"]) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT source_id, source_group, enabled, market_relevance, last_success_at,
                       last_failure_at, last_error, items_last_fetch, inserted_last_fetch,
                       duplicates_last_fetch
                FROM sources
                ORDER BY source_group, source_id
                """
            ).fetchall()
        return [
            (
                row["source_id"],
                row["source_group"],
                row["enabled"],
                row["market_relevance"],
                row["last_success_at"],
                row["last_failure_at"],
                row["last_error"],
                row["items_last_fetch"],
                row["inserted_last_fetch"],
                row["duplicates_last_fetch"],
            )
            for row in rows
        ]

    def recent_articles_rows(
        self, profile: CollectorProfile, state: dict[str, Any], *, limit: int = 200
    ) -> list[dict[str, Any]]:
        paths = self.paths(profile, str(state.get("data_dir") or ""), str(state.get("config_path") or ""), str(state.get("db_path") or ""))
        if not paths["db"].exists():
            return []
        query_limit = max(1, min(int(limit), 1000))
        try:
            with sqlite_readonly_connection(paths["db"]) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT
                        COALESCE(published_at, collected_at, '') AS article_time,
                        source_id,
                        market_relevance,
                        title,
                        COALESCE(canonical_url, source_url, '') AS article_url
                    FROM articles
                    ORDER BY COALESCE(published_at, collected_at) DESC, collected_at DESC
                    LIMIT ?
                    """,
                    (query_limit,),
                ).fetchall()
        except sqlite3.Error:
            return []
        return [
            {
                "article_time": row["article_time"] or "",
                "source_id": row["source_id"] or "",
                "market_relevance": row["market_relevance"] or "",
                "title": row["title"] or "",
                "article_url": row["article_url"] or "",
            }
            for row in rows
        ]

    def export_articles_csv(self, profile: CollectorProfile, state: dict[str, Any]) -> tuple[Path, int]:
        paths = self.paths(profile, str(state.get("data_dir") or ""), str(state.get("config_path") or ""), str(state.get("db_path") or ""))
        export_dir = paths["data_dir"] / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        csv_path = export_dir / f"{profile.key}_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        count = export_query_csv_readonly(paths["db"], csv_path, "SELECT * FROM articles ORDER BY collected_at DESC, title ASC")
        return csv_path, count

    def export_source_health_csv(self, profile: CollectorProfile, state: dict[str, Any]) -> tuple[Path, int]:
        paths = self.paths(profile, str(state.get("data_dir") or ""), str(state.get("config_path") or ""), str(state.get("db_path") or ""))
        export_dir = paths["data_dir"] / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        filename = "source_health" if profile.key == "news" else "web_source_health"
        csv_path = export_dir / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        count = export_query_csv_readonly(paths["db"], csv_path, "SELECT * FROM sources ORDER BY source_group, source_id")
        return csv_path, count

    def open_data_folder(self, profile: CollectorProfile, state: dict[str, Any]) -> None:
        paths = self.paths(profile, str(state.get("data_dir") or ""), str(state.get("config_path") or ""), str(state.get("db_path") or ""))
        paths["data_dir"].mkdir(parents=True, exist_ok=True)
        open_path(paths["data_dir"])

    def open_log(self, profile: CollectorProfile, state: dict[str, Any]) -> None:
        paths = self.paths(profile, str(state.get("data_dir") or ""), str(state.get("config_path") or ""), str(state.get("db_path") or ""))
        paths["log"].parent.mkdir(parents=True, exist_ok=True)
        paths["log"].touch(exist_ok=True)
        open_path(paths["log"])

    def open_article_url(self, url: str) -> None:
        raw = str(url or "").strip()
        if not raw:
            raise ValueError("No article URL available for this row.")
        parsed = urlparse(raw)
        if not parsed.scheme:
            raw = f"https://{raw}"
            parsed = urlparse(raw)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
        webbrowser.open(raw)

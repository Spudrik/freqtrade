from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os
import subprocess
import sys

from .collector_service import NEWS_PROFILE, WEB_PROFILE, ResearchCollectorService, is_process_running, open_path, utc_now
from .global_context_service import GlobalContextService
from .orderbook_service import OrderBookService


DEFAULT_TASK_NAME = "FreqtradeDataToolsWatchdog"
DEFAULT_CHECK_INTERVAL_MINUTES = 60
DEFAULT_HEARTBEAT_STALE_MINUTES = 10
SERVICE_KEYS = ("news", "web", "global_context", "orderbook")


@dataclass(frozen=True)
class ToolSpec:
    key: str
    label: str


TOOL_SPECS = {
    "news": ToolSpec("news", "News"),
    "web": ToolSpec("web", "Web"),
    "global_context": ToolSpec("global_context", "Global Context"),
    "orderbook": ToolSpec("orderbook", "Order Book"),
}


class DataToolsWatchdogService:
    """Checks detached data collectors and restarts dead processes.

    The watchdog is intentionally one-shot friendly so Windows Task Scheduler can
    run it periodically. Task Scheduler is the durable supervisor; this service
    just performs one health pass and records incidents.
    """

    def __init__(self, app_dir: Path, python_exe: str | None = None) -> None:
        self.app_dir = Path(app_dir)
        self.python_exe = python_exe or sys.executable
        self.runtime_dir = self.app_dir / "launcher_v2" / "runtime"
        self.status_path = self.runtime_dir / "data_tools_watchdog_status.json"
        self.events_path = self.runtime_dir / "data_tools_watchdog_events.jsonl"
        self.research_service = ResearchCollectorService(self.app_dir, self.python_exe)
        self.global_service = GlobalContextService(self.app_dir, self.python_exe)
        self.orderbook_service = OrderBookService(self.app_dir, self.python_exe)

    def default_state(self) -> dict[str, Any]:
        return {
            "task_name": DEFAULT_TASK_NAME,
            "check_interval_minutes": str(DEFAULT_CHECK_INTERVAL_MINUTES),
            "heartbeat_stale_minutes": str(DEFAULT_HEARTBEAT_STALE_MINUTES),
            "restart_dead": True,
            "services": list(SERVICE_KEYS),
        }

    def preset_path(self) -> Path:
        return self.app_dir / "launcher_v2" / "config" / "presets.json"

    def load_auto_preset(self, preset_path: str | Path | None = None) -> dict[str, Any]:
        path = Path(preset_path) if preset_path else self.preset_path()
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {}
        preset = payload.get("LauncherV2-auto")
        if isinstance(preset, dict):
            return preset
        for value in payload.values():
            if isinstance(value, dict):
                return value
        return {}

    def normalize_state(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        defaults = self.default_state()
        merged = {**defaults, **(state or {})}
        services = merged.get("services")
        if isinstance(services, str):
            selected = [part.strip() for part in services.replace(",", " ").split() if part.strip()]
        elif isinstance(services, list):
            selected = [str(item).strip() for item in services if str(item).strip()]
        else:
            selected = list(SERVICE_KEYS)
        merged["services"] = [key for key in SERVICE_KEYS if key in set(selected)]
        merged["task_name"] = str(merged.get("task_name") or DEFAULT_TASK_NAME).strip() or DEFAULT_TASK_NAME
        merged["check_interval_minutes"] = str(_positive_int(merged.get("check_interval_minutes"), DEFAULT_CHECK_INTERVAL_MINUTES))
        merged["heartbeat_stale_minutes"] = str(_positive_int(merged.get("heartbeat_stale_minutes"), DEFAULT_HEARTBEAT_STALE_MINUTES))
        merged["restart_dead"] = bool(merged.get("restart_dead", True))
        return merged

    def run_once(self, state: dict[str, Any] | None = None, *, preset_path: str | Path | None = None) -> dict[str, Any]:
        state = self.normalize_state(state)
        preset = self.load_auto_preset(preset_path)
        heartbeat_stale_minutes = _positive_int(state.get("heartbeat_stale_minutes"), DEFAULT_HEARTBEAT_STALE_MINUTES)
        restart_dead = bool(state.get("restart_dead", True))
        selected = set(state.get("services") or SERVICE_KEYS)
        rows: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []

        for key in SERVICE_KEYS:
            if key not in selected:
                continue
            row = self._check_tool(key, preset, heartbeat_stale_minutes, restart_dead)
            rows.append(row)
            if row.get("event"):
                events.append(row["event"])

        payload = {
            "checked_at": utc_now(),
            "heartbeat_stale_minutes": heartbeat_stale_minutes,
            "restart_dead": restart_dead,
            "rows": rows,
            "events": events,
        }
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        for event in events:
            self._append_event(event)
        return payload

    def _check_tool(self, key: str, preset: dict[str, Any], heartbeat_stale_minutes: int, restart_dead: bool) -> dict[str, Any]:
        spec = TOOL_SPECS[key]
        try:
            state = self._collector_state(key, preset)
            status = self._read_status(key, state)
            pid = _status_pid(status)
            running = bool(pid and is_process_running(pid))
            heartbeat = _status_heartbeat(status)
            effective_stale_minutes = _effective_stale_minutes(key, state, heartbeat_stale_minutes)
            stale = _is_stale(heartbeat, effective_stale_minutes)
            last_error = str(status.get("last_error") or "")
            action = "none"
            event: dict[str, Any] | None = None

            if not running and restart_dead:
                new_pid = self._start_tool(key, state, preset)
                action = f"restarted pid {new_pid}"
                event = _event(spec.label, "restart", f"{spec.label} was stopped or stale; restarted PID {new_pid}.", pid, new_pid)
                pid = new_pid
                running = is_process_running(pid)
                status = self._read_status(key, state)
                heartbeat = _status_heartbeat(status) or heartbeat
            elif not running:
                action = "dead"
                event = _event(spec.label, "dead", f"{spec.label} is not running.", pid, None)
            elif stale:
                action = "stale_heartbeat"
                event = _event(spec.label, "stale_heartbeat", f"{spec.label} heartbeat is stale but the process is still alive.", pid, None)
            elif last_error:
                action = "warning_last_error"

            return {
                "key": key,
                "label": spec.label,
                "pid": pid,
                "running": running,
                "status": status.get("status") or "-",
                "heartbeat_at": heartbeat or "",
                "heartbeat_stale_minutes": effective_stale_minutes,
                "last_fetch_at": status.get("last_fetch_at") or status.get("last_message_at") or status.get("last_metric_at") or "",
                "last_error": last_error,
                "action": action,
                "event": event,
            }
        except Exception as exc:
            event = _event(spec.label, "error", f"{spec.label} watchdog check failed: {exc}", None, None)
            return {
                "key": key,
                "label": spec.label,
                "pid": None,
                "running": False,
                "status": "watchdog_error",
                "heartbeat_at": "",
                "last_fetch_at": "",
                "last_error": str(exc),
                "action": "watchdog_error",
                "event": event,
            }

    def _collector_state(self, key: str, preset: dict[str, Any]) -> dict[str, Any]:
        if key == "news":
            return {
                "config_path": preset.get("news_config_path"),
                "data_dir": preset.get("news_data_dir"),
                "db_path": preset.get("news_db_path"),
                "interval_minutes": preset.get("news_interval_minutes"),
                "once": False,
            }
        if key == "web":
            return {
                "config_path": preset.get("web_config_path"),
                "data_dir": preset.get("web_data_dir"),
                "db_path": preset.get("web_db_path"),
                "interval_minutes": preset.get("web_interval_minutes"),
                "once": False,
            }
        if key == "global_context":
            return {
                "config_path": preset.get("global_context_config_path"),
                "data_dir": preset.get("global_context_data_dir"),
                "db_path": preset.get("global_context_db_path"),
                "key_file": preset.get("global_context_key_file"),
                "fred_key_json_path": preset.get("global_context_fred_key_json_path"),
                "enable_fred": bool(preset.get("global_context_enable_fred", True)),
                "interval_minutes": preset.get("global_context_interval_minutes"),
                "once": False,
            }
        if key == "orderbook":
            return {
                "config_path": preset.get("orderbook_config_path"),
                "data_dir": preset.get("orderbook_data_dir"),
                "market_profiles": preset.get("orderbook_market_profiles"),
                "depth_levels": preset.get("orderbook_depth_levels"),
                "stream_update_ms": preset.get("orderbook_stream_update_ms"),
                "metric_interval_seconds": preset.get("orderbook_metric_interval_seconds"),
                "context_poll_seconds": preset.get("orderbook_context_poll_seconds"),
                "context_period": preset.get("orderbook_context_period"),
                "snapshot_interval_seconds": preset.get("orderbook_snapshot_interval_seconds"),
                "capacity_warning_mb": preset.get("orderbook_capacity_warning_mb"),
                "capacity_critical_mb": preset.get("orderbook_capacity_critical_mb"),
                "max_symbols": preset.get("orderbook_max_symbols"),
                "store_snapshots": bool(preset.get("orderbook_store_snapshots", False)),
            }
        raise KeyError(key)

    def _read_status(self, key: str, state: dict[str, Any]) -> dict[str, Any]:
        if key == "news":
            return self.research_service.read_status(NEWS_PROFILE, state)
        if key == "web":
            return self.research_service.read_status(WEB_PROFILE, state)
        if key == "global_context":
            return self.global_service.read_status(state)
        if key == "orderbook":
            return self.orderbook_service.read_status(state)
        raise KeyError(key)

    def _start_tool(self, key: str, state: dict[str, Any], preset: dict[str, Any]) -> int:
        if key == "news":
            return int(self.research_service.start_detached(NEWS_PROFILE, state))
        if key == "web":
            return int(self.research_service.start_detached(WEB_PROFILE, state))
        if key == "global_context":
            return int(self.global_service.start_detached(state))
        if key == "orderbook":
            return int(self.orderbook_service.start_detached(state, _pairs_from_preset(preset)))
        raise KeyError(key)

    def read_latest_status(self) -> dict[str, Any]:
        if not self.status_path.exists():
            return {}
        try:
            loaded = json.loads(self.status_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def read_events(self, limit: int = 200) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.events_path.read_text(encoding="utf-8", errors="replace").splitlines()[-max(1, limit) :]:
            try:
                loaded = json.loads(line)
            except Exception:
                continue
            if isinstance(loaded, dict):
                rows.append(loaded)
        return rows

    def open_events_log(self) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.touch(exist_ok=True)
        open_path(self.events_path)

    def _append_event(self, event: dict[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=False) + "\n")

    def build_runner_command(self, state: dict[str, Any] | None = None) -> str:
        state = self.normalize_state(state)
        runner = self.app_dir / "data_tools_watchdog.py"
        parts = [
            _quote(str(self.python_exe)),
            _quote(str(runner)),
            "--once",
            "--preset",
            _quote(str(self.preset_path())),
            "--heartbeat-stale-minutes",
            str(_positive_int(state.get("heartbeat_stale_minutes"), DEFAULT_HEARTBEAT_STALE_MINUTES)),
            "--services",
            ",".join(state.get("services") or SERVICE_KEYS),
        ]
        if not state.get("restart_dead", True):
            parts.append("--no-restart-dead")
        return " ".join(parts)

    def task_wrapper_path(self, task_name: str | None = None) -> Path:
        safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(task_name or DEFAULT_TASK_NAME))
        return self.runtime_dir / f"{safe_name}.cmd"

    def task_hidden_wrapper_path(self, task_name: str | None = None) -> Path:
        safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(task_name or DEFAULT_TASK_NAME))
        return self.runtime_dir / f"{safe_name}.vbs"

    def write_task_wrapper(self, state: dict[str, Any] | None = None) -> Path:
        state = self.normalize_state(state)
        wrapper = self.task_wrapper_path(str(state["task_name"]))
        log_path = self.runtime_dir / "data_tools_watchdog_task.log"
        command = self.build_runner_command(state)
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        wrapper.write_text(
            "@echo off\n"
            f"{command} >> {_quote(str(log_path))} 2>&1\n",
            encoding="utf-8",
        )
        return wrapper

    def write_hidden_task_wrapper(self, state: dict[str, Any] | None = None) -> Path:
        state = self.normalize_state(state)
        command_wrapper = self.write_task_wrapper(state)
        hidden_wrapper = self.task_hidden_wrapper_path(str(state["task_name"]))
        hidden_wrapper.parent.mkdir(parents=True, exist_ok=True)
        hidden_wrapper.write_text(
            'Set shell = CreateObject("WScript.Shell")\n'
            f'shell.Run {_vbs_quote(str(command_wrapper))}, 0, True\n',
            encoding="ascii",
        )
        return hidden_wrapper

    def install_scheduled_task(self, state: dict[str, Any] | None = None) -> subprocess.CompletedProcess[str]:
        if os.name != "nt":
            raise RuntimeError("Windows Task Scheduler is only available on Windows.")
        state = self.normalize_state(state)
        task_name = str(state["task_name"])
        interval_minutes = _positive_int(state.get("check_interval_minutes"), DEFAULT_CHECK_INTERVAL_MINUTES)
        hidden_wrapper = self.write_hidden_task_wrapper(state)
        action = f'C:\\Windows\\System32\\wscript.exe //B //NoLogo "{hidden_wrapper}"'
        hourly = subprocess.run(
            ["schtasks", "/Create", "/TN", task_name, "/SC", "MINUTE", "/MO", str(interval_minutes), "/TR", action, "/F"],
            text=True,
            capture_output=True,
            check=False,
        )
        startup_entry = self.write_startup_entry(state)
        stdout = (
            f"[{task_name}]\n{hourly.stdout or ''}\n"
            f"[startup_entry]\nCreated {startup_entry}\n"
        )
        stderr = (
            f"[{task_name}]\n{hourly.stderr or ''}\n"
            "[startup_entry]\n"
        )
        return subprocess.CompletedProcess(
            args=["schtasks", task_name, str(startup_entry)],
            returncode=0 if hourly.returncode == 0 else 1,
            stdout=stdout,
            stderr=stderr,
        )

    def remove_scheduled_task(self, task_name: str | None = None) -> subprocess.CompletedProcess[str]:
        if os.name != "nt":
            raise RuntimeError("Windows Task Scheduler is only available on Windows.")
        name = str(task_name or DEFAULT_TASK_NAME).strip() or DEFAULT_TASK_NAME
        hourly = subprocess.run(["schtasks", "/Delete", "/TN", name, "/F"], text=True, capture_output=True, check=False)
        startup_entry = self.startup_entry_path(name)
        if startup_entry.exists():
            startup_entry.unlink()
        returncode = 0 if hourly.returncode == 0 else 1
        return subprocess.CompletedProcess(
            args=["schtasks", name, str(startup_entry)],
            returncode=returncode,
            stdout=f"[{name}]\n{hourly.stdout or ''}\n[startup_entry]\nRemoved {startup_entry}\n",
            stderr=f"[{name}]\n{hourly.stderr or ''}\n[startup_entry]\n",
        )

    def query_scheduled_task(self, task_name: str | None = None) -> subprocess.CompletedProcess[str]:
        if os.name != "nt":
            raise RuntimeError("Windows Task Scheduler is only available on Windows.")
        name = str(task_name or DEFAULT_TASK_NAME).strip() or DEFAULT_TASK_NAME
        hourly = subprocess.run(["schtasks", "/Query", "/TN", name, "/FO", "LIST", "/V"], text=True, capture_output=True, check=False)
        startup_entry = self.startup_entry_path(name)
        return subprocess.CompletedProcess(
            args=["schtasks", name, str(startup_entry)],
            returncode=hourly.returncode,
            stdout=f"[{name}]\n{hourly.stdout or ''}\n[startup_entry]\n{startup_entry} | exists={startup_entry.exists()}\n",
            stderr=f"[{name}]\n{hourly.stderr or ''}\n[startup_entry]\n",
        )

    def startup_dir(self) -> Path:
        appdata = str(os.environ.get("APPDATA") or "").strip()
        if not appdata:
            raise RuntimeError("APPDATA is not defined for the current user.")
        return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"

    def startup_entry_path(self, task_name: str) -> Path:
        safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(task_name or DEFAULT_TASK_NAME))
        return self.startup_dir() / f"{safe_name}_OnLogon.vbs"

    def write_startup_entry(self, state: dict[str, Any] | None = None) -> Path:
        state = self.normalize_state(state)
        source_wrapper = self.write_hidden_task_wrapper(state)
        startup_entry = self.startup_entry_path(str(state["task_name"]))
        startup_entry.parent.mkdir(parents=True, exist_ok=True)
        startup_entry.write_text(source_wrapper.read_text(encoding="ascii"), encoding="ascii")
        return startup_entry

def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(round(float(str(value).strip())))
    except Exception:
        parsed = default
    return max(1, parsed)


def _pairs_from_preset(preset: dict[str, Any]) -> list[str]:
    text = str(preset.get("pairs") or "")
    return [part.strip() for part in text.replace(",", "\n").splitlines() if part.strip()]


def _status_pid(status: dict[str, Any]) -> int | None:
    for key in ("pid_text", "pid"):
        try:
            pid = int(str(status.get(key) or "").strip())
        except Exception:
            continue
        return pid
    return None


def _status_heartbeat(status: dict[str, Any]) -> str:
    return str(status.get("heartbeat_at") or status.get("last_heartbeat_at") or "")


def _effective_stale_minutes(key: str, state: dict[str, Any], minimum_minutes: int) -> int:
    if key in {"news", "web", "global_context"}:
        interval = _positive_int(state.get("interval_minutes"), minimum_minutes)
        return max(minimum_minutes, interval * 2)
    if key == "orderbook":
        try:
            context_minutes = int(round(float(str(state.get("context_poll_seconds") or "0").strip()) / 60.0))
        except Exception:
            context_minutes = 0
        return max(minimum_minutes, context_minutes * 2)
    return minimum_minutes


def _is_stale(timestamp_text: str, max_age_minutes: int) -> bool:
    if not timestamp_text:
        return True
    try:
        timestamp = datetime.fromisoformat(str(timestamp_text).replace("Z", "+00:00"))
    except Exception:
        return True
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    age_seconds = (datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)).total_seconds()
    return age_seconds > max_age_minutes * 60


def _event(label: str, event_type: str, message: str, old_pid: int | None, new_pid: int | None) -> dict[str, Any]:
    return {
        "ts": utc_now(),
        "tool": label,
        "event": event_type,
        "message": message,
        "old_pid": old_pid,
        "new_pid": new_pid,
    }


def _quote(value: str) -> str:
    escaped = str(value).replace('"', r'\"')
    return f'"{escaped}"'


def _vbs_quote(value: str) -> str:
    escaped = str(value).replace('"', '""')
    return f'"{escaped}"'

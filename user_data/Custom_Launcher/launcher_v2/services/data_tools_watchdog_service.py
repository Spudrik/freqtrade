from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import csv
import io
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

from .collector_service import (
    NEWS_PROFILE,
    WEB_PROFILE,
    ResearchCollectorService,
    open_path,
    preferred_data_tools_python,
    utc_now,
)
from .global_context_service import GlobalContextService
from .orderbook_service import OrderBookService
from ..preset_manager import read_presets


DEFAULT_TASK_NAME = "FreqtradeDataToolsWatchdog"
DEFAULT_CHECK_INTERVAL_MINUTES = 60
DEFAULT_HEARTBEAT_STALE_MINUTES = 10
DEFAULT_OPERATOR_NOTE = (
    "Resolve simple operational blockers only: restart stale/dead collectors through their managed service paths. "
    "Do not auto-edit source configs, code, credentials, or disable sources; leave unresolved source/API/TLS blockers in status for operator review."
)
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
        self.python_exe = python_exe
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
            "operator_note": DEFAULT_OPERATOR_NOTE,
        }

    def preset_path(self) -> Path:
        return self.app_dir / "launcher_v2" / "config" / "presets.json"

    def load_auto_preset(self, preset_path: str | Path | None = None) -> dict[str, Any]:
        path = Path(preset_path) if preset_path else self.preset_path()
        payload = read_presets(path)
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
        merged["operator_note"] = str(merged.get("operator_note") or DEFAULT_OPERATOR_NOTE).strip() or DEFAULT_OPERATOR_NOTE
        return merged

    def run_once(self, state: dict[str, Any] | None = None, *, preset_path: str | Path | None = None) -> dict[str, Any]:
        preset = self.load_auto_preset(preset_path)
        preset_state = self._watchdog_state_from_preset(preset)
        overrides = {key: value for key, value in (state or {}).items() if value is not None}
        state = self.normalize_state({**preset_state, **overrides})
        heartbeat_stale_minutes = _positive_int(state.get("heartbeat_stale_minutes"), DEFAULT_HEARTBEAT_STALE_MINUTES)
        restart_dead = bool(state.get("restart_dead", True))
        selected = set(state["services"])
        desired = preset.get("collector_desired", {})
        if not isinstance(desired, dict):
            raise ValueError("LauncherV2-auto collector_desired must be an object.")
        rows: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []

        for key in SERVICE_KEYS:
            if key not in selected:
                continue
            row = self._check_tool(
                key,
                preset,
                heartbeat_stale_minutes,
                restart_dead,
                desired.get(key, True) is not False,
                preset_path or self.preset_path(),
            )
            rows.append(row)
            if row.get("event"):
                events.append(row["event"])

        payload = {
            "checked_at": utc_now(),
            "heartbeat_stale_minutes": heartbeat_stale_minutes,
            "restart_dead": restart_dead,
            "services": [key for key in SERVICE_KEYS if key in selected],
            "operator_note": state.get("operator_note") or DEFAULT_OPERATOR_NOTE,
            "rows": rows,
            "events": events,
        }
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        for event in events:
            self._append_event(event)
        return payload

    def _check_tool(
        self,
        key: str,
        preset: dict[str, Any],
        heartbeat_stale_minutes: int,
        restart_dead: bool,
        desired: bool = True,
        preset_path: str | Path | None = None,
    ) -> dict[str, Any]:
        spec = TOOL_SPECS[key]
        try:
            state = self._collector_state(key, preset)
            status = self._read_status(key, state)
            if not desired:
                return {
                    "key": key,
                    "label": spec.label,
                    "pid": _status_pid(status),
                    "running": status.get("verified_running") is True,
                    "desired": False,
                    "status": status.get("status") or "stopped_by_user",
                    "heartbeat_at": _status_heartbeat(status),
                    "heartbeat_stale_minutes": _effective_stale_minutes(key, state, heartbeat_stale_minutes),
                    "last_fetch_at": status.get("last_fetch_at") or status.get("last_message_at") or status.get("last_metric_at") or "",
                    "last_error": str(status.get("last_error") or ""),
                    "action": "stopped_by_user",
                    "event": None,
                }
            if not isinstance(status.get("verified_running"), bool):
                raise RuntimeError("Collector status is missing verified_running; refusing to infer ownership from PID liveness.")
            pid = _status_pid(status)
            running = status["verified_running"]
            heartbeat = _status_heartbeat(status)
            effective_stale_minutes = _effective_stale_minutes(key, state, heartbeat_stale_minutes)
            stale = _is_stale(heartbeat, effective_stale_minutes)
            last_error = str(status.get("last_error") or "")
            action = "none"
            event: dict[str, Any] | None = None

            if not running and restart_dead:
                new_pid = self._start_tool(key, state, preset, preset_path)
                if new_pid is None:
                    status = self._read_status(key, state)
                    return {
                        "key": key,
                        "label": spec.label,
                        "pid": _status_pid(status),
                        "running": status.get("verified_running") is True,
                        "desired": False,
                        "status": status.get("status") or "stopped_by_user",
                        "heartbeat_at": _status_heartbeat(status),
                        "heartbeat_stale_minutes": effective_stale_minutes,
                        "last_fetch_at": status.get("last_fetch_at") or status.get("last_message_at") or status.get("last_metric_at") or "",
                        "last_error": str(status.get("last_error") or ""),
                        "action": "stopped_by_user",
                        "event": None,
                    }
                action = f"restarted pid {new_pid}"
                event = _event(spec.label, "restart", f"{spec.label} was stopped or stale; restarted PID {new_pid}.", pid, new_pid)
                pid = new_pid
                status = self._read_status(key, state)
                if not isinstance(status.get("verified_running"), bool):
                    raise RuntimeError("Collector status is missing verified_running after restart; refusing to infer ownership from PID liveness.")
                pid = _status_pid(status) or new_pid
                running = status["verified_running"]
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
                "desired": desired,
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
                "desired": desired,
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
                "python_exe": self._service_python(preset),
                "config_path": preset.get("news_config_path"),
                "data_dir": preset.get("news_data_dir"),
                "db_path": preset.get("news_db_path"),
                "interval_minutes": preset.get("news_interval_minutes"),
                "once": False,
            }
        if key == "web":
            return {
                "python_exe": self._service_python(preset),
                "config_path": preset.get("web_config_path"),
                "data_dir": preset.get("web_data_dir"),
                "db_path": preset.get("web_db_path"),
                "interval_minutes": preset.get("web_interval_minutes"),
                "once": False,
            }
        if key == "global_context":
            return {
                "python_exe": self._service_python(preset),
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
                "python_exe": self._service_python(preset),
                "config_path": preset.get("orderbook_config_path"),
                "data_dir": preset.get("orderbook_data_dir"),
                "market_profiles": preset.get("orderbook_market_profiles"),
                "depth_levels": preset.get("orderbook_depth_levels"),
                "stream_update_ms": preset.get("orderbook_stream_update_ms"),
                "metric_interval_seconds": preset.get("orderbook_metric_interval_seconds"),
                "context_poll_seconds": preset.get("orderbook_context_poll_seconds"),
                "context_period": preset.get("orderbook_context_period"),
                "bar_intervals_seconds": preset.get("orderbook_bar_intervals_seconds", [60]),
                "snapshot_interval_seconds": preset.get("orderbook_snapshot_interval_seconds"),
                "capacity_warning_mb": preset.get("orderbook_capacity_warning_mb"),
                "capacity_critical_mb": preset.get("orderbook_capacity_critical_mb"),
                "max_symbols": preset.get("orderbook_max_symbols"),
                "store_metric_ticks": False,
                "store_snapshots": False,
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

    def _start_tool(
        self,
        key: str,
        state: dict[str, Any],
        preset: dict[str, Any],
        preset_path: str | Path | None = None,
    ) -> int | None:
        if key == "news":
            return self.research_service.start_detached(NEWS_PROFILE, state, automatic=True, preset_path=preset_path)
        if key == "web":
            return self.research_service.start_detached(WEB_PROFILE, state, automatic=True, preset_path=preset_path)
        if key == "global_context":
            return self.global_service.start_detached(state, automatic=True, preset_path=preset_path)
        if key == "orderbook":
            return self.orderbook_service.start_detached(state, _pairs_from_preset(preset), automatic=True, preset_path=preset_path)
        raise KeyError(key)

    def _service_python(self, preset: dict[str, Any]) -> str:
        configured = self.python_exe if self.python_exe is not None else preset.get("python_exe")
        return preferred_data_tools_python(self.app_dir, configured)

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

    def build_runner_command(self, state: dict[str, Any] | None = None, *, preset_path: str | Path | None = None) -> str:
        runner = self.app_dir / "data_tools_watchdog.py"
        preset = self.load_auto_preset(preset_path)
        python_override = self.python_exe if self.python_exe is not None else (state or {}).get("python_exe")
        if python_override is None:
            python_override = preset.get("python_exe")
        parts = [
            _quote(preferred_data_tools_python(self.app_dir, python_override)),
            _quote(str(runner)),
            "--once",
            "--preset",
            _quote(str(preset_path or self.preset_path())),
        ]
        return " ".join(parts)

    def _watchdog_state_from_preset(self, preset: dict[str, Any]) -> dict[str, Any]:
        state = {
            "task_name": preset.get("data_watchdog_task_name"),
            "check_interval_minutes": preset.get("data_watchdog_check_interval_minutes"),
            "heartbeat_stale_minutes": preset.get("data_watchdog_heartbeat_stale_minutes"),
            "restart_dead": preset.get("data_watchdog_restart_dead"),
            "operator_note": preset.get("data_watchdog_operator_note"),
            "services": preset.get("data_watchdog_services"),
        }
        return {key: value for key, value in state.items() if value is not None}

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
        note = str(state.get("operator_note") or DEFAULT_OPERATOR_NOTE).replace("\r", " ").replace("\n", " ")
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        wrapper.write_text(
            "@echo off\n"
            f"echo Watchdog note: {note}\n"
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
        user_sid = _current_user_sid()
        hidden_wrapper = self.write_hidden_task_wrapper(state)
        task_xml = _scheduled_task_xml(hidden_wrapper, interval_minutes, user_sid)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", encoding="utf-16", delete=False) as handle:
                handle.write(task_xml)
                temp_path = Path(handle.name)
            installed = subprocess.run(
                ["schtasks", "/Create", "/TN", task_name, "/XML", str(temp_path), "/F"],
                text=True,
                capture_output=True,
                check=False,
            )
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

        startup_entry = self.startup_entry_path(task_name)
        verified = None
        if installed.returncode == 0:
            queried = subprocess.run(
                ["schtasks", "/Query", "/TN", task_name, "/XML"],
                text=True,
                capture_output=True,
                check=False,
            )
            verified = queried.returncode == 0 and _scheduled_task_xml_matches(
                queried.stdout,
                hidden_wrapper,
                interval_minutes,
                user_sid,
            )
        else:
            queried = None

        stdout = f"[{task_name}]\n{installed.stdout or ''}\n"
        stderr = f"[{task_name}]\n{installed.stderr or ''}\n"
        if installed.returncode == 0:
            if verified:
                stdout += f"[scheduler_verification]\nVerified current-user InteractiveToken task XML.\n"
            else:
                stdout += "[scheduler_verification]\nTask XML did not match the requested user, triggers, action, or instance policy.\n"
                if queried is not None:
                    stderr += f"{queried.stderr or queried.stdout or ''}\n"
        stdout += f"[startup_entry]\nRetained {startup_entry} | exists={startup_entry.exists()}\n"
        return subprocess.CompletedProcess(
            args=["schtasks", task_name, str(startup_entry)],
            returncode=0 if installed.returncode == 0 and verified else 1,
            stdout=stdout,
            stderr=stderr,
        )

    def remove_scheduled_task(self, task_name: str | None = None) -> subprocess.CompletedProcess[str]:
        if os.name != "nt":
            raise RuntimeError("Windows Task Scheduler is only available on Windows.")
        name = str(task_name or DEFAULT_TASK_NAME).strip() or DEFAULT_TASK_NAME
        hourly = subprocess.run(["schtasks", "/Delete", "/TN", name, "/F"], text=True, capture_output=True, check=False)
        startup_entry = self.startup_entry_path(name)
        startup_entry_removed = startup_entry.exists()
        if startup_entry.exists():
            startup_entry.unlink()
        returncode = 0 if hourly.returncode == 0 else 1
        startup_entry_status = f"Removed {startup_entry}" if startup_entry_removed else f"Not present {startup_entry}"
        return subprocess.CompletedProcess(
            args=["schtasks", name, str(startup_entry)],
            returncode=returncode,
            stdout=f"[{name}]\n{hourly.stdout or ''}\n[startup_entry]\n{startup_entry_status}\n",
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


def _current_user_sid() -> str:
    try:
        result = subprocess.run(
            ["whoami", "/user", "/fo", "csv", "/nh"],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"Could not determine the current Windows user SID with whoami: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no command output").strip()
        raise RuntimeError(f"Could not determine the current Windows user SID with whoami: {detail}")
    try:
        rows = list(csv.reader(io.StringIO(result.stdout)))
        user_sid = rows[0][1].strip() if len(rows) == 1 and len(rows[0]) == 2 else ""
    except (csv.Error, IndexError):
        user_sid = ""
    if not user_sid.startswith("S-"):
        raise RuntimeError("Could not determine the current Windows user SID: whoami returned unexpected CSV output.")
    return user_sid


def _scheduled_task_xml(hidden_wrapper: Path, interval_minutes: int, user_sid: str) -> str:
    if not user_sid.strip():
        raise ValueError("user_sid must be provided when building the scheduled task XML.")
    start_boundary = datetime.now().astimezone().replace(microsecond=0).isoformat()
    arguments = escape(f'//B //NoLogo "{hidden_wrapper}"')
    return f'''<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>Run LauncherV2 data collectors for the signed-in user on logon and hourly.</Description></RegistrationInfo>
  <Triggers>
    <LogonTrigger><Enabled>true</Enabled></LogonTrigger>
    <TimeTrigger>
      <Repetition><Interval>PT{interval_minutes}M</Interval><StopAtDurationEnd>false</StopAtDurationEnd></Repetition>
      <StartBoundary>{start_boundary}</StartBoundary><Enabled>true</Enabled>
    </TimeTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author"><UserId>{escape(user_sid.strip())}</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand><Enabled>true</Enabled><Hidden>false</Hidden>
    <ExecutionTimeLimit>PT10M</ExecutionTimeLimit><Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec><Command>C:\\Windows\\System32\\wscript.exe</Command><Arguments>{arguments}</Arguments></Exec>
  </Actions>
</Task>
'''


def _scheduled_task_xml_matches(xml_text: str, hidden_wrapper: Path, interval_minutes: int, user_sid: str) -> bool:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return False
    namespace = {"task": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
    principal = root.find(".//task:Principals/task:Principal", namespace)
    user_id = principal.findtext("task:UserId", default="", namespaces=namespace) if principal is not None else ""
    logon_type = principal.findtext("task:LogonType", default="", namespaces=namespace) if principal is not None else ""
    multiple_instances = root.findtext(".//task:Settings/task:MultipleInstancesPolicy", default="", namespaces=namespace)
    actions = root.find(".//task:Actions/task:Exec", namespace)
    command = actions.findtext("task:Command", default="", namespaces=namespace) if actions is not None else ""
    arguments = actions.findtext("task:Arguments", default="", namespaces=namespace) if actions is not None else ""
    triggers = root.find("task:Triggers", namespace)
    has_logon = triggers is not None and any(
        trigger.tag == f"{{{namespace['task']}}}LogonTrigger"
        and trigger.findtext("task:Enabled", default="true", namespaces=namespace).lower() == "true"
        for trigger in triggers
    )
    time_triggers = (
        [trigger for trigger in triggers if trigger.tag == f"{{{namespace['task']}}}TimeTrigger"]
        if triggers is not None
        else []
    )
    has_hourly = any(
        trigger.findtext("task:Enabled", default="true", namespaces=namespace).lower() == "true"
        and trigger.findtext("task:Repetition/task:Interval", default="", namespaces=namespace) == f"PT{interval_minutes}M"
        for trigger in time_triggers
    )
    expected_arguments = f'//B //NoLogo "{hidden_wrapper}"'
    return (
        user_id.strip().casefold() == user_sid.strip().casefold()
        and logon_type == "InteractiveToken"
        and multiple_instances == "IgnoreNew"
        and command.rstrip("\\/").casefold() == r"C:\Windows\System32\wscript.exe".rstrip("\\/").casefold()
        and arguments == expected_arguments
        and has_logon
        and has_hourly
    )


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

from __future__ import annotations

from typing import Any
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from ..base_tab import BaseTab
from ..services.data_tools_watchdog_service import (
    DEFAULT_CHECK_INTERVAL_MINUTES,
    DEFAULT_TASK_NAME,
    SERVICE_KEYS,
    DataToolsWatchdogService,
)
from ..ui_helpers import labeled_entry, set_tree_rows


class DataWatchdogTab(BaseTab):
    tab_key = "data_watchdog"
    tab_title = "Watchdog"

    def __init__(self, master: tk.Misc, context: Any) -> None:
        super().__init__(master, context)
        self.service = DataToolsWatchdogService(context.app_dir, context.shared.python_exe.get())
        self.task_name_var = tk.StringVar(value=DEFAULT_TASK_NAME)
        self.check_interval_minutes_var = tk.StringVar(value=str(DEFAULT_CHECK_INTERVAL_MINUTES))
        self.heartbeat_stale_minutes_var = tk.StringVar(value="10")
        self.restart_dead_var = tk.BooleanVar(value=True)
        self.service_vars = {key: tk.BooleanVar(value=True) for key in SERVICE_KEYS}
        self.status_var = tk.StringVar(value="Not checked")
        self.command_var = tk.StringVar(value="")
        self.rows_tree: ttk.Treeview | None = None
        self.events_tree: ttk.Treeview | None = None
        self.scheduler_text: scrolledtext.ScrolledText | None = None
        self._build_ui()
        self.refresh_command()
        self.refresh_status()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)
        self.grid_rowconfigure(5, weight=1)

        intro = ttk.Label(
            self,
            text="Watchdog runs as a Windows scheduled task. Each pass checks collector PID + heartbeat, restarts dead collectors, and records warning events.",
            wraplength=1120,
        )
        intro.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 6))

        settings = ttk.LabelFrame(self, text="Scheduled task settings")
        settings.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        settings.grid_columnconfigure(1, weight=1)
        settings.grid_columnconfigure(3, weight=1)
        labeled_entry(settings, 0, 0, "Task name", self.task_name_var)
        labeled_entry(settings, 0, 2, "Check interval minutes", self.check_interval_minutes_var)
        labeled_entry(settings, 1, 0, "Heartbeat stale minutes", self.heartbeat_stale_minutes_var)
        ttk.Checkbutton(settings, text="Restart dead collectors", variable=self.restart_dead_var, command=self.refresh_command).grid(row=1, column=2, sticky="w", padx=8, pady=4)
        services_frame = ttk.LabelFrame(settings, text="Services")
        services_frame.grid(row=2, column=0, columnspan=4, sticky="ew", padx=8, pady=4)
        labels = {
            "news": "News",
            "web": "Web",
            "global_context": "Global Context",
            "orderbook": "Order Book",
        }
        for column, key in enumerate(SERVICE_KEYS):
            ttk.Checkbutton(services_frame, text=labels[key], variable=self.service_vars[key], command=self.refresh_command).grid(row=0, column=column, sticky="w", padx=8, pady=4)
        for var in (self.task_name_var, self.check_interval_minutes_var, self.heartbeat_stale_minutes_var):
            var.trace_add("write", lambda *_: self.refresh_command())

        preview = ttk.LabelFrame(self, text="Scheduled command")
        preview.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        preview.grid_columnconfigure(0, weight=1)
        ttk.Entry(preview, textvariable=self.command_var).grid(row=0, column=0, sticky="ew", padx=8, pady=8)

        buttons = ttk.Frame(self)
        buttons.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(buttons, text="Run Check Now", command=self.run_check_now).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Install/Update Scheduled Task", command=self.install_task).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Remove Scheduled Task", command=self.remove_task).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Query Scheduled Task", command=self.query_task).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Refresh Status", command=self.refresh_status).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Open Event Log", command=self.open_event_log).pack(side="left")
        ttk.Label(buttons, textvariable=self.status_var).pack(side="right")

        current = ttk.LabelFrame(self, text="Latest watchdog check")
        current.grid(row=4, column=0, sticky="nsew", padx=8, pady=(0, 8))
        current.grid_columnconfigure(0, weight=1)
        current.grid_rowconfigure(0, weight=1)
        columns = ("label", "pid", "running", "status", "heartbeat_at", "stale_minutes", "last_fetch_at", "action", "last_error")
        self.rows_tree = ttk.Treeview(current, columns=columns, show="headings", height=8)
        widths = {
            "label": 130,
            "pid": 80,
            "running": 80,
            "status": 110,
            "heartbeat_at": 210,
            "stale_minutes": 100,
            "last_fetch_at": 210,
            "action": 150,
            "last_error": 360,
        }
        for column in columns:
            self.rows_tree.heading(column, text=column)
            self.rows_tree.column(column, width=widths[column], anchor="w")
        self.rows_tree.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        ttk.Scrollbar(current, orient="vertical", command=self.rows_tree.yview).grid(row=0, column=1, sticky="ns", pady=8)

        events = ttk.LabelFrame(self, text="Recent watchdog events")
        events.grid(row=5, column=0, sticky="nsew", padx=8, pady=(0, 8))
        events.grid_columnconfigure(0, weight=1)
        events.grid_rowconfigure(0, weight=1)
        event_columns = ("ts", "tool", "event", "message", "old_pid", "new_pid")
        self.events_tree = ttk.Treeview(events, columns=event_columns, show="headings", height=8)
        for column in event_columns:
            self.events_tree.heading(column, text=column)
            self.events_tree.column(column, width=180 if column != "message" else 560, anchor="w")
        self.events_tree.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        ttk.Scrollbar(events, orient="vertical", command=self.events_tree.yview).grid(row=0, column=1, sticky="ns", pady=8)

        scheduler = ttk.LabelFrame(self, text="Scheduler output")
        scheduler.grid(row=6, column=0, sticky="ew", padx=8, pady=(0, 8))
        scheduler.grid_columnconfigure(0, weight=1)
        self.scheduler_text = scrolledtext.ScrolledText(scheduler, height=5, wrap="word")
        self.scheduler_text.grid(row=0, column=0, sticky="ew", padx=8, pady=8)

    def _state_for_service(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name_var.get(),
            "check_interval_minutes": self.check_interval_minutes_var.get(),
            "heartbeat_stale_minutes": self.heartbeat_stale_minutes_var.get(),
            "restart_dead": self.restart_dead_var.get(),
            "services": [key for key, var in self.service_vars.items() if var.get()],
        }

    def refresh_command(self) -> None:
        try:
            self.command_var.set(self.service.build_runner_command(self._state_for_service()))
        except Exception as exc:
            self.command_var.set(f"Invalid watchdog config: {exc}")

    def run_check_now(self) -> None:
        try:
            payload = self.service.run_once(self._state_for_service())
        except Exception as exc:
            messagebox.showerror("Data watchdog", f"Watchdog check failed:\n{exc}", parent=self)
            return
        self.status_var.set(f"Checked {payload.get('checked_at', '')}")
        self.refresh_status()
        self.context.emit("save_state", {"reason": "data_watchdog_check"})

    def install_task(self) -> None:
        result = self.service.install_scheduled_task(self._state_for_service())
        self._show_scheduler_result(result)
        self.context.emit("save_state", {"reason": "data_watchdog_install"})
        if result.returncode != 0:
            messagebox.showerror("Data watchdog", f"Could not install scheduled task:\n{result.stderr or result.stdout}", parent=self)
        else:
            self.status_var.set("Scheduled task installed/updated")

    def remove_task(self) -> None:
        result = self.service.remove_scheduled_task(self.task_name_var.get())
        self._show_scheduler_result(result)
        self.context.emit("save_state", {"reason": "data_watchdog_remove"})
        if result.returncode != 0:
            messagebox.showwarning("Data watchdog", f"Scheduled task remove returned:\n{result.stderr or result.stdout}", parent=self)
        else:
            self.status_var.set("Scheduled task removed")

    def query_task(self) -> None:
        result = self.service.query_scheduled_task(self.task_name_var.get())
        self._show_scheduler_result(result)
        self.status_var.set("Scheduled task query complete")

    def _show_scheduler_result(self, result: Any) -> None:
        if self.scheduler_text is None:
            return
        self.scheduler_text.delete("1.0", tk.END)
        self.scheduler_text.insert(tk.END, f"returncode={result.returncode}\n")
        if result.stdout:
            self.scheduler_text.insert(tk.END, result.stdout)
        if result.stderr:
            self.scheduler_text.insert(tk.END, result.stderr)

    def refresh_status(self) -> None:
        latest = self.service.read_latest_status()
        rows = latest.get("rows") if isinstance(latest, dict) else []
        if self.rows_tree is not None:
            set_tree_rows(
                self.rows_tree,
                [
                    (
                        row.get("label", ""),
                        row.get("pid", ""),
                        row.get("running", ""),
                        row.get("status", ""),
                        row.get("heartbeat_at", ""),
                        row.get("heartbeat_stale_minutes", ""),
                        row.get("last_fetch_at", ""),
                        row.get("action", ""),
                        row.get("last_error", ""),
                    )
                    for row in rows or []
                ],
            )
        if latest.get("checked_at"):
            self.status_var.set(f"Last checked {latest.get('checked_at')}")
        if self.events_tree is not None:
            set_tree_rows(
                self.events_tree,
                [
                    (
                        row.get("ts", ""),
                        row.get("tool", ""),
                        row.get("event", ""),
                        row.get("message", ""),
                        row.get("old_pid", ""),
                        row.get("new_pid", ""),
                    )
                    for row in reversed(self.service.read_events(limit=200))
                ],
            )

    def open_event_log(self) -> None:
        try:
            self.service.open_events_log()
        except Exception as exc:
            messagebox.showerror("Data watchdog", f"Could not open event log:\n{exc}", parent=self)

    def get_state(self) -> dict[str, Any]:
        return self._state_for_service()

    def set_state(self, state: dict[str, Any]) -> None:
        normalized = self.service.normalize_state(state)
        self.task_name_var.set(str(normalized.get("task_name") or DEFAULT_TASK_NAME))
        self.check_interval_minutes_var.set(str(normalized.get("check_interval_minutes") or str(DEFAULT_CHECK_INTERVAL_MINUTES)))
        self.heartbeat_stale_minutes_var.set(str(normalized.get("heartbeat_stale_minutes") or "10"))
        self.restart_dead_var.set(bool(normalized.get("restart_dead", True)))
        selected = set(normalized.get("services") or SERVICE_KEYS)
        for key, var in self.service_vars.items():
            var.set(key in selected)
        self.refresh_command()

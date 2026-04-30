from __future__ import annotations

from typing import Any
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ..base_tab import BaseTab
from ..command_builder import command_text
from ..services.global_context_service import GlobalContextService
from ..ui_helpers import labeled_entry, set_tree_rows


class GlobalContextTab(BaseTab):
    tab_key = "global_context"
    tab_title = "Global Context"

    def __init__(self, master: tk.Misc, context: Any) -> None:
        super().__init__(master, context)
        self.service = GlobalContextService(context.app_dir, context.shared.python_exe.get())
        paths = self.service.paths({})
        self.config_path_var = tk.StringVar(value=str(paths["config"]))
        self.data_dir_var = tk.StringVar(value=str(paths["data_dir"]))
        self.db_path_var = tk.StringVar(value=str(paths["db"]))
        self.key_file_var = tk.StringVar(value="")
        self.fred_key_json_path_var = tk.StringVar(value="fred.api_key")
        self.enable_fred_var = tk.BooleanVar(value=False)
        self.fred_status_var = tk.StringVar(value="No FRED key file selected")
        self.interval_minutes_var = tk.StringVar(value="30")
        self.once_var = tk.BooleanVar(value=False)
        self.preview_var = tk.StringVar(value="")
        self.score_summary_vars = {
            key: tk.StringVar(value="-")
            for key in ("average_score", "signal", "score_count", "source_score_count", "calc_score_count")
        }
        self.status_vars = {
            key: tk.StringVar(value="-")
            for key in (
                "status",
                "pid",
                "started_at",
                "heartbeat_at",
                "last_fetch_at",
                "sources_enabled",
                "ticks_total",
                "new_ticks_last_cycle",
                "stored_ticks",
                "last_error",
            )
        }
        self._build_ui()
        self._bind_preview()
        self.refresh_preview()
        self.refresh_status()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)

        intro = ttk.Label(
            self,
            text="Global Context polls free public APIs such as CoinGecko, DeFiLlama, Alternative.me, Stooq, and optional FRED into SQLite. It is data-only and does not change strategy logic.",
            wraplength=1120,
        )
        intro.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 6))

        settings = ttk.LabelFrame(self, text="Collector settings")
        settings.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        settings.grid_columnconfigure(1, weight=1)
        settings.grid_columnconfigure(3, weight=1)
        self._path_row(settings, 0, "Config path", self.config_path_var, directory=False)
        self._path_row(settings, 1, "Data dir", self.data_dir_var, directory=True)
        self._path_row(settings, 2, "Database", self.db_path_var, directory=False)
        labeled_entry(settings, 3, 0, "Poll interval minutes", self.interval_minutes_var)
        ttk.Checkbutton(settings, text="Run once", variable=self.once_var).grid(row=3, column=2, sticky="w", padx=8, pady=4)
        fred = ttk.LabelFrame(settings, text="FRED key file")
        fred.grid(row=0, column=3, rowspan=4, sticky="nsew", padx=(12, 8), pady=4)
        fred.grid_columnconfigure(1, weight=1)
        self._path_row(fred, 0, "Key file", self.key_file_var, directory=False)
        labeled_entry(fred, 1, 0, "JSON path", self.fred_key_json_path_var)
        ttk.Checkbutton(fred, text="Enable FRED sources for this run", variable=self.enable_fred_var).grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=4)
        ttk.Label(fred, textvariable=self.fred_status_var).grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=4)
        ttk.Button(fred, text="Check Key", command=self.refresh_fred_key_status).grid(row=3, column=2, sticky="e", padx=8, pady=4)

        preview = ttk.LabelFrame(self, text="Generated command")
        preview.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        preview.grid_columnconfigure(0, weight=1)
        ttk.Entry(preview, textvariable=self.preview_var).grid(row=0, column=0, sticky="ew", padx=8, pady=8)

        buttons = ttk.Frame(self)
        buttons.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(buttons, text="Start Detached Collector", command=self.start_collector).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Request Stop", command=self.request_stop).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Refresh", command=self.refresh_status).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Open Data Folder", command=self.open_data_folder).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Open Log", command=self.open_log).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Export Latest CSV", command=self.export_latest).pack(side="left")

        status = ttk.LabelFrame(self, text="Status")
        status.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 8))
        status.grid_columnconfigure(4, weight=1)
        items = [
            ("Status", "status"),
            ("PID", "pid"),
            ("Started at", "started_at"),
            ("Heartbeat at", "heartbeat_at"),
            ("Last fetch at", "last_fetch_at"),
            ("Sources enabled", "sources_enabled"),
            ("Ticks total", "ticks_total"),
            ("New ticks last cycle", "new_ticks_last_cycle"),
            ("Stored ticks", "stored_ticks"),
            ("Last error", "last_error"),
        ]
        for index, (label, key) in enumerate(items):
            row = index // 2
            col = (index % 2) * 2
            ttk.Label(status, text=f"{label}:").grid(row=row, column=col, sticky="w", padx=8, pady=3)
            ttk.Label(status, textvariable=self.status_vars[key]).grid(row=row, column=col + 1, sticky="w", padx=8, pady=3)
        score_panel = ttk.LabelFrame(status, text="Score Summary")
        score_panel.grid(row=0, column=4, rowspan=5, sticky="nsew", padx=(16, 8), pady=4)
        score_panel.grid_columnconfigure(0, weight=1)
        score_panel.grid_rowconfigure(2, weight=1)
        score_header = ttk.Frame(score_panel)
        score_header.grid(row=0, column=0, sticky="ew", padx=8, pady=(4, 2))
        labels = [
            ("Average", "average_score"),
            ("Signal", "signal"),
            ("Scores", "score_count"),
            ("Source", "source_score_count"),
            ("Calc", "calc_score_count"),
        ]
        for index, (label, key) in enumerate(labels):
            ttk.Label(score_header, text=f"{label}:").grid(row=0, column=index * 2, sticky="w", padx=(0, 3))
            ttk.Label(score_header, textvariable=self.score_summary_vars[key]).grid(row=0, column=index * 2 + 1, sticky="w", padx=(0, 12))
        score_columns = ("source_id", "metric_key", "signal", "effective_score", "source_score", "calc_score")
        self.score_tree = ttk.Treeview(score_panel, columns=score_columns, show="headings", height=4)
        for column in score_columns:
            self.score_tree.heading(column, text=column)
            self.score_tree.column(column, width=120, anchor="w")
        self.score_tree.column("source_id", width=170, anchor="w")
        self.score_tree.column("metric_key", width=220, anchor="w")
        self.score_tree.grid(row=2, column=0, sticky="nsew", padx=8, pady=(2, 6))

        views = ttk.Notebook(self)
        views.grid(row=5, column=0, sticky="nsew", padx=8, pady=(0, 8))

        summary_tab = ttk.Frame(views)
        crypto_tab = ttk.Frame(views)
        equities_tab = ttk.Frame(views)
        health_tab = ttk.Frame(views)
        views.add(summary_tab, text="Summary")
        views.add(crypto_tab, text="Crypto/Liquidity")
        views.add(equities_tab, text="Equities/Risk")
        views.add(health_tab, text="Source Health")

        summary_tab.grid_columnconfigure(0, weight=1)
        summary_tab.grid_rowconfigure(0, weight=1)
        summary_columns = ("source_group", "signal", "avg_effective_score", "metric_count", "notes", "latest_at")
        self.summary_tree = ttk.Treeview(summary_tab, columns=summary_columns, show="headings", height=12)
        for column in summary_columns:
            self.summary_tree.heading(column, text=column)
            self.summary_tree.column(column, width=150, anchor="w")
        self.summary_tree.column("notes", width=700, anchor="w")
        self.summary_tree.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        summary_scroll = ttk.Scrollbar(summary_tab, orient="vertical", command=self.summary_tree.yview)
        summary_scroll.grid(row=0, column=1, sticky="ns", pady=8)
        self.summary_tree.configure(yscrollcommand=summary_scroll.set)

        latest_columns = ("source_id", "source_group", "metric_key", "signal", "effective_score", "source_score", "calc_score", "value", "unit", "notes", "source_ts", "collected_at")
        self.crypto_tree = self._context_tree(crypto_tab, latest_columns)
        self.equities_tree = self._context_tree(equities_tab, latest_columns)

        health_tab.grid_columnconfigure(0, weight=1)
        health_tab.grid_rowconfigure(0, weight=1)
        health_columns = ("source_id", "source_group", "source_type", "enabled", "relevance", "last_success_at", "last_failure_at", "last_error", "last_effective_score", "last_source_score", "last_calc_score", "last_signal", "last_notes", "updated_at")
        self.health_tree = ttk.Treeview(health_tab, columns=health_columns, show="headings", height=12)
        for column in health_columns:
            self.health_tree.heading(column, text=column)
            self.health_tree.column(column, width=145, anchor="w")
        self.health_tree.column("last_notes", width=420, anchor="w")
        self.health_tree.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        health_scroll = ttk.Scrollbar(health_tab, orient="vertical", command=self.health_tree.yview)
        health_scroll.grid(row=0, column=1, sticky="ns", pady=8)
        self.health_tree.configure(yscrollcommand=health_scroll.set)

    def _context_tree(self, parent: ttk.Frame, columns: tuple[str, ...]) -> ttk.Treeview:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=12)
        for column in columns:
            tree.heading(column, text=column)
            tree.column(column, width=150, anchor="w")
        tree.column("notes", width=620, anchor="w")
        tree.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        scroll = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        scroll.grid(row=0, column=1, sticky="ns", pady=8)
        tree.configure(yscrollcommand=scroll.set)
        return tree

    def _path_row(self, parent: tk.Misc, row: int, label: str, variable: tk.StringVar, *, directory: bool) -> None:
        labeled_entry(parent, row, 0, label, variable)
        ttk.Button(parent, text="Browse", command=lambda: self._browse(variable, directory)).grid(row=row, column=2, padx=8, pady=4)

    def _browse(self, variable: tk.StringVar, directory: bool) -> None:
        path = filedialog.askdirectory(parent=self) if directory else filedialog.askopenfilename(parent=self, filetypes=[("All files", "*.*")])
        if path:
            variable.set(path)

    def _bind_preview(self) -> None:
        for var in (self.config_path_var, self.data_dir_var, self.db_path_var, self.key_file_var, self.fred_key_json_path_var, self.interval_minutes_var):
            var.trace_add("write", lambda *_: self.refresh_preview())
        self.once_var.trace_add("write", lambda *_: self.refresh_preview())
        self.enable_fred_var.trace_add("write", lambda *_: self.refresh_preview())

    def _state(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path_var.get(),
            "data_dir": self.data_dir_var.get(),
            "db_path": self.db_path_var.get(),
            "key_file": self.key_file_var.get(),
            "fred_key_json_path": self.fred_key_json_path_var.get(),
            "enable_fred": self.enable_fred_var.get(),
            "interval_minutes": self.interval_minutes_var.get(),
            "once": self.once_var.get(),
        }

    def refresh_preview(self) -> None:
        try:
            self.preview_var.set(command_text(self.service.build_command(self._state())))
        except Exception as exc:
            self.preview_var.set(f"Invalid Global Context configuration: {exc}")
        self.refresh_fred_key_status()

    def refresh_fred_key_status(self) -> None:
        try:
            self.fred_status_var.set(self.service.fred_key_status(self._state()))
        except Exception as exc:
            self.fred_status_var.set(f"FRED key status error: {exc}")

    def start_collector(self) -> None:
        try:
            pid = self.service.start_detached(self._state())
        except Exception as exc:
            messagebox.showerror("Global Context", f"Could not start collector:\n{exc}", parent=self)
            return
        self.status_vars["status"].set(f"started detached PID {pid}")
        self.status_vars["pid"].set(str(pid))
        self.context.emit("save_state", {"reason": "global_context_collector_start"})
        self.refresh_status()

    def request_stop(self) -> None:
        try:
            self.service.request_stop(self._state())
        except Exception as exc:
            messagebox.showerror("Global Context", f"Could not request stop:\n{exc}", parent=self)
            return
        self.status_vars["status"].set("stop requested")
        self.refresh_status()

    def refresh_status(self) -> None:
        status = self.service.read_status(self._state())
        mapping = {
            "status": status.get("status") or "unknown",
            "pid": status.get("pid_text") or status.get("pid") or "-",
            "started_at": status.get("started_at") or "-",
            "heartbeat_at": status.get("heartbeat_at") or "-",
            "last_fetch_at": status.get("last_fetch_at") or "-",
            "sources_enabled": status.get("sources_enabled") if status.get("sources_enabled") is not None else "-",
            "ticks_total": status.get("ticks_total") if status.get("ticks_total") is not None else "-",
            "new_ticks_last_cycle": status.get("new_ticks_last_cycle") if status.get("new_ticks_last_cycle") is not None else "-",
            "stored_ticks": status.get("stored_ticks") if status.get("stored_ticks") is not None else "-",
            "last_error": status.get("last_error") or "-",
        }
        for key, value in mapping.items():
            self.status_vars[key].set(str(value))
        try:
            summary = self.service.score_summary(self._state())
            for key, variable in self.score_summary_vars.items():
                variable.set(str(summary.get(key) or "-"))
            set_tree_rows(self.score_tree, [row[:6] for row in self.service.score_detail_rows(self._state())])
        except Exception:
            for variable in self.score_summary_vars.values():
                variable.set("-")
            set_tree_rows(self.score_tree, [])
        try:
            set_tree_rows(self.summary_tree, self.service.summary_rows(self._state()))
        except Exception:
            set_tree_rows(self.summary_tree, [])
        try:
            set_tree_rows(self.crypto_tree, self.service.latest_context_rows(self._state(), {"sentiment", "market", "defi"}))
        except Exception:
            set_tree_rows(self.crypto_tree, [])
        try:
            set_tree_rows(self.equities_tree, self.service.latest_context_rows(self._state(), {"equity_indices", "rates_fx"}))
        except Exception:
            set_tree_rows(self.equities_tree, [])
        try:
            set_tree_rows(self.health_tree, self.service.source_health_rows(self._state()))
        except Exception:
            set_tree_rows(self.health_tree, [])

    def open_data_folder(self) -> None:
        try:
            self.service.open_data_folder(self._state())
        except Exception as exc:
            messagebox.showerror("Global Context", f"Could not open data folder:\n{exc}", parent=self)

    def open_log(self) -> None:
        try:
            self.service.open_log(self._state())
        except Exception as exc:
            messagebox.showerror("Global Context", f"Could not open log:\n{exc}", parent=self)

    def export_latest(self) -> None:
        try:
            path, count = self.service.export_latest_csv(self._state())
        except Exception as exc:
            messagebox.showerror("Global Context", f"Could not export CSV:\n{exc}", parent=self)
            return
        messagebox.showinfo("Global Context", f"Exported {count} rows.\n\n{path}", parent=self)

    def get_state(self) -> dict[str, Any]:
        return self._state()

    def set_state(self, state: dict[str, Any]) -> None:
        paths = self.service.paths({})
        self.config_path_var.set(str(state.get("config_path") or paths["config"]))
        self.data_dir_var.set(str(state.get("data_dir") or paths["data_dir"]))
        self.db_path_var.set(str(state.get("db_path") or paths["db"]))
        self.key_file_var.set(str(state.get("key_file") or ""))
        self.fred_key_json_path_var.set(str(state.get("fred_key_json_path") or "fred.api_key"))
        self.enable_fred_var.set(bool(state.get("enable_fred", False)))
        self.interval_minutes_var.set(str(state.get("interval_minutes") or "30"))
        self.once_var.set(bool(state.get("once", False)))
        self.refresh_preview()
        self.refresh_status()

from __future__ import annotations

from typing import Any
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ..base_tab import BaseTab
from ..command_builder import command_text
from ..services.collector_service import CollectorProfile, ResearchCollectorService
from ..ui_helpers import labeled_entry, set_tree_rows


class ResearchCollectorTab(BaseTab):
    profile: CollectorProfile

    def __init__(self, master: tk.Misc, context: Any) -> None:
        super().__init__(master, context)
        self.service = ResearchCollectorService(context.app_dir, context.shared.python_exe.get())
        paths = self.service.paths(self.profile, "", "", "")
        self.config_path_var = tk.StringVar(value=str(paths["config"]))
        self.data_dir_var = tk.StringVar(value=str(paths["data_dir"]))
        self.db_path_var = tk.StringVar(value=str(paths["db"]))
        self.interval_minutes_var = tk.StringVar(value=str(self.profile.default_interval_minutes))
        self.once_var = tk.BooleanVar(value=False)
        self.preview_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="-")
        self.pid_var = tk.StringVar(value="-")
        self.started_at_var = tk.StringVar(value="-")
        self.heartbeat_at_var = tk.StringVar(value="-")
        self.last_fetch_at_var = tk.StringVar(value="-")
        self.total_articles_var = tk.StringVar(value="-")
        self.new_articles_last_cycle_var = tk.StringVar(value="-")
        self.last_error_var = tk.StringVar(value="-")
        self.article_tree: ttk.Treeview | None = None
        self._article_url_by_item: dict[str, str] = {}
        self._build_ui()
        self._bind_preview()
        self.refresh_preview()
        self.refresh_articles()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)
        self.grid_rowconfigure(6, weight=1)

        intro = ttk.Label(self, text=f"{self.profile.label} controls the existing detached collector and stores all runtime files in its data directory.", wraplength=1120)
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

        preview = ttk.LabelFrame(self, text="Generated command")
        preview.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        preview.grid_columnconfigure(0, weight=1)
        ttk.Entry(preview, textvariable=self.preview_var).grid(row=0, column=0, sticky="ew", padx=8, pady=8)

        buttons = ttk.Frame(self)
        buttons.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(buttons, text="Start Detached Collector", command=self.start_collector).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Request Stop", command=self.request_stop).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Refresh Status", command=self.refresh_status).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Open Data Folder", command=self.open_data_folder).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Open Log", command=self.open_log).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Export CSV", command=self.export_articles).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Export Source Health CSV", command=self.export_source_health).pack(side="left")
        ttk.Button(buttons, text="Refresh Articles", command=self.refresh_articles).pack(side="left", padx=(6, 0))

        status = ttk.LabelFrame(self, text="Status")
        status.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 8))
        for col in range(4):
            status.grid_columnconfigure(col, weight=1 if col in (1, 3) else 0)
        items = [
            ("Status", self.status_var),
            ("PID", self.pid_var),
            ("Started at", self.started_at_var),
            ("Heartbeat at", self.heartbeat_at_var),
            ("Last fetch at", self.last_fetch_at_var),
            ("Total articles", self.total_articles_var),
            ("New articles last cycle", self.new_articles_last_cycle_var),
            ("Last error", self.last_error_var),
        ]
        for index, (label, variable) in enumerate(items):
            row = index // 2
            col = (index % 2) * 2
            ttk.Label(status, text=f"{label}:").grid(row=row, column=col, sticky="w", padx=8, pady=4)
            ttk.Label(status, textvariable=variable).grid(row=row, column=col + 1, sticky="w", padx=8, pady=4)

        health = ttk.LabelFrame(self, text="Source health")
        health.grid(row=5, column=0, sticky="nsew", padx=8, pady=(0, 8))
        health.grid_columnconfigure(0, weight=1)
        health.grid_rowconfigure(0, weight=1)
        columns = ("source_id", "source_group", "enabled", "market_relevance", "last_success_at", "last_failure_at", "last_error", "items_last_fetch", "inserted_last_fetch", "duplicates_last_fetch")
        self.health_tree = ttk.Treeview(health, columns=columns, show="headings", height=12)
        for column in columns:
            self.health_tree.heading(column, text=column)
            self.health_tree.column(column, width=150, anchor="w")
        self.health_tree.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        scrollbar = ttk.Scrollbar(health, orient="vertical", command=self.health_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns", pady=8)
        self.health_tree.configure(yscrollcommand=scrollbar.set)

        articles = ttk.LabelFrame(self, text="Recent articles")
        articles.grid(row=6, column=0, sticky="nsew", padx=8, pady=(0, 8))
        articles.grid_columnconfigure(0, weight=1)
        articles.grid_rowconfigure(1, weight=1)
        action_row = ttk.Frame(articles)
        action_row.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))
        ttk.Button(action_row, text="Open Selected Article", command=self.open_selected_article).pack(side="left")
        columns = ("article_time", "source_id", "market_relevance", "title", "article_url")
        self.article_tree = ttk.Treeview(articles, columns=columns, show="headings", height=10)
        self.article_tree.heading("article_time", text="Time")
        self.article_tree.heading("source_id", text="Source")
        self.article_tree.heading("market_relevance", text="Relevance")
        self.article_tree.heading("title", text="Title")
        self.article_tree.heading("article_url", text="URL")
        self.article_tree.column("article_time", width=170, anchor="w", stretch=False)
        self.article_tree.column("source_id", width=180, anchor="w", stretch=False)
        self.article_tree.column("market_relevance", width=100, anchor="w", stretch=False)
        self.article_tree.column("title", width=560, anchor="w")
        self.article_tree.column("article_url", width=380, anchor="w")
        self.article_tree.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        y_scroll = ttk.Scrollbar(articles, orient="vertical", command=self.article_tree.yview)
        y_scroll.grid(row=1, column=1, sticky="ns", pady=8)
        x_scroll = ttk.Scrollbar(articles, orient="horizontal", command=self.article_tree.xview)
        x_scroll.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        self.article_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.article_tree.bind("<Double-1>", self._on_article_double_click, add="+")

    def _path_row(self, parent: tk.Misc, row: int, label: str, variable: tk.StringVar, *, directory: bool) -> None:
        labeled_entry(parent, row, 0, label, variable)
        ttk.Button(parent, text="Browse", command=lambda: self._browse(variable, directory)).grid(row=row, column=2, padx=8, pady=4)

    def _browse(self, variable: tk.StringVar, directory: bool) -> None:
        path = filedialog.askdirectory(parent=self) if directory else filedialog.askopenfilename(parent=self, filetypes=[("All files", "*.*")])
        if path:
            variable.set(path)

    def _bind_preview(self) -> None:
        for var in (self.config_path_var, self.data_dir_var, self.db_path_var, self.interval_minutes_var):
            var.trace_add("write", lambda *_: self.refresh_preview())
        self.once_var.trace_add("write", lambda *_: self.refresh_preview())

    def _state_for_service(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path_var.get(),
            "data_dir": self.data_dir_var.get(),
            "db_path": self.db_path_var.get(),
            "interval_minutes": self.interval_minutes_var.get(),
            "once": self.once_var.get(),
        }

    def refresh_preview(self) -> None:
        try:
            self.preview_var.set(command_text(self.service.build_command(self.profile, self._state_for_service())))
        except Exception as exc:
            self.preview_var.set(f"Invalid {self.profile.label} configuration: {exc}")

    def start_collector(self) -> None:
        try:
            pid = self.service.start_detached(self.profile, self._state_for_service())
        except Exception as exc:
            messagebox.showerror(self.profile.label, f"Could not start collector:\n{exc}", parent=self)
            return
        self.status_var.set(f"started detached PID {pid}")
        self.pid_var.set(str(pid))
        self.context.emit("save_state", {"reason": f"{self.profile.key}_collector_start"})
        self.refresh_status()

    def request_stop(self) -> None:
        try:
            self.service.request_stop(self.profile, self._state_for_service())
        except Exception as exc:
            messagebox.showerror(self.profile.label, f"Could not request stop:\n{exc}", parent=self)
            return
        self.status_var.set("stop requested")
        self.refresh_status()

    def refresh_status(self) -> None:
        status = self.service.read_status(self.profile, self._state_for_service())
        self.status_var.set(str(status.get("status") or "unknown"))
        self.pid_var.set(str(status.get("pid_text") or status.get("pid") or "-"))
        self.started_at_var.set(str(status.get("started_at") or "-"))
        self.heartbeat_at_var.set(str(status.get("heartbeat_at") or "-"))
        self.last_fetch_at_var.set(str(status.get("last_fetch_at") or "-"))
        self.total_articles_var.set(str(status.get("total_articles") if status.get("total_articles") is not None else "-"))
        self.new_articles_last_cycle_var.set(str(status.get("new_articles_last_cycle") if status.get("new_articles_last_cycle") is not None else "-"))
        self.last_error_var.set(str(status.get("last_error") or "-"))
        try:
            set_tree_rows(self.health_tree, self.service.source_health_rows(self.profile, self._state_for_service()))
        except Exception:
            set_tree_rows(self.health_tree, [])
        self.refresh_articles()

    def refresh_articles(self) -> None:
        if self.article_tree is None:
            return
        self._article_url_by_item.clear()
        for item in self.article_tree.get_children():
            self.article_tree.delete(item)
        try:
            rows = self.service.recent_articles_rows(self.profile, self._state_for_service(), limit=300)
        except Exception:
            rows = []
        for row in rows:
            values = (
                row.get("article_time") or "",
                row.get("source_id") or "",
                row.get("market_relevance") or "",
                row.get("title") or "",
                row.get("article_url") or "",
            )
            item = self.article_tree.insert("", tk.END, values=values)
            self._article_url_by_item[item] = str(row.get("article_url") or "")

    def _selected_article_url(self) -> str:
        if self.article_tree is None:
            return ""
        selected = self.article_tree.selection()
        if not selected:
            return ""
        return self._article_url_by_item.get(selected[0], "").strip()

    def _on_article_double_click(self, _event: tk.Event | None = None) -> None:
        self.open_selected_article()

    def open_selected_article(self) -> None:
        url = self._selected_article_url()
        if not url:
            messagebox.showinfo(self.profile.label, "Select an article row with a URL first.", parent=self)
            return
        try:
            self.service.open_article_url(url)
        except Exception as exc:
            messagebox.showerror(self.profile.label, f"Could not open article URL:\n{exc}", parent=self)

    def open_data_folder(self) -> None:
        try:
            self.service.open_data_folder(self.profile, self._state_for_service())
        except Exception as exc:
            messagebox.showerror(self.profile.label, f"Could not open data folder:\n{exc}", parent=self)

    def open_log(self) -> None:
        try:
            self.service.open_log(self.profile, self._state_for_service())
        except Exception as exc:
            messagebox.showerror(self.profile.label, f"Could not open log:\n{exc}", parent=self)

    def export_articles(self) -> None:
        try:
            path, count = self.service.export_articles_csv(self.profile, self._state_for_service())
        except Exception as exc:
            messagebox.showerror(self.profile.label, f"Could not export CSV:\n{exc}", parent=self)
            return
        messagebox.showinfo(self.profile.label, f"Exported {count} row(s).\n\n{path}", parent=self)

    def export_source_health(self) -> None:
        try:
            path, count = self.service.export_source_health_csv(self.profile, self._state_for_service())
        except Exception as exc:
            messagebox.showerror(self.profile.label, f"Could not export source health:\n{exc}", parent=self)
            return
        messagebox.showinfo(self.profile.label, f"Exported {count} row(s).\n\n{path}", parent=self)

    def get_state(self) -> dict[str, Any]:
        return self._state_for_service()

    def set_state(self, state: dict[str, Any]) -> None:
        paths = self.service.paths(self.profile, "", "", "")
        self.config_path_var.set(str(state.get("config_path") or paths["config"]))
        self.data_dir_var.set(str(state.get("data_dir") or paths["data_dir"]))
        self.db_path_var.set(str(state.get("db_path") or paths["db"]))
        self.interval_minutes_var.set(str(state.get("interval_minutes") or self.profile.default_interval_minutes))
        self.once_var.set(bool(state.get("once", False)))
        self.refresh_preview()

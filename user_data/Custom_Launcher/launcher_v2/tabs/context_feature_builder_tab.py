from __future__ import annotations

from typing import Any
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ..base_tab import BaseTab
from ..services.context_feature_service import ContextFeatureService
from ..ui_helpers import labeled_entry


class ContextFeatureBuilderTab(BaseTab):
    tab_key = "context_feature_builder"
    tab_title = "Context Feature Builder"

    def __init__(self, master: tk.Misc, context: Any) -> None:
        super().__init__(master, context)
        self.service = ContextFeatureService(context.app_dir)
        paths = self.service.paths({})
        self.news_db_var = tk.StringVar(value=str(paths.news_db))
        self.web_db_var = tk.StringVar(value=str(paths.web_db))
        self.global_db_var = tk.StringVar(value=str(paths.global_db))
        self.feature_db_var = tk.StringVar(value=str(paths.feature_db))
        self.export_dir_var = tk.StringVar(value=str(paths.export_dir))
        self.report_dir_var = tk.StringVar(value=str(paths.report_dir))
        self.btc_ohlcv_path_var = tk.StringVar(value=str(paths.btc_ohlcv_path))
        self.overlap_days_var = tk.StringVar(value="7")
        self.status_vars = {
            key: tk.StringVar(value="-")
            for key in (
                "feature_rows",
                "first_feature_hour",
                "last_feature_hour",
                "last_successful_build_time",
                "last_raw_event_seen",
                "feature_column_count",
                "numeric_export_column_count",
            )
        }
        self.result_text: tk.Text | None = None
        self._build_ui()
        self.refresh_status()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        intro = ttk.Label(
            self,
            text="Builds research-only hourly numeric context features from existing News, Web, and Global Context SQL. Raw source databases and strategies are not changed.",
            wraplength=1120,
        )
        intro.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 6))

        paths = ttk.LabelFrame(self, text="Inputs and outputs")
        paths.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        paths.grid_columnconfigure(1, weight=1)
        self._path_row(paths, 0, "News DB", self.news_db_var, directory=False)
        self._path_row(paths, 1, "Web DB", self.web_db_var, directory=False)
        self._path_row(paths, 2, "Global DB", self.global_db_var, directory=False)
        self._path_row(paths, 3, "Feature DB", self.feature_db_var, directory=False)
        self._path_row(paths, 4, "Parquet export dir", self.export_dir_var, directory=True)
        self._path_row(paths, 5, "Report dir", self.report_dir_var, directory=True)
        self._path_row(paths, 6, "BTC 1h OHLCV", self.btc_ohlcv_path_var, directory=False)
        labeled_entry(paths, 7, 0, "Overlap days", self.overlap_days_var, width=12)

        buttons = ttk.Frame(self)
        buttons.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(buttons, text="Refresh", command=self.refresh_status).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Dry Run", command=self.dry_run).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Update Features", command=self.update_features).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Full Rebuild", command=self.full_rebuild).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Export Parquet", command=self.export_parquet).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Validate No-Lookahead", command=self.validate_alignment).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Create BTC Report", command=self.create_report).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Open Feature Folder", command=self.open_feature_folder).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Open Export Folder", command=self.open_export_folder).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Open Report Folder", command=self.open_report_folder).pack(side="left")

        status = ttk.LabelFrame(self, text="Feature store status")
        status.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))
        status.grid_columnconfigure(1, weight=1)
        status.grid_columnconfigure(3, weight=1)
        labels = [
            ("Rows", "feature_rows"),
            ("First hour", "first_feature_hour"),
            ("Last hour", "last_feature_hour"),
            ("Last build", "last_successful_build_time"),
            ("Last raw event", "last_raw_event_seen"),
            ("Feature columns", "feature_column_count"),
            ("Numeric export columns", "numeric_export_column_count"),
        ]
        for index, (label, key) in enumerate(labels):
            row = index // 2
            col = (index % 2) * 2
            ttk.Label(status, text=f"{label}:").grid(row=row, column=col, sticky="w", padx=8, pady=4)
            ttk.Label(status, textvariable=self.status_vars[key]).grid(row=row, column=col + 1, sticky="w", padx=8, pady=4)

        result = ttk.LabelFrame(self, text="Latest action")
        result.grid(row=4, column=0, sticky="nsew", padx=8, pady=(0, 8))
        result.grid_columnconfigure(0, weight=1)
        result.grid_rowconfigure(0, weight=1)
        self.result_text = tk.Text(result, height=12, wrap="word")
        self.result_text.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        scroll = ttk.Scrollbar(result, orient="vertical", command=self.result_text.yview)
        scroll.grid(row=0, column=1, sticky="ns", pady=8)
        self.result_text.configure(yscrollcommand=scroll.set)

    def _path_row(self, parent: tk.Misc, row: int, label: str, variable: tk.StringVar, *, directory: bool) -> None:
        labeled_entry(parent, row, 0, label, variable)
        ttk.Button(parent, text="Browse", command=lambda: self._browse(variable, directory)).grid(row=row, column=2, padx=8, pady=4)

    def _browse(self, variable: tk.StringVar, directory: bool) -> None:
        path = filedialog.askdirectory(parent=self) if directory else filedialog.askopenfilename(parent=self, filetypes=[("All files", "*.*")])
        if path:
            variable.set(path)

    def _state(self) -> dict[str, Any]:
        return {
            "news_db": self.news_db_var.get(),
            "web_db": self.web_db_var.get(),
            "global_db": self.global_db_var.get(),
            "feature_db": self.feature_db_var.get(),
            "export_dir": self.export_dir_var.get(),
            "report_dir": self.report_dir_var.get(),
            "btc_ohlcv_path": self.btc_ohlcv_path_var.get(),
            "overlap_days": self.overlap_days_var.get(),
        }

    def refresh_status(self) -> None:
        try:
            status = self.service.status(self._state())
        except Exception as exc:
            self._show_result({"error": str(exc)})
            return
        for key, variable in self.status_vars.items():
            variable.set(str(status.get(key) if status.get(key) is not None else "-"))

    def dry_run(self) -> None:
        self._run_action("Dry run", self.service.dry_run)

    def update_features(self) -> None:
        self._run_action("Update features", self.service.update)

    def full_rebuild(self) -> None:
        if not messagebox.askyesno("Context Feature Builder", "Full rebuild replaces derived feature rows. Raw source databases are not changed. Continue?", parent=self):
            return
        self._run_action("Full rebuild", self.service.full_rebuild)

    def export_parquet(self) -> None:
        try:
            path, count = self.service.export(self._state())
        except Exception as exc:
            messagebox.showerror("Context Feature Builder", f"Export failed:\n{exc}", parent=self)
            return
        self._show_result({"export_path": str(path), "rows": count})
        messagebox.showinfo("Context Feature Builder", f"Exported {count} row(s).\n\n{path}", parent=self)

    def validate_alignment(self) -> None:
        self._run_action("Validate no-lookahead", self.service.validate)

    def create_report(self) -> None:
        try:
            path = self.service.report(self._state())
        except Exception as exc:
            messagebox.showerror("Context Feature Builder", f"Report failed:\n{exc}", parent=self)
            return
        self._show_result({"report_path": str(path)})
        messagebox.showinfo("Context Feature Builder", f"Created report:\n\n{path}", parent=self)

    def open_feature_folder(self) -> None:
        self.service.open_feature_folder(self._state())

    def open_export_folder(self) -> None:
        self.service.open_export_folder(self._state())

    def open_report_folder(self) -> None:
        self.service.open_report_folder(self._state())

    def _run_action(self, label: str, func: Any) -> None:
        try:
            result = func(self._state())
        except Exception as exc:
            messagebox.showerror("Context Feature Builder", f"{label} failed:\n{exc}", parent=self)
            self._show_result({"error": str(exc)})
            return
        self._show_result(result)
        self.refresh_status()

    def _show_result(self, result: Any) -> None:
        if self.result_text is None:
            return
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, json.dumps(result, indent=2, sort_keys=True, default=str))

    def get_state(self) -> dict[str, Any]:
        return self._state()

    def set_state(self, state: dict[str, Any]) -> None:
        defaults = self.service.paths({})
        self.news_db_var.set(str(state.get("news_db") or defaults.news_db))
        self.web_db_var.set(str(state.get("web_db") or defaults.web_db))
        self.global_db_var.set(str(state.get("global_db") or defaults.global_db))
        self.feature_db_var.set(str(state.get("feature_db") or defaults.feature_db))
        self.export_dir_var.set(str(state.get("export_dir") or defaults.export_dir))
        self.report_dir_var.set(str(state.get("report_dir") or defaults.report_dir))
        self.btc_ohlcv_path_var.set(str(state.get("btc_ohlcv_path") or defaults.btc_ohlcv_path))
        self.overlap_days_var.set(str(state.get("overlap_days") or "7"))
        self.refresh_status()

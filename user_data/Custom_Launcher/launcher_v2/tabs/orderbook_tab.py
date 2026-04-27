from __future__ import annotations

from typing import Any
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk

from ..base_tab import BaseTab
from ..command_builder import command_text, freqtrade_command
from ..services.orderbook_service import OrderBookService, tokens
from ..ui_helpers import labeled_entry, set_tree_rows
from .pairs_tab import parse_pairs


class OrderBookTab(BaseTab):
    tab_key = "orderbook"
    tab_title = "Order Book Lab"

    def __init__(self, master: tk.Misc, context: Any) -> None:
        super().__init__(master, context)
        self.service = OrderBookService(context.app_dir, context.shared.python_exe.get())
        paths = self.service.paths({})
        self.config_path_var = tk.StringVar(value=str(paths["config"]))
        self.data_dir_var = tk.StringVar(value=str(paths["data_dir"]))
        self.exchange_var = tk.StringVar(value="Binance USD-M Futures")
        self.market_type_var = tk.StringVar(value="futures")
        self.depth_levels_var = tk.StringVar(value="20")
        self.stream_update_ms_var = tk.StringVar(value="500")
        self.metric_interval_seconds_var = tk.StringVar(value="1")
        self.snapshot_interval_seconds_var = tk.StringVar(value="60")
        self.capacity_warning_mb_var = tk.StringVar(value="500")
        self.capacity_critical_mb_var = tk.StringVar(value="2000")
        self.max_symbols_var = tk.StringVar(value="12")
        self.store_snapshots_var = tk.BooleanVar(value=True)
        self.collector_preview_var = tk.StringVar(value="")
        self.status_vars = {key: tk.StringVar(value="-") for key in ("status", "pid", "started_at", "heartbeat_at", "last_message_at", "last_metric_at", "pair_count", "active_streams", "messages", "metrics", "db_mb", "data_dir_mb", "capacity", "last_error")}
        self.estimate_vars = {key: tk.StringVar(value="-") for key in ("metric_rows", "snapshot_rows", "mb_per_day", "days_to_warning", "pair_count", "symbol_count")}
        self.pair_warning_var = tk.StringVar(value="")

        self.history_datadir_var = tk.StringVar(value=str(context.app_dir / "../data/bybit_orderbook"))
        self.history_exchange_var = tk.StringVar(value="bybit")
        self.history_trading_mode_var = tk.StringVar(value="futures")
        self.history_category_var = tk.StringVar(value="linear")
        self.history_depth_var = tk.StringVar(value="500")
        self.history_timerange_var = tk.StringVar(value="")
        self.history_feature_timeframes_var = tk.StringVar(value="1h")
        self.history_feature_format_var = tk.StringVar(value="feather")
        self.history_max_rows_var = tk.StringVar(value="")
        self.history_erase_var = tk.BooleanVar(value=False)
        self.history_download_preview_var = tk.StringVar(value="")
        self.history_convert_preview_var = tk.StringVar(value="")
        self.history_status_var = tk.StringVar(value="Download raw archives to create availability reports, then refresh the summary.")
        self._build_ui()
        self._bind_refresh()
        self.refresh_pair_preview()
        self.refresh_history_previews()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        live_tab = ttk.Frame(notebook)
        history_tab = ttk.Frame(notebook)
        notebook.add(live_tab, text="Live Collector")
        notebook.add(history_tab, text="Bybit History")
        self._build_live_tab(live_tab)
        self._build_history_tab(history_tab)

    def _build_live_tab(self, root: ttk.Frame) -> None:
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(5, weight=1)
        settings = ttk.LabelFrame(root, text="Collector settings")
        settings.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        settings.grid_columnconfigure(1, weight=1)
        settings.grid_columnconfigure(3, weight=1)
        self._path_row(settings, 0, "Config path", self.config_path_var, directory=False)
        self._path_row(settings, 1, "Data dir", self.data_dir_var, directory=True)
        ttk.Label(settings, text="Exchange").grid(row=2, column=0, sticky="w", padx=8, pady=4)
        ttk.Combobox(settings, textvariable=self.exchange_var, values=["Binance USD-M Futures"], state="readonly").grid(row=2, column=1, sticky="ew", padx=8, pady=4)
        ttk.Label(settings, text="Market type").grid(row=2, column=2, sticky="w", padx=8, pady=4)
        ttk.Combobox(settings, textvariable=self.market_type_var, values=["futures"], state="readonly").grid(row=2, column=3, sticky="ew", padx=8, pady=4)
        rows = [
            ("Depth levels", self.depth_levels_var, "Stream update ms", self.stream_update_ms_var),
            ("Metric interval seconds", self.metric_interval_seconds_var, "Snapshot interval seconds", self.snapshot_interval_seconds_var),
            ("Capacity warning MB", self.capacity_warning_mb_var, "Capacity critical MB", self.capacity_critical_mb_var),
            ("Max symbols", self.max_symbols_var, "", None),
        ]
        for offset, (left_label, left_var, right_label, right_var) in enumerate(rows, start=3):
            labeled_entry(settings, offset, 0, left_label, left_var)
            if right_var is not None:
                labeled_entry(settings, offset, 2, right_label, right_var)
        ttk.Checkbutton(settings, text="Store snapshots", variable=self.store_snapshots_var).grid(row=7, column=0, columnspan=2, sticky="w", padx=8, pady=4)

        preview = ttk.LabelFrame(root, text="Generated command")
        preview.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        preview.grid_columnconfigure(0, weight=1)
        ttk.Entry(preview, textvariable=self.collector_preview_var).grid(row=0, column=0, sticky="ew", padx=8, pady=8)

        actions = ttk.Frame(root)
        actions.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(actions, text="Start Detached Collector", command=self.start_collector).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Request Stop", command=self.request_stop).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Refresh Status", command=self.refresh_status).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Open Data Folder", command=self.open_data_folder).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Open Log", command=self.open_log).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Export Latest Metrics CSV", command=self.export_latest_metrics).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Normalize Whitelist", command=self.normalize_main_whitelist).pack(side="left")

        status = ttk.LabelFrame(root, text="Status")
        status.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))
        items = [("Status", "status"), ("PID", "pid"), ("Started at", "started_at"), ("Heartbeat at", "heartbeat_at"), ("Last message at", "last_message_at"), ("Last metric at", "last_metric_at"), ("Pair count", "pair_count"), ("Active streams", "active_streams"), ("Messages", "messages"), ("Metrics", "metrics"), ("DB MB", "db_mb"), ("Data dir MB", "data_dir_mb"), ("Capacity", "capacity"), ("Last error", "last_error")]
        for index, (label, key) in enumerate(items):
            row = index // 2
            col = (index % 2) * 2
            ttk.Label(status, text=f"{label}:").grid(row=row, column=col, sticky="w", padx=8, pady=3)
            ttk.Label(status, textvariable=self.status_vars[key]).grid(row=row, column=col + 1, sticky="w", padx=8, pady=3)

        estimate = ttk.LabelFrame(root, text="Storage estimate")
        estimate.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 8))
        estimate_items = [("Metric rows/day", "metric_rows"), ("Snapshot rows/day", "snapshot_rows"), ("Estimated MB/day", "mb_per_day"), ("Days to warning", "days_to_warning"), ("Whitelist pairs", "pair_count"), ("Active symbols", "symbol_count")]
        for index, (label, key) in enumerate(estimate_items):
            ttk.Label(estimate, text=f"{label}:").grid(row=0, column=index * 2, sticky="w", padx=8, pady=6)
            ttk.Label(estimate, textvariable=self.estimate_vars[key]).grid(row=0, column=index * 2 + 1, sticky="w", padx=8, pady=6)

        lower = ttk.Panedwindow(root, orient=tk.VERTICAL)
        lower.grid(row=5, column=0, sticky="nsew", padx=8, pady=(0, 8))
        pair_frame = ttk.LabelFrame(lower, text="Whitelist preview")
        pair_frame.grid_columnconfigure(0, weight=1)
        pair_frame.grid_rowconfigure(0, weight=1)
        self.pair_tree = ttk.Treeview(pair_frame, columns=("pair", "symbol", "status"), show="headings", height=5)
        for column in ("pair", "symbol", "status"):
            self.pair_tree.heading(column, text=column)
            self.pair_tree.column(column, width=170, anchor="w")
        self.pair_tree.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        ttk.Label(pair_frame, textvariable=self.pair_warning_var).grid(row=1, column=0, sticky="w", padx=8, pady=(0, 8))
        lower.add(pair_frame, weight=1)

        metrics = ttk.LabelFrame(lower, text="Latest metrics")
        metrics.grid_columnconfigure(0, weight=1)
        metrics.grid_rowconfigure(0, weight=1)
        columns = ("pair", "symbol", "status", "best_bid", "best_ask", "spread_bps", "imbalance_top20", "bid_pressure_ratio_60s", "ask_pressure_ratio_60s", "nearest_bid_wall_distance_bps", "nearest_ask_wall_distance_bps", "last_metric_at")
        self.metrics_tree = ttk.Treeview(metrics, columns=columns, show="headings", height=8)
        for column in columns:
            self.metrics_tree.heading(column, text=column)
            self.metrics_tree.column(column, width=150, anchor="w")
        self.metrics_tree.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        lower.add(metrics, weight=2)

    def _build_history_tab(self, root: ttk.Frame) -> None:
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(5, weight=1)
        settings = ttk.LabelFrame(root, text="Bybit archive settings")
        settings.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        settings.grid_columnconfigure(1, weight=1)
        settings.grid_columnconfigure(3, weight=1)
        self._path_row(settings, 0, "Archive datadir", self.history_datadir_var, directory=True)
        labeled_entry(settings, 1, 0, "Exchange", self.history_exchange_var)
        labeled_entry(settings, 1, 2, "Trading mode", self.history_trading_mode_var)
        labeled_entry(settings, 2, 0, "Category", self.history_category_var)
        labeled_entry(settings, 2, 2, "Depth", self.history_depth_var)
        labeled_entry(settings, 3, 0, "Timerange", self.history_timerange_var)
        labeled_entry(settings, 3, 2, "Feature timeframes", self.history_feature_timeframes_var)
        ttk.Label(settings, text="Feature format").grid(row=4, column=0, sticky="w", padx=8, pady=4)
        ttk.Combobox(settings, textvariable=self.history_feature_format_var, values=["feather", "parquet"], state="readonly").grid(row=4, column=1, sticky="ew", padx=8, pady=4)
        labeled_entry(settings, 4, 2, "Max rows", self.history_max_rows_var)
        ttk.Checkbutton(settings, text="Redownload matching raw archives", variable=self.history_erase_var).grid(row=5, column=0, columnspan=2, sticky="w", padx=8, pady=4)

        pair_frame = ttk.LabelFrame(root, text="Archive pairs")
        pair_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        pair_frame.grid_columnconfigure(0, weight=1)
        pair_frame.grid_rowconfigure(1, weight=1)
        pair_buttons = ttk.Frame(pair_frame)
        pair_buttons.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        ttk.Button(pair_buttons, text="Add typed", command=self.add_history_pairs).pack(side="left", padx=(0, 6))
        ttk.Button(pair_buttons, text="Remove selected", command=self.remove_selected_history_pairs).pack(side="left", padx=(0, 6))
        ttk.Button(pair_buttons, text="Normalize", command=self.normalize_history_pairs).pack(side="left", padx=(0, 6))
        ttk.Button(pair_buttons, text="Use whitelist", command=self.copy_whitelist_to_history_pairs).pack(side="left")
        self.history_pairs_text = scrolledtext.ScrolledText(pair_frame, wrap="word", height=6)
        self.history_pairs_text.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.history_pairs_text.bind("<KeyRelease>", lambda _event: self.refresh_history_previews())

        preview = ttk.LabelFrame(root, text="Generated commands")
        preview.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        preview.grid_columnconfigure(1, weight=1)
        ttk.Label(preview, text="Download raw").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(preview, textvariable=self.history_download_preview_var).grid(row=0, column=1, sticky="ew", padx=8, pady=4)
        ttk.Label(preview, text="Convert features").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(preview, textvariable=self.history_convert_preview_var).grid(row=1, column=1, sticky="ew", padx=8, pady=4)

        actions = ttk.Frame(root)
        actions.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(actions, text="Download Raw Archives", command=self.run_history_download).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Convert To Features", command=self.run_history_convert).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Refresh Summary", command=self.refresh_history_summary).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Open Data Folder", command=self.open_history_data_folder).pack(side="left")

        summary = ttk.LabelFrame(root, text="Archive summary")
        summary.grid(row=5, column=0, sticky="nsew", padx=8, pady=(0, 8))
        summary.grid_columnconfigure(0, weight=1)
        summary.grid_rowconfigure(1, weight=1)
        ttk.Label(summary, textvariable=self.history_status_var).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 0))
        columns = ("pair", "symbol", "available_days", "missing_days", "first_available", "last_available", "raw_archives", "feature_timeframes")
        self.history_summary_tree = ttk.Treeview(summary, columns=columns, show="headings", height=8)
        for column in columns:
            self.history_summary_tree.heading(column, text=column)
            self.history_summary_tree.column(column, width=145, anchor="w")
        self.history_summary_tree.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

    def _path_row(self, parent: tk.Misc, row: int, label: str, variable: tk.StringVar, *, directory: bool) -> None:
        labeled_entry(parent, row, 0, label, variable)
        ttk.Button(parent, text="Browse", command=lambda: self._browse(variable, directory)).grid(row=row, column=2, padx=8, pady=4)

    def _browse(self, variable: tk.StringVar, directory: bool) -> None:
        path = filedialog.askdirectory(parent=self) if directory else filedialog.askopenfilename(parent=self, filetypes=[("All files", "*.*")])
        if path:
            variable.set(path)

    def _bind_refresh(self) -> None:
        for var in (self.config_path_var, self.data_dir_var, self.depth_levels_var, self.stream_update_ms_var, self.metric_interval_seconds_var, self.snapshot_interval_seconds_var, self.capacity_warning_mb_var, self.capacity_critical_mb_var, self.max_symbols_var):
            var.trace_add("write", lambda *_: self.refresh_pair_preview())
        self.store_snapshots_var.trace_add("write", lambda *_: self.refresh_pair_preview())
        for var in (self.history_datadir_var, self.history_exchange_var, self.history_trading_mode_var, self.history_category_var, self.history_depth_var, self.history_timerange_var, self.history_feature_timeframes_var, self.history_feature_format_var, self.history_max_rows_var):
            var.trace_add("write", lambda *_: self.refresh_history_previews())
        self.history_erase_var.trace_add("write", lambda *_: self.refresh_history_previews())

    def _state(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path_var.get(),
            "data_dir": self.data_dir_var.get(),
            "depth_levels": self.depth_levels_var.get(),
            "stream_update_ms": self.stream_update_ms_var.get(),
            "metric_interval_seconds": self.metric_interval_seconds_var.get(),
            "snapshot_interval_seconds": self.snapshot_interval_seconds_var.get(),
            "capacity_warning_mb": self.capacity_warning_mb_var.get(),
            "capacity_critical_mb": self.capacity_critical_mb_var.get(),
            "max_symbols": self.max_symbols_var.get(),
            "store_snapshots": self.store_snapshots_var.get(),
            "history_datadir": self.history_datadir_var.get(),
            "history_exchange": self.history_exchange_var.get(),
            "history_trading_mode": self.history_trading_mode_var.get(),
            "history_category": self.history_category_var.get(),
            "history_depth": self.history_depth_var.get(),
            "history_timerange": self.history_timerange_var.get(),
            "history_feature_timeframes": self.history_feature_timeframes_var.get(),
            "history_feature_format": self.history_feature_format_var.get(),
            "history_max_rows": self.history_max_rows_var.get(),
            "history_erase": self.history_erase_var.get(),
            "history_pairs": self.history_pairs_text.get("1.0", tk.END).strip() if hasattr(self, "history_pairs_text") else "",
        }

    def _main_pairs(self) -> list[str]:
        pairs_tab = self.context.registry.get("pairs")
        if pairs_tab is None:
            return []
        return parse_pairs(str(pairs_tab.get_state().get("pairs") or ""))

    def refresh_pair_preview(self) -> None:
        state = self._state()
        valid, preview = self.service.normalized_pairs(self._main_pairs(), state)
        set_tree_rows(self.pair_tree, [(row["pair"], row["symbol"], row["status"]) for row in preview] if hasattr(self, "pair_tree") else [])
        estimate = self.service.estimate(len(valid), state)
        self.estimate_vars["metric_rows"].set(f"{estimate['metric_rows_per_day']:.0f}")
        self.estimate_vars["snapshot_rows"].set(f"{estimate['snapshot_rows_per_day']:.0f}")
        self.estimate_vars["mb_per_day"].set(f"{estimate['estimated_total_mb_per_day']:.2f}")
        warning = max(1.0, float(state["capacity_warning_mb"] or 500))
        mb_day = float(estimate["estimated_total_mb_per_day"])
        self.estimate_vars["days_to_warning"].set("-" if mb_day <= 0 else f"{warning / mb_day:.1f}")
        self.estimate_vars["pair_count"].set(f"Whitelist pairs: {len(self._main_pairs())}")
        self.estimate_vars["symbol_count"].set(f"Active symbols: {len(valid)}")
        self.pair_warning_var.set("" if valid else "No usable whitelist pairs found. Add pairs on the Pairs tab first.")
        try:
            self.collector_preview_var.set(command_text(self.service.build_collector_command(state, self._main_pairs())))
        except Exception as exc:
            self.collector_preview_var.set(f"Invalid Order Book configuration: {exc}")

    def start_collector(self) -> None:
        try:
            pid = self.service.start_detached(self._state(), self._main_pairs())
        except Exception as exc:
            messagebox.showerror("Order Book Lab", f"Cannot start Order Book Lab:\n{exc}", parent=self)
            return
        self.status_vars["status"].set(f"started detached PID {pid}")
        self.status_vars["pid"].set(str(pid))
        self.context.emit("save_state", {"reason": "orderbook_start"})
        self.refresh_status()

    def request_stop(self) -> None:
        try:
            self.service.request_stop(self._state())
        except Exception as exc:
            messagebox.showerror("Order Book Lab", f"Could not request stop:\n{exc}", parent=self)
            return
        self.status_vars["status"].set("stop requested")

    def refresh_status(self) -> None:
        status = self.service.read_status(self._state())
        mapping = {
            "status": status.get("status") or "unknown",
            "pid": status.get("pid_text") or status.get("pid") or "-",
            "started_at": status.get("started_at") or "-",
            "heartbeat_at": status.get("heartbeat_at") or "-",
            "last_message_at": status.get("last_message_at") or "-",
            "last_metric_at": status.get("last_metric_at") or "-",
            "pair_count": status.get("pair_count") if status.get("pair_count") is not None else "-",
            "active_streams": status.get("active_streams") if status.get("active_streams") is not None else "-",
            "messages": status.get("message_count_total") if status.get("message_count_total") is not None else "-",
            "metrics": status.get("metric_count_total") if status.get("metric_count_total") is not None else "-",
            "db_mb": status.get("db_mb") if status.get("db_mb") is not None else "-",
            "data_dir_mb": status.get("data_dir_mb") if status.get("data_dir_mb") is not None else "-",
            "capacity": status.get("capacity_level") or "-",
            "last_error": status.get("last_error") or "-",
        }
        for key, value in mapping.items():
            self.status_vars[key].set(str(value))
        try:
            set_tree_rows(self.metrics_tree, self.service.latest_metric_rows(self._state()))
        except Exception:
            set_tree_rows(self.metrics_tree, [])
        self.refresh_pair_preview()

    def open_data_folder(self) -> None:
        self.service.open_data_folder(self._state())

    def open_log(self) -> None:
        self.service.open_log(self._state())

    def export_latest_metrics(self) -> None:
        try:
            path, count = self.service.export_latest_metrics(self._state())
        except Exception as exc:
            messagebox.showerror("Order Book Lab", f"Could not export metrics:\n{exc}", parent=self)
            return
        messagebox.showinfo("Order Book Lab", f"Exported {count} rows.\n\n{path}", parent=self)

    def normalize_main_whitelist(self) -> None:
        pairs_tab = self.context.registry.get("pairs")
        if pairs_tab is not None and hasattr(pairs_tab, "_normalize"):
            pairs_tab._normalize(pairs_tab.pairs_text)
        self.refresh_pair_preview()

    def _set_history_pairs(self, pairs: list[str]) -> None:
        self.history_pairs_text.delete("1.0", tk.END)
        if pairs:
            self.history_pairs_text.insert("1.0", "\n".join(pairs))
        self.refresh_history_previews()

    def add_history_pairs(self) -> None:
        value = simpledialog.askstring("Archive pairs", "Pair(s) to add, comma/space/newline separated:", parent=self)
        if value:
            self._set_history_pairs(parse_pairs(self.history_pairs_text.get("1.0", tk.END) + "\n" + value))

    def remove_selected_history_pairs(self) -> None:
        try:
            start = self.history_pairs_text.index("sel.first")
            end = self.history_pairs_text.index("sel.last")
            self.history_pairs_text.delete(start, end)
        except tk.TclError:
            line = self.history_pairs_text.index("insert").split(".", 1)[0]
            self.history_pairs_text.delete(f"{line}.0", f"{line}.end+1c")
        self.normalize_history_pairs()

    def normalize_history_pairs(self) -> None:
        self._set_history_pairs(parse_pairs(self.history_pairs_text.get("1.0", tk.END)))

    def copy_whitelist_to_history_pairs(self) -> None:
        self._set_history_pairs(self._main_pairs())

    def _fallback_datadir(self) -> str:
        common = self.context.registry.get("common")
        if common is not None:
            state = common.get_state()
            return str(state.get("datadir") or "")
        return self.context.shared.datadir.get()

    def refresh_history_previews(self) -> None:
        state = self._state()
        pairs = self._main_pairs()
        try:
            command = freqtrade_command(self.context.shared.python_exe.get(), self.service.build_history_download_args(state, pairs, self._fallback_datadir()))
            self.history_download_preview_var.set(command_text(command.preview_command))
        except Exception as exc:
            self.history_download_preview_var.set(f"Invalid Bybit history download configuration: {exc}")
        try:
            command = freqtrade_command(self.context.shared.python_exe.get(), self.service.build_history_convert_args(state, pairs, self._fallback_datadir()))
            self.history_convert_preview_var.set(command_text(command.preview_command))
        except Exception as exc:
            self.history_convert_preview_var.set(f"Invalid Bybit history convert configuration: {exc}")

    def run_history_download(self) -> None:
        try:
            args = self.service.build_history_download_args(self._state(), self._main_pairs(), self._fallback_datadir())
        except Exception as exc:
            messagebox.showerror("Order Book Lab", f"Cannot build Bybit History download command:\n{exc}", parent=self)
            return
        result = freqtrade_command(self.context.shared.python_exe.get(), args)
        self.context.emit("save_state", {"reason": "orderbook_history_download"})
        self.context.process_runner.run(result.preview_command, cwd=self.context.shared.project_root.get() or None, owner=self.tab_key)

    def run_history_convert(self) -> None:
        try:
            args = self.service.build_history_convert_args(self._state(), self._main_pairs(), self._fallback_datadir())
        except Exception as exc:
            messagebox.showerror("Order Book Lab", f"Cannot build Bybit History convert command:\n{exc}", parent=self)
            return
        result = freqtrade_command(self.context.shared.python_exe.get(), args)
        self.context.emit("save_state", {"reason": "orderbook_history_convert"})
        self.context.process_runner.run(result.preview_command, cwd=self.context.shared.project_root.get() or None, owner=self.tab_key)

    def refresh_history_summary(self) -> None:
        try:
            rows, status = self.service.history_summary_rows(self._state(), self._main_pairs(), self._fallback_datadir())
        except Exception as exc:
            rows, status = [], f"Could not refresh summary: {exc}"
        set_tree_rows(self.history_summary_tree, rows)
        self.history_status_var.set(status)

    def open_history_data_folder(self) -> None:
        from ..services.collector_service import open_path

        path = self.service.history_datadir(self._state(), self._fallback_datadir()).resolve()
        path.mkdir(parents=True, exist_ok=True)
        open_path(path)

    def get_state(self) -> dict[str, Any]:
        return self._state()

    def set_state(self, state: dict[str, Any]) -> None:
        paths = self.service.paths({})
        self.config_path_var.set(str(state.get("config_path") or paths["config"]))
        self.data_dir_var.set(str(state.get("data_dir") or paths["data_dir"]))
        self.depth_levels_var.set(str(state.get("depth_levels") or "20"))
        self.stream_update_ms_var.set(str(state.get("stream_update_ms") or "500"))
        self.metric_interval_seconds_var.set(str(state.get("metric_interval_seconds") or "1"))
        self.snapshot_interval_seconds_var.set(str(state.get("snapshot_interval_seconds") or "60"))
        self.capacity_warning_mb_var.set(str(state.get("capacity_warning_mb") or "500"))
        self.capacity_critical_mb_var.set(str(state.get("capacity_critical_mb") or "2000"))
        self.max_symbols_var.set(str(state.get("max_symbols") or "12"))
        self.store_snapshots_var.set(bool(state.get("store_snapshots", True)))
        self.history_datadir_var.set(str(state.get("history_datadir") or self.history_datadir_var.get()))
        self.history_exchange_var.set(str(state.get("history_exchange") or "bybit"))
        self.history_trading_mode_var.set(str(state.get("history_trading_mode") or "futures"))
        self.history_category_var.set(str(state.get("history_category") or "linear"))
        self.history_depth_var.set(str(state.get("history_depth") or "500"))
        self.history_timerange_var.set(str(state.get("history_timerange") or ""))
        self.history_feature_timeframes_var.set(str(state.get("history_feature_timeframes") or "1h"))
        self.history_feature_format_var.set(str(state.get("history_feature_format") or "feather"))
        self.history_max_rows_var.set(str(state.get("history_max_rows") or ""))
        self.history_erase_var.set(bool(state.get("history_erase", False)))
        self._set_history_pairs(parse_pairs(str(state.get("history_pairs") or "")))

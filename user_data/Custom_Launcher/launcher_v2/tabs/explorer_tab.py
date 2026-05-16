from __future__ import annotations

import json
import math
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from ..base_tab import BaseTab
from ..services.collector_service import open_path
from ..services.entry_sieve_service import EntrySieveService, EntrySieveSettings
from ..services.explorer_service import ExplorerRunSettings, ExplorerService
from ..ui_helpers import labeled_entry, set_tree_rows


def _split_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").replace(";", ",").replace("\n", ",")
    return [item.strip() for item in text.split(",") if item.strip()]


SIEVE_RESULT_COLUMNS = (
    "score",
    "strategy_batch",
    "speed_run_mode",
    "strategy",
    "side",
    "core_behavior",
    "training_window",
    "validation_window",
    "take_profit_pct",
    "stoploss_pct",
    "status",
    "analysis_read",
    "analysis_next",
    "hyperopt_loss",
    "objective",
    "best_params_count",
    "epoch_count",
    "profit_total_abs",
    "profit_total",
    "trade_count",
    "winrate",
    "max_drawdown_pct",
    "backtest_file",
    "params_file",
)

SIEVE_DEFAULT_COLUMN_ORDER = (
    "score",
    "strategy_batch",
    "speed_run_mode",
    "winrate",
    "max_drawdown_pct",
    "profit_total_abs",
    "profit_total",
    "trade_count",
    "objective",
    "hyperopt_loss",
    "best_params_count",
    "epoch_count",
    "take_profit_pct",
    "stoploss_pct",
    "status",
    "strategy",
    "side",
    "core_behavior",
    "training_window",
    "validation_window",
    "analysis_read",
    "analysis_next",
    "backtest_file",
    "params_file",
)


class ExplorerTab(BaseTab):
    """Normal Explorer controls: family/mode target, breadth, windows, loop size."""

    tab_key = "explorer"
    tab_title = "Explorer"

    def __init__(self, master: tk.Misc, context: Any) -> None:
        super().__init__(master, context)
        self.service = ExplorerService(context.app_dir, context.shared.python_exe.get())
        self.sieve_service = EntrySieveService(context.app_dir, context.shared.python_exe.get())
        self.preset_name_var = tk.StringVar(value="LauncherV2-auto")
        self.target_type_var = tk.StringVar(value="family")
        self.target_selection_var = tk.StringVar(value="random")
        self.target_name_var = tk.StringVar(value="")
        self.search_breadth_var = tk.StringVar(value="targeted")
        self.max_loops_var = tk.StringVar(value="0")
        self.epochs_var = tk.StringVar(value="200")
        self.auto_epochs_var = tk.BooleanVar(value=False)
        self.auto_epochs_cap_var = tk.StringVar(value="")
        self.random_state_var = tk.StringVar(value="")
        self.sampling_seed_var = tk.StringVar(value="")
        default_settings = ExplorerRunSettings.from_state({}, context.app_dir)
        self.split_venv_pipeline_var = tk.BooleanVar(value=False)
        self.backtest_python_exe_var = tk.StringVar(value=default_settings.backtest_python_exe)
        self.backtest_worker_count_var = tk.StringVar(value=default_settings.backtest_worker_count)
        self.pipeline_handoff_dir_var = tk.StringVar(value=default_settings.pipeline_handoff_dir)
        self.sieve_strategy_filter_var = tk.StringVar(value="sieve1_*.py")
        self.sieve_strategy_batch_var = tk.StringVar(value="all")
        self.sieve_batch_queue_var = tk.StringVar(value="volume_profile,structure_levels,continuation_patterns,reversal_patterns,market_state_pressure,multi_confluence,small_concepts,avwap,zones")
        self.sieve_batch_priority_var = tk.StringVar(value="least_run_first")
        self.sieve_speed_run_var = tk.BooleanVar(value=False)
        self.sieve_speed_pair_count_var = tk.StringVar(value="5")
        self.sieve_take_profit_var = tk.StringVar(value="2")
        self.sieve_stoploss_var = tk.StringVar(value="2")
        self.sieve_auto_windows_var = tk.BooleanVar(value=True)
        self.sieve_auto_window_count_var = tk.StringVar(value="2")
        self.sieve_target_sweep_var = tk.BooleanVar(value=False)
        self.sieve_target_pairs_var = tk.StringVar(value="1/1, 1.5/1.5, 2/2, 3/2, 4/2, 2/3, 3/3")
        self.sieve_result_batch_var = tk.StringVar(value="")
        self.sieve_result_batch_filter_var = tk.StringVar(value="")
        self.sieve_filter_var = tk.StringVar(value="")
        self.sieve_filter_winrate_min_var = tk.StringVar(value="")
        self.sieve_filter_profit_min_var = tk.StringVar(value="")
        self.sieve_filter_drawdown_max_var = tk.StringVar(value="")
        self.sieve_filter_trades_min_var = tk.StringVar(value="")
        self.sieve_filter_tp_eq_var = tk.StringVar(value="")
        self.sieve_filter_sl_eq_var = tk.StringVar(value="")
        self.sieve_status_var = tk.StringVar(value="Run status: idle")
        self._sieve_sort_column = "score"
        self._sieve_sort_reverse = True
        self._catalog_rows: list[dict[str, Any]] = []
        self._window_rows: list[dict[str, str]] = []
        self.specific_target_combo: ttk.Combobox | None = None
        self.catalog_tree: ttk.Treeview | None = None
        self.coverage_tree: ttk.Treeview | None = None
        self.target_params_tree: ttk.Treeview | None = None
        self.open_support_params_tree: ttk.Treeview | None = None
        self.sieve_result_batch_combo: ttk.Combobox | None = None
        self.sieve_strategy_batch_combo: ttk.Combobox | None = None
        self.sieve_results_tree: ttk.Treeview | None = None
        self._sieve_result_batch_ids: list[str] = []
        self.sieve_result_columns: tuple[str, ...] = ()
        self.sieve_column_order: list[str] = list(SIEVE_DEFAULT_COLUMN_ORDER)
        self._sieve_selected_column = "winrate"
        self.target_params_label_var = tk.StringVar(value="Select a target to view child params")
        self.open_support_label_var = tk.StringVar(value="Open support params appear when Search breadth = open")
        self._active_target_label = ""
        self.training_listbox: tk.Listbox | None = None
        self.validation_listbox: tk.Listbox | None = None
        self.editable_entries: list[ttk.Entry] = []
        self.epochs_entry: ttk.Entry | None = None
        self.auto_epochs_cap_entry: ttk.Entry | None = None
        self._build_ui()
        self._bind_events()
        self.refresh()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=0, sticky="nsew")

        main = ttk.Frame(notebook, style="App.TFrame", padding=4)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(2, weight=1)
        notebook.add(main, text="Run")

        controls = ttk.LabelFrame(main, text="Explorer")
        controls.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        for col in (1, 3, 5):
            controls.grid_columnconfigure(col, weight=1)

        ttk.Label(controls, text="Target type").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        ttk.Combobox(controls, textvariable=self.target_type_var, values=["family", "mode"], state="readonly").grid(row=0, column=1, sticky="ew", padx=8, pady=4)
        ttk.Label(controls, text="Target selection").grid(row=0, column=2, sticky="w", padx=8, pady=4)
        ttk.Combobox(controls, textvariable=self.target_selection_var, values=["random", "specific"], state="readonly").grid(row=0, column=3, sticky="ew", padx=8, pady=4)
        ttk.Label(controls, text="Search breadth").grid(row=0, column=4, sticky="w", padx=8, pady=4)
        ttk.Combobox(controls, textvariable=self.search_breadth_var, values=["targeted", "open"], state="readonly").grid(row=0, column=5, sticky="ew", padx=8, pady=4)

        ttk.Label(controls, text="Specific target").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        self.specific_target_combo = ttk.Combobox(controls, textvariable=self.target_name_var, state="disabled")
        self.specific_target_combo.grid(row=1, column=1, sticky="ew", padx=8, pady=4)
        self._editable_entry(controls, 1, 2, "Max loops", self.max_loops_var)
        self.epochs_entry = self._editable_entry(controls, 1, 4, "Epochs", self.epochs_var)
        self._editable_entry(controls, 2, 0, "Random seed", self.random_state_var)
        self._editable_entry(controls, 2, 2, "Sampling seed", self.sampling_seed_var)
        ttk.Checkbutton(controls, text="Auto epochs (20x params)", variable=self.auto_epochs_var).grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=4)
        self.auto_epochs_cap_entry = self._editable_entry(controls, 3, 2, "Auto epoch cap", self.auto_epochs_cap_var)
        ttk.Checkbutton(controls, text="Split-venv pipeline", variable=self.split_venv_pipeline_var).grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=4)
        self._editable_entry(controls, 4, 2, "Backtest Python", self.backtest_python_exe_var)
        self._editable_entry(controls, 4, 4, "Handoff dir", self.pipeline_handoff_dir_var)
        ttk.Label(controls, text="Backtest workers").grid(row=5, column=0, sticky="w", padx=8, pady=4)
        ttk.Combobox(controls, textvariable=self.backtest_worker_count_var, values=[str(index) for index in range(1, 21)], state="readonly").grid(row=5, column=1, sticky="ew", padx=8, pady=4)

        windows = ttk.Frame(main, style="App.TFrame")
        windows.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        windows.grid_columnconfigure(0, weight=1)
        windows.grid_columnconfigure(1, weight=1)
        windows.grid_rowconfigure(0, weight=1)
        self.training_listbox = self._window_selector(windows, "Training windows", 0)
        self.validation_listbox = self._window_selector(windows, "Validation windows", 1)

        catalog = ttk.LabelFrame(main, text="Catalog")
        catalog.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        catalog.grid_columnconfigure(0, weight=1)
        catalog.grid_rowconfigure(0, weight=1)
        columns = ("type", "name", "param_count", "buy_params", "sell_params", "targeted_runs", "open_runs", "accepted", "last_run")
        self.catalog_tree = ttk.Treeview(catalog, columns=columns, show="headings", height=9)
        labels = {
            "type": "Type",
            "name": "Name",
            "param_count": "Param count",
            "buy_params": "Buy params",
            "sell_params": "Sell params",
            "targeted_runs": "Targeted runs",
            "open_runs": "Open runs",
            "accepted": "Accepted",
            "last_run": "Last run",
        }
        for column in columns:
            self.catalog_tree.heading(column, text=labels[column])
            self.catalog_tree.column(column, width=105 if column != "name" else 220, stretch=column in {"name", "last_run"})
        self.catalog_tree.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        buttons = ttk.Frame(main, style="App.TFrame")
        buttons.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(buttons, text="Run Explorer", command=self._run_explorer).pack(side="left")
        ttk.Button(buttons, text="Refresh catalog", command=self.refresh).pack(side="left", padx=(8, 0))

        coverage_tab = ttk.Frame(notebook, style="App.TFrame", padding=4)
        coverage_tab.grid_columnconfigure(0, weight=2)
        coverage_tab.grid_columnconfigure(1, weight=1)
        coverage_tab.grid_rowconfigure(0, weight=1)
        notebook.add(coverage_tab, text="Target Coverage")
        coverage = ttk.LabelFrame(coverage_tab, text="Target coverage")
        coverage.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        coverage.grid_columnconfigure(0, weight=1)
        coverage.grid_rowconfigure(0, weight=1)
        coverage_columns = ("label", "targeted_runs", "open_runs", "accepted", "last_score_delta", "last_run")
        self.coverage_tree = ttk.Treeview(coverage, columns=coverage_columns, show="headings")
        coverage_labels = {
            "label": "Target",
            "targeted_runs": "Targeted",
            "open_runs": "Open",
            "accepted": "Accepted",
            "last_score_delta": "Last score delta",
            "last_run": "Last run",
        }
        for column in coverage_columns:
            self.coverage_tree.heading(column, text=coverage_labels[column])
            self.coverage_tree.column(column, width=120 if column != "label" else 260, stretch=column in {"label", "last_run"})
        self.coverage_tree.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=(8, 0))
        coverage_scroll_y = ttk.Scrollbar(coverage, orient="vertical", command=self.coverage_tree.yview)
        coverage_scroll_y.grid(row=0, column=1, sticky="ns", pady=(8, 0), padx=(4, 8))
        coverage_scroll_x = ttk.Scrollbar(coverage, orient="horizontal", command=self.coverage_tree.xview)
        coverage_scroll_x.grid(row=1, column=0, sticky="ew", padx=(8, 0), pady=(4, 8))
        self.coverage_tree.configure(yscrollcommand=coverage_scroll_y.set, xscrollcommand=coverage_scroll_x.set)

        params = ttk.LabelFrame(coverage_tab, text="Child params")
        params.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=8)
        params.grid_columnconfigure(0, weight=1)
        params.grid_rowconfigure(1, weight=1)
        params.grid_rowconfigure(3, weight=1)

        ttk.Label(params, textvariable=self.target_params_label_var).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 0))
        target_frame = ttk.Frame(params, style="App.TFrame")
        target_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(4, 8))
        target_frame.grid_columnconfigure(0, weight=1)
        target_frame.grid_rowconfigure(0, weight=1)
        self.target_params_tree = ttk.Treeview(target_frame, columns=("name", "space"), show="headings", height=8)
        self.target_params_tree.heading("name", text="Param")
        self.target_params_tree.heading("space", text="Space")
        self.target_params_tree.column("name", width=260, stretch=True)
        self.target_params_tree.column("space", width=90, stretch=False)
        self.target_params_tree.grid(row=0, column=0, sticky="nsew")
        target_scroll_y = ttk.Scrollbar(target_frame, orient="vertical", command=self.target_params_tree.yview)
        target_scroll_y.grid(row=0, column=1, sticky="ns", padx=(4, 0))
        target_scroll_x = ttk.Scrollbar(target_frame, orient="horizontal", command=self.target_params_tree.xview)
        target_scroll_x.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.target_params_tree.configure(yscrollcommand=target_scroll_y.set, xscrollcommand=target_scroll_x.set)

        ttk.Label(params, textvariable=self.open_support_label_var).grid(row=2, column=0, sticky="w", padx=8, pady=(0, 0))
        open_frame = ttk.Frame(params, style="App.TFrame")
        open_frame.grid(row=3, column=0, sticky="nsew", padx=8, pady=(4, 8))
        open_frame.grid_columnconfigure(0, weight=1)
        open_frame.grid_rowconfigure(0, weight=1)
        self.open_support_params_tree = ttk.Treeview(open_frame, columns=("name", "space", "source_family"), show="headings", height=8)
        self.open_support_params_tree.heading("name", text="Param")
        self.open_support_params_tree.heading("space", text="Space")
        self.open_support_params_tree.heading("source_family", text="Source family")
        self.open_support_params_tree.column("name", width=230, stretch=True)
        self.open_support_params_tree.column("space", width=90, stretch=False)
        self.open_support_params_tree.column("source_family", width=200, stretch=True)
        self.open_support_params_tree.grid(row=0, column=0, sticky="nsew")
        open_scroll_y = ttk.Scrollbar(open_frame, orient="vertical", command=self.open_support_params_tree.yview)
        open_scroll_y.grid(row=0, column=1, sticky="ns", padx=(4, 0))
        open_scroll_x = ttk.Scrollbar(open_frame, orient="horizontal", command=self.open_support_params_tree.xview)
        open_scroll_x.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.open_support_params_tree.configure(yscrollcommand=open_scroll_y.set, xscrollcommand=open_scroll_x.set)

        self._build_entry_sieve_tab(notebook)

    def _build_entry_sieve_tab(self, notebook: ttk.Notebook) -> None:
        sieve_tab = ttk.Frame(notebook, style="App.TFrame", padding=4)
        sieve_tab.grid_columnconfigure(0, weight=1)
        sieve_tab.grid_rowconfigure(2, weight=1)
        notebook.add(sieve_tab, text="Entry Sieve")

        controls = ttk.LabelFrame(sieve_tab, text="Entry Sieve")
        controls.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        for col in (1, 3, 5):
            controls.grid_columnconfigure(col, weight=1)
        ttk.Label(controls, text="Strategy batch").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        self.sieve_strategy_batch_combo = ttk.Combobox(controls, textvariable=self.sieve_strategy_batch_var, state="readonly")
        self.sieve_strategy_batch_combo.grid(row=0, column=1, sticky="ew", padx=8, pady=4)
        self._editable_entry(controls, 0, 2, "Strategy filter", self.sieve_strategy_filter_var)
        ttk.Checkbutton(controls, text="Speed run", variable=self.sieve_speed_run_var).grid(row=0, column=4, sticky="w", padx=8, pady=4)
        self._editable_entry(controls, 0, 5, "Speed pairs", self.sieve_speed_pair_count_var)
        self._editable_entry(controls, 1, 0, "Take profit %", self.sieve_take_profit_var)
        self._editable_entry(controls, 1, 2, "Stoploss %", self.sieve_stoploss_var)
        ttk.Label(controls, text="Result batches").grid(row=1, column=4, sticky="w", padx=8, pady=4)
        self.sieve_result_batch_combo = ttk.Combobox(controls, textvariable=self.sieve_result_batch_var, state="normal")
        self.sieve_result_batch_combo.grid(row=1, column=5, sticky="ew", padx=8, pady=4)
        self._editable_entry(controls, 2, 0, "Batch queue", self.sieve_batch_queue_var)
        ttk.Label(controls, text="Queue priority").grid(row=2, column=2, sticky="w", padx=8, pady=4)
        ttk.Combobox(controls, textvariable=self.sieve_batch_priority_var, values=("least_run_first", "configured"), state="readonly").grid(row=2, column=3, sticky="ew", padx=8, pady=4)
        self._editable_entry(controls, 2, 4, "Result file filter", self.sieve_result_batch_filter_var)
        ttk.Checkbutton(controls, text="Auto windows", variable=self.sieve_auto_windows_var).grid(row=3, column=0, sticky="w", padx=8, pady=4)
        ttk.Label(controls, text="Windows/file").grid(row=3, column=1, sticky="e", padx=8, pady=4)
        ttk.Combobox(controls, textvariable=self.sieve_auto_window_count_var, values=("1", "2", "3"), state="readonly", width=6).grid(row=3, column=2, sticky="w", padx=8, pady=4)
        ttk.Checkbutton(controls, text="Target sweep", variable=self.sieve_target_sweep_var).grid(row=3, column=3, sticky="w", padx=8, pady=4)
        self._editable_entry(controls, 3, 4, "TP/SL grid", self.sieve_target_pairs_var)
        ttk.Button(controls, text="Run Entry Sieve", command=self._run_entry_sieve).grid(row=4, column=0, sticky="w", padx=8, pady=4)
        ttk.Button(controls, text="Run batch queue", command=self._run_entry_sieve_batch_queue).grid(row=4, column=1, sticky="w", padx=8, pady=4)
        ttk.Button(controls, text="Refresh results", command=self._refresh_sieve_results).grid(row=4, column=2, sticky="w", padx=8, pady=4)
        ttk.Button(controls, text="Move column left", command=lambda: self._move_sieve_column(-1)).grid(row=4, column=3, sticky="w", padx=8, pady=4)
        ttk.Button(controls, text="Move column right", command=lambda: self._move_sieve_column(1)).grid(row=4, column=4, sticky="w", padx=8, pady=4)
        ttk.Button(controls, text="Reset columns", command=self._reset_sieve_columns).grid(row=4, column=5, sticky="w", padx=8, pady=4)
        ttk.Button(controls, text="Open results folder", command=self._open_sieve_results_folder).grid(row=5, column=0, sticky="w", padx=8, pady=4)
        ttk.Button(controls, text="Delete selected result", command=self._delete_selected_sieve_result_batches).grid(row=5, column=1, sticky="w", padx=8, pady=4)
        ttk.Label(controls, textvariable=self.sieve_status_var).grid(row=6, column=0, columnspan=6, sticky="w", padx=8, pady=4)

        filters = ttk.LabelFrame(sieve_tab, text="Result column filters")
        filters.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        for col in (1, 3, 5, 7):
            filters.grid_columnconfigure(col, weight=1)
        self._editable_entry(filters, 0, 0, "Winrate >= %", self.sieve_filter_winrate_min_var)
        self._editable_entry(filters, 0, 2, "Profit >= %", self.sieve_filter_profit_min_var)
        self._editable_entry(filters, 0, 4, "Max DD <= %", self.sieve_filter_drawdown_max_var)
        self._editable_entry(filters, 0, 6, "Trades >=", self.sieve_filter_trades_min_var)
        self._editable_entry(filters, 1, 0, "TP % =", self.sieve_filter_tp_eq_var)
        self._editable_entry(filters, 1, 2, "SL % =", self.sieve_filter_sl_eq_var)
        self._editable_entry(filters, 1, 4, "Row contains", self.sieve_filter_var)
        ttk.Button(filters, text="Clear filters", command=self._clear_sieve_column_filters).grid(row=1, column=6, sticky="w", padx=8, pady=4)

        results = ttk.LabelFrame(sieve_tab, text="Runtime results")
        results.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        results.grid_columnconfigure(0, weight=1)
        results.grid_rowconfigure(0, weight=1)
        columns = SIEVE_RESULT_COLUMNS
        self.sieve_results_tree = ttk.Treeview(results, columns=columns, show="headings", height=14)
        headings = {
            "score": "Score",
            "strategy_batch": "Batch",
            "speed_run_mode": "Speed",
            "strategy": "Strategy",
            "side": "Side",
            "core_behavior": "Core behaviour",
            "training_window": "Training window",
            "validation_window": "Validation window",
            "take_profit_pct": "TP %",
            "stoploss_pct": "SL %",
            "status": "Status",
            "analysis_read": "Read",
            "analysis_next": "Next",
            "hyperopt_loss": "Hyperopt loss",
            "objective": "Objective",
            "best_params_count": "Params",
            "epoch_count": "Epochs",
            "profit_total_abs": "Profit abs",
            "profit_total": "Profit %",
            "trade_count": "Trades",
            "winrate": "Winrate",
            "max_drawdown_pct": "Max DD",
            "backtest_file": "Backtest file",
            "params_file": "Params file",
        }
        self.sieve_result_columns = columns
        self.sieve_column_order = self._normalized_sieve_column_order(self.sieve_column_order, columns)
        for column in columns:
            self.sieve_results_tree.heading(column, text=headings[column], command=lambda col=column: self._sieve_heading_clicked(col))
            width = 95
            minwidth = 80
            stretch = False
            if column == "score":
                width = 90
                minwidth = 70
            elif column == "strategy":
                width = 360
                minwidth = 320
            elif column == "strategy_batch":
                width = 150
                minwidth = 110
            elif column == "speed_run_mode":
                width = 70
                minwidth = 60
            elif column == "side":
                width = 70
                minwidth = 60
            elif column == "core_behavior":
                width = 120
                minwidth = 100
            elif column in {"take_profit_pct", "stoploss_pct"}:
                width = 70
                minwidth = 60
            elif column in {"analysis_read", "analysis_next"}:
                width = 110
                minwidth = 90
            elif column in {"backtest_file", "params_file"}:
                width = 360
                minwidth = 220
                stretch = True
            elif column in {"training_window", "validation_window"}:
                width = 160
                minwidth = 130
            self.sieve_results_tree.column(column, width=width, minwidth=minwidth, stretch=stretch)
        self.sieve_results_tree.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=(8, 0))
        scroll_y = ttk.Scrollbar(results, orient="vertical", command=self.sieve_results_tree.yview)
        scroll_y.grid(row=0, column=1, sticky="ns", pady=(8, 0), padx=(4, 8))
        scroll_x = ttk.Scrollbar(results, orient="horizontal", command=self.sieve_results_tree.xview)
        scroll_x.grid(row=1, column=0, sticky="ew", padx=(8, 0), pady=(4, 8))
        self.sieve_results_tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        self._apply_sieve_column_order()

    def _editable_entry(self, parent: tk.Misc, row: int, column: int, label: str, variable: tk.StringVar) -> ttk.Entry:
        _, entry = labeled_entry(parent, row, column, label, variable)
        entry.configure(takefocus=True)
        entry.bind("<Button-1>", lambda event: event.widget.focus_set(), add="+")
        self.editable_entries.append(entry)
        return entry

    def _window_selector(self, parent: tk.Misc, title: str, column: int) -> tk.Listbox:
        frame = ttk.LabelFrame(parent, text=title)
        frame.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 6, 0 if column == 1 else 6), pady=0)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=1)
        listbox = tk.Listbox(frame, selectmode=tk.EXTENDED, height=9, exportselection=False)
        listbox.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        return listbox

    def _bind_events(self) -> None:
        self.target_type_var.trace_add("write", lambda *_: self._refresh_target_options())
        self.target_selection_var.trace_add("write", lambda *_: self._update_specific_target_state())
        self.search_breadth_var.trace_add("write", lambda *_: self._show_target_params(self._active_target_label))
        self.auto_epochs_var.trace_add("write", lambda *_: self._update_epochs_mode_state())
        self.sieve_speed_run_var.trace_add("write", lambda *_: self._update_epochs_mode_state())
        self.sieve_result_batch_filter_var.trace_add("write", lambda *_: self._refresh_sieve_results())
        self.sieve_filter_var.trace_add("write", lambda *_: self._refresh_sieve_results())
        for variable in (
            self.sieve_filter_var,
            self.sieve_filter_winrate_min_var,
            self.sieve_filter_profit_min_var,
            self.sieve_filter_drawdown_max_var,
            self.sieve_filter_trades_min_var,
            self.sieve_filter_tp_eq_var,
            self.sieve_filter_sl_eq_var,
        ):
            variable.trace_add("write", lambda *_: self._refresh_sieve_results())
        if self.sieve_result_batch_combo is not None:
            self.sieve_result_batch_combo.bind("<<ComboboxSelected>>", lambda event: self._refresh_sieve_results(), add="+")
        if self.catalog_tree is not None:
            self.catalog_tree.bind("<<TreeviewSelect>>", self._catalog_selected, add="+")
        if self.coverage_tree is not None:
            self.coverage_tree.bind("<<TreeviewSelect>>", self._coverage_selected, add="+")
        self._update_epochs_mode_state()
        self._update_sieve_speed_run_state()

    def _update_epochs_mode_state(self) -> None:
        auto = bool(self.auto_epochs_var.get())
        for entry in self.editable_entries:
            entry.configure(state="normal")
        if self.epochs_entry is not None:
            self.epochs_entry.configure(state="disabled" if auto else "normal")
        if self.auto_epochs_cap_entry is not None:
            self.auto_epochs_cap_entry.configure(state="normal" if auto else "disabled")
        self._update_sieve_speed_run_state()

    def _update_sieve_speed_run_state(self) -> None:
        if not hasattr(self, "sieve_speed_run_var") or not bool(self.sieve_speed_run_var.get()):
            return
        if self.epochs_entry is not None:
            self.epochs_entry.configure(state="disabled")
        if self.auto_epochs_cap_entry is not None:
            self.auto_epochs_cap_entry.configure(state="disabled")

    def _strategy_context(self) -> tuple[str, str]:
        common = self.context.registry.get("common")
        if common is not None:
            state = common.get_state()
            strategy_file = str(state.get("strategy_file") or "")
            strategy_class = str(state.get("strategy_class") or "")
            if strategy_file and strategy_class:
                return strategy_file, strategy_class
        preset = self._load_preset()
        return str(preset.get("strategy_file") or ""), str(preset.get("strategy_class") or "")

    def _load_preset(self) -> dict[str, Any]:
        path = self.service.resolve_path("launcher_v2/config/presets.json")
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return {}
        preset = data.get(self.preset_name_var.get()) if isinstance(data, dict) else {}
        return preset if isinstance(preset, dict) else {}

    def refresh(self) -> None:
        self._load_windows()
        self._load_sieve_batches()
        self._load_catalog()
        self._update_specific_target_state()

    def _load_sieve_batches(self) -> None:
        if self.sieve_strategy_batch_combo is None:
            return
        try:
            batches = self.sieve_service.load_strategy_batches()
            values = [str(batch.get("id") or "") for batch in batches if str(batch.get("id") or "").strip()]
        except Exception as exc:
            values = ["all"]
            self.context.shared.status.set(f"Entry Sieve batch load failed: {exc}")
        self.sieve_strategy_batch_combo.configure(values=values)
        if self.sieve_strategy_batch_var.get() not in values:
            self.sieve_strategy_batch_var.set(values[0] if values else "all")

    def _load_windows(self) -> None:
        self._window_rows = self.service.load_windows("explorer/config/market_windows.json")
        names = [row.get("name", "") for row in self._window_rows if row.get("name")]
        for listbox in (self.training_listbox, self.validation_listbox):
            if listbox is None:
                continue
            current = self._selected_values(listbox)
            listbox.delete(0, tk.END)
            for row in self._window_rows:
                name = row.get("name") or row.get("timerange") or ""
                regime = row.get("regime") or ""
                timerange = row.get("timerange") or ""
                listbox.insert(tk.END, f"{name}    {regime}    {timerange}")
            self._select_values(listbox, current)
        if self.validation_listbox is not None and not self.validation_listbox.curselection() and "full_cycle_2020_2026" in names:
            self._select_values(self.validation_listbox, ["full_cycle_2020_2026"])

    def _load_catalog(self) -> None:
        strategy_file, strategy_class = self._strategy_context()
        try:
            rows = self.service.load_target_catalog(strategy_file, strategy_class, "../explorer_reports/hyperopt_explorer_state.json")
            if not rows:
                preset = self._load_preset()
                preset_strategy_file = str(preset.get("strategy_file") or "")
                preset_strategy_class = str(preset.get("strategy_class") or "")
                if (preset_strategy_file, preset_strategy_class) != (strategy_file, strategy_class):
                    rows = self.service.load_target_catalog(preset_strategy_file, preset_strategy_class, "../explorer_reports/hyperopt_explorer_state.json")
        except Exception as exc:
            rows = []
            self.context.shared.status.set(f"Explorer catalog load failed: {exc}")
        self._catalog_rows = rows
        self._refresh_target_options()
        self._render_catalog()

    def _refresh_target_options(self) -> None:
        target_type = self.target_type_var.get()
        values = [row["name"] for row in self._catalog_rows if row.get("type") == target_type and int(row.get("param_count") or 0) > 0]
        if self.specific_target_combo is not None:
            self.specific_target_combo.configure(values=values)
        if self.target_name_var.get() and self.target_name_var.get() not in values:
            self.target_name_var.set("")
        self._update_specific_target_state()
        self._render_catalog()

    def _update_specific_target_state(self) -> None:
        if self.specific_target_combo is None:
            return
        if self.target_selection_var.get() == "specific":
            self.specific_target_combo.configure(state="readonly")
        else:
            self.specific_target_combo.configure(state="disabled")

    def _render_catalog(self) -> None:
        if self.catalog_tree is None or self.coverage_tree is None:
            return
        target_type = self.target_type_var.get()
        rows = [
            (
                row.get("type", ""),
                row.get("name", ""),
                row.get("param_count", ""),
                row.get("buy_params", ""),
                row.get("sell_params", ""),
                row.get("targeted_runs", ""),
                row.get("open_runs", ""),
                row.get("accepted", ""),
                row.get("last_run", ""),
            )
            for row in self._catalog_rows
            if row.get("type") == target_type and int(row.get("param_count") or 0) > 0
        ]
        set_tree_rows(self.catalog_tree, rows)
        coverage_rows = [
            (
                row.get("label", ""),
                row.get("targeted_runs", ""),
                row.get("open_runs", ""),
                row.get("accepted", ""),
                row.get("last_score_delta", ""),
                row.get("last_run", ""),
            )
            for row in self._catalog_rows
            if int(row.get("param_count") or 0) > 0
        ]
        set_tree_rows(self.coverage_tree, coverage_rows)
        self._show_target_params(self._active_target_label)

    def _catalog_selected(self, _event: tk.Event | None = None) -> None:
        if self.catalog_tree is None:
            return
        selection = self.catalog_tree.selection()
        if not selection:
            return
        values = self.catalog_tree.item(selection[0], "values")
        if len(values) >= 2:
            self.target_type_var.set(str(values[0]))
            self.target_selection_var.set("specific")
            self.target_name_var.set(str(values[1]))
            self._show_target_params(f"{values[0]}:{values[1]}")

    def _coverage_selected(self, _event: tk.Event | None = None) -> None:
        if self.coverage_tree is None:
            return
        selection = self.coverage_tree.selection()
        if not selection:
            return
        values = self.coverage_tree.item(selection[0], "values")
        if not values:
            return
        label = str(values[0])
        self._show_target_params(label)

    def _show_target_params(self, target_label: str) -> None:
        if self.target_params_tree is None or self.open_support_params_tree is None:
            return
        selected_label = str(target_label or "").strip()
        self._active_target_label = selected_label
        row = next((item for item in self._catalog_rows if str(item.get("label") or "") == selected_label), None)
        details = row.get("param_details") if isinstance(row, dict) else []
        target_rows = []
        for detail in details or []:
            if not isinstance(detail, dict):
                continue
            target_rows.append((str(detail.get("name") or ""), str(detail.get("space") or "")))
        set_tree_rows(self.target_params_tree, target_rows)

        breadth = self.search_breadth_var.get().strip().lower()
        open_rows: list[tuple[str, str, str]] = []
        if breadth == "open" and isinstance(row, dict):
            for detail in (row.get("open_support_param_details") or []):
                if not isinstance(detail, dict):
                    continue
                open_rows.append(
                    (
                        str(detail.get("name") or ""),
                        str(detail.get("space") or ""),
                        str(detail.get("source_family") or ""),
                    )
                )
        set_tree_rows(self.open_support_params_tree, open_rows)

        if selected_label:
            self.target_params_label_var.set(f"{selected_label} target params ({len(target_rows)})")
        else:
            self.target_params_label_var.set("Select a target to view child params")
        if not selected_label:
            self.open_support_label_var.set("Open support params appear when Search breadth = open")
            return
        if breadth != "open":
            self.open_support_label_var.set("Open support params appear when Search breadth = open")
            return
        families = ", ".join((row.get("open_support_families") or [])) if isinstance(row, dict) else ""
        if families:
            self.open_support_label_var.set(f"{selected_label} open support extras ({len(open_rows)}) from {families}")
        else:
            self.open_support_label_var.set(f"{selected_label} open support extras ({len(open_rows)})")

    def _selected_values(self, listbox: tk.Listbox) -> list[str]:
        selected: list[str] = []
        for index in listbox.curselection():
            name = str(listbox.get(index)).split()[0]
            if name:
                selected.append(name)
        return selected

    def _select_values(self, listbox: tk.Listbox, values: list[str]) -> None:
        wanted = set(values)
        listbox.selection_clear(0, tk.END)
        for index in range(listbox.size()):
            name = str(listbox.get(index)).split()[0]
            if name in wanted:
                listbox.selection_set(index)

    def _settings(self) -> ExplorerRunSettings:
        state = {
            "preset_name": self.preset_name_var.get(),
            "target_type": self.target_type_var.get(),
            "target_selection": self.target_selection_var.get(),
            "target_name": self.target_name_var.get(),
            "search_breadth": self.search_breadth_var.get(),
            "training_windows": self._selected_values(self.training_listbox) if self.training_listbox is not None else [],
            "validation_windows": self._selected_values(self.validation_listbox) if self.validation_listbox is not None else [],
            "max_loops": self.max_loops_var.get(),
            "epochs": self.epochs_var.get(),
            "auto_epochs": self.auto_epochs_var.get(),
            "auto_epochs_cap": self.auto_epochs_cap_var.get(),
            "random_state": self.random_state_var.get(),
            "sampling_seed": self.sampling_seed_var.get(),
            "split_venv_pipeline": self.split_venv_pipeline_var.get(),
            "backtest_python_exe": self.backtest_python_exe_var.get(),
            "backtest_worker_count": self.backtest_worker_count_var.get(),
            "pipeline_handoff_dir": self.pipeline_handoff_dir_var.get(),
        }
        return ExplorerRunSettings.from_state(state, self.context.app_dir)

    def _sieve_settings(self) -> EntrySieveSettings:
        return EntrySieveSettings(
            preset_name="LauncherV2-auto",
            auto_window_mode=self.sieve_auto_windows_var.get(),
            auto_window_count=self.sieve_auto_window_count_var.get(),
            training_windows=self._selected_values(self.training_listbox) if self.training_listbox is not None else [],
            validation_windows=self._selected_values(self.validation_listbox) if self.validation_listbox is not None else [],
            epochs=self.epochs_var.get(),
            auto_epochs=self.auto_epochs_var.get(),
            auto_epochs_cap=self.auto_epochs_cap_var.get(),
            random_state=self.random_state_var.get(),
            sampling_seed=self.sampling_seed_var.get(),
            split_venv_pipeline=self.split_venv_pipeline_var.get(),
            backtest_python_exe=self.backtest_python_exe_var.get(),
            backtest_worker_count=self.backtest_worker_count_var.get(),
            pipeline_handoff_dir=self.pipeline_handoff_dir_var.get(),
            strategy_batch=self.sieve_strategy_batch_var.get(),
            batch_queue_priority=self.sieve_batch_priority_var.get(),
            strategy_filter=self.sieve_strategy_filter_var.get(),
            speed_run_mode=self.sieve_speed_run_var.get(),
            speed_pair_count=self.sieve_speed_pair_count_var.get(),
            take_profit_pct=self.sieve_take_profit_var.get(),
            stoploss_pct=self.sieve_stoploss_var.get(),
            target_sweep_enabled=self.sieve_target_sweep_var.get(),
            target_sweep_pairs=self.sieve_target_pairs_var.get(),
        )

    def _run_explorer(self) -> None:
        self.refresh()
        self.preset_name_var.set("LauncherV2-auto")
        self.context.emit("save_state", {"reason": "explorer_run"})
        try:
            command = self.service.build_command(self._settings())
        except Exception as exc:
            messagebox.showerror("Explorer configuration", str(exc), parent=self)
            return
        self.context.process_runner.run(command, cwd=str(self.context.app_dir), owner=self.tab_key)
        self.context.shared.status.set("Explorer running")

    def _run_entry_sieve(self) -> None:
        self.refresh()
        self.preset_name_var.set("LauncherV2-auto")
        self.context.emit("save_state", {"reason": "entry_sieve_run"})
        busy_message = self.sieve_service.busy_message()
        if busy_message:
            messagebox.showerror("Entry Sieve already running", busy_message, parent=self)
            return
        try:
            command = self.sieve_service.build_command(self._sieve_settings())
        except Exception as exc:
            messagebox.showerror("Entry Sieve configuration", str(exc), parent=self)
            return
        self.context.process_runner.run(command, cwd=str(self.context.app_dir), owner=self.tab_key)
        self.context.shared.status.set("Entry Sieve running")
        self.sieve_status_var.set("Run status: starting")

    def _run_entry_sieve_batch_queue(self) -> None:
        self.refresh()
        self.preset_name_var.set("LauncherV2-auto")
        self.context.emit("save_state", {"reason": "entry_sieve_batch_queue_run"})
        batch_ids = _split_list(self.sieve_batch_queue_var.get())
        busy_message = self.sieve_service.busy_message()
        if busy_message:
            messagebox.showerror("Entry Sieve already running", busy_message, parent=self)
            return
        try:
            command = self.sieve_service.build_batch_queue_command(self._sieve_settings(), batch_ids)
        except Exception as exc:
            messagebox.showerror("Entry Sieve batch queue", str(exc), parent=self)
            return
        self.context.process_runner.run(command, cwd=str(self.context.app_dir), owner=self.tab_key)
        self.context.shared.status.set("Entry Sieve batch queue running")
        self.sieve_status_var.set("Run status: batch queue starting")

    def _open_sieve_results_folder(self) -> None:
        try:
            self.sieve_service.results_dir.mkdir(parents=True, exist_ok=True)
            open_path(self.sieve_service.results_dir)
        except Exception as exc:
            messagebox.showerror("Entry Sieve results", f"Could not open results folder:\n{exc}", parent=self)

    def _delete_selected_sieve_result_batches(self) -> None:
        selected_batches = self._selected_sieve_result_batches(
            self.sieve_result_batch_var.get().strip(),
            self._sieve_result_batch_ids,
        )
        if not selected_batches:
            messagebox.showinfo("Delete Entry Sieve result", "Select one or more result batches to delete.", parent=self)
            return
        label = ", ".join(selected_batches[:5])
        if len(selected_batches) > 5:
            label += f", and {len(selected_batches) - 5} more"
        if not messagebox.askyesno("Delete Entry Sieve result", f"Delete {len(selected_batches)} selected result batch(es)?\n\n{label}", parent=self):
            return
        try:
            deleted = self.sieve_service.delete_result_batches(selected_batches)
        except Exception as exc:
            messagebox.showerror("Delete Entry Sieve result", str(exc), parent=self)
            return
        self.sieve_result_batch_var.set("")
        self._refresh_sieve_results()
        self.context.shared.status.set(f"Deleted {len(deleted)} Entry Sieve result file(s).")

    def _clear_sieve_column_filters(self) -> None:
        for variable in (
            self.sieve_filter_winrate_min_var,
            self.sieve_filter_profit_min_var,
            self.sieve_filter_drawdown_max_var,
            self.sieve_filter_trades_min_var,
            self.sieve_filter_tp_eq_var,
            self.sieve_filter_sl_eq_var,
        ):
            variable.set("")

    def _refresh_sieve_results(self) -> None:
        if self.sieve_results_tree is None:
            return
        try:
            status = self.sieve_service.load_run_status()
            self.sieve_status_var.set(self._format_sieve_status(status))
            batches = self.sieve_service.load_result_batches()
            batch_filter = self.sieve_result_batch_filter_var.get().strip()
            if batch_filter:
                batches = [batch for batch in batches if self._sieve_result_batch_matches_filter(batch, batch_filter)]
            batch_ids = [str(batch.get("id") or "") for batch in batches]
            self._sieve_result_batch_ids = batch_ids
            if self.sieve_result_batch_combo is not None:
                self.sieve_result_batch_combo.configure(values=["all", *batch_ids])
            selected_batch = self.sieve_result_batch_var.get().strip()
            selected_batches = self._selected_sieve_result_batches(selected_batch, batch_ids)
            if not batch_ids:
                if selected_batch:
                    self.sieve_result_batch_var.set("")
                selected_batches = []
                rows = []
            elif not selected_batches:
                latest = next((str(batch.get("id") or "") for batch in batches if batch.get("latest")), "")
                selected_batch = latest or batch_ids[0]
                self.sieve_result_batch_var.set(selected_batch)
                selected_batches = [selected_batch]
                rows = self.sieve_service.load_results(selected_batch)
            else:
                rows = self.sieve_service.load_results_many(selected_batches) if len(selected_batches) > 1 else self.sieve_service.load_results(selected_batches[0])
        except Exception as exc:
            self.context.shared.status.set(f"Entry Sieve results load failed: {exc}")
            rows = []
        filter_text = self.sieve_filter_var.get().strip().lower()
        if filter_text:
            rows = [row for row in rows if filter_text in json.dumps(row, sort_keys=True, default=str).lower()]
        rows = self._apply_sieve_column_filters(rows)
        rows = sorted(rows, key=lambda row: self._sieve_sort_key(row, self._sieve_sort_column), reverse=self._sieve_sort_reverse)
        rendered = [
            (
                self._fmt_result(self._sieve_score(row)),
                row.get("strategy_batch_label", "") or row.get("strategy_batch", ""),
                "yes" if row.get("speed_run_mode") else "",
                row.get("strategy", ""),
                row.get("side", "") or self._strategy_side(row.get("strategy", "")),
                row.get("core_behavior", "") or self._strategy_core_behavior(row.get("strategy", "")),
                row.get("training_window", ""),
                row.get("validation_window", ""),
                row.get("take_profit_pct", ""),
                row.get("stoploss_pct", ""),
                row.get("status", ""),
                *self._sieve_analysis(row),
                self._fmt_result(row.get("hyperopt_loss")),
                self._fmt_result(row.get("objective")),
                row.get("best_params_count", ""),
                row.get("epoch_count", ""),
                self._fmt_result(row.get("profit_total_abs")),
                self._fmt_percent(row.get("profit_total")),
                row.get("trade_count", ""),
                self._fmt_percent(row.get("winrate")),
                self._fmt_percent(row.get("max_drawdown_pct")),
                row.get("backtest_file", ""),
                row.get("params_file", ""),
            )
            for row in rows
        ]
        set_tree_rows(self.sieve_results_tree, rendered)

    def _selected_sieve_result_batches(self, selection: str, batch_ids: list[str]) -> list[str]:
        selected = _split_list(selection)
        if not selected:
            return []
        known = set(batch_ids)
        if any(item.lower() == "all" for item in selected):
            return batch_ids
        return [item for item in selected if item in known]

    @staticmethod
    def _sieve_result_batch_matches_filter(batch: dict[str, Any], filter_text: str) -> bool:
        tokens = [token.lower() for token in _split_list(filter_text)]
        if not tokens:
            return True
        haystack = " ".join(
            str(batch.get(key) or "")
            for key in ("id", "label", "path", "strategy_batch", "strategy_batch_label", "status", "phase")
        ).lower()
        return all(token in haystack for token in tokens)

    def _apply_sieve_column_filters(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        active_filters: list[tuple[str, str, float, float]] = []
        filter_specs = (
            ("winrate", self.sieve_filter_winrate_min_var.get(), ">=", 100.0),
            ("profit_total", self.sieve_filter_profit_min_var.get(), ">=", 100.0),
            ("max_drawdown_pct", self.sieve_filter_drawdown_max_var.get(), "<=", 100.0),
            ("trade_count", self.sieve_filter_trades_min_var.get(), ">=", 1.0),
            ("take_profit_pct", self.sieve_filter_tp_eq_var.get(), "=", 1.0),
            ("stoploss_pct", self.sieve_filter_sl_eq_var.get(), "=", 1.0),
        )
        for column, text, operator, multiplier in filter_specs:
            threshold = self._parse_filter_number(text)
            if threshold is None:
                continue
            active_filters.append((column, operator, threshold, multiplier))
        if not active_filters:
            return rows

        filtered: list[dict[str, Any]] = []
        for row in rows:
            keep = True
            for column, operator, threshold, multiplier in active_filters:
                raw_value = self._parse_filter_number(row.get(column))
                if raw_value is None:
                    keep = False
                    break
                value = raw_value * multiplier
                if operator == ">=" and value < threshold:
                    keep = False
                    break
                if operator == "<=" and value > threshold:
                    keep = False
                    break
                if operator == "=" and abs(value - threshold) > 1e-9:
                    keep = False
                    break
            if keep:
                filtered.append(row)
        return filtered

    @staticmethod
    def _parse_filter_number(value: Any) -> float | None:
        if value in (None, ""):
            return None
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("%"):
            text = text[:-1].strip()
        text = text.replace(",", "")
        try:
            return float(text)
        except (TypeError, ValueError):
            return None

    def _sieve_heading_clicked(self, column: str) -> None:
        self._sieve_selected_column = column
        self._sort_sieve_results(column)

    def _move_sieve_column(self, offset: int) -> None:
        if not self.sieve_result_columns:
            return
        order = self._normalized_sieve_column_order(self.sieve_column_order, self.sieve_result_columns)
        selected = self._sieve_selected_column if self._sieve_selected_column in order else order[0]
        index = order.index(selected)
        target = max(0, min(len(order) - 1, index + offset))
        if target == index:
            return
        order.pop(index)
        order.insert(target, selected)
        self.sieve_column_order = order
        self._apply_sieve_column_order()
        self.context.emit("save_state", {"reason": "entry_sieve_column_order"})

    def _reset_sieve_columns(self) -> None:
        self.sieve_column_order = list(SIEVE_DEFAULT_COLUMN_ORDER)
        self._apply_sieve_column_order()
        self.context.emit("save_state", {"reason": "entry_sieve_column_order_reset"})

    def _apply_sieve_column_order(self) -> None:
        if self.sieve_results_tree is None or not self.sieve_result_columns:
            return
        self.sieve_column_order = self._normalized_sieve_column_order(self.sieve_column_order, self.sieve_result_columns)
        self.sieve_results_tree.configure(displaycolumns=tuple(self.sieve_column_order))

    @staticmethod
    def _normalized_sieve_column_order(order: list[str], columns: tuple[str, ...]) -> list[str]:
        if not order:
            order = list(SIEVE_DEFAULT_COLUMN_ORDER)
        valid = set(columns)
        normalized = [column for column in order if column in valid]
        normalized.extend(column for column in columns if column not in normalized)
        return normalized

    @staticmethod
    def _is_legacy_sieve_column_order(order: list[str]) -> bool:
        if "score" not in order:
            return True
        batch_columns = {"strategy_batch", "speed_run_mode"}
        return order == list(SIEVE_RESULT_COLUMNS) or order == [column for column in SIEVE_DEFAULT_COLUMN_ORDER if column not in batch_columns]

    def _sort_sieve_results(self, column: str) -> None:
        if self._sieve_sort_column == column:
            self._sieve_sort_reverse = not self._sieve_sort_reverse
        else:
            self._sieve_sort_column = column
            self._sieve_sort_reverse = column in {"score", "profit_total_abs", "profit_total", "trade_count", "winrate", "best_params_count", "epoch_count"}
        self._refresh_sieve_results()

    @staticmethod
    def _sieve_sort_key(row: dict[str, Any], column: str) -> tuple[int, Any]:
        if column == "score":
            return (1, ExplorerTab._sieve_score(row))
        if column in {"analysis_read", "analysis_next"}:
            value = ExplorerTab._sieve_analysis(row)[0 if column == "analysis_read" else 1]
            return (1, value.lower())
        value = row.get(column)
        if value in (None, ""):
            return (0, 0)
        if isinstance(value, (int, float)):
            return (1, float(value))
        try:
            return (1, float(str(value)))
        except (TypeError, ValueError):
            return (1, str(value).lower())

    @staticmethod
    def _sieve_analysis(row: dict[str, Any]) -> tuple[str, str]:
        status = str(row.get("status") or "").strip().lower()
        error_text = json.dumps(row, sort_keys=True, default=str).lower()
        trades = ExplorerTab._to_float(row.get("trade_count"))
        profit = ExplorerTab._to_float(row.get("profit_total"))
        profit_factor = ExplorerTab._to_float(row.get("profit_factor"))
        drawdown = ExplorerTab._to_float(row.get("max_drawdown_pct"))

        if status and status != "ok":
            if "dry_rvol" in error_text or "expansion_rvol" in error_text:
                return "volume config", "fix rvol order"
            return "run error", "inspect log"
        if trades <= 0:
            return "no trades", "relax trigger"
        if trades < 20:
            return "too sparse", "relax gates"
        if profit > 0 and profit_factor >= 1.2 and drawdown <= 0.03:
            return "clean edge", "freeze/test"
        if profit > 0:
            return "mild edge", "refine gates"
        if profit < -0.02 and trades >= 50:
            return "overtrades", "tighten gates"
        if profit < 0:
            return "weak entry", "rethink gate"
        return "flat/noisy", "add context"

    @staticmethod
    def _sieve_score(row: dict[str, Any]) -> float:
        if str(row.get("status") or "").strip().lower() != "ok":
            return -999.0
        trades = max(0.0, ExplorerTab._to_float(row.get("trade_count")))
        if trades <= 0:
            return -100.0

        winrate = min(1.0, max(0.0, ExplorerTab._to_float(row.get("winrate"))))
        take_profit = max(0.0, ExplorerTab._to_float(row.get("take_profit_pct")))
        stoploss = max(0.0, ExplorerTab._to_float(row.get("stoploss_pct")))
        profit_pct = ExplorerTab._to_float(row.get("profit_total")) * 100.0

        if take_profit <= 0 or stoploss <= 0:
            expectancy_r = 0.0
        else:
            expectancy_r = (winrate * take_profit) - ((1.0 - winrate) * stoploss)
            expectancy_r /= stoploss
        sample_factor = min(1.0, (trades / 50.0) ** 0.5)
        activity_bonus = min(15.0, math.log1p(trades) * 3.0)

        score = sample_factor * ((35.0 * winrate) + (45.0 * expectancy_r))
        score += max(-60.0, min(80.0, profit_pct * 2.5))
        score += activity_bonus
        return score

    @staticmethod
    def _to_float(value: Any) -> float:
        if value in (None, ""):
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _fmt_result(value: Any) -> str:
        if value in (None, ""):
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return f"{value:.4f}"
        try:
            return f"{float(str(value)):.4f}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _fmt_percent(value: Any) -> str:
        if value in (None, ""):
            return ""
        try:
            return f"{float(value) * 100.0:.2f}%"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _format_sieve_status(status: dict[str, Any]) -> str:
        if not status:
            return "Run status: idle"
        job_id = str(status.get("job_id") or "")
        state = str(status.get("status") or "unknown")
        phase = str(status.get("phase") or "")
        message = str(status.get("message") or "")
        hyper_done = status.get("completed_hyperopts")
        hyper_total = status.get("total_hyperopts")
        back_done = status.get("completed_backtests")
        back_total = status.get("total_backtests")
        parts = [f"Run status: {state}"]
        if phase:
            parts.append(phase)
        if hyper_total not in (None, ""):
            parts.append(f"hyperopt {hyper_done or 0}/{hyper_total}")
        if back_total not in (None, ""):
            parts.append(f"backtest {back_done or 0}/{back_total}")
        if job_id:
            parts.append(job_id)
        if message:
            parts.append(message)
        return " | ".join(parts)

    @staticmethod
    def _strategy_side(name: Any) -> str:
        lowered = str(name or "").lower()
        if "short" in lowered:
            return "short"
        if "long" in lowered:
            return "long"
        return ""

    @staticmethod
    def _strategy_core_behavior(name: Any) -> str:
        lowered = str(name or "").lower()
        if "volume_profile" in lowered or lowered.startswith("volume_"):
            return "volume"
        if "support" in lowered or "_sup" in lowered or "sup_" in lowered:
            return "support"
        if "resistance" in lowered or "_res" in lowered or "res_" in lowered:
            return "resistance"
        if any(token in lowered for token in ("breakout", "breakdown", "reclaim", "pullback", "retest", "reject", "fail")):
            return "pattern"
        return "other"

    def get_state(self) -> dict[str, Any]:
        return {
            "target_type": self.target_type_var.get(),
            "target_selection": self.target_selection_var.get(),
            "target_name": self.target_name_var.get(),
            "search_breadth": self.search_breadth_var.get(),
            "training_windows": self._selected_values(self.training_listbox) if self.training_listbox is not None else [],
            "validation_windows": self._selected_values(self.validation_listbox) if self.validation_listbox is not None else [],
            "max_loops": self.max_loops_var.get(),
            "epochs": self.epochs_var.get(),
            "auto_epochs": self.auto_epochs_var.get(),
            "auto_epochs_cap": self.auto_epochs_cap_var.get(),
            "random_state": self.random_state_var.get(),
            "sampling_seed": self.sampling_seed_var.get(),
            "split_venv_pipeline": self.split_venv_pipeline_var.get(),
            "backtest_python_exe": self.backtest_python_exe_var.get(),
            "backtest_python_exes": self._settings().backtest_python_exes,
            "backtest_worker_count": self.backtest_worker_count_var.get(),
            "pipeline_handoff_dir": self.pipeline_handoff_dir_var.get(),
            "sieve_strategy_batch": self.sieve_strategy_batch_var.get(),
            "sieve_batch_queue": self.sieve_batch_queue_var.get(),
            "sieve_batch_priority": self.sieve_batch_priority_var.get(),
            "sieve_strategy_filter": self.sieve_strategy_filter_var.get(),
            "sieve_speed_run": self.sieve_speed_run_var.get(),
            "sieve_speed_pair_count": self.sieve_speed_pair_count_var.get(),
            "sieve_take_profit_pct": self.sieve_take_profit_var.get(),
            "sieve_stoploss_pct": self.sieve_stoploss_var.get(),
            "sieve_auto_windows": self.sieve_auto_windows_var.get(),
            "sieve_auto_window_count": self.sieve_auto_window_count_var.get(),
            "sieve_target_sweep": self.sieve_target_sweep_var.get(),
            "sieve_target_pairs": self.sieve_target_pairs_var.get(),
            "sieve_result_batch": self.sieve_result_batch_var.get(),
            "sieve_result_batch_filter": self.sieve_result_batch_filter_var.get(),
            "sieve_result_filter": self.sieve_filter_var.get(),
            "sieve_filter_winrate_min": self.sieve_filter_winrate_min_var.get(),
            "sieve_filter_profit_min": self.sieve_filter_profit_min_var.get(),
            "sieve_filter_drawdown_max": self.sieve_filter_drawdown_max_var.get(),
            "sieve_filter_trades_min": self.sieve_filter_trades_min_var.get(),
            "sieve_filter_tp_eq": self.sieve_filter_tp_eq_var.get(),
            "sieve_filter_sl_eq": self.sieve_filter_sl_eq_var.get(),
            "sieve_column_order": list(self.sieve_column_order),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        settings = ExplorerRunSettings.from_state(state, self.context.app_dir)
        self.preset_name_var.set(settings.preset_name)
        self.target_type_var.set(settings.target_type)
        self.target_selection_var.set(settings.target_selection)
        self.target_name_var.set(settings.target_name)
        self.search_breadth_var.set(settings.search_breadth)
        self.max_loops_var.set(settings.max_loops)
        self.epochs_var.set(settings.epochs)
        self.auto_epochs_var.set(bool(settings.auto_epochs))
        self.auto_epochs_cap_var.set(settings.auto_epochs_cap)
        self.random_state_var.set(settings.random_state)
        self.sampling_seed_var.set(settings.sampling_seed)
        self.split_venv_pipeline_var.set(bool(settings.split_venv_pipeline))
        self.backtest_python_exe_var.set(settings.backtest_python_exe)
        self.backtest_worker_count_var.set(settings.backtest_worker_count)
        self.pipeline_handoff_dir_var.set(settings.pipeline_handoff_dir)
        self.sieve_strategy_batch_var.set(str(state.get("sieve_strategy_batch") or "all"))
        self.sieve_batch_queue_var.set(str(state.get("sieve_batch_queue") or "volume_profile,structure_levels,continuation_patterns,reversal_patterns,market_state_pressure,multi_confluence,small_concepts,avwap,zones"))
        self.sieve_batch_priority_var.set(str(state.get("sieve_batch_priority") or "least_run_first"))
        self.sieve_strategy_filter_var.set(str(state.get("sieve_strategy_filter") or "sieve1_*.py"))
        speed_run = state.get("sieve_speed_run", False)
        self.sieve_speed_run_var.set(speed_run if isinstance(speed_run, bool) else str(speed_run).strip().lower() in {"1", "true", "yes", "on"})
        self.sieve_speed_pair_count_var.set(str(state.get("sieve_speed_pair_count") or "5"))
        self.sieve_take_profit_var.set(str(state.get("sieve_take_profit_pct") or "2"))
        self.sieve_stoploss_var.set(str(state.get("sieve_stoploss_pct") or "2"))
        auto_windows = state.get("sieve_auto_windows")
        self.sieve_auto_windows_var.set(True if auto_windows is None else str(auto_windows).strip().lower() not in {"0", "false", "no", "off"})
        self.sieve_auto_window_count_var.set(str(state.get("sieve_auto_window_count") or "2"))
        self.sieve_target_sweep_var.set(bool(state.get("sieve_target_sweep")))
        self.sieve_target_pairs_var.set(str(state.get("sieve_target_pairs") or "1/1, 1.5/1.5, 2/2, 3/2, 4/2, 2/3, 3/3"))
        self.sieve_result_batch_var.set(str(state.get("sieve_result_batch") or ""))
        self.sieve_result_batch_filter_var.set(str(state.get("sieve_result_batch_filter") or ""))
        self.sieve_filter_var.set(str(state.get("sieve_result_filter") or ""))
        self.sieve_filter_winrate_min_var.set(str(state.get("sieve_filter_winrate_min") or ""))
        self.sieve_filter_profit_min_var.set(str(state.get("sieve_filter_profit_min") or ""))
        self.sieve_filter_drawdown_max_var.set(str(state.get("sieve_filter_drawdown_max") or ""))
        self.sieve_filter_trades_min_var.set(str(state.get("sieve_filter_trades_min") or ""))
        self.sieve_filter_tp_eq_var.set(str(state.get("sieve_filter_tp_eq") or ""))
        self.sieve_filter_sl_eq_var.set(str(state.get("sieve_filter_sl_eq") or ""))
        saved_order = state.get("sieve_column_order")
        if isinstance(saved_order, list):
            order = [str(column) for column in saved_order]
            self.sieve_column_order = list(SIEVE_DEFAULT_COLUMN_ORDER) if self._is_legacy_sieve_column_order(order) else order
        else:
            self.sieve_column_order = list(SIEVE_DEFAULT_COLUMN_ORDER)
        self._apply_sieve_column_order()
        self._update_epochs_mode_state()
        self.refresh()
        if self.training_listbox is not None:
            self._select_values(self.training_listbox, settings.training_windows)
        if self.validation_listbox is not None:
            self._select_values(self.validation_listbox, settings.validation_windows)

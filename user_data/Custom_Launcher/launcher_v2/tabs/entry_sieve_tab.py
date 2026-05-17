from __future__ import annotations

from typing import Any
import tkinter as tk
from tkinter import ttk

from .explorer_tab import ExplorerTab, SIEVE_RESULT_COLUMNS, _split_list


DEFAULT_BATCH_QUEUE = "volume_profile,structure_levels,continuation_patterns,reversal_patterns,market_state_pressure,multi_confluence,small_concepts,avwap,zones"
FILTER_OPERATORS = (">=", ">", "<=", "<", "=", "!=")


class EntrySieveTab(ExplorerTab):
    """Entry Sieve controls and result review.

    This tab intentionally reuses the existing Entry Sieve command/result logic
    from ExplorerTab while presenting Sieve as its own top-level workflow.
    """

    tab_key = "entry_sieve"
    tab_title = "Entry Sieve"

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=0, sticky="nsew")

        config_tab = ttk.Frame(notebook, style="App.TFrame", padding=4)
        config_tab.grid_columnconfigure(0, weight=1)
        config_tab.grid_rowconfigure(1, weight=1)
        notebook.add(config_tab, text="Config")

        results_tab = ttk.Frame(notebook, style="App.TFrame", padding=4)
        results_tab.grid_columnconfigure(0, weight=1)
        results_tab.grid_rowconfigure(1, weight=1)
        notebook.add(results_tab, text="Results")

        self._build_config_tab(config_tab)
        self._build_results_tab(results_tab)

    def _build_config_tab(self, parent: ttk.Frame) -> None:
        controls = ttk.LabelFrame(parent, text="Entry Sieve Run Config")
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
        ttk.Label(controls, text="Backtest workers").grid(row=1, column=4, sticky="w", padx=8, pady=4)
        ttk.Combobox(controls, textvariable=self.backtest_worker_count_var, values=[str(index) for index in range(1, 21)], state="readonly").grid(row=1, column=5, sticky="ew", padx=8, pady=4)

        ttk.Checkbutton(controls, text="Auto windows", variable=self.sieve_auto_windows_var).grid(row=2, column=0, sticky="w", padx=8, pady=4)
        ttk.Label(controls, text="Windows/file").grid(row=2, column=1, sticky="e", padx=8, pady=4)
        ttk.Combobox(controls, textvariable=self.sieve_auto_window_count_var, values=("1", "2", "3"), state="readonly", width=6).grid(row=2, column=2, sticky="w", padx=8, pady=4)
        ttk.Checkbutton(controls, text="Target sweep", variable=self.sieve_target_sweep_var).grid(row=2, column=3, sticky="w", padx=8, pady=4)
        self._editable_entry(controls, 2, 4, "TP/SL grid", self.sieve_target_pairs_var)

        ttk.Label(controls, text="Queue priority").grid(row=3, column=0, sticky="w", padx=8, pady=4)
        ttk.Combobox(controls, textvariable=self.sieve_batch_priority_var, values=("least_run_first", "configured"), state="readonly").grid(row=3, column=1, sticky="ew", padx=8, pady=4)
        self.epochs_entry = self._editable_entry(controls, 3, 2, "Epochs", self.epochs_var)
        ttk.Checkbutton(controls, text="Auto epochs (20x params)", variable=self.auto_epochs_var).grid(row=3, column=4, sticky="w", padx=8, pady=4)
        self.auto_epochs_cap_entry = self._editable_entry(controls, 3, 5, "Auto epoch cap", self.auto_epochs_cap_var)

        ttk.Checkbutton(controls, text="Split-venv pipeline", variable=self.split_venv_pipeline_var).grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=4)
        self._editable_entry(controls, 4, 2, "Backtest Python", self.backtest_python_exe_var)
        self._editable_entry(controls, 4, 4, "Handoff dir", self.pipeline_handoff_dir_var)
        ttk.Label(controls, textvariable=self.sieve_status_var).grid(row=5, column=0, columnspan=6, sticky="w", padx=8, pady=4)

        queue_frame = ttk.LabelFrame(parent, text="Batch Queue")
        queue_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        queue_frame.grid_columnconfigure(0, weight=1)
        queue_frame.grid_rowconfigure(0, weight=1)
        self.sieve_batch_queue_listbox = tk.Listbox(queue_frame, selectmode=tk.EXTENDED, height=9, exportselection=False)
        self.sieve_batch_queue_listbox.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=(8, 0))
        queue_scroll = ttk.Scrollbar(queue_frame, orient="vertical", command=self.sieve_batch_queue_listbox.yview)
        queue_scroll.grid(row=0, column=1, sticky="ns", padx=(4, 8), pady=(8, 0))
        self.sieve_batch_queue_listbox.configure(yscrollcommand=queue_scroll.set)

        queue_buttons = ttk.Frame(queue_frame, style="App.TFrame")
        queue_buttons.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=8)
        ttk.Button(queue_buttons, text="Select all", command=self._select_all_sieve_batches).pack(side="left")
        ttk.Button(queue_buttons, text="Clear queue", command=self._clear_sieve_batch_queue).pack(side="left", padx=(8, 0))
        ttk.Button(queue_buttons, text="Run Entry Sieve", command=self._run_entry_sieve).pack(side="left", padx=(16, 0))
        ttk.Button(queue_buttons, text="Run selected batch queue", command=self._run_entry_sieve_batch_queue).pack(side="left", padx=(8, 0))
        ttk.Button(queue_buttons, text="Stop", command=self._stop_entry_sieve).pack(side="left", padx=(8, 0))
        ttk.Button(queue_buttons, text="Refresh batches", command=self.refresh).pack(side="left", padx=(8, 0))

        windows = ttk.Frame(parent, style="App.TFrame")
        windows.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        windows.grid_columnconfigure(0, weight=1)
        windows.grid_columnconfigure(1, weight=1)
        windows.grid_rowconfigure(0, weight=1)
        self.training_listbox = self._window_selector(windows, "Manual training windows", 0)
        self.validation_listbox = self._window_selector(windows, "Manual validation windows", 1)

    def _build_results_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, textvariable=self.sieve_status_var).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 0))

        files = ttk.LabelFrame(parent, text="Result File Management")
        files.grid(row=1, column=0, sticky="ew", padx=8, pady=8)
        for col in (1, 3):
            files.grid_columnconfigure(col, weight=1)

        ttk.Label(files, text="Result batches").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        self.sieve_result_batch_combo = ttk.Combobox(files, textvariable=self.sieve_result_batch_var, state="normal", width=96)
        self.sieve_result_batch_combo.grid(row=0, column=1, columnspan=5, sticky="ew", padx=8, pady=4)
        self._editable_entry(files, 1, 0, "Result file filter", self.sieve_result_batch_filter_var)
        ttk.Button(files, text="Refresh results", command=self._refresh_sieve_results).grid(row=1, column=2, sticky="w", padx=8, pady=4)
        ttk.Button(files, text="Open results folder", command=self._open_sieve_results_folder).grid(row=1, column=3, sticky="w", padx=8, pady=4)
        ttk.Button(files, text="Delete selected result", command=self._delete_selected_sieve_result_batches).grid(row=1, column=4, sticky="w", padx=8, pady=4)

        table = ttk.LabelFrame(parent, text="Table Filters and Columns")
        table.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        for col in (1, 4, 7, 10):
            table.grid_columnconfigure(col, weight=1)

        self._editable_entry(table, 0, 0, "Row contains", self.sieve_filter_var)
        self._filter_control(table, 0, 2, "Winrate %", self.sieve_filter_winrate_op_var, self.sieve_filter_winrate_min_var)
        self._filter_control(table, 0, 5, "Profit %", self.sieve_filter_profit_op_var, self.sieve_filter_profit_min_var)
        self._filter_control(table, 0, 8, "Max DD %", self.sieve_filter_drawdown_op_var, self.sieve_filter_drawdown_max_var)

        self._filter_control(table, 1, 0, "Trades", self.sieve_filter_trades_op_var, self.sieve_filter_trades_min_var)
        self._filter_control(table, 1, 3, "TP %", self.sieve_filter_tp_op_var, self.sieve_filter_tp_eq_var)
        self._filter_control(table, 1, 6, "SL %", self.sieve_filter_sl_op_var, self.sieve_filter_sl_eq_var)
        ttk.Button(table, text="Clear filters", command=self._clear_sieve_column_filters).grid(row=1, column=9, sticky="w", padx=8, pady=4)
        ttk.Button(table, text="Move column left", command=lambda: self._move_sieve_column(-1)).grid(row=2, column=0, sticky="w", padx=8, pady=4)
        ttk.Button(table, text="Move column right", command=lambda: self._move_sieve_column(1)).grid(row=2, column=1, sticky="w", padx=8, pady=4)
        ttk.Button(table, text="Reset columns", command=self._reset_sieve_columns).grid(row=2, column=2, sticky="w", padx=8, pady=4)

        results = ttk.LabelFrame(parent, text="Runtime results")
        results.grid(row=3, column=0, sticky="nsew", padx=8, pady=(0, 8))
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

    def _filter_control(self, parent: tk.Misc, row: int, column: int, label: str, operator_var: tk.StringVar, value_var: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=8, pady=4)
        ttk.Combobox(parent, textvariable=operator_var, values=FILTER_OPERATORS, state="readonly", width=4).grid(row=row, column=column + 1, sticky="w", padx=(0, 4), pady=4)
        entry = ttk.Entry(parent, textvariable=value_var, width=10)
        entry.grid(row=row, column=column + 2, sticky="ew", padx=(0, 8), pady=4)
        self.editable_entries.append(entry)

    def _bind_events(self) -> None:
        self.auto_epochs_var.trace_add("write", lambda *_: self._update_epochs_mode_state())
        self.sieve_speed_run_var.trace_add("write", lambda *_: self._update_epochs_mode_state())
        self.sieve_result_batch_filter_var.trace_add("write", lambda *_: self._refresh_sieve_results())
        for variable in (
            self.sieve_filter_var,
            self.sieve_filter_winrate_min_var,
            self.sieve_filter_profit_min_var,
            self.sieve_filter_drawdown_max_var,
            self.sieve_filter_trades_min_var,
            self.sieve_filter_tp_eq_var,
            self.sieve_filter_sl_eq_var,
            self.sieve_filter_winrate_op_var,
            self.sieve_filter_profit_op_var,
            self.sieve_filter_drawdown_op_var,
            self.sieve_filter_trades_op_var,
            self.sieve_filter_tp_op_var,
            self.sieve_filter_sl_op_var,
        ):
            variable.trace_add("write", lambda *_: self._refresh_sieve_results())
        if self.sieve_result_batch_combo is not None:
            self.sieve_result_batch_combo.bind("<<ComboboxSelected>>", lambda event: self._refresh_sieve_results(), add="+")
        if getattr(self, "sieve_batch_queue_listbox", None) is not None:
            self.sieve_batch_queue_listbox.bind("<<ListboxSelect>>", lambda event: self._sync_sieve_batch_queue_var(), add="+")
        self._update_epochs_mode_state()
        self._update_sieve_speed_run_state()

    def refresh(self) -> None:
        self._load_windows()
        self._load_sieve_batches()
        self._refresh_sieve_results()

    def _load_sieve_batches(self) -> None:
        try:
            batches = self.sieve_service.load_strategy_batches()
            values = [str(batch.get("id") or "") for batch in batches if str(batch.get("id") or "").strip()]
        except Exception as exc:
            batches = [{"id": "all", "label": "All Sieve1"}]
            values = ["all"]
            self.context.shared.status.set(f"Entry Sieve batch load failed: {exc}")

        if self.sieve_strategy_batch_combo is not None:
            self.sieve_strategy_batch_combo.configure(values=values)
        if self.sieve_strategy_batch_var.get() not in values:
            self.sieve_strategy_batch_var.set(values[0] if values else "all")

        listbox = getattr(self, "sieve_batch_queue_listbox", None)
        if listbox is None:
            return
        current = _split_list(self.sieve_batch_queue_var.get())
        if not current:
            current = [batch_id for batch_id in values if batch_id != "all"] or values
        wanted = set(current)
        listbox.delete(0, tk.END)
        for batch in batches:
            batch_id = str(batch.get("id") or "").strip()
            if not batch_id:
                continue
            label = str(batch.get("label") or batch_id).strip()
            listbox.insert(tk.END, f"{batch_id}    {label}")
        self._select_values(listbox, [batch_id for batch_id in values if batch_id in wanted])
        if not listbox.curselection():
            self._select_values(listbox, [batch_id for batch_id in values if batch_id != "all"] or values)
        self._sync_sieve_batch_queue_var()

    def _sync_sieve_batch_queue_var(self) -> None:
        listbox = getattr(self, "sieve_batch_queue_listbox", None)
        if listbox is None:
            return
        self.sieve_batch_queue_var.set(",".join(self._selected_values(listbox)))

    def _select_all_sieve_batches(self) -> None:
        listbox = getattr(self, "sieve_batch_queue_listbox", None)
        if listbox is None:
            return
        listbox.selection_clear(0, tk.END)
        for index in range(listbox.size()):
            batch_id = str(listbox.get(index)).split()[0]
            if batch_id != "all":
                listbox.selection_set(index)
        self._sync_sieve_batch_queue_var()

    def _clear_sieve_batch_queue(self) -> None:
        listbox = getattr(self, "sieve_batch_queue_listbox", None)
        if listbox is None:
            return
        listbox.selection_clear(0, tk.END)
        self._sync_sieve_batch_queue_var()

    def _run_entry_sieve_batch_queue(self) -> None:
        self._sync_sieve_batch_queue_var()
        super()._run_entry_sieve_batch_queue()

    def _stop_entry_sieve(self) -> None:
        self.context.process_runner.stop(timeout_seconds=8.0)
        self.sieve_status_var.set("Run status: stop requested")
        self.context.shared.status.set("Entry Sieve stop requested")

    def on_app_event(self, event: str, payload: dict[str, Any]) -> None:
        if event != "process_output":
            return
        text = str(payload.get("text") or "").strip()
        if not text:
            return
        stream = str(payload.get("stream") or "")
        if stream.startswith("entry_sieve"):
            self.sieve_status_var.set(text.splitlines()[-1])

    def set_state(self, state: dict[str, Any]) -> None:
        if not str(state.get("sieve_batch_queue") or "").strip():
            state = dict(state)
            state["sieve_batch_queue"] = DEFAULT_BATCH_QUEUE
        super().set_state(state)

    def get_state(self) -> dict[str, Any]:
        return {
            "training_windows": self._selected_values(self.training_listbox) if self.training_listbox is not None else [],
            "validation_windows": self._selected_values(self.validation_listbox) if self.validation_listbox is not None else [],
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
            "sieve_filter_winrate_op": self.sieve_filter_winrate_op_var.get(),
            "sieve_filter_profit_op": self.sieve_filter_profit_op_var.get(),
            "sieve_filter_drawdown_op": self.sieve_filter_drawdown_op_var.get(),
            "sieve_filter_trades_op": self.sieve_filter_trades_op_var.get(),
            "sieve_filter_tp_op": self.sieve_filter_tp_op_var.get(),
            "sieve_filter_sl_op": self.sieve_filter_sl_op_var.get(),
            "sieve_column_order": list(self.sieve_column_order),
        }

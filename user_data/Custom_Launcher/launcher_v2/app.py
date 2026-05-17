from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tkinter as tk
from tkinter import ttk
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "launcher_v2"

from .context import LauncherContext, SharedVars
from .process_runner import ProcessRunner
from .tabs.run_tab import RunTab
from .tabs.common_tab import CommonTab
from .tabs.pairs_tab import PairsTab
from .tabs.mode_options_tab import ModeOptionsTab
from .tabs.data_management_tab import DataManagementTab
from .tabs.explorer_tab import ExplorerTab
from .tabs.entry_sieve_tab import EntrySieveTab
from .tabs.review_tab import ReviewTab
from .tabs.file_converter_tab import FileConverterTab


APP_TITLE = "Freqtrade Launcher V2"
AUTO_PRESET_NAME = "LauncherV2-auto"
FALLBACK_PRESET_NAME = "BackTest2021-26"
OUTPUT_DRAIN_MAX_LINES = 250
OUTPUT_DRAIN_MAX_CHARS = 120_000
ENTRY_SIEVE_STATUS_POLL_MS = 3000


class LauncherV2(tk.Tk):
    """Small app shell for LauncherV2.

    This class should stay small. Feature logic belongs in tabs/services.
    """

    tab_classes = [
        RunTab,
        CommonTab,
        PairsTab,
        ModeOptionsTab,
        DataManagementTab,
        ExplorerTab,
        EntrySieveTab,
        ReviewTab,
        FileConverterTab,
    ]

    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1240x900")
        self.minsize(1000, 720)
        self.configure(background="#f4f6fa")
        self._configure_style()

        app_dir = Path(__file__).resolve().parent.parent
        user_data_dir = app_dir.parent
        project_root = user_data_dir.parent
        self.output_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.shared = SharedVars(
            project_root=tk.StringVar(value=str(project_root)),
            python_exe=tk.StringVar(value=sys.executable),
            userdir=tk.StringVar(value=str(user_data_dir)),
            datadir=tk.StringVar(value=str(user_data_dir / "data")),
            command_preview=tk.StringVar(value=""),
            status=tk.StringVar(value="Ready"),
        )
        self.state_path = app_dir / "launcher_v2" / "runtime" / "launcher_v2_state.json"
        self.process_runner = ProcessRunner(self.output_queue, log_dir=app_dir / "launcher_v2" / "runtime" / "process_logs")
        self._closing = False
        self.context = LauncherContext(
            app_dir=app_dir,
            preset_path=self.state_path,
            output_queue=self.output_queue,
            shared=self.shared,
            process_runner=self.process_runner,
            preset_manager=None,
            notify=self._notify_tabs,
        )
        self.tabs: dict[str, Any] = {}
        self._entry_sieve_console_signature = ""
        self._entry_sieve_console_had_active = False
        self._entry_sieve_log_offsets: dict[str, int] = {}
        self._build_ui()
        self.load_execute_preset()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._drain_output_queue)
        self.after(1000, self._poll_entry_sieve_console_status)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.option_add("*Font", ("Segoe UI", 10))
        style.configure(".", font=("Segoe UI", 10), background="#f4f6fa", foreground="#172033")
        style.configure("App.TFrame", background="#f4f6fa")
        style.configure("Header.TFrame", background="#172033")
        style.configure("HeaderTitle.TLabel", background="#172033", foreground="#ffffff", font=("Segoe UI Semibold", 13))
        style.configure("HeaderMeta.TLabel", background="#172033", foreground="#b9c3d6")
        style.configure("Status.TLabel", background="#172033", foreground="#dbe7ff")
        style.configure("TNotebook", background="#f4f6fa", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 8), font=("Segoe UI", 10))
        style.map("TNotebook.Tab", background=[("selected", "#ffffff")], foreground=[("selected", "#172033")])
        style.configure("TLabelframe", background="#f4f6fa", bordercolor="#d8dee9", relief="solid")
        style.configure("TLabelframe.Label", background="#f4f6fa", foreground="#334155", font=("Segoe UI Semibold", 10))
        style.configure("TButton", padding=(10, 5))
        style.configure("Treeview", rowheight=25, background="#ffffff", fieldbackground="#ffffff")
        style.configure("Treeview.Heading", font=("Segoe UI Semibold", 10), background="#e9eef7")

    def _build_ui(self) -> None:
        top = ttk.Frame(self, style="Header.TFrame")
        top.pack(fill="x")
        title_area = ttk.Frame(top, style="Header.TFrame")
        title_area.pack(side="left", padx=14, pady=10)
        ttk.Label(title_area, text="Freqtrade Launcher V2", style="HeaderTitle.TLabel").pack(anchor="w")
        ttk.Label(title_area, text="Auto preset saved on execute actions", style="HeaderMeta.TLabel").pack(anchor="w")
        ttk.Label(top, textvariable=self.shared.status, style="Status.TLabel").pack(side="right", padx=14)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        for tab_cls in self.tab_classes:
            tab = tab_cls(self.notebook, self.context)
            self.tabs[tab.tab_key] = tab
            self.context.registry[tab.tab_key] = tab
            if hasattr(tab, "child_tabs"):
                self.tabs.update(tab.child_tabs)
                self.context.registry.update(tab.child_tabs)
            self.notebook.add(tab, text=tab.tab_title)

    def collect_state(self) -> dict[str, Any]:
        return {key: tab.get_state() for key, tab in self.tabs.items()}

    def apply_state(self, state: dict[str, Any]) -> None:
        for key, tab in self.tabs.items():
            tab_state = state.get(key, {}) if isinstance(state, dict) else {}
            if isinstance(tab_state, dict):
                tab.set_state(tab_state)

    def _preset_path(self) -> Path:
        return self.context.app_dir / "launcher_v2" / "config" / "presets.json"

    def _load_presets(self) -> dict[str, Any]:
        path = self._preset_path()
        if not path.exists():
            return {}
        for encoding in ("utf-8", "utf-8-sig"):
            try:
                payload = json.loads(path.read_text(encoding=encoding))
                if isinstance(payload, dict):
                    return payload
            except Exception:
                continue
        return {}

    def load_execute_preset(self) -> None:
        presets = self._load_presets()
        preset = presets.get(AUTO_PRESET_NAME) or presets.get(FALLBACK_PRESET_NAME)
        if not isinstance(preset, dict) and presets:
            first_value = next(iter(presets.values()))
            preset = first_value if isinstance(first_value, dict) else None
        if not isinstance(preset, dict):
            self.shared.status.set("No preset file found; using tab defaults")
            return
        self._apply_execute_preset(preset)
        self.shared.status.set(f"Loaded preset: {AUTO_PRESET_NAME if AUTO_PRESET_NAME in presets else FALLBACK_PRESET_NAME}")

    def _apply_execute_preset(self, preset: dict[str, Any]) -> None:
        mapping = {
            "run": {
                "run_type": preset.get("run_type"),
                "extra_args": preset.get("run_extra_args"),
                "raw_console": preset.get("run_raw_console"),
                "results_console": preset.get("run_results_console"),
            },
            "common": {
                "project_root": preset.get("project_root"),
                "python_exe": preset.get("python_exe"),
                "userdir": preset.get("userdir"),
                "datadir": preset.get("datadir"),
                "config_files": preset.get("config_files"),
                "strategy_file": preset.get("strategy_file"),
                "strategy_class": preset.get("strategy_class"),
                "recursive_strategy_search": preset.get("recursive_strategy_search"),
            },
            "pairs": {
                "pair_mode": preset.get("pair_mode"),
                "pairs": preset.get("pairs"),
                "blacklist": preset.get("blacklist"),
            },
            "mode_options": {
                "timeframe": preset.get("timeframe"),
                "timerange": preset.get("timerange"),
                "max_open_trades": preset.get("max_open_trades"),
                "stake_amount": preset.get("stake_amount"),
                "dry_run_wallet": preset.get("dry_run_wallet"),
                "fee": preset.get("fee"),
                "export": preset.get("backtest_export"),
                "breakdown": preset.get("backtest_breakdown"),
                "epochs": preset.get("hyperopt_epochs"),
                "spaces": preset.get("hyperopt_spaces"),
                "jobs": preset.get("hyperopt_jobs"),
                "random_state": preset.get("hyperopt_random_state"),
                "hyperopt_loss": preset.get("hyperopt_loss"),
                "ignore_missing_spaces": preset.get("hyperopt_ignore_missing_spaces"),
                "disable_param_export": preset.get("hyperopt_disable_param_export"),
            },
            "download": {
                "exchange": preset.get("download_exchange"),
                "pairs_file": preset.get("download_pairs_file"),
                "pairs": preset.get("download_pairs"),
                "timeframes": preset.get("download_timeframes"),
                "days": preset.get("download_days"),
                "new_pairs_days": preset.get("download_new_pairs_days"),
                "timerange": preset.get("download_timerange"),
                "trading_mode": preset.get("download_trading_mode"),
                "candle_types": preset.get("download_candle_types"),
                "data_format_ohlcv": preset.get("download_data_format_ohlcv"),
                "data_format_trades": preset.get("download_data_format_trades"),
                "include_inactive": preset.get("download_include_inactive"),
                "no_parallel": preset.get("download_no_parallel"),
                "dl_trades": preset.get("download_dl_trades"),
                "convert": preset.get("download_convert"),
                "erase": preset.get("download_erase"),
                "prepend": preset.get("download_prepend"),
            },
            "review": {
                "hyperopt_file": preset.get("review_hyperopt_file"),
                "hyperopt_limit": preset.get("review_hyperopt_limit"),
                "hyperopt_index": preset.get("review_hyperopt_index"),
                "hyperopt_best": preset.get("review_hyperopt_best"),
                "hyperopt_profitable": preset.get("review_hyperopt_profitable"),
                "hyperopt_print_json": preset.get("review_hyperopt_print_json"),
                "hyperopt_no_header": preset.get("review_hyperopt_no_header"),
                "hyperopt_no_details": preset.get("review_hyperopt_no_details"),
                "hyperopt_breakdown": preset.get("review_hyperopt_breakdown"),
                "backtest_file": preset.get("review_backtest_file"),
                "backtest_directory": preset.get("review_backtest_directory"),
                "backtest_show_pair_list": preset.get("review_backtest_show_pair_list"),
                "backtest_breakdown": preset.get("review_backtest_breakdown"),
                "analysis_groups": preset.get("review_analysis_groups"),
                "enter_reasons": preset.get("review_enter_reasons"),
                "exit_reasons": preset.get("review_exit_reasons"),
                "indicator_list": preset.get("review_indicator_list"),
                "entry_only": preset.get("review_entry_only"),
                "exit_only": preset.get("review_exit_only"),
                "rejected_signals": preset.get("review_rejected_signals"),
                "analysis_to_csv": preset.get("review_analysis_to_csv"),
                "analysis_csv_path": preset.get("review_analysis_csv_path"),
                "console": preset.get("review_console"),
            },
            "explorer": {
                "target_type": preset.get("explorer_target_type"),
                "target_selection": preset.get("explorer_target_selection"),
                "target_name": preset.get("explorer_target_name"),
                "search_breadth": preset.get("explorer_search_breadth"),
                "training_windows": preset.get("explorer_training_windows"),
                "validation_windows": preset.get("explorer_validation_windows"),
                "max_loops": preset.get("explorer_max_loops"),
                "epochs": preset.get("explorer_epochs"),
                "auto_epochs": preset.get("explorer_auto_epochs"),
                "auto_epochs_cap": preset.get("explorer_auto_epochs_cap"),
                "random_state": preset.get("explorer_random_state"),
                "sampling_seed": preset.get("explorer_sampling_seed"),
                "split_venv_pipeline": preset.get("explorer_split_venv_pipeline"),
                "backtest_python_exe": preset.get("explorer_backtest_python_exe"),
                "backtest_python_exes": preset.get("explorer_backtest_python_exes"),
                "backtest_worker_count": preset.get("explorer_backtest_worker_count"),
                "pipeline_handoff_dir": preset.get("explorer_pipeline_handoff_dir"),
            },
            "entry_sieve": {
                "training_windows": preset.get("entry_sieve_training_windows", preset.get("explorer_training_windows")),
                "validation_windows": preset.get("entry_sieve_validation_windows", preset.get("explorer_validation_windows")),
                "epochs": preset.get("entry_sieve_epochs", preset.get("explorer_epochs")),
                "auto_epochs": preset.get("entry_sieve_auto_epochs", preset.get("explorer_auto_epochs")),
                "auto_epochs_cap": preset.get("entry_sieve_auto_epochs_cap", preset.get("explorer_auto_epochs_cap")),
                "sieve_hyperopt_jobs": preset.get("entry_sieve_hyperopt_jobs", preset.get("hyperopt_jobs")),
                "random_state": preset.get("entry_sieve_random_state", preset.get("explorer_random_state")),
                "sampling_seed": preset.get("entry_sieve_sampling_seed", preset.get("explorer_sampling_seed")),
                "split_venv_pipeline": preset.get("entry_sieve_split_venv_pipeline", preset.get("explorer_split_venv_pipeline")),
                "backtest_python_exe": preset.get("entry_sieve_backtest_python_exe", preset.get("explorer_backtest_python_exe")),
                "backtest_python_exes": preset.get("entry_sieve_backtest_python_exes", preset.get("explorer_backtest_python_exes")),
                "backtest_worker_count": preset.get("entry_sieve_backtest_worker_count", preset.get("explorer_backtest_worker_count")),
                "pipeline_handoff_dir": preset.get("entry_sieve_pipeline_handoff_dir", preset.get("explorer_pipeline_handoff_dir")),
                "sieve_strategy_batch": preset.get("entry_sieve_strategy_batch", preset.get("explorer_sieve_strategy_batch")),
                "sieve_batch_queue": preset.get("entry_sieve_batch_queue", preset.get("explorer_sieve_batch_queue")),
                "sieve_batch_priority": preset.get("entry_sieve_batch_priority", preset.get("explorer_sieve_batch_priority")),
                "sieve_strategy_filter": preset.get("entry_sieve_strategy_filter", preset.get("explorer_sieve_strategy_filter")),
                "sieve_speed_run": preset.get("entry_sieve_speed_run", preset.get("explorer_sieve_speed_run")),
                "sieve_speed_pair_count": preset.get("entry_sieve_speed_pair_count", preset.get("explorer_sieve_speed_pair_count")),
                "sieve_take_profit_pct": preset.get("entry_sieve_take_profit_pct", preset.get("explorer_sieve_take_profit_pct")),
                "sieve_stoploss_pct": preset.get("entry_sieve_stoploss_pct", preset.get("explorer_sieve_stoploss_pct")),
                "sieve_auto_windows": preset.get("entry_sieve_auto_windows", preset.get("explorer_sieve_auto_windows")),
                "sieve_auto_window_count": preset.get("entry_sieve_auto_window_count", preset.get("explorer_sieve_auto_window_count")),
                "sieve_target_sweep": preset.get("entry_sieve_target_sweep", preset.get("explorer_sieve_target_sweep")),
                "sieve_target_pairs": preset.get("entry_sieve_target_pairs", preset.get("explorer_sieve_target_pairs")),
                "sieve_result_batch": preset.get("entry_sieve_result_batch", preset.get("explorer_sieve_result_batch")),
                "sieve_result_batch_filter": preset.get("entry_sieve_result_batch_filter", preset.get("explorer_sieve_result_batch_filter")),
                "sieve_result_filter": preset.get("entry_sieve_result_filter", preset.get("explorer_sieve_result_filter")),
                "sieve_filter_winrate_min": preset.get("entry_sieve_filter_winrate_min", preset.get("explorer_sieve_filter_winrate_min")),
                "sieve_filter_profit_min": preset.get("entry_sieve_filter_profit_min", preset.get("explorer_sieve_filter_profit_min")),
                "sieve_filter_drawdown_max": preset.get("entry_sieve_filter_drawdown_max", preset.get("explorer_sieve_filter_drawdown_max")),
                "sieve_filter_trades_min": preset.get("entry_sieve_filter_trades_min", preset.get("explorer_sieve_filter_trades_min")),
                "sieve_filter_tp_eq": preset.get("entry_sieve_filter_tp_eq", preset.get("explorer_sieve_filter_tp_eq")),
                "sieve_filter_sl_eq": preset.get("entry_sieve_filter_sl_eq", preset.get("explorer_sieve_filter_sl_eq")),
                "sieve_filter_winrate_op": preset.get("entry_sieve_filter_winrate_op", preset.get("explorer_sieve_filter_winrate_op")),
                "sieve_filter_profit_op": preset.get("entry_sieve_filter_profit_op", preset.get("explorer_sieve_filter_profit_op")),
                "sieve_filter_drawdown_op": preset.get("entry_sieve_filter_drawdown_op", preset.get("explorer_sieve_filter_drawdown_op")),
                "sieve_filter_trades_op": preset.get("entry_sieve_filter_trades_op", preset.get("explorer_sieve_filter_trades_op")),
                "sieve_filter_tp_op": preset.get("entry_sieve_filter_tp_op", preset.get("explorer_sieve_filter_tp_op")),
                "sieve_filter_sl_op": preset.get("entry_sieve_filter_sl_op", preset.get("explorer_sieve_filter_sl_op")),
                "sieve_column_order": preset.get("entry_sieve_column_order", preset.get("explorer_sieve_column_order")),
            },
            "news": {
                "config_path": preset.get("news_config_path"),
                "data_dir": preset.get("news_data_dir"),
                "db_path": preset.get("news_db_path"),
                "interval_minutes": preset.get("news_interval_minutes"),
                "once": preset.get("news_once"),
            },
            "web": {
                "config_path": preset.get("web_config_path"),
                "data_dir": preset.get("web_data_dir"),
                "db_path": preset.get("web_db_path"),
                "interval_minutes": preset.get("web_interval_minutes"),
                "once": preset.get("web_once"),
            },
            "global_context": {
                "config_path": preset.get("global_context_config_path"),
                "data_dir": preset.get("global_context_data_dir"),
                "db_path": preset.get("global_context_db_path"),
                "key_file": preset.get("global_context_key_file"),
                "fred_key_json_path": preset.get("global_context_fred_key_json_path"),
                "enable_fred": preset.get("global_context_enable_fred"),
                "interval_minutes": preset.get("global_context_interval_minutes"),
                "once": preset.get("global_context_once"),
            },
            "orderbook": {
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
                "store_snapshots": preset.get("orderbook_store_snapshots"),
                "history_datadir": preset.get("orderbook_history_datadir"),
                "history_exchange": preset.get("orderbook_history_exchange"),
                "history_trading_mode": preset.get("orderbook_history_trading_mode"),
                "history_category": preset.get("orderbook_history_category"),
                "history_depth": preset.get("orderbook_history_depth"),
                "history_timerange": preset.get("orderbook_history_timerange"),
                "history_feature_timeframes": preset.get("orderbook_history_feature_timeframes"),
                "history_feature_format": preset.get("orderbook_history_feature_format"),
                "history_max_rows": preset.get("orderbook_history_max_rows"),
                "history_erase": preset.get("orderbook_history_erase"),
                "history_pairs": preset.get("orderbook_history_pairs"),
            },
            "data_watchdog": {
                "task_name": preset.get("data_watchdog_task_name"),
                "check_interval_minutes": preset.get("data_watchdog_check_interval_minutes"),
                "heartbeat_stale_minutes": preset.get("data_watchdog_heartbeat_stale_minutes"),
                "restart_dead": preset.get("data_watchdog_restart_dead"),
                "services": preset.get("data_watchdog_services"),
            },
            "file_converter": {
                "root_paths": preset.get("file_converter_root_paths"),
                "output_extension": preset.get("file_converter_output_extension"),
                "recursive": preset.get("file_converter_recursive"),
                "filter": preset.get("file_converter_filter"),
                "replace_existing": preset.get("file_converter_replace_existing"),
                "console": preset.get("file_converter_console"),
            },
        }
        for tab_key, state in mapping.items():
            tab = self.tabs.get(tab_key)
            if tab is None:
                continue
            filtered = {key: value for key, value in state.items() if value is not None}
            if filtered:
                tab.set_state(filtered)

    def save_execute_preset(self, reason: str = "execute") -> None:
        presets = self._load_presets()
        tabs = self.collect_state()
        common = tabs.get("common", {})
        pairs = tabs.get("pairs", {})
        mode = tabs.get("mode_options", {})
        download = tabs.get("download", {})
        review = tabs.get("review", {})
        explorer = tabs.get("explorer", {})
        entry_sieve = tabs.get("entry_sieve", {})
        news = tabs.get("news", {})
        web = tabs.get("web", {})
        global_context = tabs.get("global_context", {})
        orderbook = tabs.get("orderbook", {})
        data_watchdog = tabs.get("data_watchdog", {})
        file_converter = tabs.get("file_converter", {})
        run = tabs.get("run", {})
        preset = {
            "run_type": run.get("run_type", "Backtest"),
            "run_extra_args": run.get("extra_args", ""),
            "run_raw_console": dict(run.get("raw_console") or {}),
            "run_results_console": dict(run.get("results_console") or {}),
            "reason": reason,
            "project_root": common.get("project_root", ""),
            "python_exe": common.get("python_exe", ""),
            "userdir": common.get("userdir", ""),
            "datadir": common.get("datadir", ""),
            "config_files": list(common.get("config_files") or []),
            "strategy_file": common.get("strategy_file", ""),
            "strategy_class": common.get("strategy_class", ""),
            "recursive_strategy_search": bool(common.get("recursive_strategy_search", False)),
            "pair_mode": pairs.get("pair_mode", "manual"),
            "pairs": pairs.get("pairs", ""),
            "blacklist": pairs.get("blacklist", ""),
            "timeframe": mode.get("timeframe", ""),
            "timerange": mode.get("timerange", ""),
            "max_open_trades": mode.get("max_open_trades", ""),
            "stake_amount": mode.get("stake_amount", ""),
            "dry_run_wallet": mode.get("dry_run_wallet", ""),
            "fee": mode.get("fee", ""),
            "backtest_export": mode.get("export", "trades"),
            "backtest_breakdown": mode.get("breakdown", "day"),
            "hyperopt_epochs": mode.get("epochs", ""),
            "hyperopt_spaces": mode.get("spaces", ""),
            "hyperopt_jobs": mode.get("jobs", ""),
            "hyperopt_random_state": mode.get("random_state", ""),
            "hyperopt_loss": mode.get("hyperopt_loss", ""),
            "hyperopt_ignore_missing_spaces": bool(mode.get("ignore_missing_spaces", True)),
            "hyperopt_disable_param_export": bool(mode.get("disable_param_export", False)),
            "download_exchange": download.get("exchange", ""),
            "download_pairs_file": download.get("pairs_file", ""),
            "download_pairs": download.get("pairs", ""),
            "download_timeframes": download.get("timeframes", ""),
            "download_days": download.get("days", ""),
            "download_new_pairs_days": download.get("new_pairs_days", ""),
            "download_timerange": download.get("timerange", ""),
            "download_trading_mode": download.get("trading_mode", ""),
            "download_candle_types": download.get("candle_types", ""),
            "download_data_format_ohlcv": download.get("data_format_ohlcv", ""),
            "download_data_format_trades": download.get("data_format_trades", ""),
            "download_include_inactive": bool(download.get("include_inactive", False)),
            "download_no_parallel": bool(download.get("no_parallel", False)),
            "download_dl_trades": bool(download.get("dl_trades", False)),
            "download_convert": bool(download.get("convert", False)),
            "download_erase": bool(download.get("erase", False)),
            "download_prepend": bool(download.get("prepend", False)),
            "review_hyperopt_file": review.get("hyperopt_file", ""),
            "review_hyperopt_limit": review.get("hyperopt_limit", "20"),
            "review_hyperopt_index": review.get("hyperopt_index", "-1"),
            "review_hyperopt_best": bool(review.get("hyperopt_best", False)),
            "review_hyperopt_profitable": bool(review.get("hyperopt_profitable", False)),
            "review_hyperopt_print_json": bool(review.get("hyperopt_print_json", False)),
            "review_hyperopt_no_header": bool(review.get("hyperopt_no_header", False)),
            "review_hyperopt_no_details": bool(review.get("hyperopt_no_details", False)),
            "review_hyperopt_breakdown": review.get("hyperopt_breakdown", "none"),
            "review_backtest_file": review.get("backtest_file", ""),
            "review_backtest_directory": review.get("backtest_directory", ""),
            "review_backtest_show_pair_list": bool(review.get("backtest_show_pair_list", False)),
            "review_backtest_breakdown": review.get("backtest_breakdown", "none"),
            "review_analysis_groups": review.get("analysis_groups", ""),
            "review_enter_reasons": review.get("enter_reasons", ""),
            "review_exit_reasons": review.get("exit_reasons", ""),
            "review_indicator_list": review.get("indicator_list", ""),
            "review_entry_only": bool(review.get("entry_only", False)),
            "review_exit_only": bool(review.get("exit_only", False)),
            "review_rejected_signals": bool(review.get("rejected_signals", False)),
            "review_analysis_to_csv": bool(review.get("analysis_to_csv", False)),
            "review_analysis_csv_path": review.get("analysis_csv_path", ""),
            "review_console": dict(review.get("console") or {}),
            "explorer_target_type": explorer.get("target_type", "family"),
            "explorer_target_selection": explorer.get("target_selection", "random"),
            "explorer_target_name": explorer.get("target_name", ""),
            "explorer_search_breadth": explorer.get("search_breadth", "targeted"),
            "explorer_training_windows": list(explorer.get("training_windows") or []),
            "explorer_validation_windows": list(explorer.get("validation_windows") or []),
            "explorer_max_loops": explorer.get("max_loops", "0"),
            "explorer_epochs": explorer.get("epochs", "200"),
            "explorer_auto_epochs": bool(explorer.get("auto_epochs", False)),
            "explorer_auto_epochs_cap": explorer.get("auto_epochs_cap", ""),
            "explorer_random_state": explorer.get("random_state", ""),
            "explorer_sampling_seed": explorer.get("sampling_seed", ""),
            "explorer_split_venv_pipeline": bool(explorer.get("split_venv_pipeline", False)),
            "explorer_backtest_python_exe": explorer.get("backtest_python_exe", ""),
            "explorer_backtest_python_exes": list(explorer.get("backtest_python_exes") or []),
            "explorer_backtest_worker_count": explorer.get("backtest_worker_count", "2"),
            "explorer_pipeline_handoff_dir": explorer.get("pipeline_handoff_dir", ""),
            "entry_sieve_training_windows": list(entry_sieve.get("training_windows") or []),
            "entry_sieve_validation_windows": list(entry_sieve.get("validation_windows") or []),
            "entry_sieve_epochs": entry_sieve.get("epochs", "200"),
            "entry_sieve_auto_epochs": bool(entry_sieve.get("auto_epochs", False)),
            "entry_sieve_auto_epochs_cap": entry_sieve.get("auto_epochs_cap", ""),
            "entry_sieve_hyperopt_jobs": entry_sieve.get("sieve_hyperopt_jobs", ""),
            "entry_sieve_random_state": entry_sieve.get("random_state", ""),
            "entry_sieve_sampling_seed": entry_sieve.get("sampling_seed", ""),
            "entry_sieve_split_venv_pipeline": bool(entry_sieve.get("split_venv_pipeline", False)),
            "entry_sieve_backtest_python_exe": entry_sieve.get("backtest_python_exe", ""),
            "entry_sieve_backtest_python_exes": list(entry_sieve.get("backtest_python_exes") or []),
            "entry_sieve_backtest_worker_count": entry_sieve.get("backtest_worker_count", "2"),
            "entry_sieve_pipeline_handoff_dir": entry_sieve.get("pipeline_handoff_dir", ""),
            "entry_sieve_strategy_batch": entry_sieve.get("sieve_strategy_batch", "all"),
            "entry_sieve_batch_queue": entry_sieve.get("sieve_batch_queue", ""),
            "entry_sieve_batch_priority": entry_sieve.get("sieve_batch_priority", "least_run_first"),
            "entry_sieve_strategy_filter": entry_sieve.get("sieve_strategy_filter", "sieve1_*.py"),
            "entry_sieve_speed_run": bool(entry_sieve.get("sieve_speed_run", False)),
            "entry_sieve_speed_pair_count": entry_sieve.get("sieve_speed_pair_count", "5"),
            "entry_sieve_take_profit_pct": entry_sieve.get("sieve_take_profit_pct", "2"),
            "entry_sieve_stoploss_pct": entry_sieve.get("sieve_stoploss_pct", "2"),
            "entry_sieve_auto_windows": bool(entry_sieve.get("sieve_auto_windows", True)),
            "entry_sieve_auto_window_count": entry_sieve.get("sieve_auto_window_count", "2"),
            "entry_sieve_target_sweep": bool(entry_sieve.get("sieve_target_sweep", False)),
            "entry_sieve_target_pairs": entry_sieve.get("sieve_target_pairs", ""),
            "entry_sieve_result_batch": entry_sieve.get("sieve_result_batch", ""),
            "entry_sieve_result_batch_filter": entry_sieve.get("sieve_result_batch_filter", ""),
            "entry_sieve_result_filter": entry_sieve.get("sieve_result_filter", ""),
            "entry_sieve_filter_winrate_min": entry_sieve.get("sieve_filter_winrate_min", ""),
            "entry_sieve_filter_profit_min": entry_sieve.get("sieve_filter_profit_min", ""),
            "entry_sieve_filter_drawdown_max": entry_sieve.get("sieve_filter_drawdown_max", ""),
            "entry_sieve_filter_trades_min": entry_sieve.get("sieve_filter_trades_min", ""),
            "entry_sieve_filter_tp_eq": entry_sieve.get("sieve_filter_tp_eq", ""),
            "entry_sieve_filter_sl_eq": entry_sieve.get("sieve_filter_sl_eq", ""),
            "entry_sieve_filter_winrate_op": entry_sieve.get("sieve_filter_winrate_op", ">="),
            "entry_sieve_filter_profit_op": entry_sieve.get("sieve_filter_profit_op", ">="),
            "entry_sieve_filter_drawdown_op": entry_sieve.get("sieve_filter_drawdown_op", "<="),
            "entry_sieve_filter_trades_op": entry_sieve.get("sieve_filter_trades_op", ">="),
            "entry_sieve_filter_tp_op": entry_sieve.get("sieve_filter_tp_op", "="),
            "entry_sieve_filter_sl_op": entry_sieve.get("sieve_filter_sl_op", "="),
            "entry_sieve_column_order": list(entry_sieve.get("sieve_column_order") or []),
            "news_config_path": news.get("config_path", ""),
            "news_data_dir": news.get("data_dir", ""),
            "news_db_path": news.get("db_path", ""),
            "news_interval_minutes": news.get("interval_minutes", ""),
            "news_once": bool(news.get("once", False)),
            "web_config_path": web.get("config_path", ""),
            "web_data_dir": web.get("data_dir", ""),
            "web_db_path": web.get("db_path", ""),
            "web_interval_minutes": web.get("interval_minutes", ""),
            "web_once": bool(web.get("once", False)),
            "global_context_config_path": global_context.get("config_path", ""),
            "global_context_data_dir": global_context.get("data_dir", ""),
            "global_context_db_path": global_context.get("db_path", ""),
            "global_context_key_file": global_context.get("key_file", ""),
            "global_context_fred_key_json_path": global_context.get("fred_key_json_path", "fred.api_key"),
            "global_context_enable_fred": bool(global_context.get("enable_fred", True)),
            "global_context_interval_minutes": global_context.get("interval_minutes", ""),
            "global_context_once": bool(global_context.get("once", False)),
            "orderbook_config_path": orderbook.get("config_path", ""),
            "orderbook_data_dir": orderbook.get("data_dir", ""),
            "orderbook_market_profiles": list(orderbook.get("market_profiles") or []),
            "orderbook_depth_levels": orderbook.get("depth_levels", ""),
            "orderbook_stream_update_ms": orderbook.get("stream_update_ms", ""),
            "orderbook_metric_interval_seconds": orderbook.get("metric_interval_seconds", ""),
            "orderbook_context_poll_seconds": orderbook.get("context_poll_seconds", ""),
            "orderbook_context_period": orderbook.get("context_period", ""),
            "orderbook_snapshot_interval_seconds": orderbook.get("snapshot_interval_seconds", ""),
            "orderbook_capacity_warning_mb": orderbook.get("capacity_warning_mb", ""),
            "orderbook_capacity_critical_mb": orderbook.get("capacity_critical_mb", ""),
            "orderbook_max_symbols": orderbook.get("max_symbols", ""),
            "orderbook_store_snapshots": bool(orderbook.get("store_snapshots", True)),
            "orderbook_history_datadir": orderbook.get("history_datadir", ""),
            "orderbook_history_exchange": orderbook.get("history_exchange", ""),
            "orderbook_history_trading_mode": orderbook.get("history_trading_mode", ""),
            "orderbook_history_category": orderbook.get("history_category", ""),
            "orderbook_history_depth": orderbook.get("history_depth", ""),
            "orderbook_history_timerange": orderbook.get("history_timerange", ""),
            "orderbook_history_feature_timeframes": orderbook.get("history_feature_timeframes", ""),
            "orderbook_history_feature_format": orderbook.get("history_feature_format", ""),
            "orderbook_history_max_rows": orderbook.get("history_max_rows", ""),
            "orderbook_history_erase": bool(orderbook.get("history_erase", False)),
            "orderbook_history_pairs": orderbook.get("history_pairs", ""),
            "data_watchdog_task_name": data_watchdog.get("task_name", ""),
            "data_watchdog_check_interval_minutes": data_watchdog.get("check_interval_minutes", "60"),
            "data_watchdog_heartbeat_stale_minutes": data_watchdog.get("heartbeat_stale_minutes", "10"),
            "data_watchdog_restart_dead": bool(data_watchdog.get("restart_dead", True)),
            "data_watchdog_services": list(data_watchdog.get("services") or []),
            "file_converter_root_paths": list(file_converter.get("root_paths") or []),
            "file_converter_output_extension": file_converter.get("output_extension", ".txt"),
            "file_converter_recursive": bool(file_converter.get("recursive", True)),
            "file_converter_filter": file_converter.get("filter", "All"),
            "file_converter_replace_existing": bool(file_converter.get("replace_existing", True)),
            "file_converter_console": dict(file_converter.get("console") or {}),
        }
        presets[AUTO_PRESET_NAME] = preset
        path = self._preset_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(presets, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    def load_last_state(self) -> None:
        if not self.state_path.exists():
            return
        last_error: Exception | None = None
        for encoding in ("utf-8", "utf-8-sig"):
            try:
                data = json.loads(self.state_path.read_text(encoding=encoding))
                if isinstance(data, dict):
                    self.apply_state(data.get("tabs", data))
                    self.shared.status.set("Loaded last state")
                return
            except Exception as exc:
                last_error = exc
        if last_error:
            self.shared.status.set(f"State load failed: {last_error}")

    def save_last_state(self, reason: str = "manual") -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "reason": reason,
                "tabs": self.collect_state(),
            }
            self.state_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        except Exception as exc:
            self.shared.status.set(f"State save failed: {exc}")

    def _notify_tabs(self, event: str, payload: dict[str, Any]) -> None:
        if event == "save_state":
            self.save_execute_preset(reason=str(payload.get("reason") or "event"))
            return
        for tab in self.tabs.values():
            tab.on_app_event(event, payload)

    def _drain_output_queue(self) -> None:
        lines: list[str] = []
        streams: set[str] = set()
        processed = 0
        char_count = 0
        try:
            while processed < OUTPUT_DRAIN_MAX_LINES and char_count < OUTPUT_DRAIN_MAX_CHARS:
                stream, text = self.output_queue.get_nowait()
                text = str(text)
                lines.append(text)
                streams.add(str(stream))
                processed += 1
                char_count += len(text)
        except queue.Empty:
            pass
        if lines:
            stream_name = next(iter(streams)) if len(streams) == 1 else "mixed"
            self._dispatch_process_output(
                {
                    "stream": stream_name,
                    "text": "".join(lines),
                    "line_count": processed,
                    "owner": self.process_runner.output_owner,
                    "log_file": str(self.process_runner.last_log_file or ""),
                }
            )
        if not self._closing:
            self.after(10 if processed >= OUTPUT_DRAIN_MAX_LINES or char_count >= OUTPUT_DRAIN_MAX_CHARS else 75, self._drain_output_queue)

    def _dispatch_process_output(self, payload: dict[str, Any]) -> None:
        owner = str(payload.get("owner") or "")
        target_keys: list[str] = []
        if owner and owner in self.tabs:
            target_keys.append(owner)
        if owner in {"explorer", "entry_sieve"} and "run" in self.tabs:
            target_keys.append("run")
        if owner == "explorer" and "explorer_summary" in self.tabs:
            target_keys.append("explorer_summary")
        if not target_keys and "run" in self.tabs:
            target_keys.append("run")
        seen: set[str] = set()
        for key in target_keys:
            if key in seen:
                continue
            seen.add(key)
            tab = self.tabs.get(key)
            if tab is not None:
                tab.on_app_event("process_output", payload)

    def _poll_entry_sieve_console_status(self) -> None:
        try:
            active, signature, line = self._entry_sieve_console_line()
            log_text = self._entry_sieve_log_delta() if active and not self.process_runner.is_running() else ""
            if log_text:
                self._dispatch_process_output(
                    {
                        "stream": "entry_sieve_log",
                        "text": log_text,
                        "owner": "entry_sieve",
                        "log_file": "",
                    }
                )
            should_emit = bool(line) and signature != self._entry_sieve_console_signature
            if should_emit and (active or self._entry_sieve_console_had_active):
                self._entry_sieve_console_signature = signature
                self._entry_sieve_console_had_active = active
                self._dispatch_process_output(
                    {
                        "stream": "entry_sieve_status",
                        "text": line,
                        "owner": "entry_sieve",
                        "log_file": "",
                    }
                )
            elif active:
                self._entry_sieve_console_had_active = True
                self._entry_sieve_console_signature = signature
            elif not active:
                self._entry_sieve_console_had_active = False
                self._entry_sieve_console_signature = signature
        except Exception:
            pass
        if not self._closing:
            self.after(ENTRY_SIEVE_STATUS_POLL_MS, self._poll_entry_sieve_console_status)

    def _entry_sieve_log_delta(self) -> str:
        runtime_dir = self.context.app_dir / "launcher_v2" / "runtime" / "entry_sieve"
        queue = self._latest_entry_sieve_queue(runtime_dir)
        status = self._entry_sieve_status(runtime_dir)
        job_id = str(status.get("job_id") or queue.get("current_job_id") or "").strip()
        if not job_id:
            return ""
        explicit_log = str(queue.get("log_file") or "").strip()
        log_path = Path(explicit_log) if explicit_log else runtime_dir / "logs" / f"{job_id}.log"
        if not log_path.exists():
            return ""
        key = str(log_path)
        try:
            size = log_path.stat().st_size
        except OSError:
            return ""
        offset = self._entry_sieve_log_offsets.get(key)
        if offset is None:
            offset = max(0, size - 80_000)
        if size < offset:
            offset = 0
        if size == offset:
            self._entry_sieve_log_offsets[key] = offset
            return ""
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(offset)
                text = handle.read(120_000)
                self._entry_sieve_log_offsets[key] = handle.tell()
        except OSError:
            return ""
        return text

    def _entry_sieve_console_line(self) -> tuple[bool, str, str]:
        runtime_dir = self.context.app_dir / "launcher_v2" / "runtime" / "entry_sieve"
        queue = self._latest_entry_sieve_queue(runtime_dir)
        status = self._entry_sieve_status(runtime_dir)
        queue_status = str(queue.get("status") or "").lower()
        run_status = str(status.get("status") or "").lower()
        active = queue_status in {"pending", "running", "waiting"} or run_status == "running"
        signature_payload = {
            "queue_id": queue.get("queue_id"),
            "queue_status": queue.get("status"),
            "queue_phase": queue.get("phase"),
            "current_batch": queue.get("current_batch"),
            "batch_index": queue.get("batch_index"),
            "batch_total": queue.get("batch_total"),
            "job_id": status.get("job_id"),
            "run_status": status.get("status"),
            "run_phase": status.get("phase"),
            "run_index": status.get("run_index"),
            "completed_hyperopts": status.get("completed_hyperopts"),
            "total_hyperopts": status.get("total_hyperopts"),
            "completed_backtests": status.get("completed_backtests"),
            "total_backtests": status.get("total_backtests"),
            "waiting_backtest_batches": status.get("waiting_backtest_batches"),
            "running_backtest_batches": status.get("running_backtest_batches"),
            "message": status.get("message") or queue.get("message"),
        }
        signature = json.dumps(signature_payload, sort_keys=True, default=str)
        parts = ["Entry Sieve"]
        if queue:
            batch = str(queue.get("current_batch") or "")
            batch_index = queue.get("batch_index")
            batch_total = queue.get("batch_total")
            batch_text = f"batch {batch_index}/{batch_total} {batch}" if batch and batch_index and batch_total else batch
            queue_bits = [str(queue.get("status") or ""), str(queue.get("phase") or "")]
            if batch_text:
                queue_bits.append(batch_text)
            parts.append("queue " + " ".join(bit for bit in queue_bits if bit))
        if status:
            job_id = str(status.get("job_id") or "")
            phase = str(status.get("phase") or "")
            hyper = self._progress_text(status.get("completed_hyperopts"), status.get("total_hyperopts"))
            backtest = self._progress_text(status.get("completed_backtests"), status.get("total_backtests"))
            run_bits = [job_id, str(status.get("status") or ""), phase]
            if hyper:
                run_bits.append(f"hyperopt {hyper}")
            if backtest:
                run_bits.append(f"backtest {backtest}")
            waiting = status.get("waiting_backtest_batches")
            running = status.get("running_backtest_batches")
            if waiting not in ("", None) or running not in ("", None):
                run_bits.append(f"bt_wait/running {waiting or 0}/{running or 0}")
            current_strategy = str(status.get("current_strategy") or "")
            if current_strategy and current_strategy != "final_backtest_drain":
                run_bits.append(current_strategy)
            parts.append("run " + " ".join(bit for bit in run_bits if bit))
        message = str(status.get("message") or queue.get("message") or "")
        if message:
            parts.append(message)
        return active, signature, " | ".join(parts) + "\n"

    def _latest_entry_sieve_queue(self, runtime_dir: Path) -> dict[str, Any]:
        queue_dir = runtime_dir / "queues"
        if not queue_dir.exists():
            return {}
        queues = sorted(queue_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        loaded = [self._read_json_file(path) for path in queues[:10]]
        for queue in loaded:
            if str(queue.get("status") or "").lower() in {"pending", "running", "waiting"}:
                return queue
        return loaded[0] if loaded else {}

    def _entry_sieve_status(self, runtime_dir: Path) -> dict[str, Any]:
        active = self._read_json_file(runtime_dir / "active.json")
        status_file = str(active.get("status_file") or "").strip()
        status_path = Path(status_file) if status_file else Path()
        if status_file and status_path.exists():
            status = self._read_json_file(status_path)
            if status:
                return status
        status_dir = runtime_dir / "status"
        if not status_dir.exists():
            return active
        statuses = sorted(status_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        for path in statuses[:10]:
            status = self._read_json_file(path)
            if str(status.get("status") or "").lower() == "running":
                return status
        return self._read_json_file(statuses[0]) if statuses else active

    @staticmethod
    def _progress_text(done: Any, total: Any) -> str:
        if done in ("", None) and total in ("", None):
            return ""
        return f"{done or 0}/{total or 0}"

    @staticmethod
    def _read_json_file(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _on_close(self) -> None:
        self._closing = True
        if self.process_runner.is_running():
            self.shared.status.set("Stopping Freqtrade process tree")
            self.update_idletasks()
            self.process_runner.stop(timeout_seconds=8.0)
        self._terminate_owned_entry_sieve_processes()
        self.destroy()

    def _terminate_owned_entry_sieve_processes(self) -> None:
        if sys.platform != "win32":
            return
        try:
            completed = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_Process -Filter \"name = 'python.exe'\" | "
                    "Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Depth 3",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
        except Exception:
            return
        try:
            data = json.loads(completed.stdout or "[]")
        except json.JSONDecodeError:
            return
        processes = data if isinstance(data, list) else [data]
        by_pid: dict[int, dict[str, Any]] = {}
        for process in processes:
            if not isinstance(process, dict):
                continue
            try:
                pid = int(process.get("ProcessId") or 0)
            except (TypeError, ValueError):
                continue
            if pid > 0:
                by_pid[pid] = process
        own_pid = os.getpid()
        targets: list[int] = []
        for pid, process in by_pid.items():
            command = str(process.get("CommandLine") or "")
            if "launcher_v2.services.entry_sieve" not in command:
                continue
            if self._process_has_ancestor(pid, own_pid, by_pid):
                targets.append(pid)
        for pid in targets:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
            except Exception:
                continue

    @staticmethod
    def _process_has_ancestor(pid: int, ancestor_pid: int, processes: dict[int, dict[str, Any]]) -> bool:
        seen: set[int] = set()
        current = pid
        while current and current not in seen:
            seen.add(current)
            process = processes.get(current)
            if not process:
                return False
            try:
                parent = int(process.get("ParentProcessId") or 0)
            except (TypeError, ValueError):
                return False
            if parent == ancestor_pid:
                return True
            current = parent
        return False


def main() -> None:
    app = LauncherV2()
    app.mainloop()


if __name__ == "__main__":
    main()

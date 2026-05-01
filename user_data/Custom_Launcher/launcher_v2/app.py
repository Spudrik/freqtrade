from __future__ import annotations

import json
from pathlib import Path
import queue
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
from .tabs.explorer_summary_tab import ExplorerSummaryTab
from .tabs.review_tab import ReviewTab
from .tabs.file_converter_tab import FileConverterTab
from .tabs.indicator_external_validator_tab import IndicatorExternalValidatorTab


APP_TITLE = "Freqtrade Launcher V2"
AUTO_PRESET_NAME = "LauncherV2-auto"
FALLBACK_PRESET_NAME = "BackTest2021-26"
OUTPUT_DRAIN_MAX_LINES = 250
OUTPUT_DRAIN_MAX_CHARS = 120_000


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
        ExplorerSummaryTab,
        ReviewTab,
        FileConverterTab,
        IndicatorExternalValidatorTab,
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
        self._build_ui()
        self.load_execute_preset()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._drain_output_queue)

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
                "pipeline_handoff_dir": preset.get("explorer_pipeline_handoff_dir"),
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
            "file_converter": {
                "root_paths": preset.get("file_converter_root_paths"),
                "output_extension": preset.get("file_converter_output_extension"),
                "recursive": preset.get("file_converter_recursive"),
                "filter": preset.get("file_converter_filter"),
                "replace_existing": preset.get("file_converter_replace_existing"),
                "console": preset.get("file_converter_console"),
            },
            "indicator_external_validator": {
                "datadir": preset.get("indicator_validator_datadir"),
                "pairs": preset.get("indicator_validator_pairs"),
                "timeframes": preset.get("indicator_validator_timeframes"),
                "timerange": preset.get("indicator_validator_timerange"),
                "indicators": preset.get("indicator_validator_indicators"),
                "score_scope": preset.get("indicator_validator_score_scope"),
                "benchmark": preset.get("indicator_validator_benchmark"),
                "forward_windows": preset.get("indicator_validator_forward_windows"),
                "deciles": preset.get("indicator_validator_deciles"),
                "score_threshold": preset.get("indicator_validator_score_threshold"),
                "max_pairs": preset.get("indicator_validator_max_pairs"),
                "min_rows": preset.get("indicator_validator_min_rows"),
                "output_dir": preset.get("indicator_validator_output_dir"),
                "profile_window": preset.get("indicator_validator_profile_window"),
                "profile_bins": preset.get("indicator_validator_profile_bins"),
                "profile_chunk_size": preset.get("indicator_validator_profile_chunk_size"),
                "quiet": preset.get("indicator_validator_quiet"),
                "console": preset.get("indicator_validator_console"),
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
        news = tabs.get("news", {})
        web = tabs.get("web", {})
        global_context = tabs.get("global_context", {})
        orderbook = tabs.get("orderbook", {})
        file_converter = tabs.get("file_converter", {})
        indicator_validator = tabs.get("indicator_external_validator", {})
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
            "explorer_pipeline_handoff_dir": explorer.get("pipeline_handoff_dir", ""),
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
            "file_converter_root_paths": list(file_converter.get("root_paths") or []),
            "file_converter_output_extension": file_converter.get("output_extension", ".txt"),
            "file_converter_recursive": bool(file_converter.get("recursive", True)),
            "file_converter_filter": file_converter.get("filter", "All"),
            "file_converter_replace_existing": bool(file_converter.get("replace_existing", True)),
            "file_converter_console": dict(file_converter.get("console") or {}),
            "indicator_validator_datadir": indicator_validator.get("datadir", ""),
            "indicator_validator_pairs": indicator_validator.get("pairs", ""),
            "indicator_validator_timeframes": indicator_validator.get("timeframes", "1h"),
            "indicator_validator_timerange": indicator_validator.get("timerange", ""),
            "indicator_validator_indicators": indicator_validator.get("indicators", "all"),
            "indicator_validator_score_scope": indicator_validator.get("score_scope", "base"),
            "indicator_validator_benchmark": indicator_validator.get("benchmark", "BTC/USDT:USDT"),
            "indicator_validator_forward_windows": indicator_validator.get("forward_windows", "3 6 12 24"),
            "indicator_validator_deciles": indicator_validator.get("deciles", "10"),
            "indicator_validator_score_threshold": indicator_validator.get("score_threshold", "0.70"),
            "indicator_validator_max_pairs": indicator_validator.get("max_pairs", "20"),
            "indicator_validator_min_rows": indicator_validator.get("min_rows", "250"),
            "indicator_validator_output_dir": indicator_validator.get("output_dir", ""),
            "indicator_validator_profile_window": indicator_validator.get("profile_window", "96"),
            "indicator_validator_profile_bins": indicator_validator.get("profile_bins", "48"),
            "indicator_validator_profile_chunk_size": indicator_validator.get("profile_chunk_size", "512"),
            "indicator_validator_quiet": bool(indicator_validator.get("quiet", False)),
            "indicator_validator_console": dict(indicator_validator.get("console") or {}),
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
        if owner == "explorer" and "run" in self.tabs:
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

    def _on_close(self) -> None:
        self._closing = True
        if self.process_runner.is_running():
            self.shared.status.set("Stopping Freqtrade process tree")
            self.update_idletasks()
            self.process_runner.stop(timeout_seconds=8.0)
        self.destroy()


def main() -> None:
    app = LauncherV2()
    app.mainloop()


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import re
import shlex
from typing import Any
import tkinter as tk
from tkinter import messagebox, ttk

from ..base_tab import BaseTab
from ..command_builder import command_text, freqtrade_command
from ..console_pane import ConsolePane
from ..services.collector_service import is_process_running
from .pairs_tab import parse_pairs
from ..ui_helpers import labeled_entry


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class RunTab(BaseTab):
    tab_key = "run"
    tab_title = "Run"

    def __init__(self, master: tk.Misc, context: Any) -> None:
        super().__init__(master, context)
        self.run_type_var = tk.StringVar(value="Backtest")
        self.extra_args_var = tk.StringVar(value="")
        self.raw_console: ConsolePane | None = None
        self.results_console: ConsolePane | None = None
        self._result_partial_line = ""
        self._result_context_lines = 0
        self._build_ui()
        self._refresh_preview()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        controls = ttk.LabelFrame(self, text="Session")
        controls.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        controls.grid_columnconfigure(1, weight=1)
        ttk.Label(controls, text="Run type").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        combo = ttk.Combobox(controls, textvariable=self.run_type_var, values=["Backtest", "Hyperopt", "Dry-run", "Live", "Download Data"], state="readonly")
        combo.grid(row=0, column=1, sticky="ew", padx=8, pady=6)
        combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_preview())
        labeled_entry(controls, 1, 0, "Extra args", self.extra_args_var)
        self.extra_args_var.trace_add("write", lambda *_: self._refresh_preview())

        buttons = ttk.Frame(self)
        buttons.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(buttons, text="Refresh preview", command=self._refresh_preview).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Run", command=self._run).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Launch FreqUI", command=self.run_frequi).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Launch/Refresh Data Tools", command=self.ensure_data_tools_running).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Launch Watchdog", command=self.launch_watchdog).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Stop", command=self._stop).pack(side="left", padx=(0, 6))

        preview = ttk.LabelFrame(self, text="Generated command")
        preview.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        preview.grid_columnconfigure(0, weight=1)
        ttk.Entry(preview, textvariable=self.context.shared.command_preview).grid(row=0, column=0, sticky="ew", padx=8, pady=8)

        console_frame = ttk.LabelFrame(self, text="Console")
        console_frame.grid(row=3, column=0, sticky="nsew", padx=8, pady=(0, 8))
        console_frame.grid_columnconfigure(0, weight=1)
        console_frame.grid_rowconfigure(0, weight=1)
        console_tabs = ttk.Notebook(console_frame)
        console_tabs.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.raw_console = ConsolePane(console_tabs)
        self.results_console = ConsolePane(console_tabs)
        console_tabs.add(self.raw_console, text="Raw Console")
        console_tabs.add(self.results_console, text="Results Console")

    def _build_freqtrade_args(self, run_type_override: str | None = None) -> list[str]:
        command_map = {
            "Backtest": "backtesting",
            "Hyperopt": "hyperopt",
            "Dry-run": "trade",
            "Live": "trade",
            "Download Data": "download-data",
        }
        run_type = run_type_override or self.run_type_var.get()
        command = command_map.get(run_type, "backtesting")
        args = [command]
        self._append_setup_args(args, command)
        self._append_mode_args(args, command)
        self._append_download_args(args, command)
        self._append_pair_args(args, command)
        extra = self.extra_args_var.get().strip()
        if extra:
            args.extend(shlex.split(extra))
        return args

    def _tab_state(self, key: str) -> dict[str, Any]:
        tab = self.context.registry.get(key)
        if tab is None:
            return {}
        try:
            state = tab.get_state()
        except Exception:
            return {}
        return state if isinstance(state, dict) else {}

    def _append_setup_args(self, args: list[str], command: str) -> None:
        common = self._tab_state("common")
        for config_path in common.get("config_files") or []:
            if str(config_path).strip():
                args.extend(["-c", str(config_path).strip()])

        userdir = str(common.get("userdir") or self.context.shared.userdir.get() or "").strip()
        datadir = str(common.get("datadir") or self.context.shared.datadir.get() or "").strip()
        strategy_file = str(common.get("strategy_file") or "").strip()
        strategy_class = str(common.get("strategy_class") or "").strip()
        if userdir:
            args.extend(["--userdir", userdir])
        if datadir:
            args.extend(["--datadir", datadir])
        if command != "download-data":
            if strategy_file:
                strategy_dir = Path(strategy_file).expanduser().resolve().parent
                default_strategy_dir = (Path(userdir).expanduser().resolve() / "strategies") if userdir else None
                if default_strategy_dir is None or strategy_dir != default_strategy_dir:
                    args.extend(["--strategy-path", str(strategy_dir)])
            if strategy_class:
                args.extend(["--strategy", strategy_class])
            if common.get("recursive_strategy_search"):
                args.append("--recursive-strategy-search")

    def _append_mode_args(self, args: list[str], command: str) -> None:
        if command == "download-data":
            return
        mode = self._tab_state("mode_options")
        shared_switches = [
            ("timeframe", "-i"),
            ("timerange", "--timerange"),
            ("max_open_trades", "--max-open-trades"),
            ("stake_amount", "--stake-amount"),
            ("fee", "--fee"),
        ]
        for key, switch in shared_switches:
            value = str(mode.get(key) or "").strip()
            if value:
                args.extend([switch, value])

        if command == "backtesting":
            for key, switch in (("export", "--export"), ("breakdown", "--breakdown")):
                value = str(mode.get(key) or "").strip()
                if value:
                    args.extend([switch, value])
        elif command == "hyperopt":
            for key, switch in (
                ("epochs", "--epochs"),
                ("spaces", "--spaces"),
                ("jobs", "--job-workers"),
                ("random_state", "--random-state"),
                ("hyperopt_loss", "--hyperopt-loss"),
            ):
                value = str(mode.get(key) or "").strip()
                if not value:
                    continue
                if key == "spaces":
                    args.append(switch)
                    args.extend(value.replace(",", " ").split())
                else:
                    args.extend([switch, value])
            if mode.get("ignore_missing_spaces"):
                args.append("--ignore-missing-spaces")
            if mode.get("disable_param_export"):
                args.append("--disable-param-export")
        elif command == "trade":
            dry_wallet = str(mode.get("dry_run_wallet") or "").strip()
            if dry_wallet:
                args.extend(["--dry-run-wallet", dry_wallet])

    def _append_download_args(self, args: list[str], command: str) -> None:
        if command != "download-data":
            return
        download = self._tab_state("download")
        scalar_switches = [
            ("exchange", "--exchange"),
            ("pairs_file", "--pairs-file"),
            ("days", "--days"),
            ("new_pairs_days", "--new-pairs-days"),
            ("timerange", "--timerange"),
            ("trading_mode", "--trading-mode"),
            ("data_format_ohlcv", "--data-format-ohlcv"),
            ("data_format_trades", "--data-format-trades"),
        ]
        for key, switch in scalar_switches:
            value = str(download.get(key) or "").strip()
            if value:
                args.extend([switch, value])
        timeframes = str(download.get("timeframes") or "").replace(",", " ").split()
        if timeframes:
            args.append("-t")
            args.extend(timeframes)
        candle_types = str(download.get("candle_types") or "").replace(",", " ").split()
        if candle_types:
            args.append("--candle-types")
            args.extend(candle_types)
        for key, switch in (
            ("include_inactive", "--include-inactive-pairs"),
            ("no_parallel", "--no-parallel-download"),
            ("dl_trades", "--dl-trades"),
            ("convert", "--convert"),
            ("erase", "--erase"),
            ("prepend", "--prepend"),
        ):
            if download.get(key):
                args.append(switch)

    def _append_pair_args(self, args: list[str], command: str) -> None:
        pairs = self._tab_state("pairs")
        if str(pairs.get("pair_mode") or "manual") != "manual":
            return
        if command not in {"backtesting", "hyperopt", "download-data"}:
            return
        if command == "download-data":
            download_pairs = parse_pairs(str(self._tab_state("download").get("pairs") or ""))
            parsed = download_pairs or parse_pairs(str(pairs.get("pairs") or ""))
        else:
            parsed = parse_pairs(str(pairs.get("pairs") or ""))
        if parsed:
            args.append("-p")
            args.extend(parsed)

    def _refresh_preview(self) -> None:
        result = freqtrade_command(self.context.shared.python_exe.get(), self._build_freqtrade_args())
        self.context.shared.command_preview.set(command_text(result.preview_command))

    def _run(self) -> None:
        result = freqtrade_command(self.context.shared.python_exe.get(), self._build_freqtrade_args())
        self._refresh_preview()
        self.context.emit("save_state", {"reason": "run"})
        self.context.process_runner.run(result.preview_command, cwd=self.context.shared.project_root.get() or None, owner=self.tab_key)
        self.context.shared.status.set("Process running")

    def run_download(self) -> None:
        result = freqtrade_command(self.context.shared.python_exe.get(), self._build_freqtrade_args("Download Data"))
        self.context.shared.command_preview.set(command_text(result.preview_command))
        self.context.emit("save_state", {"reason": "download_data"})
        self.context.process_runner.run(result.preview_command, cwd=self.context.shared.project_root.get() or None, owner=self.tab_key)
        self.context.shared.status.set("Download running")

    def _build_frequi_args(self) -> list[str]:
        common = self._tab_state("common")
        args = ["webserver"]
        for config_path in common.get("config_files") or []:
            if str(config_path).strip():
                args.extend(["-c", str(config_path).strip()])
        userdir = str(common.get("userdir") or self.context.shared.userdir.get() or "").strip()
        datadir = str(common.get("datadir") or self.context.shared.datadir.get() or "").strip()
        if userdir:
            args.extend(["--userdir", userdir])
        if datadir:
            args.extend(["--datadir", datadir])
        return args

    def run_frequi(self) -> None:
        result = freqtrade_command(self.context.shared.python_exe.get(), self._build_frequi_args())
        self.context.shared.command_preview.set(command_text(result.preview_command))
        self.context.emit("save_state", {"reason": "frequi_launch"})
        self.context.process_runner.run(result.preview_command, cwd=self.context.shared.project_root.get() or None, owner=self.tab_key)
        self.context.shared.status.set("FreqUI running")

    def ensure_data_tools_running(self) -> None:
        results = [
            self._ensure_research_collector("News", "news"),
            self._ensure_research_collector("Web", "web"),
            self._ensure_global_context_collector(),
            self._ensure_orderbook_collector(),
        ]
        ok = [message for ok, message in results if ok]
        errors = [message for ok, message in results if not ok]
        summary = " | ".join(ok + errors)
        self.context.shared.status.set(f"Data tools: {summary}")
        if self.raw_console is not None:
            self.raw_console.append(f"Data tools ensure-running: {summary}\n")
        self.context.emit("save_state", {"reason": "data_tools_ensure_running"})
        if errors:
            messagebox.showwarning("Data tools", "Some data tools could not be started:\n\n" + "\n".join(errors), parent=self)

    def launch_watchdog(self) -> None:
        tab = self.context.registry.get("data_watchdog")
        if tab is None:
            messagebox.showerror("Data watchdog", "Watchdog tab is not loaded.", parent=self)
            return
        try:
            tab.install_task()
        except Exception as exc:
            messagebox.showerror("Data watchdog", f"Could not launch watchdog:\n{exc}", parent=self)
            return
        self.context.shared.status.set("Data watchdog launched/updated")
        if self.raw_console is not None:
            self.raw_console.append("Data watchdog launched/updated.\n")

    def _ensure_research_collector(self, label: str, tab_key: str) -> tuple[bool, str]:
        tab = self.context.registry.get(tab_key)
        if tab is None:
            return False, f"{label}: tab not loaded"
        try:
            state = dict(tab.get_state())
            state["once"] = False
            status = tab.service.read_status(tab.profile, state)
            pid = self._running_pid(status)
            if pid is None:
                pid = int(tab.service.start_detached(tab.profile, state))
            self._refresh_tab_status(tab)
            return True, f"{label}: PID {pid}"
        except Exception as exc:
            self._refresh_tab_status(tab)
            return False, f"{label}: {exc}"

    def _ensure_global_context_collector(self) -> tuple[bool, str]:
        tab = self.context.registry.get("global_context")
        if tab is None:
            return False, "Global: tab not loaded"
        try:
            state = dict(tab.get_state())
            state["once"] = False
            status = tab.service.read_status(state)
            pid = self._running_pid(status)
            if pid is None:
                pid = int(tab.service.start_detached(state))
            self._refresh_tab_status(tab)
            return True, f"Global: PID {pid}"
        except Exception as exc:
            self._refresh_tab_status(tab)
            return False, f"Global: {exc}"

    def _ensure_orderbook_collector(self) -> tuple[bool, str]:
        tab = self.context.registry.get("orderbook")
        if tab is None:
            return False, "Order book: tab not loaded"
        try:
            state = tab.get_state()
            status = tab.service.read_status(state)
            pid = self._running_pid(status)
            if pid is None:
                pid = int(tab.service.start_detached(state, self._main_pairs()))
            self._refresh_tab_status(tab)
            return True, f"Order book: PID {pid}"
        except Exception as exc:
            self._refresh_tab_status(tab)
            return False, f"Order book: {exc}"

    def _main_pairs(self) -> list[str]:
        return parse_pairs(str(self._tab_state("pairs").get("pairs") or ""))

    @staticmethod
    def _running_pid(status: dict[str, Any]) -> int | None:
        for key in ("pid_text", "pid"):
            try:
                pid = int(str(status.get(key) or "").strip())
            except (TypeError, ValueError):
                continue
            if is_process_running(pid):
                return pid
        return None

    @staticmethod
    def _refresh_tab_status(tab: Any) -> None:
        try:
            tab.refresh_status()
        except Exception:
            return

    def _stop(self) -> None:
        self.context.process_runner.stop()
        self.context.shared.status.set("Stop requested")

    def _append_console(self, text: str) -> None:
        if self.raw_console is not None:
            self.raw_console.append(text)
        filtered = self._filter_result_text(text)
        if filtered and self.results_console is not None:
            self.results_console.append(filtered)

    def _filter_result_text(self, text: str) -> str:
        if not text:
            return ""
        combined = f"{self._result_partial_line}{text}"
        lines = combined.splitlines(keepends=True)
        self._result_partial_line = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._result_partial_line = lines.pop()
        included: list[str] = []
        for line in lines:
            if self._include_result_line(line):
                included.append(ANSI_ESCAPE_RE.sub("", line))
        return "".join(included)

    def _include_result_line(self, line: str) -> bool:
        stripped = ANSI_ESCAPE_RE.sub("", line).strip()
        lowered = stripped.lower()
        if not stripped:
            if self._result_context_lines > 0:
                self._result_context_lines -= 1
                return True
            return False
        if self._is_hyperopt_epoch_line(lowered):
            self._result_context_lines = 0
            return False
        if "epoch" in lowered and not any(marker in lowered for marker in ("best result", "best loss", "hyperopt result")):
            return False
        if self._is_result_table_line(stripped):
            self._result_context_lines = 0
            return True
        if self._is_hyperopt_result_header(lowered):
            self._result_context_lines = 30
            return True
        if self._is_result_header_line(lowered):
            self._result_context_lines = 0
            return True
        if self._result_context_lines > 0:
            if self._is_result_noise_line(lowered):
                self._result_context_lines = 0
                return False
            self._result_context_lines -= 1
            return True
        return False

    def _is_result_table_line(self, stripped: str) -> bool:
        if not stripped:
            return False
        first = stripped[0]
        if first in {"|", "+"}:
            return True
        if 0x2500 <= ord(first) <= 0x257F:
            return True
        border_chars = set("-+|= ")
        return bool(stripped) and set(stripped) <= border_chars and ("|" in stripped or "+" in stripped)

    def _is_hyperopt_epoch_line(self, lowered: str) -> bool:
        if lowered == "hyperopt results":
            return True
        if "|" not in lowered:
            return False
        if "/" not in lowered:
            return False
        return "best" in lowered or "current" in lowered or "epoch" in lowered

    def _is_hyperopt_result_header(self, lowered: str) -> bool:
        result_markers = (
            "best result",
            "best loss",
            "objective:",
        )
        return any(marker in lowered for marker in result_markers)

    def _is_result_header_line(self, lowered: str) -> bool:
        result_markers = (
            "backtesting report",
            "backtested ",
            "enter tag stats",
            "exit reason stats",
            "mixed tag stats",
            "left open trades report",
            "summary metrics",
            "strategy summary",
            "result for strategy",
            "process exited with code",
        )
        return any(marker in lowered for marker in result_markers)

    def _is_result_noise_line(self, lowered: str) -> bool:
        noise_markers = (
            "starting challenger validation backtest",
            "-m freqtrade ",
            " freqtrade ",
            "process running",
        )
        return any(marker in lowered for marker in noise_markers)

    def on_app_event(self, event: str, payload: dict[str, Any]) -> None:
        if event == "process_output":
            self._append_console(str(payload.get("text") or ""))

    def get_state(self) -> dict[str, Any]:
        raw_state = self.raw_console.get_state() if self.raw_console is not None else {}
        result_state = self.results_console.get_state() if self.results_console is not None else {}
        return {
            "run_type": self.run_type_var.get(),
            "extra_args": self.extra_args_var.get(),
            "raw_console": raw_state,
            "results_console": result_state,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.run_type_var.set(str(state.get("run_type") or "Backtest"))
        self.extra_args_var.set(str(state.get("extra_args") or ""))
        if self.raw_console is not None:
            raw_state = state.get("raw_console")
            if not isinstance(raw_state, dict):
                raw_state = {
                    "search_term": state.get("search_term", ""),
                    "search_terms": state.get("search_terms", []),
                    "follow_tail": state.get("follow_tail", True),
                }
            self.raw_console.set_state(raw_state)
        if self.results_console is not None:
            result_state = state.get("results_console")
            if not isinstance(result_state, dict):
                result_state = {
                    "search_term": state.get("result_search_term", ""),
                    "search_terms": state.get("result_search_terms", []),
                    "follow_tail": state.get("result_follow_tail", True),
                }
            self.results_console.set_state(result_state)
        self._refresh_preview()

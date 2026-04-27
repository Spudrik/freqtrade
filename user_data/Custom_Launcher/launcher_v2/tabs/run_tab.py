from __future__ import annotations

from pathlib import Path
import re
import shlex
from typing import Any
import tkinter as tk
from tkinter import ttk

from ..base_tab import BaseTab
from ..command_builder import command_text, freqtrade_command
from .pairs_tab import parse_pairs
from ..ui_helpers import labeled_entry


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
MAX_CONSOLE_LINES = 5000


class ConsolePane(ttk.Frame):
    def __init__(self, master: tk.Misc, *, label: str = "Search") -> None:
        super().__init__(master)
        self.search_term_var = tk.StringVar(value="")
        self.search_status_var = tk.StringVar(value="")
        self.follow_tail_var = tk.BooleanVar(value=True)
        self.search_terms: list[str] = []
        self.search_matches: list[tuple[str, str]] = []
        self.active_match_index = -1
        self._last_search_term = ""

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        tools = ttk.Frame(self)
        tools.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 4))
        ttk.Label(tools, text=label).pack(side="left")
        self.search_combo = ttk.Combobox(tools, textvariable=self.search_term_var, values=self.search_terms, width=34)
        self.search_combo.pack(side="left", padx=(6, 6))
        self.search_combo.bind("<Return>", lambda _event: self.find_next(reset=True))
        ttk.Button(tools, text="Find", command=lambda: self.find_next(reset=True)).pack(side="left", padx=(0, 6))
        ttk.Button(tools, text="Next", command=self.find_next).pack(side="left", padx=(0, 6))
        ttk.Button(tools, text="Bottom", command=self.scroll_bottom).pack(side="left", padx=(0, 6))
        ttk.Checkbutton(tools, text="Follow newest", variable=self.follow_tail_var).pack(side="left", padx=(0, 6))
        ttk.Label(tools, textvariable=self.search_status_var).pack(side="left", padx=(6, 0))

        text_frame = ttk.Frame(self)
        text_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        text_frame.grid_columnconfigure(0, weight=1)
        text_frame.grid_rowconfigure(0, weight=1)
        self.text = tk.Text(text_frame, wrap="none", font=("Consolas", 9), undo=False)
        yscroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.text.yview)
        xscroll = ttk.Scrollbar(text_frame, orient="horizontal", command=self.text.xview)
        self.text.configure(
            yscrollcommand=yscroll.set,
            xscrollcommand=xscroll.set,
            background="#101827",
            foreground="#d8e2f0",
            insertbackground="#d8e2f0",
            relief="flat",
        )
        self.text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self.text.tag_configure("search_match", background="#fff2a8", foreground="#000000")
        self.text.tag_configure("search_active", background="#f6a23a", foreground="#000000")

    def at_bottom(self) -> bool:
        try:
            return self.text.yview()[1] >= 0.999
        except Exception:
            return True

    def append(self, text: str) -> None:
        follow = self.follow_tail_var.get() and self.at_bottom()
        self.text.insert(tk.END, text)
        self._trim_buffer()
        if follow:
            self.text.see(tk.END)

    def _trim_buffer(self) -> None:
        try:
            line_count = int(self.text.index("end-1c").split(".", 1)[0])
        except Exception:
            return
        extra = line_count - MAX_CONSOLE_LINES
        if extra > 0:
            self.text.delete("1.0", f"{extra + 1}.0")

    def scroll_bottom(self) -> None:
        self.text.see(tk.END)

    def remember_search_term(self, term: str) -> None:
        normalized = term.strip()
        if not normalized:
            return
        lowered = normalized.lower()
        self.search_terms = [item for item in self.search_terms if item.lower() != lowered]
        self.search_terms.insert(0, normalized)
        self.search_terms = self.search_terms[:5]
        self.search_combo.configure(values=self.search_terms)

    def refresh_search_matches(self) -> None:
        term = self.search_term_var.get().strip()
        self.text.tag_remove("search_match", "1.0", tk.END)
        self.text.tag_remove("search_active", "1.0", tk.END)
        self.search_matches = []
        self.active_match_index = -1
        self._last_search_term = term
        if not term:
            self.search_status_var.set("")
            return
        start = "1.0"
        count = tk.IntVar()
        while True:
            index = self.text.search(term, start, stopindex=tk.END, nocase=True, count=count)
            if not index or count.get() <= 0:
                break
            end = f"{index}+{count.get()}c"
            self.search_matches.append((index, end))
            self.text.tag_add("search_match", index, end)
            start = end
        self.search_status_var.set(f"{len(self.search_matches)} matches")

    def set_active_match(self, index: int) -> None:
        if not self.search_matches:
            self.search_status_var.set("No matches")
            return
        self.text.tag_remove("search_active", "1.0", tk.END)
        self.active_match_index = index % len(self.search_matches)
        start, end = self.search_matches[self.active_match_index]
        self.text.tag_add("search_active", start, end)
        self.text.see(start)
        self.search_status_var.set(f"{self.active_match_index + 1}/{len(self.search_matches)}")

    def find_next(self, reset: bool = False) -> None:
        term = self.search_term_var.get().strip()
        if not term:
            self.search_status_var.set("Type search text")
            return
        self.remember_search_term(term)
        if reset or self._last_search_term != term or not self.search_matches:
            self.refresh_search_matches()
            if not self.search_matches:
                return
            self.set_active_match(0)
            return
        self.set_active_match(self.active_match_index + 1)

    def get_state(self) -> dict[str, Any]:
        return {
            "search_term": self.search_term_var.get(),
            "search_terms": list(self.search_terms),
            "follow_tail": self.follow_tail_var.get(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.search_term_var.set(str(state.get("search_term") or ""))
        terms = state.get("search_terms") or []
        if isinstance(terms, list):
            self.search_terms = [str(item).strip() for item in terms if str(item).strip()][:5]
        self.search_combo.configure(values=self.search_terms)
        self.follow_tail_var.set(bool(state.get("follow_tail", True)))


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
            "search_term": raw_state.get("search_term", ""),
            "search_terms": raw_state.get("search_terms", []),
            "follow_tail": raw_state.get("follow_tail", True),
            "result_search_term": result_state.get("search_term", ""),
            "result_search_terms": result_state.get("search_terms", []),
            "result_follow_tail": result_state.get("follow_tail", True),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.run_type_var.set(str(state.get("run_type") or "Backtest"))
        self.extra_args_var.set(str(state.get("extra_args") or ""))
        if self.raw_console is not None:
            self.raw_console.set_state(
                {
                    "search_term": state.get("search_term", ""),
                    "search_terms": state.get("search_terms", []),
                    "follow_tail": state.get("follow_tail", True),
                }
            )
        if self.results_console is not None:
            self.results_console.set_state(
                {
                    "search_term": state.get("result_search_term", ""),
                    "search_terms": state.get("result_search_terms", []),
                    "follow_tail": state.get("result_follow_tail", True),
                }
            )
        self._refresh_preview()

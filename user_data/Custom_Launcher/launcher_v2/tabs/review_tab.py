from __future__ import annotations

from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk

from ..base_tab import BaseTab
from ..command_builder import command_text, freqtrade_command
from ..ui_helpers import append_bounded_text, labeled_entry


BREAKDOWN_VALUES = ["none", "day", "week", "month", "year"]


def _tokens(value: str) -> list[str]:
    return str(value or "").replace(",", " ").split()


class ReviewTab(BaseTab):
    tab_key = "review"
    tab_title = "Review Results"

    def __init__(self, master: tk.Misc, context: Any) -> None:
        super().__init__(master, context)
        self.hyperopt_file_var = tk.StringVar(value="")
        self.hyperopt_limit_var = tk.StringVar(value="20")
        self.hyperopt_index_var = tk.StringVar(value="-1")
        self.hyperopt_best_var = tk.BooleanVar(value=False)
        self.hyperopt_profitable_var = tk.BooleanVar(value=False)
        self.hyperopt_print_json_var = tk.BooleanVar(value=False)
        self.hyperopt_no_header_var = tk.BooleanVar(value=False)
        self.hyperopt_no_details_var = tk.BooleanVar(value=False)
        self.hyperopt_breakdown_var = tk.StringVar(value="none")
        self.backtest_file_var = tk.StringVar(value="")
        self.backtest_directory_var = tk.StringVar(value="")
        self.backtest_show_pair_list_var = tk.BooleanVar(value=False)
        self.backtest_breakdown_var = tk.StringVar(value="none")
        self.analysis_groups_var = tk.StringVar(value="0 1 2 5")
        self.enter_reasons_var = tk.StringVar(value="")
        self.exit_reasons_var = tk.StringVar(value="")
        self.indicator_list_var = tk.StringVar(value="")
        self.entry_only_var = tk.BooleanVar(value=False)
        self.exit_only_var = tk.BooleanVar(value=False)
        self.rejected_signals_var = tk.BooleanVar(value=False)
        self.analysis_to_csv_var = tk.BooleanVar(value=False)
        self.analysis_csv_path_var = tk.StringVar(value="")
        self._build_ui()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        top.grid_columnconfigure(0, weight=1)
        top.grid_columnconfigure(1, weight=1)

        hyper = ttk.LabelFrame(top, text="Hyperopt")
        hyper.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        hyper.grid_columnconfigure(1, weight=1)
        self._path_row(hyper, 0, "Hyperopt file", self.hyperopt_file_var, [("Hyperopt files", "*.fthypt *.json"), ("All files", "*.*")])
        labeled_entry(hyper, 1, 0, "Limit", self.hyperopt_limit_var)
        labeled_entry(hyper, 1, 2, "Epoch index", self.hyperopt_index_var)
        ttk.Checkbutton(hyper, text="Best", variable=self.hyperopt_best_var).grid(row=2, column=0, sticky="w", padx=8, pady=4)
        ttk.Checkbutton(hyper, text="Profitable", variable=self.hyperopt_profitable_var).grid(row=2, column=1, sticky="w", padx=8, pady=4)
        ttk.Checkbutton(hyper, text="Print JSON", variable=self.hyperopt_print_json_var).grid(row=2, column=2, sticky="w", padx=8, pady=4)
        ttk.Checkbutton(hyper, text="No header", variable=self.hyperopt_no_header_var).grid(row=3, column=0, sticky="w", padx=8, pady=4)
        ttk.Checkbutton(hyper, text="No details", variable=self.hyperopt_no_details_var).grid(row=3, column=1, sticky="w", padx=8, pady=4)
        ttk.Label(hyper, text="Breakdown").grid(row=4, column=0, sticky="w", padx=8, pady=4)
        ttk.Combobox(hyper, textvariable=self.hyperopt_breakdown_var, values=BREAKDOWN_VALUES, state="readonly").grid(row=4, column=1, sticky="ew", padx=8, pady=4)
        buttons = ttk.Frame(hyper)
        buttons.grid(row=5, column=0, columnspan=4, sticky="ew", padx=8, pady=8)
        ttk.Button(buttons, text="Hyperopt List", command=self._run_hyperopt_list).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Hyperopt Show", command=self._run_hyperopt_show).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Show Best", command=self._run_hyperopt_show_best).pack(side="left")

        back = ttk.LabelFrame(top, text="Backtest")
        back.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        back.grid_columnconfigure(1, weight=1)
        self._path_row(back, 0, "Backtest file", self.backtest_file_var, [("Backtest zips", "*.zip"), ("All files", "*.*")])
        self._directory_row(back, 1, "Backtest directory", self.backtest_directory_var)
        ttk.Checkbutton(back, text="Show pair list", variable=self.backtest_show_pair_list_var).grid(row=2, column=0, sticky="w", padx=8, pady=4)
        ttk.Label(back, text="Breakdown").grid(row=2, column=2, sticky="w", padx=8, pady=4)
        ttk.Combobox(back, textvariable=self.backtest_breakdown_var, values=BREAKDOWN_VALUES, state="readonly").grid(row=2, column=3, sticky="ew", padx=8, pady=4)
        labeled_entry(back, 3, 0, "Analysis groups", self.analysis_groups_var)
        labeled_entry(back, 3, 2, "Indicators", self.indicator_list_var)
        labeled_entry(back, 4, 0, "Entry reasons", self.enter_reasons_var)
        labeled_entry(back, 4, 2, "Exit reasons", self.exit_reasons_var)
        ttk.Checkbutton(back, text="Entry only", variable=self.entry_only_var).grid(row=5, column=0, sticky="w", padx=8, pady=4)
        ttk.Checkbutton(back, text="Exit only", variable=self.exit_only_var).grid(row=5, column=1, sticky="w", padx=8, pady=4)
        ttk.Checkbutton(back, text="Rejected signals", variable=self.rejected_signals_var).grid(row=5, column=2, sticky="w", padx=8, pady=4)
        ttk.Checkbutton(back, text="Analysis to CSV", variable=self.analysis_to_csv_var).grid(row=6, column=0, sticky="w", padx=8, pady=4)
        ttk.Label(back, text="CSV path").grid(row=6, column=1, sticky="w", padx=8, pady=4)
        ttk.Entry(back, textvariable=self.analysis_csv_path_var).grid(row=6, column=2, sticky="ew", padx=8, pady=4)
        ttk.Button(back, text="Browse", command=lambda: self._browse_save(self.analysis_csv_path_var, [("CSV files", "*.csv"), ("All files", "*.*")])).grid(row=6, column=3, padx=8, pady=4)
        buttons = ttk.Frame(back)
        buttons.grid(row=7, column=0, columnspan=4, sticky="ew", padx=8, pady=8)
        ttk.Button(buttons, text="Backtesting Show", command=self._run_backtest_show).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Backtesting Analysis", command=self._run_backtest_analysis).pack(side="left")

        preview = ttk.LabelFrame(self, text="Last command")
        preview.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        preview.grid_columnconfigure(0, weight=1)
        self.preview_var = tk.StringVar(value="")
        ttk.Entry(preview, textvariable=self.preview_var).grid(row=0, column=0, sticky="ew", padx=8, pady=8)

        console = ttk.LabelFrame(self, text="Review output")
        console.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        console.grid_columnconfigure(0, weight=1)
        console.grid_rowconfigure(0, weight=1)
        self.console = scrolledtext.ScrolledText(console, wrap="word")
        self.console.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    def _path_row(self, parent: tk.Misc, row: int, label: str, variable: tk.StringVar, filetypes: list[tuple[str, str]], *, save: bool = False) -> None:
        labeled_entry(parent, row, 0, label, variable)
        if save:
            action = lambda: self._browse_save(variable, filetypes)
        else:
            action = lambda: self._browse_open(variable, filetypes)
        ttk.Button(parent, text="Browse", command=action).grid(row=row, column=2, padx=8, pady=4)

    def _directory_row(self, parent: tk.Misc, row: int, label: str, variable: tk.StringVar) -> None:
        labeled_entry(parent, row, 0, label, variable)
        ttk.Button(parent, text="Browse", command=lambda: self._browse_dir(variable)).grid(row=row, column=2, padx=8, pady=4)

    def _browse_open(self, variable: tk.StringVar, filetypes: list[tuple[str, str]]) -> None:
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            variable.set(path)

    def _browse_save(self, variable: tk.StringVar, filetypes: list[tuple[str, str]]) -> None:
        path = filedialog.asksaveasfilename(filetypes=filetypes)
        if path:
            variable.set(path)

    def _browse_dir(self, variable: tk.StringVar) -> None:
        path = filedialog.askdirectory()
        if path:
            variable.set(path)

    def _common_args(self) -> list[str]:
        common = self.context.registry.get("common")
        state = common.get_state() if common is not None else {}
        args: list[str] = []
        for config in state.get("config_files") or []:
            if str(config).strip():
                args.extend(["-c", str(config).strip()])
        if str(state.get("datadir") or "").strip():
            args.extend(["--datadir", str(state.get("datadir")).strip()])
        if str(state.get("userdir") or "").strip():
            args.extend(["--userdir", str(state.get("userdir")).strip()])
        return args

    def _run_review_command(self, args: list[str], title: str) -> None:
        result = freqtrade_command(self.context.shared.python_exe.get(), args)
        self.preview_var.set(command_text(result.preview_command))
        self.console.insert(tk.END, "\n" + "=" * 90 + "\n" + title + "\n" + self.preview_var.get() + "\n")
        self.console.see(tk.END)
        self.context.emit("save_state", {"reason": "review_run"})
        self.context.process_runner.run(result.preview_command, cwd=self.context.shared.project_root.get() or None, owner=self.tab_key)

    def _hyperopt_filename(self) -> str:
        path = self.hyperopt_file_var.get().strip()
        return Path(path).name if path else ""

    def _backtest_filename(self) -> str:
        return self.backtest_file_var.get().strip()

    def _run_hyperopt_list(self) -> None:
        args = ["hyperopt-list", *self._common_args()]
        if self.hyperopt_best_var.get():
            args.append("--best")
        if self.hyperopt_profitable_var.get():
            args.append("--profitable")
        if self.hyperopt_print_json_var.get():
            args.append("--print-json")
        if self.hyperopt_no_details_var.get():
            args.append("--no-details")
        if self.hyperopt_limit_var.get().strip():
            args.extend(["--limit", self.hyperopt_limit_var.get().strip()])
        filename = self._hyperopt_filename()
        if filename:
            args.extend(["--hyperopt-filename", filename])
        self._run_review_command(args, "Hyperopt List")

    def _run_hyperopt_show(self) -> None:
        args = ["hyperopt-show", *self._common_args()]
        if self.hyperopt_best_var.get():
            args.append("--best")
        if self.hyperopt_profitable_var.get():
            args.append("--profitable")
        if self.hyperopt_print_json_var.get():
            args.append("--print-json")
        if self.hyperopt_no_header_var.get():
            args.append("--no-header")
        filename = self._hyperopt_filename()
        if filename:
            args.extend(["--hyperopt-filename", filename])
        if self.hyperopt_index_var.get().strip():
            args.extend(["-n", self.hyperopt_index_var.get().strip()])
        breakdown = self.hyperopt_breakdown_var.get().strip()
        if breakdown and breakdown != "none":
            args.extend(["--breakdown", breakdown])
        self._run_review_command(args, "Hyperopt Show")

    def _run_hyperopt_show_best(self) -> None:
        self.hyperopt_best_var.set(True)
        if not self.hyperopt_index_var.get().strip():
            self.hyperopt_index_var.set("-1")
        self._run_hyperopt_show()

    def _backtest_has_signals(self, filename: str) -> bool | None:
        path = Path(filename)
        if path.suffix.lower() != ".zip" or not path.is_file():
            return None
        try:
            with ZipFile(path) as zip_file:
                return any(name.endswith("_signals.pkl") for name in zip_file.namelist())
        except BadZipFile:
            return None

    def _run_backtest_show(self) -> None:
        args = ["backtesting-show", *self._common_args()]
        filename = self._backtest_filename()
        if filename:
            args.extend(["--backtest-filename", filename])
        if self.backtest_show_pair_list_var.get():
            args.append("--show-pair-list")
        breakdown = self.backtest_breakdown_var.get().strip()
        if breakdown and breakdown != "none":
            args.extend(["--breakdown", breakdown])
        self._run_review_command(args, "Backtesting Show")

    def _run_backtest_analysis(self) -> None:
        args = ["backtesting-analysis", *self._common_args()]
        filename = self._backtest_filename()
        if filename:
            if self._backtest_has_signals(filename) is False:
                self.console.insert(tk.END, "\nSelected backtest zip has no signal data. Re-run backtest with signal export before analysis.\n")
                return
            args.extend(["--backtest-filename", filename])
        for key, switch in (
            (self.analysis_groups_var, "--analysis-groups"),
            (self.enter_reasons_var, "--enter-reason-list"),
            (self.exit_reasons_var, "--exit-reason-list"),
            (self.indicator_list_var, "--indicator-list"),
        ):
            values = _tokens(key.get())
            if values:
                args.extend([switch, *values])
        if self.entry_only_var.get():
            args.append("--entry-only")
        if self.exit_only_var.get():
            args.append("--exit-only")
        if self.rejected_signals_var.get():
            args.append("--rejected-signals")
        if self.analysis_to_csv_var.get():
            args.append("--analysis-to-csv")
        if self.analysis_csv_path_var.get().strip():
            args.extend(["--analysis-csv-path", self.analysis_csv_path_var.get().strip()])
        self._run_review_command(args, "Backtesting Analysis")

    def on_app_event(self, event: str, payload: dict[str, Any]) -> None:
        if event == "process_output":
            append_bounded_text(self.console, str(payload.get("text") or ""))

    def get_state(self) -> dict[str, Any]:
        return {
            "hyperopt_file": self.hyperopt_file_var.get(),
            "hyperopt_limit": self.hyperopt_limit_var.get(),
            "hyperopt_index": self.hyperopt_index_var.get(),
            "hyperopt_best": self.hyperopt_best_var.get(),
            "hyperopt_profitable": self.hyperopt_profitable_var.get(),
            "hyperopt_print_json": self.hyperopt_print_json_var.get(),
            "hyperopt_no_header": self.hyperopt_no_header_var.get(),
            "hyperopt_no_details": self.hyperopt_no_details_var.get(),
            "hyperopt_breakdown": self.hyperopt_breakdown_var.get(),
            "backtest_file": self.backtest_file_var.get(),
            "backtest_directory": self.backtest_directory_var.get(),
            "backtest_show_pair_list": self.backtest_show_pair_list_var.get(),
            "backtest_breakdown": self.backtest_breakdown_var.get(),
            "analysis_groups": self.analysis_groups_var.get(),
            "enter_reasons": self.enter_reasons_var.get(),
            "exit_reasons": self.exit_reasons_var.get(),
            "indicator_list": self.indicator_list_var.get(),
            "entry_only": self.entry_only_var.get(),
            "exit_only": self.exit_only_var.get(),
            "rejected_signals": self.rejected_signals_var.get(),
            "analysis_to_csv": self.analysis_to_csv_var.get(),
            "analysis_csv_path": self.analysis_csv_path_var.get(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.hyperopt_file_var.set(str(state.get("hyperopt_file") or ""))
        self.hyperopt_limit_var.set(str(state.get("hyperopt_limit") or "20"))
        self.hyperopt_index_var.set(str(state.get("hyperopt_index") or "-1"))
        self.hyperopt_best_var.set(bool(state.get("hyperopt_best", False)))
        self.hyperopt_profitable_var.set(bool(state.get("hyperopt_profitable", False)))
        self.hyperopt_print_json_var.set(bool(state.get("hyperopt_print_json", False)))
        self.hyperopt_no_header_var.set(bool(state.get("hyperopt_no_header", False)))
        self.hyperopt_no_details_var.set(bool(state.get("hyperopt_no_details", False)))
        self.hyperopt_breakdown_var.set(str(state.get("hyperopt_breakdown") or "none"))
        self.backtest_file_var.set(str(state.get("backtest_file") or ""))
        self.backtest_directory_var.set(str(state.get("backtest_directory") or ""))
        self.backtest_show_pair_list_var.set(bool(state.get("backtest_show_pair_list", False)))
        self.backtest_breakdown_var.set(str(state.get("backtest_breakdown") or "none"))
        self.analysis_groups_var.set(str(state.get("analysis_groups") or "0 1 2 5"))
        self.enter_reasons_var.set(str(state.get("enter_reasons") or ""))
        self.exit_reasons_var.set(str(state.get("exit_reasons") or ""))
        self.indicator_list_var.set(str(state.get("indicator_list") or ""))
        self.entry_only_var.set(bool(state.get("entry_only", False)))
        self.exit_only_var.set(bool(state.get("exit_only", False)))
        self.rejected_signals_var.set(bool(state.get("rejected_signals", False)))
        self.analysis_to_csv_var.set(bool(state.get("analysis_to_csv", False)))
        self.analysis_csv_path_var.set(str(state.get("analysis_csv_path") or ""))

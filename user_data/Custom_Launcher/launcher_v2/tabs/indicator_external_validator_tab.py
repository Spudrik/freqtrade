from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ..base_tab import BaseTab
from ..command_builder import command_text
from ..console_pane import ConsolePane
from ..services.collector_service import open_path
from ..ui_helpers import labeled_entry


class IndicatorExternalValidatorTab(BaseTab):
    tab_key = "indicator_external_validator"
    tab_title = "Indicator External Validator"

    def __init__(self, master: tk.Misc, context: Any) -> None:
        super().__init__(master, context)
        output_dir = context.app_dir.parent / "Indicator_External_Validator"
        self.runner_path = context.app_dir / "indicator_validation_runner.py"
        self.docs_path = context.app_dir / "docs" / "INDICATOR_EXTERNAL_VALIDATOR_DRAFT.md"
        self.datadir_var = tk.StringVar(value=context.shared.datadir.get())
        self.pairs_var = tk.StringVar(value="")
        self.timeframes_var = tk.StringVar(value="1h")
        self.timerange_var = tk.StringVar(value="")
        self.indicators_var = tk.StringVar(value="all")
        self.score_scope_var = tk.StringVar(value="base")
        self.benchmark_var = tk.StringVar(value="BTC/USDT:USDT")
        self.forward_windows_var = tk.StringVar(value="3 6 12 24")
        self.deciles_var = tk.StringVar(value="10")
        self.score_threshold_var = tk.StringVar(value="0.70")
        self.max_pairs_var = tk.StringVar(value="20")
        self.min_rows_var = tk.StringVar(value="250")
        self.output_dir_var = tk.StringVar(value=str(output_dir))
        self.profile_window_var = tk.StringVar(value="96")
        self.profile_bins_var = tk.StringVar(value="48")
        self.profile_chunk_size_var = tk.StringVar(value="512")
        self.quiet_var = tk.BooleanVar(value=False)
        self.preview_var = tk.StringVar(value="")
        self.summary_path_var = tk.StringVar(value="-")
        self.contract_pass_var = tk.StringVar(value="-")
        self.behavior_pass_var = tk.StringVar(value="-")
        self.score_rows_var = tk.StringVar(value="-")
        self.files_var = tk.StringVar(value="-")
        self.top_scores_var = tk.StringVar(value="-")
        self._build_ui()
        self._bind_refresh()
        self.refresh_preview()
        self.refresh_summary()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)

        intro = ttk.Label(
            self,
            text=(
                "Draft feature: validates external indicator score behaviour. "
                "Subject to deletion/replacement. Use for exploratory checks only."
            ),
            wraplength=1120,
            foreground="#8b1e1e",
        )
        intro.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 6))

        settings = ttk.LabelFrame(self, text="Validation settings")
        settings.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        settings.grid_columnconfigure(1, weight=1)
        settings.grid_columnconfigure(3, weight=1)
        self._path_row(settings, 0, "Data dir", self.datadir_var, directory=True)
        self._path_row(settings, 1, "Output dir", self.output_dir_var, directory=True)
        labeled_entry(settings, 2, 0, "Pairs", self.pairs_var)
        labeled_entry(settings, 2, 2, "Timeframes", self.timeframes_var)
        labeled_entry(settings, 3, 0, "Timerange", self.timerange_var)
        labeled_entry(settings, 3, 2, "Indicators", self.indicators_var)
        ttk.Label(settings, text="Score scope").grid(row=4, column=0, sticky="w", padx=8, pady=4)
        ttk.Combobox(settings, textvariable=self.score_scope_var, values=["base", "all"], state="readonly").grid(row=4, column=1, sticky="ew", padx=8, pady=4)
        labeled_entry(settings, 4, 2, "Benchmark", self.benchmark_var)
        labeled_entry(settings, 5, 0, "Forward windows", self.forward_windows_var)
        labeled_entry(settings, 5, 2, "Deciles", self.deciles_var)
        labeled_entry(settings, 6, 0, "Score threshold", self.score_threshold_var)
        labeled_entry(settings, 6, 2, "Max pairs", self.max_pairs_var)
        labeled_entry(settings, 7, 0, "Min rows", self.min_rows_var)
        labeled_entry(settings, 7, 2, "Profile window", self.profile_window_var)
        labeled_entry(settings, 8, 0, "Profile bins", self.profile_bins_var)
        labeled_entry(settings, 8, 2, "Profile chunk size", self.profile_chunk_size_var)
        ttk.Checkbutton(settings, text="Quiet mode", variable=self.quiet_var).grid(row=9, column=0, sticky="w", padx=8, pady=4)

        preview = ttk.LabelFrame(self, text="Generated command")
        preview.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        preview.grid_columnconfigure(0, weight=1)
        ttk.Entry(preview, textvariable=self.preview_var).grid(row=0, column=0, sticky="ew", padx=8, pady=8)

        actions = ttk.Frame(self)
        actions.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(actions, text="Refresh preview", command=self.refresh_preview).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Run validation", command=self.run_validation).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Refresh summary", command=self.refresh_summary).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Open output folder", command=self.open_output_folder).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Open draft docs", command=self.open_draft_docs).pack(side="left")

        summary = ttk.LabelFrame(self, text="Latest summary")
        summary.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 8))
        items = [
            ("Summary JSON", self.summary_path_var),
            ("Contract pass", self.contract_pass_var),
            ("Behaviour pass", self.behavior_pass_var),
            ("Score rows", self.score_rows_var),
            ("Files discovered", self.files_var),
            ("Top candidates", self.top_scores_var),
        ]
        for index, (label, variable) in enumerate(items):
            row = index // 2
            col = (index % 2) * 2
            ttk.Label(summary, text=f"{label}:").grid(row=row, column=col, sticky="w", padx=8, pady=4)
            ttk.Label(summary, textvariable=variable, wraplength=500, justify="left").grid(row=row, column=col + 1, sticky="w", padx=8, pady=4)

        console = ttk.LabelFrame(self, text="Console")
        console.grid(row=5, column=0, sticky="nsew", padx=8, pady=(0, 8))
        console.grid_columnconfigure(0, weight=1)
        console.grid_rowconfigure(0, weight=1)
        self.console = ConsolePane(console)
        self.console.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    def _path_row(self, parent: tk.Misc, row: int, label: str, variable: tk.StringVar, *, directory: bool) -> None:
        labeled_entry(parent, row, 0, label, variable)
        ttk.Button(parent, text="Browse", command=lambda: self._browse(variable, directory)).grid(row=row, column=2, padx=8, pady=4)

    def _browse(self, variable: tk.StringVar, directory: bool) -> None:
        path = filedialog.askdirectory(parent=self) if directory else filedialog.askopenfilename(parent=self, filetypes=[("All files", "*.*")])
        if path:
            variable.set(path)

    def _bind_refresh(self) -> None:
        for var in (
            self.datadir_var,
            self.pairs_var,
            self.timeframes_var,
            self.timerange_var,
            self.indicators_var,
            self.score_scope_var,
            self.benchmark_var,
            self.forward_windows_var,
            self.deciles_var,
            self.score_threshold_var,
            self.max_pairs_var,
            self.min_rows_var,
            self.output_dir_var,
            self.profile_window_var,
            self.profile_bins_var,
            self.profile_chunk_size_var,
        ):
            var.trace_add("write", lambda *_: self.refresh_preview())
        self.quiet_var.trace_add("write", lambda *_: self.refresh_preview())

    def _state(self) -> dict[str, Any]:
        return {
            "datadir": self.datadir_var.get(),
            "pairs": self.pairs_var.get(),
            "timeframes": self.timeframes_var.get(),
            "timerange": self.timerange_var.get(),
            "indicators": self.indicators_var.get(),
            "score_scope": self.score_scope_var.get(),
            "benchmark": self.benchmark_var.get(),
            "forward_windows": self.forward_windows_var.get(),
            "deciles": self.deciles_var.get(),
            "score_threshold": self.score_threshold_var.get(),
            "max_pairs": self.max_pairs_var.get(),
            "min_rows": self.min_rows_var.get(),
            "output_dir": self.output_dir_var.get(),
            "profile_window": self.profile_window_var.get(),
            "profile_bins": self.profile_bins_var.get(),
            "profile_chunk_size": self.profile_chunk_size_var.get(),
            "quiet": self.quiet_var.get(),
        }

    def _build_command(self) -> list[str]:
        if not self.runner_path.exists():
            raise FileNotFoundError(self.runner_path)
        deciles = int(str(self.deciles_var.get() or "").strip())
        max_pairs = int(str(self.max_pairs_var.get() or "").strip())
        min_rows = int(str(self.min_rows_var.get() or "").strip())
        profile_window = int(str(self.profile_window_var.get() or "").strip())
        profile_bins = int(str(self.profile_bins_var.get() or "").strip())
        profile_chunk = int(str(self.profile_chunk_size_var.get() or "").strip())
        score_threshold = float(str(self.score_threshold_var.get() or "").strip())
        output_dir = Path(str(self.output_dir_var.get() or "").strip()).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._write_runtime_draft_notice(output_dir)

        command = [
            str(self.context.shared.python_exe.get() or "python"),
            "-u",
            str(self.runner_path),
            "--datadir",
            str(self.datadir_var.get().strip()),
            "--timeframes",
            str(self.timeframes_var.get().strip()),
            "--indicators",
            str(self.indicators_var.get().strip()),
            "--score-scope",
            str(self.score_scope_var.get().strip()),
            "--benchmark",
            str(self.benchmark_var.get().strip()),
            "--forward-windows",
            str(self.forward_windows_var.get().strip()),
            "--deciles",
            str(deciles),
            "--score-threshold",
            str(score_threshold),
            "--max-pairs",
            str(max_pairs),
            "--min-rows",
            str(min_rows),
            "--output-dir",
            str(output_dir),
            "--profile-window",
            str(profile_window),
            "--profile-bins",
            str(profile_bins),
            "--profile-chunk-size",
            str(profile_chunk),
        ]
        self._append(command, "--pairs", self.pairs_var.get())
        self._append(command, "--timerange", self.timerange_var.get())
        if self.quiet_var.get():
            command.append("--quiet")
        return command

    @staticmethod
    def _append(command: list[str], flag: str, value: str) -> None:
        text = str(value or "").strip()
        if text:
            command.extend([flag, text])

    def _write_runtime_draft_notice(self, output_dir: Path) -> None:
        notice = output_dir / "README_DRAFT_SUBJECT_TO_DELETION.md"
        if notice.exists():
            return
        notice.write_text(
            (
                "# Indicator External Validator (Draft)\n\n"
                "This folder is runtime output for an experimental validator.\n\n"
                "- Subject to deletion or full redesign.\n"
                "- Not a stable production pipeline.\n"
                "- Keep only files you actively need.\n"
            ),
            encoding="utf-8",
        )

    def refresh_preview(self) -> None:
        try:
            self.preview_var.set(command_text(self._build_command()))
        except Exception as exc:
            self.preview_var.set(f"Invalid Indicator External Validator configuration: {exc}")

    def run_validation(self) -> None:
        try:
            command = self._build_command()
        except Exception as exc:
            messagebox.showerror(self.tab_title, f"Cannot build validator command:\n{exc}", parent=self)
            return
        self.preview_var.set(command_text(command))
        self.context.shared.command_preview.set(command_text(command))
        self.context.emit("save_state", {"reason": "indicator_external_validator_run"})
        try:
            self.context.process_runner.run(command, cwd=self.context.shared.project_root.get() or None, owner=self.tab_key)
        except Exception as exc:
            messagebox.showerror(self.tab_title, f"Could not start validator:\n{exc}", parent=self)
            return
        self.context.shared.status.set("Indicator External Validator running")

    def open_output_folder(self) -> None:
        path = Path(str(self.output_dir_var.get() or "").strip()).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        self._write_runtime_draft_notice(path)
        open_path(path.resolve())

    def open_draft_docs(self) -> None:
        if not self.docs_path.exists():
            messagebox.showinfo(self.tab_title, f"Draft doc not found:\n{self.docs_path}", parent=self)
            return
        open_path(self.docs_path.resolve())

    def refresh_summary(self) -> None:
        output_dir = Path(str(self.output_dir_var.get() or "").strip()).expanduser()
        summary_path = output_dir / "latest_indicator_validation_summary.json"
        self.summary_path_var.set(str(summary_path) if summary_path.exists() else "-")
        if not summary_path.exists():
            self.contract_pass_var.set("-")
            self.behavior_pass_var.set("-")
            self.score_rows_var.set("-")
            self.files_var.set("-")
            self.top_scores_var.set("-")
            return
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Summary payload is not a JSON object.")
        except Exception as exc:
            self.contract_pass_var.set(f"read error: {exc}")
            self.behavior_pass_var.set("-")
            self.score_rows_var.set("-")
            self.files_var.set("-")
            self.top_scores_var.set("-")
            return
        self.contract_pass_var.set(self._format_pct(payload.get("contract_pass_rate")))
        self.behavior_pass_var.set(self._format_pct(payload.get("behavior_pass_rate")))
        self.score_rows_var.set(str(payload.get("score_metric_rows") if payload.get("score_metric_rows") is not None else "-"))
        self.files_var.set(str(payload.get("files_discovered") if payload.get("files_discovered") is not None else "-"))
        top_scores = payload.get("top_scores") or []
        if isinstance(top_scores, list) and top_scores:
            snippets: list[str] = []
            for row in top_scores[:3]:
                if not isinstance(row, dict):
                    continue
                snippets.append(
                    f"{row.get('score', '?')} ({float(row.get('avg_validity') or 0):.3f})"
                )
            self.top_scores_var.set(", ".join(snippets) if snippets else "-")
        else:
            self.top_scores_var.set("-")

    @staticmethod
    def _format_pct(value: Any) -> str:
        try:
            return f"{float(value) * 100.0:.1f}%"
        except Exception:
            return "-"

    def on_app_event(self, event: str, payload: dict[str, Any]) -> None:
        if event != "process_output":
            return
        text = str(payload.get("text") or "")
        self.console.append(text)
        if "Process exited with code" in text:
            self.refresh_summary()

    def get_state(self) -> dict[str, Any]:
        return self._state()

    def set_state(self, state: dict[str, Any]) -> None:
        self.datadir_var.set(str(state.get("datadir") or self.context.shared.datadir.get()))
        self.pairs_var.set(str(state.get("pairs") or ""))
        self.timeframes_var.set(str(state.get("timeframes") or "1h"))
        self.timerange_var.set(str(state.get("timerange") or ""))
        self.indicators_var.set(str(state.get("indicators") or "all"))
        self.score_scope_var.set(str(state.get("score_scope") or "base"))
        self.benchmark_var.set(str(state.get("benchmark") or "BTC/USDT:USDT"))
        self.forward_windows_var.set(str(state.get("forward_windows") or "3 6 12 24"))
        self.deciles_var.set(str(state.get("deciles") or "10"))
        self.score_threshold_var.set(str(state.get("score_threshold") or "0.70"))
        self.max_pairs_var.set(str(state.get("max_pairs") or "20"))
        self.min_rows_var.set(str(state.get("min_rows") or "250"))
        self.output_dir_var.set(str(state.get("output_dir") or (self.context.app_dir.parent / "Indicator_External_Validator")))
        self.profile_window_var.set(str(state.get("profile_window") or "96"))
        self.profile_bins_var.set(str(state.get("profile_bins") or "48"))
        self.profile_chunk_size_var.set(str(state.get("profile_chunk_size") or "512"))
        self.quiet_var.set(bool(state.get("quiet", False)))
        self.refresh_preview()
        self.refresh_summary()

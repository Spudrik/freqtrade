from __future__ import annotations

from typing import Any
import tkinter as tk
from tkinter import ttk
import webbrowser

from ..base_tab import BaseTab
from ..command_builder import command_text, freqtrade_command
from ..console_pane import ConsolePane


class FreqUITab(BaseTab):
    tab_key = "frequi"
    tab_title = "FreqUI"

    def __init__(self, master: tk.Misc, context: Any) -> None:
        super().__init__(master, context)
        self._build_ui()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        controls = ttk.LabelFrame(self, text="FreqUI")
        controls.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        ttk.Button(controls, text="Launch FreqUI", command=self._launch).pack(side="left", padx=8, pady=8)
        ttk.Button(controls, text="Open Browser", command=lambda: webbrowser.open("http://127.0.0.1:8080")).pack(side="left", padx=(0, 8), pady=8)
        ttk.Button(controls, text="Stop", command=self._stop).pack(side="left", padx=(0, 8), pady=8)

        console_frame = ttk.LabelFrame(self, text="Console")
        console_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        console_frame.grid_columnconfigure(0, weight=1)
        console_frame.grid_rowconfigure(0, weight=1)
        self.console = ConsolePane(console_frame)
        self.console.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    def _setup_state(self) -> dict[str, Any]:
        common = self.context.registry.get("common")
        if common is None:
            return {}
        state = common.get_state()
        return state if isinstance(state, dict) else {}

    def _build_args(self) -> list[str]:
        common = self._setup_state()
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

    def _launch(self) -> None:
        result = freqtrade_command(self.context.shared.python_exe.get(), self._build_args())
        self.context.shared.command_preview.set(command_text(result.preview_command))
        self.context.emit("save_state", {"reason": "frequi_launch"})
        self.context.process_runner.run(result.preview_command, cwd=self.context.shared.project_root.get() or None, owner=self.tab_key)
        self.context.shared.status.set("FreqUI running")

    def _stop(self) -> None:
        self.context.process_runner.stop()
        self.context.shared.status.set("Stop requested")

    def on_app_event(self, event: str, payload: dict[str, Any]) -> None:
        if event == "process_output":
            self.console.append(str(payload.get("text") or ""))

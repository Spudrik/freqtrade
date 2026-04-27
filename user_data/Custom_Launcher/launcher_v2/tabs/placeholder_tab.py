from __future__ import annotations

from typing import Any
import tkinter as tk
from tkinter import ttk

from ..base_tab import BaseTab


class PlaceholderTab(BaseTab):
    """Simple placeholder used until old tab logic is migrated."""

    tab_key = "placeholder"
    tab_title = "Placeholder"
    description = "This tab is a migration placeholder."

    def __init__(self, master: tk.Misc, context: Any) -> None:
        super().__init__(master, context)
        self.note_var = tk.StringVar(value="")
        self._build_ui()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        ttk.Label(self, text=self.description, wraplength=900).grid(row=0, column=0, sticky="ew", padx=12, pady=12)
        ttk.Label(self, text="State note").grid(row=1, column=0, sticky="nw", padx=12, pady=(0, 4))
        self.note_entry = ttk.Entry(self, textvariable=self.note_var)
        self.note_entry.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))

    def get_state(self) -> dict[str, Any]:
        return {"note": self.note_var.get()}

    def set_state(self, state: dict[str, Any]) -> None:
        self.note_var.set(str(state.get("note") or ""))

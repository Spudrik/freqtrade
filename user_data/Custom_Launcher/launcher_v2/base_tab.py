from __future__ import annotations

from typing import Any
import tkinter as tk
from tkinter import ttk


class BaseTab(ttk.Frame):
    """Small base class used by all LauncherV2 tabs.

    Tabs own their own Tk variables and widgets. The app shell owns tab
    registration and shared services. Presets call get_state/set_state by tab key.
    """

    tab_key = "base"
    tab_title = "Base"

    def __init__(self, master: tk.Misc, context: Any) -> None:
        super().__init__(master, style="App.TFrame", padding=4)
        self.context = context

    def get_state(self) -> dict[str, Any]:
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        return None

    def refresh(self) -> None:
        return None

    def on_app_event(self, event: str, payload: dict[str, Any]) -> None:
        return None

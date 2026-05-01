from __future__ import annotations

from typing import Any
import tkinter as tk
from tkinter import ttk

from ..base_tab import BaseTab
from .download_tab import DownloadTab
from .global_context_tab import GlobalContextTab
from .news_tab import NewsTab
from .web_tab import WebTab
from .orderbook_tab import OrderBookTab


class DataManagementTab(BaseTab):
    tab_key = "data_management"
    tab_title = "Data Management"

    child_tab_classes = [
        DownloadTab,
        NewsTab,
        WebTab,
        GlobalContextTab,
        OrderBookTab,
    ]

    def __init__(self, master: tk.Misc, context: Any) -> None:
        super().__init__(master, context)
        self.child_tabs: dict[str, BaseTab] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=0, sticky="nsew")
        for tab_cls in self.child_tab_classes:
            tab = tab_cls(notebook, self.context)
            self.child_tabs[tab.tab_key] = tab
            self.context.registry[tab.tab_key] = tab
            notebook.add(tab, text=tab.tab_title)

    def get_state(self) -> dict[str, Any]:
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        return None

    def on_app_event(self, event: str, payload: dict[str, Any]) -> None:
        return None

from __future__ import annotations

from typing import Any
import tkinter as tk
from tkinter import ttk

from ..base_tab import BaseTab
from .download_tab import DownloadTab
from .global_context_tab import GlobalContextTab
from .context_feature_builder_tab import ContextFeatureBuilderTab
from .historical_data_sources_tab import HistoricalDataSourcesTab
from .gdelt_backfill_tab import GdeltBackfillTab
from .news_backfill_sources_tab import NewsBackfillSourcesTab
from .market_context_sources_tab import MarketContextSourcesTab
from .social_context_sources_tab import SocialContextSourcesTab
from .news_tab import NewsTab
from .web_tab import WebTab
from .orderbook_tab import OrderBookTab
from .orderbook_history_tab import OrderBookHistoryTab
from .data_watchdog_tab import DataWatchdogTab


class DataManagementTab(BaseTab):
    tab_key = "data_management"
    tab_title = "Data Management"

    live_tab_classes = [
        NewsTab,
        WebTab,
        GlobalContextTab,
        OrderBookTab,
        DataWatchdogTab,
    ]

    historical_tab_classes = [
        DownloadTab,
        HistoricalDataSourcesTab,
        GdeltBackfillTab,
        NewsBackfillSourcesTab,
        MarketContextSourcesTab,
        SocialContextSourcesTab,
        ContextFeatureBuilderTab,
        OrderBookHistoryTab,
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

        live_frame = ttk.Frame(notebook)
        historical_frame = ttk.Frame(notebook)
        notebook.add(live_frame, text="Live Data")
        notebook.add(historical_frame, text="Historical Data")

        self._build_group_notebook(live_frame, self.live_tab_classes)
        self._build_group_notebook(historical_frame, self.historical_tab_classes)

    def _build_group_notebook(self, frame: ttk.Frame, tab_classes: list[type[BaseTab]]) -> None:
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=1)
        notebook = ttk.Notebook(frame)
        notebook.grid(row=0, column=0, sticky="nsew")
        for tab_cls in tab_classes:
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

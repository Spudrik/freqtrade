from __future__ import annotations

from typing import Any
import tkinter as tk

from .orderbook_tab import OrderBookTab
from .pairs_tab import parse_pairs


class OrderBookHistoryTab(OrderBookTab):
    tab_key = "orderbook_history"
    tab_title = "Order Book History"

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)
        self._build_history_tab(self)

    def refresh_pair_preview(self) -> None:
        return None

    def get_state(self) -> dict[str, Any]:
        return {
            "history_datadir": self.history_datadir_var.get(),
            "history_exchange": self.history_exchange_var.get(),
            "history_trading_mode": self.history_trading_mode_var.get(),
            "history_category": self.history_category_var.get(),
            "history_depth": self.history_depth_var.get(),
            "history_timerange": self.history_timerange_var.get(),
            "history_feature_timeframes": self.history_feature_timeframes_var.get(),
            "history_feature_format": self.history_feature_format_var.get(),
            "history_max_rows": self.history_max_rows_var.get(),
            "history_erase": self.history_erase_var.get(),
            "history_pairs": self.history_pairs_text.get("1.0", tk.END).strip(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.history_datadir_var.set(str(state.get("history_datadir") or self.history_datadir_var.get()))
        self.history_exchange_var.set(str(state.get("history_exchange") or "bybit"))
        self.history_trading_mode_var.set(str(state.get("history_trading_mode") or "futures"))
        self.history_category_var.set(str(state.get("history_category") or "linear"))
        self.history_depth_var.set(str(state.get("history_depth") or "500"))
        self.history_timerange_var.set(str(state.get("history_timerange") or ""))
        self.history_feature_timeframes_var.set(str(state.get("history_feature_timeframes") or "1h"))
        self.history_feature_format_var.set(str(state.get("history_feature_format") or "feather"))
        self.history_max_rows_var.set(str(state.get("history_max_rows") or ""))
        self.history_erase_var.set(bool(state.get("history_erase", False)))
        self._set_history_pairs(parse_pairs(str(state.get("history_pairs") or "")))

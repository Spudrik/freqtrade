from __future__ import annotations

from typing import Any
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk

from ..base_tab import BaseTab
from ..ui_helpers import labeled_entry
from .pairs_tab import parse_pairs


class DownloadTab(BaseTab):
    tab_key = "download"
    tab_title = "Download Data"

    def __init__(self, master: tk.Misc, context: Any) -> None:
        super().__init__(master, context)
        self.exchange_var = tk.StringVar(value="")
        self.pairs_file_var = tk.StringVar(value="")
        self.timeframes_var = tk.StringVar(value="5m 15m 1h")
        self.days_var = tk.StringVar(value="")
        self.new_pairs_days_var = tk.StringVar(value="")
        self.timerange_var = tk.StringVar(value="")
        self.trading_mode_var = tk.StringVar(value="futures")
        self.candle_types_var = tk.StringVar(value="")
        self.data_format_ohlcv_var = tk.StringVar(value="feather")
        self.data_format_trades_var = tk.StringVar(value="feather")
        self.include_inactive_var = tk.BooleanVar(value=False)
        self.no_parallel_var = tk.BooleanVar(value=False)
        self.dl_trades_var = tk.BooleanVar(value=False)
        self.convert_var = tk.BooleanVar(value=False)
        self.erase_var = tk.BooleanVar(value=False)
        self.prepend_var = tk.BooleanVar(value=False)
        self._build_ui()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        options = ttk.LabelFrame(self, text="Download options")
        options.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        options.grid_columnconfigure(1, weight=1)
        options.grid_columnconfigure(3, weight=1)
        ttk.Label(options, text="Exchange").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        ttk.Combobox(options, textvariable=self.exchange_var, values=["binance", "bybit", "coinbase", "kraken", "okx"], state="normal").grid(row=0, column=1, sticky="ew", padx=8, pady=4)
        self._path_row(options, 0, 2, "Pairs file", self.pairs_file_var)
        ttk.Label(options, text="Timeframes").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        ttk.Combobox(options, textvariable=self.timeframes_var, values=["5m 15m 1h", "15m 1h", "1h", "4h", "1d"], state="normal").grid(row=1, column=1, sticky="ew", padx=8, pady=4)
        labeled_entry(options, 1, 2, "Days", self.days_var)
        labeled_entry(options, 2, 0, "New pairs days", self.new_pairs_days_var)
        labeled_entry(options, 2, 2, "Timerange", self.timerange_var)

        ttk.Label(options, text="Trading mode").grid(row=3, column=0, sticky="w", padx=8, pady=4)
        ttk.Combobox(options, textvariable=self.trading_mode_var, values=["spot", "margin", "futures"], state="readonly").grid(row=3, column=1, sticky="ew", padx=8, pady=4)
        ttk.Label(options, text="Candle types").grid(row=3, column=2, sticky="w", padx=8, pady=4)
        ttk.Combobox(options, textvariable=self.candle_types_var, values=["", "spot", "mark", "index", "premiumIndex", "funding_rate"], state="normal").grid(row=3, column=3, sticky="ew", padx=8, pady=4)

        ttk.Label(options, text="OHLCV format").grid(row=4, column=0, sticky="w", padx=8, pady=4)
        ttk.Combobox(options, textvariable=self.data_format_ohlcv_var, values=["feather", "json", "jsongz", "parquet"], state="readonly").grid(row=4, column=1, sticky="ew", padx=8, pady=4)
        ttk.Label(options, text="Trades format").grid(row=4, column=2, sticky="w", padx=8, pady=4)
        ttk.Combobox(options, textvariable=self.data_format_trades_var, values=["feather", "json", "jsongz", "parquet"], state="readonly").grid(row=4, column=3, sticky="ew", padx=8, pady=4)

        flags = ttk.LabelFrame(self, text="Flags")
        flags.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        for column in range(3):
            flags.grid_columnconfigure(column, weight=1)
        ttk.Checkbutton(flags, text="Include inactive pairs", variable=self.include_inactive_var).grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(flags, text="No parallel download", variable=self.no_parallel_var).grid(row=0, column=1, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(flags, text="Download trades", variable=self.dl_trades_var).grid(row=0, column=2, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(flags, text="Convert existing", variable=self.convert_var).grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(flags, text="Erase first", variable=self.erase_var).grid(row=1, column=1, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(flags, text="Prepend older data", variable=self.prepend_var).grid(row=1, column=2, sticky="w", padx=8, pady=6)

        runbar = ttk.Frame(self)
        runbar.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(runbar, text="Download", command=self._run_download).pack(side="left")
        ttk.Button(runbar, text="Stop", command=self._stop).pack(side="left", padx=(8, 0))

        pairs = ttk.LabelFrame(self, text="Download pairs")
        pairs.grid(row=3, column=0, sticky="nsew", padx=8, pady=(0, 8))
        pairs.grid_columnconfigure(0, weight=1)
        pairs.grid_rowconfigure(1, weight=1)
        buttons = ttk.Frame(pairs)
        buttons.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        ttk.Button(buttons, text="Add typed", command=self._add_typed).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Remove selected", command=self._remove_selected).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Normalize", command=self._normalize).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Use whitelist", command=self._copy_whitelist).pack(side="left")
        self.pairs_text = scrolledtext.ScrolledText(pairs, height=10, wrap="word")
        self.pairs_text.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

    def _run_download(self) -> None:
        run_tab = self.context.registry.get("run")
        if run_tab is None or not hasattr(run_tab, "run_download"):
            messagebox.showerror("Download Data", "Run service is not available.", parent=self)
            return
        run_tab.run_download()

    def _stop(self) -> None:
        self.context.process_runner.stop()
        self.context.shared.status.set("Stop requested")

    def _path_row(self, parent: tk.Misc, row: int, column: int, label: str, variable: tk.StringVar) -> None:
        labeled_entry(parent, row, column, label, variable)
        ttk.Button(parent, text="Browse", command=lambda: self._browse_file(variable)).grid(row=row, column=column + 2, padx=8, pady=4)

    def _browse_file(self, variable: tk.StringVar) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if path:
            variable.set(path)

    def _set_pairs(self, pairs: list[str]) -> None:
        self.pairs_text.delete("1.0", tk.END)
        if pairs:
            self.pairs_text.insert("1.0", "\n".join(pairs))

    def _add_typed(self) -> None:
        value = simpledialog.askstring("Download pairs", "Pair(s) to add, comma/space/newline separated:")
        if value:
            existing = parse_pairs(self.pairs_text.get("1.0", tk.END))
            self._set_pairs(parse_pairs("\n".join([*existing, value])))

    def _normalize(self) -> None:
        self._set_pairs(parse_pairs(self.pairs_text.get("1.0", tk.END)))

    def _remove_selected(self) -> None:
        try:
            start = self.pairs_text.index("sel.first")
            end = self.pairs_text.index("sel.last")
            self.pairs_text.delete(start, end)
        except tk.TclError:
            line = self.pairs_text.index("insert").split(".", 1)[0]
            self.pairs_text.delete(f"{line}.0", f"{line}.end+1c")
        self._normalize()

    def _copy_whitelist(self) -> None:
        pairs_tab = self.context.registry.get("pairs")
        if pairs_tab is None:
            return
        state = pairs_tab.get_state()
        self._set_pairs(parse_pairs(str(state.get("pairs") or "")))

    def get_state(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange_var.get(),
            "pairs_file": self.pairs_file_var.get(),
            "timeframes": self.timeframes_var.get(),
            "days": self.days_var.get(),
            "new_pairs_days": self.new_pairs_days_var.get(),
            "timerange": self.timerange_var.get(),
            "trading_mode": self.trading_mode_var.get(),
            "candle_types": self.candle_types_var.get(),
            "data_format_ohlcv": self.data_format_ohlcv_var.get(),
            "data_format_trades": self.data_format_trades_var.get(),
            "include_inactive": self.include_inactive_var.get(),
            "no_parallel": self.no_parallel_var.get(),
            "dl_trades": self.dl_trades_var.get(),
            "convert": self.convert_var.get(),
            "erase": self.erase_var.get(),
            "prepend": self.prepend_var.get(),
            "pairs": self.pairs_text.get("1.0", tk.END).strip(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.exchange_var.set(str(state.get("exchange") or ""))
        self.pairs_file_var.set(str(state.get("pairs_file") or ""))
        self.timeframes_var.set(str(state.get("timeframes") or "5m 15m 1h"))
        self.days_var.set(str(state.get("days") or ""))
        self.new_pairs_days_var.set(str(state.get("new_pairs_days") or ""))
        self.timerange_var.set(str(state.get("timerange") or ""))
        self.trading_mode_var.set(str(state.get("trading_mode") or "futures"))
        self.candle_types_var.set(str(state.get("candle_types") or ""))
        self.data_format_ohlcv_var.set(str(state.get("data_format_ohlcv") or "feather"))
        self.data_format_trades_var.set(str(state.get("data_format_trades") or "feather"))
        self.include_inactive_var.set(bool(state.get("include_inactive", False)))
        self.no_parallel_var.set(bool(state.get("no_parallel", False)))
        self.dl_trades_var.set(bool(state.get("dl_trades", False)))
        self.convert_var.set(bool(state.get("convert", False)))
        self.erase_var.set(bool(state.get("erase", False)))
        self.prepend_var.set(bool(state.get("prepend", False)))
        self._set_pairs(parse_pairs(str(state.get("pairs") or "")))

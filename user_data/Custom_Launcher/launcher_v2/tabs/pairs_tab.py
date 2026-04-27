from __future__ import annotations

from typing import Any, Iterable
import tkinter as tk
from tkinter import scrolledtext, simpledialog, ttk

from ..base_tab import BaseTab


PAIR_REFERENCE_GROUPS: list[dict[str, Any]] = [
    {"name": "Top 10 Market Cap", "note": "Large-cap crypto reference list.", "pairs": ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "TRX", "AVAX", "LINK"]},
    {"name": "Top Volume Candidates", "note": "Static proxy for liquid pairs.", "pairs": ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX", "WIF", "PEPE", "SUI", "NEAR", "LTC", "BCH"]},
    {"name": "High Beta", "note": "Higher volatility candidates.", "pairs": ["WIF", "PEPE", "BONK", "BOME", "DOGE", "SHIB", "ORDI", "TIA", "SEI", "SUI", "INJ", "JUP", "PENDLE", "ENA", "STRK"]},
    {"name": "L1 Networks", "note": "Base-layer networks.", "pairs": ["BTC", "ETH", "SOL", "BNB", "ADA", "AVAX", "TRX", "DOT", "ATOM", "NEAR", "ICP", "APT", "SUI", "SEI", "ALGO"]},
    {"name": "DeFi", "note": "DEX, lending, yield, and governance names.", "pairs": ["UNI", "AAVE", "MKR", "LDO", "CRV", "COMP", "SNX", "SUSHI", "YFI", "1INCH", "PENDLE", "ENA", "DYDX", "GMX", "CAKE"]},
    {"name": "AI/Data", "note": "AI, data, compute, and indexing narratives.", "pairs": ["TAO", "RENDER", "RNDR", "FET", "AGIX", "OCEAN", "ARKM", "GRT", "WLD", "NMR", "PHB", "AI"]},
    {"name": "Payments", "note": "Payments, settlement, and fast-transfer networks.", "pairs": ["XRP", "XLM", "LTC", "BCH", "TRX", "DASH", "CELO", "HBAR", "IOTA", "ALGO"]},
]


def parse_pairs(text: str) -> list[str]:
    raw = str(text or "").replace(",", " ").replace(";", " ").split()
    pairs: list[str] = []
    seen: set[str] = set()
    for item in raw:
        pair = item.strip().upper()
        if not pair:
            continue
        if "/" not in pair:
            pair = f"{pair}/USDT"
        if ":" not in pair:
            pair = f"{pair}:USDT"
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


class PairsTab(BaseTab):
    tab_key = "pairs"
    tab_title = "Pairs"

    def __init__(self, master: tk.Misc, context: Any) -> None:
        super().__init__(master, context)
        self.pair_mode_var = tk.StringVar(value="manual")
        self.pair_reference_format_var = tk.StringVar(value="USDT futures")
        self.reference_tree: ttk.Treeview | None = None
        self.reference_details: scrolledtext.ScrolledText | None = None
        self._build_ui()
        self._populate_reference_tree()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)

        mode = ttk.LabelFrame(self, text="Pair selection")
        mode.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=8)
        ttk.Radiobutton(mode, text="Use config pairlist", variable=self.pair_mode_var, value="config").pack(side="left", padx=8, pady=8)
        ttk.Radiobutton(mode, text="Manual pairs override", variable=self.pair_mode_var, value="manual").pack(side="left", padx=8, pady=8)

        self.pairs_text = self._pair_box("Whitelist pairs", 1, 0)
        self.blacklist_text = self._pair_box("Blacklist pairs", 1, 1)

        ref = ttk.LabelFrame(self, text="Reference pair lists")
        ref.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=8, pady=8)
        ref.grid_columnconfigure(0, weight=1)
        ref.grid_columnconfigure(1, weight=2)
        ref.grid_rowconfigure(1, weight=1)
        controls = ttk.Frame(ref)
        controls.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=8)
        ttk.Label(controls, text="Format").pack(side="left")
        combo = ttk.Combobox(controls, textvariable=self.pair_reference_format_var, values=["USDT futures", "USDT spot"], state="readonly", width=14)
        combo.pack(side="left", padx=(6, 14))
        combo.bind("<<ComboboxSelected>>", lambda _event: self._populate_reference_tree())
        ttk.Button(controls, text="Add to whitelist", command=lambda: self._add_reference(self.pairs_text)).pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="Add to blacklist", command=lambda: self._add_reference(self.blacklist_text)).pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="Replace whitelist", command=self._replace_whitelist).pack(side="left")
        self.reference_tree = ttk.Treeview(ref, columns=("count", "note"), show="tree headings", height=8, selectmode="browse")
        self.reference_tree.heading("#0", text="List")
        self.reference_tree.heading("count", text="Pairs")
        self.reference_tree.heading("note", text="Use")
        self.reference_tree.column("#0", width=220)
        self.reference_tree.column("count", width=70, anchor="center", stretch=False)
        self.reference_tree.column("note", width=420)
        self.reference_tree.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=(0, 8))
        self.reference_tree.bind("<<TreeviewSelect>>", lambda _event: self._update_reference_details())
        self.reference_details = scrolledtext.ScrolledText(ref, height=8, wrap="word")
        self.reference_details.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=(0, 8))

    def _pair_box(self, title: str, row: int, column: int) -> scrolledtext.ScrolledText:
        frame = ttk.LabelFrame(self, text=title)
        frame.grid(row=row, column=column, sticky="nsew", padx=8, pady=8)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)
        buttons = ttk.Frame(frame)
        buttons.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        widget = scrolledtext.ScrolledText(frame, height=10, wrap="word")
        ttk.Button(buttons, text="Add typed", command=lambda: self._add_typed(widget)).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Remove selected", command=lambda: self._remove_selected(widget)).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Normalize", command=lambda: self._normalize(widget)).pack(side="left")
        widget.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        return widget

    def _format_reference_pairs(self, symbols: Iterable[str]) -> list[str]:
        suffix = "" if self.pair_reference_format_var.get() == "USDT spot" else ":USDT"
        return [f"{str(symbol).upper()}/USDT{suffix}" for symbol in symbols]

    def _selected_group(self) -> dict[str, Any] | None:
        if self.reference_tree is None:
            return None
        selection = self.reference_tree.selection()
        if not selection:
            return None
        index = int(selection[0])
        return PAIR_REFERENCE_GROUPS[index] if 0 <= index < len(PAIR_REFERENCE_GROUPS) else None

    def _populate_reference_tree(self) -> None:
        if self.reference_tree is None:
            return
        self.reference_tree.delete(*self.reference_tree.get_children())
        for index, group in enumerate(PAIR_REFERENCE_GROUPS):
            self.reference_tree.insert("", tk.END, iid=str(index), text=str(group["name"]), values=(len(group["pairs"]), group["note"]))
        if PAIR_REFERENCE_GROUPS:
            self.reference_tree.selection_set("0")
            self._update_reference_details()

    def _update_reference_details(self) -> None:
        group = self._selected_group()
        if self.reference_details is None:
            return
        self.reference_details.delete("1.0", tk.END)
        if not group:
            return
        pairs = self._format_reference_pairs(group["pairs"])
        self.reference_details.insert("1.0", f"{group['name']}\n{group['note']}\n\n" + "\n".join(pairs))

    def _append_pairs(self, widget: scrolledtext.ScrolledText, pairs: Iterable[str]) -> None:
        existing = parse_pairs(widget.get("1.0", tk.END))
        merged = parse_pairs("\n".join([*existing, *pairs]))
        widget.delete("1.0", tk.END)
        widget.insert("1.0", "\n".join(merged))
        self.pair_mode_var.set("manual")

    def _add_typed(self, widget: scrolledtext.ScrolledText) -> None:
        value = simpledialog.askstring("Pairs", "Pair(s) to add, comma/space/newline separated:")
        if value:
            self._append_pairs(widget, parse_pairs(value))

    def _normalize(self, widget: scrolledtext.ScrolledText) -> None:
        self._append_pairs(widget, [])

    def _remove_selected(self, widget: scrolledtext.ScrolledText) -> None:
        try:
            start = widget.index("sel.first")
            end = widget.index("sel.last")
            widget.delete(start, end)
        except tk.TclError:
            line = widget.index("insert").split(".", 1)[0]
            widget.delete(f"{line}.0", f"{line}.end+1c")
        self._normalize(widget)

    def _add_reference(self, widget: scrolledtext.ScrolledText) -> None:
        group = self._selected_group()
        if group:
            self._append_pairs(widget, self._format_reference_pairs(group["pairs"]))

    def _replace_whitelist(self) -> None:
        group = self._selected_group()
        if not group:
            return
        self.pairs_text.delete("1.0", tk.END)
        self.pairs_text.insert("1.0", "\n".join(self._format_reference_pairs(group["pairs"])))
        self.pair_mode_var.set("manual")

    def get_state(self) -> dict[str, Any]:
        return {
            "pair_mode": self.pair_mode_var.get(),
            "pair_reference_format": self.pair_reference_format_var.get(),
            "pairs": self.pairs_text.get("1.0", tk.END).strip(),
            "blacklist": self.blacklist_text.get("1.0", tk.END).strip(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.pair_mode_var.set(str(state.get("pair_mode") or "manual"))
        self.pair_reference_format_var.set(str(state.get("pair_reference_format") or "USDT futures"))
        self.pairs_text.delete("1.0", tk.END)
        self.pairs_text.insert("1.0", str(state.get("pairs") or "BTC/USDT:USDT\nETH/USDT:USDT\nSOL/USDT:USDT"))
        self.blacklist_text.delete("1.0", tk.END)
        self.blacklist_text.insert("1.0", str(state.get("blacklist") or ""))
        self._populate_reference_tree()

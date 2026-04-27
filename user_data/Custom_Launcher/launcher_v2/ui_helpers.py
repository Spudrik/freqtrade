from __future__ import annotations

from collections.abc import Iterable
import tkinter as tk
from tkinter import ttk
from typing import Any


class ToolTip:
    """Minimal dependency-free tooltip.

    Use this for short field help only. Do not turn tooltips into long design docs.
    """

    def __init__(self, widget: tk.Widget, text: str, *, delay_ms: int = 450, wraplength: int = 440) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.wraplength = wraplength
        self.tipwindow: tk.Toplevel | None = None
        self._after_id: str | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
        widget.bind("<Destroy>", self._hide, add="+")

    def _schedule(self, _event: tk.Event | None = None) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self) -> None:
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self) -> None:
        self._cancel()
        if self.tipwindow is not None or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 18
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        except Exception:
            return
        self.tipwindow = tk.Toplevel(self.widget)
        self.tipwindow.wm_overrideredirect(True)
        self.tipwindow.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self.tipwindow,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            wraplength=self.wraplength,
            padx=8,
            pady=6,
        )
        label.pack()

    def _hide(self, _event: tk.Event | None = None) -> None:
        self._cancel()
        if self.tipwindow is not None:
            try:
                self.tipwindow.destroy()
            except Exception:
                pass
            self.tipwindow = None


def labeled_entry(parent: tk.Misc, row: int, column: int, label: str, variable: tk.StringVar, *, width: int | None = None) -> tuple[ttk.Label, ttk.Entry]:
    lbl = ttk.Label(parent, text=label)
    lbl.grid(row=row, column=column, sticky="w", padx=8, pady=4)
    entry = ttk.Entry(parent, textvariable=variable, width=width)
    entry.grid(row=row, column=column + 1, sticky="ew", padx=8, pady=4)
    return lbl, entry


def set_tree_rows(tree: ttk.Treeview, rows: Iterable[Iterable[Any]]) -> None:
    for item in tree.get_children():
        tree.delete(item)
    for row in rows:
        tree.insert("", tk.END, values=list(row))


def append_bounded_text(widget: tk.Text, text: str, *, max_lines: int = 5000, follow: bool = True) -> None:
    widget.insert(tk.END, text)
    try:
        line_count = int(widget.index("end-1c").split(".", 1)[0])
    except Exception:
        line_count = 0
    extra = line_count - max_lines
    if extra > 0:
        widget.delete("1.0", f"{extra + 1}.0")
    if follow:
        widget.see(tk.END)


def configure_grid_weights(widget: tk.Misc, columns: Iterable[int] = (), rows: Iterable[int] = ()) -> None:
    for column in columns:
        widget.grid_columnconfigure(column, weight=1)
    for row in rows:
        widget.grid_rowconfigure(row, weight=1)

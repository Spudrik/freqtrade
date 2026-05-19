from __future__ import annotations

from typing import Any
import tkinter as tk
from tkinter import messagebox, ttk

from ..base_tab import BaseTab
from ..services.context_source_catalog_service import (
    DATA_MANAGEMENT_CONTEXT_SOURCE_TODO,
    ContextSourceCatalogService,
    ContextSourceItem,
)


class ContextSourceCatalogView(BaseTab):
    catalog_group = ""
    intro_text = ""

    def __init__(self, master: tk.Misc, context: Any) -> None:
        super().__init__(master, context)
        self.service = ContextSourceCatalogService(context.app_dir)
        self.items = self.service.items(self.catalog_group)
        self._item_by_tree_id: dict[str, ContextSourceItem] = {}
        self.tree: ttk.Treeview | None = None
        self.detail_text: tk.Text | None = None
        self.todo_text: tk.Text | None = None
        self._build_ui()
        self._load_rows()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=1)

        intro = ttk.Label(self, text=self.intro_text, wraplength=1120)
        intro.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 6))

        buttons = ttk.Frame(self)
        buttons.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(buttons, text="Open Source / Download", command=self.open_source).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Open Docs", command=self.open_docs).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Open Planned Storage Folder", command=self.open_storage).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Refresh", command=self._load_rows).pack(side="left")

        list_frame = ttk.LabelFrame(self, text="Candidate sources")
        list_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        list_frame.grid_columnconfigure(0, weight=1)
        list_frame.grid_rowconfigure(0, weight=1)
        columns = ("priority", "name", "status", "merge_target", "timestamp", "storage")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=10)
        for column in columns:
            self.tree.heading(column, text=column)
            self.tree.column(column, width=150, anchor="w")
        self.tree.column("name", width=230)
        self.tree.column("merge_target", width=260)
        self.tree.column("timestamp", width=330)
        self.tree.column("storage", width=260)
        self.tree.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        y_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        y_scroll.grid(row=0, column=1, sticky="ns", pady=8)
        x_scroll = ttk.Scrollbar(list_frame, orient="horizontal", command=self.tree.xview)
        x_scroll.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        self.tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.tree.bind("<<TreeviewSelect>>", self._on_select, add="+")

        detail_frame = ttk.LabelFrame(self, text="Selected source notes")
        detail_frame.grid(row=3, column=0, sticky="nsew", padx=8, pady=(0, 8))
        detail_frame.grid_columnconfigure(0, weight=1)
        detail_frame.grid_rowconfigure(0, weight=1)
        self.detail_text = tk.Text(detail_frame, height=9, wrap="word")
        self.detail_text.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        detail_scroll = ttk.Scrollbar(detail_frame, orient="vertical", command=self.detail_text.yview)
        detail_scroll.grid(row=0, column=1, sticky="ns", pady=8)
        self.detail_text.configure(yscrollcommand=detail_scroll.set)

        todo_frame = ttk.LabelFrame(self, text="Data Management TODO")
        todo_frame.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 8))
        todo_frame.grid_columnconfigure(0, weight=1)
        self.todo_text = tk.Text(todo_frame, height=8, wrap="word")
        self.todo_text.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        self.todo_text.insert(tk.END, DATA_MANAGEMENT_CONTEXT_SOURCE_TODO.strip())
        self.todo_text.configure(state="disabled")

    def _load_rows(self) -> None:
        if self.tree is None:
            return
        self._item_by_tree_id.clear()
        for item_id in self.tree.get_children():
            self.tree.delete(item_id)
        for item in self.items:
            item_id = self.tree.insert(
                "",
                tk.END,
                values=(item.priority, item.name, item.status, item.merge_target, item.timestamp, item.storage),
            )
            self._item_by_tree_id[item_id] = item
        first = self.tree.get_children()
        if first:
            self.tree.selection_set(first[0])
            self.tree.focus(first[0])
            self._show_item(self._item_by_tree_id[first[0]])

    def _selected_item(self) -> ContextSourceItem | None:
        if self.tree is None:
            return None
        selected = self.tree.selection()
        if not selected:
            return None
        return self._item_by_tree_id.get(selected[0])

    def _on_select(self, _event: tk.Event | None = None) -> None:
        item = self._selected_item()
        if item is not None:
            self._show_item(item)

    def _show_item(self, item: ContextSourceItem) -> None:
        if self.detail_text is None:
            return
        details = (
            f"Name: {item.name}\n"
            f"Priority: {item.priority}\n"
            f"Status: {item.status}\n\n"
            f"Contents:\n{item.contents}\n\n"
            f"Timestamp / availability:\n{item.timestamp}\n\n"
            f"Merge target:\n{item.merge_target}\n\n"
            f"Planned storage:\n{item.storage}\n\n"
            f"License / provenance note:\n{item.license_note}\n\n"
            f"Next step:\n{item.next_step}\n\n"
            f"Source:\n{item.source_url}\n\n"
            f"Docs:\n{item.docs_url}"
        )
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert(tk.END, details)
        self.detail_text.configure(state="disabled")

    def open_source(self) -> None:
        item = self._selected_item()
        if item is None:
            messagebox.showinfo(self.tab_title, "Select a source first.", parent=self)
            return
        try:
            self.service.open_url(item.source_url)
        except Exception as exc:
            messagebox.showerror(self.tab_title, f"Could not open source URL:\n{exc}", parent=self)

    def open_docs(self) -> None:
        item = self._selected_item()
        if item is None:
            messagebox.showinfo(self.tab_title, "Select a source first.", parent=self)
            return
        try:
            self.service.open_url(item.docs_url)
        except Exception as exc:
            messagebox.showerror(self.tab_title, f"Could not open docs URL:\n{exc}", parent=self)

    def open_storage(self) -> None:
        item = self._selected_item()
        if item is None:
            messagebox.showinfo(self.tab_title, "Select a source first.", parent=self)
            return
        try:
            self.service.open_storage_folder(item)
        except Exception as exc:
            messagebox.showerror(self.tab_title, f"Could not open planned storage folder:\n{exc}", parent=self)

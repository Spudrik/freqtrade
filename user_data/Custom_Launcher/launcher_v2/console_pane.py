from __future__ import annotations

from typing import Any
import tkinter as tk
from tkinter import ttk


MAX_CONSOLE_LINES = 5000
MAX_SEARCH_TERMS = 20


class ConsolePane(ttk.Frame):
    def __init__(self, master: tk.Misc, *, label: str = "Search") -> None:
        super().__init__(master)
        self.search_term_var = tk.StringVar(value="")
        self.search_status_var = tk.StringVar(value="")
        self.follow_tail_var = tk.BooleanVar(value=True)
        self.search_terms: list[str] = []
        self.search_matches: list[tuple[str, str]] = []
        self.active_match_index = -1
        self._last_search_term = ""

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        tools = ttk.Frame(self)
        tools.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 4))
        ttk.Label(tools, text=label).pack(side="left")
        self.search_combo = ttk.Combobox(tools, textvariable=self.search_term_var, values=self.search_terms, width=34)
        self.search_combo.pack(side="left", padx=(6, 6))
        self.search_combo.bind("<Return>", lambda _event: self.find_next(reset=True))
        self.search_combo.bind("<<ComboboxSelected>>", lambda _event: self.find_next(reset=True))
        ttk.Button(tools, text="Find", command=lambda: self.find_next(reset=True)).pack(side="left", padx=(0, 6))
        ttk.Button(tools, text="Next", command=self.find_next).pack(side="left", padx=(0, 6))
        ttk.Button(tools, text="Back", command=self.find_previous).pack(side="left", padx=(0, 6))
        ttk.Button(tools, text="Clear", command=self.clear).pack(side="left", padx=(0, 6))
        ttk.Button(tools, text="Bottom", command=self.scroll_bottom).pack(side="left", padx=(0, 6))
        ttk.Checkbutton(tools, text="Follow newest", variable=self.follow_tail_var).pack(side="left", padx=(0, 6))
        ttk.Label(tools, textvariable=self.search_status_var).pack(side="left", padx=(6, 0))

        text_frame = ttk.Frame(self)
        text_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        text_frame.grid_columnconfigure(0, weight=1)
        text_frame.grid_rowconfigure(0, weight=1)
        self.text = tk.Text(text_frame, wrap="none", font=("Consolas", 9), undo=False)
        yscroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.text.yview)
        xscroll = ttk.Scrollbar(text_frame, orient="horizontal", command=self.text.xview)
        self.text.configure(
            yscrollcommand=yscroll.set,
            xscrollcommand=xscroll.set,
            background="#101827",
            foreground="#d8e2f0",
            insertbackground="#d8e2f0",
            relief="flat",
        )
        self.text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self.text.tag_configure("search_match", background="#fff2a8", foreground="#000000")
        self.text.tag_configure("search_active", background="#f6a23a", foreground="#000000")

    def at_bottom(self) -> bool:
        try:
            return self.text.yview()[1] >= 0.999
        except Exception:
            return True

    def append(self, text: str) -> None:
        follow = self.follow_tail_var.get() and self.at_bottom()
        self.text.insert(tk.END, text)
        self._trim_buffer()
        if text:
            self._last_search_term = ""
        if follow:
            self.text.see(tk.END)

    def clear(self) -> None:
        self.text.delete("1.0", tk.END)
        self.search_matches = []
        self.active_match_index = -1
        self._last_search_term = ""
        self.search_status_var.set("")

    def _trim_buffer(self) -> None:
        try:
            line_count = int(self.text.index("end-1c").split(".", 1)[0])
        except Exception:
            return
        extra = line_count - MAX_CONSOLE_LINES
        if extra > 0:
            self.text.delete("1.0", f"{extra + 1}.0")

    def scroll_bottom(self) -> None:
        self.text.see(tk.END)

    def remember_search_term(self, term: str) -> None:
        normalized = term.strip()
        if not normalized:
            return
        lowered = normalized.lower()
        self.search_terms = [item for item in self.search_terms if item.lower() != lowered]
        self.search_terms.insert(0, normalized)
        self.search_terms = self.search_terms[:MAX_SEARCH_TERMS]
        self.search_combo.configure(values=self.search_terms)

    def refresh_search_matches(self) -> None:
        term = self.search_term_var.get().strip()
        self.text.tag_remove("search_match", "1.0", tk.END)
        self.text.tag_remove("search_active", "1.0", tk.END)
        self.search_matches = []
        self.active_match_index = -1
        self._last_search_term = term
        if not term:
            self.search_status_var.set("")
            return
        start = "1.0"
        count = tk.IntVar()
        while True:
            index = self.text.search(term, start, stopindex=tk.END, nocase=True, count=count)
            if not index or count.get() <= 0:
                break
            end = f"{index}+{count.get()}c"
            self.search_matches.append((index, end))
            self.text.tag_add("search_match", index, end)
            start = end
        self.search_status_var.set(f"{len(self.search_matches)} matches")

    def set_active_match(self, index: int) -> None:
        if not self.search_matches:
            self.search_status_var.set("No matches")
            return
        self.text.tag_remove("search_active", "1.0", tk.END)
        self.active_match_index = index % len(self.search_matches)
        start, end = self.search_matches[self.active_match_index]
        self.text.tag_add("search_active", start, end)
        self.text.see(start)
        self.search_status_var.set(f"{self.active_match_index + 1}/{len(self.search_matches)}")

    def _ensure_matches(self, *, reset: bool) -> bool:
        term = self.search_term_var.get().strip()
        if not term:
            self.search_status_var.set("Type search text")
            return False
        self.remember_search_term(term)
        if reset or self._last_search_term != term or not self.search_matches:
            self.refresh_search_matches()
        return bool(self.search_matches)

    def find_next(self, reset: bool = False) -> None:
        if not self._ensure_matches(reset=reset):
            return
        self.set_active_match(0 if reset or self.active_match_index < 0 else self.active_match_index + 1)

    def find_previous(self) -> None:
        if not self._ensure_matches(reset=False):
            return
        self.set_active_match(len(self.search_matches) - 1 if self.active_match_index < 0 else self.active_match_index - 1)

    def get_state(self) -> dict[str, Any]:
        return {
            "search_term": self.search_term_var.get(),
            "search_terms": list(self.search_terms),
            "follow_tail": self.follow_tail_var.get(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.search_term_var.set(str(state.get("search_term") or ""))
        terms = state.get("search_terms") or []
        if isinstance(terms, list):
            self.search_terms = [str(item).strip() for item in terms if str(item).strip()][:MAX_SEARCH_TERMS]
        self.search_combo.configure(values=self.search_terms)
        self.follow_tail_var.set(bool(state.get("follow_tail", True)))

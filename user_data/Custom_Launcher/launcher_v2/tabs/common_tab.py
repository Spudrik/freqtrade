from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
import tkinter as tk
from tkinter import filedialog, ttk

from ..base_tab import BaseTab
from ..ui_helpers import labeled_entry


def _class_names(path: str) -> list[str]:
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"), filename=path)
    except Exception:
        return []
    return [node.name for node in tree.body if isinstance(node, ast.ClassDef)]


class CommonTab(BaseTab):
    tab_key = "common"
    tab_title = "Setup"

    def __init__(self, master: tk.Misc, context: Any) -> None:
        super().__init__(master, context)
        self.config_files_var = tk.StringVar(value=str(context.app_dir.parent / "configs" / "config_new2026.example.json"))
        self.strategy_file_var = tk.StringVar(value="")
        self.strategy_class_var = tk.StringVar(value="")
        self.recursive_strategy_search_var = tk.BooleanVar(value=False)
        self.config_listbox: tk.Listbox | None = None
        self.strategy_class_combo: ttk.Combobox | None = None
        self._build_ui()
        self._set_config_files([self.config_files_var.get()])

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        paths = ttk.LabelFrame(self, text="Project and runtime")
        paths.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        paths.grid_columnconfigure(1, weight=1)
        self._path_row(paths, 0, "Project root", self.context.shared.project_root, directory=True)
        self._path_row(paths, 1, "Python exe", self.context.shared.python_exe, filetypes=[("Python", "python.exe *.exe"), ("All files", "*.*")])
        self._path_row(paths, 2, "User data", self.context.shared.userdir, directory=True)
        self._path_row(paths, 3, "Data dir", self.context.shared.datadir, directory=True)

        configs = ttk.LabelFrame(self, text="Config files")
        configs.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        configs.grid_columnconfigure(0, weight=1)
        configs.grid_rowconfigure(1, weight=1)
        buttons = ttk.Frame(configs)
        buttons.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        ttk.Button(buttons, text="Add config", command=self._add_config_files).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Remove", command=self._remove_selected_config).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Up", command=lambda: self._move_config(-1)).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Down", command=lambda: self._move_config(1)).pack(side="left")
        self.config_listbox = tk.Listbox(configs, height=5, exportselection=False)
        self.config_listbox.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        strategy = ttk.LabelFrame(self, text="Strategy")
        strategy.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        strategy.grid_columnconfigure(1, weight=1)
        self._path_row(strategy, 0, "Strategy file", self.strategy_file_var, filetypes=[("Python files", "*.py"), ("All files", "*.*")], callback=self._browse_strategy)
        ttk.Label(strategy, text="Strategy class").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        self.strategy_class_combo = ttk.Combobox(strategy, textvariable=self.strategy_class_var)
        self.strategy_class_combo.grid(row=1, column=1, sticky="ew", padx=8, pady=6)
        ttk.Button(strategy, text="Refresh classes", command=self._refresh_strategy_classes).grid(row=1, column=2, padx=8, pady=6)
        ttk.Checkbutton(strategy, text="Recursive strategy search", variable=self.recursive_strategy_search_var).grid(row=2, column=1, sticky="w", padx=8, pady=(0, 8))

    def _path_row(
        self,
        parent: tk.Misc,
        row: int,
        label: str,
        variable: tk.StringVar,
        *,
        directory: bool = False,
        filetypes: list[tuple[str, str]] | None = None,
        callback: Any | None = None,
    ) -> None:
        labeled_entry(parent, row, 0, label, variable)
        ttk.Button(parent, text="Browse", command=callback or (lambda: self._browse_path(variable, directory, filetypes))).grid(row=row, column=2, padx=8, pady=4)

    def _browse_path(self, variable: tk.StringVar, directory: bool, filetypes: list[tuple[str, str]] | None = None) -> None:
        path = filedialog.askdirectory() if directory else filedialog.askopenfilename(filetypes=filetypes or [("All files", "*.*")])
        if path:
            variable.set(path)

    def _browse_strategy(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Python files", "*.py"), ("All files", "*.*")])
        if path:
            self.strategy_file_var.set(path)
            self._refresh_strategy_classes()

    def _refresh_strategy_classes(self) -> None:
        names = _class_names(self.strategy_file_var.get())
        if self.strategy_class_combo is not None:
            self.strategy_class_combo.configure(values=names)
        if names and self.strategy_class_var.get() not in names:
            self.strategy_class_var.set(names[-1])

    def _config_files(self) -> list[str]:
        if self.config_listbox is None:
            return []
        return [str(self.config_listbox.get(index)) for index in range(self.config_listbox.size()) if str(self.config_listbox.get(index)).strip()]

    def _set_config_files(self, paths: list[str]) -> None:
        if self.config_listbox is None:
            return
        self.config_listbox.delete(0, tk.END)
        for path in paths:
            if str(path).strip():
                self.config_listbox.insert(tk.END, str(path).strip())

    def _add_config_files(self) -> None:
        paths = filedialog.askopenfilenames(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        existing = self._config_files()
        for path in paths:
            if path not in existing and self.config_listbox is not None:
                self.config_listbox.insert(tk.END, path)

    def _remove_selected_config(self) -> None:
        if self.config_listbox is None:
            return
        for index in reversed(self.config_listbox.curselection()):
            self.config_listbox.delete(index)

    def _move_config(self, direction: int) -> None:
        if self.config_listbox is None:
            return
        selection = self.config_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        new_index = index + direction
        if new_index < 0 or new_index >= self.config_listbox.size():
            return
        value = self.config_listbox.get(index)
        self.config_listbox.delete(index)
        self.config_listbox.insert(new_index, value)
        self.config_listbox.selection_set(new_index)

    def get_state(self) -> dict[str, Any]:
        userdir_value = self.context.shared.userdir.get()
        return {
            "project_root": self.context.shared.project_root.get(),
            "python_exe": self.context.shared.python_exe.get(),
            "userdir": userdir_value,
            "datadir": self.context.shared.datadir.get(),
            "config_files": self._config_files(),
            "strategy_file": self.strategy_file_var.get(),
            "strategy_class": self.strategy_class_var.get(),
            "recursive_strategy_search": self.recursive_strategy_search_var.get(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        userdir_value = str(state.get("userdir") or self.context.shared.userdir.get())
        self.context.shared.project_root.set(str(state.get("project_root") or self.context.shared.project_root.get()))
        self.context.shared.python_exe.set(str(state.get("python_exe") or self.context.shared.python_exe.get()))
        self.context.shared.userdir.set(userdir_value)
        self.context.shared.datadir.set(str(state.get("datadir") or self.context.shared.datadir.get()))
        config_files = state.get("config_files")
        if isinstance(config_files, str):
            config_files = [item.strip() for item in config_files.replace(";", "\n").splitlines() if item.strip()]
        config_list = [str(item).strip() for item in (config_files or []) if str(item).strip()]
        self._set_config_files(config_list or [self.config_files_var.get()])
        self.strategy_file_var.set(str(state.get("strategy_file") or ""))
        self.strategy_class_var.set(str(state.get("strategy_class") or ""))
        self.recursive_strategy_search_var.set(bool(state.get("recursive_strategy_search")))
        self._refresh_strategy_classes()

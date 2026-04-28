from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ..base_tab import BaseTab
from ..console_pane import ConsolePane


SUPPORTED_INPUT_EXTENSIONS = {".py", ".json"}
OUTPUT_EXTENSIONS = {".txt", ".md"}
GOOGLEDRIVE_ROOT = Path(__file__).resolve().parents[3] / "googledrive"


@dataclass(frozen=True)
class FileItem:
    path: Path
    root: Path

    @property
    def extension(self) -> str:
        return self.path.suffix.lower()


class FileConverterTab(BaseTab):
    tab_key = "file_converter"
    tab_title = "File Converter"

    def __init__(self, master: tk.Misc, context: Any) -> None:
        super().__init__(master, context)
        self.root_paths: list[Path] = []
        self.scanned_items: list[FileItem] = []
        self.item_vars: dict[str, tk.BooleanVar] = {}
        self.output_extension_var = tk.StringVar(value=".txt")
        self.recursive_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready")
        self.filter_var = tk.StringVar(value="All")
        self.replace_existing_var = tk.BooleanVar(value=True)
        self._build_ui()

    def _build_ui(self) -> None:
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        top = ttk.LabelFrame(self, text="Conversion")
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        top.grid_columnconfigure(7, weight=1)
        ttk.Label(top, text="Output").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Combobox(top, textvariable=self.output_extension_var, values=[".txt", ".md"], width=8, state="readonly").grid(row=0, column=1, sticky="w", padx=(0, 16), pady=6)
        ttk.Checkbutton(top, text="Recursive", variable=self.recursive_var, command=self.scan_files).grid(row=0, column=2, sticky="w", padx=(0, 16), pady=6)
        ttk.Checkbutton(top, text="Replace existing", variable=self.replace_existing_var).grid(row=0, column=3, sticky="w", padx=(0, 16), pady=6)
        ttk.Label(top, text="Filter").grid(row=0, column=4, sticky="w", pady=6)
        ttk.Combobox(top, textvariable=self.filter_var, values=["All", ".py", ".json"], width=10, state="readonly").grid(row=0, column=5, sticky="w", padx=(8, 16), pady=6)
        ttk.Button(top, text="Execute", command=self.execute_conversion).grid(row=0, column=6, sticky="e", padx=8, pady=6)

        folders = ttk.LabelFrame(self, text="Folders")
        folders.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        folders.grid_columnconfigure(0, weight=1)
        self.roots_list = tk.Listbox(folders, height=5, selectmode=tk.EXTENDED)
        self.roots_list.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        root_buttons = ttk.Frame(folders)
        root_buttons.grid(row=0, column=1, sticky="ns", padx=(0, 8), pady=8)
        ttk.Button(root_buttons, text="Add Folder", command=self.add_folder).pack(fill="x", pady=(0, 6))
        ttk.Button(root_buttons, text="Remove Selected", command=self.remove_selected_roots).pack(fill="x", pady=(0, 6))
        ttk.Button(root_buttons, text="Scan", command=self.scan_files).pack(fill="x", pady=(0, 6))
        ttk.Button(root_buttons, text="Select All", command=self.select_all_items).pack(fill="x", pady=(0, 6))
        ttk.Button(root_buttons, text="Select None", command=self.select_none_items).pack(fill="x")

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))

        files_frame = ttk.Frame(body)
        files_frame.grid_rowconfigure(0, weight=1)
        files_frame.grid_columnconfigure(0, weight=1)
        body.add(files_frame, weight=3)
        self.tree = ttk.Treeview(
            files_frame,
            columns=("selected", "file", "root", "type", "size"),
            show="headings",
            selectmode="extended",
        )
        for column, title, width, anchor in (
            ("selected", "Use", 60, "center"),
            ("file", "File", 460, "w"),
            ("root", "Root", 420, "w"),
            ("type", "Type", 80, "center"),
            ("size", "Size", 100, "e"),
        ):
            self.tree.heading(column, text=title)
            self.tree.column(column, width=width, anchor=anchor, stretch=column in {"file", "root"})
        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll = ttk.Scrollbar(files_frame, orient="vertical", command=self.tree.yview)
        tree_scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.bind("<Double-1>", self._toggle_selected_row)

        side = ttk.Frame(body)
        side.grid_rowconfigure(1, weight=1)
        side.grid_columnconfigure(0, weight=1)
        body.add(side, weight=2)
        ttk.Label(side, text="Log").grid(row=0, column=0, sticky="w")
        self.log = ConsolePane(side)
        self.log.grid(row=1, column=0, sticky="nsew", pady=(6, 0))

        ttk.Label(self, textvariable=self.status_var).grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))

    def add_folder(self) -> None:
        folder = filedialog.askdirectory(parent=self, title="Select folder to scan")
        if not folder:
            return
        path = Path(folder).resolve()
        if path not in self.root_paths:
            self.root_paths.append(path)
            self.roots_list.insert(tk.END, str(path))
        self.scan_files()

    def remove_selected_roots(self) -> None:
        for index in reversed(self.roots_list.curselection()):
            del self.root_paths[index]
            self.roots_list.delete(index)
        self.scan_files()

    def scan_files(self) -> None:
        self.scanned_items.clear()
        self.item_vars.clear()
        self.tree.delete(*self.tree.get_children())
        if not self.root_paths:
            self.status_var.set("Add one or more folders to scan.")
            return

        filter_ext = self.filter_var.get().strip().lower()
        recursive = self.recursive_var.get()
        total = 0
        for root in self.root_paths:
            candidates = root.rglob("*") if recursive else root.glob("*")
            for path in candidates:
                if not path.is_file():
                    continue
                if path.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
                    continue
                if filter_ext in {".py", ".json"} and path.suffix.lower() != filter_ext:
                    continue
                item = FileItem(path=path, root=root)
                self.scanned_items.append(item)
                var = tk.BooleanVar(value=True)
                self.item_vars[str(path)] = var
                self._insert_tree_item(item, var)
                total += 1
        self.status_var.set(f"Scanned {total} eligible file(s).")
        self._log(f"Scan complete: {total} eligible file(s).")

    def _insert_tree_item(self, item: FileItem, var: tk.BooleanVar) -> None:
        size_text = self._format_size(item.path.stat().st_size)
        selected_text = "Yes" if var.get() else "No"
        self.tree.insert("", tk.END, iid=str(item.path), values=(selected_text, str(item.path), str(item.root), item.path.suffix.lower(), size_text))

    def _refresh_tree_selection(self) -> None:
        for item in self.scanned_items:
            key = str(item.path)
            var = self.item_vars.get(key)
            if var is None:
                continue
            values = self.tree.item(key, "values")
            self.tree.item(key, values=("Yes" if var.get() else "No", values[1], values[2], values[3], values[4]))

    def _toggle_selected_row(self, _event: Any) -> None:
        for iid in self.tree.selection():
            var = self.item_vars.get(iid)
            if var is not None:
                var.set(not var.get())
        self._refresh_tree_selection()

    def select_all_items(self) -> None:
        for var in self.item_vars.values():
            var.set(True)
        self._refresh_tree_selection()

    def select_none_items(self) -> None:
        for var in self.item_vars.values():
            var.set(False)
        self._refresh_tree_selection()

    def execute_conversion(self) -> None:
        if not self.scanned_items:
            messagebox.showinfo("No files", "Scan a folder first.", parent=self)
            return
        target_ext = self.output_extension_var.get().strip().lower()
        if target_ext not in OUTPUT_EXTENSIONS:
            messagebox.showerror("Invalid output", "Choose either .txt or .md.", parent=self)
            return
        selected = [item for item in self.scanned_items if self._is_selected(item.path)]
        if not selected:
            messagebox.showinfo("No files selected", "Select at least one file to convert.", parent=self)
            return

        written = 0
        skipped = 0
        failures = 0
        for item in selected:
            try:
                output_path = self._build_output_path(item, target_ext)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if output_path.exists() and not self.replace_existing_var.get():
                    skipped += 1
                    self._log(f"Skipped existing: {output_path}")
                    continue
                output_path.write_text(self._render_output(item.path, target_ext), encoding="utf-8")
                written += 1
                self._log(f"Wrote: {output_path}")
            except Exception as exc:
                failures += 1
                self._log(f"Failed {item.path}: {exc}")
        self.status_var.set(f"Done. Wrote {written}, skipped {skipped}, failed {failures}.")
        messagebox.showinfo("Conversion complete", f"Wrote {written} file(s).\nSkipped {skipped} existing file(s).\nFailed {failures} file(s).", parent=self)

    def _render_output(self, source_path: Path, output_ext: str) -> str:
        content = source_path.read_text(encoding="utf-8", errors="replace")
        if output_ext == ".txt":
            return content
        language = "python" if source_path.suffix.lower() == ".py" else "json"
        return f"```{language}\n{content.rstrip()}\n```\n"

    def _build_output_path(self, item: FileItem, output_ext: str) -> Path:
        try:
            relative = item.path.relative_to(item.root)
        except ValueError:
            relative = Path(item.path.name)
        relative_target = Path(relative).with_suffix(output_ext)
        return GOOGLEDRIVE_ROOT / item.root.name / relative_target

    def _is_selected(self, path: Path) -> bool:
        var = self.item_vars.get(str(path))
        return bool(var.get()) if var is not None else False

    def _format_size(self, size: int) -> str:
        units = ["B", "KB", "MB", "GB"]
        value = float(size)
        for unit in units:
            if value < 1024 or unit == units[-1]:
                return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
            value /= 1024
        return f"{size} B"

    def _log(self, message: str) -> None:
        self.log.append(message + "\n")

    def get_state(self) -> dict[str, Any]:
        return {
            "root_paths": [str(path) for path in self.root_paths],
            "output_extension": self.output_extension_var.get(),
            "recursive": self.recursive_var.get(),
            "filter": self.filter_var.get(),
            "replace_existing": self.replace_existing_var.get(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.root_paths = [Path(str(path)).resolve() for path in state.get("root_paths", []) if str(path).strip()]
        self.roots_list.delete(0, tk.END)
        for path in self.root_paths:
            self.roots_list.insert(tk.END, str(path))
        self.output_extension_var.set(str(state.get("output_extension") or ".txt"))
        self.recursive_var.set(bool(state.get("recursive", True)))
        self.filter_var.set(str(state.get("filter") or "All"))
        self.replace_existing_var.set(bool(state.get("replace_existing", True)))

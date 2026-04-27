from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import queue
import tkinter as tk
from typing import Any, Callable


@dataclass
class SharedVars:
    """Small set of cross-tab variables shared by command-building tabs.

    Keep this intentionally small. Tab-specific fields belong in the tab modules.
    """

    project_root: tk.StringVar
    python_exe: tk.StringVar
    userdir: tk.StringVar
    datadir: tk.StringVar
    command_preview: tk.StringVar
    status: tk.StringVar


@dataclass
class LauncherContext:
    """Shared services and paths passed to every tab."""

    app_dir: Path
    preset_path: Path
    output_queue: "queue.Queue[tuple[str, str]]"
    shared: SharedVars
    process_runner: Any | None = None
    preset_manager: Any | None = None
    notify: Callable[[str, dict[str, Any]], None] | None = None
    registry: dict[str, Any] = field(default_factory=dict)

    def emit(self, event: str, payload: dict[str, Any] | None = None) -> None:
        if self.notify:
            self.notify(event, payload or {})

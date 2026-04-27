from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
from typing import Any


class PresetManager:
    """JSON preset storage for LauncherV2.

    Presets are stored as a mapping of preset name to a mapping of tab_key -> tab state.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.presets: dict[str, dict[str, Any]] = {}

    def load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            self.presets = {}
            return self.presets
        last_error: Exception | None = None
        for encoding in ("utf-8", "utf-8-sig"):
            try:
                data = json.loads(self.path.read_text(encoding=encoding))
                if isinstance(data, dict):
                    self.presets = {str(key): value for key, value in data.items() if isinstance(value, dict)}
                else:
                    self.presets = {}
                return self.presets
            except Exception as exc:
                last_error = exc
        if last_error:
            raise last_error
        self.presets = {}
        return self.presets

    def save_all(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.presets, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    def names(self) -> list[str]:
        return sorted(self.presets)

    def get(self, name: str) -> dict[str, Any]:
        return deepcopy(self.presets.get(str(name), {}))

    def set(self, name: str, state: dict[str, Any]) -> None:
        clean_name = str(name).strip()
        if not clean_name:
            raise ValueError("Preset name is required.")
        self.presets[clean_name] = deepcopy(state)
        self.save_all()

    def delete(self, name: str) -> None:
        self.presets.pop(str(name), None)
        self.save_all()

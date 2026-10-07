from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
from typing import Any, Callable

from user_data.Custom_Launcher.collector_runtime import atomic_write_text, lock_resources


AUTO_PRESET_NAME = "LauncherV2-auto"
COLLECTOR_KEYS = frozenset({"news", "web", "global_context", "orderbook"})


def read_presets(path: Path) -> dict[str, Any]:
    """Read the flat LauncherV2 preset file, preserving all preset data."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read presets from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Preset file must contain a JSON object: {path}")
    return payload


def update_auto_preset(path: Path, updater: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    """Apply an update to LauncherV2-auto under the shared preset resource lock."""
    path = Path(path)
    with lock_resources([path]):
        presets = read_presets(path)
        existing = presets.get(AUTO_PRESET_NAME, {})
        if not isinstance(existing, dict):
            raise ValueError(f"Preset {AUTO_PRESET_NAME!r} must contain a JSON object: {path}")
        updated = updater(deepcopy(existing))
        if not isinstance(updated, dict):
            raise ValueError("Preset updater must return a JSON object.")
        presets[AUTO_PRESET_NAME] = updated
        atomic_write_text(path, json.dumps(presets, indent=2, sort_keys=False) + "\n")
        return presets


def _preset_path(appdir: Path, preset_path: Path | None) -> Path:
    return Path(preset_path) if preset_path is not None else Path(appdir) / "launcher_v2" / "config" / "presets.json"


def _collector_key(key: str) -> str:
    if key not in COLLECTOR_KEYS:
        raise ValueError(f"Unknown collector key: {key!r}")
    return key


def _read_collector_desired(preset: dict[str, Any], key: str) -> bool:
    if "collector_desired" not in preset:
        return True
    desired = preset["collector_desired"]
    if not isinstance(desired, dict):
        raise ValueError("Preset collector_desired must be an object.")
    unknown = set(desired) - COLLECTOR_KEYS
    if unknown:
        raise ValueError(f"Unknown collector_desired keys: {', '.join(sorted(map(str, unknown)))}")
    for collector_key, value in desired.items():
        if type(value) is not bool:
            raise ValueError(f"Preset collector_desired[{collector_key!r}] must be a boolean.")
    return desired.get(key, True)


def collector_desired(appdir: Path, key: str, preset_path: Path | None = None) -> bool:
    """Read persisted collector intent; missing legacy intent means enabled."""
    key = _collector_key(key)
    presets = read_presets(_preset_path(appdir, preset_path))
    preset = presets.get(AUTO_PRESET_NAME, {})
    if not isinstance(preset, dict):
        raise ValueError(f"Preset {AUTO_PRESET_NAME!r} must contain a JSON object.")
    return _read_collector_desired(preset, key)


def set_collector_desired(appdir: Path, key: str, value: bool, preset_path: Path | None = None) -> dict[str, Any]:
    """Persist explicit collector intent without replacing unrelated preset data."""
    key = _collector_key(key)
    if type(value) is not bool:
        raise ValueError("Collector desired state must be a boolean.")

    def update(preset: dict[str, Any]) -> dict[str, Any]:
        if "collector_desired" not in preset:
            desired: dict[str, Any] = {}
        else:
            existing = preset["collector_desired"]
            if not isinstance(existing, dict):
                raise ValueError("Preset collector_desired must be an object.")
            desired = dict(existing)
        unknown = set(desired) - COLLECTOR_KEYS
        if unknown:
            raise ValueError(f"Unknown collector_desired keys: {', '.join(sorted(map(str, unknown)))}")
        for collector_key, current in desired.items():
            if type(current) is not bool:
                raise ValueError(f"Preset collector_desired[{collector_key!r}] must be a boolean.")
        desired[key] = value
        preset["collector_desired"] = desired
        return preset

    presets = update_auto_preset(_preset_path(appdir, preset_path), update)
    return presets[AUTO_PRESET_NAME]


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

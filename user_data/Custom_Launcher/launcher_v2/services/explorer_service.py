from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import sys
from typing import Any

from ..command_builder import command_text


def _split_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").replace(";", ",").replace("\n", ",")
    return [item.strip() for item in text.split(",") if item.strip()]


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_max_loops(value: Any) -> str:
    try:
        count = int(str(value).strip())
    except (TypeError, ValueError):
        return "0"
    return "0" if count <= 0 else "1"


def _normalize_preset_file(value: Any) -> str:
    text = str(value or "").strip()
    return text or "launcher_v2/config/presets.json"


def _normalize_windows_file(value: Any) -> str:
    text = str(value or "").strip()
    return text or "explorer/config/market_windows.json"


def _normalize_runner_path(value: Any) -> str:
    text = str(value or "").strip()
    return text or "explorer/explorer_runner.py"


def _normalize_metadata_file(value: Any) -> str:
    text = str(value or "").strip()
    return text or "../explorer_reports/latest_summary.json"


def _normalize_state_file(value: Any) -> str:
    text = str(value or "").strip()
    return text or "../explorer_reports/hyperopt_explorer_state.json"


@dataclass
class ExplorerRunSettings:
    preset_name: str = "BackTest2021-26"
    preset_file: str = "launcher_v2/config/presets.json"
    market_windows_file: str = "explorer/config/market_windows.json"
    runner_path: str = "explorer/explorer_runner.py"
    metadata_file: str = "../explorer_reports/latest_summary.json"
    state_file: str = "../explorer_reports/hyperopt_explorer_state.json"
    target_type: str = "family"
    target_selection: str = "random"
    target_name: str = ""
    search_breadth: str = "targeted"
    training_windows: list[str] = field(default_factory=list)
    validation_windows: list[str] = field(default_factory=lambda: ["full_cycle_2020_2026"])
    max_loops: str = "0"
    epochs: str = "200"
    auto_epochs: bool = False
    auto_epochs_cap: str = ""
    random_state: str = ""
    sampling_seed: str = ""
    backtest_workers: str = "12"
    strategy_param_file: str = ""

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "ExplorerRunSettings":
        preset_name = str(state.get("preset_name") or "BackTest2021-26")
        if preset_name in {"test-hyperopt", "Backup"}:
            preset_name = "BackTest2021-26"
        return cls(
            preset_name=preset_name,
            preset_file=_normalize_preset_file(state.get("preset_file")),
            market_windows_file=_normalize_windows_file(state.get("market_windows_file")),
            runner_path=_normalize_runner_path(state.get("runner_path")),
            metadata_file=_normalize_metadata_file(state.get("metadata_file")),
            state_file=_normalize_state_file(state.get("state_file")),
            target_type=str(state.get("target_type") or "family").lower(),
            target_selection=str(state.get("target_selection") or "random").lower(),
            target_name=str(state.get("target_name") or ""),
            search_breadth=str(state.get("search_breadth") or "targeted").lower(),
            training_windows=_split_list(state.get("training_windows")),
            validation_windows=_split_list(state.get("validation_windows")) or ["full_cycle_2020_2026"],
            max_loops=_normalize_max_loops(state.get("max_loops")),
            epochs=str(state.get("epochs") or "200"),
            auto_epochs=_to_bool(state.get("auto_epochs")),
            auto_epochs_cap=str(state.get("auto_epochs_cap") or ""),
            random_state=str(state.get("random_state") or ""),
            sampling_seed=str(state.get("sampling_seed") or ""),
            backtest_workers=str(state.get("backtest_workers") or "12"),
            strategy_param_file=str(state.get("strategy_param_file") or ""),
        )

    def to_state(self) -> dict[str, Any]:
        return {
            "preset_name": self.preset_name,
            "preset_file": self.preset_file,
            "market_windows_file": self.market_windows_file,
            "runner_path": self.runner_path,
            "metadata_file": self.metadata_file,
            "state_file": self.state_file,
            "target_type": self.target_type,
            "target_selection": self.target_selection,
            "target_name": self.target_name,
            "search_breadth": self.search_breadth,
            "training_windows": list(self.training_windows),
            "validation_windows": list(self.validation_windows),
            "max_loops": self.max_loops,
            "epochs": self.epochs,
            "auto_epochs": self.auto_epochs,
            "auto_epochs_cap": self.auto_epochs_cap,
            "random_state": self.random_state,
            "sampling_seed": self.sampling_seed,
            "backtest_workers": self.backtest_workers,
            "strategy_param_file": self.strategy_param_file,
        }


class ExplorerService:
    """LauncherV2 adapter for the simplified Explorer runner.

    This service owns path/command adaptation only. It does not score, validate,
    parse raw console output, or expose legacy Explorer modes.
    """

    def __init__(self, app_dir: Path, python_exe: str | None = None) -> None:
        self.app_dir = Path(app_dir)
        self.python_exe = python_exe or sys.executable

    def resolve_path(self, value: str) -> Path:
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else (self.app_dir / path)

    def build_command(self, settings: ExplorerRunSettings) -> list[str]:
        self._validate(settings)
        command = [
            self.python_exe,
            "-u",
            "-m",
            self._runner_module(settings.runner_path),
            "--preset-file",
            str(self.resolve_path(settings.preset_file)),
            "--preset",
            settings.preset_name,
            "--market-windows-file",
            str(self.resolve_path(settings.market_windows_file)),
            "--target-type",
            settings.target_type,
            "--target-selection",
            settings.target_selection,
            "--search-breadth",
            settings.search_breadth,
            "--training-windows-json",
            json.dumps(settings.training_windows, separators=(",", ":")),
            "--validation-windows-json",
            json.dumps(settings.validation_windows, separators=(",", ":")),
            "--max-loops",
            str(settings.max_loops),
            "--metadata-file",
            str(self.resolve_path(settings.metadata_file)),
            "--state-file",
            str(self.resolve_path(settings.state_file)),
        ]
        self._append(command, "--target-name", settings.target_name)
        if settings.auto_epochs:
            command.append("--auto-epochs")
            self._append(command, "--auto-epochs-cap", settings.auto_epochs_cap)
        else:
            self._append(command, "--epochs", settings.epochs)
        self._append(command, "--random-state", settings.random_state)
        self._append(command, "--sampling-seed", settings.sampling_seed)
        self._append(command, "--backtest-workers", settings.backtest_workers)
        self._append(command, "--strategy-param-file", settings.strategy_param_file)
        return command

    def preview_text(self, settings: ExplorerRunSettings) -> str:
        return command_text(self.build_command(settings))

    def load_latest_summary(self, metadata_file: str) -> dict[str, Any]:
        path = self.resolve_path(metadata_file)
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}

    def load_window_names(self, market_windows_file: str) -> list[str]:
        path = self.resolve_path(market_windows_file)
        if not path.exists():
            return ["full_cycle_2020_2026"]
        data = json.loads(path.read_text(encoding="utf-8"))
        windows = data.get("market_windows") if isinstance(data, dict) else []
        names = [str(item.get("name")) for item in windows if isinstance(item, dict) and str(item.get("name") or "")]
        if "full_cycle_2020_2026" not in names:
            names.insert(0, "full_cycle_2020_2026")
        return names

    def load_windows(self, market_windows_file: str) -> list[dict[str, str]]:
        path = self.resolve_path(market_windows_file)
        if not path.exists():
            return [
                {
                    "name": "full_cycle_2020_2026",
                    "regime": "mixed",
                    "segment_type": "full_cycle",
                    "timerange": "20200101-20260101",
                }
            ]
        from explorer.explorer_windows import compact_window, load_window_manifest

        return [compact_window(window) for window in load_window_manifest(path)]

    def load_target_catalog(self, strategy_file: str, strategy_class: str, state_file: str) -> list[dict[str, Any]]:
        if not strategy_file or not strategy_class:
            return []
        from explorer.explorer_catalog import catalog_table, load_catalog
        from explorer.explorer_commands import load_json, strategy_parameter_spaces

        strategy_path = Path(strategy_file).expanduser()
        if not strategy_path.is_absolute():
            strategy_path = (self.app_dir.parent / strategy_path).resolve()
        if not strategy_path.exists():
            return []
        catalog = load_catalog(strategy_path, strategy_class)
        spaces = strategy_parameter_spaces(strategy_path, strategy_class)
        active_params = set(spaces.keys())
        catalog["params"] = {
            name: value
            for name, value in (catalog.get("params") or {}).items()
            if str(name) in active_params
        }
        state = load_json(self.resolve_path(state_file), {})
        return catalog_table(catalog, state if isinstance(state, dict) else {})

    def _validate(self, settings: ExplorerRunSettings) -> None:
        if settings.target_type not in {"family", "mode"}:
            raise ValueError("Explorer target_type must be 'family' or 'mode'.")
        if settings.target_selection not in {"random", "specific"}:
            raise ValueError("Explorer target_selection must be 'random' or 'specific'.")
        if settings.search_breadth not in {"targeted", "open"}:
            raise ValueError("Explorer search_breadth must be 'targeted' or 'open'.")
        if settings.target_selection == "specific" and not settings.target_name.strip():
            raise ValueError("Specific Explorer target selection requires a target name.")
        if not settings.training_windows:
            raise ValueError("Explorer requires at least one training window.")
        if not settings.validation_windows:
            raise ValueError("Explorer requires at least one validation window.")
        try:
            max_loops = int(str(settings.max_loops).strip())
        except (TypeError, ValueError):
            raise ValueError("Explorer max_loops must be 0 (infinite) or 1.") from None
        if max_loops not in {0, 1}:
            raise ValueError("Explorer max_loops must be 0 (infinite) or 1.")
        if settings.auto_epochs:
            cap_text = str(settings.auto_epochs_cap or "").strip()
            if cap_text:
                try:
                    cap_value = int(cap_text)
                except (TypeError, ValueError):
                    raise ValueError("Auto epoch cap must be an integer. Use 0 or blank for no limit.") from None
                if cap_value < 0:
                    raise ValueError("Auto epoch cap must be >= 0. Use 0 or blank for no limit.")
        else:
            epochs_text = str(settings.epochs or "").strip()
            try:
                epochs_value = int(epochs_text)
            except (TypeError, ValueError):
                raise ValueError("Explorer epochs must be a positive integer when Auto epochs is disabled.") from None
            if epochs_value < 1:
                raise ValueError("Explorer epochs must be >= 1 when Auto epochs is disabled.")

    @staticmethod
    def _append(command: list[str], flag: str, value: Any) -> None:
        text = str(value or "").strip()
        if text:
            command.extend([flag, text])

    @staticmethod
    def _runner_module(runner_path: str) -> str:
        normalized = str(runner_path or "").replace("\\", "/").strip().lower()
        if normalized in {"", "explorer/explorer_runner.py", "explorer.explorer_runner"}:
            return "explorer.explorer_runner"
        raise ValueError("Explorer runner must be the packaged explorer.explorer_runner module.")

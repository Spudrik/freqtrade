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


def _dedupe_paths(paths: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for raw_path in paths:
        text = str(raw_path or "").strip()
        if not text:
            continue
        try:
            key = str(Path(text).expanduser().resolve()).lower()
        except OSError:
            key = str(Path(text).expanduser()).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def _normalize_worker_count(value: Any, default: str = "2") -> str:
    try:
        count = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return str(min(9, max(1, count)))


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


def _default_repo_root(app_dir: Path) -> Path:
    return Path(app_dir).resolve().parent.parent


def _default_backtest_python(app_dir: Path) -> str:
    return str(_default_repo_root(app_dir) / "runtime" / "venvs" / "freqtrade-backtest" / "Scripts" / "python.exe")


def _default_backtest_pythons(app_dir: Path) -> list[str]:
    repo_root = _default_repo_root(app_dir)
    worker_root = repo_root / "runtime" / "venvs"
    return [
        str(worker_root / "freqtrade-backtest" / "Scripts" / "python.exe"),
        *[
            str(worker_root / f"freqtrade-backtest-{index:02d}" / "Scripts" / "python.exe")
            for index in range(1, 9)
        ],
    ]


def _default_handoff_dir(app_dir: Path) -> str:
    return str(Path(app_dir).resolve().parent / "explorer_reports" / "pipeline")


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
    strategy_param_file: str = ""
    split_venv_pipeline: bool = False
    backtest_python_exe: str = ""
    backtest_python_exes: list[str] = field(default_factory=list)
    backtest_worker_count: str = "2"
    pipeline_handoff_dir: str = ""

    @classmethod
    def from_state(cls, state: dict[str, Any], app_dir: Path | None = None) -> "ExplorerRunSettings":
        preset_name = str(state.get("preset_name") or "BackTest2021-26")
        if preset_name in {"test-hyperopt", "Backup"}:
            preset_name = "BackTest2021-26"
        default_backtest_python = _default_backtest_python(app_dir) if app_dir is not None else ""
        default_backtest_pythons = _default_backtest_pythons(app_dir) if app_dir is not None else ([default_backtest_python] if default_backtest_python else [])
        default_handoff_dir = _default_handoff_dir(app_dir) if app_dir is not None else "../explorer_reports/pipeline"
        primary_backtest_python = str(state.get("backtest_python_exe") or state.get("explorer_backtest_python_exe") or default_backtest_python)
        configured_backtest_pythons = _split_list(state.get("backtest_python_exes") or state.get("explorer_backtest_python_exes"))
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
            strategy_param_file=str(state.get("strategy_param_file") or ""),
            split_venv_pipeline=_to_bool(state.get("split_venv_pipeline") or state.get("explorer_split_venv_pipeline")),
            backtest_python_exe=primary_backtest_python,
            backtest_python_exes=_dedupe_paths([primary_backtest_python, *configured_backtest_pythons, *default_backtest_pythons]),
            backtest_worker_count=_normalize_worker_count(state.get("backtest_worker_count") or state.get("explorer_backtest_worker_count"), "2"),
            pipeline_handoff_dir=str(state.get("pipeline_handoff_dir") or state.get("explorer_pipeline_handoff_dir") or default_handoff_dir),
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
            "strategy_param_file": self.strategy_param_file,
            "split_venv_pipeline": self.split_venv_pipeline,
            "backtest_python_exe": self.backtest_python_exe,
            "backtest_python_exes": list(self.backtest_python_exes),
            "backtest_worker_count": self.backtest_worker_count,
            "pipeline_handoff_dir": self.pipeline_handoff_dir,
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
        runner_module = "explorer.explorer_pipeline_runner" if settings.split_venv_pipeline else self._runner_module(settings.runner_path)
        command = [
            self.python_exe,
            "-u",
            "-m",
            runner_module,
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
        self._append(command, "--strategy-param-file", settings.strategy_param_file)
        if settings.split_venv_pipeline:
            self._append(command, "--backtest-python-exe", settings.backtest_python_exe)
            self._append(command, "--backtest-python-exes-json", json.dumps(settings.backtest_python_exes, separators=(",", ":")))
            self._append(command, "--backtest-worker-count", settings.backtest_worker_count)
            self._append(command, "--handoff-dir", settings.pipeline_handoff_dir)
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
        if settings.split_venv_pipeline:
            if not str(settings.backtest_python_exe or "").strip():
                raise ValueError("Split-venv pipeline requires a backtest Python executable.")
            try:
                worker_count = int(str(settings.backtest_worker_count or "").strip())
            except (TypeError, ValueError):
                raise ValueError("Split-venv backtest workers must be an integer from 1 to 9.") from None
            if worker_count < 1 or worker_count > 9:
                raise ValueError("Split-venv backtest workers must be between 1 and 9.")
            if not str(settings.pipeline_handoff_dir or "").strip():
                raise ValueError("Split-venv pipeline requires a handoff directory.")
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

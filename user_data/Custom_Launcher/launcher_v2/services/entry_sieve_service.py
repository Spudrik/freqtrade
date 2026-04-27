from __future__ import annotations

import ast
from dataclasses import dataclass, field
from datetime import datetime
import fnmatch
import json
from pathlib import Path
import sys
from typing import Any


EXCLUDED_STRATEGY_FILES = {
    "__init__.py",
    "test_entry_research_base.py",
    "PivotTrendlineMTFResearchStrategy.py",
}


def _split_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").replace(";", ",").replace("\n", ",")
    return [item.strip() for item in text.split(",") if item.strip()]


@dataclass
class EntrySieveSettings:
    preset_name: str = "LauncherV2-auto"
    preset_file: str = "launcher_v2/config/presets.json"
    market_windows_file: str = "explorer/config/market_windows.json"
    training_windows: list[str] = field(default_factory=list)
    validation_windows: list[str] = field(default_factory=lambda: ["full_cycle_2020_2026"])
    epochs: str = "200"
    random_state: str = ""
    sampling_seed: str = ""
    backtest_workers: str = "1"
    strategy_filter: str = "*.py"
    take_profit_pct: str = "2"
    stoploss_pct: str = "2"


class EntrySieveService:
    def __init__(self, app_dir: Path, python_exe: str | None = None) -> None:
        self.app_dir = Path(app_dir)
        self.python_exe = python_exe or sys.executable
        self.runtime_dir = self.app_dir / "launcher_v2" / "runtime" / "entry_sieve"

    def resolve_path(self, value: str) -> Path:
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else (self.app_dir / path)

    def strategy_dir(self) -> Path:
        return (self.app_dir.parent / "strategies").resolve()

    def discover_strategies(self, strategy_filter: str = "*.py") -> list[dict[str, str]]:
        patterns = _split_list(strategy_filter) or ["*.py"]
        rows: list[dict[str, str]] = []
        for path in sorted(self.strategy_dir().glob("*.py")):
            if path.name in EXCLUDED_STRATEGY_FILES:
                continue
            if not any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns):
                continue
            class_name = self._strategy_class_name(path)
            if not class_name:
                continue
            rows.append({"strategy_file": str(path), "strategy_class": class_name, "name": path.stem})
        return rows

    def build_job(self, settings: EntrySieveSettings) -> Path:
        self._validate(settings)
        strategies = self.discover_strategies(settings.strategy_filter)
        if not strategies:
            raise ValueError("Entry Sieve found no top-level strategy files for the current filter.")
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        job_id = datetime.now().strftime("entry_sieve_%Y%m%dT%H%M%S")
        job_path = self.runtime_dir / f"{job_id}.json"
        job = {
            "schema_version": 1,
            "job_id": job_id,
            "created_at": datetime.now().astimezone().isoformat(),
            "preset_name": settings.preset_name,
            "preset_file": str(self.resolve_path(settings.preset_file)),
            "market_windows_file": str(self.resolve_path(settings.market_windows_file)),
            "training_windows": list(settings.training_windows),
            "validation_windows": list(settings.validation_windows),
            "epochs": str(settings.epochs),
            "python_exe": str(self.python_exe),
            "random_state": str(settings.random_state),
            "sampling_seed": str(settings.sampling_seed),
            "backtest_workers": str(settings.backtest_workers),
            "strategy_filter": str(settings.strategy_filter),
            "take_profit_pct": str(settings.take_profit_pct),
            "stoploss_pct": str(settings.stoploss_pct),
            "runtime_dir": str(self.runtime_dir),
            "strategies": strategies,
        }
        job_path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
        return job_path

    def build_command(self, settings: EntrySieveSettings) -> list[str]:
        job_path = self.build_job(settings)
        return [self.python_exe, "-u", "-m", "launcher_v2.services.entry_sieve_runner", "--job-file", str(job_path)]

    def load_results(self) -> list[dict[str, Any]]:
        path = self.runtime_dir / "results.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("rows") if isinstance(data, dict) else []
        return rows if isinstance(rows, list) else []

    def _validate(self, settings: EntrySieveSettings) -> None:
        if not settings.training_windows:
            raise ValueError("Entry Sieve requires at least one selected training window.")
        if not settings.validation_windows:
            raise ValueError("Entry Sieve requires at least one selected validation window.")
        try:
            epochs = int(str(settings.epochs).strip())
        except (TypeError, ValueError):
            raise ValueError("Entry Sieve epochs must be a positive integer.") from None
        if epochs < 1:
            raise ValueError("Entry Sieve epochs must be >= 1.")

    @staticmethod
    def _strategy_class_name(path: Path) -> str:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except Exception:
            return ""
        candidates: list[str] = []
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            base_names = {EntrySieveService._base_name(base) for base in node.bases}
            if "IStrategy" in base_names or any(name.endswith("Mixin") for name in base_names) or any(name.startswith("Daily") for name in base_names):
                candidates.append(node.name)
        if path.stem in candidates:
            return path.stem
        return candidates[-1] if candidates else ""

    @staticmethod
    def _base_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

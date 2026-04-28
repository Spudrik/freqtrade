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
    auto_epochs: bool = False
    auto_epochs_cap: str = ""
    random_state: str = ""
    sampling_seed: str = ""
    split_venv_pipeline: bool = False
    backtest_python_exe: str = ""
    pipeline_handoff_dir: str = ""
    strategy_filter: str = "*.py"
    take_profit_pct: str = "2"
    stoploss_pct: str = "2"
    target_sweep_enabled: bool = False
    target_sweep_pairs: str = ""


class EntrySieveService:
    def __init__(self, app_dir: Path, python_exe: str | None = None) -> None:
        self.app_dir = Path(app_dir)
        self.python_exe = python_exe or sys.executable
        self.runtime_dir = self.app_dir / "launcher_v2" / "runtime" / "entry_sieve"

    @property
    def jobs_dir(self) -> Path:
        return self.runtime_dir / "jobs"

    @property
    def results_dir(self) -> Path:
        return self.runtime_dir / "results"

    @property
    def archive_dir(self) -> Path:
        return self.runtime_dir / "archive"

    @property
    def status_dir(self) -> Path:
        return self.runtime_dir / "status"

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
            rows.append(
                {
                    "strategy_file": str(path),
                    "strategy_class": class_name,
                    "name": path.stem,
                    "side": self._strategy_side(path.stem),
                    "core_behavior": self._strategy_core_behavior(path.stem),
                }
            )
        return rows

    def build_job(self, settings: EntrySieveSettings) -> Path:
        self._validate(settings)
        strategies = self.discover_strategies(settings.strategy_filter)
        if not strategies:
            raise ValueError("Entry Sieve found no top-level strategy files for the current filter.")
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._archive_legacy_results()
        job_id = datetime.now().strftime("entry_sieve_%Y%m%dT%H%M%S")
        job_path = self.jobs_dir / f"{job_id}.json"
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
            "auto_epochs": bool(settings.auto_epochs),
            "auto_epochs_cap": str(settings.auto_epochs_cap),
            "python_exe": str(self.python_exe),
            "random_state": str(settings.random_state),
            "sampling_seed": str(settings.sampling_seed),
            "split_venv_pipeline": bool(settings.split_venv_pipeline),
            "backtest_python_exe": str(settings.backtest_python_exe),
            "pipeline_handoff_dir": str(settings.pipeline_handoff_dir),
            "strategy_filter": str(settings.strategy_filter),
            "take_profit_pct": str(settings.take_profit_pct),
            "stoploss_pct": str(settings.stoploss_pct),
            "target_sweep_enabled": bool(settings.target_sweep_enabled),
            "target_sweep_pairs": str(settings.target_sweep_pairs),
            "runtime_dir": str(self.runtime_dir),
            "strategies": strategies,
        }
        job_path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
        return job_path

    def build_command(self, settings: EntrySieveSettings) -> list[str]:
        job_path = self.build_job(settings)
        return [self.python_exe, "-u", "-m", "launcher_v2.services.entry_sieve_runner", "--job-file", str(job_path)]

    def load_result_batches(self) -> list[dict[str, Any]]:
        batches: list[dict[str, Any]] = []
        latest_id = self._latest_result_batch_id()
        for path in sorted(self.results_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            batch_id = path.stem
            batches.append(self._batch_summary(batch_id, path, latest_id == batch_id))
        for path in sorted(self.archive_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            batch_id = f"archive/{path.stem}"
            batches.append(self._batch_summary(batch_id, path, latest_id == batch_id))
        legacy = self.runtime_dir / "results.json"
        if legacy.exists():
            batches.append(self._batch_summary("legacy/results", legacy, False))
        return batches

    def load_results(self, batch_id: str = "") -> list[dict[str, Any]]:
        path = self._resolve_result_batch(batch_id)
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("rows") if isinstance(data, dict) else []
        return rows if isinstance(rows, list) else []

    def load_run_status(self) -> dict[str, Any]:
        active = self.runtime_dir / "active.json"
        if not active.exists():
            return {}
        try:
            active_data = json.loads(active.read_text(encoding="utf-8"))
        except Exception:
            return {}
        job_id = str(active_data.get("job_id") or "").strip()
        if not job_id:
            return active_data if isinstance(active_data, dict) else {}
        status_file = self.status_dir / f"{Path(job_id).name}.json"
        if not status_file.exists():
            return active_data if isinstance(active_data, dict) else {}
        try:
            status_data = json.loads(status_file.read_text(encoding="utf-8"))
        except Exception:
            return active_data if isinstance(active_data, dict) else {}
        return status_data if isinstance(status_data, dict) else {}

    def _archive_legacy_results(self) -> None:
        legacy = self.runtime_dir / "results.json"
        if not legacy.exists():
            return
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        target = self.archive_dir / f"legacy_results_{timestamp}.json"
        legacy.replace(target)

    def _latest_result_batch_id(self) -> str:
        latest = self.runtime_dir / "latest.json"
        if not latest.exists():
            return ""
        try:
            data = json.loads(latest.read_text(encoding="utf-8"))
        except Exception:
            return ""
        return str(data.get("job_id") or "")

    def _resolve_result_batch(self, batch_id: str) -> Path:
        clean_id = str(batch_id or "").strip()
        if clean_id.startswith("archive/"):
            return self.archive_dir / f"{Path(clean_id).name}.json"
        if clean_id.startswith("legacy/"):
            return self.runtime_dir / "results.json"
        if not clean_id:
            latest_id = self._latest_result_batch_id()
            if latest_id:
                return self.results_dir / f"{latest_id}.json"
            batches = self.load_result_batches()
            if batches:
                return Path(str(batches[0].get("path") or ""))
            return self.results_dir / "missing.json"
        return self.results_dir / f"{Path(clean_id).name}.json"

    def _batch_summary(self, batch_id: str, path: Path, is_latest: bool) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        rows = data.get("rows") if isinstance(data, dict) else []
        row_count = len(rows) if isinstance(rows, list) else 0
        updated_at = str(data.get("updated_at") or data.get("finished_at") or "")
        status = str(data.get("status") or "")
        status_marker = f" {status}" if status else ""
        latest_marker = " latest" if is_latest else ""
        return {
            "id": batch_id,
            "label": f"{batch_id} ({row_count} rows{status_marker}{latest_marker})",
            "path": str(path),
            "row_count": row_count,
            "updated_at": updated_at,
            "latest": is_latest,
            "status": status,
            "phase": str(data.get("phase") or ""),
        }

    def _validate(self, settings: EntrySieveSettings) -> None:
        if not settings.training_windows:
            raise ValueError("Entry Sieve requires at least one selected training window.")
        if not settings.validation_windows:
            raise ValueError("Entry Sieve requires at least one selected validation window.")
        if settings.auto_epochs:
            cap_text = str(settings.auto_epochs_cap or "").strip()
            if cap_text:
                try:
                    cap_value = int(cap_text)
                except (TypeError, ValueError):
                    raise ValueError("Entry Sieve auto epoch cap must be an integer. Use 0 or blank for no limit.") from None
                if cap_value < 0:
                    raise ValueError("Entry Sieve auto epoch cap must be >= 0. Use 0 or blank for no limit.")
        else:
            try:
                epochs = int(str(settings.epochs).strip())
            except (TypeError, ValueError):
                raise ValueError("Entry Sieve epochs must be a positive integer when Auto epochs is disabled.") from None
            if epochs < 1:
                raise ValueError("Entry Sieve epochs must be >= 1 when Auto epochs is disabled.")
        if settings.target_sweep_enabled:
            pairs = self._parse_target_pairs(settings.target_sweep_pairs)
            if not pairs:
                raise ValueError("Entry Sieve target sweep requires at least one TP/SL pair such as 1/1, 2/2, 3/2.")

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

    @staticmethod
    def _strategy_side(name: str) -> str:
        lowered = str(name or "").lower()
        if "short" in lowered:
            return "short"
        if "long" in lowered:
            return "long"
        return ""

    @staticmethod
    def _strategy_core_behavior(name: str) -> str:
        lowered = str(name or "").lower()
        if "volume_profile" in lowered or lowered.startswith("volume_"):
            return "volume"
        if "support" in lowered or "_sup" in lowered or "sup_" in lowered:
            return "support"
        if "resistance" in lowered or "_res" in lowered or "res_" in lowered:
            return "resistance"
        if any(token in lowered for token in ("breakout", "breakdown", "reclaim", "pullback", "retest", "reject", "fail")):
            return "pattern"
        return "other"

    @staticmethod
    def _parse_target_pairs(value: str) -> list[tuple[float, float]]:
        pairs: list[tuple[float, float]] = []
        text = str(value or "").replace(";", ",").replace("\n", ",")
        for raw_token in text.split(","):
            token = raw_token.strip()
            if not token:
                continue
            if "/" in token:
                left, right = token.split("/", 1)
            elif ":" in token:
                left, right = token.split(":", 1)
            else:
                parts = token.split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid target sweep pair '{token}'. Use TP/SL, for example 2/2.")
                left, right = parts
            try:
                take_profit = abs(float(str(left).strip().lstrip("+")))
                stoploss = abs(float(str(right).strip().lstrip("+")))
            except (TypeError, ValueError):
                raise ValueError(f"Invalid target sweep pair '{token}'. TP and SL must be numbers.") from None
            if take_profit <= 0.0 or stoploss <= 0.0:
                raise ValueError(f"Invalid target sweep pair '{token}'. TP and SL must be greater than 0.")
            pairs.append((take_profit, stoploss))
        return pairs

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
import fnmatch
import json
import os
from pathlib import Path
import sys
from typing import Any

from .entry_sieve_lock import live_entry_sieve_runs


EXCLUDED_STRATEGY_FILES = {
    "__init__.py",
    "test_entry_research_base.py",
    "PivotTrendlineMTFResearchStrategy.py",
}
MAX_BACKTEST_WORKERS = 20
SPEED_RUN_EPOCH_CAP = 120

DEFAULT_STRATEGY_BATCHES = [
    {
        "id": "all",
        "label": "All Sieve1",
        "include": ["sieve1_*.py"],
        "exclude": [],
    }
]


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
    return str(min(MAX_BACKTEST_WORKERS, max(1, count)))


def _default_repo_root(app_dir: Path) -> Path:
    return Path(app_dir).resolve().parent.parent


def _default_backtest_pythons(app_dir: Path) -> list[str]:
    worker_root = _default_repo_root(app_dir) / "runtime" / "venvs"
    return [
        str(worker_root / "freqtrade-backtest" / "Scripts" / "python.exe"),
        *[
            str(worker_root / f"freqtrade-backtest-{index:02d}" / "Scripts" / "python.exe")
            for index in range(1, MAX_BACKTEST_WORKERS)
        ],
    ]


@dataclass
class EntrySieveSettings:
    preset_name: str = "LauncherV2-auto"
    preset_file: str = "launcher_v2/config/presets.json"
    market_windows_file: str = "explorer/config/market_windows.json"
    auto_windows_file: str = "explorer/config/sieve_auto_windows.json"
    auto_window_mode: bool = True
    auto_window_count: str = "2"
    auto_validation_window: str = "full_cycle_2020_2026"
    training_windows: list[str] = field(default_factory=list)
    validation_windows: list[str] = field(default_factory=lambda: ["full_cycle_2020_2026"])
    epochs: str = "200"
    auto_epochs: bool = False
    auto_epochs_cap: str = ""
    random_state: str = ""
    sampling_seed: str = ""
    split_venv_pipeline: bool = False
    backtest_python_exe: str = ""
    backtest_python_exes: list[str] = field(default_factory=list)
    backtest_worker_count: str = "2"
    pipeline_handoff_dir: str = ""
    strategy_batch: str = "all"
    batch_queue_priority: str = "least_run_first"
    strategy_batch_file: str = "explorer/config/sieve_strategy_batches.json"
    strategy_filter: str = "sieve1_*.py"
    speed_run_mode: bool = False
    speed_pair_count: str = "5"
    take_profit_pct: str = "2"
    stoploss_pct: str = "2"
    target_sweep_enabled: bool = False
    target_sweep_pairs: str = ""


class EntrySieveService:
    def __init__(self, app_dir: Path, python_exe: str | None = None) -> None:
        self.app_dir = Path(app_dir)
        self.python_exe = python_exe or sys.executable
        self.runtime_dir = self.app_dir / "launcher_v2" / "runtime" / "entry_sieve"
        self.default_backtest_pythons = _default_backtest_pythons(self.app_dir)

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

    @property
    def queues_dir(self) -> Path:
        return self.runtime_dir / "queues"

    def resolve_path(self, value: str) -> Path:
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else (self.app_dir / path)

    def strategy_dir(self) -> Path:
        return (self.app_dir.parent / "strategies").resolve()

    def load_strategy_batches(self, strategy_batch_file: str = "explorer/config/sieve_strategy_batches.json") -> list[dict[str, Any]]:
        path = self.resolve_path(strategy_batch_file)
        if not path.exists():
            return [dict(batch) for batch in DEFAULT_STRATEGY_BATCHES]
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Entry Sieve strategy batch config is invalid JSON: {path}") from exc
        raw_batches = payload.get("batches") if isinstance(payload, dict) else payload
        if not isinstance(raw_batches, list):
            raise ValueError(f"Entry Sieve strategy batch config must contain a batches list: {path}")
        batches: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_batch in raw_batches:
            if not isinstance(raw_batch, dict):
                continue
            batch_id = str(raw_batch.get("id") or "").strip()
            if not batch_id or batch_id in seen:
                continue
            include = _split_list(raw_batch.get("include")) or ["*.py"]
            exclude = _split_list(raw_batch.get("exclude"))
            batches.append(
                {
                    "id": batch_id,
                    "label": str(raw_batch.get("label") or batch_id),
                    "include": include,
                    "exclude": exclude,
                    "description": str(raw_batch.get("description") or ""),
                }
            )
            seen.add(batch_id)
        if not batches:
            raise ValueError(f"Entry Sieve strategy batch config has no usable batches: {path}")
        return batches

    def _strategy_batch_definition(self, strategy_batch: str, strategy_batch_file: str) -> dict[str, Any]:
        batches = self.load_strategy_batches(strategy_batch_file)
        selected = str(strategy_batch or "all").strip() or "all"
        for batch in batches:
            if str(batch.get("id") or "") == selected:
                return batch
        available = ", ".join(str(batch.get("id") or "") for batch in batches)
        raise ValueError(f"Unknown Entry Sieve strategy batch '{selected}'. Available batches: {available}")

    def discover_strategies(
        self,
        strategy_filter: str = "*.py",
        *,
        strategy_batch: str = "all",
        strategy_batch_file: str = "explorer/config/sieve_strategy_batches.json",
    ) -> list[dict[str, str]]:
        patterns = _split_list(strategy_filter) or ["*.py"]
        batch = self._strategy_batch_definition(strategy_batch, strategy_batch_file)
        include_patterns = _split_list(batch.get("include")) or ["*.py"]
        exclude_patterns = _split_list(batch.get("exclude"))
        rows: list[dict[str, str]] = []
        for path in sorted(self.strategy_dir().glob("*.py")):
            if path.name in EXCLUDED_STRATEGY_FILES:
                continue
            if not any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns):
                continue
            if not any(fnmatch.fnmatch(path.name, pattern) for pattern in include_patterns):
                continue
            if any(fnmatch.fnmatch(path.name, pattern) for pattern in exclude_patterns):
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
        settings = self._effective_settings(settings)
        self._validate(settings)
        batch = self._strategy_batch_definition(settings.strategy_batch, settings.strategy_batch_file)
        strategies = self.discover_strategies(
            settings.strategy_filter,
            strategy_batch=settings.strategy_batch,
            strategy_batch_file=settings.strategy_batch_file,
        )
        if not strategies:
            raise ValueError("Entry Sieve found no top-level strategy files for the current filter.")
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._archive_legacy_results()
        batch_token = self._job_id_batch_token(str(settings.strategy_batch or "all"))
        job_id = datetime.now().strftime(f"%Y%m%dT%H%M%S_entry_{batch_token}")
        job_path = self.jobs_dir / f"{job_id}.json"
        job = {
            "schema_version": 1,
            "job_id": job_id,
            "created_at": datetime.now().astimezone().isoformat(),
            "preset_name": settings.preset_name,
            "preset_file": str(self.resolve_path(settings.preset_file)),
            "market_windows_file": str(self.resolve_path(settings.market_windows_file)),
            "auto_windows_file": str(self.resolve_path(settings.auto_windows_file)),
            "auto_window_mode": bool(settings.auto_window_mode),
            "auto_window_count": str(settings.auto_window_count),
            "auto_validation_window": str(settings.auto_validation_window or "full_cycle_2020_2026"),
            "training_windows": list(settings.training_windows),
            "validation_windows": ["full_cycle_2020_2026"] if settings.auto_window_mode else list(settings.validation_windows),
            "epochs": str(settings.epochs),
            "auto_epochs": bool(settings.auto_epochs),
            "auto_epochs_cap": str(settings.auto_epochs_cap),
            "speed_run_mode": bool(settings.speed_run_mode),
            "speed_pair_count": str(settings.speed_pair_count),
            "python_exe": str(self.python_exe),
            "random_state": str(settings.random_state),
            "sampling_seed": str(settings.sampling_seed),
            "split_venv_pipeline": bool(settings.split_venv_pipeline),
            "backtest_python_exe": str(settings.backtest_python_exe),
            "backtest_python_exes": _dedupe_paths([
                str(settings.backtest_python_exe),
                *list(settings.backtest_python_exes),
                *self.default_backtest_pythons,
            ]),
            "backtest_worker_count": _normalize_worker_count(settings.backtest_worker_count, "2"),
            "pipeline_handoff_dir": str(settings.pipeline_handoff_dir),
            "strategy_batch": str(settings.strategy_batch or "all"),
            "strategy_batch_label": str(batch.get("label") or settings.strategy_batch or "all"),
            "strategy_batch_file": str(self.resolve_path(settings.strategy_batch_file)),
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

    @staticmethod
    def _effective_settings(settings: EntrySieveSettings) -> EntrySieveSettings:
        if not bool(settings.speed_run_mode):
            return settings
        return replace(
            settings,
            auto_window_mode=True,
            auto_window_count="1",
            auto_epochs_cap=str(SPEED_RUN_EPOCH_CAP),
            target_sweep_enabled=False,
            target_sweep_pairs="",
        )

    def build_command(self, settings: EntrySieveSettings) -> list[str]:
        job_path = self.build_job(settings)
        return [self.python_exe, "-u", "-m", "launcher_v2.services.entry_sieve_runner", "--job-file", str(job_path)]

    def build_batch_queue(
        self,
        settings: EntrySieveSettings,
        batch_ids: list[str],
        *,
        wait_for_active: bool = True,
        poll_seconds: int = 60,
    ) -> Path:
        settings = self._effective_settings(settings)
        batches = self.load_strategy_batches(settings.strategy_batch_file)
        known = {str(batch.get("id") or ""): batch for batch in batches}
        resolved_ids: list[str] = []
        for raw_batch_id in batch_ids:
            batch_id = str(raw_batch_id or "").strip()
            if not batch_id or batch_id == "all":
                continue
            if batch_id not in known:
                raise ValueError(f"Unknown Entry Sieve queue batch '{batch_id}'.")
            if batch_id not in resolved_ids:
                resolved_ids.append(batch_id)
        if not resolved_ids:
            raise ValueError("Entry Sieve batch queue needs at least one non-all batch.")
        resolved_ids = self._prioritized_queue_batches(resolved_ids, settings.batch_queue_priority)
        self.queues_dir.mkdir(parents=True, exist_ok=True)
        queue_id = datetime.now().strftime("entry_sieve_queue_%Y%m%dT%H%M%S")
        queue_path = self.queues_dir / f"{queue_id}.json"
        queue = {
            "schema_version": 1,
            "queue_id": queue_id,
            "created_at": datetime.now().astimezone().isoformat(),
            "status": "pending",
            "app_dir": str(self.app_dir),
            "python_exe": str(self.python_exe),
            "settings": asdict(settings),
            "batch_ids": resolved_ids,
            "wait_for_active": bool(wait_for_active),
            "poll_seconds": max(10, int(poll_seconds)),
            "completed_batches": [],
            "failed_batches": [],
        }
        queue_path.write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
        return queue_path

    def build_batch_queue_command(self, settings: EntrySieveSettings, batch_ids: list[str]) -> list[str]:
        queue_path = self.build_batch_queue(settings, batch_ids)
        return [self.python_exe, "-u", "-m", "launcher_v2.services.entry_sieve_batch_queue_runner", "--queue-file", str(queue_path)]

    def load_result_batches(self) -> list[dict[str, Any]]:
        batches: list[dict[str, Any]] = []
        latest_id = self._latest_result_batch_id()
        for path in sorted(self.results_dir.glob("*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True):
            if self._count_jsonl_rows(path) == 0:
                continue
            batch_id = path.stem
            batches.append(self._batch_summary(batch_id, path, latest_id == batch_id))
        for path in sorted(self.results_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            if path.name.endswith(".summary.json"):
                continue
            if (self.results_dir / f"{path.stem}.jsonl").exists():
                continue
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
        metadata = self._result_batch_metadata(path)
        if path.suffix.lower() == ".jsonl":
            rows: list[dict[str, Any]] = []
            changed = False
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        row.pop("metrics", None)
                        changed = self._row_missing_result_metadata(row, metadata) or changed
                        self._apply_result_metadata(row, metadata)
                        rows.append(row)
            if changed:
                self._backfill_jsonl_metadata(path, rows, metadata)
            return rows
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("rows") if isinstance(data, dict) else []
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    row.pop("metrics", None)
                    self._apply_result_metadata(row, metadata)
        return rows if isinstance(rows, list) else []

    def load_results_many(self, batch_ids: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for batch_id in batch_ids:
            path = self._resolve_result_batch(batch_id)
            try:
                key = str(path.resolve()).lower()
            except OSError:
                key = str(path).lower()
            if key in seen_paths:
                continue
            seen_paths.add(key)
            rows.extend(self.load_results(batch_id))
        return rows

    def live_runs(self, *, exclude_job_id: str = "") -> list[dict[str, Any]]:
        return live_entry_sieve_runs(self.runtime_dir, exclude_job_id=exclude_job_id)

    def busy_message(self, *, exclude_job_id: str = "") -> str:
        runs = self.live_runs(exclude_job_id=exclude_job_id)
        if not runs:
            return ""
        labels = [
            f"{run.get('job_id')} (pid {run.get('pid')}, {run.get('phase') or run.get('source') or 'running'})"
            for run in runs
        ]
        return "Entry Sieve is already running: " + "; ".join(labels)

    def delete_result_batches(self, batch_ids: list[str]) -> list[str]:
        deleted: list[str] = []
        deleted_job_ids: set[str] = set()
        running_job_ids = {str(run.get("job_id") or "").strip() for run in self.live_runs()}
        deletions: list[tuple[str, list[Path]]] = []
        for batch_id in batch_ids:
            path = self._resolve_result_batch(batch_id)
            if not path.exists():
                continue
            if not self._is_deletable_result_path(path):
                raise ValueError(f"Refusing to delete outside Entry Sieve results: {path}")
            job_id = path.stem
            if job_id in running_job_ids:
                raise ValueError(f"Refusing to delete active running result batch: {job_id}")
            deletions.append((job_id, self._related_result_paths(path)))
        for job_id, related_paths in deletions:
            for related_path in related_paths:
                if related_path.exists():
                    related_path.unlink()
                    deleted.append(str(related_path))
            deleted_job_ids.add(job_id)
        if deleted_job_ids:
            self._clear_latest_if_deleted(deleted_job_ids)
        return deleted

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
        if not isinstance(status_data, dict):
            return {}
        for key, value in self._job_metadata(job_id).items():
            if key not in status_data or status_data.get(key) in ("", None):
                status_data[key] = value
        return self._reconcile_run_status(job_id, status_data)

    def _reconcile_run_status(self, job_id: str, status_data: dict[str, Any]) -> dict[str, Any]:
        if str(status_data.get("status") or "").lower() != "running":
            return status_data
        try:
            pid = int(status_data.get("pid") or 0)
        except (TypeError, ValueError):
            pid = 0
        if pid > 0 and self._process_is_running(pid):
            return status_data

        now = datetime.now().astimezone().isoformat()
        reconciled = dict(status_data)
        reconciled.update(
            {
                "status": "stopped",
                "phase": "stale",
                "updated_at": now,
                "message": "Entry Sieve runner process is not running; marked stale.",
            }
        )
        self._save_json(self.status_dir / f"{Path(job_id).name}.json", reconciled)
        self._mark_result_batch_status(job_id, "stopped", "stale", now)
        self._mark_pointer_status(self.runtime_dir / "active.json", job_id, "stopped", "stale", now)
        self._mark_pointer_status(self.runtime_dir / "latest.json", job_id, "stopped", "stale", now)
        return reconciled

    @staticmethod
    def _process_is_running(pid: int) -> bool:
        if pid <= 0:
            return False
        if sys.platform == "win32":
            import ctypes

            synchronize = 0x00100000
            wait_timeout = 0x00000102
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(synchronize, False, pid)
            if not handle:
                return False
            try:
                return kernel32.WaitForSingleObject(handle, 0) == wait_timeout
            finally:
                kernel32.CloseHandle(handle)
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    @staticmethod
    def _save_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _mark_result_batch_status(self, job_id: str, status: str, phase: str, updated_at: str) -> None:
        clean_id = Path(job_id).name
        jsonl_path = self.results_dir / f"{clean_id}.jsonl"
        legacy_path = self.results_dir / f"{clean_id}.json"
        result_path = jsonl_path if jsonl_path.exists() else legacy_path
        if not result_path.exists():
            return
        summary_path = self.results_dir / f"{clean_id}.summary.json"
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        row_count = data.get("row_count")
        if row_count in (None, "") and result_path.suffix.lower() == ".jsonl":
            row_count = self._count_jsonl_rows(result_path)
        data["status"] = status
        data["phase"] = phase
        data["updated_at"] = updated_at
        data.setdefault("schema_version", 3)
        data.setdefault("storage", result_path.suffix.lower().lstrip("."))
        data.setdefault("job_id", clean_id)
        data.setdefault("path", str(result_path))
        data.setdefault("summary_path", str(summary_path))
        if row_count not in (None, ""):
            data["row_count"] = row_count
        self._save_json(summary_path, data)

    def _mark_pointer_status(self, path: Path, job_id: str, status: str, phase: str, updated_at: str) -> None:
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, dict) or str(data.get("job_id") or "") != job_id:
            return
        data["status"] = status
        data["phase"] = phase
        data["updated_at"] = updated_at
        self._save_json(path, data)

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
                jsonl_path = self.results_dir / f"{latest_id}.jsonl"
                return jsonl_path if jsonl_path.exists() else self.results_dir / f"{latest_id}.json"
            batches = self.load_result_batches()
            if batches:
                return Path(str(batches[0].get("path") or ""))
            return self.results_dir / "missing.json"
        name = Path(clean_id).name
        jsonl_path = self.results_dir / f"{name}.jsonl"
        return jsonl_path if jsonl_path.exists() else self.results_dir / f"{name}.json"

    def _is_deletable_result_path(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
            results_dir = self.results_dir.resolve()
            archive_dir = self.archive_dir.resolve()
            legacy = (self.runtime_dir / "results.json").resolve()
        except OSError:
            return False
        return resolved.parent in {results_dir, archive_dir} or resolved == legacy

    def _related_result_paths(self, path: Path) -> list[Path]:
        paths = [path]
        if path.parent == self.results_dir:
            paths.append(self.results_dir / f"{path.stem}.summary.json")
        return paths

    def _clear_latest_if_deleted(self, job_ids: set[str]) -> None:
        latest = self.runtime_dir / "latest.json"
        if not latest.exists():
            return
        try:
            data = json.loads(latest.read_text(encoding="utf-8"))
        except Exception:
            return
        if isinstance(data, dict) and str(data.get("job_id") or "") in job_ids:
            latest.unlink()

    def _batch_summary(self, batch_id: str, path: Path, is_latest: bool) -> dict[str, Any]:
        data = self._summary_metadata(path)
        if path.suffix.lower() == ".jsonl":
            row_count = data.get("row_count")
            if row_count in (None, ""):
                row_count = self._count_jsonl_rows(path)
        else:
            row_count = data.get("row_count")
            if row_count in (None, ""):
                row_count = "legacy json"
        updated_at = str(data.get("updated_at") or data.get("finished_at") or "")
        status = str(data.get("status") or "")
        batch_label = str(data.get("strategy_batch_label") or data.get("strategy_batch") or "")
        speed = " speed" if data.get("speed_run_mode") else ""
        status_marker = f" {status}" if status else ""
        latest_marker = " latest" if is_latest else ""
        batch_marker = f"{batch_label}{speed} | " if batch_label else ""
        return {
            "id": batch_id,
            "label": f"{batch_marker}{batch_id} ({row_count} rows{status_marker}{latest_marker})",
            "path": str(path),
            "row_count": row_count,
            "updated_at": updated_at,
            "latest": is_latest,
            "status": status,
            "phase": str(data.get("phase") or ""),
            "strategy_batch": str(data.get("strategy_batch") or ""),
            "strategy_batch_label": batch_label,
            "speed_run_mode": bool(data.get("speed_run_mode")),
        }

    def _summary_metadata(self, result_path: Path) -> dict[str, Any]:
        summary_path = result_path.with_suffix(".summary.json") if result_path.suffix.lower() == ".jsonl" else self.results_dir / f"{result_path.stem}.summary.json"
        data: dict[str, Any] = {}
        if summary_path.exists():
            try:
                loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                loaded = {}
            if isinstance(loaded, dict):
                data.update(loaded)
        job_data = self._job_metadata(result_path.stem)
        for key, value in job_data.items():
            if key not in data or data.get(key) in ("", None):
                data[key] = value
        if data:
            return data
        if result_path.suffix.lower() == ".jsonl":
            return {"row_count": self._count_jsonl_rows(result_path), "updated_at": datetime.fromtimestamp(result_path.stat().st_mtime).astimezone().isoformat()}
        return {"updated_at": datetime.fromtimestamp(result_path.stat().st_mtime).astimezone().isoformat()}

    def _result_batch_metadata(self, result_path: Path) -> dict[str, Any]:
        data = self._summary_metadata(result_path)
        return {
            "strategy_batch": str(data.get("strategy_batch") or ""),
            "strategy_batch_label": str(data.get("strategy_batch_label") or data.get("strategy_batch") or ""),
            "strategy_filter": str(data.get("strategy_filter") or ""),
            "speed_run_mode": bool(data.get("speed_run_mode")),
            "speed_pair_count": str(data.get("speed_pair_count") or ""),
            "auto_window_mode": bool(data.get("auto_window_mode")),
            "auto_window_count": str(data.get("auto_window_count") or ""),
            "target_sweep_enabled": bool(data.get("target_sweep_enabled")),
        }

    @staticmethod
    def _apply_result_metadata(row: dict[str, Any], metadata: dict[str, Any]) -> None:
        for key, value in metadata.items():
            if key not in row or row.get(key) in ("", None):
                row[key] = value

    @staticmethod
    def _row_missing_result_metadata(row: dict[str, Any], metadata: dict[str, Any]) -> bool:
        return any(value not in ("", None) and (key not in row or row.get(key) in ("", None)) for key, value in metadata.items())

    def _backfill_jsonl_metadata(self, path: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
        if not rows or not metadata.get("strategy_batch"):
            return
        summary = self._summary_metadata(path)
        if str(summary.get("status") or "").lower() == "running":
            return
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=False, default=str, separators=(",", ":")) + "\n")
        summary_path = path.with_suffix(".summary.json")
        if summary_path.exists():
            summary.update(metadata)
            self._save_json(summary_path, summary)

    def _job_metadata(self, job_id: str) -> dict[str, Any]:
        job_path = self.jobs_dir / f"{Path(str(job_id)).name}.json"
        if not job_path.exists():
            return {}
        try:
            data = json.loads(job_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        batch_id = str(data.get("strategy_batch") or "").strip()
        batch_label = str(data.get("strategy_batch_label") or batch_id).strip()
        return {
            "strategy_batch": batch_id,
            "strategy_batch_label": batch_label,
            "strategy_filter": str(data.get("strategy_filter") or ""),
            "speed_run_mode": bool(data.get("speed_run_mode")),
            "speed_pair_count": str(data.get("speed_pair_count") or ""),
            "auto_window_mode": bool(data.get("auto_window_mode")),
            "auto_window_count": str(data.get("auto_window_count") or ""),
            "target_sweep_enabled": bool(data.get("target_sweep_enabled")),
        }

    @staticmethod
    def _count_jsonl_rows(path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    @staticmethod
    def _job_id_batch_token(batch_id: str) -> str:
        tokens = {
            "volume_profile": "vol_profile",
            "geometry": "pattern_geometry",
            "continuation_patterns": "pattern_continuation",
            "reversal_patterns": "pattern_reversal",
            "structure_levels": "structure",
            "tlv2_vp": "tlv2_vp",
            "tlv2_boschoch": "tlv2_boschoch",
            "vp_prior_levels": "vp_prior",
            "market_state_pressure": "pressure",
            "multi_confluence": "confluence",
            "small_concepts": "other",
            "avwap": "avwap",
            "zones": "zones",
            "all": "all",
        }
        raw = str(batch_id or "all").strip().lower() or "all"
        token = tokens.get(raw, raw)
        return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in token).strip("_") or "all"

    def _prioritized_queue_batches(self, batch_ids: list[str], priority: str) -> list[str]:
        mode = str(priority or "configured").strip().lower()
        if mode not in {"least_run_first", "least_runs_first"}:
            return batch_ids
        counts = self._result_run_counts_by_batch()
        return sorted(batch_ids, key=lambda batch_id: (counts.get(batch_id, 0), batch_ids.index(batch_id)))

    def _result_run_counts_by_batch(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for path in self.results_dir.glob("*.jsonl"):
            if self._count_jsonl_rows(path) == 0:
                continue
            metadata = self._summary_metadata(path)
            batch_id = str(metadata.get("strategy_batch") or "").strip()
            if not batch_id:
                continue
            counts[batch_id] = counts.get(batch_id, 0) + 1
        return counts

    def _validate(self, settings: EntrySieveSettings) -> None:
        if settings.auto_window_mode:
            try:
                auto_count = int(str(settings.auto_window_count or "").strip())
            except (TypeError, ValueError):
                raise ValueError("Entry Sieve auto windows per file must be 1, 2, or 3.") from None
            if auto_count < 1 or auto_count > 3:
                raise ValueError("Entry Sieve auto windows per file must be 1, 2, or 3.")
            if not self.resolve_path(settings.auto_windows_file).exists():
                raise ValueError(f"Entry Sieve auto window manifest does not exist: {self.resolve_path(settings.auto_windows_file)}")
        elif not settings.training_windows:
            raise ValueError("Entry Sieve requires at least one selected training window.")
        if not settings.auto_window_mode and not settings.validation_windows:
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
        if settings.speed_run_mode:
            try:
                pair_count = int(str(settings.speed_pair_count or "").strip())
            except (TypeError, ValueError):
                raise ValueError("Entry Sieve speed run pair count must be a positive integer.") from None
            if pair_count < 1:
                raise ValueError("Entry Sieve speed run pair count must be >= 1.")
        if settings.split_venv_pipeline:
            try:
                worker_count = int(str(settings.backtest_worker_count or "").strip())
            except (TypeError, ValueError):
                raise ValueError(f"Entry Sieve backtest workers must be an integer from 1 to {MAX_BACKTEST_WORKERS}.") from None
            if worker_count < 1 or worker_count > MAX_BACKTEST_WORKERS:
                raise ValueError(f"Entry Sieve backtest workers must be between 1 and {MAX_BACKTEST_WORKERS}.")

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

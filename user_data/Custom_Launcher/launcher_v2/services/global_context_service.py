from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import csv
import json
import os
import sqlite3
import subprocess
import sys

from .collector_service import is_process_running, open_path, parse_minutes_to_seconds, utc_now, utf8_subprocess_env
from research.collectors.global_context_collector import load_fred_api_key_from_file
from research.collectors.global_context_store import init_db


class GlobalContextService:
    def __init__(self, app_dir: Path, python_exe: str | None = None) -> None:
        self.app_dir = Path(app_dir)
        self.python_exe = python_exe or sys.executable

    def app_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.app_dir / path

    def _resolve_data_dir(self, value: Any) -> Path:
        text = str(value or "").strip()
        default_dir = self.app_path("../research_news_data/global_context")
        return Path(text) if text else default_dir

    def _resolve_config_path(self, value: Any) -> Path:
        text = str(value or "").strip()
        default_config = self.app_path("research/config/global_context_sources.json")
        if not text:
            return default_config
        path = Path(text)
        if not path.exists() and path.name.lower() == default_config.name.lower():
            return default_config
        return path

    def _resolve_db_path(self, value: Any, data_dir: Path) -> Path:
        text = str(value or "").strip()
        return Path(text) if text else data_dir / "global_context.sqlite"

    def paths(self, state: dict[str, Any]) -> dict[str, Path]:
        data_dir = self._resolve_data_dir(state.get("data_dir"))
        config_path = self._resolve_config_path(state.get("config_path"))
        db_path = self._resolve_db_path(state.get("db_path"), data_dir)
        return {
            "collector": self.app_path("research/collectors/global_context_collector.py"),
            "config": config_path,
            "data_dir": data_dir,
            "db": db_path,
            "status": data_dir / "collector_status.json",
            "pid": data_dir / "collector.pid",
            "stop": data_dir / "collector.stop",
            "log": data_dir / "logs" / "global_context_collector.log",
        }

    def build_command(self, state: dict[str, Any]) -> list[str]:
        paths = self.paths(state)
        if not paths["collector"].exists():
            raise FileNotFoundError(paths["collector"])
        command = [
            self.python_exe,
            "-u",
            "-m",
            "research.collectors.global_context_collector",
            "--config",
            str(paths["config"]),
            "--data-dir",
            str(paths["data_dir"]),
            "--db",
            str(paths["db"]),
            "--status-file",
            str(paths["status"]),
            "--pid-file",
            str(paths["pid"]),
            "--log-file",
            str(paths["log"]),
            "--stop-file",
            str(paths["stop"]),
            "--interval-seconds",
            str(parse_minutes_to_seconds(str(state.get("interval_minutes") or ""), 30)),
        ]
        key_file = str(state.get("key_file") or "").strip()
        if key_file:
            command.extend(["--key-file", key_file])
        fred_key_json_path = str(state.get("fred_key_json_path") or "").strip()
        if fred_key_json_path:
            command.extend(["--fred-key-json-path", fred_key_json_path])
        if state.get("enable_fred"):
            command.append("--enable-fred")
        if state.get("once"):
            command.append("--once")
        return command

    def start_detached(self, state: dict[str, Any]) -> int:
        paths = self.paths(state)
        paths["data_dir"].mkdir(parents=True, exist_ok=True)
        paths["log"].parent.mkdir(parents=True, exist_ok=True)
        try:
            paths["stop"].unlink(missing_ok=True)
        except Exception:
            pass
        command = self.build_command(state)
        with open(paths["log"], "ab") as log_handle:
            kwargs: dict[str, Any] = {
                "stdin": subprocess.DEVNULL,
                "stdout": log_handle,
                "stderr": subprocess.STDOUT,
                "cwd": str(self.app_dir),
                "env": utf8_subprocess_env(),
                "close_fds": True,
            }
            if os.name == "nt":
                kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
            else:
                kwargs["start_new_session"] = True
            process = subprocess.Popen(command, **kwargs)
        return int(process.pid)

    def request_stop(self, state: dict[str, Any]) -> Path:
        paths = self.paths(state)
        paths["stop"].parent.mkdir(parents=True, exist_ok=True)
        paths["stop"].write_text(utc_now() + "\n", encoding="utf-8")
        return paths["stop"]

    def read_status(self, state: dict[str, Any]) -> dict[str, Any]:
        paths = self.paths(state)
        status: dict[str, Any] = {}
        if paths["status"].exists():
            try:
                loaded = json.loads(paths["status"].read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    status = loaded
            except Exception:
                status = {}
        if paths["pid"].exists():
            try:
                pid_text = paths["pid"].read_text(encoding="utf-8").strip()
                status["pid_text"] = pid_text
                if pid_text and not is_process_running(int(pid_text)) and str(status.get("status") or "").lower() == "running":
                    status["status"] = "stale/unknown"
            except Exception:
                pass
        return status

    def _latest_context_records(self, state: dict[str, Any], source_groups: set[str] | None = None) -> list[sqlite3.Row]:
        paths = self.paths(state)
        if not paths["db"].exists():
            return []
        init_db(paths["db"])
        with sqlite3.connect(str(paths["db"]), timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT t.*
                FROM global_context_ticks t
                JOIN (
                    SELECT source_id, metric_key, MAX(id) AS latest_id
                    FROM global_context_ticks
                    GROUP BY source_id, metric_key
                ) latest ON latest.latest_id = t.id
                ORDER BY t.source_group, t.source_id, t.metric_key
                """
            ).fetchall()
        if source_groups is not None:
            rows = [row for row in rows if str(row["source_group"]) in source_groups]
        return rows

    def latest_context_rows(self, state: dict[str, Any], source_groups: set[str] | None = None) -> list[tuple[Any, ...]]:
        rows = self._latest_context_records(state, source_groups=source_groups)
        return [
            (
                row["source_id"],
                row["source_group"],
                row["metric_key"],
                _display_signal(row["signal"], row["score"]),
                _fmt(row["score"]),
                _fmt(row["source_score"]),
                _fmt(row["calc_score"]),
                _fmt(row["value"]),
                row["unit"] or "",
                row["notes"] or "",
                row["source_ts"] or "",
                row["ts"] or "",
            )
            for row in rows
        ]

    def score_summary(self, state: dict[str, Any]) -> dict[str, Any]:
        rows = self._latest_context_records(state)
        scored = [row for row in rows if _float_or_none(row["score"]) is not None]
        valid_scores = [score for score in (_float_or_none(row["score"]) for row in scored) if score is not None]
        mean_score = (sum(valid_scores) / len(valid_scores)) if valid_scores else None
        return {
            "average_score": _fmt(mean_score),
            "min_score": _fmt(min(valid_scores) if valid_scores else None),
            "max_score": _fmt(max(valid_scores) if valid_scores else None),
            "mean_score": _fmt(mean_score),
        }

    def score_detail_rows(self, state: dict[str, Any]) -> list[tuple[Any, ...]]:
        rows = self._latest_context_records(state)
        scored = [row for row in rows if _float_or_none(row["score"]) is not None]
        return [
            (
                row["metric_key"],
                _fmt(row["score"]),
            )
            for row in scored
        ]

    def summary_rows(self, state: dict[str, Any]) -> list[tuple[Any, ...]]:
        rows = self._latest_context_records(state)
        groups: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            groups.setdefault(str(row["source_group"] or "unknown"), []).append(row)
        output: list[tuple[Any, ...]] = []
        for group, group_rows in sorted(groups.items()):
            scores = [_float_or_none(row["score"]) for row in group_rows]
            valid_scores = [score for score in scores if score is not None]
            avg_score = (sum(valid_scores) / len(valid_scores)) if valid_scores else None
            signal = _score_signal(avg_score)
            notes = " | ".join(str(row["notes"] or "") for row in group_rows[:3])
            latest_ts = max((str(row["ts"] or "") for row in group_rows), default="")
            output.append((group, signal, _fmt(avg_score), len(group_rows), notes, latest_ts))
        return output

    def source_health_rows(self, state: dict[str, Any], source_groups: set[str] | None = None) -> list[tuple[Any, ...]]:
        paths = self.paths(state)
        if not paths["db"].exists():
            return []
        init_db(paths["db"])
        with sqlite3.connect(str(paths["db"]), timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT source_id, source_group, source_type, enabled, market_relevance, last_success_at,
                       last_failure_at, last_error, last_score, last_source_score, last_calc_score, last_signal, last_notes, updated_at
                FROM context_sources
                ORDER BY source_group, source_id
                """
            ).fetchall()
        if source_groups is not None:
            rows = [row for row in rows if str(row["source_group"]) in source_groups]
        return [
            (
                row["source_id"],
                row["source_group"],
                row["source_type"],
                row["enabled"],
                row["market_relevance"],
                row["last_success_at"],
                row["last_failure_at"],
                row["last_error"],
                _fmt(row["last_score"]),
                _fmt(row["last_source_score"]),
                _fmt(row["last_calc_score"]),
                _display_signal(row["last_signal"], row["last_score"]),
                row["last_notes"] or "",
                row["updated_at"],
            )
            for row in rows
        ]

    def export_latest_csv(self, state: dict[str, Any]) -> tuple[Path, int]:
        rows = self.latest_context_rows(state)
        paths = self.paths(state)
        export_dir = paths["data_dir"] / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        output = export_dir / f"global_context_latest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        headers = ["source_id", "source_group", "metric_key", "signal", "effective_score", "source_score", "calc_score", "value", "unit", "notes", "source_ts", "collected_at"]
        with open(output, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerows(rows)
        return output, len(rows)

    def open_data_folder(self, state: dict[str, Any]) -> None:
        paths = self.paths(state)
        paths["data_dir"].mkdir(parents=True, exist_ok=True)
        open_path(paths["data_dir"])

    def open_log(self, state: dict[str, Any]) -> None:
        paths = self.paths(state)
        paths["log"].parent.mkdir(parents=True, exist_ok=True)
        paths["log"].touch(exist_ok=True)
        open_path(paths["log"])

    def fred_key_status(self, state: dict[str, Any]) -> str:
        key_file_text = str(state.get("key_file") or "").strip()
        if not key_file_text:
            return "No FRED key file selected"
        key_file = Path(key_file_text)
        if not key_file.exists():
            return "FRED key file not found"
        key = load_fred_api_key_from_file(key_file, str(state.get("fred_key_json_path") or ""))
        if not key:
            return "FRED key not found at selected JSON path"
        return f"FRED key detected ({len(key)} chars)"


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return "-"
    return f"{number:.6g}"


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _score_signal(score: Any) -> str:
    value = _float_or_none(score)
    if value is None:
        return "Neutral"
    if value >= 65:
        return "Greed"
    if value <= 35:
        return "Fear"
    return "Neutral"


def _display_signal(signal: Any, score: Any) -> str:
    text = str(signal or "").strip()
    replacements = {"risk_off": "Fear", "risk_on": "Greed", "neutral": "Neutral"}
    if text in replacements:
        return replacements[text]
    return text or _score_signal(score)

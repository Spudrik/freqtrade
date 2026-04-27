from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import csv
import json
import os
import sqlite3
import subprocess
import sys
import webbrowser
from urllib.parse import urlparse


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utf8_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def parse_minutes_to_seconds(value: str, default_minutes: int) -> int:
    text = str(value or "").strip()
    minutes = default_minutes
    if text:
        try:
            minutes = int(round(float(text)))
        except Exception:
            minutes = default_minutes
    return max(1, minutes) * 60


def is_process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
    except Exception:
        return False


def open_path(path: Path) -> None:
    path = path.resolve()
    if path.is_dir():
        webbrowser.open(path.as_uri())
        return
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        webbrowser.open(path.as_uri())


@dataclass(frozen=True)
class CollectorProfile:
    key: str
    label: str
    collector_file: str
    sources_file: str
    data_dir_name: str
    db_file: str
    log_file: str
    default_interval_minutes: int


NEWS_PROFILE = CollectorProfile(
    key="news",
    label="News Lab",
    collector_file="research/collectors/news_research_collector.py",
    sources_file="research/config/news_research_sources.json",
    data_dir_name="../research_news_data/news",
    db_file="news_events.sqlite",
    log_file="news_collector.log",
    default_interval_minutes=15,
)


WEB_PROFILE = CollectorProfile(
    key="web",
    label="Web Lab",
    collector_file="research/collectors/web_research_collector.py",
    sources_file="research/config/web_research_sources.json",
    data_dir_name="../research_news_data/web",
    db_file="web_events.sqlite",
    log_file="web_collector.log",
    default_interval_minutes=360,
)


class ResearchCollectorService:
    def __init__(self, app_dir: Path, python_exe: str | None = None) -> None:
        self.app_dir = Path(app_dir)
        self.python_exe = python_exe or sys.executable

    def app_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.app_dir / path

    def _resolve_data_dir(self, profile: CollectorProfile, data_dir: str) -> Path:
        text = str(data_dir or "").strip()
        default_dir = self.app_path(profile.data_dir_name)
        if not text:
            return default_dir
        return Path(text)

    def _resolve_config_path(self, profile: CollectorProfile, config_path: str) -> Path:
        text = str(config_path or "").strip()
        default_config = self.app_path(profile.sources_file)
        if not text:
            return default_config
        path = Path(text)
        if not path.exists() and path.name.lower() == default_config.name.lower():
            return default_config
        return path

    def _resolve_db_path(self, profile: CollectorProfile, db_path: str, resolved_data_dir: Path) -> Path:
        text = str(db_path or "").strip()
        default_db = resolved_data_dir / profile.db_file
        if not text:
            return default_db
        return Path(text)

    def paths(self, profile: CollectorProfile, data_dir: str, config_path: str, db_path: str) -> dict[str, Path]:
        resolved_data_dir = self._resolve_data_dir(profile, data_dir)
        resolved_config = self._resolve_config_path(profile, config_path)
        resolved_db = self._resolve_db_path(profile, db_path, resolved_data_dir)
        return {
            "collector": self.app_path(profile.collector_file),
            "config": resolved_config,
            "data_dir": resolved_data_dir,
            "db": resolved_db,
            "status": resolved_data_dir / "collector_status.json",
            "pid": resolved_data_dir / "collector.pid",
            "stop": resolved_data_dir / "collector.stop",
            "log": resolved_data_dir / "logs" / profile.log_file,
        }

    def build_command(self, profile: CollectorProfile, state: dict[str, Any]) -> list[str]:
        paths = self.paths(profile, str(state.get("data_dir") or ""), str(state.get("config_path") or ""), str(state.get("db_path") or ""))
        if not paths["collector"].exists():
            raise FileNotFoundError(paths["collector"])
        command = [
            self.python_exe,
            "-u",
            "-m",
            f"research.collectors.{Path(profile.collector_file).stem}",
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
            str(parse_minutes_to_seconds(str(state.get("interval_minutes") or ""), profile.default_interval_minutes)),
        ]
        if state.get("once"):
            command.append("--once")
        return command

    def start_detached(self, profile: CollectorProfile, state: dict[str, Any]) -> int:
        paths = self.paths(profile, str(state.get("data_dir") or ""), str(state.get("config_path") or ""), str(state.get("db_path") or ""))
        paths["data_dir"].mkdir(parents=True, exist_ok=True)
        paths["log"].parent.mkdir(parents=True, exist_ok=True)
        try:
            paths["stop"].unlink(missing_ok=True)
        except Exception:
            pass
        command = self.build_command(profile, state)
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

    def request_stop(self, profile: CollectorProfile, state: dict[str, Any]) -> Path:
        paths = self.paths(profile, str(state.get("data_dir") or ""), str(state.get("config_path") or ""), str(state.get("db_path") or ""))
        paths["stop"].parent.mkdir(parents=True, exist_ok=True)
        paths["stop"].write_text(utc_now() + "\n", encoding="utf-8")
        return paths["stop"]

    def read_status(self, profile: CollectorProfile, state: dict[str, Any]) -> dict[str, Any]:
        paths = self.paths(profile, str(state.get("data_dir") or ""), str(state.get("config_path") or ""), str(state.get("db_path") or ""))
        status: dict[str, Any] = {}
        if paths["status"].exists():
            try:
                loaded = json.loads(paths["status"].read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    status = loaded
            except Exception:
                status = {}
        pid_text = ""
        if paths["pid"].exists():
            try:
                pid_text = paths["pid"].read_text(encoding="utf-8").strip()
            except Exception:
                pid_text = ""
        if pid_text:
            status["pid_text"] = pid_text
            try:
                if not is_process_running(int(pid_text)) and str(status.get("status") or "").lower() == "running":
                    status["status"] = "stale/unknown"
            except Exception:
                pass
        return status

    def source_health_rows(self, profile: CollectorProfile, state: dict[str, Any]) -> list[tuple[Any, ...]]:
        paths = self.paths(profile, str(state.get("data_dir") or ""), str(state.get("config_path") or ""), str(state.get("db_path") or ""))
        if not paths["db"].exists():
            return []
        with sqlite3.connect(str(paths["db"]), timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT source_id, source_group, enabled, market_relevance, last_success_at,
                       last_failure_at, last_error, items_last_fetch, inserted_last_fetch,
                       duplicates_last_fetch
                FROM sources
                ORDER BY source_group, source_id
                """
            ).fetchall()
        return [
            (
                row["source_id"],
                row["source_group"],
                row["enabled"],
                row["market_relevance"],
                row["last_success_at"],
                row["last_failure_at"],
                row["last_error"],
                row["items_last_fetch"],
                row["inserted_last_fetch"],
                row["duplicates_last_fetch"],
            )
            for row in rows
        ]

    def recent_articles_rows(
        self, profile: CollectorProfile, state: dict[str, Any], *, limit: int = 200
    ) -> list[dict[str, Any]]:
        paths = self.paths(profile, str(state.get("data_dir") or ""), str(state.get("config_path") or ""), str(state.get("db_path") or ""))
        if not paths["db"].exists():
            return []
        query_limit = max(1, min(int(limit), 1000))
        try:
            with sqlite3.connect(str(paths["db"]), timeout=5.0) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT
                        COALESCE(published_at, collected_at, '') AS article_time,
                        source_id,
                        market_relevance,
                        title,
                        COALESCE(canonical_url, source_url, '') AS article_url
                    FROM articles
                    ORDER BY COALESCE(published_at, collected_at) DESC, collected_at DESC
                    LIMIT ?
                    """,
                    (query_limit,),
                ).fetchall()
        except sqlite3.Error:
            return []
        return [
            {
                "article_time": row["article_time"] or "",
                "source_id": row["source_id"] or "",
                "market_relevance": row["market_relevance"] or "",
                "title": row["title"] or "",
                "article_url": row["article_url"] or "",
            }
            for row in rows
        ]

    def export_articles_csv(self, profile: CollectorProfile, state: dict[str, Any]) -> tuple[Path, int]:
        from research.collectors.news_research_store import export_articles_csv

        paths = self.paths(profile, str(state.get("data_dir") or ""), str(state.get("config_path") or ""), str(state.get("db_path") or ""))
        export_dir = paths["data_dir"] / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        csv_path = export_dir / f"{profile.key}_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return csv_path, int(export_articles_csv(paths["db"], csv_path))

    def export_source_health_csv(self, profile: CollectorProfile, state: dict[str, Any]) -> tuple[Path, int]:
        from research.collectors.news_research_store import export_source_health_csv

        paths = self.paths(profile, str(state.get("data_dir") or ""), str(state.get("config_path") or ""), str(state.get("db_path") or ""))
        export_dir = paths["data_dir"] / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        filename = "source_health" if profile.key == "news" else "web_source_health"
        csv_path = export_dir / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return csv_path, int(export_source_health_csv(paths["db"], csv_path))

    def open_data_folder(self, profile: CollectorProfile, state: dict[str, Any]) -> None:
        paths = self.paths(profile, str(state.get("data_dir") or ""), str(state.get("config_path") or ""), str(state.get("db_path") or ""))
        paths["data_dir"].mkdir(parents=True, exist_ok=True)
        open_path(paths["data_dir"])

    def open_log(self, profile: CollectorProfile, state: dict[str, Any]) -> None:
        paths = self.paths(profile, str(state.get("data_dir") or ""), str(state.get("config_path") or ""), str(state.get("db_path") or ""))
        paths["log"].parent.mkdir(parents=True, exist_ok=True)
        paths["log"].touch(exist_ok=True)
        open_path(paths["log"])

    def open_article_url(self, url: str) -> None:
        raw = str(url or "").strip()
        if not raw:
            raise ValueError("No article URL available for this row.")
        parsed = urlparse(raw)
        if not parsed.scheme:
            raw = f"https://{raw}"
            parsed = urlparse(raw)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
        webbrowser.open(raw)

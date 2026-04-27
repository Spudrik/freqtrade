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

from orderbook.metrics import estimate_storage_usage, normalize_freqtrade_pair_to_binance_symbol, normalize_whitelist_pairs

from .collector_service import is_process_running, open_path, utc_now, utf8_subprocess_env


def tokens(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").replace(",", " ").split() if part.strip()]


class OrderBookService:
    def __init__(self, app_dir: Path, python_exe: str | None = None) -> None:
        self.app_dir = Path(app_dir)
        self.python_exe = python_exe or sys.executable

    def app_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.app_dir / path

    def _resolve_data_dir(self, value: Any) -> Path:
        text = str(value or "").strip()
        default_dir = self.app_path("../orderbook_data/live")
        if not text:
            return default_dir
        return Path(text)

    def _resolve_config_path(self, value: Any) -> Path:
        text = str(value or "").strip()
        default_config = self.app_path("orderbook/config/sources.json")
        if not text:
            return default_config
        path = Path(text)
        if not path.exists() and path.name.lower() == default_config.name.lower():
            return default_config
        return path

    def paths(self, state: dict[str, Any]) -> dict[str, Path]:
        data_dir = self._resolve_data_dir(state.get("data_dir"))
        config_path = self._resolve_config_path(state.get("config_path"))
        return {
            "collector": self.app_path("orderbook/collector.py"),
            "config": config_path,
            "data_dir": data_dir,
            "db": data_dir / "orderbook_events.sqlite",
            "status": data_dir / "collector_status.json",
            "pid": data_dir / "collector.pid",
            "stop": data_dir / "collector.stop",
            "log": data_dir / "logs" / "orderbook_collector.log",
        }

    def build_runtime_config(self, state: dict[str, Any]) -> dict[str, Any]:
        depth = int(str(state.get("depth_levels") or "20"))
        if depth not in {5, 10, 20}:
            raise ValueError("Depth must be one of: 5, 10, 20.")
        update_ms = int(str(state.get("stream_update_ms") or "500"))
        if update_ms not in {100, 250, 500}:
            raise ValueError("Stream update ms must be one of: 100, 250, 500.")
        metric_interval = max(1, int(str(state.get("metric_interval_seconds") or "1")))
        snapshot_interval = max(1, int(str(state.get("snapshot_interval_seconds") or "60")))
        max_symbols = max(1, int(str(state.get("max_symbols") or "12")))
        warning_mb = max(1, int(str(state.get("capacity_warning_mb") or "500")))
        critical_mb = max(1, int(str(state.get("capacity_critical_mb") or "2000")))
        return {
            "version": 1,
            "exchange": "binance_usdm_futures",
            "market_type": "futures",
            "stream_mode": "partial_depth",
            "depth_levels": depth,
            "stream_update_ms": update_ms,
            "metric_interval_seconds": metric_interval,
            "snapshot_interval_seconds": snapshot_interval,
            "store_snapshots": bool(state.get("store_snapshots", True)),
            "max_symbols": max_symbols,
            "capacity_warning_mb": warning_mb,
            "capacity_critical_mb": critical_mb,
        }

    def write_runtime_config(self, state: dict[str, Any]) -> Path:
        paths = self.paths(state)
        payload: dict[str, Any] = {}
        if paths["config"].exists():
            try:
                loaded = json.loads(paths["config"].read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    payload = loaded
            except Exception:
                payload = {}
        payload.update(self.build_runtime_config(state))
        paths["config"].parent.mkdir(parents=True, exist_ok=True)
        paths["config"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return paths["config"]

    def normalized_pairs(self, pairs: list[str], state: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        max_symbols = max(1, int(str(state.get("max_symbols") or "12")))
        valid = normalize_whitelist_pairs(pairs, max_symbols=max_symbols)
        seen = {item["symbol"] for item in valid}
        preview: list[dict[str, str]] = []
        added: set[str] = set()
        for pair in pairs:
            symbol = normalize_freqtrade_pair_to_binance_symbol(pair)
            if not symbol:
                preview.append({"pair": pair, "symbol": "-", "status": "invalid"})
            elif symbol in added:
                preview.append({"pair": pair, "symbol": symbol, "status": "duplicate"})
            elif symbol not in seen:
                preview.append({"pair": pair, "symbol": symbol, "status": "max_symbols_limit"})
            else:
                preview.append({"pair": pair, "symbol": symbol, "status": "ok"})
                added.add(symbol)
        return valid, preview

    def estimate(self, pair_count: int, state: dict[str, Any]) -> dict[str, float]:
        return estimate_storage_usage(
            pair_count=pair_count,
            metric_interval_seconds=max(1, int(str(state.get("metric_interval_seconds") or "1"))),
            snapshot_interval_seconds=max(1, int(str(state.get("snapshot_interval_seconds") or "60"))),
            depth_levels=max(1, int(str(state.get("depth_levels") or "20"))),
            store_snapshots=bool(state.get("store_snapshots", True)),
        )

    def build_collector_command(self, state: dict[str, Any], pairs: list[str]) -> list[str]:
        paths = self.paths(state)
        if not paths["collector"].exists():
            raise FileNotFoundError(paths["collector"])
        valid, _ = self.normalized_pairs(pairs, state)
        if not valid:
            raise ValueError("No usable whitelist pairs found. Add pairs on the Pairs tab first.")
        return [
            self.python_exe,
            "-u",
            "-m",
            "orderbook.collector",
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
            "--stop-file",
            str(paths["stop"]),
            "--log-file",
            str(paths["log"]),
            "--pairs",
            ",".join(item["pair"] for item in valid),
        ]

    def start_detached(self, state: dict[str, Any], pairs: list[str]) -> int:
        self.write_runtime_config(state)
        paths = self.paths(state)
        paths["data_dir"].mkdir(parents=True, exist_ok=True)
        paths["log"].parent.mkdir(parents=True, exist_ok=True)
        try:
            paths["stop"].unlink(missing_ok=True)
        except Exception:
            pass
        if paths["pid"].exists():
            try:
                pid = int(paths["pid"].read_text(encoding="utf-8").strip())
                if is_process_running(pid):
                    return pid
            except Exception:
                pass
        command = self.build_collector_command(state, pairs)
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

    def latest_metric_rows(self, state: dict[str, Any]) -> list[tuple[Any, ...]]:
        paths = self.paths(state)
        if not paths["db"].exists():
            return []
        with sqlite3.connect(str(paths["db"]), timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT pair, symbol, status, best_bid, best_ask, spread_bps, imbalance_top20,
                       bid_pressure_ratio_60s, ask_pressure_ratio_60s,
                       nearest_bid_wall_distance_bps, nearest_ask_wall_distance_bps, last_metric_at
                FROM stream_status
                ORDER BY pair
                """
            ).fetchall()
        return [tuple(row[key] for key in row.keys()) for row in rows]

    def export_latest_metrics(self, state: dict[str, Any]) -> tuple[Path, int]:
        rows = self.latest_metric_rows(state)
        paths = self.paths(state)
        export_dir = paths["data_dir"] / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        output = export_dir / f"orderbook_latest_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        headers = ["pair", "symbol", "status", "best_bid", "best_ask", "spread_bps", "imbalance_top20", "bid_pressure_ratio_60s", "ask_pressure_ratio_60s", "nearest_bid_wall_distance_bps", "nearest_ask_wall_distance_bps", "last_metric_at"]
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

    def history_datadir(self, state: dict[str, Any], fallback_datadir: str = "") -> Path:
        raw = str(state.get("history_datadir") or "").strip()
        if raw:
            return Path(raw)
        if fallback_datadir:
            return Path(fallback_datadir)
        return self.app_path("../data/bybit_orderbook")

    def history_pairs(self, state: dict[str, Any], fallback_pairs: list[str]) -> list[str]:
        own = tokens(str(state.get("history_pairs") or ""))
        return own or fallback_pairs

    def history_depth(self, state: dict[str, Any]) -> int:
        depth = int(str(state.get("history_depth") or "500"))
        if depth <= 0:
            raise ValueError("Depth must be a positive integer.")
        if depth != 500:
            raise ValueError("Bybit historical archives currently support depth 500 only.")
        return depth

    def history_category(self, state: dict[str, Any]) -> str:
        category = str(state.get("history_category") or "linear").strip() or "linear"
        if category != "linear":
            raise ValueError("Bybit historical archives currently support category 'linear' only.")
        return category

    def build_history_download_args(self, state: dict[str, Any], pairs: list[str], fallback_datadir: str = "") -> list[str]:
        selected_pairs = self.history_pairs(state, pairs)
        if not selected_pairs:
            raise ValueError("Add at least one pair in Bybit History or on the Pairs tab.")
        args = [
            "download-data",
            "--exchange",
            str(state.get("history_exchange") or "bybit"),
            "--trading-mode",
            str(state.get("history_trading_mode") or "futures"),
            "--dl-orderbook",
            "--datadir",
            str(self.history_datadir(state, fallback_datadir)),
            "--orderbook-category",
            self.history_category(state),
            "--orderbook-depth",
            str(self.history_depth(state)),
            "-p",
            *selected_pairs,
        ]
        timerange = str(state.get("history_timerange") or "").strip()
        if timerange:
            args.extend(["--timerange", timerange])
        if state.get("history_erase"):
            args.append("--erase")
        return args

    def build_history_convert_args(self, state: dict[str, Any], pairs: list[str], fallback_datadir: str = "") -> list[str]:
        selected_pairs = self.history_pairs(state, pairs)
        if not selected_pairs:
            raise ValueError("Add at least one pair in Bybit History or on the Pairs tab.")
        timeframes = tokens(str(state.get("history_feature_timeframes") or ""))
        if not timeframes:
            raise ValueError("Feature timeframes are required. Example: 1h")
        args = [
            "orderbook-to-features",
            "--exchange",
            str(state.get("history_exchange") or "bybit"),
            "--trading-mode",
            str(state.get("history_trading_mode") or "futures"),
            "--datadir",
            str(self.history_datadir(state, fallback_datadir)),
            "--orderbook-category",
            self.history_category(state),
            "--orderbook-depth",
            str(self.history_depth(state)),
            "--data-format-orderbook-features",
            str(state.get("history_feature_format") or "feather"),
            "--pairs",
            *selected_pairs,
            "--timeframes",
            *timeframes,
        ]
        timerange = str(state.get("history_timerange") or "").strip()
        if timerange:
            args.extend(["--timerange", timerange])
        max_rows = str(state.get("history_max_rows") or "").strip()
        if max_rows:
            args.extend(["--max-rows", max_rows])
        return args

    def history_summary_rows(self, state: dict[str, Any], pairs: list[str], fallback_datadir: str = "") -> tuple[list[tuple[Any, ...]], str]:
        selected_pairs = self.history_pairs(state, pairs)
        datadir = self.history_datadir(state, fallback_datadir)
        category = self.history_category(state)
        depth_text = str(self.history_depth(state))
        rows: list[tuple[Any, ...]] = []
        availability_count = 0
        raw_total = 0
        feature_total = 0
        for pair in selected_pairs:
            symbol = normalize_freqtrade_pair_to_binance_symbol(pair) or "-"
            available = missing = first = last = "-"
            raw_count = 0
            feature_timeframes = "-"
            if symbol != "-":
                archive_dir = datadir / "orderbook" / category / symbol
                availability_path = archive_dir / f"availability_ob{depth_text}.json"
                raw_count = len(list(archive_dir.glob(f"*_ob{depth_text}.data.zip")))
                raw_total += raw_count
                if availability_path.exists():
                    try:
                        payload = json.loads(availability_path.read_text(encoding="utf-8"))
                    except Exception:
                        payload = {}
                    if isinstance(payload, dict):
                        availability_count += 1
                        available = str(payload.get("available_count", "-"))
                        missing = str(payload.get("missing_count", "-"))
                        first = str(payload.get("first_available") or "-")
                        last = str(payload.get("last_available") or "-")
                storage_name = self.pair_to_storage_name(pair)
                feature_dir = datadir / "orderbook_features" / category
                found: list[str] = []
                for path in sorted(feature_dir.glob(f"{storage_name}-*-ob{depth_text}.*")):
                    stem = path.stem
                    prefix = f"{storage_name}-"
                    suffix = f"-ob{depth_text}"
                    if stem.startswith(prefix) and suffix in stem:
                        timeframe = stem[len(prefix) : stem.rfind(suffix)]
                        if timeframe:
                            found.append(timeframe)
                unique = sorted(set(found))
                feature_total += len(unique)
                if unique:
                    feature_timeframes = ", ".join(unique)
            rows.append((pair, symbol, available, missing, first, last, raw_count, feature_timeframes))
        status = f"Pairs: {len(selected_pairs)} | availability reports: {availability_count} | raw archives: {raw_total} | feature timeframe files: {feature_total}"
        return rows, status

    @staticmethod
    def pair_to_storage_name(pair: str) -> str:
        value = str(pair)
        for ch in ["/", " ", ".", "@", "$", "+", ":"]:
            value = value.replace(ch, "_")
        return value

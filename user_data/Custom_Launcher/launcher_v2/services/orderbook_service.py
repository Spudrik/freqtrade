from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import csv
import json
import os
import shutil
import sqlite3
import subprocess
import sys

from orderbook.markets import (
    MARKET_PROFILES,
    market_profile_options,
    normalize_market_profile_keys,
    normalize_pairs_for_profiles,
    pair_to_symbol,
    resolve_profile_depth,
    resolve_profile_update_ms,
)
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
        market_profiles = normalize_market_profile_keys(state.get("market_profiles"))
        if not market_profiles:
            raise ValueError("Select at least one supported market profile.")
        valid_depths = {depth for profile in MARKET_PROFILES.values() for depth in profile.supported_depths}
        valid_updates = {update for profile in MARKET_PROFILES.values() for update in profile.supported_update_ms}
        depth = int(str(state.get("depth_levels") or "20"))
        if depth not in valid_depths:
            raise ValueError(f"Depth must be one of: {', '.join(str(value) for value in sorted(valid_depths))}.")
        update_ms = int(str(state.get("stream_update_ms") or "500"))
        if update_ms not in valid_updates:
            raise ValueError(f"Stream update ms must be one of: {', '.join(str(value) for value in sorted(valid_updates))}.")
        metric_interval = max(1, int(str(state.get("metric_interval_seconds") or "1")))
        snapshot_interval = max(1, int(str(state.get("snapshot_interval_seconds") or "60")))
        context_poll_seconds = max(30, int(str(state.get("context_poll_seconds") or "300")))
        max_symbols = max(1, int(str(state.get("max_symbols") or "12")))
        warning_mb = max(1, int(str(state.get("capacity_warning_mb") or "500")))
        critical_mb = max(1, int(str(state.get("capacity_critical_mb") or "2000")))
        return {
            "version": 1,
            "exchange": "multi_market",
            "market_type": "multi",
            "market_profiles": market_profiles,
            "stream_mode": "partial_depth",
            "depth_levels": depth,
            "stream_update_ms": update_ms,
            "metric_interval_seconds": metric_interval,
            "context_poll_seconds": context_poll_seconds,
            "context_period": str(state.get("context_period") or "5m"),
            "snapshot_interval_seconds": snapshot_interval,
            "store_snapshots": bool(state.get("store_snapshots", False)),
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

    def normalized_pairs(self, pairs: list[str], state: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        max_symbols = max(1, int(str(state.get("max_symbols") or "12")))
        profile_keys = normalize_market_profile_keys(state.get("market_profiles"))
        valid = normalize_pairs_for_profiles(
            pairs,
            profile_keys,
            max_symbols=max_symbols,
            depth_levels=max(1, int(str(state.get("depth_levels") or "20"))),
            stream_update_ms=max(1, int(str(state.get("stream_update_ms") or "500"))),
        )
        records_by_pair_market = {
            (str(item["canonical_pair"]), str(item["market_key"])): item
            for item in valid
        }
        seen = {item["canonical_pair"] for item in normalize_whitelist_pairs(pairs, max_symbols=max_symbols)}
        preview: list[dict[str, str]] = []
        for pair in pairs:
            canonical = str(pair or "").strip().upper()
            if "/" not in canonical:
                preview.append({"pair": pair, "market": "-", "symbol": "-", "depth": "-", "update_ms": "-", "status": "invalid"})
                continue
            canonical_pair = canonical.split(":", 1)[0]
            if canonical_pair not in seen:
                preview.append({"pair": pair, "market": "-", "symbol": "-", "depth": "-", "update_ms": "-", "status": "max_symbols_limit"})
                continue
            for profile_key in profile_keys:
                profile = MARKET_PROFILES[profile_key]
                symbol = pair_to_symbol(pair, profile)
                record = records_by_pair_market.get((canonical_pair, profile.market_key))
                preview.append(
                    {
                        "pair": pair,
                        "market": profile.market_key,
                        "symbol": symbol or "-",
                        "depth": str(record.get("stream_depth")) if record else "-",
                        "update_ms": str(record.get("stream_update_ms")) if record else "-",
                        "status": "ok" if symbol else "unsupported_quote",
                    }
                )
        return valid, preview

    def estimate(self, pair_count: int, state: dict[str, Any]) -> dict[str, float]:
        requested_depth = max(1, int(str(state.get("depth_levels") or "20")))
        profile_keys = normalize_market_profile_keys(state.get("market_profiles"))
        if not profile_keys:
            return estimate_storage_usage(
                pair_count=0,
                metric_interval_seconds=max(1, int(str(state.get("metric_interval_seconds") or "1"))),
                snapshot_interval_seconds=max(1, int(str(state.get("snapshot_interval_seconds") or "60"))),
                depth_levels=requested_depth,
                store_snapshots=bool(state.get("store_snapshots", False)),
            )
        retained_depth = max(
            min(requested_depth, resolve_profile_depth(MARKET_PROFILES[key], requested_depth))
            for key in profile_keys
        )
        return estimate_storage_usage(
            pair_count=pair_count,
            metric_interval_seconds=max(1, int(str(state.get("metric_interval_seconds") or "1"))),
            snapshot_interval_seconds=max(1, int(str(state.get("snapshot_interval_seconds") or "60"))),
            depth_levels=retained_depth,
            store_snapshots=bool(state.get("store_snapshots", False)),
        )

    def effective_profile_settings(self, state: dict[str, Any]) -> list[tuple[str, str, str, str]]:
        requested_depth = max(1, int(str(state.get("depth_levels") or "20")))
        requested_update = max(1, int(str(state.get("stream_update_ms") or "500")))
        rows: list[tuple[str, str, str, str]] = []
        for profile_key in normalize_market_profile_keys(state.get("market_profiles")):
            profile = MARKET_PROFILES[profile_key]
            rows.append(
                (
                    profile.market_key,
                    str(resolve_profile_depth(profile, requested_depth)),
                    str(resolve_profile_update_ms(profile, requested_update)),
                    "yes" if profile.context_supported else "no",
                )
            )
        return rows

    def build_collector_command(self, state: dict[str, Any], pairs: list[str]) -> list[str]:
        paths = self.paths(state)
        if not paths["collector"].exists():
            raise FileNotFoundError(paths["collector"])
        valid, _ = self.normalized_pairs(pairs, state)
        if not valid:
            raise ValueError("No usable whitelist pairs found. Add pairs on the Pairs tab first.")
        pair_args: list[str] = []
        seen_pairs: set[str] = set()
        for item in valid:
            canonical = str(item.get("canonical_pair") or item.get("pair") or "")
            if canonical in seen_pairs:
                continue
            seen_pairs.add(canonical)
            pair_args.append(str(item["pair"]))
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
            ",".join(pair_args),
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

    def _drive_usage_status(self, path: Path) -> dict[str, Any]:
        target = path.expanduser()
        if not target.is_absolute():
            target = self.app_dir / target
        target = target.resolve(strict=False)
        existing = next((candidate for candidate in (target, *target.parents) if candidate.exists()), target)
        usage = shutil.disk_usage(existing)
        total_gb = usage.total / (1024 ** 3)
        used_gb = usage.used / (1024 ** 3)
        free_gb = usage.free / (1024 ** 3)
        used_pct = (usage.used / usage.total * 100.0) if usage.total else 0.0
        free_pct = (usage.free / usage.total * 100.0) if usage.total else 0.0
        return {
            "drive_path": str(existing),
            "drive_total_gb": round(total_gb, 2),
            "drive_used_gb": round(used_gb, 2),
            "drive_free_gb": round(free_gb, 2),
            "drive_used_pct": round(used_pct, 1),
            "drive_free_pct": round(free_pct, 1),
            "drive_free_total": f"{free_gb:.1f} GB free / {total_gb:.1f} GB total",
            "drive_used": f"{used_gb:.1f} GB used ({used_pct:.1f}%)",
        }

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
        try:
            status.update(self._drive_usage_status(paths["data_dir"]))
        except OSError as exc:
            status["drive_free_total"] = f"unavailable: {exc}"
            status["drive_used"] = "-"
        return status

    def latest_metric_rows(self, state: dict[str, Any]) -> list[tuple[Any, ...]]:
        paths = self.paths(state)
        if not paths["db"].exists():
            return []
        with sqlite3.connect(str(paths["db"]), timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT market_key, canonical_pair, pair, symbol, status, best_bid, best_ask, spread_bps, imbalance_top20,
                       bid_pressure_ratio_60s, ask_pressure_ratio_60s,
                       nearest_bid_wall_distance_bps, nearest_ask_wall_distance_bps, last_metric_at
                FROM stream_status
                ORDER BY canonical_pair, market_key
                """
            ).fetchall()
        return [tuple(row[key] for key in row.keys()) for row in rows]

    def stream_health_rows(self, state: dict[str, Any]) -> list[tuple[Any, ...]]:
        paths = self.paths(state)
        if not paths["db"].exists():
            return []
        with sqlite3.connect(str(paths["db"]), timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT market_key, canonical_pair, symbol, status, message_count, metric_count,
                       reconnect_count, error_count, last_message_at, last_metric_at, last_error
                FROM stream_status
                ORDER BY canonical_pair, market_key
                """
            ).fetchall()
        return [tuple(row[key] for key in row.keys()) for row in rows]

    def export_latest_metrics(self, state: dict[str, Any]) -> tuple[Path, int]:
        rows = self.latest_metric_rows(state)
        paths = self.paths(state)
        export_dir = paths["data_dir"] / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        output = export_dir / f"orderbook_latest_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        headers = ["market_key", "canonical_pair", "pair", "symbol", "status", "best_bid", "best_ask", "spread_bps", "imbalance_top20", "bid_pressure_ratio_60s", "ask_pressure_ratio_60s", "nearest_bid_wall_distance_bps", "nearest_ask_wall_distance_bps", "last_metric_at"]
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

    def market_profile_options(self) -> list[tuple[str, str]]:
        return market_profile_options()

    def comparison_rows(self, state: dict[str, Any]) -> list[tuple[Any, ...]]:
        paths = self.paths(state)
        if not paths["db"].exists():
            return []
        with sqlite3.connect(str(paths["db"]), timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT t.*
                FROM orderbook_metric_ticks t
                JOIN (
                    SELECT stream_id, MAX(id) AS latest_id
                    FROM orderbook_metric_ticks
                    GROUP BY stream_id
                ) latest ON latest.latest_id = t.id
                ORDER BY t.canonical_pair, t.market_key
                """
            ).fetchall()
        by_pair: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            by_pair.setdefault(str(row["canonical_pair"]), []).append(row)
        output: list[tuple[Any, ...]] = []
        for canonical_pair, pair_rows in sorted(by_pair.items()):
            spot_rows = [row for row in pair_rows if row["market_type"] == "spot"]
            future_rows = [row for row in pair_rows if row["market_type"] == "futures"]
            for spot in spot_rows:
                for future in future_rows:
                    spot_mid = _float_or_none(spot["mid_price"])
                    future_mid = _float_or_none(future["mid_price"])
                    basis = ((future_mid - spot_mid) / spot_mid * 10000.0) if spot_mid and future_mid else None
                    spot_depth = _float_or_none(spot["bid_notional_top20"]) or 0.0
                    spot_depth += _float_or_none(spot["ask_notional_top20"]) or 0.0
                    future_depth = _float_or_none(future["bid_notional_top20"]) or 0.0
                    future_depth += _float_or_none(future["ask_notional_top20"]) or 0.0
                    depth_ratio = (future_depth / spot_depth) if spot_depth > 0 else None
                    spread_delta = _delta(future["spread_bps"], spot["spread_bps"])
                    imbalance_delta = _delta(future["imbalance_top20"], spot["imbalance_top20"])
                    pressure_delta = _delta(future["strong_bid_pressure"], spot["strong_bid_pressure"])
                    pressure_lead_lag = _pressure_lead_lag(spot, future)
                    wall_delta = _delta(future["nearest_bid_wall_distance_bps"], spot["nearest_bid_wall_distance_bps"])
                    output.append(
                        (
                            canonical_pair,
                            spot["market_key"],
                            future["market_key"],
                            _fmt(spot_mid),
                            _fmt(future_mid),
                            _fmt(basis),
                            _fmt(spread_delta),
                            _fmt(depth_ratio),
                            _fmt(imbalance_delta),
                            _fmt(pressure_delta),
                            pressure_lead_lag,
                            _fmt(wall_delta),
                            future["ts"],
                        )
                    )
        return output

    def latest_context_rows(self, state: dict[str, Any]) -> list[tuple[Any, ...]]:
        paths = self.paths(state)
        if not paths["db"].exists():
            return []
        with sqlite3.connect(str(paths["db"]), timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT c.*
                FROM market_context_ticks c
                JOIN (
                    SELECT market_key, canonical_pair, MAX(id) AS latest_id
                    FROM market_context_ticks
                    GROUP BY market_key, canonical_pair
                ) latest ON latest.latest_id = c.id
                ORDER BY c.canonical_pair, c.market_key
                """
            ).fetchall()
        return [
            (
                row["market_key"],
                row["canonical_pair"],
                _fmt(row["funding_rate"]),
                _fmt(row["open_interest"]),
                _fmt(row["long_ratio"]),
                _fmt(row["short_ratio"]),
                _fmt(row["long_short_ratio"]),
                _fmt(row["taker_buy_volume"]),
                _fmt(row["taker_sell_volume"]),
                _fmt(row["taker_buy_sell_ratio"]),
                row["source_ts"] or row["ts"],
            )
            for row in rows
        ]

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


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _delta(left: Any, right: Any) -> float | None:
    left_value = _float_or_none(left)
    right_value = _float_or_none(right)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def _fmt(value: Any) -> str:
    number = _float_or_none(value)
    if number is None:
        return "-"
    return f"{number:.6g}"


def _pressure_lead_lag(spot: sqlite3.Row, future: sqlite3.Row) -> str:
    spot_score = _pressure_score(spot)
    future_score = _pressure_score(future)
    if spot_score == future_score:
        if spot_score > 0:
            return "aligned_bid"
        if spot_score < 0:
            return "aligned_ask"
        return "neutral"
    if future_score > 0 and spot_score <= 0:
        return "futures_bid_leads"
    if future_score < 0 and spot_score >= 0:
        return "futures_ask_leads"
    if spot_score > 0 and future_score <= 0:
        return "spot_bid_leads"
    if spot_score < 0 and future_score >= 0:
        return "spot_ask_leads"
    return "divergent"


def _pressure_score(row: sqlite3.Row) -> int:
    bid = int(_float_or_none(row["strong_bid_pressure"]) or 0)
    ask = int(_float_or_none(row["strong_ask_pressure"]) or 0)
    return bid - ask

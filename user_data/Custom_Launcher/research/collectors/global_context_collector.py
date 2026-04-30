from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .global_context_store import (
    connect_db,
    fetch_storage_summary,
    init_db,
    insert_context_tick,
    update_collector_run,
    upsert_collector_run,
    upsert_source_status,
)
from .research_collector_common import configure_logging, evaluate_startup_stop_file, load_json, save_json, sleep_with_stop, utc_now


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = THIS_DIR.parents[0] / "config" / "global_context_sources.json"
DEFAULT_DATA_DIR = THIS_DIR.parents[2] / "research_news_data" / "global_context"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "global_context.sqlite"
DEFAULT_STATUS_PATH = DEFAULT_DATA_DIR / "collector_status.json"
DEFAULT_PID_PATH = DEFAULT_DATA_DIR / "collector.pid"
DEFAULT_STOP_PATH = DEFAULT_DATA_DIR / "collector.stop"
DEFAULT_LOG_PATH = DEFAULT_DATA_DIR / "logs" / "global_context_collector.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone global market context API collector")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--status-file", type=Path, default=DEFAULT_STATUS_PATH)
    parser.add_argument("--pid-file", type=Path, default=DEFAULT_PID_PATH)
    parser.add_argument("--stop-file", type=Path, default=DEFAULT_STOP_PATH)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--interval-seconds", type=int, default=1800)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def ensure_dirs(data_dir: Path) -> None:
    for path in [data_dir, data_dir / "logs", data_dir / "exports", data_dir / "raw"]:
        path.mkdir(parents=True, exist_ok=True)


def read_config(path: Path) -> dict[str, Any]:
    payload = load_json(path, {})
    if not isinstance(payload, dict):
        payload = {}
    defaults = load_json(DEFAULT_CONFIG_PATH, {})
    if isinstance(defaults, dict):
        for key, value in defaults.items():
            payload.setdefault(key, value)
    payload.setdefault("poll_interval_seconds", 1800)
    payload.setdefault("request_timeout_seconds", 20)
    payload.setdefault("store_raw_payloads", True)
    payload.setdefault("user_agent", "FreQ-GlobalContextCollector/1.0")
    payload.setdefault("sources", [])
    return payload


def fetch_public_json(url: str, *, timeout_seconds: int, user_agent: str) -> Any:
    request = Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_source(source: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    source_type = str(source.get("type") or "").strip()
    url = str(source.get("url") or "").strip()
    timeout_seconds = int(config.get("request_timeout_seconds", 20))
    user_agent = str(config.get("user_agent") or "FreQ-GlobalContextCollector/1.0")
    if source_type == "coingecko_markets":
        params = {
            "vs_currency": str(source.get("vs_currency") or "usd"),
            "ids": ",".join(str(value) for value in source.get("coin_ids") or ["bitcoin", "ethereum"]),
            "sparkline": "false",
            "price_change_percentage": "1h,24h,7d",
        }
        separator = "&" if "?" in url else "?"
        url = url + separator + urlencode(params)
    payload = fetch_public_json(url, timeout_seconds=timeout_seconds, user_agent=user_agent)
    return normalize_source_payload(source, payload, store_raw=bool(config.get("store_raw_payloads", True)))


def normalize_source_payload(source: dict[str, Any], payload: Any, *, store_raw: bool = True) -> dict[str, Any]:
    source_type = str(source.get("type") or "").strip()
    if source_type == "alternative_fng":
        return normalize_fear_greed(source, payload, store_raw=store_raw)
    if source_type == "coingecko_global":
        return normalize_coingecko_global(source, payload, store_raw=store_raw)
    if source_type == "coingecko_markets":
        return normalize_coingecko_markets(source, payload, store_raw=store_raw)
    if source_type == "defillama_stablecoins":
        return normalize_defillama_stablecoins(source, payload, store_raw=store_raw)
    if source_type == "defillama_chains":
        return normalize_defillama_chains(source, payload, store_raw=store_raw)
    raise ValueError(f"Unsupported global context source type: {source_type}")


def normalize_fear_greed(source: dict[str, Any], payload: Any, *, store_raw: bool = True) -> dict[str, Any]:
    rows = payload.get("data") if isinstance(payload, dict) else []
    row = rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], dict) else {}
    value = _float_or_none(row.get("value"))
    classification = str(row.get("value_classification") or "")
    score = _clamp(value if value is not None else 50.0)
    notes = classification or _score_label(score)
    return _result(
        source,
        metric_key="fear_greed_index",
        score=score,
        signal=_score_signal(score),
        value=value,
        unit="index",
        notes=notes,
        source_ts=_unix_to_iso(row.get("timestamp")),
        raw=payload if store_raw else None,
    )


def normalize_coingecko_global(source: dict[str, Any], payload: Any, *, store_raw: bool = True) -> dict[str, Any]:
    data = payload.get("data") if isinstance(payload, dict) else {}
    if not isinstance(data, dict):
        data = {}
    market_cap = _nested_float(data, "total_market_cap", "usd")
    volume = _nested_float(data, "total_volume", "usd")
    btc_dom = _nested_float(data, "market_cap_percentage", "btc")
    change_24h = _float_or_none(data.get("market_cap_change_percentage_24h_usd"))
    score = _clamp(50.0 + (change_24h or 0.0) * 5.0)
    notes = f"cap ${_compact_usd(market_cap)}, volume ${_compact_usd(volume)}, BTC dom {_fmt_pct(btc_dom)}, 24h cap {_fmt_pct(change_24h)}"
    return _result(
        source,
        metric_key="global_market_cap_change_24h",
        score=score,
        signal=_score_signal(score),
        value=change_24h,
        unit="percent",
        notes=notes,
        raw=payload if store_raw else None,
    )


def normalize_coingecko_markets(source: dict[str, Any], payload: Any, *, store_raw: bool = True) -> dict[str, Any]:
    rows = payload if isinstance(payload, list) else []
    changes_24h: list[float] = []
    changes_7d: list[float] = []
    note_parts: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or row.get("id") or "").upper()
        price = _float_or_none(row.get("current_price"))
        change_24h = _float_or_none(row.get("price_change_percentage_24h"))
        change_7d = _float_or_none(row.get("price_change_percentage_7d_in_currency"))
        if change_24h is not None:
            changes_24h.append(change_24h)
        if change_7d is not None:
            changes_7d.append(change_7d)
        if symbol:
            note_parts.append(f"{symbol} ${_compact_usd(price)} 24h {_fmt_pct(change_24h)} 7d {_fmt_pct(change_7d)}")
    avg_24h = _mean(changes_24h)
    avg_7d = _mean(changes_7d)
    score = _clamp(50.0 + (avg_24h or 0.0) * 4.0 + (avg_7d or 0.0) * 1.5)
    return _result(
        source,
        metric_key="btc_eth_avg_change_24h",
        score=score,
        signal=_score_signal(score),
        value=avg_24h,
        unit="percent",
        notes="; ".join(note_parts) or "No coin rows returned.",
        raw=payload if store_raw else None,
    )


def normalize_defillama_stablecoins(source: dict[str, Any], payload: Any, *, store_raw: bool = True) -> dict[str, Any]:
    assets = payload.get("peggedAssets") if isinstance(payload, dict) else []
    totals = {"current": 0.0, "day": 0.0, "week": 0.0, "month": 0.0}
    top_assets: list[tuple[str, float]] = []
    for asset in assets if isinstance(assets, list) else []:
        if not isinstance(asset, dict):
            continue
        current = _nested_float(asset, "circulating", "peggedUSD")
        day = _nested_float(asset, "circulatingPrevDay", "peggedUSD")
        week = _nested_float(asset, "circulatingPrevWeek", "peggedUSD")
        month = _nested_float(asset, "circulatingPrevMonth", "peggedUSD")
        totals["current"] += current or 0.0
        totals["day"] += day or 0.0
        totals["week"] += week or 0.0
        totals["month"] += month or 0.0
        name = str(asset.get("symbol") or asset.get("name") or "")
        if name and current:
            top_assets.append((name, current))
    change_1d = _pct_change(totals["current"], totals["day"])
    change_7d = _pct_change(totals["current"], totals["week"])
    score = _clamp(50.0 + (change_7d or 0.0) * 10.0 + (change_1d or 0.0) * 5.0)
    top = ", ".join(f"{name} ${_compact_usd(value)}" for name, value in sorted(top_assets, key=lambda item: item[1], reverse=True)[:3])
    notes = f"supply ${_compact_usd(totals['current'])}, 1d {_fmt_pct(change_1d)}, 7d {_fmt_pct(change_7d)}, top {top}"
    return _result(
        source,
        metric_key="stablecoin_supply_change_7d",
        score=score,
        signal=_score_signal(score),
        value=change_7d,
        unit="percent",
        notes=notes,
        raw=payload if store_raw else None,
    )


def normalize_defillama_chains(source: dict[str, Any], payload: Any, *, store_raw: bool = True) -> dict[str, Any]:
    rows = payload if isinstance(payload, list) else []
    total_tvl = 0.0
    weighted_1d = 0.0
    weighted_7d = 0.0
    top_chains: list[tuple[str, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        tvl = _float_or_none(row.get("tvl")) or 0.0
        if tvl <= 0:
            continue
        change_1d = _float_or_none(row.get("change_1d")) or 0.0
        change_7d = _float_or_none(row.get("change_7d")) or 0.0
        total_tvl += tvl
        weighted_1d += tvl * change_1d
        weighted_7d += tvl * change_7d
        name = str(row.get("name") or "")
        if name:
            top_chains.append((name, tvl))
    avg_1d = (weighted_1d / total_tvl) if total_tvl > 0 else None
    avg_7d = (weighted_7d / total_tvl) if total_tvl > 0 else None
    score = _clamp(50.0 + (avg_7d or 0.0) * 2.0 + (avg_1d or 0.0) * 3.0)
    top = ", ".join(f"{name} ${_compact_usd(value)}" for name, value in sorted(top_chains, key=lambda item: item[1], reverse=True)[:3])
    notes = f"TVL ${_compact_usd(total_tvl)}, weighted 1d {_fmt_pct(avg_1d)}, 7d {_fmt_pct(avg_7d)}, top {top}"
    return _result(
        source,
        metric_key="defi_tvl_weighted_change_7d",
        score=score,
        signal=_score_signal(score),
        value=avg_7d,
        unit="percent",
        notes=notes,
        raw=payload if store_raw else None,
    )


def main() -> int:
    args = parse_args()
    ensure_dirs(args.data_dir)
    configure_logging(args.log_file)
    init_db(args.db)
    config = read_config(args.config)
    sources = [source for source in config.get("sources", []) if isinstance(source, dict)]
    enabled_sources = [source for source in sources if bool(source.get("enabled", True))]
    interval_seconds = max(60, int(args.interval_seconds or config.get("poll_interval_seconds", 1800)))

    startup = evaluate_startup_stop_file(args.stop_file)
    if startup.get("fresh_stop_requested"):
        save_json(args.status_file, {"status": "stopped", "last_error": "Fresh stop file found at startup."})
        return 0

    run_id = hashlib.sha256(f"{os.getpid()}|{utc_now()}|global_context".encode("utf-8")).hexdigest()
    args.pid_file.parent.mkdir(parents=True, exist_ok=True)
    args.pid_file.write_text(str(os.getpid()), encoding="utf-8")
    status_payload: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "pid": os.getpid(),
        "started_at": utc_now(),
        "heartbeat_at": utc_now(),
        "last_fetch_at": None,
        "sources_total": len(sources),
        "sources_enabled": len(enabled_sources),
        "success_count_total": 0,
        "failure_count_total": 0,
        "ticks_total": 0,
        "new_ticks_last_cycle": 0,
        "last_error": None,
        "db_path": str(args.db),
        "config_path": str(args.config),
        "data_dir": str(args.data_dir),
    }
    save_json(args.status_file, status_payload)

    conn = connect_db(args.db)
    try:
        upsert_collector_run(
            conn,
            run_id=run_id,
            started_at=str(status_payload["started_at"]),
            status="running",
            pid=os.getpid(),
            config_path=str(args.config),
            db_path=str(args.db),
        )
        for source in sources:
            upsert_source_status(conn, _source_status_payload(source, enabled=bool(source.get("enabled", True))))
        conn.commit()
    finally:
        conn.close()

    terminal_status = "stopped"
    try:
        while True:
            if args.stop_file.exists():
                terminal_status = "stopped"
                break
            new_ticks, failures = run_collection_cycle(args.db, enabled_sources, config)
            status_payload["last_fetch_at"] = utc_now()
            status_payload["heartbeat_at"] = utc_now()
            status_payload["ticks_total"] = int(status_payload.get("ticks_total", 0)) + new_ticks
            status_payload["success_count_total"] = int(status_payload.get("success_count_total", 0)) + new_ticks
            status_payload["failure_count_total"] = int(status_payload.get("failure_count_total", 0)) + failures
            status_payload["new_ticks_last_cycle"] = new_ticks
            conn = connect_db(args.db)
            try:
                summary = fetch_storage_summary(conn)
            finally:
                conn.close()
            status_payload["stored_ticks"] = summary["tick_rows"]
            status_payload["stored_sources"] = summary["source_rows"]
            save_json(args.status_file, status_payload)
            if args.once:
                break
            if sleep_with_stop(args.stop_file, interval_seconds):
                terminal_status = "stopped"
                break
    except Exception as exc:
        logging.exception("Global context collector crashed")
        terminal_status = "error"
        status_payload["status"] = "error"
        status_payload["last_error"] = str(exc)
        save_json(args.status_file, status_payload)
        return 1
    finally:
        status_payload["status"] = terminal_status
        status_payload["pid"] = None
        status_payload["heartbeat_at"] = utc_now()
        save_json(args.status_file, status_payload)
        conn = connect_db(args.db)
        try:
            update_collector_run(conn, run_id, terminal_status, status_payload.get("last_error"))
            conn.commit()
        finally:
            conn.close()
        try:
            args.pid_file.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            if args.stop_file.exists():
                args.stop_file.unlink()
        except Exception:
            pass
    return 0


def run_collection_cycle(db_path: Path, enabled_sources: list[dict[str, Any]], config: dict[str, Any]) -> tuple[int, int]:
    inserted = 0
    failures = 0
    conn = connect_db(db_path)
    try:
        for source in enabled_sources:
            try:
                row = fetch_source(source, config)
                insert_context_tick(conn, row)
                upsert_source_status(conn, _source_status_payload(source, enabled=True, row=row))
                inserted += 1
            except Exception as exc:
                failures += 1
                logging.warning("Global context source failed source_id=%s error=%s", source.get("id"), exc)
                upsert_source_status(conn, _source_status_payload(source, enabled=True, error=str(exc)))
        conn.commit()
    finally:
        conn.close()
    return inserted, failures


def _source_status_payload(
    source: dict[str, Any],
    *,
    enabled: bool,
    row: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload = {
        "source_id": source.get("id"),
        "source_group": source.get("source_group") or "",
        "source_type": source.get("type") or "",
        "enabled": int(bool(enabled)),
        "market_relevance": source.get("market_relevance") or "",
        "url": source.get("url") or "",
        "last_success_at": None,
        "last_failure_at": utc_now() if error else None,
        "last_error": error,
        "last_score": None,
        "last_signal": None,
        "last_value": None,
        "last_unit": None,
        "last_notes": None,
        "updated_at": utc_now(),
    }
    if row:
        payload.update(
            {
                "last_success_at": row.get("ts"),
                "last_failure_at": None,
                "last_error": None,
                "last_score": row.get("score"),
                "last_signal": row.get("signal"),
                "last_value": row.get("value"),
                "last_unit": row.get("unit"),
                "last_notes": row.get("notes"),
            }
        )
    return payload


def _result(
    source: dict[str, Any],
    *,
    metric_key: str,
    score: float | None,
    signal: str,
    value: float | None,
    unit: str,
    notes: str,
    source_ts: str | None = None,
    raw: Any = None,
) -> dict[str, Any]:
    return {
        "ts": utc_now(),
        "source_ts": source_ts,
        "source_id": source.get("id"),
        "source_group": source.get("source_group") or "",
        "source_type": source.get("type") or "",
        "metric_key": metric_key,
        "score": score,
        "signal": signal,
        "value": value,
        "unit": unit,
        "notes": notes[:2000],
        "raw_json": raw,
    }


def _nested_float(payload: dict[str, Any], *keys: str) -> float | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return _float_or_none(value)


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _pct_change(current: float, previous: float) -> float | None:
    if previous <= 0:
        return None
    return (current - previous) / previous * 100.0


def _clamp(value: float | None, low: float = 0.0, high: float = 100.0) -> float:
    if value is None:
        return 50.0
    return max(low, min(high, float(value)))


def _score_signal(score: float | None) -> str:
    value = _clamp(score)
    if value >= 65:
        return "risk_on"
    if value <= 35:
        return "risk_off"
    return "neutral"


def _score_label(score: float | None) -> str:
    value = _clamp(score)
    if value >= 75:
        return "strong risk-on"
    if value >= 60:
        return "risk-on"
    if value <= 25:
        return "strong risk-off"
    if value <= 40:
        return "risk-off"
    return "neutral"


def _compact_usd(value: float | None) -> str:
    if value is None:
        return "-"
    abs_value = abs(value)
    if abs_value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f}T"
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:.2f}"


def _fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}%"


def _unix_to_iso(value: Any) -> str | None:
    try:
        number = float(value)
        if number > 10_000_000_000:
            number = number / 1000.0
        return datetime.fromtimestamp(number, timezone.utc).isoformat()
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())

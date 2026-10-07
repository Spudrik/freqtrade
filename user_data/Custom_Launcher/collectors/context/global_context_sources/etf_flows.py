from __future__ import annotations

from datetime import date, datetime, time, timezone
from io import StringIO
from typing import Any

import pandas as pd

from ..research_collector_common import slugify, utc_now
from .common import clamp, fetch_public_text, score_signal


def fetch_farside_btc_etf_flows(source: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    url = str(source.get("url") or "https://farside.co.uk/bitcoin-etf-flow-all-data/")
    text = fetch_public_text(
        url,
        timeout_seconds=int(source.get("timeout_seconds") or config.get("request_timeout_seconds") or 20),
        user_agent=str(config.get("user_agent") or "FreQ-GlobalContextCollector/1.0"),
    )
    tables = pd.read_html(StringIO(text))
    if "table_index" in source:
        table_index = int(source.get("table_index", 0))
        if len(tables) <= table_index:
            raise ValueError(f"Farside page returned {len(tables)} tables; expected table index {table_index}.")
        table = tables[table_index]
    else:
        table = _find_flow_table(tables)
    rows = _parse_flow_table(table)
    if not rows:
        raise ValueError("Farside ETF flow table produced no dated rows.")
    return {
        "url": url,
        "rows": rows,
        "collected_at": utc_now(),
    }


def _find_flow_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    for table in tables:
        columns = {str(column).strip().lower() for column in table.columns}
        if "date" in columns and "total" in columns:
            return table
    raise ValueError(f"Farside page returned {len(tables)} tables, but none looked like the ETF flow table.")


def normalize_farside_btc_etf_flows(source: dict[str, Any], payload: Any, *, store_raw: bool = True) -> list[dict[str, Any]]:
    rows = payload.get("rows") if isinstance(payload, dict) else []
    if not isinstance(rows, list) or not rows:
        raise ValueError("Farside ETF flow payload has no rows.")

    emit = str(source.get("emit") or "latest_reported").lower()
    selected_rows = rows if emit == "all" else [_latest_reported_row(rows)]
    metric_prefix = str(source.get("metric_prefix") or "btc_spot_etf")
    normalized: list[dict[str, Any]] = []
    for raw_row in selected_rows:
        if not isinstance(raw_row, dict):
            continue
        source_ts = str(raw_row.get("date") or "")
        ts = _available_at(source_ts, int(source.get("available_hour_utc", 0)))
        total = _float_or_none(raw_row.get("Total"))
        if total is None:
            continue
        scale = float(source.get("score_scale_musd") or 500.0)
        score = clamp(50.0 + (total / scale * 25.0))
        normalized.append(
            _flow_row(
                source,
                metric_key=f"{metric_prefix}_net_flow_musd",
                value=total,
                score=score,
                ts=ts,
                source_ts=source_ts,
                notes="Daily US spot BTC ETF net flow in USD millions. Available timestamp is conservatively shifted after the flow date.",
                raw=raw_row if store_raw else None,
            )
        )
        funds = [key for key in raw_row if key not in {"date", "Total"}]
        positive = sum(1 for key in funds if (_float_or_none(raw_row.get(key)) or 0.0) > 0)
        negative = sum(1 for key in funds if (_float_or_none(raw_row.get(key)) or 0.0) < 0)
        normalized.extend(
            [
                _flow_row(
                    source,
                    metric_key=f"{metric_prefix}_positive_fund_count",
                    value=float(positive),
                    score=clamp(float(positive) * 10.0),
                    ts=ts,
                    source_ts=source_ts,
                    notes="Number of spot BTC ETF funds with positive reported daily flow.",
                    raw=raw_row if store_raw else None,
                ),
                _flow_row(
                    source,
                    metric_key=f"{metric_prefix}_negative_fund_count",
                    value=float(negative),
                    score=clamp(float(negative) * 10.0),
                    ts=ts,
                    source_ts=source_ts,
                    notes="Number of spot BTC ETF funds with negative reported daily flow.",
                    raw=raw_row if store_raw else None,
                ),
            ]
        )
        for fund in source.get("fund_metrics") or ["IBIT", "FBTC", "GBTC"]:
            value = _float_or_none(raw_row.get(str(fund).upper()))
            if value is None:
                continue
            normalized.append(
                _flow_row(
                    source,
                    metric_key=f"{metric_prefix}_{slugify(str(fund))}_flow_musd",
                    value=value,
                    score=clamp(50.0 + (value / scale * 25.0)),
                    ts=ts,
                    source_ts=source_ts,
                    notes=f"Daily US spot BTC ETF flow for {str(fund).upper()} in USD millions.",
                    raw=raw_row if store_raw else None,
                )
            )
    if not normalized:
        raise ValueError("Farside ETF flow payload produced no numeric rows.")
    return normalized


def _parse_flow_table(table: pd.DataFrame) -> list[dict[str, Any]]:
    frame = table.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    if "Date" not in frame.columns:
        return []
    frame = frame.dropna(how="all")
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        parsed_date = pd.to_datetime(row.get("Date"), utc=True, errors="coerce", dayfirst=True)
        if pd.isna(parsed_date):
            continue
        item: dict[str, Any] = {"date": pd.Timestamp(parsed_date).date().isoformat()}
        for column in frame.columns:
            if column == "Date":
                continue
            item[str(column).strip()] = _flow_value(row.get(column))
        rows.append(item)
    rows.sort(key=lambda item: str(item["date"]))
    return rows


def _latest_reported_row(rows: list[Any]) -> dict[str, Any]:
    for row in reversed(rows):
        if isinstance(row, dict) and _has_reported_flow(row):
            return row
    return rows[-1] if isinstance(rows[-1], dict) else {}


def _has_reported_flow(row: dict[str, Any]) -> bool:
    return any(
        abs(_float_or_none(row.get(key)) or 0.0) > 0.0
        for key in row
        if key != "date"
    )


def _flow_value(value: Any) -> float:
    text = str(value or "").strip().replace(",", "")
    if not text or text == "-" or text.lower() == "nan":
        return 0.0
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    number = _float_or_none(text) or 0.0
    return -number if negative else number


def _available_at(source_date: str, hour_utc: int) -> str:
    parsed = pd.to_datetime(source_date, utc=True, errors="coerce")
    if pd.isna(parsed):
        return utc_now()
    safe_date = pd.Timestamp(parsed).date() + pd.Timedelta(days=1)
    return datetime.combine(safe_date, time(hour=max(0, min(23, hour_utc))), timezone.utc).isoformat()


def _flow_row(
    source: dict[str, Any],
    *,
    metric_key: str,
    value: float,
    score: float,
    ts: str,
    source_ts: str,
    notes: str,
    raw: Any,
) -> dict[str, Any]:
    return {
        "ts": ts,
        "source_ts": source_ts,
        "source_id": source.get("id"),
        "source_group": source.get("source_group") or "institutional_flows",
        "source_type": source.get("type") or "farside_btc_etf_flows",
        "metric_key": metric_key,
        "score": score,
        "source_score": score,
        "calc_score": score,
        "signal": score_signal(score),
        "value": value,
        "unit": "USD_millions" if metric_key.endswith("_musd") else "count",
        "notes": notes[:2000],
        "raw_json": _json_safe(raw),
    }


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value

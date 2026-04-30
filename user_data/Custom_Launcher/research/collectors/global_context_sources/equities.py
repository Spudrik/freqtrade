from __future__ import annotations

import csv
from io import StringIO
from typing import Any
from urllib.parse import quote

from .common import clamp, fetch_public_text, float_or_none, fmt_pct, iso_from_date_time, mean, pct_change, result, score_signal


def fetch_stooq_quotes(source: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    url = str(source.get("url") or "https://stooq.com/q/l/").strip()
    symbols = [str(item).strip() for item in source.get("symbols") or [] if str(item).strip()]
    fields = str(source.get("fields") or "sd2t2ohlcvp")
    if not symbols:
        raise ValueError("Stooq source requires symbols.")
    separator = "&" if "?" in url else "?"
    request_url = f"{url}{separator}s={quote(' '.join(symbols))}&f={fields}&h&e=csv"
    text = fetch_public_text(
        request_url,
        timeout_seconds=int(config.get("request_timeout_seconds", 20)),
        user_agent=str(config.get("user_agent") or "FreQ-GlobalContextCollector/1.0"),
    )
    rows = list(csv.DictReader(StringIO(text)))
    return {"request_url": request_url, "rows": rows}


def normalize_stooq_quotes(source: dict[str, Any], payload: Any, *, store_raw: bool = True) -> dict[str, Any]:
    rows = payload.get("rows") if isinstance(payload, dict) else []
    quote_rows = [row for row in rows if isinstance(row, dict) and str(row.get("Date") or "").upper() != "N/D"]
    if not quote_rows:
        raise ValueError("Stooq returned no usable quote rows.")
    changes: list[float] = []
    notes: list[str] = []
    source_ts_values: list[str] = []
    for row in quote_rows:
        symbol = str(row.get("Symbol") or "").upper()
        close = float_or_none(row.get("Close"))
        prev = float_or_none(row.get("Prev"))
        change = pct_change(close, prev)
        if change is not None:
            changes.append(change)
        source_ts = iso_from_date_time(row.get("Date"), row.get("Time"))
        if source_ts:
            source_ts_values.append(source_ts)
        label = _symbol_label(source, symbol)
        notes.append(f"{label} {fmt_pct(change)}")
    avg_change = mean(changes)
    score = _risk_score_from_change(avg_change, inverse=bool(source.get("inverse_risk_score", False)))
    return result(
        source,
        metric_key=str(source.get("metric_key") or "stooq_quote_basket_change"),
        score=score,
        signal=score_signal(score),
        value=avg_change,
        unit="percent",
        notes=", ".join(notes),
        source_ts=max(source_ts_values) if source_ts_values else None,
        raw=payload if store_raw else None,
    )


def _risk_score_from_change(change: float | None, *, inverse: bool) -> float:
    if change is None:
        return 50.0
    direction = -1.0 if inverse else 1.0
    return clamp(50.0 + direction * change * 12.0)


def _symbol_label(source: dict[str, Any], symbol: str) -> str:
    labels = source.get("symbol_labels")
    if isinstance(labels, dict):
        value = labels.get(symbol) or labels.get(symbol.lower()) or labels.get(symbol.upper())
        if value:
            return str(value)
    return symbol.replace(".US", "")

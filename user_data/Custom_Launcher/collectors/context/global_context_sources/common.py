from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from urllib.request import Request, urlopen

from ..research_collector_common import utc_now


def fetch_public_json(url: str, *, timeout_seconds: int, user_agent: str) -> Any:
    request = Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_public_text(url: str, *, timeout_seconds: int, user_agent: str) -> str:
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read().decode("utf-8")


def result(
    source: dict[str, Any],
    *,
    metric_key: str,
    score: float | None,
    signal: str,
    value: float | None,
    unit: str,
    notes: str,
    source_score: float | None = None,
    calc_score: float | None = None,
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
        "source_score": source_score,
        "calc_score": calc_score,
        "signal": signal,
        "value": value,
        "unit": unit,
        "notes": notes[:2000],
        "raw_json": raw,
    }


def nested_float(payload: dict[str, Any], *keys: str) -> float | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return float_or_none(value)


def float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "" or value == "N/D":
            return None
        return float(value)
    except Exception:
        return None


def mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def pct_change(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None or previous <= 0:
        return None
    return (current - previous) / previous * 100.0


def clamp(value: float | None, low: float = 0.0, high: float = 100.0) -> float:
    if value is None:
        return 50.0
    return max(low, min(high, float(value)))


def score_signal(score: float | None) -> str:
    value = clamp(score)
    if value >= 65:
        return "Greed"
    if value <= 35:
        return "Fear"
    return "Neutral"


def score_label(score: float | None) -> str:
    value = clamp(score)
    if value >= 75:
        return "Strong Greed"
    if value >= 60:
        return "Greed"
    if value <= 25:
        return "Strong Fear"
    if value <= 40:
        return "Fear"
    return "Neutral"


def compact_usd(value: float | None) -> str:
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


def fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}%"


def unix_to_iso(value: Any) -> str | None:
    try:
        number = float(value)
        if number > 10_000_000_000:
            number = number / 1000.0
        return datetime.fromtimestamp(number, timezone.utc).isoformat()
    except Exception:
        return None


def iso_from_date_time(date_text: Any, time_text: Any = "") -> str | None:
    date_value = str(date_text or "").strip()
    time_value = str(time_text or "").strip()
    if not date_value or date_value == "N/D":
        return None
    for candidate in (f"{date_value}T{time_value}", date_value):
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat()
        except Exception:
            continue
    return None

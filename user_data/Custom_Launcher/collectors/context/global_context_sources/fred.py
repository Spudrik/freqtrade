from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlencode

from .common import clamp, fetch_public_json, float_or_none, fmt_pct, pct_change, result, score_signal


def fetch_fred_series_basket(source: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    api_key = str(source.get("api_key") or config.get("fred_api_key") or "").strip()
    env_name = str(source.get("api_key_env") or config.get("fred_api_key_env") or "FRED_API_KEY")
    if not api_key and env_name:
        api_key = str(os.environ.get(env_name, "")).strip()
    if not api_key:
        raise ValueError(f"FRED API key missing. Set {env_name} or add api_key in local config.")
    series_ids = [str(item).strip() for item in source.get("series_ids") or [] if str(item).strip()]
    if not series_ids:
        raise ValueError("FRED source requires series_ids.")
    base_url = str(source.get("url") or "https://api.stlouisfed.org/fred/series/observations")
    payload: dict[str, Any] = {"series": {}}
    for series_id in series_ids:
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 2,
        }
        payload["series"][series_id] = fetch_public_json(
            base_url + "?" + urlencode(params),
            timeout_seconds=int(config.get("request_timeout_seconds", 20)),
            user_agent=str(config.get("user_agent") or "FreQ-GlobalContextCollector/1.0"),
        )
    return payload


def normalize_fred_series_basket(source: dict[str, Any], payload: Any, *, store_raw: bool = True) -> dict[str, Any]:
    series_payload = payload.get("series") if isinstance(payload, dict) else {}
    if not isinstance(series_payload, dict):
        series_payload = {}
    changes: list[float] = []
    levels: dict[str, float] = {}
    notes: list[str] = []
    source_ts_values: list[str] = []
    labels = source.get("series_labels") if isinstance(source.get("series_labels"), dict) else {}
    inverse_series = {str(item) for item in source.get("inverse_risk_series") or []}
    inverse_weight = 0.0
    normal_weight = 0.0
    for series_id, data in series_payload.items():
        observations = data.get("observations") if isinstance(data, dict) else []
        usable = [row for row in observations if isinstance(row, dict) and float_or_none(row.get("value")) is not None]
        if not usable:
            continue
        latest = usable[0]
        previous = usable[1] if len(usable) > 1 else {}
        latest_value = float_or_none(latest.get("value"))
        previous_value = float_or_none(previous.get("value"))
        change = pct_change(latest_value, previous_value)
        if change is not None:
            changes.append(change)
            if str(series_id) in inverse_series:
                inverse_weight += -change
            else:
                normal_weight += change
        if latest_value is not None:
            levels[str(series_id)] = latest_value
        date_text = str(latest.get("date") or "")
        if date_text:
            source_ts_values.append(f"{date_text}T00:00:00+00:00")
        label = str(labels.get(series_id) or series_id)
        notes.append(f"{label} {fmt_pct(change)} level {latest_value:g}" if latest_value is not None else f"{label} {fmt_pct(change)}")
    combined_change = (sum(changes) / len(changes)) if changes else None
    score = clamp(50.0 + normal_weight * 8.0 + inverse_weight * 8.0)
    return result(
        source,
        metric_key=str(source.get("metric_key") or "fred_series_basket_change"),
        score=score,
        calc_score=score,
        signal=score_signal(score),
        value=combined_change,
        unit="percent",
        notes=", ".join(notes) or "No usable FRED observations.",
        source_ts=max(source_ts_values) if source_ts_values else None,
        raw=payload if store_raw else None,
    )

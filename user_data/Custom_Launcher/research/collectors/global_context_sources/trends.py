from __future__ import annotations

import inspect
from datetime import date, datetime
from typing import Any

import pandas as pd

from ..research_collector_common import slugify, utc_now
from .common import clamp, mean, score_signal


def fetch_google_trends_interest(source: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    try:
        _patch_pytrends_retry_compat()
        from pytrends.request import TrendReq
    except Exception as exc:
        raise RuntimeError("Google Trends source requires the optional pytrends package.") from exc

    keywords = [str(item).strip() for item in source.get("keywords") or [] if str(item).strip()]
    if not keywords:
        raise ValueError("Google Trends source requires at least one keyword.")
    if len(keywords) > 5:
        raise ValueError("Google Trends supports at most 5 keywords per request.")

    hl = str(source.get("hl") or config.get("google_trends_hl") or "en-US")
    tz = int(source.get("tz", config.get("google_trends_tz", 0)))
    timeframe = str(source.get("timeframe") or "now 7-d")
    geo = str(source.get("geo") or "")
    gprop = str(source.get("gprop") or "")
    cat = int(source.get("cat") or 0)
    pytrends = TrendReq(
        hl=hl,
        tz=tz,
        retries=int(source.get("retries", 0)),
        backoff_factor=float(source.get("backoff_factor", 0.0)),
    )
    pytrends.build_payload(keywords, cat=cat, timeframe=timeframe, geo=geo, gprop=gprop)
    frame = pytrends.interest_over_time()
    if frame.empty:
        raise ValueError("Google Trends returned no interest-over-time rows.")
    frame = frame.drop(columns=["isPartial"], errors="ignore").reset_index()
    return {
        "keywords": keywords,
        "timeframe": timeframe,
        "geo": geo,
        "gprop": gprop,
        "cat": cat,
        "rows": _json_safe(frame.to_dict(orient="records")),
        "collected_at": utc_now(),
    }


def _patch_pytrends_retry_compat() -> None:
    try:
        from urllib3.util import retry
    except Exception:
        return
    signature = inspect.signature(retry.Retry.__init__)
    if "method_whitelist" in signature.parameters or "allowed_methods" not in signature.parameters:
        return
    original_init = retry.Retry.__init__
    if getattr(original_init, "_freqtrade_pytrends_compat", False):
        return

    def compatible_init(self, *args: Any, **kwargs: Any) -> None:
        if "method_whitelist" in kwargs and "allowed_methods" not in kwargs:
            kwargs["allowed_methods"] = kwargs.pop("method_whitelist")
        original_init(self, *args, **kwargs)

    compatible_init._freqtrade_pytrends_compat = True  # type: ignore[attr-defined]
    retry.Retry.__init__ = compatible_init  # type: ignore[method-assign]


def normalize_google_trends_interest(source: dict[str, Any], payload: Any, *, store_raw: bool = True) -> list[dict[str, Any]]:
    rows = payload.get("rows") if isinstance(payload, dict) else []
    keywords = [str(item).strip() for item in payload.get("keywords", [])] if isinstance(payload, dict) else []
    if not isinstance(rows, list) or not rows:
        raise ValueError("Google Trends payload has no rows.")

    emit = str(source.get("emit") or "latest").lower()
    selected_rows = rows if emit == "all" else [rows[-1]]
    collected_at = str(payload.get("collected_at") or utc_now()) if isinstance(payload, dict) else utc_now()
    availability_mode = str(source.get("availability_mode") or "collected_at").lower()
    metric_prefix = str(source.get("metric_prefix") or "google_trends")
    normalized: list[dict[str, Any]] = []
    for raw_row in selected_rows:
        if not isinstance(raw_row, dict):
            continue
        source_ts = _source_ts(raw_row)
        ts = source_ts if availability_mode == "source_ts" and source_ts else collected_at
        values: list[float] = []
        for keyword in keywords:
            value = _float_or_none(raw_row.get(keyword))
            if value is None:
                continue
            values.append(value)
            normalized.append(
                _trend_row(
                    source,
                    metric_key=f"{metric_prefix}_{slugify(keyword)}",
                    value=value,
                    ts=ts,
                    source_ts=source_ts,
                    notes=f"Google Trends interest for '{keyword}' over {payload.get('timeframe')}; availability_mode={availability_mode}.",
                    raw=_json_safe(raw_row) if store_raw else None,
                )
            )
        composite = mean(values)
        if composite is not None:
            normalized.append(
                _trend_row(
                    source,
                    metric_key=str(source.get("metric_key") or f"{metric_prefix}_attention_composite"),
                    value=composite,
                    ts=ts,
                    source_ts=source_ts,
                    notes=f"Mean Google Trends interest across {', '.join(keywords)} over {payload.get('timeframe')}; availability_mode={availability_mode}.",
                    raw=_json_safe(payload) if store_raw else None,
                )
            )
    if not normalized:
        raise ValueError("Google Trends payload produced no numeric rows.")
    return normalized


def _trend_row(
    source: dict[str, Any],
    *,
    metric_key: str,
    value: float,
    ts: str,
    source_ts: str | None,
    notes: str,
    raw: Any,
) -> dict[str, Any]:
    score = clamp(value)
    return {
        "ts": ts,
        "source_ts": source_ts,
        "source_id": source.get("id"),
        "source_group": source.get("source_group") or "attention",
        "source_type": source.get("type") or "google_trends_interest",
        "metric_key": metric_key,
        "score": score,
        "source_score": score,
        "calc_score": score,
        "signal": score_signal(score),
        "value": value,
        "unit": "index",
        "notes": notes[:2000],
        "raw_json": raw,
    }


def _source_ts(row: dict[str, Any]) -> str | None:
    for key in ("date", "datetime", "time"):
        value = row.get(key)
        if value is None or value == "":
            continue
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.notna(parsed):
            return pd.Timestamp(parsed).isoformat()
    return None


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

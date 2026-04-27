"""Training and validation window helpers for simplified Explorer runs."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

FULL_CYCLE_WINDOW: dict[str, Any] = {
    "name": "full_cycle_2020_2026",
    "regime": "mixed",
    "segment_type": "full_cycle",
    "market_state": "mixed",
    "timerange": "20200101-20260101",
    "segment_length_months": 72.0,
    "rationale": "Full available 2020-2026 validation cycle used as the primary long-range sanity check.",
}


def load_json(path: str | Path, default: Any) -> Any:
    path = Path(path)
    if not path.exists():
        return deepcopy(default)
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except Exception:
            continue
    return deepcopy(default)


def normalize_window(window: dict[str, Any]) -> dict[str, Any]:
    name = str(window.get("name") or "")
    timerange = str(window.get("timerange") or "")
    regime = str(window.get("regime") or window.get("market_state") or "")
    segment_type = str(window.get("segment_type") or regime or "market_state")
    normalized = dict(window)
    normalized.update(
        {
            "name": name,
            "timerange": timerange,
            "regime": regime,
            "segment_type": segment_type,
            "market_state": str(window.get("market_state") or regime),
        }
    )
    return normalized


def load_window_manifest(path: str | Path) -> list[dict[str, Any]]:
    payload = load_json(path, {})
    raw_windows = payload.get("market_windows") if isinstance(payload, dict) else []
    windows = [normalize_window(window) for window in raw_windows if isinstance(window, dict)]
    full = normalize_window(FULL_CYCLE_WINDOW)
    by_key: dict[str, dict[str, Any]] = {str(full["name"]): full}
    for window in windows:
        key = str(window.get("name") or window.get("timerange") or "")
        if key:
            by_key[key] = window
    return list(by_key.values())


def _parse_selection(raw: str | list[Any] | None) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    text = str(raw).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    return [part.strip() for part in text.replace(",", " ").split() if part.strip()]


def resolve_windows(all_windows: list[dict[str, Any]], selection: str | list[Any] | None, *, label: str) -> list[dict[str, Any]]:
    requested = _parse_selection(selection)
    if not requested:
        raise ValueError(f"Explorer needs at least one selected {label} window.")
    by_name = {str(window.get("name") or ""): window for window in all_windows if str(window.get("name") or "")}
    by_timerange = {str(window.get("timerange") or ""): window for window in all_windows if str(window.get("timerange") or "")}
    resolved: list[dict[str, Any]] = []
    missing: list[str] = []
    seen: set[str] = set()
    for item in requested:
        if isinstance(item, dict):
            window = normalize_window(item)
        else:
            key = str(item).strip()
            window = by_name.get(key) or by_timerange.get(key)
            if window is None:
                missing.append(key)
                continue
            window = normalize_window(window)
        dedupe_key = str(window.get("name") or window.get("timerange") or "")
        if dedupe_key and dedupe_key not in seen:
            resolved.append(window)
            seen.add(dedupe_key)
    if missing:
        raise ValueError(f"Unknown {label} window(s): " + ", ".join(missing))
    if not resolved:
        raise ValueError(f"Explorer needs at least one selected {label} window.")
    return resolved


def compact_window(window: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(window.get("name") or ""),
        "regime": str(window.get("regime") or ""),
        "segment_type": str(window.get("segment_type") or ""),
        "timerange": str(window.get("timerange") or ""),
    }


def window_label(window: dict[str, Any]) -> str:
    return str(window.get("name") or window.get("timerange") or "-")

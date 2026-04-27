"""Compact summary and full audit metadata for simplified Explorer runs."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from .explorer_catalog import catalog_table
from .explorer_scoring import score_table_rows

SCHEMA_VERSION = 2


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")


def compact_validation_breakdown(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    champ_windows = {str(row.get("window")): row for row in ((comparison.get("champion") or {}).get("windows") or [])}
    chall_windows = {str(row.get("window")): row for row in ((comparison.get("challenger") or {}).get("windows") or [])}
    ordered = list(dict.fromkeys(list(champ_windows.keys()) + list(chall_windows.keys())))
    rows: list[dict[str, Any]] = []
    for key in ordered:
        champion = champ_windows.get(key) or {}
        challenger = chall_windows.get(key) or {}
        rows.append(
            {
                "window": key,
                "regime": challenger.get("regime") or champion.get("regime") or "",
                "champion_objective": champion.get("objective"),
                "challenger_objective": challenger.get("objective"),
                "champion_max_drawdown_pct": champion.get("max_drawdown_pct"),
                "challenger_max_drawdown_pct": challenger.get("max_drawdown_pct"),
                "champion_score": champion.get("final_score"),
                "challenger_score": challenger.get("final_score"),
                "delta_score": (challenger.get("final_score") - champion.get("final_score"))
                if isinstance(challenger.get("final_score"), (int, float)) and isinstance(champion.get("final_score"), (int, float))
                else None,
                "guard": challenger.get("hard_guard") or "",
            }
        )
    return rows


def latest_summary(
    *,
    run_id: str,
    loop_index: int,
    loop_total: int,
    strategy_file: str | Path,
    strategy_class: str,
    target: dict[str, Any],
    target_selection: str,
    training_windows: list[dict[str, Any]],
    validation_windows: list[dict[str, Any]],
    comparison: dict[str, Any],
    params_changed_count: int,
    catalog: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    summary = {
        "schema_version": SCHEMA_VERSION,
        "saved_at": datetime.now().astimezone().isoformat(),
        "run_id": run_id,
        "loop_index": int(loop_index),
        "loop_total": int(loop_total),
        "strategy_file": str(Path(strategy_file).resolve()),
        "strategy_class": str(strategy_class or ""),
        "target_type": str(target.get("target_type") or ""),
        "target_name": str(target.get("target_name") or ""),
        "target_label": str(target.get("target_label") or ""),
        "target_selection": str(target_selection or ""),
        "search_breadth": str(target.get("search_breadth") or ""),
        "training_windows": training_windows,
        "validation_windows": validation_windows,
        "decision_code": str(comparison.get("decision_code") or ""),
        "guard": str(comparison.get("guard") or ""),
        "accepted": bool(comparison.get("accepted")),
        "params_changed_count": int(params_changed_count),
        "champion": deepcopy(comparison.get("champion") or {}),
        "challenger": deepcopy(comparison.get("challenger") or {}),
        "delta": deepcopy(comparison.get("delta") or {}),
        "score_table": score_table_rows(comparison),
        "validation_breakdown": compact_validation_breakdown(comparison),
        "target_coverage": catalog_table(catalog, state),
    }
    return summary


def write_latest_summary(path: str | Path, summary: dict[str, Any]) -> None:
    save_json(path, summary)


def write_audit(path: str | Path, audit: dict[str, Any]) -> None:
    audit = dict(audit)
    audit.setdefault("schema_version", SCHEMA_VERSION)
    audit.setdefault("saved_at", datetime.now().astimezone().isoformat())
    save_json(path, audit)


def emit_status(event: str, payload: dict[str, Any]) -> None:
    record = {"event": event, **payload}
    print("EXPLORER_STATUS_JSON " + json.dumps(record, sort_keys=True, default=str), flush=True)

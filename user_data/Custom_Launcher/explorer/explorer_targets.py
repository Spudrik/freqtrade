"""Family/mode target selection and Targeted/Open breadth resolution."""

from __future__ import annotations

from datetime import datetime
import random
from typing import Any

from .explorer_catalog import (
    params_for_target,
    params_for_family,
    spaces_for_target_params,
    target_pool,
    target_label,
    open_support_families_for_target,
)

CANONICAL_FAMILIES = {"entries", "exits", "adjust_position", "stake", "risk"}


def target_category(target_type: str, name: str) -> str:
    normalized_name = str(name or "").strip().lower()
    text = f"{target_type}:{normalized_name}"
    if target_type == "family" and normalized_name in CANONICAL_FAMILIES:
        return normalized_name
    if any(token in text for token in ("adjust_position", "adjust", "add", "rebuy", "dca", "ladder", "scale")):
        return "adjust_position"
    if any(token in text for token in ("exit", "peel", "trail", "target")):
        return "exits"
    if any(token in text for token in ("stake", "capital", "size", "leverage", "allocation")):
        return "stake"
    if any(token in text for token in ("risk", "stop", "drawdown", "guard", "liquidat")):
        return "risk"
    if any(token in text for token in ("structure", "volume", "trendline", "line_quality", "hourly", "execution")):
        return "entries"
    if any(token in text for token in ("entry", "seed", "break", "reclaim", "retest", "support", "res_", "sup_")):
        return "entries"
    return "default"


def _usage_bucket(state: dict[str, Any], target_type: str, breadth: str) -> dict[str, int]:
    key = f"{target_type}_{'open_' if breadth == 'open' else ''}usage_counts"
    bucket = state.setdefault(key, {})
    return bucket if isinstance(bucket, dict) else {}


def choose_target(catalog: dict[str, Any], target_type: str, target_selection: str, target_name: str, breadth: str, state: dict[str, Any], rng: random.Random) -> str:
    names = target_pool(catalog, target_type)
    if not names:
        raise ValueError(f"No {target_type} targets were found in the strategy catalog.")
    if target_selection == "specific":
        if not target_name:
            raise ValueError(f"Specific {target_type} mode requires --target-name.")
        if target_name not in names:
            raise ValueError(f"Unknown {target_type} target: {target_name}")
        return target_name
    if target_selection != "random":
        raise ValueError(f"Unsupported target_selection: {target_selection}")
    usage = _usage_bucket(state, target_type, breadth)
    weights = [1.0 / (1.0 + max(0, int(usage.get(name) or 0))) for name in names]
    return rng.choices(names, weights=weights, k=1)[0]


def resolve_params(catalog: dict[str, Any], target_type: str, name: str, breadth: str, strategy_spaces: dict[str, str]) -> dict[str, Any]:
    if breadth not in {"targeted", "open"}:
        raise ValueError(f"Unsupported search breadth: {breadth}")
    primary = set(params_for_target(catalog, target_type, name))
    resolved = set(primary)
    missing_support_families: list[str] = []
    support_families: list[str] = []
    if breadth == "open":
        support_families = open_support_families_for_target(catalog, target_type, name)
        for family in support_families:
            family_params = set(params_for_family(catalog, family))
            if not family_params:
                missing_support_families.append(family)
                continue
            resolved.update(family_params)
    active = {param for param in resolved if param in strategy_spaces}
    if not active:
        raise ValueError(f"Target {target_label(target_type, name)} resolved zero active tunable params.")
    spaces = spaces_for_target_params(catalog, active)
    return {
        "target_type": target_type,
        "target_name": name,
        "target_label": target_label(target_type, name),
        "search_breadth": breadth,
        "primary_params": sorted(param for param in primary if param in strategy_spaces),
        "resolved_params": sorted(active),
        "spaces": spaces,
        "support_families": support_families,
        "missing_open_support_families": missing_support_families,
    }


def update_usage_counts(state: dict[str, Any], target_type: str, name: str, breadth: str, accepted: bool, score_delta: float | None) -> None:
    now = datetime.now().astimezone().isoformat()
    bucket = _usage_bucket(state, target_type, breadth)
    bucket[name] = int(bucket.get(name) or 0) + 1
    state[f"{target_type}_{'open_' if breadth == 'open' else ''}usage_counts"] = bucket
    label = target_label(target_type, name)
    state.setdefault("last_run_by_target", {})[label] = now
    if score_delta is not None:
        state.setdefault("last_score_delta_by_target", {})[label] = float(score_delta)
    if accepted:
        accept_key = f"{target_type}_accept_counts"
        accept_bucket = state.setdefault(accept_key, {})
        accept_bucket[name] = int(accept_bucket.get(name) or 0) + 1

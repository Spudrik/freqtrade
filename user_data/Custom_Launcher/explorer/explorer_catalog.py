"""Family/mode catalog helpers for the simplified Explorer workflow.

The underlying strategy may contain many tag namespaces for legacy reasons.
Normal Explorer targeting intentionally exposes only family:* and mode:* tags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .tag_catalog import build_param_catalog, params_for_family, params_for_tag, spaces_for_params


CANONICAL_FAMILIES = {"entries", "exits", "adjust_position", "stake", "risk"}

OPEN_SUPPORT_CATEGORIES: dict[str, list[str]] = {
    "entries": ["entries", "exits", "risk"],
    "exits": ["exits", "risk"],
    "adjust_position": ["adjust_position", "stake", "exits", "risk"],
    "stake": ["stake", "adjust_position", "risk"],
    "risk": ["risk", "exits"],
    "default": ["entries", "exits", "adjust_position", "stake", "risk"],
}

@dataclass(frozen=True)
class ExplorerTarget:
    target_type: str
    name: str
    label: str
    params: frozenset[str]
    buy_count: int
    sell_count: int


def load_catalog(strategy_file: str | Path, strategy_class: str) -> dict[str, Any]:
    return build_param_catalog(strategy_file, strategy_class)


def family_names(catalog: dict[str, Any]) -> list[str]:
    return sorted(str(name) for name in (catalog.get("family_index") or {}).keys() if str(name))


def mode_names(catalog: dict[str, Any]) -> list[str]:
    tags = catalog.get("tag_index") or {}
    modes: set[str] = set()
    for tag in tags:
        text = str(tag)
        if text.startswith("mode:"):
            modes.add(text.split(":", 1)[1])
    return sorted(modes)


def params_for_mode(catalog: dict[str, Any], mode: str) -> set[str]:
    return params_for_tag(catalog, f"mode:{mode}")


def params_for_target(catalog: dict[str, Any], target_type: str, name: str) -> set[str]:
    if target_type == "family":
        return params_for_family(catalog, name)
    if target_type == "mode":
        return params_for_mode(catalog, name)
    raise ValueError(f"Unsupported target_type: {target_type}")


def target_label(target_type: str, name: str) -> str:
    if target_type not in {"family", "mode"}:
        raise ValueError(f"Unsupported target_type: {target_type}")
    return f"{target_type}:{name}"


def target_pool(catalog: dict[str, Any], target_type: str) -> list[str]:
    if target_type == "family":
        return family_names(catalog)
    if target_type == "mode":
        return mode_names(catalog)
    raise ValueError(f"Unsupported target_type: {target_type}")


def space_counts(catalog: dict[str, Any], params: set[str]) -> dict[str, int]:
    catalog_params = catalog.get("params") or {}
    counts: dict[str, int] = {"buy": 0, "sell": 0}
    for name in params:
        space = str((catalog_params.get(name) or {}).get("space") or "")
        if space in counts:
            counts[space] += 1
    return counts


def make_target(catalog: dict[str, Any], target_type: str, name: str, active_params: set[str] | None = None) -> ExplorerTarget:
    params = params_for_target(catalog, target_type, name)
    if active_params is not None:
        params = {param for param in params if param in active_params}
    counts = space_counts(catalog, params)
    return ExplorerTarget(
        target_type=target_type,
        name=name,
        label=target_label(target_type, name),
        params=frozenset(sorted(params)),
        buy_count=int(counts.get("buy") or 0),
        sell_count=int(counts.get("sell") or 0),
    )


def _target_category(target_type: str, name: str) -> str:
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


def _families_by_category(catalog: dict[str, Any]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for family in family_names(catalog):
        category = _target_category("family", family)
        grouped.setdefault(category, []).append(family)
    return {category: sorted(set(names)) for category, names in grouped.items()}


def open_support_families_for_target(catalog: dict[str, Any], target_type: str, name: str) -> list[str]:
    """Return real strategy family names that support open-breadth testing.

    The open-breadth rules are expressed in canonical categories such as
    ``entries`` and ``exits``, but strategies are allowed to use concrete family
    names like ``daily_structure`` or ``exit_profile``. This bridges the two.
    """

    category = _target_category(target_type, name)
    support_categories = OPEN_SUPPORT_CATEGORIES.get(category) or OPEN_SUPPORT_CATEGORIES["default"]
    grouped = _families_by_category(catalog)
    support: list[str] = []
    for support_category in support_categories:
        support.extend(grouped.get(support_category, []))
        if support_category in (catalog.get("family_index") or {}):
            support.append(support_category)
    return sorted(dict.fromkeys(support))


def _open_support_param_details(catalog: dict[str, Any], target_type: str, name: str, active_params: set[str]) -> tuple[list[str], list[dict[str, Any]]]:
    primary = {param for param in params_for_target(catalog, target_type, name) if param in active_params}
    support_families = open_support_families_for_target(catalog, target_type, name)
    by_param: dict[str, set[str]] = {}
    for family in support_families:
        family_params = {param for param in params_for_family(catalog, family) if param in active_params}
        for param_name in family_params:
            if param_name in primary:
                continue
            by_param.setdefault(param_name, set()).add(family)
    catalog_params = catalog.get("params") or {}
    rows = [
        {
            "name": param_name,
            "space": str((catalog_params.get(param_name) or {}).get("space") or ""),
            "source_family": ", ".join(sorted(by_param.get(param_name) or [])),
        }
        for param_name in sorted(by_param.keys())
    ]
    return support_families, rows


def catalog_table(catalog: dict[str, Any], state: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    state = state or {}
    rows: list[dict[str, Any]] = []
    active_params = set(str(name) for name in (catalog.get("params") or {}).keys())
    catalog_params = catalog.get("params") or {}
    for target_type, names in (("family", family_names(catalog)), ("mode", mode_names(catalog))):
        targeted_counts = state.get(f"{target_type}_usage_counts") or {}
        open_counts = state.get(f"{target_type}_open_usage_counts") or {}
        accepted_counts = state.get(f"{target_type}_accept_counts") or {}
        for name in names:
            target = make_target(catalog, target_type, name, active_params)
            support_families, open_support_details = _open_support_param_details(catalog, target_type, name, active_params)
            param_details = [
                {
                    "name": param_name,
                    "space": str((catalog_params.get(param_name) or {}).get("space") or ""),
                }
                for param_name in sorted(target.params)
            ]
            rows.append(
                {
                    "type": target_type,
                    "name": name,
                    "label": target.label,
                    "param_count": len(target.params),
                    "buy_params": target.buy_count,
                    "sell_params": target.sell_count,
                    "targeted_runs": int(targeted_counts.get(name) or 0),
                    "open_runs": int(open_counts.get(name) or 0),
                    "accepted": int(accepted_counts.get(name) or 0),
                    "last_score_delta": (state.get("last_score_delta_by_target") or {}).get(target.label, None),
                    "last_run": (state.get("last_run_by_target") or {}).get(target.label, ""),
                    "param_details": param_details,
                    "open_support_families": support_families,
                    "open_support_param_details": open_support_details,
                }
            )
    return rows


def spaces_for_target_params(catalog: dict[str, Any], params: set[str]) -> list[str]:
    return spaces_for_params(catalog, params)

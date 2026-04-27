#!/usr/bin/env python3
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from typing import Any, Iterable

PARAMETER_CALL_NAMES = {
    "BooleanParameter",
    "CategoricalParameter",
    "DecimalParameter",
    "IntParameter",
}

PREFERRED_NAMESPACE_ORDER = [
    "family",
    "mode",
    "batch",
    "domain",
    "action",
    "switch",
    "controls",
    "signal",
    "scope",
    "regime",
    "role",
    "side",
    "timeframe",
]


def split_tag(tag: str) -> tuple[str, str]:
    if ":" not in str(tag):
        return "", str(tag)
    namespace, value = str(tag).split(":", 1)
    return namespace, value


def tag_namespace(tag: str) -> str:
    return split_tag(tag)[0]


def tag_value(tag: str) -> str:
    return split_tag(tag)[1]


def ordered_namespaces(namespaces: dict[str, Any]) -> list[str]:
    known = [namespace for namespace in PREFERRED_NAMESPACE_ORDER if namespace in namespaces]
    unknown = sorted(namespace for namespace in namespaces if namespace not in PREFERRED_NAMESPACE_ORDER)
    return known + unknown


def _literal_or_none(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except Exception:
        if isinstance(node, ast.Name):
            return node.id
        return None


def _call_name(node: ast.AST | None) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _extract_param_info(value: ast.AST) -> tuple[dict[str, Any] | None, list[str]]:
    tags: list[str] = []
    param_call: ast.Call | None = None
    if isinstance(value, ast.Call) and _call_name(value) == "tagged_parameter":
        if value.args and isinstance(value.args[0], ast.Call):
            param_call = value.args[0]
            tags = [str(_literal_or_none(arg)) for arg in value.args[1:] if _literal_or_none(arg) is not None]
    elif isinstance(value, ast.Call) and (_call_name(value) in PARAMETER_CALL_NAMES):
        param_call = value
    if param_call is None:
        return None, []
    call_name = _call_name(param_call)
    if call_name not in PARAMETER_CALL_NAMES:
        return None, []
    keywords = {kw.arg: kw.value for kw in param_call.keywords if kw.arg}
    info = {
        "parameter_type": call_name,
        "space": _literal_or_none(keywords.get("space")) or "",
        "default": _literal_or_none(keywords.get("default")),
        "optimize": bool(_literal_or_none(keywords.get("optimize"))),
        "load": _literal_or_none(keywords.get("load")),
        "tags": tags,
    }
    return info, tags


def _is_parameter_object(value: Any) -> bool:
    return bool(value is not None and value.__class__.__name__.endswith("Parameter"))


def _runtime_default(value: Any) -> Any:
    if hasattr(value, "default"):
        return getattr(value, "default")
    if hasattr(value, "value"):
        return getattr(value, "value")
    return None


def _load_runtime_strategy_class(strategy_path: Path, strategy_class: str) -> type:
    module_name = f"_explorer_strategy_{strategy_path.stem}_{abs(hash(str(strategy_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, strategy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load strategy module spec for {strategy_path}")
    candidate_paths = [str(strategy_path.parent), str(strategy_path.parent.parent.parent)]
    inserted_paths: list[str] = []
    for added_path in candidate_paths:
        if added_path and added_path not in sys.path:
            sys.path.insert(0, added_path)
            inserted_paths.append(added_path)
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        for added_path in inserted_paths:
            try:
                sys.path.remove(added_path)
            except ValueError:
                pass
    cls = getattr(module, strategy_class, None)
    if cls is None:
        raise ValueError(f"Strategy class '{strategy_class}' not found in {strategy_path.name}")
    if not isinstance(cls, type):
        raise TypeError(f"Strategy attribute '{strategy_class}' is not a class in {strategy_path.name}")
    return cls


def _runtime_param_infos(strategy_path: Path, strategy_class: str) -> dict[str, dict[str, Any]]:
    cls = _load_runtime_strategy_class(strategy_path, strategy_class)
    params: dict[str, dict[str, Any]] = {}
    for name in dir(cls):
        value = getattr(cls, name, None)
        if not _is_parameter_object(value):
            continue
        tags = [str(tag) for tag in (getattr(value, "batch_tags", ()) or []) if str(tag)]
        params[str(name)] = {
            "parameter_type": value.__class__.__name__,
            "space": str(getattr(value, "space", "") or ""),
            "default": _runtime_default(value),
            "optimize": bool(getattr(value, "optimize", False)),
            "load": getattr(value, "load", None),
            "tags": tags,
            "name": str(name),
        }
    return params


def _catalog_from_params(strategy_path: Path, strategy_class: str, params: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tag_index: dict[str, list[str]] = {}
    family_index: dict[str, list[str]] = {}
    for target_name, info in params.items():
        for tag in info.get("tags") or []:
            tag_index.setdefault(str(tag), []).append(target_name)
            if str(tag).startswith("family:"):
                family_index.setdefault(str(tag).split(":", 1)[1], []).append(target_name)
    namespace_values: dict[str, list[str]] = {}
    namespace_index: dict[str, dict[str, list[str]]] = {}
    for tag in tag_index:
        namespace, value = split_tag(tag)
        if namespace:
            namespace_values.setdefault(namespace, []).append(value)
            namespace_index.setdefault(namespace, {})[tag] = sorted(set(tag_index[tag]))
    for namespace, values in namespace_values.items():
        namespace_values[namespace] = sorted(set(values))
    for namespace in list(namespace_index):
        namespace_index[namespace] = dict(sorted(namespace_index[namespace].items()))
    for key in list(tag_index):
        tag_index[key] = sorted(set(tag_index[key]))
    for key in list(family_index):
        family_index[key] = sorted(set(family_index[key]))
    untagged = sorted(name for name, info in params.items() if not info.get("tags"))
    return {
        "strategy_file": str(strategy_path),
        "strategy_class": strategy_class,
        "params": params,
        "tag_index": dict(sorted(tag_index.items())),
        "family_index": dict(sorted(family_index.items())),
        "namespaces": dict(sorted(namespace_values.items())),
        "namespace_index": dict(sorted(namespace_index.items())),
        "untagged_params": untagged,
        "family_names": sorted(family_index.keys()),
        "param_names": sorted(params.keys()),
        "known_namespace_order": list(PREFERRED_NAMESPACE_ORDER),
        "preferred_namespace_order": ordered_namespaces(namespace_values),
    }


def build_param_catalog(strategy_file: str | Path, strategy_class: str) -> dict[str, Any]:
    strategy_path = Path(strategy_file).resolve()
    if not strategy_path.exists():
        raise FileNotFoundError(strategy_path)
    runtime_params = _runtime_param_infos(strategy_path, strategy_class)
    if runtime_params:
        return _catalog_from_params(strategy_path, strategy_class, runtime_params)
    source = strategy_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(strategy_path))
    class_node: ast.ClassDef | None = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == strategy_class:
            class_node = node
            break
    if class_node is None:
        raise ValueError(f"Strategy class '{strategy_class}' not found in {strategy_path.name}")

    params: dict[str, dict[str, Any]] = {}
    tag_index: dict[str, list[str]] = {}
    family_index: dict[str, list[str]] = {}
    for node in class_node.body:
        target_name: str | None = None
        value_node: ast.AST | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
            value_node = node.value
        if not target_name or value_node is None:
            continue
        info, tags = _extract_param_info(value_node)
        if info is None:
            continue
        info["name"] = target_name
        params[target_name] = info
        for tag in tags:
            tag_index.setdefault(tag, []).append(target_name)
            if tag.startswith("family:"):
                family_index.setdefault(tag.split(":", 1)[1], []).append(target_name)
    namespace_values: dict[str, list[str]] = {}
    namespace_index: dict[str, dict[str, list[str]]] = {}
    for tag in tag_index:
        namespace, value = split_tag(tag)
        if namespace:
            namespace_values.setdefault(namespace, []).append(value)
            namespace_index.setdefault(namespace, {})[tag] = sorted(set(tag_index[tag]))
    for namespace, values in namespace_values.items():
        namespace_values[namespace] = sorted(set(values))
    for namespace in list(namespace_index):
        namespace_index[namespace] = dict(sorted(namespace_index[namespace].items()))
    for key in list(tag_index):
        tag_index[key] = sorted(set(tag_index[key]))
    for key in list(family_index):
        family_index[key] = sorted(set(family_index[key]))
    untagged = sorted(name for name, info in params.items() if not info.get("tags"))
    return _catalog_from_params(strategy_path, strategy_class, params)


def params_for_tag(catalog: dict[str, Any], tag: str) -> set[str]:
    return {str(param) for param in (catalog.get("tag_index") or {}).get(str(tag), []) if str(param)}


def params_for_namespace_value(catalog: dict[str, Any], namespace: str, value: str) -> set[str]:
    return params_for_tag(catalog, f"{namespace}:{value}")


def params_for_namespace(catalog: dict[str, Any], namespace: str) -> set[str]:
    namespace_tags = (catalog.get("namespace_index") or {}).get(str(namespace), {})
    return {str(param) for params in namespace_tags.values() for param in params if str(param)}


def params_for_family(catalog: dict[str, Any], family: str) -> set[str]:
    return {str(param) for param in (catalog.get("family_index") or {}).get(str(family), []) if str(param)}


def params_for_family_tag_intersection(catalog: dict[str, Any], family: str, tag: str) -> set[str]:
    return params_for_family(catalog, family) & params_for_tag(catalog, tag)


def params_for_source(catalog: dict[str, Any], source: dict[str, Any]) -> set[str]:
    source_type = str(source.get("type") or "")
    if source_type == "family":
        return params_for_family(catalog, str(source.get("id") or source.get("family") or ""))
    if source_type == "tag":
        return params_for_tag(catalog, str(source.get("id") or source.get("tag") or ""))
    if source_type == "namespace":
        return params_for_namespace(catalog, str(source.get("id") or source.get("namespace") or ""))
    if source_type == "namespace_value":
        namespace = str(source.get("namespace") or "")
        value = str(source.get("value") or source.get("id") or "")
        return params_for_namespace_value(catalog, namespace, value)
    if source_type == "family_tag_intersection":
        return params_for_family_tag_intersection(catalog, str(source.get("family") or ""), str(source.get("tag") or ""))
    if source_type == "param":
        param = str(source.get("id") or "")
        return {param} if param in (catalog.get("params") or {}) else set()
    return set()


def spaces_for_params(catalog: dict[str, Any], params: Iterable[str]) -> list[str]:
    catalog_params = catalog.get("params") or {}
    return sorted(
        {
            str((catalog_params.get(str(param)) or {}).get("space") or "")
            for param in params
            if str((catalog_params.get(str(param)) or {}).get("space") or "")
        }
    )


def namespace_values(catalog: dict[str, Any], namespace: str) -> list[str]:
    return list((catalog.get("namespaces") or {}).get(str(namespace), []))


def catalog_summary(catalog: dict[str, Any]) -> dict[str, Any]:
    namespaces = catalog.get("namespaces") or {}
    return {
        "param_count": len(catalog.get("params") or {}),
        "tag_count": len(catalog.get("tag_index") or {}),
        "family_count": len(catalog.get("family_index") or {}),
        "namespace_count": len(namespaces),
        "namespaces": {
            namespace: len(values)
            for namespace, values in sorted(namespaces.items())
        },
        "preferred_namespace_order": list(catalog.get("preferred_namespace_order") or ordered_namespaces(namespaces)),
    }


def parse_tag_tokens(value: str | None) -> list[str]:
    if not value:
        return []
    normalized = str(value).replace(";", ",").replace("|", ",").replace("\n", ",")
    tokens: list[str] = []
    for part in normalized.split(","):
        token = part.strip()
        if token:
            tokens.append(token)
    return tokens


def matching_params_for_tag_set(catalog: dict[str, Any], tags: list[str]) -> set[str]:
    if not tags:
        return set()
    params = catalog.get("params") or {}
    matching: set[str] = set()
    for name, info in params.items():
        param_tags = set(info.get("tags") or [])
        if all(tag in param_tags for tag in tags):
            matching.add(str(name))
    return matching


def matching_params_for_tag_sets(catalog: dict[str, Any], tag_sets: list[list[str]]) -> set[str]:
    non_empty_sets = [list(dict.fromkeys(tags)) for tags in tag_sets if tags]
    if not non_empty_sets:
        return set(catalog.get("param_names") or [])
    combined: set[str] = set()
    for tags in non_empty_sets:
        combined.update(matching_params_for_tag_set(catalog, tags))
    return combined

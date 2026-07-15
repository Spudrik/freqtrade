from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from .entry_sieve_lock import _atomic_write_json
from .sieve3_exit_autobatch_manager import PATTERN_TOKENS


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_STRATEGY_DIR = REPO_ROOT / "user_data" / "strategies"
DEFAULT_RUNTIME_ROOT = REPO_ROOT / "user_data" / "Custom_Launcher" / "launcher_v2" / "runtime"
DEFAULT_LEDGER_PATH = (
    DEFAULT_RUNTIME_ROOT
    / "entry_sieve_v2"
    / "migration"
    / "sieve3_v2_migration_ledger.json"
)
DEFAULT_BASELINE_LOOKUP = (
    REPO_ROOT / "ai_guidance_docs" / "04_results" / "sieve3_sieve2_baseline_lookup.md"
)

PILOT_PREFIX = "sieve3_rework_exit_"
V2_PREFIX = "sieve3_V2_"
ENTRY_METHODS = {
    "informative_pairs",
    "populate_buy_trend",
    "populate_entry_trend",
}
BASELINE_ENTRY_METHODS = ENTRY_METHODS | {"populate_indicators"}
ENTRY_SIGNATURE_METADATA_KEYS = ("SIDE", "TIMEFRAME", "ENTRY_MODE")
RESULT_SUMMARY_KEYS = (
    "job_id",
    "finished_at",
    "training_window",
    "training_timerange",
    "validation_window",
    "validation_timerange",
    "random_state",
    "sampling_seed",
    "speed_pair_count",
    "profit_total",
    "profit_total_pct",
    "trade_count",
    "winrate",
    "profit_factor",
    "max_drawdown_pct",
    "params_file",
    "backtest_file",
)

# Only these historical promotion names were verified to map by removing the
# ``complete_pattern_`` prefix.  Unknown names must remain unresolved rather
# than inheriting a broad naming convention by accident.
VALIDATED_COMPLETE_PATTERN_ALIASES = frozenset(
    {
        "complete_pattern_avwap_reject_short",
        "complete_pattern_continuation_flag_present_long_1h",
        "complete_pattern_geometry_ascending_channel_lower_bounce_long_4h",
        "complete_pattern_geometry_ascending_channel_lower_bounce_long_8h",
        "complete_pattern_geometry_descending_channel_lower_breakdown_short_1h",
        "complete_pattern_geometry_descending_channel_lower_breakdown_short_4h",
        "complete_pattern_geometry_rectangle_breakdown_short_1h",
        "complete_pattern_geometry_triangle_squeeze_breakdown_short_4h",
        "complete_pattern_geometry_triangle_squeeze_breakout_long_1h",
        "complete_pattern_geometry_wedge_breakdown_short_4h",
        "complete_pattern_geometry_wedge_breakout_long_1h",
        "complete_pattern_geometry_wedge_breakout_long_8h",
        "complete_pattern_mtf_h4_supply_reject_short_1h_local_break",
        "complete_pattern_multi2_tlv2_boschoch_fall_res_ride_bos_bear_short_1h",
        "complete_pattern_multi2_tlv2_boschoch_res_break_bos_bull_long_1h",
        "complete_pattern_multi2_tlv2_boschoch_res_break_bos_bull_long_8h",
        "complete_pattern_multi2_tlv2_boschoch_res_prox_reject_choch_bear_short_4h",
        "complete_pattern_multi2_tlv2_boschoch_sup_break_bos_bear_short_1h",
        "complete_pattern_multi2_tlv2_vp_res_break_vp_node_long_4h",
        "complete_pattern_multi2_vp_prior_equal_highs_reject_vp_vah_short",
        "complete_pattern_reversal_double_bottom_confirmed_long_4h",
        "complete_pattern_reversal_double_bottom_present_long_4h",
    }
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _relative_path(path: Path, root: Path = REPO_ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _strategy_leaf(value: Any) -> str:
    normalized = str(value or "").strip().replace("\\", "/")
    return normalized.rsplit("/", 1)[-1]


def _path_key(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").casefold()


def _resolve_local_module(module: str) -> Path | None:
    normalized = str(module or "").strip().lstrip(".")
    if not normalized:
        return None
    candidates: list[Path] = []
    if normalized == "user_data" or normalized.startswith("user_data."):
        relative = Path(*normalized.split("."))
        candidates.extend((REPO_ROOT / relative.with_suffix(".py"), REPO_ROOT / relative / "__init__.py"))
    elif "." not in normalized:
        for root in (
            REPO_ROOT / "user_data" / "Indicators",
            REPO_ROOT / "user_data" / "strategies",
            REPO_ROOT / "user_data" / "Custom_Launcher",
        ):
            candidates.extend((root / f"{normalized}.py", root / normalized / "__init__.py"))
    return next((path for path in candidates if path.is_file()), None)


def _imported_modules(tree: ast.AST) -> list[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return sorted(modules)


def _local_dependency_closure(source_bytes: bytes) -> dict[str, Any]:
    try:
        tree = ast.parse(source_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, SyntaxError):
        return {
            "status": "source_parse_error",
            "fingerprint_sha256": "",
            "dependencies": [],
            "unresolved_local_modules": [],
        }

    pending = [(module, 1) for module in _imported_modules(tree)]
    seen_paths: set[str] = set()
    unresolved_local_modules: set[str] = set()
    dependencies: list[dict[str, Any]] = []
    while pending:
        module, depth = pending.pop(0)
        path = _resolve_local_module(module)
        if path is None:
            if module == "user_data" or module.startswith("user_data."):
                unresolved_local_modules.add(module)
            continue
        key = _path_key(path)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        payload = path.read_bytes()
        dependencies.append(
            {
                "module": module,
                "path": _relative_path(path),
                "current_sha256": _sha256_bytes(payload),
                "depth": depth,
                "historical_version_status": "not_archived_with_backtest",
            }
        )
        try:
            dependency_tree = ast.parse(payload.decode("utf-8-sig"), filename=str(path))
        except (UnicodeDecodeError, SyntaxError):
            continue
        pending.extend((child, depth + 1) for child in _imported_modules(dependency_tree))

    dependencies.sort(key=lambda item: (item["path"], item["module"], item["depth"]))
    unresolved = sorted(unresolved_local_modules)
    fingerprint = _sha256_bytes(
        json.dumps(
            {"dependencies": dependencies, "unresolved_local_modules": unresolved},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return {
        "status": (
            "current_dependencies_missing_historical_versions_unavailable"
            if unresolved
            else "current_dependencies_recorded_historical_versions_unavailable"
        ),
        "fingerprint_sha256": fingerprint,
        "dependencies": dependencies,
        "unresolved_local_modules": unresolved,
    }


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _parameter_call(node: ast.AST) -> ast.Call | None:
    """Return the concrete Freqtrade parameter call, including inside wrappers."""
    if not isinstance(node, ast.Call):
        return None
    if _call_name(node.func).endswith("Parameter"):
        return node
    for child in [*node.args, *(keyword.value for keyword in node.keywords)]:
        nested = _parameter_call(child)
        if nested is not None:
            return nested
    return None


def _literal_assignments(nodes: Iterable[ast.stmt]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for node in nodes:
        name = ""
        value_node: ast.AST | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            value_node = node.value
        if not name or value_node is None:
            continue
        try:
            values[name] = ast.literal_eval(value_node)
        except (ValueError, TypeError, SyntaxError):
            continue
    return values


def _literal_assignment_values(nodes: Iterable[ast.stmt]) -> dict[str, list[Any]]:
    values: dict[str, list[Any]] = defaultdict(list)
    for node in nodes:
        name = ""
        value_node: ast.AST | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            value_node = node.value
        if not name or value_node is None:
            continue
        try:
            value = ast.literal_eval(value_node)
        except (ValueError, TypeError, SyntaxError):
            continue
        if value not in values[name]:
            values[name].append(value)
    return dict(values)


def _literal_node(node: ast.AST | None, constants: dict[str, Any]) -> Any:
    if node is None:
        raise ValueError("missing literal")
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    return ast.literal_eval(node)


def _buy_parameter_defaults(
    tree: ast.Module,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    defaults: dict[str, Any] = {}
    conflicts: set[str] = set()
    unresolved: set[str] = set()
    for class_node in (node for node in tree.body if isinstance(node, ast.ClassDef)):
        class_constants = {**metadata, **_literal_assignments(class_node.body)}
        for node in class_node.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            target = (
                node.targets[0]
                if isinstance(node, ast.Assign) and len(node.targets) == 1
                else getattr(node, "target", None)
            )
            if not isinstance(target, ast.Name):
                continue
            parameter = _parameter_call(node.value)
            if parameter is None:
                continue
            space_node = next(
                (keyword.value for keyword in parameter.keywords if keyword.arg == "space"),
                None,
            )
            try:
                space = _literal_node(space_node, class_constants) if space_node is not None else ""
            except (ValueError, TypeError, SyntaxError):
                space = ""
            if space != "buy" and not (not space and target.id.startswith("buy_")):
                continue
            default_node = next(
                (keyword.value for keyword in parameter.keywords if keyword.arg == "default"),
                None,
            )
            try:
                default = _literal_node(default_node, class_constants)
            except (ValueError, TypeError, SyntaxError):
                unresolved.add(target.id)
                continue
            if target.id in defaults and defaults[target.id] != default:
                conflicts.add(target.id)
                continue
            defaults[target.id] = default
    return defaults, sorted(conflicts), sorted(unresolved)


def _buy_parameter_domains(
    tree: ast.Module,
    metadata: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    domains: dict[str, dict[str, Any]] = {}
    unresolved: set[str] = set()
    for class_node in (node for node in tree.body if isinstance(node, ast.ClassDef)):
        class_constants = {**metadata, **_literal_assignments(class_node.body)}
        for node in class_node.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            target = (
                node.targets[0]
                if isinstance(node, ast.Assign) and len(node.targets) == 1
                else getattr(node, "target", None)
            )
            if not isinstance(target, ast.Name):
                continue
            parameter = _parameter_call(node.value)
            if parameter is None:
                continue
            space_node = next(
                (keyword.value for keyword in parameter.keywords if keyword.arg == "space"),
                None,
            )
            try:
                space = _literal_node(space_node, class_constants) if space_node is not None else ""
            except (ValueError, TypeError, SyntaxError):
                space = ""
            if space != "buy" and not (not space and target.id.startswith("buy_")):
                continue

            parameter_type = _call_name(parameter.func)
            try:
                if parameter_type == "BooleanParameter":
                    domain = {"type": "boolean", "choices": [False, True]}
                elif parameter_type == "CategoricalParameter":
                    choices = _literal_node(parameter.args[0], class_constants)
                    if not isinstance(choices, (list, tuple, set, frozenset)):
                        raise ValueError("categorical choices are not a literal sequence")
                    domain = {"type": "categorical", "choices": list(choices)}
                elif parameter_type in {
                    "IntParameter",
                    "DecimalParameter",
                    "RealParameter",
                }:
                    low = _literal_node(parameter.args[0], class_constants)
                    high = _literal_node(parameter.args[1], class_constants)
                    domain = {
                        "type": (
                            "integer" if parameter_type == "IntParameter" else "numeric"
                        ),
                        "low": low,
                        "high": high,
                    }
                else:
                    unresolved.add(target.id)
                    continue
            except (IndexError, ValueError, TypeError, SyntaxError):
                unresolved.add(target.id)
                continue
            if target.id in domains and domains[target.id] != domain:
                unresolved.add(target.id)
                domains.pop(target.id, None)
                continue
            domains[target.id] = domain
    return domains, sorted(unresolved)


def _parameter_domain_violations(
    values: dict[str, Any],
    domains: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for name, value in sorted(values.items()):
        domain = domains.get(name)
        if domain is None:
            # Parameters inherited from a base class or imported factory are
            # not necessarily declared in the archived leaf module.  Record
            # them separately as unavailable domains; only a known domain can
            # prove an out-of-range promotion value.
            continue
        domain_type = domain.get("type")
        valid = True
        if domain_type in {"boolean", "categorical"}:
            valid = value in domain.get("choices", [])
        elif domain_type in {"integer", "numeric"}:
            valid = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and domain.get("low") <= value <= domain.get("high")
            )
            if valid and domain_type == "integer":
                valid = isinstance(value, int) and not isinstance(value, bool)
        if not valid:
            violations.append(
                {
                    "parameter": name,
                    "value": value,
                    "reason": "value_outside_archived_parameter_domain",
                    "domain": domain,
                }
            )
    return violations


def _entry_signature(
    tree: ast.Module,
    metadata: dict[str, Any],
    entry_methods: set[str] | None = None,
) -> str:
    method_names = entry_methods or ENTRY_METHODS
    fragments: list[str] = []
    for key in ENTRY_SIGNATURE_METADATA_KEYS:
        value = metadata.get(key)
        if value not in (None, ""):
            fragments.append(f"{key}={value}")
    if metadata.get("TIMEFRAME") in (None, "") and metadata.get("timeframe") not in (None, ""):
        fragments.append(f"TIMEFRAME={metadata['timeframe']}")

    module_functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for class_node in (node for node in tree.body if isinstance(node, ast.ClassDef)):
        class_methods = {
            node.name: node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        pending: list[tuple[str, str]] = [
            ("class", name) for name in sorted(method_names & class_methods.keys())
        ]
        seen: set[tuple[str, str]] = set()
        while pending:
            kind, name = pending.pop()
            key = kind, name
            if key in seen:
                continue
            seen.add(key)
            method = class_methods.get(name) if kind == "class" else module_functions.get(name)
            if method is None:
                continue
            # Generated branch class names differ even when the locked entry surface is
            # identical.  Keep only the callable role and body in the semantic hash.
            fragments.append(
                f"{kind}:{name}:"
                + ast.dump(method, annotate_fields=True, include_attributes=False)
            )
            for call in (node for node in ast.walk(method) if isinstance(node, ast.Call)):
                if isinstance(call.func, ast.Name) and call.func.id in module_functions:
                    pending.append(("module", call.func.id))
                elif (
                    isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id in {"self", "cls"}
                    and call.func.attr in class_methods
                ):
                    pending.append(("class", call.func.attr))

        for node in class_node.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            target = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else getattr(node, "target", None)
            if not isinstance(target, ast.Name):
                continue
            parameter = _parameter_call(node.value)
            if parameter is None:
                continue
            space = next((keyword.value for keyword in parameter.keywords if keyword.arg == "space"), None)
            try:
                is_buy = ast.literal_eval(space) == "buy" if space is not None else target.id.startswith("buy_")
            except (ValueError, TypeError, SyntaxError):
                is_buy = False
            if is_buy:
                fragments.append(
                    f"buy_parameter:{target.id}:"
                    + ast.dump(parameter, annotate_fields=True, include_attributes=False)
                )

    if not fragments:
        return ""
    return _sha256_bytes("\n".join(sorted(fragments)).encode("utf-8"))


def parse_strategy(path: Path) -> dict[str, Any]:
    payload = path.read_text(encoding="utf-8-sig", errors="replace")
    try:
        tree = ast.parse(payload, filename=str(path))
    except SyntaxError as exc:
        return {
            "metadata": {},
            "metadata_values": {},
            "class_names": [],
            "entry_signature_sha256": "",
            "buy_parameter_defaults": {},
            "buy_parameter_defaults_sha256": "",
            "buy_parameter_default_conflicts": [],
            "unresolved_buy_parameter_defaults": [],
            "parse_error": f"{exc.msg} at line {exc.lineno}",
        }

    metadata = _literal_assignments(tree.body)
    metadata_values = _literal_assignment_values(tree.body)
    class_names: list[str] = []
    for class_node in (node for node in tree.body if isinstance(node, ast.ClassDef)):
        class_names.append(class_node.name)
        for key, value in _literal_assignments(class_node.body).items():
            metadata.setdefault(key, value)
        for key, values in _literal_assignment_values(class_node.body).items():
            destination = metadata_values.setdefault(key, [])
            for value in values:
                if value not in destination:
                    destination.append(value)

    buy_defaults, buy_default_conflicts, unresolved_buy_defaults = _buy_parameter_defaults(
        tree,
        metadata,
    )

    return {
        "metadata": metadata,
        "metadata_values": metadata_values,
        "class_names": class_names,
        "entry_signature_sha256": _entry_signature(tree, metadata),
        "buy_parameter_defaults": buy_defaults,
        "buy_parameter_defaults_sha256": (
            _sha256_bytes(
                json.dumps(buy_defaults, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
            if buy_defaults
            else ""
        ),
        "buy_parameter_default_conflicts": buy_default_conflicts,
        "unresolved_buy_parameter_defaults": unresolved_buy_defaults,
        "parse_error": "",
    }


def canonical_source_slug(value: str) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        return ""
    python_reference = re.match(r"^(.*?\.py)(?::[^/]*)?$", raw, flags=re.IGNORECASE)
    reference = python_reference.group(1) if python_reference else raw
    if not python_reference and ":" in reference and not re.match(r"^[a-z]:/", reference, flags=re.IGNORECASE):
        reference = reference.rsplit(":", 1)[0]
    stem = Path(reference.rsplit("/", 1)[-1]).stem.lower()
    if "_from_" in stem:
        stem = stem.split("_from_", 1)[1]
    for prefix in ("sieve3_exit_", "sieve3_", "sieve2_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    stem = re.sub(r"[^a-z0-9_]+", "_", stem)
    return re.sub(r"_+", "_", stem).strip("_")


def filename_source_slug(path: Path) -> str:
    return canonical_source_slug(path.stem)


def _branch_family(path: Path, metadata: dict[str, Any]) -> str:
    explicit = str(metadata.get("S3_BRANCH_FAMILY") or metadata.get("RESEARCH_PATH") or "").strip()
    if explicit:
        return explicit
    stem = path.stem
    if stem.startswith(PILOT_PREFIX) and "_from_" in stem:
        return stem[len(PILOT_PREFIX) :].split("_from_", 1)[0]
    if "_from_" in stem:
        return stem.split("_from_", 1)[0].removeprefix("sieve3_exit_")
    return "source_foundation"


def _is_pattern(source: str) -> bool:
    lowered = f"_{source.lower()}_"
    return any(token in lowered for token in PATTERN_TOKENS)


def _v2_filename(source: str, pattern: bool) -> str:
    role = "pattern_integrated" if pattern else "integrated"
    return f"sieve3_V2_{role}_from_{source}.py"


def _result_files(runtime_root: Path) -> list[Path]:
    files: list[Path] = []
    for lane in sorted(path for path in runtime_root.glob("entry_sieve*") if path.is_dir()):
        files.extend(sorted(path for path in lane.rglob("*.jsonl") if path.parent.name == "results"))
    return files


def _float_cell(value: str) -> float:
    return float(value.strip().strip("`"))


def parse_baseline_lookup(path: Path) -> dict[str, dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not path.is_file():
        return {}
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if not line.startswith("| `sieve2_"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 10:
            continue
        source_strategy = cells[0].strip("`")
        tp_sl = cells[2].strip("`").split("/", 1)
        if len(tp_sl) != 2:
            continue
        try:
            record = {
                "sieve2_source": source_strategy,
                "canonical_source": canonical_source_slug(source_strategy),
                "s3_files_at_lookup": int(cells[1]),
                "take_profit_pct": _float_cell(tp_sl[0]),
                "stoploss_pct": _float_cell(tp_sl[1]),
                "profit_total_pct": _float_cell(cells[3]),
                "winrate_pct": _float_cell(cells[4]),
                "trade_count": int(cells[5]),
                "max_drawdown_pct": _float_cell(cells[6]),
                "profit_factor": _float_cell(cells[7]),
                "training_window": cells[8].strip("`"),
                "job_id": cells[9].strip("`"),
            }
        except ValueError:
            continue
        if record["canonical_source"]:
            candidates[record["canonical_source"]].append(record)

    selected: dict[str, dict[str, Any]] = {}
    for source, rows in candidates.items():
        rows.sort(
            key=lambda row: (
                0
                if row["take_profit_pct"] == 2.0 and row["stoploss_pct"] == 2.0
                else 1,
                -row["winrate_pct"],
                -row["profit_total_pct"],
                -row["trade_count"],
                row["job_id"],
            )
        )
        selected[source] = {**rows[0], "lookup_candidate_count": len(rows)}
    return selected


def _baseline_metrics_match(row: dict[str, Any], baseline: dict[str, Any]) -> bool:
    try:
        return (
            int(row.get("trade_count")) == baseline["trade_count"]
            and abs(float(row.get("take_profit_pct")) - baseline["take_profit_pct"]) < 0.000001
            and abs(float(row.get("stoploss_pct")) - baseline["stoploss_pct"]) < 0.000001
            and abs((float(row.get("profit_total")) * 100.0) - baseline["profit_total_pct"]) <= 0.011
            and abs((float(row.get("winrate")) * 100.0) - baseline["winrate_pct"]) <= 0.011
            and abs((float(row.get("max_drawdown_pct")) * 100.0) - baseline["max_drawdown_pct"]) <= 0.011
            and abs(float(row.get("profit_factor")) - baseline["profit_factor"]) <= 0.011
        )
    except (TypeError, ValueError):
        return False


def _baseline_result_fingerprint(row: dict[str, Any]) -> str:
    keys = (
        "job_id",
        "finished_at",
        "strategy",
        "strategy_class",
        "strategy_file",
        "training_window",
        "training_timerange",
        "validation_window",
        "validation_timerange",
        "random_state",
        "sampling_seed",
        "take_profit_pct",
        "stoploss_pct",
        "profit_total",
        "profit_total_pct",
        "trade_count",
        "winrate",
        "profit_factor",
        "max_drawdown_pct",
        "params_file",
        "hyperopt_file",
        "backtest_file",
    )
    semantic = {key: row.get(key) for key in keys if row.get(key) not in (None, "")}
    return _sha256_bytes(
        json.dumps(semantic, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    )


def _result_index(runtime_root: Path) -> dict[tuple[str, str], list[tuple[dict[str, Any], Path]]]:
    index: dict[tuple[str, str], list[tuple[dict[str, Any], Path]]] = defaultdict(list)
    for result_path in _result_files(runtime_root):
        with result_path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict) or str(row.get("status") or "").lower() != "ok":
                    continue
                strategy = str(row.get("strategy") or "").strip().lower()
                job_id = str(row.get("job_id") or result_path.stem).strip()
                if strategy.startswith("sieve2_") and job_id:
                    index[(strategy, job_id)].append((row, result_path))
    return index


def _all_sieve2_results(
    runtime_root: Path,
) -> dict[str, list[tuple[dict[str, Any], Path, int, str]]]:
    index: dict[str, list[tuple[dict[str, Any], Path, int, str]]] = defaultdict(list)
    for result_path in _result_files(runtime_root):
        with result_path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict) or str(row.get("status") or "").lower() != "ok":
                    continue
                strategy = str(row.get("strategy") or "").strip().lower()
                if not strategy.startswith("sieve2_"):
                    continue
                source = canonical_source_slug(strategy)
                if source:
                    index[source].append(
                        (
                            row,
                            result_path,
                            line_number,
                            _sha256_bytes(line.encode("utf-8")),
                        )
                    )
    return index


def _source_contexts(
    source_records: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    for source, members in source_records.items():
        nonpilot_members = [row for row in members if row["category"] != "refined_pilot"]
        explicit_members = [
            row
            for row in nonpilot_members
            if row.get("source_resolution") == "source_entry_stem"
        ]
        # Explicit SOURCE_ENTRY_STEM branches are the maintained descendants.
        # Older filename-only predecessors can have stale defaults and must not
        # outvote them when reconstructing the promoted entry lock.
        foundation_members = explicit_members or nonpilot_members
        excluded_predecessors = [
            row for row in nonpilot_members if row not in foundation_members
        ]
        aliases = {source}
        if source in VALIDATED_COMPLETE_PATTERN_ALIASES:
            aliases.add(source.removeprefix("complete_pattern_"))
        for row in foundation_members:
            for value in row.get("source_sieve2_strategy_values", []):
                direct = canonical_source_slug(value)
                if direct:
                    aliases.add(direct)

        default_counts = Counter(
            row["buy_parameter_defaults_sha256"]
            for row in foundation_members
            if row["buy_parameter_defaults_sha256"]
        )
        default_maps = {
            row["buy_parameter_defaults_sha256"]: row["buy_parameter_defaults"]
            for row in foundation_members
            if row["buy_parameter_defaults_sha256"]
        }
        signature_counts = Counter(
            row["entry_signature_sha256"]
            for row in foundation_members
            if row["entry_signature_sha256"]
        )
        selected_default_hash = ""
        selected_signature_hash = ""
        flags: list[str] = []
        if default_counts:
            top_count = max(default_counts.values())
            leaders = sorted(key for key, count in default_counts.items() if count == top_count)
            if len(leaders) == 1:
                selected_default_hash = leaders[0]
            else:
                flags.append("current_buy_default_consensus_tied")
        else:
            flags.append("current_buy_defaults_unavailable")
        if len(default_counts) > 1:
            flags.append("multiple_current_buy_default_maps")

        if signature_counts:
            top_count = max(signature_counts.values())
            leaders = sorted(key for key, count in signature_counts.items() if count == top_count)
            if len(leaders) == 1:
                selected_signature_hash = leaders[0]
            else:
                flags.append("current_entry_signature_consensus_tied")
        else:
            flags.append("current_entry_signature_unavailable")
        if len(signature_counts) > 1:
            flags.append("multiple_current_entry_signatures")

        contexts[source] = {
            "canonical_source": source,
            "sieve2_source_aliases": sorted(aliases),
            "current_buy_default_candidates": [
                {
                    "sha256": key,
                    "file_count": count,
                    "defaults": default_maps[key],
                }
                for key, count in sorted(
                    default_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
            "selected_buy_defaults_sha256": selected_default_hash,
            "selected_buy_defaults": default_maps.get(selected_default_hash, {}),
            "current_entry_signature_candidates": [
                {"sha256": key, "file_count": count}
                for key, count in sorted(
                    signature_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
            "selected_entry_signature_sha256": selected_signature_hash,
            "foundation_member_policy": (
                "explicit_source_entry_stem_members"
                if explicit_members
                else "all_nonpilot_members"
            ),
            "foundation_member_count": len(foundation_members),
            "excluded_filename_predecessors": sorted(
                str(row.get("legacy_file") or "") for row in excluded_predecessors
            ),
            "has_explicit_sieve2_source_metadata": any(
                row.get("source_sieve2_strategy_values") for row in foundation_members
            ),
            "source_result_batches": sorted(
                {
                    str(batch)
                    for row in foundation_members
                    for batch in row.get("source_result_batches", [])
                    if batch
                }
            ),
            "source_result_metadata_pairs": sorted(
                [
                    {
                        "sieve2_source": canonical_source_slug(
                            row.get("source_sieve2_strategy_raw") or ""
                        ),
                        "job_id": str(row.get("source_result_batch") or ""),
                        "declaring_file": str(row.get("legacy_file") or ""),
                    }
                    for row in foundation_members
                    if row.get("source_sieve2_strategy_raw")
                    and row.get("source_result_batch")
                ],
                key=lambda item: (
                    item["sieve2_source"],
                    item["job_id"],
                    item["declaring_file"],
                ),
            ),
            "ambiguity_flags": sorted(set(flags)),
        }
    return contexts


def _params_match_current_defaults(params: dict[str, Any], defaults: dict[str, Any]) -> bool:
    return bool(params) and bool(defaults) and params == defaults


def _candidate_rank(
    row: dict[str, Any],
    *,
    entry_lock_match: bool,
    lookup_match: bool,
    metadata_job_match: bool,
) -> tuple[Any, ...]:
    # Performance must never choose lineage.  It is evidence about a selected
    # promotion, not a key for deciding which source/parameter identity it was.
    return (
        int(entry_lock_match),
        int(metadata_job_match),
        int(lookup_match),
        str(row.get("finished_at") or ""),
        str(row.get("job_id") or ""),
    )


def resolve_context_baseline_lineage(
    runtime_root: Path,
    baseline_lookup: Path,
    source_contexts: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    hints = parse_baseline_lookup(baseline_lookup)
    result_index = _all_sieve2_results(runtime_root)
    resolved: dict[str, dict[str, Any]] = {}
    stats = {
        "source_groups": len(source_contexts),
        "lookup_hints": len(hints),
        "verified_param_snapshots": 0,
        "missing_result_rows": 0,
        "ambiguous_result_rows": 0,
        "unavailable_param_snapshots": 0,
        "duplicate_result_row_copies": 0,
        "params_and_signature_matches": 0,
        "params_only_matches": 0,
        "explicit_source_result_metadata_matches": 0,
        "lookup_only_matches": 0,
        "unique_snapshot_lineages": 0,
        "effective_lock_matches": 0,
        "promotion_param_drift": 0,
        "domain_violations": 0,
        "statically_clean": 0,
    }

    for source, context in sorted(source_contexts.items()):
        hint = hints.get(source)
        aliases = set(context["sieve2_source_aliases"])
        if hint:
            aliases.add(canonical_source_slug(hint["sieve2_source"]))
        raw_candidates = [item for alias in aliases for item in result_index.get(alias, [])]
        unique_candidates: dict[str, tuple[dict[str, Any], Path, int, str]] = {}
        for item in raw_candidates:
            unique_candidates.setdefault(_baseline_result_fingerprint(item[0]), item)
        stats["duplicate_result_row_copies"] += len(raw_candidates) - len(unique_candidates)
        metadata_pairs = {
            (item["sieve2_source"], item["job_id"])
            for item in context["source_result_metadata_pairs"]
        }

        candidate_rows: list[dict[str, Any]] = []
        for row, result_path, line_number, row_hash in unique_candidates.values():
            snapshot = _baseline_snapshot(row)
            params_match = _params_match_current_defaults(
                snapshot.get("effective_buy_lock", {}),
                context["selected_buy_defaults"],
            )
            signature_match = bool(
                snapshot.get("baseline_entry_trend_signature_sha256")
                and snapshot.get("baseline_entry_trend_signature_sha256")
                == context["selected_entry_signature_sha256"]
            )
            lookup_match = bool(
                hint
                and str(row.get("strategy") or "").lower() == hint["sieve2_source"].lower()
                and str(row.get("job_id") or "") == hint["job_id"]
                and _baseline_metrics_match(row, hint)
            )
            metadata_job_match = (
                canonical_source_slug(row.get("strategy") or ""),
                str(row.get("job_id") or ""),
            ) in metadata_pairs
            candidate_rows.append(
                {
                    "row": row,
                    "result_path": result_path,
                    "line_number": line_number,
                    "row_sha256": row_hash,
                    "snapshot": snapshot,
                    "params_match": params_match,
                    "signature_match": signature_match,
                    "lookup_match": lookup_match,
                    "metadata_job_match": metadata_job_match,
                }
            )

        exact_lock = [
            item
            for item in candidate_rows
            if item["snapshot"]["snapshot_status"] == "verified"
            and item["params_match"]
        ]
        lookup_only = [
            item
            for item in candidate_rows
            if item["lookup_match"]
            and item["snapshot"]["snapshot_status"] == "verified"
        ]
        explicit_metadata = [
            item
            for item in candidate_rows
            if context["has_explicit_sieve2_source_metadata"]
            and item["metadata_job_match"]
            and item["snapshot"]["snapshot_status"] == "verified"
        ]
        if exact_lock:
            pool = exact_lock
            selected_by = "exact_current_params_and_entry_signature"
            if any(item["signature_match"] for item in pool):
                stats["params_and_signature_matches"] += 1
            else:
                selected_by = "exact_current_effective_buy_lock"
                stats["params_only_matches"] += 1
        elif explicit_metadata:
            pool = explicit_metadata
            selected_by = "explicit_sieve2_source_and_result_metadata"
            stats["explicit_source_result_metadata_matches"] += 1
        elif lookup_only:
            pool = lookup_only
            selected_by = "lookup_hint_only"
            stats["lookup_only_matches"] += 1
        else:
            pool = []
            selected_by = ""

        flags = list(context["ambiguity_flags"])
        if not pool:
            stats["missing_result_rows"] += 1
            flags.append("baseline_result_not_found_for_current_entry_surface")
            resolved[source] = {
                "canonical_source": source,
                "status": "baseline_result_not_found",
                "selected_by": selected_by,
                "baseline_lookup_hint": hint or {},
                "source_context": context,
                "candidate_count": len(candidate_rows),
                "ambiguity_flags": sorted(set(flags)),
            }
            continue

        distinct_snapshot_identities = {
            (
                item["snapshot"]["snapshot_strategy_sha256"],
                item["snapshot"]["effective_buy_lock_sha256"],
            )
            for item in pool
            if item["snapshot"]["snapshot_strategy_sha256"]
            and item["snapshot"]["effective_buy_lock_sha256"]
        }
        if len(distinct_snapshot_identities) > 1:
            stats["ambiguous_result_rows"] += 1
            flags.append("multiple_distinct_eligible_sieve2_source_lock_identities")
        pool.sort(
            key=lambda item: _candidate_rank(
                item["row"],
                entry_lock_match=item["params_match"],
                lookup_match=item["lookup_match"],
                metadata_job_match=item["metadata_job_match"],
            ),
            reverse=True,
        )
        selected = pool[0]
        row = selected["row"]
        snapshot = selected["snapshot"]
        if snapshot["snapshot_status"] != "verified":
            stats["unavailable_param_snapshots"] += 1
            flags.append(f"baseline_{snapshot['snapshot_status']}")
        if selected_by == "lookup_hint_only":
            flags.append("lookup_hint_not_matched_to_current_entry_surface")

        candidate_evidence = []
        for item in sorted(
            candidate_rows,
            key=lambda candidate: str(candidate["row"].get("finished_at") or ""),
        ):
            candidate_row = item["row"]
            candidate_snapshot = item["snapshot"]
            candidate_evidence.append(
                {
                    "strategy": str(candidate_row.get("strategy") or ""),
                    "job_id": str(candidate_row.get("job_id") or ""),
                    "finished_at": str(candidate_row.get("finished_at") or ""),
                    "result_file": _relative_path(item["result_path"]),
                    "line_number": item["line_number"],
                    "row_sha256": item["row_sha256"],
                    "training_window": str(candidate_row.get("training_window") or ""),
                    "take_profit_pct": candidate_row.get("take_profit_pct"),
                    "stoploss_pct": candidate_row.get("stoploss_pct"),
                    "profit_total": candidate_row.get("profit_total"),
                    "trade_count": candidate_row.get("trade_count"),
                    "winrate": candidate_row.get("winrate"),
                    "max_drawdown_pct": candidate_row.get("max_drawdown_pct"),
                    "profit_factor": candidate_row.get("profit_factor"),
                    "snapshot_status": candidate_snapshot["snapshot_status"],
                    "backtest_file": candidate_snapshot["backtest_file"],
                    "backtest_archive_sha256": candidate_snapshot[
                        "backtest_archive_sha256"
                    ],
                    "params_member": candidate_snapshot["snapshot_member"],
                    "params_sha256": candidate_snapshot["snapshot_sha256"],
                    "strategy_member": candidate_snapshot["snapshot_strategy_member"],
                    "archived_source_buy_defaults_sha256": candidate_snapshot[
                        "archived_source_buy_defaults_sha256"
                    ],
                    "selected_buy_params_sha256": candidate_snapshot[
                        "selected_buy_params_sha256"
                    ],
                    "effective_buy_lock_sha256": candidate_snapshot[
                        "effective_buy_lock_sha256"
                    ],
                    "locked_buy_params_sha256": candidate_snapshot["locked_buy_params_sha256"],
                    "snapshot_strategy_sha256": candidate_snapshot["snapshot_strategy_sha256"],
                    "baseline_entry_signature_sha256": candidate_snapshot[
                        "baseline_entry_signature_sha256"
                    ],
                    "baseline_entry_trend_signature_sha256": candidate_snapshot[
                        "baseline_entry_trend_signature_sha256"
                    ],
                    "params_match": item["params_match"],
                    "entry_signature_match": item["signature_match"],
                    "lookup_hint_match": item["lookup_match"],
                    "metadata_job_match": item["metadata_job_match"],
                }
            )

        domain_violations = snapshot.get("effective_buy_lock_domain_violations", [])
        unique_snapshot = (
            snapshot["snapshot_status"] == "verified"
            and len(distinct_snapshot_identities) == 1
        )
        if unique_snapshot:
            stats["unique_snapshot_lineages"] += 1
        if unique_snapshot and selected["params_match"]:
            stats["effective_lock_matches"] += 1
        if unique_snapshot and not selected["params_match"]:
            selected_status = "promotion_params_not_locked"
            stats["promotion_param_drift"] += 1
            flags.append("current_entry_defaults_do_not_match_promoted_effective_buy_lock")
        elif unique_snapshot and domain_violations:
            selected_status = "promoted_params_outside_declared_domain"
            stats["domain_violations"] += 1
            flags.append("promoted_effective_buy_lock_outside_archived_parameter_domain")
        elif unique_snapshot and selected["params_match"]:
            selected_status = "verified"
            stats["statically_clean"] += 1
        else:
            selected_status = "ambiguous"
        if selected_status == "verified":
            stats["verified_param_snapshots"] += 1
        resolved[source] = {
            "canonical_source": source,
            "sieve2_source": str(row.get("strategy") or ""),
            "status": selected_status,
            "selected_by": selected_by,
            "job_id": str(row.get("job_id") or ""),
            "finished_at": str(row.get("finished_at") or ""),
            "result_file": _relative_path(selected["result_path"]),
            "result_line_number": selected["line_number"],
            "result_row_sha256": selected["row_sha256"],
            "strategy_class": str(row.get("strategy_class") or ""),
            "training_window": str(row.get("training_window") or ""),
            "take_profit_pct": float(row.get("take_profit_pct") or 0.0),
            "stoploss_pct": float(row.get("stoploss_pct") or 0.0),
            "profit_total_pct": float(row.get("profit_total") or 0.0) * 100.0,
            "winrate_pct": float(row.get("winrate") or 0.0) * 100.0,
            "trade_count": int(row.get("trade_count") or 0),
            "max_drawdown_pct": float(row.get("max_drawdown_pct") or 0.0) * 100.0,
            "profit_factor": float(row.get("profit_factor") or 0.0),
            "result_params_file": str(row.get("params_file") or ""),
            "result_hyperopt_file": str(row.get("hyperopt_file") or ""),
            "external_result_paths_authoritative": False,
            "baseline_lookup_hint": hint or {},
            "source_context": context,
            "candidate_count": len(candidate_rows),
            "snapshot_lineage_status": "verified" if unique_snapshot else "ambiguous",
            "entry_lock_matches_current_defaults": selected["params_match"],
            "candidate_evidence": candidate_evidence,
            "ambiguity_flags": sorted(set(flags)),
            **snapshot,
        }
    return resolved, stats


def _baseline_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    output = {
        "snapshot_status": "not_available",
        "snapshot_member": "",
        "snapshot_sha256": "",
        "snapshot_strategy_member": "",
        "snapshot_strategy_sha256": "",
        "baseline_entry_signature_sha256": "",
        "baseline_entry_trend_signature_sha256": "",
        "backtest_file": "",
        "backtest_archive_sha256": "",
        "archived_source_buy_defaults": {},
        "archived_source_buy_defaults_sha256": "",
        "selected_buy_params": {},
        "selected_buy_params_sha256": "",
        "effective_buy_lock": {},
        "effective_buy_lock_sha256": "",
        "locked_buy_params": {},
        "locked_buy_params_sha256": "",
        "buy_parameter_domains": {},
        "unresolved_buy_parameter_domains": [],
        "effective_buy_lock_domain_unavailable_parameters": [],
        "effective_buy_lock_domain_violations": [],
        "imported_local_dependencies_status": "not_available",
        "imported_local_dependencies_fingerprint_sha256": "",
        "imported_local_dependencies": [],
        "unresolved_local_dependency_modules": [],
    }
    archive_path = _resolve_result_path(row.get("backtest_file"))
    if archive_path is None:
        return output
    output["backtest_file"] = _relative_path(archive_path)
    if not archive_path.is_file():
        output["snapshot_status"] = "archive_missing"
        return output
    strategy_class = str(row.get("strategy_class") or "").strip()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            json_candidates = [
                name
                for name in archive.namelist()
                if strategy_class and name.lower().endswith(f"_{strategy_class.lower()}.json")
            ]
            python_candidates = [
                name
                for name in archive.namelist()
                if strategy_class and name.lower().endswith(f"_{strategy_class.lower()}.py")
            ]
            if len(json_candidates) != 1 or len(python_candidates) != 1:
                output["snapshot_status"] = "snapshot_ambiguous"
                return output
            params_member = json_candidates[0]
            strategy_member = python_candidates[0]
            params_bytes = archive.read(params_member)
            strategy_bytes = archive.read(strategy_member)
            payload = json.loads(params_bytes.decode("utf-8-sig"))
    except (OSError, zipfile.BadZipFile, KeyError, UnicodeDecodeError, json.JSONDecodeError):
        output["snapshot_status"] = "archive_error"
        return output
    params = payload.get("params") if isinstance(payload, dict) else None
    buy = params.get("buy") if isinstance(params, dict) else None
    if str(payload.get("strategy_name") or "") != strategy_class or not isinstance(buy, dict):
        output["snapshot_status"] = "params_invalid"
        return output
    try:
        source_tree = ast.parse(strategy_bytes.decode("utf-8-sig"), filename=strategy_member)
    except (UnicodeDecodeError, SyntaxError):
        output["snapshot_status"] = "strategy_source_invalid"
        return output
    source_metadata = _literal_assignments(source_tree.body)
    for class_node in (node for node in source_tree.body if isinstance(node, ast.ClassDef)):
        for key, value in _literal_assignments(class_node.body).items():
            source_metadata.setdefault(key, value)
    source_buy_defaults, source_default_conflicts, unresolved_source_defaults = (
        _buy_parameter_defaults(source_tree, source_metadata)
    )
    if source_default_conflicts or unresolved_source_defaults:
        output["snapshot_status"] = "source_buy_defaults_ambiguous"
        output["source_buy_default_conflicts"] = source_default_conflicts
        output["unresolved_source_buy_defaults"] = unresolved_source_defaults
        return output
    domains, unresolved_domains = _buy_parameter_domains(source_tree, source_metadata)
    effective_buy_lock = {**source_buy_defaults, **buy}
    source_defaults_payload = json.dumps(
        source_buy_defaults, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    selected_params_payload = json.dumps(
        buy, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    effective_lock_payload = json.dumps(
        effective_buy_lock, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    domain_violations = _parameter_domain_violations(effective_buy_lock, domains)
    unavailable_domain_parameters = sorted(set(effective_buy_lock) - set(domains))
    dependencies = _local_dependency_closure(strategy_bytes)
    output.update(
        {
            "snapshot_status": "verified",
            "backtest_archive_sha256": _sha256_file(archive_path),
            "snapshot_member": params_member,
            "snapshot_sha256": _sha256_bytes(params_bytes),
            "snapshot_strategy_member": strategy_member,
            "snapshot_strategy_sha256": _sha256_bytes(strategy_bytes),
            "baseline_entry_signature_sha256": _entry_signature(
                source_tree,
                source_metadata,
                BASELINE_ENTRY_METHODS,
            ),
            "baseline_entry_trend_signature_sha256": _entry_signature(
                source_tree,
                source_metadata,
                ENTRY_METHODS,
            ),
            "archived_source_buy_defaults": source_buy_defaults,
            "archived_source_buy_defaults_sha256": _sha256_bytes(
                source_defaults_payload
            ),
            "selected_buy_params": buy,
            "selected_buy_params_sha256": _sha256_bytes(selected_params_payload),
            "effective_buy_lock": effective_buy_lock,
            "effective_buy_lock_sha256": _sha256_bytes(effective_lock_payload),
            # Compatibility name retained for existing consumers.  The lock is
            # the full effective map, not merely the sparse JSON override.
            "locked_buy_params": effective_buy_lock,
            "locked_buy_params_sha256": _sha256_bytes(effective_lock_payload),
            "buy_parameter_domains": domains,
            "unresolved_buy_parameter_domains": unresolved_domains,
            "effective_buy_lock_domain_unavailable_parameters": (
                unavailable_domain_parameters
            ),
            "effective_buy_lock_domain_violations": domain_violations,
            "imported_local_dependencies_status": dependencies["status"],
            "imported_local_dependencies_fingerprint_sha256": dependencies[
                "fingerprint_sha256"
            ],
            "imported_local_dependencies": dependencies["dependencies"],
            "unresolved_local_dependency_modules": dependencies[
                "unresolved_local_modules"
            ],
        }
    )
    return output


def resolve_baseline_lineage(
    runtime_root: Path,
    baseline_lookup: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    baselines = parse_baseline_lookup(baseline_lookup)
    result_index = _result_index(runtime_root)
    resolved: dict[str, dict[str, Any]] = {}
    stats = {
        "lookup_sources": len(baselines),
        "matched_result_rows": 0,
        "verified_param_snapshots": 0,
        "missing_result_rows": 0,
        "ambiguous_result_rows": 0,
        "duplicate_result_row_copies": 0,
        "unavailable_param_snapshots": 0,
    }
    for source, baseline in baselines.items():
        key = (baseline["sieve2_source"].lower(), baseline["job_id"])
        matches = [
            (row, result_path)
            for row, result_path in result_index.get(key, [])
            if _baseline_metrics_match(row, baseline)
        ]
        unique_matches: dict[str, tuple[dict[str, Any], Path]] = {}
        for row, result_path in matches:
            unique_matches.setdefault(_baseline_result_fingerprint(row), (row, result_path))
        stats["duplicate_result_row_copies"] += len(matches) - len(unique_matches)
        matches = list(unique_matches.values())
        flags: list[str] = []
        if not matches:
            stats["missing_result_rows"] += 1
            resolved[source] = {
                **baseline,
                "status": "baseline_result_not_found",
                "ambiguity_flags": ["baseline_result_not_found"],
            }
            continue
        matches.sort(key=lambda item: str(item[0].get("finished_at") or ""), reverse=True)
        if len(matches) > 1:
            stats["ambiguous_result_rows"] += 1
            flags.append("duplicate_matching_baseline_rows")
        row, result_path = matches[0]
        stats["matched_result_rows"] += 1
        snapshot = _baseline_snapshot(row)
        if snapshot["snapshot_status"] == "verified":
            stats["verified_param_snapshots"] += 1
        else:
            stats["unavailable_param_snapshots"] += 1
            flags.append(f"baseline_{snapshot['snapshot_status']}")
        resolved[source] = {
            **baseline,
            "status": "verified" if snapshot["snapshot_status"] == "verified" else "snapshot_unavailable",
            "result_file": _relative_path(result_path),
            "finished_at": str(row.get("finished_at") or ""),
            "strategy_class": str(row.get("strategy_class") or ""),
            "result_params_file": str(row.get("params_file") or ""),
            "result_hyperopt_file": str(row.get("hyperopt_file") or ""),
            "ambiguity_flags": flags,
            **snapshot,
        }
    return resolved, stats


def scan_successful_results(
    runtime_root: Path,
    strategy_paths: Iterable[Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    current_paths = list(strategy_paths)
    current_by_path = {_path_key(path): path for path in current_paths}
    current_by_leaf: dict[str, list[Path]] = defaultdict(list)
    for path in current_paths:
        current_by_leaf[path.name.casefold()].append(path)
    evidence: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "successful_row_count": 0,
            "exact_successful_row_count": 0,
            "alias_successful_row_count": 0,
            "successful_result_ids": set(),
            "successful_result_files": set(),
            "historical_strategy_paths": set(),
            "path_resolution_flags": set(),
            "latest_row": None,
            "successful_rows": [],
            "successful_evidence": [],
        }
    )
    files = _result_files(runtime_root)
    stats = {
        "result_files": len(files),
        "physical_rows": 0,
        "successful_rows": 0,
        "malformed_rows": 0,
        "non_object_rows": 0,
        "exact_path_rows": 0,
        "basename_fallback_rows": 0,
        "unmapped_successful_rows": 0,
        "ambiguous_successful_rows": 0,
        "logical_successful_rows": 0,
        "duplicate_successful_row_occurrences": 0,
        "classified_successful_rows": 0,
        "orphan_successful_evidence": [],
        "ambiguous_successful_evidence": [],
    }
    logical_occurrence_counts: Counter[str] = Counter()
    for result_path in files:
        with result_path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                stats["physical_rows"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    stats["malformed_rows"] += 1
                    continue
                if not isinstance(row, dict):
                    stats["non_object_rows"] += 1
                    continue
                if str(row.get("status") or "").strip().lower() != "ok":
                    continue
                strategy_path_raw = str(row.get("strategy_file") or "").strip()
                leaf = _strategy_leaf(strategy_path_raw)
                strategy = str(row.get("strategy") or "").strip()
                if not leaf and strategy:
                    leaf = f"{strategy}.py"
                if not leaf.lower().startswith("sieve3") or not leaf.lower().endswith(".py"):
                    continue
                stats["successful_rows"] += 1
                raw_row_sha256 = _sha256_bytes(line.encode("utf-8"))
                logical_row_id = _sha256_bytes(
                    json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        default=str,
                    ).encode("utf-8")
                )
                occurrence_id = _sha256_bytes(
                    (
                        f"{_relative_path(result_path)}\n{line_number}\n{raw_row_sha256}"
                    ).encode("utf-8")
                )
                logical_occurrence_counts[logical_row_id] += 1
                matched_path: Path | None = None
                resolution = ""
                resolution_flag = ""
                unmatched_kind = ""
                if strategy_path_raw:
                    historical_path = Path(strategy_path_raw)
                    if not historical_path.is_absolute():
                        historical_path = REPO_ROOT / historical_path
                    matched_path = current_by_path.get(_path_key(historical_path))
                    if matched_path is not None:
                        resolution = "exact_path"
                        stats["exact_path_rows"] += 1
                    else:
                        candidates = current_by_leaf.get(leaf.casefold(), [])
                        if len(candidates) == 1:
                            matched_path = candidates[0]
                            resolution = "basename_alias"
                            resolution_flag = "historical_strategy_path_mismatch"
                            stats["basename_fallback_rows"] += 1
                        elif len(candidates) > 1:
                            stats["ambiguous_successful_rows"] += 1
                            unmatched_kind = "ambiguous_basename"
                        else:
                            stats["unmapped_successful_rows"] += 1
                            unmatched_kind = "orphan_missing_current_file"
                else:
                    candidates = current_by_leaf.get(leaf.casefold(), [])
                    if len(candidates) == 1:
                        matched_path = candidates[0]
                        resolution = "missing_path_basename_alias"
                        resolution_flag = "result_missing_strategy_path"
                        stats["basename_fallback_rows"] += 1
                    elif len(candidates) > 1:
                        stats["ambiguous_successful_rows"] += 1
                        unmatched_kind = "ambiguous_missing_path_basename"
                    else:
                        stats["unmapped_successful_rows"] += 1
                        unmatched_kind = "orphan_missing_path_and_current_file"
                if matched_path is None:
                    unmatched_evidence = {
                        "occurrence_id": occurrence_id,
                        "logical_row_id": logical_row_id,
                        "result_file": _relative_path(result_path),
                        "line_number": line_number,
                        "raw_row_sha256": raw_row_sha256,
                        "raw_strategy_path": strategy_path_raw,
                        "strategy": strategy,
                        "strategy_class": str(row.get("strategy_class") or ""),
                        "filename": leaf,
                        "job_id": str(row.get("job_id") or result_path.stem),
                        "backtest_file": str(row.get("backtest_file") or ""),
                        "resolution": unmatched_kind or "unmapped",
                    }
                    target = (
                        "ambiguous_successful_evidence"
                        if unmatched_kind.startswith("ambiguous")
                        else "orphan_successful_evidence"
                    )
                    stats[target].append(unmatched_evidence)
                    continue
                item = evidence[_path_key(matched_path)]
                item["successful_row_count"] += 1
                if resolution == "exact_path":
                    item["exact_successful_row_count"] += 1
                else:
                    item["alias_successful_row_count"] += 1
                item["successful_result_ids"].add(str(row.get("job_id") or result_path.stem))
                item["successful_result_files"].add(_relative_path(result_path))
                if strategy_path_raw:
                    item["historical_strategy_paths"].add(strategy_path_raw)
                if resolution_flag:
                    item["path_resolution_flags"].add(resolution_flag)
                item["successful_rows"].append(row)
                item["successful_evidence"].append(
                    {
                        "result_file": _relative_path(result_path),
                        "line_number": line_number,
                        "occurrence_id": occurrence_id,
                        "logical_row_id": logical_row_id,
                        "row_sha256": raw_row_sha256,
                        "raw_strategy_path": strategy_path_raw,
                        "resolution": resolution,
                        "job_id": str(row.get("job_id") or result_path.stem),
                        "finished_at": str(row.get("finished_at") or ""),
                        "strategy_class": str(row.get("strategy_class") or ""),
                        "training_window": str(row.get("training_window") or ""),
                        "training_timerange": str(row.get("training_timerange") or ""),
                        "validation_window": str(row.get("validation_window") or ""),
                        "validation_timerange": str(row.get("validation_timerange") or ""),
                        "random_state": str(row.get("random_state") or ""),
                        "sampling_seed": str(row.get("sampling_seed") or ""),
                        "backtest_file": str(row.get("backtest_file") or ""),
                    }
                )
                previous = item["latest_row"]
                if previous is None or str(row.get("finished_at") or "") >= str(previous.get("finished_at") or ""):
                    item["latest_row"] = row
    stats["logical_successful_rows"] = len(logical_occurrence_counts)
    stats["duplicate_successful_row_occurrences"] = sum(
        count - 1 for count in logical_occurrence_counts.values() if count > 1
    )
    stats["classified_successful_rows"] = (
        stats["exact_path_rows"]
        + stats["basename_fallback_rows"]
        + stats["unmapped_successful_rows"]
        + stats["ambiguous_successful_rows"]
    )
    return dict(evidence), stats


def _resolve_result_path(value: Any) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


@lru_cache(maxsize=None)
def _read_archive_snapshot(archive_path_text: str, strategy_class: str) -> dict[str, Any]:
    archive_path = Path(archive_path_text)
    empty = {
        "archive_status": "not_available",
        "archive_sha256": "",
        "snapshot_member": "",
        "snapshot_sha256": "",
        "params_member": "",
        "params_sha256": "",
        "params_status": "not_available",
        "dependency_status": "not_available",
        "dependency_fingerprint_sha256": "",
        "dependencies": [],
        "unresolved_local_modules": [],
    }
    if not archive_path.exists():
        empty["archive_status"] = "archive_missing"
        return empty
    if not strategy_class:
        empty["archive_status"] = "strategy_class_missing"
        return empty
    try:
        with zipfile.ZipFile(archive_path) as archive:
            python_members = [name for name in archive.namelist() if name.lower().endswith(".py")]
            exact = [
                name
                for name in python_members
                if strategy_class and name.lower().endswith(f"_{strategy_class.casefold()}.py")
            ]
            if len(exact) != 1:
                empty["archive_status"] = "snapshot_ambiguous"
                return empty
            member = exact[0]
            source_bytes = archive.read(member)
            snapshot_hash = _sha256_bytes(source_bytes)
            json_members = [name for name in archive.namelist() if name.lower().endswith(".json")]
            params_candidates = [
                name
                for name in json_members
                if strategy_class and name.lower().endswith(f"_{strategy_class.casefold()}.json")
            ]
            if len(params_candidates) != 1:
                empty["archive_status"] = "params_ambiguous"
                return empty
            params_member = params_candidates[0]
            params_bytes = archive.read(params_member)
            params_hash = _sha256_bytes(params_bytes)
            params_payload = json.loads(params_bytes.decode("utf-8-sig"))
            valid_params = (
                isinstance(params_payload, dict)
                and str(params_payload.get("strategy_name") or "") == strategy_class
                and isinstance(params_payload.get("params"), dict)
            )
            if not valid_params:
                empty["archive_status"] = "params_invalid"
                return empty
            params_status = "verified"
    except (
        OSError,
        zipfile.BadZipFile,
        KeyError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        empty["archive_status"] = "archive_error"
        return empty

    dependencies = _local_dependency_closure(source_bytes)
    empty.update(
        {
            "archive_status": "verified",
            "archive_sha256": _sha256_file(archive_path),
            "snapshot_member": member,
            "snapshot_sha256": snapshot_hash,
            "params_member": params_member,
            "params_sha256": params_hash,
            "params_status": params_status,
            "dependency_status": dependencies["status"],
            "dependency_fingerprint_sha256": dependencies["fingerprint_sha256"],
            "dependencies": dependencies["dependencies"],
            "unresolved_local_modules": dependencies["unresolved_local_modules"],
        }
    )
    return empty


def _inspect_result_snapshot(current_path: Path, row: dict[str, Any] | None) -> dict[str, Any]:
    empty = {
        "snapshot_status": "not_available",
        "snapshot_member": "",
        "snapshot_sha256": "",
        "snapshot_backtest_file": "",
        "snapshot_archive_sha256": "",
        "snapshot_params_member": "",
        "snapshot_params_sha256": "",
        "snapshot_params_status": "not_available",
        "snapshot_dependency_status": "not_available",
        "snapshot_dependency_fingerprint_sha256": "",
        "snapshot_dependencies": [],
        "snapshot_unresolved_local_modules": [],
    }
    if not row:
        return empty
    archive_path = _resolve_result_path(row.get("backtest_file"))
    if archive_path is None:
        return empty
    empty["snapshot_backtest_file"] = _relative_path(archive_path)
    archived = _read_archive_snapshot(
        str(archive_path.resolve()),
        str(row.get("strategy_class") or "").strip(),
    )
    empty.update(
        {
            "snapshot_member": archived["snapshot_member"],
            "snapshot_sha256": archived["snapshot_sha256"],
            "snapshot_archive_sha256": archived["archive_sha256"],
            "snapshot_params_member": archived["params_member"],
            "snapshot_params_sha256": archived["params_sha256"],
            "snapshot_params_status": archived["params_status"],
            "snapshot_dependency_status": archived["dependency_status"],
            "snapshot_dependency_fingerprint_sha256": archived[
                "dependency_fingerprint_sha256"
            ],
            "snapshot_dependencies": archived["dependencies"],
            "snapshot_unresolved_local_modules": archived["unresolved_local_modules"],
        }
    )
    if archived["archive_status"] != "verified":
        empty["snapshot_status"] = archived["archive_status"]
        return empty
    empty["snapshot_status"] = (
        "match" if archived["snapshot_sha256"] == _sha256_file(current_path) else "mismatch"
    )
    return empty


def inspect_latest_snapshot(current_path: Path, latest_row: dict[str, Any] | None) -> dict[str, Any]:
    return _inspect_result_snapshot(current_path, latest_row)


def inspect_snapshot_history(current_path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    unique_rows: dict[tuple[str, str], dict[str, Any]] = {}
    rows_seen = 0
    for row in rows:
        rows_seen += 1
        archive = _resolve_result_path(row.get("backtest_file"))
        archive_key = _path_key(archive) if archive is not None else ""
        key = (archive_key, str(row.get("strategy_class") or "").casefold())
        unique_rows.setdefault(key, row)

    snapshots = [_inspect_result_snapshot(current_path, row) for row in unique_rows.values()]
    status_counts: dict[str, int] = defaultdict(int)
    hashes: set[str] = set()
    members: list[dict[str, str]] = []
    dependency_catalog: dict[str, dict[str, Any]] = {}
    for snapshot in snapshots:
        status = snapshot["snapshot_status"]
        status_counts[status] += 1
        if snapshot["snapshot_sha256"]:
            hashes.add(snapshot["snapshot_sha256"])
        if snapshot["snapshot_backtest_file"]:
            members.append(
                {
                    "backtest_file": snapshot["snapshot_backtest_file"],
                    "member": snapshot["snapshot_member"],
                    "sha256": snapshot["snapshot_sha256"],
                    "archive_sha256": snapshot["snapshot_archive_sha256"],
                    "params_member": snapshot["snapshot_params_member"],
                    "params_sha256": snapshot["snapshot_params_sha256"],
                    "params_status": snapshot["snapshot_params_status"],
                    "dependency_status": snapshot["snapshot_dependency_status"],
                    "dependency_fingerprint_sha256": snapshot[
                        "snapshot_dependency_fingerprint_sha256"
                    ],
                    "status": status,
                }
            )
        source_hash = snapshot["snapshot_sha256"]
        if source_hash and source_hash not in dependency_catalog:
            dependency_catalog[source_hash] = {
                "source_sha256": source_hash,
                "status": snapshot["snapshot_dependency_status"],
                "fingerprint_sha256": snapshot[
                    "snapshot_dependency_fingerprint_sha256"
                ],
                "dependencies": snapshot["snapshot_dependencies"],
                "unresolved_local_modules": snapshot[
                    "snapshot_unresolved_local_modules"
                ],
            }

    match_count = status_counts.get("match", 0)
    mismatch_count = status_counts.get("mismatch", 0)
    unavailable_count = len(snapshots) - match_count - mismatch_count
    if not snapshots:
        history_status = "not_available"
    elif len(hashes) > 1:
        history_status = "historical_divergence"
    elif mismatch_count:
        history_status = "current_mismatch"
    elif match_count and unavailable_count:
        history_status = "partially_verified_current"
    elif match_count:
        history_status = "verified_current"
    else:
        history_status = "unavailable"

    return {
        "historical_snapshot_status": history_status,
        "historical_successful_rows_seen": rows_seen,
        "historical_snapshot_archives_checked": len(snapshots),
        "historical_snapshot_hashes": sorted(hashes),
        "historical_snapshot_match_count": match_count,
        "historical_snapshot_mismatch_count": mismatch_count,
        "historical_snapshot_unavailable_count": unavailable_count,
        "historical_snapshot_status_counts": dict(sorted(status_counts.items())),
        "historical_snapshot_members": sorted(
            members,
            key=lambda item: (item["backtest_file"], item["member"], item["sha256"]),
        ),
        "historical_snapshot_dependency_catalog": [
            dependency_catalog[key] for key in sorted(dependency_catalog)
        ],
    }


def _evidence_rows_with_archives(
    current_path: Path,
    evidence: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not evidence:
        return []
    output: list[dict[str, Any]] = []
    rows = evidence.get("successful_rows", ())
    summaries = evidence.get("successful_evidence", ())
    for row, summary in zip(rows, summaries, strict=True):
        snapshot = _inspect_result_snapshot(current_path, row)
        output.append(
            {
                **summary,
                "archive_path": snapshot["snapshot_backtest_file"],
                "archive_sha256": snapshot["snapshot_archive_sha256"],
                "source_member": snapshot["snapshot_member"],
                "source_sha256": snapshot["snapshot_sha256"],
                "source_status": snapshot["snapshot_status"],
                "params_member": snapshot["snapshot_params_member"],
                "params_sha256": snapshot["snapshot_params_sha256"],
                "params_status": snapshot["snapshot_params_status"],
                "dependency_status": snapshot["snapshot_dependency_status"],
                "dependency_fingerprint_sha256": snapshot[
                    "snapshot_dependency_fingerprint_sha256"
                ],
                "unresolved_local_dependency_modules": snapshot[
                    "snapshot_unresolved_local_modules"
                ],
            }
        )
    return output


def _compact_result(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    return {key: row.get(key) for key in RESULT_SUMMARY_KEYS if row.get(key) not in (None, "")}


def _record_for_strategy(path: Path, evidence: dict[str, Any] | None, *, inspect_snapshots: bool) -> dict[str, Any]:
    parsed = parse_strategy(path)
    metadata = parsed["metadata"]
    metadata_values = parsed["metadata_values"]
    source_entry_stem_raw = str(metadata.get("SOURCE_ENTRY_STEM") or "")
    source_entry_stem = canonical_source_slug(source_entry_stem_raw)
    source_strategy_raw = str(metadata.get("SOURCE_STRATEGY") or "")
    metadata_source = canonical_source_slug(source_strategy_raw)
    filename_source = filename_source_slug(path)
    metadata_matches = bool(metadata_source and metadata_source == filename_source)
    if source_entry_stem:
        source = source_entry_stem
        source_resolution = "source_entry_stem"
    else:
        source = metadata_source if metadata_matches or not filename_source else filename_source
        source_resolution = (
            "module_metadata_agrees_filename"
            if metadata_matches
            else "filename_metadata_mismatch"
            if metadata_source and filename_source
            else "module_metadata"
            if metadata_source
            else "filename"
        )
    pilot = path.name.startswith(PILOT_PREFIX)
    exact_tested = bool(evidence and evidence.get("exact_successful_row_count"))
    protected_alias = bool(
        evidence and evidence.get("alias_successful_row_count") and not exact_tested
    )
    flags: list[str] = []
    if parsed["parse_error"]:
        flags.append("strategy_parse_error")
    if parsed["buy_parameter_default_conflicts"]:
        flags.append("conflicting_buy_parameter_defaults")
    if parsed["unresolved_buy_parameter_defaults"]:
        flags.append("unresolved_buy_parameter_defaults")
    if len(metadata_values.get("SOURCE_RESULT_BATCH", [])) > 1:
        flags.append("multiple_source_result_batch_metadata_values")
    if not source:
        flags.append("missing_source_lineage")
    if source_entry_stem and filename_source and source_entry_stem != filename_source:
        flags.append("source_entry_stem_filename_mismatch")
    elif metadata_source and filename_source and metadata_source != filename_source:
        flags.append("metadata_filename_source_mismatch")
    if evidence:
        flags.extend(sorted(evidence.get("path_resolution_flags", ())))

    latest_row = evidence.get("latest_row") if evidence else None
    has_evidence = bool(evidence)
    snapshot = inspect_latest_snapshot(path, latest_row) if inspect_snapshots and has_evidence else {
        "snapshot_status": "not_checked" if has_evidence else "not_available",
        "snapshot_member": "",
        "snapshot_sha256": "",
        "snapshot_backtest_file": "",
        "snapshot_archive_sha256": "",
        "snapshot_params_member": "",
        "snapshot_params_sha256": "",
        "snapshot_params_status": "not_checked" if has_evidence else "not_available",
        "snapshot_dependency_status": "not_checked" if has_evidence else "not_available",
        "snapshot_dependency_fingerprint_sha256": "",
        "snapshot_dependencies": [],
        "snapshot_unresolved_local_modules": [],
    }
    snapshot_history = (
        inspect_snapshot_history(path, evidence.get("successful_rows", ()))
        if inspect_snapshots and has_evidence and evidence
        else {
            "historical_snapshot_status": "not_checked" if has_evidence else "not_available",
            "historical_successful_rows_seen": evidence.get("successful_row_count", 0) if evidence else 0,
            "historical_snapshot_archives_checked": 0,
            "historical_snapshot_hashes": [],
            "historical_snapshot_match_count": 0,
            "historical_snapshot_mismatch_count": 0,
            "historical_snapshot_unavailable_count": 0,
            "historical_snapshot_status_counts": {},
            "historical_snapshot_members": [],
            "historical_snapshot_dependency_catalog": [],
        }
    )
    if snapshot["snapshot_status"] == "mismatch":
        flags.append("current_differs_from_latest_tested_snapshot")
    elif snapshot["snapshot_status"] in {"snapshot_ambiguous", "archive_error"}:
        flags.append(snapshot["snapshot_status"])
    if len(snapshot_history["historical_snapshot_hashes"]) > 1:
        flags.append("multiple_historical_tested_snapshots")
    if snapshot_history["historical_snapshot_mismatch_count"]:
        flags.append("current_differs_from_historical_tested_snapshot")
    if (
        len(snapshot_history["historical_snapshot_hashes"]) > 1
        or snapshot_history["historical_snapshot_mismatch_count"]
    ):
        flags.append("historical_tested_snapshot_divergence")

    pattern = _is_pattern(source or filename_source)
    if pilot:
        category = "refined_pilot"
        deletion_status = "preserve_refined_pilot"
    elif exact_tested:
        category = "tested_legacy"
        deletion_status = "preserve_tested_original"
    elif protected_alias:
        category = "protected_alias"
        deletion_status = "preserve_ambiguous_alias"
    else:
        category = "untested_legacy"
        deletion_status = "blocked_ambiguous" if flags else "pending_validated_v2_replacement"

    hypothesis = str(
        metadata.get("EXIT_HYPOTHESIS")
        or metadata.get("RESEARCH_PATH")
        or _branch_family(path, metadata)
    ).strip()
    return {
        "legacy_file": _relative_path(path),
        "filename": path.name,
        "category": category,
        "canonical_source": source,
        "source_resolution": source_resolution,
        "source_entry_stem": source_entry_stem,
        "source_entry_stem_raw": source_entry_stem_raw,
        "metadata_source": metadata_source,
        "filename_source": filename_source,
        "source_strategy_raw": source_strategy_raw,
        "source_sieve2_strategy_raw": str(metadata.get("SOURCE_SIEVE2_STRATEGY") or ""),
        "source_sieve2_strategy_values": [
            str(value) for value in metadata_values.get("SOURCE_SIEVE2_STRATEGY", [])
        ],
        "source_result_batch": str(metadata.get("SOURCE_RESULT_BATCH") or ""),
        "source_result_batches": [
            str(value) for value in metadata_values.get("SOURCE_RESULT_BATCH", [])
        ],
        "branch_family": _branch_family(path, metadata),
        "retained_hypothesis": hypothesis,
        "pattern_source": pattern,
        "strategy_classes": parsed["class_names"],
        "entry_signature_sha256": parsed["entry_signature_sha256"],
        "buy_parameter_defaults": parsed["buy_parameter_defaults"],
        "buy_parameter_defaults_sha256": parsed["buy_parameter_defaults_sha256"],
        "buy_parameter_default_conflicts": parsed["buy_parameter_default_conflicts"],
        "unresolved_buy_parameter_defaults": parsed["unresolved_buy_parameter_defaults"],
        "current_sha256": _sha256_file(path),
        "tested_successfully": exact_tested,
        "protected_by_alias_evidence": protected_alias,
        "successful_row_count": evidence.get("successful_row_count", 0) if evidence else 0,
        "exact_successful_row_count": evidence.get("exact_successful_row_count", 0) if evidence else 0,
        "alias_successful_row_count": evidence.get("alias_successful_row_count", 0) if evidence else 0,
        "successful_result_ids": sorted(evidence.get("successful_result_ids", ())) if evidence else [],
        "successful_result_files": sorted(evidence.get("successful_result_files", ())) if evidence else [],
        "historical_strategy_paths": sorted(evidence.get("historical_strategy_paths", ())) if evidence else [],
        "latest_successful_result": _compact_result(latest_row),
        "external_result_paths_authoritative": False,
        "successful_evidence_rows": (
            _evidence_rows_with_archives(path, evidence) if inspect_snapshots else []
        ),
        **snapshot,
        **snapshot_history,
        "v2_replacement": _v2_filename(source, pattern) if source else "",
        "validation_status": "inventory_pending_independent_review",
        "deletion_eligible": False,
        "deletion_status": deletion_status,
        "ambiguity_flags": flags,
        "source_group_ambiguity_flags": [],
        "parse_error": parsed["parse_error"],
    }


def build_migration_ledger(
    strategy_dir: Path = DEFAULT_STRATEGY_DIR,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    *,
    inspect_snapshots: bool = True,
    baseline_lookup: Path = DEFAULT_BASELINE_LOOKUP,
) -> dict[str, Any]:
    all_sieve3 = sorted(path for path in strategy_dir.glob("sieve3*.py") if path.is_file())
    existing_v2 = [path for path in all_sieve3 if path.name.startswith(V2_PREFIX)]
    legacy = [path for path in all_sieve3 if not path.name.startswith(V2_PREFIX)]
    evidence_by_path, result_stats = scan_successful_results(runtime_root, legacy)
    records = [
        _record_for_strategy(path, evidence_by_path.get(_path_key(path)), inspect_snapshots=inspect_snapshots)
        for path in legacy
    ]

    source_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["canonical_source"]:
            source_records[record["canonical_source"]].append(record)
    source_contexts = _source_contexts(source_records)
    baseline_lineage, baseline_stats = resolve_context_baseline_lineage(
        runtime_root,
        baseline_lookup,
        source_contexts,
    )

    signature_sources: dict[str, set[str]] = defaultdict(set)
    for record in records:
        signature = record["entry_signature_sha256"]
        source = record["canonical_source"]
        if signature and source:
            signature_sources[signature].add(source)
    baseline_signature_sources: dict[str, set[str]] = defaultdict(set)
    for source, baseline in baseline_lineage.items():
        if baseline.get("status") != "verified":
            continue
        signature = str(baseline.get("baseline_entry_signature_sha256") or "")
        if signature:
            baseline_signature_sources[signature].add(source)

    source_groups: list[dict[str, Any]] = []
    source_group_flags: dict[str, list[str]] = {}
    for source, members in sorted(source_records.items()):
        signatures = sorted({row["entry_signature_sha256"] for row in members if row["entry_signature_sha256"]})
        cross_source_duplicates = sorted(
            {
                duplicate_source
                for signature in signatures
                for duplicate_source in signature_sources[signature]
                if duplicate_source != source
            }
        )
        baseline = baseline_lineage.get(source)
        baseline_signature = (
            str((baseline or {}).get("baseline_entry_signature_sha256") or "")
            if baseline and baseline.get("status") == "verified"
            else ""
        )
        baseline_cross_source_duplicates = sorted(
            duplicate_source
            for duplicate_source in baseline_signature_sources.get(baseline_signature, set())
            if duplicate_source != source
        )
        flags = sorted({flag for row in members for flag in row["ambiguity_flags"]})
        if baseline is None:
            flags.append("missing_sieve2_baseline_lookup")
        else:
            flags.extend(baseline.get("ambiguity_flags", []))
            if baseline.get("status") != "verified":
                flags.append("sieve2_baseline_lineage_unverified")
        if len(signatures) > 1:
            flags.append("multiple_entry_signatures_within_source")
        if baseline_cross_source_duplicates:
            flags.append("baseline_entry_signature_shared_across_sources")
        unique_flags = sorted(set(flags))
        source_group_flags[source] = unique_flags
        source_groups.append(
            {
                "canonical_source": source,
                "pattern_source": any(row["pattern_source"] for row in members),
                "v2_replacement": _v2_filename(source, any(row["pattern_source"] for row in members)),
                "legacy_file_count": len(members),
                "tested_legacy_files": sorted(
                    row["filename"] for row in members if row["category"] == "tested_legacy"
                ),
                "protected_alias_files": sorted(
                    row["filename"] for row in members if row["category"] == "protected_alias"
                ),
                "untested_legacy_files": sorted(
                    row["filename"] for row in members if row["category"] == "untested_legacy"
                ),
                "refined_pilot_files": sorted(
                    row["filename"] for row in members if row["category"] == "refined_pilot"
                ),
                "branch_families": sorted({row["branch_family"] for row in members}),
                "retained_hypotheses": sorted({row["retained_hypothesis"] for row in members if row["retained_hypothesis"]}),
                "entry_signature_sha256": signatures,
                "duplicate_entry_signature_sources": baseline_cross_source_duplicates,
                "current_duplicate_entry_signature_sources": cross_source_duplicates,
                "baseline_entry_signature_sha256": baseline_signature,
                "sieve2_baseline": baseline or {},
                "ambiguity_flags": unique_flags,
                "validation_status": "inventory_pending_independent_review",
                "deletion_approved": False,
            }
        )

    for record in records:
        group_flags = source_group_flags.get(record["canonical_source"], [])
        record["source_group_ambiguity_flags"] = group_flags
        if record["category"] == "untested_legacy" and group_flags:
            record["ambiguity_flags"] = sorted(set([*record["ambiguity_flags"], "source_group_ambiguity"]))
            record["deletion_status"] = "blocked_ambiguous"

    tested_legacy = [row for row in records if row["category"] == "tested_legacy"]
    untested_legacy = [row for row in records if row["category"] == "untested_legacy"]
    protected_aliases = [row for row in records if row["category"] == "protected_alias"]
    pilots = [row for row in records if row["category"] == "refined_pilot"]
    ambiguous = [row for row in records if row["ambiguity_flags"]]
    return {
        "schema": "sieve3_v2_migration_ledger",
        "schema_version": 3,
        "generator": {
            "path": _relative_path(Path(__file__)),
            "sha256": _sha256_file(Path(__file__)),
        },
        "generated_at": datetime.now().astimezone().isoformat(),
        "strategy_dir": _relative_path(strategy_dir),
        "runtime_root": _relative_path(runtime_root),
        "baseline_lookup": _relative_path(baseline_lookup),
        "policy": {
            "tested_definition": "current exact path referenced by at least one physical status=ok Sieve result row",
            "tested_originals": "preserve unchanged",
            "protected_alias": "preserve basename-only historical matches as ambiguous controls; never count them as exact tested",
            "refined_pilot": "preserve unchanged for comparison",
            "untested_originals": "not deletable until validated V2 coverage and independent approval",
            "historical_results": "preserve all artifacts",
        },
        "summary": {
            "top_level_sieve3_files": len(all_sieve3),
            "existing_v2_files": len(existing_v2),
            "legacy_and_pilot_files": len(legacy),
            "tested_legacy_files": len(tested_legacy),
            "untested_legacy_files": len(untested_legacy),
            "protected_alias_files": len(protected_aliases),
            "refined_pilot_files": len(pilots),
            "canonical_source_groups": len(source_groups),
            "pattern_source_groups": sum(1 for row in source_groups if row["pattern_source"]),
            "ambiguous_files": len(ambiguous),
            "deletion_eligible_files": 0,
            "verified_sieve2_baselines": baseline_stats["verified_param_snapshots"],
            "blocked_sieve2_baselines": len(source_groups)
            - baseline_stats["verified_param_snapshots"],
            "unique_verified_baseline_entry_signatures": len(baseline_signature_sources),
        },
        "result_scan": result_stats,
        "baseline_lineage_scan": baseline_stats,
        "existing_v2_files": [_relative_path(path) for path in existing_v2],
        "source_groups": source_groups,
        "records": records,
    }


def _summary_for_stdout(ledger: dict[str, Any], output: Path | None) -> dict[str, Any]:
    compact_result_scan = {
        key: value
        for key, value in ledger["result_scan"].items()
        if key not in {"orphan_successful_evidence", "ambiguous_successful_evidence"}
    }
    return {
        "output": _relative_path(output) if output is not None else "",
        "summary": ledger["summary"],
        "result_scan": compact_result_scan,
        "baseline_lineage_scan": ledger["baseline_lineage_scan"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the non-destructive Sieve3 V2 migration ledger.")
    parser.add_argument("--strategy-dir", type=Path, default=DEFAULT_STRATEGY_DIR)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--baseline-lookup", type=Path, default=DEFAULT_BASELINE_LOOKUP)
    parser.add_argument("--output", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--no-snapshot-check", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Build and print counts without writing the ledger.")
    args = parser.parse_args(argv)

    ledger = build_migration_ledger(
        strategy_dir=args.strategy_dir,
        runtime_root=args.runtime_root,
        inspect_snapshots=not args.no_snapshot_check,
        baseline_lookup=args.baseline_lookup,
    )
    output = None if args.check_only else args.output
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(output, ledger)
    print(json.dumps(_summary_for_stdout(ledger, output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

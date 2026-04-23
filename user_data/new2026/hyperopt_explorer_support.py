#!/usr/bin/env python3
"""Shared Explorer support."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time
from typing import Any
import zipfile


for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


THIS_DIR = Path(__file__).resolve().parent
MARKET_WINDOWS_FILE = THIS_DIR / "explorer_market_windows.json"
PRESET_FILE = THIS_DIR / "freqtrade_launcher_presets.json"
STRATEGY_PARAM_FILE = THIS_DIR.parent / "strategies" / "HybridRecoveryGridStrategy_v11.json"
CUSTOM_BATCH_FILE = THIS_DIR / "hyperopt_custom_batches.json"
PARAM_ENV = "HYBRID_RECOVERY_HYPEROPT_PARAMS"
TAIL_LINE_COUNT = 80
MAX_HYPEROPT_EPOCHS = 800

OBJECTIVE_PROFIT_WEIGHT = 420.0
OBJECTIVE_WINRATE_WEIGHT = 160.0
OBJECTIVE_MIN_WINRATE = 0.52
OBJECTIVE_WINRATE_BELOW_FLOOR_PENALTY = 90.0
OBJECTIVE_MIN_TRADES_PER_DAY = 0.35
OBJECTIVE_LOW_ACTIVITY_PENALTY_WEIGHT = 45.0
OBJECTIVE_DD_SOFT = 0.22
OBJECTIVE_DD_HARD = 0.38
OBJECTIVE_DD_SOFT_PENALTY_WEIGHT = 140.0
OBJECTIVE_DD_HARD_STEP_PENALTY = 55.0
OBJECTIVE_NON_PROFIT_BASE_PENALTY = 120.0
OBJECTIVE_NON_PROFIT_RATIO_PENALTY = 240.0
OBJECTIVE_MAX_LOSS = 90000.0
OBJECTIVE_NOISE_BAND = 20.0

RISK_PRIORITY_HOLDOUT_POSITIVE_COUNT_WEIGHT = 180.0
RISK_PRIORITY_COMPLETED_POSITIVE_COUNT_WEIGHT = 140.0
RISK_PRIORITY_HOLDOUT_POSITIVE_TOTAL_WEIGHT = 1.0
RISK_PRIORITY_COMPLETED_POSITIVE_TOTAL_WEIGHT = 0.8
RISK_PRIORITY_HOLDOUT_WORST_POSITIVE_WEIGHT = 0.7
RISK_PRIORITY_COMPLETED_WORST_POSITIVE_WEIGHT = 0.5
RISK_PRIORITY_COMPLETED_OBJECTIVE_TOTAL_WEIGHT = 0.15


def token_list(value: str | None) -> list[str]:
    if not value:
        return []
    normalized = value.replace(",", " ").replace(";", " ")
    return [item.strip().strip('"').strip("'") for item in normalized.split() if item.strip()]


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return deepcopy(default)
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return deepcopy(default)


def load_market_windows_manifest(path: Path | str = MARKET_WINDOWS_FILE) -> dict[str, Any]:
    payload = load_json(Path(path), {})
    if not isinstance(payload, dict):
        return {"market_windows": []}
    windows = payload.get("market_windows")
    payload["market_windows"] = windows if isinstance(windows, list) else []
    return payload


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def slugify_batch_id(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return slug or "custom_batch"


def unique_batch_id(name: str, existing_ids: set[str]) -> str:
    base = slugify_batch_id(name)
    candidate = base
    index = 2
    while candidate in existing_ids:
        candidate = f"{base}_{index}"
        index += 1
    return candidate


def empty_custom_batches_payload() -> dict[str, Any]:
    return {"version": 1, "updated_at": datetime.now().astimezone().isoformat(), "batches": []}


def load_custom_batches(path: Path | str = CUSTOM_BATCH_FILE) -> dict[str, Any]:
    payload = load_json(Path(path), empty_custom_batches_payload())
    if not isinstance(payload, dict):
        payload = empty_custom_batches_payload()
    payload.setdefault("version", 1)
    payload.setdefault("updated_at", datetime.now().astimezone().isoformat())
    batches = payload.get("batches")
    payload["batches"] = batches if isinstance(batches, list) else []
    return payload


def save_custom_batches(path: Path | str, payload: dict[str, Any]) -> None:
    payload["version"] = int(payload.get("version") or 1)
    payload["updated_at"] = datetime.now().astimezone().isoformat()
    save_json(Path(path), payload)


def strategy_identity(strategy_file: str | Path, strategy_class: str) -> dict[str, str]:
    path = Path(strategy_file).resolve()
    return {
        "strategy_file": str(path),
        "strategy_class": str(strategy_class or ""),
        "strategy_file_name": path.name,
    }


def batch_matches_strategy(batch: dict[str, Any], catalog: dict[str, Any]) -> bool:
    """Return true when a custom batch belongs to the catalog strategy."""
    batch_file = str(batch.get("strategy_file") or "").strip()
    batch_class = str(batch.get("strategy_class") or "").strip()
    if not batch_file and not batch_class:
        return True

    catalog_file = str(catalog.get("strategy_file") or "").strip()
    catalog_class = str(catalog.get("strategy_class") or "").strip()
    if batch_class and catalog_class and batch_class != catalog_class:
        return False
    if batch_file and catalog_file:
        try:
            return Path(batch_file).resolve() == Path(catalog_file).resolve()
        except Exception:
            return Path(batch_file).name == Path(catalog_file).name
    return True


def custom_batch_source_label(source: dict[str, Any]) -> str:
    source_type = str(source.get("type") or "")
    source_id = str(source.get("id") or "")
    context = source.get("context") if isinstance(source.get("context"), dict) else None
    if source_type == "tag" and context and context.get("family"):
        return f"family:{context['family']} / tag:{source_id}"
    if source_type == "namespace_value":
        namespace = str(source.get("namespace") or "")
        value = str(source.get("value") or source_id or "")
        if not namespace and ":" in source_id:
            namespace, value = source_id.split(":", 1)
        return f"{namespace}:{value}" if namespace and value else f"namespace_value:{source_id}"
    if source_type == "family_tag_intersection":
        family = str(source.get("family") or "")
        tag = str(source.get("tag") or "")
        if (not family or not tag) and "|" in source_id:
            family, tag = source_id.split("|", 1)
        return f"family:{family} / tag:{tag}" if family and tag else f"family_tag_intersection:{source_id}"
    if source_type == "namespace":
        return f"namespace:{source_id}"
    return f"{source_type}:{source_id}"


def normalize_custom_batch_source(raw_source: dict[str, Any]) -> dict[str, Any]:
    source_type = str(raw_source.get("type") or "")
    source_id = str(raw_source.get("id") or "")
    namespace = str(raw_source.get("namespace") or "")
    value = str(raw_source.get("value") or "")
    family = str(raw_source.get("family") or "")
    tag = str(raw_source.get("tag") or "")
    context = raw_source.get("context") if isinstance(raw_source.get("context"), dict) else None

    if source_type == "namespace_value":
        if not namespace and ":" in source_id:
            namespace, value_from_id = source_id.split(":", 1)
            value = value or value_from_id
        if not source_id and namespace and value:
            source_id = value
    if source_type == "family_tag_intersection":
        if (not family or not tag) and "|" in source_id:
            family_from_id, tag_from_id = source_id.split("|", 1)
            family = family or family_from_id
            tag = tag or tag_from_id
        if not source_id and family and tag:
            source_id = f"{family}|{tag}"
    return {
        "type": source_type,
        "id": source_id,
        "namespace": namespace,
        "value": value,
        "family": family,
        "tag": tag,
        "context": context,
    }


def resolve_custom_batch(batch: dict[str, Any], catalog: dict[str, Any], strategy_spaces: dict[str, str]) -> dict[str, Any]:
    params = catalog.get("params") or {}
    family_index = catalog.get("family_index") or {}
    tag_index = catalog.get("tag_index") or {}
    namespaces = catalog.get("namespaces") or {}

    resolved: set[str] = set()
    stale_sources: list[dict[str, Any]] = []
    for raw_source in batch.get("sources") or []:
        if not isinstance(raw_source, dict):
            continue
        source = normalize_custom_batch_source(raw_source)
        source_params: set[str] = set()
        if source["type"] == "family":
            source_params = {str(param) for param in family_index.get(source["id"], []) if str(param)}
        elif source["type"] == "tag":
            tag_params = {str(param) for param in tag_index.get(source["id"], []) if str(param)}
            context = source.get("context") or {}
            family = str(context.get("family") or "")
            if family:
                family_params = {str(param) for param in family_index.get(family, []) if str(param)}
                source_params = tag_params & family_params
            else:
                source_params = tag_params
        elif source["type"] == "namespace":
            namespace = source["id"]
            namespace_tags = [f"{namespace}:{value}" for value in namespaces.get(namespace, [])]
            for tag in namespace_tags:
                source_params.update(str(param) for param in tag_index.get(tag, []) if str(param))
        elif source["type"] == "namespace_value":
            namespace = source["namespace"]
            value = source["value"] or source["id"]
            source_params = {str(param) for param in tag_index.get(f"{namespace}:{value}", []) if str(param)}
        elif source["type"] == "family_tag_intersection":
            family_params = {str(param) for param in family_index.get(source["family"], []) if str(param)}
            tag_params = {str(param) for param in tag_index.get(source["tag"], []) if str(param)}
            source_params = family_params & tag_params
        elif source["type"] == "param":
            if source["id"] in params:
                source_params = {source["id"]}

        if not source_params:
            stale_sources.append(source)
        resolved.update(source_params)

    excluded = {str(param) for param in batch.get("excluded_params") or [] if str(param)}
    stale_excluded_params = sorted(param for param in excluded if param not in params)
    final_params = sorted(param for param in (resolved - excluded) if param in params)
    stale_params = sorted(param for param in resolved if param not in params)
    spaces = sorted({str(strategy_spaces.get(param) or (params.get(param) or {}).get("space") or "") for param in final_params if str(strategy_spaces.get(param) or (params.get(param) or {}).get("space") or "")})
    return {
        "params": final_params,
        "spaces": spaces,
        "excluded_params": sorted(excluded),
        "stale_sources": stale_sources,
        "stale_params": stale_params,
        "stale_excluded_params": stale_excluded_params,
        "valid": bool(str(batch.get("name") or "").strip()) and bool(final_params) and bool(spaces),
    }


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def strategy_parameter_spaces(strategy_file: Path, strategy_class: str = "HybridRecoveryGridStrategy") -> dict[str, str]:
    """Return strategy hyperopt parameter spaces using the AST tag catalog."""
    if not strategy_file.exists() or not strategy_file.is_file():
        return {}
    try:
        from hyperopt_tag_catalog import build_param_catalog

        catalog = build_param_catalog(strategy_file, strategy_class)
        spaces: dict[str, str] = {}
        for name, info in (catalog.get("params") or {}).items():
            space = str((info or {}).get("space") or "").strip()
            if space:
                spaces[str(name)] = space
        if spaces:
            return spaces
        raise ValueError("AST catalog did not return any strategy parameter spaces.")
    except Exception as exc:
        raise ValueError(
            "Failed to parse strategy parameters via AST catalog. "
            "Fix catalog parsing instead of falling back to regex."
        ) from exc


def build_common_args(preset: dict[str, Any], include_strategy: bool) -> list[str]:
    args: list[str] = []
    for _ in range(int(preset.get("verbose") or 0)):
        args.append("-v")
    if preset.get("no_color"):
        args.append("--no-color")
    if preset.get("logfile"):
        args.extend(["--logfile", str(preset["logfile"])])
    for config_file in preset.get("config_files", []):
        if str(config_file).strip():
            args.extend(["-c", str(config_file).strip()])
    if preset.get("datadir"):
        args.extend(["--datadir", str(preset["datadir"])])
    if preset.get("userdir"):
        args.extend(["--userdir", str(preset["userdir"])])
    if include_strategy:
        strategy_file = str(preset.get("strategy_file") or "").strip()
        if strategy_file:
            args.extend(["--strategy-path", str(Path(strategy_file).resolve().parent)])
        if preset.get("strategy_class"):
            args.extend(["--strategy", str(preset["strategy_class"])])
        if preset.get("recursive_strategy_search"):
            args.append("--recursive-strategy-search")
    return args


def build_hyperopt_args(
    preset: dict[str, Any],
    group: dict[str, Any],
    timerange: str,
    epochs: str | None,
    random_state: str | None,
) -> list[str]:
    args = ["hyperopt", *build_common_args(preset, include_strategy=True)]
    if preset.get("timeframe"):
        args.extend(["-i", str(preset["timeframe"])])
    args.extend(["--timerange", timerange])
    if preset.get("timeframe_detail"):
        args.extend(["--timeframe-detail", str(preset["timeframe_detail"])])
    if preset.get("max_open_trades"):
        args.extend(["--max-open-trades", str(preset["max_open_trades"])])
    if preset.get("stake_amount"):
        args.extend(["--stake-amount", str(preset["stake_amount"])])
    if preset.get("fee"):
        args.extend(["--fee", str(preset["fee"])])
    if preset.get("dry_run_wallet"):
        args.extend(["--dry-run-wallet", str(preset["dry_run_wallet"])])
    if str(preset.get("pair_mode") or "") == "manual":
        pairs = token_list(str(preset.get("pairs") or ""))
        if pairs:
            args.extend(["-p", *pairs])
    if preset.get("enable_protections"):
        args.append("--enable-protections")
    if preset.get("enable_position_stacking"):
        args.append("--eps")
    epoch_raw = str(epochs or preset.get("hyperopt_epochs") or "100").strip()
    try:
        epoch_count = int(epoch_raw)
    except ValueError:
        epoch_count = 100
    epoch_count = max(1, min(MAX_HYPEROPT_EPOCHS, epoch_count))
    args.extend(["-e", str(epoch_count)])
    spaces = group.get("spaces") or token_list(str(preset.get("hyperopt_spaces") or ""))
    if spaces:
        args.extend(["--spaces", *spaces])
    if preset.get("hyperopt_jobs"):
        jobs_raw = str(preset["hyperopt_jobs"]).strip()
        try:
            jobs_value = int(jobs_raw)
        except ValueError:
            jobs_value = 0
        if jobs_value < 1:
            cpu_total = max(1, int(os.cpu_count() or 1))
            jobs_value = max(1, cpu_total + 1 + jobs_value)
        args.extend(["-j", str(jobs_value)])
    if random_state or preset.get("hyperopt_random_state"):
        args.extend(["--random-state", str(random_state or preset["hyperopt_random_state"])])
    if preset.get("hyperopt_min_trades"):
        args.extend(["--min-trades", str(preset["hyperopt_min_trades"])])
    if preset.get("hyperopt_loss"):
        args.extend(["--hyperopt-loss", str(preset["hyperopt_loss"])])
    if preset.get("hyperopt_early_stop"):
        args.extend(["--early-stop", str(preset["hyperopt_early_stop"])])
    if preset.get("hyperopt_ignore_missing_spaces"):
        args.append("--ignore-missing-spaces")
    if preset.get("hyperopt_analyze_per_epoch"):
        args.append("--analyze-per-epoch")
    if preset.get("hyperopt_print_all"):
        args.append("--print-all")
    args.append("--disable-param-export")
    return args


def build_backtest_args(preset: dict[str, Any], timerange: str) -> list[str]:
    args = ["backtesting", *build_common_args(preset, include_strategy=True)]
    if preset.get("timeframe"):
        args.extend(["-i", str(preset["timeframe"])])
    args.extend(["--timerange", timerange])
    if preset.get("timeframe_detail"):
        args.extend(["--timeframe-detail", str(preset["timeframe_detail"])])
    if preset.get("max_open_trades"):
        args.extend(["--max-open-trades", str(preset["max_open_trades"])])
    if preset.get("stake_amount"):
        args.extend(["--stake-amount", str(preset["stake_amount"])])
    if preset.get("fee"):
        args.extend(["--fee", str(preset["fee"])])
    if preset.get("dry_run_wallet"):
        args.extend(["--dry-run-wallet", str(preset["dry_run_wallet"])])
    if str(preset.get("pair_mode") or "") == "manual":
        pairs = token_list(str(preset.get("pairs") or ""))
        if pairs:
            args.extend(["-p", *pairs])
    if preset.get("enable_protections"):
        args.append("--enable-protections")
    if preset.get("enable_position_stacking"):
        args.append("--eps")
    # Request a result file so metrics can be parsed after the run.
    # Launcher presets may store "default", but Freqtrade backtesting only accepts:
    # none, trades, signals.
    export_value = str(preset.get("backtest_export") or "").strip().lower()
    if export_value in ("", "default"):
        export_value = "trades"
    if export_value not in {"none", "trades", "signals"}:
        export_value = "trades"
    args.extend(["--export", export_value])
    # Older Freqtrade builds do not support --backtest-directory, but they do
    # honor --export-filename with a directory path and write results plus
    # .last_result.json into that directory.
    if preset.get("backtest_directory"):
        args.extend(["--export-filename", str(preset["backtest_directory"])])
    return args


def hyperopt_results_dir(preset: dict[str, Any]) -> Path:
    userdir = Path(str(preset.get("userdir") or "user_data"))
    return userdir / "hyperopt_results"


def backtest_results_dir(preset: dict[str, Any]) -> Path:
    userdir = Path(str(preset.get("userdir") or "user_data"))
    return userdir / "backtest_results"


def marker_latest_file(directory: Path) -> Path | None:
    marker = directory / ".last_result.json"
    if not marker.exists():
        return None
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return None
    for key in ("latest_hyperopt", "latest_backtest", "latest_backtest_result", "latest_result"):
        filename = data.get(key)
        if not filename:
            continue
        path = directory / str(filename)
        if path.exists():
            return path
    return None


def result_file_snapshot(directory: Path, patterns: tuple[str, ...]) -> dict[Path, tuple[int, int]]:
    snapshot: dict[Path, tuple[int, int]] = {}
    for pattern in patterns:
        for path in directory.glob(pattern):
            try:
                stat = path.stat()
            except OSError:
                continue
            snapshot[path] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def changed_result_file(directory: Path, patterns: tuple[str, ...], before: dict[Path, tuple[int, int]]) -> Path | None:
    changed: list[Path] = []
    for pattern in patterns:
        for path in directory.glob(pattern):
            if is_result_sidecar(path):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            previous = before.get(path)
            current = (stat.st_mtime_ns, stat.st_size)
            if previous != current:
                changed.append(path)
    if not changed:
        return None
    return newest_result_file(changed)


def is_result_sidecar(path: Path) -> bool:
    name = path.name.lower()
    return name == ".last_result.json" or name.endswith(".meta.json")


def result_file_sort_key(path: Path) -> tuple[int, int]:
    suffix = path.suffix.lower()
    priority = 2 if suffix == ".zip" else 1 if suffix == ".json" else 0
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        mtime = 0
    return priority, mtime


def newest_result_file(paths: list[Path]) -> Path | None:
    candidates = [path for path in paths if not is_result_sidecar(path)]
    if not candidates:
        return None
    return max(candidates, key=result_file_sort_key)


def latest_result_file(
    directory: Path,
    patterns: tuple[str, ...],
    started_at: float,
    previous_result: Path | None,
    before_snapshot: dict[Path, tuple[int, int]],
) -> Path | None:
    deadline = time.time() + 5.0
    while True:
        changed = changed_result_file(directory, patterns, before_snapshot)
        if changed:
            return changed

        marker_result = marker_latest_file(directory)
        if marker_result and not is_result_sidecar(marker_result):
            if marker_result != previous_result:
                return marker_result
            if marker_result.stat().st_mtime >= started_at - 60:
                return marker_result

        if time.time() >= deadline:
            return None
        time.sleep(0.25)


def run_command(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    dry_run: bool,
    stream_output: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    print(command_text(command))
    if dry_run:
        return None
    if not stream_output:
        return subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    tail: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip("\n"))
        if len(tail) > TAIL_LINE_COUNT:
            tail = tail[-TAIL_LINE_COUNT:]
    returncode = process.wait()
    return subprocess.CompletedProcess(command, returncode, "\n".join(tail))


def build_child_env(base_env: dict[str, str], cwd: Path) -> dict[str, str]:
    env = base_env.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    candidates = [cwd]
    if cwd.name.lower() == "freqtrade":
        candidates.append(cwd.parent)
    existing = env.get("PYTHONPATH", "")
    extra = os.pathsep.join(str(path) for path in candidates)
    env["PYTHONPATH"] = extra + (os.pathsep + existing if existing else "")
    return env


def load_hyperopt_epochs(result_file: Path) -> list[dict[str, Any]]:
    epochs: list[dict[str, Any]] = []
    with result_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                epochs.append(value)
    return epochs


def best_epoch_from_file(result_file: Path) -> tuple[dict[str, Any], int]:
    epochs = load_hyperopt_epochs(result_file)
    if not epochs:
        raise ValueError(f"No epochs found in {result_file}")
    valid_epochs = [epoch for epoch in epochs if isinstance(epoch.get("loss"), (int, float))]
    if not valid_epochs:
        raise ValueError(f"No epoch with numeric loss found in {result_file}")
    return min(valid_epochs, key=lambda epoch: float(epoch["loss"])), len(epochs)


def find_loss(result: dict[str, Any]) -> float | None:
    for key in ("loss", "loss_function", "objective"):
        value = result.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    for container_key in ("results_metrics", "results", "metrics"):
        container = result.get(container_key)
        if not isinstance(container, dict):
            continue
        for key in ("loss", "loss_function", "objective"):
            value = container.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def params_from_result(result: dict[str, Any]) -> dict[str, Any]:
    params = result.get("params")
    if isinstance(params, dict):
        return params
    nested = result.get("params_details")
    if isinstance(nested, dict):
        return nested
    return {}


def filtered_params(best_params: dict[str, Any], group_params: set[str]) -> dict[str, dict[str, Any]]:
    accepted_spaces = ("buy", "sell", "stoploss", "roi", "trailing", "protection", "max_open_trades")
    filtered: dict[str, dict[str, Any]] = {}
    for space in accepted_spaces:
        values = best_params.get(space)
        if not isinstance(values, dict):
            continue
        kept = {key: value for key, value in values.items() if key in group_params or key in ("stoploss", "max_open_trades")}
        if kept:
            filtered[space] = kept
    return filtered


def flatten_params(params: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    flat: dict[str, dict[str, Any]] = {}
    for space, values in params.items():
        if not isinstance(values, dict):
            continue
        for name, value in values.items():
            flat[name] = {"space": space, "value": value}
    return flat


def current_strategy_values(strategy_param_file: Path) -> dict[str, dict[str, Any]]:
    data = load_json(strategy_param_file, {})
    params = data.get("params", {})
    return flatten_params(params if isinstance(params, dict) else {})


def value_delta(previous: Any, new: Any) -> dict[str, Any]:
    changed = previous != new
    delta: dict[str, Any] = {
        "previous": previous,
        "new": new,
        "changed": changed,
        "direction": "same",
    }
    if isinstance(previous, bool) or isinstance(new, bool):
        delta["direction"] = "toggled" if changed else "same"
        return delta
    if isinstance(previous, (int, float)) and isinstance(new, (int, float)):
        numeric_delta = float(new) - float(previous)
        delta["delta"] = numeric_delta
        if float(previous) != 0.0:
            delta["pct_delta"] = numeric_delta / abs(float(previous))
        if numeric_delta > 0:
            delta["direction"] = "up"
        elif numeric_delta < 0:
            delta["direction"] = "down"
        return delta
    if changed:
        delta["direction"] = "changed"
    return delta


def describe_param_changes(
    accepted: dict[str, dict[str, Any]],
    current_values: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for space, values in accepted.items():
        for name, new in values.items():
            current = current_values.get(name, {})
            previous = current.get("value")
            delta = value_delta(previous, new)
            changes.append(
                {
                    "param": name,
                    "space": space,
                    **delta,
                }
            )
    return changes


def merge_params_into_snapshot(snapshot: dict[str, Any], params: dict[str, dict[str, Any]]) -> dict[str, Any]:
    updated = deepcopy(snapshot)
    strategy_params = updated.setdefault("params", {})
    for space, values in params.items():
        strategy_params.setdefault(space, {})
        strategy_params[space].update(values)
    updated["export_time"] = datetime.now().astimezone().isoformat()
    return updated


def recursive_metric_candidates(value: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if isinstance(value, dict):
        score = sum(1 for key in ("profit_total", "profit_total_abs", "profit_total_pct", "total_trades", "wins", "losses", "max_drawdown_account", "max_relative_drawdown", "profit_factor") if isinstance(value.get(key), (int, float)))
        if score >= 2:
            candidates.append(value)
        for nested in value.values():
            candidates.extend(recursive_metric_candidates(nested))
    elif isinstance(value, list):
        for nested in value:
            candidates.extend(recursive_metric_candidates(nested))
    return candidates


def read_json_candidates_from_zip(path: Path) -> list[Any]:
    values: list[Any] = []
    with zipfile.ZipFile(path, "r") as handle:
        for name in handle.namelist():
            if not name.lower().endswith(".json"):
                continue
            try:
                payload = json.loads(handle.read(name).decode("utf-8"))
            except Exception:
                continue
            values.append(payload)
    return values


def parse_backtest_metrics(result_file: Path) -> dict[str, Any]:
    payloads: list[Any] = []
    if result_file.suffix.lower() == ".zip":
        payloads.extend(read_json_candidates_from_zip(result_file))
    elif result_file.suffix.lower() == ".json":
        payloads.append(load_json(result_file, {}))
    candidates: list[dict[str, Any]] = []
    for payload in payloads:
        candidates.extend(recursive_metric_candidates(payload))
    if not candidates:
        raise ValueError(f"Could not find backtest metrics inside {result_file.name}")
    return max(
        candidates,
        key=lambda item: sum(
            1 for key in ("profit_total", "profit_total_abs", "profit_total_pct", "total_trades", "wins", "losses", "max_drawdown_account", "max_relative_drawdown", "profit_factor")
            if isinstance(item.get(key), (int, float))
        ),
    )


def metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "total_trades",
        "profit_total",
        "profit_total_abs",
        "profit_total_pct",
        "profit_factor",
        "max_drawdown_account",
        "max_relative_drawdown",
        "winrate",
        "wins",
        "draws",
        "losses",
        "backtest_days",
        "starting_balance",
        "dry_run_wallet",
    ]
    return {key: metrics.get(key) for key in keys if key in metrics}


def numeric_metric(metrics: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def segment_score(metrics: dict[str, Any]) -> float:
    profit = numeric_metric(metrics, ("profit_total_abs", "profit_total_pct", "profit_total"))
    drawdown = numeric_metric(metrics, ("max_relative_drawdown", "max_drawdown_account"))
    score = 0.0
    if profit is not None:
        score += profit
    if drawdown is not None:
        score -= 0.35 * drawdown
    return score


def segment_objective(metrics: dict[str, Any]) -> float:
    """Lower-is-better backtest objective mirroring ProfitWinrateModerateRiskHyperOptLoss."""
    starting_balance = numeric_metric(metrics, ("dry_run_wallet", "starting_balance"))
    profit_abs = numeric_metric(metrics, ("profit_total_abs", "profit_total_pct", "profit_total"))
    trade_count = numeric_metric(metrics, ("total_trades",))
    day_count = numeric_metric(metrics, ("backtest_days",))
    winrate = numeric_metric(metrics, ("winrate",))
    relative_drawdown = numeric_metric(metrics, ("max_relative_drawdown", "max_drawdown_account"))

    if starting_balance is None or starting_balance <= 0.0 or profit_abs is None:
        return -segment_score(metrics)

    day_count = max(1.0, float(day_count or 1.0))
    trade_count = float(trade_count or 0.0)
    winrate = float(winrate or 0.0)
    relative_drawdown = float(relative_drawdown or 0.0)

    total_profit_ratio = float(profit_abs) / float(starting_balance)
    avg_trades_per_day = trade_count / day_count

    profit_term = -OBJECTIVE_PROFIT_WEIGHT * total_profit_ratio
    winrate_term = -OBJECTIVE_WINRATE_WEIGHT * winrate

    winrate_floor_gap = max(0.0, OBJECTIVE_MIN_WINRATE - winrate)
    winrate_floor_penalty = OBJECTIVE_WINRATE_BELOW_FLOOR_PENALTY * (winrate_floor_gap**2)

    low_activity_gap = max(0.0, OBJECTIVE_MIN_TRADES_PER_DAY - avg_trades_per_day)
    low_activity_penalty = OBJECTIVE_LOW_ACTIVITY_PENALTY_WEIGHT * (low_activity_gap**2)

    dd_soft_gap = max(0.0, relative_drawdown - OBJECTIVE_DD_SOFT)
    dd_penalty = OBJECTIVE_DD_SOFT_PENALTY_WEIGHT * (dd_soft_gap**2)
    if relative_drawdown > OBJECTIVE_DD_HARD:
        dd_penalty += OBJECTIVE_DD_HARD_STEP_PENALTY

    non_profit_penalty = 0.0
    if profit_abs <= 0.0:
        non_profit_penalty = OBJECTIVE_NON_PROFIT_BASE_PENALTY + (abs(total_profit_ratio) * OBJECTIVE_NON_PROFIT_RATIO_PENALTY)

    loss = profit_term + winrate_term + winrate_floor_penalty + low_activity_penalty + dd_penalty + non_profit_penalty
    if loss > OBJECTIVE_MAX_LOSS:
        return OBJECTIVE_MAX_LOSS
    return float(loss)


def median_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def format_number(value: Any, decimals: int = 4) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.{decimals}f}"
    if value is None:
        return "-"
    return str(value)


def format_metric_line(metrics: dict[str, Any]) -> str:
    return (
        f"trades={format_number(metrics.get('total_trades'), 0)} "
        f"profit_abs={format_number(metrics.get('profit_total_abs'))} "
        f"pf={format_number(metrics.get('profit_factor'))} "
        f"dd={format_number(numeric_metric(metrics, ('max_relative_drawdown', 'max_drawdown_account')))} "
        f"winrate={format_number(metrics.get('winrate'))} "
        f"objective={format_number(segment_objective(metrics))}"
    )


def objective_delta_bucket(value: float) -> int:
    if value < -OBJECTIVE_NOISE_BAND:
        return -1
    if value > OBJECTIVE_NOISE_BAND:
        return 1
    return 0


def objective_bad_positive(value: float) -> bool:
    return float(value) > OBJECTIVE_NOISE_BAND

def describe_segment(segment: dict[str, Any]) -> str:
    return f"{segment.get('name')} [{segment.get('segment_type')}] {segment.get('regime')} {segment.get('timerange')}"


def segment_key(segment: dict[str, Any]) -> str:
    return f"{segment.get('segment_type')}|{segment.get('timerange')}|{segment.get('name')}"

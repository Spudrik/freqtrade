#!/usr/bin/env python3
"""
Random explorer hyperopt runner for HybridRecoveryGridStrategy.

Purpose
-------
- Keep the newer explorer workflow independent from the staged batch runner.
- Search random strategy-catalog families, tags, namespace values, or custom batches.
- Validate challengers across the remaining selected market-state windows.
- Prefer fewer loss-making backtests, but not as a hard lexicographic rule.
  Objective quality still matters enough to overcome a modest loss-count gap.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import os
import random
import re
import shutil
import statistics
import sys
import time
from typing import Any

from hyperopt_explorer_support import (
    PRESET_FILE,
    MARKET_WINDOWS_FILE,
    CUSTOM_BATCH_FILE,
    PARAM_ENV,
    build_backtest_args,
    build_child_env,
    build_hyperopt_args,
    command_text,
    current_strategy_values,
    describe_param_changes,
    describe_segment,
    filtered_params,
    find_loss,
    hyperopt_results_dir,
    backtest_results_dir,
    latest_result_file,
    load_json,
    load_custom_batches,
    batch_matches_strategy,
    marker_latest_file,
    metric_summary,
    numeric_metric,
    params_from_result,
    parse_backtest_metrics,
    result_file_snapshot,
    run_command,
    save_json,
    segment_key,
    segment_objective,
    strategy_parameter_spaces,
    resolve_custom_batch,
    merge_params_into_snapshot,
    best_epoch_from_file,
    token_list,
    load_market_windows_manifest,
)
from hyperopt_tag_catalog import (
    build_param_catalog,
    params_for_tag,
    spaces_for_params as catalog_spaces_for_params,
    tag_namespace,
    tag_value,
)

THIS_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = THIS_DIR.parent / "runtime"
STATE_FILE = RUNTIME_DIR / "hyperopt_explorer_state.json"
METADATA_DIR = RUNTIME_DIR / "explorer_metadata"
LATEST_SUMMARY_FILE = METADATA_DIR / "latest_summary.json"
RUN_HISTORY_DIR = METADATA_DIR / "runs"
RUN_HISTORY_MAX_BYTES = 1_073_741_824
RUN_HISTORY_MAX_FILES = 2000
METADATA_FILE = LATEST_SUMMARY_FILE
STRATEGY_PARAM_FILE = THIS_DIR.parent / "strategies" / "HybridRecoveryGridStrategy_v11.json"

LOSS_WEIGHT_MIN = 50.0
LOSS_COUNT_WEIGHT_MULT = 0.50
LOSS_FREE_PCT = 20.0
LOSS_MODERATE_PCT = 50.0
LOSS_SEVERE_PCT = 80.0
LOSS_SEVERITY_WEIGHT_MULT = 0.08
LOSS_MODERATE_EXTRA_WEIGHT_MULT = 0.12
LOSS_SEVERE_EXTRA_WEIGHT_MULT = 0.25
ACCEPTANCE_MIN_PROFIT_RETENTION_SAME_OR_MORE_LOSSES = 0.50
ACCEPTANCE_MIN_TRADE_RETENTION_SAME_OR_MORE_LOSSES = 0.35
MARKET_TYPE_VALUES = ("bull", "bear", "chop", "crash", "crossover")
EXPLORER_STATUS_MARKER = "EXPLORER_STATUS_JSON"


def normalize_market_types(values: list[str] | None) -> set[str]:
    selected = {str(value).lower() for value in token_list(" ".join(values or [])) if str(value).lower() in MARKET_TYPE_VALUES}
    if not selected or len(selected) == len(MARKET_TYPE_VALUES):
        return set(MARKET_TYPE_VALUES)
    return selected


def normalize_segment(window: dict[str, Any]) -> dict[str, Any]:
    """Normalize a manifest market segment without assuming fixed 3-month windows."""
    name = str(window.get("name") or "")
    timerange = str(window.get("timerange") or "")
    regime = str(window.get("regime") or window.get("market_state") or "")
    segment_type = str(window.get("segment_type") or regime or "market_state")
    length_months = window.get("segment_length_months", window.get("length_months"))
    try:
        length_months = float(length_months) if length_months is not None else None
    except (TypeError, ValueError):
        length_months = None
    source_names = window.get("source_names")
    if not isinstance(source_names, list) or not source_names:
        source_names = [name] if name else []
    normalized = dict(window)
    normalized.update(
        {
            "name": name,
            "regime": regime,
            "timerange": timerange,
            "segment_type": segment_type,
            "segment_length_months": length_months,
            "source_names": [str(item) for item in source_names if str(item)],
        }
    )
    return normalized


def status_segment_label(segment: dict[str, Any] | None) -> str:
    if not isinstance(segment, dict):
        return "-"
    return str(segment.get("name") or segment.get("timerange") or "-")


def emit_explorer_status(event: str, payload: dict[str, Any]) -> None:
    record = {"event": event, **payload}
    print(f"{EXPLORER_STATUS_MARKER} {json.dumps(record, sort_keys=True, default=str)}", flush=True)



def parse_backtest_count_requests(raw: str | None) -> dict[str, int]:
    requests = {regime: 0 for regime in MARKET_TYPE_VALUES}
    if not raw:
        return requests
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return requests
    if not isinstance(data, dict):
        return requests
    for regime in MARKET_TYPE_VALUES:
        try:
            requests[regime] = max(0, int(data.get(regime) or 0))
        except (TypeError, ValueError):
            requests[regime] = 0
    return requests


def normalize_holdout_window(window: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_segment(window)
    normalized["segment_type"] = "holdout_12m"
    normalized["regime"] = "holdout"
    normalized["label"] = str(window.get("label") or window.get("timerange") or window.get("name") or "")
    for key in ("market_sequence", "event_note", "btc_low_note", "btc_high_note"):
        normalized[key] = str(window.get(key) or "")
    return normalized


def parse_holdout_windows(raw: str | None) -> list[dict[str, Any]]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    holdouts: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            normalized = normalize_holdout_window(item)
            if normalized.get("timerange"):
                holdouts.append(normalized)
    return holdouts


def dedupe_segments(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for window in windows:
        key = segment_key(window) or str(window.get("name") or window.get("timerange") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(window)
    return deduped


def select_normal_backtest_windows(
    windows: list[dict[str, Any]],
    count_requests: dict[str, int],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, list[dict[str, Any]]], list[str]]:
    by_regime: dict[str, list[dict[str, Any]]] = {regime: [] for regime in MARKET_TYPE_VALUES}
    for window in windows:
        regime = str(window.get("regime") or window.get("market_state") or "").lower()
        if regime in by_regime:
            by_regime[regime].append(deepcopy(window))
    available_by_type = {regime: len(by_regime[regime]) for regime in MARKET_TYPE_VALUES}
    selected_by_type: dict[str, list[dict[str, Any]]] = {regime: [] for regime in MARKET_TYPE_VALUES}
    warnings: list[str] = []
    selected: list[dict[str, Any]] = []
    for regime in MARKET_TYPE_VALUES:
        requested = max(0, int(count_requests.get(regime) or 0))
        if requested <= 0:
            continue
        bucket = by_regime[regime]
        if requested > len(bucket):
            warnings.append(f"Requested {requested} {regime} backtest windows, but only {len(bucket)} were available. Using {len(bucket)}.")
        rng.shuffle(bucket)
        picked = bucket[: min(requested, len(bucket))]
        selected_by_type[regime] = picked
        selected.extend(picked)
    return dedupe_segments(selected), available_by_type, selected_by_type, warnings



def weighted_name_sample(
    names: list[str],
    count: int,
    usage_counts: dict[str, int],
    rng: random.Random,
) -> list[str]:
    available = list(dict.fromkeys(str(name) for name in names if str(name)))
    if count <= 0 or count >= len(available):
        rng.shuffle(available)
        return available

    selected: list[str] = []
    while available and len(selected) < count:
        weights = [1.0 / (1.0 + max(0, int(usage_counts.get(name, 0)))) for name in available]
        total = sum(weights)
        if total <= 0.0:
            pick_index = rng.randrange(len(available))
        else:
            pick_index = rng.choices(range(len(available)), weights=weights, k=1)[0]
        selected.append(available.pop(pick_index))
    return selected



def tag_matches_filters(tag: str, filter_tokens: list[str]) -> bool:
    if not filter_tokens:
        return True
    lowered = tag.lower()
    return any(token.lower() in lowered for token in filter_tokens)



def explorer_spaces_for_params(catalog: dict[str, Any], group_params: set[str], preset: dict[str, Any]) -> list[str]:
    spaces = catalog_spaces_for_params(catalog, group_params)
    if spaces:
        return spaces
    return token_list(str(preset.get("hyperopt_spaces") or "default")) or ["buy", "sell"]


def active_catalog_params_for_tag(catalog: dict[str, Any], tag: str, strategy_spaces: dict[str, str]) -> set[str]:
    return {param for param in params_for_tag(catalog, tag) if param in strategy_spaces}


def tag_padding_priority(primary_tag: str, candidate_tag: str, current_params: set[str], primary_params: set[str], tag_params: set[str]) -> tuple[int, int, int, int, int]:
    namespace_rank = {
        "domain": 7,
        "action": 6,
        "family": 5,
        "mode": 5,
        "signal": 4,
        "role": 3,
        "regime": 2,
        "controls": 1,
        "switch": 0,
    }
    new_params = tag_params - current_params
    return (
        1 if tag_params & primary_params else 0,
        1 if tag_namespace(candidate_tag) == tag_namespace(primary_tag) else 0,
        namespace_rank.get(tag_namespace(candidate_tag), 0),
        len(new_params),
        len(tag_params),
    )


def pad_tag_target_to_min_params(
    target: dict[str, Any],
    *,
    tag_index: dict[str, Any],
    strategy_spaces: dict[str, str],
    min_param_count: int,
    rng: random.Random,
) -> dict[str, Any]:
    primary_tag = str(target.get("tag") or target.get("selection_label") or "")
    primary_params = {
        str(param)
        for param in (target.get("group_params") or [])
        if str(param) and str(param) in strategy_spaces
    }
    current_params = set(primary_params)
    target["primary_group_params"] = set(primary_params)
    target["supplemental_tags"] = []
    target["supplemental_params"] = []
    target["min_param_count_per_hyper_run"] = int(max(0, min_param_count))
    target["primary_param_count"] = len(primary_params)

    if min_param_count <= 0 or len(current_params) >= min_param_count:
        target["group_params"] = current_params
        target["active_param_count"] = len(current_params)
        return target

    candidates: list[tuple[tuple[int, int, int, int, int], float, str, set[str]]] = []
    for candidate_tag in sorted(str(tag) for tag in tag_index.keys() if str(tag) and str(tag) != primary_tag):
        candidate_params = active_catalog_params_for_tag({"tag_index": tag_index}, candidate_tag, strategy_spaces)
        if not (candidate_params - current_params):
            continue
        priority = tag_padding_priority(primary_tag, candidate_tag, current_params, primary_params, candidate_params)
        candidates.append((priority, rng.random(), candidate_tag, candidate_params))

    candidates.sort(reverse=True)
    supplemental_tags: list[str] = []
    for _priority, _tie_breaker, candidate_tag, candidate_params in candidates:
        new_params = candidate_params - current_params
        if not new_params:
            continue
        current_params.update(candidate_params)
        supplemental_tags.append(candidate_tag)
        if len(current_params) >= min_param_count:
            break

    target["group_params"] = current_params
    target["supplemental_tags"] = supplemental_tags
    target["supplemental_params"] = sorted(current_params - primary_params)
    target["active_param_count"] = len(current_params)
    return target


def pad_family_target_to_min_params(
    target: dict[str, Any],
    *,
    groups: dict[str, Any],
    strategy_spaces: dict[str, str],
    min_param_count: int,
    rng: random.Random,
) -> dict[str, Any]:
    primary_family = str(target.get("family") or target.get("selection_label") or "")
    primary_params = {
        str(param)
        for param in (target.get("group_params") or [])
        if str(param) and str(param) in strategy_spaces
    }
    current_params = set(primary_params)
    target["primary_group_params"] = set(primary_params)
    target["supplemental_families"] = []
    target["supplemental_params"] = []
    target["min_param_count_per_hyper_run"] = int(max(0, min_param_count))
    target["primary_param_count"] = len(primary_params)

    if min_param_count <= 0 or len(current_params) >= min_param_count:
        target["group_params"] = current_params
        target["active_param_count"] = len(current_params)
        return target

    primary_regimes = {
        str(regime)
        for regime in ((groups.get(primary_family) or {}).get("regimes") or [])
        if str(regime)
    }
    candidates: list[tuple[tuple[int, int, int], float, str, set[str]]] = []
    for family_name, family in sorted(groups.items()):
        family_name = str(family_name)
        if family_name == primary_family or not isinstance(family, dict):
            continue
        family_params = {
            str(param)
            for param in (family.get("params") or [])
            if str(param) and str(param) in strategy_spaces
        }
        new_params = family_params - current_params
        if not new_params:
            continue
        family_regimes = {str(regime) for regime in (family.get("regimes") or []) if str(regime)}
        priority = (
            1 if primary_regimes and family_regimes and bool(primary_regimes & family_regimes) else 0,
            len(new_params),
            len(family_params),
        )
        candidates.append((priority, rng.random(), family_name, family_params))

    candidates.sort(reverse=True)
    supplemental_families: list[str] = []
    for _priority, _tie_breaker, family_name, family_params in candidates:
        new_params = family_params - current_params
        if not new_params:
            continue
        current_params.update(family_params)
        supplemental_families.append(family_name)
        if len(current_params) >= min_param_count:
            break

    target["group_params"] = current_params
    target["supplemental_families"] = supplemental_families
    target["supplemental_params"] = sorted(current_params - primary_params)
    target["active_param_count"] = len(current_params)
    return target


def parse_custom_batch_ids(value: str | None) -> list[str]:
    if not value:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass
    return [item.strip() for item in re.split(r"[,;\s]+", text) if item.strip()]



def build_selection_targets(
    *,
    args: argparse.Namespace,
    rng: random.Random,
    plan: dict[str, Any],
    preset: dict[str, Any],
    strategy_spaces: dict[str, str],
    catalog: dict[str, Any],
    state: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    family_index = catalog.get("family_index") or {}
    family_names = [str(name) for name in family_index.keys()]
    tag_index = catalog.get("tag_index") or {}
    namespace_index = catalog.get("namespace_index") or {}

    usage_counts = {
        "families": {str(key): int(value) for key, value in (state.get("family_usage_counts") or {}).items()},
        "tags": {str(key): int(value) for key, value in (state.get("tag_usage_counts") or {}).items()},
        "custom_batches": {str(key): int(value) for key, value in (state.get("custom_batch_usage_counts") or {}).items()},
    }

    selection_mode = str(args.selection_mode or "random_families")
    targets: list[dict[str, Any]] = []
    selection_info: dict[str, Any] = {
        "selection_mode": selection_mode,
        "selected_families": [],
        "selected_tags": [],
        "selected_namespace_tags": [],
        "target_namespace": str(getattr(args, "target_namespace", "") or ""),
        "tag_filter_tokens": token_list(args.tag_filter),
    }

    try:
        family_count = int(str(args.family_count).strip() or 0)
    except (TypeError, ValueError):
        family_count = 0
    try:
        tag_count = int(str(args.tag_count).strip() or 0)
    except (TypeError, ValueError):
        tag_count = 0
    try:
        min_param_count = max(0, int(str(args.min_param_count_per_hyper_run).strip() or 0))
    except (TypeError, ValueError):
        min_param_count = 0
    selection_info["min_param_count_per_hyper_run"] = min_param_count

    if selection_mode == "custom_batches":
        batch_ids = parse_custom_batch_ids(getattr(args, "custom_batch_ids", ""))
        if not batch_ids:
            raise SystemExit("Custom batches mode needs at least one selected custom batch.")
        custom_batches_file = Path(getattr(args, "custom_batches_file", CUSTOM_BATCH_FILE))
        payload = load_custom_batches(custom_batches_file)
        batches_by_id = {
            str(batch.get("id")): batch
            for batch in payload.get("batches") or []
            if isinstance(batch, dict)
            and str(batch.get("id") or "")
            and batch_matches_strategy(batch, catalog)
        }
        missing_ids = [batch_id for batch_id in batch_ids if batch_id not in batches_by_id]
        if missing_ids:
            raise SystemExit("Custom batch IDs not found for current strategy: " + ", ".join(missing_ids))
        selected_batches: list[dict[str, Any]] = []
        for batch_id in batch_ids:
            batch = batches_by_id[batch_id]
            resolution = resolve_custom_batch(batch, catalog, strategy_spaces)
            if not resolution.get("valid"):
                raise SystemExit(f"Custom batch '{batch.get('name') or batch_id}' has no valid resolved HyperOpt params.")
            group_params = {str(param) for param in resolution.get("params") or [] if str(param) in strategy_spaces}
            if not group_params:
                raise SystemExit(f"Custom batch '{batch.get('name') or batch_id}' resolved no active strategy-space params.")
            target = {
                "selection_type": "custom_batch",
                "selection_mode": selection_mode,
                "selection_label": str(batch.get("name") or batch_id),
                "namespace": None,
                "namespace_value": None,
                "batch_id": batch_id,
                "batch_name": str(batch.get("name") or batch_id),
                "batch_description": str(batch.get("description") or ""),
                "family": None,
                "tag": None,
                "group_params": group_params,
                "primary_group_params": set(group_params),
                "primary_param_count": len(group_params),
                "active_param_count": len(group_params),
                "min_param_count_per_hyper_run": 0,
                "supplemental_tags": [],
                "supplemental_families": [],
                "supplemental_params": [],
                "excluded_params": list(resolution.get("excluded_params") or []),
                "stale_sources": list(resolution.get("stale_sources") or []),
                "stale_params": list(resolution.get("stale_params") or []),
                "stale_excluded_params": list(resolution.get("stale_excluded_params") or []),
                "spaces": list(resolution.get("spaces") or explorer_spaces_for_params(catalog, group_params, preset)),
            }
            usage_counts["custom_batches"][batch_id] = usage_counts["custom_batches"].get(batch_id, 0) + 1
            targets.append(target)
            selected_batches.append(
                {
                    "id": batch_id,
                    "name": target["batch_name"],
                    "active_param_count": target["active_param_count"],
                    "spaces": target["spaces"],
                    "group_params": sorted(group_params),
                    "excluded_params": target["excluded_params"],
                    "stale_sources": target["stale_sources"],
                    "stale_params": target["stale_params"],
                    "stale_excluded_params": target["stale_excluded_params"],
                }
            )
        selection_info["custom_batches_file"] = str(custom_batches_file)
        selection_info["selected_custom_batches"] = selected_batches
    elif selection_mode == "random_tags":
        tag_filter_tokens = token_list(args.tag_filter)
        candidate_tags = [str(tag) for tag in tag_index.keys() if tag_matches_filters(str(tag), tag_filter_tokens)]
        if not candidate_tags:
            raise SystemExit("No strategy tags matched --tag-filter.")
        chosen_tags = weighted_name_sample(candidate_tags, tag_count, usage_counts["tags"], rng)
        for tag in chosen_tags:
            usage_counts["tags"][tag] = usage_counts["tags"].get(tag, 0) + 1
            group_params = active_catalog_params_for_tag(catalog, tag, strategy_spaces)
            if not group_params:
                continue
            target = pad_tag_target_to_min_params(
                {
                    "selection_type": "tag",
                    "selection_mode": selection_mode,
                    "selection_label": tag,
                    "namespace": tag_namespace(tag) or None,
                    "namespace_value": tag_value(tag),
                    "family": None,
                    "tag": tag,
                    "group_params": group_params,
                },
                tag_index=tag_index,
                strategy_spaces=strategy_spaces,
                min_param_count=min_param_count,
                rng=rng,
            )
            target["spaces"] = explorer_spaces_for_params(catalog, set(target.get("group_params") or []), preset)
            targets.append(target)
        selection_info["selected_tags"] = list(chosen_tags)
    elif selection_mode == "random_namespace":
        target_namespace = str(getattr(args, "target_namespace", "") or "").strip()
        if not target_namespace:
            raise SystemExit("--target-namespace is required when --selection-mode random_namespace.")
        namespace_tags = namespace_index.get(target_namespace) or {}
        tag_filter_tokens = token_list(args.tag_filter)
        candidate_tags = [str(tag) for tag in namespace_tags.keys() if tag_matches_filters(str(tag), tag_filter_tokens)]
        if not candidate_tags:
            raise SystemExit(f"No strategy tags matched namespace '{target_namespace}'.")
        chosen_tags = weighted_name_sample(candidate_tags, tag_count, usage_counts["tags"], rng)
        for tag in chosen_tags:
            usage_counts["tags"][tag] = usage_counts["tags"].get(tag, 0) + 1
            group_params = active_catalog_params_for_tag(catalog, tag, strategy_spaces)
            if not group_params:
                continue
            target = pad_tag_target_to_min_params(
                {
                    "selection_type": "namespace_tag",
                    "selection_mode": selection_mode,
                    "selection_label": tag,
                    "namespace": target_namespace,
                    "namespace_value": tag_value(tag),
                    "family": None,
                    "tag": tag,
                    "group_params": group_params,
                },
                tag_index=tag_index,
                strategy_spaces=strategy_spaces,
                min_param_count=min_param_count,
                rng=rng,
            )
            target["spaces"] = explorer_spaces_for_params(catalog, set(target.get("group_params") or []), preset)
            targets.append(target)
        selection_info["selected_namespace_tags"] = list(chosen_tags)
    else:
        if not family_names:
            raise SystemExit("No family tags found in strategy catalog.")
        chosen_families = weighted_name_sample(family_names, family_count, usage_counts["families"], rng)
        catalog_family_groups = {
            family_name: {"params": list(params)}
            for family_name, params in family_index.items()
        }
        for family_name in chosen_families:
            usage_counts["families"][family_name] = usage_counts["families"].get(family_name, 0) + 1
            group_params = {str(param) for param in family_index.get(family_name, []) if str(param)}
            if not group_params:
                continue
            target = pad_family_target_to_min_params(
                {
                    "selection_type": "family",
                    "selection_mode": selection_mode,
                    "selection_label": f"family:{family_name}",
                    "namespace": "family",
                    "namespace_value": family_name,
                    "family": family_name,
                    "tag": f"family:{family_name}",
                    "group_params": group_params,
                    "supplemental_tags": [],
                },
                groups=catalog_family_groups,
                strategy_spaces=strategy_spaces,
                min_param_count=min_param_count,
                rng=rng,
            )
            target["spaces"] = explorer_spaces_for_params(catalog, set(target.get("group_params") or []), preset)
            targets.append(target)
        selection_info["selected_families"] = list(chosen_families)

    state["family_usage_counts"] = usage_counts["families"]
    state["tag_usage_counts"] = usage_counts["tags"]
    state["custom_batch_usage_counts"] = usage_counts["custom_batches"]
    return targets, selection_info



def validation_penalty_unit(*metric_sets: dict[str, dict[str, Any]]) -> float:
    objective_magnitudes: list[float] = []
    for metrics_by_segment in metric_sets:
        for metrics in metrics_by_segment.values():
            objective_magnitudes.append(abs(float(segment_objective(metrics))))
    return max(LOSS_WEIGHT_MIN, float(statistics.median(objective_magnitudes)) if objective_magnitudes else LOSS_WEIGHT_MIN)


def profit_pct_of_wallet(metrics: dict[str, Any]) -> float:
    profit_abs = float(numeric_metric(metrics, ("profit_total_abs",)) or 0.0)
    wallet = float(numeric_metric(metrics, ("starting_balance", "dry_run_wallet")) or 0.0)
    if wallet <= 0.0:
        return 0.0
    return 100.0 * profit_abs / wallet


def loss_severity_penalty(metrics: dict[str, Any], penalty_unit: float) -> float:
    loss_pct = max(0.0, -profit_pct_of_wallet(metrics))
    if loss_pct <= LOSS_FREE_PCT:
        return 0.0

    penalty = (loss_pct - LOSS_FREE_PCT) * penalty_unit * LOSS_SEVERITY_WEIGHT_MULT

    if loss_pct > LOSS_MODERATE_PCT:
        penalty += (loss_pct - LOSS_MODERATE_PCT) * penalty_unit * LOSS_MODERATE_EXTRA_WEIGHT_MULT

    if loss_pct > LOSS_SEVERE_PCT:
        penalty += (loss_pct - LOSS_SEVERE_PCT) * penalty_unit * LOSS_SEVERE_EXTRA_WEIGHT_MULT

    return float(penalty)


def validation_rank(metrics_by_segment: dict[str, dict[str, Any]], penalty_unit: float | None = None) -> dict[str, float | int]:
    loss_count = 0
    objective_total = 0.0
    profit_total = 0.0
    trade_total = 0.0
    worst_loss_pct = 0.0
    objective_magnitudes: list[float] = []
    segment_metrics = list(metrics_by_segment.values())
    for metrics in segment_metrics:
        objective = float(segment_objective(metrics))
        profit_abs = float(numeric_metric(metrics, ("profit_total_abs", "profit_total_pct", "profit_total")) or 0.0)
        trade_count = float(numeric_metric(metrics, ("total_trades",)) or 0.0)
        if profit_abs <= 0.0:
            loss_count += 1
        profit_total += profit_abs
        trade_total += trade_count
        objective_total += objective
        objective_magnitudes.append(abs(objective))
        worst_loss_pct = max(worst_loss_pct, max(0.0, -profit_pct_of_wallet(metrics)))

    penalty_unit = float(penalty_unit) if penalty_unit is not None else max(LOSS_WEIGHT_MIN, float(statistics.median(objective_magnitudes)) if objective_magnitudes else LOSS_WEIGHT_MIN)
    loss_count_penalty_total = float(loss_count) * penalty_unit * LOSS_COUNT_WEIGHT_MULT
    loss_severity_penalty_total = float(sum(loss_severity_penalty(metrics, penalty_unit) for metrics in segment_metrics))
    weighted_score = objective_total + loss_count_penalty_total + loss_severity_penalty_total
    return {
        "loss_window_count": int(loss_count),
        "objective_total": float(objective_total),
        "profit_total": float(profit_total),
        "trade_total": float(trade_total),
        "worst_loss_pct": float(worst_loss_pct),
        "loss_penalty_unit": float(penalty_unit),
        "loss_count_penalty_total": float(loss_count_penalty_total),
        "loss_severity_penalty_total": float(loss_severity_penalty_total),
        "loss_penalty_total": float(loss_count_penalty_total),
        "weighted_acceptance_score": float(weighted_score),
        "weighted_score": float(weighted_score),
    }



def validation_summary(
    champion_metrics: dict[str, dict[str, Any]],
    challenger_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    common_keys = [key for key in champion_metrics if key in challenger_metrics]
    champion_used = {key: champion_metrics[key] for key in common_keys}
    challenger_used = {key: challenger_metrics[key] for key in common_keys}
    shared_penalty_unit = validation_penalty_unit(champion_used, challenger_used)
    champion_rank = validation_rank(champion_used, shared_penalty_unit)
    challenger_rank = validation_rank(challenger_used, shared_penalty_unit)
    accepted_by_score = float(challenger_rank["weighted_acceptance_score"]) < float(champion_rank["weighted_acceptance_score"])
    objective_delta = float(challenger_rank["objective_total"]) - float(champion_rank["objective_total"])
    profit_total_delta = float(challenger_rank["profit_total"]) - float(champion_rank["profit_total"])
    loss_window_count_delta = int(challenger_rank["loss_window_count"]) - int(champion_rank["loss_window_count"])
    weighted_score_delta = float(challenger_rank["weighted_acceptance_score"]) - float(champion_rank["weighted_acceptance_score"])
    loss_count_penalty_delta = float(challenger_rank["loss_count_penalty_total"]) - float(champion_rank["loss_count_penalty_total"])
    loss_severity_penalty_delta = float(challenger_rank["loss_severity_penalty_total"]) - float(champion_rank["loss_severity_penalty_total"])
    worst_loss_pct_delta = float(challenger_rank["worst_loss_pct"]) - float(champion_rank["worst_loss_pct"])
    guard_reason = ""
    if accepted_by_score and loss_window_count_delta >= 0:
        champion_profit = float(champion_rank["profit_total"])
        challenger_profit = float(challenger_rank["profit_total"])
        champion_trades = float(champion_rank["trade_total"])
        challenger_trades = float(challenger_rank["trade_total"])
        min_profit = champion_profit * ACCEPTANCE_MIN_PROFIT_RETENTION_SAME_OR_MORE_LOSSES
        min_trades = champion_trades * ACCEPTANCE_MIN_TRADE_RETENTION_SAME_OR_MORE_LOSSES
        if champion_profit > 0.0 and challenger_profit < min_profit:
            guard_reason = (
                f"profit_retention_guard: challenger profit {challenger_profit:.4f} "
                f"is below {ACCEPTANCE_MIN_PROFIT_RETENTION_SAME_OR_MORE_LOSSES:.0%} of champion profit {champion_profit:.4f}"
            )
        elif champion_trades > 0.0 and challenger_trades < min_trades:
            guard_reason = (
                f"trade_activity_guard: challenger trades {challenger_trades:.0f} "
                f"is below {ACCEPTANCE_MIN_TRADE_RETENTION_SAME_OR_MORE_LOSSES:.0%} of champion trades {champion_trades:.0f}"
            )
    accepted = bool(accepted_by_score and not guard_reason)
    decision_reason = "weighted_score_improved" if accepted else (guard_reason or "weighted_score_not_improved")
    return {
        "completed_segments": len(common_keys),
        "backtest_window_count": len(common_keys),
        "champion_rank": champion_rank,
        "challenger_rank": challenger_rank,
        "champion_loss_windows": int(champion_rank["loss_window_count"]),
        "challenger_loss_windows": int(challenger_rank["loss_window_count"]),
        "champion_loss_window_count": int(champion_rank["loss_window_count"]),
        "challenger_loss_window_count": int(challenger_rank["loss_window_count"]),
        "champion_objective_total": float(champion_rank["objective_total"]),
        "challenger_objective_total": float(challenger_rank["objective_total"]),
        "champion_profit_total": float(champion_rank["profit_total"]),
        "challenger_profit_total": float(challenger_rank["profit_total"]),
        "champion_trade_total": float(champion_rank["trade_total"]),
        "challenger_trade_total": float(challenger_rank["trade_total"]),
        "champion_worst_loss_pct": float(champion_rank["worst_loss_pct"]),
        "challenger_worst_loss_pct": float(challenger_rank["worst_loss_pct"]),
        "champion_loss_penalty_unit": float(champion_rank["loss_penalty_unit"]),
        "challenger_loss_penalty_unit": float(challenger_rank["loss_penalty_unit"]),
        "champion_loss_count_penalty_total": float(champion_rank["loss_count_penalty_total"]),
        "challenger_loss_count_penalty_total": float(challenger_rank["loss_count_penalty_total"]),
        "champion_loss_severity_penalty_total": float(champion_rank["loss_severity_penalty_total"]),
        "challenger_loss_severity_penalty_total": float(challenger_rank["loss_severity_penalty_total"]),
        "champion_loss_penalty_total": float(champion_rank["loss_penalty_total"]),
        "challenger_loss_penalty_total": float(challenger_rank["loss_penalty_total"]),
        "champion_weighted_acceptance_score": float(champion_rank["weighted_acceptance_score"]),
        "challenger_weighted_acceptance_score": float(challenger_rank["weighted_acceptance_score"]),
        "champion_weighted_score": float(champion_rank["weighted_acceptance_score"]),
        "challenger_weighted_score": float(challenger_rank["weighted_acceptance_score"]),
        "objective_delta": objective_delta,
        "profit_total_delta": profit_total_delta,
        "worst_loss_pct_delta": worst_loss_pct_delta,
        "loss_window_count_delta": loss_window_count_delta,
        "loss_count_penalty_delta": loss_count_penalty_delta,
        "loss_severity_penalty_delta": loss_severity_penalty_delta,
        "weighted_score_delta": weighted_score_delta,
        "accepted_by_score": bool(accepted_by_score),
        "acceptance_guard_reason": guard_reason,
        "decision_reason": decision_reason,
        "accepted": accepted,
        "decision": "ACCEPTED" if accepted else "REJECTED",
    }



def changed_param_count(changes: list[dict[str, Any]] | None) -> int:
    return sum(1 for change in (changes or []) if bool(change.get("changed")))



def metadata_target(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "selection_mode": target.get("selection_mode"),
        "selection_type": target.get("selection_type"),
        "selection_label": target.get("selection_label"),
        "namespace": target.get("namespace"),
        "namespace_value": target.get("namespace_value"),
        "family": target.get("family"),
        "tag": target.get("tag"),
        "batch_id": target.get("batch_id"),
        "batch_name": target.get("batch_name"),
        "group_params": sorted(str(param) for param in (target.get("group_params") or []) if str(param)),
        "primary_group_params": sorted(str(param) for param in (target.get("primary_group_params") or []) if str(param)),
        "primary_param_count": int(target.get("primary_param_count") or 0),
        "active_param_count": int(target.get("active_param_count") or len(target.get("group_params") or [])),
        "min_param_count_per_hyper_run": int(target.get("min_param_count_per_hyper_run") or 0),
        "supplemental_tags": list(target.get("supplemental_tags") or []),
        "supplemental_families": list(target.get("supplemental_families") or []),
        "supplemental_params": sorted(str(param) for param in (target.get("supplemental_params") or []) if str(param)),
        "excluded_params": list(target.get("excluded_params") or []),
        "stale_sources": list(target.get("stale_sources") or []),
        "stale_params": list(target.get("stale_params") or []),
        "stale_excluded_params": list(target.get("stale_excluded_params") or []),
        "spaces": list(target.get("spaces") or []),
    }



def summarize_loop_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    loops = metadata.get("loops", [])
    accepted_loops = [loop for loop in loops if loop.get("loop_applied")]
    return {
        "loop_count": len(loops),
        "search_run_count": len(metadata.get("search_runs", [])),
        "backtest_run_count": len(metadata.get("backtest_runs", [])),
        "validation_run_count": len(metadata.get("validation_runs", [])),
        "accepted_loop_count": len(accepted_loops),
        "family_usage_counts": dict(sorted((metadata.get("family_usage_counts") or {}).items())),
        "tag_usage_counts": dict(sorted((metadata.get("tag_usage_counts") or {}).items())),
        "custom_batch_usage_counts": dict(sorted((metadata.get("custom_batch_usage_counts") or {}).items())),
    }



def atomic_save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def run_history_file(started_at: str, run_id: str) -> Path:
    safe_started = re.sub(r"[^0-9A-Za-z]+", "", started_at)[:20] or datetime.now().strftime("%Y%m%d%H%M%S")
    safe_id = re.sub(r"[^0-9A-Za-z_-]+", "_", run_id).strip("_") or "run"
    return RUN_HISTORY_DIR / f"explorer_run_{safe_started}_{safe_id}.json"


def latest_summary_metadata(metadata: dict[str, Any], run_file: Path) -> dict[str, Any]:
    summary = summarize_loop_metadata(metadata)
    return {
        "schema_version": 1,
        "updated_at": datetime.now().astimezone().isoformat(),
        "started_at": metadata.get("started_at"),
        "finished_at": metadata.get("finished_at"),
        "latest_run_file": str(run_file),
        "preset": metadata.get("preset"),
        "strategy_file": metadata.get("strategy_file"),
        "strategy_class": metadata.get("strategy_class"),
        "selection_mode": metadata.get("selection_mode"),
        "target_namespace": metadata.get("target_namespace"),
        "selected_target_label": metadata.get("selected_target_label"),
        "selected_target_labels": list(metadata.get("selected_target_labels") or []),
        "hyperopt_window": deepcopy(metadata.get("hyperopt_window")),
        "backtest_windows": deepcopy(metadata.get("backtest_windows") or []),
        "summary": summary,
        "last_loop": deepcopy((metadata.get("loops") or [])[-1]) if metadata.get("loops") else None,
    }


def enforce_run_history_retention(current_run_file: Path) -> None:
    RUN_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    files = [path for path in RUN_HISTORY_DIR.glob("explorer_run_*.json") if path.is_file()]
    if not files:
        return
    files.sort(key=lambda path: (path.stat().st_mtime, path.name))
    current_resolved = current_run_file.resolve()
    total_size = sum(path.stat().st_size for path in files)
    current_size = current_run_file.stat().st_size if current_run_file.exists() else 0
    if current_size > RUN_HISTORY_MAX_BYTES:
        print(
            f"Warning: current Explorer run metadata file is larger than retention cap "
            f"({current_size} > {RUN_HISTORY_MAX_BYTES}); keeping current run file.",
            flush=True,
        )
        return

    for path in list(files):
        if len(files) <= RUN_HISTORY_MAX_FILES and total_size <= RUN_HISTORY_MAX_BYTES:
            break
        try:
            if path.resolve() == current_resolved:
                continue
            size = path.stat().st_size
            path.unlink()
            total_size -= size
            files.remove(path)
        except FileNotFoundError:
            files.remove(path)
        except Exception as exc:
            print(f"Warning: could not delete old Explorer run metadata {path}: {exc}", flush=True)


def flush_metadata_checkpoint(metadata: dict[str, Any], summary_file: Path, run_file: Path) -> None:
    metadata["summary"] = summarize_loop_metadata(metadata)
    metadata["last_checkpoint_at"] = datetime.now().astimezone().isoformat()
    atomic_save_json(run_file, metadata)
    enforce_run_history_retention(run_file)
    atomic_save_json(summary_file, latest_summary_metadata(metadata, run_file))


def timerange_day_count(timerange: str) -> int:
    try:
        start_text, end_text = str(timerange).split("-", 1)
        start = datetime.strptime(start_text[:8], "%Y%m%d")
        end = datetime.strptime(end_text[:8], "%Y%m%d")
    except Exception:
        return 1
    return max(1, (end - start).days)


def empty_backtest_metrics(preset: dict[str, Any], segment: dict[str, Any]) -> dict[str, Any]:
    try:
        starting_balance = float(preset.get("dry_run_wallet") or 0.0)
    except (TypeError, ValueError):
        starting_balance = 0.0
    if starting_balance <= 0.0:
        starting_balance = 1000.0
    return {
        "total_trades": 0,
        "profit_total": 0.0,
        "profit_total_abs": 0.0,
        "profit_total_pct": 0.0,
        "profit_factor": 0.0,
        "max_drawdown_account": 0.0,
        "max_relative_drawdown": 0.0,
        "winrate": 0.0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "backtest_days": timerange_day_count(str(segment.get("timerange") or "")),
        "starting_balance": starting_balance,
        "dry_run_wallet": starting_balance,
    }


def tail_text(value: str | None, line_count: int = 80) -> str:
    if not value:
        return ""
    lines = value.splitlines()
    return "\n".join(lines[-line_count:])



def slugify_worker_part(value: Any) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        slug = "item"
    return slug[:80]


def make_explorer_run_temp_root(temp_root: Path) -> Path:
    temp_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    base_name = f"explorer_{timestamp}_{pid}"
    run_root = temp_root / base_name
    suffix = 2
    while run_root.exists():
        run_root = temp_root / f"{base_name}_{suffix}"
        suffix += 1
    run_root.mkdir(parents=True, exist_ok=False)
    return run_root


def prepare_isolated_backtest_preset(
    base_preset: dict[str, Any],
    *,
    run_temp_root: Path,
    role: str,
    loop_index: int,
    candidate_id: str,
    segment: dict[str, Any],
    snapshot: dict[str, Any],
    strategy_path: Path,
) -> tuple[dict[str, Any], Path]:
    candidate_slug = slugify_worker_part(candidate_id)
    segment_source = segment.get("name") or segment.get("timerange") or "segment"
    segment_slug = slugify_worker_part(segment_source)
    job_id = f"loop{int(loop_index):03d}__{slugify_worker_part(role)}__{candidate_slug}__{segment_slug}"
    if role == "champion":
        job_root = run_temp_root / "champion" / job_id
    else:
        job_root = run_temp_root / "challenger" / candidate_slug / job_id

    strategy_dir = job_root / "strategy"
    results_dir = job_root / "results"
    logs_dir = job_root / "logs"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    job_strategy_file = strategy_dir / strategy_path.name
    shutil.copy2(strategy_path, job_strategy_file)
    save_json(strategy_dir / f"{strategy_path.stem}.json", snapshot)

    isolated_preset = deepcopy(base_preset)
    isolated_preset["strategy_file"] = str(job_strategy_file)
    isolated_preset["backtest_directory"] = str(results_dir)
    isolated_preset["stdout_log_file"] = str(logs_dir / "stdout.txt")
    return isolated_preset, job_root


def run_one_validation_backtest_job(
    *,
    base_preset: dict[str, Any],
    cwd: Path,
    base_child_env: dict[str, str],
    run_temp_root: Path,
    role: str,
    loop_index: int,
    candidate_id: str,
    segment: dict[str, Any],
    snapshot: dict[str, Any],
    strategy_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    job_root: Path | None = None
    isolated_preset: dict[str, Any] | None = None
    try:
        isolated_preset, job_root = prepare_isolated_backtest_preset(
            base_preset,
            run_temp_root=run_temp_root,
            role=role,
            loop_index=loop_index,
            candidate_id=candidate_id,
            segment=segment,
            snapshot=snapshot,
            strategy_path=strategy_path,
        )
        record = backtest_segment(isolated_preset, cwd, base_child_env.copy(), segment, dry_run)
        record["loop_index"] = loop_index
        record["candidate_id"] = candidate_id
        record["role"] = role
        record["job_root"] = str(job_root)
        record["isolated_strategy_file"] = str(isolated_preset.get("strategy_file") or "")
        record["results_dir"] = str(isolated_preset.get("backtest_directory") or "")
        return record
    except Exception as exc:
        return {
            "segment": dict(segment),
            "status": "backtest_exception",
            "exception": repr(exc),
            "loop_index": loop_index,
            "candidate_id": candidate_id,
            "role": role,
            "command": "",
            "job_root": str(job_root) if job_root is not None else "",
            "isolated_strategy_file": str((isolated_preset or {}).get("strategy_file") or ""),
            "results_dir": str((isolated_preset or {}).get("backtest_directory") or ""),
        }


def run_validation_backtests(
    *,
    base_preset: dict[str, Any],
    cwd: Path,
    base_child_env: dict[str, str],
    run_temp_root: Path,
    role: str,
    loop_index: int,
    candidate_id: str,
    snapshot: dict[str, Any],
    validation_segments: list[dict[str, Any]],
    strategy_path: Path,
    dry_run: bool,
    max_workers: int,
    champion_cache_get: Any = None,
    champion_cache_put: Any = None,
) -> list[dict[str, Any]]:
    print(f"Starting {role} validation backtests: windows={len(validation_segments)} workers={max_workers}")
    records_by_index: dict[int, dict[str, Any]] = {}
    scheduled: list[tuple[int, dict[str, Any]]] = []

    for index, segment in enumerate(validation_segments):
        cached_record = champion_cache_get(segment) if champion_cache_get else None
        if cached_record is not None:
            record = deepcopy(cached_record)
            record["cache_hit"] = True
            records_by_index[index] = record
            continue
        segment_desc = describe_segment(segment)
        if role == "champion":
            segment_label = str(segment.get("name") or segment.get("timerange") or "segment")
            print(f"Champion baseline backtest executed: {segment_label}")
            print(f"Starting champion baseline backtest on {segment_desc}")
        else:
            print(f"Starting backtest with challenger {candidate_id} on {segment_desc}")
        print(f"Queued {role} backtest: {candidate_id} on {segment_desc}")
        scheduled.append((index, segment))

    if max_workers <= 1:
        for index, segment in scheduled:
            record = run_one_validation_backtest_job(
                base_preset=base_preset,
                cwd=cwd,
                base_child_env=base_child_env,
                run_temp_root=run_temp_root,
                role=role,
                loop_index=loop_index,
                candidate_id=candidate_id,
                segment=segment,
                snapshot=snapshot,
                strategy_path=strategy_path,
                dry_run=dry_run,
            )
            if champion_cache_put and record.get("status") == "completed":
                champion_cache_put(segment, record)
            segment_label = str(segment.get("name") or segment.get("timerange") or "segment")
            print(f"Completed {role} backtest: {candidate_id} on {segment_label} status={record.get('status')}")
            record.setdefault("cache_hit", False)
            records_by_index[index] = record
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    run_one_validation_backtest_job,
                    base_preset=base_preset,
                    cwd=cwd,
                    base_child_env=base_child_env,
                    run_temp_root=run_temp_root,
                    role=role,
                    loop_index=loop_index,
                    candidate_id=candidate_id,
                    segment=segment,
                    snapshot=snapshot,
                    strategy_path=strategy_path,
                    dry_run=dry_run,
                ): (index, segment)
                for index, segment in scheduled
            }
            for future in as_completed(future_map):
                index, segment = future_map[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = {
                        "segment": dict(segment),
                        "status": "backtest_exception",
                        "exception": repr(exc),
                        "loop_index": loop_index,
                        "candidate_id": candidate_id,
                        "role": role,
                    }
                if champion_cache_put and record.get("status") == "completed":
                    champion_cache_put(segment, record)
                segment_label = str(segment.get("name") or segment.get("timerange") or "segment")
                print(f"Completed {role} backtest: {candidate_id} on {segment_label} status={record.get('status')}")
                record.setdefault("cache_hit", False)
                records_by_index[index] = record

    return [records_by_index[index] for index in range(len(validation_segments))]


def backtest_segment(
    preset: dict[str, Any],
    cwd: Path,
    env: dict[str, str],
    segment: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    timerange = str(segment["timerange"])
    command = [str(preset.get("python_exe") or sys.executable), "-u", "-m", "freqtrade", *build_backtest_args(preset, timerange)]
    if str(preset.get("backtest_directory") or "").strip():
        results_dir = Path(str(preset.get("backtest_directory")))
    else:
        results_dir = backtest_results_dir(preset)
    results_dir.mkdir(parents=True, exist_ok=True)
    previous_result = marker_latest_file(results_dir)
    before_snapshot = result_file_snapshot(results_dir, ("*.zip", "*.json"))
    started_at = time.time()
    result = run_command(command, cwd, env, dry_run, stream_output=False)
    record: dict[str, Any] = {
        "segment": dict(segment),
        "command": command_text(command),
        "results_dir": str(results_dir),
    }
    if dry_run:
        record["status"] = "dry_run"
        return record
    assert result is not None
    stdout_log_file = str(preset.get("stdout_log_file") or "").strip()
    if stdout_log_file:
        try:
            stdout_path = Path(stdout_log_file)
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_path.write_text(result.stdout or "", encoding="utf-8")
        except Exception:
            pass
    record["exit_code"] = result.returncode
    if result.returncode != 0:
        record["status"] = "backtest_failed"
        record["stdout_tail"] = tail_text(result.stdout)
        return record
    result_file = latest_result_file(results_dir, ("*.zip", "*.json"), started_at, previous_result, before_snapshot)
    if not result_file:
        record["status"] = "completed"
        record["result_missing"] = True
        record["metrics"] = metric_summary(empty_backtest_metrics(preset, segment))
        record["stdout_tail"] = tail_text(result.stdout)
        return record
    record["result_file"] = str(result_file)
    try:
        metrics = parse_backtest_metrics(result_file)
    except Exception as exc:
        record["status"] = "result_parse_failed"
        record["error"] = str(exc)
        return record
    record["status"] = "completed"
    record["metrics"] = metric_summary(metrics)
    return record



def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()



def sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return sha256_text(f"missing:{path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def hash_json_value(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))



def preset_config_hash(preset: dict[str, Any]) -> str:
    config_files = []
    for entry in preset.get("config_files", []) or []:
        path = Path(str(entry)).expanduser()
        config_files.append(
            {
                "path": str(path.resolve()) if path.exists() else str(path),
                "hash": sha256_file(path),
            }
        )
    return hash_json_value(config_files)



def preset_pair_list_hash(preset: dict[str, Any]) -> str:
    payload = {
        "pair_mode": str(preset.get("pair_mode") or ""),
        "pairs": token_list(str(preset.get("pairs") or "")),
        "blacklist": token_list(str(preset.get("blacklist") or "")),
    }
    return hash_json_value(payload)



def print_family_catalog(catalog: dict[str, Any]) -> None:
    print("Available families from strategy catalog:")
    for family_name, params in sorted((catalog.get("family_index") or {}).items()):
        param_names = [str(param) for param in params if str(param)]
        print(f"- family:{family_name}: {len(param_names)} params -> {', '.join(param_names)}")



def print_tag_catalog_summary(catalog: dict[str, Any], selected_tags: list[str]) -> None:
    if not selected_tags:
        return
    print("Selected random tags:")
    tag_index = catalog.get("tag_index") or {}
    for tag in selected_tags:
        params = [str(param) for param in tag_index.get(tag, []) if str(param)]
        preview = ", ".join(params[:10])
        if len(params) > 10:
            preview += ", ..."
        print(f"- {tag}: {len(params)} params -> {preview}")


def print_namespace_target_summary(catalog: dict[str, Any], selected_tags: list[str]) -> None:
    if not selected_tags:
        return
    print("Selected namespace targets:")
    tag_index = catalog.get("tag_index") or {}
    params_by_name = catalog.get("params") or {}
    for tag in selected_tags:
        params = [str(param) for param in tag_index.get(tag, []) if str(param)]
        spaces = sorted({str((params_by_name.get(param) or {}).get("space") or "") for param in params if str((params_by_name.get(param) or {}).get("space") or "")})
        print(f"  - {tag} [{len(params)} params | {', '.join(spaces) or '-'}]")


def print_launch_selection_summary(catalog: dict[str, Any], selection_info: dict[str, Any], *, sampling_seed_input: int | None, effective_sampling_seed: int) -> None:
    if sampling_seed_input is None:
        print(f"Sampling seed: random per launch -> {effective_sampling_seed}")
    else:
        print(f"Sampling seed: {effective_sampling_seed}")

    selection_mode = str(selection_info.get("selection_mode") or "")
    if selection_mode == "custom_batches":
        selected_batches = [batch for batch in selection_info.get("selected_custom_batches") or [] if isinstance(batch, dict)]
        print(f"Selected custom batches for this Explorer launch: {len(selected_batches)} batches")
        for batch in selected_batches:
            excluded = ", ".join(batch.get("excluded_params") or []) or "-"
            spaces = ", ".join(batch.get("spaces") or []) or "-"
            print(f"- {batch.get('name')}: {batch.get('active_param_count')} params | spaces={spaces} | excluded={excluded}")
        return
    if selection_mode == "random_tags":
        selected_tags = [str(tag) for tag in selection_info.get("selected_tags") or [] if str(tag)]
        print(f"Selected random tag pool for this Explorer launch: {len(selected_tags)} tags")
        print_tag_catalog_summary(catalog, selected_tags)
        return
    if selection_mode == "random_namespace":
        target_namespace = str(selection_info.get("target_namespace") or "")
        selected_tags = [str(tag) for tag in selection_info.get("selected_namespace_tags") or [] if str(tag)]
        print(f"Selection mode: Random namespace values")
        print(f"Target namespace: {target_namespace}")
        print(f"Selected targets: {len(selected_tags)}")
        print_namespace_target_summary(catalog, selected_tags)
        return

    selected_families = [str(family) for family in selection_info.get("selected_families") or [] if str(family)]
    if selected_families:
        print("Selected random family pool for this Explorer launch: " + ", ".join(f"family:{family}" for family in selected_families))



def print_validation_table(
    segments: list[dict[str, Any]],
    champion_metrics: dict[str, dict[str, Any]],
    challenger_metrics: dict[str, dict[str, Any]],
) -> None:
    print("\nBacktest comparison")
    print("  Window                              Champion Profit   Champion %   Challenger Profit  Challenger %    Champion Obj     Challenger Obj")
    print("  -------------------------------------------------------------------------------------------------------------------------------")
    for segment in segments:
        key = segment_key(segment)
        champion = champion_metrics.get(key)
        challenger = challenger_metrics.get(key)
        if not champion or not challenger:
            continue
        c_profit = float(numeric_metric(champion, ("profit_total_abs", "profit_total_pct", "profit_total")) or 0.0)
        h_profit = float(numeric_metric(challenger, ("profit_total_abs", "profit_total_pct", "profit_total")) or 0.0)
        c_pct = float(profit_pct_of_wallet(champion))
        h_pct = float(profit_pct_of_wallet(challenger))
        c_obj = float(segment_objective(champion))
        h_obj = float(segment_objective(challenger))
        label = str(segment.get("name") or segment.get("timerange") or "segment")
        print(f"  {label[:34]:34} {c_profit:16.4f} {c_pct:11.2f} {h_profit:18.4f} {h_pct:12.2f} {c_obj:15.4f} {h_obj:17.4f}")


def print_acceptance_summary(summary: dict[str, Any]) -> None:
    print(
        f"Acceptance: {summary['decision']} | "
        f"backtest_windows={summary['backtest_window_count']} | "
        f"params_changed={summary['params_changed_count']} | "
        "lower weighted score is better"
    )
    print("  Role          Weighted score    Objective total   Total profit   Worst loss %   Loss windows      Count pen   Severity pen")
    print("  --------------------------------------------------------------------------------------------------------------------------")
    print(
        f"  Champion        {summary['champion_weighted_acceptance_score']:14.4f} "
        f"{summary['champion_objective_total']:16.4f} "
        f"{summary['champion_profit_total']:14.4f} "
        f"{summary['champion_worst_loss_pct']:13.2f} "
        f"{summary['champion_loss_window_count']:14d} "
        f"{summary['champion_loss_count_penalty_total']:14.4f} "
        f"{summary['champion_loss_severity_penalty_total']:14.4f}"
    )
    print(
        f"  Challenger      {summary['challenger_weighted_acceptance_score']:14.4f} "
        f"{summary['challenger_objective_total']:16.4f} "
        f"{summary['challenger_profit_total']:14.4f} "
        f"{summary['challenger_worst_loss_pct']:13.2f} "
        f"{summary['challenger_loss_window_count']:14d} "
        f"{summary['challenger_loss_count_penalty_total']:14.4f} "
        f"{summary['challenger_loss_severity_penalty_total']:14.4f}"
    )
    print(
        f"  Delta           {summary['weighted_score_delta']:14.4f} "
        f"{summary['objective_delta']:16.4f} "
        f"{summary['profit_total_delta']:14.4f} "
        f"{summary['worst_loss_pct_delta']:13.2f} "
        f"{summary['loss_window_count_delta']:14d} "
        f"{summary['loss_count_penalty_delta']:14.4f} "
        f"{summary['loss_severity_penalty_delta']:14.4f}"
    )
    if summary.get("decision_reason") and summary.get("decision_reason") != "weighted_score_improved":
        print(f"  Reason: {summary['decision_reason']}")



def candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, int, float]:
    validation = candidate.get("validation") or {}
    weighted = float(validation.get("challenger_weighted_score") or 0.0)
    loss_count = int(validation.get("challenger_loss_windows") or 0)
    objective_total = float(validation.get("challenger_objective_total") or 0.0)
    return weighted, loss_count, objective_total



def main() -> int:
    parser = argparse.ArgumentParser(description="Run random explorer hyperopt loops.")
    parser.add_argument("--preset", default="test-hyperopt", help="Launcher preset name to reuse.")
    parser.add_argument("--preset-file", default=str(PRESET_FILE), help="Preset JSON file to read.")
    parser.add_argument("--market-windows-file", default=str(MARKET_WINDOWS_FILE), help="Explorer market-window manifest JSON file.")
    parser.add_argument("--source-windows", nargs="*", help="Allowed selected market windows by manifest name or timerange.")
    parser.add_argument("--hyperopt-market-types", nargs="*", help="Allowed market types for the HyperOpt window.")
    parser.add_argument("--backtest-market-types", nargs="*", help="Allowed market types for backtest windows.")
    parser.add_argument("--backtest-counts-json", default="{}", help=argparse.SUPPRESS)
    parser.add_argument("--holdout-windows-json", default="[]", help=argparse.SUPPRESS)
    parser.add_argument("--selection-mode", choices=["random_families", "random_tags", "random_namespace", "custom_batches"], default="random_families", help="Whether each loop samples random families, random tags, namespace values, or committed custom batches.")
    parser.add_argument("--target-namespace", default="", help="Catalog namespace to sample when using random_namespace mode, for example mode, switch, domain, or action.")
    parser.add_argument("--family-count", type=int, default=3, help="How many unique families to sample across the whole Explorer run when using random_families mode.")
    parser.add_argument("--tag-count", type=int, default=5, help="How many unique tags to sample across the whole Explorer run when using random_tags mode.")
    parser.add_argument("--tag-filter", default="", help="Optional comma/space separated text filters for candidate tags, for example: switch:enable controls:.")
    parser.add_argument("--custom-batches-file", default=str(CUSTOM_BATCH_FILE), help="Custom Explorer target batches JSON file.")
    parser.add_argument("--custom-batch-ids", default="", help="Comma-separated or JSON-list custom batch IDs to run.")
    parser.add_argument("--min-param-count-per-hyper-run", type=int, default=0, help="Pad undersized family/tag selections with related strategy params until each hyperopt run has at least this many params. 0 disables padding.")
    parser.add_argument("--max-loops", type=int, default=1, help="How many exploration loops to run. 0 means uncapped.")
    parser.add_argument("--sampling-seed", type=int, default=None, help="Explorer target/window sampling seed. Leave unset for random per launch.")
    parser.add_argument("--epochs", help="Override effective epoch count (already multiplied by job workers, capped at 800).")
    parser.add_argument("--random-state", help="Override preset hyperopt random state.")
    parser.add_argument("--hyperopt-loss", help="Override preset hyperopt loss class.")
    parser.add_argument("--metadata-file", default=str(METADATA_FILE), help="Where to write the latest Explorer summary JSON.")
    parser.add_argument("--state-file", default=str(STATE_FILE), help="Where to store family/tag usage counters.")
    parser.add_argument("--strategy-param-file", default="", help="Strategy parameter JSON file to read and update.")
    parser.add_argument("--backtest-workers", type=int, default=12, help="Maximum parallel Explorer validation backtests. 1 preserves serial execution. Default 12.")
    parser.add_argument("--temp-backtest-root", default="", help="Parent directory for isolated Explorer worker strategy/result folders.")
    args = parser.parse_args()
    backtest_workers = max(1, int(args.backtest_workers or 12))
    temp_backtest_root = (
        Path(str(args.temp_backtest_root)).expanduser()
        if str(args.temp_backtest_root or "").strip()
        else (RUNTIME_DIR / "tempbacktest")
    )
    run_temp_root = make_explorer_run_temp_root(temp_backtest_root)

    presets = load_json(Path(args.preset_file), {})
    if args.preset not in presets:
        raise SystemExit(f"Preset not found: {args.preset}")
    preset = deepcopy(presets[args.preset])
    if args.hyperopt_loss:
        preset["hyperopt_loss"] = args.hyperopt_loss
    if args.epochs:
        preset["hyperopt_epochs"] = args.epochs
    if args.random_state:
        preset["hyperopt_random_state"] = args.random_state

    python_exe = str(preset.get("python_exe") or sys.executable)
    cwd = Path(str(preset.get("project_root") or Path.cwd()))

    strategy_file_value = str(preset.get("strategy_file") or "")
    strategy_class_value = str(preset.get("strategy_class") or "HybridRecoveryGridStrategy")
    if not strategy_file_value:
        raise SystemExit("Preset must include strategy_file for explorer runs.")
    strategy_path = Path(strategy_file_value)
    strategy_param_file = (
        Path(str(args.strategy_param_file)).expanduser()
        if str(args.strategy_param_file or "").strip()
        else strategy_path.with_suffix(".json")
    )
    if not strategy_param_file.is_absolute():
        strategy_param_file = (cwd / strategy_param_file).resolve()
    strategy_spaces = strategy_parameter_spaces(strategy_path, strategy_class_value)
    catalog = build_param_catalog(strategy_path, strategy_class_value)

    plan = load_market_windows_manifest(Path(args.market_windows_file))

    windows = [normalize_segment(window) for window in list(plan.get("market_windows", []))]
    if args.source_windows:
        allowed = {str(value) for value in args.source_windows}
        windows = [window for window in windows if window.get("name") in allowed or window.get("timerange") in allowed]
    if not windows:
        raise SystemExit("Explorer needs at least one selected normal market window for HyperOpt.")
    hyperopt_market_types = normalize_market_types(args.hyperopt_market_types)
    backtest_market_types = normalize_market_types(args.backtest_market_types)
    backtest_count_requests = parse_backtest_count_requests(args.backtest_counts_json)
    selected_holdout_windows = parse_holdout_windows(args.holdout_windows_json)
    if not selected_holdout_windows:
        raise SystemExit("Select at least one 12-month holdout backtest window. Holdout windows are required and are never used for HyperOpt.")
    hyperopt_windows = [window for window in windows if str(window.get("regime") or "") in hyperopt_market_types]
    normal_backtest_windows = [window for window in windows if str(window.get("regime") or "") in backtest_market_types]
    if not hyperopt_windows:
        raise SystemExit("No eligible HyperOpt windows match the selected HyperOpt market types.")

    hyperopt_market_types_list = [] if hyperopt_market_types == set(MARKET_TYPE_VALUES) else [value for value in MARKET_TYPE_VALUES if value in hyperopt_market_types]
    backtest_market_types_list = [] if backtest_market_types == set(MARKET_TYPE_VALUES) else [value for value in MARKET_TYPE_VALUES if value in backtest_market_types]

    print(f"Explorer HyperOpt loss: {preset.get('hyperopt_loss') or '-'}")
    print(f"Selection mode: {args.selection_mode}")
    if args.selection_mode == "random_namespace":
        print(f"Target namespace: {args.target_namespace or '-'}")
    print_family_catalog(catalog)

    base_child_env = build_child_env(os.environ.copy(), cwd)

    state_file = Path(args.state_file)
    state = load_json(
        state_file,
        {
            "family_usage_counts": {},
            "tag_usage_counts": {},
            "custom_batch_usage_counts": {},
            "accepted_changes": [],
        },
    )
    champion_backtest_cache: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    active_champion_cache_hash: str | None = None

    sampling_seed_input = args.sampling_seed
    if sampling_seed_input is not None:
        effective_sampling_seed = int(sampling_seed_input)
    else:
        effective_sampling_seed = int(random.SystemRandom().randint(0, 2_147_483_647))

    run_targets, selection_info = build_selection_targets(
        args=args,
        rng=random.Random(effective_sampling_seed),
        plan=plan,
        preset=preset,
        strategy_spaces=strategy_spaces,
        catalog=catalog,
        state=state,
    )
    print_launch_selection_summary(
        catalog,
        selection_info,
        sampling_seed_input=sampling_seed_input,
        effective_sampling_seed=effective_sampling_seed,
    )

    metadata_file = Path(args.metadata_file)
    legacy_root_metadata = RUNTIME_DIR / "hyperopt_explorer_metadata_latest.json"
    if metadata_file.resolve() == legacy_root_metadata.resolve():
        metadata_file = LATEST_SUMMARY_FILE
    metadata: dict[str, Any] = {
        "started_at": datetime.now().astimezone().isoformat(),
        "preset": args.preset,
        "preset_file": str(Path(args.preset_file)),
        "market_windows_file": str(Path(args.market_windows_file)),
        "strategy_param_file": str(strategy_param_file),
        "taxonomy_source": "strategy_catalog",
        "catalog_strategy_file": str(strategy_path),
        "catalog_strategy_class": strategy_class_value,
        "catalog_namespace_count": len(catalog.get("namespaces") or {}),
        "catalog_namespaces": deepcopy(catalog.get("namespaces") or {}),
        "strategy_file": str(strategy_path),
        "strategy_class": strategy_class_value,
        "hyperopt_loss": preset.get("hyperopt_loss"),
        "selection_mode": args.selection_mode,
        "target_namespace": args.target_namespace if args.selection_mode == "random_namespace" else None,
        "custom_batches_file": str(Path(args.custom_batches_file)) if args.selection_mode == "custom_batches" else None,
        "selected_custom_batches": list(selection_info.get("selected_custom_batches") or []),
        "family_count": int(str(args.family_count).strip() or 0) if str(args.family_count).strip() else 0,
        "tag_count": int(str(args.tag_count).strip() or 0) if str(args.tag_count).strip() else 0,
        "tag_filter": args.tag_filter,
        "min_param_count_per_hyper_run": args.min_param_count_per_hyper_run,
        "max_loops": args.max_loops,
        "sampling_seed_input": sampling_seed_input,
        "effective_sampling_seed": effective_sampling_seed,
        "sampling_seed": effective_sampling_seed,
        "selected_targets": [metadata_target(target) for target in run_targets],
        "selected_target_labels": [target.get("selection_label") for target in run_targets],
        "explorer_market_windows": [window.get("name") for window in windows],
        "selected_market_windows": [window.get("name") for window in windows],
        "hyperopt_window": None,
        "backtest_windows": [],
        "selected_windows": [window.get("name") for window in windows],
        "selected_hyperopt_market_types": hyperopt_market_types_list,
        "selected_backtest_market_types": backtest_market_types_list,
        "hyperopt_market_types": hyperopt_market_types_list,
        "backtest_market_types": backtest_market_types_list,
        "backtest_count_requests": dict(backtest_count_requests),
        "selected_12m_holdout_windows": [deepcopy(window) for window in selected_holdout_windows],
        "epochs": preset.get("hyperopt_epochs"),
        "random_state": preset.get("hyperopt_random_state"),
        "backtest_workers": backtest_workers,
        "temp_backtest_root": str(temp_backtest_root),
        "explorer_run_temp_root": str(run_temp_root),
        "search_runs": [],
        "backtest_runs": [],
        "validation_runs": [],
        "loops": [],
        "family_usage_counts": dict(sorted((state.get("family_usage_counts") or {}).items())),
        "tag_usage_counts": dict(sorted((state.get("tag_usage_counts") or {}).items())),
        "custom_batch_usage_counts": dict(sorted((state.get("custom_batch_usage_counts") or {}).items())),
        "summary": {},
    }
    save_json(
        run_temp_root / "manifest.json",
        {
            "started_at": metadata["started_at"],
            "backtest_workers": backtest_workers,
            "preset": args.preset,
            "runner_pid": os.getpid(),
            "strategy_file": str(strategy_path),
            "temp_root": str(temp_backtest_root),
        },
    )
    run_id_seed = f"{metadata['started_at']}|{args.preset}|{effective_sampling_seed}|{os.getpid()}"
    run_id = hashlib.sha1(run_id_seed.encode("utf-8")).hexdigest()[:10]
    metadata["run_id"] = run_id
    metadata_run_file = run_history_file(str(metadata["started_at"]), run_id)

    initial_snapshot = load_json(strategy_param_file, {})
    current_champion_snapshot = deepcopy(initial_snapshot)
    current_champion_label = "champion"
    loop_index = 0
    try:
        while True:
            loop_index += 1
            if args.max_loops > 0 and loop_index > args.max_loops:
                break

            rng = random.Random(effective_sampling_seed + loop_index)
            source_segment = deepcopy(rng.choice(hyperopt_windows))
            normal_validation_pool = [deepcopy(window) for window in normal_backtest_windows if segment_key(window) != segment_key(source_segment)]
            (
                selected_normal_backtest_windows,
                normal_backtest_available_by_type,
                normal_backtest_selected_by_type,
                backtest_selection_warnings,
            ) = select_normal_backtest_windows(normal_validation_pool, backtest_count_requests, rng)
            validation_segments = dedupe_segments(selected_normal_backtest_windows + [deepcopy(window) for window in selected_holdout_windows])
            if not validation_segments:
                raise SystemExit("No eligible backtest windows remain after applying the selected backtest market types.")
            for warning in backtest_selection_warnings:
                print(f"Warning: {warning}")
            if not run_targets:
                raise SystemExit("No eligible Explorer targets were selected for this run.")
            loop_target = deepcopy(run_targets[(loop_index - 1) % len(run_targets)])
            current_champion_hash = hash_json_value(current_champion_snapshot.get("params") or current_champion_snapshot)
            if current_champion_hash != active_champion_cache_hash:
                champion_backtest_cache.clear()
                active_champion_cache_hash = current_champion_hash

            metadata["family_usage_counts"] = dict(sorted((state.get("family_usage_counts") or {}).items()))
            metadata["tag_usage_counts"] = dict(sorted((state.get("tag_usage_counts") or {}).items()))
            metadata["custom_batch_usage_counts"] = dict(sorted((state.get("custom_batch_usage_counts") or {}).items()))
            metadata["selected_targets"] = [metadata_target(target) for target in run_targets]
            metadata["selected_target_labels"] = [target.get("selection_label") for target in run_targets]
            metadata["hyperopt_window"] = deepcopy(source_segment)
            metadata["backtest_windows"] = [deepcopy(segment) for segment in validation_segments]
            metadata["normal_backtest_count_requests"] = dict(backtest_count_requests)
            metadata["normal_backtest_available_by_type"] = dict(normal_backtest_available_by_type)
            metadata["normal_backtest_selected_by_type"] = {
                regime: [deepcopy(segment) for segment in normal_backtest_selected_by_type.get(regime, [])]
                for regime in MARKET_TYPE_VALUES
            }
            metadata["selected_12m_holdout_windows"] = [deepcopy(window) for window in selected_holdout_windows]
            metadata["final_backtest_windows"] = [deepcopy(segment) for segment in validation_segments]
            metadata["backtest_selection_warnings"] = list(backtest_selection_warnings)
            metadata["selected_target_label"] = loop_target.get("selection_label")

            loop_record: dict[str, Any] = {
                "loop_index": loop_index,
                "hyperopt_window": deepcopy(source_segment),
                "backtest_windows": [deepcopy(segment) for segment in validation_segments],
                "normal_backtest_count_requests": dict(backtest_count_requests),
                "normal_backtest_available_by_type": dict(normal_backtest_available_by_type),
                "normal_backtest_selected_by_type": {
                    regime: [deepcopy(segment) for segment in normal_backtest_selected_by_type.get(regime, [])]
                    for regime in MARKET_TYPE_VALUES
                },
                "selected_12m_holdout_windows": [deepcopy(window) for window in selected_holdout_windows],
                "final_backtest_windows": [deepcopy(segment) for segment in validation_segments],
                "backtest_selection_warnings": list(backtest_selection_warnings),
                "source_segment": deepcopy(source_segment),
                "validation_segments": [deepcopy(segment) for segment in validation_segments],
                "loop_target": metadata_target(loop_target),
                "loop_target_label": loop_target.get("selection_label"),
                "selection_mode": args.selection_mode,
                "target_namespace": args.target_namespace if args.selection_mode == "random_namespace" else None,
                "selected_families": list(selection_info.get("selected_families") or []),
                "selected_tags": list(selection_info.get("selected_tags") or []),
                "selected_namespace_tags": list(selection_info.get("selected_namespace_tags") or []),
                "tag_filter_tokens": list(selection_info.get("tag_filter_tokens") or []),
                "sampling_seed_input": sampling_seed_input,
                "effective_sampling_seed": effective_sampling_seed,
                "backtest_workers": backtest_workers,
                "challengers": [],
                "loop_applied": False,
                "applied_candidate_id": None,
            }
            metadata["loops"].append(loop_record)

            print("\n" + "=" * 100)
            print(f"Explorer loop {loop_index}")
            print(f"HyperOpt window: {describe_segment(source_segment)}")
            print(f"Backtest windows: {len(validation_segments)}")
            if loop_target.get("selection_type") == "custom_batch":
                excluded_text = ", ".join(loop_target.get("excluded_params") or []) or "-"
                spaces_text = ", ".join(loop_target.get("spaces") or []) or "-"
                print(f"Target: {loop_target.get('selection_label')}")
                print("Mode: Custom batches")
                print(f"Params: {int(loop_target.get('active_param_count') or len(loop_target.get('group_params') or []))}")
                print(f"Spaces: {spaces_text}")
                print(f"Excluded: {excluded_text}")
            else:
                spaces_text = ", ".join(loop_target.get("spaces") or []) or "-"
                print(f"Target: {loop_target.get('selection_label')}")
                print(f"Namespace: {loop_target.get('namespace') or '-'}")
                print(f"Params: {int(loop_target.get('active_param_count') or len(loop_target.get('group_params') or []))}")
                print(f"Spaces: {spaces_text}")
            print(
                "Active HyperOpt params: "
                f"{int(loop_target.get('active_param_count') or len(loop_target.get('group_params') or []))}"
                f" (primary={int(loop_target.get('primary_param_count') or 0)}, "
                f"minimum={int(loop_target.get('min_param_count_per_hyper_run') or 0)})"
            )
            if loop_target.get("supplemental_tags"):
                print(f"Min-param padding tags: {', '.join(loop_target['supplemental_tags'])}")
            if loop_target.get("supplemental_families"):
                print(f"Min-param padding families: {', '.join(loop_target['supplemental_families'])}")
            emit_explorer_status(
                "loop_start",
                {
                    "loop_index": loop_index,
                    "target": loop_target.get("selection_label"),
                    "namespace": loop_target.get("namespace"),
                    "namespace_value": loop_target.get("namespace_value"),
                    "family": loop_target.get("family"),
                    "tag": loop_target.get("tag"),
                    "hyperopt_window": status_segment_label(source_segment),
                    "hyperopt_window_detail": describe_segment(source_segment),
                    "hyperopt_regime": source_segment.get("regime"),
                    "backtest_windows": [status_segment_label(segment) for segment in validation_segments],
                    "backtest_window_count": len(validation_segments),
                    "selection_mode": args.selection_mode,
                    "target_namespace": args.target_namespace if args.selection_mode == "random_namespace" else None,
                    "sampling_seed_input": sampling_seed_input,
                    "effective_sampling_seed": effective_sampling_seed,
                },
            )

            challengers: list[dict[str, Any]] = []
            target = loop_target
            selection_label = str(target["selection_label"])
            group_params = {str(param) for param in target["group_params"] if str(param)}
            if group_params:
                active_param_count = len({param for param in group_params if param in strategy_spaces})
                min_param_count = int(target.get("min_param_count_per_hyper_run") or 0)
                if min_param_count > 0 and active_param_count < min_param_count:
                    raise SystemExit(
                        f"Explorer target {selection_label} only has {active_param_count} active HyperOpt params "
                        f"after min-param padding; required minimum is {min_param_count}."
                    )
                env = base_child_env.copy()
                env.pop("HYBRID_RECOVERY_HYPEROPT_GROUPS", None)
                env[PARAM_ENV] = ",".join(sorted(group_params))
                pseudo_group = {"spaces": list(target.get("spaces") or ["buy", "sell"])}
                command = [python_exe, "-u", "-m", "freqtrade", *build_hyperopt_args(preset, pseudo_group, str(source_segment["timerange"]), args.epochs, args.random_state)]
                run_record: dict[str, Any] = {
                    "loop_index": loop_index,
                    "selection_type": target.get("selection_type"),
                    "selection_mode": target.get("selection_mode"),
                    "selection_label": selection_label,
                    "namespace": target.get("namespace"),
                    "namespace_value": target.get("namespace_value"),
                    "family": target.get("family"),
                    "tag": target.get("tag"),
                    "hyperopt_window": deepcopy(source_segment),
                    "source_segment": deepcopy(source_segment),
                    "group_params": sorted(group_params),
                    "primary_group_params": sorted(str(param) for param in (target.get("primary_group_params") or []) if str(param)),
                    "primary_param_count": int(target.get("primary_param_count") or 0),
                    "active_param_count": active_param_count,
                    "min_param_count_per_hyper_run": min_param_count,
                    "supplemental_tags": list(target.get("supplemental_tags") or []),
                    "supplemental_families": list(target.get("supplemental_families") or []),
                    "supplemental_params": sorted(str(param) for param in (target.get("supplemental_params") or []) if str(param)),
                    "batch_id": target.get("batch_id"),
                    "batch_name": target.get("batch_name"),
                    "excluded_params": list(target.get("excluded_params") or []),
                    "stale_sources": list(target.get("stale_sources") or []),
                    "stale_params": list(target.get("stale_params") or []),
                    "stale_excluded_params": list(target.get("stale_excluded_params") or []),
                    "spaces": list(target.get("spaces") or []),
                    "command": command_text(command),
                    "status": "started",
                    "loop_target": metadata_target(loop_target),
                    "loop_target_label": loop_target.get("selection_label"),
                }
                metadata["search_runs"].append(run_record)

                results_dir = hyperopt_results_dir(preset)
                previous_result = marker_latest_file(results_dir)
                before_snapshot = result_file_snapshot(results_dir, ("*.fthypt", "*.pickle"))
                started_at = time.time()
                result = run_command(command, cwd, env, False, stream_output=True)
                assert result is not None
                run_record["exit_code"] = result.returncode
                if result.returncode != 0:
                    run_record["status"] = "hyperopt_failed"
                    print(f"HyperOpt failed for {selection_label} on {describe_segment(source_segment)}")
                else:
                    result_file = latest_result_file(results_dir, ("*.fthypt", "*.pickle"), started_at, previous_result, before_snapshot)
                    if not result_file:
                        run_record["status"] = "result_missing"
                        print(f"No hyperopt result file found for {selection_label}")
                    else:
                        run_record["result_file"] = str(result_file)
                        try:
                            best, total_epochs = best_epoch_from_file(result_file)
                        except Exception as exc:
                            run_record["status"] = "result_parse_failed"
                            run_record["error"] = str(exc)
                            print(str(exc))
                        else:
                            loss = find_loss(best)
                            if loss is None:
                                run_record["status"] = "loss_missing"
                                print(f"Could not read loss from best result in {result_file.name}")
                            else:
                                accepted_params = filtered_params(params_from_result(best), group_params)
                                changes = describe_param_changes(accepted_params, current_strategy_values(strategy_param_file))
                                params_changed_count = changed_param_count(changes)
                                run_record.update(
                                    {
                                        "status": "completed",
                                        "best_epoch": best.get("current_epoch"),
                                        "total_epochs": total_epochs,
                                        "loss": loss,
                                        "proposed_params": accepted_params,
                                        "changes": changes,
                                        "params_changed_count": params_changed_count,
                                    }
                                )
                                if not accepted_params:
                                    print(f"Search run for {selection_label} completed, but no params from this target were returned.")
                                else:
                                    candidate_id = f"loop{loop_index:03d}__{selection_label.replace(':', '_').replace('/', '_').replace(' ', '_')}"
                                    candidate = {
                                        "candidate_id": candidate_id,
                                        "selection_type": target.get("selection_type"),
                                        "selection_mode": target.get("selection_mode"),
                                        "selection_label": selection_label,
                                        "namespace": target.get("namespace"),
                                        "namespace_value": target.get("namespace_value"),
                                        "family": target.get("family"),
                                        "tag": target.get("tag"),
                                        "hyperopt_window": deepcopy(source_segment),
                                        "source_segment": deepcopy(source_segment),
                                        "loss": loss,
                                        "params": accepted_params,
                                        "changes": changes,
                                        "result_file": str(result_file),
                                        "spaces": list(target.get("spaces") or []),
                                        "loop_target": metadata_target(loop_target),
                                        "loop_target_label": loop_target.get("selection_label"),
                                        "params_changed_count": params_changed_count,
                                    }
                                    challengers.append(candidate)
                                    loop_record["challengers"].append(candidate)
                                    print(f"Collected challenger: {candidate_id} loss={loss}")
                if run_record.get("status") == "started":
                    run_record["status"] = "completed"
            else:
                run_record = {
                    "loop_index": loop_index,
                    "selection_type": target.get("selection_type"),
                    "selection_mode": target.get("selection_mode"),
                    "selection_label": selection_label,
                    "namespace": target.get("namespace"),
                    "namespace_value": target.get("namespace_value"),
                    "family": target.get("family"),
                    "tag": target.get("tag"),
                    "hyperopt_window": deepcopy(source_segment),
                    "source_segment": deepcopy(source_segment),
                    "group_params": [],
                    "spaces": list(target.get("spaces") or []),
                    "status": "skipped_empty_target",
                    "loop_target": metadata_target(loop_target),
                    "loop_target_label": loop_target.get("selection_label"),
                }
                metadata["search_runs"].append(run_record)

            if not challengers:
                loop_record["status"] = "no_challengers"
                print("No challengers were collected for this loop.")
                flush_metadata_checkpoint(metadata, metadata_file, metadata_run_file)
                save_json(state_file, state)
                continue

            champion_snapshot = deepcopy(current_champion_snapshot)
            save_json(strategy_param_file, champion_snapshot)

            champion_metrics: dict[str, dict[str, Any]] = {}
            print(f"Champion backtest baseline: {current_champion_label} across {len(validation_segments)} segment(s)")
            champion_pair_hash = preset_pair_list_hash(preset)
            champion_timeframe = str(preset.get("timeframe") or "")
            champion_config_hash = preset_config_hash(preset)
            champion_strategy_hash = sha256_file(strategy_path)

            def champion_cache_key(segment: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
                segment_label = str(segment.get("name") or segment.get("timerange") or "segment")
                return (
                    current_champion_hash,
                    segment_label,
                    champion_pair_hash,
                    champion_timeframe,
                    champion_config_hash,
                    champion_strategy_hash,
                )

            for segment in validation_segments:
                segment_label = str(segment.get("name") or segment.get("timerange") or "segment")
                if champion_backtest_cache.get(champion_cache_key(segment)) is not None:
                    print(f"Champion baseline reused from cache: {segment_label}")

            champion_records = run_validation_backtests(
                base_preset=preset,
                cwd=cwd,
                base_child_env=base_child_env,
                run_temp_root=run_temp_root,
                role="champion",
                loop_index=loop_index,
                candidate_id=current_champion_label,
                snapshot=champion_snapshot,
                validation_segments=validation_segments,
                strategy_path=strategy_path,
                dry_run=False,
                max_workers=backtest_workers,
                champion_cache_get=lambda segment: champion_backtest_cache.get(champion_cache_key(segment)),
                champion_cache_put=lambda segment, record: champion_backtest_cache.__setitem__(champion_cache_key(segment), deepcopy(record)),
            )

            for record, segment in zip(champion_records, validation_segments):
                record["loop_index"] = loop_index
                record["candidate_id"] = current_champion_label
                record["role"] = "champion"
                metadata["backtest_runs"].append(record)
                metadata["validation_runs"].append(record)
                if record.get("status") != "completed":
                    loop_record["status"] = "champion_backtest_failed"
                    print(f"Champion backtest failed on {describe_segment(segment)}; status={record.get('status')}")
                    if record.get("stdout_tail"):
                        print(record.get("stdout_tail"))
                    flush_metadata_checkpoint(metadata, metadata_file, metadata_run_file)
                    save_json(state_file, state)
                    return 1
                champion_metrics[segment_key(segment)] = dict(record.get("metrics") or {})

            best_candidate: dict[str, Any] | None = None
            best_candidate_sort_key: tuple[float, int, float] | None = None

            for candidate in challengers:
                candidate_snapshot = merge_params_into_snapshot(champion_snapshot, candidate["params"])
                candidate_metrics: dict[str, dict[str, Any]] = {}
                candidate_records: list[dict[str, Any]] = []
                print(f"\n=== Challenger {candidate['candidate_id']} ({candidate['selection_label']}) ===")
                print(f"HyperOpt window: {describe_segment(candidate.get('hyperopt_window', candidate.get('source_segment', {})))}")
                print(f"HyperOpt loss: {candidate.get('loss')}")
                for change in candidate.get("changes", []):
                    if change.get("changed"):
                        print(f"  {change['param']}: {change.get('previous')} -> {change.get('new')}")
                candidate_records = run_validation_backtests(
                    base_preset=preset,
                    cwd=cwd,
                    base_child_env=base_child_env,
                    run_temp_root=run_temp_root,
                    role="challenger",
                    loop_index=loop_index,
                    candidate_id=str(candidate["candidate_id"]),
                    snapshot=candidate_snapshot,
                    validation_segments=validation_segments,
                    strategy_path=strategy_path,
                    dry_run=False,
                    max_workers=backtest_workers,
                )
                failed_validation = False
                for record, segment in zip(candidate_records, validation_segments):
                    record["loop_index"] = loop_index
                    record["candidate_id"] = candidate["candidate_id"]
                    record["role"] = "challenger"
                    metadata["backtest_runs"].append(record)
                    metadata["validation_runs"].append(record)
                    if record.get("status") != "completed":
                        failed_validation = True
                        print(f"Backtest failed for {candidate['candidate_id']} on {describe_segment(segment)}; status={record.get('status')}")
                        if record.get("stdout_tail"):
                            print(record.get("stdout_tail"))
                        if record.get("exception"):
                            print(record.get("exception"))
                    else:
                        candidate_metrics[segment_key(segment)] = dict(record.get("metrics") or {})
                save_json(strategy_param_file, champion_snapshot)
                if failed_validation or len(candidate_metrics) != len(validation_segments):
                    candidate["validation"] = {"accepted": False, "reason": "backtest_failed"}
                    continue

                summary = validation_summary(champion_metrics, candidate_metrics)
                summary["params_changed_count"] = int(candidate.get("params_changed_count") or changed_param_count(candidate.get("changes")))
                candidate["validation"] = summary
                candidate["backtest_runs"] = candidate_records
                candidate["validation_runs"] = candidate_records
                print_validation_table(validation_segments, champion_metrics, candidate_metrics)
                print_acceptance_summary(summary)
                emit_explorer_status(
                    "challenger_decision",
                    {
                        "loop_index": loop_index,
                        "candidate_id": candidate.get("candidate_id"),
                        "target": candidate.get("selection_label"),
                        "hyperopt_window": status_segment_label(candidate.get("hyperopt_window", candidate.get("source_segment"))),
                        "hyperopt_window_detail": describe_segment(candidate.get("hyperopt_window", candidate.get("source_segment", {}))),
                        "hyperopt_regime": (candidate.get("hyperopt_window") or candidate.get("source_segment") or {}).get("regime")
                        if isinstance(candidate.get("hyperopt_window") or candidate.get("source_segment"), dict)
                        else None,
                        "backtest_windows": [status_segment_label(segment) for segment in validation_segments],
                        "backtest_window_count": int(summary["backtest_window_count"]),
                        "champion_score": float(summary["champion_weighted_acceptance_score"]),
                        "challenger_score": float(summary["challenger_weighted_acceptance_score"]),
                        "champion_profit_total": float(summary["champion_profit_total"]),
                        "challenger_profit_total": float(summary["challenger_profit_total"]),
                        "champion_objective_total": float(summary["champion_objective_total"]),
                        "challenger_objective_total": float(summary["challenger_objective_total"]),
                        "champion_loss_window_count": int(summary["champion_loss_window_count"]),
                        "challenger_loss_window_count": int(summary["challenger_loss_window_count"]),
                        "decision": summary["decision"],
                        "decision_reason": summary.get("decision_reason"),
                        "params_changed": int(summary["params_changed_count"]),
                        "weighted_score_delta": float(summary["weighted_score_delta"]),
                        "namespace": candidate.get("namespace"),
                        "namespace_value": candidate.get("namespace_value"),
                        "family": candidate.get("family"),
                        "tag": candidate.get("tag"),
                    },
                )

                if summary["accepted"]:
                    sort_key = candidate_sort_key(candidate)
                    if best_candidate_sort_key is None or sort_key < best_candidate_sort_key:
                        best_candidate = candidate
                        best_candidate_sort_key = sort_key

            save_json(strategy_param_file, current_champion_snapshot)

            if best_candidate is not None:
                current_champion_snapshot = merge_params_into_snapshot(current_champion_snapshot, best_candidate["params"])
                current_champion_label = best_candidate["candidate_id"]
                loop_record["loop_applied"] = True
                loop_record["applied_candidate_id"] = best_candidate["candidate_id"]
                save_json(strategy_param_file, current_champion_snapshot)
                print("Accepted challenger applied to the champion.")
                state.setdefault("accepted_changes", []).append(
                    {
                        "updated_at": datetime.now().astimezone().isoformat(),
                        "candidate_id": best_candidate["candidate_id"],
                        "selection_type": best_candidate.get("selection_type"),
                        "selection_label": best_candidate.get("selection_label"),
                        "family": best_candidate.get("family"),
                        "tag": best_candidate.get("tag"),
                        "hyperopt_window": best_candidate.get("hyperopt_window", best_candidate.get("source_segment")),
                        "source_segment": best_candidate.get("source_segment"),
                        "changes": best_candidate.get("changes", []),
                    }
                )
            else:
                print("No challenger beat the champion baseline.")

            loop_record["status"] = "completed"
            loop_record["final_champion_label"] = current_champion_label
            flush_metadata_checkpoint(metadata, metadata_file, metadata_run_file)
            save_json(state_file, state)

    finally:
        try:
            save_json(strategy_param_file, current_champion_snapshot)
        except Exception as exc:
            print(f"Warning: cleanup failed: {exc}")

    metadata["finished_at"] = datetime.now().astimezone().isoformat()
    flush_metadata_checkpoint(metadata, metadata_file, metadata_run_file)
    save_json(state_file, state)
    print(f"\nExplorer latest summary JSON: {metadata_file}")
    print(f"Explorer detailed run JSON: {metadata_run_file}")
    print(f"Explorer state JSON: {state_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

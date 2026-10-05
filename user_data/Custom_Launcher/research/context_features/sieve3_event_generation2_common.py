"""Shared integrity and comparison helpers for frozen Sieve3 Generation 2."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet
from pandas import DataFrame
from sieve3_event_generation1_b1_freqai import METRICS, add_majority_consistency
from sieve3_event_generation1_review import model_output_audit


KEY = ["pair", "window", "scope", "target"]
RANK_METRICS = (
    "prediction_actual_spearman",
    "top_minus_bottom",
    "r_squared",
    "roc_auc",
    "average_precision",
)
ERROR_METRICS = (
    "mean_absolute_error",
    "root_mean_squared_error",
    "brier_score_clipped",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def generation_batch(generation_manifest: dict[str, Any], batch_id: str) -> dict[str, Any]:
    matches = [item for item in generation_manifest["batches"] if item["id"] == batch_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one frozen {batch_id} definition, found {len(matches)}")
    return matches[0]


def review_batch(review: dict[str, Any], batch_id: str) -> dict[str, Any]:
    matches = [item for item in review["batch_audits"] if item["batch"] == batch_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one sealed {batch_id} audit, found {len(matches)}")
    return matches[0]


def _normalise_path(value: str | Path) -> str:
    return str(Path(value).resolve()).casefold()


def cache_contract_audit(
    cache_dir: Path,
    pairs: tuple[str, ...],
    required_columns: Iterable[str],
) -> dict[str, Any]:
    columns = tuple(dict.fromkeys(str(column) for column in required_columns))
    pair_audits: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for pair in pairs:
        token = pair.split("/", 1)[0].lower()
        path = cache_dir / f"{token}_sieve3_events_1h.parquet"
        audit: dict[str, Any] = {"path": str(path)}
        if not path.is_file():
            audit["errors"] = ["cache file missing"]
            pair_audits[pair] = audit
            errors.append(f"{pair}: cache file missing")
            continue
        available = set(parquet.read_schema(path).names)
        missing = sorted(set(columns).difference(available))
        selected = ["decision_time", "coverage_present", *columns]
        selected = list(dict.fromkeys(column for column in selected if column in available))
        frame = pd.read_parquet(path, columns=selected)
        decision_time = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
        coverage = pd.to_numeric(frame["coverage_present"], errors="coerce")
        present_required = [column for column in columns if column in frame]
        numeric = (
            frame[present_required].apply(pd.to_numeric, errors="coerce")
            if present_required
            else DataFrame(index=frame.index)
        )
        active = coverage.eq(1.0)
        null_cells = int(numeric.loc[active].isna().sum().sum()) if present_required else 0
        infinite_cells = (
            int(np.isinf(numeric.loc[active].to_numpy(dtype=float)).sum())
            if present_required
            else 0
        )
        pair_errors: list[str] = []
        if missing:
            pair_errors.append(f"missing required columns: {missing}")
        if decision_time.isna().any():
            pair_errors.append("invalid decision timestamps")
        if decision_time.duplicated().any():
            pair_errors.append("duplicate decision timestamps")
        if not decision_time.is_monotonic_increasing:
            pair_errors.append("decision timestamps are not ordered")
        if not coverage.eq(1.0).all():
            pair_errors.append("coverage_present is not one on every cached row")
        if null_cells:
            pair_errors.append(f"{null_cells} selected feature cells are null")
        if infinite_cells:
            pair_errors.append(f"{infinite_cells} selected feature cells are infinite")
        audit.update(
            {
                "rows": len(frame),
                "start": None if frame.empty else decision_time.iloc[0].isoformat(),
                "end": None if frame.empty else decision_time.iloc[-1].isoformat(),
                "required_columns": len(columns),
                "missing_columns": missing,
                "duplicate_decision_times": int(decision_time.duplicated().sum()),
                "invalid_decision_times": int(decision_time.isna().sum()),
                "coverage_rows": int(active.sum()),
                "null_feature_cells": null_cells,
                "infinite_feature_cells": infinite_cells,
                "errors": pair_errors,
            }
        )
        pair_audits[pair] = audit
        errors.extend(f"{pair}: {error}" for error in pair_errors)
    return {
        "cache_dir": str(cache_dir),
        "pairs": len(pairs),
        "required_columns": list(columns),
        "pair_audits": pair_audits,
        "errors": errors,
        "passed": not errors,
    }


def reuse_profiles(  # noqa: C901
    *,
    alias_to_source: dict[str, str],
    expected_profiles: dict[str, dict[str, Any]],
    source_manifest: dict[str, Any],
    sealed_batch_audit: dict[str, Any],
    expected_pairs: tuple[str, ...],
    expected_targets: list[str],
    expected_timerange: str,
    expected_cache: Path,
    expected_timeframes: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    commands = {item["profile_id"]: item for item in source_manifest["commands"]}
    sealed = {
        item["profile_id"]: item for item in sealed_batch_audit["completed_profile_output_audits"]
    }
    reused_commands: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    source_surface_errors: list[str] = []
    if tuple(source_manifest["pairs"]) != expected_pairs:
        source_surface_errors.append("source pair surface differs")
    if list(source_manifest["targets"]) != list(expected_targets):
        source_surface_errors.append("source target list differs")
    if str(source_manifest["timerange"]) != expected_timerange:
        source_surface_errors.append("source timerange differs")
    if list(source_manifest["include_timeframes"]) != list(expected_timeframes):
        source_surface_errors.append("source include_timeframes differs")
    for alias, source_profile in alias_to_source.items():
        definition = expected_profiles[alias]
        source = commands.get(source_profile)
        sealed_output = sealed.get(source_profile)
        errors = list(source_surface_errors)
        if source is None:
            errors.append("source command missing")
            audits.append({"profile_id": alias, "source_profile": source_profile, "errors": errors})
            continue
        if sealed_output is None:
            errors.append("sealed output audit missing")
        if source.get("status") != "completed" or int(source.get("returncode", 1)) != 0:
            errors.append("source command is not completed successfully")
        config_path = Path(source["config_path"])
        config = read_json(config_path) if config_path.is_file() else {}
        settings = config.get("sieve3_event_reaction", {})
        actual_columns = list(settings.get("event_feature_columns") or [])
        expected_columns = list(definition.get("feature_columns") or [])
        actual_shift = int(settings.get("event_feature_shift_hours", 0) or 0)
        expected_shift = int(definition.get("shift_hours", 0) or 0)
        if source.get("strategy") != definition["strategy"]:
            errors.append("strategy differs")
        if actual_columns != expected_columns:
            errors.append("event feature list differs")
        if actual_shift != expected_shift:
            errors.append("event feature shift differs")
        if list(settings.get("target_names") or []) != list(expected_targets):
            errors.append("config target list differs")
        if tuple(config.get("exchange", {}).get("pair_whitelist", [])) != expected_pairs:
            errors.append("config pair whitelist differs")
        if _normalise_path(settings.get("event_cache_dir", "")) != _normalise_path(expected_cache):
            errors.append("config cache path differs")
        if list(
            config.get("freqai", {}).get("feature_parameters", {}).get("include_timeframes", [])
        ) != list(expected_timeframes):
            errors.append("config include_timeframes differs")
        current_output = model_output_audit(source, source_manifest)
        if current_output["errors"]:
            errors.extend(f"current output: {item}" for item in current_output["errors"])
        if sealed_output is not None:
            for field in (
                "identifier",
                "trained_submodels",
                "prediction_files",
                "prediction_pairs",
                "expected_rows_per_pair",
                "eligible_mask_digest",
                "coverage",
                "eligible_coverage",
            ):
                if current_output.get(field) != sealed_output.get(field):
                    errors.append(f"current output differs from sealed audit: {field}")
        audit = {
            "profile_id": alias,
            "source_profile": source_profile,
            "identifier": source.get("identifier"),
            "strategy": source.get("strategy"),
            "feature_columns": actual_columns,
            "shift_hours": actual_shift,
            "eligible_mask_digest": current_output.get("eligible_mask_digest"),
            "prediction_files": current_output.get("prediction_files"),
            "prediction_pairs": current_output.get("prediction_pairs"),
            "expected_rows_per_pair": current_output.get("expected_rows_per_pair"),
            "errors": errors,
            "passed": not errors,
        }
        audits.append(audit)
        cloned = copy.deepcopy(source)
        cloned["profile_id"] = alias
        cloned["theory"] = definition["theory"]
        cloned["reuse_source"] = {
            "batch": source_manifest["batch"],
            "profile_id": source_profile,
            "identifier": source["identifier"],
            "sealed_eligible_mask_digest": None
            if sealed_output is None
            else sealed_output.get("eligible_mask_digest"),
        }
        reused_commands.append(cloned)
    return reused_commands, audits


def add_comparisons(
    scores: DataFrame,
    comparator_prefixes: dict[str, str],
) -> DataFrame:
    scores = scores.copy()
    for metric in METRICS:
        if metric not in scores:
            scores[metric] = np.nan
    output = scores.copy()
    for profile_id, prefix in comparator_prefixes.items():
        comparator = scores[scores["profile_id"].eq(profile_id)][[*KEY, *METRICS]].rename(
            columns={metric: f"{prefix}_{metric}" for metric in METRICS}
        )
        output = output.merge(comparator, on=KEY, how="left", validate="many_to_one")
        for metric in RANK_METRICS:
            output[f"{metric}_delta_vs_{prefix}"] = pd.to_numeric(
                output[metric], errors="coerce"
            ) - pd.to_numeric(output[f"{prefix}_{metric}"], errors="coerce")
        for metric in ERROR_METRICS:
            output[f"{metric}_skill_vs_{prefix}"] = pd.to_numeric(
                output[f"{prefix}_{metric}"], errors="coerce"
            ) - pd.to_numeric(output[metric], errors="coerce")
    return output


def controlled_portability(
    *,
    scores: DataFrame,
    comparison_contract: dict[str, list[str]],
    comparator_prefixes: dict[str, str],
    metadata_columns: list[str],
    include_binary: bool,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    records: list[DataFrame] = []
    intersections: list[DataFrame] = []
    for profile_id, comparators in comparison_contract.items():
        profile = scores[
            scores["status"].eq("scored")
            & scores["profile_id"].eq(profile_id)
            & ~scores["scope"].eq("all")
        ].copy()
        if profile.empty:
            continue
        rank_flags: list[pd.Series] = []
        binary_flags: list[pd.Series] = []
        worst_rank: list[pd.Series] = []
        worst_mae: list[pd.Series] = []
        worst_auc: list[pd.Series] = []
        worst_brier: list[pd.Series] = []
        for comparator_id in comparators:
            prefix = comparator_prefixes[comparator_id]
            rank_delta = pd.to_numeric(
                profile[f"prediction_actual_spearman_delta_vs_{prefix}"], errors="coerce"
            )
            mae_skill = pd.to_numeric(
                profile[f"mean_absolute_error_skill_vs_{prefix}"], errors="coerce"
            )
            work_columns = ["pair", "window", "scope", "target", *metadata_columns, "rows"]
            work = profile[work_columns].copy()
            work["profile_id"] = profile_id
            work["comparator_id"] = comparator_id
            work["spearman_delta"] = rank_delta
            work["mae_skill"] = mae_skill
            work["rank_and_mae_positive"] = rank_delta.gt(0.0) & mae_skill.gt(0.0)
            rank_flags.append(work["rank_and_mae_positive"])
            worst_rank.append(rank_delta)
            worst_mae.append(mae_skill)
            if include_binary:
                auc_delta = pd.to_numeric(profile[f"roc_auc_delta_vs_{prefix}"], errors="coerce")
                brier_skill = pd.to_numeric(
                    profile[f"brier_score_clipped_skill_vs_{prefix}"], errors="coerce"
                )
                work["auc_delta"] = auc_delta
                work["brier_skill"] = brier_skill
                work["auc_and_brier_positive"] = auc_delta.gt(0.0) & brier_skill.gt(0.0)
                binary_flags.append(work["auc_and_brier_positive"])
                worst_auc.append(auc_delta)
                worst_brier.append(brier_skill)
            records.append(work)
        intersection = profile[
            ["pair", "window", "scope", "target", *metadata_columns, "rows"]
        ].copy()
        intersection["profile_id"] = profile_id
        intersection["comparators"] = ",".join(comparators)
        intersection["all_controls_rank_mae_positive"] = pd.concat(rank_flags, axis=1).all(axis=1)
        intersection["worst_spearman_delta"] = pd.concat(worst_rank, axis=1).min(axis=1)
        intersection["worst_mae_skill"] = pd.concat(worst_mae, axis=1).min(axis=1)
        if include_binary:
            intersection["all_controls_auc_brier_positive"] = pd.concat(binary_flags, axis=1).all(
                axis=1
            )
            intersection["worst_auc_delta"] = pd.concat(worst_auc, axis=1).min(axis=1)
            intersection["worst_brier_skill"] = pd.concat(worst_brier, axis=1).min(axis=1)
        intersections.append(intersection)
    long = pd.concat(records, ignore_index=True) if records else DataFrame()
    intersection_long = (
        pd.concat(intersections, ignore_index=True) if intersections else DataFrame()
    )
    if long.empty:
        return long, DataFrame(), DataFrame()
    pairwise_group = ["profile_id", "comparator_id", "scope", "target", *metadata_columns]
    aggregations: dict[str, tuple[str, Any]] = {
        "pair_window_tests": ("pair", "size"),
        "distinct_pairs": ("pair", "nunique"),
        "distinct_windows": ("window", "nunique"),
        "sample_rows": ("rows", "sum"),
        "median_spearman_delta": ("spearman_delta", "median"),
        "median_mae_skill": ("mae_skill", "median"),
    }
    if include_binary:
        aggregations.update(
            median_auc_delta=("auc_delta", "median"),
            median_brier_skill=("brier_skill", "median"),
        )
    pairwise = (
        long.groupby(pairwise_group, dropna=False, observed=True).agg(**aggregations).reset_index()
    )
    pairwise = add_majority_consistency(
        pairwise, long, pairwise_group, "rank_and_mae_positive", "rank_mae"
    )
    if include_binary:
        pairwise = add_majority_consistency(
            pairwise, long, pairwise_group, "auc_and_brier_positive", "auc_brier"
        )
    intersection_group = ["profile_id", "scope", "target", *metadata_columns]
    intersection_aggregations: dict[str, tuple[str, Any]] = {
        "pair_window_tests": ("pair", "size"),
        "distinct_pairs": ("pair", "nunique"),
        "distinct_windows": ("window", "nunique"),
        "sample_rows": ("rows", "sum"),
        "median_worst_spearman_delta": ("worst_spearman_delta", "median"),
        "median_worst_mae_skill": ("worst_mae_skill", "median"),
    }
    if include_binary:
        intersection_aggregations.update(
            median_worst_auc_delta=("worst_auc_delta", "median"),
            median_worst_brier_skill=("worst_brier_skill", "median"),
        )
    intersection = (
        intersection_long.groupby(intersection_group, dropna=False, observed=True)
        .agg(**intersection_aggregations)
        .reset_index()
    )
    intersection = add_majority_consistency(
        intersection,
        intersection_long,
        intersection_group,
        "all_controls_rank_mae_positive",
        "all_controls_rank_mae",
    )
    if include_binary:
        intersection = add_majority_consistency(
            intersection,
            intersection_long,
            intersection_group,
            "all_controls_auc_brier_positive",
            "all_controls_auc_brier",
        )
    return long, pairwise, intersection


def completed_output_audit(manifest: dict[str, Any]) -> dict[str, Any]:
    audits = [model_output_audit(item, manifest) for item in manifest["commands"]]
    mask_digests = {item["eligible_mask_digest"] for item in audits}
    errors = [f"{item['profile_id']}: {error}" for item in audits for error in item["errors"]]
    if len(mask_digests) != 1:
        errors.append(f"eligible mask digests differ: {sorted(mask_digests)}")
    return {
        "profiles": audits,
        "eligible_mask_digests": sorted(mask_digests),
        "errors": errors,
        "passed": not errors,
    }


def sibling_preflights_pass(paths: Iterable[Path]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for path in paths:
        if not path.is_file():
            errors.append(f"missing sibling preflight: {path}")
            continue
        payload = read_json(path)
        if not payload.get("passed"):
            errors.append(f"sibling preflight did not pass: {path}")
    return not errors, errors

"""Freeze and materialize Generation 24 FreqAI robustness caches."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_freqai_cache as g23c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_common as g24c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_freeze as g24z,
)


ANALYSIS_PATH = Path(__file__).resolve()
BRANCH_ID = "g24a_freqai_activity_lead_stability_and_attribution"
CACHE_ID = "g24_freqai_activity_stability_20260828a"
RECORD_ROOT = g24z.OUTPUT_ROOT / "freqai_cache" / CACHE_ID
ARTIFACT_ROOT = (
    Path(
        "D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones/"
        "generation24_branches/g24_broad_siblings/freqai_cache"
    )
    / CACHE_ID
)
REGISTRY_PATH = RECORD_ROOT / "g24_freqai_profile_registry.json"
HORIZONS = tuple(g24z.HORIZONS_HOURS)
READY_BLOCK = g23c.READY_BLOCK
FEATURE_COLUMNS = tuple(g23c.FEATURE_COLUMNS)
LEVEL_FEATURES = tuple(g23c.LEVEL_FEATURES)
STATE_FEATURES = tuple(g23c.STATE_FEATURES)
MODEL_CELLS = (
    ("LightGBMRegressorMultiTarget", 2026082401),
    ("LightGBMRegressorMultiTarget", 2026082402),
    ("LightGBMRegressorMultiTarget", 2026082403),
    ("XGBoostRegressorMultiTarget", 2026082401),
)
PROFILE_ROLES = (
    "full_interaction",
    "level_geometry_only",
    "market_state_only",
    "within_pair_time_shuffled_training_labels",
    "permuted_level_feature_block",
)
CONTROLS = (
    "constant_training_median",
    "level_geometry_only_model",
    "market_state_only_model",
    "within_pair_time_shuffled_training_labels",
    "generation23_calibrated_parent",
    "permuted_level_feature_block",
)
PARENT_TARGETS = tuple(
    f"&-g23_percentile_{metric}_h{horizon}"
    for metric in ("future_volume_ratio", "future_range_ratio")
    for horizon in HORIZONS
)
TARGETS = tuple(target.replace("&-g23_", "&-g24_") for target in PARENT_TARGETS)
TARGET_METADATA = tuple(
    {
        "target": target,
        "parent_target": parent,
        "transform": "pair_training_empirical_percentile",
        "raw_target_name": parent.removeprefix("&-g23_percentile_"),
    }
    for target, parent in zip(TARGETS, PARENT_TARGETS, strict=True)
)


def artifact(path: Path) -> dict[str, Any]:
    return g24z.artifact(path)


def support_manifest_path(cohort: str) -> Path:
    return RECORD_ROOT / f"{cohort}_outcome_blind_support_freeze.json"


def cache_manifest_path(cohort: str) -> Path:
    return RECORD_ROOT / f"{cohort}_manifest.json"


def calibration_path(cohort: str) -> Path:
    return RECORD_ROOT / f"{cohort}_target_calibration.csv"


def load_branch() -> tuple[dict[str, Any], dict[str, Any]]:
    frozen, branch = g24c.load_branch(BRANCH_ID)
    if tuple(branch["features"]) != FEATURE_COLUMNS:
        raise ValueError("Generation 24 FreqAI feature contract drifted.")
    if tuple(branch["targets"]) != tuple(
        target.removeprefix("&-g24_percentile_") for target in TARGETS
    ):
        raise ValueError("Generation 24 FreqAI target contract drifted.")
    if branch["target_transform"] != "pair_training_empirical_percentile":
        raise ValueError("Generation 24 FreqAI target transform drifted.")
    declared_models = tuple(
        (str(item["model"]), int(seed))
        for item in branch["candidate_models"]
        for seed in item["seeds"]
    )
    if declared_models != MODEL_CELLS:
        raise ValueError("Generation 24 FreqAI model/seed registry drifted.")
    if tuple(branch["controls"]) != CONTROLS:
        raise ValueError("Generation 24 FreqAI controls drifted.")
    return frozen, branch


def model_tag(model_class: str) -> str:
    return "lgbm" if model_class.startswith("LightGBM") else "xgb"


def profile_id(cohort: str, model_class: str, seed: int, role: str) -> str:
    return f"g24__{cohort}__{model_tag(model_class)}__seed{seed}__{role}"


def build_registry() -> dict[str, Any]:
    profiles: dict[str, dict[str, Any]] = {}
    comparisons: list[dict[str, Any]] = []
    for cohort in ("normal", "meme"):
        for model_class, seed in MODEL_CELLS:
            ids: dict[str, str] = {}
            for role in PROFILE_ROLES:
                identifier = profile_id(cohort, model_class, seed, role)
                ids[role] = identifier
                if role == "level_geometry_only":
                    columns = LEVEL_FEATURES
                elif role == "market_state_only":
                    columns = STATE_FEATURES
                else:
                    columns = FEATURE_COLUMNS
                profiles[identifier] = {
                    "profile_id": identifier,
                    "cohort": cohort,
                    "role": role,
                    "model_class": model_class,
                    "seed": seed,
                    "feature_columns": list(columns),
                    "required_ready_blocks": [READY_BLOCK],
                    "targets": list(TARGETS),
                    "target_cache": (
                        "shuffled"
                        if role == "within_pair_time_shuffled_training_labels"
                        else "actual"
                    ),
                    "feature_cache": (
                        "permuted" if role == "permuted_level_feature_block" else "actual"
                    ),
                }
            for control in CONTROLS:
                baseline = {
                    "constant_training_median": None,
                    "level_geometry_only_model": ids["level_geometry_only"],
                    "market_state_only_model": ids["market_state_only"],
                    "within_pair_time_shuffled_training_labels": ids[
                        "within_pair_time_shuffled_training_labels"
                    ],
                    "generation23_calibrated_parent": None,
                    "permuted_level_feature_block": ids["permuted_level_feature_block"],
                }[control]
                comparisons.append(
                    {
                        "comparison_id": (
                            f"{cohort}__{model_tag(model_class)}__seed{seed}__vs_{control}"
                        ),
                        "cohort": cohort,
                        "model_class": model_class,
                        "seed": seed,
                        "candidate": ids["full_interaction"],
                        "control_type": control,
                        "baseline": baseline,
                        "targets": list(TARGETS),
                    }
                )
    return {
        "schema_version": 1,
        "generation": 24,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation24_freqai_target_materialization",
        "profiles": profiles,
        "comparisons": comparisons,
        "model_cells": [
            {"model_class": model_class, "seed": seed} for model_class, seed in MODEL_CELLS
        ],
        "profile_roles": list(PROFILE_ROLES),
        "feature_columns": list(FEATURE_COLUMNS),
        "level_features": list(LEVEL_FEATURES),
        "state_features": list(STATE_FEATURES),
        "targets": list(TARGETS),
        "target_metadata": list(TARGET_METADATA),
        "controls": list(CONTROLS),
        "permuted_level_control": {
            "method": "deterministic past-only whole-block shift within each pair",
            "minimum_lag_hours": 168,
            "maximum_lag_hours": 336,
            "future_feature_source_allowed": False,
        },
    }


def freeze_registry() -> dict[str, Any]:
    registry = build_registry()
    if REGISTRY_PATH.is_file():
        existing = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        left = {key: value for key, value in existing.items() if key != "created_at_utc"}
        right = {key: value for key, value in registry.items() if key != "created_at_utc"}
        if left != right:
            raise ValueError("Generation 24 FreqAI registry changed after freeze.")
        return existing
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(registry, REGISTRY_PATH)
    return registry


def verified_parent_support(cohort: str) -> dict[str, Any]:
    path = g23c.support_manifest_path(cohort)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_outcome_blind_generation23_freqai_support":
        raise ValueError(f"Generation 23 {cohort} FreqAI support is invalid.")
    for item in manifest["inventory"]:
        for kind in ("feature", "support"):
            source = Path(item[f"{kind}_path"])
            if g0.sha256_file(source) != item[f"{kind}_sha256"]:
                raise ValueError(f"Generation 23 {cohort} {kind} support changed.")
    return manifest


def permuted_level_lag(pair: str) -> int:
    return 168 + g0.stable_hash_int(f"g24-level-block-lag|{pair}") % 169


def past_shift_level_block(frame: DataFrame, pair: str) -> DataFrame:
    output = frame.copy()
    lag = permuted_level_lag(pair)
    output[list(LEVEL_FEATURES)] = output[list(LEVEL_FEATURES)].shift(lag)
    ready_column = f"ready__{READY_BLOCK}"
    output[ready_column] = output[ready_column].fillna(False).astype(bool) & output[
        list(LEVEL_FEATURES)
    ].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    return output


def freeze_support(cohort: str, *, overwrite: bool = False) -> dict[str, Any]:
    load_branch()
    freeze_registry()
    path = support_manifest_path(cohort)
    if path.is_file() and not overwrite:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "frozen_outcome_blind_generation24_freqai_support":
            raise ValueError(f"Invalid Generation 24 FreqAI support: {path}")
        return manifest
    parent = verified_parent_support(cohort)
    inventory: list[dict[str, Any]] = []
    for item in parent["inventory"]:
        pair = str(item["pair"])
        source = Path(item["feature_path"])
        if g0.sha256_file(source) != item["feature_sha256"]:
            raise ValueError(f"Generation 23 feature changed before G24 freeze: {pair}")
        feature = pd.read_parquet(source)
        permuted = past_shift_level_block(feature, pair)
        permuted_path = (
            ARTIFACT_ROOT / cohort / "permuted_feature_cache" / f"{g0.pair_file_stem(pair)}.parquet"
        )
        permuted_path.parent.mkdir(parents=True, exist_ok=True)
        g0.atomic_write_parquet(permuted, permuted_path)
        inventory.append(
            {
                "pair": pair,
                "cohort": cohort,
                "rows": int(item["rows"]),
                "ready_rows": int(item["ready_rows"]),
                "feature_path": item["feature_path"],
                "feature_sha256": item["feature_sha256"],
                "permuted_feature_path": str(permuted_path.resolve()),
                "permuted_feature_sha256": g0.sha256_file(permuted_path),
                "permuted_level_lag_hours": permuted_level_lag(pair),
                "support_path": item["support_path"],
                "support_sha256": item["support_sha256"],
                "future_outcomes_read": False,
            }
        )
    if len(inventory) != 10:
        raise ValueError(f"Expected ten {cohort} pairs, got {len(inventory)}.")
    manifest = {
        "schema_version": 1,
        "generation": 24,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_outcome_blind_generation24_freqai_support",
        "branch_id": BRANCH_ID,
        "cohort": cohort,
        "pairs": list(parent["pairs"]),
        "inventory": inventory,
        "feature_columns": list(FEATURE_COLUMNS),
        "targets_declared_without_values": list(TARGETS),
        "future_outcome_values_read": False,
        "profile_registry": artifact(REGISTRY_PATH),
        "source_contracts": {
            "generation24_freeze": artifact(g24z.FREEZE_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
            "generation23_outcome_blind_support": artifact(g23c.support_manifest_path(cohort)),
        },
    }
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(manifest, path)
    return manifest


def verified_parent_outcomes(cohort: str) -> dict[str, Any]:
    path = g23c.cache_manifest_path(cohort)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed_generation23_freqai_cache":
        raise ValueError(f"Generation 23 {cohort} FreqAI cache is not terminal.")
    for item in manifest["inventory"]:
        for kind in ("event", "shuffled_event", "evaluation"):
            source = Path(item[f"{kind}_path"])
            if g0.sha256_file(source) != item[f"{kind}_sha256"]:
                raise ValueError(f"Generation 23 {cohort} {kind} cache changed.")
    return manifest


def renamed_targets(frame: DataFrame) -> DataFrame:
    columns = ["date", "period", f"ready__{READY_BLOCK}", *PARENT_TARGETS]
    output = frame[columns].rename(columns=dict(zip(PARENT_TARGETS, TARGETS, strict=True)))
    return output[["date", "period", f"ready__{READY_BLOCK}", *TARGETS]]


def materialize_targets(cohort: str, *, overwrite: bool = False) -> dict[str, Any]:
    support = freeze_support(cohort, overwrite=False)
    sibling_contracts = g24c.require_all_frozen_supports()
    path = cache_manifest_path(cohort)
    if path.is_file() and not overwrite:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("status") != "completed_generation24_freqai_cache":
            raise ValueError(f"Invalid Generation 24 FreqAI cache: {path}")
        return existing
    parent = verified_parent_outcomes(cohort)
    parent_by_pair = {str(item["pair"]): item for item in parent["inventory"]}
    completed: list[dict[str, Any]] = []
    root = ARTIFACT_ROOT / cohort
    for number, support_item in enumerate(support["inventory"], start=1):
        pair = str(support_item["pair"])
        parent_item = parent_by_pair[pair]
        actual = renamed_targets(pd.read_parquet(parent_item["event_path"]))
        shuffled = renamed_targets(pd.read_parquet(parent_item["shuffled_event_path"]))
        stem = g0.pair_file_stem(pair)
        event_path = root / "event_cache" / f"{stem}.parquet"
        shuffled_path = root / "shuffled_event_cache" / f"{stem}.parquet"
        evaluation_path = root / "evaluation_cache" / f"{stem}.parquet"
        for output in (event_path, shuffled_path, evaluation_path):
            output.parent.mkdir(parents=True, exist_ok=True)
        g0.atomic_write_parquet(actual, event_path)
        g0.atomic_write_parquet(shuffled, shuffled_path)
        g0.atomic_write_parquet(actual[["date", "period", *TARGETS]], evaluation_path)
        completed.append(
            {
                **support_item,
                "event_path": str(event_path.resolve()),
                "event_sha256": g0.sha256_file(event_path),
                "shuffled_event_path": str(shuffled_path.resolve()),
                "shuffled_event_sha256": g0.sha256_file(shuffled_path),
                "evaluation_path": str(evaluation_path.resolve()),
                "evaluation_sha256": g0.sha256_file(evaluation_path),
                "event_rows": len(actual),
                "targets_opened_after_all_five_sibling_supports_frozen": True,
            }
        )
        print(
            json.dumps(
                {
                    "phase": "g24_freqai_targets",
                    "cohort": cohort,
                    "processed": number,
                    "total": len(support["inventory"]),
                }
            ),
            flush=True,
        )
    parent_calibration = pd.read_csv(g23c.calibration_path(cohort))
    wanted = {item["raw_target_name"] for item in TARGET_METADATA}
    calibration = parent_calibration.loc[
        parent_calibration["raw_target"].astype(str).map(g23c.raw_target_name).isin(wanted)
    ].copy()
    calibration_file = calibration_path(cohort)
    g0.atomic_write_csv(calibration, calibration_file)
    manifest = {
        "schema_version": 1,
        "generation": 24,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation24_freqai_cache",
        "cohort": cohort,
        "pairs": list(support["pairs"]),
        "inventory": completed,
        "targets": list(TARGETS),
        "target_metadata": list(TARGET_METADATA),
        "profile_registry": artifact(REGISTRY_PATH),
        "target_calibration": artifact(calibration_file),
        "outcome_blind_support_freeze": artifact(support_manifest_path(cohort)),
        "all_sibling_support_contracts": sibling_contracts,
        "generation23_parent_target_cache": artifact(g23c.cache_manifest_path(cohort)),
        "integrity": {
            "features_frozen_before_targets": True,
            "all_five_sibling_supports_frozen_before_targets": True,
            "permuted_level_block_uses_past_only": True,
            "profit_used": False,
            "future_signed_direction_used": False,
        },
    }
    g0.atomic_write_json(manifest, path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("normal", "meme", "all"), default="all")
    parser.add_argument("--prepare-support", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    cohorts = ("normal", "meme") if args.cohort == "all" else (args.cohort,)
    outputs = []
    for cohort in cohorts:
        if args.prepare_support:
            outputs.append(freeze_support(cohort, overwrite=args.overwrite))
        else:
            outputs.append(materialize_targets(cohort, overwrite=args.overwrite))
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Freeze and materialize Generation 23's calibrated unsigned FreqAI cache."""

from __future__ import annotations

# Bound numerical pools before pandas/numpy imports.
# ruff: noqa: E402
import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation18_freeze as g18z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_freqai_cache as g22c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_freeze as g23z,
)


ANALYSIS_PATH = Path(__file__).resolve()
BRANCH_ID = "g23c_freqai_calibrated_unsigned_reaction"
CACHE_ID = "g23_freqai_calibrated_unsigned_reaction_20260828a"
RECORD_ROOT = g23z.OUTPUT_ROOT / "freqai_cache" / CACHE_ID
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation23_branches"
    / "g23_broad_siblings"
    / "freqai_cache"
    / CACHE_ID
)
REGISTRY_PATH = RECORD_ROOT / "g23_freqai_profile_registry.json"
SEED = 2026082823
HORIZONS = tuple(g22c.HORIZONS)
READY_BLOCK = g22c.READY_BLOCK
FEATURE_COLUMNS = tuple(g22c.FEATURE_COLUMNS)
LEVEL_FEATURES = tuple(g22c.LEVEL_FEATURES)
STATE_FEATURES = tuple(g22c.STATE_FEATURES)
RAW_TARGETS = tuple(g22c.TARGETS)
TRANSFORMS = (
    "pair_training_median_residual",
    "pair_training_empirical_percentile",
)
PROFILE_ROLES = tuple(g22c.PROFILE_ROLES)
TARGET_PURGE_HOURS = max(HORIZONS)
TRAIN_DAYS = {"normal": 365, "meme": 160}


def raw_target_name(raw_target: str) -> str:
    prefix = "&-g22_"
    if not raw_target.startswith(prefix):
        raise ValueError(f"Unexpected Generation 22 target: {raw_target}")
    return raw_target.removeprefix(prefix)


def target_name(transform: str, raw_target: str) -> str:
    short = {
        "pair_training_median_residual": "residual",
        "pair_training_empirical_percentile": "percentile",
    }.get(transform)
    if short is None:
        raise ValueError(f"Unknown Generation 23 target transform: {transform}")
    return f"&-g23_{short}_{raw_target_name(raw_target)}"


TARGET_METADATA = tuple(
    {
        "target": target_name(transform, raw_target),
        "transform": transform,
        "raw_target": raw_target,
        "raw_target_name": raw_target_name(raw_target),
    }
    for transform in TRANSFORMS
    for raw_target in RAW_TARGETS
)
TARGETS = tuple(str(item["target"]) for item in TARGET_METADATA)


def artifact(path: Path) -> dict[str, Any]:
    return g23z.artifact(path)


def support_manifest_path(cohort: str) -> Path:
    return RECORD_ROOT / f"{cohort}_outcome_blind_support_freeze.json"


def cache_manifest_path(cohort: str) -> Path:
    return RECORD_ROOT / f"{cohort}_manifest.json"


def calibration_path(cohort: str) -> Path:
    return RECORD_ROOT / f"{cohort}_target_calibration.csv"


def load_branch() -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = json.loads(g23z.FREEZE_PATH.read_text(encoding="utf-8"))
    branch = next(
        item for item in frozen["branches"] if item["branch_id"] == BRANCH_ID
    )
    if frozen.get("status") != "frozen_before_generation23_outcomes":
        raise ValueError("Generation 23 batch is not frozen.")
    if tuple(branch["features"]) != FEATURE_COLUMNS:
        raise ValueError("Generation 23 FreqAI feature contract drifted.")
    expected_raw = tuple(raw_target_name(target) for target in RAW_TARGETS)
    if tuple(branch["raw_targets"]) != expected_raw:
        raise ValueError("Generation 23 FreqAI raw-target contract drifted.")
    if tuple(branch["target_transforms"]) != TRANSFORMS:
        raise ValueError("Generation 23 FreqAI transform contract drifted.")
    if branch.get("required_tool") != "FreqAI":
        raise ValueError("Generation 23 FreqAI route lost its required tool.")
    return frozen, branch


def profile_id(cohort: str, role: str) -> str:
    return f"g23__{cohort}__{role}__seed{SEED}"


def build_registry() -> dict[str, Any]:
    profiles: dict[str, dict[str, Any]] = {}
    comparisons: list[dict[str, Any]] = []
    for cohort in ("normal", "meme"):
        for role in PROFILE_ROLES:
            identifier = profile_id(cohort, role)
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
                "feature_columns": list(columns),
                "required_ready_blocks": [READY_BLOCK],
                "targets": list(TARGETS),
                "seed": SEED,
                "target_cache": (
                    "shuffled"
                    if role == "within_pair_time_shuffled_training_labels"
                    else "actual"
                ),
            }
        candidate = profile_id(cohort, "full_interaction")
        for control in (
            "constant_training_median",
            "level_geometry_only_model",
            "market_state_only_model",
            "within_pair_time_shuffled_training_labels",
            "generation22_uncalibrated_parent",
        ):
            baseline_role = control.removesuffix("_model")
            comparisons.append(
                {
                    "comparison_id": f"{cohort}__full_vs_{control}",
                    "cohort": cohort,
                    "route_id": "full_interaction",
                    "control_type": control,
                    "candidate": candidate,
                    "baseline": (
                        None
                        if control in {
                            "constant_training_median",
                            "generation22_uncalibrated_parent",
                        }
                        else profile_id(cohort, baseline_role)
                    ),
                    "targets": list(TARGETS),
                    "expected_controls_for_route": 5,
                }
            )
    return {
        "schema_version": 1,
        "generation": 23,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation23_freqai_target_materialization",
        "profiles": profiles,
        "comparisons": comparisons,
        "feature_columns": list(FEATURE_COLUMNS),
        "raw_targets": list(RAW_TARGETS),
        "target_metadata": list(TARGET_METADATA),
        "targets": list(TARGETS),
        "target_transforms": list(TRANSFORMS),
        "calibration_rule": (
            "For each pair and raw target, fit one median and empirical distribution "
            "using only the frozen training window ending eight hours before the first "
            "later evaluation block. Reuse that calibration in both later blocks."
        ),
        "shuffled_label_control": {
            "method": "reuse Generation 22 deterministic earlier-date within-pair labels",
            "future_label_source_allowed": False,
        },
    }


def freeze_registry() -> dict[str, Any]:
    registry = build_registry()
    if REGISTRY_PATH.is_file():
        existing = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        left = {key: value for key, value in existing.items() if key != "created_at_utc"}
        right = {key: value for key, value in registry.items() if key != "created_at_utc"}
        if left != right:
            raise ValueError("Generation 23 FreqAI registry changed after its freeze.")
        return existing
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(registry, REGISTRY_PATH)
    return registry


def _verified_parent_support(cohort: str) -> dict[str, Any]:
    path = g22c.support_manifest_path(cohort)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_outcome_blind_generation22_freqai_support":
        raise ValueError(f"Generation 22 {cohort} FreqAI support is not frozen.")
    for item in manifest["inventory"]:
        for kind in ("feature", "support"):
            source = Path(item[f"{kind}_path"])
            if g0.sha256_file(source) != item[f"{kind}_sha256"]:
                raise ValueError(f"Generation 22 {cohort} {kind} support changed: {source}")
    return manifest


def freeze_support(cohort: str, *, overwrite: bool = False) -> dict[str, Any]:
    load_branch()
    freeze_registry()
    path = support_manifest_path(cohort)
    if path.is_file() and not overwrite:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "frozen_outcome_blind_generation23_freqai_support":
            raise ValueError(f"Invalid Generation 23 FreqAI support freeze: {path}")
        return manifest
    parent = _verified_parent_support(cohort)
    inventory = [
        {
            "pair": item["pair"],
            "cohort": item["cohort"],
            "rows": item["rows"],
            "ready_rows": item["ready_rows"],
            "feature_path": item["feature_path"],
            "feature_sha256": item["feature_sha256"],
            "support_path": item["support_path"],
            "support_sha256": item["support_sha256"],
            "future_outcomes_read": False,
        }
        for item in parent["inventory"]
    ]
    if len(inventory) != 10:
        raise ValueError(f"Expected ten {cohort} pairs, got {len(inventory)}.")
    manifest = {
        "schema_version": 1,
        "generation": 23,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_outcome_blind_generation23_freqai_support",
        "branch_id": BRANCH_ID,
        "cohort": cohort,
        "pairs": list(parent["pairs"]),
        "inventory": inventory,
        "feature_columns": list(FEATURE_COLUMNS),
        "targets_declared_without_values": list(TARGETS),
        "future_outcome_values_read": False,
        "profile_registry": artifact(REGISTRY_PATH),
        "source_contracts": {
            "generation23_freeze": artifact(g23z.FREEZE_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
            "generation22_outcome_blind_support": artifact(
                g22c.support_manifest_path(cohort)
            ),
        },
    }
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(manifest, path)
    return manifest


def sibling_support_contracts() -> tuple[tuple[Path, str], ...]:
    from user_data.Custom_Launcher.research.context_features import (
        market_reaction_zone_generation23_common as common,
    )

    return common.all_support_contracts()


def require_all_sibling_supports() -> list[dict[str, Any]]:
    from user_data.Custom_Launcher.research.context_features import (
        market_reaction_zone_generation23_common as common,
    )

    return common.require_all_frozen_supports()


def calibration_bounds(cohort: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    periods = list(g18z.CONFIRMATION_PERIODS[cohort])
    first_start = pd.Timestamp(periods[0]["start_utc"])
    cutoff = first_start - pd.Timedelta(hours=TARGET_PURGE_HOURS)
    start = first_start - pd.Timedelta(days=TRAIN_DAYS[cohort])
    return start, cutoff


def empirical_percentile(values: Series, reference: Series) -> Series:
    clean_reference = (
        pd.to_numeric(reference, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    output = np.full(len(numeric), np.nan, dtype=float)
    if not len(clean_reference):
        return Series(output, index=values.index)
    ordered = np.sort(clean_reference)
    valid = np.isfinite(numeric)
    left = np.searchsorted(ordered, numeric[valid], side="left")
    right = np.searchsorted(ordered, numeric[valid], side="right")
    output[valid] = (left + right) / (2.0 * len(ordered))
    return Series(output, index=values.index)


def transform_targets(
    actual: DataFrame,
    source: DataFrame,
    *,
    cohort: str,
    pair: str,
) -> tuple[DataFrame, list[dict[str, Any]]]:
    start, cutoff = calibration_bounds(cohort)
    dates = pd.to_datetime(actual["date"], utc=True, errors="raise")
    ready = actual[f"ready__{READY_BLOCK}"].fillna(False).astype(bool)
    calibration_mask = ready & dates.ge(start) & dates.lt(cutoff)
    output = source[["date", "period", f"ready__{READY_BLOCK}"]].copy()
    rows: list[dict[str, Any]] = []
    for raw_target in RAW_TARGETS:
        reference = pd.to_numeric(
            actual.loc[calibration_mask, raw_target], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if reference.empty:
            raise ValueError(f"No calibration values for {pair}/{raw_target}.")
        median = float(reference.median())
        raw_values = pd.to_numeric(source[raw_target], errors="coerce")
        residual_target = target_name("pair_training_median_residual", raw_target)
        percentile_target = target_name(
            "pair_training_empirical_percentile", raw_target
        )
        output[residual_target] = raw_values - median
        output[percentile_target] = empirical_percentile(raw_values, reference)
        rows.append(
            {
                "cohort": cohort,
                "pair": pair,
                "raw_target": raw_target,
                "calibration_start_utc": start,
                "calibration_cutoff_exclusive_utc": cutoff,
                "calibration_rows": len(reference),
                "training_median": median,
                "training_minimum": float(reference.min()),
                "training_maximum": float(reference.max()),
            }
        )
    return output[["date", "period", f"ready__{READY_BLOCK}", *TARGETS]], rows


def _verified_parent_outcomes(cohort: str) -> dict[str, Any]:
    path = g22c.cache_manifest_path(cohort)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed_generation22_freqai_cache":
        raise ValueError(f"Generation 22 {cohort} FreqAI cache is not terminal.")
    for item in manifest["inventory"]:
        for kind in ("event", "shuffled_event"):
            source = Path(item[f"{kind}_path"])
            if g0.sha256_file(source) != item[f"{kind}_sha256"]:
                raise ValueError(f"Generation 22 {cohort} {kind} cache changed: {source}")
    return manifest


def materialize_targets(cohort: str, *, overwrite: bool = False) -> dict[str, Any]:
    support = freeze_support(cohort, overwrite=False)
    for key, source in (
        ("generation23_freeze", g23z.FREEZE_PATH),
        ("analysis_script", ANALYSIS_PATH),
        ("generation22_outcome_blind_support", g22c.support_manifest_path(cohort)),
    ):
        if g0.sha256_file(source) != support["source_contracts"][key]["sha256"]:
            raise ValueError(f"Frozen Generation 23 FreqAI support dependency changed: {key}")
    sibling_contracts = require_all_sibling_supports()
    path = cache_manifest_path(cohort)
    if path.is_file() and not overwrite:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("status") != "completed_generation23_freqai_cache":
            raise ValueError(f"Invalid Generation 23 FreqAI cache: {path}")
        return existing
    parent = _verified_parent_outcomes(cohort)
    parent_by_pair = {str(item["pair"]): item for item in parent["inventory"]}
    completed: list[dict[str, Any]] = []
    calibrations: list[dict[str, Any]] = []
    for number, support_item in enumerate(support["inventory"], start=1):
        pair = str(support_item["pair"])
        parent_item = parent_by_pair[pair]
        actual_raw = pd.read_parquet(parent_item["event_path"])
        shuffled_raw = pd.read_parquet(parent_item["shuffled_event_path"])
        actual, rows = transform_targets(
            actual_raw, actual_raw, cohort=cohort, pair=pair
        )
        shuffled, _ = transform_targets(
            actual_raw, shuffled_raw, cohort=cohort, pair=pair
        )
        calibrations.extend(rows)
        root = ARTIFACT_ROOT / cohort
        stem = g0.pair_file_stem(pair)
        event_path = root / "event_cache" / f"{stem}.parquet"
        shuffled_path = root / "shuffled_event_cache" / f"{stem}.parquet"
        evaluation_path = root / "evaluation_cache" / f"{stem}.parquet"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        shuffled_path.parent.mkdir(parents=True, exist_ok=True)
        evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        g0.atomic_write_parquet(actual, event_path)
        g0.atomic_write_parquet(shuffled, shuffled_path)
        g0.atomic_write_parquet(
            actual[["date", "period", *TARGETS]], evaluation_path
        )
        completed.append(
            {
                **support_item,
                "event_path": str(event_path.resolve()),
                "event_sha256": g0.sha256_file(event_path),
                "evaluation_path": str(evaluation_path.resolve()),
                "evaluation_sha256": g0.sha256_file(evaluation_path),
                "shuffled_event_path": str(shuffled_path.resolve()),
                "shuffled_event_sha256": g0.sha256_file(shuffled_path),
                "event_rows": len(actual),
                "targets_opened_after_all_five_sibling_supports_frozen": True,
            }
        )
        print(
            json.dumps(
                {
                    "phase": "g23_freqai_targets",
                    "cohort": cohort,
                    "processed": number,
                    "total": len(support["inventory"]),
                }
            ),
            flush=True,
        )
    calibration_file = calibration_path(cohort)
    g0.atomic_write_csv(DataFrame.from_records(calibrations), calibration_file)
    manifest = {
        "schema_version": 1,
        "generation": 23,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation23_freqai_cache",
        "cohort": cohort,
        "pairs": list(support["pairs"]),
        "inventory": completed,
        "targets": list(TARGETS),
        "target_metadata": list(TARGET_METADATA),
        "profile_registry": artifact(REGISTRY_PATH),
        "target_calibration": artifact(calibration_file),
        "outcome_blind_support_freeze": artifact(support_manifest_path(cohort)),
        "all_sibling_support_contracts": sibling_contracts,
        "generation22_raw_target_cache": artifact(g22c.cache_manifest_path(cohort)),
        "integrity": {
            "features_frozen_before_targets": True,
            "all_five_sibling_supports_frozen_before_targets": True,
            "one_pre_evaluation_calibration_reused_across_both_later_blocks": True,
            "shuffled_source_dates_always_earlier": True,
            "profit_used": False,
            "future_signed_direction_used": False,
        },
    }
    g0.atomic_write_json(manifest, path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("all", "normal", "meme"), default="all")
    parser.add_argument("--phase", choices=("support", "targets", "all"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    cohorts = ("normal", "meme") if args.cohort == "all" else (args.cohort,)
    results: list[dict[str, Any]] = []
    if args.phase in {"support", "all"}:
        results.extend(freeze_support(cohort, overwrite=args.overwrite) for cohort in cohorts)
    if args.phase in {"targets", "all"}:
        results.extend(materialize_targets(cohort, overwrite=args.overwrite) for cohort in cohorts)
    print(
        json.dumps(
            [
                {
                    "status": item["status"],
                    "cohort": item["cohort"],
                    "pairs": len(item["pairs"]),
                }
                for item in results
            ],
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

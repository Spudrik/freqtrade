"""Freeze and materialize Generation 19's direction-neutral FreqAI combination cache."""

from __future__ import annotations

# Bound dataframe/parquet pools before importing numerical libraries.
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
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation16_freqai_cache as g16c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_freqai_cache as g17c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation19_freeze as g19z,
)


CACHE_ID = "g19_freqai_combinations_20260823a"
RECORD_ROOT = g19z.OUTPUT_ROOT / "freqai_cache" / CACHE_ID
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation19_branches"
    / "g19_broad_combinations"
    / "freqai_cache"
    / CACHE_ID
)
REGISTRY_PATH = RECORD_ROOT / "g19_freqai_profile_registry.json"
SEED = 2026082319
HORIZONS = tuple(g19z.HORIZONS_HOURS)
TARGETS = tuple(
    column
    for horizon in HORIZONS
    for column in (
        f"&-g19_future_volume_ratio_h{horizon}",
        f"&-g19_crossing_count_h{horizon}",
    )
)

LOCAL_TREND = g17c.LOCAL_TREND
LOCAL_ACTIVITY = g17c.LOCAL_ACTIVITY
LEVEL = g17c.MINIMAL
STALE_LEVEL = g17c.STALE_MINIMAL
WIDER = g17c.WIDER
DISPERSION_COLUMN = f"{WIDER}__cohort_dispersion"

PROFILE_SPECS: dict[str, dict[str, tuple[str, ...]]] = {
    "price_and_indicator_baseline": {
        "parts": ("trend",),
        "ready": (LOCAL_TREND,),
    },
    "level_only": {
        "parts": ("trend", "level"),
        "ready": (LOCAL_TREND, LEVEL),
    },
    "dispersion_only": {
        "parts": ("trend", "dispersion"),
        "ready": (LOCAL_TREND, WIDER),
    },
    "local_activity_only": {
        "parts": ("trend", "local_activity"),
        "ready": (LOCAL_TREND, LOCAL_ACTIVITY),
    },
    "level_plus_dispersion": {
        "parts": ("trend", "level", "dispersion"),
        "ready": (LOCAL_TREND, LEVEL, WIDER),
    },
    "level_plus_local_activity": {
        "parts": ("trend", "level", "local_activity"),
        "ready": (LOCAL_TREND, LEVEL, LOCAL_ACTIVITY),
    },
    "stale_level_plus_dispersion": {
        "parts": ("trend", "stale_level", "dispersion"),
        "ready": (LOCAL_TREND, STALE_LEVEL, WIDER),
    },
    "stale_level_plus_local_activity": {
        "parts": ("trend", "stale_level", "local_activity"),
        "ready": (LOCAL_TREND, STALE_LEVEL, LOCAL_ACTIVITY),
    },
}


def artifact(path: Path) -> dict[str, Any]:
    return g19z.artifact(path)


def load_freeze() -> dict[str, Any]:
    frozen = json.loads(g19z.FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation19_outcomes":
        raise ValueError("Generation 19 branch layer is not frozen.")
    branch = next(
        item
        for item in frozen["branches"]
        if item["branch_id"] == "g19b_regime_level_incremental_combinations"
    )
    if set(branch["profiles"]) != set(PROFILE_SPECS):
        raise ValueError("Generation 19 FreqAI profiles drifted after the freeze.")
    if tuple(branch["targets"]) != tuple(
        f"{metric}_h{horizon}"
        for metric in ("future_volume_ratio", "crossing_count")
        for horizon in HORIZONS
    ):
        raise ValueError("Generation 19 FreqAI target surface drifted after the freeze.")
    return frozen


def load_g17_manifest(cohort: str) -> tuple[dict[str, Any], Path]:
    path = g17c.RECORD_ROOT / f"{cohort}_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed_generation17_freqai_cache":
        raise ValueError(f"Generation 17 {cohort} cache is not terminal.")
    return manifest, path


def prefixed(columns: Sequence[str], block: str) -> tuple[str, ...]:
    selected = tuple(column for column in columns if column.startswith(f"{block}__"))
    if not selected:
        raise ValueError(f"No feature columns found for block {block!r}.")
    return selected


def part_columns(columns: Sequence[str], part: str) -> tuple[str, ...]:
    if part == "dispersion":
        if DISPERSION_COLUMN not in columns:
            raise ValueError(f"Missing cross-coin dispersion feature {DISPERSION_COLUMN!r}.")
        return (DISPERSION_COLUMN,)
    blocks = {
        "trend": LOCAL_TREND,
        "level": LEVEL,
        "stale_level": STALE_LEVEL,
        "local_activity": LOCAL_ACTIVITY,
    }
    if part not in blocks:
        raise ValueError(f"Unknown Generation 19 feature part: {part}")
    return prefixed(columns, blocks[part])


def profile_id(cohort: str, role: str) -> str:
    return f"g19__{cohort}__{role}__seed{SEED}"


def build_registry(feature_columns: Sequence[str]) -> dict[str, Any]:
    profiles: dict[str, dict[str, Any]] = {}
    comparisons: list[dict[str, Any]] = []
    for cohort in ("normal", "meme"):
        for role, spec in PROFILE_SPECS.items():
            columns = [
                column
                for part in spec["parts"]
                for column in part_columns(feature_columns, part)
            ]
            identifier = profile_id(cohort, role)
            profiles[identifier] = {
                "profile_id": identifier,
                "cohort": cohort,
                "role": role,
                "seed": SEED,
                "feature_columns": list(dict.fromkeys(columns)),
                "required_ready_blocks": list(spec["ready"]),
                "targets": list(TARGETS),
            }

        def add(
            route_id: str,
            question: str,
            candidate: str,
            controls: Sequence[tuple[str, str]],
            cohort_name: str = cohort,
        ) -> None:
            for control_type, baseline in controls:
                comparisons.append(
                    {
                        "comparison_id": (
                            f"g19__{cohort_name}__{route_id}__vs_{control_type}"
                            f"__seed{SEED}"
                        ),
                        "cohort": cohort_name,
                        "branch_id": "g19b_regime_level_incremental_combinations",
                        "question_id": route_id,
                        "route_id": route_id,
                        "route_type": "bounded_incremental_information",
                        "plain_question": question,
                        "candidate": profile_id(cohort_name, candidate),
                        "baseline": profile_id(cohort_name, baseline),
                        "control_type": control_type,
                        "expected_controls_for_route": len(controls),
                        "seed": SEED,
                        "targets": list(TARGETS),
                    }
                )

        add(
            "level_plus_dispersion",
            "Do current levels plus cross-coin dispersion beat every simpler component?",
            "level_plus_dispersion",
            (
                ("price_and_indicators", "price_and_indicator_baseline"),
                ("level_only", "level_only"),
                ("dispersion_only", "dispersion_only"),
                ("causal_72h_old_level", "stale_level_plus_dispersion"),
            ),
        )
        add(
            "level_plus_local_activity",
            "Do current levels plus local activity beat every simpler component?",
            "level_plus_local_activity",
            (
                ("price_and_indicators", "price_and_indicator_baseline"),
                ("level_only", "level_only"),
                ("local_activity_only", "local_activity_only"),
                ("causal_72h_old_level", "stale_level_plus_local_activity"),
            ),
        )
    return {
        "schema_version": 1,
        "generation": 19,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation19_freqai_target_materialization",
        "seed": SEED,
        "profiles": profiles,
        "comparisons": comparisons,
        "targets": list(TARGETS),
        "horizons_hours": list(HORIZONS),
        "maximum_active_interaction_blocks": 2,
        "price_indicator_baseline_definition": (
            "completed 1h RSI, MACD, ADX, moving-average separation, slopes, and returns"
        ),
        "cross_coin_context_definition": "cohort dispersion only",
        "profit_used": False,
        "future_signed_direction_used": False,
    }


def feature_inventory(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair": item["pair"],
        "source_path": item["source_path"],
        "source_sha256": item["source_sha256"],
        "feature_path": item["feature_path"],
        "feature_sha256": item["feature_sha256"],
        "support_path": item["support_path"],
        "support_sha256": item["support_sha256"],
        "parent_evaluation_path": item["evaluation_path"],
        "parent_evaluation_sha256": item["evaluation_sha256"],
        "rows": item["rows"],
        "future_outcome_values_read": False,
    }


def target_frame(item: dict[str, Any]) -> DataFrame:
    metadata = [
        "date",
        "period",
        "pair",
        "cohort",
        "market_group",
        "smart_contract_platform",
        "anchor_level_family",
        "anchor_level_name",
        "anchor_source_timeframe",
        "anchor_representation",
    ]
    anchors = pd.read_parquet(item["parent_evaluation_path"], columns=metadata)
    anchors["date"] = pd.to_datetime(anchors["date"], utc=True, errors="raise")
    source_columns = [
        *g16c.SOURCE_KEYS,
        "control",
        *(f"volume_ratio_h{horizon}" for horizon in HORIZONS),
        *(f"crossings_h{horizon}" for horizon in HORIZONS),
    ]
    source = pd.read_parquet(item["source_path"], columns=source_columns)
    source = source.loc[source["control"].eq("actual")].drop(columns="control")
    source["event_time"] = pd.to_datetime(source["event_time"], utc=True, errors="raise")
    if source.duplicated(list(g16c.SOURCE_KEYS)).any():
        raise ValueError(f"Duplicate exact actual anchors in {item['source_path']}.")
    merged = anchors.merge(
        source,
        left_on=list(g16c.ANCHOR_KEYS),
        right_on=list(g16c.SOURCE_KEYS),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if merged["_merge"].ne("both").any():
        raise ValueError(f"Unmatched Generation 19 target anchors for {item['pair']}.")
    output = anchors.copy()
    for horizon in HORIZONS:
        output[f"&-g19_future_volume_ratio_h{horizon}"] = pd.to_numeric(
            merged[f"volume_ratio_h{horizon}"], errors="coerce"
        ).to_numpy()
        output[f"&-g19_crossing_count_h{horizon}"] = pd.to_numeric(
            merged[f"crossings_h{horizon}"], errors="coerce"
        ).to_numpy()
    return output


def materialize_targets(item: dict[str, Any], cohort: str) -> dict[str, Any]:
    targets = target_frame(item)
    support = pd.read_parquet(item["support_path"])
    support["date"] = pd.to_datetime(support["date"], utc=True, errors="raise")
    ready = [column for column in support if column.startswith("ready__")]
    evaluation = targets.merge(
        support[["date", *ready]], on="date", how="inner", validate="one_to_one"
    )
    event = evaluation.drop(
        columns=[
            "anchor_level_family",
            "anchor_level_name",
            "anchor_source_timeframe",
            "anchor_representation",
        ]
    )
    stem = g0.pair_file_stem(item["pair"])
    event_path = ARTIFACT_ROOT / cohort / "event_cache" / f"{stem}.parquet"
    evaluation_path = ARTIFACT_ROOT / cohort / "evaluation_cache" / f"{stem}.parquet"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(event, event_path)
    g0.atomic_write_parquet(evaluation, evaluation_path)
    return {
        **item,
        "event_path": str(event_path.resolve()),
        "event_sha256": g0.sha256_file(event_path),
        "evaluation_path": str(evaluation_path.resolve()),
        "evaluation_sha256": g0.sha256_file(evaluation_path),
        "event_rows": len(event),
        "targets_opened_after_support_freeze": True,
    }


def build_cohort(cohort: str, *, overwrite: bool = False) -> dict[str, Any]:
    load_freeze()
    manifest_path = RECORD_ROOT / f"{cohort}_manifest.json"
    if manifest_path.is_file() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed_generation19_freqai_cache":
            return existing
        raise ValueError(f"Incomplete Generation 19 cache exists: {manifest_path}")
    parent, parent_path = load_g17_manifest(cohort)
    feature_columns = pd.read_parquet(parent["inventory"][0]["feature_path"]).columns
    registry = build_registry(feature_columns)
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    if REGISTRY_PATH.is_file():
        current = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        left = {key: value for key, value in current.items() if key != "created_at_utc"}
        right = {key: value for key, value in registry.items() if key != "created_at_utc"}
        if left != right:
            raise ValueError("Generation 19 registry changed after its first freeze.")
    else:
        g0.atomic_write_json(registry, REGISTRY_PATH)
    inventory = [feature_inventory(item) for item in parent["inventory"]]
    support_freeze = {
        "schema_version": 1,
        "generation": 19,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_outcome_blind_support_before_target_materialization",
        "cohort": cohort,
        "generation19_freeze": artifact(g19z.FREEZE_PATH),
        "generation17_cache_manifest": artifact(parent_path),
        "profile_registry": artifact(REGISTRY_PATH),
        "inventory": [
            {
                key: value
                for key, value in item.items()
                if key
                not in {
                    "source_path",
                    "source_sha256",
                    "parent_evaluation_path",
                    "parent_evaluation_sha256",
                }
            }
            for item in inventory
        ],
        "future_outcome_values_read": False,
    }
    support_freeze_path = RECORD_ROOT / f"{cohort}_outcome_blind_support_freeze.json"
    g0.atomic_write_json(support_freeze, support_freeze_path)
    completed = [materialize_targets(item, cohort) for item in inventory]
    result = {
        "schema_version": 1,
        "generation": 19,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation19_freqai_cache",
        "cohort": cohort,
        "pairs": list(parent["pairs"]),
        "inventory": completed,
        "profile_registry": artifact(REGISTRY_PATH),
        "outcome_blind_support_freeze": artifact(support_freeze_path),
        "source_contracts": {
            "generation19_freeze": artifact(g19z.FREEZE_PATH),
            "generation17_cache_manifest": artifact(parent_path),
        },
        "integrity": {
            "profile_registry_frozen_before_targets": True,
            "identical_parent_feature_cache_reused": True,
            "targets_joined_after_support_freeze": True,
            "profit_used": False,
            "future_signed_direction_used": False,
        },
    }
    g0.atomic_write_json(result, manifest_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("all", "normal", "meme"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    cohorts = ("normal", "meme") if args.cohort == "all" else (args.cohort,)
    results = [build_cohort(cohort, overwrite=args.overwrite) for cohort in cohorts]
    print(
        json.dumps(
            [
                {
                    "cohort": item["cohort"],
                    "status": item["status"],
                    "pairs": len(item["pairs"]),
                    "rows": sum(entry["rows"] for entry in item["inventory"]),
                }
                for item in results
            ],
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

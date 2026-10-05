"""Build Generation 17's outcome-blind support and direction-neutral targets."""

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
    market_reaction_zone_generation16_freqai_cache as g16c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_freeze as g17z,
)


CACHE_ID = "g17_freqai_support_20260822a"
RECORD_ROOT = g17z.OUTPUT_ROOT / "freqai_cache" / CACHE_ID
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation17_branches"
    / "g17_broad_branch_layer"
    / "freqai_cache"
    / CACHE_ID
)
REGISTRY_PATH = RECORD_ROOT / "g17_freqai_profile_registry.json"
SEED = 2026082217
HORIZONS = (1, 2, 4)

CROSSING_TARGETS = tuple(
    column
    for horizon in HORIZONS
    for column in (
        f"&-g17_crossing_count_h{horizon}",
        f"&-g17_repeated_recross_h{horizon}",
        f"&-g17_two_sided_traversal_h{horizon}",
    )
)
REACTION_TARGETS = tuple(f"&-g17_reaction_h{horizon}" for horizon in HORIZONS)
TARGETS = (*CROSSING_TARGETS, *REACTION_TARGETS)

CONTACT = g16c.CONTACT
MINIMAL = g16c.MINIMAL
GEOMETRY = g16c.GEOMETRY
LOCAL_ACTIVITY = "g11_local_activity_volatility"
LOCAL_TREND = "g11_local_trend_momentum"
WIDER = g16c.WIDER
EIGHT_HOUR_ACTIVITY = g16c.EIGHT_HOUR
EIGHT_HOUR_TREND = "g13_mtf_8h_trend_momentum"
ORDERBOOK = "g11_orderbook_context"
STALE_MINIMAL = f"{MINIMAL}_stale"
STALE_GEOMETRY = f"{GEOMETRY}_stale"
SHUFFLED_MINIMAL = f"{MINIMAL}_shuffled"
SHUFFLED_GEOMETRY = f"{GEOMETRY}_shuffled"
STALE_ORDERBOOK = f"{ORDERBOOK}_stale"
SHUFFLED_ORDERBOOK = f"{ORDERBOOK}_shuffled"

CONTACT_COLUMNS = g16c.CONTACT_COLUMNS
COUNT_SUFFIXES = (
    "contacted_level_count",
    "distinct_family_count",
    "distinct_timeframe_count",
    "distinct_mechanism_count",
)
PROXIMITY_SUFFIXES = (
    "mean_zone_half_width_atr",
    "minimum_contact_distance_atr",
    "anchor_zone_half_width_atr",
    "anchor_pre_distance_atr",
    "anchor_contact_distance_atr",
)

PROFILE_SPECS: dict[str, dict[str, Any]] = {
    "contact_only": {"parts": ("contact",), "ready": (CONTACT,)},
    "contact_plus_density_counts": {
        "parts": ("contact", "counts"),
        "ready": (CONTACT, MINIMAL),
    },
    "contact_plus_proximity_width": {
        "parts": ("contact", "proximity"),
        "ready": (CONTACT, MINIMAL),
    },
    "contact_plus_cluster_geometry": {
        "parts": ("contact", "geometry"),
        "ready": (CONTACT, GEOMETRY),
    },
    "contact_plus_counts_and_geometry": {
        "parts": ("contact", "counts", "geometry"),
        "ready": (CONTACT, MINIMAL, GEOMETRY),
    },
    "contact_plus_full_density_geometry": {
        "parts": ("contact", "minimal", "geometry"),
        "ready": (CONTACT, MINIMAL, GEOMETRY),
    },
    "contact_plus_stale_full_density_geometry": {
        "parts": ("contact", "stale_minimal", "stale_geometry"),
        "ready": (CONTACT, STALE_MINIMAL, STALE_GEOMETRY),
    },
    "contact_plus_shuffled_full_density_geometry": {
        "parts": ("contact", "shuffled_minimal", "shuffled_geometry"),
        "ready": (CONTACT, SHUFFLED_MINIMAL, SHUFFLED_GEOMETRY),
    },
    "contact_plus_completed_8h": {
        "parts": ("contact", "eight_hour_activity", "eight_hour_trend"),
        "ready": (CONTACT, EIGHT_HOUR_ACTIVITY, EIGHT_HOUR_TREND),
    },
    "contact_plus_local_activity": {
        "parts": ("contact", "local_activity", "local_trend"),
        "ready": (CONTACT, LOCAL_ACTIVITY, LOCAL_TREND),
    },
    "contact_plus_wider_crypto": {
        "parts": ("contact", "wider"),
        "ready": (CONTACT, WIDER),
    },
    "contact_plus_density_and_completed_8h": {
        "parts": (
            "contact",
            "minimal",
            "geometry",
            "eight_hour_activity",
            "eight_hour_trend",
        ),
        "ready": (
            CONTACT,
            MINIMAL,
            GEOMETRY,
            EIGHT_HOUR_ACTIVITY,
            EIGHT_HOUR_TREND,
        ),
    },
    "contact_plus_density_and_local": {
        "parts": ("contact", "minimal", "geometry", "local_activity", "local_trend"),
        "ready": (CONTACT, MINIMAL, GEOMETRY, LOCAL_ACTIVITY, LOCAL_TREND),
    },
    "contact_plus_density_and_wider": {
        "parts": ("contact", "minimal", "geometry", "wider"),
        "ready": (CONTACT, MINIMAL, GEOMETRY, WIDER),
    },
    "contact_plus_density_local_wider_8h": {
        "parts": (
            "contact",
            "minimal",
            "geometry",
            "local_activity",
            "local_trend",
            "wider",
            "eight_hour_activity",
            "eight_hour_trend",
        ),
        "ready": (
            CONTACT,
            MINIMAL,
            GEOMETRY,
            LOCAL_ACTIVITY,
            LOCAL_TREND,
            WIDER,
            EIGHT_HOUR_ACTIVITY,
            EIGHT_HOUR_TREND,
        ),
    },
    "contact_plus_density_and_orderbook": {
        "parts": ("contact", "minimal", "geometry", "orderbook"),
        "ready": (CONTACT, MINIMAL, GEOMETRY, ORDERBOOK),
    },
    "contact_plus_density_and_stale_orderbook": {
        "parts": ("contact", "minimal", "geometry", "stale_orderbook"),
        "ready": (CONTACT, MINIMAL, GEOMETRY, STALE_ORDERBOOK),
    },
    "contact_plus_density_and_shuffled_orderbook": {
        "parts": ("contact", "minimal", "geometry", "shuffled_orderbook"),
        "ready": (CONTACT, MINIMAL, GEOMETRY, SHUFFLED_ORDERBOOK),
    },
}


def artifact(path: Path) -> dict[str, Any]:
    return g16c.artifact(path)


def load_freeze() -> dict[str, Any]:
    frozen = json.loads(g17z.FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation17_outcomes":
        raise ValueError("Generation 17 branch layer is not frozen.")
    branch_profiles = {
        profile
        for branch in frozen["branches"]
        for profile in branch.get("profiles", [])
    }
    unknown = sorted(branch_profiles.difference(PROFILE_SPECS))
    if unknown:
        raise ValueError(f"Generation 17 frozen profiles lack implementations: {unknown}")
    return frozen


def prefixed(columns: Sequence[str], block: str) -> tuple[str, ...]:
    selected = tuple(column for column in columns if column.startswith(f"{block}__"))
    if not selected:
        raise ValueError(f"No feature columns found for block {block!r}.")
    return selected


def part_columns(columns: Sequence[str], part: str) -> tuple[str, ...]:
    if part == "contact":
        return CONTACT_COLUMNS
    if part == "counts":
        return tuple(f"{MINIMAL}__{suffix}" for suffix in COUNT_SUFFIXES)
    if part == "proximity":
        return tuple(f"{MINIMAL}__{suffix}" for suffix in PROXIMITY_SUFFIXES)
    blocks = {
        "minimal": MINIMAL,
        "geometry": GEOMETRY,
        "stale_minimal": STALE_MINIMAL,
        "stale_geometry": STALE_GEOMETRY,
        "shuffled_minimal": SHUFFLED_MINIMAL,
        "shuffled_geometry": SHUFFLED_GEOMETRY,
        "local_activity": LOCAL_ACTIVITY,
        "local_trend": LOCAL_TREND,
        "wider": WIDER,
        "eight_hour_activity": EIGHT_HOUR_ACTIVITY,
        "eight_hour_trend": EIGHT_HOUR_TREND,
        "orderbook": ORDERBOOK,
        "stale_orderbook": STALE_ORDERBOOK,
        "shuffled_orderbook": SHUFFLED_ORDERBOOK,
    }
    if part not in blocks:
        raise ValueError(f"Unknown Generation 17 feature part: {part}")
    return prefixed(columns, blocks[part])


def profile_id(cohort: str, role: str) -> str:
    return f"g17__{cohort}__{role}__seed{SEED}"


def build_registry(feature_columns: Sequence[str]) -> dict[str, Any]:
    profiles: dict[str, dict[str, Any]] = {}
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

    comparisons: list[dict[str, Any]] = []

    def add(
        cohort: str,
        branch_id: str,
        route_id: str,
        question: str,
        candidate: str,
        controls: Sequence[tuple[str, str]],
        targets: Sequence[str],
    ) -> None:
        for control_type, baseline in controls:
            comparisons.append(
                {
                    "comparison_id": (
                        f"g17__{cohort}__{route_id}__vs_{control_type}__seed{SEED}"
                    ),
                    "cohort": cohort,
                    "branch_id": branch_id,
                    "question_id": route_id,
                    "route_id": route_id,
                    "route_type": "bounded_incremental_information",
                    "plain_question": question,
                    "candidate": profile_id(cohort, candidate),
                    "baseline": profile_id(cohort, baseline),
                    "control_type": control_type,
                    "expected_controls_for_route": len(controls),
                    "seed": SEED,
                    "targets": list(targets),
                }
            )

    density_routes = (
        (
            "density_counts_increment",
            "Do level counts improve crossing behaviour beyond contact state?",
            "contact_plus_density_counts",
            (("contact_only", "contact_only"),),
        ),
        (
            "proximity_width_increment",
            "Do distance and zone width improve crossing behaviour beyond contact state?",
            "contact_plus_proximity_width",
            (("contact_only", "contact_only"),),
        ),
        (
            "cluster_geometry_increment",
            "Does cluster geometry improve crossing behaviour beyond contact state?",
            "contact_plus_cluster_geometry",
            (("contact_only", "contact_only"),),
        ),
        (
            "counts_geometry_combination",
            "Do counts plus geometry beat each simpler component?",
            "contact_plus_counts_and_geometry",
            (
                ("counts", "contact_plus_density_counts"),
                ("geometry", "contact_plus_cluster_geometry"),
                ("contact_only", "contact_only"),
            ),
        ),
        (
            "full_density_geometry",
            "Does full density and geometry beat every simpler component?",
            "contact_plus_full_density_geometry",
            (
                ("counts_geometry", "contact_plus_counts_and_geometry"),
                ("proximity_width", "contact_plus_proximity_width"),
                ("contact_only", "contact_only"),
            ),
        ),
        (
            "current_density_vs_stale",
            "Does current density beat the causal 72-hour-old equivalent?",
            "contact_plus_full_density_geometry",
            (("causal_72h_old", "contact_plus_stale_full_density_geometry"),),
        ),
        (
            "current_density_vs_shuffled",
            "Does current density beat a within-period shuffled equivalent?",
            "contact_plus_full_density_geometry",
            (("within_period_shuffled", "contact_plus_shuffled_full_density_geometry"),),
        ),
    )
    reaction_routes = (
        (
            "completed_8h_reaction_increment",
            "Does completed 8h state improve reaction probability beyond contact state?",
            "contact_plus_completed_8h",
            (("contact_only", "contact_only"),),
        ),
        (
            "local_reaction_increment",
            "Does completed local state improve reaction probability beyond contact state?",
            "contact_plus_local_activity",
            (("contact_only", "contact_only"),),
        ),
        (
            "wider_reaction_increment",
            "Does wider crypto state improve reaction probability beyond contact state?",
            "contact_plus_wider_crypto",
            (("contact_only", "contact_only"),),
        ),
        (
            "density_8h_reaction_combination",
            "Does density plus completed 8h state beat both components?",
            "contact_plus_density_and_completed_8h",
            (
                ("density", "contact_plus_full_density_geometry"),
                ("completed_8h", "contact_plus_completed_8h"),
                ("contact_only", "contact_only"),
            ),
        ),
        (
            "density_local_reaction_combination",
            "Does density plus local state beat both components?",
            "contact_plus_density_and_local",
            (
                ("density", "contact_plus_full_density_geometry"),
                ("local", "contact_plus_local_activity"),
                ("contact_only", "contact_only"),
            ),
        ),
        (
            "density_wider_reaction_combination",
            "Does density plus wider crypto state beat both components?",
            "contact_plus_density_and_wider",
            (
                ("density", "contact_plus_full_density_geometry"),
                ("wider", "contact_plus_wider_crypto"),
                ("contact_only", "contact_only"),
            ),
        ),
        (
            "complete_reaction_context",
            "Does the complete reaction model beat every contained combination?",
            "contact_plus_density_local_wider_8h",
            (
                ("density_8h", "contact_plus_density_and_completed_8h"),
                ("density_local", "contact_plus_density_and_local"),
                ("density_wider", "contact_plus_density_and_wider"),
                ("density", "contact_plus_full_density_geometry"),
            ),
        ),
    )
    external_routes = (
        (
            "orderbook_current_increment",
            "Does source-ready current order-book state improve reaction probability?",
            "contact_plus_density_and_orderbook",
            (
                ("density_only", "contact_plus_full_density_geometry"),
                ("causal_stale", "contact_plus_density_and_stale_orderbook"),
                (
                    "within_period_shuffled",
                    "contact_plus_density_and_shuffled_orderbook",
                ),
            ),
        ),
    )
    for cohort in ("normal", "meme"):
        for route in density_routes:
            add(
                cohort,
                "g17a_density_geometry_decomposition",
                *route,
                CROSSING_TARGETS,
            )
        for route in reaction_routes:
            add(
                cohort,
                "g17c_reaction_probability_calibration",
                *route,
                REACTION_TARGETS,
            )
        for route in external_routes:
            add(
                cohort,
                "g17e_external_context_regimes",
                *route,
                REACTION_TARGETS,
            )
    return {
        "schema_version": 1,
        "generation": 17,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation17_freqai_target_materialization",
        "seed": SEED,
        "profiles": profiles,
        "comparisons": comparisons,
        "targets": list(TARGETS),
        "profit_used": False,
        "future_signed_direction_used": False,
    }


def load_g13_manifest(cohort: str) -> tuple[dict[str, Any], Path]:
    return g16c.load_g13_manifest(cohort)


def support_pair(
    item: dict[str, Any],
    parent_frozen: dict[str, Any],
    registry: dict[str, Any],
    cohort: str,
) -> dict[str, Any]:
    anchors = g16c.anchor_frame(item, parent_frozen)
    contact = g16c.merge_contact_state(anchors, item)
    feature_source = pd.read_parquet(item["mtf_feature_path"])
    support_source = pd.read_parquet(item["mtf_support_path"])
    for frame in (feature_source, support_source):
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    profiles = [
        profile for profile in registry["profiles"].values() if profile["cohort"] == cohort
    ]
    required_columns = sorted(
        {
            column
            for profile in profiles
            for column in profile["feature_columns"]
            if column not in CONTACT_COLUMNS
        }
    )
    features = anchors[["date"]].merge(
        feature_source[["date", *required_columns]],
        on="date",
        how="left",
        validate="one_to_one",
    )
    features[CONTACT_COLUMNS[0]] = pd.to_numeric(
        contact["contact_volume_ratio"], errors="coerce"
    ).to_numpy()
    features[CONTACT_COLUMNS[1]] = pd.to_numeric(
        contact["contact_range_ratio"], errors="coerce"
    ).to_numpy()
    features[CONTACT_COLUMNS[2]] = pd.to_numeric(
        contact["contact_pressure_change"], errors="coerce"
    ).abs().to_numpy()
    required_blocks = sorted(
        {
            block
            for profile in profiles
            for block in profile["required_ready_blocks"]
            if block != CONTACT
        }
    )
    ready_columns = [f"ready__{block}" for block in required_blocks]
    missing_ready = sorted(set(ready_columns).difference(support_source.columns))
    if missing_ready:
        raise ValueError(f"Missing readiness fields for {item['pair']}: {missing_ready}")
    support = anchors[["date", "period"]].merge(
        support_source[["date", *ready_columns]],
        on="date",
        how="left",
        validate="one_to_one",
    )
    support[f"ready__{CONTACT}"] = features[list(CONTACT_COLUMNS)].notna().all(axis=1)
    stem = g0.pair_file_stem(item["pair"])
    feature_path = ARTIFACT_ROOT / cohort / "feature_cache" / f"{stem}.parquet"
    support_path = ARTIFACT_ROOT / cohort / "support_cache" / f"{stem}.parquet"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    support_path.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(features, feature_path)
    g0.atomic_write_parquet(support, support_path)
    return {
        "pair": item["pair"],
        "source_path": item["source_path"],
        "source_sha256": item["source_sha256"],
        "feature_path": str(feature_path.resolve()),
        "feature_sha256": g0.sha256_file(feature_path),
        "support_path": str(support_path.resolve()),
        "support_sha256": g0.sha256_file(support_path),
        "rows": len(features),
        "future_outcomes_read": False,
    }


def target_frame(anchors: DataFrame, item: dict[str, Any]) -> DataFrame:
    outcome_columns = [
        column
        for horizon in HORIZONS
        for column in (
            f"volume_ratio_h{horizon}",
            f"crossings_h{horizon}",
            f"abs_excursion_atr_h{horizon}",
            f"away_excursion_atr_h{horizon}",
            f"through_excursion_atr_h{horizon}",
        )
    ]
    source = pd.read_parquet(
        item["source_path"],
        columns=[*g16c.SOURCE_KEYS, "control", "zone_half_width_atr", *outcome_columns],
    )
    source = source.loc[source["control"].eq("actual")].drop(columns="control")
    source["event_time"] = pd.to_datetime(source["event_time"], utc=True, errors="raise")
    merged = anchors.merge(
        source,
        left_on=list(g16c.ANCHOR_KEYS),
        right_on=list(g16c.SOURCE_KEYS),
        how="left",
        validate="one_to_one",
    )
    output = anchors.copy()
    threshold = pd.to_numeric(merged["zone_half_width_atr"], errors="coerce").clip(
        lower=0.5
    )
    for horizon in HORIZONS:
        crossings = pd.to_numeric(merged[f"crossings_h{horizon}"], errors="coerce")
        away = pd.to_numeric(merged[f"away_excursion_atr_h{horizon}"], errors="coerce")
        through = pd.to_numeric(
            merged[f"through_excursion_atr_h{horizon}"], errors="coerce"
        )
        volume = pd.to_numeric(merged[f"volume_ratio_h{horizon}"], errors="coerce")
        excursion = pd.to_numeric(
            merged[f"abs_excursion_atr_h{horizon}"], errors="coerce"
        )
        output[f"&-g17_crossing_count_h{horizon}"] = crossings.to_numpy()
        repeated = crossings.ge(2).astype(float)
        repeated.loc[crossings.isna()] = np.nan
        output[f"&-g17_repeated_recross_h{horizon}"] = repeated.to_numpy()
        two_sided = ((away >= threshold) & (through >= threshold)).astype(float)
        two_sided.loc[away.isna() | through.isna() | threshold.isna()] = np.nan
        output[f"&-g17_two_sided_traversal_h{horizon}"] = two_sided.to_numpy()
        reaction = ((excursion >= threshold) & (volume >= 1.25)).astype(float)
        reaction.loc[excursion.isna() | volume.isna() | threshold.isna()] = np.nan
        output[f"&-g17_reaction_h{horizon}"] = reaction.to_numpy()
    return output


def materialize_targets(
    item: dict[str, Any],
    source_item: dict[str, Any],
    parent_frozen: dict[str, Any],
) -> dict[str, Any]:
    anchors = g16c.anchor_frame(source_item, parent_frozen)
    targets = target_frame(anchors, source_item)
    support = pd.read_parquet(item["support_path"])
    support["date"] = pd.to_datetime(support["date"], utc=True, errors="raise")
    ready = [column for column in support if column.startswith("ready__")]
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
    evaluation = targets[metadata + list(TARGETS)].merge(
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
    cohort = str(evaluation["cohort"].iloc[0])
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
        if existing.get("status") == "completed_generation17_freqai_cache":
            return existing
        raise ValueError(f"Incomplete Generation 17 cache exists: {manifest_path}")
    g13, g13_path = load_g13_manifest(cohort)
    parent_frozen = g16c.load_freeze()
    columns = pd.read_parquet(g13["inventory"][0]["mtf_feature_path"]).columns.tolist()
    registry = build_registry(columns)
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    if REGISTRY_PATH.is_file():
        current = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        left = {key: value for key, value in current.items() if key != "created_at_utc"}
        right = {key: value for key, value in registry.items() if key != "created_at_utc"}
        if left != right:
            raise ValueError("Generation 17 registry changed after its first freeze.")
    else:
        g0.atomic_write_json(registry, REGISTRY_PATH)
    inventory = [
        support_pair(item, parent_frozen, registry, cohort)
        for item in g13["inventory"]
    ]
    support_freeze = {
        "schema_version": 1,
        "generation": 17,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_outcome_blind_support_before_target_materialization",
        "cohort": cohort,
        "generation17_freeze": artifact(g17z.FREEZE_PATH),
        "generation13_manifest": artifact(g13_path),
        "profile_registry": artifact(REGISTRY_PATH),
        "inventory": [
            {key: value for key, value in item.items() if not key.startswith("source_")}
            for item in inventory
        ],
        "future_outcome_values_read": False,
    }
    support_freeze_path = RECORD_ROOT / f"{cohort}_outcome_blind_support_freeze.json"
    g0.atomic_write_json(support_freeze, support_freeze_path)
    source_by_pair = {item["pair"]: item for item in g13["inventory"]}
    completed = [
        materialize_targets(item, source_by_pair[item["pair"]], parent_frozen)
        for item in inventory
    ]
    result = {
        "schema_version": 1,
        "generation": 17,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation17_freqai_cache",
        "cohort": cohort,
        "pairs": list(g13["pairs"]),
        "inventory": completed,
        "profile_registry": artifact(REGISTRY_PATH),
        "outcome_blind_support_freeze": artifact(support_freeze_path),
        "source_contracts": {
            "generation17_freeze": artifact(g17z.FREEZE_PATH),
            "generation13_manifest": artifact(g13_path),
        },
        "integrity": {
            "retained_parent_surface_only": True,
            "contact_features_use_completed_contact_candle": True,
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

"""Build Generation 16's causal contact-state FreqAI cache and frozen registry."""

from __future__ import annotations

# Keep dataframe/parquet work bounded when this module is used as a script.
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
    market_reaction_zone_generation13_cache as g13c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation16_freeze as g16z,
)


CACHE_ID = "g16_freqai_support_20260822a"
RECORD_ROOT = g16z.FREEZE_PATH.parent / "g16_broad_attribution" / "freqai_cache" / CACHE_ID
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
    / "generation16_branches"
    / "g16_broad_attribution"
    / "freqai_cache"
    / CACHE_ID
)
REGISTRY_PATH = RECORD_ROOT / "g16_freqai_profile_registry.json"
SEED = 2026082216
HORIZONS = (1, 2, 4)
TARGETS = tuple(
    column
    for horizon in HORIZONS
    for column in (
        f"&-g16_volume_ratio_h{horizon}",
        f"&-g16_range_ratio_h{horizon}",
        f"&-g16_crossings_h{horizon}",
        f"&-g16_reaction_h{horizon}",
    )
)

CONTACT = "g16_completed_contact_activity"
LEVEL = "g11_level_identity_timeframe"
MINIMAL = "g11_minimal_contact"
GEOMETRY = "g11_cluster_geometry"
WIDER = "g11_wider_crypto_market"
EIGHT_HOUR = "g13_mtf_8h_activity_volatility"
STALE_MINIMAL = f"{MINIMAL}_stale"
STALE_LEVEL = f"{LEVEL}_stale"
SHUFFLED_MINIMAL = f"{MINIMAL}_shuffled"
SHUFFLED_LEVEL = f"{LEVEL}_shuffled"

CONTACT_COLUMNS = (
    f"{CONTACT}__contact_volume_ratio",
    f"{CONTACT}__contact_range_ratio",
    f"{CONTACT}__absolute_contact_pressure_change",
)
PROFILE_ROLES = (
    "contact_activity_only",
    "contact_plus_level_identity",
    "contact_plus_level_density",
    "contact_plus_wider_crypto_activity",
    "contact_plus_completed_8h_activity",
    "contact_plus_level_plus_wider_activity",
    "causal_72h_old_added_block",
    "within_period_shuffled_added_block",
)
PROFILE_BLOCKS: dict[str, tuple[str, ...]] = {
    "contact_activity_only": (CONTACT,),
    "contact_plus_level_identity": (CONTACT, LEVEL),
    "contact_plus_level_density": (CONTACT, MINIMAL, GEOMETRY),
    "contact_plus_wider_crypto_activity": (CONTACT, WIDER),
    "contact_plus_completed_8h_activity": (CONTACT, EIGHT_HOUR),
    "contact_plus_level_plus_wider_activity": (CONTACT, LEVEL, WIDER),
    "causal_72h_old_added_block": (
        CONTACT,
        LEVEL,
        STALE_MINIMAL,
        STALE_LEVEL,
    ),
    "within_period_shuffled_added_block": (
        CONTACT,
        LEVEL,
        SHUFFLED_MINIMAL,
        SHUFFLED_LEVEL,
    ),
}

ANCHOR_KEYS = (
    "date",
    "anchor_level_family",
    "anchor_level_name",
    "anchor_source_timeframe",
    "anchor_representation",
)
SOURCE_KEYS = (
    "event_time",
    "level_family",
    "level_name",
    "source_timeframe",
    "representation",
)


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def load_freeze() -> dict[str, Any]:
    frozen = json.loads(g16z.FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation16_outcomes":
        raise ValueError("Generation 16 freeze is not valid.")
    roles = next(
        item["profiles"]
        for item in frozen["main_direction_neutral_family"]["routes"]
        if item["route_id"] == "freqai_incremental_activity_attribution"
    )
    if tuple(roles) != PROFILE_ROLES:
        raise ValueError("Generation 16 FreqAI profile roles drifted after freeze.")
    return frozen


def load_g13_manifest(cohort: str) -> tuple[dict[str, Any], Path]:
    path = g13c.RECORD_ROOT / f"{cohort}_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed_generation13_cache":
        raise ValueError(f"Generation 13 {cohort} cache is not terminal.")
    if manifest.get("cohort") != cohort:
        raise ValueError(f"Generation 13 cache cohort mismatch: {path}")
    return manifest, path


def block_columns(columns: Sequence[str], block: str) -> tuple[str, ...]:
    prefix = f"{block}__"
    selected = tuple(column for column in columns if column.startswith(prefix))
    if not selected:
        raise ValueError(f"No Generation 13 feature columns for block {block!r}.")
    return selected


def profile_id(cohort: str, role: str) -> str:
    return f"g16__{cohort}__{role}__seed{SEED}"


def build_registry(feature_columns: Sequence[str]) -> dict[str, Any]:
    profiles: dict[str, dict[str, Any]] = {}
    comparisons: list[dict[str, Any]] = []
    available = tuple(feature_columns)
    for cohort in ("normal", "meme"):
        for role in PROFILE_ROLES:
            blocks = PROFILE_BLOCKS[role]
            columns: list[str] = []
            for block in blocks:
                columns.extend(
                    CONTACT_COLUMNS if block == CONTACT else block_columns(available, block)
                )
            identifier = profile_id(cohort, role)
            profiles[identifier] = {
                "profile_id": identifier,
                "cohort": cohort,
                "role": role,
                "seed": SEED,
                "feature_columns": list(dict.fromkeys(columns)),
                "required_ready_blocks": list(blocks),
                "targets": list(TARGETS),
            }

        def add(
            route_id: str,
            plain_question: str,
            candidate_role: str,
            controls: Sequence[tuple[str, str]],
            cohort_name: str = cohort,
        ) -> None:
            for control_type, baseline_role in controls:
                comparisons.append(
                    {
                        "comparison_id": (
                            f"g16__{cohort_name}__{route_id}__vs_{control_type}__seed{SEED}"
                        ),
                        "cohort": cohort_name,
                        "question_id": route_id,
                        "route_id": route_id,
                        "route_type": "bounded_incremental_information",
                        "plain_question": plain_question,
                        "candidate": profile_id(cohort_name, candidate_role),
                        "baseline": profile_id(cohort_name, baseline_role),
                        "control_type": control_type,
                        "expected_controls_for_route": len(controls),
                        "seed": SEED,
                        "targets": list(TARGETS),
                    }
                )

        add(
            "level_identity_increment",
            "Does current calculated-level identity improve forecasts beyond the "
            "completed contact candle?",
            "contact_plus_level_identity",
            (("contact_activity_only", "contact_activity_only"),),
        )
        add(
            "level_density_increment",
            "Does current level density and cluster geometry improve forecasts beyond "
            "the completed contact candle?",
            "contact_plus_level_density",
            (("contact_activity_only", "contact_activity_only"),),
        )
        add(
            "wider_crypto_increment",
            "Does the wider crypto market improve forecasts beyond the completed contact candle?",
            "contact_plus_wider_crypto_activity",
            (("contact_activity_only", "contact_activity_only"),),
        )
        add(
            "completed_8h_increment",
            "Does completed 8h activity improve forecasts beyond the completed contact candle?",
            "contact_plus_completed_8h_activity",
            (("contact_activity_only", "contact_activity_only"),),
        )
        add(
            "level_plus_wider_interaction",
            "Does level identity plus wider-market state beat every immediately simpler component?",
            "contact_plus_level_plus_wider_activity",
            (
                ("contact_plus_level", "contact_plus_level_identity"),
                ("contact_plus_wider", "contact_plus_wider_crypto_activity"),
                ("contact_only", "contact_activity_only"),
            ),
        )
        add(
            "current_model_vs_causal_72h_noise",
            "Does the compact current-level model outperform the same model polluted "
            "by a causal 72h-old block?",
            "contact_plus_level_identity",
            (("causal_72h_old_added_block", "causal_72h_old_added_block"),),
        )
        add(
            "current_model_vs_shuffled_noise",
            "Does the compact current-level model outperform the same model polluted "
            "by within-period shuffled level data?",
            "contact_plus_level_identity",
            (("within_period_shuffled_added_block", "within_period_shuffled_added_block"),),
        )
    return {
        "schema_version": 1,
        "generation": 16,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation16_freqai_target_materialization",
        "seed": SEED,
        "profiles": profiles,
        "comparisons": comparisons,
        "targets": list(TARGETS),
        "profit_used": False,
        "future_signed_direction_used": False,
    }


def retained_surface(frozen: dict[str, Any]) -> tuple[set[str], set[str]]:
    parent = frozen["parent_surface"]
    return (
        set(map(str, parent["strict_level_families"])),
        set(map(str, parent["strict_source_timeframes"])),
    )


def anchor_frame(item: dict[str, Any], frozen: dict[str, Any]) -> DataFrame:
    families, timeframes = retained_surface(frozen)
    frame = pd.read_parquet(
        item["mtf_evaluation_path"],
        columns=list(ANCHOR_KEYS)
        + ["period", "pair", "cohort", "market_group", "smart_contract_platform"],
    )
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    selected = frame["anchor_level_family"].isin(families) & frame["anchor_source_timeframe"].isin(
        timeframes
    )
    output = frame.loc[selected].copy()
    if output["date"].duplicated().any():
        raise ValueError(f"Duplicate retained anchor dates for {item['pair']}.")
    return output.sort_values("date").reset_index(drop=True)


def merge_contact_state(anchors: DataFrame, item: dict[str, Any]) -> DataFrame:
    source_columns = [
        *SOURCE_KEYS,
        "control",
        "contact_volume_ratio",
        "contact_range_ratio",
        "contact_pressure_change",
    ]
    source = pd.read_parquet(item["source_path"], columns=source_columns)
    source = source.loc[source["control"].eq("actual")].copy()
    source["event_time"] = pd.to_datetime(source["event_time"], utc=True, errors="raise")
    if source.duplicated(list(SOURCE_KEYS)).any():
        raise ValueError(f"Duplicate exact actual anchors in {item['source_path']}.")
    merged = anchors.merge(
        source.drop(columns="control"),
        left_on=list(ANCHOR_KEYS),
        right_on=list(SOURCE_KEYS),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if merged["_merge"].ne("both").any():
        raise ValueError(f"Unmatched Generation 16 contact anchors for {item['pair']}.")
    return merged.drop(columns="_merge")


def build_support_pair(
    item: dict[str, Any], frozen: dict[str, Any], registry: dict[str, Any], cohort: str
) -> dict[str, Any]:
    anchors = anchor_frame(item, frozen)
    contact = merge_contact_state(anchors, item)
    feature_source = pd.read_parquet(item["mtf_feature_path"])
    support_source = pd.read_parquet(item["mtf_support_path"])
    for frame in (feature_source, support_source):
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    required_columns = sorted(
        {
            column
            for profile in registry["profiles"].values()
            if profile["cohort"] == cohort
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
    features[CONTACT_COLUMNS[2]] = (
        pd.to_numeric(contact["contact_pressure_change"], errors="coerce").abs().to_numpy()
    )
    required_blocks = sorted(
        {
            block
            for profile in registry["profiles"].values()
            if profile["cohort"] == cohort
            for block in profile["required_ready_blocks"]
            if block != CONTACT
        }
    )
    ready_columns = [f"ready__{block}" for block in required_blocks]
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
        "contact_outcome_columns_read": False,
    }


def target_frame(anchors: DataFrame, item: dict[str, Any]) -> DataFrame:
    outcome_columns = [
        column
        for horizon in HORIZONS
        for column in (
            f"volume_ratio_h{horizon}",
            f"range_ratio_h{horizon}",
            f"crossings_h{horizon}",
            f"abs_excursion_atr_h{horizon}",
        )
    ]
    source = pd.read_parquet(
        item["source_path"],
        columns=[*SOURCE_KEYS, "control", "zone_half_width_atr", *outcome_columns],
    )
    source = source.loc[source["control"].eq("actual")].drop(columns="control")
    source["event_time"] = pd.to_datetime(source["event_time"], utc=True, errors="raise")
    merged = anchors.merge(
        source,
        left_on=list(ANCHOR_KEYS),
        right_on=list(SOURCE_KEYS),
        how="left",
        validate="one_to_one",
    )
    output = anchors.copy()
    threshold = pd.to_numeric(merged["zone_half_width_atr"], errors="coerce").clip(lower=0.5)
    for horizon in HORIZONS:
        volume = pd.to_numeric(merged[f"volume_ratio_h{horizon}"], errors="coerce")
        excursion = pd.to_numeric(merged[f"abs_excursion_atr_h{horizon}"], errors="coerce")
        output[f"&-g16_volume_ratio_h{horizon}"] = volume.to_numpy()
        output[f"&-g16_range_ratio_h{horizon}"] = pd.to_numeric(
            merged[f"range_ratio_h{horizon}"], errors="coerce"
        ).to_numpy()
        output[f"&-g16_crossings_h{horizon}"] = pd.to_numeric(
            merged[f"crossings_h{horizon}"], errors="coerce"
        ).to_numpy()
        reaction = ((excursion >= threshold) & (volume >= 1.25)).astype(float)
        reaction.loc[excursion.isna() | volume.isna() | threshold.isna()] = np.nan
        output[f"&-g16_reaction_h{horizon}"] = reaction.to_numpy()
    return output


def materialize_targets(
    item: dict[str, Any], source_item: dict[str, Any], frozen: dict[str, Any], cohort: str
) -> dict[str, Any]:
    anchors = anchor_frame(source_item, frozen)
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
    frozen = load_freeze()
    manifest_path = RECORD_ROOT / f"{cohort}_manifest.json"
    if manifest_path.is_file() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed_generation16_freqai_cache":
            return existing
        raise ValueError(f"Incomplete Generation 16 cache exists: {manifest_path}")
    g13, g13_path = load_g13_manifest(cohort)
    first_columns = pd.read_parquet(g13["inventory"][0]["mtf_feature_path"]).columns.tolist()
    registry = build_registry(first_columns)
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    if REGISTRY_PATH.is_file():
        existing_registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        comparable = {
            key: value for key, value in existing_registry.items() if key != "created_at_utc"
        }
        expected = {key: value for key, value in registry.items() if key != "created_at_utc"}
        if comparable != expected:
            raise ValueError("Generation 16 FreqAI registry changed after it was frozen.")
    else:
        g0.atomic_write_json(registry, REGISTRY_PATH)
    inventory: list[dict[str, Any]] = []
    for source_item in g13["inventory"]:
        inventory.append(build_support_pair(source_item, frozen, registry, cohort))
    support_freeze = {
        "schema_version": 1,
        "generation": 16,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_outcome_blind_support_before_target_materialization",
        "cohort": cohort,
        "generation16_freeze": artifact(g16z.FREEZE_PATH),
        "generation13_manifest": artifact(g13_path),
        "profile_registry": artifact(REGISTRY_PATH),
        "inventory": [
            {
                key: value
                for key, value in item.items()
                if key not in {"source_path", "source_sha256"}
            }
            for item in inventory
        ],
        "future_outcome_values_read": False,
    }
    support_freeze_path = RECORD_ROOT / f"{cohort}_outcome_blind_support_freeze.json"
    g0.atomic_write_json(support_freeze, support_freeze_path)
    source_by_pair = {item["pair"]: item for item in g13["inventory"]}
    completed = [
        materialize_targets(item, source_by_pair[item["pair"]], frozen, cohort)
        for item in inventory
    ]
    result = {
        "schema_version": 1,
        "generation": 16,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation16_freqai_cache",
        "cohort": cohort,
        "pairs": list(g13["pairs"]),
        "inventory": completed,
        "profile_registry": artifact(REGISTRY_PATH),
        "outcome_blind_support_freeze": artifact(support_freeze_path),
        "source_contracts": {
            "generation16_freeze": artifact(g16z.FREEZE_PATH),
            "generation13_manifest": artifact(g13_path),
        },
        "integrity": {
            "retained_parent_surface_only": True,
            "contact_features_use_completed_contact_candle": True,
            "future_outcomes_joined_after_support_freeze": True,
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

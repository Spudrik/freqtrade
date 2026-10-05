from __future__ import annotations

# Keep numerical libraries single-threaded. Pair-level parallelism is owned by callers.
# ruff: noqa: E402
import os


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.market_reaction_zone_cluster_generation0 import (  # noqa: E501
    component_dependency_group,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    LARGE_ARTIFACT_ROOT,
    OUTPUT_ROOT,
    atomic_write_csv,
    atomic_write_json,
    load_manifest,
    numeric_array,
    prepare_base_market_frame,
    sha256_file,
    timeframe_hours,
    utc_now,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_localization import (  # noqa: E501
    pair_stem,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_vp_roles import (  # noqa: E501
    stable_json_sha256,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_lvn_attribution import (  # noqa: E501
    aligned_selected_levels_from_verified_prefix,
)


FROZEN_BATCH = OUTPUT_ROOT / "generation2_review" / "g3_frozen_branch_batch.json"
PARENT_REPORT_ROOT = OUTPUT_ROOT / "generation3_branches" / "g3a_thin_lvn_attribution"
REPORT_ROOT = OUTPUT_ROOT / "generation3_branches" / "g3g_thin_lvn_one_minute_replay"
ARTIFACT_ROOT = (
    LARGE_ARTIFACT_ROOT / "generation3_branches" / "g3g_thin_lvn_one_minute_replay"
)
DATA_DIR = REPO_ROOT / "user_data" / "data" / "binance"
FUTURES_DATA_DIR = DATA_DIR / "futures"
OUTPUT_SCHEMA_VERSION = 1
SELECTION_SEED = "g3g_thin_lvn_replay_v1"
CONTACT_VOLUME_THRESHOLD = 2.0
INDEPENDENCE_HOURS = 48
FEATURE_WARMUP_HOURS = 24
LEVEL_NAMES = ("lvn_above", "lvn_below")
EXPECTED_APPROACH = {"lvn_above": "from_below", "lvn_below": "from_above"}
WINDOW_HOURS = {
    "1h": (24, 12),
    "4h": (72, 48),
    "8h": (7 * 24, 4 * 24),
    "1d": (30 * 24, 14 * 24),
}


@dataclass(frozen=True)
class CohortSource:
    cohort: str
    parent_run_id: str
    manifest_path: Path
    periods: tuple[str, ...]


COHORT_SOURCES = (
    CohortSource(
        cohort="normal",
        parent_run_id="g3a_lvn_cluster_attribution_normal10_full_20260814b",
        manifest_path=OUTPUT_ROOT / "generation0_manifest.json",
        periods=("development", "validation_early", "validation_late"),
    ),
    CohortSource(
        cohort="meme",
        parent_run_id="g3a_lvn_cluster_attribution_meme_full_20260814b",
        manifest_path=OUTPUT_ROOT / "generation2_shared" / "g2_meme_reaction_manifest.json",
        periods=(
            "meme_development",
            "meme_validation_early",
            "meme_validation_late",
        ),
    ),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the direction-blind Generation 3G thin-LVN episode pool, its bounded "
            "normal-versus-meme diagnostic sample, exact peer components, and targeted "
            "one-minute acquisition intervals. This command does not read one-minute data."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--coverage-audit",
        action="store_true",
        help="Audit already-downloaded frozen 1m intervals without reading signed paths.",
    )
    args = parser.parse_args(argv)

    validate_frozen_branch()
    if args.coverage_audit:
        return audit_one_minute_coverage(args.run_id)
    run_dir = REPORT_ROOT / args.run_id
    artifact_dir = ARTIFACT_ROOT / args.run_id
    record_path = run_dir / "g3g_freeze_record.json"
    if record_path.is_file() and not args.overwrite:
        raise FileExistsError(f"Run already exists: {record_path}")

    parent_contracts = [validate_parent(source) for source in COHORT_SOURCES]
    request = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_batch_sha256": sha256_file(FROZEN_BATCH),
        "parent_contracts": parent_contracts,
        "selection": {
            "parent_event": (
                "actual high-thinness one-hour LVN from corrected G3A, with an independent "
                "peer cluster surviving LVN removal and at least two connected contacted "
                "peer dependency groups"
            ),
            "clear_approach": {
                "lvn_above": "from_below",
                "lvn_below": "from_above",
            },
            "direction_neutral_activity": (
                "contact-hour volume at least 2.0 times the causal previous-24-hour median"
            ),
            "contact_volume_ratio_minimum": CONTACT_VOLUME_THRESHOLD,
            "episode_independence_hours": INDEPENDENCE_HOURS,
            "independence_priority": "ascending SHA-256 of the frozen episode key",
            "sample": (
                "one episode per cohort x frozen period x LVN side, preferring six unique "
                "pairs per cohort, selected by the same frozen hash"
            ),
            "selection_seed": SELECTION_SEED,
            "future_signed_path_used": False,
        },
        "windows": {
            "visible_pre_post_hours_by_causal_anchor": {
                key: {"pre": value[0], "post": value[1]}
                for key, value in WINDOW_HOURS.items()
            },
            "feature_warmup_hours": FEATURE_WARMUP_HOURS,
            "cluster_anchor_rule": (
                "highest timeframe in the primary connected contacted independent peer "
                "cluster, retaining the one-hour LVN as the parent level"
            ),
            "contact_alignment_after_download": (
                "first one-minute entry into the frozen LVN zone during the parent contact hour"
            ),
        },
        "exchange": "binance",
        "market_type": "futures",
        "timeframe": "1m",
        "direction_prediction": False,
        "profit_optimization": False,
    }
    request_sha256 = stable_json_sha256(request)
    record = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "status": "freezing",
        "started_at_utc": utc_now(),
        "request_sha256": request_sha256,
        "request_contract": request,
    }
    atomic_write_json(record, record_path)
    try:
        pools: list[DataFrame] = []
        pre_independence_counts: dict[str, int] = {}
        for source in COHORT_SOURCES:
            pool, before_independence = build_cohort_pool(source)
            pools.append(pool)
            pre_independence_counts[source.cohort] = before_independence
        episode_pool = pd.concat(pools, ignore_index=True)
        components, cluster_summary = freeze_cluster_components(episode_pool)
        episode_pool = episode_pool.merge(
            cluster_summary,
            on=["episode_id", "cohort", "pair", "event_time"],
            how="left",
            validate="one_to_one",
        )
        validate_cluster_summary(episode_pool)
        sample = select_diagnostic_sample(episode_pool)
        acquisition = acquisition_intervals(sample)
        sample_components = components.loc[
            components["episode_id"].isin(set(sample["episode_id"].astype(str)))
        ].copy()

        pool_path = artifact_dir / "g3g_complete_qualifying_episode_pool.csv"
        components_path = artifact_dir / "g3g_complete_cluster_components.csv"
        sample_path = run_dir / "g3g_frozen_episode_sample.csv"
        sample_components_path = run_dir / "g3g_frozen_sample_components.csv"
        acquisition_path = run_dir / "g3g_one_minute_acquisition_intervals.csv"
        atomic_write_csv(sort_episode_rows(episode_pool), pool_path)
        atomic_write_csv(sort_component_rows(components), components_path)
        atomic_write_csv(sort_episode_rows(sample), sample_path)
        atomic_write_csv(sort_component_rows(sample_components), sample_components_path)
        atomic_write_csv(acquisition.sort_values(["pair", "interval_start_utc"]), acquisition_path)

        record.update(
            {
                "status": "frozen_before_one_minute_paths",
                "completed_at_utc": utc_now(),
                "qualifying_before_48h_independence": pre_independence_counts,
                "independent_episode_counts": count_by(episode_pool, "cohort"),
                "sample_episode_counts": count_by(sample, "cohort"),
                "sample_pair_counts": sample.groupby("cohort")["pair"].nunique().to_dict(),
                "sample_period_counts": nested_counts(sample, ["cohort", "period"]),
                "sample_side_counts": nested_counts(sample, ["cohort", "level_name"]),
                "causal_anchor_counts": count_by(sample, "cluster_causal_anchor_timeframe"),
                "component_rows": len(components),
                "acquisition_intervals": len(acquisition),
                "artifacts": {
                    "complete_episode_pool_csv": artifact_record(pool_path),
                    "complete_cluster_components_csv": artifact_record(components_path),
                    "frozen_sample_csv": artifact_record(sample_path),
                    "frozen_sample_components_csv": artifact_record(sample_components_path),
                    "acquisition_intervals_csv": artifact_record(acquisition_path),
                },
                "direction_prediction": False,
                "profit_optimization": False,
            }
        )
        atomic_write_json(record, record_path)
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "failed_at_utc": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_write_json(record, record_path)
        raise
    return 0


def validate_frozen_branch() -> None:
    frozen = json.loads(FROZEN_BATCH.read_text(encoding="utf-8"))
    branch = next(
        (
            item
            for item in frozen.get("branches", [])
            if item.get("id") == "g3g_thin_lvn_evidence_triggered_1m_replay"
        ),
        None,
    )
    if branch is None or branch.get("status") != "frozen_next_batch":
        raise ValueError("The frozen Generation 3G one-minute branch is unavailable.")
    if int(branch.get("iteration_cap", -1)) != 1:
        raise ValueError("Generation 3G must retain its single-iteration cap.")


def validate_parent(source: CohortSource) -> dict[str, Any]:
    run_dir = PARENT_REPORT_ROOT / source.parent_run_id
    record_path = run_dir / "g3a_run_record.json"
    integrity_path = run_dir / "g3a_integrity.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    request = record.get("request_contract", {})
    component_scope = request.get("component_scope", {})
    if record.get("status") != "completed" or int(record.get("schema_version", -1)) != 2:
        raise ValueError(f"Corrected G3A parent is not complete: {source.parent_run_id}")
    if not integrity.get("passed"):
        raise ValueError(f"Corrected G3A integrity failed: {source.parent_run_id}")
    if component_scope.get("independent_dependency_groups") is not True:
        raise ValueError(f"G3A parent lacks independent dependency groups: {source.parent_run_id}")
    if component_scope.get("requires_two_connected_contacted_peer_dependency_groups") is not True:
        raise ValueError(f"G3A parent lacks connected contacted groups: {source.parent_run_id}")
    if request.get("direction_prediction") is not False:
        raise ValueError("G3A parent crossed the direction boundary.")
    return {
        "cohort": source.cohort,
        "run_id": source.parent_run_id,
        "run_record": str(record_path.resolve()),
        "run_record_sha256": sha256_file(record_path),
        "integrity": str(integrity_path.resolve()),
        "integrity_sha256": sha256_file(integrity_path),
        "manifest": str(source.manifest_path.resolve()),
        "manifest_sha256": sha256_file(source.manifest_path),
        "independent_outcome_pairs": int(record["independent_outcome_pairs"]),
    }


def build_cohort_pool(source: CohortSource) -> tuple[DataFrame, int]:
    run_dir = PARENT_REPORT_ROOT / source.parent_run_id
    record = json.loads((run_dir / "g3a_run_record.json").read_text(encoding="utf-8"))
    artifact_dir = Path(str(record["bulky_artifacts"]))
    independent_path = artifact_dir / "g3a_independent_outcome_pairs.parquet"
    matches = pd.read_parquet(independent_path)
    matches = matches.loc[
        matches["attribute_assignment"].eq("actual")
        & matches["outcome"].eq("contact_volume_ratio")
    ].copy()
    state_columns = sorted(column for column in matches if column.startswith("high_state__"))
    identity_columns = [
        "pair",
        "period",
        "level_name",
        "approach_state",
        "high_base_index",
        "high_event_time",
        "high_attribute_value",
        *state_columns,
    ]
    events = collapse_parent_events(matches[identity_columns])
    source_paths = {
        str(item["pair"]): Path(str(item["event_path"]))
        for item in record["request_contract"]["source_contracts"]
        if item["timeframe"] == "1h"
    }
    joined: list[DataFrame] = []
    event_columns = [
        "pair",
        "batch",
        "level_family",
        "level_name",
        "level_column",
        "representation",
        "control",
        "zone_method",
        "base_index",
        "event_time",
        "source_available_at",
        "source_open",
        "level_price",
        "zone_half_width",
        "zone_half_width_atr",
        "base_atr",
        "level_score",
        "contact_volume_ratio",
        "contact_pressure_change",
        "contact_range_ratio",
        "dwell_fraction_h4",
        "crossings_h4",
        "volume_ratio_h48",
    ]
    for pair, group in events.groupby("pair", sort=False):
        atlas = pd.read_parquet(source_paths[str(pair)], columns=event_columns)
        atlas = atlas.loc[
            atlas["control"].eq("actual")
            & atlas["zone_method"].eq("wide_base_atr")
            & atlas["level_name"].isin(LEVEL_NAMES)
            & atlas["level_family"].eq("volume_profile_nodes")
            & atlas["representation"].eq("settled")
        ].copy()
        renamed = group.rename(
            columns={
                "high_base_index": "base_index",
                "high_event_time": "event_time",
                "high_attribute_value": "parent_high_thinness_score",
                **{
                    column: column.removeprefix("high_state__")
                    for column in state_columns
                },
            }
        )
        joined.append(
            renamed.merge(
                atlas,
                on=["pair", "base_index", "event_time", "level_name"],
                how="inner",
                validate="one_to_one",
            )
        )
    pool = pd.concat(joined, ignore_index=True)
    if not np.allclose(
        pd.to_numeric(pool["parent_high_thinness_score"]),
        pd.to_numeric(pool["level_score"]),
        equal_nan=False,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(f"G3A high-thinness values do not reproduce for {source.cohort}.")
    pool["event_time"] = pd.to_datetime(pool["event_time"], utc=True)
    pool["source_available_at"] = pd.to_datetime(pool["source_available_at"], utc=True)
    pool["source_open"] = pd.to_datetime(pool["source_open"], utc=True)
    pool = pool.loc[
        pool["period"].isin(source.periods)
        & pool["contact_volume_ratio"].ge(CONTACT_VOLUME_THRESHOLD)
        & pool.apply(
            lambda row: str(row["approach_state"]) == EXPECTED_APPROACH[str(row["level_name"])],
            axis=1,
        )
    ].copy()
    if (pool["source_available_at"] > pool["event_time"]).any():
        raise ValueError(f"Future LVN source admitted for {source.cohort}.")
    pool["cohort"] = source.cohort
    pool["parent_run_id"] = source.parent_run_id
    pool["selection_key"] = pool.apply(
        lambda row: (
            f"{SELECTION_SEED}|{source.cohort}|{row['period']}|{row['pair']}|"
            f"{pd.Timestamp(row['event_time']).isoformat()}|{row['level_name']}"
        ),
        axis=1,
    )
    pool["selection_hash"] = pool["selection_key"].map(sha256_text)
    before_independence = len(pool)
    pool = select_time_independent(pool, hours=INDEPENDENCE_HOURS)
    pool["episode_id"] = pool["selection_hash"].map(lambda value: f"g3g_{source.cohort}_{value[:16]}")
    pool["qualification_reason"] = (
        "corrected G3A high-thinness 1h LVN; clear outside approach; independent mixed "
        "peer cluster; at least two connected contacted peer mechanisms; contact-hour "
        "volume >= 2.0x prior 24h median"
    )
    pool["direction_used_for_selection"] = False
    pool["retrospective_price_reference"] = False
    pool["price_reference_known_at_event"] = True
    pool["level_role"] = "activity_transit"
    pool["source_timeframe"] = "1h"
    return pool.reset_index(drop=True), before_independence


def collapse_parent_events(frame: DataFrame) -> DataFrame:
    key = ["pair", "high_base_index", "high_event_time", "level_name"]
    checked = [column for column in frame if column not in key]
    conflicts = frame.groupby(key, observed=True)[checked].nunique(dropna=False).gt(1).any(axis=1)
    if conflicts.any():
        raise ValueError("A G3A high event has conflicting parent attributes.")
    return frame.drop_duplicates(key).reset_index(drop=True)


def select_time_independent(frame: DataFrame, *, hours: int) -> DataFrame:
    kept_indexes: list[int] = []
    kept_times: dict[str, list[pd.Timestamp]] = {}
    for index, row in frame.sort_values("selection_hash").iterrows():
        pair = str(row["pair"])
        event_time = pd.Timestamp(row["event_time"])
        if any(
            abs((event_time - previous).total_seconds()) < hours * 3600
            for previous in kept_times.get(pair, [])
        ):
            continue
        kept_indexes.append(index)
        kept_times.setdefault(pair, []).append(event_time)
    output = frame.loc[kept_indexes].copy()
    output["independence_hours"] = hours
    output["independence_selection_order"] = np.arange(len(output), dtype=np.int64)
    return output


def freeze_cluster_components(pool: DataFrame) -> tuple[DataFrame, DataFrame]:
    component_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    sources_by_cohort = {source.cohort: source for source in COHORT_SOURCES}
    for (cohort, pair), episodes in pool.groupby(["cohort", "pair"], sort=False):
        source = sources_by_cohort[str(cohort)]
        manifest = load_manifest(source.manifest_path)
        base = prepare_base_market_frame(str(pair), manifest)
        aligned = aligned_selected_levels_from_verified_prefix(
            pair=str(pair),
            base=base,
            manifest_path=source.manifest_path,
            timeframes=tuple(manifest["data"]["source_timeframes"]),
        )
        base_high = numeric_array(base["high"])
        base_low = numeric_array(base["low"])
        for _, episode in episodes.iterrows():
            rows, summary = episode_components(
                episode,
                aligned=aligned,
                base_high=base_high,
                base_low=base_low,
            )
            component_rows.extend(rows)
            summaries.append(summary)
    return DataFrame(component_rows), DataFrame(summaries)


def episode_components(
    episode: pd.Series,
    *,
    aligned: list[Any],
    base_high: np.ndarray,
    base_low: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_index = int(episode["base_index"])
    event_time = pd.Timestamp(episode["event_time"])
    target_matches = [
        (index, item)
        for index, item in enumerate(aligned)
        if item.timeframe == "1h" and item.spec.name == str(episode["level_name"])
    ]
    if len(target_matches) != 1:
        raise ValueError(f"Expected one causal LVN target for {episode['episode_id']}.")
    target_index, target = target_matches[0]
    target_price = float(target.level[base_index])
    frozen_price = float(episode["level_price"])
    if not bool(target.valid[base_index]) or not np.isclose(
        target_price, frozen_price, rtol=1e-10, atol=1e-10
    ):
        raise ValueError(f"Frozen LVN no longer reproduces for {episode['episode_id']}.")
    atr = float(episode["base_atr"])
    geometry_half_width = max(0.5 * atr, abs(target_price) * 0.0005)
    target_group = component_dependency_group(target.spec)
    peers: list[dict[str, Any]] = []
    for index, item in enumerate(aligned):
        if index == target_index or component_dependency_group(item.spec) == target_group:
            continue
        if not bool(item.valid[base_index]):
            continue
        level = float(item.level[base_index])
        if not np.isfinite(level) or level <= 0.0:
            continue
        half_width = max(0.5 * atr, abs(level) * 0.0005)
        if abs(level - target_price) > half_width + geometry_half_width:
            continue
        lower = level - half_width
        upper = level + half_width
        peers.append(
            {
                "aligned_index": index,
                "lower": lower,
                "upper": upper,
                "contacted": bool(
                    base_high[base_index] >= lower and base_low[base_index] <= upper
                ),
                "dependency_group": component_dependency_group(item.spec),
                "item": item,
                "level_price": level,
                "zone_half_width": half_width,
            }
        )
    assign_contact_components(peers)
    contact_components: dict[int, list[dict[str, Any]]] = {}
    for peer in peers:
        if peer["contact_component_id"] is not None:
            contact_components.setdefault(int(peer["contact_component_id"]), []).append(peer)
    eligible_components = [
        (component_id, rows)
        for component_id, rows in contact_components.items()
        if len({row["dependency_group"] for row in rows}) >= 2
    ]
    if not eligible_components:
        raise ValueError(
            f"Corrected parent event lost its connected contacted peer groups: "
            f"{episode['episode_id']}"
        )
    primary_id, primary = sorted(
        eligible_components,
        key=lambda item: (
            -len({row["dependency_group"] for row in item[1]}),
            -len(item[1]),
            item[0],
        ),
    )[0]
    primary_groups = sorted({str(row["dependency_group"]) for row in primary})
    primary_timeframes = [str(row["item"].timeframe) for row in primary]
    causal_anchor = max(("1h", *primary_timeframes), key=timeframe_hours)
    common = {
        "episode_id": str(episode["episode_id"]),
        "cohort": str(episode["cohort"]),
        "pair": str(episode["pair"]),
        "event_time": event_time,
        "parent_level_name": str(episode["level_name"]),
        "parent_level_price": frozen_price,
        "parent_zone_half_width": float(episode["zone_half_width"]),
        "geometry_anchor_half_width": geometry_half_width,
        "cluster_causal_anchor_timeframe": causal_anchor,
    }
    rows: list[dict[str, Any]] = [
        {
            **common,
            "component_kind": "parent_lvn",
            "component_family": target.spec.family,
            "component_name": target.spec.name,
            "component_column": target.spec.column,
            "component_representation": target.spec.representation,
            "component_timeframe": target.timeframe,
            "component_dependency_group": target_group,
            "component_level_price": target_price,
            "component_zone_half_width": geometry_half_width,
            "component_contacted_on_parent_hour": True,
            "contact_component_id": primary_id,
            "in_primary_connected_contact_cluster": True,
            "component_source_available_at": target.source_available.iloc[base_index],
            "component_source_open": target.source_open.iloc[base_index],
            "component_active": bool(target.valid[base_index]),
        }
    ]
    for peer in peers:
        item = peer["item"]
        rows.append(
            {
                **common,
                "component_kind": "peer_level",
                "component_family": item.spec.family,
                "component_name": item.spec.name,
                "component_column": item.spec.column,
                "component_representation": item.spec.representation,
                "component_timeframe": item.timeframe,
                "component_dependency_group": peer["dependency_group"],
                "component_level_price": peer["level_price"],
                "component_zone_half_width": peer["zone_half_width"],
                "component_contacted_on_parent_hour": peer["contacted"],
                "contact_component_id": peer["contact_component_id"],
                "in_primary_connected_contact_cluster": (
                    peer["contact_component_id"] == primary_id
                ),
                "component_source_available_at": item.source_available.iloc[base_index],
                "component_source_open": item.source_open.iloc[base_index],
                "component_active": bool(item.valid[base_index]),
            }
        )
    for row in rows:
        available = pd.Timestamp(row["component_source_available_at"])
        if available > event_time:
            raise ValueError(f"Future peer component admitted for {episode['episode_id']}.")
        row["component_source_age_hours"] = (
            event_time - available
        ).total_seconds() / 3600.0
        row["component_source_key"] = (
            f"{row['component_timeframe']}|{row['component_family']}|"
            f"{row['component_column']}|{row['component_representation']}|"
            f"{available.isoformat()}"
        )
    persistence = int(max(1, round(float(episode["state_vp_level_persistence_bars"]))))
    prior_start = max(0, base_index - persistence)
    prior_inside = (
        (base_high[prior_start:base_index] >= frozen_price - float(episode["zone_half_width"]))
        & (base_low[prior_start:base_index] <= frozen_price + float(episode["zone_half_width"]))
    )
    prior_episode_count = int(
        np.sum(prior_inside & ~np.r_[False, prior_inside[:-1]]) if len(prior_inside) else 0
    )
    summary = {
        "episode_id": str(episode["episode_id"]),
        "cohort": str(episode["cohort"]),
        "pair": str(episode["pair"]),
        "event_time": event_time,
        "cluster_causal_anchor_timeframe": causal_anchor,
        "nearby_peer_level_count": len(peers),
        "contacted_peer_level_count": int(sum(bool(row["contacted"]) for row in peers)),
        "nearby_peer_dependency_group_count": len(
            {str(row["dependency_group"]) for row in peers}
        ),
        "contacted_peer_dependency_group_count": len(
            {str(row["dependency_group"]) for row in peers if row["contacted"]}
        ),
        "primary_connected_contact_peer_level_count": len(primary),
        "primary_connected_contact_dependency_group_count": len(primary_groups),
        "primary_connected_contact_dependency_groups": ";".join(primary_groups),
        "primary_connected_contact_timeframes": ";".join(
            sorted(set(primary_timeframes), key=timeframe_hours)
        ),
        "parent_level_persistence_bars": persistence,
        "parent_prior_contact_hours_in_current_persistence": int(prior_inside.sum()),
        "parent_prior_contact_episodes_in_current_persistence": prior_episode_count,
    }
    return rows, summary


def assign_contact_components(peers: list[dict[str, Any]]) -> None:
    contacted = sorted(
        (row for row in peers if row["contacted"]),
        key=lambda row: (row["lower"], row["upper"], row["aligned_index"]),
    )
    component_id = -1
    running_upper: float | None = None
    for row in contacted:
        if running_upper is None or float(row["lower"]) > running_upper:
            component_id += 1
            running_upper = float(row["upper"])
        else:
            running_upper = max(running_upper, float(row["upper"]))
        row["contact_component_id"] = component_id
    for row in peers:
        row.setdefault("contact_component_id", None)


def validate_cluster_summary(pool: DataFrame) -> None:
    if pool["cluster_causal_anchor_timeframe"].isna().any():
        raise ValueError("At least one qualifying episode lacks a causal cluster anchor.")
    if pool["primary_connected_contact_dependency_group_count"].lt(2).any():
        raise ValueError("A qualifying episode has fewer than two connected peer mechanisms.")
    for source in COHORT_SOURCES:
        selected = pool.loc[pool["cohort"].eq(source.cohort)]
        cells = selected.groupby(["period", "level_name"], observed=True).size()
        for period in source.periods:
            for level_name in LEVEL_NAMES:
                if int(cells.get((period, level_name), 0)) < 1:
                    raise ValueError(
                        f"No qualifying {source.cohort} episode for {period} {level_name}."
                    )


def select_diagnostic_sample(pool: DataFrame) -> DataFrame:
    chosen_ids: list[str] = []
    ranks: dict[str, int] = {}
    rank = 0
    for source in COHORT_SOURCES:
        used_pairs: set[str] = set()
        cohort = pool.loc[pool["cohort"].eq(source.cohort)]
        for period in source.periods:
            for level_name in LEVEL_NAMES:
                cell = cohort.loc[
                    cohort["period"].eq(period) & cohort["level_name"].eq(level_name)
                ].sort_values("selection_hash")
                if cell.empty:
                    raise ValueError(f"Empty sample cell: {source.cohort} {period} {level_name}")
                unused = cell.loc[~cell["pair"].isin(used_pairs)]
                selected = unused.iloc[0] if not unused.empty else cell.iloc[0]
                episode_id = str(selected["episode_id"])
                chosen_ids.append(episode_id)
                ranks[episode_id] = rank
                rank += 1
                used_pairs.add(str(selected["pair"]))
        if len(used_pairs) != 6:
            raise ValueError(f"The {source.cohort} sample did not retain six distinct pairs.")
    sample = pool.loc[pool["episode_id"].isin(chosen_ids)].copy()
    sample["sample_selection_order"] = sample["episode_id"].map(ranks).astype(int)
    sample["selected_for_one_minute_replay"] = True
    sample["sample_stratum"] = sample.apply(
        lambda row: f"{row['cohort']}|{row['period']}|{row['level_name']}", axis=1
    )
    return sample.sort_values("sample_selection_order").reset_index(drop=True)


def acquisition_intervals(sample: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for _, episode in sample.iterrows():
        anchor = str(episode["cluster_causal_anchor_timeframe"])
        if anchor not in WINDOW_HOURS:
            raise ValueError(f"Unsupported causal anchor timeframe: {anchor}")
        pre_hours, post_hours = WINDOW_HOURS[anchor]
        event_time = pd.Timestamp(episode["event_time"])
        visible_start = event_time - pd.Timedelta(hours=pre_hours)
        # One extra hour includes a contact that occurs at the end of the parent candle.
        visible_end_exclusive = event_time + pd.Timedelta(hours=post_hours + 1)
        interval_start = visible_start - pd.Timedelta(hours=FEATURE_WARMUP_HOURS)
        interval_end_exclusive = visible_end_exclusive
        pair = str(episode["pair"])
        rows.append(
            {
                "episode_ids": str(episode["episode_id"]),
                "cohorts": str(episode["cohort"]),
                "pair": pair,
                "parent_contact_hours_utc": event_time.isoformat(),
                "causal_anchor_timeframes": anchor,
                "maximum_visible_pre_hours": pre_hours,
                "maximum_visible_post_hours": post_hours,
                "feature_warmup_hours": FEATURE_WARMUP_HOURS,
                "visible_start_utc": visible_start,
                "visible_end_exclusive_utc": visible_end_exclusive,
                "interval_start_utc": interval_start,
                "interval_end_exclusive_utc": interval_end_exclusive,
                "exchange": "binance",
                "market_type": "futures",
                "timeframe": "1m",
                "data_format": "feather",
            }
        )
    return merge_overlapping_acquisition_intervals(DataFrame(rows))


def merge_overlapping_acquisition_intervals(frame: DataFrame) -> DataFrame:
    merged: list[dict[str, Any]] = []
    for _, group in frame.groupby("pair", sort=False):
        current: dict[str, Any] | None = None
        for _, source in group.sort_values("interval_start_utc").iterrows():
            row = source.to_dict()
            if current is None:
                current = row
                continue
            if pd.Timestamp(row["interval_start_utc"]) <= pd.Timestamp(
                current["interval_end_exclusive_utc"]
            ) + pd.Timedelta(minutes=1):
                current["episode_ids"] = ";".join(
                    [str(current["episode_ids"]), str(row["episode_ids"])]
                )
                current["cohorts"] = ";".join(
                    sorted(set(str(current["cohorts"]).split(";") + str(row["cohorts"]).split(";")))
                )
                current["parent_contact_hours_utc"] = ";".join(
                    [
                        str(current["parent_contact_hours_utc"]),
                        str(row["parent_contact_hours_utc"]),
                    ]
                )
                current["causal_anchor_timeframes"] = ";".join(
                    sorted(
                        set(
                            str(current["causal_anchor_timeframes"]).split(";")
                            + str(row["causal_anchor_timeframes"]).split(";")
                        ),
                        key=timeframe_hours,
                    )
                )
                current["maximum_visible_pre_hours"] = max(
                    int(current["maximum_visible_pre_hours"]),
                    int(row["maximum_visible_pre_hours"]),
                )
                current["maximum_visible_post_hours"] = max(
                    int(current["maximum_visible_post_hours"]),
                    int(row["maximum_visible_post_hours"]),
                )
                current["visible_start_utc"] = min(
                    pd.Timestamp(current["visible_start_utc"]),
                    pd.Timestamp(row["visible_start_utc"]),
                )
                current["visible_end_exclusive_utc"] = max(
                    pd.Timestamp(current["visible_end_exclusive_utc"]),
                    pd.Timestamp(row["visible_end_exclusive_utc"]),
                )
                current["interval_start_utc"] = min(
                    pd.Timestamp(current["interval_start_utc"]),
                    pd.Timestamp(row["interval_start_utc"]),
                )
                current["interval_end_exclusive_utc"] = max(
                    pd.Timestamp(current["interval_end_exclusive_utc"]),
                    pd.Timestamp(row["interval_end_exclusive_utc"]),
                )
            else:
                merged.append(finalize_acquisition_interval(current))
                current = row
        if current is not None:
            merged.append(finalize_acquisition_interval(current))
    return DataFrame(merged)


def finalize_acquisition_interval(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    start = pd.Timestamp(output["interval_start_utc"])
    end = pd.Timestamp(output["interval_end_exclusive_utc"])
    # Freqtrade accepts exact intraday boundaries as ten-digit UTC Unix seconds.
    timerange = f"{int(start.timestamp())}-{int(end.timestamp())}"
    pair = str(output["pair"])
    local_file = FUTURES_DATA_DIR / f"{pair_stem(pair)}-1m-futures.feather"
    output["expected_one_minute_rows"] = int((end - start).total_seconds() // 60)
    output["timerange"] = timerange
    output["data_directory"] = str(DATA_DIR.resolve())
    output["expected_local_file"] = str(local_file.resolve())
    output["download_command"] = (
        f".venv\\Scripts\\python.exe -m freqtrade download-data --exchange binance "
        f"--trading-mode futures --candle-types futures --data-format-ohlcv feather "
        f"--datadir {DATA_DIR} -p {pair} -t 1m --timerange {timerange}"
    )
    return output


def audit_one_minute_coverage(
    run_id: str,
    *,
    report_root: Path = REPORT_ROOT,
    record_filename: str = "g3g_freeze_record.json",
    coverage_prefix: str = "g3g",
    update_freeze_record: bool = True,
    acquisition_note: str | None = None,
) -> int:
    run_dir = report_root / run_id
    record_path = run_dir / record_filename
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("status") != "frozen_before_one_minute_paths":
        raise ValueError(f"One-minute episode pool is not frozen: {record_path}")
    artifact = record["artifacts"]["acquisition_intervals_csv"]
    acquisition_path = Path(str(artifact["path"]))
    if sha256_file(acquisition_path) != str(artifact["sha256"]):
        raise ValueError("The frozen acquisition interval file changed after selection.")
    intervals = pd.read_csv(acquisition_path).sort_values(
        ["expected_local_file", "interval_start_utc"]
    )
    rows: list[dict[str, Any]] = []
    loaded_path: Path | None = None
    loaded_frame: DataFrame | None = None
    for _, interval in intervals.iterrows():
        path = Path(str(interval["expected_local_file"]))
        result: dict[str, Any] = {
            "episode_ids": str(interval["episode_ids"]),
            "cohorts": str(interval["cohorts"]),
            "pair": str(interval["pair"]),
            "interval_start_utc": str(interval["interval_start_utc"]),
            "interval_end_exclusive_utc": str(interval["interval_end_exclusive_utc"]),
            "expected_rows": int(interval["expected_one_minute_rows"]),
            "source_file": str(path.resolve()),
            "source_file_exists": path.is_file(),
            "download_command": str(interval["download_command"]),
        }
        if not path.is_file():
            result.update(
                {
                    "actual_rows": 0,
                    "missing_rows": int(interval["expected_one_minute_rows"]),
                    "duplicate_minutes": 0,
                    "non_one_minute_gaps": 0,
                    "invalid_ohlcv_rows": 0,
                    "passed": False,
                    "failure": "source_file_missing",
                }
            )
            rows.append(result)
            continue
        if loaded_path != path or loaded_frame is None:
            loaded_path = path
            loaded_frame = pd.read_feather(path)
        frame = loaded_frame
        required = {"date", "open", "high", "low", "close", "volume"}
        missing_columns = sorted(required.difference(frame.columns))
        if missing_columns:
            raise ValueError(f"Minute file lacks OHLCV columns: {path}: {missing_columns}")
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        start = pd.Timestamp(interval["interval_start_utc"])
        end = pd.Timestamp(interval["interval_end_exclusive_utc"])
        selected = frame.loc[(frame["date"] >= start) & (frame["date"] < end)].copy()
        selected = selected.sort_values("date").reset_index(drop=True)
        differences = selected["date"].diff().dropna()
        numeric = selected[["open", "high", "low", "close", "volume"]].apply(
            pd.to_numeric, errors="coerce"
        )
        invalid = (
            ~np.isfinite(numeric).all(axis=1)
            | numeric["volume"].lt(0.0)
            | numeric["high"].lt(numeric["low"])
            | numeric["high"].lt(numeric[["open", "close"]].max(axis=1))
            | numeric["low"].gt(numeric[["open", "close"]].min(axis=1))
        )
        expected = int(interval["expected_one_minute_rows"])
        actual = len(selected)
        duplicate_minutes = int(selected["date"].duplicated().sum())
        gaps = int(differences.ne(pd.Timedelta(minutes=1)).sum())
        first_expected = start
        last_expected = end - pd.Timedelta(minutes=1)
        first_actual = selected["date"].min() if actual else pd.NaT
        last_actual = selected["date"].max() if actual else pd.NaT
        passed = bool(
            actual == expected
            and duplicate_minutes == 0
            and gaps == 0
            and int(invalid.sum()) == 0
            and first_actual == first_expected
            and last_actual == last_expected
        )
        result.update(
            {
                "actual_rows": actual,
                "missing_rows": expected - actual,
                "first_expected_utc": first_expected,
                "first_actual_utc": first_actual,
                "last_expected_utc": last_expected,
                "last_actual_utc": last_actual,
                "duplicate_minutes": duplicate_minutes,
                "non_one_minute_gaps": gaps,
                "maximum_gap_minutes": (
                    float(differences.max() / pd.Timedelta(minutes=1))
                    if not differences.empty
                    else 0.0
                ),
                "invalid_ohlcv_rows": int(invalid.sum()),
                "source_file_rows_total": len(frame),
                "source_file_bytes": path.stat().st_size,
                "source_file_sha256": sha256_file(path),
                "passed": passed,
                "failure": "" if passed else "coverage_or_ohlcv_integrity_failure",
            }
        )
        rows.append(result)
    coverage = DataFrame(rows).sort_values(["pair", "interval_start_utc"])
    coverage_path = run_dir / f"{coverage_prefix}_one_minute_coverage_audit.csv"
    coverage_record_path = run_dir / f"{coverage_prefix}_one_minute_coverage_record.json"
    atomic_write_csv(coverage, coverage_path)
    coverage_record = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "status": "passed" if bool(coverage["passed"].all()) else "failed",
        "intervals": len(coverage),
        "passed_intervals": int(coverage["passed"].sum()),
        "expected_rows": int(coverage["expected_rows"].sum()),
        "actual_rows": int(coverage["actual_rows"].sum()),
        "duplicates": int(coverage["duplicate_minutes"].sum()),
        "gaps": int(coverage["non_one_minute_gaps"].sum()),
        "invalid_ohlcv_rows": int(coverage["invalid_ohlcv_rows"].sum()),
        "audit_csv": artifact_record(coverage_path),
        "selection_request_sha256": record["request_sha256"],
        "acquisition_note": acquisition_note
        or (
            "The first 1000PEPE smoke command used an unsupported minute-form timerange, "
            "failed before download, and was replaced by exact ten-digit UTC Unix-second "
            "boundaries before any interval was accepted."
        ),
        "directional_path_read": False,
    }
    atomic_write_json(coverage_record, coverage_record_path)
    if update_freeze_record:
        record["one_minute_coverage"] = {
            "status": coverage_record["status"],
            "coverage_csv": artifact_record(coverage_path),
            "coverage_record": artifact_record(coverage_record_path),
        }
        atomic_write_json(record, record_path)
    if coverage_record["status"] != "passed":
        raise RuntimeError(f"One-minute coverage audit failed: {coverage_path}")
    return 0


def sort_episode_rows(frame: DataFrame) -> DataFrame:
    columns = [column for column in ("cohort", "period", "level_name", "event_time") if column in frame]
    return frame.sort_values(columns).reset_index(drop=True)


def sort_component_rows(frame: DataFrame) -> DataFrame:
    return frame.sort_values(
        [
            "cohort",
            "episode_id",
            "component_kind",
            "component_timeframe",
            "component_family",
            "component_name",
        ]
    ).reset_index(drop=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def count_by(frame: DataFrame, column: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame[column].value_counts().items()}


def nested_counts(frame: DataFrame, columns: list[str]) -> dict[str, int]:
    counts = frame.groupby(columns, observed=True).size()
    return {"|".join(map(str, key if isinstance(key, tuple) else (key,))): int(value) for key, value in counts.items()}


if __name__ == "__main__":
    raise SystemExit(main())

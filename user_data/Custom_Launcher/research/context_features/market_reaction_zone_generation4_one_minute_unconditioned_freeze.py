from __future__ import annotations

# G4A repair iteration: freeze a high-thinness LVN plus independent geometric-cluster
# sample without using completed contact-hour volume or later peer-contact outcomes.
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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_lvn_attribution as g3a,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_one_minute_replay as g3g,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    atomic_write_csv,
    atomic_write_json,
    load_manifest,
    numeric_array,
    prepare_base_market_frame,
    sha256_file,
    timeframe_hours,
    utc_now,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_volume_profile_roles import (  # noqa: E501
    assign_attribute_tertiles,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_vp_roles import (  # noqa: E501
    stable_json_sha256,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_lvn_attribution import (  # noqa: E501
    aligned_selected_levels_from_verified_prefix,
    component_dependency_group,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation4_one_minute_breadth import (  # noqa: E501
    ARTIFACT_ROOT,
    PERIODS,
    REPORT_ROOT,
    TARGET_EPISODES_PER_COHORT,
    balanced_cell_quotas,
    count_by,
    cross_cohort_bridge_duplicates,
    select_globally_independent_pool,
    validate_g4_branch,
    validate_sample,
)


SCHEMA_VERSION = 1
SELECTION_SEED = "g4a_unconditioned_geometric_cluster_v1"
LEVEL_NAMES = ("lvn_above", "lvn_below")
EXPECTED_APPROACH = {"lvn_above": "from_below", "lvn_below": "from_above"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the repaired G4A sample without any post-contact volume, movement, "
            "direction, pressure, profit, or later peer-contact selection."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    validate_g4_branch()
    return freeze_unconditioned_sample(args.run_id, overwrite=bool(args.overwrite))


def freeze_unconditioned_sample(run_id: str, *, overwrite: bool) -> int:
    run_dir = REPORT_ROOT / run_id
    artifact_dir = ARTIFACT_ROOT / run_id
    record_path = run_dir / "g4a_freeze_record.json"
    if record_path.is_file() and not overwrite:
        raise FileExistsError(f"Repaired G4A run already exists: {record_path}")
    parent_contracts = [g3g.validate_parent(source) for source in g3g.COHORT_SOURCES]
    request = request_contract(run_id, parent_contracts)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "freezing",
        "started_at_utc": utc_now(),
        "request_sha256": stable_json_sha256(request),
        "request_contract": request,
    }
    atomic_write_json(record, record_path)
    try:
        raw_parts: list[DataFrame] = []
        raw_counts: dict[str, int] = {}
        for source in g3g.COHORT_SOURCES:
            part = build_unconditioned_cohort(source)
            raw_parts.append(part)
            raw_counts[source.cohort] = len(part)
        raw_pool = pd.concat(raw_parts, ignore_index=True, sort=False)
        globally_independent = select_globally_independent_pool(raw_pool)
        sample = select_bridge_safe_sample(globally_independent)
        components, summaries = freeze_geometric_components(sample)
        sample = sample.merge(
            summaries,
            on=["episode_id", "cohort", "pair", "event_time"],
            how="left",
            validate="one_to_one",
        )
        validate_geometric_summary(sample)
        validate_sample(sample)
        acquisition = g3g.acquisition_intervals(sample)

        sample_path = run_dir / "g4a_frozen_episode_sample.csv"
        component_path = run_dir / "g4a_frozen_sample_components.csv"
        acquisition_path = run_dir / "g4a_one_minute_acquisition_intervals.csv"
        candidate_path = artifact_dir / "g4a_unconditioned_candidate_pool.csv"
        atomic_write_csv(g3g.sort_episode_rows(sample), sample_path)
        atomic_write_csv(g3g.sort_component_rows(components), component_path)
        atomic_write_csv(acquisition.sort_values(["pair", "interval_start_utc"]), acquisition_path)
        atomic_write_csv(g3g.sort_episode_rows(globally_independent), candidate_path)

        record.update(
            {
                "status": "frozen_before_one_minute_paths",
                "completed_at_utc": utc_now(),
                "unconditioned_candidate_counts_before_global_independence": raw_counts,
                "global_48h_candidate_counts": count_by(globally_independent, "cohort"),
                "sample_episode_counts": count_by(sample, "cohort"),
                "sample_pair_counts": {
                    str(key): int(value)
                    for key, value in sample.groupby("cohort")["pair"].nunique().items()
                },
                "sample_period_counts": g3g.nested_counts(sample, ["cohort", "period"]),
                "sample_side_counts": g3g.nested_counts(sample, ["cohort", "level_name"]),
                "sample_anchor_counts": g3g.nested_counts(
                    sample, ["cohort", "cluster_causal_anchor_timeframe"]
                ),
                "cross_cohort_doge_bridge_duplicate_events": (
                    cross_cohort_bridge_duplicates(sample)
                ),
                "post_contact_volume_used_for_selection": False,
                "parent_hour_peer_contact_used_for_selection": False,
                "future_signed_path_used_for_selection": False,
                "acquisition_intervals": len(acquisition),
                "expected_one_minute_rows_before_file_overlap": int(
                    acquisition["expected_one_minute_rows"].sum()
                ),
                "artifacts": {
                    "frozen_sample_csv": g3g.artifact_record(sample_path),
                    "frozen_sample_components_csv": g3g.artifact_record(component_path),
                    "acquisition_intervals_csv": g3g.artifact_record(acquisition_path),
                    "globally_independent_candidate_pool_csv": g3g.artifact_record(candidate_path),
                },
            }
        )
        atomic_write_json(record, record_path)
        print(
            json.dumps(
                {
                    "status": record["status"],
                    "raw_candidates": raw_counts,
                    "globally_independent": record["global_48h_candidate_counts"],
                    "sample": record["sample_episode_counts"],
                    "pairs": record["sample_pair_counts"],
                    "anchors": record["sample_anchor_counts"],
                    "acquisition_intervals": len(acquisition),
                },
                indent=2,
            )
        )
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


def request_contract(run_id: str, parent_contracts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "parent_contracts": parent_contracts,
        "repair_reason": (
            "The first G4A pass inherited G3G's completed contact-hour volume >=2x "
            "condition and parent-hour peer-contact condition. Those post-contact "
            "conditions invalidate a prospective reaction confirmation."
        ),
        "selection": {
            "parent": (
                "actual high-tertile one-hour LVN thinness from the corrected G3A "
                "causal event surface"
            ),
            "clear_outside_approach": EXPECTED_APPROACH,
            "cluster": (
                "at least two independent non-LVN calculation mechanisms in a "
                "transitively connected peer-zone segment available before the hour"
            ),
            "cluster_contact_required_after_event": False,
            "contact_hour_volume_used": False,
            "contact_hour_pressure_used": False,
            "contact_hour_range_used": False,
            "future_price_path_used": False,
            "profit_used": False,
            "direction_used": False,
            "global_independence_hours": 48,
            "global_independence_scope": "within cohort across all pairs",
            "sample_target_per_cohort": TARGET_EPISODES_PER_COHORT,
            "sample_balance": "period x LVN side, one episode per pair before repeats",
            "bridge_rule": (
                "freeze normal first, then exclude identical DOGE event keys before freezing memes"
            ),
            "selection_seed": SELECTION_SEED,
        },
        "carried_but_not_used_for_selection": [
            "contact_volume_ratio",
            "contact_pressure_change",
            "contact_range_ratio",
            "dwell_fraction_h4",
            "crossings_h4",
            "volume_ratio_h48",
            "parent-hour peer contact flags",
        ],
        "one_minute_windows": {
            timeframe: {"pre_hours": values[0], "post_hours": values[1]}
            for timeframe, values in g3g.WINDOW_HOURS.items()
        },
        "iteration": 2,
        "g4a_iteration_cap": 3,
        "profit_optimization": False,
        "direction_prediction_during_selection": False,
    }


def build_unconditioned_cohort(source: Any) -> DataFrame:
    manifest = load_manifest(source.manifest_path)
    parent_record = json.loads(
        (g3g.PARENT_REPORT_ROOT / source.parent_run_id / "g3a_run_record.json").read_text(
            encoding="utf-8"
        )
    )
    g3a_cohort = str(parent_record["request_contract"]["cohort"])
    parts: list[DataFrame] = []
    for pair in manifest["data"]["pairs"]:
        parts.append(
            build_unconditioned_pair(
                str(pair),
                source=source,
                manifest=manifest,
                g3a_cohort=g3a_cohort,
            )
        )
    output = pd.concat(parts, ignore_index=True, sort=False)
    output["cohort"] = source.cohort
    output["parent_run_id"] = source.parent_run_id
    output["selection_key"] = output.apply(
        lambda row: (
            f"{SELECTION_SEED}|{source.cohort}|{row['period']}|{row['pair']}|"
            f"{pd.Timestamp(row['event_time']).isoformat()}|{row['level_name']}"
        ),
        axis=1,
    )
    output["selection_hash"] = output["selection_key"].map(sha256_text)
    output["episode_id"] = output["selection_hash"].map(
        lambda value: f"g4a_unconditioned_{source.cohort}_{value[:16]}"
    )
    output["g4a_selection_hash"] = output["episode_id"].map(
        lambda value: sha256_text(f"{SELECTION_SEED}|sample|{value}")
    )
    output["qualification_reason"] = (
        "corrected G3A causal high-tertile 1h LVN thinness; clear outside approach; "
        "independent mixed peer-zone geometry; no later contact, volume, movement, "
        "pressure, direction, or profit condition"
    )
    output["direction_used_for_selection"] = False
    output["retrospective_price_reference"] = False
    output["price_reference_known_at_event"] = True
    output["level_role"] = "prospective_reaction_location"
    output["source_timeframe"] = "1h"
    return output.reset_index(drop=True)


def build_unconditioned_pair(
    pair: str,
    *,
    source: Any,
    manifest: dict[str, Any],
    g3a_cohort: str,
) -> DataFrame:
    event_path, _, event_metadata = g3a.event_source(
        cohort=g3a_cohort,
        manifest=manifest,
        manifest_path=source.manifest_path,
        pair=pair,
        timeframe="1h",
    )
    events = pd.read_parquet(
        event_path,
        filters=[
            ("control", "==", "actual"),
            ("level_name", "in", list(g3a.QUESTION.profile.level_names)),
            ("zone_method", "==", g3a.QUESTION.zone),
        ],
    )
    events = g3a.eligible_period_events(events, manifest, embargo_hours=48)
    if events.empty:
        raise ValueError(f"No eligible causal one-hour LVN events for {pair}")
    base = prepare_base_market_frame(pair, manifest)
    local_state = g3a.causal_local_state(base).merge(
        g3a.causal_market_context(manifest),
        on="date",
        how="left",
        validate="one_to_one",
    )
    aligned = aligned_selected_levels_from_verified_prefix(
        pair=pair,
        base=base,
        manifest_path=source.manifest_path,
        timeframes=tuple(manifest["data"]["source_timeframes"]),
    )
    matrix = np.column_stack([item.level for item in aligned])
    base_atr = numeric_array(base["base_atr"])
    pre_close = numeric_array(base["pre_close"])
    local_state["state_selected_level_density_2atr"] = np.sum(
        np.isfinite(matrix) & (np.abs(matrix - pre_close[:, None]) <= 2.0 * base_atr[:, None]),
        axis=1,
    ).astype(float)
    events = g3a.attach_states(
        events,
        base=base,
        local_state=local_state,
        cache_path=Path(str(event_metadata["cache"])),
    )
    events["state_vp_value_area_width_pct"] = pd.to_numeric(
        events["attr_vp_value_area_width_pct"], errors="coerce"
    )
    events = g3a.attach_vp_level_persistence(
        events,
        base=base,
        cache_path=Path(str(event_metadata["cache"])),
    )
    events = g3a.attach_peer_geometry(events, base=base, aligned=aligned)
    selected = events.loc[
        pd.to_numeric(events[g3a.QUESTION.profile.attribute_column], errors="coerce").notna()
    ].copy()
    tiered = assign_attribute_tertiles(
        selected,
        attribute_column=g3a.QUESTION.profile.attribute_column,
        assignment="actual",
        seed_key=f"{pair}|{g3a.QUESTION.id}",
    )
    tiered = tiered.loc[
        tiered["period"].isin(source.periods)
        & tiered["attribute_tier"].eq("high")
        & tiered["peer_cluster_survives_without_lvn"].eq(True)
        & tiered["state_peer_connected_dependency_group_count"].ge(2)
        & tiered.apply(
            lambda row: str(row["approach_state"]) == EXPECTED_APPROACH[str(row["level_name"])],
            axis=1,
        )
    ].copy()
    tiered["event_time"] = pd.to_datetime(tiered["event_time"], utc=True)
    tiered["source_available_at"] = pd.to_datetime(tiered["source_available_at"], utc=True)
    tiered["source_open"] = pd.to_datetime(tiered["source_open"], utc=True)
    if (tiered["source_available_at"] > tiered["event_time"]).any():
        raise ValueError(f"Future LVN source admitted for {pair}")
    tiered["parent_high_thinness_score"] = pd.to_numeric(
        tiered[g3a.QUESTION.profile.attribute_column], errors="coerce"
    )
    return tiered


def select_bridge_safe_sample(candidates: DataFrame) -> DataFrame:
    normal = select_one_cohort(
        candidates.loc[candidates["cohort"].eq("normal")].copy(), cohort="normal"
    )
    normal_bridge = set(
        normal.loc[normal["pair"].eq("DOGE/USDT:USDT")]
        .apply(
            lambda row: (
                str(row["pair"]),
                pd.Timestamp(row["event_time"]),
                str(row["level_name"]),
            ),
            axis=1,
        )
        .tolist()
    )
    meme_source = candidates.loc[candidates["cohort"].eq("meme")].copy()
    meme_source = meme_source.loc[
        ~meme_source.apply(
            lambda row: (
                (
                    str(row["pair"]),
                    pd.Timestamp(row["event_time"]),
                    str(row["level_name"]),
                )
                in normal_bridge
            ),
            axis=1,
        )
    ].copy()
    meme = select_one_cohort(meme_source, cohort="meme")
    sample = pd.concat([normal, meme], ignore_index=True, sort=False)
    sample = sample.sort_values(["cohort", "g4a_selection_hash"]).reset_index(drop=True)
    sample["sample_selection_order"] = np.arange(len(sample), dtype=np.int64)
    sample["selected_for_one_minute_replay"] = True
    sample["sample_stratum"] = (
        sample["cohort"].astype(str)
        + "|"
        + sample["period"].astype(str)
        + "|"
        + sample["level_name"].astype(str)
    )
    return sample


def select_one_cohort(source: DataFrame, *, cohort: str) -> DataFrame:
    periods = PERIODS[cohort]
    if len(source) <= TARGET_EPISODES_PER_COHORT:
        return source.copy()
    cells = [(period, level_name) for period in periods for level_name in LEVEL_NAMES]
    quotas = balanced_cell_quotas(source, cells=cells, target=TARGET_EPISODES_PER_COHORT)
    parts: list[DataFrame] = []
    for period, level_name in cells:
        cell = source.loc[source["period"].eq(period) & source["level_name"].eq(level_name)].copy()
        cell["_pair_rank"] = cell.groupby("pair", observed=True).cumcount()
        cell = cell.sort_values(["_pair_rank", "g4a_selection_hash"], kind="mergesort")
        parts.append(cell.head(quotas[(period, level_name)]))
    return pd.concat(parts, ignore_index=True, sort=False).drop(
        columns="_pair_rank", errors="ignore"
    )


def freeze_geometric_components(sample: DataFrame) -> tuple[DataFrame, DataFrame]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    source_map = {source.cohort: source for source in g3g.COHORT_SOURCES}
    for (cohort, pair), group in sample.groupby(["cohort", "pair"], sort=False):
        source = source_map[str(cohort)]
        manifest = load_manifest(source.manifest_path)
        base = prepare_base_market_frame(str(pair), manifest)
        aligned = aligned_selected_levels_from_verified_prefix(
            pair=str(pair),
            base=base,
            manifest_path=source.manifest_path,
            timeframes=tuple(manifest["data"]["source_timeframes"]),
        )
        high = numeric_array(base["high"])
        low = numeric_array(base["low"])
        for _, episode in group.iterrows():
            component_rows, summary = geometric_episode_components(
                episode, aligned=aligned, base_high=high, base_low=low
            )
            rows.extend(component_rows)
            summaries.append(summary)
    return DataFrame(rows), DataFrame(summaries)


def geometric_episode_components(
    episode: Series,
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
        raise ValueError(f"Expected one causal LVN for {episode['episode_id']}")
    target_index, target = target_matches[0]
    target_price = float(target.level[base_index])
    if not bool(target.valid[base_index]) or not np.isclose(
        target_price, float(episode["level_price"]), rtol=1e-10, atol=1e-10
    ):
        raise ValueError(f"Frozen LVN does not reproduce: {episode['episode_id']}")
    atr = float(episode["base_atr"])
    target_width = max(0.5 * atr, abs(target_price) * 0.0005)
    target_group = component_dependency_group(target.spec)
    peers: list[dict[str, Any]] = []
    for index, item in enumerate(aligned):
        dependency = component_dependency_group(item.spec)
        if index == target_index or dependency == target_group or not bool(item.valid[base_index]):
            continue
        level = float(item.level[base_index])
        if not np.isfinite(level) or level <= 0.0:
            continue
        width = max(0.5 * atr, abs(level) * 0.0005)
        if abs(level - target_price) > width + target_width:
            continue
        lower = level - width
        upper = level + width
        peers.append(
            {
                "aligned_index": index,
                "item": item,
                "dependency_group": dependency,
                "level_price": level,
                "zone_half_width": width,
                "lower": lower,
                "upper": upper,
                "contacted_on_parent_hour": bool(
                    base_high[base_index] >= lower and base_low[base_index] <= upper
                ),
            }
        )
    components = connected_geometric_components(peers)
    eligible = [
        component
        for component in components
        if len({row["dependency_group"] for row in component}) >= 2
    ]
    if not eligible:
        raise ValueError(f"No independent geometric peer cluster: {episode['episode_id']}")
    primary = sorted(
        eligible,
        key=lambda component: (
            -len({row["dependency_group"] for row in component}),
            -len(component),
            min(row["aligned_index"] for row in component),
        ),
    )[0]
    primary_indexes = {int(row["aligned_index"]) for row in primary}
    groups = sorted({str(row["dependency_group"]) for row in primary})
    timeframes = sorted({str(row["item"].timeframe) for row in primary}, key=timeframe_hours)
    anchor = max(("1h", *timeframes), key=timeframe_hours)
    common = {
        "episode_id": str(episode["episode_id"]),
        "cohort": str(episode["cohort"]),
        "pair": str(episode["pair"]),
        "event_time": event_time,
        "parent_level_name": str(episode["level_name"]),
        "parent_level_price": target_price,
        "parent_zone_half_width": float(episode["zone_half_width"]),
        "geometry_anchor_half_width": target_width,
        "cluster_causal_anchor_timeframe": anchor,
        "cluster_selection_basis": "precontact_available_geometric_zone_connection",
        "parent_hour_peer_contact_used_for_selection": False,
    }
    component_rows: list[dict[str, Any]] = [
        component_row(
            common,
            item=target,
            kind="parent_lvn",
            dependency_group=target_group,
            level_price=target_price,
            width=target_width,
            base_index=base_index,
            in_primary=True,
            contacted=True,
        )
    ]
    for peer in peers:
        component_rows.append(
            component_row(
                common,
                item=peer["item"],
                kind="peer_level",
                dependency_group=str(peer["dependency_group"]),
                level_price=float(peer["level_price"]),
                width=float(peer["zone_half_width"]),
                base_index=base_index,
                in_primary=int(peer["aligned_index"]) in primary_indexes,
                contacted=bool(peer["contacted_on_parent_hour"]),
            )
        )
    persistence = int(max(1, round(float(episode["state_vp_level_persistence_bars"]))))
    prior_start = max(0, base_index - persistence)
    prior_inside = (
        base_high[prior_start:base_index] >= target_price - float(episode["zone_half_width"])
    ) & (base_low[prior_start:base_index] <= target_price + float(episode["zone_half_width"]))
    summary = {
        "episode_id": str(episode["episode_id"]),
        "cohort": str(episode["cohort"]),
        "pair": str(episode["pair"]),
        "event_time": event_time,
        "cluster_causal_anchor_timeframe": anchor,
        "nearby_peer_level_count": len(peers),
        "contacted_peer_level_count": int(
            sum(bool(row["contacted_on_parent_hour"]) for row in peers)
        ),
        "nearby_peer_dependency_group_count": len({str(row["dependency_group"]) for row in peers}),
        "contacted_peer_dependency_group_count": len(
            {str(row["dependency_group"]) for row in peers if row["contacted_on_parent_hour"]}
        ),
        "primary_connected_contact_peer_level_count": len(primary),
        "primary_connected_contact_dependency_group_count": len(groups),
        "primary_connected_contact_dependency_groups": ";".join(groups),
        "primary_connected_contact_timeframes": ";".join(timeframes),
        "primary_connected_geometric_peer_level_count": len(primary),
        "primary_connected_geometric_dependency_group_count": len(groups),
        "primary_connected_geometric_dependency_groups": ";".join(groups),
        "primary_connected_geometric_timeframes": ";".join(timeframes),
        "parent_level_persistence_bars": persistence,
        "parent_prior_contact_hours_in_current_persistence": int(prior_inside.sum()),
        "parent_prior_contact_episodes_in_current_persistence": int(
            np.sum(prior_inside & ~np.r_[False, prior_inside[:-1]]) if len(prior_inside) else 0
        ),
        "parent_hour_peer_contact_used_for_selection": False,
    }
    return component_rows, summary


def connected_geometric_components(peers: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    ordered = sorted(peers, key=lambda row: (row["lower"], row["upper"], row["aligned_index"]))
    components: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    running_upper: float | None = None
    for row in ordered:
        if running_upper is None or float(row["lower"]) > running_upper:
            if current:
                components.append(current)
            current = [row]
            running_upper = float(row["upper"])
        else:
            current.append(row)
            running_upper = max(running_upper, float(row["upper"]))
    if current:
        components.append(current)
    return components


def component_row(
    common: dict[str, Any],
    *,
    item: Any,
    kind: str,
    dependency_group: str,
    level_price: float,
    width: float,
    base_index: int,
    in_primary: bool,
    contacted: bool,
) -> dict[str, Any]:
    available = pd.Timestamp(item.source_available.iloc[base_index])
    event_time = pd.Timestamp(common["event_time"])
    if available > event_time:
        raise ValueError(f"Future component admitted for {common['episode_id']}")
    return {
        **common,
        "component_kind": kind,
        "component_family": item.spec.family,
        "component_name": item.spec.name,
        "component_column": item.spec.column,
        "component_representation": item.spec.representation,
        "component_timeframe": item.timeframe,
        "component_dependency_group": dependency_group,
        "component_level_price": level_price,
        "component_zone_half_width": width,
        "component_contacted_on_parent_hour": contacted,
        "component_contact_used_for_selection": False,
        "contact_component_id": 0 if in_primary else None,
        "in_primary_connected_contact_cluster": in_primary,
        "in_primary_connected_geometric_cluster": in_primary,
        "component_source_available_at": available,
        "component_source_open": item.source_open.iloc[base_index],
        "component_active": bool(item.valid[base_index]),
        "component_source_age_hours": (event_time - available).total_seconds() / 3600.0,
        "component_source_key": (
            f"{item.timeframe}|{item.spec.family}|{item.spec.column}|"
            f"{item.spec.representation}|{available.isoformat()}"
        ),
    }


def validate_geometric_summary(sample: DataFrame) -> None:
    if sample["cluster_causal_anchor_timeframe"].isna().any():
        raise ValueError("A repaired G4A event lacks a causal geometric anchor")
    if sample["primary_connected_geometric_dependency_group_count"].lt(2).any():
        raise ValueError("A repaired G4A event has fewer than two geometric mechanisms")
    if sample["parent_hour_peer_contact_used_for_selection"].astype(bool).any():
        raise ValueError("Parent-hour peer contact leaked into repaired selection")
    if cross_cohort_bridge_duplicates(sample) != 0:
        raise ValueError("Repaired sample duplicates a DOGE event across cohorts")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

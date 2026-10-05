from __future__ import annotations

# Bound numerical libraries before importing numpy/pandas or research helpers.
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
import json
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.market_reaction_zone_freqai_generation4 import (  # noqa: E501
    APPROACH_STATES,
    ARRIVAL_STATES,
    independently_spaced_events,
    one_hot,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    OUTPUT_ROOT,
    atomic_write_json,
    atomic_write_parquet,
    future_path_matrices,
    normalize_dates,
    numeric_array,
    prepare_base_market_frame,
    sha256_file,
    utc_now,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_localization import (  # noqa: E501
    causal_local_state,
    causal_market_context,
    pair_stem,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_volume_profile_roles import (  # noqa: E501
    assign_attribute_tertiles,
    attach_states,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_density_reaction import (  # noqa: E501  # noqa: E501
    attach_structure_overlaps,
    build_density_surfaces,
    classify_actual_scope,
    density_matrices,
    purge_density_matches,
    reference_matrices,
    selected_reference_levels,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_localization import (  # noqa: E501
    eligible_period_events,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_vp_roles import (  # noqa: E501
    atlas_event_source,
    matched_attribute_event_pairs,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_density_arrival import (  # noqa: E501
    OUTCOME_EMBARGO_HOURS,
    surface_event_geometry,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_density_arrival_states import (  # noqa: E501
    attach_geometry_event_outcomes,
    direct_state_match_rows,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_lvn_attribution import (  # noqa: E501
    MATCH_SEPARATION_HOURS,
    QUESTION,
    STATE_FEATURES,
    aligned_selected_levels_from_verified_prefix,
    attach_peer_geometry,
    attach_vp_level_persistence,
)


G0_SCRIPT = (
    REPO_ROOT
    / "user_data"
    / "Custom_Launcher"
    / "research"
    / "context_features"
    / "market_reaction_zone_generation0.py"
)
BASE_MANIFEST = OUTPUT_ROOT / "generation0_manifest.json"
FROZEN_BATCH = OUTPUT_ROOT / "generation4_review" / "g5_frozen_branch_batch.json"
DOWNLOAD_RECORD = (
    OUTPUT_ROOT
    / "generation5_branches"
    / "g5c_new_period_normal_excursion_confirmation"
    / "g5c_source_update_20260820a"
    / "g5c_download_record.json"
)
RECORD_ROOT = (
    OUTPUT_ROOT
    / "generation5_branches"
    / "g5c_new_period_normal_excursion_confirmation"
)
LARGE_ROOT = (
    Path("D:/FreqTradeStuffLargeData")
    / "research_outputs"
    / "market_reaction_zones"
    / "generation5_branches"
    / "g5c_new_period_normal_excursion_confirmation"
)

OUTPUT_SCHEMA_VERSION = 1
MAX_WORKERS = 4
MIN_SCORABLE_ROWS = 20
MIN_CONFIRMATION_COINS = 5
TARGET_COLUMN = "&-g5c_abs_excursion_atr_h1"
CONFIRMATION_PERIODS = ("g5c_confirmation_early", "g5c_confirmation_late")
SURFACES = (
    "normal_thin_lvn_mixed_cluster",
    "normal_isolated_confirmed_swing_density",
    "normal_confirmed_swing_density_cluster",
)
DENSITY_SCOPES = {
    "normal_isolated_confirmed_swing_density": "single_density_zone",
    "normal_confirmed_swing_density_cluster": "density_cluster",
}
WIDE_COLUMNS = (
    "state_btc_return_1h",
    "state_btc_return_24h",
    "state_btc_atr_pct",
    "state_top10_breadth",
    "state_top10_mean_abs_return",
    "state_top10_return_dispersion",
)


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def validate_frozen_branch() -> dict[str, Any]:
    frozen = json.loads(FROZEN_BATCH.read_text(encoding="utf-8"))
    branches = {
        str(item["id"]): item for item in frozen.get("branches", [])
    }
    branch = branches.get("g5c_new_period_normal_excursion_confirmation")
    if branch is None or branch.get("status") != "frozen_next_batch":
        raise ValueError("The frozen G5C branch is unavailable.")
    if tuple(branch.get("fixed_surfaces", ())) != SURFACES:
        raise ValueError("The G5C surface list changed after its outcome freeze.")
    if branch.get("fixed_target") != "next_hour_absolute_excursion_in_prior_atr":
        raise ValueError("The G5C target changed after its outcome freeze.")
    return branch


def source_manifest_contract(run_id: str) -> tuple[dict[str, Any], Path, Path, Path]:
    validate_frozen_branch()
    download = json.loads(DOWNLOAD_RECORD.read_text(encoding="utf-8"))
    base = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    record_dir = RECORD_ROOT / run_id
    artifact_dir = LARGE_ROOT / run_id
    storage_root = artifact_dir / "fresh_source"
    slices = download["verification"]["candidate_confirmation_slices_frozen_from_coverage_only"]
    expected = [
        {
            "id": "g5c_confirmation_early",
            "start_utc": "2026-07-20T00:00:00Z",
            "end_utc_exclusive": "2026-08-05T00:00:00Z",
        },
        {
            "id": "g5c_confirmation_late",
            "start_utc": "2026-08-05T00:00:00Z",
            "end_utc_exclusive": "2026-08-20T00:00:00Z",
        },
    ]
    if slices != expected:
        raise ValueError("The coverage-frozen G5C slices changed.")
    base["generation"] = "5c_fresh_source"
    base["frozen_at_utc"] = download["completed_at_utc"]
    base["research_boundary"]["purpose"] = (
        "Build causal level/event geometry for the exposed training bridge and two "
        "outcome-unopened G5C confirmation slices. Direction and profit remain disabled."
    )
    base["data"]["common_analysis_start_utc"] = "2026-04-01T00:00:00Z"
    base["data"]["analysis_end"] = "2026-08-20T00:00:00Z exclusive"
    base["data"]["chronological_periods"] = [
        {
            "id": "g5c_exposed_training_bridge",
            "start_utc": "2026-04-01T00:00:00Z",
            "end_utc_exclusive": "2026-07-20T00:00:00Z",
            "role": "exposed_training_only_not_confirmation",
        },
        {
            **expected[0],
            "role": "newly_accumulated_confirmation",
        },
        {
            **expected[1],
            "role": "newly_accumulated_confirmation",
        },
    ]
    base["data"]["untouched_confirmation"] = (
        "Only g5c_confirmation_early and g5c_confirmation_late are confirmation. "
        "The April-July interval is exposed and may train models but may not confirm them."
    )
    base["storage"] = {
        "cache_dir": str((storage_root / "level_cache").resolve()),
        "report_dir": str((record_dir / "fresh_source_reports").resolve()),
        "event_dir": str((storage_root / "atlas_events").resolve()),
        "detailed_summary_dir": str((storage_root / "atlas_summaries").resolve()),
    }
    base["g5c_source_contract"] = {
        "run_id": run_id,
        "download_record": str(DOWNLOAD_RECORD.resolve()),
        "download_record_sha256": sha256_file(DOWNLOAD_RECORD),
        "base_manifest": str(BASE_MANIFEST.resolve()),
        "base_manifest_sha256": sha256_file(BASE_MANIFEST),
        "outcomes_opened_before_manifest_freeze": False,
        "confirmation_slices": expected,
    }
    manifest_path = record_dir / "g5c_source_manifest.json"
    return base, manifest_path, record_dir, artifact_dir


def write_source_manifest(run_id: str) -> tuple[dict[str, Any], Path, Path, Path]:
    manifest, path, record_dir, artifact_dir = source_manifest_contract(run_id)
    record_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if stable_json(existing) != stable_json(manifest):
            raise ValueError(f"Existing G5C source manifest changed: {path}")
    else:
        atomic_write_json(manifest, path)
    return manifest, path, record_dir, artifact_dir


def run_source_command(
    command: Sequence[str],
    *,
    log_path: Path,
) -> dict[str, Any]:
    started = utc_now()
    result = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "STDOUT\n" + result.stdout + "\nSTDERR\n" + result.stderr,
        encoding="utf-8",
    )
    record = {
        "command": list(command),
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "returncode": int(result.returncode),
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
    }
    if result.returncode:
        raise RuntimeError(f"G5C source command failed; inspect {log_path}")
    return record


def build_fresh_source(run_id: str, *, workers: int) -> dict[str, Any]:
    manifest, manifest_path, record_dir, _ = write_source_manifest(run_id)
    if workers < 1 or workers > MAX_WORKERS:
        raise ValueError(f"G5C source workers must be in 1..{MAX_WORKERS}.")
    commands = [
        (
            "preflight",
            [
                sys.executable,
                str(G0_SCRIPT),
                "--manifest",
                str(manifest_path),
                "preflight",
                "--skip-indicator-smoke",
            ],
        ),
        (
            "build_cache",
            [
                sys.executable,
                str(G0_SCRIPT),
                "--manifest",
                str(manifest_path),
                "build-cache",
                "--pairs",
                "all",
                "--timeframes",
                "all",
                "--families",
                "core,generic",
                "--workers",
                str(workers),
            ],
        ),
        (
            "atlas",
            [
                sys.executable,
                str(G0_SCRIPT),
                "--manifest",
                str(manifest_path),
                "atlas",
                "--pairs",
                "all",
                "--timeframes",
                "all",
                "--cache-families",
                "core,generic",
                "--level-batches",
                "g0b1,g0b2",
                "--zones",
                "tight_base_atr,wide_base_atr",
                "--controls",
                "actual",
                "--workers",
                str(workers),
            ],
        ),
    ]
    records = []
    for name, command in commands:
        records.append(
            {
                "phase": name,
                **run_source_command(
                    command,
                    log_path=record_dir / f"g5c_source_{name}.log",
                ),
            }
        )
    output = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "pairs": list(manifest["data"]["pairs"]),
        "timeframes": list(manifest["data"]["source_timeframes"]),
        "commands": records,
        "outcomes_opened": False,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(output, record_dir / "g5c_source_build_record.json")
    return output


def source_wide_context(manifest: dict[str, Any]) -> DataFrame:
    frame = causal_market_context(manifest)[["date", *WIDE_COLUMNS]].copy()
    frame = frame.rename(
        columns={column: f"wide__{column.removeprefix('state_')}" for column in WIDE_COLUMNS}
    )
    frame["date"] = normalize_dates(frame["date"])
    return frame


def thin_candidate_events(
    pair: str,
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> DataFrame:
    event_path, _, event_metadata = atlas_event_source(
        manifest=manifest,
        manifest_path=manifest_path,
        pair=pair,
        timeframe="1h",
    )
    raw_events = pd.read_parquet(
        event_path,
        filters=[
            ("control", "==", "actual"),
            ("level_name", "in", list(QUESTION.profile.level_names)),
            ("zone_method", "==", QUESTION.zone),
        ],
    )
    raw_events = eligible_period_events(raw_events, manifest, embargo_hours=48)
    if raw_events.empty:
        raise ValueError(f"No eligible fresh thin-LVN events for {pair}.")
    base = prepare_base_market_frame(pair, manifest)
    local_state = causal_local_state(base).merge(
        causal_market_context(manifest),
        on="date",
        how="left",
        validate="one_to_one",
    )
    aligned = aligned_selected_levels_from_verified_prefix(
        pair=pair,
        base=base,
        manifest_path=manifest_path,
        timeframes=tuple(manifest["data"]["source_timeframes"]),
    )
    matrix = np.column_stack([item.level for item in aligned])
    base_atr = numeric_array(base["base_atr"])
    pre_close = numeric_array(base["pre_close"])
    local_state["state_selected_level_density_2atr"] = np.sum(
        np.isfinite(matrix)
        & (np.abs(matrix - pre_close[:, None]) <= 2.0 * base_atr[:, None]),
        axis=1,
    ).astype(float)
    events = attach_states(
        raw_events,
        base=base,
        local_state=local_state,
        cache_path=Path(str(event_metadata["cache"])),
    )
    events["state_vp_value_area_width_pct"] = pd.to_numeric(
        events["attr_vp_value_area_width_pct"], errors="coerce"
    )
    events = attach_vp_level_persistence(
        events,
        base=base,
        cache_path=Path(str(event_metadata["cache"])),
    )
    events = attach_peer_geometry(events, base=base, aligned=aligned)
    selected = events.loc[
        pd.to_numeric(events[QUESTION.profile.attribute_column], errors="coerce").notna()
    ].copy()
    tiered = assign_attribute_tertiles(
        selected,
        attribute_column=QUESTION.profile.attribute_column,
        assignment="actual",
        seed_key=f"{pair}|{QUESTION.id}",
    )
    tiered = tiered.loc[
        tiered["peer_cluster_survives_without_lvn"]
        & tiered["state_peer_contact_count"].ge(2)
        & tiered["state_peer_contact_connected_dependency_group_count"].ge(2)
    ].copy()
    matches = DataFrame(
        matched_attribute_event_pairs(
            tiered,
            question=QUESTION,
            assignment="actual",
            state_features=STATE_FEATURES,
            independence_hours=MATCH_SEPARATION_HOURS,
        )
    )
    if matches.empty:
        raise ValueError(f"No matchable fresh thin-LVN events for {pair}.")

    rows: list[dict[str, Any]] = []
    for side in ("high", "low"):
        for row in matches.to_dict("records"):
            event: dict[str, Any] = {
                "pair": pair,
                "date": row[f"{side}_event_time"],
                "base_index": int(row[f"{side}_base_index"]),
                "period": str(row["period"]),
                "event_state": "thin_lvn_mixed_cluster_contact",
                "level_identity": str(row["level_name"]),
                "approach_state": str(row["approach_state"]),
                "level__pre_distance_atr": row[f"{side}_pre_distance_atr"],
                "level__thinness": row[f"{side}_attribute_value"],
                "level__vp_value_area_width_pct": row[
                    f"{side}_state__state_vp_value_area_width_pct"
                ],
                "level__vp_persistence_bars": row[
                    f"{side}_state__state_vp_level_persistence_bars"
                ],
                "geometry__peer_level_count": row[
                    f"{side}_state__state_peer_level_count"
                ],
                "geometry__peer_contact_count": row[
                    f"{side}_state__state_peer_contact_count"
                ],
                "geometry__peer_dependency_group_count": row[
                    f"{side}_state__state_peer_dependency_group_count"
                ],
                "geometry__peer_contact_connected_dependency_group_count": row[
                    f"{side}_state__state_peer_contact_connected_dependency_group_count"
                ],
                "mtf__peer_higher_tf_count": row[
                    f"{side}_state__state_peer_higher_tf_count"
                ],
                "mtf__peer_daily_count": row[f"{side}_state__state_peer_daily_count"],
            }
            event.update(one_hot(row["level_name"], ("lvn_above", "lvn_below"), "level__id"))
            event.update(one_hot(row["approach_state"], APPROACH_STATES, "level__approach"))
            rows.append(event)
    output = DataFrame(rows)
    key = ["pair", "date", "base_index", "level_identity"]
    value_columns = [column for column in output if column not in key]
    conflicts = output.groupby(key, dropna=False)[value_columns].nunique(dropna=False).max(axis=1)
    if conflicts.gt(1).any():
        raise ValueError(f"Conflicting fresh thin-LVN event attributes for {pair}.")
    output = output.drop_duplicates(key, keep="first")

    raw = raw_events[
        [
            "base_index",
            "event_time",
            "period",
            "level_name",
            "source_available_at",
            "abs_excursion_atr_h1",
        ]
    ].rename(columns={"event_time": "date", "level_name": "level_identity"})
    raw["date"] = normalize_dates(raw["date"])
    output["date"] = normalize_dates(output["date"])
    output = output.merge(
        raw,
        on=["base_index", "date", "period", "level_identity"],
        how="left",
        validate="one_to_one",
    )
    output[TARGET_COLUMN] = pd.to_numeric(output["abs_excursion_atr_h1"], errors="coerce")
    output["source_available_at"] = normalize_dates(output["source_available_at"])
    output["interaction__thinness_x_contact_dependency_groups"] = (
        pd.to_numeric(output["level__thinness"], errors="coerce")
        * pd.to_numeric(
            output["geometry__peer_contact_connected_dependency_group_count"],
            errors="coerce",
        )
    )
    output["interaction__thinness_x_higher_tf_count"] = (
        pd.to_numeric(output["level__thinness"], errors="coerce")
        * pd.to_numeric(output["mtf__peer_higher_tf_count"], errors="coerce")
    )
    return output.drop(columns=["abs_excursion_atr_h1"])


def density_geometry(
    pair: str,
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> tuple[DataFrame, DataFrame]:
    base = prepare_base_market_frame(pair, manifest)
    surfaces = build_density_surfaces(base)
    references = selected_reference_levels(
        pair=pair,
        base=base,
        manifest_path=manifest_path,
    )
    density_levels, density_widths, density_surface_keys = density_matrices(surfaces)
    reference_levels, reference_widths, reference_groups = reference_matrices(
        references,
        base_atr=numeric_array(base["base_atr"]),
    )
    frames: list[DataFrame] = []
    for surface in surfaces:
        if surface.family != "confirmed_swing_price_density" or surface.history_hours != 168:
            continue
        events, _ = surface_event_geometry(pair=pair, base=base, surface=surface)
        events = eligible_period_events(
            events,
            manifest,
            embargo_hours=OUTCOME_EMBARGO_HOURS,
        )
        if events.empty:
            continue
        events = attach_structure_overlaps(
            events,
            source_surface_key=surface.key,
            density_levels=density_levels,
            density_widths=density_widths,
            density_surface_keys=density_surface_keys,
            reference_levels=reference_levels,
            reference_widths=reference_widths,
            reference_groups=reference_groups,
        )
        events["actual_scope"] = classify_actual_scope(events)
        frames.append(events)
    if not frames:
        raise ValueError(f"No fresh density geometry for {pair}.")
    return base, pd.concat(frames, ignore_index=True)


def density_candidate_events_by_scope(
    pair: str,
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> dict[str, DataFrame]:
    base, geometry = density_geometry(
        pair,
        manifest=manifest,
        manifest_path=manifest_path,
    )
    state = causal_local_state(base).merge(
        causal_market_context(manifest),
        on="date",
        how="left",
        validate="one_to_one",
    )
    paths = future_path_matrices(base, max_horizon=24)
    outputs: dict[str, DataFrame] = {}
    for surface_id, scope in DENSITY_SCOPES.items():
        selected = geometry.loc[geometry["actual_scope"].eq(scope)].copy()
        if selected.empty:
            outputs[surface_id] = DataFrame()
            continue
        events = attach_geometry_event_outcomes(
            pair=pair,
            base=base,
            state=state,
            paths=paths,
            geometry=selected,
        )
        rows, _ = direct_state_match_rows(events, pair=pair)
        matches = DataFrame(rows)
        if matches.empty:
            outputs[surface_id] = DataFrame()
            continue
        matches = purge_density_matches(matches)
        allowed_controls = {
            "event_state__near_miss",
            "event_state__already_inside",
            "event_state__repeat_contact",
        }
        matches = matches.loc[matches["control"].isin(allowed_controls)].copy()
        h1 = matches.loc[matches["response_window"].eq("h1")].copy()
        h4 = matches.loc[matches["response_window"].eq("h4")].copy()
        if h1.empty or h4.empty:
            outputs[surface_id] = DataFrame()
            continue
        one_hour_excursion: dict[tuple[str, int, pd.Timestamp, str], float] = {}
        for side in ("actual", "control"):
            for row in h1.to_dict("records"):
                key = (
                    side,
                    int(row[f"{side}_base_index"]),
                    pd.Timestamp(row[f"{side}_event_time"]),
                    str(row[f"{side}_event_state"]),
                )
                value = float(row[f"{side}__abs_excursion_atr_h1"])
                previous = one_hour_excursion.get(key)
                if previous is not None and not np.isclose(
                    previous,
                    value,
                    rtol=1e-12,
                    atol=1e-12,
                ):
                    raise ValueError(f"Conflicting fresh density target for {pair} and {key}.")
                one_hour_excursion[key] = value
        event_rows: list[dict[str, Any]] = []
        for side in ("actual", "control"):
            for row in h4.to_dict("records"):
                event_state = str(row[f"{side}_event_state"])
                key = (
                    side,
                    int(row[f"{side}_base_index"]),
                    pd.Timestamp(row[f"{side}_event_time"]),
                    event_state,
                )
                event: dict[str, Any] = {
                    "pair": pair,
                    "date": row[f"{side}_event_time"],
                    "base_index": int(row[f"{side}_base_index"]),
                    "period": str(row["period"]),
                    "event_state": event_state,
                    "level_identity": "confirmed_swing_price_density",
                    "approach_state": "not_available_as_an_event_specific_g3d_field",
                    "source_available_at": row[f"{side}_source_available_at"],
                    "level__pre_distance_atr": row[f"{side}_raw_pre_distance_atr"],
                    "level__zone_support_fraction": row[f"{side}_zone_support_fraction"],
                    "level__zone_half_width_atr": row[
                        f"{side}_state__state_g3d_zone_half_width_atr"
                    ],
                    "geometry__other_density_zone_count": row[
                        f"{side}_state__state_g3d_other_density_zone_count"
                    ],
                    "mtf__reference_level_count": row[
                        f"{side}_state__state_g3d_reference_level_count"
                    ],
                    "mtf__overlap_reference_level_count": row[
                        f"{side}_overlap_reference_level_count"
                    ],
                    TARGET_COLUMN: one_hour_excursion.get(key, np.nan),
                }
                event.update(one_hot(event_state, ARRIVAL_STATES, "level__arrival"))
                event.update(
                    one_hot(
                        scope,
                        ("single_density_zone", "density_cluster"),
                        "level__scope",
                    )
                )
                event_rows.append(event)
        output = DataFrame(event_rows)
        output["date"] = normalize_dates(output["date"])
        output["source_available_at"] = normalize_dates(output["source_available_at"])
        key_columns = ["pair", "date", "base_index", "event_state", "level_identity"]
        value_columns = [column for column in output if column not in key_columns]
        conflicts = (
            output.groupby(key_columns, dropna=False)[value_columns]
            .nunique(dropna=False)
            .max(axis=1)
        )
        if conflicts.gt(1).any():
            raise ValueError(f"Conflicting fresh density event attributes for {pair} and {scope}.")
        output = output.drop_duplicates(key_columns, keep="first")
        fresh = pd.to_numeric(
            output["level__arrival_first_arrival_after_outside_interval"],
            errors="coerce",
        )
        support = pd.to_numeric(output["level__zone_support_fraction"], errors="coerce")
        reference_count = pd.to_numeric(
            output["mtf__reference_level_count"], errors="coerce"
        )
        output["interaction__fresh_arrival_x_support_fraction"] = fresh * support
        output["interaction__support_fraction_x_reference_count"] = (
            support * reference_count
        )
        outputs[surface_id] = output
    return outputs


def eligible_model_events(
    events: DataFrame,
    *,
    wide: DataFrame,
    manifest: dict[str, Any],
) -> DataFrame:
    if events.empty:
        return events
    output = events.copy()
    output["date"] = normalize_dates(output["date"])
    output["source_available_at"] = normalize_dates(output["source_available_at"])
    if output["source_available_at"].isna().any():
        raise ValueError("Fresh G5C events contain missing source-availability timestamps.")
    if output["source_available_at"].gt(output["date"]).any():
        raise ValueError("Fresh G5C events admit future source information.")
    period_map = {item["id"]: item for item in manifest["data"]["chronological_periods"]}
    inside = pd.Series(False, index=output.index)
    for period, contract in period_map.items():
        selected = output["period"].eq(period)
        start = pd.Timestamp(contract["start_utc"])
        end = pd.Timestamp(contract["end_utc_exclusive"]) - pd.Timedelta(hours=1)
        inside.loc[selected] = output.loc[selected, "date"].ge(start) & output.loc[
            selected, "date"
        ].lt(end)
    output = output.loc[inside].copy()
    output = output.sort_values(
        ["date", "event_state", "level_identity"],
        kind="mergesort",
    ).drop_duplicates("date", keep="first")
    output = independently_spaced_events(output, 1)
    output = output.merge(wide, on="date", how="left", validate="one_to_one")
    features = [
        column
        for column in output
        if column.startswith(("wide__", "level__", "geometry__", "mtf__", "interaction__"))
    ]
    numeric = output[[*features, TARGET_COLUMN]].apply(pd.to_numeric, errors="coerce")
    output = output.loc[numeric.notna().all(axis=1)].copy()
    return output.reset_index(drop=True)


def build_pair_event_surfaces(
    pair: str,
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> dict[str, DataFrame]:
    wide = source_wide_context(manifest)
    thin = eligible_model_events(
        thin_candidate_events(pair, manifest=manifest, manifest_path=manifest_path),
        wide=wide,
        manifest=manifest,
    )
    density = density_candidate_events_by_scope(
        pair,
        manifest=manifest,
        manifest_path=manifest_path,
    )
    return {
        "normal_thin_lvn_mixed_cluster": thin,
        **{
            surface: eligible_model_events(
                frame,
                wide=wide,
                manifest=manifest,
            )
            for surface, frame in density.items()
        },
    }


def safe_pair_event_surfaces(
    pair: str,
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> tuple[str, dict[str, DataFrame] | None, str | None]:
    try:
        return (
            pair,
            build_pair_event_surfaces(
                pair,
                manifest=manifest,
                manifest_path=manifest_path,
            ),
            None,
        )
    except Exception as exc:
        return pair, None, f"{type(exc).__name__}: {exc}"


def coverage_table(frames: dict[tuple[str, str], DataFrame], pairs: Sequence[str]) -> DataFrame:
    rows: list[dict[str, Any]] = []
    primary = tuple(pair for pair in pairs if not pair.startswith("BTC/"))
    for surface in SURFACES:
        for period in (
            "g5c_exposed_training_bridge",
            *CONFIRMATION_PERIODS,
        ):
            counts = {
                pair: int(
                    frames[(surface, pair)]["period"].eq(period).sum()
                    if not frames[(surface, pair)].empty
                    else 0
                )
                for pair in pairs
            }
            primary_rows = sum(counts[pair] for pair in primary)
            primary_coins = sum(counts[pair] > 0 for pair in primary)
            confirmation = period in CONFIRMATION_PERIODS
            supported = bool(
                not confirmation
                or (
                    primary_rows >= MIN_SCORABLE_ROWS
                    and primary_coins >= MIN_CONFIRMATION_COINS
                )
            )
            rows.append(
                {
                    "surface_id": surface,
                    "period": period,
                    "primary_rows": primary_rows,
                    "primary_coins": primary_coins,
                    "btc_rows": counts.get("BTC/USDT:USDT", 0),
                    "pair_rows": json.dumps(counts, sort_keys=True),
                    "technical_minimum_rows": MIN_SCORABLE_ROWS,
                    "frozen_minimum_coins": MIN_CONFIRMATION_COINS,
                    "status": "supported" if supported else "insufficient_common_support",
                    "outcome_values_read_for_coverage_decision": False,
                }
            )
    return DataFrame(rows)


def build_event_preflight(run_id: str, *, workers: int) -> dict[str, Any]:
    manifest, manifest_path, record_dir, artifact_dir = write_source_manifest(run_id)
    if workers < 1 or workers > MAX_WORKERS:
        raise ValueError(f"G5C event workers must be in 1..{MAX_WORKERS}.")
    source_record = record_dir / "g5c_source_build_record.json"
    if not source_record.is_file():
        raise FileNotFoundError(source_record)
    source = json.loads(source_record.read_text(encoding="utf-8"))
    if source.get("status") != "completed" or source.get("manifest_sha256") != sha256_file(
        manifest_path
    ):
        raise ValueError("G5C source build is incomplete or uses another manifest.")
    pairs = tuple(str(pair) for pair in manifest["data"]["pairs"])
    results: list[tuple[str, dict[str, DataFrame] | None, str | None]] = []
    if workers == 1:
        results = [
            safe_pair_event_surfaces(
                pair,
                manifest=manifest,
                manifest_path=manifest_path,
            )
            for pair in pairs
        ]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    safe_pair_event_surfaces,
                    pair,
                    manifest=manifest,
                    manifest_path=manifest_path,
                ): pair
                for pair in pairs
            }
            results = [future.result() for future in as_completed(futures)]
    failures = [
        {"pair": pair, "error": error}
        for pair, frames, error in results
        if frames is None or error is not None
    ]
    if failures:
        failure_record = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "run_id": run_id,
            "status": "event_preflight_failed",
            "failures": failures,
            "outcomes_interpreted": False,
        }
        atomic_write_json(failure_record, record_dir / "g5c_event_preflight_record.json")
        raise RuntimeError(f"{len(failures)} G5C pair event build(s) failed.")
    frames: dict[tuple[str, str], DataFrame] = {}
    inventory_rows: list[dict[str, Any]] = []
    event_root = artifact_dir / "fresh_event_surfaces"
    for pair, pair_frames, _ in sorted(results, key=lambda item: item[0]):
        if pair_frames is None:
            raise AssertionError(f"Missing successful G5C pair frames for {pair}.")
        for surface, frame in pair_frames.items():
            frames[(surface, pair)] = frame
            path = event_root / surface / f"{pair_stem(pair)}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_parquet(frame, path)
            inventory_rows.append(
                {
                    "pair": pair,
                    "surface_id": surface,
                    "rows": len(frame),
                    "period_rows": json.dumps(frame.groupby("period").size().to_dict()),
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "feature_columns": json.dumps(
                        [
                            column
                            for column in frame
                            if column.startswith(
                                ("wide__", "level__", "geometry__", "mtf__", "interaction__")
                            )
                        ]
                    ),
                }
            )
    coverage = coverage_table(frames, pairs)
    confirmation = coverage.loc[coverage["period"].isin(CONFIRMATION_PERIODS)]
    surface_status = {
        surface: bool(
            len(selected := confirmation.loc[confirmation["surface_id"].eq(surface)])
            == len(CONFIRMATION_PERIODS)
            and selected["status"].eq("supported").all()
        )
        for surface in SURFACES
    }
    status = (
        "coverage_gate_passed_all_surfaces"
        if all(surface_status.values())
        else "coverage_gate_partially_passed"
        if any(surface_status.values())
        else "parked_insufficient_new_period_event_coverage"
    )
    inventory = DataFrame(inventory_rows)
    atomic_write_parquet(inventory, record_dir / "g5c_fresh_event_inventory.parquet")
    atomic_write_parquet(coverage, record_dir / "g5c_fresh_event_coverage.parquet")
    record = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": status,
        "completed_at_utc": utc_now(),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_build_record": str(source_record),
        "source_build_record_sha256": sha256_file(source_record),
        "pairs": list(pairs),
        "surfaces": list(SURFACES),
        "surface_coverage_gate": surface_status,
        "coverage_path": str(record_dir / "g5c_fresh_event_coverage.parquet"),
        "inventory_path": str(record_dir / "g5c_fresh_event_inventory.parquet"),
        "fresh_event_root": str(event_root),
        "coverage_decision_fields": ["event timestamp", "pair", "period", "surface"],
        "target_values_compared_or_interpreted": False,
        "profit_optimization": False,
        "direction_prediction": False,
    }
    atomic_write_json(record, record_dir / "g5c_event_preflight_record.json")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 5C coverage-first fresh-period source and event preflight. "
            "It freezes usable rows before any target comparison."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", choices=("source", "events", "all"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.workers > MAX_WORKERS:
        raise ValueError(f"workers must be in 1..{MAX_WORKERS}")
    if args.phase in {"source", "all"}:
        result = build_fresh_source(args.run_id, workers=args.workers)
        print(json.dumps(result, indent=2), flush=True)
    if args.phase in {"events", "all"}:
        result = build_event_preflight(args.run_id, workers=args.workers)
        print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

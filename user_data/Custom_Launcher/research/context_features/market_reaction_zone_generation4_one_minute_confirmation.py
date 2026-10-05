from __future__ import annotations

# G4A is a fixed confirmation of the G3G reaction-location and recent-path lead.
# It deliberately does not reopen the 375-input technical-indicator screen.
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
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_one_minute_replay_analysis as g3g,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    atomic_write_csv,
    atomic_write_json,
    load_manifest,
    prepare_base_market_frame,
    sha256_file,
    utc_now,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_vp_roles import (  # noqa: E501
    stable_json_sha256,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_lvn_attribution import (  # noqa: E501
    aligned_selected_levels_from_verified_prefix,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_one_minute_replay import (  # noqa: E501
    COHORT_SOURCES,
    FUTURES_DATA_DIR,
    WINDOW_HOURS,
    artifact_record,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation4_one_minute_breadth import (  # noqa: E501
    ARTIFACT_ROOT,
    REPORT_ROOT,
    validate_g4_branch,
)


SCHEMA_VERSION = 1
DEFAULT_FREEZE_RUN_ID = "g4a_thin_lvn_1m_breadth_freeze_20260814a"
CONTROL_GLOBAL_SEPARATION_MINUTES = 240
REACTION_HORIZON_MINUTES = 60
CONTROL_HORIZON_MINUTES = 120
VALIDATION_JOINT_MINIMUM = 0.55
MAIN_TARGET = 0.65
MINIMUM_SUCCESS_PAIRS = 3
MAXIMUM_SUCCESS_PAIR_SHARE = 0.35
MINIMUM_BALANCED_VALIDATION_PAIRS = 10


@dataclass(frozen=True)
class MinimalPairSurfaces:
    minute: DataFrame
    match: DataFrame
    base: DataFrame
    aligned_levels: list[Any]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze controls, audit selective context expansion, or analyze the fixed "
            "G4A one-minute thin-LVN confirmation batch."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--freeze-run-id", default=DEFAULT_FREEZE_RUN_ID)
    parser.add_argument("--overwrite", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-method", action="store_true")
    mode.add_argument("--audit-expansion", action="store_true")
    mode.add_argument("--analyze", action="store_true")
    args = parser.parse_args(argv)
    validate_g4_branch()
    if args.freeze_method:
        return freeze_method(
            run_id=args.run_id,
            freeze_run_id=args.freeze_run_id,
            overwrite=bool(args.overwrite),
        )
    if args.audit_expansion:
        return audit_expansion(run_id=args.run_id, freeze_run_id=args.freeze_run_id)
    return analyze(
        run_id=args.run_id,
        freeze_run_id=args.freeze_run_id,
        overwrite=bool(args.overwrite),
    )


def freeze_method(*, run_id: str, freeze_run_id: str, overwrite: bool) -> int:
    run_dir = REPORT_ROOT / run_id
    record_path = run_dir / "g4a_method_freeze_record.json"
    if record_path.is_file() and not overwrite:
        raise FileExistsError(f"Method run already exists: {record_path}")
    inputs = validated_freeze_inputs(freeze_run_id)
    request = method_request_contract(run_id, freeze_run_id, inputs)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "freezing_before_reaction_outcomes",
        "started_at_utc": utc_now(),
        "request_sha256": stable_json_sha256(request),
        "request_contract": request,
    }
    atomic_write_json(record, record_path)
    try:
        sample = load_sample(inputs["sample_path"])
        surfaces = load_all_surfaces(sample)
        contacts = actual_contact_table(sample, surfaces)
        boundaries = initial_boundary_audit(sample, contacts, surfaces)
        assignments, match_audit = freeze_control_assignments(
            sample=sample,
            contacts=contacts,
            surfaces=surfaces,
        )
        expansion = selective_expansion_intervals(sample, boundaries)

        assignment_path = run_dir / "g4a_frozen_control_assignments.csv"
        match_path = run_dir / "g4a_frozen_control_match_audit.csv"
        boundary_path = run_dir / "g4a_initial_boundary_audit.csv"
        expansion_path = run_dir / "g4a_selective_expansion_intervals.csv"
        atomic_write_csv(assignments, assignment_path)
        atomic_write_csv(match_audit, match_path)
        atomic_write_csv(boundaries, boundary_path)
        atomic_write_csv(expansion, expansion_path)

        assigned = assignments["control_assigned"].eq(True)
        usable = assigned & assignments["control_balance_usable"].eq(True)
        record.update(
            {
                "status": "frozen_before_reaction_outcomes",
                "completed_at_utc": utc_now(),
                "actual_episode_count": len(sample),
                "assigned_control_count": int(assigned.sum()),
                "balanced_control_count": int(usable.sum()),
                "balanced_control_counts_by_cohort": count_true_by(
                    assignments, "cohort", "control_balance_usable"
                ),
                "initial_boundary_flagged_count": int(
                    boundaries["boundary_extension_indicated"].sum()
                ),
                "selective_expansion_interval_count": len(expansion),
                "artifacts": {
                    "control_assignments_csv": artifact_record(assignment_path),
                    "control_match_audit_csv": artifact_record(match_path),
                    "initial_boundary_audit_csv": artifact_record(boundary_path),
                    "selective_expansion_intervals_csv": artifact_record(expansion_path),
                },
            }
        )
        atomic_write_json(record, record_path)
        print(
            json.dumps(
                {
                    "status": record["status"],
                    "episodes": len(sample),
                    "assigned_controls": int(assigned.sum()),
                    "balanced_controls": int(usable.sum()),
                    "boundary_flagged": record["initial_boundary_flagged_count"],
                    "expansion_intervals": len(expansion),
                },
                indent=2,
            )
        )
    except Exception as exc:
        record.update(
            {
                "status": "failed_before_reaction_outcomes",
                "failed_at_utc": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_write_json(record, record_path)
        raise
    return 0


def audit_expansion(*, run_id: str, freeze_run_id: str) -> int:
    run_dir = REPORT_ROOT / run_id
    record_path = run_dir / "g4a_method_freeze_record.json"
    record = validated_method_record(record_path)
    inputs = validated_freeze_inputs(freeze_run_id)
    sample = load_sample(inputs["sample_path"])
    boundary_path = validated_method_artifact(record, "initial_boundary_audit_csv")
    boundaries = pd.read_csv(boundary_path)
    flagged_ids = set(
        boundaries.loc[
            boundaries["boundary_extension_indicated"].astype(str).str.lower().eq("true"),
            "episode_id",
        ].astype(str)
    )
    expanded = sample.loc[sample["episode_id"].astype(str).isin(flagged_ids)].copy()
    coverage = audit_episode_windows(expanded, expanded=True)
    coverage_path = run_dir / "g4a_selective_expansion_coverage_audit.csv"
    coverage_record_path = run_dir / "g4a_selective_expansion_coverage_record.json"
    atomic_write_csv(coverage, coverage_path)
    passed = bool(coverage["passed"].all()) if len(coverage) else True
    coverage_record = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "status": "passed" if passed else "failed",
        "flagged_episode_count": len(expanded),
        "passed_episode_count": int(coverage["passed"].sum()) if len(coverage) else 0,
        "expected_rows": int(coverage["expected_rows"].sum()) if len(coverage) else 0,
        "actual_rows": int(coverage["actual_rows"].sum()) if len(coverage) else 0,
        "coverage_csv": artifact_record(coverage_path),
    }
    atomic_write_json(coverage_record, coverage_record_path)
    record["selective_expansion_coverage"] = {
        "status": coverage_record["status"],
        "coverage_csv": artifact_record(coverage_path),
        "coverage_record": artifact_record(coverage_record_path),
    }
    atomic_write_json(record, record_path)
    if not passed:
        raise RuntimeError(f"Selective expansion coverage failed: {coverage_path}")
    return 0


def analyze(*, run_id: str, freeze_run_id: str, overwrite: bool) -> int:
    run_dir = REPORT_ROOT / run_id
    artifact_dir = ARTIFACT_ROOT / run_id
    method_path = run_dir / "g4a_method_freeze_record.json"
    method = validated_method_record(method_path)
    coverage = method.get("selective_expansion_coverage", {})
    if coverage.get("status") != "passed":
        raise ValueError("Selective expanded context has not passed its coverage audit")
    analysis_path = run_dir / "g4a_confirmation_record.json"
    if analysis_path.is_file() and not overwrite:
        raise FileExistsError(f"Confirmation already exists: {analysis_path}")
    inputs = validated_freeze_inputs(freeze_run_id)
    request = {
        "schema_version": SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "method_record_sha256": sha256_file(method_path),
        "method_request_sha256": method["request_sha256"],
        "freeze_record_sha256": sha256_file(inputs["freeze_record_path"]),
        "coverage_record_sha256": sha256_file(inputs["coverage_record_path"]),
        "outcomes": method["request_contract"]["fixed_outcomes"],
        "classification": method["request_contract"]["classification"],
        "technical_relationship_screen": False,
        "model_training": False,
        "profit_optimization": False,
    }
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "running_fixed_confirmation",
        "started_at_utc": utc_now(),
        "request_sha256": stable_json_sha256(request),
        "request_contract": request,
    }
    atomic_write_json(record, analysis_path)
    try:
        sample = load_sample(inputs["sample_path"])
        components = load_components(inputs["components_path"])
        assignments_path = validated_method_artifact(method, "control_assignments_csv")
        boundary_path = validated_method_artifact(method, "initial_boundary_audit_csv")
        assignments = pd.read_csv(assignments_path)
        boundaries = pd.read_csv(boundary_path)
        for column in ("actual_contact_time", "control_time"):
            assignments[column] = pd.to_datetime(assignments[column], utc=True, errors="coerce")
        flagged_ids = set(
            boundaries.loc[
                boundaries["boundary_extension_indicated"].astype(str).str.lower().eq("true"),
                "episode_id",
            ].astype(str)
        )
        surfaces = load_all_surfaces(sample)
        actual_rows: list[dict[str, Any]] = []
        control_rows: list[dict[str, Any]] = []
        path_rows: list[dict[str, Any]] = []
        final_boundaries: list[dict[str, Any]] = []
        for _, episode in sample.sort_values("sample_selection_order").iterrows():
            episode_id = str(episode["episode_id"])
            assignment = assignments.loc[assignments["episode_id"].astype(str).eq(episode_id)]
            if len(assignment) != 1:
                raise ValueError(f"Expected one frozen control assignment: {episode_id}")
            pair_surfaces = surfaces[(str(episode["cohort"]), str(episode["pair"]))]
            component_slice = components.loc[
                components["episode_id"].astype(str).eq(episode_id)
            ].copy()
            result = analyze_fixed_episode(
                episode=episode,
                assignment=assignment.iloc[0],
                components=component_slice,
                surfaces=pair_surfaces,
            )
            actual_rows.append(result["actual"])
            if result["control"] is not None:
                control_rows.append(result["control"])
            path_rows.extend(result["paths"])
            windows = expanded_windows() if episode_id in flagged_ids else WINDOW_HOURS
            final_boundaries.append(
                g3g.boundary_audit(
                    episode=episode,
                    contact_time=pd.Timestamp(assignment.iloc[0]["actual_contact_time"]),
                    minute=pair_surfaces.minute,
                    window_hours=windows,
                )
            )

        actual = DataFrame(actual_rows).sort_values("sample_selection_order")
        controls = DataFrame(control_rows).sort_values("sample_selection_order")
        events = pd.concat([actual, controls], ignore_index=True, sort=False)
        paths = DataFrame(path_rows).sort_values(
            ["cohort", "episode_id", "event_kind", "checkpoint_minutes"]
        )
        final_boundary = DataFrame(final_boundaries).sort_values("sample_selection_order")
        comparisons = g3g.control_comparisons(events)
        phase_summary = summarize_scopes(actual, controls)
        summary = classify_confirmation(phase_summary, final_boundary)

        event_path = artifact_dir / "g4a_event_replay.csv"
        paths_path = artifact_dir / "g4a_checkpoint_paths.csv"
        comparison_path = run_dir / "g4a_control_comparisons.csv"
        phase_path = run_dir / "g4a_scope_phase_summary.csv"
        boundary_final_path = run_dir / "g4a_final_boundary_audit.csv"
        summary_path = run_dir / "g4a_confirmation_summary.json"
        atomic_write_csv(events, event_path)
        atomic_write_csv(paths, paths_path)
        atomic_write_csv(comparisons, comparison_path)
        atomic_write_csv(phase_summary, phase_path)
        atomic_write_csv(final_boundary, boundary_final_path)
        atomic_write_json(summary, summary_path)
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "actual_episode_count": len(actual),
                "control_episode_count": len(controls),
                "classification": summary["classification"],
                "summary": summary,
                "artifacts": {
                    "event_replay_csv": artifact_record(event_path),
                    "checkpoint_paths_csv": artifact_record(paths_path),
                    "control_comparisons_csv": artifact_record(comparison_path),
                    "scope_phase_summary_csv": artifact_record(phase_path),
                    "final_boundary_audit_csv": artifact_record(boundary_final_path),
                    "summary_json": artifact_record(summary_path),
                },
            }
        )
        atomic_write_json(record, analysis_path)
        print(json.dumps(summary, indent=2, sort_keys=True))
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "failed_at_utc": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_write_json(record, analysis_path)
        raise
    return 0


def validated_freeze_inputs(freeze_run_id: str) -> dict[str, Any]:
    freeze_dir = REPORT_ROOT / freeze_run_id
    freeze_record_path = freeze_dir / "g4a_freeze_record.json"
    coverage_record_path = freeze_dir / "g4a_one_minute_coverage_record.json"
    freeze = json.loads(freeze_record_path.read_text(encoding="utf-8"))
    coverage = json.loads(coverage_record_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen_before_one_minute_paths":
        raise ValueError("G4A parent selection is not frozen")
    if coverage.get("status") != "passed":
        raise ValueError("G4A one-minute coverage has not passed")
    sample_path = Path(str(freeze["artifacts"]["frozen_sample_csv"]["path"]))
    components_path = Path(str(freeze["artifacts"]["frozen_sample_components_csv"]["path"]))
    for key, path in (
        ("frozen_sample_csv", sample_path),
        ("frozen_sample_components_csv", components_path),
    ):
        if sha256_file(path) != str(freeze["artifacts"][key]["sha256"]):
            raise ValueError(f"Frozen G4A input changed: {path}")
    return {
        "freeze": freeze,
        "coverage": coverage,
        "freeze_record_path": freeze_record_path,
        "coverage_record_path": coverage_record_path,
        "sample_path": sample_path,
        "components_path": components_path,
    }


def method_request_contract(
    run_id: str, freeze_run_id: str, inputs: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "freeze_run_id": freeze_run_id,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "freeze_record_sha256": sha256_file(inputs["freeze_record_path"]),
        "coverage_record_sha256": sha256_file(inputs["coverage_record_path"]),
        "sample_sha256": sha256_file(inputs["sample_path"]),
        "components_sha256": sha256_file(inputs["components_path"]),
        "hypothesis": (
            "Qualified thin-LVN mixed-cluster contacts react more often than earlier "
            "same-state no-level minutes, while the causal preceding sixty-minute path "
            "predicts immediate direction better than retrospective majority direction."
        ),
        "strongest_alternative": (
            "G3G was lucky, already-active market state explains the response, or ordinary "
            "recent trend works equally well without a calculated level."
        ),
        "fixed_inputs": [
            "return over the preceding 15, 60, and 240 completed minutes",
            "realized return volatility over the preceding 60 completed minutes",
            "one-minute ATR(14) as a percentage of price",
            "last completed minute volume divided by its trailing 20-minute mean",
            "volume-weighted candle-body pressure over the preceding 20 minutes",
            "price position inside the preceding 240-minute range",
            "exact frozen LVN centre, half-width, side, and approach",
            "complete independent cluster components and source timeframe",
        ],
        "fixed_outcomes": [
            "declared 60-minute joint price-and-volume reaction",
            "continuous absolute, through, and away excursion",
            "through-first versus away-first after the contact candle closes",
            "far-edge breakout versus extra-half-width rejection",
            "post/pre volume, candle-pressure, and realized-volatility change",
            "zone overlap fraction and centre crossings",
        ],
        "reaction_definition": {
            "horizon_minutes": REACTION_HORIZON_MINUTES,
            "minimum_excursion_parent_half_widths": 1.0,
            "minimum_post_pre_volume_ratio": 1.5,
            "movement_starts": "first minute after the completed contact candle",
            "native_tool_score": False,
        },
        "control_assignment": {
            "same_pair": True,
            "strictly_before_actual": True,
            "same_pre_60m_return_sign": True,
            "no_causally_available_selected_level_contact": True,
            "match_features": list(g3g.MATCH_FEATURES),
            "mean_robust_distance_limit": g3g.CONTROL_MEAN_DISTANCE_LIMIT,
            "maximum_robust_distance_limit": g3g.CONTROL_MAX_DISTANCE_LIMIT,
            "actual_and_control_global_separation_minutes": (CONTROL_GLOBAL_SEPARATION_MINUTES),
            "control_path_horizon_minutes": CONTROL_HORIZON_MINUTES,
            "selection_order": ("fewest eligible candidates first, then frozen G4A selection hash"),
            "future_outcomes_used": False,
        },
        "classification": {
            "primary_period": "validation_early plus validation_late",
            "primary_scopes": ["normal_excluding_btc", "meme"],
            "btc": "descriptive separate scope",
            "retain_requires": [
                "paired actual reaction rate greater than paired no-level control",
                "simple recent-trend direction accuracy greater than retrospective majority",
                (
                    "joint reaction-and-correct-direction rate at least "
                    f"{VALIDATION_JOINT_MINIMUM:.0%}"
                ),
                f"joint successes from at least {MINIMUM_SUCCESS_PAIRS} pairs",
                f"no pair supplies more than {MAXIMUM_SUCCESS_PAIR_SHARE:.0%} of joint successes",
                (f"at least {MINIMUM_BALANCED_VALIDATION_PAIRS} balanced validation controls"),
                "early and late validation do not reverse the reaction uplift",
            ],
            "main_target": MAIN_TARGET,
            "mixed_result": (
                "combined validation passes but an early/late phase contradicts, or only "
                "one rational coin cohort repeats"
            ),
            "park": "combined validation fails the fixed lead criteria",
        },
        "selective_boundary_expansion": {
            "initial_windows": {
                key: {"pre_hours": value[0], "post_hours": value[1]}
                for key, value in WINDOW_HOURS.items()
            },
            "flagged_episode_action": "double only that episode's pre/post context once",
            "selection_or_primary_outcome_changed_by_expansion": False,
        },
        "technical_indicator_screen": False,
        "freqai_model": False,
        "profit_or_strategy_optimization": False,
        "worker_count": 1,
    }


def load_sample(path: Path) -> DataFrame:
    sample = pd.read_csv(path)
    for column in ("event_time", "source_available_at", "source_open"):
        if column in sample:
            sample[column] = pd.to_datetime(sample[column], utc=True)
    return sample


def load_components(path: Path) -> DataFrame:
    components = pd.read_csv(path)
    for column in (
        "event_time",
        "component_source_available_at",
        "component_source_open",
    ):
        if column in components:
            components[column] = pd.to_datetime(components[column], utc=True)
    return components


def load_all_surfaces(
    sample: DataFrame,
) -> dict[tuple[str, str], MinimalPairSurfaces]:
    output: dict[tuple[str, str], MinimalPairSurfaces] = {}
    for cohort, pair in (
        sample[["cohort", "pair"]]
        .drop_duplicates()
        .sort_values(["cohort", "pair"])
        .itertuples(index=False, name=None)
    ):
        output[(str(cohort), str(pair))] = load_minimal_surfaces(str(pair), cohort=str(cohort))
    return output


def load_minimal_surfaces(pair: str, *, cohort: str) -> MinimalPairSurfaces:
    minute_path = FUTURES_DATA_DIR / f"{g3g.pair_stem(pair)}-1m-futures.feather"
    minute = g3g.load_ohlcv(minute_path)
    source = next(item for item in COHORT_SOURCES if item.cohort == cohort)
    manifest = load_manifest(source.manifest_path)
    base = prepare_base_market_frame(pair, manifest)
    aligned_levels = aligned_selected_levels_from_verified_prefix(
        pair=pair,
        base=base,
        manifest_path=source.manifest_path,
        timeframes=tuple(manifest["data"]["source_timeframes"]),
    )
    match = g3g.minute_match_frame(minute)
    match["causal_level_contact_count"] = vectorized_level_contact_count(
        match, base=base, aligned_levels=aligned_levels
    )
    match["complete_control_path"] = complete_path_mask(
        match["date"],
        before_minutes=CONTROL_HORIZON_MINUTES,
        after_minutes=CONTROL_HORIZON_MINUTES,
    )
    return MinimalPairSurfaces(
        minute=minute,
        match=match,
        base=base,
        aligned_levels=aligned_levels,
    )


def vectorized_level_contact_count(
    candles: DataFrame, *, base: DataFrame, aligned_levels: list[Any]
) -> np.ndarray:
    dates = pd.DatetimeIndex(pd.to_datetime(candles["date"], utc=True))
    base_dates = pd.DatetimeIndex(pd.to_datetime(base["date"], utc=True))
    positions = base_dates.searchsorted(dates.floor("h"), side="right") - 1
    usable = (positions >= 0) & (positions < len(base))
    safe_positions = np.clip(positions, 0, max(len(base) - 1, 0))
    base_atr = pd.to_numeric(base["base_atr"], errors="coerce").to_numpy(dtype=float)
    atr = np.full(len(candles), np.nan, dtype=float)
    if len(base):
        atr[usable] = base_atr[safe_positions[usable]]
    high = pd.to_numeric(candles["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(candles["low"], errors="coerce").to_numpy(dtype=float)
    counts = np.zeros(len(candles), dtype=np.int32)
    for item in aligned_levels:
        valid_source = np.asarray(item.valid, dtype=bool)
        level_source = np.asarray(pd.to_numeric(item.level, errors="coerce"), dtype=float)
        available_source = (
            pd.DatetimeIndex(pd.to_datetime(item.source_available, utc=True))
            .tz_convert(None)
            .to_numpy(dtype="datetime64[ns]")
        )
        valid = usable.copy()
        valid[usable] &= valid_source[safe_positions[usable]]
        levels = np.full(len(candles), np.nan, dtype=float)
        available = np.full(len(candles), np.datetime64("NaT"), dtype="datetime64[ns]")
        levels[usable] = level_source[safe_positions[usable]]
        available[usable] = available_source[safe_positions[usable]]
        timestamp_values = dates.tz_convert(None).to_numpy(dtype="datetime64[ns]")
        valid &= available <= timestamp_values
        valid &= np.isfinite(levels) & (levels > 0.0) & np.isfinite(atr) & (atr > 0.0)
        half_width = np.maximum(0.5 * atr, np.abs(levels) * 0.0005)
        counts += (valid & (high >= levels - half_width) & (low <= levels + half_width)).astype(
            np.int32
        )
    return counts


def complete_path_mask(dates: Series, *, before_minutes: int, after_minutes: int) -> Series:
    timestamps = pd.to_datetime(dates, utc=True)
    before = timestamps.shift(before_minutes).eq(timestamps - pd.Timedelta(minutes=before_minutes))
    after = timestamps.shift(-after_minutes).eq(timestamps + pd.Timedelta(minutes=after_minutes))
    return before & after


def actual_contact_table(
    sample: DataFrame,
    surfaces: dict[tuple[str, str], MinimalPairSurfaces],
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for _, episode in sample.sort_values("sample_selection_order").iterrows():
        pair_surfaces = surfaces[(str(episode["cohort"]), str(episode["pair"]))]
        contact = g3g.first_parent_zone_contact(episode, pair_surfaces.minute)
        snapshot = g3g.match_snapshot(pair_surfaces.match, contact)
        rows.append(
            {
                "episode_id": str(episode["episode_id"]),
                "cohort": str(episode["cohort"]),
                "pair": str(episode["pair"]),
                "actual_contact_time": contact,
                **snapshot,
            }
        )
    return DataFrame(rows)


def initial_boundary_audit(
    sample: DataFrame,
    contacts: DataFrame,
    surfaces: dict[tuple[str, str], MinimalPairSurfaces],
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    contacts_by_id = contacts.set_index("episode_id")
    for _, episode in sample.sort_values("sample_selection_order").iterrows():
        episode_id = str(episode["episode_id"])
        pair_surfaces = surfaces[(str(episode["cohort"]), str(episode["pair"]))]
        rows.append(
            g3g.boundary_audit(
                episode=episode,
                contact_time=pd.Timestamp(contacts_by_id.loc[episode_id, "actual_contact_time"]),
                minute=pair_surfaces.minute,
                window_hours=WINDOW_HOURS,
            )
        )
    return DataFrame(rows).sort_values("sample_selection_order")


def freeze_control_assignments(
    *,
    sample: DataFrame,
    contacts: DataFrame,
    surfaces: dict[tuple[str, str], MinimalPairSurfaces],
) -> tuple[DataFrame, DataFrame]:
    contacts_by_id = contacts.set_index("episode_id")
    actual_times = {
        cohort: list(group["actual_contact_time"].map(pd.Timestamp))
        for cohort, group in contacts.groupby("cohort", observed=True)
    }
    pools: dict[str, DataFrame] = {}
    episode_lookup: dict[str, Series] = {}
    for _, episode in sample.iterrows():
        episode_id = str(episode["episode_id"])
        episode_lookup[episode_id] = episode
        contact_row = contacts_by_id.loc[episode_id]
        pair_surfaces = surfaces[(str(episode["cohort"]), str(episode["pair"]))]
        pools[episode_id] = ranked_control_candidates(
            episode=episode,
            actual_contact_time=pd.Timestamp(contact_row["actual_contact_time"]),
            actual_match={feature: float(contact_row[feature]) for feature in g3g.MATCH_FEATURES},
            surfaces=pair_surfaces,
            cohort_actual_times=actual_times[str(episode["cohort"])],
        )

    selected_control_times: dict[str, list[pd.Timestamp]] = {
        cohort: [] for cohort in sample["cohort"].astype(str).unique()
    }
    assignment_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    order = sorted(
        episode_lookup,
        key=lambda episode_id: (
            len(pools[episode_id]),
            str(episode_lookup[episode_id]["g4a_selection_hash"]),
        ),
    )
    for episode_id in order:
        episode = episode_lookup[episode_id]
        cohort = str(episode["cohort"])
        pool = pools[episode_id]
        occupied_control_times = selected_control_times[cohort]
        eligible = pool.loc[
            pool["date"].map(
                lambda timestamp, occupied=tuple(occupied_control_times): separated_from_times(
                    pd.Timestamp(timestamp), list(occupied)
                )
            )
        ].copy()
        contact_row = contacts_by_id.loc[episode_id]
        if eligible.empty:
            assignment_rows.append(
                {
                    "episode_id": episode_id,
                    "sample_selection_order": int(episode["sample_selection_order"]),
                    "cohort": cohort,
                    "period": str(episode["period"]),
                    "pair": str(episode["pair"]),
                    "actual_contact_time": contact_row["actual_contact_time"],
                    "control_assigned": False,
                    "control_time": pd.NaT,
                    "control_balance_usable": False,
                    "control_state_distance_mean": math.nan,
                    "control_state_distance_max": math.nan,
                    "eligible_candidate_count_before_global_separation": len(pool),
                    "eligible_candidate_count_after_global_separation": 0,
                    "assignment_failure": "no_globally_independent_control_candidate",
                }
            )
            continue
        selected = eligible.iloc[0]
        control_time = pd.Timestamp(selected["date"])
        selected_control_times[cohort].append(control_time)
        usable = bool(
            float(selected["distance_mean"]) <= g3g.CONTROL_MEAN_DISTANCE_LIMIT
            and float(selected["distance_max"]) <= g3g.CONTROL_MAX_DISTANCE_LIMIT
        )
        row = {
            "episode_id": episode_id,
            "sample_selection_order": int(episode["sample_selection_order"]),
            "cohort": cohort,
            "period": str(episode["period"]),
            "pair": str(episode["pair"]),
            "actual_contact_time": contact_row["actual_contact_time"],
            "control_assigned": True,
            "control_time": control_time,
            "control_reference_open": float(selected["open"]),
            "control_balance_usable": usable,
            "control_state_distance_mean": float(selected["distance_mean"]),
            "control_state_distance_max": float(selected["distance_max"]),
            "eligible_candidate_count_before_global_separation": len(pool),
            "eligible_candidate_count_after_global_separation": len(eligible),
            "selected_global_candidate_rank": int(selected["candidate_rank"]),
            "assignment_failure": "",
        }
        for feature in g3g.MATCH_FEATURES:
            row[f"actual_{feature}"] = float(contact_row[feature])
            row[f"control_{feature}"] = float(selected[feature])
        assignment_rows.append(row)
        audit_rows.extend(control_match_audit_rows(row, selected))
    assignments = DataFrame(assignment_rows).sort_values("sample_selection_order")
    audit = DataFrame(audit_rows).sort_values(["sample_selection_order", "match_feature"])
    return assignments, audit


def ranked_control_candidates(
    *,
    episode: Series,
    actual_contact_time: pd.Timestamp,
    actual_match: dict[str, float],
    surfaces: MinimalPairSurfaces,
    cohort_actual_times: list[pd.Timestamp],
) -> DataFrame:
    anchor = str(episode["cluster_causal_anchor_timeframe"])
    pre_hours = int(WINDOW_HOURS[anchor][0])
    visible_start = pd.Timestamp(episode["event_time"]) - pd.Timedelta(hours=pre_hours)
    latest = actual_contact_time - pd.Timedelta(minutes=CONTROL_GLOBAL_SEPARATION_MINUTES)
    scale_source = surfaces.match.loc[
        (surfaces.match["date"] >= visible_start) & (surfaces.match["date"] <= latest)
    ].dropna(subset=list(g3g.MATCH_FEATURES))
    if len(scale_source) < 10:
        return DataFrame()
    scales = g3g.robust_scales(scale_source[list(g3g.MATCH_FEATURES)].to_numpy(dtype=float))
    candidates = (
        surfaces.match.loc[
            (surfaces.match["date"] >= visible_start)
            & (surfaces.match["date"] <= latest)
            & surfaces.match["date"].dt.minute.mod(5).eq(0)
            & surfaces.match["complete_control_path"].eq(True)
            & surfaces.match["causal_level_contact_count"].eq(0)
        ]
        .dropna(subset=list(g3g.MATCH_FEATURES))
        .copy()
    )
    actual_vector = np.asarray(
        [actual_match[feature] for feature in g3g.MATCH_FEATURES], dtype=float
    )
    if not np.isfinite(actual_vector).all():
        raise ValueError(f"Actual match state is incomplete: {episode['episode_id']}")
    candidates = candidates.loc[
        np.sign(candidates["match_return_60m"].to_numpy(dtype=float))
        == int(np.sign(actual_match["match_return_60m"]))
    ].copy()
    candidates = candidates.loc[
        candidates["date"].map(
            lambda timestamp: separated_from_times(pd.Timestamp(timestamp), cohort_actual_times)
        )
    ].copy()
    if candidates.empty:
        return candidates
    values = candidates[list(g3g.MATCH_FEATURES)].to_numpy(dtype=float)
    standardized = np.abs(values - actual_vector[None, :]) / scales[None, :]
    candidates["distance_mean"] = standardized.mean(axis=1)
    candidates["distance_max"] = standardized.max(axis=1)
    for index, feature in enumerate(g3g.MATCH_FEATURES):
        candidates[f"distance__{feature}"] = standardized[:, index]
        candidates[f"scale__{feature}"] = scales[index]
    candidates = candidates.sort_values(
        ["distance_mean", "distance_max", "date"], kind="mergesort"
    ).reset_index(drop=True)
    candidates["candidate_rank"] = np.arange(1, len(candidates) + 1)
    return candidates


def separated_from_times(timestamp: pd.Timestamp, others: list[pd.Timestamp]) -> bool:
    minimum = pd.Timedelta(minutes=CONTROL_GLOBAL_SEPARATION_MINUTES)
    return all(abs(timestamp - other) >= minimum for other in others)


def control_match_audit_rows(assignment: dict[str, Any], selected: Series) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in g3g.MATCH_FEATURES:
        rows.append(
            {
                "episode_id": assignment["episode_id"],
                "sample_selection_order": assignment["sample_selection_order"],
                "cohort": assignment["cohort"],
                "pair": assignment["pair"],
                "actual_contact_time": assignment["actual_contact_time"],
                "control_time": assignment["control_time"],
                "match_feature": feature,
                "actual_value": assignment[f"actual_{feature}"],
                "control_value": assignment[f"control_{feature}"],
                "robust_scale": float(selected[f"scale__{feature}"]),
                "absolute_robust_distance": float(selected[f"distance__{feature}"]),
                "control_distance_mean": assignment["control_state_distance_mean"],
                "control_distance_max": assignment["control_state_distance_max"],
                "control_balance_usable": assignment["control_balance_usable"],
            }
        )
    return rows


def selective_expansion_intervals(sample: DataFrame, boundaries: DataFrame) -> DataFrame:
    flagged = set(
        boundaries.loc[boundaries["boundary_extension_indicated"].eq(True), "episode_id"].astype(
            str
        )
    )
    selected = sample.loc[sample["episode_id"].astype(str).isin(flagged)].copy()
    if selected.empty:
        return DataFrame(columns=["episode_ids", "pair", "download_command"])
    return g3g.analysis_acquisition_intervals(selected, expanded_boundaries=True).sort_values(
        ["pair", "interval_start_utc"]
    )


def audit_episode_windows(sample: DataFrame, *, expanded: bool) -> DataFrame:
    if sample.empty:
        return DataFrame(columns=["episode_id", "expected_rows", "actual_rows", "passed"])
    windows = expanded_windows() if expanded else WINDOW_HOURS
    return g3g.audit_analysis_coverage(sample=sample, window_hours=windows)


def expanded_windows() -> dict[str, tuple[int, int]]:
    return {timeframe: (hours[0] * 2, hours[1] * 2) for timeframe, hours in WINDOW_HOURS.items()}


def analyze_fixed_episode(
    *,
    episode: Series,
    assignment: Series,
    components: DataFrame,
    surfaces: MinimalPairSurfaces,
) -> dict[str, Any]:
    episode_id = str(episode["episode_id"])
    contact_time = pd.Timestamp(assignment["actual_contact_time"])
    reproduced = g3g.first_parent_zone_contact(episode, surfaces.minute)
    if reproduced != contact_time:
        raise ValueError(f"Frozen contact time changed: {episode_id}")
    checkpoints = g3g.CHECKPOINT_MINUTES[str(episode["cluster_causal_anchor_timeframe"])]
    actual_paths = g3g.checkpoint_paths(
        minute=surfaces.minute,
        timestamp=contact_time,
        reference_price=float(episode["level_price"]),
        zone_half_width=float(episode["zone_half_width"]),
        level_name=str(episode["level_name"]),
        checkpoints=checkpoints,
        episode_id=episode_id,
        cohort=str(episode["cohort"]),
        event_kind="actual_cluster_contact",
    )
    actual_match = {
        feature: float(assignment[f"actual_{feature}"]) for feature in g3g.MATCH_FEATURES
    }
    actual = {
        **g3g.episode_identity(episode),
        "event_kind": "actual_cluster_contact",
        "contact_time": contact_time,
        "reference_price": float(episode["level_price"]),
        "zone_half_width": float(episode["zone_half_width"]),
        "base_atr": float(episode["base_atr"]),
        **actual_match,
        **g3g.cluster_snapshot(episode, components),
        **g3g.primary_event_outcomes(actual_paths, actual_match),
        "control_state_distance_mean": math.nan,
        "control_state_distance_max": math.nan,
        "control_balance_usable": False,
        "causal_level_contacts_at_event": int(
            surfaces.match.loc[
                surfaces.match["date"].eq(contact_time), "causal_level_contact_count"
            ].iloc[0]
        ),
    }
    paths = list(actual_paths)
    control: dict[str, Any] | None = None
    if str(assignment["control_assigned"]).lower() == "true":
        control_time = pd.Timestamp(assignment["control_time"])
        control_reference = float(assignment["control_reference_open"])
        control_match = {
            feature: float(assignment[f"control_{feature}"]) for feature in g3g.MATCH_FEATURES
        }
        control_paths = g3g.checkpoint_paths(
            minute=surfaces.minute,
            timestamp=control_time,
            reference_price=control_reference,
            zone_half_width=float(episode["zone_half_width"]),
            level_name=str(episode["level_name"]),
            checkpoints=tuple(value for value in checkpoints if value <= CONTROL_HORIZON_MINUTES),
            episode_id=episode_id,
            cohort=str(episode["cohort"]),
            event_kind="same_state_no_level",
        )
        control = {
            **g3g.episode_identity(episode),
            "event_kind": "same_state_no_level",
            "contact_time": control_time,
            "reference_price": control_reference,
            "zone_half_width": float(episode["zone_half_width"]),
            "base_atr": float(episode["base_atr"]),
            **control_match,
            **g3g.empty_cluster_snapshot(),
            **g3g.primary_event_outcomes(control_paths, control_match),
            "control_state_distance_mean": float(assignment["control_state_distance_mean"]),
            "control_state_distance_max": float(assignment["control_state_distance_max"]),
            "control_balance_usable": parse_bool(assignment["control_balance_usable"]),
            "causal_level_contacts_at_event": 0,
        }
        paths.extend(control_paths)
    return {"actual": actual, "control": control, "paths": paths}


def summarize_scopes(actual: DataFrame, controls: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = {
        "normal_all": actual["cohort"].eq("normal"),
        "normal_excluding_btc": actual["cohort"].eq("normal") & ~actual["pair"].eq("BTC/USDT:USDT"),
        "btc_only": actual["pair"].eq("BTC/USDT:USDT"),
        "meme": actual["cohort"].eq("meme"),
        "pooled_descriptive": pd.Series(True, index=actual.index),
    }
    for scope, scope_mask in scopes.items():
        scope_actual = actual.loc[scope_mask].copy()
        periods = period_masks(scope_actual)
        for phase, phase_mask in periods.items():
            left = scope_actual.loc[phase_mask].copy()
            right = controls.loc[
                controls["episode_id"].isin(left["episode_id"])
                & controls["control_balance_usable"].eq(True)
            ].copy()
            shared = sorted(set(left["episode_id"]).intersection(right["episode_id"]))
            paired_left = left.loc[left["episode_id"].isin(shared)].set_index("episode_id")
            paired_right = right.loc[right["episode_id"].isin(shared)].set_index("episode_id")
            callable_rows = left.loc[left["direction_callable"].eq(True)]
            direction_counts = callable_rows["first_direction"].value_counts()
            majority = (
                float(direction_counts.max() / len(callable_rows))
                if len(callable_rows)
                else math.nan
            )
            success = left.loc[left["simple_trend_joint_reaction_and_direction"].eq(True)]
            success_counts = success["pair"].value_counts()
            paired_actual_reaction = (
                float(paired_left["reaction_60m"].mean()) if shared else math.nan
            )
            paired_control_reaction = (
                float(paired_right["reaction_60m"].mean()) if shared else math.nan
            )
            rows.append(
                {
                    "scope": scope,
                    "phase": phase,
                    "actual_episode_count": len(left),
                    "actual_pair_count": int(left["pair"].nunique()),
                    "balanced_pair_count": len(shared),
                    "actual_reaction_count": int(left["reaction_60m"].sum()),
                    "actual_reaction_rate": safe_mean(left["reaction_60m"]),
                    "paired_actual_reaction_rate": paired_actual_reaction,
                    "paired_control_reaction_rate": paired_control_reaction,
                    "paired_reaction_uplift": (
                        paired_actual_reaction - paired_control_reaction
                        if np.isfinite(paired_actual_reaction)
                        and np.isfinite(paired_control_reaction)
                        else math.nan
                    ),
                    "paired_excursion_difference_median": paired_difference_median(
                        paired_left, paired_right, "excursion_strength_60_half_widths"
                    ),
                    "paired_volume_ratio_difference_median": paired_difference_median(
                        paired_left, paired_right, "volume_ratio_post_pre_60m"
                    ),
                    "paired_volatility_ratio_difference_median": paired_difference_median(
                        paired_left, paired_right, "realized_volatility_ratio_60m"
                    ),
                    "paired_zone_overlap_difference_median": paired_difference_median(
                        paired_left, paired_right, "zone_overlap_fraction_60m"
                    ),
                    "paired_crossings_difference_median": paired_difference_median(
                        paired_left, paired_right, "reference_crossings_60m"
                    ),
                    "direction_callable_count": len(callable_rows),
                    "simple_trend_direction_correct_count": int(
                        callable_rows["simple_trend_direction_correct"].sum()
                    ),
                    "simple_trend_direction_accuracy": safe_mean(
                        callable_rows["simple_trend_direction_correct"]
                    ),
                    "retrospective_majority_direction_accuracy": majority,
                    "trend_minus_majority_accuracy": (
                        safe_mean(callable_rows["simple_trend_direction_correct"]) - majority
                        if np.isfinite(majority)
                        else math.nan
                    ),
                    "joint_reaction_direction_success_count": len(success),
                    "joint_reaction_direction_rate": safe_mean(
                        left["simple_trend_joint_reaction_and_direction"]
                    ),
                    "joint_success_pair_count": int(success["pair"].nunique()),
                    "maximum_joint_success_pair_share": (
                        float(success_counts.max() / len(success)) if len(success) else math.nan
                    ),
                    "first_through_count": int(left["first_path_through"].sum()),
                    "first_away_count": int(left["first_path_away"].sum()),
                    "zone_breakout_count": int(left["first_zone_breakout"].sum()),
                    "zone_rejection_count": int(left["first_zone_rejection"].sum()),
                    "zone_unresolved_or_tied_count": int(
                        (~left["first_zone_resolution"].isin(["breakout", "rejection"])).sum()
                    ),
                }
            )
    return DataFrame(rows)


def period_masks(frame: DataFrame) -> dict[str, Series]:
    periods = frame["period"].astype(str)
    return {
        "development": periods.str.contains("development", regex=False),
        "validation_early": periods.str.contains("validation_early", regex=False),
        "validation_late": periods.str.contains("validation_late", regex=False),
        "validation_combined": periods.str.contains("validation_", regex=False),
        "all": pd.Series(True, index=frame.index),
    }


def classify_confirmation(phase_summary: DataFrame, boundaries: DataFrame) -> dict[str, Any]:
    scope_results: dict[str, Any] = {}
    primary_passes: list[bool] = []
    for scope in ("normal_excluding_btc", "meme"):
        combined = one_summary_row(phase_summary, scope, "validation_combined")
        early = one_summary_row(phase_summary, scope, "validation_early")
        late = one_summary_row(phase_summary, scope, "validation_late")
        combined_pass = bool(
            combined["balanced_pair_count"] >= MINIMUM_BALANCED_VALIDATION_PAIRS
            and combined["paired_reaction_uplift"] > 0.0
            and combined["trend_minus_majority_accuracy"] > 0.0
            and combined["joint_reaction_direction_rate"] >= VALIDATION_JOINT_MINIMUM
            and combined["joint_success_pair_count"] >= MINIMUM_SUCCESS_PAIRS
            and combined["maximum_joint_success_pair_share"] <= MAXIMUM_SUCCESS_PAIR_SHARE
        )
        phase_repeat = bool(
            early["paired_reaction_uplift"] >= 0.0 and late["paired_reaction_uplift"] >= 0.0
        )
        retained = combined_pass and phase_repeat
        primary_passes.append(retained)
        scope_results[scope] = {
            "combined_validation_pass": combined_pass,
            "early_late_reaction_uplift_does_not_reverse": phase_repeat,
            "retained": retained,
            "combined_validation": serializable_row(combined),
            "validation_early": serializable_row(early),
            "validation_late": serializable_row(late),
        }
    if all(primary_passes):
        classification = "retained_in_both_primary_cohorts"
    elif any(primary_passes):
        classification = "mixed_one_primary_cohort_retained"
    else:
        classification = "parked_larger_confirmation_did_not_repeat"
    return {
        "status": "fixed_confirmation_complete",
        "classification": classification,
        "primary_scope_results": scope_results,
        "btc_validation_descriptive": serializable_row(
            one_summary_row(phase_summary, "btc_only", "validation_combined")
        ),
        "normal_all_validation_descriptive": serializable_row(
            one_summary_row(phase_summary, "normal_all", "validation_combined")
        ),
        "main_target": MAIN_TARGET,
        "main_target_is_promotion": False,
        "final_boundary_warning_count": int(boundaries["boundary_extension_indicated"].sum()),
        "interpretation_limits": [
            "The result tests reaction and immediate resolution, not profit.",
            "The 55% and 65% values are user programme targets, not native ML scores.",
            "No generic technical indicator or cross was selected from these outcomes.",
            "BTC remains separate because it is structurally different from the altcoin cohort.",
            (
                "Normal and meme cohorts may legitimately differ; neither is generalized "
                "to all crypto."
            ),
        ],
    }


def one_summary_row(frame: DataFrame, scope: str, phase: str) -> Series:
    selected = frame.loc[frame["scope"].eq(scope) & frame["phase"].eq(phase)]
    if len(selected) != 1:
        raise ValueError(f"Expected one summary row for {scope} {phase}")
    return selected.iloc[0]


def paired_difference_median(left: DataFrame, right: DataFrame, column: str) -> float:
    paired = pd.concat(
        [
            pd.to_numeric(left[column], errors="coerce").rename("actual"),
            pd.to_numeric(right[column], errors="coerce").rename("control"),
        ],
        axis=1,
    ).dropna()
    return float((paired["actual"] - paired["control"]).median()) if len(paired) else math.nan


def safe_mean(values: Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.mean()) if len(numeric.dropna()) else math.nan


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def serializable_row(row: Series) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            output[str(key)] = None
        elif isinstance(value, (np.integer, int)):
            output[str(key)] = int(value)
        elif isinstance(value, (np.floating, float)):
            output[str(key)] = float(value)
        elif isinstance(value, (np.bool_, bool)):
            output[str(key)] = bool(value)
        else:
            output[str(key)] = str(value)
    return output


def validated_method_record(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "frozen_before_reaction_outcomes":
        raise ValueError(f"G4A method is not frozen: {path}")
    expected = str(record["request_contract"]["analysis_script_sha256"])
    if sha256_file(Path(__file__).resolve()) != expected:
        raise ValueError("G4A confirmation script changed after method freeze")
    return record


def validated_method_artifact(record: dict[str, Any], name: str) -> Path:
    artifact = record["artifacts"][name]
    path = Path(str(artifact["path"]))
    if not path.is_file() or sha256_file(path) != str(artifact["sha256"]):
        raise ValueError(f"Frozen G4A method artifact changed: {path}")
    return path


def count_true_by(frame: DataFrame, group: str, value: str) -> dict[str, int]:
    return {
        str(key): int(group_frame[value].astype(str).str.lower().eq("true").sum())
        for key, group_frame in frame.groupby(group, observed=True)
    }


if __name__ == "__main__":
    raise SystemExit(main())

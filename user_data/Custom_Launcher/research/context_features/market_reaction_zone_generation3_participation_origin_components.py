from __future__ import annotations

# Fix numerical-library thread counts before importing numpy/pandas.
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
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    LARGE_ARTIFACT_ROOT,
    OUTPUT_ROOT,
    atomic_write_json,
    atomic_write_parquet,
    load_manifest,
    prepare_base_market_frame,
    sha256_file,
    utc_now,
    validate_worker_count,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_localization import (  # noqa: E501
    pair_stem,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_anchored_vwap import (  # noqa: E501
    SEPARATION_HOURS_BY_RESPONSE_WINDOW,
    stable_json_sha256,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_density_reaction import (  # noqa: E501
    ARTIFACT_ROOT as G2E_ARTIFACT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_density_reaction import (  # noqa: E501
    REPORT_ROOT as G2E_REPORT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_density_reaction import (  # noqa: E501
    build_density_surfaces,
    cohort_period_results,
    leave_one_coin_out_results,
    pair_period_results,
    purge_density_matches,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_lvn_attribution import (  # noqa: E501
    select_pairs,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_participation_origin import (  # noqa: E501
    ARTIFACT_ROOT as G3E_PREFLIGHT_ARTIFACT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_participation_origin import (  # noqa: E501
    REPORT_ROOT as G3E_PREFLIGHT_REPORT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_participation_origin import (  # noqa: E501
    validate_frozen_branch,
)


FROZEN_BATCH = OUTPUT_ROOT / "generation2_review" / "g3_frozen_branch_batch.json"
REPORT_ROOT = OUTPUT_ROOT / "generation3_branches" / "g3e_participation_origin_components"
ARTIFACT_ROOT = (
    LARGE_ARTIFACT_ROOT
    / "generation3_branches"
    / "g3e_participation_origin_components"
)
OUTPUT_SCHEMA_VERSION = 1
DENSITY_ORIGIN_SCOPES = (
    "origin_plus_density",
    "origin_plus_density_and_reference",
)
COMPONENT_CONTROLS = (
    "current_vp_hvn",
    "current_vp_lvn",
    "current_vp_poc",
    "simple_prior_high_same_history",
    "simple_prior_low_same_history",
    "shift_-1atr",
    "shift_+1atr",
    "shuffled_swing_residual_assignment",
    "same_state_no_level",
    "same_density_coverage_no_level",
)


@dataclass(frozen=True)
class PairTask:
    pair: str
    manifest_path: str
    geometry_path: str
    geometry_sha256: str
    g2e_pair_path: str
    g2e_pair_sha256: str
    supported_scopes: tuple[str, ...]
    run_id: str
    request_sha256: str
    overwrite: bool


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 3E component-attribution test for participation-origin contacts "
            "that overlap a causal density zone."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cohort", choices=("large", "meme"), required=True)
    parser.add_argument("--preflight-run-id", required=True)
    parser.add_argument("--g2e-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    validate_worker_count(args.workers, manifest=manifest)
    validate_frozen_branch(args.cohort, manifest)
    pairs = select_pairs(manifest, args.pairs)

    preflight_dir = G3E_PREFLIGHT_REPORT_ROOT / args.preflight_run_id
    preflight_artifact_dir = G3E_PREFLIGHT_ARTIFACT_ROOT / args.preflight_run_id
    g2e_dir = G2E_REPORT_ROOT / args.g2e_run_id
    g2e_artifact_dir = G2E_ARTIFACT_ROOT / args.g2e_run_id
    preflight_record_path = preflight_dir / "g3e_preflight_run_record.json"
    preflight_integrity_path = preflight_dir / "g3e_preflight_integrity.json"
    support_path = preflight_dir / "g3e_comparison_support.parquet"
    g2e_record_path = g2e_dir / "g2e_reaction_run_record.json"
    g2e_integrity_path = g2e_dir / "g2e_reaction_integrity.json"
    supported_scopes = validate_sources(
        cohort=args.cohort,
        pairs=pairs,
        preflight_record_path=preflight_record_path,
        preflight_integrity_path=preflight_integrity_path,
        support_path=support_path,
        g2e_record_path=g2e_record_path,
        g2e_integrity_path=g2e_integrity_path,
    )
    pair_contracts = []
    for pair in pairs:
        geometry_path = (
            preflight_artifact_dir / "pair_geometry" / f"{pair_stem(pair)}.parquet"
        )
        g2e_path = g2e_artifact_dir / "pair_matches" / f"{pair_stem(pair)}.parquet"
        pair_contracts.append(
            {
                "pair": pair,
                "geometry_path": str(geometry_path),
                "geometry_sha256": sha256_file(geometry_path),
                "g2e_pair_path": str(g2e_path),
                "g2e_pair_sha256": sha256_file(g2e_path),
            }
        )

    request = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "frozen_batch_sha256": sha256_file(FROZEN_BATCH),
        "cohort": args.cohort,
        "pairs": list(pairs),
        "preflight_run_id": args.preflight_run_id,
        "preflight_record_sha256": sha256_file(preflight_record_path),
        "preflight_integrity_sha256": sha256_file(preflight_integrity_path),
        "preflight_support_sha256": sha256_file(support_path),
        "g2e_run_id": args.g2e_run_id,
        "g2e_record_sha256": sha256_file(g2e_record_path),
        "g2e_integrity_sha256": sha256_file(g2e_integrity_path),
        "pair_source_contracts": pair_contracts,
        "supported_origin_scopes_opened": list(supported_scopes),
        "controls": list(COMPONENT_CONTROLS),
        "selection": (
            "Keep only G2E density events whose exact causal density zone overlaps the "
            "participation-origin zone and whose event timestamp is a G3E contact."
        ),
        "interpretation_boundary": (
            "This asks whether the complete origin-plus-density arrangement exceeds the "
            "existing density control ladder. It does not by itself isolate the added "
            "value of the participation origin."
        ),
        "reaction_outcomes_loaded": True,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    request_sha256 = stable_json_sha256(request)
    run_dir = REPORT_ROOT / args.run_id
    artifact_dir = ARTIFACT_ROOT / args.run_id
    pair_dir = artifact_dir / "pair_independent_matches"
    run_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    record_path = run_dir / "g3e_component_run_record.json"
    if record_path.is_file() and not args.overwrite:
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        if existing.get("request_sha256") != request_sha256:
            raise ValueError("Run ID already exists with an incompatible request contract.")
    record = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "status": "running",
        "started_at_utc": utc_now(),
        "orchestrator_pid": os.getpid(),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "request_sha256": request_sha256,
        "request_contract": request,
        "baseline": (
            "The direct G3E comparison left one normal-coin activity lead only when the "
            "origin overlapped both density and another reference level."
        ),
        "hypothesis": (
            "The named origin-plus-density arrangement has a repeated direction-neutral "
            "response beyond alternative density locations, VP nodes, prior extremes, "
            "and same-state no-level times."
        ),
        "pass_fail": (
            "Retain only a named outcome with fair matching, at least five coins and 50 "
            "independent pairs, the same sign in both validations, and leave-one-coin-out "
            "sign stability against every applicable component control."
        ),
        "workers": args.workers,
        "reaction_outcomes_loaded": True,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(record, record_path)
    tasks = [
        PairTask(
            pair=row["pair"],
            manifest_path=str(manifest_path),
            geometry_path=row["geometry_path"],
            geometry_sha256=row["geometry_sha256"],
            g2e_pair_path=row["g2e_pair_path"],
            g2e_pair_sha256=row["g2e_pair_sha256"],
            supported_scopes=supported_scopes,
            run_id=args.run_id,
            request_sha256=request_sha256,
            overwrite=args.overwrite,
        )
        for row in pair_contracts
    ]
    try:
        results = run_tasks(tasks, workers=args.workers)
        inventory = DataFrame(results)
        atomic_write_parquet(inventory, run_dir / "g3e_component_pair_inventory.parquet")
        failures = inventory.loc[inventory["status"].eq("failed")]
        if not failures.empty:
            raise RuntimeError(
                f"{len(failures)} pair task(s) failed; inspect the pair inventory."
            )
        independent = combine_pair_outputs(
            pair_dir=pair_dir,
            pairs=pairs,
            request_sha256=request_sha256,
        )
        atomic_write_parquet(
            independent,
            artifact_dir / "g3e_component_independent_pairs.parquet",
        )
        pair_period = pair_period_results(independent)
        cohort_period = cohort_period_results(pair_period, independent)
        leave_one_out = leave_one_coin_out_results(pair_period)
        atomic_write_parquet(pair_period, run_dir / "g3e_pair_period_results.parquet")
        atomic_write_parquet(cohort_period, run_dir / "g3e_cohort_period_results.parquet")
        atomic_write_parquet(leave_one_out, run_dir / "g3e_leave_one_coin_out.parquet")
        integrity = integrity_record(
            results=results,
            independent=independent,
            supported_scopes=supported_scopes,
            requested_pairs=pairs,
        )
        atomic_write_json(integrity, run_dir / "g3e_component_integrity.json")
        if not integrity["passed"]:
            raise RuntimeError("G3E component-control integrity validation failed.")
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "pair_tasks": results,
                "independent_comparisons": len(independent),
                "pair_period_rows": len(pair_period),
                "cohort_period_rows": len(cohort_period),
                "leave_one_coin_out_rows": len(leave_one_out),
                "evidence_eligible_rows": evidence_eligible_count(cohort_period),
                "integrity": str(run_dir / "g3e_component_integrity.json"),
                "bulky_artifacts": str(artifact_dir),
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


def validate_sources(
    *,
    cohort: str,
    pairs: Sequence[str],
    preflight_record_path: Path,
    preflight_integrity_path: Path,
    support_path: Path,
    g2e_record_path: Path,
    g2e_integrity_path: Path,
) -> tuple[str, ...]:
    preflight_record = json.loads(preflight_record_path.read_text(encoding="utf-8"))
    preflight_integrity = json.loads(preflight_integrity_path.read_text(encoding="utf-8"))
    g2e_record = json.loads(g2e_record_path.read_text(encoding="utf-8"))
    g2e_integrity = json.loads(g2e_integrity_path.read_text(encoding="utf-8"))
    if preflight_record.get("status") != "completed" or not preflight_integrity.get("passed"):
        raise ValueError("The G3E coverage preflight is not complete and clean.")
    preflight_request = preflight_record["request_contract"]
    if preflight_request.get("cohort") != cohort:
        raise ValueError("The G3E preflight cohort does not match this component run.")
    if not set(pairs).issubset(preflight_request.get("pairs", [])):
        raise ValueError("The G3E preflight does not contain every requested pair.")
    if g2e_record.get("status") != "completed" or not g2e_integrity.get("passed"):
        raise ValueError("The G2E source reaction run is not complete and clean.")
    g2e_request = g2e_record["request"]
    if not set(pairs).issubset(g2e_request.get("pairs", [])):
        raise ValueError("The G2E source does not contain every requested pair.")
    for source in (preflight_request, g2e_request):
        if source.get("direction_prediction") is not False:
            raise ValueError("A source violates direction_prediction=False.")
        if source.get("profit_optimization") is not False:
            raise ValueError("A source violates profit_optimization=False.")
    support = pd.read_parquet(support_path)
    supported = set(
        support.loc[
            support["both_validation_periods_supported"].fillna(False), "origin_scope"
        ].astype(str)
    )
    scopes = tuple(scope for scope in DENSITY_ORIGIN_SCOPES if scope in supported)
    if not scopes:
        raise ValueError("No density-overlap G3E scope passed both validation preflights.")
    return scopes


def evidence_eligible_count(frame: DataFrame) -> int:
    if "evidence_eligible" not in frame:
        return 0
    return int(frame["evidence_eligible"].fillna(False).sum())


def run_tasks(tasks: Sequence[PairTask], *, workers: int) -> list[dict[str, Any]]:
    if workers == 1:
        return [safe_build_pair(task) for task in tasks]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(safe_build_pair, task): task for task in tasks}
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda row: row["pair"])


def safe_build_pair(task: PairTask) -> dict[str, Any]:
    try:
        return build_pair(task)
    except Exception as exc:
        return {"pair": task.pair, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def build_pair(task: PairTask) -> dict[str, Any]:
    started = time.perf_counter()
    output_path = (
        ARTIFACT_ROOT
        / task.run_id
        / "pair_independent_matches"
        / f"{pair_stem(task.pair)}.parquet"
    )
    if output_path.is_file() and not task.overwrite:
        existing = validate_pair_output(
            output_path,
            pair=task.pair,
            request_sha256=task.request_sha256,
        )
        return {
            "pair": task.pair,
            "status": "existing",
            "independent_comparisons": len(existing),
            "seconds": round(time.perf_counter() - started, 3),
        }
    geometry_path = Path(task.geometry_path)
    g2e_path = Path(task.g2e_pair_path)
    if sha256_file(geometry_path) != task.geometry_sha256:
        raise ValueError(f"G3E preflight geometry changed: {geometry_path}")
    if sha256_file(g2e_path) != task.g2e_pair_sha256:
        raise ValueError(f"G2E pair evidence changed: {g2e_path}")
    geometry = pd.read_parquet(geometry_path)
    contacts = geometry.loc[
        geometry["event_kind"].eq("contact")
        & geometry["origin_scope"].isin(task.supported_scopes)
    ].copy()
    if contacts.empty:
        raise ValueError(f"No supported density-overlap G3E contacts for {task.pair}.")
    matches = pd.read_parquet(g2e_path, filters=[("control", "in", list(COMPONENT_CONTROLS))])
    manifest = load_manifest(Path(task.manifest_path))
    base = prepare_base_market_frame(task.pair, manifest)
    surfaces = build_density_surfaces(base)
    selected = origin_overlapping_density_matches(
        matches=matches,
        contacts=contacts,
        surfaces=surfaces,
    )
    if selected.empty:
        raise ValueError(f"No exact G2E density component overlaps for {task.pair}.")
    selected["route_id"] = "g3e_origin_density_cluster_vs_component_control"
    independent = purge_density_matches(selected)
    if independent.empty:
        raise ValueError(f"No independent G3E component comparisons for {task.pair}.")
    independent["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    independent["run_request_sha256"] = task.request_sha256
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(independent, output_path)
    return {
        "pair": task.pair,
        "status": "completed",
        "origin_contact_events": len(contacts),
        "matched_comparisons_before_overlap_purge": len(selected),
        "independent_comparisons": len(independent),
        "controls": ";".join(sorted(independent["control"].astype(str).unique())),
        "seconds": round(time.perf_counter() - started, 3),
    }


def origin_overlapping_density_matches(
    *,
    matches: DataFrame,
    contacts: DataFrame,
    surfaces: Sequence[Any],
) -> DataFrame:
    contact_keys = contacts.rename(
        columns={
            "event_time": "actual_event_time",
            "base_index": "actual_base_index",
            "level_price": "origin_level_price",
            "zone_half_width": "origin_zone_half_width",
            "actual_scope": "preflight_actual_scope",
        }
    )[
        [
            "actual_event_time",
            "actual_base_index",
            "origin_scope",
            "origin_level_price",
            "origin_zone_half_width",
            "overlap_density_zone_count",
            "overlap_reference_level_count",
        ]
    ].drop_duplicates(["actual_event_time", "actual_base_index"])
    joined = matches.merge(
        contact_keys,
        on=["actual_event_time", "actual_base_index"],
        how="inner",
        validate="many_to_one",
    )
    if joined.empty:
        return joined
    lookup = {(surface.family, int(surface.history_hours)): surface for surface in surfaces}
    density_level = np.full(len(joined), np.nan, dtype=float)
    density_width = np.full(len(joined), np.nan, dtype=float)
    for (family, history), positions in joined.groupby(
        ["density_family", "history_hours"], observed=True
    ).groups.items():
        surface = lookup.get((str(family), int(history)))
        if surface is None:
            continue
        row_positions = np.asarray(list(positions), dtype=int)
        indexes = joined.loc[row_positions, "actual_base_index"].to_numpy(dtype=np.int64)
        ranks = joined.loc[row_positions, "actual_zone_rank"].to_numpy(dtype=np.int64)
        valid = (
            (indexes >= 0)
            & (indexes < surface.zones.levels.shape[0])
            & (ranks >= 0)
            & (ranks < surface.zones.levels.shape[1])
        )
        density_level[row_positions[valid]] = surface.zones.levels[
            indexes[valid], ranks[valid]
        ]
        density_width[row_positions[valid]] = surface.zones.half_widths[
            indexes[valid], ranks[valid]
        ]
    joined["component_density_level_price"] = density_level
    joined["component_density_zone_half_width"] = density_width
    origin_level = pd.to_numeric(joined["origin_level_price"], errors="coerce").to_numpy()
    origin_width = pd.to_numeric(
        joined["origin_zone_half_width"], errors="coerce"
    ).to_numpy()
    overlap = (
        np.isfinite(density_level)
        & np.isfinite(density_width)
        & np.isfinite(origin_level)
        & np.isfinite(origin_width)
        & (np.abs(density_level - origin_level) <= density_width + origin_width)
    )
    reference_scope_ok = np.where(
        joined["origin_scope"].eq("origin_plus_density_and_reference"),
        joined["actual_scope"].eq("density_plus_reference_cluster"),
        ~joined["actual_scope"].eq("density_plus_reference_cluster"),
    )
    output = joined.loc[overlap & reference_scope_ok].copy()
    if output.empty:
        return output
    output["density_actual_scope"] = output["actual_scope"].astype(str)
    output["actual_scope"] = output["origin_scope"].astype(str)
    output["origin_density_centre_gap"] = (
        pd.to_numeric(output["component_density_level_price"], errors="coerce")
        - pd.to_numeric(output["origin_level_price"], errors="coerce")
    ).abs()
    return output.reset_index(drop=True)


def integrity_record(
    *,
    results: Sequence[dict[str, Any]],
    independent: DataFrame,
    supported_scopes: Sequence[str],
    requested_pairs: Sequence[str],
) -> dict[str, Any]:
    failures = [row for row in results if row["status"] == "failed"]
    invalid_scopes = sorted(
        set(independent["actual_scope"].astype(str)).difference(supported_scopes)
    )
    present_pairs = sorted(independent["pair"].astype(str).unique())
    missing_pairs = sorted(set(requested_pairs).difference(present_pairs))
    separation = pd.to_numeric(independent["event_separation_hours"], errors="coerce")
    required = independent["response_window"].map(SEPARATION_HOURS_BY_RESPONSE_WINDOW)
    separation_violations = int((separation < required).sum())
    origin_level = pd.to_numeric(independent["origin_level_price"], errors="coerce")
    origin_width = pd.to_numeric(independent["origin_zone_half_width"], errors="coerce")
    density_level = pd.to_numeric(
        independent["component_density_level_price"], errors="coerce"
    )
    density_width = pd.to_numeric(
        independent["component_density_zone_half_width"], errors="coerce"
    )
    overlap_violations = int(
        ((density_level - origin_level).abs() > density_width + origin_width).sum()
    )
    reference_scope_violations = int(
        (
            independent["actual_scope"].eq("origin_plus_density_and_reference")
            ^ independent["density_actual_scope"].eq("density_plus_reference_cluster")
        ).sum()
    )
    direction_violations = int(independent["direction_prediction"].ne(False).sum())
    profit_violations = int(independent["profit_optimization"].ne(False).sum())
    duplicate_rows = int(
        independent.duplicated(
            [
                "route_id",
                "density_family",
                "history_hours",
                "actual_scope",
                "control",
                "pair",
                "period",
                "approach_state",
                "response_window",
                "actual_event_time",
                "control_event_time",
            ]
        ).sum()
    )
    return {
        "created_at_utc": utc_now(),
        "passed": bool(
            not failures
            and not independent.empty
            and not invalid_scopes
            and not missing_pairs
            and separation_violations == 0
            and overlap_violations == 0
            and reference_scope_violations == 0
            and direction_violations == 0
            and profit_violations == 0
            and duplicate_rows == 0
        ),
        "failures": failures,
        "independent_comparisons": len(independent),
        "supported_scopes": list(supported_scopes),
        "observed_scopes": sorted(independent["actual_scope"].astype(str).unique()),
        "invalid_scopes": invalid_scopes,
        "requested_pairs": list(requested_pairs),
        "present_pairs": present_pairs,
        "missing_pairs": missing_pairs,
        "controls_present": sorted(independent["control"].astype(str).unique()),
        "event_separation_violations": separation_violations,
        "exact_origin_density_overlap_violations": overlap_violations,
        "reference_scope_violations": reference_scope_violations,
        "duplicate_independent_rows": duplicate_rows,
        "direction_prediction_violations": direction_violations,
        "profit_optimization_violations": profit_violations,
        "direction_prediction": False,
        "profit_optimization": False,
    }


def validate_pair_output(path: Path, *, pair: str, request_sha256: str) -> DataFrame:
    frame = pd.read_parquet(path)
    if frame.empty or set(frame["pair"].astype(str)) != {pair}:
        raise ValueError(f"Existing output has an incompatible pair: {path}")
    if set(frame["output_schema_version"].astype(int)) != {OUTPUT_SCHEMA_VERSION}:
        raise ValueError(f"Existing output has an incompatible schema: {path}")
    if set(frame["run_request_sha256"].astype(str)) != {request_sha256}:
        raise ValueError(f"Existing output belongs to another request: {path}")
    return frame


def combine_pair_outputs(
    *, pair_dir: Path, pairs: Sequence[str], request_sha256: str
) -> DataFrame:
    return pd.concat(
        [
            validate_pair_output(
                pair_dir / f"{pair_stem(pair)}.parquet",
                pair=pair,
                request_sha256=request_sha256,
            )
            for pair in pairs
        ],
        ignore_index=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())

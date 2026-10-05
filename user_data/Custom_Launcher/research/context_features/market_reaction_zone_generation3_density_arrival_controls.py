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
    cohort_period_results,
    leave_one_coin_out_results,
    pair_period_results,
    purge_density_matches,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_density_arrival import (  # noqa: E501
    ARTIFACT_ROOT as G3D_PREFLIGHT_ARTIFACT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_density_arrival import (  # noqa: E501
    REPORT_ROOT as G3D_PREFLIGHT_REPORT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_density_arrival import (  # noqa: E501
    validate_frozen_branch,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_lvn_attribution import (  # noqa: E501
    select_pairs,
)


FROZEN_BATCH = OUTPUT_ROOT / "generation2_review" / "g3_frozen_branch_batch.json"
REPORT_ROOT = OUTPUT_ROOT / "generation3_branches" / "g3d_density_arrival_controls"
ARTIFACT_ROOT = LARGE_ARTIFACT_ROOT / "generation3_branches" / "g3d_density_arrival_controls"
OUTPUT_SCHEMA_VERSION = 1
FRESH_STATE = "first_arrival_after_outside_interval"
GEOMETRY_CONTROLS = (
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
    preflight_geometry_path: str
    preflight_geometry_sha256: str
    g2e_pair_path: str
    g2e_pair_sha256: str
    supported_cells: tuple[tuple[str, int, str], ...]
    run_id: str
    request_sha256: str
    overwrite: bool


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 3D fresh-density-arrival reaction test against the frozen G2E "
            "geometry and no-level control ladder."
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
    preflight_dir = G3D_PREFLIGHT_REPORT_ROOT / args.preflight_run_id
    preflight_artifact_dir = G3D_PREFLIGHT_ARTIFACT_ROOT / args.preflight_run_id
    g2e_dir = G2E_REPORT_ROOT / args.g2e_run_id
    g2e_artifact_dir = G2E_ARTIFACT_ROOT / args.g2e_run_id
    preflight_record_path = preflight_dir / "g3d_preflight_run_record.json"
    preflight_integrity_path = preflight_dir / "g3d_preflight_integrity.json"
    support_path = preflight_dir / "g3d_comparison_support.parquet"
    g2e_record_path = g2e_dir / "g2e_reaction_run_record.json"
    g2e_integrity_path = g2e_dir / "g2e_reaction_integrity.json"
    supported_cells = validate_sources_and_supported_cells(
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
        preflight_path = preflight_artifact_dir / "pair_geometry" / f"{pair_stem(pair)}.parquet"
        g2e_path = g2e_artifact_dir / "pair_matches" / f"{pair_stem(pair)}.parquet"
        pair_contracts.append(
            {
                "pair": pair,
                "preflight_geometry_path": str(preflight_path),
                "preflight_geometry_sha256": sha256_file(preflight_path),
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
        "actual_event_state": FRESH_STATE,
        "supported_cells_opened": supported_cells,
        "controls": list(GEOMETRY_CONTROLS),
        "control_interpretation": {
            "current_vp_hvn_lvn_poc": "Does another current volume-profile node behave similarly?",
            "prior_high_low": "Does a simple prior extreme explain the same response?",
            "shifted_density_zone": (
                "Does nearby price, rather than the exact density area, explain it?"
            ),
            "shuffled_swing_location": "Does the causal swing-density location matter?",
            "same_state_no_level": "Does the already-active market state explain it?",
            "same_density_coverage_no_level": (
                "Does broad density-zone coverage explain it without a level at that price?"
            ),
        },
        "outcomes": (
            "range, absolute movement, volume, absolute pressure change, dwell, and crossings "
            "over 1h, 4h, and 24h where available"
        ),
        "independence": (
            "Re-run the G2E future-path purge after filtering to fresh arrivals; one-, four-, "
            "and twenty-four-hour outcomes use their own separation horizons."
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
    record_path = run_dir / "g3d_control_run_record.json"
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
            "G3D coverage separated fresh arrivals from occupancy and admitted outcomes only "
            "for cells with fresh, near-miss, and occupancy support in both validations."
        ),
        "hypothesis": (
            "A clean first arrival at a continuously known density zone changes at least one "
            "direction-neutral reaction measure beyond the frozen geometry ladder."
        ),
        "pass_fail": (
            "Retain only a named response that has usable matching balance and repeated "
            "multi-coin support with the same sign in both validation periods against the "
            "applicable exact-location, alternative-level, and no-level controls."
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
            preflight_geometry_path=row["preflight_geometry_path"],
            preflight_geometry_sha256=row["preflight_geometry_sha256"],
            g2e_pair_path=row["g2e_pair_path"],
            g2e_pair_sha256=row["g2e_pair_sha256"],
            supported_cells=tuple(
                (
                    str(cell["density_family"]),
                    int(cell["history_hours"]),
                    str(cell["actual_scope"]),
                )
                for cell in supported_cells
            ),
            run_id=args.run_id,
            request_sha256=request_sha256,
            overwrite=args.overwrite,
        )
        for row in pair_contracts
    ]
    try:
        results = run_tasks(tasks, workers=args.workers)
        inventory = DataFrame(results)
        atomic_write_parquet(inventory, run_dir / "g3d_control_pair_inventory.parquet")
        failures = inventory.loc[inventory["status"].eq("failed")]
        if not failures.empty:
            raise RuntimeError(
                f"{len(failures)} pair task(s) failed; inspect g3d_control_pair_inventory.parquet."
            )
        independent = combine_pair_outputs(
            pair_dir=pair_dir,
            pairs=pairs,
            request_sha256=request_sha256,
        )
        atomic_write_parquet(independent, artifact_dir / "g3d_independent_pairs.parquet")
        pair_period = pair_period_results(independent)
        cohort_period = cohort_period_results(pair_period, independent)
        leave_one_out = leave_one_coin_out_results(pair_period)
        atomic_write_parquet(pair_period, run_dir / "g3d_pair_period_results.parquet")
        atomic_write_parquet(cohort_period, run_dir / "g3d_cohort_period_results.parquet")
        atomic_write_parquet(leave_one_out, run_dir / "g3d_leave_one_coin_out.parquet")
        integrity = integrity_record(
            results=results,
            independent=independent,
            supported_cells=supported_cells,
            requested_pairs=pairs,
        )
        atomic_write_json(integrity, run_dir / "g3d_control_integrity.json")
        if not integrity["passed"]:
            raise RuntimeError("G3D control-run integrity validation failed.")
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "pair_tasks": results,
                "independent_comparisons": len(independent),
                "pair_period_rows": len(pair_period),
                "cohort_period_rows": len(cohort_period),
                "leave_one_coin_out_rows": len(leave_one_out),
                "evidence_eligible_rows": int(cohort_period["evidence_eligible"].sum()),
                "integrity": str(run_dir / "g3d_control_integrity.json"),
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


def validate_sources_and_supported_cells(
    *,
    cohort: str,
    pairs: Sequence[str],
    preflight_record_path: Path,
    preflight_integrity_path: Path,
    support_path: Path,
    g2e_record_path: Path,
    g2e_integrity_path: Path,
) -> list[dict[str, Any]]:
    preflight_record = json.loads(preflight_record_path.read_text(encoding="utf-8"))
    preflight_integrity = json.loads(preflight_integrity_path.read_text(encoding="utf-8"))
    g2e_record = json.loads(g2e_record_path.read_text(encoding="utf-8"))
    g2e_integrity = json.loads(g2e_integrity_path.read_text(encoding="utf-8"))
    if preflight_record.get("status") != "completed" or not preflight_integrity.get("passed"):
        raise ValueError("The G3D coverage preflight is not complete and clean.")
    preflight_request = preflight_record["request_contract"]
    preflight_pairs = list(preflight_request.get("pairs", []))
    if preflight_request.get("cohort") != cohort or not set(pairs).issubset(preflight_pairs):
        raise ValueError("The G3D preflight cohort does not contain every requested pair.")
    if g2e_record.get("status") != "completed" or not g2e_integrity.get("passed"):
        raise ValueError("The G2E source reaction run is not complete and clean.")
    g2e_request = g2e_record["request"]
    if not set(pairs).issubset(g2e_request.get("pairs", [])):
        raise ValueError("The G2E source does not contain every requested pair.")
    if (
        g2e_request.get("direction_prediction") is not False
        or g2e_request.get("profit_optimization") is not False
    ):
        raise ValueError("The G2E source crossed the direction/profit research boundary.")
    support = pd.read_parquet(support_path)
    supported = (
        support.loc[
            support["both_validation_periods_supported"],
            ["density_family", "history_hours", "actual_scope"],
        ]
        .drop_duplicates()
        .sort_values(["density_family", "history_hours", "actual_scope"], kind="stable")
    )
    if supported.empty:
        raise ValueError("The G3D preflight admitted no common-support cells.")
    return supported.to_dict(orient="records")


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
        ARTIFACT_ROOT / task.run_id / "pair_independent_matches" / f"{pair_stem(task.pair)}.parquet"
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
    preflight_path = Path(task.preflight_geometry_path)
    g2e_path = Path(task.g2e_pair_path)
    if sha256_file(preflight_path) != task.preflight_geometry_sha256:
        raise ValueError(f"G3D preflight geometry changed: {preflight_path}")
    if sha256_file(g2e_path) != task.g2e_pair_sha256:
        raise ValueError(f"G2E pair evidence changed: {g2e_path}")
    geometry = pd.read_parquet(preflight_path)
    geometry = filter_supported_cells(geometry, task.supported_cells)
    fresh = geometry.loc[geometry["event_state"].eq(FRESH_STATE)].copy()
    if fresh.empty:
        raise ValueError(f"No fresh-arrival geometry for {task.pair}.")
    event_keys = ["density_family", "history_hours", "actual_scope"]
    fresh = fresh.rename(
        columns={"event_time": "actual_event_time", "base_index": "actual_base_index"}
    )
    fresh_keys = fresh[
        [*event_keys, "actual_event_time", "actual_base_index", "event_state"]
    ].drop_duplicates()
    matches = pd.read_parquet(g2e_path, filters=[("control", "in", list(GEOMETRY_CONTROLS))])
    matches = matches.merge(
        fresh_keys,
        on=[*event_keys, "actual_event_time", "actual_base_index"],
        how="inner",
        validate="many_to_one",
    )
    if matches.empty:
        raise ValueError(f"No G2E controls matched fresh arrivals for {task.pair}.")
    matches["route_id"] = "g3d_fresh_arrival_vs_geometry_control"
    independent = purge_density_matches(matches)
    if independent.empty:
        raise ValueError(f"No independent G3D control matches for {task.pair}.")
    independent["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    independent["run_request_sha256"] = task.request_sha256
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(independent, output_path)
    return {
        "pair": task.pair,
        "status": "completed",
        "fresh_geometry_events": len(fresh_keys),
        "matched_comparisons_before_overlap_purge": len(matches),
        "independent_comparisons": len(independent),
        "controls": ";".join(sorted(independent["control"].astype(str).unique())),
        "seconds": round(time.perf_counter() - started, 3),
    }


def filter_supported_cells(
    geometry: DataFrame,
    supported_cells: Sequence[tuple[str, int, str]],
) -> DataFrame:
    allowed = set(supported_cells)
    keys = zip(
        geometry["density_family"].astype(str),
        geometry["history_hours"].astype(int),
        geometry["actual_scope"].astype(str),
        strict=True,
    )
    keep = [key in allowed for key in keys]
    return geometry.loc[keep].copy()


def integrity_record(
    *,
    results: Sequence[dict[str, Any]],
    independent: DataFrame,
    supported_cells: Sequence[dict[str, Any]],
    requested_pairs: Sequence[str],
) -> dict[str, Any]:
    failures = [row for row in results if row["status"] == "failed"]
    observed_cells = (
        independent[["density_family", "history_hours", "actual_scope"]]
        .drop_duplicates()
        .sort_values(["density_family", "history_hours", "actual_scope"], kind="stable")
        .to_dict(orient="records")
    )
    source_cells = sorted(
        (str(row["density_family"]), int(row["history_hours"]), str(row["actual_scope"]))
        for row in supported_cells
    )
    actual_cells = sorted(
        (str(row["density_family"]), int(row["history_hours"]), str(row["actual_scope"]))
        for row in observed_cells
    )
    separation = pd.to_numeric(independent["event_separation_hours"], errors="coerce")
    required = independent["response_window"].map(SEPARATION_HOURS_BY_RESPONSE_WINDOW)
    separation_violations = int((separation < required).sum())
    direction_violations = int(independent["direction_prediction"].ne(False).sum())
    profit_violations = int(independent["profit_optimization"].ne(False).sum())
    state_violations = int(independent["event_state"].ne(FRESH_STATE).sum())
    present_pairs = sorted(independent["pair"].astype(str).unique())
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
            and source_cells == actual_cells
            and set(present_pairs) == set(requested_pairs)
            and separation_violations == 0
            and direction_violations == 0
            and profit_violations == 0
            and state_violations == 0
            and duplicate_rows == 0
        ),
        "failures": failures,
        "independent_comparisons": len(independent),
        "supported_cells": supported_cells,
        "observed_cells": observed_cells,
        "requested_pairs": list(requested_pairs),
        "present_pairs": present_pairs,
        "controls_present": sorted(independent["control"].astype(str).unique()),
        "periods_present": sorted(independent["period"].astype(str).unique()),
        "separation_violations": separation_violations,
        "direction_prediction_violations": direction_violations,
        "profit_optimization_violations": profit_violations,
        "fresh_state_violations": state_violations,
        "duplicate_independent_rows": duplicate_rows,
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


def combine_pair_outputs(*, pair_dir: Path, pairs: Sequence[str], request_sha256: str) -> DataFrame:
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

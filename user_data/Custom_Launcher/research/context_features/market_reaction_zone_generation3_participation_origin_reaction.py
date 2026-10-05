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
    LevelSpec,
    atomic_write_json,
    atomic_write_parquet,
    event_frame,
    future_path_matrices,
    load_manifest,
    prepare_base_market_frame,
    sha256_file,
    utc_now,
    validate_worker_count,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_localization import (  # noqa: E501
    causal_local_state,
    nearest_state_pairs,
    pair_stem,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_anchored_vwap import (  # noqa: E501
    OUTCOMES_BY_RESPONSE_WINDOW,
    SEPARATION_HOURS_BY_RESPONSE_WINDOW,
    attach_state_and_absolute_outcomes,
    stable_json_sha256,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_density_reaction import (  # noqa: E501
    CORE_STATE_FEATURES,
    cohort_period_results,
    leave_one_coin_out_results,
    pair_period_results,
    purge_density_matches,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_density_reaction import (  # noqa: E501
    REPORT_ROOT as G2E_REPORT_ROOT,
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
REPORT_ROOT = OUTPUT_ROOT / "generation3_branches" / "g3e_participation_origin_reaction"
ARTIFACT_ROOT = (
    LARGE_ARTIFACT_ROOT
    / "generation3_branches"
    / "g3e_participation_origin_reaction"
)
OUTPUT_SCHEMA_VERSION = 2
ACTUAL_KIND = "contact"
CONTROL_KIND = "near_miss"
CONTROL_NAME = "clean_near_miss"
ORIGIN_MEMORY_HOURS = 168
ORIGIN_STATE_FEATURES = (
    "state_g3e_zone_half_width_atr",
    "state_g3e_origin_age_fraction",
    "state_g3e_origin_volume_vs_prior_median",
    "state_g3e_origin_range_atr",
    "state_g3e_origin_abs_pressure",
    "state_g3e_origin_pressure_high",
    "state_g3e_origin_atr_high",
    "state_g3e_origin_impulse_sign",
    "state_g3e_density_level_count",
    "state_g3e_reference_level_count",
)
MATCH_STATE_FEATURES = (*CORE_STATE_FEATURES, *ORIGIN_STATE_FEATURES)


@dataclass(frozen=True)
class PairTask:
    pair: str
    manifest_path: str
    context_path: str
    context_sha256: str
    geometry_path: str
    geometry_sha256: str
    supported_scopes: tuple[str, ...]
    run_id: str
    request_sha256: str
    overwrite: bool


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 3E direction-neutral comparison of later participation-origin "
            "contacts with clean near misses."
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
    preflight_record_path = preflight_dir / "g3e_preflight_run_record.json"
    preflight_integrity_path = preflight_dir / "g3e_preflight_integrity.json"
    support_path = preflight_dir / "g3e_comparison_support.parquet"
    g2e_record_path = g2e_dir / "g2e_reaction_run_record.json"
    g2e_integrity_path = g2e_dir / "g2e_reaction_integrity.json"
    context_path = g2e_dir / "g2e_causal_market_context.parquet"
    supported_scopes = validate_sources_and_supported_scopes(
        cohort=args.cohort,
        preflight_record_path=preflight_record_path,
        preflight_integrity_path=preflight_integrity_path,
        support_path=support_path,
        g2e_record_path=g2e_record_path,
        g2e_integrity_path=g2e_integrity_path,
        context_path=context_path,
    )
    context_sha256 = sha256_file(context_path)
    pair_contracts = []
    for pair in pairs:
        geometry_path = (
            preflight_artifact_dir / "pair_geometry" / f"{pair_stem(pair)}.parquet"
        )
        pair_contracts.append(
            {
                "pair": pair,
                "geometry_path": str(geometry_path),
                "geometry_sha256": sha256_file(geometry_path),
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
        "g2e_context_run_id": args.g2e_run_id,
        "context_path": str(context_path),
        "context_sha256": context_sha256,
        "pair_geometry_contracts": pair_contracts,
        "supported_scopes_opened": list(supported_scopes),
        "actual_event_kind": ACTUAL_KIND,
        "control_event_kind": CONTROL_KIND,
        "matching": {
            "same_pair_period_scope_and_approach": True,
            "origin_impulse_sign_handling": (
                "balance it as a prior-state feature; do not split small event pools by it"
            ),
            "prior_state_features": list(MATCH_STATE_FEATURES),
            "geometry": (
                "previous-close distance from the outer zone edge, matched within "
                "0.10 ATR"
            ),
            "minimum_event_separation": (
                "greater than the 1h, 4h, or 24h response horizon before final path purge"
            ),
        },
        "outcomes": (
            "encounter-candle range and volume plus absolute movement, range, volume, "
            "absolute pressure change, dwell, and crossings over 1h, 4h, and 24h"
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
    record_path = run_dir / "g3e_reaction_run_record.json"
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
            "The outcome-blind preflight found repeated later contacts and clean near "
            "misses for fixed participation-origin zones in both market groups."
        ),
        "hypothesis": (
            "After matching the approach and preceding market state, touching a remembered "
            "participation-origin area produces more direction-neutral activity than "
            "approaching the same kind of area without touching it."
        ),
        "pass_fail": (
            "Retain only a named response with usable state balance, at least five coins "
            "and 50 independent event pairs, the same sign in both validation periods, "
            "and leave-one-coin-out sign stability. A contact/near-miss difference alone "
            "does not prove unique origin-level attribution."
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
            context_path=str(context_path),
            context_sha256=context_sha256,
            geometry_path=row["geometry_path"],
            geometry_sha256=row["geometry_sha256"],
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
        atomic_write_parquet(inventory, run_dir / "g3e_reaction_pair_inventory.parquet")
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
            artifact_dir / "g3e_independent_contact_near_miss_pairs.parquet",
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
        atomic_write_json(integrity, run_dir / "g3e_reaction_integrity.json")
        if not integrity["passed"]:
            raise RuntimeError("G3E contact/near-miss integrity validation failed.")
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
                "integrity": str(run_dir / "g3e_reaction_integrity.json"),
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


def validate_sources_and_supported_scopes(
    *,
    cohort: str,
    preflight_record_path: Path,
    preflight_integrity_path: Path,
    support_path: Path,
    g2e_record_path: Path,
    g2e_integrity_path: Path,
    context_path: Path,
) -> tuple[str, ...]:
    preflight_record = json.loads(preflight_record_path.read_text(encoding="utf-8"))
    preflight_integrity = json.loads(preflight_integrity_path.read_text(encoding="utf-8"))
    g2e_record = json.loads(g2e_record_path.read_text(encoding="utf-8"))
    g2e_integrity = json.loads(g2e_integrity_path.read_text(encoding="utf-8"))
    if preflight_record.get("status") != "completed":
        raise ValueError("G3E preflight is not completed.")
    if preflight_record.get("request_contract", {}).get("cohort") != cohort:
        raise ValueError("G3E preflight cohort does not match this reaction run.")
    if preflight_integrity.get("passed") is not True:
        raise ValueError("G3E preflight integrity did not pass.")
    if preflight_integrity.get("reaction_outcomes_loaded") is not False:
        raise ValueError("G3E preflight unexpectedly opened reaction outcomes.")
    if g2e_record.get("status") != "completed" or g2e_integrity.get("passed") is not True:
        raise ValueError("The causal market-context source is not a completed valid run.")
    if not context_path.is_file():
        raise ValueError(f"Missing causal market-context table: {context_path}")
    for source in (preflight_record, preflight_integrity, g2e_record, g2e_integrity):
        if source.get("direction_prediction") is not False:
            raise ValueError("A source violates direction_prediction=False.")
        if source.get("profit_optimization") is not False:
            raise ValueError("A source violates profit_optimization=False.")
    support = pd.read_parquet(support_path)
    supported = support.loc[
        support["both_validation_periods_supported"].fillna(False), "origin_scope"
    ]
    scopes = tuple(sorted(supported.astype(str).unique()))
    if not scopes:
        raise ValueError("G3E preflight exposed no scope supported in both validations.")
    return scopes


def filter_supported_scopes(geometry: DataFrame, scopes: Sequence[str]) -> DataFrame:
    allowed = set(scopes)
    output = geometry.loc[
        geometry["origin_scope"].astype(str).isin(allowed)
        & geometry["event_kind"].isin((ACTUAL_KIND, CONTROL_KIND))
    ].copy()
    return output.reset_index(drop=True)


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
    context_path = Path(task.context_path)
    if sha256_file(geometry_path) != task.geometry_sha256:
        raise ValueError(f"G3E preflight geometry changed: {geometry_path}")
    if sha256_file(context_path) != task.context_sha256:
        raise ValueError(f"G2E causal market context changed: {context_path}")
    geometry = filter_supported_scopes(pd.read_parquet(geometry_path), task.supported_scopes)
    if geometry.empty:
        raise ValueError(f"No supported G3E geometry for {task.pair}.")
    manifest = load_manifest(Path(task.manifest_path))
    base = prepare_base_market_frame(task.pair, manifest)
    context = pd.read_parquet(context_path)
    state = causal_local_state(base).merge(context, on="date", how="left", validate="one_to_one")
    paths = future_path_matrices(base, max_horizon=24)
    events = attach_origin_event_outcomes(
        pair=task.pair,
        base=base,
        state=state,
        paths=paths,
        geometry=geometry,
    )
    rows, audit = direct_contact_near_miss_rows(events, pair=task.pair)
    matches = DataFrame(rows)
    if matches.empty:
        raise ValueError(f"No matchable G3E contact/near-miss pairs for {task.pair}.")
    independent = purge_density_matches(matches)
    if independent.empty:
        raise ValueError(f"No independent G3E contact/near-miss pairs for {task.pair}.")
    independent["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    independent["run_request_sha256"] = task.request_sha256
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(independent, output_path)
    return {
        "pair": task.pair,
        "status": "completed",
        "supported_geometry_events": len(geometry),
        "outcome_event_rows": len(events),
        "matched_comparisons_before_overlap_purge": len(matches),
        "independent_comparisons": len(independent),
        "eligible_contact_events": audit["eligible_actual_events"],
        "state_matchable_contact_events": audit["state_matchable_actual_events"],
        "seconds": round(time.perf_counter() - started, 3),
    }


def outside_edge_gap_atr(frame: DataFrame) -> np.ndarray:
    distance = pd.to_numeric(frame["pre_distance_atr"], errors="coerce").to_numpy(dtype=float)
    width = pd.to_numeric(frame["zone_half_width_atr"], errors="coerce").to_numpy(dtype=float)
    return np.maximum(distance - width, 0.0)


def attach_origin_event_outcomes(
    *,
    pair: str,
    base: DataFrame,
    state: DataFrame,
    paths: dict[str, np.ndarray],
    geometry: DataFrame,
) -> DataFrame:
    source = geometry.sort_values("base_index", kind="stable").reset_index(drop=True)
    indexes = source["base_index"].to_numpy(dtype=np.int64)
    spec = LevelSpec(
        name="participation_origin_168h",
        family="participation_origin",
        batch="g3e",
        column="g3e_participation_origin",
    )
    event_time = pd.to_datetime(source["event_time"], utc=True).reset_index(drop=True)
    events = event_frame(
        merged=base,
        paths=paths,
        indexes=indexes,
        levels=source["level_price"].to_numpy(dtype=float),
        widths=source["zone_half_width"].to_numpy(dtype=float),
        spec=spec,
        pair=pair,
        timeframe="1h_causal_participation_origin",
        control="g3e_contact_vs_near_miss",
        zone_method="completed_impulse_body_zone",
        source_available=pd.to_datetime(source["origin_available_at"], utc=True),
        source_open=pd.to_datetime(source["origin_time"], utc=True),
        horizons=(1, 4, 24),
        match_tier=None,
    )
    if not events["base_index"].equals(source["base_index"]):
        raise ValueError("G3E outcome reconstruction changed event order.")
    if not pd.to_datetime(events["event_time"], utc=True).equals(event_time):
        raise ValueError("G3E outcome reconstruction changed event timestamps.")
    reconstructed_approach = events["approach_state"].astype(str).to_numpy()
    source_approach = source["approach_state"].astype(str).to_numpy()
    if not np.array_equal(reconstructed_approach, source_approach):
        raise ValueError("G3E outcome reconstruction changed the causal approach side.")
    copy_columns = (
        "event_kind",
        "origin_scope",
        "origin_index",
        "origin_age_hours",
        "origin_volume_vs_prior_median",
        "origin_volume_vs_q90",
        "origin_range_atr",
        "origin_range_vs_q75",
        "origin_abs_pressure",
        "origin_pressure_above_q75",
        "origin_atr_above_q75",
        "origin_impulse_sign",
        "overlap_density_zone_count",
        "overlap_other_density_surface_count",
        "overlap_reference_level_count",
        "overlap_reference_group_count",
    )
    for column in copy_columns:
        events[column] = source[column].to_numpy()
    events = attach_state_and_absolute_outcomes(events, state)
    events["outside_edge_gap_atr"] = outside_edge_gap_atr(events)
    events["state_g3e_zone_half_width_atr"] = pd.to_numeric(
        events["zone_half_width_atr"], errors="coerce"
    )
    events["state_g3e_origin_age_fraction"] = (
        pd.to_numeric(events["origin_age_hours"], errors="coerce") / ORIGIN_MEMORY_HOURS
    )
    events["state_g3e_origin_volume_vs_prior_median"] = pd.to_numeric(
        events["origin_volume_vs_prior_median"], errors="coerce"
    )
    events["state_g3e_origin_range_atr"] = pd.to_numeric(
        events["origin_range_atr"], errors="coerce"
    )
    events["state_g3e_origin_abs_pressure"] = pd.to_numeric(
        events["origin_abs_pressure"], errors="coerce"
    )
    events["state_g3e_origin_pressure_high"] = events[
        "origin_pressure_above_q75"
    ].astype(float)
    events["state_g3e_origin_atr_high"] = events["origin_atr_above_q75"].astype(float)
    events["state_g3e_origin_impulse_sign"] = pd.to_numeric(
        events["origin_impulse_sign"], errors="coerce"
    )
    events["state_g3e_density_level_count"] = pd.to_numeric(
        events["overlap_density_zone_count"], errors="coerce"
    )
    events["state_g3e_reference_level_count"] = pd.to_numeric(
        events["overlap_reference_level_count"], errors="coerce"
    )
    return events


def direct_contact_near_miss_rows(
    events: DataFrame,
    *,
    pair: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    audit_totals = {"eligible_actual_events": 0, "state_matchable_actual_events": 0}
    cell_columns = ["origin_scope", "period"]
    for key, cell in events.groupby(cell_columns, observed=True):
        contacts = cell.loc[cell["event_kind"].eq(ACTUAL_KIND)].reset_index(drop=True)
        misses = cell.loc[cell["event_kind"].eq(CONTROL_KIND)].reset_index(drop=True)
        if contacts.empty or misses.empty:
            continue
        for approach in ("from_below", "from_above"):
            left = contacts.loc[contacts["approach_state"].eq(approach)].copy()
            right = misses.loc[misses["approach_state"].eq(approach)].copy()
            if left.empty or right.empty:
                continue
            left["raw_pre_distance_atr"] = left["pre_distance_atr"]
            right["raw_pre_distance_atr"] = right["pre_distance_atr"]
            left["pre_distance_atr"] = left["outside_edge_gap_atr"]
            right["pre_distance_atr"] = right["outside_edge_gap_atr"]
            for response_window, outcomes in OUTCOMES_BY_RESPONSE_WINDOW.items():
                independence_hours = SEPARATION_HOURS_BY_RESPONSE_WINDOW[response_window]
                pairs, audit = nearest_state_pairs(
                    left,
                    right,
                    state_columns=MATCH_STATE_FEATURES,
                    pre_distance_atr_caliper=0.10,
                    minimum_event_separation_hours=independence_hours,
                )
                audit_totals["eligible_actual_events"] += int(audit["eligible_actual"])
                audit_totals["state_matchable_actual_events"] += int(
                    audit["state_matchable_actual"]
                )
                for left_position, right_position, distance in pairs:
                    actual = left.iloc[left_position]
                    control = right.iloc[right_position]
                    separation = abs(
                        (
                            pd.Timestamp(actual["event_time"])
                            - pd.Timestamp(control["event_time"])
                        ).total_seconds()
                        / 3600.0
                    )
                    row: dict[str, Any] = {
                        "pair": pair,
                        "route_id": "g3e_participation_origin_contact_vs_near_miss",
                        "density_family": "participation_origin",
                        "history_hours": ORIGIN_MEMORY_HOURS,
                        "level_name": "participation_origin__168h",
                        "actual_scope": key[0],
                        "control": CONTROL_NAME,
                        "period": key[1],
                        "approach_state": approach,
                        "response_window": response_window,
                        "actual_origin_impulse_sign": int(actual["origin_impulse_sign"]),
                        "control_origin_impulse_sign": int(control["origin_impulse_sign"]),
                        "actual_event_kind": ACTUAL_KIND,
                        "control_event_kind": CONTROL_KIND,
                        "actual_event_time": actual["event_time"],
                        "control_event_time": control["event_time"],
                        "actual_base_index": int(actual["base_index"]),
                        "control_base_index": int(control["base_index"]),
                        "actual_source_available_at": actual["source_available_at"],
                        "control_source_available_at": control["source_available_at"],
                        "actual_source_open": actual["source_open"],
                        "control_source_open": control["source_open"],
                        "actual_zone_rank": -1,
                        "actual_zone_support_fraction": np.nan,
                        "match_distance": float(distance),
                        "pre_distance_atr_abs_difference": abs(
                            float(actual["outside_edge_gap_atr"])
                            - float(control["outside_edge_gap_atr"])
                        ),
                        "actual_raw_pre_distance_atr": float(
                            actual["raw_pre_distance_atr"]
                        ),
                        "control_raw_pre_distance_atr": float(
                            control["raw_pre_distance_atr"]
                        ),
                        "event_separation_hours": separation,
                        "actual_overlap_density_zone_count": int(
                            actual["overlap_density_zone_count"]
                        ),
                        "control_overlap_density_zone_count": int(
                            control["overlap_density_zone_count"]
                        ),
                        "actual_overlap_other_density_surface_count": int(
                            actual["overlap_other_density_surface_count"]
                        ),
                        "control_overlap_other_density_surface_count": int(
                            control["overlap_other_density_surface_count"]
                        ),
                        "actual_overlap_reference_level_count": int(
                            actual["overlap_reference_level_count"]
                        ),
                        "control_overlap_reference_level_count": int(
                            control["overlap_reference_level_count"]
                        ),
                        "matching_state_features": ";".join(MATCH_STATE_FEATURES),
                        "eligible_actual_events": int(audit["eligible_actual"]),
                        "eligible_control_events": int(audit["eligible_control"]),
                        "geometry_eligible_actual_events": int(
                            audit["geometry_eligible_actual"]
                        ),
                        "state_matchable_actual_events": int(audit["state_matchable_actual"]),
                        "direction_prediction": False,
                        "profit_optimization": False,
                    }
                    for outcome in outcomes:
                        actual_value = float(actual[outcome])
                        control_value = float(control[outcome])
                        row[f"actual__{outcome}"] = actual_value
                        row[f"control__{outcome}"] = control_value
                        row[f"delta__{outcome}"] = actual_value - control_value
                    for feature in MATCH_STATE_FEATURES:
                        row[f"actual_state__{feature}"] = float(actual[feature])
                        row[f"control_state__{feature}"] = float(control[feature])
                    rows.append(row)
    return rows, audit_totals


def integrity_record(
    *,
    results: Sequence[dict[str, Any]],
    independent: DataFrame,
    supported_scopes: Sequence[str],
    requested_pairs: Sequence[str],
) -> dict[str, Any]:
    failures = [row for row in results if row["status"] == "failed"]
    observed_scopes = set(independent["actual_scope"].astype(str))
    invalid_scopes = sorted(observed_scopes.difference(supported_scopes))
    missing_pairs = sorted(set(requested_pairs).difference(independent["pair"].astype(str)))
    separation = pd.to_numeric(independent["event_separation_hours"], errors="coerce")
    required = independent["response_window"].map(SEPARATION_HOURS_BY_RESPONSE_WINDOW)
    separation_violations = int((separation <= required).sum())
    geometry_difference = pd.to_numeric(
        independent["pre_distance_atr_abs_difference"], errors="coerce"
    )
    actual_availability = pd.to_datetime(
        independent["actual_source_available_at"], utc=True, errors="coerce"
    )
    control_availability = pd.to_datetime(
        independent["control_source_available_at"], utc=True, errors="coerce"
    )
    actual_times = pd.to_datetime(independent["actual_event_time"], utc=True, errors="coerce")
    control_times = pd.to_datetime(independent["control_event_time"], utc=True, errors="coerce")
    future_source_violations = int(
        (actual_availability > actual_times).sum() + (control_availability > control_times).sum()
    )
    actual_kind_violations = int(independent["actual_event_kind"].ne(ACTUAL_KIND).sum())
    control_kind_violations = int(independent["control_event_kind"].ne(CONTROL_KIND).sum())
    direction_violations = int(independent["direction_prediction"].ne(False).sum())
    profit_violations = int(independent["profit_optimization"].ne(False).sum())
    return {
        "created_at_utc": utc_now(),
        "passed": bool(
            not failures
            and not independent.empty
            and not invalid_scopes
            and not missing_pairs
            and separation_violations == 0
            and geometry_difference.le(0.1000001).all()
            and future_source_violations == 0
            and actual_kind_violations == 0
            and control_kind_violations == 0
            and direction_violations == 0
            and profit_violations == 0
        ),
        "failures": failures,
        "independent_comparisons": len(independent),
        "requested_pairs": list(requested_pairs),
        "observed_pairs": sorted(independent["pair"].astype(str).unique()),
        "missing_pairs": missing_pairs,
        "supported_scopes": list(supported_scopes),
        "observed_scopes": sorted(observed_scopes),
        "invalid_scopes": invalid_scopes,
        "controls_present": sorted(independent["control"].astype(str).unique()),
        "matching_state_features": list(MATCH_STATE_FEATURES),
        "event_separation_violations": separation_violations,
        "pre_distance_atr_abs_difference_max": float(geometry_difference.max()),
        "future_source_violations": future_source_violations,
        "actual_event_kind_violations": actual_kind_violations,
        "control_event_kind_violations": control_kind_violations,
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

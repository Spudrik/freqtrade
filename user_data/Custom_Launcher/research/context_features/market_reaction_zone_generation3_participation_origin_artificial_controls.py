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
    future_path_matrices,
    load_manifest,
    normalize_dates,
    numeric_array,
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
    stable_json_sha256,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_density_reaction import (  # noqa: E501
    REPORT_ROOT as G2E_REPORT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_density_reaction import (  # noqa: E501
    attach_structure_overlaps,
    build_density_surfaces,
    cohort_period_results,
    density_matrices,
    leave_one_coin_out_results,
    pair_period_results,
    purge_density_matches,
    reference_matrices,
    selected_reference_levels,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_localization import (  # noqa: E501
    eligible_period_events,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_lvn_attribution import (  # noqa: E501
    select_pairs,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_participation_origin import (  # noqa: E501
    ARTIFACT_ROOT as G3E_PREFLIGHT_ARTIFACT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_participation_origin import (  # noqa: E501
    CONTEXT_QUANTILE,
    MAX_ZONE_HALF_WIDTH_ATR,
    MIN_ZONE_HALF_WIDTH_ATR,
    ORIGIN_MEMORY_HOURS,
    RANGE_QUANTILE,
    VOLUME_QUANTILE,
    classify_origin_scope,
    origin_revisit_events,
    participation_origins,
    prior_rolling_quantile,
    validate_frozen_branch,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_participation_origin import (  # noqa: E501
    REPORT_ROOT as G3E_PREFLIGHT_REPORT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_participation_origin_reaction import (  # noqa: E501
    MATCH_STATE_FEATURES,
    attach_origin_event_outcomes,
    outside_edge_gap_atr,
    validate_sources_and_supported_scopes,
)


FROZEN_BATCH = OUTPUT_ROOT / "generation2_review" / "g3_frozen_branch_batch.json"
REPORT_ROOT = OUTPUT_ROOT / "generation3_branches" / "g3e_participation_origin_artificial"
ARTIFACT_ROOT = LARGE_ARTIFACT_ROOT / "generation3_branches" / "g3e_participation_origin_artificial"
OUTPUT_SCHEMA_VERSION = 1
SHIFT_CONTROLS = (-1.0, 1.0)
SHUFFLE_FRACTION = 1 / 3
STALE_OFFSET_HOURS = ORIGIN_MEMORY_HOURS
CONTROL_NAMES = (
    "origin_shift_-1atr",
    "origin_shift_+1atr",
    "shuffled_origin_source",
    "stale_origin_169_336h",
)
STALE_CONTROL = "stale_origin_169_336h"


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
            "Generation 3E direction-neutral comparison of real participation-origin "
            "contacts with shifted, shuffled-source, and stale-origin contacts."
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
        geometry_path = preflight_artifact_dir / "pair_geometry" / f"{pair_stem(pair)}.parquet"
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
        "context_path": str(context_path),
        "context_sha256": context_sha256,
        "pair_geometry_contracts": pair_contracts,
        "supported_scopes_opened": list(supported_scopes),
        "controls": list(CONTROL_NAMES),
        "control_construction": {
            "origin_shift": (
                "Move each real origin centre by exactly plus or minus one origin-time ATR; "
                "keep its source properties and run the same first-revisit rule."
            ),
            "shuffled_origin_source": (
                "Move the source identity by one third of each fixed chronological period, "
                "construct the ordinary body zone at that timestamp, and exclude real "
                "participation-origin timestamps."
            ),
            "stale_origin": (
                "Keep the real source zone but restart the six-candle outside test only "
                "after its 168-hour memory expires; search ages 169 through 336 hours."
            ),
        },
        "matching": (
            "Same pair, period, origin arrangement, approach side and source-body sign; "
            "prior market and source state plus outer-edge distance within 0.10 ATR. "
            "Age is deliberately omitted only for the stale control."
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
    record_path = run_dir / "g3e_artificial_run_record.json"
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
            "Near misses showed a narrow normal three-way local-churn lead, but the "
            "existing component ladder did not retain an origin-specific activity claim."
        ),
        "hypothesis": (
            "The exact causal origin location and its unexpired memory add a repeatable "
            "direction-neutral response beyond artificial source locations."
        ),
        "pass_fail": (
            "Retain only a named response with fair state balance, at least five coins and "
            "50 independent pairs, the same sign in both validations, and leave-one-coin-"
            "out sign stability against every supported artificial control."
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
        atomic_write_parquet(inventory, run_dir / "g3e_artificial_pair_inventory.parquet")
        failures = inventory.loc[inventory["status"].eq("failed")]
        if not failures.empty:
            raise RuntimeError(f"{len(failures)} pair task(s) failed; inspect the pair inventory.")
        independent = combine_pair_outputs(
            pair_dir=pair_dir,
            pairs=pairs,
            request_sha256=request_sha256,
        )
        atomic_write_parquet(
            independent,
            artifact_dir / "g3e_artificial_independent_pairs.parquet",
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
        atomic_write_json(integrity, run_dir / "g3e_artificial_integrity.json")
        if not integrity["passed"]:
            raise RuntimeError("G3E artificial-control integrity validation failed.")
        eligible_count = (
            int(cohort_period["evidence_eligible"].sum())
            if "evidence_eligible" in cohort_period
            else 0
        )
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "pair_tasks": results,
                "independent_comparisons": len(independent),
                "pair_period_rows": len(pair_period),
                "cohort_period_rows": len(cohort_period),
                "leave_one_coin_out_rows": len(leave_one_out),
                "evidence_eligible_rows": eligible_count,
                "integrity": str(run_dir / "g3e_artificial_integrity.json"),
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
    geometry_path = Path(task.geometry_path)
    context_path = Path(task.context_path)
    if sha256_file(geometry_path) != task.geometry_sha256:
        raise ValueError(f"G3E preflight geometry changed: {geometry_path}")
    if sha256_file(context_path) != task.context_sha256:
        raise ValueError(f"G2E causal market context changed: {context_path}")
    geometry = pd.read_parquet(geometry_path)
    actual_geometry = geometry.loc[
        geometry["event_kind"].eq("contact") & geometry["origin_scope"].isin(task.supported_scopes)
    ].reset_index(drop=True)
    if actual_geometry.empty:
        raise ValueError(f"No supported G3E contact geometry for {task.pair}.")
    manifest_path = Path(task.manifest_path)
    manifest = load_manifest(manifest_path)
    base = prepare_base_market_frame(task.pair, manifest)
    context = pd.read_parquet(context_path)
    state = causal_local_state(base).merge(context, on="date", how="left", validate="one_to_one")
    paths = future_path_matrices(base, max_horizon=24)
    surfaces = build_density_surfaces(base)
    references = selected_reference_levels(
        pair=task.pair,
        base=base,
        manifest_path=manifest_path,
    )
    density_levels, density_widths, density_surface_keys = density_matrices(surfaces)
    reference_levels, reference_widths, reference_groups = reference_matrices(
        references,
        base_atr=numeric_array(base["base_atr"]),
    )
    actual = attach_origin_event_outcomes(
        pair=task.pair,
        base=base,
        state=state,
        paths=paths,
        geometry=actual_geometry,
    )
    origins = participation_origins(base)
    control_geometry = artificial_control_geometry(
        pair=task.pair,
        base=base,
        origins=origins,
        manifest=manifest,
        density_levels=density_levels,
        density_widths=density_widths,
        density_surface_keys=density_surface_keys,
        reference_levels=reference_levels,
        reference_widths=reference_widths,
        reference_groups=reference_groups,
    )
    if control_geometry.empty:
        raise ValueError(f"No G3E artificial control geometry for {task.pair}.")
    controls = attach_control_outcomes(
        pair=task.pair,
        base=base,
        state=state,
        paths=paths,
        geometry=control_geometry,
    )
    rows, audit = artificial_match_rows(actual=actual, controls=controls, pair=task.pair)
    matches = DataFrame(rows)
    if matches.empty:
        raise ValueError(f"No matchable G3E artificial controls for {task.pair}.")
    independent = purge_density_matches(matches)
    if independent.empty:
        raise ValueError(f"No independent G3E artificial controls for {task.pair}.")
    independent["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    independent["run_request_sha256"] = task.request_sha256
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(independent, output_path)
    return {
        "pair": task.pair,
        "status": "completed",
        "actual_contact_events": len(actual),
        "artificial_control_events": len(controls),
        "matched_comparisons_before_overlap_purge": len(matches),
        "independent_comparisons": len(independent),
        "controls": ";".join(sorted(independent["control"].astype(str).unique())),
        "eligible_actual_events": audit["eligible_actual_events"],
        "state_matchable_actual_events": audit["state_matchable_actual_events"],
        "seconds": round(time.perf_counter() - started, 3),
    }


def shifted_origin_indexes(
    base: DataFrame,
    origins: DataFrame,
) -> np.ndarray:
    selected = set(origins["origin_index"].astype(int))
    used: set[int] = set()
    targets: list[int] = []
    periods = base["period"].astype(str).to_numpy()
    for period in sorted(set(periods)):
        period_indexes = np.flatnonzero(periods == period)
        if not len(period_indexes):
            continue
        position_by_index = {int(index): position for position, index in enumerate(period_indexes)}
        source_indexes = [index for index in sorted(selected) if periods[index] == period]
        shift = max(1, round(len(period_indexes) * SHUFFLE_FRACTION))
        for source_index in source_indexes:
            source_position = position_by_index[source_index]
            for extra in range(len(period_indexes)):
                candidate = int(
                    period_indexes[(source_position + shift + extra) % len(period_indexes)]
                )
                if candidate in selected or candidate in used:
                    continue
                used.add(candidate)
                targets.append(candidate)
                break
    return np.asarray(sorted(targets), dtype=np.int64)


def origin_rows_for_indexes(base: DataFrame, indexes: np.ndarray) -> DataFrame:
    if not len(indexes):
        return DataFrame()
    open_price = numeric_array(base["open"])
    high = numeric_array(base["high"])
    low = numeric_array(base["low"])
    close = numeric_array(base["close"])
    volume = numeric_array(base["volume"])
    atr = numeric_array(base["base_atr"])
    candle_range = high - low
    range_atr = np.divide(
        candle_range,
        atr,
        out=np.full(len(base), np.nan),
        where=np.isfinite(atr) & (atr > 0.0),
    )
    pressure = np.divide(
        np.abs(close - open_price),
        candle_range,
        out=np.zeros(len(base)),
        where=np.isfinite(candle_range) & (candle_range > 0.0),
    )
    volume_median = prior_rolling_quantile(volume, 0.50)
    volume_q90 = prior_rolling_quantile(volume, VOLUME_QUANTILE)
    range_q75 = prior_rolling_quantile(range_atr, RANGE_QUANTILE)
    pressure_q75 = prior_rolling_quantile(pressure, CONTEXT_QUANTILE)
    atr_q75 = prior_rolling_quantile(atr, CONTEXT_QUANTILE)
    valid = (
        np.isfinite(open_price[indexes])
        & np.isfinite(close[indexes])
        & np.isfinite(atr[indexes])
        & (atr[indexes] > 0.0)
        & np.isfinite(volume_median[indexes])
        & (volume_median[indexes] > 0.0)
        & np.isfinite(volume_q90[indexes])
        & (volume_q90[indexes] > 0.0)
        & np.isfinite(range_q75[indexes])
        & (range_q75[indexes] > 0.0)
        & np.isfinite(pressure_q75[indexes])
        & np.isfinite(atr_q75[indexes])
    )
    indexes = indexes[valid]
    if not len(indexes):
        return DataFrame()
    width = np.clip(
        np.abs(close[indexes] - open_price[indexes]) / 2.0,
        MIN_ZONE_HALF_WIDTH_ATR * atr[indexes],
        MAX_ZONE_HALF_WIDTH_ATR * atr[indexes],
    )
    dates = normalize_dates(base["date"])
    origin_time = dates.iloc[indexes].reset_index(drop=True)
    return DataFrame(
        {
            "origin_index": indexes,
            "origin_time": origin_time,
            "origin_available_at": origin_time + pd.Timedelta(hours=1),
            "level_price": (open_price[indexes] + close[indexes]) / 2.0,
            "zone_half_width": width,
            "zone_half_width_atr": width / atr[indexes],
            "origin_volume_vs_prior_median": volume[indexes] / volume_median[indexes],
            "origin_volume_vs_q90": volume[indexes] / volume_q90[indexes],
            "origin_range_atr": range_atr[indexes],
            "origin_range_vs_q75": range_atr[indexes] / range_q75[indexes],
            "origin_abs_pressure": pressure[indexes],
            "origin_pressure_above_q75": pressure[indexes] >= pressure_q75[indexes],
            "origin_atr_above_q75": atr[indexes] >= atr_q75[indexes],
            "origin_impulse_sign": np.sign(close[indexes] - open_price[indexes]).astype(np.int8),
        }
    )


def artificial_control_geometry(
    *,
    pair: str,
    base: DataFrame,
    origins: DataFrame,
    manifest: dict[str, Any],
    density_levels: np.ndarray,
    density_widths: np.ndarray,
    density_surface_keys: np.ndarray,
    reference_levels: np.ndarray,
    reference_widths: np.ndarray,
    reference_groups: np.ndarray,
) -> DataFrame:
    controls: list[DataFrame] = []
    origin_atr = numeric_array(base["base_atr"])[origins["origin_index"].to_numpy(dtype=int)]
    for shift in SHIFT_CONTROLS:
        shifted = origins.copy()
        shifted["level_price"] = shifted["level_price"].to_numpy(dtype=float) + shift * origin_atr
        events = origin_revisit_events(pair=pair, base=base, origins=shifted)
        if not events.empty:
            events["artificial_control"] = f"origin_shift_{shift:+g}atr"
            controls.append(events)
    shuffled = origin_rows_for_indexes(base, shifted_origin_indexes(base, origins))
    events = origin_revisit_events(pair=pair, base=base, origins=shuffled)
    if not events.empty:
        events["artificial_control"] = "shuffled_origin_source"
        controls.append(events)
    stale = origins.loc[
        origins["origin_index"].to_numpy(dtype=int) + STALE_OFFSET_HOURS < len(base)
    ].copy()
    stale["origin_index"] = stale["origin_index"].astype(int) + STALE_OFFSET_HOURS
    events = origin_revisit_events(pair=pair, base=base, origins=stale)
    if not events.empty:
        events["origin_index"] = events["origin_index"].astype(int) - STALE_OFFSET_HOURS
        events["origin_age_hours"] = events["origin_age_hours"].astype(int) + STALE_OFFSET_HOURS
        events["artificial_control"] = STALE_CONTROL
        controls.append(events)
    if not controls:
        return DataFrame()
    output = pd.concat(controls, ignore_index=True)
    output = eligible_period_events(output, manifest, embargo_hours=24)
    output = output.loc[output["event_kind"].eq("contact")].reset_index(drop=True)
    if output.empty:
        return output
    output = attach_structure_overlaps(
        output,
        source_surface_key=None,
        density_levels=density_levels,
        density_widths=density_widths,
        density_surface_keys=density_surface_keys,
        reference_levels=reference_levels,
        reference_widths=reference_widths,
        reference_groups=reference_groups,
    )
    output["origin_scope"] = classify_origin_scope(output)
    return output


def attach_control_outcomes(
    *,
    pair: str,
    base: DataFrame,
    state: DataFrame,
    paths: dict[str, np.ndarray],
    geometry: DataFrame,
) -> DataFrame:
    frames: list[DataFrame] = []
    for control_name, source in geometry.groupby("artificial_control", observed=True):
        events = attach_origin_event_outcomes(
            pair=pair,
            base=base,
            state=state,
            paths=paths,
            geometry=source,
        )
        events["artificial_control"] = str(control_name)
        frames.append(events)
    return pd.concat(frames, ignore_index=True) if frames else DataFrame()


def artificial_match_rows(
    *,
    actual: DataFrame,
    controls: DataFrame,
    pair: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    audit_totals = {"eligible_actual_events": 0, "state_matchable_actual_events": 0}
    cell_columns = ["origin_scope", "period", "origin_impulse_sign"]
    for key, left_cell in actual.groupby(cell_columns, observed=True):
        for control_name in CONTROL_NAMES:
            right_cell = controls.loc[
                controls["artificial_control"].eq(control_name)
                & controls["origin_scope"].eq(key[0])
                & controls["period"].eq(key[1])
                & controls["origin_impulse_sign"].eq(key[2])
            ]
            if right_cell.empty:
                continue
            state_features = tuple(
                feature
                for feature in MATCH_STATE_FEATURES
                if control_name != STALE_CONTROL or feature != "state_g3e_origin_age_fraction"
            )
            for approach in ("from_below", "from_above"):
                left = left_cell.loc[left_cell["approach_state"].eq(approach)].copy()
                right = right_cell.loc[right_cell["approach_state"].eq(approach)].copy()
                if left.empty or right.empty:
                    continue
                left["raw_pre_distance_atr"] = left["pre_distance_atr"]
                right["raw_pre_distance_atr"] = right["pre_distance_atr"]
                left["pre_distance_atr"] = outside_edge_gap_atr(left)
                right["pre_distance_atr"] = outside_edge_gap_atr(right)
                for response_window, outcomes in OUTCOMES_BY_RESPONSE_WINDOW.items():
                    independence_hours = SEPARATION_HOURS_BY_RESPONSE_WINDOW[response_window]
                    pairs, audit = nearest_state_pairs(
                        left,
                        right,
                        state_columns=state_features,
                        pre_distance_atr_caliper=0.10,
                        minimum_event_separation_hours=independence_hours,
                    )
                    audit_totals["eligible_actual_events"] += int(audit["eligible_actual"])
                    audit_totals["state_matchable_actual_events"] += int(
                        audit["state_matchable_actual"]
                    )
                    for left_position, right_position, distance in pairs:
                        actual_row = left.iloc[left_position]
                        control_row = right.iloc[right_position]
                        separation = abs(
                            (
                                pd.Timestamp(actual_row["event_time"])
                                - pd.Timestamp(control_row["event_time"])
                            ).total_seconds()
                            / 3600.0
                        )
                        row: dict[str, Any] = {
                            "pair": pair,
                            "route_id": "g3e_real_origin_vs_artificial_origin",
                            "density_family": "participation_origin",
                            "history_hours": ORIGIN_MEMORY_HOURS,
                            "level_name": "participation_origin__168h",
                            "actual_scope": key[0],
                            "control": control_name,
                            "period": key[1],
                            "approach_state": approach,
                            "response_window": response_window,
                            "origin_impulse_sign": int(key[2]),
                            "actual_event_time": actual_row["event_time"],
                            "control_event_time": control_row["event_time"],
                            "actual_base_index": int(actual_row["base_index"]),
                            "control_base_index": int(control_row["base_index"]),
                            "actual_source_available_at": actual_row["source_available_at"],
                            "control_source_available_at": control_row["source_available_at"],
                            "actual_source_open": actual_row["source_open"],
                            "control_source_open": control_row["source_open"],
                            "actual_zone_rank": -1,
                            "actual_zone_support_fraction": np.nan,
                            "match_distance": float(distance),
                            "pre_distance_atr_abs_difference": abs(
                                float(actual_row["pre_distance_atr"])
                                - float(control_row["pre_distance_atr"])
                            ),
                            "event_separation_hours": separation,
                            "actual_origin_age_hours": int(actual_row["origin_age_hours"]),
                            "control_origin_age_hours": int(control_row["origin_age_hours"]),
                            "actual_overlap_density_zone_count": int(
                                actual_row["overlap_density_zone_count"]
                            ),
                            "control_overlap_density_zone_count": int(
                                control_row["overlap_density_zone_count"]
                            ),
                            "actual_overlap_other_density_surface_count": int(
                                actual_row["overlap_other_density_surface_count"]
                            ),
                            "control_overlap_other_density_surface_count": int(
                                control_row["overlap_other_density_surface_count"]
                            ),
                            "actual_overlap_reference_level_count": int(
                                actual_row["overlap_reference_level_count"]
                            ),
                            "control_overlap_reference_level_count": int(
                                control_row["overlap_reference_level_count"]
                            ),
                            "matching_state_features": ";".join(state_features),
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
                            actual_value = float(actual_row[outcome])
                            control_value = float(control_row[outcome])
                            row[f"actual__{outcome}"] = actual_value
                            row[f"control__{outcome}"] = control_value
                            row[f"delta__{outcome}"] = actual_value - control_value
                        for feature in MATCH_STATE_FEATURES:
                            row[f"actual_state__{feature}"] = float(actual_row[feature])
                            row[f"control_state__{feature}"] = float(control_row[feature])
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
    invalid_scopes = sorted(
        set(independent["actual_scope"].astype(str)).difference(supported_scopes)
    )
    invalid_controls = sorted(set(independent["control"].astype(str)).difference(CONTROL_NAMES))
    present_pairs = sorted(independent["pair"].astype(str).unique())
    missing_pairs = sorted(set(requested_pairs).difference(present_pairs))
    separation = pd.to_numeric(independent["event_separation_hours"], errors="coerce")
    required = independent["response_window"].map(SEPARATION_HOURS_BY_RESPONSE_WINDOW)
    separation_violations = int((separation <= required).sum())
    geometry_difference = pd.to_numeric(
        independent["pre_distance_atr_abs_difference"], errors="coerce"
    )
    stale = independent["control"].eq(STALE_CONTROL)
    stale_age_violations = int(
        (
            pd.to_numeric(independent.loc[stale, "control_origin_age_hours"]) <= ORIGIN_MEMORY_HOURS
        ).sum()
    )
    ordinary_age_violations = int(
        (
            pd.to_numeric(independent.loc[~stale, "control_origin_age_hours"]) > ORIGIN_MEMORY_HOURS
        ).sum()
    )
    direction_violations = int(independent["direction_prediction"].ne(False).sum())
    profit_violations = int(independent["profit_optimization"].ne(False).sum())
    return {
        "created_at_utc": utc_now(),
        "passed": bool(
            not failures
            and not independent.empty
            and not invalid_scopes
            and not invalid_controls
            and not missing_pairs
            and separation_violations == 0
            and geometry_difference.le(0.1000001).all()
            and stale_age_violations == 0
            and ordinary_age_violations == 0
            and direction_violations == 0
            and profit_violations == 0
        ),
        "failures": failures,
        "independent_comparisons": len(independent),
        "supported_scopes": list(supported_scopes),
        "observed_scopes": sorted(independent["actual_scope"].astype(str).unique()),
        "invalid_scopes": invalid_scopes,
        "controls_present": sorted(independent["control"].astype(str).unique()),
        "invalid_controls": invalid_controls,
        "requested_pairs": list(requested_pairs),
        "present_pairs": present_pairs,
        "missing_pairs": missing_pairs,
        "event_separation_violations": separation_violations,
        "pre_distance_atr_abs_difference_max": float(geometry_difference.max()),
        "stale_age_violations": stale_age_violations,
        "ordinary_control_age_violations": ordinary_age_violations,
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

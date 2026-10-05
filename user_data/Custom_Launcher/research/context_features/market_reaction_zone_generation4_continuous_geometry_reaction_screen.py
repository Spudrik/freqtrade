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
    stable_json_sha256,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_density_arrival import (  # noqa: E501
    period_roles,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation4_continuous_geometry_preflight import (  # noqa: E501
    ARTIFACT_ROOT as PREFLIGHT_ARTIFACT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation4_continuous_geometry_preflight import (  # noqa: E501
    CANDIDATE_FEATURES,
    analysis_scope,
    apply_frozen_bands,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation4_continuous_geometry_preflight import (  # noqa: E501
    REPORT_ROOT as PREFLIGHT_REPORT_ROOT,
)


OUTPUT_SCHEMA_VERSION = 1
REPORT_ROOT = (
    OUTPUT_ROOT
    / "generation4_branches"
    / "g4c_continuous_active_region_geometry"
)
ARTIFACT_ROOT = (
    LARGE_ARTIFACT_ROOT
    / "generation4_branches"
    / "g4c_continuous_active_region_geometry"
)
SOURCE_LEVEL_NAMES = ("lvn_above", "lvn_below")
SOURCE_ZONE_METHOD = "wide_base_atr"
FUTURE_WINDOW_HOURS = 4
PRE_VOLATILITY_WINDOW_HOURS = 24

OUTCOMES: dict[str, dict[str, str]] = {
    "contact_volume_ratio": {
        "plain_language": "Contact-hour volume divided by its causal earlier baseline.",
        "source": "frozen_generation0_event",
    },
    "contact_range_ratio": {
        "plain_language": "Contact-hour high-low range divided by its causal earlier baseline.",
        "source": "frozen_generation0_event",
    },
    "abs_excursion_atr_h1": {
        "plain_language": "Largest absolute one-hour price excursion in prior ATR.",
        "source": "frozen_generation0_event",
    },
    "realized_volatility_ratio_h4": {
        "plain_language": (
            "Standard deviation of four strictly post-contact hourly returns divided by "
            "the preceding twenty-four-hour return volatility."
        ),
        "source": "causal_ohlcv_reconstruction",
    },
    "pressure_change_abs_h4": {
        "plain_language": "Absolute size of the four-hour candle-pressure change.",
        "source": "absolute_frozen_generation0_event_value",
    },
    "dwell_fraction_h4": {
        "plain_language": "Fraction of the following four closes that remain in the zone.",
        "source": "frozen_generation0_event",
    },
    "crossings_h4": {
        "plain_language": "Centre-line crossings during the following four hours.",
        "source": "frozen_generation0_event",
    },
    "time_to_leave_zone_h4": {
        "plain_language": (
            "First strictly post-contact hourly close outside the frozen zone: one to "
            "four, or five when price remains inside for the whole four-hour window."
        ),
        "source": "causal_ohlcv_reconstruction",
    },
}

EVENT_SOURCE_COLUMNS = (
    "pair",
    "control",
    "zone_method",
    "base_index",
    "event_time",
    "period",
    "level_name",
    "level_price",
    "zone_half_width",
    "contact_volume_ratio",
    "contact_range_ratio",
    "abs_excursion_atr_h1",
    "pressure_change_h4",
    "dwell_fraction_h4",
    "crossings_h4",
)


@dataclass(frozen=True)
class PairTask:
    pair: str
    manifest_path: str
    preflight_pair_path: str
    preflight_pair_sha256: str
    event_source_path: str
    event_source_bytes: int
    event_source_modified_ns: int
    run_id: str
    request_sha256: str
    overwrite: bool


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 4C raw monotonic reaction screen for development-frozen "
            "continuous geometry bands."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cohort", choices=("large", "meme"), required=True)
    parser.add_argument("--preflight-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    validate_worker_count(args.workers, manifest=manifest)
    pairs = tuple(manifest["data"]["pairs"])
    source = validate_preflight_source(
        cohort=args.cohort,
        pairs=pairs,
        preflight_run_id=args.preflight_run_id,
    )
    supported_features = tuple(source["supported_features"])
    if not supported_features:
        raise ValueError("The G4C preflight admitted no feature for reaction screening.")
    request = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "cohort": args.cohort,
        "pairs": list(pairs),
        "preflight_run_id": args.preflight_run_id,
        "preflight_record_sha256": source["preflight_record_sha256"],
        "preflight_integrity_sha256": source["preflight_integrity_sha256"],
        "preflight_edges_sha256": source["preflight_edges_sha256"],
        "supported_features": list(supported_features),
        "candidate_feature_definitions": {
            feature: CANDIDATE_FEATURES[feature] for feature in supported_features
        },
        "outcomes": OUTCOMES,
        "strictly_post_contact_reconstructions": {
            "realized_volatility_window_hours": FUTURE_WINDOW_HOURS,
            "pre_volatility_window_hours": PRE_VOLATILITY_WINDOW_HOURS,
            "time_to_leave_window_hours": FUTURE_WINDOW_HOURS,
        },
        "screen_method": (
            "Within each pair, period, feature, and low/middle/high band, average the "
            "named behaviour. Then take the equal-coin median. A raw lead requires the "
            "same non-flat low-to-middle-to-high orientation in both chronological "
            "validation periods."
        ),
        "control_boundary": (
            "This is only the smallest raw monotonic screen. A repeated cell remains "
            "interim until it survives OHLCV/broad-state, total-density, matched geometry "
            "permutation, same-state no-level, and named-identity controls."
        ),
        "reaction_outcomes_loaded": True,
        "direction_prediction": False,
        "profit_optimization": False,
        "source_contracts": source["source_contracts"],
    }
    request_sha256 = stable_json_sha256(request)
    run_dir = REPORT_ROOT / args.run_id
    artifact_dir = ARTIFACT_ROOT / args.run_id
    pair_dir = artifact_dir / "pair_reaction_rows"
    run_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    record_path = run_dir / "g4c_reaction_screen_run_record.json"
    if record_path.is_file() and not args.overwrite:
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        if existing.get("request_sha256") != request_sha256:
            raise ValueError("Run ID already exists with an incompatible request contract.")
        if existing.get("status") == "completed":
            print(json.dumps(existing, indent=2, sort_keys=True))
            return 0

    record: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "status": "running",
        "started_at_utc": utc_now(),
        "orchestrator_pid": os.getpid(),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "request_sha256": request_sha256,
        "request_contract": request,
        "baseline": (
            "The outcome-blind G4C preflight found usable three-band validation coverage "
            "for the named continuous geometry features."
        ),
        "hypothesis": (
            "At least one supported room, congestion, coverage, distribution-stretch, or "
            "MA-bundle value changes a direction-neutral reaction behaviour monotonically "
            "in both chronological validation periods."
        ),
        "pass_fail": (
            "Queue controls only for a feature/outcome cell whose equal-coin low, middle, "
            "and high medians have the same increasing or decreasing order in both "
            "validation periods. Non-monotonic cells stop here."
        ),
        "workers": args.workers,
        "reaction_outcomes_loaded": True,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(record, record_path)

    tasks = [
        PairTask(
            pair=pair,
            manifest_path=str(manifest_path),
            preflight_pair_path=source["pair_contracts"][pair]["preflight_pair_path"],
            preflight_pair_sha256=source["pair_contracts"][pair][
                "preflight_pair_sha256"
            ],
            event_source_path=source["pair_contracts"][pair]["event_source_path"],
            event_source_bytes=source["pair_contracts"][pair]["event_source_bytes"],
            event_source_modified_ns=source["pair_contracts"][pair][
                "event_source_modified_ns"
            ],
            run_id=args.run_id,
            request_sha256=request_sha256,
            overwrite=args.overwrite,
        )
        for pair in pairs
    ]
    try:
        results = run_tasks(tasks, workers=args.workers)
        inventory = DataFrame(results)
        atomic_write_parquet(inventory, run_dir / "g4c_reaction_pair_inventory.parquet")
        failures = inventory.loc[inventory["status"].eq("failed")]
        if not failures.empty:
            raise RuntimeError(
                f"{len(failures)} pair outcome task(s) failed; inspect the pair inventory."
            )
        event_rows = combine_pair_outputs(
            pair_dir=pair_dir,
            pairs=pairs,
            request_sha256=request_sha256,
        )
        event_rows["analysis_scope"] = analysis_scope(event_rows, cohort=args.cohort)
        event_rows["period_role"] = event_rows["period"].map(period_roles(manifest)).fillna(
            "unassigned"
        )
        edges = pd.read_parquet(source["preflight_edges_path"])
        banded = apply_frozen_bands(event_rows, edges)
        banded = banded.loc[banded["feature"].isin(supported_features)].copy()
        pair_band = pair_band_results(banded)
        cohort_band = cohort_band_results(pair_band, banded)
        screen = monotonic_screen(pair_band, cohort_band)
        candidate = screen.loc[screen["raw_repeated_monotonic_lead"]]
        integrity = integrity_record(
            event_rows=event_rows,
            banded=banded,
            pair_band=pair_band,
            cohort_band=cohort_band,
            screen=screen,
            supported_features=supported_features,
            requested_pairs=pairs,
            results=results,
        )
        if not integrity["passed"]:
            raise RuntimeError("G4C reaction-screen integrity failed.")

        event_path = artifact_dir / "g4c_reaction_event_rows.parquet"
        banded_path = artifact_dir / "g4c_reaction_banded_rows.parquet"
        atomic_write_parquet(event_rows, event_path)
        atomic_write_parquet(banded, banded_path)
        atomic_write_parquet(pair_band, run_dir / "g4c_pair_band_results.parquet")
        atomic_write_parquet(cohort_band, run_dir / "g4c_cohort_band_results.parquet")
        atomic_write_parquet(screen, run_dir / "g4c_raw_monotonic_screen.parquet")
        atomic_write_json(integrity, run_dir / "g4c_reaction_screen_integrity.json")
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "event_rows": len(event_rows),
                "banded_rows": len(banded),
                "screen_cells": len(screen),
                "raw_repeated_monotonic_leads": len(candidate),
                "candidate_control_cells": candidate.to_dict(orient="records"),
                "screen_decision": (
                    "run_frozen_controls_for_raw_repeated_cells"
                    if not candidate.empty
                    else "parked_no_repeated_monotonic_relationship"
                ),
                "interim_not_terminal": bool(not candidate.empty),
                "integrity": str(run_dir / "g4c_reaction_screen_integrity.json"),
                "compact_outputs": str(run_dir),
                "bulky_outputs": str(artifact_dir),
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

    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def validate_preflight_source(
    *, cohort: str, pairs: Sequence[str], preflight_run_id: str
) -> dict[str, Any]:
    run_dir = PREFLIGHT_REPORT_ROOT / preflight_run_id
    artifact_dir = PREFLIGHT_ARTIFACT_ROOT / preflight_run_id
    record_path = run_dir / "g4c_preflight_run_record.json"
    integrity_path = run_dir / "g4c_preflight_integrity.json"
    edges_path = run_dir / "g4c_development_band_edges.parquet"
    for path in (record_path, integrity_path, edges_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    request = record.get("request_contract", {})
    if record.get("status") != "completed" or not integrity.get("passed"):
        raise ValueError("The G4C preflight is not complete and clean.")
    if request.get("cohort") != cohort or list(request.get("pairs", [])) != list(pairs):
        raise ValueError("The G4C preflight cohort or pair order changed.")
    if request.get("reaction_outcomes_loaded") is not False:
        raise ValueError("The G4C preflight was not outcome blind.")
    if request.get("direction_prediction") is not False:
        raise ValueError("The G4C preflight crossed the direction boundary.")
    if request.get("profit_optimization") is not False:
        raise ValueError("The G4C preflight crossed the profit boundary.")
    supported = [
        row["feature"]
        for row in record.get("supported_feature_scopes", [])
        if row.get("both_validation_periods_supported")
    ]
    if len(supported) != len(set(supported)):
        raise ValueError("The G4C preflight duplicated a supported feature.")

    g3b_record_path = (
        OUTPUT_ROOT
        / "generation3_branches"
        / "g3b_room_obstacles"
        / request["g3b_run_id"]
        / "g3b_preflight_run_record.json"
    )
    g3b_record = json.loads(g3b_record_path.read_text(encoding="utf-8"))
    g3b_sources = g3b_record["request_contract"]["source_contracts"]
    source_contracts: list[dict[str, Any]] = [
        {
            "stage": "preflight_record",
            "path": str(record_path),
            "sha256": sha256_file(record_path),
        },
        {
            "stage": "preflight_integrity",
            "path": str(integrity_path),
            "sha256": sha256_file(integrity_path),
        },
        {
            "stage": "preflight_edges",
            "path": str(edges_path),
            "sha256": sha256_file(edges_path),
        },
        {
            "stage": "g3b_record",
            "path": str(g3b_record_path),
            "sha256": sha256_file(g3b_record_path),
        },
    ]
    pair_contracts, pair_source_contracts = validate_pair_sources(
        pairs=pairs,
        artifact_dir=artifact_dir,
        g3b_sources=g3b_sources,
    )
    source_contracts.extend(pair_source_contracts)
    return {
        "supported_features": supported,
        "preflight_record_sha256": sha256_file(record_path),
        "preflight_integrity_sha256": sha256_file(integrity_path),
        "preflight_edges_path": str(edges_path),
        "preflight_edges_sha256": sha256_file(edges_path),
        "pair_contracts": pair_contracts,
        "source_contracts": source_contracts,
    }


def validate_pair_sources(
    *,
    pairs: Sequence[str],
    artifact_dir: Path,
    g3b_sources: Sequence[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    pair_contracts: dict[str, dict[str, Any]] = {}
    source_contracts: list[dict[str, Any]] = []
    for pair in pairs:
        preflight_pair_path = artifact_dir / "pair_geometry" / f"{pair_stem(pair)}.parquet"
        source_rows = [
            row
            for row in g3b_sources
            if row["pair"] == pair and row["timeframe"] == "1h"
        ]
        if len(source_rows) != 1:
            raise ValueError(f"Expected one G3B one-hour event source for {pair}.")
        event = source_rows[0]
        event_path = Path(event["event_path"])
        stat = event_path.stat()
        if stat.st_size != int(event["event_bytes"]):
            raise ValueError(f"Frozen event-source size changed: {event_path}")
        if stat.st_mtime_ns != int(event["event_modified_ns"]):
            raise ValueError(f"Frozen event-source timestamp changed: {event_path}")
        contract = {
            "preflight_pair_path": str(preflight_pair_path),
            "preflight_pair_sha256": sha256_file(preflight_pair_path),
            "event_source_path": str(event_path),
            "event_source_bytes": stat.st_size,
            "event_source_modified_ns": stat.st_mtime_ns,
        }
        pair_contracts[pair] = contract
        source_contracts.append({"stage": "pair_sources", "pair": pair, **contract})
    return pair_contracts, source_contracts


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
        / "pair_reaction_rows"
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
            "event_rows": len(existing),
            "seconds": round(time.perf_counter() - started, 3),
        }
    feature_path = Path(task.preflight_pair_path)
    if sha256_file(feature_path) != task.preflight_pair_sha256:
        raise ValueError(f"G4C preflight pair geometry changed: {feature_path}")
    event_path = Path(task.event_source_path)
    stat = event_path.stat()
    if stat.st_size != task.event_source_bytes or stat.st_mtime_ns != task.event_source_modified_ns:
        raise ValueError(f"Frozen Generation-0 event source changed: {event_path}")
    features = pd.read_parquet(feature_path)
    outcomes = pd.read_parquet(
        event_path,
        filters=[
            ("control", "==", "actual"),
            ("level_name", "in", list(SOURCE_LEVEL_NAMES)),
            ("zone_method", "==", SOURCE_ZONE_METHOD),
        ],
        columns=list(EVENT_SOURCE_COLUMNS),
    )
    keys = ["pair", "period", "event_time", "base_index", "level_name"]
    outcomes = outcomes.drop_duplicates(keys, keep="first")
    merged = features.merge(
        outcomes,
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("", "__source"),
    )
    required = (
        "contact_volume_ratio",
        "contact_range_ratio",
        "abs_excursion_atr_h1",
        "pressure_change_h4",
        "dwell_fraction_h4",
        "crossings_h4",
    )
    if merged[list(required)].isna().all(axis=1).any():
        raise ValueError(
            f"At least one G4C event did not rejoin its frozen outcomes for {task.pair}."
        )
    merged["pressure_change_abs_h4"] = pd.to_numeric(
        merged["pressure_change_h4"], errors="coerce"
    ).abs()
    base = prepare_base_market_frame(task.pair, load_manifest(Path(task.manifest_path)))
    realized, leave = reconstructed_outcomes(merged, base=base)
    merged["realized_volatility_ratio_h4"] = realized
    merged["time_to_leave_zone_h4"] = leave
    merged["reaction_outcomes_loaded"] = True
    merged["direction_prediction"] = False
    merged["profit_optimization"] = False
    merged["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    merged["run_request_sha256"] = task.request_sha256
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(merged, output_path)
    return {
        "pair": task.pair,
        "status": "completed",
        "event_rows": len(merged),
        "seconds": round(time.perf_counter() - started, 3),
    }


def reconstructed_outcomes(events: DataFrame, *, base: DataFrame) -> tuple[np.ndarray, np.ndarray]:
    close = pd.to_numeric(base["close"], errors="coerce").to_numpy(dtype=float)
    log_return = np.full(len(close), np.nan, dtype=float)
    valid = np.isfinite(close[1:]) & np.isfinite(close[:-1]) & (close[1:] > 0) & (close[:-1] > 0)
    log_return_tail = np.full(len(close) - 1, np.nan, dtype=float)
    log_return_tail[valid] = np.log(close[1:][valid] / close[:-1][valid])
    log_return[1:] = log_return_tail
    realized = np.full(len(events), np.nan, dtype=float)
    leave = np.full(len(events), np.nan, dtype=float)
    for position, row in events.reset_index(drop=True).iterrows():
        index = int(row["base_index"])
        if index < PRE_VOLATILITY_WINDOW_HOURS or index + FUTURE_WINDOW_HOURS >= len(close):
            continue
        prior = log_return[index - PRE_VOLATILITY_WINDOW_HOURS + 1 : index + 1]
        future = log_return[index + 1 : index + FUTURE_WINDOW_HOURS + 1]
        prior_vol = float(np.nanstd(prior, ddof=0))
        future_vol = float(np.nanstd(future, ddof=0))
        if np.isfinite(prior_vol) and prior_vol > 0.0 and np.isfinite(future_vol):
            realized[position] = future_vol / prior_vol
        level = float(row["level_price"])
        width = float(row["zone_half_width"])
        future_closes = close[index + 1 : index + FUTURE_WINDOW_HOURS + 1]
        outside = np.flatnonzero((future_closes < level - width) | (future_closes > level + width))
        leave[position] = float(outside[0] + 1) if len(outside) else FUTURE_WINDOW_HOURS + 1
    return realized, leave


def pair_band_results(banded: DataFrame) -> DataFrame:
    selected = banded.loc[banded["feature_band"].notna()].copy()
    rows: list[DataFrame] = []
    group_columns = [
        "analysis_scope",
        "pair",
        "period",
        "period_role",
        "feature",
        "feature_band",
    ]
    for outcome in OUTCOMES:
        grouped = (
            selected.groupby(group_columns, observed=True)
            .agg(
                events=("event_time", "size"),
                outcome_mean=(outcome, "mean"),
                outcome_median=(outcome, "median"),
                feature_value_median=("feature_value", "median"),
            )
            .reset_index()
        )
        grouped["outcome"] = outcome
        rows.append(grouped)
    output = pd.concat(rows, ignore_index=True)
    output["direction_prediction"] = False
    output["profit_optimization"] = False
    return output


def cohort_band_results(pair_band: DataFrame, banded: DataFrame) -> DataFrame:
    group_columns = [
        "analysis_scope",
        "period",
        "period_role",
        "feature",
        "feature_band",
        "outcome",
    ]
    output = (
        pair_band.groupby(group_columns, observed=True)
        .agg(
            coins=("pair", "nunique"),
            independent_events=("events", "sum"),
            equal_coin_outcome_median=("outcome_mean", "median"),
            equal_coin_outcome_q25=("outcome_mean", lambda values: values.quantile(0.25)),
            equal_coin_outcome_q75=("outcome_mean", lambda values: values.quantile(0.75)),
            equal_coin_feature_median=("feature_value_median", "median"),
        )
        .reset_index()
    )
    pooled_rows: list[dict[str, Any]] = []
    for key, group in banded.groupby(
        ["analysis_scope", "period", "feature", "feature_band"],
        observed=True,
    ):
        for outcome in OUTCOMES:
            values = pd.to_numeric(group[outcome], errors="coerce")
            pooled_rows.append(
                {
                    "analysis_scope": key[0],
                    "period": key[1],
                    "feature": key[2],
                    "feature_band": key[3],
                    "outcome": outcome,
                    "pooled_event_outcome_median": float(values.median()),
                }
            )
    pooled = DataFrame(pooled_rows)
    output = output.merge(
        pooled,
        on=["analysis_scope", "period", "feature", "feature_band", "outcome"],
        how="left",
        validate="one_to_one",
    )
    output["direction_prediction"] = False
    output["profit_optimization"] = False
    return output


def monotonic_orientation(low: float, middle: float, high: float) -> str:
    if not all(np.isfinite(value) for value in (low, middle, high)) or high == low:
        return "not_monotonic"
    if low <= middle <= high:
        return "increasing"
    if low >= middle >= high:
        return "decreasing"
    return "not_monotonic"


def monotonic_screen(pair_band: DataFrame, cohort_band: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    validation = cohort_band.loc[
        cohort_band["period_role"].eq("chronological_internal_validation")
        & cohort_band["analysis_scope"].ne("btc")
    ]
    for (scope, feature, outcome), group in validation.groupby(
        ["analysis_scope", "feature", "outcome"],
        observed=True,
    ):
        periods = group["period"].astype(str).drop_duplicates().tolist()
        period_rows: list[dict[str, Any]] = []
        for period in periods:
            cell = group.loc[group["period"].astype(str).eq(period)].set_index("feature_band")
            values = {
                band: float(cell.loc[band, "equal_coin_outcome_median"])
                if band in cell.index
                else np.nan
                for band in ("low", "middle", "high")
            }
            orientation = monotonic_orientation(
                values["low"], values["middle"], values["high"]
            )
            pair_cell = pair_band.loc[
                pair_band["analysis_scope"].eq(scope)
                & pair_band["period"].astype(str).eq(period)
                & pair_band["feature"].eq(feature)
                & pair_band["outcome"].eq(outcome)
            ]
            pivot = pair_cell.pivot(index="pair", columns="feature_band", values="outcome_mean")
            if {"low", "high"}.issubset(pivot.columns):
                differences = pivot["high"] - pivot["low"]
                orientation_fraction = (
                    float(differences.gt(0).mean())
                    if orientation == "increasing"
                    else float(differences.lt(0).mean())
                    if orientation == "decreasing"
                    else np.nan
                )
                endpoint_coins = int(differences.notna().sum())
            else:
                orientation_fraction = np.nan
                endpoint_coins = 0
            period_rows.append(
                {
                    "period": period,
                    "orientation": orientation,
                    "low": values["low"],
                    "middle": values["middle"],
                    "high": values["high"],
                    "endpoint_difference": values["high"] - values["low"],
                    "endpoint_coin_count": endpoint_coins,
                    "endpoint_coin_orientation_fraction": orientation_fraction,
                }
            )
        orientations = [row["orientation"] for row in period_rows]
        repeated = (
            len(period_rows) == 2
            and orientations[0] in {"increasing", "decreasing"}
            and orientations[0] == orientations[1]
        )
        rows.append(
            {
                "analysis_scope": scope,
                "feature": feature,
                "feature_plain_language": CANDIDATE_FEATURES[feature]["plain_language"],
                "outcome": outcome,
                "outcome_plain_language": OUTCOMES[outcome]["plain_language"],
                "validation_periods": ";".join(periods),
                "validation_period_count": len(periods),
                "early_orientation": period_rows[0]["orientation"] if period_rows else None,
                "late_orientation": period_rows[1]["orientation"] if len(period_rows) > 1 else None,
                "early_low": period_rows[0]["low"] if period_rows else np.nan,
                "early_middle": period_rows[0]["middle"] if period_rows else np.nan,
                "early_high": period_rows[0]["high"] if period_rows else np.nan,
                "late_low": period_rows[1]["low"] if len(period_rows) > 1 else np.nan,
                "late_middle": period_rows[1]["middle"] if len(period_rows) > 1 else np.nan,
                "late_high": period_rows[1]["high"] if len(period_rows) > 1 else np.nan,
                "early_endpoint_difference": (
                    period_rows[0]["endpoint_difference"] if period_rows else np.nan
                ),
                "late_endpoint_difference": (
                    period_rows[1]["endpoint_difference"] if len(period_rows) > 1 else np.nan
                ),
                "early_endpoint_coin_count": (
                    period_rows[0]["endpoint_coin_count"] if period_rows else 0
                ),
                "late_endpoint_coin_count": (
                    period_rows[1]["endpoint_coin_count"] if len(period_rows) > 1 else 0
                ),
                "early_endpoint_coin_orientation_fraction": (
                    period_rows[0]["endpoint_coin_orientation_fraction"]
                    if period_rows
                    else np.nan
                ),
                "late_endpoint_coin_orientation_fraction": (
                    period_rows[1]["endpoint_coin_orientation_fraction"]
                    if len(period_rows) > 1
                    else np.nan
                ),
                "raw_repeated_monotonic_lead": repeated,
                "controls_completed": False,
                "direction_prediction": False,
                "profit_optimization": False,
            }
        )
    return DataFrame(rows)


def integrity_record(
    *,
    event_rows: DataFrame,
    banded: DataFrame,
    pair_band: DataFrame,
    cohort_band: DataFrame,
    screen: DataFrame,
    supported_features: Sequence[str],
    requested_pairs: Sequence[str],
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    failures = [row for row in results if row.get("status") == "failed"]
    missing_pairs = sorted(set(requested_pairs) - set(event_rows["pair"].astype(str)))
    missing_outcomes = sorted(set(OUTCOMES) - set(event_rows.columns))
    direction_violations = int(event_rows["direction_prediction"].ne(False).sum())
    profit_violations = int(event_rows["profit_optimization"].ne(False).sum())
    feature_set = set(banded["feature"].astype(str))
    expected_banded_rows = len(event_rows) * len(supported_features)
    return {
        "passed": bool(
            not failures
            and not missing_pairs
            and not missing_outcomes
            and not event_rows.empty
            and direction_violations == 0
            and profit_violations == 0
            and feature_set == set(supported_features)
            and len(banded) == expected_banded_rows
            and not pair_band.empty
            and not cohort_band.empty
            and not screen.empty
        ),
        "pair_failures": failures,
        "missing_pairs": missing_pairs,
        "missing_outcomes": missing_outcomes,
        "event_rows": len(event_rows),
        "banded_rows": len(banded),
        "expected_banded_rows": expected_banded_rows,
        "pair_band_rows": len(pair_band),
        "cohort_band_rows": len(cohort_band),
        "screen_rows": len(screen),
        "raw_repeated_monotonic_leads": int(screen["raw_repeated_monotonic_lead"].sum()),
        "direction_violations": direction_violations,
        "profit_violations": profit_violations,
        "direction_prediction": False,
        "profit_optimization": False,
    }


def validate_pair_output(path: Path, *, pair: str, request_sha256: str) -> DataFrame:
    frame = pd.read_parquet(path)
    if set(frame["pair"].astype(str)) != {pair}:
        raise ValueError(f"Existing G4C reaction pair output has the wrong pair: {path}")
    if set(frame["output_schema_version"].astype(int)) != {OUTPUT_SCHEMA_VERSION}:
        raise ValueError(f"Existing G4C reaction pair output has the wrong schema: {path}")
    if set(frame["run_request_sha256"].astype(str)) != {request_sha256}:
        raise ValueError(f"Existing G4C reaction pair output has the wrong request hash: {path}")
    return frame


def combine_pair_outputs(
    *, pair_dir: Path, pairs: Sequence[str], request_sha256: str
) -> DataFrame:
    frames = [
        validate_pair_output(
            pair_dir / f"{pair_stem(pair)}.parquet",
            pair=pair,
            request_sha256=request_sha256,
        )
        for pair in pairs
    ]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    raise SystemExit(main())

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
    atomic_write_json,
    atomic_write_parquet,
    load_manifest,
    prepare_base_market_frame,
    sha256_file,
    stable_hash_int,
    utc_now,
    validate_worker_count,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_localization import (  # noqa: E501
    balance_diagnostics,
    causal_local_state,
    causal_market_context,
    nearest_state_pairs,
    pair_stem,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_anchored_vwap import (  # noqa: E501
    MAX_MEDIAN_STATE_SMD,
    MAX_STATE_SMD,
    stable_json_sha256,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_density_reaction import (  # noqa: E501
    CORE_STATE_FEATURES,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_density_arrival import (  # noqa: E501
    period_roles,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation4_continuous_geometry_preflight import (  # noqa: E501
    REPORT_ROOT as PREFLIGHT_REPORT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation4_continuous_geometry_preflight import (  # noqa: E501
    analysis_scope,
    apply_frozen_bands,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation4_continuous_geometry_reaction_screen import (  # noqa: E501
    ARTIFACT_ROOT as REACTION_ARTIFACT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation4_continuous_geometry_reaction_screen import (  # noqa: E501
    EVENT_SOURCE_COLUMNS,
    FUTURE_WINDOW_HOURS,
    SOURCE_LEVEL_NAMES,
    SOURCE_ZONE_METHOD,
    reconstructed_outcomes,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation4_continuous_geometry_reaction_screen import (  # noqa: E501
    REPORT_ROOT as REACTION_REPORT_ROOT,
)


OUTPUT_SCHEMA_VERSION = 1
REPORT_ROOT = REACTION_REPORT_ROOT
ARTIFACT_ROOT = REACTION_ARTIFACT_ROOT
MIN_MATCHED_PAIRS = 50
MIN_MATCHED_COINS = 5
MIN_IDENTITY_PAIRS = 25
MIN_IDENTITY_COINS = 5
MIN_POSITIVE_IDENTITY_FRACTION = 0.5
PERMUTATION_ALPHA = 0.05
MINIMUM_EVENT_SEPARATION_HOURS = FUTURE_WINDOW_HOURS
PRE_DISTANCE_ATR_CALIPER = 0.10
TOTAL_DENSITY_FEATURES = (
    "total_higher_tf_level_count_within_2atr",
    "total_higher_tf_dependency_group_count_within_2atr",
    "causal_ma_bundle_level_count",
)
STATE_CONTROL_FEATURES = tuple(CORE_STATE_FEATURES)
DENSITY_CONTROL_FEATURES = (*STATE_CONTROL_FEATURES, *TOTAL_DENSITY_FEATURES)
VALIDATION_ROLE = "chronological_internal_validation"

NO_LEVEL_SOURCE_COLUMNS = tuple(
    dict.fromkeys(
        (
            *EVENT_SOURCE_COLUMNS,
            "approach_state",
            "pre_distance_atr",
            "random_match_tier",
        )
    )
)


@dataclass(frozen=True)
class PairTask:
    pair: str
    manifest_path: str
    context_path: str
    reaction_pair_path: str
    reaction_pair_sha256: str
    event_source_path: str
    event_source_bytes: int
    event_source_modified_ns: int
    run_id: str
    request_sha256: str
    overwrite: bool


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generation 4C frozen control ladder for continuous geometry leads."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cohort", choices=("large", "meme"), required=True)
    parser.add_argument("--reaction-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--permutations", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.permutations < 99:
        raise ValueError("Use at least 99 deterministic matched-pair permutations.")

    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    validate_worker_count(args.workers, manifest=manifest)
    pairs = tuple(manifest["data"]["pairs"])
    source = validate_reaction_source(
        cohort=args.cohort,
        pairs=pairs,
        reaction_run_id=args.reaction_run_id,
    )
    candidates = DataFrame(source["candidate_control_cells"])
    if candidates.empty:
        raise ValueError("The reaction screen has no raw lead requiring controls.")

    request = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "cohort": args.cohort,
        "pairs": list(pairs),
        "reaction_run_id": args.reaction_run_id,
        "reaction_record_sha256": source["reaction_record_sha256"],
        "reaction_integrity_sha256": source["reaction_integrity_sha256"],
        "candidate_cells": source["candidate_control_cells"],
        "core_state_features": list(STATE_CONTROL_FEATURES),
        "total_density_features": list(TOTAL_DENSITY_FEATURES),
        "controls": {
            "ohlcv_and_broad_market_state_only": (
                "Match adjacent geometry bands within pair, period, LVN identity, and "
                "approach using causal local OHLCV, technical, BTC, and cohort state."
            ),
            "total_density_without_geometry": (
                "Repeat the state match while additionally matching total nearby "
                "higher-timeframe line, dependency-group, and MA-bundle counts."
            ),
            "permuted_geometry_within_pair_period_state": (
                "Deterministically swap the two members of state-matched geometry "
                "pairs and compare the actual equal-coin effect with the null."
            ),
            "same_state_no_level": (
                "Match each actual geometry band to frozen Generation-0 random-time "
                "LVN controls with similar causal state and no same-LVN contact nearby."
            ),
            "named_level_identity_ablation": (
                "Require controlled effects to remain separately usable for lvn_above "
                "and lvn_below rather than being carried by one named anchor."
            ),
            "normal_meme_and_btc_separation": (
                "Normal alts, memes, and descriptive BTC rows remain separate."
            ),
        },
        "declared_review_thresholds_not_native_scores": {
            "minimum_matched_pairs_per_adjacent_contrast_period": MIN_MATCHED_PAIRS,
            "minimum_coins_per_adjacent_contrast_period": MIN_MATCHED_COINS,
            "maximum_state_smd": MAX_STATE_SMD,
            "maximum_median_state_smd": MAX_MEDIAN_STATE_SMD,
            "permutation_alpha": PERMUTATION_ALPHA,
            "permutations": args.permutations,
            "minimum_identity_pairs": MIN_IDENTITY_PAIRS,
            "minimum_identity_coins": MIN_IDENTITY_COINS,
            "minimum_positive_identity_coin_fraction": MIN_POSITIVE_IDENTITY_FRACTION,
        },
        "source_contracts": source["source_contracts"],
        "reaction_outcomes_loaded": True,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    request_sha256 = stable_json_sha256(request)
    run_dir = REPORT_ROOT / args.run_id
    artifact_dir = ARTIFACT_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    record_path = run_dir / "g4c_control_run_record.json"
    if record_path.is_file() and not args.overwrite:
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        if existing.get("request_sha256") != request_sha256:
            raise ValueError("Run ID already exists with an incompatible control contract.")
        if existing.get("status") == "completed":
            print(json.dumps(existing, indent=2, sort_keys=True))
            return 0

    record: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "status": "running",
        "started_at_utc": utc_now(),
        "orchestrator_pid": os.getpid(),
        "workers": args.workers,
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "request_sha256": request_sha256,
        "request_contract": request,
        "baseline": (
            "The raw G4C screen found low-middle-high outcome ordering in both "
            "validation periods, without yet excluding state, density, random geometry, "
            "no-level timing, or named-anchor explanations."
        ),
        "hypothesis": (
            "At least one raw cell keeps both adjacent ordinal effects in both validation "
            "periods after every frozen control and across at least five coins."
        ),
        "pass_fail": (
            "Retain only cells for which state-matched and density-matched adjacent "
            "contrasts have adequate balanced support, positive trader-readable effect, "
            "at least five agreeing coins, one-sided matched-permutation p <= 0.05, "
            "positive same-state no-level-adjusted contrasts, and both LVN identities. "
            "These are project review thresholds, not native market relevance scores."
        ),
        "reaction_outcomes_loaded": True,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(record, record_path)

    try:
        context = causal_market_context(manifest)
        context_path = run_dir / "g4c_causal_market_context.parquet"
        atomic_write_parquet(context, context_path)
        tasks = build_pair_tasks(
            pairs=pairs,
            manifest_path=manifest_path,
            context_path=context_path,
            source=source,
            run_id=args.run_id,
            request_sha256=request_sha256,
            overwrite=args.overwrite,
        )
        pair_results = run_pair_tasks(tasks, workers=args.workers)
        inventory = DataFrame(pair_results)
        atomic_write_parquet(inventory, run_dir / "g4c_control_pair_inventory.parquet")
        failures = inventory.loc[inventory["status"].eq("failed")]
        if not failures.empty:
            raise RuntimeError(
                f"{len(failures)} G4C control pair task(s) failed; inspect inventory."
            )
        actual, no_level = combine_pair_control_outputs(
            artifact_dir=artifact_dir,
            pairs=pairs,
            request_sha256=request_sha256,
        )
        edges = pd.read_parquet(source["preflight_edges_path"])
        banded = apply_frozen_bands(actual, edges)
        candidate_features = set(candidates["feature"].astype(str))
        banded = banded.loc[banded["feature"].isin(candidate_features)].copy()
        outputs = run_control_ladder(
            banded=banded,
            no_level=no_level,
            candidates=candidates,
            permutations=args.permutations,
        )
        integrity = control_integrity(
            actual=actual,
            no_level=no_level,
            banded=banded,
            candidates=candidates,
            outputs=outputs,
            pairs=pairs,
        )
        if not integrity["passed"]:
            raise RuntimeError("G4C control-ladder integrity failed.")

        atomic_write_parquet(actual, artifact_dir / "g4c_control_actual_state_rows.parquet")
        atomic_write_parquet(no_level, artifact_dir / "g4c_control_no_level_state_rows.parquet")
        atomic_write_parquet(banded, artifact_dir / "g4c_control_banded_rows.parquet")
        detailed_outputs = {
            "state_matched_pairs",
            "density_matched_pairs",
            "no_level_matched_pairs",
        }
        for name, frame in outputs.items():
            destination = artifact_dir if name in detailed_outputs else run_dir
            atomic_write_parquet(frame, destination / f"g4c_{name}.parquet")
        atomic_write_json(integrity, run_dir / "g4c_control_integrity.json")

        verdicts = outputs["candidate_verdicts"]
        retained = verdicts.loc[verdicts["retained_after_all_controls"]]
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "actual_event_rows": len(actual),
                "no_level_control_rows": len(no_level),
                "banded_rows": len(banded),
                "candidate_cells": len(candidates),
                "retained_cells": len(retained),
                "candidate_verdicts": verdicts.to_dict(orient="records"),
                "screen_decision": (
                    "retain_controlled_continuous_geometry_lead"
                    if not retained.empty
                    else "parked_all_raw_leads_explained_or_unsupported_after_controls"
                ),
                "terminal_for_cohort": True,
                "integrity": str(run_dir / "g4c_control_integrity.json"),
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


def validate_reaction_source(  # noqa: C901 - source contracts remain explicit
    *, cohort: str, pairs: Sequence[str], reaction_run_id: str
) -> dict[str, Any]:
    run_dir = REACTION_REPORT_ROOT / reaction_run_id
    artifact_dir = REACTION_ARTIFACT_ROOT / reaction_run_id
    record_path = run_dir / "g4c_reaction_screen_run_record.json"
    integrity_path = run_dir / "g4c_reaction_screen_integrity.json"
    for path in (record_path, integrity_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    request = record.get("request_contract", {})
    if record.get("status") != "completed" or not integrity.get("passed"):
        raise ValueError("The G4C reaction screen is not complete and clean.")
    if request.get("cohort") != cohort or list(request.get("pairs", [])) != list(pairs):
        raise ValueError("The G4C reaction cohort or pair order changed.")
    if request.get("reaction_outcomes_loaded") is not True:
        raise ValueError("The G4C reaction screen did not open its declared outcomes.")
    if request.get("direction_prediction") is not False:
        raise ValueError("The G4C reaction screen crossed the direction boundary.")
    if request.get("profit_optimization") is not False:
        raise ValueError("The G4C reaction screen crossed the profit boundary.")
    candidates = record.get("candidate_control_cells", [])
    if int(record.get("raw_repeated_monotonic_leads", -1)) != len(candidates):
        raise ValueError("The G4C reaction candidate count is inconsistent.")
    preflight_run_id = str(request["preflight_run_id"])
    edges_path = PREFLIGHT_REPORT_ROOT / preflight_run_id / "g4c_development_band_edges.parquet"
    if not edges_path.is_file():
        raise FileNotFoundError(edges_path)

    source_rows = {
        str(row["pair"]): row
        for row in request.get("source_contracts", [])
        if row.get("stage") == "pair_sources"
    }
    pair_contracts: dict[str, dict[str, Any]] = {}
    contracts: list[dict[str, Any]] = [
        {"stage": "reaction_record", "path": str(record_path), "sha256": sha256_file(record_path)},
        {
            "stage": "reaction_integrity",
            "path": str(integrity_path),
            "sha256": sha256_file(integrity_path),
        },
        {"stage": "preflight_edges", "path": str(edges_path), "sha256": sha256_file(edges_path)},
    ]
    for pair in pairs:
        if pair not in source_rows:
            raise ValueError(f"The G4C reaction record lacks pair sources for {pair}.")
        row = source_rows[pair]
        event_path = Path(row["event_source_path"])
        event_stat = event_path.stat()
        if event_stat.st_size != int(row["event_source_bytes"]):
            raise ValueError(f"Frozen event source size changed: {event_path}")
        if event_stat.st_mtime_ns != int(row["event_source_modified_ns"]):
            raise ValueError(f"Frozen event source timestamp changed: {event_path}")
        reaction_pair = artifact_dir / "pair_reaction_rows" / f"{pair_stem(pair)}.parquet"
        contract = {
            "reaction_pair_path": str(reaction_pair),
            "reaction_pair_sha256": sha256_file(reaction_pair),
            "event_source_path": str(event_path),
            "event_source_bytes": event_stat.st_size,
            "event_source_modified_ns": event_stat.st_mtime_ns,
        }
        pair_contracts[pair] = contract
        contracts.append({"stage": "pair_sources", "pair": pair, **contract})
    return {
        "reaction_record_sha256": sha256_file(record_path),
        "reaction_integrity_sha256": sha256_file(integrity_path),
        "candidate_control_cells": candidates,
        "preflight_edges_path": str(edges_path),
        "pair_contracts": pair_contracts,
        "source_contracts": contracts,
    }


def build_pair_tasks(
    *,
    pairs: Sequence[str],
    manifest_path: Path,
    context_path: Path,
    source: dict[str, Any],
    run_id: str,
    request_sha256: str,
    overwrite: bool,
) -> list[PairTask]:
    return [
        PairTask(
            pair=pair,
            manifest_path=str(manifest_path),
            context_path=str(context_path),
            run_id=run_id,
            request_sha256=request_sha256,
            overwrite=overwrite,
            **source["pair_contracts"][pair],
        )
        for pair in pairs
    ]


def run_pair_tasks(tasks: Sequence[PairTask], *, workers: int) -> list[dict[str, Any]]:
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
        return {
            "pair": task.pair,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_pair(task: PairTask) -> dict[str, Any]:
    started = time.perf_counter()
    pair_root = ARTIFACT_ROOT / task.run_id
    actual_path = pair_root / "pair_actual_state" / f"{pair_stem(task.pair)}.parquet"
    no_level_path = pair_root / "pair_no_level_state" / f"{pair_stem(task.pair)}.parquet"
    if actual_path.is_file() and no_level_path.is_file() and not task.overwrite:
        actual = validate_pair_state_output(
            actual_path,
            pair=task.pair,
            request_sha256=task.request_sha256,
            row_kind="actual_geometry",
        )
        no_level = validate_pair_state_output(
            no_level_path,
            pair=task.pair,
            request_sha256=task.request_sha256,
            row_kind="same_state_no_level_pool",
        )
        return {
            "pair": task.pair,
            "status": "existing",
            "actual_rows": len(actual),
            "no_level_rows": len(no_level),
            "seconds": round(time.perf_counter() - started, 3),
        }

    reaction_path = Path(task.reaction_pair_path)
    if sha256_file(reaction_path) != task.reaction_pair_sha256:
        raise ValueError(f"Frozen G4C reaction pair changed: {reaction_path}")
    event_path = Path(task.event_source_path)
    stat = event_path.stat()
    if stat.st_size != task.event_source_bytes or stat.st_mtime_ns != task.event_source_modified_ns:
        raise ValueError(f"Frozen Generation-0 event source changed: {event_path}")

    manifest = load_manifest(Path(task.manifest_path))
    cohort = manifest_cohort(manifest)
    base = prepare_base_market_frame(task.pair, manifest)
    context = pd.read_parquet(task.context_path)
    state = causal_local_state(base).merge(context, on="date", how="left", validate="one_to_one")
    if not pd.to_datetime(state["date"], utc=True).equals(pd.to_datetime(base["date"], utc=True)):
        raise ValueError(f"Causal state alignment changed for {task.pair}.")

    actual = pd.read_parquet(reaction_path)
    actual = attach_control_state(actual, base=base, state=state)
    actual["analysis_scope"] = analysis_scope(actual, cohort=cohort)
    actual["period_role"] = actual["period"].map(period_roles(manifest)).fillna("unassigned")
    actual["row_kind"] = "actual_geometry"

    no_level = pd.read_parquet(
        event_path,
        filters=[
            ("control", "==", "matched_random_time"),
            ("level_name", "in", list(SOURCE_LEVEL_NAMES)),
            ("zone_method", "==", SOURCE_ZONE_METHOD),
        ],
        columns=list(NO_LEVEL_SOURCE_COLUMNS),
    )
    keys = ["pair", "period", "event_time", "base_index", "level_name"]
    no_level = no_level.drop_duplicates(keys, keep="first").reset_index(drop=True)
    no_level["pressure_change_abs_h4"] = pd.to_numeric(
        no_level["pressure_change_h4"], errors="coerce"
    ).abs()
    realized, leave = reconstructed_outcomes(no_level, base=base)
    no_level["realized_volatility_ratio_h4"] = realized
    no_level["time_to_leave_zone_h4"] = leave
    no_level = attach_control_state(no_level, base=base, state=state)
    no_level["analysis_scope"] = analysis_scope(no_level, cohort=cohort)
    no_level["period_role"] = no_level["period"].map(period_roles(manifest)).fillna("unassigned")
    no_level["row_kind"] = "same_state_no_level_pool"

    for frame in (actual, no_level):
        frame["output_schema_version"] = OUTPUT_SCHEMA_VERSION
        frame["run_request_sha256"] = task.request_sha256
        frame["reaction_outcomes_loaded"] = True
        frame["direction_prediction"] = False
        frame["profit_optimization"] = False
    actual_path.parent.mkdir(parents=True, exist_ok=True)
    no_level_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(actual, actual_path)
    atomic_write_parquet(no_level, no_level_path)
    return {
        "pair": task.pair,
        "status": "completed",
        "actual_rows": len(actual),
        "no_level_rows": len(no_level),
        "seconds": round(time.perf_counter() - started, 3),
    }


def manifest_cohort(manifest: dict[str, Any]) -> str:
    pairs = set(manifest["data"]["pairs"])
    return "large" if "BTC/USDT:USDT" in pairs else "meme"


def attach_control_state(events: DataFrame, *, base: DataFrame, state: DataFrame) -> DataFrame:
    output = events.copy()
    indexes = output["base_index"].to_numpy(dtype=np.int64)
    if (indexes < 0).any() or (indexes >= len(base)).any():
        raise ValueError("A control event base index is outside the frozen OHLCV frame.")
    expected = pd.to_datetime(base["date"].iloc[indexes].reset_index(drop=True), utc=True)
    observed = pd.to_datetime(output["event_time"].reset_index(drop=True), utc=True)
    if not expected.equals(observed):
        raise ValueError("Control event timestamps no longer match their causal base rows.")
    for column in STATE_CONTROL_FEATURES:
        output[column] = pd.to_numeric(state[column], errors="coerce").to_numpy()[indexes]
    pre_close = pd.to_numeric(base["pre_close"], errors="coerce").to_numpy(dtype=float)[indexes]
    atr = pd.to_numeric(base["base_atr"], errors="coerce").to_numpy(dtype=float)[indexes]
    level = pd.to_numeric(output["level_price"], errors="coerce").to_numpy(dtype=float)
    output["pre_distance_atr"] = np.abs(level - pre_close) / atr
    return output.replace([np.inf, -np.inf], np.nan)


def validate_pair_state_output(
    path: Path, *, pair: str, request_sha256: str, row_kind: str
) -> DataFrame:
    frame = pd.read_parquet(path)
    if set(frame["pair"].astype(str)) != {pair}:
        raise ValueError(f"Existing G4C control pair output has the wrong pair: {path}")
    if set(frame["row_kind"].astype(str)) != {row_kind}:
        raise ValueError(f"Existing G4C control pair output has the wrong row kind: {path}")
    if set(frame["output_schema_version"].astype(int)) != {OUTPUT_SCHEMA_VERSION}:
        raise ValueError(f"Existing G4C control pair output has the wrong schema: {path}")
    if set(frame["run_request_sha256"].astype(str)) != {request_sha256}:
        raise ValueError(f"Existing G4C control pair output has the wrong request hash: {path}")
    return frame


def combine_pair_control_outputs(
    *, artifact_dir: Path, pairs: Sequence[str], request_sha256: str
) -> tuple[DataFrame, DataFrame]:
    actual = [
        validate_pair_state_output(
            artifact_dir / "pair_actual_state" / f"{pair_stem(pair)}.parquet",
            pair=pair,
            request_sha256=request_sha256,
            row_kind="actual_geometry",
        )
        for pair in pairs
    ]
    no_level = [
        validate_pair_state_output(
            artifact_dir / "pair_no_level_state" / f"{pair_stem(pair)}.parquet",
            pair=pair,
            request_sha256=request_sha256,
            row_kind="same_state_no_level_pool",
        )
        for pair in pairs
    ]
    return pd.concat(actual, ignore_index=True), pd.concat(no_level, ignore_index=True)


def run_control_ladder(
    *,
    banded: DataFrame,
    no_level: DataFrame,
    candidates: DataFrame,
    permutations: int,
) -> dict[str, DataFrame]:
    state_matches: list[DataFrame] = []
    density_matches: list[DataFrame] = []
    no_level_matches: list[DataFrame] = []
    state_summaries: list[DataFrame] = []
    density_summaries: list[DataFrame] = []
    no_level_summaries: list[DataFrame] = []
    identity_summaries: list[DataFrame] = []

    for _, candidate in candidates.iterrows():
        cell = candidate_rows(banded, candidate)
        state_match = adjacent_geometry_matches(
            cell,
            candidate=candidate,
            state_columns=STATE_CONTROL_FEATURES,
            control_name="ohlcv_and_broad_market_state_only",
        )
        density_match = adjacent_geometry_matches(
            cell,
            candidate=candidate,
            state_columns=DENSITY_CONTROL_FEATURES,
            control_name="total_density_without_geometry",
        )
        no_level_match = same_state_no_level_matches(
            cell,
            no_level=no_level,
            candidate=candidate,
        )
        state_matches.append(state_match)
        density_matches.append(density_match)
        no_level_matches.append(no_level_match)
        state_summaries.append(
            summarize_geometry_controls(
                state_match,
                candidate=candidate,
                state_columns=STATE_CONTROL_FEATURES,
                permutations=permutations,
                control_name="ohlcv_and_broad_market_state_only",
            )
        )
        density_summaries.append(
            summarize_geometry_controls(
                density_match,
                candidate=candidate,
                state_columns=DENSITY_CONTROL_FEATURES,
                permutations=permutations,
                control_name="total_density_without_geometry",
            )
        )
        no_level_summaries.append(
            summarize_no_level_controls(no_level_match, candidate=candidate)
        )
        identity_summaries.append(
            summarize_identity_ablation(density_match, candidate=candidate)
        )

    state_match_frame = concat_or_empty(state_matches)
    density_match_frame = concat_or_empty(density_matches)
    no_level_match_frame = concat_or_empty(no_level_matches)
    state_summary = pd.concat(state_summaries, ignore_index=True)
    density_summary = pd.concat(density_summaries, ignore_index=True)
    no_level_summary = pd.concat(no_level_summaries, ignore_index=True)
    identity_summary = pd.concat(identity_summaries, ignore_index=True)
    verdicts = candidate_verdicts(
        candidates=candidates,
        state_summary=state_summary,
        density_summary=density_summary,
        no_level_summary=no_level_summary,
        identity_summary=identity_summary,
    )
    return {
        "state_matched_pairs": state_match_frame,
        "density_matched_pairs": density_match_frame,
        "no_level_matched_pairs": no_level_match_frame,
        "state_control_summary": state_summary,
        "density_control_summary": density_summary,
        "no_level_control_summary": no_level_summary,
        "identity_ablation_summary": identity_summary,
        "btc_descriptive_separation": btc_descriptive_rows(banded, candidates=candidates),
        "candidate_verdicts": verdicts,
    }


def candidate_key(candidate: pd.Series | dict[str, Any]) -> str:
    return "|".join(
        (
            str(candidate["analysis_scope"]),
            str(candidate["feature"]),
            str(candidate["outcome"]),
            str(candidate["early_orientation"]),
        )
    )


def candidate_periods(candidate: pd.Series | dict[str, Any]) -> tuple[str, ...]:
    periods = tuple(
        value for value in str(candidate["validation_periods"]).split(";") if value
    )
    if len(periods) != 2:
        raise ValueError(f"A G4C candidate does not have two validation periods: {periods}")
    return periods


def band_comparisons(orientation: str) -> tuple[tuple[str, str, str], ...]:
    if orientation == "increasing":
        return (
            ("low_to_middle", "middle", "low"),
            ("middle_to_high", "high", "middle"),
        )
    if orientation == "decreasing":
        return (
            ("high_to_middle", "middle", "high"),
            ("middle_to_low", "low", "middle"),
        )
    raise ValueError(f"Unsupported monotonic orientation: {orientation}")


def candidate_rows(banded: DataFrame, candidate: pd.Series) -> DataFrame:
    periods = set(candidate_periods(candidate))
    selected = banded.loc[
        banded["analysis_scope"].eq(str(candidate["analysis_scope"]))
        & banded["feature"].eq(str(candidate["feature"]))
        & banded["period"].astype(str).isin(periods)
        & banded["period_role"].eq(VALIDATION_ROLE)
        & banded["feature_band"].notna()
    ].copy()
    if selected.empty:
        raise ValueError(f"No banded validation rows for {candidate_key(candidate)}")
    return selected


def adjacent_geometry_matches(
    cell: DataFrame,
    *,
    candidate: pd.Series,
    state_columns: Sequence[str],
    control_name: str,
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    outcome = str(candidate["outcome"])
    for period in candidate_periods(candidate):
        period_rows = cell.loc[cell["period"].astype(str).eq(period)]
        for comparison, stronger_band, weaker_band in band_comparisons(
            str(candidate["early_orientation"])
        ):
            for group_key, group in period_rows.groupby(
                ["pair", "level_name", "approach_state"],
                observed=True,
                sort=False,
            ):
                stronger = group.loc[group["feature_band"].eq(stronger_band)].reset_index(
                    drop=True
                )
                weaker = group.loc[group["feature_band"].eq(weaker_band)].reset_index(
                    drop=True
                )
                matches, _ = nearest_state_pairs(
                    stronger,
                    weaker,
                    state_columns=state_columns,
                    pre_distance_atr_caliper=PRE_DISTANCE_ATR_CALIPER,
                    minimum_event_separation_hours=MINIMUM_EVENT_SEPARATION_HOURS,
                )
                for stronger_position, weaker_position, distance in matches:
                    left = stronger.iloc[stronger_position]
                    right = weaker.iloc[weaker_position]
                    row: dict[str, Any] = {
                        "candidate_id": candidate_key(candidate),
                        "analysis_scope": str(candidate["analysis_scope"]),
                        "feature": str(candidate["feature"]),
                        "outcome": outcome,
                        "orientation": str(candidate["early_orientation"]),
                        "control": control_name,
                        "period": period,
                        "comparison": comparison,
                        "stronger_band": stronger_band,
                        "weaker_band": weaker_band,
                        "pair": str(group_key[0]),
                        "level_name": str(group_key[1]),
                        "approach_state": str(group_key[2]),
                        "stronger_event_time": left["event_time"],
                        "weaker_event_time": right["event_time"],
                        "state_distance": float(distance),
                        "stronger_outcome": float(left[outcome]),
                        "weaker_outcome": float(right[outcome]),
                        "oriented_delta": float(left[outcome] - right[outcome]),
                        "direction_prediction": False,
                        "profit_optimization": False,
                    }
                    for column in state_columns:
                        row[f"stronger_state__{column}"] = float(left[column])
                        row[f"weaker_state__{column}"] = float(right[column])
                    rows.append(row)
    return DataFrame(rows)


def same_state_no_level_matches(
    cell: DataFrame,
    *,
    no_level: DataFrame,
    candidate: pd.Series,
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    outcome = str(candidate["outcome"])
    for period in candidate_periods(candidate):
        period_actual = cell.loc[cell["period"].astype(str).eq(period)]
        period_control = no_level.loc[
            no_level["analysis_scope"].eq(str(candidate["analysis_scope"]))
            & no_level["period"].astype(str).eq(period)
            & no_level["period_role"].eq(VALIDATION_ROLE)
        ]
        for band in ("low", "middle", "high"):
            band_actual = period_actual.loc[period_actual["feature_band"].eq(band)]
            for group_key, actual_group in band_actual.groupby(
                ["pair", "level_name", "approach_state"],
                observed=True,
                sort=False,
            ):
                control_group = period_control.loc[
                    period_control["pair"].eq(group_key[0])
                    & period_control["level_name"].eq(group_key[1])
                    & period_control["approach_state"].eq(group_key[2])
                ]
                actual_group = actual_group.reset_index(drop=True)
                control_group = control_group.reset_index(drop=True)
                matches, _ = nearest_state_pairs(
                    actual_group,
                    control_group,
                    state_columns=STATE_CONTROL_FEATURES,
                    pre_distance_atr_caliper=PRE_DISTANCE_ATR_CALIPER,
                    minimum_event_separation_hours=MINIMUM_EVENT_SEPARATION_HOURS,
                )
                for actual_position, control_position, distance in matches:
                    left = actual_group.iloc[actual_position]
                    right = control_group.iloc[control_position]
                    row: dict[str, Any] = {
                        "candidate_id": candidate_key(candidate),
                        "analysis_scope": str(candidate["analysis_scope"]),
                        "feature": str(candidate["feature"]),
                        "outcome": outcome,
                        "orientation": str(candidate["early_orientation"]),
                        "control": "same_state_no_level",
                        "period": period,
                        "feature_band": band,
                        "pair": str(group_key[0]),
                        "level_name": str(group_key[1]),
                        "approach_state": str(group_key[2]),
                        "actual_event_time": left["event_time"],
                        "no_level_event_time": right["event_time"],
                        "state_distance": float(distance),
                        "actual_outcome": float(left[outcome]),
                        "no_level_outcome": float(right[outcome]),
                        "actual_minus_no_level": float(left[outcome] - right[outcome]),
                        "direction_prediction": False,
                        "profit_optimization": False,
                    }
                    for column in STATE_CONTROL_FEATURES:
                        row[f"actual_state__{column}"] = float(left[column])
                        row[f"no_level_state__{column}"] = float(right[column])
                    rows.append(row)
    return DataFrame(rows)


def summarize_geometry_controls(
    matches: DataFrame,
    *,
    candidate: pd.Series,
    state_columns: Sequence[str],
    permutations: int,
    control_name: str,
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for period in candidate_periods(candidate):
        for comparison, stronger_band, weaker_band in band_comparisons(
            str(candidate["early_orientation"])
        ):
            if matches.empty:
                group = DataFrame()
            else:
                group = matches.loc[
                    matches["period"].astype(str).eq(period)
                    & matches["comparison"].eq(comparison)
                ].copy()
            rows.append(
                geometry_control_cell(
                    group,
                    candidate=candidate,
                    period=period,
                    comparison=comparison,
                    stronger_band=stronger_band,
                    weaker_band=weaker_band,
                    state_columns=state_columns,
                    permutations=permutations,
                    control_name=control_name,
                )
            )
    return DataFrame(rows)


def geometry_control_cell(
    group: DataFrame,
    *,
    candidate: pd.Series,
    period: str,
    comparison: str,
    stronger_band: str,
    weaker_band: str,
    state_columns: Sequence[str],
    permutations: int,
    control_name: str,
) -> dict[str, Any]:
    valid = (
        group.loc[np.isfinite(pd.to_numeric(group["oriented_delta"], errors="coerce"))].copy()
        if not group.empty
        else DataFrame()
    )
    if valid.empty:
        pair_means = pd.Series(dtype=float)
        balance = {
            "features_scored": 0,
            "max_absolute_smd": np.nan,
            "median_absolute_smd": np.nan,
        }
        observed = np.nan
        permutation_p = np.nan
        null_q95 = np.nan
    else:
        pair_means = valid.groupby("pair", observed=True)["oriented_delta"].mean()
        stronger_state = DataFrame(
            {
                column: valid[f"stronger_state__{column}"].to_numpy()
                for column in state_columns
            }
        )
        weaker_state = DataFrame(
            {
                column: valid[f"weaker_state__{column}"].to_numpy()
                for column in state_columns
            }
        )
        balance = balance_diagnostics(stronger_state, weaker_state, state_columns)
        observed, permutation_p, null_q95 = matched_permutation_test(
            valid,
            permutations=permutations,
            seed_key=f"{candidate_key(candidate)}|{control_name}|{period}|{comparison}",
        )
    balance_pass = bool(
        balance["features_scored"] >= 4
        and np.isfinite(balance["max_absolute_smd"])
        and balance["max_absolute_smd"] <= MAX_STATE_SMD
        and np.isfinite(balance["median_absolute_smd"])
        and balance["median_absolute_smd"] <= MAX_MEDIAN_STATE_SMD
    )
    positive_coins = int(pair_means.gt(0.0).sum())
    support_pass = bool(
        len(valid) >= MIN_MATCHED_PAIRS
        and len(pair_means) >= MIN_MATCHED_COINS
        and balance_pass
    )
    effect_pass = bool(
        support_pass
        and np.isfinite(observed)
        and observed > 0.0
        and positive_coins >= MIN_MATCHED_COINS
        and np.isfinite(permutation_p)
        and permutation_p <= PERMUTATION_ALPHA
    )
    return {
        "candidate_id": candidate_key(candidate),
        "analysis_scope": str(candidate["analysis_scope"]),
        "feature": str(candidate["feature"]),
        "outcome": str(candidate["outcome"]),
        "orientation": str(candidate["early_orientation"]),
        "control": control_name,
        "period": period,
        "comparison": comparison,
        "stronger_band": stronger_band,
        "weaker_band": weaker_band,
        "matched_pairs": len(valid),
        "matched_coins": len(pair_means),
        "positive_effect_coins": positive_coins,
        "positive_effect_coin_fraction": (
            float(pair_means.gt(0.0).mean()) if len(pair_means) else np.nan
        ),
        "equal_coin_oriented_delta_median": observed,
        "equal_coin_oriented_delta_q25": (
            float(pair_means.quantile(0.25)) if len(pair_means) else np.nan
        ),
        "equal_coin_oriented_delta_q75": (
            float(pair_means.quantile(0.75)) if len(pair_means) else np.nan
        ),
        "state_features_scored": int(balance["features_scored"]),
        "max_absolute_state_smd": balance["max_absolute_smd"],
        "median_absolute_state_smd": balance["median_absolute_smd"],
        "state_balance_pass": balance_pass,
        "support_pass": support_pass,
        "matched_geometry_permutation_p_one_sided": permutation_p,
        "matched_geometry_null_q95": null_q95,
        "control_pass": effect_pass,
        "direction_prediction": False,
        "profit_optimization": False,
    }


def matched_permutation_test(
    matches: DataFrame, *, permutations: int, seed_key: str
) -> tuple[float, float, float]:
    selected = matches.loc[
        np.isfinite(pd.to_numeric(matches["oriented_delta"], errors="coerce")),
        ["pair", "oriented_delta"],
    ].copy()
    if selected.empty:
        return np.nan, np.nan, np.nan
    values = selected["oriented_delta"].to_numpy(dtype=float)
    pair_codes, pair_names = pd.factorize(selected["pair"].astype(str), sort=True)
    pair_counts = np.bincount(pair_codes).astype(float)

    def equal_coin_median(candidate_values: np.ndarray) -> float:
        sums = np.bincount(pair_codes, weights=candidate_values, minlength=len(pair_names))
        return float(np.median(sums / pair_counts))

    observed = equal_coin_median(values)
    rng = np.random.default_rng(stable_hash_int(seed_key))
    null = np.empty(permutations, dtype=float)
    for index in range(permutations):
        signs = rng.choice(np.asarray((-1.0, 1.0)), size=len(values), replace=True)
        null[index] = equal_coin_median(values * signs)
    p_value = float((1 + np.count_nonzero(null >= observed)) / (permutations + 1))
    return observed, p_value, float(np.quantile(null, 0.95))


def summarize_no_level_controls(matches: DataFrame, *, candidate: pd.Series) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for period in candidate_periods(candidate):
        for comparison, stronger_band, weaker_band in band_comparisons(
            str(candidate["early_orientation"])
        ):
            period_rows = (
                matches.loc[matches["period"].astype(str).eq(period)].copy()
                if not matches.empty
                else DataFrame()
            )
            rows.append(
                no_level_control_cell(
                    period_rows,
                    candidate=candidate,
                    period=period,
                    comparison=comparison,
                    stronger_band=stronger_band,
                    weaker_band=weaker_band,
                )
            )
    return DataFrame(rows)


def no_level_control_cell(
    period_rows: DataFrame,
    *,
    candidate: pd.Series,
    period: str,
    comparison: str,
    stronger_band: str,
    weaker_band: str,
) -> dict[str, Any]:
    band_records: dict[str, dict[str, Any]] = {}
    band_pair_means: dict[str, pd.Series] = {}
    for band in (stronger_band, weaker_band):
        group = (
            period_rows.loc[period_rows["feature_band"].eq(band)].copy()
            if not period_rows.empty
            else DataFrame()
        )
        if group.empty:
            valid = DataFrame()
            balance = {
                "features_scored": 0,
                "max_absolute_smd": np.nan,
                "median_absolute_smd": np.nan,
            }
            pair_means = pd.Series(dtype=float)
        else:
            valid = group.loc[
                np.isfinite(pd.to_numeric(group["actual_minus_no_level"], errors="coerce"))
            ].copy()
            pair_means = valid.groupby("pair", observed=True)["actual_minus_no_level"].mean()
            actual_state = DataFrame(
                {
                    column: valid[f"actual_state__{column}"].to_numpy()
                    for column in STATE_CONTROL_FEATURES
                }
            )
            control_state = DataFrame(
                {
                    column: valid[f"no_level_state__{column}"].to_numpy()
                    for column in STATE_CONTROL_FEATURES
                }
            )
            balance = balance_diagnostics(
                actual_state,
                control_state,
                STATE_CONTROL_FEATURES,
            )
        balance_pass = bool(
            balance["features_scored"] >= 4
            and np.isfinite(balance["max_absolute_smd"])
            and balance["max_absolute_smd"] <= MAX_STATE_SMD
            and np.isfinite(balance["median_absolute_smd"])
            and balance["median_absolute_smd"] <= MAX_MEDIAN_STATE_SMD
        )
        support_pass = bool(
            len(valid) >= MIN_MATCHED_PAIRS
            and len(pair_means) >= MIN_MATCHED_COINS
            and balance_pass
        )
        band_pair_means[band] = pair_means
        band_records[band] = {
            "matches": len(valid),
            "coins": len(pair_means),
            "features_scored": int(balance["features_scored"]),
            "max_smd": balance["max_absolute_smd"],
            "median_smd": balance["median_absolute_smd"],
            "balance_pass": balance_pass,
            "support_pass": support_pass,
        }

    joined = pd.concat(
        (
            band_pair_means[stronger_band].rename("stronger"),
            band_pair_means[weaker_band].rename("weaker"),
        ),
        axis=1,
        join="inner",
    ).dropna()
    contrast = joined["stronger"] - joined["weaker"] if not joined.empty else pd.Series(dtype=float)
    positive_coins = int(contrast.gt(0.0).sum())
    support_pass = bool(
        band_records[stronger_band]["support_pass"]
        and band_records[weaker_band]["support_pass"]
        and len(contrast) >= MIN_MATCHED_COINS
    )
    effect = float(contrast.median()) if len(contrast) else np.nan
    control_pass = bool(
        support_pass
        and np.isfinite(effect)
        and effect > 0.0
        and positive_coins >= MIN_MATCHED_COINS
    )
    return {
        "candidate_id": candidate_key(candidate),
        "analysis_scope": str(candidate["analysis_scope"]),
        "feature": str(candidate["feature"]),
        "outcome": str(candidate["outcome"]),
        "orientation": str(candidate["early_orientation"]),
        "control": "same_state_no_level",
        "period": period,
        "comparison": comparison,
        "stronger_band": stronger_band,
        "weaker_band": weaker_band,
        "stronger_band_matches": band_records[stronger_band]["matches"],
        "weaker_band_matches": band_records[weaker_band]["matches"],
        "stronger_band_coins": band_records[stronger_band]["coins"],
        "weaker_band_coins": band_records[weaker_band]["coins"],
        "coins_with_both_adjusted_bands": len(contrast),
        "positive_adjusted_contrast_coins": positive_coins,
        "positive_adjusted_coin_fraction": (
            float(contrast.gt(0.0).mean()) if len(contrast) else np.nan
        ),
        "equal_coin_adjusted_contrast_median": effect,
        "stronger_band_max_state_smd": band_records[stronger_band]["max_smd"],
        "weaker_band_max_state_smd": band_records[weaker_band]["max_smd"],
        "stronger_band_median_state_smd": band_records[stronger_band]["median_smd"],
        "weaker_band_median_state_smd": band_records[weaker_band]["median_smd"],
        "state_balance_pass": bool(
            band_records[stronger_band]["balance_pass"]
            and band_records[weaker_band]["balance_pass"]
        ),
        "support_pass": support_pass,
        "control_pass": control_pass,
        "direction_prediction": False,
        "profit_optimization": False,
    }


def summarize_identity_ablation(matches: DataFrame, *, candidate: pd.Series) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for period in candidate_periods(candidate):
        for comparison, stronger_band, weaker_band in band_comparisons(
            str(candidate["early_orientation"])
        ):
            for identity in SOURCE_LEVEL_NAMES:
                group = (
                    matches.loc[
                        matches["period"].astype(str).eq(period)
                        & matches["comparison"].eq(comparison)
                        & matches["level_name"].eq(identity)
                    ].copy()
                    if not matches.empty
                    else DataFrame()
                )
                valid = (
                    group.loc[
                        np.isfinite(pd.to_numeric(group["oriented_delta"], errors="coerce"))
                    ].copy()
                    if not group.empty
                    else DataFrame()
                )
                pair_means = (
                    valid.groupby("pair", observed=True)["oriented_delta"].mean()
                    if not valid.empty
                    else pd.Series(dtype=float)
                )
                effect = float(pair_means.median()) if len(pair_means) else np.nan
                positive_fraction = (
                    float(pair_means.gt(0.0).mean()) if len(pair_means) else np.nan
                )
                support_pass = bool(
                    len(valid) >= MIN_IDENTITY_PAIRS
                    and len(pair_means) >= MIN_IDENTITY_COINS
                )
                control_pass = bool(
                    support_pass
                    and np.isfinite(effect)
                    and effect > 0.0
                    and positive_fraction >= MIN_POSITIVE_IDENTITY_FRACTION
                )
                rows.append(
                    {
                        "candidate_id": candidate_key(candidate),
                        "analysis_scope": str(candidate["analysis_scope"]),
                        "feature": str(candidate["feature"]),
                        "outcome": str(candidate["outcome"]),
                        "orientation": str(candidate["early_orientation"]),
                        "control": "named_level_identity_ablation",
                        "period": period,
                        "comparison": comparison,
                        "stronger_band": stronger_band,
                        "weaker_band": weaker_band,
                        "level_name": identity,
                        "matched_pairs": len(valid),
                        "matched_coins": len(pair_means),
                        "positive_effect_coins": int(pair_means.gt(0.0).sum()),
                        "positive_effect_coin_fraction": positive_fraction,
                        "equal_coin_oriented_delta_median": effect,
                        "support_pass": support_pass,
                        "control_pass": control_pass,
                        "direction_prediction": False,
                        "profit_optimization": False,
                    }
                )
    return DataFrame(rows)


def candidate_verdicts(
    *,
    candidates: DataFrame,
    state_summary: DataFrame,
    density_summary: DataFrame,
    no_level_summary: DataFrame,
    identity_summary: DataFrame,
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        identifier = candidate_key(candidate)
        state = state_summary.loc[state_summary["candidate_id"].eq(identifier)]
        density = density_summary.loc[density_summary["candidate_id"].eq(identifier)]
        no_level = no_level_summary.loc[no_level_summary["candidate_id"].eq(identifier)]
        identity = identity_summary.loc[identity_summary["candidate_id"].eq(identifier)]
        expected_adjacent_cells = len(candidate_periods(candidate)) * len(
            band_comparisons(str(candidate["early_orientation"]))
        )
        expected_identity_cells = expected_adjacent_cells * len(SOURCE_LEVEL_NAMES)
        state_pass = complete_control_pass(state, expected=expected_adjacent_cells)
        density_pass = complete_control_pass(density, expected=expected_adjacent_cells)
        no_level_pass = complete_control_pass(no_level, expected=expected_adjacent_cells)
        identity_pass = complete_control_pass(identity, expected=expected_identity_cells)
        retained = bool(state_pass and density_pass and no_level_pass and identity_pass)
        failed = [
            name
            for name, passed in (
                ("ohlcv_and_broad_market_state_only", state_pass),
                ("total_density_without_geometry", density_pass),
                ("same_state_no_level", no_level_pass),
                ("named_level_identity_ablation", identity_pass),
            )
            if not passed
        ]
        rows.append(
            {
                "candidate_id": identifier,
                "analysis_scope": str(candidate["analysis_scope"]),
                "feature": str(candidate["feature"]),
                "outcome": str(candidate["outcome"]),
                "orientation": str(candidate["early_orientation"]),
                "raw_repeated_monotonic_lead": True,
                "state_and_permutation_controls_pass": state_pass,
                "density_and_permutation_controls_pass": density_pass,
                "same_state_no_level_control_pass": no_level_pass,
                "named_level_identity_ablation_pass": identity_pass,
                "retained_after_all_controls": retained,
                "failed_or_unsupported_controls": ";".join(failed),
                "classification": (
                    "controlled_continuous_geometry_lead"
                    if retained
                    else "raw_relationship_explained_or_control_support_incomplete"
                ),
                "direction_prediction": False,
                "profit_optimization": False,
            }
        )
    return DataFrame(rows)


def complete_control_pass(frame: DataFrame, *, expected: int) -> bool:
    return bool(
        len(frame) == expected
        and frame["support_pass"].fillna(False).all()
        and frame["control_pass"].fillna(False).all()
    )


def btc_descriptive_rows(banded: DataFrame, *, candidates: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    btc = banded.loc[
        banded["analysis_scope"].eq("btc")
        & banded["period_role"].eq(VALIDATION_ROLE)
    ]
    if btc.empty:
        return DataFrame(
            columns=[
                "candidate_id",
                "feature",
                "outcome",
                "period",
                "feature_band",
                "events",
                "outcome_mean",
            ]
        )
    for _, candidate in candidates.iterrows():
        selected = btc.loc[btc["feature"].eq(str(candidate["feature"]))]
        outcome = str(candidate["outcome"])
        for (period, band), group in selected.groupby(
            ["period", "feature_band"], observed=True
        ):
            values = pd.to_numeric(group[outcome], errors="coerce")
            rows.append(
                {
                    "candidate_id": candidate_key(candidate),
                    "feature": str(candidate["feature"]),
                    "outcome": outcome,
                    "period": str(period),
                    "feature_band": str(band),
                    "events": len(group),
                    "outcome_mean": float(values.mean()),
                    "outcome_median": float(values.median()),
                    "descriptive_only_single_coin": True,
                    "direction_prediction": False,
                    "profit_optimization": False,
                }
            )
    return DataFrame(rows)


def concat_or_empty(frames: Sequence[DataFrame]) -> DataFrame:
    populated = [frame for frame in frames if not frame.empty]
    return pd.concat(populated, ignore_index=True) if populated else DataFrame()


def control_integrity(
    *,
    actual: DataFrame,
    no_level: DataFrame,
    banded: DataFrame,
    candidates: DataFrame,
    outputs: dict[str, DataFrame],
    pairs: Sequence[str],
) -> dict[str, Any]:
    verdicts = outputs["candidate_verdicts"]
    missing_pairs = sorted(set(pairs) - set(actual["pair"].astype(str)))
    missing_no_level_pairs = sorted(set(pairs) - set(no_level["pair"].astype(str)))
    missing_outputs = sorted(
        name
        for name in (
            "state_control_summary",
            "density_control_summary",
            "no_level_control_summary",
            "identity_ablation_summary",
            "candidate_verdicts",
        )
        if name not in outputs or outputs[name].empty
    )
    direction_violations = int(actual["direction_prediction"].ne(False).sum()) + int(
        no_level["direction_prediction"].ne(False).sum()
    )
    profit_violations = int(actual["profit_optimization"].ne(False).sum()) + int(
        no_level["profit_optimization"].ne(False).sum()
    )
    candidate_ids = {candidate_key(row) for _, row in candidates.iterrows()}
    verdict_ids = set(verdicts["candidate_id"].astype(str)) if not verdicts.empty else set()
    return {
        "passed": bool(
            not missing_pairs
            and not missing_no_level_pairs
            and not missing_outputs
            and not actual.empty
            and not no_level.empty
            and not banded.empty
            and len(verdicts) == len(candidates)
            and candidate_ids == verdict_ids
            and direction_violations == 0
            and profit_violations == 0
        ),
        "actual_rows": len(actual),
        "no_level_rows": len(no_level),
        "banded_rows": len(banded),
        "candidate_rows": len(candidates),
        "verdict_rows": len(verdicts),
        "missing_actual_pairs": missing_pairs,
        "missing_no_level_pairs": missing_no_level_pairs,
        "missing_outputs": missing_outputs,
        "direction_violations": direction_violations,
        "profit_violations": profit_violations,
        "retained_cells": int(verdicts["retained_after_all_controls"].sum()),
        "direction_prediction": False,
        "profit_optimization": False,
    }


if __name__ == "__main__":
    raise SystemExit(main())

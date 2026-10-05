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
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    LARGE_ARTIFACT_ROOT,
    OUTPUT_ROOT,
    atomic_write_json,
    atomic_write_parquet,
    load_manifest,
    load_ohlcv,
    normalize_dates,
    numeric_array,
    ohlcv_path,
    prepare_base_market_frame,
    sha256_file,
    stable_hash_int,
    timeframe_delta,
    utc_now,
    validate_worker_count,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_localization import (  # noqa: E501
    balance_diagnostics,
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
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_generic_levels import (  # noqa: E501
    aligned_source_frame,
    indicator_source_levels,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_generic_levels_artificial_controls import (  # noqa: E501
    ARTIFACT_ROOT as G3F_ARTIFICIAL_ARTIFACT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_generic_levels_artificial_controls import (  # noqa: E501
    REPORT_ROOT as G3F_ARTIFICIAL_REPORT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_generic_levels_reaction import (  # noqa: E501
    ARTIFACT_ROOT as G3F_DIRECT_ARTIFACT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation3_generic_levels_reaction import (  # noqa: E501
    REPORT_ROOT as G3F_DIRECT_REPORT_ROOT,
)


OUTPUT_SCHEMA_VERSION = 1
FROZEN_BATCH = OUTPUT_ROOT / "generation3_review" / "g4_frozen_branch_batch.json"
REPORT_ROOT = (
    OUTPUT_ROOT / "generation4_branches" / "g4e_generic_level_representation_decomposition"
)
ARTIFACT_ROOT = (
    LARGE_ARTIFACT_ROOT / "generation4_branches" / "g4e_generic_level_representation_decomposition"
)

SOURCE_TIMEFRAMES = ("1h", "4h", "8h")
TIMEFRAME_HOURS = {"1h": 1, "4h": 4, "8h": 8}
PRICE_DISTRIBUTION_HISTORY_HOURS = 168
MIN_PAIR_PERIOD_EVENTS = 5
MIN_COHORT_PERIOD_EVENTS = 50
MIN_COHORT_PERIOD_COINS = 5
REPRESENTATION_CALIPERS = {
    "distribution_stretch_percentile": 0.15,
    "ma_bundle_distance_atr": 0.25,
}
SMA50_BB_POSITION_CALIPER = 0.15
SMA50_EMA50_GAP_ATR_CALIPER = 0.25
PERMUTATIONS = 1024
VALIDATION_ROLE = "chronological_internal_validation"

QUESTION_BOLLINGER = "bollinger_contact_vs_distribution_stretch"
QUESTION_MA_BUNDLE = "ma_contact_vs_ma_bundle_distance"
QUESTION_SMA50 = "isolated_1h_sma50_low_volume_partial"
QUESTIONS = (QUESTION_BOLLINGER, QUESTION_MA_BUNDLE, QUESTION_SMA50)

DIRECT_NO_LEVEL = "same_state_same_density_no_level"
COMPONENT_ABLATION = "component_ablation_isolated_anchor"
SHIFT_MINUS = "shift_-1atr"
SHIFT_PLUS = "shift_+1atr"
STALE = "stale_same_nominal_history"
IDENTITY_SHUFFLE = "shuffled_indicator_identity"
PRIOR_HIGH = "simple_prior_high_same_history"
PRIOR_LOW = "simple_prior_low_same_history"
LOCATION_CONTROLS = (
    DIRECT_NO_LEVEL,
    SHIFT_MINUS,
    SHIFT_PLUS,
    STALE,
    IDENTITY_SHUFFLE,
    PRIOR_HIGH,
    PRIOR_LOW,
)
ALL_CONTROLS = (*LOCATION_CONTROLS, COMPONENT_ABLATION)

OUTCOMES = {
    "contact_volume_ratio": {
        "response_window": "h4",
        "plain_language": "Contact-hour volume divided by its earlier causal median.",
    },
    "contact_range_ratio": {
        "response_window": "h4",
        "plain_language": "Contact-hour high-low range divided by its earlier causal median.",
    },
    "abs_excursion_atr_h1": {
        "response_window": "h1",
        "plain_language": "Largest next-hour move either way, divided by prior ATR.",
    },
    "range_ratio_h4": {
        "response_window": "h4",
        "plain_language": "Mean next-four-hour range divided by its earlier causal median.",
    },
    "volume_ratio_h4": {
        "response_window": "h4",
        "plain_language": "Mean next-four-hour volume divided by its earlier causal median.",
    },
}

BOLLINGER_STATE_FEATURES = tuple(
    feature for feature in CORE_STATE_FEATURES if feature != "state_local_bb_position"
)
MA_STATE_FEATURES = tuple(
    feature for feature in CORE_STATE_FEATURES if feature != "state_local_ema50_gap_atr"
)

BASE_MATCH_COLUMNS = (
    "pair",
    "density_family",
    "level_name",
    "source_timeframe",
    "actual_scope",
    "actual_generic_scope",
    "actual_cluster_relationship",
    "actual_cluster_dependency_relationship",
    "control",
    "period",
    "response_window",
    "actual_event_time",
    "control_event_time",
    "actual_base_index",
    "control_base_index",
)


@dataclass(frozen=True)
class PairTask:
    pair: str
    manifest_path: str
    run_id: str
    request_sha256: str
    overwrite: bool


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 4E decomposition of exact generic-indicator contacts from "
            "their broader causal market-state representations."
        )
    )
    parser.add_argument("--phase", choices=("preflight", "analyze", "review"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cohort", choices=("large", "meme"))
    parser.add_argument("--g3f-direct-run-id")
    parser.add_argument("--g3f-artificial-run-id")
    parser.add_argument("--preflight-run-id")
    parser.add_argument("--normal-run-id")
    parser.add_argument("--meme-run-id")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.phase == "preflight":
        required_args(args, "manifest", "cohort", "g3f_direct_run_id", "g3f_artificial_run_id")
        return run_preflight(args)
    if args.phase == "analyze":
        required_args(args, "preflight_run_id")
        return run_analysis(args)
    required_args(args, "normal_run_id", "meme_run_id")
    return run_review(args)


def required_args(args: argparse.Namespace, *names: str) -> None:
    missing = [name for name in names if getattr(args, name) in (None, "")]
    if missing:
        raise ValueError(f"Phase {args.phase} requires: {', '.join(missing)}")


def run_preflight(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    validate_worker_count(args.workers, manifest=manifest)
    frozen_branch = validate_frozen_branch()
    pairs = tuple(manifest["data"]["pairs"])
    period_map = canonical_period_map(manifest)
    sources = validate_g3f_sources(
        cohort=args.cohort,
        pairs=pairs,
        direct_run_id=args.g3f_direct_run_id,
        artificial_run_id=args.g3f_artificial_run_id,
    )
    request = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "frozen_batch_path": str(FROZEN_BATCH),
        "frozen_batch_sha256": sha256_file(FROZEN_BATCH),
        "frozen_branch": frozen_branch,
        "cohort": args.cohort,
        "pairs": list(pairs),
        "g3f_direct_run_id": args.g3f_direct_run_id,
        "g3f_artificial_run_id": args.g3f_artificial_run_id,
        "source_contracts": sources,
        "questions": list(QUESTIONS),
        "source_timeframes": list(SOURCE_TIMEFRAMES),
        "canonical_period_map": period_map,
        "representations": {
            "distribution_stretch_percentile": (
                "Absolute distance from the middle of the trailing 168-clock-hour "
                "causal source-timeframe close-rank distribution."
            ),
            "ma_bundle_distance_atr": (
                "Prior completed one-hour close's minimum distance to a causal SMA/EMA "
                "on the contacted source timeframe, divided by prior one-hour ATR."
            ),
        },
        "development_band_rule": (
            "Freeze one-third and two-third edges from all development observations "
            "of normal altcoins or memes, separately by representation and timeframe. "
            "BTC uses normal-alt edges and is descriptive."
        ),
        "common_support_calipers": REPRESENTATION_CALIPERS,
        "sma50_extra_common_support": {
            "bollinger_position_absolute_difference": SMA50_BB_POSITION_CALIPER,
            "ema50_gap_atr_absolute_difference": SMA50_EMA50_GAP_ATR_CALIPER,
        },
        "outcomes_opened": False,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    request_sha256 = stable_json_sha256(request)
    report_dir = REPORT_ROOT / args.run_id
    artifact_dir = ARTIFACT_ROOT / args.run_id
    record_path = report_dir / "g4e_preflight_run_record.json"
    integrity_path = report_dir / "g4e_preflight_integrity.json"
    if record_path.is_file() and not args.overwrite:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") == "completed" and record.get("request_sha256") == request_sha256:
            print(json.dumps(record, indent=2, sort_keys=True))
            return 0
        raise FileExistsError(record_path)

    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    tasks = [
        PairTask(
            pair=pair,
            manifest_path=str(manifest_path),
            run_id=args.run_id,
            request_sha256=request_sha256,
            overwrite=args.overwrite,
        )
        for pair in pairs
    ]
    results = run_pair_tasks(tasks, workers=args.workers)
    failures = [row for row in results if row.get("status") == "failed"]
    if failures:
        raise RuntimeError(f"G4E representation preflight pair failures: {failures}")
    states = read_pair_states(artifact_dir, pairs, request_sha256=request_sha256)
    states["period"] = canonical_periods(states["period"], period_map)
    states["analysis_scope"] = analysis_scope(states, cohort=args.cohort)
    edges = freeze_representation_edges(states, cohort=args.cohort)
    atomic_write_parquet(edges, report_dir / "g4e_representation_edges.parquet")
    support = preflight_support(
        pairs=pairs,
        cohort=args.cohort,
        states=states,
        edges=edges,
        direct_run_id=args.g3f_direct_run_id,
        artificial_run_id=args.g3f_artificial_run_id,
        period_map=period_map,
    )
    atomic_write_parquet(support, report_dir / "g4e_common_support.parquet")
    integrity = preflight_integrity(
        pairs=pairs,
        results=results,
        states=states,
        edges=edges,
        support=support,
    )
    atomic_write_json(integrity, integrity_path)
    if not integrity["passed"]:
        raise ValueError(f"G4E preflight integrity failed: {integrity}")
    record = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "status": "completed",
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "workers": args.workers,
        "request_contract": request,
        "request_sha256": request_sha256,
        "pair_tasks": results,
        "representation_rows": len(states),
        "edge_rows": len(edges),
        "support_rows": len(support),
        "outcomes_opened": False,
        "direction_prediction": False,
        "profit_optimization": False,
        "bulky_artifacts": str(artifact_dir),
        "integrity": str(integrity_path),
    }
    atomic_write_json(record, record_path)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def run_analysis(args: argparse.Namespace) -> int:
    preflight_dir = REPORT_ROOT / args.preflight_run_id
    record_path = preflight_dir / "g4e_preflight_run_record.json"
    integrity_path = preflight_dir / "g4e_preflight_integrity.json"
    edges_path = preflight_dir / "g4e_representation_edges.parquet"
    support_path = preflight_dir / "g4e_common_support.parquet"
    for path in (record_path, integrity_path, edges_path, support_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    preflight = json.loads(record_path.read_text(encoding="utf-8"))
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "completed" or not integrity.get("passed"):
        raise ValueError("G4E preflight is not complete and clean.")
    contract = preflight["request_contract"]
    manifest_path = Path(contract["manifest_path"])
    manifest = load_manifest(manifest_path)
    validate_worker_count(args.workers, manifest=manifest)
    pairs = tuple(contract["pairs"])
    cohort = str(contract["cohort"])
    period_map = dict(contract["canonical_period_map"])
    source_artifact_dir = ARTIFACT_ROOT / args.preflight_run_id
    states = read_pair_states(
        source_artifact_dir,
        pairs,
        request_sha256=preflight["request_sha256"],
    )
    states["period"] = canonical_periods(states["period"], period_map)
    states["analysis_scope"] = analysis_scope(states, cohort=cohort)
    edges = pd.read_parquet(edges_path)
    support = pd.read_parquet(support_path)
    opened_cells, parked_preflight = supported_question_timeframes(support)
    if not opened_cells:
        raise ValueError("G4E preflight left no supported question/timeframe for outcomes.")
    request = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "preflight_run_id": args.preflight_run_id,
        "preflight_record_sha256": sha256_file(record_path),
        "preflight_integrity_sha256": sha256_file(integrity_path),
        "preflight_edges_sha256": sha256_file(edges_path),
        "preflight_support_sha256": sha256_file(support_path),
        "cohort": cohort,
        "pairs": list(pairs),
        "g3f_direct_run_id": contract["g3f_direct_run_id"],
        "g3f_artificial_run_id": contract["g3f_artificial_run_id"],
        "outcomes": OUTCOMES,
        "opened_question_timeframes": [
            {"question": question, "source_timeframe": timeframe}
            for question, timeframe in sorted(opened_cells)
        ],
        "preflight_parked_question_timeframes": parked_preflight.to_dict("records"),
        "controls": list(ALL_CONTROLS),
        "exact_line_removed_surface": (
            "Use the matched G3F no-level member at the same broad state and inherited "
            "pseudo-zone geometry, then test representation bands after removing contact."
        ),
        "state_representation_removed": (
            "Report the original state-matched line-control contrast before the new "
            "representation common-support restriction."
        ),
        "representation_common_support": (
            "Require the same frozen representation band and the predeclared numeric "
            "caliper; the SMA50 partial additionally shares Bollinger position and EMA50 gap."
        ),
        "representation_gate": (
            "Development no-level rows freeze orientation. Both exact-contact and "
            "exact-line-removed surfaces must repeat both adjacent steps in both validation "
            "periods after causal state matching, common support, and 1024 matched sign swaps."
        ),
        "exact_residual_gate": (
            "The representation-conditioned exact-minus-control effect must keep its "
            "development sign in both validation periods against no-level, both ATR shifts, "
            "stale history, identity shuffle, and both simple-prior controls."
        ),
        "minimum_matched_pairs": MIN_COHORT_PERIOD_EVENTS,
        "minimum_coins": MIN_COHORT_PERIOD_COINS,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    request_sha256 = stable_json_sha256(request)
    report_dir = REPORT_ROOT / args.run_id
    artifact_dir = ARTIFACT_ROOT / args.run_id
    analysis_record_path = report_dir / "g4e_analysis_run_record.json"
    analysis_integrity_path = report_dir / "g4e_analysis_integrity.json"
    if analysis_record_path.is_file() and not args.overwrite:
        existing = json.loads(analysis_record_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "completed"
            and existing.get("request_sha256") == request_sha256
        ):
            print(json.dumps(existing, indent=2, sort_keys=True))
            return 0
        raise FileExistsError(analysis_record_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    started = utc_now()

    pair_frames: list[DataFrame] = []
    for pair in pairs:
        pair_state = states.loc[states["pair"].eq(pair)].reset_index(drop=True)
        frame = analysis_pairs_for_pair(
            pair=pair,
            cohort=cohort,
            states=pair_state,
            edges=edges,
            direct_run_id=contract["g3f_direct_run_id"],
            artificial_run_id=contract["g3f_artificial_run_id"],
            period_map=period_map,
            opened_cells=opened_cells,
        )
        frame["output_schema_version"] = OUTPUT_SCHEMA_VERSION
        frame["run_request_sha256"] = request_sha256
        output_path = artifact_dir / "pair_decomposition_rows" / f"{pair_stem(pair)}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_parquet(frame, output_path)
        pair_frames.append(frame)
    pairs_long = pd.concat(pair_frames, ignore_index=True)
    exact_summary = exact_residual_summaries(pairs_long)
    exact_assessment = exact_residual_assessment(exact_summary)
    events = representation_event_rows(pairs_long)
    band_summary = representation_band_summary(events)
    representation_controls = representation_state_controls(events, band_summary)
    representation_assessment = assess_representations(
        band_summary=band_summary,
        control_summary=representation_controls,
    )
    component_summary = component_ablation_summary(pairs_long)

    outputs = {
        "g4e_exact_residual_summary.parquet": exact_summary,
        "g4e_exact_residual_assessment.parquet": exact_assessment,
        "g4e_representation_band_summary.parquet": band_summary,
        "g4e_representation_state_controls.parquet": representation_controls,
        "g4e_representation_assessment.parquet": representation_assessment,
        "g4e_independent_component_ablation.parquet": component_summary,
        "g4e_preflight_parked_questions.parquet": parked_preflight,
    }
    for name, frame in outputs.items():
        atomic_write_parquet(frame, report_dir / name)
    analysis_integrity = analysis_integrity_record(
        pairs=pairs,
        pair_rows=pairs_long,
        outputs=outputs,
    )
    atomic_write_json(analysis_integrity, analysis_integrity_path)
    if not analysis_integrity["passed"]:
        raise ValueError(f"G4E analysis integrity failed: {analysis_integrity}")
    analysis_record = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "status": "completed",
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "workers": 1,
        "request_contract": request,
        "request_sha256": request_sha256,
        "pair_decomposition_rows": len(pairs_long),
        "opened_question_timeframes": len(opened_cells),
        "preflight_parked_question_timeframes": len(parked_preflight),
        "representation_cells_retained": int(
            representation_assessment["retained_representation_lead"].sum()
        ),
        "exact_residual_cells_retained": int(
            exact_assessment["retained_exact_location_residual"].sum()
        ),
        "bulky_artifacts": str(artifact_dir),
        "integrity": str(analysis_integrity_path),
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(analysis_record, analysis_record_path)
    print(json.dumps(analysis_record, indent=2, sort_keys=True))
    return 0


def run_review(args: argparse.Namespace) -> int:
    sources: list[tuple[str, Path, dict[str, Any]]] = []
    for label, run_id in (("normal", args.normal_run_id), ("meme", args.meme_run_id)):
        root = REPORT_ROOT / run_id
        record_path = root / "g4e_analysis_run_record.json"
        integrity_path = root / "g4e_analysis_integrity.json"
        if not record_path.is_file() or not integrity_path.is_file():
            raise FileNotFoundError(root)
        record = json.loads(record_path.read_text(encoding="utf-8"))
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        if record.get("status") != "completed" or not integrity.get("passed"):
            raise ValueError(f"G4E {label} analysis is not clean.")
        sources.append((label, root, record))
    representation_frames = []
    exact_frames = []
    contracts = []
    for label, root, record in sources:
        representation = pd.read_parquet(root / "g4e_representation_assessment.parquet")
        exact = pd.read_parquet(root / "g4e_exact_residual_assessment.parquet")
        representation["source_cohort"] = label
        exact["source_cohort"] = label
        representation_frames.append(representation)
        exact_frames.append(exact)
        contracts.append(
            {
                "cohort": label,
                "run_id": record["run_id"],
                "record_sha256": sha256_file(root / "g4e_analysis_run_record.json"),
                "integrity_sha256": sha256_file(root / "g4e_analysis_integrity.json"),
            }
        )
    representation = pd.concat(representation_frames, ignore_index=True)
    exact = pd.concat(exact_frames, ignore_index=True)
    report_dir = REPORT_ROOT / args.run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(representation, report_dir / "g4e_joint_representation_assessment.parquet")
    atomic_write_parquet(exact, report_dir / "g4e_joint_exact_residual_assessment.parquet")
    retained_representation = representation.loc[
        representation["retained_representation_lead"].astype(bool)
    ]
    retained_exact = exact.loc[exact["retained_exact_location_residual"].astype(bool)]
    review = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "source_contracts": contracts,
        "representation_cells_reviewed": len(representation),
        "representation_cells_retained": len(retained_representation),
        "exact_residual_cells_reviewed": len(exact),
        "exact_residual_cells_retained": len(retained_exact),
        "retained_representation_cells": retained_representation[
            ["analysis_scope", "question", "source_timeframe", "outcome"]
        ].to_dict("records"),
        "retained_exact_residual_cells": retained_exact[
            ["analysis_scope", "question", "source_timeframe", "identity", "outcome"]
        ].to_dict("records"),
        "interpretation_boundary": (
            "A retained representation is a direction-neutral market-behaviour lead. "
            "An exact residual is an attribution lead. Neither is a trading rule, "
            "direction forecast, profit claim, or indicator-edit instruction."
        ),
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(review, report_dir / "g4e_joint_review.json")
    print(json.dumps(review, indent=2, sort_keys=True))
    return 0


def validate_frozen_branch() -> dict[str, Any]:
    frozen = json.loads(FROZEN_BATCH.read_text(encoding="utf-8"))
    matches = [
        branch
        for branch in frozen.get("branches", [])
        if branch.get("id") == "g4e_generic_level_representation_decomposition"
    ]
    if len(matches) != 1:
        raise ValueError("The frozen Generation 4 batch lacks exactly one G4E branch.")
    branch = matches[0]
    if branch.get("status") != "frozen_next_batch":
        raise ValueError("G4E is not frozen for execution.")
    if int(branch.get("iteration_cap", 0)) != 4:
        raise ValueError("G4E iteration cap changed from four.")
    expected = {
        "exact_line_removed",
        "state_representation_removed",
        "plus_and_minus_one_atr",
        "stale_nominal_history",
        "identity_shuffle",
        "same_state_no_level",
        "simple_prior_level",
        "independent_cluster_component_ablation",
    }
    if set(branch.get("required_controls", [])) != expected:
        raise ValueError("G4E required controls changed after freezing.")
    return branch


def canonical_period_map(manifest: dict[str, Any]) -> dict[str, str]:
    periods = manifest.get("data", {}).get("chronological_periods", [])
    development = [str(row["id"]) for row in periods if row.get("role") == "development"]
    validations = [str(row["id"]) for row in periods if row.get("role") == VALIDATION_ROLE]
    if len(development) != 1 or len(validations) != 2:
        raise ValueError("G4E requires one development and two chronological validation periods.")
    return {
        development[0]: "development",
        validations[0]: "validation_early",
        validations[1]: "validation_late",
    }


def canonical_periods(values: Series, mapping: dict[str, str]) -> Series:
    source = values.astype(str)
    return source.map(mapping).fillna(source)


def validate_g3f_sources(
    *, cohort: str, pairs: Sequence[str], direct_run_id: str, artificial_run_id: str
) -> list[dict[str, Any]]:
    definitions = (
        (
            "g3f_direct",
            G3F_DIRECT_REPORT_ROOT / direct_run_id / "g3f_reaction_run_record.json",
            G3F_DIRECT_REPORT_ROOT / direct_run_id / "g3f_reaction_integrity.json",
            G3F_DIRECT_ARTIFACT_ROOT / direct_run_id / "pair_independent_matches",
        ),
        (
            "g3f_artificial",
            G3F_ARTIFICIAL_REPORT_ROOT / artificial_run_id / "g3f_artificial_run_record.json",
            G3F_ARTIFICIAL_REPORT_ROOT / artificial_run_id / "g3f_artificial_integrity.json",
            G3F_ARTIFICIAL_ARTIFACT_ROOT / artificial_run_id / "pair_independent_matches",
        ),
    )
    contracts: list[dict[str, Any]] = []
    for label, record_path, integrity_path, pair_root in definitions:
        if not record_path.is_file() or not integrity_path.is_file():
            raise FileNotFoundError(record_path)
        record = json.loads(record_path.read_text(encoding="utf-8"))
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        request = record.get("request_contract", {})
        if record.get("status") != "completed" or not integrity.get("passed"):
            raise ValueError(f"{label} is not complete and clean.")
        if request.get("cohort") != cohort:
            raise ValueError(f"{label} cohort mismatch.")
        if not set(pairs).issubset(request.get("pairs", [])):
            raise ValueError(f"{label} lacks a requested pair.")
        if request.get("direction_prediction") is not False:
            raise ValueError(f"{label} crossed the direction boundary.")
        if request.get("profit_optimization") is not False:
            raise ValueError(f"{label} crossed the profit boundary.")
        contracts.extend(
            (
                {
                    "stage": f"{label}_record",
                    "path": str(record_path),
                    "sha256": sha256_file(record_path),
                },
                {
                    "stage": f"{label}_integrity",
                    "path": str(integrity_path),
                    "sha256": sha256_file(integrity_path),
                },
            )
        )
        for pair in pairs:
            path = pair_root / f"{pair_stem(pair)}.parquet"
            if not path.is_file():
                raise FileNotFoundError(path)
            stat = path.stat()
            contracts.append(
                {
                    "stage": f"{label}_pair_matches",
                    "pair": pair,
                    "path": str(path),
                    "bytes": stat.st_size,
                    "modified_ns": stat.st_mtime_ns,
                }
            )
    return contracts


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
        return build_pair_representation(task)
    except Exception as exc:
        return {"pair": task.pair, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def build_pair_representation(task: PairTask) -> dict[str, Any]:
    started = time.perf_counter()
    output_path = (
        ARTIFACT_ROOT / task.run_id / "pair_representations" / f"{pair_stem(task.pair)}.parquet"
    )
    if output_path.is_file() and not task.overwrite:
        existing = pd.read_parquet(output_path)
        validate_representation_frame(existing, pair=task.pair, request_sha256=task.request_sha256)
        return {
            "pair": task.pair,
            "status": "existing",
            "rows": len(existing),
            "seconds": round(time.perf_counter() - started, 3),
        }
    manifest = load_manifest(Path(task.manifest_path))
    base = prepare_base_market_frame(task.pair, manifest)
    frame = representation_frame(task.pair, base=base)
    frame["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    frame["run_request_sha256"] = task.request_sha256
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(frame, output_path)
    return {
        "pair": task.pair,
        "status": "completed",
        "rows": len(frame),
        "seconds": round(time.perf_counter() - started, 3),
    }


def representation_frame(pair: str, *, base: DataFrame) -> DataFrame:
    output = DataFrame(
        {
            "pair": pair,
            "event_time": normalize_dates(base["date"]),
            "base_index": np.arange(len(base), dtype=np.int64),
            "period": base["period"].astype(str),
            "base_atr": pd.to_numeric(base["base_atr"], errors="coerce"),
            "pre_close": pd.to_numeric(base["pre_close"], errors="coerce"),
        }
    )
    pre_close = numeric_array(base["pre_close"])
    base_atr = numeric_array(base["base_atr"])
    for timeframe in SOURCE_TIMEFRAMES:
        stretch = aligned_distribution_stretch(pair=pair, base=base, timeframe=timeframe)
        levels = [
            level
            for level in indicator_source_levels(pair=pair, base=base, timeframe=timeframe)
            if level.family in {"simple_moving_average", "exponential_moving_average"}
        ]
        matrix = np.column_stack([np.where(level.valid, level.level, np.nan) for level in levels])
        distance = np.abs(matrix - pre_close[:, None])
        all_missing = ~np.isfinite(distance).any(axis=1)
        minimum = np.full(len(base), np.nan, dtype=float)
        valid_rows = ~all_missing & np.isfinite(base_atr) & (base_atr > 0.0)
        minimum[valid_rows] = np.nanmin(distance[valid_rows], axis=1) / base_atr[valid_rows]
        output[f"distribution_stretch_percentile__{timeframe}"] = stretch
        output[f"ma_bundle_distance_atr__{timeframe}"] = minimum
    output["reaction_outcomes_loaded"] = False
    output["direction_prediction"] = False
    output["profit_optimization"] = False
    return output


def aligned_distribution_stretch(*, pair: str, base: DataFrame, timeframe: str) -> np.ndarray:
    source = load_ohlcv(ohlcv_path(pair, timeframe)).sort_values("date").drop_duplicates("date")
    source = source.reset_index(drop=True)
    close = pd.to_numeric(source["close"], errors="coerce")
    history_bars = max(20, PRICE_DISTRIBUTION_HISTORY_HOURS // TIMEFRAME_HOURS[timeframe])
    percentile = close.rolling(history_bars, min_periods=history_bars).rank(
        method="average", pct=True
    )
    source_frame = DataFrame(
        {
            "source_open": normalize_dates(source["date"]),
            "available_at": normalize_dates(source["date"]) + timeframe_delta(timeframe),
            "stretch": (2.0 * percentile - 1.0).abs(),
        }
    )
    aligned = aligned_source_frame(base["date"], source_frame)
    available = normalize_dates(aligned["available_at"])
    event_time = normalize_dates(aligned["event_time"])
    if (available.dropna() > event_time.loc[available.notna()]).any():
        raise ValueError(f"A future {timeframe} distribution value entered G4E.")
    return pd.to_numeric(aligned["stretch"], errors="coerce").to_numpy(dtype=float)


def validate_representation_frame(frame: DataFrame, *, pair: str, request_sha256: str) -> None:
    required = {
        "pair",
        "event_time",
        "base_index",
        "period",
        "reaction_outcomes_loaded",
        "direction_prediction",
        "profit_optimization",
        "output_schema_version",
        "run_request_sha256",
    }
    required.update(
        f"{metric}__{timeframe}"
        for metric in ("distribution_stretch_percentile", "ma_bundle_distance_atr")
        for timeframe in SOURCE_TIMEFRAMES
    )
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"G4E representation frame lacks columns: {missing}")
    if set(frame["pair"].astype(str)) != {pair}:
        raise ValueError("G4E representation pair mismatch.")
    if set(frame["run_request_sha256"].astype(str)) != {request_sha256}:
        raise ValueError("G4E representation request mismatch.")
    if frame["reaction_outcomes_loaded"].ne(False).any():
        raise ValueError("G4E preflight representation opened outcomes.")
    if (
        frame["direction_prediction"].ne(False).any()
        or frame["profit_optimization"].ne(False).any()
    ):
        raise ValueError("G4E representation crossed a research boundary.")


def read_pair_states(artifact_dir: Path, pairs: Sequence[str], *, request_sha256: str) -> DataFrame:
    frames = []
    for pair in pairs:
        path = artifact_dir / "pair_representations" / f"{pair_stem(pair)}.parquet"
        frame = pd.read_parquet(path)
        validate_representation_frame(frame, pair=pair, request_sha256=request_sha256)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def analysis_scope(frame: DataFrame, *, cohort: str) -> Series:
    if cohort == "meme":
        return Series("memes", index=frame.index, dtype="object")
    return Series(
        np.where(frame["pair"].astype(str).eq("BTC/USDT:USDT"), "btc", "normal_alts"),
        index=frame.index,
        dtype="object",
    )


def freeze_representation_edges(states: DataFrame, *, cohort: str) -> DataFrame:
    edge_scope = "memes" if cohort == "meme" else "normal_alts"
    development = states.loc[
        states["period"].astype(str).eq("development") & states["analysis_scope"].eq(edge_scope)
    ]
    if development.empty:
        raise ValueError(f"No {edge_scope} development observations for G4E edges.")
    rows: list[dict[str, Any]] = []
    for timeframe in SOURCE_TIMEFRAMES:
        for metric in ("distribution_stretch_percentile", "ma_bundle_distance_atr"):
            column = f"{metric}__{timeframe}"
            values = pd.to_numeric(development[column], errors="coerce")
            finite = values.loc[np.isfinite(values)]
            low, high = (
                np.quantile(finite.to_numpy(dtype=float), [1 / 3, 2 / 3])
                if len(finite)
                else (np.nan, np.nan)
            )
            rows.append(
                {
                    "edge_scope": edge_scope,
                    "source_timeframe": timeframe,
                    "representation": metric,
                    "development_rows": len(development),
                    "finite_development_rows": len(finite),
                    "development_coins": int(development["pair"].nunique()),
                    "low_upper_edge": float(low),
                    "middle_upper_edge": float(high),
                    "distinct_three_band_edges": bool(
                        np.isfinite(low) and np.isfinite(high) and low < high
                    ),
                    "outcomes_opened": False,
                    "direction_prediction": False,
                    "profit_optimization": False,
                }
            )
    return DataFrame(rows)


def assign_bands(values: Series, edge: pd.Series | dict[str, Any]) -> Series:
    output = Series(pd.NA, index=values.index, dtype="object")
    if not bool(edge["distinct_three_band_edges"]):
        return output
    numeric = pd.to_numeric(values, errors="coerce")
    finite = np.isfinite(numeric)
    low = float(edge["low_upper_edge"])
    middle = float(edge["middle_upper_edge"])
    output.loc[finite & numeric.le(low)] = "low"
    output.loc[finite & numeric.gt(low) & numeric.le(middle)] = "middle"
    output.loc[finite & numeric.gt(middle)] = "high"
    return output


def question_mask(frame: DataFrame, question: str) -> Series:
    timeframe = frame["source_timeframe"].astype(str).isin(SOURCE_TIMEFRAMES)
    if question == QUESTION_BOLLINGER:
        return (
            timeframe
            & frame["density_family"].astype(str).eq("bollinger_band")
            & frame["level_name"].astype(str).isin(("bollinger_20_upper", "bollinger_20_lower"))
        )
    if question == QUESTION_MA_BUNDLE:
        return timeframe & frame["density_family"].astype(str).isin(
            ("simple_moving_average", "exponential_moving_average")
        )
    if question == QUESTION_SMA50:
        return (
            frame["source_timeframe"].astype(str).eq("1h")
            & frame["density_family"].astype(str).eq("simple_moving_average")
            & frame["level_name"].astype(str).eq("sma_50")
            & frame["actual_generic_scope"].astype(str).eq("isolated_generic_level")
        )
    raise ValueError(f"Unknown G4E question: {question}")


def question_representation(question: str) -> str:
    if question == QUESTION_BOLLINGER:
        return "distribution_stretch_percentile"
    return "ma_bundle_distance_atr"


def question_identity(frame: DataFrame, question: str) -> Series:
    if question == QUESTION_MA_BUNDLE:
        return frame["density_family"].astype(str)
    return frame["level_name"].astype(str)


def edge_for(edges: DataFrame, *, timeframe: str, representation: str) -> pd.Series:
    selected = edges.loc[
        edges["source_timeframe"].astype(str).eq(timeframe)
        & edges["representation"].astype(str).eq(representation)
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one G4E edge for {timeframe}/{representation}.")
    return selected.iloc[0]


def attach_question_rows(frame: DataFrame, *, states: DataFrame, edges: DataFrame) -> DataFrame:
    frames: list[DataFrame] = []
    state_index = states.set_index("base_index")
    if not state_index.index.is_unique:
        raise ValueError("G4E pair representation contains duplicate base indexes.")
    for question in QUESTIONS:
        selected = frame.loc[question_mask(frame, question)].copy()
        if selected.empty:
            continue
        selected["question"] = question
        selected["identity"] = question_identity(selected, question)
        representation = question_representation(question)
        selected["representation"] = representation
        actual_values = np.full(len(selected), np.nan, dtype=float)
        control_values = np.full(len(selected), np.nan, dtype=float)
        actual_bands = Series(pd.NA, index=selected.index, dtype="object")
        control_bands = Series(pd.NA, index=selected.index, dtype="object")
        for timeframe in sorted(selected["source_timeframe"].astype(str).unique()):
            local = selected["source_timeframe"].astype(str).eq(timeframe)
            column = f"{representation}__{timeframe}"
            actual_index = selected.loc[local, "actual_base_index"].astype(int)
            control_index = selected.loc[local, "control_base_index"].astype(int)
            actual = state_index.loc[actual_index, column].to_numpy(dtype=float)
            control = state_index.loc[control_index, column].to_numpy(dtype=float)
            actual_values[np.flatnonzero(local.to_numpy())] = actual
            control_values[np.flatnonzero(local.to_numpy())] = control
            edge = edge_for(edges, timeframe=timeframe, representation=representation)
            actual_bands.loc[local] = assign_bands(
                Series(actual, index=selected.index[local]), edge
            )
            control_bands.loc[local] = assign_bands(
                Series(control, index=selected.index[local]), edge
            )
        selected["actual_representation_value"] = actual_values
        selected["control_representation_value"] = control_values
        selected["actual_representation_band"] = actual_bands
        selected["control_representation_band"] = control_bands
        selected["representation_absolute_difference"] = np.abs(actual_values - control_values)
        same_observed_band = (
            selected["actual_representation_band"]
            .fillna("__missing__")
            .eq(selected["control_representation_band"].fillna("__missing__"))
        )
        selected["representation_common_support"] = (
            selected["actual_representation_band"].notna()
            & selected["control_representation_band"].notna()
            & same_observed_band
            & selected["representation_absolute_difference"]
            .le(REPRESENTATION_CALIPERS[representation])
            .fillna(False)
        )
        if question == QUESTION_SMA50:
            bb_difference = (
                pd.to_numeric(selected["actual_state__state_local_bb_position"], errors="coerce")
                - pd.to_numeric(selected["control_state__state_local_bb_position"], errors="coerce")
            ).abs()
            ema_difference = (
                pd.to_numeric(selected["actual_state__state_local_ema50_gap_atr"], errors="coerce")
                - pd.to_numeric(
                    selected["control_state__state_local_ema50_gap_atr"], errors="coerce"
                )
            ).abs()
            selected["sma50_bb_position_difference"] = bb_difference
            selected["sma50_ema50_gap_atr_difference"] = ema_difference
            selected["representation_common_support"] &= bb_difference.le(
                SMA50_BB_POSITION_CALIPER
            ).fillna(False) & ema_difference.le(SMA50_EMA50_GAP_ATR_CALIPER).fillna(False)
        else:
            selected["sma50_bb_position_difference"] = np.nan
            selected["sma50_ema50_gap_atr_difference"] = np.nan
        frames.append(selected)
    if not frames:
        return DataFrame()
    return pd.concat(frames, ignore_index=True)


def preflight_support(
    *,
    pairs: Sequence[str],
    cohort: str,
    states: DataFrame,
    edges: DataFrame,
    direct_run_id: str,
    artificial_run_id: str,
    period_map: dict[str, str],
) -> DataFrame:
    columns = [
        *BASE_MATCH_COLUMNS,
        "actual_state__state_local_bb_position",
        "control_state__state_local_bb_position",
        "actual_state__state_local_ema50_gap_atr",
        "control_state__state_local_ema50_gap_atr",
    ]
    frames = []
    for pair in pairs:
        pair_states = states.loc[states["pair"].eq(pair)]
        for root, run_id in (
            (G3F_DIRECT_ARTIFACT_ROOT, direct_run_id),
            (G3F_ARTIFICIAL_ARTIFACT_ROOT, artificial_run_id),
        ):
            path = root / run_id / "pair_independent_matches" / f"{pair_stem(pair)}.parquet"
            source = pd.read_parquet(path, columns=columns)
            frames.append(attach_question_rows(source, states=pair_states, edges=edges))
    long = pd.concat(frames, ignore_index=True)
    long["period"] = canonical_periods(long["period"], period_map)
    long["analysis_scope"] = analysis_scope(long, cohort=cohort)
    grouped = (
        long.groupby(
            [
                "analysis_scope",
                "question",
                "source_timeframe",
                "identity",
                "control",
                "period",
                "response_window",
            ],
            observed=True,
        )
        .agg(
            matched_rows=("actual_event_time", "size"),
            coins=("pair", "nunique"),
            common_support_rows=("representation_common_support", "sum"),
            common_support_coins=(
                "pair",
                lambda values: values[
                    long.loc[values.index, "representation_common_support"]
                ].nunique(),
            ),
        )
        .reset_index()
    )
    grouped["common_support_gate_passed"] = grouped["common_support_rows"].ge(
        MIN_COHORT_PERIOD_EVENTS
    ) & grouped["common_support_coins"].ge(MIN_COHORT_PERIOD_COINS)
    grouped["outcomes_opened"] = False
    grouped["direction_prediction"] = False
    grouped["profit_optimization"] = False
    return grouped


def preflight_integrity(
    *,
    pairs: Sequence[str],
    results: Sequence[dict[str, Any]],
    states: DataFrame,
    edges: DataFrame,
    support: DataFrame,
) -> dict[str, Any]:
    failures = [row for row in results if row.get("status") == "failed"]
    missing_pairs = sorted(set(pairs) - set(states["pair"].astype(str)))
    expected_edges = len(SOURCE_TIMEFRAMES) * 2
    return {
        "passed": bool(
            not failures
            and not missing_pairs
            and not states.empty
            and len(edges) == expected_edges
            and edges["distinct_three_band_edges"].all()
            and not support.empty
            and states["reaction_outcomes_loaded"].eq(False).all()
            and states["direction_prediction"].eq(False).all()
            and states["profit_optimization"].eq(False).all()
        ),
        "pair_failures": failures,
        "missing_pairs": missing_pairs,
        "representation_rows": len(states),
        "edge_rows": len(edges),
        "expected_edge_rows": expected_edges,
        "support_rows": len(support),
        "questions_with_support_rows": sorted(support["question"].astype(str).unique()),
        "outcome_columns_detected": sorted(column for column in states if "outcome" in column),
        "direction_violations": int(states["direction_prediction"].ne(False).sum()),
        "profit_violations": int(states["profit_optimization"].ne(False).sum()),
    }


def supported_question_timeframes(
    support: DataFrame,
) -> tuple[set[tuple[str, str]], DataFrame]:
    validation = support.loc[
        support["analysis_scope"].ne("btc")
        & support["control"].eq(DIRECT_NO_LEVEL)
        & support["period"].astype(str).isin(("validation_early", "validation_late"))
        & support["response_window"].astype(str).isin(("h1", "h4"))
    ].copy()
    pooled = (
        validation.groupby(
            ["analysis_scope", "question", "source_timeframe", "period", "response_window"],
            observed=True,
        )
        .agg(
            common_support_rows=("common_support_rows", "sum"),
            common_support_coins=("common_support_coins", "max"),
        )
        .reset_index()
    )
    pooled["support_pass"] = pooled["common_support_rows"].ge(MIN_COHORT_PERIOD_EVENTS) & pooled[
        "common_support_coins"
    ].ge(MIN_COHORT_PERIOD_COINS)
    rows = []
    opened: set[tuple[str, str]] = set()
    expected = {
        (question, timeframe)
        for question in (QUESTION_BOLLINGER, QUESTION_MA_BUNDLE)
        for timeframe in SOURCE_TIMEFRAMES
    } | {(QUESTION_SMA50, "1h")}
    for question, timeframe in sorted(expected):
        cells = pooled.loc[
            pooled["question"].eq(question) & pooled["source_timeframe"].astype(str).eq(timeframe)
        ]
        passed = bool(len(cells) == 4 and cells["support_pass"].all())
        if passed:
            opened.add((question, timeframe))
        rows.append(
            {
                "analysis_scope": (
                    str(cells.iloc[0]["analysis_scope"]) if not cells.empty else "unavailable"
                ),
                "question": question,
                "source_timeframe": timeframe,
                "required_period_response_cells": 4,
                "observed_period_response_cells": len(cells),
                "passed_period_response_cells": int(cells["support_pass"].sum()),
                "outcomes_opened": passed,
                "status": "opened_after_preflight" if passed else "parked_preflight_common_support",
                "direction_prediction": False,
                "profit_optimization": False,
            }
        )
    assessment = DataFrame(rows)
    return opened, assessment.loc[~assessment["outcomes_opened"]].reset_index(drop=True)


def analysis_source_columns() -> list[str]:
    columns = [*BASE_MATCH_COLUMNS]
    for feature in CORE_STATE_FEATURES:
        columns.extend((f"actual_state__{feature}", f"control_state__{feature}"))
    for outcome in OUTCOMES:
        columns.extend((f"actual__{outcome}", f"control__{outcome}"))
    return list(dict.fromkeys(columns))


def analysis_pairs_for_pair(
    *,
    pair: str,
    cohort: str,
    states: DataFrame,
    edges: DataFrame,
    direct_run_id: str,
    artificial_run_id: str,
    period_map: dict[str, str],
    opened_cells: set[tuple[str, str]],
) -> DataFrame:
    frames = []
    columns = analysis_source_columns()
    for source_kind, root, run_id in (
        ("direct", G3F_DIRECT_ARTIFACT_ROOT, direct_run_id),
        ("artificial", G3F_ARTIFICIAL_ARTIFACT_ROOT, artificial_run_id),
    ):
        path = root / run_id / "pair_independent_matches" / f"{pair_stem(pair)}.parquet"
        frame = pd.read_parquet(path, columns=columns)
        frame["source_kind"] = source_kind
        frames.append(attach_question_rows(frame, states=states, edges=edges))
    output = pd.concat(frames, ignore_index=True)
    output["period"] = canonical_periods(output["period"], period_map)
    keep = Series(False, index=output.index)
    for question, timeframe in opened_cells:
        keep |= output["question"].eq(question) & output["source_timeframe"].astype(str).eq(
            timeframe
        )
    output = output.loc[keep].reset_index(drop=True)
    output["analysis_scope"] = analysis_scope(output, cohort=cohort)
    output["direction_prediction"] = False
    output["profit_optimization"] = False
    return output


def exact_residual_summaries(pairs_long: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = (
        "analysis_scope",
        "question",
        "source_timeframe",
        "identity",
        "outcome",
        "control",
        "period",
        "conditioning",
    )
    for outcome, spec in OUTCOMES.items():
        source = pairs_long.loc[
            pairs_long["response_window"].astype(str).eq(spec["response_window"])
        ].copy()
        source["actual_outcome"] = pd.to_numeric(source[f"actual__{outcome}"], errors="coerce")
        source["control_outcome"] = pd.to_numeric(source[f"control__{outcome}"], errors="coerce")
        source["delta"] = source["actual_outcome"] - source["control_outcome"]
        source["outcome"] = outcome
        for conditioning, mask in (
            ("state_representation_removed", Series(True, index=source.index)),
            ("representation_common_support", source["representation_common_support"].astype(bool)),
        ):
            selected = source.loc[mask].copy()
            if selected.empty:
                continue
            selected["conditioning"] = conditioning
            for keys, group in selected.groupby(list(group_columns[:-1]), observed=True):
                valid = group.loc[np.isfinite(group["delta"])].copy()
                pair_means = valid.groupby("pair", observed=True)["delta"].mean()
                balance = matched_balance(valid)
                row = dict(zip(group_columns[:-1], keys, strict=True))
                row.update(
                    {
                        "conditioning": conditioning,
                        "matched_pairs": len(valid),
                        "coins": len(pair_means),
                        "equal_coin_delta_median": float(pair_means.median())
                        if len(pair_means)
                        else np.nan,
                        "positive_coins": int(pair_means.gt(0.0).sum()),
                        "negative_coins": int(pair_means.lt(0.0).sum()),
                        "positive_coin_fraction": float(pair_means.gt(0.0).mean())
                        if len(pair_means)
                        else np.nan,
                        "max_state_smd": balance["max_absolute_smd"],
                        "median_state_smd": balance["median_absolute_smd"],
                        "state_features_scored": int(balance["features_scored"]),
                        "state_balance_pass": balance_pass(balance),
                        "support_pass": bool(
                            len(valid) >= MIN_COHORT_PERIOD_EVENTS
                            and len(pair_means) >= MIN_COHORT_PERIOD_COINS
                            and balance_pass(balance)
                        ),
                        "direction_prediction": False,
                        "profit_optimization": False,
                    }
                )
                rows.append(row)
    return DataFrame(rows)


def matched_balance(group: DataFrame) -> dict[str, Any]:
    if group.empty:
        return {"features_scored": 0, "max_absolute_smd": np.nan, "median_absolute_smd": np.nan}
    columns = list(CORE_STATE_FEATURES)
    actual = DataFrame({column: group[f"actual_state__{column}"].to_numpy() for column in columns})
    control = DataFrame(
        {column: group[f"control_state__{column}"].to_numpy() for column in columns}
    )
    actual["g4e_representation"] = group["actual_representation_value"].to_numpy()
    control["g4e_representation"] = group["control_representation_value"].to_numpy()
    return balance_diagnostics(actual, control, [*columns, "g4e_representation"])


def balance_pass(balance: dict[str, Any]) -> bool:
    return bool(
        balance["features_scored"] >= 4
        and np.isfinite(balance["max_absolute_smd"])
        and balance["max_absolute_smd"] <= MAX_STATE_SMD
        and np.isfinite(balance["median_absolute_smd"])
        and balance["median_absolute_smd"] <= MAX_MEDIAN_STATE_SMD
    )


def exact_residual_assessment(summary: DataFrame) -> DataFrame:
    conditioned = summary.loc[summary["conditioning"].eq("representation_common_support")]
    rows: list[dict[str, Any]] = []
    key_columns = ("analysis_scope", "question", "source_timeframe", "identity", "outcome")
    for keys, group in conditioned.groupby(list(key_columns), observed=True):
        scope, question, _timeframe, _identity, outcome = keys
        development = group.loc[
            group["period"].astype(str).eq("development") & group["control"].eq(DIRECT_NO_LEVEL)
        ]
        if question == QUESTION_SMA50 and outcome == "volume_ratio_h4":
            expected_sign = -1
            sign_source = "frozen_g3f_low_volume_partial"
        elif len(development) == 1 and np.isfinite(development.iloc[0]["equal_coin_delta_median"]):
            expected_sign = int(np.sign(float(development.iloc[0]["equal_coin_delta_median"])))
            sign_source = "development_same_state_no_level"
        else:
            expected_sign = 0
            sign_source = "unsupported_development_orientation"
        validation = group.loc[
            group["period"].astype(str).isin(("validation_early", "validation_late"))
        ]
        checks = []
        missing = []
        for control in LOCATION_CONTROLS:
            for period in ("validation_early", "validation_late"):
                cell = validation.loc[
                    validation["control"].eq(control) & validation["period"].astype(str).eq(period)
                ]
                if len(cell) != 1:
                    missing.append(f"{control}:{period}")
                    checks.append(False)
                    continue
                row = cell.iloc[0]
                effect = float(row["equal_coin_delta_median"])
                agreement = (
                    int(row["positive_coins"]) if expected_sign > 0 else int(row["negative_coins"])
                )
                checks.append(
                    bool(
                        expected_sign != 0
                        and row["support_pass"]
                        and expected_sign * effect > 0.0
                        and agreement >= MIN_COHORT_PERIOD_COINS
                    )
                )
        retained = bool(scope != "btc" and checks and all(checks) and not missing)
        rows.append(
            {
                **dict(zip(key_columns, keys, strict=True)),
                "expected_effect_sign": expected_sign,
                "orientation_source": sign_source,
                "required_control_period_checks": len(checks),
                "passed_control_period_checks": int(sum(checks)),
                "missing_control_period_cells": ";".join(missing),
                "retained_exact_location_residual": retained,
                "status": (
                    "retained_exact_location_residual"
                    if retained
                    else "parked_exact_location_not_attributed"
                ),
                "direction_prediction": False,
                "profit_optimization": False,
            }
        )
    return DataFrame(rows)


def representation_event_rows(pairs_long: DataFrame) -> DataFrame:
    source = pairs_long.loc[pairs_long["control"].eq(DIRECT_NO_LEVEL)].copy()
    rows: list[DataFrame] = []
    for outcome, spec in OUTCOMES.items():
        selected = source.loc[
            source["response_window"].astype(str).eq(spec["response_window"])
            & source["question"].ne(QUESTION_SMA50)
        ].copy()
        for surface, prefix in (("exact_contact", "actual"), ("exact_line_removed", "control")):
            event = DataFrame(
                {
                    "analysis_scope": selected["analysis_scope"],
                    "question": selected["question"],
                    "source_timeframe": selected["source_timeframe"],
                    "identity": selected["identity"],
                    "representation": selected["representation"],
                    "surface": surface,
                    "pair": selected["pair"],
                    "period": selected["period"],
                    "event_time": selected[f"{prefix}_event_time"],
                    "base_index": selected[f"{prefix}_base_index"],
                    "feature_band": selected[f"{prefix}_representation_band"],
                    "feature_value": selected[f"{prefix}_representation_value"],
                    "outcome": outcome,
                    "outcome_value": selected[f"{prefix}__{outcome}"],
                    "pre_distance_atr": 0.0,
                }
            )
            for feature in CORE_STATE_FEATURES:
                event[feature] = selected[f"{prefix}_state__{feature}"].to_numpy()
            rows.append(event)
    output = pd.concat(rows, ignore_index=True)
    dedupe = [
        "analysis_scope",
        "question",
        "source_timeframe",
        "identity",
        "surface",
        "pair",
        "period",
        "base_index",
        "outcome",
    ]
    output = output.sort_values("event_time", kind="stable").drop_duplicates(dedupe, keep="first")
    output["period_role"] = output["period"].map(period_roles_from_values(output["period"]))
    output["direction_prediction"] = False
    output["profit_optimization"] = False
    return output.reset_index(drop=True)


def period_roles_from_values(periods: Series) -> dict[str, str]:
    unique = list(dict.fromkeys(periods.astype(str)))
    # The frozen manifests use these exact role names. Keep exposed diagnostics separate.
    return {
        period: (
            "development"
            if period == "development"
            else VALIDATION_ROLE
            if period in {"validation_early", "validation_late"}
            else "previously_exposed_diagnostic"
        )
        for period in unique
    }


def representation_band_summary(events: DataFrame) -> DataFrame:
    pair = (
        events.loc[
            events["feature_band"].notna()
            & np.isfinite(pd.to_numeric(events["outcome_value"], errors="coerce"))
        ]
        .groupby(
            [
                "analysis_scope",
                "question",
                "source_timeframe",
                "surface",
                "pair",
                "period",
                "period_role",
                "outcome",
                "feature_band",
            ],
            observed=True,
        )
        .agg(events=("event_time", "size"), outcome_mean=("outcome_value", "mean"))
        .reset_index()
    )
    cohort = (
        pair.groupby(
            [
                "analysis_scope",
                "question",
                "source_timeframe",
                "surface",
                "period",
                "period_role",
                "outcome",
                "feature_band",
            ],
            observed=True,
        )
        .agg(
            coins=("pair", "nunique"),
            independent_events=("events", "sum"),
            equal_coin_outcome_median=("outcome_mean", "median"),
        )
        .reset_index()
    )
    cohort["support_pass"] = cohort["independent_events"].ge(MIN_COHORT_PERIOD_EVENTS) & cohort[
        "coins"
    ].ge(MIN_COHORT_PERIOD_COINS)
    cohort["direction_prediction"] = False
    cohort["profit_optimization"] = False
    return cohort


def monotonic_orientation(values: dict[str, float]) -> str:
    low, middle, high = (
        values.get("low", np.nan),
        values.get("middle", np.nan),
        values.get("high", np.nan),
    )
    if not all(np.isfinite(value) for value in (low, middle, high)) or low == high:
        return "not_monotonic"
    if low <= middle <= high:
        return "increasing"
    if low >= middle >= high:
        return "decreasing"
    return "not_monotonic"


def frozen_development_orientations(band_summary: DataFrame) -> DataFrame:
    rows = []
    source = band_summary.loc[
        band_summary["surface"].eq("exact_line_removed")
        & band_summary["period"].astype(str).eq("development")
    ]
    keys = ("analysis_scope", "question", "source_timeframe", "outcome")
    for values, group in source.groupby(list(keys), observed=True):
        band = group.set_index("feature_band")["equal_coin_outcome_median"].to_dict()
        support = group.set_index("feature_band")["support_pass"].to_dict()
        orientation = monotonic_orientation(band)
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "development_orientation": orientation,
                "development_all_bands_supported": all(
                    support.get(name, False) for name in ("low", "middle", "high")
                ),
            }
        )
    return DataFrame(rows)


def representation_state_controls(events: DataFrame, band_summary: DataFrame) -> DataFrame:
    orientations = frozen_development_orientations(band_summary)
    rows: list[dict[str, Any]] = []
    for _, candidate in orientations.iterrows():
        orientation = str(candidate["development_orientation"])
        if orientation not in {"increasing", "decreasing"}:
            continue
        selected = events.loc[
            events["analysis_scope"].eq(candidate["analysis_scope"])
            & events["question"].eq(candidate["question"])
            & events["source_timeframe"].eq(candidate["source_timeframe"])
            & events["outcome"].eq(candidate["outcome"])
            & events["period"].astype(str).isin(("validation_early", "validation_late"))
            & events["feature_band"].notna()
        ]
        state_features = (
            BOLLINGER_STATE_FEATURES
            if candidate["question"] == QUESTION_BOLLINGER
            else MA_STATE_FEATURES
        )
        sign = 1.0 if orientation == "increasing" else -1.0
        for surface in ("exact_contact", "exact_line_removed"):
            surface_rows = selected.loc[selected["surface"].eq(surface)]
            for period in ("validation_early", "validation_late"):
                period_rows = surface_rows.loc[surface_rows["period"].astype(str).eq(period)]
                for comparison, lower_band, upper_band in (
                    ("low_to_middle", "low", "middle"),
                    ("middle_to_high", "middle", "high"),
                ):
                    matches = []
                    for _, group in period_rows.groupby(["pair", "identity"], observed=True):
                        lower = group.loc[group["feature_band"].eq(lower_band)].reset_index(
                            drop=True
                        )
                        upper = group.loc[group["feature_band"].eq(upper_band)].reset_index(
                            drop=True
                        )
                        pairs, _ = nearest_state_pairs(
                            upper,
                            lower,
                            state_columns=state_features,
                            pre_distance_atr_caliper=1e-9,
                            minimum_event_separation_hours=(
                                1 if candidate["outcome"] == "abs_excursion_atr_h1" else 4
                            ),
                        )
                        for upper_position, lower_position, distance in pairs:
                            high = upper.iloc[upper_position]
                            low = lower.iloc[lower_position]
                            row = {
                                "pair": high["pair"],
                                "oriented_delta": sign
                                * (float(high["outcome_value"]) - float(low["outcome_value"])),
                                "state_distance": distance,
                            }
                            for feature in state_features:
                                row[f"upper__{feature}"] = high[feature]
                                row[f"lower__{feature}"] = low[feature]
                            matches.append(row)
                    matched = DataFrame(matches)
                    summary = summarize_band_matches(
                        matched,
                        state_features=state_features,
                        seed_key="|".join(
                            str(candidate[name])
                            for name in (
                                "analysis_scope",
                                "question",
                                "source_timeframe",
                                "outcome",
                            )
                        )
                        + f"|{surface}|{period}|{comparison}",
                    )
                    rows.append(
                        {
                            "analysis_scope": candidate["analysis_scope"],
                            "question": candidate["question"],
                            "source_timeframe": candidate["source_timeframe"],
                            "outcome": candidate["outcome"],
                            "development_orientation": orientation,
                            "surface": surface,
                            "period": period,
                            "comparison": comparison,
                            **summary,
                            "direction_prediction": False,
                            "profit_optimization": False,
                        }
                    )
    return DataFrame(rows)


def summarize_band_matches(
    matches: DataFrame, *, state_features: Sequence[str], seed_key: str
) -> dict[str, Any]:
    if matches.empty:
        return {
            "matched_pairs": 0,
            "coins": 0,
            "positive_oriented_coins": 0,
            "equal_coin_oriented_delta_median": np.nan,
            "permutation_p_value": np.nan,
            "permutation_null_95": np.nan,
            "max_state_smd": np.nan,
            "median_state_smd": np.nan,
            "state_balance_pass": False,
            "support_pass": False,
            "control_pass": False,
        }
    valid = matches.loc[
        np.isfinite(pd.to_numeric(matches["oriented_delta"], errors="coerce"))
    ].copy()
    pair_means = valid.groupby("pair", observed=True)["oriented_delta"].mean()
    upper = DataFrame(
        {feature: valid[f"upper__{feature}"].to_numpy() for feature in state_features}
    )
    lower = DataFrame(
        {feature: valid[f"lower__{feature}"].to_numpy() for feature in state_features}
    )
    balance = balance_diagnostics(upper, lower, state_features)
    observed, p_value, null_95 = matched_sign_permutation(valid, seed_key=seed_key)
    support = bool(
        len(valid) >= MIN_COHORT_PERIOD_EVENTS
        and len(pair_means) >= MIN_COHORT_PERIOD_COINS
        and balance_pass(balance)
    )
    positive = int(pair_means.gt(0.0).sum())
    passed = bool(
        support
        and np.isfinite(observed)
        and observed > 0.0
        and positive >= MIN_COHORT_PERIOD_COINS
        and np.isfinite(p_value)
        and p_value <= 0.05
        and np.isfinite(null_95)
        and observed > null_95
    )
    return {
        "matched_pairs": len(valid),
        "coins": len(pair_means),
        "positive_oriented_coins": positive,
        "equal_coin_oriented_delta_median": observed,
        "permutation_p_value": p_value,
        "permutation_null_95": null_95,
        "max_state_smd": balance["max_absolute_smd"],
        "median_state_smd": balance["median_absolute_smd"],
        "state_balance_pass": balance_pass(balance),
        "support_pass": support,
        "control_pass": passed,
    }


def matched_sign_permutation(matches: DataFrame, *, seed_key: str) -> tuple[float, float, float]:
    values = pd.to_numeric(matches["oriented_delta"], errors="coerce").to_numpy(dtype=float)
    pairs, names = pd.factorize(matches["pair"].astype(str), sort=True)
    counts = np.bincount(pairs).astype(float)

    def equal_coin_median(candidate: np.ndarray) -> float:
        sums = np.bincount(pairs, weights=candidate, minlength=len(names))
        return float(np.median(sums / counts))

    observed = equal_coin_median(values)
    rng = np.random.default_rng(stable_hash_int(seed_key))
    null = np.empty(PERMUTATIONS, dtype=float)
    for index in range(PERMUTATIONS):
        signs = rng.choice(np.asarray((-1.0, 1.0)), size=len(values), replace=True)
        null[index] = equal_coin_median(values * signs)
    p_value = float((1 + np.count_nonzero(null >= observed)) / (PERMUTATIONS + 1))
    return observed, p_value, float(np.quantile(null, 0.95))


def assess_representations(*, band_summary: DataFrame, control_summary: DataFrame) -> DataFrame:
    orientations = frozen_development_orientations(band_summary)
    rows = []
    for _, candidate in orientations.iterrows():
        key_mask = (
            band_summary["analysis_scope"].eq(candidate["analysis_scope"])
            & band_summary["question"].eq(candidate["question"])
            & band_summary["source_timeframe"].eq(candidate["source_timeframe"])
            & band_summary["outcome"].eq(candidate["outcome"])
        )
        validation = band_summary.loc[
            key_mask
            & band_summary["period"].astype(str).isin(("validation_early", "validation_late"))
        ]
        orientation = str(candidate["development_orientation"])
        raw_checks = []
        for surface in ("exact_contact", "exact_line_removed"):
            for period in ("validation_early", "validation_late"):
                cell = validation.loc[
                    validation["surface"].eq(surface) & validation["period"].astype(str).eq(period)
                ]
                values = cell.set_index("feature_band")["equal_coin_outcome_median"].to_dict()
                supports = cell.set_index("feature_band")["support_pass"].to_dict()
                raw_checks.append(
                    bool(
                        orientation in {"increasing", "decreasing"}
                        and monotonic_orientation(values) == orientation
                        and all(supports.get(band, False) for band in ("low", "middle", "high"))
                    )
                )
        control = control_summary.loc[
            control_summary["analysis_scope"].eq(candidate["analysis_scope"])
            & control_summary["question"].eq(candidate["question"])
            & control_summary["source_timeframe"].eq(candidate["source_timeframe"])
            & control_summary["outcome"].eq(candidate["outcome"])
        ]
        expected_control_cells = 2 * 2 * 2
        control_passes = int(control["control_pass"].sum()) if not control.empty else 0
        retained = bool(
            candidate["analysis_scope"] != "btc"
            and candidate["development_all_bands_supported"]
            and len(raw_checks) == 4
            and all(raw_checks)
            and len(control) == expected_control_cells
            and control_passes == expected_control_cells
        )
        rows.append(
            {
                "analysis_scope": candidate["analysis_scope"],
                "question": candidate["question"],
                "source_timeframe": candidate["source_timeframe"],
                "outcome": candidate["outcome"],
                "development_orientation": orientation,
                "raw_surface_period_checks": len(raw_checks),
                "raw_surface_period_checks_passed": int(sum(raw_checks)),
                "state_control_checks": len(control),
                "state_control_checks_passed": control_passes,
                "retained_representation_lead": retained,
                "status": (
                    "retained_representation_lead"
                    if retained
                    else "parked_representation_not_repeated_through_controls"
                ),
                "direction_prediction": False,
                "profit_optimization": False,
            }
        )
    return DataFrame(rows)


def component_ablation_summary(pairs_long: DataFrame) -> DataFrame:
    source = pairs_long.loc[
        pairs_long["control"].eq(COMPONENT_ABLATION)
        & pairs_long["representation_common_support"].astype(bool)
    ]
    rows = []
    for outcome, spec in OUTCOMES.items():
        selected = source.loc[
            source["response_window"].astype(str).eq(spec["response_window"])
        ].copy()
        selected["delta"] = pd.to_numeric(
            selected[f"actual__{outcome}"], errors="coerce"
        ) - pd.to_numeric(selected[f"control__{outcome}"], errors="coerce")
        for keys, group in selected.groupby(
            ["analysis_scope", "question", "source_timeframe", "identity", "period"],
            observed=True,
        ):
            valid = group.loc[np.isfinite(group["delta"])]
            pair_means = valid.groupby("pair", observed=True)["delta"].mean()
            rows.append(
                {
                    "analysis_scope": keys[0],
                    "question": keys[1],
                    "source_timeframe": keys[2],
                    "identity": keys[3],
                    "period": keys[4],
                    "outcome": outcome,
                    "matched_pairs": len(valid),
                    "coins": len(pair_means),
                    "equal_coin_cluster_minus_isolated_median": float(pair_means.median())
                    if len(pair_means)
                    else np.nan,
                    "positive_coins": int(pair_means.gt(0.0).sum()),
                    "negative_coins": int(pair_means.lt(0.0).sum()),
                    "support_pass": bool(
                        len(valid) >= MIN_COHORT_PERIOD_EVENTS
                        and len(pair_means) >= MIN_COHORT_PERIOD_COINS
                    ),
                    "interpretation": (
                        "This compares an independently crossed cluster with the same named "
                        "anchor isolated; it does not by itself attribute the exact line."
                    ),
                    "direction_prediction": False,
                    "profit_optimization": False,
                }
            )
    return DataFrame(rows)


def analysis_integrity_record(
    *, pairs: Sequence[str], pair_rows: DataFrame, outputs: dict[str, DataFrame]
) -> dict[str, Any]:
    missing_pairs = sorted(set(pairs) - set(pair_rows["pair"].astype(str)))
    empty_outputs = sorted(name for name, frame in outputs.items() if frame.empty)
    invalid_controls = sorted(set(pair_rows["control"].astype(str)) - set(ALL_CONTROLS))
    return {
        "passed": bool(
            not pair_rows.empty
            and not missing_pairs
            and not empty_outputs
            and not invalid_controls
            and pair_rows["direction_prediction"].eq(False).all()
            and pair_rows["profit_optimization"].eq(False).all()
        ),
        "pair_rows": len(pair_rows),
        "missing_pairs": missing_pairs,
        "questions": sorted(pair_rows["question"].astype(str).unique()),
        "controls": sorted(pair_rows["control"].astype(str).unique()),
        "invalid_controls": invalid_controls,
        "empty_outputs": empty_outputs,
        "output_rows": {name: len(frame) for name, frame in outputs.items()},
        "direction_violations": int(pair_rows["direction_prediction"].ne(False).sum()),
        "profit_violations": int(pair_rows["profit_optimization"].ne(False).sum()),
    }


if __name__ == "__main__":
    raise SystemExit(main())

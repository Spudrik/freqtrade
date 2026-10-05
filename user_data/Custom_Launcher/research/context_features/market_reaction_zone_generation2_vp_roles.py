from __future__ import annotations

# Fix numerical-library thread counts before importing numpy/pandas/sklearn.
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
    DEFAULT_MANIFEST,
    OUTPUT_ROOT,
    atomic_write_json,
    atomic_write_parquet,
    load_manifest,
    manifest_storage_paths,
    normalize_dates,
    numeric_array,
    numeric_series,
    prepare_base_market_frame,
    sha256_file,
    utc_now,
    validate_cache_metadata,
    validate_worker_count,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_localization import (  # noqa: E501
    LOCAL_MATCH_FEATURES,
    aligned_selected_levels,
    balance_diagnostics,
    causal_local_state,
    causal_market_context,
    nearest_state_pairs,
    outcome_metadata,
    outcome_values,
    purge_overlapping_event_pairs_by_key,
    role_oriented_delta,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_volume_profile_roles import (  # noqa: E501
    AttributeProfile,
    assign_attribute_tertiles,
    attach_states,
    matched_attribute_cells,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_localization import (  # noqa: E501
    eligible_period_events,
)


FROZEN_BATCH = OUTPUT_ROOT / "generation1_review" / "g2_frozen_branch_batch.json"
REPORT_ROOT = OUTPUT_ROOT / "generation2_branches" / "g2b_volume_profile_roles"
MIN_CELL_EVENTS = 20
MAX_STATE_SMD = 0.50
MAX_MEDIAN_STATE_SMD = 0.20
OUTPUT_SCHEMA_VERSION = 1
MIN_PAIR_PERIOD_EPISODES = 5
MIN_COHORT_PERIOD_EPISODES = 50
MIN_COHORT_PERIOD_COINS = 5


@dataclass(frozen=True)
class Question:
    id: str
    timeframe: str
    zone: str
    profile: AttributeProfile
    outcomes: tuple[str, ...]


QUESTIONS = (
    Question(
        id="g2b_hvn_strength",
        timeframe="4h",
        zone="tight_base_atr",
        profile=AttributeProfile(
            "hvn_strength",
            ("hvn_above", "hvn_below"),
            "level_score",
            "acceptance_stickiness",
        ),
        outcomes=("volume_ratio_h2", "range_ratio_h4", "crossings_h4"),
    ),
    Question(
        id="g2b_lvn_thinness",
        timeframe="1h",
        zone="wide_base_atr",
        profile=AttributeProfile(
            "lvn_thinness",
            ("lvn_above", "lvn_below"),
            "level_score",
            "activity_transit",
        ),
        outcomes=("contact_volume_ratio", "dwell_fraction_h4", "volume_ratio_h48"),
    ),
    Question(
        id="g2b_poc_evidence",
        timeframe="1h",
        zone="tight_base_atr",
        profile=AttributeProfile(
            "poc_profile_evidence",
            ("poc",),
            "level_score",
            "acceptance_stickiness",
        ),
        outcomes=(
            "volume_ratio_h24",
            "range_ratio_h24",
            "volume_ratio_h48",
            "range_ratio_h48",
        ),
    ),
)

QUESTION_HORIZONS = {
    question.id: tuple(
        sorted(
            {
                outcome_metadata(outcome)[1]
                for outcome in question.outcomes
                if outcome_metadata(outcome)[1]
            }
        )
    )
    for question in QUESTIONS
}
INDEPENDENCE_HOURS_BY_QUESTION = {
    question_id: max(horizons) for question_id, horizons in QUESTION_HORIZONS.items()
}

FULL_STATE_FEATURES = (
    *LOCAL_MATCH_FEATURES,
    "state_vp_value_area_width_pct",
    "state_vp_level_persistence_bars",
)
CORE_STATE_FEATURES = (
    "state_local_return_24h",
    "state_local_atr_pct",
    "state_local_range_ratio",
    "state_local_volume_ratio",
    "state_local_pressure_6h",
    "state_source_atr_pct",
    "state_source_volume_ratio",
    "state_btc_return_24h",
    "state_top10_breadth",
    "state_selected_level_density_2atr",
    "state_vp_value_area_width_pct",
    "state_vp_level_persistence_bars",
)
STATE_PROFILES = {
    "core": CORE_STATE_FEATURES,
    "full_sensitivity": FULL_STATE_FEATURES,
}


@dataclass(frozen=True)
class PairTask:
    pair: str
    manifest_path: str
    run_id: str
    overwrite: bool
    request_sha256: str
    state_features: tuple[str, ...]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 2B fixed Volume Profile attribute-role confirmation. It compares "
            "high and low causal attributes with state matching and an in-stratum shuffle."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--state-profile",
        choices=tuple(STATE_PROFILES),
        default="core",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    validate_worker_count(args.workers, manifest=manifest)
    validate_frozen_branch()
    pairs = select_pairs(manifest, args.pairs)
    source_contract = source_contracts(
        manifest=manifest,
        manifest_path=args.manifest,
        pairs=pairs,
        timeframes=tuple(sorted({question.timeframe for question in QUESTIONS})),
    )
    request = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "attribute_helper_sha256": sha256_file(
            Path(__file__).with_name("market_reaction_zone_generation1_volume_profile_roles.py")
        ),
        "localization_helper_sha256": sha256_file(
            Path(__file__).with_name("market_reaction_zone_generation1_localization.py")
        ),
        "manifest_sha256": sha256_file(args.manifest),
        "frozen_batch_sha256": sha256_file(FROZEN_BATCH),
        "pairs": list(pairs),
        "questions": [question_contract(question) for question in QUESTIONS],
        "state_profile": args.state_profile,
        "state_features": list(STATE_PROFILES[args.state_profile]),
        "source_contracts": source_contract,
        "period_end_embargo_hours": max(manifest["reaction_definition"]["horizons_hours"]),
        "independent_pair_pooling": {
            "exact_match_strata": [
                "pair",
                "source_timeframe",
                "level_name",
                "zone_method",
                "period",
                "approach_state",
            ],
            "greedy_overlap_purge_hours_by_question": INDEPENDENCE_HOURS_BY_QUESTION,
            "minimum_pair_period_episodes": MIN_PAIR_PERIOD_EPISODES,
            "minimum_cohort_period_episodes": MIN_COHORT_PERIOD_EPISODES,
            "minimum_cohort_period_coins": MIN_COHORT_PERIOD_COINS,
            "cohort_effect_weights_coins_equally": True,
        },
        "direction_prediction": False,
        "profit_optimization": False,
    }
    request_sha256 = stable_json_sha256(request)
    run_dir = REPORT_ROOT / args.run_id
    pair_dir = run_dir / "pair_cells"
    run_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    record_path = run_dir / "g2b_run_record.json"
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
        "baseline": "The three fixed G1B VP attribute leads on the original large-coin cohort.",
        "hypothesis": (
            "Higher HVN strength and POC evidence order acceptance-like behaviour, while "
            "higher LVN thinness orders activity/transit behaviour, on the frozen meme cohort."
        ),
        "controls": [
            "deterministic attribute shuffle inside pair, level, zone, period, and approach",
            "continuous pre-contact local, source, BTC, meme-breadth, and level-density state",
            "hard 0.10 ATR pre-contact distance gate",
            "same-profile value-area width",
            "emitted-level persistence in source bars",
            "outcome-horizon-specific non-overlap and period-end embargo",
            "separate current-versus-stale-shuffled-and-symmetric-location G2AB run",
            "HVN, LVN, and POC role results reported side by side as opposite-role checks",
        ],
        "workers": args.workers,
        "state_profile": args.state_profile,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(record, record_path)
    tasks = [
        PairTask(
            pair=pair,
            manifest_path=str(args.manifest.resolve()),
            run_id=args.run_id,
            overwrite=args.overwrite,
            request_sha256=request_sha256,
            state_features=STATE_PROFILES[args.state_profile],
        )
        for pair in pairs
    ]
    try:
        results = run_tasks(tasks, workers=args.workers)
        inventory = DataFrame(results)
        atomic_write_parquet(inventory, run_dir / "g2b_pair_inventory.parquet")
        failures = inventory.loc[inventory["status"].eq("failed")]
        if not failures.empty:
            raise RuntimeError(
                f"{len(failures)} pair task(s) failed; inspect g2b_pair_inventory.parquet."
            )
        cells = combine_pair_outputs(
            pair_dir=pair_dir,
            pairs=pairs,
            request_sha256=request_sha256,
        )
        atomic_write_parquet(cells, run_dir / "g2b_attribute_cells.parquet")
        screen = repeatability_screen(cells)
        atomic_write_parquet(screen, run_dir / "g2b_repeatability_screen.parquet")
        matched_pairs = combine_pair_outputs(
            pair_dir=run_dir / "pair_event_matches",
            pairs=pairs,
            request_sha256=request_sha256,
        )
        independent_pairs = purge_overlapping_event_pairs_by_key(
            matched_pairs,
            separation_hours_by_key=INDEPENDENCE_HOURS_BY_QUESTION,
            key_column="question_id",
            group_columns=(
                "pair",
                "question_id",
                "attribute_assignment",
                "period",
            ),
            left_time_column="high_event_time",
            right_time_column="low_event_time",
        )
        atomic_write_parquet(
            independent_pairs,
            run_dir / "g2b_independent_event_pairs.parquet",
        )
        pair_period = pair_period_results(independent_pairs)
        atomic_write_parquet(pair_period, run_dir / "g2b_pair_period_results.parquet")
        cohort_period = cohort_period_results(pair_period, independent_pairs)
        atomic_write_parquet(cohort_period, run_dir / "g2b_cohort_period_results.parquet")
        leave_one_out = leave_one_coin_out_results(pair_period)
        atomic_write_parquet(leave_one_out, run_dir / "g2b_leave_one_coin_out.parquet")
        integrity = integrity_record(cells, results, independent_pairs=independent_pairs)
        atomic_write_json(integrity, run_dir / "g2b_integrity.json")
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "pair_tasks": results,
                "cell_rows": len(cells),
                "repeatability_rows": len(screen),
                "matched_event_pairs_before_overlap_purge": len(matched_pairs),
                "independent_event_pairs": len(independent_pairs),
                "pair_period_rows": len(pair_period),
                "cohort_period_rows": len(cohort_period),
                "leave_one_coin_out_rows": len(leave_one_out),
                "integrity": str(run_dir / "g2b_integrity.json"),
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


def question_contract(question: Question) -> dict[str, Any]:
    return {
        "id": question.id,
        "timeframe": question.timeframe,
        "zone": question.zone,
        "attribute_profile": question.profile.__dict__,
        "outcomes": list(question.outcomes),
    }


def validate_frozen_branch() -> None:
    frozen = json.loads(FROZEN_BATCH.read_text(encoding="utf-8"))
    branches = {branch["id"] for branch in frozen["branches"]}
    if frozen.get("status") != "frozen_before_generation2_reaction_outcomes":
        raise ValueError("The frozen Generation 2 batch is not ready.")
    if "g2b_volume_profile_attribute_role_confirmation" not in branches:
        raise ValueError("The frozen Generation 2B branch is unavailable.")
    boundary = frozen["research_boundary"]
    if boundary.get("direction_prediction") is not False:
        raise ValueError("Generation 2B must keep direction disabled.")
    if boundary.get("profit_optimization") is not False:
        raise ValueError("Generation 2B must keep profit optimization disabled.")


def select_pairs(manifest: dict[str, Any], requested: str) -> tuple[str, ...]:
    allowed = tuple(manifest["data"]["pairs"])
    if requested.strip().lower() == "all":
        return allowed
    selected = tuple(value.strip() for value in requested.split(",") if value.strip())
    unknown = sorted(set(selected).difference(allowed))
    if not selected or unknown:
        raise ValueError(f"Invalid pair selection: selected={selected}, unknown={unknown}")
    return selected


def atlas_event_source(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    pair: str,
    timeframe: str,
) -> tuple[Path, Path, dict[str, Any]]:
    storage = manifest_storage_paths(manifest)
    stem = pair_stem(pair)
    eligible: list[tuple[Path, Path, dict[str, Any]]] = []
    for event_path in sorted(
        storage.event_dir.glob(f"{stem}-{timeframe}-core-generic-g0b1-g0b2-*.parquet")
    ):
        metadata_path = storage.report_dir / f"{event_path.stem}.meta.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("manifest_sha256") != sha256_file(manifest_path):
            continue
        if set(metadata.get("zone_methods", ())) != {
            "tight_base_atr",
            "wide_base_atr",
        }:
            continue
        if "actual" not in set(metadata.get("controls", ())):
            continue
        eligible.append((event_path, metadata_path, metadata))
    if len(eligible) != 1:
        raise ValueError(
            f"Expected one matching Generation 2 atlas source for {pair} {timeframe}; "
            f"found {len(eligible)}."
        )
    return eligible[0]


def source_contracts(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    pairs: Sequence[str],
    timeframes: Sequence[str],
) -> list[dict[str, Any]]:
    contracts = []
    for pair in pairs:
        for timeframe in timeframes:
            event_path, metadata_path, metadata = atlas_event_source(
                manifest=manifest,
                manifest_path=manifest_path,
                pair=pair,
                timeframe=timeframe,
            )
            cache_path = Path(str(metadata["cache"]))
            validate_cache_metadata(cache_path, manifest_path)
            if sha256_file(cache_path) != metadata.get("cache_sha256"):
                raise ValueError(f"Atlas source cache changed: {event_path}")
            contracts.append(
                {
                    "pair": pair,
                    "timeframe": timeframe,
                    "event_path": str(event_path.resolve()),
                    "event_bytes": int(event_path.stat().st_size),
                    "event_modified_ns": int(event_path.stat().st_mtime_ns),
                    "event_metadata_sha256": sha256_file(metadata_path),
                    "cache_sha256": metadata["cache_sha256"],
                }
            )
    return contracts


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        return {
            "pair": task.pair,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_pair(task: PairTask) -> dict[str, Any]:
    started = time.perf_counter()
    output_path = REPORT_ROOT / task.run_id / "pair_cells" / f"{pair_stem(task.pair)}.parquet"
    event_match_path = (
        REPORT_ROOT / task.run_id / "pair_event_matches" / f"{pair_stem(task.pair)}.parquet"
    )
    if output_path.is_file() and not task.overwrite:
        if not event_match_path.is_file():
            raise FileNotFoundError(
                f"Existing cell output has no matched-event companion: {event_match_path}"
            )
        existing = validate_pair_output(
            output_path,
            pair=task.pair,
            request_sha256=task.request_sha256,
        )
        return {
            "pair": task.pair,
            "status": "existing",
            "cell_rows": len(existing),
            "state_features": list(task.state_features),
            "seconds": round(time.perf_counter() - started, 3),
        }
    manifest_path = Path(task.manifest_path)
    manifest = load_manifest(manifest_path)
    base = prepare_base_market_frame(task.pair, manifest)
    local_state = causal_local_state(base).merge(
        causal_market_context(manifest),
        on="date",
        how="left",
        validate="one_to_one",
    )
    aligned_levels = aligned_selected_levels(
        pair=task.pair,
        base=base,
        manifest_path=manifest_path,
        timeframes=tuple(manifest["data"]["source_timeframes"]),
    )
    matrix = np.column_stack([item.level for item in aligned_levels])
    pre_close = numeric_array(base["pre_close"])
    base_atr = numeric_array(base["base_atr"])
    local_state["state_selected_level_density_2atr"] = np.sum(
        np.isfinite(matrix) & (np.abs(matrix - pre_close[:, None]) <= 2.0 * base_atr[:, None]),
        axis=1,
    ).astype(float)
    rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    source_events = 0
    for question in QUESTIONS:
        question_horizons = QUESTION_HORIZONS[question.id]
        question_independence_hours = INDEPENDENCE_HOURS_BY_QUESTION[question.id]
        event_path, _, metadata = atlas_event_source(
            manifest=manifest,
            manifest_path=manifest_path,
            pair=task.pair,
            timeframe=question.timeframe,
        )
        events = pd.read_parquet(
            event_path,
            filters=[
                ("control", "==", "actual"),
                ("level_name", "in", list(question.profile.level_names)),
                ("zone_method", "==", question.zone),
            ],
        )
        events = eligible_period_events(
            events,
            manifest,
            embargo_hours=question_independence_hours,
        )
        if events.empty:
            continue
        source_events += len(events)
        events = attach_states(
            events,
            base=base,
            local_state=local_state,
            cache_path=Path(str(metadata["cache"])),
        )
        events["state_vp_value_area_width_pct"] = pd.to_numeric(
            events["attr_vp_value_area_width_pct"], errors="coerce"
        )
        events = attach_vp_level_persistence(
            events,
            base=base,
            cache_path=Path(str(metadata["cache"])),
        )
        selected = events.loc[
            pd.to_numeric(events[question.profile.attribute_column], errors="coerce").notna()
        ].copy()
        for assignment in ("actual", "shuffled_attribute"):
            tiered = assign_attribute_tertiles(
                selected,
                attribute_column=question.profile.attribute_column,
                assignment=assignment,
                seed_key=f"{task.pair}|{question.id}",
            )
            cells = matched_attribute_cells(
                tiered,
                attribute_profile=question.profile,
                assignment=assignment,
                horizons=question_horizons,
                state_features=task.state_features,
            )
            rows.extend({**row, "question_id": question.id} for row in cells)
            match_rows.extend(
                matched_attribute_event_pairs(
                    tiered,
                    question=question,
                    assignment=assignment,
                    state_features=task.state_features,
                    independence_hours=question_independence_hours,
                )
            )
    output = DataFrame(rows)
    event_matches = DataFrame(match_rows)
    if output.empty or event_matches.empty:
        raise ValueError(f"No matchable Generation 2B cells for {task.pair}.")
    output["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    output["run_request_sha256"] = task.request_sha256
    event_matches["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    event_matches["run_request_sha256"] = task.request_sha256
    output_path.parent.mkdir(parents=True, exist_ok=True)
    event_match_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(output, output_path)
    atomic_write_parquet(event_matches, event_match_path)
    return {
        "pair": task.pair,
        "status": "completed",
        "source_event_rows": source_events,
        "cell_rows": len(output),
        "matched_event_pairs_before_overlap_purge": len(event_matches),
        "state_features": list(task.state_features),
        "seconds": round(time.perf_counter() - started, 3),
    }


def matched_attribute_event_pairs(
    events: DataFrame,
    *,
    question: Question,
    assignment: str,
    state_features: Sequence[str],
    independence_hours: int,
) -> list[dict[str, Any]]:
    if events.empty:
        return []
    rows: list[dict[str, Any]] = []
    group_columns = [
        "source_timeframe",
        "level_name",
        "zone_method",
        "period",
        "approach_state",
    ]
    for key, group in events.groupby(group_columns, observed=True, sort=False):
        high = group.loc[group["attribute_tier"].eq("high")]
        low = group.loc[group["attribute_tier"].eq("low")]
        pairs, _ = nearest_state_pairs(
            high,
            low,
            state_columns=state_features,
            pre_distance_atr_caliper=0.10,
            minimum_event_separation_hours=independence_hours,
        )
        if not pairs:
            continue
        high_outcomes = {outcome: outcome_values(high, outcome) for outcome in question.outcomes}
        low_outcomes = {outcome: outcome_values(low, outcome) for outcome in question.outcomes}
        for high_position, low_position, distance in pairs:
            high_row = high.iloc[high_position]
            low_row = low.iloc[low_position]
            row: dict[str, Any] = {
                "pair": str(high_row["pair"]),
                "question_id": question.id,
                "attribute_assignment": assignment,
                "source_timeframe": str(key[0]),
                "level_name": str(key[1]),
                "zone_method": str(key[2]),
                "period": str(key[3]),
                "approach_state": str(key[4]),
                "high_base_index": int(high_row["base_index"]),
                "low_base_index": int(low_row["base_index"]),
                "high_event_time": pd.Timestamp(high_row["event_time"]),
                "low_event_time": pd.Timestamp(low_row["event_time"]),
                "high_attribute_value": float(high_row["attribute_value_for_tier"]),
                "low_attribute_value": float(low_row["attribute_value_for_tier"]),
                "high_pre_distance_atr": float(high_row["pre_distance_atr"]),
                "low_pre_distance_atr": float(low_row["pre_distance_atr"]),
                "pre_distance_atr_abs_difference": abs(
                    float(high_row["pre_distance_atr"]) - float(low_row["pre_distance_atr"])
                ),
                "event_separation_hours": abs(
                    (
                        pd.Timestamp(high_row["event_time"]) - pd.Timestamp(low_row["event_time"])
                    ).total_seconds()
                    / 3600.0
                ),
                "match_distance": float(distance),
                "level_role": question.profile.role,
                "direction_prediction": False,
                "profit_optimization": False,
            }
            for feature in state_features:
                row[f"high_state__{feature}"] = float(high_row[feature])
                row[f"low_state__{feature}"] = float(low_row[feature])
            for outcome in question.outcomes:
                high_value = float(high_outcomes[outcome][high_position])
                low_value = float(low_outcomes[outcome][low_position])
                delta = high_value - low_value
                metric, _ = outcome_metadata(outcome)
                row[f"delta__{outcome}"] = delta
                row[f"role_delta__{outcome}"] = role_oriented_delta(
                    question.profile.role,
                    metric,
                    delta,
                )
            rows.append(row)
    return rows


def pair_period_results(independent_pairs: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    state_features = matched_state_features(independent_pairs)
    group_columns = ["question_id", "attribute_assignment", "pair", "period"]
    questions = {question.id: question for question in QUESTIONS}
    for key, group in independent_pairs.groupby(group_columns, observed=True, sort=False):
        question = questions[str(key[0])]
        balance = matched_pair_balance(group, state_features)
        for outcome in question.outcomes:
            raw = pd.to_numeric(group[f"delta__{outcome}"], errors="coerce")
            role = pd.to_numeric(group[f"role_delta__{outcome}"], errors="coerce")
            valid = raw.notna() & role.notna()
            if not valid.any():
                continue
            rows.append(
                {
                    "question_id": key[0],
                    "attribute_assignment": key[1],
                    "pair": key[2],
                    "period": key[3],
                    "outcome": outcome,
                    "metric_family": outcome_metadata(outcome)[0],
                    "horizon_hours": outcome_metadata(outcome)[1],
                    "independent_event_pairs": int(valid.sum()),
                    "pair_period_coverage_eligible": int(valid.sum()) >= MIN_PAIR_PERIOD_EPISODES,
                    "raw_delta_mean": float(raw.loc[valid].mean()),
                    "raw_delta_median": float(raw.loc[valid].median()),
                    "raw_positive_fraction": float(raw.loc[valid].gt(0.0).mean()),
                    "role_oriented_delta_mean": float(role.loc[valid].mean()),
                    "role_oriented_delta_median": float(role.loc[valid].median()),
                    "role_positive_fraction": float(role.loc[valid].gt(0.0).mean()),
                    "attribute_separation_median": float(
                        (group["high_attribute_value"] - group["low_attribute_value"]).median()
                    ),
                    "match_distance_median": float(group["match_distance"].median()),
                    "pre_distance_atr_abs_difference_max": float(
                        group["pre_distance_atr_abs_difference"].max()
                    ),
                    "max_absolute_state_smd": balance["max_absolute_smd"],
                    "median_absolute_state_smd": balance["median_absolute_smd"],
                    "state_features_scored": balance["features_scored"],
                    "direction_prediction": False,
                    "profit_optimization": False,
                }
            )
    return DataFrame(rows)


def matched_state_features(frame: DataFrame) -> tuple[str, ...]:
    return tuple(
        sorted(
            column.removeprefix("high_state__")
            for column in frame
            if column.startswith("high_state__")
        )
    )


def matched_pair_balance(
    frame: DataFrame,
    state_features: Sequence[str],
) -> dict[str, Any]:
    high = DataFrame(
        {feature: frame[f"high_state__{feature}"].to_numpy() for feature in state_features}
    )
    low = DataFrame(
        {feature: frame[f"low_state__{feature}"].to_numpy() for feature in state_features}
    )
    return balance_diagnostics(high, low, state_features)


def cohort_period_results(
    pair_period: DataFrame,
    independent_pairs: DataFrame,
) -> DataFrame:
    eligible = pair_period.loc[pair_period["pair_period_coverage_eligible"]].copy()
    actual = eligible.loc[eligible["attribute_assignment"].eq("actual")].copy()
    placebo = eligible.loc[eligible["attribute_assignment"].eq("shuffled_attribute")].copy()
    join_keys = ["question_id", "pair", "period", "outcome"]
    paired = actual.merge(
        placebo,
        on=join_keys,
        how="inner",
        suffixes=("__actual", "__placebo"),
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    state_features = matched_state_features(independent_pairs)
    for key, group in paired.groupby(
        ["question_id", "period", "outcome"], observed=True, sort=False
    ):
        pairs = sorted(group["pair"].astype(str).unique().tolist())
        actual_events = independent_pairs.loc[
            independent_pairs["question_id"].eq(key[0])
            & independent_pairs["period"].eq(key[1])
            & independent_pairs["attribute_assignment"].eq("actual")
            & independent_pairs["pair"].isin(pairs)
        ]
        placebo_events = independent_pairs.loc[
            independent_pairs["question_id"].eq(key[0])
            & independent_pairs["period"].eq(key[1])
            & independent_pairs["attribute_assignment"].eq("shuffled_attribute")
            & independent_pairs["pair"].isin(pairs)
        ]
        actual_balance = matched_pair_balance(actual_events, state_features)
        placebo_balance = matched_pair_balance(placebo_events, state_features)
        actual_delta = group["role_oriented_delta_mean__actual"]
        placebo_delta = group["role_oriented_delta_mean__placebo"]
        lift = actual_delta - placebo_delta
        actual_count = int(group["independent_event_pairs__actual"].sum())
        placebo_count = int(group["independent_event_pairs__placebo"].sum())
        coverage = (
            len(pairs) >= MIN_COHORT_PERIOD_COINS
            and actual_count >= MIN_COHORT_PERIOD_EPISODES
            and placebo_count >= MIN_COHORT_PERIOD_EPISODES
        )
        rows.append(
            {
                "question_id": key[0],
                "period": key[1],
                "outcome": key[2],
                "metric_family": outcome_metadata(str(key[2]))[0],
                "horizon_hours": outcome_metadata(str(key[2]))[1],
                "paired_coin_count": len(pairs),
                "paired_coins": ";".join(pairs),
                "actual_independent_event_pairs": actual_count,
                "placebo_independent_event_pairs": placebo_count,
                "coverage_gate_passed": coverage,
                "actual_pair_delta_median": float(actual_delta.median()),
                "actual_pair_positive_fraction": float(actual_delta.gt(0.0).mean()),
                "placebo_pair_delta_median": float(placebo_delta.median()),
                "placebo_pair_positive_fraction": float(placebo_delta.gt(0.0).mean()),
                "actual_minus_placebo_pair_lift_median": float(lift.median()),
                "lift_positive_fraction": float(lift.gt(0.0).mean()),
                "actual_max_absolute_state_smd": actual_balance["max_absolute_smd"],
                "actual_median_absolute_state_smd": actual_balance["median_absolute_smd"],
                "placebo_max_absolute_state_smd": placebo_balance["max_absolute_smd"],
                "placebo_median_absolute_state_smd": placebo_balance["median_absolute_smd"],
                "direction_prediction": False,
                "profit_optimization": False,
            }
        )
    return DataFrame(rows)


def leave_one_coin_out_results(pair_period: DataFrame) -> DataFrame:
    eligible = pair_period.loc[pair_period["pair_period_coverage_eligible"]].copy()
    actual = eligible.loc[eligible["attribute_assignment"].eq("actual")].copy()
    placebo = eligible.loc[eligible["attribute_assignment"].eq("shuffled_attribute")].copy()
    keys = ["question_id", "pair", "period", "outcome"]
    paired = actual.merge(
        placebo,
        on=keys,
        how="inner",
        suffixes=("__actual", "__placebo"),
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for key, group in paired.groupby(
        ["question_id", "period", "outcome"], observed=True, sort=False
    ):
        for omitted in sorted(group["pair"].astype(str).unique()):
            retained = group.loc[~group["pair"].eq(omitted)]
            if retained.empty:
                continue
            actual_delta = retained["role_oriented_delta_mean__actual"]
            lift = actual_delta - retained["role_oriented_delta_mean__placebo"]
            rows.append(
                {
                    "question_id": key[0],
                    "period": key[1],
                    "outcome": key[2],
                    "omitted_pair": omitted,
                    "remaining_coin_count": int(retained["pair"].nunique()),
                    "actual_pair_delta_median": float(actual_delta.median()),
                    "actual_pair_positive_fraction": float(actual_delta.gt(0.0).mean()),
                    "actual_minus_placebo_pair_lift_median": float(lift.median()),
                    "lift_positive_fraction": float(lift.gt(0.0).mean()),
                }
            )
    return DataFrame(rows)


def attach_vp_level_persistence(
    events: DataFrame,
    *,
    base: DataFrame,
    cache_path: Path,
) -> DataFrame:
    mapping = {
        "hvn_above": "vp_hvn_above",
        "hvn_below": "vp_hvn_below",
        "lvn_above": "vp_lvn_above",
        "lvn_below": "vp_lvn_below",
        "poc": "vp_poc",
    }
    cache = pd.read_parquet(
        cache_path,
        columns=["available_at", "vp_native_half_width", *mapping.values()],
    ).sort_values("available_at")
    cache["available_at"] = normalize_dates(cache["available_at"])
    width = numeric_series(cache["vp_native_half_width"])
    persistence = DataFrame({"available_at": cache["available_at"]})
    for level_name, column in mapping.items():
        persistence[f"persistence__{level_name}"] = persistence_bars(
            numeric_series(cache[column]), width
        )
    aligned = pd.merge_asof(
        base[["date"]].sort_values("date"),
        persistence,
        left_on="date",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    indexes = events["base_index"].to_numpy(dtype=np.int64)
    values = np.full(len(events), np.nan, dtype=float)
    names = events["level_name"].astype(str).to_numpy()
    for level_name in mapping:
        selected = names == level_name
        if selected.any():
            values[selected] = numeric_array(aligned[f"persistence__{level_name}"])[
                indexes[selected]
            ]
    output = events.copy()
    output["state_vp_level_persistence_bars"] = values
    return output


def persistence_bars(level: Series, width: Series) -> Series:
    previous_level = level.shift(1)
    tolerance = pd.concat([width, width.shift(1)], axis=1).max(axis=1)
    same = (
        level.notna()
        & previous_level.notna()
        & tolerance.notna()
        & level.sub(previous_level).abs().le(tolerance)
    )
    groups = (~same).cumsum()
    result = level.groupby(groups).cumcount().add(1).astype(float)
    return result.where(level.notna())


def validate_pair_output(
    path: Path,
    *,
    pair: str,
    request_sha256: str,
) -> DataFrame:
    frame = pd.read_parquet(path)
    if frame.empty or set(frame["pair"].astype(str)) != {pair}:
        raise ValueError(f"Existing output has an incompatible pair: {path}")
    if set(frame["output_schema_version"].astype(int)) != {OUTPUT_SCHEMA_VERSION}:
        raise ValueError(f"Existing output has an incompatible schema: {path}")
    if set(frame["run_request_sha256"].astype(str)) != {request_sha256}:
        raise ValueError(f"Existing output belongs to another request: {path}")
    return frame


def combine_pair_outputs(
    *,
    pair_dir: Path,
    pairs: Sequence[str],
    request_sha256: str,
) -> DataFrame:
    frames = []
    for pair in pairs:
        path = pair_dir / f"{pair_stem(pair)}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(validate_pair_output(path, pair=pair, request_sha256=request_sha256))
    return pd.concat(frames, ignore_index=True)


def repeatability_screen(cells: DataFrame) -> DataFrame:
    eligible = cells.loc[
        cells["matched_events"].ge(MIN_CELL_EVENTS)
        & cells["max_absolute_state_smd"].le(MAX_STATE_SMD)
        & cells["median_absolute_state_smd"].le(MAX_MEDIAN_STATE_SMD)
    ].copy()
    if eligible.empty:
        return DataFrame()
    frames = []
    for question in QUESTIONS:
        selected = eligible.loc[eligible["question_id"].eq(question.id)].copy()
        if selected.empty:
            continue
        raw_columns = [f"delta_mean__{outcome}" for outcome in question.outcomes]
        role_columns = [f"role_delta_mean__{outcome}" for outcome in question.outcomes]
        identifiers = [
            "pair",
            "period",
            "question_id",
            "source_timeframe",
            "level_name",
            "zone_method",
            "attribute_assignment",
            "level_role",
            "matched_events",
            "max_absolute_state_smd",
            "median_absolute_state_smd",
        ]
        raw = selected.melt(
            id_vars=identifiers,
            value_vars=[column for column in raw_columns if column in selected],
            var_name="outcome",
            value_name="raw_delta",
        )
        raw["outcome"] = raw["outcome"].str.removeprefix("delta_mean__")
        role = selected.melt(
            id_vars=identifiers,
            value_vars=[column for column in role_columns if column in selected],
            var_name="outcome",
            value_name="role_oriented_delta",
        )
        role["outcome"] = role["outcome"].str.removeprefix("role_delta_mean__")
        frames.append(raw.merge(role, on=[*identifiers, "outcome"], validate="one_to_one"))
    if not frames:
        return DataFrame()
    long = pd.concat(frames, ignore_index=True).dropna(subset=["raw_delta", "role_oriented_delta"])
    metadata = long["outcome"].apply(outcome_metadata)
    long["metric_family"] = [item[0] for item in metadata]
    long["horizon_hours"] = [item[1] for item in metadata]
    long["raw_positive"] = long["raw_delta"].gt(0.0)
    long["role_positive"] = long["role_oriented_delta"].gt(0.0)
    keys = [
        "question_id",
        "source_timeframe",
        "level_name",
        "zone_method",
        "attribute_assignment",
        "level_role",
        "outcome",
        "metric_family",
        "horizon_hours",
    ]
    return (
        long.groupby(keys, observed=True, dropna=False, sort=False)
        .agg(
            pair_period_cells=("pair", "size"),
            pair_count=("pair", "nunique"),
            period_count=("period", "nunique"),
            matched_events=("matched_events", "sum"),
            raw_delta_median=("raw_delta", "median"),
            raw_positive_fraction=("raw_positive", "mean"),
            role_oriented_delta_median=("role_oriented_delta", "median"),
            role_positive_fraction=("role_positive", "mean"),
            max_state_smd=("max_absolute_state_smd", "max"),
        )
        .reset_index()
    )


def integrity_record(
    cells: DataFrame,
    results: Sequence[dict[str, Any]],
    *,
    independent_pairs: DataFrame,
) -> dict[str, Any]:
    adequate = cells.loc[cells["matched_events"].ge(MIN_CELL_EVENTS)]
    usable = adequate.loc[
        adequate["max_absolute_state_smd"].le(MAX_STATE_SMD)
        & adequate["median_absolute_state_smd"].le(MAX_MEDIAN_STATE_SMD)
    ]
    return {
        "created_at_utc": utc_now(),
        "pairs": len(results),
        "pair_failures": sum(row["status"] == "failed" for row in results),
        "source_event_rows": int(sum(row.get("source_event_rows", 0) for row in results)),
        "cell_rows": len(cells),
        "cells_with_at_least_20_matches": len(adequate),
        "usable_cells": len(usable),
        "matched_event_pairs_before_overlap_purge": int(
            sum(row.get("matched_event_pairs_before_overlap_purge", 0) for row in results)
        ),
        "independent_event_pairs_after_horizon_purge": len(independent_pairs),
        "independence_hours_by_question": INDEPENDENCE_HOURS_BY_QUESTION,
        "independent_pairs_by_question_assignment": {
            f"{question}|{assignment}": int(len(group))
            for (question, assignment), group in independent_pairs.groupby(
                ["question_id", "attribute_assignment"], observed=True
            )
        },
        "actual_attribute_cells": int(cells["attribute_assignment"].eq("actual").sum()),
        "shuffled_attribute_cells": int(
            cells["attribute_assignment"].eq("shuffled_attribute").sum()
        ),
        "questions_present": sorted(cells["question_id"].unique().tolist()),
        "periods_present": sorted(cells["period"].unique().tolist()),
        "pre_distance_atr_abs_difference_max": float(
            cells["pre_distance_atr_abs_difference_max"].max()
        ),
        "event_separation_hours_min": float(cells["event_separation_hours_min"].min()),
        "state_features": sorted(
            {
                feature
                for task in results
                if task.get("status") != "failed"
                for feature in task.get("state_features", ())
            }
        ),
        "attributes_use_outcomes_for_tiering": False,
        "direction_prediction": False,
        "profit_optimization": False,
    }


def pair_stem(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")


if __name__ == "__main__":
    raise SystemExit(main())

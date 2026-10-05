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

from user_data.Custom_Launcher.research.context_features.market_reaction_zone_cluster_generation0 import (  # noqa: E501
    component_dependency_group,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    LARGE_ARTIFACT_ROOT,
    OUTPUT_ROOT,
    atomic_write_json,
    atomic_write_parquet,
    bool_array,
    level_cache_path,
    level_specs,
    load_manifest,
    manifest_storage_paths,
    normalize_dates,
    numeric_array,
    prepare_base_market_frame,
    resolved_level_values,
    sha256_file,
    utc_now,
    validate_worker_count,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_localization import (  # noqa: E501
    SELECTED_LEVEL_COLUMNS,
    AlignedLevel,
    causal_local_state,
    causal_market_context,
    causal_source_state,
    outcome_metadata,
    pair_stem,
    purge_overlapping_event_pairs_by_key,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_volume_profile_roles import (  # noqa: E501
    AttributeProfile,
    assign_attribute_tertiles,
    attach_states,
    generation0_event_source,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_localization import (  # noqa: E501
    eligible_period_events,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation2_vp_roles import (  # noqa: E501
    CORE_STATE_FEATURES,
    Question,
    atlas_event_source,
    attach_vp_level_persistence,
    matched_attribute_event_pairs,
    matched_pair_balance,
    stable_json_sha256,
)


FROZEN_BATCH = OUTPUT_ROOT / "generation2_review" / "g3_frozen_branch_batch.json"
REPORT_ROOT = OUTPUT_ROOT / "generation3_branches" / "g3a_thin_lvn_attribution"
ARTIFACT_ROOT = LARGE_ARTIFACT_ROOT / "generation3_branches" / "g3a_thin_lvn_attribution"
OUTPUT_SCHEMA_VERSION = 2
MIN_PAIR_PERIOD_EPISODES = 5
MIN_COHORT_PERIOD_EPISODES = 50
MIN_COHORT_PERIOD_COINS = 5
MAX_STATE_SMD = 0.50
MAX_MEDIAN_STATE_SMD = 0.20
MATCH_SEPARATION_HOURS = 6

OUTCOMES = (
    "contact_volume_ratio",
    "dwell_fraction_h4",
    "crossings_h4",
    "volume_ratio_h48",
)
OUTCOME_INDEPENDENCE_HOURS = {
    "contact_volume_ratio": 6,
    "dwell_fraction_h4": 6,
    "crossings_h4": 6,
    "volume_ratio_h48": 48,
}
QUESTION = Question(
    id="g3a_lvn_thinness_after_surviving_cluster",
    timeframe="1h",
    zone="wide_base_atr",
    profile=AttributeProfile(
        "lvn_thinness",
        ("lvn_above", "lvn_below"),
        "level_score",
        "activity_transit",
    ),
    outcomes=OUTCOMES,
)
GEOMETRY_MATCH_FEATURES = (
    "state_peer_level_count",
    "state_peer_contact_count",
    "state_peer_dependency_group_count",
    "state_peer_contact_dependency_group_count",
    "state_peer_connected_dependency_group_count",
    "state_peer_contact_connected_dependency_group_count",
    "state_peer_higher_tf_count",
    "state_peer_daily_count",
)
GEOMETRY_OBSERVATION_FEATURES = (
    *GEOMETRY_MATCH_FEATURES,
    "state_peer_prior_range_count",
    "state_peer_bollinger_count",
    "state_peer_round_count",
)
STATE_FEATURES = (*CORE_STATE_FEATURES, *GEOMETRY_MATCH_FEATURES)


@dataclass(frozen=True)
class PairTask:
    pair: str
    cohort: str
    manifest_path: str
    run_id: str
    overwrite: bool
    request_sha256: str
    cache_sha256_by_timeframe: tuple[tuple[str, str], ...]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 3A thin-LVN component attribution. It compares high and low "
            "causal LVN thinness only where a contacted independent peer cluster remains "
            "after removing the LVN. It does not predict direction or optimize profit."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cohort", choices=("large", "meme"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    validate_worker_count(args.workers, manifest=manifest)
    validate_frozen_branch(args.cohort, manifest)
    pairs = select_pairs(manifest, args.pairs)
    contracts = source_contracts(
        cohort=args.cohort,
        manifest=manifest,
        manifest_path=args.manifest,
        pairs=pairs,
    )
    request = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "frozen_batch_sha256": sha256_file(FROZEN_BATCH),
        "cohort": args.cohort,
        "pairs": list(pairs),
        "question": question_contract(),
        "state_features": list(STATE_FEATURES),
        "geometry_observation_features": list(GEOMETRY_OBSERVATION_FEATURES),
        "source_contracts": contracts,
        "component_scope": {
            "source_timeframes": list(manifest["data"]["source_timeframes"]),
            "zone": "wide_base_atr",
            "independent_dependency_groups": True,
            "requires_peer_cluster_after_lvn_removal": True,
            "requires_two_contacted_peer_levels": True,
            "requires_two_connected_contacted_peer_dependency_groups": True,
        },
        "matching": {
            "attribute_tiers": "within-pair period level approach tertiles",
            "attribute_placebo": "deterministic within-stratum shuffle",
            "minimum_high_low_separation_hours": MATCH_SEPARATION_HOURS,
            "outcome_independence_hours": OUTCOME_INDEPENDENCE_HOURS,
            "pre_distance_atr_caliper": 0.10,
        },
        "prefix_equivalence_rule": (
            "Current data may be used at frozen historical event times only after exact "
            "base-metric and selected-level/source-availability equivalence succeeds."
        ),
        "direction_prediction": False,
        "profit_optimization": False,
    }
    request_sha256 = stable_json_sha256(request)
    run_dir = REPORT_ROOT / args.run_id
    artifact_dir = ARTIFACT_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    record_path = run_dir / "g3a_run_record.json"
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
            "Generation 2 high-versus-low LVN thinness and same-LVN near-miss results "
            "before direct attribution to the surviving peer cluster."
        ),
        "hypothesis": (
            "High one-hour LVN thinness retains greater contact or later activity than "
            "low thinness after both events contact a matched peer cluster that remains "
            "connected without the LVN."
        ),
        "strongest_alternative": (
            "The broad peer cluster or already-active state, not the LVN, explains the "
            "earlier activity difference."
        ),
        "controls": [
            "low-thinness LVN in the same level, period, approach and wide-zone definition",
            "deterministic shuffled thinness label",
            "surviving non-LVN peer-cluster requirement",
            "at least two peer levels contacted on the same candle",
            "continuous local, source, BTC, breadth, density and component-geometry matching",
            "outcome-specific future-path independence and period-end embargo",
            "per-pair, per-period, pair-equal cohort and leave-one-coin-out reporting",
        ],
        "workers": args.workers,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(record, record_path)
    tasks = [
        PairTask(
            pair=pair,
            cohort=args.cohort,
            manifest_path=str(args.manifest.resolve()),
            run_id=args.run_id,
            overwrite=args.overwrite,
            request_sha256=request_sha256,
            cache_sha256_by_timeframe=tuple(
                (str(row["timeframe"]), str(row["current_cache_sha256"]))
                for row in contracts
                if row["pair"] == pair
            ),
        )
        for pair in pairs
    ]
    try:
        results = run_tasks(tasks, workers=args.workers)
        inventory = DataFrame(results)
        atomic_write_parquet(inventory, run_dir / "g3a_pair_inventory.parquet")
        failures = inventory.loc[inventory["status"].eq("failed")]
        if not failures.empty:
            raise RuntimeError(
                f"{len(failures)} pair task(s) failed; inspect g3a_pair_inventory.parquet."
            )
        prefix = combine_pair_outputs(
            directory=run_dir / "pair_prefix_audits",
            pairs=pairs,
            request_sha256=request_sha256,
        )
        coverage = combine_pair_outputs(
            directory=run_dir / "pair_geometry_coverage",
            pairs=pairs,
            request_sha256=request_sha256,
        )
        matched = combine_pair_outputs(
            directory=artifact_dir / "pair_event_matches",
            pairs=pairs,
            request_sha256=request_sha256,
        )
        independent = outcome_independent_pairs(matched)
        pair_period = pair_period_results(independent)
        cohort_period = cohort_period_results(pair_period, independent)
        leave_one_out = leave_one_coin_out_results(pair_period)
        atomic_write_parquet(prefix, run_dir / "g3a_prefix_equivalence_audit.parquet")
        atomic_write_parquet(coverage, run_dir / "g3a_geometry_coverage.parquet")
        atomic_write_parquet(pair_period, run_dir / "g3a_pair_period_results.parquet")
        atomic_write_parquet(cohort_period, run_dir / "g3a_cohort_period_results.parquet")
        atomic_write_parquet(leave_one_out, run_dir / "g3a_leave_one_coin_out.parquet")
        atomic_write_parquet(
            independent,
            artifact_dir / "g3a_independent_outcome_pairs.parquet",
        )
        integrity = integrity_record(
            prefix=prefix,
            coverage=coverage,
            matched=matched,
            independent=independent,
            results=results,
        )
        atomic_write_json(integrity, run_dir / "g3a_integrity.json")
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "pair_tasks": results,
                "prefix_audit_rows": len(prefix),
                "geometry_coverage_rows": len(coverage),
                "matched_event_pairs_before_outcome_purge": len(matched),
                "independent_outcome_pairs": len(independent),
                "pair_period_rows": len(pair_period),
                "cohort_period_rows": len(cohort_period),
                "leave_one_coin_out_rows": len(leave_one_out),
                "integrity": str(run_dir / "g3a_integrity.json"),
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


def validate_frozen_branch(cohort: str, manifest: dict[str, Any]) -> None:
    frozen = json.loads(FROZEN_BATCH.read_text(encoding="utf-8"))
    branches = {branch["id"] for branch in frozen["branches"]}
    if frozen.get("status") != "frozen_before_generation3_reaction_outcomes":
        raise ValueError("The frozen Generation 3 batch is not ready.")
    if "g3a_thin_lvn_component_attribution" not in branches:
        raise ValueError("The frozen Generation 3A branch is unavailable.")
    boundary = frozen["research_boundary"]
    if boundary.get("direction_prediction") is not False:
        raise ValueError("Generation 3A must keep direction disabled.")
    if boundary.get("profit_optimization") is not False:
        raise ValueError("Generation 3A must keep profit optimization disabled.")
    scope_key = "large_coin_scope" if cohort == "large" else "frozen_meme_scope"
    allowed = set(frozen["common_scope"][scope_key])
    requested = set(manifest["data"]["pairs"])
    if requested != allowed:
        raise ValueError(
            f"The {cohort} manifest does not match the frozen cohort: "
            f"missing={sorted(allowed - requested)}, extra={sorted(requested - allowed)}"
        )


def select_pairs(manifest: dict[str, Any], requested: str) -> tuple[str, ...]:
    allowed = tuple(manifest["data"]["pairs"])
    if requested.strip().lower() == "all":
        return allowed
    selected = tuple(value.strip() for value in requested.split(",") if value.strip())
    unknown = sorted(set(selected).difference(allowed))
    if not selected or unknown:
        raise ValueError(f"Invalid pair selection: selected={selected}, unknown={unknown}")
    return selected


def question_contract() -> dict[str, Any]:
    return {
        "id": QUESTION.id,
        "timeframe": QUESTION.timeframe,
        "zone": QUESTION.zone,
        "attribute_profile": QUESTION.profile.__dict__,
        "outcomes": list(QUESTION.outcomes),
    }


def event_source(
    *,
    cohort: str,
    manifest: dict[str, Any],
    manifest_path: Path,
    pair: str,
    timeframe: str,
) -> tuple[Path, Path, dict[str, Any]]:
    if cohort == "large":
        event_path, metadata_path = generation0_event_source(pair, timeframe)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return event_path, metadata_path, metadata
    return atlas_event_source(
        manifest=manifest,
        manifest_path=manifest_path,
        pair=pair,
        timeframe=timeframe,
    )


def source_contracts(
    *,
    cohort: str,
    manifest: dict[str, Any],
    manifest_path: Path,
    pairs: Sequence[str],
) -> list[dict[str, Any]]:
    rows = []
    manifest_sha = sha256_file(manifest_path)
    storage = manifest_storage_paths(manifest)
    frozen_at = pd.Timestamp(json.loads(FROZEN_BATCH.read_text())["frozen_at_utc"])
    for pair in pairs:
        for timeframe in manifest["data"]["source_timeframes"]:
            cache_path = level_cache_path(
                pair,
                timeframe,
                ("core", "generic"),
                cache_dir=storage.cache_dir,
            )
            cache_metadata_path = cache_path.with_suffix(".meta.json")
            cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
            validate_cache_boundaries(
                cache_metadata,
                source=cache_path,
                manifest_sha=manifest_sha,
            )
            indicator_mismatches = indicator_hash_mismatches(cache_metadata)
            if indicator_mismatches:
                raise ValueError(f"Indicator source changed for {cache_path}.")

            event_reference = frozen_event_reference(
                cohort=cohort,
                manifest=manifest,
                manifest_path=manifest_path,
                pair=pair,
                timeframe=timeframe,
            )
            event_path: Path | None = None
            event_metadata_path: Path | None = None
            recorded_cache_sha256 = sha256_file(cache_path)
            reference_mode = "pre_outcome_cache_snapshot"
            if event_reference is not None:
                event_path, event_metadata_path, event_metadata = event_reference
                validate_cache_boundaries(
                    event_metadata,
                    source=event_path,
                    manifest_sha=manifest_sha,
                )
                if Path(str(event_metadata["cache"])).resolve() != cache_path.resolve():
                    raise ValueError(f"Event source points to another cache: {event_path}")
                recorded_cache_sha256 = str(event_metadata["cache_sha256"])
                reference_mode = "frozen_event_metadata"
            else:
                created_at = pd.Timestamp(cache_metadata["created_at_utc"])
                if created_at > frozen_at:
                    raise ValueError(f"Cache was created after the G3 outcome freeze: {cache_path}")
                source_path = Path(str(cache_metadata["source"]))
                if sha256_file(source_path) != cache_metadata.get("source_sha256"):
                    raise ValueError(f"Pre-outcome cache source changed: {source_path}")

            rows.append(
                {
                    "pair": pair,
                    "timeframe": timeframe,
                    "reference_mode": reference_mode,
                    "event_path": str(event_path.resolve()) if event_path else None,
                    "event_bytes": int(event_path.stat().st_size) if event_path else None,
                    "event_modified_ns": (
                        int(event_path.stat().st_mtime_ns) if event_path else None
                    ),
                    "event_metadata_sha256": (
                        sha256_file(event_metadata_path) if event_metadata_path else None
                    ),
                    "cache_path": str(cache_path.resolve()),
                    "cache_metadata_sha256": sha256_file(cache_metadata_path),
                    "recorded_cache_sha256": recorded_cache_sha256,
                    "current_cache_sha256": sha256_file(cache_path),
                }
            )
    return rows


def frozen_event_reference(
    *,
    cohort: str,
    manifest: dict[str, Any],
    manifest_path: Path,
    pair: str,
    timeframe: str,
) -> tuple[Path, Path, dict[str, Any]] | None:
    # The frozen meme atlas intentionally has event surfaces at 1h/4h/8h only.
    # Its causally built 1d cache predates the G3 outcome freeze and remains useful
    # as peer geometry, so that one structural case is frozen directly by cache hash.
    if cohort == "meme" and timeframe == "1d":
        return None
    return event_source(
        cohort=cohort,
        manifest=manifest,
        manifest_path=manifest_path,
        pair=pair,
        timeframe=timeframe,
    )


def validate_cache_boundaries(
    metadata: dict[str, Any],
    *,
    source: Path,
    manifest_sha: str,
) -> None:
    if metadata.get("manifest_sha256") != manifest_sha:
        raise ValueError(f"Source used another manifest: {source}")
    if metadata.get("direction_prediction") is not False:
        raise ValueError(f"Direction boundary failed: {source}")
    if metadata.get("profit_optimization") is not False:
        raise ValueError(f"Profit boundary failed: {source}")


def indicator_hash_mismatches(metadata: dict[str, Any]) -> int:
    recorded = metadata.get("indicator_sha256")
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("Cache metadata does not identify its indicator sources.")
    return sum(
        not (REPO_ROOT / relative).is_file() or sha256_file(REPO_ROOT / relative) != expected
        for relative, expected in recorded.items()
    )


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
    run_dir = REPORT_ROOT / task.run_id
    artifact_dir = ARTIFACT_ROOT / task.run_id
    prefix_path = run_dir / "pair_prefix_audits" / f"{pair_stem(task.pair)}.parquet"
    coverage_path = run_dir / "pair_geometry_coverage" / f"{pair_stem(task.pair)}.parquet"
    matches_path = artifact_dir / "pair_event_matches" / f"{pair_stem(task.pair)}.parquet"
    if (
        all(path.is_file() for path in (prefix_path, coverage_path, matches_path))
        and not task.overwrite
    ):
        validate_pair_output(prefix_path, pair=task.pair, request_sha256=task.request_sha256)
        validate_pair_output(coverage_path, pair=task.pair, request_sha256=task.request_sha256)
        matches = validate_pair_output(
            matches_path,
            pair=task.pair,
            request_sha256=task.request_sha256,
        )
        return {
            "pair": task.pair,
            "status": "existing",
            "matched_event_pairs": len(matches),
            "seconds": round(time.perf_counter() - started, 3),
        }

    manifest_path = Path(task.manifest_path)
    manifest = load_manifest(manifest_path)
    event_path, _, event_metadata = event_source(
        cohort=task.cohort,
        manifest=manifest,
        manifest_path=manifest_path,
        pair=task.pair,
        timeframe="1h",
    )
    events = pd.read_parquet(
        event_path,
        filters=[
            ("control", "==", "actual"),
            ("level_name", "in", list(QUESTION.profile.level_names)),
            ("zone_method", "==", QUESTION.zone),
        ],
    )
    events = eligible_period_events(events, manifest, embargo_hours=48)
    if events.empty:
        raise ValueError(f"No eligible one-hour LVN events for {task.pair}.")
    base = prepare_base_market_frame(task.pair, manifest)
    prefix = prefix_equivalence_audit(
        pair=task.pair,
        cohort=task.cohort,
        manifest=manifest,
        manifest_path=manifest_path,
        base=base,
        lvn_events=events,
        expected_cache_sha256=dict(task.cache_sha256_by_timeframe),
    )
    if not prefix["passed"].all():
        raise ValueError(f"Historical prefix equivalence failed for {task.pair}.")

    local_state = causal_local_state(base).merge(
        causal_market_context(manifest),
        on="date",
        how="left",
        validate="one_to_one",
    )
    aligned = aligned_selected_levels_from_verified_prefix(
        pair=task.pair,
        base=base,
        manifest_path=manifest_path,
        timeframes=tuple(manifest["data"]["source_timeframes"]),
    )
    matrix = np.column_stack([item.level for item in aligned])
    base_atr = numeric_array(base["base_atr"])
    pre_close = numeric_array(base["pre_close"])
    local_state["state_selected_level_density_2atr"] = np.sum(
        np.isfinite(matrix) & (np.abs(matrix - pre_close[:, None]) <= 2.0 * base_atr[:, None]),
        axis=1,
    ).astype(float)
    events = attach_states(
        events,
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
    actual_tiered = assign_attribute_tertiles(
        selected,
        attribute_column=QUESTION.profile.attribute_column,
        assignment="actual",
        seed_key=f"{task.pair}|{QUESTION.id}",
    )
    coverage = geometry_coverage(actual_tiered)
    match_rows: list[dict[str, Any]] = []
    for assignment in ("actual", "shuffled_attribute"):
        tiered = assign_attribute_tertiles(
            selected,
            attribute_column=QUESTION.profile.attribute_column,
            assignment=assignment,
            seed_key=f"{task.pair}|{QUESTION.id}",
        )
        tiered = tiered.loc[
            tiered["peer_cluster_survives_without_lvn"]
            & tiered["state_peer_contact_count"].ge(2)
            & tiered["state_peer_contact_connected_dependency_group_count"].ge(2)
        ].copy()
        match_rows.extend(
            matched_attribute_event_pairs(
                tiered,
                question=QUESTION,
                assignment=assignment,
                state_features=STATE_FEATURES,
                independence_hours=MATCH_SEPARATION_HOURS,
            )
        )
    matches = DataFrame(match_rows)
    if matches.empty:
        raise ValueError(f"No matchable G3A event pairs for {task.pair}.")
    for frame in (prefix, coverage, matches):
        frame["output_schema_version"] = OUTPUT_SCHEMA_VERSION
        frame["run_request_sha256"] = task.request_sha256
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    matches_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(prefix, prefix_path)
    atomic_write_parquet(coverage, coverage_path)
    atomic_write_parquet(matches, matches_path)
    return {
        "pair": task.pair,
        "status": "completed",
        "eligible_lvn_events": len(events),
        "matched_event_pairs": len(matches),
        "prefix_audit_rows": len(prefix),
        "seconds": round(time.perf_counter() - started, 3),
    }


def prefix_equivalence_audit(
    *,
    pair: str,
    cohort: str,
    manifest: dict[str, Any],
    manifest_path: Path,
    base: DataFrame,
    lvn_events: DataFrame,
    expected_cache_sha256: dict[str, str],
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    current = base[
        [
            "date",
            "base_atr",
            "pre_close",
            "pre_range_median_24",
            "pre_volume_median_24",
            "pre_pressure_mean_24",
            "high",
            "low",
            "volume",
            "candle_pressure",
        ]
    ]
    joined = lvn_events.merge(
        current,
        left_on="event_time",
        right_on="date",
        how="left",
        validate="many_to_one",
        suffixes=("__frozen", "__current"),
    )
    current_values = {
        "base_atr": pd.to_numeric(joined["base_atr__current"], errors="coerce"),
        "contact_range_ratio": (joined["high"] - joined["low"]).div(joined["pre_range_median_24"]),
        "contact_volume_ratio": joined["volume"].div(joined["pre_volume_median_24"]),
        "contact_pressure_change": joined["candle_pressure"].sub(joined["pre_pressure_mean_24"]),
        "zone_half_width": pd.Series(
            np.maximum(
                0.5 * pd.to_numeric(joined["base_atr__current"], errors="coerce"),
                0.0005 * pd.to_numeric(joined["level_price"], errors="coerce").abs(),
            ),
            index=joined.index,
        ),
        "pre_distance_atr": pd.to_numeric(joined["level_price"], errors="coerce")
        .sub(joined["pre_close"])
        .abs()
        .div(joined["base_atr__current"]),
    }
    frozen_columns = {
        "base_atr": "base_atr__frozen",
        "contact_range_ratio": "contact_range_ratio",
        "contact_volume_ratio": "contact_volume_ratio",
        "contact_pressure_change": "contact_pressure_change",
        "zone_half_width": "zone_half_width",
        "pre_distance_atr": "pre_distance_atr",
    }
    for metric, current_value in current_values.items():
        mismatches, maximum = equivalence_mismatches(joined[frozen_columns[metric]], current_value)
        rows.append(
            {
                "pair": pair,
                "audit_type": "base_event_metric",
                "timeframe": "1h",
                "metric": metric,
                "rows_checked": len(joined),
                "mismatches": mismatches,
                "max_absolute_difference": maximum,
                "cache_contract_mode": "frozen_event_semantic_prefix",
                "passed": mismatches == 0 and joined["date"].notna().all(),
            }
        )
    manifest_sha = sha256_file(manifest_path)
    storage = manifest_storage_paths(manifest)
    frozen_at = pd.Timestamp(json.loads(FROZEN_BATCH.read_text())["frozen_at_utc"])
    for timeframe in manifest["data"]["source_timeframes"]:
        cache_path = level_cache_path(
            pair,
            timeframe,
            ("core", "generic"),
            cache_dir=storage.cache_dir,
        )
        cache_metadata = json.loads(cache_path.with_suffix(".meta.json").read_text())
        validate_cache_boundaries(
            cache_metadata,
            source=cache_path,
            manifest_sha=manifest_sha,
        )
        indicator_mismatches = indicator_hash_mismatches(cache_metadata)
        current_cache_sha = sha256_file(cache_path)
        if current_cache_sha != expected_cache_sha256.get(timeframe):
            raise ValueError(f"Cache changed after the run request was frozen: {cache_path}")

        event_reference = frozen_event_reference(
            cohort=cohort,
            manifest=manifest,
            manifest_path=manifest_path,
            pair=pair,
            timeframe=timeframe,
        )
        semantic_mismatches = 0
        semantic_rows = 0
        semantic_max = 0.0
        availability_mismatches = 0
        if event_reference is None:
            source_path = Path(str(cache_metadata["source"]))
            source_matches = sha256_file(source_path) == cache_metadata.get("source_sha256")
            created_before_freeze = pd.Timestamp(cache_metadata["created_at_utc"]) <= frozen_at
            exact_cache = True
            semantic_rows = len(pd.read_parquet(cache_path, columns=["available_at"]))
            mode = "pre_outcome_cache_snapshot"
            passed = (
                cache_metadata.get("manifest_sha256") == manifest_sha
                and indicator_mismatches == 0
                and source_matches
                and created_before_freeze
            )
        else:
            event_path, _, metadata = event_reference
            exact_cache = current_cache_sha == metadata["cache_sha256"]
            if not exact_cache:
                (
                    semantic_rows,
                    semantic_mismatches,
                    semantic_max,
                    availability_mismatches,
                ) = selected_level_prefix_audit(event_path, cache_path)
            mode = "exact_frozen_cache_sha256" if exact_cache else "semantic_prefix_equivalence"
            passed = (
                metadata.get("manifest_sha256") == manifest_sha
                and cache_metadata.get("manifest_sha256") == manifest_sha
                and indicator_mismatches == 0
                and (exact_cache or (semantic_rows > 0 and semantic_mismatches == 0))
                and availability_mismatches == 0
            )
        rows.append(
            {
                "pair": pair,
                "audit_type": "cache_contract",
                "timeframe": timeframe,
                "metric": "selected_level_price_and_availability",
                "rows_checked": semantic_rows,
                "mismatches": semantic_mismatches + availability_mismatches,
                "max_absolute_difference": semantic_max,
                "cache_contract_mode": mode,
                "indicator_hash_mismatches": indicator_mismatches,
                "passed": passed,
            }
        )
    return DataFrame(rows)


def selected_level_prefix_audit(
    event_path: Path,
    cache_path: Path,
) -> tuple[int, int, float, int]:
    events = pd.read_parquet(
        event_path,
        filters=[("control", "==", "actual"), ("zone_method", "==", "wide_base_atr")],
        columns=["event_time", "source_available_at", "level_column", "level_price"],
    )
    events = events.loc[events["level_column"].isin(SELECTED_LEVEL_COLUMNS)].copy()
    cache = pd.read_parquet(cache_path)
    events["event_time"] = normalize_dates(events["event_time"]).astype("datetime64[ns, UTC]")
    events["source_available_at"] = normalize_dates(events["source_available_at"]).astype(
        "datetime64[ns, UTC]"
    )
    cache["available_at"] = normalize_dates(cache["available_at"]).astype("datetime64[ns, UTC]")
    aligned = pd.merge_asof(
        events.sort_values("event_time"),
        cache.sort_values("available_at"),
        left_on="event_time",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
        suffixes=("__event", "__cache"),
    ).reset_index(drop=True)
    current = np.full(len(aligned), np.nan, dtype=float)
    for column in aligned["level_column"].drop_duplicates():
        if column not in aligned:
            raise ValueError(f"Current cache is missing selected level column {column}.")
        selected = aligned["level_column"].eq(column)
        current[selected.to_numpy()] = pd.to_numeric(
            aligned.loc[selected, column], errors="coerce"
        ).to_numpy(dtype=float)
    mismatches, maximum = equivalence_mismatches(
        aligned["level_price"], pd.Series(current, index=aligned.index)
    )
    availability_mismatches = int(aligned["source_available_at"].ne(aligned["available_at"]).sum())
    return len(aligned), mismatches, maximum, availability_mismatches


def equivalence_mismatches(left: pd.Series, right: pd.Series) -> tuple[int, float]:
    a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    scale = np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
    different = finite & (np.abs(a - b) > 1e-10 * scale)
    different |= np.isfinite(a) ^ np.isfinite(b)
    maximum = float(np.nanmax(np.abs(a - b))) if finite.any() else 0.0
    return int(different.sum()), maximum


def aligned_selected_levels_from_verified_prefix(
    *,
    pair: str,
    base: DataFrame,
    manifest_path: Path,
    timeframes: Sequence[str],
) -> list[AlignedLevel]:
    manifest = load_manifest(manifest_path)
    storage = manifest_storage_paths(manifest)
    selected: list[AlignedLevel] = []
    for timeframe in timeframes:
        cache_path = level_cache_path(
            pair,
            timeframe,
            ("core", "generic"),
            cache_dir=storage.cache_dir,
        )
        cache = pd.read_parquet(cache_path).sort_values("available_at").reset_index(drop=True)
        cache["available_at"] = normalize_dates(cache["available_at"])
        cache["source_open"] = normalize_dates(cache["source_open"])
        cache = pd.concat([cache, causal_source_state(cache)], axis=1)
        aligned = pd.merge_asof(
            base[["date"]].sort_values("date"),
            cache,
            left_on="date",
            right_on="available_at",
            direction="backward",
            allow_exact_matches=True,
        )
        causal = aligned["available_at"].notna()
        if (aligned.loc[causal, "available_at"] > aligned.loc[causal, "date"]).any():
            raise AssertionError(f"Future level row admitted for {pair} {timeframe}.")
        source_columns = [column for column in aligned if column.startswith("state_source_")]
        aligned_source_state = aligned[source_columns].reset_index(drop=True).copy()
        for spec in level_specs(cache, ("g0b1", "g0b2")):
            if spec.column not in SELECTED_LEVEL_COLUMNS:
                continue
            level = resolved_level_values(aligned, spec, timeframe)
            valid = np.isfinite(level) & (level > 0.0)
            if spec.active_columns:
                active = np.zeros(len(aligned), dtype=bool)
                for column in spec.active_columns:
                    if column in aligned:
                        active |= bool_array(aligned[column])
                valid &= active
            selected.append(
                AlignedLevel(
                    timeframe=timeframe,
                    spec=spec,
                    level=level,
                    valid=valid,
                    source_available=normalize_dates(aligned["available_at"]),
                    source_open=normalize_dates(aligned["source_open"]),
                    source_state=aligned_source_state,
                )
            )
    return selected


def attach_peer_geometry(
    events: DataFrame,
    *,
    base: DataFrame,
    aligned: Sequence[AlignedLevel],
) -> DataFrame:
    output = events.copy()
    for column in GEOMETRY_OBSERVATION_FEATURES:
        output[column] = 0.0
    output["peer_cluster_survives_without_lvn"] = False
    high = numeric_array(base["high"])
    low = numeric_array(base["low"])
    for level_name in QUESTION.profile.level_names:
        targets = [
            (index, item)
            for index, item in enumerate(aligned)
            if item.timeframe == "1h" and item.spec.name == level_name
        ]
        if len(targets) != 1:
            raise ValueError(f"Expected one 1h {level_name} target; found {len(targets)}.")
        source_index, target_item = targets[0]
        target_group = component_dependency_group(target_item.spec)
        peers = [
            item
            for index, item in enumerate(aligned)
            if index != source_index and component_dependency_group(item.spec) != target_group
        ]
        for row_index, row in output.loc[output["level_name"].eq(level_name)].iterrows():
            base_index = int(row["base_index"])
            target = float(target_item.level[base_index])
            frozen_target = float(row["level_price"])
            if not np.isfinite(target) or abs(target - frozen_target) > max(
                1e-10, abs(frozen_target) * 1e-10
            ):
                raise ValueError(
                    f"Current target differs from frozen {level_name} for row {base_index}."
                )
            atr = float(row["base_atr"])
            target_width = max(0.5 * atr, abs(target) * 0.0005)
            intervals: list[tuple[float, float, str]] = []
            contacted_intervals: list[tuple[float, float, str]] = []
            contacted = 0
            higher_timeframe = 0
            daily = 0
            family_counts = {
                "generic_prior_range": 0,
                "generic_bollinger": 0,
                "generic_round_number": 0,
            }
            for peer in peers:
                level = float(peer.level[base_index])
                if not np.isfinite(level) or level <= 0.0:
                    continue
                width = max(0.5 * atr, abs(level) * 0.0005)
                if abs(level - target) > width + target_width:
                    continue
                lower = level - width
                upper = level + width
                dependency_group = component_dependency_group(peer.spec)
                interval = (lower, upper, dependency_group)
                intervals.append(interval)
                is_contacted = bool(high[base_index] >= lower and low[base_index] <= upper)
                contacted += is_contacted
                if is_contacted:
                    contacted_intervals.append(interval)
                higher_timeframe += peer.timeframe != "1h"
                daily += peer.timeframe == "1d"
                if peer.spec.family in family_counts:
                    family_counts[peer.spec.family] += 1
            output.at[row_index, "state_peer_level_count"] = len(intervals)
            output.at[row_index, "state_peer_contact_count"] = contacted
            output.at[row_index, "state_peer_dependency_group_count"] = len(
                {interval[2] for interval in intervals}
            )
            output.at[row_index, "state_peer_contact_dependency_group_count"] = len(
                {interval[2] for interval in contacted_intervals}
            )
            output.at[row_index, "state_peer_connected_dependency_group_count"] = (
                largest_connected_dependency_group_count(intervals)
            )
            output.at[
                row_index, "state_peer_contact_connected_dependency_group_count"
            ] = largest_connected_dependency_group_count(contacted_intervals)
            output.at[row_index, "state_peer_higher_tf_count"] = higher_timeframe
            output.at[row_index, "state_peer_daily_count"] = daily
            output.at[row_index, "state_peer_prior_range_count"] = family_counts[
                "generic_prior_range"
            ]
            output.at[row_index, "state_peer_bollinger_count"] = family_counts["generic_bollinger"]
            output.at[row_index, "state_peer_round_count"] = family_counts["generic_round_number"]
            output.at[row_index, "peer_cluster_survives_without_lvn"] = (
                largest_connected_dependency_group_count(intervals) >= 2
            )
    return output


def largest_connected_interval_count(intervals: Sequence[tuple[float, float]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    largest = 1
    current = 1
    running_upper = float(ordered[0][1])
    for lower, upper in ordered[1:]:
        if float(lower) > running_upper:
            current = 1
            running_upper = float(upper)
        else:
            current += 1
            running_upper = max(running_upper, float(upper))
        largest = max(largest, current)
    return largest


def largest_connected_dependency_group_count(
    intervals: Sequence[tuple[float, float, str]],
) -> int:
    """Count distinct mechanisms in the largest transitively connected segment."""
    if not intervals:
        return 0
    ordered = sorted(intervals, key=lambda interval: (interval[0], interval[1], interval[2]))
    largest = 1
    running_groups = {str(ordered[0][2])}
    running_upper = float(ordered[0][1])
    for lower, upper, dependency_group in ordered[1:]:
        if float(lower) > running_upper:
            running_groups = {str(dependency_group)}
            running_upper = float(upper)
        else:
            running_groups.add(str(dependency_group))
            running_upper = max(running_upper, float(upper))
        largest = max(largest, len(running_groups))
    return largest


def geometry_coverage(tiered: DataFrame) -> DataFrame:
    rows = []
    scopes = {
        "all_high_or_low_lvn": pd.Series(True, index=tiered.index),
        "isolated_lvn": tiered["state_peer_level_count"].eq(0),
        "lvn_only_contact_inside_peer_geometry": tiered["state_peer_level_count"].gt(0)
        & tiered["state_peer_contact_count"].eq(0),
        "peer_cluster_survives_without_lvn": tiered["peer_cluster_survives_without_lvn"],
        "surviving_cluster_and_two_peer_contacts": tiered["peer_cluster_survives_without_lvn"]
        & tiered["state_peer_contact_count"].ge(2),
    }
    for scope, mask in scopes.items():
        selected = tiered.loc[mask]
        for key, group in selected.groupby(
            ["pair", "period", "attribute_tier"], observed=True, sort=False
        ):
            rows.append(
                {
                    "pair": key[0],
                    "period": key[1],
                    "attribute_tier": key[2],
                    "geometry_scope": scope,
                    "events": len(group),
                    "peer_level_count_median": float(group["state_peer_level_count"].median()),
                    "peer_contact_count_median": float(group["state_peer_contact_count"].median()),
                    "peer_dependency_group_count_median": float(
                        group["state_peer_dependency_group_count"].median()
                    ),
                    "peer_contact_dependency_group_count_median": float(
                        group["state_peer_contact_dependency_group_count"].median()
                    ),
                    "peer_connected_dependency_group_count_median": float(
                        group["state_peer_connected_dependency_group_count"].median()
                    ),
                    "peer_contact_connected_dependency_group_count_median": float(
                        group["state_peer_contact_connected_dependency_group_count"].median()
                    ),
                    "higher_timeframe_peer_count_median": float(
                        group["state_peer_higher_tf_count"].median()
                    ),
                    "daily_peer_count_median": float(group["state_peer_daily_count"].median()),
                    "prior_range_peer_count_median": float(
                        group["state_peer_prior_range_count"].median()
                    ),
                    "bollinger_peer_count_median": float(
                        group["state_peer_bollinger_count"].median()
                    ),
                    "round_peer_count_median": float(group["state_peer_round_count"].median()),
                    "direction_prediction": False,
                    "profit_optimization": False,
                }
            )
    return DataFrame(rows)


def outcome_independent_pairs(matched: DataFrame) -> DataFrame:
    frames = []
    for outcome in OUTCOMES:
        selected = matched.copy()
        selected["outcome"] = outcome
        selected["metric_family"] = outcome_metadata(outcome)[0]
        selected["horizon_hours"] = outcome_metadata(outcome)[1]
        selected["raw_delta"] = selected[f"delta__{outcome}"]
        selected["role_oriented_delta"] = selected[f"role_delta__{outcome}"]
        frames.append(selected)
    long = pd.concat(frames, ignore_index=True)
    return purge_overlapping_event_pairs_by_key(
        long,
        separation_hours_by_key=OUTCOME_INDEPENDENCE_HOURS,
        key_column="outcome",
        group_columns=(
            "pair",
            "question_id",
            "attribute_assignment",
            "period",
            "outcome",
        ),
        left_time_column="high_event_time",
        right_time_column="low_event_time",
    )


def pair_period_results(independent: DataFrame) -> DataFrame:
    rows = []
    group_columns = ["question_id", "attribute_assignment", "pair", "period", "outcome"]
    for key, group in independent.groupby(group_columns, observed=True, sort=False):
        valid = group["raw_delta"].notna() & group["role_oriented_delta"].notna()
        if not valid.any():
            continue
        selected = group.loc[valid]
        balance = matched_pair_balance(selected, STATE_FEATURES)
        rows.append(
            {
                "question_id": key[0],
                "attribute_assignment": key[1],
                "pair": key[2],
                "period": key[3],
                "outcome": key[4],
                "metric_family": selected["metric_family"].iloc[0],
                "horizon_hours": selected["horizon_hours"].iloc[0],
                "independent_event_pairs": len(selected),
                "pair_period_coverage_eligible": len(selected) >= MIN_PAIR_PERIOD_EPISODES,
                "raw_delta_mean": float(selected["raw_delta"].mean()),
                "raw_delta_median": float(selected["raw_delta"].median()),
                "raw_positive_fraction": float(selected["raw_delta"].gt(0.0).mean()),
                "role_oriented_delta_mean": float(selected["role_oriented_delta"].mean()),
                "role_oriented_delta_median": float(selected["role_oriented_delta"].median()),
                "role_positive_fraction": float(selected["role_oriented_delta"].gt(0.0).mean()),
                "attribute_separation_median": float(
                    (selected["high_attribute_value"] - selected["low_attribute_value"]).median()
                ),
                "match_distance_median": float(selected["match_distance"].median()),
                "max_absolute_state_smd": balance["max_absolute_smd"],
                "median_absolute_state_smd": balance["median_absolute_smd"],
                "state_features_scored": balance["features_scored"],
                "direction_prediction": False,
                "profit_optimization": False,
            }
        )
    return DataFrame(rows)


def cohort_period_results(pair_period: DataFrame, independent: DataFrame) -> DataFrame:
    eligible = pair_period.loc[pair_period["pair_period_coverage_eligible"]].copy()
    actual = eligible.loc[eligible["attribute_assignment"].eq("actual")]
    placebo = eligible.loc[eligible["attribute_assignment"].eq("shuffled_attribute")]
    keys = ["question_id", "pair", "period", "outcome"]
    paired = actual.merge(
        placebo,
        on=keys,
        suffixes=("__actual", "__placebo"),
        validate="one_to_one",
    )
    rows = []
    for key, group in paired.groupby(
        ["question_id", "period", "outcome"], observed=True, sort=False
    ):
        pairs = sorted(group["pair"].astype(str).unique())
        actual_events = independent.loc[
            independent["question_id"].eq(key[0])
            & independent["period"].eq(key[1])
            & independent["outcome"].eq(key[2])
            & independent["attribute_assignment"].eq("actual")
            & independent["pair"].isin(pairs)
        ]
        placebo_events = independent.loc[
            independent["question_id"].eq(key[0])
            & independent["period"].eq(key[1])
            & independent["outcome"].eq(key[2])
            & independent["attribute_assignment"].eq("shuffled_attribute")
            & independent["pair"].isin(pairs)
        ]
        actual_balance = matched_pair_balance(actual_events, STATE_FEATURES)
        placebo_balance = matched_pair_balance(placebo_events, STATE_FEATURES)
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
                "actual_balance_gate_passed": (
                    actual_balance["max_absolute_smd"] <= MAX_STATE_SMD
                    and actual_balance["median_absolute_smd"] <= MAX_MEDIAN_STATE_SMD
                ),
                "placebo_balance_gate_passed": (
                    placebo_balance["max_absolute_smd"] <= MAX_STATE_SMD
                    and placebo_balance["median_absolute_smd"] <= MAX_MEDIAN_STATE_SMD
                ),
                "direction_prediction": False,
                "profit_optimization": False,
            }
        )
    return DataFrame(rows)


def leave_one_coin_out_results(pair_period: DataFrame) -> DataFrame:
    eligible = pair_period.loc[pair_period["pair_period_coverage_eligible"]].copy()
    actual = eligible.loc[eligible["attribute_assignment"].eq("actual")]
    placebo = eligible.loc[eligible["attribute_assignment"].eq("shuffled_attribute")]
    paired = actual.merge(
        placebo,
        on=["question_id", "pair", "period", "outcome"],
        suffixes=("__actual", "__placebo"),
        validate="one_to_one",
    )
    rows = []
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


def integrity_record(
    *,
    prefix: DataFrame,
    coverage: DataFrame,
    matched: DataFrame,
    independent: DataFrame,
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    failures = [row for row in results if row["status"] == "failed"]
    direction_violations = int(matched["direction_prediction"].ne(False).sum())
    profit_violations = int(matched["profit_optimization"].ne(False).sum())
    expected_independence = independent["outcome"].map(OUTCOME_INDEPENDENCE_HOURS)
    independence_violations = int(
        pd.to_numeric(independent["independence_hours"], errors="coerce")
        .ne(expected_independence)
        .sum()
    )
    return {
        "created_at_utc": utc_now(),
        "passed": (
            not failures
            and bool(prefix["passed"].all())
            and direction_violations == 0
            and profit_violations == 0
            and independence_violations == 0
            and not coverage.empty
            and not independent.empty
        ),
        "failures": failures,
        "prefix_audit_rows": len(prefix),
        "prefix_audit_failures": int(prefix["passed"].ne(True).sum()),
        "geometry_coverage_rows": len(coverage),
        "matched_event_pairs_before_outcome_purge": len(matched),
        "independent_outcome_pairs": len(independent),
        "independent_pairs_by_outcome": {
            str(key): len(group) for key, group in independent.groupby("outcome", observed=True)
        },
        "direction_prediction_violations": direction_violations,
        "profit_optimization_violations": profit_violations,
        "outcome_independence_violations": independence_violations,
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
    *,
    directory: Path,
    pairs: Sequence[str],
    request_sha256: str,
) -> DataFrame:
    frames = []
    for pair in pairs:
        path = directory / f"{pair_stem(pair)}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(validate_pair_output(path, pair=pair, request_sha256=request_sha256))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    raise SystemExit(main())

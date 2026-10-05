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
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import hashlib
import sys
import time
from collections import defaultdict, deque
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
    LARGE_ARTIFACT_ROOT,
    OUTPUT_ROOT,
    LevelSpec,
    atomic_write_json,
    atomic_write_parquet,
    bool_array,
    event_frame,
    future_path_matrices,
    level_cache_path,
    level_specs,
    load_manifest,
    manifest_storage_paths,
    manifest_worker_cap,
    normalize_dates,
    numeric_array,
    prepare_base_market_frame,
    resolved_level_values,
    sha256_file,
    stable_hash_int,
    system_snapshot,
    utc_now,
    validate_cache_metadata,
    validate_worker_count,
)


CLUSTER_SCALES: dict[str, float] = {
    "tight": 0.10,
    "standard": 0.25,
    "wide": 0.50,
}
REPRESENTATION_MODES = ("projected", "held_sensitivity")
REPORT_ROOT = OUTPUT_ROOT / "generation0_clusters"
ARTIFACT_ROOT = LARGE_ARTIFACT_ROOT / "generation0_clusters"
COOLDOWN_BARS = 6
RANDOM_MAX_TRIES_PER_TIER = 192


@dataclass(frozen=True)
class ClusterComponent:
    index: int
    key: str
    source_key: str
    family: str
    timeframe: str
    name: str
    column: str
    representation: str
    interpretation: str
    dynamic_representation: bool
    mechanism: str = "unclassified"
    dependency_group: str = "unclassified"


@dataclass(frozen=True)
class PairTask:
    pair: str
    manifest_path: str
    run_id: str
    representation_modes: tuple[str, ...]
    scales: tuple[str, ...]
    max_rows: int | None
    overwrite: bool
    artifact_root: str


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 0D causal level-cluster atlas. It measures non-directional "
            "market reactions and does not optimize profit or predict final direction."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--representation-modes", default=",".join(REPRESENTATION_MODES))
    parser.add_argument("--scales", default=",".join(CLUSTER_SCALES))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    validate_worker_count(args.workers, manifest=manifest)
    pairs = select_pairs(manifest, args.pairs)
    modes = parse_choices(
        args.representation_modes,
        allowed=set(REPRESENTATION_MODES),
        label="representation mode",
    )
    scales = parse_choices(args.scales, allowed=set(CLUSTER_SCALES), label="cluster scale")
    if args.max_rows is not None and args.max_rows < 300:
        raise ValueError(
            "--max-rows must be at least 300 so future paths and warm-up remain useful."
        )

    report_root, artifact_root = cluster_storage_roots(manifest)
    compact_dir = report_root / args.run_id
    bulky_dir = artifact_root / args.run_id
    compact_dir.mkdir(parents=True, exist_ok=True)
    bulky_dir.mkdir(parents=True, exist_ok=True)
    record_path = compact_dir / "g0d_cluster_run_record.json"
    record = build_run_record(
        manifest=manifest,
        manifest_path=args.manifest,
        run_id=args.run_id,
        pairs=pairs,
        modes=modes,
        scales=scales,
        workers=args.workers,
        max_rows=args.max_rows,
        compact_dir=compact_dir,
        bulky_dir=bulky_dir,
    )
    atomic_write_json(record, record_path)

    try:
        audit = preflight_cache_audit(manifest, args.manifest, pairs)
        atomic_write_parquet(audit, compact_dir / "g0d_cache_audit.parquet")
        if not audit["eligible"].all():
            failures = audit.loc[
                ~audit["eligible"], ["pair", "timeframe", "cache_families", "error"]
            ]
            raise ValueError(f"Cluster cache preflight failed:\n{failures.to_string(index=False)}")

        tasks = [
            PairTask(
                pair=pair,
                manifest_path=str(args.manifest.resolve()),
                run_id=args.run_id,
                representation_modes=modes,
                scales=scales,
                max_rows=args.max_rows,
                overwrite=args.overwrite,
                artifact_root=str(artifact_root),
            )
            for pair in pairs
        ]
        results = run_pair_tasks(tasks, workers=args.workers)
        atomic_write_parquet(DataFrame(results), compact_dir / "g0d_pair_run_inventory.parquet")
        failures = [row for row in results if row["status"] == "failed"]
        if failures:
            raise RuntimeError(f"{len(failures)} pair task(s) failed; inspect the pair inventory.")

        combined_summary, comparison = combine_pair_summaries(
            manifest=manifest,
            bulky_dir=bulky_dir,
            pairs=pairs,
        )
        combined_review_dir = bulky_dir / "review"
        combined_review_dir.mkdir(parents=True, exist_ok=True)
        combined_summary_path = combined_review_dir / "g0d_cluster_summary.parquet"
        control_comparison_path = (
            combined_review_dir / "g0d_cluster_control_comparison.parquet"
        )
        atomic_write_parquet(combined_summary, combined_summary_path)
        atomic_write_parquet(comparison, control_comparison_path)
        review = build_cluster_review(
            manifest=manifest,
            run_id=args.run_id,
            pairs=pairs,
            compact_dir=compact_dir,
            bulky_dir=bulky_dir,
            technical_smoke=args.max_rows is not None,
            required_scales=scales,
            required_modes=modes,
        )
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "cache_audit_rows": len(audit),
                "pair_tasks": results,
                "combined_summary_rows": len(combined_summary),
                "control_comparison_rows": len(comparison),
                "combined_summary": str(combined_summary_path),
                "control_comparison": str(control_comparison_path),
                "review": review,
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


def select_pairs(manifest: dict[str, Any], requested: str) -> list[str]:
    available = list(manifest["data"]["pairs"])
    if requested.strip().lower() == "all":
        return available
    selected = [value.strip() for value in requested.split(",") if value.strip()]
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ValueError(f"Pairs are outside the frozen Generation 0 surface: {unknown}")
    return selected


def cluster_storage_roots(manifest: dict[str, Any]) -> tuple[Path, Path]:
    """Keep legacy roots by default and isolate explicitly configured manifests."""
    if not manifest.get("storage"):
        return REPORT_ROOT, ARTIFACT_ROOT
    storage = manifest_storage_paths(manifest)
    return (
        storage.report_dir.parent / "cluster_reports",
        storage.event_dir.parent / "cluster_artifacts",
    )


def cache_family_groups(manifest: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    configured = set(manifest.get("cache_surface", {}).get("families", ()))
    if not configured:
        return (("core", "generic"), ("sparse",))
    groups: list[tuple[str, ...]] = []
    if {"core", "generic"}.issubset(configured):
        groups.append(("core", "generic"))
    elif configured.intersection({"core", "generic"}):
        raise ValueError("Cluster caches require core and generic together.")
    if "sparse" in configured:
        groups.append(("sparse",))
    unknown = configured.difference({"core", "generic", "sparse"})
    if unknown or not groups:
        raise ValueError(f"Unsupported cluster cache families: {sorted(unknown)}")
    return tuple(groups)


def reporting_cohort_map(manifest: dict[str, Any]) -> dict[str, str]:
    groups = manifest.get("reporting_groups", {})
    if "coin_cohorts" in groups:
        return {
            pair: str(cohort)
            for cohort, members in groups["coin_cohorts"].items()
            for pair in members
        }
    primary = groups.get("primary", {})
    return {
        pair: str(cohort)
        for cohort, members in primary.items()
        for pair in members
    }


def parse_choices(value: str, *, allowed: set[str], label: str) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(selected).difference(allowed))
    if not selected or unknown:
        raise ValueError(f"Invalid {label} selection; unknown={unknown}, selected={selected}")
    return selected


def build_run_record(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    run_id: str,
    pairs: Sequence[str],
    modes: Sequence[str],
    scales: Sequence[str],
    workers: int,
    max_rows: int | None,
    compact_dir: Path,
    bulky_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "started_at_utc": utc_now(),
        "orchestrator_pid": os.getpid(),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "objective": (
            "Test whether causally available clusters of independently calculated price "
            "levels mark repeatable changes in price-path magnitude, range, volume, pressure, "
            "dwell, or crossings beyond single-family density and artificial cluster controls."
        ),
        "hypothesis": (
            "Cross-family and cross-timeframe clusters may identify reaction areas more "
            "reliably than same-family duplicates, shifted clusters, or matched random zones."
        ),
        "baseline": "Generation 0B unchanged single-level atlas and its matched controls.",
        "evidence_interpretation": {
            "retain_lead": (
                "An actual cluster class differs from matched-random and shifted-cluster controls "
                "in repeated chronological periods and a defensible multi-coin scope, and later "
                "comparison shows incremental information beyond its strongest component."
            ),
            "conditional_lead": (
                "The difference repeats only in a named timeframe, coin cohort, width, or "
                "non-directional interpretation state and is labelled at that narrower scope."
            ),
            "revise": (
                "Control matching, cluster density, width, representation, or component coverage "
                "does not permit a fair conclusion but has a specific repair path."
            ),
            "park_or_null": (
                "Controls explain the response, chronological slices disagree, or coverage is too "
                "small. No trading rule or indicator edit follows from this classification."
            ),
        },
        "cluster_classes": [
            "same_family_same_timeframe",
            "different_family_same_timeframe",
            "same_family_cross_timeframe",
            "different_family_cross_timeframe",
        ],
        "nested_zone_test": (
            "Tight, standard, and wide connected-zone constructions are retained together so a "
            "narrow core inside a wider cluster can be compared without selecting the best width."
        ),
        "conflict_definition": (
            "A non-directional local-interpretation conflict contains at least one level whose "
            "Generation 0B behaviour was activity/transit-like and at least one whose behaviour "
            "was acceptance/stickiness-like. It does not mean bullish versus bearish."
        ),
        "controls": [
            "matched_random_cluster_same_state_shape_width_and_density",
            "deterministic_cluster_price_shift_of_two_causal_atr",
            "same_family_cluster_as_duplicate_or_shared-mechanism_reference",
            "component_family_and_leave_one_family_out_structural_ablation",
            "Generation_0B_single_component_benchmark_in_final_review",
        ],
        "random_overlap_rule": (
            "Matched artificial zones may occur on a candle that also contains a real cluster; "
            "whether the artificial zone overlaps a real cluster is retained explicitly. This "
            "avoids deleting the dense market states the control is required to match."
        ),
        "pairs": list(pairs),
        "source_timeframes": list(manifest["data"]["source_timeframes"]),
        "representation_modes": list(modes),
        "cluster_scales": {name: CLUSTER_SCALES[name] for name in scales},
        "horizons_hours": list(manifest["reaction_definition"]["horizons_hours"]),
        "max_rows": max_rows,
        "technical_smoke_not_evidence": max_rows is not None,
        "workers": workers,
        "worker_thread_contract": (
            f"At most {workers} pair workers; numerical-library thread limits are fixed at "
            f"one per process and the manifest cap is {manifest_worker_cap(manifest)}."
        ),
        "system_prelaunch": system_snapshot(manifest_worker_cap(manifest)),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "storage": {
            "compact": str(compact_dir),
            "bulky": str(bulky_dir),
            "cleanup": (
                "Failed pilots and superseded detailed rows may be removed only after the final "
                "replacement passes row, control, metadata, and hash checks where applicable."
            ),
        },
        "expected_runtime": (
            "Technical smokes normally finish in under one minute. The complete ten-pair "
            "surface is expected to take roughly one to two hours with two one-thread workers; "
            "poll at pair-completion boundaries rather than per candle."
        ),
        "expected_outputs": [
            str(compact_dir / "g0d_cache_audit.parquet"),
            str(compact_dir / "g0d_pair_run_inventory.parquet"),
            str(bulky_dir / "review" / "g0d_cluster_summary.parquet"),
            str(bulky_dir / "review" / "g0d_cluster_control_comparison.parquet"),
            str(compact_dir / "g0d_repeatability_screen.parquet"),
            str(compact_dir / "g0d_integrity.json"),
            str(bulky_dir / "pair_events"),
            str(bulky_dir / "pair_summaries"),
            str(bulky_dir / "review"),
        ],
        "summary_script": (
            "The same runner creates the integrity audit, paired-control cells, compact "
            "repeatability screen, and run record after all pair workers complete."
        ),
        "direction_prediction": False,
        "profit_optimization": False,
        "stop_conditions": [
            "future source availability detected",
            "frozen cache or manifest mismatch",
            "matched-random coverage below 75 percent for any pair",
            "missing required cluster class, scale, or representation in a full evidence run",
            "pair worker failure",
        ],
    }


def preflight_cache_audit(
    manifest: dict[str, Any], manifest_path: Path, pairs: Sequence[str]
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    cache_dir = manifest_storage_paths(manifest).cache_dir
    for pair in pairs:
        for timeframe in manifest["data"]["source_timeframes"]:
            for families in cache_family_groups(manifest):
                path = level_cache_path(
                    pair,
                    timeframe,
                    families,
                    cache_dir=cache_dir,
                )
                row: dict[str, Any] = {
                    "pair": pair,
                    "timeframe": timeframe,
                    "cache_families": ",".join(families),
                    "path": str(path),
                    "eligible": False,
                    "error": None,
                }
                try:
                    metadata = validate_cache_metadata(path, manifest_path)
                    frame = pd.read_parquet(path, columns=["source_open", "available_at"])
                    expected_delta = pd.Timedelta(hours=timeframe_hours(timeframe))
                    available = normalize_dates(frame["available_at"])
                    source_open = normalize_dates(frame["source_open"])
                    future_violations = int((available < source_open + expected_delta).sum())
                    row.update(
                        {
                            "eligible": future_violations == 0,
                            "rows": len(frame),
                            "future_availability_violations": future_violations,
                            "sha256": sha256_file(path),
                            "metadata_sha256": metadata.get("manifest_sha256"),
                        }
                    )
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
    return DataFrame(rows)


def timeframe_hours(timeframe: str) -> int:
    value = int(timeframe[:-1])
    unit = timeframe[-1].lower()
    if unit == "h":
        return value
    if unit == "d":
        return value * 24
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def run_pair_tasks(tasks: Sequence[PairTask], *, workers: int) -> list[dict[str, Any]]:
    if workers == 1:
        return [safe_build_pair(task) for task in tasks]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(safe_build_pair, task): task for task in tasks}
        for future in as_completed(future_map):
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
    manifest_path = Path(task.manifest_path)
    manifest = load_manifest(manifest_path)
    pair_stem = pair_file_stem(task.pair)
    pair_dir = Path(task.artifact_root) / task.run_id / "pair_events"
    summary_dir = Path(task.artifact_root) / task.run_id / "pair_summaries"
    pair_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    event_path = pair_dir / f"{pair_stem}.parquet"
    summary_path = summary_dir / f"{pair_stem}.parquet"
    if event_path.is_file() and summary_path.is_file() and not task.overwrite:
        existing = pd.read_parquet(event_path, columns=["control"])
        return {
            "pair": task.pair,
            "status": "existing",
            "event_rows": len(existing),
            "path": str(event_path),
            "seconds": round(time.perf_counter() - started, 3),
        }

    base = prepare_base_market_frame(task.pair, manifest)
    if task.max_rows is not None:
        base = base.tail(task.max_rows).reset_index(drop=True)
    horizons = tuple(int(value) for value in manifest["reaction_definition"]["horizons_hours"])
    paths = future_path_matrices(base, max(horizons))
    all_events: list[DataFrame] = []
    mode_stats: list[dict[str, Any]] = []
    for mode in task.representation_modes:
        values, components, available_by_timeframe = build_component_matrix(
            pair=task.pair,
            base=base,
            manifest=manifest,
            manifest_path=manifest_path,
            representation_mode=mode,
        )
        contacts, actual_zones_by_row, density_state, active_count_state, stats = (
            scan_cluster_contacts(
            base=base,
            values=values,
            components=components,
            available_by_timeframe=available_by_timeframe,
            representation_mode=mode,
            scales=task.scales,
            )
        )
        mode_stats.append(stats)
        actual_raw = dedupe_contact_candidates(
            [row for row in contacts if row["control"] == "actual_cluster"]
        )
        shifted_raw = dedupe_contact_candidates(
            [row for row in contacts if row["control"] == "price_shift_cluster"]
        )
        actual = cluster_event_frame(base, paths, actual_raw, task.pair, horizons)
        shifted = cluster_event_frame(base, paths, shifted_raw, task.pair, horizons)
        if not actual.empty:
            actual["cluster_event_id"] = [
                cluster_event_id(row) for _, row in actual.iterrows()
            ]
            actual["control_pair_id"] = actual["cluster_event_id"]
        random_raw = matched_random_cluster_candidates(
            base=base,
            actual_events=actual,
            actual_zones_by_row=actual_zones_by_row,
            density_state=density_state,
            active_count_state=active_count_state,
            pair=task.pair,
            representation_mode=mode,
        )
        random_events = cluster_event_frame(base, paths, random_raw, task.pair, horizons)
        for frame in (actual, shifted, random_events):
            if not frame.empty:
                all_events.append(frame)

    events = pd.concat(all_events, ignore_index=True) if all_events else DataFrame()
    if events.empty:
        raise ValueError("No eligible cluster events were produced.")
    actual_count = int(events["control"].eq("actual_cluster").sum())
    random_count = int(events["control"].eq("matched_random_cluster").sum())
    random_match_rate = random_count / actual_count if actual_count else 0.0
    if random_match_rate < 0.75:
        raise ValueError(
            "Matched-random cluster coverage is below the frozen 75% technical minimum: "
            f"{random_count}/{actual_count} ({random_match_rate:.1%})."
        )
    events.sort_values(
        ["event_time", "representation_mode", "cluster_scale", "control", "level_price"],
        inplace=True,
        ignore_index=True,
    )
    summary = summarize_cluster_events(events, horizons=horizons)
    atomic_write_parquet(events, event_path)
    atomic_write_parquet(summary, summary_path)
    counts = events.groupby("control", observed=True).size().to_dict()
    return {
        "pair": task.pair,
        "status": "completed",
        "event_rows": len(events),
        "summary_rows": len(summary),
        "control_rows": {str(key): int(value) for key, value in counts.items()},
        "random_match_rate": random_match_rate,
        "mode_stats": mode_stats,
        "path": str(event_path),
        "summary_path": str(summary_path),
        "seconds": round(time.perf_counter() - started, 3),
    }


def build_component_matrix(
    *,
    pair: str,
    base: DataFrame,
    manifest: dict[str, Any],
    manifest_path: Path,
    representation_mode: str,
) -> tuple[np.ndarray, list[ClusterComponent], dict[str, Series]]:
    arrays: list[np.ndarray] = []
    components: list[ClusterComponent] = []
    available_by_timeframe: dict[str, Series] = {}
    cache_dir = manifest_storage_paths(manifest).cache_dir
    for timeframe in manifest["data"]["source_timeframes"]:
        for cache_families in cache_family_groups(manifest):
            batches = (
                ("g0b1", "g0b2")
                if cache_families == ("core", "generic")
                else ("g0b3",)
            )
            cache_path = level_cache_path(
                pair,
                timeframe,
                cache_families,
                cache_dir=cache_dir,
            )
            validate_cache_metadata(cache_path, manifest_path)
            cache = pd.read_parquet(cache_path).sort_values("available_at").reset_index(drop=True)
            cache["available_at"] = normalize_dates(cache["available_at"])
            cache["source_open"] = normalize_dates(cache["source_open"])
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
                raise AssertionError(f"Future cache row admitted for {pair} {timeframe}.")
            if cache_families == ("core", "generic"):
                available_by_timeframe[timeframe] = normalize_dates(aligned["available_at"])
            specs = select_representation_specs(
                level_specs(cache, batches), representation_mode=representation_mode
            )
            duplicated_sources = dynamic_source_keys(level_specs(cache, batches))
            for spec in specs:
                level = resolved_level_values(aligned, spec, timeframe)
                valid = np.isfinite(level) & (level > 0.0)
                if spec.active_columns:
                    active = np.zeros(len(aligned), dtype=bool)
                    for column in spec.active_columns:
                        if column in aligned.columns:
                            active |= bool_array(aligned[column])
                    valid &= active
                level = level.astype(np.float64, copy=True)
                level[~valid] = np.nan
                if not np.isfinite(level).any():
                    continue
                source_key = f"{timeframe}|{spec.family}|{spec.column}"
                component = ClusterComponent(
                    index=len(components),
                    key=(
                        f"{timeframe}|{spec.family}|{spec.name}|{spec.representation}"
                    ),
                    source_key=source_key,
                    family=spec.family,
                    timeframe=timeframe,
                    name=spec.name,
                    column=spec.column,
                    representation=spec.representation,
                    interpretation=component_interpretation(spec),
                    dynamic_representation=(spec.family, spec.column) in duplicated_sources,
                    mechanism=component_mechanism(spec),
                    dependency_group=component_dependency_group(spec),
                )
                arrays.append(level)
                components.append(component)
    if not arrays:
        raise ValueError(f"No level components were available for {pair}.")
    values = np.column_stack(arrays)
    if len({component.source_key for component in components}) != len(components):
        raise AssertionError("Representation selection left duplicate physical source keys.")
    return values, components, available_by_timeframe


def select_representation_specs(
    specs: Sequence[LevelSpec], *, representation_mode: str
) -> list[LevelSpec]:
    grouped: dict[tuple[str, str], list[LevelSpec]] = defaultdict(list)
    for spec in specs:
        grouped[(spec.family, spec.column)].append(spec)
    selected: list[LevelSpec] = []
    wanted = "projected" if representation_mode == "projected" else "held"
    for group in grouped.values():
        representations = {spec.representation for spec in group}
        if {"projected", "held"}.issubset(representations):
            selected.append(next(spec for spec in group if spec.representation == wanted))
        else:
            selected.append(group[0])
    return selected


def dynamic_source_keys(specs: Sequence[LevelSpec]) -> set[tuple[str, str]]:
    representations: dict[tuple[str, str], set[str]] = defaultdict(set)
    for spec in specs:
        key = (spec.family, spec.column)
        representations[key].add(spec.representation)
    return {
        key
        for key, values in representations.items()
        if {"projected", "held"}.issubset(values)
    }


def component_interpretation(spec: LevelSpec) -> str:
    name = spec.name.lower()
    if spec.family == "generic_prior_range":
        return "activity_transit"
    if spec.family == "generic_bollinger" and name.endswith(("upper", "lower")):
        return "activity_transit"
    if spec.family == "generic_round_number" and "nearest" in name:
        return "activity_transit"
    if spec.family == "volume_profile_nodes" and name.startswith("lvn_"):
        return "activity_transit"
    if spec.family == "volume_profile_nodes" and name.startswith("hvn_"):
        return "acceptance_stickiness"
    if spec.family == "volume_profile_settled" and name == "poc":
        return "acceptance_stickiness"
    if spec.family == "volume_profile_explicit_prior" and name == "prior_poc":
        return "acceptance_stickiness"
    return "unclassified"


def component_mechanism(spec: LevelSpec) -> str:
    return {
        "volume_profile_settled": "volume_profile",
        "volume_profile_nodes": "volume_profile",
        "volume_profile_explicit_prior": "volume_profile",
        "confirmed_swing": "confirmed_swing_structure",
        "tlv2_ranked": "trendline_v2",
        "tlv2_forecast_zone": "trendline_v2",
        "geometry_boundary": "pattern_geometry_v2",
        "generic_prior_range": "rolling_price_extreme",
        "generic_rolling_vwap": "rolling_vwap",
        "generic_moving_average": "moving_average",
        "generic_bollinger": "bollinger_band",
        "generic_round_number": "round_number",
        "pattern_reversal": "pattern_reversal",
        "pattern_continuation": "pattern_continuation",
        "pattern_multi_peak": "pattern_multi_peak",
        "pattern_wolfe_wave": "pattern_wolfe_wave",
    }.get(spec.family, spec.family)


def component_dependency_group(spec: LevelSpec) -> str:
    if spec.family in {"generic_moving_average", "generic_bollinger"}:
        return "price_average_family"
    return component_mechanism(spec)


def scan_cluster_contacts(  # noqa: C901 - explicit cluster controls are auditable
    *,
    base: DataFrame,
    values: np.ndarray,
    components: Sequence[ClusterComponent],
    available_by_timeframe: dict[str, Series],
    representation_mode: str,
    scales: Sequence[str],
) -> tuple[
    list[dict[str, Any]],
    dict[int, list[tuple[float, float]]],
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    atr = numeric_array(base["base_atr"])
    pre_close = numeric_array(base["pre_close"])
    high = numeric_array(base["high"])
    low = numeric_array(base["low"])
    last_eligible = len(base) - 48
    contacts: list[dict[str, Any]] = []
    actual_zones_by_row: dict[int, list[tuple[float, float]]] = defaultdict(list)
    density_state = np.zeros(len(base), dtype=np.int16)
    active_count_state = np.zeros(len(base), dtype=np.int16)
    available_clusters = 0
    contacted_clusters = 0
    class_counts: dict[str, int] = defaultdict(int)
    availability_counts: dict[tuple[str, str], int] = defaultdict(int)
    contact_counts: dict[tuple[str, str], int] = defaultdict(int)
    availability_rows: dict[tuple[str, str], int] = defaultdict(int)
    contact_rows: dict[tuple[str, str], int] = defaultdict(int)
    last_availability_row: dict[tuple[str, str], int] = {}
    last_contact_row: dict[tuple[str, str], int] = {}
    for row_index in range(max(last_eligible, 0)):
        if not np.isfinite(atr[row_index]) or atr[row_index] <= 0.0:
            continue
        row_values = values[row_index]
        finite = np.isfinite(row_values)
        if finite.sum() < 2:
            continue
        active_count = int(finite.sum())
        local_density = int(
            (np.abs(row_values[finite] - pre_close[row_index]) <= 2.0 * atr[row_index]).sum()
        )
        active_count_state[row_index] = active_count
        density_state[row_index] = local_density
        by_scale: dict[str, list[dict[str, Any]]] = {}
        for scale in scales:
            clusters = connected_level_clusters(
                values=row_values,
                atr=float(atr[row_index]),
                components=components,
                scale=scale,
                require_dynamic=representation_mode == "held_sensitivity",
            )
            by_scale[scale] = clusters
            available_clusters += len(clusters)
            for cluster in clusters:
                coverage_key = (scale, cluster["cluster_class"])
                availability_counts[coverage_key] += 1
                if last_availability_row.get(coverage_key) != row_index:
                    availability_rows[coverage_key] += 1
                    last_availability_row[coverage_key] = row_index
        add_nested_relationships(by_scale)
        for scale, clusters in by_scale.items():
            for cluster in clusters:
                cluster.update(
                    {
                        "row_index": row_index,
                        "representation_mode": representation_mode,
                        "cluster_scale": scale,
                        "active_level_count": active_count,
                        "local_level_density_2atr": local_density,
                        "density_band": density_band(local_density),
                        **source_availability_metadata(
                            row_index,
                            cluster["component_timeframes_list"],
                            available_by_timeframe,
                            event_time=pd.Timestamp(base["date"].iloc[row_index]),
                        ),
                    }
                )
                if high[row_index] >= cluster["lower"] and low[row_index] <= cluster["upper"]:
                    actual = dict(cluster)
                    actual.update(
                        contacted_component_metadata(
                            cluster=cluster,
                            row_values=row_values,
                            components=components,
                            atr=float(atr[row_index]),
                            candle_high=float(high[row_index]),
                            candle_low=float(low[row_index]),
                            scale=scale,
                            price_shift=0.0,
                        )
                    )
                    actual["control"] = "actual_cluster"
                    contacts.append(actual)
                    actual_zones_by_row[row_index].append(
                        (float(cluster["lower"]), float(cluster["upper"]))
                    )
                    contacted_clusters += 1
                    class_counts[cluster["cluster_class"]] += 1
                    coverage_key = (scale, cluster["cluster_class"])
                    contact_counts[coverage_key] += 1
                    if last_contact_row.get(coverage_key) != row_index:
                        contact_rows[coverage_key] += 1
                        last_contact_row[coverage_key] = row_index

                sign = 1.0 if stable_hash_int(cluster["cluster_signature"]) % 2 == 0 else -1.0
                shift = sign * 2.0 * atr[row_index]
                shifted_lower = cluster["lower"] + shift
                shifted_upper = cluster["upper"] + shift
                if high[row_index] >= shifted_lower and low[row_index] <= shifted_upper:
                    shifted = dict(cluster)
                    shifted.update(
                        contacted_component_metadata(
                            cluster=cluster,
                            row_values=row_values,
                            components=components,
                            atr=float(atr[row_index]),
                            candle_high=float(high[row_index]),
                            candle_low=float(low[row_index]),
                            scale=scale,
                            price_shift=shift,
                        )
                    )
                    shifted.update(
                        {
                            "control": "price_shift_cluster",
                            "source_cluster_center": cluster["center"],
                            "center": cluster["center"] + shift,
                            "lower": shifted_lower,
                            "upper": shifted_upper,
                        }
                    )
                    contacts.append(shifted)
    for row in contacts:
        if row["control"] == "actual_cluster":
            row["control_zone_overlaps_real_cluster"] = True
        else:
            row["control_zone_overlaps_real_cluster"] = overlaps_any_zone(
                row["lower"], row["upper"], actual_zones_by_row.get(row["row_index"], ())
            )
    coverage_rows = []
    for key in sorted(availability_counts):
        scale, cluster_class = key
        available = availability_counts[key]
        contacted = contact_counts.get(key, 0)
        rows_available = availability_rows[key]
        rows_contacted = contact_rows.get(key, 0)
        coverage_rows.append(
            {
                "cluster_scale": scale,
                "cluster_class": cluster_class,
                "available_cluster_observations": available,
                "contacted_cluster_observations": contacted,
                "contact_fraction_of_cluster_observations": contacted / available,
                "rows_with_available_cluster": rows_available,
                "rows_with_contacted_cluster": rows_contacted,
                "chart_contact_fraction_when_class_available": rows_contacted / rows_available,
            }
        )
    return contacts, dict(actual_zones_by_row), density_state, active_count_state, {
        "representation_mode": representation_mode,
        "component_columns": len(components),
        "available_clusters": available_clusters,
        "raw_contacted_clusters": contacted_clusters,
        "raw_class_counts": dict(class_counts),
        "coverage": coverage_rows,
    }


def connected_level_clusters(
    *,
    values: np.ndarray,
    atr: float,
    components: Sequence[ClusterComponent],
    scale: str,
    require_dynamic: bool,
) -> list[dict[str, Any]]:
    multiplier = CLUSTER_SCALES[scale]
    indexes = np.flatnonzero(np.isfinite(values) & (values > 0.0))
    if len(indexes) < 2:
        return []
    level_values = values[indexes]
    widths = np.maximum(multiplier * atr, np.abs(level_values) * 0.0005)
    lower = level_values - widths
    upper = level_values + widths
    order = np.argsort(lower, kind="stable")
    indexes = indexes[order]
    level_values = level_values[order]
    lower = lower[order]
    upper = upper[order]
    clusters: list[dict[str, Any]] = []
    start = 0
    running_upper = float(upper[0])
    for position in range(1, len(indexes) + 1):
        connected = position < len(indexes) and float(lower[position]) <= running_upper
        if connected:
            running_upper = max(running_upper, float(upper[position]))
            continue
        member_indexes = indexes[start:position]
        if len(member_indexes) >= 2:
            member_components = [components[int(index)] for index in member_indexes]
            unique_sources = {component.source_key for component in member_components}
            has_dynamic = any(component.dynamic_representation for component in member_components)
            if len(unique_sources) >= 2 and (not require_dynamic or has_dynamic):
                clusters.append(
                    cluster_record(
                        member_indexes=member_indexes,
                        member_values=values[member_indexes],
                        member_widths=widths[order][start:position],
                        components=member_components,
                        lower=float(lower[start]),
                        upper=float(running_upper),
                        atr=atr,
                    )
                )
        if position < len(indexes):
            start = position
            running_upper = float(upper[position])
    return clusters


def cluster_record(
    *,
    member_indexes: np.ndarray,
    member_values: np.ndarray,
    member_widths: np.ndarray,
    components: Sequence[ClusterComponent],
    lower: float,
    upper: float,
    atr: float,
) -> dict[str, Any]:
    families = sorted({component.family for component in components})
    mechanisms = sorted({component.mechanism for component in components})
    dependency_groups = sorted(
        {component.dependency_group for component in components}
    )
    timeframes = sorted(
        {component.timeframe for component in components}, key=timeframe_hours
    )
    source_keys = sorted({component.source_key for component in components})
    component_keys = sorted(component.key for component in components)
    interpretations = {component.interpretation for component in components}
    cluster_class = classify_cluster(len(families), len(timeframes))
    ablation_component_counts = []
    for family in families:
        keep = np.asarray([component.family != family for component in components], dtype=bool)
        ablation_component_counts.append(
            largest_connected_interval_component(
                member_values[keep], member_widths[keep]
            )
        )
    mechanism_ablation_counts = []
    for mechanism in mechanisms:
        keep = np.asarray(
            [component.mechanism != mechanism for component in components], dtype=bool
        )
        mechanism_ablation_counts.append(
            largest_connected_interval_component(member_values[keep], member_widths[keep])
        )
    signature_source = ";".join(component_keys)
    return {
        "center": (lower + upper) / 2.0,
        "lower": lower,
        "upper": upper,
        "source_cluster_center": (lower + upper) / 2.0,
        "cluster_width_atr": (upper - lower) / atr,
        "component_dispersion_atr": float(np.std(member_values) / atr),
        "component_count": len(components),
        "distinct_source_count": len(source_keys),
        "family_count": len(families),
        "mechanism_count": len(mechanisms),
        "dependency_group_count": len(dependency_groups),
        "timeframe_count": len(timeframes),
        "component_count_bucket": count_bucket(len(components)),
        "component_keys": signature_source,
        "component_source_keys": ";".join(source_keys),
        "component_families": ";".join(families),
        "component_mechanisms": ";".join(mechanisms),
        "component_dependency_groups": ";".join(dependency_groups),
        "component_timeframes": ";".join(timeframes),
        "component_timeframes_list": tuple(timeframes),
        "cluster_signature": hashlib.sha256(signature_source.encode()).hexdigest()[:20],
        "cluster_class": cluster_class,
        "contains_activity_transit": "activity_transit" in interpretations,
        "contains_acceptance_stickiness": "acceptance_stickiness" in interpretations,
        "interpretation_conflict": (
            "activity_transit" in interpretations and "acceptance_stickiness" in interpretations
        ),
        "interpretation_class": interpretation_class(interpretations),
        "same_family_duplicate_reference": len(families) == 1,
        "same_mechanism_reference": len(mechanisms) == 1,
        "independent_mechanism_cluster": len(mechanisms) >= 2,
        "same_dependency_group_reference": len(dependency_groups) == 1,
        "independent_dependency_group_cluster": len(dependency_groups) >= 2,
        "ablation_survives_every_family_removal": bool(
            ablation_component_counts and min(ablation_component_counts) >= 2
        ),
        "indispensable_family_count": int(
            sum(value < 2 for value in ablation_component_counts)
        ),
        "min_cluster_components_after_family_removal": int(
            min(ablation_component_counts, default=0)
        ),
        "max_cluster_components_after_family_removal": int(
            max(ablation_component_counts, default=0)
        ),
        "ablation_survives_every_mechanism_removal": bool(
            mechanism_ablation_counts and min(mechanism_ablation_counts) >= 2
        ),
        "indispensable_mechanism_count": int(
            sum(value < 2 for value in mechanism_ablation_counts)
        ),
        "min_cluster_components_after_mechanism_removal": int(
            min(mechanism_ablation_counts, default=0)
        ),
        "member_index_signature": ";".join(str(int(value)) for value in member_indexes),
        "member_indexes_list": tuple(int(value) for value in member_indexes),
        "nested_relationship": "none",
        "nested_scale_count": 1,
    }


def contacted_component_metadata(
    *,
    cluster: dict[str, Any],
    row_values: np.ndarray,
    components: Sequence[ClusterComponent],
    atr: float,
    candle_high: float,
    candle_low: float,
    scale: str,
    price_shift: float,
) -> dict[str, Any]:
    indexes = np.asarray(cluster["member_indexes_list"], dtype=np.int64)
    values = row_values[indexes] + price_shift
    widths = np.maximum(CLUSTER_SCALES[scale] * atr, np.abs(values) * 0.0005)
    contacted = (candle_high >= values - widths) & (candle_low <= values + widths)
    selected_records = sorted(
        [
            (components[int(index)], int(index), float(value))
            for index, value, is_contacted in zip(
                indexes, values, contacted, strict=True
            )
            if is_contacted
        ],
        key=lambda record: record[0].key,
    )
    selected = [record[0] for record in selected_records]
    component_keys = [component.key for component in selected]
    source_keys = sorted({component.source_key for component in selected})
    families = sorted({component.family for component in selected})
    mechanisms = sorted({component.mechanism for component in selected})
    dependency_groups = sorted(
        {component.dependency_group for component in selected}
    )
    timeframes = sorted(
        {component.timeframe for component in selected}, key=timeframe_hours
    )
    interpretations = {component.interpretation for component in selected}
    if not selected:
        contact_structure = "no_individual_component_contact"
    elif len(selected) == 1:
        contact_structure = "single_component_contact"
    else:
        contact_structure = classify_cluster(len(families), len(timeframes))
    return {
        "contacted_component_count": len(selected),
        "contacted_family_count": len(families),
        "contacted_mechanism_count": len(mechanisms),
        "contacted_dependency_group_count": len(dependency_groups),
        "contacted_timeframe_count": len(timeframes),
        "contacted_distinct_source_count": len(source_keys),
        "contacted_component_count_bucket": count_bucket(len(selected)),
        "contacted_component_keys": ";".join(component_keys),
        "contacted_component_indexes": ";".join(
            str(record[1]) for record in selected_records
        ),
        "contacted_component_prices": ";".join(
            format(record[2], ".17g") for record in selected_records
        ),
        "contacted_component_source_keys": ";".join(source_keys),
        "contacted_component_source_keys_aligned": ";".join(
            component.source_key for component in selected
        ),
        "contacted_component_families": ";".join(families),
        "contacted_component_families_aligned": ";".join(
            component.family for component in selected
        ),
        "contacted_component_mechanisms": ";".join(mechanisms),
        "contacted_component_mechanisms_aligned": ";".join(
            component.mechanism for component in selected
        ),
        "contacted_component_dependency_groups": ";".join(dependency_groups),
        "contacted_component_dependency_groups_aligned": ";".join(
            component.dependency_group for component in selected
        ),
        "contacted_component_timeframes": ";".join(timeframes),
        "contacted_component_timeframes_aligned": ";".join(
            component.timeframe for component in selected
        ),
        "contacted_component_interpretations_aligned": ";".join(
            component.interpretation for component in selected
        ),
        "contacted_component_signature": (
            hashlib.sha256(";".join(component_keys).encode()).hexdigest()[:20]
            if component_keys
            else "none"
        ),
        "contacted_independent_mechanisms": len(mechanisms) >= 2,
        "contacted_independent_dependency_groups": len(dependency_groups) >= 2,
        "contact_structure_class": contact_structure,
        "all_cluster_components_contacted": len(selected) == len(indexes),
        "multiple_components_contacted": len(selected) >= 2,
        "contacted_interpretation_class": interpretation_class(interpretations),
        "contacted_interpretation_conflict": (
            "activity_transit" in interpretations
            and "acceptance_stickiness" in interpretations
        ),
    }


def largest_connected_interval_component(values: np.ndarray, widths: np.ndarray) -> int:
    if len(values) == 0:
        return 0
    lower = values - widths
    upper = values + widths
    order = np.argsort(lower, kind="stable")
    lower = lower[order]
    upper = upper[order]
    largest = 1
    current = 1
    running_upper = float(upper[0])
    for position in range(1, len(values)):
        if float(lower[position]) <= running_upper:
            current += 1
            running_upper = max(running_upper, float(upper[position]))
            largest = max(largest, current)
        else:
            current = 1
            running_upper = float(upper[position])
    return largest


def cluster_event_id(row: Series) -> str:
    key = "|".join(
        (
            str(row["pair"]),
            str(row["representation_mode"]),
            str(row["cluster_scale"]),
            pd.Timestamp(row["event_time"]).isoformat(),
            str(row["cluster_signature"]),
        )
    )
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def classify_cluster(family_count: int, timeframe_count: int) -> str:
    if family_count == 1 and timeframe_count == 1:
        return "same_family_same_timeframe"
    if family_count > 1 and timeframe_count == 1:
        return "different_family_same_timeframe"
    if family_count == 1 and timeframe_count > 1:
        return "same_family_cross_timeframe"
    return "different_family_cross_timeframe"


def interpretation_class(values: set[str]) -> str:
    activity = "activity_transit" in values
    acceptance = "acceptance_stickiness" in values
    if activity and acceptance:
        return "activity_acceptance_conflict"
    if activity:
        return "activity_transit_only"
    if acceptance:
        return "acceptance_stickiness_only"
    return "unclassified_only"


def count_bucket(value: int) -> str:
    return str(value) if value <= 4 else "5_plus"


def density_band(value: int) -> str:
    if value <= 5:
        return "00_05"
    if value <= 10:
        return "06_10"
    if value <= 20:
        return "11_20"
    if value <= 40:
        return "21_40"
    return "41_plus"


def width_band(value: float) -> str:
    if value <= 0.5:
        return "le_0_5atr"
    if value <= 1.0:
        return "0_5_to_1atr"
    if value <= 2.0:
        return "1_to_2atr"
    if value <= 4.0:
        return "2_to_4atr"
    return "gt_4atr"


def add_nested_relationships(by_scale: dict[str, list[dict[str, Any]]]) -> None:
    scale_order = [name for name in CLUSTER_SCALES if name in by_scale]
    for scale_index, scale in enumerate(scale_order):
        for cluster in by_scale[scale]:
            members = set(cluster["member_index_signature"].split(";"))
            containing_scales: list[str] = []
            contained_scales: list[str] = []
            exact_scales = {scale}
            for other_scale in scale_order:
                if other_scale == scale:
                    continue
                for other in by_scale[other_scale]:
                    other_members = set(other["member_index_signature"].split(";"))
                    if members == other_members:
                        exact_scales.add(other_scale)
                    if members.issubset(other_members):
                        containing_scales.append(other_scale)
                    if other_members.issubset(members):
                        contained_scales.append(other_scale)
            cluster["nested_scale_count"] = len(exact_scales)
            if scale_index == 0 and containing_scales:
                cluster["nested_relationship"] = "narrow_core_of_wider_cluster"
            elif scale_index == len(scale_order) - 1 and contained_scales:
                cluster["nested_relationship"] = "wide_envelope_with_narrow_core"
            elif containing_scales or contained_scales:
                cluster["nested_relationship"] = "intermediate_nested_cluster"


def source_availability_metadata(
    row_index: int,
    timeframes: Sequence[str],
    available_by_timeframe: dict[str, Series],
    *,
    event_time: pd.Timestamp,
) -> dict[str, Any]:
    values = [available_by_timeframe[timeframe].iloc[row_index] for timeframe in timeframes]
    finite = [pd.Timestamp(value) for value in values if pd.notna(value)]
    if not finite:
        return {
            "source_available_at": None,
            "oldest_source_available_at": None,
            "oldest_source_update_age_hours": np.nan,
            "newest_source_update_age_hours": np.nan,
            "source_update_age_span_hours": np.nan,
            "oldest_source_update_age_band": "missing",
        }
    oldest = min(finite)
    newest = max(finite)
    oldest_age = (event_time - oldest).total_seconds() / 3600.0
    newest_age = (event_time - newest).total_seconds() / 3600.0
    return {
        "source_available_at": newest,
        "oldest_source_available_at": oldest,
        "oldest_source_update_age_hours": oldest_age,
        "newest_source_update_age_hours": newest_age,
        "source_update_age_span_hours": (newest - oldest).total_seconds() / 3600.0,
        "oldest_source_update_age_band": source_update_age_band(oldest_age),
    }


def source_update_age_band(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    if value <= 1.0:
        return "le_1h"
    if value <= 4.0:
        return "1_to_4h"
    if value <= 8.0:
        return "4_to_8h"
    if value <= 24.0:
        return "8_to_24h"
    return "gt_24h"


def dedupe_contact_candidates(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            row["representation_mode"],
            row["cluster_scale"],
            row["row_index"],
            row["lower"],
        ),
    )
    histories: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
    selected: list[dict[str, Any]] = []
    for row in ordered:
        key = (row["representation_mode"], row["cluster_scale"])
        history = histories[key]
        while history and history[0]["row_index"] < row["row_index"] - COOLDOWN_BARS:
            history.popleft()
        overlapping_recent = any(
            candidate["lower"] <= row["upper"] and row["lower"] <= candidate["upper"]
            for candidate in history
        )
        if not overlapping_recent:
            selected.append(row)
        history.append(row)
    return selected


def cluster_event_frame(
    base: DataFrame,
    paths: dict[str, np.ndarray],
    rows: Sequence[dict[str, Any]],
    pair: str,
    horizons: tuple[int, ...],
) -> DataFrame:
    if not rows:
        return DataFrame()
    ordered = sorted(rows, key=lambda row: (row["row_index"], row["center"]))
    indexes = np.asarray([row["row_index"] for row in ordered], dtype=np.int64)
    levels = np.asarray([row["center"] for row in ordered], dtype=np.float64)
    widths = np.asarray(
        [(row["upper"] - row["lower"]) / 2.0 for row in ordered], dtype=np.float64
    )
    source_available = pd.Series(
        [row.get("source_available_at") for row in ordered], dtype="datetime64[ns, UTC]"
    )
    source_open = pd.Series(pd.NaT, index=range(len(ordered)), dtype="datetime64[ns, UTC]")
    control = str(ordered[0]["control"])
    if any(row["control"] != control for row in ordered):
        raise ValueError("cluster_event_frame received mixed controls.")
    synthetic = LevelSpec(
        name="calculated_cluster_zone",
        family="level_cluster",
        batch="g0d",
        column="synthetic_cluster_center",
    )
    result = event_frame(
        merged=base,
        paths=paths,
        indexes=indexes,
        levels=levels,
        widths=widths,
        spec=synthetic,
        pair=pair,
        timeframe="multi",
        control=control,
        zone_method="cluster_envelope",
        source_available=source_available,
        source_open=source_open,
        horizons=horizons,
        match_tier=[row.get("random_match_tier") for row in ordered],
    )
    reserved = {
        "row_index",
        "center",
        "lower",
        "upper",
        "control",
        "source_available_at",
        "component_timeframes_list",
        "member_indexes_list",
    }
    metadata_columns = sorted(set().union(*(row.keys() for row in ordered)).difference(reserved))
    metadata = DataFrame(
        {column: [row.get(column) for row in ordered] for column in metadata_columns}
    )
    overlapping_columns = sorted(set(result.columns).intersection(metadata.columns))
    if overlapping_columns:
        metadata.drop(columns=overlapping_columns, inplace=True)
    result = pd.concat([result.reset_index(drop=True), metadata.reset_index(drop=True)], axis=1)
    result["cluster_width_band"] = result["cluster_width_atr"].map(width_band)
    return result


def matched_random_cluster_candidates(
    *,
    base: DataFrame,
    actual_events: DataFrame,
    actual_zones_by_row: dict[int, list[tuple[float, float]]],
    density_state: np.ndarray,
    active_count_state: np.ndarray,
    pair: str,
    representation_mode: str,
) -> list[dict[str, Any]]:
    if actual_events.empty:
        return []
    pre_close = numeric_array(base["pre_close"])
    atr = numeric_array(base["base_atr"])
    high = numeric_array(base["high"])
    low = numeric_array(base["low"])
    periods = base["period"].astype(str).to_numpy()
    volatility = base["volatility_band"].to_numpy(dtype=np.int16)
    contact_range = base["contact_range_band"].to_numpy(dtype=np.int16)
    density_labels = np.asarray(
        [density_band(int(value)) for value in density_state], dtype=object
    )
    eligible = np.isfinite(pre_close) & np.isfinite(atr) & (atr > 0.0)
    eligible[max(len(base) - 48, 0) :] = False

    selected: list[dict[str, Any]] = []
    group_columns = ["cluster_scale", "cluster_class", "interpretation_class"]
    for group_key, group in actual_events.groupby(group_columns, observed=True, sort=False):
        blocked = np.zeros(len(base), dtype=bool)
        rng = np.random.default_rng(stable_hash_int(f"{pair}|{representation_mode}|{group_key}"))
        shuffled = group.sample(
            frac=1.0, random_state=stable_hash_int(str(group_key)) % 2**32
        )
        for _, actual in shuffled.iterrows():
            period = str(actual["period"])
            approach = str(actual["approach_state"])
            wanted_volatility = int(actual["volatility_band"])
            wanted_range = int(actual["contact_range_band"])
            wanted_density = str(actual["density_band"])
            distance_atr = float(actual["pre_distance_atr"])
            half_width_atr = float(actual["zone_half_width_atr"])
            if not np.isfinite(distance_atr) or not np.isfinite(half_width_atr):
                continue
            sign = 1.0 if approach == "from_below" else -1.0 if approach == "from_above" else 0.0
            if sign != 0.0:
                distance_atr = max(distance_atr, half_width_atr * 1.01)
            tier_masks = (
                (
                    "exact_period_volatility_range_density",
                    (periods == period)
                    & (volatility == wanted_volatility)
                    & (contact_range == wanted_range)
                    & (density_labels == wanted_density),
                ),
                (
                    "relaxed_density",
                    (periods == period)
                    & (volatility == wanted_volatility)
                    & (contact_range == wanted_range),
                ),
                (
                    "relaxed_range_and_density",
                    (periods == period) & (volatility == wanted_volatility),
                ),
                ("relaxed_state_bands", periods == period),
            )
            chosen: tuple[int, float, float, str] | None = None
            for tier, tier_mask in tier_masks:
                candidates = np.flatnonzero(eligible & tier_mask & ~blocked)
                if len(candidates) == 0:
                    continue
                tries = min(RANDOM_MAX_TRIES_PER_TIER, len(candidates))
                positions = rng.choice(candidates, size=tries, replace=False)
                for position in positions:
                    position = int(position)
                    centre = pre_close[position] + sign * distance_atr * atr[position]
                    half_width = max(half_width_atr * atr[position], abs(centre) * 0.0005)
                    if (
                        high[position] >= centre - half_width
                        and low[position] <= centre + half_width
                    ):
                        chosen = (position, centre, half_width, tier)
                        break
                if chosen is not None:
                    break
            if chosen is None:
                continue
            position, centre, half_width, tier = chosen
            overlaps_real_cluster = overlaps_any_zone(
                centre - half_width,
                centre + half_width,
                actual_zones_by_row.get(position, ()),
            )
            left = max(0, position - COOLDOWN_BARS)
            right = min(len(blocked), position + COOLDOWN_BARS + 1)
            blocked[left:right] = True
            record = {
                column: actual[column]
                for column in actual_events.columns
                if column
                in {
                    "representation_mode",
                    "cluster_scale",
                    "cluster_class",
                    "component_count",
                    "distinct_source_count",
                    "family_count",
                    "mechanism_count",
                    "dependency_group_count",
                    "timeframe_count",
                    "component_count_bucket",
                    "component_families",
                    "component_mechanisms",
                    "component_dependency_groups",
                    "component_timeframes",
                    "contains_activity_transit",
                    "contains_acceptance_stickiness",
                    "interpretation_conflict",
                    "interpretation_class",
                    "same_family_duplicate_reference",
                    "same_mechanism_reference",
                    "independent_mechanism_cluster",
                    "same_dependency_group_reference",
                    "independent_dependency_group_cluster",
                    "ablation_survives_every_family_removal",
                    "indispensable_family_count",
                    "min_cluster_components_after_family_removal",
                    "max_cluster_components_after_family_removal",
                    "ablation_survives_every_mechanism_removal",
                    "indispensable_mechanism_count",
                    "min_cluster_components_after_mechanism_removal",
                    "contacted_component_count",
                    "contacted_family_count",
                    "contacted_mechanism_count",
                    "contacted_dependency_group_count",
                    "contacted_timeframe_count",
                    "contacted_component_count_bucket",
                    "contact_structure_class",
                    "all_cluster_components_contacted",
                    "multiple_components_contacted",
                    "contacted_independent_mechanisms",
                    "contacted_independent_dependency_groups",
                    "contacted_interpretation_class",
                    "contacted_interpretation_conflict",
                    "nested_relationship",
                    "nested_scale_count",
                    "active_level_count",
                    "local_level_density_2atr",
                    "density_band",
                    "cluster_width_atr",
                    "component_dispersion_atr",
                }
            }
            record.update(
                {
                    "row_index": position,
                    "center": centre,
                    "lower": centre - half_width,
                    "upper": centre + half_width,
                    "source_cluster_center": np.nan,
                    "component_keys": None,
                    "component_source_keys": None,
                    "cluster_signature": f"random_{stable_hash_int(f'{group_key}|{position}')}",
                    "control": "matched_random_cluster",
                    "source_available_at": None,
                    "random_match_tier": tier,
                    "control_zone_overlaps_real_cluster": overlaps_real_cluster,
                    "control_pair_id": actual["control_pair_id"],
                    "matched_actual_event_time": actual["event_time"],
                    "active_level_count": int(active_count_state[position]),
                    "local_level_density_2atr": int(density_state[position]),
                    "density_band": density_band(int(density_state[position])),
                }
            )
            selected.append(record)
    return selected


def overlaps_any_zone(
    lower: float, upper: float, zones: Sequence[tuple[float, float]]
) -> bool:
    return any(zone_lower <= upper and lower <= zone_upper for zone_lower, zone_upper in zones)


def summarize_cluster_events(events: DataFrame, *, horizons: Sequence[int]) -> DataFrame:
    working_frames = [events]
    pooled_period = events.copy()
    pooled_period["period"] = "all_periods_descriptive"
    working_frames.append(pooled_period)
    working = pd.concat(working_frames, ignore_index=True)
    pooled_approach = working.copy()
    pooled_approach["approach_state"] = "all_approaches"
    working = pd.concat([working, pooled_approach], ignore_index=True).copy()
    group_columns = [
        "pair",
        "control",
        "representation_mode",
        "cluster_scale",
        "cluster_class",
        "interpretation_class",
        "contact_structure_class",
        "contacted_interpretation_class",
        "contacted_component_count_bucket",
        "component_count_bucket",
        "family_count",
        "timeframe_count",
        "density_band",
        "cluster_width_band",
        "nested_relationship",
        "period",
        "approach_state",
    ]
    working["_event_day"] = normalize_dates(working["event_time"]).dt.floor("1D")
    grouped = working.groupby(group_columns, observed=True, dropna=False, sort=False)
    counts = grouped.size().rename("event_count")
    unique_days = grouped["_event_day"].nunique().rename("unique_event_days")
    structural = grouped[
        [
            "cluster_width_atr",
            "component_dispersion_atr",
            "component_count",
            "distinct_source_count",
            "local_level_density_2atr",
            "nested_scale_count",
            "min_cluster_components_after_family_removal",
            "max_cluster_components_after_family_removal",
            "contacted_component_count",
            "contacted_family_count",
            "contacted_timeframe_count",
        ]
    ].median()
    structural.rename(
        columns={column: f"{column}_median" for column in structural.columns}, inplace=True
    )
    overlap = grouped["control_zone_overlaps_real_cluster"].mean().rename(
        "control_zone_overlaps_real_cluster_fraction"
    )
    frames: list[DataFrame] = []
    for horizon in horizons:
        columns = {
            "abs_excursion": f"abs_excursion_atr_h{horizon}",
            "away_excursion": f"away_excursion_atr_h{horizon}",
            "through_excursion": f"through_excursion_atr_h{horizon}",
            "range_ratio": f"range_ratio_h{horizon}",
            "volume_ratio": f"volume_ratio_h{horizon}",
            "pressure_change_magnitude": f"pressure_change_h{horizon}",
            "dwell_fraction": f"dwell_fraction_h{horizon}",
            "crossings": f"crossings_h{horizon}",
        }
        metric_working = working[[*group_columns, *columns.values()]].copy()
        metric_working[f"pressure_change_h{horizon}"] = metric_working[
            f"pressure_change_h{horizon}"
        ].abs()
        metric_grouped = metric_working.groupby(
            group_columns, observed=True, dropna=False, sort=False
        )
        mean = metric_grouped.mean().rename(
            columns={column: f"{name}_mean" for name, column in columns.items()}
        )
        median = metric_grouped.median().rename(
            columns={column: f"{name}_median" for name, column in columns.items()}
        )
        frame = pd.concat(
            [counts, unique_days, structural, overlap, mean, median], axis=1
        ).reset_index()
        frame["horizon_hours"] = int(horizon)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def combine_pair_summaries(
    *, manifest: dict[str, Any], bulky_dir: Path, pairs: Sequence[str]
) -> tuple[DataFrame, DataFrame]:
    summary_dir = bulky_dir / "pair_summaries"
    frames = [pd.read_parquet(summary_dir / f"{pair_file_stem(pair)}.parquet") for pair in pairs]
    summary = pd.concat(frames, ignore_index=True)
    cohort_map = reporting_cohort_map(manifest)
    summary["coin_cohort"] = summary["pair"].map(cohort_map).fillna("unassigned")
    comparison_keys = [
        "pair",
        "coin_cohort",
        "representation_mode",
        "cluster_scale",
        "cluster_class",
        "interpretation_class",
        "component_count_bucket",
        "family_count",
        "timeframe_count",
        "density_band",
        "cluster_width_band",
        "nested_relationship",
        "period",
        "approach_state",
        "horizon_hours",
    ]
    metric_columns = [
        column
        for column in summary.columns
        if column.endswith(("_mean", "_median"))
        and column
        not in {
            "cluster_width_atr_median",
            "component_dispersion_atr_median",
            "component_count_median",
            "distinct_source_count_median",
            "local_level_density_2atr_median",
            "nested_scale_count_median",
        }
    ]
    actual = summary.loc[summary["control"].eq("actual_cluster")].copy()
    controls = summary.loc[summary["control"].ne("actual_cluster")].copy()
    comparison_frames: list[DataFrame] = []
    for control_name, control in controls.groupby("control", observed=True):
        joined = actual.merge(
            control,
            on=comparison_keys,
            how="inner",
            suffixes=("_actual", "_control"),
        )
        if joined.empty:
            continue
        output = joined[comparison_keys].copy()
        output["control"] = control_name
        output["actual_event_count"] = joined["event_count_actual"]
        output["control_event_count"] = joined["event_count_control"]
        for column in metric_columns:
            output[f"delta_{column}"] = joined[f"{column}_actual"] - joined[f"{column}_control"]
        comparison_frames.append(output)
    comparison = (
        pd.concat(comparison_frames, ignore_index=True) if comparison_frames else DataFrame()
    )
    return summary, comparison


def reaction_metric_specs(horizons: Sequence[int]) -> list[dict[str, Any]]:
    specs = [
        metric_spec("contact_range_ratio", "contact_activity", 0),
        metric_spec("contact_volume_ratio", "contact_activity", 0),
        metric_spec(
            "contact_pressure_change",
            "contact_pressure_change_magnitude",
            0,
            absolute=True,
        ),
    ]
    for column in (
        "time_to_abs_0_5atr",
        "time_to_abs_1_0atr",
        "time_to_away_0_5atr",
        "time_to_away_1_0atr",
        "time_to_through_0_5atr",
        "time_to_through_1_0atr",
    ):
        specs.append(
            metric_spec(
                column,
                "reaction_timing",
                max(horizons),
                lower_is_stronger=True,
                missing_fill=float(max(horizons) + 1),
            )
        )
    for horizon in horizons:
        for name, family, absolute in (
            ("abs_excursion_atr", "absolute_price_excursion", False),
            ("away_excursion_atr", "approach_relative_excursion", False),
            ("through_excursion_atr", "approach_relative_excursion", False),
            ("close_abs_displacement_atr", "absolute_price_displacement", False),
            ("range_ratio", "range_activity", False),
            ("volume_ratio", "volume_activity", False),
            ("pressure_change", "pressure_change_magnitude", True),
            ("dwell_fraction", "acceptance_and_dwell", False),
            ("crossings", "crossing_activity", False),
        ):
            specs.append(
                metric_spec(
                    f"{name}_h{horizon}", family, horizon, absolute=absolute
                )
            )
    return specs


def metric_spec(
    column: str,
    family: str,
    horizon: int,
    *,
    absolute: bool = False,
    lower_is_stronger: bool = False,
    missing_fill: float | None = None,
) -> dict[str, Any]:
    return {
        "metric": column,
        "metric_family": family,
        "horizon_hours": horizon,
        "absolute": absolute,
        "orientation": -1.0 if lower_is_stronger else 1.0,
        "missing_fill": missing_fill,
    }


def review_lenses() -> dict[str, str | None]:
    return {
        "primary_cluster_class": None,
        "direct_contact_structure": "contact_structure_class",
        "nearby_interpretation": "interpretation_class",
        "direct_contact_interpretation": "contacted_interpretation_class",
        "component_count": "component_count_bucket",
        "directly_contacted_component_count": "contacted_component_count_bucket",
        "level_density": "density_band",
        "cluster_width": "cluster_width_band",
        "nested_zone_relationship": "nested_relationship",
        "family_ablation": "ablation_survives_every_family_removal",
        "mechanism_independence": "independent_mechanism_cluster",
        "mechanism_ablation": "ablation_survives_every_mechanism_removal",
        "source_update_age": "oldest_source_update_age_band",
    }


def build_cluster_review(
    *,
    manifest: dict[str, Any],
    run_id: str,
    pairs: Sequence[str],
    compact_dir: Path,
    bulky_dir: Path,
    technical_smoke: bool,
    required_scales: Sequence[str],
    required_modes: Sequence[str],
) -> dict[str, Any]:
    horizons = tuple(int(value) for value in manifest["reaction_definition"]["horizons_hours"])
    specs = reaction_metric_specs(horizons)
    cohort_map = reporting_cohort_map(manifest)
    paired_cells: list[DataFrame] = []
    shifted_cells: list[DataFrame] = []
    integrity_rows: list[dict[str, Any]] = []
    pair_event_dir = bulky_dir / "pair_events"
    for pair in pairs:
        events = pd.read_parquet(pair_event_dir / f"{pair_file_stem(pair)}.parquet")
        events["event_time"] = normalize_dates(events["event_time"])
        events["coin_cohort"] = cohort_map[pair]
        actual = events.loc[events["control"].eq("actual_cluster")].copy()
        random = events.loc[events["control"].eq("matched_random_cluster")].copy()
        shifted = events.loc[events["control"].eq("price_shift_cluster")].copy()
        integrity_rows.append(cluster_integrity_row(pair, actual, random, shifted))
        paired_cells.extend(
            aggregate_paired_random_cells(actual=actual, random=random, specs=specs)
        )
        shifted_cells.extend(
            aggregate_shifted_cells(actual=actual, shifted=shifted, specs=specs)
        )

    paired = pd.concat(paired_cells, ignore_index=True) if paired_cells else DataFrame()
    shifted = pd.concat(shifted_cells, ignore_index=True) if shifted_cells else DataFrame()
    review_dir = bulky_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    paired_path = review_dir / "g0d_paired_random_cells.parquet"
    shifted_path = review_dir / "g0d_shifted_control_cells.parquet"
    atomic_write_parquet(paired, paired_path)
    atomic_write_parquet(shifted, shifted_path)
    control_cells = pd.concat([paired, shifted], ignore_index=True)
    repeatability = cluster_repeatability_screen(control_cells, manifest=manifest)
    repeatability_path = compact_dir / "g0d_repeatability_screen.parquet"
    atomic_write_parquet(repeatability, repeatability_path)
    integrity = cluster_integrity_record(
        rows=integrity_rows,
        run_id=run_id,
        technical_smoke=technical_smoke,
        required_scales=required_scales,
        required_modes=required_modes,
    )
    integrity_path = compact_dir / "g0d_integrity.json"
    atomic_write_json(integrity, integrity_path)
    return {
        "created_at_utc": utc_now(),
        "paired_random_cells": str(paired_path),
        "paired_random_cell_rows": len(paired),
        "shifted_control_cells": str(shifted_path),
        "shifted_control_cell_rows": len(shifted),
        "repeatability_screen": str(repeatability_path),
        "repeatability_rows": len(repeatability),
        "integrity": str(integrity_path),
        "integrity_passed": integrity["passed"],
        "minimum_cell_events_for_repeatability": 10,
        "timing_missing_value_rule": (
            f"A threshold not reached within {max(horizons)} hours is represented as "
            f"{max(horizons) + 1} hours in timing-control comparisons."
        ),
        "raw_event_rows_not_duplicated": True,
    }


def cluster_integrity_row(
    pair: str, actual: DataFrame, random: DataFrame, shifted: DataFrame
) -> dict[str, Any]:
    future_violations = int(
        (
            actual["source_available_at"].notna()
            & (actual["source_available_at"] > actual["event_time"])
        ).sum()
    )
    actual_ids = set(actual["control_pair_id"].dropna().astype(str))
    random_ids = set(random["control_pair_id"].dropna().astype(str))
    return {
        "pair": pair,
        "analysis_rows": int(len(actual) + len(random) + len(shifted)),
        "actual_events": len(actual),
        "matched_random_events": len(random),
        "price_shift_events": len(shifted),
        "matched_random_fraction": len(random) / len(actual) if len(actual) else 0.0,
        "exact_random_match_fraction": float(
            random["random_match_tier"]
            .eq("exact_period_volatility_range_density")
            .mean()
        )
        if len(random)
        else 0.0,
        "random_zone_real_cluster_overlap_fraction": float(
            random["control_zone_overlaps_real_cluster"].mean()
        )
        if len(random)
        else 0.0,
        "shifted_zone_real_cluster_overlap_fraction": float(
            shifted["control_zone_overlaps_real_cluster"].mean()
        )
        if len(shifted)
        else 0.0,
        "future_availability_violations": future_violations,
        "actual_event_id_duplicates": int(actual["control_pair_id"].duplicated().sum()),
        "random_control_pair_id_duplicates": int(random["control_pair_id"].duplicated().sum()),
        "random_ids_outside_actual": len(random_ids.difference(actual_ids)),
        "cluster_classes": sorted(actual["cluster_class"].dropna().unique().tolist()),
        "cluster_scales": sorted(actual["cluster_scale"].dropna().unique().tolist()),
        "representation_modes": sorted(
            actual["representation_mode"].dropna().unique().tolist()
        ),
        "multiple_component_contact_fraction": float(
            actual["multiple_components_contacted"].mean()
        )
        if len(actual)
        else 0.0,
    }


def cluster_integrity_record(
    *,
    rows: Sequence[dict[str, Any]],
    run_id: str,
    technical_smoke: bool,
    required_scales: Sequence[str] = tuple(CLUSTER_SCALES),
    required_modes: Sequence[str] = REPRESENTATION_MODES,
) -> dict[str, Any]:
    required_classes = {
        "same_family_same_timeframe",
        "different_family_same_timeframe",
        "same_family_cross_timeframe",
        "different_family_cross_timeframe",
    }
    required_scale_set = set(required_scales)
    required_mode_set = set(required_modes)
    failures: list[str] = []
    for row in rows:
        if row["future_availability_violations"]:
            failures.append(f"{row['pair']}: future availability violations")
        if row["actual_event_id_duplicates"] or row["random_control_pair_id_duplicates"]:
            failures.append(f"{row['pair']}: duplicate control-pair identifiers")
        if row["random_ids_outside_actual"]:
            failures.append(f"{row['pair']}: unmatched random control identifiers")
        if row["matched_random_fraction"] < 0.75:
            failures.append(f"{row['pair']}: random match coverage below 75%")
        if not technical_smoke:
            if set(row["cluster_classes"]) != required_classes:
                failures.append(f"{row['pair']}: incomplete cluster classes")
            if set(row["cluster_scales"]) != required_scale_set:
                failures.append(f"{row['pair']}: incomplete cluster scales")
            if set(row["representation_modes"]) != required_mode_set:
                failures.append(f"{row['pair']}: incomplete representation modes")
    return {
        "created_at_utc": utc_now(),
        "run_id": run_id,
        "technical_smoke_not_evidence": technical_smoke,
        "required_cluster_scales": sorted(required_scale_set),
        "required_representation_modes": sorted(required_mode_set),
        "pairs": list(rows),
        "failures": failures,
        "passed": not failures,
        "direction_prediction": False,
        "profit_optimization": False,
    }


def aggregate_paired_random_cells(
    *, actual: DataFrame, random: DataFrame, specs: Sequence[dict[str, Any]]
) -> list[DataFrame]:
    if actual.empty or random.empty:
        return []
    metadata = review_metadata_columns(actual)
    metric_columns = [str(spec["metric"]) for spec in specs]
    left = actual[["control_pair_id", *metadata, *metric_columns]].copy()
    right = random[["control_pair_id", *metric_columns]].copy()
    paired = left.merge(
        right,
        on="control_pair_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_actual", "_control"),
    )
    metric_working = prepare_metric_working(
        paired,
        specs=specs,
        actual_suffix="_actual",
        control_suffix="_control",
    )
    outputs: list[DataFrame] = []
    for lens_name, lens_column in review_lenses().items():
        lens_values = (
            pd.Series("all_clusters", index=paired.index, dtype="string")
            if lens_column is None
            else paired[lens_column].astype("string")
        )
        working = pd.concat(
            [
                metric_working,
                DataFrame(
                    {
                        "lens": pd.Series(lens_name, index=paired.index, dtype="string"),
                        "lens_value": lens_values,
                    }
                ),
            ],
            axis=1,
        )
        keys = review_cell_keys(include_state_bands=False)
        outputs.append(
            aggregate_metric_working(
                working,
                keys=keys,
                specs=specs,
                control="matched_random_cluster",
            )
        )
    return outputs


def aggregate_shifted_cells(
    *, actual: DataFrame, shifted: DataFrame, specs: Sequence[dict[str, Any]]
) -> list[DataFrame]:
    if actual.empty or shifted.empty:
        return []
    metadata = review_metadata_columns(actual)
    metric_columns = [str(spec["metric"]) for spec in specs]
    outputs: list[DataFrame] = []
    for lens_name, lens_column in review_lenses().items():
        control_frames: list[DataFrame] = []
        for name, frame in (("actual", actual), ("control", shifted)):
            metric_working = prepare_single_metric_working(
                frame[[*metadata, *metric_columns]], specs=specs
            )
            output_index = pd.RangeIndex(len(frame))
            lens_values = (
                pd.Series("all_clusters", index=output_index, dtype="string")
                if lens_column is None
                else frame[lens_column].astype("string").reset_index(drop=True)
            )
            working = pd.concat(
                [
                    metric_working,
                    DataFrame(
                        {
                            "lens": pd.Series(lens_name, index=output_index, dtype="string"),
                            "lens_value": lens_values,
                            "_control_role": pd.Series(
                                name, index=output_index, dtype="string"
                            ),
                        }
                    ),
                ],
                axis=1,
            )
            control_frames.append(working)
        combined = pd.concat(control_frames, ignore_index=True)
        keys = review_cell_keys(include_state_bands=True)
        grouped_frames: list[DataFrame] = []
        for role, frame in combined.groupby("_control_role", observed=True, sort=False):
            complete = frame_with_complete_keys(frame, keys)
            grouped_object = complete.groupby(
                keys, observed=True, dropna=False, sort=False
            )
            counts = grouped_object.size().rename("event_count")
            value_columns = [f"value__{spec['metric']}" for spec in specs]
            means = grouped_object[value_columns].mean()
            metric_counts = grouped_object[value_columns].count().rename(
                columns={column: column.replace("value__", "count__") for column in value_columns}
            )
            grouped = pd.concat([counts, means, metric_counts], axis=1).reset_index()
            grouped["_control_role"] = role
            grouped_frames.append(grouped)
        role_groups = pd.concat(grouped_frames, ignore_index=True)
        actual_grouped = role_groups.loc[role_groups["_control_role"].eq("actual")].drop(
            columns="_control_role"
        )
        control_grouped = role_groups.loc[
            role_groups["_control_role"].eq("control")
        ].drop(columns="_control_role")
        joined = actual_grouped.merge(
            control_grouped,
            on=keys,
            how="inner",
            validate="one_to_one",
            suffixes=("_actual", "_control"),
        )
        outputs.append(
            unpaired_join_to_long(
                joined,
                keys=keys,
                specs=specs,
                control="price_shift_cluster",
            )
        )
    return outputs


def review_metadata_columns(frame: DataFrame) -> list[str]:
    columns = [
        "pair",
        "coin_cohort",
        "period",
        "approach_state",
        "volatility_band",
        "contact_range_band",
        "representation_mode",
        "cluster_scale",
        "cluster_class",
        "contact_structure_class",
        "interpretation_class",
        "contacted_interpretation_class",
        "component_count_bucket",
        "contacted_component_count_bucket",
        "density_band",
        "cluster_width_band",
        "nested_relationship",
        "ablation_survives_every_family_removal",
        "independent_mechanism_cluster",
        "ablation_survives_every_mechanism_removal",
        "oldest_source_update_age_band",
    ]
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Cluster review metadata columns are missing: {missing}")
    return columns


def review_cell_keys(*, include_state_bands: bool) -> list[str]:
    keys = [
        "pair",
        "coin_cohort",
        "period",
        "approach_state",
        "representation_mode",
        "cluster_scale",
        "cluster_class",
        "lens",
        "lens_value",
    ]
    if include_state_bands:
        keys[4:4] = ["volatility_band", "contact_range_band"]
    return keys


def prepare_metric_working(
    frame: DataFrame,
    *,
    specs: Sequence[dict[str, Any]],
    actual_suffix: str,
    control_suffix: str,
) -> DataFrame:
    columns: dict[str, Series] = {}
    for spec in specs:
        metric = str(spec["metric"])
        actual = transformed_metric(frame[f"{metric}{actual_suffix}"], spec)
        control = transformed_metric(frame[f"{metric}{control_suffix}"], spec)
        delta = actual - control
        columns[f"actual__{metric}"] = actual
        columns[f"control__{metric}"] = control
        columns[f"delta__{metric}"] = delta
        columns[f"oriented__{metric}"] = delta * float(spec["orientation"])
    return pd.concat(
        [frame.reset_index(drop=True), DataFrame(columns).reset_index(drop=True)], axis=1
    ).copy()


def prepare_single_metric_working(
    frame: DataFrame, *, specs: Sequence[dict[str, Any]]
) -> DataFrame:
    columns = {
        f"value__{spec['metric']}": transformed_metric(frame[str(spec["metric"])], spec)
        for spec in specs
    }
    return pd.concat(
        [frame.reset_index(drop=True), DataFrame(columns).reset_index(drop=True)], axis=1
    ).copy()


def transformed_metric(series: Series, spec: dict[str, Any]) -> Series:
    values = pd.to_numeric(series, errors="coerce")
    if spec["absolute"]:
        values = values.abs()
    if spec["missing_fill"] is not None:
        values = values.fillna(float(spec["missing_fill"]))
    return values


def frame_with_complete_keys(frame: DataFrame, keys: Sequence[str]) -> DataFrame:
    working = frame.copy()
    for key in keys:
        if working[key].dtype == object or isinstance(working[key].dtype, pd.StringDtype):
            working[key] = working[key].fillna("missing")
    return working


def aggregate_metric_working(
    working: DataFrame,
    *,
    keys: Sequence[str],
    specs: Sequence[dict[str, Any]],
    control: str,
) -> DataFrame:
    complete = frame_with_complete_keys(working, keys)
    grouped_object = complete.groupby(list(keys), observed=True, dropna=False, sort=False)
    counts = grouped_object.size().rename("actual_events")
    metric_names = [str(spec["metric"]) for spec in specs]
    actual_columns = [f"actual__{metric}" for metric in metric_names]
    control_columns = [f"control__{metric}" for metric in metric_names]
    delta_columns = [f"delta__{metric}" for metric in metric_names]
    oriented_columns = [f"oriented__{metric}" for metric in metric_names]
    means = grouped_object[
        [*actual_columns, *control_columns, *delta_columns, *oriented_columns]
    ].mean()
    means.rename(
        columns={
            **{column: column.replace("actual__", "actual_mean__") for column in actual_columns},
            **{
                column: column.replace("control__", "control_mean__")
                for column in control_columns
            },
            **{column: column.replace("delta__", "delta_mean__") for column in delta_columns},
            **{
                column: column.replace("oriented__", "oriented_delta_mean__")
                for column in oriented_columns
            },
        },
        inplace=True,
    )
    medians = grouped_object[delta_columns].median()
    medians.rename(
        columns={column: column.replace("delta__", "delta_median__") for column in delta_columns},
        inplace=True,
    )
    metric_counts = grouped_object[delta_columns].count()
    metric_counts.rename(
        columns={
            column: column.replace("delta__", "scored_events__")
            for column in delta_columns
        },
        inplace=True,
    )
    positive_working = pd.concat(
        [
            complete[list(keys)].reset_index(drop=True),
            DataFrame(
                {
                    column: complete[column]
                    .gt(0.0)
                    .where(complete[column].notna())
                    .reset_index(drop=True)
                    for column in oriented_columns
                }
            ),
        ],
        axis=1,
    )
    positives = positive_working.groupby(
        list(keys), observed=True, dropna=False, sort=False
    )[oriented_columns].mean()
    positives.rename(
        columns={
            column: column.replace("oriented__", "oriented_positive_fraction__")
            for column in oriented_columns
        },
        inplace=True,
    )
    grouped = pd.concat(
        [counts, metric_counts, means, medians, positives], axis=1
    ).reset_index()
    grouped["control_events"] = grouped["actual_events"]
    return paired_group_to_long(grouped, keys=keys, specs=specs, control=control)


def paired_group_to_long(
    grouped: DataFrame,
    *,
    keys: Sequence[str],
    specs: Sequence[dict[str, Any]],
    control: str,
) -> DataFrame:
    outputs: list[DataFrame] = []
    for spec in specs:
        metric = str(spec["metric"])
        frame = grouped[[*keys, "actual_events", "control_events"]].copy()
        frame["actual_scored_events"] = grouped[f"scored_events__{metric}"]
        frame["control_scored_events"] = grouped[f"scored_events__{metric}"]
        frame["control"] = control
        frame["metric"] = metric
        frame["metric_family"] = spec["metric_family"]
        frame["horizon_hours"] = int(spec["horizon_hours"])
        frame["actual_mean"] = grouped[f"actual_mean__{metric}"]
        frame["control_mean"] = grouped[f"control_mean__{metric}"]
        frame["delta_mean"] = grouped[f"delta_mean__{metric}"]
        frame["delta_median"] = grouped[f"delta_median__{metric}"]
        frame["oriented_delta_mean"] = grouped[f"oriented_delta_mean__{metric}"]
        frame["oriented_positive_fraction"] = grouped[
            f"oriented_positive_fraction__{metric}"
        ]
        outputs.append(frame)
    return pd.concat(outputs, ignore_index=True)


def unpaired_join_to_long(
    joined: DataFrame,
    *,
    keys: Sequence[str],
    specs: Sequence[dict[str, Any]],
    control: str,
) -> DataFrame:
    outputs: list[DataFrame] = []
    for spec in specs:
        metric = str(spec["metric"])
        frame = joined[list(keys)].copy()
        frame["actual_events"] = joined["event_count_actual"]
        frame["control_events"] = joined["event_count_control"]
        frame["actual_scored_events"] = joined[f"count__{metric}_actual"]
        frame["control_scored_events"] = joined[f"count__{metric}_control"]
        frame["control"] = control
        frame["metric"] = metric
        frame["metric_family"] = spec["metric_family"]
        frame["horizon_hours"] = int(spec["horizon_hours"])
        frame["actual_mean"] = joined[f"value__{metric}_actual"]
        frame["control_mean"] = joined[f"value__{metric}_control"]
        frame["delta_mean"] = frame["actual_mean"] - frame["control_mean"]
        frame["delta_median"] = np.nan
        frame["oriented_delta_mean"] = frame["delta_mean"] * float(spec["orientation"])
        frame["oriented_positive_fraction"] = np.nan
        outputs.append(frame)
    return pd.concat(outputs, ignore_index=True)


def cluster_repeatability_screen(
    cells: DataFrame, *, manifest: dict[str, Any]
) -> DataFrame:
    if cells.empty:
        return DataFrame()
    repeat_periods = {
        str(period["id"])
        for period in manifest["data"]["chronological_periods"]
        if period["role"] != "diagnostic_only_not_confirmation"
    }
    eligible = cells.loc[
        cells["period"].isin(repeat_periods)
        & cells["actual_scored_events"].ge(10)
        & cells["control_scored_events"].ge(10)
    ].copy()
    scopes: list[DataFrame] = []
    primary = eligible.copy()
    primary["market_scope"] = "frozen_cohort:" + primary["coin_cohort"].astype(str)
    scopes.append(primary)
    all_pairs = eligible.copy()
    all_pairs["market_scope"] = "all_top10"
    scopes.append(all_pairs)
    non_btc = eligible.loc[~eligible["pair"].eq("BTC/USDT:USDT")].copy()
    non_btc["market_scope"] = "all_non_btc"
    scopes.append(non_btc)
    asset = eligible.copy()
    asset["market_scope"] = "asset:" + asset["pair"].astype(str)
    scopes.append(asset)
    working = pd.concat(scopes, ignore_index=True)
    keys = [
        "control",
        "market_scope",
        "representation_mode",
        "cluster_scale",
        "cluster_class",
        "lens",
        "lens_value",
        "metric",
        "metric_family",
        "horizon_hours",
    ]
    grouped = working.groupby(keys, observed=True, dropna=False, sort=False)
    result = grouped.agg(
        pair_period_cells=("pair", "size"),
        pair_count=("pair", "nunique"),
        period_count=("period", "nunique"),
        actual_events=("actual_events", "sum"),
        control_events=("control_events", "sum"),
        actual_scored_events=("actual_scored_events", "sum"),
        control_scored_events=("control_scored_events", "sum"),
        delta_mean_median=("delta_mean", "median"),
        delta_mean_average=("delta_mean", "mean"),
        oriented_delta_median=("oriented_delta_mean", "median"),
        oriented_positive_cell_fraction=(
            "oriented_delta_mean",
            positive_fraction,
        ),
    ).reset_index()
    weighted = working.copy()
    weighted["minimum_events"] = weighted[
        ["actual_scored_events", "control_scored_events"]
    ].min(axis=1)
    weighted.loc[weighted["oriented_delta_mean"].isna(), "minimum_events"] = 0
    weighted["weighted_oriented_delta"] = (
        weighted["oriented_delta_mean"] * weighted["minimum_events"]
    )
    weights = weighted.groupby(keys, observed=True, dropna=False, sort=False).agg(
        weighted_oriented_delta_sum=("weighted_oriented_delta", "sum"),
        weight_sum=("minimum_events", "sum"),
    ).reset_index()
    weights["oriented_delta_weighted_mean"] = (
        weights["weighted_oriented_delta_sum"] / weights["weight_sum"]
    )
    result = result.merge(
        weights[[*keys, "oriented_delta_weighted_mean"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    per_pair = working.groupby(
        [*keys, "pair"], observed=True, dropna=False, sort=False
    ).agg(
        pair_periods=("period", "nunique"),
        pair_oriented_delta=("oriented_delta_mean", "median"),
    ).reset_index()
    pair_support = per_pair.groupby(keys, observed=True, dropna=False, sort=False).agg(
        pairs_positive=(
            "pair_oriented_delta",
            lambda values: int((values.dropna() > 0.0).sum()),
        ),
        positive_pair_fraction=(
            "pair_oriented_delta",
            positive_fraction,
        ),
        pairs_with_two_or_more_periods=("pair_periods", lambda values: int((values >= 2).sum())),
    ).reset_index()
    result = result.merge(pair_support, on=keys, how="left", validate="one_to_one")
    result["broad_scope"] = result["market_scope"].isin({"all_top10", "all_non_btc"})
    result["direction_prediction"] = False
    result["profit_optimization"] = False
    return result


def positive_fraction(values: Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float((clean > 0.0).mean()) if len(clean) else np.nan


def pair_file_stem(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")


if __name__ == "__main__":
    raise SystemExit(main())

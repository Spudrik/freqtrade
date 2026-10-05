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
    ARTIFACT_ROOT as GENERATION0_CLUSTER_ARTIFACT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_cluster_generation0 import (  # noqa: E501
    REPORT_ROOT as GENERATION0_CLUSTER_REPORT_ROOT,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_cluster_generation0 import (  # noqa: E501
    ClusterComponent,
    build_component_matrix,
    contacted_component_metadata,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    DEFAULT_MANIFEST,
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


REPORT_ROOT = OUTPUT_ROOT / "generation1_cluster_increment"
ARTIFACT_ROOT = LARGE_ARTIFACT_ROOT / "generation1_cluster_increment"
FROZEN_BATCH = OUTPUT_ROOT / "generation0_review" / "g1_frozen_branch_batch.json"
DEFAULT_SOURCE_RUN_ID = "g0d_main_20260812_v1"
ALLOWED_SCALES = ("tight", "standard")
ALLOWED_REPRESENTATIONS = ("projected", "held_sensitivity")
OUTPUT_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PairTask:
    pair: str
    manifest_path: str
    source_run_id: str
    run_id: str
    scales: tuple[str, ...]
    representations: tuple[str, ...]
    max_events_per_representation: int | None
    overwrite: bool
    request_sha256: str


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 1C contact-identity and anchor-support preflight. It reconstructs "
            "the exact causally available components touched in Generation 0D cluster "
            "episodes and audits separate anchor-only episodes without predicting direction "
            "or optimizing profit."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--scales", default=",".join(ALLOWED_SCALES))
    parser.add_argument(
        "--representations", default=",".join(ALLOWED_REPRESENTATIONS)
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-events-per-representation", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    validate_worker_count(args.workers)
    if (
        args.max_events_per_representation is not None
        and args.max_events_per_representation < 100
    ):
        raise ValueError("--max-events-per-representation must be at least 100.")
    manifest = load_manifest(args.manifest)
    pairs = select_values(args.pairs, tuple(manifest["data"]["pairs"]), "pair")
    scales = select_values(args.scales, ALLOWED_SCALES, "cluster scale")
    representations = select_values(
        args.representations,
        ALLOWED_REPRESENTATIONS,
        "representation mode",
    )
    validate_frozen_branch()
    source_record = validate_source_run(
        source_run_id=args.source_run_id,
        manifest_path=args.manifest,
        pairs=pairs,
        scales=scales,
        representations=representations,
    )
    request = request_contract(
        manifest_path=args.manifest,
        source_run_id=args.source_run_id,
        source_record=source_record,
        pairs=pairs,
        scales=scales,
        representations=representations,
        max_events_per_representation=args.max_events_per_representation,
    )
    request_sha256 = stable_json_sha256(request)

    compact_dir = REPORT_ROOT / args.run_id
    bulky_dir = ARTIFACT_ROOT / args.run_id
    compact_dir.mkdir(parents=True, exist_ok=True)
    bulky_dir.mkdir(parents=True, exist_ok=True)
    record_path = compact_dir / "g1c_run_record.json"
    resumed_from_status: str | None = None
    if record_path.is_file() and not args.overwrite:
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        validate_existing_run_record(existing, request_sha256=request_sha256)
        resumed_from_status = str(existing.get("status", "unknown"))
    record = build_run_record(
        args=args,
        pairs=pairs,
        scales=scales,
        representations=representations,
        compact_dir=compact_dir,
        bulky_dir=bulky_dir,
        request=request,
        request_sha256=request_sha256,
        resumed_from_status=resumed_from_status,
    )
    atomic_write_json(record, record_path)

    tasks = [
        PairTask(
            pair=pair,
            manifest_path=str(args.manifest.resolve()),
            source_run_id=args.source_run_id,
            run_id=args.run_id,
            scales=scales,
            representations=representations,
            max_events_per_representation=args.max_events_per_representation,
            overwrite=args.overwrite,
            request_sha256=request_sha256,
        )
        for pair in pairs
    ]
    try:
        results = run_pair_tasks(tasks, workers=args.workers)
        inventory = DataFrame(results)
        atomic_write_parquet(inventory, compact_dir / "g1c_pair_inventory.parquet")
        failures = inventory.loc[inventory["status"].eq("failed")]
        if not failures.empty:
            raise RuntimeError(
                f"{len(failures)} G1C pair task(s) failed; inspect g1c_pair_inventory.parquet"
            )
        support = combine_pair_tables(
            run_id=args.run_id,
            pairs=pairs,
            table_name="anchor_support",
            request_sha256=request_sha256,
        )
        patterns = combine_pair_tables(
            run_id=args.run_id,
            pairs=pairs,
            table_name="pattern_inventory",
            request_sha256=request_sha256,
        )
        global_support = global_anchor_support(support)
        global_patterns = global_pattern_inventory(patterns)
        support_path = compact_dir / "g1c_anchor_support.parquet"
        pattern_path = compact_dir / "g1c_pattern_inventory.parquet"
        global_support_path = compact_dir / "g1c_global_anchor_support.parquet"
        global_pattern_path = compact_dir / "g1c_global_pattern_inventory.parquet"
        atomic_write_parquet(support, support_path)
        atomic_write_parquet(patterns, pattern_path)
        atomic_write_parquet(global_support, global_support_path)
        atomic_write_parquet(global_patterns, global_pattern_path)
        integrity = integrity_record(
            results=results,
            support=support,
            patterns=patterns,
            technical_smoke=args.max_events_per_representation is not None,
        )
        integrity_path = compact_dir / "g1c_integrity.json"
        atomic_write_json(integrity, integrity_path)
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "pair_tasks": results,
                "anchor_support": str(support_path),
                "pattern_inventory": str(pattern_path),
                "global_anchor_support": str(global_support_path),
                "global_pattern_inventory": str(global_pattern_path),
                "integrity": str(integrity_path),
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


def select_values(
    requested: str, allowed: tuple[str, ...], label: str
) -> tuple[str, ...]:
    if requested.strip().lower() == "all":
        return allowed
    selected = tuple(value.strip() for value in requested.split(",") if value.strip())
    unknown = sorted(set(selected).difference(allowed))
    if not selected or unknown:
        raise ValueError(f"Invalid {label} selection: selected={selected}, unknown={unknown}")
    return selected


def validate_frozen_branch() -> None:
    if not FROZEN_BATCH.is_file():
        raise FileNotFoundError(FROZEN_BATCH)
    frozen = json.loads(FROZEN_BATCH.read_text(encoding="utf-8"))
    branches = {branch["id"]: branch for branch in frozen["branches"]}
    branch = branches.get("g1c_same_episode_independent_cluster_increment")
    if branch is None or frozen.get("status") != "frozen_ready_for_execution":
        raise ValueError("The frozen Generation 1C branch is unavailable.")
    if frozen["common_scope"].get("direction_prediction") is not False:
        raise ValueError("Generation 1C must keep direction prediction disabled.")
    if frozen["common_scope"].get("profit_optimization") is not False:
        raise ValueError("Generation 1C must keep profit optimization disabled.")


def validate_source_run(
    *,
    source_run_id: str,
    manifest_path: Path,
    pairs: Sequence[str],
    scales: Sequence[str],
    representations: Sequence[str],
) -> dict[str, Any]:
    path = GENERATION0_CLUSTER_REPORT_ROOT / source_run_id / "g0d_cluster_run_record.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "completed":
        raise ValueError(f"Generation 0D source run is not complete: {path}")
    if record.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Generation 0D source run used a different frozen manifest.")
    if not set(pairs).issubset(record.get("pairs", [])):
        raise ValueError("Generation 0D source run does not contain every requested pair.")
    if not set(scales).issubset(record.get("cluster_scales", {})):
        raise ValueError("Generation 0D source run does not contain every requested scale.")
    if not set(representations).issubset(record.get("representation_modes", [])):
        raise ValueError(
            "Generation 0D source run does not contain every requested representation."
        )
    return record


def source_event_path(source_run_id: str, pair: str) -> Path:
    return (
        GENERATION0_CLUSTER_ARTIFACT_ROOT
        / source_run_id
        / "pair_events"
        / f"{pair_stem(pair)}.parquet"
    )


def request_contract(
    *,
    manifest_path: Path,
    source_run_id: str,
    source_record: dict[str, Any],
    pairs: Sequence[str],
    scales: Sequence[str],
    representations: Sequence[str],
    max_events_per_representation: int | None,
) -> dict[str, Any]:
    inputs = []
    for pair in pairs:
        path = source_event_path(source_run_id, pair)
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        inputs.append(
            {
                "pair": pair,
                "path": str(path.resolve()),
                "bytes": int(stat.st_size),
                "modified_ns": int(stat.st_mtime_ns),
                "sha256": sha256_file(path),
            }
        )
    return {
        "run_schema_version": RUN_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "cluster_helper_sha256": sha256_file(
            Path(__file__).with_name("market_reaction_zone_cluster_generation0.py")
        ),
        "manifest_sha256": sha256_file(manifest_path),
        "frozen_batch_sha256": sha256_file(FROZEN_BATCH),
        "source_run_id": source_run_id,
        "source_completed_at_utc": source_record.get("completed_at_utc"),
        "source_inputs": inputs,
        "pairs": list(pairs),
        "scales": list(scales),
        "representations": list(representations),
        "max_events_per_representation": max_events_per_representation,
        "direction_prediction": False,
        "profit_optimization": False,
    }


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_existing_run_record(
    record: dict[str, Any], *, request_sha256: str
) -> None:
    if record.get("request_sha256") != request_sha256:
        raise ValueError(
            "Run ID already exists with incompatible settings, code, or source artifacts. "
            "Use a new --run-id or explicitly pass --overwrite."
        )


def build_run_record(
    *,
    args: argparse.Namespace,
    pairs: Sequence[str],
    scales: Sequence[str],
    representations: Sequence[str],
    compact_dir: Path,
    bulky_dir: Path,
    request: dict[str, Any],
    request_sha256: str,
    resumed_from_status: str | None,
) -> dict[str, Any]:
    record = {
        "schema_version": RUN_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "status": "running",
        "started_at_utc": utc_now(),
        "orchestrator_pid": os.getpid(),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "frozen_branch": "g1c_same_episode_independent_cluster_increment",
        "source_run_id": args.source_run_id,
        "request_contract": request,
        "request_sha256": request_sha256,
        "pairs": list(pairs),
        "scales": list(scales),
        "representations": list(representations),
        "technical_smoke_not_evidence": (
            args.max_events_per_representation is not None
        ),
        "objective": (
            "Retain exact contacted component identities and audit whether independent "
            "multi-component episodes have separate same-component anchor-only support."
        ),
        "interpretation_boundary": (
            "This stage establishes representation and common support only. It does not "
            "claim a cluster effect, predict direction, or optimize profit."
        ),
        "same_future_path_rule": (
            "Counts use unique pair/base-candle future_path_key values so nested clusters, "
            "scales, representations, or multiple anchor lenses do not become independent "
            "market outcomes."
        ),
        "storage": {"compact": str(compact_dir), "bulky": str(bulky_dir)},
        "direction_prediction": False,
        "profit_optimization": False,
    }
    if resumed_from_status is not None:
        record["resumed_from_status"] = resumed_from_status
    return record


def run_pair_tasks(
    tasks: Sequence[PairTask], *, workers: int
) -> list[dict[str, Any]]:
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
    bulky_dir = ARTIFACT_ROOT / task.run_id / "pair_contacts"
    compact_support_dir = REPORT_ROOT / task.run_id / "pair_anchor_support"
    compact_pattern_dir = REPORT_ROOT / task.run_id / "pair_pattern_inventory"
    for path in (bulky_dir, compact_support_dir, compact_pattern_dir):
        path.mkdir(parents=True, exist_ok=True)
    stem = pair_stem(task.pair)
    contact_path = bulky_dir / f"{stem}.parquet"
    anchor_path = bulky_dir / f"{stem}.anchors.parquet"
    support_path = compact_support_dir / f"{stem}.parquet"
    pattern_path = compact_pattern_dir / f"{stem}.parquet"
    outputs = (contact_path, anchor_path, support_path, pattern_path)
    if all(path.is_file() for path in outputs) and not task.overwrite:
        validate_pair_outputs(outputs, pair=task.pair, request_sha256=task.request_sha256)
        contacts = pd.read_parquet(contact_path, columns=["future_path_key"])
        anchors = pd.read_parquet(anchor_path, columns=["future_path_key"])
        return {
            "pair": task.pair,
            "status": "existing",
            "contact_rows": len(contacts),
            "anchor_rows": len(anchors),
            "seconds": round(time.perf_counter() - started, 3),
        }

    manifest_path = Path(task.manifest_path)
    manifest = load_manifest(manifest_path)
    base = prepare_base_market_frame(task.pair, manifest)
    source = pd.read_parquet(
        source_event_path(task.source_run_id, task.pair),
        filters=[
            ("control", "==", "actual_cluster"),
            ("cluster_scale", "in", list(task.scales)),
            ("representation_mode", "in", list(task.representations)),
        ],
    )
    source = source.sort_values(
        ["representation_mode", "event_time", "cluster_scale", "level_price"]
    ).reset_index(drop=True)
    enriched_frames = []
    mismatched_counts = 0
    mismatched_structures = 0
    for representation in task.representations:
        selected = source.loc[source["representation_mode"].eq(representation)].copy()
        if selected.empty:
            continue
        selected = technical_smoke_selection(
            selected,
            task.max_events_per_representation,
        )
        values, components, _ = build_component_matrix(
            pair=task.pair,
            base=base,
            manifest=manifest,
            manifest_path=manifest_path,
            representation_mode=representation,
        )
        enriched, audit = reconstruct_exact_contacts(
            selected,
            base=base,
            values=values,
            components=components,
        )
        mismatched_counts += audit["mismatched_contact_counts"]
        mismatched_structures += audit["mismatched_contact_structures"]
        enriched_frames.append(enriched)
    if not enriched_frames:
        raise ValueError(f"No Generation 1C source contacts were available for {task.pair}.")
    contacts = pd.concat(enriched_frames, ignore_index=True)
    contacts["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    contacts["run_request_sha256"] = task.request_sha256
    anchors = explode_contacted_anchors(contacts)
    anchors["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    anchors["run_request_sha256"] = task.request_sha256
    support = anchor_support_inventory(anchors)
    patterns = recurring_pattern_inventory(contacts)
    for frame in (support, patterns):
        frame["output_schema_version"] = OUTPUT_SCHEMA_VERSION
        frame["run_request_sha256"] = task.request_sha256
    atomic_write_parquet(contacts, contact_path)
    atomic_write_parquet(anchors, anchor_path)
    atomic_write_parquet(support, support_path)
    atomic_write_parquet(patterns, pattern_path)
    return {
        "pair": task.pair,
        "status": "completed",
        "contact_rows": len(contacts),
        "unique_future_paths": int(contacts["future_path_key"].nunique()),
        "anchor_rows": len(anchors),
        "anchor_support_rows": len(support),
        "pattern_rows": len(patterns),
        "mismatched_contact_counts": mismatched_counts,
        "mismatched_contact_structures": mismatched_structures,
        "contact_path": str(contact_path),
        "anchor_path": str(anchor_path),
        "seconds": round(time.perf_counter() - started, 3),
    }


def technical_smoke_selection(frame: DataFrame, maximum: int | None) -> DataFrame:
    if maximum is None or len(frame) <= maximum:
        return frame
    positions = np.linspace(0, len(frame) - 1, num=maximum, dtype=np.int64)
    return frame.iloc[np.unique(positions)].copy()


def reconstruct_exact_contacts(
    events: DataFrame,
    *,
    base: DataFrame,
    values: np.ndarray,
    components: Sequence[ClusterComponent],
) -> tuple[DataFrame, dict[str, int]]:
    high = base["high"].to_numpy(dtype=float)
    low = base["low"].to_numpy(dtype=float)
    atr = base["base_atr"].to_numpy(dtype=float)
    pre_close = base["pre_close"].to_numpy(dtype=float)
    metadata_rows: list[dict[str, Any]] = []
    mismatched_counts = 0
    mismatched_structures = 0
    for row in events.itertuples(index=False):
        row_index = int(row.base_index)
        member_indexes = tuple(
            int(value) for value in str(row.member_index_signature).split(";") if value
        )
        metadata = contacted_component_metadata(
            cluster={"member_indexes_list": member_indexes},
            row_values=values[row_index],
            components=components,
            atr=float(atr[row_index]),
            candle_high=float(high[row_index]),
            candle_low=float(low[row_index]),
            scale=str(row.cluster_scale),
            price_shift=0.0,
        )
        mismatched_counts += int(
            int(row.contacted_component_count)
            != int(metadata["contacted_component_count"])
        )
        mismatched_structures += int(
            str(row.contact_structure_class) != str(metadata["contact_structure_class"])
        )
        metadata_rows.append(metadata)
    if mismatched_counts or mismatched_structures:
        raise ValueError(
            "Exact contact reconstruction disagreed with Generation 0D counts or "
            f"structures: count={mismatched_counts}, structure={mismatched_structures}"
        )
    output = events.copy().reset_index(drop=True)
    metadata_frame = DataFrame(metadata_rows)
    for column in metadata_frame:
        output[column] = metadata_frame[column]
    indexes = output["base_index"].to_numpy(dtype=np.int64)
    output["pre_close_at_event"] = pre_close[indexes]
    output["future_path_key"] = [
        f"{pair}|{index}"
        for pair, index in zip(output["pair"], output["base_index"], strict=True)
    ]
    output["cluster_contact_episode_key"] = [
        hashlib.sha256(
            "|".join(
                (
                    str(pair),
                    str(representation),
                    str(scale),
                    str(index),
                    str(signature),
                )
            ).encode()
        ).hexdigest()[:24]
        for pair, representation, scale, index, signature in zip(
            output["pair"],
            output["representation_mode"],
            output["cluster_scale"],
            output["base_index"],
            output["contacted_component_signature"],
            strict=True,
        )
    ]
    output["direction_prediction"] = False
    output["profit_optimization"] = False
    return output, {
        "mismatched_contact_counts": mismatched_counts,
        "mismatched_contact_structures": mismatched_structures,
    }


def split_aligned(value: Any, *, expected: int, label: str) -> list[str]:
    values = [] if value is None or str(value) == "" else str(value).split(";")
    if len(values) != expected:
        raise ValueError(
            f"Aligned contacted-component field {label} has {len(values)} values; "
            f"expected {expected}."
        )
    return values


def explode_contacted_anchors(events: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    aligned_fields = {
        "anchor_component_key": "contacted_component_keys",
        "anchor_component_index": "contacted_component_indexes",
        "anchor_level_price": "contacted_component_prices",
        "anchor_source_key": "contacted_component_source_keys_aligned",
        "anchor_family": "contacted_component_families_aligned",
        "anchor_mechanism": "contacted_component_mechanisms_aligned",
        "anchor_dependency_group": "contacted_component_dependency_groups_aligned",
        "anchor_timeframe": "contacted_component_timeframes_aligned",
        "anchor_interpretation": "contacted_component_interpretations_aligned",
    }
    event_columns = (
        "pair",
        "representation_mode",
        "cluster_scale",
        "base_index",
        "event_time",
        "period",
        "approach_state",
        "future_path_key",
        "cluster_contact_episode_key",
        "contacted_component_signature",
        "contacted_component_count",
        "contacted_mechanism_count",
        "contacted_dependency_group_count",
        "contacted_independent_mechanisms",
        "contacted_independent_dependency_groups",
        "cluster_class",
        "contact_structure_class",
        "cluster_width_atr",
        "local_level_density_2atr",
        "active_level_count",
        "pre_close_at_event",
        "base_atr",
    )
    for row in events.to_dict(orient="records"):
        count = int(row["contacted_component_count"])
        aligned = {
            output: split_aligned(row[source], expected=count, label=source)
            for output, source in aligned_fields.items()
        }
        for position in range(count):
            record = {column: row[column] for column in event_columns}
            for output, values in aligned.items():
                record[output] = values[position]
            record["anchor_component_index"] = int(record["anchor_component_index"])
            record["anchor_level_price"] = float(record["anchor_level_price"])
            atr = float(record["base_atr"])
            record["anchor_pre_distance_atr"] = (
                abs(
                    float(record["anchor_level_price"])
                    - float(record["pre_close_at_event"])
                )
                / atr
                if np.isfinite(atr) and atr > 0.0
                else np.nan
            )
            if count == 1:
                record["anchor_contact_kind"] = "single_component_contact"
            elif bool(row["contacted_independent_dependency_groups"]):
                record["anchor_contact_kind"] = "independent_multi_component_contact"
            else:
                record["anchor_contact_kind"] = "shared_multi_component_contact"
            rows.append(record)
    output = DataFrame(rows)
    if output.empty:
        raise ValueError("No contacted anchors were reconstructed.")
    output["direction_prediction"] = False
    output["profit_optimization"] = False
    return output


def anchor_support_inventory(anchors: DataFrame) -> DataFrame:
    keys = [
        "pair",
        "representation_mode",
        "cluster_scale",
        "anchor_component_key",
        "anchor_source_key",
        "anchor_family",
        "anchor_mechanism",
        "anchor_dependency_group",
        "anchor_timeframe",
        "anchor_interpretation",
    ]
    unique = anchors.drop_duplicates(keys + ["future_path_key", "anchor_contact_kind"])
    counts = (
        unique.groupby(keys + ["anchor_contact_kind"], observed=True, sort=False)
        .agg(
            future_path_episodes=("future_path_key", "nunique"),
            period_count=("period", "nunique"),
            first_event_time=("event_time", "min"),
            last_event_time=("event_time", "max"),
        )
        .reset_index()
    )
    value_columns = [
        "future_path_episodes",
        "period_count",
        "first_event_time",
        "last_event_time",
    ]
    wide = counts.pivot(index=keys, columns="anchor_contact_kind", values=value_columns)
    wide.columns = [f"{metric}__{kind}" for metric, kind in wide.columns]
    wide = wide.reset_index()
    for kind in (
        "single_component_contact",
        "independent_multi_component_contact",
        "shared_multi_component_contact",
    ):
        column = f"future_path_episodes__{kind}"
        if column not in wide:
            wide[column] = 0
        wide[column] = wide[column].fillna(0).astype(int)
    period_presence = (
        unique.assign(present=True)
        .pivot_table(
            index=keys + ["period"],
            columns="anchor_contact_kind",
            values="present",
            aggfunc="any",
            fill_value=False,
        )
        .reset_index()
    )
    for kind in (
        "single_component_contact",
        "independent_multi_component_contact",
    ):
        if kind not in period_presence:
            period_presence[kind] = False
    common = (
        period_presence.assign(
            common_anchor_period=(
                period_presence["single_component_contact"]
                & period_presence["independent_multi_component_contact"]
            )
        )
        .groupby(keys, observed=True, sort=False)["common_anchor_period"]
        .sum()
        .rename("common_single_and_independent_multi_period_count")
        .reset_index()
    )
    result = wide.merge(common, on=keys, how="left", validate="one_to_one")
    result["has_anchor_only_common_support"] = (
        result["future_path_episodes__single_component_contact"].gt(0)
        & result["future_path_episodes__independent_multi_component_contact"].gt(0)
        & result["common_single_and_independent_multi_period_count"].gt(0)
    )
    result["direction_prediction"] = False
    result["profit_optimization"] = False
    return result


def recurring_pattern_inventory(events: DataFrame) -> DataFrame:
    keys = [
        "pair",
        "representation_mode",
        "cluster_scale",
        "contacted_component_signature",
        "contacted_component_keys",
        "contacted_component_families",
        "contacted_component_mechanisms",
        "contacted_component_dependency_groups",
        "contacted_component_timeframes",
        "contacted_component_count",
        "contacted_mechanism_count",
        "contacted_dependency_group_count",
        "contacted_independent_mechanisms",
        "contacted_independent_dependency_groups",
        "contacted_interpretation_class",
    ]
    unique = events.drop_duplicates(keys + ["future_path_key"])
    result = (
        unique.groupby(keys, observed=True, sort=False)
        .agg(
            future_path_episodes=("future_path_key", "nunique"),
            period_count=("period", "nunique"),
            first_event_time=("event_time", "min"),
            last_event_time=("event_time", "max"),
        )
        .reset_index()
    )
    result["repeated_exact_pattern"] = result["future_path_episodes"].ge(2)
    result["direction_prediction"] = False
    result["profit_optimization"] = False
    return result


def validate_pair_outputs(
    paths: Sequence[Path], *, pair: str, request_sha256: str
) -> None:
    for path in paths:
        try:
            frame = pd.read_parquet(
                path,
                columns=[
                    "pair",
                    "output_schema_version",
                    "run_request_sha256",
                ],
            )
        except Exception as exc:
            raise ValueError(f"Existing G1C output has no valid contract: {path}") from exc
        if frame.empty or set(frame["pair"].astype(str)) != {pair}:
            raise ValueError(f"Existing G1C output has an incompatible pair: {path}")
        if set(frame["output_schema_version"].astype(int)) != {
            OUTPUT_SCHEMA_VERSION
        }:
            raise ValueError(f"Existing G1C output has an incompatible schema: {path}")
        if set(frame["run_request_sha256"].astype(str)) != {request_sha256}:
            raise ValueError(f"Existing G1C output belongs to another request: {path}")


def combine_pair_tables(
    *,
    run_id: str,
    pairs: Sequence[str],
    table_name: str,
    request_sha256: str,
) -> DataFrame:
    directory = REPORT_ROOT / run_id / f"pair_{table_name}"
    frames = []
    for pair in pairs:
        path = directory / f"{pair_stem(pair)}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        validate_pair_outputs((path,), pair=pair, request_sha256=request_sha256)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True) if frames else DataFrame()


def global_anchor_support(support: DataFrame) -> DataFrame:
    if support.empty:
        return DataFrame()
    keys = [
        "representation_mode",
        "cluster_scale",
        "anchor_component_key",
        "anchor_source_key",
        "anchor_family",
        "anchor_mechanism",
        "anchor_dependency_group",
        "anchor_timeframe",
        "anchor_interpretation",
    ]
    return (
        support.groupby(keys, observed=True, sort=False)
        .agg(
            pair_count=("pair", "nunique"),
            pairs_with_common_support=("has_anchor_only_common_support", "sum"),
            single_component_future_paths=(
                "future_path_episodes__single_component_contact",
                "sum",
            ),
            independent_multi_future_paths=(
                "future_path_episodes__independent_multi_component_contact",
                "sum",
            ),
        )
        .reset_index()
    )


def global_pattern_inventory(patterns: DataFrame) -> DataFrame:
    if patterns.empty:
        return DataFrame()
    keys = [
        "representation_mode",
        "cluster_scale",
        "contacted_component_signature",
        "contacted_component_keys",
        "contacted_component_families",
        "contacted_component_mechanisms",
        "contacted_component_dependency_groups",
        "contacted_component_timeframes",
        "contacted_component_count",
        "contacted_mechanism_count",
        "contacted_dependency_group_count",
        "contacted_independent_mechanisms",
        "contacted_independent_dependency_groups",
        "contacted_interpretation_class",
    ]
    return (
        patterns.groupby(keys, observed=True, sort=False)
        .agg(
            pair_count=("pair", "nunique"),
            future_path_episodes=("future_path_episodes", "sum"),
            pair_period_cells=("period_count", "sum"),
            first_event_time=("first_event_time", "min"),
            last_event_time=("last_event_time", "max"),
        )
        .reset_index()
    )


def integrity_record(
    *,
    results: Sequence[dict[str, Any]],
    support: DataFrame,
    patterns: DataFrame,
    technical_smoke: bool,
) -> dict[str, Any]:
    return {
        "created_at_utc": utc_now(),
        "technical_smoke_not_evidence": technical_smoke,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "pairs": len(results),
        "contact_rows": int(sum(row.get("contact_rows", 0) for row in results)),
        "unique_future_paths": int(
            sum(row.get("unique_future_paths", 0) for row in results)
        ),
        "anchor_rows": int(sum(row.get("anchor_rows", 0) for row in results)),
        "mismatched_contact_counts": int(
            sum(row.get("mismatched_contact_counts", 0) for row in results)
        ),
        "mismatched_contact_structures": int(
            sum(row.get("mismatched_contact_structures", 0) for row in results)
        ),
        "anchors_with_common_single_and_independent_multi_support": int(
            support["has_anchor_only_common_support"].sum()
        )
        if len(support)
        else 0,
        "repeated_exact_patterns": int(patterns["repeated_exact_pattern"].sum())
        if len(patterns)
        else 0,
        "same_future_paths_counted_as_independent_outcomes": False,
        "outcome_comparison_performed": False,
        "direction_prediction": False,
        "profit_optimization": False,
    }


def pair_stem(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")


if __name__ == "__main__":
    raise SystemExit(main())

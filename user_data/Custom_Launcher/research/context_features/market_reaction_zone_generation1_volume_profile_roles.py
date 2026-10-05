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

from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    DEFAULT_MANIFEST,
    EVENT_DIR,
    atomic_write_json,
    atomic_write_parquet,
    load_manifest,
    normalize_dates,
    numeric_array,
    prepare_base_market_frame,
    sha256_file,
    stable_hash_int,
    utc_now,
    validate_cache_metadata,
    validate_worker_count,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    REPORT_DIR as GENERATION0_REPORT_DIR,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_localization import (  # noqa: E501
    LOCAL_MATCH_FEATURES,
    balance_diagnostics,
    causal_local_state,
    causal_market_context,
    causal_source_state,
    nearest_state_pairs,
    outcome_columns,
    outcome_metadata,
    outcome_values,
    role_oriented_delta,
)


OUTPUT_ROOT = DEFAULT_MANIFEST.parent
LARGE_OUTPUT_ROOT = EVENT_DIR.parent
REPORT_ROOT = OUTPUT_ROOT / "generation1_volume_profile_roles"
ARTIFACT_ROOT = LARGE_OUTPUT_ROOT / "generation1_volume_profile_roles"
FROZEN_BATCH = OUTPUT_ROOT / "generation0_review" / "g1_frozen_branch_batch.json"
ZONE_METHODS = ("tight_base_atr", "standard_base_atr", "wide_base_atr")
OUTPUT_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1
MIN_CELL_EVENTS = 20

VP_EVENT_FAMILIES = (
    "volume_profile_nodes",
    "volume_profile_settled",
    "volume_profile_explicit_prior",
)
VP_LEVELS = ("lvn_above", "lvn_below", "hvn_above", "hvn_below", "poc", "prior_poc")
VP_STATE_FEATURES = (*LOCAL_MATCH_FEATURES[:-1],)


@dataclass(frozen=True)
class AttributeProfile:
    name: str
    level_names: tuple[str, ...]
    attribute_column: str
    role: str


ATTRIBUTE_PROFILES = (
    AttributeProfile(
        "lvn_thinness",
        ("lvn_above", "lvn_below"),
        "level_score",
        "activity_transit",
    ),
    AttributeProfile(
        "hvn_strength",
        ("hvn_above", "hvn_below"),
        "level_score",
        "acceptance_stickiness",
    ),
    AttributeProfile(
        "poc_profile_evidence",
        ("poc", "prior_poc"),
        "level_score",
        "acceptance_stickiness",
    ),
    AttributeProfile(
        "hvn_value_area_width",
        ("hvn_above", "hvn_below"),
        "attr_vp_value_area_width_pct",
        "acceptance_stickiness",
    ),
    AttributeProfile(
        "poc_value_area_width",
        ("poc", "prior_poc"),
        "attr_vp_value_area_width_pct",
        "acceptance_stickiness",
    ),
)


@dataclass(frozen=True)
class PairTask:
    pair: str
    manifest_path: str
    run_id: str
    timeframes: tuple[str, ...]
    zones: tuple[str, ...]
    max_events_per_timeframe: int | None
    overwrite: bool
    request_sha256: str


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 1B unchanged Volume Profile role and attribute test. It compares "
            "high versus low causal attribute values after continuous pre-contact state "
            "matching and against a deterministic shuffled-attribute placebo."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--timeframes", default="all")
    parser.add_argument("--zones", default=",".join(ZONE_METHODS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-events-per-timeframe", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    validate_worker_count(args.workers)
    if args.max_events_per_timeframe is not None and args.max_events_per_timeframe < 500:
        raise ValueError("--max-events-per-timeframe must be at least 500.")
    manifest = load_manifest(args.manifest)
    pairs = select_values(args.pairs, tuple(manifest["data"]["pairs"]), "pair")
    timeframes = select_values(
        args.timeframes,
        tuple(manifest["data"]["source_timeframes"]),
        "timeframe",
    )
    zones = select_values(args.zones, ZONE_METHODS, "zone")
    validate_frozen_branch()
    sources = source_contracts(
        pairs=pairs,
        timeframes=timeframes,
        manifest_path=args.manifest,
    )
    request = {
        "run_schema_version": RUN_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "frozen_batch_sha256": sha256_file(FROZEN_BATCH),
        "pairs": list(pairs),
        "timeframes": list(timeframes),
        "zones": list(zones),
        "attribute_profiles": [profile.__dict__ for profile in ATTRIBUTE_PROFILES],
        "state_features": list(VP_STATE_FEATURES),
        "source_contracts": sources,
        "max_events_per_timeframe": args.max_events_per_timeframe,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    request_sha256 = stable_json_sha256(request)
    compact_dir = REPORT_ROOT / args.run_id
    bulky_dir = ARTIFACT_ROOT / args.run_id
    compact_dir.mkdir(parents=True, exist_ok=True)
    bulky_dir.mkdir(parents=True, exist_ok=True)
    record_path = compact_dir / "g1b_run_record.json"
    if record_path.is_file() and not args.overwrite:
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        if existing.get("request_sha256") != request_sha256:
            raise ValueError("Run ID already exists with an incompatible request.")
    record = {
        "schema_version": RUN_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "status": "running",
        "started_at_utc": utc_now(),
        "orchestrator_pid": os.getpid(),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "frozen_branch": "g1b_existing_volume_profile_roles",
        "request_contract": request,
        "request_sha256": request_sha256,
        "pairs": list(pairs),
        "timeframes": list(timeframes),
        "zones": list(zones),
        "technical_smoke_not_evidence": args.max_events_per_timeframe is not None,
        "baseline": (
            "Generation 0B found a descriptive LVN transit versus HVN/POC acceptance "
            "split and small conditional Volume Profile attribute leads."
        ),
        "hypothesis": (
            "Higher unchanged LVN thinness orders more activity/transit, while stronger "
            "HVN/POC evidence and wider value areas order more acceptance, after causal "
            "continuous market state and level approach geometry are balanced."
        ),
        "controls": [
            "deterministic attribute shuffle inside identical exact strata",
            "high-versus-low attribute tertiles on the same level family",
            "continuous pre-contact local source BTC and top-ten state matching",
            "hard 0.10 ATR pre-contact level-distance gate",
            "non-overlapping 48-hour outcome windows",
            "separate G1A current-versus-stale-shuffled-shifted location controls",
        ],
        "attribute_tiers": (
            "Within pair, timeframe, level, zone, chronological period, and approach, "
            "rank the causal attribute without using outcomes; compare upper and lower "
            "thirds. The placebo permutes the attribute deterministically in the same stratum."
        ),
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(record, record_path)
    tasks = [
        PairTask(
            pair=pair,
            manifest_path=str(args.manifest.resolve()),
            run_id=args.run_id,
            timeframes=timeframes,
            zones=zones,
            max_events_per_timeframe=args.max_events_per_timeframe,
            overwrite=args.overwrite,
            request_sha256=request_sha256,
        )
        for pair in pairs
    ]
    try:
        results = run_pair_tasks(tasks, workers=args.workers)
        inventory = DataFrame(results)
        atomic_write_parquet(inventory, compact_dir / "g1b_pair_inventory.parquet")
        failures = inventory.loc[inventory["status"].eq("failed")]
        if not failures.empty:
            raise RuntimeError(
                f"{len(failures)} G1B pair task(s) failed; inspect the pair inventory."
            )
        cells = combine_pair_outputs(
            args.run_id,
            pairs,
            request_sha256=request_sha256,
        )
        screen = repeatability_screen(cells, manifest)
        screen_path = compact_dir / "g1b_repeatability_screen.parquet"
        atomic_write_parquet(screen, screen_path)
        integrity = integrity_record(
            results=results,
            cells=cells,
            technical_smoke=args.max_events_per_timeframe is not None,
        )
        integrity_path = compact_dir / "g1b_integrity.json"
        atomic_write_json(integrity, integrity_path)
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "pair_tasks": results,
                "cell_rows": len(cells),
                "repeatability_rows": len(screen),
                "repeatability_screen": str(screen_path),
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


def select_values(requested: str, allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    if requested.strip().lower() == "all":
        return allowed
    selected = tuple(value.strip() for value in requested.split(",") if value.strip())
    unknown = sorted(set(selected).difference(allowed))
    if not selected or unknown:
        raise ValueError(f"Invalid {label} selection: selected={selected}, unknown={unknown}")
    return selected


def validate_frozen_branch() -> None:
    frozen = json.loads(FROZEN_BATCH.read_text(encoding="utf-8"))
    branch_ids = {branch["id"] for branch in frozen["branches"]}
    if (
        frozen.get("status") != "frozen_ready_for_execution"
        or "g1b_existing_volume_profile_roles" not in branch_ids
    ):
        raise ValueError("The frozen Generation 1B branch is unavailable.")
    if frozen["common_scope"].get("direction_prediction") is not False:
        raise ValueError("Generation 1B must keep direction prediction disabled.")
    if frozen["common_scope"].get("profit_optimization") is not False:
        raise ValueError("Generation 1B must keep profit optimization disabled.")


def generation0_event_source(pair: str, timeframe: str) -> tuple[Path, Path]:
    stem = pair_stem(pair)
    candidates = sorted(EVENT_DIR.glob(f"{stem}-{timeframe}-core-generic-g0b1-g0b2-*.parquet"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one full Generation 0B event source for {pair} {timeframe}; "
            f"found {len(candidates)}."
        )
    event_path = candidates[0]
    metadata_path = GENERATION0_REPORT_DIR / f"{event_path.stem}.meta.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    return event_path, metadata_path


def source_contracts(
    *, pairs: Sequence[str], timeframes: Sequence[str], manifest_path: Path
) -> list[dict[str, Any]]:
    contracts = []
    manifest_sha = sha256_file(manifest_path)
    for pair in pairs:
        for timeframe in timeframes:
            event_path, metadata_path = generation0_event_source(pair, timeframe)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("manifest_sha256") != manifest_sha:
                raise ValueError(f"Generation 0B source used another manifest: {event_path}")
            if metadata.get("direction_prediction") is not False:
                raise ValueError(f"Direction boundary failed: {event_path}")
            if metadata.get("profit_optimization") is not False:
                raise ValueError(f"Profit boundary failed: {event_path}")
            cache_path = Path(str(metadata["cache"]))
            validate_cache_metadata(cache_path, manifest_path)
            if sha256_file(cache_path) != metadata.get("cache_sha256"):
                raise ValueError(f"Generation 0B event source cache changed: {event_path}")
            stat = event_path.stat()
            contracts.append(
                {
                    "pair": pair,
                    "timeframe": timeframe,
                    "event_path": str(event_path.resolve()),
                    "event_bytes": int(stat.st_size),
                    "event_modified_ns": int(stat.st_mtime_ns),
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


def pair_output_path(run_id: str, pair: str) -> Path:
    return ARTIFACT_ROOT / run_id / "pair_cells" / f"{pair_stem(pair)}.parquet"


def build_pair(task: PairTask) -> dict[str, Any]:
    started = time.perf_counter()
    output_path = pair_output_path(task.run_id, task.pair)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file() and not task.overwrite:
        existing = validate_pair_output(
            output_path,
            pair=task.pair,
            request_sha256=task.request_sha256,
        )
        return {
            "pair": task.pair,
            "status": "existing",
            "cell_rows": len(existing),
            "seconds": round(time.perf_counter() - started, 3),
        }
    manifest_path = Path(task.manifest_path)
    manifest = load_manifest(manifest_path)
    horizons = tuple(int(value) for value in manifest["reaction_definition"]["horizons_hours"])
    base = prepare_base_market_frame(task.pair, manifest)
    local_state = causal_local_state(base)
    local_state = local_state.merge(
        causal_market_context(manifest),
        on="date",
        how="left",
        validate="one_to_one",
    )
    cell_rows: list[dict[str, Any]] = []
    event_rows = 0
    for timeframe in task.timeframes:
        event_path, metadata_path = generation0_event_source(task.pair, timeframe)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        events = pd.read_parquet(
            event_path,
            filters=[
                ("control", "==", "actual"),
                ("level_family", "in", list(VP_EVENT_FAMILIES)),
                ("zone_method", "in", list(task.zones)),
            ],
        )
        events = events.loc[events["level_name"].isin(VP_LEVELS)].copy()
        if task.max_events_per_timeframe is not None:
            events = bounded_sample(events, task.max_events_per_timeframe)
        event_rows += len(events)
        events = attach_states(
            events,
            base=base,
            local_state=local_state,
            cache_path=Path(str(metadata["cache"])),
        )
        for profile in ATTRIBUTE_PROFILES:
            selected = events.loc[
                events["level_name"].isin(profile.level_names)
                & pd.to_numeric(events[profile.attribute_column], errors="coerce").notna()
            ].copy()
            if selected.empty:
                continue
            for assignment in ("actual", "shuffled_attribute"):
                tiered = assign_attribute_tertiles(
                    selected,
                    attribute_column=profile.attribute_column,
                    assignment=assignment,
                    seed_key=f"{task.pair}|{timeframe}|{profile.name}",
                )
                cell_rows.extend(
                    matched_attribute_cells(
                        tiered,
                        attribute_profile=profile,
                        assignment=assignment,
                        horizons=horizons,
                    )
                )
    output = DataFrame(cell_rows)
    if output.empty:
        raise ValueError(f"No matchable G1B cells were produced for {task.pair}.")
    output["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    output["run_request_sha256"] = task.request_sha256
    atomic_write_parquet(output, output_path)
    return {
        "pair": task.pair,
        "status": "completed",
        "source_event_rows": event_rows,
        "cell_rows": len(output),
        "seconds": round(time.perf_counter() - started, 3),
    }


def attach_states(
    events: DataFrame,
    *,
    base: DataFrame,
    local_state: DataFrame,
    cache_path: Path,
) -> DataFrame:
    output = events.copy()
    indexes = output["base_index"].to_numpy(dtype=np.int64)
    for column in local_state:
        if column.startswith("state_"):
            output[column] = numeric_array(local_state[column])[indexes]
    cache = pd.read_parquet(
        cache_path,
        columns=[
            "available_at",
            "source_close",
            "source_high",
            "source_low",
            "source_volume",
            "source_atr_14",
        ],
    ).sort_values("available_at")
    cache["available_at"] = normalize_dates(cache["available_at"])
    source_state = causal_source_state(cache)
    source = pd.concat(
        [cache[["available_at"]].reset_index(drop=True), source_state.reset_index(drop=True)],
        axis=1,
    )
    aligned = pd.merge_asof(
        base[["date"]].sort_values("date"),
        source,
        left_on="date",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    for column in source_state:
        output[column] = numeric_array(aligned[column])[indexes]
    return output


def bounded_sample(frame: DataFrame, maximum: int) -> DataFrame:
    if len(frame) <= maximum:
        return frame
    positions = np.linspace(0, len(frame) - 1, num=maximum, dtype=np.int64)
    return frame.iloc[np.unique(positions)].copy()


def assign_attribute_tertiles(
    events: DataFrame,
    *,
    attribute_column: str,
    assignment: str,
    seed_key: str,
) -> DataFrame:
    group_columns = [
        "source_timeframe",
        "level_name",
        "zone_method",
        "period",
        "approach_state",
    ]
    frames = []
    for key, group in events.groupby(group_columns, observed=True, sort=False):
        if len(group) < 12:
            continue
        values = pd.to_numeric(group[attribute_column], errors="coerce").to_numpy(dtype=float)
        if assignment == "shuffled_attribute":
            rng = np.random.default_rng(stable_hash_int(f"{seed_key}|{key}"))
            values = values[rng.permutation(len(values))]
        elif assignment != "actual":
            raise ValueError(f"Unknown attribute assignment: {assignment}")
        ranks = pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=float)
        tier = np.full(len(group), "middle", dtype=object)
        tier[ranks <= 1.0 / 3.0] = "low"
        tier[ranks >= 2.0 / 3.0] = "high"
        selected = group.copy()
        selected["attribute_value_for_tier"] = values
        selected["attribute_tier"] = tier
        frames.append(selected.loc[selected["attribute_tier"].isin(("high", "low"))])
    return pd.concat(frames, ignore_index=True) if frames else DataFrame()


def matched_attribute_cells(
    events: DataFrame,
    *,
    attribute_profile: AttributeProfile,
    assignment: str,
    horizons: Sequence[int],
    state_features: Sequence[str] = VP_STATE_FEATURES,
) -> list[dict[str, Any]]:
    if events.empty:
        return []
    rows = []
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
        pairs, audit = nearest_state_pairs(
            high,
            low,
            state_columns=state_features,
            pre_distance_atr_caliper=0.10,
            minimum_event_separation_hours=max(horizons),
        )
        if not pairs:
            continue
        left = high.iloc[[pair[0] for pair in pairs]].reset_index(drop=True)
        right = low.iloc[[pair[1] for pair in pairs]].reset_index(drop=True)
        rows.append(
            attribute_cell_row(
                left,
                right,
                key=key,
                profile=attribute_profile,
                assignment=assignment,
                audit=audit,
                distances=[pair[2] for pair in pairs],
                horizons=horizons,
                state_features=state_features,
            )
        )
    return rows


def attribute_cell_row(
    high: DataFrame,
    low: DataFrame,
    *,
    key: tuple[Any, ...],
    profile: AttributeProfile,
    assignment: str,
    audit: dict[str, int],
    distances: Sequence[float],
    horizons: Sequence[int],
    state_features: Sequence[str] = VP_STATE_FEATURES,
) -> dict[str, Any]:
    timeframe, level, zone, period, approach = key
    balance = balance_diagnostics(high, low, state_features)
    pre_difference = np.abs(
        high["pre_distance_atr"].to_numpy(dtype=float)
        - low["pre_distance_atr"].to_numpy(dtype=float)
    )
    separation = (
        (pd.to_datetime(high["event_time"], utc=True) - pd.to_datetime(low["event_time"], utc=True))
        .abs()
        .dt.total_seconds()
        .div(3600.0)
        .to_numpy(dtype=float)
    )
    row: dict[str, Any] = {
        "pair": high["pair"].iloc[0],
        "source_timeframe": timeframe,
        "level_name": level,
        "zone_method": zone,
        "period": period,
        "approach_state": approach,
        "attribute_profile": profile.name,
        "attribute_column": profile.attribute_column,
        "level_role": profile.role,
        "attribute_assignment": assignment,
        "eligible_high_events": audit["eligible_actual"],
        "eligible_low_events": audit["eligible_control"],
        "geometry_eligible_high_events": audit["geometry_eligible_actual"],
        "state_matchable_high_events": audit["state_matchable_actual"],
        "matched_events": len(high),
        "high_attribute_median": float(high["attribute_value_for_tier"].median()),
        "low_attribute_median": float(low["attribute_value_for_tier"].median()),
        "match_distance_median": float(np.median(distances)),
        "match_distance_q90": float(np.quantile(distances, 0.90)),
        "max_absolute_state_smd": balance["max_absolute_smd"],
        "median_absolute_state_smd": balance["median_absolute_smd"],
        "state_features_scored": balance["features_scored"],
        "pre_distance_atr_abs_difference_median": float(np.median(pre_difference)),
        "pre_distance_atr_abs_difference_max": float(np.max(pre_difference)),
        "event_separation_hours_min": float(np.min(separation)),
        "direction_prediction": False,
        "profit_optimization": False,
    }
    for column in outcome_columns(horizons):
        left = outcome_values(high, column)
        right = outcome_values(low, column)
        valid = np.isfinite(left) & np.isfinite(right)
        if not valid.any():
            continue
        delta = float(np.mean(left[valid] - right[valid]))
        metric, _ = outcome_metadata(column)
        row[f"scored__{column}"] = int(valid.sum())
        row[f"high_mean__{column}"] = float(np.mean(left[valid]))
        row[f"low_mean__{column}"] = float(np.mean(right[valid]))
        row[f"delta_mean__{column}"] = delta
        row[f"role_delta_mean__{column}"] = role_oriented_delta(
            profile.role,
            metric,
            delta,
        )
    return row


def combine_pair_outputs(run_id: str, pairs: Sequence[str], *, request_sha256: str) -> DataFrame:
    frames = []
    for pair in pairs:
        path = pair_output_path(run_id, pair)
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(
            validate_pair_output(
                path,
                pair=pair,
                request_sha256=request_sha256,
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else DataFrame()


def validate_pair_output(path: Path, *, pair: str, request_sha256: str) -> DataFrame:
    required = [
        "pair",
        "output_schema_version",
        "run_request_sha256",
        "attribute_profile",
        "attribute_assignment",
        "pre_distance_atr_abs_difference_max",
        "event_separation_hours_min",
    ]
    try:
        contract = pd.read_parquet(path, columns=required)
    except Exception as exc:
        raise ValueError(f"Existing G1B output lacks a valid contract: {path}") from exc
    if contract.empty or set(contract["pair"].astype(str)) != {pair}:
        raise ValueError(f"Existing G1B output has an incompatible pair: {path}")
    if set(contract["output_schema_version"].astype(int)) != {OUTPUT_SCHEMA_VERSION}:
        raise ValueError(f"Existing G1B output has an incompatible schema: {path}")
    if set(contract["run_request_sha256"].astype(str)) != {request_sha256}:
        raise ValueError(f"Existing G1B output belongs to another request: {path}")
    return pd.read_parquet(path)


def repeatability_screen(cells: DataFrame, manifest: dict[str, Any]) -> DataFrame:
    eligible = cells.loc[cells["matched_events"].ge(MIN_CELL_EVENTS)].copy()
    if eligible.empty:
        return DataFrame()
    eligible["balance_quality"] = np.select(
        [
            eligible["max_absolute_state_smd"].le(0.25)
            & eligible["median_absolute_state_smd"].le(0.10),
            eligible["max_absolute_state_smd"].le(0.50)
            & eligible["median_absolute_state_smd"].le(0.20),
        ],
        ["strong", "usable"],
        default="weak_or_sparse",
    )
    cohorts = {
        pair: cohort
        for cohort, pairs in manifest["reporting_groups"]["coin_cohorts"].items()
        for pair in pairs
    }
    scopes = []
    cohort = eligible.copy()
    cohort["market_scope"] = "frozen_cohort:" + cohort["pair"].map(cohorts)
    scopes.append(cohort)
    broad = eligible.copy()
    broad["market_scope"] = "all_top10"
    scopes.append(broad)
    non_btc = eligible.loc[~eligible["pair"].eq("BTC/USDT:USDT")].copy()
    non_btc["market_scope"] = "all_non_btc"
    scopes.append(non_btc)
    working = pd.concat(scopes, ignore_index=True)
    role_columns = [column for column in working if column.startswith("role_delta_mean__")]
    identifiers = [
        "pair",
        "period",
        "market_scope",
        "source_timeframe",
        "level_name",
        "zone_method",
        "attribute_profile",
        "attribute_assignment",
        "level_role",
        "balance_quality",
        "matched_events",
        "max_absolute_state_smd",
    ]
    long = working.melt(
        id_vars=identifiers,
        value_vars=role_columns,
        var_name="outcome",
        value_name="role_oriented_delta",
    )
    long["outcome"] = long["outcome"].str.removeprefix("role_delta_mean__")
    metadata = long["outcome"].apply(outcome_metadata)
    long["metric_family"] = [value[0] for value in metadata]
    long["horizon_hours"] = [value[1] for value in metadata]
    long["role_oriented_positive"] = long["role_oriented_delta"].gt(0.0)
    keys = [
        "market_scope",
        "source_timeframe",
        "zone_method",
        "attribute_profile",
        "attribute_assignment",
        "level_role",
        "balance_quality",
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
            max_absolute_state_smd_max=("max_absolute_state_smd", "max"),
            role_oriented_delta_median=("role_oriented_delta", "median"),
            role_oriented_positive_fraction=("role_oriented_positive", "mean"),
        )
        .reset_index()
    )


def integrity_record(
    *,
    results: Sequence[dict[str, Any]],
    cells: DataFrame,
    technical_smoke: bool,
) -> dict[str, Any]:
    adequate = cells.loc[cells["matched_events"].ge(MIN_CELL_EVENTS)]
    return {
        "created_at_utc": utc_now(),
        "technical_smoke_not_evidence": technical_smoke,
        "pairs": len(results),
        "cell_rows": len(cells),
        "cells_with_at_least_20_matches": len(adequate),
        "cells_at_least_20_with_max_state_smd_le_0_50": int(
            adequate["max_absolute_state_smd"].le(0.50).sum()
        ),
        "actual_attribute_cells": int(cells["attribute_assignment"].eq("actual").sum()),
        "shuffled_attribute_cells": int(
            cells["attribute_assignment"].eq("shuffled_attribute").sum()
        ),
        "pre_distance_atr_abs_difference_max": float(
            cells["pre_distance_atr_abs_difference_max"].max()
        ),
        "event_separation_hours_min": float(cells["event_separation_hours_min"].min()),
        "attributes_use_outcomes_for_tiering": False,
        "direction_prediction": False,
        "profit_optimization": False,
    }


def pair_stem(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")


if __name__ == "__main__":
    raise SystemExit(main())

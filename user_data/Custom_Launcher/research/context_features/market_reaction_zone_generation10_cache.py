from __future__ import annotations

# Bound numerical libraries before pandas and the research helpers are imported.
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
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation6 as g6f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation7 as g7f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation6_direct_screen as g6d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation6_event_cache as g6e,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation8_cache as g8c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation10_freeze as g10z,
)


CACHE_ID = "g10_untouched_confirmation_cache_20260821a"
RECORD_ROOT = g10z.OUTPUT_ROOT / "generation10_shared" / CACHE_ID
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData")
    / "research_outputs"
    / "market_reaction_zones"
    / "generation10_shared"
    / CACHE_ID
)
RAW_EVENT_DIR = ARTIFACT_ROOT / "raw_standard_events"
RAW_SUMMARY_DIR = ARTIFACT_ROOT / "raw_standard_summaries"
G6_FEATURE_DIR = ARTIFACT_ROOT / "g6_feature_cache"
G6_EVENT_DIR = ARTIFACT_ROOT / "g6_event_cache"
G6_EVALUATION_DIR = ARTIFACT_ROOT / "g6_evaluation_cache"
FEATURE_DIR = ARTIFACT_ROOT / "g10_feature_cache"
EVENT_DIR = ARTIFACT_ROOT / "g10_event_cache"
MAX_WORKERS = 4
G6_SHARED_MANIFEST = (
    g10z.OUTPUT_ROOT
    / "generation6_branches"
    / "g6_common_event_cache"
    / "g6_common_event_full_20260821a"
    / "manifest.json"
)

LEVEL_SUFFIXES = (
    "event_count",
    "distinct_family_count",
    "distinct_timeframe_count",
    "mean_zone_half_width_atr",
    "median_pre_distance_atr",
    "mean_contact_close_distance_atr",
)
VOLATILITY_ABSOLUTE = ("atr_fraction", "prior_range_atr")
VOLATILITY_COMPRESSION = ("bollinger_width", "range_contraction_ratio")


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def load_freeze() -> dict[str, Any]:
    return g10z.validate_existing_freeze(g10z.FREEZE_PATH)


def load_g6_shared_features() -> dict[str, Any]:
    manifest = json.loads(G6_SHARED_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed_shared_causal_event_cache":
        raise ValueError("Generation 6 shared causal features are not terminal.")
    return manifest["shared_features"]


def raw_pair_task(pair: str, shared: dict[str, Any]) -> g6e.PairTask:
    return g6e.PairTask(
        cohort="normal",
        pair=pair,
        manifest_path=str(g10z.G5_SOURCE_MANIFEST),
        supported_cells_path=str(g6e.SUPPORTED_LEVEL_CELLS),
        cross_market_path=str(shared["normal_cross_market"]["path"]),
        context_path=str(shared["gdelt_and_topics"]["path"]),
        orderbook_path=str(shared["btc_orderbook"]["path"]),
        event_dir=str(RAW_EVENT_DIR),
        summary_dir=str(RAW_SUMMARY_DIR),
        overwrite=False,
    )


def build_pair_events_g10(task: g6e.PairTask) -> dict[str, Any]:
    """Build the Generation 6-compatible surface from the fixed G5C period schema.

    The G5C source manifest intentionally stores its terminal boundary in the frozen
    chronological periods rather than the later Generation 6 convenience field. This
    adapter keeps metadata validation against the original frozen manifest and derives
    the processing end only from those explicit period boundaries.
    """

    manifest_path = Path(task.manifest_path)
    manifest = g0.load_manifest(manifest_path)
    storage = g0.manifest_storage_paths(manifest)
    stem = f"{task.cohort}__{g0.pair_file_stem(task.pair)}"
    event_path = Path(task.event_dir) / f"{stem}.parquet"
    summary_path = Path(task.summary_dir) / f"{stem}.parquet"
    if event_path.is_file() and summary_path.is_file() and not task.overwrite:
        existing = pd.read_parquet(event_path, columns=["control"])
        return g6e.inventory_record(
            task,
            status="existing",
            event_path=event_path,
            summary_path=summary_path,
            events=existing,
            supported_specs=-1,
            causal_violations=0,
        )

    supported = pd.read_csv(task.supported_cells_path)
    supported = supported.loc[
        supported["cohort"].eq(task.cohort)
        & supported["supported"].astype(str).str.casefold().eq("true")
    ].copy()
    allowed = {
        (
            str(row.timeframe),
            str(row.level_family),
            str(row.level_name),
            str(row.representation),
        )
        for row in supported.itertuples(index=False)
    }
    base = g0.prepare_base_market_frame(task.pair, manifest)
    end = max(
        pd.Timestamp(period["end_utc_exclusive"])
        for period in manifest["data"]["chronological_periods"]
    )
    base = base.loc[base["date"].lt(end)].reset_index(drop=True)
    paths = g0.future_path_matrices(base, max(g6e.HORIZONS))
    state_features, _ = g6e.g6.causal_state_features(base)
    state_features = state_features.add_prefix("state__")
    enrichment = g6e.build_enrichment(task, base, state_features)

    merged_by_timeframe: dict[str, DataFrame] = {}
    specs_by_timeframe: dict[str, list[Any]] = {}
    mechanism_contacts: dict[tuple[str, str], np.ndarray] = {}
    side_contacts: dict[tuple[str, str], np.ndarray] = {}
    causal_violations = 0
    for timeframe in manifest["data"]["source_timeframes"]:
        cache_path = g0.level_cache_path(
            task.pair,
            timeframe,
            ("core", "generic"),
            cache_dir=storage.cache_dir,
        )
        g0.validate_cache_metadata(cache_path, manifest_path)
        level_cache = pd.read_parquet(cache_path).sort_values("available_at")
        level_cache = level_cache.reset_index(drop=True)
        level_cache["available_at"] = g0.normalize_dates(level_cache["available_at"])
        level_cache["source_open"] = g0.normalize_dates(level_cache["source_open"])
        merged = pd.merge_asof(
            base.sort_values("date"),
            level_cache,
            left_on="date",
            right_on="available_at",
            direction="backward",
            allow_exact_matches=True,
        )
        present = merged["available_at"].notna()
        causal_violations += int(
            (merged.loc[present, "available_at"] > merged.loc[present, "date"]).sum()
        )
        selected = g6e.g6.selected_level_specs(level_cache)
        merged_by_timeframe[timeframe] = merged
        specs_by_timeframe[timeframe] = list(selected)
        g6e.collect_contact_masks(
            merged,
            selected,
            timeframe=timeframe,
            mechanism_contacts=mechanism_contacts,
            side_contacts=side_contacts,
        )
    if causal_violations:
        raise AssertionError(f"Future level rows for {task.pair}: {causal_violations}")
    relationship_masks = g6e.g6.timeframe_relationship_masks(
        mechanism_contacts=mechanism_contacts,
        side_contacts=side_contacts,
        row_count=len(base),
    )

    event_frames: list[DataFrame] = []
    opened_specs = 0
    for timeframe, merged in merged_by_timeframe.items():
        for spec in specs_by_timeframe[timeframe]:
            identity = (timeframe, spec.family, spec.name, spec.representation)
            if identity not in allowed:
                continue
            frames = g0.build_spec_events(
                merged=merged,
                paths=paths,
                spec=spec,
                pair=task.pair,
                timeframe=timeframe,
                zone_methods=g6e.ZONE_METHODS,
                controls=g6e.CONTROLS,
                horizons=g6e.HORIZONS,
            )
            if not frames:
                continue
            opened_specs += 1
            events = pd.concat(frames, ignore_index=True)
            events.insert(0, "cohort", task.cohort)
            events["mechanism_group"] = g6e.g6.mechanism_group(spec)
            events["level_side"] = g6e.g6.level_side(spec)
            g6e.attach_relationship_flags(events, relationship_masks)
            events = g6e.attach_enrichment(events, enrichment)
            event_frames.append(events)

    events = pd.concat(event_frames, ignore_index=True) if event_frames else DataFrame()
    if not events.empty:
        events.sort_values(
            ["event_time", "source_timeframe", "level_family", "level_name", "control"],
            inplace=True,
            ignore_index=True,
        )
    # Generation 10 consumes the event rows directly. The old atlas summary expects
    # every long Generation 0 horizon, while this frozen adapter intentionally rebuilds
    # only the Generation 6 horizons (1h, 2h, 4h, 8h). Keep a small inventory sidecar
    # instead of fabricating unused 12h/24h/48h outcomes.
    summary = DataFrame.from_records(
        [
            {
                "cohort": task.cohort,
                "pair": task.pair,
                "event_rows": len(events),
                "summary_scope": "generation10_inventory_only",
            }
        ]
    )
    event_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(events, event_path)
    g0.atomic_write_parquet(summary, summary_path)
    return g6e.inventory_record(
        task,
        status="built",
        event_path=event_path,
        summary_path=summary_path,
        events=events,
        supported_specs=opened_specs,
        causal_violations=causal_violations,
    )


def build_raw_events(pairs: Sequence[str], *, workers: int) -> list[dict[str, Any]]:
    shared = load_g6_shared_features()
    RAW_EVENT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [raw_pair_task(pair, shared) for pair in pairs]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, min(workers, MAX_WORKERS))) as pool:
        futures = {pool.submit(build_pair_events_g10, task): task.pair for task in tasks}
        for future in as_completed(futures):
            pair = futures[future]
            item = future.result()
            results.append(item)
            print(
                json.dumps(
                    {
                        "phase": "g10_raw_standard_events",
                        "pair": pair,
                        "rows": item["event_rows"],
                    }
                ),
                flush=True,
            )
    order = {pair: index for index, pair in enumerate(pairs)}
    return sorted(results, key=lambda item: order[str(item["pair"])])


def build_g6_pair_cache(item: dict[str, Any]) -> dict[str, Any]:
    return g6f.build_pair_cache(
        cohort="normal",
        pair=str(item["pair"]),
        source_path=Path(item["event_path"]),
        feature_dir=G6_FEATURE_DIR,
        event_dir=G6_EVENT_DIR,
        evaluation_dir=G6_EVALUATION_DIR,
    )


def build_g6_caches(
    raw_inventory: Sequence[dict[str, Any]], *, workers: int
) -> list[dict[str, Any]]:
    for path in (G6_FEATURE_DIR, G6_EVENT_DIR, G6_EVALUATION_DIR):
        path.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, MAX_WORKERS))) as pool:
        futures = {
            pool.submit(build_g6_pair_cache, item): str(item["pair"])
            for item in raw_inventory
        }
        for future in as_completed(futures):
            pair = futures[future]
            item = future.result()
            results.append(item)
            print(
                json.dumps(
                    {
                        "phase": "g10_g6_compatible_cache",
                        "pair": pair,
                        "events": item["independent_event_rows"],
                    }
                ),
                flush=True,
            )
    order = {str(item["pair"]): index for index, item in enumerate(raw_inventory)}
    return sorted(results, key=lambda item: order[str(item["pair"])])


def rename_block(
    source: DataFrame,
    *,
    source_prefix: str,
    target_block: str,
    suffixes: Sequence[str],
) -> DataFrame:
    output = DataFrame({"date": pd.to_datetime(source["date"], utc=True)})
    for suffix in suffixes:
        column = f"{source_prefix}__{suffix}"
        if column not in source:
            raise ValueError(f"Missing source feature {column!r}")
        output[f"{target_block}__{suffix}"] = pd.to_numeric(
            source[column], errors="coerce"
        )
    return output


def activity_regime(frame: DataFrame) -> pd.Series:
    column = "g8_participation_duration__relative_volume_band"
    values = pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(
        np.select(
            [values.le(-1.0), values.ge(1.0), values.eq(0.0)],
            ["quiet", "active", "typical"],
            default="unavailable",
        ),
        index=frame.index,
        dtype="string",
    )


def common_support(frame: DataFrame, candidate_block: str) -> pd.Series:
    ready = pd.Series(True, index=frame.index)
    for block in (
        *g10z.BASE_BLOCKS,
        candidate_block,
        f"{candidate_block}_stale",
        f"{candidate_block}_shuffled",
    ):
        ready &= g8c.complete_block(frame, block)
    return ready


def build_g10_pair_cache(item: dict[str, Any]) -> dict[str, Any]:
    pair = str(item["pair"])
    source_features = pd.read_parquet(item["feature_path"])
    source_features["date"] = pd.to_datetime(
        source_features["date"], utc=True, errors="coerce"
    )
    source_events = pd.read_parquet(item["event_path"])
    source_events["date"] = pd.to_datetime(
        source_events["date"], utc=True, errors="coerce"
    )
    timeline = source_events[["date", "period"]].copy()
    if timeline["date"].duplicated().any():
        raise ValueError(f"Duplicate source event dates for {pair}")

    blocks = [
        rename_block(
            source_features,
            source_prefix="level",
            target_block="g8_level_base",
            suffixes=LEVEL_SUFFIXES,
        ),
        rename_block(
            source_features,
            source_prefix="ohlcv_volatility_range",
            target_block="g8_volatility_absolute",
            suffixes=VOLATILITY_ABSOLUTE,
        ),
        rename_block(
            source_features,
            source_prefix="ohlcv_volatility_range",
            target_block="g8_volatility_compression",
            suffixes=VOLATILITY_COMPRESSION,
        ),
        rename_block(
            source_features,
            source_prefix="ohlcv_volume_pressure",
            target_block="g8_participation_relative_volume",
            suffixes=("relative_volume",),
        ),
    ]
    features = timeline.copy()
    for block in blocks:
        features = features.merge(block, on="date", how="left", validate="one_to_one")

    source_manifest = json.loads(g10z.G5_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    base = g0.prepare_base_market_frame(pair, source_manifest)
    end = max(
        pd.Timestamp(period["end_utc_exclusive"])
        for period in source_manifest["data"]["chronological_periods"]
    )
    base = base.loc[base["date"].lt(end)].reset_index(drop=True)
    continuous = g8c.continuous_state(base)
    duration = g8c.duration_block(
        block="g8_participation_duration",
        cell=g8c.cell_for_branch("normal", "g8b"),
        timeline=timeline,
        source=continuous,
        definitions={
            name: definition
            for name, definition in g6d.STATE_FEATURES.items()
            if name
            in {
                "relative_volume",
                "volume_acceleration_magnitude",
                "absolute_pressure",
                "pressure_persistence",
            }
        },
        predictor_prefix="ohlcv_volume_pressure",
    )
    features = features.merge(duration, on="date", how="left", validate="one_to_one")
    features, placebo_audit = g8c.attach_stale_and_shuffled(
        features,
        pair=pair,
        blocks=(
            "g8_participation_relative_volume",
            "g8_participation_duration",
        ),
    )
    if any(
        row["stale_timestamp_violations"] or row["shuffled_self_matches"]
        for row in placebo_audit
    ):
        raise ValueError(f"Generation 10 placebo violation for {pair}")

    ready_relative = common_support(features, "g8_participation_relative_volume")
    ready_duration = common_support(features, "g8_participation_duration")
    events = source_events[
        [
            "date",
            "period",
            "pair",
            "cohort",
            "market_group",
            "smart_contract_platform",
            *g6f.TARGET_COLUMNS,
        ]
    ].copy()
    events["activity_regime"] = activity_regime(features)
    events["ready__g10_relative_volume_common_support"] = ready_relative.to_numpy()
    events["ready__g10_participation_duration_common_support"] = (
        ready_duration.to_numpy()
    )

    feature_columns = [column for column in features if "__" in column]
    output_features = features[["date", *feature_columns]].copy()
    for column in feature_columns:
        output_features[column] = pd.to_numeric(
            output_features[column], errors="coerce"
        ).astype("float32")
    feature_path = FEATURE_DIR / f"{g0.pair_file_stem(pair)}.parquet"
    event_path = EVENT_DIR / f"{g0.pair_file_stem(pair)}.parquet"
    g0.atomic_write_parquet(output_features, feature_path)
    g0.atomic_write_parquet(events, event_path)

    support_rows: list[dict[str, Any]] = []
    for mechanism, ready in (
        ("relative_volume", ready_relative),
        ("participation_duration", ready_duration),
    ):
        for period, indexes in timeline.groupby("period", observed=True).groups.items():
            mask = ready.loc[indexes]
            support_rows.append(
                {
                    "pair": pair,
                    "mechanism": mechanism,
                    "period": period,
                    "rows": int(mask.sum()),
                    "activity_quiet": int(
                        (mask & events.loc[indexes, "activity_regime"].eq("quiet")).sum()
                    ),
                    "activity_typical": int(
                        (mask & events.loc[indexes, "activity_regime"].eq("typical")).sum()
                    ),
                    "activity_active": int(
                        (mask & events.loc[indexes, "activity_regime"].eq("active")).sum()
                    ),
                }
            )
    return {
        "pair": pair,
        "feature_path": str(feature_path),
        "feature_sha256": g0.sha256_file(feature_path),
        "event_path": str(event_path),
        "event_sha256": g0.sha256_file(event_path),
        "g6_feature_path": item["feature_path"],
        "g6_feature_sha256": item["feature_sha256"],
        "g6_event_path": item["event_path"],
        "g6_event_sha256": item["event_sha256"],
        "evaluation_path": item["evaluation_path"],
        "evaluation_sha256": item["evaluation_sha256"],
        "feature_rows": len(output_features),
        "event_rows": len(events),
        "support_rows": support_rows,
        "placebo_audit": placebo_audit,
    }


def build_final_caches(
    g6_inventory: Sequence[dict[str, Any]], *, workers: int
) -> list[dict[str, Any]]:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, MAX_WORKERS))) as pool:
        futures = {
            pool.submit(build_g10_pair_cache, item): str(item["pair"])
            for item in g6_inventory
        }
        for future in as_completed(futures):
            pair = futures[future]
            item = future.result()
            results.append(item)
            print(
                json.dumps(
                    {
                        "phase": "g10_final_cache",
                        "pair": pair,
                        "rows": item["event_rows"],
                    }
                ),
                flush=True,
            )
    order = {str(item["pair"]): index for index, item in enumerate(g6_inventory)}
    return sorted(results, key=lambda item: order[str(item["pair"])])


def support_table(inventory: Sequence[dict[str, Any]]) -> DataFrame:
    return DataFrame.from_records(
        row for item in inventory for row in item["support_rows"]
    ).sort_values(["mechanism", "period", "pair"])


def validate_support(support: DataFrame) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for mechanism in ("relative_volume", "participation_duration"):
        for period in g10z.CONFIRMATION_PERIODS:
            selected = support.loc[
                support["mechanism"].eq(mechanism) & support["period"].eq(period)
            ]
            scorable = selected.loc[selected["rows"].ge(g7f.MIN_PAIR_SCORABLE_ROWS)]
            decisions.append(
                {
                    "mechanism": mechanism,
                    "period": period,
                    "rows": int(selected["rows"].sum()),
                    "scorable_pairs": len(scorable),
                    "minimum_scorable_pair_rows": (
                        int(scorable["rows"].min()) if len(scorable) else 0
                    ),
                    "quiet_rows": int(selected["activity_quiet"].sum()),
                    "typical_rows": int(selected["activity_typical"].sum()),
                    "active_rows": int(selected["activity_active"].sum()),
                    "supported": bool(
                        len(scorable) >= 5
                        and int(scorable["rows"].sum())
                        >= g7f.MIN_AGGREGATE_SCORABLE_ROWS
                    ),
                }
            )
    return decisions


def build_cache(*, workers: int, overwrite: bool = False) -> dict[str, Any]:
    frozen = load_freeze()
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = RECORD_ROOT / "g10_cache_manifest.json"
    if manifest_path.is_file() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") != "completed_generation10_confirmation_cache":
            raise ValueError("Existing Generation 10 cache is incomplete.")
        for item in existing["inventory"]:
            for key in ("feature", "event"):
                path = Path(item[f"{key}_path"])
                if not path.is_file() or g0.sha256_file(path) != item[f"{key}_sha256"]:
                    raise ValueError(f"Generation 10 {key} cache changed: {path}")
        return existing

    raw = build_raw_events(tuple(frozen["pairs"]), workers=workers)
    g6_inventory = build_g6_caches(raw, workers=workers)
    inventory = build_final_caches(g6_inventory, workers=workers)
    support = support_table(inventory)
    decisions = validate_support(support)
    support_path = RECORD_ROOT / "g10_exact_common_support.csv"
    decisions_path = RECORD_ROOT / "g10_support_decisions.json"
    g0.atomic_write_csv(support, support_path)
    g0.atomic_write_json(decisions, decisions_path)
    if not all(item["supported"] for item in decisions):
        raise ValueError("Generation 10 exact common support is insufficient.")
    manifest = {
        "schema_version": 1,
        "cache_id": CACHE_ID,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation10_confirmation_cache",
        "pairs": list(frozen["pairs"]),
        "periods": list(g10z.CONFIRMATION_PERIODS),
        "inventory": inventory,
        "support_decisions": decisions,
        "source_contracts": {
            "generation10_freeze": artifact(g10z.FREEZE_PATH),
            "fresh_source_manifest": artifact(g10z.G5_SOURCE_MANIFEST),
            "generation6_supported_level_cells": artifact(g6e.SUPPORTED_LEVEL_CELLS),
        },
        "storage": {
            "feature_dir": str(FEATURE_DIR.resolve()),
            "event_dir": str(EVENT_DIR.resolve()),
            "artifact_root": str(ARTIFACT_ROOT.resolve()),
        },
        "integrity": {
            "stale_timestamp_violations": sum(
                row["stale_timestamp_violations"]
                for item in inventory
                for row in item["placebo_audit"]
            ),
            "shuffled_self_matches": sum(
                row["shuffled_self_matches"]
                for item in inventory
                for row in item["placebo_audit"]
            ),
            "profit_features": 0,
            "future_signed_direction_features": 0,
            "volume_target_opened_only_after_generation10_freeze": True,
        },
        "artifacts": {
            "support": artifact(support_path),
            "support_decisions": artifact(decisions_path),
        },
    }
    g0.atomic_write_json(manifest, manifest_path)
    return {**manifest, "manifest_path": str(manifest_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the frozen Generation 10 confirmation cache."
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    result = build_cache(workers=args.workers, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "status": result["status"],
                "pairs": len(result["pairs"]),
                "support_decisions": result["support_decisions"],
                "manifest_path": str(RECORD_ROOT / "g10_cache_manifest.json"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

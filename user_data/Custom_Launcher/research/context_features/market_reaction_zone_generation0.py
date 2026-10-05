from __future__ import annotations

# Importing pandas/numpy only after the thread environment is fixed is intentional.
# ruff: noqa: E402
import os


# Keep numerical libraries from quietly multiplying every process into many threads.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
import re
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
USER_DATA_DIR = REPO_ROOT / "user_data"
DATA_DIR = USER_DATA_DIR / "data" / "binance" / "futures"
OUTPUT_ROOT = USER_DATA_DIR / "research_news_data" / "context_features" / "market_reaction_zones"
LARGE_ARTIFACT_ROOT = (
    Path(r"D:\FreqTradeStuffLargeData") / "research_outputs" / "market_reaction_zones"
)
DEFAULT_MANIFEST = OUTPUT_ROOT / "generation0_manifest.json"
CACHE_DIR = OUTPUT_ROOT / "generation0_cache"
REPORT_DIR = OUTPUT_ROOT / "generation0_reports"
EVENT_DIR = LARGE_ARTIFACT_ROOT / "generation0_events"
DETAILED_SUMMARY_DIR = LARGE_ARTIFACT_ROOT / "generation0_detailed_summaries"
MAX_NEW_WORKERS = 4

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


INDICATOR_FILES = (
    USER_DATA_DIR / "Indicators" / "complex_volume_profile.py",
    USER_DATA_DIR / "Indicators" / "complex_trendline_projection_v2.py",
    USER_DATA_DIR / "Indicators" / "pattern_bos_choch.py",
    USER_DATA_DIR / "Indicators" / "pattern_geometry_v2.py",
    USER_DATA_DIR / "Indicators" / "pattern_reversal.py",
    USER_DATA_DIR / "Indicators" / "pattern_continuation.py",
    USER_DATA_DIR / "Indicators" / "pattern_multi_peak.py",
    USER_DATA_DIR / "Indicators" / "pattern_wolfe_waves.py",
)

VP_COLUMNS = (
    "vp_profile_low",
    "vp_profile_high",
    "vp_poc",
    "vp_vah",
    "vp_val",
    "vp_prior_poc",
    "vp_prior_vah",
    "vp_prior_val",
    "vp_value_area_width",
    "vp_value_area_width_pct",
    "vp_hvn_above",
    "vp_hvn_below",
    "vp_lvn_above",
    "vp_lvn_below",
    "vp_hvn_above_strength",
    "vp_hvn_below_strength",
    "vp_lvn_above_thinness",
    "vp_lvn_below_thinness",
    "vp_score_abs",
    "vp_state",
)

BOS_COLUMNS = (
    "bc_prev_swing_high",
    "bc_prev_swing_low",
    "bc_last_swing_high",
    "bc_last_swing_low",
    "bc_break_level",
    "bc_invalidation_level",
    "bc_bullish_break_level",
    "bc_bearish_break_level",
    "bc_state",
)


@dataclass(frozen=True)
class CacheTask:
    pair: str
    timeframe: str
    families: tuple[str, ...]
    manifest_path: str
    overwrite: bool


@dataclass(frozen=True)
class AtlasTask:
    pair: str
    timeframe: str
    cache_families: tuple[str, ...]
    level_batches: tuple[str, ...]
    zone_methods: tuple[str, ...]
    controls: tuple[str, ...]
    manifest_path: str
    overwrite: bool


@dataclass(frozen=True)
class StoragePaths:
    cache_dir: Path
    report_dir: Path
    event_dir: Path
    detailed_summary_dir: Path


@dataclass(frozen=True)
class LevelSpec:
    name: str
    family: str
    batch: str
    column: str
    representation: str = "settled"
    slope_column: str | None = None
    native_width_column: str | None = None
    native_width_atr_column: str | None = None
    active_columns: tuple[str, ...] = ()
    score_column: str | None = None
    identity_column: str | None = None
    attribute_columns: tuple[str, ...] = ()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 0 market-reaction-zone audit and causal level-cache builder. "
            "It measures level availability; it does not predict direction or profit."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser(
        "preflight", help="Audit source coverage, manifest integrity, and indicator contracts."
    )
    preflight_parser.add_argument("--smoke-rows", type=int, default=1500)
    preflight_parser.add_argument("--skip-indicator-smoke", action="store_true")

    cache_parser = subparsers.add_parser(
        "build-cache", help="Build causally timestamped unchanged-indicator level caches."
    )
    cache_parser.add_argument(
        "--pairs",
        default="all",
        help="Comma-separated symbols/pair strings, or 'all' from the frozen manifest.",
    )
    cache_parser.add_argument(
        "--timeframes",
        default="all",
        help="Comma-separated source timeframes, or 'all' from the frozen manifest.",
    )
    cache_parser.add_argument(
        "--families",
        default="core,generic",
        help="Comma-separated cache families: core, generic, sparse.",
    )
    cache_parser.add_argument("--workers", type=int, default=1)
    cache_parser.add_argument("--overwrite", action="store_true")

    inventory_parser = subparsers.add_parser(
        "cache-inventory", help="Summarize causal level availability in existing caches."
    )
    inventory_parser.add_argument("--pairs", default="all")
    inventory_parser.add_argument("--timeframes", default="all")

    atlas_parser = subparsers.add_parser(
        "atlas", help="Build unchanged-indicator contact episodes and reaction paths."
    )
    atlas_parser.add_argument("--pairs", default="all")
    atlas_parser.add_argument("--timeframes", default="all")
    atlas_parser.add_argument("--cache-families", default="core,generic")
    atlas_parser.add_argument("--level-batches", default="g0b1,g0b2")
    atlas_parser.add_argument(
        "--zones",
        default="tight_base_atr,standard_base_atr,wide_base_atr,native_width",
    )
    atlas_parser.add_argument(
        "--controls",
        default="actual,stale_72h,stale_168h,price_shift,near_miss,matched_random_time",
    )
    atlas_parser.add_argument("--workers", type=int, default=1)
    atlas_parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    storage = manifest_storage_paths(manifest)

    if args.command == "preflight":
        report = run_preflight(
            manifest,
            args.manifest,
            smoke_rows=args.smoke_rows,
            run_indicator_smoke=not args.skip_indicator_smoke,
        )
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        return 0

    pairs = select_pairs(manifest, args.pairs)
    timeframes = select_timeframes(manifest, args.timeframes)

    if args.command == "build-cache":
        families = parse_families(args.families)
        validate_worker_count(args.workers, manifest=manifest)
        results = build_cache_batch(
            pairs=pairs,
            timeframes=timeframes,
            families=families,
            manifest_path=args.manifest,
            workers=args.workers,
            overwrite=args.overwrite,
        )
        failed = [row for row in results if row["status"] == "failed"]
        print(
            json.dumps(
                {
                    "tasks": len(results),
                    "built": sum(row["status"] == "built" for row in results),
                    "existing": sum(row["status"] == "existing" for row in results),
                    "failed": len(failed),
                    "report": str(storage.report_dir / "g0a_cache_build.json"),
                },
                indent=2,
            )
        )
        return 1 if failed else 0

    if args.command == "cache-inventory":
        inventory = write_cache_inventory(pairs, timeframes, manifest=manifest)
        print(
            json.dumps(
                {
                    "cache_files": int(inventory["cache_file"].nunique())
                    if not inventory.empty
                    else 0,
                    "level_rows": len(inventory),
                    "report": str(storage.report_dir / "g0a_level_availability.csv"),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "atlas":
        cache_families = parse_families(args.cache_families)
        level_batches = parse_choice_csv(
            args.level_batches, {"g0b1", "g0b2", "g0b3"}, "level batches"
        )
        zone_methods = parse_choice_csv(
            args.zones,
            {
                "tight_base_atr",
                "standard_base_atr",
                "wide_base_atr",
                "native_width",
            },
            "zone methods",
        )
        controls = parse_choice_csv(
            args.controls,
            {
                "actual",
                "stale_72h",
                "stale_168h",
                "price_shift",
                "near_miss",
                "matched_random_time",
            },
            "controls",
        )
        if "matched_random_time" in controls and "actual" not in controls:
            raise ValueError("matched_random_time requires actual episodes to match.")
        validate_worker_count(args.workers, manifest=manifest)
        results = build_atlas_batch(
            pairs=pairs,
            timeframes=timeframes,
            cache_families=cache_families,
            level_batches=level_batches,
            zone_methods=zone_methods,
            controls=controls,
            manifest_path=args.manifest,
            workers=args.workers,
            overwrite=args.overwrite,
        )
        failed = [row for row in results if row["status"] == "failed"]
        print(
            json.dumps(
                {
                    "tasks": len(results),
                    "built": sum(row["status"] == "built" for row in results),
                    "existing": sum(row["status"] == "existing" for row in results),
                    "failed": len(failed),
                    "report": str(storage.report_dir / "g0b_atlas_build.json"),
                },
                indent=2,
            )
        )
        return 1 if failed else 0

    raise AssertionError(f"Unhandled command: {args.command}")


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen market-reaction-zone manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("objective") != "objective_02b_market_reaction_zone_discovery":
        raise ValueError("Manifest routes to a different objective.")
    if manifest.get("research_boundary", {}).get("predict_direction") is not False:
        raise ValueError("Reaction-zone manifest must explicitly disable direction prediction.")
    if manifest.get("research_boundary", {}).get("optimize_trade_profit") is not False:
        raise ValueError("Reaction-zone manifest must explicitly disable profit optimization.")
    return manifest


def manifest_storage_paths(manifest: dict[str, Any]) -> StoragePaths:
    """Resolve optional per-manifest artifact roots without changing Generation 0 defaults."""
    configured = manifest.get("storage", {})
    if not isinstance(configured, dict):
        raise ValueError("Manifest storage must be a JSON object when provided.")
    return StoragePaths(
        cache_dir=manifest_storage_path(configured, "cache_dir", CACHE_DIR),
        report_dir=manifest_storage_path(configured, "report_dir", REPORT_DIR),
        event_dir=manifest_storage_path(configured, "event_dir", EVENT_DIR),
        detailed_summary_dir=manifest_storage_path(
            configured, "detailed_summary_dir", DETAILED_SUMMARY_DIR
        ),
    )


def manifest_storage_path(configured: dict[str, Any], key: str, default: Path) -> Path:
    value = configured.get(key)
    if value is None:
        return default
    path = Path(str(value))
    if not path.is_absolute():
        raise ValueError(f"Manifest storage.{key} must be an absolute path: {value!r}")
    return path


def select_pairs(manifest: dict[str, Any], requested: str) -> list[str]:
    allowed = list(manifest["data"]["pairs"])
    if requested.strip().lower() == "all":
        return allowed
    lookup: dict[str, str] = {}
    for pair in allowed:
        symbol = pair.split("/", maxsplit=1)[0].upper()
        lookup[symbol] = pair
        lookup[pair.upper()] = pair
    selected: list[str] = []
    for item in split_csv(requested):
        key = item.upper()
        if key not in lookup:
            raise ValueError(f"Pair {item!r} is outside the frozen manifest.")
        if lookup[key] not in selected:
            selected.append(lookup[key])
    return selected


def select_timeframes(manifest: dict[str, Any], requested: str) -> list[str]:
    allowed = list(manifest["data"]["source_timeframes"])
    if requested.strip().lower() == "all":
        return allowed
    selected = split_csv(requested)
    unknown = sorted(set(selected).difference(allowed))
    if unknown:
        raise ValueError(f"Timeframes outside the frozen manifest: {unknown}")
    return selected


def parse_families(value: str) -> tuple[str, ...]:
    allowed = {"core", "generic", "sparse"}
    families = tuple(split_csv(value))
    unknown = sorted(set(families).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown cache families: {unknown}")
    if not families:
        raise ValueError("At least one cache family is required.")
    return families


def parse_choice_csv(value: str, allowed: set[str], description: str) -> tuple[str, ...]:
    selected = tuple(split_csv(value))
    unknown = sorted(set(selected).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown {description}: {unknown}")
    if not selected:
        raise ValueError(f"At least one of {description} is required.")
    return selected


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def manifest_worker_cap(manifest: dict[str, Any] | None = None) -> int:
    if manifest is None:
        return MAX_NEW_WORKERS
    compute = manifest.get("compute", {})
    if not isinstance(compute, dict):
        raise ValueError("Manifest compute must be a JSON object when provided.")
    configured = compute.get(
        "maximum_threads_after_live_process_check",
        compute.get("default_new_worker_cap", MAX_NEW_WORKERS),
    )
    if isinstance(configured, bool) or not isinstance(configured, int):
        raise ValueError("Manifest worker cap must be an integer.")
    logical_processors = os.cpu_count() or 1
    if configured < 1 or configured > logical_processors:
        raise ValueError(
            "Manifest worker cap must be between one and the host logical-processor count "
            f"({logical_processors}); received {configured}."
        )
    return configured


def validate_worker_count(workers: int, *, manifest: dict[str, Any] | None = None) -> None:
    worker_cap = manifest_worker_cap(manifest)
    if workers < 1:
        raise ValueError("Worker count must be at least one.")
    if workers > worker_cap:
        raise ValueError(
            f"This manifest permits at most {worker_cap} new workers; requested {workers}."
        )


def run_preflight(
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    smoke_rows: int,
    run_indicator_smoke: bool,
) -> dict[str, Any]:
    storage = manifest_storage_paths(manifest)
    storage.report_dir.mkdir(parents=True, exist_ok=True)
    coverage = coverage_audit(manifest)
    coverage_path = storage.report_dir / "g0a_ohlcv_coverage.csv"
    atomic_write_csv(coverage, coverage_path)

    smoke: list[dict[str, Any]] = []
    if run_indicator_smoke:
        smoke = indicator_contract_smoke(smoke_rows)
        atomic_write_json(smoke, storage.report_dir / "g0a_indicator_contract_smoke.json")

    hashes = {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in INDICATOR_FILES}
    expected_pairs = len(manifest["data"]["pairs"])
    expected_active = expected_pairs * len(manifest["data"]["source_timeframes"])
    active_rows = coverage[coverage["timeframe"].isin(manifest["data"]["source_timeframes"])]
    missing_active = int((~active_rows["exists"]).sum())
    parked_3d_gappy = int(
        ((coverage["timeframe"] == "3d") & coverage["exists"] & (coverage["gap_count"] > 0)).sum()
    )
    smoke_failures = sum(row["status"] == "failed" for row in smoke)
    report: dict[str, Any] = {
        "created_at_utc": utc_now(),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "indicator_sha256": hashes,
        "system": system_snapshot(manifest_worker_cap(manifest)),
        "summary": {
            "expected_active_pair_timeframes": expected_active,
            "present_active_pair_timeframes": int(active_rows["exists"].sum()),
            "missing_active_pair_timeframes": missing_active,
            "parked_3d_files_with_gaps": parked_3d_gappy,
            "indicator_smoke_checks": len(smoke),
            "indicator_smoke_failures": smoke_failures,
            "direction_prediction_enabled": False,
            "profit_optimization_enabled": False,
        },
    }
    atomic_write_json(report, storage.report_dir / "g0a_preflight.json")
    return report


def coverage_audit(manifest: dict[str, Any]) -> DataFrame:
    rows: list[dict[str, Any]] = []
    timeframes = list(manifest["data"]["source_timeframes"])
    timeframes.extend(manifest["data"].get("parked_timeframes", {}).keys())
    for pair in manifest["data"]["pairs"]:
        for timeframe in dict.fromkeys(timeframes):
            path = ohlcv_path(pair, timeframe)
            row: dict[str, Any] = {
                "pair": pair,
                "timeframe": timeframe,
                "path": str(path),
                "exists": path.is_file(),
                "rows": 0,
                "start_utc": None,
                "end_utc": None,
                "gap_count": 0,
                "max_gap_hours": None,
            }
            if path.is_file():
                dates = pd.read_feather(path, columns=["date"])["date"]
                dates = normalize_dates(dates).sort_values().drop_duplicates()
                expected_hours = timeframe_hours(timeframe)
                gaps = dates.diff().dt.total_seconds().div(3600.0)
                material = gaps[gaps > expected_hours * 1.5]
                row.update(
                    {
                        "rows": len(dates),
                        "start_utc": iso_timestamp(dates.iloc[0]) if len(dates) else None,
                        "end_utc": iso_timestamp(dates.iloc[-1]) if len(dates) else None,
                        "gap_count": len(material),
                        "max_gap_hours": float(material.max()) if len(material) else None,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def indicator_contract_smoke(rows: int) -> list[dict[str, Any]]:
    if rows < 300:
        raise ValueError("Indicator smoke needs at least 300 rows for meaningful warm-up.")
    base = load_ohlcv(ohlcv_path("BTC/USDT:USDT", "1h")).tail(rows).copy()
    checks: list[tuple[str, Callable[[DataFrame], DataFrame]]] = []

    from user_data.Indicators.complex_trendline_projection_v2 import (
        add_trendline_projection_v2,
        build_trendline_projection_v2_state,
    )
    from user_data.Indicators.complex_volume_profile import add_volume_profile
    from user_data.Indicators.pattern_bos_choch import add_bos_choch
    from user_data.Indicators.pattern_continuation import add_pattern_continuation
    from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2
    from user_data.Indicators.pattern_multi_peak import add_pattern_peaks
    from user_data.Indicators.pattern_reversal import add_pattern_reversal
    from user_data.Indicators.pattern_wolfe_waves import add_pattern_wolfe_waves

    def trendline(frame: DataFrame) -> DataFrame:
        state = build_trendline_projection_v2_state(frame, timeframe="1h")
        return add_trendline_projection_v2(frame, state=state, timeframe="1h")

    def geometry(frame: DataFrame) -> DataFrame:
        state = build_trendline_projection_v2_state(frame, timeframe="1h")
        return add_pattern_geometry_v2(frame, trendline_state=state, timeframe="1h")

    checks.extend(
        [
            ("volume_profile", lambda frame: add_volume_profile(frame, prefix="vp")),
            ("trendline_v2", trendline),
            (
                "bos_choch",
                lambda frame: add_bos_choch(
                    frame, prefix="bc", include_sequence=True, include_diagnostics=True
                ),
            ),
            ("geometry_v2", geometry),
            (
                "pattern_reversal",
                lambda frame: add_pattern_reversal(frame, timeframe="1h", output_prefix="pr"),
            ),
            (
                "pattern_continuation",
                lambda frame: add_pattern_continuation(frame, timeframe="1h", output_prefix="pc"),
            ),
            (
                "pattern_multi_peak",
                lambda frame: add_pattern_peaks(frame, timeframe="1h", output_prefix="pp"),
            ),
            (
                "wolfe_waves",
                lambda frame: add_pattern_wolfe_waves(frame, output_prefix="pw"),
            ),
        ]
    )

    result: list[dict[str, Any]] = []
    for name, function in checks:
        original = set(base.columns)
        started = time.perf_counter()
        try:
            output = function(base.copy())
            new_columns = [column for column in output.columns if column not in original]
            numeric = output[new_columns].select_dtypes(include=[np.number, "bool"]).columns
            populated = [column for column in new_columns if output[column].notna().any()]
            level_like = [column for column in new_columns if is_level_like(column)]
            populated_levels = [column for column in level_like if output[column].notna().any()]
            status = "passed" if populated else "passed_sparse"
            result.append(
                {
                    "indicator": name,
                    "status": status,
                    "seconds": round(time.perf_counter() - started, 4),
                    "new_columns": len(new_columns),
                    "numeric_columns": len(numeric),
                    "populated_columns": len(populated),
                    "level_like_columns": len(level_like),
                    "populated_level_like_columns": len(populated_levels),
                    "sample_populated_levels": populated_levels[:12],
                }
            )
        except Exception as exc:  # surfaced in the machine-readable audit
            result.append(
                {
                    "indicator": name,
                    "status": "failed",
                    "seconds": round(time.perf_counter() - started, 4),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return result


def build_cache_batch(
    *,
    pairs: Sequence[str],
    timeframes: Sequence[str],
    families: tuple[str, ...],
    manifest_path: Path,
    workers: int,
    overwrite: bool,
) -> list[dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    storage = manifest_storage_paths(manifest)
    storage.cache_dir.mkdir(parents=True, exist_ok=True)
    storage.report_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        CacheTask(pair, timeframe, families, str(manifest_path), overwrite)
        for pair in pairs
        for timeframe in timeframes
    ]
    results: list[dict[str, Any]] = []
    if workers == 1:
        for task in tasks:
            results.append(build_one_cache(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(build_one_cache, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(
                        {
                            "pair": task.pair,
                            "timeframe": task.timeframe,
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
    results.sort(key=lambda row: (row["pair"], timeframe_hours(row["timeframe"])))
    report = {
        "created_at_utc": utc_now(),
        "manifest": str(manifest_path),
        "families": list(families),
        "workers": workers,
        "system_at_launch": system_snapshot(manifest_worker_cap(manifest)),
        "results": results,
    }
    atomic_write_json(report, storage.report_dir / "g0a_cache_build.json")
    return results


def build_one_cache(task: CacheTask) -> dict[str, Any]:
    started = time.perf_counter()
    manifest_path = Path(task.manifest_path)
    manifest = load_manifest(manifest_path)
    storage = manifest_storage_paths(manifest)
    cache_path = level_cache_path(
        task.pair, task.timeframe, task.families, cache_dir=storage.cache_dir
    )
    metadata_path = cache_path.with_suffix(".meta.json")
    if cache_path.is_file() and not task.overwrite:
        validate_cache_metadata(cache_path, manifest_path)
        return {
            "pair": task.pair,
            "timeframe": task.timeframe,
            "status": "existing",
            "path": str(cache_path),
            "seconds": round(time.perf_counter() - started, 4),
        }

    source_path = ohlcv_path(task.pair, task.timeframe)
    source = load_ohlcv(source_path)
    source = source.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    source["_source_bar_index"] = np.arange(len(source), dtype=np.int64)

    output = DataFrame(
        {
            "source_open": source["date"],
            "available_at": source["date"] + timeframe_delta(task.timeframe),
            "source_bar_index": source["_source_bar_index"],
            "source_open_price": source["open"],
            "source_high": source["high"],
            "source_low": source["low"],
            "source_close": source["close"],
            "source_volume": source["volume"],
            "source_atr_14": wilder_atr(source, 14),
        }
    )

    family_columns: dict[str, list[str]] = {}
    if "core" in task.families:
        core_frames, core_family_columns = build_core_indicator_columns(source, task.timeframe)
        output = pd.concat([output, *core_frames], axis=1)
        family_columns.update(core_family_columns)

    if "generic" in task.families:
        generic = generic_level_frame(source)
        output = pd.concat([output, generic], axis=1)
        family_columns["generic"] = list(generic.columns)

    if "sparse" in task.families:
        sparse, sparse_family_columns = build_sparse_indicator_columns(source, task.timeframe)
        output = pd.concat([output, *sparse], axis=1)
        family_columns.update(sparse_family_columns)

    earliest = pd.Timestamp(manifest["data"]["common_analysis_start_utc"])
    earliest -= pd.Timedelta(days=45)
    output = output.loc[output["available_at"] >= earliest].reset_index(drop=True)
    output = output.loc[:, ~output.columns.duplicated()].copy()
    output.replace([np.inf, -np.inf], np.nan, inplace=True)

    storage.cache_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(output, cache_path)
    metadata = {
        "created_at_utc": utc_now(),
        "pair": task.pair,
        "timeframe": task.timeframe,
        "source": str(source_path),
        "source_sha256": sha256_file(source_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "families": list(task.families),
        "family_columns": family_columns,
        "rows": len(output),
        "start_available_utc": iso_timestamp(output["available_at"].iloc[0])
        if len(output)
        else None,
        "end_available_utc": iso_timestamp(output["available_at"].iloc[-1])
        if len(output)
        else None,
        "columns": list(output.columns),
        "indicator_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in INDICATOR_FILES
        },
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(metadata, metadata_path)
    return {
        "pair": task.pair,
        "timeframe": task.timeframe,
        "status": "built",
        "path": str(cache_path),
        "rows": len(output),
        "columns": len(output.columns),
        "seconds": round(time.perf_counter() - started, 4),
    }


def build_core_indicator_columns(
    source: DataFrame, timeframe: str
) -> tuple[list[DataFrame], dict[str, list[str]]]:
    from user_data.Indicators.complex_trendline_projection_v2 import (
        add_trendline_projection_v2,
        build_trendline_projection_v2_state,
    )
    from user_data.Indicators.complex_volume_profile import (
        VolumeProfileConfig,
        add_volume_profile,
    )
    from user_data.Indicators.pattern_bos_choch import add_bos_choch
    from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2

    frames: list[DataFrame] = []
    family_columns: dict[str, list[str]] = {}

    vp = add_volume_profile(source.copy(), prefix="vp")
    vp_selected = existing_columns(vp, VP_COLUMNS)
    bins = int(VolumeProfileConfig().bins)
    vp_selected["vp_native_half_width"] = (
        pd.to_numeric(vp["vp_profile_high"], errors="coerce")
        - pd.to_numeric(vp["vp_profile_low"], errors="coerce")
    ).abs() / (2.0 * bins)
    frames.append(vp_selected)
    family_columns["volume_profile"] = list(vp_selected.columns)

    bos = add_bos_choch(source.copy(), prefix="bc", include_sequence=True, include_diagnostics=True)
    bos_selected = existing_columns(bos, BOS_COLUMNS)
    frames.append(bos_selected)
    family_columns["bos_choch"] = list(bos_selected.columns)

    state = build_trendline_projection_v2_state(source.copy(), timeframe=timeframe)
    tlv2 = add_trendline_projection_v2(source.copy(), state=state, timeframe=timeframe)
    tlv2_columns = selected_tlv2_columns(tlv2)
    tlv2_selected = tlv2[tlv2_columns].copy()
    frames.append(tlv2_selected)
    family_columns["trendline_projection_v2"] = list(tlv2_selected.columns)

    geometry = add_pattern_geometry_v2(source.copy(), trendline_state=state, timeframe=timeframe)
    geometry_columns = selected_geometry_columns(geometry)
    geometry_selected = geometry[geometry_columns].copy()
    frames.append(geometry_selected)
    family_columns["pattern_geometry_v2"] = list(geometry_selected.columns)

    return frames, family_columns


def build_sparse_indicator_columns(
    source: DataFrame, timeframe: str
) -> tuple[list[DataFrame], dict[str, list[str]]]:
    from user_data.Indicators.pattern_continuation import add_pattern_continuation
    from user_data.Indicators.pattern_multi_peak import add_pattern_peaks
    from user_data.Indicators.pattern_reversal import add_pattern_reversal
    from user_data.Indicators.pattern_wolfe_waves import add_pattern_wolfe_waves

    calls: tuple[tuple[str, Callable[[DataFrame], DataFrame], str], ...] = (
        (
            "pattern_reversal",
            lambda frame: add_pattern_reversal(frame, timeframe=timeframe, output_prefix="pr"),
            "pr_",
        ),
        (
            "pattern_continuation",
            lambda frame: add_pattern_continuation(frame, timeframe=timeframe, output_prefix="pc"),
            "pc_",
        ),
        (
            "pattern_multi_peak",
            lambda frame: add_pattern_peaks(frame, timeframe=timeframe, output_prefix="pp"),
            "pp_",
        ),
        (
            "wolfe_waves",
            lambda frame: add_pattern_wolfe_waves(frame, output_prefix="pw"),
            "pw_",
        ),
    )
    frames: list[DataFrame] = []
    family_columns: dict[str, list[str]] = {}
    original = set(source.columns)
    for family, function, prefix in calls:
        computed = function(source.copy())
        selected = [
            column
            for column in computed.columns
            if column not in original
            and column.startswith(prefix)
            and sparse_column_is_relevant(column)
        ]
        frame = computed[selected].copy()
        frames.append(frame)
        family_columns[family] = selected
    return frames, family_columns


def selected_tlv2_columns(frame: DataFrame) -> list[str]:
    selected: list[str] = []
    ranked_suffixes = (
        "line_rank",
        "score_rank",
        "slope_rank",
        "line_width_rank",
        "line_id_rank",
        "pivot_count_rank",
        "absorbed_pivot_count_rank",
        "projection_end_index_rank",
        "last_confirm_index_rank",
    )
    for column in frame.columns:
        if not column.startswith("tlv2_"):
            continue
        if any(token in column for token in ranked_suffixes):
            selected.append(column)
            continue
        if column.startswith(("tlv2_forecast_support_", "tlv2_forecast_resistance_")):
            if any(
                token in column
                for token in (
                    "zone_center",
                    "zone_lower",
                    "zone_upper",
                    "zone_half_width_atr",
                    "score",
                    "pivot_count",
                    "line_id",
                    "slope",
                    "projection_end_index",
                    "active",
                    "watch_active",
                )
            ):
                selected.append(column)
    return selected


def selected_geometry_columns(frame: DataFrame) -> list[str]:
    selected: list[str] = []
    for column in frame.columns:
        if not column.startswith("pg2_"):
            continue
        if column.startswith("pg2_slot_") and any(
            token in column
            for token in (
                "_active",
                "_family",
                "_direction",
                "_upper",
                "_lower",
                "_upper_slope",
                "_lower_slope",
                "_end_index",
                "_line_score",
                "_width_atr",
                "_confirmation_tier",
            )
        ):
            selected.append(column)
    return selected


def sparse_column_is_relevant(column: str) -> bool:
    return any(
        token in column
        for token in (
            "level",
            "upper",
            "lower",
            "line",
            "target",
            "active",
            "present",
            "confirmed",
            "score",
            "direction",
            "state",
        )
    )


def generic_level_frame(source: DataFrame) -> DataFrame:
    close = numeric_series(source["close"])
    high = numeric_series(source["high"])
    low = numeric_series(source["low"])
    volume = numeric_series(source["volume"]).clip(lower=0.0)
    typical = (high + low + close) / 3.0
    atr = wilder_atr(source, 14)
    output: dict[str, Series] = {}

    for window in (24, 168, 720):
        output[f"generic_rolling_high_{window}"] = high.rolling(window, min_periods=window).max()
        output[f"generic_rolling_low_{window}"] = low.rolling(window, min_periods=window).min()

    for window in (20, 50, 200):
        output[f"generic_ema_{window}"] = close.ewm(
            span=window, adjust=False, min_periods=window
        ).mean()
        output[f"generic_sma_{window}"] = close.rolling(window, min_periods=window).mean()

    for window in (24, 168):
        numerator = (typical * volume).rolling(window, min_periods=window).sum()
        denominator = volume.rolling(window, min_periods=window).sum()
        output[f"generic_vwap_{window}"] = numerator.div(denominator.replace(0.0, np.nan))

    mid = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std(ddof=0)
    output["generic_bb20_mid"] = mid
    output["generic_bb20_upper"] = mid + 2.0 * std
    output["generic_bb20_lower"] = mid - 2.0 * std

    nice_step = nice_number_step(4.0 * atr)
    output["generic_round_step"] = nice_step
    output["generic_round_below"] = np.floor(close / nice_step) * nice_step
    output["generic_round_nearest"] = np.round(close / nice_step) * nice_step
    output["generic_round_above"] = np.ceil(close / nice_step) * nice_step
    return pd.DataFrame(output, index=source.index)


def nice_number_step(target: Series) -> Series:
    positive = numeric_series(target).where(target > 0.0)
    exponent = np.floor(np.log10(positive))
    scale = np.power(10.0, exponent)
    fraction = positive / scale
    nice_fraction = np.select(
        [fraction < 1.5, fraction < 3.5, fraction < 7.5],
        [1.0, 2.0, 5.0],
        default=10.0,
    )
    return pd.Series(nice_fraction * scale, index=target.index, dtype="float64")


def level_specs(  # noqa: C901 - explicit frozen family registry is kept together
    frame: DataFrame, batches: Sequence[str]
) -> list[LevelSpec]:
    selected_batches = set(batches)
    specs: list[LevelSpec] = []

    def add(spec: LevelSpec) -> None:
        if spec.batch in selected_batches and spec.column in frame.columns:
            specs.append(spec)

    for column in ("vp_poc", "vp_vah", "vp_val"):
        add(
            LevelSpec(
                name=column.removeprefix("vp_"),
                family="volume_profile_settled",
                batch="g0b1",
                column=column,
                native_width_column="vp_native_half_width",
                score_column="vp_score_abs",
                attribute_columns=("vp_value_area_width_pct", "vp_state"),
            )
        )
    for column in ("vp_hvn_above", "vp_hvn_below", "vp_lvn_above", "vp_lvn_below"):
        attribute = (
            column.replace("vp_hvn_", "vp_hvn_") + "_strength"
            if "hvn" in column
            else column.replace("vp_lvn_", "vp_lvn_") + "_thinness"
        )
        add(
            LevelSpec(
                name=column.removeprefix("vp_"),
                family="volume_profile_nodes",
                batch="g0b1",
                column=column,
                native_width_column="vp_native_half_width",
                score_column=attribute,
                attribute_columns=("vp_value_area_width_pct", "vp_state"),
            )
        )
    for column in ("vp_prior_poc", "vp_prior_vah", "vp_prior_val"):
        add(
            LevelSpec(
                name=column.removeprefix("vp_"),
                family="volume_profile_explicit_prior",
                batch="g0b1",
                column=column,
                native_width_column="vp_native_half_width",
                score_column="vp_score_abs",
                attribute_columns=("vp_value_area_width_pct", "vp_state"),
            )
        )

    for column in ("bc_last_swing_high", "bc_last_swing_low"):
        add(
            LevelSpec(
                name=column.removeprefix("bc_"),
                family="confirmed_swing",
                batch="g0b1",
                column=column,
                attribute_columns=("bc_state",),
            )
        )

    for side in ("support", "resistance"):
        for rank in range(3):
            base = f"tlv2_{side}"
            column = f"{base}_line_rank{rank}"
            for representation in ("projected", "held"):
                add(
                    LevelSpec(
                        name=f"{side}_rank{rank}",
                        family="tlv2_ranked",
                        batch="g0b1",
                        column=column,
                        representation=representation,
                        slope_column=f"{base}_slope_rank{rank}"
                        if representation == "projected"
                        else None,
                        native_width_column=f"{base}_line_width_rank{rank}",
                        score_column=f"{base}_score_rank{rank}",
                        identity_column=f"{base}_line_id_rank{rank}",
                        attribute_columns=(
                            f"{base}_pivot_count_rank{rank}",
                            f"{base}_absorbed_pivot_count_rank{rank}",
                        ),
                    )
                )

        base = f"tlv2_forecast_{side}"
        for boundary in ("center", "lower", "upper"):
            column = f"{base}_zone_{boundary}"
            for representation in ("projected", "held"):
                add(
                    LevelSpec(
                        name=f"forecast_{side}_{boundary}",
                        family="tlv2_forecast_zone",
                        batch="g0b1",
                        column=column,
                        representation=representation,
                        slope_column=f"{base}_slope" if representation == "projected" else None,
                        native_width_atr_column=f"{base}_zone_half_width_atr"
                        if boundary == "center"
                        else None,
                        active_columns=(f"{base}_active", f"{base}_watch_active"),
                        score_column=f"{base}_score",
                        identity_column=f"{base}_line_id",
                        attribute_columns=(f"{base}_pivot_count",),
                    )
                )

    for slot in range(1, 5):
        base = f"pg2_slot_{slot}"
        for boundary in ("upper", "lower"):
            column = f"{base}_{boundary}"
            for representation in ("projected", "held"):
                add(
                    LevelSpec(
                        name=f"slot{slot}_{boundary}",
                        family="geometry_boundary",
                        batch="g0b1",
                        column=column,
                        representation=representation,
                        slope_column=f"{base}_{boundary}_slope"
                        if representation == "projected"
                        else None,
                        active_columns=(f"{base}_active",),
                        score_column=f"{base}_line_score",
                        identity_column=f"{base}_family",
                        attribute_columns=(
                            f"{base}_direction",
                            f"{base}_width_atr",
                            f"{base}_confirmation_tier",
                        ),
                    )
                )

    for column in candidate_level_columns(frame):
        if not column.startswith("generic_"):
            continue
        if column.startswith("generic_rolling_"):
            family = "generic_prior_range"
        elif column.startswith("generic_vwap_"):
            family = "generic_rolling_vwap"
        elif column.startswith(("generic_ema_", "generic_sma_")):
            family = "generic_moving_average"
        elif column.startswith("generic_bb20_"):
            family = "generic_bollinger"
        else:
            family = "generic_round_number"
        add(
            LevelSpec(
                name=column.removeprefix("generic_"),
                family=family,
                batch="g0b2",
                column=column,
                attribute_columns=("generic_round_step",)
                if family == "generic_round_number"
                else (),
            )
        )

    for column in candidate_level_columns(frame):
        if not column.startswith(("pr_", "pc_", "pp_", "pw_")):
            continue
        prefix = column.split("_", maxsplit=1)[0]
        family = {
            "pr": "pattern_reversal",
            "pc": "pattern_continuation",
            "pp": "pattern_multi_peak",
            "pw": "pattern_wolfe_wave",
        }[prefix]
        pattern_base = sparse_pattern_base(column)
        present_column = f"{pattern_base}_pattern_present"
        score_column = f"{pattern_base}_indicator_score"
        direction_column = f"{pattern_base}_direction"
        confirmed_direction_column = f"{pattern_base}_confirmed_direction"
        attribute_columns = tuple(
            candidate
            for candidate in (
                f"{pattern_base}_pattern_confirmed",
                direction_column,
                confirmed_direction_column,
            )
            if candidate in frame.columns
        )
        add(
            LevelSpec(
                name=column.removeprefix(prefix + "_"),
                family=family,
                batch="g0b3",
                column=column,
                active_columns=(present_column,) if present_column in frame.columns else (),
                score_column=score_column if score_column in frame.columns else None,
                identity_column=(
                    direction_column
                    if direction_column in frame.columns
                    else confirmed_direction_column
                    if confirmed_direction_column in frame.columns
                    else None
                ),
                attribute_columns=attribute_columns,
            )
        )

    # Preserve declared order but remove exact duplicates caused by broad discovery.
    unique: dict[tuple[str, str, str], LevelSpec] = {}
    for spec in specs:
        unique[(spec.column, spec.representation, spec.family)] = spec
    return list(unique.values())


def sparse_pattern_base(column: str) -> str:
    for suffix in (
        "_confirmation_level",
        "_invalidation_level",
        "_target_level",
        "_upper",
        "_lower",
    ):
        if column.endswith(suffix):
            return column.removesuffix(suffix)
    raise ValueError(f"Sparse candidate does not have a supported price-field suffix: {column}")


def build_atlas_batch(
    *,
    pairs: Sequence[str],
    timeframes: Sequence[str],
    cache_families: tuple[str, ...],
    level_batches: tuple[str, ...],
    zone_methods: tuple[str, ...],
    controls: tuple[str, ...],
    manifest_path: Path,
    workers: int,
    overwrite: bool,
) -> list[dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    storage = manifest_storage_paths(manifest)
    storage.event_dir.mkdir(parents=True, exist_ok=True)
    storage.detailed_summary_dir.mkdir(parents=True, exist_ok=True)
    storage.report_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        AtlasTask(
            pair=pair,
            timeframe=timeframe,
            cache_families=cache_families,
            level_batches=level_batches,
            zone_methods=zone_methods,
            controls=controls,
            manifest_path=str(manifest_path),
            overwrite=overwrite,
        )
        for pair in pairs
        for timeframe in timeframes
    ]
    results: list[dict[str, Any]] = []
    if workers == 1:
        for task in tasks:
            try:
                results.append(build_one_atlas(task))
            except Exception as exc:
                results.append(atlas_failure(task, exc))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(build_one_atlas, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(atlas_failure(task, exc))
    results.sort(key=lambda row: (row["pair"], timeframe_hours(row["timeframe"])))
    atomic_write_json(
        {
            "created_at_utc": utc_now(),
            "manifest": str(manifest_path),
            "workers": workers,
            "cache_families": list(cache_families),
            "level_batches": list(level_batches),
            "zone_methods": list(zone_methods),
            "controls": list(controls),
            "system_at_launch": system_snapshot(manifest_worker_cap(manifest)),
            "results": results,
        },
        storage.report_dir / "g0b_atlas_build.json",
    )
    return results


def atlas_failure(task: AtlasTask, exc: Exception) -> dict[str, Any]:
    return {
        "pair": task.pair,
        "timeframe": task.timeframe,
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
    }


def build_one_atlas(task: AtlasTask) -> dict[str, Any]:
    started = time.perf_counter()
    manifest_path = Path(task.manifest_path)
    manifest = load_manifest(manifest_path)
    storage = manifest_storage_paths(manifest)
    cache_path = level_cache_path(
        task.pair,
        task.timeframe,
        task.cache_families,
        cache_dir=storage.cache_dir,
    )
    if not cache_path.is_file():
        raise FileNotFoundError(f"Required causal level cache does not exist: {cache_path}")
    cache_metadata = validate_cache_metadata(cache_path, manifest_path)
    output_path = atlas_event_path(task, event_dir=storage.event_dir)
    summary_path = atlas_summary_path(task, detailed_summary_dir=storage.detailed_summary_dir)
    metadata_path = atlas_metadata_path(task, report_dir=storage.report_dir)
    if output_path.is_file() and summary_path.is_file() and not task.overwrite:
        return {
            "pair": task.pair,
            "timeframe": task.timeframe,
            "status": "existing",
            "path": str(output_path),
            "seconds": round(time.perf_counter() - started, 4),
        }

    base = prepare_base_market_frame(task.pair, manifest)
    cache = pd.read_parquet(cache_path).sort_values("available_at").reset_index(drop=True)
    cache["available_at"] = normalize_dates(cache["available_at"])
    cache["source_open"] = normalize_dates(cache["source_open"])
    merged = pd.merge_asof(
        base.sort_values("date"),
        cache,
        left_on="date",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    if (merged["available_at"].dropna() > merged.loc[merged["available_at"].notna(), "date"]).any():
        raise AssertionError("Causal as-of merge admitted a future source row.")

    specs = level_specs(cache, task.level_batches)
    if not specs:
        raise ValueError("No candidate level columns matched the requested batches.")
    paths = future_path_matrices(merged, max(manifest["reaction_definition"]["horizons_hours"]))
    durable_controls = {"actual", "matched_random_time"}
    stored_event_frames: list[DataFrame] = []
    summary_frames: list[DataFrame] = []
    evaluated_event_rows = 0
    first_event: pd.Timestamp | None = None
    last_event: pd.Timestamp | None = None
    for spec in specs:
        spec_frames = build_spec_events(
            merged=merged,
            paths=paths,
            spec=spec,
            pair=task.pair,
            timeframe=task.timeframe,
            zone_methods=task.zone_methods,
            controls=task.controls,
            horizons=tuple(manifest["reaction_definition"]["horizons_hours"]),
        )
        if not spec_frames:
            continue
        spec_events = pd.concat(spec_frames, ignore_index=True)
        evaluated_event_rows += len(spec_events)
        spec_first = pd.Timestamp(spec_events["event_time"].min())
        spec_last = pd.Timestamp(spec_events["event_time"].max())
        first_event = spec_first if first_event is None else min(first_event, spec_first)
        last_event = spec_last if last_event is None else max(last_event, spec_last)
        summary_frames.append(summarize_atlas_events(spec_events, manifest))
        durable = spec_events.loc[spec_events["control"].isin(durable_controls)]
        if not durable.empty:
            stored_event_frames.append(durable.reset_index(drop=True))

    stored_events = (
        pd.concat(stored_event_frames, ignore_index=True) if stored_event_frames else DataFrame()
    )
    if not stored_events.empty:
        stored_events.sort_values(
            ["event_time", "level_family", "level_name", "zone_method", "control"],
            inplace=True,
            ignore_index=True,
        )
    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else DataFrame()
    storage.event_dir.mkdir(parents=True, exist_ok=True)
    storage.detailed_summary_dir.mkdir(parents=True, exist_ok=True)
    storage.report_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(stored_events, output_path)
    atomic_write_parquet(summary, summary_path)
    metadata = {
        "created_at_utc": utc_now(),
        "pair": task.pair,
        "timeframe": task.timeframe,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "cache": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "cache_created_at_utc": cache_metadata["created_at_utc"],
        "event_output": str(output_path),
        "summary_output": str(summary_path),
        "storage_tier": "event_rows_and_detailed_summary_on_d_metadata_on_c",
        "cache_families": list(task.cache_families),
        "level_batches": list(task.level_batches),
        "zone_methods": list(task.zone_methods),
        "controls": list(task.controls),
        "level_specs": [spec_to_dict(spec) for spec in specs],
        "evaluated_event_rows": evaluated_event_rows,
        "stored_event_rows": len(stored_events),
        "stored_event_controls": sorted(durable_controls.intersection(task.controls)),
        "summary_rows": len(summary),
        "first_event_utc": iso_timestamp(first_event) if first_event is not None else None,
        "last_event_utc": iso_timestamp(last_event) if last_event is not None else None,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(metadata, metadata_path)
    return {
        "pair": task.pair,
        "timeframe": task.timeframe,
        "status": "built",
        "path": str(output_path),
        "evaluated_events": evaluated_event_rows,
        "stored_events": len(stored_events),
        "summary_rows": len(summary),
        "specs": len(specs),
        "seconds": round(time.perf_counter() - started, 4),
    }


def prepare_base_market_frame(pair: str, manifest: dict[str, Any]) -> DataFrame:
    frame = load_ohlcv(ohlcv_path(pair, manifest["data"]["base_timeframe"]))
    frame = frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    high = numeric_series(frame["high"])
    low = numeric_series(frame["low"])
    close = numeric_series(frame["close"])
    volume = numeric_series(frame["volume"])
    candle_range = high - low
    pressure = (2.0 * close - high - low).div(candle_range.replace(0.0, np.nan))
    frame["base_atr"] = wilder_atr(frame, 14).shift(1)
    frame["pre_close"] = close.shift(1)
    frame["pre_range_median_24"] = candle_range.rolling(24, min_periods=24).median().shift(1)
    frame["pre_volume_median_24"] = volume.rolling(24, min_periods=24).median().shift(1)
    frame["pre_pressure_mean_24"] = pressure.rolling(24, min_periods=24).mean().shift(1)
    frame["candle_pressure"] = pressure
    frame["contact_range_atr"] = candle_range.div(frame["base_atr"])
    frame["causal_volatility"] = frame["base_atr"].div(frame["pre_close"].abs())
    common_start = pd.Timestamp(manifest["data"]["common_analysis_start_utc"])
    frame = frame.loc[frame["date"] >= common_start].reset_index(drop=True)
    frame["period"] = assign_periods(frame["date"], manifest)
    frame["volatility_band"] = within_period_quantile_band(frame, "causal_volatility", bins=5)
    frame["contact_range_band"] = within_period_quantile_band(frame, "contact_range_atr", bins=5)
    return frame


def validate_cache_metadata(cache_path: Path, manifest_path: Path) -> dict[str, Any]:
    metadata_path = cache_path.with_suffix(".meta.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Causal cache metadata is missing: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError(f"Cache {cache_path.name} was built from a different frozen manifest.")
    source_value = metadata.get("source")
    if not source_value:
        raise ValueError(f"Cache {cache_path.name} metadata does not identify its OHLCV source.")
    source_path = Path(str(source_value))
    if not source_path.is_file():
        raise FileNotFoundError(f"Cache {cache_path.name} OHLCV source is missing: {source_path}")
    recorded_source_sha256 = metadata.get("source_sha256")
    current_source_sha256 = sha256_file(source_path)
    if recorded_source_sha256 != current_source_sha256:
        raise ValueError(
            f"Cache {cache_path.name} is stale because its OHLCV source changed: {source_path}"
        )
    recorded_indicators = metadata.get("indicator_sha256")
    if not isinstance(recorded_indicators, dict) or not recorded_indicators:
        raise ValueError(
            f"Cache {cache_path.name} metadata does not identify its indicator sources."
        )
    for relative_path, recorded_sha256 in recorded_indicators.items():
        indicator_path = (REPO_ROOT / str(relative_path)).resolve()
        try:
            indicator_path.relative_to(REPO_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Cache {cache_path.name} contains an indicator path outside the repository: "
                f"{relative_path}"
            ) from exc
        if not indicator_path.is_file():
            raise FileNotFoundError(
                f"Cache {cache_path.name} indicator source is missing: {indicator_path}"
            )
        if sha256_file(indicator_path) != recorded_sha256:
            raise ValueError(
                f"Cache {cache_path.name} is stale because an indicator source changed: "
                f"{indicator_path}"
            )
    if metadata.get("direction_prediction") is not False:
        raise ValueError(f"Cache {cache_path.name} does not preserve the direction boundary.")
    if metadata.get("profit_optimization") is not False:
        raise ValueError(f"Cache {cache_path.name} does not preserve the profit boundary.")
    return metadata


def assign_periods(dates: Series, manifest: dict[str, Any]) -> Series:
    result = pd.Series("outside_frozen_periods", index=dates.index, dtype="object")
    for period in manifest["data"]["chronological_periods"]:
        start = pd.Timestamp(period["start_utc"])
        end = pd.Timestamp(period["end_utc_exclusive"])
        result.loc[(dates >= start) & (dates < end)] = period["id"]
    return result


def within_period_quantile_band(frame: DataFrame, column: str, bins: int) -> Series:
    result = pd.Series(-1, index=frame.index, dtype="int16")
    for _, positions in frame.groupby("period", sort=False).groups.items():
        values = numeric_series(frame.loc[positions, column])
        valid = values.notna()
        if valid.sum() < bins:
            continue
        ranks = values.loc[valid].rank(method="average", pct=True)
        band = np.minimum((ranks * bins).astype(int), bins - 1)
        result.loc[band.index] = band.astype("int16")
    return result


def future_path_matrices(frame: DataFrame, max_horizon: int) -> dict[str, np.ndarray]:
    if max_horizon < 1:
        raise ValueError("Future horizon must be positive.")
    arrays = {
        "high": numeric_series(frame["high"]).to_numpy(),
        "low": numeric_series(frame["low"]).to_numpy(),
        "close": numeric_series(frame["close"]).to_numpy(),
        "volume": numeric_series(frame["volume"]).to_numpy(),
        "pressure": numeric_series(frame["candle_pressure"]).to_numpy(),
    }
    result: dict[str, np.ndarray] = {}
    rows = len(frame)
    for name, values in arrays.items():
        matrix = np.full((rows, max_horizon), np.nan, dtype=np.float64)
        for offset in range(1, max_horizon + 1):
            matrix[:-offset, offset - 1] = values[offset:]
        result[name] = matrix
    return result


def build_spec_events(  # noqa: C901 - control ladder is intentionally explicit
    *,
    merged: DataFrame,
    paths: dict[str, np.ndarray],
    spec: LevelSpec,
    pair: str,
    timeframe: str,
    zone_methods: Sequence[str],
    controls: Sequence[str],
    horizons: tuple[int, ...],
) -> list[DataFrame]:
    level = resolved_level_values(merged, spec, timeframe)
    valid = np.isfinite(level) & (level > 0.0)
    if spec.active_columns:
        active = np.zeros(len(merged), dtype=bool)
        for column in spec.active_columns:
            if column in merged.columns:
                active |= bool_array(merged[column])
        valid &= active
    base_atr = numeric_array(merged["base_atr"])
    valid &= np.isfinite(base_atr) & (base_atr > 0.0)
    if valid.sum() == 0:
        return []

    native_width = resolved_native_width(merged, spec)
    source_available = normalize_dates(merged["available_at"])
    source_open = normalize_dates(merged["source_open"])
    outputs: list[DataFrame] = []
    actual_by_zone: dict[str, DataFrame] = {}
    ordinary_controls = [
        control for control in controls if control not in {"near_miss", "matched_random_time"}
    ]

    for zone_method in zone_methods:
        actual_width = zone_half_width(zone_method, level, base_atr, native_width)
        if not np.isfinite(actual_width).any():
            continue
        for control in ordinary_controls:
            transformed = transform_control(
                control=control,
                level=level,
                valid=valid,
                native_width=native_width,
                source_available=source_available,
                source_open=source_open,
                base_atr=base_atr,
                stable_key=f"{pair}|{timeframe}|{spec.column}|{spec.representation}",
            )
            control_level = transformed["level"]
            control_valid = transformed["valid"]
            width = zone_half_width(
                zone_method,
                control_level,
                base_atr,
                transformed["native_width"],
            )
            events = extract_episode_events(
                merged=merged,
                paths=paths,
                spec=spec,
                pair=pair,
                timeframe=timeframe,
                level=control_level,
                valid=control_valid,
                half_width=width,
                source_available=transformed["source_available"],
                source_open=transformed["source_open"],
                control=control,
                zone_method=zone_method,
                horizons=horizons,
                event_kind="contact",
            )
            if not events.empty:
                outputs.append(events)
                if control == "actual":
                    actual_by_zone[zone_method] = events

        if "near_miss" in controls:
            near_miss = extract_episode_events(
                merged=merged,
                paths=paths,
                spec=spec,
                pair=pair,
                timeframe=timeframe,
                level=level,
                valid=valid,
                half_width=actual_width,
                source_available=source_available,
                source_open=source_open,
                control="near_miss",
                zone_method=zone_method,
                horizons=horizons,
                event_kind="near_miss",
            )
            if not near_miss.empty:
                outputs.append(near_miss)

        if "matched_random_time" in controls and zone_method in actual_by_zone:
            random_events = matched_random_time_events(
                merged=merged,
                paths=paths,
                actual=actual_by_zone[zone_method],
                spec=spec,
                pair=pair,
                timeframe=timeframe,
                zone_method=zone_method,
                horizons=horizons,
            )
            if not random_events.empty:
                outputs.append(random_events)
    return outputs


def resolved_level_values(merged: DataFrame, spec: LevelSpec, timeframe: str) -> np.ndarray:
    level = numeric_array(merged[spec.column])
    if spec.slope_column is None or spec.slope_column not in merged.columns:
        return level
    slope = numeric_array(merged[spec.slope_column])
    elapsed = (
        (normalize_dates(merged["date"]) - normalize_dates(merged["source_open"]))
        .dt.total_seconds()
        .div(timeframe_hours(timeframe) * 3600.0)
    )
    elapsed_array = np.maximum(numeric_array(elapsed), 0.0)
    return level + slope * elapsed_array


def resolved_native_width(merged: DataFrame, spec: LevelSpec) -> np.ndarray:
    if spec.native_width_column and spec.native_width_column in merged.columns:
        return np.abs(numeric_array(merged[spec.native_width_column]))
    if spec.native_width_atr_column and spec.native_width_atr_column in merged.columns:
        return np.abs(numeric_array(merged[spec.native_width_atr_column])) * numeric_array(
            merged["source_atr_14"]
        )
    return np.full(len(merged), np.nan, dtype=np.float64)


def zone_half_width(
    method: str,
    level: np.ndarray,
    base_atr: np.ndarray,
    native_width: np.ndarray,
) -> np.ndarray:
    price_floor = np.abs(level) * 0.0005
    multiplier = {
        "tight_base_atr": 0.10,
        "standard_base_atr": 0.25,
        "wide_base_atr": 0.50,
    }
    if method in multiplier:
        return np.maximum(multiplier[method] * base_atr, price_floor)
    if method == "native_width":
        lower = 0.05 * base_atr
        upper = 1.00 * base_atr
        clipped = np.minimum(np.maximum(native_width, lower), upper)
        clipped[~np.isfinite(native_width)] = np.nan
        return clipped
    raise ValueError(f"Unknown zone method: {method}")


def transform_control(
    *,
    control: str,
    level: np.ndarray,
    valid: np.ndarray,
    native_width: np.ndarray,
    source_available: Series,
    source_open: Series,
    base_atr: np.ndarray,
    stable_key: str,
) -> dict[str, Any]:
    if control == "actual":
        return {
            "level": level.copy(),
            "valid": valid.copy(),
            "native_width": native_width.copy(),
            "source_available": source_available.copy(),
            "source_open": source_open.copy(),
        }
    if control in {"stale_72h", "stale_168h"}:
        shift = int(control.removeprefix("stale_").removesuffix("h"))
        return {
            "level": shift_array(level, shift),
            "valid": shift_array(valid.astype(float), shift, fill=np.nan) == 1.0,
            "native_width": shift_array(native_width, shift),
            "source_available": source_available.shift(shift),
            "source_open": source_open.shift(shift),
        }
    if control == "price_shift":
        sign = 1.0 if stable_hash_int(stable_key) % 2 == 0 else -1.0
        shifted = level + sign * 2.0 * base_atr
        return {
            "level": shifted,
            "valid": valid & np.isfinite(shifted),
            "native_width": native_width.copy(),
            "source_available": source_available.copy(),
            "source_open": source_open.copy(),
        }
    raise ValueError(f"Unsupported ordinary control: {control}")


def shift_array(values: np.ndarray, periods: int, fill: float = np.nan) -> np.ndarray:
    result = np.full(len(values), fill, dtype=np.float64)
    if periods < len(values):
        result[periods:] = values[:-periods]
    return result


def extract_episode_events(
    *,
    merged: DataFrame,
    paths: dict[str, np.ndarray],
    spec: LevelSpec,
    pair: str,
    timeframe: str,
    level: np.ndarray,
    valid: np.ndarray,
    half_width: np.ndarray,
    source_available: Series,
    source_open: Series,
    control: str,
    zone_method: str,
    horizons: tuple[int, ...],
    event_kind: str,
) -> DataFrame:
    high = numeric_array(merged["high"])
    low = numeric_array(merged["low"])
    finite = valid & np.isfinite(half_width) & (half_width > 0.0)
    contact = finite & (high >= level - half_width) & (low <= level + half_width)
    if event_kind == "near_miss":
        expanded = finite & (high >= level - 2.0 * half_width) & (low <= level + 2.0 * half_width)
        condition = expanded & ~contact
    elif event_kind == "contact":
        condition = contact
    else:
        raise ValueError(f"Unknown event kind: {event_kind}")
    starts = episode_start_mask(condition, level, half_width, cooldown=6)
    max_horizon = max(horizons)
    starts[max(len(starts) - max_horizon, 0) :] = False
    indexes = np.flatnonzero(starts)
    if len(indexes) == 0:
        return DataFrame()
    return event_frame(
        merged=merged,
        paths=paths,
        indexes=indexes,
        levels=level[indexes],
        widths=half_width[indexes],
        spec=spec,
        pair=pair,
        timeframe=timeframe,
        control=control,
        zone_method=zone_method,
        source_available=source_available.iloc[indexes].reset_index(drop=True),
        source_open=source_open.iloc[indexes].reset_index(drop=True),
        horizons=horizons,
        match_tier=None,
    )


def episode_start_mask(
    condition: np.ndarray,
    level: np.ndarray,
    half_width: np.ndarray,
    *,
    cooldown: int,
) -> np.ndarray:
    basic_start = cooldown_start_mask(condition, cooldown=cooldown)
    moved = np.zeros(len(level), dtype=bool)
    moved[1:] = np.abs(level[1:] - level[:-1]) > half_width[1:]
    return basic_start | (condition & moved)


def cooldown_start_mask(condition: np.ndarray, *, cooldown: int) -> np.ndarray:
    recent = (
        pd.Series(condition.astype(np.int8))
        .shift(1)
        .rolling(cooldown, min_periods=1)
        .max()
        .fillna(0.0)
        .to_numpy()
        > 0.0
    )
    return condition & ~recent


def event_frame(
    *,
    merged: DataFrame,
    paths: dict[str, np.ndarray],
    indexes: np.ndarray,
    levels: np.ndarray,
    widths: np.ndarray,
    spec: LevelSpec,
    pair: str,
    timeframe: str,
    control: str,
    zone_method: str,
    source_available: Series,
    source_open: Series,
    horizons: tuple[int, ...],
    match_tier: str | Sequence[str] | None,
) -> DataFrame:
    atr = numeric_array(merged["base_atr"])[indexes]
    pre_close = numeric_array(merged["pre_close"])[indexes]
    below = pre_close < levels - widths
    above = pre_close > levels + widths
    approach_code = np.where(below, 1, np.where(above, -1, 0)).astype(np.int8)
    approach = np.select(
        [approach_code == 1, approach_code == -1],
        ["from_below", "from_above"],
        default="already_inside_or_unclear",
    )
    output = DataFrame(
        {
            "pair": pair,
            "source_timeframe": timeframe,
            "batch": spec.batch,
            "level_family": spec.family,
            "level_name": spec.name,
            "level_column": spec.column,
            "representation": spec.representation,
            "control": control,
            "zone_method": zone_method,
            "base_index": indexes.astype(np.int64),
            "event_time": merged["date"].iloc[indexes].reset_index(drop=True),
            "period": merged["period"].iloc[indexes].reset_index(drop=True),
            "source_available_at": source_available,
            "source_open": source_open,
            "level_price": levels,
            "zone_half_width": widths,
            "zone_half_width_atr": widths / atr,
            "base_atr": atr,
            "approach_state": approach,
            "approach_code": approach_code,
            "pre_distance_atr": np.abs(levels - pre_close) / atr,
            "volatility_band": merged["volatility_band"].iloc[indexes].to_numpy(dtype=np.int16),
            "contact_range_band": merged["contact_range_band"]
            .iloc[indexes]
            .to_numpy(dtype=np.int16),
        }
    )
    if match_tier is None:
        output["random_match_tier"] = None
    elif isinstance(match_tier, str):
        output["random_match_tier"] = match_tier
    else:
        output["random_match_tier"] = list(match_tier)

    actual_attributes = control in {"actual", "near_miss"}
    output["level_score"] = (
        attribute_values(merged, spec.score_column, indexes, numeric=True)
        if actual_attributes
        else np.nan
    )
    output["level_identity"] = (
        attribute_values(merged, spec.identity_column, indexes, numeric=False)
        if actual_attributes
        else None
    )
    for column in spec.attribute_columns:
        if column not in merged.columns:
            continue
        values = merged[column].iloc[indexes].reset_index(drop=True)
        output[f"attr_{column}"] = values if actual_attributes else None

    append_reaction_metrics(
        output=output,
        merged=merged,
        paths=paths,
        indexes=indexes,
        levels=levels,
        widths=widths,
        atr=atr,
        approach_code=approach_code,
        horizons=horizons,
    )
    return output


def append_reaction_metrics(
    *,
    output: DataFrame,
    merged: DataFrame,
    paths: dict[str, np.ndarray],
    indexes: np.ndarray,
    levels: np.ndarray,
    widths: np.ndarray,
    atr: np.ndarray,
    approach_code: np.ndarray,
    horizons: tuple[int, ...],
) -> None:
    future_high = paths["high"][indexes]
    future_low = paths["low"][indexes]
    future_close = paths["close"][indexes]
    future_volume = paths["volume"][indexes]
    future_pressure = paths["pressure"][indexes]
    level_matrix = levels[:, None]
    atr_matrix = atr[:, None]
    high_distance = (future_high - level_matrix) / atr_matrix
    low_distance = (level_matrix - future_low) / atr_matrix
    absolute_path = np.maximum(np.abs(high_distance), np.abs(low_distance))
    through_path = np.where(
        approach_code[:, None] == 1,
        high_distance,
        np.where(approach_code[:, None] == -1, low_distance, np.nan),
    )
    away_path = np.where(
        approach_code[:, None] == 1,
        low_distance,
        np.where(approach_code[:, None] == -1, high_distance, np.nan),
    )
    through_path = np.maximum(through_path, 0.0)
    away_path = np.maximum(away_path, 0.0)

    output["time_to_abs_0_5atr"] = first_threshold_time(absolute_path, 0.5)
    output["time_to_abs_1_0atr"] = first_threshold_time(absolute_path, 1.0)
    output["time_to_away_0_5atr"] = first_threshold_time(away_path, 0.5)
    output["time_to_away_1_0atr"] = first_threshold_time(away_path, 1.0)
    output["time_to_through_0_5atr"] = first_threshold_time(through_path, 0.5)
    output["time_to_through_1_0atr"] = first_threshold_time(through_path, 1.0)

    pre_range = numeric_array(merged["pre_range_median_24"])[indexes]
    pre_volume = numeric_array(merged["pre_volume_median_24"])[indexes]
    pre_pressure = numeric_array(merged["pre_pressure_mean_24"])[indexes]
    contact_range = numeric_array(merged["high"])[indexes] - numeric_array(merged["low"])[indexes]
    contact_volume = numeric_array(merged["volume"])[indexes]
    contact_pressure = numeric_array(merged["candle_pressure"])[indexes]
    contact_close = numeric_array(merged["close"])[indexes]
    output["contact_range_ratio"] = safe_ratio(contact_range, pre_range)
    output["contact_volume_ratio"] = safe_ratio(contact_volume, pre_volume)
    output["contact_pressure_change"] = contact_pressure - pre_pressure
    output["contact_close_distance_atr"] = np.abs(contact_close - levels) / atr

    previous_sign = np.sign(numeric_array(merged["pre_close"])[indexes] - levels)
    for horizon in horizons:
        sl = slice(0, horizon)
        high_slice = future_high[:, sl]
        low_slice = future_low[:, sl]
        close_slice = future_close[:, sl]
        volume_slice = future_volume[:, sl]
        pressure_slice = future_pressure[:, sl]
        range_slice = high_slice - low_slice
        output[f"abs_excursion_atr_h{horizon}"] = np.nanmax(absolute_path[:, sl], axis=1)
        output[f"away_excursion_atr_h{horizon}"] = nanmax_or_nan(away_path[:, sl])
        output[f"through_excursion_atr_h{horizon}"] = nanmax_or_nan(through_path[:, sl])
        output[f"close_abs_displacement_atr_h{horizon}"] = np.abs(close_slice[:, -1] - levels) / atr
        output[f"range_ratio_h{horizon}"] = safe_ratio(np.nanmean(range_slice, axis=1), pre_range)
        output[f"volume_ratio_h{horizon}"] = safe_ratio(
            np.nanmean(volume_slice, axis=1), pre_volume
        )
        output[f"pressure_change_h{horizon}"] = np.nanmean(pressure_slice, axis=1) - pre_pressure
        output[f"dwell_fraction_h{horizon}"] = np.nanmean(
            np.abs(close_slice - level_matrix) <= widths[:, None], axis=1
        )
        output[f"crossings_h{horizon}"] = crossing_counts(close_slice, levels, previous_sign)


def first_threshold_time(values: np.ndarray, threshold: float) -> np.ndarray:
    reached = np.isfinite(values) & (values >= threshold)
    any_reached = reached.any(axis=1)
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    result[any_reached] = np.argmax(reached[any_reached], axis=1) + 1.0
    return result


def crossing_counts(
    close_path: np.ndarray, levels: np.ndarray, previous_sign: np.ndarray
) -> np.ndarray:
    signs = np.sign(close_path - levels[:, None])
    combined = np.column_stack([previous_sign, signs])
    left = combined[:, :-1]
    right = combined[:, 1:]
    return ((left * right) < 0.0).sum(axis=1).astype(np.int16)


def matched_random_time_events(  # noqa: C901 - explicit matching tiers are auditable
    *,
    merged: DataFrame,
    paths: dict[str, np.ndarray],
    actual: DataFrame,
    spec: LevelSpec,
    pair: str,
    timeframe: str,
    zone_method: str,
    horizons: tuple[int, ...],
) -> DataFrame:
    if actual.empty:
        return DataFrame()
    match = actual[
        [
            "base_index",
            "period",
            "approach_state",
            "pre_distance_atr",
            "zone_half_width_atr",
            "volatility_band",
            "contact_range_band",
        ]
    ].copy()
    match["distance_band"] = np.digitize(
        match["pre_distance_atr"].to_numpy(), [0.10, 0.25, 0.50, 1.0, 2.0]
    ).astype(np.int8)
    pre_close = numeric_array(merged["pre_close"])
    atr = numeric_array(merged["base_atr"])
    high = numeric_array(merged["high"])
    low = numeric_array(merged["low"])
    periods = merged["period"].astype(str).to_numpy()
    vol_band = merged["volatility_band"].to_numpy(dtype=np.int16)
    range_band = merged["contact_range_band"].to_numpy(dtype=np.int16)
    last_eligible = len(merged) - max(horizons)
    eligible_base = (
        np.isfinite(pre_close)
        & np.isfinite(atr)
        & (atr > 0.0)
        & (np.arange(len(merged)) < last_eligible)
    )
    excluded_actual = np.zeros(len(merged), dtype=bool)
    excluded_actual[actual["base_index"].to_numpy(dtype=int)] = True
    excluded_actual = (
        np.convolve(excluded_actual.astype(np.int8), np.ones(13, dtype=np.int8), mode="same") > 0
    )
    selected_indexes: list[int] = []
    selected_levels: list[float] = []
    selected_widths: list[float] = []
    selected_tiers: list[str] = []
    group_columns = [
        "period",
        "approach_state",
        "distance_band",
        "volatility_band",
        "contact_range_band",
    ]
    for key, group in match.groupby(group_columns, sort=False, dropna=False):
        period, approach, _, wanted_vol, wanted_range = key
        count = len(group)
        distance = float(group["pre_distance_atr"].median())
        width_atr = float(group["zone_half_width_atr"].median())
        if not np.isfinite(distance) or not np.isfinite(width_atr):
            continue
        if approach == "from_below":
            sign = 1.0
        elif approach == "from_above":
            sign = -1.0
        else:
            sign = 0.0
            distance = 0.0
        if sign != 0.0:
            # Preserve the declared approach side even when medians from a broad
            # matching cell put distance and zone width nearly on top of each other.
            distance = max(distance, width_atr * 1.01)
        pseudo_level = pre_close + sign * distance * atr
        pseudo_width = np.maximum(width_atr * atr, np.abs(pseudo_level) * 0.0005)
        if sign != 0.0:
            for _ in range(2):
                row_distance = np.maximum(distance * atr, pseudo_width * 1.01)
                pseudo_level = pre_close + sign * row_distance
                pseudo_width = np.maximum(width_atr * atr, np.abs(pseudo_level) * 0.0005)
        contact = (high >= pseudo_level - pseudo_width) & (low <= pseudo_level + pseudo_width)
        tier_masks = (
            (
                "exact_period_volatility_range",
                (periods == period)
                & (vol_band == int(wanted_vol))
                & (range_band == int(wanted_range)),
            ),
            (
                "relaxed_range_band",
                (periods == period) & (vol_band == int(wanted_vol)),
            ),
            ("relaxed_volatility_and_range", periods == period),
        )
        seed = stable_hash_int(
            f"{pair}|{timeframe}|{spec.column}|{spec.representation}|{zone_method}|{key}"
        )
        rng = np.random.default_rng(seed)
        chosen: list[int] = []
        chosen_tiers: list[str] = []
        blocked = np.zeros(len(merged), dtype=bool)
        for candidate_tier, band_mask in tier_masks:
            candidate = np.flatnonzero(
                eligible_base & contact & band_mask & ~excluded_actual & ~blocked
            )
            for position in rng.permutation(candidate):
                position = int(position)
                if blocked[position]:
                    continue
                chosen.append(position)
                chosen_tiers.append(candidate_tier)
                left = max(0, position - 6)
                right = min(len(blocked), position + 7)
                blocked[left:right] = True
                if len(chosen) >= count:
                    break
            if len(chosen) >= count:
                break
        if not chosen:
            continue
        chosen_array = np.asarray(chosen, dtype=int)
        selected_indexes.extend(chosen)
        selected_levels.extend(pseudo_level[chosen_array].tolist())
        selected_widths.extend(pseudo_width[chosen_array].tolist())
        selected_tiers.extend(chosen_tiers)

    if not selected_indexes:
        return DataFrame()
    order = np.argsort(np.asarray(selected_indexes))
    indexes = np.asarray(selected_indexes, dtype=int)[order]
    levels = np.asarray(selected_levels, dtype=float)[order]
    widths = np.asarray(selected_widths, dtype=float)[order]
    tiers = np.asarray(selected_tiers, dtype=object)[order].tolist()
    nat = pd.Series(pd.NaT, index=range(len(indexes)), dtype="datetime64[ns, UTC]")
    return event_frame(
        merged=merged,
        paths=paths,
        indexes=indexes,
        levels=levels,
        widths=widths,
        spec=spec,
        pair=pair,
        timeframe=timeframe,
        control="matched_random_time",
        zone_method=zone_method,
        source_available=nat.copy(),
        source_open=nat.copy(),
        horizons=horizons,
        match_tier=tiers,
    )


def summarize_atlas_events(events: DataFrame, manifest: dict[str, Any]) -> DataFrame:
    if events.empty:
        return DataFrame()
    frames = [events.copy()]
    pooled_period = events.copy()
    pooled_period["period"] = "all_periods_descriptive"
    frames.append(pooled_period)
    with_all_approaches = pd.concat(frames, ignore_index=True)
    pooled_approach = with_all_approaches.copy()
    pooled_approach["approach_state"] = "all_approaches"
    working = pd.concat([with_all_approaches, pooled_approach], ignore_index=True)
    group_columns = [
        "pair",
        "source_timeframe",
        "batch",
        "level_family",
        "level_name",
        "representation",
        "control",
        "zone_method",
        "period",
        "approach_state",
    ]
    working["_event_day"] = normalize_dates(working["event_time"]).dt.floor("1D")
    grouped = working.groupby(group_columns, sort=False, dropna=False, observed=True)
    counts = grouped.size().rename("event_count")
    per_day = working.groupby(
        [*group_columns, "_event_day"], sort=False, dropna=False, observed=True
    ).size()
    day_stats = per_day.groupby(
        level=list(range(len(group_columns))), sort=False, dropna=False, observed=True
    ).agg(["size", "max"])
    day_stats.rename(columns={"size": "unique_event_days"}, inplace=True)
    day_stats["largest_day_event_fraction"] = day_stats["max"].div(counts)
    day_stats.drop(columns=["max"], inplace=True)

    contact_columns = (
        "contact_range_ratio",
        "contact_volume_ratio",
        "contact_pressure_change",
        "contact_close_distance_atr",
        "zone_half_width_atr",
        "pre_distance_atr",
    )
    contact = grouped[list(contact_columns)].median()
    contact.rename(
        columns={column: f"{column}_median" for column in contact_columns},
        inplace=True,
    )
    frames_by_horizon: list[DataFrame] = []
    for horizon in manifest["reaction_definition"]["horizons_hours"]:
        metric_columns = {
            "abs_excursion": f"abs_excursion_atr_h{horizon}",
            "away_excursion": f"away_excursion_atr_h{horizon}",
            "through_excursion": f"through_excursion_atr_h{horizon}",
            "close_abs_displacement": f"close_abs_displacement_atr_h{horizon}",
            "range_ratio": f"range_ratio_h{horizon}",
            "volume_ratio": f"volume_ratio_h{horizon}",
            "pressure_change": f"pressure_change_h{horizon}",
            "dwell_fraction": f"dwell_fraction_h{horizon}",
            "crossings": f"crossings_h{horizon}",
        }
        selected_columns = list(metric_columns.values())
        mean = (
            grouped[selected_columns]
            .mean()
            .rename(columns={column: f"{short}_mean" for short, column in metric_columns.items()})
        )
        median = (
            grouped[selected_columns]
            .median()
            .rename(columns={column: f"{short}_median" for short, column in metric_columns.items()})
        )
        q25 = (
            grouped[selected_columns]
            .quantile(0.25)
            .rename(columns={column: f"{short}_q25" for short, column in metric_columns.items()})
        )
        q75 = (
            grouped[selected_columns]
            .quantile(0.75)
            .rename(columns={column: f"{short}_q75" for short, column in metric_columns.items()})
        )
        horizon_frame = pd.concat(
            [counts, day_stats, mean, median, q25, q75, contact], axis=1
        ).reset_index()
        horizon_frame["horizon_hours"] = int(horizon)
        frames_by_horizon.append(horizon_frame)
    return pd.concat(frames_by_horizon, ignore_index=True)


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(len(numerator), np.nan, dtype=np.float64)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
    result[valid] = numerator[valid] / denominator[valid]
    return result


def nanmax_or_nan(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    valid = np.isfinite(values).any(axis=1)
    if valid.any():
        result[valid] = np.nanmax(values[valid], axis=1)
    return result


def numeric_array(value: Series | np.ndarray) -> np.ndarray:
    if isinstance(value, Series):
        return pd.to_numeric(value, errors="coerce").to_numpy(dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def bool_array(series: Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).to_numpy(dtype=bool)
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.fillna(0.0).to_numpy(dtype=float) > 0.0


def attribute_values(
    frame: DataFrame,
    column: str | None,
    indexes: np.ndarray,
    *,
    numeric: bool,
) -> Any:
    if column is None or column not in frame.columns:
        return np.full(len(indexes), np.nan) if numeric else [None] * len(indexes)
    values = frame[column].iloc[indexes].reset_index(drop=True)
    if numeric:
        return pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    return values.astype("string").where(values.notna(), None).tolist()


def stable_hash_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def spec_to_dict(spec: LevelSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "family": spec.family,
        "batch": spec.batch,
        "column": spec.column,
        "representation": spec.representation,
        "slope_column": spec.slope_column,
        "native_width_column": spec.native_width_column,
        "native_width_atr_column": spec.native_width_atr_column,
        "active_columns": list(spec.active_columns),
        "score_column": spec.score_column,
        "identity_column": spec.identity_column,
        "attribute_columns": list(spec.attribute_columns),
    }


def atlas_event_path(task: AtlasTask, *, event_dir: Path = EVENT_DIR) -> Path:
    return event_dir / f"{atlas_artifact_stem(task)}.parquet"


def atlas_summary_path(
    task: AtlasTask, *, detailed_summary_dir: Path = DETAILED_SUMMARY_DIR
) -> Path:
    return detailed_summary_dir / f"{atlas_artifact_stem(task)}-summary.parquet"


def atlas_metadata_path(task: AtlasTask, *, report_dir: Path = REPORT_DIR) -> Path:
    return report_dir / f"{atlas_artifact_stem(task)}.meta.json"


def atlas_artifact_stem(task: AtlasTask) -> str:
    batch_tag = "-".join(task.level_batches)
    zone_tag = hashlib.sha256("|".join(task.zone_methods).encode()).hexdigest()[:8]
    control_tag = hashlib.sha256("|".join(task.controls).encode()).hexdigest()[:8]
    family_tag = "-".join(sorted(task.cache_families))
    return (
        f"{pair_file_stem(task.pair)}-{task.timeframe}-{family_tag}-{batch_tag}-"
        f"z{zone_tag}-c{control_tag}"
    )


def write_cache_inventory(
    pairs: Sequence[str],
    timeframes: Sequence[str],
    *,
    manifest: dict[str, Any] | None = None,
) -> DataFrame:
    storage = (
        manifest_storage_paths(manifest)
        if manifest is not None
        else StoragePaths(
            cache_dir=CACHE_DIR,
            report_dir=REPORT_DIR,
            event_dir=EVENT_DIR,
            detailed_summary_dir=DETAILED_SUMMARY_DIR,
        )
    )
    storage.report_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        for timeframe in timeframes:
            candidates = sorted(
                storage.cache_dir.glob(f"{pair_file_stem(pair)}-{timeframe}-*.parquet")
            )
            for path in candidates:
                frame = pd.read_parquet(path)
                fixed = {
                    "source_open",
                    "available_at",
                    "source_bar_index",
                    "source_open_price",
                    "source_high",
                    "source_low",
                    "source_close",
                    "source_volume",
                    "source_atr_14",
                }
                for column in candidate_level_columns(frame):
                    if column in fixed:
                        continue
                    values = pd.to_numeric(frame[column], errors="coerce")
                    finite = np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
                    rows.append(
                        {
                            "pair": pair,
                            "timeframe": timeframe,
                            "cache_file": str(path),
                            "level_column": column,
                            "rows": len(frame),
                            "finite_rows": int(finite.sum()),
                            "availability_fraction": float(finite.mean()) if len(finite) else 0.0,
                            "first_available_utc": first_finite_timestamp(
                                frame["available_at"], finite
                            ),
                            "last_available_utc": last_finite_timestamp(
                                frame["available_at"], finite
                            ),
                        }
                    )
    inventory = pd.DataFrame(rows)
    atomic_write_csv(inventory, storage.report_dir / "g0a_level_availability.csv")
    return inventory


def existing_columns(frame: DataFrame, columns: Iterable[str]) -> DataFrame:
    selected = [column for column in columns if column in frame.columns]
    return frame[selected].copy()


def is_level_like(column: str) -> bool:
    name = column.lower()
    return any(
        token in name
        for token in (
            "level",
            "poc",
            "vah",
            "val",
            "hvn",
            "lvn",
            "line",
            "zone",
            "upper",
            "lower",
            "swing",
            "pivot",
            "vwap",
            "ema_",
            "sma_",
            "rolling_high",
            "rolling_low",
            "round_",
        )
    )


def candidate_level_columns(frame: DataFrame) -> list[str]:
    """Return price-valued candidate columns, excluding scores, ids and counts."""
    exact = {
        "vp_poc",
        "vp_vah",
        "vp_val",
        "vp_prior_poc",
        "vp_prior_vah",
        "vp_prior_val",
        "vp_hvn_above",
        "vp_hvn_below",
        "vp_lvn_above",
        "vp_lvn_below",
        "bc_prev_swing_high",
        "bc_prev_swing_low",
        "bc_last_swing_high",
        "bc_last_swing_low",
        "bc_break_level",
        "bc_invalidation_level",
        "bc_bullish_break_level",
        "bc_bearish_break_level",
    }
    patterns = (
        re.compile(r"^tlv2_(?:support|resistance)_line_rank[0-2]$"),
        re.compile(r"^tlv2_forecast_(?:support|resistance)_zone_(?:center|lower|upper)$"),
        re.compile(r"^pg2_slot_[1-4]_(?:upper|lower)$"),
        re.compile(
            r"^generic_(?:rolling_(?:high|low)_\d+|(?:ema|sma|vwap)_\d+|"
            r"bb20_(?:upper|mid|lower)|round_(?:below|nearest|above))$"
        ),
        re.compile(r"^(?:pr|pp|pw)_.+_(?:confirmation|target)_level$"),
        re.compile(r"^pc_(?:flag|pennant)_(?:upper|lower|confirmation_level|invalidation_level)$"),
    )
    return [
        column
        for column in frame.columns
        if column in exact or any(pattern.match(column) for pattern in patterns)
    ]


def wilder_atr(frame: DataFrame, period: int) -> Series:
    high = numeric_series(frame["high"])
    low = numeric_series(frame["low"])
    close = numeric_series(frame["close"])
    true_range = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def numeric_series(series: Series) -> Series:
    return pd.to_numeric(series, errors="coerce").astype("float64")


def load_ohlcv(path: Path) -> DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"OHLCV file not found: {path}")
    frame = pd.read_feather(path)
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing OHLCV columns: {missing}")
    frame = frame[["date", "open", "high", "low", "close", "volume"]].copy()
    frame["date"] = normalize_dates(frame["date"])
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = numeric_series(frame[column])
    return frame


def normalize_dates(series: Series) -> Series:
    dates = pd.to_datetime(series, utc=True, errors="raise")
    return dates.dt.as_unit("ns")


def ohlcv_path(pair: str, timeframe: str) -> Path:
    return DATA_DIR / f"{pair_file_stem(pair)}-{timeframe}-futures.feather"


def pair_file_stem(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")


def level_cache_path(
    pair: str,
    timeframe: str,
    families: Sequence[str],
    *,
    cache_dir: Path = CACHE_DIR,
) -> Path:
    family_tag = "-".join(sorted(families))
    return cache_dir / f"{pair_file_stem(pair)}-{timeframe}-{family_tag}.parquet"


def timeframe_hours(timeframe: str) -> int:
    unit = timeframe[-1].lower()
    value = int(timeframe[:-1])
    if unit == "h":
        return value
    if unit == "d":
        return value * 24
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def timeframe_delta(timeframe: str) -> pd.Timedelta:
    return pd.Timedelta(hours=timeframe_hours(timeframe))


def system_snapshot(new_worker_cap: int = MAX_NEW_WORKERS) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "logical_processors": os.cpu_count(),
        "new_worker_cap": new_worker_cap,
    }
    try:
        import psutil

        snapshot["cpu_percent_half_second"] = psutil.cpu_percent(interval=0.5)
        snapshot["available_memory_gib"] = round(psutil.virtual_memory().available / (1024**3), 3)
        project_processes = 0
        project_threads = 0
        for process in psutil.process_iter(["cmdline", "num_threads"]):
            try:
                command = " ".join(process.info.get("cmdline") or [])
                if str(REPO_ROOT).lower() in command.lower():
                    project_processes += 1
                    project_threads += int(process.info.get("num_threads") or 0)
            except (psutil.AccessDenied, psutil.NoSuchProcess, TypeError):
                continue
        snapshot["visible_project_processes"] = project_processes
        snapshot["visible_project_process_threads"] = project_threads
    except ImportError:
        snapshot["load_note"] = "psutil is not installed; caller must inspect live load."
    return snapshot


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=json_default)
        handle.write("\n")
    temporary.replace(path)


def atomic_write_csv(frame: DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_write_parquet(frame: DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return iso_timestamp(pd.Timestamp(value))
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def iso_timestamp(value: pd.Timestamp) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.isoformat().replace("+00:00", "Z")


def first_finite_timestamp(dates: Series, finite: np.ndarray) -> str | None:
    positions = np.flatnonzero(finite)
    return iso_timestamp(pd.Timestamp(dates.iloc[positions[0]])) if len(positions) else None


def last_finite_timestamp(dates: Series, finite: np.ndarray) -> str | None:
    positions = np.flatnonzero(finite)
    return iso_timestamp(pd.Timestamp(dates.iloc[positions[-1]])) if len(positions) else None


if __name__ == "__main__":
    raise SystemExit(main())

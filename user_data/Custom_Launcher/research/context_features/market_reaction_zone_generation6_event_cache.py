"""Build the shared causal event dataset for supported Generation 6 siblings.

Every later sibling analysis reads the same event clock, level controls, causal OHLCV
state, cross-market state, GDELT/topic state, and BTC orderbook state. Unsupported
preflight cells are not allowed to open reaction outcomes here.
"""

from __future__ import annotations

# Bind native numerical pools before importing pandas and the research modules.
# ruff: noqa: E402
import argparse
import json
import os
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation5_external_context as g5e,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_external_context as g3h,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation5_external_context_preflight as g5e_preflight,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation6_preflight as g6,
)


USER_DATA_DIR = REPO_ROOT / "user_data"
OUTPUT_ROOT = USER_DATA_DIR / "research_news_data" / "context_features" / "market_reaction_zones"
PRELIGHT_RUN_ID = "g6_outcome_blind_preflight_20260821a"
PREFLIGHT_DIR = OUTPUT_ROOT / "generation6_branches" / PRELIGHT_RUN_ID
PREFLIGHT_RECORD = PREFLIGHT_DIR / "g6_preflight_record.json"
SUPPORTED_LEVEL_CELLS = PREFLIGHT_DIR / "g6_level_supported_cells.csv"
FROZEN_BATCH = OUTPUT_ROOT / "generation5_review" / "g6_frozen_branch_batch.json"
NORMAL_MANIFEST = OUTPUT_ROOT / "generation6_shared" / "g6_normal_source_manifest.json"
MEME_MANIFEST = OUTPUT_ROOT / "generation6_shared" / "g6_meme_source_manifest.json"
RECORD_ROOT = OUTPUT_ROOT / "generation6_branches" / "g6_common_event_cache"
LARGE_ROOT = Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
ARTIFACT_ROOT = LARGE_ROOT / "generation6_branches" / "g6_common_event_cache"
MAX_WORKERS = 4
HORIZONS = (1, 2, 4, 8)
ZONE_METHODS = ("standard_base_atr",)
CONTROLS = ("actual", "stale_72h", "price_shift", "near_miss", "matched_random_time")

TOPIC_GROUPS: dict[str, tuple[str, ...]] = {
    "geopolitics_and_energy": (
        "war_geopolitics_count_1h",
        "sanctions_trade_count_1h",
        "oil_energy_count_1h",
    ),
    "macro_policy_and_growth": (
        "rates_count_1h",
        "inflation_count_1h",
        "central_bank_count_1h",
        "official_data_release_count_1h",
        "recession_growth_count_1h",
        "jobs_labor_count_1h",
    ),
    "crypto_policy_and_institutions": (
        "regulation_legal_count_1h",
        "etf_institutional_count_1h",
        "crypto_native_count_1h",
    ),
    "crypto_liquidity_and_security": (
        "liquidity_stablecoin_count_1h",
        "security_exploit_count_1h",
        "exchange_listing_delisting_count_1h",
    ),
}


@dataclass(frozen=True)
class PairTask:
    cohort: str
    pair: str
    manifest_path: str
    supported_cells_path: str
    cross_market_path: str
    context_path: str
    orderbook_path: str
    event_dir: str
    summary_dir: str
    overwrite: bool


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build controlled reaction events for every supported Generation 6 level "
            "cell and attach causal sibling-source features."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1 or args.workers > MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    preflight = validate_contracts()
    record_dir = RECORD_ROOT / args.run_id
    artifact_dir = ARTIFACT_ROOT / args.run_id
    manifest_path = record_dir / "manifest.json"
    if manifest_path.is_file() and not args.overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(json.dumps(existing["summary"], indent=2, sort_keys=True))
        return 0
    record_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = artifact_dir / "shared_features"
    event_dir = artifact_dir / "pair_events"
    summary_dir = record_dir / "pair_summaries"
    shared_dir.mkdir(parents=True, exist_ok=True)
    event_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    shared = build_shared_feature_caches(shared_dir, overwrite=args.overwrite)
    tasks = build_tasks(
        pairs=args.pairs,
        shared=shared,
        event_dir=event_dir,
        summary_dir=summary_dir,
        overwrite=args.overwrite,
    )
    results = run_tasks(tasks, workers=args.workers)
    failures = [item for item in results if item.get("status") == "failed"]
    if failures:
        raise RuntimeError(f"Generation 6 event-cache tasks failed: {failures}")

    inventory = DataFrame.from_records(results)
    inventory_path = record_dir / "pair_event_inventory.csv"
    g0.atomic_write_csv(inventory, inventory_path)
    combined_summary = combine_summaries(results)
    combined_summary_path = record_dir / "g6_supported_level_control_summary.parquet"
    g0.atomic_write_parquet(combined_summary, combined_summary_path)
    control_counts = inventory.filter(regex=r"^control_rows__").sum(numeric_only=True).to_dict()
    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "completed_shared_causal_event_cache",
        "created_at_utc": g0.utc_now(),
        "objective": (
            "Create one controlled causal event dataset for the seven supported "
            "Generation 6 sibling analyses."
        ),
        "preflight": artifact_record(PREFLIGHT_RECORD),
        "frozen_batch": artifact_record(FROZEN_BATCH),
        "supported_level_cells": artifact_record(SUPPORTED_LEVEL_CELLS),
        "source_manifests": {
            "normal": artifact_record(NORMAL_MANIFEST),
            "meme": artifact_record(MEME_MANIFEST),
        },
        "shared_features": shared,
        "controls": list(CONTROLS),
        "zone_methods": list(ZONE_METHODS),
        "horizons_hours": list(HORIZONS),
        "information_clock": {
            "ohlcv_and_indicators": "previous completed base candle",
            "higher_timeframe_levels": "source candle available_at not after event time",
            "cross_market": "previous completed one-hour candle",
            "gdelt_and_topics": "source availability not after feature hour",
            "orderbook": "source_max_ts not after feature hour",
        },
        "research_boundary": {
            "profit_used": False,
            "direction_prediction": False,
            "unsupported_preflight_cells_opened": False,
            "broad_feature_soup_scored": False,
        },
        "pair_inventory": artifact_record(inventory_path),
        "combined_level_control_summary": artifact_record(combined_summary_path),
        "tasks": results,
        "summary": {
            "pair_tasks": len(results),
            "built_tasks": sum(item["status"] == "built" for item in results),
            "existing_tasks": sum(item["status"] == "existing" for item in results),
            "event_rows": int(inventory["event_rows"].sum()),
            "actual_event_rows": int(inventory["actual_event_rows"].sum()),
            "control_rows": control_counts,
            "supported_specs_opened": int(inventory["supported_specs"].sum()),
            "causal_timestamp_violations": int(inventory["causal_timestamp_violations"].sum()),
            "profit_used": False,
            "direction_prediction": False,
            "preflight_reaction_outcomes_opened": bool(
                preflight["summary"]["reaction_outcomes_opened"]
            ),
        },
    }
    if manifest["summary"]["causal_timestamp_violations"]:
        raise AssertionError("A future source row entered the shared Generation 6 cache.")
    g0.atomic_write_json(manifest, manifest_path)
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
    return 0


def validate_contracts() -> dict[str, Any]:
    for path in (
        PREFLIGHT_RECORD,
        SUPPORTED_LEVEL_CELLS,
        FROZEN_BATCH,
        NORMAL_MANIFEST,
        MEME_MANIFEST,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    preflight = json.loads(PREFLIGHT_RECORD.read_text(encoding="utf-8"))
    if preflight.get("status") != "completed_outcome_blind":
        raise ValueError("Generation 6 outcome-blind preflight is not terminal.")
    summary = preflight.get("summary", {})
    if summary.get("reaction_outcomes_opened") is not False:
        raise ValueError("The preflight unexpectedly opened reaction outcomes.")
    if summary.get("sibling_branches_supported_or_partly_supported") != 7:
        raise ValueError("All seven sibling preflights must be terminal before outcomes.")
    frozen = json.loads(FROZEN_BATCH.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation6_reaction_outcomes":
        raise ValueError("The Generation 6 batch is not frozen.")
    return preflight


def build_shared_feature_caches(shared_dir: Path, *, overwrite: bool) -> dict[str, dict[str, Any]]:
    paths = {
        "normal_cross_market": shared_dir / "normal_cross_market.parquet",
        "meme_cross_market": shared_dir / "meme_cross_market.parquet",
        "gdelt_and_topics": shared_dir / "gdelt_and_topics.parquet",
        "btc_orderbook": shared_dir / "btc_orderbook.parquet",
    }
    if overwrite or not paths["normal_cross_market"].is_file():
        g0.atomic_write_parquet(
            cross_market_frame(g0.load_manifest(NORMAL_MANIFEST)),
            paths["normal_cross_market"],
        )
    if overwrite or not paths["meme_cross_market"].is_file():
        g0.atomic_write_parquet(
            cross_market_frame(g0.load_manifest(MEME_MANIFEST)),
            paths["meme_cross_market"],
        )
    if overwrite or not paths["gdelt_and_topics"].is_file():
        g0.atomic_write_parquet(context_feature_frame(), paths["gdelt_and_topics"])
    if overwrite or not paths["btc_orderbook"].is_file():
        g0.atomic_write_parquet(orderbook_feature_frame(), paths["btc_orderbook"])
    return {key: artifact_record(path) for key, path in paths.items()}


def cross_market_frame(manifest: dict[str, Any]) -> DataFrame:
    pairs = tuple(str(pair) for pair in manifest["data"]["pairs"])
    all_pairs = tuple(dict.fromkeys((*pairs, "BTC/USDT:USDT", "ETH/USDT:USDT")))
    close_frames: list[DataFrame] = []
    volume_frames: list[DataFrame] = []
    for pair in all_pairs:
        frame = g0.load_ohlcv(g0.ohlcv_path(pair, "1h"))[["date", "close", "volume"]]
        frame = frame.drop_duplicates("date").set_index("date")
        close_frames.append(frame[["close"]].rename(columns={"close": pair}))
        volume_frames.append(frame[["volume"]].rename(columns={"volume": pair}))
    close = pd.concat(close_frames, axis=1, sort=False).sort_index()
    volume = pd.concat(volume_frames, axis=1, sort=False).sort_index()
    returns_1 = close.pct_change(fill_method=None).shift(1)
    returns_4 = close.pct_change(4, fill_method=None).shift(1)
    returns_24 = close.pct_change(24, fill_method=None).shift(1)
    previous_volume = volume.shift(1)
    relative_volume = previous_volume.div(previous_volume.rolling(24, min_periods=24).median())
    cohort_returns = returns_1.loc[:, list(pairs)]
    output = DataFrame({"date": close.index})
    output["xm_btc_return_1h"] = returns_1["BTC/USDT:USDT"].to_numpy()
    output["xm_btc_return_4h"] = returns_4["BTC/USDT:USDT"].to_numpy()
    output["xm_btc_return_24h"] = returns_24["BTC/USDT:USDT"].to_numpy()
    output["xm_btc_relative_volume"] = relative_volume["BTC/USDT:USDT"].to_numpy()
    output["xm_eth_return_1h"] = returns_1["ETH/USDT:USDT"].to_numpy()
    output["xm_eth_return_4h"] = returns_4["ETH/USDT:USDT"].to_numpy()
    output["xm_eth_return_24h"] = returns_24["ETH/USDT:USDT"].to_numpy()
    output["xm_cohort_breadth_positive"] = cohort_returns.gt(0.0).mean(axis=1).to_numpy()
    output["xm_cohort_dispersion"] = cohort_returns.std(axis=1).to_numpy()
    output["xm_cohort_absolute_activity"] = cohort_returns.abs().mean(axis=1).to_numpy()
    output["xm_cohort_ready_members"] = cohort_returns.notna().sum(axis=1).to_numpy()
    btc = returns_1["BTC/USDT:USDT"]
    btc_variance = btc.rolling(168, min_periods=168).var()
    for pair in pairs:
        stem = g0.pair_file_stem(pair).lower()
        pair_return = returns_1[pair]
        output[f"xm_pair_minus_btc_1h__{stem}"] = (pair_return - btc).to_numpy()
        output[f"xm_pair_minus_btc_24h__{stem}"] = (
            returns_24[pair] - returns_24["BTC/USDT:USDT"]
        ).to_numpy()
        output[f"xm_pair_btc_corr_168h__{stem}"] = (
            pair_return.rolling(168, min_periods=168).corr(btc).to_numpy()
        )
        covariance = pair_return.rolling(168, min_periods=168).cov(btc)
        output[f"xm_pair_btc_beta_168h__{stem}"] = covariance.div(
            btc_variance.replace(0.0, np.nan)
        ).to_numpy()
    return output.replace([np.inf, -np.inf], np.nan)


def context_feature_frame() -> DataFrame:
    sources, _ = g5e.load_context_sources()
    gdelt = sources["gdelt_aggregate"].copy()
    output = DataFrame({"date": pd.to_datetime(gdelt["date"], utc=True)})
    output["news_gdelt_ready"] = gdelt["ready"].fillna(False).astype(bool)
    output["news_gdelt_regime"] = gdelt["regime"].astype("string")
    output["news_gdelt_activity"] = pd.to_numeric(gdelt["activity_score"], errors="coerce")
    for column in gdelt.columns:
        if column.startswith("value__"):
            output[f"news_gdelt__{column.removeprefix('value__')}"] = pd.to_numeric(
                gdelt[column], errors="coerce"
            )

    context, _, masks = g5e_preflight.load_context_masks(g6.CONTEXT_SNAPSHOT)
    topic_ready = (
        masks.get("context_topic_severity", {})
        .get("usable", Series(False, index=context.index))
        .reindex(context.index)
        .fillna(False)
        .astype(bool)
    )
    topics = DataFrame({"date": pd.to_datetime(context["date"], utc=True)})
    topics["news_topics_ready"] = topic_ready
    for group, raw_columns in TOPIC_GROUPS.items():
        columns = [f"ctx_{column}" for column in raw_columns]
        present = [column for column in columns if column in context]
        if not present:
            topics[f"news_topic__{group}"] = np.nan
            continue
        values = context[present].apply(pd.to_numeric, errors="coerce").clip(lower=0.0)
        topics[f"news_topic__{group}"] = np.log1p(values).sum(axis=1).where(topic_ready)
    output = output.merge(topics, on="date", how="outer", validate="one_to_one")
    return output.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def orderbook_feature_frame() -> DataFrame:
    frame = g3h.load_causal_pressure_surface(g6.HISTORICAL_ORDERBOOK)
    output = DataFrame({"date": pd.to_datetime(frame["date"], utc=True)})
    output["ob_btc_source_ready"] = frame["source_row_usable"].fillna(False).astype(bool)
    output["ob_btc_model_ready"] = frame["pressure_regime"].ne("unavailable")
    output["ob_btc_regime"] = frame["pressure_regime"].astype("string")
    for source, target in (
        ("obts_coverage_ratio", "ob_btc_coverage_ratio"),
        ("obts_pressure_25bps_mean", "ob_btc_pressure_25bps"),
        ("absolute_pressure", "ob_btc_absolute_pressure"),
        ("pressure_lower_tertile", "ob_btc_pressure_lower_tertile"),
        ("pressure_upper_tertile", "ob_btc_pressure_upper_tertile"),
    ):
        output[target] = pd.to_numeric(frame[source], errors="coerce")
    return output


def build_tasks(
    *,
    pairs: str,
    shared: dict[str, dict[str, Any]],
    event_dir: Path,
    summary_dir: Path,
    overwrite: bool,
) -> list[PairTask]:
    requested = None if pairs.strip().lower() == "all" else set(g0.split_csv(pairs))
    tasks: list[PairTask] = []
    for cohort, manifest_path, cross_key in (
        ("normal", NORMAL_MANIFEST, "normal_cross_market"),
        ("meme", MEME_MANIFEST, "meme_cross_market"),
    ):
        manifest = g0.load_manifest(manifest_path)
        for pair in manifest["data"]["pairs"]:
            if (
                requested is not None
                and pair not in requested
                and g0.pair_file_stem(pair) not in requested
            ):
                continue
            tasks.append(
                PairTask(
                    cohort=cohort,
                    pair=str(pair),
                    manifest_path=str(manifest_path),
                    supported_cells_path=str(SUPPORTED_LEVEL_CELLS),
                    cross_market_path=shared[cross_key]["path"],
                    context_path=shared["gdelt_and_topics"]["path"],
                    orderbook_path=shared["btc_orderbook"]["path"],
                    event_dir=str(event_dir),
                    summary_dir=str(summary_dir),
                    overwrite=overwrite,
                )
            )
    return tasks


def run_tasks(tasks: Sequence[PairTask], *, workers: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(build_pair_events, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    {
                        "status": "failed",
                        "cohort": task.cohort,
                        "pair": task.pair,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return sorted(results, key=lambda item: (item.get("cohort", ""), item.get("pair", "")))


def build_pair_events(task: PairTask) -> dict[str, Any]:
    manifest_path = Path(task.manifest_path)
    manifest = g0.load_manifest(manifest_path)
    storage = g0.manifest_storage_paths(manifest)
    stem = f"{task.cohort}__{g0.pair_file_stem(task.pair)}"
    event_path = Path(task.event_dir) / f"{stem}.parquet"
    summary_path = Path(task.summary_dir) / f"{stem}.parquet"
    if event_path.is_file() and summary_path.is_file() and not task.overwrite:
        existing = pd.read_parquet(event_path, columns=["control"])
        return inventory_record(
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
        & supported["supported"].astype(str).str.lower().eq("true")
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
    end = pd.Timestamp(manifest["data"]["analysis_end_utc_exclusive"])
    base = base.loc[base["date"] < end].reset_index(drop=True)
    paths = g0.future_path_matrices(base, max(HORIZONS))
    state_features, _ = g6.causal_state_features(base)
    state_features = state_features.add_prefix("state__")
    enrichment = build_enrichment(task, base, state_features)

    merged_by_timeframe: dict[str, DataFrame] = {}
    specs_by_timeframe: dict[str, list[g0.LevelSpec]] = {}
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
        cache = pd.read_parquet(cache_path).sort_values("available_at").reset_index(drop=True)
        cache["available_at"] = g0.normalize_dates(cache["available_at"])
        cache["source_open"] = g0.normalize_dates(cache["source_open"])
        merged = pd.merge_asof(
            base.sort_values("date"),
            cache,
            left_on="date",
            right_on="available_at",
            direction="backward",
            allow_exact_matches=True,
        )
        present = merged["available_at"].notna()
        causal_violations += int(
            (merged.loc[present, "available_at"] > merged.loc[present, "date"]).sum()
        )
        selected = g6.selected_level_specs(cache)
        merged_by_timeframe[timeframe] = merged
        specs_by_timeframe[timeframe] = selected
        collect_contact_masks(
            merged,
            selected,
            timeframe=timeframe,
            mechanism_contacts=mechanism_contacts,
            side_contacts=side_contacts,
        )
    if causal_violations:
        raise AssertionError(f"Future level rows for {task.pair}: {causal_violations}")
    relationship_masks = g6.timeframe_relationship_masks(
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
                zone_methods=ZONE_METHODS,
                controls=CONTROLS,
                horizons=HORIZONS,
            )
            if not frames:
                continue
            opened_specs += 1
            events = pd.concat(frames, ignore_index=True)
            events.insert(0, "cohort", task.cohort)
            events["mechanism_group"] = g6.mechanism_group(spec)
            events["level_side"] = g6.level_side(spec)
            attach_relationship_flags(events, relationship_masks)
            events = attach_enrichment(events, enrichment)
            event_frames.append(events)

    events = pd.concat(event_frames, ignore_index=True) if event_frames else DataFrame()
    if not events.empty:
        events.sort_values(
            ["event_time", "source_timeframe", "level_family", "level_name", "control"],
            inplace=True,
            ignore_index=True,
        )
    summary = g0.summarize_atlas_events(events, manifest) if not events.empty else DataFrame()
    if not summary.empty:
        summary.insert(0, "cohort", task.cohort)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(events, event_path)
    g0.atomic_write_parquet(summary, summary_path)
    return inventory_record(
        task,
        status="built",
        event_path=event_path,
        summary_path=summary_path,
        events=events,
        supported_specs=opened_specs,
        causal_violations=causal_violations,
    )


def collect_contact_masks(
    merged: DataFrame,
    specs: Sequence[g0.LevelSpec],
    *,
    timeframe: str,
    mechanism_contacts: dict[tuple[str, str], np.ndarray],
    side_contacts: dict[tuple[str, str], np.ndarray],
) -> None:
    high = g0.numeric_array(merged["high"])
    low = g0.numeric_array(merged["low"])
    base_atr = g0.numeric_array(merged["base_atr"])
    for spec in specs:
        level = g0.resolved_level_values(merged, spec, timeframe)
        valid = np.isfinite(level) & (level > 0.0) & np.isfinite(base_atr) & (base_atr > 0.0)
        if spec.active_columns:
            active = np.zeros(len(merged), dtype=bool)
            for column in spec.active_columns:
                if column in merged:
                    active |= g0.bool_array(merged[column])
            valid &= active
        width = g0.zone_half_width(
            "standard_base_atr", level, base_atr, g0.resolved_native_width(merged, spec)
        )
        contact = (
            valid
            & np.isfinite(width)
            & (width > 0.0)
            & (high >= level - width)
            & (low <= level + width)
        )
        mechanism = g6.mechanism_group(spec)
        side = g6.level_side(spec)
        mechanism_contacts.setdefault((mechanism, timeframe), np.zeros(len(merged), dtype=bool))
        mechanism_contacts[(mechanism, timeframe)] |= contact
        side_contacts.setdefault((side, timeframe), np.zeros(len(merged), dtype=bool))
        side_contacts[(side, timeframe)] |= contact


def build_enrichment(task: PairTask, base: DataFrame, state: DataFrame) -> DataFrame:
    output = DataFrame({"date": pd.to_datetime(base["date"], utc=True)})
    output = pd.concat([output, state.reset_index(drop=True)], axis=1)
    cross = pd.read_parquet(task.cross_market_path)
    cross["date"] = pd.to_datetime(cross["date"], utc=True)
    stem = g0.pair_file_stem(task.pair).lower()
    shared_cross_columns = [
        column
        for column in cross
        if column == "date"
        or (
            not column.endswith(
                tuple(f"__{g0.pair_file_stem(pair).lower()}" for pair in all_manifest_pairs())
            )
        )
    ]
    pair_columns = [column for column in cross if column.endswith(f"__{stem}")]
    cross = cross[[*dict.fromkeys([*shared_cross_columns, *pair_columns])]]
    cross = cross.rename(
        columns={column: column.replace(f"__{stem}", "") for column in pair_columns}
    )
    output = output.merge(cross, on="date", how="left", validate="one_to_one")

    context = pd.read_parquet(task.context_path)
    context["date"] = pd.to_datetime(context["date"], utc=True)
    output = output.merge(context, on="date", how="left", validate="one_to_one")
    orderbook = pd.read_parquet(task.orderbook_path)
    orderbook["date"] = pd.to_datetime(orderbook["date"], utc=True)
    output = output.merge(orderbook, on="date", how="left", validate="one_to_one")
    output["ob_pair_local"] = task.pair == "BTC/USDT:USDT"
    output["market_group"] = market_group(task.cohort, task.pair)
    output["smart_contract_platform"] = task.pair in g6.GROUPS["smart_contract_platforms"]
    return output


def all_manifest_pairs() -> tuple[str, ...]:
    normal = g0.load_manifest(NORMAL_MANIFEST)["data"]["pairs"]
    meme = g0.load_manifest(MEME_MANIFEST)["data"]["pairs"]
    return tuple(dict.fromkeys([*normal, *meme]))


def market_group(cohort: str, pair: str) -> str:
    if pair == "BTC/USDT:USDT":
        return "btc_separate"
    if cohort == "meme":
        return "frozen_top_ten_memes"
    if pair == "DOGE/USDT:USDT":
        return "doge_bridge"
    return "established_altcoins"


def attach_relationship_flags(
    events: DataFrame, relationship_masks: dict[tuple[str, str], np.ndarray]
) -> None:
    indexes = events["base_index"].to_numpy(dtype=np.int64)
    for (higher, relationship), mask in relationship_masks.items():
        events[f"rel__{higher}__{relationship}"] = mask[indexes]


def attach_enrichment(events: DataFrame, enrichment: DataFrame) -> DataFrame:
    indexes = events["base_index"].to_numpy(dtype=np.int64)
    values = enrichment.iloc[indexes].reset_index(drop=True).drop(columns=["date"])
    duplicate = set(events.columns).intersection(values.columns)
    if duplicate:
        raise ValueError(f"Duplicate enrichment columns: {sorted(duplicate)}")
    return pd.concat([events.reset_index(drop=True), values], axis=1)


def inventory_record(
    task: PairTask,
    *,
    status: str,
    event_path: Path,
    summary_path: Path,
    events: DataFrame,
    supported_specs: int,
    causal_violations: int,
) -> dict[str, Any]:
    controls = events["control"].astype(str).value_counts().to_dict() if len(events) else {}
    record: dict[str, Any] = {
        "status": status,
        "cohort": task.cohort,
        "pair": task.pair,
        "event_path": str(event_path),
        "event_sha256": g0.sha256_file(event_path),
        "summary_path": str(summary_path),
        "summary_sha256": g0.sha256_file(summary_path),
        "event_rows": len(events),
        "actual_event_rows": int(controls.get("actual", 0)),
        "supported_specs": supported_specs,
        "causal_timestamp_violations": causal_violations,
    }
    for control in CONTROLS:
        record[f"control_rows__{control}"] = int(controls.get(control, 0))
    return record


def combine_summaries(results: Sequence[dict[str, Any]]) -> DataFrame:
    frames = [pd.read_parquet(item["summary_path"]) for item in results]
    return pd.concat(frames, ignore_index=True) if frames else DataFrame()


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


if __name__ == "__main__":
    raise SystemExit(main())

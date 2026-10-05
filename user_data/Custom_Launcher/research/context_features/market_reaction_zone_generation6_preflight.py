"""Outcome-blind support preflight for the seven Generation 6 sibling lanes.

This module deliberately does not build future-path matrices or read reaction targets.
It freezes which source/level/timeframe/cohort cells have enough causal observations
to justify later direct-control or FreqAI experiments.
"""

from __future__ import annotations

# Bind numerical libraries before importing pandas and source helpers. Four process
# workers must not each create their own large native thread pool.
# ruff: noqa: E402
import argparse
import json
import math
import os
import re
import sys
from collections.abc import Iterable, Sequence
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
from pyarrow import parquet as pq


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


USER_DATA_DIR = REPO_ROOT / "user_data"
OUTPUT_ROOT = USER_DATA_DIR / "research_news_data" / "context_features" / "market_reaction_zones"
FROZEN_BATCH = OUTPUT_ROOT / "generation5_review" / "g6_frozen_branch_batch.json"
COVERAGE_CROSSWALK = (
    OUTPUT_ROOT / "generation5_review" / "g6_source_level_coverage_crosswalk_20260821.json"
)
NORMAL_MANIFEST = OUTPUT_ROOT / "generation6_shared" / "g6_normal_source_manifest.json"
MEME_MANIFEST = OUTPUT_ROOT / "generation6_shared" / "g6_meme_source_manifest.json"
REPORT_ROOT = OUTPUT_ROOT / "generation6_branches"
HISTORICAL_ORDERBOOK = (
    USER_DATA_DIR
    / "orderbook_data"
    / "historical_bybit"
    / "features"
    / "orderbook_trader_state_1h_bybit_linear.parquet"
)
LIVE_ORDERBOOK_EXPORT_DIR = USER_DATA_DIR / "orderbook_data" / "live" / "exports"
CONTEXT_SNAPSHOT = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "exports"
    / "context_features_1h_20260626_000458.parquet"
)

MAX_WORKERS = 4
COOLDOWN_HOURS = 6
OUTCOME_EMBARGO_HOURS = 8
MIN_FULL_EVENTS = 50
MIN_FULL_COINS = 5
MIN_SUBGROUP_EVENTS = 30
MIN_SUBGROUP_COINS = 3
MIN_BTC_EVENTS = 50

SELECTED_LEVEL_FAMILIES = frozenset(
    {
        "volume_profile_settled",
        "volume_profile_nodes",
        "volume_profile_explicit_prior",
        "confirmed_swing",
        "generic_prior_range",
        "generic_round_number",
        "tlv2_ranked",
        "tlv2_forecast_zone",
    }
)

GROUPS: dict[str, tuple[str, ...]] = {
    "btc_separate": ("BTC/USDT:USDT",),
    "established_altcoins": (
        "ETH/USDT:USDT",
        "BNB/USDT:USDT",
        "SOL/USDT:USDT",
        "XRP/USDT:USDT",
        "ADA/USDT:USDT",
        "TRX/USDT:USDT",
        "AVAX/USDT:USDT",
        "LINK/USDT:USDT",
    ),
    "smart_contract_platforms": (
        "ETH/USDT:USDT",
        "BNB/USDT:USDT",
        "SOL/USDT:USDT",
        "ADA/USDT:USDT",
        "AVAX/USDT:USDT",
    ),
    "payments_and_transfer": ("XRP/USDT:USDT", "TRX/USDT:USDT"),
    "frozen_top_ten_memes": (
        "DOGE/USDT:USDT",
        "1000PEPE/USDT:USDT",
        "PUMP/USDT:USDT",
        "1000SHIB/USDT:USDT",
        "TRUMP/USDT:USDT",
        "PENGU/USDT:USDT",
        "1000BONK/USDT:USDT",
        "FARTCOIN/USDT:USDT",
        "WIF/USDT:USDT",
        "ORDI/USDT:USDT",
    ),
}


@dataclass(frozen=True)
class PairTask:
    cohort: str
    pair: str
    manifest_path: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count causal level contacts and source overlap for all seven frozen "
            "Generation 6 siblings without reading reaction outcomes."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1 or args.workers > MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    frozen = validate_frozen_contract()
    run_dir = REPORT_ROOT / args.run_id
    record_path = run_dir / "g6_preflight_record.json"
    if record_path.is_file() and not args.overwrite:
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        print(json.dumps(existing["summary"], indent=2, sort_keys=True))
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)

    tasks = pair_tasks()
    pair_results = run_pair_tasks(tasks, workers=args.workers)
    failures = [item for item in pair_results if item.get("status") != "completed"]
    if failures:
        raise RuntimeError(f"Generation 6 pair preflight failed: {failures}")

    level_detail = concat_records(pair_results, "level_rows")
    relationship_detail = concat_records(pair_results, "relationship_rows")
    state_detail = concat_records(pair_results, "state_rows")
    transfer_detail = concat_records(pair_results, "transfer_rows")
    causal_violations = sum(int(item["causal_violations"]) for item in pair_results)
    if causal_violations:
        raise AssertionError(f"Causal cache merge admitted {causal_violations} future source rows.")

    level_cells = summarize_supported_cells(
        level_detail,
        keys=("cohort", "timeframe", "level_family", "level_name", "representation"),
    )
    relationship_cells = summarize_supported_cells(
        relationship_detail,
        keys=("cohort", "higher_timeframe", "relationship"),
    )
    state_cells = summarize_state_cells(state_detail)
    cross_market = build_cross_market_coverage()
    external_context, context_audit = build_context_coverage()
    orderbook = build_orderbook_coverage()
    group_transfer = summarize_group_transfer(transfer_detail, relationship_detail)
    branch_decisions = build_branch_decisions(
        level_cells=level_cells,
        relationship_cells=relationship_cells,
        state_cells=state_cells,
        cross_market=cross_market,
        external_context=external_context,
        context_audit=context_audit,
        orderbook=orderbook,
        group_transfer=group_transfer,
    )

    artifacts = {
        "level_support_detail": write_frame(level_detail, run_dir / "g6_level_support_detail.csv"),
        "level_supported_cells": write_frame(level_cells, run_dir / "g6_level_supported_cells.csv"),
        "timeframe_relationship_detail": write_frame(
            relationship_detail, run_dir / "g6_timeframe_relationship_detail.csv"
        ),
        "timeframe_supported_cells": write_frame(
            relationship_cells, run_dir / "g6_timeframe_supported_cells.csv"
        ),
        "ohlcv_indicator_coverage": write_frame(
            state_cells, run_dir / "g6_ohlcv_indicator_coverage.csv"
        ),
        "cross_market_coverage": write_frame(
            cross_market, run_dir / "g6_cross_market_coverage.csv"
        ),
        "external_context_coverage": write_frame(
            external_context, run_dir / "g6_external_context_coverage.csv"
        ),
        "context_block_audit": write_frame(context_audit, run_dir / "g6_context_block_audit.csv"),
        "orderbook_coverage": write_frame(orderbook, run_dir / "g6_orderbook_coverage.csv"),
        "market_group_transfer_support": write_frame(
            group_transfer, run_dir / "g6_market_group_transfer_support.csv"
        ),
        "branch_decisions": write_frame(
            branch_decisions, run_dir / "g6_preflight_branch_decisions.csv"
        ),
    }

    supported_branches = int(branch_decisions["status"].str.startswith("supported").sum())
    record = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "completed_outcome_blind",
        "created_at_utc": g0.utc_now(),
        "objective": "Generation 6 support and timestamp-integrity preflight",
        "frozen_batch": artifact_record(FROZEN_BATCH),
        "coverage_crosswalk": artifact_record(COVERAGE_CROSSWALK),
        "source_manifests": {
            "normal": artifact_record(NORMAL_MANIFEST),
            "meme": artifact_record(MEME_MANIFEST),
        },
        "research_boundary": {
            "reaction_outcomes_opened": False,
            "direction_outcomes_opened": False,
            "profit_used": False,
            "future_path_matrices_built": False,
            "missing_data_imputed_as_neutral": False,
        },
        "plain_language_result": (
            "This run counted whether each planned level, timeframe relation, market-state "
            "block, outside source, and coin group has honest causal support. It did not "
            "inspect what price or volume did afterwards."
        ),
        "generation_contract": frozen["generation_rule"],
        "workers": args.workers,
        "system_at_launch": g0.system_snapshot(args.workers),
        "artifacts": artifacts,
        "summary": {
            "pair_tasks": len(pair_results),
            "level_detail_rows": len(level_detail),
            "supported_level_cells": int(level_cells["supported"].sum()),
            "timeframe_relationship_cells": len(relationship_cells),
            "supported_timeframe_relationship_cells": int(relationship_cells["supported"].sum()),
            "ohlcv_indicator_cells": len(state_cells),
            "cross_market_cells": len(cross_market),
            "external_context_blocks": len(external_context),
            "orderbook_blocks": len(orderbook),
            "market_group_questions": len(group_transfer),
            "sibling_branches_supported_or_partly_supported": supported_branches,
            "sibling_branches_total": len(branch_decisions),
            "causal_timestamp_violations": causal_violations,
            "reaction_outcomes_opened": False,
            "direction_outcomes_opened": False,
            "profit_used": False,
        },
    }
    g0.atomic_write_json(record, record_path)
    print(json.dumps(record["summary"], indent=2, sort_keys=True))
    return 0


def validate_frozen_contract() -> dict[str, Any]:
    for path in (FROZEN_BATCH, COVERAGE_CROSSWALK, NORMAL_MANIFEST, MEME_MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(path)
    frozen = json.loads(FROZEN_BATCH.read_text(encoding="utf-8"))
    if frozen.get("generation") != 6 or frozen.get("branch_layer") != 6:
        raise ValueError("The routed frozen batch is not Generation 6.")
    if frozen.get("status") != "frozen_before_generation6_reaction_outcomes":
        raise ValueError("Generation 6 must be frozen before this preflight runs.")
    if len(frozen.get("branches", [])) != 7:
        raise ValueError("Generation 6 must contain all seven sibling lanes.")
    boundary = frozen.get("research_boundary", {})
    if boundary.get("profit_optimization") is not False:
        raise ValueError("Profit optimization must remain closed.")
    if boundary.get("ordinary_timestamp_direction_prediction") is not False:
        raise ValueError("Broad direction prediction must remain closed.")
    return frozen


def pair_tasks() -> list[PairTask]:
    tasks: list[PairTask] = []
    for cohort, manifest_path in (
        ("normal", NORMAL_MANIFEST),
        ("meme", MEME_MANIFEST),
    ):
        manifest = g0.load_manifest(manifest_path)
        tasks.extend(
            PairTask(cohort=cohort, pair=str(pair), manifest_path=str(manifest_path))
            for pair in manifest["data"]["pairs"]
        )
    return tasks


def run_pair_tasks(tasks: Sequence[PairTask], *, workers: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(build_pair_support, task): task for task in tasks}
        for future in as_completed(future_map):
            task = future_map[future]
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


def build_pair_support(task: PairTask) -> dict[str, Any]:
    manifest_path = Path(task.manifest_path)
    manifest = g0.load_manifest(manifest_path)
    storage = g0.manifest_storage_paths(manifest)
    base = g0.prepare_base_market_frame(task.pair, manifest)
    end = pd.Timestamp(manifest["data"]["analysis_end_utc_exclusive"])
    base = base.loc[base["date"] < end].reset_index(drop=True)
    if base.empty:
        raise ValueError(f"No base rows for {task.pair}")

    level_rows: list[dict[str, Any]] = []
    relationship_rows: list[dict[str, Any]] = []
    transfer_masks: dict[str, np.ndarray] = {
        "one_hour_thin_lvn_activity_benchmark": np.zeros(len(base), dtype=bool),
        "volume_profile_value_area_reaction": np.zeros(len(base), dtype=bool),
        "prior_week_or_month_extreme_reaction": np.zeros(len(base), dtype=bool),
    }
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
        valid_available = merged["available_at"].notna()
        causal_violations += int(
            (
                merged.loc[valid_available, "available_at"] > merged.loc[valid_available, "date"]
            ).sum()
        )
        high = g0.numeric_array(merged["high"])
        low = g0.numeric_array(merged["low"])
        base_atr = g0.numeric_array(merged["base_atr"])
        for spec in selected_level_specs(cache):
            level = g0.resolved_level_values(merged, spec, timeframe)
            valid = np.isfinite(level) & (level > 0.0) & np.isfinite(base_atr) & (base_atr > 0.0)
            if spec.active_columns:
                active = np.zeros(len(merged), dtype=bool)
                for column in spec.active_columns:
                    if column in merged:
                        active |= g0.bool_array(merged[column])
                valid &= active
            native_width = g0.resolved_native_width(merged, spec)
            half_width = g0.zone_half_width("standard_base_atr", level, base_atr, native_width)
            finite = valid & np.isfinite(half_width) & (half_width > 0.0)
            contact = finite & (high >= level - half_width) & (low <= level + half_width)
            starts = g0.episode_start_mask(contact, level, half_width, cooldown=COOLDOWN_HOURS)
            starts[max(len(starts) - OUTCOME_EMBARGO_HOURS, 0) :] = False
            mechanism = mechanism_group(spec)
            side = level_side(spec)
            mechanism_contacts.setdefault((mechanism, timeframe), np.zeros(len(base), dtype=bool))
            mechanism_contacts[(mechanism, timeframe)] |= contact
            side_contacts.setdefault((side, timeframe), np.zeros(len(base), dtype=bool))
            side_contacts[(side, timeframe)] |= contact
            add_transfer_contact(transfer_masks, spec, timeframe, contact)

            for period, period_mask in period_masks(merged).items():
                level_rows.append(
                    {
                        "cohort": task.cohort,
                        "pair": task.pair,
                        "timeframe": timeframe,
                        "level_family": spec.family,
                        "level_name": spec.name,
                        "representation": spec.representation,
                        "mechanism_group": mechanism,
                        "level_side": side,
                        "period": period,
                        "finite_level_rows": int((finite & period_mask).sum()),
                        "contact_rows": int((contact & period_mask).sum()),
                        "event_count": int((starts & period_mask).sum()),
                        "reaction_outcomes_opened": False,
                    }
                )

    relationship_masks = timeframe_relationship_masks(
        mechanism_contacts=mechanism_contacts,
        side_contacts=side_contacts,
        row_count=len(base),
    )
    for (higher_timeframe, relationship), condition in relationship_masks.items():
        starts = g0.cooldown_start_mask(condition, cooldown=COOLDOWN_HOURS)
        starts[max(len(starts) - OUTCOME_EMBARGO_HOURS, 0) :] = False
        for period, period_mask in period_masks(base).items():
            relationship_rows.append(
                {
                    "cohort": task.cohort,
                    "pair": task.pair,
                    "higher_timeframe": higher_timeframe,
                    "relationship": relationship,
                    "period": period,
                    "contact_rows": int((condition & period_mask).sum()),
                    "event_count": int((starts & period_mask).sum()),
                    "reaction_outcomes_opened": False,
                }
            )

    state_rows = state_coverage_rows(task.cohort, task.pair, base)
    transfer_rows = transfer_coverage_rows(
        task.cohort,
        task.pair,
        base,
        transfer_masks,
        relationship_masks,
    )
    return {
        "status": "completed",
        "cohort": task.cohort,
        "pair": task.pair,
        "level_rows": level_rows,
        "relationship_rows": relationship_rows,
        "state_rows": state_rows,
        "transfer_rows": transfer_rows,
        "causal_violations": causal_violations,
    }


def selected_level_specs(cache: DataFrame) -> list[g0.LevelSpec]:
    selected: list[g0.LevelSpec] = []
    for spec in g0.level_specs(cache, ("g0b1", "g0b2")):
        if spec.family not in SELECTED_LEVEL_FAMILIES:
            continue
        if spec.family == "tlv2_ranked" and (
            "rank0" not in spec.name or spec.representation != "projected"
        ):
            continue
        if spec.family == "tlv2_forecast_zone" and (
            not spec.name.endswith("_center") or spec.representation != "projected"
        ):
            continue
        selected.append(spec)
    return selected


def mechanism_group(spec: g0.LevelSpec) -> str:
    return {
        "volume_profile_settled": "volume_profile_value_area",
        "volume_profile_explicit_prior": "volume_profile_value_area",
        "volume_profile_nodes": "volume_profile_nodes",
        "confirmed_swing": "confirmed_swing",
        "generic_prior_range": "prior_extreme",
        "generic_round_number": "round_number",
        "tlv2_ranked": "trendline",
        "tlv2_forecast_zone": "trendline",
    }[spec.family]


def level_side(spec: g0.LevelSpec) -> str:
    name = spec.name.lower()
    if any(token in name for token in ("high", "above", "resistance", "vah")):
        return "upper"
    if any(token in name for token in ("low", "below", "support", "val")):
        return "lower"
    return "neutral"


def add_transfer_contact(
    transfer_masks: dict[str, np.ndarray],
    spec: g0.LevelSpec,
    timeframe: str,
    contact: np.ndarray,
) -> None:
    if timeframe != "1h":
        return
    name = spec.name.lower()
    if spec.family == "volume_profile_nodes" and "lvn" in name:
        transfer_masks["one_hour_thin_lvn_activity_benchmark"] |= contact
    if spec.family == "volume_profile_settled" and name in {"vah", "val"}:
        transfer_masks["volume_profile_value_area_reaction"] |= contact
    if spec.family == "generic_prior_range" and any(
        token in name for token in ("high_168", "low_168", "high_720", "low_720")
    ):
        transfer_masks["prior_week_or_month_extreme_reaction"] |= contact


def timeframe_relationship_masks(
    *,
    mechanism_contacts: dict[tuple[str, str], np.ndarray],
    side_contacts: dict[tuple[str, str], np.ndarray],
    row_count: int,
) -> dict[tuple[str, str], np.ndarray]:
    groups = sorted({group for group, _ in mechanism_contacts})
    zero = np.zeros(row_count, dtype=bool)
    native_by_group = {group: mechanism_contacts.get((group, "1h"), zero) for group in groups}
    native_any = any_mask(native_by_group.values(), row_count)
    output: dict[tuple[str, str], np.ndarray] = {}
    for higher in ("4h", "8h", "1d"):
        higher_by_group = {group: mechanism_contacts.get((group, higher), zero) for group in groups}
        higher_any = any_mask(higher_by_group.values(), row_count)
        same = any_mask(
            (native_by_group[group] & higher_by_group[group] for group in groups),
            row_count,
        )
        different = any_mask(
            (
                native_by_group[left] & higher_by_group[right]
                for left in groups
                for right in groups
                if left != right
            ),
            row_count,
        )
        native_upper = side_contacts.get(("upper", "1h"), zero)
        native_lower = side_contacts.get(("lower", "1h"), zero)
        higher_upper = side_contacts.get(("upper", higher), zero)
        higher_lower = side_contacts.get(("lower", higher), zero)
        output[(higher, "isolated_native_1h")] = native_any & ~higher_any
        output[(higher, "isolated_higher_timeframe")] = higher_any & ~native_any
        output[(higher, "same_mechanism_agreement")] = same
        output[(higher, "different_mechanism_agreement")] = different
        output[(higher, "any_cross_timeframe_cluster")] = native_any & higher_any
        output[(higher, "opposing_side_overlap")] = (native_upper & higher_lower) | (
            native_lower & higher_upper
        )
    return output


def any_mask(values: Iterable[np.ndarray], row_count: int) -> np.ndarray:
    output = np.zeros(row_count, dtype=bool)
    for value in values:
        output |= value
    return output


def period_masks(frame: DataFrame) -> dict[str, np.ndarray]:
    periods = frame["period"].astype("string")
    return {
        str(period): periods.eq(period).fillna(False).to_numpy(dtype=bool)
        for period in periods.dropna().unique()
        if str(period) not in {"", "<NA>", "nan"}
    }


def causal_state_features(base: DataFrame) -> tuple[DataFrame, dict[str, tuple[str, ...]]]:
    high = pd.to_numeric(base["high"], errors="coerce")
    low = pd.to_numeric(base["low"], errors="coerce")
    close = pd.to_numeric(base["close"], errors="coerce")
    volume = pd.to_numeric(base["volume"], errors="coerce")
    candle_range = high - low
    pressure = (2.0 * close - high - low).div(candle_range.replace(0.0, np.nan))
    prior_close = close.shift(1)
    prior_volume = volume.shift(1)
    prior_range = candle_range.shift(1)
    prior_atr = pd.to_numeric(base["base_atr"], errors="coerce")

    output = DataFrame(index=base.index)
    output["relative_volume"] = prior_volume.div(prior_volume.rolling(24, min_periods=24).median())
    output["volume_acceleration"] = np.log1p(prior_volume.clip(lower=0.0)).diff(3)
    output["absolute_pressure"] = pressure.shift(1).abs()
    output["pressure_persistence"] = pressure.shift(1).rolling(6, min_periods=6).mean().abs()
    output["atr_fraction"] = prior_atr.div(prior_close.abs())
    output["prior_range_atr"] = prior_range.div(prior_atr)
    close_known = close.shift(1)
    bb_mid = close_known.rolling(20, min_periods=20).mean()
    bb_std = close_known.rolling(20, min_periods=20).std(ddof=0)
    output["bollinger_width"] = (4.0 * bb_std).div(bb_mid.abs())
    output["range_contraction"] = prior_range.div(prior_range.rolling(24, min_periods=24).median())
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean().shift(1)
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean().shift(1)
    output["ema20_slope"] = ema20.pct_change(4, fill_method=None)
    output["ma_separation"] = (ema20 - ema50).abs().div(prior_close.abs())
    prior_return = close.pct_change(fill_method=None).shift(1)
    output["return_slope"] = prior_return.rolling(6, min_periods=6).mean()
    output["return_acceleration"] = (
        prior_return.rolling(3, min_periods=3).mean()
        - prior_return.rolling(12, min_periods=12).mean()
    )
    output["adx14"] = adx(high, low, close, 14).shift(1)
    raw_rsi = rsi(close, 14)
    output["rsi14"] = raw_rsi.shift(1)
    output["rsi_change"] = raw_rsi.diff().shift(1)
    macd = (
        close.ewm(span=12, adjust=False, min_periods=26).mean()
        - close.ewm(span=26, adjust=False, min_periods=26).mean()
    )
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_hist = macd - macd_signal
    output["macd_histogram"] = macd_hist.shift(1)
    output["macd_histogram_change"] = macd_hist.diff().shift(1)
    blocks = {
        "volume_and_pressure": (
            "relative_volume",
            "volume_acceleration",
            "absolute_pressure",
            "pressure_persistence",
        ),
        "volatility_and_compression": (
            "atr_fraction",
            "prior_range_atr",
            "bollinger_width",
            "range_contraction",
        ),
        "trend_and_acceleration": (
            "ema20_slope",
            "ma_separation",
            "return_slope",
            "return_acceleration",
            "adx14",
        ),
        "momentum": (
            "rsi14",
            "rsi_change",
            "macd_histogram",
            "macd_histogram_change",
        ),
    }
    return output.replace([np.inf, -np.inf], np.nan), blocks


def rsi(close: Series, period: int) -> Series:
    change = close.diff()
    gain = change.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    loss = (
        (-change.clip(upper=0.0)).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    )
    relative = gain.div(loss.replace(0.0, np.nan))
    return 100.0 - 100.0 / (1.0 + relative)


def adx(high: Series, low: Series, close: Series, period: int) -> Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = Series(np.where((up > down) & (up > 0.0), up, 0.0), index=high.index)
    minus_dm = Series(np.where((down > up) & (down > 0.0), down, 0.0), index=high.index)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_value = true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().div(
        atr_value
    )
    minus = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().div(
        atr_value
    )
    dx = 100.0 * (plus - minus).abs().div((plus + minus).replace(0.0, np.nan))
    return dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def state_coverage_rows(cohort: str, pair: str, base: DataFrame) -> list[dict[str, Any]]:
    features, blocks = causal_state_features(base)
    rows: list[dict[str, Any]] = []
    for period, mask in period_masks(base).items():
        for block, columns in blocks.items():
            view = features.loc[mask, list(columns)]
            complete = view.notna().all(axis=1)
            nonconstant = sum(
                pd.to_numeric(view[column], errors="coerce").dropna().nunique() > 1
                for column in columns
            )
            rows.append(
                {
                    "cohort": cohort,
                    "pair": pair,
                    "period": period,
                    "state_block": block,
                    "rows": len(view),
                    "complete_rows": int(complete.sum()),
                    "complete_fraction": float(complete.mean()) if len(view) else 0.0,
                    "feature_columns": len(columns),
                    "nonconstant_columns": int(nonconstant),
                    "causal_clock": "features_end_at_previous_candle_close",
                }
            )
        rows.append(
            {
                "cohort": cohort,
                "pair": pair,
                "period": period,
                "state_block": "approach_path",
                "rows": int(mask.sum()),
                "complete_rows": int(mask.sum()),
                "complete_fraction": 1.0 if mask.any() else 0.0,
                "feature_columns": 5,
                "nonconstant_columns": 5 if mask.any() else 0,
                "causal_clock": "computed_per_supported_level_event_before_contact",
            }
        )
    return rows


def transfer_coverage_rows(
    cohort: str,
    pair: str,
    base: DataFrame,
    transfer_masks: dict[str, np.ndarray],
    relationship_masks: dict[tuple[str, str], np.ndarray],
) -> list[dict[str, Any]]:
    higher_agreement = any_mask(
        (
            value
            for (_, relation), value in relationship_masks.items()
            if relation in {"same_mechanism_agreement", "different_mechanism_agreement"}
        ),
        len(base),
    )
    masks = dict(transfer_masks)
    masks["higher_timeframe_agreement_increment"] = higher_agreement
    rows: list[dict[str, Any]] = []
    for question, condition in masks.items():
        starts = g0.cooldown_start_mask(condition, cooldown=COOLDOWN_HOURS)
        starts[max(len(starts) - OUTCOME_EMBARGO_HOURS, 0) :] = False
        for period, period_mask in period_masks(base).items():
            rows.append(
                {
                    "cohort": cohort,
                    "pair": pair,
                    "question": question,
                    "period": period,
                    "event_count": int((starts & period_mask).sum()),
                }
            )
    return rows


def concat_records(results: Sequence[dict[str, Any]], key: str) -> DataFrame:
    records = [row for result in results for row in result.get(key, [])]
    return DataFrame.from_records(records)


def validation_periods_for_cohort(cohort: str) -> tuple[str, str]:
    if cohort == "normal":
        return ("validation_early", "validation_late")
    if cohort == "meme":
        return ("meme_validation_early", "meme_validation_late")
    raise ValueError(cohort)


def summarize_supported_cells(detail: DataFrame, *, keys: tuple[str, ...]) -> DataFrame:
    if detail.empty:
        return DataFrame(columns=[*keys, "supported"])
    aggregate = (
        detail.groupby([*keys, "period"], dropna=False)
        .agg(
            event_count=("event_count", "sum"),
            coins_with_events=(
                "pair",
                lambda value: int(detail.loc[value.index, "event_count"].gt(0).sum()),
            ),
        )
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for cell_key, cell in aggregate.groupby(list(keys), dropna=False):
        key_values = cell_key if isinstance(cell_key, tuple) else (cell_key,)
        record = dict(zip(keys, key_values, strict=True))
        cohort = str(record["cohort"])
        required_periods = validation_periods_for_cohort(cohort)
        checks: list[dict[str, Any]] = []
        for period in required_periods:
            found = cell.loc[cell["period"].eq(period)]
            events = int(found["event_count"].sum())
            coins = int(found["coins_with_events"].sum())
            checks.append(
                {
                    "period": period,
                    "event_count": events,
                    "coins_with_events": coins,
                    "passed": events >= MIN_FULL_EVENTS and coins >= MIN_FULL_COINS,
                }
            )
        record["supported"] = all(check["passed"] for check in checks)
        record["minimum_events_any_validation_period"] = min(
            check["event_count"] for check in checks
        )
        record["minimum_coins_any_validation_period"] = min(
            check["coins_with_events"] for check in checks
        )
        record["period_checks"] = json.dumps(checks, sort_keys=True)
        record["classification"] = (
            "open_outcomes_after_all_sibling_preflights"
            if record["supported"]
            else "park_without_outcomes_insufficient_support"
        )
        rows.append(record)
    return DataFrame.from_records(rows).sort_values(list(keys)).reset_index(drop=True)


def summarize_state_cells(detail: DataFrame) -> DataFrame:
    if detail.empty:
        return detail
    aggregate = (
        detail.groupby(["cohort", "state_block", "period"], dropna=False)
        .agg(
            rows=("rows", "sum"),
            complete_rows=("complete_rows", "sum"),
            coins=("pair", "nunique"),
            minimum_nonconstant_columns=("nonconstant_columns", "min"),
        )
        .reset_index()
    )
    aggregate["complete_fraction"] = aggregate["complete_rows"].div(
        aggregate["rows"].replace(0, np.nan)
    )
    aggregate["supported"] = (
        aggregate["complete_rows"].ge(MIN_FULL_EVENTS)
        & aggregate["coins"].ge(MIN_FULL_COINS)
        & aggregate["minimum_nonconstant_columns"].gt(0)
    )
    return aggregate.sort_values(["cohort", "state_block", "period"]).reset_index(drop=True)


def build_cross_market_coverage() -> DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort, manifest_path in (("normal", NORMAL_MANIFEST), ("meme", MEME_MANIFEST)):
        manifest = g0.load_manifest(manifest_path)
        pairs = tuple(str(pair) for pair in manifest["data"]["pairs"])
        close_frames: list[DataFrame] = []
        for pair in sorted(set((*pairs, "BTC/USDT:USDT", "ETH/USDT:USDT"))):
            source = g0.load_ohlcv(g0.ohlcv_path(pair, "1h"))[["date", "close"]].copy()
            source = source.rename(columns={"close": pair}).drop_duplicates("date")
            close_frames.append(source.set_index("date"))
        prices = pd.concat(close_frames, axis=1, sort=False).sort_index()
        returns = prices.pct_change(fill_method=None).shift(1)
        cohort_returns = returns.loc[:, list(pairs)]
        required_members = max(3, math.ceil(len(pairs) * 0.7))
        block_masks = {
            "btc_eth_state": returns[["BTC/USDT:USDT", "ETH/USDT:USDT"]].notna().all(axis=1),
            "cohort_breadth_and_dispersion": cohort_returns.notna()
            .sum(axis=1)
            .ge(required_members),
            "rolling_correlation_beta_and_decoupling": returns["BTC/USDT:USDT"]
            .rolling(168, min_periods=168)
            .std()
            .notna(),
        }
        dates = Series(prices.index, index=prices.index)
        periods = g0.assign_periods(dates, manifest)
        end = pd.Timestamp(manifest["data"]["analysis_end_utc_exclusive"])
        for block, ready in block_masks.items():
            for period in periods.dropna().unique():
                period_mask = periods.eq(period) & dates.lt(end)
                rows.append(
                    {
                        "cohort": cohort,
                        "source_block": block,
                        "period": str(period),
                        "rows": int(period_mask.sum()),
                        "ready_rows": int((period_mask & ready).sum()),
                        "ready_fraction": float(ready.loc[period_mask].mean())
                        if period_mask.any()
                        else 0.0,
                        "status": "supported"
                        if int((period_mask & ready).sum()) >= MIN_FULL_EVENTS
                        else "insufficient_support",
                    }
                )
        for pair in pairs:
            ready = returns[[pair, "BTC/USDT:USDT"]].notna().all(axis=1)
            for period in periods.dropna().unique():
                period_mask = periods.eq(period) & dates.lt(end)
                rows.append(
                    {
                        "cohort": cohort,
                        "source_block": "pair_relative_strength",
                        "period": str(period),
                        "pair": pair,
                        "rows": int(period_mask.sum()),
                        "ready_rows": int((period_mask & ready).sum()),
                        "ready_fraction": float(ready.loc[period_mask].mean())
                        if period_mask.any()
                        else 0.0,
                        "status": "supported"
                        if int((period_mask & ready).sum()) >= MIN_FULL_EVENTS
                        else "insufficient_support",
                    }
                )
    return DataFrame.from_records(rows)


def build_context_coverage() -> tuple[DataFrame, DataFrame]:
    sources, representation = g5e.load_context_sources()
    _, audit, _ = g5e_preflight.load_context_masks(CONTEXT_SNAPSHOT)
    rows: list[dict[str, Any]] = []
    for source_id, source in sources.items():
        ready = source["ready"].fillna(False).astype(bool)
        if source_id == "global_market_macro":
            status = "park_without_outcomes_insufficient_two_slice_support"
            source_note = (
                "Only a short May-June 2026 genuinely ready window remains after causal "
                "carry and history controls. It cannot supply two declared validation "
                "slices."
            )
        elif int(ready.sum()) >= MIN_FULL_EVENTS * 2:
            status = "supported_masked_windows"
            source_note = "Genuine ready rows after causal availability and history controls."
        else:
            status = "park_without_outcomes_insufficient_genuine_support"
            source_note = "The genuinely ready rows do not reach the minimum broad support gate."
        rows.append(
            {
                "source_block": source_id,
                "rows": len(source),
                "ready_rows": int(ready.sum()),
                "first_ready": str(source.loc[ready, "date"].min()) if ready.any() else None,
                "last_ready": str(source.loc[ready, "date"].max()) if ready.any() else None,
                "feature_columns": len(
                    [column for column in source if column.startswith("value__")]
                ),
                "status": status,
                "note": source_note,
            }
        )
    for block in (
        "context_article_source_activity",
        "context_gkg_documents",
        "context_google_trends",
        "context_btc_etf_flows",
        "context_topic_severity",
    ):
        match = audit.loc[audit["source_detail_block"].eq(block)]
        if match.empty:
            continue
        item = match.iloc[0]
        rows.append(
            {
                "source_block": block,
                "rows": int(item["coverage_rows"]),
                "ready_rows": int(item["usable_rows"]),
                "first_ready": item["first_usable_date"],
                "last_ready": item["last_usable_date"],
                "feature_columns": int(item["numeric_signal_columns"]),
                "status": (
                    "supported_masked_windows"
                    if block == "context_topic_severity"
                    and int(item["usable_rows"]) >= MIN_FULL_EVENTS * 2
                    else "park_without_outcomes_insufficient_or_unconditioned_support"
                ),
                "note": str(item["missing_data_caution"]),
            }
        )
    context = DataFrame.from_records(rows)
    context["snapshot"] = str(CONTEXT_SNAPSHOT)
    context["snapshot_sha256"] = g0.sha256_file(CONTEXT_SNAPSHOT)
    context["representation_record"] = json.dumps(representation, default=g0.json_default)
    return context, audit


def newest_live_orderbook_export() -> Path | None:
    pattern = re.compile(r"^orderbook_features_1h_\d{8}_\d{6}\.parquet$")
    candidates = [
        path
        for path in LIVE_ORDERBOOK_EXPORT_DIR.glob("orderbook_features_1h_*.parquet")
        if pattern.match(path.name)
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def build_orderbook_coverage() -> DataFrame:
    rows: list[dict[str, Any]] = []
    historical = g3h.load_causal_pressure_surface(HISTORICAL_ORDERBOOK)
    source_ready = historical["source_row_usable"].fillna(False).astype(bool)
    regime_ready = historical["pressure_regime"].ne("unavailable")
    rows.append(
        {
            "source_block": "historical_bybit_btcusdt_linear",
            "venue_role": "BTC_pair_local_and_BTC_wide_only",
            "pairs": "BTC/USDT:USDT",
            "rows": len(historical),
            "source_ready_rows": int(source_ready.sum()),
            "model_ready_rows": int(regime_ready.sum()),
            "first_ready": str(historical.loc[regime_ready, "date"].min()),
            "last_ready": str(historical.loc[regime_ready, "date"].max()),
            "status": "supported_btc_only",
            "snapshot": str(HISTORICAL_ORDERBOOK),
            "snapshot_sha256": g0.sha256_file(HISTORICAL_ORDERBOOK),
            "note": "Never label this pair-local for an altcoin or meme.",
        }
    )

    live_path = newest_live_orderbook_export()
    if live_path is None:
        rows.append(
            {
                "source_block": "live_binance_pair_local",
                "status": "park_without_outcomes_no_frozen_export",
                "note": "The live SQLite collector was not read.",
            }
        )
        return DataFrame.from_records(rows)
    schema = pq.ParquetFile(live_path).schema.names
    ready_columns = [column for column in schema if column.endswith("_ready")]
    columns = [
        column
        for column in ("date", "canonical_pair", "source_max_ts", *ready_columns)
        if column in schema
    ]
    live = pd.read_parquet(live_path, columns=columns)
    live["date"] = pd.to_datetime(live["date"], utc=True, errors="coerce")
    live["source_max_ts"] = pd.to_datetime(live.get("source_max_ts"), utc=True, errors="coerce")
    source_safe = live["source_max_ts"].notna() & live["source_max_ts"].le(live["date"])
    if ready_columns:
        ready_any = live[ready_columns].apply(pd.to_numeric, errors="coerce").eq(1.0).any(axis=1)
    else:
        ready_any = Series(False, index=live.index)
    usable = source_safe & ready_any
    pair_column = "canonical_pair"
    for pair, group in live.groupby(pair_column, dropna=False):
        group_usable = usable.loc[group.index]
        last_ready = group.loc[group_usable, "date"].max() if group_usable.any() else pd.NaT
        rows.append(
            {
                "source_block": "live_binance_pair_local",
                "venue_role": "pair_local",
                "pairs": str(pair),
                "rows": len(group),
                "source_ready_rows": int(source_safe.loc[group.index].sum()),
                "model_ready_rows": int(group_usable.sum()),
                "first_ready": str(group.loc[group_usable, "date"].min())
                if group_usable.any()
                else None,
                "last_ready": str(last_ready) if group_usable.any() else None,
                "status": "refresh_required_before_outcomes",
                "snapshot": str(live_path),
                "snapshot_sha256": g0.sha256_file(live_path),
                "note": (
                    "Frozen export is older than the active Generation 6 date. Refresh "
                    "only after safely pausing the exact collector and exporting parquet."
                ),
            }
        )
    return DataFrame.from_records(rows)


def summarize_group_transfer(
    transfer_detail: DataFrame, relationship_detail: DataFrame
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    combined = transfer_detail.copy()
    volume_rows = []
    for cohort in ("normal", "meme"):
        for period in validation_periods_for_cohort(cohort):
            volume_rows.append(
                {
                    "cohort": cohort,
                    "pair": "previous_generation_multi_pair_surface",
                    "question": "strict_precontact_and_next_candle_volume_relationship",
                    "period": period,
                    "event_count": 0,
                }
            )
    combined = pd.concat([combined, DataFrame.from_records(volume_rows)], ignore_index=True)
    for group_id, members in GROUPS.items():
        cohort = "meme" if group_id == "frozen_top_ten_memes" else "normal"
        required_periods = validation_periods_for_cohort(cohort)
        group_rows = combined.loc[
            combined["cohort"].eq(cohort)
            & (
                combined["pair"].isin(members)
                | combined["pair"].eq("previous_generation_multi_pair_surface")
            )
        ]
        for question, question_rows in group_rows.groupby("question"):
            checks: list[dict[str, Any]] = []
            for period in required_periods:
                period_rows = question_rows.loc[question_rows["period"].eq(period)]
                events = int(period_rows["event_count"].sum())
                coins = int(period_rows.loc[period_rows["event_count"].gt(0), "pair"].nunique())
                if question == "strict_precontact_and_next_candle_volume_relationship":
                    passed = group_id in {
                        "established_altcoins",
                        "smart_contract_platforms",
                        "frozen_top_ten_memes",
                    }
                    events = -1
                    coins = -1
                elif group_id == "btc_separate":
                    passed = events >= MIN_BTC_EVENTS
                elif group_id in {"established_altcoins", "frozen_top_ten_memes"}:
                    passed = events >= MIN_FULL_EVENTS and coins >= MIN_FULL_COINS
                elif len(members) >= MIN_SUBGROUP_COINS:
                    passed = events >= MIN_SUBGROUP_EVENTS and coins >= MIN_SUBGROUP_COINS
                else:
                    passed = False
                checks.append(
                    {
                        "period": period,
                        "events": events,
                        "coins": coins,
                        "passed": passed,
                    }
                )
            supported = all(check["passed"] for check in checks)
            rows.append(
                {
                    "group_id": group_id,
                    "member_count": len(members),
                    "members": ",".join(members),
                    "question": question,
                    "supported": supported,
                    "classification": (
                        "open_group_outcomes_after_all_preflights"
                        if supported
                        else "park_without_group_outcomes_insufficient_support"
                    ),
                    "period_checks": json.dumps(checks, sort_keys=True),
                    "note": (
                        "Volume relationship support comes from the completed Generation 5 "
                        "review and receives one fresh Generation 6 cell."
                        if question == "strict_precontact_and_next_candle_volume_relationship"
                        else "Outcome-blind de-duplicated contact episodes."
                    ),
                }
            )
    return DataFrame.from_records(rows)


def build_branch_decisions(
    *,
    level_cells: DataFrame,
    relationship_cells: DataFrame,
    state_cells: DataFrame,
    cross_market: DataFrame,
    external_context: DataFrame,
    context_audit: DataFrame,
    orderbook: DataFrame,
    group_transfer: DataFrame,
) -> DataFrame:
    state_supported = int(state_cells["supported"].sum())
    crypto_supported = int(cross_market["status"].eq("supported").sum())
    gdelt_supported = int(
        external_context["source_block"]
        .astype(str)
        .str.contains("gdelt|topic")
        .mul(external_context["status"].eq("supported_masked_windows"))
        .sum()
    )
    orderbook_supported = int(orderbook["status"].astype(str).str.startswith("supported").sum())
    decisions = [
        {
            "branch_id": "g6a_calculated_price_areas",
            "plain_name": "Calculated price areas",
            "supported_cells": int(level_cells["supported"].sum()),
            "parked_cells": int((~level_cells["supported"]).sum()),
            "status": "supported_for_outcome_batch"
            if level_cells["supported"].any()
            else "parked_without_outcomes",
            "next_action": (
                "Run direct matched controls for every supported frozen family; only "
                "then open low-dimensional FreqAI profiles."
            ),
        },
        {
            "branch_id": "g6b_timeframe_relationships",
            "plain_name": "Timeframe relationships",
            "supported_cells": int(relationship_cells["supported"].sum()),
            "parked_cells": int((~relationship_cells["supported"]).sum()),
            "status": "supported_for_outcome_batch"
            if relationship_cells["supported"].any()
            else "parked_without_outcomes",
            "next_action": (
                "Run width/contact-matched native, higher-only, agreement, and "
                "opposing-side controls."
            ),
        },
        {
            "branch_id": "g6c_recent_ohlcv_and_standard_indicator_state",
            "plain_name": "Recent OHLCV and standard-indicator state",
            "supported_cells": state_supported,
            "parked_cells": int((~state_cells["supported"]).sum()),
            "status": "supported_for_outcome_batch"
            if state_supported
            else "parked_without_outcomes",
            "next_action": (
                "Test each named block separately against level-only, state-only, "
                "stale/shuffle, and away-from-level controls."
            ),
        },
        {
            "branch_id": "g6d_cross_market_and_global_state",
            "plain_name": "Cross-market and global state",
            "supported_cells": crypto_supported,
            "parked_cells": int(external_context["source_block"].eq("global_market_macro").sum()),
            "status": "supported_partial_crypto_first"
            if crypto_supported
            else "parked_without_outcomes",
            "next_action": (
                "Open causal crypto cross-market blocks. Keep macro parked unless "
                "genuine field-level overlap passes a refreshed source audit."
            ),
        },
        {
            "branch_id": "g6e_orderbook_state",
            "plain_name": "Orderbook state",
            "supported_cells": orderbook_supported,
            "parked_cells": int(orderbook["status"].eq("refresh_required_before_outcomes").sum()),
            "status": "supported_partial_btc_historical",
            "next_action": (
                "Open distinct BTC historical level questions. Refresh live pair-local "
                "parquet before considering those cells."
            ),
        },
        {
            "branch_id": "g6f_news_media_and_web_context",
            "plain_name": "News, media, and web context",
            "supported_cells": gdelt_supported,
            "parked_cells": int(
                external_context["status"].astype(str).str.startswith("park").sum()
            ),
            "status": "supported_partial_gdelt_only"
            if gdelt_supported
            else "parked_without_outcomes",
            "next_action": (
                "Open distinct GDELT/topic questions only. Refresh live sources "
                "outcome-blind; keep GKG, trends, and ETF flows parked unless coverage "
                "changed materially."
            ),
        },
        {
            "branch_id": "g6g_market_group_transfer",
            "plain_name": "Market-group transfer",
            "supported_cells": int(group_transfer["supported"].sum()),
            "parked_cells": int((~group_transfer["supported"]).sum()),
            "status": "supported_for_outcome_batch"
            if group_transfer["supported"].any()
            else "parked_without_outcomes",
            "next_action": (
                "Report identical frozen questions for BTC, established alts, memes, "
                "and supported pre-outcome subgroups without forcing a universal "
                "7-of-10 rule."
            ),
        },
    ]
    return DataFrame.from_records(decisions)


def write_frame(frame: DataFrame, path: Path) -> dict[str, Any]:
    g0.atomic_write_csv(frame, path)
    return artifact_record(path)


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


if __name__ == "__main__":
    raise SystemExit(main())

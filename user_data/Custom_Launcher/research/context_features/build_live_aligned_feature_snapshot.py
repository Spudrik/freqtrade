from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CONTEXT = USER_DATA_DIR / "research_news_data" / "context_features" / "exports" / "context_features_1h_latest.parquet"
DEFAULT_ORDERBOOK_EXPORT_DIR = USER_DATA_DIR / "orderbook_data" / "live" / "exports"
DEFAULT_MARKET_BASKET = USER_DATA_DIR / "market_context_data" / "market_basket_features" / "market_basket_features_wide_freqai_1h.parquet"
DEFAULT_OHLCV_DIR = USER_DATA_DIR / "data" / "binance" / "futures"
DEFAULT_OUTPUT_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "live_aligned_snapshots"
DEFAULT_PAIRS = (
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "TRX/USDT",
    "AVAX/USDT",
    "LINK/USDT",
)
SCHEMA_VERSION = "live_context_orderbook_aligned_features_v2"
ORDERBOOK_STRICT_MIN_COVERAGE = 0.50


CONTEXT_TRANSFORM_PATTERNS = (
    "article_count_24h",
    "news_rows_24h",
    "web_rows_24h",
    "source_count_24h",
    "source_group_count_24h",
    "max_sources_same_topic_24h",
    "topic_count_24h",
    "top_topic_share_6h",
    "fear_greed",
    "btc_dominance",
    "global_",
    "google_trends_",
    "btc_spot_etf",
    "gdelt_",
    "gkg_",
    "intensity_24h",
    "confluence_24h",
    "persistence_hours_24h",
    "severity_max_24h",
    "novelty_z_30d",
    "stack_intensity_24h",
)

ORDERBOOK_TRANSFORM_PATTERNS = (
    "imbalance_top20_mean",
    "imbalance_top20_last",
    "microprice_offset_bps_mean",
    "microprice_offset_bps_last",
    "spread_bps_mean",
    "depth_top20_bid_share",
    "depth_thinness_score",
    "pressure_delta",
    "extreme_pressure_delta",
    "pressure_flip_count",
    "pressure_agreement_ratio",
    "nearest_bid_wall_distance_bps_mean",
    "nearest_ask_wall_distance_bps_mean",
    "nearest_bid_wall_score_mean",
    "nearest_ask_wall_score_mean",
    "nearest_bid_wall_notional_mean",
    "nearest_ask_wall_notional_mean",
    "strongest_bid_wall_score",
    "strongest_ask_wall_score",
    "liquidity_vacuum_up",
    "liquidity_vacuum_down",
    "liquidity_10bps_mean",
    "liquidity_25bps_mean",
    "total_depth_top20_mean",
)

ORDERBOOK_AVAILABILITY_MARKERS = (
    "tick_rows",
    "coverage",
    "ready",
    "max_gap",
    "feature_schema",
    "feature_config",
)


@dataclass(frozen=True)
class SourcePaths:
    context: Path
    orderbook: Path
    market_basket: Path
    ohlcv_dir: Path
    output_dir: Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a feature-only, hour-close-aligned snapshot from recent live context, orderbook, market basket, and OHLCV sources."
    )
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--orderbook", type=Path, default=None, help="Parquet export. Defaults to newest orderbook_features_1h_*.parquet.")
    parser.add_argument("--market-basket", type=Path, default=DEFAULT_MARKET_BASKET)
    parser.add_argument("--ohlcv-dir", type=Path, default=DEFAULT_OHLCV_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pairs", nargs="*", default=list(DEFAULT_PAIRS))
    parser.add_argument("--start", default="", help="Optional UTC bound on aligned hour-close date.")
    parser.add_argument("--end", default="", help="Optional UTC bound on aligned hour-close date.")
    parser.add_argument("--copy-latest", action="store_true", help="Also update live_context_orderbook_aligned_1h_latest.parquet.")
    parser.add_argument("--execute", action="store_true", help="Write the snapshot. Without this, only print the plan.")
    args = parser.parse_args()

    orderbook = args.orderbook or newest_orderbook_export(DEFAULT_ORDERBOOK_EXPORT_DIR)
    paths = SourcePaths(
        context=args.context,
        orderbook=orderbook,
        market_basket=args.market_basket,
        ohlcv_dir=args.ohlcv_dir,
        output_dir=args.output_dir,
    )
    pairs = [normalise_pair(pair) for pair in args.pairs]
    plan = {
        "schema_version": SCHEMA_VERSION,
        "mode": "feature-processing-only",
        "testing_or_future_labels": "not included",
        "timestamp_rule": "OHLCV and market-basket candle-open timestamps are shifted +1h to hour-close before joining context/orderbook features.",
        "market_basket_timestamp_contract": "The market_basket source matches unshifted OHLCV candle-open returns; this builder stores market_basket_open_date and merges the shifted +1h date.",
        "missing_source_rule": "missing source rows stay missing; availability masks are emitted so zero-filled upstream data is not treated as quiet news or neutral orderbook.",
        "orderbook_strict_min_coverage": ORDERBOOK_STRICT_MIN_COVERAGE,
        "pairs": pairs,
        "inputs": input_manifest(paths, pairs),
        "output_dir": str(paths.output_dir),
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0

    frame, metadata = build_snapshot(paths, pairs, parse_bound(args.start), parse_bound(args.end))
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    output = paths.output_dir / f"live_context_orderbook_aligned_1h_{stamp}.parquet"
    validation = paths.output_dir / f"live_context_orderbook_aligned_1h_{stamp}.validation.json"
    columns = paths.output_dir / f"live_context_orderbook_aligned_1h_{stamp}.columns.csv"
    metadata_path = paths.output_dir / f"live_context_orderbook_aligned_1h_{stamp}.meta.json"
    frame.to_parquet(output, index=False)
    column_dictionary(frame).to_csv(columns, index=False)
    metadata = {**plan, **metadata, "output": str(output), "validation": str(validation), "columns_csv": str(columns)}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    validation.write_text(json.dumps(metadata["validation_summary"], indent=2) + "\n", encoding="utf-8")
    latest = None
    if args.copy_latest:
        latest = paths.output_dir / "live_context_orderbook_aligned_1h_latest.parquet"
        shutil.copy2(output, latest)
    print(
        json.dumps(
            {
                "output": str(output),
                "latest": str(latest) if latest else None,
                "metadata": str(metadata_path),
                "validation": str(validation),
                "columns_csv": str(columns),
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "date_min": str(frame["date"].min()) if not frame.empty else None,
                "date_max": str(frame["date"].max()) if not frame.empty else None,
            },
            indent=2,
        )
    )
    return 0


def build_snapshot(paths: SourcePaths, pairs: list[str], start: pd.Timestamp | None, end: pd.Timestamp | None) -> tuple[DataFrame, dict[str, Any]]:
    context = load_context(paths.context)
    orderbook = load_orderbook(paths.orderbook, pairs)
    market = load_market_basket(paths.market_basket)
    per_pair: list[DataFrame] = []
    ohlcv_summaries: dict[str, Any] = {}

    for pair in pairs:
        ohlcv = load_ohlcv(paths.ohlcv_dir, pair)
        ohlcv_summaries[pair] = frame_summary(pair, ohlcv)
        if ohlcv.empty:
            continue
        pair_start = per_pair_start(pair, orderbook, start)
        pair_end = per_pair_end(pair, orderbook, market, context, end)
        if pair_start is not None:
            ohlcv = ohlcv[ohlcv["date"] >= pair_start]
        if pair_end is not None:
            ohlcv = ohlcv[ohlcv["date"] <= pair_end]
        if ohlcv.empty:
            continue
        frame = append_price_features(ohlcv)
        frame = merge_context(frame, context)
        frame = merge_market_basket(frame, market)
        frame = merge_orderbook(frame, orderbook[orderbook["canonical_pair"].eq(pair)].copy())
        frame = append_transforms(frame)
        frame = append_confluence_features(frame)
        frame["snapshot_feature_schema_version"] = SCHEMA_VERSION
        frame["snapshot_generated_at"] = pd.Timestamp.utcnow()
        per_pair.append(frame)

    if per_pair:
        out = pd.concat(per_pair, ignore_index=True)
        out = out.sort_values(["pair", "date"]).drop_duplicates(["pair", "date"], keep="last").reset_index(drop=True)
    else:
        out = DataFrame()

    metadata = {
        "input_summaries": {
            "context": frame_summary("context", context),
            "orderbook": frame_summary("orderbook", orderbook),
            "market_basket": frame_summary("market_basket", market),
            "ohlcv": ohlcv_summaries,
        },
        "validation_summary": validate_snapshot(out, context, orderbook, market),
        "timestamp_contracts": {
            "ohlcv": "source date is candle-open; output date is source date + 1h close",
            "market_basket": "source date is candle-open; output date is source date + 1h close",
            "context": "source date is already feature-hour date; max_source_available_at must be <= output date",
            "orderbook": "source date is feature-hour date; orderbook_source_max_ts must be <= output date",
        },
    }
    return out, metadata


def newest_orderbook_export(export_dir: Path) -> Path:
    candidates = sorted(
        [
            path
            for path in export_dir.glob("orderbook_features_1h_*.parquet")
            if "latest" not in path.name and "alpha" not in path.name and "behaviour" not in path.name
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No orderbook_features_1h_*.parquet exports found in {export_dir}")
    return candidates[0]


def load_ohlcv(ohlcv_dir: Path, pair: str) -> DataFrame:
    symbol = pair.replace("/", "_")
    path = ohlcv_dir / f"{symbol}_USDT-1h-futures.feather"
    if not path.exists():
        return DataFrame(columns=["date", "pair", "open", "high", "low", "close", "volume"])
    frame = pd.read_feather(path, columns=["date", "open", "high", "low", "close", "volume"])
    frame["ohlcv_open_date"] = to_utc(frame["date"])
    frame["date"] = frame["ohlcv_open_date"] + pd.Timedelta(hours=1)
    frame["pair"] = pair
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return frame[["pair", "date", "ohlcv_open_date", "open", "high", "low", "close", "volume"]]


def load_context(path: Path) -> DataFrame:
    frame = pd.read_parquet(path)
    frame["date"] = to_utc(frame["date"])
    for column in ("generated_at", "min_source_available_at", "max_source_available_at"):
        if column in frame:
            frame[column] = to_utc(frame[column])
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    rename = {column: f"ctx_{column}" for column in frame.columns if column != "date" and not column.startswith("ctx_")}
    return frame.rename(columns=rename)


def load_orderbook(path: Path, pairs: list[str]) -> DataFrame:
    frame = pd.read_parquet(path)
    frame["date"] = to_utc(frame["date"])
    frame["canonical_pair"] = frame["canonical_pair"].map(normalise_pair)
    for column in ("source_min_ts", "source_max_ts", "generated_at"):
        if column in frame:
            frame[column] = to_utc(frame[column])
    frame = frame.rename(
        columns={
            "generated_at": "orderbook_generated_at",
            "source_min_ts": "orderbook_source_min_ts",
            "source_max_ts": "orderbook_source_max_ts",
            "raw_tick_rows": "orderbook_raw_tick_rows",
            "market_context_rows": "orderbook_market_context_rows",
            "feature_schema_version": "orderbook_feature_schema_version",
            "feature_config_hash": "orderbook_feature_config_hash",
        }
    )
    frame = frame[frame["canonical_pair"].isin(pairs)].copy()
    return frame.dropna(subset=["date", "canonical_pair"]).sort_values(["canonical_pair", "date"]).drop_duplicates(["canonical_pair", "date"], keep="last").reset_index(drop=True)


def load_market_basket(path: Path) -> DataFrame:
    frame = pd.read_parquet(path)
    frame["market_basket_open_date"] = to_utc(frame["date"])
    frame["date"] = frame["market_basket_open_date"] + pd.Timedelta(hours=1)
    drop = [column for column in frame.columns if column.startswith("&-")]
    frame = frame.drop(columns=drop, errors="ignore")
    return frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def per_pair_start(pair: str, orderbook: DataFrame, explicit: pd.Timestamp | None) -> pd.Timestamp | None:
    if explicit is not None:
        return explicit
    pair_orderbook = orderbook.loc[orderbook["canonical_pair"].eq(pair), "date"]
    if pair_orderbook.empty:
        return None
    return pair_orderbook.min()


def per_pair_end(
    pair: str,
    orderbook: DataFrame,
    market: DataFrame,
    context: DataFrame,
    explicit: pd.Timestamp | None,
) -> pd.Timestamp | None:
    if explicit is not None:
        return explicit
    candidates = []
    pair_orderbook = orderbook.loc[orderbook["canonical_pair"].eq(pair), "date"]
    for values in (pair_orderbook, market.get("date"), context.get("date")):
        if values is not None and len(values):
            candidates.append(values.max())
    return min(candidates) if candidates else None


def append_price_features(frame: DataFrame) -> DataFrame:
    out = frame.copy()
    group = out.groupby("pair", group_keys=False)
    close = group["close"]
    high = group["high"]
    low = group["low"]
    volume = group["volume"]
    out["px_ret_1h"] = close.pct_change()
    for window in (3, 6, 12, 24, 72):
        out[f"px_ret_{window}h"] = close.pct_change(window)
    out["px_log_ret_1h"] = np.log(out["close"]).groupby(out["pair"]).diff()
    out["px_range_pct"] = (out["high"] - out["low"]) / out["close"].replace(0.0, np.nan)
    out["px_body_pct"] = (out["close"] - out["open"]) / out["open"].replace(0.0, np.nan)
    out["px_upper_wick_pct"] = (out["high"] - out[["open", "close"]].max(axis=1)) / out["close"].replace(0.0, np.nan)
    out["px_lower_wick_pct"] = (out[["open", "close"]].min(axis=1) - out["low"]) / out["close"].replace(0.0, np.nan)
    out["px_quote_volume"] = out["close"] * out["volume"]
    out["px_realized_vol_24h"] = group["px_log_ret_1h"].rolling(24, min_periods=12).std().reset_index(level=0, drop=True) * math.sqrt(24.0)
    out["px_volume_z_24h"] = rolling_z(volume, 24, min_periods=12)
    out["px_quote_volume_z_24h"] = rolling_z(group["px_quote_volume"], 24, min_periods=12)
    out["px_range_z_24h"] = rolling_z(group["px_range_pct"], 24, min_periods=12)
    rolling_high = high.rolling(24, min_periods=12).max().reset_index(level=0, drop=True)
    rolling_low = low.rolling(24, min_periods=12).min().reset_index(level=0, drop=True)
    prev_high = group["high"].rolling(24, min_periods=12).max().reset_index(level=0, drop=True).groupby(out["pair"]).shift(1)
    prev_low = group["low"].rolling(24, min_periods=12).min().reset_index(level=0, drop=True).groupby(out["pair"]).shift(1)
    out["px_compression_range_24h"] = (rolling_high - rolling_low) / out["close"].replace(0.0, np.nan)
    out["px_breakout_above_24h_pct"] = (out["close"] - prev_high) / prev_high.replace(0.0, np.nan)
    out["px_breakdown_below_24h_pct"] = (out["close"] - prev_low) / prev_low.replace(0.0, np.nan)
    out["px_volume_expansion_6h_vs_24h"] = (
        volume.rolling(6, min_periods=3).mean().reset_index(level=0, drop=True)
        / volume.rolling(24, min_periods=12).mean().reset_index(level=0, drop=True).replace(0.0, np.nan)
    )
    return out


def merge_context(frame: DataFrame, context: DataFrame) -> DataFrame:
    out = frame.merge(context, on="date", how="left", sort=False)
    out["context_row_present"] = out[[column for column in context.columns if column != "date"]].notna().any(axis=1).astype(float)
    date = to_utc(out["date"])
    max_available = to_utc(out.get("ctx_max_source_available_at"))
    out["context_source_age_hours"] = ((date - max_available).dt.total_seconds() / 3600.0).where(max_available.notna())
    out["context_source_future_violation"] = (max_available.notna() & max_available.gt(date)).astype(float)
    out["context_news_web_present_24h"] = source_activity(out, ["ctx_news_rows_24h", "ctx_web_rows_24h"])
    out["context_gdelt_present_24h"] = source_activity(out, ["ctx_gdelt_event_count_24h", "ctx_gdelt_num_articles_24h"])
    out["context_gkg_present_24h"] = source_activity(out, ["ctx_gkg_doc_count_24h", "ctx_gkg_file_count_1h"])
    out["context_global_present"] = source_activity(out, ["ctx_global_metrics_available"])
    out["context_any_source_raw_present"] = (
        (out["context_news_web_present_24h"] > 0)
        | (out["context_gdelt_present_24h"] > 0)
        | (out["context_gkg_present_24h"] > 0)
        | (out["context_global_present"] > 0)
    ).astype(float)
    out["context_row_time_safe"] = (
        (out["context_row_present"] > 0)
        & max_available.notna()
        & (out["context_source_future_violation"] <= 0)
    ).astype(float)
    out["context_usable_time_safe"] = (
        (out["context_row_time_safe"] > 0)
        & (out["context_any_source_raw_present"] > 0)
    ).astype(float)
    out["context_any_source_present"] = out["context_usable_time_safe"]
    gkg_mask = out["context_gkg_present_24h"].gt(0)
    gkg_columns = [column for column in out.columns if column.startswith("ctx_gkg_") and pd.api.types.is_numeric_dtype(out[column])]
    if gkg_columns:
        out.loc[~gkg_mask, gkg_columns] = np.nan
    out["context_pair_specific_available"] = (
        pair_specific_context_available(out).gt(0)
        & out["context_usable_time_safe"].gt(0)
    ).astype(float)
    out["context_global_only_for_pair"] = (
        (out["context_any_source_present"] > 0)
        & (out["context_pair_specific_available"] <= 0)
    ).astype(float)
    out["context_fresh_6h"] = pd.to_numeric(out["context_source_age_hours"], errors="coerce").between(0, 6, inclusive="both").astype(float)
    out["context_stale_6h_to_24h"] = pd.to_numeric(out["context_source_age_hours"], errors="coerce").gt(6).astype(float) * pd.to_numeric(out["context_source_age_hours"], errors="coerce").le(24).astype(float)
    out["context_very_stale_over_24h"] = pd.to_numeric(out["context_source_age_hours"], errors="coerce").gt(24).astype(float)
    out = mask_context_signal_columns(out)
    return out


def merge_market_basket(frame: DataFrame, market: DataFrame) -> DataFrame:
    out = frame.merge(market, on="date", how="left", sort=False)
    market_columns = [column for column in market.columns if column != "date"]
    out["market_basket_row_present"] = out[market_columns].notna().any(axis=1).astype(float) if market_columns else 0.0
    for column in ("mkt_top10_coverage_ratio", "mkt_btc_coverage_ratio", "mkt_top100_coverage_ratio"):
        if column in out:
            out[f"{column}_available"] = pd.to_numeric(out[column], errors="coerce").gt(0).astype(float)
    return out


def merge_orderbook(frame: DataFrame, orderbook: DataFrame) -> DataFrame:
    if orderbook.empty:
        frame["orderbook_row_present"] = 0.0
        frame["orderbook_tick_present"] = 0.0
        frame["orderbook_usable_loose"] = 0.0
        frame["orderbook_ready_any_market"] = 0.0
        frame["orderbook_source_future_violation"] = 0.0
        return frame
    out = frame.merge(orderbook.drop(columns=["canonical_pair"], errors="ignore"), on="date", how="left", sort=False)
    out["orderbook_row_present"] = out["orderbook_raw_tick_rows"].notna().astype(float) if "orderbook_raw_tick_rows" in out else 0.0
    raw_tick_rows = pd.to_numeric(out.get("orderbook_raw_tick_rows"), errors="coerce").fillna(0.0)
    coverage = pd.to_numeric(out.get("ob1h_x_coverage_mean"), errors="coerce").fillna(0.0)
    ready_count = pd.to_numeric(out.get("ob1h_x_market_ready_count"), errors="coerce").fillna(0.0)
    source_max = to_utc(out.get("orderbook_source_max_ts"))
    date = to_utc(out["date"])
    future = source_max.notna() & source_max.gt(date)
    out["orderbook_source_future_violation"] = future.astype(float)
    out["orderbook_source_age_minutes"] = ((date - source_max).dt.total_seconds() / 60.0).where(source_max.notna())
    out["orderbook_tick_present"] = raw_tick_rows.gt(0).astype(float)
    out["orderbook_usable_loose"] = (raw_tick_rows.gt(0) & coverage.gt(0.0) & source_max.notna() & ~future).astype(float)
    out["orderbook_ready_any_market"] = (ready_count.gt(0) & source_max.notna() & ~future).astype(float)
    out["orderbook_usable_strict"] = (
        ready_count.gt(0)
        & coverage.ge(ORDERBOOK_STRICT_MIN_COVERAGE)
        & source_max.notna()
        & ~future
    ).astype(float)
    out["orderbook_sparse_ticks"] = (raw_tick_rows.gt(0) & ready_count.le(0)).astype(float)
    age = pd.to_numeric(out["orderbook_source_age_minutes"], errors="coerce")
    out["orderbook_fresh_under_5m"] = age.between(0, 5, inclusive="both").astype(float)
    out["orderbook_stale_5m_to_60m"] = age.gt(5).astype(float) * age.le(60).astype(float)
    out["orderbook_very_stale_over_60m"] = age.gt(60).astype(float)
    signal_columns = [
        column
        for column in out.columns
        if column.startswith("ob1h_") and not any(marker in column for marker in ORDERBOOK_AVAILABILITY_MARKERS)
    ]
    if signal_columns:
        strict = out["orderbook_usable_strict"].gt(0)
        out.loc[~strict, signal_columns] = np.nan
    return out


def append_transforms(frame: DataFrame) -> DataFrame:
    out = frame.copy()
    selected = select_transform_columns(out)
    derived: dict[str, Series] = {}
    for column in selected:
        if column not in out or not pd.api.types.is_numeric_dtype(out[column]):
            continue
        safe = safe_name(column)
        active = source_mask_for_column(out, column)
        values = pd.to_numeric(out[column], errors="coerce").where(active)
        segment_group = contiguous_group(values, out["pair"], active)
        diff_1h = segment_group.diff()
        diff_3h = segment_group.diff(3)
        diff_6h = segment_group.diff(6)
        z_24h = grouped_rolling_z(segment_group, 24, min_periods=24)
        previous = segment_group.shift(1)
        current = values
        cross_up_zero = (previous.le(0) & current.gt(0)).astype(float).where(previous.notna() & current.notna())
        cross_down_zero = (previous.ge(0) & current.lt(0)).astype(float).where(previous.notna() & current.notna())
        derived[f"drv_{safe}_chg_1h"] = diff_1h
        derived[f"drv_{safe}_chg_3h"] = diff_3h
        derived[f"drv_{safe}_chg_6h"] = diff_6h
        derived[f"drv_{safe}_z_24h"] = z_24h
        derived[f"drv_{safe}_rising_3h"] = diff_3h.gt(0).astype(float).where(active & diff_3h.notna())
        derived[f"drv_{safe}_falling_3h"] = diff_3h.lt(0).astype(float).where(active & diff_3h.notna())
        derived[f"drv_{safe}_cross_up_zero"] = cross_up_zero.where(active)
        derived[f"drv_{safe}_cross_down_zero"] = cross_down_zero.where(active)
        derived[f"drv_{safe}_extreme_high_z24"] = z_24h.gt(1.5).astype(float).where(active & z_24h.notna())
        derived[f"drv_{safe}_extreme_low_z24"] = z_24h.lt(-1.5).astype(float).where(active & z_24h.notna())
    if not derived:
        return out
    return pd.concat([out, DataFrame(derived, index=out.index)], axis=1).copy()


def select_transform_columns(frame: DataFrame) -> list[str]:
    context = [
        column
        for column in frame.columns
        if column.startswith("ctx_") and any(pattern in column for pattern in CONTEXT_TRANSFORM_PATTERNS)
    ]
    orderbook = [
        column
        for column in frame.columns
        if column.startswith("ob1h_") and any(pattern in column for pattern in ORDERBOOK_TRANSFORM_PATTERNS)
    ]
    return unique(context[:120] + orderbook[:180])


def append_confluence_features(frame: DataFrame) -> DataFrame:
    out = frame.copy()
    out["conf_context_activity_pressure"] = row_mean_rolling_z(out, ["ctx_article_count_24h", "ctx_news_rows_24h", "ctx_web_rows_24h", "ctx_gdelt_num_articles_24h", "ctx_gkg_doc_count_24h"])
    out["conf_context_risk_attention"] = row_mean_rolling_z(
        out,
        [
            "ctx_systemic_risk_stack_intensity_24h",
            "ctx_crypto_stress_stack_intensity_24h",
            "ctx_dollar_risk_off_intensity_24h",
            "ctx_google_trends_market_stress_attention_composite",
            "ctx_google_trends_systemic_risk_attention_composite",
        ],
    )
    out["conf_context_crypto_attention"] = row_mean_rolling_z(
        out,
        [
            "ctx_crypto_market_structure_intensity_24h",
            "ctx_stablecoin_liquidity_intensity_24h",
            "ctx_liquidation_leverage_intensity_24h",
            "ctx_gkg_crypto_doc_count_24h",
            "ctx_google_trends_crypto_attention_composite",
        ],
    )
    out["conf_orderbook_buy_pressure"] = row_mean_rolling_z(
        out,
        [
            "ob1h_binance_spot_imbalance_top20_mean",
            "ob1h_binance_usdm_futures_imbalance_top20_mean",
            "ob1h_bybit_spot_imbalance_top20_mean",
            "ob1h_bybit_linear_imbalance_top20_mean",
            "ob1h_binance_spot_pressure_delta",
            "ob1h_binance_usdm_futures_pressure_delta",
            "ob1h_bybit_spot_pressure_delta",
            "ob1h_bybit_linear_pressure_delta",
        ],
    )
    out["conf_orderbook_sell_pressure"] = -out["conf_orderbook_buy_pressure"]
    out["conf_bid_wall_support"] = row_mean_rolling_z(
        out,
        [
            "ob1h_binance_spot_nearest_bid_wall_score_mean",
            "ob1h_binance_usdm_futures_nearest_bid_wall_score_mean",
            "ob1h_bybit_spot_nearest_bid_wall_score_mean",
            "ob1h_bybit_linear_nearest_bid_wall_score_mean",
        ],
    )
    out["conf_ask_wall_resistance"] = row_mean_rolling_z(
        out,
        [
            "ob1h_binance_spot_nearest_ask_wall_score_mean",
            "ob1h_binance_usdm_futures_nearest_ask_wall_score_mean",
            "ob1h_bybit_spot_nearest_ask_wall_score_mean",
            "ob1h_bybit_linear_nearest_ask_wall_score_mean",
        ],
    )
    out["conf_orderbook_microprice_pressure"] = row_mean_rolling_z(
        out,
        [
            "ob1h_binance_spot_microprice_offset_bps_mean",
            "ob1h_binance_usdm_futures_microprice_offset_bps_mean",
            "ob1h_bybit_spot_microprice_offset_bps_mean",
            "ob1h_bybit_linear_microprice_offset_bps_mean",
        ],
    )
    out["conf_orderbook_pressure_agreement"] = row_mean_rolling_z(
        out,
        [
            "ob1h_binance_spot_pressure_agreement_ratio",
            "ob1h_binance_usdm_futures_pressure_agreement_ratio",
            "ob1h_bybit_spot_pressure_agreement_ratio",
            "ob1h_bybit_linear_pressure_agreement_ratio",
        ],
    )
    out["conf_orderbook_pressure_flip_instability"] = row_mean_rolling_z(
        out,
        [
            "ob1h_binance_spot_pressure_flip_count",
            "ob1h_binance_usdm_futures_pressure_flip_count",
            "ob1h_bybit_spot_pressure_flip_count",
            "ob1h_bybit_linear_pressure_flip_count",
        ],
    )
    out["conf_orderbook_thin_book_fragility"] = row_mean_rolling_z(
        out,
        [
            "ob1h_binance_spot_depth_thinness_score",
            "ob1h_binance_usdm_futures_depth_thinness_score",
            "ob1h_bybit_spot_depth_thinness_score",
            "ob1h_bybit_linear_depth_thinness_score",
        ],
    )
    out["conf_futures_minus_spot_pressure"] = row_mean_rolling_z(
        out,
        [
            "ob1h_x_binance_futures_minus_spot_pressure_delta",
            "ob1h_x_binance_futures_minus_spot_imbalance_top20_mean",
            "ob1h_x_bybit_linear_minus_spot_pressure_delta",
            "ob1h_x_bybit_linear_minus_spot_imbalance_top20_mean",
        ],
    )
    out["conf_liquidity_vacuum_up_venue_count"] = row_positive_count(
        out,
        [
            "ob1h_binance_spot_beh_liquidity_vacuum_up",
            "ob1h_binance_usdm_futures_beh_liquidity_vacuum_up",
            "ob1h_bybit_spot_beh_liquidity_vacuum_up",
            "ob1h_bybit_linear_beh_liquidity_vacuum_up",
        ],
    ).where(pd.to_numeric(out.get("orderbook_usable_strict"), errors="coerce").fillna(0.0).gt(0))
    out["conf_liquidity_vacuum_down_venue_count"] = row_positive_count(
        out,
        [
            "ob1h_binance_spot_beh_liquidity_vacuum_down",
            "ob1h_binance_usdm_futures_beh_liquidity_vacuum_down",
            "ob1h_bybit_spot_beh_liquidity_vacuum_down",
            "ob1h_bybit_linear_beh_liquidity_vacuum_down",
        ],
    ).where(pd.to_numeric(out.get("orderbook_usable_strict"), errors="coerce").fillna(0.0).gt(0))
    out["conf_market_breadth_risk_on"] = row_mean_rolling_z(
        out,
        [
            "%-mkt_top10_pressure_score",
            "%-mkt_top10_breadth_up_6",
            "%-mkt_top10_breadth_up_24",
            "%-mkt_top100_alt_pressure_score",
            "%-mkt_top100_alt_breadth_up_24",
        ],
    )
    out["conf_market_breadth_risk_off"] = row_mean_rolling_z(
        out,
        [
            "%-mkt_top10_breadth_down_6",
            "%-mkt_top10_breadth_down_24",
            "%-mkt_top100_alt_breadth_down_24",
            "%-mkt_top10_alt_pressure_minus_btc",
        ],
    )
    out["conf_context_topic_escalation"] = row_mean_rolling_z(
        out,
        [
            "ctx_high_severity_topic_count_24h",
            "ctx_cross_topic_confluence_max_24h",
            "ctx_cross_topic_persistence_count_24h",
            "ctx_systemic_risk_stack_intensity_24h",
            "ctx_crypto_stress_stack_intensity_24h",
            "ctx_gdelt_conflict_event_count_24h",
        ],
    )
    out["conf_breakout_setup_pressure"] = row_mean_rolling_z(
        out,
        ["px_breakout_above_24h_pct", "px_volume_z_24h", "px_volume_expansion_6h_vs_24h", "conf_orderbook_buy_pressure", "conf_market_breadth_risk_on"],
    )
    out["conf_downside_fragility"] = row_mean_rolling_z(
        out,
        ["px_breakdown_below_24h_pct", "px_volume_z_24h", "conf_orderbook_sell_pressure", "conf_context_risk_attention", "conf_market_breadth_risk_off"],
    )
    ready = pd.to_numeric(out.get("orderbook_usable_strict"), errors="coerce").fillna(0.0).gt(0)
    loose = pd.to_numeric(out.get("orderbook_usable_loose"), errors="coerce").fillna(0.0).gt(0)
    context_safe = pd.to_numeric(out.get("context_usable_time_safe"), errors="coerce").fillna(0.0).gt(0)
    market_safe = pd.to_numeric(out.get("market_basket_row_present"), errors="coerce").fillna(0.0).gt(0)
    out["conf_context_safe_source_family_count"] = (
        pd.to_numeric(out.get("context_news_web_present_24h"), errors="coerce").fillna(0.0).gt(0).astype(int)
        + pd.to_numeric(out.get("context_gdelt_present_24h"), errors="coerce").fillna(0.0).gt(0).astype(int)
        + pd.to_numeric(out.get("context_gkg_present_24h"), errors="coerce").fillna(0.0).gt(0).astype(int)
        + pd.to_numeric(out.get("context_global_present"), errors="coerce").fillna(0.0).gt(0).astype(int)
    ).where(context_safe, 0)
    out["conf_orderbook_loose_available"] = loose.astype(float)
    out["conf_orderbook_ready_market_count"] = pd.to_numeric(out.get("ob1h_x_market_ready_count"), errors="coerce").where(ready)
    out["conf_orderbook_ready_market_ratio"] = pd.to_numeric(out.get("ob1h_x_market_ready_ratio"), errors="coerce").where(ready)
    out["conf_external_safe_source_count"] = (
        context_safe.astype(int)
        + ready.astype(int)
        + market_safe.astype(int)
    )
    out["conf_external_non_market_safe_source_count"] = context_safe.astype(int) + ready.astype(int)
    out["conf_orderbook_buy_pressure_ready_gated"] = out["conf_orderbook_buy_pressure"].where(ready)
    out["conf_orderbook_sell_pressure_ready_gated"] = out["conf_orderbook_sell_pressure"].where(ready)
    out["conf_bid_wall_support_ready_gated"] = out["conf_bid_wall_support"].where(ready)
    out["conf_ask_wall_resistance_ready_gated"] = out["conf_ask_wall_resistance"].where(ready)
    out["state_price_up_bid_pressure_rising"] = (
        out["px_ret_1h"].gt(0)
        & out["conf_orderbook_buy_pressure"].groupby(out["pair"]).diff(3).gt(0)
        & loose
    ).astype(float).where(out["px_ret_1h"].notna() & out["conf_orderbook_buy_pressure"].notna())
    out["state_price_up_ask_wall_weakening"] = (
        out["px_ret_1h"].gt(0)
        & out["conf_ask_wall_resistance"].groupby(out["pair"]).diff(3).lt(0)
        & loose
    ).astype(float).where(out["px_ret_1h"].notna() & out["conf_ask_wall_resistance"].notna())
    out["state_breakout_liquidity_vacuum_up"] = (
        out["px_breakout_above_24h_pct"].gt(0)
        & row_any_positive(out, ["ob1h_binance_spot_beh_liquidity_vacuum_up", "ob1h_binance_usdm_futures_beh_liquidity_vacuum_up", "ob1h_bybit_spot_beh_liquidity_vacuum_up", "ob1h_bybit_linear_beh_liquidity_vacuum_up"])
        & loose
    ).astype(float).where(out["px_breakout_above_24h_pct"].notna())
    out["state_breakdown_liquidity_vacuum_down"] = (
        out["px_breakdown_below_24h_pct"].lt(0)
        & row_any_positive(out, ["ob1h_binance_spot_beh_liquidity_vacuum_down", "ob1h_binance_usdm_futures_beh_liquidity_vacuum_down", "ob1h_bybit_spot_beh_liquidity_vacuum_down", "ob1h_bybit_linear_beh_liquidity_vacuum_down"])
        & loose
    ).astype(float).where(out["px_breakdown_below_24h_pct"].notna())
    out["state_sell_pressure_rising_into_green_price"] = (
        out["px_ret_1h"].gt(0)
        & out["conf_orderbook_sell_pressure"].groupby(out["pair"]).diff(3).gt(0)
        & loose
    ).astype(float).where(out["px_ret_1h"].notna() & out["conf_orderbook_sell_pressure"].notna())
    out["state_ob_buy_pressure_agreement_ready"] = (
        out["conf_orderbook_buy_pressure"].gt(0)
        & out["conf_orderbook_microprice_pressure"].gt(0)
        & out["conf_orderbook_pressure_agreement"].gt(0)
        & ready
    ).astype(float).where(out["conf_orderbook_buy_pressure"].notna())
    out["state_ob_sell_pressure_agreement_ready"] = (
        out["conf_orderbook_sell_pressure"].gt(0)
        & out["conf_orderbook_microprice_pressure"].lt(0)
        & out["conf_orderbook_pressure_agreement"].gt(0)
        & ready
    ).astype(float).where(out["conf_orderbook_sell_pressure"].notna())
    out["state_ob_pressure_flip_instability_ready"] = (
        out["conf_orderbook_pressure_flip_instability"].gt(1.0)
        & ready
    ).astype(float).where(out["conf_orderbook_pressure_flip_instability"].notna())
    out["state_ob_thin_book_fragility_ready"] = (
        out["conf_orderbook_thin_book_fragility"].gt(1.0)
        & ready
    ).astype(float).where(out["conf_orderbook_thin_book_fragility"].notna())
    risk_prev_24h_max = (
        out["conf_context_risk_attention"]
        .groupby(out["pair"])
        .rolling(24, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
        .groupby(out["pair"])
        .shift(1)
    )
    out["state_context_first_risk_escalation"] = (
        out["conf_context_risk_attention"].gt(0.5)
        & risk_prev_24h_max.le(0.25)
        & context_safe
    ).astype(float).where(out["conf_context_risk_attention"].notna())
    out["state_context_risk_relief"] = (
        out["conf_context_risk_attention"].groupby(out["pair"]).diff(6).lt(-0.5)
        & out["conf_market_breadth_risk_on"].gt(0)
        & context_safe
    ).astype(float).where(out["conf_context_risk_attention"].notna())
    out["state_risk_news_downside_orderbook_fragility"] = (
        out["conf_context_risk_attention"].gt(0.5)
        & out["conf_orderbook_sell_pressure"].gt(0)
        & out["conf_orderbook_thin_book_fragility"].gt(0)
        & ready
        & context_safe
    ).astype(float).where(out["conf_context_risk_attention"].notna() & out["conf_orderbook_sell_pressure"].notna())
    out["state_breakout_buy_pressure_breadth_risk_on"] = (
        out["px_breakout_above_24h_pct"].gt(0)
        & out["conf_orderbook_buy_pressure"].gt(0)
        & out["conf_market_breadth_risk_on"].gt(0)
        & ready
    ).astype(float).where(out["px_breakout_above_24h_pct"].notna() & out["conf_orderbook_buy_pressure"].notna())
    out["state_good_context_fades_under_ask_wall"] = (
        out["conf_market_breadth_risk_on"].gt(0)
        & out["conf_context_risk_attention"].lt(0)
        & out["conf_ask_wall_resistance"].gt(0.75)
        & ready
    ).astype(float).where(out["conf_market_breadth_risk_on"].notna() & out["conf_ask_wall_resistance"].notna())
    for column in [
        "state_price_up_bid_pressure_rising",
        "state_price_up_ask_wall_weakening",
        "state_breakout_liquidity_vacuum_up",
        "state_breakdown_liquidity_vacuum_down",
        "state_sell_pressure_rising_into_green_price",
    ]:
        out[f"{column}_ready_gated"] = out[column].where(ready)
    return out


def validate_snapshot(frame: DataFrame, context: DataFrame, orderbook: DataFrame, market: DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0, "warnings": ["aligned snapshot is empty"]}
    numeric = frame.select_dtypes(include=["number", "bool"]).columns
    grouped = frame.groupby("pair")
    per_pair = {}
    for pair, data in grouped:
        dates = to_utc(data["date"]).sort_values()
        gaps = dates.diff().dropna()
        per_pair[pair] = {
            "rows": int(len(data)),
            "first_date": str(dates.min()),
            "last_date": str(dates.max()),
            "duplicate_dates": int(dates.duplicated().sum()),
            "hourly_gap_count": int(gaps.gt(pd.Timedelta(hours=1)).sum()),
            "context_any_source_rows": int(pd.to_numeric(data.get("context_any_source_present"), errors="coerce").fillna(0.0).gt(0).sum()),
            "orderbook_tick_rows": int(pd.to_numeric(data.get("orderbook_tick_present"), errors="coerce").fillna(0.0).gt(0).sum()),
            "orderbook_ready_rows": int(pd.to_numeric(data.get("orderbook_ready_any_market"), errors="coerce").fillna(0.0).gt(0).sum()),
            "market_basket_rows": int(pd.to_numeric(data.get("market_basket_row_present"), errors="coerce").fillna(0.0).gt(0).sum()),
        }
    context_future = int(pd.to_numeric(frame.get("context_source_future_violation"), errors="coerce").fillna(0.0).gt(0).sum())
    orderbook_future = int(pd.to_numeric(frame.get("orderbook_source_future_violation"), errors="coerce").fillna(0.0).gt(0).sum())
    unsafe_context_cells = unsafe_numeric_signal_cells(frame, "ctx_", "context_usable_time_safe", keep={"ctx_generated_at", "ctx_min_source_available_at", "ctx_max_source_available_at", "ctx_feature_config_hash"})
    unsafe_orderbook_cells = unsafe_numeric_signal_cells(frame, "ob1h_", "orderbook_usable_strict")
    active_context_missing_available_at = int(
        (
            pd.to_numeric(frame.get("context_any_source_present"), errors="coerce").fillna(0.0).gt(0)
            & to_utc(frame.get("ctx_max_source_available_at")).isna()
        ).sum()
    )
    duplicate_pair_dates = int(frame.duplicated(["pair", "date"]).sum())
    warnings = []
    if duplicate_pair_dates:
        warnings.append("duplicate pair/date rows are present")
    if context_future:
        warnings.append("context source timestamp is after aligned feature date on some rows")
    if orderbook_future:
        warnings.append("orderbook source timestamp is after aligned feature date on some rows")
    if unsafe_context_cells:
        warnings.append("unsafe context signal cells are populated outside context_usable_time_safe rows")
    if unsafe_orderbook_cells:
        warnings.append("unsafe orderbook signal cells are populated outside strict orderbook usable rows")
    if active_context_missing_available_at:
        warnings.append("active context rows are missing max_source_available_at")
    if len(numeric):
        inf_cells = int(np.isinf(frame[numeric].to_numpy(dtype=float, na_value=np.nan)).sum())
    else:
        inf_cells = 0
    if inf_cells:
        warnings.append("infinite numeric cells are present")
    aligned_max = to_utc(frame["date"]).max()
    context_extra = int(to_utc(context["date"]).gt(aligned_max).sum()) if not context.empty else 0
    orderbook_extra = int(to_utc(orderbook["date"]).gt(aligned_max).sum()) if not orderbook.empty else 0
    market_extra = int(to_utc(market["date"]).gt(aligned_max).sum()) if not market.empty else 0
    if context_extra or orderbook_extra:
        warnings.append("external source rows exist after the latest aligned OHLCV/market row; they are parked until price data catches up")
    return {
        "schema_version": SCHEMA_VERSION,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "pairs": int(frame["pair"].nunique()),
        "first_date": str(to_utc(frame["date"]).min()),
        "last_date": str(aligned_max),
        "duplicate_pair_dates": duplicate_pair_dates,
        "numeric_columns": int(len(numeric)),
        "non_numeric_columns": int(len(frame.columns) - len(numeric)),
        "infinite_numeric_cells": inf_cells,
        "context_source_future_violations": context_future,
        "orderbook_source_future_violations": orderbook_future,
        "unsafe_context_numeric_signal_cells": unsafe_context_cells,
        "unsafe_orderbook_numeric_signal_cells": unsafe_orderbook_cells,
        "active_context_rows_missing_max_source_available_at": active_context_missing_available_at,
        "orderbook_usable_rows": {
            "tick_present": int(pd.to_numeric(frame.get("orderbook_tick_present"), errors="coerce").fillna(0.0).gt(0).sum()),
            "loose": int(pd.to_numeric(frame.get("orderbook_usable_loose"), errors="coerce").fillna(0.0).gt(0).sum()),
            "strict": int(pd.to_numeric(frame.get("orderbook_usable_strict"), errors="coerce").fillna(0.0).gt(0).sum()),
            "ready_any_market": int(pd.to_numeric(frame.get("orderbook_ready_any_market"), errors="coerce").fillna(0.0).gt(0).sum()),
            "strict_min_coverage": ORDERBOOK_STRICT_MIN_COVERAGE,
        },
        "source_rows_after_aligned_price_end": {
            "context": context_extra,
            "orderbook": orderbook_extra,
            "market_basket": market_extra,
        },
        "per_pair": per_pair,
        "warnings": warnings,
    }


def frame_summary(name: str, frame: DataFrame) -> dict[str, Any]:
    if frame.empty or "date" not in frame:
        return {"name": name, "rows": int(len(frame)), "columns": int(len(frame.columns)), "first_date": None, "last_date": None}
    dates = to_utc(frame["date"])
    summary = {
        "name": name,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "first_date": str(dates.min()),
        "last_date": str(dates.max()),
        "duplicate_dates": int(dates.duplicated().sum()),
    }
    if "canonical_pair" in frame:
        summary["duplicate_pair_dates"] = int(frame.duplicated(["canonical_pair", "date"]).sum())
    if "pair" in frame:
        summary["duplicate_pair_dates"] = int(frame.duplicated(["pair", "date"]).sum())
    return summary


def unsafe_numeric_signal_cells(frame: DataFrame, prefix: str, safe_column: str, keep: set[str] | None = None) -> int:
    keep = keep or set()
    if safe_column not in frame:
        return 0
    unsafe = pd.to_numeric(frame[safe_column], errors="coerce").fillna(0.0).le(0)
    columns = [
        column
        for column in frame.columns
        if column.startswith(prefix)
        and column not in keep
        and pd.api.types.is_numeric_dtype(frame[column])
        and not any(token in column for token in ("present", "available", "ready", "coverage", "violation", "age", "fresh", "stale", "sparse", "tick_rows", "valid_tick_rows", "max_gap", "message_count"))
    ]
    if not columns:
        return 0
    return int(frame.loc[unsafe, columns].notna().sum().sum())


def column_dictionary(frame: DataFrame) -> DataFrame:
    rows = []
    for column in frame.columns:
        family = column_family(column)
        if column in ("pair", "date", "ohlcv_open_date", "market_basket_open_date", "snapshot_generated_at", "snapshot_feature_schema_version"):
            role = "identifier"
        elif any(token in column for token in ("present", "violation", "usable", "ready", "coverage", "available", "fresh", "stale", "age", "sparse", "tick_rows", "valid_tick_rows", "max_gap", "message_count")):
            role = "availability_or_safety_mask"
        elif column.startswith("drv_"):
            role = "derived_transform"
        elif column.startswith("conf_"):
            role = "trader_confluence_feature"
        elif column.startswith("state_"):
            role = "trader_state_feature"
        elif column.startswith("px_"):
            role = "price_volume_feature"
        else:
            role = "source_feature"
        numeric = bool(pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column]))
        panel = feature_panel(column, family, role)
        rows.append(
            {
                "column": column,
                "family": family,
                "role": role,
                "feature_panel": panel,
                "dtype": str(frame[column].dtype),
                "freqai_candidate_numeric": bool(numeric and role in {"price_volume_feature", "derived_transform", "trader_confluence_feature", "trader_state_feature"}),
            }
        )
    return DataFrame(rows)


def input_manifest(paths: SourcePaths, pairs: list[str]) -> dict[str, Any]:
    manifest = {
        "context": file_manifest(paths.context),
        "orderbook": file_manifest(paths.orderbook),
        "market_basket": file_manifest(paths.market_basket),
    }
    manifest["ohlcv"] = {}
    for pair in pairs:
        symbol = pair.replace("/", "_")
        manifest["ohlcv"][pair] = file_manifest(paths.ohlcv_dir / f"{symbol}_USDT-1h-futures.feather")
    return manifest


def file_manifest(path: Path) -> dict[str, Any]:
    item = {"path": str(path), "exists": bool(path.exists())}
    if path.exists():
        stat = path.stat()
        item["size_bytes"] = int(stat.st_size)
        item["mtime_utc"] = str(pd.Timestamp(stat.st_mtime, unit="s", tz="UTC"))
        item["sha256_16"] = sha256_prefix(path)
    return item


def sha256_prefix(path: Path, max_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        remaining = max_bytes
        while remaining > 0:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()[:16]


def source_activity(frame: DataFrame, columns: list[str]) -> Series:
    existing = [column for column in columns if column in frame]
    if not existing:
        return Series(0.0, index=frame.index)
    return frame[existing].apply(pd.to_numeric, errors="coerce").fillna(0.0).gt(0).any(axis=1).astype(float)


def mask_context_signal_columns(frame: DataFrame) -> DataFrame:
    safe = pd.to_numeric(frame.get("context_usable_time_safe"), errors="coerce").fillna(0.0).gt(0)
    keep = {
        "ctx_generated_at",
        "ctx_min_source_available_at",
        "ctx_max_source_available_at",
        "ctx_feature_config_hash",
    }
    for column in [column for column in frame.columns if column.startswith("ctx_") and column not in keep]:
        if pd.api.types.is_numeric_dtype(frame[column]):
            frame.loc[~safe, column] = np.nan
    return frame


def pair_specific_context_available(frame: DataFrame) -> Series:
    pair = frame["pair"].astype(str).str.upper()
    btc = pair.eq("BTC/USDT") & source_activity(frame, ["ctx_btc_mentions_6h", "ctx_gkg_bitcoin_doc_count_24h"]).gt(0)
    eth = pair.eq("ETH/USDT") & source_activity(frame, ["ctx_eth_mentions_6h", "ctx_gkg_ethereum_doc_count_24h"]).gt(0)
    return (btc | eth).astype(float)


def source_mask_for_column(frame: DataFrame, column: str) -> Series:
    if column.startswith("ctx_gkg_"):
        return pd.to_numeric(frame.get("context_gkg_present_24h"), errors="coerce").fillna(0.0).gt(0)
    if column.startswith("ctx_"):
        return pd.to_numeric(frame.get("context_usable_time_safe"), errors="coerce").fillna(0.0).gt(0)
    if column.startswith("ob1h_"):
        return pd.to_numeric(frame.get("orderbook_usable_strict"), errors="coerce").fillna(0.0).gt(0)
    if column.startswith("%-mkt") or column.startswith("mkt_"):
        return pd.to_numeric(frame.get("market_basket_row_present"), errors="coerce").fillna(0.0).gt(0)
    return Series(True, index=frame.index)


def row_any_positive(frame: DataFrame, columns: list[str]) -> Series:
    existing = [column for column in columns if column in frame]
    if not existing:
        return Series(False, index=frame.index)
    return frame[existing].apply(pd.to_numeric, errors="coerce").gt(0).any(axis=1)


def row_positive_count(frame: DataFrame, columns: list[str]) -> Series:
    existing = [column for column in columns if column in frame]
    if not existing:
        return Series(np.nan, index=frame.index)
    return frame[existing].apply(pd.to_numeric, errors="coerce").gt(0).sum(axis=1)


def contiguous_group(values: Series, pair: Series, active: Series) -> Any:
    clean_active = active.reindex(values.index).fillna(False).astype(bool)
    segments = (~clean_active).astype(int).groupby(pair).cumsum()
    return values.where(clean_active).groupby([pair, segments], group_keys=False)


def grouped_rolling_z(grouped: Any, window: int, min_periods: int) -> Series:
    mean = grouped.rolling(window, min_periods=min_periods).mean().reset_index(level=[0, 1], drop=True)
    std = grouped.rolling(window, min_periods=min_periods).std().reset_index(level=[0, 1], drop=True).replace(0.0, np.nan)
    raw = grouped.obj if hasattr(grouped, "obj") else grouped
    return (raw - mean) / std


def row_mean_rolling_z(frame: DataFrame, columns: list[str], window: int = 168, min_periods: int = 24) -> Series:
    existing = [column for column in columns if column in frame and pd.api.types.is_numeric_dtype(frame[column])]
    if not existing:
        return Series(np.nan, index=frame.index)
    z_values: list[Series] = []
    for column in existing:
        values = pd.to_numeric(frame[column], errors="coerce")
        active = source_mask_for_column(frame, column)
        z_values.append(grouped_rolling_z(contiguous_group(values, frame["pair"], active), window, min_periods=min_periods))
    return pd.concat(z_values, axis=1).mean(axis=1, skipna=True)


def rolling_z(grouped: Any, window: int, min_periods: int) -> Series:
    mean = grouped.rolling(window, min_periods=min_periods).mean().reset_index(level=0, drop=True)
    std = grouped.rolling(window, min_periods=min_periods).std().reset_index(level=0, drop=True).replace(0.0, np.nan)
    raw = grouped.obj if hasattr(grouped, "obj") else grouped
    return (raw - mean) / std


def to_utc(values: Any) -> Series:
    return pd.to_datetime(values, utc=True, errors="coerce", format="mixed")


def parse_bound(value: str) -> pd.Timestamp | None:
    if not value:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def normalise_pair(pair: str) -> str:
    return pair.strip().upper().replace("-", "/").replace("_", "/").replace("//", "/")


def safe_name(column: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", column).strip("_").lower()


def unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def column_family(column: str) -> str:
    if column in ("pair", "date", "ohlcv_open_date", "market_basket_open_date"):
        return "id"
    if column.startswith("px_") or column in ("open", "high", "low", "close", "volume"):
        return "price_volume"
    if column.startswith("ctx_") or column.startswith("context_"):
        return "context_news_global"
    if column.startswith("ob1h_") or column.startswith("orderbook_"):
        return "orderbook"
    if column.startswith("%-mkt") or column.startswith("mkt_") or column.startswith("market_basket"):
        return "market_basket"
    if column.startswith("drv_"):
        return "derived_transform"
    if column.startswith("conf_"):
        return "confluence"
    if column.startswith("state_"):
        return "trader_state"
    return "other"


def feature_panel(column: str, family: str, role: str) -> str:
    if role == "identifier":
        return "metadata"
    if role == "availability_or_safety_mask":
        if family == "orderbook":
            return "availability_orderbook"
        if family == "context_news_global":
            return "availability_context"
        return "availability_general"
    if column.startswith("state_"):
        if "context" in column or "news" in column:
            return "fs_context_state"
        if "orderbook" in column or "_ob_" in column or "wall" in column or "pressure" in column or "liquidity" in column:
            return "fs_orderbook_state"
        return "fs_cross_confluence"
    if column.startswith("conf_"):
        if "context" in column:
            return "fs_context_escalation"
        if "orderbook" in column or "wall" in column or "liquidity" in column or "futures_minus_spot" in column:
            return "fs_orderbook_pressure_walls"
        if "market" in column:
            return "fs_market_breadth"
        return "fs_cross_confluence"
    if column.startswith("px_") or column in ("open", "high", "low", "close", "volume"):
        return "fs_price_volume"
    if column.startswith("drv_ctx_"):
        return "fs_context_transforms"
    if column.startswith("drv_ob1h_"):
        if "wall" in column:
            return "fs_orderbook_wall_transforms"
        if "liquidity" in column or "vacuum" in column or "depth" in column or "thinness" in column:
            return "fs_orderbook_liquidity_transforms"
        if "pressure" in column or "imbalance" in column or "microprice" in column:
            return "fs_orderbook_pressure_transforms"
        return "fs_orderbook_transforms"
    if column.startswith("drv_"):
        return "fs_derived_transforms"
    if family == "market_basket":
        return "fs_market_breadth_raw"
    if family == "context_news_global":
        return "raw_context_not_default_candidate"
    if family == "orderbook":
        return "raw_orderbook_not_default_candidate"
    return "not_default_candidate"


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


USER_DATA_DIR = Path(__file__).resolve().parents[3]
REPO_ROOT = USER_DATA_DIR.parent
DEFAULT_DB = USER_DATA_DIR / "orderbook_data" / "live" / "orderbook_events.sqlite"
DEFAULT_OHLCV = USER_DATA_DIR / "data" / "binance" / "futures" / "BTC_USDT_USDT-1h-futures.feather"
DEFAULT_OUTPUT_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "orderbook_timeframe_value"
DEFAULT_PAIR = "BTC/USDT"
DEFAULT_DB_PAIR = "BTC/USDT:USDT"
MARKETS = ("binance_spot", "binance_usdm_futures", "bybit_spot", "bybit_linear")
RESOLUTIONS = ("1s", "1m", "5m", "1h")
BAR_SECONDS = {"1m": 60, "5m": 300, "1h": 3600}


TICK_COLUMNS = [
    "ts",
    "market_key",
    "book_valid",
    "message_count_interval",
    "spread_bps",
    "microprice_offset_bps",
    "imbalance_top20",
    "imbalance_10bps",
    "imbalance_25bps",
    "bid_liquidity_5bps",
    "ask_liquidity_5bps",
    "nearest_bid_wall_distance_bps",
    "nearest_ask_wall_distance_bps",
    "nearest_bid_wall_score",
    "nearest_ask_wall_score",
    "strongest_bid_wall_score_50bps",
    "strongest_ask_wall_score_50bps",
    "strong_bid_pressure",
    "strong_ask_pressure",
    "extreme_bid_pressure",
    "extreme_ask_pressure",
]


BAR_COLUMNS = [
    "ts_start",
    "ts_end",
    "timeframe_seconds",
    "market_key",
    "valid_samples",
    "expected_samples",
    "spread_bps_mean",
    "spread_bps_max",
    "microprice_offset_bps_mean",
    "imbalance_top20_mean",
    "imbalance_top20_min",
    "imbalance_top20_max",
    "imbalance_10bps_mean",
    "imbalance_25bps_mean",
    "bid_pressure_seconds",
    "ask_pressure_seconds",
    "bid_pressure_ratio",
    "ask_pressure_ratio",
    "nearest_bid_wall_min_distance_bps",
    "nearest_ask_wall_min_distance_bps",
    "strongest_bid_wall_score",
    "strongest_ask_wall_score",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build matched orderbook timeframe comparison features.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--ohlcv", type=Path, default=DEFAULT_OHLCV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pair", default=DEFAULT_PAIR)
    parser.add_argument("--db-pair", default=DEFAULT_DB_PAIR)
    parser.add_argument("--start", default="2026-05-08T00:00:00+00:00")
    parser.add_argument("--end", default="2026-06-22T00:00:00+00:00")
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--strict-coverage", type=float, default=0.50)
    args = parser.parse_args()

    if not args.db.exists():
        raise FileNotFoundError(args.db)
    if not args.ohlcv.exists():
        raise FileNotFoundError(args.ohlcv)

    start = pd.Timestamp(args.start).tz_convert("UTC") if pd.Timestamp(args.start).tzinfo else pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end).tz_convert("UTC") if pd.Timestamp(args.end).tzinfo else pd.Timestamp(args.end, tz="UTC")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prices = load_prices(args.ohlcv, start, end)
    with connect(args.db) as conn:
        tick_frame = load_ticks(conn, args.db_pair, start, end)
        bar_frames = {name: load_bars(conn, args.db_pair, seconds, start, end) for name, seconds in BAR_SECONDS.items()}

    pieces = [prices]
    diagnostics: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "db": str(args.db),
        "ohlcv": str(args.ohlcv),
        "pair": args.pair,
        "db_pair": args.db_pair,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "min_coverage": args.min_coverage,
        "strict_coverage": args.strict_coverage,
    }

    one_second = aggregate_ticks_to_hour(tick_frame, args.min_coverage, args.strict_coverage)
    pieces.append(one_second)
    diagnostics["1s"] = frame_diagnostics(one_second, "obtf_1s")
    for name, frame in bar_frames.items():
        hourly = aggregate_bars_to_hour(frame, name, args.min_coverage, args.strict_coverage)
        pieces.append(hourly)
        diagnostics[name] = frame_diagnostics(hourly, f"obtf_{name}")

    merged = pieces[0]
    for piece in pieces[1:]:
        merged = merged.merge(piece, on="date", how="left")
    merged["pair"] = args.pair
    ready_columns = [f"obtf_{name}_usable" for name in RESOLUTIONS if f"obtf_{name}_usable" in merged]
    strict_columns = [f"obtf_{name}_strict" for name in RESOLUTIONS if f"obtf_{name}_strict" in merged]
    merged["obtf_all_usable"] = all_true(merged, ready_columns)
    merged["obtf_all_strict"] = all_true(merged, strict_columns)
    merged = append_cross_resolution_diagnostics(merged)
    merged = merged.replace([np.inf, -np.inf], np.nan)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parquet_path = args.output_dir / f"orderbook_timeframe_value_{stamp}.parquet"
    latest_path = args.output_dir / "orderbook_timeframe_value_latest.parquet"
    meta_path = args.output_dir / f"orderbook_timeframe_value_{stamp}.meta.json"
    latest_meta_path = args.output_dir / "orderbook_timeframe_value_latest.meta.json"
    columns_path = args.output_dir / f"orderbook_timeframe_value_{stamp}.columns.csv"
    latest_columns_path = args.output_dir / "orderbook_timeframe_value_latest.columns.csv"

    merged.to_parquet(parquet_path, index=False)
    merged.to_parquet(latest_path, index=False)
    columns = column_dictionary(merged)
    columns.to_csv(columns_path, index=False)
    columns.to_csv(latest_columns_path, index=False)
    diagnostics["rows"] = int(len(merged))
    diagnostics["date_min"] = str(merged["date"].min()) if not merged.empty else None
    diagnostics["date_max"] = str(merged["date"].max()) if not merged.empty else None
    diagnostics["all_usable_rows"] = int(pd.to_numeric(merged["obtf_all_usable"], errors="coerce").fillna(0).gt(0).sum())
    diagnostics["all_strict_rows"] = int(pd.to_numeric(merged["obtf_all_strict"], errors="coerce").fillna(0).gt(0).sum())
    diagnostics["parquet_path"] = str(parquet_path)
    diagnostics["latest_path"] = str(latest_path)
    meta_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    latest_meta_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))
    return 0


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA query_only=ON")
    return conn


def load_prices(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> DataFrame:
    frame = pd.read_feather(path, columns=["date", "open", "high", "low", "close", "volume"])
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce") + pd.Timedelta(hours=1)
    frame = frame[(frame["date"] >= start) & (frame["date"] < end)].copy()
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    close = pd.to_numeric(frame["close"], errors="coerce").replace(0.0, np.nan)
    frame["px_ret_1h"] = close.pct_change(1, fill_method=None)
    frame["px_ret_2h"] = close.pct_change(2, fill_method=None)
    frame["px_ret_4h"] = close.pct_change(4, fill_method=None)
    frame["px_ret_6h"] = close.pct_change(6, fill_method=None)
    volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    log_volume = np.log1p(volume)
    frame["px_volume_z_24h"] = (log_volume - log_volume.shift(1).rolling(24, min_periods=8).mean()) / log_volume.shift(1).rolling(24, min_periods=8).std().replace(0.0, np.nan)
    return frame.reset_index(drop=True)


def load_ticks(conn: sqlite3.Connection, db_pair: str, start: pd.Timestamp, end: pd.Timestamp) -> DataFrame:
    sql = f"""
        SELECT {', '.join(TICK_COLUMNS)}
        FROM orderbook_metric_ticks INDEXED BY idx_orderbook_metric_ticks_pair_ts
        WHERE pair = ? AND ts >= ? AND ts < ?
        ORDER BY ts
    """
    return pd.read_sql_query(sql, conn, params=(db_pair, start.isoformat(), end.isoformat()))


def load_bars(conn: sqlite3.Connection, db_pair: str, timeframe_seconds: int, start: pd.Timestamp, end: pd.Timestamp) -> DataFrame:
    sql = f"""
        SELECT {', '.join(BAR_COLUMNS)}
        FROM orderbook_metric_bars INDEXED BY idx_orderbook_metric_bars_pair_tf_ts
        WHERE pair = ? AND timeframe_seconds = ? AND ts_start >= ? AND ts_start < ?
        ORDER BY ts_start
    """
    return pd.read_sql_query(sql, conn, params=(db_pair, timeframe_seconds, start.isoformat(), end.isoformat()))


def aggregate_ticks_to_hour(frame: DataFrame, min_coverage: float, strict_coverage: float) -> DataFrame:
    if frame.empty:
        return empty_resolution("1s")
    out = frame.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out = out.dropna(subset=["ts", "market_key"])
    out["date"] = out["ts"].dt.floor("h") + pd.Timedelta(hours=1)
    out["book_valid"] = pd.to_numeric(out["book_valid"], errors="coerce").fillna(0.0)
    out["pressure_delta"] = pd.to_numeric(out["strong_bid_pressure"], errors="coerce").fillna(0.0) - pd.to_numeric(out["strong_ask_pressure"], errors="coerce").fillna(0.0)
    out["extreme_pressure_delta"] = pd.to_numeric(out["extreme_bid_pressure"], errors="coerce").fillna(0.0) - pd.to_numeric(out["extreme_ask_pressure"], errors="coerce").fillna(0.0)
    market = out.groupby(["date", "market_key"], sort=True).apply(aggregate_tick_market, include_groups=False).reset_index()
    return aggregate_resolution_market_rows(market, "1s", min_coverage, strict_coverage)


def aggregate_tick_market(group: DataFrame) -> Series:
    valid = group[pd.to_numeric(group["book_valid"], errors="coerce").fillna(0.0).gt(0.0)]
    result = {
        "sample_rows": float(len(group)),
        "valid_samples": float(len(valid)),
        "expected_samples": 3600.0,
        "coverage": float(len(valid) / 3600.0),
        "message_count_sum": safe_sum(group.get("message_count_interval")),
        "pressure_delta_mean": safe_mean(valid.get("pressure_delta")),
        "pressure_delta_max_abs": safe_absmax(valid.get("pressure_delta")),
        "extreme_pressure_delta_mean": safe_mean(valid.get("extreme_pressure_delta")),
        "spread_bps_mean": safe_mean(valid.get("spread_bps")),
        "spread_bps_max": safe_max(valid.get("spread_bps")),
        "microprice_offset_bps_mean": safe_mean(valid.get("microprice_offset_bps")),
        "imbalance_top20_mean": safe_mean(valid.get("imbalance_top20")),
        "imbalance_top20_std": safe_std(valid.get("imbalance_top20")),
        "imbalance_10bps_mean": safe_mean(valid.get("imbalance_10bps")),
        "imbalance_25bps_mean": safe_mean(valid.get("imbalance_25bps")),
        "liquidity_delta_5bps_mean": safe_mean(pd.to_numeric(valid.get("bid_liquidity_5bps"), errors="coerce") - pd.to_numeric(valid.get("ask_liquidity_5bps"), errors="coerce")),
        "nearest_bid_wall_distance_min": safe_min(valid.get("nearest_bid_wall_distance_bps")),
        "nearest_ask_wall_distance_min": safe_min(valid.get("nearest_ask_wall_distance_bps")),
        "bid_wall_score_max": safe_max(valid.get("strongest_bid_wall_score_50bps")),
        "ask_wall_score_max": safe_max(valid.get("strongest_ask_wall_score_50bps")),
    }
    result["wall_support_resistance_delta"] = result["bid_wall_score_max"] - result["ask_wall_score_max"] if np.isfinite(result["bid_wall_score_max"]) and np.isfinite(result["ask_wall_score_max"]) else np.nan
    return pd.Series(result)


def aggregate_bars_to_hour(frame: DataFrame, resolution: str, min_coverage: float, strict_coverage: float) -> DataFrame:
    if frame.empty:
        return empty_resolution(resolution)
    out = frame.copy()
    out["ts_end"] = pd.to_datetime(out["ts_end"], utc=True, errors="coerce")
    out = out.dropna(subset=["ts_end", "market_key"])
    out["date"] = out["ts_end"].dt.floor("h") + pd.Timedelta(hours=1)
    out["pressure_delta"] = pd.to_numeric(out["bid_pressure_ratio"], errors="coerce").fillna(0.0) - pd.to_numeric(out["ask_pressure_ratio"], errors="coerce").fillna(0.0)
    out["wall_support_resistance_delta"] = pd.to_numeric(out["strongest_bid_wall_score"], errors="coerce") - pd.to_numeric(out["strongest_ask_wall_score"], errors="coerce")
    market = out.groupby(["date", "market_key"], sort=True).apply(aggregate_bar_market, include_groups=False).reset_index()
    return aggregate_resolution_market_rows(market, resolution, min_coverage, strict_coverage)


def aggregate_bar_market(group: DataFrame) -> Series:
    valid_samples = pd.to_numeric(group.get("valid_samples"), errors="coerce").fillna(0.0)
    expected_samples = pd.to_numeric(group.get("expected_samples"), errors="coerce").fillna(0.0)
    result = {
        "sample_rows": float(len(group)),
        "valid_samples": float(valid_samples.sum()),
        "expected_samples": float(expected_samples.sum()),
        "coverage": float(valid_samples.sum() / expected_samples.sum()) if expected_samples.sum() else np.nan,
        "message_count_sum": np.nan,
        "pressure_delta_mean": safe_mean(group.get("pressure_delta")),
        "pressure_delta_max_abs": safe_absmax(group.get("pressure_delta")),
        "extreme_pressure_delta_mean": np.nan,
        "spread_bps_mean": safe_mean(group.get("spread_bps_mean")),
        "spread_bps_max": safe_max(group.get("spread_bps_max")),
        "microprice_offset_bps_mean": safe_mean(group.get("microprice_offset_bps_mean")),
        "imbalance_top20_mean": safe_mean(group.get("imbalance_top20_mean")),
        "imbalance_top20_std": safe_std(group.get("imbalance_top20_mean")),
        "imbalance_10bps_mean": safe_mean(group.get("imbalance_10bps_mean")),
        "imbalance_25bps_mean": safe_mean(group.get("imbalance_25bps_mean")),
        "liquidity_delta_5bps_mean": np.nan,
        "nearest_bid_wall_distance_min": safe_min(group.get("nearest_bid_wall_min_distance_bps")),
        "nearest_ask_wall_distance_min": safe_min(group.get("nearest_ask_wall_min_distance_bps")),
        "bid_wall_score_max": safe_max(group.get("strongest_bid_wall_score")),
        "ask_wall_score_max": safe_max(group.get("strongest_ask_wall_score")),
    }
    result["wall_support_resistance_delta"] = safe_mean(group.get("wall_support_resistance_delta"))
    return pd.Series(result)


def aggregate_resolution_market_rows(frame: DataFrame, resolution: str, min_coverage: float, strict_coverage: float) -> DataFrame:
    if frame.empty:
        return empty_resolution(resolution)
    p = f"obtf_{resolution}"
    frame = frame.copy()
    frame["usable_market"] = pd.to_numeric(frame["coverage"], errors="coerce").fillna(0.0).ge(min_coverage)
    frame["strict_market"] = pd.to_numeric(frame["coverage"], errors="coerce").fillna(0.0).ge(strict_coverage)
    numeric_columns = [
        "sample_rows",
        "valid_samples",
        "expected_samples",
        "coverage",
        "message_count_sum",
        "pressure_delta_mean",
        "pressure_delta_max_abs",
        "extreme_pressure_delta_mean",
        "spread_bps_mean",
        "spread_bps_max",
        "microprice_offset_bps_mean",
        "imbalance_top20_mean",
        "imbalance_top20_std",
        "imbalance_10bps_mean",
        "imbalance_25bps_mean",
        "liquidity_delta_5bps_mean",
        "nearest_bid_wall_distance_min",
        "nearest_ask_wall_distance_min",
        "bid_wall_score_max",
        "ask_wall_score_max",
        "wall_support_resistance_delta",
    ]
    rows: list[dict[str, Any]] = []
    for date, group in frame.groupby("date", sort=True):
        row: dict[str, Any] = {"date": date}
        row[f"{p}_usable"] = float(group["usable_market"].any())
        row[f"{p}_strict"] = float(group["strict_market"].any())
        row[f"{p}_usable_market_count"] = float(group["usable_market"].sum())
        row[f"{p}_strict_market_count"] = float(group["strict_market"].sum())
        row[f"{p}_coverage_mean"] = safe_mean(group["coverage"])
        row[f"{p}_coverage_max"] = safe_max(group["coverage"])
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{p}_{column}_xmean"] = safe_mean(values)
            row[f"{p}_{column}_xmaxabs"] = safe_absmax(values)
        pressure = pd.to_numeric(group["pressure_delta_mean"], errors="coerce").dropna()
        row[f"{p}_pressure_agreement"] = sign_agreement(pressure)
        row[f"{p}_wall_delta_agreement"] = sign_agreement(pd.to_numeric(group["wall_support_resistance_delta"], errors="coerce").dropna())
        for market in MARKETS:
            market_rows = group[group["market_key"].astype(str).eq(market)]
            if market_rows.empty:
                continue
            suffix = safe_col(market)
            row[f"{p}_{suffix}_coverage"] = safe_max(market_rows["coverage"])
            row[f"{p}_{suffix}_pressure_delta"] = safe_mean(market_rows["pressure_delta_mean"])
            row[f"{p}_{suffix}_spread_bps"] = safe_mean(market_rows["spread_bps_mean"])
            row[f"{p}_{suffix}_imbalance_top20"] = safe_mean(market_rows["imbalance_top20_mean"])
            row[f"{p}_{suffix}_wall_delta"] = safe_mean(market_rows["wall_support_resistance_delta"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def append_cross_resolution_diagnostics(frame: DataFrame) -> DataFrame:
    out = frame.copy()
    base_features = (
        "pressure_delta_mean_xmean",
        "spread_bps_mean_xmean",
        "imbalance_top20_mean_xmean",
        "wall_support_resistance_delta_xmean",
    )
    for feature in base_features:
        one = pd.to_numeric(out.get(f"obtf_1s_{feature}"), errors="coerce")
        for other in ("1m", "5m", "1h"):
            value = pd.to_numeric(out.get(f"obtf_{other}_{feature}"), errors="coerce")
            out[f"obtf_cmp_1s_minus_{other}_{feature}"] = one - value
    return out


def all_true(frame: DataFrame, columns: list[str]) -> Series:
    if not columns:
        return pd.Series(0.0, index=frame.index)
    mask = pd.Series(True, index=frame.index)
    for column in columns:
        mask &= pd.to_numeric(frame[column], errors="coerce").fillna(0.0).gt(0.0)
    return mask.astype(float)


def empty_resolution(resolution: str) -> DataFrame:
    return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns, UTC]"), f"obtf_{resolution}_usable": pd.Series(dtype=float), f"obtf_{resolution}_strict": pd.Series(dtype=float)})


def frame_diagnostics(frame: DataFrame, prefix: str) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0, "usable_rows": 0, "strict_rows": 0}
    usable = pd.to_numeric(frame.get(f"{prefix}_usable"), errors="coerce").fillna(0).gt(0)
    strict = pd.to_numeric(frame.get(f"{prefix}_strict"), errors="coerce").fillna(0).gt(0)
    return {
        "rows": int(len(frame)),
        "date_min": str(frame["date"].min()),
        "date_max": str(frame["date"].max()),
        "usable_rows": int(usable.sum()),
        "strict_rows": int(strict.sum()),
        "coverage_mean_nonzero": safe_mean(pd.to_numeric(frame.get(f"{prefix}_coverage_mean"), errors="coerce").replace(0.0, np.nan)),
        "coverage_max": safe_max(frame.get(f"{prefix}_coverage_max")),
    }


def column_dictionary(frame: DataFrame) -> DataFrame:
    rows = []
    for column in frame.columns:
        if column in {"date", "pair", "open", "high", "low", "close", "volume"}:
            role = "base"
        elif column.startswith("obtf_cmp_"):
            role = "cross_resolution_diagnostic"
        elif column.endswith("_usable") or column.endswith("_strict") or "coverage" in column:
            role = "availability_mask"
        elif column.startswith("obtf_"):
            role = "orderbook_feature"
        elif column.startswith("px_"):
            role = "price_control_feature"
        else:
            role = "other"
        rows.append({"column": column, "role": role, "nonnull": int(frame[column].notna().sum())})
    return pd.DataFrame(rows)


def safe_col(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_")


def to_numeric(series: Series | None) -> Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def safe_mean(series: Series | None) -> float:
    values = to_numeric(series).dropna()
    return float(values.mean()) if not values.empty else np.nan


def safe_std(series: Series | None) -> float:
    values = to_numeric(series).dropna()
    return float(values.std()) if len(values) > 1 else np.nan


def safe_min(series: Series | None) -> float:
    values = to_numeric(series).dropna()
    return float(values.min()) if not values.empty else np.nan


def safe_max(series: Series | None) -> float:
    values = to_numeric(series).dropna()
    return float(values.max()) if not values.empty else np.nan


def safe_sum(series: Series | None) -> float:
    values = to_numeric(series).dropna()
    return float(values.sum()) if not values.empty else 0.0


def safe_absmax(series: Series | None) -> float:
    values = to_numeric(series).dropna()
    return float(values.abs().max()) if not values.empty else np.nan


def sign_agreement(series: Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    signs = np.sign(values.to_numpy(dtype=float))
    signs = signs[signs != 0]
    if signs.size == 0:
        return 0.0
    return float(abs(signs.sum()) / signs.size)


if __name__ == "__main__":
    raise SystemExit(main())

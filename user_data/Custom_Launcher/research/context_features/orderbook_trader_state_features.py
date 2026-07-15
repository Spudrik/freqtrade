from __future__ import annotations

import argparse
import json
import math
import zipfile
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = Path("D:/FreqTradeStuffLargeData/orderbook_data/historical_bybit/spot_raw")
DEFAULT_OUTPUT_DIR = USER_DATA_DIR / "orderbook_data" / "historical_bybit" / "features"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "orderbook_trader_state_1h.parquet"
DEFAULT_LATEST = DEFAULT_OUTPUT_DIR / "orderbook_trader_state_1h_latest.parquet"

DEPTH_BANDS_BPS = (5, 10, 25, 50, 100)
SCHEMA_VERSION = "trader_state_v1"


@dataclass
class BuildSummary:
    raw_dir: str
    output: str
    latest: str | None
    files_planned: int
    files_processed: int = 0
    sample_rows_1m: int = 0
    zone_rows_5m: int = 0
    feature_rows_1h: int = 0
    first_feature_date: str | None = None
    last_feature_date: str | None = None
    warnings: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = self.__dict__.copy()
        payload["warnings"] = self.warnings or []
        return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build stateful trader-style Bybit orderbook features from raw archive ZIPs."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--pattern", default="*_BTCUSDT_ob200.data.zip")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--latest", type=Path, default=DEFAULT_LATEST)
    parser.add_argument("--canonical-pair", default="BTC/USDT")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--market-key", default="bybit_spot")
    parser.add_argument("--market-id", type=float, default=1.0)
    parser.add_argument("--sample-seconds", type=int, default=60)
    parser.add_argument("--zone-bps", type=float, default=10.0)
    parser.add_argument("--wall-lookout-bps", type=float, default=100.0)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--recompute-second-pass-from", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.recompute_second_pass_from:
        features = pd.read_parquet(args.recompute_second_pass_from)
        features["market_key"] = str(args.market_key)
        features["obts_feature_present"] = 1.0
        features["obts_market_id"] = float(args.market_id)
        features = _append_trader_state_second_pass(features)
        features["date"] = pd.to_datetime(features["date"], utc=True)
        features["source_min_ts"] = pd.to_datetime(features["source_min_ts"], utc=True)
        features["source_max_ts"] = pd.to_datetime(features["source_max_ts"], utc=True)
        summary = BuildSummary(
            raw_dir=str(args.raw_dir),
            output=str(args.output),
            latest=str(args.latest) if args.latest else None,
            files_planned=0,
            files_processed=0,
            feature_rows_1h=int(len(features)),
            first_feature_date=str(features["date"].min()) if not features.empty else None,
            last_feature_date=str(features["date"].max()) if not features.empty else None,
            warnings=[f"Recomputed second-pass trader-state columns from {args.recompute_second_pass_from}."],
        )
        if args.dry_run:
            print(json.dumps(summary.to_dict(), indent=2))
            return 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(args.output)
        if args.latest:
            args.latest.parent.mkdir(parents=True, exist_ok=True)
            features.to_parquet(args.latest)
        print(json.dumps(summary.to_dict(), indent=2))
        return 0

    files = _selected_archives(args.raw_dir, args.pattern, args.start, args.end)
    if args.limit_files is not None:
        files = files[: max(0, args.limit_files)]
    summary = BuildSummary(
        raw_dir=str(args.raw_dir),
        output=str(args.output),
        latest=str(args.latest) if args.latest else None,
        files_planned=len(files),
        warnings=[],
    )
    if args.dry_run:
        print(json.dumps(summary.to_dict(), indent=2))
        return 0
    if not files:
        summary.warnings.append("No raw Bybit archives matched the requested selection.")
        print(json.dumps(summary.to_dict(), indent=2))
        return 0

    samples, files_processed = build_1m_samples_from_archives(
        files,
        sample_seconds=args.sample_seconds,
        zone_bps=args.zone_bps,
        wall_lookout_bps=args.wall_lookout_bps,
    )
    summary.files_processed = files_processed
    summary.sample_rows_1m = int(len(samples))
    if samples.empty:
        summary.warnings.append("No valid 1m orderbook samples were produced.")
        print(json.dumps(summary.to_dict(), indent=2))
        return 0

    zones_5m = build_5m_zone_states(samples)
    features = build_1h_trader_state_features(
        samples,
        zones_5m,
        canonical_pair=args.canonical_pair,
        market_key=args.market_key,
        market_id=float(args.market_id),
        sample_seconds=args.sample_seconds,
        zone_bps=args.zone_bps,
        wall_lookout_bps=args.wall_lookout_bps,
    )
    summary.zone_rows_5m = int(len(zones_5m))
    summary.feature_rows_1h = int(len(features))
    if not features.empty:
        summary.first_feature_date = str(features["date"].min())
        summary.last_feature_date = str(features["date"].max())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output)
    if args.latest:
        args.latest.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(args.latest)
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


def _selected_archives(raw_dir: Path, pattern: str, start: str, end: str) -> list[Path]:
    files = sorted(Path(raw_dir).glob(pattern))
    if not start and not end:
        return files
    start_ts = pd.Timestamp(start, tz="UTC") if start else None
    end_ts = pd.Timestamp(end, tz="UTC") if end else None
    selected: list[Path] = []
    for path in files:
        day = _archive_day(path)
        if day is None:
            continue
        if start_ts is not None and day < start_ts.normalize():
            continue
        if end_ts is not None and day >= end_ts.normalize():
            continue
        selected.append(path)
    return selected


def _archive_day(path: Path) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(path.name[:10], tz="UTC")
    except Exception:
        return None


def build_1m_samples_from_archives(
    files: Iterable[Path],
    *,
    sample_seconds: int = 60,
    zone_bps: float = 10.0,
    wall_lookout_bps: float = 100.0,
) -> tuple[DataFrame, int]:
    sample_ms = max(1, int(sample_seconds)) * 1000
    bid_levels: list[float] = []
    ask_levels: list[float] = []
    bid_sizes: dict[float, float] = {}
    ask_sizes: dict[float, float] = {}
    next_sample_ms: int | None = None
    last_event_ms: int | None = None
    zone_width_price: float | None = None
    rows: list[dict[str, Any]] = []
    processed_files = 0

    for path in sorted(files):
        processed_files += 1
        bid_levels = []
        ask_levels = []
        bid_sizes = {}
        ask_sizes = {}
        next_sample_ms = None
        last_event_ms = None
        with zipfile.ZipFile(path) as zipf:
            names = zipf.namelist()
            if not names:
                continue
            with zipf.open(names[0]) as fp:
                for raw_line in fp:
                    event = json.loads(raw_line)
                    event_ms = int(event.get("cts") or event.get("ts"))
                    if next_sample_ms is not None and last_event_ms is not None:
                        while next_sample_ms <= event_ms:
                            row = _book_sample(
                                next_sample_ms,
                                bid_levels,
                                bid_sizes,
                                ask_levels,
                                ask_sizes,
                                zone_width_price,
                                zone_bps=zone_bps,
                                wall_lookout_bps=wall_lookout_bps,
                            )
                            if row:
                                rows.append(row)
                                if zone_width_price is None:
                                    zone_width_price = row["zone_width_price"]
                            next_sample_ms += sample_ms

                    if event["type"] == "snapshot":
                        bid_levels, bid_sizes = _reset_book_side(event["data"].get("b") or [])
                        ask_levels, ask_sizes = _reset_book_side(event["data"].get("a") or [])
                        next_sample_ms = ((event_ms // sample_ms) + 1) * sample_ms
                    else:
                        _apply_delta(bid_levels, bid_sizes, event["data"].get("b") or [])
                        _apply_delta(ask_levels, ask_sizes, event["data"].get("a") or [])
                    last_event_ms = event_ms

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, processed_files
    frame["date"] = pd.to_datetime(frame["sample_ms"], unit="ms", utc=True)
    frame = frame.drop(columns=["sample_ms"]).sort_values("date").reset_index(drop=True)
    return frame, processed_files


def _reset_book_side(levels: list[list[str]]) -> tuple[list[float], dict[float, float]]:
    prices: list[float] = []
    sizes: dict[float, float] = {}
    for price_raw, size_raw in levels:
        price = float(price_raw)
        size = float(size_raw)
        if size <= 0:
            continue
        prices.append(price)
        sizes[price] = size
    prices.sort()
    return prices, sizes


def _apply_delta(price_levels: list[float], sizes: dict[float, float], updates: list[list[str]]) -> None:
    for price_raw, size_raw in updates:
        price = float(price_raw)
        size = float(size_raw)
        if price in sizes:
            if size <= 0:
                del sizes[price]
                idx = bisect_left(price_levels, price)
                if idx < len(price_levels) and price_levels[idx] == price:
                    del price_levels[idx]
            else:
                sizes[price] = size
        elif size > 0:
            sizes[price] = size
            price_levels.insert(bisect_left(price_levels, price), price)


def _book_sample(
    sample_ms: int,
    bid_levels: list[float],
    bid_sizes: dict[float, float],
    ask_levels: list[float],
    ask_sizes: dict[float, float],
    zone_width_price: float | None,
    *,
    zone_bps: float,
    wall_lookout_bps: float,
) -> dict[str, Any] | None:
    if not bid_levels or not ask_levels:
        return None
    best_bid = bid_levels[-1]
    best_ask = ask_levels[0]
    if best_bid <= 0 or best_ask <= 0 or best_ask < best_bid:
        return None
    mid = (best_bid + best_ask) / 2.0
    if zone_width_price is None:
        zone_width_price = max(1e-9, mid * zone_bps / 10000.0)

    bid = _side_snapshot(
        bid_levels,
        bid_sizes,
        mid,
        side="bid",
        zone_width_price=zone_width_price,
        wall_lookout_bps=wall_lookout_bps,
    )
    ask = _side_snapshot(
        ask_levels,
        ask_sizes,
        mid,
        side="ask",
        zone_width_price=zone_width_price,
        wall_lookout_bps=wall_lookout_bps,
    )
    bid_top = bid_levels[-1]
    ask_top = ask_levels[0]
    bid_top_notional = bid_top * bid_sizes[bid_top]
    ask_top_notional = ask_top * ask_sizes[ask_top]
    top_total = bid_top_notional + ask_top_notional
    microprice = (
        ((ask_top * bid_top_notional) + (bid_top * ask_top_notional)) / top_total
        if top_total > 0.0
        else mid
    )
    row: dict[str, Any] = {
        "sample_ms": sample_ms,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid_price": mid,
        "spread_bps": (best_ask - best_bid) / mid * 10000.0,
        "microprice_offset_bps": (microprice - mid) / mid * 10000.0,
        "zone_width_price": zone_width_price,
    }
    for band in DEPTH_BANDS_BPS:
        bid_depth = bid[f"depth_{band}bps"]
        ask_depth = ask[f"depth_{band}bps"]
        row[f"bid_depth_{band}bps"] = bid_depth
        row[f"ask_depth_{band}bps"] = ask_depth
        row[f"imbalance_{band}bps"] = _ratio(bid_depth - ask_depth, bid_depth + ask_depth)
    for side_name, side_data in (("bid", bid), ("ask", ask)):
        for key, value in side_data.items():
            if key.startswith("depth_"):
                continue
            row[f"{side_name}_{key}"] = value
    return row


def _side_snapshot(
    levels: list[float],
    sizes: dict[float, float],
    mid: float,
    *,
    side: str,
    zone_width_price: float,
    wall_lookout_bps: float,
) -> dict[str, Any]:
    ordered = levels[::-1] if side == "bid" else levels
    depth_by_band = {f"depth_{band}bps": 0.0 for band in DEPTH_BANDS_BPS}
    records: list[dict[str, float]] = []
    for price in ordered:
        distance_bps = abs(price - mid) / mid * 10000.0
        if distance_bps > max(max(DEPTH_BANDS_BPS), wall_lookout_bps):
            break
        notional = price * sizes[price]
        for band in DEPTH_BANDS_BPS:
            if distance_bps <= band:
                depth_by_band[f"depth_{band}bps"] += notional
        if distance_bps <= wall_lookout_bps:
            records.append({"price": price, "distance_bps": distance_bps, "notional": notional})

    candidates = _wall_candidates(records, mid, zone_width_price)
    strongest = max(candidates, key=lambda item: item["score"], default={})
    nearest = min(candidates, key=lambda item: item["distance_bps"], default={})
    out: dict[str, Any] = dict(depth_by_band)
    out.update(
        {
            "wall_candidate_count": float(len(candidates)),
            "wall_candidates_json": json.dumps(candidates, separators=(",", ":")),
            "strongest_wall_price": _nan_if_missing(strongest.get("price")),
            "strongest_wall_zone": _nan_if_missing(strongest.get("zone_key")),
            "strongest_wall_distance_bps": _nan_if_missing(strongest.get("distance_bps")),
            "strongest_wall_notional": _nan_if_missing(strongest.get("notional")),
            "strongest_wall_score": _nan_if_missing(strongest.get("score")),
            "nearest_wall_price": _nan_if_missing(nearest.get("price")),
            "nearest_wall_zone": _nan_if_missing(nearest.get("zone_key")),
            "nearest_wall_distance_bps": _nan_if_missing(nearest.get("distance_bps")),
            "nearest_wall_notional": _nan_if_missing(nearest.get("notional")),
            "nearest_wall_score": _nan_if_missing(nearest.get("score")),
        }
    )
    return out


def _wall_candidates(records: list[dict[str, float]], mid: float, zone_width_price: float) -> list[dict[str, float]]:
    if not records:
        return []
    notionals = np.array([record["notional"] for record in records], dtype=float)
    if not np.isfinite(notionals).any():
        return []
    q90 = float(np.nanquantile(notionals, 0.90)) if len(notionals) >= 5 else float(np.nanmax(notionals))
    q95 = float(np.nanquantile(notionals, 0.95)) if len(notionals) >= 10 else q90
    threshold = max(q90, q95 * 0.80)
    floor = max(q90, 1.0)
    candidates = [record for record in records if record["notional"] >= threshold]
    if len(candidates) < 3:
        candidates = sorted(records, key=lambda item: item["notional"], reverse=True)[:3]
    output = []
    for record in sorted(candidates, key=lambda item: item["notional"], reverse=True)[:8]:
        zone_key = round(record["price"] / zone_width_price) * zone_width_price
        output.append(
            {
                "price": float(record["price"]),
                "zone_key": float(zone_key),
                "distance_bps": float(record["distance_bps"]),
                "notional": float(record["notional"]),
                "score": float(min(10.0, record["notional"] / floor)),
            }
        )
    return output


def build_5m_zone_states(samples: DataFrame) -> DataFrame:
    if samples.empty:
        return DataFrame()
    records: list[dict[str, Any]] = []
    for row in samples.itertuples(index=False):
        ts = getattr(row, "date")
        mid = float(getattr(row, "mid_price"))
        for side in ("bid", "ask"):
            raw = getattr(row, f"{side}_wall_candidates_json")
            for candidate in _json_records(raw):
                zone = _float(candidate.get("zone_key"))
                score = _float(candidate.get("score"))
                price = _float(candidate.get("price"))
                if math.isnan(zone) or math.isnan(score) or math.isnan(price):
                    continue
                records.append(
                    {
                        "date_1m": ts,
                        "date_5m": pd.Timestamp(ts).floor("5min") + pd.Timedelta(minutes=5),
                        "side": side,
                        "zone_key": zone,
                        "zone_price": price,
                        "distance_bps": abs(price - mid) / mid * 10000.0 if mid else math.nan,
                        "notional": _float(candidate.get("notional")),
                        "strength": score,
                    }
                )
    if not records:
        return DataFrame()
    frame = pd.DataFrame(records)
    frame["date_1m"] = pd.to_datetime(frame["date_1m"], utc=True, errors="coerce")
    frame["date_5m"] = pd.to_datetime(frame["date_5m"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date_1m", "date_5m", "zone_key", "strength"])
    rows = []
    for (date_5m, side, zone_key), group in frame.groupby(["date_5m", "side", "zone_key"], sort=True):
        ordered = group.sort_values("date_1m")
        rows.append(
            {
                "date_5m": date_5m,
                "side": side,
                "zone_key": float(zone_key),
                "zone_price_mean": float(ordered["zone_price"].mean()),
                "zone_price_last": float(ordered["zone_price"].iloc[-1]),
                "distance_bps_min": float(ordered["distance_bps"].min()),
                "distance_bps_last": float(ordered["distance_bps"].iloc[-1]),
                "notional_max": float(ordered["notional"].max()),
                "notional_mean": float(ordered["notional"].mean()),
                "strength_max": float(ordered["strength"].max()),
                "strength_mean": float(ordered["strength"].mean()),
                "strength_last": float(ordered["strength"].iloc[-1]),
                "seen_minutes": float(len(ordered)),
            }
        )
    return pd.DataFrame(rows).sort_values(["date_5m", "side", "zone_key"]).reset_index(drop=True)


def build_1h_trader_state_features(
    samples: DataFrame,
    zones_5m: DataFrame,
    *,
    canonical_pair: str,
    market_key: str,
    market_id: float,
    sample_seconds: int,
    zone_bps: float,
    wall_lookout_bps: float,
) -> DataFrame:
    prepared = samples.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], utc=True, errors="coerce")
    prepared = prepared.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    prepared["hour"] = prepared["date"].dt.floor("h") + pd.Timedelta(hours=1)
    rows: list[dict[str, Any]] = []
    for hour, group in prepared.groupby("hour", sort=True):
        ordered = group.sort_values("date")
        row: dict[str, Any] = {
            "date": hour,
            "canonical_pair": canonical_pair,
            "market_key": market_key,
            "source_min_ts": ordered["date"].min(),
            "source_max_ts": ordered["date"].max(),
            "obts_schema_version": SCHEMA_VERSION,
            "obts_feature_present": 1.0,
            "obts_market_id": float(market_id),
            "obts_sample_rows_1m": float(len(ordered)),
            "obts_expected_samples_1h": float(3600 / max(1, sample_seconds)),
            "obts_coverage_ratio": float(min(1.0, len(ordered) / max(1.0, 3600 / max(1, sample_seconds)))),
            "obts_zone_bps": float(zone_bps),
            "obts_wall_lookout_bps": float(wall_lookout_bps),
        }
        row.update(_hourly_sample_stats(ordered))
        row.update(_hourly_zone_features(zones_5m, pd.Timestamp(hour), row))
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if frame.empty:
        return frame
    frame = _append_trader_state_second_pass(frame)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["source_min_ts"] = pd.to_datetime(frame["source_min_ts"], utc=True)
    frame["source_max_ts"] = pd.to_datetime(frame["source_max_ts"], utc=True)
    return frame


def _hourly_sample_stats(group: DataFrame) -> dict[str, float]:
    out: dict[str, float] = {
        "obts_mid_open": float(group["mid_price"].iloc[0]),
        "obts_mid_high": float(group["mid_price"].max()),
        "obts_mid_low": float(group["mid_price"].min()),
        "obts_mid_close": float(group["mid_price"].iloc[-1]),
        "obts_spread_bps_mean": _mean(group["spread_bps"]),
        "obts_spread_bps_p95": _quantile(group["spread_bps"], 0.95),
        "obts_spread_bps_max": _max(group["spread_bps"]),
        "obts_microprice_offset_bps_mean": _mean(group["microprice_offset_bps"]),
        "obts_microprice_offset_bps_last": _last(group["microprice_offset_bps"]),
    }
    pressure = pd.to_numeric(group["imbalance_25bps"], errors="coerce").fillna(0.0)
    out.update(
        {
            "obts_pressure_25bps_mean": float(pressure.mean()),
            "obts_pressure_25bps_last": float(pressure.iloc[-1]),
            "obts_pressure_25bps_min": float(pressure.min()),
            "obts_pressure_25bps_max": float(pressure.max()),
            "obts_pressure_25bps_slope": _slope(pressure),
            "obts_pressure_flip_count_1h": float(_sign_flip_count(pressure.to_numpy())),
        }
    )
    for band in DEPTH_BANDS_BPS:
        bid = pd.to_numeric(group[f"bid_depth_{band}bps"], errors="coerce")
        ask = pd.to_numeric(group[f"ask_depth_{band}bps"], errors="coerce")
        out[f"obts_bid_depth_{band}bps_mean"] = _mean(bid)
        out[f"obts_ask_depth_{band}bps_mean"] = _mean(ask)
        out[f"obts_bid_depth_{band}bps_min"] = _min(bid)
        out[f"obts_ask_depth_{band}bps_min"] = _min(ask)
        out[f"obts_imbalance_{band}bps_mean"] = _mean(group[f"imbalance_{band}bps"])
        out[f"obts_imbalance_{band}bps_last"] = _last(group[f"imbalance_{band}bps"])
    for side in ("bid", "ask"):
        out[f"obts_{side}_wall_candidate_count_mean"] = _mean(group[f"{side}_wall_candidate_count"])
        for kind in ("strongest", "nearest"):
            for suffix in ("wall_price", "wall_zone", "wall_distance_bps", "wall_notional", "wall_score"):
                column = f"{side}_{kind}_{suffix}"
                out[f"obts_{side}_{kind}_{suffix}_last"] = _last(group[column])
                out[f"obts_{side}_{kind}_{suffix}_max"] = _max(group[column])
                out[f"obts_{side}_{kind}_{suffix}_min"] = _min(group[column])
    return out


def _hourly_zone_features(zones_5m: DataFrame, hour: pd.Timestamp, row: dict[str, Any]) -> dict[str, float]:
    if zones_5m.empty:
        return _empty_zone_features()
    close = _float(row.get("obts_mid_close"))
    high = _float(row.get("obts_mid_high"))
    low = _float(row.get("obts_mid_low"))
    out: dict[str, float] = {}
    resistance = _select_zone(zones_5m, hour, "ask", close, high, low)
    support = _select_zone(zones_5m, hour, "bid", close, high, low)
    for label, payload in (("resistance", resistance), ("support", support)):
        for key, value in payload.items():
            out[f"obts_{label}_{key}"] = value
    res_price = out.get("obts_resistance_zone_price", math.nan)
    sup_price = out.get("obts_support_zone_price", math.nan)
    out["obts_zone_width_bps"] = (
        (res_price - sup_price) / close * 10000.0
        if close and not math.isnan(res_price) and not math.isnan(sup_price)
        else math.nan
    )
    out["obts_zone_compression_score"] = _clip01(1.0 - _safe_div(out["obts_zone_width_bps"], 150.0))
    return out


def _empty_zone_features() -> dict[str, float]:
    out: dict[str, float] = {"obts_zone_width_bps": math.nan, "obts_zone_compression_score": 0.0}
    for label in ("resistance", "support"):
        for key in _zone_feature_keys():
            out[f"obts_{label}_{key}"] = math.nan if key.endswith(("price", "distance_bps", "age_hours")) else 0.0
    return out


def _zone_feature_keys() -> tuple[str, ...]:
    return (
        "zone_price",
        "zone_distance_bps",
        "zone_age_hours",
        "zone_strength",
        "zone_persistence_6h",
        "zone_persistence_24h",
        "zone_stability_score",
        "zone_added_strength_1h",
        "zone_removed_strength_1h",
        "zone_evaporation_1h",
        "zone_current_seen_5m",
        "zone_current_strength",
        "zone_recent_strength",
        "zone_near_touch_count_1h",
    )


def _select_zone(zones_5m: DataFrame, hour: pd.Timestamp, side: str, close: float, high: float, low: float) -> dict[str, float]:
    empty = {key: math.nan if key.endswith(("price", "distance_bps", "age_hours")) else 0.0 for key in _zone_feature_keys()}
    if math.isnan(close) or close <= 0.0:
        return empty
    start_24h = hour - pd.Timedelta(hours=24)
    start_6h = hour - pd.Timedelta(hours=6)
    start_1h = hour - pd.Timedelta(hours=1)
    start_prev = hour - pd.Timedelta(hours=2)
    hist = zones_5m[(zones_5m["side"].eq(side)) & (zones_5m["date_5m"] > start_24h) & (zones_5m["date_5m"] <= hour)].copy()
    if hist.empty:
        return empty
    current = hist[hist["date_5m"] > start_1h]
    prev = hist[(hist["date_5m"] > start_prev) & (hist["date_5m"] <= start_1h)]
    grouped = []
    for zone_key, group in hist.groupby("zone_key", sort=False):
        ordered = group.sort_values("date_5m")
        last = ordered.iloc[-1]
        last_seen = pd.Timestamp(last["date_5m"])
        price = float(last["zone_price_last"])
        if side == "ask" and price < close:
            continue
        if side == "bid" and price > close:
            continue
        seen_6h = ordered[ordered["date_5m"] > start_6h]
        seen_1h = ordered[ordered["date_5m"] > start_1h]
        persistence_6h = len(seen_6h) / 72.0
        persistence_24h = len(ordered) / 288.0
        recency = _clip01(1.0 - ((hour - last_seen).total_seconds() / 3600.0) / 6.0)
        price_std_bps = (
            ordered["zone_price_last"].std(ddof=0) / close * 10000.0
            if len(ordered) > 1 and close
            else 0.0
        )
        stability = _clip01(persistence_6h * (1.0 - price_std_bps / 50.0))
        strength = _clip01(math.log1p(float(ordered["strength_mean"].tail(12).mean())) / math.log1p(10.0))
        strength = _row_score(strength, persistence_6h, persistence_24h, recency, stability)
        distance = abs(price - close) / close * 10000.0
        grouped.append(
            {
                "zone_key": float(zone_key),
                "price": price,
                "distance": distance,
                "age_hours": max(0.0, (last_seen - pd.Timestamp(ordered["date_5m"].min())).total_seconds() / 3600.0),
                "strength": strength,
                "persistence_6h": persistence_6h,
                "persistence_24h": persistence_24h,
                "stability": stability,
                "recency": recency,
                "current_seen": float(len(seen_1h)),
                "current_strength": float(seen_1h["strength_mean"].mean()) if not seen_1h.empty else 0.0,
                "recent_strength": float(ordered["strength_mean"].tail(12).mean()),
            }
        )
    if not grouped:
        return empty
    candidates = pd.DataFrame(grouped)
    active = candidates[candidates["current_seen"] > 0.0].copy()
    if active.empty:
        active = candidates[candidates["recency"] > 0.0].copy()
    if active.empty:
        return empty
    active["selection_score"] = (
        active["strength"] * 0.45
        + active["stability"] * 0.25
        + _clip01_series(1.0 - active["distance"] / 150.0) * 0.30
    )
    selected = active.sort_values(["selection_score", "distance"], ascending=[False, True]).iloc[0]
    curr_strength_by_zone = current.groupby("zone_key")["strength_mean"].mean() if not current.empty else Series(dtype=float)
    prev_strength_by_zone = prev.groupby("zone_key")["strength_mean"].mean() if not prev.empty else Series(dtype=float)
    added = 0.0
    removed = 0.0
    for zone_key, curr_strength in curr_strength_by_zone.items():
        added += max(0.0, float(curr_strength) - float(prev_strength_by_zone.get(zone_key, 0.0)))
    for zone_key, prev_strength in prev_strength_by_zone.items():
        removed += max(0.0, float(prev_strength) - float(curr_strength_by_zone.get(zone_key, 0.0)))
    normalizer = max(1.0, added + removed)
    touch_count = 0.0
    zone_price = float(selected["price"])
    if side == "ask":
        touch_count = 1.0 if high >= zone_price * (1.0 - 5.0 / 10000.0) else 0.0
    else:
        touch_count = 1.0 if low <= zone_price * (1.0 + 5.0 / 10000.0) else 0.0
    return {
        "zone_price": zone_price,
        "zone_distance_bps": float(selected["distance"]),
        "zone_age_hours": float(selected["age_hours"]),
        "zone_strength": float(selected["strength"]),
        "zone_persistence_6h": float(selected["persistence_6h"]),
        "zone_persistence_24h": float(selected["persistence_24h"]),
        "zone_stability_score": float(selected["stability"]),
        "zone_added_strength_1h": _clip01(added / normalizer),
        "zone_removed_strength_1h": _clip01(removed / normalizer),
        "zone_evaporation_1h": _clip01((removed / normalizer) * (1.0 - float(selected["current_seen"]) / 12.0)),
        "zone_current_seen_5m": float(selected["current_seen"]),
        "zone_current_strength": float(selected["current_strength"]),
        "zone_recent_strength": float(selected["recent_strength"]),
        "zone_near_touch_count_1h": touch_count,
    }


def _append_trader_state_second_pass(frame: DataFrame) -> DataFrame:
    prepared = frame.copy()
    dates = pd.to_datetime(prepared["date"], utc=True, errors="coerce")
    segments = dates.diff().gt(pd.Timedelta(hours=2)).fillna(False).cumsum()
    chunks = []
    for _, chunk in prepared.groupby(segments, sort=False):
        chunks.append(_append_trader_state_second_pass_segment(chunk.copy()))
    if not chunks:
        return prepared
    return pd.concat(chunks).sort_index()


def _append_trader_state_second_pass_segment(frame: DataFrame) -> DataFrame:
    out = frame.copy()
    close = pd.to_numeric(out["obts_mid_close"], errors="coerce")
    high = pd.to_numeric(out["obts_mid_high"], errors="coerce")
    low = pd.to_numeric(out["obts_mid_low"], errors="coerce")
    pressure = pd.to_numeric(out["obts_pressure_25bps_mean"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    pressure_direction = Series(np.select([pressure >= 0.10, pressure <= -0.10], [1.0, -1.0], default=0.0), index=out.index)
    price_return = close.pct_change(1, fill_method=None).fillna(0.0)
    price_direction = Series(np.sign(price_return), index=out.index)

    out["obts_pressure_direction"] = pressure_direction
    out["obts_pressure_duration_hours"] = _consecutive_signed_duration(pressure_direction)
    out["obts_pressure_flip_strength"] = (pressure.diff().abs() * (pressure_direction != pressure_direction.shift(1)).astype(float)).clip(0.0, 2.0) / 2.0
    out["obts_pressure_acceleration"] = pressure.diff().clip(-1.0, 1.0).fillna(0.0)
    out["obts_pressure_price_agreement"] = ((pressure_direction != 0.0) & (price_direction != 0.0) & (pressure_direction == price_direction)).astype(float)
    out["obts_pressure_divergence"] = (pressure.abs() * ((pressure_direction != 0.0) & (price_direction != 0.0) & (pressure_direction != price_direction)).astype(float)).clip(0.0, 1.0)

    ask_depth = pd.to_numeric(out["obts_ask_depth_25bps_mean"], errors="coerce")
    bid_depth = pd.to_numeric(out["obts_bid_depth_25bps_mean"], errors="coerce")
    ask_median = ask_depth.rolling(24, min_periods=6).median()
    bid_median = bid_depth.rolling(24, min_periods=6).median()
    out["obts_upside_liquidity_vacuum_near"] = _clip01_series(1.0 - ask_depth / ask_median.replace(0.0, np.nan))
    out["obts_downside_liquidity_vacuum_near"] = _clip01_series(1.0 - bid_depth / bid_median.replace(0.0, np.nan))
    out["obts_upside_vacuum_after_resistance_removed"] = out["obts_upside_liquidity_vacuum_near"] * pd.to_numeric(out["obts_resistance_zone_removed_strength_1h"], errors="coerce").fillna(0.0)
    out["obts_downside_vacuum_after_support_removed"] = out["obts_downside_liquidity_vacuum_near"] * pd.to_numeric(out["obts_support_zone_removed_strength_1h"], errors="coerce").fillna(0.0)
    out["obts_vacuum_direction_agrees_with_price"] = (
        out["obts_upside_liquidity_vacuum_near"] * _clip01_series(price_return / 0.01)
        + out["obts_downside_liquidity_vacuum_near"] * _clip01_series(-price_return / 0.01)
    ).clip(0.0, 1.0)

    prior_res = pd.to_numeric(out["obts_resistance_zone_price"], errors="coerce").shift(1)
    prior_sup = pd.to_numeric(out["obts_support_zone_price"], errors="coerce").shift(1)
    res = pd.to_numeric(out["obts_resistance_zone_price"], errors="coerce")
    sup = pd.to_numeric(out["obts_support_zone_price"], errors="coerce")
    out["obts_price_approached_resistance_1h"] = ((res - high) / close * 10000.0 <= 25.0).astype(float).where(res.notna(), 0.0)
    out["obts_price_touched_resistance_1h"] = (high >= res * (1.0 - 5.0 / 10000.0)).astype(float).where(res.notna(), 0.0)
    out["obts_price_approached_support_1h"] = ((low - sup) / close * 10000.0 <= 25.0).astype(float).where(sup.notna(), 0.0)
    out["obts_price_touched_support_1h"] = (low <= sup * (1.0 + 5.0 / 10000.0)).astype(float).where(sup.notna(), 0.0)
    close_above_prior_res = (close > prior_res * (1.0 + 5.0 / 10000.0)).astype(float).where(prior_res.notna(), 0.0)
    close_below_prior_sup = (close < prior_sup * (1.0 - 5.0 / 10000.0)).astype(float).where(prior_sup.notna(), 0.0)
    recent_touch_res = out["obts_price_touched_resistance_1h"].rolling(3, min_periods=1).sum() > 0.0
    recent_touch_sup = out["obts_price_touched_support_1h"].rolling(3, min_periods=1).sum() > 0.0
    out["obts_price_accepted_above_resistance_3h"] = (close_above_prior_res.rolling(3, min_periods=1).sum() >= 2.0).astype(float)
    out["obts_price_accepted_below_support_3h"] = (close_below_prior_sup.rolling(3, min_periods=1).sum() >= 2.0).astype(float)
    out["obts_price_rejected_resistance_3h"] = (recent_touch_res & (close < prior_res * (1.0 - 5.0 / 10000.0))).astype(float).where(prior_res.notna(), 0.0)
    out["obts_price_bounced_support_3h"] = (recent_touch_sup & (close > prior_sup * (1.0 + 5.0 / 10000.0))).astype(float).where(prior_sup.notna(), 0.0)

    pressure_pos = pressure.clip(lower=0.0)
    pressure_neg = (-pressure).clip(lower=0.0)
    out["obts_ask_absorption_score"] = (
        out["obts_price_touched_resistance_1h"]
        * pressure_pos
        * pd.to_numeric(out["obts_resistance_zone_stability_score"], errors="coerce").fillna(0.0)
        * (1.0 - out["obts_price_accepted_above_resistance_3h"])
    )
    out["obts_bid_absorption_score"] = (
        out["obts_price_touched_support_1h"]
        * pressure_neg
        * pd.to_numeric(out["obts_support_zone_stability_score"], errors="coerce").fillna(0.0)
        * (1.0 - out["obts_price_accepted_below_support_3h"])
    )
    out["obts_ask_absorption_attempt_count_6h"] = (
        (out["obts_price_touched_resistance_1h"] * (pressure_pos > 0.10).astype(float)).rolling(6, min_periods=1).sum()
    )
    out["obts_bid_absorption_attempt_count_6h"] = (
        (out["obts_price_touched_support_1h"] * (pressure_neg > 0.10).astype(float)).rolling(6, min_periods=1).sum()
    )
    out["obts_absorption_exhaustion_score"] = (
        out[["obts_ask_absorption_score", "obts_bid_absorption_score", "obts_pressure_divergence"]].mean(axis=1).fillna(0.0)
    )
    out["obts_breakout_acceptance_score"] = (
        out["obts_price_accepted_above_resistance_3h"]
        * out[["obts_support_zone_added_strength_1h", "obts_pressure_price_agreement", "obts_upside_liquidity_vacuum_near"]].mean(axis=1).fillna(0.0)
    )
    out["obts_breakout_failure_score"] = (
        out["obts_price_rejected_resistance_3h"]
        * out[["obts_resistance_zone_stability_score", "obts_ask_absorption_score", "obts_pressure_divergence"]].mean(axis=1).fillna(0.0)
    )
    out["obts_breakdown_acceptance_score"] = (
        out["obts_price_accepted_below_support_3h"]
        * out[["obts_resistance_zone_added_strength_1h", "obts_pressure_price_agreement", "obts_downside_liquidity_vacuum_near"]].mean(axis=1).fillna(0.0)
    )
    out["obts_breakdown_failure_score"] = (
        out["obts_price_bounced_support_3h"]
        * out[["obts_support_zone_stability_score", "obts_bid_absorption_score", "obts_pressure_divergence"]].mean(axis=1).fillna(0.0)
    )
    out["obts_acceptance_hold_hours"] = pd.concat(
        [
            _consecutive_boolean_duration(close_above_prior_res > 0.0),
            _consecutive_boolean_duration(close_below_prior_sup > 0.0),
        ],
        axis=1,
    ).max(axis=1)

    resistance_distance = ((res - close) / close.replace(0.0, np.nan) * 10000.0).where(res.notna())
    support_distance = ((close - sup) / close.replace(0.0, np.nan) * 10000.0).where(sup.notna())
    resistance_proximity = _clip01_series(1.0 - resistance_distance / 50.0)
    support_proximity = _clip01_series(1.0 - support_distance / 50.0)
    resistance_stability = pd.to_numeric(out["obts_resistance_zone_stability_score"], errors="coerce").fillna(0.0)
    support_stability = pd.to_numeric(out["obts_support_zone_stability_score"], errors="coerce").fillna(0.0)
    resistance_removed = pd.to_numeric(out["obts_resistance_zone_removed_strength_1h"], errors="coerce").fillna(0.0)
    support_removed = pd.to_numeric(out["obts_support_zone_removed_strength_1h"], errors="coerce").fillna(0.0)
    support_added = pd.to_numeric(out["obts_support_zone_added_strength_1h"], errors="coerce").fillna(0.0)
    resistance_added = pd.to_numeric(out["obts_resistance_zone_added_strength_1h"], errors="coerce").fillna(0.0)
    accepted_above = pd.to_numeric(out["obts_price_accepted_above_resistance_3h"], errors="coerce").fillna(0.0)
    accepted_below = pd.to_numeric(out["obts_price_accepted_below_support_3h"], errors="coerce").fillna(0.0)
    rejected_resistance = pd.to_numeric(out["obts_price_rejected_resistance_3h"], errors="coerce").fillna(0.0)
    bounced_support = pd.to_numeric(out["obts_price_bounced_support_3h"], errors="coerce").fillna(0.0)
    price_progress_3h = (close / close.shift(3) - 1.0).abs()
    price_stall_score = _clip01_series(1.0 - price_progress_3h / 0.006)
    post_rejection_distance = _clip01_series((prior_res - close) / prior_res.replace(0.0, np.nan) * 10000.0 / 60.0)
    post_bounce_distance = _clip01_series((close - prior_sup) / prior_sup.replace(0.0, np.nan) * 10000.0 / 60.0)
    pre_failure_pressure = pd.concat(
        [pressure_pos, pd.to_numeric(out["obts_ask_absorption_score"], errors="coerce").fillna(0.0), out["obts_pressure_divergence"]],
        axis=1,
    ).max(axis=1)
    pre_breakdown_pressure = pd.concat(
        [pressure_neg, pd.to_numeric(out["obts_bid_absorption_score"], errors="coerce").fillna(0.0), out["obts_pressure_divergence"]],
        axis=1,
    ).max(axis=1)
    out["obts_pre_breakout_failure_risk_score"] = (
        resistance_proximity
        * resistance_stability
        * pre_failure_pressure
        * (1.0 - accepted_above)
        * (1.0 - rejected_resistance)
    ).clip(0.0, 1.0)
    out["obts_resistance_rejection_resolved_score"] = (
        rejected_resistance
        * resistance_stability
        * post_rejection_distance
        * (1.0 - accepted_above)
    ).clip(0.0, 1.0)
    out["obts_resistance_cleared_score"] = (
        accepted_above
        * pd.concat([support_added, resistance_removed, out["obts_pressure_price_agreement"], out["obts_upside_liquidity_vacuum_near"]], axis=1).mean(axis=1).fillna(0.0)
    ).clip(0.0, 1.0)
    out["obts_post_failure_chop_score"] = (
        out["obts_resistance_rejection_resolved_score"]
        * pd.to_numeric(out["obts_zone_compression_score"], errors="coerce").fillna(0.0)
        * price_stall_score
    ).clip(0.0, 1.0)
    out["obts_ask_absorption_before_reaction_score"] = (
        out["obts_price_touched_resistance_1h"]
        * pressure_pos
        * resistance_stability
        * price_stall_score
        * (1.0 - accepted_above)
        * (1.0 - rejected_resistance)
    ).clip(0.0, 1.0)
    out["obts_pre_breakdown_failure_risk_score"] = (
        support_proximity
        * support_stability
        * pre_breakdown_pressure
        * (1.0 - accepted_below)
        * (1.0 - bounced_support)
    ).clip(0.0, 1.0)
    out["obts_support_bounce_resolved_score"] = (
        bounced_support
        * support_stability
        * post_bounce_distance
        * (1.0 - accepted_below)
    ).clip(0.0, 1.0)
    out["obts_support_cleared_score"] = (
        accepted_below
        * pd.concat([resistance_added, support_removed, out["obts_pressure_price_agreement"], out["obts_downside_liquidity_vacuum_near"]], axis=1).mean(axis=1).fillna(0.0)
    ).clip(0.0, 1.0)
    out["obts_post_breakdown_chop_score"] = (
        out["obts_support_bounce_resolved_score"]
        * pd.to_numeric(out["obts_zone_compression_score"], errors="coerce").fillna(0.0)
        * price_stall_score
    ).clip(0.0, 1.0)
    out["obts_bid_absorption_before_reaction_score"] = (
        out["obts_price_touched_support_1h"]
        * pressure_neg
        * support_stability
        * price_stall_score
        * (1.0 - accepted_below)
        * (1.0 - bounced_support)
    ).clip(0.0, 1.0)

    for window in (24, 72):
        prior_high = high.shift(1).rolling(window, min_periods=max(3, window // 2)).max()
        prior_low = low.shift(1).rolling(window, min_periods=max(3, window // 2)).min()
        prior_range = (prior_high - prior_low).replace(0.0, np.nan)
        out[f"obts_range_position_{window}h"] = ((close - prior_low) / prior_range).clip(0.0, 1.0)
        trend_return = close / close.shift(window) - 1.0
        out[f"obts_price_trend_state_{window}h"] = np.select(
            [trend_return >= 0.01, trend_return <= -0.01],
            [1.0, -1.0],
            default=0.0,
        )
    higher_high = (high > high.shift(1)).astype(float)
    higher_low = (low > low.shift(1)).astype(float)
    lower_high = (high < high.shift(1)).astype(float)
    lower_low = (low < low.shift(1)).astype(float)
    out["obts_higher_high_sequence"] = (higher_high + higher_low).rolling(6, min_periods=3).sum()
    out["obts_lower_low_sequence"] = (lower_high + lower_low).rolling(6, min_periods=3).sum()
    short_vol = ((high - low) / close.replace(0.0, np.nan)).rolling(6, min_periods=3).mean()
    medium_vol = ((high - low) / close.replace(0.0, np.nan)).rolling(24, min_periods=12).mean()
    out["obts_volatility_expansion_state"] = (short_vol / medium_vol.replace(0.0, np.nan) - 1.0).clip(-3.0, 3.0)
    return out


def _json_records(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    try:
        loaded = json.loads(str(value))
    except Exception:
        return []
    if not isinstance(loaded, list):
        return []
    return [item for item in loaded if isinstance(item, dict)]


def _nan_if_missing(value: Any) -> float:
    result = _float(value)
    return result


def _float(value: Any) -> float:
    try:
        result = float(value)
    except Exception:
        return math.nan
    return result if math.isfinite(result) else math.nan


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator and math.isfinite(denominator) else math.nan


def _safe_div(value: Any, denominator: float) -> float:
    number = _float(value)
    if math.isnan(number) or denominator == 0.0:
        return math.nan
    return number / denominator


def _clip01(value: float) -> float:
    if value is None or not math.isfinite(float(value)):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _clip01_series(series: Series) -> Series:
    return pd.to_numeric(series, errors="coerce").clip(lower=0.0, upper=1.0).fillna(0.0)


def _row_score(*values: float) -> float:
    valid = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(valid)) if valid else 0.0


def _mean(series: Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else math.nan


def _min(series: Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.min()) if not numeric.empty else math.nan


def _max(series: Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.max()) if not numeric.empty else math.nan


def _last(series: Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.iloc[-1]) if not numeric.empty else math.nan


def _quantile(series: Series, quantile: float) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.quantile(quantile)) if not numeric.empty else math.nan


def _slope(series: Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 2:
        return math.nan
    x = np.linspace(0.0, 1.0, len(numeric))
    try:
        return float(np.polyfit(x, numeric.to_numpy(dtype=float), 1)[0])
    except Exception:
        return math.nan


def _sign_flip_count(values: np.ndarray) -> int:
    signs = [int(np.sign(value)) for value in values if math.isfinite(float(value)) and np.sign(value) != 0]
    if len(signs) < 2:
        return 0
    return sum(1 for left, right in zip(signs, signs[1:]) if left != right)


def _consecutive_signed_duration(sign: Series) -> Series:
    output = Series(0.0, index=sign.index, dtype="float64")
    count = 0.0
    previous = 0.0
    for index, value in pd.to_numeric(sign, errors="coerce").fillna(0.0).items():
        if value != 0.0 and value == previous:
            count += 1.0
        elif value != 0.0:
            count = 1.0
        else:
            count = 0.0
        previous = float(value)
        output.loc[index] = count
    return output


def _consecutive_boolean_duration(mask: Series) -> Series:
    output = Series(0.0, index=mask.index, dtype="float64")
    count = 0.0
    for index, value in mask.fillna(False).astype(bool).items():
        count = count + 1.0 if bool(value) else 0.0
        output.loc[index] = count
    return output


if __name__ == "__main__":
    raise SystemExit(main())

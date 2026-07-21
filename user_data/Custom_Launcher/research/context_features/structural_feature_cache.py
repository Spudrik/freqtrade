from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Indicators.complex_trendline_projection_v2 import (
    add_trendline_projection_v2,
    build_trendline_projection_v2_state,
)
from user_data.Indicators.complex_volume_profile import add_volume_profile
from user_data.Indicators.pattern_bos_choch import add_bos_choch
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = USER_DATA_DIR / "data" / "binance"
DEFAULT_OUTPUT_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "structural_cache"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "btc_structural_features_1h.parquet"
DEFAULT_LATEST = DEFAULT_OUTPUT_DIR / "btc_structural_features_1h_latest.parquet"

TIMEFRAME_HOURS = {"1h": 1, "4h": 4, "8h": 8, "1d": 24, "3d": 72}

VP_COLUMNS = (
    "prior_poc",
    "prior_vah",
    "prior_val",
    "poc",
    "vah",
    "val",
    "value_area_width_pct",
    "value_area_position",
    "distance_to_poc_pct",
    "distance_to_vah_pct",
    "distance_to_val_pct",
    "close_bin_volume_share",
    "close_bin_volume_percentile",
    "delta_ratio",
    "poc_delta_ratio",
    "entropy",
    "concentration",
    "hvn_above_distance_pct",
    "hvn_below_distance_pct",
    "lvn_above_distance_pct",
    "lvn_below_distance_pct",
    "hvn_above_strength",
    "hvn_below_strength",
    "lvn_above_thinness",
    "lvn_below_thinness",
    "in_value_area",
    "above_value_area",
    "below_value_area",
    "upper_rejection",
    "lower_rejection",
    "vah_breakout",
    "val_breakdown",
    "bull_pressure",
    "bear_pressure",
    "hvn_below_reclaim",
    "hvn_above_reject",
    "vah_breakout_with_pressure",
    "val_breakdown_with_pressure",
    "lower_rejection_with_pressure",
    "upper_rejection_with_pressure",
    "entry_trigger_long",
    "entry_trigger_short",
    "node_entry_long",
    "node_entry_short",
    "node_hold_long",
    "node_hold_short",
    "node_exit_long",
    "node_exit_short",
    "market_context",
    "score_long",
    "score_short",
    "score_abs",
    "state",
)

TLV2_COLUMNS = (
    "resistance_line_rank0",
    "resistance_score_rank0",
    "resistance_distance_atr_rank0",
    "resistance_slope_atr_per_bar_rank0",
    "support_line_rank0",
    "support_score_rank0",
    "support_distance_atr_rank0",
    "support_slope_atr_per_bar_rank0",
)

MS_COLUMNS = (
    "bos_to_bull",
    "bos_to_bear",
    "choch_to_bull",
    "choch_to_bear",
    "structure_event",
    "state",
    "higher_high",
    "lower_high",
    "higher_low",
    "lower_low",
    "last_swing_high",
    "last_swing_low",
    "break_level",
    "invalidation_level",
    "bullish_break_level",
    "bearish_break_level",
)

GEOMETRY_COLUMNS = (
    "triangle_pattern_present",
    "triangle_indicator_score",
    "triangle_direction",
    "triangle_width_atr",
    "triangle_squeeze_active",
    "triangle_upper",
    "triangle_lower",
    "wedge_pattern_present",
    "wedge_indicator_score",
    "wedge_direction",
    "wedge_width_atr",
    "wedge_squeeze_active",
    "wedge_upper",
    "wedge_lower",
    "compression_pattern_present",
    "compression_indicator_score",
    "compression_direction",
    "compression_width_atr",
    "compression_squeeze_active",
    "compression_upper",
    "compression_lower",
    "rectangle_pattern_present",
    "rectangle_indicator_score",
    "rectangle_direction",
    "rectangle_width_atr",
    "rectangle_squeeze_active",
    "rectangle_upper",
    "rectangle_lower",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build cached custom-structural features for FreqAI/context research.")
    parser.add_argument("--pair-stem", default="BTC_USDT")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--latest", type=Path, default=DEFAULT_LATEST)
    parser.add_argument("--timeframes", default="1h,4h,1d")
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--warmup-days", type=int, default=240)
    parser.add_argument("--include-geometry", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    timeframes = [item.strip() for item in args.timeframes.split(",") if item.strip()]
    config = {
        "pair_stem": args.pair_stem,
        "timeframes": timeframes,
        "include_geometry": bool(args.include_geometry),
        "start": args.start,
        "end": args.end,
        "warmup_days": int(args.warmup_days),
        "vp": {"window": 96, "bins": 48, "value_area_pct": 0.70, "price_source": "hlc3"},
        "tlv2": {"raw_line_output_count": 1, "min_output_line_score": 0.50},
        "ms": {"include_sequence": True, "include_diagnostics": True},
    }
    plan = {
        "mode": "setup-only unless --execute is supplied",
        "data_dir": str(args.data_dir),
        "output": str(args.output),
        "latest": str(args.latest),
        "config": config,
        "timestamp_rule": "features are stamped at candle close/availability time; informative timeframes merge backward onto the 1h close timeline",
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0

    frame, summary = build_structural_cache(
        args.pair_stem,
        args.data_dir,
        timeframes,
        bool(args.include_geometry),
        config,
        parse_bound(args.start),
        parse_bound(args.end),
        int(args.warmup_days),
    )
    frame, dropped_all_null = drop_all_null_numeric_columns(frame)
    summary["dropped_all_null_numeric_columns"] = dropped_all_null
    summary["all_null_numeric_columns"] = []
    summary["numeric_columns"] = int(len(frame.select_dtypes(include=["number", "bool"]).columns))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    frame.to_parquet(args.latest, index=False)
    meta_path = args.output.with_suffix(".meta.json")
    meta = {**plan, **summary, "rows": int(len(frame)), "columns": int(len(frame.columns))}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "latest": str(args.latest), "meta": str(meta_path), **summary}, indent=2))
    return 0


def build_structural_cache(
    pair_stem: str,
    data_dir: Path,
    timeframes: list[str],
    include_geometry: bool,
    config: dict[str, Any],
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    warmup_days: int = 240,
) -> tuple[DataFrame, dict[str, Any]]:
    warmup_start = start - pd.Timedelta(days=int(warmup_days)) if start is not None else None
    base = load_ohlcv(data_dir / f"{pair_stem}-1h.feather", "1h", warmup_start, end)
    output = base[["date", "open", "high", "low", "close", "volume"]].copy()
    output["canonical_pair"] = pair_stem.replace("_", "/")
    stats: dict[str, Any] = {"timeframe_rows": {}, "config_hash": config_hash(config)}
    for timeframe in timeframes:
        print(f"building {timeframe} structural indicators on {pair_stem}...", flush=True)
        source = load_ohlcv(data_dir / f"{pair_stem}-{timeframe}.feather", timeframe, warmup_start, end)
        featured = add_indicator_family(source, timeframe, include_geometry)
        compact = select_structural_columns(featured, timeframe, include_geometry)
        stats["timeframe_rows"][timeframe] = int(len(compact))
        if timeframe == "1h":
            output = output.merge(compact, on="date", how="left", sort=False)
        else:
            output = pd.merge_asof(
                output.sort_values("date"),
                compact.sort_values("date"),
                on="date",
                direction="backward",
                tolerance=pd.Timedelta(hours=TIMEFRAME_HOURS[timeframe] * 2),
            ).sort_values("date")
    if start is not None:
        output = output[output["date"] >= start].copy()
    if end is not None:
        output = output[output["date"] <= end].copy()
    output["feature_generated_at"] = pd.Timestamp.utcnow()
    output["schema_version"] = 1
    output["feature_config_hash"] = stats["config_hash"]
    output = append_trader_setup_features(output)
    numeric = output.select_dtypes(include=["number", "bool"]).columns
    output[numeric] = output[numeric].replace([np.inf, -np.inf], np.nan)
    stats["first_date"] = str(output["date"].min())
    stats["last_date"] = str(output["date"].max())
    stats["numeric_columns"] = int(len(numeric))
    stats["all_null_numeric_columns"] = [column for column in numeric if output[column].isna().all()]
    return output, stats


def drop_all_null_numeric_columns(frame: DataFrame) -> tuple[DataFrame, list[str]]:
    numeric = frame.select_dtypes(include=["number", "bool"]).columns
    dropped = [column for column in numeric if frame[column].isna().all()]
    if not dropped:
        return frame, []
    return frame.drop(columns=dropped), dropped


def load_ohlcv(path: Path, timeframe: str, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> DataFrame:
    if timeframe not in TIMEFRAME_HOURS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    frame = pd.read_feather(path)
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame = frame[["date", "open", "high", "low", "close", "volume"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce") + pd.Timedelta(hours=TIMEFRAME_HOURS[timeframe])
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if start is not None:
        frame = frame[frame["date"] >= start]
    if end is not None:
        frame = frame[frame["date"] <= end]
    return frame.reset_index(drop=True)


def add_indicator_family(frame: DataFrame, timeframe: str, include_geometry: bool) -> DataFrame:
    out = add_volume_profile(frame, prefix="vp", window=96, bins=48, value_area_pct=0.70, price_source="hlc3")
    trendline_state = build_trendline_projection_v2_state(
        out,
        timeframe=timeframe,
        output_prefix="tlv2",
        raw_line_output_count=1,
        min_output_line_score=0.50,
        include_diagnostics=False,
    )
    out = add_trendline_projection_v2(
        out,
        timeframe=timeframe,
        output_prefix="tlv2",
        raw_line_output_count=1,
        min_output_line_score=0.50,
        include_diagnostics=False,
        state=trendline_state,
    )
    out = add_bos_choch(out, prefix="ms", include_sequence=True, include_diagnostics=True)
    if include_geometry:
        out = add_pattern_geometry_v2(
            out,
            timeframe=timeframe,
            output_prefix="pg2",
            output_slots=2,
            trendline_state=trendline_state,
        )
    return out


def select_structural_columns(frame: DataFrame, timeframe: str, include_geometry: bool) -> DataFrame:
    selected: dict[str, Any] = {"date": frame["date"]}
    for source_prefix, output_prefix, columns in (
        ("vp", "vp", VP_COLUMNS),
        ("tlv2", "tlv2", TLV2_COLUMNS),
        ("ms", "ms", MS_COLUMNS),
    ):
        for suffix in columns:
            source = f"{source_prefix}_{suffix}"
            if source in frame:
                selected[f"st_{timeframe}_{output_prefix}_{suffix}"] = numeric_feature(frame[source])
    if include_geometry:
        for suffix in GEOMETRY_COLUMNS:
            source = f"pg2_{suffix}"
            if source in frame:
                selected[f"st_{timeframe}_pg2_{suffix}"] = numeric_feature(frame[source])
    compact = pd.DataFrame(selected)
    return compact.sort_values("date").reset_index(drop=True)


def append_trader_setup_features(frame: DataFrame) -> DataFrame:
    out = frame.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    volume = pd.to_numeric(out["volume"], errors="coerce").clip(lower=0.0)
    candle_range = (high - low).replace(0.0, np.nan)
    pressure = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0).fillna(0.0)
    signed_volume = pressure * volume.fillna(0.0)
    out["st_price_return_1h"] = close.pct_change(1)
    out["st_price_return_3h"] = close.pct_change(3)
    out["st_price_return_6h"] = close.pct_change(6)
    out["st_volume_pressure_6h"] = signed_volume.rolling(6, min_periods=3).sum() / volume.rolling(6, min_periods=3).sum().replace(0.0, np.nan)
    out["st_volume_pressure_24h"] = signed_volume.rolling(24, min_periods=12).sum() / volume.rolling(24, min_periods=12).sum().replace(0.0, np.nan)
    out["st_volume_z_24h"] = (volume - volume.rolling(24, min_periods=12).mean()) / volume.rolling(24, min_periods=12).std(ddof=0).replace(0.0, np.nan)
    out["st_range_position_24h"] = (close - low.shift(1).rolling(24, min_periods=12).min()) / (
        high.shift(1).rolling(24, min_periods=12).max() - low.shift(1).rolling(24, min_periods=12).min()
    ).replace(0.0, np.nan)
    out["st_range_position_72h"] = (close - low.shift(1).rolling(72, min_periods=36).min()) / (
        high.shift(1).rolling(72, min_periods=36).max() - low.shift(1).rolling(72, min_periods=36).min()
    ).replace(0.0, np.nan)
    out["st_near_tlv2_resistance_1h"] = (out.get("st_1h_tlv2_resistance_distance_atr_rank0", pd.Series(np.nan, index=out.index)).abs() <= 1.0).astype(float)
    out["st_near_tlv2_support_1h"] = (out.get("st_1h_tlv2_support_distance_atr_rank0", pd.Series(np.nan, index=out.index)).abs() <= 1.0).astype(float)
    out["st_breakout_structure_setup"] = (
        boolish(out, "st_1h_ms_bos_to_bull")
        + boolish(out, "st_1h_vp_vah_breakout_with_pressure")
        + boolish(out, "st_near_tlv2_resistance_1h")
        + positive(out, "st_1h_vp_score_long", 0.25)
    )
    out["st_breakdown_structure_setup"] = (
        boolish(out, "st_1h_ms_bos_to_bear")
        + boolish(out, "st_1h_vp_val_breakdown_with_pressure")
        + boolish(out, "st_near_tlv2_support_1h")
        + positive(out, "st_1h_vp_score_short", 0.25)
    )
    out["st_failed_breakout_structure_risk"] = (
        boolish(out, "st_1h_vp_upper_rejection_with_pressure")
        + boolish(out, "st_1h_vp_hvn_above_reject")
        + positive(out, "st_1h_tlv2_resistance_score_rank0", 0.50)
        + (1.0 - positive(out, "st_volume_pressure_6h", 0.0))
    )
    out["st_failed_breakdown_structure_risk"] = (
        boolish(out, "st_1h_vp_lower_rejection_with_pressure")
        + boolish(out, "st_1h_vp_hvn_below_reclaim")
        + positive(out, "st_1h_tlv2_support_score_rank0", 0.50)
        + positive(out, "st_volume_pressure_6h", 0.0)
    )
    return out


def numeric_feature(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    return pd.to_numeric(series, errors="coerce")


def boolish(frame: DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0).gt(0.0).astype(float)


def positive(frame: DataFrame, column: str, threshold: float) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0).ge(float(threshold)).astype(float)


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def parse_bound(value: str) -> pd.Timestamp | None:
    clean = str(value or "").strip()
    if not clean:
        return None
    return pd.Timestamp(clean, tz="UTC")


if __name__ == "__main__":
    raise SystemExit(main())

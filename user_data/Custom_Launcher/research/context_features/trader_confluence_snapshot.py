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

from user_data.Custom_Launcher.research.context_features.trader_confluence_features import (  # noqa: E402
    append_confluence_features,
    append_future_labels,
    append_price_features,
)


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OHLCV = USER_DATA_DIR / "data" / "binance" / "BTC_USDT-1h.feather"
DEFAULT_STRUCTURAL = USER_DATA_DIR / "research_news_data" / "context_features" / "structural_cache" / "btc_structural_features_1h_latest.parquet"
DEFAULT_CONTEXT = USER_DATA_DIR / "research_news_data" / "context_features" / "exports" / "context_features_1h_latest.parquet"
DEFAULT_OB_SPOT = USER_DATA_DIR / "orderbook_data" / "historical_bybit" / "features" / "orderbook_trader_state_1h_latest.parquet"
DEFAULT_OB_LINEAR = USER_DATA_DIR / "orderbook_data" / "historical_bybit" / "features" / "orderbook_trader_state_1h_bybit_linear_latest.parquet"
DEFAULT_OB_INVERSE = USER_DATA_DIR / "orderbook_data" / "historical_bybit" / "features" / "orderbook_trader_state_1h_bybit_inverse_latest.parquet"
DEFAULT_OUTPUT_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "confluence_cache"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "trader_confluence_1h.parquet"
DEFAULT_LATEST = DEFAULT_OUTPUT_DIR / "trader_confluence_1h_latest.parquet"
SCHEMA_VERSION = "trader_confluence_snapshot_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a frozen 1h trader-confluence research snapshot from validated parquet sources.")
    parser.add_argument("--ohlcv", type=Path, default=DEFAULT_OHLCV)
    parser.add_argument("--structural", type=Path, default=DEFAULT_STRUCTURAL)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--orderbook-spot", type=Path, default=DEFAULT_OB_SPOT)
    parser.add_argument("--orderbook-linear", type=Path, default=DEFAULT_OB_LINEAR)
    parser.add_argument("--orderbook-inverse", type=Path, default=DEFAULT_OB_INVERSE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--latest", type=Path, default=DEFAULT_LATEST)
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--orderbook-min-coverage", type=float, default=0.80)
    parser.add_argument("--context-max-age-hours", type=float, default=48.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    plan = {
        "mode": "setup-only unless --execute is supplied",
        "schema_version": SCHEMA_VERSION,
        "output": str(args.output),
        "latest": str(args.latest),
        "timestamp_rule": "all feature rows are joined on the 1h candle close timestamp; missing sources stay missing",
        "lookahead_rule": "orderbook source_max_ts and context max_source_available_at must be <= the feature date",
        "inputs": input_manifest(args),
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0

    frame, validation = build_snapshot(
        args.ohlcv,
        args.structural,
        args.context,
        args.orderbook_spot,
        args.orderbook_linear,
        args.orderbook_inverse,
        parse_bound(args.start),
        parse_bound(args.end),
        float(args.orderbook_min_coverage),
        float(args.context_max_age_hours),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    if args.latest:
        args.latest.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(args.latest, index=False)
    meta_path = args.output.with_suffix(".meta.json")
    validation_path = args.output.with_suffix(".validation.json")
    meta = {**plan, "output_rows": int(len(frame)), "output_columns": int(len(frame.columns)), "validation": validation}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    validation_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "latest": str(args.latest), "meta": str(meta_path), "validation": str(validation_path), "rows": int(len(frame)), "columns": int(len(frame.columns))}, indent=2))
    return 0


def build_snapshot(
    ohlcv_path: Path,
    structural_path: Path,
    context_path: Path,
    orderbook_spot_path: Path,
    orderbook_linear_path: Path,
    orderbook_inverse_path: Path,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    orderbook_min_coverage: float,
    context_max_age_hours: float,
) -> tuple[DataFrame, dict[str, Any]]:
    base = load_ohlcv(ohlcv_path, start, end)
    frame = append_price_features(base)
    validation: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "base": frame_summary("ohlcv", frame),
        "inputs": {},
        "warnings": [],
    }

    structure = load_parquet_source(structural_path, "structure")
    frame = merge_structural(frame, structure)
    validation["inputs"]["structure"] = frame_summary("structure", structure)

    for name, path, prefix in (
        ("orderbook_spot", orderbook_spot_path, "ob_spot_"),
        ("orderbook_linear", orderbook_linear_path, "ob_linear_"),
        ("orderbook_inverse", orderbook_inverse_path, "ob_inverse_"),
    ):
        source = load_parquet_source(path, name)
        frame = merge_orderbook(frame, source, prefix, orderbook_min_coverage)
        validation["inputs"][name] = frame_summary(name, source)

    context = load_parquet_source(context_path, "context")
    frame = merge_context(frame, context, context_max_age_hours)
    validation["inputs"]["context"] = frame_summary("context", context)

    frame = append_confluence_features(frame)
    frame = append_future_labels(frame)
    frame["confluence_schema_version"] = SCHEMA_VERSION
    frame["confluence_generated_at"] = pd.Timestamp.utcnow()
    validation.update(validate_snapshot(frame, orderbook_min_coverage, context_max_age_hours))
    validation["warnings"].extend(validation_warnings(validation))
    return frame, validation


def load_ohlcv(path: Path, start: pd.Timestamp | None, end: pd.Timestamp | None) -> DataFrame:
    frame = pd.read_feather(path, columns=["date", "open", "high", "low", "close", "volume"])
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce") + pd.Timedelta(hours=1)
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if start is not None:
        frame = frame[frame["date"] >= start]
    if end is not None:
        frame = frame[frame["date"] <= end]
    return frame.reset_index(drop=True)


def load_parquet_source(path: Path, source_name: str) -> DataFrame:
    if not path.exists():
        return DataFrame({"date": pd.Series(dtype="datetime64[ns, UTC]"), f"_{source_name}_missing_file": pd.Series(dtype=float)})
    frame = pd.read_parquet(path)
    if "date" not in frame:
        raise ValueError(f"{path} has no date column")
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    return frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def merge_structural(frame: DataFrame, structure: DataFrame) -> DataFrame:
    if structure.empty:
        frame["structure_present"] = 0.0
        return frame
    keep = ["date", *[column for column in structure.columns if column.startswith("st_")]]
    compact = structure[keep].copy()
    compact["_structure_row_present"] = 1.0
    out = frame.merge(compact, on="date", how="left", sort=False)
    out["structure_present"] = pd.to_numeric(out.pop("_structure_row_present"), errors="coerce").fillna(0.0)
    return out


def merge_orderbook(frame: DataFrame, source: DataFrame, prefix: str, min_coverage: float) -> DataFrame:
    present_column = f"{prefix}present"
    if source.empty:
        frame[present_column] = 0.0
        return frame
    renamed = {"date": "date"}
    for column in source.columns:
        if column == "date":
            continue
        renamed[column] = f"{prefix}{column}"
    compact = source.rename(columns=renamed)
    compact[f"{prefix}row_present"] = 1.0
    out = frame.merge(compact, on="date", how="left", sort=False)
    feature_present = pd.to_numeric(out.get(f"{prefix}obts_feature_present"), errors="coerce").fillna(0.0).gt(0.0)
    row_present = pd.to_numeric(out.get(f"{prefix}row_present"), errors="coerce").fillna(0.0).gt(0.0)
    coverage = pd.to_numeric(out.get(f"{prefix}obts_coverage_ratio"), errors="coerce")
    coverage_ok = coverage.fillna(0.0).ge(float(min_coverage))
    source_max = pd.to_datetime(out.get(f"{prefix}source_max_ts"), utc=True, errors="coerce")
    date = pd.to_datetime(out["date"], utc=True, errors="coerce")
    source_not_future = source_max.notna() & source_max.le(date)
    out[present_column] = (row_present & feature_present & coverage_ok & source_not_future).astype(float)
    out[f"{prefix}low_coverage"] = (row_present & ~coverage_ok).astype(float)
    out[f"{prefix}source_future_violation"] = (source_max.notna() & source_max.gt(date)).astype(float)
    out[f"{prefix}source_age_hours"] = ((date - source_max).dt.total_seconds() / 3600.0).where(source_max.notna())
    out = mask_orderbook_source_columns(out, prefix, out[present_column].gt(0.0))
    return out


def merge_context(frame: DataFrame, context: DataFrame, max_age_hours: float) -> DataFrame:
    if context.empty:
        frame["context_present"] = 0.0
        frame["context_row_present"] = 0.0
        return frame
    renamed = {"date": "date"}
    for column in context.columns:
        if column == "date":
            continue
        renamed[column] = f"ctx_{column}"
    compact = context.rename(columns=renamed)
    compact["context_row_present"] = 1.0
    out = frame.merge(compact, on="date", how="left", sort=False)
    date = pd.to_datetime(out["date"], utc=True, errors="coerce")
    max_available = pd.to_datetime(out.get("ctx_max_source_available_at"), utc=True, errors="coerce")
    age_hours = (date - max_available).dt.total_seconds() / 3600.0
    row_present = pd.to_numeric(out.get("context_row_present"), errors="coerce").fillna(0.0).gt(0.0)
    source_ok = max_available.notna() & max_available.le(date) & age_hours.ge(0.0) & age_hours.le(float(max_age_hours))
    out["context_source_age_hours"] = age_hours.where(max_available.notna())
    out["context_source_future_violation"] = (max_available.notna() & max_available.gt(date)).astype(float)
    out["context_present"] = (row_present & source_ok).astype(float)
    out = mask_context_source_columns(out, out["context_present"].gt(0.0))
    return out


def mask_orderbook_source_columns(frame: DataFrame, prefix: str, present: pd.Series) -> DataFrame:
    keep = {
        f"{prefix}present",
        f"{prefix}row_present",
        f"{prefix}low_coverage",
        f"{prefix}source_future_violation",
        f"{prefix}source_age_hours",
        f"{prefix}source_min_ts",
        f"{prefix}source_max_ts",
    }
    mask = present.reindex(frame.index).fillna(False)
    for column in [column for column in frame.columns if column.startswith(prefix) and column not in keep]:
        if pd.api.types.is_numeric_dtype(frame[column]):
            frame.loc[~mask, column] = np.nan
    return frame


def mask_context_source_columns(frame: DataFrame, present: pd.Series) -> DataFrame:
    keep = {
        "context_present",
        "context_row_present",
        "context_source_age_hours",
        "context_source_future_violation",
        "ctx_min_source_available_at",
        "ctx_max_source_available_at",
    }
    mask = present.reindex(frame.index).fillna(False)
    for column in [column for column in frame.columns if column.startswith("ctx_") and column not in keep]:
        if pd.api.types.is_numeric_dtype(frame[column]):
            frame.loc[~mask, column] = np.nan
    return frame


def validate_snapshot(frame: DataFrame, min_coverage: float, max_context_age_hours: float) -> dict[str, Any]:
    date = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    gaps = date.sort_values().diff().dropna()
    numeric = frame.select_dtypes(include=["number", "bool"]).columns
    validation: dict[str, Any] = {
        "output": frame_summary("trader_confluence_snapshot", frame),
        "duplicate_dates": int(date.duplicated().sum()),
        "hourly_gap_count": int(gaps.gt(pd.Timedelta(hours=1)).sum()),
        "numeric_columns": int(len(numeric)),
        "non_numeric_columns": int(len(frame.columns) - len(numeric)),
        "infinite_numeric_cells": int(np.isinf(frame[numeric].to_numpy(dtype=float, na_value=np.nan)).sum()) if len(numeric) else 0,
        "source_presence": {},
        "thresholds": {
            "orderbook_min_coverage": float(min_coverage),
            "context_max_age_hours": float(max_context_age_hours),
        },
    }
    for source in ("structure", "context", "ob_spot", "ob_linear", "ob_inverse"):
        column = f"{source}_present" if source.startswith("ob_") else f"{source}_present"
        if column in frame:
            validation["source_presence"][source] = {
                "present_rows": int(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).gt(0.0).sum()),
                "present_ratio": float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).gt(0.0).mean()),
            }
    for source in ("ob_spot", "ob_linear", "ob_inverse"):
        low_col = f"{source}_low_coverage"
        future_col = f"{source}_source_future_violation"
        if low_col in frame:
            validation["source_presence"].setdefault(source, {})["low_coverage_rows"] = int(pd.to_numeric(frame[low_col], errors="coerce").fillna(0.0).gt(0.0).sum())
        if future_col in frame:
            validation["source_presence"].setdefault(source, {})["source_future_violations"] = int(pd.to_numeric(frame[future_col], errors="coerce").fillna(0.0).gt(0.0).sum())
    if "context_source_future_violation" in frame:
        validation["source_presence"].setdefault("context", {})["source_future_violations"] = int(pd.to_numeric(frame["context_source_future_violation"], errors="coerce").fillna(0.0).gt(0.0).sum())
    validation["inactive_source_signal_rows"] = inactive_source_signal_rows(frame)
    return validation


def inactive_source_signal_rows(frame: DataFrame) -> dict[str, int]:
    context_signals = [
        column
        for column in (
            "conf_context_activity_24h",
            "conf_context_activity_acceleration",
            "conf_context_topic_confluence_24h",
            "conf_context_topic_persistence_24h",
            "conf_context_topic_severity_24h",
            "conf_context_risk_event_pressure",
            "conf_context_event_regime_score",
            "conf_context_quiet_regime",
        )
        if column in frame
    ]
    checks: dict[str, tuple[str, list[str]]] = {
        "context": ("context_present", context_signals),
        "orderbook": ("conf_ob_venue_count_present", [column for column in frame.columns if column.startswith("conf_ob_")]),
    }
    result: dict[str, int] = {}
    for source, (present_column, signal_columns) in checks.items():
        if present_column not in frame or not signal_columns:
            result[source] = 0
            continue
        inactive = pd.to_numeric(frame[present_column], errors="coerce").fillna(0.0).le(0.0)
        signals = frame[signal_columns].select_dtypes(include=["number", "bool"])
        if signals.empty:
            result[source] = 0
            continue
        active_signal = signals.fillna(0.0).abs().gt(1e-12).any(axis=1)
        result[source] = int((inactive & active_signal).sum())
    return result


def validation_warnings(validation: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if validation.get("duplicate_dates", 0):
        warnings.append("Snapshot has duplicate dates.")
    if validation.get("hourly_gap_count", 0):
        warnings.append("Snapshot base timeline has hourly gaps; tests must not assume continuous candles through those gaps.")
    if validation.get("infinite_numeric_cells", 0):
        warnings.append("Snapshot has infinite numeric cells.")
    for source, item in validation.get("source_presence", {}).items():
        if item.get("source_future_violations", 0):
            warnings.append(f"{source} has source timestamp lookahead violations.")
        if item.get("low_coverage_rows", 0):
            warnings.append(f"{source} has low-coverage rows excluded from present flags.")
        if item.get("present_rows", 0) == 0:
            warnings.append(f"{source} has no present rows in the selected window.")
    for source, rows in validation.get("inactive_source_signal_rows", {}).items():
        if rows:
            warnings.append(f"{source} has {rows} inactive rows with nonzero confluence source signals.")
    return warnings


def frame_summary(name: str, frame: DataFrame) -> dict[str, Any]:
    if frame.empty or "date" not in frame:
        return {"name": name, "rows": int(len(frame)), "columns": int(len(frame.columns)), "first_date": None, "last_date": None}
    date = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    gaps = date.sort_values().diff().dropna()
    return {
        "name": name,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "first_date": str(date.min()),
        "last_date": str(date.max()),
        "duplicate_dates": int(date.duplicated().sum()),
        "hourly_gap_count": int(gaps.gt(pd.Timedelta(hours=1)).sum()),
    }


def input_manifest(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "ohlcv": args.ohlcv,
        "structural": args.structural,
        "context": args.context,
        "orderbook_spot": args.orderbook_spot,
        "orderbook_linear": args.orderbook_linear,
        "orderbook_inverse": args.orderbook_inverse,
    }
    return {name: file_manifest(path) for name, path in paths.items()}


def file_manifest(path: Path) -> dict[str, Any]:
    item = {"path": str(path), "exists": bool(path.exists())}
    if path.exists():
        stat = path.stat()
        item["size_bytes"] = int(stat.st_size)
        item["sha256_16"] = sha256_prefix(path)
    return item


def sha256_prefix(path: Path, max_bytes: int = 16 * 1024 * 1024) -> str:
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


def parse_bound(value: str) -> pd.Timestamp | None:
    if not value:
        return None
    return pd.Timestamp(value, tz="UTC") if pd.Timestamp(value).tzinfo is None else pd.Timestamp(value).tz_convert("UTC")


if __name__ == "__main__":
    raise SystemExit(main())

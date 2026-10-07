from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


APP_DIR = Path(__file__).resolve().parents[2]
USER_DATA_DIR = APP_DIR.parent
DEFAULT_RAW_DIR = Path("D:/FreqTradeStuffLargeData/orderbook_data/historical_bybit/spot_raw")
DEFAULT_FEATURE_PATH = USER_DATA_DIR / "orderbook_data" / "historical_bybit" / "features" / "orderbook_trader_state_1h_latest.parquet"


@dataclass(frozen=True)
class RawArchive:
    path: Path
    day: date


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate derived 1h Bybit orderbook features before deleting raw historical ZIP archives."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--archive-pattern", default="*.zip")
    parser.add_argument("--min-hours-per-day", type=int, default=20)
    parser.add_argument("--delete-raw", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    summary = validate_and_cleanup(
        raw_dir=args.raw_dir,
        feature_path=args.feature_path,
        archive_pattern=args.archive_pattern,
        min_hours_per_day=max(1, int(args.min_hours_per_day)),
        delete_raw=bool(args.delete_raw),
        execute=bool(args.execute),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not summary["blocking_errors"] else 2


def validate_and_cleanup(
    *,
    raw_dir: Path,
    feature_path: Path,
    archive_pattern: str,
    min_hours_per_day: int,
    delete_raw: bool,
    execute: bool,
) -> dict[str, Any]:
    archives = _raw_archives(raw_dir, archive_pattern)
    tmp_files = sorted(str(path) for path in raw_dir.glob("*.tmp"))
    summary: dict[str, Any] = {
        "raw_dir": str(raw_dir),
        "feature_path": str(feature_path),
        "archive_pattern": archive_pattern,
        "execute": execute,
        "delete_raw": delete_raw,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "raw_archive_count": len(archives),
        "raw_bytes": int(sum(item.path.stat().st_size for item in archives)),
        "tmp_files": tmp_files,
        "feature_rows": 0,
        "feature_start": None,
        "feature_end": None,
        "duplicate_feature_dates": 0,
        "source_after_feature_rows": 0,
        "all_null_numeric_columns": [],
        "days_below_min_hours": [],
        "raw_files_deleted": 0,
        "raw_bytes_deleted": 0,
        "blocking_errors": [],
        "warnings": [],
    }
    if tmp_files:
        summary["blocking_errors"].append("Raw directory still contains .tmp download files.")
    if not archives:
        summary["blocking_errors"].append("No raw archives matched the requested pattern.")
    if not feature_path.exists():
        summary["blocking_errors"].append("Feature parquet does not exist.")
        return summary

    frame = pd.read_parquet(feature_path)
    if "date" not in frame.columns:
        summary["blocking_errors"].append("Feature parquet is missing date column.")
        return summary
    if "source_min_ts" not in frame.columns or "source_max_ts" not in frame.columns:
        summary["blocking_errors"].append("Feature parquet is missing source_min_ts/source_max_ts metadata.")
        return summary

    feature_dates = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    source_min = pd.to_datetime(frame["source_min_ts"], utc=True, errors="coerce")
    source_max = pd.to_datetime(frame["source_max_ts"], utc=True, errors="coerce")
    summary["feature_rows"] = int(len(frame))
    summary["feature_start"] = str(feature_dates.min()) if not frame.empty else None
    summary["feature_end"] = str(feature_dates.max()) if not frame.empty else None
    summary["duplicate_feature_dates"] = int(feature_dates.duplicated().sum())
    summary["source_after_feature_rows"] = int((source_max > feature_dates).sum())
    numeric = frame.select_dtypes(include=["number", "bool"])
    summary["all_null_numeric_columns"] = [str(column) for column in numeric.columns if frame[column].isna().all()]

    if feature_dates.isna().any():
        summary["blocking_errors"].append("Feature parquet contains null/unparseable dates.")
    if summary["duplicate_feature_dates"]:
        summary["blocking_errors"].append("Feature parquet contains duplicate dates.")
    if summary["source_after_feature_rows"]:
        summary["blocking_errors"].append("Feature parquet has rows where source_max_ts is after the feature date.")
    if summary["all_null_numeric_columns"]:
        summary["blocking_errors"].append("Feature parquet contains all-null numeric columns.")

    by_source_day = source_min.dt.date.value_counts()
    for archive in archives:
        hours = int(by_source_day.get(archive.day, 0))
        if hours < min_hours_per_day:
            summary["days_below_min_hours"].append({"date": archive.day.isoformat(), "feature_hours": hours})
    if summary["days_below_min_hours"]:
        summary["blocking_errors"].append("One or more raw archive days have insufficient feature-hour coverage.")

    if not delete_raw:
        summary["warnings"].append("Dry retention check only. Pass --delete-raw --execute to delete after validation.")
        return summary
    if not execute:
        summary["warnings"].append("--delete-raw was requested without --execute; no files were deleted.")
        return summary
    if summary["blocking_errors"]:
        return summary

    for archive in archives:
        size = archive.path.stat().st_size
        archive.path.unlink()
        summary["raw_files_deleted"] += 1
        summary["raw_bytes_deleted"] += int(size)
    return summary


def _raw_archives(raw_dir: Path, archive_pattern: str) -> list[RawArchive]:
    archives: list[RawArchive] = []
    for path in sorted(raw_dir.glob(archive_pattern)):
        if path.suffix != ".zip":
            continue
        try:
            day = date.fromisoformat(path.name[:10])
        except ValueError:
            continue
        archives.append(RawArchive(path=path, day=day))
    return archives


if __name__ == "__main__":
    raise SystemExit(main())

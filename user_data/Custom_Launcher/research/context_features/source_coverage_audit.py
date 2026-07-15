from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.trader_confluence_feature_taxonomy import (  # noqa: E402
    source_detail,
)


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT = USER_DATA_DIR / "research_news_data" / "context_features" / "confluence_cache" / "trader_confluence_1h_latest.parquet"
DEFAULT_OUTPUT_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"

REQUIRED_SOURCE_BLOCKS = (
    "price_ohlcv",
    "structure_volume_profile",
    "structure_tlv2_support_resistance",
    "structure_bos_choch_market_structure",
    "structure_pattern_geometry",
    "orderbook_spot",
    "orderbook_bybit_linear",
    "orderbook_bybit_inverse",
    "context_article_source_activity",
    "context_topic_severity",
    "context_gdelt_events",
    "context_gkg_documents",
    "context_google_trends",
    "context_btc_etf_flows",
    "context_global_market_macro",
)

WINDOW_DEFINITIONS: dict[str, tuple[str, ...]] = {
    "full_confluence": REQUIRED_SOURCE_BLOCKS,
    "structure_orderbook": (
        "price_ohlcv",
        "structure_volume_profile",
        "structure_tlv2_support_resistance",
        "structure_bos_choch_market_structure",
        "structure_pattern_geometry",
        "orderbook_spot",
        "orderbook_bybit_linear",
        "orderbook_bybit_inverse",
    ),
    "orderbook_only": (
        "price_ohlcv",
        "orderbook_spot",
        "orderbook_bybit_linear",
        "orderbook_bybit_inverse",
    ),
    "gdelt_only": (
        "price_ohlcv",
        "context_gdelt_events",
    ),
    "gdelt_gkg_documents": (
        "price_ohlcv",
        "context_gdelt_events",
        "context_gkg_documents",
    ),
    "recent_live_news_context": (
        "price_ohlcv",
        "context_article_source_activity",
        "context_topic_severity",
    ),
}

BLOCK_NOTE_BY_PREFIX = {
    "price_ohlcv": "Price rows come from frozen OHLCV joined on 1h candle-close timestamps.",
    "structure_": "Structure rows come from the frozen structural parquet cache; no live collector or database is read.",
    "orderbook_": "Orderbook rows rely on frozen parquet present flags, min coverage, and source_max_ts <= candle date.",
    "context_": "Context rows rely on frozen context parquet and max_source_available_at <= candle date.",
}


@dataclass(frozen=True)
class Segment:
    start: pd.Timestamp
    end: pd.Timestamp
    rows: int

    @property
    def hours(self) -> float:
        return float((self.end - self.start).total_seconds() / 3600.0) + 1.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit frozen trader-confluence source-detail coverage and clean testing windows."
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="")
    parser.add_argument("--min-window-hours", type=float, default=168.0)
    parser.add_argument("--recent-min-window-hours", type=float, default=24.0)
    parser.add_argument("--min-source-coverage-ratio", type=float, default=0.01)
    parser.add_argument("--top-segments", type=int, default=5)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    tag = safe_tag(args.tag) if args.tag else pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    plan = {
        "mode": "setup-only unless --execute is supplied",
        "snapshot": str(args.snapshot),
        "output_dir": str(args.output_dir),
        "tag": tag,
        "reports": {
            "source_coverage_csv": str(args.output_dir / f"source_coverage_audit_{tag}.csv"),
            "source_coverage_md": str(args.output_dir / f"source_coverage_audit_{tag}.md"),
            "clean_windows_csv": str(args.output_dir / f"clean_testing_windows_{tag}.csv"),
            "clean_windows_md": str(args.output_dir / f"clean_testing_windows_{tag}.md"),
        },
        "inputs": "Reads only the frozen trader confluence parquet snapshot and adjacent metadata if present.",
        "timestamp_rule": "Coverage means timestamp-safe non-null/present rows. Usable windows require a source-detail usable mask, which is stricter for sparse/zero-filled context blocks.",
        "window_definitions": {name: list(blocks) for name, blocks in WINDOW_DEFINITIONS.items()},
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.snapshot.exists():
        raise FileNotFoundError(args.snapshot)

    frame = load_snapshot(args.snapshot)
    meta = load_meta(args.snapshot)
    audit, masks = build_source_audit(frame, float(args.min_source_coverage_ratio))
    windows = build_window_candidates(
        frame,
        audit,
        masks,
        min_window_hours=float(args.min_window_hours),
        recent_min_window_hours=float(args.recent_min_window_hours),
        top_segments=max(1, int(args.top_segments)),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage_csv = args.output_dir / f"source_coverage_audit_{tag}.csv"
    coverage_md = args.output_dir / f"source_coverage_audit_{tag}.md"
    windows_csv = args.output_dir / f"clean_testing_windows_{tag}.csv"
    windows_md = args.output_dir / f"clean_testing_windows_{tag}.md"
    audit.to_csv(coverage_csv, index=False)
    windows.to_csv(windows_csv, index=False)
    coverage_md.write_text(source_markdown(args.snapshot, audit, meta), encoding="utf-8")
    windows_md.write_text(windows_markdown(args.snapshot, windows, audit, meta), encoding="utf-8")

    print(
        json.dumps(
            {
                "source_coverage_csv": str(coverage_csv),
                "source_coverage_md": str(coverage_md),
                "clean_windows_csv": str(windows_csv),
                "clean_windows_md": str(windows_md),
                "source_blocks": int(len(audit)),
                "window_rows": int(len(windows)),
            },
            indent=2,
        )
    )
    return 0


def load_snapshot(path: Path) -> DataFrame:
    frame = pd.read_parquet(path)
    if "date" not in frame:
        raise ValueError(f"{path} has no date column")
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    return frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def load_meta(snapshot_path: Path) -> dict[str, Any]:
    candidates = [
        snapshot_path.with_suffix(".meta.json"),
        snapshot_path.with_name(snapshot_path.name.replace("_latest.parquet", ".meta.json")),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {"meta_read_error": str(candidate)}
    return {}


def build_source_audit(frame: DataFrame, min_source_coverage_ratio: float) -> tuple[DataFrame, dict[str, dict[str, Series]]]:
    dates = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    detail_columns: dict[str, list[str]] = {}
    for column in frame.columns:
        detail_columns.setdefault(source_detail(str(column)), []).append(str(column))
    for block in REQUIRED_SOURCE_BLOCKS:
        detail_columns.setdefault(block, [])

    rows: list[dict[str, Any]] = []
    masks: dict[str, dict[str, Series]] = {}
    total_rows = max(1, len(frame))
    for block in sorted(detail_columns):
        columns = [column for column in detail_columns[block] if column in frame.columns]
        numeric_columns = [
            column
            for column in columns
            if pd.api.types.is_numeric_dtype(frame[column]) and is_signal_column(column, block)
        ]
        coverage_mask = coverage_mask_for_block(frame, block, columns)
        active_mask = coverage_mask & active_mask_for_columns(frame, numeric_columns)
        usable_mask = usable_mask_for_block(frame, block, coverage_mask, active_mask)
        masks[block] = {"coverage": coverage_mask, "active": active_mask, "usable": usable_mask}
        coverage_segments = contiguous_segments(dates, coverage_mask)
        active_segments = contiguous_segments(dates, active_mask)
        usable_segments = contiguous_segments(dates, usable_mask)
        longest_coverage = longest_segment(coverage_segments)
        longest_active = longest_segment(active_segments)
        longest_usable = longest_segment(usable_segments)
        coverage_rows = int(coverage_mask.sum())
        active_rows = int(active_mask.sum())
        usable_rows = int(usable_mask.sum())
        status = block_status(block, coverage_rows, active_rows, usable_rows, total_rows, min_source_coverage_ratio)
        rows.append(
            {
                "source_detail_block": block,
                "required_block": bool(block in REQUIRED_SOURCE_BLOCKS),
                "status": status,
                "columns_total": int(len(columns)),
                "numeric_signal_columns": int(len(numeric_columns)),
                "coverage_rows": coverage_rows,
                "coverage_ratio": coverage_rows / total_rows,
                "first_coverage_date": fmt_ts(dates[coverage_mask].min()) if coverage_rows else "",
                "last_coverage_date": fmt_ts(dates[coverage_mask].max()) if coverage_rows else "",
                "coverage_segments": int(len(coverage_segments)),
                "longest_coverage_start": fmt_ts(longest_coverage.start) if longest_coverage else "",
                "longest_coverage_end": fmt_ts(longest_coverage.end) if longest_coverage else "",
                "longest_coverage_hours": longest_coverage.hours if longest_coverage else 0.0,
                "active_nonzero_rows": active_rows,
                "active_nonzero_ratio": active_rows / total_rows,
                "first_active_nonzero_date": fmt_ts(dates[active_mask].min()) if active_rows else "",
                "last_active_nonzero_date": fmt_ts(dates[active_mask].max()) if active_rows else "",
                "active_nonzero_segments": int(len(active_segments)),
                "longest_active_nonzero_start": fmt_ts(longest_active.start) if longest_active else "",
                "longest_active_nonzero_end": fmt_ts(longest_active.end) if longest_active else "",
                "longest_active_nonzero_hours": longest_active.hours if longest_active else 0.0,
                "usable_rows": usable_rows,
                "usable_ratio": usable_rows / total_rows,
                "first_usable_date": fmt_ts(dates[usable_mask].min()) if usable_rows else "",
                "last_usable_date": fmt_ts(dates[usable_mask].max()) if usable_rows else "",
                "usable_segments": int(len(usable_segments)),
                "longest_usable_start": fmt_ts(longest_usable.start) if longest_usable else "",
                "longest_usable_end": fmt_ts(longest_usable.end) if longest_usable else "",
                "longest_usable_hours": longest_usable.hours if longest_usable else 0.0,
                "sample_columns": ", ".join(columns[:10]),
                "timestamp_safety_note": timestamp_note(block, frame),
                "missing_data_caution": missing_data_caution(block, coverage_rows, active_rows, usable_rows, total_rows, coverage_segments),
            }
        )
    return pd.DataFrame(rows), masks


def coverage_mask_for_block(frame: DataFrame, block: str, columns: list[str]) -> Series:
    if block == "price_ohlcv":
        price_columns = [column for column in ("open", "high", "low", "close", "volume") if column in frame]
        if price_columns:
            return frame[price_columns].notna().all(axis=1)
    present_column = {
        "orderbook_spot": "ob_spot_present",
        "orderbook_bybit_linear": "ob_linear_present",
        "orderbook_bybit_inverse": "ob_inverse_present",
    }.get(block)
    if present_column and present_column in frame:
        prefix = {
            "orderbook_spot": "ob_spot_",
            "orderbook_bybit_linear": "ob_linear_",
            "orderbook_bybit_inverse": "ob_inverse_",
        }[block]
        base = pd.to_numeric(frame[present_column], errors="coerce").fillna(0.0).gt(0.0)
        low_ok = ~numeric_column(frame, f"{prefix}low_coverage", 0.0).gt(0.0)
        future_ok = ~numeric_column(frame, f"{prefix}source_future_violation", 0.0).gt(0.0)
        max_ts = datetime_column(frame, f"{prefix}source_max_ts")
        date = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        max_ts_ok = max_ts.isna() | max_ts.le(date)
        return base & low_ok & future_ok & max_ts_ok
    if block.startswith("context_") and "context_present" in frame:
        base = pd.to_numeric(frame["context_present"], errors="coerce").fillna(0.0).gt(0.0)
        base &= ~numeric_column(frame, "context_source_future_violation", 0.0).gt(0.0)
        max_available = datetime_column(frame, "ctx_max_source_available_at")
        date = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        base &= max_available.isna() | max_available.le(date)
    elif block.startswith("structure_") and "structure_present" in frame:
        base = pd.to_numeric(frame["structure_present"], errors="coerce").fillna(0.0).gt(0.0)
    else:
        base = pd.Series(True, index=frame.index)
    useful = [column for column in columns if column != "date" and is_signal_column(column, block)]
    if not useful:
        return pd.Series(False, index=frame.index)
    return base & frame[useful].notna().any(axis=1)


def usable_mask_for_block(frame: DataFrame, block: str, coverage_mask: Series, active_mask: Series) -> Series:
    if block in {"price_ohlcv"} or block.startswith(("structure_", "orderbook_")):
        return coverage_mask
    if block.startswith("context_"):
        if block == "context_global_market_macro":
            explicit_availability = numeric_column(frame, "ctx_global_metrics_available", 0.0).gt(0.0)
            return coverage_mask & (active_mask | explicit_availability)
        return coverage_mask & active_mask
    return coverage_mask


def numeric_column(frame: DataFrame, column: str, default: float = np.nan) -> Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def datetime_column(frame: DataFrame, column: str) -> Series:
    if column not in frame:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    return pd.to_datetime(frame[column], utc=True, errors="coerce")


def active_mask_for_columns(frame: DataFrame, numeric_columns: list[str]) -> Series:
    if not numeric_columns:
        return pd.Series(False, index=frame.index)
    values = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return values.fillna(0.0).abs().gt(1e-12).any(axis=1)


def is_signal_column(column: str, block: str) -> bool:
    lower = column.lower()
    if lower == "date":
        return False
    if lower.startswith("future_") or lower.startswith(("hit_", "time_to_")):
        return False
    if lower.endswith(("_next_6h", "_next_24h")):
        return False
    if "generated_at" in lower or "schema_version" in lower:
        return False
    diagnostic_tokens = (
        "source_future_violation",
        "source_age",
        "source_min_ts",
        "source_max_ts",
        "min_source_available_at",
        "max_source_available_at",
        "row_present",
        "low_coverage",
    )
    if any(token in lower for token in diagnostic_tokens):
        return False
    if lower.endswith("_present") and block != "price_ohlcv":
        return False
    return True


def build_window_candidates(
    frame: DataFrame,
    audit: DataFrame,
    masks: dict[str, dict[str, Series]],
    *,
    min_window_hours: float,
    recent_min_window_hours: float,
    top_segments: int,
) -> DataFrame:
    dates = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    rows: list[dict[str, Any]] = []
    for window_type, blocks in WINDOW_DEFINITIONS.items():
        mask = combined_usable_mask(frame, masks, blocks)
        segments = sorted(contiguous_segments(dates, mask), key=lambda item: (item.hours, item.rows), reverse=True)
        if window_type == "recent_live_news_context" and segments:
            segments = [max(segments, key=lambda item: item.end)]
        limit = 1 if window_type == "recent_live_news_context" else top_segments
        minimum = recent_min_window_hours if window_type == "recent_live_news_context" else min_window_hours
        missing = missing_or_unusable_blocks(masks, blocks)
        if not segments:
            rows.append(window_row(window_type, "invalid_no_overlap", blocks, None, missing, audit))
            continue
        for rank, segment in enumerate(segments[:limit], start=1):
            status = "candidate" if segment.hours >= minimum else "too_short_for_default_minimum"
            rows.append(window_row(window_type, status, blocks, segment, missing, audit, rank=rank, minimum_hours=minimum))

    for _, row in audit[audit["required_block"].astype(bool)].iterrows():
        if str(row["status"]) in {"missing", "too_sparse", "no_nonzero_signal", "not_usable", "covered_mostly_inactive"}:
            rows.append(
                {
                    "window_type": "invalid_too_sparse_source",
                    "rank": "",
                    "status": str(row["status"]),
                    "required_blocks": str(row["source_detail_block"]),
                    "start": "",
                    "end": "",
                    "hours": 0.0,
                    "rows": int(row["coverage_rows"]),
                    "minimum_hours": min_window_hours,
                    "missing_or_sparse_blocks": str(row["source_detail_block"]),
                    "timestamp_safety_notes": str(row["timestamp_safety_note"]),
                    "missing_data_cautions": str(row["missing_data_caution"]),
                }
            )
    return pd.DataFrame(rows)


def combined_usable_mask(frame: DataFrame, masks: dict[str, dict[str, Series]], blocks: tuple[str, ...]) -> Series:
    mask = pd.Series(True, index=frame.index)
    for block in blocks:
        block_mask = masks.get(block, {}).get("usable")
        if block_mask is None:
            return pd.Series(False, index=frame.index)
        mask &= block_mask.reindex(frame.index).fillna(False)
    return mask


def missing_or_unusable_blocks(masks: dict[str, dict[str, Series]], blocks: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for block in blocks:
        block_mask = masks.get(block, {}).get("usable")
        if block_mask is None or int(block_mask.sum()) == 0:
            missing.append(block)
    return missing


def window_row(
    window_type: str,
    status: str,
    blocks: tuple[str, ...],
    segment: Segment | None,
    missing: list[str],
    audit: DataFrame,
    *,
    rank: int | str = "",
    minimum_hours: float = 0.0,
) -> dict[str, Any]:
    notes = [timestamp_note_for_window(blocks)]
    cautions = source_cautions_for_window(audit, blocks)
    sparse = sparse_blocks(audit, blocks)
    if missing:
        cautions.append("Missing coverage blocks: " + ", ".join(missing))
    missing_or_sparse = sorted(set(missing + sparse))
    effective_status = status
    if status == "candidate" and missing_or_sparse:
        effective_status = "conditional_sparse_source_candidate"
    return {
        "window_type": window_type,
        "rank": rank,
        "status": effective_status,
        "required_blocks": ", ".join(blocks),
        "start": fmt_ts(segment.start) if segment else "",
        "end": fmt_ts(segment.end) if segment else "",
        "hours": segment.hours if segment else 0.0,
        "rows": int(segment.rows) if segment else 0,
        "minimum_hours": minimum_hours,
        "missing_or_sparse_blocks": ", ".join(missing_or_sparse),
        "required_block_usable_rows": usable_rows_summary(audit, blocks),
        "required_block_active_rows": active_rows_summary(audit, blocks),
        "timestamp_safety_notes": " ".join(notes),
        "missing_data_cautions": " ".join(cautions),
    }


def usable_rows_summary(audit: DataFrame, blocks: tuple[str, ...]) -> str:
    audit_by_block = {str(row["source_detail_block"]): row for _, row in audit.iterrows()}
    return "; ".join(f"{block}={int(audit_by_block.get(block, {}).get('usable_rows', 0))}" for block in blocks)


def active_rows_summary(audit: DataFrame, blocks: tuple[str, ...]) -> str:
    audit_by_block = {str(row["source_detail_block"]): row for _, row in audit.iterrows()}
    return "; ".join(f"{block}={int(audit_by_block.get(block, {}).get('active_nonzero_rows', 0))}" for block in blocks)


def contiguous_segments(dates: Series, mask: Series) -> list[Segment]:
    work = DataFrame({"date": dates, "mask": mask.fillna(False).astype(bool)})
    work = work[work["mask"]].sort_values("date")
    if work.empty:
        return []
    segments: list[Segment] = []
    start = work.iloc[0]["date"]
    prev = start
    rows = 1
    for current in work["date"].iloc[1:]:
        if current - prev > pd.Timedelta(hours=1):
            segments.append(Segment(pd.Timestamp(start), pd.Timestamp(prev), rows))
            start = current
            rows = 1
        else:
            rows += 1
        prev = current
    segments.append(Segment(pd.Timestamp(start), pd.Timestamp(prev), rows))
    return segments


def longest_segment(segments: list[Segment]) -> Segment | None:
    if not segments:
        return None
    return max(segments, key=lambda item: (item.hours, item.rows))


def block_status(block: str, coverage_rows: int, active_rows: int, usable_rows: int, total_rows: int, min_source_coverage_ratio: float) -> str:
    if block not in REQUIRED_SOURCE_BLOCKS and coverage_rows == 0:
        return "non_signal_metadata"
    if coverage_rows == 0:
        return "missing"
    if block in REQUIRED_SOURCE_BLOCKS and usable_rows == 0:
        return "not_usable"
    if block in REQUIRED_SOURCE_BLOCKS and usable_rows / max(1, total_rows) < min_source_coverage_ratio:
        return "too_sparse"
    if active_rows == 0 and block in REQUIRED_SOURCE_BLOCKS:
        return "no_nonzero_signal"
    if active_rows / max(1, coverage_rows) < 0.05 and block.startswith("context_"):
        return "covered_mostly_inactive"
    return "covered_active"


def timestamp_note(block: str, frame: DataFrame) -> str:
    base_note = next((note for prefix, note in BLOCK_NOTE_BY_PREFIX.items() if block.startswith(prefix)), "")
    if block.startswith("orderbook_"):
        prefix = {
            "orderbook_spot": "ob_spot_",
            "orderbook_bybit_linear": "ob_linear_",
            "orderbook_bybit_inverse": "ob_inverse_",
        }.get(block, "")
        future_col = f"{prefix}source_future_violation"
        low_col = f"{prefix}low_coverage"
        extras = []
        if future_col in frame:
            extras.append(f"future_violation_rows={int(pd.to_numeric(frame[future_col], errors='coerce').fillna(0.0).gt(0.0).sum())}")
        if low_col in frame:
            extras.append(f"low_coverage_rows={int(pd.to_numeric(frame[low_col], errors='coerce').fillna(0.0).gt(0.0).sum())}")
        return f"{base_note} {'; '.join(extras)}".strip()
    if block.startswith("context_") and "context_source_future_violation" in frame:
        rows = int(pd.to_numeric(frame["context_source_future_violation"], errors="coerce").fillna(0.0).gt(0.0).sum())
        return f"{base_note} future_violation_rows={rows}".strip()
    return base_note


def missing_data_caution(block: str, coverage_rows: int, active_rows: int, usable_rows: int, total_rows: int, segments: list[Segment]) -> str:
    if coverage_rows == 0:
        return "No coverage in the snapshot; do not use this block for tests."
    cautions: list[str] = []
    if len(segments) > 1:
        cautions.append(f"{len(segments)} coverage segments; do not carry state through gaps.")
    if coverage_rows < total_rows:
        cautions.append("Partial coverage; tests must intersect this block with required source windows.")
    if active_rows == 0:
        cautions.append("Coverage exists but no nonzero numeric signal was found; verify whether zero is meaningful before testing.")
    if usable_rows == 0:
        cautions.append("No usable source-detail rows under the conservative clean-window mask.")
    elif usable_rows < coverage_rows:
        cautions.append("Usable rows are fewer than coverage rows; zero-filled or inactive periods are excluded from clean windows.")
    return " ".join(cautions) if cautions else "Coverage appears continuous for its available range."


def timestamp_note_for_window(blocks: tuple[str, ...]) -> str:
    notes: list[str] = []
    if any(block.startswith("orderbook_") for block in blocks):
        notes.append("Orderbook windows use present flags that exclude low-coverage/future source rows.")
    if any(block.startswith("context_") for block in blocks):
        notes.append("Context windows require source availability timestamps not later than the candle date.")
    if any(block.startswith("structure_") for block in blocks):
        notes.append("Structure windows use frozen structural cache rows and should not be forward-filled across gaps.")
    return " ".join(notes) if notes else "Uses frozen snapshot candle timestamps."


def source_cautions_for_window(audit: DataFrame, blocks: tuple[str, ...]) -> list[str]:
    cautions: list[str] = []
    audit_by_block = {str(row["source_detail_block"]): row for _, row in audit.iterrows()}
    for block in blocks:
        row = audit_by_block.get(block)
        if row is None:
            cautions.append(f"{block}: not found in audit.")
            continue
        if str(row.get("status")) != "covered_active":
            cautions.append(f"{block}: {row.get('status')}.")
        if int(row.get("coverage_segments", 0)) > 1:
            cautions.append(f"{block}: segmented coverage.")
    return cautions


def sparse_blocks(audit: DataFrame, blocks: tuple[str, ...]) -> list[str]:
    audit_by_block = {str(row["source_detail_block"]): row for _, row in audit.iterrows()}
    sparse: list[str] = []
    for block in blocks:
        row = audit_by_block.get(block)
        if row is not None and str(row.get("status")) not in {"covered_active", "non_signal_metadata"}:
            sparse.append(block)
    return sparse


def source_markdown(snapshot: Path, audit: DataFrame, meta: dict[str, Any]) -> str:
    required = audit[audit["required_block"].astype(bool)].copy()
    lines = [
        "# Source Coverage Audit",
        "",
        f"- Snapshot: `{snapshot}`",
        f"- Rows: `{meta.get('output_rows', 'unknown')}`; columns: `{meta.get('output_columns', 'unknown')}`",
        "- Reads frozen parquet artifacts only; no live databases were queried.",
        "- Coverage range means timestamp-safe non-null/present source rows.",
        "- Active range means at least one nonzero numeric signal in the source-detail block.",
        "- Usable range is the conservative mask used for clean windows. Sparse/zero-filled context blocks require usable source activity unless a source-specific availability flag exists.",
        "",
        "## Required Source-Detail Blocks",
        "",
        markdown_table(
            required,
            [
                "source_detail_block",
                "status",
                "coverage_rows",
                "first_coverage_date",
                "last_coverage_date",
                "active_nonzero_rows",
                "first_active_nonzero_date",
                "last_active_nonzero_date",
                "usable_rows",
                "first_usable_date",
                "last_usable_date",
                "coverage_segments",
            ],
        ),
        "",
        "## Timestamp Safety And Missing-Data Cautions",
        "",
    ]
    for _, row in required.iterrows():
        lines.append(f"- `{row['source_detail_block']}`: {row['timestamp_safety_note']} {row['missing_data_caution']}")
    lines.extend(["", "## All Source-Detail Blocks", ""])
    lines.append(
        markdown_table(
            audit,
            [
                "source_detail_block",
                "required_block",
                "status",
                "columns_total",
                "coverage_rows",
                "active_nonzero_rows",
                "usable_rows",
                "longest_coverage_start",
                "longest_coverage_end",
                "longest_coverage_hours",
            ],
        )
    )
    return "\n".join(lines) + "\n"


def windows_markdown(snapshot: Path, windows: DataFrame, audit: DataFrame, meta: dict[str, Any]) -> str:
    lines = [
        "# Clean Testing Window Candidates",
        "",
        f"- Snapshot: `{snapshot}`",
        f"- Snapshot output range: `{nested(meta, 'validation', 'output', 'first_date')}` to `{nested(meta, 'validation', 'output', 'last_date')}`",
        "- Candidate windows are contiguous 1h rows where every listed source-detail block has coverage in the frozen snapshot.",
        "- Invalid/too-sparse rows mark source-detail blocks that should not be silently treated as full confluence.",
        "",
        markdown_table(
            windows,
            [
                "window_type",
                "rank",
                "status",
                "required_blocks",
                "start",
                "end",
                "hours",
                "rows",
                "missing_or_sparse_blocks",
                "required_block_usable_rows",
            ],
        ),
        "",
        "## Notes",
        "",
    ]
    for _, row in windows.iterrows():
        lines.append(f"- `{row['window_type']}` {row['rank']}: {row['timestamp_safety_notes']} {row['missing_data_cautions']}")
    return "\n".join(lines) + "\n"


def markdown_table(frame: DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    available = [column for column in columns if column in frame.columns]
    rows = [available, ["---" for _ in available]]
    for _, item in frame[available].iterrows():
        rows.append([fmt_cell(item[column]) for column in available])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def fmt_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def fmt_ts(value: Any) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).isoformat()


def nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return "unknown"
        current = current.get(key, "unknown")
    return current


def safe_tag(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip())
    return safe or "latest"


if __name__ == "__main__":
    raise SystemExit(main())

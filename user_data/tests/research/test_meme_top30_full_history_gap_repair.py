# ruff: noqa: S101

"""Tests for top-30 meme OHLCV continuity auditing and exact gap repair."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    meme_top30_full_history_gap_repair as repair,
)


def frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = len(dates)
    return pd.DataFrame(
        {
            "date": dates,
            "open": np.full(rows, 100.0),
            "high": np.full(rows, 101.0),
            "low": np.full(rows, 99.0),
            "close": np.full(rows, 100.5),
            "volume": np.ones(rows),
        }
    )


def test_audit_frame_finds_exact_fixed_timeframe_gap(tmp_path: Path) -> None:
    path = tmp_path / "sample.feather"
    source = frame(
        pd.DatetimeIndex(
            [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:01:00Z",
                "2026-01-01T00:05:00Z",
            ]
        )
    )
    source.to_feather(path)

    summary, gaps = repair.audit_frame("DOGE/USDT:USDT", "1m", path, source)

    assert summary["gap_events"] == 1
    assert summary["missing_candle_intervals"] == 3
    assert gaps[0]["interval_start_utc"] == pd.Timestamp("2026-01-01T00:02:00Z")
    assert gaps[0]["interval_end_exclusive_utc"] == pd.Timestamp("2026-01-01T00:05:00Z")


def test_audit_frame_finds_missing_calendar_month(tmp_path: Path) -> None:
    path = tmp_path / "monthly.feather"
    source = frame(pd.DatetimeIndex(["2026-01-01T00:00:00Z", "2026-03-01T00:00:00Z"]))
    source.to_feather(path)

    summary, gaps = repair.audit_frame("DOGE/USDT:USDT", "1M", path, source)

    assert summary["gap_events"] == 1
    assert summary["missing_candle_intervals"] == 1
    assert gaps[0]["interval_start_utc"] == pd.Timestamp("2026-02-01T00:00:00Z")


def test_interval_quality_requires_exact_continuous_range() -> None:
    source = frame(pd.date_range("2026-01-01", periods=60, freq="1min", tz="UTC"))

    quality = repair.interval_quality(
        source,
        timeframe="1m",
        start=pd.Timestamp("2026-01-01T00:00:00Z"),
        end=pd.Timestamp("2026-01-01T01:00:00Z"),
        expected_rows=60,
    )

    assert quality["complete"]


def test_monthly_filename_uses_freqtrade_normalization() -> None:
    assert repair.data_path("DOGE/USDT:USDT", "1M").name.endswith("-1Mo-futures.feather")

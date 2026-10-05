# ruff: noqa: S101

"""Tests for targeted CPI BTC one-minute coverage repair."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_event_cpi_links_one_minute_acquisition as acquisition,
)


def candles(start: str, periods: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range(start, periods=periods, freq="1min", tz="UTC"),
            "open": np.full(periods, 100.0),
            "high": np.full(periods, 101.0),
            "low": np.full(periods, 99.0),
            "close": np.full(periods, 100.5),
            "volume": np.ones(periods),
        }
    )


def test_missing_sample_rows_selects_only_incomplete_frozen_hours() -> None:
    samples = pd.DataFrame(
        [
            {
                "sample_id": "event|event|0",
                "event_id": "event",
                "sample_type": "event",
                "control_rank": 0,
                "sample_anchor_utc": pd.Timestamp("2026-01-01T00:00:00Z"),
            },
            {
                "sample_id": "event|control|1",
                "event_id": "event",
                "sample_type": "control",
                "control_rank": 1,
                "sample_anchor_utc": pd.Timestamp("2026-01-08T00:00:00Z"),
            },
        ]
    )
    frame = candles("2026-01-01T00:00:00Z", 60)

    missing = acquisition.missing_sample_rows(samples, frame)

    assert list(missing["sample_id"]) == ["event|control|1"]
    assert missing.iloc[0]["prior_coverage_status"] == "missing_anchor"
    assert missing.iloc[0]["interval_end_exclusive_utc"] - missing.iloc[0][
        "interval_start_utc"
    ] == pd.Timedelta(hours=1)


def test_merge_missing_intervals_combines_touching_hours_only() -> None:
    missing = pd.DataFrame(
        [
            {
                "sample_id": "one",
                "event_id": "event_one",
                "interval_start_utc": pd.Timestamp("2026-01-01T00:00:00Z"),
                "interval_end_exclusive_utc": pd.Timestamp("2026-01-01T01:00:00Z"),
            },
            {
                "sample_id": "two",
                "event_id": "event_two",
                "interval_start_utc": pd.Timestamp("2026-01-01T01:00:00Z"),
                "interval_end_exclusive_utc": pd.Timestamp("2026-01-01T02:00:00Z"),
            },
            {
                "sample_id": "three",
                "event_id": "event_three",
                "interval_start_utc": pd.Timestamp("2026-01-03T00:00:00Z"),
                "interval_end_exclusive_utc": pd.Timestamp("2026-01-03T01:00:00Z"),
            },
        ]
    )

    intervals = acquisition.merge_missing_intervals(
        missing, Path("C:/FreqTradeStuff/user_data/data/binance/futures/test.feather")
    )

    assert list(intervals["expected_rows"]) == [120, 60]
    assert intervals.iloc[0]["sample_ids"] == "one|two"
    assert intervals.iloc[0]["event_ids"] == "event_one|event_two"

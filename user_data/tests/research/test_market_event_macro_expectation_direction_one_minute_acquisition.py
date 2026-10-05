# ruff: noqa: S101

from __future__ import annotations

import importlib

import pandas as pd
from pandas import DataFrame


MODULE = (
    "user_data.Custom_Launcher.research.context_features."
    "market_event_macro_expectation_direction_one_minute_acquisition"
)
acquisition = importlib.import_module(MODULE)


def test_pair_spanning_intervals_combines_one_target_write_per_pair() -> None:
    missing = DataFrame(
        {
            "pair": ["ETH/USDT:USDT", "ETH/USDT:USDT", "BTC/USDT:USDT"],
            "expectation_episode_id": ["one", "two", "three"],
            "interval_start_utc": pd.to_datetime(
                [
                    "2026-04-03T12:30:00Z",
                    "2026-06-10T12:30:00Z",
                    "2026-03-06T13:30:00Z",
                ],
                utc=True,
            ),
            "interval_end_exclusive_utc": pd.to_datetime(
                [
                    "2026-04-03T13:30:00Z",
                    "2026-06-10T13:30:00Z",
                    "2026-03-06T14:30:00Z",
                ],
                utc=True,
            ),
            "target_file": ["eth.feather", "eth.feather", "btc.feather"],
        }
    )
    intervals = acquisition.pair_spanning_intervals(missing).set_index("pair")
    assert len(intervals) == 2
    assert intervals.loc["ETH/USDT:USDT", "expected_rows"] == 97_980
    assert intervals.loc["BTC/USDT:USDT", "expected_rows"] == 60

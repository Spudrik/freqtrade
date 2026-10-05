# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation20_one_minute_acquisition as g20a,
)


def test_merge_intervals_combines_only_overlapping_same_pair_windows() -> None:
    frame = pd.DataFrame(
        {
            "episode_id": ["a", "b", "c"],
            "pair": ["BTC/USDT:USDT", "BTC/USDT:USDT", "ETH/USDT:USDT"],
            "interval_start_utc": pd.to_datetime(
                ["2026-01-01T00:00Z", "2026-01-01T01:00Z", "2026-01-01T01:00Z"]
            ),
            "interval_end_exclusive_utc": pd.to_datetime(
                ["2026-01-01T02:00Z", "2026-01-01T03:00Z", "2026-01-01T03:00Z"]
            ),
        }
    )

    result = g20a.merge_intervals(frame)

    assert len(result) == 2
    btc = result.loc[result["pair"].eq("BTC/USDT:USDT")].iloc[0]
    assert btc["episode_ids"] == "a|b"
    assert btc["expected_rows"] == 180


def test_acquisition_worker_cap_is_bounded() -> None:
    assert g20a.MAXIMUM_WORKERS == 2

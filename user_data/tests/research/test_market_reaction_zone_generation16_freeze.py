# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation16_freeze as g16,
)


def test_main_batch_retains_real_breadth() -> None:
    routes = g16.main_routes()

    assert len(routes) >= 5
    assert len({item["route_id"] for item in routes}) == len(routes)
    assert any(item["route_id"].startswith("freqai") for item in routes)


def test_episode_freeze_never_reads_future_outcomes() -> None:
    forbidden = ("excursion", "future_", "time_to_", "dwell", "crossings")

    assert not any(
        token in column for column in g16.EPISODE_COLUMNS for token in forbidden
    )


def test_sample_selection_requires_every_frozen_stratum_and_distinct_pairs() -> None:
    rows = []
    number = 0
    for cohort in ("normal", "meme"):
        for period in g16.PERIODS[cohort]:
            for timeframe in g16.ONE_MINUTE_TIMEFRAMES:
                rows.append(
                    {
                        "cohort": cohort,
                        "analysis_period": period,
                        "source_timeframe": timeframe,
                        "pair": f"PAIR{number}/USDT:USDT",
                        "episode_id": f"episode-{number}",
                        "selection_hash": f"{number:064d}",
                    }
                )
                number += 1
    sample = g16.select_one_minute_sample(pd.DataFrame(rows))

    assert len(sample) == 12
    assert sample["pair"].nunique() == 12
    assert sample["sample_stratum"].nunique() == 12

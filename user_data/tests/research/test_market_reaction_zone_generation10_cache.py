# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation10_cache as cache,
)


def test_activity_regime_uses_frozen_band_meanings() -> None:
    frame = pd.DataFrame(
        {
            "g8_participation_duration__relative_volume_band": [
                -1.0,
                0.0,
                1.0,
                np.nan,
            ]
        }
    )

    assert cache.activity_regime(frame).tolist() == [
        "quiet",
        "typical",
        "active",
        "unavailable",
    ]


def test_rename_block_keeps_only_declared_features() -> None:
    source = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"], utc=True),
            "source__one": [1.0],
            "source__two": [2.0],
            "source__ignored": [3.0],
        }
    )

    result = cache.rename_block(
        source,
        source_prefix="source",
        target_block="target",
        suffixes=("one", "two"),
    )

    assert list(result) == ["date", "target__one", "target__two"]


def test_support_decision_requires_five_scorable_pairs() -> None:
    rows = []
    for mechanism in ("relative_volume", "participation_duration"):
        for period in cache.g10z.CONFIRMATION_PERIODS:
            for index in range(5):
                rows.append(
                    {
                        "pair": f"P{index}",
                        "mechanism": mechanism,
                        "period": period,
                        "rows": cache.g7f.MIN_PAIR_SCORABLE_ROWS,
                        "activity_quiet": 1,
                        "activity_typical": 1,
                        "activity_active": 1,
                    }
                )

    decisions = cache.validate_support(pd.DataFrame.from_records(rows))

    assert len(decisions) == 4
    assert all(item["supported"] for item in decisions)

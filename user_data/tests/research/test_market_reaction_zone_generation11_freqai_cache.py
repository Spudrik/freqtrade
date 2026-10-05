# ruff: noqa: S101

from __future__ import annotations

import pandas as pd
import pytest

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation11_freqai_cache as cache,
)


def test_feature_contract_rejects_outcomes() -> None:
    with pytest.raises(ValueError, match="Outcome columns"):
        cache.validate_feature_column_contract(["pair", "volume_ratio_h4"])


def test_anchor_selection_uses_nearest_then_eight_hour_cooldown() -> None:
    rows = []
    for hour, distance in ((0, 0.2), (0, 0.1), (4, 0.05), (8, 0.3)):
        rows.append(
            {
                "cohort": "normal",
                "pair": "A/USDT:USDT",
                "event_time": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(hours=hour),
                "period": "validation_early",
                "source_timeframe": "1h" if distance != 0.1 else "4h",
                "level_family": f"family_{distance}",
                "control": "actual",
                "contact_close_distance_atr": distance,
                "level_score": 1.0,
            }
        )
    frame = pd.DataFrame.from_records(rows)

    result = cache.select_independent_anchors(frame)

    assert len(result) == 2
    assert result.iloc[0]["contact_close_distance_atr"] == 0.1
    assert result.iloc[1]["event_time"] == pd.Timestamp("2025-01-01 08:00", tz="UTC")


def test_target_reaction_requires_price_and_volume() -> None:
    rows = []
    for index, (excursion, volume) in enumerate(((0.6, 1.3), (0.4, 1.3), (0.6, 1.2))):
        row = {
            "cohort": "normal",
            "pair": "A/USDT:USDT",
            "event_time": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(hours=8 * index),
            "period": "validation_early",
            "source_timeframe": "1h",
            "level_family": "generic_prior_range",
            "level_name": "prior_high",
            "representation": "line",
            "control": "actual",
            "contact_close_distance_atr": 0.1,
            "level_score": 1.0,
            "zone_half_width_atr": 0.2,
            "market_group": "test",
            "smart_contract_platform": False,
        }
        for horizon in cache.g11f.g11z.HORIZONS:
            row[f"abs_excursion_atr_h{horizon}"] = excursion
            row[f"volume_ratio_h{horizon}"] = volume
        rows.append(row)

    targets, _ = cache.target_frame(pd.DataFrame.from_records(rows))

    assert targets["&-g11_reaction_h1"].tolist() == [1.0, 0.0, 0.0]

# ruff: noqa: S101

from __future__ import annotations

import pandas as pd

from user_data.strategies.MarketReactionZoneFreqAIGeneration12Strategy import (
    MarketReactionZoneG12ConfigurableFreqAIResearchStrategy,
)


def test_generation12_strategy_selects_only_frozen_exact_columns(monkeypatch) -> None:
    strategy = object.__new__(
        MarketReactionZoneG12ConfigurableFreqAIResearchStrategy
    )
    strategy.config = {
        "market_reaction_zone_g12": {
            "feature_columns": ["block__first", "block__second"]
        }
    }
    cache = pd.DataFrame(
        {
            "block__first": [1.0, 2.0],
            "block__second": [3.0, 4.0],
            "block__unselected": [5.0, 6.0],
        }
    )
    monkeypatch.setattr(
        strategy,
        "_aligned_feature_cache",
        lambda dataframe, pair: cache,
    )
    frame = pd.DataFrame(
        {"date": pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")}
    )

    result = strategy.feature_engineering_standard(
        frame, metadata={"pair": "BTC/USDT:USDT"}
    )

    assert "%-g12_block__first" in result
    assert "%-g12_block__second" in result
    assert "%-g12_block__unselected" not in result


def test_generation12_strategy_rejects_duplicate_feature_columns() -> None:
    strategy = object.__new__(
        MarketReactionZoneG12ConfigurableFreqAIResearchStrategy
    )
    strategy.config = {
        "market_reaction_zone_g12": {
            "feature_columns": ["block__same", "block__same"]
        }
    }

    try:
        strategy._configured_feature_columns()
    except ValueError as exc:
        assert "must be unique" in str(exc)
    else:
        raise AssertionError("Duplicate feature columns should fail closed")

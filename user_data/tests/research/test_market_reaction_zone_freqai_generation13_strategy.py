# ruff: noqa: S101

from __future__ import annotations

import pandas as pd
import pytest

from user_data.strategies.MarketReactionZoneFreqAIGeneration13Strategy import (
    MarketReactionZoneG13FreqAIBaseStrategy,
)


def test_generation13_strategy_selects_exact_configured_columns(monkeypatch) -> None:
    strategy = object.__new__(MarketReactionZoneG13FreqAIBaseStrategy)
    strategy.config = {
        "market_reaction_zone_g13": {
            "feature_columns": ["base__one", "mtf__two"],
            "target_columns": ["&-g11_reaction_h1"],
            "maximum_target_horizon_hours": 8,
        }
    }
    cache = pd.DataFrame(
        {"base__one": [1.0], "mtf__two": [2.0], "not_selected": [3.0]}
    )
    monkeypatch.setattr(strategy, "_aligned_feature_cache", lambda *_args, **_kw: cache)

    result = strategy.feature_engineering_standard(
        pd.DataFrame({"date": pd.to_datetime(["2026-01-01T00:00:00Z"])}),
        {"pair": "BTC/USDT:USDT"},
    )

    assert "%-g13_base__one" in result
    assert "%-g13_mtf__two" in result
    assert "not_selected" not in result


def test_generation13_strategy_accepts_long_targets_and_rejects_bad_horizon() -> None:
    strategy = object.__new__(MarketReactionZoneG13FreqAIBaseStrategy)
    strategy.config = {
        "market_reaction_zone_g13": {
            "feature_columns": ["x"],
            "target_columns": ["&-g13_reaction_h48", "&-g13_volume_ratio_h48"],
            "maximum_target_horizon_hours": 48,
        }
    }

    assert strategy._configured_target_columns() == (
        "&-g13_reaction_h48",
        "&-g13_volume_ratio_h48",
    )
    assert strategy._maximum_target_horizon_hours() == 48
    strategy.config["market_reaction_zone_g13"]["maximum_target_horizon_hours"] = 24
    with pytest.raises(ValueError, match="8 or 48"):
        strategy._maximum_target_horizon_hours()

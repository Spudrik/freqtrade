# ruff: noqa: S101

from __future__ import annotations

from user_data.strategies.MarketReactionZoneFreqAIGeneration14Strategy import (
    MarketReactionZoneG14ConfigurableFreqAIResearchStrategy,
)


def test_generation14_strategy_reads_generation14_config() -> None:
    strategy = object.__new__(MarketReactionZoneG14ConfigurableFreqAIResearchStrategy)
    strategy.config = {
        "market_reaction_zone_g14": {
            "feature_columns": ["feature_a"],
            "target_columns": ["&-g11_reaction_h8"],
            "maximum_target_horizon_hours": 8,
        }
    }

    assert strategy._configured_feature_columns() == ("feature_a",)
    assert strategy._configured_target_columns() == ("&-g11_reaction_h8",)
    assert strategy._maximum_target_horizon_hours() == 8

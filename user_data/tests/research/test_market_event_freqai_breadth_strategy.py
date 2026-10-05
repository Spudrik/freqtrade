# ruff: noqa: S101

from __future__ import annotations

import pytest

from user_data.strategies.MarketEventFreqAIBreadthStrategy import (
    TARGET_COLUMNS,
    MarketEventFreqAIBreadthResearchStrategy,
)


def strategy_with_config(config: dict) -> MarketEventFreqAIBreadthResearchStrategy:
    strategy = MarketEventFreqAIBreadthResearchStrategy.__new__(
        MarketEventFreqAIBreadthResearchStrategy
    )
    strategy.config = config
    return strategy


def test_strategy_accepts_only_the_frozen_targets_and_horizon() -> None:
    strategy = strategy_with_config(
        {
            "market_event_freqai_breadth": {
                "target_columns": list(TARGET_COLUMNS),
                "maximum_target_horizon_hours": 8,
            }
        }
    )

    assert strategy._configured_target_columns() == TARGET_COLUMNS
    assert strategy._maximum_target_horizon_hours() == 8


def test_strategy_rejects_an_unknown_target() -> None:
    strategy = strategy_with_config(
        {
            "market_event_freqai_breadth": {
                "target_columns": ["&-future_profit"],
                "maximum_target_horizon_hours": 8,
            }
        }
    )

    with pytest.raises(ValueError, match="Unknown event breadth targets"):
        strategy._configured_target_columns()

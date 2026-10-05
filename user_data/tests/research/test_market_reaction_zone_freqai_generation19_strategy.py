# ruff: noqa: S101

from __future__ import annotations

import pytest

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation19_freqai_cache as g19c,
)
from user_data.strategies.MarketReactionZoneFreqAIGeneration19Strategy import (
    TARGET_COLUMNS,
    MarketReactionZoneG19FreqAIBaseStrategy,
)


def test_generation19_strategy_uses_exact_frozen_targets() -> None:
    strategy = object.__new__(MarketReactionZoneG19FreqAIBaseStrategy)
    strategy.config = {
        "market_reaction_zone_g19": {
            "feature_columns": ["g11_local_trend_momentum__adx14"],
            "target_columns": list(TARGET_COLUMNS),
            "maximum_target_horizon_hours": 8,
        }
    }

    assert TARGET_COLUMNS == g19c.TARGETS
    assert strategy._configured_target_columns() == TARGET_COLUMNS
    assert strategy._configured_feature_columns() == (
        "g11_local_trend_momentum__adx14",
    )
    assert strategy._maximum_target_horizon_hours() == 8


def test_generation19_strategy_rejects_unknown_target_and_horizon() -> None:
    strategy = object.__new__(MarketReactionZoneG19FreqAIBaseStrategy)
    strategy.config = {
        "market_reaction_zone_g19": {
            "feature_columns": ["feature"],
            "target_columns": ["&-g19_signed_direction_h8"],
            "maximum_target_horizon_hours": 4,
        }
    }

    with pytest.raises(ValueError, match="Unknown Generation 19"):
        strategy._configured_target_columns()
    with pytest.raises(ValueError, match="must be 8 hours"):
        strategy._maximum_target_horizon_hours()

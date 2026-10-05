from __future__ import annotations

from typing import Any

from user_data.strategies.MarketReactionZoneFreqAIGeneration13Strategy import (
    MarketReactionZoneG13FreqAIBaseStrategy,
)


class MarketReactionZoneG14FreqAIBaseStrategy(MarketReactionZoneG13FreqAIBaseStrategy):
    """Generation 14 exact-feature combination research base."""

    def _research_config(self) -> dict[str, Any]:
        return dict(self.config.get("market_reaction_zone_g14", {}))


class MarketReactionZoneG14ConfigurableFreqAIResearchStrategy(
    MarketReactionZoneG14FreqAIBaseStrategy
):
    """One strategy configured with one frozen Generation 14 profile."""

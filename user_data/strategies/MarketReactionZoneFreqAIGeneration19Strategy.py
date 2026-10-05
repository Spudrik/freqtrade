from __future__ import annotations

from typing import Any

from user_data.strategies.MarketReactionZoneFreqAIGeneration13Strategy import (
    MarketReactionZoneG13FreqAIBaseStrategy,
)


TARGET_COLUMNS = tuple(
    column
    for horizon in (1, 2, 4, 8)
    for column in (
        f"&-g19_future_volume_ratio_h{horizon}",
        f"&-g19_crossing_count_h{horizon}",
    )
)


class MarketReactionZoneG19FreqAIBaseStrategy(MarketReactionZoneG13FreqAIBaseStrategy):
    """Generation 19 direction-neutral level/context combination research base."""

    def _research_config(self) -> dict[str, Any]:
        return dict(self.config.get("market_reaction_zone_g19", {}))

    def _configured_target_columns(self) -> tuple[str, ...]:
        configured = self._research_config().get("target_columns")
        if not isinstance(configured, list) or not configured:
            raise ValueError("market_reaction_zone_g19.target_columns must be non-empty.")
        targets = tuple(str(column) for column in configured)
        unknown = sorted(set(targets).difference(TARGET_COLUMNS))
        if unknown:
            raise ValueError(f"Unknown Generation 19 reaction targets: {unknown}")
        return targets

    def _maximum_target_horizon_hours(self) -> int:
        value = int(self._research_config().get("maximum_target_horizon_hours", 0))
        if value != 8:
            raise ValueError("Generation 19 maximum target horizon must be 8 hours.")
        return value


class MarketReactionZoneG19ConfigurableFreqAIResearchStrategy(
    MarketReactionZoneG19FreqAIBaseStrategy
):
    """One strategy configured with one frozen Generation 19 profile."""

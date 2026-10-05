from __future__ import annotations

from typing import Any

from user_data.strategies.MarketReactionZoneFreqAIGeneration13Strategy import (
    MarketReactionZoneG13FreqAIBaseStrategy,
)


TARGET_COLUMNS = tuple(
    column
    for horizon in (1, 2, 4)
    for column in (
        f"&-g16_volume_ratio_h{horizon}",
        f"&-g16_range_ratio_h{horizon}",
        f"&-g16_crossings_h{horizon}",
        f"&-g16_reaction_h{horizon}",
    )
)


class MarketReactionZoneG16FreqAIBaseStrategy(MarketReactionZoneG13FreqAIBaseStrategy):
    """Generation 16 completed-contact and context attribution research base."""

    def _research_config(self) -> dict[str, Any]:
        return dict(self.config.get("market_reaction_zone_g16", {}))

    def _configured_target_columns(self) -> tuple[str, ...]:
        configured = self._research_config().get("target_columns")
        if not isinstance(configured, list) or not configured:
            raise ValueError("market_reaction_zone_g16.target_columns must be a non-empty list.")
        targets = tuple(str(column) for column in configured)
        unknown = sorted(set(targets).difference(TARGET_COLUMNS))
        if unknown:
            raise ValueError(f"Unknown Generation 16 reaction targets: {unknown}")
        return targets

    def _maximum_target_horizon_hours(self) -> int:
        value = int(self._research_config().get("maximum_target_horizon_hours", 0))
        if value != 4:
            raise ValueError("Generation 16 maximum target horizon must be 4 hours.")
        return value


class MarketReactionZoneG16ConfigurableFreqAIResearchStrategy(
    MarketReactionZoneG16FreqAIBaseStrategy
):
    """One strategy configured with one frozen Generation 16 profile."""

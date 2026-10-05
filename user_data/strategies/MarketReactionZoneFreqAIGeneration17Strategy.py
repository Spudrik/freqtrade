from __future__ import annotations

from typing import Any

from user_data.strategies.MarketReactionZoneFreqAIGeneration13Strategy import (
    MarketReactionZoneG13FreqAIBaseStrategy,
)


_CROSSING_TARGETS = tuple(
    column
    for horizon in (1, 2, 4)
    for column in (
        f"&-g17_crossing_count_h{horizon}",
        f"&-g17_repeated_recross_h{horizon}",
        f"&-g17_two_sided_traversal_h{horizon}",
    )
)
TARGET_COLUMNS = (
    *_CROSSING_TARGETS,
    *(f"&-g17_reaction_h{horizon}" for horizon in (1, 2, 4)),
)


class MarketReactionZoneG17FreqAIBaseStrategy(MarketReactionZoneG13FreqAIBaseStrategy):
    """Generation 17 direction-neutral decomposition research base."""

    def _research_config(self) -> dict[str, Any]:
        return dict(self.config.get("market_reaction_zone_g17", {}))

    def _configured_target_columns(self) -> tuple[str, ...]:
        configured = self._research_config().get("target_columns")
        if not isinstance(configured, list) or not configured:
            raise ValueError("market_reaction_zone_g17.target_columns must be non-empty.")
        targets = tuple(str(column) for column in configured)
        unknown = sorted(set(targets).difference(TARGET_COLUMNS))
        if unknown:
            raise ValueError(f"Unknown Generation 17 reaction targets: {unknown}")
        return targets

    def _maximum_target_horizon_hours(self) -> int:
        value = int(self._research_config().get("maximum_target_horizon_hours", 0))
        if value != 4:
            raise ValueError("Generation 17 maximum target horizon must be 4 hours.")
        return value


class MarketReactionZoneG17ConfigurableFreqAIResearchStrategy(
    MarketReactionZoneG17FreqAIBaseStrategy
):
    """One strategy configured with one frozen Generation 17 profile."""

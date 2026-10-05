from __future__ import annotations

from typing import Any

from user_data.strategies.MarketReactionZoneFreqAIGeneration23Strategy import (
    MarketReactionZoneG23FreqAIBaseStrategy,
)


TARGET_COLUMNS = tuple(
    f"&-g24_percentile_{metric}_h{horizon}"
    for metric in ("future_volume_ratio", "future_range_ratio")
    for horizon in (1, 2, 4, 8)
)


class MarketReactionZoneG24FreqAIBaseStrategy(MarketReactionZoneG23FreqAIBaseStrategy):
    """Generation 24 multi-model robustness research base."""

    def _research_config(self) -> dict[str, Any]:
        return dict(self.config.get("market_reaction_zone_g24", {}))

    def _configured_target_columns(self) -> tuple[str, ...]:
        configured = self._research_config().get("target_columns")
        if not isinstance(configured, list) or not configured:
            raise ValueError("market_reaction_zone_g24.target_columns must be non-empty.")
        targets = tuple(str(column) for column in configured)
        unknown = sorted(set(targets).difference(TARGET_COLUMNS))
        if unknown:
            raise ValueError(f"Unknown Generation 24 activity targets: {unknown}")
        return targets

    def _maximum_target_horizon_hours(self) -> int:
        value = int(self._research_config().get("maximum_target_horizon_hours", 0))
        if value != 8:
            raise ValueError("Generation 24 maximum target horizon must be 8 hours.")
        return value


class MarketReactionZoneG24ConfigurableFreqAIResearchStrategy(
    MarketReactionZoneG24FreqAIBaseStrategy
):
    """One strategy configured with one frozen Generation 24 profile."""

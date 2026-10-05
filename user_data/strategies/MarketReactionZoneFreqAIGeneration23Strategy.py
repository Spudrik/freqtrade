from __future__ import annotations

from typing import Any

from user_data.strategies.MarketReactionZoneFreqAIGeneration22Strategy import (
    MarketReactionZoneG22FreqAIBaseStrategy,
)


TARGET_COLUMNS = tuple(
    f"&-g23_{transform}_{metric}_h{horizon}"
    for transform in ("residual", "percentile")
    for metric in (
        "maximum_absolute_excursion_atr",
        "future_volume_ratio",
        "future_range_ratio",
        "crossing_count",
        "dwell_fraction",
    )
    for horizon in (1, 2, 4, 8)
)


class MarketReactionZoneG23FreqAIBaseStrategy(MarketReactionZoneG22FreqAIBaseStrategy):
    """Pair-calibrated, direction-neutral Generation 23 FreqAI research base."""

    def _research_config(self) -> dict[str, Any]:
        return dict(self.config.get("market_reaction_zone_g23", {}))

    def _configured_target_columns(self) -> tuple[str, ...]:
        configured = self._research_config().get("target_columns")
        if not isinstance(configured, list) or not configured:
            raise ValueError("market_reaction_zone_g23.target_columns must be non-empty.")
        targets = tuple(str(column) for column in configured)
        unknown = sorted(set(targets).difference(TARGET_COLUMNS))
        if unknown:
            raise ValueError(f"Unknown Generation 23 reaction targets: {unknown}")
        return targets

    def _maximum_target_horizon_hours(self) -> int:
        value = int(self._research_config().get("maximum_target_horizon_hours", 0))
        if value != 8:
            raise ValueError("Generation 23 maximum target horizon must be 8 hours.")
        return value


class MarketReactionZoneG23ConfigurableFreqAIResearchStrategy(
    MarketReactionZoneG23FreqAIBaseStrategy
):
    """One strategy configured with one frozen Generation 23 profile."""

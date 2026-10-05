from __future__ import annotations

from typing import Any

from user_data.strategies.MarketReactionZoneFreqAIGeneration22Strategy import (
    MarketReactionZoneG22FreqAIBaseStrategy,
)


TARGET_COLUMNS = tuple(
    f"&-meb_{metric}_h{horizon}"
    for metric in (
        "log_range_atr",
        "log_volume_ratio",
        "close_return_atr",
        "excursion_balance_atr",
    )
    for horizon in (1, 4, 8)
)


class MarketEventFreqAIBreadthBaseStrategy(MarketReactionZoneG22FreqAIBaseStrategy):
    """Frozen event/market/level breadth comparison through actual FreqAI."""

    def _research_config(self) -> dict[str, Any]:
        return dict(self.config.get("market_event_freqai_breadth", {}))

    def _configured_target_columns(self) -> tuple[str, ...]:
        configured = self._research_config().get("target_columns")
        if not isinstance(configured, list) or not configured:
            raise ValueError("market_event_freqai_breadth.target_columns must be non-empty.")
        targets = tuple(str(column) for column in configured)
        unknown = sorted(set(targets).difference(TARGET_COLUMNS))
        if unknown:
            raise ValueError(f"Unknown event breadth targets: {unknown}")
        return targets

    def _maximum_target_horizon_hours(self) -> int:
        value = int(self._research_config().get("maximum_target_horizon_hours", 0))
        if value != 8:
            raise ValueError("Event breadth maximum target horizon must be 8 hours.")
        return value


class MarketEventFreqAIBreadthResearchStrategy(MarketEventFreqAIBreadthBaseStrategy):
    """One strategy configured with one frozen breadth profile."""

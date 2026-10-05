from __future__ import annotations

from typing import Any

from pandas import DataFrame

from user_data.strategies.MarketReactionZoneFreqAIGeneration13Strategy import (
    MarketReactionZoneG13FreqAIBaseStrategy,
)


TARGET_COLUMNS = tuple(
    f"&-g22_{metric}_h{horizon}"
    for metric in (
        "maximum_absolute_excursion_atr",
        "future_volume_ratio",
        "future_range_ratio",
        "crossing_count",
        "dwell_fraction",
    )
    for horizon in (1, 2, 4, 8)
)


class MarketReactionZoneG22FreqAIBaseStrategy(MarketReactionZoneG13FreqAIBaseStrategy):
    """Low-dimensional, direction-neutral Generation 22 FreqAI research base."""

    def _research_config(self) -> dict[str, Any]:
        return dict(self.config.get("market_reaction_zone_g22", {}))

    def _configured_target_columns(self) -> tuple[str, ...]:
        configured = self._research_config().get("target_columns")
        if not isinstance(configured, list) or not configured:
            raise ValueError("market_reaction_zone_g22.target_columns must be non-empty.")
        targets = tuple(str(column) for column in configured)
        unknown = sorted(set(targets).difference(TARGET_COLUMNS))
        if unknown:
            raise ValueError(f"Unknown Generation 22 reaction targets: {unknown}")
        return targets

    def _maximum_target_horizon_hours(self) -> int:
        value = int(self._research_config().get("maximum_target_horizon_hours", 0))
        if value != 8:
            raise ValueError("Generation 22 maximum target horizon must be 8 hours.")
        return value

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs: Any
    ) -> DataFrame:
        """Use only the frozen cache columns; do not add implicit TA or time features."""
        _ = metadata, kwargs
        return dataframe


class MarketReactionZoneG22ConfigurableFreqAIResearchStrategy(
    MarketReactionZoneG22FreqAIBaseStrategy
):
    """One strategy configured with one frozen Generation 22 profile."""

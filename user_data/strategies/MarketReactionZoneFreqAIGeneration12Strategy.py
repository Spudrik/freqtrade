from __future__ import annotations

from typing import Any

import pandas as pd
from pandas import DataFrame

from user_data.strategies.MarketReactionZoneFreqAIGeneration11Strategy import (
    MarketReactionZoneG11FreqAIBaseStrategy,
)


class MarketReactionZoneG12FreqAIBaseStrategy(
    MarketReactionZoneG11FreqAIBaseStrategy
):
    """Generation 12 exact-subfamily and bounded-combination research base."""

    def _research_config(self) -> dict[str, Any]:
        return dict(self.config.get("market_reaction_zone_g12", {}))

    def _configured_feature_columns(self) -> tuple[str, ...]:
        configured = self._research_config().get("feature_columns")
        if not isinstance(configured, list) or not configured:
            raise ValueError(
                "market_reaction_zone_g12.feature_columns must be a non-empty list."
            )
        columns = tuple(str(column) for column in configured)
        if len(columns) != len(set(columns)):
            raise ValueError("Generation 12 feature columns must be unique.")
        return columns

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs: Any
    ) -> DataFrame:
        _ = kwargs
        pair = str(metadata.get("pair") or "")
        cache = self._aligned_feature_cache(dataframe, pair=pair)
        columns = self._configured_feature_columns()
        missing = sorted(set(columns).difference(cache.columns))
        if missing:
            raise ValueError(
                f"Generation 12 feature cache lacks columns for {pair}: {missing}"
            )
        selected = cache[list(columns)].copy()
        selected.columns = [f"%-g12_{column}" for column in columns]
        selected.index = dataframe.index
        return pd.concat([dataframe, selected], axis=1)


class MarketReactionZoneG12ConfigurableFreqAIResearchStrategy(
    MarketReactionZoneG12FreqAIBaseStrategy
):
    """One strategy configured with one frozen Generation 12 feature profile."""

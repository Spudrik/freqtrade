from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from user_data.strategies.MarketReactionZoneFreqAIGeneration4Strategy import (
    utc_nanoseconds,
)
from user_data.strategies.MarketReactionZoneFreqAIGeneration6Strategy import (
    MAX_TARGET_HORIZON_HOURS,
    TARGET_COLUMNS,
    MarketReactionZoneG6FreqAIBaseStrategy,
)


class MarketReactionZoneG7FreqAIBaseStrategy(
    MarketReactionZoneG6FreqAIBaseStrategy
):
    """Generation 7 direction-neutral pairwise interaction research base."""

    def _research_config(self) -> dict[str, Any]:
        return dict(self.config.get("market_reaction_zone_g7", {}))

    def _configured_target_columns(self) -> tuple[str, ...]:
        configured = self._research_config().get("target_columns")
        if not isinstance(configured, list) or not configured:
            raise ValueError(
                "market_reaction_zone_g7.target_columns must be a non-empty list."
            )
        targets = tuple(str(column) for column in configured)
        unknown = sorted(set(targets).difference(TARGET_COLUMNS))
        if unknown:
            raise ValueError(f"Unknown Generation 7 reaction targets: {unknown}")
        return targets

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict, **kwargs: Any
    ) -> DataFrame:
        _ = kwargs
        pair = str(metadata.get("pair") or "")
        events = self._event_cache(pair)
        dates = utc_nanoseconds(dataframe["date"])
        left = DataFrame(
            {"date": dates, "_row": np.arange(len(dataframe), dtype=np.int64)}
        )
        labels = left.merge(events, on="date", how="left", validate="many_to_one")
        labels.sort_values("_row", inplace=True)
        eligible = np.ones(len(labels), dtype=bool)
        for block in self._configured_ready_blocks():
            column = f"ready__{block}"
            if column not in labels:
                raise ValueError(
                    f"Generation 7 target cache lacks readiness column {column!r}."
                )
            eligible &= labels[column].fillna(False).astype(bool).to_numpy()
        maximum_date = dates.max()
        purge_cutoff = maximum_date - pd.Timedelta(hours=MAX_TARGET_HORIZON_HOURS)
        purged = dates.gt(purge_cutoff).to_numpy()
        for column in self._configured_target_columns():
            values = pd.to_numeric(labels[column], errors="coerce").to_numpy(
                dtype=float,
                copy=True,
            )
            values[~eligible | purged] = np.nan
            dataframe[column] = values
        return dataframe


class MarketReactionZoneG7ConfigurableFreqAIResearchStrategy(
    MarketReactionZoneG7FreqAIBaseStrategy
):
    """One strategy configured with a frozen Generation 7 interaction profile."""

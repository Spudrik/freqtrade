from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from user_data.strategies.MarketReactionZoneFreqAIGeneration4Strategy import (
    pair_file_stem,
    utc_nanoseconds,
)
from user_data.strategies.MarketReactionZoneFreqAIGeneration6Strategy import (
    MarketReactionZoneG6FreqAIBaseStrategy,
)


TARGET_COLUMNS = tuple(
    [
        *(f"&-g11_reaction_h{horizon}" for horizon in (1, 2, 4, 8)),
        *(f"&-g11_volume_ratio_h{horizon}" for horizon in (1, 2, 4, 8)),
    ]
)
MAX_TARGET_HORIZON_HOURS = 8


class MarketReactionZoneG11FreqAIBaseStrategy(
    MarketReactionZoneG6FreqAIBaseStrategy
):
    """Generation 11 multi-horizon, direction-neutral attribution research base."""

    def _research_config(self) -> dict[str, Any]:
        return dict(self.config.get("market_reaction_zone_g11", {}))

    def _event_cache(self, pair: str) -> DataFrame:
        path = self._event_cache_dir() / f"{pair_file_stem(pair)}.parquet"
        key = str(path)
        cached = self._event_cache_frames.get(key)
        if cached is not None:
            return cached
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        required = {"date", *TARGET_COLUMNS}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"Generation 11 event cache is missing columns {sorted(missing)}: {path}"
            )
        frame["date"] = utc_nanoseconds(frame["date"])
        if frame["date"].duplicated().any():
            raise ValueError(f"Duplicate Generation 11 event-cache dates: {path}")
        frame = frame.sort_values("date").reset_index(drop=True)
        self._event_cache_frames[key] = frame
        return frame

    def _configured_target_columns(self) -> tuple[str, ...]:
        configured = self._research_config().get("target_columns")
        if not isinstance(configured, list) or not configured:
            raise ValueError(
                "market_reaction_zone_g11.target_columns must be a non-empty list."
            )
        targets = tuple(str(column) for column in configured)
        unknown = sorted(set(targets).difference(TARGET_COLUMNS))
        if unknown:
            raise ValueError(f"Unknown Generation 11 reaction targets: {unknown}")
        return targets

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict, **kwargs: Any
    ) -> DataFrame:
        _ = kwargs
        pair = str(metadata.get("pair") or "")
        events = self._event_cache(pair)
        dates = utc_nanoseconds(dataframe["date"])
        labels = DataFrame(
            {"date": dates, "_row": np.arange(len(dataframe), dtype=np.int64)}
        ).merge(events, on="date", how="left", validate="many_to_one")
        labels.sort_values("_row", inplace=True)
        eligible = np.ones(len(labels), dtype=bool)
        for block in self._configured_ready_blocks():
            column = f"ready__{block}"
            if column not in labels:
                raise ValueError(
                    f"Generation 11 target cache lacks readiness column {column!r}."
                )
            eligible &= labels[column].fillna(False).astype(bool).to_numpy()
        maximum_date = dates.max()
        purge_cutoff = maximum_date - pd.Timedelta(
            hours=MAX_TARGET_HORIZON_HOURS
        )
        purged = dates.gt(purge_cutoff).to_numpy()
        for column in self._configured_target_columns():
            values = pd.to_numeric(labels[column], errors="coerce").to_numpy(
                dtype=float,
                copy=True,
            )
            values[~eligible | purged] = np.nan
            dataframe[column] = values
        return dataframe


class MarketReactionZoneG11ConfigurableFreqAIResearchStrategy(
    MarketReactionZoneG11FreqAIBaseStrategy
):
    """One strategy configured with one frozen Generation 11 information profile."""

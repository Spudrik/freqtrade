from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from user_data.strategies.MarketReactionZoneFreqAIGeneration4Strategy import (
    pair_file_stem,
    utc_nanoseconds,
)
from user_data.strategies.MarketReactionZoneFreqAIGeneration11Strategy import (
    MarketReactionZoneG11FreqAIBaseStrategy,
)


G11_TARGETS = tuple(
    [
        *(f"&-g11_reaction_h{horizon}" for horizon in (1, 2, 4, 8)),
        *(f"&-g11_volume_ratio_h{horizon}" for horizon in (1, 2, 4, 8)),
    ]
)
G13_LONG_TARGETS = tuple(
    [
        *(f"&-g13_reaction_h{horizon}" for horizon in (12, 24, 48)),
        *(f"&-g13_volume_ratio_h{horizon}" for horizon in (12, 24, 48)),
    ]
)
TARGET_COLUMNS = (*G11_TARGETS, *G13_LONG_TARGETS)


class MarketReactionZoneG13FreqAIBaseStrategy(
    MarketReactionZoneG11FreqAIBaseStrategy
):
    """Generation 13 exact-feature, variable-horizon reaction research base."""

    def _research_config(self) -> dict[str, Any]:
        return dict(self.config.get("market_reaction_zone_g13", {}))

    def _configured_feature_columns(self) -> tuple[str, ...]:
        configured = self._research_config().get("feature_columns")
        if not isinstance(configured, list) or not configured:
            raise ValueError(
                "market_reaction_zone_g13.feature_columns must be a non-empty list."
            )
        columns = tuple(str(column) for column in configured)
        if len(columns) != len(set(columns)):
            raise ValueError("Generation 13 feature columns must be unique.")
        return columns

    def _configured_target_columns(self) -> tuple[str, ...]:
        configured = self._research_config().get("target_columns")
        if not isinstance(configured, list) or not configured:
            raise ValueError(
                "market_reaction_zone_g13.target_columns must be a non-empty list."
            )
        targets = tuple(str(column) for column in configured)
        unknown = sorted(set(targets).difference(TARGET_COLUMNS))
        if unknown:
            raise ValueError(f"Unknown Generation 13 reaction targets: {unknown}")
        return targets

    def _maximum_target_horizon_hours(self) -> int:
        value = int(self._research_config().get("maximum_target_horizon_hours", 0))
        if value not in {8, 48}:
            raise ValueError("Generation 13 maximum target horizon must be 8 or 48 hours.")
        return value

    def _event_cache(self, pair: str) -> DataFrame:
        path = self._event_cache_dir() / f"{pair_file_stem(pair)}.parquet"
        key = str(path)
        cached = self._event_cache_frames.get(key)
        if cached is not None:
            return cached
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        required = {"date", *self._configured_target_columns()}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"Generation 13 event cache is missing columns {sorted(missing)}: {path}"
            )
        frame["date"] = utc_nanoseconds(frame["date"])
        if frame["date"].duplicated().any():
            raise ValueError(f"Duplicate Generation 13 event-cache dates: {path}")
        frame = frame.sort_values("date").reset_index(drop=True)
        self._event_cache_frames[key] = frame
        return frame

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
                f"Generation 13 feature cache lacks columns for {pair}: {missing}"
            )
        selected = cache[list(columns)].copy()
        selected.columns = [f"%-g13_{column}" for column in columns]
        selected.index = dataframe.index
        return pd.concat([dataframe, selected], axis=1)

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
                    f"Generation 13 target cache lacks readiness column {column!r}."
                )
            eligible &= labels[column].fillna(False).astype(bool).to_numpy()
        purge_cutoff = dates.max() - pd.Timedelta(
            hours=self._maximum_target_horizon_hours()
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


class MarketReactionZoneG13ConfigurableFreqAIResearchStrategy(
    MarketReactionZoneG13FreqAIBaseStrategy
):
    """One strategy configured with one frozen Generation 13 profile."""

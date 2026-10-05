from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from pandas import DataFrame

from user_data.strategies.MarketReactionZoneFreqAIGeneration4Strategy import (
    MarketReactionZoneG4DFreqAIBaseStrategy,
    pair_file_stem,
    utc_nanoseconds,
)


USER_DATA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_CACHE_DIR = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "market_reaction_zones"
    / "generation6_branches"
    / "g6_freqai_source_ladders"
    / "feature_cache"
)
DEFAULT_EVENT_CACHE_DIR = DEFAULT_FEATURE_CACHE_DIR.parent / "event_cache"

TARGET_COLUMNS = (
    "&-g6_future_volume_ratio_h1",
    "&-g6_future_range_ratio_h1",
    "&-g6_absolute_excursion_atr_h4",
    "&-g6_absolute_pressure_change_h1",
    "&-g6_dwell_fraction_h4",
)
MAX_TARGET_HORIZON_HOURS = 4


class MarketReactionZoneG6FreqAIBaseStrategy(
    MarketReactionZoneG4DFreqAIBaseStrategy
):
    """Generation 6 non-directional source-versus-level FreqAI research base."""

    startup_candle_count = 2200
    include_local_state = False
    required_ready_blocks: tuple[str, ...] = ()

    _feature_cache_frames: ClassVar[dict[str, DataFrame]] = {}
    _event_cache_frames: ClassVar[dict[str, DataFrame]] = {}

    def _research_config(self) -> dict[str, Any]:
        return dict(self.config.get("market_reaction_zone_g6", {}))

    def _feature_cache_dir(self) -> Path:
        configured = self._research_config().get("feature_cache_dir")
        path = Path(configured) if configured else DEFAULT_FEATURE_CACHE_DIR
        if not path.is_absolute():
            path = USER_DATA_DIR / path
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path.resolve()

    def _event_cache_dir(self) -> Path:
        configured = self._research_config().get("event_cache_dir")
        path = Path(configured) if configured else DEFAULT_EVENT_CACHE_DIR
        if not path.is_absolute():
            path = USER_DATA_DIR / path
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path.resolve()

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
            raise ValueError(f"G6 event cache is missing columns {sorted(missing)}: {path}")
        frame["date"] = utc_nanoseconds(frame["date"])
        if frame["date"].duplicated().any():
            raise ValueError(f"Duplicate G6 event-cache dates: {path}")
        frame = frame.sort_values("date").reset_index(drop=True)
        self._event_cache_frames[key] = frame
        return frame

    def _configured_feature_blocks(self) -> tuple[str, ...]:
        configured = self._research_config().get("feature_blocks")
        if not isinstance(configured, list) or not configured:
            raise ValueError("market_reaction_zone_g6.feature_blocks must be a non-empty list.")
        return tuple(str(block) for block in configured)

    def _configured_ready_blocks(self) -> tuple[str, ...]:
        configured = self._research_config().get("required_ready_blocks")
        if configured is None:
            return self._configured_feature_blocks()
        if not isinstance(configured, list) or not configured:
            raise ValueError(
                "market_reaction_zone_g6.required_ready_blocks must be a non-empty list."
            )
        return tuple(str(block) for block in configured)

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs: Any
    ) -> DataFrame:
        _ = kwargs
        blocks = self._configured_feature_blocks()
        pair = str(metadata.get("pair") or "")
        cache = self._aligned_feature_cache(dataframe, pair=pair)
        prefixes = tuple(f"{block}__" for block in blocks)
        selected = [column for column in cache if column.startswith(prefixes)]
        missing = [
            block
            for block in blocks
            if not any(column.startswith(f"{block}__") for column in selected)
        ]
        if missing:
            raise ValueError(f"No G6 feature-cache columns for blocks {missing!r} and {pair}.")
        block = cache[selected].copy()
        block.columns = [f"%-g6_{column}" for column in selected]
        block.index = dataframe.index
        return pd.concat([dataframe, block], axis=1)

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
        labels = left.merge(events, on="date", how="left", validate="many_to_one").sort_values(
            "_row"
        )
        eligible = np.ones(len(labels), dtype=bool)
        for block in self._configured_ready_blocks():
            column = f"ready__{block}"
            if column not in labels:
                raise ValueError(f"G6 target cache lacks required readiness column {column!r}.")
            eligible &= labels[column].fillna(False).astype(bool).to_numpy()
        maximum_date = dates.max()
        purge_cutoff = maximum_date - pd.Timedelta(hours=MAX_TARGET_HORIZON_HOURS)
        purged = dates.gt(purge_cutoff).to_numpy()
        for column in TARGET_COLUMNS:
            values = pd.to_numeric(labels[column], errors="coerce").to_numpy(
                dtype=float,
                copy=True,
            )
            values[~eligible | purged] = np.nan
            dataframe[column] = values
        return dataframe


class MarketReactionZoneG6ConfigurableFreqAIResearchStrategy(
    MarketReactionZoneG6FreqAIBaseStrategy
):
    """One audited strategy class configured with one frozen feature-block profile."""

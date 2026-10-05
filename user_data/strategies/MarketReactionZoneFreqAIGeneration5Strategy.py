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
    / "generation5_branches"
    / "g5b_freqai_volume_clock_and_feature_ablation"
    / "feature_cache"
)
DEFAULT_EVENT_CACHE_DIR = DEFAULT_FEATURE_CACHE_DIR.parent / "event_cache"

TARGET_COLUMNS = (
    "&-g5b_contact_volume_ratio",
    "&-g5b_next_candle_volume_ratio",
)
MAX_TARGET_HORIZON_HOURS = 1


class MarketReactionZoneG5BFreqAIBaseStrategy(
    MarketReactionZoneG4DFreqAIBaseStrategy
):
    """Generation 5B volume-clock research with explicit cache-block availability."""

    startup_candle_count = 2200
    _feature_cache_frames: ClassVar[dict[str, DataFrame]] = {}
    _event_cache_frames: ClassVar[dict[str, DataFrame]] = {}

    def _research_config(self) -> dict[str, Any]:
        return dict(self.config.get("market_reaction_zone_g5b", {}))

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
        frame = pd.read_parquet(path, columns=["date", *TARGET_COLUMNS])
        frame["date"] = utc_nanoseconds(frame["date"])
        if frame["date"].duplicated().any():
            raise ValueError(f"Duplicate G5B event-cache dates: {path}")
        frame = frame.sort_values("date").reset_index(drop=True)
        self._event_cache_frames[key] = frame
        return frame

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict, **kwargs: Any
    ) -> DataFrame:
        _ = kwargs
        pair = str(metadata.get("pair") or "")
        events = self._event_cache(pair)
        dates = utc_nanoseconds(dataframe["date"])
        left = DataFrame({"date": dates, "_row": np.arange(len(dataframe), dtype=np.int64)})
        labels = left.merge(events, on="date", how="left", validate="many_to_one").sort_values(
            "_row"
        )
        maximum_date = dates.max()
        purge_cutoff = maximum_date - pd.Timedelta(hours=MAX_TARGET_HORIZON_HOURS)
        purged = dates.gt(purge_cutoff).to_numpy()
        for column in TARGET_COLUMNS:
            values = pd.to_numeric(labels[column], errors="coerce").to_numpy(
                dtype=float,
                copy=True,
            )
            values[purged] = np.nan
            dataframe[column] = values
        return dataframe


class MarketReactionZoneG5BStateOnlyFreqAIResearchStrategy(
    MarketReactionZoneG5BFreqAIBaseStrategy
):
    include_local_state = True
    cache_blocks = ("wide",)


class MarketReactionZoneG5BPreContactFreqAIResearchStrategy(
    MarketReactionZoneG5BFreqAIBaseStrategy
):
    include_local_state = True
    cache_blocks = ("wide", "pre_path", "pre_level", "pre_proximity")


class MarketReactionZoneG5BPreContactShuffledFreqAIResearchStrategy(
    MarketReactionZoneG5BFreqAIBaseStrategy
):
    include_local_state = True
    cache_blocks = (
        "wide",
        "shuffle_path",
        "shuffle_level",
        "shuffle_proximity",
    )


class MarketReactionZoneG5BPreContactStaleFreqAIResearchStrategy(
    MarketReactionZoneG5BFreqAIBaseStrategy
):
    include_local_state = True
    cache_blocks = ("wide", "stale_path", "stale_level", "stale_proximity")


class MarketReactionZoneG5BEventResolutionFreqAIResearchStrategy(
    MarketReactionZoneG5BPreContactFreqAIResearchStrategy
):
    cache_blocks = (
        "wide",
        "pre_path",
        "pre_level",
        "pre_proximity",
        "resolution",
    )


class MarketReactionZoneG5BContactCloseFreqAIResearchStrategy(
    MarketReactionZoneG5BEventResolutionFreqAIResearchStrategy
):
    cache_blocks = (
        "wide",
        "pre_path",
        "pre_level",
        "pre_proximity",
        "resolution",
        "contact",
    )


class MarketReactionZoneG5BNoPathFreqAIResearchStrategy(
    MarketReactionZoneG5BFreqAIBaseStrategy
):
    include_local_state = True
    cache_blocks = ("wide", "pre_level", "pre_proximity")


class MarketReactionZoneG5BNoLevelFreqAIResearchStrategy(
    MarketReactionZoneG5BFreqAIBaseStrategy
):
    include_local_state = True
    cache_blocks = ("wide", "pre_path", "pre_proximity")


class MarketReactionZoneG5BNoProximityFreqAIResearchStrategy(
    MarketReactionZoneG5BFreqAIBaseStrategy
):
    include_local_state = True
    cache_blocks = ("wide", "pre_path", "pre_level")


class MarketReactionZoneG5BCandidateOnlyFreqAIResearchStrategy(
    MarketReactionZoneG5BFreqAIBaseStrategy
):
    include_local_state = False
    cache_blocks = ("pre_path", "pre_level", "pre_proximity")

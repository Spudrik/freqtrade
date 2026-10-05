from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.strategy import IStrategy


USER_DATA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_CACHE_DIR = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "market_reaction_zones"
    / "generation4_branches"
    / "g4d_freqai_reaction_ablation"
    / "feature_cache"
)
DEFAULT_EVENT_CACHE_DIR = DEFAULT_FEATURE_CACHE_DIR.parent / "event_cache"

TARGET_COLUMNS = (
    "&-g4d_contact_volume_ratio",
    "&-g4d_one_hour_absolute_excursion_in_prior_atr",
    "&-g4d_four_hour_dwell_fraction",
)
MAX_TARGET_HORIZON_HOURS = 4


def pair_file_stem(pair: str) -> str:
    normalized = pair.strip().upper()
    if "/" not in normalized:
        raise ValueError(f"Unsupported market-reaction pair: {pair!r}")
    return normalized.replace("/", "_").replace(":", "_")


def numeric(frame: DataFrame, column: str, default: float = np.nan) -> Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def utc_nanoseconds(values: Series) -> Series:
    converted = pd.to_datetime(values, utc=True, errors="raise")
    return converted.astype("datetime64[ns, UTC]")


def atr(dataframe: DataFrame, period: int = 14) -> Series:
    high = numeric(dataframe, "high")
    low = numeric(dataframe, "low")
    close = numeric(dataframe, "close")
    previous_close = close.shift(1)
    true_range = pd.concat(
        (
            (high - low).abs(),
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ),
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period, min_periods=period).mean()


class MarketReactionZoneG4DFreqAIBaseStrategy(IStrategy):
    """Direction-neutral Generation 4D FreqAI regression research base."""

    INTERFACE_VERSION = 3
    timeframe = "1h"
    startup_candle_count = 2200
    process_only_new_candles = True
    can_short = True

    minimal_roi = {"0": 100.0}
    stoploss = -0.99
    use_exit_signal = False
    trailing_stop = False

    include_local_state = False
    cache_blocks: tuple[str, ...] = ()

    _feature_cache_frames: ClassVar[dict[str, DataFrame]] = {}
    _event_cache_frames: ClassVar[dict[str, DataFrame]] = {}

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs: Any
    ) -> DataFrame:
        _ = period, metadata, kwargs
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs: Any
    ) -> DataFrame:
        """Expose only completed-candle state known before the event candle opens."""
        _ = metadata, kwargs
        dataframe["%-g4d_unconditional_constant"] = 1.0
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="raise")
        dataframe["%-g4d_causal_calendar_time_years"] = (
            dates.astype("int64") / (365.25 * 24.0 * 3600.0 * 1e9)
        )
        if not self.include_local_state:
            return dataframe

        close = numeric(dataframe, "close").replace(0.0, np.nan)
        open_ = numeric(dataframe, "open")
        high = numeric(dataframe, "high")
        low = numeric(dataframe, "low")
        volume = numeric(dataframe, "volume").clip(lower=0.0)
        candle_range = (high - low).replace(0.0, np.nan)
        known_close = close.shift(1)
        known_range = candle_range.shift(1)
        known_volume = volume.shift(1)
        known_atr = atr(dataframe).shift(1).replace(0.0, np.nan)
        known_close_location = (
            ((close - low) / candle_range) * 2.0 - 1.0
        ).clip(-1.0, 1.0).shift(1)

        for horizon in (1, 4, 24):
            dataframe[f"%-g4d_local_return_{horizon}h"] = close.pct_change(horizon).shift(1)
        dataframe["%-g4d_local_atr_pct"] = known_atr / known_close.abs()
        dataframe["%-g4d_local_body_fraction"] = (
            ((close - open_) / candle_range).clip(-1.0, 1.0).shift(1)
        )
        dataframe["%-g4d_local_close_location"] = known_close_location
        dataframe["%-g4d_local_range_vs_prior24"] = (
            known_range
            / candle_range.shift(2).rolling(24, min_periods=12).median().replace(0.0, np.nan)
        ).clip(0.0, 30.0)
        dataframe["%-g4d_local_volume_vs_prior24"] = (
            known_volume
            / volume.shift(2).rolling(24, min_periods=12).median().replace(0.0, np.nan)
        ).clip(0.0, 30.0)
        dataframe["%-g4d_local_volume_z_24"] = (
            (known_volume - volume.shift(2).rolling(24, min_periods=12).mean())
            / volume.shift(2).rolling(24, min_periods=12).std(ddof=0).replace(0.0, np.nan)
        ).clip(-10.0, 10.0)

        signed_volume = known_close_location.fillna(0.0) * known_volume.fillna(0.0)
        for horizon in (6, 24):
            minimum = max(3, horizon // 2)
            dataframe[f"%-g4d_local_pressure_{horizon}h"] = signed_volume.rolling(
                horizon, min_periods=minimum
            ).sum() / known_volume.rolling(horizon, min_periods=minimum).sum().replace(
                0.0, np.nan
            )
        dataframe["%-g4d_local_pressure_change_6h_vs_24h"] = (
            dataframe["%-g4d_local_pressure_6h"]
            - dataframe["%-g4d_local_pressure_24h"]
        )
        self._append_fixed_context(dataframe, close, known_atr)
        return dataframe

    @staticmethod
    def _append_fixed_context(dataframe: DataFrame, close: Series, known_atr: Series) -> None:
        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(
            alpha=1.0 / 14.0, min_periods=14, adjust=False
        ).mean()
        loss = (-delta.clip(upper=0.0)).ewm(
            alpha=1.0 / 14.0, min_periods=14, adjust=False
        ).mean()
        relative_strength = gain / loss.replace(0.0, np.nan)
        dataframe["%-g4d_local_rsi_14"] = (
            100.0 - (100.0 / (1.0 + relative_strength))
        ).shift(1) / 100.0

        ema_12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, min_periods=9, adjust=False).mean()
        dataframe["%-g4d_local_macd_histogram_atr"] = (
            (macd - macd_signal).shift(1) / known_atr
        ).clip(-20.0, 20.0)

        average = close.ewm(span=50, min_periods=50, adjust=False).mean()
        dataframe["%-g4d_local_ema_50_distance_atr"] = (
            (close.shift(1) - average.shift(1)) / known_atr
        ).clip(-30.0, 30.0)

        rolling_mean = close.rolling(20, min_periods=20).mean()
        rolling_std = close.rolling(20, min_periods=20).std(ddof=0)
        dataframe["%-g4d_local_bollinger_position"] = (
            (close.shift(1) - rolling_mean.shift(1))
            / (2.0 * rolling_std.shift(1)).replace(0.0, np.nan)
        ).clip(-5.0, 5.0)
        dataframe["%-g4d_local_bollinger_width_atr"] = (
            4.0 * rolling_std.shift(1) / known_atr
        ).clip(0.0, 30.0)
        dataframe["%-g4d_local_realised_volatility_24"] = (
            close.pct_change().rolling(24, min_periods=12).std(ddof=0).shift(1)
        )

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs: Any
    ) -> DataFrame:
        _ = kwargs
        if not self.cache_blocks:
            return dataframe
        pair = str(metadata.get("pair") or "")
        cache = self._aligned_feature_cache(dataframe, pair=pair)
        prefixes = tuple(f"{block}__" for block in self.cache_blocks)
        selected = [column for column in cache if column.startswith(prefixes)]
        if not selected:
            raise ValueError(
                f"No G4D feature-cache columns for blocks {self.cache_blocks!r} and {pair}."
            )
        block = cache[selected].copy()
        block.columns = [f"%-g4d_{column}" for column in selected]
        block.index = dataframe.index
        return pd.concat([dataframe, block], axis=1)

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
            values = pd.to_numeric(labels[column], errors="coerce").to_numpy(dtype=float, copy=True)
            values[purged] = np.nan
            dataframe[column] = values
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self.freqai.start(dataframe, metadata, self)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe

    def _research_config(self) -> dict[str, Any]:
        return dict(self.config.get("market_reaction_zone_g4d", {}))

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

    def _feature_cache(self, pair: str) -> DataFrame:
        path = self._feature_cache_dir() / f"{pair_file_stem(pair)}.parquet"
        key = str(path)
        cached = self._feature_cache_frames.get(key)
        if cached is not None:
            return cached
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        frame["date"] = utc_nanoseconds(frame["date"])
        if frame["date"].duplicated().any():
            raise ValueError(f"Duplicate feature-cache dates: {path}")
        frame = frame.sort_values("date").reset_index(drop=True)
        self._feature_cache_frames[key] = frame
        return frame

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
            raise ValueError(f"Duplicate event-cache dates: {path}")
        frame = frame.sort_values("date").reset_index(drop=True)
        self._event_cache_frames[key] = frame
        return frame

    def _aligned_feature_cache(self, dataframe: DataFrame, *, pair: str) -> DataFrame:
        cache = self._feature_cache(pair)
        left = DataFrame(
            {
                "date": utc_nanoseconds(dataframe["date"]),
                "_row": np.arange(len(dataframe), dtype=np.int64),
            }
        )
        aligned = left.merge(cache, on="date", how="left", validate="many_to_one").sort_values(
            "_row"
        )
        output = aligned.drop(columns=["date", "_row"]).reset_index(drop=True)
        return output.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


class MarketReactionZoneG4DUnconditionalFreqAIResearchStrategy(
    MarketReactionZoneG4DFreqAIBaseStrategy
):
    include_local_state = False
    cache_blocks = ()


class MarketReactionZoneG4DMarketStateFreqAIResearchStrategy(
    MarketReactionZoneG4DFreqAIBaseStrategy
):
    include_local_state = True
    cache_blocks = ("wide",)


class MarketReactionZoneG4DLevelOnlyFreqAIResearchStrategy(
    MarketReactionZoneG4DFreqAIBaseStrategy
):
    include_local_state = False
    cache_blocks = ("level",)


class MarketReactionZoneG4DCombinedCurrentFreqAIResearchStrategy(
    MarketReactionZoneG4DFreqAIBaseStrategy
):
    include_local_state = True
    cache_blocks = ("wide", "level", "geometry")


class MarketReactionZoneG4DCombinedShuffledFreqAIResearchStrategy(
    MarketReactionZoneG4DFreqAIBaseStrategy
):
    include_local_state = True
    cache_blocks = ("wide", "placebo")


class MarketReactionZoneG4DSelectedMTFFreqAIResearchStrategy(
    MarketReactionZoneG4DFreqAIBaseStrategy
):
    include_local_state = True
    cache_blocks = ("wide", "level", "geometry", "mtf")


class MarketReactionZoneG4DLimitedInteractionsFreqAIResearchStrategy(
    MarketReactionZoneG4DFreqAIBaseStrategy
):
    include_local_state = True
    cache_blocks = ("wide", "level", "geometry", "mtf", "interaction")

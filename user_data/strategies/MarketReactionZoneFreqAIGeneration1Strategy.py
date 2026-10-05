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
    / "generation1_freqai"
    / "feature_cache"
)
DEFAULT_EVENT_CACHE_DIR = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "market_reaction_zones"
    / "generation1_freqai"
    / "event_cache"
)
TARGET_HORIZONS = (1, 2, 4, 8, 12, 24, 48)
TARGET_FAMILIES = (
    "absolute_excursion_atr",
    "range_ratio",
    "volume_ratio",
    "pressure_change_magnitude",
    "dwell_fraction",
    "crossing_rate",
)
TARGET_COLUMNS = tuple(
    f"&-event_{family}_{horizon}h" for horizon in TARGET_HORIZONS for family in TARGET_FAMILIES
)
MAX_TARGET_HORIZON_HOURS = max(TARGET_HORIZONS)


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
    """Give cache and Freqtrade candle timestamps one explicit merge dtype."""
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


class MarketReactionZoneG1DFreqAIBaseStrategy(IStrategy):
    """Contact-focused, direction-neutral Generation 1D FreqAI research base."""

    INTERFACE_VERSION = 3
    timeframe = "1h"
    startup_candle_count = 2200
    process_only_new_candles = True
    can_short = True

    minimal_roi = {"0": 100.0}
    stoploss = -0.99
    use_exit_signal = False
    trailing_stop = False

    include_crypto_wide_state = False
    feature_block: str | None = None
    block_shift_hours = 0

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
        """Expose only information known before the contact candle opens."""
        _ = metadata, kwargs
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
        known_close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0).shift(1)

        for horizon in TARGET_HORIZONS:
            dataframe[f"%-g1d_local_return_{horizon}h"] = close.pct_change(horizon).shift(1)
        dataframe["%-g1d_local_atr_pct"] = known_atr / known_close.abs()
        dataframe["%-g1d_local_body_fraction"] = (
            ((close - open_) / candle_range).clip(-1.0, 1.0).shift(1)
        )
        dataframe["%-g1d_local_close_location"] = known_close_location
        dataframe["%-g1d_local_range_vs_prior24"] = (
            known_range
            / candle_range.shift(2).rolling(24, min_periods=12).median().replace(0.0, np.nan)
        ).clip(0.0, 30.0)
        dataframe["%-g1d_local_volume_vs_prior24"] = (
            known_volume / volume.shift(2).rolling(24, min_periods=12).median().replace(0.0, np.nan)
        ).clip(0.0, 30.0)
        dataframe["%-g1d_local_volume_z_24"] = (
            (known_volume - volume.shift(2).rolling(24, min_periods=12).mean())
            / volume.shift(2).rolling(24, min_periods=12).std(ddof=0).replace(0.0, np.nan)
        ).clip(-10.0, 10.0)
        signed_volume = known_close_location.fillna(0.0) * known_volume.fillna(0.0)
        for horizon in (6, 24):
            dataframe[f"%-g1d_local_pressure_{horizon}h"] = signed_volume.rolling(
                horizon, min_periods=max(3, horizon // 2)
            ).sum() / known_volume.rolling(horizon, min_periods=max(3, horizon // 2)).sum().replace(
                0.0, np.nan
            )
        dataframe["%-g1d_local_pressure_change_6h_vs_24h"] = (
            dataframe["%-g1d_local_pressure_6h"] - dataframe["%-g1d_local_pressure_24h"]
        )

        self._append_fixed_context(dataframe, close, high, low, known_atr)
        return dataframe

    @staticmethod
    def _append_fixed_context(
        dataframe: DataFrame,
        close: Series,
        high: Series,
        low: Series,
        known_atr: Series,
    ) -> None:
        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
        loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
        relative_strength = gain / loss.replace(0.0, np.nan)
        dataframe["%-g1d_local_rsi_14"] = (100.0 - (100.0 / (1.0 + relative_strength))).shift(
            1
        ) / 100.0

        ema_12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, min_periods=9, adjust=False).mean()
        dataframe["%-g1d_local_macd_12_26_atr"] = (macd.shift(1) / known_atr).clip(-20.0, 20.0)
        dataframe["%-g1d_local_macd_histogram_atr"] = (
            (macd - macd_signal).shift(1) / known_atr
        ).clip(-20.0, 20.0)

        for period in (20, 50, 200):
            average = close.rolling(period, min_periods=period).mean()
            dataframe[f"%-g1d_local_sma_{period}_distance_atr"] = (
                (close.shift(1) - average.shift(1)) / known_atr
            ).clip(-30.0, 30.0)
        for period in (12, 26, 50):
            average = close.ewm(span=period, min_periods=period, adjust=False).mean()
            dataframe[f"%-g1d_local_ema_{period}_distance_atr"] = (
                (close.shift(1) - average.shift(1)) / known_atr
            ).clip(-30.0, 30.0)

        previous_close = close.shift(1)
        true_range = pd.concat(
            (
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ),
            axis=1,
        ).max(axis=1)
        upward = high.diff()
        downward = -low.diff()
        plus_dm = upward.where((upward > downward) & (upward > 0.0), 0.0)
        minus_dm = downward.where((downward > upward) & (downward > 0.0), 0.0)
        smoothed_tr = true_range.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
        plus_di = (
            100.0
            * plus_dm.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
            / smoothed_tr.replace(0.0, np.nan)
        )
        minus_di = (
            100.0
            * minus_dm.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
            / smoothed_tr.replace(0.0, np.nan)
        )
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
        dataframe["%-g1d_local_adx_14"] = (
            dx.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean().shift(1) / 100.0
        )
        dataframe["%-g1d_local_realised_volatility_24"] = (
            close.pct_change().rolling(24, min_periods=12).std(ddof=0).shift(1)
        )

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs: Any
    ) -> DataFrame:
        _ = kwargs
        pair = str(metadata.get("pair") or "")
        cache = self._aligned_feature_cache(dataframe, pair=pair)
        prefixes: list[str] = []
        if self.include_crypto_wide_state:
            prefixes.append("wide__")
        if self.feature_block:
            prefixes.append(f"{self.feature_block}__")
        selected = [
            column
            for column in cache.columns
            if any(column.startswith(prefix) for prefix in prefixes)
        ]
        if not selected:
            return dataframe
        block = cache[selected].copy()
        block.columns = [f"%-g1d_{column}" for column in selected]
        block.index = dataframe.index
        return pd.concat([dataframe, block], axis=1)

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs: Any) -> DataFrame:
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
        return dict(self.config.get("market_reaction_zone_g1d", {}))

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
        cache = self._feature_cache(pair).copy()
        if self.feature_block and self.block_shift_hours:
            block_prefix = f"{self.feature_block}__"
            block_columns = [column for column in cache if column.startswith(block_prefix)]
            shifted = cache[["date", *block_columns]].copy()
            shifted["date"] = shifted["date"] + pd.Timedelta(hours=self.block_shift_hours)
            cache = cache.drop(columns=block_columns).merge(
                shifted, on="date", how="left", validate="one_to_one"
            )
        left = DataFrame(
            {
                "date": utc_nanoseconds(dataframe["date"]),
                "_row": np.arange(len(dataframe), dtype=np.int64),
            }
        ).sort_values("date")
        aligned = pd.merge_asof(
            left,
            cache.sort_values("date"),
            on="date",
            direction="backward",
            allow_exact_matches=True,
            tolerance=pd.Timedelta(hours=1),
        ).sort_values("_row")
        output = aligned.drop(columns=["date", "_row"]).reset_index(drop=True)
        for column in output:
            values = pd.to_numeric(output[column], errors="coerce")
            if "distance_atr" in column:
                values = values.fillna(25.0)
            elif column.endswith("source_age_hours"):
                values = values.fillna(240.0)
            else:
                values = values.fillna(0.0)
            output[column] = values
        return output


class MarketReactionZoneG1DLocalStateFreqAIResearchStrategy(
    MarketReactionZoneG1DFreqAIBaseStrategy
):
    include_crypto_wide_state = False
    feature_block = None


class MarketReactionZoneG1DCryptoWideStateFreqAIResearchStrategy(
    MarketReactionZoneG1DFreqAIBaseStrategy
):
    include_crypto_wide_state = True
    feature_block = None


class MarketReactionZoneG1DNativeCurrentFreqAIResearchStrategy(
    MarketReactionZoneG1DFreqAIBaseStrategy
):
    include_crypto_wide_state = True
    feature_block = "native"


class MarketReactionZoneG1DNativePlaceboFreqAIResearchStrategy(
    MarketReactionZoneG1DFreqAIBaseStrategy
):
    include_crypto_wide_state = True
    feature_block = "native"
    block_shift_hours = 168


class MarketReactionZoneG1DVolumeProfileCurrentFreqAIResearchStrategy(
    MarketReactionZoneG1DFreqAIBaseStrategy
):
    include_crypto_wide_state = True
    feature_block = "vp"


class MarketReactionZoneG1DVolumeProfilePlaceboFreqAIResearchStrategy(
    MarketReactionZoneG1DFreqAIBaseStrategy
):
    include_crypto_wide_state = True
    feature_block = "vp"
    block_shift_hours = 168


class MarketReactionZoneG1DClusterCurrentFreqAIResearchStrategy(
    MarketReactionZoneG1DFreqAIBaseStrategy
):
    include_crypto_wide_state = True
    feature_block = "cluster"


class MarketReactionZoneG1DClusterPlaceboFreqAIResearchStrategy(
    MarketReactionZoneG1DFreqAIBaseStrategy
):
    include_crypto_wide_state = True
    feature_block = "cluster"
    block_shift_hours = 168

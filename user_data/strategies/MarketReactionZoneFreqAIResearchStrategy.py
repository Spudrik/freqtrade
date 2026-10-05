from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pandas import DataFrame, Series

from freqtrade.strategy import IStrategy


USER_DATA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "market_reaction_zones"
    / "generation0_cache"
)
TARGET_HORIZONS = (1, 2, 4, 8, 12, 24, 48)
NATIVE_LEVEL_TIMEFRAMES = ("1h",)
SELECTED_MTF_LEVEL_TIMEFRAMES = ("1h", "4h", "8h", "1d")
MISSING_DISTANCE_ATR = 25.0

# These are the unchanged level outputs that produced the clearest direct 0B
# expansion/transit or acceptance/stickiness differences. Sparse pattern scores
# are intentionally absent because their shuffled-score calibration added no
# repeatable ordering information.
SELECTED_LEVEL_COLUMNS = (
    "generic_rolling_high_24",
    "generic_rolling_low_24",
    "generic_rolling_high_168",
    "generic_rolling_low_168",
    "generic_rolling_high_720",
    "generic_rolling_low_720",
    "generic_bb20_upper",
    "generic_bb20_lower",
    "generic_round_nearest",
    "vp_lvn_above",
    "vp_lvn_below",
    "vp_hvn_above",
    "vp_hvn_below",
    "vp_poc",
    "vp_prior_poc",
)

SELECTED_ATTRIBUTE_COLUMNS = (
    "vp_value_area_width_pct",
    "vp_hvn_above_strength",
    "vp_hvn_below_strength",
    "vp_lvn_above_thinness",
    "vp_lvn_below_thinness",
    "vp_score_abs",
)


def _pair_file_stem(pair: str) -> str:
    normalized = pair.strip().upper()
    if "/" not in normalized:
        raise ValueError(f"Unsupported market-reaction pair: {pair!r}")
    return normalized.replace("/", "_").replace(":", "_")


def _numeric(frame: DataFrame, column: str, default: float = np.nan) -> Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _atr(dataframe: DataFrame, period: int = 14) -> Series:
    high = pd.to_numeric(dataframe["high"], errors="coerce")
    low = pd.to_numeric(dataframe["low"], errors="coerce")
    close = pd.to_numeric(dataframe["close"], errors="coerce")
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


def _future_extreme(series: Series, horizon: int, method: str) -> Series:
    ordered = pd.to_numeric(series, errors="coerce")
    rolling = ordered.iloc[::-1].rolling(horizon, min_periods=horizon)
    inclusive = rolling.max() if method == "max" else rolling.min()
    # The reversed rolling window at row i contains i..i+h-1. Move it back one
    # row so each label contains exactly the later candles i+1..i+h.
    return inclusive.iloc[::-1].shift(-1)


def _future_mean(series: Series, horizon: int) -> Series:
    ordered = pd.to_numeric(series, errors="coerce")
    inclusive = ordered.iloc[::-1].rolling(horizon, min_periods=horizon).mean()
    return inclusive.iloc[::-1].shift(-1)


def build_market_reaction_targets(dataframe: DataFrame) -> DataFrame:
    """Create direction-neutral future activity labels from later candles only."""
    out = dataframe.copy()
    close = pd.to_numeric(out["close"], errors="coerce").replace(0.0, np.nan)
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    volume = pd.to_numeric(out["volume"], errors="coerce").clip(lower=0.0)
    atr = _atr(out).replace(0.0, np.nan)
    candle_range = (high - low).replace(0.0, np.nan)
    close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)

    pre_range = candle_range.shift(1).rolling(24, min_periods=12).median()
    pre_volume = volume.shift(1).rolling(24, min_periods=12).median().replace(0.0, np.nan)
    pre_pressure = close_location.shift(1).rolling(24, min_periods=12).mean()

    for horizon in TARGET_HORIZONS:
        future_high = _future_extreme(high, horizon, "max")
        future_low = _future_extreme(low, horizon, "min")
        excursion = pd.concat(
            ((future_high - close).abs(), (close - future_low).abs()), axis=1
        ).max(axis=1)
        out[f"&-future_abs_excursion_{horizon}h_atr"] = (excursion / atr).clip(0.0, 30.0)
        out[f"&-future_range_ratio_{horizon}h"] = (
            _future_mean(candle_range, horizon) / pre_range.replace(0.0, np.nan)
        ).clip(0.0, 30.0)
        out[f"&-future_volume_ratio_{horizon}h"] = (
            _future_mean(volume, horizon) / pre_volume
        ).clip(0.0, 30.0)
        out[f"&-future_pressure_change_magnitude_{horizon}h"] = (
            (_future_mean(close_location, horizon) - pre_pressure).abs().clip(0.0, 2.0)
        )
    return out


class MarketReactionZoneFreqAIResearchStrategy(IStrategy):
    """Research-only FreqAI ladder for non-directional level reactions."""

    INTERFACE_VERSION = 3
    timeframe = "1h"
    startup_candle_count = 2200
    process_only_new_candles = True
    can_short = True

    minimal_roi = {"0": 100.0}
    stoploss = -0.99
    use_exit_signal = False
    trailing_stop = False

    include_market_context = True
    include_level_context = False
    level_timeframes = NATIVE_LEVEL_TIMEFRAMES
    _level_cache_frames: dict[str, DataFrame] = {}

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs: Any
    ) -> DataFrame:
        _ = period, metadata, kwargs
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs: Any
    ) -> DataFrame:
        _ = metadata, kwargs
        if not self.include_market_context:
            return dataframe

        close = pd.to_numeric(dataframe["close"], errors="coerce").replace(0.0, np.nan)
        open_ = pd.to_numeric(dataframe["open"], errors="coerce")
        high = pd.to_numeric(dataframe["high"], errors="coerce")
        low = pd.to_numeric(dataframe["low"], errors="coerce")
        volume = pd.to_numeric(dataframe["volume"], errors="coerce").clip(lower=0.0)
        candle_range = (high - low).replace(0.0, np.nan)
        atr = _atr(dataframe).replace(0.0, np.nan)
        close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)

        for horizon in TARGET_HORIZONS:
            dataframe[f"%-return_{horizon}h"] = close.pct_change(horizon)
        dataframe["%-atr_pct"] = atr / close
        dataframe["%-body_fraction"] = ((close - open_) / candle_range).clip(-1.0, 1.0)
        dataframe["%-close_location"] = close_location
        dataframe["%-range_vs_prior24"] = (
            candle_range
            / candle_range.shift(1).rolling(24, min_periods=12).median().replace(0.0, np.nan)
        ).clip(0.0, 30.0)
        dataframe["%-volume_vs_prior24"] = (
            volume / volume.shift(1).rolling(24, min_periods=12).median().replace(0.0, np.nan)
        ).clip(0.0, 30.0)
        dataframe["%-volume_z_24"] = (
            (volume - volume.shift(1).rolling(24, min_periods=12).mean())
            / volume.shift(1).rolling(24, min_periods=12).std(ddof=0).replace(0.0, np.nan)
        ).clip(-10.0, 10.0)
        signed_volume = close_location.fillna(0.0) * volume.fillna(0.0)
        for horizon in (6, 24):
            dataframe[f"%-pressure_{horizon}h"] = signed_volume.rolling(
                horizon, min_periods=max(3, horizon // 2)
            ).sum() / volume.rolling(horizon, min_periods=max(3, horizon // 2)).sum().replace(
                0.0, np.nan
            )
        dataframe["%-pressure_change_6h_vs_24h"] = (
            dataframe["%-pressure_6h"] - dataframe["%-pressure_24h"]
        )

        self._append_fixed_indicator_context(dataframe, close, high, low, volume, atr)
        return dataframe

    @staticmethod
    def _append_fixed_indicator_context(
        dataframe: DataFrame,
        close: Series,
        high: Series,
        low: Series,
        volume: Series,
        atr: Series,
    ) -> None:
        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
        loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
        relative_strength = gain / loss.replace(0.0, np.nan)
        dataframe["%-rsi_14"] = (100.0 - (100.0 / (1.0 + relative_strength))) / 100.0

        ema_12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, min_periods=9, adjust=False).mean()
        dataframe["%-macd_12_26_atr"] = (macd / atr).clip(-20.0, 20.0)
        dataframe["%-macd_histogram_atr"] = ((macd - macd_signal) / atr).clip(-20.0, 20.0)

        for period in (20, 50, 200):
            sma = close.rolling(period, min_periods=period).mean()
            dataframe[f"%-sma_{period}_distance_atr"] = ((close - sma) / atr).clip(-30.0, 30.0)
        for period in (12, 26, 50):
            ema = close.ewm(span=period, min_periods=period, adjust=False).mean()
            dataframe[f"%-ema_{period}_distance_atr"] = ((close - ema) / atr).clip(-30.0, 30.0)

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
        dataframe["%-adx_14"] = (
            dx.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean() / 100.0
        )
        dataframe["%-realised_volatility_24"] = (
            close.pct_change().rolling(24, min_periods=12).std(ddof=0)
        )

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs: Any
    ) -> DataFrame:
        _ = kwargs
        if not self.include_level_context:
            return dataframe
        pair = str(metadata.get("pair") or "")
        close = pd.to_numeric(dataframe["close"], errors="coerce").replace(0.0, np.nan)
        base_atr = _atr(dataframe).replace(0.0, np.nan)
        shift_hours = int(
            self.config.get("market_reaction_zone_freqai", {}).get("level_feature_shift_hours", 0)
            or 0
        )
        feature_blocks: list[DataFrame] = []
        for timeframe in self.level_timeframes:
            aligned = self._aligned_level_cache(
                dataframe,
                pair=pair,
                timeframe=timeframe,
                shift_hours=shift_hours,
            )
            feature_blocks.append(
                self._level_feature_frame(
                    aligned=aligned,
                    close=close,
                    base_atr=base_atr,
                    timeframe=timeframe,
                    dataframe=dataframe,
                )
            )
        return pd.concat([dataframe, *feature_blocks], axis=1)

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs: Any) -> DataFrame:
        _ = metadata, kwargs
        return build_market_reaction_targets(dataframe)

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

    def _cache_dir(self) -> Path:
        configured = self.config.get("market_reaction_zone_freqai", {}).get("level_cache_dir")
        path = Path(configured) if configured else DEFAULT_CACHE_DIR
        if not path.is_absolute():
            path = USER_DATA_DIR / path
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path.resolve()

    def _load_level_cache(self, pair: str, timeframe: str) -> DataFrame:
        path = self._cache_dir() / (f"{_pair_file_stem(pair)}-{timeframe}-core-generic.parquet")
        cache_key = str(path)
        cached = self._level_cache_frames.get(cache_key)
        if cached is not None:
            return cached
        if not path.is_file():
            raise FileNotFoundError(path)
        required = {
            "available_at",
            *SELECTED_LEVEL_COLUMNS,
            *SELECTED_ATTRIBUTE_COLUMNS,
        }
        schema_names = set(pq.ParquetFile(path).schema.names)
        missing = sorted(required.difference(schema_names))
        if missing:
            raise ValueError(f"{path} is missing required level columns: {missing}")
        frame = pd.read_parquet(path, columns=sorted(required))
        frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True, errors="raise")
        frame = (
            frame.drop_duplicates("available_at", keep="last")
            .sort_values("available_at")
            .reset_index(drop=True)
        )
        self._level_cache_frames[cache_key] = frame
        return frame

    def _aligned_level_cache(
        self,
        dataframe: DataFrame,
        *,
        pair: str,
        timeframe: str,
        shift_hours: int,
    ) -> DataFrame:
        cache = self._load_level_cache(pair, timeframe).copy()
        if shift_hours:
            cache["available_at"] = cache["available_at"] + pd.Timedelta(hours=shift_hours)
        decision_time = pd.to_datetime(dataframe["date"], utc=True, errors="raise") + pd.Timedelta(
            hours=1
        )
        left = DataFrame(
            {
                "_decision_time": decision_time,
                "_row_order": np.arange(len(dataframe), dtype=np.int64),
            }
        ).sort_values("_decision_time")
        aligned = pd.merge_asof(
            left,
            cache.sort_values("available_at"),
            left_on="_decision_time",
            right_on="available_at",
            direction="backward",
            allow_exact_matches=True,
        ).sort_values("_row_order")
        aligned.index = dataframe.index
        return aligned

    @staticmethod
    def _level_feature_frame(
        *,
        dataframe: DataFrame,
        aligned: DataFrame,
        close: Series,
        base_atr: Series,
        timeframe: str,
    ) -> DataFrame:
        features: dict[str, Series] = {}
        decision_time = pd.to_datetime(dataframe["date"], utc=True, errors="coerce") + pd.Timedelta(
            hours=1
        )
        available_at = pd.to_datetime(aligned["available_at"], utc=True, errors="coerce")
        features[f"%-level_{timeframe}_source_age_hours"] = (
            (decision_time - available_at).dt.total_seconds() / 3600.0
        ).clip(0.0, 240.0)
        for column in SELECTED_LEVEL_COLUMNS:
            level = _numeric(aligned, column)
            present = level.notna() & level.gt(0.0)
            signed_distance = ((level - close) / base_atr).clip(
                -MISSING_DISTANCE_ATR, MISSING_DISTANCE_ATR
            )
            absolute_distance = signed_distance.abs()
            prefix = f"%-level_{timeframe}_{column}"
            features[f"{prefix}_present"] = present.astype(float)
            features[f"{prefix}_signed_distance_atr"] = signed_distance.where(
                present, MISSING_DISTANCE_ATR
            )
            features[f"{prefix}_absolute_distance_atr"] = absolute_distance.where(
                present, MISSING_DISTANCE_ATR
            )
            for threshold, label in ((0.10, "10"), (0.25, "25"), (0.50, "50")):
                features[f"{prefix}_within_{label}pct_atr"] = (
                    present & absolute_distance.le(threshold)
                ).astype(float)
        for column in SELECTED_ATTRIBUTE_COLUMNS:
            values = _numeric(aligned, column)
            features[f"%-level_{timeframe}_{column}"] = values.fillna(0.0).clip(-100.0, 100.0)
        return DataFrame(features, index=dataframe.index)


class MarketReactionZoneMarketStateFreqAIResearchStrategy(MarketReactionZoneFreqAIResearchStrategy):
    include_market_context = True
    include_level_context = False


class MarketReactionZoneLevelOnlyFreqAIResearchStrategy(MarketReactionZoneFreqAIResearchStrategy):
    include_market_context = False
    include_level_context = True
    level_timeframes = NATIVE_LEVEL_TIMEFRAMES


class MarketReactionZoneCombinedFreqAIResearchStrategy(MarketReactionZoneFreqAIResearchStrategy):
    include_market_context = True
    include_level_context = True
    level_timeframes = NATIVE_LEVEL_TIMEFRAMES


class MarketReactionZoneCombinedMtfFreqAIResearchStrategy(MarketReactionZoneFreqAIResearchStrategy):
    include_market_context = True
    include_level_context = True
    level_timeframes = SELECTED_MTF_LEVEL_TIMEFRAMES

from __future__ import annotations

from functools import reduce
from pathlib import Path
import re

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy


USER_DATA_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "orderbook_timeframe_value"
    / "orderbook_timeframe_value_latest.parquet"
)


class OrderbookTimeframeValueBaseFreqAIStrategy(IStrategy):
    timeframe = "1h"
    startup_candle_count = 24
    minimal_roi = {"0": 0.0}
    stoploss = -0.99
    can_short = True
    process_only_new_candles = True
    use_exit_signal = True

    orderbook_prefixes: tuple[str, ...] = ()
    require_all_usable = True
    include_comparison_features = False
    min_feature_nonnull_ratio = 0.95
    _dataset_cache: DataFrame | None = None

    @classmethod
    def _load_dataset(cls) -> DataFrame:
        if cls._dataset_cache is not None:
            return cls._dataset_cache
        if not DATASET_PATH.exists():
            raise FileNotFoundError(DATASET_PATH)
        frame = pd.read_parquet(DATASET_PATH)
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
        cls._dataset_cache = frame.reset_index(drop=True)
        return cls._dataset_cache

    @classmethod
    def _feature_columns(cls, frame: DataFrame) -> list[str]:
        columns = [
            column
            for column in frame.columns
            if column.startswith("px_")
            or any(column.startswith(prefix) for prefix in cls.orderbook_prefixes)
            or (cls.include_comparison_features and column.startswith("obtf_cmp_"))
        ]
        blocked = ("_usable", "_strict", "_coverage", "_market_count")
        selected = [column for column in columns if not any(token in column for token in blocked)]
        ready = pd.to_numeric(frame.get("obtf_all_usable"), errors="coerce").fillna(0.0).gt(0.0)
        if not ready.any():
            ready = pd.Series(True, index=frame.index)
        feature_columns = []
        for column in selected:
            if not pd.api.types.is_numeric_dtype(frame[column]):
                continue
            values = pd.to_numeric(frame.loc[ready, column], errors="coerce")
            if values.notna().mean() < cls.min_feature_nonnull_ratio:
                continue
            if values.nunique(dropna=True) < 2:
                continue
            feature_columns.append(column)
        return feature_columns

    @staticmethod
    def _freqai_name(source: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", source).strip("_")
        return f"%-{safe}"

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, metadata: dict, **kwargs) -> DataFrame:
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataset = self._load_dataset()
        selected_columns = self._feature_columns(dataset)
        features = dataset[["date", "obtf_all_usable", "obtf_all_strict", *selected_columns]].copy()
        ready = pd.to_numeric(features["obtf_all_usable"], errors="coerce").fillna(0.0).gt(0.0)
        for column in selected_columns:
            values = pd.to_numeric(features[column], errors="coerce")
            median = values[ready].median()
            if pd.notna(median):
                features[column] = values.fillna(median)
        rename = {column: self._freqai_name(column) for column in features.columns if column not in {"date", "obtf_all_usable", "obtf_all_strict"}}
        features = features.rename(columns=rename)
        dataframe = dataframe.copy()
        dataframe["date"] = pd.to_datetime(dataframe["date"], utc=True, errors="coerce")
        return dataframe.merge(features, on="date", how="left", sort=False)

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe = dataframe.copy()
        for horizon in (1, 2, 4, 6):
            dataframe[f"&-future_return_{horizon}h"] = dataframe["close"].shift(-horizon) / dataframe["close"] - 1.0
            upside = self._future_extreme(dataframe["high"], horizon, "max") / dataframe["close"] - 1.0
            drawdown = self._future_extreme(dataframe["low"], horizon, "min") / dataframe["close"] - 1.0
            dataframe[f"&-future_max_upside_{horizon}h"] = upside
            dataframe[f"&-future_max_drawdown_{horizon}h"] = drawdown
            reward, danger, large_move = self._path_thresholds(horizon)
            dataframe[f"&-long_reward_before_danger_{horizon}h"] = self._reward_before_danger(
                dataframe["close"], horizon=horizon, reward=reward, danger=-danger, long_side=True
            )
            dataframe[f"&-short_reward_before_danger_{horizon}h"] = self._reward_before_danger(
                dataframe["close"], horizon=horizon, reward=reward, danger=-danger, long_side=False
            )
            dataframe[f"&-large_upside_next_{horizon}h"] = dataframe[f"&-future_max_upside_{horizon}h"].ge(large_move).astype(float).where(dataframe[f"&-future_max_upside_{horizon}h"].notna())
            dataframe[f"&-large_drawdown_next_{horizon}h"] = dataframe[f"&-future_max_drawdown_{horizon}h"].le(-large_move).astype(float).where(dataframe[f"&-future_max_drawdown_{horizon}h"].notna())
        if self.require_all_usable:
            mask = pd.to_numeric(dataframe.get("obtf_all_usable"), errors="coerce").fillna(0.0).gt(0.0)
            for column in [column for column in dataframe.columns if column.startswith("&-")]:
                dataframe[column] = dataframe[column].where(mask)
        return dataframe

    @staticmethod
    def _future_extreme(series: pd.Series, horizon: int, method: str) -> pd.Series:
        future = series.shift(-1).iloc[::-1]
        rolling = future.rolling(horizon, min_periods=horizon)
        result = rolling.max() if method == "max" else rolling.min()
        return result.iloc[::-1]

    @staticmethod
    def _path_thresholds(horizon: int) -> tuple[float, float, float]:
        if horizon <= 1:
            return 0.006, 0.004, 0.010
        if horizon <= 2:
            return 0.010, 0.006, 0.014
        if horizon <= 4:
            return 0.015, 0.010, 0.020
        return 0.025, 0.015, 0.030

    @staticmethod
    def _reward_before_danger(close: pd.Series, *, horizon: int, reward: float, danger: float, long_side: bool) -> pd.Series:
        values = close.to_numpy(dtype=float)
        out = np.full(len(values), np.nan)
        for idx, current in enumerate(values):
            if not np.isfinite(current) or idx + horizon >= len(values):
                continue
            future = values[idx + 1 : idx + horizon + 1] / current - 1.0
            if long_side:
                reward_hits = np.where(future >= reward)[0]
                danger_hits = np.where(future <= danger)[0]
            else:
                reward_hits = np.where(future <= -reward)[0]
                danger_hits = np.where(future >= -danger)[0]
            first_reward = reward_hits[0] if reward_hits.size else None
            first_danger = danger_hits[0] if danger_hits.size else None
            if first_reward is None:
                out[idx] = 0.0
            elif first_danger is None:
                out[idx] = 1.0
            elif first_reward == first_danger:
                out[idx] = np.nan
            else:
                out[idx] = float(first_reward < first_danger)
        return pd.Series(out, index=close.index)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self.freqai.start(dataframe, metadata, self)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [dataframe.get("do_predict", 0) == 1, dataframe.get("&-future_return_4h", 0) > 0]
        dataframe.loc[reduce(lambda left, right: left & right, conditions), ["enter_long", "enter_tag"]] = (
            1,
            "orderbook_tf_research_probe_long",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [dataframe.get("do_predict", 0) == 1, dataframe.get("&-future_return_4h", 0) < 0]
        dataframe.loc[reduce(lambda left, right: left & right, conditions), ["exit_long", "exit_tag"]] = (
            1,
            "orderbook_tf_research_probe_exit",
        )
        return dataframe


class OrderbookTfPriceOnlyFreqAIStrategy(OrderbookTimeframeValueBaseFreqAIStrategy):
    orderbook_prefixes = ()


class OrderbookTf1sFreqAIStrategy(OrderbookTimeframeValueBaseFreqAIStrategy):
    orderbook_prefixes = ("obtf_1s_",)


class OrderbookTf1mFreqAIStrategy(OrderbookTimeframeValueBaseFreqAIStrategy):
    orderbook_prefixes = ("obtf_1m_",)


class OrderbookTf5mFreqAIStrategy(OrderbookTimeframeValueBaseFreqAIStrategy):
    orderbook_prefixes = ("obtf_5m_",)


class OrderbookTf1hFreqAIStrategy(OrderbookTimeframeValueBaseFreqAIStrategy):
    orderbook_prefixes = ("obtf_1h_",)


class OrderbookTfAllFreqAIStrategy(OrderbookTimeframeValueBaseFreqAIStrategy):
    orderbook_prefixes = ("obtf_1s_", "obtf_1m_", "obtf_5m_", "obtf_1h_")
    include_comparison_features = True

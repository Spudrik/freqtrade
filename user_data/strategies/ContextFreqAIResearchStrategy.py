from __future__ import annotations

from functools import reduce
from pathlib import Path
import sqlite3

import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class ContextFreqAIResearchStrategy(IStrategy):
    """
    Research-only FreqAI strategy for numeric external context features.

    It reads the derived context_features_1h SQL table. It does not scrape,
    parse article text, call APIs, or alter any production strategy behavior.
    """

    minimal_roi = {"0": 0.0}
    stoploss = -0.99
    timeframe = "1h"
    startup_candle_count = 72
    process_only_new_candles = True
    can_short = False
    use_exit_signal = True

    _context_cache: DataFrame | None = None

    @classmethod
    def _context_db_path(cls) -> Path:
        user_data_dir = Path(__file__).resolve().parents[1]
        return user_data_dir / "research_news_data" / "context_features" / "context_features.sqlite"

    @classmethod
    def _load_context_features(cls) -> DataFrame:
        if cls._context_cache is not None:
            return cls._context_cache
        db_path = cls._context_db_path()
        if not db_path.exists():
            cls._context_cache = DataFrame({"date": pd.Series(dtype="datetime64[ns, UTC]")})
            return cls._context_cache
        with sqlite3.connect(str(db_path), timeout=10.0) as conn:
            frame = pd.read_sql_query("SELECT * FROM context_features_1h ORDER BY date", conn)
        if frame.empty:
            cls._context_cache = DataFrame({"date": pd.Series(dtype="datetime64[ns, UTC]")})
            return cls._context_cache
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        skip_columns = {
            "date",
            "generated_at",
            "min_source_available_at",
            "max_source_available_at",
            "schema_version",
            "feature_config_hash",
            "news_rows_24h",
            "web_rows_24h",
            "global_metrics_available",
            "missing_available_at_rows",
            "feature_history_hours",
        }
        feature_columns = [column for column in frame.columns if column not in skip_columns]
        cleaned = frame[["date", *feature_columns]].dropna(subset=["date"]).copy()
        for column in feature_columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        cls._context_cache = cleaned.sort_values("date").reset_index(drop=True)
        return cls._context_cache

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe["%-pct_change"] = dataframe["close"].pct_change()
        dataframe["%-raw_close"] = dataframe["close"]
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-volume_change"] = dataframe["volume"].pct_change()
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        context = self._load_context_features()
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        if context.empty:
            return dataframe
        left = dataframe[["date"]].copy()
        left["date"] = pd.to_datetime(left["date"], utc=True, errors="coerce")
        left["_rowid"] = range(len(left))
        merged = pd.merge_asof(
            left.sort_values("date"),
            context,
            on="date",
            direction="backward",
        ).sort_values("_rowid")
        for column in context.columns:
            if column == "date":
                continue
            dataframe[f"%-ctx_{column}"] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0).to_numpy()
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe["&-future_return_1h"] = dataframe["close"].shift(-1) / dataframe["close"] - 1.0
        dataframe["&-future_return_6h"] = dataframe["close"].shift(-6) / dataframe["close"] - 1.0
        dataframe["&-future_return_24h"] = dataframe["close"].shift(-24) / dataframe["close"] - 1.0
        dataframe["&-future_up_6h"] = (dataframe["&-future_return_6h"] > 0).astype(float)
        dataframe["&-future_up_24h"] = (dataframe["&-future_return_24h"] > 0).astype(float)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self.freqai.start(dataframe, metadata, self)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_conditions = [
            dataframe["do_predict"] == 1,
            dataframe["&-future_return_6h"] > 0,
        ]
        if long_conditions:
            dataframe.loc[
                reduce(lambda left, right: left & right, long_conditions),
                ["enter_long", "enter_tag"],
            ] = (1, "context_freqai_research_long")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        exit_conditions = [
            dataframe["do_predict"] == 1,
            dataframe["&-future_return_6h"] < 0,
        ]
        if exit_conditions:
            dataframe.loc[
                reduce(lambda left, right: left & right, exit_conditions),
                ["exit_long", "exit_tag"],
            ] = (1, "context_freqai_research_exit")
        return dataframe

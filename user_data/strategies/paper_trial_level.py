"""Paper B: price-confirmed rejection at previously calculated levels."""

from __future__ import annotations

import numpy as np
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import merge_informative_pair
from user_data.Indicators.complex_volume_profile import add_volume_profile
from user_data.strategies.paper_trial_common import PaperTrialBase


class PaperTrialLevel(PaperTrialBase):
    timeframe = "1h"
    informative_timeframe = "4h"
    level_columns = (
        "prior_24h_high", "prior_24h_low", "rolling_20_high_4h", "rolling_20_low_4h",
        "vp_poc_4h", "vp_hvn_above_4h", "vp_hvn_below_4h",
        "vp_lvn_above_4h", "vp_lvn_below_4h",
    )

    def informative_pairs(self):
        return [(pair, self.informative_timeframe) for pair in self.dp.current_whitelist()]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["paper_atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["prior_24h_high"] = dataframe["high"].shift(1).rolling(24).max()
        dataframe["prior_24h_low"] = dataframe["low"].shift(1).rolling(24).min()
        dataframe["prior_20_volume_median"] = dataframe["volume"].shift(1).rolling(20).median()

        four_hour = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="4h").copy()
        four_hour["rolling_20_high"] = four_hour["high"].rolling(20).max()
        four_hour["rolling_20_low"] = four_hour["low"].rolling(20).min()
        four_hour = add_volume_profile(four_hour)
        selected = [
            "date", "rolling_20_high", "rolling_20_low", "vp_poc",
            "vp_hvn_above", "vp_hvn_below", "vp_lvn_above", "vp_lvn_below",
        ]
        return merge_informative_pair(
            dataframe, four_hour[selected], self.timeframe, self.informative_timeframe,
            ffill=True,
        )

    def _volume_gate(self, dataframe: DataFrame):
        return dataframe["volume"] > 0

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        zone = 0.25 * dataframe["paper_atr"]
        previous_close = dataframe["close"].shift(1)
        long_hits = []
        short_hits = []
        for name in self.level_columns:
            level = dataframe[name]
            touch = (dataframe["low"] <= level + zone) & (dataframe["high"] >= level - zone)
            long_hits.append((previous_close > level + zone) & touch & (dataframe["close"] > level + zone))
            short_hits.append((previous_close < level - zone) & touch & (dataframe["close"] < level - zone))

        long_count = np.asarray(long_hits, dtype=np.int8).sum(axis=0)
        short_count = np.asarray(short_hits, dtype=np.int8).sum(axis=0)
        allowed = dataframe["paper_atr"].notna() & self._volume_gate(dataframe)
        long = allowed & (long_count > 0) & (short_count == 0)
        short = allowed & (short_count > 0) & (long_count == 0)
        dataframe.loc[long, "enter_long"] = 1
        dataframe.loc[short, "enter_short"] = 1
        dataframe.loc[long, "enter_tag"] = np.where(long_count[long] > 1, "level_cluster_long", "level_single_long")
        dataframe.loc[short, "enter_tag"] = np.where(short_count[short] > 1, "level_cluster_short", "level_single_short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

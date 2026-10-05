"""Paper A: completed 1h range breakout, without context gates."""

from __future__ import annotations

import talib.abstract as ta
from pandas import DataFrame

from user_data.strategies.paper_trial_common import PaperTrialBase


class PaperTrialBreakout(PaperTrialBase):
    timeframe = "1h"

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["paper_atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["prior_24h_high"] = dataframe["high"].shift(1).rolling(24).max()
        dataframe["prior_24h_low"] = dataframe["low"].shift(1).rolling(24).min()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ready = dataframe["paper_atr"].notna() & (dataframe["volume"] > 0)
        long = ready & (dataframe["close"] > dataframe["prior_24h_high"])
        short = ready & (dataframe["close"] < dataframe["prior_24h_low"])
        dataframe.loc[long, ["enter_long", "enter_tag"]] = (1, "range_break_long")
        dataframe.loc[short, ["enter_short", "enter_tag"]] = (1, "range_break_short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

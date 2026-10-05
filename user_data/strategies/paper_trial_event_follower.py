"""Paper D: observed BTC-led post-release movement into four follower coins."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from user_data.strategies.paper_trial_common import PaperTrialBase


CALENDAR = Path(__file__).resolve().parents[1] / "configs" / "paper_trial_event_calendar_2026.json"
LEADERS = ("BTC/USDT:USDT", "ETH/USDT:USDT")
FOLLOWERS = ("SOL/USDT:USDT", "BNB/USDT:USDT", "DOGE/USDT:USDT", "1000PEPE/USDT:USDT")


def _eligible_clock_rows() -> list[tuple[pd.Timestamp, str, int]]:
    with CALENDAR.open(encoding="utf-8") as handle:
        rows = json.load(handle)["events"]
    result = []
    for ordinal, event in enumerate(row for row in rows if row["auto_eligible"]):
        event_time = pd.Timestamp(event["release_utc"])
        result.append((event_time.ceil("15min"), event["id"], ordinal))
    if len({row[0] for row in result}) != len(result):
        raise ValueError("Two automatic paper events share a 15-minute decision candle")
    return result


class PaperTrialEventFollower(PaperTrialBase):
    timeframe = "15m"
    paper_hold_hours = 4
    startup_candle_count = 100

    def informative_pairs(self):
        return [(pair, self.timeframe) for pair in LEADERS]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["paper_atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["paper_prior_atr"] = dataframe["paper_atr"].shift(1)
        dataframe["paper_prior_close"] = dataframe["close"].shift(1)
        dataframe["paper_contiguous"] = dataframe["date"].diff() == pd.Timedelta(minutes=15)
        for leader_name, pair in (("btc", LEADERS[0]), ("eth", LEADERS[1])):
            leader = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe).copy()
            leader[f"{leader_name}_atr"] = ta.ATR(leader, timeperiod=14).shift(1)
            leader[f"{leader_name}_prior_close"] = leader["close"].shift(1)
            leader[f"{leader_name}_contiguous"] = leader["date"].diff() == pd.Timedelta(minutes=15)
            leader = leader.rename(columns={"close": f"{leader_name}_close"})
            selected = ["date", f"{leader_name}_close", f"{leader_name}_prior_close",
                        f"{leader_name}_atr", f"{leader_name}_contiguous"]
            dataframe = dataframe.merge(leader[selected], on="date", how="left", validate="one_to_one")
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if metadata["pair"] not in FOLLOWERS:
            return dataframe
        clocks = _eligible_clock_rows()
        clock_by_date = {time: (event_id, ordinal) for time, event_id, ordinal in clocks}
        event_dates = dataframe["date"].isin(clock_by_date)
        selected = pd.Series(False, index=dataframe.index)
        for time, (_, ordinal) in clock_by_date.items():
            rotated = FOLLOWERS[ordinal % len(FOLLOWERS):] + FOLLOWERS[:ordinal % len(FOLLOWERS)]
            if metadata["pair"] in rotated[:3]:
                selected |= dataframe["date"] == time

        btc_move = dataframe["btc_close"] - dataframe["btc_prior_close"]
        eth_move = dataframe["eth_close"] - dataframe["eth_prior_close"]
        own_move = dataframe["close"] - dataframe["paper_prior_close"]
        ready = (
            event_dates & selected & dataframe["paper_contiguous"]
            & dataframe["btc_contiguous"] & dataframe["eth_contiguous"]
            & dataframe["paper_prior_atr"].gt(0)
            & dataframe["btc_atr"].gt(0) & dataframe["eth_atr"].gt(0)
            & (dataframe["volume"] > 0)
        )
        leader_up = btc_move >= (0.5 * dataframe["btc_atr"])
        leader_down = btc_move <= (-0.5 * dataframe["btc_atr"])
        long = ready & leader_up & (eth_move > -0.5 * dataframe["eth_atr"])
        short = ready & leader_down & (eth_move < 0.5 * dataframe["eth_atr"])
        long &= own_move >= -0.5 * dataframe["paper_prior_atr"]
        long &= own_move <= dataframe["paper_prior_atr"]
        short &= own_move <= 0.5 * dataframe["paper_prior_atr"]
        short &= own_move >= -dataframe["paper_prior_atr"]
        dataframe.loc[long, "enter_long"] = 1
        dataframe.loc[short, "enter_short"] = 1
        event_ids = dataframe["date"].map({date: event_id for date, (event_id, _) in clock_by_date.items()})
        dataframe.loc[long, "enter_tag"] = "event:" + event_ids[long].astype(str) + ":long"
        dataframe.loc[short, "enter_tag"] = "event:" + event_ids[short].astype(str) + ":short"
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

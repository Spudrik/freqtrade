"""Paper-only, higher-risk directional test after an observed Bitcoin impulse.

FreqAI research supports activity persistence and calculated reaction locations,
not a proven direction forecast.  This strategy tests the explicit hypothesis
that a completed, busy BTC move plus a same-side local break in another coin can
be traded in the observed direction.  It does not trade before the BTC move.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from math import isfinite

import pandas as pd
from pandas import DataFrame

from freqtrade.enums import RunMode
from user_data.strategies.integrated_paper import OUTPUT
from user_data.strategies.integrated_paper_context import load_luna_context
from user_data.strategies.paper_trial_level import PaperTrialLevel


LOG = logging.getLogger(__name__)
BTC = "BTC/USDT:USDT"
LUNA_CONTEXT = OUTPUT / "luna_context.json"


class PaperLeaderImpulse(PaperTrialLevel):
    """Test high-risk follower continuation; never claim that BTC or news caused it."""

    paper_hold_hours = 48
    stoploss = -0.08  # Emergency fallback; normal stop is the inherited 1.5-entry-ATR rule.

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._pending_stake: dict[tuple[str, str, str], tuple[datetime, float]] = {}

    def bot_start(self, **kwargs) -> None:
        _ = kwargs
        if self.config.get("dry_run") is not True or self.config.get("runmode") != RunMode.DRY_RUN:
            raise RuntimeError("PaperLeaderImpulse is dry-run only")
        if self.config.get("bot_name") != "paper_leader_impulse":
            raise RuntimeError("Unknown high-risk paper account")
        if self.config.get("trading_mode") != "futures" or self.config.get("exchange", {}).get("name") != "binance":
            raise RuntimeError("PaperLeaderImpulse requires the Binance futures paper configuration")
        if any(self.config.get("exchange", {}).get(key) for key in ("key", "secret")):
            raise RuntimeError("Paper account must not contain exchange credentials")
        if self.config.get("force_entry_enable") or self.config.get("api_server", {}).get("enabled"):
            raise RuntimeError("PaperLeaderImpulse forbids force entries and an order API")

    def informative_pairs(self) -> list[tuple[str, str]]:
        return sorted(set(super().informative_pairs()) | {(BTC, "1h")})

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        frame = super().populate_indicators(dataframe, metadata)
        frame["local_prior_6h_high"] = frame["high"].shift(1).rolling(6).max()
        frame["local_prior_6h_low"] = frame["low"].shift(1).rolling(6).min()
        frame["local_prior_volume"] = frame["volume"].shift(1).rolling(20).median()
        leader = self.dp.get_pair_dataframe(pair=BTC, timeframe="1h").copy()
        leader["btc_change"] = leader["close"].pct_change()
        leader["btc_prior_volume"] = leader["volume"].shift(1).rolling(20).median()
        leader = leader[["date", "btc_change", "volume", "btc_prior_volume"]].rename(
            columns={"volume": "btc_volume"}
        )
        return frame.merge(leader, on="date", how="left", validate="one_to_one")

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if metadata["pair"] == BTC:
            dataframe["enter_long"] = 0
            dataframe["enter_short"] = 0
            dataframe["enter_tag"] = None
            return dataframe
        ready = (
            dataframe["paper_atr"].notna()
            & dataframe["local_prior_volume"].gt(0)
            & dataframe["btc_prior_volume"].gt(0)
            & dataframe["volume"].ge(dataframe["local_prior_volume"])
            & dataframe["btc_volume"].ge(1.25 * dataframe["btc_prior_volume"])
        )
        long = ready & dataframe["btc_change"].ge(0.006) & dataframe["close"].gt(dataframe["local_prior_6h_high"])
        short = ready & dataframe["btc_change"].le(-0.006) & dataframe["close"].lt(dataframe["local_prior_6h_low"])
        dataframe.loc[long, ["enter_long", "enter_tag"]] = (1, "btc_impulse_local_break_long")
        dataframe.loc[short, ["enter_short", "enter_tag"]] = (1, "btc_impulse_local_break_short")
        return dataframe

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs) -> float:
        _ = (pair, current_time, current_rate, proposed_leverage, entry_tag, side, kwargs)
        if float(max_leverage) < 2.0:
            raise RuntimeError("High-risk paper rule requires 2x permitted leverage")
        return 2.0

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, leverage, entry_tag, side, **kwargs) -> float:
        _ = (proposed_stake, leverage, kwargs)
        key = (pair, side, str(entry_tag))
        self._pending_stake.pop(key, None)
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if frame.empty:
            raise RuntimeError(f"Missing analyzed paper candle for {pair}")
        row = frame.iloc[-1]
        candle_end = pd.Timestamp(row["date"])
        if candle_end.tzinfo is None:
            candle_end = candle_end.tz_localize("UTC")
        if not timedelta(0) <= current_time - (candle_end.to_pydatetime() + timedelta(hours=1)) < timedelta(hours=2):
            raise RuntimeError(f"Stale or future paper candle for {pair}")
        atr = float(row["paper_atr"])
        if not isfinite(atr) or atr <= 0:
            raise RuntimeError(f"Missing current ATR for {pair}")
        opposite = ("rolling_20_high_4h", "vp_hvn_above_4h", "vp_lvn_above_4h") if side == "long" else (
            "rolling_20_low_4h", "vp_hvn_below_4h", "vp_lvn_below_4h"
        )
        nearby = any(
            pd.notna(row[name]) and 0 < (float(row[name]) - current_rate) * (1 if side == "long" else -1) <= atr
            for name in opposite
        )
        luna = load_luna_context(current_time, LUNA_CONTEXT)
        opposed_news = (
            luna.status == "observed" and luna.event_scale == "major"
            and luna.risk_bias == ("risk_off" if side == "long" else "risk_on")
        )
        factor = 0.5 if nearby or opposed_news else 1.0
        equity = float(self.wallets.get_total_stake_amount())
        stake = min(equity * 0.10 * factor, float(max_stake))
        if not isfinite(stake) or stake <= 0 or (min_stake is not None and stake < min_stake):
            return 0.0
        self._pending_stake[key] = (current_time, stake)
        LOG.info(
            "paper_leader_impulse_decision pair=%s side=%s stake=%.8f nearby_opposing_level=%s "
            "luna_status=%s luna_bias=%s opposed_news=%s",
            pair, side, stake, nearby, luna.status, luna.risk_bias, opposed_news,
        )
        return stake

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force,
                            current_time, entry_tag, side, **kwargs) -> bool:
        _ = (order_type, amount, rate, time_in_force, kwargs)
        pending = self._pending_stake.pop((pair, side, str(entry_tag)), None)
        if pending is None or not timedelta(0) <= current_time - pending[0] <= timedelta(seconds=30):
            LOG.error("PaperLeaderImpulse refused entry without a fresh sizing decision")
            return False
        return pending[1] > 0
